// ds41h-phys Stage 2f (indexer2f): the indexer q path of one index layer in ONE kernel (v2 of idx2f_q.cu's q_kernel).
//
//   q = qr @ Wq^T (bf16 mma.sync m16n8k16, fp32 accumulation), then the stock fused_indexer_q_rope_quant epilogue
//   (idx2f_q.cu's, bit-identical to the stock op given the same bf16 q / w): GPT-J RoPE on dims 64..127, per-(row, head)
//   ue8m0 scale, e4m3 satfinite, weights_out = (w * wscale) * 2^(e - 127). w = weights_proj(x) [R, 32] bf16 is an input
//   (cuBLAS, on the graph's side branch next to this kernel's GEMM).
// Layout: grid (4 K-parts x 32 heads, ceil(R / 160)), cluster (4, 1, 1) = the 4 K-parts of one head. CTA (kp, h, c) loads
//   rows [160 c, +160) of qr[:, 320 kp .. +320) (100 KB) and Wq[128 h .. +128, 320 kp .. +320) (80 KB) once, in 4
//   cp.async stages along K, and accumulates its partial [160, 128] in registers (8 warps x <= 5 m16 x 4 n8 tiles). Every
//   weight byte is read ONCE per call (idx2f_q.cu re-read each head's weights per 32-row tile: ~3-5x the L2 traffic).
//   The partials go to shared memory; after a cluster barrier CTA kp reduces rows [40 kp, +40) of the chunk over the
//   4 CTAs' partials through DSMEM in the fixed order ((P0 + P1) + P2) + P3 and runs the epilogue (a warp per row, 4
//   dims per lane). Deterministic; the GEMM differs from cuBLAS only by the fp32 summation order.
// Debug: qin [R, 4096] bf16 replaces the GEMM (epilogue-only check vs the stock op); qdbg [R, 4096] bf16 = q before RoPE.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

namespace cg = cooperative_groups;

namespace idx2f_q2 {

constexpr int kHeads = 32, kHD = 128, kQK = 1280;
constexpr int kKP = 4, kKPart = kQK / kKP;   // 320 per K-part
constexpr int kRC = 160;                     // rows per chunk (10 m16 tiles)
constexpr int kThreads = 256;                // 8 warps
constexpr int kStages = 4;                   // cp.async groups along K (80 elements = 10 chunks of 16 B each)
constexpr int kChunks = kKPart / 8;          // 40 chunks of 16 B per smem row
constexpr int kPLd = kHD + 4;                // partial row stride (floats)
constexpr int kSmemA = kRC * kKPart * 2, kSmemB = kHD * kKPart * 2;
constexpr int kSmem = kSmemA + kSmemB;       // 184,320 B (the partials [160, 132] fp32 = 84,480 B reuse it)
static_assert(kRC * kPLd * 4 <= kSmem, "partials fit");

__device__ __forceinline__ void cp16(void* s, const void* g, bool ok) {
  const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(s));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(unsigned (&r)[2], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
// rows of 320 bf16 = 40 chunks of 16 B; XOR swizzle inside each group of 8 chunks (= idx2f_q.cu swzp)
__device__ __forceinline__ int swz(int row, int chunk) { return row * kKPart + (((chunk & ~7) | ((chunk ^ row) & 7)) << 3); }

struct Args {
  const __nv_bfloat16* qr;   // [R, 1280]
  const __nv_bfloat16* wq;   // [4096, 1280]
  const __nv_bfloat16* w;    // [R, 32] weights_proj(x)
  const long* pos;           // [R]
  const float* cs;           // [P, 64] cos | sin
  long cs_ld;
  float wscale;
  uint8_t* qf8;              // [R, 32, 128] e4m3
  float* wout;               // [R, 32]
  const __nv_bfloat16* qin;  // debug: [R, 4096] replaces the GEMM
  __nv_bfloat16* qdbg;       // debug: [R, 4096] q before RoPE
  int R;
};

__global__ void __cluster_dims__(kKP, 1, 1) __launch_bounds__(kThreads, 1) q2_kernel(const __grid_constant__ Args a) {
  extern __shared__ __align__(128) uint8_t smem[];
  __nv_bfloat16* sa = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sb = reinterpret_cast<__nv_bfloat16*>(smem + kSmemA);
  float* part = reinterpret_cast<float*>(smem);
  cg::cluster_group cl = cg::this_cluster();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kp = int(cl.block_rank()), h = blockIdx.x / kKP, r0 = blockIdx.y * kRC;
  const int k0 = kp * kKPart;
  const int R = a.R;
  const bool gemm = a.qin == nullptr;
  if (gemm) {
    // ---- loads: stage s = chunks [10 s, 10 s + 10) of every row (A then B), one commit group per stage
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      constexpr int CPS = kChunks / kStages;  // 10
      for (int i = tid; i < kRC * CPS; i += kThreads) {
        const int row = i / CPS, ch = s * CPS + (i - row * CPS);
        const bool ok = r0 + row < R;
        cp16(&sa[swz(row, ch)], a.qr + long(ok ? r0 + row : 0) * kQK + k0 + ch * 8, ok);
      }
      for (int i = tid; i < kHD * CPS; i += kThreads) {
        const int row = i / CPS, ch = s * CPS + (i - row * CPS);
        cp16(&sb[swz(row, ch)], a.wq + long(h * kHD + row) * kQK + k0 + ch * 8, true);
      }
      cp_commit();
    }
  }
  // ---- GEMM: warp tile = m16 tiles 2 i + (warp & 1) (interleaved: a partial chunk's tiles split evenly over the two
  // halves; tiles past the chunk's rows are skipped, warp-uniform) x dims [32 (warp >> 1), +32)
  const int wh = warp & 1, wc = (warp >> 1) * 32;
  const int nmt = (min(kRC, R - r0) + 15) / 16;
  float acc[5][4][4];
#pragma unroll
  for (int i = 0; i < 5; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  if (gemm) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      if (s == 0) cp_wait<3>(); else if (s == 1) cp_wait<2>(); else if (s == 2) cp_wait<1>(); else cp_wait<0>();
      __syncthreads();
#pragma unroll
      for (int kk = 0; kk < 5; ++kk) {   // 5 k16 steps per stage (80 elements)
        const int k16 = s * 5 + kk;
        unsigned bfr[4][2];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          ldsm_x2(bfr[j], &sb[swz(wc + j * 8 + (lane & 7), k16 * 2 + ((lane >> 3) & 1))]);
#pragma unroll
        for (int i = 0; i < 5; ++i) {
          if (2 * i + wh >= nmt) break;
          unsigned afr[4];
          ldsm_x4(afr, &sa[swz((2 * i + wh) * 16 + (lane & 15), k16 * 2 + (lane >> 4))]);
#pragma unroll
          for (int j = 0; j < 4; ++j) mma16816(acc[i][j], afr, bfr[j]);
        }
      }
    }
    __syncthreads();   // every warp is done with the operands: the partials overwrite them
  }
  {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 5; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int ra = (2 * i + wh) * 16 + g, c = wc + j * 8 + 2 * t;
        *reinterpret_cast<float2*>(&part[ra * kPLd + c]) = make_float2(acc[i][j][0], acc[i][j][1]);
        *reinterpret_cast<float2*>(&part[(ra + 8) * kPLd + c]) = make_float2(acc[i][j][2], acc[i][j][3]);
      }
  }
  cl.sync();   // all 4 K-parts' partials written (release / acquire across the cluster)
  // ---- reduce + epilogue: rows [40 kp, +40) of the chunk, warp w -> rows 40 kp + 5 w + i, lane -> dims 4 lane .. + 3
  const float* p0 = cl.map_shared_rank(part, 0);
  const float* p1 = cl.map_shared_rank(part, 1);
  const float* p2 = cl.map_shared_rank(part, 2);
  const float* p3 = cl.map_shared_rank(part, 3);
  const int d = 4 * lane;
  for (int i = 0; i < 5; ++i) {
    const int lr = kp * 40 + warp * 5 + i, r = r0 + lr;
    if (r >= R) break;   // warp-uniform
    float e[4];
    if (gemm) {
      const float4 x0 = *reinterpret_cast<const float4*>(p0 + lr * kPLd + d);
      const float4 x1 = *reinterpret_cast<const float4*>(p1 + lr * kPLd + d);
      const float4 x2 = *reinterpret_cast<const float4*>(p2 + lr * kPLd + d);
      const float4 x3 = *reinterpret_cast<const float4*>(p3 + lr * kPLd + d);
      e[0] = bf16r(__fadd_rn(__fadd_rn(__fadd_rn(x0.x, x1.x), x2.x), x3.x));
      e[1] = bf16r(__fadd_rn(__fadd_rn(__fadd_rn(x0.y, x1.y), x2.y), x3.y));
      e[2] = bf16r(__fadd_rn(__fadd_rn(__fadd_rn(x0.z, x1.z), x2.z), x3.z));
      e[3] = bf16r(__fadd_rn(__fadd_rn(__fadd_rn(x0.w, x1.w), x2.w), x3.w));
    } else {
#pragma unroll
      for (int k = 0; k < 4; ++k) e[k] = __bfloat162float(a.qin[long(r) * 4096 + h * kHD + d + k]);
    }
    if (a.qdbg != nullptr) {
#pragma unroll
      for (int k = 0; k < 4; ++k) a.qdbg[long(r) * 4096 + h * kHD + d + k] = __float2bfloat16_rn(e[k]);
    }
    if (d >= 64) {   // GPT-J pairs (d, d + 1), (d + 2, d + 3)  (= idx2f_q.cu q_kernel / the stock op)
      const float* csr = a.cs + a.pos[r] * a.cs_ld;
#pragma unroll
      for (int k = 0; k < 4; k += 2) {
        const int ii = (d + k - 64) >> 1;
        const float c = __ldg(csr + ii), s_ = __ldg(csr + 32 + ii);
        const float x0 = e[k], x1 = e[k + 1];
        e[k] = bf16r(__fmaf_rn(x0, c, -__fmul_rn(x1, s_)));
        e[k + 1] = bf16r(__fmaf_rn(x0, s_, __fmul_rn(x1, c)));
      }
    }
    float am = fmaxf(fmaxf(fabsf(e[0]), fabsf(e[1])), fmaxf(fabsf(e[2]), fabsf(e[3])));
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const unsigned ex = ((__float_as_uint(fmaxf(am, 1e-4f) * (1.0f / 448.0f)) + 0x7fffffu) >> 23) & 0xffu;
    const float inv = __uint_as_float((254u - ex) << 23);
    const __nv_fp8x2_storage_t v0 = __nv_cvt_float2_to_fp8x2(make_float2(e[0] * inv, e[1] * inv), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t v1 = __nv_cvt_float2_to_fp8x2(make_float2(e[2] * inv, e[3] * inv), __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint32_t*>(a.qf8 + long(r) * 4096 + h * kHD + d) = uint32_t(v0) | (uint32_t(v1) << 16);
    if (lane == 0) {
      const float wv = __bfloat162float(a.w[long(r) * kHeads + h]);
      a.wout[long(r) * kHeads + h] = __fmul_rn(__fmul_rn(wv, a.wscale), __uint_as_float(ex << 23));
    }
  }
  cl.sync();   // the other CTAs may still read this CTA's partials
}

void qpath2(at::Tensor qr, at::Tensor wq, at::Tensor w, at::Tensor positions, at::Tensor cos_sin, double wscale,
            at::Tensor q_fp8, at::Tensor wout, std::optional<at::Tensor> qin, std::optional<at::Tensor> qdbg) {
  const int R = int(qr.size(0));
  TORCH_CHECK(qr.is_cuda() && qr.scalar_type() == at::kBFloat16 && qr.is_contiguous() && qr.size(1) == kQK, "qr [R, 1280] bf16");
  TORCH_CHECK(wq.scalar_type() == at::kBFloat16 && wq.is_contiguous() && wq.size(0) == kHeads * kHD && wq.size(1) == kQK, "wq [4096, 1280] bf16");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.size(0) >= R && w.size(1) == kHeads, "w [R, 32] bf16");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.is_contiguous() && positions.numel() >= R, "positions int64");
  TORCH_CHECK(cos_sin.scalar_type() == at::kFloat && cos_sin.dim() == 2 && cos_sin.size(1) == 64 && cos_sin.stride(1) == 1, "cos_sin [P, 64]");
  TORCH_CHECK(q_fp8.element_size() == 1 && q_fp8.is_contiguous() && q_fp8.numel() >= long(R) * 4096, "q_fp8 [R, 32, 128]");
  TORCH_CHECK(wout.scalar_type() == at::kFloat && wout.is_contiguous() && wout.numel() >= long(R) * kHeads, "wout [R, 32]");
  if (qin) TORCH_CHECK(qin->scalar_type() == at::kBFloat16 && qin->is_contiguous() && qin->numel() >= long(R) * 4096, "qin");
  if (qdbg) TORCH_CHECK(qdbg->scalar_type() == at::kBFloat16 && qdbg->is_contiguous() && qdbg->numel() >= long(R) * 4096, "qdbg");
  if (!R) return;
  const c10::cuda::CUDAGuard guard(qr.device());
  static bool set[64] = {};
  const int dev = qr.get_device();
  if (!set[dev]) {
    TORCH_CHECK(cudaFuncSetAttribute(q2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) == cudaSuccess, "q2 smem");
    set[dev] = true;
  }
  Args a{reinterpret_cast<const __nv_bfloat16*>(qr.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(wq.data_ptr()),
         reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), positions.data_ptr<long>(), cos_sin.data_ptr<float>(),
         cos_sin.stride(0), float(wscale), reinterpret_cast<uint8_t*>(q_fp8.data_ptr()), wout.data_ptr<float>(),
         qin ? reinterpret_cast<const __nv_bfloat16*>(qin->data_ptr()) : nullptr,
         qdbg ? reinterpret_cast<__nv_bfloat16*>(qdbg->data_ptr()) : nullptr, R};
  q2_kernel<<<dim3(kKP * kHeads, (R + kRC - 1) / kRC), kThreads, kSmem, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// debug: how many 4-CTA clusters of q2_kernel can be resident at once (cudaOccupancyMaxActiveClusters)
int64_t max_clusters(int64_t R) {
  static bool set = false;
  if (!set) {
    TORCH_CHECK(cudaFuncSetAttribute(q2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) == cudaSuccess, "q2 smem");
    set = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(kKP * kHeads, unsigned((R + kRC - 1) / kRC));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = kSmem;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = kKP, attr[0].val.clusterDim.y = 1, attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int n = -1;
  TORCH_CHECK(cudaOccupancyMaxActiveClusters(&n, q2_kernel, &cfg) == cudaSuccess, "cudaOccupancyMaxActiveClusters");
  return n;
}

}  // namespace idx2f_q2

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("max_clusters", &idx2f_q2::max_clusters, "resident 4-CTA clusters of q2_kernel (debug)");
  m.def("qpath2", &idx2f_q2::qpath2, "indexer q path v2: wq_b GEMM (cluster K-split) + stock RoPE / ue8m0 fp8 quant / weight fold");
}
