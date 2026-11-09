// indexer2f: the indexer q path of one index layer on the owned rows, in two kernels.
//
//   wp_kernel : weights_proj partials P[ks][r][h] = sum_{k in part ks} x[r, k] * Wp[h, k]   (bf16 mma.sync, fp32; 16 k-parts
//               of 320, deterministic: no atomics; the q kernel sums the parts in order 0..15 and rounds to bf16 = the stock
//               bf16 GEMM output up to fp32 summation order)
//   q_kernel  : q = qr @ Wq^T for one (head, 32-row tile) per CTA (bf16 mma.sync on the exactly dequantized fp8 weights, fp32
//               accumulation over K = 1280 in order, bf16 round = the W8A16 GEMM output up to summation order), then IN
//               REGISTERS the stock fused_indexer_q_rope_quant (deepseek_v4 IndexerQFp8Kernel, as bit-reproduced by the ds41h
//               prologue v3_idx_q): GPT-J RoPE on dims 64..127 (pair (2i, 2i+1), cos / sin [pos, i] / [pos, 32 + i]),
//               r = bf16(fma(e0, c, -e1 s)), bf16(fma(e0, s, e1 c)); per-(row, head) amax -> power-of-two scale
//               e = ((bits(max(amax, 1e-4) * (1 / 448)) + 0x7fffff) >> 23) & 0xff; q_fp8 = e4m3(r * 2^(127 - e)) (satfinite);
//               weights_out = (bf16(w) * wscale) * 2^(e - 127), wscale = softmax_scale * n_head^-0.5 (fp32).
// Given the same bf16 q / w the epilogue is bit-identical to stock (checked against the stock op).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

namespace idx2f_q {

constexpr int kHeads = 32, kHD = 128, kQK = 1280, kXK = 5120;
constexpr int kKS = 16, kKPart = kXK / kKS;  // weights_proj k-parts of 320 (single-shot loads: 30 KB per CTA)
constexpr int kStages = 8;                    // q GEMM pipeline depth (20 KB per stage: ~140 KB in flight per SM)
constexpr int kBK = 64;                     // k chunk (smem stage)
constexpr int kRows = 32;                   // q tile rows per CTA

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

// smem tiles are [rows][kBK] bf16 with a 16-byte XOR swizzle on the 8 16-byte chunks of a 128-byte row
__device__ __forceinline__ int swz(int row, int chunk) { return row * kBK + ((chunk ^ (row & 7)) << 3); }

// ---------------------------------------------------------------------------------------------- weights_proj partials
// grid (ceil(R / 16), kKS), 128 threads: warp w = output columns [8w, 8w + 8) of the 32 heads, rows [16 blockIdx.x, +16); the whole
// k-part (16 x 320 activations + 32 x 320 weights, bf16) is loaded in one shot (latency once, not per chunk)
constexpr int kWPC = kKPart / 8;  // 16-byte chunks per row of a part (40)
__device__ __forceinline__ int swzp(int row, int chunk) { return row * kKPart + (((chunk & ~7) | ((chunk ^ row) & 7)) << 3); }
__global__ void __launch_bounds__(128) wp_kernel(const __nv_bfloat16* __restrict__ x, int R, const __nv_bfloat16* __restrict__ wp,
                                                 float* __restrict__ part) {
  __shared__ __align__(128) __nv_bfloat16 sa[16 * kKPart];
  __shared__ __align__(128) __nv_bfloat16 sb[kHeads * kKPart];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int r0 = blockIdx.x * 16, ks = blockIdx.y, k0 = ks * kKPart;
  for (int i = tid; i < 16 * kWPC; i += 128) {
    const int row = i / kWPC, ch = i - row * kWPC;
    const bool ok = r0 + row < R;
    cp16(&sa[swzp(row, ch)], x + long(ok ? r0 + row : 0) * kXK + k0 + ch * 8, ok);
  }
  for (int i = tid; i < kHeads * kWPC; i += 128) {
    const int row = i / kWPC, ch = i - row * kWPC;
    cp16(&sb[swzp(row, ch)], wp + long(row) * kXK + k0 + ch * 8, true);
  }
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
  for (int k16 = 0; k16 < kKPart / 16; ++k16) {
    unsigned a[4], b[2];
    ldsm_x4(a, &sa[swzp(lane & 15, k16 * 2 + (lane >> 4))]);
    ldsm_x2(b, &sb[swzp(warp * 8 + (lane & 7), k16 * 2 + ((lane >> 3) & 1))]);
    mma16816(acc, a, b);
  }
  const int g = lane >> 2, t = lane & 3, col = warp * 8 + 2 * t;
  float* p = part + long(ks) * R * kHeads;
  if (r0 + g < R) p[long(r0 + g) * kHeads + col] = acc[0], p[long(r0 + g) * kHeads + col + 1] = acc[1];
  if (r0 + g + 8 < R) p[long(r0 + g + 8) * kHeads + col] = acc[2], p[long(r0 + g + 8) * kHeads + col + 1] = acc[3];
}

// ---------------------------------------------------------------------------------------------- q GEMM + RoPE + quant
// grid (32 heads, ceil(R / 32)), 128 threads: warp w = rows [16 (w & 1), +16) x head dims [64 (w >> 1), +64) (8 n-tiles)
__global__ void __launch_bounds__(128) q_kernel(const __nv_bfloat16* __restrict__ qr, int R, const __nv_bfloat16* __restrict__ wq,
                                                const long* __restrict__ pos, const float* __restrict__ cs, long cs_ld,
                                                const float* __restrict__ part, float wscale, uint8_t* __restrict__ qf8,
                                                float* __restrict__ wout, __nv_bfloat16* __restrict__ qdbg) {
  extern __shared__ __align__(128) uint8_t qsm[];
  auto sa = reinterpret_cast<__nv_bfloat16(*)[kRows * kBK]>(qsm);
  auto sb = reinterpret_cast<__nv_bfloat16(*)[kHD * kBK]>(qsm + kStages * kRows * kBK * 2);
  __shared__ float s_amax[2][kRows];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int h = blockIdx.x, r0 = blockIdx.y * kRows;
  const int wr = (warp & 1) * 16, wc = (warp >> 1) * 64;
  auto load = [&](int st, int kc) {
    const int kk = kc * kBK;
    for (int i = tid; i < kRows * 8; i += 128) {
      const int row = i >> 3, ch = i & 7;
      const bool ok = r0 + row < R;
      cp16(&sa[st][swz(row, ch)], qr + long(ok ? r0 + row : 0) * kQK + kk + ch * 8, ok);
    }
    for (int i = tid; i < kHD * 8; i += 128) {
      const int row = i >> 3, ch = i & 7;
      cp16(&sb[st][swz(row, ch)], wq + long(h * kHD + row) * kQK + kk + ch * 8, true);
    }
    cp_commit();
  };
  float acc[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  constexpr int NC = kQK / kBK;  // 20
#pragma unroll
  for (int i = 0; i < kStages - 1; ++i) load(i, i);  // NC >= kStages - 1
  for (int kc = 0; kc < NC; ++kc) {
    cp_wait<kStages - 2>();
    __syncthreads();
    if (kc + kStages - 1 < NC) load((kc + kStages - 1) % kStages, kc + kStages - 1); else cp_commit();
    const int st = kc % kStages;
#pragma unroll
    for (int k16 = 0; k16 < kBK / 16; ++k16) {
      unsigned a[4];
      {
        const int row = wr + (lane & 15), ch = k16 * 2 + (lane >> 4);
        ldsm_x4(a, &sa[st][swz(row, ch)]);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        unsigned b[2];
        const int row = wc + j * 8 + (lane & 7), ch = k16 * 2 + ((lane >> 3) & 1);
        ldsm_x2(b, &sb[st][swz(row, ch)]);
        mma16816(acc[j], a, b);
      }
    }
  }
  cp_wait<0>();
  // ---- epilogue: rows ra = r0 + wr + g (acc[.][0..1]) and rb = ra + 8 (acc[.][2..3]); head dims wc + 8 j + 2 t, + 1
  const int g = lane >> 2, t = lane & 3;
  const int lra = wr + g, lrb = wr + g + 8, ra = r0 + lra, rb = r0 + lrb;
  const float* csa = cs + (ra < R ? pos[ra] : 0) * cs_ld;
  const float* csb = cs + (rb < R ? pos[rb] : 0) * cs_ld;
  float am_a = 0.f, am_b = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int d = wc + j * 8 + 2 * t;
    float e0 = bf16r(acc[j][0]), e1 = bf16r(acc[j][1]), f0 = bf16r(acc[j][2]), f1 = bf16r(acc[j][3]);
    if (qdbg != nullptr) {
      if (ra < R) qdbg[long(ra) * 4096 + h * kHD + d] = __float2bfloat16_rn(e0), qdbg[long(ra) * 4096 + h * kHD + d + 1] = __float2bfloat16_rn(e1);
      if (rb < R) qdbg[long(rb) * 4096 + h * kHD + d] = __float2bfloat16_rn(f0), qdbg[long(rb) * 4096 + h * kHD + d + 1] = __float2bfloat16_rn(f1);
    }
    if (d >= 64) {
      const int i = (d - 64) >> 1;
      const float ca = __ldg(csa + i), sa_ = __ldg(csa + 32 + i), cb = __ldg(csb + i), sb_ = __ldg(csb + 32 + i);
      const float x0 = e0, x1 = e1, y0 = f0, y1 = f1;
      e0 = bf16r(__fmaf_rn(x0, ca, -__fmul_rn(x1, sa_)));
      e1 = bf16r(__fmaf_rn(x0, sa_, __fmul_rn(x1, ca)));
      f0 = bf16r(__fmaf_rn(y0, cb, -__fmul_rn(y1, sb_)));
      f1 = bf16r(__fmaf_rn(y0, sb_, __fmul_rn(y1, cb)));
    }
    acc[j][0] = e0, acc[j][1] = e1, acc[j][2] = f0, acc[j][3] = f1;
    am_a = fmaxf(am_a, fmaxf(fabsf(e0), fabsf(e1)));
    am_b = fmaxf(am_b, fmaxf(fabsf(f0), fabsf(f1)));
  }
#pragma unroll
  for (int o = 1; o < 4; o <<= 1) {
    am_a = fmaxf(am_a, __shfl_xor_sync(0xffffffffu, am_a, o));
    am_b = fmaxf(am_b, __shfl_xor_sync(0xffffffffu, am_b, o));
  }
  if (t == 0) s_amax[warp >> 1][lra] = am_a, s_amax[warp >> 1][lrb] = am_b;
  __syncthreads();
  const float amax_a = fmaxf(s_amax[0][lra], s_amax[1][lra]), amax_b = fmaxf(s_amax[0][lrb], s_amax[1][lrb]);
  const unsigned ea = ((__float_as_uint(fmaxf(amax_a, 1e-4f) * (1.0f / 448.0f)) + 0x7fffffu) >> 23) & 0xffu;
  const unsigned eb = ((__float_as_uint(fmaxf(amax_b, 1e-4f) * (1.0f / 448.0f)) + 0x7fffffu) >> 23) & 0xffu;
  const float inva = __uint_as_float((254u - ea) << 23), invb = __uint_as_float((254u - eb) << 23);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int d = wc + j * 8 + 2 * t;
    if (ra < R) {
      const __nv_fp8x2_storage_t v = __nv_cvt_float2_to_fp8x2(make_float2(acc[j][0] * inva, acc[j][1] * inva), __NV_SATFINITE, __NV_E4M3);
      *reinterpret_cast<__nv_fp8x2_storage_t*>(qf8 + long(ra) * 4096 + h * kHD + d) = v;
    }
    if (rb < R) {
      const __nv_fp8x2_storage_t v = __nv_cvt_float2_to_fp8x2(make_float2(acc[j][2] * invb, acc[j][3] * invb), __NV_SATFINITE, __NV_E4M3);
      *reinterpret_cast<__nv_fp8x2_storage_t*>(qf8 + long(rb) * 4096 + h * kHD + d) = v;
    }
  }
  // weights_out: one thread per (row, head); w = bf16(sum of the 8 k-parts in order)
  if (warp < 2 && t == 0 && (warp >> 1) == 0) {
    for (int k = 0; k < 2; ++k) {
      const int lr = k ? lrb : lra, r = r0 + lr;
      if (r >= R) continue;
      float w = 0.f;
#pragma unroll
      for (int ks = 0; ks < kKS; ++ks) w += part[(long(ks) * R + r) * kHeads + h];
      const unsigned e = k ? eb : ea;
      wout[long(r) * kHeads + h] = __fmul_rn(__fmul_rn(bf16r(w), wscale), __uint_as_float(e << 23));
    }
  }
}

// ---------------------------------------------------------------------------------------------- host
void qpath(at::Tensor x, at::Tensor qr, at::Tensor wp, at::Tensor wq, at::Tensor positions, at::Tensor cos_sin, double wscale,
           at::Tensor part, at::Tensor q_fp8, at::Tensor wout, std::optional<at::Tensor> qdbg, int64_t which) {
  const int R = int(x.size(0));
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(1) == kXK, "x [R, 5120] bf16");
  TORCH_CHECK(qr.scalar_type() == at::kBFloat16 && qr.is_contiguous() && qr.size(0) == R && qr.size(1) == kQK, "qr [R, 1280] bf16");
  TORCH_CHECK(wp.scalar_type() == at::kBFloat16 && wp.is_contiguous() && wp.size(0) == kHeads && wp.size(1) == kXK, "wp [32, 5120]");
  TORCH_CHECK(wq.scalar_type() == at::kBFloat16 && wq.is_contiguous() && wq.size(0) == kHeads * kHD && wq.size(1) == kQK, "wq [4096, 1280]");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.is_contiguous() && positions.numel() >= R, "positions int64");
  TORCH_CHECK(cos_sin.scalar_type() == at::kFloat && cos_sin.dim() == 2 && cos_sin.size(1) == 64 && cos_sin.stride(1) == 1, "cos_sin [P, 64]");
  TORCH_CHECK(part.scalar_type() == at::kFloat && part.is_contiguous() && part.numel() >= long(kKS) * R * kHeads, "part [16, R, 32]");
  TORCH_CHECK(q_fp8.element_size() == 1 && q_fp8.is_contiguous() && q_fp8.numel() >= long(R) * 4096, "q_fp8 [R, 32, 128]");
  TORCH_CHECK(wout.scalar_type() == at::kFloat && wout.is_contiguous() && wout.numel() >= long(R) * kHeads, "wout [R, 32]");
  if (!R) return;
  const c10::cuda::CUDAGuard guard(x.device());
  auto st = at::cuda::getCurrentCUDAStream();
  if (which & 1) wp_kernel<<<dim3((R + 15) / 16, kKS), 128, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), R,
                                                      reinterpret_cast<const __nv_bfloat16*>(wp.data_ptr()), part.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  constexpr int smem = kStages * (kRows + kHD) * kBK * 2;
  static bool set[64] = {};
  const int dev = x.get_device();
  if (!set[dev]) {
    TORCH_CHECK(cudaFuncSetAttribute(q_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess, "q smem");
    set[dev] = true;
  }
  if (which & 2) q_kernel<<<dim3(kHeads, (R + kRows - 1) / kRows), 128, smem, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(qr.data_ptr()), R, reinterpret_cast<const __nv_bfloat16*>(wq.data_ptr()),
      positions.data_ptr<long>(), cos_sin.data_ptr<float>(), cos_sin.stride(0), part.data_ptr<float>(), float(wscale),
      reinterpret_cast<uint8_t*>(q_fp8.data_ptr()), wout.data_ptr<float>(),
      qdbg ? reinterpret_cast<__nv_bfloat16*>(qdbg->data_ptr()) : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace idx2f_q

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qpath", &idx2f_q::qpath, "indexer q path: weights_proj + wq_b + stock RoPE / ue8m0 fp8 quant / weight fold");
}
