// dense.cu: attention-layer K2 / K4 helpers (the GEMMs themselves are cuBLASLt bf16 in the baseline).
//
//  rope_q_insert: K2 epilogue work after q = bf16(qr @ wq_b^T):
//     q[t, h, 448:512] = bf16(rope(q))           interleaved pairs, stock fp32 cos/sin cache, no FMA contraction
//     window KV store: kv_raw[t] (bf16, post-norm, post-RoPE) -> stock sm90 fp8_ds_mla record at slot_mapping[t]
//       (7 x 64 NoPE dims: amax >= 1e-4, s = 2^ceil(log2(amax * fp32(1/448))), e4m3(x / s) RNE; 64 bf16 RoPE dims;
//        scale bytes e + 127 and one zero pad byte), cache_utils.py / ds41h ref quant.v4_record_quant.
//  inv_rope: o_rot = bf16(rope^-1(o)) on dims 448..511 (K4's first op when it is not fused into the attention).
#ifndef DENSE_PDL
#define DENSE_PDL 0  // 1: programmatic dependent launch (trigger at start, wait before reading inputs); opt-in build (ATTN2B_PDL=1)
#endif
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cstdint>

namespace kd {

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }

template <bool INV>
__device__ __forceinline__ void rope_pair(__nv_bfloat16* x, const float* cs, int i) {
  const float c = __ldg(cs + i), sn = __ldg(cs + 32 + i);
  const float e = bf(x[2 * i]), o = bf(x[2 * i + 1]);
  float a, b;
  if (INV) {  // sin -> -sin
    a = __fadd_rn(__fmul_rn(e, c), __fmul_rn(o, sn));
    b = __fsub_rn(__fmul_rn(o, c), __fmul_rn(e, sn));
  } else {
    a = __fsub_rn(__fmul_rn(e, c), __fmul_rn(o, sn));
    b = __fadd_rn(__fmul_rn(e, sn), __fmul_rn(o, c));
  }
  x[2 * i] = __float2bfloat16_rn(a);
  x[2 * i + 1] = __float2bfloat16_rn(b);
}

// grid T, block 256: all threads: nh x 32 q pairs (nh = 16 local heads, or 64 in the sequence-parallel layout);
// warps 0..6: NoPE group w of the KV row; warp 7: RoPE part.
__global__ void __launch_bounds__(256) rope_q_insert_kernel(__nv_bfloat16* q, const __nv_bfloat16* kv, const long long* pos,
                                                            const float* cos_sin, const long long* slot_mapping, uint8_t* cache,
                                                            long long page_stride, int shift, int do_insert, int nh) {
  // PDL: let the next kernel launch now (it only reads constants until its own griddepcontrol.wait), then wait for
  // the previous kernel's outputs
  if (DENSE_PDL) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  const int t = blockIdx.x, tid = threadIdx.x;
  const float* cs = cos_sin + __ldg(pos + t) * 64;
  __nv_bfloat16* qt = q + static_cast<long long>(t) * nh * 512;
#pragma unroll 2
  for (int k = 0; k < nh / 8; ++k) {
    const int pi = tid + 256 * k, h = pi >> 5, i = pi & 31;
    rope_pair<false>(qt + h * 512 + 448, cs, i);
  }
  if (!do_insert) return;
  const long long slot = __ldg(slot_mapping + t);
  if (slot < 0) return;
  const int w = tid >> 5, lane = tid & 31;
  const long long pg = slot >> shift, ps = slot & ((1LL << shift) - 1);
  uint8_t* base = cache + pg * page_stride;
  uint8_t* data = base + ps * 576;
  uint8_t* scl = base + (576LL << shift) + ps * 8;
  const __nv_bfloat16* kt = kv + static_cast<long long>(t) * 512;
  if (w < 7) {
    const float x0 = bf(kt[64 * w + 2 * lane]), x1 = bf(kt[64 * w + 2 * lane + 1]);
    float am = fmaxf(fabsf(x0), fabsf(x1));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    am = fmaxf(am, 1e-4f);
    const float v = __fmul_rn(am, 1.0f / 448.0f);
    const unsigned bits = __float_as_uint(v);
    const int ex = static_cast<int>(bits >> 23) - ((bits & 0x7fffffu) == 0 ? 127 : 126);  // s = 2^ex >= v
    const float inv = __uint_as_float(static_cast<unsigned>(127 - ex) << 23);            // 2^-ex, exact
    const __nv_fp8x2_storage_t qv = __nv_cvt_float2_to_fp8x2(make_float2(x0 * inv, x1 * inv), __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<__nv_fp8x2_storage_t*>(data + 64 * w + 2 * lane) = qv;
    if (lane == 0) scl[w] = static_cast<uint8_t>(ex + 127);
  } else {
    // 64 bf16 RoPE dims = 128 bytes: lane l copies 4 bytes
    reinterpret_cast<unsigned*>(data + 448)[lane] = reinterpret_cast<const unsigned*>(kt + 448)[lane];
    if (lane == 0) scl[7] = 0;
  }
}

// grid T, block 256: in-place inverse RoPE on o[t, :, 448:512] (nh x 32 pairs per row).
__global__ void __launch_bounds__(256) inv_rope_kernel(__nv_bfloat16* o, const long long* pos, const float* cos_sin, int nh) {
  // PDL: let the next kernel launch now (it only reads constants until its own griddepcontrol.wait), then wait for
  // the previous kernel's outputs
  if (DENSE_PDL) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  const int t = blockIdx.x, tid = threadIdx.x;
  const float* cs = cos_sin + __ldg(pos + t) * 64;
  __nv_bfloat16* ot = o + static_cast<long long>(t) * nh * 512;
#pragma unroll 2
  for (int k = 0; k < nh / 8; ++k) {
    const int pi = tid + 256 * k, h = pi >> 5, i = pi & 31;
    rope_pair<true>(ot + h * 512 + 448, cs, i);
  }
}

// grid R, block 256: out[r] = bf16(x[r] * rsqrt(mean(x[r]^2) + eps) * w) (the reference RMSNorm: fp32, two rounded
// multiplies, one bf16 rounding); with pos: then RoPE on the last 64 dims (bf16 input, stock fp32 formula, bf16) = the
// kv path (kv_norm + RoPE). x rows may be strided (a view of a fused projection).
// one row of norm_rope (shared by the single and the paired kernel)
__device__ __forceinline__ void norm_rope_row(const __nv_bfloat16* x, long long ldx, const float* w, __nv_bfloat16* out,
                                              long long ldo, int D, float eps, const long long* pos, const float* cos_sin,
                                              int r) {
  const int tid = threadIdx.x;
  const __nv_bfloat16* xr = x + r * ldx;
  __nv_bfloat16* orow = out + r * ldo;
  float ss = 0.f;
  for (int i = tid; i < D; i += 256) {
    const float v = bf(xr[i]);
    ss = __fadd_rn(ss, __fmul_rn(v, v));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  __shared__ float red[8];
  if ((tid & 31) == 0) red[tid >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) tot += red[k];
  const float rs = rsqrtf(__fadd_rn(__fdiv_rn(tot, static_cast<float>(D)), eps));
  const int rope0 = pos != nullptr ? D - 64 : D;
  for (int i = tid; i < rope0; i += 256) orow[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(bf(xr[i]), rs), __ldg(w + i)));
  if (pos != nullptr && tid < 32) {
    const float* cs = cos_sin + __ldg(pos + r) * 64;
    const int i = tid, d0 = rope0 + 2 * i;
    const float e = bf(__float2bfloat16_rn(__fmul_rn(__fmul_rn(bf(xr[d0]), rs), __ldg(w + d0))));
    const float o = bf(__float2bfloat16_rn(__fmul_rn(__fmul_rn(bf(xr[d0 + 1]), rs), __ldg(w + d0 + 1))));
    const float c = __ldg(cs + i), sn = __ldg(cs + 32 + i);
    orow[d0] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(e, c), __fmul_rn(o, sn)));
    orow[d0 + 1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(e, sn), __fmul_rn(o, c)));
  }
}

__global__ void __launch_bounds__(256) norm_rope_kernel(const __nv_bfloat16* x, long long ldx, const float* w, __nv_bfloat16* out,
                                                        long long ldo, int D, float eps, const long long* pos, const float* cos_sin) {
  // PDL: let the next kernel launch now (it only reads constants until its own griddepcontrol.wait), then wait for
  // the previous kernel's outputs
  if (DENSE_PDL) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  norm_rope_row(x, ldx, w, out, ldo, D, eps, pos, cos_sin, blockIdx.x);
}

// both norms of the attention prologue in ONE launch (grid 2R): blocks [0, R) = q_a -> q_r (plain RMSNorm), blocks
// [R, 2R) = kv -> RoPE(RMSNorm(kv)); same per-row math as two norm_rope_kernel launches
__global__ void __launch_bounds__(256) norm_rope2_kernel(const __nv_bfloat16* xq, long long ldxq, const float* wq, __nv_bfloat16* oq,
                                                         long long ldoq, int Dq, const __nv_bfloat16* xk, long long ldxk,
                                                         const float* wk, __nv_bfloat16* ok, long long ldok, int Dk, int R,
                                                         float eps, const long long* pos, const float* cos_sin) {
  if (DENSE_PDL) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  if (static_cast<int>(blockIdx.x) < R)
    norm_rope_row(xq, ldxq, wq, oq, ldoq, Dq, eps, nullptr, nullptr, blockIdx.x);
  else
    norm_rope_row(xk, ldxk, wk, ok, ldok, Dk, eps, pos, cos_sin, blockIdx.x - R);
}

// launch with programmatic stream serialization (PDL): the kernel may start while the previous one drains
template <typename... KArgs, typename... Args>
static void launch_pdl(void (*kern)(KArgs...), int grid, const char* what, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid, 1, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = DENSE_PDL;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...) == cudaSuccess, what);
}

}  // namespace kd

void norm_rope(torch::Tensor x, torch::Tensor w, torch::Tensor out, double eps, c10::optional<torch::Tensor> pos,
               c10::optional<torch::Tensor> cos_sin) {
  const int R = static_cast<int>(x.size(0)), D = static_cast<int>(x.size(1));
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.stride(1) == 1 && out.dtype() == torch::kBFloat16 && out.stride(1) == 1, "x / out");
  TORCH_CHECK(w.dtype() == torch::kFloat32 && w.numel() == D && out.size(0) == R && out.size(1) == D, "w / out shape");
  const long long* pp = nullptr;
  const float* cp = nullptr;
  if (pos.has_value()) {
    TORCH_CHECK(cos_sin.has_value() && cos_sin->dtype() == torch::kFloat32 && cos_sin->size(1) == 64 && D >= 64, "cos_sin");
    pp = reinterpret_cast<const long long*>(pos->data_ptr<int64_t>());
    cp = cos_sin->data_ptr<float>();
  }
  kd::launch_pdl(kd::norm_rope_kernel, R, "norm_rope launch", reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0),
                 w.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), D, static_cast<float>(eps), pp, cp);
}

void norm_rope2(torch::Tensor xq, torch::Tensor wq, torch::Tensor oq, torch::Tensor xk, torch::Tensor wk, torch::Tensor ok,
                double eps, torch::Tensor pos, torch::Tensor cos_sin) {
  const int R = static_cast<int>(xq.size(0));
  TORCH_CHECK(xk.size(0) == R && oq.size(0) == R && ok.size(0) == R, "rows");
  TORCH_CHECK(xq.dtype() == torch::kBFloat16 && xq.stride(1) == 1 && oq.dtype() == torch::kBFloat16 && oq.stride(1) == 1, "q");
  TORCH_CHECK(xk.dtype() == torch::kBFloat16 && xk.stride(1) == 1 && ok.dtype() == torch::kBFloat16 && ok.stride(1) == 1, "kv");
  TORCH_CHECK(wq.dtype() == torch::kFloat32 && wq.numel() == xq.size(1) && wk.dtype() == torch::kFloat32 && wk.numel() == xk.size(1), "w");
  TORCH_CHECK(cos_sin.dtype() == torch::kFloat32 && cos_sin.size(1) == 64 && xk.size(1) >= 64, "cos_sin");
  kd::launch_pdl(kd::norm_rope2_kernel, 2 * R, "norm_rope2 launch", reinterpret_cast<const __nv_bfloat16*>(xq.data_ptr()), xq.stride(0),
                 wq.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(oq.data_ptr()), oq.stride(0), static_cast<int>(xq.size(1)),
                 reinterpret_cast<const __nv_bfloat16*>(xk.data_ptr()), xk.stride(0), wk.data_ptr<float>(),
                 reinterpret_cast<__nv_bfloat16*>(ok.data_ptr()), ok.stride(0), static_cast<int>(xk.size(1)), R, static_cast<float>(eps),
                 reinterpret_cast<const long long*>(pos.data_ptr<int64_t>()), cos_sin.data_ptr<float>());
}

void rope_q_insert(torch::Tensor q, torch::Tensor kv, torch::Tensor pos, torch::Tensor cos_sin, torch::Tensor slot_mapping,
                   torch::Tensor cache, int64_t page_stride, int64_t shift, bool do_insert) {
  const int T = static_cast<int>(q.size(0));
  const int nh = static_cast<int>(q.size(1));
  TORCH_CHECK(q.is_contiguous() && (nh == 16 || nh == 64) && q.size(2) == 512 && q.dtype() == torch::kBFloat16, "q");
  TORCH_CHECK(kv.is_contiguous() && kv.size(0) == T && kv.size(1) == 512, "kv");
  TORCH_CHECK(cos_sin.dtype() == torch::kFloat32 && cos_sin.size(1) == 64, "cos_sin");
  kd::launch_pdl(kd::rope_q_insert_kernel, T, "rope_q_insert launch", reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
                 reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()), reinterpret_cast<const long long*>(pos.data_ptr<int64_t>()),
                 cos_sin.data_ptr<float>(), reinterpret_cast<const long long*>(slot_mapping.data_ptr<int64_t>()), cache.data_ptr<uint8_t>(),
                 page_stride, static_cast<int>(shift), do_insert ? 1 : 0, nh);
}

void inv_rope(torch::Tensor o, torch::Tensor pos, torch::Tensor cos_sin) {
  const int T = static_cast<int>(o.size(0));
  const int nh = static_cast<int>(o.size(1));
  TORCH_CHECK(o.is_contiguous() && (nh == 16 || nh == 64) && o.size(2) == 512 && o.dtype() == torch::kBFloat16, "o");
  kd::launch_pdl(kd::inv_rope_kernel, T, "inv_rope launch", reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
                 reinterpret_cast<const long long*>(pos.data_ptr<int64_t>()), cos_sin.data_ptr<float>(), nh);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_q_insert", &rope_q_insert, "K2: RoPE(q) in place + window KV record insert");
  m.def("inv_rope", &inv_rope, "K4: inverse RoPE of the attention output in place");
  m.def("norm_rope", &norm_rope, "RMSNorm (reference rounding), optionally + RoPE of the last 64 dims (kv path)");
  m.def("norm_rope2", &norm_rope2, "both attention-prologue norms (q plain, kv + RoPE) in one launch");
}
