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

// launch with programmatic stream serialization (PDL): the kernel may start while the previous one drains; on `stream`
// (C ABI: the caller's), the launch's own return code
template <typename... KArgs, typename... Args>
static cudaError_t launch_pdl(void (*kern)(KArgs...), int grid, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid, 1, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = DENSE_PDL;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kern, static_cast<KArgs>(args)...);
}

}  // namespace kd

// C ABI: the host launches are C functions on plain argument structs
// and an explicit stream, exported through fpr_head_dense_table(); the pybind functions build the structs and call the
// same functions. Kernels unchanged.
#include "head_dense_cabi.h"
#include "../../../engine/mixed_cxx/fpr_cabi_rec.h"   // rule 9: optional record hook
FPR_CABI_RECORD_DECL(head_dense)

namespace kd {

static inline int rc(cudaError_t e) { return e == cudaSuccess ? FPR_OK : FPR_E_CUDA; }

#ifdef FPR_CABI_DEBUG
static inline bool dev_ok(const FprCall& c) {
  int d = -1;
  return cudaGetDevice(&d) == cudaSuccess && d == c.device;
}
#define FPR_DEV_CHECK(c) \
  if (!dev_ok(c)) return FPR_E_ARG
#else
#define FPR_DEV_CHECK(c)
#endif

// ---- the C functions (the table's entries; the pybind wrappers below call the same ones)
static int c_norm_rope2(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprAttnDenseNormRope2Args*>(p);
  FPR_DEV_CHECK(a.call);
  if (a.dk < 64 || a.pos == nullptr || a.cos_sin == nullptr) return FPR_E_ARG;
  const int R = static_cast<int>(a.rows);
  return rc(launch_pdl(norm_rope2_kernel, 2 * R, a.call.stream, static_cast<const __nv_bfloat16*>(a.xq), a.ldxq, a.wq,
                       static_cast<__nv_bfloat16*>(a.oq), a.ldoq, static_cast<int>(a.dq),
                       static_cast<const __nv_bfloat16*>(a.xk), a.ldxk, a.wk, static_cast<__nv_bfloat16*>(a.ok), a.ldok,
                       static_cast<int>(a.dk), R, a.eps, reinterpret_cast<const long long*>(a.pos), a.cos_sin));
}

static int c_norm_rope(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprAttnDenseNormRopeArgs*>(p);
  FPR_DEV_CHECK(a.call);
  if (a.pos != nullptr && (a.cos_sin == nullptr || a.d < 64)) return FPR_E_ARG;
  return rc(launch_pdl(norm_rope_kernel, static_cast<int>(a.rows), a.call.stream, static_cast<const __nv_bfloat16*>(a.x),
                       a.ldx, a.w, static_cast<__nv_bfloat16*>(a.out), a.ldo, static_cast<int>(a.d), a.eps,
                       reinterpret_cast<const long long*>(a.pos), a.cos_sin));
}

static int c_rope_q_insert(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprAttnDenseRopeQInsertArgs*>(p);
  FPR_DEV_CHECK(a.call);
  if (a.nh != 16 && a.nh != 64) return FPR_E_ARG;
  return rc(launch_pdl(rope_q_insert_kernel, static_cast<int>(a.t), a.call.stream, static_cast<__nv_bfloat16*>(a.q),
                       static_cast<const __nv_bfloat16*>(a.kv), reinterpret_cast<const long long*>(a.pos), a.cos_sin,
                       reinterpret_cast<const long long*>(a.slot_mapping), a.cache, a.page_stride, static_cast<int>(a.shift),
                       a.do_insert ? 1 : 0, static_cast<int>(a.nh)));
}

static int c_inv_rope(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprAttnDenseInvRopeArgs*>(p);
  FPR_DEV_CHECK(a.call);
  if (a.nh != 16 && a.nh != 64) return FPR_E_ARG;
  return rc(launch_pdl(inv_rope_kernel, static_cast<int>(a.t), a.call.stream, static_cast<__nv_bfloat16*>(a.o),
                       reinterpret_cast<const long long*>(a.pos), a.cos_sin, static_cast<int>(a.nh)));
}

}  // namespace kd

extern "C" __attribute__((visibility("default"))) const FprExtTable* fpr_head_dense_table(void) {
  static const FprFnEntry fns[] = {
      {"head_dense.norm_rope2", kd::c_norm_rope2, nullptr, static_cast<int64_t>(sizeof(FprAttnDenseNormRope2Args))},
      {"head_dense.norm_rope", kd::c_norm_rope, nullptr, static_cast<int64_t>(sizeof(FprAttnDenseNormRopeArgs))},
      {"head_dense.rope_q_insert", kd::c_rope_q_insert, nullptr, static_cast<int64_t>(sizeof(FprAttnDenseRopeQInsertArgs))},
      {"head_dense.inv_rope", kd::c_inv_rope, nullptr, static_cast<int64_t>(sizeof(FprAttnDenseInvRopeArgs))},
  };
  static const FprExtTable t = {FPR_CABI_VERSION, DENSE_PDL ? "head_dense_pdl" : "head_dense",
                                "b85def80e80484c97d03ea1fe5067272", 4, fns};
  return &t;
}

// ---- pybind (signatures, checks and behaviour unchanged; the launch is the C function above)
static FprCall call_of(const torch::Tensor& t) {
  FprCall c;
  c.stream = at::cuda::getCurrentCUDAStream();
  c.device = t.get_device();
  c.rank = 0;
  return c;
}

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
  FprAttnDenseNormRopeArgs a{};
  a.call = call_of(x);
  a.x = x.data_ptr();
  a.ldx = x.stride(0);
  a.w = w.data_ptr<float>();
  a.out = out.data_ptr();
  a.ldo = out.stride(0);
  a.d = D;
  a.rows = R;
  a.eps = static_cast<float>(eps);
  a.pos = reinterpret_cast<const int64_t*>(pp);
  a.cos_sin = cp;
  FPR_CABI_RECORD("head_dense.norm_rope", a);   // rule 9 (fpr_cabi_rec.h)
  TORCH_CHECK(kd::c_norm_rope(&a) == FPR_OK, "norm_rope launch");
}

void norm_rope2(torch::Tensor xq, torch::Tensor wq, torch::Tensor oq, torch::Tensor xk, torch::Tensor wk, torch::Tensor ok,
                double eps, torch::Tensor pos, torch::Tensor cos_sin) {
  const int R = static_cast<int>(xq.size(0));
  TORCH_CHECK(xk.size(0) == R && oq.size(0) == R && ok.size(0) == R, "rows");
  TORCH_CHECK(xq.dtype() == torch::kBFloat16 && xq.stride(1) == 1 && oq.dtype() == torch::kBFloat16 && oq.stride(1) == 1, "q");
  TORCH_CHECK(xk.dtype() == torch::kBFloat16 && xk.stride(1) == 1 && ok.dtype() == torch::kBFloat16 && ok.stride(1) == 1, "kv");
  TORCH_CHECK(wq.dtype() == torch::kFloat32 && wq.numel() == xq.size(1) && wk.dtype() == torch::kFloat32 && wk.numel() == xk.size(1), "w");
  TORCH_CHECK(cos_sin.dtype() == torch::kFloat32 && cos_sin.size(1) == 64 && xk.size(1) >= 64, "cos_sin");
  FprAttnDenseNormRope2Args a{};
  a.call = call_of(xq);
  a.xq = xq.data_ptr();
  a.ldxq = xq.stride(0);
  a.wq = wq.data_ptr<float>();
  a.oq = oq.data_ptr();
  a.ldoq = oq.stride(0);
  a.dq = static_cast<int>(xq.size(1));
  a.xk = xk.data_ptr();
  a.ldxk = xk.stride(0);
  a.wk = wk.data_ptr<float>();
  a.ok = ok.data_ptr();
  a.ldok = ok.stride(0);
  a.dk = static_cast<int>(xk.size(1));
  a.rows = R;
  a.eps = static_cast<float>(eps);
  a.pos = pos.data_ptr<int64_t>();
  a.cos_sin = cos_sin.data_ptr<float>();
  FPR_CABI_RECORD("head_dense.norm_rope2", a);   // rule 9 (fpr_cabi_rec.h)
  TORCH_CHECK(kd::c_norm_rope2(&a) == FPR_OK, "norm_rope2 launch");
}

void rope_q_insert(torch::Tensor q, torch::Tensor kv, torch::Tensor pos, torch::Tensor cos_sin, torch::Tensor slot_mapping,
                   torch::Tensor cache, int64_t page_stride, int64_t shift, bool do_insert) {
  const int T = static_cast<int>(q.size(0));
  const int nh = static_cast<int>(q.size(1));
  TORCH_CHECK(q.is_contiguous() && (nh == 16 || nh == 64) && q.size(2) == 512 && q.dtype() == torch::kBFloat16, "q");
  TORCH_CHECK(kv.is_contiguous() && kv.size(0) == T && kv.size(1) == 512, "kv");
  TORCH_CHECK(cos_sin.dtype() == torch::kFloat32 && cos_sin.size(1) == 64, "cos_sin");
  FprAttnDenseRopeQInsertArgs a{};
  a.call = call_of(q);
  a.q = q.data_ptr();
  a.kv = kv.data_ptr();
  a.pos = pos.data_ptr<int64_t>();
  a.cos_sin = cos_sin.data_ptr<float>();
  a.slot_mapping = slot_mapping.data_ptr<int64_t>();
  a.cache = cache.data_ptr<uint8_t>();
  a.page_stride = page_stride;
  a.shift = shift;
  a.t = T;
  a.nh = nh;
  a.do_insert = do_insert ? 1 : 0;
  FPR_CABI_RECORD("head_dense.rope_q_insert", a);   // rule 9 (fpr_cabi_rec.h)
  TORCH_CHECK(kd::c_rope_q_insert(&a) == FPR_OK, "rope_q_insert launch");
}

void inv_rope(torch::Tensor o, torch::Tensor pos, torch::Tensor cos_sin) {
  const int T = static_cast<int>(o.size(0));
  const int nh = static_cast<int>(o.size(1));
  TORCH_CHECK(o.is_contiguous() && (nh == 16 || nh == 64) && o.size(2) == 512 && o.dtype() == torch::kBFloat16, "o");
  FprAttnDenseInvRopeArgs a{};
  a.call = call_of(o);
  a.o = o.data_ptr();
  a.pos = pos.data_ptr<int64_t>();
  a.cos_sin = cos_sin.data_ptr<float>();
  a.t = T;
  a.nh = nh;
  FPR_CABI_RECORD("head_dense.inv_rope", a);   // rule 9 (fpr_cabi_rec.h)
  TORCH_CHECK(kd::c_inv_rope(&a) == FPR_OK, "inv_rope launch");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_q_insert", &rope_q_insert, "K2: RoPE(q) in place + window KV record insert");
  m.def("inv_rope", &inv_rope, "K4: inverse RoPE of the attention output in place");
  m.def("norm_rope", &norm_rope, "RMSNorm (reference rounding), optionally + RoPE of the last 64 dims (kv path)");
  m.def("norm_rope2", &norm_rope2, "both attention-prologue norms (q plain, kv + RoPE) in one launch");
}
