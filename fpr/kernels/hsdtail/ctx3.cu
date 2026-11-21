// ctx3.cu: the drafter's context KV epilogue of its 3 blocks in ONE launch, for
// hsdt_c1e.py's ctx3 switch. Per (row r, block i): out_i[r] = RoPE(RMSNorm(kv3[r, i*512:(i+1)*512]) * kv_norm_i) and the
// window KV record of that row into block i's paged cache at slot_mapping[r] -- the same device code, copied verbatim,
// as kernels/head/vendor/dense.cu's norm_rope_kernel (norm_rope_row) followed by rope_q_insert_kernel's insert part.
// rope_q_insert's in-place RoPE of the q rows is skipped: the context path passes the drafter's dq rows, which are all
// zeros and stay zeros (RoPE of 0 is 0). Bitwise equal to the 3 x (norm_rope + rope_q_insert) launches it replaces.
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cstdint>

namespace c3 {

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }

// = dense.cu norm_rope_row (verbatim)
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

struct Args {
  const __nv_bfloat16* kv3;   // [R, 3 * 512], row stride ld
  long long ld;
  const float* w[3];          // kv_norm of block i (fp32 [512])
  __nv_bfloat16* out[3];      // the normed + roped rows of block i ([R, 512] contiguous)
  uint8_t* cache[3];          // block i's paged window cache
  const long long* pos;       // ctx_pos [R]
  const float* cos_sin;       // [max_pos, 64]
  const long long* slot;      // ctx_slot [R]
  long long page_stride;
  int shift;
  float eps;
};

// grid (R, 3), block 256
__global__ void __launch_bounds__(256) k_norm_insert3(Args a) {
  const int r = blockIdx.x, i = blockIdx.y, tid = threadIdx.x;
  norm_rope_row(a.kv3 + i * 512, a.ld, a.w[i], a.out[i], 512, 512, a.eps, a.pos, a.cos_sin, r);
  __syncthreads();   // the row's out values (written by all threads) before the insert reads them
  // = dense.cu rope_q_insert_kernel's insert part (verbatim, kv = out[i])
  const long long slot = __ldg(a.slot + r);
  if (slot < 0) return;
  const int w = tid >> 5, lane = tid & 31, shift = a.shift;
  const long long pg = slot >> shift, ps = slot & ((1LL << shift) - 1);
  uint8_t* base = a.cache[i] + pg * a.page_stride;
  uint8_t* data = base + ps * 576;
  uint8_t* scl = base + (576LL << shift) + ps * 8;
  const __nv_bfloat16* kt = a.out[i] + static_cast<long long>(r) * 512;
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
    reinterpret_cast<unsigned*>(data + 448)[lane] = reinterpret_cast<const unsigned*>(kt + 448)[lane];
    if (lane == 0) scl[7] = 0;
  }
}

}  // namespace c3

// ---- C ABI (ctx3_cabi.h; fpr_cabi.h rules): the launch of pybind norm_insert3, called by it and by the C++ driver
#include "ctx3_cabi.h"
#include "../../engine/mixed_cxx/fpr_cabi_rec.h"
FPR_CABI_RECORD_DECL(ctx3)

static int norm_insert3_core(const FprCtx3NormInsert3Args* p) {   // "ctx3.norm_insert3"
  try {
    if (p->R < 0) return FPR_E_ARG;
    c3::Args a;
    a.kv3 = reinterpret_cast<const __nv_bfloat16*>(p->kv3);
    a.ld = p->ld;
    for (int i = 0; i < 3; ++i) {
      a.w[i] = p->w[i];
      a.out[i] = reinterpret_cast<__nv_bfloat16*>(p->out[i]);
      a.cache[i] = reinterpret_cast<uint8_t*>(p->cache[i]);
    }
    a.pos = reinterpret_cast<const long long*>(p->pos);
    a.cos_sin = p->cos_sin;
    a.slot = reinterpret_cast<const long long*>(p->slot);
    a.page_stride = p->page_stride;
    a.shift = p->shift;
    a.eps = p->eps;
    if (p->R == 0) return FPR_OK;
    c3::k_norm_insert3<<<dim3(p->R, 3), 256, 0, p->c.stream>>>(a);
    return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
  } catch (...) {
    return FPR_E_CUDA;
  }
}

// kv3 [R, 3 x 512] (row stride any, column stride 1); w / out / cache: 3 each; pos / slot int64 [R]
void norm_insert3(torch::Tensor kv3, std::vector<torch::Tensor> w, std::vector<torch::Tensor> out, torch::Tensor pos,
                  torch::Tensor cos_sin, torch::Tensor slot, std::vector<torch::Tensor> cache, int64_t page_stride,
                  int64_t shift, double eps) {
  const int R = static_cast<int>(kv3.size(0));
  TORCH_CHECK(kv3.dtype() == torch::kBFloat16 && kv3.stride(1) == 1 && kv3.size(1) == 3 * 512, "kv3");
  TORCH_CHECK(w.size() == 3 && out.size() == 3 && cache.size() == 3, "3 blocks");
  TORCH_CHECK(cos_sin.dtype() == torch::kFloat32 && cos_sin.size(1) == 64, "cos_sin");
  TORCH_CHECK(pos.dtype() == torch::kInt64 && slot.dtype() == torch::kInt64 && pos.size(0) >= R && slot.size(0) >= R, "pos / slot");
  FprCtx3NormInsert3Args ca{};
  ca.c = FprCall{c10::cuda::getCurrentCUDAStream(), (int)kv3.get_device(), 0};
  ca.kv3 = kv3.data_ptr();
  ca.ld = kv3.stride(0);
  ca.R = R;
  ca.shift = static_cast<int32_t>(shift);
  for (int i = 0; i < 3; ++i) {
    TORCH_CHECK(w[i].dtype() == torch::kFloat32 && w[i].numel() == 512, "w");
    TORCH_CHECK(out[i].dtype() == torch::kBFloat16 && out[i].is_contiguous() && out[i].size(0) >= R && out[i].size(1) == 512, "out");
    TORCH_CHECK(cache[i].dtype() == torch::kUInt8, "cache");
    ca.w[i] = w[i].data_ptr<float>();
    ca.out[i] = out[i].data_ptr();
    ca.cache[i] = cache[i].data_ptr<uint8_t>();
  }
  ca.pos = pos.data_ptr<int64_t>();
  ca.cos_sin = cos_sin.data_ptr<float>();
  ca.slot = slot.data_ptr<int64_t>();
  ca.page_stride = page_stride;
  ca.eps = static_cast<float>(eps);
  FPR_CABI_RECORD("ctx3.norm_insert3", ca);
  TORCH_CHECK(norm_insert3_core(&ca) == FPR_OK, "ctx3: launch error");
}

extern "C" {
static int c_norm_insert3(const void* a) { return norm_insert3_core((const FprCtx3NormInsert3Args*)a); }
static const FprFnEntry k_ctx3_fns[] = {
    {"ctx3.norm_insert3", c_norm_insert3, nullptr, (int64_t)sizeof(FprCtx3NormInsert3Args)}};
static const FprExtTable k_ctx3_table = {FPR_CABI_VERSION, "ctx3", "", 1, k_ctx3_fns};
__attribute__((visibility("default"))) const FprExtTable* fpr_ctx3_table(void) { return &k_ctx3_table; }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("norm_insert3", &norm_insert3, "ctx3: the 3 drafter blocks' context KV norm_rope + window-record insert, one launch");
}
