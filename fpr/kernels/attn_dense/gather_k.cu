// gather_k.cu -- DSv4 fp8 K-cache dequantize + gather: vLLM's
// dequantize_and_gather_k_cache (CuTe DSL dequant_gather_k_cutedsl, fp8_dim 448, group 64) without Python, for the C++
// mixed-step driver's prefill attention (stage 1c, attn_prefill.attend). The per-element arithmetic is the CuTe kernel's
// own PTX (vllm.cute_utils.cvt.fp8x4_to_bf16x4 + _bf16x2_mul), so the output is bit-identical; work split and staging
// differ (one warp per gathered row, direct loads; every output element is written exactly once either way).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "gather_k_cabi.h"

namespace gk {

constexpr int kData = 576;      // 448 e4m3 + 64 bf16 bytes per token
constexpr int kScale = 8;       // ue8m0 bytes per token (7 used + 1 pad)
constexpr int kWarps = 8;

// 4 e4m3 -> 4 bf16 (two bf16x2), exactly cvt.fp8x4_to_bf16x4: e4m3x2 -> f16x2 -> bf16 per element
__device__ __forceinline__ void fp8x4_to_bf16x4(uint32_t x, uint32_t& lo, uint32_t& hi) {
  asm("{\n\t"
      ".reg .b16 x0, x1;\n\t"
      ".reg .b16 t00, t01, t10, t11;\n\t"
      "mov.b32 {x0, x1}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, x0;\n\t"
      "cvt.rn.f16x2.e4m3x2 %1, x1;\n\t"
      "mov.b32 {t00, t01}, %0;\n\t"
      "mov.b32 {t10, t11}, %1;\n\t"
      "cvt.rn.bf16.f16 t00, t00;\n\t"
      "cvt.rn.bf16.f16 t01, t01;\n\t"
      "cvt.rn.bf16.f16 t10, t10;\n\t"
      "cvt.rn.bf16.f16 t11, t11;\n\t"
      "mov.b32 %0, {t00, t01};\n\t"
      "mov.b32 %1, {t10, t11};\n\t"
      "}\n"
      : "=r"(lo), "=r"(hi)
      : "r"(x));
}

__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

// 8 e4m3 (uint2) x the scale byte s -> 8 bf16 (uint4)
__device__ __forceinline__ uint4 dq8(uint2 v, uint32_t s) {
  const uint32_t sb = (s << 23) | (s << 7);     // bf16x2 {2^(s-127), 2^(s-127)}
  uint4 o;
  uint32_t a, b;
  fp8x4_to_bf16x4(v.x, a, b);
  o.x = bf16x2_mul(a, sb);
  o.y = bf16x2_mul(b, sb);
  fp8x4_to_bf16x4(v.y, a, b);
  o.z = bf16x2_mul(a, sb);
  o.w = bf16x2_mul(b, sb);
  return o;
}

__global__ void __launch_bounds__(kWarps * 32) gather_kernel(uint16_t* __restrict__ out, int64_t out_sr, int64_t out_st,
                                                             const uint8_t* __restrict__ kc, int64_t page_stride,
                                                             const int32_t* __restrict__ seq_lens,
                                                             const int32_t* __restrict__ gather_lens,
                                                             const int32_t* __restrict__ bt, int64_t ld_bt, int bs,
                                                             int64_t offset) {
  const int req = blockIdx.x, workers = gridDim.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int seq_len = seq_lens[req];
  const int gl = gather_lens != nullptr ? gather_lens[req] : seq_len;
  const int start = seq_len - gl;
  for (int i = blockIdx.y * kWarps + warp; i < gl; i += workers * kWarps) {
    const int pos = start + i;
    const int64_t page = bt[(int64_t)req * ld_bt + pos / bs];
    const int o = pos % bs;
    const uint8_t* pg = kc + page * page_stride;
    const uint8_t* rec = pg + (int64_t)o * kData;
    const uint8_t* sc = pg + (int64_t)bs * kData + (int64_t)o * kScale;
    uint16_t* dst = out + (int64_t)req * out_sr + (offset + i) * out_st;
    // elements 8 lane .. 8 lane + 7 (all e4m3)
    const uint2 v0 = *reinterpret_cast<const uint2*>(rec + 8 * lane);
    const uint4 d0 = dq8(v0, sc[(8 * lane) >> 6]);
    *reinterpret_cast<uint4*>(dst + 8 * lane) = d0;
    // elements 256 + 8 lane ..: e4m3 for lane < 24 (up to 447), the bf16 RoPE tail (bytes 448 .. 575) for lanes 24 .. 31
    uint4 d1;
    if (lane < 24) {
      const uint2 v1 = *reinterpret_cast<const uint2*>(rec + 256 + 8 * lane);
      d1 = dq8(v1, sc[(256 + 8 * lane) >> 6]);
    } else {
      d1 = *reinterpret_cast<const uint4*>(rec + 448 + 16 * (lane - 24));
    }
    *reinterpret_cast<uint4*>(dst + 256 + 8 * lane) = d1;
  }
}

static int64_t workers_of(int64_t max_rows, int64_t num_reqs) {   // the CuTe kernel's num_workers (launch shape only)
  const int64_t a = (max_rows + 3) / 4, b = (8192 + num_reqs - 1) / num_reqs;
  return std::max<int64_t>(1, std::min(a, b));
}

static cudaError_t launch(const FprGatherkGatherArgs& a) {
  if (a.num_reqs <= 0) return cudaSuccess;
  const dim3 grid(static_cast<unsigned>(a.num_reqs), static_cast<unsigned>(workers_of(a.max_rows, a.num_reqs)));
  gather_kernel<<<grid, kWarps * 32, 0, a.call.stream>>>(static_cast<uint16_t*>(a.out), a.out_sr, a.out_st, a.k_cache,
                                                        a.page_stride, a.seq_lens, a.gather_lens, a.block_table, a.ld_bt,
                                                        static_cast<int>(a.block_size), a.offset);
  return cudaGetLastError();
}

static int c_gather(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprGatherkGatherArgs*>(p);
  if (a.num_reqs < 0 || a.block_size <= 0 || a.offset < 0 || a.out_sr % 8 || a.out_st % 8 || a.page_stride % 16 ||
      reinterpret_cast<uintptr_t>(a.out) % 16 || reinterpret_cast<uintptr_t>(a.k_cache) % 16 ||
      (a.num_reqs > 0 && (a.out == nullptr || a.k_cache == nullptr || a.seq_lens == nullptr || a.block_table == nullptr)))
    return FPR_E_ARG;
  return launch(a) == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}

}  // namespace gk

extern "C" __attribute__((visibility("default"))) const FprExtTable* fpr_gatherk_table(void) {
  static const FprFnEntry fns[] = {
      {"gatherk.gather", gk::c_gather, nullptr, static_cast<int64_t>(sizeof(FprGatherkGatherArgs))},
  };
  static const FprExtTable t = {FPR_CABI_VERSION, "gatherk", "new (gather_k.cu)", 1, fns};
  return &t;
}

// pybind: vLLM's argument order (out [num_reqs, M, 512] bf16 written in place)
void gather(torch::Tensor out, torch::Tensor k_cache, torch::Tensor seq_lens, c10::optional<torch::Tensor> gather_lens,
            torch::Tensor block_table, int64_t block_size, int64_t offset) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 3 && out.size(2) == 512 &&
                  out.stride(2) == 1, "out must be bf16 [num_reqs, M, 512]");
  TORCH_CHECK(k_cache.is_cuda() && k_cache.element_size() == 1 && k_cache.size(-1) == 584, "k_cache: the 584-B fp8 record");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && block_table.scalar_type() == torch::kInt32 &&
                  block_table.dim() == 2 && block_table.stride(1) == 1, "int32 seq_lens / block_table");
  FprGatherkGatherArgs a{};
  a.call.stream = c10::cuda::getCurrentCUDAStream();
  a.call.device = out.get_device();
  a.out = out.data_ptr();
  a.out_sr = out.stride(0);
  a.out_st = out.stride(1);
  a.max_rows = out.size(1) - offset;
  a.k_cache = static_cast<const uint8_t*>(k_cache.data_ptr());
  a.page_stride = k_cache.stride(0);
  a.seq_lens = seq_lens.data_ptr<int32_t>();
  a.gather_lens = gather_lens.has_value() ? gather_lens->data_ptr<int32_t>() : nullptr;
  a.block_table = block_table.data_ptr<int32_t>();
  a.ld_bt = block_table.stride(0);
  a.num_reqs = out.size(0);
  a.block_size = block_size;
  a.offset = offset;
  const int rc = gk::c_gather(&a);
  TORCH_CHECK(rc == FPR_OK, "gatherk.gather: status ", rc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather", &gather, "DSv4 fp8 K-cache dequantize + gather (= vLLM dequantize_and_gather_k_cache, CuTe numerics)",
        py::arg("out"), py::arg("k_cache"), py::arg("seq_lens"), py::arg("gather_lens"), py::arg("block_table"),
        py::arg("block_size"), py::arg("offset"));
}

// ---- fpr_kwarm(dev): cudaFuncGetAttributes on every __global__ this .so launches (incl. the
// template instances), on device dev (current device restored). Under CUDA_MODULE_LOADING=LAZY that loads the module at
// driver init instead of at the first launch on a pool thread. Host code only: the kernels are unchanged.
extern "C" __attribute__((visibility("default"))) int fpr_kwarm(int dev) {
  int prev = 0;
  if (cudaGetDevice(&prev) != cudaSuccess) return -1;
  if (cudaSetDevice(dev) != cudaSuccess) return -1;
  cudaFuncAttributes fa;
  int bad = 0;
  const void* fns[] = {
      (const void*)&gk::gather_kernel,
  };
  for (const void* fn : fns) bad += cudaFuncGetAttributes(&fa, fn) != cudaSuccess;
  cudaGetLastError();
  cudaSetDevice(prev);
  return bad ? -1 : (int)(sizeof(fns) / sizeof(fns[0]));   // kernels warmed (gatherk)
}
