// pagecopy: one launch copies a list of whole pages (src page ids -> dst page ids) between two
// pool base pointers, 16 B per load / store. Replaces serve's per-run copy_ (Worker.copies): with scattered free lists
// almost every run is one page, and an HtoH pinned -> pinned copy_ costs ~60 us of host time per page.
//
// The pointers are raw device addresses of the pools (paged: the tier range = cuMemHostRegister'ed (PORTABLE | DEVICEMAP)
// host memory with its leading pages mbind'ed to HBM; window / state: cudaMalloc'ed HBM on the owning GPU, read here over
// peer access). A pure byte copy: the destination bytes equal the source bytes (bitwise by construction).
// C ABI: the pybind function and the table entry share launch(): same kernel, same
// grid, same arguments (fpr_cabi.h rule 6); the kernel itself is unchanged (SASS gate).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

#include "pagecopy_cabi.h"

__global__ void k_pagecopy(const uint4* __restrict__ src, uint4* __restrict__ dst, const int64_t* __restrict__ sp,
                           const int64_t* __restrict__ dp, int64_t words) {
  const int64_t p = blockIdx.y;
  const uint4* s = src + sp[p] * words;
  uint4* d = dst + dp[p] * words;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (int64_t)gridDim.x * blockDim.x)
    d[i] = s[i];
}

namespace pc {

static cudaError_t launch(const FprPagecopyArgs& a) {
  if (a.n == 0) return cudaSuccess;
  const int64_t words = a.page_bytes / 16;
  const int threads = 256;
  int64_t bx = (words + threads * 4 - 1) / (threads * 4);          // ~4 words per thread
  if (bx < 1) bx = 1;
  if (bx > 64) bx = 64;
  for (int64_t p0 = 0; p0 < a.n; p0 += 65535) {                     // gridDim.y <= 65535
    const int64_t m = std::min<int64_t>(65535, a.n - p0);
    dim3 grid((unsigned)bx, (unsigned)m);
    k_pagecopy<<<grid, threads, 0, a.call.stream>>>(reinterpret_cast<const uint4*>(a.src), reinterpret_cast<uint4*>(a.dst),
                                                     a.sp + p0, a.dp + p0, words);
  }
  return cudaGetLastError();
}

static int c_copy(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprPagecopyArgs*>(p);
  if (a.page_bytes <= 0 || a.page_bytes % 16 || reinterpret_cast<uintptr_t>(a.src) % 16 ||
      reinterpret_cast<uintptr_t>(a.dst) % 16 || a.n < 0 || (a.n > 0 && (a.sp == nullptr || a.dp == nullptr)))
    return FPR_E_ARG;
  return launch(a) == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}

}  // namespace pc

extern "C" __attribute__((visibility("default"))) const FprExtTable* fpr_pagecopy_table(void) {
  static const FprFnEntry fns[] = {
      {"pagecopy.copy", pc::c_copy, nullptr, static_cast<int64_t>(sizeof(FprPagecopyArgs))},
  };
  static const FprExtTable t = {FPR_CABI_VERSION, "pagecopy", "pagecopy.cu", 1, fns};
  return &t;
}

// src_ptr / dst_ptr: pool base addresses; sp / dp: int64 page ids on the CURRENT device (the launch device, normally the
// destination's GPU); page_bytes: bytes per page (multiple of 16). Runs on the current stream.
void pagecopy(int64_t src_ptr, int64_t dst_ptr, torch::Tensor sp, torch::Tensor dp, int64_t page_bytes) {
  TORCH_CHECK(page_bytes > 0 && page_bytes % 16 == 0, "pagecopy: page_bytes must be a positive multiple of 16");
  TORCH_CHECK(src_ptr % 16 == 0 && dst_ptr % 16 == 0, "pagecopy: pool base pointers must be 16-byte aligned");
  TORCH_CHECK(sp.is_cuda() && dp.is_cuda() && sp.device() == dp.device(), "pagecopy: page ids on one CUDA device");
  TORCH_CHECK(sp.scalar_type() == torch::kInt64 && dp.scalar_type() == torch::kInt64, "pagecopy: int64 page ids");
  TORCH_CHECK(sp.is_contiguous() && dp.is_contiguous() && sp.numel() == dp.numel(), "pagecopy: matching contiguous id lists");
  const int64_t n = sp.numel();
  if (n == 0) return;
  const c10::cuda::CUDAGuard guard(sp.device());
  FprPagecopyArgs a{};
  a.call.stream = at::cuda::getCurrentCUDAStream();
  a.call.device = sp.device().index();
  a.call.rank = 0;
  a.src = reinterpret_cast<const void*>(src_ptr);
  a.dst = reinterpret_cast<void*>(dst_ptr);
  a.sp = sp.data_ptr<int64_t>();
  a.dp = dp.data_ptr<int64_t>();
  a.n = n;
  a.page_bytes = page_bytes;
  C10_CUDA_CHECK(pc::launch(a));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pagecopy", &pagecopy, "copy whole pages src_ptr[sp] -> dst_ptr[dp] (page_bytes each) on the current stream");
}
