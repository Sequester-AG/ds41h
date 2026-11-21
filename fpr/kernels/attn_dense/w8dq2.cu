// w8dq2.cu: w8dq.cu's exact bf16 [N, K] of a w8a16-packed fp8 weight, as a PERSISTENT kernel
// with a bulk-copy (TMA engine) ring, so each CTA's loads overlap its own convert + stores. w8dq.cu's k_dq loads its whole
// 32,896-B block, syncs, converts and stores: load and store phases never overlap inside a CTA and the grid runs 1.6 waves
// Per item (tile, chunk) the arithmetic, the lane map and the stores are k_dq's code
// verbatim (same bits by construction; w8dq2_test.py checks bitwise vs w8dq.cu and vs the raw checkpoint dequant).
//   grid = SMs x CTAs per SM (occupancy), item i = blockIdx.x + j * gridDim.x in packed memory order (chunk-major, as k_dq's
//   blockIdx (tile, chunk)); a ring of NS stages of 32,896 B + one mbarrier each; thread 0 refills stage s with item j + NS
//   after the CTA's __syncthreads that ends item j's reads of it.
// Same API as w8dq.cu: dequant_packed(packed uint8 [K / 128, N / 256, 32896], out bf16 [N, >= K]).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <atomic>
#include <cstdint>

namespace w8dq2 {
constexpr int TR = 256, KC = 128, HW = 128 * 128, HB = HW + 64, CHUNK = 2 * HB;
#ifndef W8DQ2_NS
#define W8DQ2_NS 2
#endif
constexpr int NS = W8DQ2_NS;
constexpr int SMEM = NS * CHUNK + 64;   // stages + mbarriers

__device__ __forceinline__ float e4m3f(uint32_t b) {   // = w8dq.cu
  const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
  return __half2float(__half(h));
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {   // = w8dq.cu
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t s32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void load(uint8_t* dst, const uint8_t* src, uint64_t* bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(s32(bar)), "r"(CHUNK) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(s32(dst)), "l"(src), "r"(CHUNK), "r"(s32(bar)) : "memory");
}
__device__ __forceinline__ void wait(uint64_t* bar, uint32_t parity) {
  uint32_t done = 0;
  while (!done) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(done) : "r"(s32(bar)), "r"(parity) : "memory");
  }
}

__global__ void __launch_bounds__(256) k_dq2(const uint8_t* __restrict__ packed, int tiles, int items, int64_t ldo,
                                             __nv_bfloat16* __restrict__ out) {
  extern __shared__ __align__(128) uint8_t dsm[];
  uint64_t* bar = reinterpret_cast<uint64_t*>(dsm + NS * CHUNK);
  const int G = gridDim.x;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(s32(bar + s)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int s = 0; s < NS; ++s) {
      const int it = blockIdx.x + s * G;
      if (it < items) load(dsm + s * CHUNK, packed + static_cast<int64_t>(it) * CHUNK, bar + s);
    }
  for (int j = 0;; ++j) {
    const int it = blockIdx.x + j * G;
    if (it >= items) break;
    const int s = j % NS;
    const uint8_t* sm = dsm + s * CHUNK;
    wait(bar + s, static_cast<uint32_t>((j / NS) & 1));
    const int tile = it % tiles, ch = it / tiles;
    // ---- k_dq's per-block body, verbatim (sm = this item's 32,896-B block)
    for (int task = threadIdx.x; task < 128 * 16; task += 256) {
      const int o = task & 15, rp = task >> 4;   // 8-k octet of the chunk; row pair (r, r + 8)
      const int g = rp & 7, w = (rp >> 3) & 3, m = (rp >> 5) & 1, c = rp >> 6;
      const int r = c * 128 + m * 64 + w * 16 + g;
      const int kb = o >> 2, s16 = (o >> 1) & 1, khi = o & 1;
      const int base = c * HB + kb * 4096 + m * 2048 + w * 512 + g * 64 + (s16 * 2 + khi) * 4;
      const int rb = r >> 5;   // r and r + 8 share the 32-row block (r % 16 < 8)
      const float sc = __uint_as_float(static_cast<uint32_t>(sm[(rb >> 2) * HB + HW + kb * 4 + (rb & 3)]) << 23);
      uint4 v0, v1;
      uint32_t* a = reinterpret_cast<uint32_t*>(&v0);
      uint32_t* b = reinterpret_cast<uint32_t*>(&v1);
#pragma unroll
      for (int t = 0; t < 4; ++t) {   // k = 8 o + 2 t, + 1
        const uint32_t wd = *reinterpret_cast<const uint32_t*>(sm + base + 16 * t);
        a[t] = pack2(e4m3f(wd & 0xffu) * sc, e4m3f((wd >> 8) & 0xffu) * sc);
        b[t] = pack2(e4m3f((wd >> 16) & 0xffu) * sc, e4m3f(wd >> 24) * sc);
      }
      const int64_t col = static_cast<int64_t>(ch) * KC + 8 * o;
      const int64_t row = static_cast<int64_t>(tile) * TR + r;
      *reinterpret_cast<uint4*>(out + row * ldo + col) = v0;
      *reinterpret_cast<uint4*>(out + (row + 8) * ldo + col) = v1;
    }
    // ---- end of k_dq's body
    __syncthreads();   // every thread is done reading stage s
    if (threadIdx.x == 0) {
      const int nx = it + NS * G;
      if (nx < items) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // generic reads of stage s before the async refill
        load(dsm + s * CHUNK, packed + static_cast<int64_t>(nx) * CHUNK, bar + s);
      }
    }
  }
}

// per device: the dynamic-SMEM opt-in (cudaFuncSetAttribute is per device) and the grid (SMs x resident CTAs)
std::atomic<int> g_grid[64];
int grid_of(int dev) {
  int g = g_grid[dev].load(std::memory_order_acquire);
  if (g) return g;
  C10_CUDA_CHECK(cudaFuncSetAttribute(k_dq2, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  int per = 0, nsm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_dq2, 256, SMEM));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  g = nsm * (per > 0 ? per : 1);
  g_grid[dev].store(g, std::memory_order_release);
  return g;
}
}  // namespace w8dq2

// grid > 0 (test knob): that many persistent CTAs instead of SMs x resident CTAs per SM
void dequant_packed(torch::Tensor packed, torch::Tensor out, int64_t grid_req) {
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8 && packed.dim() == 3 && packed.size(2) == w8dq2::CHUNK &&
                  packed.is_contiguous() && reinterpret_cast<uintptr_t>(packed.data_ptr()) % 16 == 0,
              "packed: uint8 [K/128, N/256, 32896] contiguous, 16-B aligned");
  const int nchunk = packed.size(0), tiles = packed.size(1);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.stride(1) == 1 &&
                  out.size(0) == static_cast<int64_t>(tiles) * w8dq2::TR && out.size(1) >= static_cast<int64_t>(nchunk) * w8dq2::KC &&
                  out.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "out: bf16 [N, >= K], 16-B rows");
  if (!nchunk || !tiles) return;
  const c10::cuda::CUDAGuard guard(packed.device());
  const int items = nchunk * tiles;
  const int gmax = w8dq2::grid_of(packed.get_device());
  const int grid = std::min<int64_t>(items, grid_req > 0 ? std::min<int64_t>(grid_req, gmax) : gmax);
  w8dq2::k_dq2<<<grid, 256, w8dq2::SMEM, c10::cuda::getCurrentCUDAStream()>>>(
      packed.data_ptr<uint8_t>(), tiles, items, out.stride(0), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant_packed", &dequant_packed, "w8dq2: w8dq.cu's exact bf16 [N, K] (persistent, bulk-copy ring)",
        pybind11::arg("packed"), pybind11::arg("out"), pybind11::arg("grid") = 0);
  m.def("grid", [](int64_t dev) { return w8dq2::grid_of(static_cast<int>(dev)); }, "the default grid on a device");
}
