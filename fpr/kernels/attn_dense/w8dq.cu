// w8dq.cu: the EXACT bf16 [N, K] value of a w8a16-packed fp8 weight
// (w8a16.py repack: [K / 128 chunks][N / 256 tiles][32,896 B], two 16,448-B halves of RS-wgmma fragments + 32 x 32 ue8m0
// scales) -- the inverse of repack times the block scale: out[n, k] = e4m3(w[n, k]) * 2^(s[n / 32, k / 32] - 127), exact in
// bf16 (3-bit mantissa x a power of two in bf16's range; w8a16's check_scales keeps the bytes in [16, 134]).
// Layout (w8a16.py _maps / _scale_pos): fragment byte q of half c (rows 128 c ..): kb = q / 4096, m = (q / 2048) % 2,
// w = (q / 512) % 4, lane ln = (q / 16) % 32 = 4 g + t, b = q % 16; word = b / 4 = 2 s16 + khi, bi = b % 4;
// row = 128 c + 64 m + 16 w + g + 8 (bi / 2), k = 32 kb + 16 s16 + 8 khi + 2 t + bi % 2. Scale of (32-row block rb, kb) at
// (rb / 4) * 16,448 + 16,384 + 4 kb + rb % 4. So ONE 32-bit word = a k pair of rows g and g + 8.
// One CTA per (tile, chunk): the 32,896-B block into shared memory (coalesced), then a lane per (row pair, 8-k octet) reads
// 4 words and stores 16 B of each row (consecutive lanes = consecutive octets of the same row: coalesced stores).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace w8dq {
constexpr int TR = 256, KC = 128, HW = 128 * 128, HB = HW + 64, CHUNK = 2 * HB;

__device__ __forceinline__ float e4m3f(uint32_t b) {
  const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
  return __half2float(__half(h));
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {   // exact (see the header)
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__global__ void __launch_bounds__(256) k_dq(const uint8_t* __restrict__ packed, int tiles, int64_t ldo,
                                            __nv_bfloat16* __restrict__ out) {
  __shared__ __align__(16) uint8_t sm[CHUNK];
  const int tile = blockIdx.x, ch = blockIdx.y;
  const uint4* src = reinterpret_cast<const uint4*>(packed + (static_cast<int64_t>(ch) * tiles + tile) * CHUNK);
  for (int i = threadIdx.x; i < CHUNK / 16; i += 256) reinterpret_cast<uint4*>(sm)[i] = __ldg(src + i);
  __syncthreads();
  for (int task = threadIdx.x; task < 128 * 16; task += 256) {
    const int o = task & 15, rp = task >> 4;   // 8-k octet of the chunk; row pair (r, r + 8)
    const int g = rp & 7, w = (rp >> 3) & 3, m = (rp >> 5) & 1, c = rp >> 6;
    const int r = c * 128 + m * 64 + w * 16 + g;
    const int kb = o >> 2, s16 = (o >> 1) & 1, khi = o & 1;
    const int base = c * HB + kb * 4096 + m * 2048 + w * 512 + g * 64 + (s16 * 2 + khi) * 4;
    const int rb = r >> 5;   // r and r + 8 share the 32-row block (r % 16 < 8)
    const float s = __uint_as_float(static_cast<uint32_t>(sm[(rb >> 2) * HB + HW + kb * 4 + (rb & 3)]) << 23);
    uint4 v0, v1;
    uint32_t* a = reinterpret_cast<uint32_t*>(&v0);
    uint32_t* b = reinterpret_cast<uint32_t*>(&v1);
#pragma unroll
    for (int t = 0; t < 4; ++t) {   // k = 8 o + 2 t, + 1
      const uint32_t wd = *reinterpret_cast<const uint32_t*>(sm + base + 16 * t);
      a[t] = pack2(e4m3f(wd & 0xffu) * s, e4m3f((wd >> 8) & 0xffu) * s);
      b[t] = pack2(e4m3f((wd >> 16) & 0xffu) * s, e4m3f(wd >> 24) * s);
    }
    const int64_t col = static_cast<int64_t>(ch) * KC + 8 * o;
    const int64_t row = static_cast<int64_t>(tile) * TR + r;
    *reinterpret_cast<uint4*>(out + row * ldo + col) = v0;
    *reinterpret_cast<uint4*>(out + (row + 8) * ldo + col) = v1;
  }
}
}  // namespace w8dq

// packed uint8 [K / 128, N / 256, 32896] (w8a16.Gemm.packed) -> out bf16 [N, >= K] (unit inner stride, 16-B rows)
void dequant_packed(torch::Tensor packed, torch::Tensor out) {
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kUInt8 && packed.dim() == 3 && packed.size(2) == w8dq::CHUNK &&
                  packed.is_contiguous(), "packed: uint8 [K/128, N/256, 32896] contiguous");
  const int nchunk = packed.size(0), tiles = packed.size(1);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.stride(1) == 1 &&
                  out.size(0) == static_cast<int64_t>(tiles) * w8dq::TR && out.size(1) >= static_cast<int64_t>(nchunk) * w8dq::KC &&
                  out.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "out: bf16 [N, >= K], 16-B rows");
  if (!nchunk || !tiles) return;
  const c10::cuda::CUDAGuard guard(packed.device());
  w8dq::k_dq<<<dim3(tiles, nchunk), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      packed.data_ptr<uint8_t>(), tiles, out.stride(0), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant_packed", &dequant_packed, "exact bf16 [N, K] of a w8a16-packed fp8 weight (inverse of w8a16.repack)");
}
