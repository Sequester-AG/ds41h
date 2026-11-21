// act_quant.cuh: THE activation quantization of the fp8 activation paths (W8A8 dense, W4A8 experts, the C1 persistent
// kernel); One definition, include it; do not re-implement.
//
// Numerics = ds41h/ref/quant.py fp8_quant(x, block=32, amax_min=1e-4, scale_fmt="ue8m0") = the official
// inference/kernel.py act_quant(round_scale=True). Per row, per block
// of 32 consecutive elements along K (the weights' ue8m0 32 x 32 blocks align with it):
//   amax = max(max |x| over the block, fp32(1e-4))            (bf16 / fp32 -> fp32, exact)
//   s    = 2^ceil(log2(fl32(amax * fl32(1/448))))             (kernel.py:22-37 bit trick on the fp32 product)
//   q    = e4m3(x / s), round to nearest even, satfinite      (x / s is exact: a power-of-2 division)
//   the block's scale is stored as its ue8m0 byte: e + 127 for s = 2^e (the checkpoint's weight-scale format); e >= -22
// Dequant: x ~= q * s. A GEMM block partial (sum over the block's 32 k of qx * qw) is multiplied by sx * sw (powers of 2:
// exact) and accumulated in fp32.
// Reference: fpr/kernels/w8a8/act_quant_ref.py.
#pragma once
#include <cuda_fp8.h>

namespace aq {
constexpr int BLOCK = 32;
constexpr float FP8_MAX = 448.f;

constexpr float AMAX_MIN = 1e-4f;                 // fp32(1e-4), as torch's clamp_min(1e-4) on an fp32 tensor
constexpr float FP8_MAX_INV = 1.f / 448.f;         // fp32(1/448) = 0x3B124925, kernel.py's fp8_max_inv

// e with s = 2^e = pow2_ceil(fl32(max(amax, 1e-4) * fl32(1/448))) (amax >= 0, finite); the product is a normal fp32
// (>= 2.2e-11), so ceil(log2) = its exponent, + 1 unless the mantissa is 0 (an exact power of 2)
__device__ __forceinline__ int scale_exp(float amax) {
  const float y = __fmul_rn(fmaxf(amax, AMAX_MIN), FP8_MAX_INV);
  const unsigned b = __float_as_uint(y);
  return (int)((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u);
}
__device__ __forceinline__ float exp2i(int e) { return ldexpf(1.f, e); }
__device__ __forceinline__ unsigned char e8m0_byte(int e) { return (unsigned char)(e + 127); }
__device__ __forceinline__ float e8m0_value(unsigned char b) { return __uint_as_float((unsigned)b << 23); }   // 2^(b - 127)

// 4 values (already divided by s) -> 4 e4m3 bytes packed little-endian (element 0 in bits 0..7)
__device__ __forceinline__ unsigned pack4(float a, float b, float c, float d) {
  return (unsigned)__nv_cvt_float_to_fp8(a, __NV_SATFINITE, __NV_E4M3) |
         ((unsigned)__nv_cvt_float_to_fp8(b, __NV_SATFINITE, __NV_E4M3) << 8) |
         ((unsigned)__nv_cvt_float_to_fp8(c, __NV_SATFINITE, __NV_E4M3) << 16) |
         ((unsigned)__nv_cvt_float_to_fp8(d, __NV_SATFINITE, __NV_E4M3) << 24);
}

// Warp-cooperative form: each lane holds 8 consecutive elements v[8] of one row; the 4 lanes 4b .. 4b + 3 (aligned
// group of 4) hold one 32-element block. Every lane of the warp must call it (shuffles). Returns the 8 e4m3 bytes of
// this lane (uint2) and the block's scale exponent e (the same in the 4 lanes). Lanes with no data pass zeros.
__device__ __forceinline__ uint2 quant8_group4(const float v[8], int& e_out) {
  float m = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) m = fmaxf(m, fabsf(v[j]));
  m = fmaxf(m, __shfl_xor_sync(~0u, m, 1));
  m = fmaxf(m, __shfl_xor_sync(~0u, m, 2));
  const int e = scale_exp(m);
  e_out = e;
  const float inv = exp2i(-e);
  return make_uint2(pack4(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv), pack4(v[4] * inv, v[5] * inv, v[6] * inv, v[7] * inv));
}
}  // namespace aq
