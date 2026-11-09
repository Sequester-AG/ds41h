// Inverse RoPE + 1x128 FP8 quant of the attention output for the FP8 wo_a einsum, any number of 8-head groups.
// ds41h/kernels/fp8_linear.cu's inv_rope_fp8_quant_kernel<8> (production's
// DS41H_PREFILL_WOA_ROPEQ: one CTA per token with all G * 8 head warps, instantiated for TP4's G = 2) with only the launch
// shape changed: one CTA per (token, group), 8 head warps each, so G = 8 (64 heads, the fork's sequence-parallel layout)
// fits the 1,024-thread limit. Per-element arithmetic, layouts and build flags (--use_fast_math) are production's:
// the bytes and scales of a (token, head) equal production's per rank.
// Layout: out [G, T, 4096] e4m3 (group-major), scales fp32 at g * 32 * TA + t + c * TA (TA = T rounded up to 4).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace rq {

__device__ __forceinline__ float2 unpack_bf16(unsigned v) { return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u)); }
__device__ __forceinline__ float pow2(int e) { return __uint_as_float(unsigned(e + 127) << 23); }
__device__ __forceinline__ int quant_exp(float amax, float floor) {
  const unsigned bits = __float_as_uint(fmaxf(amax, floor) * (1.0f / 448.0f));
  return int((bits >> 23) & 0xff) - 127 + ((bits & 0x7fffffu) != 0);
}
__device__ __forceinline__ unsigned e4m3x2(float lo, float hi) {
  unsigned short r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

template <int HPG>
__global__ void __launch_bounds__(HPG * 32) kernel(const __nv_bfloat16* __restrict__ o, long o_stride_t, long o_stride_h,
                                                   const int64_t* __restrict__ positions, const float* __restrict__ cache,
                                                   long cache_stride, int T, int TA, uint8_t* __restrict__ out,
                                                   float* __restrict__ scales) {
  constexpr int NOPE = 448, HALF = 32, D = 512, CH = D / 128;
  const int t = blockIdx.x, g = blockIdx.y;
  const int hg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int head = g * HPG + hg;
  const int sub = lane & 7, chunk = lane >> 3;
  float* srow = scales + long(g) * (HPG * CH) * TA + t + long(hg * CH + chunk) * TA;
  if (t >= T) {
    if (sub == 0) *srow = 0.0f;
    return;
  }
  const uint4* src = reinterpret_cast<const uint4*>(o + long(t) * o_stride_t + long(head) * o_stride_h + 16 * lane);
  const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
  float x[16];
  {
    const unsigned w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float2 f = unpack_bf16(w[i]);
      x[2 * i] = f.x;
      x[2 * i + 1] = f.y;
    }
  }
  if (16 * lane >= NOPE) {
    const long pos = positions[t];
    const int c0 = (16 * lane - NOPE) / 2;
    const float4* cs = reinterpret_cast<const float4*>(cache + pos * cache_stride + c0);
    const float4* sn = reinterpret_cast<const float4*>(cache + pos * cache_stride + HALF + c0);
    const float4 ca = __ldg(cs), cb = __ldg(cs + 1), sa = __ldg(sn), sb = __ldg(sn + 1);
    const float cv[8] = {ca.x, ca.y, ca.z, ca.w, cb.x, cb.y, cb.z, cb.w};
    const float sv[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float e = x[2 * k], od = x[2 * k + 1];
      x[2 * k] = fmaf(od, sv[k], e * cv[k]);
      x[2 * k + 1] = fmaf(-e, sv[k], od * cv[k]);
    }
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
  const int e = quant_exp(amax, 1e-10f);
  const float inv = pow2(-e);
  unsigned q[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    q[i] = e4m3x2(x[4 * i] * inv, x[4 * i + 1] * inv) | (e4m3x2(x[4 * i + 2] * inv, x[4 * i + 3] * inv) << 16);
  uint8_t* dst = out + (long(g) * T + t) * (HPG * D) + hg * D + 16 * lane;
  *reinterpret_cast<uint4*>(dst) = make_uint4(q[0], q[1], q[2], q[3]);
  if (sub == 0) *srow = pow2(e);
}

}  // namespace rq

std::vector<torch::Tensor> inv_rope_fp8_quant(torch::Tensor o, torch::Tensor positions, torch::Tensor cache, int64_t n_groups) {
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == torch::kBFloat16 && o.dim() == 3 && o.size(2) == 512 && o.stride(2) == 1,
              "o must be bf16 [T, H, 512]");
  TORCH_CHECK(o.stride(1) % 8 == 0 && o.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0, "o alignment");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 && positions.numel() >= o.size(0), "positions");
  TORCH_CHECK(cache.is_cuda() && cache.scalar_type() == torch::kFloat32 && cache.dim() == 2 && cache.size(1) == 64 &&
                  cache.stride(1) == 1 && cache.stride(0) % 4 == 0, "cos_sin_cache must be fp32 [max_pos, 64]");
  const long T = o.size(0), H = o.size(1);
  TORCH_CHECK(H % n_groups == 0 && H / n_groups == 8, "8 heads per wo_a group");
  const long G = n_groups, TA = (T + 3) / 4 * 4;
  auto out = torch::empty({G, T, 8 * 512}, o.options().dtype(torch::kFloat8_e4m3fn));
  auto sbuf = torch::empty({G * 32 * TA}, o.options().dtype(torch::kFloat32));
  auto sc = sbuf.as_strided({G, T, 32}, {32 * TA, 1, TA});
  if (T > 0) {
    const dim3 grid(static_cast<unsigned>(TA), static_cast<unsigned>(G));
    rq::kernel<8><<<grid, dim3(8 * 32), 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()), o.stride(0), o.stride(1), positions.data_ptr<int64_t>(),
        cache.data_ptr<float>(), cache.stride(0), int(T), int(TA), reinterpret_cast<uint8_t*>(out.data_ptr()), sbuf.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out.transpose(0, 1), sc.transpose(0, 1)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inv_rope_fp8_quant", &inv_rope_fp8_quant, "inverse RoPE + 1x128 FP8 quant of the attention output, G x 8 heads");
}
