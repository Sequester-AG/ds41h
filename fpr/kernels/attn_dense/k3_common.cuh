// k3_common.cuh: helpers shared by the K3 attention kernels (copied from attn_seq.cu v2).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace k3c {


constexpr int D = 512;
constexpr int BK = 64;  // keys per tile
constexpr int NT = 256;
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void sts128(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void sts32(uint32_t a, unsigned v) { asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void stsf(uint32_t a, float v) { asm volatile("st.shared.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory"); }
__device__ __forceinline__ float ldsf(uint32_t a) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ldg_stream16(const void* p) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned ldg_u8(const uint8_t* p) {
  unsigned short v;
  asm("ld.global.nc.u8 %0, [%1];" : "=h"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
// K-major, 128-B rows, 128B swizzle, 8-row groups 1 KB apart (LBO unused).
__device__ __forceinline__ uint64_t desc_k(uint32_t a) {
  return ((uint64_t(a) & 0x3FFFFull) >> 4) | (uint64_t(1) << 16) | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
}
// MN-major 128B swizzle: lbo = bytes between 64-element atoms along N, sbo = bytes between 8-row groups along K.
__device__ __forceinline__ uint64_t desc_mn(uint32_t a, int lbo, int sbo) {
  return ((uint64_t(a) & 0x3FFFFull) >> 4) | (uint64_t((lbo >> 4) & 0x3FFF) << 16) | (uint64_t((sbo >> 4) & 0x3FFF) << 32) |
         (uint64_t(1) << 62);
}
__device__ __forceinline__ unsigned bf16x2_mul(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ unsigned e4m3x2_to_f16x2(unsigned short v) {
  unsigned h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(v));
  return h;
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&v);
}
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
// 16 e4m3 (dims 16c..16c+15) x 2^(e-127) -> 16 bf16, exact (sparse_decode.cu dequant16, fast path e <= 142).
__device__ __forceinline__ void dequant16(uint4 raw, unsigned e, uint4& lo, uint4& hi) {
  const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
  unsigned o[8];
  if (e <= 142u) {
    const unsigned b = (e + 112u) << 7;
    const unsigned s2 = b | (b << 16);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned h0 = e4m3x2_to_f16x2(static_cast<unsigned short>(w[j] & 0xffffu));
      const unsigned h1 = e4m3x2_to_f16x2(static_cast<unsigned short>(w[j] >> 16));
      o[2 * j] = bf16x2_mul(((h0 & 0x7fff7fffu) >> 3) | (h0 & 0x80008000u), s2);
      o[2 * j + 1] = bf16x2_mul(((h1 & 0x7fff7fffu) >> 3) | (h1 & 0x80008000u), s2);
    }
  } else {
    const float s = __uint_as_float(e << 23);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const unsigned h = e4m3x2_to_f16x2(static_cast<unsigned short>(hh ? w[j] >> 16 : w[j] & 0xffffu));
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
        o[2 * j + hh] = pack_bf16(f.x * s, f.y * s);
      }
    }
  }
  lo = make_uint4(o[0], o[1], o[2], o[3]);
  hi = make_uint4(o[4], o[5], o[6], o[7]);
}


}  // namespace k3c
