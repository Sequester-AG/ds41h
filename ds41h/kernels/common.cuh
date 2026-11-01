// Common device helpers for ds41h sm90 kernels.
// Portions adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
// src/brrr/kernels/common.cuh). See LICENSES/BRRR-Apache-2.0.txt.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

constexpr int RANKS = 4;             // TP size (4x H200 NVL, NVLink all-to-all)
constexpr float AMAX_FLOOR = 1e-4f;  // reference act_quant amax floor (inference/kernel.py)

__host__ __device__ __forceinline__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }

// ---------------------------------------------------------------- async copies / barriers
__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async16_hint(void* dst, const void* src, unsigned long long policy) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(smem_u32(dst)), "l"(src), "l"(policy) : "memory");
}
__device__ __forceinline__ unsigned long long evict_first_policy() {
  unsigned long long policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ void mbar_init(unsigned long long* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned long long* bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n LAB_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra LAB_WAIT;\n}" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive(unsigned long long* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}

// ---------------------------------------------------------------- memory ordering (gpu / sys scope)
__device__ __forceinline__ unsigned ld_volatile(const unsigned* p) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release(unsigned* p, unsigned v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v) { asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_relaxed_sys(unsigned* p, unsigned v) { asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned atom_add_acq_rel(unsigned* p, unsigned v) {
  unsigned old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ bool before(unsigned a, unsigned b) { return int(a - b) < 0; }

// Programmatic dependent launch (sm90+)
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// Streaming (no L1 allocate) loads for weights read once per step
__device__ __forceinline__ uint4 ld_stream16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned ld_stream4(const void* p) {
  unsigned v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

// ---------------------------------------------------------------- warp reductions
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// ---------------------------------------------------------------- bf16 / fp8 packing
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&v);
}
__device__ __forceinline__ float2 unpack_bf16(unsigned v) { return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u)); }
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ uint4 pack8_bf16(const float (&f)[8]) {
  return make_uint4(pack_bf16(f[0], f[1]), pack_bf16(f[2], f[3]), pack_bf16(f[4], f[5]), pack_bf16(f[6], f[7]));
}
__device__ __forceinline__ void add8_bf16(float (&acc)[8], uint4 v) {
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = unpack_bf16(u[i]);
    acc[2 * i] += f.x;
    acc[2 * i + 1] += f.y;
  }
}
__device__ __forceinline__ unsigned pack_f16(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&v);
}

__device__ __forceinline__ float pow2(int e) { return __uint_as_float(unsigned(e + 127) << 23); }  // valid for e in [-126, 127]
__device__ __forceinline__ float e8m0_to_float(unsigned e) { return __uint_as_float(e << 23); }    // ue8m0 byte -> 2^(e-127)

// Reference activation quant (inference/kernel.py act_quant): per-32 ue8m0 scale = 2^ceil(log2(amax/448)).
__device__ __forceinline__ int quant_exp(float amax, float floor) {
  const unsigned bits = __float_as_uint(fmaxf(amax, floor) * (1.0f / 448.0f));
  return int((bits >> 23) & 0xff) - 127 + ((bits & 0x7fffffu) != 0);
}
__device__ __forceinline__ unsigned e4m3x2(float lo, float hi) {
  unsigned short r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint8_t quant_byte(float x, int e) { return uint8_t(e4m3x2(x * pow2(-e), 0.0f) & 0xffu); }
__device__ __forceinline__ unsigned float_order(float v) {
  const unsigned bits = __float_as_uint(v);
  return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}
