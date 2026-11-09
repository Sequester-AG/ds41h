// spa_common.cuh: device helpers shared by attn_sp3.cu (copied from attn_sp.cu v2, which keeps its own copy):
// SMEM / cp.async / wgmma / barrier wrappers, SW128 descriptors, exact e4m3 x 2^(E-127) -> bf16 dequant, record address.
// Needs spa::Params (q, swa, comp, stride, ...) declared before inclusion.
#pragma once
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void sts128(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ float4 lds128f(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void sts128f(uint32_t a, float x, float y, float z, float w) {
  asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" ::"r"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ unsigned lds_u8(uint32_t a) {
  unsigned v;
  asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void cp16(uint32_t s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void cp8(uint32_t s, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
// K-major SW128, 128-B rows, 8-row groups 1 KB apart
__device__ __forceinline__ uint64_t desc_k(uint32_t a) {
  return ((uint64_t(a) & 0x3FFFFull) >> 4) | (uint64_t(1) << 16) | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
}
// MN-major SW128: lbo = bytes between 64-element atoms along N, sbo = bytes between 8-row groups along K
__device__ __forceinline__ uint64_t desc_mn(uint32_t a) {
  return ((uint64_t(a) & 0x3FFFFull) >> 4) | (uint64_t((MN_LBO >> 4) & 0x3FFF) << 16) |
         (uint64_t((MN_SBO >> 4) & 0x3FFF) << 32) | (uint64_t(1) << 62);
}
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned b, unsigned sel) {
  unsigned d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
__device__ __forceinline__ unsigned bf16x2_mul(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&v);
}
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// e4m3 bits s eeee mmm placed as bf16 bits are the value x 2^-120 exactly (subnormals included); x 2^(E-7) as a bf16
// gives value x 2^(E-127) exactly for E in [16, 134] (w8a16.cu's integer dequant). 4 bytes -> 2 bf16x2 words.
__device__ __forceinline__ void e4m3x4_to_w(unsigned v, unsigned m2, unsigned& lo, unsigned& hi) {
  const unsigned mg = v & 0x7f7f7f7fu, sg = v & 0x80808080u;
  lo = bf16x2_mul(prmt(mg, 0u, 0x4140u) * 16u + prmt(sg, 0u, 0x1404u), m2);
  hi = bf16x2_mul(prmt(mg, 0u, 0x4342u) * 16u + prmt(sg, 0u, 0x3424u), m2);
}
__device__ __forceinline__ float e4m3_float(unsigned b) {
  const unsigned ex = (b >> 3) & 15u, m = b & 7u;
  const float v = ex ? __uint_as_float(((ex + 120u) << 23) | (m << 20)) : static_cast<float>(m) * 0.001953125f;
  return (b & 0x80u) ? -v : v;
}
// 16 e4m3 x 2^(E-127) -> 16 bf16 (lo = dims 0..7, hi = 8..15), exact
__device__ __forceinline__ void dequant16(uint4 raw, unsigned E, uint4& lo, uint4& hi) {
  if (E >= 16u && E <= 134u) {
    const unsigned b = (E + 120u) << 7, m2 = b | (b << 16);
    e4m3x4_to_w(raw.x, m2, lo.x, lo.y);
    e4m3x4_to_w(raw.y, m2, lo.z, lo.w);
    e4m3x4_to_w(raw.z, m2, hi.x, hi.y);
    e4m3x4_to_w(raw.w, m2, hi.z, hi.w);
  } else {  // out of the fast range (never on real records): fp32 path, one bf16 rounding
    const float s = exp2f(static_cast<float>(static_cast<int>(E) - 127));
    const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
    unsigned o[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      o[2 * j] = pack_bf16(e4m3_float(w[j] & 0xffu) * s, e4m3_float((w[j] >> 8) & 0xffu) * s);
      o[2 * j + 1] = pack_bf16(e4m3_float((w[j] >> 16) & 0xffu) * s, e4m3_float(w[j] >> 24) * s);
    }
    lo = make_uint4(o[0], o[1], o[2], o[3]);
    hi = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

__device__ __forceinline__ void rec_addr(const Params& p, int idx, bool win, const uint8_t*& d, const uint8_t*& s) {
  idx = max(idx, 0);
  const int sh = win ? 5 : 6;
  const uint8_t* pb = (win ? p.swa : p.comp) + static_cast<long long>(idx >> sh) * p.stride;
  const int off = idx & ((1 << sh) - 1);
  d = pb + off * REC;
  s = pb + (REC << sh) + off * 8;
}

