// attn_seq.cu: K3 -- sparse MQA decode attention of one consumer layer, one CTA per sequence.
//
// DeepSeek-V4.1-Flash, TP4 rank (16 local heads), MQA K = V = 512-dim latent in stock vLLM 0.32.0 sm90 "fp8_ds_mla"
// records (584 B per token: 448 e4m3 NoPE + 64 bf16 RoPE, 7 ue8m0 exponents + 1 pad in the page's scale area).
// A decode step has 4 rows per sequence (decode + 3 draft tokens). Each row attends to its 128-position window
// (SWA cache) and its top-512 compressed rows (compressed cache) plus the attention sink.
//
// Design:
//  * The 4 rows x 16 heads of one sequence are the M = 64 rows of SS-wgmma; every key of the sequence's key UNION
//    (window union <= 131 + top-k union) is read from HBM and dequantized once, then used by all 64 queries with a
//    per-(row, key) membership mask. Measured on decode steps: the union of the 4 rows' top-512 is ~1,060-1,280
//    keys, so ~half of the QK / PV MACs are masked.
//  * The key list (+ 4-bit membership) is built ONCE per step / index group by meta_kernel (bitonic sort + unique of
//    the 4 x 512 local indices, block-table lookup), not per layer.
//  * 256 threads: warpgroup 0 = S = Q K^T (32 x m64n64k16, Q and K in SMEM) + online softmax (P -> SMEM bf16) + P V for
//    dims 0..255; warpgroup 1 = gather + dequant of a later 64-key tile (LDG -> registers -> swizzled bf16 tile) +
//    P V for dims 256..511 (4 x m64n256k16, B = the same K tile read MN-major). Double-buffered K and P tiles.
//  * v2 pipeline: in iteration i both warpgroups issue P_i V_i and warpgroup 0 also S_{i+1} = Q K_{i+1}^T in the same
//    wgmma batch; then warpgroup 0 runs softmax_{i+1} while warpgroup 1 gathers tile i+2 into the buffer of tile i.
//    The metadata hands the kernel absolute record addresses (padded with a zero record), so the gather is
//    branch-free (v1 spent ~80 instructions of address math per 16-byte item: 20K cycles per tile).
//  * Numerics as stock FlashMLA / ds41h sparse_decode.cu: exact fp8 -> bf16 key dequant, bf16 operands, fp32
//    accumulation, scores x (scale * log2 e), exp2, P rounded to bf16 for PV, l summed in fp32 from the unrounded
//    exponentials, sink in the denominator, a row with no keys outputs 0. Only the summation order differs.
//  * Optional fused inverse RoPE (K4's first op): o_rot = bf16(rope^-1(bf16(o))) with stock's fp32 cos/sin cache,
//    non-contracted fp32 math (same roundings as the reference / torch).
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

#include "wgmma_bf16_gen.cuh"

namespace k3 {

constexpr int D = 512;
constexpr int BK = 64;  // keys per tile
constexpr int NT = 256;
constexpr float LOG2E = 1.4426950408889634f;
constexpr int OFF_Q = 0, OFF_K0 = 65536, OFF_K1 = 131072, OFF_P0 = 196608, OFF_P1 = 204800, OFF_ALPHA = 212992, OFF_SO = 213504;  // alpha: 2 x 64 f32
constexpr int SMEM_BYTES = 213760 + 1024;

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

struct Params {
  const __nv_bfloat16* q;  // [T, 16, 512] post-RoPE; the 64 query rows of CTA s are q[64 s .. 64 s + 63] (4 rows x 16 heads)
  __nv_bfloat16* out;      // same layout
  const int* codes;        // [nseq, code_stride]: bits 0..3 = membership of the CTA's 4 rows (0 for padding)
  int code_stride;
  const ulonglong2* addr;  // [nseq, addr_stride]: {data row address, scale row address}; padding = a zero record
  int addr_stride;
  const int* nkeys;        // [nseq]
  const float* sink;       // [sink_rows]: query row m uses sink[m % sink_rows] (16: 4 rows x 16 heads; 64: 1 row x 64 heads)
  int sink_rows;
  float scale_log2;
  const long long* positions;  // [T] (inverse RoPE only)
  const float* cos_sin;        // [P, 64] fp32 (cos 0..31 | sin 32..63) or nullptr: no inverse RoPE
  int mn_lbo, mn_sbo;
  float* dbg_s;                // optional [nseq, 64, 64] raw scores of tile 0
  unsigned long long* prof;    // K3_PROF builds: [nseq][40 tiles][2 warpgroups][8] clock64 stamps
  int abl;                     // ablation (timing only, results invalid): 1 no gather, 2 no QK, 3 no PV, 4 no softmax
};
#ifdef K3_PROF
#define KP(it_, slot)                                                                                     \
  do {                                                                                                    \
    if (ltid == 0 && p.prof != nullptr && (it_) >= 0 && (it_) < 40)                                       \
      p.prof[((static_cast<long long>(s) * 40 + (it_)) * 2 + wg) * 8 + (slot)] = (unsigned long long)clock64(); \
  } while (0)
#else
#define KP(it_, slot) \
  do {                \
  } while (0)
#endif

__device__ __forceinline__ ulonglong2 ldg_u64x2(const ulonglong2* p) {
  ulonglong2 v;
  asm("ld.global.nc.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p));
  return v;
}

// --------------------------------------------------------------------------------------------- tile gather
// TPK = NTH / 64 threads per key: thread ltid gathers key kk = ltid / TPK of the tile, 16-byte chunks
// c = ltid % TPK + TPK * j of its 576-byte data row (c < 28: 16 e4m3 NoPE dims with exponent c / 4; c >= 28: 8 bf16
// RoPE dims) and the key's 8-byte scale row once. Branch-free: padding keys point at a zero record.
template <int NTH>
struct Gather {
  static constexpr int TPK = NTH / BK;
  static constexpr int NI = 36 / TPK;
  static_assert(TPK * NI == 36, "chunks per thread");
  uint4 raw[NI];
  uint2 scl;
  __device__ __forceinline__ void issue(const ulonglong2* addr, int tile, int ltid) {
    const ulonglong2 a = ldg_u64x2(addr + tile * BK + ltid / TPK);
    const uint8_t* d = reinterpret_cast<const uint8_t*>(a.x) + 16 * (ltid % TPK);
#pragma unroll
    for (int j = 0; j < NI; ++j) raw[j] = ldg_stream16(d + 16 * TPK * j);
    asm("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(scl.x), "=r"(scl.y) : "l"(a.y));
  }
  __device__ __forceinline__ void store(uint32_t kbase, int ltid) const {
    const int kk = ltid / TPK, c0 = ltid % TPK;
    const uint32_t row = kbase + kk * 128;
#pragma unroll
    for (int j = 0; j < NI; ++j) {
      const int c = c0 + TPK * j;
      if (c < 28) {
        const int sidx = c >> 2;
        const unsigned e = ((sidx < 4 ? scl.x : scl.y) >> (8 * (sidx & 3))) & 0xffu;
        uint4 lo, hi;
        dequant16(raw[j], e, lo, hi);
        const uint32_t a = row + sidx * 8192;
        const int ch = (c & 3) * 2;
        sts128(a + (((ch) ^ (kk & 7)) << 4), lo);
        sts128(a + (((ch + 1) ^ (kk & 7)) << 4), hi);
      } else {
        sts128(row + 7 * 8192 + (((c - 28) ^ (kk & 7)) << 4), raw[j]);
      }
    }
  }
};

// S = Q K^T for one 64-key tile (warpgroup 0), issued without waiting.
__device__ __forceinline__ void issue_qk(float (&sacc)[32], uint32_t qb, uint32_t kb) {
#pragma unroll
  for (int ks = 0; ks < 32; ++ks) {
    const uint32_t off = (ks >> 2) * 8192 + (ks & 3) * 32;
    WgSS<64, 0>::mma(sacc, desc_k(qb + off), desc_k(kb + off), ks > 0);
  }
}

// Online softmax of one tile for this thread's rows (rho 0: row mrow0, 1: mrow0 + 8); P (bf16) -> SMEM, alpha -> SMEM.
__device__ __forceinline__ void softmax_tile(float (&sacc)[32], unsigned long long mask, float (&m_run)[2], float (&l_run)[2],
                                             float sl2, uint32_t pbuf, uint32_t alpha_s, int mrow0, int t4) {
#pragma unroll
  for (int rho = 0; rho < 2; ++rho) {
    float x[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int c = 8 * j + 2 * t4 + e;
        x[2 * j + e] = ((mask >> c) & 1ull) ? sacc[4 * j + 2 * rho + e] * sl2 : -INFINITY;
      }
    }
    float t[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) t[i] = fmaxf(x[2 * i], x[2 * i + 1]);
#pragma unroll
    for (int i = 0; i < 4; ++i) t[i] = fmaxf(t[2 * i], t[2 * i + 1]);
    float tmax = fmaxf(fmaxf(t[0], t[1]), fmaxf(t[2], t[3]));
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
    const float mnew = fmaxf(m_run[rho], tmax);
    const float alpha = mnew == -INFINITY ? 1.f : exp2f(m_run[rho] - mnew);
    float ev[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) ev[i] = x[i] == -INFINITY ? 0.f : exp2f(x[i] - mnew);
    const int mr = mrow0 + 8 * rho;
#pragma unroll
    for (int j = 0; j < 8; ++j) sts32(pbuf + mr * 128 + ((j ^ (mr & 7)) << 4) + 4 * t4, pack_bf16(ev[2 * j], ev[2 * j + 1]));
#pragma unroll
    for (int i = 0; i < 8; ++i) t[i] = ev[2 * i] + ev[2 * i + 1];
#pragma unroll
    for (int i = 0; i < 4; ++i) t[i] = t[2 * i] + t[2 * i + 1];
    const float lsum = (t[0] + t[1]) + (t[2] + t[3]);
    l_run[rho] = l_run[rho] * alpha + lsum;
    m_run[rho] = mnew;
    if (t4 == 0) stsf(alpha_s + 4 * mr, alpha);
  }
}

__device__ __forceinline__ unsigned long long tile_mask(const int* codes, int tile, int lane, int row) {
  const int c0 = __ldg(codes + tile * BK + lane), c1 = __ldg(codes + tile * BK + 32 + lane);
  const unsigned lo = __ballot_sync(0xffffffffu, (c0 >> row) & 1), hi = __ballot_sync(0xffffffffu, (c1 >> row) & 1);
  return (static_cast<unsigned long long>(hi) << 32) | lo;
}

// --------------------------------------------------------------------------------------------- attention kernel
__global__ void __launch_bounds__(NT, 1) attn_seq_kernel(const __grid_constant__ Params p) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t sraw = smem_u32(smem_raw);
  const uint32_t sb = (sraw + 1023u) & ~1023u;
  const int s = blockIdx.x, tid = threadIdx.x, wg = tid >> 7, ltid = tid & 127, warp = (tid >> 5) & 3, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int nk = __ldg(p.nkeys + s), ntiles = (nk + BK - 1) / BK;
  const int* codes = p.codes + static_cast<long long>(s) * p.code_stride;
  const ulonglong2* addr = p.addr + static_cast<long long>(s) * p.addr_stride;
  const int mrow0 = 16 * warp + g;  // this thread's query rows mrow0, mrow0 + 8 (CTA row = warp, heads g, g + 8)
  const uint32_t KB[2] = {sb + OFF_K0, sb + OFF_K1}, PB[2] = {sb + OFF_P0, sb + OFF_P1};

  // Q (64 rows x 512) -> SMEM, 8 column blocks of 64 dims, SW128
  const __nv_bfloat16* qs = p.q + static_cast<long long>(s) * 64 * D;
  for (int i = tid; i < 64 * 64; i += NT) {
    const int m = i >> 6, ch = i & 63, cb = ch >> 3, c8 = ch & 7;
    cp_async16(sb + OFF_Q + cb * 8192 + m * 128 + ((c8 ^ (m & 7)) << 4), qs + m * D + ch * 8);
  }
  cp_async_commit();
  if (ntiles > 0) {
    Gather<NT> g0;
    g0.issue(addr, 0, tid);
    g0.store(KB[0], tid);
  }
  cp_async_wait_all();
  fence_proxy_async();
  __syncthreads();

  float o[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) o[i] = 0.f;
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
  float sacc[32];

  // prologue: S_0 + softmax_0 (warpgroup 0) | tile 1 (warpgroup 1)
  if (ntiles > 0) {
    if (wg == 0) {
      wg_fence();
      if (p.abl != 2) issue_qk(sacc, sb + OFF_Q, KB[0]);
      wg_commit();
      const unsigned long long mask = tile_mask(codes, 0, lane, warp);
      wg_wait<0>();
      if (p.dbg_s != nullptr) {
        float* d = p.dbg_s + static_cast<long long>(s) * 64 * 64;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int c = 8 * j + 2 * t4;
          d[mrow0 * 64 + c] = sacc[4 * j];
          d[mrow0 * 64 + c + 1] = sacc[4 * j + 1];
          d[(mrow0 + 8) * 64 + c] = sacc[4 * j + 2];
          d[(mrow0 + 8) * 64 + c + 1] = sacc[4 * j + 3];
        }
      }
      if (p.abl != 4) softmax_tile(sacc, mask, m_run, l_run, p.scale_log2, PB[0], sb + OFF_ALPHA, mrow0, t4);
      fence_proxy_async();
    } else if (ntiles > 1 && p.abl != 1) {
      Gather<128> gn;
      gn.issue(addr, 1, ltid);
      gn.store(KB[1], ltid);
      fence_proxy_async();
    }
    named_bar(1, NT);
  }

  for (int it = 0; it < ntiles; ++it) {
    const int cur = it & 1;
    KP(it, 0);
    const uint32_t al = sb + OFF_ALPHA + 256 * cur;
    const float a0 = ldsf(al + 4 * mrow0), a1 = ldsf(al + 4 * (mrow0 + 8));
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      o[4 * j] *= a0;
      o[4 * j + 1] *= a0;
      o[4 * j + 2] *= a1;
      o[4 * j + 3] *= a1;
    }
    const bool more = it + 1 < ntiles;
    wg_fence();
    if (p.abl != 3) {
#pragma unroll
      for (int ks = 0; ks < 4; ++ks)
        WgSS<256, 1>::mma(o, desc_k(PB[cur] + ks * 32), desc_mn(KB[cur] + 4 * wg * 8192 + ks * 2048, p.mn_lbo, p.mn_sbo), 1);
    }
    if (wg == 0 && more && p.abl != 2) issue_qk(sacc, sb + OFF_Q, KB[cur ^ 1]);
    wg_commit();
    unsigned long long mask = 0ull;
    if (wg == 0 && more) mask = tile_mask(codes, it + 1, lane, warp);
    wg_wait<0>();
    KP(it, 1);
    if (wg == 0) {
      asm volatile("bar.arrive 2, %0;" ::"r"(NT) : "memory");  // P_it / K_it no longer read by warpgroup 0
      if (more) {
        if (p.abl != 4) softmax_tile(sacc, mask, m_run, l_run, p.scale_log2, PB[cur ^ 1], sb + OFF_ALPHA + 256 * (cur ^ 1), mrow0, t4);
        fence_proxy_async();
      }
      KP(it, 2);
    } else {
      named_bar(2, NT);  // both warpgroups' P_it V_it done: buffer K[cur] is free
      if (it + 2 < ntiles && p.abl != 1) {
        Gather<128> gn;
        gn.issue(addr, it + 2, ltid);
        gn.store(KB[cur], ltid);
        fence_proxy_async();
      }
      KP(it, 2);
    }
    named_bar(1, NT);  // P_{it+1}, alpha_{it+1} and tile it+2 are in SMEM
    KP(it, 3);
  }

  // ---- final normalization: o / (l + e^(sink - m)) in log2 units, zero for sink = +inf / no keys with sink = -inf
  if (wg == 0) {
#pragma unroll
    for (int rho = 0; rho < 2; ++rho) {
      float l = l_run[rho];
      l += __shfl_xor_sync(0xffffffffu, l, 1);
      l += __shfl_xor_sync(0xffffffffu, l, 2);
      const float m = m_run[rho];
      const float sink2 = __ldg(p.sink + (mrow0 + 8 * rho) % p.sink_rows) * LOG2E;
      const float mf = fmaxf(m, sink2);
      const float f = m == -INFINITY ? 0.f : exp2f(m - mf);
      const float den = l * f + (sink2 == -INFINITY ? 0.f : exp2f(sink2 - mf));
      const float so = (den > 0.f && mf < INFINITY) ? f / den : 0.f;
      if (t4 == 0) stsf(sb + OFF_SO + 4 * (mrow0 + 8 * rho), so);
    }
  }
  named_bar(3, NT);
  {
    const float so0 = ldsf(sb + OFF_SO + 4 * mrow0), so1 = ldsf(sb + OFF_SO + 4 * (mrow0 + 8));
    const long long trow = 4LL * s + warp;
    __nv_bfloat16* out0 = p.out + (trow * 16 + g) * D + 256 * wg;
    __nv_bfloat16* out1 = out0 + 8 * D;
    const float* cs = nullptr;
    if (p.cos_sin != nullptr && wg == 1) cs = p.cos_sin + __ldg(p.positions + trow) * 64;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int d = 8 * j + 2 * t4;
      float v0 = bf16r(o[4 * j] * so0), v1 = bf16r(o[4 * j + 1] * so0);
      float v2 = bf16r(o[4 * j + 2] * so1), v3 = bf16r(o[4 * j + 3] * so1);
      if (cs != nullptr && j >= 24) {  // dims 448..511: inverse RoPE on the bf16 output (pair i)
        const int i = 4 * (j - 24) + t4;
        const float c = __ldg(cs + i), sn = __ldg(cs + 32 + i);
        const float a0 = __fadd_rn(__fmul_rn(v0, c), __fmul_rn(v1, sn)), b0 = __fsub_rn(__fmul_rn(v1, c), __fmul_rn(v0, sn));
        const float a1 = __fadd_rn(__fmul_rn(v2, c), __fmul_rn(v3, sn)), b1 = __fsub_rn(__fmul_rn(v3, c), __fmul_rn(v2, sn));
        v0 = a0, v1 = b0, v2 = a1, v3 = b1;
      }
      *reinterpret_cast<unsigned*>(out0 + d) = pack_bf16(v0, v1);
      *reinterpret_cast<unsigned*>(out1 + d) = pack_bf16(v2, v3);
    }
  }
}

// --------------------------------------------------------------------------------------------- per-step metadata
// One CTA per sequence: the key union of its 4 rows. Window part first (positions [lo, hi] of the rows' 128-position
// windows, membership per row), then the compressed top-k union in ascending index order (bitonic sort of
// (index << 2 | row), unique, OR of the rows). Per key: codes = (slot << 5) | (compressed << 4) | mask, addr = absolute
// {data row, scale row} addresses; the list is padded to a multiple of 64 with code 0 / the zero record.
struct MetaParams {
  const int* topk;  // [T, topk_stride] local compressed indices (-1 = empty)
  long long topk_stride;
  int topk_w;                   // <= 512
  const long long* positions;  // [T]
  const uint8_t* valid;         // [T] or nullptr (all valid)
  const int* swa_bt;
  long long swa_bt_stride;
  int swa_shift;
  const int* comp_bt;
  long long comp_bt_stride;
  int comp_shift;
  int window;
  int* codes;
  int code_stride;
  int* nkeys;
  unsigned long long cache0, cache1, zero;  // absolute addresses
  long long stride0, stride1;
  ulonglong2* addr;
  int addr_stride;
};

__device__ __forceinline__ ulonglong2 rec_addr(unsigned long long base, long long stride, int shift, unsigned slot) {
  const unsigned long long pg = slot >> shift, ps = slot & ((1u << shift) - 1u);
  const unsigned long long b = base + pg * static_cast<unsigned long long>(stride);
  return make_ulonglong2(b + ps * 576ull, b + (576ull << shift) + ps * 8ull);
}

__global__ void __launch_bounds__(512) meta_kernel(const __grid_constant__ MetaParams p) {
  __shared__ uint32_t key[2048];
  __shared__ int wsum[16];
  const int s = blockIdx.x, tid = threadIdx.x;
  long long pr[4];
  bool vr[4];
  long long lo = 0x7fffffffffffffffLL, hi = -1;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    pr[r] = __ldg(p.positions + 4 * s + r);
    vr[r] = p.valid == nullptr || __ldg(p.valid + 4 * s + r) != 0;
    if (vr[r]) {
      lo = min(lo, max(pr[r] - (p.window - 1), 0LL));
      hi = max(hi, pr[r]);
    }
  }
  const int nwin = hi >= lo ? static_cast<int>(hi - lo + 1) : 0;
  int* out = p.codes + static_cast<long long>(s) * p.code_stride;
  ulonglong2* aout = p.addr + static_cast<long long>(s) * p.addr_stride;
  for (int j = tid; j < nwin; j += 512) {
    const long long ps = lo + j;
    int m = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r)
      if (vr[r] && ps <= pr[r] && ps >= pr[r] - (p.window - 1)) m |= 1 << r;
    const int pg = __ldg(p.swa_bt + s * p.swa_bt_stride + (ps >> p.swa_shift));
    const unsigned slot = (static_cast<unsigned>(pg) << p.swa_shift) | static_cast<unsigned>(ps & ((1 << p.swa_shift) - 1));
    out[j] = static_cast<int>((slot << 5) | m);
    aout[j] = rec_addr(p.cache0, p.stride0, p.swa_shift, slot);
  }
  // compressed union
  for (int i = tid; i < 2048; i += 512) {
    const int r = i >> 9, j = i & 511;
    uint32_t k = 0xffffffffu;
    if (j < p.topk_w && vr[r]) {
      const int v = __ldg(p.topk + (4LL * s + r) * p.topk_stride + j);
      if (v >= 0) k = (static_cast<uint32_t>(v) << 2) | r;
    }
    key[i] = k;
  }
  __syncthreads();
  for (int k = 2; k <= 2048; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < 2048; i += 512) {
        const int ixj = i ^ j;
        if (ixj > i) {
          const uint32_t a = key[i], b = key[ixj];
          const bool asc = (i & k) == 0;
          if ((a > b) == asc) {
            key[i] = b;
            key[ixj] = a;
          }
        }
      }
      __syncthreads();
    }
  }
  // thread tid owns entries [4 tid, 4 tid + 4)
  int head[4], msk[4], cnt = 0;
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    const int i = 4 * tid + d;
    const uint32_t kv = key[i];
    head[d] = kv != 0xffffffffu && (i == 0 || (key[i - 1] >> 2) != (kv >> 2));
    int m = 0;
    if (head[d]) {
      for (int e = 0; e < 8 && i + e < 2048; ++e) {
        const uint32_t k2 = key[i + e];
        if (k2 == 0xffffffffu || (k2 >> 2) != (kv >> 2)) break;
        m |= 1 << (k2 & 3);
      }
    }
    msk[d] = m;
    cnt += head[d];
  }
  const int lane = tid & 31, w = tid >> 5;
  int incl = cnt;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) wsum[w] = incl;
  __syncthreads();
  int wbase = 0, total = 0;
  for (int i = 0; i < 16; ++i) {
    const int v = wsum[i];
    if (i < w) wbase += v;
    total += v;
  }
  int pos = nwin + wbase + incl - cnt;
  const int cap = min(p.code_stride, p.addr_stride);
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    if (!head[d]) continue;
    const uint32_t idx = key[4 * tid + d] >> 2;
    const int pg = __ldg(p.comp_bt + s * p.comp_bt_stride + (idx >> p.comp_shift));
    const unsigned slot = (static_cast<unsigned>(pg) << p.comp_shift) | (idx & ((1u << p.comp_shift) - 1u));
    if (pos < cap) {
      out[pos] = static_cast<int>((slot << 5) | 16u | static_cast<unsigned>(msk[d]));
      aout[pos] = rec_addr(p.cache1, p.stride1, p.comp_shift, slot);
    }
    ++pos;
  }
  const int n = min(nwin + total, cap), npad = min((n + BK - 1) / BK * BK, cap);
  for (int i = n + tid; i < npad; i += 512) {
    out[i] = 0;
    aout[i] = make_ulonglong2(p.zero, p.zero);
  }
  if (tid == 0) p.nkeys[s] = n;
  if (tid == 0 && nwin + total > cap) printf("attn_seq meta: seq %d needs %d keys > capacity %d\n", s, nwin + total, cap);
}

}  // namespace k3

// --------------------------------------------------------------------------------------------- host bindings
static void check_cuda(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }

void meta(torch::Tensor topk, torch::Tensor positions, c10::optional<torch::Tensor> valid, torch::Tensor swa_bt, int64_t swa_shift,
          torch::Tensor comp_bt, int64_t comp_shift, int64_t window, torch::Tensor codes, torch::Tensor nkeys, torch::Tensor cache0,
          int64_t stride0, torch::Tensor cache1, int64_t stride1, torch::Tensor zero, torch::Tensor addr) {
  const int nseq = static_cast<int>(nkeys.size(0));
  TORCH_CHECK(topk.dtype() == torch::kInt32 && topk.size(1) <= 512 && positions.dtype() == torch::kInt64, "meta: dtypes");
  TORCH_CHECK(topk.size(0) == 4 * nseq && positions.size(0) == 4 * nseq, "meta: T must be 4 x nseq");
  TORCH_CHECK(addr.dtype() == torch::kInt64 && addr.dim() == 3 && addr.size(2) == 2 && addr.size(1) % 64 == 0, "meta: addr [nseq, K, 2] int64");
  TORCH_CHECK(zero.numel() >= 1024, "meta: zero record >= 1 KB");
  k3::MetaParams p{};
  p.topk = topk.data_ptr<int>();
  p.topk_stride = topk.stride(0);
  p.topk_w = static_cast<int>(topk.size(1));
  p.positions = reinterpret_cast<const long long*>(positions.data_ptr<int64_t>());
  p.valid = valid.has_value() ? valid->data_ptr<uint8_t>() : nullptr;
  p.swa_bt = swa_bt.data_ptr<int>();
  p.swa_bt_stride = swa_bt.stride(0);
  p.swa_shift = static_cast<int>(swa_shift);
  p.comp_bt = comp_bt.data_ptr<int>();
  p.comp_bt_stride = comp_bt.stride(0);
  p.comp_shift = static_cast<int>(comp_shift);
  p.window = static_cast<int>(window);
  p.codes = codes.data_ptr<int>();
  p.code_stride = static_cast<int>(codes.stride(0));
  p.nkeys = nkeys.data_ptr<int>();
  p.cache0 = reinterpret_cast<unsigned long long>(cache0.data_ptr());
  p.cache1 = reinterpret_cast<unsigned long long>(cache1.data_ptr());
  p.zero = reinterpret_cast<unsigned long long>(zero.data_ptr());
  p.stride0 = stride0;
  p.stride1 = stride1;
  p.addr = reinterpret_cast<ulonglong2*>(addr.data_ptr<int64_t>());
  p.addr_stride = static_cast<int>(addr.size(1));
  k3::meta_kernel<<<nseq, 512, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  check_cuda(cudaGetLastError(), "meta launch");
}

void attn(torch::Tensor q, torch::Tensor out, torch::Tensor codes, torch::Tensor addr, torch::Tensor nkeys, torch::Tensor sink,
          double scale, c10::optional<torch::Tensor> positions, c10::optional<torch::Tensor> cos_sin, int64_t mn_lbo, int64_t mn_sbo,
          c10::optional<torch::Tensor> dbg_s, c10::optional<torch::Tensor> prof, int64_t abl) {
  const int nseq = static_cast<int>(nkeys.size(0));
  TORCH_CHECK(q.dtype() == torch::kBFloat16 && q.is_contiguous() && q.size(0) == 4 * nseq && q.size(1) == 16 && q.size(2) == 512, "attn: q");
  TORCH_CHECK(out.is_contiguous() && out.sizes() == q.sizes(), "attn: out");
  TORCH_CHECK(addr.dtype() == torch::kInt64 && addr.dim() == 3 && addr.size(0) == nseq && addr.size(2) == 2, "attn: addr");
  k3::Params p{};
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.codes = codes.data_ptr<int>();
  p.code_stride = static_cast<int>(codes.stride(0));
  p.addr = reinterpret_cast<const ulonglong2*>(addr.data_ptr<int64_t>());
  p.addr_stride = static_cast<int>(addr.size(1));
  p.nkeys = nkeys.data_ptr<int>();
  p.sink = sink.data_ptr<float>();
  p.sink_rows = static_cast<int>(sink.numel());
  TORCH_CHECK(p.sink_rows == 16 || p.sink_rows == 64, "attn: sink has 16 or 64 entries");
  p.scale_log2 = static_cast<float>(scale) * k3::LOG2E;
  p.positions = positions.has_value() ? reinterpret_cast<const long long*>(positions->data_ptr<int64_t>()) : nullptr;
  p.cos_sin = cos_sin.has_value() ? cos_sin->data_ptr<float>() : nullptr;
  TORCH_CHECK(p.cos_sin == nullptr || p.positions != nullptr, "attn: inverse RoPE needs positions");
  p.mn_lbo = static_cast<int>(mn_lbo);
  p.mn_sbo = static_cast<int>(mn_sbo);
  p.dbg_s = dbg_s.has_value() ? dbg_s->data_ptr<float>() : nullptr;
  p.prof = prof.has_value() ? reinterpret_cast<unsigned long long*>(prof->data_ptr<int64_t>()) : nullptr;
  p.abl = static_cast<int>(abl);
  static bool attr_dev[64] = {};  // per device (one process may launch on several GPUs)
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  if (!attr_dev[cur_dev & 63]) {
    check_cuda(cudaFuncSetAttribute(k3::attn_seq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, k3::SMEM_BYTES), "smem attr");
    attr_dev[cur_dev & 63] = true;
  }
  k3::attn_seq_kernel<<<nseq, k3::NT, k3::SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(p);
  check_cuda(cudaGetLastError(), "attn launch");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("meta", &meta, "per-step key union + membership + record addresses (K3 metadata)");
  m.def("attn", &attn, "per-sequence sparse MQA decode attention (K3)");
}
