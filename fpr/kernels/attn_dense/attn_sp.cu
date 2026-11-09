// attn_sp.cu: sparse MQA decode attention in the sequence-parallel layout: one CTA per row,
// all 64 heads, the row's window (block-32 pages) + top-k (block-64 pages) records, sink; fused inverse RoPE of
// o dims 448..511. Replaces stock FlashMLA (split-KV main + combine) and dense.cu inv_rope in attn_half
// (ATTN2B_SPA=1).
//
// Why a new kernel: FlashMLA takes 53 us + 2.5 us combine + 3.6 us inv_rope in
// the 4-GPU chain for 9.7 GFLOP (204 TFLOP/s). Gather microbenchmark on measured index sets: one CTA per row with
// 8 warps issuing 16-B cp.async from SMEM-staged indices gathers a layer's 43 MB of row records in ~15 us (LDG + STS
// 54 us, one bulk copy per record 26 us: TMA issue-bound, per-chunk global index loads 39 us). MMA microbenchmark: 2
// warpgroups x (SS m64n64k16 Q K^T over 256 dims + RS m64n256k16 P V) run at the ideal 2,048 cycles per 64-key tile;
// Q in registers (RS Q K^T) does not fit next to the 128 O registers (255 + spills, wgmma serialized).
//
// v2 design (256 threads = 2 warpgroups, no producer warps, 1 CTA per SM, ~225 KB SMEM):
//  * Warpgroup w owns dims [256 w, 256 w + 256): its Q quarter (64 heads x 256 dims, SW128) in SMEM, its O half
//    (64 x 256 fp32) in 128 accumulator registers, its half of every bf16 key tile (64 keys x 256 dims).
//  * bf16 tiles double-buffered: while P V of tile t runs (async), both warpgroups dequantize tile t + 1 from the single
//    raw slot into the other buffer; while Q K^T of tile t runs (async), all 256 threads issue the cp.async gather of
//    tile t + 1 (e4m3 bytes + scale rows -> raw slot; the bf16 RoPE dims straight into the next bf16 tile).
//  * S = S_0 + S_1 (each warpgroup's SS partial over its 256 dims). The softmax is split by rows: warpgroup w owns
//    accumulator row rho = w of every thread (heads m0 + 8 rho), receives the other warpgroup's partial of that row
//    (8 KB exchange each way), runs the online softmax, and sends its P row (bf16 A-fragment words) and rescale factor
//    back (4.5 KB each way). The exchange buffers live in the next bf16 tile before it is dequantized.
//  * Numerics as stock FlashMLA / K3: exact fp8 x 2^(e-127) -> bf16 dequant, bf16 operands, fp32 accumulation, scores
//    x (scale * log2 e), exp2, P rounded to bf16 for P V, l summed in fp32 from the unrounded exponentials, the sink
//    in the denominator, o rounded to bf16 then inverse-rotated with stock's fp32 cos / sin (dense.cu's formula).
//    Only the fp32 summation order of S (two 256-dim halves) and of the online softmax (64-key tiles, no split-KV)
//    differs from FlashMLA.
//  * PDL: indices staged and the optional L2 weight prefetch issued before griddepcontrol.wait; q and the window
//    records the previous kernel inserted are read after it. launch_dependents after the main loop.
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

#include "wgmma_bf16_gen.cuh"

#ifndef SPA_PDL
#define SPA_PDL 1
#endif
#ifndef SPA_PROF
#define SPA_PROF 0  // 1: clock64 phase sums per (row, warpgroup) into p.prof
#endif
#ifndef SPA_SPLITG
#define SPA_SPLITG 0  // 1: tile t + 1's gather in two commit groups: records 0..39 under Q K^T, 40..63 after the softmax;
                      // the dequant does 0..39 first (cp.async.wait_group 1) and 40..63 after (wait_group 0)
#endif
#ifndef SPA_EXP
#define SPA_EXP 1  // softmax exponentials: 1 (default) = ex2.approx.ftz (1 MUFU; same agreement with FlashMLA), 0 = exp2f (denormal-exact, ~4 instructions)
#endif

namespace spa {
constexpr int BK = 64, NT = 256, REC = 576, MAXK = 1024;
constexpr int RW = 448;                              // raw slot: e4m3 bytes per record
constexpr int RAW_B = BK * RW + BK * 8;              // + 8-B scale rows = 29,184 B
constexpr int OFF_Q = 0;                             // 2 x 32 KB Q quarters (A operand, SW128 K-major)
constexpr int OFF_BF = 65536;                        // bf16 tiles [buffer 2][warpgroup 2] x 32 KB
constexpr int OFF_RAW = OFF_BF + 4 * 32768;
constexpr int OFF_IDX = OFF_RAW + RAW_B;             // MAXK int32 indices
constexpr int SMEM = OFF_IDX + MAXK * 4 + 1024;
static_assert(SMEM <= 232448, "smem");
// exchange regions inside the NEXT bf16 half tile of each warpgroup (free until its dequant): score row partial
// [4 float4][128 threads] at +0, P row words + alpha [9 words][128 threads] at +8 KB (warpgroup 1: below its RoPE
// column block 3 at +24 KB)
constexpr int XOFF = 0, POFF = 8192;
constexpr int MN_LBO = 8192, MN_SBO = 1024;
constexpr float LOG2E = 1.4426950408889634f;

struct Params {
  const __nv_bfloat16* q;  // [R, 64, 512] (RoPE'd)
  __nv_bfloat16* out;      // [R, 64, 512]
  const uint8_t* swa;      // window pages (block 32): [page][32 x 576 data | 32 x 8 scales], page stride `stride`
  const uint8_t* comp;     // compressed pages (block 64)
  long long stride;
  const int* si;  // [R, nsi] window slots
  const int* ti;  // [R, nti] compressed slots
  int nsi, nti;
  const int* sl;  // [R] window lengths
  const int* tl;  // [R] top-k lengths
  const float* sink;  // [64]
  float scale_log2;
  const long long* pos;  // [R] (inverse RoPE)
  const float* cos_sin;  // [P, 64] fp32 (cos 0..31 | sin 32..63) or nullptr
  unsigned long long* prof;  // SPA_PROF: [R][2][16] (8 loop phases, prologue, epilogue, globaltimer start / end, 4 prologue parts)
  const uint8_t* pf;         // optional L2 prefetch of the next kernel's weights: pf_bytes split over the CTAs
  long long pf_bytes;
  int ins;                   // trailing window entries per row the previous kernel may write (rows per sequence)
};

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

// tile t's records, all 256 threads, a warp per record: lanes 0..27 the 28 e4m3 chunks -> raw slot, lanes 28..31 and
// (second instruction) 0..3 the 8 bf16 RoPE chunks -> warpgroup 1's next bf16 half tile (column block 3, swizzled),
// lane 4 the 8-B scale row -> raw slot. The caller commits.
__device__ __forceinline__ void gather(const Params& p, const int* sidx, int sl, int n, int t, int warp8, int lane,
                                       uint32_t raw, uint32_t bf1, int k0 = 0, int k1 = BK) {
#pragma unroll 2
  for (int k = k0 + warp8; k < k1; k += 8) {
    const int j = min(t * BK + k, n - 1);
    const uint8_t *d, *s;
    rec_addr(p, sidx[j], j < sl, d, s);
    if (lane < 28)
      cp16(raw + k * RW + lane * 16, d + lane * 16);
    else
      cp16(bf1 + 3 * 8192 + k * 128 + (((lane - 28) ^ (k & 7)) << 4), d + RW + (lane - 28) * 16);
    if (lane < 4)
      cp16(bf1 + 3 * 8192 + k * 128 + (((lane + 4) ^ (k & 7)) << 4), d + RW + (lane + 4) * 16);
    else if (lane == 4)
      cp8(raw + BK * RW + k * 8, s);
  }
}

// raw slot -> this warpgroup's bf16 half tile (64 keys x 256 dims, 4 column blocks of 64 dims, SW128 K-major rows of
// 128 B). Warpgroup 0: e4m3 dims 0..255 (16 chunks per key); warpgroup 1: e4m3 dims 256..447 (12 chunks; its RoPE dims
// came in by cp.async).
__device__ __forceinline__ void dequant_tile(uint32_t raw, uint32_t bf, int wg, int ltid, int k0 = 0, int k1 = BK) {
  if (wg == 0) {
    const int c = ltid & 15;
#pragma unroll 4
    for (int jj = k0 / 8; jj < k1 / 8; ++jj) {  // k0, k1 multiples of 8
      const int k = (ltid >> 4) + 8 * jj;
      const uint4 v = lds128(raw + k * RW + c * 16);
      const unsigned E = lds_u8(raw + BK * RW + k * 8 + (c >> 2));
      uint4 lo, hi;
      dequant16(v, E, lo, hi);
      const uint32_t a = bf + (c >> 2) * 8192 + k * 128;
      const int q = (c & 3) * 2;
      sts128(a + ((q ^ (k & 7)) << 4), lo);
      sts128(a + (((q + 1) ^ (k & 7)) << 4), hi);
    }
  } else {
#pragma unroll 3
    for (int i = 12 * k0 + ltid; i < 12 * k1; i += 128) {  // 12 chunks per record
      const int k = i / 12, c = i - 12 * k;
      const uint4 v = lds128(raw + k * RW + 256 + c * 16);
      const unsigned E = lds_u8(raw + BK * RW + k * 8 + 4 + (c >> 2));
      uint4 lo, hi;
      dequant16(v, E, lo, hi);
      const uint32_t a = bf + (c >> 2) * 8192 + k * 128;
      const int q = (c & 3) * 2;
      sts128(a + ((q ^ (k & 7)) << 4), lo);
      sts128(a + (((q + 1) ^ (k & 7)) << 4), hi);
    }
  }
}

__device__ __forceinline__ float ex2(float x) {
  if (SPA_EXP) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  }
  return exp2f(x);
}
// online softmax of one accumulator row (rho) of a 64-key tile, in place on the full scores x[16]
// (x[2 j + e] = key 8 j + 2 t4 + e); returns the row's rescale factor
__device__ __forceinline__ float softmax_row(float (&x)[16], unsigned long long mask, float& m_run, float& l_run, float sl2, int t4) {
  if (mask == ~0ull) {  // full tile (warp-uniform): no masking
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] *= sl2;
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int c = 8 * j + 2 * t4 + e;
        x[2 * j + e] = ((mask >> c) & 1ull) ? x[2 * j + e] * sl2 : -INFINITY;
      }
  }
  float t[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) t[i] = fmaxf(fmaxf(x[4 * i], x[4 * i + 1]), fmaxf(x[4 * i + 2], x[4 * i + 3]));
  float tmax = fmaxf(fmaxf(t[0], t[1]), fmaxf(t[2], t[3]));
  tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
  tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
  const float mnew = fmaxf(m_run, tmax);
  const float alpha = mnew == -INFINITY ? 1.f : ex2(m_run - mnew);
  if (SPA_EXP) {  // ex2.approx(-inf) = +0; a fully masked row (mnew = -inf) subtracts 0 instead
    const float ms = mnew == -INFINITY ? 0.f : mnew;
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = ex2(x[i] - ms);
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = x[i] == -INFINITY ? 0.f : exp2f(x[i] - mnew);
  }
  float u[8];  // K3's order: pairs (key 8 j + 2 t4, + 1), then a tree over j
#pragma unroll
  for (int j = 0; j < 8; ++j) u[j] = x[2 * j] + x[2 * j + 1];
#pragma unroll
  for (int i = 0; i < 4; ++i) u[i] = u[2 * i] + u[2 * i + 1];
  l_run = l_run * alpha + ((u[0] + u[1]) + (u[2] + u[3]));
  m_run = mnew;
  return alpha;
}

__global__ void __launch_bounds__(NT, 1) attn_sp_kernel(const __grid_constant__ Params p) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t s0 = smem_u32(smem_raw), sb = (s0 + 1023u) & ~1023u;
  int* sidx = reinterpret_cast<int*>(smem_raw + (sb - s0) + OFF_IDX);
  const int r = blockIdx.x, tid = threadIdx.x, wg = tid >> 7, ltid = tid & 127;
  const int warp8 = __shfl_sync(0xffffffffu, tid >> 5, 0), warp = warp8 & 3, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  unsigned long long gt0 = 0;
  long long ck0 = 0;
  if (SPA_PROF) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt0));
    ck0 = clock64();
  }
  if (p.pf != nullptr && tid < 32) {  // static weights: prefetch before the PDL wait, 32 KB per lane-request
    const long long per = ((p.pf_bytes + gridDim.x - 1) / gridDim.x + 15) & ~15ll;
    const long long b0 = per * r, b1 = min(b0 + per, p.pf_bytes);
    for (long long b = b0 + 32768ll * tid; b < b1; b += 32768ll * 32)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.pf + b), "r"(static_cast<unsigned>(min(32768ll, b1 - b)))
                   : "memory");
  }
  const int sl = __ldg(p.sl + r), n = min(sl + __ldg(p.tl + r), MAXK), ntiles = (n + BK - 1) / BK;
  {  // indices -> SMEM, all loads of a thread in flight together
    int v[MAXK / NT];
#pragma unroll
    for (int q = 0; q < MAXK / NT; ++q) {
      const int j = tid + NT * q;
      v[q] = j < n ? (j < sl ? __ldg(p.si + static_cast<long long>(r) * p.nsi + j) : __ldg(p.ti + static_cast<long long>(r) * p.nti + (j - sl)))
                   : -1;
    }
#pragma unroll
    for (int q = 0; q < MAXK / NT; ++q)
      if (tid + NT * q < n) sidx[tid + NT * q] = v[q];
  }
  __syncthreads();
  long long pq[4] = {0, 0, 0, 0};
  (void)pq;
  if (SPA_PROF) pq[0] = clock64() - ck0;
  const uint32_t QS = sb + OFF_Q + 32768 * wg, RAW = sb + OFF_RAW;
  auto bfh = [&](int b, int w) -> uint32_t { return sb + OFF_BF + 65536 * b + 32768 * w; };
  // tile 0 before the PDL wait when none of its window entries is one the previous kernel may write (the last p.ins)
  const bool early = ntiles > 0 && max(0, sl - p.ins) >= min(BK, sl);
  if (early) gather(p, sidx, sl, n, 0, warp8, lane, RAW, bfh(0, 1));
  if (SPA_PROF) pq[1] = clock64() - ck0;
  if (SPA_PDL) asm volatile("griddepcontrol.wait;" ::: "memory");  // q, inserted window records
  if (SPA_PROF) pq[2] = clock64() - ck0;
  {  // Q quarter -> SMEM; tile 0 -> raw slot + RoPE into buffer 0
    const __nv_bfloat16* qs = p.q + static_cast<long long>(r) * 64 * 512 + 256 * wg;
#pragma unroll 4
    for (int i = ltid; i < 64 * 32; i += 128) {
      const int m = i >> 5, ch = i & 31, cb = ch >> 3, c8 = ch & 7;
      cp16(QS + cb * 8192 + m * 128 + ((c8 ^ (m & 7)) << 4), qs + m * 512 + ch * 8);
    }
    if (ntiles > 0 && !early) gather(p, sidx, sl, n, 0, warp8, lane, RAW, bfh(0, 1));
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    if (SPA_PROF) pq[3] = clock64() - ck0;
    if (ntiles > 0) dequant_tile(RAW, bfh(0, wg), wg, ltid);
    fence_proxy_async();
    __syncthreads();
  }
  float o[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) o[i] = 0.f;
  float m_run = -INFINITY, l_run = 0.f;  // this warpgroup's softmax row (rho = wg)
  unsigned long long pt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  (void)pt;
  if (SPA_PROF) {
    pt[8] = clock64() - ck0;
#pragma unroll
    for (int i = 0; i < 4; ++i) pt[12 + i] = pq[i];
  }
  const int rho = wg;

  for (int t = 0; t < ntiles; ++t) {
    const int cur = t & 1, nxt = cur ^ 1;
    const bool more = t + 1 < ntiles;
    long long tq = SPA_PROF ? clock64() : 0;
    // ---- A: Q K^T partial of tile t over my 256 dims (async); B: gather tile t + 1 meanwhile
    float sacc[32];
    wg_fence();
#pragma unroll
    for (int s = 0; s < 16; ++s) {
      const uint32_t off = (s >> 2) * 8192 + (s & 3) * 32;
      WgSS<64, 0>::mma(sacc, desc_k(QS + off), desc_k(bfh(cur, wg) + off), s > 0);
    }
    wg_commit();
    if (more) gather(p, sidx, sl, n, t + 1, warp8, lane, RAW, bfh(nxt, 1), 0, SPA_SPLITG ? 40 : BK);
    cp_commit();
    if (SPA_PROF) { const long long c = clock64(); pt[0] += c - tq; tq = c; }
    unsigned long long mask;
    {
      const int j0 = t * BK + lane, j1 = j0 + 32;
      const bool v0 = j0 < n && sidx[j0 < n ? j0 : 0] >= 0, v1 = j1 < n && sidx[j1 < n ? j1 : 0] >= 0;
      mask = (static_cast<unsigned long long>(__ballot_sync(0xffffffffu, v1)) << 32) | __ballot_sync(0xffffffffu, v0);
    }
    wg_wait<0>();
    if (SPA_PROF) { const long long c = clock64(); pt[1] += c - tq; tq = c; }
    // ---- exchange: I send the other row's partial, receive mine; x = full scores of my row
    const uint32_t xm = bfh(nxt, wg) + XOFF + ltid * 16, xo = bfh(nxt, wg ^ 1) + XOFF + ltid * 16;
    // row 0 values of sacc: [4 j + e], row 1: [4 j + 2 + e] (static indices + selects: no local-memory accumulator)
    const bool r1 = rho == 1;
#pragma unroll
    for (int v = 0; v < 4; ++v) {  // the other row's 16 values (keys 16 v + 2 t4 + e and 16 v + 8 + 2 t4 + e)
      sts128f(xm + v * 2048, r1 ? sacc[8 * v] : sacc[8 * v + 2], r1 ? sacc[8 * v + 1] : sacc[8 * v + 3],
              r1 ? sacc[8 * v + 4] : sacc[8 * v + 6], r1 ? sacc[8 * v + 5] : sacc[8 * v + 7]);
    }
    named_bar(1, NT);
    float x[16];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const float4 y = lds128f(xo + v * 2048);
      // S = S_0 + S_1 (fadd is commutative: the same value in either warpgroup's order)
      x[4 * v] = (r1 ? sacc[8 * v + 2] : sacc[8 * v]) + y.x;
      x[4 * v + 1] = (r1 ? sacc[8 * v + 3] : sacc[8 * v + 1]) + y.y;
      x[4 * v + 2] = (r1 ? sacc[8 * v + 6] : sacc[8 * v + 4]) + y.z;
      x[4 * v + 3] = (r1 ? sacc[8 * v + 7] : sacc[8 * v + 5]) + y.w;
    }
    if (SPA_PROF) { const long long c = clock64(); pt[2] += c - tq; tq = c; }
    const float alpha_me = softmax_row(x, mask, m_run, l_run, p.scale_log2, t4);
    if (SPA_SPLITG && more) {  // the second part of tile t + 1's gather: its own commit group
      gather(p, sidx, sl, n, t + 1, warp8, lane, RAW, bfh(nxt, 1), 40, BK);
      cp_commit();
    }
    // my P row: words for k-steps ks = 0..3, (keys 16 ks + 2 t4, +1) and (+8, +9) -> A-fragment slots 2 ks' + rho
    unsigned pm[8];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      pm[2 * ks] = pack_bf16(x[4 * ks], x[4 * ks + 1]);
      pm[2 * ks + 1] = pack_bf16(x[4 * ks + 2], x[4 * ks + 3]);
    }
    const uint32_t pmy = bfh(nxt, wg) + POFF + ltid * 4, poth = bfh(nxt, wg ^ 1) + POFF + ltid * 4;
#pragma unroll
    for (int i = 0; i < 8; ++i) asm volatile("st.shared.u32 [%0], %1;" ::"r"(pmy + i * 512), "r"(pm[i]) : "memory");
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(pmy + 8 * 512), "f"(alpha_me) : "memory");
    if (SPA_PROF) { const long long c = clock64(); pt[3] += c - tq; tq = c; }
    named_bar(2, NT);
    unsigned po[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(po[i]) : "r"(poth + i * 512) : "memory");
    float alpha_o;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(alpha_o) : "r"(poth + 8 * 512) : "memory");
    unsigned pf[4][4];
    const float a0 = rho == 0 ? alpha_me : alpha_o, a1 = rho == 0 ? alpha_o : alpha_me;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {  // a0 = (row g, keys 16 ks + 2 t4), a1 = (g + 8, ..), a2 = (g, +8), a3 = (g + 8, +8)
      pf[ks][0] = rho == 0 ? pm[2 * ks] : po[2 * ks];
      pf[ks][1] = rho == 0 ? po[2 * ks] : pm[2 * ks];
      pf[ks][2] = rho == 0 ? pm[2 * ks + 1] : po[2 * ks + 1];
      pf[ks][3] = rho == 0 ? po[2 * ks + 1] : pm[2 * ks + 1];
    }
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      o[4 * j] *= a0;
      o[4 * j + 1] *= a0;
      o[4 * j + 2] *= a1;
      o[4 * j + 3] *= a1;
    }
    // ---- P V of tile t (async); dequant tile t + 1 meanwhile
    wg_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) WgRS<256, 1>::mma(o, pf[ks], desc_mn(bfh(cur, wg) + ks * 2048), 1);
    wg_commit();
    if (SPA_PROF) { const long long c = clock64(); pt[4] += c - tq; tq = c; }
    if (more) {
      if (SPA_SPLITG) {
        cp_wait<1>();
        named_bar(3, NT);  // records 0..39 of tile t + 1 landed (all threads); every exchange read of this tile is done
        if (SPA_PROF) { const long long c = clock64(); pt[5] += c - tq; tq = c; }
        dequant_tile(RAW, bfh(nxt, wg), wg, ltid, 0, 40);
        cp_wait<0>();
        named_bar(7, NT);  // records 40..63
        dequant_tile(RAW, bfh(nxt, wg), wg, ltid, 40, BK);
      } else {
        cp_wait<0>();
        named_bar(3, NT);  // tile t + 1's copies landed (all threads); every exchange read of this tile is done
        if (SPA_PROF) { const long long c = clock64(); pt[5] += c - tq; tq = c; }
        dequant_tile(RAW, bfh(nxt, wg), wg, ltid);
      }
      fence_proxy_async();
    }
    if (SPA_PROF) { const long long c = clock64(); pt[6] += c - tq; tq = c; }
    wg_wait<0>();
    named_bar(4, NT);  // tile t + 1 complete in buffer nxt; P V t done reading buffer cur; raw slot free
    if (SPA_PROF) { const long long c = clock64(); pt[7] += c - tq; }
  }
  cp_wait<0>();
  if (SPA_PDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const long long ck1 = SPA_PROF ? clock64() : 0;

  // ---- o / (l + e^(sink - m)) (log2 units) for my row, the other row's factor from the other warpgroup
  float csv[16];
  const bool csv_ok = p.cos_sin != nullptr;
  if (wg == 1 && csv_ok) {  // this thread's 8 RoPE pairs i = 4 jj + t4: cos, sin (loads issued before the barriers)
    const float* cs = p.cos_sin + __ldg(p.pos + r) * 64;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      csv[jj] = __ldg(cs + 4 * jj + t4);
      csv[8 + jj] = __ldg(cs + 32 + 4 * jj + t4);
    }
  }
  {
    float l = l_run;
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const float m = m_run;
    const float sink2 = __ldg(p.sink + 16 * warp + g + 8 * rho) * LOG2E;
    const float mf = fmaxf(m, sink2);
    const float f = m == -INFINITY ? 0.f : exp2f(m - mf);
    const float den = l * f + (sink2 == -INFINITY ? 0.f : exp2f(sink2 - mf));
    const float so_me = (den > 0.f && mf < INFINITY) ? f / den : 0.f;
    const uint32_t a = sb + OFF_RAW + (wg * 128 + ltid) * 4;  // raw slot is free now
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(a), "f"(so_me) : "memory");
    named_bar(5, NT);
    float so_o;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(so_o) : "r"(sb + OFF_RAW + ((wg ^ 1) * 128 + ltid) * 4) : "memory");
    const float so0 = rho == 0 ? so_me : so_o, so1 = rho == 0 ? so_o : so_me;
    const int m0 = 16 * warp + g;
    // o tile (64 heads x 512 dims bf16) -> SMEM (the free bf16 buffers), 16-B chunks XOR-swizzled by (head & 7)
    const uint32_t ST = sb + OFF_BF;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int d = 256 * wg + 8 * j + 2 * t4;
      float v0 = bf16r(o[4 * j] * so0), v1 = bf16r(o[4 * j + 1] * so0);
      float v2 = bf16r(o[4 * j + 2] * so1), v3 = bf16r(o[4 * j + 3] * so1);
      if (wg == 1 && j >= 24 && csv_ok) {  // inverse RoPE, pair i = (d - 448) / 2
        const float c = csv[j - 24], sn = csv[8 + j - 24];
        const float a0 = __fadd_rn(__fmul_rn(v0, c), __fmul_rn(v1, sn)), b0 = __fsub_rn(__fmul_rn(v1, c), __fmul_rn(v0, sn));
        const float a1 = __fadd_rn(__fmul_rn(v2, c), __fmul_rn(v3, sn)), b1 = __fsub_rn(__fmul_rn(v3, c), __fmul_rn(v2, sn));
        v0 = a0, v1 = b0, v2 = a1, v3 = b1;
      }
      const int ch = d >> 3, off = (d & 7) * 2;
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(ST + m0 * 1024 + ((ch ^ (m0 & 7)) << 4) + off), "r"(pack_bf16(v0, v1)) : "memory");
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(ST + (m0 + 8) * 1024 + ((ch ^ ((m0 + 8) & 7)) << 4) + off), "r"(pack_bf16(v2, v3))
                   : "memory");
    }
    named_bar(6, NT);
    __nv_bfloat16* orow = p.out + static_cast<long long>(r) * 64 * 512;
#pragma unroll 4
    for (int i = tid; i < 64 * 64; i += NT) {  // 16-B chunks, a warp writes 512 contiguous bytes
      const int m = i >> 6, ch = i & 63;
      const uint4 v = lds128(ST + m * 1024 + ((ch ^ (m & 7)) << 4));
      *reinterpret_cast<uint4*>(orow + m * 512 + ch * 8) = v;
    }
  }
  if (SPA_PROF && p.prof != nullptr && ltid == 0) {
    pt[9] = clock64() - ck1;
    pt[10] = gt0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(pt[11]));
#pragma unroll
    for (int i = 0; i < 16; ++i) p.prof[(static_cast<long long>(r) * 2 + wg) * 16 + i] = pt[i];
  }
}
}  // namespace spa

void attn_sp(torch::Tensor q, torch::Tensor out, torch::Tensor swa, torch::Tensor comp, int64_t stride, torch::Tensor si,
             torch::Tensor ti, torch::Tensor sl, torch::Tensor tl, torch::Tensor sink, double scale,
             c10::optional<torch::Tensor> pos, c10::optional<torch::Tensor> cos_sin, c10::optional<torch::Tensor> prof,
             c10::optional<torch::Tensor> pf, int64_t pf_bytes, int64_t ins) {
  const int R = static_cast<int>(sl.numel());
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous() && q.numel() == static_cast<long long>(R) * 64 * 512, "q / out [R, 64, 512]");
  TORCH_CHECK(si.dtype() == torch::kInt32 && ti.dtype() == torch::kInt32 && sl.dtype() == torch::kInt32 &&
                  tl.dtype() == torch::kInt32 && si.is_contiguous() && ti.is_contiguous(),
              "int32 contiguous indices / lengths");
  spa::Params p{};
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.swa = swa.data_ptr<uint8_t>();
  p.comp = comp.data_ptr<uint8_t>();
  p.stride = stride;
  p.si = si.data_ptr<int>();
  p.ti = ti.data_ptr<int>();
  p.nsi = static_cast<int>(si.numel() / R);
  p.nti = static_cast<int>(ti.numel() / R);
  p.sl = sl.data_ptr<int>();
  p.tl = tl.data_ptr<int>();
  p.sink = sink.data_ptr<float>();
  p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  p.pos = pos.has_value() ? reinterpret_cast<const long long*>(pos->data_ptr<int64_t>()) : nullptr;
  p.cos_sin = cos_sin.has_value() ? cos_sin->data_ptr<float>() : nullptr;
  TORCH_CHECK(p.cos_sin == nullptr || p.pos != nullptr, "inverse RoPE needs positions");
  p.prof = prof.has_value() ? reinterpret_cast<unsigned long long*>(prof->data_ptr<int64_t>()) : nullptr;
  p.pf = pf.has_value() && pf_bytes > 0 ? reinterpret_cast<const uint8_t*>(pf->data_ptr()) : nullptr;
  p.pf_bytes = p.pf != nullptr ? (std::min<long long>(pf_bytes, static_cast<long long>(pf->nbytes())) & ~15ll) : 0;
  p.ins = static_cast<int>(ins);
  auto kern = spa::attn_sp_kernel;
  static bool init_dev[64] = {};  // per device (one process may launch on several GPUs)
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  if (!init_dev[cur_dev & 63]) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, spa::SMEM) == cudaSuccess, "smem attr");
    init_dev[cur_dev & 63] = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(R);
  cfg.blockDim = dim3(spa::NT);
  cfg.dynamicSmemBytes = spa::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = SPA_PDL ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, p) == cudaSuccess, "attn_sp launch");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_sp", &attn_sp, "SP sparse MQA decode attention (one CTA per row, 64 heads)"); }
