// c1e0.cuh: E0 = the Engram wkv projections of BOTH Engram layers (1, 14) inside c1mega, TP4. Self-contained (namespace c1e0, own PTX helpers).
//
// The engine runs them on the owner GPU before / beside layer 0 (EngramPhys.prepare: hash, host-table gathers, two
// W8A16 GEMMs over 157 MB of e4m3 each, 77-172 us); a persistent 132-CTA launch cannot overlap them. v2 streams the
// engine's OWN fp8 weights in place: EngramPhys.gemm8[l].packed (kernels/attn_dense/w8a16.py repack: chunk-major
// [48 k-chunks of 128][100 row tiles of 256][32,896 B]; a tile = two 16,448-B halves of 128 rows: 16 KB of e4m3 A
// fragments (k32 block kb, 64-row block m, 16-row warp slot w, lane l = 4 g + t: 16 B = 4 words (rows g | g + 8) x
// (k 2t, 2t + 1 | + 8 | + 16 | + 24)), then 16 B of ue8m0 scales [kb][32-row block]). GPU r computes row tiles
// 25 r .. 25 r + 24 of both layers (no copies):
//   hook 2 (consumers, layer 0): the owner pushes the gathered rows x (EngramPhys.rows[l]: [NR][6144] bf16 per layer)
//       and keep [NR] to every peer (posted stores; one red.release.sys per owner CTA on the peer's x counter);
//   hook 1 (producer, layer 0, CTA c < 100): unit c = (layer c / 50, half-tile c % 50) of this GPU: 48 chunks, each the
//       half's 16,448 B + the NR x rows' 256 B of that k-chunk (peers: after the x counter);
//   hook 3 (consumers, layer 0): warp (m, w) owns rows 64 m + 16 w .. + 15 of the half: per k32 block cvt e4m3x2 -> f16
//       A fragments, B = x with an exact per-(row, k32) power-of-2 prescale, 2 f16 MMAs, fp32 accumulation with the
//       block scale; the 128 outputs x NR rows (bf16) posted to all 4 GPUs' kv buffers; then every CTA one
//       red.release.sys per layer on every GPU's kv counter (target 528 x call);
//   layers 1 / 14 hook 2: the injection reads kv locally after kvcnt[li] >= 528 x call (c1eng.cuh, local mode).
// Numerics: products exact (e4m3 x bf16 as f16 x f16 after an exact rescale), fp32 accumulation in another order than
// the W8A16 kernel / cuBLAS, one bf16 rounding (kv): same math (rounding-level differences in kv).
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace c1e0 {
constexpr int NRM = 4, KX = 6144, NO = 25600, NSL = NO / 4, NL = 2;
constexpr int KC = 128, NCH = KX / KC, HALF = 16448, TILE_B = 2 * HALF, NTILE = NO / 256;   // packed geometry
constexpr int UNITS_L = NSL / 128, UNITS = NL * UNITS_L;                                     // 50 per layer, 100
constexpr int XO = HALF, XST = 272;                       // x rows in the stage: NR x 256 B at XO + n XST (bank shift)
constexpr int STAGE_BYTES = HALF + NRM * XST;
constexpr int NCTA = 132, CTH = 256, W = 4;
static_assert(UNITS <= NCTA && NCH == 48 && NTILE == 100, "E0 geometry");

struct E0Args {
  int NR, owner;
  const uint8_t* wp[NL];           // this GPU's EngramPhys.gemm8[l].packed (chunk-major, see above)
  long long cstride;               // bytes between k-chunks (NTILE x 32,896)
  const __nv_bfloat16* xo[NL];     // owner: the gathered rows [NR][6144] (null elsewhere)
  const unsigned char* keepo;      // owner: [NR]
  __nv_bfloat16* xl[W][NL];        // rank s: landing rows [NRM][6144] (s != owner)
  unsigned char* keepl[W];         // rank s: [NRM] landing keep (the owner's own included)
  unsigned* xcnt[W];               // rank s: x arrivals (the owner's 132 CTAs per call)
  __nv_bfloat16* kvd[W][NL];       // rank s: kv [NRM][25600] bf16 per layer (key 0..3 | value, 5120 each)
  unsigned* kvcnt[W];              // rank s: [NL] arrivals (132 CTAs x 4 GPUs per layer per call)
  unsigned* tick;                  // this GPU: [132] per-CTA E0 epoch
};

__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, %0;" ::"n"(CTH) : "memory"); }
__device__ __forceinline__ void red_rel_sys(unsigned* p, unsigned v) {
  asm volatile("red.release.sys.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acq_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// waits poll with RELAXED loads and acquire ONCE: an acquire load compiles to LDG.STRONG + CCTL.IVALL (the whole
// SM's L1 invalidated per poll), which in situ stalls the other warps' LSU traffic (smem included) on the same SM
__device__ __forceinline__ unsigned ld_rlx_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_rlx(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fence_acq_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_acq() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  unsigned v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}
__device__ __forceinline__ void cvt_e4m3x4(unsigned v, unsigned& lo2, unsigned& hi2) {
  asm("{\n.reg .b16 l, h;\nmov.b32 {l, h}, %2;\ncvt.rn.f16x2.e4m3x2 %0, l;\ncvt.rn.f16x2.e4m3x2 %1, h;\n}"
      : "=r"(lo2), "=r"(hi2)
      : "r"(v));
}
__device__ __forceinline__ void mma16816(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned h2pack(float a, float b) {   // a -> low half
  const __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ float pow2f(int p) { return __int_as_float((127 + p) << 23); }   // |p| <= 126

__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }

// ---- hook 1 (producer warp, lane 0): unit c's 48 chunks. call = this CTA's E0 epoch (tick + 1); ns = the ring's stages.
// slot(bytes) -> stage index (waits empty, expects bytes); bulk(stage, offset, src, bytes) issues into the stage. Each
// stage expects the half-tile's 16,448 B AND the x rows' 256 B per row: the weights go out at once; on a peer the x
// copies follow as soon as the owner's push has landed (polled, never waited on while a stage is free), so a late push
// never holds back the weight stream; the producer only blocks on x before reusing a stage whose x is still missing.
template <typename Slot, typename Bulk>
__device__ __forceinline__ void e0_produce(const E0Args& E, int rank, int c, unsigned call, int ns, Slot slot, Bulk bulk) {
  if (c >= UNITS) return;
  const int li = c / UNITS_L, hu = c % UNITS_L;
  const uint8_t* wb = E.wp[li] + (size_t)(25 * rank + (hu >> 1)) * TILE_B + (size_t)(hu & 1) * HALF;
  const __nv_bfloat16* xs = rank == E.owner ? E.xo[li] : E.xl[rank][li];
  bool xok = rank == E.owner;
  int pend = 0;          // first chunk whose x copies are not issued yet
  int stg[8];            // stage of chunk k (k % 8; ns <= 8)
  auto xready = [&](bool block) {
    if (xok) return true;
    do {
      xok = (int)(ld_rlx_sys(E.xcnt[rank]) - (unsigned)NCTA * call) >= 0;
    } while (block && !xok);
    if (xok) {
      fence_acq_sys();
      fence_proxy_async_global();   // (the owner's generic stores -> the bulk copies' async proxy)
    }
    return xok;
  };
  auto issue_x = [&](int upto) {   // x of chunks pend .. upto - 1
    for (; pend < upto; ++pend)
      for (int n = 0; n < E.NR; ++n) bulk(stg[pend & 7], XO + n * XST, xs + (size_t)n * KX + pend * KC, 256u);
  };
#pragma unroll 1
  for (int kc = 0; kc < NCH; ++kc) {
    if (kc - pend >= ns && xready(true)) issue_x(kc - ns + 1);   // (the stage about to be reused needs its x first)
    const int s = slot((unsigned)(HALF + E.NR * 256));
    stg[kc & 7] = s;
    bulk(s, 0, wb + (size_t)kc * E.cstride, (unsigned)HALF);
    if (xready(false)) issue_x(kc + 1);
  }
  xready(true);
  issue_x(NCH);
}

// ---- hook 2 (consumers, layer 0, owner): push x (both layers) and keep to the peers, then one arrival per peer
__device__ void e0_push_x(const E0Args& E, int rank, int c, int tid) {
  if (rank != E.owner) return;
  constexpr int NQ = NL * NRM * (KX / 8);   // 16-B pieces
  constexpr int PER = (NQ + NCTA - 1) / NCTA;
  if (tid < PER) {
    const int i = c * PER + tid;
    if (i < NQ) {
      const int li = i / (NRM * (KX / 8)), n = (i / (KX / 8)) % NRM, q = i % (KX / 8);
      if (n < E.NR) {
        const uint4 v = *(reinterpret_cast<const uint4*>(E.xo[li] + (size_t)n * KX) + q);
#pragma unroll
        for (int s = 0; s < W; ++s)
          if (s != rank) *(reinterpret_cast<uint4*>(E.xl[s][li] + (size_t)n * KX) + q) = v;
      }
    }
  }
  if (c == 0 && tid < NRM) {
    const unsigned char kp = tid < E.NR ? E.keepo[tid] : (unsigned char)0;
#pragma unroll
    for (int s = 0; s < W; ++s) E.keepl[s][tid] = kp;
  }
  cbar();
  if (tid == 0) {   // every GPU's counter, the owner's own included: 132 arrivals per call everywhere (owners change)
    __threadfence_system();
#pragma unroll
    for (int s = 0; s < W; ++s) red_rel_sys(E.xcnt[s], 1u);
  }
}

// ---- hook 3 (consumers): unit c's GEMV. rwait() -> stage shared address (waits full), rrel() releases it (every warp).
// ob: >= NRM x 128 bf16 of shared scratch (c1mega's S.red); call = this CTA's E0 epoch (tick + 1).
template <typename RWait, typename RRel>
__device__ void e0_consume(const E0Args& E, int rank, int c, int tid, unsigned call, void* ob_, RWait rwait, RRel rrel) {
  const int lane = tid & 31, wv = tid >> 5, g = lane >> 2, t = lane & 3, m = wv >> 2, w4 = wv & 3;
  __nv_bfloat16* ob = static_cast<__nv_bfloat16*>(ob_);   // [NRM][128]
  if (c < UNITS) {
    const int li = c / UNITS_L, hu = c % UNITS_L;
    const bool hasg = g < E.NR;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int kc = 0; kc < NCH; ++kc) {
      const uint32_t st = rwait();
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        // B: row g's elements 32 kb + 2t + {0, 1, 8, 9, 16, 17, 24, 25} (bf16 pairs), scaled by 2^p of the k32 block
        uint32_t xb[4] = {0u, 0u, 0u, 0u};
        if (hasg)
#pragma unroll
          for (int j = 0; j < 4; ++j) xb[j] = lds32(st + XO + g * XST + (32 * kb + 2 * t + 8 * j) * 2);
        float f[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) f[2 * j] = __uint_as_float(xb[j] << 16), f[2 * j + 1] = __uint_as_float(xb[j] & 0xffff0000u);
        float mx = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) mx = fmaxf(mx, fabsf(f[i]));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        int p = 0;
        if (mx > 0.f) p = 140 - (int)((__float_as_uint(mx) >> 23) & 0xff);   // mx 2^p in [2^13, 2^14)
        p = max(-126, min(126, p));
        const float sx = pow2f(p);
        const unsigned b0 = h2pack(f[0] * sx, f[1] * sx), b1 = h2pack(f[2] * sx, f[3] * sx);
        const unsigned b2 = h2pack(f[4] * sx, f[5] * sx), b3 = h2pack(f[6] * sx, f[7] * sx);
        const int pa = __shfl_sync(0xffffffffu, p, 8 * t), pb = __shfl_sync(0xffffffffu, p, 8 * t + 4);   // rows 2t, 2t + 1
        // A: this lane's 16 B of the (kb, m, w4) fragment block: words (k 2t | +8 | +16 | +24) x (rows g | g + 8)
        const uint4 a = lds128(st + kb * 4096 + m * 2048 + w4 * 512 + lane * 16);
        unsigned a0, a1, a2, a3, c0, c1, c2, c3;
        cvt_e4m3x4(a.x, a0, a1);
        cvt_e4m3x4(a.y, a2, a3);
        cvt_e4m3x4(a.z, c0, c1);
        cvt_e4m3x4(a.w, c2, c3);
        float tmp[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816(tmp, a0, a1, a2, a3, b0, b1);
        mma16816(tmp, c0, c1, c2, c3, b2, b3);
        unsigned sb;
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(sb) : "r"(st + 16384 + 4 * kb + 2 * m + (w4 >> 1)));
        const float sc = __uint_as_float(sb << 23);   // 2^(sb - 127)
        const float ua = pow2f(-pa), ub = pow2f(-pb);
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[e] = fmaf(tmp[e] * ((e & 1) ? ub : ua), sc, acc[e]);
      }
      rrel();
    }
    // the half's outputs -> ob [n][row in half] (bf16), then posted to every GPU (one 16-B store per thread)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = 64 * m + 16 * w4 + g + 8 * (e >> 1), n = 2 * t + (e & 1);
      if (n < NRM) ob[n * 128 + row] = __float2bfloat16_rn(acc[e]);
    }
    cbar();
    const int s = tid >> 6, n = (tid >> 4) & 3, q = tid & 15;
    const int row0 = 256 * (25 * rank + (hu >> 1)) + 128 * (hu & 1);
    if (n < E.NR)
      *reinterpret_cast<uint4*>(E.kvd[s][li] + (size_t)n * NO + row0 + 8 * q) = reinterpret_cast<const uint4*>(ob)[n * 16 + q];
    cbar();
  }
  if (tid == 0) {   // every CTA, both layers (a CTA without a unit contributes nothing): the kv of this CTA is on every GPU
    __threadfence_system();
#pragma unroll
    for (int s = 0; s < W; ++s)
#pragma unroll
      for (int li = 0; li < NL; ++li) red_rel_sys(E.kvcnt[s] + li, 1u);
  }
  (void)call;
}
}  // namespace c1e0
