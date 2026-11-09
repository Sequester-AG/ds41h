// moe_raw: the v3 MoE kernels reading ds41h's RAW expert layout (for serving: no second copy).
//
// Raw TP-sliced layout (ds41h.moe get_raw_experts, moe_dec / moe_prefill / decoder.cu): w13 [E, 2I, H/2] (gate rows 0..I-1,
// up rows I..2I-1; low nibble = even k), s13 [E, 2I, H/32] ue8m0, w2 [E, H, I/2], s2 [E, H, I/32]. One copy of the
// experts serves every path (a repacked copy would cost ~72 GB per GPU).
// Against moe_stream3.cu:
//  * weight tiles by TMA 2D boxes from the raw rows: GEMM1 gate / up 64-row boxes of 128 B (128B swizzle), GEMM2 128-row
//    boxes of CH x 32 B (no swizzle, 288-B rows); scales: s13 16-B boxes, s2 one 2,304-B bulk copy per tile;
//  * A fragments by ldmatrix.x4 (4 words: rows g / g+8 x two k32 blocks); a word = 8 consecutive k of one row, so the
//    activations use a fixed K order inside every 32-block (logical L = 16 s + kk <-> physical 8t + 4 (kk >> 3) + 2e + s,
//    t = (kk & 7) >> 1, e = kk & 1): the token images (k_gather) and GEMM1's H image are written in that order;
//  * raw scale bytes biased in-kernel ((s - 106) << 2 per byte, 2 ops per 4 scales) for the f16 multiplier 2^(s-121).
// Numerics as v3 (exact products, fp32 accumulation, routing weight before w2, bf16 per-slot outputs + fixed-order combine).
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

#include "wgmma_rs_gen.cuh"

#ifndef MS3_MMA_MAXN
#define MS3_MMA_MAXN 32  // n_pad <= this: mma.sync path; larger: RS-wgmma (sweep: 32 best, 16 -3 %, 64 -5 %)
#endif
#ifndef MS3_CHS1
#define MS3_CHS1 8  // GEMM1 k64 steps per stage for n_pad <= 32 (stage-depth sweep)
#endif
#ifndef MS3_S
#define MS3_S 3  // ring stages
#endif
#ifndef MS3_CTAS
#define MS3_CTAS 1  // GEMM CTAs per SM (2: smaller rings, 16 consumer warps per SM, <= 112 registers)
#endif
#ifndef MS3_CHS2
#define MS3_CHS2 9  // GEMM2 k64 steps per stage for n_pad <= 24 (9 = whole tile)
#endif
#ifndef MS3_CHB2
#define MS3_CHB2 3  // GEMM2 k64 steps per stage above 24 tokens
#endif
#ifndef MS3_S2
#define MS3_S2 3  // GEMM2 ring stages
#endif
#ifndef MS3_PSLEEP
#define MS3_PSLEEP 0  // producer: nanoseconds of sleep between empty-barrier polls (0 = spin)
#endif
#ifndef MS3_EPF
#define MS3_EPF 1  // 1: the producer copies the tile's EC (tokens, weights, 2^-p) into the first stage, consumers keep it in
                   //    registers, so the epilogue has no dependent global loads
#endif
#ifndef MS3_RED2
#define MS3_RED2 0  // 1: GEMM2 output as red.global.add.v2.f32 on row pairs (half the requests); measured no gain (L2
                   //    atomic units are per element), so off
#endif
#ifndef MS3_YSTG
#define MS3_YSTG 1  // 1 (default): stage GEMM2 bf16 Y as [token][128 rows] in SMEM, coalesced 16-B stores (-12 % GEMM2 energy, measured)
#endif
#ifndef MSR_G2BULK
#define MSR_G2BULK 1  // GEMM2 tiles with the whole K in one stage (CH = 9): ONE 36,864-B bulk copy of the 128 contiguous
                      // w2 rows instead of three 96-B x 128-row 2D boxes; SMEM layout [128 rows][288 B]
#endif
#ifndef MSR_G2PAIR
#define MSR_G2PAIR 0  // (off: measured 197 vs 185 us; the strided 256-row boxes lose to the 36 KB bulk) GEMM2 tiles of ECs with n_pad <= MS3_MMA_MAXN: 256 rows (q even; odd q tiles skipped), one H image
                      // per stage shared by both 128-row halves, one staged epilogue (half the per-tile overhead)
#endif
#ifndef MSR_G1PAIR
#define MSR_G1PAIR 0  // (off: measured 366 vs 331 us; half the K-depth per stage costs more than the halved B traffic) GEMM1 tiles q = 0..7 of ECs with n_pad <= 16: pairs (q, q+1) = 128 intermediates (256 weight rows);
                      // each token-image chunk and B fragment serves twice the rows (half the B-side traffic)
#endif
#ifndef MSR_G1PAIR_CH
#define MSR_G1PAIR_CH 4  // k64 steps per stage of a paired GEMM1 tile (8 needs A_MAX 64 KB -> use MS3_S=2)
#endif
#ifndef MS4_HQ
#define MS4_HQ 1  // moe_v4: the tile's q from the stage header instead of a global Tile load in the epilogue
#endif
#ifndef MS4_YWARP
#define MS4_YWARP 1  // moe_v4: GEMM2 Y staged per warp ([token][16 rows], __syncwarp only, 2 x 16-B stores per token) instead of
                     // per CTA ([token][128 rows], two 256-thread barriers per tile); same bytes
#endif
#ifndef MS4_G2P
#define MS4_G2P 0  // moe_v4: GEMM2 items flagged q | 0x100 (route2.cu built with R2_G2PAIR=1: ECs with n_pad <= 16) are
                   // 256-row pairs: ONE 73,728-B bulk copy of the contiguous rows, the H image once, B fragments shared
                   // by both 128-row halves, warp-staged Y; 2 ring stages of 96 KB
#endif
#ifndef MS4_DYN
#define MS4_DYN 0  // moe_v4: k_gemm producers take items from an atomic queue in counts[7] (list order, one grab ahead)
                   // instead of the static round-robin; the CTA making the last failing grab resets the counter to 0
#endif
#ifndef MS4_FSIG
#define MS4_FSIG 1  // fused kernel: 1 = consumers arrive on a CTA-local mbarrier per GEMM1 tile and the producer lane does the
                    // GPU-scope fence + count (no consumer stalls); 0 = per-warp fence + atomic (v2)
#endif
#ifndef MS4_PDL
#define MS4_PDL 0  // moe_v4: programmatic dependent launch: GEMM1 lets its dependents launch at once; GEMM2 is launched with
                   // programmatic stream serialization and runs griddepcontrol.wait after its prologue (barrier init)
#endif
#ifndef MS4_SPOL
#define MS4_SPOL 0  // moe_v4: GEMM1 scale boxes (16 B of each row per stage = half a 32-B sector; the next stage reads the
                    // other half) with the L2 evict_normal policy (1) / evict_last (2) instead of evict_first
#endif
#ifndef MS4_NS2
#define MS4_NS2 24  // moe_v4: GEMM2 tiles with n_pad <= this take the whole K in one stage (40 fits the MS4_G2P stage)
#endif
#ifndef MS4_G1P
#define MS4_G1P 0  // moe_v4: GEMM1 items flagged q | 0x100 (route2.cu R2_G1PAIR=1) are 128-intermediate pairs (blocks q, q+1):
                   // MSR_G1PAIR's consumer (B fragments shared by both row sets) on a balanced item list
#endif
#ifndef MS4_NOIMG
#define MS4_NOIMG 0  // timing / energy ablation: the producer skips the token-image / H copies (stale data; results invalid)
#endif
#ifndef MS4_NOW
#define MS4_NOW 0  // timing / energy ablation: the producer skips the weight + scale copies (stale data; results invalid)
#endif
#ifndef MS4_TRACE
#define MS4_TRACE 0  // moe_v4: 1 = consumer thread 0 records per tile {globaltimer, first-wait, mainloop, total, inner waits (cycles), smid, npad}
#endif
#ifndef MS4_NOSTREAM
#define MS4_NOSTREAM 0  // moe_v4 ablation: 1 = the producer copies only the EC records (consumer-bound timing; results invalid)
#endif
#ifndef MS3_ABL
#define MS3_ABL 0  // timing-only ablations (results invalid): 1 stream + A fragments + scales; 3 + B (image) fragments;
                   // 4 + dequant, no MMA; 5 MMA on the raw words, no dequant; 6 = 1 without the image / H loads; 7 = 6 without the GEMM2 Y stores;
                   // 8 = 6 without GEMM1's scale boxes; 9 = 8 with the GEMM1 weight boxes replaced by contiguous bulk copies (same bytes)
#endif

namespace msr {
constexpr int TRACE_MAX = 16384;
__device__ unsigned long long g_trace[2][TRACE_MAX][8];
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

constexpr int D = 5120, IR = 576, NMAX = 64, NCW = 8, HDR = 1024;  // Hdr at +0, EC copy at +128
template <int MODE>
constexpr int SS = MODE == 1 ? MS3_S : MS3_S2;  // ring stages per GEMM

struct EC {  // one expert chunk (<= 64 tokens); v3 adds inv (2^-p of each token, written by k_gather)
  int e, n_real, n_pad, pad;
  long long img_off, h_off;
  int tok[NMAX];
  float wt[NMAX];
  float inv[NMAX];
};
static_assert(sizeof(EC) == 32 + 3 * 4 * NMAX, "EC layout must match bench.py (ec_inv=True)");
struct Tile {
  int ec, q, e, npad;
  long long boff;
};
static_assert(sizeof(Tile) == 24, "Tile layout must match bench.py");
struct Hdr {
  int tile, ch, npad, nch, pair, q, ec;  // moe_v4: q = the tile's row block (no global Tile load in the epilogue), ec index
};

template <int MODE>
struct Cfg;
template <>
struct Cfg<1> {  // GEMM1: K = 5120 = 80 k64 steps; 8 per stage (n_pad <= 32) or 4
  static constexpr int KB = 80, QPE = 9, NSMALL = 32, CHS = MS3_CHS1, CHB = MS3_CHS1 / 2;
};
static_assert(MS3_CHS2 % 3 == 0 && MS3_CHB2 % 3 == 0, "raw GEMM2 loads w2 in 3-step (96-byte) boxes");
template <>
struct Cfg<2> {  // GEMM2: K = 576 = 9 k64 steps; the whole tile per stage (n_pad <= 24) or 3
  static constexpr int KB = 9, QPE = 40, NSMALL = MS4_NS2, CHS = MS3_CHS2, CHB = MS3_CHB2;
};
template <int MODE>
struct Lay {
  using C = Cfg<MODE>;
  static constexpr int B_MAX = (C::CHS * C::NSMALL > C::CHB * NMAX ? C::CHS * C::NSMALL : C::CHB * NMAX) * 128;
  static constexpr int A_MAX = (MODE == 1 && (MSR_G1PAIR || MS4_G1P) && MSR_G1PAIR_CH * 8192 > C::CHS * 4096) ? MSR_G1PAIR_CH * 8192 : C::CHS * 4096,
                       S_MAX = MODE == 1 ? ((MSR_G1PAIR || MS4_G1P) ? 4096 : 2048) : (MSR_G2PAIR ? 4608 : 2304);
  static constexpr int OFF_A = B_MAX, OFF_S = B_MAX + A_MAX;
  // MS4_G2P pair stages (GEMM2, n_pad <= 16): H image [0, 18,432), 256 rows x 288 B, 256 x 18 B scales
  static constexpr int P_OFF_A = 16 * 1152, P_OFF_S = P_OFF_A + 256 * (IR / 2);
  static constexpr int OFF_H = (MODE == 2 && MS4_G2P) ? (P_OFF_S + 256 * 18 > OFF_S + S_MAX ? P_OFF_S + 256 * 18 : OFF_S + S_MAX)
                                                      : OFF_S + S_MAX;
  static constexpr int STAGE = ((OFF_H + HDR) + 1023) / 1024 * 1024;
  static constexpr int STG = (MODE == 2 && MS3_YSTG) ? NCW * 16 * NMAX * 2 : 0;  // bf16 Y staging [token][16 rows]
  static constexpr int SMEM = 1024 + SS<MODE> * STAGE + 1024 + STG;
};
static_assert(Lay<1>::SMEM <= 227 * 1024 / MS3_CTAS - 1024 * (MS3_CTAS - 1) && Lay<2>::SMEM <= 227 * 1024 / MS3_CTAS - 1024 * (MS3_CTAS - 1),
              "ring must fit in SMEM (per CTA)");

// ------------------------------------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t a, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, unsigned b) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(b) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W_%=;\n}\n" ::"r"(
                   a),
               "r"(parity)
               : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t a, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(a), "r"(parity) : "memory");
  return ok;
}
__device__ __forceinline__ void mbar_wait_sleep(uint32_t a, uint32_t parity, unsigned ns) {
  while (!mbar_try(a, parity)) __nanosleep(ns);
}
__device__ __forceinline__ uint64_t pol_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t pol_normal() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t pol_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, unsigned bytes, uint32_t mbar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
          dst),
      "l"(src), "r"(bytes), "r"(mbar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void tma_row(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t mbar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
      "l"(tm), "r"(c0), "r"(c1), "r"(mbar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_arrive(uint32_t mbar) {  // arrives when this thread's cp.asyncs land
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(mbar) : "memory");
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ unsigned lds32(uint32_t a) {
  unsigned v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], uint32_t a) {
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}
__device__ __forceinline__ void mma16816(float* d, const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void red_v2(float* p, float x, float y) {  // p 8-byte aligned
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(p), "f"(x), "f"(y) : "memory");
}
// K order inside a 32-block: logical L (mma / token-image order) <-> physical k (raw nibble order)
__host__ __device__ __forceinline__ int kperm(int L) {  // logical -> physical
  const int s = L >> 4, kk = L & 15;
  return 8 * ((kk & 7) >> 1) + 4 * (kk >> 3) + 2 * (kk & 1) + s;
}
__host__ __device__ __forceinline__ int kinv(int p) {  // physical -> logical
  return 16 * (p & 1) + 8 * ((p >> 2) & 1) + 2 * (p >> 3) + ((p >> 1) & 1);
}
__device__ __forceinline__ void ldsm_x4a(unsigned (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ unsigned bias4(unsigned u) { return (u - 0x6a6a6a6au) << 2; }  // 4 bytes s -> (s - 106) << 2
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t mbar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(mbar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, int c2, uint32_t mbar,
                                      uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(c2), "r"(mbar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {  // K-major, 128-B rows, 128B swizzle, SBO 1024
  return ((uint64_t(saddr) & 0x3FFFFull) >> 4) | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
}
__device__ __forceinline__ unsigned fp4_lo_e4m3(unsigned w) { return ((w << 4) & 0x80808080u) | ((w << 2) & 0x1c1c1c1cu); }
__device__ __forceinline__ unsigned fp4_hi_e4m3(unsigned w) { return (w & 0x80808080u) | ((w >> 2) & 0x1c1c1c1cu); }
__device__ __forceinline__ unsigned cvt_e4m3x2(unsigned v16) {
  unsigned h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)v16));
  return h;
}
__device__ __forceinline__ unsigned hmul2(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ unsigned swz(int n, int b) {
  return (n >> 3) * 1024 + (n & 7) * 128 + ((((b >> 4) ^ (n & 7)) & 7) << 4) + (b & 15);
}
#ifndef MS3_CVT2
#define MS3_CVT2 1  // 1: both e4m3 pairs of a register in one asm block (mov.b32 {l, h}): no shift for the high half
#endif
// e4m3x4 register -> two f16x2 (low pair, high pair)
__device__ __forceinline__ void cvt_e4m3x4(unsigned v, unsigned& lo2, unsigned& hi2) {
  if (MS3_CVT2) {
    asm("{\n.reg .b16 l, h;\nmov.b32 {l, h}, %2;\ncvt.rn.f16x2.e4m3x2 %0, l;\ncvt.rn.f16x2.e4m3x2 %1, h;\n}"
        : "=r"(lo2), "=r"(hi2) : "r"(v));
  } else {
    lo2 = cvt_e4m3x2(v & 0xffff);
    hi2 = cvt_e4m3x2(v >> 16);
  }
}
// one word (8 FP4 codes of rows g / g+8, k 2t..2t+1, 2t+8..2t+9) -> the m16n8k16 / RS-wgmma A fragment
__device__ __forceinline__ void dequant_word(unsigned (&a)[4], unsigned w, unsigned mg, unsigned mg8) {
  const unsigned lo = fp4_lo_e4m3(w), hi = fp4_hi_e4m3(w);
  unsigned l0, l1, h0, h1;
  cvt_e4m3x4(lo, l0, l1);
  cvt_e4m3x4(hi, h0, h1);
  a[0] = hmul2(l0, mg);   // row g,   k 2t..2t+1
  a[2] = hmul2(l1, mg);   // row g,   k 2t+8..
  a[1] = hmul2(h0, mg8);  // row g+8, k 2t..
  a[3] = hmul2(h1, mg8);  // row g+8, k 2t+8..
}

// one k32 block of rows g (word wg) and g+8 (word wh): fragments of k16 step s=0 (even nibbles) and s=1 (odd)
__device__ __forceinline__ void dequant_pair(unsigned (&f0)[4], unsigned (&f1)[4], unsigned wg, unsigned wh, unsigned mg,
                                             unsigned mh) {
  unsigned g0, g1, h0, h1;
  cvt_e4m3x4(fp4_lo_e4m3(wg), g0, g1);  // phys 8t + {0,2} | 8t + {4,6}
  cvt_e4m3x4(fp4_lo_e4m3(wh), h0, h1);
  f0[0] = hmul2(g0, mg), f0[2] = hmul2(g1, mg), f0[1] = hmul2(h0, mh), f0[3] = hmul2(h1, mh);
  cvt_e4m3x4(fp4_hi_e4m3(wg), g0, g1);  // phys 8t + {1,3} | 8t + {5,7}
  cvt_e4m3x4(fp4_hi_e4m3(wh), h0, h1);
  f1[0] = hmul2(g0, mg), f1[2] = hmul2(g1, mg), f1[1] = hmul2(h0, mh), f1[3] = hmul2(h1, mh);
}
// ------------------------------------------------------------------------------------------------ small kernels
__global__ void k_prescale(const __nv_bfloat16* __restrict__ x, int* __restrict__ ptok, float* __restrict__ inv) {
  const __nv_bfloat16* r = x + (size_t)blockIdx.x * D;
  float m = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) m = fmaxf(m, fabsf(__bfloat162float(r[i])));
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  __shared__ float wm[32];
  if (threadIdx.x % 32 == 0) wm[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < blockDim.x / 32; ++i) m = fmaxf(m, wm[i]);
    m = fmaxf(m, wm[0]);
    int p = 0;
    if (m > 0.f) {
      int e;
      frexpf(m, &e);
      p = 14 - e;
    }
    ptok[blockIdx.x] = p;
    inv[blockIdx.x] = exp2f((float)-p);
  }
}
// k_prescale2 (moe_v4): k_prescale with one 16-B load per thread (640 threads = one 5,120-element row); same p / inv
__global__ void __launch_bounds__(640) k_prescale2(const __nv_bfloat16* __restrict__ x, int* __restrict__ ptok,
                                                   float* __restrict__ inv) {
  const uint4 v = *((const uint4*)(x + (size_t)blockIdx.x * D) + threadIdx.x);
  const __nv_bfloat162* h = (const __nv_bfloat162*)&v;
  float m = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 f = __bfloat1622float2(h[k]);
    m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
  }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  __shared__ float wm[20];
  if (threadIdx.x % 32 == 0) wm[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    m = wm[0];
    for (int i = 1; i < 20; ++i) m = fmaxf(m, wm[i]);
    int p = 0;
    if (m > 0.f) {
      int e;
      frexpf(m, &e);
      p = 14 - e;
    }
    ptok[blockIdx.x] = p;
    inv[blockIdx.x] = exp2f((float)-p);
  }
}
// k_prescale16: per-token power of two p (as k_prescale) and the prescaled f16 copy x16 [T, D] (exact for elements
// >= 2^-28 of the token max) that GEMM1 gathers rows from with TMA (no per-expert token images).
__global__ void k_prescale16(const __nv_bfloat16* __restrict__ x, int* __restrict__ ptok, float* __restrict__ inv,
                             __half* __restrict__ x16) {
  const __nv_bfloat16* r = x + (size_t)blockIdx.x * D;
  __shared__ float wm[32];
  __shared__ float sc_s;
  float m = 0.f;
  for (int i = threadIdx.x * 8; i < D; i += blockDim.x * 8) {
    const uint4 v = *(const uint4*)(r + i);
    const __nv_bfloat162* h = (const __nv_bfloat162*)&v;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 f = __bfloat1622float2(h[k]);
      m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  if (threadIdx.x % 32 == 0) wm[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < blockDim.x / 32; ++i) m = fmaxf(m, wm[i]);
    m = fmaxf(m, wm[0]);
    int p = 0;
    if (m > 0.f) {
      int e;
      frexpf(m, &e);
      p = 14 - e;
    }
    ptok[blockIdx.x] = p;
    inv[blockIdx.x] = exp2f((float)-p);
    sc_s = exp2f((float)p);
  }
  __syncthreads();
  const float sc = sc_s;
  for (int i = threadIdx.x * 8; i < D; i += blockDim.x * 8) {
    const uint4 v = *(const uint4*)(r + i);
    const __nv_bfloat162* h = (const __nv_bfloat162*)&v;
    uint4 o;
    __half2* o2 = (__half2*)&o;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 f = __bfloat1622float2(h[k]);
      o2[k] = __floats2half2_rn(f.x * sc, f.y * sc);
    }
    *(uint4*)(x16 + (size_t)blockIdx.x * D + i) = o;
  }
}
__global__ void k_gather(const __nv_bfloat16* __restrict__ x, EC* __restrict__ ecs, const int* __restrict__ ptok,
                         char* __restrict__ img, const int* __restrict__ nec_dev) {
  // image [80 slabs][n_pad][128 B swizzled] f16 in the LOGICAL K order of every 32-block (kperm)
  if ((int)blockIdx.x >= *nec_dev) return;
  EC& ec = ecs[blockIdx.x];
  for (int n = blockIdx.y; n < ec.n_pad; n += gridDim.y) {
    const int tok = ec.tok[n], slab = threadIdx.x >> 3, cc = threadIdx.x & 7;
    if (threadIdx.x == 0) ec.inv[n] = tok >= 0 ? exp2f((float)-ptok[tok]) : 0.f;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (tok >= 0) {
      const float sc = exp2f((float)ptok[tok]);
      const int blk = cc >> 2, L0 = 8 * (cc & 3);
      const __nv_bfloat16* src = x + (size_t)tok * D + slab * 64 + blk * 32;
      __half h[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) h[i] = __float2half_rn(__bfloat162float(src[kperm(L0 + i)]) * sc);
      v = *(uint4*)h;
    }
    *(uint4*)(img + ec.img_off + (size_t)slab * ec.n_pad * 128 + swz(n, cc * 16)) = v;
  }
}

// k_gather2 (MSR_GATHER2=1, default): k_gather's thread mapping (thread = one 16-B output chunk, 8 threads per 128-B
// image row: coalesced stores), but the 64-B source block is read with 4 x 16-B vector loads (L1 hits for 3 of the 4
// threads sharing it) instead of 8 scalar loads, and the thread's 8 permuted elements are picked in registers (switch on
// its quarter c so every index is a compile-time constant). Same image bytes as k_gather.
#ifndef MSR_GATHER2
#define MSR_GATHER2 1
#endif
__device__ __forceinline__ float bf_at(const uint4 (&v)[4], int p) {  // p compile-time after unrolling
  const int w = p >> 1, q = w >> 2, c = w & 3;
  const unsigned u = c == 0 ? v[q].x : c == 1 ? v[q].y : c == 2 ? v[q].z : v[q].w;
  return __uint_as_float((p & 1) ? (u & 0xffff0000u) : (u << 16));
}
template <int C>
__device__ __forceinline__ uint4 emit8(const uint4 (&v)[4], float sc) {
  unsigned w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2 h = __halves2half2(__float2half_rn(bf_at(v, kperm(8 * C + 2 * i)) * sc),
                                     __float2half_rn(bf_at(v, kperm(8 * C + 2 * i + 1)) * sc));
    w[i] = *(const unsigned*)&h;
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}
__global__ void __launch_bounds__(640) k_gather2(const __nv_bfloat16* __restrict__ x, EC* __restrict__ ecs,
                                                 const int* __restrict__ ptok, char* __restrict__ img,
                                                 const int* __restrict__ nec_dev, int pstride = 1) {
  if ((int)blockIdx.x >= *nec_dev) return;
  EC& ec = ecs[blockIdx.x];
  const int slab = threadIdx.x >> 3, cc = threadIdx.x & 7, blk = cc >> 2, c = cc & 3;
  for (int n = blockIdx.y; n < ec.n_pad; n += gridDim.y) {
    const int tok = ec.tok[n];
    if (threadIdx.x == 0) ec.inv[n] = tok >= 0 ? exp2f((float)-ptok[(size_t)tok * pstride]) : 0.f;
    uint4 o = make_uint4(0, 0, 0, 0);
    if (tok >= 0) {
      const float sc = exp2f((float)ptok[(size_t)tok * pstride]);
      const uint4* src = (const uint4*)(x + (size_t)tok * D + slab * 64 + blk * 32);
      uint4 v[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) v[q] = __ldg(src + q);
      switch (c) {
        case 0: o = emit8<0>(v, sc); break;
        case 1: o = emit8<1>(v, sc); break;
        case 2: o = emit8<2>(v, sc); break;
        default: o = emit8<3>(v, sc); break;
      }
    }
    *(uint4*)(img + ec.img_off + (size_t)slab * ec.n_pad * 128 + swz(n, cc * 16)) = o;
  }
}

// k_gather5 (prepr5; route2's EC numbering: slot base pre = img_off / 10240 non-decreasing in the EC index):
// k_gather2's thread mapping and arithmetic (640 threads = one slot row: 80 slabs x 8 chunks, emit8; same image bytes,
// same ec.inv), but PERSISTENT: grid = MSR_G5B x SMs blocks over the flat slot list [0, ptot) (ptot = counts[3] / 10240),
// MSR_G5U slots per round in flight (their tok / ptok / source loads issued together), slot -> (EC, n) by a binary search
// over the EC slot bases staged in shared memory. Why: k_gather2 runs ~max_ec x 8 short blocks (~1 slot each): ~8 waves of
// launch + one dependent load round trip each, 38 us for a ~30 MB image write (floor ~8 us at ~4 TB/s).
#ifndef MSR_G5U
#define MSR_G5U 4
#endif
#ifndef MSR_G5B
#define MSR_G5B 1
#endif
__global__ void __launch_bounds__(640, MSR_G5B) k_gather5(const __nv_bfloat16* __restrict__ x, EC* __restrict__ ecs,
                                                          const int* __restrict__ ptok, char* __restrict__ img,
                                                          const int* __restrict__ counts, int pstride) {
  constexpr int U = MSR_G5U, MAXE = 512, MAXJ = 64;
  __shared__ int sbase[MAXE + 1];
  __shared__ long long soff[MAXJ];  // per slot of this batch: image byte offset of slab 0, n_pad * 128 (slab stride)
  __shared__ int sstr[MAXJ], stok[MAXJ], snn[MAXJ];
  __shared__ float ssc[MAXJ];
  const int nec = counts[0], S = counts[3] / (80 * 128), G = gridDim.x;
  for (int i = threadIdx.x; i < nec; i += blockDim.x) sbase[i] = (int)(ecs[i].img_off / (80 * 128));
  if (threadIdx.x == 0) sbase[nec] = S;
  __syncthreads();
  const int slab = threadIdx.x >> 3, cc = threadIdx.x & 7, blk = cc >> 2, c = cc & 3;
  // this block's slots: j = blockIdx.x + G * i, i = 0, 1, ...; batches of MAXJ
  const int nmine = blockIdx.x < S ? (S - 1 - blockIdx.x) / G + 1 : 0;
  for (int b0 = 0; b0 < nmine; b0 += MAXJ) {
    const int nb = min(MAXJ, nmine - b0);
    if (threadIdx.x < nb) {  // resolve slot -> (EC, n, token, scale) for the whole batch in parallel; write ec.inv
      const int j = blockIdx.x + G * (b0 + threadIdx.x);
      int lo = 0, hi = nec - 1;
      while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (sbase[mid] <= j) lo = mid; else hi = mid - 1;
      }
      const int n = j - sbase[lo], np = sbase[lo + 1] - sbase[lo];
      const int tok = ecs[lo].tok[n];
      float sc = 0.f, iv = 0.f;
      if (tok >= 0) {
        const int p = ptok[(size_t)tok * pstride];
        sc = exp2f((float)p), iv = exp2f((float)-p);
      }
      ecs[lo].inv[n] = iv;
      soff[threadIdx.x] = (long long)sbase[lo] * (80 * 128);
      sstr[threadIdx.x] = np * 128, stok[threadIdx.x] = tok, snn[threadIdx.x] = n, ssc[threadIdx.x] = sc;
    }
    __syncthreads();
    for (int i0 = 0; i0 < nb; i0 += U) {
      uint4 v[U][4];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int i = i0 + u, tok = i < nb ? stok[i] : -1;
        if (tok >= 0) {
          const uint4* src = (const uint4*)(x + (size_t)tok * D + slab * 64 + blk * 32);
#pragma unroll
          for (int q = 0; q < 4; ++q) v[u][q] = __ldg(src + q);
        }
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int i = i0 + u;
        if (i >= nb) break;
        uint4 o = make_uint4(0, 0, 0, 0);
        if (stok[i] >= 0) {
          const float sc = ssc[i];
          switch (c) {
            case 0: o = emit8<0>(v[u], sc); break;
            case 1: o = emit8<1>(v[u], sc); break;
            case 2: o = emit8<2>(v[u], sc); break;
            default: o = emit8<3>(v[u], sc); break;
          }
        }
        *(uint4*)(img + soff[i] + (size_t)slab * sstr[i] + swz(snn[i], cc * 16)) = o;
      }
    }
    __syncthreads();  // the batch's SMEM slot table is reused
  }
}

#ifndef MSR_GATHER3
#define MSR_GATHER3 0  // 1: prescale + gather fused (bit-identical, but 37.6 / 46.3 us vs 7.0 + 23.7: each warp store hits 16 lines
                       //    with 2 x 16 B, where k_gather2 writes whole 128-B lines; off)
#endif
#ifndef MSR_G3Y
#define MSR_G3Y 2  // k_gather3 grid.y: blocks per EC (each G3G tokens per pass)
#endif
// k_gather3 = k_prescale + k_gather2 in one kernel. Block (EC, y) handles slots n = y, y + gridDim.y, ...; for each, its 160
// threads own the token row's 160 k32 blocks (64 B loaded once), reduce max |x| over the row (-> p, the per-token power
// of two, as k_prescale), then each thread writes its block's 4 logical 16-B chunks. Writes ptok / inv per token (any of
// the token's ECs; identical values) and ec.inv. Same image bytes as k_prescale + k_gather2.
// k_gather4 (MSR_GATHER3 = 2): k_gather2's layout (640 threads = 80 slabs x 8 chunks of one token, whole 128-B line
// stores) with the per-token max folded in (block max over the row's 640 x 32 values, 4x redundant like the loads:
// coalesced), so no k_prescale. Same p / inv / image bytes.
__global__ void __launch_bounds__(640) k_gather4(const __nv_bfloat16* __restrict__ x, EC* __restrict__ ecs,
                                                 int* __restrict__ ptok, float* __restrict__ inv, char* __restrict__ img,
                                                 const int* __restrict__ nec_dev) {
  if ((int)blockIdx.x >= *nec_dev) return;
  EC& ec = ecs[blockIdx.x];
  const int slab = threadIdx.x >> 3, cc = threadIdx.x & 7, blk = cc >> 2, c = cc & 3, w = threadIdx.x >> 5,
            lane = threadIdx.x & 31;
  __shared__ float wm[2][20];
  int it = 0;
  for (int n = blockIdx.y; n < ec.n_pad; n += gridDim.y, ++it) {
    const int tok = ec.tok[n];  // block-uniform
    uint4 o = make_uint4(0, 0, 0, 0);
    if (tok >= 0) {
      const uint4* src = (const uint4*)(x + (size_t)tok * D + slab * 64 + blk * 32);
      uint4 v[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) v[q] = __ldg(src + q);
      float m = 0.f;
#pragma unroll
      for (int pp = 0; pp < 32; ++pp) m = fmaxf(m, fabsf(bf_at(v, pp)));
#pragma unroll
      for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, off));
      if (lane == 0) wm[it & 1][w] = m;
      __syncthreads();
      m = 0.f;
#pragma unroll
      for (int i = 0; i < 20; ++i) m = fmaxf(m, wm[it & 1][i]);
      int p = 0;
      if (m > 0.f) {
        int e;
        frexpf(m, &e);
        p = 14 - e;
      }
      if (threadIdx.x == 0) {
        const float iv = exp2f((float)-p);
        ec.inv[n] = iv;
        ptok[tok] = p;
        inv[tok] = iv;
      }
      const float sc = exp2f((float)p);
      switch (c) {
        case 0: o = emit8<0>(v, sc); break;
        case 1: o = emit8<1>(v, sc); break;
        case 2: o = emit8<2>(v, sc); break;
        default: o = emit8<3>(v, sc); break;
      }
    } else if (threadIdx.x == 0) {
      ec.inv[n] = 0.f;
    }
    *(uint4*)(img + ec.img_off + (size_t)slab * ec.n_pad * 128 + swz(n, cc * 16)) = o;
  }
}

constexpr int G3G = 4;  // k_gather3: tokens per block in parallel (one 160-thread group each)
__global__ void __launch_bounds__(160 * G3G) k_gather3(const __nv_bfloat16* __restrict__ x, EC* __restrict__ ecs,
                                                       int* __restrict__ ptok, float* __restrict__ inv,
                                                       char* __restrict__ img, const int* __restrict__ nec_dev) {
  if ((int)blockIdx.x >= *nec_dev) return;
  EC& ec = ecs[blockIdx.x];
  const int grp = threadIdx.x / 160, t = threadIdx.x % 160, slab = t >> 1, blk = t & 1, w = t >> 5, lane = t & 31;
  __shared__ float wm[2][G3G][5];
  const int np = ec.n_pad, step = gridDim.y * G3G;
  int it = 0;
  for (int n = blockIdx.y * G3G + grp; n < np; n += step, ++it) {
    const int tok = ec.tok[n];  // group-uniform
    char* dst = img + ec.img_off + (size_t)slab * np * 128;
    if (tok < 0) {
#pragma unroll
      for (int c = 0; c < 4; ++c) *(uint4*)(dst + swz(n, (blk * 4 + c) * 16)) = make_uint4(0, 0, 0, 0);
      if (t == 0) ec.inv[n] = 0.f;
      continue;
    }
    const uint4* src = (const uint4*)(x + (size_t)tok * D + t * 32);
    uint4 v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = __ldg(src + q);
    float m = 0.f;
#pragma unroll
    for (int pp = 0; pp < 32; ++pp) m = fmaxf(m, fabsf(bf_at(v, pp)));
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
    float* wmg = wm[it & 1][grp];
    if (lane == 0) wmg[w] = m;
    asm volatile("bar.sync %0, 160;" ::"r"(1 + grp) : "memory");
    m = fmaxf(fmaxf(fmaxf(wmg[0], wmg[1]), fmaxf(wmg[2], wmg[3])), wmg[4]);
    int p = 0;
    if (m > 0.f) {
      int e;
      frexpf(m, &e);
      p = 14 - e;
    }
    if (t == 0) {
      const float iv = exp2f((float)-p);
      ec.inv[n] = iv;
      ptok[tok] = p;
      inv[tok] = iv;
    }
    const float sc = exp2f((float)p);
    *(uint4*)(dst + swz(n, (blk * 4 + 0) * 16)) = emit8<0>(v, sc);
    *(uint4*)(dst + swz(n, (blk * 4 + 1) * 16)) = emit8<1>(v, sc);
    *(uint4*)(dst + swz(n, (blk * 4 + 2) * 16)) = emit8<2>(v, sc);
    *(uint4*)(dst + swz(n, (blk * 4 + 3) * 16)) = emit8<3>(v, sc);
  }
}

// ------------------------------------------------------------------------------------------------ GEMM kernels
template <int MODE>
struct Args {
  const unsigned char* s2;  // GEMM2: raw s2 [E, H, I/32] (bulk-copied per tile)
  const unsigned char* w2;  // GEMM2: raw w2 [E, H, I/2] (MSR_G2BULK: one bulk copy per whole-K tile)
  const EC* ecs;
  const Tile* tiles;
  const char* bimg;  // GEMM1: x image; GEMM2: H image (both in the logical K order)
  char* himg;        // GEMM1 output
  const int* nt_dev;
  __nv_bfloat16* yout;  // GEMM2: bf16 [slots][D]
  alignas(64) CUtensorMap tmw;  // GEMM1: w13 [E 2I rows, H/2 B] u8 box 128 x 64, SW128; GEMM2: w2 [E H rows, I/2 B] u16
  alignas(64) CUtensorMap tms;  // GEMM1: s13 [E 2I rows, H/32 B] u8 box 16 x 64
  alignas(64) CUtensorMap tmw2;  // paired tiles: GEMM2 w2 u16 box 48 x 256 rows; GEMM1 w13 box 128 B x 128 rows (SW128)
  alignas(64) CUtensorMap tms2;  // GEMM1 paired tiles: s13 box 16 B x 128 rows
  const int* ec8;  // gemm1r8: EC indices with n_pad 8 (route2x)
  int skip8;       // gemm1rs: the tile-stream GEMM1 skips n_pad-8 tiles (gemm1r8 runs them)
};

template <int MODE, int NT, bool PAIR = false>
__device__ __forceinline__ void run_tile(const Args<MODE>& a, char* ring, uint64_t* full, uint64_t* empty, long& it,
                                         int tile, int c, int t, long long* tr = nullptr, int hq = -1, int* done1 = nullptr,
                                         int hec = 0, uint32_t hdbar = 0) {
  using C = Cfg<MODE>;
  using Ly = Lay<MODE>;
  constexpr int N = NT * 8;
  constexpr int PS = PAIR ? 2 : 1;  // GEMM2 paired tile: rows R and R + 128 of a 256-row tile share the B fragments
  constexpr int CH = PAIR ? (MODE == 1 ? MSR_G1PAIR_CH : (MS4_G2P ? C::KB : C::CHB)) : (N <= C::NSMALL ? C::CHS : C::CHB);
  constexpr int OFFA = (MODE == 2 && PAIR && MS4_G2P) ? Ly::P_OFF_A : Ly::OFF_A;
  constexpr int OFFS = (MODE == 2 && PAIR && MS4_G2P) ? Ly::P_OFF_S : Ly::OFF_S;
  static_assert(!PAIR || N <= MS3_MMA_MAXN, "paired tiles: mma.sync path only");
  constexpr int NCH = C::KB / CH;
  constexpr bool MMA = N <= MS3_MMA_MAXN;
  static_assert(C::KB % CH == 0, "stages must tile K");
  const int w = t / 32, lane = t % 32, g = lane / 4, tq = lane % 4, R = 64 * c + 16 * w + g;
  // A rows of this warp: GEMM1 gate / up rows 32c + 8w + (0..7) of the tile's 64 intermediates; GEMM2 rows 64c + 16w + (0..15)
  const int r0 = MODE == 1 ? 32 * c + 8 * w : 64 * c + 16 * w;
  const int lr = lane & 7, mi = lane >> 3;  // ldmatrix: this lane's row within matrix mi
  int etok[NT][2];
  float ewt[NT][2], einv[NT][2];
  long long eh_off;
  int eslot, enr;
  {
    const EC* e0 = (const EC*)(ring + (size_t)(it % SS<MODE>) * Ly::STAGE + Ly::OFF_H + 128);
#pragma unroll
    for (int nb = 0; nb < NT; ++nb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nb + 2 * tq + hh;
        etok[nb][hh] = e0->tok[n];
        ewt[nb][hh] = e0->wt[n];
        einv[nb][hh] = e0->inv[n];
      }
    eh_off = e0->h_off, eslot = e0->pad, enr = e0->n_real;
  }
  float acc[PS][2][N / 2];
  unsigned abl_sink = 0;  // MS3_ABL: keeps ablated loads / conversions live
#pragma unroll
  for (int ps = 0; ps < PS; ++ps)
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
      for (int i = 0; i < N / 2; ++i) acc[ps][q][i] = 0.f;
  for (int ch = 0; ch < NCH; ++ch) {
    const int s = it % SS<MODE>;
    if (ch > 0) {
      const long long q0 = MS4_TRACE ? clock64() : 0;
      mbar_wait(smem_u32(full + s), (it / SS<MODE>) & 1);
      if (MS4_TRACE && tr) tr[0] += clock64() - q0;
    }
    char* st = ring + (size_t)s * Ly::STAGE;
    const uint32_t aB = smem_u32(st), aA = aB + OFFA;
    // scale bytes of this thread's two rows for the stage, biased: GEMM1 16 B per row (gate / up), GEMM2 2 per k64 step
    unsigned sg[PS][4], sh[PS][4];
    if (MODE == 1) {  // scale boxes [64 PS rows][16 B]: gate at OFF_S, up at OFF_S + 1024 PS
#pragma unroll
      for (int ps = 0; ps < PS; ++ps) {
        const int rr = r0 + g + 64 * ps;
        const uint4 vg = *(const uint4*)(st + Ly::OFF_S + rr * 16), vh = *(const uint4*)(st + Ly::OFF_S + 1024 * PS + rr * 16);
        sg[ps][0] = bias4(vg.x), sg[ps][1] = bias4(vg.y), sg[ps][2] = bias4(vg.z), sg[ps][3] = bias4(vg.w);
        sh[ps][0] = bias4(vh.x), sh[ps][1] = bias4(vh.y), sh[ps][2] = bias4(vh.z), sh[ps][3] = bias4(vh.w);
      }
    }
#pragma unroll
    for (int kb = 0; kb < CH; ++kb) {
      // ---- A words: w[0] row g blk 0, w[1] row g+8 blk 0, w[2] row g blk 1, w[3] row g+8 blk 1 (PAIR: row set ps adds 128)
      unsigned wv[PS][4];
      unsigned mg[PS][2], mh[PS][2];
#pragma unroll
      for (int ps = 0; ps < PS; ++ps) {
        if (MODE == 1) {
          const int h = kb >> 2, cbase = 2 * (kb & 3), m = mi & 1, chunk = cbase + (mi >> 1), r = r0 + lr;
          ldsm_x4a(wv[ps], aA + (unsigned)(((h * 2 + m) * (64 * PS) + r + 64 * ps) * 128 + ((chunk ^ (r & 7)) << 4)));
        } else {
          const int r = r0 + 8 * (mi & 1) + lr + 128 * ps, chunk = 2 * (kb % 3) + (mi >> 1);  // [CH/3 boxes][rows][96 B]
          if (MSR_G2BULK && CH == C::KB)  // [128 rows][288 B]: row byte 32 kb + 16 (mi >> 1) (2-way bank conflicts)
            ldsm_x4a(wv[ps], aA + (unsigned)(r * (IR / 2) + 32 * kb + 16 * (mi >> 1)));
          else
            ldsm_x4a(wv[ps], aA + (unsigned)((kb / 3) * (12288 * PS) + r * 96 + chunk * 16));
        }
        // ---- multipliers (f16x2 2^(s-121)) of rows g, g+8 for the two k32 blocks of this k64 step
        if (MODE == 1) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {  // byte soff + 2 kb + j of the 16-B box; soff is 0 (CH 8) or 8 (odd stage, CH 4)
            const int bi = 2 * kb + j;   // compile-time
            const unsigned wg = (CH == 4 && (ch & 1)) ? sg[ps][2 + (bi >> 2)] : sg[ps][bi >> 2];
            const unsigned wh = (CH == 4 && (ch & 1)) ? sh[ps][2 + (bi >> 2)] : sh[ps][bi >> 2];
            mg[ps][j] = __byte_perm(wg, 0, 0x0404 + 0x1010 * (bi & 3));
            mh[ps][j] = __byte_perm(wh, 0, 0x0404 + 0x1010 * (bi & 3));
          }
        } else {
          const int blk0 = 2 * (ch * CH + kb), RR = R + 128 * ps;
          const unsigned short bgs = *(const unsigned short*)(st + OFFS + RR * 18 + blk0);
          const unsigned short bhs = *(const unsigned short*)(st + OFFS + (RR + 8) * 18 + blk0);
          const unsigned u = bias4((unsigned)bgs | ((unsigned)bhs << 16));  // bytes: g blk0, g blk1, g+8 blk0, g+8 blk1
          mg[ps][0] = __byte_perm(u, 0, 0x0404), mg[ps][1] = __byte_perm(u, 0, 0x1414);
          mh[ps][0] = __byte_perm(u, 0, 0x2424), mh[ps][1] = __byte_perm(u, 0, 0x3434);
        }
      }
      if (MS3_ABL == 1 || MS3_ABL >= 6) {
        abl_sink ^= wv[0][0] ^ wv[0][1] ^ wv[0][2] ^ wv[0][3] ^ mg[0][0] ^ mh[0][1];
        continue;
      }
      if constexpr (MMA) {
        unsigned bf[NT][2][4];
        const uint32_t slab = aB + kb * N * 128;
#pragma unroll
        for (int nb = 0; nb < NT; ++nb)
#pragma unroll
          for (int kk = 0; kk < 2; ++kk)
            ldsm_x4(bf[nb][kk], slab + swz(nb * 8 + (lane & 7), (kk * 4 + (lane >> 3)) * 16));
        if (MS3_ABL == 3) {
#pragma unroll
          for (int nb = 0; nb < NT; ++nb)
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) abl_sink ^= bf[nb][kk][0] ^ bf[nb][kk][1] ^ bf[nb][kk][2] ^ bf[nb][kk][3];
          abl_sink ^= wv[0][0] ^ wv[0][1] ^ wv[0][2] ^ wv[0][3] ^ mg[0][0] ^ mh[0][1];
          continue;
        }
#pragma unroll
        for (int ps = 0; ps < PS; ++ps)  // PAIR: both row sets use the same B fragments
#pragma unroll
          for (int j = 0; j < 2; ++j) {  // k32 block j: k16 steps 2j (even nibbles), 2j+1 (odd)
            unsigned f0[4], f1[4];
            if (MS3_ABL == 5) {  // MMA work without the dequant (garbage values; timing / energy only)
#pragma unroll
              for (int q = 0; q < 4; ++q) f0[q] = wv[ps][(2 * j + q) & 3] ^ mg[ps][j], f1[q] = wv[ps][(2 * j + 1 + q) & 3] ^ mh[ps][j];
            } else {
              dequant_pair(f0, f1, wv[ps][2 * j], wv[ps][2 * j + 1], mg[ps][j], mh[ps][j]);
            }
            if (MS3_ABL == 4) {
#pragma unroll
              for (int q = 0; q < 4; ++q) abl_sink ^= f0[q] ^ f1[q];
#pragma unroll
              for (int nb = 0; nb < NT; ++nb) abl_sink ^= bf[nb][j][0] ^ bf[nb][j][3];
              continue;
            }
#pragma unroll
            for (int nb = 0; nb < NT; ++nb) {
              mma16816(&acc[ps][0][nb * 4], f0, bf[nb][j][0], bf[nb][j][1]);
              mma16816(&acc[ps][1][nb * 4], f1, bf[nb][j][2], bf[nb][j][3]);
            }
          }
      } else {
        unsigned A[2][4][4];  // [set][k16 step][4]
        const int set = kb & 1;
        if (kb >= 2) wg_wait<1>();
#pragma unroll
        for (int j = 0; j < 2; ++j) dequant_pair(A[set][2 * j], A[set][2 * j + 1], wv[0][2 * j], wv[0][2 * j + 1], mg[0][j], mh[0][j]);
        wg_fence();
        const uint64_t db = desc_sw128(aB + kb * N * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) WgRS<N>::mma(acc[0][0], A[set][j], db + 2 * j, 1);
        wg_commit();
      }
    }
    if (!MMA && MS3_ABL != 1 && MS3_ABL < 6) wg_wait<0>();
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_u32(empty + s));
    ++it;
  }
  if (MMA) {
#pragma unroll
    for (int ps = 0; ps < PS; ++ps)
#pragma unroll
      for (int i = 0; i < N / 2; ++i) acc[ps][0][i] += acc[ps][1][i];
  }
  if (MS3_ABL && abl_sink == 0x9e3779b9u) acc[0][0][0] += 1.f;  // never true in practice; the compiler cannot know
  if (MS4_TRACE && tr) tr[1] = clock64();
  // ---- epilogue. acc[0][4 nb + {0,1,2,3}] = (row g, col 8nb+2tq), (g, +1), (g+8, 2tq), (g+8, +1)
  struct {
    int q;
  } tl{MS4_HQ ? hq : a.tiles[tile].q};
  if (MODE == 1) {  // h for intermediate i = 64 (q + ps) + il -> GEMM2's K, written at its logical position in slab q + ps
    const int il = 32 * c + 8 * w + g, pos = 32 * (il >> 5) + kinv(il & 31);
#pragma unroll
    for (int ps = 0; ps < PS; ++ps) {
    char* hs = a.himg + eh_off + (size_t)(tl.q + ps) * N * 128;
#pragma unroll
    for (int nb = 0; nb < NT; ++nb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nb + 2 * tq + hh;
        const int tok = etok[nb][hh];
        float h = 0.f;
        if (tok >= 0) {
          const float iv = einv[nb][hh];
          float gt = fminf(acc[ps][0][4 * nb + hh] * iv, 10.f);
          float up = fminf(fmaxf(acc[ps][0][4 * nb + 2 + hh] * iv, -10.f), 10.f);
          h = gt / (1.f + expf(-gt)) * up * ewt[nb][hh] * 256.f;
        }
        *(__half*)(hs + swz(n, pos * 2)) = __float2half_rn(h);
      }
    }
    if (hdbar) {  // fused kernel (MS4_FSIG 1): CTA-local release; the producer lane does the GPU-scope fence + count
      __syncwarp();
      if (lane == 0) mbar_arrive(hdbar);
    } else if (done1) {  // fused kernel v2: each warp releases its own H stores of this tile (8 warps x 9 tiles per EC = 72)
      __syncwarp();
      if (lane == 0) {
        __threadfence();
        atomicAdd(done1 + hec, PS);
      }
    }
  } else if (MS4_YWARP && (!PAIR || MS4_G2P)) {  // per-warp staging [ps][token][16 rows] bf16 (32 B per token), no CTA barrier
    static_assert(PS * N <= NMAX, "warp staging");
    __nv_bfloat16* stg = (__nv_bfloat16*)(ring + SS<MODE> * Ly::STAGE + 1024) + (size_t)(4 * c + w) * 16 * NMAX;
    __syncwarp();  // this warp's previous write-out has read stg
#pragma unroll
    for (int ps = 0; ps < PS; ++ps)
#pragma unroll
      for (int nb = 0; nb < NT; ++nb)
#pragma unroll
        for (int rs = 0; rs < 2; ++rs)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            stg[(ps * N + 8 * nb + 2 * tq + j) * 16 + g + 8 * rs] = __float2bfloat16_rn(acc[ps][0][4 * nb + 2 * rs + j] * (1.f / 256.f));
    __syncwarp();
#pragma unroll
    for (int k = lane; k < 2 * PS * N; k += 32) {  // chunk k: row set k / 2N, token (k / 2) % N, rows 8 (k & 1) .. + 7
      const int n = (k >> 1) % N, ps = (k >> 1) / N;
      if (MS3_ABL != 7 && n < enr)
        *(uint4*)(a.yout + (size_t)(eslot + n) * D + 128 * tl.q + 128 * ps + 64 * c + 16 * w + 8 * (k & 1)) =
            *(const uint4*)(stg + k * 8);
    }
  } else if (MS3_YSTG) {  // bf16 per (token, slot) staged as [token][128 rows] in SMEM, then 16-B coalesced stores
    __nv_bfloat16* stg = (__nv_bfloat16*)(ring + SS<MODE> * Ly::STAGE + 1024);
    asm volatile("bar.sync 1, 256;" ::: "memory");  // the previous tile's write-out has read stg
    constexpr int RW = 128 * PS;  // staged rows per token
#pragma unroll
    for (int ps = 0; ps < PS; ++ps)
#pragma unroll
      for (int nb = 0; nb < NT; ++nb)
#pragma unroll
        for (int rs = 0; rs < 2; ++rs)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            stg[(8 * nb + 2 * tq + j) * RW + 128 * ps + R + 8 * rs] =
                __float2bfloat16_rn(acc[ps][0][4 * nb + 2 * rs + j] * (1.f / 256.f));
    asm volatile("bar.sync 1, 256;" ::: "memory");
    constexpr int CPT = RW / 8, TPP = 256 / CPT;  // 16-B chunks per token, tokens per pass
    const int ct = 128 * c + t, ck = ct % CPT;
    for (int n = ct / CPT; n < N; n += TPP)
      if (MS3_ABL != 7 && n < enr)
        *(uint4*)(a.yout + (size_t)(eslot + n) * D + 128 * tl.q + 8 * ck) = *(const uint4*)(stg + n * RW + 8 * ck);
  } else {  // bf16 per (token, slot), lanes g / g^1 swap one value -> 4-byte row-pair stores
    const bool odd = g & 1;
    const int rb = 128 * tl.q + (R & ~1);
#pragma unroll
    for (int nb = 0; nb < NT; ++nb)
#pragma unroll
      for (int rs = 0; rs < 2; ++rs) {
        const float v0 = acc[0][0][4 * nb + 2 * rs], v1 = acc[0][0][4 * nb + 2 * rs + 1];
        const float got = __shfl_xor_sync(0xffffffffu, odd ? v0 : v1, 4);
        const int n = 8 * nb + 2 * tq + (odd ? 1 : 0);
        if (MS3_ABL != 7 && etok[nb][odd ? 1 : 0] >= 0)
          *(__nv_bfloat162*)(a.yout + (size_t)(eslot + n) * D + rb + 8 * rs) =
              __floats2bfloat162_rn((odd ? got : v0) * (1.f / 256.f), (odd ? v1 : got) * (1.f / 256.f));
      }
  }
  (void)enr;
}

// one work item (all its stages) or the end marker: the k_gemm / k_fused producer body
struct NoPoll {
  __device__ __forceinline__ void operator()(bool) const {}
};
template <int MODE, class Poll = NoPoll>
__device__ __forceinline__ void produce_item(const Args<MODE>& a, char* ring, uint64_t* full, uint64_t* empty, long& it,
                                             int tile, Tile tl, bool end, uint64_t pw, uint64_t pb, int endv = -1,
                                             const int* dep = nullptr, int dep_target = 0, Poll poll = Poll{}) {
  using C = Cfg<MODE>;
  using Ly = Lay<MODE>;
      const int npad = end ? 0 : tl.npad;
  if (!end && (npad < 8 || npad > NMAX || (npad & 7))) __trap();
  const bool g2p = MS4_G2P && MODE == 2 && !end && (tl.q & 0x100);
  const bool g1p = MS4_G1P && MODE == 1 && !end && (tl.q & 0x100);
  if (g2p || g1p) tl.q &= 0xff;
  const bool pair = !end && ((MODE == 2 && MSR_G2PAIR && npad <= MS3_MMA_MAXN) ||
                         (MODE == 1 && MSR_G1PAIR && npad <= 16 && tl.q < 8)) || g2p || g1p;
  if (pair && !g2p && !g1p && (tl.q & 1)) return;  // the even tile of the pair covers these rows
  const int CH = pair ? (MODE == 1 ? MSR_G1PAIR_CH : (g2p ? C::KB : C::CHB)) : (npad <= C::NSMALL ? C::CHS : C::CHB);
  const int nch = end ? 1 : C::KB / CH;
  for (int ch = 0; ch < nch; ++ch, ++it) {
    const int s = it % SS<MODE>;
    poll(false);
    if (it >= SS<MODE>) mbar_wait(smem_u32(empty + s), ((it / SS<MODE>) + 1) & 1);
    char* st = ring + (size_t)s * Ly::STAGE;
    const uint32_t fb = smem_u32(full + s), dst = smem_u32(st);
    Hdr* h = (Hdr*)(st + Ly::OFF_H);
    h->tile = end ? endv : tile;
    h->ch = ch;
    h->npad = npad;
    h->nch = nch;
    h->pair = pair ? 1 : 0;
    h->q = end ? 0 : tl.q;
    h->ec = end ? 0 : tl.ec;
    if (end) {
      mbar_arrive(fb);
      break;
    }
    const unsigned bA = MS4_NOW ? 0u : g2p ? 256u * (IR / 2) : pair ? (MODE == 1 ? (unsigned)CH * 8192u : 256u * 96u) : CH * 4096,
               bS = MS4_NOW ? 0u : (MODE == 1 && MS3_ABL >= 8) ? 0u : MODE == 1 ? (pair ? 4096u : 2048u) : (pair ? 4608u : 2304u),
               bB = (MS3_ABL >= 6 || MS4_NOIMG) ? 0u : CH * npad * 128;
    const unsigned bE = ch == 0 ? (unsigned)sizeof(EC) : 0u;
    if (MS4_NOSTREAM) {  // timing only: consumer-bound time (stale stage data; only the EC record is copied)
      mbar_expect_tx(fb, bE);
      if (bE) bulk(dst + Ly::OFF_H + 128, a.ecs + tl.ec, bE, fb, pb);
      continue;
    }
    mbar_expect_tx(fb, bA + bS + bB + bE);
    if (bE) bulk(dst + Ly::OFF_H + 128, a.ecs + tl.ec, bE, fb, pb);
    if (MS4_NOW) {
    } else if (MODE == 1) {  // gate rows e 2I + 64 q .., up rows e 2I + I + 64 q ..; K bytes [ch CH 32, + CH 32)
      const int rg = tl.e * 1152 + 64 * tl.q, ru = rg + 576;
      const CUtensorMap* tw = pair ? &a.tmw2 : &a.tmw;  // pair: 128-row boxes (blocks q, q + 1)
      const unsigned bx = pair ? 16384u : 8192u;
      if (MS3_ABL == 9) {  // timing only: the same bytes as contiguous bulk copies (gate / up blocks, CH 2 KB each)
    const unsigned hb = (unsigned)CH * 2048u;
    bulk(dst + Ly::OFF_A, a.w2 + (size_t)rg * (D / 2) + (size_t)ch * hb, hb, fb, pw);
    bulk(dst + Ly::OFF_A + hb, a.w2 + (size_t)ru * (D / 2) + (size_t)ch * hb, hb, fb, pw);
      } else {
    for (int hh = 0; hh < CH / 4; ++hh) {
      const int c0 = ch * CH * 32 + hh * 128;
      tma2d(dst + Ly::OFF_A + (unsigned)((hh * 2 + 0) * bx), tw, c0, rg, fb, pw);
      tma2d(dst + Ly::OFF_A + (unsigned)((hh * 2 + 1) * bx), tw, c0, ru, fb, pw);
    }
      }
      if (MS3_ABL < 8) {
    const int cs = (ch * CH * 2) & ~15;
    const uint64_t ps_ = MS4_SPOL == 1 ? pol_normal() : MS4_SPOL == 2 ? pb : pw;
    tma2d(dst + Ly::OFF_S, pair ? &a.tms2 : &a.tms, cs, rg, fb, ps_);
    tma2d(dst + Ly::OFF_S + (pair ? 2048 : 1024), pair ? &a.tms2 : &a.tms, cs, ru, fb, ps_);
      }
    } else {  // 128 output rows e H + 128 rt ..; K u16 [ch CH 16, + CH 16); s2 rows of the tile in one bulk copy
      const int rr = tl.e * D + 128 * tl.q;
      if (g2p) {  // 256 contiguous rows + their scales, one bulk copy each
    bulk(dst + Ly::P_OFF_A, a.w2 + (size_t)rr * (IR / 2), 256 * (IR / 2), fb, pw);
    bulk(dst + Ly::P_OFF_S, a.s2 + (size_t)rr * 18, 256 * 18, fb, pw);
      } else if (pair)
    tma2d(dst + Ly::OFF_A, &a.tmw2, ch * CH * 16, rr, fb, pw);
      else if (MSR_G2BULK && CH == C::KB)
    bulk(dst + Ly::OFF_A, a.w2 + (size_t)rr * (IR / 2), 128 * (IR / 2), fb, pw);
      else
    for (int bx = 0; bx < CH / 3; ++bx) tma2d(dst + Ly::OFF_A + (unsigned)(bx * 12288), &a.tmw, (ch * CH + 3 * bx) * 16, rr, fb, pw);
      if (!g2p) bulk(dst + Ly::OFF_S, a.s2 + (size_t)rr * 18, pair ? 4608 : 2304, fb, pw);
    }
    if (dep && ch == 0) {  // fused kernel: this EC's GEMM1 tiles must have written H (release by their epilogues)
      const int* cnt = dep + tl.ec;
      for (int k = 0;; ++k) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(cnt) : "memory");
    if (v >= dep_target) break;
    if (k > 4) __nanosleep(256);
      }
      asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    if (bB) bulk(dst, a.bimg + tl.boff + (size_t)ch * CH * npad * 128, bB, fb, pb);
  }
}

// one work item on the consumer side; false at the end marker (k_gemm / k_fused consumer body)
template <int MODE>
__device__ __forceinline__ bool consume_item(const Args<MODE>& a, char* ring, uint64_t* full, uint64_t* empty, long& it,
                                             int c, int t, int* done1 = nullptr, uint32_t hdbase = 0, long* ntile = nullptr) {
  using Ly = Lay<MODE>;
  const int ct = 128 * c + t;
    const int s = it % SS<MODE>;
  const long long c0 = MS4_TRACE ? clock64() : 0;
  const unsigned long long g0 = MS4_TRACE ? gtimer() : 0;
  mbar_wait(smem_u32(full + s), (it / SS<MODE>) & 1);
  const long long c1 = MS4_TRACE ? clock64() : 0;
  const Hdr h = *(const Hdr*)(ring + (size_t)s * Ly::STAGE + Ly::OFF_H);
  if (h.tile < 0) return false;
  long long trv[2] = {0, 0};
  long long* tr = (MS4_TRACE && ct == 0) ? trv : nullptr;
  if constexpr (MODE == 2 && (MSR_G2PAIR || MS4_G2P)) {
    if (h.pair) {
      if constexpr (MS4_G2P != 0) {  // route2p pairs only n_pad 8 / 16
        if (h.npad == 8)
          run_tile<2, 1, true>(a, ring, full, empty, it, h.tile, c, t, tr, h.q);
        else
          run_tile<2, 2, true>(a, ring, full, empty, it, h.tile, c, t, tr, h.q);
        if (MS4_TRACE && ct == 0 && h.tile < TRACE_MAX) {
          unsigned smid;
          asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
          unsigned long long* q = g_trace[MODE - 1][h.tile];
          q[0] = g0, q[1] = c1 - c0, q[2] = trv[1] - c0, q[3] = clock64() - c0, q[4] = trv[0], q[5] = smid, q[6] = h.npad + 1000, q[7] = blockIdx.x;
        }
        return true;
      } else {
        switch (h.npad / 8) {
          case 1: run_tile<2, 1, true>(a, ring, full, empty, it, h.tile, c, t, nullptr, h.q); break;
          case 2: run_tile<2, 2, true>(a, ring, full, empty, it, h.tile, c, t, nullptr, h.q); break;
          case 3: run_tile<2, 3, true>(a, ring, full, empty, it, h.tile, c, t, nullptr, h.q); break;
          default: run_tile<2, 4, true>(a, ring, full, empty, it, h.tile, c, t, nullptr, h.q); break;
        }
      }
      return true;
    }
  }
  if constexpr (MODE == 1 && (MSR_G1PAIR || MS4_G1P)) {
    if (h.pair) {
      if (h.npad == 8)
        run_tile<1, 1, true>(a, ring, full, empty, it, h.tile, c, t, nullptr, h.q);
      else
        run_tile<1, 2, true>(a, ring, full, empty, it, h.tile, c, t, nullptr, h.q);
      return true;
    }
  }
  const uint32_t hdb = (MODE == 1 && hdbase) ? hdbase + 8u * (unsigned)((*ntile)++ % 4) : 0u;
  switch (h.npad / 8) {
    case 1: run_tile<MODE, 1>(a, ring, full, empty, it, h.tile, c, t, tr, h.q, done1, h.ec, hdb); break;
    case 2: run_tile<MODE, 2>(a, ring, full, empty, it, h.tile, c, t, tr, h.q, done1, h.ec, hdb); break;
    case 3: run_tile<MODE, 3>(a, ring, full, empty, it, h.tile, c, t, tr, h.q, done1, h.ec, hdb); break;
    case 4: run_tile<MODE, 4>(a, ring, full, empty, it, h.tile, c, t, tr, h.q, done1, h.ec, hdb); break;
    case 5: run_tile<MODE, 5>(a, ring, full, empty, it, h.tile, c, t, tr, h.q, done1, h.ec, hdb); break;
    case 6: run_tile<MODE, 6>(a, ring, full, empty, it, h.tile, c, t, tr, h.q, done1, h.ec, hdb); break;
    case 7: run_tile<MODE, 7>(a, ring, full, empty, it, h.tile, c, t, tr, h.q, done1, h.ec, hdb); break;
    default: run_tile<MODE, 8>(a, ring, full, empty, it, h.tile, c, t, tr, h.q, done1, h.ec, hdb); break;
  }
  if (MS4_TRACE && ct == 0 && h.tile < TRACE_MAX) {
    unsigned smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    unsigned long long* q = g_trace[MODE - 1][h.tile];
    q[0] = g0, q[1] = c1 - c0, q[2] = trv[1] - c0, q[3] = clock64() - c0, q[4] = trv[0], q[5] = smid, q[6] = h.npad, q[7] = blockIdx.x;
  }
  return true;
}

template <int MODE>
__global__ void __launch_bounds__(32 + 32 * NCW, 1) k_gemm(const __grid_constant__ Args<MODE> a) {
  using Ly = Lay<MODE>;
  extern __shared__ __align__(1024) char smem_raw[];
  char* ring = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  uint64_t* full = (uint64_t*)(ring + SS<MODE> * Ly::STAGE);
  uint64_t* empty = full + SS<MODE>;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int s = 0; s < SS<MODE>; ++s) mbar_init(smem_u32(full + s), 1), mbar_init(smem_u32(empty + s), NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (MS4_PDL) {
    if (MODE == 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    else asm volatile("griddepcontrol.wait;" ::: "memory");  // GEMM1 (and everything before it) complete and visible
  }
  __syncthreads();
  if (warp == NCW) {  // ---------------------------------------------------------------- producer (lane 0 of the last warp)
    if (lane != 0) return;
    const uint64_t pw = pol_first(), pb = pol_last();
    long it = 0;
    const int ntiles = *a.nt_dev;
    if (MS4_DYN && !a.skip8) {  // dynamic queue: counts[7] = nt_dev[7 - MODE]
      int* qc = const_cast<int*>(a.nt_dev) + (7 - MODE);
      int item = atomicAdd(qc, 1);
      for (;;) {
        const bool end = item >= ntiles;
        if (end && item == ntiles + (int)gridDim.x - 1) atomicExch(qc, 0);  // every CTA has made its last grab
        const Tile tl = end ? Tile{0, 0, 0, 0, 0} : a.tiles[item];
        const int next = end ? 0 : atomicAdd(qc, 1);
        produce_item<MODE>(a, ring, full, empty, it, item, tl, end, pw, pb);
        if (end) break;
        item = next;
      }
      return;
    }
    int tile = blockIdx.x;
    Tile tl = tile < ntiles ? a.tiles[tile] : Tile{0, 0, 0, 0, 0};
    for (;;) {
      const bool end = tile >= ntiles;
      const int ntile = tile + gridDim.x;
      const Tile nx = ntile < ntiles ? a.tiles[ntile] : Tile{0, 0, 0, 0, 0};
      if (!(a.skip8 && !end && tl.npad == 8))  // n_pad-8 tiles are run by the row-stream kernels when skip8
        produce_item<MODE>(a, ring, full, empty, it, tile, tl, end, pw, pb);
      if (end) break;
      tile = ntile;
      tl = nx;
    }
    return;
  }
  // ------------------------------------------------------------------ consumers (warps 0..7 = warpgroups 0, 1)
  const int ct = threadIdx.x, c = ct / 128, t = ct % 128;
  long it = 0;
  while (consume_item<MODE>(a, ring, full, empty, it, c, t)) {
  }
}

// ================================================================================================ fused GEMM1 -> GEMM2
// (moe_v4, MS4_FUSE) One persistent kernel: the GEMM1 items in their static snake schedule, then GEMM2 items grabbed from
// an atomic queue, so SMs that finish GEMM1 early start GEMM2 at once (GEMM1's CTA end times spread ~20 us). A GEMM1 tile
// adds 1 to done1[ec] after its H stores (256-thread barrier, fence, atomic: release); a GEMM2 item's producer waits for
// done1[ec] == 9 (acquire + proxy fence) before its H copy. done1 / the GEMM2 queue counter are zeroed by a memset node before
// each launch. Phase rings: GEMM1 3 x Lay<1>::STAGE, GEMM2 Lay<2> (2 x 96 KB + Y staging) in the same SMEM, with their
// own barriers; the producer drains the GEMM1 ring before the first GEMM2 copy. Same bytes as gemm1r + gemm2r.
struct FArgs {
  Args<1> a1;
  Args<2> a2;
  int* done1;  // [n_ec] per-EC GEMM1 tiles finished
  int* g2ctr;  // GEMM2 queue head
};
namespace fz {
constexpr int R1 = SS<1> * Lay<1>::STAGE, R2 = SS<2> * Lay<2>::STAGE + 1024 + Lay<2>::STG;
constexpr int RB = R1 > R2 ? R1 : R2;
constexpr int SMEM = 1024 + RB + 256;
static_assert(SMEM <= 227 * 1024, "fused SMEM");
}  // namespace fz
__global__ void __launch_bounds__(32 + 32 * NCW, 1) k_fused(const __grid_constant__ FArgs f) {
  static_assert(!MS4_G1P && !MSR_G1PAIR, "fused: unpaired GEMM1 only");
  extern __shared__ __align__(1024) char smem_raw[];
  char* ring = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  uint64_t* full1 = (uint64_t*)(ring + fz::RB);
  uint64_t* empty1 = full1 + SS<1>;
  uint64_t* full2 = empty1 + SS<1>;
  uint64_t* empty2 = full2 + SS<2>;
  uint64_t* hd = empty2 + SS<2>;  // [4] GEMM1 tile k's H stores done (all consumer warps), MS4_FSIG 1
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int s = 0; s < SS<1>; ++s) mbar_init(smem_u32(full1 + s), 1), mbar_init(smem_u32(empty1 + s), NCW);
    for (int s = 0; s < SS<2>; ++s) mbar_init(smem_u32(full2 + s), 1), mbar_init(smem_u32(empty2 + s), NCW);
    for (int s = 0; s < 4; ++s) mbar_init(smem_u32(hd + s), NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == NCW) {
    if (lane != 0) return;
    const uint64_t pw = pol_first(), pb = pol_last();
    {  // ---- GEMM1 phase: static snake schedule, then the phase-end marker (-2)
      long it = 0, nis = 0, nsig = 0;  // GEMM1 tiles issued / released to the GEMM2 phase by this CTA
      int ecr[4] = {0, 0, 0, 0};
      auto poll = [&](bool block) {  // MS4_FSIG 1: publish finished tiles (consumers arrived on hd[k % 4])
        if (!MS4_FSIG) return;
        while (nsig < nis) {
          const uint32_t b = smem_u32(hd + nsig % 4), par = (uint32_t)((nsig / 4) & 1);
          if (block) mbar_wait(b, par);
          else if (!mbar_try(b, par)) break;
          __threadfence();
          atomicAdd(f.done1 + ecr[nsig % 4], 1);
          ++nsig;
        }
      };
      const int ntiles = *f.a1.nt_dev;
      int tile = blockIdx.x;
      Tile tl = tile < ntiles ? f.a1.tiles[tile] : Tile{0, 0, 0, 0, 0};
      for (;;) {
        const bool end = tile >= ntiles;
        const int ntile = tile + gridDim.x;
        const Tile nx = ntile < ntiles ? f.a1.tiles[ntile] : Tile{0, 0, 0, 0, 0};
        if (!end) {
          while (MS4_FSIG && nis - nsig >= 4) poll(true);
          ecr[nis % 4] = tl.ec;
          ++nis;
        }
        produce_item<1>(f.a1, ring, full1, empty1, it, tile, tl, end, pw, pb, -2, nullptr, 0, poll);
        if (end) break;
        tile = ntile;
        tl = nx;
      }
      // drain: the consumers released every real GEMM1 stage (uses 0 .. it - 1; the marker sits at use it, no release)
      for (long u = it > SS<1> ? it - SS<1> : 0; u < it; ++u) mbar_wait(smem_u32(empty1 + u % SS<1>), (u / SS<1>) & 1);
      poll(true);
    }
    {  // ---- GEMM2 phase: dynamic queue (heaviest first), each item waits for its EC's GEMM1 tiles
      long it = 0;
      const int ntiles = *f.a2.nt_dev;
      int item = atomicAdd(f.g2ctr, 1);
      for (;;) {
        const bool end = item >= ntiles;
        const Tile tl = end ? Tile{0, 0, 0, 0, 0} : f.a2.tiles[item];
        const int next = end ? 0 : atomicAdd(f.g2ctr, 1);
        produce_item<2>(f.a2, ring, full2, empty2, it, item, tl, end, pw, pb, -1, f.done1, MS4_FSIG ? 9 : 9 * NCW);
        if (end) break;
        item = next;
      }
    }
    return;
  }
  const int ct = threadIdx.x, c = ct / 128, t = ct % 128;
  long it1 = 0, it2 = 0, nt1 = 0;
  while (consume_item<1>(f.a1, ring, full1, empty1, it1, c, t, MS4_FSIG ? nullptr : f.done1, MS4_FSIG ? smem_u32(hd) : 0u,
                         &nt1)) {
  }
  while (consume_item<2>(f.a2, ring, full2, empty2, it2, c, t)) {
  }
}


// ================================================================================================ GEMM1 row stream, n_pad 8
// For ECs of 1-8 tokens (70 % of GEMM1's weight bytes on measured decode steps). Physics: the tile
// stream re-reads the token image per stage (TMA L2 -> SMEM) and every warp re-reads the same B fragments (8x the A
// bytes in SMEM traffic), and its 2D boxes (256 B of each of 128 rows per stage) stream at 3.6 TB/s where contiguous
// bulk reaches 4.1. Here: work item = (EC, q) = 64 intermediates (128 rows) as in the tile stream; a stage = 8 gate + 8
// up rows over the WHOLE K, fetched as 16 contiguous 2,560-B row copies into padded SMEM rows (stride 2,576: ldmatrix
// conflict-free) + 2 x 1,280-B scale copies; warp w owns K slice [640 w, 640 w + 640) and keeps the EC's B fragments
// for it in registers (80) for the whole EC: the image is loaded once per EC switch (prefetched, one SMEM buffer).
// The 8 slice partials (fp32) are summed in a fixed order by one warp per stage, then the tile stream's epilogue.
// Items are split contiguously over the CTAs (equal sizes; consecutive items share their EC's image).
#ifndef MSR_R8NW
#define MSR_R8NW 8  // consumer warps of the row-stream GEMM1 (K slice 80 / NW k64 steps each; B fragments 40 / 80 regs)
#endif
#ifndef MSR_R8S
#define MSR_R8S 3  // its ring stages (44 KB each)
#endif
#ifndef MSR_R8ABL
#define MSR_R8ABL 0  // timing only: 1 = consumers skip the math (stream + B loads + reduction), 2 = also skip the reduction
#endif
namespace r8 {
constexpr int ROWS = 16;                             // 8 gate + 8 up rows per stage
constexpr int OFF_S = 2 * 20480;                     // A: [gate | up][20 chunks][8 rows][128 B] SW128 (3D TMA boxes)
constexpr int OFF_H = OFF_S + ROWS * (D / 32);       // scales [16][160 B], then the stage header
constexpr int STAGE = (OFF_H + 64 + 1023) / 1024 * 1024;  // 44,032
constexpr int IMG = 80 * 8 * 128;                    // the EC's token image [80 slabs][8 tokens][128 B]
template <int NW, int NS>
struct L {
  static constexpr int OFF_IMG = NS * STAGE, OFF_EC = OFF_IMG + IMG, OFF_RED = OFF_EC + 896,
                       OFF_BAR = OFF_RED + 2 * NW * 32 * 16, SMEM = 1024 + OFF_BAR + 128;
  static_assert(SMEM <= 227 * 1024, "r8 SMEM");
  static_assert(80 % NW == 0, "K slices");
};
struct H8 {
  int ec, q, s, pad;
};
}  // namespace r8

// Row-stream GEMM1 (n_pad 8): one stage = 8 gate + 8 up rows over the whole K as two 3D TMA boxes (128 B x 8 rows x 20
// chunks each, SW128 -> SMEM [chunk][row][128 B]: ldmatrix conflict-free, one TMA op per 20 KB of contiguous rows);
// consumer warp w owns K slice [KS w, KS w + KS) k64 steps (KS = 80 / NW) and keeps its B fragments in registers per EC.
template <int NW, int NS>
__global__ void __launch_bounds__(32 + 32 * NW, 1) k_gemm1r8(const __grid_constant__ Args<1> a) {
  using Ly = r8::L<NW, NS>;
  constexpr int KS = 80 / NW;
  extern __shared__ __align__(1024) char smem_raw[];
  char* sm = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  uint64_t* full = (uint64_t*)(sm + Ly::OFF_BAR);
  uint64_t* empty = full + NS;
  uint64_t* imfull = empty + NS;
  uint64_t* imempty = imfull + 1;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) mbar_init(smem_u32(full + s), 1), mbar_init(smem_u32(empty + s), NW);
    mbar_init(smem_u32(imfull), 1), mbar_init(smem_u32(imempty), NW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int nitems = 9 * *a.nt_dev;
  const int i0 = (int)((long long)blockIdx.x * nitems / gridDim.x);
  const int i1 = (int)((long long)(blockIdx.x + 1) * nitems / gridDim.x);
  if (warp == NW) {  // -------------------------------------------------------------------------- producer
    if (lane != 0) return;
    const uint64_t pw = pol_first(), pb = pol_last();
    long it = 0;
    int cur = -1, nim = 0, e = 0;
    for (int i = i0; i < i1; ++i) {
      const int k = i / 9, q = i - 9 * k, x = a.ec8[k];
      if (x != cur) {
        const EC* ecg = a.ecs + x;
        e = ecg->e;
        const long long boff = ecg->img_off;
        if (nim > 0) mbar_wait(smem_u32(imempty), (nim - 1) & 1);
        const uint32_t fb = smem_u32(imfull);
        mbar_expect_tx(fb, r8::IMG + (unsigned)sizeof(EC));
        for (int o = 0; o < r8::IMG; o += 16384) bulk(smem_u32(sm + Ly::OFF_IMG + o), a.bimg + boff + o, 16384, fb, pb);
        bulk(smem_u32(sm + Ly::OFF_EC), ecg, (unsigned)sizeof(EC), fb, pb);
        cur = x;
        ++nim;
      }
      const size_t rg = (size_t)e * 1152 + 64 * q, ru = rg + 576;
      for (int s8 = 0; s8 < 8; ++s8, ++it) {
        const int s = it % NS;
        if (it >= NS) mbar_wait(smem_u32(empty + s), ((it / NS) + 1) & 1);
        char* st = sm + (size_t)s * r8::STAGE;
        r8::H8* h = (r8::H8*)(st + r8::OFF_H);
        h->ec = x, h->q = q, h->s = s8;
        const uint32_t fb = smem_u32(full + s), dst = smem_u32(st);
        mbar_expect_tx(fb, r8::ROWS * (D / 2) + r8::ROWS * (D / 32));
        const size_t g0 = rg + 8 * s8, u0 = ru + 8 * s8;
        tma3d(dst, &a.tmw, 0, (int)g0, 0, fb, pw);
        tma3d(dst + 20480, &a.tmw, 0, (int)u0, 0, fb, pw);
        bulk(dst + r8::OFF_S, a.s2 + g0 * (D / 32), 8 * (D / 32), fb, pw);
        bulk(dst + r8::OFF_S + 8 * (D / 32), a.s2 + u0 * (D / 32), 8 * (D / 32), fb, pw);
      }
    }
    const int s = it % NS;  // end marker
    if (it >= NS) mbar_wait(smem_u32(empty + s), ((it / NS) + 1) & 1);
    ((r8::H8*)(sm + (size_t)s * r8::STAGE + r8::OFF_H))->ec = -1;
    mbar_arrive(smem_u32(full + s));
    return;
  }
  // ------------------------------------------------------------------------------------------- consumers (NW warps)
  const int w = warp, g = lane / 4, tq = lane % 4, lr = lane & 7, mi = lane >> 3;
  unsigned bfr[KS][2][4];
  int etok[2] = {-1, -1};
  float ewt[2] = {0.f, 0.f}, einv[2] = {0.f, 0.f};
  long long eh_off = 0;
  int cur = -1, nim = 0;
  long it = 0;
  const uint32_t aimg = smem_u32(sm + Ly::OFF_IMG);
  float4* red = (float4*)(sm + Ly::OFF_RED);
  for (;;) {
    const int s = it % NS;
    mbar_wait(smem_u32(full + s), (it / NS) & 1);
    char* st = sm + (size_t)s * r8::STAGE;
    const r8::H8 h = *(const r8::H8*)(st + r8::OFF_H);
    if (h.ec < 0) break;
    if (h.ec != cur) {  // new EC: its B fragments for this warp's K slice into registers, then release the image
      mbar_wait(smem_u32(imfull), nim & 1);
#pragma unroll
      for (int kb = 0; kb < KS; ++kb)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
          ldsm_x4(bfr[kb][kk], aimg + (unsigned)((KS * w + kb) * 1024) + swz(lane & 7, (kk * 4 + (lane >> 3)) * 16));
      const EC* e0 = (const EC*)(sm + Ly::OFF_EC);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 2 * tq + hh;
        etok[hh] = e0->tok[n], ewt[hh] = e0->wt[n], einv[hh] = e0->inv[n];
      }
      eh_off = e0->h_off;
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(imempty));
      cur = h.ec;
      ++nim;
    }
    const unsigned char* sc = (const unsigned char*)(st + r8::OFF_S);
    const uint32_t aA = smem_u32(st);
    float acc0[4] = {0.f, 0.f, 0.f, 0.f}, acc1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kb = 0; kb < (MSR_R8ABL ? 0 : KS); ++kb) {
      const int K64 = KS * w + kb;
      unsigned wv[4];  // gate row lr blk 2K, up row lr blk 2K, gate blk 2K+1, up blk 2K+1
      {
        const int b = 2 * K64 + (mi >> 1);
        ldsm_x4a(wv, aA + (unsigned)((mi & 1) * 20480 + ((b >> 3) * 8 + lr) * 128 + (((b & 7) ^ lr) << 4)));
      }
      const unsigned short gs = *(const unsigned short*)(sc + g * 160 + 2 * K64);
      const unsigned short us = *(const unsigned short*)(sc + (8 + g) * 160 + 2 * K64);
      const unsigned u = bias4((unsigned)gs | ((unsigned)us << 16));  // g blk 2K, g blk 2K+1, up blk 2K, up blk 2K+1
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const unsigned mg = __byte_perm(u, 0, 0x0404 + 0x1010 * j), mh = __byte_perm(u, 0, 0x2424 + 0x1010 * j);
        unsigned f0[4], f1[4];
        dequant_pair(f0, f1, wv[2 * j], wv[2 * j + 1], mg, mh);
        mma16816(acc0, f0, bfr[kb][j][0], bfr[kb][j][1]);
        mma16816(acc1, f1, bfr[kb][j][2], bfr[kb][j][3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_u32(empty + s));
    const int p = (int)(it & 1);
    if (MSR_R8ABL >= 2) {
      ++it;
      continue;
    }
    red[(p * NW + w) * 32 + lane] = make_float4(acc0[0] + acc1[0], acc0[1] + acc1[1], acc0[2] + acc1[2], acc0[3] + acc1[3]);
    asm volatile("bar.sync 1, %0;" ::"n"(32 * NW) : "memory");
    if (w == (int)(it % NW)) {  // fixed-order sum of the K slices, then the tile stream's epilogue
      float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float4 r = red[(p * NW + ww) * 32 + lane];
        v0 += r.x, v1 += r.y, v2 += r.z, v3 += r.w;
      }
      const float vg[2] = {v0, v1}, vu[2] = {v2, v3};
      const int il = 8 * h.s + g, pos = 32 * (il >> 5) + kinv(il & 31);
      char* hs = a.himg + eh_off + (size_t)h.q * 8 * 128;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 2 * tq + hh;
        float hv = 0.f;
        if (etok[hh] >= 0) {
          const float iv = einv[hh];
          const float gt = fminf(vg[hh] * iv, 10.f);
          const float up = fminf(fmaxf(vu[hh] * iv, -10.f), 10.f);
          hv = gt / (1.f + expf(-gt)) * up * ewt[hh] * 256.f;
        }
        *(__half*)(hs + swz(n, pos * 2)) = __float2half_rn(hv);
      }
    }
    ++it;
  }
}

// ================================================================================================ GEMM2 row stream, n_pad 8
// Same idea for w2: an EC's whole H (8 tokens x 576 = 9 KB) sits in 72 registers of every warp for all its 40 tiles, so a
// tile is just its contiguous 36,864-B weight block + 2,304-B scales (bulk, 5-stage ring); no per-tile H image fetch and
// no per-warp B ldmatrix (2x the A bytes in SMEM traffic). Rows per warp, K whole (summation order as the tile stream).
// Items (EC, 128-row tile) split contiguously over the CTAs; epilogue as MS3_YSTG with a double-buffered staging.
namespace q8 {
constexpr int OFF_S = 128 * (IR / 2);                 // 36,864
constexpr int OFF_H = OFF_S + 128 * (IR / 32);        // 39,168
constexpr int STAGE = (OFF_H + 64 + 1023) / 1024 * 1024;  // 40,960
constexpr int S = 5;
constexpr int HIMG = 9 * 8 * 128;
constexpr int OFF_IMG = S * STAGE;
constexpr int OFF_EC = OFF_IMG + HIMG;
constexpr int OFF_STG = OFF_EC + 1024;                // [2][8 tokens][128 rows] bf16
constexpr int OFF_BAR = OFF_STG + 2 * 8 * 128 * 2;
constexpr int SMEM = 1024 + OFF_BAR + 128;
static_assert(SMEM <= 227 * 1024, "q8 SMEM");
}  // namespace q8

__global__ void __launch_bounds__(32 + 32 * NCW, 1) k_gemm2r8(const __grid_constant__ Args<2> a) {
  extern __shared__ __align__(1024) char smem_raw[];
  char* sm = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  uint64_t* full = (uint64_t*)(sm + q8::OFF_BAR);
  uint64_t* empty = full + q8::S;
  uint64_t* imfull = empty + q8::S;
  uint64_t* imempty = imfull + 1;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int s = 0; s < q8::S; ++s) mbar_init(smem_u32(full + s), 1), mbar_init(smem_u32(empty + s), NCW);
    mbar_init(smem_u32(imfull), 1), mbar_init(smem_u32(imempty), NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int nitems = 40 * *a.nt_dev;
  const int i0 = (int)((long long)blockIdx.x * nitems / gridDim.x);
  const int i1 = (int)((long long)(blockIdx.x + 1) * nitems / gridDim.x);
  if (warp == NCW) {  // ------------------------------------------------------------------------- producer
    if (lane != 0) return;
    const uint64_t pw = pol_first(), pb = pol_last();
    long it = 0;
    int cur = -1, nim = 0, e = 0;
    for (int i = i0; i < i1; ++i, ++it) {
      const int k = i / 40, q = i - 40 * k, x = a.ec8[k];
      if (x != cur) {
        const EC* ecg = a.ecs + x;
        e = ecg->e;
        const long long hoff = ecg->h_off;
        if (nim > 0) mbar_wait(smem_u32(imempty), (nim - 1) & 1);
        const uint32_t fb = smem_u32(imfull);
        mbar_expect_tx(fb, q8::HIMG + (unsigned)sizeof(EC));
        bulk(smem_u32(sm + q8::OFF_IMG), a.bimg + hoff, q8::HIMG, fb, pb);
        bulk(smem_u32(sm + q8::OFF_EC), ecg, (unsigned)sizeof(EC), fb, pb);
        cur = x;
        ++nim;
      }
      const int s = it % q8::S;
      if (it >= q8::S) mbar_wait(smem_u32(empty + s), ((it / q8::S) + 1) & 1);
      char* st = sm + (size_t)s * q8::STAGE;
      r8::H8* h = (r8::H8*)(st + q8::OFF_H);
      h->ec = x, h->q = q;
      const uint32_t fb = smem_u32(full + s), dst = smem_u32(st);
      mbar_expect_tx(fb, q8::OFF_S + 128 * (IR / 32));
      const size_t rr = (size_t)e * D + 128 * q;
      bulk(dst, a.w2 + rr * (IR / 2), 128 * (IR / 2), fb, pw);
      bulk(dst + q8::OFF_S, a.s2 + rr * (IR / 32), 128 * (IR / 32), fb, pw);
    }
    const int s = it % q8::S;  // end marker
    if (it >= q8::S) mbar_wait(smem_u32(empty + s), ((it / q8::S) + 1) & 1);
    ((r8::H8*)(sm + (size_t)s * q8::STAGE + q8::OFF_H))->ec = -1;
    mbar_arrive(smem_u32(full + s));
    return;
  }
  // ------------------------------------------------------------------------------------------- consumers (8 warps)
  const int ct = threadIdx.x, c = ct / 128, t = ct % 128, w = t / 32, g = lane / 4, tq = lane % 4, lr = lane & 7,
            mi = lane >> 3;
  const int r0 = 64 * c + 16 * w, R = r0 + g;
  unsigned bfr[9][2][4];
  int eslot = 0, enr = 0, cur = -1, nim = 0;
  long it = 0;
  const uint32_t aimg = smem_u32(sm + q8::OFF_IMG);
  __nv_bfloat16* stg0 = (__nv_bfloat16*)(sm + q8::OFF_STG);
  for (;;) {
    const int s = it % q8::S;
    mbar_wait(smem_u32(full + s), (it / q8::S) & 1);
    char* st = sm + (size_t)s * q8::STAGE;
    const r8::H8 h = *(const r8::H8*)(st + q8::OFF_H);
    if (h.ec < 0) break;
    if (h.ec != cur) {
      mbar_wait(smem_u32(imfull), nim & 1);
#pragma unroll
      for (int kb = 0; kb < 9; ++kb)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
          ldsm_x4(bfr[kb][kk], aimg + (unsigned)(kb * 1024) + swz(lane & 7, (kk * 4 + (lane >> 3)) * 16));
      const EC* e0 = (const EC*)(sm + q8::OFF_EC);
      eslot = e0->pad, enr = e0->n_real;
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(imempty));
      cur = h.ec;
      ++nim;
    }
    const uint32_t aA = smem_u32(st);
    float acc0[4] = {0.f, 0.f, 0.f, 0.f}, acc1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kb = 0; kb < 9; ++kb) {
      unsigned wv[4];
      ldsm_x4a(wv, aA + (unsigned)((r0 + 8 * (mi & 1) + lr) * (IR / 2) + 32 * kb + 16 * (mi >> 1)));
      const unsigned short bgs = *(const unsigned short*)(st + q8::OFF_S + R * 18 + 2 * kb);
      const unsigned short bhs = *(const unsigned short*)(st + q8::OFF_S + (R + 8) * 18 + 2 * kb);
      const unsigned u = bias4((unsigned)bgs | ((unsigned)bhs << 16));
      const unsigned mg[2] = {__byte_perm(u, 0, 0x0404), __byte_perm(u, 0, 0x1414)};
      const unsigned mh[2] = {__byte_perm(u, 0, 0x2424), __byte_perm(u, 0, 0x3434)};
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        unsigned f0[4], f1[4];
        dequant_pair(f0, f1, wv[2 * j], wv[2 * j + 1], mg[j], mh[j]);
        mma16816(acc0, f0, bfr[kb][j][0], bfr[kb][j][1]);
        mma16816(acc1, f1, bfr[kb][j][2], bfr[kb][j][3]);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_u32(empty + s));
    __nv_bfloat16* stg = stg0 + (it & 1) * (8 * 128);
#pragma unroll
    for (int rs = 0; rs < 2; ++rs)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        stg[(2 * tq + j) * 128 + R + 8 * rs] = __float2bfloat16_rn((acc0[2 * rs + j] + acc1[2 * rs + j]) * (1.f / 256.f));
    asm volatile("bar.sync 1, 256;" ::: "memory");
    {
      const int n = ct / 16, ck = ct % 16;  // 16 chunks of 16 B per token, 16 tokens per pass (8 used)
      if (n < enr)
        *(uint4*)(a.yout + (size_t)(eslot + n) * D + 128 * h.q + 8 * ck) = *(const uint4*)(stg + n * 128 + 8 * ck);
    }
    ++it;
  }
}

// ================================================================================================ routing tables
// k_route: device routing tables from the router's top-6 (ids [T, 6] int32, w [T, 6] fp32), one CTA, deterministic.
//   1. stable radix sort of the P = 6T (token, slot) pairs by expert (CUB): sorted position i = the pair's Y slot;
//   2. per-expert counts and segments, experts split into chunks of <= 64 tokens (EC records, in expert order);
//   3. EC prefix sums of n_pad -> token-image (80 slabs) and H-image (9 slabs) offsets;
//   4. tiles (GEMM1: 9 per EC, GEMM2: 40) in snake order: heaviest n_pad first, every other round of `grid` reversed;
//   5. slotmap[p] = sorted position (for the combine), counts = {n_ec, ntiles1, ntiles2, img bytes, h bytes, P}.
constexpr int RT_THREADS = 1024, RT_ITEMS = 6, RT_MAXP = RT_THREADS * RT_ITEMS, NEXP = 384;
constexpr int RT_MAXEC = NEXP + RT_MAXP / NMAX;  // 480: sum over experts of ceil(c_e / 64)
using RtSort = cub::BlockRadixSort<int, RT_THREADS, RT_ITEMS, int>;
using RtScan = cub::BlockScan<int, RT_THREADS>;
struct RtSmem {
  union {
    typename RtSort::TempStorage sort;
    typename RtScan::TempStorage scan;
  } t;
  int sv[RT_MAXP];  // sorted pair index p = 6 t + k
  int cnt[NEXP], seg[NEXP], ecb[NEXP];
  int ec_e[RT_MAXEC], ec_c[RT_MAXEC], ec_np[RT_MAXEC], ec_pre[RT_MAXEC], ec_pos[RT_MAXEC];
  int n_ec;
};

// global scratch written by k_route for k_route_fill (int32 offsets)
constexpr int RS_SV = 0, RS_EE = RS_SV + RT_MAXP, RS_EK = RS_EE + RT_MAXEC, RS_ENP = RS_EK + RT_MAXEC,
              RS_EPRE = RS_ENP + RT_MAXEC, RS_EPOS = RS_EPRE + RT_MAXEC, RS_CNT = RS_EPOS + RT_MAXEC, RS_SEG = RS_CNT + NEXP,
              RS_NEC = RS_SEG + NEXP, RS_INTS = RS_NEC + 32;

__global__ void __launch_bounds__(RT_THREADS, 1) k_route(const int* __restrict__ ids, int T, int* __restrict__ rts,
                                                         int* __restrict__ counts, int* __restrict__ slotmap) {
  extern __shared__ __align__(16) char rt_raw[];
  RtSmem& sm = *reinterpret_cast<RtSmem*>(rt_raw);
  const int tid = threadIdx.x, P = 6 * T;
  int key[RT_ITEMS], val[RT_ITEMS];
#pragma unroll
  for (int r = 0; r < RT_ITEMS; ++r) {
    const int p = tid * RT_ITEMS + r;
    const int id = p < P ? ids[p] : -1;
    key[r] = (id >= 0 && id < NEXP) ? id : 1023;  // padding and out-of-range ids (contribute nothing) sort last
    val[r] = p;
  }
  for (int e = tid; e < NEXP; e += RT_THREADS) sm.cnt[e] = 0;
  RtSort(sm.t.sort).Sort(key, val, 0, 10);  // stable: within an expert the pairs stay in (token, slot) order
  __syncthreads();
#pragma unroll
  for (int r = 0; r < RT_ITEMS; ++r) {
    const int i = tid * RT_ITEMS + r;  // blocked arrangement
    if (i < P) {
      sm.sv[i] = val[r];
      rts[RS_SV + i] = val[r];
      if (key[r] < NEXP) {
        slotmap[val[r]] = i;
        atomicAdd(&sm.cnt[key[r]], 1);  // the counts do not depend on the atomic order
      } else {
        slotmap[val[r]] = -1;
      }
    }
  }
  __syncthreads();
  {  // segments and EC bases: exclusive scans over the experts
    const int c = tid < NEXP ? sm.cnt[tid] : 0, nc = (c + NMAX - 1) / NMAX;
    int sc, snc, tot;
    RtScan(sm.t.scan).ExclusiveSum(c, sc);
    __syncthreads();
    RtScan(sm.t.scan).ExclusiveSum(nc, snc, tot);
    if (tid < NEXP) sm.seg[tid] = sc, sm.ecb[tid] = snc;
    if (tid == 0) sm.n_ec = tot;
    __syncthreads();
    if (tid < NEXP)
      for (int k = 0; k < nc; ++k) {
        const int x = snc + k, nr = min(NMAX, c - NMAX * k);
        sm.ec_e[x] = tid, sm.ec_c[x] = k, sm.ec_np[x] = (nr + 7) & ~7;
      }
  }
  __syncthreads();
  const int n_ec = sm.n_ec;
  {  // image offsets: exclusive scan of n_pad over the ECs
    const int np = tid < n_ec ? sm.ec_np[tid] : 0;
    int pre, tot;
    RtScan(sm.t.scan).ExclusiveSum(np, pre, tot);
    if (tid < n_ec) sm.ec_pre[tid] = pre;
    if (tid == 0) counts[3] = tot * 80 * 128, counts[4] = tot * 9 * 128;
  }
  __syncthreads();
  if (tid < n_ec) {  // stable rank by n_pad descending (8 buckets)
    const int np = sm.ec_np[tid];
    int before = 0;
    for (int x = 0; x < n_ec; ++x) {
      const int o = sm.ec_np[x];
      before += (o > np) || (o == np && x < tid);
    }
    sm.ec_pos[tid] = before;
  }
  __syncthreads();
  // hand the small per-EC tables to k_route_fill (many CTAs: one SM writing ~0.5 MB of ECs and tiles took ~30 us)
  for (int x = tid; x < n_ec; x += RT_THREADS) {
    rts[RS_EE + x] = sm.ec_e[x], rts[RS_EK + x] = sm.ec_c[x], rts[RS_ENP + x] = sm.ec_np[x];
    rts[RS_EPRE + x] = sm.ec_pre[x], rts[RS_EPOS + x] = sm.ec_pos[x];
  }
  for (int e = tid; e < NEXP; e += RT_THREADS) rts[RS_CNT + e] = sm.cnt[e], rts[RS_SEG + e] = sm.seg[e];
  if (tid == 0) {
    rts[RS_NEC] = n_ec;
    counts[0] = n_ec, counts[1] = 9 * n_ec, counts[2] = 40 * n_ec, counts[5] = P;
  }
}

// k_route_fill: EC records (64 token slots each) and the snake-ordered GEMM1 / GEMM2 tiles, grid-stride over all CTAs.
__global__ void __launch_bounds__(256) k_route_fill(const int* __restrict__ rts, const float* __restrict__ w,
                                                    EC* __restrict__ ecs, Tile* __restrict__ t1, Tile* __restrict__ t2,
                                                    int grid, const float* __restrict__ inv) {
  const int n_ec = rts[RS_NEC], stride = gridDim.x * blockDim.x;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_ec * (NMAX + 49); j += stride) {
    if (j < n_ec * NMAX) {  // EC token slots
      const int x = j / NMAX, n = j % NMAX, e = rts[RS_EE + x], k = rts[RS_EK + x];
      const int nr = min(NMAX, rts[RS_CNT + e] - NMAX * k), base = rts[RS_SEG + e] + NMAX * k;
      EC& ec = ecs[x];
      const int p = n < nr ? rts[RS_SV + base + n] : -1;
      ec.tok[n] = p >= 0 ? p / 6 : -1;
      ec.wt[n] = p >= 0 ? w[p] : 0.f;
      ec.inv[n] = (inv && p >= 0) ? inv[p / 6] : 0.f;
      if (n == 0) {
        const long long pre = rts[RS_EPRE + x];
        ec.e = e, ec.n_real = nr, ec.n_pad = rts[RS_ENP + x], ec.pad = base;
        ec.img_off = pre * 80 * 128, ec.h_off = pre * 9 * 128;
      }
    } else {  // tiles: 9 GEMM1 + 40 GEMM2 per EC, heaviest first, alternate rounds reversed
      const int jj = j - n_ec * NMAX, x = jj / 49, q = jj % 49, e = rts[RS_EE + x], np = rts[RS_ENP + x];
      const long long pre = rts[RS_EPRE + x];
      const int pos0 = rts[RS_EPOS + x];
      if (q < 9) {
        const int u = pos0 * 9 + q, n1 = 9 * n_ec, r = u / grid, pos = u % grid, L = min(grid, n1 - r * grid);
        t1[r * grid + ((r & 1) ? L - 1 - pos : pos)] = Tile{x, q, e, np, pre * 80 * 128};
      } else {
        const int qq = q - 9, u = pos0 * 40 + qq, n2 = 40 * n_ec, r = u / grid, pos = u % grid,
                  L = min(grid, n2 - r * grid);
        t2[r * grid + ((r & 1) ? L - 1 - pos : pos)] = Tile{x, qq, e, np, pre * 9 * 128};
      }
    }
  }
}

#ifndef MSR_RT_W
#define MSR_RT_W 8  // k_router_top warps (tokens) per CTA
#endif
#ifndef MSR_RT_PAR
#define MSR_RT_PAR 0  // 1: the 6 outputs written by lanes 0..5 in parallel (same values) instead of lane 0 in sequence
#endif
// k_router_top: router epilogue on the fp32 gate logits [T, 384] (bf16 x bf16 GEMM, fp32 accumulate: exact products,
// as ref gate_ref): scores = sqrt(softplus(logit)) (torch softplus: threshold 20), top-6 of scores + bias (ties: lower
// expert id), weights = scores[sel] / (sum + 1e-20) * route_scale. One warp per token; ids in descending order.
__global__ void k_router_top(const float* __restrict__ logits, const float* __restrict__ bias, int T,
                             int* __restrict__ ids, float* __restrict__ wts, float route_scale,
                             unsigned char* __restrict__ row = nullptr, int nvalid = 1 << 30,
                             const int* __restrict__ ptok_in = nullptr) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= T) return;
  float sc[12], sel[12];
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    const int e = i * 32 + lane;
    const float x = logits[(size_t)warp * NEXP + e];
    sc[i] = sqrtf(x > 20.f ? x : log1pf(expf(x)));
    sel[i] = sc[i] + bias[e];
  }
  float pick[6];
  int pid[6];
  unsigned taken = 0;
#pragma unroll
  for (int r = 0; r < 6; ++r) {
    float bv = -INFINITY;
    int bi = 1 << 30;
#pragma unroll
    for (int i = 0; i < 12; ++i)
      if (!((taken >> i) & 1) && sel[i] > bv) bv = sel[i], bi = i * 32 + lane;  // within a lane: ascending ids
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    pid[r] = bi;
    if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
    float s_r = 0.f;
#pragma unroll
    for (int i = 0; i < 12; ++i)
      if (i * 32 + lane == bi) s_r = sc[i];
    pick[r] = __shfl_sync(0xffffffffu, s_r, bi & 31);
  }
#if MSR_RT_PAR
  // lane r < 6 writes output r (every lane holds all 6 picks / ids and forms the same sum in the same order: same bits)
  float sum = 0.f;
#pragma unroll
  for (int r = 0; r < 6; ++r) sum += pick[r];
  if (lane < 6) {
    float pr = pick[0];
    int ir = pid[0];
#pragma unroll
    for (int r = 1; r < 6; ++r)
      if (lane == r) pr = pick[r], ir = pid[r];
    const float wv = pr / (sum + 1e-20f) * route_scale;
    ids[warp * 6 + lane] = ir;
    wts[warp * 6 + lane] = wv;
    if (row) {
      ((int*)(row + (size_t)warp * 64))[lane] = warp < nvalid ? ir : -1;
      ((float*)(row + (size_t)warp * 64 + 24))[lane] = wv;
    }
  } else if (lane == 6 && row && ptok_in) {
    ((int*)(row + (size_t)warp * 64 + 48))[0] = warp < nvalid ? ptok_in[warp] : 0;
  }
#else
  if (lane == 0) {
    float sum = 0.f;
#pragma unroll
    for (int r = 0; r < 6; ++r) sum += pick[r];
#pragma unroll
    for (int r = 0; r < 6; ++r) {
      const float wv = pick[r] / (sum + 1e-20f) * route_scale;
      ids[warp * 6 + r] = pid[r];
      wts[warp * 6 + r] = wv;
      if (row) {  // SP all-gather row: 6 int32 ids (-1 for padding rows >= nvalid) || 6 fp32 weights || 16 B pad
        ((int*)(row + (size_t)warp * 64))[r] = warp < nvalid ? pid[r] : -1;
        ((float*)(row + (size_t)warp * 64 + 24))[r] = wv;
      }
    }
    if (row && ptok_in) ((int*)(row + (size_t)warp * 64 + 48))[0] = warp < nvalid ? ptok_in[warp] : 0;  // prescale p
  }
#endif
}

// k_swiglu: shared expert middle step on the bf16 [T, 2 I] output of the fused gate|up GEMM -> bf16 h [T, I]
// (stock: up clamp(-10, 10), gate clamp(max 10), silu in fp32, h rounded to bf16). grid T, block I / 8.
__global__ void k_swiglu(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ h, int I) {
  const int t = blockIdx.x, i0 = threadIdx.x * 8;
  const uint4 gv = *(const uint4*)(gu + (size_t)t * 2 * I + i0), uv = *(const uint4*)(gu + (size_t)t * 2 * I + I + i0);
  const __nv_bfloat162* g2 = (const __nv_bfloat162*)&gv;
  const __nv_bfloat162* u2 = (const __nv_bfloat162*)&uv;
  uint4 o;
  __nv_bfloat162* o2 = (__nv_bfloat162*)&o;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 gf = __bfloat1622float2(g2[k]), uf = __bfloat1622float2(u2[k]);
    const float ga = fminf(gf.x, 10.f), gb = fminf(gf.y, 10.f);
    const float ua = fminf(fmaxf(uf.x, -10.f), 10.f), ub = fminf(fmaxf(uf.y, -10.f), 10.f);
    o2[k] = __floats2bfloat162_rn(ga / (1.f + expf(-ga)) * ua, gb / (1.f + expf(-gb)) * ub);
  }
  *(uint4*)(h + (size_t)t * I + i0) = o;
}

// k_combine: per token, the 6 bf16 expert outputs (slot order k = 0..5, fp32 sum) + the bf16 shared-expert output
// -> bf16 rank partial (stock: y = bf16(routed + shared)). grid T, block 640 x 8 elements.
__global__ void k_combine(const __nv_bfloat16* __restrict__ Y, const int* __restrict__ slotmap,
                          const __nv_bfloat16* __restrict__ sh, __nv_bfloat16* __restrict__ out) {
  const int t = blockIdx.x, d0 = threadIdx.x * 8;
  int sl[6];
#pragma unroll
  for (int k = 0; k < 6; ++k) sl[k] = slotmap[6 * t + k];  // -1: id outside [0, E), contributes nothing
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  uint4 v[6];
#pragma unroll
  for (int k = 0; k < 6; ++k) v[k] = sl[k] >= 0 ? *(const uint4*)(Y + (size_t)sl[k] * D + d0) : make_uint4(0, 0, 0, 0);
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    const __nv_bfloat162* h = (const __nv_bfloat162*)&v[k];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(h[i]);
      acc[2 * i] += f.x, acc[2 * i + 1] += f.y;
    }
  }
  const uint4 sv = sh ? *(const uint4*)(sh + (size_t)t * D + d0) : make_uint4(0, 0, 0, 0);  // addend optional
  const __nv_bfloat162* hs = (const __nv_bfloat162*)&sv;
  uint4 o;
  __nv_bfloat162* ho = (__nv_bfloat162*)&o;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(hs[i]);
    ho[i] = __floats2bfloat162_rn(acc[2 * i] + f.x, acc[2 * i + 1] + f.y);
  }
  *(uint4*)(out + (size_t)t * D + d0) = o;
}

}  // namespace msr

// ------------------------------------------------------------------------------------------------ bindings
using namespace msr;
static int g_sms = 0;

void route(torch::Tensor ids, torch::Tensor w, int64_t T, torch::Tensor ecs, torch::Tensor t1, torch::Tensor t2,
           torch::Tensor counts, torch::Tensor slotmap, torch::Tensor rts, int64_t grid, torch::Tensor inv) {
  TORCH_CHECK(T * 6 <= RT_MAXP, "route: T too large");
  TORCH_CHECK(rts.numel() >= RS_INTS, "route: scratch too small");
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool attr_dev[64] = {};  // per device: a one-process multi-GPU harness calls this on several devices
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  bool& attr = attr_dev[cur_dev & 63];
  if (!attr) {
    cudaFuncSetAttribute(k_route, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(RtSmem));
    attr = true;
  }
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, ids.get_device());
  k_route<<<1, RT_THREADS, sizeof(RtSmem), stream>>>(ids.data_ptr<int>(), (int)T, rts.data_ptr<int>(), counts.data_ptr<int>(),
                                                     slotmap.data_ptr<int>());
  k_route_fill<<<2 * g_sms, 256, 0, stream>>>(rts.data_ptr<int>(), w.data_ptr<float>(), (EC*)ecs.data_ptr(), (Tile*)t1.data_ptr(),
                                              (Tile*)t2.data_ptr(), grid > 0 ? (int)grid : g_sms,
                                              inv.numel() ? inv.data_ptr<float>() : nullptr);
}
void swiglu(torch::Tensor gu, torch::Tensor h) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int I = (int)h.size(1);
  TORCH_CHECK(I % 8 == 0 && gu.size(1) == 2 * I, "swiglu shapes");
  k_swiglu<<<(unsigned)h.size(0), I / 8, 0, stream>>>((const __nv_bfloat16*)gu.data_ptr(), (__nv_bfloat16*)h.data_ptr(), I);
}
int64_t route_scratch_ints() { return RS_INTS; }
void router_top(torch::Tensor logits, torch::Tensor bias, torch::Tensor ids, torch::Tensor wts, double route_scale) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int T = (int)logits.size(0);
  TORCH_CHECK(logits.size(1) == NEXP, "router_top: 384 experts");
  k_router_top<<<(T + MSR_RT_W - 1) / MSR_RT_W, 32 * MSR_RT_W, 0, stream>>>(logits.data_ptr<float>(), bias.data_ptr<float>(), T, ids.data_ptr<int>(),
                                                wts.data_ptr<float>(), (float)route_scale);
}

// router_top + the SP all-gather row in one kernel: row [T, 64] uint8 (6 int32 ids, -1 for rows >= nvalid; 6 fp32 weights)
void router_top_pack(torch::Tensor logits, torch::Tensor bias, torch::Tensor ids, torch::Tensor wts, double route_scale,
                     torch::Tensor row, int64_t nvalid) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int T = (int)logits.size(0);
  TORCH_CHECK(logits.size(1) == NEXP && row.numel() >= (int64_t)T * 64, "router_top_pack: shapes");
  k_router_top<<<(T + MSR_RT_W - 1) / MSR_RT_W, 32 * MSR_RT_W, 0, stream>>>(logits.data_ptr<float>(), bias.data_ptr<float>(), T, ids.data_ptr<int>(),
                                                wts.data_ptr<float>(), (float)route_scale,
                                                (unsigned char*)row.data_ptr(), (int)nvalid);
}

// router_top_pack + the prescale exponent of each row (int32 [T], from mhc boundary2) into the row's bytes 48..51
void router_top_pack2(torch::Tensor logits, torch::Tensor bias, torch::Tensor ids, torch::Tensor wts, double route_scale,
                      torch::Tensor row, int64_t nvalid, torch::Tensor ptok) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int T = (int)logits.size(0);
  TORCH_CHECK(logits.size(1) == NEXP && row.numel() >= (int64_t)T * 64 && ptok.numel() >= T, "router_top_pack2: shapes");
  k_router_top<<<(T + MSR_RT_W - 1) / MSR_RT_W, 32 * MSR_RT_W, 0, stream>>>(logits.data_ptr<float>(), bias.data_ptr<float>(), T, ids.data_ptr<int>(),
                                                wts.data_ptr<float>(), (float)route_scale,
                                                (unsigned char*)row.data_ptr(), (int)nvalid, ptok.data_ptr<int>());
}
// prepr without k_prescale: the exponents come with the all-gathered routing rows (rows [T, 64] uint8, int32 at byte 48)
void prepr2(torch::Tensor x, torch::Tensor ecs, torch::Tensor counts, int64_t max_ec, torch::Tensor img, torch::Tensor rows) {
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(rows.numel() >= x.size(0) * 64, "prepr2: routing rows [T, 64]");
  k_gather2<<<dim3((unsigned)max_ec, 8), 640, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (EC*)ecs.data_ptr(),
                                                           (const int*)rows.data_ptr() + 12, (char*)img.data_ptr(),
                                                           counts.data_ptr<int>(), 16);
}
// prepr2 by the persistent k_gather5 (route2's EC numbering; same image bytes / ec.inv as prepr2)
void prepr5(torch::Tensor x, torch::Tensor ecs, torch::Tensor counts, torch::Tensor img, torch::Tensor rows, int64_t sms) {
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(rows.numel() >= x.size(0) * 64, "prepr5: routing rows [T, 64]");
  k_gather5<<<(unsigned)(MSR_G5B * sms), 640, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (EC*)ecs.data_ptr(),
                                                           (const int*)rows.data_ptr() + 12, (char*)img.data_ptr(),
                                                           counts.data_ptr<int>(), 16);
}
// prepr6 (moe_v4): prepr's outputs (ptok, inv, image bytes, ec.inv) by k_prescale2 + the persistent k_gather5 (route2's
// EC numbering), for callers without the routing-row exponent (the drafter)
void prepr6(torch::Tensor x, torch::Tensor ecs, torch::Tensor counts, torch::Tensor img, torch::Tensor ptok, torch::Tensor inv,
            int64_t sms) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int T = (int)x.size(0);
  TORCH_CHECK(x.size(1) == D && ptok.numel() >= T && inv.numel() >= T, "prepr6: shapes");
  k_prescale2<<<T, 640, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), ptok.data_ptr<int>(), inv.data_ptr<float>());
  k_gather5<<<(unsigned)(MSR_G5B * sms), 640, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (EC*)ecs.data_ptr(),
                                                           ptok.data_ptr<int>(), (char*)img.data_ptr(), counts.data_ptr<int>(), 1);
}
void prepr(torch::Tensor x, torch::Tensor ecs, torch::Tensor counts, int64_t max_ec, torch::Tensor img, torch::Tensor ptok,
           torch::Tensor inv) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int T = x.size(0);
  if (MSR_GATHER3 == 2) {
    k_gather4<<<dim3((unsigned)max_ec, 8), 640, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (EC*)ecs.data_ptr(),
                                                             ptok.data_ptr<int>(), inv.data_ptr<float>(),
                                                             (char*)img.data_ptr(), counts.data_ptr<int>());
    return;
  }
  if (MSR_GATHER3) {
    k_gather3<<<dim3((unsigned)max_ec, MSR_G3Y), 160 * G3G, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (EC*)ecs.data_ptr(),
                                                             ptok.data_ptr<int>(), inv.data_ptr<float>(),
                                                             (char*)img.data_ptr(), counts.data_ptr<int>());
    return;
  }
  k_prescale<<<T, 256, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), ptok.data_ptr<int>(), inv.data_ptr<float>());
  if (MSR_GATHER2)
    k_gather2<<<dim3((unsigned)max_ec, 8), 640, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (EC*)ecs.data_ptr(),
                                                             ptok.data_ptr<int>(), (char*)img.data_ptr(), counts.data_ptr<int>());
  else
    k_gather<<<dim3((unsigned)max_ec, 8), 640, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (EC*)ecs.data_ptr(),
                                                            ptok.data_ptr<int>(), (char*)img.data_ptr(), counts.data_ptr<int>());
}
static PFN_cuTensorMapEncodeTiled_v12000 tmap_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&fn, 12000, cudaEnableDefault, &q) ==
                        cudaSuccess && fn,
                "cuTensorMapEncodeTiled not available");
  }
  return fn;
}
static CUtensorMap tmap2d(void* base, CUtensorMapDataType ty, int esz, uint64_t inner, uint64_t rows, uint32_t box0,
                          uint32_t box1, CUtensorMapSwizzle sw) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)(inner * esz)};
  cuuint32_t box[2] = {box0, box1}, es[2] = {1, 1};
  CUresult r = tmap_encode()(&m, ty, 2, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
// w13 as [rows][20 chunks][128 B] with dims ordered (byte, row, chunk): a box (128, 8, 20) lands as [chunk][row][128 B]
static CUtensorMap tmap_rows3d(void* base, uint64_t rows) {
  CUtensorMap m;
  cuuint64_t dims[3] = {128, (cuuint64_t)rows, 20};
  cuuint64_t strides[2] = {(cuuint64_t)(D / 2), 128};
  cuuint32_t box[3] = {128, 8, 20}, es[3] = {1, 1, 1};
  CUresult r = tmap_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, base, dims, strides, box, es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d rows) failed: ", (int)r);
  return m;
}
template <int MODE>
static void launch(Args<MODE>& a, int64_t grid) {
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool attr_dev[64] = {};  // per device: a one-process multi-GPU harness calls this on several devices
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  bool& attr = attr_dev[cur_dev & 63];
  if (!attr) {
    cudaFuncSetAttribute(k_gemm<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, Lay<MODE>::SMEM);
    attr = true;
  }
  if (MS4_PDL && MODE == 2) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid > 0 ? (unsigned)grid : (unsigned)g_sms);
    cfg.blockDim = dim3(32 + 32 * NCW);
    cfg.dynamicSmemBytes = Lay<MODE>::SMEM;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, k_gemm<MODE>, a);
    return;
  }
  k_gemm<MODE><<<grid > 0 ? (int)grid : g_sms, 32 + 32 * NCW, Lay<MODE>::SMEM, stream>>>(a);
}
// raw w13 [E, 2I, H/2], s13 [E, 2I, H/32] (uint8), counts[1] = GEMM1 tiles
void gemm1r(torch::Tensor w13, torch::Tensor s13, torch::Tensor ecs, torch::Tensor tiles, torch::Tensor counts,
            torch::Tensor img, torch::Tensor himg, int64_t grid) {
  TORCH_CHECK(w13.size(1) == 1152 && w13.size(2) == D / 2 && s13.size(2) == D / 32, "gemm1r: raw w13 / s13 shapes");
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, w13.get_device());
  Args<1> a{};
  a.ecs = (const EC*)ecs.data_ptr(), a.tiles = (const Tile*)tiles.data_ptr(), a.bimg = (const char*)img.data_ptr();
  a.himg = (char*)himg.data_ptr(), a.nt_dev = counts.data_ptr<int>() + 1;
  a.w2 = (const unsigned char*)w13.data_ptr();  // GEMM1: raw w13 (only the MS3_ABL 9 timing ablation reads it directly)
  const uint64_t rows = (uint64_t)w13.size(0) * 1152;
  a.tmw = tmap2d(w13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 2, rows, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B);
  a.tms = tmap2d(s13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 32, rows, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE);
  a.tmw2 = tmap2d(w13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 2, rows, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  a.tms2 = tmap2d(s13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 32, rows, 16, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
  launch<1>(a, grid);
}
// gemm1r with skip8 = 1: the tile stream leaves the n_pad-8 tiles to gemm1r8
void gemm1rs(torch::Tensor w13, torch::Tensor s13, torch::Tensor ecs, torch::Tensor tiles, torch::Tensor counts,
             torch::Tensor img, torch::Tensor himg, int64_t grid, int64_t skip8) {
  TORCH_CHECK(w13.size(1) == 1152 && w13.size(2) == D / 2 && s13.size(2) == D / 32, "gemm1rs: raw w13 / s13 shapes");
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, w13.get_device());
  Args<1> a{};
  a.ecs = (const EC*)ecs.data_ptr(), a.tiles = (const Tile*)tiles.data_ptr(), a.bimg = (const char*)img.data_ptr();
  a.himg = (char*)himg.data_ptr(), a.nt_dev = counts.data_ptr<int>() + 1;
  a.w2 = (const unsigned char*)w13.data_ptr();
  a.skip8 = (int)skip8;
  const uint64_t rows = (uint64_t)w13.size(0) * 1152;
  a.tmw = tmap2d(w13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 2, rows, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B);
  a.tms = tmap2d(s13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 32, rows, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE);
  a.tmw2 = tmap2d(w13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 2, rows, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  a.tms2 = tmap2d(s13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 32, rows, 16, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
  launch<1>(a, grid);
}
// row-stream GEMM1 of the n_pad-8 ECs: ec8 / counts[6] from route2x; same H image as gemm1r
void gemm1r8(torch::Tensor w13, torch::Tensor s13, torch::Tensor ecs, torch::Tensor ec8, torch::Tensor counts,
             torch::Tensor img, torch::Tensor himg, int64_t grid) {
  TORCH_CHECK(w13.size(1) == 1152 && w13.size(2) == D / 2 && s13.size(2) == D / 32, "gemm1r8: raw w13 / s13 shapes");
  TORCH_CHECK(counts.numel() >= 7 && ec8.scalar_type() == torch::kInt32, "gemm1r8: counts >= 7, ec8 int32");
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, w13.get_device());
  Args<1> a{};
  a.ecs = (const EC*)ecs.data_ptr(), a.bimg = (const char*)img.data_ptr(), a.himg = (char*)himg.data_ptr();
  a.nt_dev = counts.data_ptr<int>() + 6, a.ec8 = ec8.data_ptr<int>();
  a.w2 = (const unsigned char*)w13.data_ptr(), a.s2 = (const unsigned char*)s13.data_ptr();
  a.tmw = tmap_rows3d(w13.data_ptr(), (uint64_t)w13.size(0) * 1152);
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int SM8 = r8::L<MSR_R8NW, MSR_R8S>::SMEM;
  static bool attr_dev[64] = {};
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  if (!attr_dev[cur_dev & 63]) {
    cudaFuncSetAttribute(k_gemm1r8<MSR_R8NW, MSR_R8S>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM8);
    attr_dev[cur_dev & 63] = true;
  }
  k_gemm1r8<MSR_R8NW, MSR_R8S><<<grid > 0 ? (int)grid : g_sms, 32 + 32 * MSR_R8NW, SM8, stream>>>(a);
}
// raw w2 [E, H, I/2], s2 [E, H, I/32] (uint8), counts[2] = GEMM2 tiles; yout bf16 [slots][H]
void gemm2r(torch::Tensor w2, torch::Tensor s2, torch::Tensor ecs, torch::Tensor tiles, torch::Tensor counts,
            torch::Tensor himg, torch::Tensor yout, int64_t grid) {
  TORCH_CHECK(w2.size(1) == D && w2.size(2) == IR / 2 && s2.size(2) == IR / 32, "gemm2r: raw w2 / s2 shapes");
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, w2.get_device());
  Args<2> a{};
  a.s2 = (const unsigned char*)s2.data_ptr(), a.ecs = (const EC*)ecs.data_ptr(), a.tiles = (const Tile*)tiles.data_ptr();
  a.w2 = (const unsigned char*)w2.data_ptr();
  a.bimg = (const char*)himg.data_ptr(), a.nt_dev = counts.data_ptr<int>() + 2;
  a.yout = (__nv_bfloat16*)yout.data_ptr();
  a.tmw = tmap2d(w2.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, IR / 4, (uint64_t)w2.size(0) * D, 48, 128,
                 CU_TENSOR_MAP_SWIZZLE_NONE);
  a.tmw2 = tmap2d(w2.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, IR / 4, (uint64_t)w2.size(0) * D, 48, 256,
                  CU_TENSOR_MAP_SWIZZLE_NONE);
  launch<2>(a, grid);
}
void gemm2rs(torch::Tensor w2, torch::Tensor s2, torch::Tensor ecs, torch::Tensor tiles, torch::Tensor counts,
             torch::Tensor himg, torch::Tensor yout, int64_t grid, int64_t skip8) {
  TORCH_CHECK(w2.size(1) == D && w2.size(2) == IR / 2 && s2.size(2) == IR / 32, "gemm2rs: raw w2 / s2 shapes");
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, w2.get_device());
  Args<2> a{};
  a.s2 = (const unsigned char*)s2.data_ptr(), a.ecs = (const EC*)ecs.data_ptr(), a.tiles = (const Tile*)tiles.data_ptr();
  a.w2 = (const unsigned char*)w2.data_ptr();
  a.bimg = (const char*)himg.data_ptr(), a.nt_dev = counts.data_ptr<int>() + 2;
  a.yout = (__nv_bfloat16*)yout.data_ptr();
  a.skip8 = (int)skip8;
  a.tmw = tmap2d(w2.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, IR / 4, (uint64_t)w2.size(0) * D, 48, 128,
                 CU_TENSOR_MAP_SWIZZLE_NONE);
  a.tmw2 = tmap2d(w2.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, IR / 4, (uint64_t)w2.size(0) * D, 48, 256,
                  CU_TENSOR_MAP_SWIZZLE_NONE);
  launch<2>(a, grid);
}
// row-stream GEMM2 of the n_pad-8 ECs (ec8 / counts[6] from route2x); same Y as gemm2r
void gemm2r8(torch::Tensor w2, torch::Tensor s2, torch::Tensor ecs, torch::Tensor ec8, torch::Tensor counts,
             torch::Tensor himg, torch::Tensor yout, int64_t grid) {
  TORCH_CHECK(w2.size(1) == D && w2.size(2) == IR / 2 && s2.size(2) == IR / 32, "gemm2r8: raw w2 / s2 shapes");
  TORCH_CHECK(counts.numel() >= 7 && ec8.scalar_type() == torch::kInt32, "gemm2r8: counts >= 7, ec8 int32");
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, w2.get_device());
  Args<2> a{};
  a.s2 = (const unsigned char*)s2.data_ptr(), a.w2 = (const unsigned char*)w2.data_ptr(), a.ecs = (const EC*)ecs.data_ptr();
  a.bimg = (const char*)himg.data_ptr(), a.nt_dev = counts.data_ptr<int>() + 6, a.ec8 = ec8.data_ptr<int>();
  a.yout = (__nv_bfloat16*)yout.data_ptr();
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool attr_dev[64] = {};
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  if (!attr_dev[cur_dev & 63]) {
    cudaFuncSetAttribute(k_gemm2r8, cudaFuncAttributeMaxDynamicSharedMemorySize, q8::SMEM);
    attr_dev[cur_dev & 63] = true;
  }
  k_gemm2r8<<<grid > 0 ? (int)grid : g_sms, 32 + 32 * NCW, q8::SMEM, stream>>>(a);
}
void combine(torch::Tensor Y, torch::Tensor slotmap, torch::Tensor sh, torch::Tensor out) {
  auto stream = at::cuda::getCurrentCUDAStream();
  k_combine<<<(unsigned)out.size(0), D / 8, 0, stream>>>((const __nv_bfloat16*)Y.data_ptr(), slotmap.data_ptr<int>(),
                                                          sh.numel() ? (const __nv_bfloat16*)sh.data_ptr() : nullptr,
                                                          (__nv_bfloat16*)out.data_ptr());
}
int64_t route_max_ec() { return RT_MAXEC; }
int64_t gemm_ctas() { return MS3_CTAS; }

int64_t smem_bytes(int64_t mode) { return mode == 1 ? Lay<1>::SMEM : Lay<2>::SMEM; }
// fused GEMM1 -> GEMM2 (MS4_FUSE builds): the arguments of gemm1r + gemm2r; sync = int32 scratch >= max_ec + 1 (zeroed here)
void gemm12(torch::Tensor w13, torch::Tensor s13, torch::Tensor w2, torch::Tensor s2, torch::Tensor ecs, torch::Tensor t1,
            torch::Tensor t2, torch::Tensor counts, torch::Tensor img, torch::Tensor himg, torch::Tensor yout,
            torch::Tensor sync) {
  TORCH_CHECK(w13.size(1) == 1152 && w13.size(2) == D / 2 && s13.size(2) == D / 32, "gemm12: raw w13 / s13 shapes");
  TORCH_CHECK(w2.size(1) == D && w2.size(2) == IR / 2 && s2.size(2) == IR / 32, "gemm12: raw w2 / s2 shapes");
  const int64_t mec = ecs.numel() / (int64_t)sizeof(EC);
  TORCH_CHECK(sync.scalar_type() == torch::kInt32 && sync.numel() >= mec + 1, "gemm12: sync int32 >= max_ec + 1");
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, w13.get_device());
  FArgs f{};
  Args<1>& a = f.a1;
  a.ecs = (const EC*)ecs.data_ptr(), a.tiles = (const Tile*)t1.data_ptr(), a.bimg = (const char*)img.data_ptr();
  a.himg = (char*)himg.data_ptr(), a.nt_dev = counts.data_ptr<int>() + 1;
  a.w2 = (const unsigned char*)w13.data_ptr();
  const uint64_t rows = (uint64_t)w13.size(0) * 1152;
  a.tmw = tmap2d(w13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 2, rows, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B);
  a.tms = tmap2d(s13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 32, rows, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE);
  a.tmw2 = tmap2d(w13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 2, rows, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  a.tms2 = tmap2d(s13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 32, rows, 16, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
  Args<2>& b = f.a2;
  b.s2 = (const unsigned char*)s2.data_ptr(), b.ecs = (const EC*)ecs.data_ptr(), b.tiles = (const Tile*)t2.data_ptr();
  b.w2 = (const unsigned char*)w2.data_ptr();
  b.bimg = (const char*)himg.data_ptr(), b.nt_dev = counts.data_ptr<int>() + 2;
  b.yout = (__nv_bfloat16*)yout.data_ptr();
  b.tmw = tmap2d(w2.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, IR / 4, (uint64_t)w2.size(0) * D, 48, 128,
                 CU_TENSOR_MAP_SWIZZLE_NONE);
  b.tmw2 = tmap2d(w2.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, IR / 4, (uint64_t)w2.size(0) * D, 48, 256,
                  CU_TENSOR_MAP_SWIZZLE_NONE);
  f.done1 = sync.data_ptr<int>();
  f.g2ctr = f.done1 + mec;
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(sync.data_ptr(), 0, (size_t)(mec + 1) * sizeof(int), stream);
  static bool attr_dev[64] = {};
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  if (!attr_dev[cur_dev & 63]) {
    cudaFuncSetAttribute(k_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, fz::SMEM);
    attr_dev[cur_dev & 63] = true;
  }
  k_fused<<<g_sms, 32 + 32 * NCW, fz::SMEM, stream>>>(f);
}
torch::Tensor trace_get(int64_t mode) {  // [TRACE_MAX, 8] int64 of the last k_gemm<mode> run (MS4_TRACE builds)
  auto out = torch::zeros({TRACE_MAX, 8}, torch::kInt64);
  cudaMemcpyFromSymbol(out.data_ptr(), g_trace, sizeof(unsigned long long) * TRACE_MAX * 8,
                       sizeof(unsigned long long) * TRACE_MAX * 8 * (mode - 1));
  return out;
}
int64_t trace_on() { return MS4_TRACE; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("route", &route);
  m.def("route_scratch_ints", &route_scratch_ints);
  m.def("route_max_ec", &route_max_ec);
  m.def("router_top", &router_top);
  m.def("router_top_pack", &router_top_pack);
  m.def("router_top_pack2", &router_top_pack2);
  m.def("prepr2", &prepr2);
  m.def("prepr5", &prepr5);
  m.def("prepr6", &prepr6);
  m.def("swiglu", &swiglu);
  m.def("prepr", &prepr);
  m.def("gemm1r", &gemm1r);
  m.def("gemm1rs", &gemm1rs);
  m.def("gemm1r8", &gemm1r8);
  m.def("gemm2r", &gemm2r);
  m.def("gemm2rs", &gemm2rs);
  m.def("gemm2r8", &gemm2r8);
  m.def("combine", &combine);
  m.def("smem_bytes", &smem_bytes);
  m.def("trace_get", &trace_get);
  m.def("gemm12", &gemm12);
  m.def("trace_on", &trace_on);
}
