// moe_x2: C128 routed-expert GEMMs on a pre-placed FP4 expert layout. Two levers on top of moe_x
//  * X2_PRE (layout, same math): the expert bytes are bit-permuted in place once (preplace(): a byte -> byte map) so that
//    byte i of every 32-bit word holds its low nibble's e4m3 bits at {7, 4, 3, 2} (s, e1, e0, m) and its high nibble's at
//    {5, 6, 1, 0}; the FP4 -> e4m3 step is then w & 0x9c9c9c9c (low nibbles) and ((w << 2) & 0x8c8c8c8c) | ((w >> 2) &
//    0x10101010) (high nibbles), shifts as IMAD (FMA pipe): 3 LOP3 + 2 IMAD per word instead of 4 LOP3 + 2 IMAD + 1 SHF.
//    Products, order and outputs bitwise equal to moe_x on the raw layout.
//  * X2_W8 (W4A8, math-changing, test only): GEMM1 as fp8 wgmma m64nNk32 RS: the e4m3 bytes above are the A operand
//    directly (FP4 x 2^-6, exact), no cvt / HMUL2; the token image is e4m3 (gather_w8: the official act_quant numerics,
//    ue8m0 power-of-two scales, X2_ASC=1 per (token, 32-block) like inference/kernel.py act_quant(block 32), X2_ASC=0 one
//    per token); every k32 block is a fresh accumulator folded into the fp32 sum with its block scales (FFMA), so the
//    accumulation is fp32 (the tensor core sums 32 products at a time).
// The rest is moe_x verbatim (persistent CTAs, in-kernel plan, heaviest-first atomic queue, epilogues, GEMM2).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

#include <vector>

#include <cuda_fp8.h>

#include "wgmma_x.cuh"  // WgX<N>::mma: wgmma.mma_async m64nNk16 f32.f16.f16, A from registers, B descriptor low word
#include "wgmma_f8.cuh"  // WgF8<N>::mma / mmac: m64nNk32 f32.e4m3.e4m3
#include "../w8a8/act_quant.cuh"  // aq::scale_exp: THE activation quantization

#ifndef X2_PRE
#define X2_PRE 1  // expert weights in the pre-placed layout (preplace()); 0: raw checkpoint layout (moe_x)
#endif
#ifndef X2_W8
#define X2_W8 0  // GEMM1 W4A8 (fp8 wgmma, e4m3 token image from gather_w8); raw (X2_PRE=0, the served FP4 -> e4m3
                 // extraction, 1 ALU op more per word) or pre-placed weights: the same e4m3 bytes in the same K order
#endif
#ifndef X2_ASC
#define X2_ASC 1  // W4A8 activation scales: 1 per (token, 32-block) (official act_quant), 0 one per token (in ec.inv)
#endif
#ifndef X2_W8G2
#define X2_W8G2 0  // with X2_W8: GEMM1 writes H as e4m3 + per (token, 32-block) scales (act_quant of the w2 input) and
                   // GEMM2 runs W4A8 too (fp8 wgmma, SW64 k64 slabs of H); 0: f16 H image + moe_x's GEMM2
#endif
#ifndef X2_ABL
#define X2_ABL 0  // W4A8 timing ablations (results invalid): 1 no wgmma (dequant + fold only), 2 no fold (wgmma only),
                 // 3 no dequant (raw words as A), 4 no scale extraction (constant scales)
#endif
#ifndef X2_SFMA
#define X2_SFMA 1  // W4A8 GEMM1: block-scale floats on the FMA pipe: (u >> 8k) << 23 as IMAD.HI / IMAD (the stray sign bit
                   // is dropped by |x| in the FFMA), 2^(s - 127) instead of 2^(s - 121): the 2^6 goes into the epilogue
#endif
#ifndef X2_KREG
#define X2_KREG 1  // W4A8 GEMM1: shift multipliers from the kernel arguments, so ptxas keeps them as IMAD (FMA pipe)
                   // instead of turning them into SHF (ALU pipe, the binding one)
#endif
#ifndef X2_GS
#define X2_GS 1  // W4A8 GEMM1: k64 steps per wgmma group (2: 4 wgmma per fence / commit / wait, n_pad <= 16 only;
                // register pressure)
#endif
#ifndef X2_FASTEPI
#define X2_FASTEPI 0  // W4A8 GEMM1 epilogue: SiLU with __expf / __fdividef (else the served expf + IEEE division)
#endif
#ifndef X2_G2CH
#define X2_G2CH 9  // W4A8 GEMM2: k64 steps per stage for n_pad <= 32 (9 = whole K, 2 ring stages; 3 = 3 per stage, more
                   // stages in flight)
#endif
#ifndef X2_SLDS
#define X2_SLDS 0  // W4A8 GEMM1: block-scale bytes read per k64 step from SMEM (2 x 16 bit) instead of 8 registers per stage
#endif
#ifndef X2_LOOP
#define X2_LOOP 0  // W4A8 GEMM1: the stage as a runtime loop of 2-step trips (small code: i-cache test) instead of the
                   // fully unrolled CH steps; scale words read per trip from SMEM, 2 wgmma groups in flight per trip
#endif
#ifndef X2_PSK
#define X2_PSK 1  // W4A8 GEMM1 with MX_KS1 (16 consumer warps, 112 registers): groups in flight (1 or 2)
#endif
#ifndef X2_PROF
#define X2_PROF 0  // W4A8 GEMM1: per consumer warp clock64 cycles waiting for stages / for wgmma groups / total, summed
                   // into Args::prof [4] (u64: full-wait, wgmma-wait, total, k64 steps) and %globaltimer ns into [4]
#endif
#ifndef X2_PS
#define X2_PS 2  // W4A8: k64 steps (groups of 2 fresh k32 accumulators) in flight before folding (1 or 2; 1 above n 32)
#endif
#ifndef MX_NWG1
#define MX_NWG1 3  // GEMM1 consumer warpgroups (2 or 3): items of 32 x NWG1 intermediates (64 x NWG1 weight rows)
#endif
#ifndef MX_NWG2
#define MX_NWG2 4  // GEMM2 consumer warpgroups (2 or 4): items of 64 x NWG2 output rows (4: measured best, g-21 / g-21d)
#endif
#ifndef MX_KS1
#define MX_KS1 0  // GEMM1 with 16 consumer warps: 4 warpgroups = 2 pairs x 2 row warpgroups (64-intermediate items); pair p
                  // takes the item's stages of parity p (each pair owns half of an even ring), pair 1 hands its fp32
                  // partials to pair 0 through SMEM (fixed order). The producer is a whole warpgroup (setmaxnreg 32,
                  // consumers 112: exactly the launch allocation 96 x 640; more makes setmaxnreg.inc spin forever),
                  // else 17 warps cap every thread at 96 registers
#endif
#ifndef MX_DYN
#define MX_DYN 1  // 1: atomic item queue (heaviest first); 0: static snake over the same order
#endif
#ifndef MX_B1
#define MX_B1 16384  // GEMM1 token bytes per stage (sets the k-depth per n_pad)
#endif
#ifndef MX_NS2
#define MX_NS2 16  // GEMM2: n_pad <= this takes the whole K (9 k64) in one stage, larger 3 stages of 3
#endif
#ifndef MX_U1
#define MX_U1 1  // GEMM1 stages per (unrolled) loop trip: inside a trip a stage is freed once the next stage's second
                 // wgmma group is issued (wait_group 1 proves its groups done); the trip ends with wait_group 0. Groups in
                 // flight across the runtime loop's back edge make ptxas serialize every wgmma (C7513), hence trips.
#endif
#ifndef MX_U2
#define MX_U2 1  // GEMM2 stages per loop trip (only used when it divides the item's stage count, else 1)
#endif
#ifndef MX_SPOL
#define MX_SPOL 1  // GEMM1 scale boxes with L2 evict_normal (the next stage reads the rest of the 32-B sectors)
#endif
#ifndef MX_S1
#define MX_S1 0  // GEMM1 ring stages (0 = as many as fit)
#endif
#ifndef MX_S2
#define MX_S2 0  // GEMM2 ring stages (0 = as many as fit)
#endif
#ifndef MX_MMA
#define MX_MMA 0  // 1: mma.sync m16n8k16 (moe_v4's per-warp B fragments and accumulation order) for n_pad <= MX_MMAN,
                  //    RS-wgmma above: the same arithmetic as moe_v4 (bitwise reference check); 0: RS-wgmma for every n_pad
#endif
#ifndef MX_MMAN
#define MX_MMAN 32
#endif
#ifndef MX_MMA1
#define MX_MMA1 MX_MMA  // per GEMM: GEMM1 path
#endif
#ifndef MX_MMA2
#define MX_MMA2 1  // per GEMM: GEMM2 path; default mma.sync (single-stage items drain the wgmma pipe before every
                   // epilogue: wgmma GEMM2 measured 10-20 % slower); bitwise equal to moe_v4's GEMM2 on the same H
#endif
#ifndef MX_WG
#define MX_WG 1  // wgmma groups (k64 steps) left in flight after each step (wait_group MX_WG); fresh A registers per step
#endif
#ifndef MX_UNIF
#define MX_UNIF 1  // B descriptors from a lane-0 broadcast (uniform datapath); 0: per-thread values (R2UR per wgmma)
#endif
#ifndef MX_SW
#define MX_SW 16  // stream mode GEMM1 (gemm1s): consumer warps (K slices: divides 80)
#endif
#ifndef MX_SW2
#define MX_SW2 15  // stream mode GEMM2 (gemm2s): consumer warps = units per stage (16 + producer would cap at 96 registers:
                   // registers are per SM sub-partition, 5 warps on one of them)
#endif
#ifndef MX_SPLIT1
#define MX_SPLIT1 1  // stream GEMM1: CTA ranges at K-half granularity (a boundary unit's two K halves run on neighbouring
                     // CTAs; the second to finish adds half 0 + half 1 and runs the epilogue): ~3 % imbalance instead of 7
#endif
#ifndef MX_RED1
#define MX_RED1 1  // stream GEMM1: K-slice partial buffers (units), so warps may drift MX_RED1 - 1 units apart around the
                   // rotating reducer; 3 measured no better at C1 (g-21e), so 1
#endif
#ifndef MX_IMGG
#define MX_IMGG 0  // stream GEMM1: 1 = each warp loads its K slice's B fragments straight from the L2-resident image at
                   // chunk switches (frees the 80 KB SMEM image buffer: a 4th ring stage); measured slower at C1 (39 vs 35
                   // us: the loads are exposed at every chunk switch), so 0
#endif
#ifndef MX_PDL
#define MX_PDL 1  // GEMM2 is a programmatic dependent launch of GEMM1 (GEMM1 triggers at once): GEMM2's weights stream
                 // during GEMM1's tail, its H copies wait (griddepcontrol.wait) for GEMM1's completion
#endif
#ifndef MX_NOSTREAM
#define MX_NOSTREAM 0  // timing ablation: the producer copies only the EC records (stale stage data; results invalid)
#endif
#ifndef MX_NOW
#define MX_NOW 0  // timing ablation: no weight / scale copies (results invalid)
#endif
#ifndef MX_NOMMA
#define MX_NOMMA 0  // timing ablation: no wgmma (dequant only; results invalid)
#endif

namespace mx {  // (own extension module: the names do not clash with moe_x)
constexpr int D = 5120, IR = 576, NMAX = 64, MAXEC = 512, SMEM_MAX = 227 * 1024, ECB = 800;

struct EC {  // = moe_v4 / route2 EC record
  int e, n_real, n_pad, pad;
  long long img_off, h_off;
  int tok[NMAX];
  float wt[NMAX];
  float inv[NMAX];
};
static_assert(sizeof(EC) == ECB, "EC layout");
struct Hdr {  // per stage, written by the producer
  int item, npad, nch, q, slot, ec, ch, pad;
};

template <int MODE>
struct Cfg;
template <>
struct Cfg<1> {  // GEMM1: K = 5120 = 80 k64 steps; rows: per warpgroup 32 gate + 32 up (interleaved per warp, as moe_v4)
  static constexpr int KSP = MX_KS1 ? 2 : 1;  // K-split pairs
  static constexpr int NWG = MX_KS1 ? 4 : MX_NWG1, RWG = NWG / KSP, NCW = 4 * NWG, BR = 32 * RWG, Q = IR / BR, KB = 80;
  static constexpr int XB = MX_KS1 ? 256 * 8 * 4 : (X2_W8 && X2_W8G2) ? 4 * 4 * NMAX * 4 : 0;  // pair hand-off / W8G2 amax
  static constexpr int PTH = MX_KS1 ? 128 : 32;        // producer threads
  static constexpr int OFF_B = 0, B_BYTES = MX_B1;
  static constexpr int OFF_A = B_BYTES, A_BYTES = 8 * 2 * BR * 32;  // 8 k64 x (gate + up) x BR rows x 32 B
  static constexpr int OFF_S = OFF_A + A_BYTES, S_BYTES = 2 * BR * 16;
  static constexpr int OFF_X = OFF_S + S_BYTES, X_BYTES = (X2_W8 && X2_ASC) ? 16 * NMAX * 4 : 0;  // W4A8 token scales
  static constexpr int OFF_H = OFF_X + X_BYTES;
  static constexpr int STAGE = (OFF_H + 64 + 1023) / 1024 * 1024;
  static constexpr int STG = 0;
  static constexpr int SREQ = MX_S1;
  // k64 steps per stage: the token part (f16: n x 128 B per k64; e4m3: n x 64 B) stays <= B_BYTES (W4A8: even, k128 slabs)
  __host__ __device__ static constexpr int ch(int n) {
    return X2_W8 ? (8 * n * 64 <= B_BYTES ? 8 : 4) : (8 * n * 128 <= B_BYTES ? 8 : 4 * n * 128 <= B_BYTES ? 4 : 2);
  }
  static_assert(IR % BR == 0 && B_BYTES % 1024 == 0 && B_BYTES >= 2 * NMAX * 128, "GEMM1 config");
  static_assert(!X2_W8 || (!MX_MMA1 && !(MX_KS1 && X2_W8G2)), "W4A8 GEMM1: wgmma");  // raw (X2_PRE=0) or pre-placed
  static_assert(!X2_SLDS || (X2_SFMA && X2_KREG), "X2_SLDS: the FMA-pipe scale path");
  static_assert(!X2_W8G2 || X2_W8, "W4A8 GEMM2 needs the W4A8 GEMM1 (e4m3 H)");
};
template <>
struct Cfg<2> {  // GEMM2: K = 576 = 9 k64 steps; rows: 64 per warpgroup
  static constexpr int KSP = 1, XB = 0, PTH = 32;
  static constexpr int NWG = MX_NWG2, RWG = NWG, NCW = 4 * NWG, BR = 64 * NWG, Q = D / BR, KB = 9;
  static constexpr int OFF_B = 0, B_BYTES = X2_W8G2 ? (X2_G2CH == 9 ? 9 * 32 * 64 : 3 * NMAX * 64)
                                                     : (9 * MX_NS2 > 3 * NMAX ? 9 * MX_NS2 : 3 * NMAX) * 128;
  static constexpr int OFF_A = (B_BYTES + 1023) / 1024 * 1024, A_BYTES = (X2_W8G2 && X2_G2CH == 3) ? BR * 96 : BR * (IR / 2);
  static constexpr int OFF_S = OFF_A + A_BYTES, S_BYTES = BR * 18;
  static constexpr int OFF_X = (OFF_S + S_BYTES + 15) / 16 * 16, X_BYTES = X2_W8G2 ? 20 * NMAX * 4 : 0;  // H block scales
  static constexpr int OFF_H = (OFF_X + X_BYTES + 15) / 16 * 16;
  static constexpr int STAGE = (OFF_H + 64 + 1023) / 1024 * 1024;
  static constexpr int STG = NCW * 512;  // per warp: bf16 Y staging [16 tokens][16 rows]
  static constexpr int SREQ = MX_S2;
  __host__ __device__ static constexpr int ch(int n) { return X2_W8G2 ? (n <= 32 ? X2_G2CH : 3) : (n <= MX_NS2 ? 9 : 3); }
  static_assert(D % BR == 0, "GEMM2 config");
};
struct Plan {  // in SMEM, identical in every CTA
  unsigned short order[MAXEC];  // EC indices, n_pad descending (stable)
  unsigned short cnt[16][8];    // per 32-EC chunk and bucket: count, then exclusive prefix over chunks
  int base[9];                  // bucket bases (bucket b = n_pad 64 - 8 b)
  int nec;
  int info[MAXEC];  // e | n_pad << 16
  int pre[MAXEC];   // slot base: img_off / 10240 = h_off / 1152
};
template <int MODE>
struct Lay {
  using C = Cfg<MODE>;
  static constexpr int FIX = 1024 + (int)sizeof(Plan) + C::STG + C::XB + 256 + ECB;
  static constexpr int SS0 = C::SREQ ? C::SREQ : (SMEM_MAX - FIX) / (C::STAGE + ECB), SS = SS0 / C::KSP * C::KSP;
  static constexpr int NSLOT = SS + 1;
  static constexpr int O_BAR = SS * C::STAGE, O_SLOT = O_BAR + 256, O_STG = O_SLOT + NSLOT * ECB,
                       O_XB = (O_STG + C::STG + 15) / 16 * 16, O_PLAN = O_XB + C::XB,
                       SMEM = O_PLAN + (int)sizeof(Plan) + 1024;
  static_assert(SS >= 2 && SMEM <= SMEM_MAX, "SMEM");
};

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
  asm volatile("{\n.reg .pred p;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W_%=;\n}\n" ::"r"(a),
               "r"(parity)
               : "memory");
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
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(mbar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t mbar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
      "l"(tm), "r"(c0), "r"(c1), "r"(mbar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}
__device__ __forceinline__ void ldsm_x4b(unsigned (&r)[4], uint32_t a) {
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
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
template <int K>
__device__ __forceinline__ void reg_fence(float (&r)[K]) {  // keeps reads of async wgmma results after the wait
#pragma unroll
  for (int i = 0; i < K; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {  // K-major, 128-B rows, 128B swizzle, SBO 1024
  return ((uint64_t(saddr) & 0x3FFFFull) >> 4) | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
}
// K order inside a 32-block: logical L (mma / image order) <-> physical k (raw nibble order), as moe_v4
__host__ __device__ __forceinline__ int kinv(int p) { return 16 * (p & 1) + 8 * ((p >> 2) & 1) + 2 * (p >> 3) + ((p >> 1) & 1); }
__device__ __forceinline__ unsigned swz(int n, int b) {
  return (n >> 3) * 1024 + (n & 7) * 128 + ((((b >> 4) ^ (n & 7)) & 7) << 4) + (b & 15);
}
__device__ __forceinline__ unsigned bias4(unsigned u) { return (u - 0x6a6a6a6au) << 2; }  // 4 scale bytes s -> (s - 106) << 2
// FP4 -> e4m3 bytes (x 2^-6, exact) of the low / high nibbles of the 4 bytes of a word
__device__ __forceinline__ unsigned shl2(unsigned w) {  // w << 2 as IMAD (FMA pipe)
  unsigned d;
  asm("mul.lo.u32 %0, %1, 4;" : "=r"(d) : "r"(w));
  return d;
}
__device__ __forceinline__ unsigned shr2(unsigned w) {  // w >> 2 as IMAD.HI (FMA pipe)
  unsigned d;
  asm("mul.hi.u32 %0, %1, 0x40000000;" : "=r"(d) : "r"(w));
  return d;
}
#if X2_PRE
__device__ __forceinline__ unsigned fp4_lo_e4m3(unsigned w) { return w & 0x9c9c9c9cu; }
__device__ __forceinline__ unsigned fp4_hi_e4m3(unsigned w) { return (shl2(w) & 0x8c8c8c8cu) | (shr2(w) & 0x10101010u); }
#else
__device__ __forceinline__ unsigned fp4_lo_e4m3(unsigned w) { return ((w << 4) & 0x80808080u) | ((w << 2) & 0x1c1c1c1cu); }
__device__ __forceinline__ unsigned fp4_hi_e4m3(unsigned w) { return (w & 0x80808080u) | ((w >> 2) & 0x1c1c1c1cu); }
#endif
__device__ __forceinline__ float pow2_ceil_f(float x) {  // 2^ceil(log2 x), x > 0 normal
  const unsigned u = __float_as_uint(x);
  return __uint_as_float((u + 0x7fffffu) & 0xff800000u);
}
__device__ __forceinline__ unsigned sw64(int n, int b) {  // [rows][64 B] slab, 64B swizzle (8-row atoms of 512 B)
  return (n >> 3) * 512 + (n & 7) * 64 + ((((b >> 4) ^ (n >> 1)) & 3) << 4) + (b & 15);
}
// raw checkpoint byte (low nibble = even k: s e1 e0 m) -> pre-placed byte: lo s,e1,e0,m -> 7,4,3,2; hi s,e1,e0,m -> 5,6,1,0
__host__ __device__ __forceinline__ unsigned preplace_byte(unsigned r) {
  return ((r >> 3) & 1) << 7 | ((r >> 2) & 1) << 4 | ((r >> 1) & 1) << 3 | (r & 1) << 2 | ((r >> 7) & 1) << 5 |
         ((r >> 6) & 1) << 6 | ((r >> 5) & 1) << 1 | ((r >> 4) & 1);
}
__device__ __forceinline__ void cvt_e4m3x4(unsigned v, unsigned& lo2, unsigned& hi2) {
  asm("{\n.reg .b16 l, h;\nmov.b32 {l, h}, %2;\ncvt.rn.f16x2.e4m3x2 %0, l;\ncvt.rn.f16x2.e4m3x2 %1, h;\n}"
      : "=r"(lo2), "=r"(hi2)
      : "r"(v));
}
__device__ __forceinline__ unsigned hmul2(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
// one k32 block of rows g (word wg) and g+8 (word wh) -> the A fragments of its two k16 steps (moe_v4 dequant_pair)
__device__ __forceinline__ void dequant_pair(unsigned (&f0)[4], unsigned (&f1)[4], unsigned wg, unsigned wh, unsigned mg,
                                             unsigned mh) {
  unsigned g0, g1, h0, h1;
  cvt_e4m3x4(fp4_lo_e4m3(wg), g0, g1);
  cvt_e4m3x4(fp4_lo_e4m3(wh), h0, h1);
  f0[0] = hmul2(g0, mg), f0[2] = hmul2(g1, mg), f0[1] = hmul2(h0, mh), f0[3] = hmul2(h1, mh);
  cvt_e4m3x4(fp4_hi_e4m3(wg), g0, g1);
  cvt_e4m3x4(fp4_hi_e4m3(wh), h0, h1);
  f1[0] = hmul2(g0, mg), f1[2] = hmul2(g1, mg), f1[1] = hmul2(h0, mh), f1[3] = hmul2(h1, mh);
}

// ------------------------------------------------------------------------------------------------ kernel
template <int MODE>
struct Args {
  const unsigned char* w;  // GEMM2: raw w2 (bulk copies)
  const unsigned char* s;  // GEMM2: raw s2 (bulk copies)
  const EC* ecs;
  const int* counts;  // counts[0] = number of EC records
  const char* bimg;   // GEMM1: token image; GEMM2: H image
  char* himg;         // GEMM1 output
  __nv_bfloat16* yout;  // GEMM2 output, bf16 [slots][D]
  int* ctr;             // MX_DYN queue counter (0 between launches)
  const float* asc;     // W4A8 GEMM1, X2_ASC=1: token block scales, per EC [160 blocks][n_pad] at slot base x 160
  char* h8;             // X2_W8G2: e4m3 H image, per EC 9 k64 slabs of [n_pad][64 B] (SW64) at slot base x 576
  float* hsc;           // X2_W8G2: H block scales, per EC [18 (20) blocks][n_pad] at slot base x 20
  unsigned long long* prof;  // X2_PROF
  unsigned km[8];            // X2_KREG multipliers: 4, 2^30, 2^23, 2^24, 2^16, 2^8
  alignas(64) CUtensorMap tmw;    // GEMM1: w13 u8 box 128 B x BR rows, SW128; GEMM2: w2 u16 box 48 x BR rows
  alignas(64) CUtensorMap tmw64;  // GEMM1: w13 u8 box 64 B x BR rows, SW64 (2 k64 per stage)
  alignas(64) CUtensorMap tms;    // GEMM1: s13 u8 box 16 B x BR rows
};

// every CTA: EC records -> plan (n_pad buckets, stable order)
__device__ __forceinline__ void build_plan(Plan& pl, const EC* __restrict__ ecs, const int* __restrict__ counts) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
  const int nec = min(counts[0], MAXEC);
  for (int k = warp; k < 16; k += nw) {
    const int x = 32 * k + lane;
    int b = 8;
    if (x < nec) {
      const int4 h = *reinterpret_cast<const int4*>(ecs + x);
      const long long io = ecs[x].img_off;
      if (h.z < 8 || h.z > NMAX || (h.z & 7)) __trap();
      pl.info[x] = h.x | (h.z << 16);
      pl.pre[x] = (int)(io / (80 * 128));
      b = 8 - h.z / 8;
    }
#pragma unroll
    for (int bb = 0; bb < 8; ++bb) {
      const unsigned m = __ballot_sync(~0u, b == bb);
      if (lane == 0) pl.cnt[k][bb] = (unsigned short)__popc(m);
    }
  }
  __syncthreads();
  if (tid < 8) {  // exclusive prefix over chunks per bucket; totals into base[] (temporarily)
    int run = 0;
    for (int k = 0; k < 16; ++k) {
      const int v = pl.cnt[k][tid];
      pl.cnt[k][tid] = (unsigned short)run;
      run += v;
    }
    pl.base[tid + 1] = run;
  }
  __syncthreads();
  if (tid == 0) {
    pl.base[0] = 0;
    for (int b = 1; b <= 8; ++b) pl.base[b] += pl.base[b - 1];
    pl.nec = nec;
  }
  __syncthreads();
  for (int k = warp; k < 16; k += nw) {
    const int x = 32 * k + lane;
    const int b = x < nec ? 8 - (pl.info[x] >> 16) / 8 : 8;
    unsigned mine = 0;
#pragma unroll
    for (int bb = 0; bb < 8; ++bb) {
      const unsigned m = __ballot_sync(~0u, b == bb);
      if (b == bb) mine = m;
    }
    if (b < 8) pl.order[pl.base[b] + pl.cnt[k][b] + __popc(mine & ((1u << lane) - 1u))] = (unsigned short)x;
  }
  __syncthreads();
}

template <int MODE>
__device__ __forceinline__ void producer(const Args<MODE>& a, char* ring, uint64_t* full, uint64_t* empty, char* slots,
                                         const Plan& pl) {
  using C = Cfg<MODE>;
  using L = Lay<MODE>;
  const uint64_t pw = pol_first(), pb = pol_last(), psc = MX_SPOL ? pol_normal() : pw;
  const int nitems = pl.nec * C::Q, G = gridDim.x, bx = blockIdx.x;
  long it = 0;
  int seq = 0, k = 0;
  if (MX_PDL && MODE == 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  // MX_PDL, GEMM2: launched as a programmatic dependent of GEMM1; the first stages' weights stream at once, their H
  // copies wait for GEMM1 (griddepcontrol.wait) and are issued before the producer first blocks on a free stage
  struct Pend {
    uint32_t dst, mb;
    const char* src;
    unsigned bytes;
  } pend[16];
  static_assert(L::SS <= 8, "deferred H copies");
  int npend = 0;
  bool waited = !(MX_PDL && MODE == 2);
  auto flush = [&]() {
    if (waited) return;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    waited = true;
    for (int i = 0; i < npend; ++i) bulk(pend[i].dst, pend[i].src, pend[i].bytes, pend[i].mb, pb);
    npend = 0;
  };
  auto grab = [&]() -> int {
    if (MX_DYN) return atomicAdd(a.ctr, 1);
    const int r = k++;
    return r * G + ((r & 1) ? G - 1 - bx : bx);
  };
  int item = grab();
  for (;;) {
    const bool end = item >= nitems;
    int next = 0;
    if (!end) next = grab();
    if (end && MX_DYN && item == nitems + G - 1) {  // every CTA has made its last grab: reset (completed at L2 before
      const int old = atomicExch(a.ctr, 0);         // this CTA exits: a PDL-launched next GEMM2 grabs from 0)
      asm volatile("" ::"r"(old));
      __threadfence();
    }
    if (end) {
      flush();
      for (int pp = 0; pp < C::KSP; ++pp, ++it) {  // one end marker per consumer pair (stage parity)
        const int s = it % L::SS;
        if (it >= L::SS) mbar_wait(smem_u32(empty + s), ((it / L::SS) + 1) & 1);
        Hdr* h = (Hdr*)(ring + (size_t)s * C::STAGE + C::OFF_H);
        h->item = -1;
        mbar_arrive(smem_u32(full + s));
      }
      return;
    }
    const int p = item / C::Q, q = item - p * C::Q, x = pl.order[p];
    const int e = pl.info[x] & 0xffff, npad = pl.info[x] >> 16;
    const long long pre = pl.pre[x];
    const int CH = C::ch(npad), nch = C::KB / CH, slot = seq % L::NSLOT;
    ++seq;
    for (int ch = 0; ch < nch; ++ch, ++it) {
      const int s = it % L::SS;
      if (it >= L::SS) {
        flush();
        mbar_wait(smem_u32(empty + s), ((it / L::SS) + 1) & 1);
      }
      char* st = ring + (size_t)s * C::STAGE;
      Hdr* h = (Hdr*)(st + C::OFF_H);
      h->item = item, h->npad = npad, h->nch = nch, h->q = q, h->slot = slot, h->ec = x, h->ch = ch;
      const uint32_t fb = smem_u32(full + s), dst = smem_u32(st);
      const unsigned bE = ch == 0 ? (unsigned)ECB : 0u;
      if (MX_NOSTREAM) {
        mbar_expect_tx(fb, bE);
        if (bE) bulk(smem_u32(slots + slot * ECB), a.ecs + x, ECB, fb, pb);
        continue;
      }
      const unsigned bB = (unsigned)(CH * npad * (((MODE == 1 && X2_W8) || (MODE == 2 && X2_W8G2)) ? 64 : 128));
      const unsigned bX = (MODE == 1 && X2_W8 && X2_ASC) || (MODE == 2 && X2_W8G2) ? (unsigned)(2 * CH * npad * 4) : 0u;
      unsigned bA, bS;
      if (MODE == 1) {
        bA = MX_NOW ? 0u : (unsigned)(CH * 2 * C::BR * 32), bS = MX_NOW ? 0u : (unsigned)C::S_BYTES;
      } else {
        bA = MX_NOW ? 0u : (unsigned)(C::BR * CH * 32), bS = MX_NOW ? 0u : (unsigned)C::S_BYTES;
      }
      mbar_expect_tx(fb, bA + bS + bB + bX + bE);
      if (bE) bulk(smem_u32(slots + slot * ECB), a.ecs + x, ECB, fb, pb);
      if (MODE == 1) {
        if (!MX_NOW) {
          const int rg = e * 1152 + C::BR * q, ru = rg + IR;
          if (CH == 8) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const int c0 = ch * 256 + hh * 128;
              tma2d(dst + C::OFF_A + (unsigned)((hh * 2 + 0) * C::BR * 128), &a.tmw, c0, rg, fb, pw);
              tma2d(dst + C::OFF_A + (unsigned)((hh * 2 + 1) * C::BR * 128), &a.tmw, c0, ru, fb, pw);
            }
          } else if (CH == 4) {
            tma2d(dst + C::OFF_A, &a.tmw, ch * 128, rg, fb, pw);
            tma2d(dst + C::OFF_A + (unsigned)(C::BR * 128), &a.tmw, ch * 128, ru, fb, pw);
          } else {
            tma2d(dst + C::OFF_A, &a.tmw64, ch * 64, rg, fb, pw);
            tma2d(dst + C::OFF_A + (unsigned)(C::BR * 64), &a.tmw64, ch * 64, ru, fb, pw);
          }
          const int cs = (ch * CH * 2) & ~15;
          tma2d(dst + C::OFF_S, &a.tms, cs, rg, fb, psc);
          tma2d(dst + C::OFF_S + (unsigned)(C::BR * 16), &a.tms, cs, ru, fb, psc);
        }
        if (X2_W8) {  // e4m3 image: per EC 40 k128 slabs of [n_pad][128 B]; CH / 2 slabs per stage
          bulk(dst + C::OFF_B, a.bimg + pre * 5120 + (long long)ch * (CH / 2) * npad * 128, bB, fb, pb);
          if (X2_ASC) bulk(dst + C::OFF_X, a.asc + pre * 160 + (long long)ch * 2 * CH * npad, (unsigned)(2 * CH * npad * 4), fb, pb);
        } else {
          bulk(dst + C::OFF_B, a.bimg + pre * (80 * 128) + (long long)ch * CH * npad * 128, bB, fb, pb);
        }
      } else {
        if (!MX_NOW) {
          const long long rr = (long long)e * D + C::BR * q;
          if (CH == 9)
            bulk(dst + C::OFF_A, a.w + rr * (IR / 2), (unsigned)(C::BR * (IR / 2)), fb, pw);
          else
            tma2d(dst + C::OFF_A, &a.tmw, ch * 48, (int)rr, fb, pw);
          bulk(dst + C::OFF_S, a.s + rr * 18, (unsigned)C::S_BYTES, fb, pw);
        }
        const char* hsrc = X2_W8G2 ? (const char*)a.h8 + pre * 576 + (long long)ch * CH * npad * 64
                                   : a.bimg + pre * (9 * 128) + (long long)ch * CH * npad * 128;
        const char* xsrc = (const char*)(a.hsc + pre * 20 + (long long)ch * 2 * CH * npad);
        if (waited) {
          bulk(dst + C::OFF_B, hsrc, bB, fb, pb);
          if (bX) bulk(dst + C::OFF_X, xsrc, bX, fb, pb);
        } else {
          pend[npend++] = Pend{dst + (uint32_t)C::OFF_B, fb, hsrc, bB};
          if (bX) pend[npend++] = Pend{dst + (uint32_t)C::OFF_X, fb, xsrc, bX};
        }
      }
    }
    item = next;
  }
}

// one stage of an item on the consumer side: CH k64 steps of dequant + 4 wgmma each
template <int MODE, int N, int CH, bool REL, bool MMA>
__device__ __forceinline__ void stage_body(const char* st, int ch, float (&acc)[2][N / 2], int& pend, uint64_t* empty, int c,
                                           int t) {
  using C = Cfg<MODE>;
  const int w = t >> 5, lane = t & 31, g = lane >> 2, lr = lane & 7, mi = lane >> 3;
  const uint32_t aB = smem_u32(st), aA = aB + C::OFF_A;
  // B descriptor start-address field of slab 0 (16-B units). MX_UNIF: broadcast from lane 0 so ptxas sees a warp-uniform
  // value and builds every wgmma descriptor with UIADD3 / UMOV on the uniform datapath (else 4 vector ops per wgmma)
  const unsigned dlo0 = ((aB + C::OFF_B) & 0x3FFFFu) >> 4;
  const unsigned dlo = MX_UNIF ? __shfl_sync(~0u, dlo0, 0) : dlo0;
  // ---- scale bytes of this thread's rows for the stage (GEMM1: biased words; GEMM2: per k64 step below)
  unsigned sg[4] = {0, 0, 0, 0}, sh[4] = {0, 0, 0, 0};
  if (MODE == 1) {
    const int rr = 32 * c + 8 * w + g, soff = (ch * CH * 2) & 15;
    const char* sp = st + C::OFF_S + rr * 16 + soff;
    if (CH == 8) {
      const uint4 vg = *(const uint4*)sp, vh = *(const uint4*)(sp + C::BR * 16);
      sg[0] = bias4(vg.x), sg[1] = bias4(vg.y), sg[2] = bias4(vg.z), sg[3] = bias4(vg.w);
      sh[0] = bias4(vh.x), sh[1] = bias4(vh.y), sh[2] = bias4(vh.z), sh[3] = bias4(vh.w);
    } else if (CH == 4) {
      const uint2 vg = *(const uint2*)sp, vh = *(const uint2*)(sp + C::BR * 16);
      sg[0] = bias4(vg.x), sg[1] = bias4(vg.y), sh[0] = bias4(vh.x), sh[1] = bias4(vh.y);
    } else {
      sg[0] = bias4(*(const unsigned*)sp), sh[0] = bias4(*(const unsigned*)(sp + C::BR * 16));
    }
  }
  auto words = [&](unsigned (&wv)[4], int kb) {
    if (MODE == 1) {
      const int m = mi & 1, r = 32 * c + 8 * w + lr;
      if (CH >= 4) {
        const int h = kb >> 2, chunk = 2 * (kb & 3) + (mi >> 1);
        ldsm_x4(wv, aA + (unsigned)(((h * 2 + m) * C::BR + r) * 128 + ((chunk ^ (r & 7)) << 4)));
      } else {
        const int chunk = 2 * kb + (mi >> 1);
        ldsm_x4(wv, aA + (unsigned)((m * C::BR + r) * 64 + ((chunk ^ ((r >> 1) & 3)) << 4)));
      }
    } else {
      const int r = 64 * c + 16 * w + 8 * (mi & 1) + lr;
      ldsm_x4(wv, aA + (unsigned)(r * (CH == 9 ? IR / 2 : 96) + 32 * kb + 16 * (mi >> 1)));
    }
  };
  unsigned wv[2][4];
  words(wv[0], 0);
#pragma unroll
  for (int kb = 0; kb < CH; ++kb) {
    if (kb + 1 < CH) words(wv[(kb + 1) & 1], kb + 1);
    unsigned mg[2], mh[2];
    if (MODE == 1) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int bi = 2 * kb + j;  // byte of the stage's scale bytes (compile-time)
        mg[j] = __byte_perm(sg[bi >> 2], 0, 0x0404 + 0x1010 * (bi & 3));
        mh[j] = __byte_perm(sh[bi >> 2], 0, 0x0404 + 0x1010 * (bi & 3));
      }
    } else {
      const int blk0 = 2 * (ch * CH + kb), RR = 64 * c + 16 * w + g;
      const unsigned short bgs = *(const unsigned short*)(st + C::OFF_S + RR * 18 + blk0);
      const unsigned short bhs = *(const unsigned short*)(st + C::OFF_S + (RR + 8) * 18 + blk0);
      const unsigned u = bias4((unsigned)bgs | ((unsigned)bhs << 16));  // g blk0, g blk1, g+8 blk0, g+8 blk1
      mg[0] = __byte_perm(u, 0, 0x0404), mg[1] = __byte_perm(u, 0, 0x1414);
      mh[0] = __byte_perm(u, 0, 0x2424), mh[1] = __byte_perm(u, 0, 0x3434);
    }
    const unsigned* cw = wv[kb & 1];
    if constexpr (MMA) {  // moe_v4 run_tile's mma.sync path, verbatim arithmetic
      unsigned bf[N / 8][2][4];
      const uint32_t slab = aB + C::OFF_B + kb * N * 128;
#pragma unroll
      for (int nb = 0; nb < N / 8; ++nb)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) ldsm_x4b(bf[nb][kk], slab + swz(nb * 8 + (lane & 7), (kk * 4 + (lane >> 3)) * 16));
      if (REL && kb == 1) {
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_u32(empty + pend));
        pend = -1;
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        unsigned f0[4], f1[4];
        dequant_pair(f0, f1, cw[2 * j], cw[2 * j + 1], mg[j], mh[j]);
        if (MX_NOMMA) {
          acc[0][0] += __uint_as_float(f0[0] ^ f0[1] ^ f0[2] ^ f0[3] ^ f1[0] ^ f1[1] ^ f1[2] ^ f1[3] ^ bf[0][j][0]);
          continue;
        }
#pragma unroll
        for (int nb = 0; nb < N / 8; ++nb) {
          mma16816(&acc[0][nb * 4], f0, bf[nb][j][0], bf[nb][j][1]);
          mma16816(&acc[1][nb * 4], f1, bf[nb][j][2], bf[nb][j][3]);
        }
      }
      continue;
    }
    // fresh A registers per k64 step (a new value for the compiler, as moe_v4's RS path): the groups in flight keep
    // theirs until a wait retires them; wait_group 1 retires the group of two steps ago
    unsigned Af[4][4];
    dequant_pair(Af[0], Af[1], cw[0], cw[1], mg[0], mh[0]);
    dequant_pair(Af[2], Af[3], cw[2], cw[3], mg[1], mh[1]);
    wg_wait<MX_WG>();
    if (REL && kb == MX_WG) {  // every group of the previous stage (same loop trip) is done: free it
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(empty + pend));
      pend = -1;
    }
    if (MX_NOMMA) {
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[0][j % (N / 2)] += __uint_as_float(Af[j][0] ^ Af[j][1] ^ Af[j][2] ^ Af[j][3]);
      continue;
    }
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) WgX<N>::mma(acc[0], Af[j], dlo + (unsigned)(kb * N * 8 + 2 * j));
    wg_commit();
  }
}

template <int MODE, int N>
__device__ __forceinline__ void run_item(const Args<MODE>& a, char* ring, uint64_t* full, uint64_t* empty, long& it, const Hdr h0,
                                      int c, int t, const char* slots, char* stgb, int pr, float* xb) {
  using C = Cfg<MODE>;
  using L = Lay<MODE>;
  constexpr int CH = C::ch(N), NCH = C::KB / CH;
  const int w = t >> 5, lane = t & 31, g = lane >> 2, tq = lane & 3;
  constexpr bool MMA = (MODE == 1 ? MX_MMA1 : MX_MMA2) && N <= MX_MMAN;
  float acc[2][N / 2];  // [1]: mma.sync's second chain (odd k16 steps), added at the end as moe_v4
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[0][i] = 0.f, acc[1][i] = 0.f;
  constexpr int KSP = C::KSP;
  constexpr int U0 = MODE == 1 ? MX_U1 : MX_U2, U = (KSP == 1 && NCH % U0 == 0 && CH > MX_WG) ? U0 : 1;
  static_assert(NCH % KSP == 0, "K-split pairs need an even stage count");
  int pend = -1;
  for (int lc = pr; lc < NCH; lc += U * KSP) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int s = it % L::SS;
      if (lc + u > pr) mbar_wait(smem_u32(full + s), (it / L::SS) & 1);
      const char* st = ring + (size_t)s * C::STAGE;
      if (u == 0)
        stage_body<MODE, N, CH, false, MMA>(st, lc + u, acc, pend, empty, c, t);
      else
        stage_body<MODE, N, CH, true, MMA>(st, lc + u, acc, pend, empty, c, t);
      pend = s;
      it += KSP;
    }
    wg_wait<0>();
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_u32(empty + pend));
  }
  if (MMA) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[0][i] += acc[1][i];
  }
  if constexpr (KSP > 1) {  // pair 1 -> pair 0: sum = pair 0 + pair 1 per element, 8 floats per thread per round
    const int tp = c * 128 + t;
#pragma unroll
    for (int r0 = 0; r0 < N / 2; r0 += 8) {
      if (pr == 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (r0 + i < N / 2) xb[i * 256 + tp] = acc[0][r0 + i];
      }
      asm volatile("bar.sync 1, 512;" ::: "memory");
      if (pr == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (r0 + i < N / 2) acc[0][r0 + i] += xb[i * 256 + tp];
      }
      asm volatile("bar.sync 1, 512;" ::: "memory");
    }
    if (pr == 1) return;
  }
  const EC* ec = (const EC*)(slots + h0.slot * ECB);
  if (MODE == 1) {  // h of intermediate i = BR q + il -> GEMM2's K, at its logical position in slab i / 64 (moe_v4 epilogue)
    const int i = C::BR * h0.q + 32 * c + 8 * w + g, i6 = i & 63, pos = 32 * (i6 >> 5) + kinv(i6 & 31);
    char* hs = a.himg + ec->h_off + (size_t)(i >> 6) * N * 128;
#pragma unroll
    for (int nb = 0; nb < N / 8; ++nb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nb + 2 * tq + hh;
        const int tok = ec->tok[n];
        float hv = 0.f;
        if (tok >= 0) {
          const float iv = ec->inv[n];
          float gt = fminf(acc[0][4 * nb + hh] * iv, 10.f);
          float up = fminf(fmaxf(acc[0][4 * nb + 2 + hh] * iv, -10.f), 10.f);
          hv = gt / (1.f + expf(-gt)) * up * ec->wt[n] * 256.f;
        }
        *(__half*)(hs + swz(n, pos * 2)) = __float2half_rn(hv);
      }
  } else {  // bf16 Y per (token, slot): per-warp staging [16 tokens][16 rows], 16-B stores (moe_v4 MS4_YWARP layout)
    __nv_bfloat16* stg = (__nv_bfloat16*)stgb + (4 * c + w) * 256;
    const int row0 = C::BR * h0.q + 64 * c + 16 * w, eslot = ec->pad, enr = ec->n_real;
#pragma unroll
    for (int t0 = 0; t0 < N; t0 += 16) {
      __syncwarp();
#pragma unroll
      for (int nb = t0 / 8; nb < t0 / 8 + 2 && nb < N / 8; ++nb)
#pragma unroll
        for (int rs = 0; rs < 2; ++rs)
#pragma unroll
          for (int jj = 0; jj < 2; ++jj)
            stg[(8 * nb + 2 * tq + jj - t0) * 16 + g + 8 * rs] = __float2bfloat16_rn(acc[0][4 * nb + 2 * rs + jj] * (1.f / 256.f));
      __syncwarp();
      const int n = t0 + (lane >> 1);
      if (n < N && n < enr)
        *(uint4*)(a.yout + (size_t)(eslot + n) * D + row0 + 8 * (lane & 1)) = *(const uint4*)(stg + (lane >> 1) * 16 + 8 * (lane & 1));
    }
  }
}


// ---- W4A8 GEMM1: one stage = CH k64 steps; per step the 4 words of rows (gate g, up g) x 2 k32 blocks -> 2 A fragments
// (e4m3: reg 0 / 2 = gate row, k 4 tq.. / 16 + 4 tq.., reg 1 / 3 = up row) -> 2 fresh m64nNk32 wgmma (P), folded one
// step later: acc += P x 2^(s - 121) (x the token's block scale with X2_ASC=1). Logical k of a 32-block <-> raw nibble
// order: L < 16 -> 2 L (low nibbles), L >= 16 -> 2 (L - 16) + 1 (high nibbles); gather_w8 writes the image in that order.
template <int N>
struct W8 {  // GEMM2 (X2_W8G2): accumulator sets in flight
  static constexpr int PS = N <= 32 ? X2_PS : 1;
};
template <int N>
struct W8G1 {  // GEMM1: k64 steps per wgmma group (GS), k32 blocks per group (NB = 2 GS), groups in flight (PS)
  static constexpr int GS = (X2_GS == 2 && N <= 16) ? 2 : 1, NB = 2 * GS;
  static constexpr int PS = N <= 32 ? X2_PS : 1;
  static_assert(!MX_KS1, "W4A8 GEMM1: 12 consumer warps (MX_KS1 measured no better)");
};
template <int N, int CH>
__device__ __forceinline__ void stage_w8(const Args<1>& a, const char* st, int ch, float (&acc)[N / 2],
                                         float (&P)[W8G1<N>::PS][W8G1<N>::NB][N / 2], int c, int t, long long& tdep) {
  using C = Cfg<1>;
  constexpr int PS = W8G1<N>::PS, GS = W8G1<N>::GS, NB = W8G1<N>::NB, NG = CH / GS;
  static_assert(CH % GS == 0, "groups");
  const int w = t >> 5, lane = t & 31, g = lane >> 2, tq = lane & 3, lr = lane & 7, mi = lane >> 3;
  const uint32_t aB = smem_u32(st), aA = aB + C::OFF_A;
  const unsigned dlo0 = ((aB + C::OFF_B) & 0x3FFFFu) >> 4;
  const unsigned dlo = MX_UNIF ? __shfl_sync(~0u, dlo0, 0) : dlo0;
  unsigned sg[4] = {0, 0, 0, 0}, sh[4] = {0, 0, 0, 0};  // raw scale bytes of rows gate g / up g, the stage's 2 CH blocks
  const char* spg = st + C::OFF_S + (32 * c + 8 * w + g) * 16 + ((ch * CH * 2) & 15);  // X2_SLDS: per-step reads
  if (!X2_SLDS) {
    const int rr = 32 * c + 8 * w + g, soff = (ch * CH * 2) & 15;
    const char* sp = st + C::OFF_S + rr * 16 + soff;
    if (CH == 8) {
      const uint4 vg = *(const uint4*)sp, vh = *(const uint4*)(sp + C::BR * 16);
      sg[0] = vg.x, sg[1] = vg.y, sg[2] = vg.z, sg[3] = vg.w, sh[0] = vh.x, sh[1] = vh.y, sh[2] = vh.z, sh[3] = vh.w;
    } else {
      const uint2 vg = *(const uint2*)sp, vh = *(const uint2*)(sp + C::BR * 16);
      sg[0] = vg.x, sg[1] = vg.y, sh[0] = vh.x, sh[1] = vh.y;
    }
  }
  // scale byte of block bi -> float: X2_SFMA 2^(s - 127) with a stray sign bit (used as |x|), else 2^(s - 121)
  auto sfl = [&](const unsigned(&sv)[4], int bi) -> float {
    const unsigned u = sv[bi >> 2];
    const int k = bi & 3;
    if (X2_SFMA) {
      unsigned x = u;
      if (k) x = X2_KREG ? __umulhi(u, a.km[3 + k]) : u >> (8 * k);  // u >> 8 k (km[4] = 2^24, [5] = 2^16, [6] = 2^8)
      return __uint_as_float(X2_KREG ? x * a.km[2] : x << 23);
    }
    const unsigned b = __byte_perm(u, 0, 0x4440 + k);
    return __uint_as_float(b * (1u << 23) + (6u << 23));
  };
  auto lo8 = [&](unsigned wd) { return fp4_lo_e4m3(wd); };
  auto hi8 = [&](unsigned wd) -> unsigned {
    if (X2_PRE && X2_KREG) return ((wd * a.km[0]) & 0x8c8c8c8cu) | (__umulhi(wd, a.km[1]) & 0x10101010u);
    return fp4_hi_e4m3(wd);
  };
  auto words = [&](unsigned(&wv)[4], int kb) {
    const int m = mi & 1, r = 32 * c + 8 * w + lr, h = kb >> 2, chunk = 2 * (kb & 3) + (mi >> 1);
    ldsm_x4(wv, aA + (unsigned)(((h * 2 + m) * C::BR + r) * 128 + ((chunk ^ (r & 7)) << 4)));
  };
  // acc += Q[j] x |block scale| (x the token block scale, X2_ASC=1) for the NB blocks j0 .. j0 + NB - 1
  auto fold = [&](const float(&Q)[NB][N / 2], const float* fg, const float* fh, int j0) {
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      float tsc[N / 4];
      if (X2_ASC) {
        const float* xs = (const float*)(st + C::OFF_X) + (j0 + j) * N + 2 * tq;
#pragma unroll
        for (int nb = 0; nb < N / 8; ++nb) {
          const float2 v = *(const float2*)(xs + 8 * nb);
          tsc[2 * nb] = v.x, tsc[2 * nb + 1] = v.y;
        }
      }
      if (X2_ABL == 2) {  // no fold: one add per k32 block
        acc[j % (N / 2)] += Q[j][j % (N / 2)] + fg[j] + fh[j];
        continue;
      }
#pragma unroll
      for (int i = 0; i < N / 2; ++i) {
        float sc = fabsf((i & 2) ? fh[j] : fg[j]);
        if (X2_ASC) sc *= tsc[2 * (i >> 2) + (i & 1)];
        acc[i] = fmaf(Q[j][i], sc, acc[i]);
      }
    }
  };
  auto waitp = [&](float(&Q)[NB][N / 2], int left) {
    const long long t0 = X2_PROF ? clock64() : 0;
    if (left) wg_wait<1>(); else wg_wait<0>();
#pragma unroll
    for (int j = 0; j < NB; ++j) reg_fence(Q[j]);
    if (X2_PROF) tdep += clock64() - t0;
  };
  unsigned wv[2][4];
  words(wv[0], 0);
  float pg[NB], ph[NB];
#pragma unroll
  for (int j = 0; j < NB; ++j) pg[j] = 0.f, ph[j] = 0.f;
#pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    unsigned A[NB][4];
    float cg[NB], chh[NB];
#pragma unroll
    for (int ss = 0; ss < GS; ++ss) {
      const int kb = gi * GS + ss;
      if (kb + 1 < CH) words(wv[(kb + 1) & 1], kb + 1);
      const unsigned* cw = wv[kb & 1];
      if (X2_ABL == 3) {
#pragma unroll
        for (int q = 0; q < 4; ++q) A[2 * ss][q] = cw[q] & 0x7f7f7f7fu, A[2 * ss + 1][q] = cw[q] & 0x1f1f1f1fu;
      } else {
        A[2 * ss][0] = lo8(cw[0]), A[2 * ss][1] = lo8(cw[1]), A[2 * ss][2] = hi8(cw[0]), A[2 * ss][3] = hi8(cw[1]);
        A[2 * ss + 1][0] = lo8(cw[2]), A[2 * ss + 1][1] = lo8(cw[3]), A[2 * ss + 1][2] = hi8(cw[2]), A[2 * ss + 1][3] = hi8(cw[3]);
      }
      if (X2_SLDS) {  // the 2 blocks' bytes of rows gate g / up g: byte 0 -> << 23 (|x| drops byte 1's bit), byte 1 -> >> 8, << 23
        const unsigned ug = *(const unsigned short*)(spg + 2 * kb), uh = *(const unsigned short*)(spg + C::BR * 16 + 2 * kb);
        cg[2 * ss] = __uint_as_float(ug * a.km[2]), cg[2 * ss + 1] = __uint_as_float(__umulhi(ug, a.km[4]) * a.km[2]);
        chh[2 * ss] = __uint_as_float(uh * a.km[2]), chh[2 * ss + 1] = __uint_as_float(__umulhi(uh, a.km[4]) * a.km[2]);
      } else {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          cg[2 * ss + j] = X2_ABL == 4 ? 1.f + j : sfl(sg, 2 * kb + j);
          chh[2 * ss + j] = X2_ABL == 4 ? 2.f + j : sfl(sh, 2 * kb + j);
        }
      }
    }
    const int ps = gi % PS, j0 = gi * NB;
    if (X2_ABL == 1) {  // no wgmma: keep the A words alive
#pragma unroll
      for (int q = 0; q < 4; ++q) P[ps][0][q % (N / 2)] += __uint_as_float((A[0][q] ^ A[NB - 1][q]) & 0x3f7fffffu);
    } else {
      wg_fence();
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        const int jj = j0 + j;
        WgF8<N>::mma(P[ps][j], A[j], dlo + (unsigned)((jj >> 2) * N * 8 + (jj & 3) * 2));
      }
      wg_commit();
    }
    if constexpr (PS == 2) {
      if (gi > 0) {
        waitp(P[(gi - 1) % PS], 1);
        fold(P[(gi - 1) % PS], pg, ph, j0 - NB);
      }
#pragma unroll
      for (int j = 0; j < NB; ++j) pg[j] = cg[j], ph[j] = chh[j];
    } else {
      waitp(P[0], 0);
      fold(P[0], cg, chh, j0);
    }
  }
  if constexpr (PS == 2) {
    waitp(P[(NG - 1) % PS], 0);
    fold(P[(NG - 1) % PS], pg, ph, (NG - 1) * NB);
  }
}

// X2_LOOP: one stage of a GEMM1 item as a runtime loop over trips of 2 k64 steps (ptxas: no wgmma group may be in flight
// across the back edge, so each trip ends with wait_group 0)
template <int N, int CH>
__device__ __forceinline__ void stage_w8_loop(const Args<1>& a, const char* st, int ch, float (&acc)[N / 2], int c, int t,
                                              long long& tdep) {
  using C = Cfg<1>;
  static_assert(CH % 2 == 0, "trips of 2 steps");
  const int w = t >> 5, lane = t & 31, g = lane >> 2, lr = lane & 7, mi = lane >> 3;
  const uint32_t aB = smem_u32(st), aA = aB + C::OFF_A;
  const unsigned dlo0 = ((aB + C::OFF_B) & 0x3FFFFu) >> 4;
  const unsigned dlo = MX_UNIF ? __shfl_sync(~0u, dlo0, 0) : dlo0;
  const int rr = 32 * c + 8 * w + g, soff = (ch * CH * 2) & 15;
  const char* spg = st + C::OFF_S + rr * 16 + soff;
  const char* sph = spg + C::BR * 16;
  auto words = [&](unsigned(&wv)[4], int kb) {
    const int m = mi & 1, r = 32 * c + 8 * w + lr, h = kb >> 2, chunk = 2 * (kb & 3) + (mi >> 1);
    ldsm_x4(wv, aA + (unsigned)(((h * 2 + m) * C::BR + r) * 128 + ((chunk ^ (r & 7)) << 4)));
  };
  auto sfk = [&](unsigned u, int k) -> float {  // byte k of a scale word -> |2^(s - 127)| (X2_SFMA) or 2^(s - 121)
    if (X2_SFMA) {
      unsigned x = u;
      if (k) x = X2_KREG ? __umulhi(u, a.km[3 + k]) : u >> (8 * k);
      return __uint_as_float(X2_KREG ? x * a.km[2] : x << 23);
    }
    return __uint_as_float(__byte_perm(u, 0, 0x4440 + k) * (1u << 23) + (6u << 23));
  };
  auto hi8 = [&](unsigned wd) -> unsigned {
    if (X2_PRE && X2_KREG) return ((wd * a.km[0]) & 0x8c8c8c8cu) | (__umulhi(wd, a.km[1]) & 0x10101010u);
    return fp4_hi_e4m3(wd);
  };
  auto fold2 = [&](const float(&Q)[2][N / 2], float g0, float g1, float h0, float h1) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      acc[i] = fmaf(Q[0][i], fabsf((i & 2) ? h0 : g0), acc[i]);
      acc[i] = fmaf(Q[1][i], fabsf((i & 2) ? h1 : g1), acc[i]);
    }
  };
  float P0[2][N / 2], P1[2][N / 2];
  unsigned wa[4], wb[4];
  words(wa, 0);
#pragma unroll 1
  for (int kb = 0; kb < CH; kb += 2) {
    words(wb, kb + 1);
    const unsigned ug = *(const unsigned*)(spg + 2 * kb), uh = *(const unsigned*)(sph + 2 * kb);  // blocks 2 kb .. 2 kb + 3
    {
      const unsigned A0[4] = {fp4_lo_e4m3(wa[0]), fp4_lo_e4m3(wa[1]), hi8(wa[0]), hi8(wa[1])};
      const unsigned A1[4] = {fp4_lo_e4m3(wa[2]), fp4_lo_e4m3(wa[3]), hi8(wa[2]), hi8(wa[3])};
      const int j0 = 2 * kb, j1 = j0 + 1;
      wg_fence();
      WgF8<N>::mma(P0[0], A0, dlo + (unsigned)((j0 >> 2) * N * 8 + (j0 & 3) * 2));
      WgF8<N>::mma(P0[1], A1, dlo + (unsigned)((j1 >> 2) * N * 8 + (j1 & 3) * 2));
      wg_commit();
    }
    if (kb + 2 < CH) words(wa, kb + 2);
    {
      const unsigned A0[4] = {fp4_lo_e4m3(wb[0]), fp4_lo_e4m3(wb[1]), hi8(wb[0]), hi8(wb[1])};
      const unsigned A1[4] = {fp4_lo_e4m3(wb[2]), fp4_lo_e4m3(wb[3]), hi8(wb[2]), hi8(wb[3])};
      const int j0 = 2 * kb + 2, j1 = j0 + 1;
      wg_fence();
      WgF8<N>::mma(P1[0], A0, dlo + (unsigned)((j0 >> 2) * N * 8 + (j0 & 3) * 2));
      WgF8<N>::mma(P1[1], A1, dlo + (unsigned)((j1 >> 2) * N * 8 + (j1 & 3) * 2));
      wg_commit();
    }
    const float g0 = sfk(ug, 0), g1 = sfk(ug, 1), g2 = sfk(ug, 2), g3 = sfk(ug, 3);
    const float h0 = sfk(uh, 0), h1 = sfk(uh, 1), h2 = sfk(uh, 2), h3 = sfk(uh, 3);
    const long long t0 = X2_PROF ? clock64() : 0;
    wg_wait<1>();
    reg_fence(P0[0]), reg_fence(P0[1]);
    if (X2_PROF) tdep += clock64() - t0;
    fold2(P0, g0, g1, h0, h1);
    const long long t1 = X2_PROF ? clock64() : 0;
    wg_wait<0>();
    reg_fence(P1[0]), reg_fence(P1[1]);
    if (X2_PROF) tdep += clock64() - t1;
    fold2(P1, g2, g3, h2, h3);
  }
}
template <int N>
__device__ __forceinline__ void run_item_w8(const Args<1>& a, char* ring, uint64_t* full, uint64_t* empty, long& it, const Hdr h0,
                                            int c, int t, const char* slots, float* xb, long long (&pf)[5], int pr) {
  using C = Cfg<1>;
  using L = Lay<1>;
  constexpr int CH = C::ch(N), NCH = C::KB / CH, KSP = C::KSP;
  static_assert(NCH % KSP == 0, "K-split pairs need an even stage count");
  const int w = t >> 5, lane = t & 31, g = lane >> 2, tq = lane & 3;
  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  float P[W8G1<N>::PS][W8G1<N>::NB][N / 2];
  long itl = it;  // a register copy (the reference spilled to local memory)
  for (int lc = pr; lc < NCH; lc += KSP) {
    const int s = itl % L::SS;
    if (lc > pr) {
      const long long t0 = X2_PROF ? clock64() : 0;
      mbar_wait(smem_u32(full + s), (itl / L::SS) & 1);
      if (X2_PROF) pf[0] += clock64() - t0;
    }
    const long long ts0 = X2_PROF ? clock64() : 0;
    if constexpr (X2_LOOP)
      stage_w8_loop<N, CH>(a, ring + (size_t)s * C::STAGE, lc, acc, c, t, pf[1]);
    else
      stage_w8<N, CH>(a, ring + (size_t)s * C::STAGE, lc, acc, P, c, t, pf[1]);
    if (X2_PROF) pf[2] += CH, pf[3] += clock64() - ts0;
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_u32(empty + s));
    itl += KSP;
  }
  it = itl;
  constexpr float EPI = X2_SFMA ? 64.f : 1.f;  // X2_SFMA: the fold used 2^(s - 127)
  if constexpr (KSP > 1) {  // pair 1 -> pair 0 through SMEM (xb), 8 floats per thread per round (moe_x KS1)
    const int tp = c * 128 + t;
#pragma unroll
    for (int r0 = 0; r0 < N / 2; r0 += 8) {
      if (pr == 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (r0 + i < N / 2) xb[i * 256 + tp] = acc[r0 + i];
      }
      asm volatile("bar.sync 1, 512;" ::: "memory");
      if (pr == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (r0 + i < N / 2) acc[r0 + i] += xb[i * 256 + tp];
      }
      asm volatile("bar.sync 1, 512;" ::: "memory");
    }
    if (pr == 1) return;
  }
  const long long te0 = X2_PROF ? clock64() : 0;
  struct EpT {  // epilogue cycles into pf[4] on every return path
    long long t0, &acc_;
    __device__ ~EpT() {
      if (X2_PROF) acc_ += clock64() - t0;
    }
  } ept{te0, pf[4]};
  const EC* ec = (const EC*)(slots + h0.slot * ECB);  // moe_v4 GEMM1 epilogue (ec.inv: X2_ASC=0 the token scale, else 1)
  if constexpr (X2_W8G2) {
    // H as e4m3 with act_quant's numerics per (token, the 32 intermediates of this warpgroup = one w2 K block): amax over
    // the 8 g-lanes (shuffles), then the 4 warps (xb[c][w][n], named barrier 2 + c); h = SiLU(gate) x up x weight
    const long long pre = ec->img_off / (80 * 128);
    float hv[N / 4], wm[N / 4];
#pragma unroll
    for (int nb = 0; nb < N / 8; ++nb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nb + 2 * tq + hh, x = 2 * nb + hh;
        hv[x] = 0.f;
        if (ec->tok[n] >= 0) {
          const float iv = ec->inv[n];
          const float gt = fminf(acc[4 * nb + hh] * EPI * iv, 10.f);
          const float up = fminf(fmaxf(acc[4 * nb + 2 + hh] * EPI * iv, -10.f), 10.f);
          hv[x] = (X2_FASTEPI ? __fdividef(gt, 1.f + __expf(-gt)) : gt / (1.f + expf(-gt))) * up * ec->wt[n];
        }
        wm[x] = fabsf(hv[x]);
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) wm[x] = fmaxf(wm[x], __shfl_xor_sync(~0u, wm[x], o));
      }
    float* eb = xb + c * 4 * NMAX;
    if (g == 0) {
#pragma unroll
      for (int x = 0; x < N / 4; ++x) eb[w * NMAX + 8 * (x >> 1) + 2 * tq + (x & 1)] = wm[x];
    }
    asm volatile("bar.sync %0, 128;" ::"r"(2 + c) : "memory");
    const int blk = (C::BR / 32) * h0.q + c, kk = 8 * w + g, L = (kk & 1) ? 16 + (kk >> 1) : (kk >> 1), kg = 32 * blk + L;
    char* hb = a.h8 + pre * 576 + (size_t)(kg >> 6) * N * 64;
#pragma unroll
    for (int x = 0; x < N / 4; ++x) {
      const int n = 8 * (x >> 1) + 2 * tq + (x & 1);
      const float am = fmaxf(fmaxf(eb[n], eb[NMAX + n]), fmaxf(eb[2 * NMAX + n], eb[3 * NMAX + n]));
      const float sc = pow2_ceil_f(fmaxf(am, 1e-4f) * (1.0f / 448.0f));
      hb[sw64(n, kg & 63)] = (char)__nv_cvt_float_to_fp8(hv[x] * (1.0f / sc), __NV_SATFINITE, __NV_E4M3);
      if (w == 0 && g == 0) a.hsc[pre * 20 + blk * N + n] = sc;
    }
    asm volatile("bar.sync %0, 128;" ::"r"(2 + c) : "memory");
    return;
  }
  const int i = C::BR * h0.q + 32 * c + 8 * w + g, i6 = i & 63, pos = 32 * (i6 >> 5) + kinv(i6 & 31);
  char* hs = a.himg + ec->h_off + (size_t)(i >> 6) * N * 128;
#pragma unroll
  for (int nb = 0; nb < N / 8; ++nb)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int n = 8 * nb + 2 * tq + hh;
      const int tok = ec->tok[n];
      float hv = 0.f;
      if (tok >= 0) {
        const float iv = ec->inv[n];
        float gt = fminf(acc[4 * nb + hh] * EPI * iv, 10.f);
        float up = fminf(fmaxf(acc[4 * nb + 2 + hh] * EPI * iv, -10.f), 10.f);
        hv = (X2_FASTEPI ? __fdividef(gt, 1.f + __expf(-gt)) : gt / (1.f + expf(-gt))) * up * ec->wt[n] * 256.f;
      }
      *(__half*)(hs + swz(n, pos * 2)) = __float2half_rn(hv);
    }
}


// ---- W4A8 GEMM2 (X2_W8G2): items of BR output rows, the e4m3 H (SW64 k64 slabs) as B; per k64 step the 4 words of rows
// (g, g + 8) x 2 k32 blocks -> 2 fresh m64nNk32 wgmma, folded with |2^(s - 127)| x the H block scale (the 2^6 in the
// epilogue), the same ALU-pipe relief as GEMM1 (X2_SFMA / X2_KREG / X2_GS)
template <int N, int CH>
__device__ __forceinline__ void stage_w8g2(const Args<2>& a, const char* st, int ch, float (&acc)[N / 2],
                                           float (&P)[W8G1<N>::PS][W8G1<N>::NB][N / 2], int c, int t) {
  using C = Cfg<2>;
  constexpr int PS = W8G1<N>::PS, NB = W8G1<N>::NB, GS = (CH % W8G1<N>::GS == 0) ? W8G1<N>::GS : 1, RB = CH == 9 ? IR / 2 : 96;
  static_assert(NB == 2 * W8G1<N>::GS, "blocks per group");
  const int w = t >> 5, lane = t & 31, g = lane >> 2, tq = lane & 3, lr = lane & 7, mi = lane >> 3;
  const uint32_t aB = smem_u32(st), aA = aB + C::OFF_A;
  const unsigned dlo0 = ((aB + C::OFF_B) & 0x3FFFFu) >> 4;
  const unsigned dlo = MX_UNIF ? __shfl_sync(~0u, dlo0, 0) : dlo0;
  const int RR = 64 * c + 16 * w + g;
  auto words = [&](unsigned(&wv)[4], int kb) {
    const int r = 64 * c + 16 * w + 8 * (mi & 1) + lr;
    ldsm_x4(wv, aA + (unsigned)(r * RB + 32 * kb + 16 * (mi >> 1)));
  };
  // two scale bytes (blocks 2 kb, 2 kb + 1) of a row -> floats: X2_SFMA |2^(s - 127)| on the FMA pipe, else 2^(s - 121)
  auto sf2 = [&](unsigned u16, float& f0, float& f1) {
    if (X2_SFMA) {
      f0 = __uint_as_float(X2_KREG ? u16 * a.km[2] : u16 << 23);  // byte 1's low bit lands in the sign: used as |x|
      f1 = __uint_as_float(X2_KREG ? __umulhi(u16, a.km[4]) * a.km[2] : (u16 >> 8) << 23);
    } else {
      f0 = __uint_as_float((u16 & 255u) * (1u << 23) + (6u << 23)), f1 = __uint_as_float((u16 >> 8) * (1u << 23) + (6u << 23));
    }
  };
  auto hi8 = [&](unsigned wd) -> unsigned {
    if (X2_PRE && X2_KREG) return ((wd * a.km[0]) & 0x8c8c8c8cu) | (__umulhi(wd, a.km[1]) & 0x10101010u);
    return fp4_hi_e4m3(wd);
  };
  auto fold = [&](const float(&Q)[NB][N / 2], const float* fg, const float* fh, int j0, int nbk) {
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      if (j >= nbk) break;
      float tsc[N / 4];
      const float* xs = (const float*)(st + C::OFF_X) + (j0 + j) * N + 2 * tq;
#pragma unroll
      for (int nb = 0; nb < N / 8; ++nb) {
        const float2 v = *(const float2*)(xs + 8 * nb);
        tsc[2 * nb] = v.x, tsc[2 * nb + 1] = v.y;
      }
#pragma unroll
      for (int i = 0; i < N / 2; ++i) acc[i] = fmaf(Q[j][i], fabsf((i & 2) ? fh[j] : fg[j]) * tsc[2 * (i >> 2) + (i & 1)], acc[i]);
    }
  };
  auto waitp = [&](float(&Q)[NB][N / 2], int left) {
    if (left) wg_wait<1>(); else wg_wait<0>();
#pragma unroll
    for (int j = 0; j < NB; ++j) reg_fence(Q[j]);
  };
  unsigned wv[2][4];
  words(wv[0], 0);
  float pg[NB], ph[NB];
#pragma unroll
  for (int j = 0; j < NB; ++j) pg[j] = 0.f, ph[j] = 0.f;
  constexpr int NG = CH / GS, NBG = 2 * GS;  // blocks used per group (NB when GS divides CH, else 2)
#pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    unsigned A[NB][4];
    float cg[NB], chh[NB];
#pragma unroll
    for (int ss = 0; ss < GS; ++ss) {
      const int kb = gi * GS + ss;
      if (kb + 1 < CH) words(wv[(kb + 1) & 1], kb + 1);
      const unsigned* cw = wv[kb & 1];
      A[2 * ss][0] = fp4_lo_e4m3(cw[0]), A[2 * ss][1] = fp4_lo_e4m3(cw[1]), A[2 * ss][2] = hi8(cw[0]), A[2 * ss][3] = hi8(cw[1]);
      A[2 * ss + 1][0] = fp4_lo_e4m3(cw[2]), A[2 * ss + 1][1] = fp4_lo_e4m3(cw[3]), A[2 * ss + 1][2] = hi8(cw[2]),
      A[2 * ss + 1][3] = hi8(cw[3]);
      const int blk0 = 2 * (ch * CH + kb);
      sf2(*(const unsigned short*)(st + C::OFF_S + RR * 18 + blk0), cg[2 * ss], cg[2 * ss + 1]);
      sf2(*(const unsigned short*)(st + C::OFF_S + (RR + 8) * 18 + blk0), chh[2 * ss], chh[2 * ss + 1]);
    }
    const int ps = gi % PS, j0 = gi * NBG;
    wg_fence();
#pragma unroll
    for (int j = 0; j < NBG; ++j) {
      const int kb = gi * GS + (j >> 1);
      WgF8<N>::template mma<F8_SW64>(P[ps][j], A[j], dlo + (unsigned)(kb * N * 4 + (j & 1) * 2));
    }
    wg_commit();
    if constexpr (PS == 2) {
      if (gi > 0) {
        waitp(P[(gi - 1) % PS], 1);
        fold(P[(gi - 1) % PS], pg, ph, j0 - NBG, NBG);
      }
#pragma unroll
      for (int j = 0; j < NB; ++j) pg[j] = cg[j], ph[j] = chh[j];
    } else {
      waitp(P[0], 0);
      fold(P[0], cg, chh, j0, NBG);
    }
  }
  if constexpr (PS == 2) {
    waitp(P[(NG - 1) % PS], 0);
    fold(P[(NG - 1) % PS], pg, ph, (NG - 1) * NBG, NBG);
  }
}
template <int N>
__device__ __forceinline__ void run_item_w8g2(const Args<2>& a, char* ring, uint64_t* full, uint64_t* empty, long& it,
                                              const Hdr h0, int c, int t, const char* slots, char* stgb) {
  using C = Cfg<2>;
  using L = Lay<2>;
  constexpr int CH = C::ch(N), NCH = C::KB / CH;
  const int w = t >> 5, lane = t & 31, g = lane >> 2, tq = lane & 3;
  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  float P[W8G1<N>::PS][W8G1<N>::NB][N / 2];
  for (int lc = 0; lc < NCH; ++lc) {
    const int s = it % L::SS;
    if (lc > 0) mbar_wait(smem_u32(full + s), (it / L::SS) & 1);
    stage_w8g2<N, CH>(a, ring + (size_t)s * C::STAGE, lc, acc, P, c, t);
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_u32(empty + s));
    ++it;
  }
  const EC* ec = (const EC*)(slots + h0.slot * ECB);  // bf16 Y per (token, slot): moe_x's staging, no 1/256 (H unscaled)
  __nv_bfloat16* stg = (__nv_bfloat16*)stgb + (4 * c + w) * 256;
  const int row0 = C::BR * h0.q + 64 * c + 16 * w, eslot = ec->pad, enr = ec->n_real;
#pragma unroll
  for (int t0 = 0; t0 < N; t0 += 16) {
    __syncwarp();
#pragma unroll
    for (int nb = t0 / 8; nb < t0 / 8 + 2 && nb < N / 8; ++nb)
#pragma unroll
      for (int rs = 0; rs < 2; ++rs)
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) stg[(8 * nb + 2 * tq + jj - t0) * 16 + g + 8 * rs] = __float2bfloat16_rn(acc[4 * nb + 2 * rs + jj] * (X2_SFMA ? 64.f : 1.f));
    __syncwarp();
    const int n = t0 + (lane >> 1);
    if (n < N && n < enr)
      *(uint4*)(a.yout + (size_t)(eslot + n) * D + row0 + 8 * (lane & 1)) = *(const uint4*)(stg + (lane >> 1) * 16 + 8 * (lane & 1));
  }
}

template <int MODE>
__global__ void __launch_bounds__(32 * Cfg<MODE>::NCW + Cfg<MODE>::PTH, 1) k_mx(const __grid_constant__ Args<MODE> a) {
  using C = Cfg<MODE>;
  using L = Lay<MODE>;
  extern __shared__ __align__(1024) char smem_raw[];
  char* ring = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  uint64_t* full = (uint64_t*)(ring + L::O_BAR);
  uint64_t* empty = full + L::SS;
  char* slots = ring + L::O_SLOT;
  char* stgb = ring + L::O_STG;
  Plan& pl = *(Plan*)(ring + L::O_PLAN);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int s = 0; s < L::SS; ++s) mbar_init(smem_u32(full + s), 1), mbar_init(smem_u32(empty + s), C::NCW / C::KSP);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  build_plan(pl, a.ecs, a.counts);  // ends with __syncthreads
  if (warp >= C::NCW) {
    // setmaxnreg moves registers only inside the CTA's launch allocation (96 x 640 = 61,440): 512 x 112 + 128 x 32
    if (C::PTH == 128) asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
    if (warp == C::NCW && lane == 0) producer<MODE>(a, ring, full, empty, slots, pl);
    return;
  }
  if (C::PTH == 128) asm volatile("setmaxnreg.inc.sync.aligned.u32 112;" ::: "memory");
  const int cg = threadIdx.x / 128, t = threadIdx.x % 128, pr = cg / C::RWG, c = cg % C::RWG;
  float* xb = (float*)(ring + L::O_XB);
  long it = pr;
  long long pf[5] = {0, 0, 0, 0, 0};
  const long long tstart = X2_PROF ? clock64() : 0;
  unsigned long long gstart = 0;
  if (X2_PROF) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gstart));
  for (;;) {
    const int s = it % L::SS;
    {
      const long long t0 = X2_PROF ? clock64() : 0;
      mbar_wait(smem_u32(full + s), (it / L::SS) & 1);
      if (X2_PROF) pf[0] += clock64() - t0;
    }
    const Hdr h = *(const Hdr*)(ring + (size_t)s * C::STAGE + C::OFF_H);
    if (h.item < 0) break;
#define X2_RUN(NN)                                                                      \
  if constexpr (MODE == 1 && X2_W8) run_item_w8<NN>(a, ring, full, empty, it, h, c, t, slots, xb, pf, pr); \
  else if constexpr (MODE == 2 && X2_W8G2) run_item_w8g2<NN>(a, ring, full, empty, it, h, c, t, slots, stgb); \
  else run_item<MODE, NN>(a, ring, full, empty, it, h, c, t, slots, stgb, pr, xb)
    switch (h.npad >> 3) {
      case 1: X2_RUN(8); break;
      case 2: X2_RUN(16); break;
      case 3: X2_RUN(24); break;
      case 4: X2_RUN(32); break;
      case 5: X2_RUN(40); break;
      case 6: X2_RUN(48); break;
      case 7: X2_RUN(56); break;
      default: X2_RUN(64); break;
    }
#undef X2_RUN
  }
  if (X2_PROF && MODE == 1 && a.prof && (threadIdx.x & 31) == 0) {
    unsigned long long gend;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gend));
    atomicAdd(a.prof + 0, (unsigned long long)pf[0]);
    atomicAdd(a.prof + 1, (unsigned long long)pf[1]);
    atomicAdd(a.prof + 2, (unsigned long long)(clock64() - tstart));
    atomicAdd(a.prof + 3, (unsigned long long)pf[2]);
    atomicAdd(a.prof + 4, gend - gstart);
    atomicAdd(a.prof + 5, 1ull);
    atomicAdd(a.prof + 6, (unsigned long long)pf[3]);
    atomicAdd(a.prof + 7, (unsigned long long)pf[4]);
  }
}


// ---- layout conversion (in place): raw <-> pre-placed bytes
__global__ void k_preplace(uint4* __restrict__ p, long long n16, int inverse) {
  __shared__ unsigned char lut[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    if (!inverse) {
      lut[i] = (unsigned char)preplace_byte((unsigned)i);
    } else {
      lut[preplace_byte((unsigned)i)] = (unsigned char)i;
    }
  }
  __syncthreads();
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n16; i += (long long)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    unsigned* u = &v.x;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      u[k] = (unsigned)lut[u[k] & 255] | (unsigned)lut[(u[k] >> 8) & 255] << 8 | (unsigned)lut[(u[k] >> 16) & 255] << 16 |
             (unsigned)lut[u[k] >> 24] << 24;
    p[i] = v;
  }
}
// ---- W4A8 token image: one warp per image slot (token row of an EC), lane l handles 32-blocks l, l + 32, ...:
// fpr/kernels/w8a8/act_quant.cuh numerics (block 32, ue8m0: s = the smallest power of 2 with amax <= 448 s, amax 0 ->
// s = 1; q = e4m3_rn(x / s), satfinite); X2_ASC=0: one s per token (aq::scale_exp of the row amax). Image: per EC 40 k128 slabs of [n_pad][128 B], 128B
// swizzle, logical order inside a 32-block (L < 16: k = 2 L, else 2 (L - 16) + 1); block scales asc (X2_ASC=1) per EC
// [160][n_pad] at slot base x 160; ec.inv = s (X2_ASC=0) / 1 (X2_ASC=1) / 0 (padding rows).
__global__ void __launch_bounds__(256) k_gather_w8(const __nv_bfloat16* __restrict__ x, EC* __restrict__ ecs,
                                                   const int* __restrict__ counts, char* __restrict__ img,
                                                   float* __restrict__ asc) {
  __shared__ int sbase[MAXEC + 1];
  const int nec = counts[0], S = counts[3] / (80 * 128);
  for (int i = threadIdx.x; i < nec; i += blockDim.x) sbase[i] = (int)(ecs[i].img_off / (80 * 128));
  if (threadIdx.x == 0) sbase[nec] = S;
  __syncthreads();
  const int lane = threadIdx.x & 31, wpb = blockDim.x >> 5;
  for (int j = blockIdx.x * wpb + (threadIdx.x >> 5); j < S; j += gridDim.x * wpb) {
    int lo = 0, hi = nec - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (sbase[mid] <= j) lo = mid; else hi = mid - 1;
    }
    const int n = j - sbase[lo], np = sbase[lo + 1] - sbase[lo], tok = ecs[lo].tok[n];
    auto load32 = [&](int b, float (&v)[32]) {
      if (tok < 0) {
#pragma unroll
        for (int k = 0; k < 32; ++k) v[k] = 0.f;
        return;
      }
      const uint4* src = (const uint4*)(x + (size_t)tok * D + 32 * b);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 u = src[q];
        const unsigned uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) v[8 * q + 2 * e] = __uint_as_float(uu[e] << 16), v[8 * q + 2 * e + 1] = __uint_as_float(uu[e] & 0xffff0000u);
      }
    };
    float am[5];
#pragma unroll
    for (int r = 0; r < 5; ++r) {
      float v[32];
      load32(lane + 32 * r, v);
      am[r] = 0.f;
#pragma unroll
      for (int k = 0; k < 32; ++k) am[r] = fmaxf(am[r], fabsf(v[k]));
    }
    float tam = fmaxf(fmaxf(fmaxf(am[0], am[1]), fmaxf(am[2], am[3])), am[4]);
#pragma unroll
    for (int o = 16; o; o >>= 1) tam = fmaxf(tam, __shfl_xor_sync(~0u, tam, o));
    const float st = aq::exp2i(aq::scale_exp(tam));
    char* base = img + (size_t)sbase[lo] * 5120;
#pragma unroll 1
    for (int r = 0; r < 5; ++r) {
      const int b = lane + 32 * r;
      float v[32];
      load32(b, v);
      const float sb = X2_ASC ? aq::exp2i(aq::scale_exp(am[r])) : st, is = 1.0f / sb;  // exact (power of two)
      unsigned o[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) {  // logical bytes 4 q .. 4 q + 3
        unsigned wv = 0;
#pragma unroll
        for (int e = 0; e < 4; e += 2) {
          const int L0 = 4 * q + e, L1 = L0 + 1;
          const int p0 = L0 < 16 ? 2 * L0 : 2 * (L0 - 16) + 1, p1 = L1 < 16 ? 2 * L1 : 2 * (L1 - 16) + 1;
          const __nv_fp8x2_storage_t pr = __nv_cvt_float2_to_fp8x2(make_float2(v[p0] * is, v[p1] * is), __NV_SATFINITE, __NV_E4M3);
          wv |= (unsigned)pr << (8 * e);
        }
        o[q] = wv;
      }
      const int slab = b >> 2, bb = (b & 3) * 32;
      char* rowp = base + (size_t)slab * np * 128;
      *(uint4*)(rowp + swz(n, bb)) = make_uint4(o[0], o[1], o[2], o[3]);
      *(uint4*)(rowp + swz(n, bb + 16)) = make_uint4(o[4], o[5], o[6], o[7]);
      if (X2_ASC) asc[(size_t)sbase[lo] * 160 + (size_t)b * np + n] = tok >= 0 ? sb : 0.f;
    }
    if (lane == 0) ecs[lo].inv[n] = tok >= 0 ? (X2_ASC ? 1.f : st) : 0.f;
  }
}
}  // namespace mx

// ================================================================================================ host
using namespace mx;
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
static CUtensorMap tmap2d(const void* base, CUtensorMapDataType ty, int esz, uint64_t inner, uint64_t rows, uint32_t box0,
                          uint32_t box1, CUtensorMapSwizzle sw) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)(inner * esz)};
  cuuint32_t box[2] = {box0, box1}, es[2] = {1, 1};
  CUresult r = tmap_encode()(&m, ty, 2, const_cast<void*>(base), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
template <int MODE>
static void launch(const Args<MODE>& a, int64_t grid, int dev) {
  static bool attr_dev[64] = {};
  if (!attr_dev[dev & 63]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k_mx<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, Lay<MODE>::SMEM));
    attr_dev[dev & 63] = true;
  }
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (MX_PDL && MODE == 2) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid > 0 ? (unsigned)grid : (unsigned)sms);
    cfg.blockDim = dim3(32 * Cfg<MODE>::NCW + Cfg<MODE>::PTH);
    cfg.dynamicSmemBytes = Lay<MODE>::SMEM;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at_;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_mx<MODE>, a));
    return;
  }
  k_mx<MODE><<<grid > 0 ? (int)grid : sms, 32 * Cfg<MODE>::NCW + Cfg<MODE>::PTH, Lay<MODE>::SMEM, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// raw w13 [E, 1152, 2560] / s13 [E, 1152, 160] uint8; ecs / counts from route2r; img from gather6; himg = GEMM1 output
// (H image, moe_v4 layout); ctr int32 [>= 2] zeros (queue counters, left zeroed)
void gemm1(torch::Tensor w13, torch::Tensor s13, torch::Tensor ecs, torch::Tensor counts, torch::Tensor img, torch::Tensor himg,
           torch::Tensor ctr, torch::Tensor asc, torch::Tensor hsc, int64_t grid, c10::optional<torch::Tensor> prof) {
  TORCH_CHECK(w13.size(1) == 1152 && w13.size(2) == D / 2 && s13.size(2) == D / 32, "moe_x gemm1: raw w13 / s13 shapes");
  TORCH_CHECK(ctr.scalar_type() == torch::kInt32 && ctr.numel() >= 2, "moe_x gemm1: ctr int32 >= 2");
  const c10::cuda::CUDAGuard dg(w13.device());
  using C = Cfg<1>;
  Args<1> a{};
  a.ecs = (const EC*)ecs.data_ptr(), a.counts = counts.data_ptr<int>(), a.bimg = (const char*)img.data_ptr();
  a.himg = (char*)himg.data_ptr(), a.ctr = ctr.data_ptr<int>();
  a.asc = (X2_W8 && X2_ASC) ? asc.data_ptr<float>() : nullptr;
  a.h8 = (char*)himg.data_ptr(), a.hsc = X2_W8G2 ? hsc.data_ptr<float>() : nullptr;  // X2_W8G2: himg is the e4m3 H image
  a.prof = prof.has_value() ? (unsigned long long*)prof->data_ptr() : nullptr;
  a.km[0] = 4u, a.km[1] = 1u << 30, a.km[2] = 1u << 23, a.km[3] = 0u, a.km[4] = 1u << 24, a.km[5] = 1u << 16, a.km[6] = 1u << 8;
  const uint64_t rows = (uint64_t)w13.size(0) * 1152;
  a.tmw = tmap2d(w13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 2, rows, 128, C::BR, CU_TENSOR_MAP_SWIZZLE_128B);
  a.tmw64 = tmap2d(w13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 2, rows, 64, C::BR, CU_TENSOR_MAP_SWIZZLE_64B);
  a.tms = tmap2d(s13.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 32, rows, 16, C::BR, CU_TENSOR_MAP_SWIZZLE_NONE);
  launch<1>(a, grid, w13.get_device());
}
// raw w2 [E, 5120, 288] / s2 [E, 5120, 18] uint8; himg = GEMM1's H image; yout bf16 [slots, 5120]; ctr as gemm1
void gemm2(torch::Tensor w2, torch::Tensor s2, torch::Tensor ecs, torch::Tensor counts, torch::Tensor himg, torch::Tensor yout,
           torch::Tensor ctr, torch::Tensor hsc, int64_t grid) {
  TORCH_CHECK(w2.size(1) == D && w2.size(2) == IR / 2 && s2.size(2) == IR / 32, "moe_x gemm2: raw w2 / s2 shapes");
  TORCH_CHECK(ctr.scalar_type() == torch::kInt32 && ctr.numel() >= 2, "moe_x gemm2: ctr int32 >= 2");
  const c10::cuda::CUDAGuard dg(w2.device());
  using C = Cfg<2>;
  Args<2> a{};
  a.w = (const unsigned char*)w2.data_ptr(), a.s = (const unsigned char*)s2.data_ptr();
  a.ecs = (const EC*)ecs.data_ptr(), a.counts = counts.data_ptr<int>(), a.bimg = (const char*)himg.data_ptr();
  a.yout = (__nv_bfloat16*)yout.data_ptr(), a.ctr = ctr.data_ptr<int>() + 1;
  a.h8 = (char*)himg.data_ptr(), a.hsc = X2_W8G2 ? hsc.data_ptr<float>() : nullptr;
  a.km[0] = 4u, a.km[1] = 1u << 30, a.km[2] = 1u << 23, a.km[3] = 0u, a.km[4] = 1u << 24, a.km[5] = 1u << 16, a.km[6] = 1u << 8;
  a.tmw = tmap2d(w2.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, IR / 4, (uint64_t)w2.size(0) * D, 48, C::BR,
                 CU_TENSOR_MAP_SWIZZLE_NONE);
  launch<2>(a, grid, w2.get_device());
}

// in place: raw checkpoint expert bytes <-> pre-placed (inverse=1 restores raw); any uint8 tensor, numel % 16 == 0
void preplace(torch::Tensor w, int64_t inverse) {
  TORCH_CHECK(w.scalar_type() == torch::kUInt8 && w.is_contiguous() && w.numel() % 16 == 0 &&
                  ((uintptr_t)w.data_ptr() % 16) == 0, "preplace: contiguous uint8, numel % 16 == 0");
  const c10::cuda::CUDAGuard dg(w.device());
  k_preplace<<<1024, 256, 0, at::cuda::getCurrentCUDAStream()>>>((uint4*)w.data_ptr(), w.numel() / 16, (int)inverse);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// W4A8 token image: x bf16 [T, 5120]; ecs / counts from route2r; img8 >= slots x 5120 B; asc float >= slots x 160 (X2_ASC=1)
void gather_w8(torch::Tensor x, torch::Tensor ecs, torch::Tensor counts, torch::Tensor img8, torch::Tensor asc, int64_t grid) {
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.size(1) == D && x.stride(0) == D, "gather_w8: x [T, 5120] bf16");
  const c10::cuda::CUDAGuard dg(x.device());
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device());
  k_gather_w8<<<grid > 0 ? (int)grid : 2 * sms, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)x.data_ptr(), (EC*)ecs.data_ptr(), counts.data_ptr<int>(), (char*)img8.data_ptr(),
      X2_ASC ? asc.data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int64_t> config() {  // NWG1, NWG2, stages1, stages2, SMEM1, SMEM2, STAGE1, STAGE2, threads1, threads2
  return {Cfg<1>::NWG, MX_NWG2, Lay<1>::SS, Lay<2>::SS, Lay<1>::SMEM, Lay<2>::SMEM, Cfg<1>::STAGE, Cfg<2>::STAGE,
          32 * Cfg<1>::NCW + Cfg<1>::PTH, 32 * Cfg<2>::NCW + Cfg<2>::PTH, X2_PRE, X2_W8, X2_ASC, X2_W8G2};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm1", &gemm1, py::arg("w13"), py::arg("s13"), py::arg("ecs"), py::arg("counts"), py::arg("img"), py::arg("himg"),
        py::arg("ctr"), py::arg("asc"), py::arg("hsc"), py::arg("grid"), py::arg("prof") = py::none());
  m.def("gemm2", &gemm2);
  m.def("config", &config);
  m.def("preplace", &preplace);
  m.def("gather_w8", &gather_w8);
}
