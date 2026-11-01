// ds41h decode-shaped routed-MoE W4A16 kernels ("moe_dec") for calls with 1..~512 tokens
// (multi-request decode c2..c64 at k=7 = 16..512 tokens, prefill replay chunks <= SMALL_T).
//
// Same layouts, numerics and outputs as moe_small16_kernel<MODE, 16> (moe_prefill.cu, forward variant 0x500), BIT FOR BIT:
//   * every weight element is expanded exactly FP4 -> bf16 with its ue8m0 scale folded in (the same raw bf16 bits
//     value * 2^-126 times 2^(S-1) as fp4_pair_bf16_raw + bf16x2_mul_rn),
//   * lane (g, t) owns physical bytes [8t, 8t+8) of every 64-wide K block of its rows, rotated right by 4t bits, and k16 step
//     s of the block uses word s >> 1, pairs 2 (s & 1) + {0, 1} -- the same A fragments as small16, hence the same
//     mma.sync m16n8k16 products in the same k16 order with fp32 accumulation over the whole K,
//   * the activations are the same W4A16 logical-K-order rows (act_perm16 / GEMM1 output), GEMM1 epilogue = stock clamped
//     SwiGLU on the fp32 accumulators (accurate expf) -> one bf16 rounding, GEMM2 = unscaled bf16 y in slot order,
//     moe_combine (fp32, k order, one rounding) applies the routing weights.
// Only WHICH lane / register computes an output column differs (rows are assigned for bank-conflict-free swizzled loads).
//
// What is different (speed):
//   1. FP4 expansion in 4 instructions per bf16x2 pair (small16: 5, 2.5 of them on the ALU pipe, which made small16 ALU-bound):
//        t = w & 0x000F000F;  r = t * (2^6 + 2^12)  (IMAD: the two shifted copies of each nibble, [9:6] and [15:12] (+16),
//        cannot overlap, so no carries);  v = r & 0x81C081C0 (mag [8:6] + sign 15 per half);  HMUL2 by 2^(S-1).
//      Pair (1,5): t = w & 0x00F000F0, r = t * (2^2 + 2^8). Pairs (2,6)/(3,7): the same on w >> 8 (IMAD.HI). The multipliers
//      are kernel arguments, so ptxas cannot strength-reduce them back into ALU shift/LEA sequences.
//   2. Units of up to NT = 8 * NGMAX tokens (small16 0x500: 16) -> hot experts (many routed rows in multi-request decode)
//      dequantize their weights once per NT tokens instead of once per 16.
//   3. KPS = 1 / 2 / 4 consecutive 64-wide K blocks per pipeline stage = one TMA weight box of 32 / 64 / 128 bytes per row
//      (no / 64B / 128B swizzle; rows assigned to lanes so every 8-byte weight load is bank-conflict-free).
//   4. Token rows in 8-row TMA boxes (small16: 16) -> half the token traffic for experts with <= 8 rows.
// Cross-proxy rule (docs/EXACTNESS.md): fence.proxy.async.shared::cta before every release of a TMA-filled stage / scale slot that
// generic loads (lds / ldmatrix) read.
#include "common.cuh"
#ifndef DS41H_DEVICE_ONLY
#include "host.cuh"
#endif

#include <cuda.h>

#ifndef DS41H_DEVICE_ONLY
#include <cudaTypedefs.h>

#include <cstdio>
#include <cstdlib>
#endif

namespace {

constexpr int MAXE = 512;     // experts per layer supported by this kernel (DeepSeek-V4.1-Flash: 384)
constexpr int META_CTR = 8;   // moe_route.cu meta layout
constexpr int NCW = 4;        // consumer warps
constexpr int NTHR = 32 * (NCW + 1);
constexpr int NSR = 2;        // scale-slot ring
constexpr int S_SLOT = 2304;  // MODE 1: 2 x 64 rows x 16 B = 2048; MODE 2: 128 rows x I/32 (I <= 576) = 2304
constexpr int PITCH1 = 144, PITCH2 = 272;  // epilogue staging rows: 64 / 128 bf16 + 16 B

// ------------------------------------------------------------------------------------------------------------- PTX
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}
__device__ __forceinline__ int lds_u8(uint32_t a) {
  unsigned v;
  asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v) : "r"(a));
  return int(v);
}
__device__ __forceinline__ void stg128(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
// mbarrier wait with a watchdog: a phase that does not complete within ~20 s (a protocol bug) traps instead of hanging
__device__ __forceinline__ void mbar_wait_s(uint32_t bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n .reg .u64 t0, t1;\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @p bra.uni LAB_DONE;\n"
      " mov.u64 t0, %%globaltimer;\n"
      "LAB_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @p bra.uni LAB_DONE;\n"
      " mov.u64 t1, %%globaltimer;\n"
      " sub.u64 t1, t1, t0;\n"
      " setp.lt.u64 p, t1, 20000000000;\n"
      " @p bra.uni LAB_WAIT;\n"
      " trap;\n"
      "LAB_DONE:\n}" ::"r"(bar), "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive_s(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx_s(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
               "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
               : "memory");
}
// L2 cache-hint variants (weights / scales are streamed exactly once per call: evict_first keeps the token rows, re-read by
// every weight block of an expert, resident in L2)
__device__ __forceinline__ void tma_load_2d_hint(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void bulk_load_hint(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar), "l"(pol)
               : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void mma16816_bf16(float (&d)[4], const unsigned (&A)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(uint32_t a, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm_x2(uint32_t a, unsigned& r0, unsigned& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(a) : "memory");
}
__device__ __forceinline__ unsigned bf16x2_mul_rn(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ unsigned mul_lo(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.lo.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ unsigned mul_hi(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.hi.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
// Loop-invariant per-lane values that ptxas must keep in registers (it otherwise rematerializes the address arithmetic from
// %tid inside the K loop: ~40 extra ALU / IMAD instructions per K block): an identity shuffle is opaque to it.
__device__ __forceinline__ uint32_t opaque(uint32_t v) { return __shfl_sync(0xffffffffu, v, threadIdx.x & 31); }
// {2^(S-1), 2^(S-1)} as bf16x2 for a ue8m0 byte S in [0, 128] (moe_prefill.cu fp4_bf16_mult)
__device__ __forceinline__ unsigned fp4_bf16_mult(int S) { return unsigned(S + 126) * 0x00800080u; }
// moe_prefill.cu a16_phys: physical column of logical column L inside a 64-wide K block (W4A16 logical order)
__host__ __device__ __forceinline__ constexpr int a16_phys(int L) {
  return 16 * ((L >> 1) & 3) + 8 * (L >> 5) + ((2 * ((L >> 4) & 1) + ((L >> 3) & 1) + 4 * (L & 1) + ((L >> 1) & 3)) & 7);
}
// accurate expf (libdevice) for the stock SwiGLU numerics (the extension is built with --use_fast_math)
extern "C" __device__ float __nv_expf(float);

// ------------------------------------------------------------------------------------------------------- arguments
struct DecArgs {
  CUtensorMap tm_b, tm_w, tm_s;  // activation rows (SW128, 8-row boxes) | weights (KPS-wide boxes) | MODE 1 scales (16 B x 64 rows)
  const uint8_t* s;              // ue8m0 scales [E * rows_e, K/32]
  const int* off;                // [E+1] routed-row offsets per expert (moe_permute)
  const int* perm_slot;          // [R] t*K + k of routed row
  __nv_bfloat16* out;            // MODE 1 h [R, I]; MODE 2 slot [T*K, H]
  long w_estride;                // bytes per expert (weights)
  int w_rstride;                 // bytes per weight row (K/2)
  int s_rstride;                 // bytes per scale row (K/32)
  int K;                         // reduction length (multiple of 64)
  int E;                         // experts
  int NB;                        // 128-row weight blocks per expert (MODE 1: I / 64 gate+up blocks; MODE 2: H / 128)
  int I;                         // MODE 1: row offset of w3 (up) inside w13
  int out_stride;                // elements per output row
  int brows;                     // token rows per activation box (8, fewer only for tiny activation tensors)
  float limit;                   // MODE 1: swiglu limit
  unsigned k01, k23, kshr8;      // 0x1040, 0x104, 1 << 24: IMAD multipliers (runtime values: ptxas must not strength-reduce)
  int whint;                     // 1: weight / scale loads with an L2 evict_first policy (DS41H_MOE_DEC_HINT)
};

template <int MODE_, int KPS_, int NST_, int NGMAX_, int CTAS_>
struct DecCfg {
  static constexpr int MODE = MODE_, KPS = KPS_, NST = NST_, NGMAX = NGMAX_, CTAS = CTAS_;
  static_assert(MODE == 1 || MODE == 2, "MODE");
  static_assert(KPS == 1 || KPS == 2 || KPS == 4, "KPS 1 / 2 / 4 (divides the 8-block MODE 1 scale groups)");
  static_assert(NGMAX >= 1 && NGMAX <= 8 && NST >= 2, "NGMAX / NST");
  static constexpr int RB = 32 * KPS;       // weight bytes per row per stage
  static constexpr int W_ST = 128 * RB;     // 4 / 8 / 16 KB
  static constexpr int NT = 8 * NGMAX;      // tokens per unit
  static constexpr int X_KB = NT * 128;     // token tile of one K block (NT rows x 64 bf16, SW128)
  static constexpr int X_ST = KPS * X_KB;
  static constexpr int STAGE = W_ST + X_ST;
  static constexpr int EPI = NT * (MODE == 1 ? PITCH1 : PITCH2);
  static constexpr int OFF_SR = NST * STAGE;
  static constexpr int OFF_EPI = OFF_SR + NSR * S_SLOT;
  static constexpr int OFF_ARR = OFF_EPI + EPI;  // s_off[E+1] | s_tp[E+1]
  static constexpr int OFF_BAR = (OFF_ARR + 2 * (MAXE + 1) * 4 + 7) / 8 * 8;
  static constexpr int B_FULL = 0, B_EMPTY = NST, B_SFULL = 2 * NST, B_SEMPTY = 2 * NST + NSR, NBAR = 2 * NST + 2 * NSR;
  static constexpr int SMEM = 1024 + OFF_BAR + 8 * NBAR;
  static_assert(STAGE % 1024 == 0 && OFF_SR % 1024 == 0 && S_SLOT % 128 == 0 && OFF_EPI % 16 == 0, "alignment");
  static_assert(CTAS * (SMEM + 1024) <= 228 * 1024, "moe_dec: CTAs per SM");
  // lane-row -> stage-row permutation inside 8 rows: bank-conflict-free 8-byte weight loads for the swizzle of the box width
  // (KPS 1: no swizzle, 32-B rows -> rows & 3 distinct per half warp; KPS 2: 64B swizzle (16-B chunk ^= (row >> 1) & 3, two
  // rows per 128-B bank line); KPS 4: 128B swizzle (chunk ^= row & 7))
  __host__ __device__ static constexpr int perm8(int g) {
    return KPS == 1 ? g : KPS == 2 ? ((0x76325410 >> (4 * g)) & 15) : ((0x75316420 >> (4 * g)) & 15);
  }
  // byte offset of the 8-byte piece (stage row r, K block j of the stage, lane quad t) inside the weight box
  __host__ __device__ static constexpr uint32_t woff(int r, int j, int t) {
    return uint32_t(r * RB + 16 * ((2 * j + (t >> 1)) ^ (KPS == 1 ? 0 : KPS == 2 ? ((r >> 1) & 3) : (r & 7))) + 8 * (t & 1));
  }
};

// unit = (expert, chunk of <= NT routed rows, 128-row weight block); the weight block runs fastest, so consecutive units (=
// neighbouring CTAs at the same time) share the token rows through L2
struct DecUnit {
  int e, nb, pos0, n;
};
template <int NT>
__device__ __forceinline__ DecUnit dec_decode(int u, const int* s_off, const int* s_tp, const DecArgs& a) {
  int lo = 0, hi = a.E;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (s_tp[mid] * a.NB <= u)
      lo = mid;
    else
      hi = mid;
  }
  DecUnit r;
  r.e = lo;
  const int local = u - s_tp[lo] * a.NB, chunk = local / a.NB;
  r.nb = local - chunk * a.NB;
  r.pos0 = s_off[lo] + NT * chunk;
  r.n = min(NT, s_off[lo + 1] - r.pos0);
  return r;
}

// DS41H_MOE_DEC_PROF=1 (diagnostic build ds41h_moe_dec_prof, never the serving build): clock64 accounting of where the warps
// spend their time; summed over all consumer warps (lane 0) / producer lanes with atomicAdd into g_dec_prof.
#ifndef DS41H_MOE_DEC_PROF
#define DS41H_MOE_DEC_PROF 0
#endif
#if DS41H_MOE_DEC_PROF
// [0] consumer FULL waits [1] SFULL waits [2] stage tail (fence + arrive) [3] K loops total [4] epilogues [5] consumer total
// [6] consumer warps [7] units [8] producer EMPTY waits [9] SEMPTY waits [10] producer total [11] producers
__device__ unsigned long long g_dec_prof[16];
#endif

struct DecCons {
#if DS41H_MOE_DEC_PROF
  unsigned long long pf[3];  // FULL wait, SFULL wait, stage tail
#endif
  uint32_t sb;
  int stage, ss;
  unsigned phase, sph;
  int t;
  uint32_t aoff[2][2][4];  // [tile][row g / g + 8][K block of the stage] byte offsets of the lane's 8 weight bytes in a stage
  uint32_t soff[2][2];     // [tile][row g / g + 8] byte offsets of the lane's scale byte inside a scale slot (incl. t >> 1)
  uint32_t xo[4];          // [k16 step] ldmatrix byte offset of this lane inside a token tile (SW128)
  unsigned rot;            // 4 t: right rotation of the lane's weight words (small16 K order)
};

// One unit's K loop with NG n8 token groups; acc[tile][group][4]
template <class C, int NG>
__device__ __forceinline__ void dec_mainloop(DecCons& c, const DecArgs& a, float (&acc)[2][C::NT / 8][4]) {
  constexpr int MODE = C::MODE, KPS = C::KPS;
  const int nkb = a.K / 64, nks = (nkb + KPS - 1) / KPS;
  const int lane = threadIdx.x & 31;
  const unsigned rot = c.rot;
  const unsigned k01 = a.k01, k23 = a.k23, kshr8 = a.kshr8;
  auto bar = [&](int i) { return c.sb + C::OFF_BAR + 8 * i; };
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int q = 0; q < NG; ++q)
#pragma unroll
      for (int v = 0; v < 4; ++v) acc[i][q][v] = 0.f;
#pragma unroll 1
  for (int ks = 0; ks < nks; ++ks) {
    // scale slot: MODE 1 one slot per 8 K blocks (KPS divides 8 -> group boundaries are stage boundaries), MODE 2 one per unit
    const bool sg_first = MODE == 1 ? ((ks * KPS) & 7) == 0 : ks == 0;
    const bool sg_last = MODE == 1 ? (((ks + 1) * KPS) & 7) == 0 || ks == nks - 1 : ks == nks - 1;
#if DS41H_MOE_DEC_PROF
    unsigned long long p0 = clock64();
    mbar_wait_s(bar(C::B_FULL + c.stage), c.phase);
    unsigned long long p1 = clock64();
    c.pf[0] += p1 - p0;
    if (sg_first) mbar_wait_s(bar(C::B_SFULL + c.ss), c.sph);
    c.pf[1] += clock64() - p1;
#else
    mbar_wait_s(bar(C::B_FULL + c.stage), c.phase);
    if (sg_first) mbar_wait_s(bar(C::B_SFULL + c.ss), c.sph);
#endif
    const uint32_t st = c.sb + c.stage * C::STAGE;
    const uint32_t ssb = c.sb + C::OFF_SR + c.ss * S_SLOT + (MODE == 1 ? 2 * ((ks * KPS) & 7) : 2 * ks * KPS);
#pragma unroll
    for (int j = 0; j < KPS; ++j) {
      if (KPS > 1 && ks * KPS + j >= nkb) break;  // K tail (MODE 2: 9 blocks): the zero-filled rest of the box is not read
      unsigned w[2][2][2], m[2][2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint2 q0 = lds64(st + c.aoff[i][0][j]);
        const uint2 q1 = lds64(st + c.aoff[i][1][j]);
        w[i][0][0] = __funnelshift_r(q0.x, q0.x, rot), w[i][0][1] = __funnelshift_r(q0.y, q0.y, rot);
        w[i][1][0] = __funnelshift_r(q1.x, q1.x, rot), w[i][1][1] = __funnelshift_r(q1.y, q1.y, rot);
        m[i][0] = fp4_bf16_mult(lds_u8(ssb + 2 * j + c.soff[i][0]));
        m[i][1] = fp4_bf16_mult(lds_u8(ssb + 2 * j + c.soff[i][1]));
      }
      const uint32_t xs = st + C::W_ST + j * C::X_KB;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        unsigned A[2][4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          // word s >> 1 of rows g / g+8 (rotated); pairs 0,1 (s even) or 2,3 (s odd = pairs 0,1 of word >> 8)
          unsigned x0 = w[i][0][s >> 1], x1 = w[i][1][s >> 1];
          if (s & 1) x0 = mul_hi(x0, kshr8), x1 = mul_hi(x1, kshr8);
          A[i][0] = bf16x2_mul_rn(mul_lo(x0 & 0x000F000Fu, k01) & 0x81C081C0u, m[i][0]);
          A[i][2] = bf16x2_mul_rn(mul_lo(x0 & 0x00F000F0u, k23) & 0x81C081C0u, m[i][0]);
          A[i][1] = bf16x2_mul_rn(mul_lo(x1 & 0x000F000Fu, k01) & 0x81C081C0u, m[i][1]);
          A[i][3] = bf16x2_mul_rn(mul_lo(x1 & 0x00F000F0u, k23) & 0x81C081C0u, m[i][1]);
        }
        const uint32_t xa = xs + c.xo[s];
#pragma unroll
        for (int gp = 0; gp < (NG + 1) / 2; ++gp) {
          unsigned b0, b1, b2 = 0, b3 = 0;
          if (2 * gp + 1 < NG)
            ldsm_x4(xa + gp * 16 * 128, b0, b1, b2, b3);
          else
            ldsm_x2(xa + gp * 16 * 128, b0, b1);  // (lanes 16..31 addresses unused)
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            mma16816_bf16(acc[i][2 * gp], A[i], b0, b1);
            if (2 * gp + 1 < NG) mma16816_bf16(acc[i][2 * gp + 1], A[i], b2, b3);
          }
        }
      }
    }
    // cross-proxy WAR fence (docs/EXACTNESS.md) before the stage (and, at a group end, the scale slot) go back to the TMA producer:
    // the generic smem reads above (lds / ldmatrix) must not be overtaken by the producer's next async-proxy writes
#if DS41H_MOE_DEC_PROF
    const unsigned long long p2 = clock64();
#endif
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) {
      mbar_arrive_s(bar(C::B_EMPTY + c.stage));
      if (sg_last) mbar_arrive_s(bar(C::B_SEMPTY + c.ss));
    }
#if DS41H_MOE_DEC_PROF
    c.pf[2] += clock64() - p2;
#endif
    if (++c.stage == C::NST) c.stage = 0, c.phase ^= 1;
    if (sg_last && ++c.ss == NSR) c.ss = 0, c.sph ^= 1;
  }
}

template <class C, int NG = C::NGMAX>
__device__ __forceinline__ void dec_dispatch(int ng, DecCons& c, const DecArgs& a, float (&acc)[2][C::NT / 8][4]) {
  if constexpr (NG >= 1) {
    if (ng == NG) {
      dec_mainloop<C, NG>(c, a, acc);
      return;
    }
    dec_dispatch<C, NG - 1>(ng, c, a, acc);
  }
}

// inverse of a16_phys inside a 64-wide block (logical column of physical column p)
__device__ __forceinline__ int a16_inv(int p) {
  int r = 0;
#pragma unroll 1
  for (int L = 0; L < 64; ++L) r = a16_phys(L) == p ? L : r;
  return r;
}

template <int MODE, int KPS, int NST, int NGMAX, int CTAS>
__global__ void __launch_bounds__(NTHR, CTAS) moe_dec_kernel(const __grid_constant__ DecArgs a) {
  using C = DecCfg<MODE, KPS, NST, NGMAX, CTAS>;
  constexpr int NT = C::NT;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t sb = smem_u32(sm);
  unsigned long long* bars = reinterpret_cast<unsigned long long*>(sm + C::OFF_BAR);
  auto bar = [&](int i) { return sb + C::OFF_BAR + 8 * i; };
  int* s_off = reinterpret_cast<int*>(sm + C::OFF_ARR);
  int* s_tp = s_off + (MAXE + 1);
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) mbar_init(&bars[C::B_FULL + s], 1), mbar_init(&bars[C::B_EMPTY + s], NCW);
    for (int s = 0; s < NSR; ++s) mbar_init(&bars[C::B_SFULL + s], 1), mbar_init(&bars[C::B_SEMPTY + s], NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  pdl_wait();  // off / activations come from the preceding kernels
  asm volatile("fence.proxy.async.global;" ::: "memory");
  const int E = a.E;
  // per-expert row offsets and the exclusive prefix of units per weight block (chunks), every CTA the same
  for (int e = tid; e <= E; e += NTHR) s_off[e] = a.off[e];
  __syncthreads();
  if (warp == 0) {
    int carry = 0;
    for (int e0 = 0; e0 < E; e0 += 32) {
      const int e = e0 + lane;
      const int v = e < E ? (s_off[e + 1] - s_off[e] + NT - 1) / NT : 0;
      int inc = v;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
      }
      if (e < E) s_tp[e] = carry + inc - v;
      carry += __shfl_sync(0xffffffffu, inc, 31);
    }
    if (lane == 0) s_tp[E] = carry;
  }
  __syncthreads();
  const int total = s_tp[E] * a.NB;
  const int nkb = a.K / 64, nks = (nkb + KPS - 1) / KPS;
  if (warp == NCW) {
    // ------------------------------------------------------------------ TMA producer (lane 0)
    if (lane != 0) return;
    const int rows_e = int(a.w_estride / a.w_rstride);
    const bool hint = a.whint != 0;
    const uint64_t pol = policy_evict_first();
    auto wload = [&](uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t b) {
      if (hint)
        tma_load_2d_hint(dst, map, c0, c1, b, pol);
      else
        tma_load_2d(dst, map, c0, c1, b);
    };
    int stage = 0, ss = 0;
    unsigned phase = 0, sph = 0;
#if DS41H_MOE_DEC_PROF
    unsigned long long q_empty = 0, q_sempty = 0;
    const unsigned long long q_t0 = clock64();
#endif
    for (int u = blockIdx.x; u < total; u += gridDim.x) {
      const DecUnit un = dec_decode<NT>(u, s_off, s_tp, a);
      const int w0 = un.e * rows_e + (MODE == 1 ? 64 * un.nb : 128 * un.nb);
      const int br = a.brows;  // token rows per box: 8 (fewer only if the activation tensor has < 8 rows: one box)
      const int nbx = (un.n + br - 1) / br;
      for (int ks = 0; ks < nks; ++ks) {
        const int kb0 = ks * KPS;
        if (MODE == 1 ? (kb0 & 7) == 0 : ks == 0) {
#if DS41H_MOE_DEC_PROF
          const unsigned long long q0 = clock64();
          mbar_wait_s(bar(C::B_SEMPTY + ss), sph ^ 1);
          q_sempty += clock64() - q0;
#else
          mbar_wait_s(bar(C::B_SEMPTY + ss), sph ^ 1);
#endif
          const uint32_t sd = sb + C::OFF_SR + ss * S_SLOT, sf = bar(C::B_SFULL + ss);
          if constexpr (MODE == 1) {
            mbar_expect_tx_s(sf, 2048u);
            wload(sd, &a.tm_s, 16 * (kb0 >> 3), w0, sf);                // gate rows: 16-byte group of 8 K blocks
            wload(sd + 1024, &a.tm_s, 16 * (kb0 >> 3), w0 + a.I, sf);  // up rows
          } else {
            const uint32_t sbytes = uint32_t(128 * a.s_rstride);
            mbar_expect_tx_s(sf, sbytes);
            if (hint)
              bulk_load_hint(sd, a.s + long(w0) * a.s_rstride, sbytes, sf, pol);  // the unit's 128 scale rows (contiguous)
            else
              bulk_load(sd, a.s + long(w0) * a.s_rstride, sbytes, sf);
          }
          if (++ss == NSR) ss = 0, sph ^= 1;
        }
        const int nv = nkb - kb0 < KPS ? nkb - kb0 : KPS;  // valid K blocks of this stage
#if DS41H_MOE_DEC_PROF
        const unsigned long long q1 = clock64();
        mbar_wait_s(bar(C::B_EMPTY + stage), phase ^ 1);
        q_empty += clock64() - q1;
#else
        mbar_wait_s(bar(C::B_EMPTY + stage), phase ^ 1);
#endif
        const uint32_t st = sb + stage * C::STAGE, fb = bar(C::B_FULL + stage);
        mbar_expect_tx_s(fb, uint32_t(C::W_ST + nv * nbx * br * 128));
        if constexpr (MODE == 1) {
          wload(st, &a.tm_w, kb0 * 32, w0, fb);                     // 64 gate rows x 32 KPS bytes
          wload(st + 64 * C::RB, &a.tm_w, kb0 * 32, w0 + a.I, fb);  // 64 up rows
        } else {
          wload(st, &a.tm_w, kb0 * 32, w0, fb);  // 128 rows (K tail: bytes beyond the row -> 0, full box counted)
        }
        for (int j = 0; j < nv; ++j)
          for (int b = 0; b < nbx; ++b)
            tma_load_2d(st + C::W_ST + j * C::X_KB + br * 128 * b, &a.tm_b, (kb0 + j) * 128, un.pos0 + br * b, fb);
        if (++stage == NST) stage = 0, phase ^= 1;
      }
    }
#if DS41H_MOE_DEC_PROF
    atomicAdd(&g_dec_prof[8], q_empty);
    atomicAdd(&g_dec_prof[9], q_sempty);
    atomicAdd(&g_dec_prof[10], clock64() - q_t0);
    atomicAdd(&g_dec_prof[11], 1ull);
#endif
    return;
  }
  // -------------------------------------------------------------------- consumer warps
  DecCons c;
  c.sb = sb, c.stage = 0, c.ss = 0, c.phase = 0, c.sph = 0;
#if DS41H_MOE_DEC_PROF
  c.pf[0] = c.pf[1] = c.pf[2] = 0;
  unsigned long long r_loop = 0, r_epi = 0, r_units = 0;
  const unsigned long long r_t0 = clock64();
#endif
  const int g = lane >> 2, t = lane & 3;
  c.t = t;
  int col[2][2];  // [tile][row g / g + 8]: MODE 1 logical h column of the gate row (index 0); MODE 2 output column in the block
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int r0, r1;  // stage rows of A rows g / g + 8
    if constexpr (MODE == 1) {
      r0 = 16 * warp + 8 * i + C::perm8(g);  // gate row (stage rows 0..63) = physical h column of the block
      r1 = 64 + r0;                          // its up row (stage rows 64..127)
      col[i][0] = a16_inv(r0), col[i][1] = 0;
    } else {
      r0 = 32 * warp + 16 * i + C::perm8(g);
      r1 = r0 + 8;
      col[i][0] = r0, col[i][1] = r1;
    }
#pragma unroll
    for (int j = 0; j < KPS; ++j) c.aoff[i][0][j] = opaque(C::woff(r0, j, t)), c.aoff[i][1][j] = opaque(C::woff(r1, j, t));
    if constexpr (MODE == 1) {
      c.soff[i][0] = opaque(uint32_t(r0 * 16 + (t >> 1))), c.soff[i][1] = opaque(uint32_t(1024 + r0 * 16 + (t >> 1)));
    } else {
      c.soff[i][0] = opaque(uint32_t(r0 * a.s_rstride + (t >> 1))), c.soff[i][1] = opaque(uint32_t(r1 * a.s_rstride + (t >> 1)));
    }
  }
  // ldmatrix row address of this lane (matrix mi = lane >> 3: tokens 8 (mi >> 1) + (lane & 7) of a 16-token group pair,
  // 16-byte chunk 2 s + (mi & 1), SW128: chunk ^ (row & 7)), relative to the token tile of a K block
  {
    const int mi = lane >> 3, r7 = lane & 7;
#pragma unroll
    for (int s = 0; s < 4; ++s) c.xo[s] = opaque(uint32_t((8 * (mi >> 1) + r7) * 128 + (((2 * s + (mi & 1)) ^ r7) << 4)));
  }
  c.rot = opaque(4u * unsigned(t));
  const uint32_t epi = sb + C::OFF_EPI;
  for (int u = blockIdx.x; u < total; u += gridDim.x) {
    const DecUnit un = dec_decode<NT>(u, s_off, s_tp, a);
    float acc[2][NT / 8][4];
#if DS41H_MOE_DEC_PROF
    const unsigned long long r0 = clock64();
    dec_dispatch<C>((un.n + 7) >> 3, c, a, acc);
    const unsigned long long r1 = clock64();
    r_loop += r1 - r0;
    ++r_units;
#else
    dec_dispatch<C>((un.n + 7) >> 3, c, a, acc);
#endif
    asm volatile("bar.sync 1, 128;" ::: "memory");  // staging tile free (previous unit copied out)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int q = 0; q < NT / 8; ++q) {
        if (8 * q >= un.n) break;
        const int tok = 8 * q + 2 * t;
        if constexpr (MODE == 1) {
#pragma unroll
          for (int hv = 0; hv < 2; ++hv) {  // stock clamped SiLU (small16 / LD 4 epilogue): row g = gate, row g + 8 = up
            const float gt = fminf(acc[i][q][hv], a.limit);
            const float up = fmaxf(fminf(acc[i][q][2 + hv], a.limit), -a.limit);
            const float h = __fmul_rn(__fdiv_rn(gt, __fadd_rn(1.f, __nv_expf(-gt))), up);
            const __nv_bfloat16 hb = __float2bfloat16_rn(h);
            asm volatile("st.shared.b16 [%0], %1;" ::"r"(epi + uint32_t((tok + hv) * PITCH1 + 2 * col[i][0])),
                         "h"(*reinterpret_cast<const unsigned short*>(&hb)) : "memory");
          }
        } else {
#pragma unroll
          for (int hv = 0; hv < 2; ++hv)
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
              const __nv_bfloat16 yb = __float2bfloat16_rn(acc[i][q][2 * rr + hv]);
              asm volatile("st.shared.b16 [%0], %1;" ::"r"(epi + uint32_t((tok + hv) * PITCH2 + 2 * col[i][rr])),
                           "h"(*reinterpret_cast<const unsigned short*>(&yb)) : "memory");
            }
        }
      }
    asm volatile("bar.sync 1, 128;" ::: "memory");  // staging tile complete
    if constexpr (MODE == 1) {  // h rows pos0 + r, columns [64 nb, 64 nb + 64): 8 x 16 B per row
      for (int idx = tid; idx < un.n * 8; idx += 128) {
        const int r = idx >> 3, ch = idx & 7;
        stg128(a.out + long(un.pos0 + r) * a.out_stride + 64 * un.nb + 8 * ch, lds128(epi + r * PITCH1 + 16 * ch));
      }
    } else {  // slot rows perm_slot[pos0 + r], columns [128 nb, 128 nb + 128): 16 x 16 B per row
      for (int idx = tid; idx < un.n * 16; idx += 128) {
        const int r = idx >> 4, ch = idx & 15;
        const long row = a.perm_slot[un.pos0 + r];
        stg128(a.out + row * a.out_stride + 128 * un.nb + 8 * ch, lds128(epi + r * PITCH2 + 16 * ch));
      }
    }
#if DS41H_MOE_DEC_PROF
    r_epi += clock64() - r1;
#endif
  }
#if DS41H_MOE_DEC_PROF
  if (lane == 0) {
    atomicAdd(&g_dec_prof[0], c.pf[0]);
    atomicAdd(&g_dec_prof[1], c.pf[1]);
    atomicAdd(&g_dec_prof[2], c.pf[2]);
    atomicAdd(&g_dec_prof[3], r_loop);
    atomicAdd(&g_dec_prof[4], r_epi);
    atomicAdd(&g_dec_prof[5], clock64() - r_t0);
    atomicAdd(&g_dec_prof[6], 1ull);
    atomicAdd(&g_dec_prof[7], r_units);
  }
#endif
}

#ifndef DS41H_DEVICE_ONLY
// ---------------------------------------------------------------------------------------- host
int num_sms() {
  static const int n = [] {
    int dev = 0, v = 0;
    TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
    TORCH_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess);
    return v;
  }();
  return n;
}

PFN_cuTensorMapEncodeTiled_v12000 tmap_encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q{};
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q) == cudaSuccess && ptr,
                "cuTensorMapEncodeTiled not available");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

void encode_u8_2d(CUtensorMap* map, const void* base, uint64_t inner, uint64_t rows, uint32_t box_inner, uint32_t box_rows,
                  CUtensorMapSwizzle swz, const char* name) {
  const cuuint64_t dims[2] = {cuuint64_t(inner), cuuint64_t(rows)}, strides[1] = {cuuint64_t(inner)};
  const cuuint32_t box[2] = {box_inner, box_rows}, es[2] = {1u, 1u};
  const CUresult r = tmap_encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box, es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "moe_dec: cuTensorMapEncodeTiled(", name, ") failed: ", int(r));
}

// Shape conditions (MODE 1: I % 64 == 0, H % 64 == 0; MODE 2: H % 128 == 0, K % 64 == 0, 128 scale rows <= S_SLOT bytes)
bool dec_ok(int mode, long rows_e, int K, int E, int s_rstride, const void* s) {
  if (K % 64 != 0 || E > MAXE || (reinterpret_cast<uintptr_t>(s) & 15) != 0) return false;
  if (mode == 1) return rows_e % 128 == 0 && (rows_e / 2) % 64 == 0;
  return rows_e % 128 == 0 && 128 * s_rstride <= S_SLOT && (128 * s_rstride) % 16 == 0;
}

template <int MODE, int KPS, int NST, int NGMAX, int CTAS>
void launch_dec(DecArgs a, const uint8_t* w, const uint8_t* act, long act_rows, long rows_total) {
  using C = DecCfg<MODE, KPS, NST, NGMAX, CTAS>;
  a.brows = act_rows < 8 ? int(act_rows) : 8;
  encode_u8_2d(&a.tm_b, act, uint64_t(2 * a.K), uint64_t(act_rows), 128u, uint32_t(a.brows), CU_TENSOR_MAP_SWIZZLE_128B, "act");
  encode_u8_2d(&a.tm_w, w, uint64_t(a.w_rstride), uint64_t(rows_total), uint32_t(C::RB), MODE == 1 ? 64u : 128u,
               KPS == 1 ? CU_TENSOR_MAP_SWIZZLE_NONE : KPS == 2 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B, "w");
  if (MODE == 1) encode_u8_2d(&a.tm_s, a.s, uint64_t(a.s_rstride), uint64_t(rows_total), 16u, 64u, CU_TENSOR_MAP_SWIZZLE_NONE, "s");
  opt_in_smem<moe_dec_kernel<MODE, KPS, NST, NGMAX, CTAS>>(C::SMEM);
  launch_pdl("moe_dec", moe_dec_kernel<MODE, KPS, NST, NGMAX, CTAS>, dim3(CTAS * num_sms()), dim3(NTHR), C::SMEM, a);
}

// Variants (both GEMMs; forward cfg = v1 | v2 << 4): KPS (K blocks per stage) / NST (stages) / NGMAX (n8 token groups per unit,
// NT = 8 NGMAX tokens) / CTAS per SM
template <int MODE>
void launch_dec_v(int v, DecArgs a, const uint8_t* w, const uint8_t* act, long act_rows, long rows_total) {
  switch (v) {
    case 0: launch_dec<MODE, 1, 6, 2, 4>(a, w, act, act_rows, rows_total); break;  // small16 shape (4 KB weight stages), NT 16
    case 1: launch_dec<MODE, 2, 3, 2, 4>(a, w, act, act_rows, rows_total); break;  // 64-byte rows
    case 2: launch_dec<MODE, 4, 2, 2, 3>(a, w, act, act_rows, rows_total); break;  // 128-byte rows
    case 3: launch_dec<MODE, 4, 3, 2, 2>(a, w, act, act_rows, rows_total); break;
    case 4: launch_dec<MODE, 1, 6, 4, 3>(a, w, act, act_rows, rows_total); break;  // NT 32
    case 5: launch_dec<MODE, 2, 3, 4, 3>(a, w, act, act_rows, rows_total); break;
    case 6: launch_dec<MODE, 4, 2, 4, 2>(a, w, act, act_rows, rows_total); break;
    case 7: launch_dec<MODE, 2, 4, 4, 2>(a, w, act, act_rows, rows_total); break;
    case 8: launch_dec<MODE, 4, 4, 2, 2>(a, w, act, act_rows, rows_total); break;  // deeper 128-byte ring
    case 9: launch_dec<MODE, 2, 4, 2, 3>(a, w, act, act_rows, rows_total); break;
    default: TORCH_CHECK(false, "moe_dec: variant must be 0..9, got ", v);
  }
}
#endif  // DS41H_DEVICE_ONLY
}  // namespace

#ifndef DS41H_DEVICE_ONLY
// Route-side kernels (moe_route.cu, compiled into this extension unchanged)
void ds41h_moe_permute(torch::Tensor topk_ids, torch::Tensor topk_w, int64_t E, torch::Tensor meta, torch::Tensor perm_w);
void ds41h_act_perm16(torch::Tensor x, torch::Tensor pos_of, int64_t K, torch::Tensor xp);
void ds41h_moe_combine(torch::Tensor slot, torch::Tensor topk_ids, int64_t E, torch::Tensor out, c10::optional<torch::Tensor> sscale,
                       c10::optional<torch::Tensor> addend);
int64_t ds41h_moe_meta_size(int64_t E, int64_t R);

static void check_u8(const torch::Tensor& t, std::initializer_list<int64_t> shape, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dtype() == torch::kUInt8, name, ": expected contiguous CUDA uint8");
  TORCH_CHECK(t.sizes() == c10::IntArrayRef(shape), name, ": shape ", t.sizes(), " != ", c10::IntArrayRef(shape));
}

// One grouped GEMM of the W4A16 small-T path (same contract as ds41h_moe_gemm with a small16 variant):
//   MODE 1: act = xp bf16 [R, H] (routed-row order, W4A16 logical K order), w = w13 [E, 2I, H/2], s = s13 -> out = h bf16 [R, I]
//   MODE 2: act = h bf16 [R, I], w = w2 [E, H, I/2], s = s2 -> out = unscaled slot rows bf16 [T*K, H]
// meta = moe_permute output (int32), R = routed rows. Returns false if the shape is unsupported (caller falls back).
bool ds41h_moe_dec_gemm(int64_t mode, torch::Tensor act, torch::Tensor w, torch::Tensor s, torch::Tensor meta, int64_t R, int64_t variant,
                        double limit, torch::Tensor out) {
  TORCH_CHECK(mode == 1 || mode == 2);
  TORCH_CHECK(w.dim() == 3 && s.dim() == 3);
  const int E = w.size(0), rows = w.size(1), K = w.size(2) * 2;
  check_u8(w, {E, rows, K / 2}, "w");
  check_u8(s, {E, rows, K / 32}, "s");
  TORCH_CHECK(act.is_cuda() && act.is_contiguous() && act.dtype() == torch::kBFloat16 && act.numel() % K == 0 && act.numel() >= R * K,
              "moe_dec: activations must be contiguous bf16 [>= R, K]");
  TORCH_CHECK(meta.dtype() == torch::kInt32 && meta.numel() >= ds41h_moe_meta_size(E, R));
  TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous());
  if (!dec_ok(int(mode), rows, K, E, K / 32, s.data_ptr())) return false;
  DecArgs a{};
  int* m = mptr<int>(meta);
  a.s = cptr<uint8_t>(s);
  a.off = m + (E + 1);
  a.perm_slot = m + 2 * (E + 1) + META_CTR + R;
  a.out = mptr<__nv_bfloat16>(out);
  a.w_estride = long(rows) * (K / 2);
  a.w_rstride = K / 2;
  a.s_rstride = K / 32;
  a.K = K;
  a.E = E;
  a.limit = float(limit);
  a.k01 = 0x1040u, a.k23 = 0x104u, a.kshr8 = 1u << 24;
  static const int env_hint = [] { const char* v = std::getenv("DS41H_MOE_DEC_HINT"); return v && *v && *v != '0' ? 1 : 0; }();
  a.whint = (variant & 0x10) ? 1 : env_hint;  // per-GEMM variant bit 4 (forward cfg bit 4 / 12) or DS41H_MOE_DEC_HINT=1
  variant &= 15;
  if (mode == 1) {
    a.I = rows / 2;
    a.NB = (rows / 2) / 64;
    a.out_stride = rows / 2;
    TORCH_CHECK(out.numel() >= R * (rows / 2), "moe_dec mode 1: h must hold R rows of I");
  } else {
    a.I = 0;
    a.NB = rows / 128;
    a.out_stride = rows;
    TORCH_CHECK(out.numel() >= R * rows, "moe_dec mode 2: slot buffer must hold R rows of H");
  }
  if (R == 0) return true;
  const long act_rows = act.numel() / K;
  if (mode == 1)
    launch_dec_v<1>(int(variant), a, cptr<uint8_t>(w), reinterpret_cast<const uint8_t*>(act.data_ptr()), act_rows, long(E) * rows);
  else
    launch_dec_v<2>(int(variant), a, cptr<uint8_t>(w), reinterpret_cast<const uint8_t*>(act.data_ptr()), act_rows, long(E) * rows);
  return true;
}

// Full routed-expert forward on the raw layout (= ds41h_moe_forward with variant 0x500, other GEMM kernels):
// permute -> act_perm16 -> GEMM1 (moe_dec MODE 1) -> GEMM2 (moe_dec MODE 2) -> moe_combine (+ addend).
// cfg = v1 | v2 << 4 | hint1 << 8 | hint2 << 9 (hint = L2 evict_first on the weight / scale loads; DS41H_MOE_DEC_HINT=1 = both).
torch::Tensor ds41h_moe_dec_forward(torch::Tensor x, torch::Tensor topk_w, torch::Tensor topk_ids, torch::Tensor w13, torch::Tensor s13,
                                    torch::Tensor w2, torch::Tensor s2, double limit, int64_t cfg, c10::optional<torch::Tensor> addend) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2);
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(0) == x.size(0) && topk_w.sizes() == topk_ids.sizes(),
              "moe_dec_forward: topk_ids/topk_weights must be [T, K] for x [T, H]");
  x = x.contiguous();
  const int64_t T = x.size(0), H = x.size(1), E = w13.size(0), I = w13.size(1) / 2, K = topk_ids.size(1), R = T * K;
  check_u8(w13, {E, 2 * I, H / 2}, "w13");
  check_u8(s13, {E, 2 * I, H / 32}, "s13");
  check_u8(w2, {E, H, I / 2}, "w2");
  check_u8(s2, {E, H, I / 32}, "s2");
  TORCH_CHECK(dec_ok(1, 2 * I, int(H), int(E), int(H / 32), s13.data_ptr()) && dec_ok(2, H, int(I), int(E), int(I / 32), s2.data_ptr()),
              "moe_dec_forward: unsupported shape");
  auto out = torch::empty({T, H}, x.options());
  if (T == 0) return out;
  auto ids = topk_ids.to(torch::kInt32).contiguous();
  auto tw = topk_w.to(torch::kFloat32).contiguous();
  auto meta = torch::empty({ds41h_moe_meta_size(E, R)}, x.options().dtype(torch::kInt32));
  auto perm_w = torch::empty({R}, x.options().dtype(torch::kFloat32));
  ds41h_moe_permute(ids, tw, E, meta, perm_w);
  const int64_t t0 = 2 * (E + 1) + META_CTR;
  auto h = torch::empty({R, I}, x.options());
  {
    auto xp = torch::empty({R, H}, x.options());  // freed before the slot buffer is allocated (same size)
    ds41h_act_perm16(x, meta.narrow(0, t0 + 2 * R, R), K, xp);
    ds41h_moe_dec_gemm(1, xp, w13, s13, meta, R, (cfg & 15) | (((cfg >> 8) & 1) << 4), limit, h);
  }
  auto slot = torch::empty({R, H}, x.options());
  ds41h_moe_dec_gemm(2, h, w2, s2, meta, R, ((cfg >> 4) & 15) | (((cfg >> 9) & 1) << 4), limit, slot);
  ds41h_moe_combine(slot, ids, E, out, tw.view({-1}), addend);
  return out;
}

#if DS41H_MOE_DEC_PROF
void ds41h_moe_dec_prof_reset() {
  unsigned long long z[16] = {};
  TORCH_CHECK(cudaMemcpyToSymbol(g_dec_prof, z, sizeof(z)) == cudaSuccess);
}
std::vector<int64_t> ds41h_moe_dec_prof_read() {
  unsigned long long v[16];
  TORCH_CHECK(cudaMemcpyFromSymbol(v, g_dec_prof, sizeof(v)) == cudaSuccess);
  return std::vector<int64_t>(v, v + 16);
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#if DS41H_MOE_DEC_PROF
  m.def("prof_reset", &ds41h_moe_dec_prof_reset);
  m.def("prof_read", &ds41h_moe_dec_prof_read);
#endif
  m.def("moe_forward", &ds41h_moe_dec_forward, "routed experts forward, small-T W4A16 (moe_dec kernels; bit-identical to small16 0x500)",
        py::arg("x"), py::arg("topk_w"), py::arg("topk_ids"), py::arg("w13"), py::arg("s13"), py::arg("w2"), py::arg("s2"),
        py::arg("limit"), py::arg("cfg") = 0, py::arg("addend") = py::none());
  m.def("moe_gemm", &ds41h_moe_dec_gemm, "one moe_dec grouped GEMM (mode 1: w13 + SwiGLU, mode 2: w2); false = unsupported shape");
  m.def("moe_permute", &ds41h_moe_permute, "expert permutation metadata (moe_route.cu)");
  m.def("act_perm16", &ds41h_act_perm16, "bf16 x -> routed-row order, W4A16 logical K order (moe_route.cu)");
  m.def("moe_combine", &ds41h_moe_combine, "sum top-k slot rows per token (moe_route.cu)", py::arg("slot"), py::arg("topk_ids"), py::arg("E"),
        py::arg("out"), py::arg("sscale") = py::none(), py::arg("addend") = py::none());
  m.def("meta_size", &ds41h_moe_meta_size, "int32 elements of the moe_permute meta buffer for (E, R)");
}
#endif  // DS41H_DEVICE_ONLY
