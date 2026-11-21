// c1mega v0.
//
// The post-attention half of one decode layer for the C1 bucket (NR <= 4 real rows, all owned by one sequence) as ONE
// persistent launch per GPU, TP4 over the node's 4 GPUs with the residual stream replicated (identical on all 4):
//   F  wo_a (this GPU's 2 groups) -> z bf16 -> wo_b (K quarter) fp32 partial -> lamport all-reduce #1 -> y bf16
//   G  FFN mHC boundary: xm2 = bf16(post y + comb^T X), in = bf16(sum lag xm2), the 24 mix dot products + sum of squares
//   H  RMSNorm -> f (bf16) + prescale p; mixes -> f_pre / f_post / f_comb (Sinkhorn, off the critical path);
//      router GEMV (bf16 gate, fp32) -> top-6 (sqrt-softplus + bias; k_router_top's statements); shared expert GEMM1
//   I  routed experts GEMM1 (FP4 raw layout, SwiGLU epilogue) -> GEMM2 + shared GEMM2 -> combine (bf16 partial)
//      -> lamport all-reduce #2 -> moe_out bf16
// Same math as the engine's pieces (ENG_C1_TPWO + MHCF + AGRT + RSC + RSL, moe_v4): exact checkpoint weights (fp8 e4m3 x
// ue8m0 32x32 blocks, FP4 e2m1 x ue8m0 per 32), fp32 accumulation, the engine's rounding points (z bf16, y bf16, xm2 /
// in / f bf16, shared gu / h / sh bf16, routed h f16 x 256 and Y bf16 x 1/256, combine bf16, moe_out bf16). Differences:
// summation order (K split over warps, mma k order), and 16-bit activations as f16 x 2^p (per-row power of two, the
// engine MoE's own scheme) for every fp8 / FP4 GEMM (the engine runs its dense GEMMs as bf16 x bf16; f16 x 2^p holds every
// bf16 value whose magnitude is >= 2^-31 of its row's maximum exactly).
//
// Execution: grid = 132 CTAs (one per SM), NCW consumer warps + 1 producer warp. The producer streams this CTA's weight
// slices of every stage, in program order, by cp.async.bulk into an NS-stage SMEM ring (it runs ahead across stage
// boundaries: the dense weights do not depend on activations; the routed experts wait for this CTA's own top-6, GEMM2
// chunks wait for the experts' h). Consumers walk the same chunk sequence. Every CTA owns a fixed column slice of the
// hidden dimension (2 or 3 tiles of 16 columns): wo_b's output, AR #1, the boundary, GEMM2, the combine and AR #2 work on
// that slice, so the all-reduces are per-CTA exchanges (CTA c of every GPU), no grid barrier. Grid-level dependencies
// are monotonic counters (release / acquire): z complete (wo_a -> wo_b), the boundary's sums of squares (-> RMSNorm),
// the router partials (-> top-6), and per expert h (GEMM1 -> GEMM2).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <torch/extension.h>

#ifndef C1M_NCW
#define C1M_NCW 8
#endif
#ifndef C1M_NS
#define C1M_NS 8
#endif

namespace c1m {
typedef __nv_bfloat16 bf16;
// build-time shape knobs (defaults = the main model; the C1 tail's drafter blocks build with -DC1M_NE=128 -DC1M_TOPK=3
// -DC1M_WIN=131 -DC1M_TKA=0): routed experts, experts per row, window slots per row, top-k keys per row in the attention
#ifndef C1M_NE
#define C1M_NE 384
#endif
#ifndef C1M_TOPK
#define C1M_TOPK 6
#endif
#ifndef C1M_WIN
#define C1M_WIN 128
#endif
#ifndef C1M_TKA
#define C1M_TKA 512
#endif
#ifndef C1M_NRM
#define C1M_NRM 4
#endif
// phase fragments textually included in k_layer (other sessions' files; alternative files for variants via -D):
#ifndef C1M_MOE_PRODUCE
#define C1M_MOE_PRODUCE "c1moe_x2b_produce.inc"
                                                   // grabbed GEMM1 items, GEMM2 interleaved; c1moe_produce.inc = before)
#endif
#ifndef C1M_MOE_CONSUME
#define C1M_MOE_CONSUME "c1moe_x2b_consume.inc"   // the expert phase (consumers; x2b; c1moe_consume.inc = before)
#endif
#ifndef C1M_HOOKS
#define C1M_HOOKS "c1hooks.inc"               // index / compressor / Engram hook points (empty by default)
#endif
constexpr int D = 5120, HC = 4, NE = C1M_NE, TOPK = C1M_TOPK, IR = 576, W = 4, NRM = C1M_NRM, NCTA = 132;
constexpr int WIN = C1M_WIN, TKA = C1M_TKA, NEW = (NE + 31) / 32;   // NEW: 32-bit words / lanes per expert set
static_assert(NE % 32 == 0 && NE <= 1024 && TOPK >= 1 && TOPK <= 8 && NRM == 4 && WIN >= 1 && WIN <= 256, "shape knobs");
constexpr int NCW = C1M_NCW, CTH = NCW * 32, THR = CTH + 64;   // consumer warps, the producer warp, the KV-push warp
constexpr int NSMAX = 8, STAGE = 20480;   // multiple of 1024 (128B-swizzled TMA boxes)
constexpr int MAXX = NRM * TOPK;
constexpr int RT_NP = 240, RT_KC = 512, RT_NKC = D / RT_KC;  // router pieces: 24 tiles of 16 experts x 10 K chunks
constexpr int SH_NU = 36, SH_NH = 72;                        // shared GEMM1: 36 tiles of 16 intermediates x 2 K halves
constexpr int IMS = D * 2 + 64;                              // activation image row stride in SMEM (bytes)
constexpr unsigned SENT = 0x80000000u;
constexpr int NTS = 48, NTIM = 4 * NTS;   // per CTA: stamps, data-wait ns, sync-wait ns, SM clock (clock64)
constexpr int KVR = WIN + TKA, STG_R = 592;   // mode-3 staging: records of 584 B at a 592-B (16-B aligned) stride, data then scales
constexpr int STG_V = NRM * KVR * STG_R, STG_P = (STG_V + NRM * KVR + 15) & ~15, STG_BYTES = STG_P + NRM * 8;   // per CTA: stamps [0, 32), data-wait ns [32, 64), sync-wait ns [64, 96) at each stamp
constexpr int AK = 32, ANCH = (KVR + AK - 1) / AK;  // FULL: attention key chunk, chunks per row
// counters (unsigned [C_N], monotonic; targets scale with the CTA's call index)
enum { C_Z = 0, C_SS = 1, C_RT = 2, C_SH = 3, C_SSA = 4, C_MIXA = 5, C_QA = 6, C_QB = 7, C_SHT = 8, C_H = 48, C_O = 72,
       C_AT = 73, C_MIXF = 77, C_INS = 78, C_N = 80 };   // C_INS: the rows' window inserts done (NR per layer, every rank)
// stage layouts (bytes). TMA boxes of [16 rows x 128 B] with the 128-B swizzle (2 KB each, 1024-B aligned): the 16-B chunk
// j of box row r lands at chunk j ^ (r & 7).
constexpr int L_D_S = 16384;                            // dense 16 x 1024 fp8 tile (8 boxes) + 32 scale bytes
constexpr int L_QB_S = 16384;                           // q_b: 64 rows x 256 K (one box: [2 kblocks][64][128]) + 2 scale rows (96 B)
constexpr int L_SH1_S = 16384;                          // shared GEMM1: gate 16 x 512 (4 boxes), up (4 boxes) at 8192, 2 x 16 B
constexpr int L_G1_S = 16384;                           // routed GEMM1: gate / up boxes as L_SH1, scale boxes [16][32] x 2
constexpr int L_G2_S = 48 * 288, L_G2_H = 14720, L_G2_HS = IR * 2 + 64;  // routed GEMM2: rows (288 B), scales, h image
constexpr int L_S2_S = 16 * IR, L_S2_HS = L_S2_S + 128, L_S2_HM = L_S2_HS + NRM * IR * 2;  // shared GEMM2, one tile per stage
constexpr int L_S2B_H = 5 * 2048;   // shared GEMM2 from the engine's bf16 w2: [5 k-blocks of 64][16 rows][128 B], then hsh
enum { TM_WOA = 0, TM_WOB, TM_WQKV, TM_WQB, TM_SW13, TM_W13, TM_S13, TM_FN32, TM_FN48, TM_AFN32, TM_AFN48, TM_WQKV16, TM_WOA16,
       TM_WOB16, TM_SW13_16, TM_SW2_16, TM_N };
static_assert(L_G1_S + 2 * 512 <= STAGE && L_G2_H + NRM * L_G2_HS <= STAGE && L_S2_HM + SH_NU * NRM * 4 <= STAGE, "stage");
static_assert(24 * 4 * 48 * 4 <= STAGE, "fn chunk");
static_assert(L_S2B_H + NRM * IR * 2 <= STAGE && 2 * 4 * 16 * 128 <= STAGE, "shared bf16 stages");

struct Args {
  int rank, NR, mode;   // mode (timing ablations, results invalid): 1 routed GEMMs without dequant + MMA, 2 no routed weight copies
  // inputs (compact real rows 0 .. NR-1)
  const bf16* og;     // [NR][8192]   this GPU's 16 heads of o (inverse RoPE applied)
  const bf16* X;      // [NR][4][D]   attention-boundary streams (xn of the attention boundary)
  const float* apost; // [NR][4]
  const float* acomb; // [NR][16]
  const float* apre;  // [NR][4]
  // weights
  const uint8_t* woa;    // [2048][4096] e4m3 (wo_a rows 2048 rank ..)
  const uint8_t* woa_s;  // [64][128] ue8m0
  const uint8_t* wob;    // [5120][8192] e4m3
  const uint8_t* wob_s;  // [160][256]
  const float* fn;       // hc_ffn_fn as per-CTA slices [132][96 = 24 x 4][48] (columns of the CTA's tiles, zero-padded)
  const float* fscale;   // [3]
  const float* fbase;    // [24]
  const float* nw;       // [D] ffn_norm weight (fp32)
  float norm_eps, hc_eps;
  int hc_iters;
  const bf16* gate_w;    // [384][D]
  const float* gate_b;   // [384]
  float route_scale;
  const uint8_t* sw13;   // [1152][D] e4m3: shared w1 rows 576 rank .. (576), then w3 rows (576)
  const uint8_t* sw13_s; // [36][160]
  const uint8_t* sw2;    // [D][576] e4m3 (shared w2 K columns 576 rank ..)
  const uint8_t* sw2_s;  // [160][32] (18 used)
  const uint8_t* w13;    // [384][1152][2560] FP4 raw (TP slice)
  const uint8_t* s13;    // [384][1152][160]
  const uint8_t* w2;     // [384][5120][288]
  const uint8_t* s2;     // [384][5120][18]
  // outputs
  bf16* y;        // [NR][D]
  bf16* xm2;      // [NR][4][D]
  bf16* fout;     // [NR][D]
  int* ptok;      // [NR]
  float* fpre;    // [NR][4]
  float* fpost;   // [NR][4]
  float* fcomb;   // [NR][16]
  float* logits;  // [NR][384]
  int* ids;       // [NR][6]
  float* wts;     // [NR][6]
  bf16* moe_out;  // [NR][D]
  // scratch
  bf16* z;        // [NRM][2048]
  bf16* inb;      // [NRM][D]
  float* ssp;     // [132][NRM]
  float* mixp;    // [132][NRM][25]
  float* rtp;     // [240][16][NRM]
  float* shp;     // [72][32][NRM]
  bf16* hsh;      // [NRM][576]
  float* hmax;    // [36][NRM]
  __half* himg;   // [24][NRM][608] (576 used, k32 blocks in P8 order; rows padded to 1216 B)
  unsigned* cnt;  // [C_N]
  unsigned* tick; // [132] per-CTA call counter
  float* ar1[W];  // rank s's AR #1 slots [2][W][NRM][D] fp32 (sentinel -0.0)
  bf16* ar2[W];   // rank s's AR #2 slots [2][W][NRM][D] bf16
  unsigned long long* tim;  // [132][NTIM] or null
  const CUtensorMap* tm;    // [TM_N] this layer's tensor maps (global memory, 64-B aligned)
  // ---- FULL (the whole layer): the front half + attention; the post half then reads og = o, X = Xo, the mixes apre_o ..
  const bf16* xin;      // [NR][D]     previous layer's output (moe_out) or the layer input
  const bf16* rsin;     // [NR][4][D]  previous streams (xm2)
  const float* pin;     // [NR][4]     previous boundary's post / comb / pre (f_post, f_comb, f_pre)
  const float* cin;     // [NR][16]
  const float* lin;     // [NR][4]
  const float* afn;     // hc_attn_fn, per-CTA slices as fn
  const float* ascale;  // [3]
  const float* abase;   // [24]
  const float* anw;     // [D] attn_norm (fp32)
  const uint8_t* wqkv;  // [1792][D] e4m3 (wq_a | wkv)
  const uint8_t* wqkv_s;  // [56][160]
  const float* q_norm;  // [1280]
  const float* kv_norm; // [512]
  float attn_eps;
  const uint8_t* wqb;   // [8192][1280] e4m3: wq_b rows of this GPU's 16 heads
  const uint8_t* wqb_s; // [256][48] (40 used)
  const float* sink;    // [64]
  float scale_log2;
  const float* cos_sin; // [pos][64]
  const long long* pos; // [NR]
  const long long* own; // [NR] window slot of the row's own record (-1: none)
  uint8_t* win;         // this layer's window pages (page p at p * win_stride: bs records of 576 B, then bs x 8 B scales)
  int win_shift, comp_shift, do_insert;
  long long win_stride, comp_stride;
  const int* si;        // [NR][128] window slots
  const int* sl;        // [NR]
  const uint8_t* comp;  // compressed pages of the layer's KV source (or null)
  const int* ti;        // [NR][512]
  const int* tl;        // [NR]
  bf16* Xo;             // [NR][4][D]
  bf16* ain;            // [NR][D]
  float* ssp_a;         // [132][NRM]
  float* mixp_a;        // [132][NRM][25]
  float* apre_o;        // [NR][4]
  float* apost_o;       // [NR][4]
  float* acomb_o;       // [NR][16]
  bf16* qakv;           // [NRM][1792]
  bf16* qrout;          // [NR][1280]
  bf16* kvout;          // [NR][512]
  bf16* q;              // [NRM][16][512] this GPU's heads, RoPE applied
  bf16* o;              // [NRM][16][512] attention output, inverse RoPE applied (= og)
  float* apart;         // [NRM * ANCH][16][512]
  float* aml;           // [NRM * ANCH][2][16] (owner mode: [NRM * 4 * ANCH])
  // attention mode: 0 = every GPU attends its 16 heads on a LOCAL copy of the KV; 1 = the owner (rank `owner`, all rows)
  // attends all 64 heads on its own KV: q heads land in the owner's qall, o slices in each GPU's ogl (lamport, -0.0)
  int amode, owner;
  bf16* qall[W];        // rank s: [2][NRM][64][512]
  bf16* ogl[W];         // rank s: [2][NRM][8192]
  // engine integration (patch_integ; null / 128 / 512 otherwise): this launch's first-layer inputs (xin rsin pin cin lin,
  // pos) are the OWNER's buffers (peer pointers on the other ranks); the owner's CTA c tells every peer's CTA c that they
  // are complete (rdy[s][c] = this launch's epoch, release.sys) before the peers read them
  unsigned* rdy[W];     // rank s: [132] input-ready flags per CTA, then [NRM] insert-done flags per row (amode 2)
  int si_rs, ti_rs;     // row strides (ints) of si / ti
  // wo_b: the whole [5120][8192] (k-block offset 16 rank, scale row stride 256, scale column offset 64 rank) or this rank's
  // K quarter [5120][2048] (0, 64, 0)
  int wob_k0, wob_sst, wob_sc0;
  // the engine's own bf16 tensors instead of the fp8 copies (patch_bf16; null = fp8): q_a | wkv [1792][5120], wo_a^T
  // [8][4096][1024], wo_b [5120][8192]; bf16 x bf16 MMA on natural bf16 images (exact products, fp32 accumulation)
  const bf16* wqkv16;
  const bf16* woaT16;
  const bf16* wob16;
  // attention mode 3 (owner push): rank s's staging [NRM][KVR] records (584 B at a 592-B stride), then valid bytes [NRM][KVR], then the
  // rows' positions [NRM] (int64); the owner pushes each row's selected KV records (window + top-k) there during the
  // layer's front half and counts every push into rdy[s][NCTA + NRM] (NCTA + NR arrivals per layer)
  uint8_t* stg[W];
  int push_topk;        // mode 3: this layer's top-k records are (re)pushed (the launch's first layer, in-kernel index layers)
  // phase extensions (null on layers without them): index / compressor and Engram phases (their own structs / files)
  const void* ix;
  const void* eg;
  // the engine's bf16 shared expert instead of the fp8 copies (patch_sh16; null = fp8): sw13 [1152][5120] (this rank's
  // w1 rows, then its w3 rows), sw2 [5120][576]; bf16 x bf16 MMA (the f16 P8 image converted to bf16: exact), fp32 sums
  const bf16* sw13_16;
  const bf16* sw2_16;
  // index layers (patch_ext): A2 also writes the normalized attention input a [NR][D] (natural bf16) here, every CTA the
  // whole rows (the same values; each CTA reads back its own writes); null elsewhere
  bf16* aout;
};
// deployment builds: -DC1M_FIX_AMODE=<mode> and -DC1M_FIX_B16=<bits: 1 q_a|wkv, 2 wo_a, 4 wo_b, 8 the
// shared expert> fix the attention mode and the bf16 / fp8 choice per matrix at compile time (the other paths compiled out,
// less code to fetch per launch; the host checks the Args against fixed_config()); -DC1M_NOTIM compiles the timeline out.
// Default: the runtime fields.
#ifdef C1M_FIX_AMODE
#define AMODE(x) (C1M_FIX_AMODE)
#else
#define AMODE(x) ((x).amode)
#endif
#ifdef C1M_FIX_B16
#define B16(x, f, bit) ((((C1M_FIX_B16) >> (bit)) & 1) != 0)
#else
#define B16(x, f, bit) ((x).f != nullptr)
#endif

// ------------------------------------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint32_t a, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_expect(uint32_t a, unsigned b) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(b) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mb_wait(uint32_t a, uint32_t par) {
  asm volatile(
      "{\n .reg .pred p;\n LAB_W%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra LAB_W%=;\n}" ::"r"(a),
      "r"(par)
      : "memory");
}
__device__ __forceinline__ uint64_t pol_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, unsigned bytes, uint32_t mbar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(mbar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void bulk_nc(uint32_t dst, const void* src, unsigned bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src),
               "r"(bytes), "r"(mbar)
               : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t mbar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
      "l"(tm), "r"(c0), "r"(c1), "r"(mbar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void tma3d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, int c2, uint32_t mbar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
      "l"(tm), "r"(c0), "r"(c1), "r"(c2), "r"(mbar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void tma4d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, int c2, int c3, uint32_t mbar,
                                      uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(dst),
      "l"(tm), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void pf_l2(const void* src, unsigned bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void red_rel(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void spin_ge(const unsigned* p, unsigned target) {
  while ((int)(ld_acq(p) - target) < 0) {
  }
}
__device__ __forceinline__ uint4 ld_vol4(const void* p) {
  uint4 u;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w) : "l"(p) : "memory");
  return u;
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
__device__ __forceinline__ unsigned lds32(uint32_t a) {
  unsigned v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void st_rel_sys(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acq_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_rel_sys(unsigned* p, unsigned v) {
  asm volatile("red.release.sys.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, %0;" ::"n"(CTH) : "memory"); }
// named barrier 2 (consumer threads): one warp arrives (a buffer is free), the other consumer warps wait for it
__device__ __forceinline__ void nb2_arrive() { asm volatile("bar.arrive 2, %0;" ::"n"(CTH) : "memory"); }
__device__ __forceinline__ void nb2_sync() { asm volatile("bar.sync 2, %0;" ::"n"(CTH) : "memory"); }
__device__ __forceinline__ void mma16816(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void hmma_bf16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                          unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
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
__device__ __forceinline__ unsigned fp4_lo_e4m3(unsigned w) { return ((w << 4) & 0x80808080u) | ((w << 2) & 0x1c1c1c1cu); }
__device__ __forceinline__ unsigned fp4_hi_e4m3(unsigned w) { return (w & 0x80808080u) | ((w >> 2) & 0x1c1c1c1cu); }
// moe_v4's dequant_pair: one k32 block of rows g (word wg) and g+8 (wh) -> the k16 step fragments s=0 (even nibbles), s=1
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
// FP4 multiplier f16x2 2^(s - 121) of one ue8m0 byte (moe_v4: bias4 + byte replicate; s >= 106)
__device__ __forceinline__ unsigned fp4_mult(unsigned s) {
  const unsigned m = ((s - 106u) << 2) & 0xffu;
  return (m << 8) | (m << 24);
}
__device__ __forceinline__ unsigned bias4(unsigned u) { return (u - 0x6a6a6a6au) << 2; }  // 4 bytes s -> (s - 106) << 2
__device__ __forceinline__ bool mb_test(uint32_t a, uint32_t par) {
  unsigned r;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(r) : "r"(a), "r"(par) : "memory");
  return r != 0;
}
__device__ __forceinline__ float ue8m0(unsigned s) { return __uint_as_float(s << 23); }  // 2^(s - 127), s in 1..254
__device__ __forceinline__ float pow2i(int p) { return __int_as_float((127 + p) << 23); }  // exact, |p| <= 126
__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float sigm(float z) { return 1.f / (1.f + expf(-z)); }
__device__ __forceinline__ int p_of(float m) {  // the engine's per-row prescale exponent (k1b_out / k_prescale)
  int p = 0;
  if (m > 0.f) {
    int ex;
    frexpf(m, &ex);
    p = 14 - ex;
  }
  return p;
}
__device__ __forceinline__ unsigned h2pack(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<unsigned*>(&h);
}
// 8 bf16 (natural order k0..k7) x sc -> f16 in P8 order [k0 k2 k4 k6 k1 k3 k5 k7]
__device__ __forceinline__ uint4 p8_f16(uint4 v, float sc) {
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
  float f[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) f[2 * i] = __uint_as_float(u[i] << 16), f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
  return make_uint4(h2pack(f[0] * sc, f[2] * sc), h2pack(f[4] * sc, f[6] * sc), h2pack(f[1] * sc, f[3] * sc),
                    h2pack(f[5] * sc, f[7] * sc));
}
__device__ __forceinline__ float bmax8(uint4 v) {
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) m = fmaxf(m, fmaxf(fabsf(__uint_as_float(u[i] << 16)), fabsf(__uint_as_float(u[i] & 0xffff0000u))));
  return m;
}
__device__ __forceinline__ int p8pos(int k) {  // position of physical k (within a 576 / D row) in the P8 image
  const int j = k & 7;
  return (k & ~7) + ((j & 1) << 2) + (j >> 1);
}

// ------------------------------------------------------------------------------------------------ work partition
// GEMM1 items (72 shared halves, then 36 per routed expert) are dealt round-robin starting at CTA 56: the 2-tile CTAs
// (56 .. 131, less GEMM2 work) take the first items of every round, so the per-CTA bytes of GEMM1 + GEMM2 even out
constexpr int NWK = 128;   // worker CTAs (router rows, GEMM1 items); CTAs 128 .. 131 compute the mHC mixes (one row each)
__device__ __forceinline__ int g1first(int c) { return c < NWK ? (c + NWK - 56) % NWK : 1 << 30; }
__device__ __forceinline__ void my_tiles(int c, int& ftb, int& nt) {
  if (c < 56) ftb = 3 * c, nt = 3;
  else ftb = 168 + 2 * (c - 56), nt = 2;
}

// one dense fp8 chunk: NT16 tiles of 16 rows, tile ti at st + ti * TS as [16 x 128 B] boxes (128-B swizzle), nkb k32
// blocks, this warp's blocks kb = w, w + NCW, ..; scale byte of (tile, kb) at scl + ti * sstr + kb; B image row n at
// img + n * IMS, k offset kofs (halves). acc[tile] += sum over blocks of 2^(s-127) * (A . B).
template <int NT16>
__device__ __forceinline__ void dense_chunk(float (&acc)[NT16][4], uint32_t st, int TS, int nkb, uint32_t scl, int sstr,
                                            uint32_t img, int kofs, int NR, int w, int lane) {
  const int g = lane >> 2, t = lane & 3;
  for (int kb = w; kb < nkb; kb += NCW) {
    uint4 bv = make_uint4(0u, 0u, 0u, 0u);
    if (g < NR) bv = lds128(img + g * IMS + (kofs + kb * 32) * 2 + 16 * t);
    const uint32_t off = (kb >> 2) * 2048 + g * 128 + (((((kb & 3) << 1) | (t >> 1)) ^ g) << 4) + ((t & 1) << 3);
#pragma unroll
    for (int ti = 0; ti < NT16; ++ti) {
      const uint32_t ra = st + ti * TS + off;
      const uint2 wg = lds64(ra), wh = lds64(ra + 1024);
      const unsigned pg = __byte_perm(wg.x, wg.y, 0x6420), qg = __byte_perm(wg.x, wg.y, 0x7531);
      const unsigned ph = __byte_perm(wh.x, wh.y, 0x6420), qh = __byte_perm(wh.x, wh.y, 0x7531);
      unsigned g02, g46, h02, h46, g13, g57, h13, h57;
      cvt_e4m3x4(pg, g02, g46);
      cvt_e4m3x4(ph, h02, h46);
      cvt_e4m3x4(qg, g13, g57);
      cvt_e4m3x4(qh, h13, h57);
      float tmp[4] = {0.f, 0.f, 0.f, 0.f};
      mma16816(tmp, g02, h02, g46, h46, bv.x, bv.y);
      mma16816(tmp, g13, h13, g57, h57, bv.z, bv.w);
      unsigned sb;
      asm volatile("ld.shared.u8 %0, [%1];" : "=r"(sb) : "r"(scl + ti * sstr + kb));
      const float sc = ue8m0(sb);
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[ti][i] = fmaf(tmp[i], sc, acc[ti][i]);
    }
  }
}

// f16 x 2 -> bf16 x 2 (exact for the activation images: bf16 values times a power of 2)
__device__ __forceinline__ unsigned h2_bf2(unsigned h) {
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
  return *reinterpret_cast<const unsigned*>(&b);
}
// k32 block kb (of a 128B-swizzled bf16 box [k-blocks of 64][16 rows][128 B] at t0) as P8-ordered A operands of two
// m16n8k16 MMAs (the pairs (0,2) (4,6) then (1,3) (5,7) of each lane's 8 physical k: the order of the P8 images), and
// the MMAs with B (b0, b1) for the first, (b2, b3) for the second
__device__ __forceinline__ void p8_bf16_mma(float (&acc)[4], uint32_t t0, int kb, int g, int t, unsigned b0, unsigned b1,
                                            unsigned b2, unsigned b3) {
  const uint32_t ra = t0 + (kb >> 1) * 2048 + g * 128 + (((((kb & 1) << 2) | t) ^ g) << 4);
  const uint4 wg = lds128(ra), wh = lds128(ra + 1024);
  hmma_bf16(acc, __byte_perm(wg.x, wg.y, 0x5410), __byte_perm(wh.x, wh.y, 0x5410), __byte_perm(wg.z, wg.w, 0x5410),
            __byte_perm(wh.z, wh.w, 0x5410), b0, b1);
  hmma_bf16(acc, __byte_perm(wg.x, wg.y, 0x7632), __byte_perm(wh.x, wh.y, 0x7632), __byte_perm(wg.z, wg.w, 0x7632),
            __byte_perm(wh.z, wh.w, 0x7632), b2, b3);
}
// shared expert GEMM1 from the engine's bf16 w1 | w3: one 256-K chunk ([2 halves][4 k-blocks of 64][16 rows][128 B] at st),
// warp w: its k32 block w; B = the f16 P8 image rows converted to bf16
__device__ __forceinline__ void sh16_chunk(float (&acc)[2][4], uint32_t st, uint32_t img, int kofs, int NR, int w, int lane) {
  const int g = lane >> 2, t = lane & 3;
  uint4 bv = make_uint4(0u, 0u, 0u, 0u);
  if (g < NR) {
    bv = lds128(img + g * IMS + (kofs + 32 * w) * 2 + 16 * t);
    bv = make_uint4(h2_bf2(bv.x), h2_bf2(bv.y), h2_bf2(bv.z), h2_bf2(bv.w));
  }
#pragma unroll
  for (int ti = 0; ti < 2; ++ti) p8_bf16_mma(acc[ti], st + ti * 8192, w, g, t, bv.x, bv.y, bv.z, bv.w);
}
// shared expert GEMM2 from the engine's bf16 w2: k32 blocks j0 .. j1 - 1 of one stage ([5 k-blocks of 64][16 rows][128 B]
// at st, then hsh [NR][576] natural bf16 at st + L_S2B_H); k32 block j of the stage is the tile's k32 block kb0 + j
__device__ __forceinline__ void sh2_16(float (&acc)[4], uint32_t st, int kb0, int nkb, int NR, int lane) {
  const int g = lane >> 2, t = lane & 3;
  for (int j = 0; j < nkb; ++j) {
    uint4 hv = make_uint4(0u, 0u, 0u, 0u);
    if (g < NR) hv = lds128(st + L_S2B_H + g * IR * 2 + (32 * (kb0 + j) + 8 * t) * 2);
    p8_bf16_mma(acc, st, j, g, t, __byte_perm(hv.x, hv.y, 0x5410), __byte_perm(hv.z, hv.w, 0x5410),
                __byte_perm(hv.x, hv.y, 0x7632), __byte_perm(hv.z, hv.w, 0x7632));
  }
}

__device__ __forceinline__ void ldm_x4(uint32_t a, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm_x4t(uint32_t a, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
// bf16 weight tiles of 16 rows x 512 K from 128B-swizzled boxes ([8 kblocks of 64][16 rows][128 B]; tile ti at st + ti * TS),
// k16 steps kk = w, w + NCW, ..; B = natural bf16 image rows (row n at img + n * IMS, k offset kofs elements)
template <int NT16>
__device__ __forceinline__ void dense16_chunk(float (&acc)[NT16][4], uint32_t st, int TS, int nk16, uint32_t img, int kofs, int NR,
                                              int w, int lane) {
  const int g = lane >> 2, t = lane & 3, lr = (lane & 7) + 8 * ((lane >> 3) & 1), lc = lane >> 4;
  for (int kk = w; kk < nk16; kk += NCW) {
    unsigned b0 = 0u, b1 = 0u;
    if (g < NR) {
      const uint32_t ba = img + g * IMS + (kofs + 16 * kk) * 2 + 4 * t;
      b0 = lds32(ba), b1 = lds32(ba + 16);
    }
    const uint32_t off = (kk >> 2) * 2048 + lr * 128 + (((((kk & 3) << 1) | lc) ^ (lr & 7)) << 4);
#pragma unroll
    for (int ti = 0; ti < NT16; ++ti) {
      unsigned a0, a1, a2, a3;
      ldm_x4(st + ti * TS + off, a0, a1, a2, a3);
      hmma_bf16(acc[ti], a0, a1, a2, a3, b0, b1);
    }
  }
}
// the same from K-major tiles ([k][16 n] bf16, 32-B rows, no swizzle: wo_a^T boxes), A fragments by ldmatrix.trans
__device__ __forceinline__ void dense16t_chunk(float (&acc)[4], uint32_t st, int nk16, uint32_t img, int kofs, int NR, int w,
                                               int lane) {
  const int g = lane >> 2, t = lane & 3, m = lane >> 3, j = lane & 7;
  for (int kk = w; kk < nk16; kk += NCW) {
    unsigned b0 = 0u, b1 = 0u;
    if (g < NR) {
      const uint32_t ba = img + g * IMS + (kofs + 16 * kk) * 2 + 4 * t;
      b0 = lds32(ba), b1 = lds32(ba + 16);
    }
    unsigned a0, a1, a2, a3;
    ldm_x4t(st + (16 * kk + j + 8 * (m >> 1)) * 32 + 16 * (m & 1), a0, a1, a2, a3);
    hmma_bf16(acc, a0, a1, a2, a3, b0, b1);
  }
}

// cross-warp fixed-order reduction of NT16 tiles' accumulators; result valid in warp 0 (all lanes)
template <int NT16>
__device__ __forceinline__ void reduce_warps(float (&acc)[NT16][4], float* red, int w, int lane) {
#pragma unroll
  for (int ti = 0; ti < NT16; ++ti)
#pragma unroll
    for (int i = 0; i < 4; ++i) red[((w * NT16 + ti) * 32 + lane) * 4 + i] = acc[ti][i];
  cbar();
  if (w == 0) {
#pragma unroll
    for (int ti = 0; ti < NT16; ++ti)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float a = red[((0 * NT16 + ti) * 32 + lane) * 4 + i];
        for (int ww = 1; ww < NCW; ++ww) a += red[((ww * NT16 + ti) * 32 + lane) * 4 + i];
        acc[ti][i] = a;
      }
  }
}

// KV record (row n, key j: window slot j < WIN, else top-k entry j - WIN < TKA) in this layer's pages: base + data / scale offsets
template <class A>
__device__ __forceinline__ bool kv_rec(const A& a, int n, int j, const uint8_t*& b, long long& d, long long& sc) {
  if (j < WIN) {
    const int idx = j < __ldg(a.sl + n) ? __ldg(a.si + n * a.si_rs + j) : -1;
    if (idx < 0) return false;
    const long long pg = idx >> a.win_shift, ps = idx & ((1 << a.win_shift) - 1);
    d = pg * a.win_stride + ps * 576, sc = pg * a.win_stride + (576LL << a.win_shift) + ps * 8, b = a.win;
    return true;
  }
  if (!a.comp || j >= KVR) return false;
  const int jj = j - WIN;
  const int idx = jj < __ldcg(a.tl + n) ? __ldcg(a.ti + n * a.ti_rs + jj) : -1;   // (.cg: in-kernel index layers rewrite ti / tl)
  if (idx < 0) return false;
  const long long pg = idx >> a.comp_shift, ps = idx & ((1 << a.comp_shift) - 1);
  d = pg * a.comp_stride + ps * 576, sc = pg * a.comp_stride + (576LL << a.comp_shift) + ps * 8, b = a.comp;
  return true;
}

struct Smem {
  alignas(16) unsigned long long full[NSMAX], empty[NSMAX], route;
  alignas(16) float red[NCW * 2 * 32 * 4];
  alignas(16) float yp[NRM][48];
  alignas(16) bf16 ybf[NRM][48];
  alignas(16) bf16 Ybf[NRM][TOPK][48];
  alignas(16) bf16 shbf[NRM][48];
  alignas(16) bf16 part[NRM][48];
  alignas(16) float xs[NRM][HC][48];
  alignas(16) float wmax[NCW][NRM];
  alignas(16) float rr[NRM];
  alignas(16) int pv[NRM];
  alignas(16) float post[NRM][HC], comb[NRM][16], lag[NRM][HC];
  alignas(16) int ids[NRM][TOPK];
  alignas(16) float wts[NRM][TOPK];
  int nexp, ticket;
  alignas(16) int xid[MAXX];
  alignas(16) float xw[MAXX][NRM];
  alignas(16) int xk[MAXX][NRM];
  alignas(16) bf16 qt[NRM][64];      // q_b staging (4 tiles)
  unsigned bm[NEW], bpre[NEW];       // chosen-expert bitmap (NE bits) and its word prefix counts
  alignas(16) Args args;             // consumers: this layer's arguments
  alignas(16) bf16 ot[16][32];       // attention combine staging (16 heads x 32 dims)
  int m3j[NRM];                      // mode 3: the window index of the new record in each row (-1: none)
};
static_assert(offsetof(Smem, yp) % 16 == 0 && offsetof(Smem, part) % 16 == 0 && offsetof(Smem, red) % 16 == 0, "align");

// ------------------------------------------------------------------------------------------------ the kernel
template <bool FULL>
struct LayCfg {
  static constexpr int NS = FULL ? 7 : NSMAX;  // ring stages (FULL: the attention stage takes the SMEM of one)
  static constexpr int ATTX = FULL ? 22528 : 0;  // extra SMEM after the image region (attention scratch)
};
// file-scope definitions of the phase fragments (headers with the device functions / constants they call), after Args,
// Smem and the helpers: C1M_HOOKS section 0 (index / Engram phases) and the optional C1M_MOE_DEFS (expert phase)
#define C1M_HOOK 0
#include C1M_HOOKS
#undef C1M_HOOK
#ifdef C1M_MOE_DEFS
#include C1M_MOE_DEFS
#endif

// ============================================================================================== consumer layer body
// One layer of the consumer warps. Regular layers: cons_layer<FULL, false> force-inlined into
// k_layer's loop (the code of before). Builds with layer phases (the hooks file defines C1M_HAS_PHASES): index /
// compressor / Engram layers run cons_layer<FULL, true> (hooks 2-4 compiled in) inside the __noinline__ cons_ix, a
// separate function: its code and register allocation stay out of the regular layers' path (one never-taken call
// at the loop top; the loop state goes in and out by value, no address escapes).
struct CState {
  long it;                          // ring stages consumed so far
  unsigned long long st_w, st_b;    // thread 0, timed calls: ns waiting for ring stages / for counters + remote data
  unsigned call;                    // this CTA's layer epoch at the launch start
  int ltim;                         // the stamped layer
  unsigned long long* tim0;         // its timeline (this CTA's row) or null
};
template <bool FULL, bool HX>
__device__ __forceinline__ void cons_layer(const Args* __restrict__ al, const int nly, const int l, CState& cs) {
  constexpr int NS = LayCfg<FULL>::NS, ATTX = LayCfg<FULL>::ATTX;
  extern __shared__ __align__(1024) char smem[];
  char* ring = smem;
  char* imgp = smem + NS * STAGE;
  Smem& S = *reinterpret_cast<Smem*>(smem + NS * STAGE + NRM * IMS + ATTX);
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, c = blockIdx.x;
  const int NR = al[0].NR, rank = al[0].rank;
  int ftb, nt;
  my_tiles(c, ftb, nt);
  const int col0 = 16 * ftb, ncols = 16 * nt;
  const uint32_t ring_u = su32(ring), img = su32(imgp);
  const unsigned call = cs.call;
  const int ltim = cs.ltim;
  unsigned long long* const tim0 = cs.tim0;
  long& it = cs.it;
  unsigned long long& st_w = cs.st_w;
  unsigned long long& st_b = cs.st_b;
  (void)rank, (void)ring_u, (void)img, (void)ftb, (void)nly, (void)ltim, (void)tim0, (void)call;

  const int g = lane >> 2, t4 = lane & 3;
  const Args& a = S.args;            // this layer's arguments (copied from al[l] at every layer start)
  unsigned par = 0, ncall = 0;       // this layer's lamport parity and counter epoch (+1)
  unsigned long long* tim = nullptr;
  auto stamp = [&](int k) {
    if (tim && tid == 0) tim[k] = gtime(), tim[NTS + k] = st_w, tim[2 * NTS + k] = st_b, tim[3 * NTS + k] = clock64();
  };
  auto cwait = [&]() -> uint32_t {
    const int s = (int)(it % NS);
    const unsigned pr = (unsigned)((it / NS) & 1);
    if (tim && tid == 0 && !mb_test(su32(&S.full[s]), pr)) {
      const unsigned long long t0 = gtime();
      mb_wait(su32(&S.full[s]), pr);
      st_w += gtime() - t0;
    } else {
      mb_wait(su32(&S.full[s]), pr);
    }
    return ring_u + s * STAGE;
  };
  auto crel = [&]() {
    __syncwarp();
    if (lane == 0) mb_arrive(su32(&S.empty[(int)(it % NS)]));
    ++it;
  };
  auto arrive = [&](int k) {
    cbar();
    if (tid == 0) {
      __threadfence();
      red_rel(a.cnt + k, 1u);
    }
  };
  auto waitc = [&](int k, unsigned target) {
    if (tid == 0) {
      const unsigned long long t0 = tim ? gtime() : 0ull;
      spin_ge(a.cnt + k, target);
      if (tim) st_b += gtime() - t0;
    }
    cbar();
  };
  // block max over consumer threads of per-row values mx[NRM] -> S.pv[n] = p_of(max)
  auto rowp = [&](float (&mx)[NRM]) {
#pragma unroll
    for (int n = 0; n < NRM; ++n) {
      float m = mx[n];
#pragma unroll
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
      if (lane == 0) S.wmax[w][n] = m;
    }
    cbar();
    if (tid < NRM) {
      float m = 0.f;
      for (int ww = 0; ww < NCW; ++ww) m = fmaxf(m, S.wmax[ww][tid]);
      S.pv[tid] = p_of(m);
    }
    cbar();
  };
  // mHC boundary slice of this CTA's columns (k_frow's arithmetic): xo = bf16(post y + comb^T rs), in = bf16(sum lag
  // xo), the sum of squares of in (RMSNorm partial) and the 24 mix dot products + the sum of squares of xo (fn chunk)
  auto bslice = [&](auto yget, const bf16* rsx, bf16* xo, bf16* ino, float* sspo, float* mixo, int ck, int sb) {
    float ss[NRM] = {0.f, 0.f, 0.f, 0.f};
    for (int i = tid; i < NR * ncols; i += CTH) {
      const int n = i / ncols, j = i % ncols, col = col0 + j;
      const float yv = yget(n, j, col);
      float rv[HC];
#pragma unroll
      for (int cc = 0; cc < HC; ++cc) rv[cc] = __bfloat162float(rsx[((size_t)n * HC + cc) * D + col]);
      float x[HC];
#pragma unroll
      for (int jj = 0; jj < HC; ++jj) {
        float v = S.post[n][jj] * yv;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) v = fmaf(S.comb[n][cc * HC + jj], rv[cc], v);
        const bf16 hb = __float2bfloat16_rn(v);
        x[jj] = __bfloat162float(hb);
        S.xs[n][jj][j] = x[jj];
        xo[((size_t)n * HC + jj) * D + col] = hb;
      }
      float ai = 0.f;
#pragma unroll
      for (int cc = 0; cc < HC; ++cc) ai = fmaf(S.lag[n][cc], x[cc], ai);
      const float v = bfr(ai);
      ino[(size_t)n * D + col] = __float2bfloat16_rn(v);
#pragma unroll
      for (int q = 0; q < NRM; ++q)   // (static indices: ss stays in registers)
        if (q == n) ss[q] = fmaf(v, v, ss[q]);
    }
    stamp(sb);
#pragma unroll
    for (int n = 0; n < NRM; ++n) {
      float m = ss[n];
#pragma unroll
      for (int o = 16; o; o >>= 1) m += __shfl_xor_sync(~0u, m, o);
      if (lane == 0) S.wmax[w][n] = m;
    }
    cbar();
    if (tid < NR) {
      float m = 0.f;
      for (int ww = 0; ww < NCW; ++ww) m += S.wmax[ww][tid];
      sspo[c * NRM + tid] = m;
    }
    stamp(sb + 1);
    // the 24 mix dot products + the sum of squares of xm2 over this slice (fn chunk from the ring)
    const uint32_t st = cwait();
    stamp(sb + 2);
    const float* fs = (const float*)(ring + (st - ring_u));
    {   // warp w: mixes m = w, w + 8, w + 16 (+ m = 24 = the sum of squares of xo, warp 0), all rows at once (independent
        // chains, interleaved butterflies); per (m, row) the same order of terms and of the butterfly as one chain
      float acc[4][NRM];
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int n = 0; n < NRM; ++n) acc[q][n] = 0.f;
#pragma unroll
      for (int cc = 0; cc < HC; ++cc)
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int j = lane + 32 * jj;
          if (j < ncols) {
            float xv[NRM];
#pragma unroll
            for (int n = 0; n < NRM; ++n) xv[n] = S.xs[n][cc][j];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const int m = w + NCW * q;
              if (m < 25) {
                const float fv = m == 24 ? 0.f : fs[(m * HC + cc) * 48 + j];
#pragma unroll
                for (int n = 0; n < NRM; ++n) acc[q][n] = fmaf(xv[n], m == 24 ? xv[n] : fv, acc[q][n]);
              }
            }
          }
        }
#pragma unroll
      for (int o = 16; o; o >>= 1)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int n = 0; n < NRM; ++n) acc[q][n] += __shfl_xor_sync(~0u, acc[q][n], o);
      if (lane == 0)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int m = w + NCW * q;
          if (m < 25)
#pragma unroll
            for (int n = 0; n < NRM; ++n)
              if (n < NR) mixo[(c * NRM + n) * 25 + (m == 24 ? 0 : 1 + m)] = acc[q][n];
        }
    }
    stamp(sb + 3);
    crel();
    arrive(ck);
  };
  // RMSNorm of the boundary rows from the slices' partial sums (every CTA, all columns): natural bf16 rows in the image
  // region (+ fo), S.pv = their prescale exponents
  auto rnorm = [&](const float* sspi, const bf16* ini, const float* nwi, bf16* fo) {
    if (w < NR) {   // warp n = row n: lane l sums CTAs l, l + 32, .. (5 loads in flight), then a butterfly
      float v[5];
#pragma unroll
      for (int k = 0; k < 5; ++k) v[k] = lane + 32 * k < NCTA ? __ldcg(sspi + (lane + 32 * k) * NRM + w) : 0.f;
      float s = (((v[0] + v[1]) + v[2]) + v[3]) + v[4];
#pragma unroll
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
      if (lane == 0) S.rr[w] = rsqrtf(s / (float)D + a.norm_eps);
    }
    cbar();
    stamp(38);
    float mx[NRM] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 3; ++j) {   // 640 chunks of 8 columns: thread tid takes tid, tid + 256, tid + 512 (all loads first)
      const int c8 = tid + CTH * j;
      if (c8 >= 640) break;
      const float4 w0 = __ldg((const float4*)(nwi + 8 * c8)), w1 = __ldg((const float4*)(nwi + 8 * c8 + 4));
      uint4 v[NRM];
#pragma unroll
      for (int n = 0; n < NRM; ++n) v[n] = n < NR ? __ldcg((const uint4*)(ini + (size_t)n * D) + c8) : make_uint4(0u, 0u, 0u, 0u);
      const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int n = 0; n < NRM; ++n) {
        if (n < NR) {   // (no break: a fully unrolled loop keeps mx in registers)
          const unsigned u[4] = {v[n].x, v[n].y, v[n].z, v[n].w};
          unsigned o[4];
          const float rr = S.rr[n];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float f0 = __uint_as_float(u[q] << 16), f1 = __uint_as_float(u[q] & 0xffff0000u);
            const bf16 b0 = __float2bfloat16_rn((f0 * rr) * wv[2 * q]), b1 = __float2bfloat16_rn((f1 * rr) * wv[2 * q + 1]);
            mx[n] = fmaxf(mx[n], fmaxf(fabsf(__bfloat162float(b0)), fabsf(__bfloat162float(b1))));
            o[q] = (unsigned)__bfloat16_as_ushort(b0) | ((unsigned)__bfloat16_as_ushort(b1) << 16);
          }
          const uint4 ov = make_uint4(o[0], o[1], o[2], o[3]);
          *(uint4*)(imgp + n * IMS + 16 * c8) = ov;  // natural bf16 (router), converted to f16 P8 after the router
          if (fo) *(uint4*)(fo + (size_t)n * D + 8 * c8) = ov;
        }
      }
    }
    rowp(mx);
  };
  // the boundary's mixes of row n (k_fbm8's statements), on service CTA 128 + n: the 25 sums over the 132 slices by all
  // threads (value q, group k of CTAs k, k + 10, ..; the 10 group sums added in order), the Sinkhorn on warp 0
  auto mixes = [&](int n, const float* mixi, const float* sc, const float* bs, float* preo, float* posto, float* combo) {
    if (tid < 250) {
      const int q = tid / 10, k = tid % 10;
      float v[14];
#pragma unroll
      for (int i = 0; i < 14; ++i) v[i] = k + 10 * i < NCTA ? __ldcg(mixi + ((k + 10 * i) * NRM + n) * 25 + q) : 0.f;
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < 14; ++i) acc += v[i];
      S.red[tid] = acc;
    }
    cbar();
    if (w != 0) return;
    const int tt = lane;
    float tv = 0.f;
    if (tt < 25) {
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < 10; ++k) acc += S.red[tt * 10 + k];
      tv = acc;
    }
    const float rsq = rsqrtf(__shfl_sync(~0u, tv, 0) / (float)(HC * D) + a.norm_eps);
    const int l = tt & 15, i = l >> 2, jj = l & 3;
    const float t_pre = __shfl_sync(~0u, tv, 1 + (tt & 3)), t_post = __shfl_sync(~0u, tv, 1 + HC + (tt & 3));
    const float t_c = __shfl_sync(~0u, tv, 1 + 2 * HC + l);
    if (tt < HC) {
      preo[n * HC + tt] = sigm(t_pre * rsq * sc[0] + bs[tt]) + a.hc_eps;
      posto[n * HC + tt] = 2.f * sigm(t_post * rsq * sc[1] + bs[HC + tt]);
    }
    const float lg = t_c * rsq * sc[2] + bs[2 * HC + l];
    float mxl = fmaxf(lg, __shfl_xor_sync(~0u, lg, 1));
    mxl = fmaxf(mxl, __shfl_xor_sync(~0u, mxl, 2));
    const float ex = expf(lg - mxl);
    auto rowsum = [&](float v) {
      const int b = tt & ~3;
      return ((__shfl_sync(~0u, v, b) + __shfl_sync(~0u, v, b + 1)) + __shfl_sync(~0u, v, b + 2)) + __shfl_sync(~0u, v, b + 3);
    };
    auto colsum = [&](float v) {
      const int b = (tt & 16) | (tt & 3);
      return ((__shfl_sync(~0u, v, b) + __shfl_sync(~0u, v, b + 4)) + __shfl_sync(~0u, v, b + 8)) + __shfl_sync(~0u, v, b + 12);
    };
    float cv = ex / rowsum(ex) + a.hc_eps;
    cv = cv / (colsum(cv) + a.hc_eps);
    for (int k = 1; k < a.hc_iters; ++k) {
      cv = cv / (rowsum(cv) + a.hc_eps);
      cv = cv / (colsum(cv) + a.hc_eps);
    }
    if (tt < 16) combo[n * 16 + i * HC + jj] = cv;
  };
  for (int i = tid; i < (int)(sizeof(Args) / 4); i += CTH) ((unsigned*)&S.args)[i] = ((const unsigned*)(al + l))[i];
  cbar();
  par = (call + l) & 1u, ncall = call + l + 1u;
  tim = (l == ltim) ? tim0 : nullptr;
  if (l > 0) stamp(0);   // (layer 0: stamp 0 = the kernel start)
  if (!FULL && tid < NR * 24) {  // mixes of the attention boundary (inputs of this boundary)
    const int n = tid / 24, k = tid % 24;
    if (k < 4) S.post[n][k] = a.apost[n * 4 + k];
    else if (k < 20) S.comb[n][k - 4] = a.acomb[n * 16 + k - 4];
    else S.lag[n][k - 20] = a.apre[n * 4 + k - 20];
  }

  if constexpr (FULL) {
    // ============================================================================================ front half
    // the previous boundary's mixes (post, comb, pre = lag of this boundary); in a multi-layer launch the previous
    // layer's FFN mixes come from its CTA 131
    if (l > 0) waitc(C_MIXF, (unsigned)NR * (ncall - 1u));
    if (l == 0 && a.rdy[0] != nullptr) {   // engine runs: the inputs are the owner's buffers (see Args::rdy)
      if (rank == a.owner) {
        if (tid < W && tid != rank) st_rel_sys(a.rdy[tid] + c, call + 1u);
      } else if (tid == 0) {
        const unsigned long long t0 = tim ? gtime() : 0ull;
        while ((int)(ld_acq_sys(a.rdy[rank] + c) - (call + 1u)) < 0) {
        }
        if (tim) st_b += gtime() - t0;
      }
      cbar();
    }
    if (tid < NR * 24) {
      const int n = tid / 24, k = tid % 24;
      if (k < 4) S.post[n][k] = __ldcg(a.pin + n * 4 + k);
      else if (k < 20) S.comb[n][k - 4] = __ldcg(a.cin + n * 16 + k - 4);
      else S.lag[n][k - 20] = __ldcg(a.lin + n * 4 + k - 20);
    }
    cbar();
    // KV records of this CTA's attention units -> L2 now (read ~40 us later; the weights stream with evict_first)
    {
      const bool attend = AMODE(a) == 0 || rank == a.owner;   // (modes 2 / 3: the peers' pages are not local)
      const int NHG = AMODE(a) == 1 ? 4 : 1;
      for (int u = c; attend && u < NR * NHG * ANCH; u += NCTA) {
        const int n = u / (NHG * ANCH), ch = u % ANCH;
        if (tid < AK * 5) {   // 32 keys x 5 lines (576 B data) + the scale row
          const int k = tid / 5, ln = tid % 5, j = ch * AK + k;
          int idx = -1;
          const uint8_t* b = nullptr;
          int sh = 0;
          long long stride = 0;
          if (j < WIN) {
            if (j < __ldg(a.sl + n)) idx = __ldg(a.si + n * a.si_rs + j), b = a.win, sh = a.win_shift, stride = a.win_stride;
          } else if (a.comp && j < KVR) {
            if (j - WIN < __ldcg(a.tl + n)) idx = __ldcg(a.ti + n * a.ti_rs + j - WIN), b = a.comp, sh = a.comp_shift, stride = a.comp_stride;
          }
          if (idx >= 0) {
            const long long pg = idx >> sh, ps = idx & ((1 << sh) - 1);
            const uint8_t* p = ln < 4 ? b + pg * stride + ps * 576 + 128 * ln + (ln == 4 ? 0 : 0) : b + pg * stride + (576LL << sh) + ps * 8;
            asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p));
            if (ln == 3) asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p + 128));   // bytes 512 .. 575
          }
        }
      }
    }
if constexpr (HX) {
#define C1M_HOOK 2
#include C1M_HOOKS
#undef C1M_HOOK
}
    // ---- A1 attention mHC boundary slice -> X (streams), in, partial sums
    bslice([&](int n, int j, int col) { return __bfloat162float(a.xin[(size_t)n * D + col]); }, a.rsin, a.Xo, a.ain, a.ssp_a,
           a.mixp_a, C_SSA, 27);
    stamp(14);
    // (attention mode 3: the owner's KV push runs on the producer warp's lanes 1..31)
    // ---- A2 a = RMSNorm(in) x attn_norm -> f16 P8 image; the attention mixes (CTA 131)
    waitc(C_SSA, 132u * ncall);
#ifdef C1M_NO_IX
    rnorm(a.ssp_a, a.ain, a.anw, nullptr);   // (deployment builds without in-kernel index layers: no aout code)
#else
    rnorm(a.ssp_a, a.ain, a.anw, a.aout);
#endif
    for (int i = tid; i < NR * 640 && !B16(a, wqkv16, 0); i += CTH) {   // (bf16 q_a | wkv: the natural bf16 image as it is)
      const int n = i / 640, c8 = i % 640;
      uint4* q = (uint4*)(imgp + n * IMS + 16 * c8);
      *q = p8_f16(*q, pow2i(S.pv[n]));
    }
    cbar();
    stamp(15);
    // ---- A3 q_a | wkv (replicated): tile c = output columns 16 c .. 16 c + 15 of [1280 q_a | 512 kv]
    if (c < 112) {
      float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
      const bool b16 = B16(a, wqkv16, 0);
      for (int kc = 0; kc < (b16 ? 10 : 5); ++kc) {
        const uint32_t st = cwait();
        if (b16) dense16_chunk<1>(acc, st, 0, 32, img, 512 * kc, NR, w, lane);
        else dense_chunk<1>(acc, st, 0, 32, st + L_D_S, 32, img, 1024 * kc, NR, w, lane);
        crel();
      }
      reduce_warps<1>(acc, S.red, w, lane);
      if (w == 0)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
          if (n < NR) a.qakv[(size_t)n * 1792 + 16 * c + row] = __float2bfloat16_rn(b16 ? acc[0][i] : acc[0][i] * pow2i(-S.pv[n]));
        }
      arrive(C_QA);
    }
    stamp(16);
    // ---- A4 q norm (every CTA: qr -> f16 P8 image), kv norm + RoPE + window record (CTA 0 when do_insert)
    waitc(C_QA, 112u * ncall);
    stamp(39);
    if (c >= NWK) arrive(C_QB);   // the service CTAs have no q_b tiles: their C_QB arrival first, the insert after it
                                  // (the attention waits for the inserts through C_INS / the mode-3 push counter)
    {
      // q_a rows: 160 chunks of 8 per row; thread tid: chunk tid (tid < 160) of every row (loads first)
      uint4 qv[NRM];
      float ss[NRM] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int n = 0; n < NRM; ++n)
        qv[n] = (n < NR && tid < 160) ? __ldcg((const uint4*)(a.qakv + (size_t)n * 1792) + tid) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int n = 0; n < NRM; ++n) {
        const unsigned u[4] = {qv[n].x, qv[n].y, qv[n].z, qv[n].w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float f0 = __uint_as_float(u[q] << 16), f1 = __uint_as_float(u[q] & 0xffff0000u);
          ss[n] = __fadd_rn(ss[n], __fmul_rn(f0, f0));
          ss[n] = __fadd_rn(ss[n], __fmul_rn(f1, f1));
        }
      }
#pragma unroll
      for (int n = 0; n < NRM; ++n) {
        float m = ss[n];
#pragma unroll
        for (int o = 16; o; o >>= 1) m += __shfl_xor_sync(~0u, m, o);
        if (lane == 0) S.wmax[w][n] = m;
      }
      cbar();
      if (tid < NR) {
        float m = 0.f;
        for (int ww = 0; ww < NCW; ++ww) m += S.wmax[ww][tid];
        S.rr[tid] = rsqrtf(__fadd_rn(__fdiv_rn(m, 1280.f), a.attn_eps));
      }
      cbar();
      float mx[NRM] = {0.f, 0.f, 0.f, 0.f};
      if (tid < 160) {
        const float4 w0 = __ldg((const float4*)(a.q_norm + 8 * tid)), w1 = __ldg((const float4*)(a.q_norm + 8 * tid + 4));
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int n = 0; n < NRM; ++n) {
          if (n >= NR) break;
          const unsigned u[4] = {qv[n].x, qv[n].y, qv[n].z, qv[n].w};
          unsigned o[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float f0 = __uint_as_float(u[q] << 16), f1 = __uint_as_float(u[q] & 0xffff0000u);
            const bf16 b0 = __float2bfloat16_rn(__fmul_rn(__fmul_rn(f0, S.rr[n]), wv[2 * q]));
            const bf16 b1 = __float2bfloat16_rn(__fmul_rn(__fmul_rn(f1, S.rr[n]), wv[2 * q + 1]));
            mx[n] = fmaxf(mx[n], fmaxf(fabsf(__bfloat162float(b0)), fabsf(__bfloat162float(b1))));
            o[q] = (unsigned)__bfloat16_as_ushort(b0) | ((unsigned)__bfloat16_as_ushort(b1) << 16);
          }
          const uint4 ov = make_uint4(o[0], o[1], o[2], o[3]);
          if (c == 0) *(uint4*)(a.qrout + (size_t)n * 1280 + 8 * tid) = ov;
          qv[n] = ov;
        }
      }
      rowp(mx);
      if (tid < 160)
#pragma unroll
        for (int n = 0; n < NRM; ++n)
          if (n < NR) *(uint4*)(imgp + n * IMS + 16 * tid) = p8_f16(qv[n], pow2i(S.pv[n]));
      if (a.do_insert && c >= NWK && c - NWK < NR) {   // service CTA 128 + n: row n's kv norm + RoPE + window record
        float* kvs = &S.xs[0][0][0];  // 512 floats of scratch
        for (int n = c - NWK; n == c - NWK; ++n) {
          const long long slot = __ldg(a.own + n);
          if (slot < 0) continue;  // uniform
          const float x0 = __bfloat162float(__ldcg(a.qakv + (size_t)n * 1792 + 1280 + tid));
          const float x1 = __bfloat162float(__ldcg(a.qakv + (size_t)n * 1792 + 1280 + 256 + tid));
          float m = __fadd_rn(__fmul_rn(x0, x0), __fmul_rn(x1, x1));
#pragma unroll
          for (int o = 16; o; o >>= 1) m += __shfl_xor_sync(~0u, m, o);
          if (lane == 0) S.wmax[w][0] = m;
          cbar();
          float tot = 0.f;
          for (int ww = 0; ww < NCW; ++ww) tot += S.wmax[ww][0];
          const float rs = rsqrtf(__fadd_rn(__fdiv_rn(tot, 512.f), a.attn_eps));
          kvs[tid] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(x0, rs), __ldg(a.kv_norm + tid))));
          kvs[256 + tid] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(x1, rs), __ldg(a.kv_norm + 256 + tid))));
          cbar();
          if (tid < 32) {  // RoPE of dims 448..511 (norm_rope_row: on the bf16 normed values)
            const float* cs = a.cos_sin + __ldg(a.pos + n) * 64;
            const int d0 = 448 + 2 * tid;
            const float e = kvs[d0], od = kvs[d0 + 1], cc = __ldg(cs + tid), sn = __ldg(cs + 32 + tid);
            kvs[d0] = __bfloat162float(__float2bfloat16_rn(__fsub_rn(__fmul_rn(e, cc), __fmul_rn(od, sn))));
            kvs[d0 + 1] = __bfloat162float(__float2bfloat16_rn(__fadd_rn(__fmul_rn(e, sn), __fmul_rn(od, cc))));
          }
          cbar();
          if (a.kvout) a.kvout[(size_t)n * 512 + tid] = __float2bfloat16_rn(kvs[tid]),
                       a.kvout[(size_t)n * 512 + 256 + tid] = __float2bfloat16_rn(kvs[256 + tid]);
          // the stock sm90 fp8_ds_mla record (dense.cu rope_q_insert_kernel's statements)
          const long long pg = slot >> a.win_shift, ps = slot & ((1LL << a.win_shift) - 1);
          uint8_t* base = a.win + pg * a.win_stride;
          uint8_t* data = base + ps * 576;
          uint8_t* scl = base + (576LL << a.win_shift) + ps * 8;
          if (w < 7) {
            const float y0 = kvs[64 * w + 2 * lane], y1 = kvs[64 * w + 2 * lane + 1];
            float am = fmaxf(fabsf(y0), fabsf(y1));
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
            am = fmaxf(am, 1e-4f);
            const float v = __fmul_rn(am, 1.0f / 448.0f);
            const unsigned bits = __float_as_uint(v);
            const int ex = (int)(bits >> 23) - ((bits & 0x7fffffu) == 0 ? 127 : 126);
            const float inv = __uint_as_float((unsigned)(127 - ex) << 23);
            const __nv_fp8x2_storage_t qv = __nv_cvt_float2_to_fp8x2(make_float2(y0 * inv, y1 * inv), __NV_SATFINITE, __NV_E4M3);
            *reinterpret_cast<__nv_fp8x2_storage_t*>(data + 64 * w + 2 * lane) = qv;
            if (lane == 0) scl[w] = (uint8_t)(ex + 127);
          } else if (w == 7) {
            const unsigned u = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(kvs[448 + 2 * lane])) |
                               ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(kvs[448 + 2 * lane + 1])) << 16);
            reinterpret_cast<unsigned*>(data + 448)[lane] = u;
            if (lane == 0) scl[7] = 0;
          }
          cbar();
        }
        __threadfence();
      }
      if (c >= NWK && c - NWK < NR) arrive(C_INS);   // row c - 128 inserted (or nothing to insert here): every rank
      if (AMODE(a) == 2 && rank == a.owner && c >= NWK && c - NWK < NR) {   // remote-KV mode: row c - 128's record is in
        cbar();                                                          // the owner's pages (every thread fenced above)
        if (tid < W && tid != rank) st_rel_sys(a.rdy[tid] + NCTA + (c - NWK), ncall);
      }
      if (AMODE(a) == 3 && rank == a.owner && c >= NWK && c - NWK < NR) {   // owner push of row c - 128's new record: every
        const int np = c - NWK;                                             // (row i, window index j) holding its slot
        const long long slot = __ldg(a.own + np);
        if (tid < NRM) S.m3j[tid] = -1;
        cbar();
        if (slot >= 0)
          for (int i = 0; i < NR; ++i)
            if (tid < WIN && tid < __ldg(a.sl + i) && (long long)__ldg(a.si + i * a.si_rs + tid) == slot) S.m3j[i] = tid;
        cbar();
        if (slot >= 0 && tid < NR * 37) {
          const int i = tid / 37, ch = tid % 37, j = S.m3j[i];
          if (j >= 0) {
            const long long pg = slot >> a.win_shift, ps = slot & ((1 << a.win_shift) - 1), r = (long long)i * KVR + j;
            if (ch < 36) {
              const uint4 x = __ldcg((const uint4*)(a.win + pg * a.win_stride + ps * 576) + ch);
#pragma unroll
              for (int p = 0; p < W; ++p)
                if (p != rank) *((uint4*)(a.stg[p] + r * STG_R) + ch) = x;
            } else {
              const uint2 x = __ldcg((const uint2*)(a.win + pg * a.win_stride + (576LL << a.win_shift) + ps * 8));
#pragma unroll
              for (int p = 0; p < W; ++p)
                if (p != rank) *(uint2*)(a.stg[p] + r * STG_R + 576) = x;
            }
          }
        }
        __threadfence_system();
        cbar();
        if (tid < W && tid != rank) red_rel_sys(a.rdy[tid] + NCTA + NRM, 1u);
      }
      cbar();
    }
    stamp(24);
if constexpr (HX) {
#define C1M_HOOK 3
#include C1M_HOOKS
#undef C1M_HOOK
}
    // ---- A5 q_b (this GPU's 16 heads): worker c takes the 4 tiles tq = 4c + t of 16 rows (rows 64c .., = dims 16 (tq & 31)
    // of head c / 8), RoPE on dims 448..511. 5 chunks of 256 K, warp w = k32 block w of every chunk for all 4 tiles (4
    // independent chains); then one cross-warp reduction (partials [warp][tile][lane][4] in the image region, free by
    // then) and the 4 tiles' epilogues on warps 0..3
    if (c < NWK) {
      static_assert(512 == 4 * NWK && NCW == 8, "q_b: 4 tiles per worker, 8 k32 blocks per chunk");
      float rc[4] = {1.f, 1.f, 1.f, 1.f}, rsn[4] = {0.f, 0.f, 0.f, 0.f};   // RoPE factors of this lane's 4 values
      if (w < 4) {
        const int tq = 4 * c + w;
        if ((tq & 31) >= 28)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1), d = 16 * (tq & 31) + row;
            if (n < NR) {
              const float* cs = a.cos_sin + __ldg(a.pos + n) * 64;
              const int ip = (d - 448) >> 1;
              rc[i] = __ldg(cs + ip), rsn[i] = __ldg(cs + 32 + ip);
            }
          }
      }
      float acc[4][4];
#pragma unroll
      for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[t][i] = 0.f;
      for (int kc = 0; kc < 5; ++kc) {
        const uint32_t st = cwait();
        const int kb = w;
        uint4 bv = make_uint4(0u, 0u, 0u, 0u);
        if (g < NR) bv = lds128(img + g * IMS + (256 * kc + 32 * kb) * 2 + 16 * t4);
        const uint32_t off = (kb >> 2) * 8192 + g * 128 + (((((kb & 3) << 1) | (t4 >> 1)) ^ g) << 4) + ((t4 & 1) << 3);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const uint32_t ra = st + t * 2048 + off;
          const uint2 wg = lds64(ra), wh = lds64(ra + 1024);
          const unsigned pg = __byte_perm(wg.x, wg.y, 0x6420), qg = __byte_perm(wg.x, wg.y, 0x7531);
          const unsigned ph = __byte_perm(wh.x, wh.y, 0x6420), qh = __byte_perm(wh.x, wh.y, 0x7531);
          unsigned g02, g46, h02, h46, g13, g57, h13, h57;
          cvt_e4m3x4(pg, g02, g46);
          cvt_e4m3x4(ph, h02, h46);
          cvt_e4m3x4(qg, g13, g57);
          cvt_e4m3x4(qh, h13, h57);
          float tmp[4] = {0.f, 0.f, 0.f, 0.f};
          mma16816(tmp, g02, h02, g46, h46, bv.x, bv.y);
          mma16816(tmp, g13, h13, g57, h57, bv.z, bv.w);
          unsigned sb;
          asm volatile("ld.shared.u8 %0, [%1];" : "=r"(sb) : "r"(st + L_QB_S + (t >> 1) * 48 + 8 * kc + kb));
          const float sc = ue8m0(sb);
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[t][i] = fmaf(tmp[i], sc, acc[t][i]);
        }
        crel();
      }
      stamp(35);
      cbar();   // every warp is done with the image: partials go there
      float* red = (float*)imgp;
#pragma unroll
      for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[((w * 4 + t) * 32 + lane) * 4 + i] = acc[t][i];
      cbar();
      if (w < 4) {
        const int t = w, tq = 4 * c + t;
        const bool rope = (tq & 31) >= 28;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float sm_ = red[((0 * 4 + t) * 32 + lane) * 4 + i];
          for (int ww = 1; ww < NCW; ++ww) sm_ += red[((ww * 4 + t) * 32 + lane) * 4 + i];
          const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
          const float v = __bfloat162float(__float2bfloat16_rn(sm_ * pow2i(-S.pv[n < NR ? n : 0])));
          const float pv = __shfl_xor_sync(~0u, v, 4);  // the pair partner: row g ^ 1
          float r = v;
          if (rope && n < NR) {
            const float e = (g & 1) ? pv : v, od = (g & 1) ? v : pv, cc = rc[i], sn = rsn[i];
            r = (g & 1) ? __fadd_rn(__fmul_rn(e, sn), __fmul_rn(od, cc)) : __fsub_rn(__fmul_rn(e, cc), __fmul_rn(od, sn));
          }
          if (n < NR) S.qt[n][16 * t + row] = __float2bfloat16_rn(r);
        }
      }
      stamp(36);
      cbar();
      if (tid < 8 * NR) {   // row n: dims dt0 + 8 p .. + 7 (16 B) of head hl, dt0 = 16 ((4c) & 31)
        const int n = tid >> 3, pp = tid & 7, hl = c >> 3, dt = 16 * ((4 * c) & 31) + 8 * pp;
        uint4 u = *(const uint4*)&S.qt[n][8 * pp];
        if (AMODE(a) == 1) {
          unsigned* uu = reinterpret_cast<unsigned*>(&u);
#pragma unroll
          for (int k = 0; k < 4; ++k) {   // -0.0 halves travel as +0.0 (the sentinel)
            if ((uu[k] & 0xffffu) == 0x8000u) uu[k] &= 0xffff0000u;
            if ((uu[k] >> 16) == 0x8000u) uu[k] &= 0x0000ffffu;
          }
          *(uint4*)(a.qall[a.owner] + (size_t)par * NRM * 64 * 512 + ((size_t)n * 64 + 16 * rank + hl) * 512 + dt) = u;
        } else {
          *(uint4*)(a.q + ((size_t)n * 16 + hl) * 512 + dt) = u;
        }
      }
    }
    if (c < NWK) arrive(C_QB);
    stamp(17);
if constexpr (HX) {
#define C1M_HOOK 4
#include C1M_HOOKS
#undef C1M_HOOK
}

    // ---- A6 attention (this GPU's 16 heads, local KV): units (row, chunk of AK keys), then combines (row, 32 dims)
    waitc(C_QB, 132u * ncall);
    stamp(18);
    {
      bf16* sk = (bf16*)imgp;                 // [AK][512] swizzled keys
      bf16* sq = sk + AK * 512;               // [16][512] swizzled q
      float* sred = (float*)(sq + 16 * 512);  // [4 dim groups][16 heads][AK]
      bf16* sp = (bf16*)(sred + 4 * 16 * AK); // [16][AK + 8]
      long long* soff = (long long*)(sp + 16 * (AK + 8));  // [AK][2]
      const uint8_t** sbs = (const uint8_t**)(soff + 2 * AK);
      auto aswz = [](int row, int col) { return row * 512 + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7); };
      const bool attend = AMODE(a) != 1 || rank == a.owner;
      const int NHG = AMODE(a) == 1 ? 4 : 1;
      if (!attend && c == 0 && tid < NR)   // the unit arrivals this rank skips: C_AT stays at NHG ANCH per call on every rank
        red_rel(a.cnt + C_AT + tid, (unsigned)(NHG * ANCH));   // (the owner may change between calls: engine runs)
      const bool staged = AMODE(a) == 3 && rank != a.owner;   // mode 3 peer: the records are in this GPU's staging
      if (staged && c < NR * NHG * ANCH) {   // every push of this layer has landed (NCTA early + NR new-record pushes)
        if (tid == 0) {
          const unsigned long long t0 = tim ? gtime() : 0ull;
          const unsigned tgt = (unsigned)(NCTA + NR) * ncall;
          while ((int)(ld_acq_sys(a.rdy[rank] + NCTA + NRM) - tgt) < 0) {
          }
          if (tim) st_b += gtime() - t0;
        }
        cbar();
      }
      if (attend && !staged && c < NR * NHG * ANCH) {   // local pages: the rows' own new records are inserted
        if (tid == 0) {
          const unsigned long long t0 = tim ? gtime() : 0ull;
          spin_ge(a.cnt + C_INS, (unsigned)NR * ncall);
          if (tim) st_b += gtime() - t0;
        }
        cbar();
      }
      for (int u = c; attend && u < NR * NHG * ANCH; u += NCTA) {
        const int n = u / (NHG * ANCH), hg = (u / ANCH) % NHG, ch = u % ANCH;
        if (AMODE(a) == 2 && rank != a.owner) {   // remote KV: the owner has inserted row n's record (this layer's epoch)
          if (tid == 0) {
            const unsigned long long t0 = tim ? gtime() : 0ull;
            while ((int)(ld_acq_sys(a.rdy[rank] + NCTA + n) - ncall) < 0) {
            }
            if (tim) st_b += gtime() - t0;
          }
          cbar();
        }
        if (tid < AK) {
          const int j = ch * AK + tid;
          long long d = -1, sc_ = -1;
          const uint8_t* b = nullptr;
          if (staged) {
            if (j < KVR && __ldcg(a.stg[rank] + STG_V + n * KVR + j)) {
              d = (long long)(n * KVR + j) * STG_R, sc_ = d + 576, b = a.stg[rank];
            }
          } else if (!kv_rec(a, n, j, b, d, sc_)) {
            d = -1, sc_ = -1, b = nullptr;
          }
          soff[2 * tid] = d, soff[2 * tid + 1] = sc_, sbs[tid] = b;
        }
        for (int i = tid; i < 16 * 64; i += CTH) {
          const int h = i >> 6, cc = i & 63;
          uint4 qv;
          if (AMODE(a) == 1) {   // poll the landed q (lamport: no -0.0 half once a GPU's store arrived)
            const bf16* qa_ = a.qall[rank] + (size_t)par * NRM * 64 * 512 + ((size_t)n * 64 + 16 * hg + h) * 512;
            bool ok;
            const unsigned long long t0_ = (tim && tid == 0) ? gtime() : 0ull;
            do {
              qv = ld_vol4((const uint4*)qa_ + cc);
              const unsigned uu[4] = {qv.x, qv.y, qv.z, qv.w};
              ok = true;
#pragma unroll
              for (int k = 0; k < 4; ++k) ok = ok && (uu[k] & 0xffffu) != 0x8000u && (uu[k] >> 16) != 0x8000u;
            } while (!ok);
            if (tim && tid == 0) st_b += gtime() - t0_;
          } else {
            qv = __ldcg((const uint4*)(a.q + ((size_t)n * 16 + h) * 512) + cc);
          }
          *(uint4*)&sq[aswz(h, 8 * cc)] = qv;
        }
        cbar();
        {  // keys: loads batched, then dequant (exact) into the swizzled tile
          constexpr int NT = (AK * 36 + CTH - 1) / CTH;
          uint4 raw[NT];
          unsigned ev[NT];
#pragma unroll
          for (int jq = 0; jq < NT; ++jq) {
            const int i = tid + CTH * jq;
            raw[jq] = make_uint4(0u, 0u, 0u, 0u), ev[jq] = 0;
            if (i < AK * 36) {
              const int k = i / 36, cc = i % 36;
              const long long d = soff[2 * k];
              if (d >= 0) {
                const uint8_t* b = sbs[k];
                raw[jq] = cc < 28 ? __ldcg((const uint4*)(b + d) + cc) : __ldcg((const uint4*)(b + d + 448) + (cc - 28));
                if (cc < 28) ev[jq] = __ldcg(b + soff[2 * k + 1] + (cc >> 2));
              }
            }
          }
#pragma unroll
          for (int jq = 0; jq < NT; ++jq) {
            const int i = tid + CTH * jq;
            if (i >= AK * 36) continue;
            const int k = i / 36, cc = i % 36;
            if (cc < 28) {
              uint4 lo = make_uint4(0u, 0u, 0u, 0u), hi = lo;
              if (soff[2 * k] >= 0) {
                const unsigned e = ev[jq];
                const float scl = e == 0 ? 5.877471754111438e-39f : __uint_as_float(e << 23);
                const unsigned wv[4] = {raw[jq].x, raw[jq].y, raw[jq].z, raw[jq].w};
                unsigned ob[8];
#pragma unroll
                for (int q4 = 0; q4 < 4; ++q4) {
                  unsigned h0, h1;
                  cvt_e4m3x4(wv[q4], h0, h1);
                  const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h0));
                  const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h1));
                  ob[2 * q4] = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(f0.x * scl)) |
                               ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(f0.y * scl)) << 16);
                  ob[2 * q4 + 1] = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(f1.x * scl)) |
                                   ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(f1.y * scl)) << 16);
                }
                lo = make_uint4(ob[0], ob[1], ob[2], ob[3]), hi = make_uint4(ob[4], ob[5], ob[6], ob[7]);
              }
              *(uint4*)&sk[aswz(k, 16 * cc)] = lo;
              *(uint4*)&sk[aswz(k, 16 * cc + 8)] = hi;
            } else {
              *(uint4*)&sk[aswz(k, 448 + 8 * (cc - 28))] = raw[jq];
            }
          }
        }
        cbar();
        {  // S partials: warp w = key half w >> 2 (16 keys), dims 128 (w & 3) ..
          const int kh = w >> 2, dg = w & 3;
          float sv[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            const int d0 = 128 * dg + 16 * ks;
            uint32_t qa[4], kb[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(qa[0]), "=r"(qa[1]), "=r"(qa[2]), "=r"(qa[3]) : "r"(su32(&sq[aswz(lane & 15, d0 + (lane >> 4) * 8)])));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(kb[0]), "=r"(kb[1]), "=r"(kb[2]), "=r"(kb[3])
                         : "r"(su32(&sk[aswz(16 * kh + (lane & 7) + ((lane >> 4) << 3), d0 + ((lane >> 3) & 1) * 8)])));
            hmma_bf16(sv[0], qa[0], qa[1], qa[2], qa[3], kb[0], kb[1]);
            hmma_bf16(sv[1], qa[0], qa[1], qa[2], qa[3], kb[2], kb[3]);
          }
#pragma unroll
          for (int ntl = 0; ntl < 2; ++ntl) {
            const int k0 = 16 * kh + 8 * ntl + 2 * t4;
            sred[(dg * 16 + g) * AK + k0] = sv[ntl][0], sred[(dg * 16 + g) * AK + k0 + 1] = sv[ntl][1];
            sred[(dg * 16 + g + 8) * AK + k0] = sv[ntl][2], sred[(dg * 16 + g + 8) * AK + k0 + 1] = sv[ntl][3];
          }
        }
        cbar();
        {  // local softmax: 16 threads per head
          const int h = tid >> 4, kl = tid & 15;
          float v[AK / 16], mxv = -INFINITY;
#pragma unroll
          for (int jj = 0; jj < AK / 16; ++jj) {
            const int k = kl + 16 * jj;
            float acc = 0.f;
#pragma unroll
            for (int dg = 0; dg < 4; ++dg) acc += sred[(dg * 16 + h) * AK + k];
            v[jj] = soff[2 * k] >= 0 ? acc * a.scale_log2 : -INFINITY;
            mxv = fmaxf(mxv, v[jj]);
          }
#pragma unroll
          for (int o = 8; o; o >>= 1) mxv = fmaxf(mxv, __shfl_xor_sync(~0u, mxv, o));
          float sum = 0.f;
#pragma unroll
          for (int jj = 0; jj < AK / 16; ++jj) {
            const float e = v[jj] == -INFINITY ? 0.f : exp2f(v[jj] - mxv);
            sum += e;
            sp[h * (AK + 8) + kl + 16 * jj] = __float2bfloat16_rn(e);
          }
#pragma unroll
          for (int o = 8; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
          if (kl == 0) a.aml[(size_t)u * 32 + h] = mxv, a.aml[(size_t)u * 32 + 16 + h] = sum;   // u = (n, hg, ch)
        }
        cbar();
        {  // O partial = P V: warp w = dims 64 w ..
          float ov[8][4];
#pragma unroll
          for (int nt8 = 0; nt8 < 8; ++nt8) ov[nt8][0] = ov[nt8][1] = ov[nt8][2] = ov[nt8][3] = 0.f;
#pragma unroll
          for (int ks = 0; ks < AK / 16; ++ks) {
            uint32_t pa[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(pa[0]), "=r"(pa[1]), "=r"(pa[2]), "=r"(pa[3])
                         : "r"(su32(&sp[(lane & 15) * (AK + 8) + ks * 16 + (lane >> 4) * 8])));
#pragma unroll
            for (int nt8 = 0; nt8 < 8; nt8 += 2) {
              uint32_t vb[4];
              asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                           : "=r"(vb[0]), "=r"(vb[1]), "=r"(vb[2]), "=r"(vb[3])
                           : "r"(su32(&sk[aswz(ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, 64 * w + nt8 * 8 + (lane >> 4) * 8)])));
              hmma_bf16(ov[nt8], pa[0], pa[1], pa[2], pa[3], vb[0], vb[1]);
              hmma_bf16(ov[nt8 + 1], pa[0], pa[1], pa[2], pa[3], vb[2], vb[3]);
            }
          }
          float* pp = a.apart + (size_t)u * 16 * 512;
#pragma unroll
          for (int nt8 = 0; nt8 < 8; ++nt8) {
            const int d = 64 * w + 8 * nt8 + 2 * t4;
            *(float2*)(pp + (size_t)g * 512 + d) = make_float2(ov[nt8][0], ov[nt8][1]);
            *(float2*)(pp + (size_t)(g + 8) * 512 + d) = make_float2(ov[nt8][2], ov[nt8][3]);
          }
        }
        cbar();
        if (tid == 0) {
          __threadfence();
          red_rel(a.cnt + C_AT + n, 1u);
        }
      }
      stamp(25);
      // combines: (row n, 32-dim slice sl); thread = (head tid / 16, 2 dims)
      for (int v = c; attend && v < NR * NHG * 16; v += NCTA) {
        const int n = v / (NHG * 16), hg = (v / 16) % NHG, sl = v & 15;
        if (tid == 0) {
          const unsigned long long t0 = tim ? gtime() : 0ull;
          spin_ge(a.cnt + C_AT + n, (unsigned)(NHG * ANCH) * ncall);
          if (tim) st_b += gtime() - t0;
        }
        cbar();
        const int h = tid >> 4, d = 32 * sl + 2 * (tid & 15), ub = (n * NHG + hg) * ANCH;
        const float sink2 = __ldg(a.sink + (AMODE(a) == 1 ? 16 * hg : 16 * rank) + h) * 1.4426950408889634f;
        float mv[ANCH], lv[ANCH];
        float2 ov[ANCH];
#pragma unroll
        for (int cc = 0; cc < ANCH; ++cc) {
          mv[cc] = __ldcg(a.aml + (size_t)(ub + cc) * 32 + h), lv[cc] = __ldcg(a.aml + (size_t)(ub + cc) * 32 + 16 + h);
          ov[cc] = __ldcg((const float2*)(a.apart + ((size_t)(ub + cc) * 16 + h) * 512 + d));
        }
        float mt = sink2;
#pragma unroll
        for (int cc = 0; cc < ANCH; ++cc) mt = fmaxf(mt, mv[cc]);
        float den = sink2 == -INFINITY ? 0.f : exp2f(sink2 - mt), a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int cc = 0; cc < ANCH; ++cc) {
          const float wc = mv[cc] == -INFINITY ? 0.f : exp2f(mv[cc] - mt);
          den += wc * lv[cc];
          a0 += wc * ov[cc].x, a1 += wc * ov[cc].y;
        }
        const float inv = (den > 0.f && mt < INFINITY) ? 1.f / den : 0.f;
        bf16 b0 = __float2bfloat16_rn(a0 * inv), b1 = __float2bfloat16_rn(a1 * inv);
        if (d >= 448) {  // inverse RoPE (comm_c1 inv_rope4 / dense.cu rope_pair<true>)
          const long long pn = staged ? __ldcg((const long long*)(a.stg[rank] + STG_P) + n) : __ldg(a.pos + n);
          const float* cs = a.cos_sin + pn * 64;
          const int ip = (d - 448) >> 1;
          const float cc = __ldg(cs + ip), sn = __ldg(cs + 32 + ip), e = __bfloat162float(b0), od = __bfloat162float(b1);
          b0 = __float2bfloat16_rn(__fadd_rn(__fmul_rn(e, cc), __fmul_rn(od, sn)));
          b1 = __float2bfloat16_rn(__fsub_rn(__fmul_rn(od, cc), __fmul_rn(e, sn)));
        }
        *(unsigned*)&S.ot[h][2 * (tid & 15)] = (unsigned)__bfloat16_as_ushort(b0) | ((unsigned)__bfloat16_as_ushort(b1) << 16);
        cbar();
        if (tid < 64) {   // 16 heads x 32 dims = 64 chunks of 16 B: local o (mode 0) or GPU hg's og landing buffer (mode 1)
          const int hh = tid >> 2, dz = 8 * (tid & 3);
          uint4 u = *(const uint4*)&S.ot[hh][dz];
          if (AMODE(a) == 1) {
            unsigned* uu = reinterpret_cast<unsigned*>(&u);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              if ((uu[k] & 0xffffu) == 0x8000u) uu[k] &= 0xffff0000u;
              if ((uu[k] >> 16) == 0x8000u) uu[k] &= 0x0000ffffu;
            }
            *(uint4*)(a.ogl[hg] + (size_t)par * NRM * 8192 + (size_t)n * 8192 + hh * 512 + 32 * sl + dz) = u;
            // this slice of the landed q is consumed (every unit of row n has loaded it): back to the sentinel
            *(uint4*)(a.qall[rank] + (size_t)par * NRM * 64 * 512 + ((size_t)n * 64 + 16 * hg + hh) * 512 + 32 * sl + dz) =
                make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
          } else {
            *(uint4*)(a.o + ((size_t)n * 16 + hh) * 512 + 32 * sl + dz) = u;
          }
        }
        cbar();
      }
      arrive(C_O);
    }
    stamp(19);
  }

  // ---------------------------------------------------------------------------------------------- F1 wo_a
  if (c < 128) {
    bool landed = false;
    if constexpr (FULL) {
      landed = AMODE(a) == 1;
      if (!landed) waitc(C_O, 132u * ncall);  // og = this GPU's attention output
    }
    const bf16* ogp = landed ? a.ogl[rank] + (size_t)par * NRM * 8192 : a.og;
    const int gi = c >> 6, tr = (c & 63) * 16;
    {   // og group gi -> f16 x 2^p image; NR x 512 chunks, thread tid: chunk tid + 256 j (2 per row), loads first
      constexpr int NJ = NRM * 512 / CTH;
      uint4 v[NJ];
      float mx[NRM] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        const int i = tid + CTH * j, n = i >> 9, c8 = i & 511;
        v[j] = make_uint4(0u, 0u, 0u, 0u);
        if (n < NR) {
          const uint4* src = (const uint4*)(ogp + (size_t)n * 8192 + 4096 * gi) + c8;
          if (landed) {   // poll the owner's o slice (lamport)
            bool ok;
            const unsigned long long t0_ = (tim && tid == 0) ? gtime() : 0ull;
            do {
              v[j] = ld_vol4(src);
              const unsigned uu[4] = {v[j].x, v[j].y, v[j].z, v[j].w};
              ok = true;
#pragma unroll
              for (int k = 0; k < 4; ++k) ok = ok && (uu[k] & 0xffffu) != 0x8000u && (uu[k] >> 16) != 0x8000u;
            } while (!ok);
            if (tim && tid == 0) st_b += gtime() - t0_;
          } else {
            v[j] = __ldcg(src);
          }
        }
      }
      if (B16(a, woaT16, 1)) {   // bf16 wo_a: the natural bf16 image
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          const int i = tid + CTH * j, n = i >> 9, c8 = i & 511;
          if (n < NR) *(uint4*)(imgp + n * IMS + 16 * c8) = v[j];
        }
      } else {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          const int n = (tid + CTH * j) >> 9;
          const float b = bmax8(v[j]);
#pragma unroll
          for (int q = 0; q < NRM; ++q)   // (static indices: mx stays in registers)
            if (q == n && q < NR) mx[q] = fmaxf(mx[q], b);
        }
        rowp(mx);
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          const int i = tid + CTH * j, n = i >> 9, c8 = i & 511;
          if (n < NR) *(uint4*)(imgp + n * IMS + 16 * c8) = p8_f16(v[j], pow2i(S.pv[n]));
        }
      }
    }
    cbar();
    stamp(26);
    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
    const bool b16 = B16(a, woaT16, 1);
    for (int kc = 0; kc < (b16 ? 8 : 4); ++kc) {
      const uint32_t st = cwait();
      if (b16) dense16t_chunk(acc[0], st, 32, img, 512 * kc, NR, w, lane);
      else dense_chunk<1>(acc, st, 0, 32, st + L_D_S, 32, img, 1024 * kc, NR, w, lane);
      crel();
    }
    reduce_warps<1>(acc, S.red, w, lane);
    if (w == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
        if (n < NR) a.z[(size_t)n * 2048 + 1024 * gi + tr + row] = __float2bfloat16_rn(b16 ? acc[0][i] : acc[0][i] * pow2i(-S.pv[n]));
      }
    }
    arrive(C_Z);
  }
  stamp(1);
  if constexpr (FULL)
    if (c >= NWK && c - NWK < NR) {   // the attention boundary's mixes of row c - 128 (needed by the FFN boundary)
      mixes(c - NWK, a.mixp_a, a.ascale, a.abase, a.apre_o, a.apost_o, a.acomb_o);
      arrive(C_MIXA);
      stamp(22);
    }
  // ---------------------------------------------------------------------------------------------- F2 wo_b
  waitc(C_Z, 128u * ncall);
  stamp(2);
  if constexpr (FULL)
    if (AMODE(a) == 1 && c == NCTA - 1)   // every wo_a CTA has read the landed og: back to the sentinel (next use: 2 calls later)
      for (int i = tid; i < NR * 1024; i += CTH)
        *((uint4*)(a.ogl[rank] + (size_t)par * NRM * 8192) + i) = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
  {
    {   // z -> f16 x 2^p image: NR x 256 chunks, thread tid: rows of chunk tid (loads first)
      uint4 v[NRM];
      float mx[NRM] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int n = 0; n < NRM; ++n) v[n] = n < NR ? __ldcg((const uint4*)(a.z + (size_t)n * 2048) + tid) : make_uint4(0u, 0u, 0u, 0u);
      if (B16(a, wob16, 2)) {   // bf16 wo_b: the natural bf16 image
#pragma unroll
        for (int n = 0; n < NRM; ++n)
          if (n < NR) *(uint4*)(imgp + n * IMS + 16 * tid) = v[n];
      } else {
#pragma unroll
        for (int n = 0; n < NRM; ++n) mx[n] = bmax8(v[n]);
        rowp(mx);
#pragma unroll
        for (int n = 0; n < NRM; ++n)
          if (n < NR) *(uint4*)(imgp + n * IMS + 16 * tid) = p8_f16(v[n], pow2i(S.pv[n]));
      }
    }
    cbar();
    const bool b16 = B16(a, wob16, 2);
    for (int ti = 0; ti < nt; ++ti) {
      float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
      for (int kc = 0; kc < (b16 ? 4 : 2); ++kc) {
        const uint32_t st = cwait();
        if (b16) dense16_chunk<1>(acc, st, 0, 32, img, 512 * kc, NR, w, lane);
        else dense_chunk<1>(acc, st, 0, 32, st + L_D_S, 32, img, 1024 * kc, NR, w, lane);
        crel();
      }
      reduce_warps<1>(acc, S.red, w, lane);
      if (w == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int row = g + 8 * (i >> 1), n = 2 * t4 + (i & 1);
          if (n < NR) S.yp[n][16 * ti + row] = b16 ? acc[0][i] : acc[0][i] * pow2i(-S.pv[n]);
        }
      }
      cbar();
    }
  }
  stamp(3);
  // ---------------------------------------------------------------------------------------------- AR #1 (lamport)
  {
    const int nq = ncols / 4;  // float4 per row
    for (int d = 1; d < W; ++d) {
      const int s = (rank + d) % W;
      float* dst = a.ar1[s] + (((size_t)par * W + rank) * NRM) * D;
      for (int i = tid; i < NR * nq; i += CTH) {
        const int n = i / nq, q = i % nq;
        float4 v = *(const float4*)&S.yp[n][4 * q];
        uint4 u = *reinterpret_cast<uint4*>(&v);
        u.x = u.x == SENT ? 0u : u.x, u.y = u.y == SENT ? 0u : u.y, u.z = u.z == SENT ? 0u : u.z, u.w = u.w == SENT ? 0u : u.w;
        *(uint4*)(dst + (size_t)n * D + col0 + 4 * q) = u;
      }
    }
    float* mine = a.ar1[rank] + (size_t)par * W * NRM * D;
    for (int i = tid; i < NR * nq; i += CTH) {
      const int n = i / nq, q = i % nq;
      float4 v[W];
#pragma unroll
      for (int s = 0; s < W; ++s) {
        if (s == rank) {
          v[s] = *(const float4*)&S.yp[n][4 * q];
          continue;
        }
        float* sl = mine + ((size_t)s * NRM + n) * D + col0 + 4 * q;
        uint4 u;
        const unsigned long long t0_ = (tim && tid == 0) ? gtime() : 0ull;
        do {
          u = ld_vol4(sl);
        } while (u.x == SENT || u.y == SENT || u.z == SENT || u.w == SENT);
        if (tim && tid == 0) st_b += gtime() - t0_;
        v[s] = *reinterpret_cast<float4*>(&u);
        *(uint4*)sl = make_uint4(SENT, SENT, SENT, SENT);
      }
      float4 r = v[0];
#pragma unroll
      for (int s = 1; s < W; ++s) r.x += v[s].x, r.y += v[s].y, r.z += v[s].z, r.w += v[s].w;
      const bf16 b0 = __float2bfloat16_rn(r.x), b1 = __float2bfloat16_rn(r.y), b2 = __float2bfloat16_rn(r.z),
                 b3 = __float2bfloat16_rn(r.w);
      S.ybf[n][4 * q] = b0, S.ybf[n][4 * q + 1] = b1, S.ybf[n][4 * q + 2] = b2, S.ybf[n][4 * q + 3] = b3;
      bf16* yo = a.y + (size_t)n * D + col0 + 4 * q;
      yo[0] = b0, yo[1] = b1, yo[2] = b2, yo[3] = b3;
    }
    cbar();
  }
  stamp(4);
  // ---------------------------------------------------------------------------------------------- G FFN boundary slice
  if constexpr (FULL) {  // the attention boundary's mixes (CTA 131 of the front half)
    waitc(C_MIXA, (unsigned)NR * ncall);
    if (tid < NR * 24) {
      const int n = tid / 24, k = tid % 24;
      if (k < 4) S.post[n][k] = __ldcg(a.apost_o + n * 4 + k);
      else if (k < 20) S.comb[n][k - 4] = __ldcg(a.acomb_o + n * 16 + k - 4);
      else S.lag[n][k - 20] = __ldcg(a.apre_o + n * 4 + k - 20);
    }
    cbar();
  }
  bslice([&](int n, int j, int col) { return __bfloat162float(S.ybf[n][j]); }, a.X, a.xm2, a.inb, a.ssp, a.mixp, C_SS, 31);
  stamp(5);
  // ---------------------------------------------------------------------------------------------- H RMSNorm -> f
  waitc(C_SS, 132u * ncall);
  stamp(6);
  rnorm(a.ssp, a.inb, a.nw, c == 0 ? a.fout : nullptr);
  stamp(20);
  if (c == 0 && tid < NR) a.ptok[tid] = S.pv[tid];
  stamp(7);
  // ---------------------------------------------------------------------------------------------- H router rows
  for (int e = c; c < NWK && e < NE; e += 2 * NWK) {
    const int nrw = e + NWK < NE ? 2 : 1;
    const uint32_t st = cwait();
    const int rs = w >> 2, lt = tid & 127;   // warps 0-3: row e, warps 4-7: row e + 128
    float sv[NRM] = {0.f, 0.f, 0.f, 0.f};
    if (rs < nrw) {
      for (int q = lt; q < D / 8; q += 128) {
        const uint4 wv = lds128(st + rs * D * 2 + 16 * q);
        const unsigned wu[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int n = 0; n < NRM; ++n) {
          if (n >= NR) break;
          const uint4 fv = lds128(img + n * IMS + 16 * q);
          const unsigned fu[4] = {fv.x, fv.y, fv.z, fv.w};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            sv[n] = fmaf(__uint_as_float(wu[k] << 16), __uint_as_float(fu[k] << 16), sv[n]);
            sv[n] = fmaf(__uint_as_float(wu[k] & 0xffff0000u), __uint_as_float(fu[k] & 0xffff0000u), sv[n]);
          }
        }
      }
    }
#pragma unroll
    for (int n = 0; n < NRM; ++n) {
      float m = sv[n];
#pragma unroll
      for (int o = 16; o; o >>= 1) m += __shfl_xor_sync(~0u, m, o);
      if (lane == 0) S.wmax[w][n] = m;
    }
    cbar();
    if (tid < 2 * NR) {
      const int r = tid / NR, n = tid % NR;
      if (r < nrw)
        a.rtp[n * NE + e + r * NWK] = ((S.wmax[4 * r][n] + S.wmax[4 * r + 1][n]) + S.wmax[4 * r + 2][n]) + S.wmax[4 * r + 3][n];
    }
    crel();
    cbar();
  }
  stamp(23);
  arrive(C_RT);
  if (c >= NWK && c - NWK < NR) {   // service CTA 128 + n (no router rows): the FFN mixes of row n, after its C_RT arrival
    mixes(c - NWK, a.mixp, a.fscale, a.fbase, a.fpre, a.fpost, a.fcomb);
    arrive(C_MIXF);
    stamp(21);
  }
  // f image: natural bf16 -> f16 x 2^p in P8 order (in place)
  for (int i = tid; i < NR * 640; i += CTH) {
    const int n = i / 640, c8 = i % 640;
    uint4* q = (uint4*)(imgp + n * IMS + 16 * c8);
    *q = p8_f16(*q, pow2i(S.pv[n]));
  }
  stamp(8);
  waitc(C_RT, 132u * ncall);
  stamp(9);
  // ---------------------------------------------------------------------------------------------- top-6 (redundant)
  // FULL: the scores (softplus, sqrt: latency-bound math) by all consumer threads into the attention scratch (free from
  // the router to the combine), then one warp per row selects; the same values as one warp per row computing them
  constexpr bool SC8 = FULL && ATTX >= 2 * NRM * NE * 4;
  float* const scS = reinterpret_cast<float*>(imgp + NRM * IMS);
  float* const selS = scS + NRM * NE;
  if constexpr (SC8) {
    for (int v = tid; v < NR * NE; v += CTH) {
      const int e = v % NE;
      const float x = __ldcg(a.rtp + v);
      if (c == 0) a.logits[v] = x;
      const float sv = sqrtf(x > 20.f ? x : log1pf(expf(x)));
      scS[v] = sv, selS[v] = sv + __ldg(a.gate_b + e);
    }
    cbar();
  }
  if (w < NR) {
    const int n = w;
    float sc[NEW], sel[NEW];
#pragma unroll
    for (int i = 0; i < NEW; ++i) {
      const int e = i * 32 + lane;
      if constexpr (SC8) {
        sc[i] = e < NE ? scS[n * NE + e] : 0.f, sel[i] = e < NE ? selS[n * NE + e] : -INFINITY;
      } else {
        const float x = __ldcg(a.rtp + n * NE + e);
        if (c == 0) a.logits[n * NE + e] = x;
        sc[i] = sqrtf(x > 20.f ? x : log1pf(expf(x)));
        sel[i] = sc[i] + __ldg(a.gate_b + e);
      }
    }
    if (n == 0) stamp(40);
    float pick[TOPK];
    int pid[TOPK];
    unsigned taken = 0;
#pragma unroll
    for (int r = 0; r < TOPK; ++r) {   // the max (lowest expert id on ties) by two warp reductions on order-preserving keys
      float bv = -INFINITY;
      int bi = 1 << 30;
#pragma unroll
      for (int i = 0; i < NEW; ++i)
        if (!((taken >> i) & 1) && sel[i] > bv) bv = sel[i], bi = i * 32 + lane;
      const unsigned ub = __float_as_uint(bv), key = (ub & 0x80000000u) ? ~ub : (ub | 0x80000000u);
      const unsigned mk = __reduce_max_sync(0xffffffffu, key);
      bi = (int)__reduce_min_sync(0xffffffffu, key == mk ? (unsigned)bi : 0xffffffffu);
      pid[r] = bi;
      if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
      float s_r = 0.f;
#pragma unroll
      for (int i = 0; i < NEW; ++i)
        if (i * 32 + lane == bi) s_r = sc[i];
      pick[r] = __shfl_sync(0xffffffffu, s_r, bi & 31);
    }
    if (lane == 0) {
      float sum = 0.f;
#pragma unroll
      for (int r = 0; r < TOPK; ++r) sum += pick[r];
#pragma unroll
      for (int r = 0; r < TOPK; ++r) {
        const float wv = pick[r] / (sum + 1e-20f) * a.route_scale;
        S.ids[n][r] = pid[r], S.wts[n][r] = wv;
        if (c == 0) a.ids[n * TOPK + r] = pid[r], a.wts[n * TOPK + r] = wv;
      }
    }
  }
  cbar();
  stamp(37);
  // distinct experts in ascending order: a 384-bit bitmap, word prefix counts, slot = rank of the id among the chosen
  if (tid < NEW) S.bm[tid] = 0u;
  if (tid < MAXX * NRM) S.xw[tid / NRM][tid % NRM] = 0.f, S.xk[tid / NRM][tid % NRM] = -1;
  cbar();
  if (tid < NR * TOPK) {
    const int e = S.ids[tid / TOPK][tid % TOPK];
    atomicOr(&S.bm[e >> 5], 1u << (e & 31));
  }
  cbar();
  if (w == 0) {
    const unsigned word = lane < NEW ? S.bm[lane] : 0u;
    int incl = __popc(word);
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(~0u, incl, o);
      if (lane >= o) incl += v;
    }
    if (lane < NEW) S.bpre[lane] = incl - __popc(word);
    if (lane == 31) S.nexp = incl;
  }
  cbar();
  if (tid < NR * TOPK) {
    const int n = tid / TOPK, r = tid % TOPK, e = S.ids[n][r];
    const int j = S.bpre[e >> 5] + __popc(S.bm[e >> 5] & ((1u << (e & 31)) - 1u));
    S.xid[j] = e, S.xw[j][n] = S.wts[n][r], S.xk[j][n] = r;
  }
  cbar();
  if (tid == 0) {
    if (c == 0)  // unused h slots: their arrivals, so every slot counts 36 per call
      for (int j = S.nexp; j < MAXX; ++j) red_rel(a.cnt + C_H + j, (unsigned)SH_NU);
    mb_arrive(su32(&S.route));
  }
  cbar();
  stamp(10);
  const int nexp = S.nexp;
  // the expert phase: GEMM1 (shared halves + routed items), GEMM2 routed + shared -> S.Ybf / S.shbf
#include C1M_MOE_CONSUME
  cbar();
  stamp(12);
  // ---------------------------------------------------------------------------------------------- combine + AR #2
  for (int i = tid; i < NR * ncols; i += CTH) {
    const int n = i / ncols, j = i % ncols;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) s += __bfloat162float(S.Ybf[n][k][j]);
    S.part[n][j] = __float2bfloat16_rn(s + __bfloat162float(S.shbf[n][j]));
  }
  cbar();
  {
    const int nq = ncols / 8;  // 16-B chunks per row
    for (int d = 1; d < W; ++d) {
      const int s = (rank + d) % W;
      bf16* dst = a.ar2[s] + (((size_t)par * W + rank) * NRM) * D;
      for (int i = tid; i < NR * nq; i += CTH) {
        const int n = i / nq, q = i % nq;
        uint4 u = *(const uint4*)&S.part[n][8 * q];
        unsigned* uu = reinterpret_cast<unsigned*>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) {  // -0.0 halves travel as +0.0
          if ((uu[k] & 0xffffu) == 0x8000u) uu[k] &= 0xffff0000u;
          if ((uu[k] >> 16) == 0x8000u) uu[k] &= 0x0000ffffu;
        }
        *(uint4*)(dst + (size_t)n * D + col0 + 8 * q) = u;
      }
    }
    bf16* mine = a.ar2[rank] + (size_t)par * W * NRM * D;
    for (int i = tid; i < NR * nq; i += CTH) {
      const int n = i / nq, q = i % nq;
      float acc[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
      for (int s = 0; s < W; ++s) {
        uint4 u;
        if (s == rank) {
          u = *(const uint4*)&S.part[n][8 * q];
        } else {
          bf16* sl = mine + ((size_t)s * NRM + n) * D + col0 + 8 * q;
          bool ok;
          const unsigned long long t0_ = (tim && tid == 0) ? gtime() : 0ull;
          do {
            u = ld_vol4(sl);
            const unsigned uu[4] = {u.x, u.y, u.z, u.w};
            ok = true;
#pragma unroll
            for (int k = 0; k < 4; ++k) ok = ok && (uu[k] & 0xffffu) != 0x8000u && (uu[k] >> 16) != 0x8000u;
          } while (!ok);
          if (tim && tid == 0) st_b += gtime() - t0_;
          *(uint4*)sl = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
        }
        const unsigned uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[2 * k] += __uint_as_float(uu[k] << 16), acc[2 * k + 1] += __uint_as_float(uu[k] & 0xffff0000u);
      }
      unsigned o[4];
#pragma unroll
      for (int k = 0; k < 4; ++k)
        o[k] = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(acc[2 * k])) |
               ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(acc[2 * k + 1])) << 16);
      *(uint4*)(a.moe_out + (size_t)n * D + col0 + 8 * q) = make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
  cbar();
  stamp(13);
}
#ifdef C1M_HAS_PHASES
template <bool FULL>
__device__ __noinline__ CState cons_ix(const Args* __restrict__ al, const int nly, const int l, CState cs) {
  cons_layer<FULL, true>(al, nly, l, cs);
  return cs;
}
#endif

template <bool FULL>
__global__ void __launch_bounds__(THR, 1) k_layer(const Args* __restrict__ al, const int nly) {
  static_assert(!FULL || NCW == 8, "FULL: 8 consumer warps (256 threads)");
  constexpr int NS = LayCfg<FULL>::NS, ATTX = LayCfg<FULL>::ATTX;
  extern __shared__ __align__(1024) char smem[];
  char* ring = smem;
  char* imgp = smem + NS * STAGE;
  Smem& S = *reinterpret_cast<Smem*>(smem + NS * STAGE + NRM * IMS + ATTX);
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, c = blockIdx.x;
  const int NR = al[0].NR, rank = al[0].rank;
  int ftb, nt;
  my_tiles(c, ftb, nt);
  const int col0 = 16 * ftb, ncols = 16 * nt;
  const uint32_t ring_u = su32(ring), img = su32(imgp);
  __shared__ unsigned call_s;
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) mb_init(su32(&S.full[i]), 1), mb_init(su32(&S.empty[i]), NCW);
    mb_init(su32(&S.route), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    call_s = al[0].tick[c];
  }
  __syncthreads();
  const unsigned call = call_s;   // layers processed by this CTA so far (all launches): layer l of this launch = epoch call + l
  int ltim = nly - 1;   // the stamped layer: the first one whose Args carry a timeline (pack: usually the last)
  for (int l2 = 0; l2 < nly; ++l2)
    if (al[l2].tim) {
      ltim = l2;
      break;
    }
#ifdef C1M_NOTIM
  unsigned long long* const tim0 = nullptr;
#else
  unsigned long long* const tim0 = al[ltim].tim ? al[ltim].tim + (size_t)c * NTIM : nullptr;
#endif
  if (tim0 && tid == 0 && ltim == 0)   // (kernel start)
    tim0[0] = gtime(), tim0[NTS] = 0ull, tim0[2 * NTS] = 0ull, tim0[3 * NTS] = clock64();

  // number of router pieces / GEMM1 items of CTA c (items g = c + 132 i: g < 72 shared halves, then routed (slot, tile))
if (w == NCW + 1) {   // the KV-push warp: attention mode 3, the owner's push of the rows' selected KV records
    if constexpr (FULL) {
      const int pl = lane;   // 0..31
      for (int l = 0; l < nly; ++l) {
        const Args& A = al[l];
        if (AMODE(A) != 3 || rank != A.owner) continue;
        const unsigned ncall = call + l + 1u;
        // layer 0 of the launch: the window AND top-k records (the top-k set is the run's: the same KV source and index
        // source on every layer of a run); later layers: the window only, once every peer is past the previous
        // layer's attention (the owner's FFN boundary of layer l - 1 runs after its AR #1, i.e. after the peers' wo_b)
        if (l > 0) spin_ge(A.cnt + C_SS, 132u * (ncall - 1u));
#define C1M_HOOK 5
#include C1M_HOOKS
#undef C1M_HOOK
        const int nj = (l == 0 || A.push_topk) ? KVR : WIN;   // index layers inside a run: their new top-k too
        const long long o0 = NR > 0 ? __ldg(A.own) : -1, o1 = NR > 1 ? __ldg(A.own + 1) : -1;
        const long long o2 = NR > 2 ? __ldg(A.own + 2) : -1, o3 = NR > 3 ? __ldg(A.own + 3) : -1;
        // record info on lane k (this CTA's k-th record rr = c + NCTA k), then all 32 lanes copy chunks in batches of 8
        // independent loads (the copy is latency-bound otherwise: ~1 us per dependent HBM round)
        const int nrec = (NR * nj - c + NCTA - 1) / NCTA;   // <= 20
        long long rk = 0;
        const uint8_t* srcd = nullptr;
        const uint8_t* srcs = nullptr;
        int cp = 0;   // this lane's record is copied (valid and not one of the rows' new records)
        if (pl < nrec) {
          const int rr = c + NCTA * pl, n = rr / nj, j = rr % nj;
          rk = (long long)n * KVR + j;
          const uint8_t* b = nullptr;
          long long d = 0, sc = 0;
          const bool v = kv_rec(A, n, j, b, d, sc);
          bool fresh = false;   // the rows' own new records: inserted in this layer's A4, pushed after the insert
          if (v && j < WIN) {
            const long long sl_ = __ldg(A.si + n * A.si_rs + j);
            fresh = sl_ == o0 || sl_ == o1 || sl_ == o2 || sl_ == o3;
          }
#pragma unroll
          for (int p = 0; p < W; ++p)
            if (p != rank) A.stg[p][STG_V + rk] = v ? 1 : 0;
          cp = v && !fresh;
          if (cp) srcd = b + d, srcs = b + sc;
        }
        const int ntask = nrec * 37;
        for (int t0 = 0; t0 < ntask; t0 += 8 * 32) {
          uint4 x[8];
          long long dst[8];
          int ok[8];
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            const int t = t0 + 32 * u + pl, k = t / 37, q = t % 37;
            const int ks = k < 32 ? k : 0;
            const int cpk = __shfl_sync(~0u, cp, ks);
            const long long rkk = __shfl_sync(~0u, rk, ks);
            const unsigned long long sd = __shfl_sync(~0u, (unsigned long long)srcd, ks);
            const unsigned long long ss = __shfl_sync(~0u, (unsigned long long)srcs, ks);
            ok[u] = t < ntask && cpk;
            dst[u] = rkk * STG_R + (q < 36 ? 16 * q : 576);
            x[u] = make_uint4(0u, 0u, 0u, 0u);
            if (ok[u]) {
              if (q < 36) x[u] = __ldcg((const uint4*)sd + q);
              else {
                const uint2 y = __ldcg((const uint2*)ss);
                x[u].x = y.x, x[u].y = y.y;
              }
            }
          }
#pragma unroll
          for (int u = 0; u < 8; ++u)
            if (ok[u]) {
              const bool sc8 = (dst[u] % STG_R) == 576;
#pragma unroll
              for (int p = 0; p < W; ++p)
                if (p != rank) {
                  if (sc8) *(uint2*)(A.stg[p] + dst[u]) = make_uint2(x[u].x, x[u].y);
                  else *(uint4*)(A.stg[p] + dst[u]) = x[u];
                }
            }
        }
        if (l == 0 && c == 0 && pl < NR)   // the rows' positions (the peers' inverse RoPE in the combine)
#pragma unroll
          for (int p = 0; p < W; ++p)
            if (p != rank) ((long long*)(A.stg[p] + STG_P))[pl] = __ldg(A.pos + pl);
        __threadfence_system();
        __syncwarp();
        if (pl == 0) {
#pragma unroll
          for (int p = 0; p < W; ++p)
            if (p != rank) red_rel_sys(A.rdy[p] + NCTA + NRM, 1u);
          if (c == 0) red_rel_sys(A.rdy[rank] + NCTA + NRM, (unsigned)(NCTA + NR));   // (the owner's own count: epochs)
        }
      }
    }
    return;
  }
  if (w == NCW) {
    // ============================================================================================ producer
    if (lane != 0) return;
    const uint64_t pol = pol_first();
    long it = 0;
    for (int l = 0; l < nly; ++l) {
    const Args& a = al[l];
    const unsigned ncall = call + l + 1u;
    auto slot = [&](unsigned bytes) -> uint32_t {
      const int s = (int)(it % NS);
      mb_wait(su32(&S.empty[s]), (unsigned)(((it / NS) & 1) ^ 1));
      mb_expect(su32(&S.full[s]), bytes);
      return (uint32_t)s;
    };
    auto sbase = [&](uint32_t s) { return ring_u + s * STAGE; };
    if constexpr (FULL) {
      // A1: hc_attn_fn slice [24][4][ncols] fp32
      {
        const uint32_t s = slot(96 * 48 * 4), b = sbase(s), fb = su32(&S.full[s]);
        bulk(b, a.afn + (size_t)c * 96 * 48, 96 * 48 * 4, fb, pol);
        ++it;
      }
      // A3: q_a | wkv tile c (16 rows x 5120): 5 chunks of 1024 K (fp8) or 10 chunks of 512 K (the engine's bf16)
      if (c < 112 && B16(a, wqkv16, 0))
        for (int kc = 0; kc < 10; ++kc) {
          const uint32_t s = slot(16 * 512 * 2), b = sbase(s), fb = su32(&S.full[s]);
          tma3d(b, a.tm + TM_WQKV16, 0, 16 * c, 8 * kc, fb, pol);
          ++it;
        }
      else if (c < 112)
        for (int kc = 0; kc < 5; ++kc) {
          const uint32_t s = slot(16 * 1024 + 32), b = sbase(s), fb = su32(&S.full[s]);
          tma3d(b, a.tm + TM_WQKV, 0, 16 * c, 8 * kc, fb, pol);
          bulk(b + L_D_S, a.wqkv_s + (size_t)((16 * c) / 32) * 160 + 32 * kc, 32, fb, pol);
          ++it;
        }
#define C1M_HOOK 1
#include C1M_HOOKS
#undef C1M_HOOK
      // A5: q_b tiles 4c .. 4c + 3 (rows 64c .., 64 x 1280): 5 chunks of 256 K (one 64-row box + the 2 scale rows)
      if (c < NWK)
        for (int kc = 0; kc < 5; ++kc) {
          const uint32_t s = slot(64 * 256 + 96), b = sbase(s), fb = su32(&S.full[s]);
          tma3d(b, a.tm + TM_WQB, 0, 64 * c, 2 * kc, fb, pol);
          bulk(b + L_QB_S, a.wqb_s + (size_t)(2 * c) * 48, 96, fb, pol);
          ++it;
        }
    }
    // F1 wo_a
    if (c < 128 && B16(a, woaT16, 1)) {   // the engine's bf16 wo_a^T: 8 chunks of 512 K, each two boxes {16 columns, 256 K}
      const int gi = c >> 6, tr = (c & 63) * 16;
      for (int kc = 0; kc < 8; ++kc) {
        const uint32_t s = slot(16 * 512 * 2), b = sbase(s), fb = su32(&S.full[s]);
        tma3d(b, a.tm + TM_WOA16, tr, 512 * kc, 2 * rank + gi, fb, pol);
        tma3d(b + 8192, a.tm + TM_WOA16, tr, 512 * kc + 256, 2 * rank + gi, fb, pol);
        ++it;
      }
    } else if (c < 128) {
      const int gi = c >> 6, tr = (c & 63) * 16;
      for (int kc = 0; kc < 4; ++kc) {
        const uint32_t s = slot(16 * 1024 + 32), b = sbase(s), fb = su32(&S.full[s]);
        tma3d(b, a.tm + TM_WOA, 0, 1024 * gi + tr, 8 * kc, fb, pol);
        bulk(b + L_D_S, a.woa_s + (size_t)((1024 * gi + tr) / 32) * 128 + 32 * kc, 32, fb, pol);
        ++it;
      }
    }
    // F2 wo_b
    for (int ti = 0; ti < nt && B16(a, wob16, 2); ++ti)   // the engine's bf16 wo_b: 4 chunks of 512 K of this rank's K quarter
      for (int kc = 0; kc < 4; ++kc) {
        const uint32_t s = slot(16 * 512 * 2), b = sbase(s), fb = su32(&S.full[s]);
        tma3d(b, a.tm + TM_WOB16, 0, 16 * (ftb + ti), 32 * rank + 8 * kc, fb, pol);
        ++it;
      }
    for (int ti = 0; ti < nt && !B16(a, wob16, 2); ++ti)
      for (int kc = 0; kc < 2; ++kc) {
        const int r0 = 16 * (ftb + ti);
        const uint32_t s = slot(16 * 1024 + 32), b = sbase(s), fb = su32(&S.full[s]);
        tma3d(b, a.tm + TM_WOB, 0, r0, a.wob_k0 + 8 * kc, fb, pol);
        bulk(b + L_D_S, a.wob_s + (size_t)(r0 / 32) * a.wob_sst + a.wob_sc0 + 32 * kc, 32, fb, pol);
        ++it;
      }
    // G: the fn slice [24][4][ncols] fp32
    {
      const uint32_t s = slot(96 * 48 * 4), b = sbase(s), fb = su32(&S.full[s]);
      bulk(b, a.fn + (size_t)c * 96 * 48, 96 * 48 * 4, fb, pol);
      ++it;
    }
    // H router: whole expert rows e = c, c + 132, c + 264 (bf16, 10 KB each), two rows per stage
    for (int e = c; c < NWK && e < NE; e += 2 * NWK) {   // rows e, e + 128 (and e + 256 alone)
      const int nrw = e + NWK < NE ? 2 : 1;
      const uint32_t s = slot(nrw * D * 2), b = sbase(s), fb = su32(&S.full[s]);
      for (int r = 0; r < nrw; ++r) bulk(b + r * D * 2, a.gate_w + (size_t)(e + r * NWK) * D, D * 2, fb, pol);
      ++it;
    }
    // the expert phase's weight stream (shared GEMM1 halves, routed GEMM1 items, GEMM2 routed / shared)
#include C1M_MOE_PRODUCE
    }   // layers
    return;
  }

  // ============================================================================================== consumers
  CState cs{0, 0ull, 0ull, call, ltim, tim0};
  for (int l = 0; l < nly; ++l) {
#ifdef C1M_HAS_PHASES
    if (al[l].ix || al[l].eg) {   // index / compressor / Engram layer: the separate function
      cs = cons_ix<FULL>(al, nly, l, cs);
      continue;
    }
#endif
    cons_layer<FULL, false>(al, nly, l, cs);
  }   // layers
  if (tid == 0) al[0].tick[c] = call + (unsigned)nly;
}

template <bool FULL>
constexpr size_t smem_of() { return (size_t)LayCfg<FULL>::NS * STAGE + NRM * IMS + LayCfg<FULL>::ATTX + sizeof(Smem); }
static_assert(smem_of<false>() <= 227 * 1024 && smem_of<true>() <= 227 * 1024, "SMEM");
}  // namespace c1m

// ------------------------------------------------------------------------------------------------ host
using namespace c1m;
static const uint8_t* u8(const torch::Tensor& t) { return (const uint8_t*)t.data_ptr(); }
template <typename T>
static T* P(const torch::Tensor& t) {
  return (T*)t.data_ptr();
}
#include <atomic>
static std::atomic<unsigned long long> g_attr[2];   // devices whose k_layer<FULL> SMEM attribute is set (C ABI: atomic)
static PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&fn, 12000, cudaEnableDefault, &q) ==
                    cudaSuccess && fn, "cuTensorMapEncodeTiled not available");
  }
  return fn;
}
// rank-r tensor map; dims / box in elements (dim 0 innermost), strides in bytes for dims 1 .. r-1
static void enc(CUtensorMap* m, CUtensorMapDataType ty, const void* p, std::initializer_list<uint64_t> dims,
                std::initializer_list<uint64_t> strides, std::initializer_list<uint32_t> box, bool swz) {
  cuuint64_t d[5], st[4];
  cuuint32_t b[5], es[5] = {1, 1, 1, 1, 1};
  int r = 0;
  for (auto x : dims) d[r++] = x;
  int k = 0;
  for (auto x : strides) st[k++] = x;
  int q = 0;
  for (auto x : box) b[q++] = x;
  TORCH_CHECK(k == r - 1 && q == r, "tensor map rank");
  const CUresult res = encode_fn()(m, ty, r, const_cast<void*>(p), d, st, b, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                   swz ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)res);
}
// one layer's tensor maps for one GPU -> CPU uint8 [TM_N * 128] (copy it to the GPU, 64-B aligned). Tensors (device):
// woa [2048][4096], wob [5120][8192], wqkv [1792][5120] or empty, wqb [8192][1280] or empty, sw13 [1152][5120],
// w13 [384][1152][2560], s13 [384][1152][160], fn [24][20480] fp32, afn (or empty: fn)
// maps3 = maps2 + the engine's bf16 shared expert (sw13_16 [1152][5120], sw2_16 [5120][576] or empty; sw13 may then be empty)
torch::Tensor maps3(torch::Tensor woa, torch::Tensor wob, torch::Tensor wqkv, torch::Tensor wqb, torch::Tensor sw13,
                    torch::Tensor w13, torch::Tensor s13, torch::Tensor fn, torch::Tensor afn, torch::Tensor wqkv16,
                    torch::Tensor woaT16, torch::Tensor wob16, torch::Tensor sw13_16, torch::Tensor sw2_16) {
  const at::cuda::CUDAGuard guard(woa.device());
  static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap");
  auto out = torch::zeros({TM_N * 128}, torch::kUInt8);
  CUtensorMap* m = (CUtensorMap*)out.data_ptr();
  const auto U8 = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  if (woa.numel()) enc(m + TM_WOA, U8, woa.data_ptr(), {128, 2048, 32}, {4096, 128}, {128, 16, 8}, true);
  if (wob.numel()) {
    TORCH_CHECK(wob.dim() == 2 && wob.size(0) == 5120 && (wob.size(1) == 8192 || wob.size(1) == 2048), "wo_b: [5120][8192] or a K quarter");
    enc(m + TM_WOB, U8, wob.data_ptr(), {128, 5120, (cuuint64_t)(wob.size(1) / 128)}, {(cuuint64_t)wob.size(1), 128}, {128, 16, 8}, true);
  }
  const auto U16 = CU_TENSOR_MAP_DATA_TYPE_UINT16;   // the engine's bf16 tensors (boxes of 64 x 2 B = 128 B rows)
  if (wqkv16.numel()) {
    TORCH_CHECK(wqkv16.dim() == 2 && wqkv16.size(0) == 1792 && wqkv16.size(1) == 5120 && wqkv16.is_contiguous(), "wqkv16");
    enc(m + TM_WQKV16, U16, wqkv16.data_ptr(), {64, 1792, 80}, {10240, 128}, {64, 16, 8}, true);
  }
  if (woaT16.numel()) {
    TORCH_CHECK(woaT16.dim() == 3 && woaT16.size(0) == 8 && woaT16.size(1) == 4096 && woaT16.size(2) == 1024 && woaT16.is_contiguous(), "woaT16");
    enc(m + TM_WOA16, U16, woaT16.data_ptr(), {1024, 4096, 8}, {2048, 4096ull * 2048}, {16, 256, 1}, false);
  }
  if (wob16.numel()) {
    TORCH_CHECK(wob16.dim() == 2 && wob16.size(0) == 5120 && wob16.size(1) == 8192 && wob16.is_contiguous(), "wob16");
    enc(m + TM_WOB16, U16, wob16.data_ptr(), {64, 5120, 128}, {16384, 128}, {64, 16, 8}, true);
  }
  if (wqkv.numel()) enc(m + TM_WQKV, U8, wqkv.data_ptr(), {128, 1792, 40}, {5120, 128}, {128, 16, 8}, true);
  if (wqb.numel()) enc(m + TM_WQB, U8, wqb.data_ptr(), {128, 8192, 10}, {1280, 128}, {128, 64, 2}, true);
  if (sw13.numel()) enc(m + TM_SW13, U8, sw13.data_ptr(), {128, 576, 40, 2}, {5120, 128, 576ull * 5120}, {128, 16, 4, 2}, true);
  if (sw13_16.numel()) {   // the engine's bf16 shared w1 | w3 rows of this rank: [2 halves][4 k-blocks of 64][16 rows] boxes
    TORCH_CHECK(sw13_16.dim() == 2 && sw13_16.size(0) == 1152 && sw13_16.size(1) == 5120 && sw13_16.is_contiguous() &&
                sw13_16.scalar_type() == torch::kBFloat16, "sw13_16");
    enc(m + TM_SW13_16, U16, sw13_16.data_ptr(), {64, 576, 80, 2}, {10240, 128, 576ull * 10240}, {64, 16, 4, 2}, true);
  }
  if (sw2_16.numel()) {    // the engine's bf16 shared w2 (K = this rank's 576): [5 k-blocks of 64][16 rows] boxes
    TORCH_CHECK(sw2_16.dim() == 2 && sw2_16.size(0) == 5120 && sw2_16.size(1) == 576 && sw2_16.is_contiguous() &&
                sw2_16.scalar_type() == torch::kBFloat16, "sw2_16");
    enc(m + TM_SW2_16, U16, sw2_16.data_ptr(), {64, 5120, 9}, {1152, 128}, {64, 16, 5}, true);
  }
  enc(m + TM_W13, U8, w13.data_ptr(), {128, 576, 20, 768}, {2560, 128, 576ull * 2560}, {128, 16, 4, 2}, true);
  enc(m + TM_S13, U8, s13.data_ptr(), {160, 576, 768}, {160, 576ull * 160}, {32, 16, 2}, false);
  (void)fn, (void)afn;   // (the mHC fn slices are per-CTA contiguous blocks: plain bulk copies, no map)
  return out;
}
torch::Tensor maps(torch::Tensor woa, torch::Tensor wob, torch::Tensor wqkv, torch::Tensor wqb, torch::Tensor sw13,
                   torch::Tensor w13, torch::Tensor s13, torch::Tensor fn, torch::Tensor afn) {
  const auto e = torch::empty({0}, torch::kUInt8);
  return maps3(woa, wob, wqkv, wqb, sw13, w13, s13, fn, afn, e, e, e, e, e);
}
torch::Tensor maps2(torch::Tensor woa, torch::Tensor wob, torch::Tensor wqkv, torch::Tensor wqb, torch::Tensor sw13,
                    torch::Tensor w13, torch::Tensor s13, torch::Tensor fn, torch::Tensor afn, torch::Tensor wqkv16,
                    torch::Tensor woaT16, torch::Tensor wob16) {
  const auto e = torch::empty({0}, torch::kUInt8);
  return maps3(woa, wob, wqkv, wqb, sw13, w13, s13, fn, afn, wqkv16, woaT16, wob16, e, e);
}
// a packed layer's Args (CPU bytes) with the 4 ranks' mode-3 staging buffers (device pointers as ints, [stg_bytes()] each)
torch::Tensor patch_stg(torch::Tensor packed, std::vector<int64_t> stg) {
  TORCH_CHECK(packed.device().is_cpu() && packed.numel() == (int64_t)sizeof(Args) && stg.size() == W, "patch_stg");
  torch::Tensor o = packed.clone();
  Args* a = (Args*)o.data_ptr();
  for (int s2 = 0; s2 < W; ++s2) a->stg[s2] = (uint8_t*)(uintptr_t)stg[s2];
  return o;
}
int64_t stg_bytes() { return (int64_t)STG_BYTES; }
// a packed layer's Args (CPU bytes) reading the engine's bf16 q_a | wkv / wo_a^T / wo_b (device pointers as ints, 0 = the
// fp8 copy); the tensor maps must come from maps2 with the same tensors
torch::Tensor patch_bf16(torch::Tensor packed, int64_t wqkv16, int64_t woaT16, int64_t wob16) {
  TORCH_CHECK(packed.device().is_cpu() && packed.numel() == (int64_t)sizeof(Args), "patch_bf16");
  torch::Tensor o = packed.clone();
  Args* a = (Args*)o.data_ptr();
  a->wqkv16 = (const bf16*)(uintptr_t)wqkv16, a->woaT16 = (const bf16*)(uintptr_t)woaT16, a->wob16 = (const bf16*)(uintptr_t)wob16;
#ifdef C1M_FIX_B16
  TORCH_CHECK((wqkv16 != 0) == (((C1M_FIX_B16) & 1) != 0) && (woaT16 != 0) == (((C1M_FIX_B16) & 2) != 0) &&
              (wob16 != 0) == (((C1M_FIX_B16) & 4) != 0), "this build fixes the bf16 set to bits ", C1M_FIX_B16);
#endif
  return o;
}
// a packed layer's Args (CPU bytes) with the phase extensions of an index / Engram layer (device pointers as ints, 0 = none):
// ix / eg (the phases' own structs), aout ([NR][D] bf16: A2's normalized attention input), push_topk (mode 3: the push
// warp re-pushes this layer's top-k records after hook 5)
torch::Tensor patch_ext(torch::Tensor packed, int64_t ix, int64_t eg, int64_t aout, int64_t push_topk) {
  TORCH_CHECK(packed.device().is_cpu() && packed.numel() == (int64_t)sizeof(Args), "patch_ext");
  torch::Tensor o = packed.clone();
  Args* a = (Args*)o.data_ptr();
  a->ix = (const void*)(uintptr_t)ix, a->eg = (const void*)(uintptr_t)eg, a->aout = (bf16*)(uintptr_t)aout;
  a->push_topk = (int)push_topk;
  return o;
}
// a packed layer's Args (CPU bytes) reading the engine's bf16 shared expert (device pointers as ints, 0 = the fp8 copies);
// the tensor maps must come from maps2 with the same tensors
torch::Tensor patch_sh16(torch::Tensor packed, int64_t sw13_16, int64_t sw2_16) {
  TORCH_CHECK(packed.device().is_cpu() && packed.numel() == (int64_t)sizeof(Args), "patch_sh16");
  torch::Tensor o = packed.clone();
  Args* a = (Args*)o.data_ptr();
  a->sw13_16 = (const bf16*)(uintptr_t)sw13_16, a->sw2_16 = (const bf16*)(uintptr_t)sw2_16;
#ifdef C1M_FIX_B16
  TORCH_CHECK((sw13_16 != 0) == (((C1M_FIX_B16) & 8) != 0) && (sw2_16 != 0) == (((C1M_FIX_B16) & 8) != 0),
              "this build fixes the bf16 set to bits ", C1M_FIX_B16);
#endif
  return o;
}

static void fill_post(Args& a, int64_t rank, int64_t NR, std::vector<torch::Tensor>& in, std::vector<torch::Tensor>& wt,
                      std::vector<torch::Tensor>& out, std::vector<torch::Tensor>& scr, std::vector<torch::Tensor>& ar1,
                      std::vector<torch::Tensor>& ar2, torch::Tensor& tim, double norm_eps, double hc_eps, int64_t hc_iters,
                      double route_scale) {
  TORCH_CHECK(NR >= 1 && NR <= NRM, "NR");
  TORCH_CHECK(in.size() == 5 && wt.size() == 18 && out.size() == 11 && scr.size() == 12 && ar1.size() == 4 && ar2.size() == 4,
              "post: argument lists (scr: ... cnt tick tm)");
  memset(&a, 0, sizeof(a));
  a.rank = (int)rank, a.NR = (int)NR, a.mode = 0;
  a.og = P<bf16>(in[0]), a.X = P<bf16>(in[1]), a.apost = P<float>(in[2]), a.acomb = P<float>(in[3]), a.apre = P<float>(in[4]);
  a.woa = u8(wt[0]), a.woa_s = u8(wt[1]), a.wob = u8(wt[2]), a.wob_s = u8(wt[3]), a.fn = P<float>(wt[4]);
  {
    const bool q = wt[2].dim() == 2 && wt[2].size(1) == 2048;   // this rank's K quarter of wo_b (scales [160][64])
    TORCH_CHECK(wt[2].numel() == 0 || (q ? wt[3].numel() == 160 * 64 : wt[3].numel() == 160 * 256), "wo_b scales");
    a.wob_k0 = q ? 0 : 16 * (int)rank, a.wob_sst = q ? 64 : 256, a.wob_sc0 = q ? 0 : 64 * (int)rank;
  }
  a.fscale = P<float>(wt[5]), a.fbase = P<float>(wt[6]), a.nw = P<float>(wt[7]), a.gate_w = P<bf16>(wt[8]);
  a.gate_b = P<float>(wt[9]), a.sw13 = u8(wt[10]), a.sw13_s = u8(wt[11]), a.sw2 = u8(wt[12]), a.sw2_s = u8(wt[13]);
  a.w13 = u8(wt[14]), a.s13 = u8(wt[15]), a.w2 = u8(wt[16]), a.s2 = u8(wt[17]);
  a.norm_eps = (float)norm_eps, a.hc_eps = (float)hc_eps, a.hc_iters = (int)hc_iters, a.route_scale = (float)route_scale;
  a.y = P<bf16>(out[0]), a.xm2 = P<bf16>(out[1]), a.fout = P<bf16>(out[2]), a.ptok = P<int>(out[3]), a.fpre = P<float>(out[4]);
  a.fpost = P<float>(out[5]), a.fcomb = P<float>(out[6]), a.logits = P<float>(out[7]), a.ids = P<int>(out[8]);
  a.wts = P<float>(out[9]), a.moe_out = P<bf16>(out[10]);
  a.z = P<bf16>(scr[0]), a.inb = P<bf16>(scr[1]), a.ssp = P<float>(scr[2]), a.mixp = P<float>(scr[3]), a.rtp = P<float>(scr[4]);
  a.shp = P<float>(scr[5]), a.hsh = P<bf16>(scr[6]), a.hmax = P<float>(scr[7]), a.himg = P<__half>(scr[8]);
  a.cnt = P<unsigned>(scr[9]), a.tick = P<unsigned>(scr[10]);
  TORCH_CHECK(scr[11].is_cuda() && scr[11].numel() >= TM_N * 128 && ((uintptr_t)scr[11].data_ptr() & 63) == 0, "tensor maps");
  a.tm = (const CUtensorMap*)scr[11].data_ptr();
  for (int s = 0; s < W; ++s) a.ar1[s] = P<float>(ar1[s]), a.ar2[s] = P<bf16>(ar2[s]);
  a.tim = tim.numel() ? (unsigned long long*)tim.data_ptr() : nullptr;
}
// ---- C ABI (../c1d_cabi.h; fpr_cabi.h rules): the launch of pybind run, called by it and by the C++ driver
#include "../c1d_cabi.h"
#include "../../../engine/mixed_cxx/fpr_cabi_rec.h"
FPR_CABI_RECORD_DECL(c1d)
template <bool FULL>
static int launch_core(const Args* al, int nly, int dev, cudaStream_t st) {
  constexpr size_t SM = smem_of<FULL>();
  if (!((g_attr[FULL].load(std::memory_order_acquire) >> dev) & 1)) {   // per device, once (idempotent if threads race)
    if (cudaFuncSetAttribute(k_layer<FULL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SM) != cudaSuccess) return FPR_E_CUDA;
    g_attr[FULL].fetch_or(1ull << dev, std::memory_order_acq_rel);
  }
  k_layer<FULL><<<NCTA, THR, SM, st>>>(al, nly);
  return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}
static int run_core(const FprC1dRunArgs* a) {   // "c1d.run"
  try {
    if (!a->packed || ((uintptr_t)a->packed & 15) || a->nly <= 0 || a->c.device < 0 || a->c.device >= 64) return FPR_E_ARG;
    return a->full ? launch_core<true>((const Args*)a->packed, a->nly, a->c.device, a->c.stream)
                   : launch_core<false>((const Args*)a->packed, a->nly, a->c.device, a->c.stream);
  } catch (...) {
    return FPR_E_CUDA;
  }
}
static torch::Tensor to_bytes(const Args& a) {
  auto t = torch::empty({(int64_t)sizeof(Args)}, torch::kUInt8);
  memcpy(t.data_ptr(), &a, sizeof(Args));
  return t;
}
// one launch of nly layers: packed = the layers' Args (pack_ffn / pack_layer, concatenated) on the GPU
void run(torch::Tensor packed, int64_t nly, bool full) {
  const at::cuda::CUDAGuard guard(packed.device());
  TORCH_CHECK(packed.is_cuda() && packed.numel() == nly * (int64_t)sizeof(Args) && ((uintptr_t)packed.data_ptr() & 15) == 0,
              "run: packed Args");
  FprC1dRunArgs ca{{at::cuda::getCurrentCUDAStream(), (int)packed.get_device(), 0}, packed.data_ptr(), (int32_t)nly, full ? 1 : 0};
  FPR_CABI_RECORD("c1d.run", ca);
  const int rc = run_core(&ca);
  TORCH_CHECK(rc != FPR_E_CUDA || cudaGetLastError() == cudaSuccess, "smem attr / k_layer launch");
  TORCH_CHECK(rc == FPR_OK, "k_layer launch");
}
extern "C" {
static int c_run(const void* a) { return run_core((const FprC1dRunArgs*)a); }
static const FprFnEntry k_c1d_fns[] = {{"c1d.run", c_run, nullptr, (int64_t)sizeof(FprC1dRunArgs)}};
static const FprExtTable k_c1d_table = {FPR_CABI_VERSION, "c1d", "", 1, k_c1d_fns};
__attribute__((visibility("default"))) const FprExtTable* fpr_c1d_table(void) { return &k_c1d_table; }
}
int64_t args_bytes() { return (int64_t)sizeof(Args); }
// the layer phases (index / compressor / Engram) are compiled in (the hooks file defines C1M_HAS_PHASES): C1M_IX=1 runs
bool has_phases() {
#ifdef C1M_HAS_PHASES
  return true;
#else
  return false;
#endif
}
// the compile-time deployment config: (attention mode or -1, bf16 bits or -1, timeline compiled in)
std::vector<int64_t> fixed_config() {
#ifdef C1M_FIX_AMODE
  const int64_t am = C1M_FIX_AMODE;
#else
  const int64_t am = -1;
#endif
#ifdef C1M_FIX_B16
  const int64_t b = C1M_FIX_B16;
#else
  const int64_t b = -1;
#endif
#ifdef C1M_NOTIM
  return {am, b, 0};
#else
  return {am, b, 1};
#endif
}

// in: og X apost acomb apre; wt: woa woa_s wob wob_s fn fscale fbase nw gate_w gate_b sw13 sw13_s sw2 sw2_s w13 s13 w2 s2;
// out: y xm2 f ptok fpre fpost fcomb logits ids wts moe_out; scr: z inb ssp mixp rtp shp hsh hmax himg cnt tick;
// ar1 / ar2: the 4 ranks' slot tensors (peer pointers); tim: [132 * 24] int64 or empty
torch::Tensor pack_ffn(int64_t rank, int64_t NR, std::vector<torch::Tensor> in, std::vector<torch::Tensor> wt, std::vector<torch::Tensor> out,
         std::vector<torch::Tensor> scr, std::vector<torch::Tensor> ar1, std::vector<torch::Tensor> ar2, torch::Tensor tim,
         double norm_eps, double hc_eps, int64_t hc_iters, double route_scale, int64_t mode) {
  const at::cuda::CUDAGuard guard(in[0].device());
  Args a{};
  fill_post(a, rank, NR, in, wt, out, scr, ar1, ar2, tim, norm_eps, hc_eps, hc_iters, route_scale);
  a.mode = (int)mode;
  return to_bytes(a);
}
// the whole layer: the post lists as ffn (in[0] og = fo[9] o, in[1] X = fo[0] Xo, in[2..4] = fo[3], fo[4], fo[2]: the
// caller passes those tensors), plus
//  fin: xin rsin pin cin lin pos own si sl ti tl     fw: afn ascale abase anw wqkv wqkv_s q_norm kv_norm wqb wqb_s sink cos_sin
//  fo:  Xo ain apre apost acomb qakv qrout kvout q o  fscr: ssp_a mixp_a apart aml
//  kv:  win (this layer's window pages, base pointer + page stride in bytes), comp (or empty); win_bs / comp_bs records per page
torch::Tensor pack_layer(int64_t rank, int64_t NR, std::vector<torch::Tensor> in, std::vector<torch::Tensor> wt, std::vector<torch::Tensor> out,
           std::vector<torch::Tensor> scr, std::vector<torch::Tensor> ar1, std::vector<torch::Tensor> ar2, torch::Tensor tim,
           double norm_eps, double hc_eps, int64_t hc_iters, double route_scale, std::vector<torch::Tensor> fin,
           std::vector<torch::Tensor> fw, std::vector<torch::Tensor> fo, std::vector<torch::Tensor> fscr, torch::Tensor win,
           int64_t win_stride, int64_t win_bs, torch::Tensor comp, int64_t comp_stride, int64_t comp_bs, double attn_eps,
           double softmax_scale, int64_t do_insert, int64_t amode, int64_t owner, std::vector<torch::Tensor> qall,
           std::vector<torch::Tensor> ogl) {
  const at::cuda::CUDAGuard guard(in[0].device());
  TORCH_CHECK(fin.size() == 11 && fw.size() == 12 && fo.size() == 10 && fscr.size() == 4, "layer: argument lists");
  Args a{};
  fill_post(a, rank, NR, in, wt, out, scr, ar1, ar2, tim, norm_eps, hc_eps, hc_iters, route_scale);
  a.xin = P<bf16>(fin[0]), a.rsin = P<bf16>(fin[1]), a.pin = P<float>(fin[2]), a.cin = P<float>(fin[3]), a.lin = P<float>(fin[4]);
  a.pos = (const long long*)fin[5].data_ptr<int64_t>(), a.own = (const long long*)fin[6].data_ptr<int64_t>();
  a.si = fin[7].data_ptr<int>(), a.sl = fin[8].data_ptr<int>(), a.ti = fin[9].data_ptr<int>(), a.tl = fin[10].data_ptr<int>();
  a.afn = P<float>(fw[0]), a.ascale = P<float>(fw[1]), a.abase = P<float>(fw[2]), a.anw = P<float>(fw[3]);
  a.wqkv = u8(fw[4]), a.wqkv_s = u8(fw[5]), a.q_norm = P<float>(fw[6]), a.kv_norm = P<float>(fw[7]);
  a.wqb = u8(fw[8]), a.wqb_s = u8(fw[9]), a.sink = P<float>(fw[10]), a.cos_sin = P<float>(fw[11]);
  a.Xo = P<bf16>(fo[0]), a.ain = P<bf16>(fo[1]), a.apre_o = P<float>(fo[2]), a.apost_o = P<float>(fo[3]);
  a.acomb_o = P<float>(fo[4]), a.qakv = P<bf16>(fo[5]), a.qrout = P<bf16>(fo[6]), a.kvout = P<bf16>(fo[7]);
  a.q = P<bf16>(fo[8]), a.o = P<bf16>(fo[9]);
  a.ssp_a = P<float>(fscr[0]), a.mixp_a = P<float>(fscr[1]), a.apart = P<float>(fscr[2]), a.aml = P<float>(fscr[3]);
  auto sh = [](int64_t bs) { int s = 0; while ((1LL << s) < bs) ++s; TORCH_CHECK((1LL << s) == bs, "page size"); return s; };
  a.win = (uint8_t*)win.data_ptr(), a.win_stride = win_stride, a.win_shift = sh(win_bs);
  a.comp = comp.numel() ? (const uint8_t*)comp.data_ptr() : nullptr, a.comp_stride = comp_stride, a.comp_shift = comp.numel() ? sh(comp_bs) : 0;
  a.attn_eps = (float)attn_eps, a.scale_log2 = (float)(softmax_scale * 1.4426950408889634), a.do_insert = (int)do_insert;
  a.amode = (int)amode, a.owner = (int)owner;
  TORCH_CHECK(amode == 0 || (qall.size() == 4 && ogl.size() == 4), "owner mode: qall / ogl of the 4 ranks");
#ifdef C1M_FIX_AMODE
  TORCH_CHECK(amode == C1M_FIX_AMODE, "this build fixes the attention mode to ", C1M_FIX_AMODE);
#endif
  for (int s2 = 0; s2 < W && amode == 1; ++s2) a.qall[s2] = P<bf16>(qall[s2]), a.ogl[s2] = P<bf16>(ogl[s2]);
  TORCH_CHECK(a.og == (const bf16*)a.o && a.X == (const bf16*)a.Xo, "layer: og / X must be fo[9] / fo[0]");
  a.si_rs = 128, a.ti_rs = 512;
  return to_bytes(a);
}
// engine integration: a packed layer's Args (CPU bytes) with the ready flags of the 4 ranks (device pointers, as ints; all
// 0 = inputs local) and the si / ti row strides
torch::Tensor patch_integ(torch::Tensor packed, std::vector<int64_t> rdy, int64_t si_rs, int64_t ti_rs) {
  TORCH_CHECK(packed.device().is_cpu() && packed.numel() == (int64_t)sizeof(Args) && rdy.size() == W, "patch_integ");
  torch::Tensor o = packed.clone();
  Args* a = (Args*)o.data_ptr();
  for (int s2 = 0; s2 < W; ++s2) a->rdy[s2] = (unsigned*)(uintptr_t)rdy[s2];
  a->si_rs = (int)si_rs, a->ti_rs = (int)ti_rs;
  return o;
}
int64_t smem_bytes(bool full) { return full ? (int64_t)smem_of<true>() : (int64_t)smem_of<false>(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_ffn", &pack_ffn);
  m.def("pack_layer", &pack_layer);
  m.def("patch_integ", &patch_integ);
  m.def("run", &run);
  m.def("args_bytes", &args_bytes);
  m.def("fixed_config", &fixed_config);
  m.def("has_phases", &has_phases);
  m.def("smem_bytes", &smem_bytes);
  m.def("maps", &maps);
  m.def("maps2", &maps2);
  m.def("maps3", &maps3);
  m.def("patch_bf16", &patch_bf16);
  m.def("patch_sh16", &patch_sh16);
  m.def("patch_ext", &patch_ext);
  m.def("patch_stg", &patch_stg);
  m.def("stg_bytes", &stg_bytes);
}
