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
// dense projection phases: each file holds one phase's
// producer (C1M_DPART 0) and consumer (C1M_DPART 1) halves; defaults = the code as it was, verbatim
// -DC1M_DOTS_WIN: the boundaries' mix dots in wait windows (A1 after the q_a tile, G after the router rows): measured a LOSS
// (m-68, ~+3 us per layer vs the default: the dots inside the slices, after the norm arrival); kept for reference
#if defined(C1M_DOTS_WIN) && defined(C1M_MX_OLD)
#error "C1M_DOTS_WIN needs the split arrivals (not C1M_MX_OLD)"
#endif
#ifndef C1M_DENSE_A3
#define C1M_DENSE_A3 "c1dense_a3.inc"
#endif
#ifndef C1M_DENSE_A5
#define C1M_DENSE_A5 "c1dense_a5.inc"
#endif
#ifndef C1M_DENSE_F1
#define C1M_DENSE_F1 "c1dense_f1.inc"
#endif
#ifndef C1M_DENSE_F2
#define C1M_DENSE_F2 "c1dense_f2.inc"
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
#define C1M_MOE_PRODUCE "c1moe_x2j_produce.inc"
                                                   // x2b + per-item scale bulk copies + non-blocking h polls; x2b:
                                                   // grabbed GEMM1 items, GEMM2 interleaved; c1moe_produce.inc = before)
#endif
#ifndef C1M_MOE_CONSUME
// CONTRACT with the producer (x2j): each grabbed item's 5 GEMM1 chunk stages are contiguous in the ring; never interleave two
// items' chunks; never abandon an item mid-way (the consumers reset their accumulators after chunk kc == 4 only)
#define C1M_MOE_CONSUME "c1moe_x2j_consume.inc"
#endif
#ifndef C1M_HOOKS
#define C1M_HOOKS "c1hooks.inc"               // index / compressor / Engram hook points (empty by default)
#endif
constexpr int D = 5120, HC = 4, NE = C1M_NE, TOPK = C1M_TOPK, IR = 576, W = 4, NRM = C1M_NRM, NCTA = 132;
constexpr int WIN = C1M_WIN, TKA = C1M_TKA, NEW = (NE + 31) / 32;   // NEW: 32-bit words / lanes per expert set
static_assert(NE % 32 == 0 && NE <= 1024 && TOPK >= 1 && TOPK <= 8 && NRM >= 4 && NRM <= 6 && WIN >= 1 && WIN <= 256, "shape knobs");
// -DC1M_NRM=6: up to 6 real rows. Rows 4, 5 on the service CTAs 128, 129 as a
// second pass (q_b keeps its 4 tiles per worker: NWK stays 128); a larger stage (the routed GEMM2 stage holds the h image of
// NRM rows) and a shallower ring (one 10 KB activation image row per row): STAGE 22528, NS 5. NRM = 4: the code as it was
// -DC1M_X16: 8 more consumer warps (10-17) that work in the expert phase only. Warps 0-7
// (all phases), 8 = producer, 9 = KV push, 10-17 = expert-only; THR = 576 (ptxas: <= 96 registers). The ring's empty
// mbarriers count NCWX = 16: the base warps' crel() arrives with count XRC = 2, a stage consumed by all 16 (between
// C1M_XENTER() and C1M_XEXIT() in an X16-aware expert fragment, C1M_XFRAG) with crel1() = count 1 per warp. xbar() =
// bar 3 over the 16 warps. Without C1M_XFRAG the extra warps exit at once (an X16 build then measures the 96-register cap)
#if defined(C1M_X16) && !defined(C1M_XN)
#define C1M_XN 8
#endif
#ifdef C1M_XN
constexpr int NCW = C1M_NCW, CTH = NCW * 32, NCWX = NCW + C1M_XN, XTH = NCWX * 32, THR = CTH + 64 + C1M_XN * 32;
#else
constexpr int NCW = C1M_NCW, CTH = NCW * 32, NCWX = NCW, XTH = CTH, THR = CTH + 64;   // consumer warps, producer, KV push
#endif
constexpr unsigned XRC = (unsigned)(NCWX / NCW);   // the base warps' arrival count per ring stage (C1M_XN: see xrc_of)
// (c1mega_xn) a base warp's arrival count on a stage the expert-only warps do not consume: the counts of warps 0..NCW-1
// sum to NCWX
__device__ __forceinline__ unsigned xrc_of(int w) {
  return 1u + (unsigned)((NCWX - NCW) / NCW) + (w < (NCWX - NCW) % NCW ? 1u : 0u);
}
#ifndef C1M_STAGEF   // (tests: -DC1M_STAGEF=22528 = the NRM 6 ring's stage at NRM 4)
#define C1M_STAGEF (C1M_NRM <= 4 ? 20480 : C1M_NRM == 5 ? 21504 : 22528)
#endif
constexpr int NSMAX = 8, STAGE = C1M_STAGEF;   // multiple of 1024 (128B-swizzled TMA boxes)
static_assert(STAGE % 1024 == 0 && STAGE >= 20480, "stage");
constexpr int MAXX = NRM * TOPK;
constexpr int RT_NP = 240, RT_KC = 512, RT_NKC = D / RT_KC;  // router pieces: 24 tiles of 16 experts x 10 K chunks
constexpr int SH_NU = 36, SH_NH = 72;                        // shared GEMM1: 36 tiles of 16 intermediates x 2 K halves
constexpr int IMS = D * 2 + 64;                              // activation image row stride in SMEM (bytes)
constexpr unsigned SENT = 0x80000000u;
constexpr int NTS = 48, NTIM = 4 * NTS;   // per CTA: stamps, data-wait ns, sync-wait ns, SM clock (clock64)
constexpr int KVR = WIN + TKA, STG_R = 592;   // mode-3 staging: records of 584 B at a 592-B (16-B aligned) stride, data then scales
constexpr int STG_V = NRM * KVR * STG_R, STG_P = (STG_V + NRM * KVR + 15) & ~15,
#ifdef C1M_PDD   // the deduplicated top-k map: [NRM][KVR] u16 = the slot holding (n, j)'s record (0xFFFF: none)
                 STG_M = STG_P + NRM * 8, STG_BYTES = STG_M + NRM * KVR * 2;
#else
                 STG_BYTES = STG_P + NRM * 8;
#endif   // per CTA: stamps [0, 32), data-wait ns [32, 64), sync-wait ns [64, 96) at each stamp
constexpr int AK = 32, ANCH = (KVR + AK - 1) / AK;  // FULL: attention key chunk, chunks per row
// counters (unsigned [C_N], monotonic; targets scale with the CTA's call index)
enum { C_Z = 0, C_SS = 1, C_RT = 2, C_SH = 3, C_SSA = 4, C_MIXA = 5, C_QA = 6, C_QB = 7, C_SHT = 8, C_H = 48, C_O = C_H + MAXX,
       C_AT = C_O + 1, C_MIXF = C_AT + NRM, C_INS = C_MIXF + 1, C_MXA = C_INS + 1, C_MXG = 47, C_N = C_MXA + 1 };
static_assert(NRM != 4 || (C_O == 72 && C_AT == 73 && C_MIXF == 77 && C_INS == 78 && C_MXA == 79 && C_N == 80), "counters");   // C_MXA / C_MXG: the boundary slices'
// mix partials (A1 / G), arrived after the norm partials' C_SSA / C_SS   // C_INS: the rows' window inserts done (NR per layer, every rank)
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
       TM_WOB16, TM_SW13_16, TM_SW2_16, TM_WQB8S, TM_GATE16, TM_WOA8, TM_WOA8S, TM_WOB8, TM_WOB8S, TM_N };   // (srv8x: + the packed wo_a / wo_b)   // TM_GATE16: C1M_RKS's gate column tiles
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
  // q_b from the engine's own W8A16 packed wq_b (patch_qb8; null = c1mega's fp8 copy wqb / wqb_s): the packed tensor
  // [10 k-chunks of 128][128 tiles of 256 rows][32,896 B] (attn_dense/w8a16.py repack: per lane 16 B = the m16n8k16 A
  // fragment of rows g, g + 8, k 2t.. of one k32 block, plus 16 B of ue8m0 scales per half tile); read by TM_WQB /
  // TM_WQB8S (maps3 with the [10][32][32896] view of this rank's 32 tiles). The same weights and scales: no copy (0.33 GB)
  const void* wqb8;
  // wo_a / wo_b from the engine's OWN W8A16 packed fp8 (AW["w8"]["wo_a"] /
  // ["wo_b"], the same w8a16 repack as wq_b), this rank's views (wo_a pk[:, 8r:8r+8] = groups 2r, 2r+1; wo_b pk[16r:16r+16] =
  // its K quarter) by TM_WOA8 / TM_WOB8 (+ the scale maps, map_wo8).
  // B16 bits 5 / 6
  const void* woa8;
  const void* wob8;
  int* idd;     // (tests) every layer's top-6 ids of the launch [nly][NRM][TOPK], CTA 0 (patch_idd; null = off)
  float* rkp;   // (C1M_RKS) the router's partial logits [2 parity][NRM][NE][NCTA] fp32, RTL_SENT-filled (patch_rks)
};
// deployment builds: -DC1M_FIX_AMODE=<mode> and -DC1M_FIX_B16=<bits: 1 q_a|wkv, 2 wo_a, 4 wo_b, 8 the
// shared expert (bf16 from the engine), 16 q_b from the engine's W8A16 packed fp8 wq_b (B16(a, wqb8, 4)), (srv8x) 32 / 64 wo_a /
// wo_b from the engine's W8A16 packed fp8 (B16(a, woa8, 5) / B16(a, wob8, 6))> fix the attention mode and the bf16 / fp8 choice per matrix at compile time (the other paths compiled out,
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
// C1M_FN_FRAG: the mHC fn matrices read IN PLACE from the engine's mhcf prep_f fragments
// (lw["fa_f"] / lw["ff_f"]: [160 ch][16 s][3 nt][32 l][2 q] fp32 = the exact fn values permuted: column k_local = hc * 32 + cw
// of chunk ch (32 columns), output n: s = k / 8, l = (n % 8) * 4 + k % 4, q = (k % 8) / 4) instead of c1mega's per-CTA
// re-layout copies (-0.20 GB per GPU). A CTA's 32 columns are one chunk; its 48 a full chunk + a half chunk (even CTAs: the
// next chunk's lower half; odd: the previous chunk's upper half). Stage: [full 3072 | half 1536] floats (even) or
// [half 1536 | full 3072] (odd); the half region keeps the s blocks 4h, 4h + 1 (lower) or 4h + 2, 4h + 3 (upper) as s' = 2h,
// 2h + 1. The consumers read fn_off(): the same values in the same order (bitwise vs the copies), another SMEM address.
#if defined(C1M_FN_FRAG2)
// C1M_FN_FRAG2: every 16-column tile t of the CTA (global tile ftb + t = chunk ch = tile >> 1, half = tile & 1) as ONE 5-D TMA
// of its half chunk (TM_FN32 / TM_AFN32: {192 floats, s2, half, h, ch}, box {192, 2, 1, 4, 1}) into region t of 1536 floats
// = [h = hc][s2][nt][l][q]: the address is a per-lane constant + immediates: t * 1536 + hc * 384 + s2 * 192 + nt * 64 +
// (n % 8) * 8 + (j % 4) * 2 + (j % 8) / 4 with t = j / 16, s2 = (j / 8) % 2 (col0 is a multiple of 16)
__device__ __forceinline__ unsigned fn_bytes(int c) { return c < 56 ? 3u * 6144u : 2u * 6144u; }
__device__ __forceinline__ int fn2_off(int m, int cc, int j) {
  return (j >> 4) * 1536 + cc * 384 + ((j >> 3) & 1) * 192 + (m >> 3) * 64 + (m & 7) * 8 + (j & 3) * 2 + ((j >> 2) & 1);
}
#define C1M_FNI(m_, cc_, j_) fn2_off((m_), (cc_), (j_))
#elif defined(C1M_FN_FRAG)
__device__ __forceinline__ unsigned fn_bytes(int c) { return c < 56 ? 4608u * 4u : 3072u * 4u; }
__device__ __forceinline__ int fn_off(int c, int m, int cc, int j) {
  const bool odd = c < 56 && (c & 1);
  const bool half = c < 56 && (odd ? j < 16 : j >= 32);
  const int cw = odd ? ((j + 16) & 31) : (j & 31);   // the column within its chunk
  const int r = cw & 7, l = (m & 7) * 4 + (r & 3), q = r >> 2, nt = m >> 3;
  const int base = half ? (odd ? 0 : 3072) : (odd ? 1536 : 0);
  const int sb = half ? 2 * cc + ((cw & 15) >> 3) : 4 * cc + (cw >> 3);
  return base + ((sb * 3 + nt) * 32 + l) * 2 + q;
}
#define C1M_FNI(m_, cc_, j_) fn_off(c, (m_), (cc_), (j_))
#else
#define C1M_FNI(m_, cc_, j_) ((((m_) * HC + (cc_)) * 48) + (j_))
#endif
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
__device__ __forceinline__ void tma5d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, int c2, int c3, int c4,
                                      uint32_t mbar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;" ::"r"(dst),
      "l"(tm), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4), "r"(mbar), "l"(pol)
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
__device__ __forceinline__ float ld_rlx_f(const float* p) {   // (C1M_RTL: Lamport-polled router logits)
  float v;
  asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_rlx_f(float* p, float v) {
  asm volatile("st.relaxed.gpu.global.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}
constexpr unsigned RTL_SENT = 0x7fc0deadu;
constexpr unsigned CML_SENT = 0x7fc0beefu;   // the combine partials' "not written yet" value (a NaN payload)   // (C1M_RTL) the logits' "not written yet" value: a NaN no logit can be
// bf16 data polled against 0xFFFF (a NaN the epilogues' cvt.rn never produces: it
// returns 0x7FFF for any NaN): parity buffers, the writer resets the other parity
constexpr unsigned short L16_SENT = 0xFFFFu;
__device__ __forceinline__ void st_rlx_h(bf16* p, bf16 v) {
  asm volatile("st.relaxed.gpu.global.b16 [%0], %1;" ::"l"(p), "h"(__bfloat16_as_ushort(v)) : "memory");
}
__device__ __forceinline__ uint4 ld_rlx_v4(const void* p) {
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_rlx_u32(const void* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ bool has_s16(unsigned u) { return (u & 0xffffu) == L16_SENT || (u >> 16) == L16_SENT; }
__device__ __forceinline__ bool has_s16(uint4 u) { return has_s16(u.x) || has_s16(u.y) || has_s16(u.z) || has_s16(u.w); }
__device__ __forceinline__ uint4 ld_poll_v4(const void* p) {   // a 16-B chunk of polled bf16 data, every half written
  uint4 v = ld_rlx_v4(p);
  while (has_s16(v)) v = ld_rlx_v4(p);
  return v;
}
__device__ __forceinline__ unsigned ld_poll_u32(const void* p) {
  unsigned v = ld_rlx_u32(p);
  while (has_s16(v)) v = ld_rlx_u32(p);
  return v;
}
__device__ __forceinline__ void red_rlx(unsigned* p, unsigned v) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_rlx(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void spin_ge_nf(const unsigned* p, unsigned target) {   // (C1M_QAG / C1M_ZG) relaxed, no fence:
  while ((int)(ld_rlx(p) - target) < 0) {                                         // the data itself is checked
  }
}
// C1M_LBM "lamport mix partials": the A1 / G slices' mix-dot partials are 8-B {value
// epoch = ncall} entries stored relaxed (no C_MXA / C_MXG arrival, so no release fence behind this SM's in-flight TMA
// loads) and polled by the 4 service CTAs' mixes (all 14 loads in flight, then each re-polled until its epoch is this
// layer's). The adapter doubles the two mix scratch buffers (scr[3], fscr[1]). BITWISE
#ifdef C1M_LBM
__device__ __forceinline__ void st_lb(float* p, float v, unsigned e) {   // entry at p (2 floats)
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"((unsigned long long)__float_as_uint(v) | ((unsigned long long)e << 32))
               : "memory");
}
__device__ __forceinline__ unsigned long long ld_lb(const float* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
#endif
#if defined(C1M_LBM) && (defined(C1M_DOTS_BR) || defined(C1M_MX_OLD) || defined(C1M_DOTS_WIN))
#error "C1M_LBM: the default (branch-free) mix dots only"
#endif
// relaxed polls + one acquire fence (C1M_SPIN_PW's waits: a lone warp polling next to computing warps)
__device__ __forceinline__ void spin_ge_rlx(const unsigned* p, unsigned target) {
  while ((int)(ld_rlx(p) - target) < 0) {
  }
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
// C1M_SPIN_RLX: an acquire load compiles to LDG.STRONG + CCTL.IVALL (the SM's whole L1
// invalidated per poll); polled relaxed instead, then ONE acquire fence once the condition holds (the same ordering)
#if defined(C1S_OGMC) && !defined(C1S_CLS)
#error "C1S_OGMC multicasts within a cluster: define C1S_CLS (2 at 132 CTAs)"
#endif
#ifdef C1S_CLS
// cluster launch (C1S_CLS CTAs per cluster) and the og multicast (C1S_OGMC)
static_assert(132 % C1S_CLS == 0 && C1S_CLS >= 2 && C1S_CLS <= 8, "C1S_CLS: a divisor of 132 (2, 4 / 6 if they fit)");
__device__ __forceinline__ unsigned cl_rank() {
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cl_sync_all() {   // every thread of every CTA of the cluster
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// one bulk copy global -> the same smem offset in every CTA of ctaMask, completing on each one's mbarrier at offset mbar
__device__ __forceinline__ void bulk_mc(uint32_t dst, const void* src, unsigned bytes, uint32_t mbar, unsigned short mask) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(mbar), "h"(mask)
               : "memory");
}
#endif
__device__ __forceinline__ void spin_ge(const unsigned* p, unsigned target) {
#ifdef C1M_SPIN_RLX
  while ((int)(ld_rlx(p) - target) < 0) {
  }
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
#else
  while ((int)(ld_acq(p) - target) < 0) {
  }
#endif
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
__device__ __forceinline__ void spin_ge_sys(const unsigned* p, unsigned target) {   // (sys scope; C1M_SPIN_RLX as spin_ge)
#ifdef C1M_SPIN_RLX
  unsigned v;
  do {
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  } while ((int)(v - target) < 0);
  asm volatile("fence.acq_rel.sys;" ::: "memory");
#else
  while ((int)(ld_acq_sys(p) - target) < 0) {
  }
#endif
}
__device__ __forceinline__ void red_rel_sys(unsigned* p, unsigned v) {
  asm volatile("red.release.sys.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, %0;" ::"n"(CTH) : "memory"); }
// named barrier 2 (consumer threads): one warp arrives (a buffer is free), the other consumer warps wait for it
__device__ __forceinline__ void nb2_arrive() { asm volatile("bar.arrive 2, %0;" ::"n"(CTH) : "memory"); }
__device__ __forceinline__ void nb2_sync() { asm volatile("bar.sync 2, %0;" ::"n"(CTH) : "memory"); }
__device__ __forceinline__ void xbar() { asm volatile("bar.sync 3, %0;" ::"n"(XTH) : "memory"); }   // (C1M_X16: warps 0-7 + 10-17)
__device__ __forceinline__ void mb_arrive_n(uint32_t a, unsigned n) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(n) : "memory");
}
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
#if defined(C1M_FN_FRAG2)
__device__ __forceinline__ void fn_produce2(uint32_t b, const CUtensorMap* tm, int c, uint32_t fb, uint64_t pol) {
  int ftb, nt;
  my_tiles(c, ftb, nt);
  for (int t = 0; t < nt; ++t) tma5d(b + t * 6144u, tm, 0, 0, (ftb + t) & 1, 0, (ftb + t) >> 1, fb, pol);
}
#elif defined(C1M_FN_FRAG)
// the CTA's fn columns from the engine's prep_f fragments into one stage (layout: fn_off): 1 bulk (32 columns) or 5 (48)
__device__ __forceinline__ void fn_produce(uint32_t b, const float* f, int c, uint32_t fb, uint64_t pol) {
  int ftb, nt;
  my_tiles(c, ftb, nt);
  const int col0 = 16 * ftb;
  if (c >= 56) {
    bulk(b, f + (size_t)(col0 / 32) * 3072, 3072 * 4, fb, pol);
    return;
  }
  const bool odd = c & 1;
  const int chf = odd ? (col0 + 16) / 32 : col0 / 32, chh = odd ? col0 / 32 : col0 / 32 + 1;   // full / half chunk
  const uint32_t bf = b + (odd ? 1536u * 4u : 0u), bh = b + (odd ? 0u : 3072u * 4u);
  bulk(bf, f + (size_t)chf * 3072, 3072 * 4, fb, pol);
#pragma unroll
  for (int h = 0; h < 4; ++h)   // s blocks 4h, 4h + 1 (lower half) or 4h + 2, 4h + 3 (upper): 2 x 192 floats each
    bulk(bh + h * 384u * 4u, f + (size_t)chh * 3072 + (4 * h + (odd ? 2 : 0)) * 192, 384 * 4, fb, pol);
}
#endif

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

#ifdef C1M_A8C
// Per k32 block: the same e4m3 -> f16 cvt and f16 MMAs as dense_chunk
// (W8A16, 16-bit activations: more precise than production's W8A8), acc += (A . B) * s128[kb / 4].
template <int NT16>
__device__ __forceinline__ void dense_chunk_s128(float (&acc)[NT16][4], uint32_t st, int TS, int nkb, uint32_t scl, int sstr,
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
      float sc;
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(sc) : "r"(scl + ti * sstr + 4 * (kb >> 2)));
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[ti][i] = fmaf(tmp[i], sc, acc[ti][i]);
    }
  }
}
#endif
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

// one 512-K chunk of a single 16-row tile with NCW = 8: this warp's 4 k16 steps
// kk = w, w + 8, w + 16, w + 24 with every load issued first, then the 4 MMAs in the same order (bitwise = dense16_chunk<1>)
__device__ __forceinline__ void dense16_c1u(float (&acc)[4], uint32_t st, uint32_t img, int kofs, int NR, int w, int lane) {
  static_assert(NCW == 8, "C1M_DU: 4 k16 steps per warp");
  const int g = lane >> 2, t = lane & 3, lr = (lane & 7) + 8 * ((lane >> 3) & 1), lc = lane >> 4;
  unsigned b[4][2], A[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int kk = w + 8 * j;
    b[j][0] = 0u, b[j][1] = 0u;
    if (g < NR) {
      const uint32_t ba = img + g * IMS + (kofs + 16 * kk) * 2 + 4 * t;
      b[j][0] = lds32(ba), b[j][1] = lds32(ba + 16);
    }
    ldm_x4(st + (kk >> 2) * 2048 + lr * 128 + (((((kk & 3) << 1) | lc) ^ (lr & 7)) << 4), A[j][0], A[j][1], A[j][2], A[j][3]);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) hmma_bf16(acc, A[j][0], A[j][1], A[j][2], A[j][3], b[j][0], b[j][1]);
}
// the same for dense16t_chunk (K-major wo_a^T tiles, ldmatrix.trans)
__device__ __forceinline__ void dense16t_c1u(float (&acc)[4], uint32_t st, uint32_t img, int kofs, int NR, int w, int lane) {
  static_assert(NCW == 8, "C1M_DU: 4 k16 steps per warp");
  const int g = lane >> 2, t = lane & 3, m = lane >> 3, j8 = lane & 7;
  unsigned b[4][2], A[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int kk = w + 8 * j;
    b[j][0] = 0u, b[j][1] = 0u;
    if (g < NR) {
      const uint32_t ba = img + g * IMS + (kofs + 16 * kk) * 2 + 4 * t;
      b[j][0] = lds32(ba), b[j][1] = lds32(ba + 16);
    }
    ldm_x4t(st + (16 * kk + j8 + 8 * (m >> 1)) * 32 + 16 * (m & 1), A[j][0], A[j][1], A[j][2], A[j][3]);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) hmma_bf16(acc, A[j][0], A[j][1], A[j][2], A[j][3], b[j][0], b[j][1]);
}

// the engine's W8A16 packed fp8 as bf16 MMA operands, bit-exact: 4 e4m3 -> 2 bf16x2 words = e4m3 x 2^(s - 127) exactly (the e4m3 bits placed as a bf16 of
// value e4m3 x 2^-120, zeros and subnormals included, then x 2^(s - 7) by mul.rn.bf16x2: an exact power-of-two scaling, every
// product a normal bf16 for scale bytes 16..134 (w8a16's check_scales)) = the engine's w8dq bf16 copy, bit for bit
__device__ __forceinline__ unsigned prmt8(unsigned a, unsigned b, unsigned sel) {
  unsigned d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
__device__ __forceinline__ unsigned bf16x2_mul8(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ void e4m3x4_to_bf(unsigned v, unsigned m2, unsigned& lo, unsigned& hi) {   // bytes 0, 1 -> lo; 2, 3 -> hi
  const unsigned mg = v & 0x7f7f7f7fu, sg = v & 0x80808080u;
  lo = bf16x2_mul8(prmt8(mg, 0u, 0x4140u) * 16u + prmt8(sg, 0u, 0x1404u), m2);
  hi = bf16x2_mul8(prmt8(mg, 0u, 0x4342u) * 16u + prmt8(sg, 0u, 0x3424u), m2);
}
__device__ __forceinline__ unsigned scale_m2b(unsigned s) {   // bf16x2 of 2^((s + 120) - 127) = 2^(s - 7)
  const unsigned b = (s + 120u) << 7;
  return b | (b << 16);
}
// one 512-K stage of ONE 16-row tile from the packed layout (the producer's 5-D box: [4 chunks of 128 K][4 k32 blocks][32 lanes][16
// B] = the m16n8k16 A fragments of rows g, g + 8, k 2t.. / 2t + 8.. of both k16 halves; then [4 chunks][16 B] ue8m0 scales =
// [k32 block][32-row block]); rb = the tile's 32-row block in its 128-row half tile. warp w's
// k16 steps kk = w + 8 j (chunk j, k32 block w >> 1, half w & 1), every load first, then the 4 MMAs in the same order -> the
// same registers and products as the bf16 quarters' ldmatrix(.trans) A fragments: BITWISE
constexpr int L_D8S = 8192;   // the stage's scale bytes (after 8 KB of fragments)
__device__ __forceinline__ void dense8_c1u(float (&acc)[4], uint32_t st, uint32_t img, int kofs, int NR, int rb, int w, int lane) {
  static_assert(NCW == 8, "srv8x: 4 k16 steps per warp per 512-K stage");
  const int g = lane >> 2, t = lane & 3;
  unsigned b[4][2], A[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int kk = w + 8 * j;
    b[j][0] = 0u, b[j][1] = 0u;
    if (g < NR) {
      const uint32_t ba = img + g * IMS + (kofs + 16 * kk) * 2 + 4 * t;
      b[j][0] = lds32(ba), b[j][1] = lds32(ba + 16);
    }
    const uint2 v = lds64(st + j * 2048 + (w >> 1) * 512 + lane * 16 + (w & 1) * 8);
    unsigned sb;
    asm volatile("ld.shared.u8 %0, [%1];" : "=r"(sb) : "r"(st + L_D8S + j * 16 + (w >> 1) * 4 + rb));
    const unsigned m2 = scale_m2b(sb);
    e4m3x4_to_bf(v.x, m2, A[j][0], A[j][1]);
    e4m3x4_to_bf(v.y, m2, A[j][2], A[j][3]);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) hmma_bf16(acc, A[j][0], A[j][1], A[j][2], A[j][3], b[j][0], b[j][1]);
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
  alignas(16) float rt2[4][NRM];     // (C1M_RT1) warps 0-3's partial sums of the CTA's third router row
  alignas(16) float fsl[NRM][48];    // (C1M_RKS) this CTA's columns of the normalized f rows
  unsigned cpush;                    // (C1M_CPUSH) the consumers' top-k push of this call is done (= ncall)
#ifdef C1M_PDD
  unsigned short umap[AK];           // (C1M_PDD) this CTA's unit: the staging slot of each top-k key (0xFFFF: invalid)
  unsigned short pdd[32];            // (C1M_PDD) this CTA's top-k records (push order): their first-occurrence slot
  int umode;                         // (C1M_PDD) 1: this CTA's staged units read their top-k keys through umap
#endif
  alignas(16) int pv[NRM];
  alignas(16) float rmax[NCW][NRM];
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
  int m3w[4][NRM];                   // (new insert path) per warp 0..3: its first window entry holding the new slot
#ifndef C1M_INS_OLD
  // the new insert path's window scan is one full-mask ballot per warp
  // over tid < WIN and m3w has 4 warps: WIN <= 128 only. Larger windows: -DC1M_INS_OLD (WIN-general) until it is generalized
  static_assert(C1M_WIN <= 128, "the new insert path (m3w[4], full-warp ballots over tid < WIN) needs WIN <= 128: define C1M_INS_OLD");
#endif
  long long xit;                     // (C1M_X16) the ring position the base warps hand to warps 10-17 at C1M_XENTER
  alignas(16) bf16 xinb[NRM][48];    // this CTA's columns of the layer's moe_out: the next layer's A1 input
  long long uoff[AK][2];             // (attention fast path) the CTA's first unit's records: data / scale offsets (-1: none)
  const uint8_t* uoffb[AK];          //   and base pointers; set by the layer-start KV prefetch
#ifdef C1S_OGMC
  alignas(8) unsigned long long ogmb;   // the og multicast's mbarrier: one phase per wo_a layer
  unsigned ogph;                        //   its current parity
#endif
};
static_assert(offsetof(Smem, yp) % 16 == 0 && offsetof(Smem, part) % 16 == 0 && offsetof(Smem, red) % 16 == 0, "align");

// ------------------------------------------------------------------------------------------------ the kernel
template <bool FULL>
struct LayCfg {
  // ring stages (FULL: the attention stage takes the SMEM of one); -DC1M_NSF=n: FULL's ring
  // depth (the NR > 4 builds' SMEM: one 10 KB activation row per row; probe of the ring-depth cost at NR = 4)
#ifndef C1M_NSF
#define C1M_NSF (C1M_NRM <= 4 ? 7 : C1M_NRM == 5 ? 6 : 5)
#endif
  static constexpr int NS = FULL ? C1M_NSF : (NRM <= 4 && STAGE <= 20480 ? NSMAX : 6);   // (k_layer<false>, NRM 6 / larger stages: 6 fit)
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
  auto cwait1 = [&]() -> uint32_t {   // (C1M_RT1) the stage after the current one, not consumed yet (no wait timing)
    const int s = (int)((it + 1) % NS);
    mb_wait(su32(&S.full[s]), (unsigned)(((it + 1) / NS) & 1));
    return ring_u + s * STAGE;
  };
  auto crel = [&]() {   // a base-warp stage: XRC arrivals per warp (C1M_X16: 2, the extra warps do not consume it)
    __syncwarp();
    if (lane == 0) {
      if constexpr (NCWX == NCW) mb_arrive(su32(&S.empty[(int)(it % NS)]));
      else mb_arrive_n(su32(&S.empty[(int)(it % NS)]), xrc_of(w));
    }
    ++it;
  };
  auto crel1 = [&]() {   // (C1M_X16) a stage all 16 expert-phase warps consume: one arrival per warp
    __syncwarp();
    if (lane == 0) mb_arrive(su32(&S.empty[(int)(it % NS)]));
    ++it;
  };
  (void)crel1;
  // grid arrival: the CTA's barrier, then thread 0's red.release.gpu (cumulative over the CTA's writes ordered before it
  // by the barrier, as CUTLASS's GenericBarrier). the extra __threadfence() (MEMBAR.SC.GPU) in
  // front of the release was redundant; -DC1M_FENCE_OLD restores it
  auto arrive = [&](int k) {
    cbar();
    if (tid == 0) {
#ifdef C1M_FENCE_OLD
      __threadfence();
#endif
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
#if defined(C1M_ROWP_OLD) || (C1M_NRM * C1M_NCW != 32)   // (NRM > 4: the two-barrier version, same values)
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
#else
    // one barrier: every warp takes the 8 warps' maxima itself (lane l = warp l >> 2, row l & 3; 3
    // shuffles; max is exact, so the same pv) and its lanes 0..3 write S.pv (the same values from every warp), in its own
    // array (S.rmax: S.wmax's later users run without a barrier after rowp)
    static_assert(NCW * NRM == 32, "rowp: one lane per (warp, row)");
#pragma unroll
    for (int n = 0; n < NRM; ++n) {
      float m = mx[n];
#pragma unroll
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
      if (lane == 0) S.rmax[w][n] = m;
    }
    cbar();
    {
      float m = S.rmax[lane >> 2][lane & 3];
      m = fmaxf(m, __shfl_xor_sync(~0u, m, 4));
      m = fmaxf(m, __shfl_xor_sync(~0u, m, 8));
      m = fmaxf(m, __shfl_xor_sync(~0u, m, 16));
      if (lane < NRM) S.pv[lane] = p_of(fmaxf(m, 0.f));
      __syncwarp();
    }
#endif
  };
  // mHC boundary slice of this CTA's columns (k_frow's arithmetic): xo = bf16(post y + comb^T rs), in = bf16(sum lag
  // xo), the sum of squares of in (RMSNorm partial) and the 24 mix dot products + the sum of squares of xo (fn chunk)
  auto bslice = [&](auto yget, const bf16* rsx, bf16* xo, bf16* ino, float* sspo, float* mixo, int ck, int ckm, int sb,
#ifdef C1S_GPF
                    bool xsin = false, bool xred = false) {   // xsin: the residual streams from S.xs (this CTA's previous G
                                                              // slice); xred (C1S_GPF): from S.red (raw bf16 [n][cc][48])
#else
                    bool xsin = false) {   // xsin: the residual streams from S.xs (this CTA's previous G slice)
#endif
    float ss[NRM] = {0.f, 0.f, 0.f, 0.f};
    for (int i = tid; i < NR * ncols; i += CTH) {
      const int n = i / ncols, j = i % ncols, col = col0 + j;
      const float yv = yget(n, j, col);
      float rv[HC];
#pragma unroll
#ifdef C1S_GPF
      for (int cc = 0; cc < HC; ++cc)
        rv[cc] = xsin ? S.xs[n][cc][j]
                      : xred ? __bfloat162float(reinterpret_cast<const bf16*>(S.red)[(n * HC + cc) * 48 + j])
                             : __bfloat162float(rsx[((size_t)n * HC + cc) * D + col]);
#else
      for (int cc = 0; cc < HC; ++cc) rv[cc] = xsin ? S.xs[n][cc][j] : __bfloat162float(rsx[((size_t)n * HC + cc) * D + col]);
#endif
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
#ifndef C1M_MX_OLD
    arrive(ck);   // the norm partials (and xo / in) first: the RMSNorm waits only for these, the mix dots below run meanwhile
#endif
    stamp(sb + 1);
#ifdef C1M_DOTS_WIN
    (void)mixo, (void)ckm;
  };
  // the slice's 24 mix dot products + the sum of squares of xo (fn chunk from the ring), split off the
  // boundary: only the service CTAs' mixes read them, much later, so each CTA runs them in a wait window (A1: after its
  // q_a tile, before waitc(C_QA); G: after its router rows) instead of in front of its RMSNorm. S.xs keeps the slice
  // until then; the producer streams the fn chunk at the matching ring position
  auto bdots = [&](float* mixo, int ckm, int sb) {
#endif
    // the 24 mix dot products + the sum of squares of xm2 over this slice (fn chunk from the ring)
    const uint32_t st = cwait();
    stamp(sb + 2);
    const float* fs = (const float*)(ring + (st - ring_u));
#ifdef C1M_DOTS_BR
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
                const float fv = m == 24 ? 0.f : fs[C1M_FNI(m, cc, j)];
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
#else
    {   // branch-free: warp w = mixes m = w, w + 8, w + 16 (< 24 always) and the sum of squares of xo
        // (m = 24: every warp computes it, warp 0 stores it); lanes past the CTA's columns read column 0 and keep their sums
        // by a select: per (m, row) the same terms in the same order as the branchy version (bitwise), no divergence
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
          const bool ok = j < ncols;
          const int jc = ok ? j : 0;
          float xv[NRM];
#pragma unroll
          for (int n = 0; n < NRM; ++n) xv[n] = S.xs[n][cc][jc];
#pragma unroll
          for (int q = 0; q < 3; ++q) {
            const float fv = fs[C1M_FNI(w + NCW * q, cc, jc)];
#pragma unroll
            for (int n = 0; n < NRM; ++n) {
              const float t = fmaf(xv[n], fv, acc[q][n]);
              acc[q][n] = ok ? t : acc[q][n];
            }
          }
#pragma unroll
          for (int n = 0; n < NRM; ++n) {
            const float t = fmaf(xv[n], xv[n], acc[3][n]);
            acc[3][n] = ok ? t : acc[3][n];
          }
        }
#pragma unroll
      for (int o = 16; o; o >>= 1)
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int n = 0; n < NRM; ++n) acc[q][n] += __shfl_xor_sync(~0u, acc[q][n], o);
      if (lane == 0) {
#pragma unroll
        for (int q = 0; q < 3; ++q)
#pragma unroll
          for (int n = 0; n < NRM; ++n)
#ifdef C1M_LBM
            if (n < NR) st_lb(mixo + 2 * ((c * NRM + n) * 25 + 1 + w + NCW * q), acc[q][n], ncall);
#else
            if (n < NR) mixo[(c * NRM + n) * 25 + 1 + w + NCW * q] = acc[q][n];
#endif
        if (w == 0)
#pragma unroll
          for (int n = 0; n < NRM; ++n)
#ifdef C1M_LBM
            if (n < NR) st_lb(mixo + 2 * ((c * NRM + n) * 25), acc[3][n], ncall);
#else
            if (n < NR) mixo[(c * NRM + n) * 25] = acc[3][n];
#endif
      }
    }
#endif
    stamp(sb + 3);
    crel();
#ifdef C1M_DOTS_WIN
    arrive(ckm);   // the mix partials (read by the service CTAs' mixes)
  };
#else
#if defined(C1M_LBM)
    (void)ckm;     // (polled by the service CTAs' mixes: no arrival)
#elif !defined(C1M_MX_OLD)
    arrive(ckm);   // the mix partials (read by the service CTAs' mixes)
#else
    (void)ckm;
    arrive(ck);
#endif
  };
#endif
  // RMSNorm of the boundary rows from the slices' partial sums (every CTA, all columns): natural bf16 rows in the image
  // region (+ fo), S.pv = their prescale exponents
  auto rnorm = [&](const float* sspi, const bf16* ini, const float* nwi, bf16* fo, bool needp = true, bool fone = false,
                   bool nwpre = false, int part = 0) {   // nwpre (C1M_NWP): the norm weights were prefetched into the scratch
                                                          // already; part (C1M_RKS): 1 = loads + rr only, 2 = the normalize only
    // needp: S.pv; fone: fo is written by all CTAs (A2's aout): each chunk once
#ifndef C1M_RN_OLD
    constexpr bool RNA = FULL;   // (FULL only: the norm weights go to the attention scratch)
#else
    constexpr bool RNA = false;
#endif
    if (part != 2) {
    if constexpr (RNA) {
    // the rows' chunks (into the image region, in place) and the norm weights (into the attention
    // scratch, free here) by cp.async first: their L2 round trip overlaps the partial-sum reduction (one warp per row, as
    // before: NOT every warp, which hot-spots the 2 KB of partials from every SM); no registers held across the barrier
      float* const nws = reinterpret_cast<float*>(imgp + NRM * IMS);   // [5120] fp32 (ATTX = 22,528 B)
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int c8 = tid + CTH * j;
        if (c8 < 640) {
#pragma unroll
          for (int n = 0; n < NRM; ++n)
            if (n < NR)
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(img + n * IMS + 16 * c8),
                           "l"(ini + (size_t)n * D + 8 * c8) : "memory");
          if (!nwpre) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(su32(nws + 8 * c8)), "l"(nwi + 8 * c8) : "memory");
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(su32(nws + 8 * c8 + 4)), "l"(nwi + 8 * c8 + 4) : "memory");
          }
        }
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
      static_assert(!RNA || ATTX >= D * 4, "rnorm: the norm weights in the attention scratch");
    }
    if (w < NR) {   // warp n = row n: lane l sums CTAs l, l + 32, .. (5 loads in flight), then a butterfly
      float v[5];
#pragma unroll
      for (int k = 0; k < 5; ++k) v[k] = lane + 32 * k < NCTA ? __ldcg(sspi + (lane + 32 * k) * NRM + w) : 0.f;
      float s = (((v[0] + v[1]) + v[2]) + v[3]) + v[4];
#pragma unroll
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
      if (lane == 0) S.rr[w] = rsqrtf(s / (float)D + a.norm_eps);
    }
    if constexpr (RNA) asm volatile("cp.async.wait_all;" ::: "memory");
    cbar();
    stamp(38);
    }
    if (part == 1) return;
    float mx[NRM] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 3; ++j) {   // 640 chunks of 8 columns: thread tid takes tid, tid + 256, tid + 512 (all loads first)
      const int c8 = tid + CTH * j;
      if (c8 >= 640) break;
      const float* const nws = reinterpret_cast<const float*>(imgp + NRM * IMS);
      const float4 w0 = RNA ? *(const float4*)(nws + 8 * c8) : __ldg((const float4*)(nwi + 8 * c8));
      const float4 w1 = RNA ? *(const float4*)(nws + 8 * c8 + 4) : __ldg((const float4*)(nwi + 8 * c8 + 4));
      uint4 v[NRM];
#pragma unroll
      for (int n = 0; n < NRM; ++n) {
        if constexpr (RNA) v[n] = n < NR ? *(const uint4*)(imgp + n * IMS + 16 * c8) : make_uint4(0u, 0u, 0u, 0u);   // (own copy)
        else v[n] = n < NR ? __ldcg((const uint4*)(ini + (size_t)n * D) + c8) : make_uint4(0u, 0u, 0u, 0u);
      }
      const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int n = 0; n < NRM; ++n) {
        if (n < NR) {   // (no break: a fully unrolled loop keeps mx in registers)
          const unsigned u[4] = {v[n].x, v[n].y, v[n].z, v[n].w};
          unsigned o[4];
          const float rr = S.rr[n];
#ifdef C1M_RNP
          // one cvt.rn.bf16x2 per pair (the same rounding per element) and the |max| on the bf16 bits
          // (non-negative bf16 order = integer order): the same image and the same row maxima
          unsigned amx = 0u;
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float f0 = __uint_as_float(u[q] << 16), f1 = __uint_as_float(u[q] & 0xffff0000u);
            const __nv_bfloat162 bb = __floats2bfloat162_rn((f0 * rr) * wv[2 * q], (f1 * rr) * wv[2 * q + 1]);
            o[q] = *reinterpret_cast<const unsigned*>(&bb);
            amx = __vmaxu2(amx, o[q] & 0x7fff7fffu);
          }
          mx[n] = fmaxf(mx[n], __uint_as_float(max(amx & 0xffffu, amx >> 16) << 16));
#else
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float f0 = __uint_as_float(u[q] << 16), f1 = __uint_as_float(u[q] & 0xffff0000u);
            const bf16 b0 = __float2bfloat16_rn((f0 * rr) * wv[2 * q]), b1 = __float2bfloat16_rn((f1 * rr) * wv[2 * q + 1]);
            mx[n] = fmaxf(mx[n], fmaxf(fabsf(__bfloat162float(b0)), fabsf(__bfloat162float(b1))));
            o[q] = (unsigned)__bfloat16_as_ushort(b0) | ((unsigned)__bfloat16_as_ushort(b1) << 16);
          }
#endif
          const uint4 ov = make_uint4(o[0], o[1], o[2], o[3]);
          *(uint4*)(imgp + n * IMS + 16 * c8) = ov;  // natural bf16 (router), converted to f16 P8 after the router
#ifndef C1M_AOUT_OLD
          // each chunk ONCE (CTA c8 % 132): every CTA has the same image; the index phases
          // read aout after C_QA, which every CTA passes after this layer's A2 barrier
          if (fo && (!fone || c8 % NCTA == c)) *(uint4*)(fo + (size_t)n * D + 8 * c8) = ov;
#else
          if (fo) *(uint4*)(fo + (size_t)n * D + 8 * c8) = ov;
#endif
        }
      }
    }
    if (needp) rowp(mx);
  };
  // the boundary's mixes of row n (k_fbm8's statements), on service CTA 128 + n: the 25 sums over the 132 slices by all
  // threads (value q, group k of CTAs k, k + 10, ..; the 10 group sums added in order), the Sinkhorn on warp 0
  auto mixes = [&](int n, const float* mixi, const float* sc, const float* bs, float* preo, float* posto, float* combo) {
    if (tid < 250) {
      const int q = tid / 10, k = tid % 10;
      float v[14];
#ifdef C1M_LBM
      {   // polled {partial, ncall} entries: all loads in flight, then each re-polled until its epoch is this layer's
        unsigned long long u[14];
#pragma unroll
        for (int i = 0; i < 14; ++i)
          u[i] = k + 10 * i < NCTA ? ld_lb(mixi + 2 * (((k + 10 * i) * NRM + n) * 25 + q)) : (unsigned long long)ncall << 32;
#pragma unroll
        for (int i = 0; i < 14; ++i)
          while ((unsigned)(u[i] >> 32) != ncall) u[i] = ld_lb(mixi + 2 * (((k + 10 * i) * NRM + n) * 25 + q));
#pragma unroll
        for (int i = 0; i < 14; ++i) v[i] = __uint_as_float((unsigned)u[i]);
      }
#else
#pragma unroll
      for (int i = 0; i < 14; ++i) v[i] = k + 10 * i < NCTA ? __ldcg(mixi + ((k + 10 * i) * NRM + n) * 25 + q) : 0.f;
#endif
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
  // with C1M_ONE_BODY every layer runs this body (HX = true); a regular layer's HX-dependent parts are
  // gated at run time here (the hooks themselves branch on a.ix / a.eg)
  const bool hxl = HX && (a.ix != nullptr || a.eg != nullptr);
  (void)hxl;
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
        spin_ge_sys(a.rdy[rank] + c, call + 1u);
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
#ifdef C1M_OWNSTG
      const bool attend = AMODE(a) == 0 || (rank == a.owner && hxl);   // (C1M_OWNSTG: regular layers read the staging)
#else
      const bool attend = AMODE(a) == 0 || rank == a.owner;   // (modes 2 / 3: the peers' pages are not local)
#endif
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
#ifdef C1M_ATT_FAST
          if (ln == 0 && u == c) {   // the first unit's record offsets for the attention (regular layers: tables unchanged)
            const long long pg = idx >> sh, ps = idx & ((1 << sh) - 1);
            S.uoff[k][0] = idx >= 0 ? pg * stride + ps * 576 : -1LL;
            S.uoff[k][1] = idx >= 0 ? pg * stride + (576LL << sh) + ps * 8 : -1LL;
            S.uoffb[k] = b;
          }
#endif
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
#ifndef C1M_A1S_OLD
    // a regular layer after the launch's first: both inputs are this CTA's own outputs of the previous
    // layer (moe_out's columns from its AR #2, the X streams from its G slice, still in S.xs): from SMEM, no L2 round trip
    const bool a1s = !hxl && l > 0;
#else
    const bool a1s = false;
#endif
    bslice([&](int n, int j, int col) { return a1s ? __bfloat162float(S.xinb[n][j]) : __bfloat162float(a.xin[(size_t)n * D + col]); },
           a.rsin, a.Xo, a.ain, a.ssp_a, a.mixp_a, C_SSA, C_MXA, 27, a1s);
    stamp(14);
    // (attention mode 3: the owner's KV push runs on the producer warp's lanes 1..31)
    // ---- A2 a = RMSNorm(in) x attn_norm -> f16 P8 image; the attention mixes (CTA 131)
    waitc(C_SSA, 132u * ncall);
#ifdef C1M_TS
    stamp(44);   // (diagnostic build: the attention boundary's sums of squares complete)
#endif
#ifdef C1M_NO_IX
#ifdef C1M_ROWP_OLD
    rnorm(a.ssp_a, a.ain, a.anw, nullptr);   // (deployment builds without in-kernel index layers: no aout code)
#else
    rnorm(a.ssp_a, a.ain, a.anw, nullptr, !B16(a, wqkv16, 0));   // (bf16 q_a | wkv: the natural image, no S.pv)
#endif
#else
#ifdef C1M_ROWP_OLD
    rnorm(a.ssp_a, a.ain, a.anw, a.aout);
#else
    rnorm(a.ssp_a, a.ain, a.anw, a.aout, !B16(a, wqkv16, 0), true);   // (hook 3 reads A4's S.pv, not A2's)
#endif
#endif
    for (int i = tid; i < NR * 640 && !B16(a, wqkv16, 0); i += CTH) {   // (bf16 q_a | wkv: the natural bf16 image as it is)
      const int n = i / 640, c8 = i % 640;
      uint4* q = (uint4*)(imgp + n * IMS + 16 * c8);
      *q = p8_f16(*q, pow2i(S.pv[n]));
    }
    cbar();
    stamp(15);
if constexpr (HX) {   // hook 7: after A2, before A3 (the image = A2's output for rows < NR); empty by default
#define C1M_HOOK 7
#include C1M_HOOKS
#undef C1M_HOOK
}
    // ---- A3 q_a | wkv (replicated): tile c = output columns 16 c .. 16 c + 15 of [1280 q_a | 512 kv]
    if (c < 112) {
#define C1M_DPART 1
#include C1M_DENSE_A3
#undef C1M_DPART
#ifdef C1M_TS
      stamp(47);   // (diagnostic build: the A3 epilogue done, before its C_QA arrival)
#endif
#if defined(C1M_QAG)
      // counter-gated Lamport: regular layers count the arrival relaxed (no release fence; the readers
      // check the data against the sentinel); phase layers keep the release (the index phases read aout after C_QA)
      if (hxl) {
        arrive(C_QA);
      } else {
        __syncwarp();
        if (tid == 0) red_rlx(a.cnt + C_QA, 1u);
      }
#elif defined(C1M_QAL)
      // only warp 0 wrote qakv (polled by the readers): warp 0 alone releases C_QA (the index phases'
      // aout ordering: A2's barrier already ordered every warp's aout writes before it); the other warps go on to A4
      __syncwarp();
      if (tid == 0) red_rel(a.cnt + C_QA, 1u);
#else
      arrive(C_QA);
#endif
    }
#ifdef C1M_DOTS_WIN
    bdots(a.mixp_a, C_MXA, 27);   // (the A1 slice's mix dots in the q_a wait window)
#endif
    stamp(16);
    // ---- A4 q norm (every CTA: qr -> f16 P8 image), kv norm + RoPE + window record (CTA 0 when do_insert)
#if defined(C1M_QAG)
    if (hxl) {
      waitc(C_QA, 112u * ncall);
    } else {   // (counter-gated: the readers start once every writer counted; the data loads still check the sentinel)
      if (tid == 0) spin_ge_nf(a.cnt + C_QA, 112u * ncall);
      cbar();
    }
    const bf16* const qak = a.qakv + (size_t)par * NRM * 1792;
#elif defined(C1M_QAL)
    if (hxl) waitc(C_QA, 112u * ncall);   // (phase layers: the index phases read aout after it; regular layers poll qakv)
    const bf16* const qak = a.qakv + (size_t)par * NRM * 1792;
#else
    waitc(C_QA, 112u * ncall);
    const bf16* const qak = a.qakv;
#endif
    stamp(39);
    if (c >= NWK) arrive(C_QB);   // the service CTAs have no q_b tiles: their C_QB arrival first, the insert after it
                                  // (the attention waits for the inserts through C_INS / the mode-3 push counter)
// the owner's pushes also fill ITS OWN staging (and count on its own counter), so its attention
// units read the records from HBM staging like the peers (not the paged pool in host memory); the same record bytes
#ifdef C1M_OWNSTG
#define C1M_PUSH_TO(p_) (true)
#define C1M_PUSH_DATA(p_, hx_) (true)   // (NOT skippable on phase layers: their new top-k is what the next regular
                                        // layers' units read from the owner's own staging)
#if defined(C1M_INS_OLD)
#error "C1M_OWNSTG: the new insert path only"
#endif
#else
#define C1M_PUSH_TO(p_) ((p_) != rank)
#define C1M_PUSH_DATA(p_, hx_) ((p_) != rank)
#endif
#if defined(C1M_RKS) && (!defined(C1M_RTL) || defined(C1M_RT1))
#error "C1M_RKS: the split-K router publishes its logits through C1M_RTL (and replaces C1M_RT1's router pass)"
#endif
#if defined(C1M_PDD) && (!defined(C1M_STF) || !defined(C1M_OWNSTG) || defined(C1M_CPUSH) || defined(C1M_ATT_FAST) || defined(C1M_PUSH_OLD))
#error "C1M_PDD: the staged-unit path (C1M_STF + C1M_OWNSTG) and the split push only"
#endif
#if defined(C1M_QAG) && !defined(C1M_QAL)
#error "C1M_QAG: counter-gated Lamport on top of C1M_QAL's data layout (define both)"
#endif
#if defined(C1M_ZG) && !defined(C1M_ZL)
#error "C1M_ZG: counter-gated Lamport on top of C1M_ZL's data layout (define both)"
#endif
#if defined(C1M_INS_OLD) && (C1M_NRM > 4 || defined(C1M_QAL))
#error "C1M_INS_OLD: one row per service CTA (NRM <= 4)"
#endif
#ifndef C1M_INS_OLD
    // the q norm + image only where it is read: the workers (q_b) and, on phase layers, the
    // service CTAs too (hook 3); a regular layer's service CTAs go straight to the insert (the owner's critical path)
    const bool qnorm = hxl || c < NWK;
#else
    const bool qnorm = true;
#endif
    if (qnorm) {
      // q_a rows: 160 chunks of 8 per row; thread tid: chunk tid (tid < 160) of every row (loads first)
      uint4 qv[NRM];
      float ss[NRM] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int n = 0; n < NRM; ++n)
#ifdef C1M_QAL
        qv[n] = (n < NR && tid < 160) ? ld_rlx_v4((const uint4*)(qak + (size_t)n * 1792) + tid) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int n = 0; n < NRM; ++n)
        if (n < NR && tid < 160)
          while (has_s16(qv[n])) qv[n] = ld_rlx_v4((const uint4*)(qak + (size_t)n * 1792) + tid);
#else
        qv[n] = (n < NR && tid < 160) ? __ldcg((const uint4*)(qak + (size_t)n * 1792) + tid) : make_uint4(0u, 0u, 0u, 0u);
#endif
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
    }
#ifdef C1M_INS_OLD
    {
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
#else
    // service CTA 128 + n: row n's kv norm + RoPE + window record with 2 barriers instead of 5.
    // Thread t: dims t, t + 256 for the sum of squares (the old order: the same rs bits) and dims 2t, 2t + 1 for the record
    // (warp w = the record's 64-dim block w, warp 7 = the RoPE dims 448..511: no SMEM exchange); the record is staged in
    // SMEM too, so the owner's mode-3 push takes it from there (no L2 read-back, no fences beyond the arrivals' releases)
    // (NRM > 4: rows 4, 5 on service CTAs 128, 129 as a second pass; every pass ends with a barrier)
    for (int n = c - NWK; c >= NWK && n < NR; n += NCTA - NWK) {
      const bool push = AMODE(a) == 3 && rank == a.owner;
      const long long slot = (a.do_insert || push) ? __ldg(a.own + n) : -1;
      uint8_t* const rec = reinterpret_cast<uint8_t*>(&S.xs[0][0][0]);   // 576 data + 8 scale bytes
      if (push && tid < WIN) {   // the window index of the new slot in each row: warp w's first hit (lanes = entries)
        for (int i = 0; i < NR; ++i) {
          const bool hit = slot >= 0 && tid < __ldg(a.sl + i) && (long long)__ldg(a.si + i * a.si_rs + tid) == slot;
          const unsigned bb = __ballot_sync(~0u, hit);
          if (lane == 0) S.m3w[w][i] = bb ? 32 * w + __ffs(bb) - 1 : -1;
        }
      }
      if (a.do_insert && slot >= 0) {   // (uniform)
        const bf16* kr = qak + (size_t)n * 1792 + 1280;
#ifdef C1M_QAL
        // (polled: dims 2t, 2t + 1 = the pair pr; t and t + 256 come from the pairs t / 2 and 128 + t / 2)
        const unsigned* const kr32 = reinterpret_cast<const unsigned*>(kr);
        unsigned pr = ld_rlx_u32(kr32 + tid), pa = ld_rlx_u32(kr32 + (tid >> 1)), pb = ld_rlx_u32(kr32 + 128 + (tid >> 1));
        while (has_s16(pr)) pr = ld_rlx_u32(kr32 + tid);   // (batched: the three first loads in flight together)
        while (has_s16(pa)) pa = ld_rlx_u32(kr32 + (tid >> 1));
        while (has_s16(pb)) pb = ld_rlx_u32(kr32 + 128 + (tid >> 1));
        const float x0 = __uint_as_float((tid & 1) ? (pa & 0xffff0000u) : (pa << 16));
        const float x1 = __uint_as_float((tid & 1) ? (pb & 0xffff0000u) : (pb << 16));
#else
        const float x0 = __bfloat162float(__ldcg(kr + tid)), x1 = __bfloat162float(__ldcg(kr + 256 + tid));
        const unsigned pr = __ldcg(reinterpret_cast<const unsigned*>(kr) + tid);   // dims 2t, 2t + 1
#endif
        const float2 kn = __ldg(reinterpret_cast<const float2*>(a.kv_norm) + tid);
        float cc = 1.f, sn = 0.f;
        if (w == 7) {
          const float* cs = a.cos_sin + __ldg(a.pos + n) * 64;
          cc = __ldg(cs + lane), sn = __ldg(cs + 32 + lane);
        }
        float m = __fadd_rn(__fmul_rn(x0, x0), __fmul_rn(x1, x1));
#pragma unroll
        for (int o = 16; o; o >>= 1) m += __shfl_xor_sync(~0u, m, o);
        if (lane == 0) S.wmax[w][0] = m;
        cbar();
        float tot = 0.f;
        for (int ww = 0; ww < NCW; ++ww) tot += S.wmax[ww][0];
        const float rs = rsqrtf(__fadd_rn(__fdiv_rn(tot, 512.f), a.attn_eps));
        const float p0 = __uint_as_float(pr << 16), p1 = __uint_as_float(pr & 0xffff0000u);
        float y0 = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(p0, rs), kn.x)));
        float y1 = __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(p1, rs), kn.y)));
        if (w == 7) {   // RoPE of dims 448..511 (norm_rope_row: on the bf16 normed values)
          const float e = y0, od = y1;
          y0 = __bfloat162float(__float2bfloat16_rn(__fsub_rn(__fmul_rn(e, cc), __fmul_rn(od, sn))));
          y1 = __bfloat162float(__float2bfloat16_rn(__fadd_rn(__fmul_rn(e, sn), __fmul_rn(od, cc))));
        }
        const unsigned yb = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(y0)) |
                            ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(y1)) << 16);
        if (a.kvout) reinterpret_cast<unsigned*>(a.kvout + (size_t)n * 512)[tid] = yb;
        // the stock sm90 fp8_ds_mla record (dense.cu rope_q_insert_kernel's statements)
        const long long pg = slot >> a.win_shift, ps = slot & ((1LL << a.win_shift) - 1);
        uint8_t* base = a.win + pg * a.win_stride;
        uint8_t* data = base + ps * 576;
        uint8_t* scl = base + (576LL << a.win_shift) + ps * 8;
        if (w < 7) {
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
          *reinterpret_cast<__nv_fp8x2_storage_t*>(rec + 64 * w + 2 * lane) = qv;
          if (lane == 0) scl[w] = (uint8_t)(ex + 127), rec[576 + w] = (uint8_t)(ex + 127);
        } else {
          reinterpret_cast<unsigned*>(data + 448)[lane] = yb;
          reinterpret_cast<unsigned*>(rec + 448)[lane] = yb;
          if (lane == 0) scl[7] = 0, rec[576 + 7] = 0;
        }
      }
      if (AMODE(a) == 2) __threadfence();   // (remote-KV mode: the peers read the owner's page after the sys flag below)
      arrive(C_INS);   // row n inserted (or nothing to insert here): every rank; its barrier also completes rec / S.m3w
      if (AMODE(a) == 2 && rank == a.owner) {   // remote-KV mode: row n's record is in the owner's pages
        cbar();
        if (tid < W && tid != rank) st_rel_sys(a.rdy[tid] + NCTA + n, ncall);
      }
      if (push) {   // owner push of row n's new record into every peer's staging, at each (row i, window index j)
        if (slot >= 0 && tid < NR * 37) {
          const int i = tid / 37, ch = tid % 37;
          int j = -1;
#pragma unroll
          for (int ww = 3; ww >= 0; --ww)
            if (S.m3w[ww][i] >= 0) j = S.m3w[ww][i];
          if (j >= 0) {
            const long long pg = slot >> a.win_shift, ps = slot & ((1 << a.win_shift) - 1), r = (long long)i * KVR + j;
            if (ch < 36) {
              const uint4 x = a.do_insert ? *reinterpret_cast<const uint4*>(rec + 16 * ch)
                                          : __ldcg((const uint4*)(a.win + pg * a.win_stride + ps * 576) + ch);
#pragma unroll
              for (int p = 0; p < W; ++p)
                if (C1M_PUSH_DATA(p, hxl)) *((uint4*)(a.stg[p] + r * STG_R) + ch) = x;
            } else {
              const uint2 x = a.do_insert ? *reinterpret_cast<const uint2*>(rec + 576)
                                          : __ldcg((const uint2*)(a.win + pg * a.win_stride + (576LL << a.win_shift) + ps * 8));
#pragma unroll
              for (int p = 0; p < W; ++p)
                if (C1M_PUSH_DATA(p, hxl)) *(uint2*)(a.stg[p] + r * STG_R + 576) = x;
            }
          }
        }
#ifdef C1M_FENCE_OLD
        __threadfence_system();
#endif
        cbar();
        if (tid < W && C1M_PUSH_TO(tid)) red_rel_sys(a.rdy[tid] + NCTA + NRM, 1u);   // (release.sys after the barrier)
      }
    }
    cbar();   // (every CTA, as the old block's last barrier: the q image is complete before q_b reads it)
#endif
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
#define C1M_DPART 1
#include C1M_DENSE_A5
#undef C1M_DPART
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

#if defined(C1M_UNIT_EARLY) && defined(C1M_STF) && defined(C1M_OWNSTG) && !defined(C1M_ATT_FAST) && !defined(C1M_PDD)
    // lever (a)) a regular layer's mode-3 unit (CTA c < NR ANCH: its one unit u = c) prepares its keys
    // HERE, while the other CTAs finish q_b: the staged-push wait; its AK staged records (592 B: data + scales) by cp.async into
    // the attention scratch's q / sred region (free until the q chunks' cp.async after the C_QB wait); the validity bytes into
    // the softmax mask soff; then the unit loop's dequant statements on the same bytes (from SMEM) into the tile: bitwise. The
    // tile + scratch (the A image region) are this CTA's own and free once its q_b tiles are done (its C_QB arrival above);
    // index / Engram layers (hxl: hook 4, the owner's pools) keep the old order.
    static_assert(AK * STG_R <= 16 * 512 * 2 + 4 * 16 * AK * 4, "the staged records fit in the q + sred scratch");
    const bool uearly = !hxl && AMODE(a) == 3 && c < NR * ANCH;
    if (uearly) {
      bf16* sk = (bf16*)imgp;                 // (the attention block's tile pointers)
      bf16* sq = sk + AK * 512;
      float* sred = (float*)(sq + 16 * 512);
      bf16* sp = (bf16*)(sred + 4 * 16 * AK);
      long long* soff = (long long*)(sp + 16 * (AK + 8));
      auto aswz = [](int row, int col) { return row * 512 + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7); };
      if (tid == 0) {   // every push of this layer has landed (NCTA early + NR new-record pushes): the block's wait, moved
        const unsigned long long t0 = tim ? gtime() : 0ull;
        spin_ge_sys(a.rdy[rank] + NCTA + NRM, (unsigned)(NCTA + NR) * ncall);
        if (tim) st_b += gtime() - t0;
      }
      cbar();
      const int n = c / ANCH, ch = c % ANCH;   // (mode 3: NHG 1)
      const uint8_t* b = a.stg[rank];
      const uint8_t* rec0 = b + (long long)(n * KVR + ch * AK) * STG_R;
      const uint32_t stg_s = su32(sq);        // [AK][STG_R] staged records
      for (int i = tid; i < AK * (STG_R / 16); i += CTH)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(stg_s + 16 * i), "l"(rec0 + 16 * i) : "memory");
      asm volatile("cp.async.commit_group;" ::: "memory");
      if (tid < AK) soff[2 * tid] = __ldcg(b + STG_V + n * KVR + ch * AK + tid) != 0 ? 0 : -1;   // (the softmax's key mask)
      asm volatile("cp.async.wait_all;" ::: "memory");
      cbar();
      constexpr int NT = (AK * 36 + CTH - 1) / CTH;
#pragma unroll 1
      for (int jq = 0; jq < NT; ++jq) {   // the unit loop's dequant (exact) into the swizzled tile, raw / ev / okv from SMEM
        const int i = tid + CTH * jq;
        if (i >= AK * 36) continue;
        const int k = i / 36, cc = i % 36;
        const bool okv = soff[2 * k] >= 0;
        const uint32_t rb = stg_s + k * STG_R;
        uint4 rawv;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(rawv.x), "=r"(rawv.y), "=r"(rawv.z), "=r"(rawv.w)
                     : "r"(cc < 28 ? rb + 16 * cc : rb + 448 + 16 * (cc - 28)));
        if (cc < 28) {
          uint4 lo = make_uint4(0u, 0u, 0u, 0u), hi = lo;
          if (okv) {
            unsigned e;
            asm volatile("ld.shared.u8 %0, [%1];" : "=r"(e) : "r"(rb + 576 + (cc >> 2)));
            const float scl = e == 0 ? 5.877471754111438e-39f : __uint_as_float(e << 23);
            const unsigned wv[4] = {rawv.x, rawv.y, rawv.z, rawv.w};
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
          *(uint4*)&sk[aswz(k, 448 + 8 * (cc - 28))] = okv ? rawv : make_uint4(0u, 0u, 0u, 0u);
        }
      }
      cbar();   // (the staging is read: the q chunks' cp.async after the C_QB wait may overwrite it)
    }
#else
    constexpr bool uearly = false;
#endif
    // ---- A6 attention (this GPU's 16 heads, local KV): units (row, chunk of AK keys), then combines (row, 32 dims)
    waitc(C_QB, 132u * ncall);
    stamp(18);
#ifdef C1M_CPUSH
    // the owner's index layers: the consumer warps push this CTA's records of the NEW top-k (j >= WIN; the
    // push warp's top-k pass, 3 serial batches on one warp, ~18-21 us after the selection) right after the selection, 256
    // threads; then the push warp only counts the arrivals. The owner's own units read the pools: not delayed by it
    if (hxl && AMODE(a) == 3 && rank == a.owner && a.ix != nullptr && a.push_topk) {
      if (w == 0) c1ixg::pushwait(a.ix, c, ncall);   // (the push warp's hook 5, a whole warp: it ends with __syncwarp)
      cbar();
      const int nj = KVR - WIN, nrec = (NR * nj - c + NCTA - 1) / NCTA;
      for (int t = tid; t < nrec * 37; t += CTH) {
        const int k = t / 37, q = t % 37, rr = c + NCTA * k, n = rr / nj, j = WIN + rr % nj;
        const long long rk = (long long)n * KVR + j;
        const uint8_t* b = nullptr;
        long long d = 0, sc = 0;
        const bool v = kv_rec(a, n, j, b, d, sc);
        if (q == 0)
#pragma unroll
          for (int p = 0; p < W; ++p)
            if (C1M_PUSH_DATA(p, hxl)) a.stg[p][STG_V + rk] = v ? 1 : 0;
        if (v) {
          if (q < 36) {
            const uint4 x = __ldcg((const uint4*)(b + d) + q);
#pragma unroll
            for (int p = 0; p < W; ++p)
              if (C1M_PUSH_DATA(p, hxl)) *((uint4*)(a.stg[p] + rk * STG_R) + q) = x;
          } else {
            const uint2 x = __ldcg((const uint2*)(b + sc));
#pragma unroll
            for (int p = 0; p < W; ++p)
              if (C1M_PUSH_DATA(p, hxl)) *(uint2*)(a.stg[p] + rk * STG_R + 576) = x;
          }
        }
      }
      __threadfence_system();
      cbar();
      if (tid == 0) *(volatile unsigned*)&S.cpush = ncall;   // (the push warp's fence + arrivals follow)
    }
#endif
#ifdef C1M_PDD
    // DEDUPLICATED top-k push: the rows' new top-k sets overlap, and the push is bandwidth-bound (~20 us with 32 or 256 threads per CTA). On the owner's index
    // layers every CTA hashes the 2,048 slots (SMEM, the attention scratch: free until this CTA's units), gives each (n, jj)
    // the smallest staging slot n' KVR + WIN + jj' holding the same KV record, writes the map entries of ITS records (the
    // push warp's order) to every GPU's staging (STG_M) and tells its push warp which records are first occurrences. The
    // units read their top-k keys through a per-CTA SMEM copy of their 32 map entries (umap), filled once per index layer,
    // so regular layers keep one round trip. Same bytes per key: bitwise.
    static_assert(TKA > 0 && KVR % AK == 0 && NRM * ANCH <= NCTA && NRM * TKA <= 3072, "C1M_PDD: shapes");
    if constexpr (HX) {   // (phase layers only: cons_ix's instantiation; the regular layers' body stays as it was)
      const bool pdd_l = AMODE(a) == 3 && a.ix != nullptr && a.push_topk;
      if (pdd_l && rank == a.owner) {
        constexpr int PHT = 4096, NE_ = NRM * TKA, PE = (NE_ + CTH - 1) / CTH;
        unsigned* const hk = reinterpret_cast<unsigned*>(imgp);                    // [PHT] slot + 1 (0: empty)
        unsigned* const hv = hk + PHT;                                             // [PHT] smallest staging slot
        int* const hx = reinterpret_cast<int*>(hv + PHT);                          // [NRM * TKA] the rows' slots (-1: none)
        unsigned short* const hm = reinterpret_cast<unsigned short*>(hx + NE_);    // [NRM * TKA] first-occurrence slots
        static_assert(2 * PHT * 4 + NE_ * 6 <= NRM * IMS + ATTX, "C1M_PDD: hash in the attention scratch");
        if (w == 0) c1ixg::pushwait(a.ix, c, ncall);   // (the selection is complete: ti / tl final; a whole warp)
        for (int i = tid; i < PHT; i += CTH) hk[i] = 0u, hv[i] = 0xffffffffu;
        cbar();   // (warp 0's pushwait -> every warp: ti / tl are read only after it)
        {   // the rows' slots into SMEM (loads first: one L2 round; short-lived registers)
          int v_[PE];
#pragma unroll
          for (int q = 0; q < PE; ++q) {
            const int e = tid + CTH * q, n = e / TKA, jj = e % TKA;
            v_[q] = (e < NE_ && n < NR && jj < __ldcg(a.tl + n)) ? __ldcg(a.ti + (size_t)n * a.ti_rs + jj) : -1;
          }
#pragma unroll
          for (int q = 0; q < PE; ++q)
            if (tid + CTH * q < NE_) hx[tid + CTH * q] = v_[q];
        }
        cbar();   // (the table is initialized, the slots staged)
#pragma unroll 1
        for (int e = tid; e < NE_; e += CTH) {
          const int ix = hx[e];
          if (ix < 0) continue;
          const unsigned key = (unsigned)ix + 1u, rk = (unsigned)((e / TKA) * KVR + WIN + e % TKA);
          unsigned h = (key * 2654435761u) >> 20;
          while (true) {
            const unsigned old = atomicCAS(hk + h, 0u, key);
            if (old == 0u || old == key) {
              atomicMin(hv + h, rk);
              break;
            }
            h = (h + 1) & (PHT - 1);
          }
        }
        cbar();
#pragma unroll 1
        for (int e = tid; e < NE_; e += CTH) {
          const int ix = hx[e];
          unsigned short m = 0xffffu;
          if (ix >= 0) {
            const unsigned key = (unsigned)ix + 1u;
            unsigned h = (key * 2654435761u) >> 20;
            while (hk[h] != key) h = (h + 1) & (PHT - 1);
            m = (unsigned short)hv[h];
          }
          hm[e] = m;
        }
        cbar();
        const int nrec = (NR * TKA - c + NCTA - 1) / NCTA;   // this CTA's records rr = c + NCTA k (the push warp's order)
        static_assert((NRM * TKA + NCTA - 1) / NCTA <= 32, "C1M_PDD: S.pdd");
        if (tid < nrec) {
          const int rr = c + NCTA * tid, n = rr / TKA, jj = rr % TKA;
          const unsigned short m = hm[rr];
          S.pdd[tid] = m;
#pragma unroll
          for (int p = 0; p < W; ++p) *((unsigned short*)(a.stg[p] + STG_M) + n * KVR + WIN + jj) = m;
        }
        if (tid < AK && c < NR * ANCH) {   // this CTA's own unit: read through umap on the regular layers after this one
          const int n = c / ANCH, j = (c % ANCH) * AK + tid;
          S.umap[tid] = j < WIN ? (unsigned short)(n * KVR + j) : (j < KVR ? hm[n * TKA + j - WIN] : (unsigned short)0xffffu);
        }
        __threadfence_system();   // (the map entries reach every GPU before the push warp's release of this CTA's push)
        cbar();                   // (also: hm / the table are read; the units may reuse the scratch)
        if (tid == 0) {
          S.umode = 1;
          __threadfence_block();
          *(volatile unsigned*)&S.cpush = ncall;
        }
      } else if (a.push_topk && tid == 0) {
        S.umode = 0;   // (a full top-k push: identity slots)
      }
    } else {
      if (a.push_topk && tid == 0) S.umode = 0;   // (a launch's first layer without phases: the full top-k push)
    }
#endif
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
#ifdef C1M_OWNSTG
      // (C1M_OWNSTG: the owner's records are in its own staging too; phase layers keep the pools: their new top-k is pushed
      // after the selection, the owner reads it in place without waiting for that push)
      const bool staged = AMODE(a) == 3 && (rank != a.owner || !hxl);
#else
      const bool staged = AMODE(a) == 3 && rank != a.owner;   // mode 3 peer: the records are in this GPU's staging
#endif
      if (staged && c < NR * NHG * ANCH && !uearly) {   // every push of this layer has landed
        if (tid == 0) {
          const unsigned long long t0 = tim ? gtime() : 0ull;
          const unsigned tgt = (unsigned)(NCTA + NR) * ncall;
          spin_ge_sys(a.rdy[rank] + NCTA + NRM, tgt);
          if (tim) st_b += gtime() - t0;
        }
        cbar();
#ifdef C1M_PDD
        // (the adapter gives a.ix to the OWNER only: a peer's index layer is a plain layer with push_topk, in either body)
        if (rank != a.owner && a.push_topk) {   // (a peer, the index layer: its unit's map entries,
          if (tid < AK) {                                          // kept for the regular layers until the next index layer)
            const int n = c / ANCH, j = (c % ANCH) * AK + tid;
            S.umap[tid] = (j >= WIN && j < KVR) ? __ldcg((const unsigned short*)(a.stg[rank] + STG_M) + n * KVR + j)
                                                : (unsigned short)(n * KVR + j);
          }
          if (tid == 0) S.umode = 1;
          cbar();
        }
#endif
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
            spin_ge_sys(a.rdy[rank] + NCTA + n, ncall);
            if (tim) st_b += gtime() - t0;
          }
          cbar();
        }
#ifndef C1M_ATT_FAST
#ifdef C1M_STF
        // staged units: the records sit at fixed staging slots, so their loads go out with the validity
        // flags in ONE round trip (no address phase, no barrier before the keys); invalid keys are zeroed / masked as before
        const bool stf = staged;
        static_assert(KVR % AK == 0, "C1M_STF: whole chunks of staging slots");
#else
        constexpr bool stf = false;
#endif
        if (!stf && tid < AK) {
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
#ifndef C1M_QL_OLD
        if (AMODE(a) != 1) {   // the 4 q chunks per thread by cp.async (the load + store loop was not
                               // unrolled: 4 serial L2 round trips per thread)
          constexpr int NQ = 16 * 64 / CTH;
          static_assert(16 * 64 % CTH == 0, "q chunks per thread");
          const bf16* const qbase = a.q + (size_t)n * 16 * 512;
#pragma unroll
          for (int k = 0; k < NQ; ++k) {   // cp.async straight into the swizzled tile: no registers, in flight during the
            const int i = tid + CTH * k;   // record-address / key loads below; waited before the S barrier
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(su32(&sq[aswz(i >> 6, 8 * (i & 63))])),
                         "l"((const uint4*)(qbase + (size_t)(i >> 6) * 512) + (i & 63)) : "memory");
          }
          asm volatile("cp.async.commit_group;" ::: "memory");
        } else
#endif
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
        if (!stf) cbar();
        if (!(uearly && u == c)) {  // keys: loads batched, then dequant (exact) into the swizzled tile
          constexpr int NT = (AK * 36 + CTH - 1) / CTH;
          uint4 raw[NT];
          unsigned ev[NT];
          bool okv[NT];
#pragma unroll
          for (int jq = 0; jq < NT; ++jq) {
            const int i = tid + CTH * jq;
            raw[jq] = make_uint4(0u, 0u, 0u, 0u), ev[jq] = 0, okv[jq] = false;
            if (i < AK * 36) {
              const int k = i / 36, cc = i % 36;
              if (stf) {   // (C1M_STF) the slot, its scale bytes and its validity flag in the same round trip
                const uint8_t* b = a.stg[rank];
#ifdef C1M_PDD   // (deduplicated top-k: the key's record sits at its first occurrence's slot; validity from the map)
                int sl_ = n * KVR + ch * AK + k;
                if (S.umode && ch * AK + k >= WIN) {
                  const unsigned short m = S.umap[k];
                  okv[jq] = m != 0xffffu;
                  if (okv[jq]) sl_ = m;
                } else {
                  okv[jq] = __ldcg(b + STG_V + sl_) != 0;
                }
                const long long d = (long long)sl_ * STG_R;
#else
                const long long d = (long long)(n * KVR + ch * AK + k) * STG_R;
                okv[jq] = __ldcg(b + STG_V + n * KVR + ch * AK + k) != 0;
#endif
                raw[jq] = cc < 28 ? __ldcg((const uint4*)(b + d) + cc) : __ldcg((const uint4*)(b + d + 448) + (cc - 28));
                if (cc < 28) ev[jq] = __ldcg(b + d + 576 + (cc >> 2));
              } else {
                const long long d = soff[2 * k];
                okv[jq] = d >= 0;
                if (d >= 0) {
                  const uint8_t* b = sbs[k];
                  raw[jq] = cc < 28 ? __ldcg((const uint4*)(b + d) + cc) : __ldcg((const uint4*)(b + d + 448) + (cc - 28));
                  if (cc < 28) ev[jq] = __ldcg(b + soff[2 * k + 1] + (cc >> 2));
                }
              }
            }
          }
#pragma unroll
          for (int jq = 0; jq < NT; ++jq) {
            const int i = tid + CTH * jq;
            if (i >= AK * 36) continue;
            const int k = i / 36, cc = i % 36;
            if (stf && cc == 0) soff[2 * k] = okv[jq] ? 0 : -1;   // (C1M_STF: the softmax's key mask)
            if (cc < 28) {
              uint4 lo = make_uint4(0u, 0u, 0u, 0u), hi = lo;
              if (okv[jq]) {
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
              *(uint4*)&sk[aswz(k, 448 + 8 * (cc - 28))] = okv[jq] ? raw[jq] : make_uint4(0u, 0u, 0u, 0u);
            }
          }
        }
#else
        // a regular layer's first unit on this CTA: its records' addresses need no table loads (the
        // owner: the layer-start prefetch's offsets in S.uoff; a mode-3 peer: the fixed staging slots, validity read with
        // the keys), so the key loads go out with the q loads: 1 L2 round trip instead of 3 (sl -> si / tl -> ti -> keys)
        const bool fastu = !hxl && u == c && (staged || (attend && AMODE(a) != 2));
        if (!fastu && tid < AK) {
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
        constexpr int NT = (AK * 36 + CTH - 1) / CTH;
        uint4 fraw[NT];
        unsigned fev[NT];
        bool fok[NT];
        if (fastu) {
#pragma unroll
          for (int jq = 0; jq < NT; ++jq) {
            const int i = tid + CTH * jq;
            fraw[jq] = make_uint4(0u, 0u, 0u, 0u), fev[jq] = 0, fok[jq] = false;
            if (i < AK * 36) {
              const int k = i / 36, cc = i % 36, j = ch * AK + k;
              const uint8_t* b;
              long long d, scx;
              if (staged) {
                b = a.stg[rank], d = (long long)(n * KVR + j) * STG_R, scx = d + 576;
                fok[jq] = j < KVR && __ldcg(a.stg[rank] + STG_V + n * KVR + j);   // (the same line per key: merged)
              } else {
                b = S.uoffb[k], d = S.uoff[k][0], scx = S.uoff[k][1];
                fok[jq] = d >= 0;
              }
              if (staged || fok[jq]) {   // (staged: unconditional, masked below; the slot always exists)
                fraw[jq] = cc < 28 ? __ldcg((const uint4*)(b + d) + cc) : __ldcg((const uint4*)(b + d + 448) + (cc - 28));
                if (cc < 28) fev[jq] = __ldcg(b + scx + (cc >> 2));
              }
            }
          }
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
        auto store_key = [&](int k, int cc, const uint4 rv, unsigned e, bool ok) {   // dequant (exact) into the swizzled tile
          if (cc < 28) {
            uint4 lo = make_uint4(0u, 0u, 0u, 0u), hi = lo;
            if (ok) {
              const float scl = e == 0 ? 5.877471754111438e-39f : __uint_as_float(e << 23);
              const unsigned wv[4] = {rv.x, rv.y, rv.z, rv.w};
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
            *(uint4*)&sk[aswz(k, 448 + 8 * (cc - 28))] = ok ? rv : make_uint4(0u, 0u, 0u, 0u);
          }
        };
        if (!fastu) {   // the table-based records (index layers, later units, mode 2): after the soff barrier
          cbar();
#pragma unroll
          for (int jq = 0; jq < NT; ++jq) {
            const int i = tid + CTH * jq;
            fraw[jq] = make_uint4(0u, 0u, 0u, 0u), fev[jq] = 0, fok[jq] = false;
            if (i < AK * 36) {
              const int k = i / 36, cc = i % 36;
              const long long d = soff[2 * k];
              fok[jq] = d >= 0;
              if (d >= 0) {
                const uint8_t* b = sbs[k];
                fraw[jq] = cc < 28 ? __ldcg((const uint4*)(b + d) + cc) : __ldcg((const uint4*)(b + d + 448) + (cc - 28));
                if (cc < 28) fev[jq] = __ldcg(b + soff[2 * k + 1] + (cc >> 2));
              }
            }
          }
        }
#pragma unroll
        for (int jq = 0; jq < NT; ++jq) {   // one dequant for both paths
          const int i = tid + CTH * jq;
          if (i >= AK * 36) continue;
          const int k = i / 36, cc = i % 36;
          store_key(k, cc, fraw[jq], fev[jq], fok[jq]);
          if (fastu && cc == 0) soff[2 * k] = fok[jq] ? 0 : -1;   // (the softmax's key mask)
        }
#endif
#ifndef C1M_QL_OLD
        if (AMODE(a) != 1) asm volatile("cp.async.wait_all;" ::: "memory");   // (the q chunks)
#endif
        cbar();
#if !defined(C1M_TS) && !defined(C1M_PSTAMP)
        if (u == c) stamp(41);
#endif   // (timeline: unit loads + dequant done)
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
#if !defined(C1M_TS) && !defined(C1M_PSTAMP)
        if (u == c) stamp(42);
#endif   // (S partials)
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
#if !defined(C1M_TS) && !defined(C1M_PSTAMP)
        if (u == c) stamp(43);
#endif   // (softmax)
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
#if !defined(C1M_TS) && !defined(C1M_PSTAMP)
        if (u == c) stamp(44);
#endif   // (PV + partial stores)
        if (tid == 0) {
#ifdef C1M_FENCE_OLD
          __threadfence();
#endif
          red_rel(a.cnt + C_AT + n, 1u);   // (release after the barrier: see arrive)
        }
      }
      stamp(25);
      // combines: (row n, 32-dim slice sl); thread = (head tid / 16, 2 dims)
      for (int v = c; attend && v < NR * NHG * 16; v += NCTA) {
        const int n = v / (NHG * 16), hg = (v / 16) % NHG, sl = v & 15;
        const int h = tid >> 4, d = 32 * sl + 2 * (tid & 15), ub = (n * NHG + hg) * ANCH;
#ifndef C1M_CMB_OLD
        // the sink and the inverse-RoPE factors do not depend on the partials: loaded before the spin
        const float sink2 = __ldg(a.sink + (AMODE(a) == 1 ? 16 * hg : 16 * rank) + h) * 1.4426950408889634f;
        float rcc = 1.f, rsn = 0.f;
        if (d >= 448) {
          const long long pn = staged ? __ldcg((const long long*)(a.stg[rank] + STG_P) + n) : __ldg(a.pos + n);
          const float* cs = a.cos_sin + pn * 64;
          const int ip = (d - 448) >> 1;
          rcc = __ldg(cs + ip), rsn = __ldg(cs + 32 + ip);
        }
#endif
#ifndef C1M_CML
        if (tid == 0) {
          const unsigned long long t0 = tim ? gtime() : 0ull;
          spin_ge(a.cnt + C_AT + n, (unsigned)(NHG * ANCH) * ncall);
          if (tim) st_b += gtime() - t0;
        }
        cbar();
#endif
#ifndef C1M_TS
        if (v == c) stamp(45);
#endif   // (timeline: the combine's C_AT wait done)
#ifdef C1M_CMB_OLD
        const float sink2 = __ldg(a.sink + (AMODE(a) == 1 ? 16 * hg : 16 * rank) + h) * 1.4426950408889634f;
#endif
#ifdef C1M_CML
        // the units' partials as they land (Lamport, no C_AT wait), lean: pass 1 = m / l with the head's 16 lanes
        // sharing the 20 units (lane j: units j and 16 + j), the max by a 16-lane butterfly (max over the same set: exact),
        // each unit's weight on its lane (the same expression), den summed in unit order from shuffled operands; pass 2 = this
        // thread's o partials, a0 / a1 summed in unit order with the shuffled weights. Every accumulator sees the same
        // operands in the same order as the one-loop code below: bitwise
        static_assert(ANCH > 16 && ANCH <= 32, "two units per lane of a head");
        const unsigned long long t0_ = (tim && tid == 0) ? gtime() : 0ull;
        const int j16 = tid & 15;
        const bool hasB = j16 < ANCH - 16;
        const float* const amA = a.aml + (size_t)(ub + j16) * 32 + h;
        const float* const amB = a.aml + (size_t)(ub + 16 + j16) * 32 + h;
        float mA = ld_rlx_f(amA), lA = ld_rlx_f(amA + 16);
        float mB = hasB ? ld_rlx_f(amB) : -INFINITY, lB = hasB ? ld_rlx_f(amB + 16) : 0.f;
        while (__float_as_uint(mA) == CML_SENT || __float_as_uint(lA) == CML_SENT ||
               (hasB && (__float_as_uint(mB) == CML_SENT || __float_as_uint(lB) == CML_SENT))) {
          if (__float_as_uint(mA) == CML_SENT) mA = ld_rlx_f(amA);
          if (__float_as_uint(lA) == CML_SENT) lA = ld_rlx_f(amA + 16);
          if (hasB && __float_as_uint(mB) == CML_SENT) mB = ld_rlx_f(amB);
          if (hasB && __float_as_uint(lB) == CML_SENT) lB = ld_rlx_f(amB + 16);
        }
        float ml = hasB ? fmaxf(mA, mB) : mA;
#pragma unroll
        for (int o = 8; o; o >>= 1) ml = fmaxf(ml, __shfl_xor_sync(~0u, ml, o, 16));
        const float mt = fmaxf(sink2, ml);
        const float wA = mA == -INFINITY ? 0.f : exp2f(mA - mt), wB = mB == -INFINITY ? 0.f : exp2f(mB - mt);
        float den = sink2 == -INFINITY ? 0.f : exp2f(sink2 - mt);
#pragma unroll
        for (int cc = 0; cc < ANCH; ++cc) {
          const float wc = __shfl_sync(~0u, cc < 16 ? wA : wB, cc & 15, 16);
          const float lv = __shfl_sync(~0u, cc < 16 ? lA : lB, cc & 15, 16);
          den += wc * lv;
        }
        float a0 = 0.f, a1 = 0.f;
        {
          float* const ap0 = a.apart + ((size_t)ub * 16 + h) * 512 + d;   // unit cc: + cc * 16 * 512
          float2 ov[ANCH];
#pragma unroll
          for (int cc = 0; cc < ANCH; ++cc) {
            const unsigned long long o_ = ld_lb(ap0 + (size_t)cc * 16 * 512);
            ov[cc] = make_float2(__uint_as_float((unsigned)o_), __uint_as_float((unsigned)(o_ >> 32)));
          }
          for (;;) {
            bool pend = false;
#pragma unroll
            for (int cc = 0; cc < ANCH; ++cc) pend |= __float_as_uint(ov[cc].x) == CML_SENT || __float_as_uint(ov[cc].y) == CML_SENT;
            if (!pend) break;
#pragma unroll
            for (int cc = 0; cc < ANCH; ++cc)
              if (__float_as_uint(ov[cc].x) == CML_SENT || __float_as_uint(ov[cc].y) == CML_SENT) {
                const unsigned long long o_ = ld_lb(ap0 + (size_t)cc * 16 * 512);
                ov[cc] = make_float2(__uint_as_float((unsigned)o_), __uint_as_float((unsigned)(o_ >> 32)));
              }
          }
#pragma unroll
          for (int cc = 0; cc < ANCH; ++cc) {
            const float wc = __shfl_sync(~0u, cc < 16 ? wA : wB, cc & 15, 16);
            a0 += wc * ov[cc].x, a1 += wc * ov[cc].y;
          }
          // this thread is each of these apart elements' only reader: the sentinel back for the next layer's units
          const unsigned long long sn_ = (unsigned long long)CML_SENT | ((unsigned long long)CML_SENT << 32);
#pragma unroll
          for (int cc = 0; cc < ANCH; ++cc) *(unsigned long long*)(ap0 + (size_t)cc * 16 * 512) = sn_;
        }
        if (tim && tid == 0) st_b += gtime() - t0_;
#else
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
#endif
        const float inv = (den > 0.f && mt < INFINITY) ? 1.f / den : 0.f;
        bf16 b0 = __float2bfloat16_rn(a0 * inv), b1 = __float2bfloat16_rn(a1 * inv);
        if (d >= 448) {  // inverse RoPE (comm_c1 inv_rope4 / dense.cu rope_pair<true>)
#ifdef C1M_CMB_OLD
          const long long pn = staged ? __ldcg((const long long*)(a.stg[rank] + STG_P) + n) : __ldg(a.pos + n);
          const float* cs = a.cos_sin + pn * 64;
          const int ip = (d - 448) >> 1;
          const float cc = __ldg(cs + ip), sn = __ldg(cs + 32 + ip), e = __bfloat162float(b0), od = __bfloat162float(b1);
#else
          const float cc = rcc, sn = rsn, e = __bfloat162float(b0), od = __bfloat162float(b1);
#endif
          b0 = __float2bfloat16_rn(__fadd_rn(__fmul_rn(e, cc), __fmul_rn(od, sn)));
          b1 = __float2bfloat16_rn(__fsub_rn(__fmul_rn(od, cc), __fmul_rn(e, sn)));
        }
        *(unsigned*)&S.ot[h][2 * (tid & 15)] = (unsigned)__bfloat16_as_ushort(b0) | ((unsigned)__bfloat16_as_ushort(b1) << 16);
        cbar();
#ifndef C1M_TS
        if (v == c) stamp(46);
#endif   // (combine math)
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
#ifdef C1S_OGMC
      // og by ONE bulk multicast per cluster instead of 128 CTAs reading the same 32 KB: every CTA arms its own
      // mbarrier for the NR rows; the cluster's rank-0 CTA waits C_O (the others do not poll), then copies group gi's rows
      // into every CTA's image rows (same SMEM offsets); the image path below reads them from SMEM (same values: bitwise)
      if (!landed) {
        const uint32_t ogm = su32(&S.ogmb);
        const unsigned ph = S.ogph;
        if (tid == 0) {
          mb_expect(ogm, (unsigned)NR * 8192u);
          if (cl_rank() == 0) {
            const unsigned long long t0_ = tim ? gtime() : 0ull;
            spin_ge(a.cnt + C_O, 132u * ncall);
            if (tim) st_b += gtime() - t0_;
            asm volatile("fence.proxy.async.global;" ::: "memory");   // og (generic stores) -> the bulk copy (async proxy)
            for (int n = 0; n < NR; ++n)
              bulk_mc(img + n * IMS, a.og + (size_t)n * 8192 + 4096 * (c >> 6), 8192u, ogm, (unsigned short)((1u << C1S_CLS) - 1u));
          }
        }
        mb_wait(ogm, ph);
      }
#else
      if (!landed) waitc(C_O, 132u * ncall);  // og = this GPU's attention output
#endif
#ifdef C1M_TS
      stamp(46);   // (diagnostic build: every CTA's o out)
#endif
    }
#ifdef C1M_CML
    // C_O is complete here (this CTA waited for it, or its cluster's rank 0 did before the og multicast it waited
    // for): every combine has read this CTA's unit's m / l; the sentinel back for the next layer
    if (!landed && AMODE(a) == 3 && c < NR * ANCH && tid < 32) a.aml[(size_t)c * 32 + tid] = __uint_as_float(CML_SENT);
#endif
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
#ifdef C1S_OGMC
            v[j] = *(const uint4*)(imgp + n * IMS + 16 * c8);   // (landed in this CTA's image row by the multicast)
            (void)src;
#else
            v[j] = __ldcg(src);
#endif
          }
        }
      }
      if (B16(a, woaT16, 1) || B16(a, woa8, 5)) {   // bf16 wo_a (srv8x: or its exact in-register dequant): the natural bf16 image
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
#ifdef C1S_OGMC
    if (FULL && !landed && tid == 0) S.ogph ^= 1u;   // (read again only in a later layer, after many barriers)
#endif
    stamp(26);
#define C1M_DPART 1
#include C1M_DENSE_F1
#undef C1M_DPART
#ifdef C1M_TS
    stamp(45);   // (diagnostic build: the wo_a epilogue done, before its C_Z arrival)
#endif
#if defined(C1M_ZG)
    cbar();
    if (tid == 0) red_rlx(a.cnt + C_Z, 1u);
#elif defined(C1M_ZL)
    cbar();
#else
    arrive(C_Z);
#endif
  }
  stamp(1);
#if defined(C1M_NWP)
    if constexpr (FULL) {   // H's norm weights into the scratch now (every CTA; free from here to the H RMSNorm)
      float* const nws = reinterpret_cast<float*>(imgp + NRM * IMS);
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int c8 = tid + CTH * j;
        if (c8 < 640) {
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(su32(nws + 8 * c8)), "l"(a.nw + 8 * c8) : "memory");
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(su32(nws + 8 * c8 + 4)), "l"(a.nw + 8 * c8 + 4) : "memory");
        }
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
#endif
  if constexpr (FULL)
    if (c >= NWK && c - NWK < NR) {   // the attention boundary's mixes of row c - 128 (needed by the FFN boundary)
#if !defined(C1M_MX_OLD) && !defined(C1M_LBM)
      waitc(C_MXA, 132u * ncall);
#endif
      for (int n = c - NWK; n < NR; n += NCTA - NWK) {   // (NRM > 4: rows 4, 5 second; arrive's barrier ends each pass)
        mixes(n, a.mixp_a, a.ascale, a.abase, a.apre_o, a.apost_o, a.acomb_o);
        arrive(C_MIXA);
      }
      stamp(22);
    }
  // ---------------------------------------------------------------------------------------------- F2 wo_b
#if defined(C1M_ZG)
  if (tid == 0) spin_ge_nf(a.cnt + C_Z, 128u * ncall);
  cbar();
#elif !defined(C1M_ZL)
  waitc(C_Z, 128u * ncall);
#endif
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
#ifdef C1M_ZL
      for (int n = 0; n < NRM; ++n)
        v[n] = n < NR ? ld_rlx_v4((const uint4*)(a.z + (size_t)(par * NRM + n) * 2048) + tid) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int n = 0; n < NRM; ++n)   // (batched polls: every row's first load in flight before any check)
        if (n < NR)
          while (has_s16(v[n])) v[n] = ld_rlx_v4((const uint4*)(a.z + (size_t)(par * NRM + n) * 2048) + tid);
#else
      for (int n = 0; n < NRM; ++n) v[n] = n < NR ? __ldcg((const uint4*)(a.z + (size_t)n * 2048) + tid) : make_uint4(0u, 0u, 0u, 0u);
#endif
      if (B16(a, wob16, 2) || B16(a, wob8, 6)) {   // bf16 wo_b (srv8x: or its exact in-register dequant): the natural bf16 image
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
#define C1M_DPART 1
#include C1M_DENSE_F2
#undef C1M_DPART
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
#ifdef C1S_GPF
    // the G slice's inputs that do not depend on y, fetched now (their L2 round trips under AR #1's
    // polls, not after them): the X streams of this CTA's columns as raw bf16 into S.red (free from F2's reduction to the
    // FFN mixes / expert phase; bslice reads them back: the same values), and the attention boundary's mixes (C_MIXA:
    // the service CTAs arrived at wo_a time) by threads 128.. (not the AR #1 pollers 0 .. NR * nq - 1)
    {
      const int ch = ncols / 8;   // 16-B chunks per (row, stream)
      for (int i = tid; i < NR * HC * ch; i += CTH) {
        const int nc = i / ch, k = i % ch;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(su32(S.red) + (uint32_t)(nc * 48 + 8 * k) * 2u),
                     "l"(a.X + (size_t)nc * D + col0 + 8 * k) : "memory");
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
      static_assert(sizeof(S.red) >= NRM * HC * 48 * 2, "C1S_GPF: X staging in S.red");
    }
    if constexpr (FULL) {
      waitc(C_MIXA, (unsigned)NR * ncall);
      const int tq = tid - 128;
      if (tq >= 0 && tq < NR * 24) {
        const int n = tq / 24, k = tq % 24;
        if (k < 4) S.post[n][k] = __ldcg(a.apost_o + n * 4 + k);
        else if (k < 20) S.comb[n][k - 4] = __ldcg(a.acomb_o + n * 16 + k - 4);
        else S.lag[n][k - 20] = __ldcg(a.apre_o + n * 4 + k - 20);
      }
      static_assert(!FULL || CTH >= 128 + NRM * 24, "C1S_GPF: mix loads on threads 128..");
    }
#endif
    float* mine = a.ar1[rank] + (size_t)par * W * NRM * D;
    for (int i = tid; i < NR * nq; i += CTH) {
      const int n = i / nq, q = i % nq;
#ifndef C1M_ARP
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
#else
      // the three peers' slots polled
      // together: all loads first, re-polls only where the sentinel is
      // still there (one round trip when the data has landed; the same values and sum order)
      float4 v[W];
      uint4 u[W];
      const unsigned long long t0_ = (tim && tid == 0) ? gtime() : 0ull;
#pragma unroll
      for (int s = 0; s < W; ++s)
        if (s != rank) u[s] = ld_vol4(mine + ((size_t)s * NRM + n) * D + col0 + 4 * q);
#pragma unroll
      for (int s = 0; s < W; ++s) {
        if (s == rank) {
          v[s] = *(const float4*)&S.yp[n][4 * q];
          continue;
        }
        float* sl = mine + ((size_t)s * NRM + n) * D + col0 + 4 * q;
        while (u[s].x == SENT || u[s].y == SENT || u[s].z == SENT || u[s].w == SENT) u[s] = ld_vol4(sl);
        v[s] = *reinterpret_cast<float4*>(&u[s]);
        *(uint4*)sl = make_uint4(SENT, SENT, SENT, SENT);
      }
      if (tim && tid == 0) st_b += gtime() - t0_;
#endif
      float4 r = v[0];
#pragma unroll
      for (int s = 1; s < W; ++s) r.x += v[s].x, r.y += v[s].y, r.z += v[s].z, r.w += v[s].w;
      const bf16 b0 = __float2bfloat16_rn(r.x), b1 = __float2bfloat16_rn(r.y), b2 = __float2bfloat16_rn(r.z),
                 b3 = __float2bfloat16_rn(r.w);
      S.ybf[n][4 * q] = b0, S.ybf[n][4 * q + 1] = b1, S.ybf[n][4 * q + 2] = b2, S.ybf[n][4 * q + 3] = b3;
      bf16* yo = a.y + (size_t)n * D + col0 + 4 * q;
      yo[0] = b0, yo[1] = b1, yo[2] = b2, yo[3] = b3;
    }
#ifdef C1S_GPF
    asm volatile("cp.async.wait_all;" ::: "memory");   // (the X staging, issued before the polls; the barrier publishes it)
#endif
    cbar();
  }
  stamp(4);
  // ---------------------------------------------------------------------------------------------- G FFN boundary slice
#ifdef C1S_GPF
  bslice([&](int n, int j, int col) { return __bfloat162float(S.ybf[n][j]); }, a.X, a.xm2, a.inb, a.ssp, a.mixp, C_SS, C_MXG, 31,
         false, true);
#else
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
  bslice([&](int n, int j, int col) { return __bfloat162float(S.ybf[n][j]); }, a.X, a.xm2, a.inb, a.ssp, a.mixp, C_SS, C_MXG, 31);
#endif
  stamp(5);
  // ---------------------------------------------------------------------------------------------- H RMSNorm -> f
  waitc(C_SS, 132u * ncall);
  stamp(6);
#ifdef C1M_RKS
  // split-K router: the RMSNorm's loads +
  // rr, this CTA's f columns (the same statements as the image's), their partial logits for all 384 experts (stored as
  // polled data), the full image while they propagate, then CTA c < 128 reduces its experts c, c + 128, c + 256 over the
  // 132 partials in a fixed order and publishes them through C1M_RTL's polled logits
  float* const rkw = a.rkp + (size_t)par * NRM * NE * NCTA;
  float* const rko = a.rkp + (size_t)(par ^ 1u) * NRM * NE * NCTA;
  {
#ifdef C1M_NWP
    rnorm(a.ssp, a.inb, a.nw, a.fout, true, true, true, 1);
#else
    rnorm(a.ssp, a.inb, a.nw, a.fout, true, true, false, 1);
#endif
    const float* const nws = reinterpret_cast<const float*>(imgp + NRM * IMS);
    for (int i = tid; i < NR * ncols; i += CTH) {
      const int n = i / ncols, j = i % ncols, col = col0 + j;
      const float xin = __bfloat162float(*reinterpret_cast<const bf16*>(imgp + n * IMS + 2 * col));
      S.fsl[n][j] = __bfloat162float(__float2bfloat16_rn((xin * S.rr[n]) * nws[col]));
    }
    cbar();
    for (int hs = 0; hs < 2; ++hs) {
      const uint32_t st = cwait();
      if (tid < 192) {
        const int e = 192 * hs + tid;
        const char* const gr = ring + (st - ring_u) + tid * 32;
        float acc[NRM] = {0.f, 0.f, 0.f, 0.f};
        for (int j8 = 0; j8 < ncols / 8; ++j8) {
          const uint4 gv = *reinterpret_cast<const uint4*>(gr + (j8 >> 1) * 6144 + (j8 & 1) * 16);
          const unsigned gu[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float g0 = __uint_as_float(gu[q] << 16), g1 = __uint_as_float(gu[q] & 0xffff0000u);
#pragma unroll
            for (int n = 0; n < NRM; ++n) {
              if (n >= NR) break;
              acc[n] = fmaf(g0, S.fsl[n][8 * j8 + 2 * q], acc[n]);
              acc[n] = fmaf(g1, S.fsl[n][8 * j8 + 2 * q + 1], acc[n]);
            }
          }
        }
#pragma unroll
        for (int n = 0; n < NRM; ++n) {
          if (n >= NR) break;
          st_rlx_f(rkw + ((size_t)n * NE + e) * NCTA + c, acc[n]);
          rko[((size_t)n * NE + e) * NCTA + c] = __uint_as_float(RTL_SENT);
        }
      }
      crel();
    }
    rnorm(a.ssp, a.inb, a.nw, a.fout, true, true, true, 2);   // the full image (+ rowp, the engine's f copy)
  }
#elif defined(C1M_FONE_H)
#ifdef C1M_NWP
  rnorm(a.ssp, a.inb, a.nw, a.fout, true, true, true);
#else
  rnorm(a.ssp, a.inb, a.nw, a.fout, true, true);   // the f rows' engine copy written once per chunk by CTA
#endif
                                                   // c8 % 132, as A2's aout; CTA 0 no longer writes all 40 KB alone)
#else
  rnorm(a.ssp, a.inb, a.nw, c == 0 ? a.fout : nullptr);
#endif
  stamp(20);
  if (c == 0 && tid < NR) a.ptok[tid] = S.pv[tid];
  stamp(7);
  // ---------------------------------------------------------------------------------------------- H router rows
#ifdef C1M_RKS
  if (c < NWK) {   // (C1M_RKS) the reduce: warp w -> values v = w, w + 8 of (r = expert c + 128 r, row n)
    float* const rtw_ = a.rtp + (size_t)par * NRM * NE;
    float* const rto_ = a.rtp + (size_t)(par ^ 1u) * NRM * NE;
    for (int v = w; v < 3 * NR; v += NCW) {
      const int r = v / NR, n = v % NR, e = c + r * NWK;
      const float* const pp = rkw + ((size_t)n * NE + e) * NCTA;
      float x[5];
#pragma unroll
      for (int k = 0; k < 5; ++k) x[k] = lane + 32 * k < NCTA ? ld_rlx_f(pp + lane + 32 * k) : 0.f;
#pragma unroll
      for (int k = 0; k < 5; ++k)
        if (lane + 32 * k < NCTA)
          while (__float_as_uint(x[k]) == RTL_SENT) x[k] = ld_rlx_f(pp + lane + 32 * k);
      float sm = (((x[0] + x[1]) + x[2]) + x[3]) + x[4];
#pragma unroll
      for (int o = 16; o; o >>= 1) sm += __shfl_xor_sync(~0u, sm, o);
      if (lane == 0) st_rlx_f(rtw_ + n * NE + e, sm), rto_[n * NE + e] = __uint_as_float(RTL_SENT);
    }
  }
  if (false) {
#endif
#if defined(C1S_RTS2) && (!defined(C1M_RTL) || !defined(C1M_LBM) || defined(C1M_RT1) || defined(C1M_RKS) || defined(C1M_RTSC) || defined(C1M_SC_OLD))
#error "C1S_RTS2: RTL + LBM path only"
#endif
#if defined(C1M_RTL) && defined(C1S_RTS2)
  // the writer publishes {logit, score} as ONE 8-B Lamport entry (score = sqrt(softplus(logit)), the
  // readers' own statement): the 132 readers no longer compute 1,536 scores each; low word = the logit (RTL_SENT: not
  // written), single-copy atomic with the score
  static_assert(sizeof(unsigned long long) == 8, "");
  float* const rtw = a.rtp + (size_t)par * NRM * NE * 2;
  float* const rto = a.rtp + (size_t)(par ^ 1u) * NRM * NE * 2;
#define C1M_RT_STORE(n_, ei_, x_)                                                                           \
  do {                                                                                                      \
    const float xx_ = (x_);                                                                                 \
    const float sv_ = sqrtf(xx_ > 20.f ? xx_ : log1pf(expf(xx_)));                                          \
    st_lb(rtw + 2 * ((n_) * NE + (ei_)), xx_, __float_as_uint(sv_));                                        \
    rto[2 * ((n_) * NE + (ei_))] = __uint_as_float(RTL_SENT);                                               \
  } while (0)
#elif defined(C1M_RTL)
  float* const rtw = a.rtp + (size_t)par * NRM * NE;          // (C1M_RTL) this layer's logits (parity par), polled by
  float* const rto = a.rtp + (size_t)(par ^ 1u) * NRM * NE;   // the readers; the other parity is reset to RTL_SENT here
#define C1M_RT_STORE(n_, ei_, x_) (st_rlx_f(rtw + (n_) * NE + (ei_), (x_)), rto[(n_) * NE + (ei_)] = __uint_as_float(RTL_SENT))
#else
#define C1M_RT_STORE(n_, ei_, x_) (a.rtp[(n_) * NE + (ei_)] = (x_))
#endif
#if defined(C1S_RT1T)
  // the CTA's 3 router rows in ONE pass, split by TOKEN row: warps 4g .. 4g + 3 take rows n = 2g, 2g + 1
  // of all three experts (c, c + 128 in the first stage, c + 256 in the second). Per (expert, row) thread lt = tid & 127
  // sums the same chunks q = lt, lt + 128, .. with the same FMA order, warp 4g + j covers the same lanes lt = 32 j .. as warp
  // 4r + j did, its butterfly and the 4-warp sum ((w0 + w1) + w2) + w3 are the old ones: BITWISE, with half the shared
  // loads (3 gate + 2 image chunks per q instead of 1 + 4, twice) and one barrier round instead of two
  static_assert(NE == 3 * NWK && NRM == 4 && NCW == 8, "C1S_RT1T: 3 router rows per worker CTA, 4 rows, 8 warps");
  if (c < NWK) {
    const uint32_t st0 = cwait(), st1 = cwait1();
    const int gq = w >> 2, lt = tid & 127, n0 = 2 * gq;
    float acc[3][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
    if (n0 < NR) {
      const bool two = n0 + 1 < NR;
      for (int q = lt; q < D / 8; q += 128) {
        const uint4 w0 = lds128(st0 + 16 * q), w1 = lds128(st0 + D * 2 + 16 * q), w2 = lds128(st1 + 16 * q);
        const uint4 f0 = lds128(img + n0 * IMS + 16 * q);
        const uint4 f1 = two ? lds128(img + (n0 + 1) * IMS + 16 * q) : make_uint4(0u, 0u, 0u, 0u);
        const unsigned wu[3][4] = {{w0.x, w0.y, w0.z, w0.w}, {w1.x, w1.y, w1.z, w1.w}, {w2.x, w2.y, w2.z, w2.w}};
        const unsigned fu[2][4] = {{f0.x, f0.y, f0.z, f0.w}, {f1.x, f1.y, f1.z, f1.w}};
#pragma unroll
        for (int r = 0; r < 3; ++r)
#pragma unroll
          for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              acc[r][t] = fmaf(__uint_as_float(wu[r][k] << 16), __uint_as_float(fu[t][k] << 16), acc[r][t]);
              acc[r][t] = fmaf(__uint_as_float(wu[r][k] & 0xffff0000u), __uint_as_float(fu[t][k] & 0xffff0000u), acc[r][t]);
            }
      }
    }
#pragma unroll
    for (int r = 0; r < 3; ++r)
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        float m = acc[r][t];
#pragma unroll
        for (int o = 16; o; o >>= 1) m += __shfl_xor_sync(~0u, m, o);
        if (lane == 0) {
          if (r < 2) S.wmax[w][2 * r + t] = m;
          else S.rt2[w >> 1][2 * (w & 1) + t] = m;
        }
      }
    cbar();
    if (tid < 3 * NR) {
      const int r = tid / NR, n = tid % NR, g2 = n >> 1, t = n & 1;
      float pw[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int ww = 4 * g2 + j;
        pw[j] = r < 2 ? S.wmax[ww][2 * r + t] : S.rt2[ww >> 1][2 * (ww & 1) + t];
      }
      const float x = ((pw[0] + pw[1]) + pw[2]) + pw[3];
      C1M_RT_STORE(n, c + r * NWK, x);
    }
    crel();
    crel();
    cbar();
  }
#elif defined(C1M_RT1)
  // the CTA's 3 router rows in ONE pass: warps 0-3 row c then row c + 256 (the second
  // stage), warps 4-7 row c + 128; one barrier. Every row's per-thread FMA order, butterfly and 4-warp sum are the old ones
  static_assert(NE == 3 * NWK, "C1M_RT1: 3 router rows per worker");
  if (c < NWK) {
    const uint32_t st0 = cwait(), st1 = cwait1();
    const int rs = w >> 2, lt = tid & 127;
    float sv[NRM] = {0.f, 0.f, 0.f, 0.f}, sv2[NRM] = {0.f, 0.f, 0.f, 0.f};
    auto rowdot = [&](uint32_t base, float (&acc)[NRM]) {
      for (int q = lt; q < D / 8; q += 128) {
        const uint4 wv = lds128(base + 16 * q);
        const unsigned wu[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int n = 0; n < NRM; ++n) {
          if (n >= NR) break;
          const uint4 fv = lds128(img + n * IMS + 16 * q);
          const unsigned fu[4] = {fv.x, fv.y, fv.z, fv.w};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            acc[n] = fmaf(__uint_as_float(wu[k] << 16), __uint_as_float(fu[k] << 16), acc[n]);
            acc[n] = fmaf(__uint_as_float(wu[k] & 0xffff0000u), __uint_as_float(fu[k] & 0xffff0000u), acc[n]);
          }
        }
      }
    };
    rowdot(st0 + rs * D * 2, sv);
    if (rs == 0) rowdot(st1, sv2);
#pragma unroll
    for (int n = 0; n < NRM; ++n) {
      float m = sv[n];
#pragma unroll
      for (int o = 16; o; o >>= 1) m += __shfl_xor_sync(~0u, m, o);
      if (lane == 0) S.wmax[w][n] = m;
      if (rs == 0) {
        float m2 = sv2[n];
#pragma unroll
        for (int o = 16; o; o >>= 1) m2 += __shfl_xor_sync(~0u, m2, o);
        if (lane == 0) S.rt2[w][n] = m2;
      }
    }
    cbar();
    if (tid < 3 * NR) {
      const int r = tid / NR, n = tid % NR;
      const float x = r < 2 ? ((S.wmax[4 * r][n] + S.wmax[4 * r + 1][n]) + S.wmax[4 * r + 2][n]) + S.wmax[4 * r + 3][n]
                            : ((S.rt2[0][n] + S.rt2[1][n]) + S.rt2[2][n]) + S.rt2[3][n];
      C1M_RT_STORE(n, c + r * NWK, x);
    }
    crel();
    crel();
    cbar();
  }
#else
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
      if (r < nrw) {
        const float x = ((S.wmax[4 * r][n] + S.wmax[4 * r + 1][n]) + S.wmax[4 * r + 2][n]) + S.wmax[4 * r + 3][n];
        const int ei = e + r * NWK;
        C1M_RT_STORE(n, ei, x);
#ifdef C1M_RTSC
        // the expert's score and
        // selection key here, once (the same statements as the top-6 below,
        // which then only loads them): rtp = [logits | scores | keys], each [NRM][NE]
        const float sv = sqrtf(x > 20.f ? x : log1pf(expf(x)));
        a.rtp[NRM * NE + n * NE + ei] = sv;
        a.rtp[2 * NRM * NE + n * NE + ei] = sv + __ldg(a.gate_b + ei);
#endif
      }
    }
    crel();
    cbar();
  }
#endif
#ifdef C1M_RKS
  }   // (if (false): the router rows' pass, replaced by the split-K reduce)
#endif
#undef C1M_RT_STORE
  stamp(23);
#ifndef C1M_RTL
  arrive(C_RT);
#endif
#ifdef C1M_TS
  stamp(43);   // (diagnostic build: the C_RT arrival done)
#endif
#ifdef C1M_DOTS_WIN
  bdots(a.mixp, C_MXG, 31);   // (the G slice's mix dots in the router wait window)
#endif
#ifndef C1S_SVC
  if (c >= NWK && c - NWK < NR) {   // service CTA 128 + n (no router rows): the FFN mixes of row n, after its C_RT arrival
#if !defined(C1M_MX_OLD) && !defined(C1M_LBM)
    waitc(C_MXG, 132u * ncall);
#endif
    for (int n = c - NWK; n < NR; n += NCTA - NWK) {   // (NRM > 4: rows 4, 5 second; arrive's barrier ends each pass)
      mixes(n, a.mixp, a.fscale, a.fbase, a.fpre, a.fpost, a.fcomb);
      arrive(C_MIXF);
    }
    stamp(21);
  }
#endif
  // f image: natural bf16 -> f16 x 2^p in P8 order (in place)
  auto fimage = [&]() {
    for (int i = tid; i < NR * 640; i += CTH) {
      const int n = i / 640, c8 = i % 640;
      uint4* q = (uint4*)(imgp + n * IMS + 16 * c8);
      *q = p8_f16(*q, pow2i(S.pv[n]));
    }
  };
#if !(defined(C1M_FIMG) && defined(C1M_RTL))
  fimage();
#endif
  stamp(8);
#ifndef C1M_RTL
  waitc(C_RT, 132u * ncall);
#endif
  stamp(9);
  // ---------------------------------------------------------------------------------------------- top-6 (redundant)
  // FULL: the scores (softplus, sqrt: latency-bound math) by all consumer threads into the attention scratch (free from
  // the router to the combine), then one warp per row selects; the same values as one warp per row computing them
  constexpr bool SC8 = FULL && ATTX >= 2 * NRM * NE * 4;
  float* const scS = reinterpret_cast<float*>(imgp + NRM * IMS);
  float* const selS = scS + NRM * NE;
#if defined(C1M_RTL) && (defined(C1M_SC_OLD) || defined(C1M_RTSC))
#error "C1M_RTL: the default scores path only"
#endif
#ifdef C1M_RTL
  static_assert(SC8, "C1M_RTL: the FULL scores path (C1M_ONLY_FULL builds)");
#endif
  if constexpr (SC8) {
#ifdef C1M_SC_OLD
    for (int v = tid; v < NR * NE; v += CTH) {
      const int e = v % NE;
      const float x = __ldcg(a.rtp + v);
      if (c == 0) a.logits[v] = x;
      const float sv = sqrtf(x > 20.f ? x : log1pf(expf(x)));
      scS[v] = sv, selS[v] = sv + __ldg(a.gate_b + e);
    }
#elif defined(C1M_RTSC)
    // the router CTAs computed the scores / keys: loads only (logits: CTA 0's copy to the engine)
    constexpr int NV = (NRM * NE + CTH - 1) / CTH;
    float sv_[NV], kv_[NV], xv_[NV];
#pragma unroll
    for (int k = 0; k < NV; ++k) {
      const int v = tid + CTH * k;
      const int n = v / NE, e = v % NE;
      sv_[k] = v < NR * NE ? __ldcg(a.rtp + NRM * NE + n * NE + e) : 0.f;
      kv_[k] = v < NR * NE ? __ldcg(a.rtp + 2 * NRM * NE + n * NE + e) : 0.f;
      xv_[k] = (c == 0 && v < NR * NE) ? __ldcg(a.rtp + v) : 0.f;
    }
#pragma unroll
    for (int k = 0; k < NV; ++k) {
      const int v = tid + CTH * k;
      if (v < NR * NE) {
        if (c == 0) a.logits[v] = xv_[k];
        scS[v] = sv_[k], selS[v] = kv_[k];
      }
    }
#else
#if defined(C1M_RTL) && defined(C1S_RTS2)
    // {logit, score} entries: loads, f image, batched re-polls, then only the key (score + bias)
    constexpr int NV = (NRM * NE + CTH - 1) / CTH;
    unsigned long long xu[NV];
    float bv[NV];
    const float* const rtr = a.rtp + (size_t)par * NRM * NE * 2;
#pragma unroll
    for (int k = 0; k < NV; ++k) {
      const int v = tid + CTH * k;
      xu[k] = v < NR * NE ? ld_lb(rtr + 2 * v) : 0ull;
      bv[k] = v < NR * NE ? __ldg(a.gate_b + v % NE) : 0.f;
    }
#ifdef C1M_FIMG
    fimage();
#endif
    for (;;) {
      bool pend = false;
#pragma unroll
      for (int k = 0; k < NV; ++k) pend |= tid + CTH * k < NR * NE && (unsigned)xu[k] == RTL_SENT;
      if (!pend) break;
#pragma unroll
      for (int k = 0; k < NV; ++k) {
        const int v = tid + CTH * k;
        if (v < NR * NE && (unsigned)xu[k] == RTL_SENT) xu[k] = ld_lb(rtr + 2 * v);
      }
    }
#pragma unroll
    for (int k = 0; k < NV; ++k) {
      const int v = tid + CTH * k;
      if (v < NR * NE) {
        const float x = __uint_as_float((unsigned)xu[k]), sv = __uint_as_float((unsigned)(xu[k] >> 32));
        if (c == 0) a.logits[v] = x;
        scS[v] = sv, selS[v] = sv + bv[k];
      }
    }
    if (false) {
#endif
    // all of this thread's logit / bias loads first (one L2 round trip, not one per value)
    constexpr int NV = (NRM * NE + CTH - 1) / CTH;
    float xv[NV], bv[NV];
#pragma unroll
    for (int k = 0; k < NV; ++k) {
      const int v = tid + CTH * k;
#ifdef C1M_RTL
      xv[k] = v < NR * NE ? ld_rlx_f(a.rtp + (size_t)par * NRM * NE + v) : 0.f;
#else
      xv[k] = v < NR * NE ? __ldcg(a.rtp + v) : 0.f;
#endif
      bv[k] = v < NR * NE ? __ldg(a.gate_b + v % NE) : 0.f;
    }
#if defined(C1M_FIMG) && defined(C1M_RTL)
    fimage();   // the f image conversion while this thread's logit loads are in flight
#endif
#ifdef C1M_RTL
    // Lamport: poll each logit until its writer's store landed (no C_RT counter, no
    // release fence on the writers); the values are the same
#pragma unroll
    for (int k = 0; k < NV; ++k) {
      const int v = tid + CTH * k;
      if (v < NR * NE)
        while (__float_as_uint(xv[k]) == RTL_SENT) xv[k] = ld_rlx_f(a.rtp + (size_t)par * NRM * NE + v);
    }
#endif
#pragma unroll
    for (int k = 0; k < NV; ++k) {
      const int v = tid + CTH * k;
      if (v < NR * NE) {
        const float x = xv[k];
        if (c == 0) a.logits[v] = x;
        const float sv = sqrtf(x > 20.f ? x : log1pf(expf(x)));
        scS[v] = sv, selS[v] = sv + bv[k];
      }
    }
#if defined(C1M_RTL) && defined(C1S_RTS2)
    }   // (if (false): the old reader path)
#endif
#endif
    cbar();
  }
#if !defined(C1M_T6_OLD) && (C1M_NRM * C1M_TOPK <= 32)
  if (tid < MAXX * NRM) S.xw[tid / NRM][tid % NRM] = 0.f, S.xk[tid / NRM][tid % NRM] = -1;   // (read after 2 barriers)
#endif
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
      if (bi < NE && (bi & 31) == lane) taken |= 1u << (bi >> 5);   // (guard: bi = 1 << 30 when no finite key is left)
      float s_r = 0.f;
#pragma unroll
      for (int i = 0; i < NEW; ++i)
        if (i * 32 + lane == bi) s_r = sc[i];
      pick[r] = __shfl_sync(0xffffffffu, s_r, bi & 31);
    }
#ifdef C1S_T6W
    {   // the six weights on lanes 0..5 at once (pick / pid are warp-uniform; every lane forms the same sum
        // in the same order as lane 0 did): one divide on the critical lane instead of six, the same values and stores
      float sum = 0.f;
#pragma unroll
      for (int r = 0; r < TOPK; ++r) sum += pick[r];
      if (lane < TOPK) {
        float pr = pick[0];
        int pd = pid[0];
#pragma unroll
        for (int r = 1; r < TOPK; ++r)
          if (lane == r) pr = pick[r], pd = pid[r];
        const float wv = pr / (sum + 1e-20f) * a.route_scale;
        S.ids[n][lane] = pd, S.wts[n][lane] = wv;
        if (c == 0) a.ids[n * TOPK + lane] = pd, a.wts[n * TOPK + lane] = wv;
        if (c == 0 && a.idd) a.idd[((size_t)l * NRM + n) * TOPK + lane] = pd;   // (tests: the routing of every layer)
      }
    }
    if (false) {
#else
    if (lane == 0) {
#endif
      float sum = 0.f;
#pragma unroll
      for (int r = 0; r < TOPK; ++r) sum += pick[r];
#pragma unroll
      for (int r = 0; r < TOPK; ++r) {
        const float wv = pick[r] / (sum + 1e-20f) * a.route_scale;
        S.ids[n][r] = pid[r], S.wts[n][r] = wv;
        if (c == 0) a.ids[n * TOPK + r] = pid[r], a.wts[n * TOPK + r] = wv;
        if (c == 0 && a.idd) a.idd[((size_t)l * NRM + n) * TOPK + r] = pid[r];   // (tests: the routing of every layer)
      }
    }
  }
  cbar();
  stamp(37);
#if defined(C1M_T6_OLD) || (C1M_NRM * C1M_TOPK > 32)   // (NRM 6: 36 picks > 32 lanes: the bitmap path, the same slots)
  // distinct experts in ascending order: a 384-bit bitmap, word prefix counts, slot = rank of the id among the chosen
  if (tid < NEW) S.bm[tid] = 0u;
  if (tid < MAXX * NRM) S.xw[tid / NRM][tid % NRM] = 0.f, S.xk[tid / NRM][tid % NRM] = -1;
  cbar();
  if (tid < NR * TOPK) {
    const int e = S.ids[tid / TOPK][tid % TOPK];
    if ((unsigned)e < (unsigned)NE) atomicOr(&S.bm[e >> 5], 1u << (e & 31));   // (guard: no-pick ids have no expert)
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
    if ((unsigned)e < (unsigned)NE) {
      const int j = S.bpre[e >> 5] + __popc(S.bm[e >> 5] & ((1u << (e & 31)) - 1u));
      S.xid[j] = e, S.xw[j][n] = S.wts[n][r], S.xk[j][n] = r;
    } else {   // (guard: a no-pick has no slot; its routed output row is zero for the combine)
#pragma unroll
      for (int q = 0; q < 48 / 8; ++q) *(uint4*)&S.Ybf[n][r][8 * q] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  cbar();
  if (tid == 0) {
    if (c == 0)  // unused h slots: their arrivals, so every slot counts 36 per call
      for (int j = S.nexp; j < MAXX; ++j) red_rel(a.cnt + C_H + j, (unsigned)SH_NU);
    mb_arrive(su32(&S.route));
  }
  cbar();
#else
  // distinct experts in ascending order on warp 0 alone (2 barriers instead of 5): lane i < NR TOPK
  // = (row i / TOPK, pick i % TOPK); slot = the number of distinct chosen ids below its id (the bitmap prefix rank of the
  // old code: the same slots); S.xw / S.xk were reset before the selection
  if (w == 0) {
    const int n = lane / TOPK, r = lane % TOPK;
    const int e0 = lane < NR * TOPK ? S.ids[n][r] : 0;
    const bool vl = lane < NR * TOPK && (unsigned)e0 < (unsigned)NE;
    if (lane < NR * TOPK && !vl)
#pragma unroll
      for (int q = 0; q < 48 / 8; ++q) *(uint4*)&S.Ybf[n][r][8 * q] = make_uint4(0u, 0u, 0u, 0u);
    const int e = vl ? e0 : (1 << 20) + lane;   // (invalid lanes: distinct ids above every expert)
    const unsigned same = __match_any_sync(~0u, e);
    const bool first = vl && (__ffs(same) - 1) == lane;
    const unsigned fm = __ballot_sync(~0u, first);
    int j = 0;
#pragma unroll
    for (int k = 0; k < MAXX; ++k) {
      const int ek = __shfl_sync(~0u, e, k);
      j += ((fm >> k) & 1u) && ek < e;
    }
    if (first) S.xid[j] = e;
    if (vl) S.xw[j][n] = S.wts[n][r], S.xk[j][n] = r;
    const int nx = __popc(fm);
    if (lane == 0) S.nexp = nx;
    __syncwarp();
#ifdef C1M_RTF
    // the producer's routing signal first; CTA 0's unused-slot arrivals in parallel and relaxed (they carry
    // no data: they only keep every slot's count at 36 per call; serial red.release cost CTA 0 ~6 us per layer)
    static_assert(MAXX <= 32, "C1M_RTF: one lane per slot");
    if (lane == 0) mb_arrive(su32(&S.route));
    if (c == 0 && lane >= nx && lane < MAXX) red_rlx(a.cnt + C_H + lane, (unsigned)SH_NU);
#else
    if (lane == 0) {
      if (c == 0)  // unused h slots: their arrivals, so every slot counts 36 per call
        for (int jj = nx; jj < MAXX; ++jj) red_rel(a.cnt + C_H + jj, (unsigned)SH_NU);
      mb_arrive(su32(&S.route));
    }
#endif
  }
  cbar();
#endif
  stamp(10);
#ifdef C1S_SVC
  // the service CTAs' FFN mixes AFTER their own routing (the next layer's boundary needs them ~60 us later
  // before, they ran between the router rows and the scores and made these CTAs route ~3.5 us after the median); S.red is
  // free here on them (no routed GEMM1 items, no shared halves); the same statements
#if defined(C1M_DOTS_WIN) || defined(C1M_MX_OLD)
#error "C1S_SVC: mix path only (LBM, no DOTS_WIN)"
#endif
  if (c >= NWK && c - NWK < NR) {   // service CTA 128 + n (no router rows): the FFN mixes of row n
#if !defined(C1M_MX_OLD) && !defined(C1M_LBM)
    waitc(C_MXG, 132u * ncall);
#endif
    for (int n = c - NWK; n < NR; n += NCTA - NWK) {   // (NRM > 4: rows 4, 5 second; arrive's barrier ends each pass)
      mixes(n, a.mixp, a.fscale, a.fbase, a.fpre, a.fpost, a.fcomb);
      arrive(C_MIXF);
    }
    stamp(21);
  }
#endif
  const int nexp = S.nexp;
  // the expert phase: GEMM1 (shared halves + routed items), GEMM2 routed + shared -> S.Ybf / S.shbf
  // (C1M_X16, an X16-aware fragment: C1M_XROLE 0 = the base warps' whole phase; its 16-warp section between C1M_XENTER()
  // and C1M_XEXIT() also runs on warps 10-17 (C1M_XROLE 1, xcons below))
#define C1M_XROLE 0
#ifdef C1M_XN
#define C1M_XENTER() do { if (tid == 0) S.xit = it; xbar(); } while (0)
#define C1M_XEXIT() xbar()
#else
#define C1M_XENTER() do { } while (0)
#define C1M_XEXIT() do { } while (0)
#endif
#include C1M_MOE_CONSUME
#undef C1M_XENTER
#undef C1M_XEXIT
#undef C1M_XROLE
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
#ifndef C1M_ARP
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
#else
      // the peers' slots polled together (as AR #1): loads first, re-polls only where a -0.0 half remains
      auto landed = [](const uint4& x) {
        const unsigned uu[4] = {x.x, x.y, x.z, x.w};
        bool ok = true;
#pragma unroll
        for (int k = 0; k < 4; ++k) ok = ok && (uu[k] & 0xffffu) != 0x8000u && (uu[k] >> 16) != 0x8000u;
        return ok;
      };
      uint4 pu[W];
      const unsigned long long t0_ = (tim && tid == 0) ? gtime() : 0ull;
#pragma unroll
      for (int s = 0; s < W; ++s)
        if (s != rank) pu[s] = ld_vol4(mine + ((size_t)s * NRM + n) * D + col0 + 8 * q);
#pragma unroll
      for (int s = 0; s < W; ++s) {
        uint4 u;
        if (s == rank) {
          u = *(const uint4*)&S.part[n][8 * q];
        } else {
          bf16* sl = mine + ((size_t)s * NRM + n) * D + col0 + 8 * q;
          while (!landed(pu[s])) pu[s] = ld_vol4(sl);
          u = pu[s];
          *(uint4*)sl = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
        }
        const unsigned uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[2 * k] += __uint_as_float(uu[k] << 16), acc[2 * k + 1] += __uint_as_float(uu[k] & 0xffff0000u);
      }
      if (tim && tid == 0) st_b += gtime() - t0_;
#endif
      unsigned o[4];
#pragma unroll
      for (int k = 0; k < 4; ++k)
        o[k] = (unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(acc[2 * k])) |
               ((unsigned)__bfloat16_as_ushort(__float2bfloat16_rn(acc[2 * k + 1])) << 16);
      *(uint4*)(a.moe_out + (size_t)n * D + col0 + 8 * q) = make_uint4(o[0], o[1], o[2], o[3]);
#ifndef C1M_A1S_OLD
      *(uint4*)&S.xinb[n][8 * q] = make_uint4(o[0], o[1], o[2], o[3]);   // (the next layer's A1 reads it here)
#endif
    }
  }
  cbar();
  stamp(13);
}
#if defined(C1M_XN) && defined(C1M_XFRAG)
// (C1M_X16) warps 10-17: per layer, only the expert fragment's 16-warp section (C1M_XROLE 1). In scope for the fragment:
// a (= S.args, this layer's), al, l, nly, c, NR, rank, col0 / ncols / nt / ftb, ring / ring_u, imgp / img, S, g / t4, lane,
// w (10..17), wx = w - 2 (8..15: the expert-phase warp), tx = 32 wx + lane (256..511), par / ncall, it (from S.xit at
// C1M_XENTER()), cwait() / crel1(), xbar(); stamp() is a no-op (tim = nullptr). Read S.nexp etc. after C1M_XENTER().
template <bool FULL>
__device__ __noinline__ void xcons(const Args* __restrict__ al, const int nly, const unsigned call) {
  constexpr int NS = LayCfg<FULL>::NS, ATTX = LayCfg<FULL>::ATTX;
  extern __shared__ __align__(1024) char smem[];
  char* ring = smem;
  char* imgp = smem + NS * STAGE;
  Smem& S = *reinterpret_cast<Smem*>(smem + NS * STAGE + NRM * IMS + ATTX);
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, c = blockIdx.x;
  const int wx = w - 2, tx = 32 * wx + lane;
  const int NR = al[0].NR, rank = al[0].rank;
  int ftb, nt;
  my_tiles(c, ftb, nt);
  const int col0 = 16 * ftb, ncols = 16 * nt;
  const uint32_t ring_u = su32(ring), img = su32(imgp);
  const int g = lane >> 2, t4 = lane & 3;
  const Args& a = S.args;
  unsigned long long* const tim = nullptr;
  unsigned long long st_w = 0ull, st_b = 0ull;
  long it = 0;
  auto stamp = [&](int) {};
  auto cwait = [&]() -> uint32_t {
    const int s = (int)(it % NS);
    mb_wait(su32(&S.full[s]), (unsigned)((it / NS) & 1));
    return ring_u + s * STAGE;
  };
  auto crel1 = [&]() {
    __syncwarp();
    if (lane == 0) mb_arrive(su32(&S.empty[(int)(it % NS)]));
    ++it;
  };
  (void)tid, (void)tx, (void)rank, (void)ring_u, (void)img, (void)ftb, (void)nt, (void)col0, (void)ncols, (void)g, (void)t4;
  (void)tim, (void)st_w, (void)st_b, (void)stamp, (void)cwait, (void)crel1, (void)a, (void)NR;
  for (int l = 0; l < nly; ++l) {
    const unsigned par = (call + l) & 1u, ncall = call + l + 1u;
    (void)par, (void)ncall;
#define C1M_XROLE 1
#define C1M_XENTER() do { xbar(); it = S.xit; } while (0)
#define C1M_XEXIT() xbar()
#include C1M_MOE_CONSUME
#undef C1M_XENTER
#undef C1M_XEXIT
#undef C1M_XROLE
  }
}
#endif
#if defined(C1M_HAS_PHASES) && !defined(C1M_ONE_BODY)
template <bool FULL>
__device__ __noinline__ CState cons_ix(const Args* __restrict__ al, const int nly, const int l, CState cs) {
  cons_layer<FULL, true>(al, nly, l, cs);
  return cs;
}
#endif

template <bool FULL>
#ifndef C1M_REGCAP_THREADS
#define C1M_REGCAP_THREADS THR   // (tests: a larger value caps the registers as that block size would, at THR threads)
#endif
__global__ void __launch_bounds__(C1M_REGCAP_THREADS, 1) k_layer(const Args* __restrict__ al, const int nly) {
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
    for (int i = 0; i < NS; ++i) mb_init(su32(&S.full[i]), 1), mb_init(su32(&S.empty[i]), NCWX);
    mb_init(su32(&S.route), 1);
#ifdef C1S_OGMC
    mb_init(su32(&S.ogmb), 1);
    S.ogph = 0u;
#endif
    S.cpush = 0u;   // (C1M_CPUSH's flag: ncall >= 1 never matches before the first push)
#ifdef C1M_PDD
    S.umode = 0;    // (a launch's first layer pushes the full top-k: identity slots)
#endif
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    call_s = al[0].tick[c];
  }
  __syncthreads();
#ifdef C1S_CLS
  cl_sync_all();   // (the cluster peers' mbarriers are initialized before any multicast)
#endif
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
#ifdef C1M_SPIN_PW   // the push warp polls through the layer front next to the consumers: no CCTL per poll
        if (l > 0) spin_ge_rlx(A.cnt + C_SS, 132u * (ncall - 1u));
#else
        if (l > 0) spin_ge(A.cnt + C_SS, 132u * (ncall - 1u));
#endif
        const int njt = (l == 0 || A.push_topk) ? KVR : WIN;   // index layers inside a run: their new top-k too
        const bool hxa = A.ix != nullptr || A.eg != nullptr;   // (a phase layer)
        (void)hxa;
        auto push = [&](const int jlo, const int jhi, const bool dd) {   // the records j in [jlo, jhi) of every row (dd: C1M_PDD's
          (void)dd;                                                      // deduplicated top-k: first occurrences only)
          const long long o0 = NR > 0 ? __ldg(A.own) : -1, o1 = NR > 1 ? __ldg(A.own + 1) : -1;
          const long long o2 = NR > 2 ? __ldg(A.own + 2) : -1, o3 = NR > 3 ? __ldg(A.own + 3) : -1;
#if C1M_NRM > 4
          const long long o4 = NR > 4 ? __ldg(A.own + 4) : -1, o5 = NR > 5 ? __ldg(A.own + 5) : -1;
#endif
          // record info on lane k (this CTA's k-th record rr = c + NCTA k), then all 32 lanes copy chunks in batches of 8
          // independent loads (the copy is latency-bound otherwise: ~1 us per dependent HBM round)
          const int nj = jhi - jlo;
          static_assert((NRM * KVR + NCTA - 1) / NCTA <= 32, "push: one record per lane");
          const int nrec = (NR * nj - c + NCTA - 1) / NCTA;   // <= 20 (NRM 6: <= 30)
          long long rk = 0;
          const uint8_t* srcd = nullptr;
          const uint8_t* srcs = nullptr;
          int cp = 0;   // this lane's record is copied (valid and not one of the rows' new records)
          if (pl < nrec) {
            const int rr = c + NCTA * pl, n = rr / nj, j = jlo + rr % nj;
            rk = (long long)n * KVR + j;
            const uint8_t* b = nullptr;
            long long d = 0, sc = 0;
            const bool v = kv_rec(A, n, j, b, d, sc);
            bool fresh = false;   // the rows' own new records: inserted in this layer's A4, pushed after the insert
            if (v && j < WIN) {
              const long long sl_ = __ldg(A.si + n * A.si_rs + j);
              fresh = sl_ == o0 || sl_ == o1 || sl_ == o2 || sl_ == o3;
#if C1M_NRM > 4
              fresh = fresh || sl_ == o4 || sl_ == o5;
#endif
            }
  #pragma unroll
            for (int p = 0; p < W; ++p)
              if (C1M_PUSH_DATA(p, hxa)) A.stg[p][STG_V + rk] = v ? 1 : 0;
            cp = v && !fresh;
#ifdef C1M_PDD
            if (dd && cp) cp = S.pdd[pl] == (unsigned short)rk;   // (a duplicate: its first occurrence's slot serves it)
#endif
            if (cp) srcd = b + d, srcs = b + sc;
          }
#ifdef C1M_PSTAMP   // (DIAGNOSTIC: the push warp's top-k pass, stamps 41-44 in place of the units')
          if (jlo > 0 && pl == 0 && l == ltim && tim0) tim0[42] = gtime();   // (record info + validity stores issued)
#endif
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
#if defined(C1M_PDIAG) && C1M_PDIAG == 2   // (DIAGNOSTIC ONLY, wrong outputs: no source reads from the paged pools)
              if (false) {
#else
              if (ok[u]) {
#endif
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
#if defined(C1M_PDIAG) && C1M_PDIAG == 1   // (DIAGNOSTIC ONLY, wrong outputs: record data to the own staging only)
                  if (C1M_PUSH_DATA(p, hxa) && p == rank) {
#else
                  if (C1M_PUSH_DATA(p, hxa)) {
#endif
                    if (sc8) *(uint2*)(A.stg[p] + dst[u]) = make_uint2(x[u].x, x[u].y);
                    else *(uint4*)(A.stg[p] + dst[u]) = x[u];
                  }
              }
          }
#ifdef C1M_PSTAMP
          if (jlo > 0 && pl == 0 && l == ltim && tim0) tim0[43] = gtime();   // (copy loop issued)
#endif
        };
#ifndef C1M_PUSH_OLD
        // an index layer's window records (j < WIN) do not depend on its selection: pushed
        // BEFORE hook 5's wait on the top-k, only the top-k part after it
        const bool split = A.push_topk != 0;
        if (split) push(0, WIN, false);
#ifdef C1M_CPUSH
        if (split && hxa && A.ix != nullptr) {   // (C1M_CPUSH: the consumers pushed the top-k part; wait for their flag)
          while ((int)(*(volatile unsigned*)&S.cpush - ncall) < 0) {
          }
        } else
#elif defined(C1M_PDD)
        if (split && A.ix != nullptr) {   // (C1M_PDD: the consumers' dedupe of the new top-k; wait for this CTA's decisions)
          while ((int)(*(volatile unsigned*)&S.cpush - ncall) < 0) {
          }
          __threadfence_block();
          push(WIN, njt, true);
        } else
#endif
        {
#define C1M_HOOK 5
#include C1M_HOOKS
#undef C1M_HOOK
#ifdef C1M_PSTAMP
        if (split && pl == 0 && l == ltim && tim0) tim0[41] = gtime();   // (hook 5 returned: the selection is final)
#endif
        push(split ? WIN : 0, njt, false);
        }
#else
#define C1M_HOOK 5
#include C1M_HOOKS
#undef C1M_HOOK
        push(0, njt, false);
#endif
        if (l == 0 && c == 0 && pl < NR)   // the rows' positions (the peers' inverse RoPE in the combine)
#pragma unroll
          for (int p = 0; p < W; ++p)
            if (C1M_PUSH_DATA(p, hxa)) ((long long*)(A.stg[p] + STG_P))[pl] = __ldg(A.pos + pl);
        __threadfence_system();
#ifdef C1M_PSTAMP
        if (pl == 0 && l == ltim && tim0) tim0[44] = gtime();   // (the system fence returned)
#endif
        __syncwarp();
#if !defined(C1M_TS) && !defined(C1M_NOTIM)
        if (pl == 0 && l == ltim && tim0) tim0[47] = gtime();   // (timeline builds: this CTA's push done, stamp 47)
#endif
        if (pl == 0) {
#pragma unroll
          for (int p = 0; p < W; ++p)
            if (C1M_PUSH_TO(p)) red_rel_sys(A.rdy[p] + NCTA + NRM, 1u);
#ifndef C1M_OWNSTG
          if (c == 0) red_rel_sys(A.rdy[rank] + NCTA + NRM, (unsigned)(NCTA + NR));   // (the owner's own count: epochs)
#endif
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
#ifndef C1M_DOTS_WIN
      // A1: hc_attn_fn slice [24][4][ncols] fp32
      {
#if defined(C1M_FN_FRAG2)
        const uint32_t s = slot(fn_bytes(c)), b = sbase(s), fb = su32(&S.full[s]);
        fn_produce2(b, a.tm + TM_AFN32, c, fb, pol);
#elif defined(C1M_FN_FRAG)
        const uint32_t s = slot(fn_bytes(c)), b = sbase(s), fb = su32(&S.full[s]);
        fn_produce(b, a.afn, c, fb, pol);
#else
        const uint32_t s = slot(96 * 48 * 4), b = sbase(s), fb = su32(&S.full[s]);
        bulk(b, a.afn + (size_t)c * 96 * 48, 96 * 48 * 4, fb, pol);
#endif
        ++it;
      }
#endif
      // hook 8: the producer's stream for hook 7 (after A1's slice, before the q_a | wkv chunks); empty by default
#define C1M_HOOK 8
#include C1M_HOOKS
#undef C1M_HOOK
#define C1M_DPART 0
#include C1M_DENSE_A3
#undef C1M_DPART
#ifdef C1M_DOTS_WIN
      // A1: hc_attn_fn slice (consumed after the q_a tile) [24][4][ncols] fp32
      {
#if defined(C1M_FN_FRAG2)
        const uint32_t s = slot(fn_bytes(c)), b = sbase(s), fb = su32(&S.full[s]);
        fn_produce2(b, a.tm + TM_AFN32, c, fb, pol);
#elif defined(C1M_FN_FRAG)
        const uint32_t s = slot(fn_bytes(c)), b = sbase(s), fb = su32(&S.full[s]);
        fn_produce(b, a.afn, c, fb, pol);
#else
        const uint32_t s = slot(96 * 48 * 4), b = sbase(s), fb = su32(&S.full[s]);
        bulk(b, a.afn + (size_t)c * 96 * 48, 96 * 48 * 4, fb, pol);
#endif
        ++it;
      }
#endif
#define C1M_HOOK 1
#include C1M_HOOKS
#undef C1M_HOOK
#define C1M_DPART 0
#include C1M_DENSE_A5
#undef C1M_DPART
    }
#define C1M_DPART 0
#include C1M_DENSE_F1
#undef C1M_DPART
#define C1M_DPART 0
#include C1M_DENSE_F2
#undef C1M_DPART
#ifndef C1M_DOTS_WIN
    // G: the fn slice [24][4][ncols] fp32
    {
#if defined(C1M_FN_FRAG2)
      const uint32_t s = slot(fn_bytes(c)), b = sbase(s), fb = su32(&S.full[s]);
      fn_produce2(b, a.tm + TM_FN32, c, fb, pol);
#elif defined(C1M_FN_FRAG)
      const uint32_t s = slot(fn_bytes(c)), b = sbase(s), fb = su32(&S.full[s]);
      fn_produce(b, a.fn, c, fb, pol);
#else
      const uint32_t s = slot(96 * 48 * 4), b = sbase(s), fb = su32(&S.full[s]);
      bulk(b, a.fn + (size_t)c * 96 * 48, 96 * 48 * 4, fb, pol);
#endif
      ++it;
    }
#endif
#ifdef C1M_RKS
    {   // H router, split-K: this CTA's gate columns for all 384 experts, 2 stages of 192 experts, one
        // {16 cols, 192 rows} box per 16-column tile (box t at t * 6144: [192][16] bf16)
      static_assert(NE == 384, "C1M_RKS: 2 x 192 experts");
      int ftb_, nt_;
      my_tiles(c, ftb_, nt_);
      for (int hs = 0; hs < 2; ++hs) {
        const uint32_t s = slot((unsigned)nt_ * 6144u), b = sbase(s), fb = su32(&S.full[s]);
        for (int t = 0; t < nt_; ++t) tma3d(b + t * 6144u, a.tm + TM_GATE16, 0, 192 * hs, ftb_ + t, fb, pol);
        ++it;
      }
    }
    for (int e = c; false; ) {
#else
    // H router: whole expert rows e = c, c + 132, c + 264 (bf16, 10 KB each), two rows per stage
    for (int e = c; c < NWK && e < NE; e += 2 * NWK) {   // rows e, e + 128 (and e + 256 alone)
#endif
      const int nrw = e + NWK < NE ? 2 : 1;
      const uint32_t s = slot(nrw * D * 2), b = sbase(s), fb = su32(&S.full[s]);
      for (int r = 0; r < nrw; ++r) bulk(b + r * D * 2, a.gate_w + (size_t)(e + r * NWK) * D, D * 2, fb, pol);
      ++it;
    }
#ifdef C1M_DOTS_WIN
    // G: the fn slice (consumed after the router rows) [24][4][ncols] fp32
    {
#if defined(C1M_FN_FRAG2)
      const uint32_t s = slot(fn_bytes(c)), b = sbase(s), fb = su32(&S.full[s]);
      fn_produce2(b, a.tm + TM_FN32, c, fb, pol);
#elif defined(C1M_FN_FRAG)
      const uint32_t s = slot(fn_bytes(c)), b = sbase(s), fb = su32(&S.full[s]);
      fn_produce(b, a.fn, c, fb, pol);
#else
      const uint32_t s = slot(96 * 48 * 4), b = sbase(s), fb = su32(&S.full[s]);
      bulk(b, a.fn + (size_t)c * 96 * 48, 96 * 48 * 4, fb, pol);
#endif
      ++it;
    }
#endif
    // the expert phase's weight stream (shared GEMM1 halves, routed GEMM1 items, GEMM2 routed / shared)
#ifdef C1M_SPIN_PW   // (the producer lane's blocking waits in the fragment: relaxed polls + one fence; its ld_acq checks stay)
#define spin_ge spin_ge_rlx
#endif
#include C1M_MOE_PRODUCE
#ifdef C1M_SPIN_PW
#undef spin_ge
#endif
    }   // layers
    return;
  }

  // ============================================================================================== consumers
#ifdef C1M_XN
  if (w >= NCW + 2) {   // the expert-only consumer warps
#ifdef C1M_XFRAG
    xcons<FULL>(al, nly, call);
#endif
    return;
  }
#endif
  CState cs{0, 0ull, 0ull, call, ltim, tim0};
  for (int l = 0; l < nly; ++l) {
#if defined(C1M_HAS_PHASES) && defined(C1M_ONE_BODY)
    cons_layer<FULL, true>(al, nly, l, cs);   // one body for every layer: the hooks gated at run time
    continue;
#elif defined(C1M_HAS_PHASES)
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
static unsigned long long g_attr[2] = {0, 0};
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
  if (wqb.numel() && wqb.dim() == 3) {   // the engine's W8A16 packed wq_b, this rank's [10][32][32896] view (patch_qb8)
    TORCH_CHECK(wqb.size(0) == 10 && wqb.size(1) == 32 && wqb.size(2) == 32896 && wqb.stride(2) == 1 && wqb.stride(1) == 32896 &&
                wqb.scalar_type() == torch::kUInt8 && ((uintptr_t)wqb.data_ptr() & 15) == 0, "wqb8: [10][32][32896] uint8 view");
    const uint64_t cs = (uint64_t)wqb.stride(0);   // chunk stride (128 tiles x 32,896 B in the engine's tensor)
    // fragments: {256 B, 16 x 256 B (64-row sub-tile m = 8 of them), 4 k32 blocks (4096 B), 64 half tiles (16,448 B), 10 chunks}
    enc(m + TM_WQB, U8, wqb.data_ptr(), {256, 16, 4, 64, 10}, {256, 4096, 16448, cs}, {256, 8, 4, 1, 2}, false);
    // scales: 16 B per half tile and chunk ([k32 block][32-row block]), after its 16 KB of fragments
    enc(m + TM_WQB8S, U8, (const uint8_t*)wqb.data_ptr() + 16384, {16, 64, 10}, {16448, cs}, {16, 1, 2}, false);
  } else if (wqb.numel()) {
    enc(m + TM_WQB, U8, wqb.data_ptr(), {128, 8192, 10}, {1280, 128}, {128, 64, 2}, true);
  }
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
#ifndef C1M_W13KB
#define C1M_W13KB 4
#endif
  enc(m + TM_W13, U8, w13.data_ptr(), {128, 576, 20, 768}, {2560, 128, 576ull * 2560}, {128, 16, C1M_W13KB, 2}, true);
  enc(m + TM_S13, U8, s13.data_ptr(), {160, 576, 768}, {160, 576ull * 160}, {32, 16, 2}, false);
  // C1M_FN_FRAG2: the engine's mhcf prep_f fragments [160][16][3][32][2] fp32 as {192, s2, half, h, ch} (s = 4h + 2 half + s2)
  for (auto pr : {std::make_pair(&fn, (int)TM_FN32), std::make_pair(&afn, (int)TM_AFN32)}) {
    const torch::Tensor& f = *pr.first;
    if (f.numel() && f.dim() == 5) {
      TORCH_CHECK(f.size(0) == 160 && f.size(1) == 16 && f.size(2) == 3 && f.size(3) == 32 && f.size(4) == 2 && f.is_contiguous() &&
                  f.scalar_type() == torch::kFloat32 && ((uintptr_t)f.data_ptr() & 15) == 0, "fn frag: [160][16][3][32][2] fp32");
      enc(m + pr.second, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, f.data_ptr(), {192, 2, 2, 4, 160}, {768, 1536, 3072, 12288},
          {192, 2, 1, 4, 1}, false);
    }
  }
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
int64_t nrm() { return (int64_t)NRM; }
int64_t lamport() {   // bit 0 RTL (logits), 1 QAL (qakv), 2 ZL (z): parity buffers + sentinels
  int64_t m = 0;
#ifdef C1M_RTL
  m |= 1;
#endif
#ifdef C1M_QAL
  m |= 2;
#endif
#ifdef C1M_ZL
  m |= 4;
#endif
#ifdef C1M_RKS
  m |= 8;
#endif
#ifdef C1M_CML
  m |= 16;   // fscr[2] apart / fscr[3] aml: CML_SENT-filled by the adapter
#endif
  return m;
}
bool rtl() {   // (C1M_RTL builds: the adapter fills the logits buffer with RTL_SENT)
#ifdef C1M_RTL
  return true;
#else
  return false;
#endif
}
int64_t cnt_n() { return (int64_t)C_N; }
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
// a packed layer's Args (CPU bytes) reading q_b's weights from the engine's W8A16 packed wq_b (its device pointer as an int,
// 0 = c1mega's fp8 copy)
// The kernel's producers keep their GLOBAL coordinates (group 2 rank + gi
// K block 32 rank + 8 kc), so each map's base is the slice minus the rank's offset and its outer dim ends at the slice's end:
// every box the kernel issues lies inside the slice (nothing below it is ever addressed). Same kernel code: bitwise.
// maps = the CPU maps tensor of maps3 (called with EMPTY woaT16 / wob16); patch_bf16 then takes the slices' pointers (flags)
void map_woq(torch::Tensor maps, torch::Tensor woaq, torch::Tensor wobq, int64_t rank) {
  TORCH_CHECK(maps.device().is_cpu() && maps.numel() >= TM_N * 128, "map_woq: the CPU maps tensor of maps3");
  TORCH_CHECK(rank >= 0 && rank < W, "map_woq: rank");
  CUtensorMap* m = (CUtensorMap*)maps.data_ptr();
  const auto U16 = CU_TENSOR_MAP_DATA_TYPE_UINT16;
  if (woaq.numel()) {
    TORCH_CHECK(woaq.scalar_type() == torch::kBFloat16 && woaq.dim() == 3 && woaq.size(0) == 2 && woaq.size(1) == 4096 &&
                woaq.size(2) == 1024 && woaq.is_contiguous(), "map_woq: wo_a slice [2][4096][1024] bf16 contiguous");
    const uint8_t* b = (const uint8_t*)woaq.data_ptr() - (size_t)(2 * rank) * 4096ull * 2048;
    enc(m + TM_WOA16, U16, b, {1024, 4096, (uint64_t)(2 * rank + 2)}, {2048, 4096ull * 2048}, {16, 256, 1}, false);
  }
  if (wobq.numel()) {
    TORCH_CHECK(wobq.scalar_type() == torch::kBFloat16 && wobq.dim() == 2 && wobq.size(0) == 5120 && wobq.size(1) == 2048 &&
                wobq.is_contiguous(), "map_woq: wo_b slice [5120][2048] bf16 contiguous");
    const uint8_t* b = (const uint8_t*)wobq.data_ptr() - (size_t)(32 * rank) * 128;
    enc(m + TM_WOB16, U16, b, {64, 5120, (uint64_t)(32 * rank + 32)}, {4096, 128}, {64, 16, 8}, true);
  }
}
// (srv8x) TM_WOA8 / TM_WOB8 (+ scales) on this rank's views of the engine's W8A16 packed wo_a / wo_b, into an existing maps tensor
// (maps3 called with EMPTY woa / woaT16 / wob / wob16): woa8 = pk[:, 8r:8r+8] of [32][32][32896] (8 tiles of 256 rows = groups
// 2r, 2r+1), wob8 = pk[16r:16r+16] of [64][20][32896] (K quarter r). Fragment maps {256 B, 16 x 256 B, 4 k32 blocks, half tiles,
// chunks}, box {256, 2 (one 16-row tile = 512 B), 4, 1, 4 chunks} = 8 KB per 512-K stage; scale maps {16 B, half tiles, chunks},
// box {16, 1, 4} (TM_WQB / TM_WQB8S's geometry, qb8)
void map_wo8(torch::Tensor maps, torch::Tensor woa8, torch::Tensor wob8) {
  TORCH_CHECK(maps.device().is_cpu() && maps.numel() >= TM_N * 128, "map_wo8: the CPU maps tensor of maps3");
  CUtensorMap* m = (CUtensorMap*)maps.data_ptr();
  const auto U8 = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  auto chk = [](const torch::Tensor& t, int64_t d0, int64_t d1, const char* nm) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kUInt8 && t.dim() == 3 && t.size(0) == d0 && t.size(1) == d1 &&
                t.size(2) == 32896 && t.stride(2) == 1 && t.stride(1) == 32896 && (t.stride(0) * 1) % 16 == 0 &&
                ((uintptr_t)t.data_ptr() & 15) == 0, nm, ": a [", d0, "][", d1, "][32896] uint8 view of the W8A16 packed tensor");
  };
  if (woa8.numel()) {
    chk(woa8, 32, 8, "map_wo8 woa8");
    const at::cuda::CUDAGuard guard(woa8.device());
    const uint64_t cs = (uint64_t)woa8.stride(0);
    enc(m + TM_WOA8, U8, woa8.data_ptr(), {256, 16, 4, 16, 32}, {256, 4096, 16448, cs}, {256, 2, 4, 1, 4}, false);
    enc(m + TM_WOA8S, U8, (const uint8_t*)woa8.data_ptr() + 16384, {16, 16, 32}, {16448, cs}, {16, 1, 4}, false);
  }
  if (wob8.numel()) {
    chk(wob8, 16, 20, "map_wo8 wob8");
    const at::cuda::CUDAGuard guard(wob8.device());
    const uint64_t cs = (uint64_t)wob8.stride(0);
    enc(m + TM_WOB8, U8, wob8.data_ptr(), {256, 16, 4, 40, 16}, {256, 4096, 16448, cs}, {256, 2, 4, 1, 4}, false);
    enc(m + TM_WOB8S, U8, (const uint8_t*)wob8.data_ptr() + 16384, {16, 40, 16}, {16448, cs}, {16, 1, 4}, false);
  }
}
torch::Tensor patch_wo8(torch::Tensor packed, int64_t woa8, int64_t wob8) {
  TORCH_CHECK(packed.device().is_cpu() && packed.numel() == (int64_t)sizeof(Args), "patch_wo8");
  torch::Tensor o = packed.clone();
  Args* a = (Args*)o.data_ptr();
  a->woa8 = (const void*)(uintptr_t)woa8, a->wob8 = (const void*)(uintptr_t)wob8;
  TORCH_CHECK(!(woa8 && a->woaT16) && !(wob8 && a->wob16), "patch_wo8: wo_a / wo_b from ONE source (packed or bf16)");
#ifdef C1M_FIX_B16
  TORCH_CHECK((woa8 != 0) == (((C1M_FIX_B16) & 32) != 0) && (wob8 != 0) == (((C1M_FIX_B16) & 64) != 0),
              "this build fixes the bf16 set to bits ", C1M_FIX_B16);
#endif
  return o;
}
void map_gate16(torch::Tensor maps, torch::Tensor gate) {
  TORCH_CHECK(maps.device().is_cpu() && maps.numel() >= TM_N * 128, "map_gate16: the CPU maps tensor of maps3");
  TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == torch::kBFloat16 && gate.dim() == 2 && gate.size(0) == 384 &&
              gate.size(1) == 5120 && gate.is_contiguous(), "map_gate16: gate [384][5120] bf16");
  const at::cuda::CUDAGuard guard(gate.device());
  enc((CUtensorMap*)maps.data_ptr() + TM_GATE16, CU_TENSOR_MAP_DATA_TYPE_UINT16, gate.data_ptr(), {16, 384, 320}, {10240, 32},
      {16, 192, 1}, false);
}
torch::Tensor patch_idd(torch::Tensor packed, int64_t idd) {
  TORCH_CHECK(packed.device().is_cpu() && packed.numel() == (int64_t)sizeof(Args), "patch_idd");
  torch::Tensor o = packed.clone();
  ((Args*)o.data_ptr())->idd = (int*)(uintptr_t)idd;
  return o;
}
torch::Tensor patch_rks(torch::Tensor packed, int64_t rkp) {
  TORCH_CHECK(packed.device().is_cpu() && packed.numel() == (int64_t)sizeof(Args), "patch_rks");
  torch::Tensor o = packed.clone();
  ((Args*)o.data_ptr())->rkp = (float*)(uintptr_t)rkp;
  return o;
}
torch::Tensor patch_qb8(torch::Tensor packed, int64_t wqb8) {
  TORCH_CHECK(packed.device().is_cpu() && packed.numel() == (int64_t)sizeof(Args), "patch_qb8");
  torch::Tensor o = packed.clone();
  Args* a = (Args*)o.data_ptr();
  a->wqb8 = (const void*)(uintptr_t)wqb8;
#ifdef C1M_FIX_B16
  TORCH_CHECK((wqb8 != 0) == (((C1M_FIX_B16) & 16) != 0), "this build fixes the bf16 set to bits ", C1M_FIX_B16);
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
#ifdef C1M_RTSC
  TORCH_CHECK(scr[4].numel() >= 3 * NRM * NE, "rtp: [3][NRM][NE] fp32 (logits | scores | keys) in this build (without C1M_RTSC: [NRM][NE])");
#endif
#ifdef C1M_RTL
  TORCH_CHECK(scr[4].numel() >= 2 * NRM * NE, "rtp: [2][NRM][NE] fp32 (C1M_RTL: parity buffers, RTL_SENT-filled by the adapter)");
#endif
#ifdef C1S_RTS2
  TORCH_CHECK(scr[4].numel() >= 4 * NRM * NE, "rtp: [2][NRM][NE] {logit, score} 8-B entries (C1S_RTS2), RTL_SENT-filled");
#endif
#ifdef C1M_ZL
  TORCH_CHECK(scr[0].numel() >= 2 * NRM * 2048, "z: [2][NRM][2048] bf16 (C1M_ZL: parity buffers, 0xFFFF-filled by the adapter)");
#endif
  a.shp = P<float>(scr[5]), a.hsh = P<bf16>(scr[6]), a.hmax = P<float>(scr[7]), a.himg = P<__half>(scr[8]);
  a.cnt = P<unsigned>(scr[9]), a.tick = P<unsigned>(scr[10]);
  TORCH_CHECK(scr[11].is_cuda() && scr[11].numel() >= TM_N * 128 && ((uintptr_t)scr[11].data_ptr() & 63) == 0, "tensor maps");
  a.tm = (const CUtensorMap*)scr[11].data_ptr();
  for (int s = 0; s < W; ++s) a.ar1[s] = P<float>(ar1[s]), a.ar2[s] = P<bf16>(ar2[s]);
  a.tim = tim.numel() ? (unsigned long long*)tim.data_ptr() : nullptr;
}
template <bool FULL>
static void launch(const Args* al, int nly, int dev) {
  constexpr size_t SM = smem_of<FULL>();
  if (!((g_attr[FULL] >> dev) & 1)) {
    TORCH_CHECK(cudaFuncSetAttribute(k_layer<FULL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SM) == cudaSuccess, "smem attr");
    g_attr[FULL] |= 1ull << dev;
  }
#ifdef C1S_CLS
  // cluster launch: every cluster must be co-resident (the kernel spins on grid counters)
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NCTA);
  cfg.blockDim = dim3(THR);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute cat[1];
  cat[0].id = cudaLaunchAttributeClusterDimension;
  cat[0].val.clusterDim.x = C1S_CLS, cat[0].val.clusterDim.y = 1, cat[0].val.clusterDim.z = 1;
  cfg.attrs = cat;
  cfg.numAttrs = 1;
  static int g_ncl[2][64] = {};
  if (!g_ncl[FULL][dev]) {
    int n = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveClusters(&n, k_layer<FULL>, &cfg) == cudaSuccess, "cluster occupancy query");
    g_ncl[FULL][dev] = n > 0 ? n : -1;
  }
  TORCH_CHECK(g_ncl[FULL][dev] >= NCTA / C1S_CLS, "C1S_CLS ", C1S_CLS, ": only ", g_ncl[FULL][dev], " clusters fit, ",
              NCTA / C1S_CLS, " needed");
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, k_layer<FULL>, al, nly) == cudaSuccess, "k_layer cluster launch");
#else
  k_layer<FULL><<<NCTA, THR, SM, at::cuda::getCurrentCUDAStream()>>>(al, nly);
#endif
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "k_layer launch");
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
#ifdef C1M_ONLY_FULL   // (deployment builds: k_layer<false>, the FFN-only test kernel, is never launched: not compiled)
  TORCH_CHECK(full, "this build has the whole-layer kernel only (C1M_ONLY_FULL)");
  launch<true>((const Args*)packed.data_ptr(), (int)nly, packed.get_device());
#else
  if (full) launch<true>((const Args*)packed.data_ptr(), (int)nly, packed.get_device());
  else launch<false>((const Args*)packed.data_ptr(), (int)nly, packed.get_device());
#endif
}
int64_t args_bytes() { return (int64_t)sizeof(Args); }
// the layer phases (index / compressor / Engram) are compiled in (the hooks file defines C1M_HAS_PHASES): C1M_IX=1 runs
// the mHC fn read in place from the engine's mhcf prep_f fragments (C1M_FN_FRAG): the adapter passes lw["fa_f"] / lw["ff_f"]
bool fn_frag() {
#if defined(C1M_FN_FRAG) || defined(C1M_FN_FRAG2)
  return true;
#else
  return false;
#endif
}
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
#ifdef C1M_QAL
  TORCH_CHECK(fo[5].numel() >= 2 * NRM * 1792, "qakv: [2][NRM][1792] bf16 (C1M_QAL: parity buffers, 0xFFFF-filled by the adapter)");
#endif
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
  m.def("a8c", []() -> int64_t {
#ifdef C1M_A8C                       // wo_a (c1dense_f1_a8.inc), bit 2 q_a|wkv (c1dense_a3_a8.inc); 0 = none)
#ifndef C1M_A8C_MASK
#define C1M_A8C_MASK 1
#endif
    return (int64_t)C1M_A8C_MASK;
#else
    return 0;
#endif
  });
  m.def("has_phases", &has_phases);
  m.def("fn_frag", &fn_frag);
  m.def("smem_bytes", &smem_bytes);
  m.def("maps", &maps);
  m.def("maps2", &maps2);
  m.def("maps3", &maps3);
  m.def("map_woq", &map_woq);
  m.def("patch_bf16", &patch_bf16);
  m.def("patch_sh16", &patch_sh16);
  m.def("patch_qb8", &patch_qb8);
  m.def("map_wo8", &map_wo8);
  m.def("patch_wo8", &patch_wo8);
  m.def("patch_ext", &patch_ext);
  m.def("patch_stg", &patch_stg);
  m.def("stg_bytes", &stg_bytes);
  m.def("nrm", &nrm);
  m.def("rtl", &rtl);
  m.def("lamport", &lamport);
  m.def("map_gate16", &map_gate16);
  m.def("patch_rks", &patch_rks);
  m.def("patch_idd", &patch_idd);
  m.def("cnt_n", &cnt_n);
}
