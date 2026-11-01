// Fused decode prologue of one DeepSeek-V4.1-Flash decoder layer on sm90a (stock vLLM 0.32.0, TP4,
// M <= 8 tokens, non-Engram layers with a residual stream).
//
// Adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
// src/brrr/kernels/decoder.cu prologue_kernel, lines ~1556-2131; see LICENSES/BRRR-Apache-2.0.txt):
// same persistent single-launch structure (all CTAs co-resident, phases chained through global
// counters with acquire/release, weights prefetched into registers before waiting) and the same lane
// transposition tricks (lane_sums, mix_reduce). Changed for sm90 / stock vLLM:
//   * GEMMs use the skinny FP16-HMMA fragment path of skinny.cu (no block-scaled MMA on Hopper) with
//     bf16 activations (stock W8A16 numerics) instead of MX-quantized activations;
//   * the attention mHC coefficients (Sinkhorn) are finished here (BRRR deferred them to its layer
//     kernel) because the next stock op consumes post/comb/pre;
//   * mHC numerics follow stock mhc_fused_post_pre_delayed_tilelang (model_executor/kernels/mhc/
//     tilelang.py:269-497; tilelang_kernels.py:668-786 fused post+GEMM, :299-503 pre-norm epilogue):
//     the 24 mixes and the RMS use the fp32 post-mapped stream, the collapse reads the bf16 stream;
//   * the SWA KV write reproduces stock fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert byte for
//     byte (vLLM csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu processDeepseekV4Slot, V4 record:
//     584 B/token = 448 e4m3 + 64 bf16 RoPE dims, then 7 ue8m0 scales of 64 dims + a zero pad byte,
//     page-segregated data | scales; models/deepseek_v41/common/ops/cache_utils.py:37-48, :83-180);
//   * q is produced as contiguous [M, 16, 512] bf16 with RoPE (no 16 -> 64 head padding).
//
// Phase graph (counters in scratch; every CTA is resident, so spin-waits cannot deadlock):
//   P1  (G2 warps, before G2 loads wq_b; fn lines L2-prefetched before the PDL wait): per (32-column slice,
//       token) P1a: residual_out = post*x + comb^T residual (bf16), collapse with the INCOMING pre_mix (bf16)
//       and its square sum -> xcb / xsq                                          -> C_P1A (per G2 CTA)
//       then (the CTA's wq_b slice is L2-prefetched) P1b: partial 24 mixes + sum(new_r^2) over the fp32
//       stream -> mixp                                                            -> C_P1B (per G2 CTA)
//   G1  (112 CTAs = 14 groups of 8 tiles x 8 k-parts of 640): wq_a|wkv weights prefetched at launch; after
//       C_P1A (early mode) or C_P1B each CTA builds its 640-column slice of a = RMSNorm(collapse) (attn_norm)
//       itself, runs its k-part, the last of the 8 k-part warps of a tile reduces (fixed order), rounds to
//       bf16 (stock GEMM output) and publishes qkv + per-tile square sums.            -> C_P3Q / C_P3K
//   G2  (128 CTAs = 4 tiles x 2 k-halves of wq_b): after C_P3Q each CTA builds qr = q_norm(qkv[:, :1280])
//       itself, runs the 8192 x 1280 GEMM slice, reduces the halves in smem, applies q RoPE.
//   H   (2 CTAs): Sinkhorn of the attention mHC mixes (after C_P1B) and kv_norm + RoPE + FP8 V4-record
//       insert (after C_P3K).
// Stock mhc_shifted_post_pre (models/deepseek_v41/nvidia/ops/mhc.py:191, model.py:565-599) collapses this
// attention input with the PREVIOUS sublayer's pre (pre_mix argument) and only hands this sublayer's
// Sinkhorn outputs (post, comb, attn_pre) to the FFN boundary, so nothing in G1/G2 depends on P1b or the
// Sinkhorn: with `early` G1 starts as soon as the collapse is published and the Sinkhorn chain (fn GEMV ->
// mix reduction -> 20 iterations) runs concurrently with both GEMMs.
//
// Switches (Params, from ds41h/prologue.py; defaults keep the Phase-1 behaviour):
//   MMA template (DS41H_PROLOGUE_MMA=hmma|wgmma): G1/G2 MMA flavour, see skinny.cu.
//   fast & FAST_EARLY   (DS41H_PROLOGUE_FAST=1): G1 waits for C_P1A instead of C_P1B.
//   fast & FAST_SINKHORN (DS41H_PROLOGUE_FAST=1): Sinkhorn of all tokens in one warp (lane = token, the
//       4x4 matrix in registers, no shuffles) with one rcp.approx per row/column instead of IEEE divisions
//       (~40 dependent cycles per half-iteration instead of ~4 shuffles + 1 division ~ 130). Exact mode keeps
//       the Phase-1 lane-parallel code (bit-identical results).
#define DS41H_SKINNY_DEVICE_ONLY
#include "skinny.cu"

#include <c10/cuda/CUDAGuard.h>

#include "host.cuh"

namespace ds41h_prologue {

using namespace ds41h_skinny;

constexpr int HC = 4, MIX = (2 + HC) * HC, HID = 5120, HCD = HC * HID, SLICES = HID / 32, MIXW = 32;
constexpr int QR = 1280, KVW = 512, QKV = QR + KVW, HDIM = 512, ROPE = 64, NOPE = HDIM - ROPE, HEADS = 16, WQB = HEADS * HDIM;
constexpr int QKV_TILES = QKV / 16, Q_TILES = QR / 16, KV_TILES = KVW / 16, WQB_TILES = WQB / 16;
constexpr int KB1 = HID / 32, KB2 = QR / 32;
constexpr int KSPLIT1 = KB1 / KW_MAX, KPART1 = KW_MAX * 32;  // 8 k-parts of 640 columns
constexpr int G1_GROUPS = QKV_TILES / WARPS;                 // 14 groups of 8 tiles
constexpr int G1_CTAS = G1_GROUPS * KSPLIT1;                 // 112
constexpr int G2_TILES = WARPS / 2;                          // 4 tiles x 2 k-halves per CTA
constexpr int G2_CTAS = WQB_TILES / G2_TILES;                // 128
constexpr int H_CTAS = 2;                                    // 16 warps: Sinkhorn x8, KV x8
constexpr int CTAS = G1_CTAS + G2_CTAS + H_CTAS;             // 242
constexpr int P1_WORKERS = G2_CTAS * WARPS;                  // 1024 warps (G2 CTAs only)
constexpr int H_KV0 = 8;                                     // first H warp doing KV jobs
constexpr int V4_DATA = 576, V4_SCALES = 8;                  // V4 record: page data then page scales
static_assert(MIX == 24 && QKV_TILES % WARPS == 0 && KB1 % KW_MAX == 0 && KB2 == 2 * KW_MAX && WQB_TILES % G2_TILES == 0);
static_assert(QR % 16 == 0, "a 16-row tile is entirely q or entirely kv");
// P1 jobs: one (slice, token) per warp for m <= 6, (slice, token pair) for m = 7, 8; one job per worker.
static_assert(SLICES * 6 <= P1_WORKERS && SLICES * 4 <= P1_WORKERS, "P1 jobs must fit one per worker warp");
static_assert(KV_TILES == 32, "kv job reads one per-tile square sum per lane");
static_assert(KW_MAX % WG_G == 0 && G2_TILES == 4, "wgmma: a warpgroup = 4 tiles of one k-part");
enum : int {
  FAST_EARLY = 1, FAST_SINKHORN = 2, FAST_PREFETCH = 4, FAST_FN_SMEM = 8, FAST_QR_LOCAL = 16, FAST_LIGHT_SYNC = 32, FAST_CLUSTER = 64,
  FAST_QR_DEFER = 128, FAST_OWN_COLLAPSE = 256, FAST_V3 = 512, FAST_V3_LATE_W2 = 1024, FAST_V3_HOP = 2048,
  FAST_V3_ADEFER = 4096, FAST_V3_W1_LATE = 8192, FAST_V3_PACE = 16384, FAST_V3_PACE1 = 32768, FAST_V3_SK_G2 = 65536,
  FAST_V3_FAC = 131072, FAST_V3_ILV = 262144, FAST_V3_PUSH = 524288, FAST_V3_SK_G1 = 1048576, FAST_V3_KV_SPREAD = 2097152,
  FAST_V3_L2PF = 4194304, FAST_V3_HOST_SPREAD = 8388608, FAST_V3_FENCE = 16777216, FAST_V3_ACQPOLL = 33554432,
  FAST_V3_SKREG = 67108864, FAST_V3_SKROW = 134217728, FAST_V3_EXT_P1A = 268435456, FAST_V3_W2_WIN4 = 536870912,
  FAST_V3_W2_OPEN = 1073741824
};
constexpr int FAST_V3_BITS = FAST_V3 | FAST_V3_LATE_W2 | FAST_V3_HOP | FAST_V3_ADEFER | FAST_V3_W1_LATE | FAST_V3_PACE | FAST_V3_PACE1 |
                             FAST_V3_SK_G2 | FAST_V3_FAC | FAST_V3_ILV | FAST_V3_PUSH | FAST_V3_SK_G1 | FAST_V3_KV_SPREAD |
                             FAST_V3_L2PF | FAST_V3_HOST_SPREAD | FAST_V3_FENCE | FAST_V3_ACQPOLL | FAST_V3_SKREG |
                             FAST_V3_SKROW | FAST_V3_EXT_P1A | FAST_V3_W2_WIN4 | FAST_V3_W2_OPEN;
constexpr int CLUSTER = KSPLIT1;                             // FAST_CLUSTER: the 8 k-parts of a G1 tile group
constexpr int CTAS_CL = G1_CTAS + G2_CTAS;                    // 240 = 30 clusters (the H jobs run on the last 2 G1 CTAs)
static_assert(G1_CTAS % CLUSTER == 0 && G2_CTAS % CLUSTER == 0 && KPART1 % 32 == 0);

// ---------------------------------------------------------------- scratch (one per device, zero-initialized)
// C_P1A / C_P1B count G2 CTAs (target G2_CTAS); C_P3Q / C_P3K count finished q / kv tiles; C_QR counts the
// QR_CTAS CTAs that built the shared e4m3 qr tile (wgmma).
enum : int { C_P1A = 0, C_P1B = 1, C_P3Q = 2, C_P3K = 3, C_QR = 4, C_FIN = 5, C_QHINT = 6, C_P1AHINT = 7, NCOUNTERS = 8 };
// DS41H_PROLOGUE_LL: C_QHINT / C_P1AHINT = relaxed hint counts (q tiles / G2 CTAs whose LL words are stored, no release);
// C_EPOCH (never reset)
// = launches of the v3 kernel, the LL flag of a launch is C_EPOCH + 1 (read after the PDL wait, bumped by the last CTA).
enum : int { C_EPOCH = 10 };
// DS41H_PROLOGUE_INDEXER_FOLD: idx_w published (v3_idx_w rank 0 -> the fold's weights_out writers). C_QR is wgmma-only and the
// indexer projections are HMMA-only, so the reset counter word is free.
enum : int { C_IDXW = C_QR };
// DS41H_PROLOGUE_TICKET (v3, opt-in): no end-of-kernel arrival / counter reset. Thread 0 of every CTA takes a 64-bit ticket
// (pcf[C_TICKET], never reset) at kernel entry; launch e = ticket / CTAS_V3 uses counter block e & 1 (pcf or pcf + PCB1),
// selected right after the PDL wait (tick_publish), and its first ticket holder zeroes the other block (launch e - 1's,
// which has completed; launch e + 1 finds it zero after this launch completes).
enum : int { C_TICKET = 12, PCB1 = 32 };
// Attention handoff (Params::aflag, DS41H_ATTN_FLAG): words of the same 16-word counter block that the prologue never resets.
// C_AQ counts the G2 CTAs whose q rows are stored + the per-token KV insert warps that are done (target G2_CTAS + m); the
// consuming sparse decode attention polls it instead of griddepcontrol.wait, and its last CTA past the wait (C_AACK)
// resets both to 0 (or ds41h/attn.py's reset kernel does, when the attention of that layer falls back to stock).
enum : int { C_AQ = 8, C_AACK = 9 };
// wgmma: G1 CTAs 0..QR_CTAS-1 quantize qr ONCE (8 quads = 8 (k32 block, token) jobs per warp) into a global
// B tile that every G2 CTA copies, instead of each of the 128 G2 CTAs normalizing + quantizing all 1280 columns.
constexpr int QR_CTAS = 5;
static_assert(QR_CTAS * WARPS * 8 >= KB2 * MAXM && QR_CTAS <= G1_CTAS && MAXM <= WARPS);
constexpr long OFF_PC = 0;                                               // 16 x u32
constexpr long OFF_TILECNT = 256;                                        // QKV_TILES x u32
constexpr long OFF_MIXP = 1024;                                          // [MAXM][SLICES][MIXW] f32
constexpr long OFF_XSQ = OFF_MIXP + long(MAXM) * SLICES * MIXW * 4;      // [MAXM][SLICES] f32
constexpr long OFF_XCB = OFF_XSQ + long(MAXM) * SLICES * 4;              // [MAXM][HID] bf16 (collapse)
constexpr long OFF_QKP = OFF_XCB + long(MAXM) * HID * 2;                 // [QKV_TILES][KSPLIT1][16][MAXM] f32
constexpr long OFF_QKV = OFF_QKP + long(QKV_TILES) * KSPLIT1 * 16 * MAXM * 4;  // [MAXM][QKV] bf16
constexpr long OFF_QSQ = OFF_QKV + long(MAXM) * QKV * 2;                 // [QKV_TILES][MAXM] f32
constexpr long OFF_QRB = (OFF_QSQ + long(QKV_TILES) * MAXM * 4 + 255) / 256 * 256;  // [KB2][WG_KB_PITCH] e4m3 qr B tile
constexpr long OFF_QRS = (OFF_QRB + long(KB2) * WG_KB_PITCH + 255) / 256 * 256;     // [KB2][MAXM] f32 qr block scales
// FAST_V3_ADEFER (wgmma): the P1a jobs quantize u = bf16(collapse * attn_norm_w) per (token, k32 block) straight into an
// e4m3 B tile of the whole 5120-wide activation (attn_norm's rsqrt deferred to the wq_a|wkv finisher).
constexpr long OFF_QAB = (OFF_QRS + long(KB2) * MAXM * 4 + 255) / 256 * 256;   // [KB1][WG_KB_PITCH] e4m3
constexpr long OFF_QAS = (OFF_QAB + long(KB1) * WG_KB_PITCH + 255) / 256 * 256;  // [KB1][MAXM] f32 block scales
// Params::idx (indexer projections, see v3_idx_w): weights_proj partial sums [k-part][32 rows][MAXM] f32
constexpr long OFF_IWP = (OFF_QAS + long(KB1) * MAXM * 4 + 255) / 256 * 256;
// DS41H_PROLOGUE_LL words: 8 bytes = {2 bf16 data, u32 launch flag}, stored / loaded single-copy atomically
constexpr long OFF_QLL = (OFF_IWP + long(KSPLIT1) * 32 * MAXM * 4 + 255) / 256 * 256;  // [MAXM][Q_TILES][8] q cols (c, c + 8)
constexpr long OFF_QSQLL = OFF_QLL + long(MAXM) * Q_TILES * 8 * 8;                      // [Q_TILES][MAXM] {f32 qsq, flag}
constexpr long OFF_XLL = (OFF_QSQLL + long(Q_TILES) * MAXM * 8 + 255) / 256 * 256;  // [MAXM][SLICES][16] collapse cols 2w, 2w + 1
constexpr long OFF_XSQLL = OFF_XLL + long(MAXM) * SLICES * 16 * 8;                    // [MAXM][SLICES] {f32 xsq, flag}
constexpr long SCRATCH_BYTES = (OFF_XSQLL + long(MAXM) * SLICES * 8 + 255) / 256 * 256;
static_assert(OFF_TILECNT + QKV_TILES * 4 <= OFF_MIXP && OFF_XSQ % 256 == 0 && OFF_XCB % 256 == 0 && OFF_QKP % 256 == 0 &&
              OFF_QKV % 256 == 0 && OFF_QSQ % 256 == 0 && OFF_QRB % 256 == 0 && OFF_QRS % 256 == 0);

// ---------------------------------------------------------------- shared memory (one layout for all roles)
constexpr int SM_ACT = 0;                                      // G2: MAXM x act_pitch(1280); G1: MAXM x act_pitch(640)
constexpr int SM_XE = SM_ACT + MAXM * act_pitch(QR);           // exponent bytes [KB2][MAXM]
constexpr int SM_AUX = SM_XE + round_up(KB2 * MAXM, 128);      // G1: xsq [MAXM][SLICES] f32; G2: qsq [Q_TILES][MAXM] f32
constexpr int SM_W = SM_AUX + MAXM * SLICES * 4;               // norm weight slice (bf16, <= 1280)
constexpr int SM_RS = SM_W + QR * 2;                           // per-token rsqrt [MAXM] f32
constexpr int SM_RED = SM_RS + 64;                             // G2: [WARPS][16][MAXM] f32
constexpr int SMEM_BYTES = SM_RED + WARPS * 16 * MAXM * 4;     // HMMA instantiation
// wgmma instantiation only (after the HMMA layout; SM_ACT holds the raw bf16 rows, SM_XE is unused)
constexpr int SM_BT = round_up(SMEM_BYTES, 128);               // e4m3 B tile, KB2 blocks of WG_KB_PITCH
constexpr int SM_SX = SM_BT + KB2 * WG_KB_PITCH;               // [KB2][MAXM] f32 activation block scales
constexpr int SM_FAC = SM_SX + KB2 * MAXM * 4;                 // [WARPS][KW_MAX][MAXM] f32 promotion factors
constexpr int SM_WF = SM_FAC + WARPS * KW_MAX * MAXM * 4;      // G1 attn_norm slice as f32
constexpr int SMEM_BYTES_WG = SM_WF + KPART1 * 4;
static_assert(MAXM * act_pitch(KPART1) <= SM_XE - SM_ACT && Q_TILES * MAXM <= MAXM * SLICES && SM_RED % 16 == 0 && SM_W % 16 == 0);
static_assert(SM_SX % 16 == 0 && SM_FAC % 16 == 0 && SM_WF % 16 == 0 && MAXM * 16 * 4 <= SM_XE, "H CTA comb-logit exchange lives in SM_ACT");
template <int MMA>
constexpr int smem_of() {
  return MMA == MMA_WGMMA ? SMEM_BYTES_WG : SMEM_BYTES;
}

struct Params {
  // inputs (previous kernels)
  const __nv_bfloat16* x;      // [m, HID] previous FFN output (all-reduced)
  const __nv_bfloat16* resid;  // [m, HC, HID]
  const float* post;           // [m, HC(, 1)] previous FFN post mix
  const float* comb;           // [m, HC, HC] previous FFN comb (res_mix), comb[k][j] mixes stream k into j
  const float* pre;            // [m, HC] previous FFN pre mix (collapse weights of this attention input)
  const long* positions;       // [>= m] int64
  const long* slots;           // [n_insert] int64 SWA slot mapping (may be null)
  // weights
  const float* fn;  // hc_attn_fn [MIX, HCD]
  const float* hc_scale;
  const float* hc_base;
  const __nv_bfloat16* attn_norm;
  const uint8_t* wqa;  // skinny-packed fused wq_a|wkv [1792, 5120]
  const uint8_t* wqa_s;
  const __nv_bfloat16* q_norm;
  const __nv_bfloat16* kv_norm;
  const uint8_t* wqb;  // skinny-packed wq_b TP slice [8192, 1280]
  const uint8_t* wqb_s;
  const float* cos_sin;  // [max_pos, 64] fp32: cos(0..31) | sin(0..31)
  // SWA cache (V4 record pages)
  uint8_t* swa;
  long swa_stride;  // bytes per page (k_cache.view(blocks, -1).stride(0))
  int swa_block;    // tokens per page
  int n_insert;
  // outputs
  __nv_bfloat16* resid_out;  // [m, HC, HID]
  float* post_out;           // [m, HC]
  float* comb_out;           // [m, HC, HC]
  float* pre_out;            // [m, HC]
  __nv_bfloat16* a;          // [m, HID] attn_norm(collapse)
  __nv_bfloat16* qr;         // [m, QR]
  __nv_bfloat16* kv;         // [m, KVW] kv_norm output (pre-RoPE, like stock)
  __nv_bfloat16* q;          // [m, HEADS, HDIM] with RoPE
  __nv_bfloat16* q_pre;      // optional [m, WQB] pre-RoPE q (tests)
  // scratch
  unsigned* pc;
  unsigned* pcf;             // fixed counter base (C_AQ / C_AACK / C_EPOCH / C_TICKET); pc = pcf or pcf + PCB1 (ticket mode)
  int tick;                  // DS41H_PROLOGUE_TICKET=1 (v3 only)
  unsigned* tilecnt;
  float* mixp;
  float* xsq;
  __nv_bfloat16* xcb;
  float* qkp;
  __nv_bfloat16* qkv;
  float* qsq;
  uint8_t* qrb;  // wgmma: shared e4m3 qr B tile (scratch)
  float* qrs;    // wgmma: its block scales
  uint8_t* qab;  // FAST_V3_ADEFER: e4m3 B tile of u = bf16(collapse * attn_norm_w) (scratch)
  float* qas;    // its block scales
  // Params::idx (DS41H_PROLOGUE_INDEXER, index-source layers, HMMA + v3 only): indexer projections computed in the prologue
  const uint8_t* iwq;        // skinny-packed indexer.wq_b [4096, 1280] (same format as wq_b)
  const uint8_t* iwq_s;
  const __nv_bfloat16* iwp;  // indexer.weights_proj.weight [32, 5120] bf16
  __nv_bfloat16* idx_q;      // [m, 4096] = indexer.wq_b(qr) (bf16, no RoPE: the stock fused q rope/quant follows)
  __nv_bfloat16* idx_w;      // [m, 32]   = indexer.weights_proj(a) (bf16)
  float* iwpart;             // scratch OFF_IWP
  int idx;
  // DS41H_PROLOGUE_INDEXER_FOLD (with idx_w): the stock fused_indexer_q_rope_quant (CuTe FP8 kernel) and the sparse indexer's top-k
  // buffer clear folded into idx_q's epilogue (see v3_idx_q):
  uint8_t* idx_qf;           // [m, 32, 128] e4m3 indexer q (RoPE on dims 64..127, per-(token, head) power-of-two scale)
  float* idx_wf;             // [m, 32] f32 weights_out = idx_w * wscale * q_scale
  const float* idx_cs;       // indexer rotary cos_sin_cache [max_pos, 64] f32
  float idx_wscale;          // float(softmax_scale * head_scale)
  int* topk_buf;             // optional: rows [0, m) x topk_w of the shared top-k buffer set to -1
  long topk_ld;
  int topk_w;
  __nv_bfloat16* aux;        // optional [m, HID]: capture_aux = residual_out.mean(dim=1) (DS41H_PROLOGUE_AUX)
  int m, tpj, iters, fast;  // fast: FAST_EARLY | FAST_SINKHORN bits
  unsigned long long* qll;   // scratch OFF_QLL (DS41H_PROLOGUE_LL)
  unsigned long long* qsqll; // scratch OFF_QSQLL
  unsigned long long* xll;   // scratch OFF_XLL
  unsigned long long* xsqll; // scratch OFF_XSQLL
  int ll;                    // DS41H_PROLOGUE_LL (opt-in bits): 1 = the G1 -> G2 q hop as LL words + relaxed hint (see v3_g2),
                             // 2 = the P1a -> G1 hop (xcb / xsq) likewise (see p1_collapse / v3_g1)
  int cvt;                  // DS41H_PROLOGUE_CVT=1 (opt-in): v3 G1 / G2 activation staging with convert_act_batched
  int mma2;                 // DS41H_PROLOGUE_MMA2=1 (opt-in): v3 HMMA MMA phases with 2 tiles per warp sharing one activation
                            // fragment (warps 0-3 compute, 4-7 idle): half the activation LDS, same per-tile FMA order
  int probe;                // DS41H_PROLOGUE_PROBE (bench-only timing probes, NOT synchronization-correct; default 0):
                            // 1: G1 finisher q/kv tile signals relaxed (no release), 2: P1a signals relaxed;
                            // 4 (correctness-safe test probe): DS41H_PROLOGUE_LL hint issued before the LL words;
                            // 8: no end-of-kernel C_FIN arrival / counter reset (bench zeroes the counters between launches)
  int aflag;                // 1: signal C_AQ for the flag-mode attention (q rows of each G2 CTA, each token's KV insert)
  float rms_eps, hc_eps, norm_eps, qkv_eps, post_mult;
  unsigned long long* prof;  // DS41H_PROLOGUE_PROF builds only: [CTAS][PROF_SLOTS] %globaltimer stamps
};

// FAST_V3_ILV: v3 cluster q of the grid (8 CTAs each) is G1 group q / 2 (even q < 28) or G2 cluster (q - 1) / 2 (odd q < 28),
// G2 cluster q - 14 for q >= 28 (instead of 14 G1 clusters followed by 16 G2 clusters), so the block scheduler pairs G1 and G2
// CTAs on SMs (2 per SM) instead of stacking two G2 CTAs (both MMA at the same time) on some SMs. Returns the logical index
// (G1: 0..111, G2: 112..239); the cluster rank (k-part) is unchanged.
__device__ __forceinline__ int v3_logical_block(int b, int fast) {
  if (!(fast & FAST_V3_ILV)) return b;
  constexpr int G1_CL = G1_CTAS / CLUSTER;  // 14 G1 clusters
  static_assert(CLUSTER == 8 && G2_CTAS / CLUSTER >= G1_CL);
  const int q = b >> 3, r = b & 7;
  if (q >= 2 * G1_CL) return G1_CTAS + (q - G1_CL) * CLUSTER + r;
  return (q & 1) ? G1_CTAS + (q >> 1) * CLUSTER + r : (q >> 1) * CLUSTER + r;
}

// ---------------------------------------------------------------- optional phase timestamps (profiling build)
// Built with -DDS41H_PROLOGUE_PROF (ds41h/prologue.py prof_ext()): CTA b writes %globaltimer at its phase
// boundaries into prof[b * PROF_SLOTS + slot], plus %clock64 at entry / exit (slots 13 / 14) to recover the
// SM clock. Slot map: 0 entry, 1 after the PDL wait, 15 exit; role-specific slots documented at each PROF. Slots 16 + s hold
// the %clock64 of stamp s (exact intra-CTA phase durations in SM cycles; %globaltimer ticks are ~0.25 us).
constexpr int PROF_SLOTS = 32;
#ifdef DS41H_PROLOGUE_PROF
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}
#define PROF_AT(slot)                                                                                        \
  do {                                                                                                       \
    if (p.prof) {                                                                                            \
      p.prof[v3_logical_block(blockIdx.x, p.fast) * PROF_SLOTS + (slot)] = gtimer();                         \
      p.prof[v3_logical_block(blockIdx.x, p.fast) * PROF_SLOTS + 16 + (slot)] = (unsigned long long)clock64(); \
    }                                                                                                        \
  } while (0)
#define PROF_T0(slot)                  \
  do {                                 \
    if (threadIdx.x == 0) PROF_AT(slot); \
  } while (0)
#define PROF_MAX(slot)                                                                                              \
  do {                                                                                                              \
    if (p.prof) {                                                                                                   \
      atomicMax(p.prof + v3_logical_block(blockIdx.x, p.fast) * PROF_SLOTS + (slot), gtimer());                     \
      atomicMax(p.prof + v3_logical_block(blockIdx.x, p.fast) * PROF_SLOTS + 16 + (slot), (unsigned long long)clock64()); \
    }                                                                                                               \
  } while (0)
#define PROF_CLK(slot)                                                                                       \
  do {                                                                                                       \
    if (threadIdx.x == 0 && p.prof) p.prof[v3_logical_block(blockIdx.x, p.fast) * PROF_SLOTS + (slot)] = (unsigned long long)clock64(); \
  } while (0)
#define PROF_SMID()                                                                       \
  do {                                                                                    \
    unsigned smid_;                                                                       \
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_));                                    \
    if (threadIdx.x == 0 && p.prof) p.prof[v3_logical_block(blockIdx.x, p.fast) * PROF_SLOTS + 12] = 1000000u + smid_; \
  } while (0)
#else
#define PROF_SMID() \
  do {              \
  } while (0)
#define PROF_AT(slot) \
  do {                \
  } while (0)
#define PROF_T0(slot) PROF_AT(slot)
#define PROF_MAX(slot) PROF_AT(slot)
#define PROF_CLK(slot) PROF_AT(slot)
#endif

// ---------------------------------------------------------------- sync helpers
// Light sync (FAST_LIGHT_SYNC, the PTX release/acquire patterns without the extra fences):
//  * waits poll with ld.relaxed.gpu (no L1 invalidation per poll: an ld.acquire is LDG + CCTL.IVALL, which
//    keeps wiping the L1 of the SM the neighbouring CTA works on) and acquire once after the target is seen;
//  * signals are a single red.release.gpu after the CTA barrier (release is cumulative over the writes the
//    barrier ordered before it), without the preceding __threadfence (fence.sc.gpu = MEMBAR.SC.GPU, which
//    also drains outstanding loads).
__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// LL words (DS41H_PROLOGUE_LL): a naturally aligned 64-bit relaxed access is single-copy atomic, so a word whose flag half
// equals this launch's flag carries this launch's data half (no producer fence, no consumer acquire: NCCL's LL protocol).
__device__ __forceinline__ void st_relaxed_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void ld_relaxed_v2u64(const unsigned long long* p, unsigned long long& a, unsigned long long& b) {
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
__device__ __forceinline__ unsigned long long ld_relaxed_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_relaxed(unsigned* p, unsigned v) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// Hang diagnosis (DS41H_PROLOGUE_HANGCHECK, default 1): every spin loop checks %globaltimer every 2048 polls and, after
// 4 s, prints what it waits for and traps -- a deadlock becomes a launch error with a message instead of a silent spin.
#ifndef DS41H_PROLOGUE_HANGCHECK
#define DS41H_PROLOGUE_HANGCHECK 1
#endif
struct HangCheck {
  unsigned long long t0 = 0;
  unsigned n = 0;
  __device__ __forceinline__ void tick(int where, unsigned value, unsigned target) {
#if DS41H_PROLOGUE_HANGCHECK
    if ((++n & 2047u) != 0) return;
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (t0 == 0) {
      t0 = t;
    } else if (t - t0 > 4000000000ull) {
      printf("ds41h prologue HANG: wait %d block %d thread %d value %u target %u\n", where, int(blockIdx.x), int(threadIdx.x), value, target);
      __trap();
    }
#endif
  }
};
// where codes: 1 spin_geq (C_P1B / C_P3K / ...), 2 cta_wait_ws, 3 weight-chunk mbarrier, 4 pushed-partials mbarrier,
// 7 LL word re-read (DS41H_PROLOGUE_LL)
__device__ __forceinline__ bool mbar_try(unsigned long long* bar, unsigned parity) {
  unsigned r;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(r)
               : "r"(smem_u32(bar)), "r"(parity)
               : "memory");
  return r != 0;
}
__device__ __forceinline__ void mbar_wait_hc(unsigned long long* bar, unsigned parity, int where) {
  HangCheck hc;
  while (!mbar_try(bar, parity)) hc.tick(where, 0u, parity);
}

// FAST_V3_FENCE: the acquire after a relaxed poll is a fence.acq_rel.gpu (PTX acquire pattern: a strong read followed by a
// fence) instead of a second load (ld.acquire = one more L2 round trip after the target was already observed).
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void spin_geq(const unsigned* p, unsigned target, bool light, bool fence = false) {
  HangCheck hc;
  if (light) {
    unsigned v;
    while ((v = ld_relaxed_gpu(p)) < target) hc.tick(1, v, target);
    if (fence)
      fence_acq_rel_gpu();
    else
      (void)ld_acquire_gpu(p);  // acquire pattern: synchronizes with the releases the relaxed poll observed
  } else {
    unsigned v;
    while ((v = ld_acquire_gpu(p)) < target) hc.tick(1, v, target);
  }
}
__device__ __forceinline__ void cta_wait(const unsigned* p, unsigned target, bool light) {
  if (threadIdx.x == 0) spin_geq(p, target, light);
  __syncthreads();
}
__device__ __forceinline__ void warp_wait(const unsigned* p, unsigned target, int lane, bool light, bool fence = false) {
  if (lane == 0) spin_geq(p, target, light, fence);
  __syncwarp();
}
__device__ __forceinline__ void cta_signal(unsigned* p, bool light) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (!light) __threadfence();
    red_release(p, 1u);
  }
}

// Attention handoff: one release increment of C_AQ by the calling thread, after the barrier that ordered the writes it
// covers (CTA: __syncthreads; KV warp: __syncwarp) -- release is cumulative over them.
__device__ __forceinline__ void aflag_signal(const unsigned* pc_base, bool on) {
  if (on) red_release(const_cast<unsigned*>(pc_base) + C_AQ, 1u);
}

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void unpack8(uint4 v, float (&f)[8]) {
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 h = unpack_bf16(u[i]);
    f[2 * i] = h.x;
    f[2 * i + 1] = h.y;
  }
}

// bf16 [8 * v8] (shared) -> f32 (shared), all threads of the CTA.
__device__ __forceinline__ void widen_bf16(float* dst, const __nv_bfloat16* src, int v8, int tid) {
  for (int i = tid; i < v8; i += THREADS) {
    float f[8];
    unpack8(*reinterpret_cast<const uint4*>(src + i * 8), f);
    reinterpret_cast<float4*>(dst)[2 * i] = make_float4(f[0], f[1], f[2], f[3]);
    reinterpret_cast<float4*>(dst)[2 * i + 1] = make_float4(f[4], f[5], f[6], f[7]);
  }
}

// ---------------------------------------------------------------- lane transposition (BRRR decoder.cu lane_sums / mix_reduce)
// v[i] on every lane -> lane L returns sum over lanes of v[L].
template <int W>
__device__ __forceinline__ void lane_sums_step(float (&v)[32], int lane) {
  const bool hi = lane & W;
#pragma unroll
  for (int i = 0; i < W; ++i) {
    const float keep = hi ? v[i + W] : v[i], send = hi ? v[i] : v[i + W];
    v[i] = keep + __shfl_xor_sync(FULL, send, W);
  }
}
__device__ __forceinline__ float lane_sums(float (&v)[32], int lane) {
  // compile-time strides so every v[] index is static and the array stays in registers
  lane_sums_step<16>(v, lane);
  lane_sums_step<8>(v, lane);
  lane_sums_step<4>(v, lane);
  lane_sums_step<2>(v, lane);
  lane_sums_step<1>(v, lane);
  return v[0];
}

// mixp[tok][s][0..31] summed over the SLICES slices; lane L returns the total of entry L.
// (Two L2 rounds of 20 independent 16-byte loads; the summation order is fixed: rows 0, 1, ..., 39.)
__device__ __forceinline__ float mix_reduce(const float* mixp, int tok, int lane) {
  constexpr int ROWS = SLICES / 4, BATCH = 20;
  static_assert(ROWS % BATCH == 0);
  const float4* src = reinterpret_cast<const float4*>(mixp + long(tok) * SLICES * MIXW) + (lane >> 3) * (MIXW / 4) + (lane & 7);
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
  for (int b = 0; b < ROWS; b += BATCH) {
    float4 v[BATCH];
#pragma unroll
    for (int i = 0; i < BATCH; ++i) v[i] = __ldcg(src + (b + i) * MIXW);  // slice 4(b+i) + (lane >> 3)
#pragma unroll
    for (int i = 0; i < BATCH; ++i) acc.x += v[i].x, acc.y += v[i].y, acc.z += v[i].z, acc.w += v[i].w;
  }
#pragma unroll
  for (int o = 8; o < 32; o <<= 1) {
    acc.x += __shfl_xor_sync(FULL, acc.x, o);
    acc.y += __shfl_xor_sync(FULL, acc.y, o);
    acc.z += __shfl_xor_sync(FULL, acc.z, o);
    acc.w += __shfl_xor_sync(FULL, acc.w, o);
  }
  const int from = (lane >> 2) & 7, part = lane & 3;
  const float a0 = __shfl_sync(FULL, acc.x, from), a1 = __shfl_sync(FULL, acc.y, from);
  const float a2 = __shfl_sync(FULL, acc.z, from), a3 = __shfl_sync(FULL, acc.w, from);
  return part == 0 ? a0 : part == 1 ? a1 : part == 2 ? a2 : a3;
}

// ---------------------------------------------------------------- P1a: mHC post + collapse, one (slice, token)
// Returns the fp32 post-mapped stream values of this column (the mixes of P1b read them).
// MODE 0: residual output + xcb / xsq scratch; MODE 2 (v3: the G1 CTAs compute the collapse and the residual
// output themselves): only nr (the fp32 stream values the mixes need).
template <int MODE = 0>
__device__ __forceinline__ void p1_collapse(const Params& p, int s, int tok, int lane, float (&nr)[HC], unsigned llflag = 0u) {
  const int col = s * 32 + lane;
  const float y = bf2f(__ldg(p.x + long(tok) * HID + col));
  float rk[HC];
#pragma unroll
  for (int k = 0; k < HC; ++k) rk[k] = bf2f(__ldg(p.resid + (long(tok) * HC + k) * HID + col));
  float post[HC], pre[HC], cm[HC * HC];
  {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p.post + tok * HC));
    const float4 b = __ldg(reinterpret_cast<const float4*>(p.pre + tok * HC));
    post[0] = a.x, post[1] = a.y, post[2] = a.z, post[3] = a.w;
    pre[0] = b.x, pre[1] = b.y, pre[2] = b.z, pre[3] = b.w;
#pragma unroll
    for (int r = 0; r < HC; ++r) {
      const float4 c = __ldg(reinterpret_cast<const float4*>(p.comb + tok * HC * HC + r * HC));
      cm[r * HC + 0] = c.x, cm[r * HC + 1] = c.y, cm[r * HC + 2] = c.z, cm[r * HC + 3] = c.w;
    }
  }
  // Stock fused post + pre-norm GEMM (tilelang_kernels.py:745-750): new_r[j] = post[j]*x, then
  // new_r[j] += comb[k, j] * residual[k] for k = 0..3 (fp32, contracted to FMA), stored as bf16.
  float rb[HC];
#pragma unroll
  for (int j = 0; j < HC; ++j) {
    float t = __fmul_rn(post[j], y);
#pragma unroll
    for (int k = 0; k < HC; ++k) t = __fmaf_rn(cm[k * HC + j], rk[k], t);
    nr[j] = t;
    const __nv_bfloat16 b = __float2bfloat16(t);
    if constexpr (MODE == 0 || MODE == 3) p.resid_out[(long(tok) * HC + j) * HID + col] = b;
    rb[j] = bf2f(b);
  }
  if constexpr (MODE != 0 && MODE != 3) return;
  if (p.aux != nullptr)  // capture_aux: bf16 mean over the 4 post-mapped bf16 streams (= stock residual.mean(dim=1); the fp32 sum of
                         // 4 bf16 values is exact, then / 4 exact, one bf16 rounding)
    p.aux[long(tok) * HID + col] = __float2bfloat16(((rb[0] + rb[1]) + (rb[2] + rb[3])) * 0.25f);
  // Collapse with the incoming pre-mix over the bf16 stream (tilelang_kernels.py:434-470): ol = 0;
  // ol += pre[j] * x[j]; rounded to bf16 before the RMSNorm.
  float xc = __fmul_rn(pre[0], rb[0]);
#pragma unroll
  for (int j = 1; j < HC; ++j) xc = __fmaf_rn(pre[j], rb[j], xc);
  const __nv_bfloat16 xb = __float2bfloat16(xc);
  if (llflag != 0u) {  // DS41H_PROLOGUE_LL & 2: word (tok, s, lane / 2) = {bf16 col 2w | bf16 col 2w + 1 << 16, flag}
    const unsigned mine = __bfloat16_as_ushort(xb), next = __shfl_down_sync(FULL, mine, 1);
    if ((lane & 1) == 0)
      st_relaxed_u64(p.xll + (long(tok) * SLICES + s) * 16 + (lane >> 1), ((unsigned long long)llflag << 32) | mine | (next << 16));
  }
  p.xcb[long(tok) * HID + col] = xb;
  const float xf = bf2f(xb);
  if constexpr (MODE == 3) {
    // u = bf16(collapse * attn_norm_w) quantized per (token, this slice = one k32 block) into the shared B tile (layout
    // of stage_e4m3: physical column 8 t4 + j -> byte 4 t4 + j (j < 4) or 128 + 4 t4 + j - 4 of token row tok)
    const float u = bf16_round(__fmul_rn(xf, bf2f(__ldg(p.attn_norm + col))));
    const float am = warp_max(fabsf(u));
    const int e = quant_exp(am, AMAX_FLOOR);
    const int t4 = lane >> 3, jj = lane & 7;
    p.qab[long(s) * WG_KB_PITCH + tok * 16 + (jj < 4 ? 4 * t4 + jj : 128 + 4 * t4 + jj - 4)] = uint8_t(e4m3x2(u * pow2(-e), 0.0f) & 0xffu);
    if (lane == 0) p.qas[s * MAXM + tok] = pow2(e);
  }
  // Collapse square sum of the 32 columns (attn_norm statistics). The xor butterfly leaves the same value on
  // every lane, equal (operand for operand) to entry 25 of the Phase-1 lane transposition: bit-identical xsq.
  const float xs = warp_sum(xf * xf);
  if (llflag != 0u && lane == 0) st_relaxed_u64(p.xsqll + tok * SLICES + s, ((unsigned long long)llflag << 32) | __float_as_uint(xs));
  if (lane == 0) __stcg(p.xsq + tok * SLICES + s, xs);
}

// ---------------------------------------------------------------- P1b: partial mixes, one (slice, token)
// FS == false: hc_attn_fn straight from global memory (96 LDG.32 per lane). FS == true: from the CTA's shared copy
// fs = f32 [MIX][HC][32] of this slice (p1_stage_fn, cp.async), 96 LDS.32 per lane: the jobs of the m tokens of a
// slice share one 12 KB copy instead of each pulling it from L2. Same values, same FMA order: bit-identical.
template <bool FS>
__device__ __forceinline__ void p1_mixes(const Params& p, const float* fs, int s, int tok, int lane, const float (&nr)[HC]) {
  const int col = s * 32 + lane;
  // Per-lane values: entries 0..23 = fn[q] . new_r (this column), 24 = sum new_r^2, 25..31 = 0; summed
  // over the 32 lanes (columns) with the lane transposition. The first transposition step (lane bit 16
  // pairs entry i with i + 16) is folded into the dot-product loop so only 16 partials are live at a time
  // (keeps the 96 fn loads from spilling).
  const float nsq = nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2] + nr[3] * nr[3];
  const bool hi16 = lane & 16;
  // The fn pointer is warp-uniform, so a shuffle from any lane returns it unchanged; taking the source
  // lane from nsq makes the 96 fn loads depend on the stream math, so ptxas cannot hoist them above it
  // (that kept the residual/mix temporaries live next to 96 loads in flight and spilled).
  const float* fn = reinterpret_cast<const float*>(
      __shfl_sync(FULL, reinterpret_cast<unsigned long long>(p.fn), nsq < 0.0f ? 1 : 0));
  const unsigned fsa = FS ? __shfl_sync(FULL, smem_u32(fs), nsq < 0.0f ? 1 : 0) + lane * 4 : 0u;
  // fn[q][j] for this lane's column
  auto dot = [&](int q) {
    float acc = 0.0f;
    if constexpr (FS) {
      float f[HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) asm volatile("ld.shared.f32 %0, [%1];" : "=f"(f[j]) : "r"(fsa + (q * HC + j) * 128) : "memory");
#pragma unroll
      for (int j = 0; j < HC; ++j) acc = fmaf(f[j], nr[j], acc);
    } else {
#pragma unroll
      for (int j = 0; j < HC; ++j) acc = fmaf(__ldg(fn + col + long(q) * HCD + j * HID), nr[j], acc);
    }
    return acc;
  };
  float v[32];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float lo = dot(i);
    float up = 0.0f;
    if (i + 16 < MIX) {
      up = dot(i + 16);
    } else if (i + 16 == MIX) {
      up = nsq;
    }
    v[i] = (hi16 ? up : lo) + __shfl_xor_sync(FULL, hi16 ? lo : up, 16);
  }
#pragma unroll
  for (int i = 16; i < 32; ++i) v[i] = 0.0f;  // unused by the remaining steps
  lane_sums_step<8>(v, lane);
  lane_sums_step<4>(v, lane);
  lane_sums_step<2>(v, lane);
  lane_sums_step<1>(v, lane);
  __stcg(p.mixp + (long(tok) * SLICES + s) * MIXW + lane, v[0]);
}

__device__ __forceinline__ void prefetch_l2(const void* ptr) { asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr)); }
__device__ __forceinline__ void prefetch_l2_bulk(const void* ptr, unsigned bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(ptr), "r"(bytes) : "memory");
}

// L2 prefetch of G2 CTA c's wq_b slice: 4 consecutive tiles (80 KB, contiguous) + their 160 scale bytes.
__device__ __forceinline__ void prefetch_wqb(const Params& p, int c, int tid) {
  constexpr unsigned PART = G2_TILES * KB2 * CHUNK / 5;
  static_assert(PART % 16 == 0);
  if (tid < 5) prefetch_l2_bulk(p.wqb + long(c) * G2_TILES * KB2 * CHUNK + tid * PART, PART);
  if (tid == 5 || tid == 6) prefetch_l2(p.wqb_s + long(c) * G2_TILES * KB2 + (tid - 5) * (G2_TILES * KB2 - 1));  // both lines
}

// P1 jobs of G2 CTA c: worker wid = 8c + warp owns (slice wid / ntg, tokens tpj * (wid % ntg) ...).
struct P1Job {
  int s = 0, tok0 = 0, tok1 = 0;  // tok1 == tok0: no job
};
__device__ __forceinline__ P1Job p1_job(const Params& p, int wid) {
  P1Job j;
  const int ntg = (p.m + p.tpj - 1) / p.tpj;
  if (wid < SLICES * ntg) {
    j.s = wid / ntg;
    j.tok0 = (wid - j.s * ntg) * p.tpj;
    j.tok1 = min(p.m, j.tok0 + p.tpj);
  }
  return j;
}
// Before the PDL wait (weights only): pull the hc_attn_fn lines the CTA's P1b jobs will read into L2.
__device__ __forceinline__ void p1_prefetch_fn(const Params& p, int c, int tid) {
  const int ntg = (p.m + p.tpj - 1) / p.tpj, w0 = c * WARPS;
  if (w0 >= SLICES * ntg) return;
  const int s0 = w0 / ntg, s1 = min(SLICES * ntg - 1, w0 + WARPS - 1) / ntg;
  for (int q = tid; q < (s1 - s0 + 1) * MIX * HC; q += THREADS) {
    const int s = s0 + q / (MIX * HC), r = q % (MIX * HC);
    prefetch_l2(p.fn + long(r >> 2) * HCD + (r & 3) * HID + s * 32);  // one 128-byte line
  }
}
// Slices [s0, s0 + ns) covered by the P1 jobs of G2 CTA c (ns = 0: no jobs).
struct P1Range {
  int s0 = 0, ns = 0;
};
__device__ __forceinline__ P1Range p1_range(const Params& p, int c) {
  const int ntg = (p.m + p.tpj - 1) / p.tpj, w0 = c * WARPS;
  P1Range r;
  if (w0 < SLICES * ntg) {
    r.s0 = w0 / ntg;
    r.ns = min(SLICES * ntg - 1, w0 + WARPS - 1) / ntg - r.s0 + 1;
  }
  return r;
}
constexpr int FN_SLICE_BYTES = MIX * HC * 32 * 4;  // 12 KB of hc_attn_fn per 32-column slice
template <int MMA>
constexpr int fn_stage_max() {  // slices whose fn fits the CTA's shared memory (free until P1 ends)
  return (MMA == MMA_WGMMA ? SMEM_BYTES_WG : SMEM_BYTES) / FN_SLICE_BYTES;
}
// Before the PDL wait (weights only): cp.async the CTA's fn slices into shared memory as f32 [ns][MIX][HC][32] (the
// 128-byte rows of hc_attn_fn); asynchronous, waited for (cp_async_wait_all + barrier) before P1b reads it.
__device__ __forceinline__ void p1_stage_fn(const Params& p, float* fs, const P1Range& r, int tid) {
  for (int i = tid; i < r.ns * MIX * HC * 8; i += THREADS) {  // 16-byte chunk i = ((si * MIX + q) * HC + j) * 8 + c8
    const int c8 = i & 7, j = (i >> 3) & 3, qs = i >> 5, q = qs % MIX, si = qs / MIX;
    cp_async16(fs + (qs * HC + j) * 32 + c8 * 4, p.fn + long(q) * HCD + j * HID + (r.s0 + si) * 32 + c8 * 4);
  }
  cp_async_commit();
}
// All warps of G2 CTA c: P1a for the CTA's jobs -> C_P1A, then P1b -> C_P1B (one release per CTA).
// fs: the staged fn slices (p1_stage_fn) or nullptr (P1b reads global memory).
__device__ __forceinline__ void p1_cta(const Params& p, const float* fs, int c, int tid, int warp, int lane) {
  const P1Job j = p1_job(p, c * WARPS + warp);
  float nr[2][HC];
#pragma unroll
  for (int t = 0; t < 2; ++t)
    if (j.tok0 + t < j.tok1) p1_collapse(p, j.s, j.tok0 + t, lane, nr[t]);
  PROF_T0(9);  // G2: thread 0's P1a job issued
  if (fs != nullptr) cp_async_wait_all();  // the staged fn slices (p1_stage_fn) are complete after the barrier
  __syncthreads();
  PROF_T0(10);  // G2: all P1a jobs of the CTA issued
  if (tid == 0) {
    if (!(p.fast & FAST_LIGHT_SYNC)) __threadfence();
    PROF_AT(11);  // G2: P1a fence done
    red_release(p.pc + C_P1A, 1u);
  }
  PROF_T0(2);  // G2: P1a published
  // The CTA's 80 KB wq_b slice (4 consecutive tiles, contiguous) + scales start streaming into L2 now, so
  // the register loads after P1b hit L2 instead of starting the HBM stream late (FAST_PREFETCH: already
  // requested at kernel entry).
  if (!(p.fast & FAST_PREFETCH)) prefetch_wqb(p, c, tid);
  if (fs != nullptr) {  // CTA-uniform
    const float* f = fs + (j.s - p1_range(p, c).s0) * (FN_SLICE_BYTES / 4);
#pragma unroll
    for (int t = 0; t < 2; ++t)
      if (j.tok0 + t < j.tok1) p1_mixes<true>(p, f, j.s, j.tok0 + t, lane, nr[t]);
  } else {
#pragma unroll
    for (int t = 0; t < 2; ++t)
      if (j.tok0 + t < j.tok1) p1_mixes<false>(p, nullptr, j.s, j.tok0 + t, lane, nr[t]);
  }
  cta_signal(p.pc + C_P1B, p.fast & FAST_LIGHT_SYNC);
  PROF_T0(3);  // G2: P1b published
}

// ---------------------------------------------------------------- H: Sinkhorn of the attention mixes (one warp per token)
// Stock's exp: the TileLang mHC kernels are compiled WITHOUT fast math (tilelang 0.1.12: TL_ENABLE_FAST_MATH defaults
// to False), so their expf is libdevice's accurate expf (<= 2 ulp over the whole range). ds41h extensions build with
// --use_fast_math, which turns expf into __expf = ex2.approx(x * log2 e) whose error grows with |x| (2 + 1.17|x| ulp),
// so the target math calls the libdevice routine directly (same algorithm as stock; only denormal results flush).
extern "C" __device__ float __nv_expf(float);
__device__ __forceinline__ float exp_stock(float x) { return __nv_expf(x); }
// stock: 1.0f / (1.0f + expf(0.0f - z)) (IEEE division)
__device__ __forceinline__ float sigmoidf_(float x) { return __fdiv_rn(1.0f, 1.0f + exp_stock(-x)); }

// Stock pre-norm epilogue (tilelang_kernels.py:350-412): rms over the 20480 fp32 stream values,
// pre = sigmoid(m*s0 + b) + eps, post = sigmoid(m*s1 + b) * 2, comb = row softmax + eps, column
// normalize, then (iters - 1) x (row, column) normalizations, all with + eps in the denominators.
__device__ __forceinline__ void sinkhorn_job(const Params& p, int tok, int lane) {
  const float tot = mix_reduce(p.mixp, tok, lane);
  const float sq = __shfl_sync(FULL, tot, MIX);
  const float rms = rsqrtf(__fdiv_rn(sq, float(HCD)) + p.rms_eps);
  const float mixv = __fmul_rn(tot, rms);
  if (lane < HC) {
    p.pre_out[tok * HC + lane] = sigmoidf_(fmaf(mixv, __ldg(p.hc_scale + 0), __ldg(p.hc_base + lane))) + p.hc_eps;
  } else if (lane < 2 * HC) {
    p.post_out[tok * HC + lane - HC] = sigmoidf_(fmaf(mixv, __ldg(p.hc_scale + 1), __ldg(p.hc_base + lane))) * p.post_mult;
  }
  // comb logits on lanes 16..31 (lane 16 + 4j + k holds comb[j][k]; row = xor 1,2, column = xor 4,8).
  const float cmv = __shfl_sync(FULL, mixv, (lane - 8) & 31);
  const int ci = lane & 15;
  const float x = fmaf(cmv, __ldg(p.hc_scale + 2), __ldg(p.hc_base + 2 * HC + ci));
  float mx = fmaxf(x, __shfl_xor_sync(FULL, x, 1));
  mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 2));
  const float e = exp_stock(x - mx);
  float rsum = e + __shfl_xor_sync(FULL, e, 1);
  rsum += __shfl_xor_sync(FULL, rsum, 2);
  float c = __fdiv_rn(e, rsum) + p.hc_eps;
  float csum = c + __shfl_xor_sync(FULL, c, 4);
  csum += __shfl_xor_sync(FULL, csum, 8);
  c = __fdiv_rn(c, csum + p.hc_eps);
  for (int it = 1; it < p.iters; ++it) {
    rsum = c + __shfl_xor_sync(FULL, c, 1);
    rsum += __shfl_xor_sync(FULL, rsum, 2);
    c = __fdiv_rn(c, rsum + p.hc_eps);
    csum = c + __shfl_xor_sync(FULL, c, 4);
    csum += __shfl_xor_sync(FULL, csum, 8);
    c = __fdiv_rn(c, csum + p.hc_eps);
  }
  if (lane >= 16) p.comb_out[tok * HC * HC + ci] = c;
}

// Exact Sinkhorn with one ROW per lane (lane 4g + j holds comb row j in registers; the 8 lane groups compute the same
// token): the row normalizations need no shuffles, the column ones 2 rounds of 4 independent shuffles, instead of
// 2 dependent shuffles + 1 division per half-iteration on one element per lane. Same operations, same IEEE divisions
// and the same summation order as sinkhorn_job: row sums (c0+c1)+(c2+c3), column sums (r0+r1)+(r2+r3) -> bit-identical.
__device__ __forceinline__ void sinkhorn_rows(const Params& p, int tok, int lane) {
  const float tot = mix_reduce(p.mixp, tok, lane);
  const float sq = __shfl_sync(FULL, tot, MIX);
  const float rms = rsqrtf(__fdiv_rn(sq, float(HCD)) + p.rms_eps);
  const float mixv = __fmul_rn(tot, rms);
  if (lane < HC) {
    p.pre_out[tok * HC + lane] = sigmoidf_(fmaf(mixv, __ldg(p.hc_scale + 0), __ldg(p.hc_base + lane))) + p.hc_eps;
  } else if (lane < 2 * HC) {
    p.post_out[tok * HC + lane - HC] = sigmoidf_(fmaf(mixv, __ldg(p.hc_scale + 1), __ldg(p.hc_base + lane))) * p.post_mult;
  }
  const int j = lane & 3;
  const float s2 = __ldg(p.hc_scale + 2), eps = p.hc_eps;
  float c[HC];
#pragma unroll
  for (int k = 0; k < HC; ++k) c[k] = fmaf(__shfl_sync(FULL, mixv, 2 * HC + 4 * j + k), s2, __ldg(p.hc_base + 2 * HC + 4 * j + k));
  {
    const float mx = fmaxf(fmaxf(c[0], c[1]), fmaxf(c[2], c[3]));  // max: order-independent
    float e[HC];
#pragma unroll
    for (int k = 0; k < HC; ++k) e[k] = exp_stock(c[k] - mx);
    const float rs = (e[0] + e[1]) + (e[2] + e[3]);
#pragma unroll
    for (int k = 0; k < HC; ++k) c[k] = __fdiv_rn(e[k], rs) + eps;
  }
  auto cols = [&]() {
    float s[HC];
#pragma unroll
    for (int k = 0; k < HC; ++k) s[k] = c[k] + __shfl_xor_sync(FULL, c[k], 1);
#pragma unroll
    for (int k = 0; k < HC; ++k) s[k] += __shfl_xor_sync(FULL, s[k], 2);
#pragma unroll
    for (int k = 0; k < HC; ++k) c[k] = __fdiv_rn(c[k], s[k] + eps);
  };
  cols();
  for (int it = 1; it < p.iters; ++it) {
    const float rs = ((c[0] + c[1]) + (c[2] + c[3])) + eps;
#pragma unroll
    for (int k = 0; k < HC; ++k) c[k] = __fdiv_rn(c[k], rs);
    cols();
  }
  if (lane < HC) *reinterpret_cast<float4*>(p.comb_out + tok * HC * HC + 4 * j) = make_float4(c[0], c[1], c[2], c[3]);
}

// ---------------------------------------------------------------- H: Sinkhorn with shared reciprocals (FAST_V3_SKREG / SKROW)
// IEEE-exact division with a shared reciprocal: the fast path of div.rn.f32 as nvcc expands __fdiv_rn (r0 = MUFU.RCP(b),
// r1 = r0 + r0 (1 - b r0), q0 = a r1, q = q0 + r1 (a - b q0)) returns the correctly rounded quotient whenever FCHK lets it
// (normal operands, quotient far from the subnormal / overflow range); the reciprocal part depends on b only, so the 4
// divisions of a Sinkhorn row / column by the same sum share it. Correct rounding is unique, so the results equal __fdiv_rn
// bit for bit. Range argument (checked at run time, never in the division itself): every Sinkhorn division a / b has
// 0 <= a <= b (a is one of the non-negative terms of the sum b), so q is in [a / b, 1] with b <= 4 (1 + eps) + eps (softmax
// step: b = sum e in [1, 4]). The callers verify eps in [2^-60, 1] (every later b >= eps), every softmax numerator e == 0 or
// >= 2^-80, every value >= 2^-80 after each column normalization (so every numerator of the next row / column step is
// >= 2^-80 / 25) and no NaN at the end; anything else recomputes the token with sinkhorn_job (__fdiv_rn).
struct Rcp {
  float b, r1;
};
__device__ __forceinline__ Rcp rcp_prep(float b) {
  Rcp r;
  r.b = b;
  float r0;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r0) : "f"(b));
  const float e = __fmaf_rn(-b, r0, 1.0f);
  r.r1 = __fmaf_rn(e, r0, r0);
  return r;
}
__device__ __forceinline__ float div_rcp(float a, const Rcp& r) {
  const float q0 = __fmul_rn(a, r.r1);
  const float rem = __fmaf_rn(-r.b, q0, a);
  return __fmaf_rn(r.r1, rem, q0);
}
constexpr float SK_MIN = 8.271806125530277e-25f;  // 2^-80
__device__ __forceinline__ bool sk_eps_ok(float eps) { return eps >= 8.673617379884035e-19f && eps <= 1.0f; }  // [2^-60, 1]
__device__ __forceinline__ bool sk_e_ok(float e) { return e == 0.0f || e >= SK_MIN; }
__device__ __forceinline__ float min4(float a, float b, float c, float d) { return fminf(fminf(a, b), fminf(c, d)); }

// FAST_V3_SKREG: one token's Sinkhorn with the whole 4x4 matrix in registers (every lane computes it redundantly, lane 0
// stores it): 16 independent chains per half-iteration, no shuffles. Same operations / order as sinkhorn_job: row sums
// (c0+c1)+(c2+c3), column sums (r0+r1)+(r2+r3), correctly rounded divisions -> bit-identical.
__device__ __forceinline__ void sk_rows_rcp(float (&c)[16], float eps) {
#pragma unroll
  for (int r = 0; r < HC; ++r) {
    const Rcp rs = rcp_prep(((c[4 * r] + c[4 * r + 1]) + (c[4 * r + 2] + c[4 * r + 3])) + eps);
#pragma unroll
    for (int k = 0; k < HC; ++k) c[4 * r + k] = div_rcp(c[4 * r + k], rs);
  }
}
__device__ __forceinline__ float sk_cols_rcp(float (&c)[16], float eps) {  // returns the matrix minimum
#pragma unroll
  for (int k = 0; k < HC; ++k) {
    const Rcp cs = rcp_prep(((c[k] + c[4 + k]) + (c[8 + k] + c[12 + k])) + eps);
#pragma unroll
    for (int r = 0; r < HC; ++r) c[4 * r + k] = div_rcp(c[4 * r + k], cs);
  }
  return fminf(fminf(min4(c[0], c[1], c[2], c[3]), min4(c[4], c[5], c[6], c[7])),
               fminf(min4(c[8], c[9], c[10], c[11]), min4(c[12], c[13], c[14], c[15])));
}
__device__ __forceinline__ void sinkhorn_regs(const Params& p, int tok, int lane) {
  const float tot = mix_reduce(p.mixp, tok, lane);
  const float sq = __shfl_sync(FULL, tot, MIX);
  const float rms = rsqrtf(__fdiv_rn(sq, float(HCD)) + p.rms_eps);
  const float mixv = __fmul_rn(tot, rms);
  if (lane < HC) {
    p.pre_out[tok * HC + lane] = sigmoidf_(fmaf(mixv, __ldg(p.hc_scale + 0), __ldg(p.hc_base + lane))) + p.hc_eps;
  } else if (lane < 2 * HC) {
    p.post_out[tok * HC + lane - HC] = sigmoidf_(fmaf(mixv, __ldg(p.hc_scale + 1), __ldg(p.hc_base + lane))) * p.post_mult;
  }
  const float s2 = __ldg(p.hc_scale + 2), eps = p.hc_eps;
  bool ok = sk_eps_ok(eps);
  float c[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) c[i] = fmaf(__shfl_sync(FULL, mixv, 2 * HC + i), s2, __ldg(p.hc_base + 2 * HC + i));
#pragma unroll
  for (int r = 0; r < HC; ++r) {  // row softmax + eps
    const float mx = fmaxf(fmaxf(c[4 * r], c[4 * r + 1]), fmaxf(c[4 * r + 2], c[4 * r + 3]));
    float e[HC];
#pragma unroll
    for (int k = 0; k < HC; ++k) e[k] = exp_stock(c[4 * r + k] - mx);
    ok = ok && sk_e_ok(e[0]) && sk_e_ok(e[1]) && sk_e_ok(e[2]) && sk_e_ok(e[3]);
    const Rcp rs = rcp_prep((e[0] + e[1]) + (e[2] + e[3]));
#pragma unroll
    for (int k = 0; k < HC; ++k) c[4 * r + k] = div_rcp(e[k], rs) + eps;
  }
  float mn = sk_cols_rcp(c, eps);
#pragma unroll 1
  for (int it = 1; it < p.iters; ++it) {
    sk_rows_rcp(c, eps);
    mn = fminf(mn, sk_cols_rcp(c, eps));
  }
  bool fin = true;
#pragma unroll
  for (int i = 0; i < 16; ++i) fin = fin && c[i] == c[i];
  if (!__all_sync(FULL, ok && fin && mn >= SK_MIN)) {  // outside the fast path's checked range (see Rcp): reference schedule
    sinkhorn_job(p, tok, lane);
    return;
  }
  if (lane == 0) {  // static indices only: a lane-indexed pick makes nvcc demote c[] to local memory
    float4* o = reinterpret_cast<float4*>(p.comb_out + tok * HC * HC);
#pragma unroll
    for (int r = 0; r < HC; ++r) o[r] = make_float4(c[4 * r], c[4 * r + 1], c[4 * r + 2], c[4 * r + 3]);
  }
}

// FAST_V3_SKROW: sinkhorn_rows' layout (lane 4g + j holds row j; the 8 lane groups compute the same token) with shared
// reciprocals: a row step is 1 reciprocal + 4 divisions without shuffles, a column step 2 rounds of 4 independent shuffles
// + 4 reciprocals + 4 divisions (~65 instructions per iteration vs ~160 for SKREG: better when the SM's issue slots are
// shared with a co-resident CTA). Same sums and division results as sinkhorn_job -> bit-identical.
__device__ __forceinline__ void sinkhorn_rows_rcp(const Params& p, int tok, int lane) {
  const float tot = mix_reduce(p.mixp, tok, lane);
  const float sq = __shfl_sync(FULL, tot, MIX);
  const float rms = rsqrtf(__fdiv_rn(sq, float(HCD)) + p.rms_eps);
  const float mixv = __fmul_rn(tot, rms);
  if (lane < HC) {
    p.pre_out[tok * HC + lane] = sigmoidf_(fmaf(mixv, __ldg(p.hc_scale + 0), __ldg(p.hc_base + lane))) + p.hc_eps;
  } else if (lane < 2 * HC) {
    p.post_out[tok * HC + lane - HC] = sigmoidf_(fmaf(mixv, __ldg(p.hc_scale + 1), __ldg(p.hc_base + lane))) * p.post_mult;
  }
  const int j = lane & 3;
  const float s2 = __ldg(p.hc_scale + 2), eps = p.hc_eps;
  bool ok = sk_eps_ok(eps);
  float c[HC];
#pragma unroll
  for (int k = 0; k < HC; ++k) c[k] = fmaf(__shfl_sync(FULL, mixv, 2 * HC + 4 * j + k), s2, __ldg(p.hc_base + 2 * HC + 4 * j + k));
  {
    const float mx = fmaxf(fmaxf(c[0], c[1]), fmaxf(c[2], c[3]));  // max: order-independent
    float e[HC];
#pragma unroll
    for (int k = 0; k < HC; ++k) e[k] = exp_stock(c[k] - mx);
    ok = ok && sk_e_ok(e[0]) && sk_e_ok(e[1]) && sk_e_ok(e[2]) && sk_e_ok(e[3]);
    const Rcp rs = rcp_prep((e[0] + e[1]) + (e[2] + e[3]));
#pragma unroll
    for (int k = 0; k < HC; ++k) c[k] = div_rcp(e[k], rs) + eps;
  }
  auto cols = [&]() {
    float s[HC];
#pragma unroll
    for (int k = 0; k < HC; ++k) s[k] = c[k] + __shfl_xor_sync(FULL, c[k], 1);
#pragma unroll
    for (int k = 0; k < HC; ++k) s[k] += __shfl_xor_sync(FULL, s[k], 2);
#pragma unroll
    for (int k = 0; k < HC; ++k) c[k] = div_rcp(c[k], rcp_prep(s[k] + eps));
    return min4(c[0], c[1], c[2], c[3]);
  };
  float mn = cols();
#pragma unroll 1
  for (int it = 1; it < p.iters; ++it) {
    const Rcp rs = rcp_prep(((c[0] + c[1]) + (c[2] + c[3])) + eps);
#pragma unroll
    for (int k = 0; k < HC; ++k) c[k] = div_rcp(c[k], rs);
    mn = fminf(mn, cols());
  }
  const bool fin = c[0] == c[0] && c[1] == c[1] && c[2] == c[2] && c[3] == c[3];
  if (!__all_sync(FULL, ok && fin && mn >= SK_MIN)) {  // outside the fast path's checked range (see Rcp): reference schedule
    sinkhorn_job(p, tok, lane);
    return;
  }
  if (lane < HC) *reinterpret_cast<float4*>(p.comb_out + tok * HC * HC + 4 * j) = make_float4(c[0], c[1], c[2], c[3]);
}

// ---------------------------------------------------------------- H: fast Sinkhorn (FAST_SINKHORN)
__device__ __forceinline__ float rcp_approx(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// Step 1 (one warp per token): mixes, rms, pre / post (exactly as sinkhorn_job); the 16 comb logits go
// to shared memory cl[tok][16] for step 2.
__device__ __forceinline__ void sinkhorn_mix(const Params& p, int tok, int lane, float* cl) {
  const float tot = mix_reduce(p.mixp, tok, lane);
  const float sq = __shfl_sync(FULL, tot, MIX);
  const float rms = rsqrtf(__fdiv_rn(sq, float(HCD)) + p.rms_eps);
  const float mixv = __fmul_rn(tot, rms);
  if (lane < HC) {
    p.pre_out[tok * HC + lane] = sigmoidf_(fmaf(mixv, __ldg(p.hc_scale + 0), __ldg(p.hc_base + lane))) + p.hc_eps;
  } else if (lane < 2 * HC) {
    p.post_out[tok * HC + lane - HC] = sigmoidf_(fmaf(mixv, __ldg(p.hc_scale + 1), __ldg(p.hc_base + lane))) * p.post_mult;
  } else if (lane < MIX) {
    cl[tok * 16 + lane - 2 * HC] = mixv;
  }
}

// Row normalization c[r][*] /= (sum_k c[r][k] + eps) with the stock summation order (c0+c1)+(c2+c3).
__device__ __forceinline__ void sk_rows(float (&c)[16], float eps) {
#pragma unroll
  for (int r = 0; r < HC; ++r) {
    const float inv = rcp_approx(((c[4 * r] + c[4 * r + 1]) + (c[4 * r + 2] + c[4 * r + 3])) + eps);
#pragma unroll
    for (int k = 0; k < HC; ++k) c[4 * r + k] *= inv;
  }
}
__device__ __forceinline__ void sk_cols(float (&c)[16], float eps) {
#pragma unroll
  for (int k = 0; k < HC; ++k) {
    const float inv = rcp_approx(((c[k] + c[4 + k]) + (c[8 + k] + c[12 + k])) + eps);
#pragma unroll
    for (int r = 0; r < HC; ++r) c[4 * r + k] *= inv;
  }
}

// Step 2 (one warp, all tokens): lane L iterates token L & 7's 4x4 matrix in registers (lanes >= m
// duplicate a valid token and store nothing). Same operation order as sinkhorn_job, with each division
// replaced by a multiply with rcp.approx of the row / column denominator (<= 2 ulp per step; stock
// tolerance for comb is ~5e-5 absolute).
__device__ __forceinline__ void sinkhorn_fast(const Params& p, const float* cl, int lane) {
  const int tok = min(lane & 7, p.m - 1);
  const float s2 = __ldg(p.hc_scale + 2), eps = p.hc_eps;
  float c[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) c[i] = fmaf(cl[tok * 16 + i], s2, __ldg(p.hc_base + 2 * HC + i));
#pragma unroll
  for (int r = 0; r < HC; ++r) {
    const float mx = fmaxf(fmaxf(c[4 * r], c[4 * r + 1]), fmaxf(c[4 * r + 2], c[4 * r + 3]));
    float e[HC];
#pragma unroll
    for (int k = 0; k < HC; ++k) e[k] = __expf(c[4 * r + k] - mx);
    const float inv = rcp_approx((e[0] + e[1]) + (e[2] + e[3]));
#pragma unroll
    for (int k = 0; k < HC; ++k) c[4 * r + k] = fmaf(e[k], inv, eps);
  }
  sk_cols(c, eps);
  for (int it = 1; it < p.iters; ++it) {
    sk_rows(c, eps);
    sk_cols(c, eps);
  }
  if (lane < p.m) {
    float4* dst = reinterpret_cast<float4*>(p.comb_out + tok * HC * HC);
#pragma unroll
    for (int r = 0; r < HC; ++r) dst[r] = make_float4(c[4 * r], c[4 * r + 1], c[4 * r + 2], c[4 * r + 3]);
  }
}

// ---------------------------------------------------------------- H: kv_norm + RoPE + V4 record insert (one warp per token)
// Lane L owns dims [16L, 16L + 16) exactly like stock processDeepseekV4Slot, so the amax groups
// (4 lanes = 64 dims), rounding points and byte layout match by construction.
__device__ __forceinline__ void kv_job(const Params& p, int tok, int lane) {
  // every load of the job is issued up front (one L2 round trip instead of three dependent ones)
  const int d0 = lane * 16;
  const uint4* src = reinterpret_cast<const uint4*>(p.qkv + long(tok) * QKV + QR + d0);
  const float sq0 = __ldcg(p.qsq + (Q_TILES + lane) * MAXM + tok);
  const uint4 r0 = __ldcg(src), r1 = __ldcg(src + 1);
  const uint4 w0 = __ldg(reinterpret_cast<const uint4*>(p.kv_norm + d0)), w1 = __ldg(reinterpret_cast<const uint4*>(p.kv_norm + d0) + 1);
  const long pos = p.positions[tok];
  const float sq = warp_sum(sq0);
  // fused_q_kv_rmsnorm (models/common/ops/fused_qk_rmsnorm.py:76-91): y = x * rsqrt(sum/N + eps) * w.
  const float rr = rsqrtf(__fdiv_rn(sq, float(KVW)) + p.qkv_eps);
  float e[16], w[16];
  {
    float t[8];
    unpack8(r0, t);
#pragma unroll
    for (int i = 0; i < 8; ++i) e[i] = t[i];
    unpack8(r1, t);
#pragma unroll
    for (int i = 0; i < 8; ++i) e[8 + i] = t[i];
    unpack8(w0, t);
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = t[i];
    unpack8(w1, t);
#pragma unroll
    for (int i = 0; i < 8; ++i) w[8 + i] = t[i];
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) e[i] = bf16_round(__fmul_rn(__fmul_rn(e[i], rr), w[i]));
  uint4* dst = reinterpret_cast<uint4*>(p.kv + long(tok) * KVW + d0);
  dst[0] = pack8_bf16(reinterpret_cast<const float(&)[8]>(e[0]));
  dst[1] = pack8_bf16(reinterpret_cast<const float(&)[8]>(e[8]));
  if (p.swa == nullptr) return;  // profile run: stock skips the insert (attention.py:1026-1042)

  // GPT-J RoPE on dims [448, 512): even' = fma(e, cos, -(o*sin)), odd' = fma(e, sin, o*cos) -- the
  // contraction stock's sm_90 SASS uses (FMUL + FFMA per output).
  if (d0 >= NOPE) {
    const float* cs = p.cos_sin + pos * ROPE;
    const int hb = (d0 - NOPE) >> 1;
    const float4 c0 = __ldg(reinterpret_cast<const float4*>(cs + hb)), c1 = __ldg(reinterpret_cast<const float4*>(cs + hb + 4));
    const float4 s0 = __ldg(reinterpret_cast<const float4*>(cs + ROPE / 2 + hb)), s1 = __ldg(reinterpret_cast<const float4*>(cs + ROPE / 2 + hb + 4));
    const float cv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float xe = e[2 * i], xo = e[2 * i + 1];
      e[2 * i] = __fmaf_rn(xe, cv[i], -__fmul_rn(xo, sv[i]));
      e[2 * i + 1] = __fmaf_rn(xe, sv[i], __fmul_rn(xo, cv[i]));
    }
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) e[i] = bf16_round(e[i]);
  float am = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(e[i]));
  am = fmaxf(am, __shfl_xor_sync(FULL, am, 1));
  am = fmaxf(am, __shfl_xor_sync(FULL, am, 2));
  am = fmaxf(am, 1e-4f);
  // ceilf(log2f(am / 448)) exactly (am is bf16-valued or the 1e-4 floor; see common.cuh quant_exp).
  const int ex = quant_exp(am, 1e-4f);
  if (tok >= p.n_insert) return;
  const long slot = p.slots[tok];
  if (slot < 0) return;
  const long blk = slot / p.swa_block, pib = slot - blk * p.swa_block;
  uint8_t* page = p.swa + blk * p.swa_stride;
  uint8_t* data = page + pib * V4_DATA;
  uint8_t* scl = page + long(p.swa_block) * V4_DATA + pib * V4_SCALES;
  if (d0 < NOPE) {
    const float inv = pow2(-ex);
    unsigned wd[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float c[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) c[i] = fminf(fmaxf(__fmul_rn(e[4 * q + i], inv), -448.0f), 448.0f);
      wd[q] = e4m3x2(c[0], c[1]) | (e4m3x2(c[2], c[3]) << 16);
    }
    *reinterpret_cast<uint4*>(data + d0) = make_uint4(wd[0], wd[1], wd[2], wd[3]);
    if ((lane & 3) == 0) scl[lane >> 2] = uint8_t(min(max(ex + 127, 0), 255));
    if (lane == 0) scl[NOPE / 64] = 0;
  } else {
    uint4* rd = reinterpret_cast<uint4*>(data + NOPE + (d0 - NOPE) * 2);
    rd[0] = pack8_bf16(reinterpret_cast<const float(&)[8]>(e[0]));
    rd[1] = pack8_bf16(reinterpret_cast<const float(&)[8]>(e[8]));
  }
}

// ---------------------------------------------------------------- G1 tile finish / shared qr tile
// The last of the 8 k-part warps of `tile`: fixed-order sum of the partials, bf16 qkv + square sums, signal.
__device__ __forceinline__ void g1_finish_tile(const Params& p, int tile, int lane) {
  const int g = lane >> 2, t = lane & 3, m = p.m;
  // Acquire side of the threadfence reduction: lane 0 observed all 8 arrivals (atomic, then fence; light sync:
  // its atom.acq_rel); bar.warp.sync then orders that observation before every lane's partial loads (a shuffle
  // alone carries no memory ordering in the PTX model).
  if (!(p.fast & FAST_LIGHT_SYNC)) __threadfence();
  __syncwarp();
  float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
#pragma unroll
  for (int k = 0; k < KSPLIT1; ++k) {
    const float* q = p.qkp + (long(tile) * KSPLIT1 + k) * 16 * MAXM;
    const float2 x0 = __ldcg(reinterpret_cast<const float2*>(q + g * MAXM + 2 * t));
    const float2 x1 = __ldcg(reinterpret_cast<const float2*>(q + (g + 8) * MAXM + 2 * t));
    v0 += x0.x, v1 += x0.y, v2 += x1.x, v3 += x1.y;
  }
  // The stock fused_wqa_wkv GEMM outputs bf16 (qr_kv), which the q/kv RMSNorm then reads.
  const __nv_bfloat16 b0 = __float2bfloat16(v0), b1 = __float2bfloat16(v1), b2 = __float2bfloat16(v2), b3 = __float2bfloat16(v3);
  const int c = tile * 16 + g;
  if (2 * t < m) p.qkv[(2 * t) * QKV + c] = b0, p.qkv[(2 * t) * QKV + c + 8] = b2;
  if (2 * t + 1 < m) p.qkv[(2 * t + 1) * QKV + c] = b1, p.qkv[(2 * t + 1) * QKV + c + 8] = b3;
  float s0 = bf2f(b0) * bf2f(b0) + bf2f(b2) * bf2f(b2), s1 = bf2f(b1) * bf2f(b1) + bf2f(b3) * bf2f(b3);
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    s0 += __shfl_xor_sync(FULL, s0, o);
    s1 += __shfl_xor_sync(FULL, s1, o);
  }
  if (g == 0) {
    if (2 * t < m) p.qsq[tile * MAXM + 2 * t] = s0;
    if (2 * t + 1 < m) p.qsq[tile * MAXM + 2 * t + 1] = s1;
  }
  __syncwarp();
  if (lane == 0) {
    p.tilecnt[tile] = 0;  // ready for the next launch (stream order makes it visible)
    if (!(p.fast & FAST_LIGHT_SYNC)) __threadfence();
    red_release(p.pc + (tile < Q_TILES ? C_P3Q : C_P3K), 1u);
    PROF_MAX(9);  // G1: latest tile finish of this CTA
  }
}

// wgmma, G1 CTAs b < QR_CTAS: once all q tiles are published (C_P3Q), quantize qr = q_norm(qkv[:, :1280])
// (reference act_quant) into the shared B tile p.qrb / p.qrs and write the `qr` output; -> C_QR.
// Warp w of CTA b owns jobs (k32 block, token) j = 8 * (8b + w) + quad; lane t4 of the quad owns physical k
// 8*t4 .. 8*t4+7 of the block. Job loads are issued before the q_norm statistics barrier (one L2 round).
__device__ __forceinline__ void qr_build(const Params& p, uint8_t* sm, int b, int tid, int warp, int lane) {
  float* rr = reinterpret_cast<float*>(sm + SM_RS);
  const int m = p.m, njobs = KB2 * m, t4 = lane & 3;
  const int job = (b * WARPS + warp) * 8 + (lane >> 2);
  const bool ok = job < njobs;
  const int jj = ok ? job : njobs - 1, kb = jj / m, tok = jj - kb * m;
  cta_wait(p.pc + C_P3Q, Q_TILES, p.fast & FAST_LIGHT_SYNC);
  PROF_T0(7);  // G1 (qr CTAs): all q tiles published
  const uint4 raw = __ldcg(reinterpret_cast<const uint4*>(p.qkv + long(tok) * QKV + kb * 32 + 8 * t4));
  const uint4 wraw = __ldg(reinterpret_cast<const uint4*>(p.q_norm + kb * 32 + 8 * t4));
  if (warp < m) {  // q_norm statistics of token `warp` (same order as role_g2)
    float sq = 0.0f;
    for (int i = lane; i < Q_TILES; i += 32) sq += __ldcg(p.qsq + i * MAXM + warp);
    sq = warp_sum(sq);
    if (lane == 0) rr[warp] = rsqrtf(__fdiv_rn(sq, float(QR)) + p.qkv_eps);
  }
  if (b == 0) {  // padding token rows of the shared tile
    for (int i = tid; i < MAXM * KB2 * 2; i += THREADS) {
      const int pt = i & (MAXM - 1);
      if (pt >= m) *reinterpret_cast<uint4*>(p.qrb + (i >> 4) * WG_KB_PITCH + ((i >> 3) & 1) * 128 + pt * 16) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int i = tid; i < MAXM * KB2; i += THREADS)
      if ((i & (MAXM - 1)) >= m) p.qrs[i] = 0.0f;
  }
  __syncthreads();
  float x[8], wf[8];
  unpack8(raw, x);
  unpack8(wraw, wf);
  const uint4 pk = rmsnorm8_bf16(x, rr[tok], make_float4(wf[0], wf[1], wf[2], wf[3]), make_float4(wf[4], wf[5], wf[6], wf[7]));
  if (ok) *reinterpret_cast<uint4*>(p.qr + long(tok) * QR + kb * 32 + 8 * t4) = pk;
  float am = fabsf(x[0]);
#pragma unroll
  for (int i = 1; i < 8; ++i) am = fmaxf(am, fabsf(x[i]));
  am = fmaxf(am, __shfl_xor_sync(FULL, am, 1));
  am = fmaxf(am, __shfl_xor_sync(FULL, am, 2));
  const int e = quant_exp(am, AMAX_FLOOR);
  const float inv = pow2(-e);
  if (ok) {
    uint8_t* dst = p.qrb + kb * WG_KB_PITCH + tok * 16 + 4 * t4;
    *reinterpret_cast<unsigned*>(dst) = e4m3x2(x[0] * inv, x[1] * inv) | (e4m3x2(x[2] * inv, x[3] * inv) << 16);
    *reinterpret_cast<unsigned*>(dst + 128) = e4m3x2(x[4] * inv, x[5] * inv) | (e4m3x2(x[6] * inv, x[7] * inv) << 16);
    if (t4 == 0) p.qrs[kb * MAXM + tok] = pow2(e);
  }
  PROF_T0(11);  // G1 (qr CTAs): thread 0's qr jobs done
  cta_signal(p.pc + C_QR, p.fast & FAST_LIGHT_SYNC);
  PROF_T0(8);  // G1 (qr CTAs): shared qr tile published
}

// wgmma + FAST_QR_LOCAL: every G2 CTA builds the e4m3 qr B tile (bt / sx in its shared memory) straight from the
// published q tiles, like qr_build but without the extra C_QR hop and global tile copy (128x redundant SIMT work
// instead of one more cross-CTA round trip); CTA 0 also writes the `qr` output. Same values as qr_build.
constexpr int QRL_PASSES = (KB2 * MAXM + THREADS / 4 - 1) / (THREADS / 4);  // (k32 block, token) jobs per lane quad
__device__ __forceinline__ void qr_local(const Params& p, uint8_t* sm, uint8_t* bt, float* sx, int c, int tid, int warp, int lane) {
  float* rr = reinterpret_cast<float*>(sm + SM_RS);
  const int m = p.m, njobs = KB2 * m, t4 = lane & 3;
  cta_wait(p.pc + C_P3Q, Q_TILES, p.fast & FAST_LIGHT_SYNC);
  PROF_T0(5);  // G2: q tiles published
  uint4 raw[QRL_PASSES];
#pragma unroll
  for (int ps = 0; ps < QRL_PASSES; ++ps) {
    const int jj = min(ps * (THREADS / 4) + (tid >> 2), njobs - 1), kb = jj / m, tok = jj - kb * m;
    raw[ps] = __ldcg(reinterpret_cast<const uint4*>(p.qkv + long(tok) * QKV + kb * 32 + 8 * t4));
  }
  if (warp < m) {  // q_norm statistics of token `warp` (same order as qr_build / role_g2)
    float sq = 0.0f;
    for (int i = lane; i < Q_TILES; i += 32) sq += __ldcg(p.qsq + i * MAXM + warp);
    sq = warp_sum(sq);
    if (lane == 0) rr[warp] = rsqrtf(__fdiv_rn(sq, float(QR)) + p.qkv_eps);
  }
  for (int i = tid; i < MAXM * KB2 * 2; i += THREADS) {  // padding token rows
    const int pt = i & (MAXM - 1);
    if (pt >= m) *reinterpret_cast<uint4*>(bt + (i >> 4) * WG_KB_PITCH + ((i >> 3) & 1) * 128 + pt * 16) = make_uint4(0u, 0u, 0u, 0u);
  }
  for (int i = tid; i < MAXM * KB2; i += THREADS)
    if ((i & (MAXM - 1)) >= m) sx[i] = 0.0f;
  __syncthreads();
#pragma unroll
  for (int ps = 0; ps < QRL_PASSES; ++ps) {
    const int job = ps * (THREADS / 4) + (tid >> 2);
    if (ps * (THREADS / 4) >= njobs) break;  // CTA-uniform
    const bool ok = job < njobs;
    const int jj = ok ? job : njobs - 1, kb = jj / m, tok = jj - kb * m;
    float x[8], wf[8];
    unpack8(raw[ps], x);
    unpack8(__ldg(reinterpret_cast<const uint4*>(p.q_norm + kb * 32 + 8 * t4)), wf);  // 2.5 KB, L1-resident
    const uint4 pk = rmsnorm8_bf16(x, rr[tok], make_float4(wf[0], wf[1], wf[2], wf[3]), make_float4(wf[4], wf[5], wf[6], wf[7]));
    if (ok && c == 0) *reinterpret_cast<uint4*>(p.qr + long(tok) * QR + kb * 32 + 8 * t4) = pk;
    float am = fabsf(x[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) am = fmaxf(am, fabsf(x[i]));
    am = fmaxf(am, __shfl_xor_sync(FULL, am, 1));
    am = fmaxf(am, __shfl_xor_sync(FULL, am, 2));
    const int e = quant_exp(am, AMAX_FLOOR);
    const float inv = pow2(-e);
    if (ok) {
      uint8_t* dst = bt + kb * WG_KB_PITCH + tok * 16 + 4 * t4;
      *reinterpret_cast<unsigned*>(dst) = e4m3x2(x[0] * inv, x[1] * inv) | (e4m3x2(x[2] * inv, x[3] * inv) << 16);
      *reinterpret_cast<unsigned*>(dst + 128) = e4m3x2(x[4] * inv, x[5] * inv) | (e4m3x2(x[6] * inv, x[7] * inv) << 16);
      if (t4 == 0) sx[kb * MAXM + tok] = pow2(e);
    }
  }
  fence_proxy_async_smem();  // generic st.shared -> wgmma operand reads
  __syncthreads();
}

// ---------------------------------------------------------------- FAST_CLUSTER G1 (clusters of the 8 k-part CTAs)
// The grid runs in clusters of 8 (the 8 k-part CTAs of a G1 tile group are one cluster). After the MMA the tile
// partials are reduced through DSMEM (CTA r finishes tile grp*8 + r, k-parts summed 0..7 like g1_finish_tile): the
// tile counter atomic + partial reload round trips become a cluster barrier.
// FAST_OWN_COLLAPSE (with FAST_CLUSTER): each G1 CTA also computes the collapse of its own 640 columns (same arithmetic
// as p1_collapse, no wait for P1a) and gets the attn_norm sum of squares from its 7 peers through DSMEM in the order
// of the global path (bit-identical rs). Measured slower (the 14 groups redo the collapse: ~6 us at m = 6).
__device__ __forceinline__ unsigned cl_rank() {
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cl_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_wait_acquire() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ unsigned cl_map(const void* local, unsigned rank) {
  unsigned r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_u32(local)), "r"(rank));
  return r;
}
__device__ __forceinline__ float cl_ld(unsigned addr) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ float2 cl_ld2(unsigned addr) {
  float2 v;
  asm volatile("ld.shared::cluster.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(addr) : "memory");
  return v;
}
constexpr int G1_SL = KPART1 / 32;  // 20 slices (32 columns) per k-part
constexpr int SM_CF = SM_W + KPART1 * 2;  // [MAXM][24] f32 post | pre | comb per token (after the attn_norm slice)
static_assert(SM_CF + MAXM * 24 * 4 <= SM_RS && MAXM * G1_SL * 4 <= MAXM * SLICES * 4);

// Collapse of columns [col0, col0 + 640) for all m tokens -> bf16 rows in act (pitch layout) and the per-slice
// square sums ssum[tok][j] (j < 20; warp butterfly over the slice's 32 columns, exactly as p1_collapse).
__device__ __forceinline__ void g1_collapse(const Params& p, uint8_t* sm, int col0, int tid, int warp, int lane) {
  constexpr int PITCH = act_pitch(KPART1), V8 = KPART1 / 8;
  uint8_t* act = sm + SM_ACT;
  float* cf = reinterpret_cast<float*>(sm + SM_CF);
  float* ssum = reinterpret_cast<float*>(sm + SM_AUX);
  const int m = p.m;
  if (tid < m * 24) {
    const int tok = tid / 24, i = tid - tok * 24;
    cf[tid] = i < HC ? __ldg(p.post + tok * HC + i) : i < 2 * HC ? __ldg(p.pre + tok * HC + i - HC) : __ldg(p.comb + tok * HC * HC + i - 2 * HC);
  }
  __syncthreads();
  // (raw bf16 words kept packed and widened per column: the G1 weight fragment (80 registers) is live here)
  auto bf_at = [](const uint4& v, int i) {
    const unsigned w = i < 2 ? v.x : i < 4 ? v.y : i < 6 ? v.z : v.w;
    return __uint_as_float((i & 1) ? (w & 0xffff0000u) : (w << 16));
  };
  for (int v = tid; v < m * V8; v += THREADS) {
    const int tok = v / V8, c8 = v - tok * V8, col = col0 + c8 * 8;
    const uint4 yv = __ldg(reinterpret_cast<const uint4*>(p.x + long(tok) * HID + col));
    uint4 rv[HC];
#pragma unroll
    for (int k = 0; k < HC; ++k) rv[k] = __ldg(reinterpret_cast<const uint4*>(p.resid + (long(tok) * HC + k) * HID + col));
    const float* c = cf + tok * 24;
    unsigned pk[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float y = bf_at(yv, i);
      float rb[HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {  // p1_collapse: t = post[j]*y; t = fma(comb[k][j], r[k], t), k = 0..3; bf16
        float t = __fmul_rn(c[j], y);
#pragma unroll
        for (int k = 0; k < HC; ++k) t = __fmaf_rn(c[2 * HC + k * HC + j], bf_at(rv[k], i), t);
        rb[j] = bf2f(__float2bfloat16(t));
      }
      float xc = __fmul_rn(c[HC], rb[0]);
#pragma unroll
      for (int j = 1; j < HC; ++j) xc = __fmaf_rn(c[HC + j], rb[j], xc);
      const unsigned short h = __bfloat16_as_ushort(__float2bfloat16(xc));
      pk[i >> 1] = (i & 1) ? (pk[i >> 1] | (unsigned(h) << 16)) : unsigned(h);
    }
    *reinterpret_cast<uint4*>(act + tok * PITCH + c8 * 16) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  }
  __syncthreads();
  // slice square sums: warp w takes (slice, token) jobs; lane = column within the slice (p1_collapse's butterfly)
  for (int jt = warp; jt < G1_SL * m; jt += WARPS) {
    const int tok = jt / G1_SL, j = jt - tok * G1_SL;
    const float xf = bf2f(*reinterpret_cast<const __nv_bfloat16*>(act + tok * PITCH + (j * 32 + lane) * 2));
    const float xs = warp_sum(xf * xf);
    if (lane == 0) ssum[tok * G1_SL + j] = xs;
  }
}

// attn_norm statistics from the cluster's 160 slice sums, in the order of the global path (role_g1): lane L adds
// slices L, L + 32, ... (slice s lives in rank s / 20), then the butterfly. Warp tok < m -> rs[tok].
__device__ __forceinline__ void g1_cluster_rs(const Params& p, uint8_t* sm, int warp, int lane) {
  const float* ssum = reinterpret_cast<const float*>(sm + SM_AUX);
  float* rs = reinterpret_cast<float*>(sm + SM_RS);
  if (warp < p.m) {
    float s = 0.0f;
    for (int i = lane; i < SLICES; i += 32) s += cl_ld(cl_map(ssum + warp * G1_SL + i % G1_SL, unsigned(i / G1_SL)));
    s = warp_sum(s);
    if (lane == 0) rs[warp] = rsqrtf(__fdiv_rn(s, float(HID)) + p.norm_eps);
  }
}

// Warp 0 of cluster rank r: tile = grp*8 + r from the 8 k-part partials red[r] of ranks 0..7 (k-part order, as
// g1_finish_tile), bf16 qkv + square sums, signal.
// rs (FAST_V3_ADEFER): per-token attn_norm rsqrt applied to the k-sum before the bf16 rounding (nullptr: none).
__device__ __forceinline__ void g1_cluster_finish(const Params& p, const float* red, int tile, unsigned r, int lane, const float* rs = nullptr) {
  const int g = lane >> 2, t = lane & 3, m = p.m;
  const float* mine = red + r * 16 * MAXM;
  float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  float2 x0[KSPLIT1], x1[KSPLIT1];
#pragma unroll
  for (int k = 0; k < KSPLIT1; ++k) {
    x0[k] = cl_ld2(cl_map(mine + g * MAXM + 2 * t, unsigned(k)));
    x1[k] = cl_ld2(cl_map(mine + (g + 8) * MAXM + 2 * t, unsigned(k)));
  }
#pragma unroll
  for (int k = 0; k < KSPLIT1; ++k) v0 += x0[k].x, v1 += x0[k].y, v2 += x1[k].x, v3 += x1[k].y;
  if (rs != nullptr) {
    const float r0 = rs[2 * t], r1 = rs[2 * t + 1];
    v0 = __fmul_rn(v0, r0), v1 = __fmul_rn(v1, r1), v2 = __fmul_rn(v2, r0), v3 = __fmul_rn(v3, r1);
  }
  const __nv_bfloat16 b0 = __float2bfloat16(v0), b1 = __float2bfloat16(v1), b2 = __float2bfloat16(v2), b3 = __float2bfloat16(v3);
  const int c = tile * 16 + g;
  if (2 * t < m) p.qkv[(2 * t) * QKV + c] = b0, p.qkv[(2 * t) * QKV + c + 8] = b2;
  if (2 * t + 1 < m) p.qkv[(2 * t + 1) * QKV + c] = b1, p.qkv[(2 * t + 1) * QKV + c + 8] = b3;
  float s0 = bf2f(b0) * bf2f(b0) + bf2f(b2) * bf2f(b2), s1 = bf2f(b1) * bf2f(b1) + bf2f(b3) * bf2f(b3);
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    s0 += __shfl_xor_sync(FULL, s0, o);
    s1 += __shfl_xor_sync(FULL, s1, o);
  }
  if (g == 0) {
    if (2 * t < m) p.qsq[tile * MAXM + 2 * t] = s0;
    if (2 * t + 1 < m) p.qsq[tile * MAXM + 2 * t + 1] = s1;
  }
  __syncwarp();
  if (lane == 0) {
    if (!(p.fast & FAST_LIGHT_SYNC)) __threadfence();
    red_release(p.pc + (tile < Q_TILES ? C_P3Q : C_P3K), 1u);
    PROF_MAX(9);  // G1: tile finish of this CTA
  }
}

// wgmma + FAST_CLUSTER + FAST_QR_DEFER, even cluster rank r of a q group: finishes tiles grp*8 + r and + r + 1 (the
// 32 columns of qr k32 block kb) like g1_cluster_finish, then quantizes u = bf16(qkv * q_norm_w) of the block (act_quant
// per token, the q_norm rsqrt deferred to the wq_b epilogue: rsqrt(mean) is a per-token scalar) straight into the
// shared B tile qrb / qrs -- no qr_build hop. Signals C_P3Q with 2 (tiles).
__device__ __forceinline__ void g1_cluster_finish_pair(const Params& p, const float* red, int tile, unsigned r, int lane, const float* rs = nullptr) {
  const int g = lane >> 2, t = lane & 3, m = p.m, kb = tile >> 1;
  float v[2][4];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float* mine = red + (r + h) * 16 * MAXM;
    float2 x0[KSPLIT1], x1[KSPLIT1];
#pragma unroll
    for (int k = 0; k < KSPLIT1; ++k) {
      x0[k] = cl_ld2(cl_map(mine + g * MAXM + 2 * t, unsigned(k)));
      x1[k] = cl_ld2(cl_map(mine + (g + 8) * MAXM + 2 * t, unsigned(k)));
    }
    v[h][0] = v[h][1] = v[h][2] = v[h][3] = 0.0f;
#pragma unroll
    for (int k = 0; k < KSPLIT1; ++k) v[h][0] += x0[k].x, v[h][1] += x0[k].y, v[h][2] += x1[k].x, v[h][3] += x1[k].y;
    if (rs != nullptr) {
      const float r0 = rs[2 * t], r1 = rs[2 * t + 1];
      v[h][0] = __fmul_rn(v[h][0], r0), v[h][1] = __fmul_rn(v[h][1], r1), v[h][2] = __fmul_rn(v[h][2], r0), v[h][3] = __fmul_rn(v[h][3], r1);
    }
  }
  // bf16 qkv (stock GEMM output), its square sums per tile, and u = bf16(qkv * w) per (column, token)
  // v[h][0] = (col 16h + g, tok 2t), [1] = (16h + g, 2t + 1), [2] = (16h + g + 8, 2t), [3] = (16h + g + 8, 2t + 1)
  float u[2][4];
  float am0 = 0.0f, am1 = 0.0f;  // tokens 2t, 2t + 1
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int c = (tile + h) * 16 + g;
    const __nv_bfloat16 b0 = __float2bfloat16(v[h][0]), b1 = __float2bfloat16(v[h][1]), b2 = __float2bfloat16(v[h][2]), b3 = __float2bfloat16(v[h][3]);
    if (2 * t < m) p.qkv[(2 * t) * QKV + c] = b0, p.qkv[(2 * t) * QKV + c + 8] = b2;
    if (2 * t + 1 < m) p.qkv[(2 * t + 1) * QKV + c] = b1, p.qkv[(2 * t + 1) * QKV + c + 8] = b3;
    float s0 = bf2f(b0) * bf2f(b0) + bf2f(b2) * bf2f(b2), s1 = bf2f(b1) * bf2f(b1) + bf2f(b3) * bf2f(b3);
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      s0 += __shfl_xor_sync(FULL, s0, o);
      s1 += __shfl_xor_sync(FULL, s1, o);
    }
    if (g == 0) {
      if (2 * t < m) p.qsq[(tile + h) * MAXM + 2 * t] = s0;
      if (2 * t + 1 < m) p.qsq[(tile + h) * MAXM + 2 * t + 1] = s1;
    }
    const float w0 = bf2f(__ldg(p.q_norm + c)), w1 = bf2f(__ldg(p.q_norm + c + 8));
    u[h][0] = bf16_round(__fmul_rn(bf2f(b0), w0));
    u[h][1] = bf16_round(__fmul_rn(bf2f(b1), w0));
    u[h][2] = bf16_round(__fmul_rn(bf2f(b2), w1));
    u[h][3] = bf16_round(__fmul_rn(bf2f(b3), w1));
    am0 = fmaxf(am0, fmaxf(fabsf(u[h][0]), fabsf(u[h][2])));
    am1 = fmaxf(am1, fmaxf(fabsf(u[h][1]), fabsf(u[h][3])));
  }
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {  // amax over the block's 32 columns (lanes with the same t)
    am0 = fmaxf(am0, __shfl_xor_sync(FULL, am0, o));
    am1 = fmaxf(am1, __shfl_xor_sync(FULL, am1, o));
  }
  const int e0 = quant_exp(am0, AMAX_FLOOR), e1 = quant_exp(am1, AMAX_FLOOR);
  const float i0 = pow2(-e0), i1 = pow2(-e1);
  // B tile layout (stage_e4m3): column cb = 8 t4 + j of the block -> byte 4 t4 + j (j < 4) or 128 + 4 t4 + j - 4
  auto put = [&](int cb, int tok, float x, float inv) {
    const int t4 = cb >> 3, j = cb & 7;
    const unsigned q = tok < m ? (e4m3x2(x * inv, 0.0f) & 0xffu) : 0u;
    p.qrb[kb * WG_KB_PITCH + tok * 16 + (j < 4 ? 4 * t4 + j : 128 + 4 * t4 + j - 4)] = uint8_t(q);
  };
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    put(16 * h + g, 2 * t, u[h][0], i0);
    put(16 * h + g, 2 * t + 1, u[h][1], i1);
    put(16 * h + g + 8, 2 * t, u[h][2], i0);
    put(16 * h + g + 8, 2 * t + 1, u[h][3], i1);
  }
  if (g == 0) {
    p.qrs[kb * MAXM + 2 * t] = 2 * t < m ? pow2(e0) : 0.0f;
    p.qrs[kb * MAXM + 2 * t + 1] = 2 * t + 1 < m ? pow2(e1) : 0.0f;
  }
  __syncwarp();
  if (lane == 0) {
    if (!(p.fast & FAST_LIGHT_SYNC)) __threadfence();
    red_release(p.pc + C_P3Q, 2u);
    PROF_MAX(9);
  }
}

// ---------------------------------------------------------------- roles
template <int MMA>
__device__ __forceinline__ void role_g1(const Params& p, uint8_t* sm, int b, int tid, int warp, int lane) {
  const int kp = b % KSPLIT1, grp = b / KSPLIT1, tile = grp * WARPS + warp, m = p.m, col0 = kp * KPART1;
  Frag f;
  const bool clustered = p.fast & FAST_CLUSTER;
  const bool own = clustered && (p.fast & FAST_OWN_COLLAPSE);
  if (own) {
    // the own-slice collapse needs registers: only the first half of the fragment now, the second half is pulled into
    // L2 (one bulk prefetch per warp: its 10 consecutive 512-byte chunks) and loaded after the collapse
    constexpr int H1 = KW_MAX / 2;
    const uint8_t* base = p.wqa + ((long(tile) * KB1 + kp * KW_MAX) * 32 + lane) * 16;
#pragma unroll
    for (int i = 0; i < H1; ++i) f.w[i] = ld_stream16(base + i * CHUNK);
    f.s = lane * 4 < KW_MAX ? ld_stream4(p.wqa_s + long(tile) * KB1 + kp * KW_MAX + lane * 4) : 0u;
    if (lane == 0) prefetch_l2_bulk(p.wqa + (long(tile) * KB1 + kp * KW_MAX + H1) * CHUNK, (KW_MAX - H1) * CHUNK);
  } else {
    load_frag(f, p.wqa, p.wqa_s, tile, KB1, KB1, kp * KW_MAX, KW_MAX, lane);  // immutable: before the PDL wait
  }
  pdl_wait();
  PROF_T0(1);
  uint8_t* act = sm + SM_ACT;
  uint8_t* xe = sm + SM_XE;
  float* xsq = reinterpret_cast<float*>(sm + SM_AUX);
  __nv_bfloat16* w = reinterpret_cast<__nv_bfloat16*>(sm + SM_W);
  float* rs = reinterpret_cast<float*>(sm + SM_RS);
  constexpr int PITCH = act_pitch(KPART1), V8 = KPART1 / 8;
  for (int i = tid; i < V8; i += THREADS) cp_async16(w + i * 8, p.attn_norm + col0 + i * 8);
  cp_async_commit();
  if (own) {
    g1_collapse(p, sm, col0, tid, warp, lane);  // own slice: no wait for P1a
    {
      constexpr int H1 = KW_MAX / 2;
      const uint8_t* base = p.wqa + ((long(tile) * KB1 + kp * KW_MAX) * 32 + lane) * 16;
#pragma unroll
      for (int i = H1; i < KW_MAX; ++i) f.w[i] = ld_stream16(base + i * CHUNK);
    }
    cp_async_wait_all();
    PROF_T0(2);  // G1: collapse slice computed
    cl_arrive_release();  // slice sums visible to the cluster
    cl_wait_acquire();
    g1_cluster_rs(p, sm, warp, lane);
    PROF_T0(3);  // G1: rs known
  } else {
    // The collapse uses the incoming pre_mix, so (early mode) only P1a has to be complete.
    cta_wait(p.pc + ((p.fast & FAST_EARLY) ? C_P1A : C_P1B), unsigned(G2_CTAS), p.fast & FAST_LIGHT_SYNC);
    PROF_T0(2);  // G1: collapse published
    for (int v = tid; v < m * V8; v += THREADS) {
      const int tok = v / V8, c8 = v - tok * V8;
      cp_async16(act + tok * PITCH + c8 * 16, p.xcb + long(tok) * HID + col0 + c8 * 8);
    }
    for (int v = tid; v < m * SLICES / 4; v += THREADS) cp_async16(xsq + v * 4, p.xsq + v * 4);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    PROF_T0(3);  // G1: collapse slice in smem
    if (warp < m) {
      float s = 0.0f;
      for (int i = lane; i < SLICES; i += 32) s += xsq[warp * SLICES + i];
      s = warp_sum(s);
      // attn_norm over the bf16 collapse (tilelang_kernels.py:472-500): rsqrt(sum / D + eps).
      if (lane == 0) rs[warp] = rsqrtf(__fdiv_rn(s, float(HID)) + p.norm_eps);
    }
  }
  if constexpr (MMA == MMA_WGMMA) widen_bf16(reinterpret_cast<float*>(sm + SM_WF), w, V8, tid);
  __syncthreads();
  // a = bf16(collapse * rsqrt * attn_norm) for the CTA's 640 columns (and the `a` output from group 0)
  const bool write_a = grp == 0;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if constexpr (MMA == MMA_HMMA) {
    convert_act(act, PITCH, xe, m, KPART1, tid, [&](int tok, int c8, float (&x)[8]) {
      const float r = rs[tok];
      float wf[8];
      unpack8(*reinterpret_cast<const uint4*>(w + c8 * 8), wf);
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = bf16_round(__fmul_rn(__fmul_rn(x[i], r), wf[i]));
      if (write_a) *reinterpret_cast<uint4*>(p.a + long(tok) * HID + col0 + c8 * 8) = pack8_bf16(x);
    });
    __syncthreads();
    PROF_T0(4);  // G1: activations staged
    mma_frag(f, act, PITCH, xe, 0, KW_MAX, lane, acc);
  } else {
    uint8_t* bt = sm + SM_BT;
    float* sx = reinterpret_cast<float*>(sm + SM_SX);
    float* fac = reinterpret_cast<float*>(sm + SM_FAC) + warp * KW_MAX * MAXM;
    const float4* wf = reinterpret_cast<const float4*>(sm + SM_WF);
    stage_e4m3<KW_MAX>(bt, sx, m, tid, THREADS, [&](int tok, int c8, float (&x)[8], bool ok) {
      unpack8(*reinterpret_cast<const uint4*>(act + tok * PITCH + c8 * 16), x);
      const uint4 pk = rmsnorm8_bf16(x, rs[tok], wf[2 * c8], wf[2 * c8 + 1]);
      if (write_a && ok) *reinterpret_cast<uint4*>(p.a + long(tok) * HID + col0 + c8 * 8) = pk;
    });
    fence_proxy_async_smem();
    __syncthreads();
    PROF_T0(4);  // G1: activations staged
    wg_factors<KW_MAX>(fac, sx, f.s, KW_MAX, lane);
    __syncwarp();
    wg_mma<KW_MAX>(f, bt, fac, lane, acc);  // warpgroup = tiles grp*8 + {0..3 | 4..7}, same k-part
  }
  PROF_T0(5);  // G1: warp 0 MMA done (includes waiting for its weight registers)

  if (clustered) {
    // k reduction inside the cluster: every warp's [16 x MAXM] partial in red[warp], CTA rank r sums red[r] of all
    // 8 ranks (their k-parts) in k-part order. The second barrier keeps every red alive until all peers read it.
    float* red = reinterpret_cast<float*>(sm + SM_RED);
    store_frag_tile(red + warp * 16 * MAXM, acc, lane);
    PROF_T0(10);
    cl_arrive_release();
    cl_wait_acquire();
    PROF_T0(6);  // G1: cluster partials visible
    const bool defer = MMA == MMA_WGMMA && (p.fast & FAST_QR_DEFER) && grp * WARPS < Q_TILES;
    if (warp == 0) {
      if (!defer)
        g1_cluster_finish(p, red, grp * WARPS + kp, unsigned(kp), lane);
      else if ((kp & 1) == 0)
        g1_cluster_finish_pair(p, red, grp * WARPS + kp, unsigned(kp), lane);
    }
    cl_arrive_release();
    cl_wait_acquire();
    if constexpr (MMA == MMA_WGMMA) {
      if (b < QR_CTAS && !(p.fast & (FAST_QR_LOCAL | FAST_QR_DEFER))) qr_build(p, sm, b, tid, warp, lane);
    }
    return;
  }
  // Deterministic cross-CTA k reduction: every k-part publishes its tile partial; the last arriving warp
  // (threadfence-reduction pattern) sums the 8 partials in k-part order.
  float* part = p.qkp + (long(tile) * KSPLIT1 + kp) * 16 * MAXM;
  const int g = lane >> 2, t = lane & 3;
  __stcg(reinterpret_cast<float2*>(part + g * MAXM + 2 * t), make_float2(acc[0], acc[1]));
  __stcg(reinterpret_cast<float2*>(part + (g + 8) * MAXM + 2 * t), make_float2(acc[2], acc[3]));
  __syncwarp();
  PROF_T0(10);  // G1: warp 0 partial stores issued
  unsigned arrived = 0;
  if (lane == 0) {
    if (p.fast & FAST_LIGHT_SYNC) {
      arrived = atom_add_acq_rel(p.tilecnt + tile, 1u);  // release our partial, acquire the others' (if last)
    } else {
      __threadfence();
      arrived = atomicAdd(p.tilecnt + tile, 1u);
    }
  }
  arrived = __shfl_sync(FULL, arrived, 0);
  PROF_T0(6);  // G1: warp 0 partial published
  if (arrived == KSPLIT1 - 1) g1_finish_tile(p, tile, lane);
  if constexpr (MMA == MMA_WGMMA) {
    if (b < QR_CTAS && !(p.fast & FAST_QR_LOCAL)) qr_build(p, sm, b, tid, warp, lane);
  }
}

template <int MMA>
__device__ __forceinline__ void role_g2(const Params& p, uint8_t* sm, int c, int tid, int warp, int lane) {
  if (p.fast & FAST_PREFETCH) prefetch_wqb(p, c, tid);  // weights only: before the PDL wait
  // hc_attn_fn of the CTA's P1 slices: staged in shared memory when it fits (m >= 2: the jobs of one slice share
  // it instead of each re-reading 12 KB from L2), else L2-prefetched for the direct loads.
  const P1Range pr = p1_range(p, c);
  float* fs = nullptr;
  if ((p.fast & FAST_FN_SMEM) && pr.ns > 0 && pr.ns <= fn_stage_max<MMA>() && p.m >= 2) {
    fs = reinterpret_cast<float*>(sm);
    p1_stage_fn(p, fs, pr, tid);
  } else {
    p1_prefetch_fn(p, c, tid);
  }
  pdl_wait();
  PROF_T0(1);
  p1_cta(p, fs, c, tid, warp, lane);
  // Warp -> (tile, k-half). HMMA: tl = warp >> 1, half = warp & 1. wgmma: a warpgroup must share the
  // k-part, so warpgroup = half and the 4 warps of it are the CTA's 4 tiles.
  const int tl = MMA == MMA_WGMMA ? (warp & 3) : (warp >> 1);
  const int half = MMA == MMA_WGMMA ? __shfl_sync(FULL, warp >> 2, 0) : (warp & 1);  // (uniform: see skinny.cu)
  const int tile = c * G2_TILES + tl, m = p.m;
  Frag f;
  load_frag(f, p.wqb, p.wqb_s, tile, KB2, KB2, half * KW_MAX, KW_MAX, lane);  // after P1 (L2-prefetched during P1b)
  float* red = reinterpret_cast<float*>(sm + SM_RED);
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if constexpr (MMA == MMA_HMMA) {
    uint8_t* act = sm + SM_ACT;
    uint8_t* xe = sm + SM_XE;
    float* qsq = reinterpret_cast<float*>(sm + SM_AUX);
    __nv_bfloat16* w = reinterpret_cast<__nv_bfloat16*>(sm + SM_W);
    float* rr = reinterpret_cast<float*>(sm + SM_RS);
    constexpr int PITCH = act_pitch(QR), V8 = QR / 8;
    for (int i = tid; i < V8; i += THREADS) cp_async16(w + i * 8, p.q_norm + i * 8);
    cp_async_commit();
    cta_wait(p.pc + C_P3Q, Q_TILES, p.fast & FAST_LIGHT_SYNC);
    PROF_T0(5);  // G2: q tiles published
    for (int v = tid; v < m * V8; v += THREADS) {
      const int tok = v / V8, c8 = v - tok * V8;
      cp_async16(act + tok * PITCH + c8 * 16, p.qkv + long(tok) * QKV + c8 * 8);
    }
    for (int v = tid; v < Q_TILES * MAXM / 4; v += THREADS) cp_async16(qsq + v * 4, p.qsq + v * 4);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    if (warp < m) {
      float s = 0.0f;
      for (int i = lane; i < Q_TILES; i += 32) s += qsq[i * MAXM + warp];
      s = warp_sum(s);
      if (lane == 0) rr[warp] = rsqrtf(__fdiv_rn(s, float(QR)) + p.qkv_eps);
    }
    __syncthreads();
    const bool write_qr = c == 0;
    convert_act(act, PITCH, xe, m, QR, tid, [&](int tok, int c8, float (&x)[8]) {
      const float r = rr[tok];
      float wf[8];
      unpack8(*reinterpret_cast<const uint4*>(w + c8 * 8), wf);
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = bf16_round(__fmul_rn(__fmul_rn(x[i], r), wf[i]));
      if (write_qr) *reinterpret_cast<uint4*>(p.qr + long(tok) * QR + c8 * 8) = pack8_bf16(x);
    });
    __syncthreads();
    PROF_T0(6);  // G2: qr staged
    mma_frag(f, act, PITCH, xe, half * KW_MAX, KW_MAX, lane, acc);
  } else {
    // the e4m3 qr tile was built once by the QR_CTAS producers (qr_build): copy it (12 KB, one L2 round)
    uint8_t* bt = sm + SM_BT;
    float* sx = reinterpret_cast<float*>(sm + SM_SX);
    float* fac = reinterpret_cast<float*>(sm + SM_FAC) + warp * KW_MAX * MAXM;
    if (p.fast & FAST_QR_DEFER) {
      // B tile built by the G1 pair finishers (u = qkv * q_norm_w, act_quant); the q_norm rsqrt goes to the epilogue
      cta_wait(p.pc + C_P3Q, Q_TILES, p.fast & FAST_LIGHT_SYNC);
      PROF_T0(5);  // G2: q blocks published
      for (int i = tid; i < KB2 * WG_KB_PITCH / 16; i += THREADS) cp_async16(bt + 16 * i, p.qrb + 16 * i);
      for (int i = tid; i < KB2 * MAXM / 4; i += THREADS) cp_async16(sx + 4 * i, p.qrs + 4 * i);
      cp_async_commit();
      float* rq = reinterpret_cast<float*>(sm + SM_RS);
      if (warp < m) {  // q_norm statistics of token `warp` (qr_build's order)
        float sq = 0.0f;
        for (int i = lane; i < Q_TILES; i += 32) sq += __ldcg(p.qsq + i * MAXM + warp);
        sq = warp_sum(sq);
        if (lane == 0) rq[warp] = rsqrtf(__fdiv_rn(sq, float(QR)) + p.qkv_eps);
      }
      cp_async_wait_all();
      fence_proxy_async_smem();
      __syncthreads();
    } else if (p.fast & FAST_QR_LOCAL) {
      qr_local(p, sm, bt, sx, c, tid, warp, lane);
    } else {
      cta_wait(p.pc + C_QR, unsigned(QR_CTAS), p.fast & FAST_LIGHT_SYNC);
      PROF_T0(5);  // G2: shared qr tile published
      for (int i = tid; i < KB2 * WG_KB_PITCH / 16; i += THREADS) cp_async16(bt + 16 * i, p.qrb + 16 * i);
      for (int i = tid; i < KB2 * MAXM / 4; i += THREADS) cp_async16(sx + 4 * i, p.qrs + 4 * i);
      cp_async_commit();
      cp_async_wait_all();
      fence_proxy_async_smem();  // cp.async (generic proxy) data -> wgmma operand reads (async proxy)
      __syncthreads();
    }
    PROF_T0(6);  // G2: qr tile in smem
    wg_factors<KW_MAX>(fac, sx + half * KW_MAX * MAXM, f.s, KW_MAX, lane);
    __syncwarp();
    wg_mma<KW_MAX>(f, bt + half * KW_MAX * WG_KB_PITCH, fac, lane, acc);
  }
  PROF_T0(7);  // G2: warp 0 MMA done (includes waiting for its wq_b registers)
  store_frag_tile(red + warp * 16 * MAXM, acc, lane);
  __syncthreads();
  PROF_T0(8);  // G2: k-halves in smem
  if (MMA == MMA_WGMMA && (p.fast & FAST_QR_DEFER) && tid < 10 * m) {
    // the `qr` output (stock q_norm: bf16((x * rsqrt) * w)): CTA c writes columns [10c, 10c + 10)
    static_assert(G2_CTAS * 10 == QR);
    const int tk = tid / 10, col = c * 10 + tid - tk * 10;
    const float x = bf2f(__ldcg(p.qkv + long(tk) * QKV + col)), w = bf2f(__ldg(p.q_norm + col));
    p.qr[long(tk) * QR + col] = __float2bfloat16(__fmul_rn(__fmul_rn(x, reinterpret_cast<const float*>(sm + SM_RS)[tk]), w));
  }
  // Epilogue: thread -> (token, adjacent row pair); 32 consecutive threads write 128 contiguous bytes.
  const int tok = tid >> 5, q32 = tid & 31, tl2 = q32 >> 3, row = 2 * (q32 & 7);
  if (tok >= m) return;
  // warp of (tile tl2, half 0) and of (tl2, half 1); halves summed in order 0 + 1
  const float* r0 = red + (MMA == MMA_WGMMA ? tl2 : 2 * tl2) * 16 * MAXM;
  const float* r1 = red + (MMA == MMA_WGMMA ? 4 + tl2 : 2 * tl2 + 1) * 16 * MAXM;
  float e0 = r0[row * MAXM + tok] + r1[row * MAXM + tok];
  float e1 = r0[(row + 1) * MAXM + tok] + r1[(row + 1) * MAXM + tok];
  if (MMA == MMA_WGMMA && (p.fast & FAST_QR_DEFER)) {  // deferred q_norm rsqrt (per token)
    const float rq = reinterpret_cast<const float*>(sm + SM_RS)[tok];
    e0 = __fmul_rn(e0, rq);
    e1 = __fmul_rn(e1, rq);
  }
  e0 = bf16_round(e0);  // stock wq_b output is bf16
  e1 = bf16_round(e1);
  const int orow = (c * G2_TILES + tl2) * 16 + row;
  if (p.q_pre != nullptr) *reinterpret_cast<unsigned*>(p.q_pre + long(tok) * WQB + orow) = pack_bf16(e0, e1);
  const int d = orow & (HDIM - 1);
  if (d >= NOPE) {
    const float* cs = p.cos_sin + p.positions[tok] * ROPE;
    const int i = (d - NOPE) >> 1;
    const float cv = __ldg(cs + i), sv = __ldg(cs + ROPE / 2 + i);
    const float xe2 = e0, xo = e1;
    e0 = __fmaf_rn(xe2, cv, -__fmul_rn(xo, sv));
    e1 = __fmaf_rn(xe2, sv, __fmul_rn(xo, cv));
  }
  *reinterpret_cast<unsigned*>(p.q + long(tok) * WQB + orow) = pack_bf16(e0, e1);
}

// H CTA 0: Sinkhorn (after C_P1B); H CTA 1: kv_norm + RoPE + V4 insert (after C_P3K). No P1 jobs (they all
// fit in the G2 warps, see P1_WORKERS).
__device__ __forceinline__ void role_h(const Params& p, uint8_t* sm, int h, int warp, int lane) {
  pdl_wait();
  PROF_T0(1);
  if (h == 0) {
    if (p.fast & FAST_SINKHORN) {
      float* cl = reinterpret_cast<float*>(sm + SM_ACT);  // [MAXM][16] comb logits
      if (warp < p.m) {
        warp_wait(p.pc + C_P1B, unsigned(G2_CTAS), lane, p.fast & FAST_LIGHT_SYNC);
        PROF_T0(2);  // H0: mixes published
        sinkhorn_mix(p, warp, lane, cl);
      }
      __syncthreads();
      if (warp == 0) sinkhorn_fast(p, cl, lane);
      PROF_T0(3);  // H0: Sinkhorn done
    } else if (warp < p.m) {
      warp_wait(p.pc + C_P1B, unsigned(G2_CTAS), lane, p.fast & FAST_LIGHT_SYNC);
      PROF_T0(2);  // H0: mixes published
      sinkhorn_job(p, warp, lane);
      PROF_T0(3);  // H0: Sinkhorn of token 0 done
    }
  } else if (h * WARPS + warp >= H_KV0 && h * WARPS + warp - H_KV0 < p.m) {
    warp_wait(p.pc + C_P3K, KV_TILES, lane, p.fast & FAST_LIGHT_SYNC);
    PROF_T0(4);  // H1: kv tiles published
    kv_job(p, h * WARPS + warp - H_KV0, lane);
    __syncwarp();
    if (lane == 0) aflag_signal(p.pcf, p.aflag);
    if (lane == 0) PROF_MAX(5);  // H1: KV insert done
  }
}

// ================================================================ v3 schedule (FAST_V3)
// Same grid as FAST_CLUSTER (240 CTAs, clusters of 8; the H jobs run on G1 CTAs 110 / 111 after their tiles), but:
//   * weights go to shared memory by one bulk copy per tile (cp.async.bulk + mbarrier, L2::evict_first) issued at
//     entry, so no thread holds weight registers and the latency-critical activation loads are not queued behind
//     the per-SM weight stream (LSU loads vs. the copy engine);
//   * G1 computes the collapse of its own 640 columns (x / residual are L2-hot in decode) instead of waiting for the
//     P1a hop, gets the exact attn_norm statistics from its 7 cluster peers through DSMEM, and reduces its k-parts
//     through DSMEM (no tile counter / partial reload hop);
//   * wgmma: q_norm deferred to the wq_b epilogue (FAST_QR_DEFER semantics, the G1 pair finishers write the e4m3 qr
//     B tile), so the qr_build hop is gone too. HMMA keeps stock rounding (bit-identical to fast=0).
//   * G1 never waits on G2 (G2 waits for the q tiles; the H jobs wait for P1b / kv tiles only after their own tile
//     work), so the grid cannot deadlock even when not every cluster is co-resident.
// MMA loops read the A fragments from shared memory (HMMA: rolled by 4, wgmma: one LDS.128 per block just before its
// wgmma), which keeps the hot code small (cold instruction fetch after the decoder layer kernel flushed the L2).
constexpr int V3_WBYTES = KW_MAX * CHUNK * WARPS;  // 81920: G1 = 8 tiles x 20 blocks, G2 = 4 tiles x 40 blocks
static_assert(V3_WBYTES == G2_TILES * KB2 * CHUNK && V3_WBYTES == WARPS * KW_MAX * CHUNK);
constexpr int V3_A = V3_WBYTES;  // role area
// G1 (offsets in the role area)
constexpr int V1_ACT = 0;                                  // MAXM x act_pitch(640): raw bf16 collapse rows (HMMA: f16 in place)
constexpr int V1_XE = V1_ACT + MAXM * act_pitch(KPART1);   // [20][MAXM] exponent bytes (HMMA)
constexpr int V1_SSUM = V1_XE + 256;                       // [MAXM][20] f32 slice square sums (read by the peers)
constexpr int V1_NW = V1_SSUM + MAXM * G1_SL * 4;          // attn_norm slice, bf16 [640]
constexpr int V1_CF = V1_NW + KPART1 * 2;                  // [MAXM][24] f32 post | pre | comb
constexpr int V1_RS = V1_CF + MAXM * 24 * 4;               // [MAXM] f32
constexpr int V1_BT = round_up(V1_RS + 64, 128);           // wgmma: e4m3 B tile [20][272]
constexpr int V1_SX = V1_BT + KW_MAX * WG_KB_PITCH;        // wgmma: [20][MAXM] f32
constexpr int V1_FAC = V1_SX + KW_MAX * MAXM * 4;          // wgmma: [WARPS][20][MAXM] f32
constexpr int V1_WF = V1_FAC + WARPS * KW_MAX * MAXM * 4;  // wgmma: attn_norm slice f32 [640]
constexpr int V1_RED = V1_WF + KPART1 * 4;                 // [WARPS][16][MAXM] f32 partials (read by the peers)
constexpr int V1_END = V1_RED + WARPS * 16 * MAXM * 4;
// G2 HMMA (red aliases the activation tile after the MMA barrier)
constexpr int V2_ACT = 0;
constexpr int V2_XE = V2_ACT + MAXM * act_pitch(QR);
constexpr int V2_QSQ = V2_XE + round_up(KB2 * MAXM, 128);  // [Q_TILES][MAXM] f32
constexpr int V2_NW = V2_QSQ + Q_TILES * MAXM * 4;         // q_norm bf16 [1280]
constexpr int V2_RS = V2_NW + QR * 2;
constexpr int V2_END = V2_RS + 64;
// G2 wgmma
constexpr int V2W_BT = 0;
constexpr int V2W_SX = V2W_BT + KB2 * WG_KB_PITCH;
constexpr int V2W_FAC = V2W_SX + KB2 * MAXM * 4;
constexpr int V2W_RS = V2W_FAC + WARPS * KW_MAX * MAXM * 4;
constexpr int V2W_RED = V2W_RS + 64;
constexpr int V2W_END = V2W_RED + WARPS * 16 * MAXM * 4;
constexpr int cmax(int a, int b) { return a > b ? a : b; }
constexpr int V3_BAR = V3_A + round_up(cmax(V1_END, cmax(V2_END, V2W_END)), 16);  // weight mbarrier
constexpr int CTAS_V3 = CTAS_CL;  // 30 clusters (a 31st does not fit: GPC limits at 2 x 113 KB smem per SM)
constexpr int V3_H0_G1 = G1_CTAS - 2;  // G1 CTA hosting the Sinkhorn after its tile work (on a G2 CTA it delays that
                                       // CTA's wq_b work: the Sinkhorn takes ~5 us under the weight stream)
constexpr int V3_H1_G1 = G1_CTAS - 1;  // G1 CTA hosting the KV insert after its tile work
constexpr int V3_NCH = 8, V3_CH = V3_WBYTES / V3_NCH;  // weight chunks of 10 KB (G1: one tile, G2: one tile half)
constexpr int V3_PACE_WINDOW = 2;                        // FAST_V3_PACE: chunks in flight per CTA
constexpr int V3_PBAR = V3_BAR + 8 * V3_NCH;  // FAST_V3_PUSH: mbarrier of the 8 pushed k-part partials (G1)
constexpr int V3_SMEM = V3_PBAR + 16;
static_assert(KSPLIT1 * 32 * 16 == WARPS * 16 * MAXM * 4, "FAST_V3_PUSH: the receive slots [k-part][lane] f32x4 fill V1_RED");
static_assert(V1_SSUM % 16 == 0 && V1_NW % 16 == 0 && V1_CF % 16 == 0 && V1_SX % 16 == 0 && V1_FAC % 16 == 0 && V1_WF % 16 == 0 &&
              V1_RED % 16 == 0 && V2_XE % 16 == 0 && V2_QSQ % 16 == 0 && V2_NW % 16 == 0 && V2W_SX % 16 == 0 && V2W_RED % 16 == 0);
static_assert(MAXM * 16 * 4 <= V3_WBYTES, "H0 comb logits reuse the (consumed) weight area");
static_assert(MAXM * SLICES * 4 <= V1_RED - V1_BT, "FAST_V3_HOP: the xsq copy aliases the B tile area");
static_assert(WARPS * 16 * MAXM * 4 <= MAXM * act_pitch(QR), "G2 HMMA red aliases the activation tile");
static_assert(WARPS * KW_MAX * MAXM * 4 <= V2_RS - V2_QSQ, "FAST_V3_FAC: the G2 HMMA factor tables alias qsq + the q_norm slice");
static_assert(WARPS * KW_MAX * MAXM * 4 <= V1_WF - V1_FAC, "FAST_V3_FAC: the G1 HMMA factor tables use the wgmma factor area");

__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(unsigned long long* bar, unsigned bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
// Global -> own shared memory bulk copy completing on `bar` (bytes % 16 == 0, both addresses 16-byte aligned).
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, unsigned long long* bar, unsigned long long pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
               : "memory");
}
__device__ __forceinline__ bool mbar_test(unsigned long long* bar, unsigned parity) {
  unsigned r;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(r)
               : "r"(smem_u32(bar)), "r"(parity)
               : "memory");
  return r != 0;
}
// A CTA's weight stream: V3_NCH bulk copies of V3_CH bytes (chunk c: src + c * stride -> dst + c * V3_CH), one mbarrier
// each; driven by thread 0 only. FAST_V3_PACE keeps at most V3_PACE_WINDOW chunks in flight (step() is called from
// thread 0's spin-waits), so the SM's request queue never holds more than ~2 x 20 KB of weights ahead of a
// latency-critical load; otherwise everything is issued at once.
struct WStream {
  uint8_t* dst;
  const uint8_t* src;
  long stride;
  unsigned long long* bars;
  unsigned long long pol;
  int issued, done;
  const unsigned* gate;  // optional: once *gate >= G2_CTAS (the P1a hop is complete everywhere) the window opens fully
  int window;            // chunks in flight (FAST_V3_PACE: V3_PACE_WINDOW, FAST_V3_PACE1: 1)
  __device__ __forceinline__ void init() {
    for (int c = 0; c < V3_NCH; ++c) mbar_init(bars + c, 1);
    fence_mbar_init();
    pol = evict_first_policy();
    issued = done = 0;
  }
  __device__ __forceinline__ void issue_one() {
    const int c = issued++;
    mbar_expect_tx(bars + c, V3_CH);
    bulk_g2s(dst + c * V3_CH, src + c * stride, V3_CH, bars + c, pol);
  }
  __device__ __forceinline__ void step() {
    if (gate != nullptr && ld_relaxed_gpu(gate) >= unsigned(G2_CTAS)) {
      all();
      return;
    }
    while (done < issued && mbar_test(bars + done, 0)) ++done;
    while (issued < V3_NCH && issued - done < window) issue_one();
  }
  __device__ __forceinline__ void all() {
    while (issued < V3_NCH) issue_one();
  }
  // FAST_V3_L2PF: pull every chunk into L2 (evict_last; the later TMA copy reads it with evict_first). A prefetch returns
  // nothing to the SM, so the HBM stream starts early without queueing the SM's latency-critical loads behind 80 KB of
  // returning weights (the TMA copy into smem is issued once the CTA's critical loads are in flight and hits L2).
  __device__ __forceinline__ void prefetch_all() const {
    unsigned long long last;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(last));
    for (int c = 0; c < V3_NCH; ++c)
      asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(src + c * stride), "r"(unsigned(V3_CH)), "l"(last)
                   : "memory");
  }
};
// cta_wait with thread 0 driving the weight stream while it polls.
// acq (FAST_V3_ACQPOLL): every poll is an ld.acquire (LDG.STRONG + CCTL.IVALL), so the load that sees the target is
// already the acquire -- one L2 round trip less after the flip than relaxed polls + a final ld.acquire.
__device__ __forceinline__ void cta_wait_ws(const unsigned* ctr, unsigned target, WStream& ws, bool pace, bool fence = false,
                                            bool acq = false, bool noacq = false) {
  if (threadIdx.x == 0) {
    HangCheck hc;
    unsigned v;
    if (acq) {
      while ((v = ld_acquire_gpu(ctr)) < target) {
        if (pace) ws.step();
        hc.tick(2, v, target);
      }
    } else {
      while ((v = ld_relaxed_gpu(ctr)) < target) {
        if (pace) ws.step();
        hc.tick(2, v, target);
      }
      if (noacq) {
        // hint only (DS41H_PROLOGUE_LL): the data carries its own flags
      } else if (fence) {
        fence_acq_rel_gpu();
      } else {
        (void)ld_acquire_gpu(ctr);
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ uint4 lds128(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(smem_u32(p)));
  return v;
}

// HMMA path of mma_frag with the fragment in shared memory (this warp's KW_MAX consecutive 512-byte chunks): same
// operations and accumulation order, so bit-identical to mma_frag; rolled by 4 (small hot code).
__device__ __forceinline__ void mma_smem(const uint8_t* wsm, unsigned sword, const uint8_t* act, int pitch, const uint8_t* xe, int kba0, int lane,
                                         float (&acc)[4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* arow = act + g * pitch + t * 16 + kba0 * 64;
  const uint8_t* xrow = xe + kba0 * MAXM + 2 * t;
  const uint8_t* wl = wsm + lane * 16;
#pragma unroll 4
  for (int i = 0; i < KW_MAX; ++i) {
    const uint4 b = *reinterpret_cast<const uint4*>(arow + i * 64);
    const uint4 v = lds128(wl + i * CHUNK);
    float d[4];
    hmma_f16_zero(d, f16x2_lo(v.x), f16x2_lo(v.z), f16x2_hi(v.x), f16x2_hi(v.z), b.x, b.y);
    hmma_f16(d, f16x2_lo(v.y), f16x2_lo(v.w), f16x2_hi(v.y), f16x2_hi(v.w), b.z, b.w);
    const unsigned sw = (__shfl_sync(FULL, sword, i >> 2) >> (8 * (i & 3))) & 0xffu;
    const unsigned ex = *reinterpret_cast<const unsigned short*>(xrow + i * MAXM);
    const float f0 = post_scale(sw, ex & 0xffu), f1 = post_scale(sw, ex >> 8);
    acc[0] = fmaf(d[0], f0, acc[0]);
    acc[1] = fmaf(d[1], f1, acc[1]);
    acc[2] = fmaf(d[2], f0, acc[2]);
    acc[3] = fmaf(d[3], f1, acc[3]);
  }
}

// FAST_V3_FAC: the per-block promotion factors of this warp's KW_MAX blocks, precomputed once per warp:
// fac[i][tok] = post_scale(sw_i, xe[kba0 + i][tok]) (sw_i = byte i of the tile's scale words: lane i / 4 holds bytes
// 4 (i / 4) .. + 3), i.e. exactly the values mma_smem derives per block with a shuffle + LDS.U16 + integer math.
__device__ __forceinline__ void hmma_factors(float* fac, const uint8_t* xe, int kba0, unsigned sword, int lane) {
  static_assert(KW_MAX * MAXM % 32 == 0);
#pragma unroll
  for (int r = 0; r < KW_MAX * MAXM / 32; ++r) {
    const int e = r * 32 + lane, i = e >> 3, tok = e & 7;  // i >> 2 == r: uniform shuffle source
    const unsigned sw = (__shfl_sync(FULL, sword, r) >> (8 * (i & 3))) & 0xffu;
    fac[e] = post_scale(sw, xe[(kba0 + i) * MAXM + tok]);
  }
}
// mma_smem with the factors read from fac (hmma_factors): one LDS.64 per block instead of ~12 instructions of scale
// bookkeeping. Same HMMAs, same factor values, same FFMA order -> bit-identical.
__device__ __forceinline__ void mma_smem_f(const uint8_t* wsm, const float* fac, const uint8_t* act, int pitch, int kba0, int lane,
                                           float (&acc)[4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* arow = act + g * pitch + t * 16 + kba0 * 64;
  const float* fp = fac + 2 * t;
  const uint8_t* wl = wsm + lane * 16;
#pragma unroll 4
  for (int i = 0; i < KW_MAX; ++i) {
    const uint4 b = *reinterpret_cast<const uint4*>(arow + i * 64);
    const uint4 v = lds128(wl + i * CHUNK);
    float d[4];
    hmma_f16_zero(d, f16x2_lo(v.x), f16x2_lo(v.z), f16x2_hi(v.x), f16x2_hi(v.z), b.x, b.y);
    hmma_f16(d, f16x2_lo(v.y), f16x2_lo(v.w), f16x2_hi(v.y), f16x2_hi(v.w), b.z, b.w);
    const float2 f = *reinterpret_cast<const float2*>(fp + i * MAXM);
    acc[0] = fmaf(d[0], f.x, acc[0]);
    acc[1] = fmaf(d[1], f.y, acc[1]);
    acc[2] = fmaf(d[2], f.x, acc[2]);
    acc[3] = fmaf(d[3], f.y, acc[3]);
  }
}

// DS41H_PROLOGUE_MMA2: mma_smem_f for two tiles sharing each activation fragment (one LDS.128 of b per block for both).
// Each tile's HMMAs, factors and FFMA order are exactly mma_smem_f's -> bit-identical per tile.
__device__ __forceinline__ void mma_smem_f2(const uint8_t* wsm0, const uint8_t* wsm1, const float* fac0, const float* fac1,
                                            const uint8_t* act, int pitch, int kba0, int lane, float (&acc0)[4], float (&acc1)[4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* arow = act + g * pitch + t * 16 + kba0 * 64;
  const float* fp0 = fac0 + 2 * t;
  const float* fp1 = fac1 + 2 * t;
  const uint8_t* wl0 = wsm0 + lane * 16;
  const uint8_t* wl1 = wsm1 + lane * 16;
#pragma unroll 4
  for (int i = 0; i < KW_MAX; ++i) {
    const uint4 b = *reinterpret_cast<const uint4*>(arow + i * 64);
    const uint4 v0 = lds128(wl0 + i * CHUNK);
    const uint4 v1 = lds128(wl1 + i * CHUNK);
    float d0[4], d1[4];
    hmma_f16_zero(d0, f16x2_lo(v0.x), f16x2_lo(v0.z), f16x2_hi(v0.x), f16x2_hi(v0.z), b.x, b.y);
    hmma_f16_zero(d1, f16x2_lo(v1.x), f16x2_lo(v1.z), f16x2_hi(v1.x), f16x2_hi(v1.z), b.x, b.y);
    hmma_f16(d0, f16x2_lo(v0.y), f16x2_lo(v0.w), f16x2_hi(v0.y), f16x2_hi(v0.w), b.z, b.w);
    hmma_f16(d1, f16x2_lo(v1.y), f16x2_lo(v1.w), f16x2_hi(v1.y), f16x2_hi(v1.w), b.z, b.w);
    const float2 f0 = *reinterpret_cast<const float2*>(fp0 + i * MAXM);
    const float2 f1 = *reinterpret_cast<const float2*>(fp1 + i * MAXM);
    acc0[0] = fmaf(d0[0], f0.x, acc0[0]);
    acc0[1] = fmaf(d0[1], f0.y, acc0[1]);
    acc0[2] = fmaf(d0[2], f0.x, acc0[2]);
    acc0[3] = fmaf(d0[3], f0.y, acc0[3]);
    acc1[0] = fmaf(d1[0], f1.x, acc1[0]);
    acc1[1] = fmaf(d1[1], f1.y, acc1[1]);
    acc1[2] = fmaf(d1[2], f1.x, acc1[2]);
    acc1[3] = fmaf(d1[3], f1.y, acc1[3]);
  }
}

// wg_mma with the A fragments read from shared memory right before their wgmma (double-buffered registers).
template <int KW>
__device__ __forceinline__ void wg_mma_smem(const uint8_t* wsm, const uint8_t* bt, const float* fac, int lane, float (&acc)[4]) {
  constexpr int NG = KW / WG_G;
  static_assert(KW % WG_G == 0 && NG >= 1, "KW must be a multiple of the group size");
  const unsigned desc0 = wg_desc_lo(bt);
  const float* fp = fac + 2 * (lane & 3);
  const uint8_t* wl = wsm + lane * 16;
  uint4 a[2][WG_G];
  float d[2][WG_G][4];
#pragma unroll
  for (int g = 0; g <= NG; ++g) {
    if (g < NG) {
#pragma unroll
      for (int s = 0; s < WG_G; ++s) a[g & 1][s] = lds128(wl + (g * WG_G + s) * CHUNK);
      wgmma_fence();
#pragma unroll
      for (int s = 0; s < WG_G; ++s) wgmma_m64n8k32(d[g & 1][s], a[g & 1][s], desc0 + unsigned((g * WG_G + s) * (WG_KB_PITCH >> 4)));
      wgmma_commit();
    }
    if (g >= 1) {
      if (g < NG)
        wgmma_wait<1>();
      else
        wgmma_wait<0>();
      const int gp = g - 1;
#pragma unroll
      for (int s = 0; s < WG_G; ++s) {
        float(&dd)[4] = d[gp & 1][s];
        reg_fence(dd[0]), reg_fence(dd[1]), reg_fence(dd[2]), reg_fence(dd[3]);
        uint4& aa = a[gp & 1][s];  // A registers stay live until their wgmma retired
        reg_fence(aa.x), reg_fence(aa.y), reg_fence(aa.z), reg_fence(aa.w);
        const float2 fc = *reinterpret_cast<const float2*>(fp + (gp * WG_G + s) * MAXM);
        acc[0] = fmaf(dd[0], fc.x, acc[0]);
        acc[1] = fmaf(dd[1], fc.y, acc[1]);
        acc[2] = fmaf(dd[2], fc.x, acc[2]);
        acc[3] = fmaf(dd[3], fc.y, acc[3]);
      }
    }
  }
}

// v3 G1: collapse of columns [col0, col0 + 640) for all tokens (same arithmetic as p1_collapse / g1_collapse), all
// global loads issued before any math; bf16 rows -> act (pitch layout), slice square sums -> ssum[tok][20].
__device__ __forceinline__ void v3_collapse(const Params& p, uint8_t* act, float* cf, float* ssum, int col0, int tid, int warp, int lane,
                                            bool write_resid) {
  constexpr int PITCH = act_pitch(KPART1), V8 = KPART1 / 8, NV = (MAXM * V8 + THREADS - 1) / THREADS;
  const int m = p.m, n = m * V8;
  uint4 yv[NV], rv[NV][HC];
#pragma unroll
  for (int j = 0; j < NV; ++j) {
    const int v = min(tid + j * THREADS, n - 1), tok = v / V8, col = col0 + (v - tok * V8) * 8;
    yv[j] = __ldg(reinterpret_cast<const uint4*>(p.x + long(tok) * HID + col));
#pragma unroll
    for (int k = 0; k < HC; ++k) rv[j][k] = __ldg(reinterpret_cast<const uint4*>(p.resid + (long(tok) * HC + k) * HID + col));
  }
  if (tid < m * 24) {
    const int tok = tid / 24, i = tid - tok * 24;
    cf[tid] = i < HC ? __ldg(p.post + tok * HC + i) : i < 2 * HC ? __ldg(p.pre + tok * HC + i - HC) : __ldg(p.comb + tok * HC * HC + i - 2 * HC);
  }
  __syncthreads();
  auto bf_at = [](const uint4& v, int i) {
    const unsigned w = i < 2 ? v.x : i < 4 ? v.y : i < 6 ? v.z : v.w;
    return __uint_as_float((i & 1) ? (w & 0xffff0000u) : (w << 16));
  };
#pragma unroll
  for (int j = 0; j < NV; ++j) {
    const int v = tid + j * THREADS;
    if (v < n) {
      const int tok = v / V8, c8 = v - tok * V8;
      const float* c = cf + tok * 24;
      unsigned pk[4], ro[HC][4];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float y = bf_at(yv[j], i);
        float rb[HC];
#pragma unroll
        for (int jj = 0; jj < HC; ++jj) {  // p1_collapse: t = post[j]*y; t = fma(comb[k][j], r[k], t), k = 0..3; bf16
          float t = __fmul_rn(c[jj], y);
#pragma unroll
          for (int k = 0; k < HC; ++k) t = __fmaf_rn(c[2 * HC + k * HC + jj], bf_at(rv[j][k], i), t);
          const unsigned short hb = __bfloat16_as_ushort(__float2bfloat16(t));
          ro[jj][i >> 1] = (i & 1) ? (ro[jj][i >> 1] | (unsigned(hb) << 16)) : unsigned(hb);
          rb[jj] = __uint_as_float(unsigned(hb) << 16);
        }
        float xc = __fmul_rn(c[HC], rb[0]);
#pragma unroll
        for (int jj = 1; jj < HC; ++jj) xc = __fmaf_rn(c[HC + jj], rb[jj], xc);
        const unsigned short h = __bfloat16_as_ushort(__float2bfloat16(xc));
        pk[i >> 1] = (i & 1) ? (pk[i >> 1] | (unsigned(h) << 16)) : unsigned(h);
      }
      *reinterpret_cast<uint4*>(act + tok * PITCH + c8 * 16) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
      if (write_resid) {  // the post-mapped residual stream output (stock mhc_shifted_post_pre `residual`)
#pragma unroll
        for (int jj = 0; jj < HC; ++jj)
          *reinterpret_cast<uint4*>(p.resid_out + (long(tok) * HC + jj) * HID + col0 + c8 * 8) = make_uint4(ro[jj][0], ro[jj][1], ro[jj][2], ro[jj][3]);
        if (p.aux != nullptr) {  // capture_aux (see p1_collapse)
          float av[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float r4[HC];
#pragma unroll
            for (int jj = 0; jj < HC; ++jj) r4[jj] = __uint_as_float((i & 1) ? (ro[jj][i >> 1] & 0xffff0000u) : (ro[jj][i >> 1] << 16));
            av[i] = ((r4[0] + r4[1]) + (r4[2] + r4[3])) * 0.25f;
          }
          *reinterpret_cast<uint4*>(p.aux + long(tok) * HID + col0 + c8 * 8) = pack8_bf16(av);
        }
      }
    }
  }
  __syncthreads();
  for (int jt = warp; jt < G1_SL * m; jt += WARPS) {  // (slice, token) jobs; lane = column (p1_collapse's butterfly)
    const int tok = jt / G1_SL, j = jt - tok * G1_SL;
    const float xf = bf2f(*reinterpret_cast<const __nv_bfloat16*>(act + tok * PITCH + (j * 32 + lane) * 2));
    const float xs = warp_sum(xf * xf);
    if (lane == 0) ssum[tok * G1_SL + j] = xs;
  }
}

// FAST_V3_PUSH: 16-byte store into a peer CTA's shared memory completing on the peer's mbarrier (DSMEM write, no read round trip).
__device__ __forceinline__ void st_async_v4(unsigned dst, const float (&v)[4], unsigned bar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst), "f"(v[0]), "f"(v[1]),
               "f"(v[2]), "f"(v[3]), "r"(bar)
               : "memory");
}
// wait for phase `parity` of a local mbarrier whose transactions come from other CTAs of the cluster (acquire.cluster)
__device__ __forceinline__ void mbar_wait_cl(unsigned long long* bar, unsigned parity) {
  HangCheck hc;
  for (;;) {
    unsigned r;
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(r)
                 : "r"(smem_u32(bar)), "r"(parity)
                 : "memory");
    if (r) break;
    hc.tick(4, 0u, parity);
  }
}
__device__ __forceinline__ void cl_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

// FAST_V3_PUSH finisher (warp 0 of cluster rank r): the 8 k-part partials of tile grp*8 + r were pushed by warp r of every
// k-part CTA into rbuf[k][lane] (f32x4 = that lane's C fragment acc[0..3]); summed in k-part order 0..7 exactly like
// g1_cluster_finish (x0[k] = acc[0..1], x1[k] = acc[2..3] of the same lane) -> bit-identical, then the same epilogue.
__device__ __forceinline__ void g1_push_finish(const Params& p, const float* rbuf, int tile, int lane, unsigned llflag = 0u) {
  const int g = lane >> 2, t = lane & 3, m = p.m;
  float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  float4 x[KSPLIT1];
#pragma unroll
  for (int k = 0; k < KSPLIT1; ++k) x[k] = *reinterpret_cast<const float4*>(rbuf + (k * 32 + lane) * 4);
#pragma unroll
  for (int k = 0; k < KSPLIT1; ++k) v0 += x[k].x, v1 += x[k].y, v2 += x[k].z, v3 += x[k].w;
  const __nv_bfloat16 b0 = __float2bfloat16(v0), b1 = __float2bfloat16(v1), b2 = __float2bfloat16(v2), b3 = __float2bfloat16(v3);
  const int c = tile * 16 + g;
  const bool ll = llflag != 0u && tile < Q_TILES;
  const unsigned long long fl = (unsigned long long)llflag << 32;
  if (ll && (p.probe & 4) && lane == 0) red_relaxed(p.pc + C_QHINT, 1u);  // test probe: hint BEFORE the words (slow path)
  if (ll) {  // DS41H_PROLOGUE_LL: word (tok, tile, g) = {bf16 col c | bf16 col c + 8 << 16, flag}, before the regular stores
    unsigned long long* w = p.qll + tile * 8 + g;
    if (2 * t < m)
      st_relaxed_u64(w + (2 * t) * (Q_TILES * 8), fl | __bfloat16_as_ushort(b0) | (unsigned(__bfloat16_as_ushort(b2)) << 16));
    if (2 * t + 1 < m)
      st_relaxed_u64(w + (2 * t + 1) * (Q_TILES * 8), fl | __bfloat16_as_ushort(b1) | (unsigned(__bfloat16_as_ushort(b3)) << 16));
  }
  if (2 * t < m) p.qkv[(2 * t) * QKV + c] = b0, p.qkv[(2 * t) * QKV + c + 8] = b2;
  if (2 * t + 1 < m) p.qkv[(2 * t + 1) * QKV + c] = b1, p.qkv[(2 * t + 1) * QKV + c + 8] = b3;
  float s0 = bf2f(b0) * bf2f(b0) + bf2f(b2) * bf2f(b2), s1 = bf2f(b1) * bf2f(b1) + bf2f(b3) * bf2f(b3);
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    s0 += __shfl_xor_sync(FULL, s0, o);
    s1 += __shfl_xor_sync(FULL, s1, o);
  }
  if (g == 0) {
    if (ll) {
      if (2 * t < m) st_relaxed_u64(p.qsqll + tile * MAXM + 2 * t, fl | __float_as_uint(s0));
      if (2 * t + 1 < m) st_relaxed_u64(p.qsqll + tile * MAXM + 2 * t + 1, fl | __float_as_uint(s1));
    }
    if (2 * t < m) p.qsq[tile * MAXM + 2 * t] = s0;
    if (2 * t + 1 < m) p.qsq[tile * MAXM + 2 * t + 1] = s1;
  }
  __syncwarp();
  if (ll && !(p.probe & 4) && lane == 0) red_relaxed(p.pc + C_QHINT, 1u);  // hint only: may overtake the LL words (re-read)
  if (lane == 0) {
    if (p.probe & 1)  // timing probe only: no release ordering (racy)
      asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" ::"l"(p.pc + (tile < Q_TILES ? C_P3Q : C_P3K)) : "memory");
    else
      red_release(p.pc + (tile < Q_TILES ? C_P3Q : C_P3K), 1u);
    PROF_MAX(9);  // G1: tile finish of this CTA
  }
}

// DS41H_PROLOGUE_LL & 2 consumer (G1 HOP path): this CTA's 640 collapse columns of the m tokens (LL words of p1_collapse)
// -> act rows (bf16, the cp.async path's layout) and all m x SLICES square sums -> xsq; one load round, then flag checks
// (slow path re-reads the words the relaxed hint overtook).
__device__ __forceinline__ void ll_put_x(uint8_t* act, int pitch, int s0, int u, unsigned long long a, unsigned long long b) {
  constexpr int UPS = 8;  // 16-byte units (word pairs) per slice
  const int tok = u / (20 * UPS), r = u - tok * (20 * UPS), sl = r / UPS, q = r - sl * UPS;
  (void)s0;
  uint8_t* dst = act + tok * pitch + (sl * 32 + 4 * q) * 2;
  *reinterpret_cast<uint2*>(dst) = make_uint2(unsigned(a), unsigned(b));
}
__device__ __forceinline__ void ll_stage_x(const Params& p, uint8_t* act, int pitch, float* xsq, int m, int col0, int tid,
                                           unsigned flag) {
  constexpr int UPS = 8, SL = KPART1 / 32;                                  // 20 slices of this k-part
  constexpr int UPT = (MAXM * SL * UPS + THREADS - 1) / THREADS;            // 5
  constexpr int QPT = (MAXM * SLICES + THREADS - 1) / THREADS;              // 5
  const int s0 = col0 / 32, nu = m * SL * UPS, nq = m * SLICES;
  auto uptr = [&](int u) {
    const int tok = u / (SL * UPS), r = u - tok * (SL * UPS), sl = r / UPS, q = r - sl * UPS;
    return p.xll + (long(tok) * SLICES + s0 + sl) * 16 + 2 * q;
  };
  unsigned long long a[UPT], b[UPT], x[QPT];
#pragma unroll
  for (int k = 0; k < UPT; ++k) {
    const int u = tid + k * THREADS;
    a[k] = b[k] = (unsigned long long)flag << 32;
    if (u < nu) ld_relaxed_v2u64(uptr(u), a[k], b[k]);
  }
#pragma unroll
  for (int k = 0; k < QPT; ++k) {
    const int i = tid + k * THREADS;
    x[k] = (unsigned long long)flag << 32;
    if (i < nq) x[k] = ld_relaxed_u64(p.xsqll + i);
  }
  bool ok = true;
#pragma unroll
  for (int k = 0; k < UPT; ++k) ok = ok && unsigned(a[k] >> 32) == flag && unsigned(b[k] >> 32) == flag;
#pragma unroll
  for (int k = 0; k < QPT; ++k) ok = ok && unsigned(x[k] >> 32) == flag;
  if (ok) {
#pragma unroll
    for (int k = 0; k < UPT; ++k)
      if (tid + k * THREADS < nu) ll_put_x(act, pitch, s0, tid + k * THREADS, a[k], b[k]);
#pragma unroll
    for (int k = 0; k < QPT; ++k)
      if (tid + k * THREADS < nq) xsq[tid + k * THREADS] = __uint_as_float(unsigned(x[k]));
    return;
  }
  HangCheck hc;
#pragma unroll 1
  for (int u = tid; u < nu; u += THREADS) {
    unsigned long long y, z;
    for (;;) {
      ld_relaxed_v2u64(uptr(u), y, z);
      if (unsigned(y >> 32) == flag && unsigned(z >> 32) == flag) break;
      hc.tick(7, unsigned(y >> 32), flag);
    }
    ll_put_x(act, pitch, s0, u, y, z);
  }
#pragma unroll 1
  for (int i = tid; i < nq; i += THREADS) {
    unsigned long long y;
    for (;;) {
      y = ld_relaxed_u64(p.xsqll + i);
      if (unsigned(y >> 32) == flag) break;
      hc.tick(7, unsigned(y >> 32), flag);
    }
    xsq[i] = __uint_as_float(unsigned(y));
  }
}

// DS41H_PROLOGUE_TICKET: after the PDL wait (the ticket atomic issued at entry has returned by then), select this launch's
// counter block; the launch's first ticket holder zeroes the other one.
__device__ __forceinline__ void tick_publish(Params& p, unsigned long long t, int tid) {
  if (!p.tick) return;
  __shared__ unsigned tick_blk;
  if (tid == 0) {
    const unsigned long long e = t / (unsigned long long)CTAS_V3;
    if (t % (unsigned long long)CTAS_V3 == 0) {
      unsigned* other = p.pcf + (((e + 1ull) & 1ull) ? PCB1 : 0);
      for (int i = 0; i < NCOUNTERS; ++i) other[i] = 0u;
    }
    tick_blk = unsigned(e & 1ull);
  }
  __syncthreads();
  p.pc = p.pcf + (tick_blk ? PCB1 : 0);
}

template <int MMA>
__device__ __forceinline__ void v3_g1(Params& p, uint8_t* sm, int b, int tid, int warp, int lane, unsigned long long tick_t) {
  const int kp = b % KSPLIT1, grp = b / KSPLIT1, tile = grp * WARPS + warp, m = p.m, col0 = kp * KPART1;
  uint8_t* ra = sm + V3_A;
  unsigned long long* wbar = reinterpret_cast<unsigned long long*>(sm + V3_BAR);
  // Weight stream (chunk t = tile grp*8 + t, k-part kp). FAST_V3_W1_LATE (with FAST_V3_HOP): it starts only once the
  // P1a hop is seen and this CTA's critical copies are queued, so the latency-critical loads of the CTAs sharing this
  // SM are not queued behind it; else at entry (immutable: before the PDL wait), all at once or paced (FAST_V3_PACE).
  const bool pace = p.fast & FAST_V3_PACE;
  WStream ws{sm, p.wqa + (long(grp * WARPS) * KB1 + kp * KW_MAX) * CHUNK, long(KB1) * CHUNK, wbar, 0ull, 0, 0, nullptr,
             (p.fast & FAST_V3_PACE1) ? 1 : V3_PACE_WINDOW};
  const bool late_w = (p.fast & (FAST_V3_W1_LATE | FAST_V3_HOP)) == (FAST_V3_W1_LATE | FAST_V3_HOP);
  auto issue_weights = [&]() {  // (late) start: everything now
    if (tid == 0) {
      ws.init();
      ws.all();
    }
  };
  const bool l2pf = p.fast & FAST_V3_L2PF;
  if (l2pf) {
    if (tid == 0) {  // HBM -> L2 now; the smem copy is issued after the C_P1A hop's critical copies
      ws.init();
      ws.prefetch_all();
    }
  } else if (!late_w && tid == 0) {
    ws.init();
    if (pace)
      ws.step();
    else
      ws.all();
  }
  // FAST_V3_PUSH (HMMA): the tile partials are pushed into the finisher CTA's smem (st.async, completing on its mbarrier)
  // instead of a cluster barrier + DSMEM reads; the barrier is initialized now and released to the cluster (relaxed
  // arrive here, the matching wait right before the pushes).
  const bool push = MMA == MMA_HMMA && (p.fast & FAST_V3_PUSH) && !(p.fast & FAST_V3_ADEFER);
  unsigned long long* pbar = reinterpret_cast<unsigned long long*>(sm + V3_PBAR);
  if (push) {
    if (tid == 0) {
      mbar_init(pbar, 1);
      fence_mbar_init();
      mbar_expect_tx(pbar, unsigned(KSPLIT1 * 32 * 16));
    }
    cl_arrive_relaxed();
  }
  const unsigned sword = lane * 4 < KW_MAX ? ld_stream4(p.wqa_s + long(tile) * KB1 + kp * KW_MAX + lane * 4) : 0u;
  pdl_wait();
  tick_publish(p, tick_t, tid);
  PROF_T0(1);
  // DS41H_PROLOGUE_LL: this launch's LL flag (the previous launch's last CTA bumped C_EPOCH before it completed)
  const unsigned epoch1 = (p.ll & 3) ? ld_relaxed_gpu(p.pcf + C_EPOCH) + 1u : 0u;
  const unsigned llflag = ((p.ll & 1) && push && grp * WARPS < Q_TILES) ? epoch1 : 0u;
  constexpr int PITCH = act_pitch(KPART1), V8 = KPART1 / 8;
  uint8_t* act = ra + V1_ACT;
  uint8_t* xe = ra + V1_XE;
  float* ssum = reinterpret_cast<float*>(ra + V1_SSUM);
  __nv_bfloat16* w = reinterpret_cast<__nv_bfloat16*>(ra + V1_NW);
  float* rs = reinterpret_cast<float*>(ra + V1_RS);
  for (int i = tid; i < V8; i += THREADS) cp_async16(w + i * 8, p.attn_norm + col0 + i * 8);
  cp_async_commit();
  const bool adefer = MMA == MMA_WGMMA && (p.fast & FAST_V3_ADEFER);
  if (adefer) {
    // the P1a jobs published the e4m3 B tile of u = bf16(collapse * attn_norm_w) + the collapse slice sums: copy this
    // k-part's 20 blocks (5.4 KB) and all sums; attn_norm's rsqrt is applied by the finisher (no staging at all)
    uint8_t* bt = ra + V1_BT;
    float* sx = reinterpret_cast<float*>(ra + V1_SX);
    float* xsq = reinterpret_cast<float*>(ra + V1_FAC);  // [MAXM][SLICES] f32; the promotion factors are written after rs
    cta_wait_ws(p.pc + C_P1A, unsigned(G2_CTAS), ws, pace && !late_w && !l2pf, p.fast & FAST_V3_FENCE, p.fast & FAST_V3_ACQPOLL);
    PROF_T0(2);  // G1: collapse / a tile published
    for (int i = tid; i < KW_MAX * WG_KB_PITCH / 16; i += THREADS) cp_async16(bt + 16 * i, p.qab + long(kp) * KW_MAX * WG_KB_PITCH + 16 * i);
    for (int i = tid; i < KW_MAX * MAXM / 4; i += THREADS) cp_async16(sx + 4 * i, p.qas + kp * KW_MAX * MAXM + 4 * i);
    for (int v = tid; v < m * SLICES / 4; v += THREADS) cp_async16(xsq + v * 4, p.xsq + v * 4);
    cp_async_commit();
    __syncwarp();
    if (late_w && warp == 0 && !l2pf) issue_weights();  // after this CTA's critical copies are queued (thread 0 is in warp 0)
    if ((l2pf || (!late_w && pace)) && tid == 0) ws.all();
    cp_async_wait_all();
    __syncthreads();
    for (int i = tid; i < MAXM * KW_MAX * 2; i += THREADS) {  // padding token rows (the tile is shared by all m)
      const int pt = i & (MAXM - 1);
      if (pt >= m) *reinterpret_cast<uint4*>(bt + (i >> 4) * WG_KB_PITCH + ((i >> 3) & 1) * 128 + pt * 16) = make_uint4(0u, 0u, 0u, 0u);
    }
    for (int i = tid; i < MAXM * KW_MAX; i += THREADS)
      if ((i & (MAXM - 1)) >= m) sx[i] = 0.0f;
    if (tid < MAXM && tid >= m) rs[tid] = 0.0f;
    if (warp < m) {  // attn_norm over the bf16 collapse, role_g1's order
      float s = 0.0f;
      for (int i = lane; i < SLICES; i += 32) s += xsq[warp * SLICES + i];
      s = warp_sum(s);
      if (lane == 0) rs[warp] = rsqrtf(__fdiv_rn(s, float(HID)) + p.norm_eps);
    }
  } else if (p.fast & FAST_V3_HOP) {
    // collapse published by the G2 P1a jobs (no redundant collapse math): its 640-column slice + all slice sums
    float* xsq = reinterpret_cast<float*>(ra + V1_BT);  // [MAXM][SLICES] f32; the B tile is written after rs
    const unsigned llx = (MMA == MMA_HMMA && (p.ll & 2) && !(p.fast & FAST_V3_EXT_P1A)) ? epoch1 : 0u;
    if (llx != 0u) {  // DS41H_PROLOGUE_LL & 2: relaxed hint, then the LL words (flags validated)
      cta_wait_ws(p.pc + C_P1AHINT, unsigned(G2_CTAS), ws, pace && !late_w && !l2pf, false, false, true);
      PROF_T0(2);  // G1: collapse published (hint)
      ll_stage_x(p, act, PITCH, xsq, m, col0, tid, llx);
    } else {
      if (!(p.fast & FAST_V3_EXT_P1A))  // EXT_P1A: published by the previous grid (visible after the PDL wait)
        cta_wait_ws(p.pc + C_P1A, unsigned(G2_CTAS), ws, pace && !late_w && !l2pf, p.fast & FAST_V3_FENCE, p.fast & FAST_V3_ACQPOLL);
      PROF_T0(2);  // G1: collapse published
      for (int v = tid; v < m * V8; v += THREADS) {
        const int tok = v / V8, c8 = v - tok * V8;
        cp_async16(act + tok * PITCH + c8 * 16, p.xcb + long(tok) * HID + col0 + c8 * 8);
      }
      for (int v = tid; v < m * SLICES / 4; v += THREADS) cp_async16(xsq + v * 4, p.xsq + v * 4);
    }
    cp_async_commit();
    __syncwarp();
    if (late_w && warp == 0 && !l2pf) issue_weights();  // after this CTA's critical copies are queued (thread 0 is in warp 0)
    if ((l2pf || (!late_w && pace)) && tid == 0) ws.all();
    cp_async_wait_all();
    __syncthreads();
    if (warp < m) {  // attn_norm over the bf16 collapse, role_g1's order
      float s = 0.0f;
      for (int i = lane; i < SLICES; i += 32) s += xsq[warp * SLICES + i];
      s = warp_sum(s);
      if (lane == 0) rs[warp] = rsqrtf(__fdiv_rn(s, float(HID)) + p.norm_eps);
    }
  } else {
    v3_collapse(p, act, reinterpret_cast<float*>(ra + V1_CF), ssum, col0, tid, warp, lane, grp == 0);
    if ((pace || l2pf) && tid == 0) ws.all();
    PROF_T0(2);  // G1: own collapse slice computed
    cl_arrive_release();  // slice sums visible to the cluster
    cl_wait_acquire();
    if (warp < m) {  // attn_norm statistics in the order of the global path (role_g1): lane L adds slices L, L + 32, ...
      float s = 0.0f;
      for (int i = lane; i < SLICES; i += 32) s += cl_ld(cl_map(ssum + warp * G1_SL + i % G1_SL, unsigned(i / G1_SL)));
      s = warp_sum(s);
      if (lane == 0) rs[warp] = rsqrtf(__fdiv_rn(s, float(HID)) + p.norm_eps);
    }
  }
  cp_async_wait_all();
  if constexpr (MMA == MMA_WGMMA) {
    __syncthreads();  // attn_norm slice landed for every thread
    widen_bf16(reinterpret_cast<float*>(ra + V1_WF), w, V8, tid);
  }
  __syncthreads();
  PROF_T0(3);  // G1: rs known
  const bool write_a = grp == 0;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc2[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // DS41H_PROLOGUE_MMA2: tile 2w + 1 of warp w < 4
  const bool mma2 = MMA == MMA_HMMA && (p.fast & FAST_V3_FAC) && p.mma2 && push;  // (the pushes carry the second tile)
  if (adefer) {
    fence_proxy_async_smem();  // the generic zero-fill of the padding rows -> wgmma operand reads
    __syncthreads();
    PROF_T0(4);  // G1: activations ready
    float* fac = reinterpret_cast<float*>(ra + V1_FAC) + warp * KW_MAX * MAXM;
    wg_factors<KW_MAX>(fac, reinterpret_cast<const float*>(ra + V1_SX), sword, KW_MAX, lane);
    __syncwarp();
    mbar_wait_hc(wbar + warp, 0, 3);  // chunk = this warp's tile
    PROF_T0(10);  // G1: weights in smem
    wg_mma_smem<KW_MAX>(sm + warp * KW_MAX * CHUNK, ra + V1_BT, fac, lane, acc);
  } else if constexpr (MMA == MMA_HMMA) {
    auto g1x = [&](int tok, int c8, float (&x)[8]) {
      const float r = rs[tok];
      float wf[8];
      unpack8(*reinterpret_cast<const uint4*>(w + c8 * 8), wf);
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = bf16_round(__fmul_rn(__fmul_rn(x[i], r), wf[i]));
      if (write_a) *reinterpret_cast<uint4*>(p.a + long(tok) * HID + col0 + c8 * 8) = pack8_bf16(x);
    };
    if (p.cvt)  // 640 x 8 / 8 = 640 vectors: 3 per thread in one batch (same values)
      convert_act_batched<(MAXM * KPART1 / 8 + THREADS - 1) / THREADS>(act, PITCH, xe, m, KPART1, tid, g1x);
    else
      convert_act(act, PITCH, xe, m, KPART1, tid, g1x);
    __syncthreads();
    PROF_T0(4);  // G1: activations staged
    if (mma2) {
      // DS41H_PROLOGUE_MMA2: every warp builds its tile's factor table, warps 0-3 run tiles (2w, 2w+1) on one act fragment
      float* fac = reinterpret_cast<float*>(ra + V1_FAC);
      hmma_factors(fac + warp * KW_MAX * MAXM, xe, 0, sword, lane);
      __syncthreads();
      if (warp < WARPS / 2) {
        mbar_wait_hc(wbar + 2 * warp, 0, 3);
        mbar_wait_hc(wbar + 2 * warp + 1, 0, 3);
        PROF_T0(10);  // G1: weights in smem
        mma_smem_f2(sm + (2 * warp) * KW_MAX * CHUNK, sm + (2 * warp + 1) * KW_MAX * CHUNK, fac + (2 * warp) * KW_MAX * MAXM,
                    fac + (2 * warp + 1) * KW_MAX * MAXM, act, PITCH, 0, lane, acc, acc2);
      }
    } else if (p.fast & FAST_V3_FAC) {
      float* fac = reinterpret_cast<float*>(ra + V1_FAC) + warp * KW_MAX * MAXM;  // (wgmma-only area in HMMA builds)
      hmma_factors(fac, xe, 0, sword, lane);
      __syncwarp();
      mbar_wait_hc(wbar + warp, 0, 3);  // chunk = this warp's tile
      PROF_T0(10);  // G1: weights in smem
      mma_smem_f(sm + warp * KW_MAX * CHUNK, fac, act, PITCH, 0, lane, acc);
    } else {
      mbar_wait_hc(wbar + warp, 0, 3);  // chunk = this warp's tile
      PROF_T0(10);  // G1: weights in smem
      mma_smem(sm + warp * KW_MAX * CHUNK, sword, act, PITCH, xe, 0, lane, acc);
    }
  } else {
    uint8_t* bt = ra + V1_BT;
    float* sx = reinterpret_cast<float*>(ra + V1_SX);
    float* fac = reinterpret_cast<float*>(ra + V1_FAC) + warp * KW_MAX * MAXM;
    const float4* wf = reinterpret_cast<const float4*>(ra + V1_WF);
    stage_e4m3<KW_MAX>(bt, sx, m, tid, THREADS, [&](int tok, int c8, float (&x)[8], bool ok) {
      unpack8(*reinterpret_cast<const uint4*>(act + tok * PITCH + c8 * 16), x);
      const uint4 pk = rmsnorm8_bf16(x, rs[tok], wf[2 * c8], wf[2 * c8 + 1]);
      if (write_a && ok) *reinterpret_cast<uint4*>(p.a + long(tok) * HID + col0 + c8 * 8) = pk;
    });
    fence_proxy_async_smem();
    __syncthreads();
    PROF_T0(4);  // G1: activations staged
    wg_factors<KW_MAX>(fac, sx, sword, KW_MAX, lane);
    __syncwarp();
    mbar_wait_hc(wbar + warp, 0, 3);  // chunk = this warp's tile
    PROF_T0(10);  // G1: weights in smem
    wg_mma_smem<KW_MAX>(sm + warp * KW_MAX * CHUNK, bt, fac, lane, acc);  // warpgroup = tiles grp*8 + {0..3 | 4..7}
  }
  PROF_T0(5);  // G1: warp 0 MMA done
  float* red = reinterpret_cast<float*>(ra + V1_RED);
  if (push) {
    cl_wait();  // every peer initialized its push mbarrier (arrived at entry)
    // warp w holds the k-part kp partial of tile grp*8 + w, finished by cluster rank w: slot [kp][lane] there
    if (!mma2) {
      st_async_v4(cl_map(red + (kp * 32 + lane) * 4, unsigned(warp)), acc, cl_map(pbar, unsigned(warp)));
    } else if (warp < WARPS / 2) {  // tiles 2w / 2w + 1 are finished by cluster ranks 2w / 2w + 1
      st_async_v4(cl_map(red + (kp * 32 + lane) * 4, unsigned(2 * warp)), acc, cl_map(pbar, unsigned(2 * warp)));
      st_async_v4(cl_map(red + (kp * 32 + lane) * 4, unsigned(2 * warp + 1)), acc2, cl_map(pbar, unsigned(2 * warp + 1)));
    }
    PROF_T0(6);  // G1: partials pushed
    if (warp == 0) {
      mbar_wait_cl(pbar, 0);  // the 8 partials of tile grp*8 + kp landed
      g1_push_finish(p, red, grp * WARPS + kp, lane, llflag);
    }
    return;  // no exit barrier: peers wrote into our smem (we waited for all of it), we only wrote into theirs
  }
  store_frag_tile(red + warp * 16 * MAXM, acc, lane);
  cl_arrive_release();
  cl_wait_acquire();
  PROF_T0(6);  // G1: cluster partials visible
  if (warp == 0) {
    const float* rsd = adefer ? rs : nullptr;  // deferred attn_norm rsqrt
    if (MMA == MMA_WGMMA && grp * WARPS < Q_TILES) {
      if ((kp & 1) == 0) g1_cluster_finish_pair(p, red, grp * WARPS + kp, unsigned(kp), lane, rsd);
    } else {
      g1_cluster_finish(p, red, grp * WARPS + kp, unsigned(kp), lane, rsd);
    }
  }
  if (adefer && write_a) {  // the `a` output (stock attn_norm: bf16((x * rsqrt) * w)) of this k-part, off the critical path
    for (int v = tid; v < m * V8; v += THREADS) {
      const int tok = v / V8, c8 = v - tok * V8;
      float x[8], wv[8];
      unpack8(__ldcg(reinterpret_cast<const uint4*>(p.xcb + long(tok) * HID + col0 + c8 * 8)), x);
      unpack8(*reinterpret_cast<const uint4*>(w + c8 * 8), wv);
      *reinterpret_cast<uint4*>(p.a + long(tok) * HID + col0 + c8 * 8) =
          rmsnorm8_bf16(x, rs[tok], make_float4(wv[0], wv[1], wv[2], wv[3]), make_float4(wv[4], wv[5], wv[6], wv[7]));
    }
  }
  cl_arrive_release();  // every peer read our red before it goes away / is reused
  cl_wait_acquire();
}

// v3 P1 on G2 CTA c (right after the PDL wait; nothing in G1 depends on it unless FAST_V3_HOP): its (slice, token)
// jobs: FAST_V3_HOP: the full P1a (residual output + collapse -> xcb / xsq, C_P1A), else only the fp32 stream values
// (the G1 CTAs compute the collapse and group 0 writes the residual output); then the partial mixes -> C_P1B.
// FAST_V3_EXT_P1A (with HOP, experimental): the residual output and xcb / xsq were produced by the PREVIOUS grid (target:
// the decoder layer kernel's AR #2 epilogue, see notes "PROPOSAL"; bench: an earlier launch on the same inputs), so the G2
// jobs only recompute the fp32 stream for the mixes and G1 reads xcb / xsq right after its PDL wait (no C_P1A hop).
template <class AfterP1a>
__device__ __forceinline__ void p1_v3_g2(const Params& p, int c, int warp, int lane, AfterP1a&& after_p1a, unsigned llx = 0u) {
  const bool hop = (p.fast & FAST_V3_HOP) && !(p.fast & FAST_V3_EXT_P1A);
  const P1Job j = p1_job(p, c * WARPS + warp);
  float nr[2][HC];
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    if (j.tok0 + t < j.tok1) {
      if (p.fast & FAST_V3_ADEFER)
        p1_collapse<3>(p, j.s, j.tok0 + t, lane, nr[t]);
      else if (hop)
        p1_collapse<0>(p, j.s, j.tok0 + t, lane, nr[t], llx);
      else
        p1_collapse<2>(p, j.s, j.tok0 + t, lane, nr[t]);
    }
  }
  if (hop && llx != 0u) {  // DS41H_PROLOGUE_LL & 2: relaxed hint only (G1 validates the words' flags), no release fence
    __syncthreads();
    if (threadIdx.x == 0) red_relaxed(p.pc + C_P1AHINT, 1u);
  } else if (hop) {
    if (p.probe & 2) {  // timing probe only: no release ordering (racy)
      __syncthreads();
      if (threadIdx.x == 0) asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" ::"l"(p.pc + C_P1A) : "memory");
    } else {
      cta_signal(p.pc + C_P1A, true);
    }
  }
  after_p1a();
  PROF_T0(9);  // G2: P1a done
#pragma unroll
  for (int t = 0; t < 2; ++t)
    if (j.tok0 + t < j.tok1) p1_mixes<false>(p, nullptr, j.s, j.tok0 + t, lane, nr[t]);
  cta_signal(p.pc + C_P1B, true);
  PROF_T0(2);  // G2: P1b published
}

// DS41H_PROLOGUE_LL consumer (G2, HMMA): the m qkv q rows as LL words (g1_push_finish) -> act rows (bf16, the layout the
// cp.async path produces), the per-tile q square sums -> qsq. All loads are issued first (one round trip); a word whose
// flag is not this launch's (the relaxed hint overtook it) is re-read until it is.
__device__ __forceinline__ void ll_put_q(uint8_t* act, int pitch, int u, unsigned long long a, unsigned long long b) {
  const int tok = u / (Q_TILES * 4), r = u - tok * (Q_TILES * 4), tile = r >> 2, g = (r & 3) * 2;
  uint8_t* row = act + tok * pitch + (tile * 16 + g) * 2;
  // word g = cols (c, c + 8), word g + 1 = cols (c + 1, c + 9)
  *reinterpret_cast<unsigned*>(row) = (unsigned(a) & 0xffffu) | (unsigned(b) << 16);
  *reinterpret_cast<unsigned*>(row + 16) = (unsigned(a) >> 16) | (unsigned(b) & 0xffff0000u);
}
__device__ __forceinline__ void ll_stage_q(const Params& p, uint8_t* act, int pitch, float* qsq, int m, int tid, unsigned flag) {
  constexpr int UPT = (MAXM * Q_TILES * 4 + THREADS - 1) / THREADS;  // 16-byte units (2 words) per thread
  constexpr int SPT = (Q_TILES * MAXM + THREADS - 1) / THREADS;
  const int nu = m * Q_TILES * 4, ns = Q_TILES * m;
  unsigned long long a[UPT], b[UPT], q[SPT];
#pragma unroll
  for (int k = 0; k < UPT; ++k) {
    const int u = tid + k * THREADS;
    a[k] = b[k] = (unsigned long long)flag << 32;
    if (u < nu) ld_relaxed_v2u64(p.qll + 2 * u, a[k], b[k]);
  }
#pragma unroll
  for (int k = 0; k < SPT; ++k) {
    const int i = tid + k * THREADS;
    q[k] = (unsigned long long)flag << 32;
    if (i < ns) {
      const int tile = i / m, tok = i - tile * m;
      q[k] = ld_relaxed_u64(p.qsqll + tile * MAXM + tok);
    }
  }
  bool ok = true;
#pragma unroll
  for (int k = 0; k < UPT; ++k) ok = ok && unsigned(a[k] >> 32) == flag && unsigned(b[k] >> 32) == flag;
#pragma unroll
  for (int k = 0; k < SPT; ++k) ok = ok && unsigned(q[k] >> 32) == flag;
  if (ok) {
#pragma unroll
    for (int k = 0; k < UPT; ++k)
      if (tid + k * THREADS < nu) ll_put_q(act, pitch, tid + k * THREADS, a[k], b[k]);
#pragma unroll
    for (int k = 0; k < SPT; ++k) {
      const int i = tid + k * THREADS;
      if (i < ns) {
        const int tile = i / m, tok = i - tile * m;
        qsq[tile * MAXM + tok] = __uint_as_float(unsigned(q[k]));
      }
    }
    return;
  }
  // slow path (the relaxed hint overtook some words): re-read this thread's words one by one until they carry the flag
  HangCheck hc;
#pragma unroll 1
  for (int u = tid; u < nu; u += THREADS) {
    unsigned long long x, y;
    for (;;) {
      ld_relaxed_v2u64(p.qll + 2 * u, x, y);
      if (unsigned(x >> 32) == flag && unsigned(y >> 32) == flag) break;
      hc.tick(7, unsigned(x >> 32), flag);
    }
    ll_put_q(act, pitch, u, x, y);
  }
#pragma unroll 1
  for (int i = tid; i < ns; i += THREADS) {
    const int tile = i / m, tok = i - tile * m;
    unsigned long long x;
    for (;;) {
      x = ld_relaxed_u64(p.qsqll + tile * MAXM + tok);
      if (unsigned(x >> 32) == flag) break;
      hc.tick(7, unsigned(x >> 32), flag);
    }
    qsq[tile * MAXM + tok] = __uint_as_float(unsigned(x));
  }
}

template <int MMA>
__device__ __forceinline__ void v3_g2(Params& p, uint8_t* sm, int c, int tid, int warp, int lane, unsigned long long tick_t) {
  uint8_t* ra = sm + V3_A;
  unsigned long long* wbar = reinterpret_cast<unsigned long long*>(sm + V3_BAR);
  // Warp -> (tile, k-half) as role_g2 (wgmma: warpgroup = k-half, its 4 warps are the CTA's 4 tiles).
  const int tl = MMA == MMA_WGMMA ? (warp & 3) : (warp >> 1);
  const int half = MMA == MMA_WGMMA ? __shfl_sync(FULL, warp >> 2, 0) : (warp & 1);
  const int tile = c * G2_TILES + tl, m = p.m;
  const bool late_w = p.fast & FAST_V3_LATE_W2;
  // weight stream: chunk ch = tile-half (tl, half) at (tl * 2 + half) * 10 KB (the CTA's 80 KB are contiguous)
  const bool pace = p.fast & FAST_V3_PACE;
  // (no gate: opening the G2 window when the P1a hop completes floods the SMs while the G1 CTAs are on their critical
  // copies / MMA -- measured 1-2 us slower)
  // FAST_V3_W2_WIN4 (experimental): window 4 instead of 2; FAST_V3_W2_OPEN (experimental): the whole rest of the stream is
  // issued once this CTA's P1b is published (G1's weights have landed by then: G1 W ready ~6.9 us vs P1b ~5.4-6.7 us)
  WStream ws{sm, p.wqb + long(c) * G2_TILES * KB2 * CHUNK, long(V3_CH), wbar, 0ull, 0, 0, nullptr,
             (p.fast & FAST_V3_PACE1) ? 1 : ((p.fast & FAST_V3_W2_WIN4) ? 4 : V3_PACE_WINDOW)};
  const int wchunk = tl * 2 + half;
  auto issue_weights = [&]() {
    if (tid == 0) {
      ws.init();
      if (pace)
        ws.step();
      else
        ws.all();
    }
  };
  // FAST_V3_L2PF: the wq_b slice goes HBM -> L2 after this CTA's P1a (G1's wq_a|wkv stream has the HBM to itself until
  // then) and L2 -> smem after its P1b, long before the q tiles arrive
  const bool l2pf = p.fast & FAST_V3_L2PF;
  if (!late_w && !l2pf) issue_weights();  // weights are immutable: before the PDL wait
  const unsigned sword = lane * 4 < KW_MAX ? ld_stream4(p.wqb_s + long(tile) * KB2 + half * KW_MAX + lane * 4) : 0u;
  p1_prefetch_fn(p, c, tid);  // the hc_attn_fn lines of this CTA's mix jobs -> L2
  pdl_wait();
  tick_publish(p, tick_t, tid);
  PROF_T0(1);
  // DS41H_PROLOGUE_LL: the q hop as LL words (written by the PUSH finishers, see g1_push_finish)
  const bool ll = MMA == MMA_HMMA && (p.ll & 1) && (p.fast & FAST_V3_PUSH) && !(p.fast & FAST_V3_ADEFER);
  const bool llp1 = MMA == MMA_HMMA && (p.ll & 2) && (p.fast & FAST_V3_HOP) && !(p.fast & (FAST_V3_EXT_P1A | FAST_V3_ADEFER));
  const unsigned epoch1 = (ll || llp1) ? ld_relaxed_gpu(p.pcf + C_EPOCH) + 1u : 0u;
  const unsigned llflag = ll ? epoch1 : 0u;
  p1_v3_g2(p, c, warp, lane, [&]() {
    if (l2pf) {
      if (tid == 0) {
        ws.init();
        ws.prefetch_all();
      }
    } else if (late_w) {
      issue_weights();  // FAST_V3_LATE_W2: after this CTA's P1a jobs, before the mixes
    } else if (pace && tid == 0) {
      ws.step();
    }
  }, llp1 ? epoch1 : 0u);
  if (l2pf) {
    if (tid == 0) ws.all();
  } else if (pace && tid == 0) {
    if (p.fast & FAST_V3_W2_OPEN)
      ws.all();
    else
      ws.step();
  }
  // FAST_V3_SK_G2: the Sinkhorn of token t runs on the last warp of G2 CTA G2_CTAS - 1 - t, a CTA without P1 jobs for every
  // m <= 8 (P1 jobs fill CTAs 0 .. 160 m / 8 - 1 for m <= 6, 0 .. 79 for m = 7, 8). It starts as soon as all mixes are
  // published (~P1b) instead of after a G1 CTA's tile work and normally ends before the q tiles arrive; meanwhile warp 0
  // (thread 0 drives the paced weight stream) waits for the q tiles. Without the bit G1 CTA V3_H0_G1 hosts it.
  if ((p.fast & FAST_V3_SK_G2) && !(p.fast & FAST_V3_SK_G1) && warp == WARPS - 1 && G2_CTAS - 1 - c < m) {
    warp_wait(p.pc + C_P1B, unsigned(G2_CTAS), lane, true, p.fast & FAST_V3_FENCE);
    if (lane == 0) PROF_AT(3);  // G2 Sinkhorn host: mixes published
    if (p.fast & FAST_V3_SKROW)  // shared-reciprocal row Sinkhorn: short enough to end inside the CTA's idle window
      sinkhorn_rows_rcp(p, G2_CTAS - 1 - c, lane);
    else
      sinkhorn_rows(p, G2_CTAS - 1 - c, lane);
    if (lane == 0) PROF_AT(4);  // G2 Sinkhorn host: Sinkhorn done
  }
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc2[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // DS41H_PROLOGUE_MMA2: second unit of warp w < 4
  const bool mma2 = MMA == MMA_HMMA && (p.fast & FAST_V3_FAC) && p.mma2;
  float* red;
  const float* rq = nullptr;  // wgmma: deferred q_norm rsqrt per token
  if constexpr (MMA == MMA_HMMA) {
    uint8_t* act = ra + V2_ACT;
    uint8_t* xe = ra + V2_XE;
    float* qsq = reinterpret_cast<float*>(ra + V2_QSQ);
    __nv_bfloat16* w = reinterpret_cast<__nv_bfloat16*>(ra + V2_NW);
    float* rr = reinterpret_cast<float*>(ra + V2_RS);
    constexpr int PITCH = act_pitch(QR), V8 = QR / 8;
    for (int i = tid; i < V8; i += THREADS) cp_async16(w + i * 8, p.q_norm + i * 8);
    cp_async_commit();
    if (llflag != 0u) {
      cta_wait_ws(p.pc + C_QHINT, Q_TILES, ws, pace && !l2pf, false, false, true);  // relaxed hint, no acquire
      PROF_T0(5);  // G2: q tiles published (hint)
      if (pace && tid == 0) ws.all();
      ll_stage_q(p, act, PITCH, qsq, m, tid, llflag);
      cp_async_wait_all();
      __syncthreads();
    } else {
      cta_wait_ws(p.pc + C_P3Q, Q_TILES, ws, pace && !l2pf, p.fast & FAST_V3_FENCE, p.fast & FAST_V3_ACQPOLL);
      PROF_T0(5);  // G2: q tiles published
      for (int v = tid; v < m * V8; v += THREADS) {
        const int tok = v / V8, c8 = v - tok * V8;
        cp_async16(act + tok * PITCH + c8 * 16, p.qkv + long(tok) * QKV + c8 * 8);
      }
      for (int v = tid; v < Q_TILES * MAXM / 4; v += THREADS) cp_async16(qsq + v * 4, p.qsq + v * 4);
      cp_async_commit();
      if (pace && tid == 0) ws.all();
      cp_async_wait_all();
      __syncthreads();
    }
    if (!(p.fast & FAST_V3_SK_G2)) PROF_T0(3);  // G2: qkv rows + q square sums in smem (slot 3: SK_G2 host otherwise)
    if (warp < m) {
      float s = 0.0f;
      for (int i = lane; i < Q_TILES; i += 32) s += qsq[i * MAXM + warp];
      s = warp_sum(s);
      if (lane == 0) rr[warp] = rsqrtf(__fdiv_rn(s, float(QR)) + p.qkv_eps);
    }
    __syncthreads();
    if (!(p.fast & FAST_V3_SK_G2)) PROF_T0(4);  // G2: q_norm rsqrt per token
    const bool write_qr = c == 0;
    auto g2x = [&](int tok, int c8, float (&x)[8]) {
      const float r = rr[tok];
      float wf[8];
      unpack8(*reinterpret_cast<const uint4*>(w + c8 * 8), wf);
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = bf16_round(__fmul_rn(__fmul_rn(x[i], r), wf[i]));
      if (write_qr) *reinterpret_cast<uint4*>(p.qr + long(tok) * QR + c8 * 8) = pack8_bf16(x);
    };
    if (p.cvt)  // 1280 vectors: 5 per thread in one batch (same values)
      convert_act_batched<(MAXM * QR / 8 + THREADS - 1) / THREADS>(act, PITCH, xe, m, QR, tid, g2x);
    else
      convert_act(act, PITCH, xe, m, QR, tid, g2x);
    __syncthreads();
    PROF_T0(6);  // G2: qr staged
    if ((p.fast & FAST_V3_FAC) && p.mma2) {
      // DS41H_PROLOGUE_MMA2: every warp builds its unit's factor table; warps 0-3 run units (tiles 2(w>>1), 2(w>>1)+1; half
      // w & 1) on one act fragment (unit u = tile * 2 + half, as the original warp -> unit map)
      float* fac = reinterpret_cast<float*>(ra + V2_QSQ);
      hmma_factors(fac + warp * KW_MAX * MAXM, xe, half * KW_MAX, sword, lane);
      __syncthreads();
      if (warp < WARPS / 2) {
        const int h2 = warp & 1, ta = (warp >> 1) * 2, u0 = ta * 2 + h2, u1 = (ta + 1) * 2 + h2;
        mbar_wait_hc(wbar + u0, 0, 3);
        mbar_wait_hc(wbar + u1, 0, 3);
        PROF_T0(10);  // G2: weights in smem
        mma_smem_f2(sm + (ta * KB2 + h2 * KW_MAX) * CHUNK, sm + ((ta + 1) * KB2 + h2 * KW_MAX) * CHUNK, fac + u0 * KW_MAX * MAXM,
                    fac + u1 * KW_MAX * MAXM, act, PITCH, h2 * KW_MAX, lane, acc, acc2);
      }
    } else if (p.fast & FAST_V3_FAC) {
      // factors alias the q-norm statistics / weight slice (both consumed by convert_act, behind its barrier)
      float* fac = reinterpret_cast<float*>(ra + V2_QSQ) + warp * KW_MAX * MAXM;
      hmma_factors(fac, xe, half * KW_MAX, sword, lane);
      __syncwarp();
      mbar_wait_hc(wbar + wchunk, 0, 3);
      PROF_T0(10);  // G2: weights in smem
      mma_smem_f(sm + (tl * KB2 + half * KW_MAX) * CHUNK, fac, act, PITCH, half * KW_MAX, lane, acc);
    } else {
      mbar_wait_hc(wbar + wchunk, 0, 3);
      PROF_T0(10);  // G2: weights in smem
      mma_smem(sm + (tl * KB2 + half * KW_MAX) * CHUNK, sword, act, PITCH, xe, half * KW_MAX, lane, acc);
    }
    __syncthreads();  // red aliases the activation tile
    red = reinterpret_cast<float*>(ra + V2_ACT);
  } else {
    uint8_t* bt = ra + V2W_BT;
    float* sx = reinterpret_cast<float*>(ra + V2W_SX);
    float* fac = reinterpret_cast<float*>(ra + V2W_FAC) + warp * KW_MAX * MAXM;
    float* rqs = reinterpret_cast<float*>(ra + V2W_RS);
    // B tile built by the G1 pair finishers (u = qkv * q_norm_w, act_quant); the q_norm rsqrt goes to the epilogue
    cta_wait_ws(p.pc + C_P3Q, Q_TILES, ws, pace && !l2pf, p.fast & FAST_V3_FENCE, p.fast & FAST_V3_ACQPOLL);
    PROF_T0(5);  // G2: q blocks published
    for (int i = tid; i < KB2 * WG_KB_PITCH / 16; i += THREADS) cp_async16(bt + 16 * i, p.qrb + 16 * i);
    for (int i = tid; i < KB2 * MAXM / 4; i += THREADS) cp_async16(sx + 4 * i, p.qrs + 4 * i);
    cp_async_commit();
    if (pace && tid == 0) ws.all();
    if (warp < m) {  // q_norm statistics of token `warp` (qr_build's order)
      float sq = 0.0f;
      for (int i = lane; i < Q_TILES; i += 32) sq += __ldcg(p.qsq + i * MAXM + warp);
      sq = warp_sum(sq);
      if (lane == 0) rqs[warp] = rsqrtf(__fdiv_rn(sq, float(QR)) + p.qkv_eps);
    }
    cp_async_wait_all();
    fence_proxy_async_smem();
    __syncthreads();
    PROF_T0(6);  // G2: qr tile in smem
    wg_factors<KW_MAX>(fac, sx + half * KW_MAX * MAXM, sword, KW_MAX, lane);
    __syncwarp();
    mbar_wait_hc(wbar + wchunk, 0, 3);
    PROF_T0(10);  // G2: weights in smem
    wg_mma_smem<KW_MAX>(sm + (tl * KB2 + half * KW_MAX) * CHUNK, bt + half * KW_MAX * WG_KB_PITCH, fac, lane, acc);
    red = reinterpret_cast<float*>(ra + V2W_RED);
    rq = rqs;
  }
  PROF_T0(7);  // G2: warp 0 MMA done
  if (!mma2) {
    store_frag_tile(red + warp * 16 * MAXM, acc, lane);
  } else if (warp < WARPS / 2) {  // units u0 / u1 of warp w at the original warp's slots
    const int h2 = warp & 1, ta = (warp >> 1) * 2;
    store_frag_tile(red + (ta * 2 + h2) * 16 * MAXM, acc, lane);
    store_frag_tile(red + ((ta + 1) * 2 + h2) * 16 * MAXM, acc2, lane);
  }
  __syncthreads();
  PROF_T0(8);  // G2: k-halves in smem
  if (MMA == MMA_WGMMA && tid < 10 * m) {
    // the `qr` output (stock q_norm: bf16((x * rsqrt) * w)): CTA c writes columns [10c, 10c + 10)
    static_assert(G2_CTAS * 10 == QR);
    const int tk = tid / 10, col = c * 10 + tid - tk * 10;
    const float x = bf2f(__ldcg(p.qkv + long(tk) * QKV + col)), wv = bf2f(__ldg(p.q_norm + col));
    p.qr[long(tk) * QR + col] = __float2bfloat16(__fmul_rn(__fmul_rn(x, rq[tk]), wv));
  }
  // Epilogue (role_g2): thread -> (token, adjacent row pair); halves summed in order 0 + 1.
  const int tok = tid >> 5, q32 = tid & 31, tl2 = q32 >> 3, row = 2 * (q32 & 7);
  if (tok >= m) return;
  const float* r0 = red + (MMA == MMA_WGMMA ? tl2 : 2 * tl2) * 16 * MAXM;
  const float* r1 = red + (MMA == MMA_WGMMA ? 4 + tl2 : 2 * tl2 + 1) * 16 * MAXM;
  float e0 = r0[row * MAXM + tok] + r1[row * MAXM + tok];
  float e1 = r0[(row + 1) * MAXM + tok] + r1[(row + 1) * MAXM + tok];
  if (MMA == MMA_WGMMA) {
    e0 = __fmul_rn(e0, rq[tok]);
    e1 = __fmul_rn(e1, rq[tok]);
  }
  e0 = bf16_round(e0);  // stock wq_b output is bf16
  e1 = bf16_round(e1);
  const int orow = (c * G2_TILES + tl2) * 16 + row;
  if (p.q_pre != nullptr) *reinterpret_cast<unsigned*>(p.q_pre + long(tok) * WQB + orow) = pack_bf16(e0, e1);
  const int d = orow & (HDIM - 1);
  if (d >= NOPE) {
    const float* cs = p.cos_sin + p.positions[tok] * ROPE;
    const int i = (d - NOPE) >> 1;
    const float cv = __ldg(cs + i), sv = __ldg(cs + ROPE / 2 + i);
    const float xe2 = e0, xo = e1;
    e0 = __fmaf_rn(xe2, cv, -__fmul_rn(xo, sv));
    e1 = __fmaf_rn(xe2, sv, __fmul_rn(xo, cv));
  }
  *reinterpret_cast<unsigned*>(p.q + long(tok) * WQB + orow) = pack_bf16(e0, e1);
}

// ================================================================ indexer projections (Params::idx)
// On index-source layers the stock glue runs two more GEMMs on the prologue's outputs before the indexer: weights_proj(a)
// (bf16 [32, 5120], unquantized ReplicatedLinear = cuBLAS bf16 GEMM, fp32 accumulation, bf16 out) and indexer.wq_b(qr)
// (FP8 [4096, 1280] through the skinny drop-in), ~13 us of serial kernels at 350 W. With p.idx the prologue computes both:
//  * idx_q: G1 CTAs 0..63 (the q-tile groups 0..7; their own tiles are finished by then) run a G2-like phase over the 256
//    indexer tiles (4 tiles per CTA): the same qr staging (q_norm with the exact rsqrt, f16 power-of-two staging) and the
//    K PARTITION OF THE SKINNY DROP-IN that the reference path runs on qr (skinny.cu choose_kpc(4096, 1280) = 4: k-parts of
//    12 / 12 / 12 / 4 k32 blocks, each accumulated from 0 in block order with the same HMMA + post-scale FMA, then
//    v = 0 + p0 + p1 + p2 + p3 in fp32, bf16) -> idx_q is BIT-IDENTICAL to indexer.wq_b(qr) of the reference config; no RoPE;
//  * idx_w: the 8 k-part CTAs of G1 group IDXW_GROUP (kv tiles: not on the q path; no Sinkhorn / KV hosts in it) multiply
//    their staged 640-column slice of `a` (exactly the bf16 values: f16 * 2^(eb-127)) by the bf16 weights_proj rows on bf16
//    HMMA (exact products, fp32 accumulation), write [k-part][32][MAXM] partials to scratch, and after a cluster barrier rank
//    0 sums them in k-part order and rounds to bf16 (the stock GEMM up to fp32 summation order).
constexpr int IDX_N = 4096, IDX_TILES = IDX_N / 16, IDX_CTAS = IDX_TILES / G2_TILES, IDXW_N = 32, IDXW_GROUP = 11;
static_assert(IDX_CTAS == 64 && IDX_CTAS <= (Q_TILES / WARPS) * KSPLIT1, "idx_q CTAs are q-tile groups");
static_assert(IDXW_GROUP * KSPLIT1 >= IDX_CTAS && (IDXW_GROUP + 1) * KSPLIT1 <= V3_H0_G1 - MAXM && IDXW_GROUP >= Q_TILES / WARPS,
              "the weights_proj group is a kv-tile group without Sinkhorn / KV hosts (CTAs >= 101 host them)");
static_assert(IDXW_N == 2 * 16 && IDXW_N * act_pitch(KPART1) <= V3_WBYTES, "weights_proj slice in the weight area");

// idx_q k-partition = the skinny drop-in's for [4096, 1280], m <= 8 (ds41h/skinny.py checks choose_split(4096, 1280) == 4).
constexpr int IDXQ_KPC = 4, IDXQ_KW = 12;
static_assert(IDXQ_KW % 4 == 0 && IDXQ_KPC * IDXQ_KW >= KB2 && (IDXQ_KPC - 1) * IDXQ_KW < KB2 && 2 * IDXQ_KW > KW_MAX,
              "idx_q parts: KW = round_up(ceil(40 / 4), 4) = 12; warp parts {0, 1} span both 20-block chunks of its tile");
// mma_smem for `cnt` (<= IDXQ_KW) blocks of one k-part: same HMMAs, post-scales and FMA order as skinny.cu mma_frag (whose
// sword holds the scale bytes of blocks kba0 + 4 lane .. + 3, exactly as here).
__device__ __forceinline__ void mma_smem_cnt(const uint8_t* wsm, unsigned sword, const uint8_t* act, int pitch, const uint8_t* xe, int kba0,
                                             int cnt, int lane, float (&acc)[4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* arow = act + g * pitch + t * 16 + kba0 * 64;
  const uint8_t* xrow = xe + kba0 * MAXM + 2 * t;
  const uint8_t* wl = wsm + lane * 16;
#pragma unroll 4
  for (int i = 0; i < IDXQ_KW; ++i) {
    if (i < cnt) {
      const uint4 b = *reinterpret_cast<const uint4*>(arow + i * 64);
      const uint4 v = lds128(wl + i * CHUNK);
      float d[4];
      hmma_f16_zero(d, f16x2_lo(v.x), f16x2_lo(v.z), f16x2_hi(v.x), f16x2_hi(v.z), b.x, b.y);
      hmma_f16(d, f16x2_lo(v.y), f16x2_lo(v.w), f16x2_hi(v.y), f16x2_hi(v.w), b.z, b.w);
      const unsigned sw = (__shfl_sync(FULL, sword, i >> 2) >> (8 * (i & 3))) & 0xffu;
      const unsigned ex = *reinterpret_cast<const unsigned short*>(xrow + i * MAXM);
      const float f0 = post_scale(sw, ex & 0xffu), f1 = post_scale(sw, ex >> 8);
      acc[0] = fmaf(d[0], f0, acc[0]);
      acc[1] = fmaf(d[1], f1, acc[1]);
      acc[2] = fmaf(d[2], f0, acc[2]);
      acc[3] = fmaf(d[3], f1, acc[3]);
    }
  }
}

__device__ __forceinline__ void v3_idx_q(const Params& p, uint8_t* sm, int c, int tid, int warp, int lane) {
  uint8_t* ra = sm + V3_A;
  unsigned long long* wbar = reinterpret_cast<unsigned long long*>(sm + V3_BAR);  // G1's chunk mbarriers: phase 0 done
  // warp w: tile w >> 1, k-parts pp, pp + 1 (pp = 0 for even warps: blocks 0..23, 2 for odd warps: blocks 24..39)
  const int tl = warp >> 1, pp = (warp & 1) * 2, tile = c * G2_TILES + tl, m = p.m;
  // every peer of this cluster is past its G1 work (own-collapse mode reads the peers' slice sums through DSMEM; PUSH writes
  // the tile partials into the finisher's smem) and every warp of this CTA past its MMA: the role + weight areas are free
  cl_arrive_release();
  cl_wait_acquire();
  if (tid == 0) {  // the CTA's 80 KB of indexer weights (contiguous: 4 tiles x 40 blocks), one 10 KB tile-half per mbarrier
    const unsigned long long pol = evict_first_policy();
    for (int ch = 0; ch < V3_NCH; ++ch) {
      mbar_expect_tx(wbar + ch, V3_CH);
      bulk_g2s(sm + ch * V3_CH, p.iwq + long(c) * G2_TILES * KB2 * CHUNK + ch * V3_CH, V3_CH, wbar + ch, pol);
    }
  }
  if (p.topk_buf != nullptr && c < 4) {  // fold: stock's topk_indices_buffer[:m] = -1 (sparse_attn_indexer), after the PDL wait
    const int vpr = p.topk_w >> 2;
    for (int v = c * THREADS + tid; v < m * vpr; v += 4 * THREADS) {
      const int r = v / vpr, q = v - r * vpr;
      *reinterpret_cast<int4*>(p.topk_buf + r * p.topk_ld + q * 4) = make_int4(-1, -1, -1, -1);
    }
  }
  int kb0[2], cnt[2];
  unsigned sword[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    kb0[j] = (pp + j) * IDXQ_KW;
    cnt[j] = min(IDXQ_KW, KB2 - kb0[j]);
    sword[j] = lane * 4 < cnt[j] ? ld_stream4(p.iwq_s + long(tile) * KB2 + kb0[j] + lane * 4) : 0u;
  }
  uint8_t* act = ra + V2_ACT;
  uint8_t* xe = ra + V2_XE;
  float* qsq = reinterpret_cast<float*>(ra + V2_QSQ);
  __nv_bfloat16* w = reinterpret_cast<__nv_bfloat16*>(ra + V2_NW);
  float* rr = reinterpret_cast<float*>(ra + V2_RS);
  constexpr int PITCH = act_pitch(QR), V8 = QR / 8;
  for (int i = tid; i < V8; i += THREADS) cp_async16(w + i * 8, p.q_norm + i * 8);
  cp_async_commit();
  if (tid == 0) {
    HangCheck hc;
    unsigned v;
    while ((v = ld_relaxed_gpu(p.pc + C_P3Q)) < unsigned(Q_TILES)) hc.tick(5, v, Q_TILES);
    if (p.fast & FAST_V3_FENCE)
      fence_acq_rel_gpu();
    else
      (void)ld_acquire_gpu(p.pc + C_P3Q);
  }
  __syncthreads();
  for (int v = tid; v < m * V8; v += THREADS) {
    const int tok = v / V8, c8 = v - tok * V8;
    cp_async16(act + tok * PITCH + c8 * 16, p.qkv + long(tok) * QKV + c8 * 8);
  }
  for (int v = tid; v < Q_TILES * MAXM / 4; v += THREADS) cp_async16(qsq + v * 4, p.qsq + v * 4);
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
  if (warp < m) {  // q_norm statistics of token `warp` (v3_g2's order)
    float s = 0.0f;
    for (int i = lane; i < Q_TILES; i += 32) s += qsq[i * MAXM + warp];
    s = warp_sum(s);
    if (lane == 0) rr[warp] = rsqrtf(__fdiv_rn(s, float(QR)) + p.qkv_eps);
  }
  __syncthreads();
  convert_act(act, PITCH, xe, m, QR, tid, [&](int tok, int c8, float (&x)[8]) {
    const float r = rr[tok];
    float wf[8];
    unpack8(*reinterpret_cast<const uint4*>(w + c8 * 8), wf);
#pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = bf16_round(__fmul_rn(__fmul_rn(x[i], r), wf[i]));
  });
  __syncthreads();
  float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
  if (pp == 0) mbar_wait_hc(wbar + tl * 2, 1, 6);  // phase 1 of the chunk mbarriers (phase 0 = this CTA's G1 tile)
  mbar_wait_hc(wbar + tl * 2 + 1, 1, 6);
#pragma unroll
  for (int j = 0; j < 2; ++j) mma_smem_cnt(sm + (tl * KB2 + kb0[j]) * CHUNK, sword[j], act, PITCH, xe, kb0[j], cnt[j], lane, acc[j]);
  __syncthreads();  // red aliases the activation tile
  float* red = reinterpret_cast<float*>(ra + V2_ACT);  // [tile][k-part][16][MAXM]
  static_assert(G2_TILES * IDXQ_KPC * 16 * MAXM * 4 <= MAXM * act_pitch(QR), "idx_q partials alias the activation tile");
#pragma unroll
  for (int j = 0; j < 2; ++j) store_frag_tile(red + (tl * IDXQ_KPC + pp + j) * 16 * MAXM, acc[j], lane);
  __syncthreads();
  const int tok = tid >> 5, q32 = tid & 31, tl2 = q32 >> 3, row = 2 * (q32 & 7);
  const int col = (c * G2_TILES + tl2) * 16 + row;  // indexer q column: head col >> 7, dim col & 127
  const bool fold = p.idx_qf != nullptr;
  float* amx = reinterpret_cast<float*>(ra + V2_RS);  // per-token |q| max of this CTA's 64 columns (rr is consumed)
  float r0 = 0.0f, r1 = 0.0f;
  if (tok < m) {  // k-parts summed 0 + p0 + p1 + p2 + p3 (skinny_kernel's order), bf16 (the drop-in's output)
    const float* rp = red + tl2 * IDXQ_KPC * 16 * MAXM;
    float v0 = 0.0f, v1 = 0.0f;
#pragma unroll
    for (int q = 0; q < IDXQ_KPC; ++q) {
      v0 += rp[q * 16 * MAXM + row * MAXM + tok];
      v1 += rp[q * 16 * MAXM + (row + 1) * MAXM + tok];
    }
    const float e0 = bf16_round(v0), e1 = bf16_round(v1);
    *reinterpret_cast<unsigned*>(p.idx_q + long(tok) * IDX_N + col) = pack_bf16(e0, e1);
    if (fold) {
      // stock fused_indexer_q_rope_quant (deepseek_v4 fused_indexer_q_cutedsl.py IndexerQFp8Kernel): GPT-J interleaved RoPE on
      // dims [64, 128) of each head (pair (2i, 2i + 1) with cos / sin [pos, i] / [pos, 32 + i]), result rounded to bf16
      r0 = e0;
      r1 = e1;
      if ((col & 127) >= 64) {
        const int i = ((col & 127) - 64) >> 1;
        const float* cs = p.idx_cs + p.positions[tok] * ROPE;
        const float cv = __ldg(cs + i), sv = __ldg(cs + ROPE / 2 + i);
        r0 = bf16_round(__fmaf_rn(e0, cv, -__fmul_rn(e1, sv)));
        r1 = bf16_round(__fmaf_rn(e0, sv, __fmul_rn(e1, cv)));
      }
      const float am = warp_max(fmaxf(fabsf(r0), fabsf(r1)));  // warp = token
      if (q32 == 0) amx[tok] = am;
    }
  }
  if (fold) {
    // the head's other 64 columns live in the partner CTA c ^ 1 = cluster rank ^ 1 (all 8 CTAs of these clusters run idx_q)
    cl_arrive_release();
    cl_wait_acquire();
    if (tok < m) {
      const float amax = fmaxf(amx[tok], cl_ld(cl_map(amx + tok, cl_rank() ^ 1u)));
      // scale = 2^ceil(log2(max(amax, 1e-4) * f32(1/448))) by the CuTe kernel's bit trick; exact power-of-two (inverse) multiplies
      const unsigned e = ((__float_as_uint(fmaxf(amax, 1e-4f) * (1.0f / 448.0f)) + 0x7fffffu) >> 23) & 0xffu;
      const float inv = __uint_as_float((254u - e) << 23), fs = __uint_as_float(e << 23);
      *reinterpret_cast<unsigned short*>(p.idx_qf + long(tok) * IDX_N + col) = static_cast<unsigned short>(e4m3x2(r0 * inv, r1 * inv));
      if ((c & 1) == 0 && q32 == 0) {  // one thread per (token, head): weights_out = (w * wscale) * q_scale (exact power-of-two)
        HangCheck hc;
        unsigned v;
        while ((v = ld_acquire_gpu(p.pc + C_IDXW)) < 1u) hc.tick(8, v, 1u);
        const int head = col >> 7;
        p.idx_wf[tok * IDXW_N + head] = __fmul_rn(__fmul_rn(bf2f(p.idx_w[tok * IDXW_N + head]), p.idx_wscale), fs);
      }
    }
    cl_arrive_relaxed();  // the partner has read our amx before either CTA moves on / exits
    cl_wait();
  }
}

__device__ __forceinline__ void v3_idx_w(const Params& p, uint8_t* sm, int kp, int tid, int warp, int lane) {
  constexpr int PITCH = act_pitch(KPART1), V8 = KPART1 / 8;
  uint8_t* wsm = sm;  // the weight area (this CTA's G1 MMA is done: every warp passed the barrier in the caller)
  __syncthreads();
  for (int i = tid; i < IDXW_N * V8; i += THREADS) {  // rows 0..31, cols [640 kp, +640) -> pitch PITCH (bank-conflict free)
    const int r = i / V8, v = i - r * V8;
    cp_async16(wsm + r * PITCH + v * 16, p.iwp + long(r) * HID + kp * KPART1 + v * 8);
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
  if (warp < 2) {  // warp w: weights_proj rows 16w .. 16w + 15
    const int g = lane >> 2, t = lane & 3;
    const uint8_t* act = sm + V3_A + V1_ACT;  // staged f16 slice of `a`: value = f16 * 2^(eb - 127), eb in V1_XE
    const uint8_t* xe = sm + V3_A + V1_XE;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
    for (int i = 0; i < KW_MAX; ++i) {
      // A = bf16 rows (16w + g, + 8) at the physical columns 32 i + 8 t .. + 7, B = token g's values at the same columns: the
      // HMMA k permutation of the packed FP8 tiles (sm90.cuh fragment map), consumed identically on both operands
      const uint4 w0 = lds128(wsm + (16 * warp + g) * PITCH + i * 64 + t * 16);
      const uint4 w1 = lds128(wsm + (16 * warp + g + 8) * PITCH + i * 64 + t * 16);
      const uint4 bv = *reinterpret_cast<const uint4*>(act + g * PITCH + i * 64 + t * 16);
      const float sc = pow2(int(xe[i * MAXM + g]) - 127);  // exact power of two; bf16 a = f16 * sc
      const unsigned bw[4] = {bv.x, bv.y, bv.z, bv.w};
      unsigned bb[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&bw[j]));
        bb[j] = pack_bf16(f.x * sc, f.y * sc);  // exact (the value is bf16)
      }
      hmma_bf16(acc, w0.x, w1.x, w0.y, w1.y, bb[0], bb[1]);
      hmma_bf16(acc, w0.z, w1.z, w0.w, w1.w, bb[2], bb[3]);
    }
    float* part = p.iwpart + long(kp) * IDXW_N * MAXM;
    *reinterpret_cast<float2*>(part + (16 * warp + g) * MAXM + 2 * t) = make_float2(acc[0], acc[1]);
    *reinterpret_cast<float2*>(part + (16 * warp + g + 8) * MAXM + 2 * t) = make_float2(acc[2], acc[3]);
  }
  cl_arrive_release();  // partials (global) -> rank 0
  cl_wait_acquire();
  if (kp == 0) {
    const int r = tid >> 3, tok = tid & 7;  // 32 rows x 8 tokens
    if (tok < p.m) {
      float s = 0.0f;
#pragma unroll
      for (int k = 0; k < KSPLIT1; ++k) s += __ldcg(p.iwpart + (long(k) * IDXW_N + r) * MAXM + tok);
      p.idx_w[long(tok) * IDXW_N + r] = __float2bfloat16(s);
    }
    if (p.idx_qf != nullptr) cta_signal(p.pc + C_IDXW, true);  // fold: idx_w is complete (CTA barrier + release)
  }
}

// FB >= 0: the fast bits are the compile-time constant FB (the launcher's effective default set, see FAST_V3_FIXED):
// a local Params copy with q.fast = FB lets the compiler drop every branch of the other modes (smaller hot code after
// the decoder layer kernel flushed L2, no runtime bit tests). FB = -1: runtime p.fast (any combination).
template <int MMA, int FB = -1, int VAR = 0>
__global__ void __launch_bounds__(THREADS, 2) prologue_kernel_v3(const Params p_in) {
  Params p = p_in;
  p.mma2 = (VAR & 1) ? 1 : 0;  // DS41H_PROLOGUE_MMA2 variant only in its own instantiation (keeps the default kernel spill-free)
  if constexpr (FB >= 0) {
    p.fast = FB;
    p.ll = 0;     // experimental runtime switches stay out of the specialized kernels (code size: the runtime kernel grew
    p.probe = 0;  // from 17.5k to 23k SASS instructions with the opt-in experiments; FIXED = 8.2k measured -1.7 us end-to-end)
    p.cvt = 0;    // (DS41H_PROLOGUE_CVT lost: +0.55 us)
  }
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, b = v3_logical_block(blockIdx.x, p.fast);
  PROF_T0(0);
  PROF_CLK(13);
  PROF_SMID();
  pdl_launch_dependents();
  // DS41H_PROLOGUE_TICKET: take the ticket now (its round trip overlaps the pre-PDL weight issue; consumed in tick_publish)
  unsigned long long tick_t = 0ull;
  if (p.tick && tid == 0) tick_t = atomicAdd(reinterpret_cast<unsigned long long*>(p.pcf + C_TICKET), 1ull);
  if (b < G1_CTAS) {
    v3_g1<MMA>(p, sm, b, tid, warp, lane, tick_t);
    // FAST_V3_HOST_SPREAD: the per-token Sinkhorn / KV hosts spread over the 4 kv-tile clusters (groups 10..13: their
    // finishers do not feed G2) instead of packed into the last one, so they do not keep one GPC busy (a 16-CTA attention
    // cluster needs 16 free SMs of one GPC).
    const bool hspread = p.fast & FAST_V3_HOST_SPREAD;
    constexpr int KVG0 = Q_TILES / WARPS;  // first kv-tile group (10)
    static_assert(G1_GROUPS - KVG0 == 4 && MAXM <= 8, "host spread: 4 kv clusters x 2 slots per role");
    const int grp_b = b / KSPLIT1, kp_b = b % KSPLIT1;
    if (p.fast & FAST_V3_SK_G1) {
      // token t's Sinkhorn on warp 1 of G1 CTA V3_H0_G1 - 1 - t (kv-tile groups, whose finishers do not feed G2): with
      // FAST_V3_PUSH warp 1 is free right after its push, so no CTA waits for the Sinkhorn (G2 hosts delayed their wq_b)
      const int t = hspread ? (grp_b >= KVG0 && kp_b < 2 ? (grp_b - KVG0) + 4 * kp_b : -1) : V3_H0_G1 - 1 - b;
      if (warp == 1 && t >= 0 && t < p.m) {
        warp_wait(p.pc + C_P1B, unsigned(G2_CTAS), lane, true, p.fast & FAST_V3_FENCE);
        if (lane == 0) PROF_AT(11);  // G1 Sinkhorn host: mixes published (start)
        if (p.fast & FAST_V3_SKROW)
          sinkhorn_rows_rcp(p, t, lane);
        else if (p.fast & FAST_V3_SKREG)
          sinkhorn_regs(p, t, lane);
        else
          sinkhorn_job(p, t, lane);
        if (lane == 0) PROF_AT(7);  // G1 Sinkhorn host: done
      }
    } else if (b == V3_H0_G1 && !(p.fast & FAST_V3_SK_G2)) {
      role_h(p, sm, 0, warp, lane);  // Sinkhorn (smem of the G1 role is free)
    }
    if (p.fast & FAST_V3_KV_SPREAD) {  // token t's KV insert on warp 2 of G1 CTA V3_H1_G1 - t (one token per SM)
      const int t = hspread ? (grp_b >= KVG0 && kp_b >= 2 && kp_b < 4 ? (grp_b - KVG0) + 4 * (kp_b - 2) : -1) : V3_H1_G1 - b;
      if (warp == 2 && t >= 0 && t < p.m) {
        warp_wait(p.pc + C_P3K, unsigned(KV_TILES), lane, true, p.fast & FAST_V3_FENCE);
        kv_job(p, t, lane);
        __syncwarp();
        if (lane == 0) aflag_signal(p.pcf, p.aflag);
        if (lane == 0) PROF_AT(8);  // G1 KV host: insert done
      }
    } else if (b == V3_H1_G1) {
      role_h(p, sm, 1, warp, lane);  // KV insert (waits for the kv tiles)
    }
    if constexpr (MMA == MMA_HMMA) {
      if (p.idx) {  // indexer projections (index-source layers): G1 CTAs after their own work
        if (b < IDX_CTAS) {
          __syncthreads();
          v3_idx_q(p, sm, b, tid, warp, lane);
        } else if (grp_b == IDXW_GROUP && p.idx_w != nullptr) {
          v3_idx_w(p, sm, kp_b, tid, warp, lane);
        }
      }
    }
  } else {
    v3_g2<MMA>(p, sm, b - G1_CTAS, tid, warp, lane, tick_t);
  }
  __syncthreads();
  if (b >= G1_CTAS && tid == 0) aflag_signal(p.pcf, p.aflag);  // this G2 CTA's q rows are stored (barrier above)
  PROF_T0(15);
  PROF_CLK(14);
  if (p.probe & 8) return;  // bench-only probe: no end-of-kernel arrival / reset (the bench zeroes the counters between launches)
  if (p.tick) return;       // ticket mode: no arrival, no reset (the next launch's first ticket holder zeroes this block)
  if (tid == 0 && atom_add_acq_rel(p.pc + C_FIN, 1u) == gridDim.x - 1) {
    for (int i = 0; i < NCOUNTERS; ++i) p.pc[i] = 0;  // light-sync reset (see prologue_kernel)
    p.pc[C_EPOCH] += 1u;  // DS41H_PROLOGUE_LL: the next launch's flag (published by kernel completion)
  }
}

template <int MMA>
__global__ void __launch_bounds__(THREADS, 2) prologue_kernel(const Params p) {
  extern __shared__ __align__(16) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, b = blockIdx.x;
  PROF_T0(0);
  PROF_CLK(13);
  PROF_SMID();
  pdl_launch_dependents();
  if (b < G1_CTAS) {
    role_g1<MMA>(p, sm, b, tid, warp, lane);
    // FAST_CLUSTER grids have no H CTAs (clusters of 8 fit 240 CTAs): the last two G1 CTAs take the Sinkhorn and
    // the KV insert once their tile work is done (both wait for other CTAs' results anyway).
    if ((p.fast & FAST_CLUSTER) && b >= G1_CTAS - H_CTAS) {
      __syncthreads();  // smem of the G1 role (H0 uses SM_ACT) is free
      role_h(p, sm, b - (G1_CTAS - H_CTAS), warp, lane);
    }
  } else if (b < G1_CTAS + G2_CTAS) {
    role_g2<MMA>(p, sm, b - G1_CTAS, tid, warp, lane);
  } else {
    role_h(p, sm, b - G1_CTAS - G2_CTAS, warp, lane);
  }
  __syncthreads();
  if (b >= G1_CTAS && b < G1_CTAS + G2_CTAS && tid == 0) aflag_signal(p.pcf, p.aflag);  // q rows of this G2 CTA stored
  PROF_T0(15);
  PROF_CLK(14);
  if (tid == 0) {
    if (p.fast & FAST_LIGHT_SYNC) {
      // acq_rel: the last arrival observes (acquire) every CTA's arrival, which follows all of that CTA's waits;
      // the resets are then ordered after them, and kernel completion publishes them to the next launch.
      if (atom_add_acq_rel(p.pc + C_FIN, 1u) == gridDim.x - 1)
        for (int i = 0; i < NCOUNTERS; ++i) p.pc[i] = 0;
    } else {
      __threadfence();
      if (atomicAdd(p.pc + C_FIN, 1u) == gridDim.x - 1) {
        // Every CTA has passed all of its waits: reset the phase counters for the next launch. The
        // fence orders the resets after every CTA's (fenced) signals that this atomic observed.
        __threadfence();
        for (int i = 0; i < NCOUNTERS; ++i) p.pc[i] = 0;
        __threadfence();
      }
    }
  }
}

template <int MMA>
int resident_ctas_t() {
  static const int ctas = [] {
    opt_in_smem<prologue_kernel<MMA>>(smem_of<MMA>());
    int device = 0, sms = 0, per = 0;
    TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess && cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess);
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, prologue_kernel<MMA>, THREADS, smem_of<MMA>()) == cudaSuccess);
    return sms * per;
  }();
  return ctas;
}
int resident_ctas(int mma) { return mma == MMA_WGMMA ? resident_ctas_t<MMA_WGMMA>() : resident_ctas_t<MMA_HMMA>(); }

// FAST_CLUSTER: co-resident CTAs when launched in clusters of 8 (max active clusters x 8; 0 if refused).
template <int MMA>
int resident_ctas_cl_t() {
  static const int ctas = [] {
    opt_in_smem<prologue_kernel<MMA>>(smem_of<MMA>());
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = CLUSTER;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(CTAS_CL);
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = smem_of<MMA>();
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    int n = 0;
    if (cudaOccupancyMaxActiveClusters(&n, prologue_kernel<MMA>, &cfg) != cudaSuccess) {
      (void)cudaGetLastError();
      n = 0;
    }
    return n * CLUSTER;
  }();
  return ctas;
}
int resident_ctas_cl(int mma) { return mma == MMA_WGMMA ? resident_ctas_cl_t<MMA_WGMMA>() : resident_ctas_cl_t<MMA_HMMA>(); }

template <int MMA>
void launch_cluster(const Params& p) {
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = CLUSTER;
  attrs[1].val.clusterDim.y = 1;
  attrs[1].val.clusterDim.z = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(CTAS_CL);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = smem_of<MMA>();
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, prologue_kernel<MMA>, p);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    TORCH_CHECK(false, "ds41h prologue (cluster) launch failed: ", cudaGetErrorString(err));
  }
}

// The fast-bit set of the default (ds41h/prologue.py FAST_DEFAULT = 4083200) as the HMMA launcher leaves it (+ LIGHT_SYNC):
// launched through the specialized instantiation prologue_kernel_v3<MMA_HMMA, FAST_V3_FIXED>.
constexpr int FAST_V3_FIXED = FAST_V3 | FAST_V3_HOP | FAST_V3_PACE | FAST_V3_LATE_W2 | FAST_V3_FAC | FAST_V3_ILV | FAST_V3_PUSH |
                              FAST_V3_SK_G1 | FAST_V3_KV_SPREAD | FAST_LIGHT_SYNC;
static_assert(FAST_V3_FIXED == 4083232);

// FAST_V3: co-resident CTAs of prologue_kernel_v3 in clusters of 8 (0 if refused).
template <int MMA>
int resident_ctas_v3_t() {
  static const int ctas = [] {
    opt_in_smem<prologue_kernel_v3<MMA>>(V3_SMEM);
    if constexpr (MMA == MMA_HMMA) {
      opt_in_smem<prologue_kernel_v3<MMA, FAST_V3_FIXED>>(V3_SMEM);
      opt_in_smem<prologue_kernel_v3<MMA, FAST_V3_FIXED | FAST_V3_ACQPOLL>>(V3_SMEM);
      opt_in_smem<prologue_kernel_v3<MMA, FAST_V3_FIXED | FAST_V3_EXT_P1A>>(V3_SMEM);
    }
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = CLUSTER;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(CTAS_V3);
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = V3_SMEM;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    int n = 0;
    if (cudaOccupancyMaxActiveClusters(&n, prologue_kernel_v3<MMA>, &cfg) != cudaSuccess) {
      (void)cudaGetLastError();
      n = 0;
    }
    return n * CLUSTER;
  }();
  return ctas;
}
int resident_ctas_v3(int mma) { return mma == MMA_WGMMA ? resident_ctas_v3_t<MMA_WGMMA>() : resident_ctas_v3_t<MMA_HMMA>(); }

static int ll_env() {
  static const int v = [] {
    const char* e = std::getenv("DS41H_PROLOGUE_LL");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  return v;
}

static int mma2_env() {
  static const int v = [] {
    const char* e = std::getenv("DS41H_PROLOGUE_MMA2");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  return v;
}

static int cvt_env() {
  static const int v = [] {
    const char* e = std::getenv("DS41H_PROLOGUE_CVT");
    return e != nullptr ? std::atoi(e) : 0;  // opt-in
  }();
  return v;
}

static int tick_env() {
  static const int v = [] {
    const char* e = std::getenv("DS41H_PROLOGUE_TICKET");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  return v;
}
static bool g_last_tick[64] = {};  // per device: the last launch ran in ticket mode (block 0 may hold its counts)

static int probe_env() {
  static const int v = [] {
    const char* e = std::getenv("DS41H_PROLOGUE_PROBE");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  return v;
}

// DS41H_PROLOGUE_FIXED (default ON): the default bits run the compile-time specialized kernel
// (bit-identical: the same code with p.fast constant-folded; end-to-end: prologue 20.70 -> 19.04 us, layer period
// -2.4 us at 350 W, k=7). DS41H_PROLOGUE_FIXED=0 = the runtime-bits kernel for every mode.
static bool no_fixed_v3() {
  static const bool v = [] {
    const char* e = std::getenv("DS41H_PROLOGUE_FIXED");
    return e != nullptr && e[0] == '0';
  }();
  return v;
}

template <int MMA>
void launch_v3(const Params& p) {
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = CLUSTER;
  attrs[1].val.clusterDim.y = 1;
  attrs[1].val.clusterDim.z = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(CTAS_V3);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = V3_SMEM;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  cudaError_t err;
  if constexpr (MMA == MMA_HMMA) {
    // (DS41H_PROLOGUE_MMA2 = prologue_kernel_v3<MMA, FAST_V3_FIXED, 1>: bit-identical but no gain -> not
    // instantiated any more; the code path stays for reference)
    if (p.fast == FAST_V3_FIXED && !no_fixed_v3())
      err = cudaLaunchKernelEx(&cfg, prologue_kernel_v3<MMA, FAST_V3_FIXED>, p);
    else if (p.fast == (FAST_V3_FIXED | FAST_V3_ACQPOLL) && !no_fixed_v3())
      err = cudaLaunchKernelEx(&cfg, prologue_kernel_v3<MMA, FAST_V3_FIXED | FAST_V3_ACQPOLL>, p);
    else if (p.fast == (FAST_V3_FIXED | FAST_V3_EXT_P1A) && !no_fixed_v3())  // P1a done by the decoder (run_prologue p1a_done)
      err = cudaLaunchKernelEx(&cfg, prologue_kernel_v3<MMA, FAST_V3_FIXED | FAST_V3_EXT_P1A>, p);
    else
      err = cudaLaunchKernelEx(&cfg, prologue_kernel_v3<MMA>, p);
  } else {
    err = cudaLaunchKernelEx(&cfg, prologue_kernel_v3<MMA>, p);
  }
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    TORCH_CHECK(false, "ds41h prologue (v3) launch failed: ", cudaGetErrorString(err));
  }
}

template <class T>
const T* in_ptr(const torch::Tensor& t, at::ScalarType dt, long numel, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == dt && t.numel() >= numel, "prologue: bad ", name);
  TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0, "prologue: ", name, " must be 16-byte aligned");
  return data_as<const T>(t);
}
template <class T>
T* out_ptr(const torch::Tensor& t, at::ScalarType dt, long numel, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == dt && t.numel() == numel, "prologue: bad output ", name);
  TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0, "prologue: ", name, " must be 16-byte aligned");
  return data_as<T>(t);
}

}  // namespace ds41h_prologue

using namespace ds41h_prologue;

static unsigned long long* g_prof = nullptr;  // profiling build: stamp buffer of the next launches
// Profiling builds: route phase stamps into `buf` (uint64 [>= CTAS * PROF_SLOTS], zeroed by the caller), or
// stop with None. A no-op in normal builds (the kernel has no stamp code).
void set_prof(c10::optional<torch::Tensor> buf) {
  if (!buf.has_value()) {
    g_prof = nullptr;
    return;
  }
  TORCH_CHECK(buf->is_cuda() && buf->is_contiguous() && buf->scalar_type() == at::kLong && buf->numel() >= long(std::max(CTAS, CTAS_V3)) * PROF_SLOTS,
              "prologue: prof buffer must be int64 [>= max(CTAS, CTAS_V3) * 16]");
  g_prof = reinterpret_cast<unsigned long long*>(buf->data_ptr<int64_t>());
}

int64_t scratch_bytes() { return SCRATCH_BYTES; }
int64_t num_ctas() { return CTAS; }
// Co-resident CTA capacity of the instantiation for `mma` (both must reach CTAS).
int64_t max_resident_ctas(int64_t mma) { return resident_ctas(int(mma)); }

void prologue(torch::Tensor x, torch::Tensor resid, torch::Tensor post, torch::Tensor comb, torch::Tensor pre, torch::Tensor fn,
              torch::Tensor hc_scale, torch::Tensor hc_base, torch::Tensor attn_norm, torch::Tensor wqa, torch::Tensor wqa_s,
              torch::Tensor q_norm, torch::Tensor kv_norm, torch::Tensor wqb, torch::Tensor wqb_s, torch::Tensor positions,
              torch::Tensor cos_sin, c10::optional<torch::Tensor> swa, c10::optional<torch::Tensor> slots, int64_t block_size,
              torch::Tensor scratch, torch::Tensor resid_out, torch::Tensor post_out, torch::Tensor comb_out, torch::Tensor pre_out,
              torch::Tensor a, torch::Tensor qr, torch::Tensor kv, torch::Tensor q, c10::optional<torch::Tensor> q_pre, double rms_eps,
              double hc_eps, double norm_eps, double qkv_eps, double post_mult, int64_t iters, int64_t mma, int64_t fast, int64_t aflag,
              c10::optional<torch::Tensor> idx_wqb, c10::optional<torch::Tensor> idx_wqb_s, c10::optional<torch::Tensor> idx_wp,
              c10::optional<torch::Tensor> idx_q, c10::optional<torch::Tensor> idx_w, c10::optional<torch::Tensor> aux,
              c10::optional<torch::Tensor> idx_qf, c10::optional<torch::Tensor> idx_wf, c10::optional<torch::Tensor> idx_cs, double idx_wscale,
              c10::optional<torch::Tensor> topk_buf) {
  const long m = x.size(0);
  TORCH_CHECK(mma == MMA_HMMA || mma == MMA_WGMMA, "prologue: mma must be 0 (hmma) or 1 (wgmma)");
  TORCH_CHECK((fast & ~int64_t(FAST_EARLY | FAST_SINKHORN | FAST_PREFETCH | FAST_FN_SMEM | FAST_QR_LOCAL | FAST_LIGHT_SYNC | FAST_CLUSTER |
                                FAST_QR_DEFER | FAST_OWN_COLLAPSE | FAST_V3 | FAST_V3_LATE_W2 | FAST_V3_HOP |
                                FAST_V3_BITS)) == 0,
              "prologue: unknown fast bits ", fast);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == HID && m >= 1 && m <= MAXM, "prologue: x must be [m <= ", MAXM, ", ", HID, "]");
  TORCH_CHECK(resid.dim() == 3 && resid.size(0) == m && resid.size(1) == HC && resid.size(2) == HID, "prologue: residual must be [m, 4, 5120]");
  TORCH_CHECK(iters >= 1, "prologue: sinkhorn iters >= 1");
  const c10::cuda::CUDAGuard guard(x.device());
  // Every pointer the kernel dereferences must live on the launch device (a stray tensor on another
  // GPU would otherwise fault or read peer memory silently).
  for (const torch::Tensor* t : {&resid, &post, &comb, &pre, &fn, &hc_scale, &hc_base, &attn_norm, &wqa, &wqa_s, &q_norm, &kv_norm, &wqb,
                                 &wqb_s, &positions, &cos_sin, &scratch, &resid_out, &post_out, &comb_out, &pre_out, &a, &qr, &kv, &q})
    TORCH_CHECK(t->device() == x.device(), "prologue: all tensors must be on ", x.device());
  if (q_pre.has_value()) TORCH_CHECK(q_pre->device() == x.device(), "prologue: q_pre must be on ", x.device());
  Params p{};
  p.x = in_ptr<__nv_bfloat16>(x, at::kBFloat16, m * HID, "x");
  p.resid = in_ptr<__nv_bfloat16>(resid, at::kBFloat16, m * HCD, "residual");
  p.post = in_ptr<float>(post, at::kFloat, m * HC, "post_mix");
  p.comb = in_ptr<float>(comb, at::kFloat, m * HC * HC, "res_mix");
  p.pre = in_ptr<float>(pre, at::kFloat, m * HC, "pre_mix");
  TORCH_CHECK(post.numel() == m * HC && comb.numel() == m * HC * HC && pre.numel() == m * HC, "prologue: mix shapes");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == at::kLong && positions.is_contiguous() && positions.numel() >= m,
              "prologue: positions must be int64 [>= m]");
  p.positions = data_as<const long>(positions);
  TORCH_CHECK(fn.dim() == 2 && fn.size(0) == MIX && fn.size(1) == HCD, "prologue: hc_attn_fn must be [24, 20480]");
  p.fn = in_ptr<float>(fn, at::kFloat, long(MIX) * HCD, "hc_attn_fn");
  p.hc_scale = in_ptr<float>(hc_scale, at::kFloat, 3, "hc_attn_scale");
  p.hc_base = in_ptr<float>(hc_base, at::kFloat, MIX, "hc_attn_base");
  p.attn_norm = in_ptr<__nv_bfloat16>(attn_norm, at::kBFloat16, HID, "attn_norm");
  p.wqa = in_ptr<uint8_t>(wqa, at::kByte, long(QKV) * HID, "wqa_wkv");
  p.wqa_s = in_ptr<uint8_t>(wqa_s, at::kByte, long(QKV_TILES) * KB1, "wqa_wkv scales");
  TORCH_CHECK(wqa.numel() == long(QKV) * HID && wqa_s.numel() == long(QKV_TILES) * KB1, "prologue: wqa_wkv packing");
  p.q_norm = in_ptr<__nv_bfloat16>(q_norm, at::kBFloat16, QR, "q_norm");
  p.kv_norm = in_ptr<__nv_bfloat16>(kv_norm, at::kBFloat16, KVW, "kv_norm");
  p.wqb = in_ptr<uint8_t>(wqb, at::kByte, long(WQB) * QR, "wq_b");
  p.wqb_s = in_ptr<uint8_t>(wqb_s, at::kByte, long(WQB_TILES) * KB2, "wq_b scales");
  TORCH_CHECK(wqb.numel() == long(WQB) * QR && wqb_s.numel() == long(WQB_TILES) * KB2, "prologue: wq_b packing");
  TORCH_CHECK(cos_sin.dim() == 2 && cos_sin.size(1) == ROPE, "prologue: cos_sin_cache must be [max_pos, 64]");
  p.cos_sin = in_ptr<float>(cos_sin, at::kFloat, ROPE, "cos_sin_cache");
  if (swa.has_value()) {
    TORCH_CHECK(slots.has_value(), "prologue: swa cache without slot mapping");
    TORCH_CHECK(swa->is_cuda() && swa->device() == x.device() && swa->scalar_type() == at::kByte && swa->dim() == 2 && swa->stride(1) == 1,
                "prologue: swa cache must be a uint8 [blocks, page_bytes] view");
    TORCH_CHECK(block_size > 0 && swa->size(1) >= block_size * (V4_DATA + V4_SCALES) && swa->stride(0) % 16 == 0 &&
                    (reinterpret_cast<uintptr_t>(swa->data_ptr()) & 15) == 0,
                "prologue: swa page too small / misaligned for block size ", block_size);
    TORCH_CHECK(slots->is_cuda() && slots->device() == x.device() && slots->scalar_type() == at::kLong && slots->is_contiguous(),
                "prologue: slot mapping must be contiguous int64");
    p.swa = data_as<uint8_t>(*swa);
    p.swa_stride = swa->stride(0);
    p.swa_block = int(block_size);
    p.slots = data_as<const long>(*slots);
    p.n_insert = int(std::min<long>(m, slots->numel()));  // stock: slot_mapping.size(0) rows are inserted
  }
  TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type() == at::kByte && scratch.numel() >= SCRATCH_BYTES &&
                  (reinterpret_cast<uintptr_t>(scratch.data_ptr()) & 255) == 0,
              "prologue: scratch must be a 256-byte aligned uint8 buffer of >= ", SCRATCH_BYTES, " bytes");
  uint8_t* s = data_as<uint8_t>(scratch);
  p.pc = reinterpret_cast<unsigned*>(s + OFF_PC);
  p.pcf = p.pc;
  p.tilecnt = reinterpret_cast<unsigned*>(s + OFF_TILECNT);
  p.mixp = reinterpret_cast<float*>(s + OFF_MIXP);
  p.xsq = reinterpret_cast<float*>(s + OFF_XSQ);
  p.xcb = reinterpret_cast<__nv_bfloat16*>(s + OFF_XCB);
  p.qkp = reinterpret_cast<float*>(s + OFF_QKP);
  p.qkv = reinterpret_cast<__nv_bfloat16*>(s + OFF_QKV);
  p.qsq = reinterpret_cast<float*>(s + OFF_QSQ);
  p.qll = reinterpret_cast<unsigned long long*>(s + OFF_QLL);
  p.qsqll = reinterpret_cast<unsigned long long*>(s + OFF_QSQLL);
  p.xll = reinterpret_cast<unsigned long long*>(s + OFF_XLL);
  p.xsqll = reinterpret_cast<unsigned long long*>(s + OFF_XSQLL);
  p.qrb = s + OFF_QRB;
  p.qrs = reinterpret_cast<float*>(s + OFF_QRS);
  p.qab = s + OFF_QAB;
  p.qas = reinterpret_cast<float*>(s + OFF_QAS);
  p.resid_out = out_ptr<__nv_bfloat16>(resid_out, at::kBFloat16, m * HCD, "residual");
  p.post_out = out_ptr<float>(post_out, at::kFloat, m * HC, "post_mix");
  p.comb_out = out_ptr<float>(comb_out, at::kFloat, m * HC * HC, "res_mix");
  p.pre_out = out_ptr<float>(pre_out, at::kFloat, m * HC, "attn_pre");
  p.a = out_ptr<__nv_bfloat16>(a, at::kBFloat16, m * HID, "a");
  p.qr = out_ptr<__nv_bfloat16>(qr, at::kBFloat16, m * QR, "qr");
  p.kv = out_ptr<__nv_bfloat16>(kv, at::kBFloat16, m * KVW, "kv");
  p.q = out_ptr<__nv_bfloat16>(q, at::kBFloat16, m * WQB, "q");
  if (q_pre.has_value()) p.q_pre = out_ptr<__nv_bfloat16>(*q_pre, at::kBFloat16, m * WQB, "q_pre");
  p.m = int(m);
  p.tpj = SLICES * m <= P1_WORKERS ? 1 : 2;
  p.iters = int(iters);
  p.fast = int(fast);
  p.probe = probe_env();
  p.cvt = cvt_env();
  p.mma2 = mma2_env();
  p.ll = ll_env();
  p.aflag = aflag != 0 ? 1 : 0;
  p.rms_eps = float(rms_eps);
  p.hc_eps = float(hc_eps);
  p.norm_eps = float(norm_eps);
  p.qkv_eps = float(qkv_eps);
  p.post_mult = float(post_mult);
  p.prof = g_prof;
  if (aux.has_value()) {
    TORCH_CHECK(aux->device() == x.device(), "prologue: aux must be on ", x.device());
    p.aux = out_ptr<__nv_bfloat16>(*aux, at::kBFloat16, m * HID, "aux");
  }
  if (idx_q.has_value()) {  // indexer projections (index-source layers): HMMA + the v3 schedule only
    // idx_w / idx_wp optional: without them only idx_q runs (bit-identical to the reference path's drop-in); the
    // weights_proj GEMM stays on stock cuBLAS
    TORCH_CHECK(idx_wqb.has_value() && idx_wqb_s.has_value() && idx_wp.has_value() == idx_w.has_value(), "prologue: idx needs all tensors");
    TORCH_CHECK(mma == MMA_HMMA && (fast & FAST_V3) && resident_ctas_v3(int(mma)) >= CTAS_V3, "prologue: idx needs hmma + the v3 schedule");
    for (const torch::Tensor* t : {&*idx_wqb, &*idx_wqb_s, &*idx_q})
      TORCH_CHECK(t->device() == x.device(), "prologue: idx tensors must be on ", x.device());
    p.iwq = in_ptr<uint8_t>(*idx_wqb, at::kByte, long(IDX_N) * QR, "indexer wq_b");
    p.iwq_s = in_ptr<uint8_t>(*idx_wqb_s, at::kByte, long(IDX_TILES) * KB2, "indexer wq_b scales");
    TORCH_CHECK(idx_wqb->numel() == long(IDX_N) * QR && idx_wqb_s->numel() == long(IDX_TILES) * KB2, "prologue: indexer wq_b packing");
    p.idx_q = out_ptr<__nv_bfloat16>(*idx_q, at::kBFloat16, m * IDX_N, "idx_q");
    if (idx_w.has_value()) {
      TORCH_CHECK(idx_wp->device() == x.device() && idx_w->device() == x.device(), "prologue: idx tensors must be on ", x.device());
      TORCH_CHECK(idx_wp->dim() == 2 && idx_wp->size(0) == IDXW_N && idx_wp->size(1) == HID, "prologue: weights_proj must be [32, 5120]");
      p.iwp = in_ptr<__nv_bfloat16>(*idx_wp, at::kBFloat16, long(IDXW_N) * HID, "weights_proj");
      p.idx_w = out_ptr<__nv_bfloat16>(*idx_w, at::kBFloat16, m * IDXW_N, "idx_w");
    }
    p.iwpart = reinterpret_cast<float*>(s + OFF_IWP);
    p.idx = 1;
    if (idx_qf.has_value()) {  // DS41H_PROLOGUE_INDEXER_FOLD (needs the prologue's idx_w for weights_out)
      TORCH_CHECK(idx_w.has_value() && idx_wf.has_value() && idx_cs.has_value(), "prologue: the indexer fold needs idx_w, idx_wf and idx_cs");
      TORCH_CHECK(idx_qf->is_cuda() && idx_qf->is_contiguous() && idx_qf->element_size() == 1 && idx_qf->numel() == m * IDX_N &&
                      idx_qf->device() == x.device(), "prologue: idx_qf must be a contiguous 1-byte [m, 32, 128] tensor");
      p.idx_qf = reinterpret_cast<uint8_t*>(idx_qf->data_ptr());
      p.idx_wf = out_ptr<float>(*idx_wf, at::kFloat, m * IDXW_N, "idx_wf");
      TORCH_CHECK(idx_cs->dim() == 2 && idx_cs->size(1) == ROPE, "prologue: indexer cos_sin must be [max_pos, 64] f32");
      p.idx_cs = in_ptr<float>(*idx_cs, at::kFloat, idx_cs->numel(), "indexer cos_sin");
      p.idx_wscale = float(idx_wscale);
      if (topk_buf.has_value()) {
        const auto& t = *topk_buf;
        TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.scalar_type() == at::kInt && t.dim() == 2 && t.stride(1) == 1 && t.size(0) >= m &&
                        t.size(1) % 4 == 0 && t.stride(0) % 4 == 0 && (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0,
                    "prologue: topk buffer must be int32 [>= m, 4k] with 16-byte aligned rows");
        p.topk_buf = t.data_ptr<int>();
        p.topk_ld = t.stride(0);
        p.topk_w = int(t.size(1));
      }
    }
  }
  // FAST_V3 (own kernel, see prologue_kernel_v3): needs the 240-CTA cluster grid to fit (checked once per instantiation
  // on the idle device; it cannot deadlock if not all clusters are resident, but then it would be slow); else the
  // regular schedule runs. It always uses light sync; wgmma always defers q_norm (QR_DEFER semantics).
  if ((p.fast & FAST_V3) && resident_ctas_v3(int(mma)) < CTAS_V3) p.fast &= ~FAST_V3_BITS;
  // DS41H_PROLOGUE_TICKET (v3 only): a launch without tickets after a ticket launch first zeroes block 0, which may still
  // hold that launch's counts (stream-ordered memset, graph-capturable); block 1 is zeroed by the ticket launches themselves
  {
    const int dev = int(x.get_device());
    p.tick = ((p.fast & FAST_V3) && tick_env()) ? 1 : 0;
    if (!p.tick && dev >= 0 && dev < 64 && g_last_tick[dev])
      TORCH_CHECK(cudaMemsetAsync(p.pcf, 0, NCOUNTERS * sizeof(unsigned), c10::cuda::getCurrentCUDAStream()) == cudaSuccess);
    if (dev >= 0 && dev < 64) g_last_tick[dev] = p.tick != 0;
  }
  if (p.fast & FAST_V3) {
    p.fast |= FAST_LIGHT_SYNC;
    if (mma != MMA_WGMMA) p.fast &= ~FAST_V3_ADEFER;  // A_DEFER: wgmma only, implies the P1a hop
    if (p.fast & FAST_V3_ADEFER) p.fast |= FAST_V3_HOP;
    if (p.fast & FAST_V3_PACE1) p.fast |= FAST_V3_PACE;
    if (mma == MMA_WGMMA) p.fast |= FAST_QR_DEFER | FAST_CLUSTER;
    if (mma == MMA_WGMMA)
      launch_v3<MMA_WGMMA>(p);
    else
      launch_v3<MMA_HMMA>(p);
    return;
  }
  p.fast &= ~FAST_V3_BITS;
  // FAST_CLUSTER needs all 240 CTAs co-resident in clusters of 8 (30 clusters; checked once per instantiation on the
  // idle device); without it the regular grid runs (same results).
  if ((p.fast & FAST_CLUSTER) && resident_ctas_cl(int(mma)) < CTAS_CL) p.fast &= ~FAST_CLUSTER;
  // FAST_QR_DEFER: wgmma only, and the pair finishers exist only in the cluster grid
  if (!(p.fast & FAST_CLUSTER) || mma != MMA_WGMMA) p.fast &= ~FAST_QR_DEFER;
  if (p.fast & FAST_CLUSTER) {
    if (mma == MMA_WGMMA)
      launch_cluster<MMA_WGMMA>(p);
    else
      launch_cluster<MMA_HMMA>(p);
    return;
  }
  const int fits = resident_ctas(int(mma));
  TORCH_CHECK(fits >= CTAS, "prologue: needs ", CTAS, " co-resident CTAs, device fits ", fits);
  if (mma == MMA_WGMMA)
    launch_pdl("ds41h prologue (wgmma)", prologue_kernel<MMA_WGMMA>, dim3(CTAS), dim3(THREADS), size_t(SMEM_BYTES_WG), p);
  else
    launch_pdl("ds41h prologue", prologue_kernel<MMA_HMMA>, dim3(CTAS), dim3(THREADS), size_t(SMEM_BYTES), p);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("MAXM") = MAXM;
  m.attr("CTAS") = CTAS;
  m.def("scratch_bytes", &scratch_bytes);
  m.def("num_ctas", &num_ctas);
  m.attr("MMA_HMMA") = int(MMA_HMMA);
  m.attr("MMA_WGMMA") = int(MMA_WGMMA);
  m.attr("FAST_EARLY") = int(FAST_EARLY);
  m.attr("FAST_SINKHORN") = int(FAST_SINKHORN);
  m.attr("FAST_PREFETCH") = int(FAST_PREFETCH);
  m.attr("FAST_FN_SMEM") = int(FAST_FN_SMEM);
  m.attr("FAST_QR_LOCAL") = int(FAST_QR_LOCAL);
  m.attr("FAST_LIGHT_SYNC") = int(FAST_LIGHT_SYNC);
  m.attr("FAST_CLUSTER") = int(FAST_CLUSTER);
  m.attr("FAST_QR_DEFER") = int(FAST_QR_DEFER);
  m.attr("FAST_OWN_COLLAPSE") = int(FAST_OWN_COLLAPSE);
  m.attr("FAST_V3") = int(FAST_V3);
  m.attr("FAST_V3_LATE_W2") = int(FAST_V3_LATE_W2);
  m.attr("FAST_V3_HOP") = int(FAST_V3_HOP);
  m.attr("FAST_V3_ADEFER") = int(FAST_V3_ADEFER);
  m.attr("FAST_V3_W1_LATE") = int(FAST_V3_W1_LATE);
  m.attr("FAST_V3_PACE") = int(FAST_V3_PACE);
  m.attr("FAST_V3_PACE1") = int(FAST_V3_PACE1);
  m.attr("FAST_V3_SK_G2") = int(FAST_V3_SK_G2);
  m.attr("FAST_V3_FAC") = int(FAST_V3_FAC);
  m.attr("FAST_V3_ILV") = int(FAST_V3_ILV);
  m.attr("FAST_V3_PUSH") = int(FAST_V3_PUSH);
  m.attr("FAST_V3_SK_G1") = int(FAST_V3_SK_G1);
  m.attr("FAST_V3_KV_SPREAD") = int(FAST_V3_KV_SPREAD);
  m.attr("FAST_V3_L2PF") = int(FAST_V3_L2PF);
  m.attr("FAST_V3_HOST_SPREAD") = int(FAST_V3_HOST_SPREAD);
  m.attr("FAST_V3_FENCE") = int(FAST_V3_FENCE);
  m.attr("FAST_V3_ACQPOLL") = int(FAST_V3_ACQPOLL);
  m.attr("FAST_V3_SKREG") = int(FAST_V3_SKREG);
  m.attr("FAST_V3_SKROW") = int(FAST_V3_SKROW);
  m.attr("FAST_V3_EXT_P1A") = int(FAST_V3_EXT_P1A);
  m.attr("FAST_V3_W2_WIN4") = int(FAST_V3_W2_WIN4);
  m.attr("FAST_V3_W2_OPEN") = int(FAST_V3_W2_OPEN);
  m.attr("C_QHINT") = int(C_QHINT);
  m.attr("OFF_XSQ") = int64_t(OFF_XSQ);
  m.attr("C_P1AHINT") = int(C_P1AHINT);
  m.attr("C_EPOCH") = int(C_EPOCH);
  m.attr("OFF_XCB") = int64_t(OFF_XCB);
  m.attr("C_AQ") = int(C_AQ);      // u32 index (in the scratch's counter block) of the attention handoff counter
  m.attr("C_AACK") = int(C_AACK);  // u32 index of the attention's pass counter
  m.attr("G2_CTAS") = G2_CTAS;      // C_AQ target = G2_CTAS + m
  m.attr("IDX_N") = IDX_N;          // Params::idx: indexer.wq_b rows
  m.attr("IDXW_N") = IDXW_N;        // weights_proj rows  // scratch offset of the bf16 collapse [MAXM][5120] (tests)
  m.attr("V3_SMEM") = V3_SMEM;
  m.def("max_resident_ctas_v3", [](int64_t mma) { return int64_t(resident_ctas_v3(int(mma))); }, py::arg("mma") = int64_t(MMA_HMMA));
  m.attr("OFF_QKV") = int64_t(OFF_QKV);  // scratch offset of the bf16 qkv [MAXM][1792] (tests)
  m.attr("CTAS_CL") = CTAS_CL;
  m.attr("CTAS_V3") = CTAS_V3;
  m.def("max_resident_ctas_cluster", [](int64_t mma) { return int64_t(resident_ctas_cl(int(mma))); }, py::arg("mma") = int64_t(MMA_HMMA));
  m.def("max_resident_ctas", &max_resident_ctas, py::arg("mma") = int64_t(MMA_HMMA));
  m.def("prologue", &prologue);
  m.def("set_prof", &set_prof, py::arg("buf") = py::none());
  m.attr("PROF_SLOTS") = PROF_SLOTS;
#ifdef DS41H_PROLOGUE_PROF
  m.attr("PROFILING") = true;
#else
  m.attr("PROFILING") = false;
#endif
  m.def("init", [] {  // smem attributes / occupancy of both instantiations, outside graph capture
    resident_ctas(MMA_HMMA);
    resident_ctas(MMA_WGMMA);
    resident_ctas_cl(MMA_HMMA);
    resident_ctas_cl(MMA_WGMMA);
    resident_ctas_v3(MMA_HMMA);
    resident_ctas_v3(MMA_WGMMA);
  });
}
