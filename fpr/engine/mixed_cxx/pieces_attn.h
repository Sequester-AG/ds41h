// pieces_attn.h -- the attention pieces of the C++ mixed-step driver. Compiled into the
// mixed_cxx extension (pieces_attn.cpp, fpr_pieces_attn_register); Python side mixed_cxx_attn.py (setup / step_params /
// piece_<kind> / the static-struct builders). Each piece issues exactly the Python piece's launches in the same order on
// the rank's main stream, for the branch the served config takes in mixed buckets; every other branch stays Python
// (piece_<kind> returns False before building anything).
//
// Documented exceptions to fpr_cabi.h
//  - DeepGEMM (wo_a / wo_b, dg_cabi): its process-global num_sms under one process mutex, warmed keys only (dg_cabi.h).
//  - FlashMLA (_flashmla_C::sparse_decode_fwd / sparse_prefill_fwd through the dispatcher): the op allocates its own
//    outputs (lse / max_logits every call; the tile-scheduler metadata on a plan's first call) through torch's caching
//    allocator, always under CUDAStreamGuard(the rank stream) so the frees stay stream-ordered; the same allocations the
//    Python path makes. The per-(step, plan key) metadata lives in this file's per-rank state (reset per segment).
//  - cuBLAS GEMMs (q_a | wkv) through ATen (at::mm_out on from_blob views, same call as torch.mm(out=)): ATen's per-
//    (thread, device) handle and workspace.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#include "pieces.h"
#include "../../kernels/attn_dense/dense_cabi.h"
#include "../../kernels/attn_dense/ropeq64_cabi.h"
#include "../../kernels/attn_dense/w8a16_cabi.h"
#include "../../kernels/attn_dense/dg_cabi.h"
#include "../../../ds41h/kernels/fp8_linear_cabi.h"
#include "../../kernels/indexer2f/idx2f_topk_cabi.h"
#include "../../kernels/indexer2f/idx2f_fcand8_cabi.h"
#include "../../kernels/indexer2f/idx2f_cand_cabi.h"
#include "../../kernels/indexer2f/idx2f_pf4_cabi.h"
#include "../../kernels/attn_dense/gather_k_cabi.h"
#include "kernel_capture.h"
#include "triton_replay.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_PIECE_ATTN_VERSION 1

// attn_a (attn.attn_a, mixed branch): qakv[R, N] = bf16(a_rows[R, K] @ wqkv_a[N, K]^T) (cuBLAS through ATen), then
// attn_dense.norm_rope2 (q_a -> qr, kv -> RoPE(norm(kv))). All pointers are the bucket's persistent buffers / weights.
typedef struct {
  int32_t version, rank, layer, device;
  int64_t R;
  cudaStream_t main;
  void* a_rows;               // bf16 [R, K], row stride a_ld (s["a"])
  int64_t a_ld;
  void* wqkv;                 // bf16 [N, K] contiguous (W["wqkv_a"])
  int64_t n, k;
  void* qakv;                 // bf16 [R, N] contiguous (ws["qakv"])
  FprAttnDenseNormRope2Args nr2;   // ws["qakv"][:, :1280] -> ws["qr"], ws["qakv"][:, 1280:] -> ws["kv"]; pos, cos_sin
} FprPieceAttnA;

// a strided tensor view for the dispatcher calls (FlashMLA): rank <= 4, sizes / strides in elements
enum { FPR_AV_U8 = 1, FPR_AV_I32 = 2, FPR_AV_F32 = 3, FPR_AV_BF16 = 4, FPR_AV_E4M3 = 5, FPR_AV_I64 = 6 };
typedef struct {
  void* ptr;
  int64_t sizes[4];
  int64_t strides[4];
  int32_t ndim, dtype;   // FPR_AV_*; ptr NULL = None
} FprAttnView;

// attn_b (attn.attn_b = attn_b_insert + attn_b_attend), the T-bucket / no-pf mixed branch:
//   q_b: w8a16.gemm (W["w8"]["wq_b"]: qr -> q with RoPE(q) + the rows' window-KV insert)  (attn_half.w8_qb, one launch)
//   FlashMLA: _flashmla_C::sparse_decode_fwd(q, k_swa, si, sl, sink, plan..., k_comp, ti, tl, 512, scale, out=o)
//   DENSE8 (W["A8c"]): ropeq64.inv_rope_fp8_quant(o) -> (oq, osc); dg.fp8_einsum (wo_a) -> z;
//                      fp8_linear.act_quant_128(z) -> (zq, zs); dg.fp8_gemm_nt (wo_b) -> y
typedef struct {
  int32_t version, rank, layer, device;
  int64_t R;
  cudaStream_t main;
  FprW8a16GemmArgs qb;                 // static: pointers of qr / q / pos / cs / kv / slot / cache / packed / ws / cnt
  int32_t plan_key;                    // FlashMLA plan id of this layer in the bucket (_new_scheds' key; 0..15)
  int32_t comp;                        // the layer has the compressed (extra) cache
  FprAttnView fq, fk_swa, fsi, fsl, fsink, fk_comp, fti, ftl, fo;   // sparse_decode_fwd's tensors
  double scale;                        // c["scale"]
  FprRopeq64InvRopeFp8QuantArgs rq;    // o -> oq / osc (persistent per-bucket buffers)
  FprDgEinsumArgs woa;                 // (oq^T, osc^T views) x A8c wo_a -> z [R, 8, 1024]
  FprFp8LinearActQuant128Args aq;      // z -> zq / zs (persistent per-bucket buffers)
  FprDgGemmArgs wob;                   // (zq, zs view) x A8c wo_b -> y [R, 5120]
  // stage 1c-1, bucket A with ENG_ATTN_PF (attn_b_attend's pf branch): pf = 1 -> the FlashMLA call above takes the decode
  // rows [:row_lo] only, on plan_key = c["sched_dec"]'s slot (none when row_lo == 0); the prefill rows [row_lo, row_hi)
  // go through attn_prefill.attend (FprAttnPfStep: gather / combine / sparse prefill), rows [row_hi, R) of o are zeroed
  int32_t pf, ratio, src, isrc;        // c["ratio"], c.get("src") / c.get("isrc") (-1 = None)
  void *pf_q, *pf_o;                   // bf16 [R, 64, 512] (ws["q"], ws["o"])
  const void* pf_kswa;                 // the window cache view (_view3: [pages, 32, 584] u8), page stride bytes
  int64_t pf_kswa_ps;
  const void* pf_kcomp;                // the compressed cache view ([pages, 64 / ratio, 584]) or NULL (ratio 0)
  int64_t pf_kcomp_ps;
  const void* pf_ti;                   // int32 [R, 512]: c["ti"] (the index source's GLOBAL top-512), ratio > 0
  const void* pf_sink;                 // f32 [64] W["sink"]
  // wo = 1 -> attn.attn_b_attend's ENG_WO_W8A16 `full` tail instead of DENSE8 (rq / woa / aq / wob
  // unused): attn_dense.inv_rope(o) in place (= kd.inv_rope), then engine._WoW8's w8a16 path (rows < ENG_DQ_MINR, so no
  // _Dq slot): wo_a o[R, 32768] -> z (grouped: 1024 rows / 4096 kx per group), wo_b z -> y; each with its Gemm's packed,
  // and its ws / cnt + splits at <= 128 rows (splits 1 above, no ws / cnt), exactly the pybind call's arguments
  int32_t wo, pad_wo;
  FprAttnDenseInvRopeArgs ir;
  FprW8a16GemmArgs woa8, wob8;
  // z^T(8, R, 1024) = bmm(o^T(8, R, 4096), w_a(8, 1024, 4096)^T); y = mm(z, w_b[5120, 8192]^T)
  // Spans only: mixed_cxx hands the slots over at every flush of a span with such entries (pending side-stream dequants
  // waited on the rank stream, owners reset), so Python's _Dq bookkeeping stays exact.
  int32_t qdq, pad_qdq;
  int64_t dq_q, dq_a, dq_b;             // dequant record handles (attn_add_dq_record), -1 = none
  void *slot_q, *slot_a, *slot_b;       // the slots' bf16 base pointers ([N, K] row-major views)
  int64_t q_n, q_k, a_n, a_k, b_n, b_k;
  void *wo_o, *wo_z, *wo_y;             // ws["o"] (contiguous [R, 32768]), ws["z"] [R, 8192], ws["y"] [R, 5120]
  int64_t z_ld, y_ld;
  FprAttnDenseRopeQInsertArgs rqi;
} FprPieceAttnB;

// stage 1c-1: the step's attn_prefill.Plan on one rank (mixed_cxx_attn.step_params -> attn_set_pf_step), pieces_attn's
// own per-rank state (not FprStepParams). Per ratio layout (Plan.group_plan / group_ws): the request groups, each with its
// workspace view and the combine launch (a Triton record index + its parameter values: pointers of this step's tables).
#define FPR_PF_MAX_GROUPS 32
#define FPR_PF_MAX_LAYOUTS 4
#define FPR_PF_MAX_RECS 8
typedef struct {
  int64_t a, b, N, M;                  // requests [a, b), the group's max compressed count, + max window gather length
  void* kv;                            // bf16 [b - a, M, 512] (the group's workspace view)
  int64_t q0, q1;                      // the group's rows within the plan (pl.qsl_h[a], pl.qsl_h[b])
  int32_t rec, gy;                     // combine record (FprAttnPfStep.rec[]) of this group's key class; its grid y
  uint64_t vals[16];                   // its parameter values (Rec.values: Python-level arguments incl. constexprs)
} FprAttnPfGroup;
typedef struct {
  int32_t ratio, ngroups, disjoint, pad;
  FprAttnPfGroup g[FPR_PF_MAX_GROUPS];
} FprAttnPfLayout;
typedef struct {
  int32_t valid, nlayouts, nrecs, pad0;
  int64_t n, row_lo, row_hi;           // plan rows (row_hi - row_lo) and their bucket rows
  const void *seq_lens, *gather_lens, *bt, *wbt, *inv;   // int32 Plan tables
  void* loc;                           // int32 [n, 512] (Plan.loc)
  int64_t ld_bt, ld_wbt, n_inv;
  const void* seq_lens_r[FPR_PF_MAX_LAYOUTS];            // pl.seq_lens // ratio of layout i (ratio > 0), this step's tensors
  void* comb[2];                       // int32 [n, cw]: [0] ratio > 0 (cw 640), [1] ratio 0 (cw 128) (Plan.comb[cw])
  void* clen[2];
  int64_t cw[2];
  FprAttnPfLayout lay[FPR_PF_MAX_LAYOUTS];
  FprTrRecord rec[FPR_PF_MAX_RECS];
  // the piece's own reuse state (= Plan.ws_ratio / comp_src / comb_key), reset by the setter and by attn_pf_invalidate
  int32_t ws_ratio, comp_src;          // -1: none
  int32_t ck_valid[2], ck_ratio[2], ck_isrc[2];
} FprAttnPfStep;

// idx (hooks_idx._score_general of a T-role mixed bucket's consumer layers 24 / 28 / 32 / 36; served flags: fast mode,
// f8 logits, lean eager workspace, no fuse): w / q projections (ATen mm), the CuTe q-quant (kernel_capture record), the
// rows' q / weights by ir2row (ATen index_select), OwnedIndexer._consumer_ws's chunks (prep2 or cand_prep, f8 candidate
// logits, block maxima, exact top-512), compute_global_topk_indices_and_lens (Triton replay, grid IR), the scatter into
// s["ti"][L] / s["tl"][L] (ATen fill_ / zero_ / index_copy_). Per-step values: FprAttnIdxStep (pieces_attn's own
// per-rank state, set by mixed_cxx_attn.step_params; not FprStepParams).
#define FPR_IDX_MAX_CHUNKS 16
typedef struct {
  int64_t b0, b1;                       // pseudo-requests [b0, b1) of the chunk (rows 4 b0 .. 4 b1)
  void *ublk, *uslot, *nunion, *cblk, *clen, *upk, *comp, *bm;   // EagerWorkspace views (prep_bufs / logits / bm)
  int64_t ld_cblk;                      // cblk row stride (prep2: the candidate rows themselves, 2048)
} FprAttnIdxChunk;
typedef struct {
  int32_t valid;                        // 1: this segment's idx tables were set (else the piece refuses: FPR_E_SHAPE)
  int32_t idle;                         // s["idx"]["idle"]: fill / zero only
  int32_t use2;                         // prep2 (ascending candidates, rows <= LONG_ROW) else cand_prep
  int32_t single;                       // one chunk: the prep of layer 24 is reused by 28 / 32 / 36
  int32_t nchunks, nwords;              // chunks; cand_prep's bitmap words per row
  int64_t B, IR, nv;                    // pseudo-requests, indexer rows, the step's valid rows (vir / vrow length)
  int32_t skip[8];                      // per LAYERS index (2, 8, 14, 20, 24, 28, 32, 36): oi.max_len <= 0 -> fill / zero
  int32_t prep_done, pad1;              // the piece's own: a consumer computed the single chunk's prep this segment
                                        // (= OwnedIndexer.prep is not None; set by the piece, reset by the setter)
  FprAttnIdxChunk ch[FPR_IDX_MAX_CHUNKS];
} FprAttnIdxStep;

typedef struct {
  int32_t version, rank, layer, device;
  int64_t R;
  cudaStream_t main;
  int32_t lidx, pad0;                   // LAYERS index of `layer`
  // projections: w = x @ idx_wp^T [R, 32]; q = qr @ idx_wq_b^T [R, 4096] (bf16, cuBLAS through ATen)
  void *x, *wp, *qr, *wqb, *wbuf, *qbuf;
  int64_t x_ld, kx, qr_ld, kq;
  int64_t pad1;                         // (explicit: FprKcRecord is 16-B aligned; ctypes mirrors every byte)
  FprKcRecord qq;                       // the CuTe q-quant on (pos, qbuf, cos_sin, wbuf) -> (q8buf, wobuf)
  void *q8buf, *wobuf;                  // e4m3 [R, 32, 128] (as u8), f32 [R, 32]
  void *ir2row, *q8i, *woi;             // int64 [>= IR]; e4m3 [>= IR, 32, 128]; f32 [>= IR, 32]
  // consumers (layer-20 CompactK of the bucket's indexer)
  void *cand, *seq_lens1, *kv, *bt, *topk_out, *slow;
  int64_t num_pages, page_stride, ld_bt;
  // global top-k (Triton replay, grid (IR,)): the record + its values with the persistent output buffers
  FprTrRecord gt;
  uint64_t gt_vals[16];
  // scatter
  void *gbuf, *lnbuf, *ti, *tl, *vir, *vrow;
  int64_t pad2;                         // (explicit tail: sizeof is a multiple of 16)
} FprPieceAttnIdx;
#ifdef __cplusplus
static_assert(offsetof(FprPieceAttnIdx, qq) % 16 == 0 && sizeof(FprPieceAttnIdx) % 16 == 0 &&
                  offsetof(FprPieceAttnIdx, pad2) + 8 == sizeof(FprPieceAttnIdx),
              "FprPieceAttnIdx: no implicit padding (ctypes mirror)");
#endif

// Shared FlashMLA sparse decode for fixed-plan callers: one _flashmla_C::sparse_decode_fwd
// (dispatcher, boxed) under CUDAStreamGuard(call.stream), = vLLM flash_mla_with_kvcache(q, k_cache, indices=..., topk_length=
// ..., attn_sink=..., extra_*, head_dim_v=d_v, softmax_scale=sm_scale, out=out) with the plan's EXISTING metadata
// (FlashMLASchedMeta.tile_scheduler_metadata / num_splits after its first Python call): tsm / ns must be given (NULL:
// FPR_E_ARG; this helper never plans). The op allocates its lse. Callable from any
// driver thread (no GIL); the op is resolved on first use (vllm._flashmla_C must be imported by Python before).
typedef struct {
  FprCall call;
  FprAttnView q, kv, indices, topk_length, attn_sink, tsm, ns, extra_kv, extra_indices, extra_topk_length, out;
  int64_t d_v;
  double sm_scale;
} FprAttnFlashMlaDecodeArgs;

int fpr_attn_flashmla_decode(const FprAttnFlashMlaDecodeArgs* a);

// rank `rank`'s FlashMLA plan slots slots[0 .. n) start over (= the pybind attn_reset_plans; core2b's native NEW_SCHEDS).
// FPR_E_ARG (nothing changed): rank outside [0, 4), n < 0, slots NULL with n > 0, a slot outside [0, 64). Host only, no GIL.
int fpr_attn_reset_plans(int rank, const int64_t* slots, int n);
// the CP message capacity this driver was built with (FPR_CP_MAX_MSGS): core2b checks a step against it
int fpr_attn_cp_max_msgs(void);

// CP prefill exchanges: the step's messages, set per
// CP step by mixed_cxx_attn.cp_step (Python thread, before the segment's first span) into pieces_attn's own state. One
// message = one cp.py _send + its _recv: SEND (kinds cp_x1a_send / cp_x2_send, on the sender's thread and rank stream):
// rows `src_idx` of the sender's pool view (base, row stride src_ld bytes) gathered (ATen index_select), byte columns
// [off, off + len) made contiguous, a [n, len] buffer allocated on the receiver under its CP mail stream (that pool), one
// cudaMemcpyAsync on the sender's stream (= comm.cu peer_copy), an event recorded after it, then published (atomic);
// RECV (kinds cp_x1a_recv / cp_x2_recv, the receiver's thread and rank stream): wait for the publish (bounded spin), the
// rank stream waits the event, rows `dst_idx` of the receiver's view at [off, off + len) = the buffer (ATen index_put_).
// Tensors live until attn_cp_done (Python, after the step's sync). Every send of a layer comes before its receives in
// every rank's program and messages go forward in the group's position order: no cycle with the driver's 4 threads
// (threads only: a serial issue in rank order could wait on a rank issued later).
#define FPR_CP_MAX_MSGS 4096   /* 4 KV-source layers x 6 pairs x 2 sections + 3 halo x 21 layers = 111 per CP group: 36 groups */
enum { FPR_CP_X1A = 1, FPR_CP_X2 = 2 };
typedef struct {
  int32_t src_rank, dst_rank, layer, kind;   // kind FPR_CP_X1A / FPR_CP_X2
  int64_t n, len, off;                       // rows; bytes per row; the section's first byte column
  const void* src_idx;                       // int64 [n] on the sender (cp.py: pages[r][r] / halo_src[r])
  const void* dst_idx;                       // int64 [n] on the receiver (pages[q][r] / halo_dst[q])
  void* src_base;                            // the sender's pool view (_pg(r) / _wsec(r)), uint8 [rows_src, src_ld]
  void* dst_base;
  int64_t src_rows, src_ld, dst_rows, dst_ld;
  int32_t src_dev, dst_dev;
  void* mail_stream;                         // cudaStream_t of the receiver's CP mail stream (cp.py _mail_stream)
} FprCpMsg;

// idx_a: hooks_idx._compress_general + _score_general of an A-ROLE mixed bucket's index
// layer (2 / 8 / 14 / 20, the KV sources) on the served flags: tables2f.GeneralTables device tables, the lean OwnedIndexer
// on the EagerWorkspace, fast mode, idx2f dense logits (idx2f.LOGITS f8 / cand) or the shared-key prefill logits
// (IDX2F_PF, idx2f_pf4) per slice, rows <= LONG_ROW, no mirror / IDX2F_DL / persistent bmax. Anything else: the handle
// refuses (-1) and the Python piece runs.
//   compress: ks = x @ comp_w^T (fp32 out, at::mm_out with out_dtype = torch.mm(out_dtype=)), fused_save_compress_norm /
//             rope_quant_insert / indexer_k_norm_rope_store (Triton replays; recorded with -1 slot mappings, so the
//             recording wrote nothing), kpre = lat @ idx_wk^T (at::mm_out)
//   score:    w / q projections, the CuTe q-quant (kernel_capture), q8i / woi by ir2row (= the T piece), then
//             OwnedIndexer._dense_sliced: per slice of whole pseudo-requests [b0, b1) the logits (pf4 when the slice's
//             groups average >= 2 pseudo-requests, else logits_dense) into the workspace, block_max (<= 1,024-row calls),
//             layer 20's select_blocks into cand, the exact top-512 (cblk NULL); compute_global_topk_indices_and_lens
//             (Triton replay, grid IR) and the scatter into s["ti"][L] / s["tl"][L] (= the T piece)
// Kinds: "idx" (handles >= FPR_IDXA_HANDLE_BASE: both halves, the non-CP p_attn), "idx_c" / "idx_s" (CP prefill: the
// compressor in P1 before the X1a send, the scoring in P2 after the X1a receive). Per-step values: FprIdxAStep
// (pieces_attn's own per-rank state, set by mixed_cxx_attn.step_params for an A-role segment).
#define FPR_IDXA_HANDLE_BASE (1 << 20)
#define FPR_IDXA_MAX_SLICES 64
#define FPR_IDXA_MAXV 32
typedef struct {
  int64_t b0, b1;                       // pseudo-requests of the slice
  int64_t gst_off;                      // int32 offset of the slice's group starts in FprIdxAStep.gst; -1: logits_dense
  int64_t g;                            // groups of the slice (pf4)
} FprIdxASlice;
typedef struct {
  int32_t valid, idle;
  int32_t logits;                       // 1 = idx2f_fcand8.logits_dense, 2 = idx2f_cand.logits_dense
  int32_t global_rows;                  // LayerIn.global_rows: the step's rows of every rank (production_backend)
  int64_t B, IR, nv;                    // pseudo-requests, indexer rows, the step's valid rows (vir / vrow length)
  int64_t nreq;                         // fused_save_compress_norm's num_reqs (ratio 2): qsl.numel() - 1
  int32_t skip[4];                      // per KV source (2, 8, 14, 20): oi.max_len <= 0 -> the score is fill / zero
  int32_t refuse[4];                    // per source, rows > LONG_ROW: 1 = the piece refuses (Python's stock select /
                                        // top-k), 2 = the piece's stock branch: the same
                                        // vLLM select_candidate_blocks / _C top-k / canon_topk calls; 0 = the fast path
  int64_t max_len[4], width[4];
  int32_t nsl[2], pad1[2];              // slices per ratio set ([0] ratio 2: layers 2 / 8 / 14, [1] ratio 1: layer 20)
  FprIdxASlice sl[2][FPR_IDXA_MAX_SLICES];
  const int32_t* gst;                   // device int32: every pf4 slice's group starts (relative to its b0)
  int64_t max_seq_len;                  // LayerIn.max_seq_len (the stock top-k's host bound)
} FprIdxAStep;

typedef struct {
  FprKcRecord qq;                       // FIRST (16-B aligned): the CuTe q-quant on (pos, qbuf, cos_sin, wbuf) -> (q8buf, wobuf)
  int32_t version, rank, layer, device;
  int64_t R;
  cudaStream_t main;
  int32_t lidx, sidx;                   // KV_SOURCES index 0..3; slice set (0 ratio 2, 1 ratio 1)
  int32_t ratio, pad0;
  // compress
  void *x, *comp_w, *ks, *lat, *idx_wk, *kpre;
  int64_t x_ld, kx, ncomp;              // ncomp = 512 * ratio
  FprTrRecord fscn;                     // fused_save_compress_norm (grid: nreq + ceil(R / ratio))
  uint64_t fscn_vals[FPR_IDXA_MAXV];
  int32_t fscn_nreq, fscn_nv;           // pyarg index of num_reqs (-1: ratio 1, constant); values used
  FprTrRecord rqi;                      // rope_quant_insert (grid R)
  uint64_t rqi_vals[FPR_IDXA_MAXV];
  FprTrRecord iks;                      // indexer_k_norm_rope_store (grid R)
  uint64_t iks_vals[FPR_IDXA_MAXV];
  int32_t rqi_nv, iks_nv;
  // score: projections (= FprPieceAttnIdx)
  void *wp, *qr, *wqb, *wbuf, *qbuf, *q8buf, *wobuf, *ir2row, *q8i, *woi;
  int64_t qr_ld, kq;
  // the dense layer
  void *kv;                             // the source's index-K pages (CompactK.kv) [num_pages, page_tok, 1, 132] u8
  int64_t num_pages, page_stride, page_tok;
  void *bt;                             // GeneralTables.bt [>= B, bt_cols] int32
  int64_t ld_bt, bt_cols;
  void *seq_lens;                       // int32 [>= IR]: sl[ratio] flat
  void *nb_lens;                        // int32 [>= IR]: layer 20's select (GeneralTables.nb_lens)
  void *cand;                           // int32 [>= IR, 2048]: layer 20's candidate blocks (the bucket's indexer)
  void *topk_out, *slow;                // the lean shared top-512 buffer [>= IR, 512]; the slow-path counter
  void *ws_f32, *ws_bm;                 // EagerWorkspace.f32 / .bmax bases
  int64_t ws_f32_n, ws_bm_n;            // their elements
  // global top-k + scatter (= FprPieceAttnIdx)
  FprTrRecord gt;
  uint64_t gt_vals[16];
  void *gbuf, *lnbuf, *ti, *tl, *vir, *vrow;
  // the stock branch (FprIdxAStep.refuse[lidx] == 2): idx2f._dense_sliced's long-row slice body =
  //   [layer 20] vLLM select_candidate_blocks: _block_scores_kernel (grid rows x cdiv(nblocks, 128)) -> torch.topk(scores,
  //   min(2048, nblocks)) -> _store_candidates_kernel (grid rows x 8); stock_topk = torch.ops._C.persistent_topk (cooperative
  //   when global rows <= 64 and slice rows <= 64); canon_topk (IDX2F_CANON_STOCK)
  FprTrRecord bs, sc;                   // the two Triton kernels (layer 20 only), recorded at build time
  uint64_t bs_vals[16], sc_vals[16];
  int32_t bs_nv, sc_nv;
  int32_t bs_i[6];                      // pyarg index of: logits, ends, scores, stride_row, width, nblocks
  int32_t sc_i[4];                      // pyarg index of: values, indices, output, k
  int32_t canon, stock_ok;              // canon_topk after the stock top-k; the stock branch is built (else it refuses)
  void* topk_ws;                        // the _C top-k's uint8 workspace (vLLM's: torch.empty RADIX_TOPK_WORKSPACE_SIZE)
  int64_t topk_ws_n;
  int64_t pad4;
  int64_t pad3;                         // (explicit tail: sizeof is a multiple of 16)
} FprPieceIdxA;
#ifdef __cplusplus
static_assert(sizeof(FprKcRecord) % 16 == 0, "FprPieceIdxA: FprKcRecord size % 16");
static_assert(offsetof(FprPieceIdxA, version) == sizeof(FprKcRecord), "FprPieceIdxA: version offset");
static_assert(offsetof(FprPieceIdxA, pad3) + 8 == sizeof(FprPieceIdxA), "FprPieceIdxA: tail padding");
static_assert(sizeof(FprPieceIdxA) % 16 == 0, "FprPieceIdxA: size % 16");
static_assert(offsetof(FprIdxAStep, max_seq_len) + 8 == sizeof(FprIdxAStep) && sizeof(FprIdxAStep) % 8 == 0,
              "FprIdxAStep: no implicit padding (ctypes mirror)");
#endif
int piece_idx_a(const FprPieceIdxA* p, const FprIdxAStep* st, int layer, int what);   // what: 1 compress, 2 score, 3 both

// the per-step setters as C entries (nd's native mixed step; the pybind setters write the same state): FPR_OK / FPR_E_ARG
int fpr_attn_set_idx_step(int rank, const FprAttnIdxStep* st);   // T-role idx (consumers)
int fpr_attn_set_idxa_step(int rank, const FprIdxAStep* st);     // A-role idx_a; NULL = none this segment
int fpr_attn_set_pf_step(int rank, const FprAttnPfStep* st);     // attn_pf; NULL = none; resets the reuse state
int fpr_attn_cp_step(const FprCpMsg* msgs, int64_t n, const void* const* rank_streams);   // the CP step's messages
int fpr_attn_cp_done(void);                                      // after the step's sync
int fpr_attn_tr_check(const void* record, const uint64_t* vals);   // fpr_tr_check (FPR_OK: the values fit the record's key class)

#ifdef __cplusplus
}
#endif
