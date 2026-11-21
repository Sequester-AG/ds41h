// pieces.h -- stage-1b piece structs of the C++ mixed-step driver.
// One static struct per (bucket state, layer, piece, rank)
// built once (Python, ctypes: mixed_cxx.py) and kept by the driver (add_static -> handle); per-step values in
// FprStepParams. Piece functions issue exactly the Python piece's launches in the same order (bitwise by construction).
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include "fpr_cabi.h"
#include "../../kernels/mhc_fuse/mhcf_cabi.h"
#include "../../kernels/sp_layer/comm_cabi.h"
#include "../../kernels/sp_layer/mhc_cabi.h"
#include "../../kernels/moe_stream/moe_raw_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_PIECES_VERSION 1
#define FPR_BD_MAX_CHUNKS 4

typedef struct {                 // per rank, per bucket segment of a step; set by the
                                 // pre-step (mixed_cxx.Driver.set_step) before the segment's pieces run
  int32_t chain_input;           // Engine._inputs: s["chain_input"] (the first layer's x0 / res0 / ... unless chained)
  int32_t moe_variant;           // FprMoeVariant (mixed_plan.h) for this bucket in this step; -1 = Python p_moe
  int64_t moe_vcount;            // host sum of valid rows over all ranks (s["vcount"])
  int32_t moe_n_wait, pad0_;     // 1d: AGCE per-source done events this rank's MoE waits first
  cudaEvent_t moe_wait[4];
} FprStepParams;

// The mHC boundary (Engine._split_bd via p_attn_bd / p_ffn_bd) on the ENG_MHCF fused path:
//   _join_mix, then
//   path 1 (R <= 320): frow; record ev; side: wait ev, fmix, fb_mix, record mix -> pending
//   path 2 (R  > 320): fa per chunk; record ev; fb_out per chunk; side: wait ev, fb_mix per chunk; record mix
//   attn_bd on AUX layers (37, 38, 39): aux[:, i] = X.mean(1) (ATen, main stream) after the boundary.
// inp: [0] = the first layer's inputs (x0, res0, post0, comb0, lag0), [1] = the chained ones (moe_out, xm2, f_post, ...):
// _inputs picks by (layer == first layer && !chain_input). ffn_bd always uses [1] (its inputs are fixed).
typedef struct {
  int32_t version, which;        // which: 0 attn_bd, 1 ffn_bd
  int32_t layer, R, path;        // path 1 frow, 2 fa; 0 = not on the fused path (the piece stays in Python)
  int32_t first_layer;           // this layer is the bucket's first (attn_bd: inputs set chosen by chain_input)
  int32_t n_chunks;              // path 2: chunks of mhc_chunk rows
  int32_t aux_index;             // attn_bd: index into AUX_LAYERS, -1 = none
  cudaStream_t main, side;
  FprMhcfFrowArgs frow[2];       // path 1, per inputs set
  FprMhcfFmixArgs fmix;          // path 1 (side)
  FprMhcfFbMixArgs fb_mix;       // path 1 (side)
  FprMhcfFaArgs fa[2][FPR_BD_MAX_CHUNKS];          // path 2, per inputs set and chunk
  FprMhcfFbOutArgs fb_out[FPR_BD_MAX_CHUNKS];      // path 2
  FprMhcfFbMixArgs fb_mix_c[FPR_BD_MAX_CHUNKS];    // path 2 (side)
  // aux (attn_bd only): X bf16 [R, HC, D] -> tmp bf16 [R, D] (mean over HC) -> aux slice [R, D] with row stride
  void* aux_X; void* aux_tmp; void* aux_dst; int64_t aux_dst_stride0; int32_t aux_HC, aux_D;
} FprPieceBd;

// per destination q: cs[q] waits
//     ev_f, cudaMemcpyAsync(gf_q + r * nb_f <- f, nb_f, D2D, cs[q]) (agrs.cu ce_copy)
//   logits = mm(f, gate_w.t(), out_dtype fp32) (ATen mm.dtype_out = cuBLAS, main stream), then moe_raw.router_top_pack3.
typedef struct {
  int32_t version, layer, R, ce_ok;  // ce_ok: the AGCE fields are filled (built with the bucket's 4 states)
  cudaStream_t main;
  cudaStream_t cs[4];                // agrs_engine._cstreams(eng, r): this rank's 4 copy streams
  void* gf_dst[4];                   // st[q]["gf"] + r * nb_f
  const void* f_src; int64_t nb_f;   // s["f"], R * D * 2 bytes
  void* f; int64_t f_st0, f_st1;     // bf16 [R, D] (s["f"], its strides)
  void* gw; int64_t gw_st0, gw_st1;  // bf16 [E, D] (lw["gate_w"]; the mm takes its .t())
  void* lg; int64_t lg_st0, lg_st1;  // fp32 [R, E] (s["logits"])
  int32_t D, E;
  FprMoeRawRouterTopPack3Args top;
} FprPieceRouter;

// per q: cs[q] waits ev_r, cudaMemcpyAsync(gr_q + r *
//     nb_r <- route_row, nb_r, D2D, cs[q]), record done[r][q] on cs[q]; receiver q's p_moe waits its senders' done events
//     first (agce_join; Python: st[q]["agrs_in"])
//   else (AG32 off): comm.ag_push2 (f + routing rows, cfg.ncta) + comm.wait (fl_f / ep_f, W * ncta signals).
typedef struct {
  int32_t version, layer, R, ce_ok;
  cudaStream_t main;
  cudaStream_t cs[4];
  void* gr_dst[4];                   // st[q]["gr"] + r * nb_r
  const void* rr_src; int64_t nb_r;  // s["route_row"], R * 64 bytes
  FprCommAgPush2Args push;
  FprCommWaitArgs wait;
} FprPieceAg;

// The reduce-scatter of a prod bucket = agrs_engine.p_rs -> Engine.p_rs (no ENG_C1_L2PF):
//   rsf (call arg = s.pop("agrs_rsf_pending")): comm.rs_sum on the RSF flag (W * RSF_GRID signals, grid sum_rsf.ncta)
//   else: comm.rs_push (part -> every rank's rs slot, cfg.ncta) + comm.rs_sum (W * ncta signals; grid 528 under
//         AGRS_SUM528 with R >= AGRS_MIN_R, else 132).
typedef struct {
  int32_t version, layer, R, rsf_ok; // rsf_ok: sum_rsf filled (the bucket had its agrs_fl_rsf / agrs_ep_rsf words)
  FprCommRsPushArgs push;
  FprCommRsSumArgs sum;
  FprCommRsSumArgs sum_rsf;
} FprPieceRs;

// ---- stage 1d: a span = one program per rank, run by the driver's 4 threads (fpr_drv run_span). Entries in the per-rank
// stream order of Engine._issue; every decision is made when the span is built (Python, per step): arg = which (BD),
// ce (ROUTER / AG), rsf (RS). Owner kinds (>= FPR_K_OWNER0) are the functions registered with fpr_drv_register_piece.
enum {
  FPR_K_BD = 1,          // handle: FprPieceBd; arg: unused (which is in the struct); chain_input from StepParams
  FPR_K_ROUTER = 2,      // handle: FprPieceRouter; arg: ce
  FPR_K_AG = 3,          // handle: FprPieceAg; arg: ce
  FPR_K_AGCE_JOIN = 4,   // the receiver's AGCE waits (before its MoE); no handle
  FPR_K_RS = 5,          // handle: FprPieceRs; arg: rsf
  FPR_K_FINAL = 6,       // handle: FprPieceFinal (hooks.head_run stays Python, after the span)
  FPR_K_JOIN = 7,        // _join_mix on the rank's main stream; no handle
  FPR_K_WAITEV = 8,      // the rank's main stream waits a Python event (handle = its cudaEvent_t): a Python boundary's
                         // pending s["mix_ev"], placed right before the first joining entry (as _join_mix would wait it)
  FPR_K_OWNER0 = 32,
};
typedef struct { int32_t kind, layer; int64_t handle; int32_t arg, pad_; } FprProgEntry;

#define FPR_FINAL_MAX_CHUNKS 16
// The head input of a prod bucket = Engine.p_final up to hooks.head_run (which stays Python):
//   join (the rank's pending C++ boundary mix), then
//   path 1 (MHCF, R <= 320): mhcf.frow(moe_out, xm2, f_post, f_comb, f_pre, norm_w, eps, Xf, h, ptok NULL)
//   path 2: per chunk of cfg.mhc_chunk rows: mhc.k1a_x(moe_out, xm2, f_post, f_comb, Xf), mhc.k1b_out(Xf, f_pre, norm_w,
//           eps, h, NULL)
typedef struct {
  int32_t version, R, path, n_chunks;
  cudaStream_t main;
  FprMhcfFrowArgs frow;
  FprMhcK1aXArgs k1a[FPR_FINAL_MAX_CHUNKS];
  FprMhcK1bOutArgs k1b[FPR_FINAL_MAX_CHUNKS];
} FprPieceFinal;

#ifdef __cplusplus
}
#endif
