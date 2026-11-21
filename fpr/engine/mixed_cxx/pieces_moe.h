// pieces_moe.h -- the moe piece of the C++ mixed-step driver. One static FprPieceMoe per (bucket state
// layer, rank, RSF or not), built once by mixed_cxx_moe.py (build_piece_moe) and kept by the driver (add_moe -> handle);
// the step's variant / vcount / AGCE waits come from FprStepParams (fpr_drv_step_params(rank)).
//
// Both prod variants run on the workspace of ENG_AGRS_WS=2 (agrs_ws: agrs_pmws_cabi.FusedRSC, one tensor per device), the
// same buffer the Python path uses; vcount above its rows -> the Python glue keeps the Python path (its counted fallback).
// AGCE: in 1b-i the agrs p_moe wrapper has already waited its events when the hook in Engine.p_moe runs; for 1d the
// driver passes them in StepParams.moe_wait (waited on the main stream first).
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include "fpr_cabi.h"
#include "pieces.h"            // FprStepParams
#include "agrs/pm_rs_cabi.h"
#include "moe_v4/moe_v4_cabi.h"
#include "moe_stream/moe_raw_cabi.h"
#include "moe_stream/route2_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_PIECE_MOE_VERSION 2

typedef struct {
  int32_t version;              // FPR_PIECE_MOE_VERSION
  int32_t layer, R, rank;
  int32_t rsf_ok;               // prod_rs filled (the RS receive buffers / flags of this bucket set, epoch, grid)
  int32_t fused_ok;             // prod filled (part)
  int32_t fork_ok;              // the fork Args below filled (the bucket has the fork tables)
  int32_t fork_route;           // 0 moe_v4r.route2r, 1 route2.route2r, 2 ids / wts copies + route2.route2
  int32_t fork_gemm;            // 0 moe_v4 gemm1r / gemm2r, 1 moe_raw gemm1r / gemm2r
  int32_t device;               // the rank's device (ATen views)
  cudaStream_t main, side;      // the rank stream (the piece's stream) / the engine's side stream (fork's shared expert)
  cudaEvent_t ev0, ev1;         // fork: main -> side / side -> main (this rank's, created by moe_init_rank); set by add_moe
  AgrsPmwsLayerArgs prod_rs;    // everything but vcount (the step's)
  AgrsPmwsLayerArgs prod;       // everything but vcount
  // fork: the shared expert slice on the side stream (ATen mm on these, all Tall rows) + moe_raw swiglu
  void* fall;                   // bf16 [Tall, H] (gf)
  void* sw13;                   // bf16 [2 Is, H]
  void* sw2;                    // bf16 [H, Is]
  void* gu;                     // bf16 [Tall, 2 Is]
  void* hsh;                    // bf16 [Tall, Is]
  void* sh;                     // bf16 [Tall, H]
  int64_t Tall, H, Is;
  FprMoeRawSwigluArgs swiglu;   // on side
  // fork: routing
  MoeV4Route2rArgs v4_route;    // fork_route 0
  FprRoute2rArgs r2r;           // fork_route 1
  FprRoute2Args r2;             // fork_route 2 (ids / w = the two byte buffers below)
  void* gr;                     // u8 [Tall, 64] routing rows
  void* ids_buf;                // u8 [Tall, 24] (int32 [Tall, 6]) = gr[:, :24].contiguous()
  void* wts_buf;                // u8 [Tall, 24] (fp32 [Tall, 6]) = gr[:, 24:48].contiguous()
  // fork: image, GEMMs, combine
  FprMoeRawPrepr5Args prepr5;
  MoeV4Gemm1rArgs v4_g1;        // fork_gemm 0
  MoeV4Gemm2rArgs v4_g2;
  FprMoeRawGemm1rArgs raw_g1;   // fork_gemm 1
  FprMoeRawGemm2rArgs raw_g2;
  FprMoeRawCombineArgs combine;
} FprPieceMoe;

// the piece (FPR_OK or FPR_E_*): sp = this rank's StepParams of the segment (fpr_drv_step_params(rank))
int piece_moe(const FprPieceMoe* st, const FprStepParams* sp, int layer);

#ifdef __cplusplus
}
#endif
