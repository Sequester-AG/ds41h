// pm_rs_cabi.h -- C ABI of the production-MoE layer of a mixed step. Extension agrs_pmws_c (pm_rs_cabi.cu: production's moe_route.cu +
// moe_prefill.cu private copies and agrs/pm_rs.cu, all #included UNCHANGED into one translation unit; built by
// kernels/agrs/agrs_pmws_cabi.py). Table symbol: fpr_agrs_pmws_table().
//
//   agrs_pmws.init         once per device before any launch (any thread): production's W4A16 hybrid setting
//                          (moe_prefill.cu set_hybrid) + the shared-expert swiglu launcher (pm_swiglu.launcher()).
//                          Process-wide, idempotent: a later call with DIFFERENT values returns FPR_E_ARG.
//   agrs_pmws.init_thread  once per driver thread, after it bound its device + rank stream: one small ATen mm on that
//                          stream creates torch's cuBLAS handle / workspace for (thread, device, stream), so no step allocates.
//   agrs_pmws.layer        = pm_fused's / pm_rs.cu's layer (the non-RSF mixed path, R < AGRS_MIN_R): part <- shared + routed.
//   agrs_pmws.layer_rs     = agrs_pm.FusedRS.layer_rs (RSF, R >= AGRS_MIN_R): the combine pushes every row into its
//                          owner's RS receive slot `rank`, then comm.cu's release signals (owners wait W x grid per round).
//   ws_bytes (both)        the caller-owned scratch for the call's T rows (x, h, xp / slot, ids, tw, perm_w, meta, hist):
//                          78,736 B per row + ~0.1 MB (4 KiB-aligned regions); size it once for the largest T.
// Bitwise: the same kernels, launch shapes and operands as pm_layer_impl (the allocating path) and pm_fused's layer;
// the only host-side changes are the buffers' addresses (caller ws), moe_permute's hist (in ws instead of torch::empty,
// the same moe_count / moe_scatter launches) and the stream (torch's current stream = Args.stream for the call).
#pragma once
#include <stdint.h>

#include "fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGRS_PMWS_MAXW 4                 // ranks a layer_rs call addresses

typedef struct {
  int32_t device;                        // the device this init is for (0..31)
  int32_t pad_;
  int64_t hybrid_split;                  // pm_fused.prod_settings(ds41h.moe)[1]
  int64_t hybrid_engine;                 // pm_fused.prod_settings(ds41h.moe)[2]
  int64_t swiglu_fn;                     // pm_fused.swiglu_ext().launcher(): void (*)(const void* gu, void* h, int T, int I, cudaStream_t)
} AgrsPmwsInitArgs;

typedef struct {
  FprCall call;                          // stream = the thread's rank stream, device = its device
  void* ws;                              // >= 48 KiB of device scratch for the warm-up mm (contents overwritten)
  int64_t ws_bytes;
} AgrsPmwsThreadArgs;

typedef struct {
  FprCall call;                          // stream, device (current), rank (layer_rs: this rank's slot in every owner)
  // rows: the engine bucket's all-gathered layout, Tall = W R rows, rank q's valid rows the prefix [q R, q R + nv[q])
  const void* fall;                      // bf16 [W R, H]
  const uint8_t* gr;                     // u8 [>= W R, 64] route rows (ids int32 x 6 | weights f32 x 6 | ...)
  const int32_t* nv;                     // device int32 [W]
  int64_t W, R;
  int64_t vcount;                        // host sum(nv): the call's T when padded == 0
  int64_t H;                             // hidden (5120)
  // this rank's routed experts (raw MXFP4 + ue8m0), contiguous
  const uint8_t* w13;                    // [E, 2 I, H / 2]
  const uint8_t* s13;                    // [E, 2 I, H / 32]
  const uint8_t* w2;                     // [E, H, I / 2]
  const uint8_t* s2;                     // [E, H, I / 32]
  int64_t E, I;                          // 384, 576
  // this rank's shared-expert slice, contiguous bf16
  const void* sw13;                      // [2 Is, H]
  const void* sw2;                       // [H, Is]
  int64_t Is;                            // 576
  // the bucket's shared-expert scratch, contiguous bf16, >= T rows each
  void* gu;                              // [>= T, 2 Is]
  void* hsh;                             // [>= T, Is]
  void* sh;                              // [>= T, H]
  int64_t scratch_rows;                  // rows gu / hsh / sh hold
  // layer: the output; layer_rs: unused (NULL)
  void* part;                            // bf16 [W R, H]
  // layer_rs only (layer: NULL / 0)
  void* rs_buf[AGRS_PMWS_MAXW];          // owner q's RS receive buffer: bf16, >= W R H elements
  uint32_t* rs_flag[AGRS_PMWS_MAXW];     // owner q's int32 flag word of this path
  uint32_t* epoch;                       // this rank's int32 epoch word of this path
  int64_t grid;                          // combine-push CTAs, a multiple of W, <= 65535 (served: 528)
  // config
  double limit;                          // 10.0
  int64_t variant;                       // production's W4A16 variant (pm_fused.prod_settings: 0xB9)
  int32_t padded;                        // 0 = compact (served), 1 = all Tall rows
  int32_t pad_;
  // caller-owned scratch (>= agrs_pmws_layer_ws_bytes(this), 512-B aligned)
  void* ws;
  int64_t ws_bytes;
} AgrsPmwsLayerArgs;

int agrs_pmws_init(const void* args);              // AgrsPmwsInitArgs
int agrs_pmws_init_thread(const void* args);       // AgrsPmwsThreadArgs
int agrs_pmws_layer(const void* args);             // AgrsPmwsLayerArgs
int agrs_pmws_layer_rs(const void* args);          // AgrsPmwsLayerArgs
int64_t agrs_pmws_layer_ws_bytes(const void* args);// AgrsPmwsLayerArgs (T from padded / vcount, H, E, I); < 0 = bad args
int64_t agrs_pmws_ws_bytes_for(int64_t T, int64_t H, int64_t E, int64_t I);
const FprExtTable* fpr_agrs_pmws_table(void);

#ifdef __cplusplus
}
#endif
