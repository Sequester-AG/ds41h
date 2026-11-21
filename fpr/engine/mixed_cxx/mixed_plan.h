// mixed_plan.h
// Kernel calls: fpr_cabi.h (every extension's table).
//
// FLOW of one mixed step (D13 _fallback -> Engine.step -> step_mixed):
//   Python: prepare A / T as today + the PRE-STEP phase (stage 1a, ENG_MIXED_PRESTEP): every
//   host sync, bucket create / evict (both buckets, before any layer), fused-RS init, prefill-attention plans with their
//   uploads (non-blocking, pinned), indexer slice tables, MoE grid / vcount, FlashMLA plans, head inputs pre-staged.
//   The result is one MixedPlan per rank (below): pointers + sizes only, nothing the issue phase must compute or read.
//   Driver (C++, stage 1b+): mixed_cxx.run(plans) with the GIL released; thread r (bound to device r + the rank stream,
//   created once at engine init) walks LayerOp[] of plan r and calls the extensions' C functions in the SAME order and
//   with the SAME arguments as the Python pieces (bitwise by construction). Cross-rank order = device flags (as today).
//   run() returns when every thread has ISSUED its list (not when the GPU finishes); Python then does the head / sync.
//
// OWNERSHIP: every pointer in a plan is engine-owned device memory (bucket state, pools, plan buffers) that stays alive
// until the step's sync; the driver allocates nothing during run(). Workspaces come from the plan (ws pointers sized by
// the extensions' *_ws_bytes at bucket creation).
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_MIXED_PLAN_VERSION 1
#define FPR_W 4                      // ranks (GPUs) per engine

// Which piece a LayerOp issues (= today's Engine pieces, _issue order). One op per (piece, layer).
typedef enum {
  FPR_OP_EMBED = 0, FPR_OP_ATTN_BD, FPR_OP_ATTN_A, FPR_OP_IDX, FPR_OP_ATTN_B, FPR_OP_FFN_BD, FPR_OP_ROUTER, FPR_OP_AG,
  FPR_OP_MOE, FPR_OP_RS, FPR_OP_FINAL, FPR_OP_JOIN,
  FPR_OP_CP_P1, FPR_OP_CP_P2, FPR_OP_CP_P3,       // stage 1e
  FPR_OP_GRAPH                                     // replay a captured segment (bucket T graphs, if 1a of graphs ships)
} FprOpKind;

// The variant a piece takes this step (decided in the pre-step, never inside the driver).
typedef enum { FPR_MOE_FORK_V4 = 0, FPR_MOE_FORK_RAW, FPR_MOE_PROD_RSF, FPR_MOE_PROD_FUSED } FprMoeVariant;
typedef enum { FPR_ATTN_DECODE = 0, FPR_ATTN_DECODE_PLUS_PREFILL } FprAttnVariant;

// One piece of one layer on one rank: kind + layer + an index into the plan's per-kind argument arrays.
typedef struct {
  int32_t kind;            // FprOpKind
  int32_t layer;           // 0..42 (-1: embed / final / join)
  int32_t variant;         // FprMoeVariant / FprAttnVariant / 0
  int32_t arg;             // index into the plan's args of this kind (the Args struct of the extension function(s))
} FprLayerOp;

// Pointers of one bucket state (A or T) on one rank: the tensors Engine._alloc creates (s["X"], s["gf"], ...), by name
// id, so the driver never parses Python dicts. The table is filled once per bucket creation (stable across steps).
typedef struct {
  int32_t R;               // rows per rank of the bucket
  int32_t n_ptrs;
  const void* const* ptrs; // indexed by FprStateField (mixed_fields.h, generated from Engine._alloc's keys)
  const int64_t* bytes;    // allocation sizes (debug checks)
} FprBucketState;

// The per-rank plan of one mixed step.
typedef struct {
  int32_t version;         // FPR_MIXED_PLAN_VERSION
  int32_t rank, device;
  cudaStream_t stream;     // the engine's rank stream; side streams below
  cudaStream_t side_stream;        // mHC mixes / shared expert (today's s["side"])
  cudaEvent_t* events; int32_t n_events;   // pre-created events the pieces record / wait (mix_ev, Engram, AGCE handoff)
  FprBucketState bucket;   // this segment's bucket (A for layers 0..20 + embed, T for 21..39 + final)
  int32_t n_ops;
  const FprLayerOp* ops;   // issue order for this rank (the same piece order as Engine._issue)
  // Per-kind argument blocks: arrays of the extensions' own Args structs (fpr_cabi.h rule 1), filled by the pre-step.
  // The driver passes &args_<kind>[op.arg] to the function the op's (kind, variant) maps to.
  const void* args[FPR_OP_GRAPH + 1];      // per FprOpKind: pointer to that kind's Args array
  int64_t args_stride[FPR_OP_GRAPH + 1];   // sizeof one Args struct of that kind (ABI drift guard)
  // Peers (for comm kernels whose Args carry 4-pointer arrays: filled at bucket creation, constant).
  const void* peer_ptrs[FPR_W];
} FprMixedPlan;

// ---- driver API (fpr/engine/mixed_cxx/driver.cpp; Python binding mixed_cxx.Driver)
typedef struct FprDriver FprDriver;

// Create once per engine: 4 threads, each bound to devices[r] and (later) the plan's streams; calls every extension's
// <ext>_init_thread on its thread (cuBLAS handles, DeepGEMM warm-up). tables[] = the extension tables (dlsym'd).
FprDriver* fpr_driver_create(const int devices[FPR_W], const FprExtTable* const* tables, int n_tables);
// Issue one mixed segment on all ranks (plans[r] for rank r) and return when all 4 threads issued (not GPU-complete).
// Returns FPR_OK or the first error (rank encoded: err * 16 + rank). Never blocks on the GPU.
int fpr_driver_run(FprDriver* d, const FprMixedPlan* const plans[FPR_W]);
// Timing of the last run per rank (host issue us), for the steplog.
void fpr_driver_last_issue_us(const FprDriver* d, double out_us[FPR_W]);
void fpr_driver_destroy(FprDriver* d);

#ifdef __cplusplus
}
#endif
