// tail_gather_cabi.h -- FPR_OP_TAIL_GATHER / T_STAGE: the
// device half of the mixed step's T stage (engine.py step_mixed: bucket A -> bucket T after layer 20) in ONE launch:
//   gather   for every key k and row i < n: dst[k] row i = src[k] row ix[i] (byte copy of row_bytes[k]; = torch.index_select
//            of contiguous rows, bitwise)
//   pads     for every pad j and row n <= i < pad_rows[j]: pad_dst[j] row i = pad_src[j] row i (a copy), or every 4-byte word
//            of the row = pad_word[j] (a fill; pad_src[j] NULL)
// Rows written by the gather and by the pads are disjoint, so the result equals Python's sequence (si20.copy_(si) of ALL rows,
// then the gather of rows [0, n); the index layers' ti[n:].fill_(-1) / tl[n:].zero_() after it).
// Follows fpr_cabi.h (rules 1-7): plain Args, caller-owned memory, the caller's stream only, no host sync, no allocation, no
// process-global mutable state. Two entries: fpr_tail_gather_run (explicit key / pad lists) and fpr_t_stage_run (the step's
// T stage from two per-view key tables FprTgKeys, = Engine._tail_gather's key order + the padding rules: the engine's
// ENG_TAIL_GATHER_KERNEL path and nd's T_STAGE call this one function).
#pragma once
#include "fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_TAIL_GATHER_MAX_KEYS 16
#define FPR_TAIL_GATHER_MAX_PADS 16

typedef struct {
  FprCall call;                                     // stream / device (rule 3); rank unused
  const int64_t* ix;                                // [n] source row of each destination row (device); NULL only if n == 0
  int64_t n;                                        // destination rows 0..n-1 of every key
  int32_t n_keys;                                   // <= FPR_TAIL_GATHER_MAX_KEYS
  int32_t n_pads;                                   // <= FPR_TAIL_GATHER_MAX_PADS
  const void* src[FPR_TAIL_GATHER_MAX_KEYS];        // key k: base of a row-major [src_rows[k], row_bytes[k]] buffer
  void* dst[FPR_TAIL_GATHER_MAX_KEYS];              // key k: base of the destination rows (>= n rows of row_bytes[k])
  int64_t row_bytes[FPR_TAIL_GATHER_MAX_KEYS];      // bytes per row (> 0)
  int64_t src_rows[FPR_TAIL_GATHER_MAX_KEYS];       // rows of src[k]: an ix value outside [0, src_rows) copies nothing
  const void* pad_src[FPR_TAIL_GATHER_MAX_PADS];    // pad j: copy source (same row layout as pad_dst), NULL = fill
  void* pad_dst[FPR_TAIL_GATHER_MAX_PADS];          // pad j: base of a row-major [pad_rows[j], pad_row_bytes[j]] buffer
  int64_t pad_row_bytes[FPR_TAIL_GATHER_MAX_PADS];  // bytes per row (> 0; a multiple of 4 for a fill)
  int64_t pad_rows[FPR_TAIL_GATHER_MAX_PADS];       // rows n .. pad_rows[j] - 1 are written (none if pad_rows[j] <= n)
  int32_t pad_word[FPR_TAIL_GATHER_MAX_PADS];       // fill: the 32-bit word written (ti: -1, tl: 0)
  int32_t* err;                                     // optional (device): += 1 per skipped out-of-range (row, key); NULL = none
} FprTailGatherArgs;

int fpr_tail_gather_run(const FprTailGatherArgs* a);   // table "tail_gather.run"

// ---- the T stage from per-view key tables
// One mixed bucket view's (rank's) per-row state keys in a FIXED order. p NULL = the key is absent (A views: si20 / sl20). ti / tl: the index layers <= 20 in
// INDEX_LAYERS order (2, 8, 14, 20), n_idx of them.
enum {
  FPR_TG_X = 0, FPR_TG_A_PRE, FPR_TG_A_POST, FPR_TG_A_COMB, FPR_TG_WS_Q, FPR_TG_SI, FPR_TG_SL, FPR_TG_SI20, FPR_TG_SL20,
  FPR_TG_MOE_OUT, FPR_TG_XM2, FPR_TG_F_POST, FPR_TG_F_COMB, FPR_TG_F_PRE,
  FPR_TG_TI0 = 14,                                  // ti of index layer j (j < n_idx) at FPR_TG_TI0 + j
  FPR_TG_TL0 = 18,                                  // tl of index layer j at FPR_TG_TL0 + j
  FPR_TG_N = 22
};
#define FPR_TG_MAX_IDX 4
#define FPR_TG_SLOTS 24
#define FPR_TG_KEYS_VERSION 1

typedef struct {
  int32_t version;                                  // FPR_TG_KEYS_VERSION
  int32_t n_idx;                                    // index layers <= 20 (<= FPR_TG_MAX_IDX)
  int64_t R;                                        // the view's rows
  void* p[FPR_TG_SLOTS];                            // key bases (row-major, contiguous), by FPR_TG_* slot
  int64_t row_bytes[FPR_TG_SLOTS];                  // bytes per row
  int64_t rows[FPR_TG_SLOTS];                       // rows of the key's tensor (shape[0])
} FprTgKeys;

typedef struct {
  FprCall call;
  const int64_t* ix;                                // [n] device: the tail rows (A row of T row i), e.g. tail_ix[:n]
  int64_t n;                                        // tail rows of this rank (T rows [0, n) are real)
  int32_t l20;                                      // sT["l20"]: X / a_* / ws.q / si -> si20 / sl -> sl20 (else moe_out ...)
  int32_t clear;                                    // BISECT clear_tail_padding: ti rows [n, R_T) = -1, tl = 0
  const FprTgKeys* A;                               // bucket A's view (sources)
  const FprTgKeys* T;                               // bucket T's view (destinations, padding)
  int32_t* err;                                     // optional, as FprTailGatherArgs.err
} FprTStageArgs;

// Builds the FprTailGatherArgs of a T stage (host only, pure: no CUDA call); FPR_E_ARG on a missing key / layout mismatch.
int fpr_t_stage_args(const FprTStageArgs* t, FprTailGatherArgs* out);
int fpr_t_stage_run(const FprTStageArgs* t);        // = fpr_t_stage_args + fpr_tail_gather_run; table "tail_gather.t_stage"

#ifdef __cplusplus
}
#endif
