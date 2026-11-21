// rowexp_cabi.h -- C ABI of kernels/rowexp/rowexp.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// D13's per-step row expansion
//   rowexp.ring_ops  k_ring_ops: the step's window-page ring ops (sets / clears) into ring[slot]
//   rowexp.expand    k_expand: the step descriptor -> per-row pos / own / si / sl / paged, per-sequence start, row -> seq
//   rowexp.advance   k_advance: nc[slot] = start + accepted (after the head)
// Every pointer is device memory of the caller's rank (rowexp.RowExp's buffers); counts live on the device (graph-capturable).
// Tables: fpr_rowexp_table(); the pybind wrappers (advance / ring_ops / expand) build these structs and call the same cores.
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {                       // = pybind ring_ops(ring, ops, counts)
  FprCall c;
  int32_t* ring;                       // int32 [slots, 512, 2] (int2 entries: block, page)
  const int32_t* ops;                  // int32 [3 * ops_max]
  const int32_t* counts;               // int32 [3]: S, rows, window ops
} FprRowexpRingOpsArgs;

typedef struct {                       // = pybind expand(slot, row0, nrows, kind, start, replay, cached, counts, ring, dbt, nc, R,
                                       //                 scratch, pos, own, si, sl, paged, start_o, seq, err)
  FprCall c;
  const int32_t* slot; const int32_t* row0; const int32_t* nrows; const int32_t* kind;    // [S_max]
  const int64_t* start; const int64_t* replay; const int64_t* cached;                     // [S_max]
  const int32_t* counts;               // [3]
  const int32_t* ring;                 // int32 [slots, 512, 2]
  const int32_t* dbt; int64_t dbt_stride;   // int32 [slots, P] (row stride in elements)
  const int32_t* nc;                   // int32 [slots]
  int64_t R;                           // bucket rows (0: no launch)
  int64_t scratch;                     // scratch window slot
  int64_t* pos; int64_t* own;          // [>= R]
  int32_t* si;                         // [>= R * 128]
  int32_t* sl;                         // [>= R]
  int64_t* paged;                      // [>= R]
  int64_t* start_o;                    // [S_max]
  int32_t* seq;                        // [>= R]
  int32_t* err;                        // [2]
} FprRowexpExpandArgs;

typedef struct {                       // = pybind advance(slot, kind, nrows, start_res, accepted, counts, nc, s_max)
  FprCall c;
  const int32_t* slot; const int32_t* kind; const int32_t* nrows;   // [S_max]
  const int64_t* start_res;            // [S_max] (rowexp out["start"])
  const int32_t* accepted;             // [S] accepted tokens per sequence (step order)
  const int32_t* counts;               // [3]
  int32_t* nc;                         // [slots]
  int64_t s_max;                       // grid = ceil(s_max / 128) x 128
} FprRowexpAdvanceArgs;

const FprExtTable* fpr_rowexp_table(void);

#ifdef __cplusplus
}
#endif
