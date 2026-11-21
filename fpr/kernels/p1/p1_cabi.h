// p1_cabi.h -- C ABI of p1_kernels.cu, rules of fpr/engine/mixed_cxx/fpr_cabi.h. Table fpr_p1_table() -> ext "p1": "p1.pf_wbt", "p1.pf_inv"
// "p1.to_local", "p1.carry". Pure index kernels (bit-exact by construction); one stream each (args->call.stream); no
// scratch, no init. FPR_OK / FPR_E_ARG / FPR_E_CUDA.
//   pf_wbt   attn_prefill.Plan's window block table (ENG_MIXED_PRESTEP): per prefill sequence s (rows row0 .. row0 + n - 1,
//            positions first .. first + n - 1, window start g0): wbt[s, :wmax] = 0, then for p in [g0, first + n):
//            slot = p <= first ? si[row0, p - g0] : si[row0 + p - first, sl[row0 + p - first] - 1];
//            wbt[s, p / 32] = slot / 32                       (= index_put_ of the cat(head, tail) slots // WPT)
//   pf_inv   Plan.inv: inv[0 .. n_pool + 1) = -1, then inv[bt[s, j]] = j for j < nb[s] (prefix-shared pages: same j)
//   to_local attn_prefill.to_local: loc[i, k] = g >= 0 && inv[g / rpp] >= 0 ? inv[g / rpp] * rpp + g % rpp : -1,
//            g = ti[i, k], rpp = 64 / ratio
//   carry    tables2f.carry_candidates_dev: T pseudo-request rows of 4 (ir2row_T int64, valid_T u8): fill = valid ? row :
//            the group's last valid row; cand_T[i] = cand_A[row2ir_A[ix[fill]]] (2,048 int32 per row)
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  FprCall call;
  const int32_t* si;          // [rows, ld_si] window slots (engine s["si"], ld 128)
  int64_t ld_si;
  const int32_t* sl;          // [rows] window lengths
  const int64_t* seq;         // [S, 4] int64: row0, n, first, g0
  int64_t S;
  int32_t* wbt;               // [S, ld_wbt] out
  int64_t ld_wbt, wmax;       // columns zeroed / written per sequence (wmax <= ld_wbt)
  int64_t wpt;                // window page tokens (32)
  // (v0.6 blob mode, seq == NULL: the plan's device blob, engine_handover.h FprEhMixedPf.blob) the same four values from
  // row0 = row_lo + qsl[s], n = qsl[s + 1] - qsl[s], first = seq_lens[s] - n, g0 = seq_lens[s] - gather_lens[s]
  const int32_t* seq_lens;    // [S] ends
  const int32_t* gather_lens; // [S] ends - g0
  const int32_t* qsl;         // [S + 1]
  int64_t row_lo;
} FprP1PfWbtArgs;

typedef struct {
  FprCall call;
  int32_t* inv;               // [n_inv] out (n_pool_pages + 1)
  int64_t n_inv;
  const int32_t* bt;          // [S, ld_bt] block tables (the plan's bt)
  int64_t ld_bt;
  const int32_t* nb;          // [S] blocks per sequence, or NULL: nb[s] = ceil(seq_lens[s] / 64) (v0.6 blob mode)
  int64_t S, max_nb;          // max_nb >= every nb[s] (grid)
  const int32_t* seq_lens;    // [S] (blob mode)
} FprP1PfInvArgs;

typedef struct {
  FprCall call;
  const int32_t* ti;          // [n, 512] global top-k slots (-1 none)
  const int32_t* inv;         // [n_inv]
  int64_t n_inv;
  int32_t* loc;               // [n, 512] out
  int64_t n, rpp;             // rows; records per page (64 / ratio)
} FprP1ToLocalArgs;

typedef struct {
  FprCall call;
  const int64_t* ir2row_T;    // [IR_T] (T rows of the T indexer rows; padding rows 0)
  const uint8_t* valid_T;     // [IR_T] bool
  const int64_t* ix;          // [m] T row -> A row
  const int64_t* row2ir_A;    // [R_A] A row -> A indexer row
  const int32_t* cand_A;      // [*, 2048]
  int32_t* cand_T;            // [IR_T, 2048] out
  int64_t IR_T;               // multiple of 4
} FprP1CarryArgs;

const FprExtTable* fpr_p1_table(void);

#ifdef __cplusplus
}
#endif
