// nd.h -- NativeDecode: engine/d13.py's per-step device ops of an async decode
// step as C++ calls.
//
// Every pointer / handle is resolved by the caller: ND-1 = the Python engine's own buffers and the CURRENT set of each of
// its double-buffered pinned staging areas (so Python's sync / fallback paths and ND share one alternation), plus
// persistent events the caller created; ND-3 = the init registry. Nothing here allocates, and the only host waits are the
// set-reuse event waits Python makes too (each set's previous upload). The caller makes the rank's device current.
// Bitwise contract: the same bytes reach the same device buffers through the same copies / graph launches / ATen ops in
// the same stream order as the Python body (ENG_D13_NATIVE=1 vs 0 harness arms).
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "../d13/d13_host.h"

namespace fpr {
namespace nd {

struct PreCommon {
  int rank = 0, device = 0;
  cudaStream_t stream = nullptr;
  int64_t R = 0;                       // the bucket's rows
  // positions D2H (d13.py: ph[:R] <- s["pos"], oh[:R] <- s["own"], then s["d13_ev"])
  int64_t* ph = nullptr; int64_t* oh = nullptr;            // pinned, set d2h_i, [>= R]
  const int64_t* pos_d = nullptr; const int64_t* own_d = nullptr;
  cudaEvent_t pos_ev = nullptr;
};

// D13._pre's full path for one rank with ENG_D13_GT (graph T): _deltas, Engine._upload_v2(dev_pos=True),
// hooks_idx.upload_d13, the owner's rslot base copy, RowExp.upload, _guard_host + graph T, the positions D2H, Engram's
// step args + upload.
struct PreFull {
  PreCommon c;
  bool hsd_on = true;                  // hooks.hsd is not None: host-given -1 tokens -> 0 (else 1000)
  // _deltas (pinned set dl_i; nu > delta_cap or the index mirror: the caller ran the Python body instead -> skip_deltas)
  bool skip_deltas = false;
  int64_t* dl_rows_h = nullptr; int64_t* dl_cols_h = nullptr; int32_t* dl_pages_h = nullptr;
  cudaEvent_t dl_ev_wait = nullptr, dl_ev = nullptr;
  int64_t* dl_rows_d = nullptr; int64_t* dl_cols_d = nullptr; int32_t* dl_pages_d = nullptr;
  int64_t delta_cap = 0;
  int64_t* n_direct = nullptr;         // (v3.28) +1 per call that took _deltas' direct branch (nu > delta_cap)
  int32_t* dbt = nullptr; int64_t dbt_rows = 0, dbt_cols = 0, dbt_stride0 = 0;   // rowexp's per-slot block table
  // Engine._upload_v2 (dev_pos): the bucket's input blob, pinned set s["in_i"]
  uint8_t* in_h = nullptr; cudaEvent_t in_ev_wait = nullptr, in_ev = nullptr;
  void* in_d = nullptr; int64_t in_bytes = 0, in_off_tok = 0, in_off_valid = 0, in_off_samp = 0;
  // hooks_idx.upload_d13: the indexer blob, pinned set d["up_i"]
  uint8_t* up_h = nullptr; cudaEvent_t up_ev_wait = nullptr, up_ev = nullptr;
  void* up_d = nullptr; int64_t up_bytes = 0, up_off_valid = 0, up_off_pslot = 0, up_off_rs = 0, up_B = 0, pslot_fill = 0;
  // ENG_D13_OWNER: _keep(s, "_own_rbase", d["rslot"]) (the buffer exists: the caller allocates it on first use)
  const void* rslot_d = nullptr; void* rbase_d = nullptr; int64_t rslot_bytes = 0;
  // RowExp.upload: pinned set rx._hi, the device descriptor blocks
  int32_t* rx_h32 = nullptr; int64_t* rx_h64 = nullptr; cudaEvent_t rx_ev_wait = nullptr, rx_ev = nullptr;
  int32_t* rx_b32 = nullptr; int64_t* rx_b64 = nullptr; int64_t S_max = 0, ops_max = 0, R_max = 0;
  // _guard_host (pinned set d2h_i) + graph T (R, rank, has_seqs)
  int64_t* nb_h = nullptr; cudaEvent_t nb_ev_wait = nullptr, nb_ev = nullptr; int64_t* nb_d = nullptr;
  cudaGraphExec_t gt_exec = nullptr;
  // Engram: EG.step_args_from_compact(ct, H._lookback, pad_to=R) + engram[r].upload(a) (async decode rows only: look = -2,
  // no host lookback; a prefill / long row is BAD_ARG): pinned stage set engram._stage_i, its persistent event, _pk
  bool engram = false;
  int32_t* eg_stage_h = nullptr; cudaEvent_t eg_ev = nullptr; int32_t* eg_pk_d = nullptr;
  int64_t eg_pk_n = 0, eg_off_row0 = 0, eg_off_nrows = 0, eg_off_start = 0, eg_off_slot = 0, eg_off_look = 0,
          eg_off_counts = 0, eg_max_rows = 0, eg_max_seqs = 0, eg_ring_w = 0;
};

// D13._owner_skip's device part: graph T' (R, rank, "own", n) or the fused tail (ENG_D13_TFUSE: d13tail.tail through its
// C table, args built by the caller), then the positions D2H.
struct PreOwner {
  PreCommon c;
  cudaGraphExec_t gto_exec = nullptr;  // null: tfuse
  int (*tfuse_fn)(const void* args) = nullptr;
  const void* tfuse_args = nullptr;    // FprD13tailArgs
};

// Status: OK, or BAD_ARG / a CUDA error with *err set (nothing issued after the failing op).
Status pre_full(const PreFull& p, const StepBatch& sb, const CompactTables& ct, const d13::D13Host& h, std::string* err);
Status pre_owner(const PreOwner& p, std::string* err);

}  // namespace nd
}  // namespace fpr
