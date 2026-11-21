// idx2f_tab_cabi.h -- C ABI of idx2f_tab.cu, rules of fpr/engine/mixed_cxx/fpr_cabi.h. Table fpr_idx2f_tab_table() -> ext "idx2f_tab"
//   "idx2f_tab.wait"  host wait for ring slot `slot`'s event (its previous H2D + kernels): call before rewriting the slot
//                     (the one host synchronization of this table, by design: tables2f.Stage.acquire's)
//   "idx2f_tab.run"   = the pybind run() on an EXPLICIT stream: one cudaMemcpyAsync of the staging words (host -> dev
//                     block), scatter + derive / general kernels, the slot's event recorded after them
// The caller makes call.device current (fpr_cabi rule 3; a mismatch is FPR_E_ARG). hid / ring_id are the extension's own
// registrations (Python create() / ring(): tables2f.GeneralTables / Stage; the
// handover exports them). Same launches as the pybind entry (which now calls the same function on the current stream).
// FPR_OK / FPR_E_ARG (bad handle / slot / device) / FPR_E_CUDA.
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  FprCall call;
  int64_t ring_id, slot;
} FprIdx2fTabWaitArgs;

typedef struct {
  FprCall call;                 // stream: the rank's stream; device: the tables' device
  int64_t hid, ring_id, slot;
  const void* host;             // pinned int32 staging block (tables2f.layout words)
  void* dev;                    // device int32 block of the same slot
  int64_t n, S, U, P, B;        // rows, sequences, delta records, delta pages, pseudo-requests (general; 0 for decode)
} FprIdx2fTabRunArgs;

const FprExtTable* fpr_idx2f_tab_table(void);

#ifdef __cplusplus
}
#endif
