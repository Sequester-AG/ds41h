// head_comm_cabi.h -- C ABI of kernels/head/vendor/comm.cu, the HSD head's head_comm, per
// fpr/engine/mixed_cxx/fpr_cabi.h. vendor/comm.cu = sp_layer/comm.cu before peer_copy (same kernels, same launches), so
// every entry takes sp_layer/comm_cabi.h's struct: ag_push (the head's PeerGather sites, hsd.py / comm_head.py:
// FprCommAgPushArgs), ag_push2 / wait / rs_push / rs_sum (the drafter's PeerComm, drafter.py). Peers: one device pointer
// per rank (W = 4) for every gather / receive buffer and flag; peer access must be on (Python enable_peers(W) at load: it
// walks all devices and changes the current device, so it is no per-device init and stays Python; no "head_comm.init").
// Flags / epochs: uint32 counters, never reset.
// Table: fpr_head_comm_table(); "head_comm.ag_push", "head_comm.ag_push2", "head_comm.wait", "head_comm.rs_push",
// "head_comm.rs_sum".
#pragma once
#include "../../sp_layer/comm_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

const FprExtTable* fpr_head_comm_table(void);

#ifdef __cplusplus
}
#endif
