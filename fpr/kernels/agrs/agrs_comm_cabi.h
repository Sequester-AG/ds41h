// agrs_comm_cabi.h
// Extension agrs_commc (agrs_comm_cabi.cu: agrs.cu
// #included UNCHANGED, its own TU as the served agrs_comm build, + this ABI). Table fpr_agrs_comm_table, ext "agrs_comm".
//
//   agrs_comm.ce_copy  [cudaStreamWaitEvent(stream, wait_before)] -> cudaMemcpyAsync(dst, src, nbytes, D2D, stream)
//                      -> [cudaEventRecord(record_after, stream)].
//     AGCE today (agrs_engine.py p_router / p_ag): the sender records "f ready" on its rank stream, each of its 4 copy
//     streams waits on it and copies f (then the route rows) into the 4 gather slots; after the route-row copy it records
//     one done event per destination, which the receiver's p_moe waits on. In the driver, wait_before = "f ready" (or
//     NULL when the caller already ordered the copy stream), record_after = the destination's done event (or NULL).
//     Both events are the caller's (created once; rule 3: the function creates none, never synchronizes the host).
//   The served pybind ce_copy(dst, dst_off, src, src_off, nbytes, stream) is the same cudaMemcpyAsync (offsets folded
//   into the pointers here).
#pragma once
#include <stdint.h>

#include "fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  FprCall call;            // stream = the copy stream (AGCE: one of the sender's 4 copy streams), device = the sender
  void* dst;               // device pointer (a peer's gather slot: peer access enabled by the caller)
  const void* src;         // device pointer
  int64_t nbytes;          // > 0
  cudaEvent_t wait_before; // NULL = no wait
  cudaEvent_t record_after;// NULL = no record
} AgrsCeCopyArgs;

int agrs_comm_ce_copy(const void* args);   // AgrsCeCopyArgs
const FprExtTable* fpr_agrs_comm_table(void);

#ifdef __cplusplus
}
#endif
