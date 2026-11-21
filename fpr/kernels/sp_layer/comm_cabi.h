// comm_cabi.h -- C ABI of kernels/sp_layer/comm.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// The mixed step's sequence-parallel collectives (Engine.p_ag / p_rs, cp.py): ag_push2 + wait (f rows + 64-B routing rows
// all-gather), rs_push + rs_sum (MoE partial reduce-scatter), peer_copy (copy-engine copy on the caller's stream).
// Peers: one device pointer per rank (W = 4) for every gather / receive buffer and flag; peer access must be on (the Python
// enable_peers(W) at load: it walks all devices and changes the current device, so it is not a per-device init and stays
// Python; no "comm.init"). Flags / epochs: uint32 counters, never reset (comm.cu). c.rank = the pushing rank.
// Table: fpr_comm_table(); "comm.ag_push2", "comm.wait", "comm.rs_push", "comm.rs_sum", "comm.peer_copy", "comm.ag_push"
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_COMM_W 4

typedef struct {                       // = pybind ag_push2(src, src2, rank, bufs, flags, bufs2, epoch, ncta)
  FprCall c;                           // c.rank: this rank (its slot in every gather buffer)
  const void* src;                     // [R, RB bytes] f rows
  const void* src2;                    // [R, RB2 bytes] routing rows
  int32_t R;
  int32_t RB;                          // % 16 == 0
  int32_t RB2;                         // % 16 == 0
  void* bufs[FPR_COMM_W];              // f gather buffer [W * R, RB] on each rank
  void* bufs2[FPR_COMM_W];             // routing gather buffer [W * R, RB2] on each rank
  uint32_t* flags[FPR_COMM_W];         // f flag on each rank (one signal per CTA covers both row sets)
  uint32_t* epoch;                     // this rank's AG round counter (bumped by the kernel)
  int32_t ncta;                        // CTAs per destination (grid W * ncta)
} FprCommAgPush2Args;

typedef struct {                       // = pybind wait(flag, epoch, signals)
  FprCall c;
  const uint32_t* flag;
  const uint32_t* epoch;
  uint32_t signals;                    // per round (W * ncta)
} FprCommWaitArgs;

typedef struct {                       // = pybind rs_push(part, rank, bufs, flags, epoch, ncta)
  FprCall c;                           // c.rank: this rank
  const void* part;                    // bf16 [W * R, H] (all rows)
  int32_t R;                           // rows per rank (part.size(0) / W)
  int32_t H;
  void* bufs[FPR_COMM_W];              // receive buffer bf16 [W, R, H] on each rank
  uint32_t* flags[FPR_COMM_W];
  uint32_t* epoch;                     // this rank's RS round counter (bumped by the kernel)
  int32_t ncta;                        // CTAs per destination (grid W * ncta)
} FprCommRsPushArgs;

typedef struct {                       // = pybind rs_sum(recv, flag, epoch, signals, out, ncta)
  FprCall c;
  const void* recv;                    // bf16 [W, R, H] (slot = source rank)
  int32_t R;
  int32_t H;
  const uint32_t* flag;
  const uint32_t* epoch;
  uint32_t signals;
  void* out;                           // bf16 [R, H]
  int32_t ncta;                        // grid
} FprCommRsSumArgs;

typedef struct {                       // = pybind peer_copy(dst, src): cudaMemcpyAsync(cudaMemcpyDefault) on c.stream
  FprCall c;
  void* dst;
  const void* src;
  int64_t bytes;
} FprCommPeerCopyArgs;

typedef struct {                       // = pybind ag_push(src, rank, bufs, flags, epoch, ncta)
  FprCall c;                           // c.rank: this rank (its slot in every gather buffer)
  const void* src;                     // [R, RB bytes] rows
  int32_t R;
  int32_t RB;                          // % 16 == 0
  void* bufs[FPR_COMM_W];              // gather buffer [W * R, RB] on each rank
  uint32_t* flags[FPR_COMM_W];         // flag on each rank (one signal per CTA)
  uint32_t* epoch;                     // this rank's round counter (bumped by the kernel)
  int32_t ncta;                        // CTAs per destination (grid W * ncta); the waiter counts W * ncta signals
} FprCommAgPushArgs;

const FprExtTable* fpr_comm_table(void);

#ifdef __cplusplus
}
#endif
