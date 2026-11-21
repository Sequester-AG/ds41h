// xg_cabi.h -- C ABI of kernels/hsdtail/xg.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// HSD's small all-gathers (sampler words g1 / g2 / g3 / d0..d2, the h / hd rows at C1 sizes) as ONE lamport kernel per rank:
// stores into every rank's mailbox, spin on the own one, copy out [W, n]. Mailboxes [2][W][nmax] int32 per rank and the
// call counter ctr[2] are the caller's (HSDT.configure(xg=True): torch tensors, never reset: the protocol is replay-safe and
// capture-safe). Peer access must be on (HSD's PeerGather init). c.rank = this rank. No init, no scratch.
// Table: fpr_xg_table(); "xg.xgather".
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_XG_W 4

typedef struct {                       // = pybind xgather(src, out, box, peers, rank, ctr)
  FprCall c;                           // c.rank: this rank (its slot in every mailbox)
  const void* src;                     // [n] 32-bit words (any 4-byte type)
  void* out;                           // [W, n] words (the gathered buffer)
  int32_t n;                           // words; 0 <= n <= nmax
  int32_t nmax;                        // words per (parity, slot) of a mailbox = box words / (2 * W)
  void* box;                           // this rank's mailbox, int32 [2 * W * nmax]
  void* peers[FPR_XG_W];               // every rank's mailbox (the same tensors' device pointers)
  int32_t* ctr;                        // this rank's call counter, int32 [2]
} FprXgGatherArgs;

const FprExtTable* fpr_xg_table(void);

#ifdef __cplusplus
}
#endif
