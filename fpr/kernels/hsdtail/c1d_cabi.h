// c1d_cabi.h
// HSDT's c1d switch: the 3 DSpark drafter blocks as ONE
// persistent launch of nly layers per GPU (132 co-resident CTAs, grid barriers + lamport exchanges between the 4 GPUs; every
// rank's launch must be issued: graph H's C1 drafter). The per-layer Args are packed on the GPU by c1d.py DrafterC1M.packs
// (pack_ffn / pack_layer, per owner rank and DraftRank set: built outside any capture). Per-device state: the kernel's max
// dynamic SMEM attribute, set by the first launch on each device (atomic mask). No scratch.
// Table: fpr_c1d_table(); "c1d.run".
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {                       // = pybind run(packed, nly, full)
  FprCall c;                           // c.device: the GPU the packed Args live on (c.rank unused: it is in the Args)
  const void* packed;                  // device: nly x the kernel's Args (pybind args_bytes()), 16-B aligned
  int32_t nly;                         // layers in this launch (the drafter: 3)
  int32_t full;                        // 1: the full-layer kernel variant (k_layer<true>; c1d always), 0: k_layer<false>
} FprC1dRunArgs;

const FprExtTable* fpr_c1d_table(void);

#ifdef __cplusplus
}
#endif
