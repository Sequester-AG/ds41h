// c1mega_cabi.h -- C ABI of the c1mega persistent C1 decode kernel, per
// fpr/engine/mixed_cxx/fpr_cabi.h (stage 4: the C1 bucket's layer run without Python).
// What the engine's C1 step launches per run of layers (Engine._issue's _mega_on branch -> C1Mega.run -> X.run): ONE
// k_layer launch over the run's packed Args (pack_layer / patch_* blobs built at init by c1mega_eng.py, device-resident,
// one blob per (bucket, owner, run, rank)). Everything else in a step is either the engine's (the mHC mix join, the
// Engram event waits: cudaStreamWaitEvent on the caller's side) or ATen (the DSpark aux mean copy after the launch).
// The index phases (c1idx), Engram inject, X12 / LBM / L3 and the expert path run INSIDE k_layer: no other launch.
// Table: fpr_c1mega_table(); "c1mega.run", "c1mega.init".
#pragma once
#include "../../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {                       // = pybind run(packed, nly, full)
  FprCall c;                           // c.device: the packed blob's device (made current by the caller)
  const void* packed;                  // device: nly x sizeof(c1m::Args), 16-B aligned (the run's layers, in order)
  int64_t nly;                         // layers in this launch (1 .. 40)
  int32_t full;                        // 1: whole-layer kernel k_layer<true> (deployment builds: the only one compiled)
  int32_t pad_;
} FprC1megaRunArgs;

typedef struct {                       // per device, once, before any c1mega.run on it (also done lazily by run)
  FprCall c;                           // c.device: the device (made current by the caller); stream unused
  int32_t full;
  int32_t pad_;
} FprC1megaInitArgs;

int64_t fpr_c1mega_args_bytes(void);   // sizeof(c1m::Args): the packed blob's per-layer stride (ABI drift guard)
const FprExtTable* fpr_c1mega_table(void);

#ifdef __cplusplus
}
#endif
