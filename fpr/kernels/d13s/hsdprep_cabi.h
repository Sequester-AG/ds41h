// hsdprep_cabi.h -- C ABI of kernels/d13s/hsdprep.cu, per fpr/engine/mixed_cxx/fpr_cabi.h.
// D13 steady decode (ENG_D13S): HSD's position-dependent per-step tables derived on the device (one CTA), in place of
// HSD's host prepare + blob upload + positions D2H. Table: fpr_hsdprep_table(); "hsdprep.hsdprep" (the pybind
// hsdprep() builds this struct and calls the same core).
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {                       // = pybind hsdprep(cs, ce, g_dslot, g_p0, pos, num_sampled, q0, D_pad, wbase, wpages,
                                       //                  lslot, ring, window, tpos, tslot, nt, rx_pos, rx_own, err)
  FprCall c;
  const int32_t* cs; const int32_t* ce;   // [WD]
  const int64_t* g_dslot;                 // [WD]
  int64_t* g_p0;                          // [WD] (in place)
  int64_t* pos;                           // [rows]
  const int32_t* num_sampled;             // [>= WD]
  int32_t WD, q0, D_pad;
  int64_t* wbase;                         // [>= D_pad]
  int64_t* wpages; int32_t nbw;           // [>= D_pad, nbw]
  const int64_t* lslot;                   // [>= D_pad]
  const int32_t* ring;                    // int32 [slots, 512, 2] (rowexp's ring)
  int32_t window;
  int64_t* tpos; int64_t* tslot; int32_t nt;   // [>= nt]
  const int64_t* rx_pos; const int64_t* rx_own; // [>= nt]
  int32_t* err;
} FprHsdprepArgs;

const FprExtTable* fpr_hsdprep_table(void);

#ifdef __cplusplus
}
#endif
