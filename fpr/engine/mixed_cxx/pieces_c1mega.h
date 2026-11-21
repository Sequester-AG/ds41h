// pieces_c1mega.h -- the c1mega launch of ONE run of layers
// i.e. Engine._issue's _mega_on branch (C1Mega.run(r, s, L, st)) for the run starting at layer L, without Python:
//   [join: _join_mix(s, pf=False)]; main waits the run's Engram events (estream side work, E0 on the owner);
//   ONE k_layer launch over the run's packed Args (c1mega C ABI "c1mega.run", fpr_c1mega_table() of the loaded package);
//   owner only: the DSpark aux of the run's AUX layers: aux[:, col].copy_(Xo[L].mean(1)) (ATen, main stream).
// One handle per (rank, run-start layer); the layers inside a run have no piece (the persistent kernel runs them). Built by
// the package's c1mega_eng.py (C1Mega.cxx_pieces) from the same packs / buffers / events the Python hook uses, so both
// paths launch the same kernel with the same Args blob on the same stream. Capturable: no host sync, no allocation outside
// ATen (the mean's temporary). C1Idx.prepare is a no-op in the served config (C1IDX_FOLD_BT=1): the builder refuses
// handles otherwise (the Python path then stays in use).
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include "pieces.h"
#include "../../kernels/c1mega/srv7c/c1mega_cabi.h"   // FprC1megaRunArgs (the table's args_size guards the layout)

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_C1M_MAX_EV 6
#define FPR_C1M_MAX_AUX 3

typedef struct {
  int32_t version, layer, nly, owner;         // the run [layer, layer + nly) of this bucket's owner
  int32_t rank, join, n_ev, n_aux;            // join: _join_mix before the launch; n_ev events; n_aux (owner only)
  cudaStream_t main;                          // the rank's main stream (the engine's stream of the decode step)
  cudaEvent_t ev[FPR_C1M_MAX_EV];             // EngramHooks.ev[...] cuda_event handles main waits, in the Python order
  FprC1megaRunArgs run;                       // run.c.stream == main, run.c.device == the rank's device
  void* xo[FPR_C1M_MAX_AUX];                  // owner: bf16 [R, HC, D] contiguous (C1Mega._bufs[r]["Xo"][L])
  int32_t col[FPR_C1M_MAX_AUX];               // aux column (AUX_LAYERS.index(L))
  int32_t R, HC, D, A;                        // aux: bf16 [R, A, D] contiguous (s["aux"])
  void* aux;
} FprPieceC1mega;

#ifdef __cplusplus
}
#endif
