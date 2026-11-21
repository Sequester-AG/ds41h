// driver_api.h -- what piece files of other owners use from the
// driver. A piece file defines `void fpr_pieces_<owner>_register(pybind11::module_& m)` (its pybind
// entries: add_<kind>(bytes) -> handle, <kind>(rank, handle, ...) -> FPR status) and is compiled into the same extension
// (mixed_cxx.py picks up every pieces_*.cpp next to driver.cpp). Shared per-rank state lives in the driver.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "fpr_cabi.h"

// an extension's function by name, checked against the caller's Args size (nullptr if the table is not registered)
FprFn fpr_drv_fn(const char* ext, const char* name, int64_t args_size);
// per-rank reusable events of the driver: 0 = boundary handoff, 1 = boundary mixes (the pending mix of _join_mix)
cudaEvent_t fpr_drv_event(int rank, int which);
// _join_mix for a rank on a stream (waits the pending boundary mixes, if any); every piece that reads the mixes' outputs
// (pre / post / comb) must call it first, as the Python piece calls self._join_mix(s)
int fpr_drv_join(int rank, cudaStream_t stream);
int fpr_drv_device(int rank);
// AGCE (agrs copy-engine AG, FPR's ag piece): this rank's MoE waits the done events of every sender that copied its f /
// route rows into this rank's gather slots since the last call (sender order), then clears them. 1b-i: the Python p_moe
// wrapper (mixed_cxx.install) calls it before the moe piece; 1d: the moe piece calls it first (instead of
// StepParams.moe_wait).
int fpr_drv_agce_join(int rank, cudaStream_t stream);
// stage 1d: an owner's piece as a C function the driver threads call: fn(rank, handle, layer) -> FPR status. Register it
// in fpr_pieces_<owner>_register (module init); returns the kind id (>= FPR_K_OWNER0) the spans use. The function runs on
// rank `rank`'s driver thread (device set, no GIL): no Python, no host sync; exceptions are caught by the driver.
typedef int (*FprPieceFn)(int rank, int64_t handle, int layer);
// the driver's boundary piece (FprPieceBd handle) with this rank's StepParams.chain_input (owners' pieces that end in a
// boundary: FPR's engram_bd)
int fpr_drv_bd(int rank, int64_t handle);
int fpr_drv_register_piece(const char* kind, FprPieceFn fn);
// this rank's StepParams of the current bucket segment (set by Python before the segment's pieces run)
const FprStepParams* fpr_drv_step_params(int rank);

// ---- C exports for the native mixed step (core2b dlopens the loaded fpr_mixed_cxx .so with RTLD_NOLOAD
// and dlsyms them; the pybind entries set_step_params / run_span / run_span_serial / owner_kind / last_error call the same
// bodies). No GIL, no Python. Spans are serialized (one at a time, whichever thread). Status: FPR_OK, FPR_E_ARG, FPR_E_INIT
// (set_streams not called), FPR_E_DEAD (-103: an earlier span failed; every later span is refused), else the first failing
// rank's status (rc_per_rank[W] has all; text: fpr_drv_last_error(rank), valid until the next span starts).
#ifdef __cplusplus
extern "C" {
#endif
int fpr_drv_set_step_params(int rank, const FprStepParams* sp);
int fpr_drv_run_span(int n_ranks, const FprProgEntry* const* progs, const int64_t* n_entries, int* rc_per_rank);
int fpr_drv_run_span_serial(int n_ranks, const FprProgEntry* const* progs, const int64_t* n_entries, int* rc_per_rank);
int32_t fpr_drv_owner_kind(const char* kind);   // -1 = not registered
const char* fpr_drv_last_error(int rank);
#ifdef __cplusplus
}
#endif
