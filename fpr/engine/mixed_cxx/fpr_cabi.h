// fpr_cabi.h -- Shared by every extension the driver calls
//
// RULES (every exported function):
//  1. One plain C struct of arguments per function: raw device pointers (typed or void*), int64_t sizes / strides / row
//     counts, scalars, and the cudaStream_t. No at::Tensor, no std:: types, no Python objects across the boundary.
//  2. The caller owns ALL memory. Nothing inside allocates (no cudaMalloc, no torch allocator, no static device buffers
//     created on first use). A function that needs scratch takes it as (ws, ws_bytes) and has a companion
//     `<name>_ws_bytes(const Args*)` returning the bytes it needs (0 if none). One-time state (lookup tables, cubins,
//     peer pointers) is created by an explicit `<ext>_init(...)` called once per device before any launch.
//  3. The caller sets the current device and passes the stream. A function launches ONLY on that stream, never on the
//     default stream or a stream of its own, never changes the current device, never synchronizes the host (no
//     cudaStreamSynchronize / cudaDeviceSynchronize / blocking cudaMemcpy / cudaEventSynchronize), never creates events.
//     Cross-stream / cross-device ordering is the caller's (events passed in as cudaEvent_t when a function must record
//     or wait one).
//  4. Thread safety: callable concurrently from 4 host threads, one per device. No process-global mutable state: a
//     setting that today is global (DeepGEMM's num_sms, a "mode" module flag) becomes an argument.
//  5. Errors: return int (FPR_OK = 0, negative = FPR_E_*); no exceptions across the boundary; no printing. The launch
//     result is cudaGetLastError() right after the launch, folded into FPR_E_CUDA (the caller checks in debug mode only).
//  6. Same code path as the pybind wrapper: the existing pybind function builds the Args struct from its tensors and calls
//     the SAME C function, so the Python path and the C++ driver launch identical kernels with identical arguments. The
//     pybind signature and behaviour stay unchanged.
//  7. Export: per extension ONE C symbol `const FprExtTable* fpr_<ext>_table(void)` (extern "C", visibility default)
//     returning {abi_version, ext name, n_fns, fns[]: {name, fn pointer, ws_bytes pointer or NULL}}. The driver finds the
//     already loaded .so (dlopen(path, RTLD_NOW | RTLD_NOLOAD), path from the Python module's __file__) and dlsym's the
//     table, so no second copy of the kernels exists in the process.
//  8. GATES before the driver uses an extension: (a) SASS identical: `cuobjdump -sass` of every kernel in the new .so ==
//     the served .so (only host code changed); (b) the pybind path's outputs bitwise == the served build on the
//     extension's own test; (c) the C path (via the table, from the C++ test harness fpr/engine/mixed_cxx/cabi_test)
//     bitwise == the pybind path for the shapes of the served mixed buckets (R = 64 .. 2,432 rows per rank, T <= 1,536).
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_CABI_VERSION 1

enum {
  FPR_OK = 0,
  FPR_E_ARG = -1,       // invalid / inconsistent arguments
  FPR_E_SHAPE = -2,     // shape not supported by this kernel (the caller must take another path)
  FPR_E_WS = -3,        // workspace too small
  FPR_E_INIT = -4,      // <ext>_init not called on this device
  FPR_E_CUDA = -5,      // a CUDA error at launch (cudaGetLastError)
};

typedef int (*FprFn)(const void* args);            // args = pointer to the function's own Args struct (rule 1)
typedef int64_t (*FprWsFn)(const void* args);       // bytes of scratch the function needs for these args (rule 2)

typedef struct {
  const char* name;      // "<ext>.<function>", e.g. "mhcf.frow"
  FprFn fn;
  FprWsFn ws_bytes;      // NULL = no scratch
  int64_t args_size;     // sizeof(the Args struct): the driver checks it against its own header (ABI drift guard)
} FprFnEntry;

typedef struct {
  int abi_version;       // FPR_CABI_VERSION the extension was built against
  const char* ext;       // extension name
  const char* src_md5;   // md5 of the kernel source(s) it was built from
  int n_fns;
  const FprFnEntry* fns;
} FprExtTable;

typedef const FprExtTable* (*FprTableFn)(void);

// Common argument conventions (rule 1): every Args struct starts with this header.
typedef struct {
  cudaStream_t stream;   // the stream to launch on (rule 3)
  int device;            // the device the caller made current (checked in debug builds)
  int rank;              // 0..3 (for kernels that address peers)
} FprCall;

#ifdef __cplusplus
}
#endif
