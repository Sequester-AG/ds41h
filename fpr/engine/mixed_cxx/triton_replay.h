// triton_replay.h -- launch Triton's OWN compiled kernels from C++ (no Python, no AOT, no second copy): the C++ mixed-step
// driver's path for the vLLM / engine Triton kernels.
//
// A FprTrRecord is made on the Python thread at warm-up by triton_replay.py from one real Triton launch: the CompiledKernel's
// CUfunction ON THAT DEVICE (Triton loads its module per device context), num_warps / num_ctas / shared memory /
// launch_pdl / launch_cooperative_grid, the global- and profile-scratch sizes, and the compiled parameter list (the
// Python-level arguments that are not constexpr, in launch order) with each one's compiled type and specialization.
// fpr_tr_check: the values the driver is about to pass must fall in the record's key class, by Triton's own rules
//   (probed on 3.7.1): ints are i32 if in [-2^31, 2^31), else i64 (< 2^63), else u64; unless do_not_specialize: == 1 is
//   constexpr 1 (a different compiled kernel), else 'D' if % 16 == 0 (alignment specialization on) or '' ; pointers:
//   'D' if 16-B aligned (alignment on) else ''; bools u1 / floats fp32 / None: not specialized. Mismatch: FPR_E_SHAPE (the
//   step falls back to Python: the pre-step warms every reachable class, so this is a coverage bug, never a compile).
// fpr_tr_launch: packs the parameters exactly as Triton's launcher (each compiled parameter by its type size, then the
//   global_scratch and profile_scratch pointers) and calls cuLaunchKernelEx with Triton's config: grid x * num_ctas,
//   block 32 * num_warps, the shared memory, attributes PDL (launch_pdl) / cooperative / cluster (num_ctas != 1: cluster
//   x = num_ctas + spread scheduling); no launch for an empty grid. Launch enter / exit hooks: none (off on the served
//   path). num_ctas == 16's non-portable-cluster function attribute is set at record time (triton_replay.py), not here.
// Values: uint64_t vals[] indexed by the Python-level argument position (pointers as addresses, ints as two's complement,
//   fp32 / fp16 / bf16 / fp64 as their bit patterns in the low bytes, exactly what Triton's extractors store).
#pragma once
#include <cuda.h>
#include <stdint.h>
#include <string.h>

#include "fpr_cabi.h"
#include "fpr_cu_launch.h"   // (cuLaunchKernelEx from cudart: no -lcuda in the driver)

#ifdef __cplusplus
extern "C" {
#endif

#define FPR_TR_MAX_PARAMS 64

enum {   // compiled parameter types (Triton signature strings)
  FPR_TR_PTR = 1,   // "*<dtype>"
  FPR_TR_I1,        // "i1" (4 bytes, Triton's extractor table)
  FPR_TR_I8,
  FPR_TR_I16,
  FPR_TR_I32,
  FPR_TR_I64,
  FPR_TR_U1,        // "u1" (bool: 4 bytes)
  FPR_TR_U8,
  FPR_TR_U16,
  FPR_TR_U32,
  FPR_TR_U64,
  FPR_TR_FP16,      // 2 bytes (bits)
  FPR_TR_BF16,      // 2 bytes (bits)
  FPR_TR_FP32,      // 4 bytes (bits)
  FPR_TR_FP64,      // 8 bytes (bits)
};

enum {   // specialization of a compiled parameter (Triton's second tuple element)
  FPR_TR_SPEC_NONE = 0,    // None: not specialized (do_not_specialize, bools, floats)
  FPR_TR_SPEC_EMPTY = 1,   // ''
  FPR_TR_SPEC_D = 2,       // 'D' (divisible by 16 / 16-B aligned)
};

typedef struct {
  int32_t type;    // FPR_TR_*
  int32_t spec;    // FPR_TR_SPEC_* of the record
  int32_t align;   // 1: alignment specialization on (not do_not_specialize_on_alignment)
  int32_t dns;     // 1: do_not_specialize
  int32_t pyarg;   // Python-level argument index (into vals)
  int32_t pad;
} FprTrParam;

typedef struct {
  CUfunction function;       // CompiledKernel.function on `device`
  int32_t device;
  int32_t num_warps;
  int32_t num_ctas;
  int32_t shared;
  int32_t launch_pdl;
  int32_t launch_coop;
  int64_t gscratch_size;     // bytes per CTA (0: none); the driver passes grid * num_ctas * size bytes, aligned
  int64_t gscratch_align;
  int64_t pscratch_size;
  int64_t pscratch_align;
  int32_t n_params;
  int32_t n_const1;          // Python-level ints specialized to constexpr 1 in this record: they must still be 1
  FprTrParam p[FPR_TR_MAX_PARAMS];
  int32_t const1_pyarg[FPR_TR_MAX_PARAMS];
  char name[96];             // kernel name
  char key[160];             // readable key class (dump / logs)
} FprTrRecord;

static inline int fpr_tr_type_size(int32_t t) {
  switch (t) {
    case FPR_TR_PTR: case FPR_TR_I64: case FPR_TR_U64: case FPR_TR_FP64: return 8;
    case FPR_TR_I1: case FPR_TR_I32: case FPR_TR_U1: case FPR_TR_U32: case FPR_TR_FP32: return 4;
    case FPR_TR_I16: case FPR_TR_U16: case FPR_TR_FP16: case FPR_TR_BF16: return 2;
    case FPR_TR_I8: case FPR_TR_U8: return 1;
    default: return 0;
  }
}

// Triton's int type of a Python int (signed view of the 64-bit value; u64 needs the caller's flag: values >= 2^63)
static inline int32_t fpr_tr_int_type(int64_t v) {
  return (v >= -(INT64_C(1) << 31) && v < (INT64_C(1) << 31)) ? FPR_TR_I32 : FPR_TR_I64;
}

// FPR_OK when vals fall in the record's key class (see the header comment), else FPR_E_SHAPE
static inline int fpr_tr_check(const FprTrRecord* r, const uint64_t* vals) {
  for (int i = 0; i < r->n_const1; ++i)
    if (static_cast<int64_t>(vals[r->const1_pyarg[i]]) != 1) return FPR_E_SHAPE;
  for (int i = 0; i < r->n_params; ++i) {
    const FprTrParam& p = r->p[i];
    const uint64_t v = vals[p.pyarg];
    if (p.type == FPR_TR_PTR) {
      const int32_t s = (p.dns || !p.align) ? (p.dns ? FPR_TR_SPEC_NONE : FPR_TR_SPEC_EMPTY) : (v % 16 == 0 ? FPR_TR_SPEC_D : FPR_TR_SPEC_EMPTY);
      if (s != p.spec) return FPR_E_SHAPE;
    } else if (p.type == FPR_TR_I32 || p.type == FPR_TR_I64) {
      const int64_t x = static_cast<int64_t>(v);
      if (fpr_tr_int_type(x) != p.type) return FPR_E_SHAPE;
      if (!p.dns && x == 1) return FPR_E_SHAPE;   // would be constexpr 1: another compiled kernel
      const int32_t s = p.dns ? FPR_TR_SPEC_NONE : (p.align && x % 16 == 0) ? FPR_TR_SPEC_D : FPR_TR_SPEC_EMPTY;
      if (s != p.spec) return FPR_E_SHAPE;
    }
  }
  return FPR_OK;
}

// one launch of the record's kernel on `stream` (current context = the record's device): FPR_OK / FPR_E_ARG / FPR_E_CUDA
static inline int fpr_tr_launch(const FprTrRecord* r, const uint64_t* vals, uint32_t gx, uint32_t gy, uint32_t gz,
                                CUstream stream, CUdeviceptr gscratch, CUdeviceptr pscratch) {
  if (r == NULL || r->function == NULL || r->n_params > FPR_TR_MAX_PARAMS) return FPR_E_ARG;
  if (static_cast<uint64_t>(gx) * gy * gz == 0) return FPR_OK;   // Triton's launcher: nothing for an empty grid
  if ((r->gscratch_size > 0 && gscratch == 0) || (r->pscratch_size > 0 && pscratch == 0)) return FPR_E_ARG;
  uint64_t store[FPR_TR_MAX_PARAMS];
  void* params[FPR_TR_MAX_PARAMS + 2];
  for (int i = 0; i < r->n_params; ++i) {
    const int sz = fpr_tr_type_size(r->p[i].type);
    if (sz == 0) return FPR_E_ARG;
    store[i] = 0;
    memcpy(&store[i], &vals[r->p[i].pyarg], sz);   // the value's low bytes (little endian), as Triton's extractors
    params[i] = &store[i];
  }
  params[r->n_params] = &gscratch;
  params[r->n_params + 1] = &pscratch;
  CUlaunchAttribute attr[4];
  int na = 0;
  if (r->launch_pdl) {
    attr[na].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attr[na].value.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  if (r->launch_coop) {
    attr[na].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attr[na].value.cooperative = 1;
    ++na;
  }
  if (r->num_ctas != 1) {
    memset(&attr[na], 0, sizeof(attr[na]));
    attr[na].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attr[na].value.clusterDim.x = r->num_ctas;
    attr[na].value.clusterDim.y = 1;
    attr[na].value.clusterDim.z = 1;
    ++na;
    memset(&attr[na], 0, sizeof(attr[na]));
    attr[na].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    attr[na].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    ++na;
  }
  CUlaunchConfig cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.gridDimX = gx * r->num_ctas;
  cfg.gridDimY = gy;
  cfg.gridDimZ = gz;
  cfg.blockDimX = 32 * r->num_warps;
  cfg.blockDimY = 1;
  cfg.blockDimZ = 1;
  cfg.sharedMemBytes = r->shared;
  cfg.hStream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = na;
  return fpr_cu_launch_kernel_ex(&cfg, r->function, params, 0) == CUDA_SUCCESS ? FPR_OK : FPR_E_CUDA;
}

#ifdef __cplusplus
}
#endif
