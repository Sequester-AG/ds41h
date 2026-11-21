// pagecopy_cabi.h -- C ABI of the pagecopy extension (pagecopy.cu) for the native mixed step's page copies, under fpr/engine/mixed_cxx/fpr_cabi.h's rules.
//
// Table: fpr_pagecopy_table() -> ext "pagecopy": "pagecopy.copy" (no init: the kernel uses no dynamic shared memory).
//   copy: dst[dp[i]] = src[sp[i]] for i < n whole pages of page_bytes (a positive multiple of 16), 16 B per load / store, on
//   call.stream only (the pybind path's grid: x = clamp(ceil(words / 1,024), 1, 64) x 256 threads, y = pages in chunks of
//   65,535), no launch when n == 0. sp / dp: int64 page ids in device memory of call.device (the caller stages them: a
//   pinned set + H2D, as the pybind path's pin + non_blocking copy). No allocation, no device change, no sync. Returns FPR_OK,
//   FPR_E_ARG (the pybind path's checks: page_bytes > 0 and % 16 == 0, 16-B aligned pool bases, n >= 0, non-NULL lists), or
//   FPR_E_CUDA (cudaGetLastError after the launches, as the pybind path's launch check).
#pragma once
#include "../../engine/mixed_cxx/fpr_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  FprCall call;
  const void* src;       // the source pool's base (device or host-mapped; peer-readable from call.device)
  void* dst;             // the destination pool's base
  const int64_t* sp;     // [n] source page ids (device memory of call.device)
  const int64_t* dp;     // [n] destination page ids
  int64_t n;
  int64_t page_bytes;
} FprPagecopyArgs;

#ifdef __cplusplus
}
#endif
