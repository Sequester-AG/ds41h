// pm_rs_lt.cpp: the HOST translation unit of the agrs_pmws_c .so that holds
// fpr_lt.h. It is a separate .cpp because
// nvcc's front end (cudafe++) crashes on fpr_lt.h inside a .cu TU ("internal error ... lexical.c ... find_allocated_name_reference",
// reproduced with a 3-line .cu). pm_rs_cabi.cu calls agrs_lt_mm() for its two shared-expert GEMMs (pool threads).
// ONE fpr_lt state for this .so; FPR's mixed_cxx_ltwarm warms it through the C entry fpr_agrs_lt_warm (+ fpr_agrs_lt_warm_dump).
#include <torch/extension.h>

#include "fpr_lt.h"

FPR_LT_DEFINE_WARM_ENTRY(fpr_agrs_lt_warm)

// y = a @ w^T (w un-transposed [N, K]) = at::mm_out(y, a, w.t()); site = the caller's label in fpr_lt's key dump
void agrs_lt_mm(at::Tensor& y, const at::Tensor& a, const at::Tensor& w, const char* site) { fpr_lt::mm(y, a, w, site); }
