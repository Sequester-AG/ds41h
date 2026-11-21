// agrs_comm_cabi.cu -- extension agrs_commc: agrs.cu #included UNCHANGED (its own translation unit
// as the served agrs_comm build of commbench.ext(): same CFLAGS) + the C ABI of agrs_comm_cabi.h + thin pybind callers.
// Device code = the served build's. agrs.cu's PYBIND11_MODULE is renamed (push_macro) to
// comm_bind and called from this module's init, so the served bindings (ce_copy, combine_push_dm, ...) are all there.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "agrs_comm_cabi.h"
#include "fpr_cabi.h"

#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) static void comm_bind(pybind11::module_& variable)
#include "agrs.cu"
#pragma pop_macro("PYBIND11_MODULE")

#if __has_include("agrs_commc_md5.h")
#include "agrs_commc_md5.h"   // written by agrs_comm_cabi.py into the build dir
#endif
#ifndef AGRS_COMMC_SRC_MD5
#define AGRS_COMMC_SRC_MD5 "unset"
#endif

extern "C" {

int agrs_comm_ce_copy(const void* p) {
  const auto* a = static_cast<const AgrsCeCopyArgs*>(p);
  if (a == nullptr || a->dst == nullptr || a->src == nullptr || a->nbytes <= 0) return FPR_E_ARG;
  if (a->wait_before != nullptr && cudaStreamWaitEvent(a->call.stream, a->wait_before, 0) != cudaSuccess) return FPR_E_CUDA;
  if (cudaMemcpyAsync(a->dst, a->src, (size_t)a->nbytes, cudaMemcpyDeviceToDevice, a->call.stream) != cudaSuccess) return FPR_E_CUDA;
  if (a->record_after != nullptr && cudaEventRecord(a->record_after, a->call.stream) != cudaSuccess) return FPR_E_CUDA;
  return FPR_OK;
}

static const FprFnEntry kCommFns[] = {
    {"agrs_comm.ce_copy", agrs_comm_ce_copy, nullptr, int64_t(sizeof(AgrsCeCopyArgs))},
};
static const FprExtTable kCommTable = {FPR_CABI_VERSION, "agrs_comm", AGRS_COMMC_SRC_MD5, 1, kCommFns};

__attribute__((visibility("default"))) const FprExtTable* fpr_agrs_comm_table(void) { return &kCommTable; }

}  // extern "C"

namespace {
// the served ce_copy's arguments (offsets, an int64 stream), through the C function
void py_ce_copy(torch::Tensor dst, int64_t dst_off, torch::Tensor src, int64_t src_off, int64_t nbytes, int64_t stream,
                int64_t wait_before, int64_t record_after) {
  AgrsCeCopyArgs a{};
  a.call.stream = (cudaStream_t)stream, a.call.device = int(src.get_device()), a.call.rank = 0;
  a.dst = (char*)dst.data_ptr() + dst_off, a.src = (const char*)src.data_ptr() + src_off, a.nbytes = nbytes;
  a.wait_before = (cudaEvent_t)wait_before, a.record_after = (cudaEvent_t)record_after;
  const int rc = agrs_comm_ce_copy(&a);
  TORCH_CHECK(rc == FPR_OK, "agrs_comm.ce_copy: error ", rc);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  comm_bind(m);   // agrs.cu's module body (the served bindings)
  m.def("cabi_ce_copy", &py_ce_copy, "agrs_comm.ce_copy through the C ABI (events: cudaEvent_t as int, 0 = none)");
  m.def("cabi_args_size", []() { return int64_t(sizeof(AgrsCeCopyArgs)); });
}
