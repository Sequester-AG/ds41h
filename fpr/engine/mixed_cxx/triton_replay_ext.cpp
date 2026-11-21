// triton_replay_ext.cpp
#include <torch/extension.h>

#include <cstring>
#include <string>

#include "triton_replay.h"

static const FprTrRecord* rec_of(const std::string& b) {
  TORCH_CHECK(b.size() == sizeof(FprTrRecord), "record bytes ", b.size(), " != sizeof(FprTrRecord) ", sizeof(FprTrRecord));
  return reinterpret_cast<const FprTrRecord*>(b.data());
}

static int64_t check(py::bytes record, py::bytes vals) {
  const std::string r = record, v = vals;
  return fpr_tr_check(rec_of(r), reinterpret_cast<const uint64_t*>(v.data()));
}

static int64_t launch(py::bytes record, py::bytes vals, int64_t gx, int64_t gy, int64_t gz, int64_t stream, int64_t gscratch,
                      int64_t pscratch) {
  const std::string r = record, v = vals;
  const FprTrRecord* rp = rec_of(r);
  if (v.size() < 8u * 1) return FPR_E_ARG;
  return fpr_tr_launch(rp, reinterpret_cast<const uint64_t*>(v.data()), static_cast<uint32_t>(gx), static_cast<uint32_t>(gy),
                       static_cast<uint32_t>(gz), reinterpret_cast<CUstream>(stream), static_cast<CUdeviceptr>(gscratch),
                       static_cast<CUdeviceptr>(pscratch));
}

static int64_t record_size() { return sizeof(FprTrRecord); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("check", &check, "fpr_tr_check(record, vals): 0 = in the record's key class, -2 = not");
  m.def("launch", &launch, "fpr_tr_launch(record, vals, gx, gy, gz, stream, gscratch, pscratch)");
  m.def("record_size", &record_size);
}
