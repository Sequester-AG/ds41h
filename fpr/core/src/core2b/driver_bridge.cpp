// driver_bridge.cpp -- see driver_bridge.h. Build: -O2 -std=c++17 (links -ldl).
#include "driver_bridge.h"

#include <dlfcn.h>

namespace fpr {
namespace core2b {

static_assert(sizeof(DrvProgEntry) == 24, "FprProgEntry is 24 bytes (pieces.h)");
static_assert(sizeof(DrvStepParams) == 56, "FprStepParams is 56 bytes on LP64 (pieces.h)");

bool DriverBridge::load(const std::string& path, std::string* err) {
  if (path.empty()) {
    if (err) *err = "driver not loaded (no fpr_mixed_cxx path)";
    return false;
  }
  void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
  if (h == nullptr) {
    if (err) *err = "dlopen(" + path + ", RTLD_NOLOAD): " + std::string(dlerror() ? dlerror() : "?");
    return false;
  }
  set_sp_ = reinterpret_cast<SetSpFn>(dlsym(h, "fpr_drv_set_step_params"));
  run_span_ = reinterpret_cast<RunSpanFn>(dlsym(h, "fpr_drv_run_span"));
  join_ = reinterpret_cast<JoinFn>(dlsym(h, "fpr_drv_join"));
  // FPR's driver has join with C++ linkage only (driver_api.h, for in-process piece files; the C
  // spans use the FPR_K_JOIN entry): every driver build exports `int fpr_drv_join(int, cudaStream_t)` as this symbol. Same
  // function FPR_K_JOIN calls; (int, CUstream_st*) and (int, void*) pass identically on LP64.
  if (join_ == nullptr) join_ = reinterpret_cast<JoinFn>(dlsym(h, "_Z12fpr_drv_joiniP11CUstream_st"));
  last_err_ = reinterpret_cast<LastErrFn>(dlsym(h, "fpr_drv_last_error"));
  if (!ok()) {
    if (err)
      *err = std::string("driver C exports missing:") + (set_sp_ ? "" : " fpr_drv_set_step_params") +
             (run_span_ ? "" : " fpr_drv_run_span") + (join_ ? "" : " fpr_drv_join (C or C++ symbol)");
    return false;
  }
  return true;
}

int DriverBridge::run_span(const std::vector<std::vector<SpanEntry>>& prog, std::vector<int>* rc) const {
  const int W = (int)prog.size();
  std::vector<std::vector<DrvProgEntry>> e((size_t)W);
  std::vector<const void*> ptrs((size_t)W);
  std::vector<int64_t> n((size_t)W);
  for (int r = 0; r < W; ++r) {
    for (const SpanEntry& x : prog[(size_t)r]) e[(size_t)r].push_back(DrvProgEntry{x.kind, x.layer, x.handle, (int32_t)x.arg, 0});
    ptrs[(size_t)r] = e[(size_t)r].data();
    n[(size_t)r] = (int64_t)e[(size_t)r].size();
  }
  rc->assign((size_t)W, 0);
  return run_span_(W, ptrs.data(), n.data(), rc->data());
}

std::string DriverBridge::last_error(int rank) const {
  const char* s = last_err_ ? last_err_(rank) : nullptr;
  return s ? std::string(s) : std::string();
}

DrvStepParams step_params(bool chain_input, int64_t vcount, int32_t moe_variant) {
  DrvStepParams sp;
  sp.chain_input = chain_input ? 1 : 0;
  sp.moe_variant = moe_variant;
  sp.moe_vcount = vcount;
  sp.moe_n_wait = 0;
  return sp;
}

}  // namespace core2b
}  // namespace fpr
