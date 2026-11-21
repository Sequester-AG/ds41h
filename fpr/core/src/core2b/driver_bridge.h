// driver_bridge.h -- the C++ mixed-step driver from core2b without CUDA headers
// The native step's SET_STEP / ISSUE / JOIN_MIX handlers call
// the driver's C exports through dlopen(path, RTLD_NOW | RTLD_NOLOAD) of the ALREADY LOADED fpr_mixed_cxx.so (Python's
// mixed_cxx.ext().__file__: one copy of the driver in the process). Struct layouts mirror engine/mixed_cxx/pieces.h
// (FprStepParams, FprProgEntry) with void* for cudaEvent_t / cudaStream_t; sizes are checked at load where the driver
// exports them. ok() == false (the .so or a symbol missing) -> the handlers are not registered -> those ops refuse.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "span_program.h"

namespace fpr {
namespace core2b {

struct DrvStepParams {                  // = FprStepParams (pieces.h)
  int32_t chain_input = 0;
  int32_t moe_variant = -1;
  int64_t moe_vcount = 0;
  int32_t moe_n_wait = 0, pad0_ = 0;
  void* moe_wait[4] = {nullptr, nullptr, nullptr, nullptr};
};
struct DrvProgEntry { int32_t kind, layer; int64_t handle; int32_t arg, pad_; };   // = FprProgEntry

class DriverBridge {
 public:
  // path: the loaded fpr_mixed_cxx.so; "" = not loaded (ok() false)
  bool load(const std::string& path, std::string* err);
  bool ok() const { return set_sp_ && run_span_ && join_; }
  int set_step_params(int rank, const DrvStepParams& sp) const { return set_sp_(rank, &sp); }
  // every rank's program (prog[r] in order) on the driver threads; blocking like mixed_cxx's run_span. rc[r] != 0 = failed
  int run_span(const std::vector<std::vector<SpanEntry>>& prog, std::vector<int>* rc) const;
  int join(int rank, void* stream) const { return join_(rank, stream); }
  std::string last_error(int rank) const;

 private:
  using SetSpFn = int (*)(int, const void*);
  using RunSpanFn = int (*)(int, const void* const*, const int64_t*, int*);
  using JoinFn = int (*)(int, void*);
  using LastErrFn = const char* (*)(int);
  SetSpFn set_sp_ = nullptr;
  RunSpanFn run_span_ = nullptr;
  JoinFn join_ = nullptr;
  LastErrFn last_err_ = nullptr;
};

DrvStepParams step_params(bool chain_input, int64_t vcount, int32_t moe_variant);

}  // namespace core2b
}  // namespace fpr
