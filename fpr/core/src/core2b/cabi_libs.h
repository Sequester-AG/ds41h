// cabi_libs.h -- the owners' C ABI functions (engine/mixed_cxx/fpr_cabi.h rule 7) from core2b without CUDA headers
// A library is found by NAME in the handover's v0.7 slice-4 table libs[] ({name, path}: the loaded
// extension's __file__), opened with dlopen(path, RTLD_NOW | RTLD_NOLOAD) (one copy of the kernels in the process), and its
// `fpr_<name>_table()` is searched for the function; the entry's args_size must equal the caller's mirror (the ABI drift
// guard). nd's cabi() sets the device and the leading FprCall {step_stream, device, rank}: the mirrors here put a
// CabiCall (16 bytes, void* stream) first.
//
// Mirrors
//   TStageArgs = kernels/tail_gather/tail_gather_cabi.h FprTStageArgs; TgKeys = FprTgKeys (FPR_TG_KEYS_VERSION 1).
#pragma once
#include <cstdint>
#include <map>
#include <string>

#include "engine_handover.h"

namespace fpr {
namespace core2b {

using CabiFn = int (*)(const void* args);

struct CabiCall { void* stream; int32_t device, rank; };   // = FprCall (fpr_cabi.h)
static_assert(sizeof(CabiCall) == 16, "FprCall is 16 bytes");

// ---- tail_gather
constexpr int kTgSlots = 24, kTgMaxIdx = 4, kTgKeysVersion = 1;
enum : int {
  TG_X = 0, TG_A_PRE, TG_A_POST, TG_A_COMB, TG_WS_Q, TG_SI, TG_SL, TG_SI20, TG_SL20, TG_MOE_OUT, TG_XM2, TG_F_POST, TG_F_COMB,
  TG_F_PRE, TG_TI0 = 14, TG_TL0 = 18
};
struct TgKeys {                         // = FprTgKeys
  int32_t version, n_idx;
  int64_t R;
  void* p[kTgSlots];
  int64_t row_bytes[kTgSlots];
  int64_t rows[kTgSlots];
};
static_assert(sizeof(TgKeys) == 592, "FprTgKeys");
struct TStageArgs {                     // = FprTStageArgs
  CabiCall call;
  const int64_t* ix;
  int64_t n;
  int32_t l20, clear;
  const TgKeys* A;
  const TgKeys* T;
  int32_t* err;
};
static_assert(sizeof(TStageArgs) == 64, "FprTStageArgs");
// fpr_t_stage_args' checks on the host: "" = the T stage of these views is buildable (every key
// present with matching row bytes, T rows >= n, the pads' sources), else why not. Used in phase 1 when the library does not
// export fpr_t_stage_args itself (CabiLibs::sym)
std::string t_stage_check(const TgKeys& A, const TgKeys& T, int64_t n, bool l20, bool clear);

class CabiLibs {
 public:
  // the handover whose libs[] to search (engine lifetime); entries are resolved lazily and cached
  void set(const FprEngineHandover* h) { h_ = h; }
  // lib = the libs[] name ("tail_gather"), name = the table entry ("tail_gather.t_stage"); nullptr + *why on any miss
  CabiFn fn(const std::string& lib, const std::string& name, int64_t args_size, std::string* why);
  // a plain exported symbol of the library (e.g. fpr_t_stage_args); nullptr if absent
  void* sym(const std::string& lib, const char* symbol, std::string* why);

 private:
  void* open_(const std::string& lib, std::string* why);
  const FprEngineHandover* h_ = nullptr;
  std::map<std::string, void*> handles_;
  std::map<std::string, CabiFn> fns_;
};

}  // namespace core2b
}  // namespace fpr
