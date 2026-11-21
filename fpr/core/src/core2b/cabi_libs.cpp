// cabi_libs.cpp -- see cabi_libs.h. Build: -O2 -std=c++17 (links -ldl).
#include "cabi_libs.h"

#include <dlfcn.h>

#include <cstring>

namespace fpr {
namespace core2b {

namespace {
struct FnEntry { const char* name; CabiFn fn; int64_t (*ws_bytes)(const void*); int64_t args_size; };   // = FprFnEntry
struct ExtTable { int abi_version; const char* ext; const char* src_md5; int n_fns; const FnEntry* fns; };  // = FprExtTable
using TableFn = const ExtTable* (*)();
}  // namespace

void* CabiLibs::open_(const std::string& lib, std::string* why) {
  auto it = handles_.find(lib);
  if (it != handles_.end()) return it->second;
  if (h_ == nullptr) {
    *why = "no handover";
    return nullptr;
  }
  const char* path = nullptr;
  for (int i = 0; i < h_->n_libs && i < FPR_EH_MAX_LIBS; ++i)
    if (std::strncmp(h_->libs[i].name, lib.c_str(), sizeof(h_->libs[i].name)) == 0) path = h_->libs[i].path;
  if (path == nullptr) {
    *why = "lib " + lib + " not exported (handover libs[])";
    return nullptr;
  }
  void* d = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
  if (d == nullptr) {
    const char* e = dlerror();
    *why = "dlopen(" + std::string(path) + ", RTLD_NOLOAD): " + (e ? e : "?");
    return nullptr;
  }
  handles_[lib] = d;
  return d;
}

CabiFn CabiLibs::fn(const std::string& lib, const std::string& name, int64_t args_size, std::string* why) {
  const std::string key = lib + "/" + name;
  auto it = fns_.find(key);
  if (it != fns_.end()) return it->second;
  void* d = open_(lib, why);
  if (d == nullptr) return nullptr;
  const std::string tsym = "fpr_" + lib + "_table";
  TableFn tf = reinterpret_cast<TableFn>(dlsym(d, tsym.c_str()));
  const ExtTable* t = tf ? tf() : nullptr;
  if (t == nullptr) {
    *why = lib + ": no " + tsym;
    return nullptr;
  }
  for (int i = 0; i < t->n_fns; ++i) {
    if (t->fns[i].name == nullptr || name != t->fns[i].name) continue;
    if (t->fns[i].args_size != args_size) {
      *why = name + ": args_size " + std::to_string(t->fns[i].args_size) + " != core2b's " + std::to_string(args_size);
      return nullptr;
    }
    fns_[key] = t->fns[i].fn;
    return t->fns[i].fn;
  }
  *why = lib + ": no table entry " + name;
  return nullptr;
}

void* CabiLibs::sym(const std::string& lib, const char* symbol, std::string* why) {
  void* d = open_(lib, why);
  return d ? dlsym(d, symbol) : nullptr;
}

std::string t_stage_check(const TgKeys& A, const TgKeys& T, int64_t n, bool l20, bool clear) {
  if (n < 0) return "n < 0";
  if (A.version != kTgKeysVersion || T.version != kTgKeysVersion) return "key table version";
  if (A.n_idx != T.n_idx || A.n_idx < 0 || A.n_idx > kTgMaxIdx) return "n_idx";
  int n_keys = 0, n_pads = 0;
  auto key = [&](int sa, int st) -> std::string {
    if (A.p[sa] == nullptr || T.p[st] == nullptr) return "key " + std::to_string(sa) + " missing";
    if (A.row_bytes[sa] != T.row_bytes[st] || A.row_bytes[sa] <= 0) return "key " + std::to_string(sa) + " row bytes";
    if (T.rows[st] < n) return "key " + std::to_string(st) + " rows < n";
    return ++n_keys > 16 ? "keys > 16" : "";
  };
  auto pad = [&](int ss, int sd, bool fill) -> std::string {
    if (T.p[sd] == nullptr || T.row_bytes[sd] <= 0) return "pad " + std::to_string(sd) + " missing";
    if (!fill && (T.p[ss] == nullptr || T.row_bytes[ss] != T.row_bytes[sd] || T.rows[ss] < T.rows[sd]))
      return "pad source " + std::to_string(ss);
    return ++n_pads > 16 ? "pads > 16" : "";
  };
  std::string e;
  if (l20) {
    for (int s : {TG_X, TG_A_PRE, TG_A_POST, TG_A_COMB, TG_WS_Q})
      if (!(e = key(s, s)).empty()) return e;
    if (!(e = key(TG_SI, TG_SI20)).empty() || !(e = key(TG_SL, TG_SL20)).empty()) return e;
  } else {
    for (int s : {TG_MOE_OUT, TG_XM2, TG_F_POST, TG_F_COMB, TG_F_PRE})
      if (!(e = key(s, s)).empty()) return e;
  }
  for (int j = 0; j < A.n_idx; ++j)
    if (!(e = key(TG_TI0 + j, TG_TI0 + j)).empty() || !(e = key(TG_TL0 + j, TG_TL0 + j)).empty()) return e;
  if (l20 && (!(e = pad(TG_SI, TG_SI20, false)).empty() || !(e = pad(TG_SL, TG_SL20, false)).empty())) return e;
  if (clear)
    for (int j = 0; j < T.n_idx; ++j)
      if (!(e = pad(-1, TG_TI0 + j, true)).empty() || !(e = pad(-1, TG_TL0 + j, true)).empty()) return e;
  return "";
}

}  // namespace core2b
}  // namespace fpr
