// Streaming JSON writer for byte-exact response bodies (no DOM).
//
// Two dialects are needed to match the Python frontend:
//  - Py:       json.dumps(..., ensure_ascii=?, separators=?) -- Starlette JSONResponse
//              (compact, ensure_ascii=False) and the SSE error chunk (", "/": ",
//              ensure_ascii=True). Floats use Python repr(): 1e-05, 1e+16, 0.1, 1.0.
//  - Pydantic: model_dump_json() (serde_json): compact, non-ASCII verbatim, floats in
//              ryu style: 1e-5 -> "0.00001", 1e16 -> "1e16", 1.0 -> "1.0".
// String escaping is the same in both except ensure_ascii.
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "common/json.h"

namespace fe::api {

enum class FloatStyle { PyRepr, Ryu };

// Escaping / float text come from fe_common when it is linked (FE_HAVE_COMMON), else from
// the local fallback in jw.cpp (identical output; checked by tests/api/jw_golden.py).
void append_escaped(std::string& out, std::string_view utf8, bool ensure_ascii);   // json.dumps
void append_escaped_pyd(std::string& out, std::string_view utf8, bool ensure_ascii);  // pydantic (DEL raw)
std::string py_float_repr(double v);   // Python repr(float)
std::string ryu_float(double v);       // pydantic-core float text

class JW {
 public:
  explicit JW(bool compact = true, bool ensure_ascii = false, FloatStyle fs = FloatStyle::PyRepr)
      : compact_(compact), ensure_ascii_(ensure_ascii), fs_(fs) {}
  JW& ob() { pre(); s_ += '{'; st_.push_back(0); return *this; }
  JW& oe() { st_.pop_back(); s_ += '}'; return *this; }
  JW& ab() { pre(); s_ += '['; st_.push_back(0); return *this; }
  JW& ae() { st_.pop_back(); s_ += ']'; return *this; }
  JW& k(std::string_view key) {
    if (!st_.empty() && st_.back()++) s_ += compact_ ? "," : ", ";
    if (fs_ == FloatStyle::PyRepr)
      append_escaped(s_, key, ensure_ascii_);
    else
      append_escaped_pyd(s_, key, ensure_ascii_);
    s_ += compact_ ? ":" : ": ";
    after_key_ = true;
    return *this;
  }
  JW& str(std::string_view v) {
    pre();
    if (fs_ == FloatStyle::PyRepr)
      append_escaped(s_, v, ensure_ascii_);
    else
      append_escaped_pyd(s_, v, ensure_ascii_);
    return *this;
  }
  JW& null() { pre(); s_ += "null"; return *this; }
  JW& b(bool v) { pre(); s_ += v ? "true" : "false"; return *this; }
  JW& i(int64_t v) { pre(); s_ += std::to_string(v); return *this; }
  JW& f(double v) { pre(); s_ += fs_ == FloatStyle::PyRepr ? py_float_repr(v) : ryu_float(v); return *this; }
  // Pre-serialised JSON value.
  JW& raw(std::string_view v) { pre(); s_.append(v); return *this; }
  // A DOM value in this writer's dialect (key order kept).
  JW& json(const fe::Json& v) {
    pre();
    fe::JsonDumpOptions o;
    o.item_sep = compact_ ? "," : ", ";
    o.key_sep = compact_ ? ":" : ": ";
    o.ensure_ascii = ensure_ascii_;
    if (fs_ == FloatStyle::Ryu) {
      o.floats = fe::JsonFloatStyle::Pydantic;
      o.escape_del = false;
      o.non_finite = fe::JsonNonFinite::Null;
    }
    fe::dump_append(s_, v, o);
    return *this;
  }
  JW& kjson(std::string_view key, const fe::Json& v) { return k(key).json(v); }
  // Helpers for "key: value".
  JW& kstr(std::string_view key, std::string_view v) { return k(key).str(v); }
  JW& knull(std::string_view key) { return k(key).null(); }
  JW& ki(std::string_view key, int64_t v) { return k(key).i(v); }
  JW& kb(std::string_view key, bool v) { return k(key).b(v); }
  JW& kraw(std::string_view key, std::string_view v) { return k(key).raw(v); }
  JW& kopt(std::string_view key, const std::string* v) { return v ? kstr(key, *v) : knull(key); }
  JW& int_array(const std::vector<int32_t>& v) {
    ab();
    for (auto x : v) i(x);
    return ae();
  }
  const std::string& out() const { return s_; }
  std::string take() { return std::move(s_); }
  bool compact() const { return compact_; }
  bool ensure_ascii() const { return ensure_ascii_; }
  FloatStyle float_style() const { return fs_; }

 private:
  void pre() {
    if (after_key_) {
      after_key_ = false;
      return;
    }
    if (!st_.empty() && st_.back()++) s_ += compact_ ? "," : ", ";
  }
  std::string s_;
  std::vector<int> st_;
  bool after_key_ = false;
  bool compact_, ensure_ascii_;
  FloatStyle fs_;
};

}  // namespace fe::api
