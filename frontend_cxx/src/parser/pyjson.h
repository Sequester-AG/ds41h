// Internal to fe_parser: a small JSON value with Python json-module semantics, plus the Python string predicates the
// DSML parser needs. Kept private to the parser so that the byte-level contract (Python json.loads acceptance,
// json.dumps output, str.isspace, the `regex` module's \s) is owned and tested here.
//   - loads():  Python json.loads(s) (strict=True): NaN / Infinity / -Infinity accepted, control characters inside
//               strings rejected, whitespace [ \t\n\r], ints vs floats as Python decides, duplicate object keys keep
//               the first key position and the last value (dict semantics).
//   - dumps():  Python json.dumps(v, ensure_ascii=...) with the default separators ", " / ": ", float repr, lowercase
//               \uXXXX escapes (surrogate pairs above U+FFFF when ensure_ascii).
// Strings are UTF-8; a lone surrogate from a \uD8xx escape is kept as its 3-byte (WTF-8) form.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace fe::pyjson {

struct Value {
  enum class T : uint8_t { Null, Bool, Int, Float, Str, Arr, Obj };
  T t = T::Null;
  bool b = false;
  double f = 0;
  std::string s;                  // Str: text; Int: canonical decimal digits (Python int repr)
  std::vector<Value> items;       // Arr items, or Obj values
  std::vector<std::string> keys;  // Obj keys (same length as items)

  static Value null() { return Value{}; }
  static Value boolean(bool v) { Value x; x.t = T::Bool; x.b = v; return x; }
  static Value str(std::string v) { Value x; x.t = T::Str; x.s = std::move(v); return x; }
  static Value integer(long long v) { Value x; x.t = T::Int; x.s = std::to_string(v); return x; }
  static Value array() { Value x; x.t = T::Arr; return x; }
  static Value object() { Value x; x.t = T::Obj; return x; }

  bool is_null() const { return t == T::Null; }
  bool is_str() const { return t == T::Str; }
  bool is_obj() const { return t == T::Obj; }
  bool is_arr() const { return t == T::Arr; }
  bool is_bool() const { return t == T::Bool; }
  const Value* get(std::string_view k) const;  // object member or nullptr
  Value* get_mut(std::string_view k);
  bool has(std::string_view k) const { return get(k) != nullptr; }
  void set(std::string k, Value v);             // dict assignment semantics (existing key keeps its position)
  void erase(std::string_view k);
  void push(Value v) { items.push_back(std::move(v)); }
  bool truthy() const;                          // Python bool(x)
};

// Python json.loads. Returns false on any JSONDecodeError (nothing else is raised for valid UTF-8 input; nesting is
// limited to 10000 levels, deeper input returns false).
bool loads(std::string_view s, Value& out);
inline bool loads_ok(std::string_view s) { Value v; return loads(s, v); }

void dumps_to(const Value& v, bool ensure_ascii, std::string& out);
inline std::string dumps(const Value& v, bool ensure_ascii) { std::string o; dumps_to(v, ensure_ascii, o); return o; }
// json.dumps(<str>, ensure_ascii=...)
void dump_str_to(std::string_view s, bool ensure_ascii, std::string& out);
inline std::string dump_str(std::string_view s, bool ensure_ascii) { std::string o; dump_str_to(s, ensure_ascii, o); return o; }
// Python repr(float), as json.dumps writes floats (inf/nan as Infinity/-Infinity/NaN)
std::string float_repr(double d);

// ---- Python string predicates over UTF-8 ----
// decode one code point at s[i]; returns its byte length (>=1). Invalid bytes decode as themselves (length 1).
uint32_t decode_cp(std::string_view s, size_t i, size_t& len);
size_t cp_len_at(std::string_view s, size_t i);
bool py_isspace(uint32_t cp);   // str.isspace()
bool re_space(uint32_t cp);     // `regex` module \s (Unicode White_Space): str.isspace() minus U+001C..U+001F
std::string_view py_strip(std::string_view s);    // str.strip()
std::string_view py_rstrip(std::string_view s);   // str.rstrip()
inline bool py_is_blank(std::string_view s) { return py_strip(s).empty(); }  // not s.strip()

}  // namespace fe::pyjson
