// fe::Json: an ordered JSON DOM with Python-compatible parse and dump (owner A0, component COMMON).
//
// Parse == Python 3.12 `json.loads(bytes)` (Starlette's request.json()), including:
//   * objects keep the received key order; a duplicate key keeps its FIRST position and takes the LAST value
//     (Python dict semantics);
//   * integers of any size (Int when it fits int64, else BigInt holding the canonical decimal text);
//     more than 4300 digits -> JsonError(Value) like CPython's int-string limit;
//   * floats as double, correctly rounded; 1e400 -> inf; NaN / Infinity / -Infinity literals are accepted;
//   * "\ud800"-style lone surrogates are kept (stored as WTF-8, see below); raw UTF-8-encoded surrogates are
//     accepted (Python decodes request bytes with 'surrogatepass'); UTF-8 BOM stripped; UTF-16/32 detected;
//   * nesting deeper than 9997 -> JsonError(Recursion) (CPython 3.12 C recursion limit at top level);
//   * errors carry Python's exact message and char position ("Expecting value: line 1 column 4 (char 3)").
// Fast path: simdjson DOM (validating). Anything simdjson rejects is re-parsed by a Python-exact scanner,
// which either returns Python's value or throws Python's error.
//
// STRING ENCODING: std::string holds UTF-8. A Python str can contain lone surrogates (U+D800..U+DFFF); they are
// stored as their 3-byte generalized-UTF-8 form (ED A0..BF xx, "WTF-8"), which is exactly what
// str.encode('utf-8', 'surrogatepass') gives. Nothing else produces those bytes.
//
// Dump: dumps_py == json.dumps(obj, ensure_ascii=X) (separators ", " / ": ", float repr, NaN/Infinity literals);
// dumps_compact == json.dumps(obj, ensure_ascii=X, separators=(",", ":")); dumps_starlette == Starlette's
// JSONResponse.render (compact, ensure_ascii=False, allow_nan=False); dumps_pydantic == pydantic 2.14 /
// pydantic-core 2.50 model_dump_json for the same data (compact, raw UTF-8, its own float format, inf/nan -> null,
// lone surrogates -> error, > 255 nesting levels -> error). Floats: see JsonFloatStyle.
// Strings that are not valid (WTF-)UTF-8 cannot exist in Python; the writers emit U+FFFD for each invalid
// sequence (maximal-subpart rule, like bytes.decode('utf-8', 'replace')).
//
// ROBUSTNESS: parse, dump, copy, ==, py_equal and destruction use explicit heap stacks beyond shallow depth, so a
// 9997-deep request is safe on a 256 KB thread stack. Object lookups use a keyed SipHash-1-3 index above 12 keys
// (no hash flooding). Thread-safe for distinct values (one thread_local simdjson parser per thread).
// Verified byte-for-byte against the venv's Python: tools/golden/common_json_dump.py + tests/common/.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace fe {

class Json;
using JsonArray = std::vector<Json>;

struct JsonError : std::runtime_error {
  enum class Kind : uint8_t {
    Decode,     // json.JSONDecodeError: msg / pos / lineno / colno are set (pos = Python char index)
    Unicode,    // UnicodeDecodeError while decoding the request bytes (FastAPI: 400 "There was an error parsing the body")
    Value,      // other ValueError: int > 4300 digits, out-of-range float with allow_nan=False
    Recursion,  // RecursionError: nesting deeper than Python's limit
    Encode,     // serialization error (pydantic: lone surrogate)
    Type,       // C++ misuse: wrong-type access on a Json value (no Python equivalent)
  };
  Kind kind;
  std::string msg;     // bare message, e.g. "Expecting value" (== JSONDecodeError.msg); == what() for other kinds
  int64_t pos = -1;       // Decode: char (code point) index into the decoded document (== JSONDecodeError.pos)
  int64_t lineno = 0;     // Decode: 1-based
  int64_t colno = 0;      // Decode: 1-based
  int64_t byte_pos = -1;  // Decode: byte offset into the decoded UTF-8 text (== input offset unless BOM / UTF-16/32)
  JsonError(Kind k, const std::string& what_arg, std::string m = {}, int64_t p = -1, int64_t ln = 0, int64_t cn = 0,
            int64_t bp = -1)
      : std::runtime_error(what_arg), kind(k), msg(m.empty() ? what_arg : std::move(m)), pos(p), lineno(ln), colno(cn),
        byte_pos(bp) {}
};

// Insertion-ordered string-keyed map with Python dict semantics.
class JsonObject {
 public:
  using Member = std::pair<std::string, Json>;
  using iterator = std::vector<Member>::iterator;
  using const_iterator = std::vector<Member>::const_iterator;

  JsonObject() noexcept;
  JsonObject(const JsonObject&);
  JsonObject(JsonObject&&) noexcept;
  JsonObject& operator=(const JsonObject&);
  JsonObject& operator=(JsonObject&&) noexcept;
  ~JsonObject();

  size_t size() const noexcept { return m_.size(); }
  bool empty() const noexcept { return m_.empty(); }
  iterator begin() noexcept { return m_.begin(); }
  iterator end() noexcept { return m_.end(); }
  const_iterator begin() const noexcept { return m_.begin(); }
  const_iterator end() const noexcept { return m_.end(); }
  const Member& at_index(size_t i) const { return m_.at(i); }
  Member& at_index(size_t i) { return m_.at(i); }

  const Json* find(std::string_view key) const noexcept;
  Json* find(std::string_view key) noexcept;
  bool contains(std::string_view key) const noexcept { return find(key) != nullptr; }
  // Python `d[key] = value`: replaces in place (keeps the position) or appends.
  Json& set(std::string_view key, Json value);
  Json& set(std::string&& key, Json value);
  Json& set(const char* key, Json value);
  // Returns the existing value, or appends a null member and returns it (Python d.setdefault(key, None)).
  Json& get_or_insert(std::string_view key);
  // Python `del d[key]`: returns false when absent. The remaining keys keep their order.
  bool erase(std::string_view key);
  // Append WITHOUT a duplicate check (caller guarantees the key is absent). Fast builder path.
  Json& append_unique(std::string key, Json value);
  void reserve(size_t n) { m_.reserve(n); }
  void clear() noexcept;

 private:
  std::vector<Member> m_;
  std::unique_ptr<std::vector<uint32_t>> idx_;  // open-addressing hash index (pos+1), built above kLinear keys
  static constexpr size_t kLinear = 12;
  int64_t lookup(std::string_view key) const noexcept;
  void index_insert(size_t pos);
  void rebuild_index();
};

class Json {
 public:
  enum class Type : uint8_t { Null, Bool, Int, BigInt, Float, String, Array, Object };

  // ---- construction ----
  Json() noexcept : t_(Type::Null), i_(0) {}
  Json(std::nullptr_t) noexcept : t_(Type::Null), i_(0) {}
  Json(bool b) noexcept : t_(Type::Bool), b_(b) {}
  template <class T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool> &&
                                          !std::is_same_v<T, char>, int> = 0>
  Json(T v) {
    if constexpr (std::is_unsigned_v<T> && sizeof(T) >= sizeof(int64_t)) {
      if (v > static_cast<T>(INT64_MAX)) { init_big_u64(static_cast<uint64_t>(v)); return; }
    }
    t_ = Type::Int;
    i_ = static_cast<int64_t>(v);
  }
  Json(double d) noexcept : t_(Type::Float), d_(d) {}
  Json(float f) noexcept : t_(Type::Float), d_(static_cast<double>(f)) {}
  Json(const char* s) : t_(Type::String) { new (&s_) std::string(s); }
  Json(std::string_view s) : t_(Type::String) { new (&s_) std::string(s); }
  Json(const std::string& s) : t_(Type::String) { new (&s_) std::string(s); }
  Json(std::string&& s) noexcept : t_(Type::String) { new (&s_) std::string(std::move(s)); }
  Json(JsonArray a) noexcept : t_(Type::Array) { new (&a_) JsonArray(std::move(a)); }
  Json(JsonObject o) noexcept : t_(Type::Object) { new (&o_) JsonObject(std::move(o)); }

  static Json array() { return Json(JsonArray{}); }
  static Json array(std::initializer_list<Json> items) { return Json(JsonArray(items)); }
  static Json object() { return Json(JsonObject{}); }
  // Later duplicates overwrite earlier ones (keeping the first position), like a Python dict literal.
  static Json object(std::initializer_list<std::pair<std::string_view, Json>> items);
  // Integer from decimal text "-?[0-9]+" of any size (normalised like Python int()); Int when it fits int64.
  // Throws JsonError(Value) on bad text or > 4300 digits.
  static Json integer_from_text(std::string_view digits);
  // BigInt from already-canonical digits (no leading zeros, outside int64). No validation.
  static Json big_int_unchecked(std::string canonical_digits) noexcept;

  Json(const Json& o);
  Json(Json&& o) noexcept;
  Json& operator=(const Json& o);
  Json& operator=(Json&& o) noexcept;
  ~Json() { destroy(); }

  // ---- type queries ----
  Type type() const noexcept { return t_; }
  bool is_null() const noexcept { return t_ == Type::Null; }
  bool is_bool() const noexcept { return t_ == Type::Bool; }
  bool is_int() const noexcept { return t_ == Type::Int; }          // fits int64
  bool is_big_int() const noexcept { return t_ == Type::BigInt; }   // Python int outside int64
  bool is_integer() const noexcept { return t_ == Type::Int || t_ == Type::BigInt; }
  bool is_float() const noexcept { return t_ == Type::Float; }
  bool is_number() const noexcept { return t_ == Type::Int || t_ == Type::BigInt || t_ == Type::Float; }
  bool is_string() const noexcept { return t_ == Type::String; }
  bool is_array() const noexcept { return t_ == Type::Array; }
  bool is_object() const noexcept { return t_ == Type::Object; }
  const char* type_name() const noexcept;  // "null","bool","int","bigint","float","string","array","object"

  // ---- typed access (throw JsonError(Type) on mismatch) ----
  bool as_bool() const;
  int64_t as_int() const;                     // Int only
  int64_t as_int64() const { return as_int(); }
  double as_double() const;                   // Float, Int, BigInt (Python float(x); huge BigInt -> JsonError(Value))
  const std::string& as_string() const;       // String only
  std::string& as_string();
  std::string_view as_str() const { return as_string(); }
  const std::string& big_int_text() const;    // BigInt only: canonical decimal digits ("-123...")
  const JsonArray& as_array() const;
  JsonArray& as_array();
  const JsonObject& as_object() const;
  JsonObject& as_object();

  // ---- convenient lookups (never throw) ----
  // Object member or nullptr (also nullptr when *this is not an object).
  const Json* get(std::string_view key) const noexcept;
  Json* get(std::string_view key) noexcept;
  bool contains(std::string_view key) const noexcept { return get(key) != nullptr; }
  // Chained READ-ONLY access: missing key / index / wrong type -> a static null. There is deliberately no
  // inserting operator[] (an accidental insert would change dump bytes): write with set() / member().
  const Json& operator[](std::string_view key) const noexcept;
  const Json& operator[](const char* key) const noexcept { return (*this)[std::string_view(key)]; }
  const Json& operator[](size_t i) const noexcept;
  const Json& operator[](int i) const noexcept { return i < 0 ? null_ref() : (*this)[static_cast<size_t>(i)]; }
  // Write access to a member: null becomes an object; appends a null member when absent.
  // Throws JsonError(Type) when *this is neither null nor an object.
  Json& member(std::string_view key);
  // Typed member lookups: nullopt when absent or of another type.
  std::optional<std::string_view> get_string(std::string_view key) const noexcept;
  std::optional<int64_t> get_int(std::string_view key) const noexcept;
  std::optional<double> get_number(std::string_view key) const noexcept;  // Int or Float (not bool)
  std::optional<bool> get_bool(std::string_view key) const noexcept;
  size_t size() const noexcept;  // array / object length, else 0
  bool empty() const noexcept { return size() == 0; }

  // ---- builders ----
  Json& push_back(Json v);                       // null -> array
  Json& set(std::string_view key, Json v);       // null -> object; Python d[key] = v
  bool erase(std::string_view key);              // object only; Python del d[key]

  // ---- comparison ----
  // Structural equality: same types, same values, same key order (floats compare by value; NaN != NaN).
  friend bool operator==(const Json& a, const Json& b) noexcept;
  friend bool operator!=(const Json& a, const Json& b) noexcept { return !(a == b); }
  // Python `==`: bool/int/float compare numerically (True == 1 == 1.0), dicts ignore key order.
  bool py_equal(const Json& o) const noexcept;

  static const Json& null_ref() noexcept;

 private:
  friend class JsonObject;
  void destroy() noexcept;
  void init_big_u64(uint64_t v);
  [[noreturn]] void type_error(const char* want) const;
  Type t_;
  union {
    bool b_;
    int64_t i_;
    double d_;
    std::string s_;   // String; BigInt digits
    JsonArray a_;
    JsonObject o_;
  };
};

inline Json& JsonObject::set(const char* key, Json value) { return set(std::string_view(key), std::move(value)); }

// ================================ parsing ================================

struct JsonParseOptions {
  // true: json.loads(bytes) (BOM stripped, UTF-16/32 detected, 'surrogatepass' decode).
  // false: json.loads(str) for a str held as WTF-8 (a leading U+FEFF -> "Unexpected UTF-8 BOM" error).
  bool from_bytes = true;
  int max_depth = 9997;  // CPython 3.12 at top level: 9997 nested containers parse, 9998 -> RecursionError
};

Json json_parse(std::string_view text);  // == json.loads(bytes); throws JsonError
Json json_parse(std::string_view text, const JsonParseOptions& opt);
inline Json json_loads_str(std::string_view wtf8) {  // == json.loads(str), e.g. tool-call `arguments`
  JsonParseOptions o;
  o.from_bytes = false;
  return json_parse(wtf8, o);
}

// ================================ writing ================================

enum class JsonFloatStyle : uint8_t {
  Python,    // float.__repr__: "1e-05", "1e+16", "100000.0", "1.5e+300"
  Pydantic,  // pydantic-core: fixed for 1e-5 <= |x| < 1e16 ("0.00001"), else "1e-6" / "1e+16" (no exponent padding)
};
enum class JsonNonFinite : uint8_t {
  Literal,  // NaN / Infinity / -Infinity (json.dumps allow_nan=True)
  Error,    // JsonError(Value, "Out of range float values are not JSON compliant") (allow_nan=False)
  Null,     // null (pydantic ser_json_inf_nan='null', the default)
};

struct JsonDumpOptions {
  std::string_view item_sep = ", ";
  std::string_view key_sep = ": ";
  bool ensure_ascii = true;
  bool escape_del = true;          // ensure_ascii escapes DEL (json does; pydantic does not)
  bool surrogates_error = false;   // lone surrogate -> JsonError(Encode) (pydantic); else emitted (WTF-8 or \udXXX)
  JsonFloatStyle floats = JsonFloatStyle::Python;
  JsonNonFinite non_finite = JsonNonFinite::Literal;
  int indent = -1;                 // >= 0: json.dumps(indent=N); item_sep should then be "," (Python's default)
  // > 0: a value nested deeper than this (root = 1, scalars count) -> JsonError(Encode). pydantic-core: 255
  // ("Circular reference detected (depth exceeded)"), measured with RootModel[Any].
  int max_depth = 0;
};

void dump_append(std::string& out, const Json& v, const JsonDumpOptions& opt);
std::string dump(const Json& v, const JsonDumpOptions& opt);

// json.dumps(v, ensure_ascii=X)
std::string dumps_py(const Json& v, bool ensure_ascii);
// json.dumps(v, ensure_ascii=X, separators=(",", ":"))  (allow_nan=True: NaN literals)
std::string dumps_compact(const Json& v, bool ensure_ascii = false);
// Starlette JSONResponse.render: json.dumps(v, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
std::string dumps_starlette(const Json& v);
// pydantic model_dump_json of the same data (dict/list/str/int/float/bool/None fields):
// compact, raw UTF-8 (ensure_ascii=False) or \u-escaped non-ASCII (ensure_ascii=True, DEL kept raw),
// pydantic float format, inf/nan -> null, lone surrogate -> JsonError(Encode).
std::string dumps_pydantic(const Json& v, bool ensure_ascii = false);

// Low-level pieces for hand-built JSON (e.g. dsml_roundtrip arguments, SSE chunks):
// a quoted, escaped JSON string as json.dumps would write it.
void append_py_string(std::string& out, std::string_view s, bool ensure_ascii);
inline void append_json_string(std::string& out, std::string_view s, bool ensure_ascii) {
  append_py_string(out, s, ensure_ascii);
}
// pydantic model_dump_json string (raw UTF-8 or \u-escaped non-ASCII; DEL raw; lone surrogate -> JsonError(Encode)).
void append_pydantic_string(std::string& out, std::string_view s, bool ensure_ascii = false);
// float.__repr__ (finite) / NaN / Infinity / -Infinity.
void append_py_float(std::string& out, double d);
std::string py_float_repr(double d);  // repr(float): "inf", "nan", "-inf" for non-finite (Python repr, not JSON)
// pydantic float text (finite only; caller handles non-finite).
void append_pydantic_float(std::string& out, double d);
std::string format_float_pydantic(double d);  // NOT ryu: "1e-7", "1e+16", "0.00001" (see JsonFloatStyle::Pydantic)

// ================================ helpers ================================

// Is `s` valid UTF-8 (strict, no surrogates)?
bool utf8_valid(std::string_view s) noexcept;
// Number of Python chars (code points) in a WTF-8 string (invalid bytes count 1 each).
size_t py_len(std::string_view wtf8) noexcept;

}  // namespace fe
