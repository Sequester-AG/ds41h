// fe::Json DOM + Python-compatible parser (simdjson fast path, Python-exact fallback scanner).
#include "common/json.h"

#include <clocale>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <locale.h>

#include "simdjson/simdjson.h"

namespace fe {

// ============================================================================================
// JsonObject
// ============================================================================================

JsonObject::JsonObject() noexcept = default;
JsonObject::JsonObject(const JsonObject& o) : m_(o.m_) {
  if (o.idx_) idx_ = std::make_unique<std::vector<uint32_t>>(*o.idx_);
}
JsonObject::JsonObject(JsonObject&& o) noexcept = default;
JsonObject& JsonObject::operator=(const JsonObject& o) {
  if (this != &o) {
    m_ = o.m_;
    idx_.reset(o.idx_ ? new std::vector<uint32_t>(*o.idx_) : nullptr);
  }
  return *this;
}
JsonObject& JsonObject::operator=(JsonObject&& o) noexcept = default;
JsonObject::~JsonObject() = default;

// Keyed SipHash-1-3 (per-process random key), like CPython's str hash: crafted keys cannot flood the index.
static inline uint64_t rotl64(uint64_t x, int b) { return (x << b) | (x >> (64 - b)); }
static uint64_t siphash13(const unsigned char* in, size_t len, uint64_t k0, uint64_t k1) noexcept {
  uint64_t v0 = 0x736f6d6570736575ULL ^ k0, v1 = 0x646f72616e646f6dULL ^ k1;
  uint64_t v2 = 0x6c7967656e657261ULL ^ k0, v3 = 0x7465646279746573ULL ^ k1;
  auto round = [&] {
    v0 += v1; v1 = rotl64(v1, 13); v1 ^= v0; v0 = rotl64(v0, 32);
    v2 += v3; v3 = rotl64(v3, 16); v3 ^= v2;
    v0 += v3; v3 = rotl64(v3, 21); v3 ^= v0;
    v2 += v1; v1 = rotl64(v1, 17); v1 ^= v2; v2 = rotl64(v2, 32);
  };
  const unsigned char* end = in + (len & ~size_t(7));
  for (; in != end; in += 8) {
    uint64_t m;
    std::memcpy(&m, in, 8);
    v3 ^= m;
    round();
    v0 ^= m;
  }
  uint64_t b = static_cast<uint64_t>(len) << 56;
  for (size_t i = 0; i < (len & 7); ++i) b |= static_cast<uint64_t>(in[i]) << (8 * i);
  v3 ^= b;
  round();
  v0 ^= b;
  v2 ^= 0xff;
  round();
  round();
  round();
  return v0 ^ v1 ^ v2 ^ v3;
}
struct SipKey {
  uint64_t k0, k1;
  SipKey() {
    std::random_device rd;
    k0 = (static_cast<uint64_t>(rd()) << 32) ^ rd();
    k1 = (static_cast<uint64_t>(rd()) << 32) ^ rd();
  }
};
static inline size_t key_hash(std::string_view k) noexcept {
  static const SipKey key;
  return static_cast<size_t>(siphash13(reinterpret_cast<const unsigned char*>(k.data()), k.size(), key.k0, key.k1));
}

int64_t JsonObject::lookup(std::string_view key) const noexcept {
  if (!idx_) {
    const size_t n = m_.size();
    for (size_t i = 0; i < n; ++i) {
      const std::string& k = m_[i].first;
      if (k.size() == key.size() && std::memcmp(k.data(), key.data(), key.size()) == 0) return static_cast<int64_t>(i);
    }
    return -1;
  }
  const std::vector<uint32_t>& t = *idx_;
  const size_t mask = t.size() - 1;
  for (size_t h = key_hash(key) & mask;; h = (h + 1) & mask) {
    uint32_t v = t[h];
    if (v == 0) return -1;
    const std::string& k = m_[v - 1].first;
    if (k.size() == key.size() && std::memcmp(k.data(), key.data(), key.size()) == 0) return v - 1;
  }
}

void JsonObject::index_insert(size_t pos) {
  if (!idx_) {
    if (m_.size() <= kLinear) return;
    rebuild_index();
    return;
  }
  std::vector<uint32_t>& t = *idx_;
  if ((m_.size() * 2) > t.size()) {  // load factor 1/2
    rebuild_index();
    return;
  }
  const size_t mask = t.size() - 1;
  size_t h = key_hash(m_[pos].first) & mask;
  while (t[h] != 0) h = (h + 1) & mask;
  t[h] = static_cast<uint32_t>(pos + 1);
}

void JsonObject::rebuild_index() {
  if (m_.size() <= kLinear) {
    idx_.reset();
    return;
  }
  size_t cap = 32;
  while (cap < m_.size() * 4) cap <<= 1;
  if (!idx_) idx_ = std::make_unique<std::vector<uint32_t>>();
  idx_->assign(cap, 0);
  std::vector<uint32_t>& t = *idx_;
  const size_t mask = cap - 1;
  for (size_t i = 0; i < m_.size(); ++i) {
    size_t h = key_hash(m_[i].first) & mask;
    while (t[h] != 0) h = (h + 1) & mask;
    t[h] = static_cast<uint32_t>(i + 1);
  }
}

const Json* JsonObject::find(std::string_view key) const noexcept {
  int64_t i = lookup(key);
  return i < 0 ? nullptr : &m_[static_cast<size_t>(i)].second;
}
Json* JsonObject::find(std::string_view key) noexcept {
  int64_t i = lookup(key);
  return i < 0 ? nullptr : &m_[static_cast<size_t>(i)].second;
}

Json& JsonObject::set(std::string_view key, Json value) {
  int64_t i = lookup(key);
  if (i >= 0) {
    Json& slot = m_[static_cast<size_t>(i)].second;
    slot = std::move(value);
    return slot;
  }
  return append_unique(std::string(key), std::move(value));
}

Json& JsonObject::set(std::string&& key, Json value) {
  int64_t i = lookup(key);
  if (i >= 0) {
    Json& slot = m_[static_cast<size_t>(i)].second;
    slot = std::move(value);
    return slot;
  }
  return append_unique(std::move(key), std::move(value));
}

Json& JsonObject::get_or_insert(std::string_view key) {
  int64_t i = lookup(key);
  if (i >= 0) return m_[static_cast<size_t>(i)].second;
  return append_unique(std::string(key), Json());
}

Json& JsonObject::append_unique(std::string key, Json value) {
  m_.emplace_back(std::move(key), std::move(value));
  index_insert(m_.size() - 1);
  return m_.back().second;
}

bool JsonObject::erase(std::string_view key) {
  int64_t i = lookup(key);
  if (i < 0) return false;
  m_.erase(m_.begin() + i);
  if (idx_) rebuild_index();
  return true;
}

void JsonObject::clear() noexcept {
  m_.clear();
  idx_.reset();
}

// ============================================================================================
// Json
// ============================================================================================

const Json& Json::null_ref() noexcept {
  static const Json n;
  return n;
}

static inline bool nonempty_container(const Json& j) noexcept {
  return (j.is_array() || j.is_object()) && j.size() > 0;
}

// Move the non-empty child containers of `j` into `work` (each left as an empty moved-from container).
static void steal_nested(Json& j, std::vector<Json>& work) {
  if (j.is_array()) {
    for (Json& c : j.as_array())
      if (nonempty_container(c)) work.push_back(std::move(c));
  } else if (j.is_object()) {
    for (auto& kv : j.as_object())
      if (nonempty_container(kv.second)) work.push_back(std::move(kv.second));
  }
}

static bool has_nested(const Json& j) noexcept {
  if (j.is_array()) {
    for (const Json& c : j.as_array())
      if (nonempty_container(c)) return true;
  } else if (j.is_object()) {
    for (const auto& kv : j.as_object())
      if (nonempty_container(kv.second)) return true;
  }
  return false;
}

// Deep containers are torn down iteratively (nested containers go to a heap worklist), so destroying a 10k-deep
// document never recurses deeply: request handlers may run on small stacks.
static thread_local int tl_destroy_depth = 0;

void Json::destroy() noexcept {
  switch (t_) {
    case Type::String:
    case Type::BigInt: s_.~basic_string(); break;
    case Type::Array:
    case Type::Object: {
      // plain recursion for the first 64 levels (fast); beyond, a heap worklist (bounded stack)
      if (tl_destroy_depth >= 64 && has_nested(*this)) {
        std::vector<Json> work;
        steal_nested(*this, work);
        while (!work.empty()) {
          Json x(std::move(work.back()));
          work.pop_back();
          steal_nested(x, work);
        }  // x has no nested containers left when it is destroyed here
      }
      ++tl_destroy_depth;
      if (t_ == Type::Array) a_.~JsonArray();
      else o_.~JsonObject();
      --tl_destroy_depth;
      break;
    }
    default: break;
  }
  t_ = Type::Null;
}

static Json shallow_copy(const Json& o) {
  switch (o.type()) {
    case Json::Type::Array: return Json::array();
    case Json::Type::Object: return Json::object();
    default: return Json(o);  // scalars / strings: no recursion
  }
}

// Iterative deep copy (explicit stack): no recursion per nesting level.
Json::Json(const Json& o) : t_(o.t_) {
  switch (t_) {
    case Type::Null: i_ = 0; return;
    case Type::Bool: b_ = o.b_; return;
    case Type::Int: i_ = o.i_; return;
    case Type::Float: d_ = o.d_; return;
    case Type::String:
    case Type::BigInt: new (&s_) std::string(o.s_); return;
    case Type::Array: new (&a_) JsonArray(); break;
    case Type::Object: new (&o_) JsonObject(); break;
  }
  std::vector<std::pair<const Json*, Json*>> st;
  st.emplace_back(&o, this);
  while (!st.empty()) {
    auto [src, dst] = st.back();
    st.pop_back();
    if (src->t_ == Type::Array) {
      const JsonArray& sa = src->a_;
      JsonArray& da = dst->a_;
      da.reserve(sa.size());
      for (const Json& c : sa) da.push_back(shallow_copy(c));
      for (size_t k = 0; k < sa.size(); ++k)
        if (nonempty_container(sa[k])) st.emplace_back(&sa[k], &da[k]);  // da is complete: pointers stay valid
    } else {
      const JsonObject& so = src->o_;
      JsonObject& dob = dst->o_;
      dob.reserve(so.size());
      for (const auto& kv : so) dob.append_unique(kv.first, shallow_copy(kv.second));
      for (size_t k = 0; k < so.size(); ++k)
        if (nonempty_container(so.at_index(k).second)) st.emplace_back(&so.at_index(k).second, &dob.at_index(k).second);
    }
  }
}

Json::Json(Json&& o) noexcept : t_(o.t_) {
  switch (t_) {
    case Type::Null: i_ = 0; break;
    case Type::Bool: b_ = o.b_; break;
    case Type::Int: i_ = o.i_; break;
    case Type::Float: d_ = o.d_; break;
    case Type::String:
    case Type::BigInt: new (&s_) std::string(std::move(o.s_)); break;
    case Type::Array: new (&a_) JsonArray(std::move(o.a_)); break;
    case Type::Object: new (&o_) JsonObject(std::move(o.o_)); break;
  }
}

Json& Json::operator=(const Json& o) {
  if (this != &o) {
    Json tmp(o);
    *this = std::move(tmp);
  }
  return *this;
}

Json& Json::operator=(Json&& o) noexcept {
  if (this == &o) return *this;
  // o may be a descendant of *this: move it out first.
  Json tmp(std::move(o));
  destroy();
  t_ = tmp.t_;
  switch (t_) {
    case Type::Null: i_ = 0; break;
    case Type::Bool: b_ = tmp.b_; break;
    case Type::Int: i_ = tmp.i_; break;
    case Type::Float: d_ = tmp.d_; break;
    case Type::String:
    case Type::BigInt: new (&s_) std::string(std::move(tmp.s_)); break;
    case Type::Array: new (&a_) JsonArray(std::move(tmp.a_)); break;
    case Type::Object: new (&o_) JsonObject(std::move(tmp.o_)); break;
  }
  return *this;
}

void Json::init_big_u64(uint64_t v) {
  t_ = Type::BigInt;
  new (&s_) std::string(std::to_string(v));
}

static constexpr size_t kMaxIntDigits = 4300;  // sys.get_int_max_str_digits() default

static std::string int_limit_msg(size_t ndigits) {
  return "Exceeds the limit (4300 digits) for integer string conversion: value has " + std::to_string(ndigits) +
         " digits; use sys.set_int_max_str_digits() to increase the limit";
}

Json Json::big_int_unchecked(std::string canonical_digits) noexcept {
  Json j;
  j.t_ = Type::BigInt;
  new (&j.s_) std::string(std::move(canonical_digits));
  return j;
}

// text: -?[0-9]+ (validated by the caller). Normalised like Python int(): leading zeros dropped, "-0" -> 0.
static Json make_integer(std::string_view text) {
  bool neg = !text.empty() && text[0] == '-';
  std::string_view d = neg ? text.substr(1) : text;
  size_t nz = 0;
  while (nz + 1 < d.size() && d[nz] == '0') ++nz;
  d.remove_prefix(nz);
  if (d.size() > kMaxIntDigits) throw JsonError(JsonError::Kind::Value, int_limit_msg(d.size()));
  if (d.size() <= 18) {
    int64_t v = 0;
    for (char c : d) v = v * 10 + (c - '0');
    return Json(neg ? -v : v);
  }
  if (d.size() == 19) {
    static constexpr std::string_view kMax = "9223372036854775807";
    static constexpr std::string_view kMinAbs = "9223372036854775808";
    if (!neg ? d <= kMax : d <= kMinAbs) {
      uint64_t u = 0;
      for (char c : d) u = u * 10 + static_cast<uint64_t>(c - '0');
      return Json(neg ? static_cast<int64_t>(0 - u) : static_cast<int64_t>(u));
    }
  }
  std::string s;
  s.reserve(d.size() + 1);
  if (neg) s.push_back('-');
  s.append(d);
  return Json::big_int_unchecked(std::move(s));
}

Json Json::integer_from_text(std::string_view digits) {
  std::string_view d = digits;
  if (!d.empty() && d[0] == '-') d.remove_prefix(1);
  bool ok = !d.empty();
  for (char c : d) ok = ok && c >= '0' && c <= '9';
  if (!ok) throw JsonError(JsonError::Kind::Value, "invalid literal for int() with base 10: '" + std::string(digits) + "'");
  return make_integer(digits);
}

Json Json::object(std::initializer_list<std::pair<std::string_view, Json>> items) {
  JsonObject o;
  o.reserve(items.size());
  for (auto& kv : items) o.set(kv.first, kv.second);
  return Json(std::move(o));
}

const char* Json::type_name() const noexcept {
  switch (t_) {
    case Type::Null: return "null";
    case Type::Bool: return "bool";
    case Type::Int: return "int";
    case Type::BigInt: return "bigint";
    case Type::Float: return "float";
    case Type::String: return "string";
    case Type::Array: return "array";
    case Type::Object: return "object";
  }
  return "?";
}

void Json::type_error(const char* want) const {
  throw JsonError(JsonError::Kind::Type, std::string("fe::Json: expected ") + want + ", have " + type_name());
}

bool Json::as_bool() const {
  if (t_ != Type::Bool) type_error("bool");
  return b_;
}
int64_t Json::as_int() const {
  if (t_ != Type::Int) type_error("int");
  return i_;
}

static locale_t c_locale() {
  static locale_t loc = newlocale(LC_ALL_MASK, "C", (locale_t)0);
  return loc;
}

double Json::as_double() const {
  switch (t_) {
    case Type::Float: return d_;
    case Type::Int: return static_cast<double>(i_);  // correctly rounded (round-half-even), == Python float(int)
    case Type::BigInt: {
      double d = strtod_l(s_.c_str(), nullptr, c_locale());
      if (std::isinf(d)) throw JsonError(JsonError::Kind::Value, "int too large to convert to float");
      return d;
    }
    default: type_error("number");
  }
}
const std::string& Json::as_string() const {
  if (t_ != Type::String) type_error("string");
  return s_;
}
std::string& Json::as_string() {
  if (t_ != Type::String) type_error("string");
  return s_;
}
const std::string& Json::big_int_text() const {
  if (t_ != Type::BigInt) type_error("bigint");
  return s_;
}
const JsonArray& Json::as_array() const {
  if (t_ != Type::Array) type_error("array");
  return a_;
}
JsonArray& Json::as_array() {
  if (t_ != Type::Array) type_error("array");
  return a_;
}
const JsonObject& Json::as_object() const {
  if (t_ != Type::Object) type_error("object");
  return o_;
}
JsonObject& Json::as_object() {
  if (t_ != Type::Object) type_error("object");
  return o_;
}

const Json* Json::get(std::string_view key) const noexcept { return t_ == Type::Object ? o_.find(key) : nullptr; }
Json* Json::get(std::string_view key) noexcept { return t_ == Type::Object ? o_.find(key) : nullptr; }

const Json& Json::operator[](std::string_view key) const noexcept {
  const Json* p = get(key);
  return p ? *p : null_ref();
}
const Json& Json::operator[](size_t i) const noexcept {
  if (t_ != Type::Array || i >= a_.size()) return null_ref();
  return a_[i];
}

Json& Json::member(std::string_view key) {
  if (t_ == Type::Null) *this = Json::object();
  if (t_ != Type::Object) type_error("object");
  return o_.get_or_insert(key);
}

std::optional<std::string_view> Json::get_string(std::string_view key) const noexcept {
  const Json* p = get(key);
  if (!p || p->t_ != Type::String) return std::nullopt;
  return std::string_view(p->s_);
}
std::optional<int64_t> Json::get_int(std::string_view key) const noexcept {
  const Json* p = get(key);
  if (!p || p->t_ != Type::Int) return std::nullopt;
  return p->i_;
}
std::optional<double> Json::get_number(std::string_view key) const noexcept {
  const Json* p = get(key);
  if (!p) return std::nullopt;
  if (p->t_ == Type::Float) return p->d_;
  if (p->t_ == Type::Int) return static_cast<double>(p->i_);
  return std::nullopt;
}
std::optional<bool> Json::get_bool(std::string_view key) const noexcept {
  const Json* p = get(key);
  if (!p || p->t_ != Type::Bool) return std::nullopt;
  return p->b_;
}
size_t Json::size() const noexcept {
  if (t_ == Type::Array) return a_.size();
  if (t_ == Type::Object) return o_.size();
  return 0;
}

Json& Json::push_back(Json v) {
  if (t_ == Type::Null) *this = Json::array();
  if (t_ != Type::Array) type_error("array");
  a_.push_back(std::move(v));
  return a_.back();
}
Json& Json::set(std::string_view key, Json v) {
  if (t_ == Type::Null) *this = Json::object();
  if (t_ != Type::Object) type_error("object");
  return o_.set(key, std::move(v));
}
bool Json::erase(std::string_view key) {
  if (t_ != Type::Object) return false;
  return o_.erase(key);
}

bool operator==(const Json& a, const Json& b) noexcept {
  std::vector<std::pair<const Json*, const Json*>> st;  // iterative: no recursion per nesting level
  st.emplace_back(&a, &b);
  while (!st.empty()) {
    auto [x, y] = st.back();
    st.pop_back();
    if (x->t_ != y->t_) return false;
    switch (x->t_) {
      case Json::Type::Null: break;
      case Json::Type::Bool: if (x->b_ != y->b_) return false; break;
      case Json::Type::Int: if (x->i_ != y->i_) return false; break;
      case Json::Type::Float: if (!(x->d_ == y->d_)) return false; break;
      case Json::Type::String:
      case Json::Type::BigInt: if (x->s_ != y->s_) return false; break;
      case Json::Type::Array:
        if (x->a_.size() != y->a_.size()) return false;
        for (size_t k = 0; k < x->a_.size(); ++k) st.emplace_back(&x->a_[k], &y->a_[k]);
        break;
      case Json::Type::Object:
        if (x->o_.size() != y->o_.size()) return false;
        for (size_t k = 0; k < x->o_.size(); ++k) {
          const auto& mx = x->o_.at_index(k);
          const auto& my = y->o_.at_index(k);
          if (mx.first != my.first) return false;
          st.emplace_back(&mx.second, &my.second);
        }
        break;
    }
  }
  return true;
}

static bool py_num(const Json& j) { return j.is_bool() || j.is_number(); }

// Python == between two numbers (bool/int/BigInt/float), exact like CPython (int vs float compares exactly).
static bool py_num_equal(const Json& a, const Json& b) {
  auto kind = [](const Json& j) { return j.is_float() ? 1 : j.is_big_int() ? 2 : 0; };
  auto ival = [](const Json& j) -> int64_t { return j.is_bool() ? (j.as_bool() ? 1 : 0) : j.as_int(); };
  int ka = kind(a), kb = kind(b);
  if (ka == 0 && kb == 0) return ival(a) == ival(b);
  if (ka == 2 && kb == 2) return a.big_int_text() == b.big_int_text();
  if (ka == 1 && kb == 1) return a.as_double() == b.as_double();
  if (ka == 2 || kb == 2) {
    if (ka == 0 || kb == 0) return false;  // BigInt is never within int64
    double d = ka == 1 ? a.as_double() : b.as_double();
    const std::string& big = ka == 2 ? a.big_int_text() : b.big_int_text();
    if (!std::isfinite(d) || std::floor(d) != d) return false;
    char buf[400];
    int n = snprintf(buf, sizeof buf, "%.0f", d);
    return n > 0 && big == std::string_view(buf, static_cast<size_t>(n));
  }
  double d = ka == 1 ? a.as_double() : b.as_double();
  int64_t i = ka == 0 ? ival(a) : ival(b);
  if (!std::isfinite(d) || std::floor(d) != d) return false;
  if (d < -9223372036854775808.0 || d >= 9223372036854775808.0) return false;
  return static_cast<int64_t>(d) == i;
}

bool Json::py_equal(const Json& o) const noexcept {
  std::vector<std::pair<const Json*, const Json*>> st;
  st.emplace_back(this, &o);
  while (!st.empty()) {
    auto [x, y] = st.back();
    st.pop_back();
    if (py_num(*x) && py_num(*y)) {
      if (!py_num_equal(*x, *y)) return false;
      continue;
    }
    if (x->t_ != y->t_) return false;
    switch (x->t_) {
      case Type::Null: break;
      case Type::String: if (x->s_ != y->s_) return false; break;
      case Type::Array:
        if (x->a_.size() != y->a_.size()) return false;
        for (size_t k = 0; k < x->a_.size(); ++k) st.emplace_back(&x->a_[k], &y->a_[k]);
        break;
      case Type::Object:
        if (x->o_.size() != y->o_.size()) return false;
        for (const auto& kv : x->o_) {
          const Json* p = y->o_.find(kv.first);
          if (!p) return false;
          st.emplace_back(&kv.second, p);
        }
        break;
      default: return false;
    }
  }
  return true;
}

// ============================================================================================
// Unicode helpers
// ============================================================================================

namespace {

inline bool is_cont(unsigned char c) { return (c & 0xC0) == 0x80; }

// Python's UTF-8 decoder (Objects/stringlib/codecs.h) + 'surrogatepass'. Returns the first error, if any.
struct Utf8Err {
  bool ok = true;
  size_t start = 0, end = 0;  // byte range
  const char* reason = nullptr;
};

Utf8Err py_utf8_check(const unsigned char* s0, size_t n, bool surrogatepass) {
  Utf8Err e;
  const unsigned char* s = s0;
  const unsigned char* end = s0 + n;
  while (s < end) {
    // ASCII run, 8 at a time
    while (end - s >= 8) {
      uint64_t w;
      std::memcpy(&w, s, 8);
      if (w & 0x8080808080808080ULL) break;
      s += 8;
    }
    if (s >= end) break;
    unsigned char ch = *s;
    if (ch < 0x80) { ++s; continue; }
    int code = -1;  // 0 eod, 1 invalid start, 2..4 invalid continuation (ch-1 bytes)
    size_t len = 0;
    if (ch < 0xC2) code = 1;
    else if (ch < 0xE0) {
      if (end - s < 2) code = 0;
      else if (!is_cont(s[1])) code = 2;
      else len = 2;
    } else if (ch < 0xF0) {
      if (end - s < 3) {
        if (end - s < 2) code = 0;
        else {
          unsigned char c2 = s[1];
          if (!is_cont(c2) || (c2 < 0xA0 ? ch == 0xE0 : ch == 0xED)) code = 2;
          else code = 0;
        }
      } else {
        unsigned char c2 = s[1], c3 = s[2];
        if (!is_cont(c2)) code = 2;
        else if (ch == 0xE0 && c2 < 0xA0) code = 2;
        else if (ch == 0xED && c2 >= 0xA0) code = 2;  // surrogate: handled by surrogatepass below
        else if (!is_cont(c3)) code = 3;
        else len = 3;
      }
    } else if (ch < 0xF5) {
      if (end - s < 4) {
        if (end - s < 2) code = 0;
        else {
          unsigned char c2 = s[1];
          if (!is_cont(c2) || (c2 < 0x90 ? ch == 0xF0 : ch == 0xF4)) code = 2;
          else if (end - s < 3) code = 0;
          else if (!is_cont(s[2])) code = 3;
          else code = 0;
        }
      } else {
        unsigned char c2 = s[1], c3 = s[2], c4 = s[3];
        if (!is_cont(c2)) code = 2;
        else if (ch == 0xF0 && c2 < 0x90) code = 2;
        else if (ch == 0xF4 && c2 >= 0x90) code = 2;
        else if (!is_cont(c3)) code = 3;
        else if (!is_cont(c4)) code = 4;
        else len = 4;
      }
    } else code = 1;
    if (len) { s += len; continue; }
    // error: surrogatepass handler first
    if (surrogatepass && end - s >= 3 && (s[0] & 0xF0) == 0xE0 && is_cont(s[1]) && is_cont(s[2])) {
      uint32_t c = ((s[0] & 0x0Fu) << 12) | ((s[1] & 0x3Fu) << 6) | (s[2] & 0x3Fu);
      if (c >= 0xD800 && c <= 0xDFFF) { s += 3; continue; }
    }
    e.ok = false;
    e.start = static_cast<size_t>(s - s0);
    if (code == 0) { e.end = n; e.reason = "unexpected end of data"; }
    else if (code == 1) { e.end = e.start + 1; e.reason = "invalid start byte"; }
    else { e.end = e.start + static_cast<size_t>(code - 1); e.reason = "invalid continuation byte"; }
    return e;
  }
  return e;
}

[[noreturn]] void throw_unicode(const char* codec, const unsigned char* bytes, const Utf8Err& e) {
  char buf[256];
  if (e.end - e.start == 1)
    snprintf(buf, sizeof buf, "'%s' codec can't decode byte 0x%02x in position %zu: %s", codec, bytes[e.start], e.start,
             e.reason);
  else
    snprintf(buf, sizeof buf, "'%s' codec can't decode bytes in position %zu-%zu: %s", codec, e.start, e.end - 1,
             e.reason);
  throw JsonError(JsonError::Kind::Unicode, buf);
}

inline void append_cp(std::string& out, uint32_t cp) {  // generalized UTF-8 (surrogates as 3 bytes)
  if (cp < 0x80) out.push_back(static_cast<char>(cp));
  else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// UTF-16 / UTF-32 -> generalized UTF-8 with 'surrogatepass' (lone surrogates kept). Approximate error messages.
std::string transcode_utf16(const unsigned char* p, size_t n, bool big_endian, const char* codec, size_t off = 0) {
  std::string out;
  out.reserve(n);
  size_t i = 0;
  auto unit = [&](size_t k) -> uint32_t {
    return big_endian ? (uint32_t(p[k]) << 8) | p[k + 1] : (uint32_t(p[k + 1]) << 8) | p[k];
  };
  while (i + 1 < n) {
    uint32_t u = unit(i);
    if (u >= 0xD800 && u <= 0xDBFF && i + 3 < n) {
      uint32_t u2 = unit(i + 2);
      if (u2 >= 0xDC00 && u2 <= 0xDFFF) {
        append_cp(out, 0x10000 + ((u - 0xD800) << 10) + (u2 - 0xDC00));
        i += 4;
        continue;
      }
    }
    append_cp(out, u);
    i += 2;
  }
  if (i < n) {
    char buf[200];
    snprintf(buf, sizeof buf, "'%s' codec can't decode byte 0x%02x in position %zu: truncated data", codec, p[i], i + off);
    throw JsonError(JsonError::Kind::Unicode, buf);
  }
  return out;
}

std::string transcode_utf32(const unsigned char* p, size_t n, bool big_endian, const char* codec, size_t off = 0) {
  std::string out;
  out.reserve(n);
  size_t i = 0;
  for (; i + 3 < n; i += 4) {
    uint32_t u = big_endian ? (uint32_t(p[i]) << 24) | (uint32_t(p[i + 1]) << 16) | (uint32_t(p[i + 2]) << 8) | p[i + 3]
                            : (uint32_t(p[i + 3]) << 24) | (uint32_t(p[i + 2]) << 16) | (uint32_t(p[i + 1]) << 8) | p[i];
    if (u > 0x10FFFF) {
      char buf[200];
      snprintf(buf, sizeof buf, "'%s' codec can't decode bytes in position %zu-%zu: code point not in range(0x110000)",
               codec, i + off, i + 3 + off);
      throw JsonError(JsonError::Kind::Unicode, buf);
    }
    append_cp(out, u);
  }
  if (i < n) {
    char buf[200];
    snprintf(buf, sizeof buf, "'%s' codec can't decode bytes in position %zu-%zu: truncated data", codec, i + off,
             n - 1 + off);
    throw JsonError(JsonError::Kind::Unicode, buf);
  }
  return out;
}

// ============================================================================================
// Python-exact scanner (CPython 3.12 Modules/_json.c + json/decoder.py), on generalized UTF-8 text.
// ============================================================================================

class PyScanner {
 public:
  PyScanner(const char* s, size_t n, int max_depth) : s_(s), n_(n), max_depth_(max_depth) {}

  Json parse_document() {
    size_t i = 0;
    skip_ws(i);
    Json v = parse_value(i);
    skip_ws(i);
    if (i != n_) fail("Extra data", i);
    return v;
  }

  [[noreturn]] void fail(const char* msg, size_t byte_pos) const {
    // char index, line, column (Python counts code points)
    int64_t pos = 0, lineno = 1, last_nl = -1;
    for (size_t k = 0; k < byte_pos && k < n_; ++k) {
      unsigned char c = static_cast<unsigned char>(s_[k]);
      if (!is_cont(c)) {
        if (c == '\n') { ++lineno; last_nl = pos; }
        ++pos;
      }
    }
    int64_t colno = pos - last_nl;
    char buf[160];
    snprintf(buf, sizeof buf, ": line %lld column %lld (char %lld)", (long long)lineno, (long long)colno, (long long)pos);
    throw JsonError(JsonError::Kind::Decode, std::string(msg) + buf, msg, pos, lineno, colno,
                    static_cast<int64_t>(byte_pos));
  }

 private:
  const char* s_;
  size_t n_;
  int max_depth_;

  struct Frame {
    Json value;
    std::string key;
    bool is_obj;
  };

  inline void skip_ws(size_t& i) const {
    while (i < n_) {
      char c = s_[i];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++i;
      else break;
    }
  }

  [[noreturn]] void recursion_error(bool obj) const {
    throw JsonError(JsonError::Kind::Recursion,
                    std::string("maximum recursion depth exceeded while decoding a JSON ") + (obj ? "object" : "array") +
                        " from a unicode string");
  }

  // Key after an opening '"' or ',' : expects '"', reads key, ':' and following whitespace.
  void read_key(size_t& i, std::string& key) {
    if (i >= n_ || s_[i] != '"') fail("Expecting property name enclosed in double quotes", i);
    ++i;
    key = scan_string(i);
    skip_ws(i);
    if (i >= n_ || s_[i] != ':') fail("Expecting ':' delimiter", i);
    ++i;
    skip_ws(i);
  }

  Json parse_value(size_t& i) {
    std::vector<Frame> st;
    Json cur;
    for (;;) {
      // ---- scan_once at i ----
      if (i >= n_) fail("Expecting value", i);
      char c = s_[i];
      if (c == '{' || c == '[') {
        const bool obj = c == '{';
        if (static_cast<int>(st.size()) >= max_depth_) recursion_error(obj);
        ++i;
        skip_ws(i);
        if (i < n_ && s_[i] == (obj ? '}' : ']')) {
          ++i;
          cur = obj ? Json::object() : Json::array();
        } else {
          st.push_back(Frame{obj ? Json::object() : Json::array(), {}, obj});
          if (obj) read_key(i, st.back().key);
          continue;
        }
      } else {
        cur = scan_scalar(i);
      }
      // ---- attach completed value(s) ----
      for (;;) {
        if (st.empty()) return cur;
        Frame& f = st.back();
        if (f.is_obj) f.value.as_object().set(std::move(f.key), std::move(cur));
        else f.value.as_array().push_back(std::move(cur));
        skip_ws(i);
        const char close = f.is_obj ? '}' : ']';
        if (i < n_ && s_[i] == close) {
          ++i;
          cur = std::move(f.value);
          st.pop_back();
          continue;
        }
        if (i >= n_ || s_[i] != ',') fail("Expecting ',' delimiter", i);
        ++i;
        skip_ws(i);
        if (f.is_obj) {
          f.key.clear();
          read_key(i, f.key);
        }
        break;
      }
    }
  }

  static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  }

  // i: first byte after the opening quote. On return i is past the closing quote.
  std::string scan_string(size_t& i) {
    const size_t begin = i - 1;
    std::string out;
    for (;;) {
      size_t j = i;
      while (j < n_) {
        unsigned char d = static_cast<unsigned char>(s_[j]);
        if (d == '"' || d == '\\') break;
        if (d <= 0x1f) fail("Invalid control character at", j);
        ++j;
      }
      if (j >= n_) fail("Unterminated string starting at", begin);
      out.append(s_ + i, j - i);
      if (s_[j] == '"') {
        i = j + 1;
        return out;
      }
      size_t next = j + 1;
      if (next == n_) fail("Unterminated string starting at", begin);
      char c = s_[next];
      if (c != 'u') {
        char r;
        switch (c) {
          case '"': r = '"'; break;
          case '\\': r = '\\'; break;
          case '/': r = '/'; break;
          case 'b': r = '\b'; break;
          case 'f': r = '\f'; break;
          case 'n': r = '\n'; break;
          case 'r': r = '\r'; break;
          case 't': r = '\t'; break;
          default: fail("Invalid \\escape", j);
        }
        out.push_back(r);
        i = next + 1;
        continue;
      }
      ++next;  // first hex digit
      size_t end = next + 4;
      if (end >= n_) fail("Invalid \\uXXXX escape", next - 1);
      uint32_t cp = 0;
      for (size_t k = next; k < end; ++k) {
        int h = hexval(s_[k]);
        if (h < 0) fail("Invalid \\uXXXX escape", end - 5);
        cp = (cp << 4) | static_cast<uint32_t>(h);
      }
      if (cp >= 0xD800 && cp <= 0xDBFF && end + 6 < n_ && s_[end] == '\\' && s_[end + 1] == 'u') {
        uint32_t c2 = 0;
        size_t e2 = end + 6;
        for (size_t k = end + 2; k < e2; ++k) {
          int h = hexval(s_[k]);
          if (h < 0) fail("Invalid \\uXXXX escape", e2 - 5);
          c2 = (c2 << 4) | static_cast<uint32_t>(h);
        }
        if (c2 >= 0xDC00 && c2 <= 0xDFFF) {
          cp = 0x10000 + ((cp - 0xD800) << 10) + (c2 - 0xDC00);
          end = e2;
        }
      }
      append_cp(out, cp);
      i = end;
    }
  }

  bool lit(size_t i, const char* w, size_t len) const {
    // Python: (idx + len - 1 < length) && chars match
    if (i + len - 1 >= n_) return false;
    return std::memcmp(s_ + i, w, len) == 0;
  }

  Json scan_scalar(size_t& i) {
    char c = s_[i];
    switch (c) {
      case '"': {
        ++i;
        return Json(scan_string(i));
      }
      case 'n':
        if (lit(i, "null", 4)) { i += 4; return Json(); }
        break;
      case 't':
        if (lit(i, "true", 4)) { i += 4; return Json(true); }
        break;
      case 'f':
        if (lit(i, "false", 5)) { i += 5; return Json(false); }
        break;
      case 'N':
        if (lit(i, "NaN", 3)) { i += 3; return Json(std::numeric_limits<double>::quiet_NaN()); }
        break;
      case 'I':
        if (lit(i, "Infinity", 8)) { i += 8; return Json(std::numeric_limits<double>::infinity()); }
        break;
      case '-':
        if (lit(i, "-Infinity", 9)) { i += 9; return Json(-std::numeric_limits<double>::infinity()); }
        break;
      default: break;
    }
    return match_number(i);
  }

  static bool dig(char c) { return c >= '0' && c <= '9'; }

  Json match_number(size_t& i) {
    const size_t start = i;
    const size_t end_idx = n_ - 1;  // n_ > i >= 0 here
    size_t idx = i;
    if (s_[idx] == '-') {
      ++idx;
      if (idx > end_idx) fail("Expecting value", start);
    }
    if (s_[idx] >= '1' && s_[idx] <= '9') {
      ++idx;
      while (idx <= end_idx && dig(s_[idx])) ++idx;
    } else if (s_[idx] == '0') {
      ++idx;
    } else {
      fail("Expecting value", start);
    }
    bool is_float = false;
    if (idx < end_idx && s_[idx] == '.' && dig(s_[idx + 1])) {
      is_float = true;
      idx += 2;
      while (idx <= end_idx && dig(s_[idx])) ++idx;
    }
    if (idx < end_idx && (s_[idx] == 'e' || s_[idx] == 'E')) {
      size_t e_start = idx;
      ++idx;
      if (idx < end_idx && (s_[idx] == '-' || s_[idx] == '+')) ++idx;
      while (idx <= end_idx && dig(s_[idx])) ++idx;
      if (dig(s_[idx - 1])) is_float = true;
      else idx = e_start;
    }
    std::string_view num(s_ + start, idx - start);
    i = idx;
    if (is_float) {
      std::string tmp(num);
      return Json(strtod_l(tmp.c_str(), nullptr, c_locale()));
    }
    return make_integer(num);
  }
};

// Decode request bytes as json.loads(bytes) does; returns the text to scan (may point into `storage`).
std::string_view py_decode_bytes(std::string_view in, std::string& storage) {
  const auto* b = reinterpret_cast<const unsigned char*>(in.data());
  const size_t n = in.size();
  auto starts = [&](const char* pfx, size_t len) { return n >= len && std::memcmp(b, pfx, len) == 0; };
  // BOM'd 'utf-16' / 'utf-32': Python reports the endian-specific codec and positions counted from the BOM
  if (starts("\x00\x00\xfe\xff", 4)) { storage = transcode_utf32(b + 4, n - 4, true, "utf-32-be", 4); return storage; }
  if (starts("\xff\xfe\x00\x00", 4)) { storage = transcode_utf32(b + 4, n - 4, false, "utf-32-le", 4); return storage; }
  if (starts("\xfe\xff", 2)) { storage = transcode_utf16(b + 2, n - 2, true, "utf-16-be", 2); return storage; }
  if (starts("\xff\xfe", 2)) { storage = transcode_utf16(b + 2, n - 2, false, "utf-16-le", 2); return storage; }
  const char* codec = "utf-8";
  std::string_view text = in;
  if (starts("\xef\xbb\xbf", 3)) {
    text.remove_prefix(3);
    codec = "utf-8";  // utf-8-sig reports errors as 'utf-8', positions relative to after the BOM
  } else if (n >= 4) {
    if (!b[0]) {
      if (b[1]) storage = transcode_utf16(b, n, true, "utf-16-be");
      else storage = transcode_utf32(b, n, true, "utf-32-be");
      return storage;
    }
    if (!b[1]) {
      if (b[2] || b[3]) storage = transcode_utf16(b, n, false, "utf-16-le");
      else storage = transcode_utf32(b, n, false, "utf-32-le");
      return storage;
    }
  } else if (n == 2) {
    if (!b[0]) { storage = transcode_utf16(b, n, true, "utf-16-be"); return storage; }
    if (!b[1]) { storage = transcode_utf16(b, n, false, "utf-16-le"); return storage; }
  }
  const auto* tb = reinterpret_cast<const unsigned char*>(text.data());
  Utf8Err e = py_utf8_check(tb, text.size(), true);
  if (!e.ok) throw_unicode(codec, tb, e);
  return text;
}

Json parse_fallback(std::string_view in, const JsonParseOptions& opt) {
  std::string storage;
  std::string_view text;
  if (opt.from_bytes) {
    text = py_decode_bytes(in, storage);
  } else {
    const auto* tb = reinterpret_cast<const unsigned char*>(in.data());
    if (in.size() >= 3 && std::memcmp(tb, "\xef\xbb\xbf", 3) == 0)
      throw JsonError(JsonError::Kind::Decode, "Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)",
                      "Unexpected UTF-8 BOM (decode using utf-8-sig)", 0, 1, 1, 0);
    Utf8Err e = py_utf8_check(tb, in.size(), true);
    if (!e.ok) throw_unicode("utf-8", tb, e);  // cannot happen for a real Python str
    text = in;
  }
  PyScanner sc(text.data(), text.size(), opt.max_depth);
  return sc.parse_document();
}

// ============================================================================================
// simdjson fast path
// ============================================================================================

struct FallbackNeeded {};
// simdjson's depth limit for the fast path: from_dom recurses once per level, so this bounds its stack use
// (~40 KB). Deeper documents (legal up to 9997 in Python) are parsed by the iterative PyScanner.
constexpr size_t kFastMaxDepth = 256;

Json from_dom(simdjson::dom::element e) {
  using simdjson::dom::element_type;
  switch (e.type()) {
    case element_type::ARRAY: {
      simdjson::dom::array a = e.get_array().value_unsafe();
      JsonArray arr;
      arr.reserve(a.size());
      for (simdjson::dom::element x : a) arr.push_back(from_dom(x));
      return Json(std::move(arr));
    }
    case element_type::OBJECT: {
      simdjson::dom::object o = e.get_object().value_unsafe();
      JsonObject obj;
      obj.reserve(o.size());
      for (simdjson::dom::key_value_pair kv : o) obj.set(kv.key, from_dom(kv.value));
      return Json(std::move(obj));
    }
    case element_type::INT64: return Json(e.get_int64().value_unsafe());
    case element_type::UINT64: return Json(e.get_uint64().value_unsafe());  // > INT64_MAX -> BigInt
    case element_type::DOUBLE: {
      double d = e.get_double().value_unsafe();
      if (d == 0.0 && std::signbit(d)) throw FallbackNeeded{};  // "-0" (int) vs "-0.0" (float): let the scanner decide
      return Json(d);
    }
    case element_type::STRING: return Json(e.get_string().value_unsafe());
    case element_type::BOOL: return Json(e.get_bool().value_unsafe());
    case element_type::NULL_VALUE: return Json();
  }
  throw FallbackNeeded{};
}

}  // namespace

Json json_parse(std::string_view text) { return json_parse(text, JsonParseOptions{}); }

Json json_parse(std::string_view text, const JsonParseOptions& opt) {
  std::string_view view = text;
  const auto* b = reinterpret_cast<const unsigned char*>(text.data());
  const size_t n = text.size();
  bool fast = n > 0;
  if (n >= 2 && (b[0] == 0 || b[1] == 0 || (b[0] == 0xFF && b[1] == 0xFE) || (b[0] == 0xFE && b[1] == 0xFF)))
    fast = false;  // UTF-16/32 candidates
  if (n >= 3 && std::memcmp(b, "\xef\xbb\xbf", 3) == 0) {
    if (!opt.from_bytes) fast = false;  // str: BOM is an error
    else view.remove_prefix(3);
    // simdjson silently skips a leading BOM itself; Python sees a second one as U+FEFF ("Expecting value")
    if (view.size() >= 3 && std::memcmp(view.data(), "\xef\xbb\xbf", 3) == 0) fast = false;
  }
  if (opt.max_depth < static_cast<int>(kFastMaxDepth)) fast = false;
  if (fast && !view.empty()) {
    try {
      if (view.size() > (64u << 20)) {
        simdjson::dom::parser big;
        if (big.allocate(view.size(), kFastMaxDepth) != simdjson::SUCCESS) throw FallbackNeeded{};
        simdjson::dom::element root;
        if (big.parse(view.data(), view.size()).get(root) == simdjson::SUCCESS) return from_dom(root);
      } else {
        thread_local simdjson::dom::parser tl;
        thread_local bool tl_init = false;
        if (!tl_init) {
          if (tl.allocate(1u << 16, kFastMaxDepth) != simdjson::SUCCESS) throw FallbackNeeded{};
          tl_init = true;
        }
        simdjson::dom::element root;
        if (tl.parse(view.data(), view.size()).get(root) == simdjson::SUCCESS) return from_dom(root);
      }
    } catch (const FallbackNeeded&) {
    }
  }
  return parse_fallback(text, opt);
}

bool utf8_valid(std::string_view s) noexcept {
  return py_utf8_check(reinterpret_cast<const unsigned char*>(s.data()), s.size(), false).ok;
}

size_t py_len(std::string_view s) noexcept {
  size_t n = 0;
  for (unsigned char c : s) n += !is_cont(c);
  return n;
}

}  // namespace fe
