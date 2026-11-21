// msgspec-compatible msgpack Packer / Reader / DOM.
#include "common/msgpack.h"

#include <cstdio>
#include <cstring>

namespace fe {

// ============================================================================================ Packer

void MsgpackPacker::pack_uint(uint64_t x) {
  if (x < 0x80) put8(static_cast<uint8_t>(x));
  else if (x < 0x100) { put8(0xcc); put8(static_cast<uint8_t>(x)); }
  else if (x < 0x10000) { put8(0xcd); put_be(x, 2); }
  else if (x < 0x100000000ULL) { put8(0xce); put_be(x, 4); }
  else { put8(0xcf); put_be(x, 8); }
}

void MsgpackPacker::pack_int(int64_t x) {
  if (x >= 0) { pack_uint(static_cast<uint64_t>(x)); return; }
  if (x >= -32) put8(static_cast<uint8_t>(x & 0xff));
  else if (x >= -128) { put8(0xd0); put8(static_cast<uint8_t>(x & 0xff)); }
  else if (x >= -32768) { put8(0xd1); put_be(static_cast<uint64_t>(x) & 0xffff, 2); }
  else if (x >= -2147483648LL) { put8(0xd2); put_be(static_cast<uint64_t>(x) & 0xffffffffULL, 4); }
  else { put8(0xd3); put_be(static_cast<uint64_t>(x), 8); }
}

void MsgpackPacker::pack_float(double v) {
  uint64_t bits;
  std::memcpy(&bits, &v, 8);
  put8(0xcb);
  put_be(bits, 8);
}

void MsgpackPacker::pack_str(std::string_view s) {
  const size_t n = s.size();
  if (n < 32) put8(static_cast<uint8_t>(0xa0 | n));
  else if (n < 0x100) { put8(0xd9); put8(static_cast<uint8_t>(n)); }
  else if (n < 0x10000) { put8(0xda); put_be(n, 2); }
  else if (n <= 0xffffffffULL) { put8(0xdb); put_be(n, 4); }
  else throw MsgpackError("msgpack: str too long");
  buf_.append(s);
}

void MsgpackPacker::pack_bin(std::string_view b) {
  const size_t n = b.size();
  if (n < 0x100) { put8(0xc4); put8(static_cast<uint8_t>(n)); }
  else if (n < 0x10000) { put8(0xc5); put_be(n, 2); }
  else if (n <= 0xffffffffULL) { put8(0xc6); put_be(n, 4); }
  else throw MsgpackError("msgpack: bin too long");
  buf_.append(b);
}

void MsgpackPacker::pack_array_header(uint32_t n) {
  if (n < 16) put8(static_cast<uint8_t>(0x90 | n));
  else if (n < 0x10000) { put8(0xdc); put_be(n, 2); }
  else { put8(0xdd); put_be(n, 4); }
}

void MsgpackPacker::pack_map_header(uint32_t n) {
  if (n < 16) put8(static_cast<uint8_t>(0x80 | n));
  else if (n < 0x10000) { put8(0xde); put_be(n, 2); }
  else { put8(0xdf); put_be(n, 4); }
}

void MsgpackPacker::pack_ext(int8_t type, std::string_view d) {
  const size_t n = d.size();
  switch (n) {
    case 1: put8(0xd4); break;
    case 2: put8(0xd5); break;
    case 4: put8(0xd6); break;
    case 8: put8(0xd7); break;
    case 16: put8(0xd8); break;
    default:
      if (n < 0x100) { put8(0xc7); put8(static_cast<uint8_t>(n)); }
      else if (n < 0x10000) { put8(0xc8); put_be(n, 2); }
      else if (n <= 0xffffffffULL) { put8(0xc9); put_be(n, 4); }
      else throw MsgpackError("msgpack: ext too long");
  }
  put8(static_cast<uint8_t>(type));
  buf_.append(d);
}

void MsgpackPacker::pack_json(const Json& top) {
  // iterative (explicit stack): deep user JSON (e.g. extra_args) cannot exhaust the thread stack
  std::vector<std::pair<const Json*, size_t>> st;
  const Json* v = &top;
  for (;;) {
    switch (v->type()) {
      case Json::Type::Null: pack_nil(); break;
      case Json::Type::Bool: pack_bool(v->as_bool()); break;
      case Json::Type::Int: pack_int(v->as_int()); break;
      case Json::Type::BigInt: {
        const std::string& t = v->big_int_text();
        // fits uint64? (BigInt is never within int64)
        if (t[0] != '-' && (t.size() < 20 || (t.size() == 20 && t <= "18446744073709551615"))) {
          uint64_t u = 0;
          for (char c : t) u = u * 10 + static_cast<uint64_t>(c - '0');
          pack_uint(u);
          break;
        }
        throw MsgpackError("can't serialize ints < -2**63 or > 2**64 - 1");
      }
      case Json::Type::Float: pack_float(v->as_double()); break;
      case Json::Type::String: {
        const std::string& s = v->as_string();
        if (!utf8_valid(s)) throw MsgpackError("'utf-8' codec can't encode string: surrogates not allowed");
        pack_str(s);
        break;
      }
      case Json::Type::Array:
        pack_array_header(static_cast<uint32_t>(v->size()));
        st.emplace_back(v, 0);
        break;
      case Json::Type::Object:
        pack_map_header(static_cast<uint32_t>(v->size()));
        st.emplace_back(v, 0);
        break;
    }
    for (;;) {
      if (st.empty()) return;
      auto& f = st.back();
      if (f.second == f.first->size()) {
        st.pop_back();
        continue;
      }
      if (f.first->is_array()) {
        v = &f.first->as_array()[f.second];
      } else {
        const auto& m = f.first->as_object().at_index(f.second);
        if (!utf8_valid(m.first)) throw MsgpackError("'utf-8' codec can't encode string: surrogates not allowed");
        pack_str(m.first);
        v = &m.second;
      }
      ++f.second;
      break;
    }
  }
}

std::string msgpack_pack_json(const Json& v) {
  MsgpackPacker p;
  p.pack_json(v);
  return p.take();
}

// ============================================================================================ Reader

const char* mp_type_name(MpType t) noexcept {
  switch (t) {
    case MpType::Nil: return "nil";
    case MpType::Bool: return "bool";
    case MpType::Int: return "int";
    case MpType::UInt: return "uint";
    case MpType::Float: return "float";
    case MpType::Str: return "str";
    case MpType::Bin: return "bin";
    case MpType::Array: return "array";
    case MpType::Map: return "map";
    case MpType::Ext: return "ext";
  }
  return "?";
}

void MsgpackReader::fail(const char* what) const {
  char buf[160];
  snprintf(buf, sizeof buf, "msgpack: %s at byte %zu", what, offset());
  throw MsgpackError(buf);
}

MpType MsgpackReader::peek() const {
  if (p_ >= e_) fail("truncated");
  uint8_t c = *p_;
  if (c <= 0x7f || c >= 0xe0) return MpType::Int;
  if (c <= 0x8f) return MpType::Map;
  if (c <= 0x9f) return MpType::Array;
  if (c <= 0xbf) return MpType::Str;
  switch (c) {
    case 0xc0: return MpType::Nil;
    case 0xc2:
    case 0xc3: return MpType::Bool;
    case 0xc4:
    case 0xc5:
    case 0xc6: return MpType::Bin;
    case 0xc7:
    case 0xc8:
    case 0xc9:
    case 0xd4:
    case 0xd5:
    case 0xd6:
    case 0xd7:
    case 0xd8: return MpType::Ext;
    case 0xca:
    case 0xcb: return MpType::Float;
    case 0xcc:
    case 0xcd:
    case 0xce:
    case 0xd0:
    case 0xd1:
    case 0xd2:
    case 0xd3: return MpType::Int;
    case 0xcf: {
      if (e_ - p_ < 9) fail("truncated");
      return p_[1] & 0x80 ? MpType::UInt : MpType::Int;
    }
    case 0xd9:
    case 0xda:
    case 0xdb: return MpType::Str;
    case 0xdc:
    case 0xdd: return MpType::Array;
    case 0xde:
    case 0xdf: return MpType::Map;
    default: fail("invalid type byte 0xc1");
  }
}

void MsgpackReader::read_nil() {
  need(1);
  if (*p_ != 0xc0) fail("expected nil");
  ++p_;
}

bool MsgpackReader::try_nil() {
  if (p_ < e_ && *p_ == 0xc0) {
    ++p_;
    return true;
  }
  return false;
}

bool MsgpackReader::read_bool() {
  need(1);
  uint8_t c = *p_;
  if (c != 0xc2 && c != 0xc3) fail("expected bool");
  ++p_;
  return c == 0xc3;
}

// returns (negative, magnitude/raw) through the out-params
static inline bool read_any_int(const uint8_t*& p, const uint8_t* e, bool& is_neg, uint64_t& u, int64_t& s) {
  if (p >= e) return false;
  uint8_t c = *p;
  auto be = [&](int n, uint64_t& out) {
    if (e - p < 1 + n) return false;
    out = 0;
    for (int i = 1; i <= n; ++i) out = (out << 8) | p[i];
    p += 1 + n;
    return true;
  };
  uint64_t raw;
  if (c <= 0x7f) { is_neg = false; u = c; ++p; return true; }
  if (c >= 0xe0) { is_neg = true; s = static_cast<int8_t>(c); ++p; return true; }
  switch (c) {
    case 0xcc: if (!be(1, raw)) return false; is_neg = false; u = raw; return true;
    case 0xcd: if (!be(2, raw)) return false; is_neg = false; u = raw; return true;
    case 0xce: if (!be(4, raw)) return false; is_neg = false; u = raw; return true;
    case 0xcf: if (!be(8, raw)) return false; is_neg = false; u = raw; return true;
    case 0xd0: if (!be(1, raw)) return false; s = static_cast<int8_t>(raw); break;
    case 0xd1: if (!be(2, raw)) return false; s = static_cast<int16_t>(raw); break;
    case 0xd2: if (!be(4, raw)) return false; s = static_cast<int32_t>(raw); break;
    case 0xd3: if (!be(8, raw)) return false; s = static_cast<int64_t>(raw); break;
    default: return false;
  }
  if (s >= 0) { is_neg = false; u = static_cast<uint64_t>(s); }
  else is_neg = true;
  return true;
}

int64_t MsgpackReader::read_int() {
  bool neg;
  uint64_t u = 0;
  int64_t s = 0;
  const uint8_t* q = p_;
  if (!read_any_int(q, e_, neg, u, s)) fail(p_ < e_ ? "expected int" : "truncated");
  if (!neg && u > static_cast<uint64_t>(INT64_MAX)) fail("int out of int64 range");
  p_ = q;
  return neg ? s : static_cast<int64_t>(u);
}

uint64_t MsgpackReader::read_uint() {
  bool neg;
  uint64_t u = 0;
  int64_t s = 0;
  const uint8_t* q = p_;
  if (!read_any_int(q, e_, neg, u, s)) fail(p_ < e_ ? "expected int" : "truncated");
  if (neg) fail("negative int where unsigned expected");
  p_ = q;
  return u;
}

double MsgpackReader::read_float() {
  need(1);
  uint8_t c = *p_;
  if (c == 0xcb) {
    ++p_;
    uint64_t b = be(8);
    double d;
    std::memcpy(&d, &b, 8);
    return d;
  }
  if (c == 0xca) {
    ++p_;
    uint32_t b = static_cast<uint32_t>(be(4));
    float f;
    std::memcpy(&f, &b, 4);
    return static_cast<double>(f);
  }
  fail("expected float");
}

double MsgpackReader::read_number() {
  MpType t = peek();
  if (t == MpType::Float) return read_float();
  if (t == MpType::UInt) return static_cast<double>(read_uint());
  if (t == MpType::Int) return static_cast<double>(read_int());
  fail("expected number");
}

std::string_view MsgpackReader::read_str(bool validate_utf8) {
  need(1);
  uint8_t c = *p_;
  size_t n;
  if (c >= 0xa0 && c <= 0xbf) { ++p_; n = c & 0x1f; }
  else if (c == 0xd9) { ++p_; n = be(1); }
  else if (c == 0xda) { ++p_; n = be(2); }
  else if (c == 0xdb) { ++p_; n = be(4); }
  else fail("expected str");
  std::string_view s = take(n);
  if (validate_utf8 && !utf8_valid(s)) fail("invalid utf-8 in str");
  return s;
}

std::string_view MsgpackReader::read_bin() {
  need(1);
  uint8_t c = *p_;
  size_t n;
  if (c == 0xc4) { ++p_; n = be(1); }
  else if (c == 0xc5) { ++p_; n = be(2); }
  else if (c == 0xc6) { ++p_; n = be(4); }
  else fail("expected bin");
  return take(n);
}

uint32_t MsgpackReader::read_array() {
  need(1);
  uint8_t c = *p_;
  if (c >= 0x90 && c <= 0x9f) { ++p_; return c & 0x0f; }
  if (c == 0xdc) { ++p_; return static_cast<uint32_t>(be(2)); }
  if (c == 0xdd) { ++p_; return static_cast<uint32_t>(be(4)); }
  fail("expected array");
}

uint32_t MsgpackReader::read_map() {
  need(1);
  uint8_t c = *p_;
  if (c >= 0x80 && c <= 0x8f) { ++p_; return c & 0x0f; }
  if (c == 0xde) { ++p_; return static_cast<uint32_t>(be(2)); }
  if (c == 0xdf) { ++p_; return static_cast<uint32_t>(be(4)); }
  fail("expected map");
}

std::pair<int8_t, std::string_view> MsgpackReader::read_ext() {
  need(1);
  uint8_t c = *p_;
  size_t n;
  switch (c) {
    case 0xd4: ++p_; n = 1; break;
    case 0xd5: ++p_; n = 2; break;
    case 0xd6: ++p_; n = 4; break;
    case 0xd7: ++p_; n = 8; break;
    case 0xd8: ++p_; n = 16; break;
    case 0xc7: ++p_; n = be(1); break;
    case 0xc8: ++p_; n = be(2); break;
    case 0xc9: ++p_; n = be(4); break;
    default: fail("expected ext");
  }
  need(1);
  int8_t type = static_cast<int8_t>(*p_++);
  return {type, take(n)};
}

void MsgpackReader::skip(int max_depth) {
  if (max_depth <= 0) fail("nesting too deep");
  switch (peek()) {
    case MpType::Nil: read_nil(); return;
    case MpType::Bool: read_bool(); return;
    case MpType::Int: read_int(); return;
    case MpType::UInt: read_uint(); return;
    case MpType::Float: read_float(); return;
    case MpType::Str: read_str(false); return;
    case MpType::Bin: read_bin(); return;
    case MpType::Ext: read_ext(); return;
    case MpType::Array: {
      uint32_t n = read_array();
      for (uint32_t i = 0; i < n; ++i) skip(max_depth - 1);
      return;
    }
    case MpType::Map: {
      uint32_t n = read_map();
      for (uint32_t i = 0; i < 2 * n; ++i) skip(max_depth - 1);
      return;
    }
  }
}

// ============================================================================================ DOM

void MpValue::type_error(const char* want) const {
  throw MsgpackError(std::string("msgpack value: expected ") + want + ", have " + mp_type_name(t_));
}
bool MpValue::as_bool() const {
  if (t_ != MpType::Bool) type_error("bool");
  return b_;
}
int64_t MpValue::as_int() const {
  if (t_ != MpType::Int) type_error("int");
  return i_;
}
uint64_t MpValue::as_uint() const {
  if (t_ == MpType::UInt) return u_;
  if (t_ == MpType::Int && i_ >= 0) return static_cast<uint64_t>(i_);
  type_error("non-negative int");
}
double MpValue::as_double() const {
  if (t_ == MpType::Float) return d_;
  if (t_ == MpType::Int) return static_cast<double>(i_);
  if (t_ == MpType::UInt) return static_cast<double>(u_);
  type_error("number");
}
std::string_view MpValue::as_str() const {
  if (t_ != MpType::Str) type_error("str");
  return s_;
}
std::string_view MpValue::as_bin() const {
  if (t_ != MpType::Bin && t_ != MpType::Ext) type_error("bin");
  return s_;
}
int8_t MpValue::ext_type() const {
  if (t_ != MpType::Ext) type_error("ext");
  return static_cast<int8_t>(i_);
}
const std::vector<MpValue>& MpValue::as_array() const {
  if (t_ != MpType::Array) type_error("array");
  return a_;
}
size_t MpValue::size() const noexcept {
  if (t_ == MpType::Array) return a_.size();
  if (t_ == MpType::Map) return a_.size() / 2;
  return 0;
}
const MpValue& MpValue::operator[](size_t i) const noexcept {
  static const MpValue nil;
  if (t_ != MpType::Array || i >= a_.size()) return nil;
  return a_[i];
}
const MpValue& MpValue::map_key(size_t i) const {
  if (t_ != MpType::Map) type_error("map");
  return a_.at(2 * i);
}
const MpValue& MpValue::map_value(size_t i) const {
  if (t_ != MpType::Map) type_error("map");
  return a_.at(2 * i + 1);
}
const MpValue* MpValue::get(std::string_view key) const noexcept {
  if (t_ != MpType::Map) return nullptr;
  const MpValue* found = nullptr;
  for (size_t i = 0; i + 1 < a_.size(); i += 2)  // last duplicate wins (Python dict)
    if (a_[i].t_ == MpType::Str && a_[i].s_ == key) found = &a_[i + 1];
  return found;
}

MpValue msgpack_unpack_one(MsgpackReader& r, int max_depth) {
  if (max_depth <= 0) throw MsgpackError("msgpack: nesting too deep");
  switch (r.peek()) {
    case MpType::Nil: r.read_nil(); return MpValue::nil();
    case MpType::Bool: return MpValue::boolean(r.read_bool());
    case MpType::Int: return MpValue::integer(r.read_int());
    case MpType::UInt: return MpValue::uinteger(r.read_uint());
    case MpType::Float: return MpValue::floating(r.read_float());
    case MpType::Str: return MpValue::str(std::string(r.read_str(true)));
    case MpType::Bin: return MpValue::bin(std::string(r.read_bin()));
    case MpType::Ext: {
      auto [t, d] = r.read_ext();
      return MpValue::ext(t, std::string(d));
    }
    case MpType::Array: {
      uint32_t n = r.read_array();
      if (n > r.remaining()) throw MsgpackError("msgpack: truncated (array length)");
      std::vector<MpValue> items;
      items.reserve(n);
      for (uint32_t i = 0; i < n; ++i) items.push_back(msgpack_unpack_one(r, max_depth - 1));
      return MpValue::array(std::move(items));
    }
    case MpType::Map: {
      uint32_t n = r.read_map();
      if (2ULL * n > r.remaining()) throw MsgpackError("msgpack: truncated (map length)");
      std::vector<MpValue> kv;
      kv.reserve(2 * static_cast<size_t>(n));
      for (uint32_t i = 0; i < 2 * n; ++i) kv.push_back(msgpack_unpack_one(r, max_depth - 1));
      return MpValue::map_flat(std::move(kv));
    }
  }
  throw MsgpackError("msgpack: bad value");
}

MpValue msgpack_unpack(std::string_view data) {
  MsgpackReader r(data);
  MpValue v = msgpack_unpack_one(r);
  if (!r.at_end()) {
    char buf[96];
    snprintf(buf, sizeof buf, "msgpack: trailing bytes at %zu", r.offset());
    throw MsgpackError(buf);
  }
  return v;
}

Json msgpack_to_json(const MpValue& v) {
  switch (v.type()) {
    case MpType::Nil: return Json();
    case MpType::Bool: return Json(v.as_bool());
    case MpType::Int: return Json(v.as_int());
    case MpType::UInt: return Json(v.as_uint());
    case MpType::Float: return Json(v.as_double());
    case MpType::Str: return Json(v.as_str());
    case MpType::Array: {
      JsonArray a;
      a.reserve(v.size());
      for (const MpValue& x : v.as_array()) a.push_back(msgpack_to_json(x));
      return Json(std::move(a));
    }
    case MpType::Map: {
      JsonObject o;
      o.reserve(v.size());
      for (size_t i = 0; i < v.size(); ++i) {
        const MpValue& k = v.map_key(i);
        if (!k.is_str()) throw MsgpackError("msgpack_to_json: non-str map key");
        o.set(k.as_str(), msgpack_to_json(v.map_value(i)));
      }
      return Json(std::move(o));
    }
    case MpType::Bin:
    case MpType::Ext: throw MsgpackError("msgpack_to_json: bin/ext has no JSON form");
  }
  return Json();
}

}  // namespace fe
