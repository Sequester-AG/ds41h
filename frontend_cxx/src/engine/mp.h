// Private minimal msgpack for the engine component.
// Byte-compatible with msgspec 0.22 for the types vLLM puts on the wire:
//   ints use the smallest format (positive -> fixint/uint8/16/32/64, negative -> fixint/int8/16/32/64),
//   floats are always float64, str/bin/array/map/ext use the smallest header.
// Switch to fe_common's msgpack once it lands (see tools/REQUESTS.md); the API here is deliberately small.
#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fe::mp {

struct DecodeError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------- packer
class Packer {
 public:
  std::string buf;

  void nil() { buf.push_back(char(0xc0)); }
  void boolean(bool b) { buf.push_back(char(b ? 0xc3 : 0xc2)); }
  void integer(int64_t v) {
    if (v >= 0) return uinteger(uint64_t(v));
    if (v >= -32) { buf.push_back(char(int8_t(v))); return; }
    if (v >= -128) { buf.push_back(char(0xd0)); buf.push_back(char(int8_t(v))); return; }
    if (v >= -32768) { buf.push_back(char(0xd1)); be16(uint16_t(int16_t(v))); return; }
    if (v >= -2147483648LL) { buf.push_back(char(0xd2)); be32(uint32_t(int32_t(v))); return; }
    buf.push_back(char(0xd3)); be64(uint64_t(v));
  }
  void uinteger(uint64_t v) {
    if (v < 128) { buf.push_back(char(v)); return; }
    if (v < 256) { buf.push_back(char(0xcc)); buf.push_back(char(v)); return; }
    if (v < 65536) { buf.push_back(char(0xcd)); be16(uint16_t(v)); return; }
    if (v < 4294967296ULL) { buf.push_back(char(0xce)); be32(uint32_t(v)); return; }
    buf.push_back(char(0xcf)); be64(v);
  }
  void float64(double d) {
    uint64_t u;
    std::memcpy(&u, &d, 8);
    buf.push_back(char(0xcb));
    be64(u);
  }
  void str(std::string_view s) {
    size_t n = s.size();
    if (n < 32) buf.push_back(char(0xa0 | n));
    else if (n < 256) { buf.push_back(char(0xd9)); buf.push_back(char(n)); }
    else if (n < 65536) { buf.push_back(char(0xda)); be16(uint16_t(n)); }
    else { buf.push_back(char(0xdb)); be32(uint32_t(n)); }
    buf.append(s.data(), n);
  }
  void bin(std::string_view s) {
    size_t n = s.size();
    if (n < 256) { buf.push_back(char(0xc4)); buf.push_back(char(n)); }
    else if (n < 65536) { buf.push_back(char(0xc5)); be16(uint16_t(n)); }
    else { buf.push_back(char(0xc6)); be32(uint32_t(n)); }
    buf.append(s.data(), n);
  }
  void array(size_t n) {
    if (n < 16) buf.push_back(char(0x90 | n));
    else if (n < 65536) { buf.push_back(char(0xdc)); be16(uint16_t(n)); }
    else { buf.push_back(char(0xdd)); be32(uint32_t(n)); }
  }
  void map(size_t n) {
    if (n < 16) buf.push_back(char(0x80 | n));
    else if (n < 65536) { buf.push_back(char(0xde)); be16(uint16_t(n)); }
    else { buf.push_back(char(0xdf)); be32(uint32_t(n)); }
  }
  void ext(int8_t type, std::string_view data) {
    size_t n = data.size();
    switch (n) {
      case 1: buf.push_back(char(0xd4)); break;
      case 2: buf.push_back(char(0xd5)); break;
      case 4: buf.push_back(char(0xd6)); break;
      case 8: buf.push_back(char(0xd7)); break;
      case 16: buf.push_back(char(0xd8)); break;
      default:
        if (n < 256) { buf.push_back(char(0xc7)); buf.push_back(char(n)); }
        else if (n < 65536) { buf.push_back(char(0xc8)); be16(uint16_t(n)); }
        else { buf.push_back(char(0xc9)); be32(uint32_t(n)); }
    }
    buf.push_back(char(type));
    buf.append(data.data(), n);
  }
  void raw(std::string_view s) { buf.append(s.data(), s.size()); }

 private:
  void be16(uint16_t v) { buf.push_back(char(v >> 8)); buf.push_back(char(v)); }
  void be32(uint32_t v) { for (int i = 3; i >= 0; --i) buf.push_back(char(v >> (8 * i))); }
  void be64(uint64_t v) { for (int i = 7; i >= 0; --i) buf.push_back(char(v >> (8 * i))); }
};

// ---------------------------------------------------------------- value DOM
enum class Type : uint8_t { Nil, Bool, Int, UInt, Float, Str, Bin, Array, Map, Ext };

struct Value {
  Type type = Type::Nil;
  bool b = false;
  int64_t i = 0;     // Int
  uint64_t u = 0;    // UInt (only for values > INT64_MAX)
  double f = 0;      // Float (float32 is widened)
  int8_t ext_type = 0;
  std::string s;     // Str / Bin / Ext payload
  std::vector<Value> arr;
  std::vector<std::pair<Value, Value>> kv;  // Map, in wire order

  static Value nil() { return Value{}; }
  static Value from_bool(bool v) { Value x; x.type = Type::Bool; x.b = v; return x; }
  static Value from_int(int64_t v) { Value x; x.type = Type::Int; x.i = v; return x; }
  static Value from_double(double v) { Value x; x.type = Type::Float; x.f = v; return x; }
  static Value from_str(std::string v) { Value x; x.type = Type::Str; x.s = std::move(v); return x; }
  static Value make_array() { Value x; x.type = Type::Array; return x; }
  static Value make_map() { Value x; x.type = Type::Map; return x; }

  bool is_nil() const { return type == Type::Nil; }
  bool is_int() const { return type == Type::Int || type == Type::UInt; }
  bool is_num() const { return is_int() || type == Type::Float; }
  bool is_str() const { return type == Type::Str; }
  bool is_array() const { return type == Type::Array; }
  bool is_map() const { return type == Type::Map; }
  int64_t as_int() const;          // Int/UInt(<=INT64_MAX)/Bool; throws otherwise
  uint64_t as_uint() const;
  double as_double() const;        // Int or Float
  const std::string& as_str() const;
  const Value* get(std::string_view key) const;  // Map lookup by str key, nullptr if absent
  Value& set(std::string key, Value v);         // append/replace a str key (keeps order)
  size_t size() const { return type == Type::Array ? arr.size() : kv.size(); }
  const Value& operator[](size_t idx) const { return arr.at(idx); }
};

// Decode one value starting at data[*pos]; advances *pos.
Value unpack(const char* data, size_t len, size_t* pos);
inline Value unpack(std::string_view s) {
  size_t p = 0;
  Value v = unpack(s.data(), s.size(), &p);
  if (p != s.size()) throw DecodeError("trailing bytes after msgpack value");
  return v;
}
// Re-encode a DOM value (msgspec-compatible formats).
void pack(Packer& p, const Value& v);
std::string to_debug_string(const Value& v);  // compact, for logs/tests (not JSON)

}  // namespace fe::mp
