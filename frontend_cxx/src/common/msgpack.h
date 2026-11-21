// msgpack compatible with msgspec 0.22 (vLLM's EngineCore wire format). Owner A0, component COMMON.
//
// Packer: emits exactly what msgspec.msgpack.encode produces for the equivalent Python object:
//   * int: smallest form; >= 0 -> fixint / uint8 / uint16 / uint32 / uint64, < 0 -> negative fixint / int8 / int16 /
//     int32 / int64 (msgspec never writes a signed form for a non-negative int);
//   * float: always float64 (0xcb);  str: fixstr / str8 / str16 / str32;  bytes: bin8 / bin16 / bin32;
//   * list/tuple: fixarray / array16 / array32;  dict: fixmap / map16 / map32;  None / True / False;
//   * msgspec.msgpack.Ext(code, data): fixext1/2/4/8/16 when len is exactly 1/2/4/8/16, else ext8 / ext16 / ext32
//     (vLLM: Ext(3, raw bytes) = CUSTOM_TYPE_RAW_VIEW for tensors below VLLM_MSGPACK_ZERO_COPY_THRESHOLD).
//   * Struct(array_like=True): a positional array (see the golden for omit_defaults truncation of trailing defaults).
// Reader / Unpacker: accepts every msgpack format (incl. float32, all int widths, str8, fixext, timestamps as Ext -1).
// Strings are validated as strict UTF-8 on read (msgspec raises on invalid UTF-8) and on pack_json (a Python str with
// a lone surrogate cannot be encoded by msgspec).
#pragma once

#include <cstdint>
#include <span>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/json.h"

namespace fe {

struct MsgpackError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class MsgpackPacker {
 public:
  MsgpackPacker() = default;
  explicit MsgpackPacker(size_t reserve) { buf_.reserve(reserve); }

  void clear() noexcept { buf_.clear(); }
  const std::string& data() const noexcept { return buf_; }
  std::string& data() noexcept { return buf_; }
  std::string take() noexcept { return std::move(buf_); }
  size_t size() const noexcept { return buf_.size(); }

  void pack_nil() { buf_.push_back('\xc0'); }
  void pack_bool(bool b) { buf_.push_back(b ? '\xc3' : '\xc2'); }
  void pack_int(int64_t v);    // msgspec int encoding (non-negative values use the unsigned forms)
  void pack_uint(uint64_t v);
  void pack_float(double v);   // float64
  void pack_str(std::string_view s);
  void pack_bin(std::string_view b);
  void pack_bin(const void* p, size_t n) { pack_bin(std::string_view(static_cast<const char*>(p), n)); }
  void pack_array_header(uint32_t n);
  void pack_map_header(uint32_t n);
  void pack_ext(int8_t type, std::string_view data);
  void pack_ext(int8_t type, const void* p, size_t n) { pack_ext(type, std::string_view(static_cast<const char*>(p), n)); }
  void pack_raw(std::string_view already_encoded) { buf_.append(already_encoded); }

  // The Json value as msgspec encodes the equivalent Python object (dict -> map with str keys, list -> array,
  // int -> int, float -> float64, ...). BigInt outside [-2**63, 2**64-1] -> MsgpackError (msgspec: OverflowError);
  // a string holding a lone surrogate -> MsgpackError (msgspec: UnicodeEncodeError).
  void pack_json(const Json& v);

  template <class T>
  void pack_int_array(std::span<const T> v) {  // list[int]
    pack_array_header(static_cast<uint32_t>(v.size()));
    for (T x : v) {
      if constexpr (std::is_signed_v<T>) pack_int(static_cast<int64_t>(x));
      else pack_uint(static_cast<uint64_t>(x));
    }
  }
  void pack_str_array(std::span<const std::string> v) {
    pack_array_header(static_cast<uint32_t>(v.size()));
    for (const auto& s : v) pack_str(s);
  }

 private:
  std::string buf_;
  void put8(uint8_t c) { buf_.push_back(static_cast<char>(c)); }
  void put_be(uint64_t v, int nbytes) {
    for (int i = nbytes - 1; i >= 0; --i) buf_.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
  }
};

enum class MpType : uint8_t { Nil, Bool, Int, UInt, Float, Str, Bin, Array, Map, Ext };
const char* mp_type_name(MpType t) noexcept;

// Zero-copy cursor over one buffer. Views returned by read_str/read_bin/read_ext point into the buffer.
class MsgpackReader {
 public:
  MsgpackReader(const void* data, size_t n) : p_(static_cast<const uint8_t*>(data)), b_(p_), e_(p_ + n) {}
  explicit MsgpackReader(std::string_view s) : MsgpackReader(s.data(), s.size()) {}

  bool at_end() const noexcept { return p_ >= e_; }
  size_t offset() const noexcept { return static_cast<size_t>(p_ - b_); }
  size_t remaining() const noexcept { return static_cast<size_t>(e_ - p_); }
  MpType peek() const;  // Int: any int that fits int64 (incl. uint64 forms <= INT64_MAX); UInt: > INT64_MAX

  void read_nil();
  bool try_nil();  // consumes and returns true when the next value is nil
  bool read_bool();
  int64_t read_int();     // any int form; MsgpackError if > INT64_MAX
  uint64_t read_uint();   // any int form; MsgpackError if negative
  double read_float();    // float32 or float64 only
  double read_number();   // int or float -> double
  std::string_view read_str(bool validate_utf8 = true);
  std::string_view read_bin();
  uint32_t read_array();  // header; returns the element count
  uint32_t read_map();    // header; returns the pair count
  std::pair<int8_t, std::string_view> read_ext();
  void skip(int max_depth = 512);  // one complete value

 private:
  const uint8_t* p_;
  const uint8_t* b_;
  const uint8_t* e_;
  [[noreturn]] void fail(const char* what) const;
  void need(size_t n) const {
    if (static_cast<size_t>(e_ - p_) < n) fail("truncated");
  }
  uint64_t be(int nbytes) {
    need(static_cast<size_t>(nbytes));
    uint64_t v = 0;
    for (int i = 0; i < nbytes; ++i) v = (v << 8) | p_[i];
    p_ += nbytes;
    return v;
  }
  std::string_view take(size_t n) {
    need(n);
    std::string_view s(reinterpret_cast<const char*>(p_), n);
    p_ += n;
    return s;
  }
};

// Owning DOM of one msgpack value ("Json-like view").
class MpValue {
 public:
  MpValue() noexcept : t_(MpType::Nil) { i_ = 0; }
  static MpValue nil() { return MpValue(); }
  static MpValue boolean(bool b) { MpValue v; v.t_ = MpType::Bool; v.b_ = b; return v; }
  static MpValue integer(int64_t i) { MpValue v; v.t_ = MpType::Int; v.i_ = i; return v; }
  static MpValue uinteger(uint64_t u) {  // normalised: <= INT64_MAX becomes Int
    MpValue v;
    if (u <= static_cast<uint64_t>(INT64_MAX)) { v.t_ = MpType::Int; v.i_ = static_cast<int64_t>(u); }
    else { v.t_ = MpType::UInt; v.u_ = u; }
    return v;
  }
  static MpValue floating(double d) { MpValue v; v.t_ = MpType::Float; v.d_ = d; return v; }
  static MpValue str(std::string s) { MpValue v; v.t_ = MpType::Str; v.s_ = std::move(s); return v; }
  static MpValue bin(std::string s) { MpValue v; v.t_ = MpType::Bin; v.s_ = std::move(s); return v; }
  static MpValue ext(int8_t type, std::string data) { MpValue v; v.t_ = MpType::Ext; v.i_ = type; v.s_ = std::move(data); return v; }
  static MpValue array(std::vector<MpValue> items) { MpValue v; v.t_ = MpType::Array; v.a_ = std::move(items); return v; }
  // pairs flattened: k0, v0, k1, v1, ...
  static MpValue map_flat(std::vector<MpValue> kv) { MpValue v; v.t_ = MpType::Map; v.a_ = std::move(kv); return v; }

  MpType type() const noexcept { return t_; }
  bool is_nil() const noexcept { return t_ == MpType::Nil; }
  bool is_bool() const noexcept { return t_ == MpType::Bool; }
  bool is_int() const noexcept { return t_ == MpType::Int; }
  bool is_uint() const noexcept { return t_ == MpType::UInt; }
  bool is_integer() const noexcept { return t_ == MpType::Int || t_ == MpType::UInt; }
  bool is_float() const noexcept { return t_ == MpType::Float; }
  bool is_str() const noexcept { return t_ == MpType::Str; }
  bool is_bin() const noexcept { return t_ == MpType::Bin; }
  bool is_array() const noexcept { return t_ == MpType::Array; }
  bool is_map() const noexcept { return t_ == MpType::Map; }
  bool is_ext() const noexcept { return t_ == MpType::Ext; }

  bool as_bool() const;
  int64_t as_int() const;      // Int
  uint64_t as_uint() const;    // UInt, or Int >= 0
  double as_double() const;    // Float, Int, UInt
  std::string_view as_str() const;   // Str
  std::string_view as_bin() const;   // Bin or Ext data
  int8_t ext_type() const;           // Ext
  const std::vector<MpValue>& as_array() const;  // Array
  size_t size() const noexcept;      // array length / map pair count
  const MpValue& operator[](size_t i) const noexcept;  // array element; nil sentinel when out of range
  const MpValue& map_key(size_t i) const;
  const MpValue& map_value(size_t i) const;
  const MpValue* get(std::string_view key) const noexcept;  // map with str keys; nullptr if absent

 private:
  MpType t_;
  union { bool b_; int64_t i_; uint64_t u_; double d_; };
  std::string s_;
  std::vector<MpValue> a_;
  [[noreturn]] void type_error(const char* want) const;
};

MpValue msgpack_unpack(std::string_view data);           // exactly one value; trailing bytes -> MsgpackError
MpValue msgpack_unpack_one(MsgpackReader& r, int max_depth = 512);  // recursive: depth-bounded
// nil/bool/int/float/str/array/map-with-str-keys -> Json (UInt -> BigInt); bin / ext / other keys -> MsgpackError.
Json msgpack_to_json(const MpValue& v);
std::string msgpack_pack_json(const Json& v);           // MsgpackPacker::pack_json in one call

}  // namespace fe
