// msgpack.h -- msgpack writer / reader with msgspec's exact encoding choices.
//
// The V1 wire (vLLM EngineCore <-> frontend) is msgspec.msgpack. Byte identity with msgspec needs its choices, which
// this writer copies (msgspec 0.22 _core.c mpack_encode_*; checked byte for byte by tests/test_wire.cpp against vLLM's
// own MsgpackEncoder output):
//   ints     positive: fixint < 128, uint8 < 256, uint16 < 65536, uint32 < 2^32, else uint64;
//            negative: fixint >= -32, int8 >= -128, int16 >= -32768, int32 >= -2^31, else int64
//   floats   always float64 (0xcb), big-endian IEEE bits (no float32 shortening)
//   str      fixstr < 32 bytes, str8 < 256, str16 < 65536, else str32
//   bin      bin8 / bin16 / bin32
//   array    fixarray < 16, array16 < 65536, else array32;  map: fixmap < 16, map16, map32
//   ext      fixext 1 / 2 / 4 / 8 / 16 for those exact sizes, else ext8 / ext16 / ext32
//   bool     0xc3 / 0xc2;  None: 0xc0
// No exceptions: the reader reports errors through ok() / error().
#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fpr {
namespace mp {

class Writer {
 public:
  std::string buf;

  void nil() { buf.push_back((char)0xc0); }
  void boolean(bool v) { buf.push_back((char)(v ? 0xc3 : 0xc2)); }
  void i64(int64_t x) {
    if (x >= 0) {
      u64((uint64_t)x);
      return;
    }
    if (x >= -32) {
      buf.push_back((char)(int8_t)x);
    } else if (x >= -128) {
      buf.push_back((char)0xd0);
      buf.push_back((char)(int8_t)x);
    } else if (x >= -32768) {
      buf.push_back((char)0xd1);
      be16((uint16_t)(int16_t)x);
    } else if (x >= -2147483648LL) {
      buf.push_back((char)0xd2);
      be32((uint32_t)(int32_t)x);
    } else {
      buf.push_back((char)0xd3);
      be64((uint64_t)x);
    }
  }
  void u64(uint64_t x) {
    if (x < 128) {
      buf.push_back((char)x);
    } else if (x < 256) {
      buf.push_back((char)0xcc);
      buf.push_back((char)x);
    } else if (x < 65536) {
      buf.push_back((char)0xcd);
      be16((uint16_t)x);
    } else if (x < (1ULL << 32)) {
      buf.push_back((char)0xce);
      be32((uint32_t)x);
    } else {
      buf.push_back((char)0xcf);
      be64(x);
    }
  }
  void f64(double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, 8);
    buf.push_back((char)0xcb);
    be64(bits);
  }
  void str(const char* s, size_t n) {
    if (n < 32) {
      buf.push_back((char)(0xa0 | n));
    } else if (n < 256) {
      buf.push_back((char)0xd9);
      buf.push_back((char)n);
    } else if (n < 65536) {
      buf.push_back((char)0xda);
      be16((uint16_t)n);
    } else {
      buf.push_back((char)0xdb);
      be32((uint32_t)n);
    }
    buf.append(s, n);
  }
  void str(const std::string& s) { str(s.data(), s.size()); }
  void bin(const char* s, size_t n) {
    if (n < 256) {
      buf.push_back((char)0xc4);
      buf.push_back((char)n);
    } else if (n < 65536) {
      buf.push_back((char)0xc5);
      be16((uint16_t)n);
    } else {
      buf.push_back((char)0xc6);
      be32((uint32_t)n);
    }
    buf.append(s, n);
  }
  void ext(int8_t type, const char* s, size_t n) {
    switch (n) {
      case 1: buf.push_back((char)0xd4); break;
      case 2: buf.push_back((char)0xd5); break;
      case 4: buf.push_back((char)0xd6); break;
      case 8: buf.push_back((char)0xd7); break;
      case 16: buf.push_back((char)0xd8); break;
      default:
        if (n < 256) {
          buf.push_back((char)0xc7);
          buf.push_back((char)n);
        } else if (n < 65536) {
          buf.push_back((char)0xc8);
          be16((uint16_t)n);
        } else {
          buf.push_back((char)0xc9);
          be32((uint32_t)n);
        }
    }
    buf.push_back((char)type);
    buf.append(s, n);
  }
  void arr(uint32_t n) {
    if (n < 16) {
      buf.push_back((char)(0x90 | n));
    } else if (n < 65536) {
      buf.push_back((char)0xdc);
      be16((uint16_t)n);
    } else {
      buf.push_back((char)0xdd);
      be32(n);
    }
  }
  void map(uint32_t n) {
    if (n < 16) {
      buf.push_back((char)(0x80 | n));
    } else if (n < 65536) {
      buf.push_back((char)0xde);
      be16((uint16_t)n);
    } else {
      buf.push_back((char)0xdf);
      be32(n);
    }
  }
  void raw(const std::string& encoded) { buf += encoded; }   // an already-encoded value

 private:
  void be16(uint16_t x) {
    buf.push_back((char)(x >> 8));
    buf.push_back((char)x);
  }
  void be32(uint32_t x) {
    for (int s = 24; s >= 0; s -= 8) buf.push_back((char)(x >> s));
  }
  void be64(uint64_t x) {
    for (int s = 56; s >= 0; s -= 8) buf.push_back((char)(x >> s));
  }
};

// ------------------------------------------------------------------ generic value (decode side, utility results)
struct Val {
  enum Kind : uint8_t { NIL, BOOL, INT, UINT, F64, STR, BIN, ARR, MAP, EXT } kind = NIL;
  bool b = false;
  int64_t i = 0;          // INT (also UINT when it fits)
  uint64_t u = 0;         // UINT
  double f = 0.0;
  int8_t ext_type = 0;
  std::string s;          // STR / BIN / EXT payload
  std::vector<Val> a;     // ARR; MAP as [k0, v0, k1, v1, ...] (order kept)

  bool is_nil() const { return kind == NIL; }
  bool is_int() const { return kind == INT || (kind == UINT && u <= (uint64_t)INT64_MAX); }
  int64_t as_int() const { return kind == UINT ? (int64_t)u : i; }
  double as_f64() const { return kind == F64 ? f : (double)as_int(); }
  size_t map_size() const { return a.size() / 2; }
  const Val* get(const std::string& key) const {   // MAP lookup (linear: maps here are small)
    for (size_t j = 0; j + 1 < a.size(); j += 2)
      if (a[j].kind == STR && a[j].s == key) return &a[j + 1];
    return nullptr;
  }
  static Val nil_() { return Val(); }
  static Val int_(int64_t x) { Val v; v.kind = INT; v.i = x; return v; }
  static Val str_(std::string x) { Val v; v.kind = STR; v.s = std::move(x); return v; }
  static Val bool_(bool x) { Val v; v.kind = BOOL; v.b = x; return v; }
  static Val f64_(double x) { Val v; v.kind = F64; v.f = x; return v; }
  static Val arr_() { Val v; v.kind = ARR; return v; }
  static Val map_() { Val v; v.kind = MAP; return v; }
};

inline void write_val(Writer& w, const Val& v) {
  switch (v.kind) {
    case Val::NIL: w.nil(); break;
    case Val::BOOL: w.boolean(v.b); break;
    case Val::INT: w.i64(v.i); break;
    case Val::UINT: w.u64(v.u); break;
    case Val::F64: w.f64(v.f); break;
    case Val::STR: w.str(v.s); break;
    case Val::BIN: w.bin(v.s.data(), v.s.size()); break;
    case Val::EXT: w.ext(v.ext_type, v.s.data(), v.s.size()); break;
    case Val::ARR:
      w.arr((uint32_t)v.a.size());
      for (const auto& x : v.a) write_val(w, x);
      break;
    case Val::MAP:
      w.map((uint32_t)(v.a.size() / 2));
      for (const auto& x : v.a) write_val(w, x);
      break;
  }
}

class Reader {
 public:
  Reader(const char* p, size_t n) : p_((const uint8_t*)p), end_((const uint8_t*)p + n) {}
  explicit Reader(const std::string& s) : Reader(s.data(), s.size()) {}

  bool ok() const { return err_.empty(); }
  const std::string& error() const { return err_; }
  bool at_end() const { return p_ == end_; }

  bool read(Val* out, int depth = 0) {
    if (!ok()) return false;
    if (depth > 64) return fail("msgpack: nesting too deep");
    if (p_ >= end_) return fail("msgpack: truncated");
    uint8_t c = *p_++;
    *out = Val();
    if (c <= 0x7f) { out->kind = Val::INT; out->i = c; return true; }
    if (c >= 0xe0) { out->kind = Val::INT; out->i = (int8_t)c; return true; }
    if ((c & 0xe0) == 0xa0) return rd_str(out, c & 0x1f, Val::STR);
    if ((c & 0xf0) == 0x90) return rd_arr(out, c & 0x0f, depth);
    if ((c & 0xf0) == 0x80) return rd_map(out, c & 0x0f, depth);
    uint64_t n;
    switch (c) {
      case 0xc0: out->kind = Val::NIL; return true;
      case 0xc2: out->kind = Val::BOOL; out->b = false; return true;
      case 0xc3: out->kind = Val::BOOL; out->b = true; return true;
      case 0xc4: return be(1, &n) && rd_str(out, n, Val::BIN);
      case 0xc5: return be(2, &n) && rd_str(out, n, Val::BIN);
      case 0xc6: return be(4, &n) && rd_str(out, n, Val::BIN);
      case 0xc7: return be(1, &n) && rd_ext(out, n);
      case 0xc8: return be(2, &n) && rd_ext(out, n);
      case 0xc9: return be(4, &n) && rd_ext(out, n);
      case 0xca: {
        if (!be(4, &n)) return false;
        uint32_t b32 = (uint32_t)n; float f; std::memcpy(&f, &b32, 4);
        out->kind = Val::F64; out->f = f; return true;
      }
      case 0xcb: {
        if (!be(8, &n)) return false;
        double d; std::memcpy(&d, &n, 8);
        out->kind = Val::F64; out->f = d; return true;
      }
      case 0xcc: if (!be(1, &n)) return false; out->kind = Val::INT; out->i = (int64_t)n; return true;
      case 0xcd: if (!be(2, &n)) return false; out->kind = Val::INT; out->i = (int64_t)n; return true;
      case 0xce: if (!be(4, &n)) return false; out->kind = Val::INT; out->i = (int64_t)n; return true;
      case 0xcf:
        if (!be(8, &n)) return false;
        if (n <= (uint64_t)INT64_MAX) { out->kind = Val::INT; out->i = (int64_t)n; }
        else { out->kind = Val::UINT; out->u = n; }
        return true;
      case 0xd0: if (!be(1, &n)) return false; out->kind = Val::INT; out->i = (int8_t)n; return true;
      case 0xd1: if (!be(2, &n)) return false; out->kind = Val::INT; out->i = (int16_t)n; return true;
      case 0xd2: if (!be(4, &n)) return false; out->kind = Val::INT; out->i = (int32_t)n; return true;
      case 0xd3: if (!be(8, &n)) return false; out->kind = Val::INT; out->i = (int64_t)n; return true;
      case 0xd4: return rd_ext(out, 1);
      case 0xd5: return rd_ext(out, 2);
      case 0xd6: return rd_ext(out, 4);
      case 0xd7: return rd_ext(out, 8);
      case 0xd8: return rd_ext(out, 16);
      case 0xd9: return be(1, &n) && rd_str(out, n, Val::STR);
      case 0xda: return be(2, &n) && rd_str(out, n, Val::STR);
      case 0xdb: return be(4, &n) && rd_str(out, n, Val::STR);
      case 0xdc: return be(2, &n) && rd_arr(out, n, depth);
      case 0xdd: return be(4, &n) && rd_arr(out, n, depth);
      case 0xde: return be(2, &n) && rd_map(out, n, depth);
      case 0xdf: return be(4, &n) && rd_map(out, n, depth);
      default: return fail("msgpack: bad type byte");
    }
  }

 private:
  bool fail(const char* m) {
    if (err_.empty()) err_ = m;
    return false;
  }
  bool be(int k, uint64_t* n) {
    if (end_ - p_ < k) return fail("msgpack: truncated");
    uint64_t x = 0;
    for (int j = 0; j < k; ++j) x = (x << 8) | *p_++;
    *n = x;
    return true;
  }
  bool rd_str(Val* out, uint64_t n, Val::Kind k) {
    if ((uint64_t)(end_ - p_) < n) return fail("msgpack: truncated string");
    out->kind = k;
    out->s.assign((const char*)p_, n);
    p_ += n;
    return true;
  }
  bool rd_ext(Val* out, uint64_t n) {
    if (p_ >= end_) return fail("msgpack: truncated ext");
    out->ext_type = (int8_t)*p_++;
    if (!rd_str(out, n, Val::EXT)) return false;
    out->kind = Val::EXT;
    return true;
  }
  bool rd_arr(Val* out, uint64_t n, int depth) {
    if (n > (uint64_t)(end_ - p_)) return fail("msgpack: array length beyond the buffer");
    out->kind = Val::ARR;
    out->a.resize(n);
    for (uint64_t j = 0; j < n; ++j)
      if (!read(&out->a[j], depth + 1)) return false;
    return true;
  }
  bool rd_map(Val* out, uint64_t n, int depth) {
    if (2 * n > (uint64_t)(end_ - p_)) return fail("msgpack: map length beyond the buffer");
    out->kind = Val::MAP;
    out->a.resize(2 * n);
    for (uint64_t j = 0; j < 2 * n; ++j)
      if (!read(&out->a[j], depth + 1)) return false;
    return true;
  }
  const uint8_t* p_;
  const uint8_t* end_;
  std::string err_;
};

}  // namespace mp
}  // namespace fpr
