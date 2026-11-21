#include "engine/mp.h"

#include <charconv>
#include <cstdio>

namespace fe::mp {

int64_t Value::as_int() const {
  if (type == Type::Int) return i;
  if (type == Type::UInt && u <= uint64_t(INT64_MAX)) return int64_t(u);
  if (type == Type::Bool) return b ? 1 : 0;
  throw DecodeError("msgpack: expected int");
}
uint64_t Value::as_uint() const {
  if (type == Type::UInt) return u;
  if (type == Type::Int && i >= 0) return uint64_t(i);
  throw DecodeError("msgpack: expected unsigned int");
}
double Value::as_double() const {
  if (type == Type::Float) return f;
  if (type == Type::Int) return double(i);
  if (type == Type::UInt) return double(u);
  throw DecodeError("msgpack: expected number");
}
const std::string& Value::as_str() const {
  if (type != Type::Str && type != Type::Bin) throw DecodeError("msgpack: expected str");
  return s;
}
const Value* Value::get(std::string_view key) const {
  if (type != Type::Map) return nullptr;
  for (auto& [k, v] : kv)
    if (k.type == Type::Str && k.s == key) return &v;
  return nullptr;
}
Value& Value::set(std::string key, Value v) {
  type = Type::Map;
  for (auto& [k, old] : kv)
    if (k.type == Type::Str && k.s == key) { old = std::move(v); return old; }
  kv.emplace_back(Value::from_str(std::move(key)), std::move(v));
  return kv.back().second;
}

namespace {
struct Reader {
  const unsigned char* d;
  size_t n;
  size_t p;
  void need(size_t k) {
    if (n - p < k) throw DecodeError("msgpack: truncated input");
  }
  uint8_t u8() { need(1); return d[p++]; }
  uint64_t be(int k) {
    need(size_t(k));
    uint64_t v = 0;
    for (int j = 0; j < k; ++j) v = (v << 8) | d[p + j];
    p += size_t(k);
    return v;
  }
  std::string bytes(size_t k) {
    need(k);
    std::string s(reinterpret_cast<const char*>(d + p), k);
    p += k;
    return s;
  }
};

Value read(Reader& r, int depth) {
  if (depth > 512) throw DecodeError("msgpack: nesting too deep");
  Value v;
  uint8_t c = r.u8();
  auto set_int = [&](int64_t x) { v.type = Type::Int; v.i = x; };
  auto set_uint = [&](uint64_t x) {
    if (x <= uint64_t(INT64_MAX)) set_int(int64_t(x));
    else { v.type = Type::UInt; v.u = x; }
  };
  auto read_arr = [&](size_t k) {
    v.type = Type::Array;
    v.arr.reserve(k < 1u << 20 ? k : 1u << 20);
    for (size_t j = 0; j < k; ++j) v.arr.push_back(read(r, depth + 1));
  };
  auto read_map = [&](size_t k) {
    v.type = Type::Map;
    v.kv.reserve(k < 1u << 16 ? k : 1u << 16);
    for (size_t j = 0; j < k; ++j) {
      Value key = read(r, depth + 1);
      Value val = read(r, depth + 1);
      v.kv.emplace_back(std::move(key), std::move(val));
    }
  };
  auto read_ext = [&](size_t k) {
    v.type = Type::Ext;
    v.ext_type = int8_t(r.u8());
    v.s = r.bytes(k);
  };
  if (c <= 0x7f) { set_int(c); return v; }
  if (c >= 0xe0) { set_int(int8_t(c)); return v; }
  if ((c & 0xf0) == 0x80) { read_map(c & 0x0f); return v; }
  if ((c & 0xf0) == 0x90) { read_arr(c & 0x0f); return v; }
  if ((c & 0xe0) == 0xa0) { v.type = Type::Str; v.s = r.bytes(c & 0x1f); return v; }
  switch (c) {
    case 0xc0: return v;
    case 0xc2: v.type = Type::Bool; v.b = false; return v;
    case 0xc3: v.type = Type::Bool; v.b = true; return v;
    case 0xc4: v.type = Type::Bin; v.s = r.bytes(r.be(1)); return v;
    case 0xc5: v.type = Type::Bin; v.s = r.bytes(r.be(2)); return v;
    case 0xc6: v.type = Type::Bin; v.s = r.bytes(r.be(4)); return v;
    case 0xc7: read_ext(r.be(1)); return v;
    case 0xc8: read_ext(r.be(2)); return v;
    case 0xc9: read_ext(r.be(4)); return v;
    case 0xca: {
      uint32_t u = uint32_t(r.be(4));
      float fl;
      std::memcpy(&fl, &u, 4);
      v.type = Type::Float; v.f = fl; return v;
    }
    case 0xcb: {
      uint64_t u = r.be(8);
      std::memcpy(&v.f, &u, 8);
      v.type = Type::Float; return v;
    }
    case 0xcc: set_uint(r.be(1)); return v;
    case 0xcd: set_uint(r.be(2)); return v;
    case 0xce: set_uint(r.be(4)); return v;
    case 0xcf: set_uint(r.be(8)); return v;
    case 0xd0: set_int(int8_t(r.be(1))); return v;
    case 0xd1: set_int(int16_t(r.be(2))); return v;
    case 0xd2: set_int(int32_t(r.be(4))); return v;
    case 0xd3: set_int(int64_t(r.be(8))); return v;
    case 0xd4: read_ext(1); return v;
    case 0xd5: read_ext(2); return v;
    case 0xd6: read_ext(4); return v;
    case 0xd7: read_ext(8); return v;
    case 0xd8: read_ext(16); return v;
    case 0xd9: v.type = Type::Str; v.s = r.bytes(r.be(1)); return v;
    case 0xda: v.type = Type::Str; v.s = r.bytes(r.be(2)); return v;
    case 0xdb: v.type = Type::Str; v.s = r.bytes(r.be(4)); return v;
    case 0xdc: read_arr(r.be(2)); return v;
    case 0xdd: read_arr(r.be(4)); return v;
    case 0xde: read_map(r.be(2)); return v;
    case 0xdf: read_map(r.be(4)); return v;
    default: break;
  }
  throw DecodeError("msgpack: invalid type byte");
}
}  // namespace

Value unpack(const char* data, size_t len, size_t* pos) {
  Reader r{reinterpret_cast<const unsigned char*>(data), len, *pos};
  Value v = read(r, 0);
  *pos = r.p;
  return v;
}

void pack(Packer& p, const Value& v) {
  switch (v.type) {
    case Type::Nil: p.nil(); break;
    case Type::Bool: p.boolean(v.b); break;
    case Type::Int: p.integer(v.i); break;
    case Type::UInt: p.uinteger(v.u); break;
    case Type::Float: p.float64(v.f); break;
    case Type::Str: p.str(v.s); break;
    case Type::Bin: p.bin(v.s); break;
    case Type::Array:
      p.array(v.arr.size());
      for (auto& e : v.arr) pack(p, e);
      break;
    case Type::Map:
      p.map(v.kv.size());
      for (auto& [k, e] : v.kv) { pack(p, k); pack(p, e); }
      break;
    case Type::Ext: p.ext(v.ext_type, v.s); break;
  }
}

static void dbg(std::string& o, const Value& v) {
  char tmp[64];
  switch (v.type) {
    case Type::Nil: o += "nil"; break;
    case Type::Bool: o += v.b ? "true" : "false"; break;
    case Type::Int: o += std::to_string(v.i); break;
    case Type::UInt: o += std::to_string(v.u); break;
    case Type::Float: {
      auto res = std::to_chars(tmp, tmp + sizeof tmp, v.f);
      o.append(tmp, res.ptr);
      break;
    }
    case Type::Str: o += '"'; o += v.s; o += '"'; break;
    case Type::Bin: o += "bin(" + std::to_string(v.s.size()) + ")"; break;
    case Type::Ext: o += "ext(" + std::to_string(v.ext_type) + "," + std::to_string(v.s.size()) + ")"; break;
    case Type::Array:
      o += '[';
      for (size_t j = 0; j < v.arr.size(); ++j) {
        if (j) o += ',';
        if (j >= 32) { o += "..." + std::to_string(v.arr.size()); break; }
        dbg(o, v.arr[j]);
      }
      o += ']';
      break;
    case Type::Map:
      o += '{';
      for (size_t j = 0; j < v.kv.size(); ++j) {
        if (j) o += ',';
        dbg(o, v.kv[j].first);
        o += ':';
        dbg(o, v.kv[j].second);
      }
      o += '}';
      break;
  }
}
std::string to_debug_string(const Value& v) {
  std::string o;
  dbg(o, v);
  return o;
}

}  // namespace fe::mp
