// Python-compatible JSON writers: json.dumps, Starlette JSONResponse, pydantic model_dump_json.
#include <charconv>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

#include "common/json.h"
#include "simdjson/simdjson.h"

namespace fe {

namespace {

constexpr char kHex[] = "0123456789abcdef";

inline void put_u4(std::string& out, uint32_t c) {
  char b[6] = {'\\', 'u', kHex[(c >> 12) & 0xF], kHex[(c >> 8) & 0xF], kHex[(c >> 4) & 0xF], kHex[c & 0xF]};
  out.append(b, 6);
}

// Decode one generalized-UTF-8 sequence at s[0..n). Returns the code point and sets len; on an invalid sequence
// returns -1 and sets len to the maximal-subpart length (>= 1), like Python's 'replace' handler.
inline int32_t decode_one(const unsigned char* s, size_t n, size_t& len) {
  unsigned char c = s[0];
  auto cont = [](unsigned char x) { return (x & 0xC0) == 0x80; };
  if (c < 0x80) { len = 1; return c; }
  if (c < 0xC2) { len = 1; return -1; }
  if (c < 0xE0) {
    if (n >= 2 && cont(s[1])) { len = 2; return ((c & 0x1F) << 6) | (s[1] & 0x3F); }
    len = 1;
    return -1;
  }
  if (c < 0xF0) {
    // E0: A0..BF; ED: 80..BF here (surrogates allowed: generalized UTF-8); others 80..BF
    if (n >= 2 && cont(s[1]) && !(c == 0xE0 && s[1] < 0xA0)) {
      if (n >= 3 && cont(s[2])) {
        len = 3;
        return ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
      }
      len = 2;
      return -1;
    }
    len = 1;
    return -1;
  }
  if (c < 0xF5) {
    if (n >= 2 && cont(s[1]) && !(c == 0xF0 && s[1] < 0x90) && !(c == 0xF4 && s[1] >= 0x90)) {
      if (n >= 3 && cont(s[2])) {
        if (n >= 4 && cont(s[3])) {
          len = 4;
          return ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        }
        len = 3;
        return -1;
      }
      len = 2;
      return -1;
    }
    len = 1;
    return -1;
  }
  len = 1;
  return -1;
}

inline bool word_needs_attention(uint64_t w, bool ascii_out) {
  // any byte < 0x20, == '"', == '\\', >= 0x80, or (ascii_out) == 0x7F
  constexpr uint64_t ones = 0x0101010101010101ULL, highs = 0x8080808080808080ULL;
  uint64_t lt20 = (w - ones * 0x20) & ~w;
  uint64_t q = w ^ (ones * 0x22);
  uint64_t bs = w ^ (ones * 0x5C);
  uint64_t zq = (q - ones) & ~q;
  uint64_t zb = (bs - ones) & ~bs;
  uint64_t m = (lt20 | zq | zb | w) & highs;
  if (ascii_out) {
    uint64_t dl = w ^ (ones * 0x7F);
    m |= ((dl - ones) & ~dl) & highs;
  }
  return m != 0;
}

[[noreturn]] void surrogate_error(std::string_view s, size_t byte_pos, uint32_t cp) {
  size_t chars = 0;
  for (size_t k = 0; k < byte_pos; ++k) chars += (static_cast<unsigned char>(s[k]) & 0xC0) != 0x80;
  char buf[200];
  snprintf(buf, sizeof buf,
           "Error serializing to JSON: UnicodeEncodeError: 'utf-8' codec can't encode character '\\u%04x' in position "
           "%zu: surrogates not allowed",
           cp, chars);
  throw JsonError(JsonError::Kind::Encode, buf);
}

inline bool word_needs_escape_utf8(uint64_t w) {
  // byte < 0x20, '"' or '\\' (bytes >= 0x80 are known-valid UTF-8 here)
  constexpr uint64_t ones = 0x0101010101010101ULL, highs = 0x8080808080808080ULL;
  uint64_t lt20 = (w - ones * 0x20) & ~w;
  uint64_t q = w ^ (ones * 0x22);
  uint64_t bs = w ^ (ones * 0x5C);
  return (((lt20 | ((q - ones) & ~q) | ((bs - ones) & ~bs)) & highs)) != 0;
}

// ensure_ascii=False on strict-valid UTF-8: only '"', '\\' and C0 controls are escaped.
void write_string_valid_utf8(std::string& out, std::string_view sv) {
  const auto* s = reinterpret_cast<const unsigned char*>(sv.data());
  const size_t n = sv.size();
  out.reserve(out.size() + n + 2);
  out.push_back('"');
  size_t i = 0, run = 0;
  while (i < n) {
    while (i + 8 <= n) {
      uint64_t w;
      std::memcpy(&w, s + i, 8);
      if (word_needs_escape_utf8(w)) break;
      i += 8;
    }
    const size_t lim = std::min(n, i + 8);
    for (; i < lim; ++i) {
      unsigned char c = s[i];
      if (c >= 0x20 && c != '"' && c != '\\') continue;
      out.append(sv.data() + run, i - run);
      out.push_back('\\');
      switch (c) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '\b': out.push_back('b'); break;
        case '\f': out.push_back('f'); break;
        case '\n': out.push_back('n'); break;
        case '\r': out.push_back('r'); break;
        case '\t': out.push_back('t'); break;
        default:
          out.pop_back();
          put_u4(out, c);
      }
      run = i + 1;
    }
  }
  out.append(sv.data() + run, n - run);
  out.push_back('"');
}

void write_string(std::string& out, std::string_view sv, bool ensure_ascii, bool escape_del, bool surrogates_error) {
  if (!ensure_ascii && sv.size() >= 16 && simdjson::validate_utf8(sv.data(), sv.size())) {
    write_string_valid_utf8(out, sv);  // strict UTF-8 has no surrogates: nothing else can differ
    return;
  }
  const auto* s = reinterpret_cast<const unsigned char*>(sv.data());
  const size_t n = sv.size();
  out.reserve(out.size() + n + 2);
  out.push_back('"');
  size_t i = 0, run = 0;
  const bool del_esc = ensure_ascii && escape_del;
  while (i < n) {
    // fast skip of plain ASCII
    while (i + 8 <= n) {
      uint64_t w;
      std::memcpy(&w, s + i, 8);
      if (word_needs_attention(w, del_esc)) break;
      i += 8;
    }
    if (i >= n) break;
    unsigned char c = s[i];
    if (c >= 0x20 && c != '"' && c != '\\' && c < 0x7F) { ++i; continue; }
    if (c == 0x7F) {
      if (!del_esc) { ++i; continue; }
      out.append(sv.data() + run, i - run);
      put_u4(out, 0x7F);
      run = ++i;
      continue;
    }
    if (c < 0x80) {
      out.append(sv.data() + run, i - run);
      out.push_back('\\');
      switch (c) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '\b': out.push_back('b'); break;
        case '\f': out.push_back('f'); break;
        case '\n': out.push_back('n'); break;
        case '\r': out.push_back('r'); break;
        case '\t': out.push_back('t'); break;
        default:
          out.pop_back();
          put_u4(out, c);
      }
      run = ++i;
      continue;
    }
    size_t len;
    int32_t cp = decode_one(s + i, n - i, len);
    const bool sur = cp >= 0xD800 && cp <= 0xDFFF;
    if (sur && surrogates_error) surrogate_error(sv, i, static_cast<uint32_t>(cp));
    if (cp >= 0 && !ensure_ascii) { i += len; continue; }  // raw (incl. lone surrogate bytes, WTF-8)
    out.append(sv.data() + run, i - run);
    if (cp < 0) {
      if (ensure_ascii) put_u4(out, 0xFFFD);
      else out.append("\xEF\xBF\xBD", 3);
    } else if (cp >= 0x10000) {
      uint32_t v = static_cast<uint32_t>(cp) - 0x10000;
      put_u4(out, 0xD800 | (v >> 10));
      put_u4(out, 0xDC00 | (v & 0x3FF));
    } else {
      put_u4(out, static_cast<uint32_t>(cp));
    }
    i += len;
    run = i;
  }
  out.append(sv.data() + run, n - run);
  out.push_back('"');
}

// Shortest round-trip digits of a finite, non-negative-or-negative double: sign, digit string, decimal exponent
// such that value = 0.DIGITS * 10^decpt.
struct Digits {
  bool neg;
  char d[32];
  int nd;
  int decpt;
};

inline Digits shortest(double v) {
  Digits r{};
  char buf[64];
  auto res = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
  const char* p = buf;
  const char* e = res.ptr;
  r.neg = false;
  if (*p == '-') { r.neg = true; ++p; }
  int nd = 0;
  while (p < e && *p != 'e') {
    if (*p != '.') r.d[nd++] = *p;
    ++p;
  }
  int ex = 0;
  if (p < e && *p == 'e') {
    ++p;
    bool eneg = false;
    if (*p == '+' || *p == '-') { eneg = *p == '-'; ++p; }
    while (p < e) ex = ex * 10 + (*p++ - '0');
    if (eneg) ex = -ex;
  }
  r.nd = nd;
  r.decpt = ex + 1;
  return r;
}

void format_digits(std::string& out, const Digits& g, bool pydantic) {
  if (g.neg) out.push_back('-');
  const int nd = g.nd, decpt = g.decpt;
  const bool use_exp = pydantic ? (decpt < -4 || decpt > 16) : (decpt <= -4 || decpt > 16);
  if (use_exp) {
    out.push_back(g.d[0]);
    if (nd > 1) {
      out.push_back('.');
      out.append(g.d + 1, static_cast<size_t>(nd - 1));
    }
    int ex = decpt - 1;
    out.push_back('e');
    out.push_back(ex < 0 ? '-' : '+');
    unsigned ax = static_cast<unsigned>(ex < 0 ? -ex : ex);
    char eb[8];
    int k = 0;
    do { eb[k++] = static_cast<char>('0' + ax % 10); ax /= 10; } while (ax);
    if (!pydantic && k < 2) eb[k++] = '0';
    while (k) out.push_back(eb[--k]);
    return;
  }
  if (decpt <= 0) {
    out.append("0.", 2);
    out.append(static_cast<size_t>(-decpt), '0');
    out.append(g.d, static_cast<size_t>(nd));
  } else if (decpt >= nd) {
    out.append(g.d, static_cast<size_t>(nd));
    out.append(static_cast<size_t>(decpt - nd), '0');
    out.append(".0", 2);
  } else {
    out.append(g.d, static_cast<size_t>(decpt));
    out.push_back('.');
    out.append(g.d + decpt, static_cast<size_t>(nd - decpt));
  }
}

void write_float(std::string& out, double d, const JsonDumpOptions& o) {
  if (!std::isfinite(d)) {
    switch (o.non_finite) {
      case JsonNonFinite::Literal:
        if (std::isnan(d)) out.append("NaN");
        else out.append(d > 0 ? "Infinity" : "-Infinity");
        return;
      case JsonNonFinite::Null: out.append("null"); return;
      case JsonNonFinite::Error:
        throw JsonError(JsonError::Kind::Value, "Out of range float values are not JSON compliant");
    }
  }
  format_digits(out, shortest(d), o.floats == JsonFloatStyle::Pydantic);
}

inline void write_int(std::string& out, int64_t v) {
  char b[24];
  auto r = std::to_chars(b, b + sizeof b, v);
  out.append(b, static_cast<size_t>(r.ptr - b));
}

struct Writer {
  std::string& out;
  const JsonDumpOptions& o;
  std::string indent_unit;

  void newline(size_t level) {
    out.push_back('\n');
    for (size_t k = 0; k < level; ++k) out.append(indent_unit);
  }

  void scalar(const Json& v) {
    switch (v.type()) {
      case Json::Type::Null: out.append("null", 4); return;
      case Json::Type::Bool: v.as_bool() ? out.append("true", 4) : out.append("false", 5); return;
      case Json::Type::Int: write_int(out, v.as_int()); return;
      case Json::Type::BigInt: out.append(v.big_int_text()); return;
      case Json::Type::Float: write_float(out, v.as_double(), o); return;
      case Json::Type::String: write_string(out, v.as_string(), o.ensure_ascii, o.escape_del, o.surrogates_error); return;
      default: return;
    }
  }

  // Iterative (explicit stack): a 10k-deep document needs no deep recursion.
  void root(const Json& top) {
    std::vector<std::pair<const Json*, size_t>> st;  // open container, next child index
    const bool ind = o.indent >= 0;
    const Json* v = &top;
    for (;;) {
      const size_t level = st.size();  // nesting level of v (root = 0)
      if (o.max_depth > 0 && level >= static_cast<size_t>(o.max_depth))
        throw JsonError(JsonError::Kind::Encode,
                        "Error serializing to JSON: ValueError: Circular reference detected (depth exceeded)");
      if (v->is_array() || v->is_object()) {
        if (v->size() == 0) out.append(v->is_array() ? "[]" : "{}", 2);
        else {
          out.push_back(v->is_array() ? '[' : '{');
          st.emplace_back(v, 0);
        }
      } else {
        scalar(*v);
      }
      for (;;) {  // next value to write
        if (st.empty()) return;
        auto& top_frame = st.back();
        const Json* c = top_frame.first;
        const size_t clevel = st.size() - 1;
        size_t& i = top_frame.second;
        if (i == c->size()) {
          if (ind) newline(clevel);
          out.push_back(c->is_array() ? ']' : '}');
          st.pop_back();
          continue;
        }
        if (i > 0) out.append(o.item_sep);
        if (ind) newline(clevel + 1);
        if (c->is_array()) {
          v = &c->as_array()[i];
        } else {
          const auto& m = c->as_object().at_index(i);
          write_string(out, m.first, o.ensure_ascii, o.escape_del, o.surrogates_error);
          out.append(o.key_sep);
          v = &m.second;
        }
        ++i;
        break;
      }
    }
  }
};

}  // namespace

void dump_append(std::string& out, const Json& v, const JsonDumpOptions& opt) {
  Writer w{out, opt, std::string(opt.indent > 0 ? static_cast<size_t>(opt.indent) : 0, ' ')};
  w.root(v);
}

std::string dump(const Json& v, const JsonDumpOptions& opt) {
  std::string out;
  dump_append(out, v, opt);
  return out;
}

std::string dumps_py(const Json& v, bool ensure_ascii) {
  JsonDumpOptions o;
  o.ensure_ascii = ensure_ascii;
  return dump(v, o);
}

std::string dumps_compact(const Json& v, bool ensure_ascii) {
  JsonDumpOptions o;
  o.item_sep = ",";
  o.key_sep = ":";
  o.ensure_ascii = ensure_ascii;
  return dump(v, o);
}

std::string dumps_starlette(const Json& v) {
  // json.dumps(..., allow_nan=False) runs first (ValueError on NaN/inf anywhere), then .encode("utf-8")
  // (UnicodeEncodeError on a lone surrogate): check the surrogates after the full dump.
  JsonDumpOptions o;
  o.item_sep = ",";
  o.key_sep = ":";
  o.ensure_ascii = false;
  o.non_finite = JsonNonFinite::Error;
  std::string out = dump(v, o);
  const char* p = out.data();
  const char* e = p + out.size();
  while ((p = static_cast<const char*>(std::memchr(p, '\xED', static_cast<size_t>(e - p)))) != nullptr) {
    if (e - p >= 3 && static_cast<unsigned char>(p[1]) >= 0xA0 && static_cast<unsigned char>(p[1]) <= 0xBF) {
      size_t pos = 0;
      for (const char* q = out.data(); q < p; ++q) pos += (static_cast<unsigned char>(*q) & 0xC0) != 0x80;
      uint32_t cp = 0xD000u | ((static_cast<unsigned char>(p[1]) & 0x3Fu) << 6) | (static_cast<unsigned char>(p[2]) & 0x3Fu);
      char buf[160];
      snprintf(buf, sizeof buf, "'utf-8' codec can't encode character '\\u%04x' in position %zu: surrogates not allowed",
               cp, pos);
      throw JsonError(JsonError::Kind::Encode, buf);
    }
    ++p;
  }
  return out;
}

std::string dumps_pydantic(const Json& v, bool ensure_ascii) {
  JsonDumpOptions o;
  o.item_sep = ",";
  o.key_sep = ":";
  o.ensure_ascii = ensure_ascii;
  o.escape_del = false;
  o.surrogates_error = true;
  o.floats = JsonFloatStyle::Pydantic;
  o.non_finite = JsonNonFinite::Null;
  o.max_depth = 255;
  return dump(v, o);
}

void append_py_string(std::string& out, std::string_view s, bool ensure_ascii) {
  write_string(out, s, ensure_ascii, true, false);
}

void append_py_float(std::string& out, double d) {
  JsonDumpOptions o;
  write_float(out, d, o);
}

std::string py_float_repr(double d) {
  if (std::isnan(d)) return "nan";
  if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
  std::string s;
  format_digits(s, shortest(d), false);
  return s;
}

void append_pydantic_float(std::string& out, double d) { format_digits(out, shortest(d), true); }

std::string format_float_pydantic(double d) {
  if (!std::isfinite(d)) return "null";
  std::string s;
  format_digits(s, shortest(d), true);
  return s;
}

void append_pydantic_string(std::string& out, std::string_view s, bool ensure_ascii) {
  write_string(out, s, ensure_ascii, false, true);
}

}  // namespace fe
