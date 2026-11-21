#include "parser/pyjson.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace fe::pyjson {

const Value* Value::get(std::string_view k) const {
  if (t != T::Obj) return nullptr;
  for (size_t i = 0; i < keys.size(); ++i)
    if (keys[i] == k) return &items[i];
  return nullptr;
}
Value* Value::get_mut(std::string_view k) {
  if (t != T::Obj) return nullptr;
  for (size_t i = 0; i < keys.size(); ++i)
    if (keys[i] == k) return &items[i];
  return nullptr;
}
void Value::set(std::string k, Value v) {
  if (Value* p = get_mut(k)) { *p = std::move(v); return; }
  keys.push_back(std::move(k));
  items.push_back(std::move(v));
}
void Value::erase(std::string_view k) {
  for (size_t i = 0; i < keys.size(); ++i)
    if (keys[i] == k) { keys.erase(keys.begin() + i); items.erase(items.begin() + i); return; }
}
bool Value::truthy() const {
  switch (t) {
    case T::Null: return false;
    case T::Bool: return b;
    case T::Int: return s != "0";
    case T::Float: return f != 0.0;  // nan is truthy
    case T::Str: return !s.empty();
    case T::Arr: case T::Obj: return !items.empty();
  }
  return false;
}

// ------------------------------------------------------------------------------------------------------------ utf-8
uint32_t decode_cp(std::string_view s, size_t i, size_t& len) {
  unsigned char c = (unsigned char)s[i];
  if (c < 0x80) { len = 1; return c; }
  size_t n = c >= 0xF0 && c < 0xF8 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
  if (n == 1 || i + n > s.size()) { len = 1; return c; }
  uint32_t cp = n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
  for (size_t k = 1; k < n; ++k) {
    unsigned char d = (unsigned char)s[i + k];
    if ((d & 0xC0) != 0x80) { len = 1; return c; }
    cp = (cp << 6) | (d & 0x3F);
  }
  len = n;
  return cp;
}
size_t cp_len_at(std::string_view s, size_t i) { size_t l; decode_cp(s, i, l); return l; }

bool re_space(uint32_t c) {
  return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
bool py_isspace(uint32_t c) { return re_space(c) || (c >= 0x1C && c <= 0x1F); }

std::string_view py_rstrip(std::string_view s) {
  size_t keep = 0;  // end of the last non-space code point
  for (size_t i = 0; i < s.size();) {
    size_t l;
    uint32_t cp = decode_cp(s, i, l);
    i += l;
    if (!py_isspace(cp)) keep = i;
  }
  return s.substr(0, keep);
}

std::string_view py_strip(std::string_view s) {
  size_t i = 0;
  while (i < s.size()) {
    size_t l;
    uint32_t cp = decode_cp(s, i, l);
    if (!py_isspace(cp)) break;
    i += l;
  }
  return py_rstrip(s.substr(i));
}

// ------------------------------------------------------------------------------------------------------------ loads
namespace {
struct Parser {
  std::string_view s;
  size_t i = 0;
  int depth = 0;
  void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i; }
  static int hexv(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  }
  static void put_cp(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
    else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
  }
  bool u4(size_t at, uint32_t& v) {
    if (at + 4 > s.size()) return false;
    v = 0;
    for (size_t k = 0; k < 4; ++k) { int h = hexv(s[at + k]); if (h < 0) return false; v = v * 16 + h; }
    return true;
  }
  bool str(std::string& o) {  // at the opening quote
    ++i;
    for (;;) {
      if (i >= s.size()) return false;
      unsigned char c = (unsigned char)s[i];
      if (c == '"') { ++i; return true; }
      if (c < 0x20) return false;  // strict: Invalid control character
      if (c != '\\') { o += (char)c; ++i; continue; }
      if (i + 1 >= s.size()) return false;
      char e = s[i + 1];
      i += 2;
      switch (e) {
        case '"': o += '"'; break;
        case '\\': o += '\\'; break;
        case '/': o += '/'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'n': o += '\n'; break;
        case 'r': o += '\r'; break;
        case 't': o += '\t'; break;
        case 'u': {
          uint32_t v;
          if (!u4(i, v)) return false;
          i += 4;
          if (v >= 0xD800 && v <= 0xDBFF && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
            uint32_t w;
            if (u4(i + 2, w) && w >= 0xDC00 && w <= 0xDFFF) {
              v = 0x10000 + (((v - 0xD800) << 10) | (w - 0xDC00));
              i += 6;
            }
          }
          put_cp(o, v);
          break;
        }
        default: return false;
      }
    }
  }
  bool lit(std::string_view w) {
    if (s.substr(i, w.size()) == w) { i += w.size(); return true; }
    return false;
  }
  bool number(Value& v) {
    size_t st = i;
    if (i < s.size() && s[i] == '-') ++i;
    if (i >= s.size()) return false;
    if (s[i] == '0') ++i;
    else if (s[i] >= '1' && s[i] <= '9') { while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i; }
    else return false;
    size_t int_end = i;
    bool isf = false;
    if (i + 1 < s.size() && s[i] == '.' && s[i + 1] >= '0' && s[i + 1] <= '9') {
      isf = true; i += 1;
      while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
      size_t j = i + 1;
      if (j < s.size() && (s[j] == '+' || s[j] == '-')) ++j;
      if (j < s.size() && s[j] >= '0' && s[j] <= '9') {
        while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j;
        isf = true; i = j;
      }
    }
    if (isf) {
      std::string t(s.substr(st, i - st));
      v.t = Value::T::Float;
      v.f = std::strtod(t.c_str(), nullptr);  // correctly rounded (glibc); overflow -> inf like float()
      return true;
    }
    v.t = Value::T::Int;
    std::string_view d = s.substr(st, int_end - st);
    v.s = d == "-0" ? std::string("0") : std::string(d);
    return true;
  }
  bool value(Value& v) {
    if (i >= s.size()) return false;
    if (++depth > 10000) return false;
    bool ok = value_inner(v);
    --depth;
    return ok;
  }
  bool value_inner(Value& v) {
    char c = s[i];
    if (c == '"') { v.t = Value::T::Str; return str(v.s); }
    if (c == '{') {
      v.t = Value::T::Obj;
      ++i; ws();
      if (i < s.size() && s[i] == '}') { ++i; return true; }
      for (;;) {
        if (i >= s.size() || s[i] != '"') return false;
        std::string k;
        if (!str(k)) return false;
        ws();
        if (i >= s.size() || s[i] != ':') return false;
        ++i; ws();
        Value x;
        if (!value(x)) return false;
        v.set(std::move(k), std::move(x));
        ws();
        if (i < s.size() && s[i] == '}') { ++i; return true; }
        if (i >= s.size() || s[i] != ',') return false;
        ++i; ws();
      }
    }
    if (c == '[') {
      v.t = Value::T::Arr;
      ++i; ws();
      if (i < s.size() && s[i] == ']') { ++i; return true; }
      for (;;) {
        Value x;
        if (!value(x)) return false;
        v.items.push_back(std::move(x));
        ws();
        if (i < s.size() && s[i] == ']') { ++i; return true; }
        if (i >= s.size() || s[i] != ',') return false;
        ++i; ws();
      }
    }
    if (c == 'n') { if (lit("null")) { v.t = Value::T::Null; return true; } return false; }
    if (c == 't') { if (lit("true")) { v = Value::boolean(true); return true; } return false; }
    if (c == 'f') { if (lit("false")) { v = Value::boolean(false); return true; } return false; }
    if ((c >= '0' && c <= '9') || c == '-') {
      size_t save = i;
      if (number(v)) return true;
      i = save;
    }
    if (c == 'N' && lit("NaN")) { v.t = Value::T::Float; v.f = std::nan(""); return true; }
    if (c == 'I' && lit("Infinity")) { v.t = Value::T::Float; v.f = INFINITY; return true; }
    if (c == '-' && lit("-Infinity")) { v.t = Value::T::Float; v.f = -INFINITY; return true; }
    return false;
  }
};
}  // namespace

bool loads(std::string_view s, Value& out) {
  Parser p{s};
  p.ws();
  out = Value{};
  if (!p.value(out)) return false;
  p.ws();
  return p.i == s.size();
}

// ------------------------------------------------------------------------------------------------------------ dumps
std::string float_repr(double d) {
  if (std::isnan(d)) return "NaN";
  if (std::isinf(d)) return d > 0 ? "Infinity" : "-Infinity";
  if (d == 0) return std::signbit(d) ? "-0.0" : "0.0";
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
  std::string sci(buf, r.ptr);
  std::string out;
  size_t p = 0;
  if (sci[0] == '-') { out += '-'; p = 1; }
  size_t e = sci.find('e');
  std::string digits;
  for (size_t k = p; k < e; ++k) if (sci[k] != '.') digits += sci[k];
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
  int exp10 = std::atoi(sci.c_str() + e + 1);
  int decpt = exp10 + 1;
  int nd = (int)digits.size();
  if (decpt <= -4 || decpt > 16) {
    out += digits[0];
    if (nd > 1) { out += '.'; out.append(digits, 1, std::string::npos); }
    int x = decpt - 1;
    out += 'e';
    out += x < 0 ? '-' : '+';
    int ax = x < 0 ? -x : x;
    if (ax < 10) out += '0';
    out += std::to_string(ax);
  } else if (decpt <= 0) {
    out += "0.";
    out.append((size_t)(-decpt), '0');
    out += digits;
  } else if (decpt >= nd) {
    out += digits;
    out.append((size_t)(decpt - nd), '0');
    out += ".0";
  } else {
    out.append(digits, 0, (size_t)decpt);
    out += '.';
    out.append(digits, (size_t)decpt, std::string::npos);
  }
  return out;
}

static const char* HEX = "0123456789abcdef";
static void u_esc(std::string& o, uint32_t v) {
  o += "\\u";
  o += HEX[(v >> 12) & 15]; o += HEX[(v >> 8) & 15]; o += HEX[(v >> 4) & 15]; o += HEX[v & 15];
}

void dump_str_to(std::string_view s, bool ensure_ascii, std::string& o) {
  o += '"';
  for (size_t i = 0; i < s.size();) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
      switch (c) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        case '\b': o += "\\b"; break;
        case '\f': o += "\\f"; break;
        default:
          if (c < 0x20 || (ensure_ascii && c == 0x7F)) u_esc(o, c);
          else o += (char)c;
      }
      ++i;
      continue;
    }
    size_t l;
    uint32_t cp = decode_cp(s, i, l);
    if (!ensure_ascii) { o.append(s.substr(i, l)); i += l; continue; }
    if (l == 1) cp = 0xFFFD;  // invalid byte (cannot come from a Python str)
    if (cp >= 0x10000) {
      uint32_t v = cp - 0x10000;
      u_esc(o, 0xD800 + (v >> 10));
      u_esc(o, 0xDC00 + (v & 0x3FF));
    } else {
      u_esc(o, cp);
    }
    i += l;
  }
  o += '"';
}

void dumps_to(const Value& v, bool ensure_ascii, std::string& o) {
  switch (v.t) {
    case Value::T::Null: o += "null"; return;
    case Value::T::Bool: o += v.b ? "true" : "false"; return;
    case Value::T::Int: o += v.s; return;
    case Value::T::Float: o += float_repr(v.f); return;
    case Value::T::Str: dump_str_to(v.s, ensure_ascii, o); return;
    case Value::T::Arr:
      if (v.items.empty()) { o += "[]"; return; }
      o += '[';
      for (size_t i = 0; i < v.items.size(); ++i) {
        if (i) o += ", ";
        dumps_to(v.items[i], ensure_ascii, o);
      }
      o += ']';
      return;
    case Value::T::Obj:
      if (v.items.empty()) { o += "{}"; return; }
      o += '{';
      for (size_t i = 0; i < v.items.size(); ++i) {
        if (i) o += ", ";
        dump_str_to(v.keys[i], ensure_ascii, o);
        o += ": ";
        dumps_to(v.items[i], ensure_ascii, o);
      }
      o += '}';
      return;
  }
}

}  // namespace fe::pyjson
