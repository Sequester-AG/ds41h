// Python value semantics on fe::Json. See pyval.h.
#include "render/pyval.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <vector>

namespace fe::render {

bool py_truthy(const Json& v) noexcept {
  switch (v.type()) {
    case Json::Type::Null: return false;
    case Json::Type::Bool: return v.as_bool();
    case Json::Type::Int: return v.as_int() != 0;
    case Json::Type::BigInt: return true;  // canonical text, never zero
    case Json::Type::Float: return v.as_double() != 0.0;  // NaN is truthy
    case Json::Type::String: return !v.as_string().empty();
    case Json::Type::Array: return !v.as_array().empty();
    case Json::Type::Object: return !v.as_object().empty();
  }
  return false;
}

const char* py_type_name(const Json& v) noexcept {
  switch (v.type()) {
    case Json::Type::Null: return "NoneType";
    case Json::Type::Bool: return "bool";
    case Json::Type::Int:
    case Json::Type::BigInt: return "int";
    case Json::Type::Float: return "float";
    case Json::Type::String: return "str";
    case Json::Type::Array: return "list";
    case Json::Type::Object: return "dict";
  }
  return "object";
}

bool py_hashable(const Json& v) noexcept { return !v.is_array() && !v.is_object(); }

// ---------------------------------------------------------------------------------------------- WTF-8 decoding
namespace {

// Decode one code point at s[i] (WTF-8: surrogates allowed). Invalid byte -> returns the byte | 0x80000000.
uint32_t next_cp(std::string_view s, size_t& i) {
  unsigned char c = static_cast<unsigned char>(s[i]);
  if (c < 0x80) { ++i; return c; }
  int n = 0;
  uint32_t cp = 0;
  if ((c & 0xE0) == 0xC0) { n = 1; cp = c & 0x1F; }
  else if ((c & 0xF0) == 0xE0) { n = 2; cp = c & 0x0F; }
  else if ((c & 0xF8) == 0xF0) { n = 3; cp = c & 0x07; }
  else { ++i; return 0x80000000u | c; }
  for (int k = 1; k <= n; ++k) {
    if (i + k >= s.size()) { ++i; return 0x80000000u | c; }
    unsigned char d = static_cast<unsigned char>(s[i + k]);
    if ((d & 0xC0) != 0x80) { ++i; return 0x80000000u | c; }
    cp = (cp << 6) | (d & 0x3F);
  }
  i += n + 1;
  return cp;
}

void put_hex(std::string& out, const char* prefix, uint32_t v, int width) {
  char buf[16];
  std::snprintf(buf, sizeof buf, "%s%0*x", prefix, width, v);
  out += buf;
}


struct Range { uint32_t lo, hi; };
// Not printable for str.isprintable(): Cc, Cf, Cs, Co, Zl, Zp, Zs (except U+0020), and the unassigned (Cn) blocks
// below. Sorted, non-overlapping.
constexpr Range kNonPrintable[] = {
    {0x0000, 0x001F}, {0x007F, 0x00A0}, {0x00AD, 0x00AD}, {0x0378, 0x0379}, {0x0380, 0x0383}, {0x038B, 0x038B},
    {0x038D, 0x038D}, {0x03A2, 0x03A2}, {0x0530, 0x0530}, {0x0557, 0x0558}, {0x058B, 0x058C}, {0x0590, 0x0590},
    {0x05C8, 0x05CF}, {0x05EB, 0x05EE}, {0x05F5, 0x0605}, {0x061C, 0x061C}, {0x06DD, 0x06DD}, {0x070E, 0x070F},
    {0x074B, 0x074C}, {0x07B2, 0x07BF}, {0x07FB, 0x07FC}, {0x082E, 0x082F}, {0x083F, 0x083F}, {0x085C, 0x085D},
    {0x085F, 0x085F}, {0x086B, 0x086F}, {0x088F, 0x0897}, {0x08E2, 0x08E2}, {0x1680, 0x1680}, {0x180E, 0x180E},
    {0x2000, 0x200F}, {0x2028, 0x202F}, {0x205F, 0x206F}, {0x2072, 0x2073}, {0x208F, 0x208F}, {0x3000, 0x3000},
    {0xD800, 0xDFFF}, {0xE000, 0xF8FF}, {0xFDD0, 0xFDEF}, {0xFEFF, 0xFEFF}, {0xFFF0, 0xFFFB}, {0xFFFE, 0xFFFF},
    {0x110BD, 0x110BD}, {0x110CD, 0x110CD}, {0x13430, 0x1343F}, {0x1BCA0, 0x1BCA3}, {0x1D173, 0x1D17A},
    {0x1FFFE, 0x1FFFF}, {0x2FFFE, 0x2FFFF}, {0x323B0, 0xE00FF}, {0xE01F0, 0x10FFFF},
};

}  // namespace

bool py_isprintable(uint32_t cp) noexcept {
  if (cp >= 0x20 && cp < 0x7F) return true;
  size_t lo = 0, hi = sizeof(kNonPrintable) / sizeof(kNonPrintable[0]);
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < kNonPrintable[mid].lo) hi = mid;
    else if (cp > kNonPrintable[mid].hi) lo = mid + 1;
    else return false;
  }
  return true;
}

void py_repr_str_append(std::string& out, std::string_view s) {
  bool has_sq = s.find('\'') != std::string_view::npos;
  bool has_dq = s.find('"') != std::string_view::npos;
  char quote = (has_sq && !has_dq) ? '"' : '\'';
  out += quote;
  size_t i = 0;
  while (i < s.size()) {
    size_t start = i;
    uint32_t cp = next_cp(s, i);
    if (cp & 0x80000000u) {  // invalid byte: cannot occur in a Python str; keep the byte
      out += s[start];
      continue;
    }
    if (cp == static_cast<uint32_t>(quote) || cp == '\\') { out += '\\'; out += static_cast<char>(cp); }
    else if (cp == '\t') out += "\\t";
    else if (cp == '\n') out += "\\n";
    else if (cp == '\r') out += "\\r";
    else if (cp < 0x20 || cp == 0x7F) put_hex(out, "\\x", cp, 2);
    else if (cp < 0x7F) out += static_cast<char>(cp);
    else if (py_isprintable(cp)) out.append(s.substr(start, i - start));
    else if (cp <= 0xFF) put_hex(out, "\\x", cp, 2);
    else if (cp <= 0xFFFF) put_hex(out, "\\u", cp, 4);
    else put_hex(out, "\\U", cp, 8);
  }
  out += quote;
}

std::string py_repr_str(std::string_view s) {
  std::string out;
  py_repr_str_append(out, s);
  return out;
}

void py_repr_append(std::string& out, const Json& v) {
  switch (v.type()) {
    case Json::Type::Null: out += "None"; return;
    case Json::Type::Bool: out += v.as_bool() ? "True" : "False"; return;
    case Json::Type::Int: out += std::to_string(v.as_int()); return;
    case Json::Type::BigInt: out += v.big_int_text(); return;
    case Json::Type::Float: out += py_float_repr(v.as_double()); return;
    case Json::Type::String: py_repr_str_append(out, v.as_string()); return;
    case Json::Type::Array: {
      out += '[';
      bool first = true;
      for (const auto& x : v.as_array()) {
        if (!first) out += ", ";
        first = false;
        py_repr_append(out, x);
      }
      out += ']';
      return;
    }
    case Json::Type::Object: {
      out += '{';
      bool first = true;
      for (const auto& [k, x] : v.as_object()) {
        if (!first) out += ", ";
        first = false;
        py_repr_str_append(out, k);
        out += ": ";
        py_repr_append(out, x);
      }
      out += '}';
      return;
    }
  }
}

std::string py_repr(const Json& v) {
  std::string out;
  py_repr_append(out, v);
  return out;
}

void py_str_append(std::string& out, const Json& v) {
  if (v.is_string()) out += v.as_string();
  else py_repr_append(out, v);
}

std::string py_str(const Json& v) {
  if (v.is_string()) return v.as_string();
  return py_repr(v);
}

// ---------------------------------------------------------------------------------------------- exceptions
std::string PyError::str() const {
  if (cls != "VLLMValidationError" || (!param && !value)) return msg;
  std::string s = msg + " (";
  bool any = false;
  if (param) { s += "parameter=" + *param; any = true; }
  if (value) { if (any) s += ", "; s += "value=" + *value; }
  return s + ")";
}

void raise_vllm(std::string msg, std::optional<std::string> param, std::optional<std::string> value) {
  throw PyError("VLLMValidationError", std::move(msg), std::move(param), std::move(value));
}

void raise_py(const char* cls, std::string msg) { throw PyError(cls, std::move(msg)); }

// ---------------------------------------------------------------------------------------------- sanitize_message
namespace {

bool is_space_cp(uint32_t cp) {
  return cp == ' ' || (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x1F) || cp == 0x85 || cp == 0xA0 ||
         cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
         cp == 0x205F || cp == 0x3000;
}

// regex-module \w (approximation outside ASCII: letters / digits / marks, not punctuation / symbols / spaces)
bool is_word_cp(uint32_t cp) {
  if (cp < 0x80) return (cp >= '0' && cp <= '9') || (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || cp == '_';
  if (cp < 0xC0) return cp == 0xAA || cp == 0xB5 || cp == 0xBA;
  if (cp == 0xD7 || cp == 0xF7) return false;
  if (is_space_cp(cp)) return false;
  if (cp >= 0x2000 && cp <= 0x2BFF) return false;   // punctuation, symbols, arrows, math, box drawing
  if (cp >= 0x3000 && cp <= 0x303F) return cp == 0x3005 || cp == 0x3006 || cp == 0x3007;
  if (cp >= 0xFF00 && cp <= 0xFF0F) return false;
  if (cp >= 0xFE30 && cp <= 0xFE4F) return false;
  if (cp >= 0xD800 && cp <= 0xDFFF) return false;
  if (cp >= 0xE000 && cp <= 0xF8FF) return false;
  if (cp >= 0x1F000 && cp <= 0x1FAFF) return false;  // emoji / symbols
  return true;
}

struct Cps {
  std::vector<uint32_t> cp;
  std::vector<size_t> off;  // byte offset of cp[i]; off[n] = size
};

Cps decode(std::string_view s) {
  Cps c;
  size_t i = 0;
  while (i < s.size()) {
    c.off.push_back(i);
    uint32_t cp = next_cp(s, i);
    c.cp.push_back(cp & 0x80000000u ? 0xFFFD : cp);
  }
  c.off.push_back(s.size());
  return c;
}

bool word_or(uint32_t cp, const char* extra) {
  if (is_word_cp(cp)) return true;
  return cp < 0x80 && std::strchr(extra, static_cast<int>(cp)) != nullptr && cp != 0;
}

// Generic "replace every non-overlapping leftmost match" driver: match(cps, i) returns the end index or -1.
template <class F>
std::string sub_all(std::string_view s, const char* repl, F match) {
  Cps c = decode(s);
  std::string out;
  size_t n = c.cp.size(), i = 0, last = 0;
  while (i < n) {
    long e = match(c, i);
    if (e >= 0 && static_cast<size_t>(e) > i) {
      out.append(s.substr(c.off[last], c.off[i] - c.off[last]));
      out += repl;
      i = static_cast<size_t>(e);
      last = i;
    } else {
      ++i;
    }
  }
  out.append(s.substr(c.off[last]));
  return out;
}

long match_addr(const Cps& c, size_t i) {  // " at 0x[0-9a-fA-F]+>"
  static const char* pre = " at 0x";
  size_t n = c.cp.size();
  for (size_t k = 0; k < 6; ++k)
    if (i + k >= n || c.cp[i + k] != static_cast<uint32_t>(pre[k])) return -1;
  size_t j = i + 6, h = j;
  while (h < n && std::isxdigit(static_cast<int>(c.cp[h] < 0x80 ? c.cp[h] : 0))) ++h;
  if (h == j || h >= n || c.cp[h] != '>') return -1;
  return static_cast<long>(h + 1);
}

long match_file(const Cps& c, size_t i) {  // \n?\s*File "[^"]+", line \d+, in \S+(\n\s+.*)?
  size_t n = c.cp.size(), j = i;
  if (j < n && c.cp[j] == '\n') ++j;
  while (j < n && is_space_cp(c.cp[j])) ++j;
  auto lit = [&](const char* t) {
    size_t k = 0;
    for (; t[k]; ++k)
      if (j + k >= n || c.cp[j + k] != static_cast<uint32_t>(static_cast<unsigned char>(t[k]))) return false;
    j += k;
    return true;
  };
  if (!lit("File \"")) return -1;
  size_t q = j;
  while (j < n && c.cp[j] != '"') ++j;
  if (j == q || j >= n) return -1;
  if (!lit("\", line ")) return -1;
  size_t d = j;
  while (j < n && c.cp[j] < 0x80 && std::isdigit(static_cast<int>(c.cp[j]))) ++j;
  if (j == d) return -1;
  if (!lit(", in ")) return -1;
  size_t w = j;
  while (j < n && !is_space_cp(c.cp[j])) ++j;
  if (j == w) return -1;
  if (j < n && c.cp[j] == '\n') {
    size_t k = j + 1, ws = k;
    while (k < n && is_space_cp(c.cp[k])) ++k;
    if (k > ws) {
      // \s+ is greedy and may cross newlines; .* then runs to the end of that line
      while (k < n && c.cp[k] != '\n') ++k;
      j = k;
    }
  }
  return static_cast<long>(j);
}

long match_sys_path(const Cps& c, size_t i) {  // /(?:home|usr|opt|var|tmp|root|lib|mnt|srv)(?:/[\w.\-]+)+
  static const char* words[] = {"home", "usr", "opt", "var", "tmp", "root", "lib", "mnt", "srv"};
  size_t n = c.cp.size();
  if (c.cp[i] != '/') return -1;
  for (const char* w : words) {
    size_t j = i + 1, k = 0;
    for (; w[k]; ++k)
      if (j + k >= n || c.cp[j + k] != static_cast<uint32_t>(w[k])) break;
    if (w[k]) continue;
    j += k;
    size_t segs = 0;
    while (j < n && c.cp[j] == '/' && j + 1 < n && word_or(c.cp[j + 1], ".-")) {
      j += 1;
      while (j < n && word_or(c.cp[j], ".-")) ++j;
      ++segs;
    }
    if (segs > 0) return static_cast<long>(j);
  }
  return -1;
}

long match_file_path(const Cps& c, size_t i) {  // (?:/[\w\-]+)+/[\w\-]+\.\w+
  size_t n = c.cp.size(), j = i, segs = 0;
  while (j < n && c.cp[j] == '/' && j + 1 < n && word_or(c.cp[j + 1], "-")) {
    j += 1;
    while (j < n && word_or(c.cp[j], "-")) ++j;
    ++segs;
  }
  if (segs < 2 || j + 1 >= n || c.cp[j] != '.' || !is_word_cp(c.cp[j + 1])) return -1;
  j += 1;
  while (j < n && is_word_cp(c.cp[j])) ++j;
  return static_cast<long>(j);
}

}  // namespace

std::string sanitize_message(std::string_view message) {
  std::string s = sub_all(message, ">", match_addr);
  s = sub_all(s, "", match_file);
  s = sub_all(s, "<path>", match_sys_path);
  s = sub_all(s, "<path>", match_file_path);
  // str.strip()
  Cps c = decode(s);
  size_t a = 0, b = c.cp.size();
  while (a < b && is_space_cp(c.cp[a])) ++a;
  while (b > a && is_space_cp(c.cp[b - 1])) --b;
  return s.substr(c.off[a], c.off[b] - c.off[a]);
}

}  // namespace fe::render
