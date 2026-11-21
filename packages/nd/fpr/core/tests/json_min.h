// json_min.h -- a minimal JSON reader for the golden-trace replays. No dependencies.
// Keeps object key order, keeps int vs float (Python's json writes ints without '.' / exponent), accepts Python's
// Infinity / -Infinity / NaN. Doubles are parsed with strtod (correctly rounded: Python's repr round-trips exactly).
#pragma once
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jmin {

struct J {
  enum T { NUL, BOOL, INT, DBL, STR, ARR, OBJ } t = NUL;
  bool b = false;
  int64_t i = 0;
  double d = 0.0;
  std::string s;
  std::vector<J> a;
  std::vector<std::pair<std::string, J>> o;

  bool is_null() const { return t == NUL; }
  bool is_num() const { return t == INT || t == DBL || t == BOOL; }
  double num() const { return t == DBL ? d : t == INT ? (double)i : t == BOOL ? (b ? 1.0 : 0.0) : NAN; }
  int64_t integer() const { return t == INT ? i : t == BOOL ? (int64_t)b : t == DBL ? (int64_t)d : 0; }
  bool truthy() const {
    switch (t) {
      case NUL: return false;
      case BOOL: return b;
      case INT: return i != 0;
      case DBL: return d != 0.0;
      case STR: return !s.empty();
      case ARR: return !a.empty();
      case OBJ: return !o.empty();
    }
    return false;
  }
  const J* get(const char* k) const {
    for (auto& e : o) if (e.first == k) return &e.second;
    return nullptr;
  }
  const J& operator[](const char* k) const {
    static const J null_;
    const J* p = get(k);
    return p ? *p : null_;
  }
  const J& operator[](int k) const { return (*this)[(size_t)k]; }
  const J& operator[](size_t k) const {
    static const J null_;
    return k < a.size() ? a[k] : null_;
  }
  size_t size() const { return t == ARR ? a.size() : t == OBJ ? o.size() : 0; }
  std::string dump(size_t max = 400) const;
};

class Parser {
 public:
  explicit Parser(const char* p, const char* e) : p_(p), e_(e) {}
  J parse() {
    J v = value();
    ws();
    return v;
  }

 private:
  const char* p_;
  const char* e_;
  void ws() { while (p_ < e_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\t' || *p_ == '\r')) ++p_; }
  [[noreturn]] void err(const char* m) { throw std::runtime_error(std::string("json: ") + m); }
  bool lit(const char* s) {
    size_t n = std::strlen(s);
    if ((size_t)(e_ - p_) >= n && std::memcmp(p_, s, n) == 0) { p_ += n; return true; }
    return false;
  }
  J value() {
    ws();
    if (p_ >= e_) err("eof");
    J v;
    char c = *p_;
    if (c == '{') {
      ++p_;
      v.t = J::OBJ;
      ws();
      if (p_ < e_ && *p_ == '}') { ++p_; return v; }
      while (true) {
        ws();
        J k = value();
        if (k.t != J::STR) err("key");
        ws();
        if (p_ >= e_ || *p_ != ':') err("colon");
        ++p_;
        v.o.emplace_back(std::move(k.s), value());
        ws();
        if (p_ < e_ && *p_ == ',') { ++p_; continue; }
        if (p_ < e_ && *p_ == '}') { ++p_; break; }
        err("object");
      }
      return v;
    }
    if (c == '[') {
      ++p_;
      v.t = J::ARR;
      ws();
      if (p_ < e_ && *p_ == ']') { ++p_; return v; }
      while (true) {
        v.a.push_back(value());
        ws();
        if (p_ < e_ && *p_ == ',') { ++p_; continue; }
        if (p_ < e_ && *p_ == ']') { ++p_; break; }
        err("array");
      }
      return v;
    }
    if (c == '"') {
      ++p_;
      v.t = J::STR;
      while (p_ < e_ && *p_ != '"') {
        if (*p_ == '\\') {
          ++p_;
          if (p_ >= e_) err("escape");
          char x = *p_++;
          switch (x) {
            case 'n': v.s += '\n'; break;
            case 't': v.s += '\t'; break;
            case 'r': v.s += '\r'; break;
            case 'b': v.s += '\b'; break;
            case 'f': v.s += '\f'; break;
            case 'u': {
              if (e_ - p_ < 4) err("\\u");
              unsigned cp = (unsigned)std::strtoul(std::string(p_, 4).c_str(), nullptr, 16);
              p_ += 4;
              if (cp >= 0xD800 && cp < 0xDC00 && e_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
                unsigned lo = (unsigned)std::strtoul(std::string(p_ + 2, 4).c_str(), nullptr, 16);
                p_ += 6;
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              }
              if (cp < 0x80) v.s += (char)cp;
              else if (cp < 0x800) { v.s += (char)(0xC0 | (cp >> 6)); v.s += (char)(0x80 | (cp & 0x3F)); }
              else if (cp < 0x10000) { v.s += (char)(0xE0 | (cp >> 12)); v.s += (char)(0x80 | ((cp >> 6) & 0x3F)); v.s += (char)(0x80 | (cp & 0x3F)); }
              else { v.s += (char)(0xF0 | (cp >> 18)); v.s += (char)(0x80 | ((cp >> 12) & 0x3F)); v.s += (char)(0x80 | ((cp >> 6) & 0x3F)); v.s += (char)(0x80 | (cp & 0x3F)); }
              break;
            }
            default: v.s += x;
          }
        } else {
          v.s += *p_++;
        }
      }
      if (p_ >= e_) err("string");
      ++p_;
      return v;
    }
    if (lit("true")) { v.t = J::BOOL; v.b = true; return v; }
    if (lit("false")) { v.t = J::BOOL; v.b = false; return v; }
    if (lit("null")) return v;
    if (lit("NaN")) { v.t = J::DBL; v.d = NAN; return v; }
    if (lit("Infinity")) { v.t = J::DBL; v.d = std::numeric_limits<double>::infinity(); return v; }
    if (lit("-Infinity")) { v.t = J::DBL; v.d = -std::numeric_limits<double>::infinity(); return v; }
    const char* s = p_;
    bool flt = false;
    while (p_ < e_ && (std::strchr("+-0123456789.eE", *p_) != nullptr)) {
      if (*p_ == '.' || *p_ == 'e' || *p_ == 'E') flt = true;
      ++p_;
    }
    if (p_ == s) err("value");
    std::string num(s, p_);
    if (!flt) {
      errno = 0;
      long long x = std::strtoll(num.c_str(), nullptr, 10);
      if (errno == 0) { v.t = J::INT; v.i = (int64_t)x; return v; }
    }
    v.t = J::DBL;
    v.d = std::strtod(num.c_str(), nullptr);
    return v;
  }
};

inline J parse(const std::string& s) { return Parser(s.data(), s.data() + s.size()).parse(); }

inline void dump_to(const J& v, std::string& o, size_t max) {
  if (o.size() > max) return;
  char buf[64];
  switch (v.t) {
    case J::NUL: o += "null"; break;
    case J::BOOL: o += v.b ? "true" : "false"; break;
    case J::INT: o += std::to_string(v.i); break;
    case J::DBL: std::snprintf(buf, sizeof buf, "%.17g", v.d); o += buf; break;
    case J::STR: o += '"'; o += v.s; o += '"'; break;
    case J::ARR:
      o += '[';
      for (size_t i = 0; i < v.a.size(); ++i) { if (i) o += ','; dump_to(v.a[i], o, max); }
      o += ']';
      break;
    case J::OBJ:
      o += '{';
      for (size_t i = 0; i < v.o.size(); ++i) {
        if (i) o += ',';
        o += '"'; o += v.o[i].first; o += "\":";
        dump_to(v.o[i].second, o, max);
      }
      o += '}';
      break;
  }
}
inline std::string J::dump(size_t max) const {
  std::string o;
  dump_to(*this, o, max);
  if (o.size() > max) { o.resize(max); o += "..."; }
  return o;
}

}  // namespace jmin
