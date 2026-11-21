// Minimal strict JSON DOM for loading tokenizer.json / tokenizer_config.json inside fe_tokenizer only (keeps the
// library free of a link-time dependency on fe_common / simdjson). Not a general-purpose JSON API.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tokenizer/tokenizer.h"

namespace fe::tokjson {

struct Value {
  enum Type : uint8_t { Null, Bool, Number, String, Array, Object } type = Null;
  bool b = false;
  std::string str;  // String value, or the raw Number text
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;

  const Value* get(std::string_view key) const {
    if (type != Object) return nullptr;
    for (auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  bool is_null() const { return type == Null; }
  bool truthy() const { return type == Bool && b; }
  long long as_int() const {
    if (type != Number) throw TokenizerError("tokenizer json: expected a number");
    return std::stoll(str);
  }
  const std::string& as_str() const {
    if (type != String) throw TokenizerError("tokenizer json: expected a string");
    return str;
  }
};

class Parser {
 public:
  explicit Parser(std::string_view s) : p_(s.data()), e_(s.data() + s.size()) {}
  Value parse() {
    Value v;
    ws();
    value(v, 0);
    ws();
    if (p_ != e_) fail("trailing data");
    return v;
  }

 private:
  const char* p_;
  const char* e_;
  [[noreturn]] void fail(const char* m) { throw TokenizerError(std::string("tokenizer json parse error: ") + m); }
  void ws() {
    while (p_ < e_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) p_++;
  }
  void lit(const char* w) {
    size_t n = strlen(w);
    if (size_t(e_ - p_) < n || memcmp(p_, w, n) != 0) fail("bad literal");
    p_ += n;
  }
  static void put_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) {
      o += char(cp);
    } else if (cp < 0x800) {
      o += char(0xC0 | (cp >> 6));
      o += char(0x80 | (cp & 63));
    } else if (cp < 0x10000) {
      o += char(0xE0 | (cp >> 12));
      o += char(0x80 | ((cp >> 6) & 63));
      o += char(0x80 | (cp & 63));
    } else {
      o += char(0xF0 | (cp >> 18));
      o += char(0x80 | ((cp >> 12) & 63));
      o += char(0x80 | ((cp >> 6) & 63));
      o += char(0x80 | (cp & 63));
    }
  }
  uint32_t hex4() {
    if (e_ - p_ < 4) fail("short \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
      char c = *p_++;
      v <<= 4;
      if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
      else fail("bad hex");
    }
    return v;
  }
  void string(std::string& o) {
    p_++;  // opening quote
    for (;;) {
      const char* q = p_;
      while (q < e_ && *q != '"' && *q != '\\') q++;
      o.append(p_, q);
      p_ = q;
      if (p_ >= e_) fail("unterminated string");
      if (*p_ == '"') {
        p_++;
        return;
      }
      p_++;  // backslash
      if (p_ >= e_) fail("bad escape");
      char c = *p_++;
      switch (c) {
        case '"': o += '"'; break;
        case '\\': o += '\\'; break;
        case '/': o += '/'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'n': o += '\n'; break;
        case 'r': o += '\r'; break;
        case 't': o += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (e_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
              p_ += 2;
              uint32_t lo = hex4();
              if (lo < 0xDC00 || lo > 0xDFFF) fail("lone surrogate");
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else {
              fail("lone surrogate");
            }
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            fail("lone surrogate");
          }
          put_utf8(o, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
  }
  void value(Value& v, int depth) {
    if (depth > 256) fail("too deep");
    if (p_ >= e_) fail("unexpected end");
    char c = *p_;
    if (c == '{') {
      v.type = Value::Object;
      p_++;
      ws();
      if (p_ < e_ && *p_ == '}') {
        p_++;
        return;
      }
      for (;;) {
        ws();
        if (p_ >= e_ || *p_ != '"') fail("expected key");
        v.obj.emplace_back();
        string(v.obj.back().first);
        ws();
        if (p_ >= e_ || *p_ != ':') fail("expected ':'");
        p_++;
        ws();
        value(v.obj.back().second, depth + 1);
        ws();
        if (p_ < e_ && *p_ == ',') {
          p_++;
          continue;
        }
        if (p_ < e_ && *p_ == '}') {
          p_++;
          return;
        }
        fail("expected ',' or '}'");
      }
    } else if (c == '[') {
      v.type = Value::Array;
      p_++;
      ws();
      if (p_ < e_ && *p_ == ']') {
        p_++;
        return;
      }
      for (;;) {
        ws();
        v.arr.emplace_back();
        value(v.arr.back(), depth + 1);
        ws();
        if (p_ < e_ && *p_ == ',') {
          p_++;
          continue;
        }
        if (p_ < e_ && *p_ == ']') {
          p_++;
          return;
        }
        fail("expected ',' or ']'");
      }
    } else if (c == '"') {
      v.type = Value::String;
      string(v.str);
    } else if (c == 't') {
      lit("true");
      v.type = Value::Bool;
      v.b = true;
    } else if (c == 'f') {
      lit("false");
      v.type = Value::Bool;
      v.b = false;
    } else if (c == 'n') {
      lit("null");
      v.type = Value::Null;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
      const char* q = p_;
      while (q < e_ && (*q == '-' || *q == '+' || *q == '.' || *q == 'e' || *q == 'E' || (*q >= '0' && *q <= '9'))) q++;
      v.type = Value::Number;
      v.str.assign(p_, q);
      p_ = q;
    } else {
      fail("unexpected character");
    }
  }
};

inline Value parse(std::string_view s) { return Parser(s).parse(); }

}  // namespace fe::tokjson
