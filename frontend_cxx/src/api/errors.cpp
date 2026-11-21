#include "api/errors.h"

#include <cmath>
#include <regex>

#include "api/jw.h"

namespace fe::api {

ApiError vllm_validation_error(std::string msg, std::optional<std::string> param,
                               std::optional<std::string> value_str) {
  std::string m = msg;
  std::vector<std::string> extras;
  if (param) extras.push_back("parameter=" + *param);
  if (value_str) extras.push_back("value=" + *value_str);
  if (!extras.empty()) {
    m += " (";
    for (size_t i = 0; i < extras.size(); ++i) {
      if (i) m += ", ";
      m += extras[i];
    }
    m += ")";
  }
  return ApiError(400, "BadRequestError", m, param);
}

namespace {
std::string py_strip(std::string_view s) {
  // str.strip(): Unicode whitespace; ASCII subset is enough for our messages.
  auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f' ||
                                c == '\x1c' || c == '\x1d' || c == '\x1e' || c == '\x1f'; };
  size_t b = 0, e = s.size();
  while (b < e && ws(s[b])) ++b;
  while (e > b && ws(s[e - 1])) --e;
  return std::string(s.substr(b, e - b));
}
}  // namespace

std::string sanitize_message(std::string_view msg) {
  std::string m(msg);
  // Fast path: none of the patterns can match without '/' or " at 0x" or 'File "'.
  if (m.find('/') == std::string::npos && m.find(" at 0x") == std::string::npos &&
      m.find("File \"") == std::string::npos)
    return py_strip(m);
  static const std::regex r1(R"( at 0x[0-9a-fA-F]+>)");
  static const std::regex r2(R"(\n?\s*File "[^"]+", line \d+, in \S+(\n\s+.*)?)");
  static const std::regex r3(R"(/(?:home|usr|opt|var|tmp|root|lib|mnt|srv)(?:/[\w.\-]+)+)");
  static const std::regex r4(R"((?:/[\w\-]+)+/[\w\-]+\.\w+)");
  m = std::regex_replace(m, r1, ">");
  m = std::regex_replace(m, r2, "");
  m = std::regex_replace(m, r3, "<path>");
  m = std::regex_replace(m, r4, "<path>");
  return py_strip(m);
}

std::string error_body(const ApiError& e) {
  JW w(true, false, FloatStyle::PyRepr);
  w.ob().k("error").ob();
  w.kstr("message", sanitize_message(e.message));
  w.kstr("type", e.type);
  if (e.param)
    w.kstr("param", *e.param);
  else
    w.knull("param");
  w.ki("code", e.status);
  w.oe().oe();
  return w.take();
}

std::string error_body_stream(const ApiError& e) {
  JW w(false, true, FloatStyle::PyRepr);
  w.ob().k("error").ob();
  w.kstr("message", sanitize_message(e.message));
  w.kstr("type", e.type);
  if (e.param)
    w.kstr("param", *e.param);
  else
    w.knull("param");
  w.ki("code", e.status);
  w.oe().oe();
  return w.take();
}

// ---------------------------------------------------------------------------------------
// Python repr helpers

namespace {
uint32_t decode_cp(std::string_view s, size_t& i) {
  unsigned char c = static_cast<unsigned char>(s[i]);
  if (c < 0x80) {
    ++i;
    return c;
  }
  int n = (c >= 0xf0) ? 3 : (c >= 0xe0) ? 2 : (c >= 0xc0) ? 1 : -1;
  if (n < 0 || i + size_t(n) >= s.size()) {
    ++i;
    return 0xfffd;
  }
  uint32_t cp = c & (0x3f >> n);
  for (int k = 1; k <= n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + size_t(k)]) & 0x3f);
  i += size_t(n) + 1;
  return cp;
}

bool py_printable(uint32_t cp) {
  if (cp < 0x20 || cp == 0x7f) return false;
  if (cp < 0x7f) return true;
  if (cp <= 0xa0) return false;            // C1 controls + NBSP (Zs)
  if (cp == 0xad) return false;            // soft hyphen (Cf)
  if (cp >= 0x600 && cp <= 0x605) return false;
  if (cp == 0x61c || cp == 0x6dd || cp == 0x70f || cp == 0x180e || cp == 0x1680) return false;
  if (cp >= 0x2000 && cp <= 0x200f) return false;
  if (cp >= 0x2028 && cp <= 0x202f) return false;
  if (cp >= 0x205f && cp <= 0x2064) return false;
  if (cp >= 0x2066 && cp <= 0x206f) return false;
  if (cp == 0x3000 || cp == 0xfeff) return false;
  if (cp >= 0xfff9 && cp <= 0xfffb) return false;
  if (cp >= 0xd800 && cp <= 0xf8ff) return false;  // surrogates + private use
  if (cp >= 0xf0000) return false;
  return true;
}

void put_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out += char(cp);
  } else if (cp < 0x800) {
    out += char(0xc0 | (cp >> 6));
    out += char(0x80 | (cp & 0x3f));
  } else if (cp < 0x10000) {
    out += char(0xe0 | (cp >> 12));
    out += char(0x80 | ((cp >> 6) & 0x3f));
    out += char(0x80 | (cp & 0x3f));
  } else {
    out += char(0xf0 | (cp >> 18));
    out += char(0x80 | ((cp >> 12) & 0x3f));
    out += char(0x80 | ((cp >> 6) & 0x3f));
    out += char(0x80 | (cp & 0x3f));
  }
}
}  // namespace

std::string py_repr_str(std::string_view s) {
  char q = '\'';
  if (s.find('\'') != std::string_view::npos && s.find('"') == std::string_view::npos) q = '"';
  std::string out;
  out += q;
  size_t i = 0;
  while (i < s.size()) {
    uint32_t cp = decode_cp(s, i);
    if (cp == uint32_t(q) || cp == '\\') {
      out += '\\';
      out += char(cp);
    } else if (cp == '\t') {
      out += "\\t";
    } else if (cp == '\n') {
      out += "\\n";
    } else if (cp == '\r') {
      out += "\\r";
    } else if (!py_printable(cp)) {
      char b[16];
      if (cp < 0x100)
        snprintf(b, sizeof b, "\\x%02x", cp);
      else if (cp < 0x10000)
        snprintf(b, sizeof b, "\\u%04x", cp);
      else
        snprintf(b, sizeof b, "\\U%08x", cp);
      out += b;
    } else {
      put_utf8(out, cp);
    }
  }
  out += q;
  return out;
}

std::string py_str_scalar(const Json& v) {
  switch (v.type()) {
    case Json::Type::Null: return "None";
    case Json::Type::Bool: return v.as_bool() ? "True" : "False";
    case Json::Type::Int: return std::to_string(v.as_int());
    case Json::Type::BigInt: return v.big_int_text();
    case Json::Type::Float: {
      double d = v.as_double();
      if (std::isnan(d)) return "nan";
      if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
      return py_float_repr(d);
    }
    case Json::Type::String: return v.as_string();
    default: return py_repr(v);
  }
}

std::string py_repr(const Json& v) {
  switch (v.type()) {
    case Json::Type::String: return py_repr_str(v.as_string());
    case Json::Type::Array: {
      std::string out = "[";
      bool first = true;
      for (auto& x : v.as_array()) {
        if (!first) out += ", ";
        first = false;
        out += py_repr(x);
      }
      return out + "]";
    }
    case Json::Type::Object: {
      std::string out = "{";
      bool first = true;
      for (auto& [k, x] : v.as_object()) {
        if (!first) out += ", ";
        first = false;
        out += py_repr_str(k);
        out += ": ";
        out += py_repr(x);
      }
      return out + "}";
    }
    default: return py_str_scalar(v);
  }
}

std::string summarize_input_repr(const Json& v) {
  constexpr size_t kMax = 200;
  if (v.is_string()) {
    const std::string& s = v.as_string();
    // len() counts code points
    size_t cps = 0, cut = std::string::npos;
    for (size_t i = 0; i < s.size();) {
      if (cps == kMax) {
        cut = i;
        break;
      }
      decode_cp(s, i);
      ++cps;
    }
    if (cut != std::string::npos) return py_repr_str(s.substr(0, cut) + "...[truncated]");
    return py_repr_str(s);
  }
  if (v.is_array()) return py_repr_str("<list of " + std::to_string(v.size()) + " items>");
  if (v.is_object()) return py_repr_str("<dict of " + std::to_string(v.size()) + " items>");
  std::string t = py_str_scalar(v);
  if (t.size() > kMax) t = t.substr(0, kMax) + "...[truncated]";
  return py_repr_str(t);
}

std::string summarize_input_repr_none() { return "'None'"; }

namespace {
bool internal_loc_segment(std::string_view seg) {
  for (char c : seg)
    if (c == '[' || c == ']' || c == '{' || c == '}' || c == '(' || c == ')') return true;
  static const char* markers[] = {"function-wrap", "function-after", "function-before", "function-plain",
                                  "json-or-python", "lax-or-strict", "chain", "default", "nullable",
                                  "tagged-union", "union", "call", "arguments", "is-instance",
                                  "is-subclass", "callable", "str", "int", "float", "bool", "bytes",
                                  "bytearray", "list", "tuple", "dict", "set", "frozenset", "complex",
                                  "none", "nonetype"};
  std::string l(seg);
  for (auto& c : l) c = char(std::tolower(static_cast<unsigned char>(c)));
  for (auto* m : markers)
    if (l == m) return true;
  return false;
}
std::string clean_loc(const std::vector<std::string>& loc) {
  std::string out;
  bool any = false;
  for (auto& p : loc) {
    if (internal_loc_segment(p)) continue;
    if (any) out += '.';
    out += p;
    any = true;
  }
  if (!any) {
    for (size_t i = 0; i < loc.size(); ++i) {
      if (i) out += '.';
      out += loc[i];
    }
  }
  return out;
}
}  // namespace

ApiError validation_error(const std::vector<PydErr>& errs) {
  constexpr size_t kMaxReported = 10, kMaxChars = 1000;
  ApiError e;
  e.status = 400;
  e.type = "Bad Request";
  if (errs.empty()) {
    e.message = "Validation error";
    return e;
  }
  e.param = clean_loc(errs[0].loc);
  size_t count = errs.size();
  const char* label = count == 1 ? "error" : "errors";
  std::string m = std::to_string(count) + " validation " + label + ":\n";
  for (size_t i = 0; i < count && i < kMaxReported; ++i) {
    const PydErr& x = errs[i];
    std::string t = "{'type': " + py_repr_str(x.type) + ", 'loc': " + py_repr_str(clean_loc(x.loc)) +
                    ", 'msg': " + py_repr_str(x.msg) + ", 'input': " + x.input_repr;
    if (!x.ctx_repr.empty()) t += ", 'ctx': " + x.ctx_repr;
    t += "}";
    if (t.size() > kMaxChars) {
      // truncate on a code-point boundary at 1000 chars
      size_t cps = 0, i2 = 0;
      while (i2 < t.size() && cps < kMaxChars) {
        decode_cp(t, i2);
        ++cps;
      }
      if (i2 < t.size()) t = t.substr(0, i2) + "...[truncated]";
    }
    m += "  " + t + "\n";
  }
  if (count > kMaxReported)
    m += "  ...and " + std::to_string(count - kMaxReported) + " more " + label + "\n";
  while (!m.empty() && (m.back() == '\n' || m.back() == ' ')) m.pop_back();
  e.message = m;
  return e;
}

ApiError validation_error_plain(std::string_view text) {
  ApiError e;
  e.status = 400;
  e.type = "Bad Request";
  e.message = "1 validation error:\n  " + std::string(text);
  while (!e.message.empty() && (e.message.back() == '\n' || e.message.back() == ' ')) e.message.pop_back();
  return e;
}

}  // namespace fe::api
