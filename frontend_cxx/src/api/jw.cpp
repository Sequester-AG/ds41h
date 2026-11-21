#include "api/jw.h"
#define FE_HAVE_COMMON 1

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace fe::api {

namespace {
const char kHex[] = "0123456789abcdef";

void u_escape(std::string& out, uint32_t cp) {
  out += "\\u";
  out += kHex[(cp >> 12) & 0xf];
  out += kHex[(cp >> 8) & 0xf];
  out += kHex[(cp >> 4) & 0xf];
  out += kHex[cp & 0xf];
}

// Decodes one UTF-8 sequence at s[i]; returns code point and advances i. Invalid bytes
// yield U+FFFD (cannot occur for strings that came from a Python str).
uint32_t next_cp(std::string_view s, size_t& i) {
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
  for (int k = 1; k <= n; ++k) {
    unsigned char cc = static_cast<unsigned char>(s[i + size_t(k)]);
    if ((cc & 0xc0) != 0x80) {
      ++i;
      return 0xfffd;
    }
    cp = (cp << 6) | (cc & 0x3f);
  }
  i += size_t(n) + 1;
  return cp;
}
}  // namespace

void append_escaped_pyd(std::string& out, std::string_view s, bool ensure_ascii) {
#ifdef FE_HAVE_COMMON
  fe::append_pydantic_string(out, s, ensure_ascii);
#else
  append_escaped(out, s, ensure_ascii);
#endif
}

void append_escaped(std::string& out, std::string_view s, bool ensure_ascii) {
#ifdef FE_HAVE_COMMON
  fe::append_py_string(out, s, ensure_ascii);
  return;
#endif
  out += '"';
  size_t i = 0;
  size_t run = 0;  // start of a verbatim run
  auto flush = [&](size_t upto) {
    if (upto > run) out.append(s.data() + run, upto - run);
  };
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c >= 0x20 && c != '"' && c != '\\' && c < 0x7f) {
      ++i;
      continue;
    }
    if (c >= 0x80 && !ensure_ascii) {
      ++i;
      continue;
    }
    if (c == 0x7f && !ensure_ascii) {
      ++i;
      continue;
    }
    flush(i);
    if (c < 0x80) {
      switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        default: u_escape(out, c); break;
      }
      ++i;
    } else {
      uint32_t cp = next_cp(s, i);
      if (cp >= 0x10000) {
        uint32_t v = cp - 0x10000;
        u_escape(out, 0xd800 | (v >> 10));
        u_escape(out, 0xdc00 | (v & 0x3ff));
      } else {
        u_escape(out, cp);
      }
    }
    run = i;
  }
  flush(i);
  out += '"';
}

namespace {
// Shortest round-trip digits and decimal exponent: value = 0.d1d2...dn * 10^decpt
// (Python's _Py_dg_dtoa mode 0 convention: decpt = position of the decimal point).
void shortest(double v, std::string& digits, int& decpt, bool& neg) {
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
  std::string_view s(buf, size_t(r.ptr - buf));
  neg = false;
  if (!s.empty() && s[0] == '-') {
    neg = true;
    s.remove_prefix(1);
  }
  size_t e = s.find('e');
  std::string_view mant = s.substr(0, e);
  int exp10 = std::atoi(std::string(s.substr(e + 1)).c_str());
  digits.clear();
  for (char c : mant)
    if (c != '.') digits += c;
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
  decpt = exp10 + 1;
}
}  // namespace

std::string py_float_repr(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
  std::string d;
  int decpt;
  bool neg;
  shortest(v, d, decpt, neg);
  std::string out = neg ? "-" : "";
  if (decpt <= -4 || decpt > 16) {
    out += d[0];
    if (d.size() > 1) {
      out += '.';
      out.append(d, 1, std::string::npos);
    }
    int e = decpt - 1;
    char eb[16];
    snprintf(eb, sizeof eb, "e%c%02d", e < 0 ? '-' : '+', e < 0 ? -e : e);
    out += eb;
  } else if (decpt <= 0) {
    out += "0.";
    out.append(size_t(-decpt), '0');
    out += d;
  } else if (size_t(decpt) >= d.size()) {
    out += d;
    out.append(size_t(decpt) - d.size(), '0');
    out += ".0";
  } else {
    out.append(d, 0, size_t(decpt));
    out += '.';
    out.append(d, size_t(decpt), std::string::npos);
  }
  return out;
}

std::string ryu_float(double v) {
  if (std::isnan(v) || std::isinf(v)) return "null";  // serde_json writes null
  if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
  std::string d;
  int decpt;
  bool neg;
  shortest(v, d, decpt, neg);
  // ryu: value = digits * 10^k, kk = len + k = decpt
  int len = int(d.size());
  int k = decpt - len;
  int kk = decpt;
  std::string out = neg ? "-" : "";
  if (0 <= k && kk <= 16) {
    out += d;
    out.append(size_t(k), '0');
    out += ".0";
  } else if (0 < kk && kk <= 16) {
    out.append(d, 0, size_t(kk));
    out += '.';
    out.append(d, size_t(kk), std::string::npos);
  } else if (-5 < kk && kk <= 0) {
    out += "0.";
    out.append(size_t(-kk), '0');
    out += d;
  } else {
    out += d[0];
    if (len > 1) {
      out += '.';
      out.append(d, 1, std::string::npos);
    }
    out += 'e';
    if (kk - 1 >= 0) out += '+';  // pydantic-core writes "1e+16", "1e-7"
    out += std::to_string(kk - 1);
  }
  return out;
}

}  // namespace fe::api
