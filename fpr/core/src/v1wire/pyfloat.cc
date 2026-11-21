// pyfloat.cc -- Python's repr(float) (float_repr_style "short") in C++. json.dumps prints floats
// with float.__repr__, so a json-schema dict that vLLM turns into a key string (get_structured_output_key) needs it:
// the shortest digits that round-trip, exponent form when the decimal exponent is < -4 or >= 16, "e-05" / "e+16" style
// exponents, ".0" added to integral values. json.dumps writes inf / nan as Infinity / NaN.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace fpr {
namespace v1 {

std::string py_float_repr(double x) {
  if (std::isnan(x)) return "NaN";
  if (std::isinf(x)) return x > 0 ? "Infinity" : "-Infinity";
  if (x == 0.0) return std::signbit(x) ? "-0.0" : "0.0";
  char buf[64];
  int prec = 0;
  for (prec = 0; prec <= 17; ++prec) {          // shortest round-trip in %e form: prec + 1 significant digits
    std::snprintf(buf, sizeof buf, "%.*e", prec, x);
    if (std::strtod(buf, nullptr) == x) break;
  }
  // buf = [-]d[.ddd]e[+-]XX
  std::string s(buf);
  bool neg = s[0] == '-';
  if (neg) s.erase(0, 1);
  size_t e = s.find('e');
  std::string mant = s.substr(0, e);
  int exp10 = std::atoi(s.c_str() + e + 1);
  std::string digits;
  for (char c : mant)
    if (c != '.') digits.push_back(c);
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
  std::string out = neg ? "-" : "";
  if (exp10 < -4 || exp10 >= 16) {
    out += digits[0];
    if (digits.size() > 1) {
      out += '.';
      out += digits.substr(1);
    }
    char eb[16];
    std::snprintf(eb, sizeof eb, "e%c%02d", exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
    out += eb;
    return out;
  }
  if (exp10 >= 0) {
    size_t ip = (size_t)exp10 + 1;
    if (digits.size() <= ip) {
      out += digits + std::string(ip - digits.size(), '0') + ".0";
    } else {
      out += digits.substr(0, ip) + "." + digits.substr(ip);
    }
  } else {
    out += "0." + std::string((size_t)(-exp10 - 1), '0') + digits;
  }
  return out;
}

}  // namespace v1
}  // namespace fpr
