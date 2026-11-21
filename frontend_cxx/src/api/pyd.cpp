#include "api/pyd.h"

#include <cmath>
#include <cstdlib>

namespace fe::api {

void Pyd::add(std::string type, std::vector<std::string> loc, std::string msg, const Json* input, std::string ctx) {
  PydErr e;
  e.type = std::move(type);
  e.loc = std::move(loc);
  e.msg = std::move(msg);
  e.input_repr = input ? summarize_input_repr(*input) : summarize_input_repr_none();
  e.ctx_repr = std::move(ctx);
  errs.push_back(std::move(e));
}

void Pyd::missing(std::vector<std::string> loc, const Json& parent) {
  add("missing", std::move(loc), "Field required", &parent);
}

namespace {
std::string strip_ws(std::string_view s) {
  size_t b = 0, e = s.size();
  auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; };
  while (b < e && ws(s[b])) ++b;
  while (e > b && ws(s[e - 1])) --e;
  return std::string(s.substr(b, e - b));
}

// Python-ish int() of a string: optional sign, digits with single underscores between digits.
bool parse_int_str(std::string_view s0, int64_t& out, bool& overflow) {
  std::string s = strip_ws(s0);
  overflow = false;
  if (s.empty()) return false;
  size_t i = 0;
  bool neg = false;
  if (s[0] == '+' || s[0] == '-') {
    neg = s[0] == '-';
    i = 1;
  }
  if (i >= s.size()) return false;
  __int128 v = 0;
  bool prev_digit = false;
  for (; i < s.size(); ++i) {
    char c = s[i];
    if (c == '_') {
      if (!prev_digit || i + 1 >= s.size() || !(s[i + 1] >= '0' && s[i + 1] <= '9')) return false;
      prev_digit = false;
      continue;
    }
    if (c < '0' || c > '9') {
      // "12.0" style: integral float text
      if (c == '.' || c == 'e' || c == 'E') {
        char* end = nullptr;
        double d = std::strtod(s.c_str(), &end);
        if (end && *end == '\0' && std::isfinite(d) && d == std::floor(d)) {
          if (std::fabs(d) > 9.2e18) {
            overflow = true;
            out = d < 0 ? INT64_MIN : INT64_MAX;
          } else {
            out = int64_t(d);
          }
          return true;
        }
      }
      return false;
    }
    prev_digit = true;
    if (v < (__int128(1) << 100)) v = v * 10 + (c - '0');
  }
  if (neg) v = -v;
  if (v > INT64_MAX) {
    overflow = true;
    out = INT64_MAX;
  } else if (v < INT64_MIN) {
    overflow = true;
    out = INT64_MIN;
  } else {
    out = int64_t(v);
  }
  return true;
}

std::string literal_expected(const std::vector<std::string>& allowed) {
  std::string e;
  for (size_t i = 0; i < allowed.size(); ++i) {
    if (i) e += (i + 1 == allowed.size()) ? " or " : ", ";
    e += "'" + allowed[i] + "'";
  }
  return e;
}
}  // namespace

std::optional<int64_t> Pyd::to_int(const Json* v, std::vector<std::string> loc, bool nullable,
                                   std::optional<int64_t> ge, std::optional<int64_t> le) {
  last_null = false;
  if (!v) return std::nullopt;
  if (v->is_null()) {
    last_null = true;
    if (nullable) return std::nullopt;
    add("int_type", loc, "Input should be a valid integer", v);
    return std::nullopt;
  }
  int64_t out = 0;
  bool big_pos = false, big_neg = false;
  switch (v->type()) {
    case Json::Type::Bool: out = v->as_bool() ? 1 : 0; break;
    case Json::Type::Int: out = v->as_int(); break;
    case Json::Type::BigInt: {
      const std::string& t = v->big_int_text();
      if (!t.empty() && t[0] == '-') {
        big_neg = true;
        out = INT64_MIN;
      } else {
        big_pos = true;
        out = INT64_MAX;
      }
      break;
    }
    case Json::Type::Float: {
      double d = v->as_double();
      if (!std::isfinite(d)) {
        add("finite_number", loc, "Input should be a finite number", v);
        return std::nullopt;
      }
      if (d != std::floor(d)) {
        add("int_from_float", loc, "Input should be a valid integer, got a number with a fractional part", v);
        return std::nullopt;
      }
      if (d >= 9.2233720368547758e18) {
        big_pos = true;
        out = INT64_MAX;
      } else if (d <= -9.2233720368547758e18) {
        big_neg = true;
        out = INT64_MIN;
      } else {
        out = int64_t(d);
      }
      break;
    }
    case Json::Type::String: {
      bool ovf = false;
      if (!parse_int_str(v->as_string(), out, ovf)) {
        add("int_parsing", loc, "Input should be a valid integer, unable to parse string as an integer", v);
        return std::nullopt;
      }
      if (ovf) (out > 0 ? big_pos : big_neg) = true;
      break;
    }
    default:
      add("int_type", loc, "Input should be a valid integer", v);
      return std::nullopt;
  }
  if (ge && (big_neg || out < *ge)) {
    add("greater_than_equal", loc, "Input should be greater than or equal to " + std::to_string(*ge), v,
        "{'ge': " + std::to_string(*ge) + "}");
    return std::nullopt;
  }
  if (le && (big_pos || out > *le)) {
    add("less_than_equal", loc, "Input should be less than or equal to " + std::to_string(*le), v,
        "{'le': " + std::to_string(*le) + "}");
    return std::nullopt;
  }
  return out;
}

std::optional<double> Pyd::to_float(const Json* v, std::vector<std::string> loc, bool nullable) {
  last_null = false;
  if (!v) return std::nullopt;
  if (v->is_null()) {
    last_null = true;
    if (nullable) return std::nullopt;
    add("float_type", loc, "Input should be a valid number", v);
    return std::nullopt;
  }
  switch (v->type()) {
    case Json::Type::Bool: return v->as_bool() ? 1.0 : 0.0;
    case Json::Type::Int:
    case Json::Type::BigInt:
    case Json::Type::Float: return v->as_double();
    case Json::Type::String: {
      std::string s = strip_ws(v->as_string());
      char* end = nullptr;
      double d = s.empty() ? 0 : std::strtod(s.c_str(), &end);
      bool hexish = s.find('x') != std::string::npos || s.find('X') != std::string::npos;
      if (s.empty() || !end || *end != '\0' || hexish) {
        add("float_parsing", loc, "Input should be a valid number, unable to parse string as a number", v);
        return std::nullopt;
      }
      return d;
    }
    default: add("float_type", loc, "Input should be a valid number", v); return std::nullopt;
  }
}

std::optional<bool> Pyd::to_bool(const Json* v, std::vector<std::string> loc, bool nullable) {
  last_null = false;
  if (!v) return std::nullopt;
  if (v->is_null()) {
    last_null = true;
    if (nullable) return std::nullopt;
    add("bool_type", loc, "Input should be a valid boolean", v);
    return std::nullopt;
  }
  switch (v->type()) {
    case Json::Type::Bool: return v->as_bool();
    case Json::Type::Int: {
      int64_t i = v->as_int();
      if (i == 0 || i == 1) return i == 1;
      break;
    }
    case Json::Type::Float: {
      double d = v->as_double();
      if (d == 0.0 || d == 1.0) return d == 1.0;
      break;
    }
    case Json::Type::String: {
      std::string s = v->as_string();
      for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
      if (s == "0" || s == "off" || s == "f" || s == "false" || s == "n" || s == "no") return false;
      if (s == "1" || s == "on" || s == "t" || s == "true" || s == "y" || s == "yes") return true;
      break;
    }
    case Json::Type::BigInt: break;
    default: add("bool_type", loc, "Input should be a valid boolean", v); return std::nullopt;
  }
  add("bool_parsing", loc, "Input should be a valid boolean, unable to interpret input", v);
  return std::nullopt;
}

std::optional<std::string> Pyd::to_str(const Json* v, std::vector<std::string> loc, bool nullable) {
  last_null = false;
  if (!v) return std::nullopt;
  if (v->is_null()) {
    last_null = true;
    if (nullable) return std::nullopt;
  }
  if (v->is_string()) return v->as_string();
  add("string_type", loc, "Input should be a valid string", v);
  return std::nullopt;
}

std::optional<std::string> Pyd::to_literal(const Json* v, std::vector<std::string> loc, bool nullable,
                                           const std::vector<std::string>& allowed) {
  last_null = false;
  if (!v) return std::nullopt;
  if (v->is_null()) {
    last_null = true;
    if (nullable) return std::nullopt;
  }
  if (v->is_string())
    for (auto& a : allowed)
      if (v->as_string() == a) return a;
  std::string exp = literal_expected(allowed);
  add("literal_error", loc, "Input should be " + exp, v, "{'expected': " + py_repr_str(exp) + "}");
  return std::nullopt;
}

bool Pyd::is_list(const Json* v, std::vector<std::string> loc, bool nullable) {
  last_null = false;
  if (!v) return false;
  if (v->is_null()) {
    last_null = true;
    if (nullable) return false;
  }
  if (v->is_array()) return true;
  add("list_type", loc, "Input should be a valid list", v);
  return false;
}

bool Pyd::is_dict(const Json* v, std::vector<std::string> loc, bool nullable) {
  last_null = false;
  if (!v) return false;
  if (v->is_null()) {
    last_null = true;
    if (nullable) return false;
  }
  if (v->is_object()) return true;
  add("dict_type", loc, "Input should be a valid dictionary", v);
  return false;
}

bool Pyd::is_model(const Json* v, std::vector<std::string> loc, bool nullable) {
  last_null = false;
  if (!v) return false;
  if (v->is_null()) {
    last_null = true;
    if (nullable) return false;
  }
  if (v->is_object()) return true;
  add("model_attributes_type", loc, "Input should be a valid dictionary or object to extract fields from", v);
  return false;
}

}  // namespace fe::api
