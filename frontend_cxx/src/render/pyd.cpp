// Mini pydantic emulation. See pyd.h.
#include "render/pyd.h"

#include <cctype>
#include <cmath>
#include <deque>
#include <string_view>

#include "render/pyval.h"

namespace fe::render::pyd {

namespace {

bool lower_eq(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::tolower(static_cast<unsigned char>(a[i])) != b[i]) return false;
  return true;
}

std::string strip_ws(std::string_view s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return std::string(s.substr(a, b - a));
}

// errs == nullptr: dry run (a smart-union member probe); nothing is recorded or copied.
void add_err(Errors* errs, const JsonArray& loc, const char* type, std::string msg, const Json& input,
             Json ctx = Json()) {
  if (errs) errs->push_back(Error{type, loc, std::move(msg), input, std::move(ctx)});
}

JsonArray with(const JsonArray& loc, Json seg) {
  JsonArray l = loc;
  l.push_back(std::move(seg));
  return l;
}

std::string literal_expected(const std::vector<std::string>& lits) {
  std::string s;
  for (size_t i = 0; i < lits.size(); ++i) {
    if (i) s += (i + 1 == lits.size()) ? " or " : ", ";
    s += py_repr_str(lits[i]);
  }
  return s;
}

// Python int(str) as pydantic parses it (optional sign, digits, surrounding whitespace, '_' separators).
bool parse_int_text(std::string_view s0, Json& out) {
  std::string s = strip_ws(s0), d;
  // pydantic-core str -> int: "12.0" / "12.000" are accepted (strip_decimal_zeros), "12.5" is not
  if (size_t dot = s.find('.'); dot != std::string::npos && dot + 1 < s.size() &&
      s.find_first_not_of('0', dot + 1) == std::string::npos)
    s.resize(dot);
  size_t i = 0;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) d += s[i++] == '-' ? "-" : "";
  bool prev_digit = false;
  for (; i < s.size(); ++i) {
    if (std::isdigit(static_cast<unsigned char>(s[i]))) { d += s[i]; prev_digit = true; }
    else if (s[i] == '_' && prev_digit && i + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[i + 1]))) {}
    else return false;
  }
  if (d.empty() || d == "-") return false;
  try {
    out = Json::integer_from_text(d);
  } catch (const JsonError&) {
    return false;
  }
  return true;
}

bool parse_float_text(std::string_view s0, double& out) {
  std::string s = strip_ws(s0);
  if (s.empty()) return false;
  std::string low;
  for (char c : s) low += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  std::string t = (low[0] == '+' || low[0] == '-') ? low.substr(1) : low;
  if (t == "inf" || t == "infinity" || t == "nan") {
    out = t == "nan" ? NAN : (low[0] == '-' ? -INFINITY : INFINITY);
    return true;
  }
  char* end = nullptr;
  out = std::strtod(s.c_str(), &end);
  return end && *end == '\0';
}

int cmp_int(const Json& a, const Json& b) {  // both Int / BigInt
  if (a.is_int() && b.is_int()) return a.as_int() < b.as_int() ? -1 : (a.as_int() > b.as_int() ? 1 : 0);
  auto sign = [](const Json& x) { return x.is_int() ? (x.as_int() < 0 ? -1 : 1) : (x.big_int_text()[0] == '-' ? -1 : 1); };
  auto mag = [](const Json& x) { return x.is_int() ? std::to_string(x.as_int()) : x.big_int_text(); };
  int sa = sign(a), sb = sign(b);
  if (sa != sb) return sa < sb ? -1 : 1;
  std::string ma = mag(a), mb = mag(b);
  if (ma[0] == '-') ma = ma.substr(1);
  if (mb[0] == '-') mb = mb.substr(1);
  int c = ma.size() != mb.size() ? (ma.size() < mb.size() ? -1 : 1) : ma.compare(mb);
  c = c < 0 ? -1 : (c > 0 ? 1 : 0);
  return sa < 0 ? -c : c;
}

// validate_consume(): value builds may move from the (owned) input instead of copying it. Builds only happen for
// values already known to validate (the union winner's rebuild), so no error ever needs a moved-from input.
thread_local bool g_move = false;
Json take(const Json& x) { return g_move ? std::move(const_cast<Json&>(x)) : x; }

Result fail() { return Result{}; }
Result okv(const Json&, int n = 0) {
  Result r;
  r.ok = true;
  r.fields_set = n;
  return r;
}
Result coerce(Json v) {
  Result r;
  r.ok = true;
  r.coerced = true;
  r.value = std::move(v);
  return r;
}

}  // namespace

Result validate_impl(const Schema& s, const Json& v, const JsonArray& loc, Errors* errs, bool build = true);

std::string Schema::label() const {
  switch (kind) {
    case K::Any: return "any";
    case K::Str: return "str";
    case K::Int: return "int";
    case K::Float: return "float";
    case K::Bool: return "bool";
    case K::Literal: {
      std::string s = "literal[";
      for (size_t i = 0; i < lits.size(); ++i) s += (i ? "," : "") + py_repr_str(lits[i]);
      return s + "]";
    }
    case K::List: return "list[" + item->label() + "]";
    case K::Iterable: return "generator[" + item->label() + "]";
    case K::Dict: return "dict[str," + (item ? item->label() : std::string("any")) + "]";
    case K::TypedDict:
    case K::Model: return vllm_model ? "function-wrap[__log_extra_fields__()]" : name;
    case K::Union: {
      std::string s = "union[";
      for (size_t i = 0; i < members.size(); ++i) s += (i ? "," : "") + members[i]->label();
      return s + "]";
    }
    case K::Nullable: return "nullable[" + item->label() + "]";
  }
  return "?";
}

// build == false: a probe; containers report `coerced` without building the value (no copies).
Result validate_impl(const Schema& s, const Json& v, const JsonArray& loc, Errors* errs, bool build) {
  switch (s.kind) {
    case K::Any: return okv(v);
    case K::Str: {
      if (!v.is_string()) { add_err(errs, loc, "string_type", "Input should be a valid string", v); return fail(); }
      if (s.min_length < 0 && s.max_length < 0) return okv(v);
      size_t n = py_len(v.as_string());
      if (s.min_length >= 0 && n < static_cast<size_t>(s.min_length)) {
        add_err(errs, loc, "string_too_short",
                "String should have at least " + std::to_string(s.min_length) + " character" +
                    (s.min_length == 1 ? "" : "s"),
                v, Json::object({{"min_length", Json(s.min_length)}}));
        return fail();
      }
      if (s.max_length >= 0 && n > static_cast<size_t>(s.max_length)) {
        add_err(errs, loc, "string_too_long",
                "String should have at most " + std::to_string(s.max_length) + " character" +
                    (s.max_length == 1 ? "" : "s"),
                v, Json::object({{"max_length", Json(s.max_length)}}));
        return fail();
      }
      return okv(v);
    }
    case K::Int: {
      Json iv;
      if (v.is_integer()) iv = v;
      else if (v.is_bool()) iv = Json(static_cast<int64_t>(v.as_bool() ? 1 : 0));
      else if (v.is_float()) {
        double d = v.as_double();
        if (!std::isfinite(d)) { add_err(errs, loc, "finite_number", "Input should be a finite number", v); return fail(); }
        if (d != std::floor(d)) {
          add_err(errs, loc, "int_from_float", "Input should be a valid integer, got a number with a fractional part", v);
          return fail();
        }
        if (std::fabs(d) < 9.2e18) iv = Json(static_cast<int64_t>(d));
        else {
          char buf[64];
          std::snprintf(buf, sizeof buf, "%.0f", d);
          iv = Json::integer_from_text(buf);
        }
      } else if (v.is_string()) {
        if (!parse_int_text(v.as_string(), iv)) {
          add_err(errs, loc, "int_parsing", "Input should be a valid integer, unable to parse string as an integer", v);
          return fail();
        }
      } else {
        add_err(errs, loc, "int_type", "Input should be a valid integer", v);
        return fail();
      }
      if (s.has_ge && cmp_int(iv, s.ge) < 0) {
        add_err(errs, loc, "greater_than_equal", "Input should be greater than or equal to " + py_str(s.ge), v,
                Json::object({{"ge", s.ge}}));
        return fail();
      }
      if (s.has_le && cmp_int(iv, s.le) > 0) {
        add_err(errs, loc, "less_than_equal", "Input should be less than or equal to " + py_str(s.le), v,
                Json::object({{"le", s.le}}));
        return fail();
      }
      if (v.is_integer()) return okv(v);
      return coerce(std::move(iv));
    }
    case K::Float: {
      if (v.is_float()) return okv(v);
      if (v.is_integer()) return coerce(Json(v.as_double()));
      if (v.is_bool()) return coerce(Json(v.as_bool() ? 1.0 : 0.0));
      if (v.is_string()) {
        double d;
        if (parse_float_text(v.as_string(), d)) return coerce(Json(d));
        add_err(errs, loc, "float_parsing", "Input should be a valid number, unable to parse string as a number", v);
        return fail();
      }
      add_err(errs, loc, "float_type", "Input should be a valid number", v);
      return fail();
    }
    case K::Bool: {
      if (v.is_bool()) return okv(v);
      if (v.is_int() && (v.as_int() == 0 || v.as_int() == 1)) return coerce(Json(v.as_int() == 1));
      if (v.is_float() && (v.as_double() == 0.0 || v.as_double() == 1.0)) return coerce(Json(v.as_double() == 1.0));
      if (v.is_string()) {
        const std::string& t = v.as_string();
        for (const char* x : {"0", "off", "f", "false", "n", "no"})
          if (lower_eq(t, x)) return coerce(Json(false));
        for (const char* x : {"1", "on", "t", "true", "y", "yes"})
          if (lower_eq(t, x)) return coerce(Json(true));
      }
      if (v.is_number() || v.is_string()) {
        add_err(errs, loc, "bool_parsing", "Input should be a valid boolean, unable to interpret input", v);
        return fail();
      }
      add_err(errs, loc, "bool_type", "Input should be a valid boolean", v);
      return fail();
    }
    case K::Literal: {
      if (v.is_string())
        for (const auto& l : s.lits)
          if (v.as_string() == l) return okv(v);
      std::string exp = literal_expected(s.lits);
      add_err(errs, loc, "literal_error", "Input should be " + exp, v, Json::object({{"expected", Json(exp)}}));
      return fail();
    }
    case K::List: {
      if (!v.is_array()) { add_err(errs, loc, "list_type", "Input should be a valid list", v); return fail(); }
      bool ok = true;
      int n = 0;
      const JsonArray& a = v.as_array();
      JsonArray out;
      bool coerced = false;
      for (size_t i = 0; i < a.size(); ++i) {
        Result r = validate_impl(*s.item, a[i], errs ? with(loc, Json(static_cast<int64_t>(i))) : loc, errs, build);
        ok = ok && r.ok;
        n += r.fields_set;
        if (!build) {
          coerced = coerced || r.coerced;
        } else if (ok && r.coerced) {
          if (!coerced) {
            for (size_t j = 0; j < i; ++j) out.push_back(take(a[j]));
            coerced = true;
          }
          out.push_back(std::move(r.value));
        } else if (ok && coerced) {
          out.push_back(take(a[i]));
        }
      }
      if (!ok) return fail();
      if (!coerced) return okv(v, n);
      if (!build) {
        Result p = okv(v, n);
        p.coerced = true;
        return p;
      }
      Result r = coerce(Json(std::move(out)));
      r.fields_set = n;
      return r;
    }
    case K::Iterable: {
      if (v.is_array() || v.is_object() || v.is_string()) return okv(v);
      add_err(errs, loc, "iterable_type", "Input should be iterable", v);
      return fail();
    }
    case K::Dict: {
      if (!v.is_object()) { add_err(errs, loc, "dict_type", "Input should be a valid dictionary", v); return fail(); }
      if (!s.item) return okv(v);
      bool ok = true, coerced = false;
      Json out;
      for (const auto& [k, x] : v.as_object()) {
        Result r = validate_impl(*s.item, x, errs ? with(loc, Json(k)) : loc, errs, build);
        ok = r.ok && ok;
        if (r.ok && r.coerced) {
          if (!build) { coerced = true; continue; }
          if (!coerced) { out = take(v); coerced = true; }
          out.set(k, std::move(r.value));
        }
      }
      if (!ok) return fail();
      if (coerced && !build) {
        Result p = okv(v);
        p.coerced = true;
        return p;
      }
      return coerced ? coerce(std::move(out)) : okv(v);
    }
    case K::TypedDict:
    case K::Model: {
      if (!v.is_object()) {
        if (s.kind == K::TypedDict) add_err(errs, loc, "dict_type", "Input should be a valid dictionary", v);
        else add_err(errs, loc, "model_attributes_type",
                     "Input should be a valid dictionary or object to extract fields from", v);
        return fail();
      }
      if (s.kind == K::Model && s.fields.empty()) {  // is-instance check of an arbitrary type (PIL.Image)
        add_err(errs, loc, "is_instance_of", "Input should be an instance of " + s.name, v,
                Json::object({{"class", Json(s.name)}}));
        return fail();
      }
      bool ok = true;
      int n = 0;
      bool coerced = false;
      std::vector<std::pair<const Json*, Json>> vals(s.fields.size());  // (input value, coerced value)
      std::vector<bool> has_coerced(s.fields.size(), false);
      for (size_t fi = 0; fi < s.fields.size(); ++fi) {
        const Field& f = s.fields[fi];
        const Json* x = v.get(f.name);
        vals[fi].first = x;
        if (!x) {
          if (f.required) {
            if (errs) add_err(errs, with(loc, Json(f.name)), "missing", "Field required", v);
            ok = false;
          }
          continue;
        }
        Result r = validate_impl(*f.schema, *x, errs ? with(loc, Json(f.name)) : loc, errs, build);
        if (!r.ok) { ok = false; continue; }
        n += 1 + r.fields_set;
        if (r.coerced) {
          coerced = true;
          has_coerced[fi] = true;
          vals[fi].second = std::move(r.value);
        }
      }
      if (!ok) return fail();
      // A validated TypedDict lists its declared fields first (declaration order), then the extras (input order).
      bool reorder = false;
      if (s.kind == K::TypedDict) {
        int last = -1;
        bool seen_extra = false;
        for (const auto& [k, x] : v.as_object()) {
          int idx = -1;
          for (size_t fi = 0; fi < s.fields.size(); ++fi)
            if (s.fields[fi].name == k) { idx = static_cast<int>(fi); break; }
          if (idx < 0) {
            seen_extra = true;
          } else {
            if (seen_extra || idx < last) reorder = true;
            last = idx;
          }
        }
      }
      if (!coerced && !reorder) return okv(v, n);
      if (!build) {
        Result p = okv(v, n);
        p.coerced = true;
        return p;
      }
      Json out;
      if (reorder) {
        JsonObject o;
        o.reserve(v.size());
        for (size_t fi = 0; fi < s.fields.size(); ++fi)
          if (vals[fi].first) o.append_unique(s.fields[fi].name, has_coerced[fi] ? std::move(vals[fi].second) : take(*vals[fi].first));
        for (const auto& [k, x] : v.as_object()) {
          bool declared = false;
          for (const Field& f : s.fields)
            if (f.name == k) { declared = true; break; }
          if (!declared) o.append_unique(k, take(x));
        }
        out = Json(std::move(o));
      } else {
        out = take(v);
        for (size_t fi = 0; fi < s.fields.size(); ++fi)
          if (has_coerced[fi]) out.set(s.fields[fi].name, std::move(vals[fi].second));
      }
      Result r = coerce(std::move(out));
      r.fields_set = n;
      return r;
    }
    case K::Union: {
      int best = -1, best_n = -1;
      Result best_r;
      for (size_t i = 0; i < s.members.size(); ++i) {
        const Schema* m = s.members[i];
        Result r = validate_impl(*m, v, loc, nullptr, false);  // probe: no errors recorded, no values built
        if (r.ok) {
          bool model_like = m->kind == K::TypedDict || m->kind == K::Model;
          if (!model_like && best < 0) {  // exact scalar match, no fields-set data: returned immediately
            if (r.coerced && build) r = validate_impl(*m, v, loc, nullptr, true);  // build the probed value
            r.member = static_cast<int>(i);
            return r;
          }
          if (r.fields_set > best_n) {
            best = static_cast<int>(i);
            best_n = r.fields_set;
            best_r = std::move(r);
          }
        }
      }
      if (best >= 0) {
        if (best_r.coerced && build) {  // the winner's value: validate it once more, building it
          int fs = best_r.fields_set;
          best_r = validate_impl(*s.members[best], v, loc, nullptr, true);
          best_r.fields_set = fs;
        }
        best_r.member = best;
        return best_r;
      }
      if (errs)  // every member failed: their errors, in member order, under the member labels
        for (const Schema* m : s.members) validate_impl(*m, v, with(loc, Json(m->label())), errs);
      return fail();
    }
    case K::Nullable: {
      if (v.is_null()) return okv(v);
      return validate_impl(*s.item, v, loc, errs, build);
    }
  }
  return fail();
}

Result validate(const Schema& s, const Json& v, const JsonArray& loc, Errors& errs) {
  return validate_impl(s, v, loc, &errs);
}

Result validate_consume(const Schema& s, Json& v, const JsonArray& loc, Errors& errs) {
  struct Guard {
    Guard() { g_move = true; }
    ~Guard() { g_move = false; }
  } guard;
  if (s.kind != K::Union) g_move = false;  // only union-winner rebuilds are failure-free
  return validate_impl(s, v, loc, &errs);
}

// ------------------------------------------------------------------------------------------- formatting
namespace {

const char* const kInternal[] = {"function-wrap", "function-after", "function-before", "function-plain",
                                 "json-or-python", "lax-or-strict", "chain", "default", "nullable", "tagged-union",
                                 "union", "call", "arguments", "is-instance", "is-subclass", "callable", "str",
                                 "int", "float", "bool", "bytes", "bytearray", "list", "tuple", "dict", "set",
                                 "frozenset", "complex", "none", "nonetype"};

bool internal_seg(const std::string& s) {
  for (char c : s)
    if (c == '[' || c == ']' || c == '{' || c == '}' || c == '(' || c == ')') return true;
  std::string low;
  for (char c : s) low += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  for (const char* m : kInternal)
    if (low == m) return true;
  return false;
}

std::string seg_str(const Json& p) { return p.is_string() ? p.as_string() : py_str(p); }

// Python s[:n] on code points
std::string py_prefix(std::string_view s, size_t n) {
  size_t i = 0, k = 0;
  while (i < s.size() && k < n) {
    ++i;
    while (i < s.size() && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) ++i;
    ++k;
  }
  return std::string(s.substr(0, i));
}
std::string py_suffix(std::string_view s, size_t n) {  // s[-n:]
  size_t len = py_len(s);
  if (n >= len) return std::string(s);
  size_t skip = len - n, i = 0, k = 0;
  while (i < s.size() && k < skip) {
    ++i;
    while (i < s.size() && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) ++i;
    ++k;
  }
  return std::string(s.substr(i));
}

std::string summarize_input(const Json& v) {
  if (v.is_string()) {
    if (py_len(v.as_string()) > 200) return py_prefix(v.as_string(), 200) + "...[truncated]";
    return v.as_string();
  }
  if (v.is_array()) return "<list of " + std::to_string(v.size()) + " items>";
  if (v.is_object()) return "<dict of " + std::to_string(v.size()) + " items>";
  std::string t = py_str(v);
  if (py_len(t) > 200) return py_prefix(t, 200) + "...[truncated]";
  return t;
}

}  // namespace

std::string clean_loc(const JsonArray& loc) {
  std::string out;
  bool any = false;
  for (const Json& p : loc) {
    std::string s = seg_str(p);
    if (internal_seg(s)) continue;
    if (any) out += '.';
    out += s;
    any = true;
  }
  if (any) return out;
  for (size_t i = 0; i < loc.size(); ++i) {
    if (i) out += '.';
    out += seg_str(loc[i]);
  }
  return out;
}

std::string format_request_errors(const Errors& errs, std::string* param_out) {
  if (errs.empty()) {
    if (param_out) *param_out = "";
    return "Validation error";
  }
  if (param_out) *param_out = clean_loc(errs[0].loc);
  size_t count = errs.size();
  const char* label = count == 1 ? "error" : "errors";
  std::string msg = std::to_string(count) + " validation " + label + ":\n";
  size_t reported = std::min<size_t>(count, 10);
  for (size_t i = 0; i < reported; ++i) {
    const Error& e = errs[i];
    std::string text = "{'type': ";
    py_repr_str_append(text, e.type);
    text += ", 'loc': ";
    py_repr_str_append(text, clean_loc(e.loc));
    text += ", 'msg': ";
    py_repr_str_append(text, e.msg);
    text += ", 'input': ";
    py_repr_str_append(text, summarize_input(e.input));
    if (!e.ctx.is_null()) {
      text += ", 'ctx': ";
      py_repr_append(text, e.ctx);
    }
    text += "}";
    if (py_len(text) > 1000) text = py_prefix(text, 1000) + "...[truncated]";
    msg += "  " + text + "\n";
  }
  if (count > reported) msg += "  ...and " + std::to_string(count - reported) + " more " + label + "\n";
  while (!msg.empty() && (msg.back() == '\n' || msg.back() == ' ')) msg.pop_back();
  return msg;
}

std::string truncated_repr(const Json& v) {
  std::string r = py_repr(v);
  if (py_len(r) > 50) return py_prefix(r, 25) + "..." + py_suffix(r, 24);
  return r;
}

std::string format_native_errors(const std::string& title, const Errors& errs) {
  std::string out = std::to_string(errs.size()) + " validation error" + (errs.size() == 1 ? "" : "s") + " for " + title;
  for (const Error& e : errs) {
    out += "\n";
    if (!e.loc.empty()) {
      for (size_t i = 0; i < e.loc.size(); ++i) {
        if (i) out += '.';
        out += seg_str(e.loc[i]);
      }
      out += "\n";
    }
    out += "  " + e.msg + " [type=" + e.type + ", input_value=" + truncated_repr(e.input) +
           ", input_type=" + py_type_name(e.input) + "]\n    For further information visit https://errors.pydantic.dev/2.14/v/" +
           e.type;
  }
  return out;
}

// ------------------------------------------------------------------------------------------- schemas
Schema* Builder::mk(K k, std::string name) {
  pool.emplace_back();
  pool.back().kind = k;
  pool.back().name = std::move(name);
  return &pool.back();
}
const Schema* Builder::lit(std::initializer_list<const char*> v) {
  Schema* s = mk(K::Literal);
  for (const char* x : v) s->lits.emplace_back(x);
  return s;
}
const Schema* Builder::wrap(K k, const Schema* item) {
  Schema* s = mk(k);
  s->item = item;
  return s;
}
const Schema* Builder::uni(std::initializer_list<const Schema*> m) {
  Schema* s = mk(K::Union);
  s->members = m;
  return s;
}
const Schema* Builder::td(const char* name, std::initializer_list<Field> f, K k) {
  Schema* s = mk(k, name);
  s->fields = f;
  return s;
}
const Schema* Builder::int_range(const Json* ge, const Json* le) {
  Schema* s = mk(K::Int);
  if (ge) { s->has_ge = true; s->ge = *ge; }
  if (le) { s->has_le = true; s->le = *le; }
  return s;
}

namespace {

Schemas build() {
  static Builder b;
  const Schema* str = b.mk(K::Str);
  const Schema* boolean = b.mk(K::Bool);
  const Schema* any = b.mk(K::Any);
  const Schema* str_opt = b.wrap(K::Nullable, str);
  const Schema* bool_opt = b.wrap(K::Nullable, boolean);
  const Schema* dict_any = b.mk(K::Dict);
  const Schema* pcb = b.td("PromptCacheBreakpoint", {{"mode", b.lit({"explicit"}), true}});

  const Schema* text = b.td("ChatCompletionContentPartTextParam",
                            {{"text", str, true}, {"type", b.lit({"text"}), true}, {"prompt_cache_breakpoint", pcb, false}});
  const Schema* image_url = b.td("ImageURL", {{"url", str, true}, {"detail", b.lit({"auto", "low", "high", "original"}), false}});
  const Schema* image = b.td("ChatCompletionContentPartImageParam", {{"image_url", image_url, true},
                                                                     {"type", b.lit({"image_url"}), true},
                                                                     {"prompt_cache_breakpoint", pcb, false}});
  const Schema* input_audio_v = b.td("InputAudio", {{"data", str, true}, {"format", b.lit({"wav", "mp3"}), true}});
  const Schema* input_audio = b.td("ChatCompletionContentPartInputAudioParam",
                                   {{"input_audio", input_audio_v, true}, {"type", b.lit({"input_audio"}), true},
                                    {"prompt_cache_breakpoint", pcb, false}});
  const Schema* filefile = b.td("FileFile", {{"file_data", str, false}, {"file_id", str, false}, {"filename", str, false}});
  const Schema* file = b.td("File", {{"file", filefile, true}, {"type", b.lit({"file"}), true},
                                     {"prompt_cache_breakpoint", pcb, false}});
  const Schema* audio_url_v = b.td("AudioURL", {{"url", str, true}});
  const Schema* audio_p = b.td("ChatCompletionContentPartAudioParam", {{"audio_url", audio_url_v, true},
                                                                       {"type", b.lit({"audio_url"}), true}});
  const Schema* video_url_v = b.td("VideoURL", {{"url", str, true}});
  const Schema* video_p = b.td("ChatCompletionContentPartVideoParam", {{"video_url", video_url_v, true},
                                                                       {"type", b.lit({"video_url"}), true}});
  const Schema* refusal = b.td("ChatCompletionContentPartRefusalParam",
                               {{"refusal", str, true}, {"type", b.lit({"refusal"}), true}});
  // PILImage is a pydantic model holding a PIL.Image: no JSON value validates, null does.
  const Schema* pil_model = b.td("PILImage", {{"image_pil", b.mk(K::Model, "Image"), true}}, K::Model);
  const Schema* pil = b.td("CustomChatCompletionContentPILImageParam",
                           {{"image_pil", b.wrap(K::Nullable, pil_model), false}, {"uuid", str_opt, false}});
  const Schema* simple_image = b.td("CustomChatCompletionContentSimpleImageParam",
                                    {{"image_url", str_opt, false}, {"uuid", str_opt, false}});
  const Schema* num = b.uni({b.mk(K::Int), b.mk(K::Float)});
  const Schema* embeds_v = b.wrap(
      K::Nullable, b.uni({str, b.wrap(K::Dict, b.uni({str, b.wrap(K::List, num)}))}));
  const Schema* image_embeds = b.td("ChatCompletionContentPartImageEmbedsParam",
                                    {{"image_embeds", embeds_v, false}, {"type", b.lit({"image_embeds"}), true},
                                     {"uuid", str_opt, false}});
  const Schema* audio_embeds = b.td("ChatCompletionContentPartAudioEmbedsParam",
                                    {{"audio_embeds", embeds_v, false}, {"type", b.lit({"audio_embeds"}), true},
                                     {"uuid", str_opt, false}});
  const Schema* video_embeds = b.td("ChatCompletionContentPartVideoEmbedsParam",
                                    {{"video_embeds", embeds_v, false}, {"type", b.lit({"video_embeds"}), true},
                                     {"uuid", str_opt, false}});
  const Schema* prompt_embeds = b.td("ChatCompletionContentPartPromptEmbedsParam",
                                     {{"data", str, true}, {"type", b.lit({"prompt_embeds"}), true}});
  const Schema* simple_audio = b.td("CustomChatCompletionContentSimpleAudioParam", {{"audio_url", str_opt, false}});
  const Schema* simple_video = b.td("CustomChatCompletionContentSimpleVideoParam",
                                    {{"video_url", str_opt, false}, {"uuid", str_opt, false}});
  const Schema* tool_ref = b.td("CustomChatCompletionContentToolReferenceParam",
                                {{"name", str, false}, {"type", b.lit({"tool_reference"}), false}});
  const Schema* think = b.td("CustomThinkCompletionContentParam",
                             {{"thinking", str, true}, {"closed", boolean, false}, {"type", b.lit({"thinking"}), true}});
  const Schema* custom_part = b.uni({text, image, input_audio, file, audio_p, video_p, refusal, pil, simple_image,
                                     image_embeds, audio_embeds, video_embeds, prompt_embeds, simple_audio,
                                     simple_video, tool_ref, str, think});

  const Schema* fn = b.td("Function", {{"arguments", str, true}, {"name", str, true}});
  const Schema* fn_call = b.td("ChatCompletionMessageFunctionToolCallParam",
                               {{"id", str, true}, {"function", fn, true}, {"type", b.lit({"function"}), true}});
  const Schema* custom_v = b.td("Custom", {{"input", str, true}, {"name", str, true}});
  const Schema* custom_call = b.td("ChatCompletionMessageCustomToolCallParam",
                                   {{"id", str, true}, {"custom", custom_v, true}, {"type", b.lit({"custom"}), true}});
  const Schema* fn_def = b.td("FunctionDefinition", {{"name", str, true}, {"description", str, false},
                                                     {"parameters", dict_any, false}, {"strict", bool_opt, false}});
  const Schema* fn_tool = b.td("ChatCompletionFunctionToolParam",
                               {{"function", fn_def, true}, {"type", b.lit({"function"}), true}});
  const Schema* audio_ref = b.td("Audio", {{"id", str, true}});
  const Schema* fcall = b.td("FunctionCall", {{"arguments", str, true}, {"name", str, true}});

  const Schema* user_parts = b.uni({text, image, input_audio, file});
  const Schema* asst_parts = b.uni({text, refusal});
  const Schema* tool_calls_u = b.uni({fn_call, custom_call});

  const Schema* dev = b.td("ChatCompletionDeveloperMessageParam",
                           {{"content", b.uni({str, b.wrap(K::Iterable, text)}), true},
                            {"role", b.lit({"developer"}), true}, {"name", str, false}});
  const Schema* sys = b.td("ChatCompletionSystemMessageParam",
                           {{"content", b.uni({str, b.wrap(K::Iterable, text)}), true},
                            {"role", b.lit({"system"}), true}, {"name", str, false}});
  const Schema* user = b.td("ChatCompletionUserMessageParam",
                            {{"content", b.uni({str, b.wrap(K::Iterable, user_parts)}), true},
                             {"role", b.lit({"user"}), true}, {"name", str, false}});
  const Schema* asst = b.td(
      "ChatCompletionAssistantMessageParam",
      {{"role", b.lit({"assistant"}), true}, {"audio", b.wrap(K::Nullable, audio_ref), false},
       {"content", b.wrap(K::Nullable, b.uni({str, b.wrap(K::Iterable, asst_parts)})), false},
       {"function_call", b.wrap(K::Nullable, fcall), false}, {"name", str, false}, {"refusal", str_opt, false},
       {"tool_calls", b.wrap(K::Iterable, tool_calls_u), false}});
  const Schema* tool = b.td("ChatCompletionToolMessageParam",
                            {{"content", b.uni({str, b.wrap(K::Iterable, text)}), true},
                             {"role", b.lit({"tool"}), true}, {"tool_call_id", str, true}});
  const Schema* func = b.td("ChatCompletionFunctionMessageParam", {{"content", str_opt, true}, {"name", str, true},
                                                                   {"role", b.lit({"function"}), true}});
  const Schema* custom = b.td("CustomChatCompletionMessageParam",
                              {{"role", str, true},
                               {"content", b.uni({str, b.wrap(K::List, custom_part)}), false},
                               {"name", str, false},
                               {"tool_call_id", str_opt, false},
                               {"tool_calls", b.wrap(K::Nullable, b.wrap(K::List, fn_call)), false},
                               {"reasoning", str_opt, false},
                               {"tools", b.wrap(K::Nullable, b.wrap(K::List, fn_tool)), false},
                               {"task", str_opt, false}});
  // openai_harmony.Message (pydantic model): author: Author (required), content: List[Content], ...
  const Schema* harmony = b.td("Message", {{"author", any, true}, {"content", b.wrap(K::List, any), false},
                                           {"channel", str_opt, false}, {"recipient", str_opt, false},
                                           {"content_type", str_opt, false}},
                               K::Model);
  const Schema* msg_union = b.uni({dev, sys, user, asst, tool, func, custom, harmony});

  const Schema* resp_image = b.td("ResponseInputImageParam",
                                  {{"detail", b.lit({"low", "high", "auto", "original"}), true},
                                   {"type", b.lit({"input_image"}), true}, {"file_id", str_opt, false},
                                   {"image_url", str_opt, false}});

  // vLLM pydantic models (extra="allow")
  const Schema* vfn_def = b.td("FunctionDefinition", {{"name", str, true}, {"description", str_opt, false},
                                                      {"parameters", b.wrap(K::Nullable, dict_any), false},
                                                      {"strict", bool_opt, false}, {"defer_loading", bool_opt, false}},
                               K::Model);
  const Schema* tools_param = b.td("ChatCompletionToolsParam", {{"type", b.lit({"function"}), false},
                                                                {"function", vfn_def, true},
                                                                {"defer_loading", bool_opt, false}},
                                   K::Model);
  const Schema* named_fn = b.td("ChatCompletionNamedFunction", {{"name", str, true}}, K::Model);
  const Schema* named = b.td("ChatCompletionNamedToolChoiceParam",
                             {{"function", named_fn, true}, {"type", b.lit({"function"}), false}}, K::Model);
  for (const Schema* m : {vfn_def, tools_param, named_fn, named}) const_cast<Schema*>(m)->vllm_model = true;

  // Responses input dict items (openai types/responses): the members a plain message / tool-output dict resolves to.
  const Schema* rpcb = b.td("PromptCacheBreakpoint", {{"mode", b.lit({"explicit"}), true}});
  const Schema* in_text = b.td("ResponseInputTextParam", {{"text", str, true}, {"type", b.lit({"input_text"}), true},
                                                         {"prompt_cache_breakpoint", rpcb, false}});
  const Schema* in_image = b.td("ResponseInputImageParam",
                                {{"detail", b.lit({"low", "high", "auto", "original"}), true},
                                 {"type", b.lit({"input_image"}), true}, {"file_id", str_opt, false},
                                 {"image_url", str_opt, false}, {"prompt_cache_breakpoint", rpcb, false}});
  const Schema* in_file = b.td("ResponseInputFileParam",
                               {{"type", b.lit({"input_file"}), true}, {"detail", b.lit({"auto", "low", "high"}), false},
                                {"file_data", str, false}, {"file_id", str_opt, false}, {"file_url", str, false},
                                {"filename", str, false}, {"prompt_cache_breakpoint", rpcb, false}});
  const Schema* in_parts = b.wrap(K::List, b.uni({in_text, in_image, in_file}));
  const Schema* easy = b.td("EasyInputMessageParam",
                            {{"content", b.uni({str, in_parts}), true},
                             {"role", b.lit({"user", "assistant", "system", "developer"}), true},
                             {"phase", b.wrap(K::Nullable, b.lit({"commentary", "final_answer"})), false},
                             {"type", b.lit({"message"}), false}});
  const Schema* rmsg = b.td("Message", {{"content", in_parts, true}, {"role", b.lit({"user", "system", "developer"}), true},
                                        {"status", b.lit({"in_progress", "completed", "incomplete"}), false},
                                        {"type", b.lit({"message"}), false}});
  const Schema* fco = b.td("FunctionCallOutput",
                           {{"output", b.uni({str, in_parts}), true}, {"type", b.lit({"function_call_output"}), true},
                            {"id", str_opt, false}, {"call_id", str_opt, false}, {"caller", any, false},
                            {"name", str_opt, false}, {"namespace", str_opt, false},
                            {"status", b.wrap(K::Nullable, b.lit({"in_progress", "completed", "incomplete"})), false}});

  Schemas s;
  s.responses_dict_item = b.uni({easy, rmsg, fco});
  s.message_union = msg_union;
  s.openai_user_parts = user_parts;
  s.openai_assistant_parts = asst_parts;
  s.text_part = text;
  s.tool_call_union = tool_calls_u;
  s.image_part = image;
  s.responses_input_image = resp_image;
  s.audio_part = audio_p;
  s.video_part = video_p;
  s.tools_param = tools_param;
  s.named_tool_choice = named;
  return s;
}

}  // namespace

const Schemas& schemas() {
  static const Schemas s = build();
  return s;
}

}  // namespace fe::render::pyd
