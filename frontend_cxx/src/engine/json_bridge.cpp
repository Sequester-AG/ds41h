#include "engine/json_bridge.h"
#ifdef FE_HAVE_COMMON
#include <cstdlib>

namespace fe::engine {

mp::Value to_mp(const fe::Json& j) {
  using T = fe::Json::Type;
  switch (j.type()) {
    case T::Null: return mp::Value::nil();
    case T::Bool: return mp::Value::from_bool(j.as_bool());
    case T::Int: return mp::Value::from_int(j.as_int());
    case T::BigInt: {
      const std::string& t = j.big_int_text();
      if (!t.empty() && t[0] != '-') {
        errno = 0;
        unsigned long long u = std::strtoull(t.c_str(), nullptr, 10);
        if (errno == 0) {
          mp::Value v;
          v.type = mp::Type::UInt;
          v.u = u;
          return v;
        }
      }
      return mp::Value::from_double(j.as_double());
    }
    case T::Float: return mp::Value::from_double(j.as_double());
    case T::String: return mp::Value::from_str(j.as_string());
    case T::Array: {
      mp::Value a = mp::Value::make_array();
      a.arr.reserve(j.size());
      for (auto& e : j.as_array()) a.arr.push_back(to_mp(e));
      return a;
    }
    case T::Object: {
      mp::Value m = mp::Value::make_map();
      m.kv.reserve(j.size());
      for (auto& [k, v] : j.as_object()) m.kv.emplace_back(mp::Value::from_str(k), to_mp(v));
      return m;
    }
  }
  return mp::Value::nil();
}

fe::Json to_json(const mp::Value& v) {
  switch (v.type) {
    case mp::Type::Nil: return fe::Json();
    case mp::Type::Bool: return fe::Json(v.b);
    case mp::Type::Int: return fe::Json(v.i);
    case mp::Type::UInt: return fe::Json(v.u);
    case mp::Type::Float: return fe::Json(v.f);
    case mp::Type::Str:
    case mp::Type::Bin:
    case mp::Type::Ext: return fe::Json(v.s);
    case mp::Type::Array: {
      fe::Json a = fe::Json::array();
      for (auto& e : v.arr) a.push_back(to_json(e));
      return a;
    }
    case mp::Type::Map: {
      fe::Json o = fe::Json::object();
      for (auto& [k, e] : v.kv) {
        std::string key = k.type == mp::Type::Str ? k.s
                          : k.is_int()            ? std::to_string(k.as_int())
                                                  : mp::to_debug_string(k);
        o.set(key, to_json(e));
      }
      return o;
    }
  }
  return fe::Json();
}

}  // namespace fe::engine
#endif
