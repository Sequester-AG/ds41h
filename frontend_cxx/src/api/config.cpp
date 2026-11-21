#include "api/config.h"

#include <cstdlib>

#include "http/http_server.h"

namespace fe::api {

bool Config::is_base_model(std::string_view name) const {
  for (auto& n : served_names)
    if (n == name) return true;
  return false;
}

namespace {
const char* env(const char* k) {
  const char* v = std::getenv(k);
  return (v && *v) ? v : nullptr;
}
bool env_flag(const char* k) {
  const char* v = std::getenv(k);
  if (!v) return false;
  std::string s(v);
  return s == "1" || s == "true" || s == "True" || s == "TRUE";
}
}  // namespace

std::string py_normpath(std::string_view p) {
  if (p.empty()) return ".";
  int initial = 0;
  if (p[0] == '/') {
    initial = 1;
    if (p.size() > 1 && p[1] == '/' && !(p.size() > 2 && p[2] == '/')) initial = 2;
  }
  std::vector<std::string_view> out;
  size_t i = 0;
  while (i <= p.size()) {
    size_t j = p.find('/', i);
    if (j == std::string_view::npos) j = p.size();
    std::string_view comp = p.substr(i, j - i);
    i = j + 1;
    if (comp.empty() || comp == ".") continue;
    if (comp != ".." || (!initial && out.empty()) || (!out.empty() && out.back() == ".."))
      out.push_back(comp);
    else if (!out.empty())
      out.pop_back();
  }
  std::string r(size_t(initial), '/');
  for (size_t k = 0; k < out.size(); ++k) {
    if (k) r += '/';
    r += out[k];
  }
  return r.empty() ? "." : r;
}

bool route_allowed(const std::vector<std::string>& allow, std::string_view path) {
  if (allow.empty()) return true;
  for (auto& e : allow) {
    if (!e.empty() && e.back() == '/') {
      if (path.substr(0, e.size()) != e) continue;
      std::string n = py_normpath(path);
      if (n == path || n + "/" == path) return true;
    } else if (path == e) {
      return true;
    }
  }
  return false;
}

Config config_from_args(const Json& a) {
  Config c;
  c.args = a;
  if (auto s = a.get_string("model_tag"))
    c.model_path = std::string(*s);
  else if (auto s2 = a.get_string("model"))
    c.model_path = std::string(*s2);
  if (auto t = a.get_string("tokenizer"))
    c.tokenizer_path = std::string(*t);
  else
    c.tokenizer_path = c.model_path;
  const Json& smn = a["served_model_name"];
  if (smn.is_array()) {
    for (auto& x : smn.as_array())
      if (x.is_string()) c.served_names.push_back(x.as_string());
  } else if (smn.is_string()) {
    c.served_names.push_back(smn.as_string());
  }
  if (c.served_names.empty()) c.served_names.push_back(c.model_path);
  if (auto m = a.get_int("max_model_len")) c.max_model_len = *m;
  const Json& ak = a["api_key"];
  if (ak.is_array()) {
    for (auto& x : ak.as_array())
      if (x.is_string() && !x.as_string().empty()) c.api_keys.push_back(x.as_string());
  } else if (ak.is_string() && !ak.as_string().empty()) {
    c.api_keys.push_back(ak.as_string());
  }
  // "--api-key option from CLI takes precedence over VLLM_API_KEY"
  if (c.api_keys.empty())
    if (const char* k = env("VLLM_API_KEY")) c.api_keys.push_back(k);
  if (auto r = a.get_string("root_path")) c.root_path = std::string(*r);
  if (auto r = a.get_string("response_role")) c.response_role = std::string(*r);
  c.enable_prompt_tokens_details = a.get_bool("enable_prompt_tokens_details").value_or(false);
  c.enable_force_include_usage = a.get_bool("enable_force_include_usage").value_or(false);
  c.enable_auto_tools = a.get_bool("enable_auto_tool_choice").value_or(false);
  if (auto t = a.get_string("tool_call_parser")) c.tool_parser = std::string(*t);
  if (auto t = a.get_string("reasoning_parser")) c.reasoning_parser = std::string(*t);
  c.return_tokens_as_token_ids = a.get_bool("return_tokens_as_token_ids").value_or(false);
  c.exclude_tools_when_tool_choice_none = a.get_bool("exclude_tools_when_tool_choice_none").value_or(false);
  if (const Json* d = a.get("default_chat_template_kwargs")) {
    if (d->is_object())
      c.default_chat_template_kwargs = *d;
    else if (d->is_string()) {
      try {
        c.default_chat_template_kwargs = json_parse(d->as_string());
      } catch (...) {
      }
    }
  }
  c.disable_log_stats = a.get_bool("disable_log_stats").value_or(false);
  if (auto v = a.get_number("sse_keep_alive_interval")) c.sse_keep_alive_interval = *v;
  if (auto v = a.get_number("http_timeout_keep_alive")) c.http_timeout_keep_alive = *v;
  else if (const char* e = env("VLLM_HTTP_TIMEOUT_KEEP_ALIVE")) c.http_timeout_keep_alive = std::atof(e);
  if (auto v = a.get_number("engine_ready_timeout_secs")) c.engine_ready_timeout_s = *v;
  else if (const char* e = env("VLLM_ENGINE_READY_TIMEOUT_S")) c.engine_ready_timeout_s = std::atof(e);
  // middleware list -> BoundedStore
  const Json& mw = a["middleware"];
  if (mw.is_array())
    for (auto& x : mw.as_array())
      if (x.is_string() && x.as_string().find("responses_store.BoundedStore") != std::string::npos)
        c.bounded_store = true;
  // route allowlist (EdgeGuard parity), see config.h
  {
    static const char* kEdgeGuardList = "/v1/,/health,/metrics,/version";
    bool mw_edge = false;
    if (mw.is_array())
      for (auto& x : mw.as_array())
        if (x.is_string() && x.as_string().find("edge_guard.EdgeGuard") != std::string::npos) mw_edge = true;
    std::string spec;
    if (const char* ra = std::getenv("FE_ROUTE_ALLOW")) {
      spec = ra;
      c.route_allow_source = "FE_ROUTE_ALLOW";
    } else if (mw_edge) {
      spec = "edgeguard";
      c.route_allow_source = "--middleware edge_guard.EdgeGuard";
    }
    if (spec == "edgeguard") spec = kEdgeGuardList;
    if (spec != "off") {
      size_t i = 0;
      while (i <= spec.size()) {
        size_t j = spec.find(',', i);
        if (j == std::string::npos) j = spec.size();
        std::string e = spec.substr(i, j - i);
        while (!e.empty() && (e.front() == ' ' || e.front() == '\t')) e.erase(0, 1);
        while (!e.empty() && (e.back() == ' ' || e.back() == '\t')) e.pop_back();
        if (!e.empty()) c.route_allow.push_back(e);
        i = j + 1;
      }
    }
    if (c.route_allow.empty()) c.route_allow_source.clear();
  }
  c.enable_store = env_flag("VLLM_ENABLE_RESPONSES_API_STORE");
  if (const char* e = env("PHYS_RESPONSES_STORE_MAX")) {
    long v = std::atol(e);
    if (v > 0) c.store_max = size_t(v);
  }
  // speculative config: dict or JSON string
  const Json* sc = a.get("speculative_config");
  Json scj;
  if (sc && sc->is_string()) {
    try {
      scj = json_parse(sc->as_string());
    } catch (...) {
    }
    sc = &scj;
  }
  if (sc && sc->is_object())
    if (auto n = sc->get_int("num_speculative_tokens")) c.num_speculative_tokens = int(*n);
  // system fingerprint: Python computes "vllm-<version>-<compute_hash()[:8]>" from the full
  // VllmConfig; we cannot. FE_SYSTEM_FINGERPRINT sets it (empty -> null).
  std::string mode = std::string(a.get_string("fingerprint_mode").value_or("full"));
  if (mode == "none") {
    c.system_fingerprint = std::nullopt;
  } else if (mode == "custom") {
    if (auto v = a.get_string("fingerprint_value")) c.system_fingerprint = std::string(*v);
  } else if (const char* fp = std::getenv("FE_SYSTEM_FINGERPRINT")) {
    if (*fp) c.system_fingerprint = std::string(fp);
  } else {
    // Same shape and length as vLLM's "vllm-<version>-<compute_hash()[:8]>", but hashed from the
    // args-json (VllmConfig.compute_hash() is not reproducible outside Python). See DEVIATIONS.md.
    // TLS settings are transport, not model configuration (Python's compute_hash() never sees them):
    // they stay out of the hash so that serving HTTPS does not change the fingerprint.
    Json fa = a;
    if (fa.is_object())
      for (const char* k : {"ssl_keyfile", "ssl_certfile", "ssl_ca_certs", "ssl_cert_reqs", "ssl_ciphers",
                            "enable_ssl_refresh"})
        fa.erase(k);
    std::string h = fe::http::sha256(dumps_py(fa, true));
    static const char* hx = "0123456789abcdef";
    std::string h8;
    for (int i = 0; i < 4; ++i) {
      h8 += hx[(static_cast<unsigned char>(h[i]) >> 4) & 15];
      h8 += hx[static_cast<unsigned char>(h[i]) & 15];
    }
    c.system_fingerprint = "vllm-" + c.vllm_version + "-" + h8;
  }
  return c;
}

}  // namespace fe::api
