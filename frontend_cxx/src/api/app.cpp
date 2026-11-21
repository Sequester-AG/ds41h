#include "api/app.h"

#include <chrono>
#include <cstring>
#include <random>

#include "api/jw.h"
#include "render/render.h"

namespace fe::api {

std::string random_uuid_hex() {
  thread_local std::mt19937_64 rng([] {
    std::random_device rd;
    uint64_t s = (uint64_t(rd()) << 32) ^ rd() ^
                 uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    return s;
  }());
  // low 64 bits of a uuid4: the variant bits 10xx lead
  uint64_t b = (rng() & 0x3fffffffffffffffULL) | 0x8000000000000000ULL;
  char buf[17];
  snprintf(buf, sizeof buf, "%016llx", (unsigned long long)b);
  return buf;
}

double now_unix() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

ApiError from_render_error(const fe::RenderError& e) {
  ApiError a;
  a.status = e.status;
  a.type = e.type;
  a.message = e.message;
  if (e.param.is_string()) a.param = e.param.as_string();
  return a;
}

App::App(Config cfg, Components comp, std::shared_ptr<GenBackend> backend)
    : cfg_(std::move(cfg)), comp_(std::move(comp)), backend_(std::move(backend)) {
  http_metrics_ = std::make_unique<HttpMetrics>();
  // init_app_state: `if args.tool_call_parser is not None: init_parser_metrics(model_name=served_model_name)`
  if (!cfg_.tool_parser.empty()) tool_parser_metrics_ = std::make_shared<ToolParserMetrics>(cfg_.model_name());
  store_ = std::make_unique<ResponsesStore>(cfg_.bounded_store, cfg_.store_max);
  for (auto& k : cfg_.api_keys) api_key_hashes_.push_back(http::sha256(k));
}

App::~App() = default;

std::function<void(bool, bool)> App::tool_parser_hook(ToolParserRequestType rt, fe::ToolChoice tc) const {
  if (!tool_parser_metrics_) return {};
  bool full_records = cfg_.enable_auto_tools || tc == fe::ToolChoice::None;
  return [m = tool_parser_metrics_, rt, full_records](bool streaming, bool tool_called) {
    if (streaming || full_records) m->record(streaming, tool_called, rt);
  };
}

void App::route_metrics(Ctx& c) {
  // Mount("/metrics", make_asgi_app(registry)): process collector, vllm:* (PrometheusStatLogger), http_*,
  // then vllm:tool_call_parser_invocations_* (registered last, by init_app_state)
  std::string body = http_metrics_->render_process();
  if (backend_) body += backend_->render_vllm_metrics();
  body += http_metrics_->render_http();
  if (tool_parser_metrics_) body += tool_parser_metrics_->render();
  // the length first: in one call, std::move(body) may run before body.size() (GCC x86_64 did: content-length 0)
  std::string length = std::to_string(body.size());
  c.resp->send(200, {{"content-type", "text/plain; version=1.0.0; charset=utf-8"}, {"content-length", std::move(length)}},
               std::move(body));
}

void App::send_json(Ctx& c, int status, std::string body, http::HeaderList extra) {
  http::HeaderList h;
  h.emplace_back("content-length", std::to_string(body.size()));
  h.emplace_back("content-type", "application/json");
  for (auto& e : extra) h.push_back(std::move(e));
  c.resp->send(status, std::move(h), std::move(body));
}

void App::send_error(Ctx& c, const ApiError& e) { send_json(c, e.status, error_body(e)); }

void App::check_model(const Json& body) {
  const Json* m = body.get("model");
  if (!m || m->is_null()) return;
  if (!m->is_string()) return;  // validated earlier
  const std::string& name = m->as_string();
  if (name.empty()) return;
  if (cfg_.is_base_model(name)) return;
  if (const char* skip = std::getenv("VLLM_SKIP_MODEL_NAME_VALIDATION"); skip && std::strcmp(skip, "1") == 0)
    return;
  std::string served;
  for (size_t i = 0; i < cfg_.served_names.size(); ++i) {
    if (i) served += ", ";
    served += cfg_.served_names[i];
  }
  throw ApiError(404, "NotFoundError",
                 "The model `" + name + "` does not exist. Valid aliases: " + served + ".", "model");
}

std::string App::base_request_id(Ctx& c, const Json* body_request_id) {
  if (const std::string* h = c.req->header("x-request-id")) return *h;
  if (body_request_id && body_request_id->is_string()) return body_request_id->as_string();
  return random_uuid_hex();
}

void App::preflight() {
  if (backend_ && backend_->dead())
    throw ApiError(500, "InternalServerError",
                   "EngineCore encountered an issue. See stack trace (above) for the root cause.");
}

Json App::parse_json_body(Ctx& c) {
  const std::string& body = c.req->body;
  const std::string* ct = c.req->header("content-type");
  Json parsed;
  bool have_json = false;
  if (!body.empty() && ct) {
    // email.message content-type: maintype "application", subtype "json" or "*+json"
    std::string t = *ct;
    size_t semi = t.find(';');
    std::string mt = t.substr(0, semi);
    while (!mt.empty() && (mt.back() == ' ' || mt.back() == '\t')) mt.pop_back();
    size_t b = 0;
    while (b < mt.size() && (mt[b] == ' ' || mt[b] == '\t')) ++b;
    mt = mt.substr(b);
    for (auto& ch : mt) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    size_t sl = mt.find('/');
    if (sl != std::string::npos && mt.substr(0, sl) == "application") {
      std::string sub = mt.substr(sl + 1);
      if (sub == "json" || (sub.size() >= 5 && sub.compare(sub.size() - 5, 5, "+json") == 0)) {
        try {
          parsed = json_parse(body);
          have_json = true;
        } catch (const JsonError& e) {
          if (e.kind == JsonError::Kind::Decode) {
            PydErr pe;
            pe.type = "json_invalid";
            pe.loc = {"body", std::to_string(e.pos)};
            pe.msg = "JSON decode error";
            pe.input_repr = "'<dict of 0 items>'";
            pe.ctx_repr = "{'error': " + py_repr_str(e.msg) + "}";
            throw validation_error({pe});
          }
          // UnicodeDecodeError / RecursionError / ValueError -> FastAPI HTTPException 400
          throw ApiError(-400, "", "There was an error parsing the body");
        }
      }
    }
  }
  // validate_json_request dependency
  std::string ctl = ct ? *ct : "";
  for (auto& ch : ctl) ch = char(std::tolower(static_cast<unsigned char>(ch)));
  if (ctl.substr(0, ctl.find(';')) != "application/json")
    throw validation_error_plain("Unsupported Media Type: Only 'application/json' is allowed");
  if (body.empty()) {
    PydErr pe;
    pe.type = "missing";
    pe.loc = {"body"};
    pe.msg = "Field required";
    pe.input_repr = summarize_input_repr_none();
    throw validation_error({pe});
  }
  if (!have_json) return Json();
  if (!parsed.is_object()) {
    PydErr pe;
    pe.type = "model_attributes_type";
    pe.loc = {"body"};
    pe.msg = "Input should be a valid dictionary or object to extract fields from";
    pe.input_repr = summarize_input_repr(parsed);
    throw validation_error({pe});
  }
  return parsed;
}

namespace {
void send_detail(http::Response& r, int status, std::string_view detail, http::HeaderList extra = {}) {
  JW w(true, false);
  w.ob().kstr("detail", detail).oe();
  std::string body = w.take();
  http::HeaderList h;
  for (auto& e : extra) h.push_back(std::move(e));
  h.emplace_back("content-length", std::to_string(body.size()));
  h.emplace_back("content-type", "application/json");
  r.send(status, std::move(h), std::move(body));
}

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }
}  // namespace

void App::route_models(Ctx& c) {
  // ModelList(data=[ModelCard(id, max_model_len, root, permission=[ModelPermission()])...])
  JW w(true, false);
  w.ob().kstr("object", "list").k("data").ab();
  int64_t now = int64_t(now_unix());
  for (auto& name : cfg_.served_names) {
    w.ob();
    w.kstr("id", name).kstr("object", "model").ki("created", now).kstr("owned_by", "vllm");
    w.kstr("root", cfg_.model_path).knull("parent").ki("max_model_len", backend_ && backend_->max_model_len() > 0 ? backend_->max_model_len() : cfg_.max_model_len);
    w.k("permission").ab().ob();
    w.kstr("id", "modelperm-" + random_uuid_hex()).kstr("object", "model_permission").ki("created", now);
    w.kb("allow_create_engine", false).kb("allow_sampling", true).kb("allow_logprobs", true);
    w.kb("allow_search_indices", false).kb("allow_view", true).kb("allow_fine_tuning", false);
    w.kstr("organization", "*").knull("group").kb("is_blocking", false);
    w.oe().ae();
    w.oe();
  }
  w.ae().oe();
  send_json(c, 200, w.take());
}

void App::handle(std::shared_ptr<http::Request> req, std::shared_ptr<http::Response> resp) {
  // ---- EdgeGuard parity
  if (!cfg_.route_allow.empty() && !route_allowed(cfg_.route_allow, req->path)) {
    resp->send(404, {{"content-type", "application/json"}, {"content-length", std::to_string(kEdgeGuardBody.size())}},
               std::string(kEdgeGuardBody));
    return;
  }
  Ctx c{req, resp, req->path};
  auto t0 = std::chrono::steady_clock::now();
  // strip it for routing and auth (Starlette get_route_path).
  if (!cfg_.root_path.empty() && starts_with(c.route_path, cfg_.root_path)) {
    c.route_path = c.route_path.substr(cfg_.root_path.size());
    if (c.route_path.empty()) c.route_path = "/";
  }
  const std::string& m = req->method;
  const std::string& p = c.route_path;
  // ---- AuthenticationMiddleware (patched: every path except /health; OPTIONS skipped) ----
  if (!api_key_hashes_.empty() && m != "OPTIONS" && p != "/health") {
    bool ok = false;
    if (const std::string* auth = req->header("authorization"); auth && !auth->empty()) {
      size_t sp = auth->find(' ');
      std::string scheme = auth->substr(0, sp);
      for (auto& ch : scheme) ch = char(std::tolower(static_cast<unsigned char>(ch)));
      std::string param = sp == std::string::npos ? std::string() : auth->substr(sp + 1);
      if (scheme == "bearer") {
        std::string h = http::sha256(param);
        bool match = false;
        for (auto& k : api_key_hashes_) match |= http::constant_time_eq(h, k);
        ok = match;
      }
    }
    if (!ok) {
      std::string body = "{\"error\":\"Unauthorized\"}";
      resp->send(401, {{"content-length", std::to_string(body.size())}, {"content-type", "application/json"}},
                 body);
      return;
    }
  }
  std::string handler_label;  // for http_* metrics
  int status_for_metrics = 0;
  try {
    // ---- routes ----
    auto method_not_allowed = [&](const char* allow) {
      send_detail(*resp, 405, "Method Not Allowed", {{"allow", allow}});
    };
    if (p == "/health") {
      if (m != "GET") return method_not_allowed("GET");
      handler_label = "/health";
      int st = engine_dead() ? 503 : 200;
      resp->send(st, {{"content-length", "0"}}, "");
      return;
    }
    if (p == "/v1/chat/completions") {
      if (m != "POST") return method_not_allowed("POST");
      handler_label = p;
      route_chat(c);
    } else if (p == "/v1/completions") {
      if (m != "POST") return method_not_allowed("POST");
      handler_label = p;
      route_completions(c);
    } else if (p == "/v1/responses") {
      if (m != "POST") return method_not_allowed("POST");
      handler_label = p;
      route_responses_create(c);
    } else if (starts_with(p, "/v1/responses/")) {
      std::string rest = p.substr(14);
      size_t sl = rest.find('/');
      if (sl == std::string::npos && !rest.empty()) {
        if (m != "GET") return method_not_allowed("GET");
        handler_label = "/v1/responses/{response_id}";
        route_responses_get(c, rest);
      } else if (sl != std::string::npos && rest.substr(sl) == "/cancel" && sl > 0) {
        if (m != "POST") return method_not_allowed("POST");
        handler_label = "/v1/responses/{response_id}/cancel";
        route_responses_cancel(c, rest.substr(0, sl));
      } else {
        send_detail(*resp, 404, "Not Found");
        status_for_metrics = 404;
      }
    } else if (p == "/tokenize") {
      if (m != "POST") return method_not_allowed("POST");
      handler_label = p;
      route_tokenize(c);
    } else if (p == "/detokenize") {
      if (m != "POST") return method_not_allowed("POST");
      handler_label = p;
      route_detokenize(c);
    } else if (p == "/v1/models") {
      if (m != "GET") return method_not_allowed("GET");
      handler_label = p;
      route_models(c);
    } else if (p == "/metrics" || starts_with(p, "/metrics/")) {
      route_metrics(c);
      return;
    } else if (p == "/version") {
      if (m != "GET") return method_not_allowed("GET");
      std::string body = "{\"version\":\"" + cfg_.vllm_version + "\"}";
      send_json(c, 200, body);
    } else if (p == "/ping") {
      if (m != "GET" && m != "POST") return method_not_allowed("GET, POST");
      int st = engine_dead() ? 503 : 200;
      resp->send(st, {{"content-length", "0"}}, "");
    } else if (p == "/load") {
      if (m != "GET") return method_not_allowed("GET");
      send_json(c, 200, "{\"server_load\":0}");
    } else {
      send_detail(*resp, 404, "Not Found");
      status_for_metrics = 404;
    }
  } catch (const ApiError& e) {
    status_for_metrics = e.status;
    if (e.status == -400) {
      status_for_metrics = 400;
      send_detail(*resp, 400, e.message);
    } else if (!resp->started()) {
      send_error(c, e);
    }
  } catch (const std::exception& e) {
    // Exception handler safety net: 500 InternalServerError with str(exc).
    if (!resp->started()) send_error(c, ApiError(500, "InternalServerError", e.what()));
  }
  (void)status_for_metrics;
  // prometheus-fastapi-instrumentator (inside the auth middleware; excluded handlers skipped).
  // Recorded when the handler returns (streams: at response start).
  static const char* excluded[] = {"/metrics", "/health", "/load", "/ping", "/version", "/server_info"};
  bool skip = false;
  for (auto* x : excluded)
    if (handler_label == x || (handler_label.empty() && p == x)) skip = true;
  if (!skip) {
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    size_t req_bytes = 0;
    if (const std::string* cl = req->header("content-length")) req_bytes = size_t(std::atoll(cl->c_str()));
    int st = status_for_metrics ? status_for_metrics : (handler_label.empty() ? 404 : 200);
    http_metrics_->on_http(handler_label.empty() ? "none" : handler_label, m, st, dt, req_bytes, 0);
  }
}

}  // namespace fe::api
