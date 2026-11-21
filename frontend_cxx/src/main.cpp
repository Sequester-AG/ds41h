// frontend_cxx: C++ replacement of vLLM 0.32.0's API-server process.
//
// Launched by vLLM's RustFrontendProcessManager (VLLM_USE_RUST_FRONTEND=1,
// VLLM_RUST_FRONTEND_PATH=<this binary>), see tools/SEAM.md and $V/v1/utils.py:280-396:
//   frontend_cxx frontend --listen-fd N --input-listener-fd N --output-listener-fd N
//       --engine-start-index 0 --engine-count 1 --data-parallel-size 1
// [--grpc-listen-fd N] --args-json '{...}'
// Test extras (not passed by vLLM): --port P [--host H] binds the HTTP socket itself when
// --listen-fd is absent.
// Checks (not passed by vLLM): --check-config parses --args-json (or --args-json-file F, which keeps secrets out of
// argv) like a real start, prints "api keys: N" and exits 0 (2 on a bad config); no engine, no socket.
// The --args-json value (it can hold api_key, e.g. from --config endpoint_auth.yaml) is blanked in our own argv right
// after parsing, so /proc/<pid>/cmdline no longer shows it.
// HTTPS: vLLM's --ssl-keyfile / --ssl-certfile / --ssl-ca-certs / --ssl-cert-reqs / --ssl-ciphers reach
// us inside --args-json (ssl_keyfile, ...; jsonify_non_default_args). The same flags are also accepted
// directly on our command line (seam and --port mode) and override args-json. The key path is never logged.
//
// Lifecycle (= run_server_worker): wait for EngineCoreReadyResponse + get_supported_tasks,
// then serve HTTP. SIGTERM / SIGINT: stop accepting, abort in-flight engine requests, exit 0.
// Engine death: /health answers 503 and the watchdog exits (code 1) like terminate_if_errored.
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "api/app.h"
#include "api/backend_engine.h"
#include "common/json.h"
#include "engine/seam.h"
#include "http/http_server.h"

namespace {

std::atomic<int> g_signal{0};
void on_signal(int s) { g_signal.store(s); }

void log(const std::string& m) {
  fprintf(stderr, "[frontend_cxx] %s\n", m.c_str());
  fflush(stderr);
}

std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool exists(const std::string& p) { return access(p.c_str(), R_OK) == 0; }

std::vector<int64_t> encode_hook(const std::string& text, void* ctx) {
  auto* tok = static_cast<fe::Tokenizer*>(ctx);
  auto ids = tok->encode(text);
  return std::vector<int64_t>(ids.begin(), ids.end());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::strcmp(argv[1], "frontend") != 0) {
    fprintf(stderr, "usage: %s frontend --listen-fd N --input-listener-fd N --output-listener-fd N "
                    "--engine-start-index I --engine-count C --data-parallel-size D --args-json JSON\n", argv[0]);
    return 2;
  }
  // test-only flags, stripped before the seam parser sees argv
  int port = -1;
  std::string host = "127.0.0.1";
  std::vector<char*> av;
  // --ssl-* flags (vLLM spelling), stripped like the test flags; they override args-json
  static const char* const kSslFlags[] = {"--ssl-keyfile", "--ssl-certfile", "--ssl-ca-certs", "--ssl-cert-reqs",
                                          "--ssl-ciphers"};
  std::vector<std::pair<std::string, std::string>> ssl_cli;  // (flag, value)
  bool check_config = false;
  std::string args_json_file;
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    bool ssl_flag = false;
    for (const char* f : kSslFlags) {
      std::string fs(f);
      if (a == fs && i + 1 < argc) {
        ssl_cli.emplace_back(fs, argv[++i]);
        ssl_flag = true;
      } else if (a.rfind(fs + "=", 0) == 0) {
        ssl_cli.emplace_back(fs, a.substr(fs.size() + 1));
        ssl_flag = true;
      }
      if (ssl_flag) break;
    }
    if (ssl_flag) continue;
    if (a == "--check-config") {
      check_config = true;
    } else if (a == "--args-json-file" && i + 1 < argc) {
      args_json_file = argv[++i];
    } else if (a == "--port" && i + 1 < argc) {
      port = std::atoi(argv[++i]);
    } else if (a.rfind("--port=", 0) == 0) {
      port = std::atoi(a.c_str() + 7);
    } else if (a == "--host" && i + 1 < argc) {
      host = argv[++i];
    } else {
      av.push_back(argv[i]);
    }
  }
  fe::engine::FrontendArgs fa;
  bool have_listen = false;
  std::string aj_arg;  // --args-json=<file content> (storage outlives the parse)
  std::vector<std::string> placeholders = {"--listen-fd=-1", "--input-listener-fd=-1", "--output-listener-fd=-1"};
  try {
    // the seam parser requires --listen-fd; supply a placeholder in --port mode (and all fds in --check-config)
    std::vector<char*> av2 = av;
    auto has = [&](const char* f) {
      for (auto* a : av2)
        if (std::string(a).rfind(f, 0) == 0) return true;
      return false;
    };
    have_listen = has("--listen-fd");
    if (!have_listen) av2.push_back(placeholders[0].data());
    if (check_config) {
      if (!has("--input-listener-fd")) av2.push_back(placeholders[1].data());
      if (!has("--output-listener-fd")) av2.push_back(placeholders[2].data());
    }
    if (!args_json_file.empty()) {
      if (!exists(args_json_file)) {
        log("--args-json-file: cannot read " + args_json_file);
        return 2;
      }
      aj_arg = "--args-json=" + read_file(args_json_file);
      av2.push_back(aj_arg.data());
    }
    fa = fe::engine::parse_frontend_args(int(av2.size()), av2.data());
  } catch (const std::exception& e) {
    log(std::string("bad command line: ") + e.what());
    return 2;
  }
  // Blank the --args-json value in our argv (it may carry api keys): /proc/<pid>/cmdline shows "<args-json>" now.
  for (int i = 1; i < argc; ++i) {
    char* v = nullptr;
    if (std::strcmp(argv[i], "--args-json") == 0 && i + 1 < argc)
      v = argv[i + 1];
    else if (std::strncmp(argv[i], "--args-json=", 12) == 0)
      v = argv[i] + 12;
    if (!v) continue;
    size_t n = std::strlen(v);
    static const char kMark[] = "<args-json>";
    size_t m = std::min(n, sizeof kMark - 1);
    std::memcpy(v, kMark, m);
    std::memset(v + m, 0, n - m);
  }
  if (check_config) have_listen = true;
  if (!have_listen && port < 0) {
    log("need --listen-fd (vLLM seam) or --port (tests)");
    return 2;
  }
  fe::Json args;
  try {
    args = fe::json_parse(fa.args_json);
  } catch (const std::exception& e) {
    log(std::string("bad --args-json: ") + e.what());
    return 2;
  }
  fe::api::Config cfg = fe::api::config_from_args(args);
  if (!check_config)
    log("model " + cfg.model_path + ", served name " + cfg.model_name() + ", api keys: " +
        std::to_string(cfg.api_keys.size()));
  if (!cfg.route_allow.empty() && !check_config) {
    std::string l;
    for (auto& e : cfg.route_allow) l += (l.empty() ? "" : ",") + e;
    log("route allowlist (EdgeGuard parity, from " + cfg.route_allow_source + "): " + l + "; other paths -> 404");
  }

  // ---- TLS options (vLLM's uvicorn: ssl_keyfile, ssl_certfile, ssl_ca_certs, ssl_cert_reqs, ssl_ciphers)
  fe::http::ServerOptions tls;
  if (auto v = args.get_string("ssl_certfile")) tls.tls_certfile = std::string(*v);
  if (auto v = args.get_string("ssl_keyfile")) tls.tls_keyfile = std::string(*v);
  if (auto v = args.get_string("ssl_ca_certs")) tls.tls_ca_certs = std::string(*v);
  if (auto v = args.get_int("ssl_cert_reqs")) tls.tls_cert_reqs = int(*v);
  if (auto v = args.get_string("ssl_ciphers")) tls.tls_ciphers = std::string(*v);
  for (auto& [f, v] : ssl_cli) {
    if (f == "--ssl-certfile") tls.tls_certfile = v;
    else if (f == "--ssl-keyfile") tls.tls_keyfile = v;
    else if (f == "--ssl-ca-certs") tls.tls_ca_certs = v;
    else if (f == "--ssl-cert-reqs") tls.tls_cert_reqs = std::atoi(v.c_str());
    else if (f == "--ssl-ciphers") tls.tls_ciphers = v;
  }
  if (const char* t = std::getenv("FE_TLS_HANDSHAKE_TIMEOUT_S")) tls.tls_handshake_timeout_s = std::atof(t);
  // uvicorn: TLS when a key or a cert is given, and then the cert is required
  const bool use_tls = !tls.tls_certfile.empty() || !tls.tls_keyfile.empty();
  if (use_tls) {
    if (args.get_bool("enable_ssl_refresh").value_or(false))
      log("warning: --enable-ssl-refresh is not supported: certificates are loaded once at start");
    if (tls.tls_certfile.empty()) {
      log("TLS: --ssl-keyfile needs --ssl-certfile");
      return 2;
    }
    try {  // fail fast, before waiting for the engine
      fe::http::Server::check_tls(tls);
    } catch (const std::exception& e) {
      log(e.what());
      return 2;
    }
  }
  if (check_config) {  // the config parsed like a real start: report the key count only
    printf("api keys: %zu\n", cfg.api_keys.size());
    return 0;
  }

  // ---- components
  fe::api::Components comp;
  try {
    std::string tj = cfg.tokenizer_path + "/tokenizer.json";
    std::string tc = cfg.tokenizer_path + "/tokenizer_config.json";
    comp.tokenizer = std::shared_ptr<fe::Tokenizer>(fe::Tokenizer::load(tj, exists(tc) ? tc : std::string()));
  } catch (const std::exception& e) {
    log(std::string("tokenizer load failed: ") + e.what());
    return 1;
  }
  comp.vocab = fe::make_parser_vocab(*comp.tokenizer);
  comp.render.dsml_strict_env = std::getenv("DSML_STRICT") ? std::getenv("DSML_STRICT") : "vllm";
  comp.render.max_token_id = comp.tokenizer->max_token_id();
  comp.vision.model_id = args.get_string("model_tag") ? std::string(*args.get_string("model_tag")) : cfg.model_path;
  if (const char* mp = std::getenv("VLLM_MAX_IMAGE_PIXELS")) comp.vision.max_image_pixels = std::atoll(mp);
  if (const char* wd = std::getenv("FE_WEBP_LIB_DIR")) comp.vision.webp_lib_dir = wd;
  try {
    std::string cj = cfg.model_path + "/config.json";
    if (exists(cj)) {
      fe::Json mc = fe::json_parse(read_file(cj));
      const fe::Json& vc = mc["vision_config"];
      if (auto v = vc.get_int("patch_size")) comp.vision.patch_size = int(*v);
      if (auto v = vc.get_int("downsample_ratio")) comp.vision.downsample_ratio = int(*v);
      if (auto v = vc.get_int("max_image_tokens")) comp.vision.max_n_token = int(*v);
      if (auto v = vc.get_int("min_pixels")) comp.vision.min_pixels = *v;
      if (auto v = vc.get_number("max_wh_ratio")) comp.vision.max_wh_ratio = *v;
      if (cfg.max_model_len == 0)
        if (auto v = mc.get_int("max_position_embeddings")) cfg.max_model_len = *v;
    }
  } catch (const std::exception& e) {
    log(std::string("config.json: ") + e.what());
  }

  // ---- engine (E): adopt the ZMQ listeners and wait for the engine's ready message
  fe::engine::EngineOptions eo;
  eo.client.input_listener_fd = fa.input_listener_fd;
  eo.client.output_listener_fd = fa.output_listener_fd;
  eo.client.engine_start_index = fa.engine_start_index;
  eo.client.engine_count = fa.engine_count;
  eo.client.ready_timeout_s = cfg.engine_ready_timeout_s;
  eo.tokenizer = comp.tokenizer.get();
  eo.metrics.model_name = cfg.model_name();
  eo.metrics.max_model_len = cfg.max_model_len > 0 ? cfg.max_model_len : 655360;
  eo.metrics.num_spec_tokens = cfg.num_speculative_tokens;
  eo.log_stats = !cfg.disable_log_stats;
  if (auto si = args.get_int("stream_interval")) eo.stream_interval = *si;
  std::unique_ptr<fe::engine::AsyncEngine> eng;
  try {
    eng = std::make_unique<fe::engine::AsyncEngine>(eo);
    const auto& ready = eng->start();
    log("engine ready: max_model_len " + std::to_string(ready.max_model_len) + ", vllm " + ready.vllm_version);
  } catch (const std::exception& e) {
    log(std::string("engine start failed: ") + e.what());
    return 1;
  }
  cfg.max_model_len = eng->max_model_len();
  comp.render.max_model_len = cfg.max_model_len;
  fe::engine::SamplingContext sctx;
  sctx.max_model_len = cfg.max_model_len;
  sctx.vocab_size = comp.tokenizer->vocab_size();
  sctx.max_token_id = comp.tokenizer->max_token_id();
  sctx.dp_num_ranks = fa.data_parallel_size;
  // The kv_transfer_params check uses the same validate_token_ids limit: max(max_token_id, vocab_size - 1).
  comp.render.max_token_id = std::max(sctx.max_token_id, sctx.vocab_size - 1);
  sctx.eos_token_id = comp.tokenizer->token_to_id("<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>").value_or(1);
  sctx.speculative = cfg.num_speculative_tokens > 0;
  sctx.encode = &encode_hook;
  sctx.encode_ctx = comp.tokenizer.get();
  // the phys engine's sampling caps (adapter/sampling_guard.py's utility): a failure (the
  // utility does not exist: not the phys engine) or None = no API-side check; the engine's _unsupported stays the backstop
  try {
    fe::mp::Value cv = eng->client().call_utility("phys_sampling_caps", std::vector<fe::mp::Value>{}, 60);
    if (cv.is_array()) {
      std::vector<std::string> caps;
      for (auto& e : cv.arr)
        if (e.is_str()) caps.push_back(e.s);
      std::string l;
      for (auto& c : caps) l += (l.empty() ? "" : ", ") + c;
      log("sampling caps from the engine: " + (caps.empty() ? std::string("none beyond temperature / seed / stop / max_tokens") : l));
      sctx.phys_caps = std::move(caps);
    } else {
      log("sampling caps from the engine: all (no check)");
    }
  } catch (const std::exception& e) {
    log(std::string("no phys_sampling_caps utility on the engine (") + std::string(e.what()).substr(0, 160) + "): no API-side check");
  }
  auto backend = std::make_shared<fe::api::EngineBackend>(std::move(eng), comp.tokenizer.get(), sctx);

  auto app = std::make_shared<fe::api::App>(cfg, comp, backend);

  // ---- HTTP
  fe::http::ServerOptions so;
  if (have_listen) {
    so.listen_fd = fa.listen_fd;
  } else {
    so.host = host;
    so.port = port;
  }
  so.keep_alive_timeout_s = cfg.http_timeout_keep_alive;
  if (use_tls) {
    so.tls_certfile = tls.tls_certfile;
    so.tls_keyfile = tls.tls_keyfile;
    so.tls_ca_certs = tls.tls_ca_certs;
    so.tls_cert_reqs = tls.tls_cert_reqs;
    so.tls_ciphers = tls.tls_ciphers;
    so.tls_handshake_timeout_s = tls.tls_handshake_timeout_s;
  }
  so.access_log = !args.get_bool("disable_uvicorn_access_log").value_or(false);
  if (const char* w = std::getenv("FE_HTTP_WORKERS")) so.workers = std::atoi(w);
  fe::http::Server server(so, [app](std::shared_ptr<fe::http::Request> rq, std::shared_ptr<fe::http::Response> rs) {
    app->handle(std::move(rq), std::move(rs));
  });
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = on_signal;
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGINT, &sa, nullptr);
  signal(SIGPIPE, SIG_IGN);
  try {
    server.start();
  } catch (const std::exception& e) {
    log(std::string("http start failed: ") + e.what());
    return 1;
  }
  if (use_tls) {
    static const char* const kReqs[] = {"none", "optional", "required"};
    log("serving HTTPS on port " + std::to_string(server.port()) + " (TLS >= 1.2, ALPN http/1.1, cert " +
        so.tls_certfile + ", client certs " + kReqs[std::clamp(so.tls_cert_reqs, 0, 2)] + ")");
  } else {
    log("serving HTTP on port " + std::to_string(server.port()));
  }

  int rc = 0;
  auto dead_since = std::chrono::steady_clock::time_point();
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (int s = g_signal.load()) {
      log("signal " + std::to_string(s) + ": shutting down");
      break;
    }
    if (backend->dead()) {
      auto now = std::chrono::steady_clock::now();
      if (dead_since == std::chrono::steady_clock::time_point()) dead_since = now;
      // vLLM's watchdog checks every 5 s and then stops the server
      if (now - dead_since > std::chrono::seconds(5)) {
        log("engine is dead: exiting");
        rc = 1;
        break;
      }
    }
  }
  server.stop(1.0);
  _exit(rc);  // the engine client's threads are joined by the OS; avoid blocking destructors
}
