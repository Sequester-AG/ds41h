#include "api/http_metrics.h"

#include <dirent.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <tuple>

#include "api/jw.h"

namespace fe::api {

namespace {
double unix_now() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

const std::vector<double> kLatency = {0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0,
                                      30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0};
const std::vector<double> kTtft = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75,
                                   1.0,   2.5,   5.0,  7.5,  10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0};
const std::vector<double> kItl = {0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                                  0.75, 1.0,   2.5,  5.0,   7.5, 10.0, 20.0, 40.0, 80.0};
const std::vector<double> kIter = {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
const std::vector<double> kN = {1, 2, 5, 10, 20};
const std::vector<double> kPreempt = {1, 2, 3, 4, 5, 10, 20};
const std::vector<double> kHighr = {0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2,
                                    2.5,  3,     3.5,  4,     4.5, 5,    7.5, 10,   30, 60};
const std::vector<double> kLowr = {0.1, 0.5, 1};

std::vector<double> buckets_125(int64_t max_value) {
  std::vector<double> b;
  for (int64_t e = 1;; e *= 10) {
    for (int m : {1, 2, 5}) {
      int64_t v = m * e;
      if (v > max_value) return b;
      b.push_back(double(v));
    }
  }
}

std::string esc_label(const std::string& v) {
  std::string o;
  for (char c : v) {
    if (c == '\\')
      o += "\\\\";
    else if (c == '"')
      o += "\\\"";
    else if (c == '\n')
      o += "\\n";
    else
      o += c;
  }
  return o;
}

// labels rendered sorted by name (prometheus_client sample_line)
std::string labels(std::vector<std::pair<std::string, std::string>> kv) {
  if (kv.empty()) return "";
  std::sort(kv.begin(), kv.end());
  std::string s = "{";
  for (size_t i = 0; i < kv.size(); ++i) {
    if (i) s += ',';
    s += kv[i].first + "=\"" + esc_label(kv[i].second) + "\"";
  }
  return s + "}";
}
}  // namespace

std::string go_float(double d) {
  if (std::isinf(d)) return d > 0 ? "+Inf" : "-Inf";
  if (std::isnan(d)) return "NaN";
  std::string s = py_float_repr(d);
  size_t dot = s.find('.');
  if (d > 0 && dot != std::string::npos && dot > 6) {
    std::string mant = std::string(1, s[0]) + "." + s.substr(1, dot - 1) + s.substr(dot + 1);
    while (!mant.empty() && (mant.back() == '0' || mant.back() == '.')) mant.pop_back();
    char e[16];
    snprintf(e, sizeof e, "e+%02d", int(dot - 1));
    return mant + e;
  }
  return s;
}

void HttpMetrics::Hist::observe(double v) {
  size_t i = 0;
  while (i < bounds.size() && v > bounds[i]) ++i;
  counts[i] += 1;
  sum += v;
  n += 1;
}


HttpMetrics::HttpMetrics() {
  start_time_ = unix_now();
  std::ifstream st("/proc/self/stat");
  std::string line;
  std::getline(st, line);
  size_t rp = line.rfind(')');
  if (rp != std::string::npos) {
    std::istringstream is(line.substr(rp + 2));
    std::vector<std::string> f;
    std::string x;
    while (is >> x) f.push_back(x);
    if (f.size() > 19) {
      double ticks = std::atof(f[19].c_str());
      std::ifstream ps("/proc/stat");
      std::string l;
      while (std::getline(ps, l))
        if (l.rfind("btime ", 0) == 0) start_time_ = std::atof(l.c_str() + 6) + ticks / double(sysconf(_SC_CLK_TCK));
    }
  }
  highr_.bounds = kHighr;
  highr_.counts.assign(kHighr.size() + 1, 0);
  highr_.created = unix_now();
}

void HttpMetrics::on_http(const std::string& handler, const std::string& method, int status, double seconds,
                          size_t req_bytes, size_t resp_bytes) {
  std::lock_guard<std::mutex> lk(mu_);
  double now = unix_now();
  HttpKey k{handler, method, std::to_string(status / 100) + "xx"};
  if (!http_requests_.count(k)) {
    order_.push_back(k);
    http_requests_created_[k] = now;
  }
  http_requests_[k] += 1;
  if (!req_size_.count(handler)) {
    handler_order_.push_back(handler);
    size_created_[handler] = now;
  }
  auto& a = req_size_[handler];
  a.first += 1;
  a.second += double(req_bytes);
  auto& b = resp_size_[handler];
  b.first += 1;
  b.second += double(resp_bytes);
  highr_.observe(seconds);
  auto key = std::make_pair(method, handler);
  auto it = duration_.find(key);
  if (it == duration_.end()) {
    Hist h;
    h.bounds = kLowr;
    h.counts.assign(kLowr.size() + 1, 0);
    h.created = now;
    dur_order_.push_back(key);
    it = duration_.emplace(key, h).first;
  }
  it->second.observe(seconds);
}

namespace {
void help(std::string& o, const std::string& name, const std::string& type, const std::string& doc) {
  o += "# HELP " + name + " " + doc + "\n# TYPE " + name + " " + type + "\n";
}
void line(std::string& o, const std::string& name, const std::vector<std::pair<std::string, std::string>>& kv, double v) {
  o += name + labels(kv) + " " + go_float(v) + "\n";
}
}  // namespace

std::string HttpMetrics::render_process() {
  std::string o;
  long size = 0, res = 0;
  {
    std::ifstream sm("/proc/self/statm");
    sm >> size >> res;
  }
  long pg = sysconf(_SC_PAGESIZE);
  double cpu = 0;
  {
    std::ifstream st("/proc/self/stat");
    std::string ln;
    std::getline(st, ln);
    size_t rp = ln.rfind(')');
    if (rp != std::string::npos) {
      std::istringstream is(ln.substr(rp + 2));
      std::vector<std::string> f;
      std::string x;
      while (is >> x) f.push_back(x);
      if (f.size() > 12) cpu = (std::atof(f[11].c_str()) + std::atof(f[12].c_str())) / double(sysconf(_SC_CLK_TCK));
    }
  }
  int fds = 0;
  if (DIR* d = opendir("/proc/self/fd")) {
    while (struct dirent* e = readdir(d))
      if (e->d_name[0] != '.') ++fds;
    closedir(d);
    --fds;  // the DIR's own fd
  }
  struct rlimit rl;
  double maxfd = getrlimit(RLIMIT_NOFILE, &rl) == 0 ? double(rl.rlim_cur) : double(sysconf(_SC_OPEN_MAX));
  help(o, "process_virtual_memory_bytes", "gauge", "Virtual memory size in bytes.");
  o += "process_virtual_memory_bytes " + go_float(double(size) * double(pg)) + "\n";
  help(o, "process_resident_memory_bytes", "gauge", "Resident memory size in bytes.");
  o += "process_resident_memory_bytes " + go_float(double(res) * double(pg)) + "\n";
  help(o, "process_start_time_seconds", "gauge", "Start time of the process since unix epoch in seconds.");
  o += "process_start_time_seconds " + go_float(start_time_) + "\n";
  help(o, "process_cpu_seconds_total", "counter", "Total user and system CPU time spent in seconds.");
  o += "process_cpu_seconds_total " + go_float(cpu) + "\n";
  help(o, "process_open_fds", "gauge", "Number of open file descriptors.");
  o += "process_open_fds " + go_float(double(fds)) + "\n";
  help(o, "process_max_fds", "gauge", "Maximum number of open file descriptors.");
  o += "process_max_fds " + go_float(maxfd) + "\n";
  return o;
}

std::string HttpMetrics::render_http() {
  std::lock_guard<std::mutex> lk(mu_);
  std::string o;
  auto hist = [&](const std::string& name, const Hist& h, const std::vector<std::pair<std::string, std::string>>& kv) {
    double acc = 0;
    for (size_t i = 0; i <= h.bounds.size(); ++i) {
      acc += h.counts[i];
      auto l = kv;
      l.emplace_back("le", i < h.bounds.size() ? go_float(h.bounds[i]) : "+Inf");
      line(o, name + "_bucket", l, acc);
    }
    line(o, name + "_count", kv, acc);
    line(o, name + "_sum", kv, h.sum);
  };
  const std::string tot_doc = "Total number of requests by method, status and handler.";
  help(o, "http_requests_total", "counter", tot_doc);
  for (auto& k : order_)
    line(o, "http_requests_total", {{"method", k.method}, {"status", k.status}, {"handler", k.handler}}, http_requests_[k]);
  help(o, "http_requests_created", "gauge", tot_doc);
  for (auto& k : order_)
    line(o, "http_requests_created", {{"method", k.method}, {"status", k.status}, {"handler", k.handler}},
         http_requests_created_[k]);
  const std::string in_doc =
      "Content length of incoming requests by handler. Only value of header is respected. Otherwise ignored. No "
      "percentile calculated. ";
  const std::string out_doc =
      "Content length of outgoing responses by handler. Only value of header is respected. Otherwise ignored. No "
      "percentile calculated. ";
  help(o, "http_request_size_bytes", "summary", in_doc);
  for (auto& h : handler_order_) {
    line(o, "http_request_size_bytes_count", {{"handler", h}}, req_size_[h].first);
    line(o, "http_request_size_bytes_sum", {{"handler", h}}, req_size_[h].second);
  }
  help(o, "http_request_size_bytes_created", "gauge", in_doc);
  for (auto& h : handler_order_) line(o, "http_request_size_bytes_created", {{"handler", h}}, size_created_[h]);
  help(o, "http_response_size_bytes", "summary", out_doc);
  for (auto& h : handler_order_) {
    line(o, "http_response_size_bytes_count", {{"handler", h}}, resp_size_[h].first);
    line(o, "http_response_size_bytes_sum", {{"handler", h}}, resp_size_[h].second);
  }
  help(o, "http_response_size_bytes_created", "gauge", out_doc);
  for (auto& h : handler_order_) line(o, "http_response_size_bytes_created", {{"handler", h}}, size_created_[h]);
  const std::string hr_doc =
      "Latency with many buckets but no API specific labels. Made for more accurate percentile calculations. ";
  help(o, "http_request_duration_highr_seconds", "histogram", hr_doc);
  hist("http_request_duration_highr_seconds", highr_, {});
  help(o, "http_request_duration_highr_seconds_created", "gauge", hr_doc);
  o += "http_request_duration_highr_seconds_created " + go_float(highr_.created) + "\n";
  const std::string lr_doc =
      "Latency with only few buckets by handler. Made to be only used if aggregation by handler is important. ";
  help(o, "http_request_duration_seconds", "histogram", lr_doc);
  for (auto& k : dur_order_) hist("http_request_duration_seconds", duration_[k], {{"method", k.first}, {"handler", k.second}});
  help(o, "http_request_duration_seconds_created", "gauge", lr_doc);
  for (auto& k : dur_order_)
    line(o, "http_request_duration_seconds_created", {{"method", k.first}, {"handler", k.second}}, duration_[k].created);
  return o;
}

// ---------------------------------------------------------------------------------------- tool-parser invocations

namespace {
const char* const kTpModes[2] = {"streaming", "non_streaming"};
const char* const kTpOutcomes[2] = {"tool_call", "no_tool_call"};      // ToolCallOutcome
const char* const kTpTypes[3] = {"chat_completions", "responses", "other"};  // RequestType
}  // namespace

ToolParserMetrics::ToolParserMetrics(std::string model_name) : model_name_(std::move(model_name)) {
  // for mode, outcome, request_type in product(...): counter.labels(...) -> one _created timestamp per child
  for (int m = 0; m < 2; ++m)
    for (int oc = 0; oc < 2; ++oc)
      for (int t = 0; t < 3; ++t) {
        n_[m][oc][t].store(0, std::memory_order_relaxed);
        created_[m][oc][t] = unix_now();
      }
}

void ToolParserMetrics::record(bool streaming, bool tool_called, ToolParserRequestType rt) {
  n_[streaming ? 0 : 1][tool_called ? 0 : 1][int(rt)].fetch_add(1, std::memory_order_relaxed);
}

std::string ToolParserMetrics::render() const {
  // prometheus_client Counter("vllm:tool_call_parser_invocations_total", labelnames=[model_name, mode, outcome,
  // request_type]); generate_latest: <name>_total samples, then the _created samples as a gauge, labels sorted.
  const std::string doc =
      "Total number of ToolParser invocations. Non-streaming increments once per choice; streaming increments once "
      "per delta.";
  std::string o;
  auto kv = [&](int m, int oc, int t) {
    return std::vector<std::pair<std::string, std::string>>{
        {"model_name", model_name_}, {"mode", kTpModes[m]}, {"outcome", kTpOutcomes[oc]}, {"request_type", kTpTypes[t]}};
  };
  help(o, "vllm:tool_call_parser_invocations_total", "counter", doc);
  for (int m = 0; m < 2; ++m)
    for (int oc = 0; oc < 2; ++oc)
      for (int t = 0; t < 3; ++t)
        line(o, "vllm:tool_call_parser_invocations_total", kv(m, oc, t),
             double(n_[m][oc][t].load(std::memory_order_relaxed)));
  help(o, "vllm:tool_call_parser_invocations_created", "gauge", doc);
  for (int m = 0; m < 2; ++m)
    for (int oc = 0; oc < 2; ++oc)
      for (int t = 0; t < 3; ++t) line(o, "vllm:tool_call_parser_invocations_created", kv(m, oc, t), created_[m][oc][t]);
  return o;
}

}  // namespace fe::api
