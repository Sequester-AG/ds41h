// process_* and http_* families of /metrics (prometheus_client process collector +
// prometheus-fastapi-instrumentator defaults). The vllm:* families come from fe_engine's
// Metrics (PrometheusStatLogger port) and are spliced in between, in Python's order.
// vllm:tool_call_parser_invocations_* (ToolParserMetrics) follows http_*.
#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fe::api {

class HttpMetrics {
 public:
  HttpMetrics();
  void on_http(const std::string& handler, const std::string& method, int status, double seconds,
               size_t req_bytes, size_t resp_bytes);
  std::string render_process();
  std::string render_http();

 private:
  struct Hist {
    std::vector<double> bounds;
    std::vector<double> counts;
    double sum = 0, n = 0, created = 0;
    void observe(double v);
  };
  struct HttpKey {
    std::string handler, method, status;
    bool operator<(const HttpKey& o) const {
      return std::tie(handler, method, status) < std::tie(o.handler, o.method, o.status);
    }
  };
  std::mutex mu_;
  double start_time_ = 0;
  std::vector<HttpKey> order_;  // first-use order of http_requests_total children
  std::map<HttpKey, double> http_requests_, http_requests_created_;
  std::vector<std::string> handler_order_;
  std::map<std::string, std::pair<double, double>> req_size_, resp_size_;
  std::map<std::string, double> size_created_;
  std::vector<std::pair<std::string, std::string>> dur_order_;
  std::map<std::pair<std::string, std::string>, Hist> duration_;
  Hist highr_;
};

// vllm:tool_call_parser_invocations_total (vllm/parser/metrics.py). init_app_state registers it when
// --tool-call-parser is set, after the instrumentator, so it renders after http_*; init_parser_metrics creates all
// 12 children (mode x outcome x request_type, in that product order) up front, so they show 0.0 before any request.
enum class ToolParserRequestType : int { ChatCompletions = 0, Responses = 1, Other = 2 };

class ToolParserMetrics {
 public:
  explicit ToolParserMetrics(std::string model_name);  // == init_parser_metrics(model_name=served_model_name)
  void record(bool streaming, bool tool_called, ToolParserRequestType rt);  // == record_tool_parser_invocation
  std::string render() const;

 private:
  std::string model_name_;
  std::atomic<uint64_t> n_[2][2][3];  // [streaming ? 0 : 1][tool_called ? 0 : 1][request_type]
  double created_[2][2][3];
};

std::string go_float(double d);

}  // namespace fe::api
