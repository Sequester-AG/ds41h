// vLLM's PrometheusStatLogger (v1/metrics/loggers.py) for the vllm:* families.
// Families, help text, label sets and histogram buckets come from metrics_families.inc, generated from the
// installed vLLM by tools/golden/engine_metrics_gen.py; record() follows PrometheusStatLogger.record().
// render_vllm() prints them like prometheus_client.generate_latest (sorted labels, floatToGoString,
// *_created gauges). The API layer appends http_* / process_* / python_* itself.
#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/mp.h"
#include "engine/output_processor.h"
#include "engine/protocol.h"

namespace fe::engine {

struct MetricsConfig {
  std::string model_name = "deepseek-v4.1-flash";  // served_model_name (the last --served-model-name)
  int64_t max_model_len = 655360;
  int num_spec_tokens = 3;
};

class Metrics {
 public:
  explicit Metrics(MetricsConfig cfg);
  // PrometheusStatLogger.record(scheduler_stats, iteration_stats, engine_idx=0). Thread-safe.
  void record(const mp::Value* scheduler_stats, const IterationStats* iteration);
  // log_engine_initialized(): fills the cache_config_info labels from the ready response.
  void set_ready(const ReadyResponse& r);
  // Prometheus text for the vllm:* families (prometheus_client 0.0.4 text format). Thread-safe.
  std::string render_vllm() const;
  // Single values for tests / the router-facing quick checks (sum over children). Thread-safe.
  double value(const std::string& sample_name) const;

  // prometheus_client.utils.floatToGoString
  static std::string go_float(double d);

 private:
  struct Child {
    std::vector<std::pair<std::string, std::string>> labels;  // sorted, without "le"
    double value = 0;               // counter total / gauge value
    std::vector<double> buckets;    // histogram: per-bucket (non-cumulative) counts
    double sum = 0, count = 0;
    double created = 0;
    bool present = true;            // cache_config_info appears only after set_ready()
  };
  struct Family {
    std::string type, name, doc;
    std::vector<double> bounds;          // histogram upper bounds (last = +Inf)
    std::vector<std::string> bound_txt;  // their "le" text
    std::vector<Child> children;
  };
  Family* fam(const std::string& name);
  Child* child(const std::string& name, const char* key = nullptr, const std::string& val = {});
  void inc(const std::string& name, double v, const char* key = nullptr, const std::string& val = {});
  void set(const std::string& name, double v, const char* key = nullptr, const std::string& val = {});
  void observe(const std::string& name, double v);

  MetricsConfig cfg_;
  mutable std::mutex mu_;
  std::vector<Family> fams_;
  std::unordered_map<std::string, size_t> idx_;
};

}  // namespace fe::engine
