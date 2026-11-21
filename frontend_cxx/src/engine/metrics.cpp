#include "engine/metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

#include "engine/sampling.h"  // py_float_repr

namespace fe::engine {

#include "engine/metrics_families.inc"

namespace {
std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  size_t a = 0;
  while (true) {
    size_t b = s.find(sep, a);
    out.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
    if (b == std::string::npos) break;
    a = b + 1;
  }
  return out;
}
std::string replace_all(std::string s, const std::string& from, const std::string& to) {
  size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) {
    s.replace(p, from.size(), to);
    p += to.size();
  }
  return s;
}
std::string escape_label(const std::string& v) {
  std::string o;
  for (char c : v) {
    if (c == '\\') o += "\\\\";
    else if (c == '\n') o += "\\n";
    else if (c == '"') o += "\\\"";
    else o += c;
  }
  return o;
}
}  // namespace

std::string Metrics::go_float(double d) {
  if (std::isinf(d)) return d > 0 ? "+Inf" : "-Inf";
  if (std::isnan(d)) return "NaN";
  std::string s = py_float_repr(d);
  size_t dot = s.find('.');
  if (d > 0 && dot != std::string::npos && dot > 6) {
    std::string mant = s.substr(0, 1) + "." + s.substr(1, dot - 1) + s.substr(dot + 1);
    while (!mant.empty() && (mant.back() == '0' || mant.back() == '.')) mant.pop_back();
    char e[16];
    std::snprintf(e, sizeof e, "e+%02d", int(dot - 1));
    return mant + e;
  }
  return s;
}

Metrics::Metrics(MetricsConfig cfg) : cfg_(std::move(cfg)) {
  double now = wall_time();
  std::istringstream in(kMetricFamilies);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto parts = split(line, '\t');
    if (parts[0] == "F" && parts.size() >= 4) {
      Family f;
      f.type = parts[1];
      f.name = parts[2];
      f.doc = parts[3];
      idx_[f.name] = fams_.size();
      fams_.push_back(std::move(f));
    } else if (parts[0] == "B" && !fams_.empty()) {
      for (auto& b : split(parts[1], ',')) {
        fams_.back().bound_txt.push_back(b);
        fams_.back().bounds.push_back(b == "+Inf" ? INFINITY : std::strtod(b.c_str(), nullptr));
      }
    } else if (parts[0] == "C" && !fams_.empty()) {
      Child c;
      if (parts.size() >= 2 && !parts[1].empty()) {
        for (auto& kv : split(parts[1], '\x1f')) {
          size_t eq = kv.find('=');
          std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
          if (v == "@MODEL@") v = cfg_.model_name;
          c.labels.emplace_back(k, v);
        }
      }
      if (parts.size() >= 3) c.value = std::strtod(parts[2].c_str(), nullptr);
      c.buckets.assign(fams_.back().bounds.size(), 0.0);
      c.created = now;
      if (fams_.back().name == "vllm:cache_config_info") c.present = false;
      fams_.back().children.push_back(std::move(c));
    }
  }
}

Metrics::Family* Metrics::fam(const std::string& name) {
  auto it = idx_.find(name);
  return it == idx_.end() ? nullptr : &fams_[it->second];
}

Metrics::Child* Metrics::child(const std::string& name, const char* key, const std::string& val) {
  Family* f = fam(name);
  if (!f) return nullptr;
  for (auto& c : f->children) {
    if (!key) return &c;
    for (auto& [k, v] : c.labels)
      if (k == key && v == val) return &c;
  }
  return nullptr;
}

void Metrics::inc(const std::string& name, double v, const char* key, const std::string& val) {
  if (Child* c = child(name, key, val)) c->value += v;
}
void Metrics::set(const std::string& name, double v, const char* key, const std::string& val) {
  if (Child* c = child(name, key, val)) c->value = v;
}
void Metrics::observe(const std::string& name, double v) {
  Family* f = fam(name);
  if (!f || f->children.empty()) return;
  Child& c = f->children[0];
  c.sum += v;
  c.count += 1;
  for (size_t i = 0; i < f->bounds.size(); ++i)
    if (v <= f->bounds[i]) {
      c.buckets[i] += 1;
      break;
    }
}

void Metrics::record(const mp::Value* sched, const IterationStats* it) {
  std::lock_guard<std::mutex> lk(mu_);
  if (sched && sched->is_map()) {
    SchedulerStatsView s = view_scheduler_stats(*sched);
    set("vllm:num_requests_running", double(s.num_running_reqs));
    set("vllm:num_requests_waiting", double(s.num_waiting_reqs + s.num_skipped_waiting_reqs));
    set("vllm:num_requests_waiting_by_reason", double(s.num_waiting_reqs), "reason", "capacity");
    set("vllm:num_requests_waiting_by_reason", double(s.num_skipped_waiting_reqs), "reason", "deferred");
    set("vllm:kv_cache_usage_perc", s.kv_cache_usage);
    inc("vllm:prefix_cache_queries", double(s.prefix_queries));
    inc("vllm:prefix_cache_hits", double(s.prefix_hits));
    inc("vllm:prompt_tokens_cached_by_source", double(s.prefix_hits), "source", "device");
    if (s.spec) {
      inc("vllm:spec_decode_num_drafts", double(s.spec->num_drafts));
      inc("vllm:spec_decode_num_draft_tokens", double(s.spec->num_draft_tokens));
      inc("vllm:spec_decode_num_accepted_tokens", double(s.spec->num_accepted_tokens));
      for (size_t p = 0; p < s.spec->num_accepted_tokens_per_pos.size(); ++p)
        inc("vllm:spec_decode_num_accepted_tokens_per_pos", double(s.spec->num_accepted_tokens_per_pos[p]), "position",
            std::to_string(p));
    }
    if (auto* ps = sched->get("perf_stats"); ps && ps->is_map()) {
      auto num = [&](const char* k) {
        auto* v = ps->get(k);
        return v && v->is_num() ? v->as_double() : 0.0;
      };
      inc("vllm:estimated_flops_per_gpu", num("num_flops_per_gpu"));
      inc("vllm:estimated_read_bytes_per_gpu", num("num_read_bytes_per_gpu"));
      inc("vllm:estimated_write_bytes_per_gpu", num("num_write_bytes_per_gpu"));
    }
  }
  if (!it) return;
  inc("vllm:num_preemptions", double(it->num_preempted_reqs));
  inc("vllm:prompt_tokens", double(it->pt_total));
  inc("vllm:prompt_tokens_by_source", double(it->pt_computed), "source", "local_compute");
  inc("vllm:prompt_tokens_by_source", double(it->pt_local_cache_hit), "source", "local_cache_hit");
  inc("vllm:prompt_tokens_by_source", double(it->pt_external_kv_transfer), "source", "external_kv_transfer");
  inc("vllm:prompt_tokens_cached", double(it->pt_cached_tokens));
  inc("vllm:generation_tokens", double(it->num_generation_tokens));
  observe("vllm:iteration_tokens_total", double(it->pt_computed + it->num_generation_tokens));
  for (auto v : it->max_num_generation_tokens_iter) observe("vllm:request_max_num_generation_tokens", double(v));
  for (auto v : it->n_params_iter) observe("vllm:request_params_n", double(v));
  for (auto v : it->time_to_first_tokens_iter) observe("vllm:time_to_first_token_seconds", v);
  for (auto v : it->inter_token_latencies_iter) observe("vllm:inter_token_latency_seconds", v);
  for (auto& f : it->finished_requests) {
    inc("vllm:request_success", 1, "finished_reason", finish_reason_str(f.finish_reason));
    observe("vllm:e2e_request_latency_seconds", f.e2e_latency);
    observe("vllm:request_queue_time_seconds", f.queued_time);
    observe("vllm:request_prefill_time_seconds", f.prefill_time);
    observe("vllm:request_inference_time_seconds", f.inference_time);
    observe("vllm:request_decode_time_seconds", f.decode_time);
    observe("vllm:request_num_preemptions", double(f.num_preemptions));
    observe("vllm:request_prefill_kv_computed_tokens",
            double(f.num_prompt_tokens - std::max<int64_t>(f.num_cached_tokens, 0)));
    observe("vllm:request_prompt_tokens", double(f.num_prompt_tokens));
    observe("vllm:request_generation_tokens", double(f.num_generation_tokens));
    observe("vllm:request_time_per_output_token_seconds", f.mean_time_per_output_token);
    if (f.max_tokens_param && *f.max_tokens_param) observe("vllm:request_params_max_tokens", double(*f.max_tokens_param));
  }
}

void Metrics::set_ready(const ReadyResponse& r) {
  std::lock_guard<std::mutex> lk(mu_);
  Family* f = fam("vllm:cache_config_info");
  if (!f) return;
  auto opt = [](const std::optional<int64_t>& v) { return v ? std::to_string(*v) : std::string("None"); };
  for (auto& c : f->children) {
    for (auto& [k, v] : c.labels) {
      v = replace_all(v, "@num_gpu_blocks@", std::to_string(r.num_gpu_blocks));
      v = replace_all(v, "@block_size@", std::to_string(r.block_size));
      v = replace_all(v, "@kv_cache_size_tokens@", opt(r.kv_cache_size_tokens));
      v = replace_all(v, "@mamba_block_size@", opt(r.mamba_block_size));
      v = replace_all(v, "@kv_cache_max_concurrency@",
                      r.kv_cache_max_concurrency ? py_float_repr(*r.kv_cache_max_concurrency) : "None");
    }
    c.present = true;
    c.value = 1;
    c.created = wall_time();
  }
}

std::string Metrics::render_vllm() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::string o;
  o.reserve(64 * 1024);
  auto labelstr = [](const std::vector<std::pair<std::string, std::string>>& l, const std::string* le) {
    // sorted(labels.items()) incl. "le"
    std::vector<std::pair<std::string, std::string>> all = l;
    if (le) all.emplace_back("le", *le);
    std::sort(all.begin(), all.end());
    if (all.empty()) return std::string();
    std::string s = "{";
    for (size_t i = 0; i < all.size(); ++i) {
      if (i) s += ',';
      s += all[i].first + "=\"" + escape_label(all[i].second) + "\"";
    }
    return s + "}";
  };
  for (auto& f : fams_) {
    bool any = false;
    for (auto& c : f.children) any = any || c.present;
    if (!any) continue;
    std::string mname = f.type == "counter" ? f.name + "_total" : f.name;
    o += "# HELP " + mname + " " + f.doc + "\n";
    o += "# TYPE " + mname + " " + f.type + "\n";
    std::string created;
    for (auto& c : f.children) {
      if (!c.present) continue;
      if (f.type == "histogram") {
        double cum = 0;
        for (size_t i = 0; i < f.bounds.size(); ++i) {
          cum += c.buckets[i];
          o += f.name + "_bucket" + labelstr(c.labels, &f.bound_txt[i]) + " " + go_float(cum) + "\n";
        }
        o += f.name + "_count" + labelstr(c.labels, nullptr) + " " + go_float(c.count) + "\n";
        o += f.name + "_sum" + labelstr(c.labels, nullptr) + " " + go_float(c.sum) + "\n";
        created += f.name + "_created" + labelstr(c.labels, nullptr) + " " + go_float(c.created) + "\n";
      } else if (f.type == "counter") {
        o += mname + labelstr(c.labels, nullptr) + " " + go_float(c.value) + "\n";
        created += f.name + "_created" + labelstr(c.labels, nullptr) + " " + go_float(c.created) + "\n";
      } else {
        o += f.name + labelstr(c.labels, nullptr) + " " + go_float(c.value) + "\n";
      }
    }
    if (!created.empty()) {
      o += "# HELP " + f.name + "_created " + f.doc + "\n";
      o += "# TYPE " + f.name + "_created gauge\n";
      o += created;
    }
  }
  return o;
}

double Metrics::value(const std::string& sample) const {
  std::lock_guard<std::mutex> lk(mu_);
  double s = 0;
  for (auto& f : fams_) {
    for (auto& c : f.children) {
      if (f.type == "counter" && sample == f.name + "_total") s += c.value;
      else if (f.type == "gauge" && sample == f.name) s += c.value;
      else if (f.type == "histogram" && sample == f.name + "_sum") s += c.sum;
      else if (f.type == "histogram" && sample == f.name + "_count") s += c.count;
    }
  }
  return s;
}

}  // namespace fe::engine
