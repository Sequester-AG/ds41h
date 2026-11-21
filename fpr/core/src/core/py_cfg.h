// py_cfg.h -- the core's configuration from the adapter's Python cfg dict: one parser for
// fpr_core_py.CoreX (2a) and fpr_core2b.Core2b (2b), so both read the same keys the same way.
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "blockhash.h"
#include "core.h"   // lifecycle.h + grammar.h (relative includes)

namespace fpr {
namespace pycfg {
namespace py = pybind11;

// cfg: core_sched._core_init's dict; reasoners: [(kwargs json, reasoner_spec dict)]
inline core::LifecycleConfig lifecycle(py::dict cfg, py::list reasoners) {
  core::LifecycleConfig c;
  c.num_spec_tokens = cfg["k"].cast<int32_t>();
  c.max_model_len = cfg["max_model_len"].cast<int64_t>();
  c.page_tokens = cfg["page_tokens"].cast<int32_t>();
  c.async_mode = cfg["async"].cast<bool>();
  c.log_stats = cfg["log_stats"].cast<bool>();
  c.spec_decode_metrics_level = cfg["spec_decode_metrics_level"].cast<std::string>();
  c.include_finished_set = cfg["include_finished_set"].cast<bool>();
  c.grammar_hold = cfg["grammar_hold"].cast<bool>();
  c.grammar_max_frac = cfg["grammar_max_frac"].cast<double>();
  c.vocab_size = cfg["vocab_size"].cast<int32_t>();
  c.t0_mono = cfg["t0"].cast<double>();
  if (cfg.contains("enable_in_reasoning")) c.enable_in_reasoning = cfg["enable_in_reasoning"].cast<bool>();
  if (!cfg["sampling_caps"].is_none()) {
    c.has_sampling_caps = true;
    for (auto x : cfg["sampling_caps"]) c.sampling_caps.insert(x.cast<std::string>());
  }
  for (auto item : reasoners) {
    py::tuple t = item.cast<py::tuple>();
    py::dict j = t[1].cast<py::dict>();
    grammar::ReasonerSpec s;
    s.present = j["present"].cast<bool>();
    s.adapter = j["adapter"].cast<bool>();
    s.end_ids = j["end_ids"].cast<std::vector<int32_t>>();
    s.end_token_id = j["end_token_id"].cast<int32_t>();
    s.start_id = j["start_id"].cast<int32_t>();
    s.boundary_ids = j["boundary_ids"].cast<std::vector<int32_t>>();
    s.wait_for_reasoning = j["wait_for_reasoning"].cast<bool>();
    s.start_opens_reasoning = j["start_opens_reasoning"].cast<bool>();
    c.reasoners.emplace_back(t[0].cast<std::string>(), s);
  }
  return c;
}

// the xgrammar backend's config (tokenizer_info: TokenizerInfo.serialize_json(), or None = no structured outputs)
inline grammar::BackendConfig backend(py::dict cfg, const std::string& tokenizer_info, const core::LifecycleConfig& c) {
  grammar::BackendConfig bc;
  bc.tokenizer_info_json = tokenizer_info;
  bc.vocab_size = c.vocab_size;
  bc.num_spec_tokens = c.num_spec_tokens;
  if (cfg.contains("disable_any_whitespace")) bc.disable_any_whitespace = cfg["disable_any_whitespace"].cast<bool>();
  return bc;
}

// the core's prefix-cache block keys (SHA-256 chain, 128 bits; salt + private rid + image keys via the scheduler)
inline core::BlockHasher hasher(int32_t block_tokens) {
  return [block_tokens](const core::VReq& r, std::vector<Hash>* keys) {
    core::BlockKeyCtx ctx;
    ctx.salt = r.cache_salt;
    if (r.skip_reading_prefix_cache) ctx.private_rid = r.rid;
    ctx.block_tokens = block_tokens;
    core::extend_block_keys(ctx, r.all, keys);
  };
}

}  // namespace pycfg
}  // namespace fpr
