// SamplingParams exactly as vLLM 0.32.0 puts it on the wire.
//
// msgspec encodes SamplingParams (omit_defaults=True, dict=True) as a MAP in field-declaration order and
// omits a field only when its value IS the default object (identity, not equality; empty default-factory
// collections also count). The observable consequences for our three endpoints (verified byte for byte
// against vLLM, see tools/golden/engine_request_dump.py):
//   * float fields that pass through to_sampling_params (penalties, temperature, top_p, min_p) are always
//     emitted - their objects come from the request or from the endpoint's default dict - EXCEPT when greedy
//     post-processing (temperature < 1e-5) resets top_p / min_p to the class-default objects;
//   * int fields with a small default (n=1, top_k=0, max_tokens=16, min_tokens=0, ...) are omitted iff equal
//     (CPython caches small ints);
//   * None-default fields are emitted iff set; stop / stop_token_ids / bad_words are always emitted (post_init
//     turns None into []); watermarking=False and skip_reading_prefix_cache=False are always emitted.
// std::optional<double> below means "explicit object" (emitted) when engaged, "class default" when empty.
#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/mp.h"

namespace fe::engine {

// vLLM VLLMValidationError: str(e) == what() == "<message>[ (parameter=..., value=...)]".
struct ValidationError : std::runtime_error {
  std::string message, parameter, value;  // value: Python str() of the rejected value ("" = None)
  ValidationError(std::string msg, std::string param = {}, std::string val = {});
};
// the sampling-caps rejection (check_phys_caps): a ValidationError the API raises before streaming starts.
struct CapsError : ValidationError {
  using ValidationError::ValidationError;
};
// ValueError raised by get_max_tokens (same text as Python; the API layer maps it to HTTP 400).
struct PromptTooLongError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class OutputKind : int { Cumulative = 0, Delta = 1, FinalOnly = 2 };
enum class Endpoint { Chat, Completions, Responses };

struct StructuredOutputs {  // StructuredOutputsParams (pydantic dataclass -> full map on the wire)
  std::optional<mp::Value> json;  // str or object
  std::optional<std::string> regex;
  std::optional<std::vector<std::string>> choice;
  std::optional<std::string> grammar;
  std::optional<bool> json_object;
  bool disable_any_whitespace = false;
  bool disable_additional_properties = false;
  std::optional<std::string> whitespace_pattern;
  std::optional<std::string> structural_tag;
  std::optional<std::string> backend = std::string("xgrammar");  // set by InputProcessor validation
  bool backend_was_auto = true;
};

struct SamplingParams {
  int64_t n = 1;
  std::optional<double> presence_penalty, frequency_penalty, repetition_penalty, temperature;
  std::optional<bool> watermarking;
  std::optional<double> top_p;
  int64_t top_k = 0;
  std::optional<double> min_p;
  std::optional<int64_t> seed;
  std::optional<std::vector<std::string>> stop;
  std::optional<std::vector<int64_t>> stop_token_ids;
  bool ignore_eos = false;
  std::optional<int64_t> max_tokens = 16;  // nullopt = None
  int64_t min_tokens = 0;
  std::optional<int64_t> logprobs, prompt_logprobs;
  std::optional<std::vector<int64_t>> logprob_token_ids;
  bool detokenize = true;
  bool skip_special_tokens = true;
  bool spaces_between_special_tokens = true;
  bool include_stop_str_in_output = false;
  OutputKind output_kind = OutputKind::Cumulative;
  std::optional<int64_t> stream_interval;
  bool skip_clone = false;
  int64_t output_text_buffer_length = 0;
  std::optional<int64_t> eos_token_id;          // _eos_token_id
  std::vector<int64_t> all_stop_token_ids;      // _all_stop_token_ids, in CPython set iteration order
  bool watermarking_skipped = false;            // _watermarking_skipped
  std::optional<StructuredOutputs> structured_outputs;
  std::optional<std::vector<std::pair<int64_t, double>>> logit_bias;  // request dict order
  std::optional<std::vector<int64_t>> allowed_token_ids;
  std::optional<mp::Value> extra_args;          // map
  std::optional<std::vector<std::string>> bad_words;
  std::optional<std::vector<std::vector<int64_t>>> bad_words_token_ids;
  std::optional<bool> skip_reading_prefix_cache;
  std::optional<int64_t> thinking_token_budget;
  std::optional<mp::Value> repetition_detection;  // RepetitionDetectionParams as a map
  int64_t routed_experts_prompt_start = 0;
  std::optional<std::vector<int64_t>> trace_decode_token_ids;

  // Effective values (what Python reads), for the output processor.
  double eff_temperature() const { return temperature.value_or(1.0); }
  const std::vector<std::string>& stop_strings() const;
};

// Static facts the builders need (from the ready response / CLI / tokenizer).
struct SamplingContext {
  int64_t max_model_len = 655360;
  int64_t vocab_size = 129280;   // model_config.get_vocab_size()
  int64_t max_token_id = 129279; // tokenizer.max_token_id (validate_token_ids)
  int64_t dp_num_ranks = 1;      // process_inputs num_ranks: --data-parallel-size (no local_engines_only)
  int64_t max_logprobs = 20;     // model_config.max_logprobs
  int64_t eos_token_id = 1;      // renderer.get_eos_token_id()
  bool speculative = true;       // --speculative-config set (production: dspark k=3)
  bool watermark_configured = false;
  // Tokenizer hook for bad_words (encode(text, add_special_tokens=False)); empty = bad_words unsupported.
  std::vector<int64_t> (*encode)(const std::string&, void*) = nullptr;
  void* encode_ctx = nullptr;
  // the phys engine's sampling caps (EngineCore utility phys_sampling_caps, fetched at startup)
  // nullopt = no check (not the phys engine, or caps None). Engaged: a request whose SamplingParams use a feature outside
  // it is a 400 before streaming (adapter/sampling_guard.py, same text), instead of the engine's mid-stream error.
  std::optional<std::vector<std::string>> phys_caps;
};

// get_max_tokens() from serve/utils/api_utils.py (default_sampling_params empty, no override).
int64_t get_max_tokens(int64_t max_model_len, std::optional<int64_t> requested, int64_t input_length);

// <Endpoint>Request.to_sampling_params(get_max_tokens(...), {}) followed by SamplingParams.__post_init__,
// then InputProcessor.process_inputs (verify, watermarking, max_tokens default, generation config: eos).
// The InputProcessor.process_inputs checks that are not SamplingParams.verify, run at their place: after verify and
// watermarking, data_parallel_rank in [0, ctx.dp_num_ranks); then _validate_model_inputs (empty prompt,
// validate_token_ids(prompt_ids), input_processor.py:647, parameter None); then routed_experts_prompt_start
// <= prompt length; all before the max_tokens default and the generation-config / tokenizer (bad_words) updates.
// build_engine_request passes them; without them (nullptr) none of these run.
struct InputChecks {
  std::span<const int32_t> prompt_ids;
  std::optional<std::string> prompt_oov_id;  // BuildInput::prompt_oov_id
  std::optional<int64_t> data_parallel_rank;
  std::string data_parallel_rank_text;       // str() of a rank that does not fit int64 ("" = data_parallel_rank)
};

// `body` is the (already schema-validated) request JSON as a msgpack DOM.
// Throws ValidationError / PromptTooLongError with vLLM's exact messages.
SamplingParams sampling_from_request(Endpoint ep, const mp::Value& body, int64_t prompt_len,
                                     const SamplingContext& ctx, const InputChecks* checks = nullptr);

// adapter/sampling_features.py: the non-default sampling features sp uses (beyond temperature / seed / stop /
// max_tokens), sorted. check_phys_caps throws sampling_guard's ValidationError when ctx.phys_caps lacks one.
std::vector<std::string> sampling_features(const SamplingParams& sp);
void check_phys_caps(const SamplingParams& sp, const SamplingContext& ctx);

// The individual stages, for callers that build SamplingParams themselves.
void post_init(SamplingParams& sp);                       // SamplingParams.__post_init__ (+ _verify_args)
void process_for_engine(SamplingParams& sp, int64_t prompt_len, const SamplingContext& ctx,
                        const InputChecks* checks = nullptr);

// BaseRenderer.validate_token_ids (renderers/base.py:420). No-op for an empty list. Otherwise min(ids) < 0 or
// max(ids) > max(tokenizer.max_token_id, vocab_size - 1) throws ValidationError("Token id <id> is out of
// vocabulary", parameter); the min is checked first. `oov_id`: Python str() of the id to report for a prompt whose
// ids do not fit int32 (BuildInput::prompt_oov_id); it wins over `ids`, which then hold clamped placeholders.
void validate_token_ids(std::span<const int32_t> ids, const SamplingContext& ctx, const std::string& parameter = {},
                        const std::optional<std::string>& oov_id = std::nullopt);

// Wire encoding (msgspec-identical map) and its inverse (absent key = class default).
void pack_sampling_params(mp::Packer& p, const SamplingParams& sp);
SamplingParams sampling_from_wire(const mp::Value& map);

// CPython set iteration order for a set of small non-negative ints built by successive add() calls.
std::vector<int64_t> python_int_set_order(const std::vector<int64_t>& inserts);

// Python repr() of a float (shortest round-trip, "1.0", "1e-05", "inf").
std::string py_float_repr(double d);

}  // namespace fe::engine
