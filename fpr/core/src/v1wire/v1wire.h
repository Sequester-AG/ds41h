// v1wire.h -- the vLLM V1 EngineCore wire in C++: EngineCoreOutputs out, EngineCoreRequest /
// ABORT / UTILITY in, as the installed vLLM (0.32.0 dev + patches, msgspec 0.22) encodes them.
//
// Encoding facts (msgspec, checked against vLLM's MsgpackEncoder by tests/test_wire.cpp):
//   * array_like Structs (EngineCoreOutputs, EngineCoreOutput, UtilityOutput, EngineCoreRequest) are arrays of ALL their
//     fields in declaration order: omit_defaults does not shorten array_like structs.
//   * dataclasses (SchedulerStats, PrefixCacheStats, PrefillStats, SpecDecodingStats, RequestSpecDecodeMetrics) are maps
//     of all their fields, in declaration order.
//   * plain Structs (EngineCoreEvent) are maps; SamplingParams (omit_defaults, not array_like) is a map of its non-default
//     fields only.
//   * numpy arrays (vLLM enc_hook): [dtype.str, shape, data]; data = Ext(3, raw bytes) when nbytes < the zero-copy
//     threshold (VLLM_MSGPACK_ZERO_COPY_THRESHOLD, default 256) or the array is 0-d, else the index of an extra frame.
//   * sets (finished_requests) are arrays in the producer's iteration order. Python's str-set order depends on the
//     process's hash seed, so that field is compared as a set by the tests (spec note in tests/README_core.md).
// Field lists are the installed vLLM's; a vLLM upgrade that adds a field changes the arrays: test_wire.cpp's golden corpus
// (made by gen_wire.py from the installed classes) catches it.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "msgpack.h"

namespace fpr {
namespace v1 {

constexpr int64_t kZeroCopyThreshold = 256;   // VLLM_MSGPACK_ZERO_COPY_THRESHOLD default

// FinishReason (vllm/v1/engine/__init__.py) -- the fpr::Finish values plus REPETITION
enum FinishReason : int { FR_NONE = -1, FR_STOP = 0, FR_LENGTH = 1, FR_ABORT = 2, FR_ERROR = 3, FR_REPETITION = 4 };

struct NdArray {                        // a numpy array as vLLM's enc_hook sends it
  std::string dtype;                    // numpy dtype.str, e.g. "<i8", "<f4"
  std::vector<int64_t> shape;
  std::string data;                     // C-order raw bytes
};
struct LogprobsLists {                  // vllm/v1/outputs.py LogprobsLists (NamedTuple -> array of 4)
  NdArray logprob_token_ids;            // [n, w] int64
  NdArray logprobs;                     // [n, w] float32
  NdArray sampled_token_ranks;          // [n] int64
  bool has_cu = false;                  // cu_num_generated_tokens (None in outputs: slice_request drops it)
  std::vector<int64_t> cu_num_generated_tokens;
};
struct Event { int32_t type; double timestamp; };
struct PrefillStats {
  int64_t num_prompt_tokens = 0, num_computed_tokens = 0, num_cached_tokens = 0, num_local_cached_tokens = 0,
          num_external_cached_tokens = 0, num_cache_creation_tokens = 0;
};
struct RequestSpecDecodeMetrics {
  int64_t num_spec_tokens = 0;
  std::vector<int64_t> histogram;
  int64_t num_draft_tokens = 0;
  std::vector<int64_t> per_step_accepted, per_step_drafted;
};
struct StopReason {                     // int | str | None
  enum Kind : uint8_t { NONE, INT, STR } kind = NONE;
  int64_t i = 0;
  std::string s;
};
using StrMap = std::vector<std::pair<std::string, std::string>>;   // Mapping[str, str], insertion order

struct EngineCoreOutput {
  std::string request_id;
  std::vector<int32_t> new_token_ids;
  std::optional<LogprobsLists> new_logprobs;
  int finish_reason = FR_NONE;
  StopReason stop_reason;
  std::optional<std::vector<Event>> events;
  std::optional<StrMap> trace_headers;
  std::optional<PrefillStats> prefill_stats;
  int64_t num_nans_in_logits = 0;
  std::optional<RequestSpecDecodeMetrics> spec_decode_metrics;
  // not produced by this engine (always None): new_prompt_logprobs_tensors, pooling_output, kv_transfer_params,
  // ec_transfer_params, routed_experts, mm_cache_miss_hashes, new_sampling_mask, prompt_token_id_logprobs
};

struct PrefixCacheStats {               // BaseCacheStats + PrefixCacheStats; hits_by_source stays {} (connector only)
  bool reset = false;
  int64_t requests = 0, queries = 0, hits = 0, preempted_requests = 0, preempted_queries = 0, preempted_hits = 0;
  void record(int64_t num_tokens, int64_t num_hits, bool preempted) {
    if (preempted) {
      preempted_requests += 1; preempted_queries += num_tokens; preempted_hits += num_hits;
      return;
    }
    requests += 1; queries += num_tokens; hits += num_hits;
  }
};
struct SpecDecodingStats {
  int64_t num_spec_tokens = 0, num_drafts = 0, num_draft_tokens = 0, num_accepted_tokens = 0;
  std::vector<int64_t> num_accepted_tokens_per_pos, num_draft_tokens_per_pos;
  static SpecDecodingStats make(int64_t k) {
    SpecDecodingStats s;
    s.num_spec_tokens = k;
    s.num_accepted_tokens_per_pos.assign((size_t)k, 0);
    s.num_draft_tokens_per_pos.assign((size_t)k, 0);
    return s;
  }
  void observe_draft(int64_t num_draft_tokens_, int64_t num_accepted) {
    num_drafts += 1; num_draft_tokens += num_draft_tokens_; num_accepted_tokens += num_accepted;
    for (int64_t i = 0; i < num_accepted && i < (int64_t)num_accepted_tokens_per_pos.size(); ++i) num_accepted_tokens_per_pos[i] += 1;
    for (int64_t i = 0; i < num_draft_tokens_ && i < (int64_t)num_draft_tokens_per_pos.size(); ++i) num_draft_tokens_per_pos[i] += 1;
  }
};
struct SchedulerStats {                 // the fields this engine fills; the rest are their (empty) defaults
  int64_t num_running_reqs = 0, num_waiting_reqs = 0, num_skipped_waiting_reqs = 0;
  // Scheduler._kv_fetch_counts.copy(): the scheduler always sends its 3 stages ({} only in a bare SchedulerStats())
  std::vector<std::pair<std::string, int64_t>> num_kv_fetch_reqs_by_stage;
  int64_t step_counter = 0, current_wave = 0;
  double kv_cache_usage = 0.0;
  PrefixCacheStats prefix_cache_stats;
  std::optional<SpecDecodingStats> spec_decoding_stats;
};
struct UtilityOutput {
  uint64_t call_id = 0;              // uuid1().int >> 64: the full unsigned 64-bit range
  std::optional<std::string> failure_message;
  bool has_result = false;              // UtilityResult present: encoded as [type_info (None), result]
  mp::Val result;
};
struct EngineCoreOutputs {
  int64_t engine_index = 0;
  std::vector<EngineCoreOutput> outputs;
  std::optional<SchedulerStats> scheduler_stats;
  double timestamp = 0.0;               // time.monotonic() of the producer (an input of the core: `now`)
  std::optional<UtilityOutput> utility_output;
  std::optional<std::vector<std::string>> finished_requests;
  std::optional<int64_t> wave_complete, start_wave;
  std::optional<std::string> weight_version;
};

// frames[0] = the message, frames[1..] = zero-copy ndarray buffers (as MsgpackEncoder.encode returns them)
void encode_outputs(const EngineCoreOutputs& o, std::vector<std::string>* frames,
                    int64_t zero_copy_threshold = kZeroCopyThreshold);
void encode_ndarray(mp::Writer& w, const NdArray& a, std::vector<std::string>* frames, int64_t threshold);
void encode_scheduler_stats(mp::Writer& w, const SchedulerStats& s);

// ------------------------------------------------------------------ inputs (frontend -> engine)
enum ReqType : uint8_t { ADD = 0x00, ABORT = 0x01, START_DP_WAVE = 0x02, UTILITY = 0x03 };

enum class SOType : int { NONE = 0, JSON, JSON_OBJECT, REGEX, CHOICE, GRAMMAR, STRUCTURAL_TAG };   // StructuredOutputOptions
struct StructuredOutputs {              // StructuredOutputsParams, as the frontend resolved it (key = (type, spec))
  SOType type = SOType::NONE;
  std::string spec;                     // get_structured_output_key: json dumps / regex / grammar / structural tag string
  std::string backend;                  // _backend ("xgrammar")
  bool disable_any_whitespace = false;
};
struct RepetitionDetection { int64_t max_pattern_size = 0, min_pattern_size = 0, min_count = 0; };
struct SamplingParams {                 // the fields the engine core reads (vllm/sampling_params.py)
  int64_t n = 1;
  double temperature = 1.0, top_p = 1.0, min_p = 0.0;
  double presence_penalty = 0.0, frequency_penalty = 0.0, repetition_penalty = 1.0;
  int64_t top_k = 0;
  bool has_seed = false; int64_t seed = 0;
  std::vector<int32_t> stop_token_ids;  // stop_token_ids (list, order kept)
  bool ignore_eos = false;
  bool has_max_tokens = true; int64_t max_tokens = 16;
  int64_t min_tokens = 0;
  bool has_logprobs = false; int64_t logprobs = 0;          // -1 = full vocab (vLLM convention)
  bool has_prompt_logprobs = false; int64_t prompt_logprobs = 0;
  bool has_eos = false; int32_t eos_token_id = -1;          // _eos_token_id (None when ignore_eos)
  std::vector<int32_t> all_stop_token_ids;                  // _all_stop_token_ids (set; wire order)
  bool has_skip_reading_prefix_cache = false; bool skip_reading_prefix_cache = false;
  std::optional<StructuredOutputs> structured_outputs;
  std::optional<RepetitionDetection> repetition_detection;
  bool has_logit_bias = false, has_allowed_token_ids = false, has_bad_words = false;
  int64_t num_logprobs() const { return has_logprobs ? logprobs : -2; }   // -2 = None
};
struct EngineCoreRequest {
  std::string request_id;
  bool has_prompt_token_ids = false;
  std::vector<int32_t> prompt_token_ids;
  bool has_mm_features = false;         // mm_features (images): carried opaque, see mm_features_raw
  mp::Val mm_features_raw;
  bool has_sampling_params = false;
  SamplingParams sampling_params;
  bool has_pooling_params = false;
  double arrival_time = 0.0;
  bool has_lora_request = false;
  bool has_cache_salt = false; std::string cache_salt;
  bool has_data_parallel_rank = false; int64_t data_parallel_rank = 0;
  bool has_prompt_embeds = false;
  bool has_prompt_is_token_ids = false;
  int64_t client_index = 0, current_wave = 0, priority = 0;
  std::optional<StrMap> trace_headers;
  bool resumable = false;
  bool has_external_req_id = false; std::string external_req_id;
  int reasoning_ended = -1;             // None / False / True = -1 / 0 / 1
  mp::Val reasoning_parser_kwargs;      // dict or None, kept as decoded
  bool abort_immediately = false;
  bool has_session_id = false; std::string session_id;
  bool has_kv_hints = false;
};
struct UtilityCall { int64_t client_index = 0; uint64_t call_id = 0; std::string method; mp::Val args; };

// decode one frontend message: type byte (frames[0][0]) + payload frames (frames[1], aux frames after it)
bool decode_request(const std::string& payload, EngineCoreRequest* out, std::string* err);
bool decode_sampling_params(const mp::Val& v, SamplingParams* out, std::string* err);
bool decode_abort(const std::string& payload, std::vector<std::string>* rids, std::string* err);
bool decode_utility(const std::string& payload, UtilityCall* out, std::string* err);

// Python json.dumps(obj) with its defaults / repr(float): vLLM's key strings for dict-valued schemas and kwargs
bool py_json_dumps(const mp::Val& v, std::string* out);
std::string py_float_repr(double x);

}  // namespace v1
}  // namespace fpr
