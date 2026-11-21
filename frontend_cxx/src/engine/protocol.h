// Frontend <-> EngineCore wire structs for vLLM 0.32.0 V1. See tools/inventory.md section 2.
//   EngineCoreRequest  : msgspec array_like -> positional array, ALL 22 fields (msgspec does not truncate).
//   EngineCoreOutputs  : array; the decoder accepts shorter arrays and ignores extra trailing elements.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "engine/mp.h"
#include "engine/sampling.h"

namespace fe::engine {

// Request-type bytes (EngineCoreRequestType).
inline constexpr char kReqAdd = '\x00';
inline constexpr char kReqAbort = '\x01';
inline constexpr char kReqStartDpWave = '\x02';
inline constexpr char kReqUtility = '\x03';
inline constexpr std::string_view kEngineCoreDead = "ENGINE_CORE_DEAD";

enum class FinishReason : int { Stop = 0, Length = 1, Abort = 2, Error = 3, Repetition = 4 };
const char* finish_reason_str(FinishReason r);  // "stop", "length", "abort", "error", "repetition"

// ---------------------------------------------------------------- multimodal (MultiModalFeatureSpec)
// A tensor goes on the wire as [dtype, shape, data]; data = Ext(3, raw) when nbytes < 256 (CPU), else the
// index of an aux frame (frame 0 is the msgpack body). dtype uses torch names without "torch." ("bfloat16").
struct MmTensor {
  std::string dtype;
  std::vector<int64_t> shape;
  std::string data;  // raw little-endian bytes, C-contiguous
};
// One MultiModalFieldElem: {"data": tensor|nil, "field": [factory ("flat"|"batched"|"shared"), kwargs map]}
struct MmFieldElem {
  std::optional<MmTensor> data;
  std::string factory;
  mp::Value factory_kwargs;  // map of the field dataclass's fields, in declaration order (slices as arrays)
};
struct MmFeature {
  std::vector<std::pair<std::string, MmFieldElem>> item;  // MultiModalKwargsItem, key order preserved
  std::string modality = "image";
  std::string identifier;  // == mm_hash without LoRA
  int64_t offset = 0, length = 0;
  std::optional<MmTensor> is_embed;  // PlaceholderRange.is_embed (bool[length]) or None
  std::optional<std::string> mm_hash;
};

// DeepSeek-V4.1 image items exactly as vLLM's processor + InputProcessor build them (mm_preprocess.py
// _get_mm_fields_config; verified against MsgpackEncoder in tools/golden/engine_request_dump.py):
//   item fields "types" (flat, keep_on_cpu, int64[n_types]), "vit_grid" (batched, keep_on_cpu, int64[2]),
//   "llm_grid" (batched, keep_on_cpu, int64[2]), "patches" (flat, keep_on_cpu false, bfloat16[n, 3, 14, 14]);
//   flat kwargs {keep_on_cpu, slices, dim: 0} where `slices` lists EVERY image of the request
//   ([[[0, n0, None]], [[n0, n0+n1, None]], ...], the batch-level field object); identifier = mm_hash;
//   mm_position {offset, length = n_types, is_embed = bool[length], all true}.
// The item key ORDER is not stable in Python (it follows a str-set iteration, i.e. PYTHONHASHSEED); the engine
// decodes by key. `key_order` picks one (default: types, vit_grid, llm_grid, patches).
struct DsImage {
  const uint16_t* patches_bf16 = nullptr;  // n_patches*3*14*14 bf16 values (raw bits)
  int64_t n_patches = 0;
  int64_t vit_grid[2] = {0, 0}, llm_grid[2] = {0, 0};
  std::vector<int64_t> types;
  std::string mm_hash;
  int64_t offset = 0;  // first placeholder position in the prompt ids
};
std::vector<MmFeature> make_deepseek_v41_image_features(
    const std::vector<DsImage>& images,
    const std::vector<std::string>& key_order = {"types", "vit_grid", "llm_grid", "patches"});

// ---------------------------------------------------------------- EngineCoreRequest
struct EngineRequest {
  std::string request_id;       // internal id: external_req_id + "-" + 8 hex chars (make_internal_request_id)
  std::string external_req_id;  // e.g. "chatcmpl-<16 hex>" or the X-Request-Id header
  std::vector<int32_t> prompt_ids;
  std::vector<MmFeature> mm_features;  // empty -> None
  SamplingParams sampling;
  double arrival_time = 0;  // time.time() at add_request
  std::optional<std::string> cache_salt;
  std::optional<int64_t> data_parallel_rank;
  int64_t priority = 0;
  std::optional<bool> reasoning_ended;              // [17]
  std::optional<mp::Value> reasoning_parser_kwargs;  // [18] {"chat_template_kwargs": {...}}
  std::optional<std::string> session_id;
  int64_t client_index = 0;
};

// Multipart frames exactly as AsyncMPClient sends them (without the ROUTER identity frame):
//   [type byte, msgpack body, aux frames...]
std::vector<std::string> encode_add(const EngineRequest& r);
std::vector<std::string> encode_abort(const std::vector<std::string>& internal_ids);
// UTILITY: (client_index, call_id, method, args tuple)
std::vector<std::string> encode_utility(int64_t client_index, uint64_t call_id, std::string_view method,
                                        const std::vector<mp::Value>& args);

// Inverse of encode_add (frames = [type, body, aux...]); used by tests and tools. Throws mp::DecodeError.
EngineRequest decode_add(const std::vector<std::string>& frames);

// "<external>-<first 8 hex of random_uuid()>"; random_uuid() = low 64 bits of uuid4 -> first char in [89ab].
std::string make_internal_request_id(std::string_view external_req_id);
std::string random_uuid_hex16();  // vllm.utils.random_uuid()
uint64_t uuid1_call_id();         // uuid.uuid1().int >> 64

// ---------------------------------------------------------------- EngineCoreOutputs
struct EngineCoreEvent {
  int type = 0;
  double timestamp = 0;  // engine monotonic
};
struct PrefillStats {
  int64_t num_prompt_tokens = 0, num_computed_tokens = 0, num_cached_tokens = 0;
  int64_t num_local_cached_tokens = 0, num_external_cached_tokens = 0, num_cache_creation_tokens = 0;
};
struct EngineOutput {
  std::string request_id;
  std::vector<int32_t> new_token_ids;
  mp::Value new_logprobs;  // LogprobsLists or nil (unused: the FPR engine rejects logprobs)
  std::optional<FinishReason> finish_reason;
  mp::Value stop_reason;  // nil | int | str
  std::vector<EngineCoreEvent> events;
  mp::Value kv_transfer_params, ec_transfer_params;
  std::optional<PrefillStats> prefill_stats;
  int64_t num_nans_in_logits = 0;
  std::vector<std::string> mm_cache_miss_hashes;
  mp::Value spec_decode_metrics;  // RequestSpecDecodeMetrics map (final output only) or nil
  bool finished() const { return finish_reason.has_value(); }
};
struct UtilityOutput {
  uint64_t call_id = 0;
  std::optional<std::string> failure_message;
  mp::Value result;  // UtilityResult encoded as [type_info|nil, value]; use utility_value()
  const mp::Value* utility_value() const;
};
struct EngineCoreOutputs {
  int64_t engine_index = 0;
  std::vector<EngineOutput> outputs;
  mp::Value scheduler_stats;  // SchedulerStats map or nil
  double timestamp = 0;       // engine monotonic
  std::optional<UtilityOutput> utility_output;
  std::optional<std::vector<std::string>> finished_requests;
  std::optional<int64_t> wave_complete, start_wave;
  std::optional<std::string> weight_version;
};
// frames[0] is the msgpack body; aux frames carry large tensors (only referenced, never needed here).
EngineCoreOutputs decode_outputs(const std::vector<std::string>& frames);
bool is_engine_dead_message(const std::vector<std::string>& frames);

// ---------------------------------------------------------------- EngineCoreReadyResponse (a map)
struct ReadyResponse {
  int64_t max_model_len = 0, num_gpu_blocks = 0, block_size = 0;
  std::optional<std::string> dp_stats_address;
  std::string dtype, vllm_version;
  int64_t world_size = 0, data_parallel_size = 0, tensor_parallel_size = 0, pipeline_parallel_size = 0;
  int64_t decode_context_parallel_size = 0, data_parallel_rank = 0, max_num_seqs = 0, max_num_batched_tokens = 0;
  std::string instance_id;
  bool supports_lora = false;
  int64_t max_loras = 0;
  std::optional<int64_t> mamba_block_size, kv_cache_size_tokens, effective_attention_block_size;
  std::optional<double> kv_cache_max_concurrency;
  mp::Value kv_events_config;
  std::optional<std::string> weight_transfer_backend;
  bool enable_sleep_mode = false, supports_draft_weight_updates = false;
  mp::Value raw;  // the full decoded map
};
ReadyResponse decode_ready_response(std::string_view payload);

// SchedulerStats fields used by /metrics (others stay in the raw map).
struct SchedulerStatsView {
  int64_t num_running_reqs = 0, num_waiting_reqs = 0, num_skipped_waiting_reqs = 0;
  double kv_cache_usage = 0;
  // prefix_cache_stats
  bool prefix_reset = false;
  int64_t prefix_requests = 0, prefix_queries = 0, prefix_hits = 0, prefix_preempted_requests = 0;
  int64_t prefix_preempted_queries = 0, prefix_preempted_hits = 0;
  // spec_decoding_stats (nullopt when absent)
  struct Spec {
    int64_t num_spec_tokens = 0, num_drafts = 0, num_draft_tokens = 0, num_accepted_tokens = 0;
    std::vector<int64_t> num_accepted_tokens_per_pos;
  };
  std::optional<Spec> spec;
};
SchedulerStatsView view_scheduler_stats(const mp::Value& m);

}  // namespace fe::engine
