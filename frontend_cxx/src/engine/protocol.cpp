#include "engine/protocol.h"

#include <chrono>
#include <cstdio>
#include <mutex>
#include <random>

namespace fe::engine {

const char* finish_reason_str(FinishReason r) {
  switch (r) {
    case FinishReason::Stop: return "stop";
    case FinishReason::Length: return "length";
    case FinishReason::Abort: return "abort";
    case FinishReason::Error: return "error";
    case FinishReason::Repetition: return "repetition";
  }
  return "error";
}

// ---------------------------------------------------------------- ids
static std::mt19937_64& rng() {
  thread_local std::mt19937_64 g([] {
    std::random_device rd;
    uint64_t s = (uint64_t(rd()) << 32) ^ rd();
    s ^= uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    return s;
  }());
  return g;
}

std::string random_uuid_hex16() {
  // uuid4().int & (2**64-1): bytes 8..15 of a version-4 uuid; byte 8 carries the RFC 4122 variant (10xxxxxx).
  uint64_t v = rng()();
  v = (v & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

std::string make_internal_request_id(std::string_view external_req_id) {
  std::string s(external_req_id);
  s += '-';
  s += random_uuid_hex16().substr(0, 8);
  return s;
}

uint64_t uuid1_call_id() {
  // uuid1().int >> 64 = time_low(32) | time_mid(16) | version(4)+time_hi(12), from 100-ns ticks since 1582.
  static std::mutex mu;
  static uint64_t last = 0;
  std::lock_guard<std::mutex> lk(mu);
  auto now = std::chrono::system_clock::now().time_since_epoch();
  uint64_t ticks = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count() / 100) +
                   0x01B21DD213814000ULL;
  if (ticks <= last) ticks = last + 1;  // uuid1 guarantees monotonic timestamps within a process
  last = ticks;
  uint64_t time_low = ticks & 0xFFFFFFFFULL, time_mid = (ticks >> 32) & 0xFFFF, time_hi = (ticks >> 48) & 0x0FFF;
  return (time_low << 32) | (time_mid << 16) | (0x1000 | time_hi);
}

// ---------------------------------------------------------------- encoding
namespace {
void pack_tensor(mp::Packer& p, const MmTensor& t, std::vector<std::string>& aux) {
  p.array(3);
  p.str(t.dtype);
  p.array(t.shape.size());
  for (auto d : t.shape) p.integer(d);
  if (t.data.size() < 256) {
    p.ext(3, t.data);
  } else {
    p.integer(int64_t(aux.size() + 1));  // index into [body, aux...]
    aux.push_back(t.data);
  }
}

void pack_mm_feature(mp::Packer& p, const MmFeature& f, std::vector<std::string>& aux) {
  // MultiModalFeatureSpec dataclass -> map {data, modality, identifier, mm_position, mm_hash}
  p.map(5);
  p.str("data");
  p.map(f.item.size());
  for (auto& [key, elem] : f.item) {
    p.str(key);
    p.map(2);
    p.str("data");
    if (elem.data) pack_tensor(p, *elem.data, aux); else p.nil();
    p.str("field");
    p.array(2);
    p.str(elem.factory);
    mp::pack(p, elem.factory_kwargs);
  }
  p.str("modality");
  p.str(f.modality);
  p.str("identifier");
  p.str(f.identifier);
  p.str("mm_position");
  p.map(3);
  p.str("offset");
  p.integer(f.offset);
  p.str("length");
  p.integer(f.length);
  p.str("is_embed");
  if (f.is_embed) pack_tensor(p, *f.is_embed, aux); else p.nil();
  p.str("mm_hash");
  if (f.mm_hash) p.str(*f.mm_hash); else p.nil();
}
}  // namespace

std::vector<std::string> encode_add(const EngineRequest& r) {
  std::vector<std::string> aux;
  mp::Packer p;
  p.buf.reserve(64 + r.prompt_ids.size() * 4);
  p.array(22);
  p.str(r.request_id);                                    // 0
  p.array(r.prompt_ids.size());                           // 1
  for (int32_t t : r.prompt_ids) p.integer(t);
  if (r.mm_features.empty()) {                            // 2
    p.nil();
  } else {
    p.array(r.mm_features.size());
    for (auto& f : r.mm_features) pack_mm_feature(p, f, aux);
  }
  pack_sampling_params(p, r.sampling);                    // 3
  p.nil();                                                // 4 pooling_params
  p.float64(r.arrival_time);                              // 5
  p.nil();                                                // 6 lora_request
  if (r.cache_salt) p.str(*r.cache_salt); else p.nil();   // 7
  if (r.data_parallel_rank) p.integer(*r.data_parallel_rank); else p.nil();  // 8
  p.nil();                                                // 9 prompt_embeds
  p.nil();                                                // 10 prompt_is_token_ids
  p.integer(r.client_index);                              // 11
  p.integer(0);                                           // 12 current_wave
  p.integer(r.priority);                                  // 13
  p.nil();                                                // 14 trace_headers
  p.boolean(false);                                       // 15 resumable
  p.str(r.external_req_id);                               // 16
  if (r.reasoning_ended) p.boolean(*r.reasoning_ended); else p.nil();  // 17
  if (r.reasoning_parser_kwargs) mp::pack(p, *r.reasoning_parser_kwargs); else p.nil();  // 18
  p.boolean(false);                                       // 19 abort_immediately
  if (r.session_id) p.str(*r.session_id); else p.nil();   // 20
  p.nil();                                                // 21 kv_hints
  std::vector<std::string> frames;
  frames.reserve(2 + aux.size());
  frames.emplace_back(1, kReqAdd);
  frames.push_back(std::move(p.buf));
  for (auto& a : aux) frames.push_back(std::move(a));
  return frames;
}

std::vector<std::string> encode_abort(const std::vector<std::string>& ids) {
  mp::Packer p;
  p.array(ids.size());
  for (auto& s : ids) p.str(s);
  return {std::string(1, kReqAbort), std::move(p.buf)};
}

std::vector<std::string> encode_utility(int64_t client_index, uint64_t call_id, std::string_view method,
                                        const std::vector<mp::Value>& args) {
  mp::Packer p;
  p.array(4);
  p.integer(client_index);
  p.uinteger(call_id);
  p.str(method);
  p.array(args.size());
  for (auto& a : args) mp::pack(p, a);
  return {std::string(1, kReqUtility), std::move(p.buf)};
}

// ---------------------------------------------------------------- decoding
namespace {
const mp::Value& at(const mp::Value& arr, size_t i) {
  static const mp::Value nil;
  return i < arr.arr.size() ? arr.arr[i] : nil;
}
int64_t get_i(const mp::Value& m, std::string_view k, int64_t d = 0) {
  auto* v = m.get(k);
  return (v && v->is_num()) ? (v->type == mp::Type::Float ? int64_t(v->f) : v->as_int()) : d;
}
double get_d(const mp::Value& m, std::string_view k, double d = 0) {
  auto* v = m.get(k);
  return (v && v->is_num()) ? v->as_double() : d;
}

EngineOutput decode_one(const mp::Value& a) {
  if (!a.is_array()) throw mp::DecodeError("EngineCoreOutput: expected array");
  EngineOutput o;
  o.request_id = at(a, 0).as_str();
  for (auto& t : at(a, 1).arr) o.new_token_ids.push_back(int32_t(t.as_int()));
  o.new_logprobs = at(a, 2);
  if (!at(a, 5).is_nil()) o.finish_reason = FinishReason(at(a, 5).as_int());
  o.stop_reason = at(a, 6);
  for (auto& e : at(a, 7).arr) {
    EngineCoreEvent ev;
    if (e.is_array()) {  // msgspec.Struct (not array_like) encodes as a map; accept both
      ev.type = int(at(e, 0).as_int());
      ev.timestamp = at(e, 1).as_double();
    } else {
      ev.type = int(get_i(e, "type"));
      ev.timestamp = get_d(e, "timestamp");
    }
    o.events.push_back(ev);
  }
  o.kv_transfer_params = at(a, 8);
  o.ec_transfer_params = at(a, 9);
  if (const auto& ps = at(a, 11); ps.is_map()) {
    PrefillStats s;
    s.num_prompt_tokens = get_i(ps, "num_prompt_tokens");
    s.num_computed_tokens = get_i(ps, "num_computed_tokens");
    s.num_cached_tokens = get_i(ps, "num_cached_tokens");
    s.num_local_cached_tokens = get_i(ps, "num_local_cached_tokens");
    s.num_external_cached_tokens = get_i(ps, "num_external_cached_tokens");
    s.num_cache_creation_tokens = get_i(ps, "num_cache_creation_tokens");
    o.prefill_stats = s;
  }
  if (at(a, 13).is_num()) o.num_nans_in_logits = at(a, 13).as_int();
  for (auto& h : at(a, 14).arr) o.mm_cache_miss_hashes.push_back(h.as_str());
  o.spec_decode_metrics = at(a, 16);
  return o;
}
}  // namespace

const mp::Value* UtilityOutput::utility_value() const {
  if (result.is_array() && result.arr.size() == 2) return &result.arr[1];
  return result.is_nil() ? nullptr : &result;
}

bool is_engine_dead_message(const std::vector<std::string>& frames) {
  return frames.size() == 1 && frames[0] == kEngineCoreDead;
}

EngineCoreOutputs decode_outputs(const std::vector<std::string>& frames) {
  if (frames.empty()) throw mp::DecodeError("EngineCoreOutputs: no frames");
  mp::Value a = mp::unpack(frames[0]);
  if (!a.is_array()) throw mp::DecodeError("EngineCoreOutputs: expected array");
  EngineCoreOutputs out;
  if (at(a, 0).is_num()) out.engine_index = at(a, 0).as_int();
  for (auto& o : at(a, 1).arr) out.outputs.push_back(decode_one(o));
  out.scheduler_stats = at(a, 2);
  if (at(a, 3).is_num()) out.timestamp = at(a, 3).as_double();
  if (const auto& u = at(a, 4); u.is_array()) {
    UtilityOutput uo;
    uo.call_id = at(u, 0).as_uint();
    if (at(u, 1).is_str()) uo.failure_message = at(u, 1).s;
    uo.result = at(u, 2);
    out.utility_output = std::move(uo);
  }
  if (const auto& f = at(a, 5); f.is_array()) {
    std::vector<std::string> ids;
    for (auto& s : f.arr) ids.push_back(s.as_str());
    out.finished_requests = std::move(ids);
  }
  if (at(a, 6).is_num()) out.wave_complete = at(a, 6).as_int();
  if (at(a, 7).is_num()) out.start_wave = at(a, 7).as_int();
  if (at(a, 8).is_str()) out.weight_version = at(a, 8).s;
  return out;
}

ReadyResponse decode_ready_response(std::string_view payload) {
  ReadyResponse r;
  if (payload.empty()) return r;
  r.raw = mp::unpack(payload);
  const mp::Value& m = r.raw;
  auto opt_i = [&](std::string_view k) -> std::optional<int64_t> {
    auto* v = m.get(k);
    if (v && v->is_num()) return v->as_int();
    return std::nullopt;
  };
  auto opt_s = [&](std::string_view k) -> std::optional<std::string> {
    auto* v = m.get(k);
    if (v && v->is_str()) return v->s;
    return std::nullopt;
  };
  auto get_b = [&](std::string_view k) {
    auto* v = m.get(k);
    return v && v->type == mp::Type::Bool && v->b;
  };
  r.max_model_len = get_i(m, "max_model_len");
  r.num_gpu_blocks = get_i(m, "num_gpu_blocks");
  r.block_size = get_i(m, "block_size");
  r.dp_stats_address = opt_s("dp_stats_address");
  r.dtype = opt_s("dtype").value_or("");
  r.vllm_version = opt_s("vllm_version").value_or("");
  r.world_size = get_i(m, "world_size");
  r.data_parallel_size = get_i(m, "data_parallel_size");
  r.tensor_parallel_size = get_i(m, "tensor_parallel_size");
  r.pipeline_parallel_size = get_i(m, "pipeline_parallel_size");
  r.decode_context_parallel_size = get_i(m, "decode_context_parallel_size");
  r.data_parallel_rank = get_i(m, "data_parallel_rank");
  r.max_num_seqs = get_i(m, "max_num_seqs");
  r.max_num_batched_tokens = get_i(m, "max_num_batched_tokens");
  r.instance_id = opt_s("instance_id").value_or("");
  r.supports_lora = get_b("supports_lora");
  r.max_loras = get_i(m, "max_loras");
  r.mamba_block_size = opt_i("mamba_block_size");
  r.kv_cache_size_tokens = opt_i("kv_cache_size_tokens");
  r.effective_attention_block_size = opt_i("effective_attention_block_size");
  if (auto* v = m.get("kv_cache_max_concurrency"); v && v->is_num()) r.kv_cache_max_concurrency = v->as_double();
  if (auto* v = m.get("kv_events_config")) r.kv_events_config = *v;
  r.weight_transfer_backend = opt_s("weight_transfer_backend");
  r.enable_sleep_mode = get_b("enable_sleep_mode");
  r.supports_draft_weight_updates = get_b("supports_draft_weight_updates");
  return r;
}

SchedulerStatsView view_scheduler_stats(const mp::Value& m) {
  SchedulerStatsView s;
  if (!m.is_map()) return s;
  s.num_running_reqs = get_i(m, "num_running_reqs");
  s.num_waiting_reqs = get_i(m, "num_waiting_reqs");
  s.num_skipped_waiting_reqs = get_i(m, "num_skipped_waiting_reqs");
  s.kv_cache_usage = get_d(m, "kv_cache_usage");
  if (auto* p = m.get("prefix_cache_stats"); p && p->is_map()) {
    auto* r = p->get("reset");
    s.prefix_reset = r && r->type == mp::Type::Bool && r->b;
    s.prefix_requests = get_i(*p, "requests");
    s.prefix_queries = get_i(*p, "queries");
    s.prefix_hits = get_i(*p, "hits");
    s.prefix_preempted_requests = get_i(*p, "preempted_requests");
    s.prefix_preempted_queries = get_i(*p, "preempted_queries");
    s.prefix_preempted_hits = get_i(*p, "preempted_hits");
  }
  if (auto* sd = m.get("spec_decoding_stats"); sd && sd->is_map()) {
    SchedulerStatsView::Spec sp;
    sp.num_spec_tokens = get_i(*sd, "num_spec_tokens");
    sp.num_drafts = get_i(*sd, "num_drafts");
    sp.num_draft_tokens = get_i(*sd, "num_draft_tokens");
    sp.num_accepted_tokens = get_i(*sd, "num_accepted_tokens");
    if (auto* pp = sd->get("num_accepted_tokens_per_pos"); pp && pp->is_array())
      for (auto& e : pp->arr) sp.num_accepted_tokens_per_pos.push_back(e.as_int());
    s.spec = sp;
  }
  return s;
}

}  // namespace fe::engine

namespace fe::engine {
EngineRequest decode_add(const std::vector<std::string>& frames) {
  if (frames.size() < 2 || frames[0].size() != 1 || frames[0][0] != kReqAdd)
    throw mp::DecodeError("decode_add: not an ADD message");
  mp::Value a = mp::unpack(frames[1]);
  if (!a.is_array()) throw mp::DecodeError("EngineCoreRequest: expected array");
  auto at = [&](size_t i) -> const mp::Value& {
    static const mp::Value nil;
    return i < a.arr.size() ? a.arr[i] : nil;
  };
  EngineRequest r;
  r.request_id = at(0).as_str();
  for (auto& t : at(1).arr) r.prompt_ids.push_back(int32_t(t.as_int()));
  // mm_features are not decoded (tests only need text requests)
  r.sampling = sampling_from_wire(at(3));
  r.arrival_time = at(5).as_double();
  if (at(7).is_str()) r.cache_salt = at(7).s;
  if (at(8).is_num()) r.data_parallel_rank = at(8).as_int();
  if (at(11).is_num()) r.client_index = at(11).as_int();
  if (at(13).is_num()) r.priority = at(13).as_int();
  if (at(16).is_str()) r.external_req_id = at(16).s;
  if (at(17).type == mp::Type::Bool) r.reasoning_ended = at(17).b;
  if (at(18).is_map()) r.reasoning_parser_kwargs = at(18);
  if (at(20).is_str()) r.session_id = at(20).s;
  return r;
}
}  // namespace fe::engine

namespace fe::engine {
static MmTensor i64_tensor(const std::vector<int64_t>& v) {
  MmTensor t;
  t.dtype = "int64";
  t.shape = {int64_t(v.size())};
  t.data.resize(v.size() * 8);
  for (size_t i = 0; i < v.size(); ++i)
    for (int b = 0; b < 8; ++b) t.data[i * 8 + b] = char((uint64_t(v[i]) >> (8 * b)) & 0xff);
  return t;
}
static mp::Value flat_kwargs(bool keep_on_cpu, const std::vector<int64_t>& sizes) {
  mp::Value kw = mp::Value::make_map();
  kw.set("keep_on_cpu", mp::Value::from_bool(keep_on_cpu));
  mp::Value sl = mp::Value::make_array();  // [(slice(a, b, None),), ...] one per image
  int64_t a = 0;
  for (int64_t n : sizes) {
    mp::Value tup = mp::Value::make_array();
    mp::Value s3 = mp::Value::make_array();
    s3.arr = {mp::Value::from_int(a), mp::Value::from_int(a + n), mp::Value::nil()};
    tup.arr.push_back(s3);
    sl.arr.push_back(tup);
    a += n;
  }
  kw.set("slices", sl);
  kw.set("dim", mp::Value::from_int(0));
  return kw;
}
static mp::Value batched_kwargs() {
  mp::Value kw = mp::Value::make_map();
  kw.set("keep_on_cpu", mp::Value::from_bool(true));
  return kw;
}

std::vector<MmFeature> make_deepseek_v41_image_features(const std::vector<DsImage>& images,
                                                        const std::vector<std::string>& key_order) {
  std::vector<int64_t> np, nt;
  for (auto& im : images) {
    np.push_back(im.n_patches);
    nt.push_back(int64_t(im.types.size()));
  }
  mp::Value kw_patches = flat_kwargs(false, np), kw_types = flat_kwargs(true, nt);
  std::vector<MmFeature> out;
  for (auto& im : images) {
    MmFeature f;
    for (auto& key : key_order) {
      MmFieldElem e;
      if (key == "types") {
        e.data = i64_tensor(im.types);
        e.factory = "flat";
        e.factory_kwargs = kw_types;
      } else if (key == "vit_grid" || key == "llm_grid") {
        const int64_t* g = key == "vit_grid" ? im.vit_grid : im.llm_grid;
        e.data = i64_tensor({g[0], g[1]});
        e.factory = "batched";
        e.factory_kwargs = batched_kwargs();
      } else if (key == "patches") {
        MmTensor p;
        p.dtype = "bfloat16";
        p.shape = {im.n_patches, 3, 14, 14};
        size_t n = size_t(im.n_patches) * 3 * 14 * 14;
        p.data.resize(n * 2);
        for (size_t i = 0; i < n; ++i) {
          p.data[2 * i] = char(im.patches_bf16[i] & 0xff);
          p.data[2 * i + 1] = char(im.patches_bf16[i] >> 8);
        }
        e.data = std::move(p);
        e.factory = "flat";
        e.factory_kwargs = kw_patches;
      } else {
        continue;
      }
      f.item.emplace_back(key, std::move(e));
    }
    f.modality = "image";
    f.identifier = im.mm_hash;
    f.mm_hash = im.mm_hash;
    f.offset = im.offset;
    f.length = int64_t(im.types.size());
    MmTensor emb;
    emb.dtype = "bool";
    emb.shape = {f.length};
    emb.data.assign(size_t(f.length), char(1));
    f.is_embed = std::move(emb);
    out.push_back(std::move(f));
  }
  return out;
}
}  // namespace fe::engine
