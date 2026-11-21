// v1wire.cc -- see v1wire.h
#include "v1wire.h"

namespace fpr {
namespace v1 {

namespace {

void key(mp::Writer& w, const char* k) { w.str(k, std::strlen(k)); }

void int_list(mp::Writer& w, const std::vector<int64_t>& v) {
  w.arr((uint32_t)v.size());
  for (int64_t x : v) w.i64(x);
}

void str_map(mp::Writer& w, const StrMap& m) {
  w.map((uint32_t)m.size());
  for (const auto& kv : m) {
    w.str(kv.first);
    w.str(kv.second);
  }
}

void prefix_stats(mp::Writer& w, const PrefixCacheStats& p) {
  w.map(8);
  key(w, "reset"); w.boolean(p.reset);
  key(w, "requests"); w.i64(p.requests);
  key(w, "queries"); w.i64(p.queries);
  key(w, "hits"); w.i64(p.hits);
  key(w, "preempted_requests"); w.i64(p.preempted_requests);
  key(w, "preempted_queries"); w.i64(p.preempted_queries);
  key(w, "preempted_hits"); w.i64(p.preempted_hits);
  key(w, "hits_by_source"); w.map(0);
}

void spec_stats(mp::Writer& w, const SpecDecodingStats& s) {
  w.map(6);
  key(w, "num_spec_tokens"); w.i64(s.num_spec_tokens);
  key(w, "num_drafts"); w.i64(s.num_drafts);
  key(w, "num_draft_tokens"); w.i64(s.num_draft_tokens);
  key(w, "num_accepted_tokens"); w.i64(s.num_accepted_tokens);
  key(w, "num_accepted_tokens_per_pos"); int_list(w, s.num_accepted_tokens_per_pos);
  key(w, "num_draft_tokens_per_pos"); int_list(w, s.num_draft_tokens_per_pos);
}

void logprobs(mp::Writer& w, const LogprobsLists& l, std::vector<std::string>* frames, int64_t th) {
  w.arr(4);
  encode_ndarray(w, l.logprob_token_ids, frames, th);
  encode_ndarray(w, l.logprobs, frames, th);
  encode_ndarray(w, l.sampled_token_ranks, frames, th);
  if (l.has_cu) int_list(w, l.cu_num_generated_tokens);
  else w.nil();
}

void output(mp::Writer& w, const EngineCoreOutput& o, std::vector<std::string>* frames, int64_t th) {
  w.arr(18);
  w.str(o.request_id);                                            // request_id
  w.arr((uint32_t)o.new_token_ids.size());                         // new_token_ids
  for (int32_t t : o.new_token_ids) w.i64(t);
  if (o.new_logprobs) logprobs(w, *o.new_logprobs, frames, th);   // new_logprobs
  else w.nil();
  w.nil();                                                         // new_prompt_logprobs_tensors
  w.nil();                                                         // pooling_output
  if (o.finish_reason >= 0) w.i64(o.finish_reason);                // finish_reason
  else w.nil();
  switch (o.stop_reason.kind) {                                    // stop_reason
    case StopReason::NONE: w.nil(); break;
    case StopReason::INT: w.i64(o.stop_reason.i); break;
    case StopReason::STR: w.str(o.stop_reason.s); break;
  }
  if (o.events) {                                                  // events: [{type, timestamp}]
    w.arr((uint32_t)o.events->size());
    for (const auto& e : *o.events) {
      w.map(2);
      key(w, "type"); w.i64(e.type);
      key(w, "timestamp"); w.f64(e.timestamp);
    }
  } else {
    w.nil();
  }
  w.nil();                                                         // kv_transfer_params
  w.nil();                                                         // ec_transfer_params
  if (o.trace_headers) str_map(w, *o.trace_headers);               // trace_headers
  else w.nil();
  if (o.prefill_stats) {                                           // prefill_stats
    const auto& p = *o.prefill_stats;
    w.map(6);
    key(w, "num_prompt_tokens"); w.i64(p.num_prompt_tokens);
    key(w, "num_computed_tokens"); w.i64(p.num_computed_tokens);
    key(w, "num_cached_tokens"); w.i64(p.num_cached_tokens);
    key(w, "num_local_cached_tokens"); w.i64(p.num_local_cached_tokens);
    key(w, "num_external_cached_tokens"); w.i64(p.num_external_cached_tokens);
    key(w, "num_cache_creation_tokens"); w.i64(p.num_cache_creation_tokens);
  } else {
    w.nil();
  }
  w.nil();                                                         // routed_experts
  w.i64(o.num_nans_in_logits);                                     // num_nans_in_logits
  w.nil();                                                         // mm_cache_miss_hashes
  w.nil();                                                         // new_sampling_mask
  if (o.spec_decode_metrics) {                                     // spec_decode_metrics
    const auto& m = *o.spec_decode_metrics;
    w.map(5);
    key(w, "num_spec_tokens"); w.i64(m.num_spec_tokens);
    key(w, "histogram"); int_list(w, m.histogram);
    key(w, "num_draft_tokens"); w.i64(m.num_draft_tokens);
    key(w, "per_step_accepted"); int_list(w, m.per_step_accepted);
    key(w, "per_step_drafted"); int_list(w, m.per_step_drafted);
  } else {
    w.nil();
  }
  w.nil();                                                         // prompt_token_id_logprobs
}

}  // namespace

void encode_ndarray(mp::Writer& w, const NdArray& a, std::vector<std::string>* frames, int64_t threshold) {
  w.arr(3);
  w.str(a.dtype);
  int_list(w, a.shape);
  if (a.shape.empty() || (int64_t)a.data.size() < threshold) {
    w.ext(3, a.data.data(), a.data.size());                        // CUSTOM_TYPE_RAW_VIEW
  } else {
    w.i64((int64_t)frames->size());                                // index of the backing buffer frame
    frames->push_back(a.data);
  }
}

void encode_scheduler_stats(mp::Writer& w, const SchedulerStats& s) {
  w.map(18);
  key(w, "num_running_reqs"); w.i64(s.num_running_reqs);
  key(w, "num_waiting_reqs"); w.i64(s.num_waiting_reqs);
  key(w, "num_skipped_waiting_reqs"); w.i64(s.num_skipped_waiting_reqs);
  key(w, "num_kv_fetch_reqs_by_stage");
  w.map((uint32_t)s.num_kv_fetch_reqs_by_stage.size());
  for (const auto& kv : s.num_kv_fetch_reqs_by_stage) {
    w.str(kv.first);
    w.i64(kv.second);
  }
  key(w, "step_counter"); w.i64(s.step_counter);
  key(w, "current_wave"); w.i64(s.current_wave);
  key(w, "kv_cache_usage"); w.f64(s.kv_cache_usage);
  key(w, "iteration_details"); w.nil();
  key(w, "prefix_cache_stats"); prefix_stats(w, s.prefix_cache_stats);
  key(w, "connector_prefix_cache_stats"); w.nil();
  key(w, "kv_cache_eviction_events"); w.arr(0);
  key(w, "spec_decoding_stats");
  if (s.spec_decoding_stats) spec_stats(w, *s.spec_decoding_stats);
  else w.nil();
  key(w, "kv_connector_stats"); w.nil();
  key(w, "ec_connector_stats"); w.nil();
  key(w, "waiting_lora_adapters"); w.map(0);
  key(w, "running_lora_adapters"); w.map(0);
  key(w, "cudagraph_stats"); w.nil();
  key(w, "perf_stats"); w.nil();
}

void encode_outputs(const EngineCoreOutputs& o, std::vector<std::string>* frames, int64_t th) {
  frames->clear();
  frames->emplace_back();                                          // frames[0]: filled at the end
  mp::Writer w;
  w.arr(9);
  w.i64(o.engine_index);                                           // engine_index
  w.arr((uint32_t)o.outputs.size());                               // outputs
  for (const auto& x : o.outputs) output(w, x, frames, th);
  if (o.scheduler_stats) encode_scheduler_stats(w, *o.scheduler_stats);   // scheduler_stats
  else w.nil();
  w.f64(o.timestamp);                                              // timestamp
  if (o.utility_output) {                                          // utility_output: [call_id, failure, result]
    const auto& u = *o.utility_output;
    w.arr(3);
    w.u64(u.call_id);
    if (u.failure_message) w.str(*u.failure_message);
    else w.nil();
    if (u.has_result) {
      w.arr(2);
      w.nil();                                                     // type info (None without insecure serialization)
      mp::write_val(w, u.result);
    } else {
      w.nil();
    }
  } else {
    w.nil();
  }
  if (o.finished_requests) {                                       // finished_requests (a set)
    w.arr((uint32_t)o.finished_requests->size());
    for (const auto& r : *o.finished_requests) w.str(r);
  } else {
    w.nil();
  }
  if (o.wave_complete) w.i64(*o.wave_complete);
  else w.nil();
  if (o.start_wave) w.i64(*o.start_wave);
  else w.nil();
  if (o.weight_version) w.str(*o.weight_version);
  else w.nil();
  (*frames)[0] = std::move(w.buf);
}

// ------------------------------------------------------------------ decode
namespace {

bool fail(std::string* err, const std::string& m) {
  if (err) *err = m;
  return false;
}

bool int_vec(const mp::Val& v, std::vector<int32_t>* out) {
  if (v.kind != mp::Val::ARR) return false;
  out->clear();
  out->reserve(v.a.size());
  for (const auto& x : v.a) {
    if (!x.is_int()) return false;
    out->push_back((int32_t)x.as_int());
  }
  return true;
}

bool str_of(const mp::Val& v, std::string* out) {
  if (v.kind != mp::Val::STR) return false;
  *out = v.s;
  return true;
}

void put_utf8(std::string* o, uint32_t cp) {
  char b[16];
  std::snprintf(b, sizeof b, "\\u%04x", (unsigned)(cp & 0xffff));
  o->append(b);
}

}  // namespace

namespace {

void dump_str(const std::string& s, std::string* o) {
  o->push_back('"');
  size_t i = 0;
  while (i < s.size()) {
    uint8_t c = (uint8_t)s[i];
    uint32_t cp;
    int n;
    if (c < 0x80) { cp = c; n = 1; }
    else if ((c >> 5) == 6) { cp = c & 0x1f; n = 2; }
    else if ((c >> 4) == 14) { cp = c & 0x0f; n = 3; }
    else { cp = c & 0x07; n = 4; }
    for (int k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | ((uint8_t)s[i + k] & 0x3f);
    i += n;
    switch (cp) {
      case '"': o->append("\\\""); continue;
      case '\\': o->append("\\\\"); continue;
      case '\n': o->append("\\n"); continue;
      case '\r': o->append("\\r"); continue;
      case '\t': o->append("\\t"); continue;
      case '\b': o->append("\\b"); continue;
      case '\f': o->append("\\f"); continue;
      default: break;
    }
    if (cp < 0x20 || cp >= 0x7f) {
      if (cp >= 0x10000) {
        uint32_t x = cp - 0x10000;
        put_utf8(o, 0xd800 | (x >> 10));
        put_utf8(o, 0xdc00 | (x & 0x3ff));
      } else {
        put_utf8(o, cp);
      }
    } else {
      o->push_back((char)cp);
    }
  }
  o->push_back('"');
}

}  // namespace

bool py_json_dumps(const mp::Val& v, std::string* o) {
  switch (v.kind) {
    case mp::Val::NIL: o->append("null"); return true;
    case mp::Val::BOOL: o->append(v.b ? "true" : "false"); return true;
    case mp::Val::INT: o->append(std::to_string(v.i)); return true;
    case mp::Val::UINT: o->append(std::to_string(v.u)); return true;
    case mp::Val::F64: o->append(py_float_repr(v.f)); return true;
    case mp::Val::STR: dump_str(v.s, o); return true;
    case mp::Val::ARR:
      o->push_back('[');
      for (size_t j = 0; j < v.a.size(); ++j) {
        if (j) o->append(", ");
        if (!py_json_dumps(v.a[j], o)) return false;
      }
      o->push_back(']');
      return true;
    case mp::Val::MAP:
      o->push_back('{');
      for (size_t j = 0; j + 1 < v.a.size(); j += 2) {
        if (j) o->append(", ");
        if (v.a[j].kind != mp::Val::STR) return false;   // (json.dumps would stringify int keys: not used by schemas)
        dump_str(v.a[j].s, o);
        o->append(": ");
        if (!py_json_dumps(v.a[j + 1], o)) return false;
      }
      o->push_back('}');
      return true;
    default: return false;
  }
}

bool decode_sampling_params(const mp::Val& v, SamplingParams* sp, std::string* err) {
  if (v.kind != mp::Val::MAP) return fail(err, "sampling_params: not a map");
  *sp = SamplingParams();
  for (size_t j = 0; j + 1 < v.a.size(); j += 2) {
    if (v.a[j].kind != mp::Val::STR) return fail(err, "sampling_params: non-string key");
    const std::string& k = v.a[j].s;
    const mp::Val& x = v.a[j + 1];
    auto num = [&](double* d) { if (x.kind == mp::Val::F64 || x.is_int()) *d = x.as_f64(); };
    if (k == "n") { if (x.is_int()) sp->n = x.as_int(); }
    else if (k == "temperature") num(&sp->temperature);
    else if (k == "top_p") num(&sp->top_p);
    else if (k == "min_p") num(&sp->min_p);
    else if (k == "presence_penalty") num(&sp->presence_penalty);
    else if (k == "frequency_penalty") num(&sp->frequency_penalty);
    else if (k == "repetition_penalty") num(&sp->repetition_penalty);
    else if (k == "top_k") { if (x.is_int()) sp->top_k = x.as_int(); }
    else if (k == "seed") { sp->has_seed = x.is_int(); if (x.is_int()) sp->seed = x.as_int(); }
    else if (k == "stop_token_ids") { if (!x.is_nil() && !int_vec(x, &sp->stop_token_ids)) return fail(err, "stop_token_ids"); }
    else if (k == "ignore_eos") { if (x.kind == mp::Val::BOOL) sp->ignore_eos = x.b; }
    else if (k == "max_tokens") { sp->has_max_tokens = x.is_int(); if (x.is_int()) sp->max_tokens = x.as_int(); }
    else if (k == "min_tokens") { if (x.is_int()) sp->min_tokens = x.as_int(); }
    else if (k == "logprobs") { sp->has_logprobs = x.is_int(); if (x.is_int()) sp->logprobs = x.as_int(); }
    else if (k == "prompt_logprobs") { sp->has_prompt_logprobs = x.is_int(); if (x.is_int()) sp->prompt_logprobs = x.as_int(); }
    else if (k == "_eos_token_id") { sp->has_eos = x.is_int(); if (x.is_int()) sp->eos_token_id = (int32_t)x.as_int(); }
    else if (k == "_all_stop_token_ids") { if (!x.is_nil() && !int_vec(x, &sp->all_stop_token_ids)) return fail(err, "_all_stop_token_ids"); }
    else if (k == "skip_reading_prefix_cache") {
      sp->has_skip_reading_prefix_cache = x.kind == mp::Val::BOOL;
      if (x.kind == mp::Val::BOOL) sp->skip_reading_prefix_cache = x.b;
    }
    else if (k == "logit_bias") sp->has_logit_bias = !x.is_nil() && !(x.kind == mp::Val::MAP && x.a.empty());
    else if (k == "allowed_token_ids") sp->has_allowed_token_ids = !x.is_nil() && !(x.kind == mp::Val::ARR && x.a.empty());
    else if (k == "bad_words") sp->has_bad_words = sp->has_bad_words || (!x.is_nil() && !(x.kind == mp::Val::ARR && x.a.empty()));
    else if (k == "repetition_detection") {
      if (x.kind == mp::Val::MAP) {
        RepetitionDetection r;
        if (auto* p = x.get("max_pattern_size")) r.max_pattern_size = p->as_int();
        if (auto* p = x.get("min_pattern_size")) r.min_pattern_size = p->as_int();
        if (auto* p = x.get("min_count")) r.min_count = p->as_int();
        sp->repetition_detection = r;
      }
    } else if (k == "structured_outputs") {
      if (x.kind != mp::Val::MAP) continue;
      // get_structured_output_key order: json, json_object, regex, choice, grammar, structural_tag
      StructuredOutputs so;
      const mp::Val* f;
      if ((f = x.get("json")) && !f->is_nil()) {
        so.type = SOType::JSON;
        if (f->kind == mp::Val::STR) so.spec = f->s;
        else if (!py_json_dumps(*f, &so.spec)) return fail(err, "structured_outputs.json: not json-dumpable");
      } else if ((f = x.get("json_object")) && f->kind == mp::Val::BOOL && f->b) {
        so.type = SOType::JSON_OBJECT;
      } else if ((f = x.get("regex")) && f->kind == mp::Val::STR) {
        so.type = SOType::REGEX; so.spec = f->s;
      } else if ((f = x.get("choice")) && !f->is_nil()) {
        so.type = SOType::CHOICE;
        if (f->kind == mp::Val::STR) so.spec = f->s;
        else if (!py_json_dumps(*f, &so.spec)) return fail(err, "structured_outputs.choice");
      } else if ((f = x.get("grammar")) && f->kind == mp::Val::STR) {
        so.type = SOType::GRAMMAR; so.spec = f->s;
      } else if ((f = x.get("structural_tag")) && f->kind == mp::Val::STR) {
        so.type = SOType::STRUCTURAL_TAG; so.spec = f->s;
      }
      if ((f = x.get("_backend")) && f->kind == mp::Val::STR) so.backend = f->s;
      if ((f = x.get("disable_any_whitespace")) && f->kind == mp::Val::BOOL) so.disable_any_whitespace = f->b;
      if (so.type != SOType::NONE) sp->structured_outputs = so;
    }
    // other fields (stop strings, detokenize, output_kind, ...) are frontend-side: ignored here
  }
  return true;
}

bool decode_request(const std::string& payload, EngineCoreRequest* r, std::string* err) {
  mp::Reader rd(payload);
  mp::Val v;
  if (!rd.read(&v)) return fail(err, rd.error());
  if (v.kind != mp::Val::ARR || v.a.size() < 9) return fail(err, "EngineCoreRequest: not an array of >= 9 fields");
  *r = EngineCoreRequest();
  const auto& a = v.a;
  if (!str_of(a[0], &r->request_id)) return fail(err, "request_id");
  r->has_prompt_token_ids = !a[1].is_nil();
  if (r->has_prompt_token_ids && !int_vec(a[1], &r->prompt_token_ids)) return fail(err, "prompt_token_ids");
  r->has_mm_features = !a[2].is_nil() && !(a[2].kind == mp::Val::ARR && a[2].a.empty());
  if (r->has_mm_features) r->mm_features_raw = a[2];
  r->has_sampling_params = !a[3].is_nil();
  if (r->has_sampling_params && !decode_sampling_params(a[3], &r->sampling_params, err)) return false;
  r->has_pooling_params = !a[4].is_nil();
  if (a[5].kind == mp::Val::F64 || a[5].is_int()) r->arrival_time = a[5].as_f64();
  r->has_lora_request = !a[6].is_nil();
  r->has_cache_salt = str_of(a[7], &r->cache_salt);
  r->has_data_parallel_rank = a[8].is_int();
  if (r->has_data_parallel_rank) r->data_parallel_rank = a[8].as_int();
  auto at = [&](size_t i) -> const mp::Val* { return i < a.size() ? &a[i] : nullptr; };
  const mp::Val* x;
  if ((x = at(9))) r->has_prompt_embeds = !x->is_nil();
  if ((x = at(10))) r->has_prompt_is_token_ids = !x->is_nil();
  if ((x = at(11)) && x->is_int()) r->client_index = x->as_int();
  if ((x = at(12)) && x->is_int()) r->current_wave = x->as_int();
  if ((x = at(13)) && x->is_int()) r->priority = x->as_int();
  if ((x = at(14)) && x->kind == mp::Val::MAP) {
    StrMap m;
    for (size_t j = 0; j + 1 < x->a.size(); j += 2) m.emplace_back(x->a[j].s, x->a[j + 1].s);
    r->trace_headers = m;
  }
  if ((x = at(15)) && x->kind == mp::Val::BOOL) r->resumable = x->b;
  if ((x = at(16))) r->has_external_req_id = str_of(*x, &r->external_req_id);
  if ((x = at(17)) && x->kind == mp::Val::BOOL) r->reasoning_ended = x->b ? 1 : 0;
  if ((x = at(18))) r->reasoning_parser_kwargs = *x;
  if ((x = at(19)) && x->kind == mp::Val::BOOL) r->abort_immediately = x->b;
  if ((x = at(20))) r->has_session_id = str_of(*x, &r->session_id);
  if ((x = at(21))) r->has_kv_hints = !x->is_nil();
  return true;
}

bool decode_abort(const std::string& payload, std::vector<std::string>* rids, std::string* err) {
  mp::Reader rd(payload);
  mp::Val v;
  if (!rd.read(&v)) return fail(err, rd.error());
  rids->clear();
  if (v.kind == mp::Val::STR) {
    rids->push_back(v.s);
    return true;
  }
  if (v.kind != mp::Val::ARR) return fail(err, "ABORT: not a list");
  for (const auto& x : v.a) {
    if (x.kind != mp::Val::STR) return fail(err, "ABORT: non-string id");
    rids->push_back(x.s);
  }
  return true;
}

bool decode_utility(const std::string& payload, UtilityCall* u, std::string* err) {
  mp::Reader rd(payload);
  mp::Val v;
  if (!rd.read(&v)) return fail(err, rd.error());
  if (v.kind != mp::Val::ARR || v.a.size() != 4) return fail(err, "UTILITY: not (client_index, call_id, method, args)");
  bool cid = v.a[1].kind == mp::Val::UINT || (v.a[1].kind == mp::Val::INT && v.a[1].i >= 0);   // up to 2^64 - 1
  if (!v.a[0].is_int() || !cid || v.a[2].kind != mp::Val::STR) return fail(err, "UTILITY: bad header");
  u->client_index = v.a[0].as_int();
  u->call_id = v.a[1].kind == mp::Val::UINT ? v.a[1].u : (uint64_t)v.a[1].i;
  u->method = v.a[2].s;
  u->args = v.a[3];
  return true;
}

}  // namespace v1
}  // namespace fpr
