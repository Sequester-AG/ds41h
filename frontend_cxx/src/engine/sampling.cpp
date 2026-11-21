#include "engine/sampling.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <unordered_set>

#include "engine/textutil.h"

namespace fe::engine {

static std::string ve_what(const std::string& m, const std::string& p, const std::string& v) {
  std::string extras;
  if (!p.empty()) extras += "parameter=" + p;
  if (!v.empty()) {
    if (!extras.empty()) extras += ", ";
    extras += "value=" + v;
  }
  return extras.empty() ? m : m + " (" + extras + ")";
}
ValidationError::ValidationError(std::string msg, std::string param, std::string val)
    : std::runtime_error(ve_what(msg, param, val)), message(std::move(msg)), parameter(std::move(param)),
      value(std::move(val)) {}

std::string py_float_repr(double d) {
  if (std::isnan(d)) return "nan";
  if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
  if (d == 0) return std::signbit(d) ? "-0.0" : "0.0";
  // Python repr ('r' format): shortest round-trip digits; fixed notation when 1e-4 <= |d| < 1e16.
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
  std::string sci(buf, r.ptr);  // e.g. "1.2345e+02" or "5e-05"
  size_t epos = sci.find('e');
  std::string mant = sci.substr(0, epos);
  int exp = std::stoi(sci.substr(epos + 1));
  bool neg = mant[0] == '-';
  if (neg) mant.erase(0, 1);
  std::string digits;
  for (char c : mant)
    if (c != '.') digits += c;
  std::string out;
  if (exp >= -4 && exp < 16) {
    int point = exp + 1;  // position of decimal point within digits
    if (point <= 0) {
      out = "0." + std::string(size_t(-point), '0') + digits;
    } else if (size_t(point) >= digits.size()) {
      out = digits + std::string(size_t(point) - digits.size(), '0') + ".0";
    } else {
      out = digits.substr(0, size_t(point)) + "." + digits.substr(size_t(point));
    }
  } else {
    out = digits.substr(0, 1);
    if (digits.size() > 1) out += "." + digits.substr(1);
    char eb[16];
    std::snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
    out += eb;
  }
  return neg ? "-" + out : out;
}

static std::string py_list(const std::vector<int64_t>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(v[i]);
  }
  return s + "]";
}

const std::vector<std::string>& SamplingParams::stop_strings() const {
  static const std::vector<std::string> empty;
  return stop ? *stop : empty;
}

// ------------------------------------------------------------------ CPython set emulation
// setobject.c (3.12): LINEAR_PROBES 9, PERTURB_SHIFT 5, PySet_MINSIZE 8, resize when fill*5 >= mask*3
// to the smallest power of two > used*4 (used*2 above 50000). hash(int) == int for 0 <= v < 2**61-1.
std::vector<int64_t> python_int_set_order(const std::vector<int64_t>& inserts) {
  struct Slot {
    bool used = false;
    int64_t key = 0;
  };
  const size_t LINEAR_PROBES = 9;
  std::vector<Slot> table(8);
  size_t mask = 7, fill = 0;
  auto hash_of = [](int64_t v) -> uint64_t {
    // CPython: hash(-1) == -2; other small negatives hash to themselves.
    if (v == -1) return uint64_t(int64_t(-2));
    const int64_t M = (int64_t(1) << 61) - 1;
    if (v >= 0) return uint64_t(v % M);
    int64_t h = -((-v) % M);
    if (h == -1) h = -2;
    return uint64_t(h);
  };
  auto insert_clean = [&](std::vector<Slot>& t, size_t m, int64_t key) {
    uint64_t h = hash_of(key);
    size_t perturb = size_t(h);
    size_t i = size_t(h) & m;
    while (true) {
      if (!t[i].used) { t[i] = {true, key}; return; }
      if (i + LINEAR_PROBES <= m) {
        for (size_t j = 1; j <= LINEAR_PROBES; ++j)
          if (!t[i + j].used) { t[i + j] = {true, key}; return; }
      }
      perturb >>= 5;
      i = (i * 5 + 1 + perturb) & m;
    }
  };
  for (int64_t key : inserts) {
    uint64_t h = hash_of(key);
    size_t perturb = size_t(h);
    size_t i = size_t(h) & mask;
    bool found = false;
    size_t target = SIZE_MAX;
    while (true) {
      size_t probes = (i + LINEAR_PROBES <= mask) ? LINEAR_PROBES : 0;
      size_t k = i;
      for (size_t c = 0; c <= probes; ++c, ++k) {
        if (!table[k].used) { target = k; break; }
        if (table[k].key == key) { found = true; break; }
      }
      if (found || target != SIZE_MAX) break;
      perturb >>= 5;
      i = (i * 5 + 1 + perturb) & mask;
    }
    if (found) continue;
    table[target] = {true, key};
    ++fill;
    if (fill * 5 < mask * 3) continue;
    size_t used = fill;  // no deletions -> fill == used
    size_t minused = used > 50000 ? used * 2 : used * 4;
    size_t newsize = 8;
    while (newsize <= minused) newsize <<= 1;
    std::vector<Slot> nt(newsize);
    for (auto& s : table)
      if (s.used) insert_clean(nt, newsize - 1, s.key);
    table.swap(nt);
    mask = newsize - 1;
  }
  std::vector<int64_t> out;
  for (auto& s : table)
    if (s.used) out.push_back(s.key);
  return out;
}

// ------------------------------------------------------------------ helpers on the body
namespace {
const mp::Value* field(const mp::Value& body, std::string_view k) {
  const mp::Value* v = body.get(k);
  return (v && !v->is_nil()) ? v : nullptr;
}
std::optional<double> opt_double(const mp::Value& b, std::string_view k) {
  auto* v = field(b, k);
  if (!v) return std::nullopt;
  if (v->type == mp::Type::Bool) return v->b ? 1.0 : 0.0;
  return v->as_double();
}
std::optional<int64_t> opt_int(const mp::Value& b, std::string_view k) {
  auto* v = field(b, k);
  if (!v) return std::nullopt;
  if (v->type == mp::Type::Float) return int64_t(v->f);
  return v->as_int();
}
bool get_bool(const mp::Value& b, std::string_view k, bool dflt) {
  auto* v = field(b, k);
  if (!v) return dflt;
  if (v->type == mp::Type::Bool) return v->b;
  if (v->is_num()) return v->as_double() != 0;
  return dflt;
}
std::optional<std::vector<int64_t>> opt_int_list(const mp::Value& b, std::string_view k) {
  auto* v = field(b, k);
  if (!v || !v->is_array()) return std::nullopt;
  std::vector<int64_t> out;
  for (auto& e : v->arr) out.push_back(e.type == mp::Type::Float ? int64_t(e.f) : e.as_int());
  return out;
}
// StopParam: str | list[str] | None
std::optional<std::vector<std::string>> opt_stop(const mp::Value& b, std::string_view k, bool* present) {
  auto* v = field(b, k);
  *present = v != nullptr;
  if (!v) return std::nullopt;
  if (v->is_str()) return std::vector<std::string>{v->s};
  std::vector<std::string> out;
  if (v->is_array())
    for (auto& e : v->arr) out.push_back(e.as_str());
  return out;
}

std::optional<StructuredOutputs> structured_from(const mp::Value& body) {
  // extract_structured_outputs(): explicit structured_outputs, else response_format.
  if (auto* so = field(body, "structured_outputs"); so && so->is_map()) {
    StructuredOutputs o;
    if (auto* x = field(*so, "json")) o.json = *x;
    if (auto* x = field(*so, "regex")) o.regex = x->as_str();
    if (auto* x = field(*so, "choice"); x && x->is_array()) {
      std::vector<std::string> c;
      for (auto& e : x->arr) c.push_back(e.as_str());
      o.choice = c;
    }
    if (auto* x = field(*so, "grammar")) o.grammar = x->as_str();
    if (auto* x = field(*so, "json_object")) o.json_object = x->b;
    o.disable_any_whitespace = get_bool(*so, "disable_any_whitespace", false);
    o.disable_additional_properties = get_bool(*so, "disable_additional_properties", false);
    if (auto* x = field(*so, "whitespace_pattern")) o.whitespace_pattern = x->as_str();
    if (auto* x = field(*so, "structural_tag")) o.structural_tag = x->as_str();
    return o;
  }
  if (auto* rf = field(body, "response_format"); rf && rf->is_map()) {
    auto* t = field(*rf, "type");
    std::string type = t ? t->as_str() : "";
    StructuredOutputs o;
    if (type == "json_object") { o.json_object = true; return o; }
    if (type == "json_schema") {
      if (auto* js = field(*rf, "json_schema"))
        if (auto* sch = field(*js, "schema")) { o.json = *sch; return o; }
      return std::nullopt;
    }
    // "text" and others: no constraint. structural_tag response_format is serialised by the API layer.
  }
  return std::nullopt;
}
}  // namespace

int64_t get_max_tokens(int64_t max_model_len, std::optional<int64_t> requested, int64_t input_length) {
  if (max_model_len < input_length)
    throw PromptTooLongError("Input length (" + std::to_string(input_length) +
                             ") exceeds model's maximum context length (" + std::to_string(max_model_len) + ").");
  int64_t m = max_model_len - input_length;
  if (requested) m = std::min(m, *requested);
  return m;
}

void post_init(SamplingParams& sp) {
  if (sp.temperature && *sp.temperature > 0 && *sp.temperature < 1e-2) sp.temperature = 1e-2;
  if (sp.seed && *sp.seed == -1) sp.seed.reset();
  if (sp.thinking_token_budget) {
    if (*sp.thinking_token_budget == -1) sp.thinking_token_budget.reset();
    else if (*sp.thinking_token_budget < 0)
      throw ValidationError("`thinking_token_budget` must be a non-negative integer or -1 for unlimited.",
                            "thinking_token_budget", std::to_string(*sp.thinking_token_budget));
  }
  if (!sp.stop) sp.stop.emplace();
  if (!sp.stop_token_ids) {
    sp.stop_token_ids.emplace();
  } else {  // list(dict.fromkeys(...))
    std::vector<int64_t> d;
    std::unordered_set<int64_t> seen;
    for (auto v : *sp.stop_token_ids)
      if (seen.insert(v).second) d.push_back(v);
    sp.stop_token_ids = d;
  }
  if (!sp.bad_words) {
    sp.bad_words.emplace();
  } else {
    std::vector<std::string> d;
    for (auto& w : *sp.bad_words)
      if (std::find(d.begin(), d.end(), w) == d.end()) d.push_back(w);
    sp.bad_words = d;
  }
  if (!sp.stop->empty() && !sp.include_stop_str_in_output) {
    size_t mx = 0;
    for (auto& s : *sp.stop) mx = std::max(mx, utf8_codepoints(s));
    sp.output_text_buffer_length = int64_t(mx) - 1;
  }
  // _verify_args
  if (sp.n < 1) throw ValidationError("n must be at least 1, got " + std::to_string(sp.n) + ".");
  if (sp.n > 16384)
    throw ValidationError("n must be at most 16384, got " + std::to_string(sp.n) +
                          ". To increase this limit, set the VLLM_MAX_N_SEQUENCES environment variable.");
  double pp = sp.presence_penalty.value_or(0.0), fp = sp.frequency_penalty.value_or(0.0);
  double rp = sp.repetition_penalty.value_or(1.0), t = sp.temperature.value_or(1.0);
  double top_p = sp.top_p.value_or(1.0), min_p = sp.min_p.value_or(0.0);
  if (!(pp >= -2.0 && pp <= 2.0))
    throw ValidationError("presence_penalty must be in [-2, 2], got " + py_float_repr(pp) + ".");
  if (!(fp >= -2.0 && fp <= 2.0))
    throw ValidationError("frequency_penalty must be in [-2, 2], got " + py_float_repr(fp) + ".");
  if (!std::isfinite(rp))
    throw ValidationError("repetition_penalty must be a finite number, got " + py_float_repr(rp) + ".");
  if (rp <= 0.0)
    throw ValidationError("repetition_penalty must be greater than zero, got " + py_float_repr(rp) + ".");
  if (!std::isfinite(t))
    throw ValidationError("temperature must be a finite number, got " + py_float_repr(t) + ".", "temperature",
                          py_float_repr(t));
  if (t < 0.0)
    throw ValidationError("temperature must be non-negative, got " + py_float_repr(t) + ".", "temperature",
                          py_float_repr(t));
  if (t > 2.0)
    throw ValidationError("temperature must be in [0, 2], got " + py_float_repr(t) + ".", "temperature",
                          py_float_repr(t));
  if (!(top_p > 0.0 && top_p <= 1.0))
    throw ValidationError("top_p must be in (0, 1], got " + py_float_repr(top_p) + ".", "top_p", py_float_repr(top_p));
  if (sp.top_k < -1)
    throw ValidationError("top_k must be 0 (disable), or at least 1, got " + std::to_string(sp.top_k) + ".");
  if (!(min_p >= 0.0 && min_p <= 1.0))
    throw ValidationError("min_p must be in [0, 1], got " + py_float_repr(min_p) + ".");
  if (sp.max_tokens && *sp.max_tokens < 1)
    throw ValidationError("max_tokens must be at least 1, got " + std::to_string(*sp.max_tokens) + ".", "max_tokens",
                          std::to_string(*sp.max_tokens));
  if (sp.min_tokens < 0)
    throw ValidationError("min_tokens must be greater than or equal to 0, got " + std::to_string(sp.min_tokens) + ".");
  if (sp.max_tokens && sp.min_tokens > *sp.max_tokens)
    throw ValidationError("min_tokens must be less than or equal to max_tokens=" + std::to_string(*sp.max_tokens) +
                          ", got " + std::to_string(sp.min_tokens) + ".");
  if (sp.stream_interval && *sp.stream_interval < 1)
    throw ValidationError("stream_interval must be at least 1, got " + std::to_string(*sp.stream_interval) + ".",
                          "stream_interval", std::to_string(*sp.stream_interval));
  if (sp.logprobs && *sp.logprobs != -1 && *sp.logprobs < 0)
    throw ValidationError("logprobs must be non-negative or -1, got " + std::to_string(*sp.logprobs) + ".", "logprobs",
                          std::to_string(*sp.logprobs));
  if (sp.prompt_logprobs && *sp.prompt_logprobs != -1 && *sp.prompt_logprobs < 0)
    throw ValidationError("prompt_logprobs must be non-negative or -1, got " + std::to_string(*sp.prompt_logprobs) + ".",
                          "prompt_logprobs", std::to_string(*sp.prompt_logprobs));
  for (auto& s : *sp.stop)
    if (s.empty()) throw ValidationError("stop cannot contain an empty string.");
  if (!sp.stop->empty() && !sp.detokenize)
    throw ValidationError("stop strings are only supported when detokenize is True. Set detokenize=True to use stop.");
  for (auto& w : *sp.bad_words)
    if (w.empty()) throw ValidationError("bad_words cannot contain an empty string. Got bad_words=[...]");
  if (t < 1e-5) {  // greedy: reset to the class-default objects (-> omitted on the wire)
    sp.top_p.reset();
    sp.top_k = 0;
    sp.min_p.reset();
    if (sp.n > 1) throw ValidationError("n must be 1 when using greedy sampling, got " + std::to_string(sp.n) + ".");
  }
  sp.all_stop_token_ids = python_int_set_order(*sp.stop_token_ids);
  if (!sp.skip_reading_prefix_cache) sp.skip_reading_prefix_cache = sp.prompt_logprobs.has_value();
}

void validate_token_ids(std::span<const int32_t> ids, const SamplingContext& ctx, const std::string& parameter,
                        const std::optional<std::string>& oov_id) {
  if (ids.empty()) return;
  if (oov_id) throw ValidationError("Token id " + *oov_id + " is out of vocabulary", parameter);
  auto [mn, mx] = std::minmax_element(ids.begin(), ids.end());
  // The tokenizer's max id or the model's vocab size - 1, whichever is larger (base.py:452).
  int64_t limit = std::max(ctx.max_token_id, ctx.vocab_size - 1);
  if (*mn < 0) throw ValidationError("Token id " + std::to_string(*mn) + " is out of vocabulary", parameter);
  if (*mx > limit) throw ValidationError("Token id " + std::to_string(*mx) + " is out of vocabulary", parameter);
}

void process_for_engine(SamplingParams& sp, int64_t prompt_len, const SamplingContext& ctx,
                        const InputChecks* checks) {
  // params.verify(...)
  int64_t max_lp = ctx.max_logprobs == -1 ? ctx.vocab_size : ctx.max_logprobs;
  if (sp.logprobs && *sp.logprobs != 0) {
    int64_t k = *sp.logprobs == -1 ? ctx.vocab_size : *sp.logprobs;
    if (k > max_lp)
      throw ValidationError("Requested sample logprobs of " + std::to_string(k) +
                                ", which is greater than max allowed: " + std::to_string(max_lp),
                            "logprobs", std::to_string(k));
  }
  if (sp.prompt_logprobs && *sp.prompt_logprobs != 0) {
    int64_t k = *sp.prompt_logprobs == -1 ? ctx.vocab_size : *sp.prompt_logprobs;
    if (k > max_lp)
      throw ValidationError("Requested prompt logprobs of " + std::to_string(k) +
                                ", which is greater than max allowed: " + std::to_string(max_lp),
                            "prompt_logprobs", std::to_string(k));
  }
  auto oov = [&](const std::vector<int64_t>& ids) {
    std::vector<int64_t> bad;
    for (auto i : ids)
      if (i < 0 || i >= ctx.vocab_size) bad.push_back(i);
    return bad;
  };
  if (sp.logit_bias && !sp.logit_bias->empty()) {
    std::vector<int64_t> ids;
    for (auto& [k, v] : *sp.logit_bias) ids.push_back(k);
    auto bad = oov(ids);
    if (!bad.empty())
      throw ValidationError("token_id(s) " + py_list(bad) +
                                " in logit_bias contain out-of-vocab token ids. Vocabulary size: " +
                                std::to_string(ctx.vocab_size),
                            "logit_bias", py_list(bad));
  }
  if (!sp.stop_token_ids->empty()) {
    auto bad = oov(*sp.stop_token_ids);
    if (!bad.empty())
      throw ValidationError("token_id(s) " + py_list(bad) +
                                " in stop_token_ids contain out-of-vocab token ids. Vocabulary size: " +
                                std::to_string(ctx.vocab_size),
                            "stop_token_ids", py_list(bad));
  }
  if (sp.allowed_token_ids) {
    if (sp.allowed_token_ids->empty())
      throw ValidationError("allowed_token_ids is not None and empty!", "allowed_token_ids", "[]");
    auto bad = oov(*sp.allowed_token_ids);
    if (!bad.empty())
      throw ValidationError("allowed_token_ids contains out-of-vocab token id!", "allowed_token_ids", py_list(bad));
  }
  if (ctx.speculative && (sp.min_p.value_or(0.0) > 1e-5 || (sp.logit_bias && !sp.logit_bias->empty())))
    throw ValidationError("The min_p and logit_bias sampling parameters are not yet supported with speculative decoding.");
  // apply_watermarking
  bool wm = false;  // no watermark config in our deployment
  sp.watermarking_skipped = sp.watermarking_skipped ||
                            (ctx.watermark_configured && !(sp.watermarking && !*sp.watermarking) && !wm);
  sp.watermarking = wm;
  if (checks) {
    // process_inputs: data_parallel_rank range (no parameter)
    if (auto r = checks->data_parallel_rank; r && !(*r >= 0 && *r < ctx.dp_num_ranks)) {
      std::string rs = checks->data_parallel_rank_text.empty() ? std::to_string(*r) : checks->data_parallel_rank_text;
      throw ValidationError("data_parallel_rank " + rs + " is out of range [0, " + std::to_string(ctx.dp_num_ranks) +
                            ").");
    }
    // _validate_model_inputs -> _validate_prompt_len (the renderer already enforced max_model_len), then
    // validate_token_ids(prompt_ids) (input_processor.py:647)
    if (prompt_len == 0) throw ValidationError("The decoder prompt cannot be empty");
    validate_token_ids(checks->prompt_ids, ctx, {}, checks->prompt_oov_id);
    if (!(sp.routed_experts_prompt_start >= 0 && sp.routed_experts_prompt_start <= prompt_len))
      throw ValidationError("routed_experts_prompt_start must be between 0 and the prompt length (" +
                                std::to_string(prompt_len) + "), inclusive.",
                            "routed_experts_prompt_start", std::to_string(sp.routed_experts_prompt_start));
  }
  if (!sp.max_tokens) {
    sp.max_tokens = ctx.max_model_len - prompt_len;
    if (sp.min_tokens > *sp.max_tokens)
      throw ValidationError("min_tokens must be less than or equal to max_tokens=" + std::to_string(*sp.max_tokens) +
                            ", got " + std::to_string(sp.min_tokens) + ".");
  }
  // update_from_generation_config({eos_token_id: 1, ...}, eos)
  if (!sp.ignore_eos) sp.eos_token_id = ctx.eos_token_id;
  {
    std::vector<int64_t> ins = *sp.stop_token_ids;
    ins.push_back(ctx.eos_token_id);
    sp.all_stop_token_ids = python_int_set_order(ins);
  }
  // update_from_tokenizer: bad words
  if (!sp.bad_words->empty()) {
    if (!ctx.encode) throw ValidationError("bad_words are not supported by this frontend.", "bad_words");
    std::vector<std::vector<int64_t>> out;
    for (auto& w : *sp.bad_words) {
      for (int add_space = 0; add_space < 2; ++add_space) {
        std::string s = w;
        size_t k = 0;
        while (k < s.size() && (s[k] == ' ' || s[k] == '\t' || s[k] == '\n' || s[k] == '\r')) ++k;
        s = (add_space ? " " : "") + s.substr(k);
        auto ids = ctx.encode(s, ctx.encode_ctx);
        if (ids.empty()) {
          if (!add_space)
            throw ValidationError("bad_words entries must tokenize to at least one token.", "bad_words");
          continue;
        }
        if (!add_space || (ids[0] != out.back()[0] && ids.size() == out.back().size())) out.push_back(ids);
      }
    }
    sp.bad_words_token_ids = out;
  }
}

SamplingParams sampling_from_request(Endpoint ep, const mp::Value& body, int64_t prompt_len,
                                     const SamplingContext& ctx, const InputChecks* checks) {
  SamplingParams sp;
  bool stop_present = false;
  if (ep == Endpoint::Responses) {
    int64_t dflt = get_max_tokens(ctx.max_model_len, opt_int(body, "max_output_tokens"), prompt_len);
    auto mot = opt_int(body, "max_output_tokens");
    sp.max_tokens = mot ? std::min(*mot, dflt) : dflt;
    sp.temperature = opt_double(body, "temperature").value_or(1.0);
    sp.top_p = opt_double(body, "top_p").value_or(1.0);
    sp.top_k = opt_int(body, "top_k").value_or(0);
    sp.min_p = opt_double(body, "min_p").value_or(0.0);
    sp.repetition_penalty = opt_double(body, "repetition_penalty").value_or(1.0);
    sp.presence_penalty = opt_double(body, "presence_penalty").value_or(0.0);
    sp.frequency_penalty = opt_double(body, "frequency_penalty").value_or(0.0);
    auto st = opt_stop(body, "stop", &stop_present);
    sp.stop = st ? *st : std::vector<std::string>{};
    sp.stop_token_ids = opt_int_list(body, "stop_token_ids");
    sp.seed = opt_int(body, "seed");
    sp.ignore_eos = get_bool(body, "ignore_eos", false);
    if (auto* inc = field(body, "include"); inc && inc->is_array()) {
      for (auto& e : inc->arr)
        if (e.is_str() && e.s == "message.output_text.logprobs") sp.logprobs = opt_int(body, "top_logprobs").value_or(0);
    }
    sp.output_kind = get_bool(body, "stream", false) ? OutputKind::Delta : OutputKind::FinalOnly;
    sp.structured_outputs = structured_from(body);
    sp.skip_special_tokens = get_bool(body, "skip_special_tokens", true);
    sp.include_stop_str_in_output = get_bool(body, "include_stop_str_in_output", false);
    if (auto* x = field(body, "vllm_xargs"); x && x->is_map()) sp.extra_args = *x;
    else sp.extra_args = mp::Value::make_map();  // Responses passes extra_args={} (not `or None`)
    if (auto* w = field(body, "watermarking")) sp.watermarking = w->b;
  } else {
    bool chat = ep == Endpoint::Chat;
    std::optional<int64_t> req_max;
    if (chat) {
      req_max = opt_int(body, "max_completion_tokens");
      if (!req_max) req_max = opt_int(body, "max_tokens");
    } else {
      auto* mt = body.get("max_tokens");
      req_max = mt ? (mt->is_nil() ? std::nullopt : std::optional<int64_t>(mt->as_int())) : std::optional<int64_t>(16);
    }
    int64_t mt = get_max_tokens(ctx.max_model_len, req_max, prompt_len);
    sp.n = opt_int(body, "n").value_or(1);
    sp.repetition_penalty = opt_double(body, "repetition_penalty").value_or(1.0);
    sp.presence_penalty = opt_double(body, "presence_penalty").value_or(0.0);
    sp.frequency_penalty = opt_double(body, "frequency_penalty").value_or(0.0);
    sp.temperature = opt_double(body, "temperature").value_or(1.0);
    sp.top_p = opt_double(body, "top_p").value_or(1.0);
    sp.top_k = opt_int(body, "top_k").value_or(0);
    sp.min_p = opt_double(body, "min_p").value_or(0.0);
    if (auto* w = field(body, "watermarking")) sp.watermarking = w->b;
    sp.seed = opt_int(body, "seed");
    sp.stop = opt_stop(body, "stop", &stop_present);  // absent: pydantic default []; null: None -> [] in post_init
    sp.stop_token_ids = opt_int_list(body, "stop_token_ids");
    auto ltids = opt_int_list(body, "logprob_token_ids");
    if (ltids && ltids->empty()) ltids.reset();
    if (chat) {
      bool lp = get_bool(body, "logprobs", false);
      int64_t tlp = opt_int(body, "top_logprobs").value_or(0);
      if (lp && !ltids) sp.logprobs = tlp;
      sp.prompt_logprobs = opt_int(body, "prompt_logprobs");
      if (!sp.prompt_logprobs && get_bool(body, "echo", false)) sp.prompt_logprobs = tlp;
    } else {
      if (!ltids) sp.logprobs = opt_int(body, "logprobs");
      sp.prompt_logprobs = opt_int(body, "prompt_logprobs");
      if (!sp.prompt_logprobs && get_bool(body, "echo", false)) sp.prompt_logprobs = opt_int(body, "logprobs");
      if (get_bool(body, "echo", false) && req_max && *req_max == 0) mt = 1;
    }
    sp.logprob_token_ids = ltids;
    sp.ignore_eos = get_bool(body, "ignore_eos", false);
    sp.max_tokens = mt;
    sp.min_tokens = opt_int(body, "min_tokens").value_or(0);
    sp.skip_special_tokens = get_bool(body, "skip_special_tokens", true);
    sp.spaces_between_special_tokens = get_bool(body, "spaces_between_special_tokens", true);
    sp.include_stop_str_in_output = get_bool(body, "include_stop_str_in_output", false);
    sp.output_kind = get_bool(body, "stream", false) ? OutputKind::Delta : OutputKind::FinalOnly;
    sp.stream_interval = opt_int(body, "stream_interval");
    sp.structured_outputs = structured_from(body);
    if (auto* lb = field(body, "logit_bias"); lb && lb->is_map()) {
      std::vector<std::pair<int64_t, double>> v;
      for (auto& [k, val] : lb->kv) {
        int64_t id;
        const std::string& ks = k.is_str() ? k.s : std::to_string(k.as_int());
        auto r = std::from_chars(ks.data(), ks.data() + ks.size(), id);
        if (r.ec != std::errc() || r.ptr != ks.data() + ks.size())
          throw ValidationError("logit_bias contains key(s) that cannot be converted to integer token IDs: ['" + ks + "']",
                                "logit_bias", "['" + ks + "']");
        v.emplace_back(id, std::min(100.0, std::max(-100.0, val.as_double())));
      }
      sp.logit_bias = v;
    }
    if (auto* bw = field(body, "bad_words"); bw && bw->is_array()) {
      std::vector<std::string> w;
      for (auto& e : bw->arr) w.push_back(e.as_str());
      sp.bad_words = w;
    } else {
      sp.bad_words = std::vector<std::string>{};
    }
    sp.thinking_token_budget = opt_int(body, "thinking_token_budget");
    sp.allowed_token_ids = opt_int_list(body, "allowed_token_ids");
    mp::Value extra = mp::Value::make_map();
    if (auto* x = field(body, "vllm_xargs"); x && x->is_map()) extra = *x;
    if (auto* x = field(body, "kv_transfer_params"); x && x->is_map() && !x->kv.empty()) extra.set("kv_transfer_params", *x);
    if (auto* x = field(body, "ec_transfer_params"); x && x->is_map() && !x->kv.empty()) extra.set("ec_transfer_params", *x);
    if (!extra.kv.empty()) sp.extra_args = extra;
    if (auto* rd = field(body, "repetition_detection"); rd && rd->is_map()) {
      mp::Value m = mp::Value::make_map();
      m.set("max_pattern_size", mp::Value::from_int(opt_int(*rd, "max_pattern_size").value_or(0)));
      m.set("min_pattern_size", mp::Value::from_int(opt_int(*rd, "min_pattern_size").value_or(0)));
      m.set("min_count", mp::Value::from_int(opt_int(*rd, "min_count").value_or(0)));
      sp.repetition_detection = m;
    }
    sp.routed_experts_prompt_start = opt_int(body, "routed_experts_prompt_start").value_or(0);
  }
  sp.skip_clone = true;
  post_init(sp);
  check_phys_caps(sp, ctx);   // sampling_guard: right after Python's to_sampling_params, before InputProcessor
  process_for_engine(sp, prompt_len, ctx, checks);
  return sp;
}

std::vector<std::string> sampling_features(const SamplingParams& sp) {
  std::vector<std::string> u;
  if (sp.top_p && *sp.top_p < 1.0) u.push_back("top_p");
  if (sp.top_k > 0) u.push_back("top_k");
  if (sp.min_p && *sp.min_p > 0.0) u.push_back("min_p");
  if (sp.presence_penalty.value_or(0.0) != 0.0 || sp.frequency_penalty.value_or(0.0) != 0.0 ||
      sp.repetition_penalty.value_or(1.0) != 1.0)
    u.push_back("penalties");
  if (sp.logit_bias && !sp.logit_bias->empty()) u.push_back("logit_bias");
  if ((sp.bad_words && !sp.bad_words->empty()) || (sp.bad_words_token_ids && !sp.bad_words_token_ids->empty()))
    u.push_back("bad_words");
  if (sp.allowed_token_ids && !sp.allowed_token_ids->empty()) u.push_back("allowed_token_ids");
  if (sp.logprobs) u.push_back("logprobs");
  if (sp.prompt_logprobs) u.push_back("prompt_logprobs");
  if (sp.min_tokens > 0) u.push_back("min_tokens");
  std::sort(u.begin(), u.end());
  return u;
}

void check_phys_caps(const SamplingParams& sp, const SamplingContext& ctx) {
  if (!ctx.phys_caps) return;
  const auto& caps = *ctx.phys_caps;
  std::vector<std::string> fields;
  for (const auto& f : sampling_features(sp)) {
    if (std::find(caps.begin(), caps.end(), f) != caps.end()) continue;
    std::string field = f;   // sampling_guard._field_of: penalties names the first non-default penalty
    if (f == "penalties") {
      if (sp.presence_penalty.value_or(0.0) != 0.0) field = "presence_penalty";
      else if (sp.frequency_penalty.value_or(0.0) != 0.0) field = "frequency_penalty";
      else field = "repetition_penalty";
    }
    fields.push_back(field);
  }
  if (fields.empty()) return;
  std::vector<std::string> sc = caps;
  std::sort(sc.begin(), sc.end());
  std::string m;
  for (size_t i = 0; i < fields.size(); ++i) m += (i ? ", " : "") + fields[i];
  m += fields.size() == 1 ? " is" : " are";
  m += " not supported by this server's sampler (supported: temperature, seed, stop, max_tokens";
  for (const auto& c : sc) m += ", " + c;
  m += ")";
  throw CapsError(m, fields[0]);
}

// ------------------------------------------------------------------ wire encoding
static void pack_str_list(mp::Packer& p, const std::vector<std::string>& v) {
  p.array(v.size());
  for (auto& s : v) p.str(s);
}
static void pack_int_list(mp::Packer& p, const std::vector<int64_t>& v) {
  p.array(v.size());
  for (auto x : v) p.integer(x);
}

static void pack_structured(mp::Packer& p, const StructuredOutputs& o) {
  p.map(11);
  p.str("json");
  if (o.json) mp::pack(p, *o.json); else p.nil();
  p.str("regex");
  if (o.regex) p.str(*o.regex); else p.nil();
  p.str("choice");
  if (o.choice) pack_str_list(p, *o.choice); else p.nil();
  p.str("grammar");
  if (o.grammar) p.str(*o.grammar); else p.nil();
  p.str("json_object");
  if (o.json_object) p.boolean(*o.json_object); else p.nil();
  p.str("disable_any_whitespace");
  p.boolean(o.disable_any_whitespace);
  p.str("disable_additional_properties");
  p.boolean(o.disable_additional_properties);
  p.str("whitespace_pattern");
  if (o.whitespace_pattern) p.str(*o.whitespace_pattern); else p.nil();
  p.str("structural_tag");
  if (o.structural_tag) p.str(*o.structural_tag); else p.nil();
  p.str("_backend");
  if (o.backend) p.str(*o.backend); else p.nil();
  p.str("_backend_was_auto");
  p.boolean(o.backend_was_auto);
}

void pack_sampling_params(mp::Packer& out, const SamplingParams& sp) {
  mp::Packer b;  // body, then prefix the map header once the count is known
  size_t n = 0;
  auto key = [&](const char* k) { b.str(k); ++n; };
  if (sp.n != 1) { key("n"); b.integer(sp.n); }
  if (sp.presence_penalty) { key("presence_penalty"); b.float64(*sp.presence_penalty); }
  if (sp.frequency_penalty) { key("frequency_penalty"); b.float64(*sp.frequency_penalty); }
  if (sp.repetition_penalty) { key("repetition_penalty"); b.float64(*sp.repetition_penalty); }
  if (sp.temperature) { key("temperature"); b.float64(*sp.temperature); }
  if (sp.watermarking) { key("watermarking"); b.boolean(*sp.watermarking); }
  if (sp.top_p) { key("top_p"); b.float64(*sp.top_p); }
  if (sp.top_k != 0) { key("top_k"); b.integer(sp.top_k); }
  if (sp.min_p) { key("min_p"); b.float64(*sp.min_p); }
  if (sp.seed) { key("seed"); b.integer(*sp.seed); }
  if (sp.stop) { key("stop"); pack_str_list(b, *sp.stop); }
  if (sp.stop_token_ids) { key("stop_token_ids"); pack_int_list(b, *sp.stop_token_ids); }
  if (sp.ignore_eos) { key("ignore_eos"); b.boolean(true); }
  if (!sp.max_tokens) { key("max_tokens"); b.nil(); }
  else if (*sp.max_tokens != 16) { key("max_tokens"); b.integer(*sp.max_tokens); }
  if (sp.min_tokens != 0) { key("min_tokens"); b.integer(sp.min_tokens); }
  if (sp.logprobs) { key("logprobs"); b.integer(*sp.logprobs); }
  if (sp.prompt_logprobs) { key("prompt_logprobs"); b.integer(*sp.prompt_logprobs); }
  if (sp.logprob_token_ids) { key("logprob_token_ids"); pack_int_list(b, *sp.logprob_token_ids); }
  if (!sp.detokenize) { key("detokenize"); b.boolean(false); }
  if (!sp.skip_special_tokens) { key("skip_special_tokens"); b.boolean(false); }
  if (!sp.spaces_between_special_tokens) { key("spaces_between_special_tokens"); b.boolean(false); }
  if (sp.include_stop_str_in_output) { key("include_stop_str_in_output"); b.boolean(true); }
  if (sp.output_kind != OutputKind::Cumulative) { key("output_kind"); b.integer(int(sp.output_kind)); }
  if (sp.stream_interval) { key("stream_interval"); b.integer(*sp.stream_interval); }
  if (sp.skip_clone) { key("skip_clone"); b.boolean(true); }
  if (sp.output_text_buffer_length != 0) { key("output_text_buffer_length"); b.integer(sp.output_text_buffer_length); }
  if (sp.eos_token_id) { key("_eos_token_id"); b.integer(*sp.eos_token_id); }
  if (!sp.all_stop_token_ids.empty()) { key("_all_stop_token_ids"); pack_int_list(b, sp.all_stop_token_ids); }
  if (sp.watermarking_skipped) { key("_watermarking_skipped"); b.boolean(true); }
  if (sp.structured_outputs) { key("structured_outputs"); pack_structured(b, *sp.structured_outputs); }
  if (sp.logit_bias) {
    key("logit_bias");
    b.map(sp.logit_bias->size());
    for (auto& [k, v] : *sp.logit_bias) { b.integer(k); b.float64(v); }
  }
  if (sp.allowed_token_ids) { key("allowed_token_ids"); pack_int_list(b, *sp.allowed_token_ids); }
  if (sp.extra_args) { key("extra_args"); mp::pack(b, *sp.extra_args); }
  if (sp.bad_words) { key("bad_words"); pack_str_list(b, *sp.bad_words); }
  if (sp.bad_words_token_ids) {
    key("_bad_words_token_ids");
    b.array(sp.bad_words_token_ids->size());
    for (auto& v : *sp.bad_words_token_ids) pack_int_list(b, v);
  }
  if (sp.skip_reading_prefix_cache) { key("skip_reading_prefix_cache"); b.boolean(*sp.skip_reading_prefix_cache); }
  if (sp.thinking_token_budget) { key("thinking_token_budget"); b.integer(*sp.thinking_token_budget); }
  if (sp.repetition_detection) { key("repetition_detection"); mp::pack(b, *sp.repetition_detection); }
  if (sp.routed_experts_prompt_start != 0) { key("routed_experts_prompt_start"); b.integer(sp.routed_experts_prompt_start); }
  if (sp.trace_decode_token_ids) { key("trace_decode_token_ids"); pack_int_list(b, *sp.trace_decode_token_ids); }
  // msgspec sizes a Struct's map header for its declared field count (43 > 15): always map16, even when
  // fewer fields survive omit_defaults (observed: de 00 0e for 14 fields).
  out.buf.push_back(char(0xde));
  out.buf.push_back(char(n >> 8));
  out.buf.push_back(char(n & 0xff));
  out.raw(b.buf);
}

}  // namespace fe::engine

namespace fe::engine {
// ------------------------------------------------------------------ wire decoding
SamplingParams sampling_from_wire(const mp::Value& m) {
  SamplingParams sp;
  sp.max_tokens = 16;
  if (!m.is_map()) return sp;
  auto ints = [](const mp::Value& v) {
    std::vector<int64_t> o;
    for (auto& e : v.arr) o.push_back(e.as_int());
    return o;
  };
  auto strs = [](const mp::Value& v) {
    std::vector<std::string> o;
    for (auto& e : v.arr) o.push_back(e.as_str());
    return o;
  };
  for (auto& [kv, v] : m.kv) {
    const std::string& k = kv.s;
    if (k == "n") sp.n = v.as_int();
    else if (k == "presence_penalty") sp.presence_penalty = v.as_double();
    else if (k == "frequency_penalty") sp.frequency_penalty = v.as_double();
    else if (k == "repetition_penalty") sp.repetition_penalty = v.as_double();
    else if (k == "temperature") sp.temperature = v.as_double();
    else if (k == "watermarking") { if (!v.is_nil()) sp.watermarking = v.b; }
    else if (k == "top_p") sp.top_p = v.as_double();
    else if (k == "top_k") sp.top_k = v.as_int();
    else if (k == "min_p") sp.min_p = v.as_double();
    else if (k == "seed") { if (!v.is_nil()) sp.seed = v.as_int(); }
    else if (k == "stop") { if (v.is_str()) sp.stop = std::vector<std::string>{v.s}; else if (v.is_array()) sp.stop = strs(v); }
    else if (k == "stop_token_ids") { if (v.is_array()) sp.stop_token_ids = ints(v); }
    else if (k == "ignore_eos") sp.ignore_eos = v.b;
    else if (k == "max_tokens") { if (v.is_nil()) sp.max_tokens.reset(); else sp.max_tokens = v.as_int(); }
    else if (k == "min_tokens") sp.min_tokens = v.as_int();
    else if (k == "logprobs") { if (!v.is_nil()) sp.logprobs = v.as_int(); }
    else if (k == "prompt_logprobs") { if (!v.is_nil()) sp.prompt_logprobs = v.as_int(); }
    else if (k == "logprob_token_ids") { if (v.is_array()) sp.logprob_token_ids = ints(v); }
    else if (k == "detokenize") sp.detokenize = v.b;
    else if (k == "skip_special_tokens") sp.skip_special_tokens = v.b;
    else if (k == "spaces_between_special_tokens") sp.spaces_between_special_tokens = v.b;
    else if (k == "include_stop_str_in_output") sp.include_stop_str_in_output = v.b;
    else if (k == "output_kind") sp.output_kind = OutputKind(v.as_int());
    else if (k == "stream_interval") { if (!v.is_nil()) sp.stream_interval = v.as_int(); }
    else if (k == "skip_clone") sp.skip_clone = v.b;
    else if (k == "output_text_buffer_length") sp.output_text_buffer_length = v.as_int();
    else if (k == "_eos_token_id") { if (!v.is_nil()) sp.eos_token_id = v.as_int(); }
    else if (k == "_all_stop_token_ids") { if (v.is_array()) sp.all_stop_token_ids = ints(v); }
    else if (k == "_watermarking_skipped") sp.watermarking_skipped = v.b;
    else if (k == "structured_outputs") {
      if (v.is_map()) {
        StructuredOutputs o;
        if (auto* x = v.get("json"); x && !x->is_nil()) o.json = *x;
        if (auto* x = v.get("regex"); x && x->is_str()) o.regex = x->s;
        if (auto* x = v.get("choice"); x && x->is_array()) o.choice = strs(*x);
        if (auto* x = v.get("grammar"); x && x->is_str()) o.grammar = x->s;
        if (auto* x = v.get("json_object"); x && !x->is_nil()) o.json_object = x->b;
        if (auto* x = v.get("disable_any_whitespace")) o.disable_any_whitespace = x->b;
        if (auto* x = v.get("disable_additional_properties")) o.disable_additional_properties = x->b;
        if (auto* x = v.get("whitespace_pattern"); x && x->is_str()) o.whitespace_pattern = x->s;
        if (auto* x = v.get("structural_tag"); x && x->is_str()) o.structural_tag = x->s;
        o.backend.reset();
        if (auto* x = v.get("_backend"); x && x->is_str()) o.backend = x->s;
        if (auto* x = v.get("_backend_was_auto")) o.backend_was_auto = x->b;
        sp.structured_outputs = o;
      }
    } else if (k == "logit_bias") {
      if (v.is_map()) {
        std::vector<std::pair<int64_t, double>> lb;
        for (auto& [a, b] : v.kv) lb.emplace_back(a.as_int(), b.as_double());
        sp.logit_bias = lb;
      }
    } else if (k == "allowed_token_ids") { if (v.is_array()) sp.allowed_token_ids = ints(v); }
    else if (k == "extra_args") { if (!v.is_nil()) sp.extra_args = v; }
    else if (k == "bad_words") { if (v.is_array()) sp.bad_words = strs(v); }
    else if (k == "_bad_words_token_ids") {
      if (v.is_array()) {
        std::vector<std::vector<int64_t>> w;
        for (auto& e : v.arr) w.push_back(ints(e));
        sp.bad_words_token_ids = w;
      }
    } else if (k == "skip_reading_prefix_cache") { if (!v.is_nil()) sp.skip_reading_prefix_cache = v.b; }
    else if (k == "thinking_token_budget") { if (!v.is_nil()) sp.thinking_token_budget = v.as_int(); }
    else if (k == "repetition_detection") { if (!v.is_nil()) sp.repetition_detection = v; }
    else if (k == "routed_experts_prompt_start") sp.routed_experts_prompt_start = v.as_int();
    else if (k == "trace_decode_token_ids") { if (v.is_array()) sp.trace_decode_token_ids = ints(v); }
  }
  return sp;
}
}  // namespace fe::engine
