#include "api/util.h"

#include <algorithm>
#include <charconv>

namespace fe::api {

std::string py_fstr(const Json& v) {
  if (v.is_string()) return v.as_string();
  if (v.is_array() || v.is_object()) return py_repr(v);
  return py_str_scalar(v);
}

std::vector<int32_t> token_prompt_ids(const Json& ids, std::optional<std::string>& oov_id, size_t begin,
                                      size_t end) {
  std::vector<int32_t> out;
  const auto& arr = ids.as_array();
  end = std::min(end, arr.size());
  begin = std::min(begin, end);
  out.reserve(end - begin);
  int64_t lo = INT64_MAX, hi = INT64_MIN;              // over the ids that fit int64
  const Json *big_lo = nullptr, *big_hi = nullptr;     // most negative / most positive id outside int64
  auto mag_less = [](const Json& a, const Json& b) {   // |a| < |b|, same sign, canonical digits
    const std::string &x = a.big_int_text(), &y = b.big_int_text();
    return x.size() != y.size() ? x.size() < y.size() : x < y;
  };
  for (size_t i = begin; i < end; ++i) {
    const Json& x = arr[i];
    if (x.is_int()) {
      int64_t v = x.as_int();
      lo = std::min(lo, v);
      hi = std::max(hi, v);
      out.push_back(v < INT32_MIN ? INT32_MIN : v > INT32_MAX ? INT32_MAX : int32_t(v));
    } else if (x.is_big_int()) {
      bool neg = x.big_int_text()[0] == '-';
      const Json*& slot = neg ? big_lo : big_hi;
      if (!slot || mag_less(*slot, x)) slot = &x;
      out.push_back(neg ? INT32_MIN : INT32_MAX);
    } else {
      out.push_back(0);
    }
  }
  oov_id.reset();
  if (big_lo || big_hi || lo < INT32_MIN || hi > INT32_MAX) {
    if (big_lo) oov_id = big_lo->big_int_text();
    else if (lo < 0) oov_id = std::to_string(lo);
    else if (big_hi) oov_id = big_hi->big_int_text();
    else oov_id = std::to_string(hi);
  }
  return out;
}

std::optional<int64_t> py_int_header(std::string_view s, std::string* text) {
  // str.strip() whitespace as latin-1 decoded header bytes: \t\n\v\f\r, \x1c-\x1f, space, \x85, \xa0
  auto ws = [](unsigned char c) { return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0; };
  size_t a = 0, b = s.size();
  while (a < b && ws(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && ws(static_cast<unsigned char>(s[b - 1]))) --b;
  bool neg = false;
  if (a < b && (s[a] == '+' || s[a] == '-')) neg = s[a++] == '-';
  std::string digits;
  bool prev_digit = false;
  for (size_t i = a; i < b; ++i) {
    char c = s[i];
    if (c >= '0' && c <= '9') {
      digits += c;
      prev_digit = true;
    } else if (c == '_' && prev_digit && i + 1 < b && s[i + 1] >= '0' && s[i + 1] <= '9') {
      prev_digit = false;
    } else {
      return std::nullopt;
    }
  }
  if (digits.empty()) return std::nullopt;
  size_t nz = digits.find_first_not_of('0');
  digits = nz == std::string::npos ? "0" : digits.substr(nz);
  if (digits == "0") neg = false;
  std::string t = (neg ? "-" : "") + digits;
  int64_t v;
  auto r = std::from_chars(t.data(), t.data() + t.size(), v);
  if (r.ec != std::errc()) v = neg ? INT64_MIN : INT64_MAX;
  if (text) *text = std::move(t);
  return v;
}

std::pair<size_t, size_t> truncation_slice(size_t n, std::optional<int64_t> truncate, bool left,
                                           int64_t max_model_len, int64_t max_output_tokens) {
  if (!truncate) return {0, n};
  // max_truncation_tokens: max_input_tokens (= max_model_len) less one token held back for output
  int64_t max_length = *truncate < 0 ? (max_output_tokens == 0 ? max_model_len : std::max<int64_t>(0, max_model_len - 1))
                                     : *truncate;
  if (max_length >= int64_t(n)) return {0, n};
  if (max_length == 0) return {0, 0};
  return left ? std::pair<size_t, size_t>{n - size_t(max_length), n} : std::pair<size_t, size_t>{0, size_t(max_length)};
}

void Pump::push(GenOutput&& o) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (done_) return;
    if (!q_.empty())
      merge_output(q_.back(), std::move(o));
    else
      q_.push_back(std::move(o));
    if (running_) return;
    if (resp_->congested() && !resp_->disconnected()) {  // resumed by kick()
      if (!self_) self_ = shared_from_this();          // the backend may drop its reference meanwhile
      return;
    }
    running_ = true;
  }
  run();
}

void Pump::kick() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (running_ || done_ || q_.empty()) return;
    running_ = true;
  }
  run();
}

void Pump::run() {
  std::shared_ptr<Pump> release;  // a dropped self-reference: destroyed after the lock, as the last action
  for (;;) {
    GenOutput o;
    {
      std::lock_guard<std::mutex> lk(mu_);
      bool parked = resp_->congested() && !resp_->disconnected();
      if (done_ || q_.empty() || parked) {
        running_ = false;
        if (parked && !done_ && !q_.empty()) {
          if (!self_) self_ = shared_from_this();
        } else {
          release = std::move(self_);
        }
        return;
      }
      o = std::move(q_.front());
      q_.pop_front();
    }
    bool more = fn_(std::move(o));
    if (!more) {
      std::lock_guard<std::mutex> lk(mu_);
      done_ = true;
      running_ = false;
      q_.clear();
      release = std::move(self_);
      return;
    }
  }
}

bool spec_decode_present(const Json& m) { return m.is_object(); }

void write_spec_decode(JW& w, const Json& m, bool full_dump) {
  // m: RequestSpecDecodeMetrics as a map {num_spec_tokens, histogram, num_draft_tokens,
  //    per_step_accepted, per_step_drafted}; or already to_dict() form.
  if (m.get("mean_acceptance_length")) {  // already to_dict()
    w.ob();
    for (auto& [k, v] : m.as_object()) w.kjson(k, v);
    if (full_dump) {
      if (!m.get("per_step_accepted")) w.knull("per_step_accepted");
      if (!m.get("per_step_drafted")) w.knull("per_step_drafted");
    }
    w.oe();
    return;
  }
  int64_t k = m.get_int("num_spec_tokens").value_or(0);
  std::vector<int64_t> hist;
  if (const Json* h = m.get("histogram"); h && h->is_array())
    for (auto& x : h->as_array()) hist.push_back(x.is_int() ? x.as_int() : 0);
  int64_t ndraft = m.get_int("num_draft_tokens").value_or(0);
  int64_t steps = 0, acc = 0;
  for (size_t j = 0; j < hist.size(); ++j) {
    steps += hist[j];
    acc += int64_t(j) * hist[j];
  }
  double mean_al = steps ? 1.0 + double(acc) / double(steps) : 1.0;
  double rate = ndraft ? double(acc) / double(ndraft) : 0.0;
  w.ob();
  w.k("mean_acceptance_length").f(mean_al);
  w.k("draft_acceptance_rate").f(rate);
  w.k("acceptance_histogram").ab();
  for (auto x : hist) w.i(x);
  w.ae();
  w.ki("num_spec_steps", steps).ki("num_accepted_draft_tokens", acc).ki("num_draft_tokens", ndraft);
  w.ki("num_spec_tokens", k);
  const Json* psa = m.get("per_step_accepted");
  const Json* psd = m.get("per_step_drafted");
  bool populated = psa && psa->is_array() && psa->size() > 0;
  if (populated) {
    w.kjson("per_step_accepted", *psa);
    w.kjson("per_step_drafted", psd ? *psd : Json::array());
  } else if (full_dump) {
    w.knull("per_step_accepted").knull("per_step_drafted");
  }
  w.oe();
}

void write_prompt_details_stream(JW& w, bool enabled, int64_t cached, int64_t creation, int64_t mm) {
  // _make_prompt_tokens_details: RequestOutput always carries ints, so the object is present
  // whenever the flag is on; model_dump_json(exclude_none) drops the None multimodal field.
  if (!enabled) return;
  w.k("prompt_tokens_details").ob().ki("cached_tokens", cached).ki("created_cache_tokens", creation);
  if (mm > 0) w.k("multimodal_tokens").ob().ki("image", mm).oe();
  w.oe();
}

void write_prompt_details_full(JW& w, bool enabled, int64_t cached, int64_t creation, int64_t mm) {
  if (!enabled) {
    w.knull("prompt_tokens_details");
    return;
  }
  w.k("prompt_tokens_details").ob().ki("cached_tokens", cached).ki("created_cache_tokens", creation);
  if (mm > 0)
    w.k("multimodal_tokens").ob().ki("image", mm).oe();
  else
    w.knull("multimodal_tokens");
  w.oe();
}

void write_metrics_full(JW& w, const Json& spec) {
  if (!spec_decode_present(spec)) {
    w.knull("metrics");
    return;
  }
  w.k("metrics").ob();
  w.knull("time_to_first_token_ms").knull("generation_time_ms").knull("queue_time_ms").knull("mean_itl_ms");
  w.knull("tokens_per_second");
  w.k("speculative_decoding");
  write_spec_decode(w, spec, true);
  w.oe();
}

}  // namespace fe::api
