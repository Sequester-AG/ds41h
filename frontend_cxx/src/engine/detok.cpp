#include "engine/detok.h"

#include <algorithm>
#include <climits>

#include "engine/textutil.h"
#ifdef FE_HAVE_TOKENIZER
#include "tokenizer/tokenizer.h"
#endif

namespace fe::engine {

#ifdef FE_HAVE_TOKENIZER
namespace {
class FastDecoder final : public TokenDecoder {
 public:
  FastDecoder(const fe::Tokenizer& tok, const std::vector<int32_t>& prompt, const SamplingParams& sp)
      : ds_(tok, sp.skip_special_tokens) {
    ds_.prime(prompt);
    ds_.set_spaces_between_special_tokens(sp.skip_special_tokens || sp.spaces_between_special_tokens);
  }
  std::string decode_next(int32_t id) override { return ds_.decode_next(id); }

 private:
  fe::DecodeStream ds_;
};
}  // namespace

std::unique_ptr<TokenDecoder> make_fast_decoder(const fe::Tokenizer& tok, const std::vector<int32_t>& prompt_ids,
                                                const SamplingParams& sp) {
  return std::make_unique<FastDecoder>(tok, prompt_ids, sp);
}
#endif

std::optional<StopMatch> check_stop_strings(const std::u32string& text, size_t new_char_count,
                                            const std::vector<std::u32string>& stop, bool include_in_output) {
  if (!new_char_count || stop.empty()) return std::nullopt;
  std::optional<size_t> best;
  size_t best_index = 0, best_end = SIZE_MAX;
  const int64_t len = int64_t(text.size());
  for (size_t si = 0; si < stop.size(); ++si) {
    const auto& s = stop[si];
    // output_text.find(stop_str, 1 - new_char_count - stop_string_len): negative start counts from the end
    int64_t start = 1 - int64_t(new_char_count) - int64_t(s.size());
    if (start < 0) start = std::max<int64_t>(0, len + start);
    if (start > len) continue;
    size_t idx = text.find(s, size_t(start));
    if (idx == std::u32string::npos) continue;
    size_t end = idx + s.size();
    if (end < best_end) {
      best = si;
      best_index = idx;
      best_end = end;
    }
  }
  if (!best) return std::nullopt;
  if (include_in_output) {
    if (best_end >= text.size()) return StopMatch{*best, -1};
    return StopMatch{*best, int64_t(best_end)};
  }
  return StopMatch{*best, int64_t(best_index)};
}

IncrementalDetokenizer::IncrementalDetokenizer(std::unique_ptr<TokenDecoder> decoder, const SamplingParams& sp)
    : dec_(std::move(decoder)) {
  if (!dec_) return;  // IncrementalDetokenizer base class: ids only
  for (auto& s : sp.stop_strings()) {
    stop_.push_back(utf8_to_u32(s));
    stop_utf8_.push_back(s);
  }
  min_tokens_ = sp.min_tokens;
  include_stop_ = sp.include_stop_str_in_output;
  if (!stop_.empty() && !include_stop_) {
    size_t mx = 0;
    for (auto& s : stop_) mx = std::max(mx, s.size());
    stop_buffer_length_ = mx - 1;
  }
}

std::optional<std::string> IncrementalDetokenizer::update(const std::vector<int32_t>& new_ids_in, bool stop_terminated) {
  if (!dec_) {
    token_ids_.insert(token_ids_.end(), new_ids_in.begin(), new_ids_in.end());
    return std::nullopt;
  }
  if (new_ids_in.empty()) return std::nullopt;
  size_t n_new = new_ids_in.size();
  std::optional<int32_t> skipped;
  if (stop_terminated && !include_stop_) {
    skipped = new_ids_in.back();
    --n_new;
  }
  const int64_t num_prev_tokens = int64_t(num_output_tokens());
  const size_t first_new_token_index = token_ids_.size();
  const bool have_stop = !stop_.empty();
  std::vector<size_t> token_offsets;
  if (have_stop) token_offsets.push_back(output_text_.size());
  for (size_t i = 0; i < n_new; ++i) {
    int32_t t = new_ids_in[i];
    token_ids_.push_back(t);
    std::string piece = dec_->decode_next(t);
    if (!piece.empty()) output_text_ += utf8_to_u32(piece);
    if (have_stop) token_offsets.push_back(output_text_.size());
  }
  if (skipped) {
    token_ids_.push_back(*skipped);
    if (have_stop) token_offsets.push_back(output_text_.size());
  }
  std::optional<std::string> stop_string;
  if (have_stop && int64_t(num_output_tokens()) > min_tokens_) {
    size_t k = size_t(std::max<int64_t>(min_tokens_ - num_prev_tokens, 0));
    size_t stop_check_offset = token_offsets[std::min(k, token_offsets.size() - 1)];
    auto m = check_stop_strings(output_text_, output_text_.size() - stop_check_offset, stop_, include_stop_);
    if (m) {
      stop_string = stop_utf8_[m->stop_index];
      size_t stop_end = m->truncate_to == -1 ? output_text_.size() : size_t(m->truncate_to);
      if (!include_stop_) stop_end += stop_[m->stop_index].size();
      // num_kept = bisect_left(token_offsets, stop_end, lo=1)
      auto it = std::lower_bound(token_offsets.begin() + 1, token_offsets.end(), stop_end);
      size_t num_kept = size_t(it - token_offsets.begin());
      if (first_new_token_index + num_kept < token_ids_.size())
        token_ids_.resize(first_new_token_index + num_kept);
      if (m->truncate_to != -1) output_text_.resize(size_t(m->truncate_to));
    }
  }
  return stop_string;
}

std::string IncrementalDetokenizer::get_next_output_text(bool finished, bool delta) {
  if (!dec_) return "";
  size_t buffer_length = finished ? 0 : stop_buffer_length_;
  if (!delta) {
    if (!buffer_length) return u32_to_utf8(output_text_);
    // output_text[:-buffer_length]
    size_t n = output_text_.size() > buffer_length ? output_text_.size() - buffer_length : 0;
    return u32_to_utf8(std::u32string_view(output_text_).substr(0, n));
  }
  int64_t length = int64_t(output_text_.size()) - int64_t(buffer_length);
  int64_t last = int64_t(last_output_text_offset_);
  if (last < length) {
    last_output_text_offset_ = size_t(length);
    return u32_to_utf8(std::u32string_view(output_text_).substr(size_t(last), size_t(length - last)));
  }
  return "";
}

}  // namespace fe::engine
