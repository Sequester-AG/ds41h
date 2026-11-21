// DecodeStream: port of tokenizers 0.23.3 step_decode_stream + PyDecodeStream(ids=...) and of vLLM 0.32.0
// FastIncrementalDetokenizer._protected_step / decode_next (v1/engine/detokenizer.py:179-260).
#include <cstring>

#include "tokenizer/tokenizer_impl.h"

namespace fe {

namespace {
inline bool ends_with_replacement(const std::string& s) {
  return s.size() >= 3 && memcmp(s.data() + s.size() - 3, "\xEF\xBF\xBD", 3) == 0;
}
}  // namespace

DecodeStream::DecodeStream(const Tokenizer& t, bool skip_special) : tok_(&t), skip_(skip_special) {}

void DecodeStream::prime(std::span<const int32_t> prompt_ids) {
  ids_.assign(prompt_ids.begin(), prompt_ids.end());
  prefix_.clear();
  prefix_index_ = 0;
  last_special_ = false;
}

std::optional<std::string> DecodeStream::raw_step(int32_t id) {
  const TokenizerImpl& im = tok_->impl();
  if (prefix_.empty() && !ids_.empty()) {
    std::string np = im.decode(ids_.data(), ids_.size(), skip_);
    if (!ends_with_replacement(np)) {
      prefix_ = std::move(np);
      prefix_index_ = ids_.size();
    }
  }
  ids_.push_back(id);
  std::string s = im.decode(ids_.data(), ids_.size(), skip_);
  if (s.size() > prefix_.size() && !ends_with_replacement(s)) {
    if (s.compare(0, prefix_.size(), prefix_) != 0) throw TokenizerError("Invalid prefix encountered");
    std::string new_text = s.substr(prefix_.size());
    size_t new_prefix_index = ids_.size() - prefix_index_;
    ids_.erase(ids_.begin(), ids_.begin() + long(prefix_index_));
    prefix_ = im.decode(ids_.data(), ids_.size(), skip_);
    prefix_index_ = new_prefix_index;
    return new_text;
  }
  return std::nullopt;
}

std::optional<std::string> DecodeStream::step(int32_t id) {
  if (id < 0) return std::nullopt;  // Python: OverflowError converting to u32 -> logged, token None
  try {
    return raw_step(id);
  } catch (const TokenizerError& e) {
    if (strncmp(e.what(), "Invalid prefix encountered", 26) != 0) throw;
    resets_++;
    ids_.clear();
    prefix_.clear();
    prefix_index_ = 0;
    return raw_step(id);
  }
}

void DecodeStream::set_spaces_between_special_tokens(bool v) { spaces_between_ = v; }

std::string DecodeStream::decode_next(int32_t id) {
  std::optional<std::string> t = step(id);
  if (!(skip_ || spaces_between_)) {
    std::optional<std::string_view> c = tok_->added_token_content(id);
    bool is_special = c.has_value();
    if (is_special && last_special_) t = std::string(*c);
    last_special_ = is_special;
  }
  return t ? std::move(*t) : std::string();
}

}  // namespace fe
