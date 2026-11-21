// Incremental detokenizer + stop strings: vLLM v1/engine/detokenizer.py
//   BaseIncrementalDetokenizer.update / get_next_output_text (:96-177) + check_stop_strings (:321-373),
//   FastIncrementalDetokenizer (:179-260) over fe::DecodeStream.
// Text offsets are Python str offsets (code points), exactly like the Python code.
#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "engine/sampling.h"

namespace fe {
class Tokenizer;
class DecodeStream;
}  // namespace fe

namespace fe::engine {

// Token -> text step function (DecodeStream.step semantics, incl. the "Invalid prefix" reset).
class TokenDecoder {
 public:
  virtual ~TokenDecoder() = default;
  virtual std::string decode_next(int32_t id) = 0;  // "" when the stream returns None
};

#ifdef FE_HAVE_TOKENIZER
// FastIncrementalDetokenizer: DecodeStream(ids=prompt, skip_special_tokens) + spaces_between rule.
std::unique_ptr<TokenDecoder> make_fast_decoder(const fe::Tokenizer& tok, const std::vector<int32_t>& prompt_ids,
                                                const SamplingParams& sp);
#endif

// check_stop_strings(): returns (stop_string_index, truncate_to) or nullopt. Text in code points.
struct StopMatch {
  size_t stop_index;   // index into `stop`
  int64_t truncate_to; // -1 = no truncation
};
std::optional<StopMatch> check_stop_strings(const std::u32string& output_text, size_t new_char_count,
                                            const std::vector<std::u32string>& stop, bool include_in_output);

class IncrementalDetokenizer {
 public:
  // decoder == nullptr: no detokenization (sp.detokenize=False / no tokenizer): ids only, no stop strings.
  IncrementalDetokenizer(std::unique_ptr<TokenDecoder> decoder, const SamplingParams& sp);

  // update(new_token_ids, stop_terminated) -> matched stop string (UTF-8) or nullopt.
  std::optional<std::string> update(const std::vector<int32_t>& new_token_ids, bool stop_terminated);
  // get_next_output_text(finished, delta) as UTF-8.
  std::string get_next_output_text(bool finished, bool delta);

  size_t num_output_tokens() const { return token_ids_.size(); }
  const std::vector<int32_t>& output_token_ids() const { return token_ids_; }
  const std::u32string& output_text() const { return output_text_; }

 private:
  std::unique_ptr<TokenDecoder> dec_;
  std::vector<std::u32string> stop_;
  std::vector<std::string> stop_utf8_;
  int64_t min_tokens_ = 0;
  bool include_stop_ = false;
  size_t stop_buffer_length_ = 0;
  size_t last_output_text_offset_ = 0;
  std::vector<int32_t> token_ids_;
  std::u32string output_text_;
};

}  // namespace fe::engine
