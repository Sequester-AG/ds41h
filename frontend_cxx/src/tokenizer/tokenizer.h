// fe_tokenizer: byte-level BPE tokenizer, bit-exact with HF `tokenizers` 0.23.3 as vLLM 0.32.0 uses it for
// DeepSeek-V4.1-Flash. See DESIGN.md "Interfaces" and tools/inventory.md 1.4 / 1.7.
//
// Supported tokenizer.json subset (anything else -> TokenizerError at load):
//   normalizer: null | Sequence[]           pre_tokenizer: Sequence[Split(Regex, Isolated)..., ByteLevel(no prefix space,
//   use_regex=false)]                       model: BPE (byte-level, no dropout / prefixes / suffixes / unk / byte_fallback)
//   decoder: ByteLevel                      post_processor: ByteLevel | null (adds nothing with add_special_tokens=False)
//   added_tokens: full AddedVocabulary semantics (special / non-special, normalized, lstrip, rstrip, single_word,
//   leftmost-longest, two passes: non-normalized tokens on the raw text, then normalized tokens on the remaining pieces).
// The DeepSeek V3/V4 3-stage Split pattern runs on a hand-written matcher (Unicode tables probed from the Oniguruma
// copy inside HF tokenizers); any other Split pattern runs on the vendored Oniguruma 6.9.10 (same tables).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fe {

struct TokenizerError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// Thrown by encode() / encode_unchunked() for text that is not valid UTF-8, e.g. a lone surrogate (fe_common keeps
// JSON "\ud800" as WTF-8 ED A0..BF xx). Python cannot pass such a str to the Rust tokenizer either: every vLLM encode
// path (encode, __call__, the chunked encode_batch) raises TypeError with exactly this what() text.
struct TokenizerInputError : TokenizerError {
  TokenizerInputError()
      : TokenizerError("TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]") {}
};

class TokenizerImpl;

class Tokenizer {
 public:
  // Arguments are FILE PATHS ($M/tokenizer.json, $M/tokenizer_config.json). The config path may be empty.
  static std::unique_ptr<Tokenizer> load(const std::string& tokenizer_json, const std::string& tokenizer_config_json);
  // Same, from the file contents.
  static std::unique_ptr<Tokenizer> from_json(std::string_view tokenizer_json, std::string_view tokenizer_config_json);
  ~Tokenizer();

  // == vLLM's tokenizer.encode(text, add_special_tokens=False) in production
  // texts of >= 8192 code points are cut where "\r"/"\n" is followed by an ASCII letter into ~4096-code-point chunks
  // (ds41h/patch_serving.py split_chunks), the chunks are encoded independently (in parallel on the internal pool when
  // parallelism > 1) and the ids concatenated. Throws TokenizerInputError on invalid UTF-8 / lone surrogates.
  // Thread-safe (concurrent calls share the pool; per-thread BPE word caches).
  std::vector<int32_t> encode(std::string_view text) const;
  // HF encode of the whole string, never chunked, single-threaded (the stock vLLM path; reference / benchmarks).
  std::vector<int32_t> encode_unchunked(std::string_view text) const;
  // == HF Tokenizer.decode(ids, skip_special_tokens): ByteLevel decoder + String::from_utf8_lossy. Unknown or
  // negative ids are dropped (HF drops ids it cannot map; Python raises OverflowError for negative ints).
  std::string decode(std::span<const int32_t> ids, bool skip_special) const;
  // == HF token_to_id (added vocabulary first, then the BPE vocab).
  std::optional<int32_t> token_to_id(std::string_view token) const;
  // == HF id_to_token (the token string as stored, byte-level chars for BPE tokens).
  std::optional<std::string> id_to_token(int32_t id) const;
  // True for the special added tokens (the set DecodeStream / decode(skip_special=true) drop).
  bool is_special(int32_t id) const;
  // Added token content for id (special or not), as vLLM's get_added_tokens_decoder(); nullopt otherwise.
  std::optional<std::string_view> added_token_content(int32_t id) const;
  // == len(tokenizer) in transformers / vLLM (vocab + added tokens, 129,280 for V4.1).
  int32_t vocab_size() const;
  // == vLLM max_token_id (max id of vocab + added tokens, 129,279 for V4.1).
  int32_t max_token_id() const;
  // Number of code points of a valid UTF-8 string (Python len()).
  static size_t count_chars(std::string_view text);
  // The production chunking rule, as byte offsets of the chunk starts (first is 0). Exposed for tests.
  static std::vector<size_t> split_chunks(std::string_view text, size_t min_chars = 8192, size_t target = 4096);

  // Threads used by encode() for chunk parallelism (1 = single-threaded). Default: env FE_TOKENIZER_THREADS, else
  // min(8, hardware_concurrency). Safe to call concurrently with encode().
  void set_parallelism(int nthreads);
  int parallelism() const;

  const TokenizerImpl& impl() const { return *impl_; }

 private:
  Tokenizer() = default;
  std::unique_ptr<TokenizerImpl> impl_;
};

// == tokenizers.decoders.DecodeStream as vLLM's FastIncrementalDetokenizer uses it (v1/engine/detokenizer.py:179-260):
//   DecodeStream(ids=prompt_token_ids, skip_special_tokens=sp.skip_special_tokens), then _protected_step per token.
class DecodeStream {
 public:
  DecodeStream(const Tokenizer&, bool skip_special);
  // == constructing DecodeStream(ids=prompt_ids): resets the state and stores the prompt ids (decoded lazily at the
  // first step, exactly like the Rust prefill).
  void prime(std::span<const int32_t> prompt_ids);
  // == FastIncrementalDetokenizer._protected_step: the new text, or nullopt (Python None) while a UTF-8 sequence is
  // incomplete. On the "Invalid prefix encountered" error the stream is reset WITHOUT prompt and the id re-stepped
  // (resets() counts these; vLLM logs a warning). A negative id (Python OverflowError) returns nullopt, state unchanged.
  std::optional<std::string> step(int32_t id);
  // == FastIncrementalDetokenizer.decode_next: step() plus the spaces_between_special_tokens=False rule (only active
  // when skip_special is false and set_spaces_between_special_tokens(false) was called); returns "" for None.
  std::string decode_next(int32_t id);
  void set_spaces_between_special_tokens(bool v);
  size_t resets() const { return resets_; }

  // The raw Rust DecodeStream.step (throws TokenizerError("Invalid prefix encountered") instead of resetting).
  std::optional<std::string> raw_step(int32_t id);

 private:
  const Tokenizer* tok_;
  bool skip_;
  bool spaces_between_ = true;
  bool last_special_ = false;
  size_t resets_ = 0;
  std::vector<int32_t> ids_;
  std::string prefix_;
  size_t prefix_index_ = 0;
};

}  // namespace fe
