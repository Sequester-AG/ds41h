// DeepSeek V4.1 prompt encoder: a line-by-line port of vllm/tokenizers/deepseek_v41_encoding.py (vLLM 0.32.0),
// itself adapted from the checkpoint's encoding/encoding.py. Owner C (render).
//
// Messages are Python dicts (fe::Json objects) with Python semantics: missing keys read as None, values are
// formatted with str(), truthiness decides branches, and the same Python exceptions are raised (as PyError) for the
// same malformed inputs (TypeError for str + list, KeyError for a missing "function", AssertionError for a bad
// task, ...).
#pragma once

#include <string>
#include <vector>

#include "common/json.h"

namespace fe::render {

// Special tokens (deepseek_v41_encoding.py:14-33)
inline constexpr const char* kBos = "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>";
inline constexpr const char* kEos = "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>";
inline constexpr const char* kImagePlaceholder = "<\xef\xbd\x9c" "deepseek_image\xef\xbd\x9c>";

// encode_messages(messages, thinking_mode, context=None, drop_thinking, add_default_bos_token=True, reasoning_effort)
// `messages` is consumed (the Python code deep-copies; callers pass a copy they no longer need).
// `drop_thinking` is the truthiness of the Python value; `reasoning_effort` is None / str / int as passed.
std::string encode_messages(JsonArray messages, bool thinking, bool drop_thinking, const Json& reasoning_effort);

// render_tools(tools): the "## Tools" system block for a list of function dicts.
std::string render_tools(const Json& functions);
std::string render_tools_ptrs(const std::vector<const Json*>& functions);

}  // namespace fe::render
