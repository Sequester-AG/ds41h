// A small pydantic 2.14 / pydantic-core 2.50 emulation for the request shapes the renderer validates.
//
// Covers: TypedDict (total=False + Required, extras kept), pydantic models with extra="allow", smart unions
// (most fields set wins, nested model-like counts add up, ties keep the earlier member), lazy Iterable fields
// (ValidatorIterator: validated when consumed), str / int / float / bool lax rules, Literal, list, dict, nullable.
// Errors render either like vLLM's RequestValidationError handler (FastAPI errors() dicts, cleaned loc) or like
// str(pydantic_core.ValidationError) ("N validation errors for X\nloc\n  msg [type=..., input_value=...]").
#pragma once

#include <deque>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "common/json.h"

namespace fe::render::pyd {

struct Error {
  std::string type;
  JsonArray loc;  // str / int segments
  std::string msg;
  Json input;
  Json ctx;  // null or object
};
using Errors = std::vector<Error>;

enum class K { Any, Str, Int, Float, Bool, Literal, List, Iterable, Dict, TypedDict, Model, Union, Nullable };

struct Field;
struct Schema {
  K kind = K::Any;
  std::string name;               // TypedDict / Model class name
  std::vector<std::string> lits;  // Literal values (str)
  const Schema* item = nullptr;   // List / Iterable item, Dict value, Nullable inner
  std::vector<Field> fields;      // TypedDict / Model
  std::vector<const Schema*> members;  // Union
  // int constraints
  bool has_ge = false, has_le = false;
  Json ge, le;
  // str constraints
  int min_length = -1, max_length = -1;
  // vLLM OpenAIBaseModel: its wrap model_validator (__log_extra_fields__) makes the smart-union label
  // "function-wrap[__log_extra_fields__()]" (dropped by vLLM's clean_loc_for_param)
  bool vllm_model = false;
  std::string label() const;      // smart-union member label (as in loc)
};
struct Field {
  std::string name;
  const Schema* schema;
  bool required;
};

struct Result {
  bool ok = false;
  int fields_set = 0;   // pydantic fields_set_count (model-like types; nested counts add up)
  int member = -1;      // Union: winning member index
  bool coerced = false; // true when the validated value differs from the input (coercion / TypedDict key order)
  Json value;           // the validated value when coerced (only built when asked for, see validate())
};

// Validate `v` against `s`. On failure, errors are appended (loc relative to `loc_prefix`).
Result validate(const Schema& s, const Json& v, const JsonArray& loc_prefix, Errors& errs);

// Like validate(), for an owned input that is discarded afterwards: when r.coerced, r.value was built by moving
// from `v` (only for a Union schema; other kinds copy).
Result validate_consume(const Schema& s, Json& v, const JsonArray& loc_prefix, Errors& errs);

// The vLLM validation_exception_handler message ("N validation errors:\n  {...}") and its param.
std::string format_request_errors(const Errors& errs, std::string* param_out);
// str(pydantic_core.ValidationError) with the given title.
std::string format_native_errors(const std::string& title, const Errors& errs);
// vLLM clean_loc_for_param
std::string clean_loc(const JsonArray& loc);

// Python repr() truncated like pydantic-core's input_value (> 50 chars: 25 + "..." + 24).
std::string truncated_repr(const Json& v);

// Schema construction (schemas live as long as the Builder).
struct Builder {
  std::deque<Schema> pool;
  Schema* mk(K k, std::string name = {});
  const Schema* lit(std::initializer_list<const char*> v);
  const Schema* wrap(K k, const Schema* item);
  const Schema* uni(std::initializer_list<const Schema*> m);
  const Schema* td(const char* name, std::initializer_list<Field> f, K k = K::TypedDict);
  const Schema* int_range(const Json* ge, const Json* le);
};

// Shared schema instances (built once).
struct Schemas {
  // chat messages
  const Schema* message_union;          // ChatCompletionMessageParam
  const Schema* openai_user_parts;      // Union[Text, Image, InputAudio, File] (lazy content of a user message)
  const Schema* openai_assistant_parts; // Union[Text, Refusal]
  const Schema* text_part;              // ChatCompletionContentPartTextParam
  const Schema* tool_call_union;        // Union[FunctionToolCall, CustomToolCall] (lazy assistant tool_calls)
  const Schema* image_part;             // ChatCompletionContentPartImageParam (MM parser TypeAdapter)
  const Schema* responses_input_image;  // ResponseInputImageParam
  const Schema* audio_part;             // ChatCompletionContentPartAudioParam
  const Schema* video_part;             // ChatCompletionContentPartVideoParam
  const Schema* tools_param;            // ChatCompletionToolsParam (model)
  const Schema* named_tool_choice;      // ChatCompletionNamedToolChoiceParam (model)
  const Schema* responses_dict_item;    // Union[EasyInputMessageParam, Message, FunctionCallOutput] (input dict items)
};
const Schemas& schemas();

// Message-union indices
enum MsgBranch { kDeveloper = 0, kSystem, kUser, kAssistant, kTool, kFunction, kCustom, kHarmony };

}  // namespace fe::render::pyd
