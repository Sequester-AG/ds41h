// Error responses of the vLLM 0.32.0 frontend (serve/exception_handling/*).
#pragma once
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "common/json.h"

namespace fe::api {

// An error that becomes an HTTP error response: {"error":{"message","type","param","code"}}.
struct ApiError : std::exception {
  int status = 400;
  std::string type = "BadRequestError";
  std::string message;
  std::optional<std::string> param;
  ApiError() = default;
  ApiError(int s, std::string t, std::string m, std::optional<std::string> p = std::nullopt)
      : status(s), type(std::move(t)), message(std::move(m)), param(std::move(p)) {}
  const char* what() const noexcept override { return message.c_str(); }
};

// create_error_response(...) with an explicit message (BaseServing.create_error_response).
inline ApiError bad_request(std::string msg, std::optional<std::string> param = std::nullopt) {
  return ApiError(400, "BadRequestError", std::move(msg), std::move(param));
}
// VLLMValidationError: str() appends " (parameter=..., value=...)".
ApiError vllm_validation_error(std::string msg, std::optional<std::string> param,
                               std::optional<std::string> value_str = std::nullopt);
// GenerationError("Internal server error"): 500 InternalServerError.
inline ApiError generation_error() {
  return ApiError(500, "InternalServerError", "Internal server error", std::nullopt);
}

// sanitize_message (serve/exception_handling/utils.py).
std::string sanitize_message(std::string_view msg);

// JSONResponse(ErrorResponse.model_dump()) body: compact, ensure_ascii=False.
std::string error_body(const ApiError& e);
// json.dumps(ErrorResponse.model_dump()): ", " / ": ", ensure_ascii=True (SSE error chunk).
std::string error_body_stream(const ApiError& e);

// ---- pydantic RequestValidationError (handlers/validation.py) ----
struct PydErr {
  std::string type;
  std::vector<std::string> loc;  // e.g. {"body","messages","0"}
  std::string msg;
  // repr()-ready summarised input: already the Python repr text (e.g. "'5'", "'<dict of 1 items>'").
  std::string input_repr;
  // optional ", 'ctx': {...}" body (without the leading ", 'ctx': "), Python repr text.
  std::string ctx_repr;
};
// _summarize_error_input(value) as a Python str, then repr() of it.
std::string summarize_input_repr(const Json& v);
std::string summarize_input_repr_none();  // input None -> 'None'
// 400 "Bad Request" ApiError from a list of pydantic errors (count, 10-entry cap, param).
ApiError validation_error(const std::vector<PydErr>& errs);
// RequestValidationError(errors=["<plain string>"]) (validate_json_request).
ApiError validation_error_plain(std::string_view text);

// Python repr() of a str.
std::string py_repr_str(std::string_view s);
// Python str() of a JSON scalar as Python would print the decoded value (int, float, bool, None).
std::string py_str_scalar(const Json& v);
// Python repr() of a decoded JSON value (dict / list / str / int / float / bool / None).
std::string py_repr(const Json& v);

}  // namespace fe::api
