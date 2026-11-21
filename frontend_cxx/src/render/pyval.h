// Python value semantics on fe::Json for the render component.
//
// The vLLM request path is Python code operating on json.loads() values: truthiness, str(), repr(), f-strings,
// exception classes and their HTTP mapping. These helpers reproduce exactly the parts the renderer needs.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "common/json.h"

namespace fe::render {

// bool(x)
bool py_truthy(const Json& v) noexcept;
// type(x).__name__: NoneType bool int float str list dict
const char* py_type_name(const Json& v) noexcept;
// str(x) / format(x, "") / f"{x}"
std::string py_str(const Json& v);
void py_str_append(std::string& out, const Json& v);
// repr(x)
std::string py_repr(const Json& v);
void py_repr_append(std::string& out, const Json& v);
// repr(s) for a Python str held as (WTF-)UTF-8
std::string py_repr_str(std::string_view s);
void py_repr_str_append(std::string& out, std::string_view s);
// str.isprintable() for one code point (Python 3.12 semantics, Unicode 15 table subset)
bool py_isprintable(uint32_t cp) noexcept;
// hash-based `x in dict` needs hashable keys: str / int / float / bool / None
bool py_hashable(const Json& v) noexcept;

// A Python exception raised on the request path. `cls` is the Python class name; it decides the HTTP mapping
// (vllm/entrypoints/serve/exception_handling/error_response.py create_error_response):
//   VLLMValidationError -> 400 BadRequestError, param = parameter, message = str(exc) incl. " (parameter=..., value=...)"
//   ValueError / TypeError / OverflowError / ValidationError (pydantic, a ValueError) -> 400 BadRequestError
//   NotImplementedError -> 501 NotImplementedError
//   anything else (KeyError, AttributeError, AssertionError, IndexError, RuntimeError, ...) -> 500 InternalServerError
struct PyError : std::runtime_error {
  std::string cls;
  std::string msg;                   // str(exc) WITHOUT the VLLMValidationError suffix
  std::optional<std::string> param;  // VLLMValidationError.parameter
  std::optional<std::string> value;  // VLLMValidationError.value, already str()-ed
  PyError(std::string c, std::string m, std::optional<std::string> p = std::nullopt,
          std::optional<std::string> v = std::nullopt)
      : std::runtime_error(m), cls(std::move(c)), msg(std::move(m)), param(std::move(p)), value(std::move(v)) {}
  std::string str() const;  // str(exc)
};

[[noreturn]] void raise_vllm(std::string msg, std::optional<std::string> param = std::nullopt,
                             std::optional<std::string> value = std::nullopt);
[[noreturn]] void raise_py(const char* cls, std::string msg);

// vllm/entrypoints/serve/exception_handling/utils.py sanitize_message
std::string sanitize_message(std::string_view message);

}  // namespace fe::render
