// pydantic 2.14 lax-mode field validation for the request models we serve, producing
// the same error list (type / loc / msg / input) as FastAPI's RequestValidationError.
#pragma once
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "api/errors.h"
#include "common/json.h"

namespace fe::api {

class Pyd {
 public:
  std::vector<PydErr> errs;
  bool ok() const { return errs.empty(); }
  void add(std::string type, std::vector<std::string> loc, std::string msg, const Json* input,
           std::string ctx = {});

  // Each validator: `v` == nullptr means "absent" (default used). A JSON null is accepted
  // when `nullable`. On error an entry is added and nullopt returned.
  // int: Int, BigInt (clamped to int64 for use), integral Float, numeric str, bool.
  std::optional<int64_t> to_int(const Json* v, std::vector<std::string> loc, bool nullable,
                                std::optional<int64_t> ge = std::nullopt,
                                std::optional<int64_t> le = std::nullopt);
  std::optional<double> to_float(const Json* v, std::vector<std::string> loc, bool nullable);
  std::optional<bool> to_bool(const Json* v, std::vector<std::string> loc, bool nullable);
  std::optional<std::string> to_str(const Json* v, std::vector<std::string> loc, bool nullable);
  // Literal["a","b",...]; returns the matched value.
  std::optional<std::string> to_literal(const Json* v, std::vector<std::string> loc, bool nullable,
                                        const std::vector<std::string>& allowed);
  bool is_list(const Json* v, std::vector<std::string> loc, bool nullable);
  bool is_dict(const Json* v, std::vector<std::string> loc, bool nullable);
  // A nested model (dict required): model_attributes_type otherwise.
  bool is_model(const Json* v, std::vector<std::string> loc, bool nullable);
  void missing(std::vector<std::string> loc, const Json& parent);
  // was the last validated value JSON null (only meaningful right after a call)?
  bool last_null = false;
};

// Field "is null or absent".
inline bool absent_or_null(const Json* v) { return v == nullptr || v->is_null(); }

}  // namespace fe::api
