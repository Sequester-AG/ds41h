// GenBackend over fe::engine::AsyncEngine.
#pragma once
#include <memory>

#include "api/backend.h"
#include "engine/engine.h"
#include "engine/sampling.h"

namespace fe {
class Tokenizer;
}

namespace fe::api {

// Inside a catch block: rethrows the current fe_engine input error (ValidationError, PromptTooLongError,
// EngineDeadError, std::invalid_argument) as the ApiError the Python frontend answers with.
[[noreturn]] void rethrow_input_error();

class EngineBackend : public GenBackend {
 public:
  EngineBackend(std::unique_ptr<fe::engine::AsyncEngine> eng, const fe::Tokenizer* tok,
                fe::engine::SamplingContext ctx);
  std::string start(GenRequest&& req, GenCallback cb) override;
  void abort(const std::string& internal_id) override;
  bool dead() const override { return eng_->dead(); }
  SamplingSummary sampling_summary(const std::string& endpoint, const Json& body, int64_t prompt_len) override;
  void precheck_caps(const std::string& endpoint, const Json& body, int64_t prompt_len) override;
  int64_t max_model_len() const override { return eng_->max_model_len(); }
  std::string render_vllm_metrics() override { return eng_->metrics().render_vllm(); }
  size_t num_unfinished() const override { return eng_->num_unfinished(); }
  fe::engine::AsyncEngine& engine() { return *eng_; }

 private:
  std::unique_ptr<fe::engine::AsyncEngine> eng_;
  const fe::Tokenizer* tok_;
  fe::engine::SamplingContext ctx_;
};

}  // namespace fe::api
