#include "api/backend_engine.h"

#include "api/errors.h"
#include "engine/json_bridge.h"
#include "engine/protocol.h"
#include "image/image.h"
#include "tokenizer/tokenizer.h"


namespace fe::api {

namespace eng = fe::engine;
namespace mp = fe::mp;

EngineBackend::EngineBackend(std::unique_ptr<eng::AsyncEngine> e, const fe::Tokenizer* tok, eng::SamplingContext ctx)
    : eng_(std::move(e)), tok_(tok), ctx_(ctx) {}

[[noreturn]] void rethrow_input_error() {
  try {
    throw;
  } catch (const eng::ValidationError& e) {
    throw ApiError(400, "BadRequestError", e.what(),
                   e.parameter.empty() ? std::nullopt : std::optional<std::string>(e.parameter));
  } catch (const eng::PromptTooLongError& e) {
    throw ApiError(400, "BadRequestError", e.what());
  } catch (const eng::EngineDeadError& e) {
    throw ApiError(500, "InternalServerError", e.what());
  } catch (const std::invalid_argument& e) {
    throw ApiError(400, "BadRequestError", e.what());
  }
}

namespace {
eng::Endpoint ep_of(const std::string& s) {
  if (s == "completion") return eng::Endpoint::Completions;
  if (s == "responses") return eng::Endpoint::Responses;
  return eng::Endpoint::Chat;
}

GenOutput convert(eng::RequestOutput&& ro) {
  GenOutput g;
  g.text = std::move(ro.text);
  g.token_ids = std::move(ro.token_ids);
  g.finish_reason = std::move(ro.finish_reason);
  g.stop_reason = eng::to_json(ro.stop_reason);
  g.finished = ro.finished;
  g.num_cached_tokens = ro.num_cached_tokens;
  g.num_cache_creation_tokens = ro.num_cache_creation_tokens;
  if (!ro.spec_decode_metrics.is_nil()) g.spec_decode = eng::to_json(ro.spec_decode_metrics);
  g.engine_dead = ro.engine_dead;
  if (g.engine_dead) g.finished = true;
  return g;
}
}  // namespace

std::string EngineBackend::start(GenRequest&& r, GenCallback cb) {
  eng::EngineRequest er;
  try {
    eng::SamplingContext ctx = ctx_;
    ctx.max_model_len = eng_->max_model_len();
    mp::Value body = eng::to_mp(r.body);
    eng::BuildInput in;
    in.endpoint = ep_of(r.endpoint);
    in.body = &body;
    in.prompt_ids = std::move(r.prompt_ids);
    in.prompt_oov_id = std::move(r.prompt_oov_id);
    in.external_req_id = r.external_id;
    in.reasoning_ended = r.reasoning_ended;
    if (!r.reasoning_parser_kwargs.is_null()) in.reasoning_parser_kwargs = eng::to_mp(r.reasoning_parser_kwargs);
    in.structural_tag = r.structural_tag;
    in.priority = r.priority;
    in.session_id = r.session_id;
    in.data_parallel_rank = r.data_parallel_rank;
    in.data_parallel_rank_text = r.data_parallel_rank_text;
    if (const Json* cs = r.body.get("cache_salt"); cs && cs->is_string()) in.cache_salt = cs->as_string();
    if (!r.images.empty()) {
      std::vector<eng::DsImage> ims;
      for (size_t i = 0; i < r.images.size(); ++i) {
        const fe::ImageFeatures& f = *r.images[i];
        eng::DsImage d;
        d.patches_bf16 = f.patches_bf16.data();
        d.n_patches = f.n_patches;
        d.vit_grid[0] = f.vit_grid[0];
        d.vit_grid[1] = f.vit_grid[1];
        d.llm_grid[0] = f.llm_grid[0];
        d.llm_grid[1] = f.llm_grid[1];
        d.types = f.types;
        d.mm_hash = f.mm_hash;
        d.offset = r.image_offsets[i];
        ims.push_back(std::move(d));
      }
      in.mm_features = eng::make_deepseek_v41_image_features(ims);
    }
    in.arrival_time = r.arrival_time;
    er = eng::build_engine_request(in, ctx);
  } catch (...) {
    rethrow_input_error();
  }
  std::string internal = er.request_id;
  try {
    eng_->add_request(std::move(er), [cb = std::move(cb)](eng::RequestOutput&& ro) { cb(convert(std::move(ro))); });
  } catch (...) {
    rethrow_input_error();
  }
  return internal;
}

void EngineBackend::abort(const std::string& internal_id) { eng_->abort(internal_id, false); }

void EngineBackend::precheck_caps(const std::string& endpoint, const Json& body, int64_t prompt_len) {
  if (!ctx_.phys_caps) return;
  try {
    eng::SamplingContext ctx = ctx_;
    ctx.max_model_len = eng_->max_model_len();
    (void)eng::sampling_from_request(ep_of(endpoint), eng::to_mp(body), prompt_len, ctx);
  } catch (const eng::CapsError&) {
    rethrow_input_error();
  } catch (...) {
    // any other input error is raised where it always was (backend_->start)
  }
}

SamplingSummary EngineBackend::sampling_summary(const std::string& endpoint, const Json& body, int64_t prompt_len) {
  SamplingSummary s;
  try {
    eng::SamplingContext ctx = ctx_;
    ctx.max_model_len = eng_->max_model_len();
    eng::SamplingParams sp = eng::sampling_from_request(ep_of(endpoint), eng::to_mp(body), prompt_len, ctx);
    s.temperature = sp.temperature.value_or(1.0);
    s.top_p = sp.top_p.value_or(1.0);
    s.presence_penalty = sp.presence_penalty.value_or(0.0);
    s.frequency_penalty = sp.frequency_penalty.value_or(0.0);
    s.max_tokens = sp.max_tokens.value_or(0);
    s.logprobs = sp.logprobs;
  } catch (...) {
    rethrow_input_error();
  }
  return s;
}

}  // namespace fe::api
