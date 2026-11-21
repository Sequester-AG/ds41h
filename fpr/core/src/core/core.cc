// core.cc -- ICore: the EngineCore loop in C++. One loop thread does what vLLM's
// EngineCoreProc.run_busy_loop does with PhysScheduler + PhysExecutor plugged in (vllm/v1/engine/core.py):
//   _process_input_queue (block while there is no work, then drain), step (sync) or step_with_batch_queue (async,
//   batch_queue_size 2: schedule step N+1 before step N's results; a step whose structured requests have tokens in
//   flight defers its sampling until the previous step's update), _process_aborts_queue before every update, post_step
//   (sync: draft validation), the output thread's engine_index + MsgpackEncoder framing.
// The device side is IWorker (dispatch at sample time with the tables built at schedule time, as PhysExecutor's
// snapshot); merge_results turns its per-rank StepResults into the ModelRunnerOutput the lifecycle consumes.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "core.h"

namespace fpr {
namespace {

double mono_now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}
double wall_now() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

struct InMsg {
  uint8_t type;                       // v1::ReqType
  v1::EngineCoreRequest add;
  bool has_images = false;            // add_mm: images admitted (v2.2)
  std::vector<Image> images;
  std::string mm_reject;
  std::vector<Rid> abort;
  v1::UtilityCall util;
};

struct Pending {                       // a batch-queue entry: the step's scheduler output + its in-flight work
  core::SchedOut so;
  StepInput in;
  StepHandle* h = nullptr;
  bool executed = false;               // dispatched to the worker
  int64_t so_id = 0;                   // the loop's count of scheduler outputs (golden tests)
};

class Core final : public ICore {
 public:
  Core(const CoreConfig& cfg, std::unique_ptr<IScheduler> rs, BatchBuilder batches, std::unique_ptr<IWorker> worker,
       core::KVFacade kv, core::BlockHasher hasher, std::unique_ptr<grammar::Manager> som)
      : cfg_(cfg), rs_(std::move(rs)), batches_(std::move(batches)), worker_(std::move(worker)),
        som_(std::move(som)), lc_(cfg.lifecycle, rs_.get(), std::move(kv), std::move(hasher), som_.get()) {
    if (!cfg_.step_log.empty()) {
      slog_ = std::fopen(cfg_.step_log.c_str(), "a");
      if (slog_) std::setvbuf(slog_, nullptr, _IOLBF, 1 << 16);   // whole lines on disk (the process may exit unclosed)
    }
  }
  ~Core() override {
    stop();
    if (slog_) std::fclose(slog_);
  }

  Status add(const v1::EngineCoreRequest& r, std::string* err) override {
    InMsg m;
    m.type = v1::ADD;
    m.add = r;
    push(std::move(m));
    return Status::OK;
  }
  Status add_mm(const v1::EngineCoreRequest& r, const std::vector<Image>& images, const std::string& mm_reject,
                std::string* err) override {
    InMsg m;
    m.type = v1::ADD;
    m.add = r;
    m.has_images = mm_reject.empty();
    if (m.has_images) m.images = images;
    m.mm_reject = mm_reject;
    push(std::move(m));
    return Status::OK;
  }
  void abort(const std::vector<Rid>& rids) override {
    {
      std::lock_guard<std::mutex> g(amu_);
      aborts_.insert(aborts_.end(), rids.begin(), rids.end());   // aborts_queue (also handled from the input queue)
    }
    InMsg m;
    m.type = v1::ABORT;
    m.abort = rids;
    push(std::move(m));
  }
  Status utility(const v1::UtilityCall& call, std::string* err) override {
    InMsg m;
    m.type = v1::UTILITY;
    m.util = call;
    push(std::move(m));
    return Status::OK;
  }
  Status input(uint8_t type, const std::string& payload, std::string* err) override {
    if (type == v1::ADD) {
      v1::EngineCoreRequest r;
      if (!v1::decode_request(payload, &r, err)) return Status::BAD_ARG;
      return add(r, err);
    }
    if (type == v1::ABORT) {
      std::vector<Rid> ids;
      if (!v1::decode_abort(payload, &ids, err)) return Status::BAD_ARG;
      abort(ids);
      return Status::OK;
    }
    if (type == v1::UTILITY) {
      v1::UtilityCall u;
      if (!v1::decode_utility(payload, &u, err)) return Status::BAD_ARG;
      return utility(u, err);
    }
    if (err) *err = "unsupported request type";
    return Status::BAD_ARG;
  }
  void start() override {
    std::lock_guard<std::mutex> g(imu_);
    if (thr_.joinable()) return;
    run_ = true;
    thr_ = std::thread([this] { loop(); });
  }
  void stop() override {
    {
      std::lock_guard<std::mutex> g(imu_);
      run_ = false;
    }
    icv_.notify_all();
    if (thr_.joinable()) thr_.join();
  }
  bool step_once() override {          // InprocClient.get_output: drain inputs, step_fn, post_step
    {
      std::unique_lock<std::mutex> l(imu_);
      while (!inq_.empty()) {
        InMsg m = std::move(inq_.front());
        inq_.pop_front();
        l.unlock();
        handle(m);
        l.lock();
      }
    }
    return cfg_.lifecycle.async_mode ? step_batch_queue() : step_sync();
  }
  std::string fatal() const override {
    std::lock_guard<std::mutex> g(imu_);
    return fatal_;
  }
  bool has_work() const override { return work_.load(); }
  bool pop_outputs(std::vector<WireOut>* batch, double timeout_s) override {
    std::unique_lock<std::mutex> l(omu_);
    if (!ocv_.wait_for(l, std::chrono::duration<double>(timeout_s), [this] { return !outq_.empty(); })) return false;
    batch->clear();
    while (!outq_.empty()) {
      batch->push_back(std::move(outq_.front()));
      outq_.pop_front();
    }
    return true;
  }

 private:
  void push(InMsg m) {
    {
      std::lock_guard<std::mutex> g(imu_);
      inq_.push_back(std::move(m));
    }
    icv_.notify_one();
  }
  bool has_work_() const { return lc_.has_requests() || !bq_.empty(); }

  void handle(InMsg& m) {   // _handle_client_request
    if (m.type == v1::ADD) {
      lc_.add_request(m.add, clk(cfg_.clocks.add), clk_wall(), m.has_images ? &m.images : nullptr, m.mm_reject);
    } else if (m.type == v1::ABORT) {
      lc_.abort_requests(m.abort);
    } else if (m.type == v1::UTILITY) {
      v1::UtilityOutput u;
      u.call_id = m.util.call_id;
      run_utility(m.util, &u);
      v1::EngineCoreOutputs e;
      e.utility_output = u;
      e.timestamp = clk(cfg_.clocks.output);
      emit(m.util.client_index, std::move(e));
    }
  }

  void run_utility(const v1::UtilityCall& c, v1::UtilityOutput* u) {
    // the EngineCore methods the frontend calls on a phys server; anything else fails as getattr would
    const std::string& n = c.method;
    auto su = cfg_.static_utilities.find(n);
    if (su != cfg_.static_utilities.end()) {
      u->has_result = true;
      u->result = su->second;
    } else if (n == "reset_prefix_cache") {
      u->has_result = true;
      u->result = mp::Val::bool_(lc_.reset_prefix_cache());
    } else if (n == "reset_mm_cache" || n == "reset_encoder_cache" || n == "profile" || n == "execute_dummy_batch") {
      u->has_result = true;                              // None
    } else if (n == "is_sleeping") {
      u->has_result = true;
      u->result = mp::Val::bool_(false);
    } else if (n == "is_scheduler_paused") {
      u->has_result = true;
      u->result = mp::Val::bool_(paused_ != core::Lifecycle::Pause::UNPAUSED);
    } else if (n == "pause_scheduler") {               // mode "keep" semantics only (requests stay queued)
      paused_ = core::Lifecycle::Pause::PAUSED_ALL;
      lc_.set_pause(paused_);
      u->has_result = true;
    } else if (n == "resume_scheduler") {
      paused_ = core::Lifecycle::Pause::UNPAUSED;
      lc_.set_pause(paused_);
      u->has_result = true;
    } else {
      u->failure_message = "Call to " + n + " method failed: 'EngineCoreProc' object has no attribute '" + n + "'";
    }
  }

  void emit(int64_t client, v1::EngineCoreOutputs e) {   // output thread: engine_index + encode
    e.engine_index = cfg_.engine_index;
    WireOut w;
    w.client_index = client;
    v1::encode_outputs(e, &w.frames, cfg_.zero_copy_threshold);
    w.msg = std::move(e);
    {
      std::lock_guard<std::mutex> g(omu_);
      outq_.push_back(std::move(w));
    }
    ocv_.notify_one();
  }

  void process_input_queue() {
    std::unique_lock<std::mutex> l(imu_);
    while (!has_work_() && run_) {
      if (inq_.empty()) {
        std::lock_guard<std::mutex> g(amu_);
        aborts_.clear();                                 // all aborts are also in the input queue
      }
      icv_.wait(l, [this] { return !inq_.empty() || !run_; });
      if (!run_) return;
      InMsg m = std::move(inq_.front());
      inq_.pop_front();
      l.unlock();
      handle(m);
      l.lock();
    }
    while (!inq_.empty()) {
      InMsg m = std::move(inq_.front());
      inq_.pop_front();
      l.unlock();
      handle(m);
      l.lock();
    }
  }

  void process_aborts() {
    std::vector<Rid> ids;
    {
      std::lock_guard<std::mutex> g(amu_);
      ids.swap(aborts_);
    }
    if (!ids.empty()) lc_.abort_requests(ids);
  }

  // PhysScheduler._phys_step + PhysExecutor.execute_model's part: the step's host input, frozen at schedule time
  void build_input(Pending* p) {
    StepInput& in = p->in;
    const core::SchedOut& so = p->so;
    in.step = so.phys_step;
    in.plan = so.has_plan ? &p->so.plan : nullptr;
    in.total_rows = so.total;
    in.new_reqs = so.new_reqs;
    in.finished = so.finished_req_ids;
    in.preempted = so.phys_preempted;
    for (const auto& kv : so.num_scheduled) in.order.push_back(kv.first);
    for (const auto& kv : so.spec) in.num_spec.emplace_back(kv.first, (int32_t)kv.second.size());
    if (so.has_plan) {
      in.est_us = so.plan.est_us;
      if (batches_) {                                  // interfaces.build_step_batch per rank
        in.batches.resize(W);
        for (int r = 0; r < W; ++r) batches_(so.plan, r, &in.batches[r]);
      }
      if (!so.plan.work.empty()) lc_.row_tokens(in.batches, &in.row_tokens, &in.prev_tokens);
      else in.row_tokens.assign(in.batches.size(), {});
    }
  }

  void add_grammar(Pending* p, const core::GrammarOut& g) {   // split_bitmask
    p->in.grammar.clear();
    int64_t row = 0;
    for (size_t i = 0; i < g.rids.size(); ++i) {
      GrammarRows gr;
      gr.rid = g.rids[i];
      int64_t ns = 0;
      for (const auto& s : p->so.spec)
        if (s.first == gr.rid) ns = (int64_t)s.second.size();
      gr.n_rows = 1 + ns;
      gr.words = g.words;
      gr.rows.assign(g.bitmask.begin() + row * g.words, g.bitmask.begin() + (row + gr.n_rows) * g.words);
      gr.num_acceptable = i < g.num_acceptable_drafts.size() ? g.num_acceptable_drafts[i] : ns;
      row += gr.n_rows;
      p->in.grammar.push_back(std::move(gr));
    }
  }

  bool sample(Pending* p) {   // sample_tokens: the dispatch (sync: the whole step)
    if (worker_->dispatch(p->in, &p->h) != Status::OK) {
      fail("worker dispatch failed at step " + std::to_string(p->so.phys_step));
      return false;
    }
    p->executed = true;
    return true;
  }
  void fail(const std::string& why) {   // the executor failed: stop the loop; the process sends ENGINE_CORE_DEAD
    std::lock_guard<std::mutex> g(imu_);
    if (fatal_.empty()) fatal_ = why;
    run_ = false;
  }

  void result(Pending* p, core::ModelOut* mo) {   // future.result(): wait + merge_results
    std::vector<StepResult> res;
    if (p->executed && worker_->wait(p->h, &res) != Status::OK) fail("worker wait failed at step " +
                                                                      std::to_string(p->so.phys_step));
    if (cfg_.hooks.merge) cfg_.hooks.merge(p->so, res, mo);
    else core::merge_results(p->so, res, mo);
    last_drafts_ = mo->has_drafts ? mo->drafts : std::vector<std::pair<Rid, std::vector<int32_t>>>();
    has_last_drafts_ = mo->has_drafts;
  }

  void deliver(const core::UpdateOut& u) {
    for (const auto& kv : u.outputs) emit(kv.first, kv.second);
  }

  bool step_sync() {   // EngineCore.step + post_step
    if (!lc_.has_requests()) return false;
    auto p = std::make_unique<Pending>();
    sched(p.get());
    build_input(p.get());
    core::ModelOut mo;
    if (p->so.total == 0) {
      if (!p->in.finished.empty() || !p->in.preempted.empty() || !p->in.new_reqs.empty()) {
        sample(p.get());                               // frees only
        std::vector<StepResult> res;
        worker_->wait(p->h, &res);
      }
      if (cfg_.hooks.merge) cfg_.hooks.merge(p->so, {}, &mo);
      else core::merge_results(p->so, {}, &mo);
    } else {
      bitmask(p.get());
      sample(p.get());
      result(p.get(), &mo);
    }
    process_aborts();
    update(p.get(), mo);
    bool executed = p->so.total > 0;
    if (cfg_.check_for_draft_tokens && executed && has_last_drafts_) {   // post_step
      lc_.update_draft_token_ids(last_drafts_);
      has_last_drafts_ = false;
      if (cfg_.hooks.on_call) cfg_.hooks.on_call("drafts", p->so, nullptr, nullptr);
    }
    return executed;
  }

  // the lifecycle calls of the loop, each reported to hooks.on_call
  void sched(Pending* p) {
    double m = clk(cfg_.clocks.sched);
    p->so = lc_.schedule(m, clk_wall());
    p->so_id = ++so_seq_;
    if (cfg_.hooks.on_call) cfg_.hooks.on_call("schedule", p->so, nullptr, nullptr);
  }
  void bitmask(Pending* p) {
    core::GrammarOut g = lc_.get_grammar_bitmask(p->so);
    add_grammar(p, g);
    if (cfg_.hooks.on_call) cfg_.hooks.on_call("bitmask", p->so, &g, nullptr);
  }
  void update(Pending* p, const core::ModelOut& mo) {
    double m = clk(cfg_.clocks.update);
    auto u = lc_.update_from_output(p->so, mo, m, [this]() { return clk(cfg_.clocks.output); });
    if (cfg_.hooks.on_call) cfg_.hooks.on_call("update", p->so, nullptr, &u);
    deliver(u);
  }
  double clk(const std::function<double()>& f) { return f ? f() : mono_now(); }
  double clk_wall() { return cfg_.clocks.wall ? cfg_.clocks.wall() : wall_now(); }

  // ---- the step log (CoreConfig.step_log): per call of step_batch_queue
  struct StepLog {
    double t0 = 0, sched = 0, disp = 0, wait = 0, upd = 0;
    int64_t rows = -1, dec = -1, step = -1, waited_step = -1;
    bool any = false;
  };
  static double mono_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void log_step(const StepLog& l) {
    if (!slog_ || !l.any) return;
    std::fprintf(slog_, "{\"t\":%.6f,\"sched_ms\":%.3f,\"disp_ms\":%.3f,\"wait_ms\":%.3f,\"upd_ms\":%.3f,\"rows\":%lld,"
                        "\"dec\":%lld,\"step\":%lld,\"waited\":%lld,\"bq\":%zu}\n",
                 l.t0, 1e3 * l.sched, 1e3 * l.disp, 1e3 * l.wait, 1e3 * l.upd, (long long)l.rows, (long long)l.dec,
                 (long long)l.step, (long long)l.waited_step, bq_.size());
  }

  bool step_batch_queue() {   // EngineCore.step_with_batch_queue (batch_queue_size 2)
    StepLog sl;
    if (slog_) sl.t0 = mono_now();
    bool r = step_batch_queue_(&sl);
    log_step(sl);
    return r;
  }
  bool step_batch_queue_(StepLog* sl) {
    bool executed = false;
    std::unique_ptr<Pending> deferred;
    if (lc_.has_requests()) {
      auto p = std::make_unique<Pending>();
      double ta = slog_ ? mono_now() : 0;
      sched(p.get());
      build_input(p.get());
      executed = p->so.total > 0;
      if (slog_) {
        sl->any = true;
        sl->sched = mono_now() - ta;
        sl->rows = p->so.total;
        sl->step = p->in.step;
        int64_t nd = 0;
        if (p->in.plan)
          for (const auto& w : p->in.plan->work) nd += w.kind == "decode";
        sl->dec = nd;
        ta = mono_now();
      }
      if (!executed) {
        if (!p->in.finished.empty() || !p->in.preempted.empty() || !p->in.new_reqs.empty()) {
          sample(p.get());
          std::vector<StepResult> res;
          worker_->wait(p->h, &res);
          p->executed = false;
        }
      } else if (!p->so.pending_structured_output_tokens) {
        bitmask(p.get());
        sample(p.get());
      } else {
        deferred = std::move(p);
      }
      if (slog_) sl->disp = mono_now() - ta;
      if (!deferred) {
        bq_.push_front(std::move(p));
        if (bq_.size() < 2 && (executed || lc_.has_requests())) return executed;
      }
    } else if (bq_.empty()) {
      return false;
    }
    std::unique_ptr<Pending> p = std::move(bq_.back());
    bq_.pop_back();
    core::ModelOut mo;
    double tw = slog_ ? mono_now() : 0;
    result(p.get(), &mo);
    if (slog_) {
      sl->any = true;
      sl->wait = mono_now() - tw;
      sl->waited_step = p->in.step;
      tw = mono_now();
    }
    process_aborts();
    update(p.get(), mo);
    if (slog_) sl->upd = mono_now() - tw;
    if (deferred) {
      if (cfg_.check_for_draft_tokens && has_last_drafts_) {
        lc_.update_draft_token_ids_in_output(last_drafts_, &deferred->so);
        has_last_drafts_ = false;
        if (cfg_.hooks.on_call) cfg_.hooks.on_call("drafts_in_output", deferred->so, nullptr, nullptr);
      }
      bitmask(deferred.get());
      sample(deferred.get());
      bq_.push_front(std::move(deferred));
    }
    return executed;
  }

  void loop() {
    for (;;) {
      {
        std::lock_guard<std::mutex> g(imu_);
        if (!run_) break;
      }
      process_input_queue();
      {
        std::lock_guard<std::mutex> g(imu_);
        if (!run_) break;
      }
      bool executed = cfg_.lifecycle.async_mode ? step_batch_queue() : step_sync();
      work_ = has_work_();
      if (!executed && lc_.has_requests()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  CoreConfig cfg_;
  std::unique_ptr<IScheduler> rs_;
  BatchBuilder batches_;
  std::unique_ptr<IWorker> worker_;
  std::unique_ptr<grammar::Manager> som_;
  core::Lifecycle lc_;
  core::Lifecycle::Pause paused_ = core::Lifecycle::Pause::UNPAUSED;
  std::deque<std::unique_ptr<Pending>> bq_;
  int64_t so_seq_ = 0;
  std::vector<std::pair<Rid, std::vector<int32_t>>> last_drafts_;
  bool has_last_drafts_ = false;
  mutable std::mutex imu_;
  std::mutex amu_, omu_;
  std::condition_variable icv_, ocv_;
  std::deque<InMsg> inq_;
  std::vector<Rid> aborts_;
  std::deque<WireOut> outq_;
  bool run_ = false;
  std::string fatal_;                  // guarded by imu_
  std::atomic<bool> work_{false};      // has_work_() after the loop's last iteration (any thread may read)
  FILE* slog_ = nullptr;               // CoreConfig.step_log (loop thread only)
  std::thread thr_;
};

}  // namespace

std::unique_ptr<ICore> make_core(const CoreConfig& cfg, std::unique_ptr<IScheduler> rs, BatchBuilder batches,
                                 std::unique_ptr<IWorker> worker, core::KVFacade kv, core::BlockHasher hasher,
                                 std::string* err) {
  std::unique_ptr<grammar::Manager> som;
  if (cfg.structured_outputs) {
    auto be = grammar::make_xgrammar_backend(cfg.grammar, err);
    if (!be) return nullptr;
    som = std::make_unique<grammar::Manager>(std::move(be), cfg.lifecycle.num_spec_tokens,
                                             cfg.lifecycle.enable_in_reasoning, cfg.grammar_async);
  }
  return std::unique_ptr<ICore>(new Core(cfg, std::move(rs), std::move(batches), std::move(worker), std::move(kv),
                                         std::move(hasher), std::move(som)));
}

}  // namespace fpr
