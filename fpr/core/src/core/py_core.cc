// py_core.cc
//   import fpr_core_py as C
//   core = C.Core(cfg: dict, rs: object, kv_usage: callable, est_cached: callable, hasher: callable | None,
//                 tokenizer_info_json: str | None, reasoners: list[(key, dict)])
//   core.add_request(payload: bytes, mono, wall)        # EngineCoreRequest msgpack (vLLM's MsgpackEncoder)
//   core.abort(rids)
//   h, so = core.schedule(mono)                         # so: dict of SchedulerOutput fields + "plan" (as rs returned it)
//   rids, rows, n_rows, words, acc = core.grammar_bitmask(h)
//   msgs, accepted = core.update(h, out: dict, mono_rs, timestamps)   # msgs: [(client, [frame bytes])]
//   core.update_draft_token_ids([(rid, ids)]); core.update_draft_token_ids_in_output(h, [(rid, ids)]) -> so dict
// `rs` is the rank scheduler: today the Python RankScheduler behind a thin shim (methods named as IScheduler, see
// PyRS), later the C++ one bound directly, with no Python on the step path. The GIL is released inside every
// core call and re-acquired only for calls into Python objects (rs / kv / hasher).
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <memory>

#include "core.h"
#include "blockhash.h"
#include "tables.h"
#include "py_cfg.h"

namespace py = pybind11;
using namespace fpr;

namespace {

ReqState to_state(const py::tuple& t) {   // (phase, rank, uid, num_computed, num_output, inflight, replay_start,
  ReqState s;                              //  cached_tokens, skip_until_step, first_pending)
  s.phase = (Phase)t[0].cast<int>();
  s.rank = t[1].cast<int32_t>();
  s.uid = t[2].cast<int32_t>();
  s.num_computed = t[3].cast<int64_t>();
  s.num_output = t[4].cast<int64_t>();
  s.inflight = t[5].cast<int64_t>();
  s.replay_start = t[6].cast<int64_t>();
  s.cached_tokens = t[7].cast<int64_t>();
  s.skip_until_step = t[8].cast<int64_t>();
  s.first_pending = t[9].cast<bool>();
  return s;
}

StepPlan to_plan(const py::dict& p) {     // the gtrace_core.plan_json form
  StepPlan out;
  out.step = p["step"].cast<int64_t>();
  for (auto wh : p["work"]) {
    py::dict w = wh.cast<py::dict>();
    SeqWork s;
    s.rid = w["rid"].cast<std::string>();
    s.rank = w["rank"].cast<int32_t>();
    s.kind = w["kind"].cast<std::string>();
    s.start = w["start"].cast<int64_t>();
    s.n = w["n"].cast<int64_t>();
    s.cached_upto = w["cached_upto"].cast<int64_t>();
    s.replay_start = w["replay_start"].cast<int64_t>();
    s.last_chunk = w["last_chunk"].cast<bool>();
    s.score = w["score"].cast<bool>();
    s.forced = w["forced"].cast<bool>();
    s.uid = w["uid"].cast<int32_t>();
    s.known = w["known"].cast<int64_t>();
    if (!w["cp"].is_none()) {
      py::dict c = w["cp"].cast<py::dict>();
      s.has_cp = true;
      s.cp.group = c["group"].cast<int64_t>();
      s.cp.rid = c["rid"].cast<std::string>();
      s.cp.owner = c["owner"].cast<int32_t>();
      s.cp.me = c["me"].cast<int32_t>();
    }
    out.work.push_back(std::move(s));
  }
  for (auto r : p["preempted"]) out.preempted.push_back(r.cast<std::string>());
  for (auto r : p["migrated"]) out.migrated.push_back(r.cast<std::string>());
  out.est_us = p["est_us"].cast<double>();
  out.idle_frac = p["idle_frac"].cast<double>();
  return out;
}

class PyRS final : public IScheduler {    // IScheduler over a Python object (GIL taken per call)
 public:
  explicit PyRS(py::object o) : o_(std::move(o)) {}
  ~PyRS() override {
    py::gil_scoped_acquire g;
    o_ = py::object();
  }
  Status add_request(const Request& r, std::string* why) override {
    py::gil_scoped_acquire g;
    py::tuple t = o_.attr("add_request")(r.rid, r.prompt_ids.size(), r.sampling.max_tokens, r.arrival, r.priority,
                                         r.sampling.prompt_logprobs);
    if (t[0].cast<bool>()) return Status::OK;
    if (why && !t[1].is_none()) *why = t[1].cast<std::string>();
    return Status::REJECTED;
  }
  void abort(const Rid& rid) override {
    py::gil_scoped_acquire g;
    o_.attr("abort")(rid);
  }
  StepPlan schedule(double now) override {
    py::gil_scoped_acquire g;
    return to_plan(o_.attr("schedule")(now).cast<py::dict>());
  }
  std::vector<Rid> update(const StepPlan& plan, const std::vector<std::pair<Rid, int32_t>>& acc, double now) override {
    py::gil_scoped_acquire g;
    return o_.attr("update")(plan.step, acc, now).cast<std::vector<Rid>>();
  }
  int64_t max_new_cap(int64_t prompt_len) const override {
    py::gil_scoped_acquire g;
    return o_.attr("max_new_cap")(prompt_len).cast<int64_t>();
  }
  int32_t seats_per_rank() const override {
    py::gil_scoped_acquire g;
    return o_.attr("seats_per_rank")().cast<int32_t>();
  }
  void extend_block_hashes(const Rid&, int64_t, const std::vector<Hash>&) override {}   // 2a: rs reads vLLM's hashes
  void hold_until(const Rid& rid, int64_t step) override {
    py::gil_scoped_acquire g;
    o_.attr("hold_until")(rid, step);
  }
  void sync_num_output(const Rid& rid, int64_t n) override {
    py::gil_scoped_acquire g;
    o_.attr("sync_num_output")(rid, n);
  }
  void set_paused_new(bool p) override {
    py::gil_scoped_acquire g;
    o_.attr("set_paused_new")(p);
  }
  void reset_kv(std::unique_ptr<IKVManager>) override {}
  bool get(const Rid& rid, ReqState* out) const override {
    py::gil_scoped_acquire g;
    py::object t = o_.attr("get")(rid);
    if (t.is_none()) return false;
    *out = to_state(t.cast<py::tuple>());
    return true;
  }
  std::vector<Rid> running() const override {
    py::gil_scoped_acquire g;
    return o_.attr("running")().cast<std::vector<Rid>>();
  }
  int64_t step_id() const override {
    py::gil_scoped_acquire g;
    return o_.attr("step_id")().cast<int64_t>();
  }
  std::string stats_json() const override { return "{}"; }

 private:
  py::object o_;
};

py::dict so_dict(const core::SchedOut& so) {
  py::dict d;
  d["num_scheduled"] = so.num_scheduled;
  d["spec"] = so.spec;
  d["finished"] = so.finished_req_ids;
  d["preempted"] = so.preempted_req_ids;
  d["phys_preempted"] = so.phys_preempted;
  d["num_invalid"] = so.num_invalid_spec_tokens;
  d["total"] = so.total;
  d["has_so"] = so.has_structured_output_requests;
  d["pending_so"] = so.pending_structured_output_tokens;
  d["step"] = so.phys_step;
  py::list work;
  if (so.has_plan)
    for (const auto& w : so.plan.work) work.append(py::make_tuple(w.rid, w.kind, w.n, w.last_chunk, w.rank));
  d["work"] = work;
  py::list nr;
  for (const auto& r : so.new_reqs) nr.append(py::make_tuple(r.rid, r.rank, r.prompt, r.output, r.structured));
  d["new_reqs"] = nr;
  return d;
}

class PyCore {
 public:
  PyCore(py::dict cfg, py::object rs, py::object kv_usage, py::object est_cached, py::object hasher,
         py::object tokenizer_info, py::list reasoners) {
    core::LifecycleConfig c;
    c.num_spec_tokens = cfg["k"].cast<int32_t>();
    c.max_model_len = cfg["max_model_len"].cast<int64_t>();
    c.page_tokens = cfg["page_tokens"].cast<int32_t>();
    c.async_mode = cfg["async"].cast<bool>();
    c.log_stats = cfg["log_stats"].cast<bool>();
    c.spec_decode_metrics_level = cfg["spec_decode_metrics_level"].cast<std::string>();
    c.include_finished_set = cfg["include_finished_set"].cast<bool>();
    c.grammar_hold = cfg["grammar_hold"].cast<bool>();
    c.grammar_max_frac = cfg["grammar_max_frac"].cast<double>();
    c.vocab_size = cfg["vocab_size"].cast<int32_t>();
    c.t0_mono = cfg["t0"].cast<double>();
    if (!cfg["sampling_caps"].is_none()) {
      c.has_sampling_caps = true;
      for (auto x : cfg["sampling_caps"]) c.sampling_caps.insert(x.cast<std::string>());
    }
    for (auto item : reasoners) {
      py::tuple t = item.cast<py::tuple>();
      py::dict j = t[1].cast<py::dict>();
      grammar::ReasonerSpec s;
      s.present = j["present"].cast<bool>();
      s.adapter = j["adapter"].cast<bool>();
      s.end_ids = j["end_ids"].cast<std::vector<int32_t>>();
      s.end_token_id = j["end_token_id"].cast<int32_t>();
      s.start_id = j["start_id"].cast<int32_t>();
      s.boundary_ids = j["boundary_ids"].cast<std::vector<int32_t>>();
      s.wait_for_reasoning = j["wait_for_reasoning"].cast<bool>();
      s.start_opens_reasoning = j["start_opens_reasoning"].cast<bool>();
      c.reasoners.emplace_back(t[0].cast<std::string>(), s);
    }
    rs_ = std::make_unique<PyRS>(rs);
    kv_usage_ = kv_usage;
    est_ = est_cached;
    core::KVFacade kv;
    kv.usage = [this]() {
      py::gil_scoped_acquire g;
      return kv_usage_().cast<double>();
    };
    kv.estimate_cached_tokens = [this](const Rid& rid) {
      py::gil_scoped_acquire g;
      return est_(rid).cast<int64_t>();
    };
    if (!tokenizer_info.is_none()) {
      grammar::BackendConfig bc;
      bc.tokenizer_info_json = tokenizer_info.cast<std::string>();
      bc.vocab_size = c.vocab_size;
      bc.num_spec_tokens = c.num_spec_tokens;
      if (cfg.contains("disable_any_whitespace")) bc.disable_any_whitespace = cfg["disable_any_whitespace"].cast<bool>();
      std::string err;
      auto be = grammar::make_xgrammar_backend(bc, &err);
      if (!be) throw std::runtime_error("fpr_core_py: grammar backend: " + err);
      som_ = std::make_unique<grammar::Manager>(std::move(be), c.num_spec_tokens, c.enable_in_reasoning, true);
    }
    lc_ = std::make_unique<core::Lifecycle>(c, rs_.get(), kv, nullptr, som_.get());
  }

  void add_request(py::bytes payload, double mono, double wall) {
    std::string p = payload;
    v1::EngineCoreRequest r;
    std::string err;
    if (!v1::decode_request(p, &r, &err)) throw std::runtime_error("fpr_core_py: EngineCoreRequest: " + err);
    py::gil_scoped_release nogil;
    lc_->add_request(r, mono, wall);
  }
  void abort(const std::vector<Rid>& rids) {
    py::gil_scoped_release nogil;
    lc_->abort_requests(rids);
  }
  py::tuple schedule(double mono, double wall) {
    int64_t h = ++next_;
    {
      py::gil_scoped_release nogil;
      sos_[h] = lc_->schedule(mono, wall);
    }
    return py::make_tuple(h, so_dict(sos_[h]));
  }
  py::tuple grammar_bitmask(int64_t h) {
    core::GrammarOut g;
    {
      py::gil_scoped_release nogil;
      g = lc_->get_grammar_bitmask(sos_.at(h));
    }
    return py::make_tuple(g.rids, py::bytes((const char*)g.bitmask.data(), g.bitmask.size() * 4), g.rows, g.words,
                          g.num_acceptable_drafts);
  }
  // out: {"req_ids": [...], "sampled": [[...]], "logprobs": None | {"ids", "vals", "ranks", "cu"}, "drafts": None | [...]}
  py::tuple update(int64_t h, py::dict out, double mono_rs, const std::vector<double>& ts) {
    core::ModelOut mo;
    mo.req_ids = out["req_ids"].cast<std::vector<Rid>>();
    mo.sampled = out["sampled"].cast<std::vector<std::vector<int32_t>>>();
    if (!out["logprobs"].is_none()) {
      py::dict l = out["logprobs"].cast<py::dict>();
      auto ids = l["ids"].cast<std::vector<std::vector<int64_t>>>();
      auto vals = l["vals"].cast<std::vector<std::vector<float>>>();
      auto ranks = l["ranks"].cast<std::vector<int64_t>>();
      mo.has_logprobs = true;
      for (size_t j = 0; j < ids.size(); ++j) {
        mo.lp_width = (int64_t)ids[j].size();
        mo.lp_rows.push_back(core::LpRow{ids[j], vals[j], ranks[j]});
      }
      mo.lp_cu = l["cu"].cast<std::vector<int64_t>>();
    }
    if (!out["drafts"].is_none()) {
      mo.has_drafts = true;
      mo.drafts = out["drafts"].cast<std::vector<std::pair<Rid, std::vector<int32_t>>>>();
    }
    core::UpdateOut u;
    std::vector<std::pair<int64_t, std::vector<std::string>>> frames;
    {
      py::gil_scoped_release nogil;
      u = lc_->update_from_output(sos_.at(h), mo, mono_rs, ts);
      sos_.erase(h);
      for (auto& kv : u.outputs) {
        std::vector<std::string> fr;
        v1::encode_outputs(kv.second, &fr);
        frames.emplace_back(kv.first, std::move(fr));
      }
    }
    py::list msgs;
    for (auto& f : frames) {
      py::list fl;
      for (auto& b : f.second) fl.append(py::bytes(b));
      msgs.append(py::make_tuple(f.first, fl));
    }
    return py::make_tuple(msgs, u.accepted);
  }
  void update_draft_token_ids(const std::vector<std::pair<Rid, std::vector<int32_t>>>& d) {
    py::gil_scoped_release nogil;
    lc_->update_draft_token_ids(d);
  }
  py::dict update_draft_token_ids_in_output(int64_t h, const std::vector<std::pair<Rid, std::vector<int32_t>>>& d) {
    {
      py::gil_scoped_release nogil;
      lc_->update_draft_token_ids_in_output(d, &sos_.at(h));
    }
    return so_dict(sos_.at(h));
  }
  py::object spec_token_ids(const Rid& rid) {
    const core::VReq* r = lc_->get(rid);
    if (!r) return py::none();
    return py::cast(r->spec_token_ids);
  }
  bool has_requests() const { return lc_->has_requests(); }
  int64_t num_unfinished() const { return lc_->num_unfinished(); }

 private:
  std::unique_ptr<PyRS> rs_;
  std::unique_ptr<grammar::Manager> som_;
  std::unique_ptr<core::Lifecycle> lc_;
  py::object kv_usage_, est_;
  std::map<int64_t, core::SchedOut> sos_;
  int64_t next_ = 0;
};

// prefix-cache keys come from the core's SHA-256 block
// hasher (blockhash.h). One schedule() call returns everything the executor needs: the SchedulerOutput fields, the plan
// (tuples; the adapter rebuilds serve's StepPlan objects), the PhysStep row / prev tokens and the new requests.
class CoreX {
 public:
  CoreX(py::dict cfg, py::capsule sched, py::capsule kvc, py::object sched_owner, py::object tokenizer_info,
        py::list reasoners)
      : owner_(sched_owner) {
    rs_ = static_cast<IScheduler*>(sched.get_pointer());
    kv_ = static_cast<IKVManager*>(kvc.get_pointer());
    if (std::string(sched.name()) != "fpr.IScheduler" || std::string(kvc.name()) != "fpr.IKVManager")
      throw std::runtime_error("fpr_core_py.CoreX: capsules must be fpr.IScheduler / fpr.IKVManager");
    core::LifecycleConfig c = pycfg::lifecycle(cfg, reasoners);
    window_page_tokens_ = cfg.contains("window_page_tokens") ? cfg["window_page_tokens"].cast<int64_t>() : WINDOW_PAGE_TOKENS;
    IKVManager* kvp = kv_;
    core::KVFacade kv = core::kv_facade_of([kvp]() -> const IKVManager* { return kvp; });
    if (!tokenizer_info.is_none()) {
      std::string err;
      auto be = grammar::make_xgrammar_backend(pycfg::backend(cfg, tokenizer_info.cast<std::string>(), c), &err);
      if (!be) throw std::runtime_error("fpr_core_py: grammar backend: " + err);
      som_ = std::make_unique<grammar::Manager>(std::move(be), c.num_spec_tokens, c.enable_in_reasoning, true);
    }
    core::BlockHasher hasher = pycfg::hasher(c.page_tokens);
    lc_ = std::make_unique<core::Lifecycle>(c, rs_, kv, hasher, som_.get());
  }

  // images: None or [(offset, length, key)]; mm_reject: the adapter's reason to reject the request's multimodal inputs;
  // forced: None or the force file's "forced" ids (the adapter's ForceSpec matched the prompt)
  void add_request(py::bytes payload, double mono, double wall, py::object images, const std::string& mm_reject,
                   py::object forced) {
    std::string p = payload;
    v1::EngineCoreRequest r;
    std::string err;
    if (!v1::decode_request(p, &r, &err)) throw std::runtime_error("fpr_core_py: EngineCoreRequest: " + err);
    std::vector<Image> ims;
    bool have = !images.is_none();
    if (have)
      for (auto it : images) {
        py::tuple t = it.cast<py::tuple>();
        ims.push_back(Image{t[0].cast<int64_t>(), t[1].cast<int64_t>(), t[2].cast<std::string>()});
      }
    if (have) r.has_mm_features = true;
    std::vector<int32_t> fids;
    bool fhave = !forced.is_none();
    if (fhave) fids = forced.cast<std::vector<int32_t>>();
    py::gil_scoped_release nogil;
    lc_->add_request(r, mono, wall, have ? &ims : nullptr, mm_reject, fhave ? &fids : nullptr);
  }
  void abort(const std::vector<Rid>& rids) {
    py::gil_scoped_release nogil;
    lc_->abort_requests(rids);
  }
  // -> (h, so dict, plan tuple | None, row_tokens [W], prev_tokens [(rid, ids)], new_reqs [(rid, rank, prompt, output, so)])
  // so dict FORCE keys: forced_to_rs [(rid, accepted)], forced_emits [(rid, ids, checked_rows, input_mismatch)]
  py::tuple schedule(double mono, double wall) {
    int64_t h = ++next_;
    std::vector<std::vector<int32_t>> rows;
    std::vector<std::pair<Rid, std::vector<int32_t>>> prev;
    std::vector<core::ForcedEmit> emits;
    std::string err;
    bool bad = false;
    {
      py::gil_scoped_release nogil;
      core::SchedOut& so = sos_[h] = lc_->schedule(mono, wall);
      if (so.has_plan && !so.plan.work.empty()) {
        KVViews v;
        IKVManager* kvp = kv_;
        v.seq = [kvp](const Rid& id) { return kvp->seq(id); };
        std::vector<StepBatch> b(W);
        for (int r = 0; r < W && !bad; ++r)
          if (build_step_batch(so.plan, v, PAGE_TOKENS, window_page_tokens_, r, &b[r], &err) != Status::OK) bad = true;
        if (!bad) lc_->row_tokens(b, &rows, &prev, &emits);
      } else if (so.has_plan) {
        rows.assign(W, {});
      }
    }
    if (bad) throw std::runtime_error("fpr_core_py: build_step_batch: " + err);
    const core::SchedOut& so = sos_[h];
    py::object plan = py::none();
    if (so.has_plan) {
      py::list work;
      for (const auto& w : so.plan.work) {
        py::object cp = py::none();
        if (w.has_cp) {
          py::list subs;
          for (const auto& t : w.cp.subs) subs.append(py::make_tuple(std::get<0>(t), std::get<1>(t), std::get<2>(t)));
          cp = py::make_tuple(w.cp.group, w.cp.rid, w.cp.owner, subs, w.cp.me);
        }
        py::list ims;
        for (const auto& im : w.images) ims.append(py::make_tuple(im.offset, im.length, im.key));
        work.append(py::make_tuple(w.rid, w.rank, w.kind, w.start, w.n, w.cached_upto, w.replay_start, w.last_chunk,
                                   w.score, w.forced, w.uid, w.known, cp, ims));
      }
      py::list copies;
      for (const auto& c : so.plan.copies)
        copies.append(py::make_tuple(c.pool, c.src_rank, c.src_page, c.dst_rank, c.dst_page, c.step));
      plan = py::make_tuple(so.plan.step, work, copies, so.plan.preempted, so.plan.migrated, so.plan.est_us,
                            so.plan.idle_frac);
    }
    py::list nr;
    for (const auto& r : so.new_reqs) nr.append(py::make_tuple(r.rid, r.rank, r.prompt, r.output, r.structured));
    py::dict sd = so_dict(so);
    py::list fe;
    for (const auto& e : emits) fe.append(py::make_tuple(e.rid, e.ids, e.checked_rows, e.input_mismatch));
    sd["forced_to_rs"] = so.forced_to_rs;
    sd["forced_emits"] = fe;
    return py::make_tuple(h, sd, plan, rows, prev, nr);
  }
  py::tuple grammar_bitmask(int64_t h) {
    core::GrammarOut g;
    {
      py::gil_scoped_release nogil;
      g = lc_->get_grammar_bitmask(sos_.at(h));
    }
    return py::make_tuple(g.rids, py::bytes((const char*)g.bitmask.data(), g.bitmask.size() * 4), g.rows, g.words,
                          g.num_acceptable_drafts);
  }
  // sampled: per req_ids entry; lp: None | (ids [[...]], vals [[...]], ranks [...], cu [...]); drafts: None | [(rid, ids)]
  // -> ([(client, [frame bytes])], accepted)
  py::tuple update(int64_t h, const std::vector<Rid>& req_ids, const std::vector<std::vector<int32_t>>& sampled,
                   py::object lp, py::object drafts, double mono_rs, double ts) {
    core::ModelOut mo;
    mo.req_ids = req_ids;
    mo.sampled = sampled;
    if (!lp.is_none()) {
      py::tuple l = lp.cast<py::tuple>();
      auto ids = l[0].cast<std::vector<std::vector<int64_t>>>();
      auto vals = l[1].cast<std::vector<std::vector<float>>>();
      auto ranks = l[2].cast<std::vector<int64_t>>();
      mo.has_logprobs = true;
      for (size_t j = 0; j < ids.size(); ++j) {
        mo.lp_width = (int64_t)ids[j].size();
        mo.lp_rows.push_back(core::LpRow{ids[j], vals[j], ranks[j]});
      }
      mo.lp_cu = l[3].cast<std::vector<int64_t>>();
    }
    if (!drafts.is_none()) {
      mo.has_drafts = true;
      mo.drafts = drafts.cast<std::vector<std::pair<Rid, std::vector<int32_t>>>>();
    }
    std::vector<std::pair<int64_t, std::vector<std::string>>> frames;
    core::UpdateOut u;
    {
      py::gil_scoped_release nogil;
      u = lc_->update_from_output(sos_.at(h), mo, mono_rs, [ts]() { return ts; });
      sos_.erase(h);
      for (auto& kv : u.outputs) {
        std::vector<std::string> fr;
        v1::encode_outputs(kv.second, &fr);
        frames.emplace_back(kv.first, std::move(fr));
      }
    }
    py::list msgs;
    for (auto& f : frames) {
      py::list fl;
      for (auto& b : f.second) fl.append(py::bytes(b));
      msgs.append(py::make_tuple(f.first, fl));
    }
    return py::make_tuple(msgs, u.accepted);
  }
  void update_draft_token_ids(const std::vector<std::pair<Rid, std::vector<int32_t>>>& d) {
    py::gil_scoped_release nogil;
    lc_->update_draft_token_ids(d);
  }
  py::dict update_draft_token_ids_in_output(int64_t h, const std::vector<std::pair<Rid, std::vector<int32_t>>>& d) {
    {
      py::gil_scoped_release nogil;
      lc_->update_draft_token_ids_in_output(d, &sos_.at(h));
    }
    return so_dict(sos_.at(h));
  }
  bool has_requests() const { return lc_->has_requests(); }
  int64_t num_unfinished() const { return lc_->num_unfinished(); }
  py::tuple counts() const { return py::make_tuple(lc_->running_ids().size(), lc_->num_unfinished()); }
  void set_pause(int p) { lc_->set_pause((core::Lifecycle::Pause)p); }
  bool reset_prefix_cache() { return lc_->reset_prefix_cache(); }
  void drop(int64_t h) { sos_.erase(h); }
  py::dict stats() const {
    const auto& c = lc_->counters();
    py::dict d;
    d["grammar_held"] = c.grammar_held; d["grammar_hold_off"] = c.grammar_hold_off;
    d["grammar_host_ids"] = c.grammar_host_ids; d["grammar_resume_no_ids"] = c.grammar_resume_no_ids;
    d["settled_host_t0"] = c.settled_host_t0; d["deferred_steps"] = c.deferred_steps; d["rejected"] = c.rejected;
    return d;
  }

 private:
  py::object owner_;                   // keeps the Python scheduler wrapper (and its C++ scheduler / KV) alive
  IScheduler* rs_ = nullptr;
  IKVManager* kv_ = nullptr;
  int64_t window_page_tokens_ = WINDOW_PAGE_TOKENS;
  std::unique_ptr<grammar::Manager> som_;
  std::unique_ptr<core::Lifecycle> lc_;
  std::map<int64_t, core::SchedOut> sos_;
  int64_t next_ = 0;
};

}  // namespace

PYBIND11_MODULE(fpr_core_py, m) {
  m.doc() = "fpr: the C++ engine-core lifecycle, structured outputs and V1 output assembly";
  py::class_<PyCore>(m, "Core")
      .def(py::init<py::dict, py::object, py::object, py::object, py::object, py::object, py::list>())
      .def("add_request", &PyCore::add_request)
      .def("abort", &PyCore::abort)
      .def("schedule", &PyCore::schedule)
      .def("grammar_bitmask", &PyCore::grammar_bitmask)
      .def("update", &PyCore::update)
      .def("update_draft_token_ids", &PyCore::update_draft_token_ids)
      .def("update_draft_token_ids_in_output", &PyCore::update_draft_token_ids_in_output)
      .def("spec_token_ids", &PyCore::spec_token_ids)
      .def("has_requests", &PyCore::has_requests)
      .def("num_unfinished", &PyCore::num_unfinished);
  py::class_<CoreX>(m, "CoreX")
      .def(py::init<py::dict, py::capsule, py::capsule, py::object, py::object, py::list>())
      .def("add_request", &CoreX::add_request, py::arg("payload"), py::arg("mono"), py::arg("wall"),
           py::arg("images") = py::none(), py::arg("mm_reject") = std::string(), py::arg("forced") = py::none())
      .def("abort", &CoreX::abort)
      .def("schedule", &CoreX::schedule)
      .def("grammar_bitmask", &CoreX::grammar_bitmask)
      .def("update", &CoreX::update)
      .def("update_draft_token_ids", &CoreX::update_draft_token_ids)
      .def("update_draft_token_ids_in_output", &CoreX::update_draft_token_ids_in_output)
      .def("has_requests", &CoreX::has_requests)
      .def("num_unfinished", &CoreX::num_unfinished)
      .def("counts", &CoreX::counts)
      .def("set_pause", &CoreX::set_pause)
      .def("reset_prefix_cache", &CoreX::reset_prefix_cache)
      .def("drop", &CoreX::drop)
      .def("stats", &CoreX::stats);
}
