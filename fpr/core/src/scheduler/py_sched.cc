// py_sched.cc
// and _ischeduler() -> PyCapsule "fpr.IScheduler"
//    (non-owning IScheduler*, valid while this object lives) so a C++ core calls the scheduler with no Python at all;
//  * the *_t calls the Python facade (cxx_sched.py: CxxRankScheduler, serve/scheduler.py's surface) uses: tuple
//    plans, request handles (live views of the C++ request state, by Python field name), queue / state reads, and
//    the out-of-band writes tests and the adapter make.
// KV: `kv` is an object exposing _ikv() -> PyCapsule "fpr.IKVManager", or None: a private C++ OwnedKVManager built from the geometry.
// Block hashes: hash_cb(rid, first, n) -> bytes of 16 * k (k <= n; big-endian hi, lo per block), the base hashes of
// blocks [first, first + k) (2a: vLLM's block hashes through the adapter's hash_fn); called lazily, with the GIL.
// The GIL is released during add / schedule / update (re-taken only inside hash_cb).
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "../kv/kv_manager.h"
#include "rank_scheduler.h"

namespace py = pybind11;
using namespace fpr;

namespace {

const char* phase_name(Phase p) {
  switch (p) {
    case Phase::WAITING: return "waiting";
    case Phase::PREFILL: return "prefill";
    case Phase::DECODE: return "decode";
    case Phase::FINISHED: return "finished";
  }
  return "waiting";
}
Phase phase_of(const std::string& s) {
  if (s == "prefill") return Phase::PREFILL;
  if (s == "decode") return Phase::DECODE;
  if (s == "finished") return Phase::FINISHED;
  return Phase::WAITING;
}

enum Fld {
  F_PHASE, F_RANK, F_NUM_COMPUTED, F_NUM_OUTPUT, F_PREFILL_END, F_CACHED_TOKENS, F_REPLAY_START, F_SKIP_UNTIL_STEP,
  F_INFLIGHT, F_FIRST_PENDING, F_MAX_NEW_TOKENS, F_PRIORITY, F_SCORE, F_FORCED, F_ARRIVAL, F_PROMPT_LEN, F_ADMIT_TIME,
  F_PF_PROGRESS_TIME, F_FIRST_CHUNK_TIME, F_FIRST_TOKEN_TIME, F_FINISH_TIME, F_LAST_PF_STEP, F_CP_FORCED, F_UID,
  F_HOME_RANK, F_N_PREEMPT, F_N_MIGRATE, F_ADMIT_STEPS_WAITED, F_HIT_TOKENS_TOTAL, F_COPIED_TOKENS_TOTAL, F_CP_PREV_SUBS,
  F_CP_SHADOW_RANKS
};
const std::unordered_map<std::string, int>& fields() {
  static const std::unordered_map<std::string, int> m = {
      {"phase", F_PHASE}, {"rank", F_RANK}, {"num_computed", F_NUM_COMPUTED}, {"num_output", F_NUM_OUTPUT},
      {"prefill_end", F_PREFILL_END}, {"cached_tokens", F_CACHED_TOKENS}, {"replay_start", F_REPLAY_START},
      {"skip_until_step", F_SKIP_UNTIL_STEP}, {"inflight", F_INFLIGHT}, {"first_pending", F_FIRST_PENDING},
      {"max_new_tokens", F_MAX_NEW_TOKENS}, {"priority", F_PRIORITY}, {"score", F_SCORE}, {"forced", F_FORCED},
      {"arrival", F_ARRIVAL}, {"prompt_len", F_PROMPT_LEN}, {"admit_time", F_ADMIT_TIME},
      {"pf_progress_time", F_PF_PROGRESS_TIME}, {"first_chunk_time", F_FIRST_CHUNK_TIME},
      {"first_token_time", F_FIRST_TOKEN_TIME}, {"finish_time", F_FINISH_TIME}, {"last_pf_step", F_LAST_PF_STEP},
      {"cp_forced", F_CP_FORCED}, {"uid", F_UID}, {"home_rank", F_HOME_RANK}, {"n_preempt", F_N_PREEMPT},
      {"n_migrate", F_N_MIGRATE}, {"admit_steps_waited", F_ADMIT_STEPS_WAITED}, {"hit_tokens_total", F_HIT_TOKENS_TOTAL},
      {"copied_tokens_total", F_COPIED_TOKENS_TOTAL}, {"cp_prev_subs", F_CP_PREV_SUBS},
      {"cp_shadow_ranks", F_CP_SHADOW_RANKS}};
  return m;
}
int field_id(const std::string& f) {
  auto it = fields().find(f);
  if (it == fields().end()) throw py::attribute_error("RankScheduler request has no C++ field " + f);
  return it->second;
}
py::object opt(const OptD& x) { return x ? py::object(py::float_(*x)) : py::object(py::none()); }
OptD to_opt(py::handle v) { return v.is_none() ? OptD() : OptD(v.cast<double>()); }

py::object get_field(const SReq& q, int f) {
  switch (f) {
    case F_PHASE: return py::str(phase_name(q.phase));
    case F_RANK: return q.rank < 0 ? py::object(py::none()) : py::object(py::int_(q.rank));
    case F_NUM_COMPUTED: return py::int_(q.num_computed);
    case F_NUM_OUTPUT: return py::int_(q.num_output);
    case F_PREFILL_END: return py::int_(q.prefill_end);
    case F_CACHED_TOKENS: return py::int_(q.cached_tokens);
    case F_REPLAY_START: return py::int_(q.replay_start);
    case F_SKIP_UNTIL_STEP: return py::int_(q.skip_until_step);
    case F_INFLIGHT: return py::int_(q.inflight);
    case F_FIRST_PENDING: return py::bool_(q.first_pending);
    case F_MAX_NEW_TOKENS: return py::int_(q.max_new_tokens);
    case F_PRIORITY: return py::int_(q.priority);
    case F_SCORE: return py::bool_(q.score);
    case F_FORCED: return py::bool_(q.forced);
    case F_ARRIVAL: return py::float_(q.arrival);
    case F_PROMPT_LEN: return py::int_(q.prompt_len);
    case F_ADMIT_TIME: return opt(q.admit_time);
    case F_PF_PROGRESS_TIME: return opt(q.pf_progress_time);
    case F_FIRST_CHUNK_TIME: return opt(q.first_chunk_time);
    case F_FIRST_TOKEN_TIME: return opt(q.first_token_time);
    case F_FINISH_TIME: return opt(q.finish_time);
    case F_LAST_PF_STEP: return py::int_(q.last_pf_step);
    case F_CP_FORCED: return py::bool_(q.cp_forced);
    case F_UID: return py::int_(q.uid);
    case F_HOME_RANK: return q.home_rank < 0 ? py::object(py::none()) : py::object(py::int_(q.home_rank));
    case F_N_PREEMPT: return py::int_(q.n_preempt);
    case F_N_MIGRATE: return py::int_(q.n_migrate);
    case F_ADMIT_STEPS_WAITED: return py::int_(q.admit_steps_waited);
    case F_HIT_TOKENS_TOTAL: return py::int_(q.hit_tokens_total);
    case F_COPIED_TOKENS_TOTAL: return py::int_(q.copied_tokens_total);
    case F_CP_PREV_SUBS: {
      py::list l;
      for (auto& s : q.cp_prev_subs) l.append(py::make_tuple(std::get<0>(s), std::get<1>(s), std::get<2>(s)));
      return std::move(l);
    }
    case F_CP_SHADOW_RANKS: {   // [(helper rank, shadow uid)] in insertion order (the facade builds Python's dict)
      py::list l;
      for (auto& e : q.cp_shadows) l.append(py::make_tuple(e.first, e.second.second));
      return std::move(l);
    }
  }
  return py::none();
}

void set_field(SReq& q, int f, py::handle v) {
  switch (f) {
    case F_PHASE: q.phase = phase_of(v.cast<std::string>()); break;
    case F_RANK: q.rank = v.is_none() ? -1 : v.cast<int32_t>(); break;
    case F_NUM_COMPUTED: q.num_computed = v.cast<int64_t>(); break;
    case F_NUM_OUTPUT: q.num_output = v.cast<int64_t>(); break;
    case F_PREFILL_END: q.prefill_end = v.cast<int64_t>(); break;
    case F_CACHED_TOKENS: q.cached_tokens = v.cast<int64_t>(); break;
    case F_REPLAY_START: q.replay_start = v.cast<int64_t>(); break;
    case F_SKIP_UNTIL_STEP: q.skip_until_step = v.cast<int64_t>(); break;
    case F_INFLIGHT: q.inflight = v.cast<int64_t>(); break;
    case F_FIRST_PENDING: q.first_pending = v.cast<bool>(); break;
    case F_MAX_NEW_TOKENS: q.max_new_tokens = v.cast<int64_t>(); break;
    case F_PRIORITY: q.priority = v.cast<int32_t>(); break;
    case F_SCORE: q.score = v.cast<bool>(); break;
    case F_FORCED: q.forced = v.cast<bool>(); break;
    case F_ARRIVAL: q.arrival = v.cast<double>(); break;
    case F_PROMPT_LEN: q.prompt_len = v.cast<int64_t>(); break;
    case F_ADMIT_TIME: q.admit_time = to_opt(v); break;
    case F_PF_PROGRESS_TIME: q.pf_progress_time = to_opt(v); break;
    case F_FIRST_CHUNK_TIME: q.first_chunk_time = to_opt(v); break;
    case F_FIRST_TOKEN_TIME: q.first_token_time = to_opt(v); break;
    case F_FINISH_TIME: q.finish_time = to_opt(v); break;
    case F_LAST_PF_STEP: q.last_pf_step = v.cast<int64_t>(); break;
    case F_CP_FORCED: q.cp_forced = v.cast<bool>(); break;
    case F_UID: q.uid = v.cast<int32_t>(); break;
    case F_HOME_RANK: q.home_rank = v.is_none() ? -1 : v.cast<int32_t>(); break;
    case F_N_PREEMPT: q.n_preempt = v.cast<int64_t>(); break;
    case F_N_MIGRATE: q.n_migrate = v.cast<int64_t>(); break;
    case F_ADMIT_STEPS_WAITED: q.admit_steps_waited = v.cast<int64_t>(); break;
    case F_HIT_TOKENS_TOTAL: q.hit_tokens_total = v.cast<int64_t>(); break;
    case F_COPIED_TOKENS_TOTAL: q.copied_tokens_total = v.cast<int64_t>(); break;
    default: throw py::attribute_error("read-only C++ request field");
  }
}

struct ReqHandle {
  ReqP q;
};

template <class T>
void load(T& obj, const py::dict& d) {
  T::visit(obj, [&](const char* name, auto& m) {
    if (!d.contains(name)) return;
    py::handle v = d[name];
    using M = std::decay_t<decltype(m)>;
    if constexpr (std::is_same_v<M, std::string>) m = v.cast<std::string>();
    else if constexpr (std::is_same_v<M, bool>) m = v.cast<bool>();
    else if constexpr (std::is_same_v<M, double>) m = v.cast<double>();
    else m = (M)v.cast<double>();     // ints (Python may hold an int field as float: PHYS_COST float())
  });
}
template <class T>
bool set_named(T& obj, const std::string& f, py::handle v) {
  bool ok = false;
  T::visit(obj, [&](const char* name, auto& m) {
    if (f != name) return;
    ok = true;
    using M = std::decay_t<decltype(m)>;
    if constexpr (std::is_same_v<M, std::string>) m = v.cast<std::string>();
    else if constexpr (std::is_same_v<M, bool>) m = v.cast<bool>();
    else if constexpr (std::is_same_v<M, double>) m = v.cast<double>();
    else m = (M)v.cast<double>();
  });
  return ok;
}

py::tuple work_t(const SeqWork& w) {
  py::object cp = py::none();
  if (w.has_cp) {
    py::list subs;
    for (auto& s : w.cp.subs) subs.append(py::make_tuple(std::get<0>(s), std::get<1>(s), std::get<2>(s)));
    cp = py::make_tuple(w.cp.group, w.cp.rid, w.cp.owner, subs, w.cp.me);
  }
  py::list ims;
  for (auto& im : w.images) ims.append(py::make_tuple(im.offset, im.length, im.key));
  return py::make_tuple(w.rid, w.rank, w.kind, w.start, w.n, w.cached_upto, w.replay_start, w.last_chunk, w.score,
                        w.forced, w.uid, w.known, cp, ims);
}
py::tuple load_t(const RankLoad& l) {
  return py::make_tuple(l.dec_rows, l.dec_idx_pos, l.dec_cand_pos, l.n_dec_seqs, l.max_dec_ctx, l.pf_rows, l.pf_tail_rows,
                        l.pf_idx_pos, l.pf_cand_pos);
}

class PySched {
 public:
  PySched(py::dict geom, py::list pools, py::dict cost, py::dict cfg, py::object kv, py::object hash_cb,
          int64_t hbm_pages, std::string poison_mode)
      : hash_cb_(std::move(hash_cb)) {
    Geometry g;
    load(g, geom);
    for (auto p : pools) {
      py::tuple t = p.cast<py::tuple>();
      g.pool_override.push_back(PoolSpec{t[0].cast<std::string>(), t[1].cast<std::string>(), t[2].cast<int64_t>(),
                                         t[3].cast<int64_t>(), t[4].cast<int64_t>()});
    }
    CostParams cp;
    load(cp, cost);
    SchedConfig sc;
    load(sc, cfg);
    if (kv.is_none()) {
      s_ = std::make_unique<RankScheduler>(g, cp, sc, make_kv_manager(kv_geometry(g, hbm_pages, poison_mode)));
    } else {
      s_ = std::make_unique<RankScheduler>(g, cp, sc, kv_ptr(kv));
      kv_obj_ = kv;
    }
    if (!hash_cb_.is_none()) install_hash_source();
  }
  ~PySched() {
    py::gil_scoped_acquire g;
    hash_cb_ = py::object();
    kv_obj_ = py::object();
  }

  // kv.cxx_capsule() -> PyCapsule "fpr::IKVManager::shared_ptr" holding a heap
  // std::shared_ptr<IKVManager>*; we keep a copy of the shared_ptr (the manager outlives either Python object)
  IKVManager* kv_ptr(py::object kv) {
    py::object cap = kv.attr("cxx_capsule")();
    void* p = PyCapsule_GetPointer(cap.ptr(), "fpr::IKVManager::shared_ptr");
    if (!p) throw py::error_already_set();
    kv_sp_ = *static_cast<std::shared_ptr<IKVManager>*>(p);
    return kv_sp_.get();
  }
  void install_hash_source() {
    s_->hash_source = [this](const SReq& q, int64_t first, int64_t n, std::vector<Hash>* out) -> int64_t {
      py::gil_scoped_acquire g;
      try {
        py::object r = hash_cb_(q.rid, first, n);
        if (r.is_none()) return 0;
        std::string b = r.cast<py::bytes>();
        int64_t k = std::min<int64_t>(n, (int64_t)(b.size() / 16));
        out->resize((size_t)k);
        const unsigned char* c = (const unsigned char*)b.data();
        for (int64_t i = 0; i < k; ++i) {
          uint64_t hi = 0, lo = 0;
          for (int j = 0; j < 8; ++j) hi = (hi << 8) | c[16 * i + j];
          for (int j = 8; j < 16; ++j) lo = (lo << 8) | c[16 * i + j];
          (*out)[(size_t)i] = Hash{hi, lo};
        }
        return k;
      } catch (py::error_already_set& e) {
        e.restore();
        PyErr_Print();                // a broken hash_fn: the blocks get private keys (never a wrong prefix reuse)
        return 0;
      }
    };
  }

  // ------------------------------------------------------------------ facade surface (*_t)
  py::tuple add_t(const std::string& rid, int64_t prompt_len, int64_t max_new, double arrival, int32_t priority, bool score,
                  bool forced, py::list images, bool final_keys) {
    std::vector<Image> ims;
    for (auto im : images) {
      py::tuple t = im.cast<py::tuple>();
      ims.push_back(Image{t[0].cast<int64_t>(), t[1].cast<int64_t>(), t[2].cast<std::string>()});
    }
    std::string why;
    int64_t mx = 0;
    Status st;
    {
      py::gil_scoped_release r;
      st = s_->add(rid, prompt_len, max_new, arrival, priority, score, forced, ims, {}, &why, &mx, final_keys);
    }
    py::object h = py::none();
    if (st == Status::OK) h = py::cast(ReqHandle{s_->reqs.at(rid)});
    return py::make_tuple(st == Status::OK, st == Status::OK ? py::object(py::none()) : py::object(py::str(why)), mx, h);
  }
  py::object abort_t(const std::string& rid) {
    s_->clear_error();
    s_->abort(rid);
    return err();
  }
  py::tuple schedule_t(double now) {
    s_->clear_error();
    StepPlan p;
    {
      py::gil_scoped_release r;
      p = s_->schedule(now);
    }
    py::list work, copies, pre, mig, loads;
    for (auto& w : p.work) work.append(work_t(w));
    for (auto& c : p.copies) copies.append(py::make_tuple(c.pool, c.src_rank, c.src_page, c.dst_rank, c.dst_page, c.step));
    for (auto& r : p.preempted) pre.append(r);
    for (auto& r : p.migrated) mig.append(r);
    for (auto& l : p.loads) loads.append(load_t(l));
    int64_t step = p.step;
    plans_[step] = std::move(p);
    while (plans_.size() > 8) plans_.erase(plans_.begin());
    const StepPlan& q = plans_[step];
    return py::make_tuple(step, work, copies, pre, mig, loads, q.est_us, q.idle_frac, err());
  }
  py::tuple update_t(int64_t step, const std::vector<std::pair<Rid, int32_t>>& acc, double now) {
    s_->clear_error();
    auto it = plans_.find(step);
    if (it == plans_.end()) throw py::key_error("update: no plan for step " + std::to_string(step) + " (only the last 8)");
    std::vector<Rid> fin;
    {
      py::gil_scoped_release r;
      fin = s_->update(it->second, acc, now);
    }
    plans_.erase(it);
    return py::make_tuple(fin, err());
  }
  py::list reqs_t() const {   // [(rid, handle)] in Python dict order
    py::list l;
    for (auto& r : s_->reqs_order) l.append(py::make_tuple(r, ReqHandle{s_->reqs.at(r)}));
    return l;
  }
  std::vector<Rid> waiting_t() const {
    std::vector<Rid> v;
    for (auto& q : s_->waiting) v.push_back(q->rid);
    return v;
  }
  std::vector<Rid> prefilling_t() const {
    std::vector<Rid> v;
    for (SReq* q : s_->prefilling()) v.push_back(q->rid);
    return v;
  }
  std::vector<Rid> on_rank_t(int r) const {
    std::vector<Rid> v;
    for (auto& q : s_->running_) if (q->rank == r) v.push_back(q->rid);
    return v;
  }
  py::object handle(const std::string& rid) const {
    auto it = s_->reqs.find(rid);
    return it == s_->reqs.end() ? py::object(py::none()) : py::cast(ReqHandle{it->second});
  }
  std::vector<std::pair<std::string, int64_t>> stats_t() const { return s_->stats; }
  py::object sched_get(const std::string& f) const {
    if (f == "step_id") return py::int_(s_->step_id_);
    if (f == "rr") return py::int_(s_->rr);
    if (f == "imbalance_steps") return py::int_(s_->imbalance_steps);
    if (f == "pf_credit") return py::float_(s_->pf_credit);
    if (f == "dec_since_mixed") return py::int_(s_->dec_since_mixed);
    if (f == "lam") return py::float_(s_->lam);
    throw py::attribute_error("RankScheduler has no C++ scalar " + f);
  }
  void sched_set(const std::string& f, py::handle v) {
    if (f == "step_id") s_->step_id_ = v.cast<int64_t>();
    else if (f == "rr") s_->rr = v.cast<int64_t>();
    else if (f == "imbalance_steps") s_->imbalance_steps = v.cast<int64_t>();
    else if (f == "pf_credit") s_->pf_credit = v.cast<double>();
    else if (f == "dec_since_mixed") s_->dec_since_mixed = v.cast<int64_t>();
    else if (f == "lam") s_->lam = v.cast<double>();
    else throw py::attribute_error("RankScheduler has no C++ scalar " + f);
  }
  std::vector<int32_t> free_uids_t() const { return {s_->free_uids.begin(), s_->free_uids.end()}; }
  // Python `s.waiting = <deque>` / `s.running = <list>` (the adapter hides the queue for PAUSED_NEW and restores it)
  void set_waiting_t(const std::vector<Rid>& rids) {
    std::deque<ReqP> w;
    for (auto& r : rids) w.push_back(s_->reqs.at(r));
    s_->waiting.swap(w);
  }
  void set_running_t(const std::vector<Rid>& rids) {
    std::vector<ReqP> v;
    for (auto& r : rids) v.push_back(s_->reqs.at(r));
    s_->set_running(v);
  }
  py::object kv_check() const {   // IKVManager::check(): None, or the first failing invariant
    if (s_->kv()->check() == Status::OK) return py::none();
    return py::str(s_->kv()->last_error());
  }
  std::vector<int32_t> free_shadow_uids_t() const { return {s_->free_shadow_uids.begin(), s_->free_shadow_uids.end()}; }
  void set_cfg(const std::string& f, py::handle v) {
    if (!set_named(s_->cfg, f, v)) throw py::attribute_error("SchedConfig has no field " + f);
  }
  void set_cost(const std::string& f, py::handle v) {
    if (!set_named(s_->cost.p, f, v)) throw py::attribute_error("CostParams has no field " + f);
  }
  void reset_kv_obj(py::object kv) {
    s_->reset_kv_unowned(kv_ptr(kv));
    kv_obj_ = kv;
  }
  py::capsule kv_capsule() const { return py::capsule((void*)s_->kv(), "fpr.IKVManager"); }
  py::capsule ischeduler() const { return py::capsule((void*)static_cast<IScheduler*>(s_.get()), "fpr.IScheduler"); }
  int64_t max_request_tokens() const { return s_->max_request_tokens(); }
  std::vector<double> rank_loads_t() const { return s_->rank_loads_pub(); }
  double slack_t(double base, double backlog, py::object frac) const {
    if (frac.is_none()) return s_->slack_pub(base, backlog, nullptr);
    double f = frac.cast<double>();
    return s_->slack_pub(base, backlog, &f);
  }
  double decode_load_t(const std::string& rid) const { return s_->decode_load_pub(*s_->reqs.at(rid)); }
  std::vector<Rid> prefill_sorted_t(const std::vector<Rid>& rids, int64_t step, double now) const {
    std::vector<SReq*> qs;
    for (auto& r : rids) qs.push_back(s_->reqs.at(r).get());
    std::vector<Rid> out;
    for (SReq* q : s_->prefill_sorted_pub(qs, step, now)) out.push_back(q->rid);
    return out;
  }

  // ------------------------------------------------------------------ IScheduler names
  py::tuple add_request(const std::string& rid, int64_t prompt_len, int64_t max_tokens, double arrival, int32_t priority,
                        bool prompt_logprobs) {
    py::tuple t = add_t(rid, prompt_len, max_tokens, arrival, priority, prompt_logprobs, false, py::list(), false);
    return py::make_tuple(t[0], t[1]);
  }
  void abort(const std::string& rid) { abort_t(rid); }
  py::dict schedule(double now) {
    s_->clear_error();
    StepPlan p;
    {
      py::gil_scoped_release r;
      p = s_->schedule(now);
    }
    py::dict d;
    d["step"] = p.step;
    py::list work;
    for (auto& w : p.work) {
      py::dict x;
      x["rid"] = w.rid; x["rank"] = w.rank; x["kind"] = w.kind; x["start"] = w.start; x["n"] = w.n;
      x["cached_upto"] = w.cached_upto; x["replay_start"] = w.replay_start; x["last_chunk"] = w.last_chunk;
      x["score"] = w.score; x["forced"] = w.forced; x["uid"] = w.uid; x["known"] = w.known;
      if (w.has_cp) {
        py::dict c;
        py::list subs;
        for (auto& s : w.cp.subs) subs.append(py::make_tuple(std::get<0>(s), std::get<1>(s), std::get<2>(s)));
        c["group"] = w.cp.group; c["rid"] = w.cp.rid; c["owner"] = w.cp.owner; c["subs"] = subs; c["me"] = w.cp.me;
        x["cp"] = c;
      } else {
        x["cp"] = py::none();
      }
      work.append(x);
    }
    d["work"] = work;
    py::list copies;
    for (auto& c : p.copies)
      copies.append(py::make_tuple(pool_name(c.pool), c.src_rank, c.src_page, c.dst_rank, c.dst_page, c.step));
    d["copies"] = copies;
    d["preempted"] = p.preempted;
    d["migrated"] = p.migrated;
    d["est_us"] = p.est_us;
    d["idle_frac"] = p.idle_frac;
    int64_t step = p.step;
    plans_[step] = std::move(p);
    while (plans_.size() > 8) plans_.erase(plans_.begin());
    return d;
  }
  std::vector<Rid> update(int64_t step, const std::vector<std::pair<Rid, int32_t>>& acc, double now) {
    py::tuple t = update_t(step, acc, now);
    return t[0].cast<std::vector<Rid>>();
  }
  py::object get(const std::string& rid) const {
    ReqState st;
    if (!s_->get(rid, &st)) return py::none();
    return py::make_tuple((int)st.phase, st.rank, st.uid, st.num_computed, st.num_output, st.inflight, st.replay_start,
                          st.cached_tokens, st.skip_until_step, st.first_pending);
  }
  std::vector<Rid> running() const { return s_->running(); }
  int64_t step_id() const { return s_->step_id(); }
  void hold_until(const std::string& rid, int64_t step) { s_->hold_until(rid, step); }
  void sync_num_output(const std::string& rid, int64_t n) { s_->sync_num_output(rid, n); }
  void set_paused_new(bool p) { s_->set_paused_new(p); }
  bool paused_new() const { return s_->paused_new_; }
  int64_t max_new_cap(int64_t n) const { return s_->max_new_cap(n); }
  int32_t seats_per_rank() const { return s_->seats_per_rank(); }
  std::string stats_json() const { return s_->stats_json(); }

 private:
  std::unique_ptr<RankScheduler> s_;
  py::object kv_obj_, hash_cb_;
  std::shared_ptr<IKVManager> kv_sp_;
  std::map<int64_t, StepPlan> plans_;
  py::object err() const { return s_->error().empty() ? py::object(py::none()) : py::object(py::str(s_->error())); }
};

}  // namespace

PYBIND11_MODULE(fpr_sched_py, m) {
  m.doc() = "the C++ RankScheduler (fpr/core/src/scheduler/py_sched.cc)";
  py::class_<ReqHandle>(m, "ReqHandle")
      .def("get", [](const ReqHandle& h, const std::string& f) { return get_field(*h.q, field_id(f)); })
      .def("set", [](ReqHandle& h, const std::string& f, py::handle v) { set_field(*h.q, field_id(f), v); })
      .def_property_readonly("rid", [](const ReqHandle& h) { return h.q->rid; })
      .def("same", [](const ReqHandle& a, const ReqHandle& b) { return a.q == b.q; });
  py::class_<PySched>(m, "Scheduler")
      .def(py::init<py::dict, py::list, py::dict, py::dict, py::object, py::object, int64_t, std::string>(),
           py::arg("geom"), py::arg("pools"), py::arg("cost"), py::arg("cfg"), py::arg("kv") = py::none(),
           py::arg("hash_cb") = py::none(), py::arg("hbm_pages") = 0, py::arg("poison_mode") = "")
      .def("add_t", &PySched::add_t)
      .def("abort_t", &PySched::abort_t)
      .def("schedule_t", &PySched::schedule_t)
      .def("update_t", &PySched::update_t)
      .def("reqs_t", &PySched::reqs_t)
      .def("waiting_t", &PySched::waiting_t)
      .def("prefilling_t", &PySched::prefilling_t)
      .def("on_rank_t", &PySched::on_rank_t)
      .def("handle", &PySched::handle)
      .def("stats_t", &PySched::stats_t)
      .def("sched_get", &PySched::sched_get)
      .def("sched_set", &PySched::sched_set)
      .def("free_uids_t", &PySched::free_uids_t)
      .def("set_waiting_t", &PySched::set_waiting_t)
      .def("set_running_t", &PySched::set_running_t)
      .def("kv_check", &PySched::kv_check)
      .def("free_shadow_uids_t", &PySched::free_shadow_uids_t)
      .def("set_cfg", &PySched::set_cfg)
      .def("set_cost", &PySched::set_cost)
      .def("reset_kv_obj", &PySched::reset_kv_obj)
      .def("kv_capsule", &PySched::kv_capsule)
      .def("_ischeduler", &PySched::ischeduler)
      .def("max_request_tokens", &PySched::max_request_tokens)
      .def("rank_loads_t", &PySched::rank_loads_t)
      .def("slack_t", &PySched::slack_t)
      .def("decode_load_t", &PySched::decode_load_t)
      .def("prefill_sorted_t", &PySched::prefill_sorted_t)
      .def("paused_new", &PySched::paused_new)
      .def("stats_json", &PySched::stats_json)
      .def("add_request", &PySched::add_request)
      .def("abort", &PySched::abort)
      .def("schedule", &PySched::schedule)
      .def("update", &PySched::update)
      .def("get", &PySched::get)
      .def("running", &PySched::running)
      .def("step_id", &PySched::step_id)
      .def("hold_until", &PySched::hold_until)
      .def("sync_num_output", &PySched::sync_num_output)
      .def("set_paused_new", &PySched::set_paused_new)
      .def("max_new_cap", &PySched::max_new_cap)
      .def("seats_per_rank", &PySched::seats_per_rank);
}
