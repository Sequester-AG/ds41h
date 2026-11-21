// py_core2b.cc -- Python runs only at init (vLLM's handshake, the device engine)
// then core2b_proc.py hands the sockets' addresses to Server and the rank scheduler / KV capsules to Core2b, and the
// C++ threads (T_in, the ICore loop, T_out) serve with the GIL released.
//   Server(): start(inputs, outputs, identity, ready_payload[, poll_ms]); attach(core); dead(); stop(); stats()
//   Core2b(cfg, sched_capsule, kv_capsule, owner, tokenizer_info | None, reasoners, worker, engine_index)
//     start() / stop() (the loop thread), step_once() + input() + pop_outputs() (tests without sockets)
// worker (2b-α0, the interim / CPU vehicle): a Python object with dispatch(step: dict) -> handle and wait(handle) ->
// [per-rank result dicts]; called under the GIL per step. NativeDecode replaces it for decode steps.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>

#include "core.h"
namespace fpr { namespace core { int64_t lifecycle_ghosts(); int64_t lifecycle_pending_dropped(); } }   // lifecycle.cc (ghost fix)
#include "fake_nd.h"
#include "native_worker.h"
#include "py_cfg.h"
#include "server.h"
#include "tables.h"

namespace py = pybind11;

namespace {
// core2b_proc._mixed_snap(eng) (a dict) -> fpr::p0::MixedSnap
fpr::p0::MixedSnap mixed_snap_of(const py::dict& d) {
  fpr::p0::MixedSnap s;
  auto i64 = [&](const char* k) { return d[k].cast<int64_t>(); };
  auto b = [&](const char* k) { return d[k].cast<bool>(); };
  auto v32 = [&](const char* k) { return d[k].cast<std::vector<int32_t>>(); };
  auto v64 = [&](const char* k) { return d[k].cast<std::vector<int64_t>>(); };
  s.W = (int)i64("W"); s.mixed_round = i64("mixed_round"); s.max_mixed_rows = i64("max_mixed_rows");
  s.max_tail_rows = i64("max_tail_rows"); s.mixed_t_round = i64("mixed_t_round"); s.layers = v32("layers");
  s.mixed_graphs = b("mixed_graphs"); s.cp = b("cp"); s.dbg_sync = b("dbg_sync"); s.l20_tail = b("l20_tail");
  s.mixed_prestep = b("mixed_prestep"); s.mixed_graphs_t = b("mixed_graphs_t"); s.pf_act8 = b("pf_act8");
  s.attn_pf = b("attn_pf"); s.mm_on = b("mm_on"); s.poison = b("poison"); s.cxx_pieces = b("cxx_pieces");
  s.mixed_cxx_head = b("mixed_cxx_head"); s.tail_bound = b("tail_bound"); s.devtok_v2 = b("devtok_v2");
  s.per_layer_plans = b("per_layer_plans"); s.clear_tail_padding = b("clear_tail_padding");
  s.mixed_plans = d["mixed_plans"].cast<std::string>(); s.upload_v2 = b("upload_v2");
  s.index_layers = v32("index_layers"); s.ratio = v32("ratio"); s.index_source = v32("index_source");
  s.sliding_window = i64("sliding_window"); s.window_page_tokens = i64("window_page_tokens");
  s.scratch = v64("scratch"); s.n_paged = v64("n_paged"); s.pf_ws = b("pf_ws"); s.hsd = b("hsd"); s.idx = b("idx");
  s.k1 = i64("k1");
  if (d.contains("image_route") && !d["image_route"].is_none()) {   // [lo, hi] (optional: else the default)
    const auto ir = d["image_route"].cast<std::vector<int64_t>>();
    if (ir.size() == 2) s.img_route_lo = ir[0], s.img_route_hi = ir[1];
  }
  return s;
}
}  // namespace
using namespace fpr;

namespace {

// the 2b-α0 worker's in-flight step (the Python handle). StepHandle stays opaque: each worker casts its own type.
struct PyHandle {
  py::object h;
};

// Core2b holds the owner)
class BorrowedScheduler final : public IScheduler {
 public:
  explicit BorrowedScheduler(IScheduler* p) : p_(p) {}
  Status add_request(const Request& r, std::string* why) override { return p_->add_request(r, why); }
  void abort(const Rid& rid) override { p_->abort(rid); }
  StepPlan schedule(double now) override { return p_->schedule(now); }
  std::vector<Rid> update(const StepPlan& plan, const std::vector<std::pair<Rid, int32_t>>& acc, double now) override {
    return p_->update(plan, acc, now);
  }
  int64_t max_new_cap(int64_t prompt_len) const override { return p_->max_new_cap(prompt_len); }
  int32_t seats_per_rank() const override { return p_->seats_per_rank(); }
  void extend_block_hashes(const Rid& rid, int64_t first, const std::vector<Hash>& h) override {
    p_->extend_block_hashes(rid, first, h);
  }
  void hold_until(const Rid& rid, int64_t step) override { p_->hold_until(rid, step); }
  void sync_num_output(const Rid& rid, int64_t n) override { p_->sync_num_output(rid, n); }
  void set_paused_new(bool paused) override { p_->set_paused_new(paused); }
  void reset_kv(std::unique_ptr<IKVManager> kv) override { p_->reset_kv(std::move(kv)); }
  bool get(const Rid& rid, ReqState* out) const override { return p_->get(rid, out); }
  std::vector<Rid> running() const override { return p_->running(); }
  int64_t step_id() const override { return p_->step_id(); }
  std::string stats_json() const override { return p_->stats_json(); }

 private:
  IScheduler* p_;
};

mp::Val to_mp(py::handle o) {   // a JSON-like Python value (None / bool / int / float / str / bytes / list / tuple / dict)
  if (o.is_none()) return mp::Val::nil_();
  if (py::isinstance<py::bool_>(o)) return mp::Val::bool_(o.cast<bool>());
  if (py::isinstance<py::int_>(o)) return mp::Val::int_(o.cast<int64_t>());
  if (py::isinstance<py::float_>(o)) return mp::Val::f64_(o.cast<double>());
  if (py::isinstance<py::str>(o)) return mp::Val::str_(o.cast<std::string>());
  if (py::isinstance<py::bytes>(o)) {
    mp::Val v;
    v.kind = mp::Val::BIN;
    v.s = o.cast<std::string>();
    return v;
  }
  if (py::isinstance<py::dict>(o)) {
    mp::Val v = mp::Val::map_();
    for (auto kv : o.cast<py::dict>()) {
      v.a.push_back(to_mp(kv.first));
      v.a.push_back(to_mp(kv.second));
    }
    return v;
  }
  if (py::isinstance<py::list>(o) || py::isinstance<py::tuple>(o)) {
    mp::Val v = mp::Val::arr_();
    for (auto x : o) v.a.push_back(to_mp(x));
    return v;
  }
  throw std::runtime_error("fpr_core2b: static utility value of type " + std::string(py::str(o.get_type())));
}

py::tuple plan_tuple(const StepPlan& plan) {   // the tuple core_sched._plan turns into serve's StepPlan
  py::list work;
  for (const auto& w : plan.work) {
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
  for (const auto& c : plan.copies) copies.append(py::make_tuple(c.pool, c.src_rank, c.src_page, c.dst_rank, c.dst_page, c.step));
  return py::make_tuple(plan.step, work, copies, plan.preempted, plan.migrated, plan.est_us, plan.idle_frac);
}

// ---- the embedded engine step's Python inputs.
// batch_tuple = fpr_kvtables.build_step_batch's tuple, which core2b_proc._Embedded assembles into
// serve's StepBatch exactly as cxx_kv.wrap_build_step_batch does.
struct PagesView { std::shared_ptr<const std::vector<int64_t>> v; };
struct WindowView { std::shared_ptr<const WindowTable> w; };

int64_t norm_index(int64_t i, int64_t n) {
  int64_t j = i < 0 ? i + n : i;
  if (j < 0 || j >= n) throw py::index_error("list index out of range");
  return j;
}

py::tuple batch_tuple(const StepPlan& plan, const StepBatch& sb) {
  std::vector<size_t> mine;           // the rank's work items in plan order (build_step_batch's `items`)
  for (size_t i = 0; i < plan.work.size(); ++i)
    if (plan.work[i].rank == sb.rank) mine.push_back(i);
  std::vector<size_t> order(mine.size());   // entries: Python's stable sort, decode first
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return (plan.work[mine[a]].kind != "decode") < (plan.work[mine[b]].kind != "decode");
  });
  if (order.size() != sb.seqs.size())
    throw std::runtime_error("fpr_core2b: rank " + std::to_string(sb.rank) + " batch has " +
                             std::to_string(sb.seqs.size()) + " entries for " + std::to_string(order.size()) + " work items");
  static const auto empty_pages = std::make_shared<const std::vector<int64_t>>();
  static const auto empty_window = std::make_shared<const WindowTable>();
  py::list seqs;
  for (size_t k = 0; k < sb.seqs.size(); ++k) {
    const auto& se = sb.seqs[k];
    auto bt = se.block_table ? se.block_table : empty_pages;
    auto wt = se.window_table ? se.window_table : empty_window;
    seqs.append(py::make_tuple(order[k], se.row0, se.n_blocks, se.tail_row0, se.wver, se.inc, se.state_slot,
                               PagesView{bt}, WindowView{wt}, (uintptr_t)bt.get(), (uintptr_t)wt.get()));
  }
  return py::make_tuple(sb.n_rows, sb.positions, sb.seq_slot, sb.paged_slot, sb.window_slot, sb.tail_rows,
                        sb.sample_rows, seqs);
}

template <class T>
py::array_t<T> arr(const std::vector<T>& v) {
  py::array_t<T> a((py::ssize_t)v.size());
  if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
  return a;
}
py::dict ct_dict(const CompactTables& ct) {   // fpr_kvtables TableBuilder.build_compact's dict (cxx_kv sets ct.rids)
  py::dict d;
#define F(x) d[#x] = arr(ct.x)
  F(slot); F(row0); F(n_rows); F(kind); F(start); F(replay_start); F(cached_upto); F(win_slot); F(win_blk); F(win_page);
  F(bt_upd_slot); F(bt_upd_start); F(bt_upd_len); F(bt_upd_pages); F(tokens); F(tail_rows); F(sample_idx); F(targets);
  F(uid); F(last_chunk); F(score); F(forced);
#undef F
  return d;
}

// nd's resolved batch (D13._fb_rows) as the Python body applies it to the assembled original batch
// every row's position / paged slot / window slot, and per entry (start, num_computed, n_blocks) (_fb_rows replaces
// those three for the device-resolved decode entries, start < 0 in the original)
py::tuple rb_tuple(const StepBatch& sb) {
  py::list seqs;
  for (const auto& se : sb.seqs) seqs.append(py::make_tuple(se.start, se.num_computed, se.n_blocks));
  return py::make_tuple(sb.positions, sb.paged_slot, sb.window_slot, seqs);
}
template <class T>
py::array_t<T> arr2(const std::vector<T>& v, py::ssize_t r, py::ssize_t c) {
  py::array_t<T> a({r, c});
  if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
  return a;
}
// nd's full tables of rb (its private TableBuilder, = D13.tb.build) as fpr_kvtables TableBuilder.build's dict
// (kvtables_py.cpp, key for key, same dtypes / shapes); core2b_proc wraps it into worker.RankTables as cxx_kv does
py::dict rank_tables_dict(const RankTables& t) {
  py::dict d;
  const py::ssize_t n = t.n;
  d["n"] = t.n;
  d["positions"] = arr(t.positions); d["tokens"] = arr(t.tokens); d["own_slot"] = arr(t.own_slot);
  d["si"] = arr2(t.si, n, (py::ssize_t)SLIDING_WINDOW); d["sl"] = arr(t.sl);
  py::dict ti, tl;
  for (size_t s = 0; s < t.ti.size(); ++s) {
    ti[py::int_(t.ti_sources[s])] = arr2(t.ti[s], n, 512);
    tl[py::int_(t.ti_sources[s])] = arr(t.tl[s]);
  }
  d["ti"] = ti; d["tl"] = tl;
  d["seq_slot"] = arr(t.seq_slot); d["paged_slot"] = arr(t.paged_slot); d["tail_rows"] = arr(t.tail_rows);
  d["sample_idx"] = arr(t.sample_idx); d["targets"] = arr(t.targets);
  const auto& q = t.seq;
  py::dict sq;
  sq["state_slot"] = arr(q.state_slot); sq["uid"] = arr(q.uid); sq["start"] = arr(q.start); sq["n_rows"] = arr(q.n_rows);
  sq["kind"] = arr(q.kind); sq["last_chunk"] = arr(q.last_chunk); sq["score"] = arr(q.score); sq["forced"] = arr(q.forced);
  sq["replay_start"] = arr(q.replay_start); sq["cached_upto"] = arr(q.cached_upto); sq["n_blocks"] = arr(q.n_blocks);
  sq["bt_reset"] = arr(q.bt_reset); sq["bt_upd_seq"] = arr(q.bt_upd_seq); sq["bt_upd_start"] = arr(q.bt_upd_start);
  sq["bt_upd_len"] = arr(q.bt_upd_len); sq["bt_upd_pages"] = arr(q.bt_upd_pages);
  if (q.has_block_table) sq["block_table"] = arr2(q.block_table, (py::ssize_t)q.bt_rows, (py::ssize_t)q.bt_cols);
  d["seq"] = sq;
  return d;
}

// StepResult dicts (core2b_proc's conversion of serve's StepResult) -> C++ (GIL held; a malformed dict throws)
void results_of(py::list res, std::vector<StepResult>* out) {
  out->clear();
  for (auto item : res) {
    py::dict r = item.cast<py::dict>();
    StepResult s;
    s.rank = r["rank"].cast<int32_t>();
    s.step = r["step"].cast<int64_t>();
    s.accepted = r["accepted"].cast<std::vector<std::pair<Rid, int32_t>>>();
    s.tokens = r["tokens"].cast<std::vector<std::pair<Rid, std::vector<int32_t>>>>();
    if (r.contains("logprobs"))
      s.logprobs = r["logprobs"].cast<std::vector<std::pair<Rid, std::vector<std::vector<std::pair<int32_t, float>>>>>>();
    if (r.contains("drafts")) s.drafts = r["drafts"].cast<std::vector<std::pair<Rid, std::vector<int32_t>>>>();
    out->push_back(std::move(s));
  }
}

// an IEngine2b that refuses every step, so the NativeWorker
// runs ALL of them through the nd EmbeddedFn (C++ tuples / views / payloads -> core2b_proc._Embedded -> serve Worker +
// backend.step_compact with the NativeWorker's CompactTables); on serve's MockEngine it compares with the Python engine
class EmbedAllNd final : public nd::IEngine2b {
 public:
  Status init(const FprEngineHandover*, nd::RunPython, std::string*) override { return Status::OK; }
  std::string async_ok(const std::vector<StepBatch>&, const std::vector<CompactTables>&, const nd::StepHost&) override {
    return "embed_all";
  }
  Status dispatch(const std::vector<StepBatch>&, const std::vector<CompactTables>&, const nd::StepHost&, nd::Handle**,
                  std::string* err) override {
    if (err) *err = "embed_all: no native steps";
    return Status::INTERNAL;
  }
  Status wait(nd::Handle*, std::vector<StepResult>*, std::string* err) override {
    if (err) *err = "embed_all: no native steps";
    return Status::INTERNAL;
  }
  Status drain(std::string*) override { return Status::OK; }
  Status embedded_begin(const nd::StepHost&, std::string*) override {
    ++n_;
    return Status::OK;
  }
  Status embedded_end(std::string*) override { return Status::OK; }
  std::string stats_json() const override { return "{\"embedded\": " + std::to_string(n_) + "}"; }

 private:
  int64_t n_ = 0;
};

// ADD payloads by request id until the request leaves the worker: the 2b-α0 worker decodes the request's vLLM
// SamplingParams from them (once per request, at its first scheduling)
class Payloads {
 public:
  void put(const Rid& rid, std::vector<std::string> frames) {   // [payload, aux...]
    std::lock_guard<std::mutex> g(mu_);
    m_[rid] = std::move(frames);
  }
  std::vector<std::string> get(const Rid& rid) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = m_.find(rid);
    return it == m_.end() ? std::vector<std::string>() : it->second;
  }
  void drop(const Rid& rid) {
    std::lock_guard<std::mutex> g(mu_);
    m_.erase(rid);
  }

 private:
  std::mutex mu_;
  std::map<Rid, std::vector<std::string>> m_;
};

// The C++ threads' Python work, run on the thread that serves it: core2b_proc's run_busy_loop = the engine process's
// main thread, the one that built the device engine. Its CUDA current devices / streams are thread-local.
// Inline mode (tests driving step_once from Python): run on the calling thread with the GIL.
class PyExec {
 public:
  // fn runs with the GIL held, on the serving thread (or inline); false = stopped before it ran
  bool call(const std::function<void()>& fn) {
    if (inline_) {
      py::gil_scoped_acquire g;
      fn();
      ++runs_;
      return true;
    }
    auto j = std::make_shared<Job>();
    j->fn = &fn;
    std::unique_lock<std::mutex> l(mu_);
    if (stop_) return false;
    q_.push_back(j);
    cv_.notify_all();
    jcv_.wait(l, [&] { return j->done || (stop_ && !j->running); });
    if (!j->done) {                     // stopped while still queued: never runs (fn's frame is about to go)
      for (auto it = q_.begin(); it != q_.end(); ++it)
        if (*it == j) {
          q_.erase(it);
          break;
        }
      return false;
    }
    return true;
  }
  // fire-and-forget (FIFO with call()'s jobs): the image frees (design §5a); false = stopped
  bool post(std::function<void()> fn) {
    if (inline_) {
      py::gil_scoped_acquire g;
      fn();
      ++runs_;
      return true;
    }
    auto j = std::make_shared<Job>();
    j->own = std::move(fn);
    j->fn = &j->own;
    {
      std::lock_guard<std::mutex> g(mu_);
      if (stop_) return false;
      q_.push_back(j);
    }
    cv_.notify_all();
    return true;
  }
  // the serving thread (GIL held by the caller): run at most one job within timeout_s; 1 = ran one, 0 = none, -1 = stopped
  int serve(double timeout_s) {
    std::shared_ptr<Job> j;
    {
      py::gil_scoped_release nogil;
      std::unique_lock<std::mutex> l(mu_);
      cv_.wait_for(l, std::chrono::duration<double>(timeout_s), [this] { return !q_.empty() || stop_; });
      if (stop_) return -1;
      if (q_.empty()) return 0;
      j = q_.front();
      q_.pop_front();
      j->running = true;
    }
    try {
      (*j->fn)();
    } catch (...) {                     // fn catches Python errors itself; anything else must not strand the caller
      std::fprintf(stderr, "core2b PyExec: a job threw\n");
    }
    ++runs_;
    {
      std::lock_guard<std::mutex> g(mu_);
      j->done = true;
    }
    jcv_.notify_all();
    return 1;
  }
  void set_inline(bool v) { inline_ = v; }
  // jobs run so far (every C++ -> Python crossing: nd's run_python, NativeWorker's callbacks, image frees): NativeWorker's
  // "Python epoch" (L3: no Python since X -> Python-visible state unchanged since X)
  uint64_t runs() const { return runs_.load(); }
  void stop() {
    {
      std::lock_guard<std::mutex> g(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    jcv_.notify_all();
  }

 private:
  struct Job {
    const std::function<void()>* fn = nullptr;
    std::function<void()> own;        // post(): the job owns its function
    bool running = false, done = false;
  };
  std::atomic<bool> inline_{true};
  std::atomic<uint64_t> runs_{0};
  std::mutex mu_;
  std::condition_variable cv_, jcv_;
  std::deque<std::shared_ptr<Job>> q_;
  bool stop_ = false;
};

class PyWorker final : public IWorker {
 public:
  // check_k1 > 0: every dispatch also hands Python the
  // C++ StepBatches as 2b's tuples + views and a shadow C++ TableBuilder's CompactTables (NativeWorker's builder) to
  // compare with its own
  PyWorker(py::object w, std::shared_ptr<Payloads> pl, std::shared_ptr<PyExec> ex, int32_t check_k1 = 0)
      : w_(std::move(w)), pl_(std::move(pl)), ex_(std::move(ex)), check_k1_(check_k1) {
    shadow_.tokens_of = [](const Rid&, int64_t) -> int64_t { return -1; };
  }
  ~PyWorker() override {
    py::gil_scoped_acquire g;
    w_ = py::object();
  }
  Status dispatch(const StepInput& in, StepHandle** h) override {
    Status st = Status::INTERNAL;
    if (!ex_->call([&] { st = dispatch_py(in, h); })) return Status::INTERNAL;
    return st;
  }
  Status wait(StepHandle* h, std::vector<StepResult>* out) override {
    Status st = Status::INTERNAL;
    if (!ex_->call([&] { st = wait_py(h, out); })) return Status::INTERNAL;
    return st;
  }

 private:
  Status dispatch_py(const StepInput& in, StepHandle** h) {   // GIL held
    try {
      py::dict d;
      d["step"] = in.step;
      d["plan"] = in.plan ? py::object(plan_tuple(*in.plan)) : py::object(py::none());
      d["rows"] = in.row_tokens;
      d["prev"] = in.prev_tokens;
      py::list nr;
      for (const auto& r : in.new_reqs) {
        py::list fr;                   // [payload, aux...] (vLLM's MsgpackDecoder takes the buffer list)
        for (const auto& f : pl_->get(r.rid)) fr.append(py::bytes(f));
        nr.append(py::make_tuple(r.rid, r.rank, r.prompt, r.output, r.structured, fr));
      }
      d["new_reqs"] = nr;
      d["finished"] = in.finished;
      d["preempted"] = in.preempted;
      d["order"] = in.order;
      d["num_spec"] = in.num_spec;
      py::list gr;
      for (const auto& x : in.grammar)
        gr.append(py::make_tuple(x.rid, py::bytes(reinterpret_cast<const char*>(x.rows.data()), x.rows.size() * 4),
                                 x.n_rows, x.words, x.num_acceptable));
      d["grammar"] = gr;
      d["est_us"] = in.est_us;
      d["total"] = in.total_rows;
      if (check_k1_ > 0 && in.plan && in.total_rows > 0) {
        py::list bt, cd;
        std::vector<int64_t> rt;
        for (size_t r = 0; r < in.batches.size(); ++r) {
          bt.append(batch_tuple(*in.plan, in.batches[r]));
          rt.assign(in.row_tokens[r].begin(), in.row_tokens[r].end());
          CompactTables ct;
          std::string err;
          if (shadow_.build_compact(in.batches[r], check_k1_, &rt, &ct, &err) != Status::OK)
            throw std::runtime_error("shadow build_compact: " + err);
          cd.append(ct_dict(ct));
        }
        d["cxx_batches"] = bt;
        d["cxx_cts"] = cd;
      }
      auto* sh = new PyHandle;
      sh->h = w_.attr("dispatch")(d);
      *h = reinterpret_cast<StepHandle*>(sh);
      for (const auto& rid : in.finished) pl_->drop(rid);
      return Status::OK;
    } catch (std::exception& e) {      // py::error_already_set + the check's runtime_error
      last_ = e.what();
      std::fprintf(stderr, "core2b worker dispatch: %s\n", last_.c_str());
      return Status::INTERNAL;
    }
  }
  Status wait_py(StepHandle* h, std::vector<StepResult>* out) {   // GIL held
    std::unique_ptr<PyHandle> own(reinterpret_cast<PyHandle*>(h));
    try {
      results_of(w_.attr("wait")(own->h), out);
      return Status::OK;
    } catch (py::error_already_set& e) {
      last_ = e.what();
      std::fprintf(stderr, "core2b worker wait: %s\n", last_.c_str());
      return Status::INTERNAL;
    }
  }

 private:
  py::object w_;
  std::shared_ptr<Payloads> pl_;
  std::shared_ptr<PyExec> ex_;
  int32_t check_k1_ = 0;
  TableBuilder shadow_;
  std::string last_;
};

// Images (design §5a): the process's Python image object (core2b_proc._MM) behind C++ calls. admit runs on T_in and
// blocks it (vLLM decodes + preprocesses an ADD on its input thread before the next message); frees are posted.
struct MmBridge {
  std::shared_ptr<PyExec> ex;
  py::object* mm = nullptr;           // Core2b::mm_ (alive as long as the core; PyExec stops first)
  std::mutex mu;
  std::set<Rid> rids;                 // admitted with images (Python's MMPlanner holds them) and not yet forgotten
  std::atomic<int64_t> admitted{0}, rejected{0}, posts{0};
  // frames -> images + reject reason ("" = admitted); false = the admitter failed (err)
  bool admit(const std::vector<std::string>& frames, std::vector<Image>* ims, std::string* reject, std::string* err) {
    bool ok = false;
    bool ran = ex->call([&] {
      try {
        py::list fr;
        for (size_t i = 1; i < frames.size(); ++i) fr.append(py::bytes(frames[i]));   // [payload, aux...]
        py::tuple t = mm->attr("admit")(fr);
        *reject = t[1].cast<std::string>();
        if (!t[0].is_none())
          for (auto it : t[0]) {
            py::tuple x = it.cast<py::tuple>();
            ims->push_back(Image{x[0].cast<int64_t>(), x[1].cast<int64_t>(), x[2].cast<std::string>()});
          }
        ok = true;
      } catch (py::error_already_set& e) {
        *err = e.what();
      }
    });
    if (!ran) *err = "the engine's Python thread stopped";
    return ran && ok;
  }
  void frees(const std::vector<Rid>& finished, const std::vector<Rid>& preempted) {
    std::vector<Rid> f, p;
    {
      std::lock_guard<std::mutex> g(mu);
      if (rids.empty()) return;
      for (const auto& r : preempted)
        if (rids.count(r)) p.push_back(r);
      for (const auto& r : finished)
        if (rids.erase(r)) f.push_back(r);
    }
    if (f.empty() && p.empty()) return;
    posts++;
    py::object* m = mm;
    ex->post([m, f, p] {
      try {
        m->attr("free")(f, p);
      } catch (py::error_already_set& e) {
        std::fprintf(stderr, "core2b mm free: %s\n", e.what());
      }
    });
  }
  // the step's image entries for MMPlanner.plan: [(rank, [(rid, kind, start, n_rows, row0, cp_owner | -1)])] (only
  // entries with images: the planner skips every other one)
  static py::list view(const std::vector<StepBatch>& b) {   // GIL held
    py::list out;
    for (const auto& sb : b) {
      py::list seqs;
      for (const auto& se : sb.seqs)
        if (!se.images.empty())
          seqs.append(py::make_tuple(se.rid, se.kind, se.start, se.n_rows, se.row0, se.has_cp ? se.cp.owner : -1));
      out.append(py::make_tuple(sb.rank, seqs));
    }
    return out;
  }
};

// ICore with the ADD payloads tapped for the 2b-α0 worker and images admitted through MmBridge (everything else forwarded)
class TapCore final : public ICore {
 public:
  TapCore(std::unique_ptr<ICore> c, std::shared_ptr<Payloads> pl, std::shared_ptr<MmBridge> mm)
      : c_(std::move(c)), pl_(std::move(pl)), mm_(std::move(mm)) {}
  Status add(const v1::EngineCoreRequest& r, std::string* err) override { return c_->add(r, err); }
  Status add_mm(const v1::EngineCoreRequest& r, const std::vector<Image>& ims, const std::string& rej,
                std::string* err) override {
    return c_->add_mm(r, ims, rej, err);
  }
  Status input_frames(const std::vector<std::string>& frames, std::string* err) override {
    uint8_t type = frames.empty() || frames[0].empty() ? 0xff : (uint8_t)frames[0][0];
    if (type != v1::ADD || frames.size() < 2) return ICore::input_frames(frames, err);
    v1::EngineCoreRequest r;
    if (!v1::decode_request(frames[1], &r, err)) return Status::BAD_ARG;
    if (pl_) pl_->put(r.request_id, std::vector<std::string>(frames.begin() + 1, frames.end()));
    if (!r.has_mm_features || !mm_) return c_->add(r, err);   // no admitter: the lifecycle rejects mm as text-only
    std::vector<Image> ims;
    std::string rej, e;
    if (!mm_->admit(frames, &ims, &rej, &e)) {
      std::fprintf(stderr, "core2b mm admit %s: %s\n", r.request_id.c_str(), e.c_str());
      rej = "image preprocessing failed in the engine: " + e.substr(0, 300);
    }
    if (rej.empty()) {
      std::lock_guard<std::mutex> g(mm_->mu);
      mm_->rids.insert(r.request_id);
      mm_->admitted++;
    } else {
      mm_->rejected++;
    }
    return c_->add_mm(r, ims, rej, err);
  }
  void abort(const std::vector<Rid>& rids) override { c_->abort(rids); }
  Status utility(const v1::UtilityCall& u, std::string* err) override { return c_->utility(u, err); }
  Status input(uint8_t type, const std::string& payload, std::string* err) override {
    if (type == v1::ADD) {
      v1::EngineCoreRequest r;
      if (!v1::decode_request(payload, &r, err)) return Status::BAD_ARG;
      if (pl_) pl_->put(r.request_id, {payload});
      return c_->add(r, err);
    }
    return c_->input(type, payload, err);
  }
  void start() override { c_->start(); }
  bool step_once() override { return c_->step_once(); }
  void stop() override { c_->stop(); }
  bool pop_outputs(std::vector<WireOut>* b, double t) override { return c_->pop_outputs(b, t); }
  std::string fatal() const override { return c_->fatal(); }
  bool has_work() const override { return c_->has_work(); }

 private:
  std::unique_ptr<ICore> c_;
  std::shared_ptr<Payloads> pl_;
  std::shared_ptr<MmBridge> mm_;
};

class Core2b {
 public:
  Core2b(py::dict cfg, py::capsule sched, py::capsule kvc, py::object owner, py::object tokenizer_info,
         py::list reasoners, py::object worker, int64_t engine_index)
      : owner_(owner) {
    if (std::string(sched.name()) != "fpr.IScheduler" || std::string(kvc.name()) != "fpr.IKVManager")
      throw std::runtime_error("fpr_core2b.Core2b: capsules must be fpr.IScheduler / fpr.IKVManager");
    auto* rs = static_cast<IScheduler*>(sched.get_pointer());
    auto* kv = static_cast<IKVManager*>(kvc.get_pointer());
    CoreConfig c;
    c.lifecycle = pycfg::lifecycle(cfg, reasoners);
    c.engine_index = engine_index;
    if (cfg.contains("step_log")) c.step_log = cfg["step_log"].cast<std::string>();   // PHYS_CXX_STEP_LOG
    if (cfg.contains("static_utilities"))
      for (auto kv : cfg["static_utilities"].cast<py::dict>()) c.static_utilities[kv.first.cast<std::string>()] = to_mp(kv.second);
    c.structured_outputs = !tokenizer_info.is_none();
    if (c.structured_outputs) c.grammar = pycfg::backend(cfg, tokenizer_info.cast<std::string>(), c.lifecycle);
    int64_t wpt = cfg.contains("window_page_tokens") ? cfg["window_page_tokens"].cast<int64_t>() : WINDOW_PAGE_TOKENS;
    // 2b-α0: the Python worker builds its own StepBatch objects at SCHEDULE time too (worker.prebuild(plan), once per
    // plan, before rank 0's): PhysScheduler builds them in _schedule; a dispatch deferred behind the previous step's
    // update (pending grammar) must not rebuild from a KV manager that update already changed
    ex_ = std::make_shared<PyExec>();
    py::object pre = py::hasattr(worker, "prebuild") ? worker.attr("prebuild") : py::object();
    std::shared_ptr<PyExec> ex = ex_;
    BatchBuilder bb = [kv, wpt, pre, ex](const StepPlan& plan, int rank, StepBatch* out) {
      if (rank == 0 && pre) {
        bool ok = false;
        ex->call([&] {
          try {
            pre(plan_tuple(plan));
            ok = true;
          } catch (py::error_already_set& e) {
            std::fprintf(stderr, "core2b worker prebuild: %s\n", e.what());
          }
        });
        if (!ok) return Status::INTERNAL;
      }
      KVViews v;
      v.seq = [kv](const Rid& id) { return kv->seq(id); };
      std::string err;
      Status s = build_step_batch(plan, v, PAGE_TOKENS, wpt, rank, out, &err);
      if (s != Status::OK) std::fprintf(stderr, "core2b build_step_batch rank %d: %s\n", rank, err.c_str());
      return s;
    };
    pl_ = std::make_shared<Payloads>();
    // images (design §5a): cfg["mm"] = core2b_proc._MM (admit / free / step); absent = image requests are rejected
    std::shared_ptr<MmBridge> mmb;
    if (cfg.contains("mm") && !cfg["mm"].is_none()) {
      mm_ = cfg["mm"];
      mmb = std::make_shared<MmBridge>();
      mmb->ex = ex_;
      mmb->mm = &mm_;
      mmb_ = mmb;
    }
    std::string err;
    // the device worker: cfg["native"] = {"kind": "fake", "vocab", "k", "seed"} -> NativeWorker over the C++ fake
    // device (no Python per step); else the 2b-α0 Python worker
    std::unique_ptr<IWorker> wk;
    if (cfg.contains("native")) {
      py::dict nv = cfg["native"].cast<py::dict>();
      std::string kind = nv["kind"].cast<std::string>();
      core2b::NativeWorkerConfig wc;
      wc.k1 = nv["k"].cast<int32_t>() + 1;
      if (nv.contains("sources")) wc.sources = nv["sources"].cast<std::vector<int>>();
      if (nv.contains("stub_indexer")) wc.stub_indexer = nv["stub_indexer"].cast<bool>();
      if (nv.contains("full_block_tables")) wc.full_block_tables = nv["full_block_tables"].cast<bool>();
      std::shared_ptr<Payloads> pl = pl_;
      // frees, every step (C++): payloads of finished requests; image frees posted to Python (MmBridge)
      core2b::MmFrees fr = [mmb, pl](const std::vector<Rid>& f, const std::vector<Rid>& p) {
        for (const auto& r : f) pl->drop(r);
        if (mmb) mmb->frees(f, p);
      };
      core2b::EmbeddedFn emb;
      core2b::PostEmbedded post;
      core2b::MixedBodyFn mbody;          // the Python halves of a native mixed step
      core2b::MixedPostFn mpost;
      core2b::MixedGuardFn mguard;
      core2b::OnCtxProbeFn oprobe;        // H2: PHYS_CXX_ONCTX_CHECK's probe (_Embedded.post_probe)
      core2b::MixedOnCtxFn monctx;        // a native step's admissions / drops -> Python's hooks.on_ctx
      core2b::MixedReplayFn mreplay;      // on_ctx lazy: queued admissions -> core2b_proc._Embedded.replay
      core2b::MixedEnsureFn mensure;      // the engine snapshot)
      std::unique_ptr<fpr::p0::MixedSnap> msnap;
      const FprEngineHandover* hov = nullptr;
      if (kind == "fake") {
        core2b::SimConfig sc;         // the simulated device timeline (sim_step_ms > 0: the overlap measurement)
        if (nv.contains("sim_step_ms")) sc.step_ms = nv["sim_step_ms"].cast<double>();
        if (nv.contains("sim_h_frac")) sc.h_frac = nv["sim_h_frac"].cast<double>();
        if (nv.contains("sim_t_ms")) sc.t_ms = nv["sim_t_ms"].cast<double>();
        if (nv.contains("sim_prep_ms")) sc.prep_ms = nv["sim_prep_ms"].cast<double>();
        nd_ = std::make_unique<core2b::FakeNd>(nv["vocab"].cast<int64_t>(), nv["k"].cast<int32_t>(),
                                               nv.contains("seed") ? nv["seed"].cast<int64_t>() : 0, sc);
        ndp_ = nd_.get();
        // the fake's embedded step (image steps): Python's planner + every rank's EncoderCache (mm.step: drop, encode,
        // gather on the Python fake worker's caches), then the fake device's own step (FakeNd's embedded_begin / end
        // are no-ops: its dispatch observes the host)
        nd::IEngine2b* ndp = ndp_;
        emb = [ndp, mmb](const StepInput& in, const std::vector<CompactTables>& cts, const nd::StepHost& host,
                         std::vector<StepResult>* out) -> Status {
          if (mmb && core2b::has_image_rows(in.batches)) {
            bool ok = false;
            if (!mmb->ex->call([&] {
                  try {
                    mmb->mm->attr("step")(MmBridge::view(in.batches));
                    ok = true;
                  } catch (py::error_already_set& e) {
                    std::fprintf(stderr, "core2b mm step: %s\n", e.what());
                  }
                }) || !ok)
              return Status::INTERNAL;
          }
          nd::Handle* h = nullptr;
          std::string e;
          if (ndp->dispatch(in.batches, cts, host, &h, &e) != Status::OK) return Status::INTERNAL;
          return ndp->wait(h, out, &e);
        };
      } else if (kind == "nd" || kind == "embed_all") {
        std::shared_ptr<PyExec> ex = ex_;
        if (kind == "nd") {
          // the
          // Python Engine2b object (nv["owner"]) and the exporter behind the handover stay alive in core2b_proc
          py::capsule cap = nv["engine"].cast<py::capsule>();
          if (std::string(cap.name()) != "fpr.IEngine2b")
            throw std::runtime_error("fpr_core2b: native engine must be an fpr.IEngine2b capsule");
          ndp_ = static_cast<nd::IEngine2b*>(cap.get_pointer());
          nd_owner_ = nv["owner"];
          nd::RunPython run_python = [ex](const std::function<void()>& f) { return ex->call(f); };
          std::string ierr;
          auto* h = reinterpret_cast<const FprEngineHandover*>((uintptr_t)nv["handover"].cast<uint64_t>());
          if (ndp_->init(h, run_python, &ierr) != Status::OK) throw std::runtime_error("fpr_core2b: nd init: " + ierr);
          hov = h;
        } else {
          nd_ = std::make_unique<EmbedAllNd>();
          ndp_ = nd_.get();
        }
        emb_ = nv["embedded"];
        wc.draw_seeds = true;
        {                                 // on_ctx lazy (default off); the check mode implies it
          const char* lz = std::getenv("PHYS_CXX_ONCTX_LAZY");
          const char* lc = std::getenv("PHYS_CXX_ONCTX_LAZY_CHECK");
          wc.onctx_lazy_check = lc != nullptr && std::string(lc) == "1";
          wc.onctx_lazy = wc.onctx_lazy_check || (lz != nullptr && std::string(lz) == "1");
        }
        py::object* embp = &emb_;
        // the embedded engine step: core2b_proc._Embedded.step on the engine's main thread (serve Worker.copies +
        // backend.step_compact with THESE CompactTables and serve StepBatches over the same page lists)
        emb = [ex, embp, mmb, pl](const StepInput& in, const std::vector<CompactTables>& cts, const nd::StepHost& host,
                                  std::vector<StepResult>* out) -> Status {
          Status st = Status::INTERNAL;
          if (!ex->call([&] {
                try {
                  py::list bt, cd;
                  for (size_t r = 0; r < in.batches.size(); ++r) {
                    if (!in.plan) throw std::runtime_error("an embedded step with batches but no plan");
                    bt.append(batch_tuple(*in.plan, in.batches[r]));
                    cd.append(ct_dict(cts[r]));
                  }
                  py::list fresh, gr;
                  for (const auto& f : host.fresh) {
                    py::list fr;      // the ADD frames: the request's own SamplingParams for its SeqInit
                    for (const auto& x : pl->get(f.rid)) fr.append(py::bytes(x));
                    fresh.append(py::make_tuple(f.rid, f.serial, f.prompt, f.output,
                                                f.sampling.has_seed ? py::object(py::int_(f.sampling.seed)) : py::object(py::none()),
                                                f.structured, fr));
                  }
                  for (const auto& g : host.grammar)
                    gr.append(py::make_tuple(g.rid, py::bytes(reinterpret_cast<const char*>(g.rows.data()), g.rows.size() * 4),
                                             g.n_rows, g.words, g.num_acceptable));
                  py::dict hd;
                  hd["fresh"] = fresh;
                  hd["dropped"] = host.dropped;
                  hd["grammar"] = gr;
                  hd["prev"] = host.prev_tokens;
                  hd["order"] = host.order;
                  py::object mmv = py::none();
                  if (mmb && core2b::has_image_rows(in.batches)) mmv = MmBridge::view(in.batches);
                  py::object pt = in.plan ? py::object(plan_tuple(*in.plan)) : py::object(py::none());
                  results_of((*embp).attr("step")(in.step, pt, bt, cd, hd, mmv), out);
                  st = Status::OK;
                } catch (std::exception& e) {   // py::error_already_set included
                  std::fprintf(stderr, "core2b embedded step %lld: %s\n", (long long)in.step, e.what());
                }
              }))
            return Status::INTERNAL;
          return st;
        };
        if (kind == "nd" && nv.contains("mixed") && nv["mixed"].cast<bool>()) {
          wc.mixed_native = true;
          if (nv.contains("tail_bound")) wc.tail_bound = nv["tail_bound"].cast<bool>();
          // core2b_proc._Embedded.mixed_body / mixed_post on the engine's main thread (PyExec)
          mbody = [ex, embp, pl](const StepInput& in, const std::vector<CompactTables>& cts, const nd::StepHost& host,
                                 const std::vector<StepBatch>& rb, const std::vector<RankTables>& tabs,
                                 core2b::MixedKeys* out) -> Status {
            Status st = Status::INTERNAL;
            if (!ex->call([&] {
                  try {
                    if (!in.plan) throw std::runtime_error("a mixed step without a plan");
                    py::list bt, cd, rbt, td;
                    for (size_t r = 0; r < in.batches.size(); ++r) {
                      bt.append(batch_tuple(*in.plan, in.batches[r]));
                      cd.append(ct_dict(cts[r]));
                      rbt.append(rb_tuple(rb[r]));
                      td.append(rank_tables_dict(tabs[r]));
                    }
                    py::list fresh, gr;
                    for (const auto& f : host.fresh) {
                      py::list fr;
                      for (const auto& x : pl->get(f.rid)) fr.append(py::bytes(x));
                      fresh.append(py::make_tuple(f.rid, f.serial, f.prompt, f.output,
                                                  f.sampling.has_seed ? py::object(py::int_(f.sampling.seed)) : py::object(py::none()),
                                                  f.structured, fr));
                    }
                    for (const auto& g : host.grammar)
                      gr.append(py::make_tuple(g.rid, py::bytes(reinterpret_cast<const char*>(g.rows.data()), g.rows.size() * 4),
                                               g.n_rows, g.words, g.num_acceptable));
                    py::dict hd;
                    hd["fresh"] = fresh;
                    hd["dropped"] = host.dropped;
                    hd["grammar"] = gr;
                    hd["prev"] = host.prev_tokens;
                    hd["order"] = host.order;
                    py::tuple k = (*embp).attr("mixed_body")(in.step, plan_tuple(*in.plan), bt, cd, hd, rbt, td).cast<py::tuple>();
                    out->r_a = k[0].cast<int64_t>();
                    out->r_t = k[1].cast<int64_t>();
                    out->key_t = k[2].cast<std::string>();
                    st = Status::OK;
                  } catch (std::exception& e) {
                    std::fprintf(stderr, "core2b mixed body %lld: %s\n", (long long)in.step, e.what());
                  }
                }))
              return Status::INTERNAL;
            return st;
          };
          monctx = [ex, embp, pl](const StepInput& in, const nd::StepHost& host) -> Status {
            Status st = Status::INTERNAL;
            if (!ex->call([&] {
                  try {
                    if (!in.plan) throw std::runtime_error("a mixed step without a plan");
                    py::list bt, fresh, gr;
                    for (size_t r = 0; r < in.batches.size(); ++r) bt.append(batch_tuple(*in.plan, in.batches[r]));
                    for (const auto& f : host.fresh) {
                      py::list fr;
                      for (const auto& x : pl->get(f.rid)) fr.append(py::bytes(x));
                      fresh.append(py::make_tuple(f.rid, f.serial, f.prompt, f.output,
                                                  f.sampling.has_seed ? py::object(py::int_(f.sampling.seed)) : py::object(py::none()),
                                                  f.structured, fr));
                    }
                    for (const auto& g : host.grammar)
                      gr.append(py::make_tuple(g.rid, py::bytes(reinterpret_cast<const char*>(g.rows.data()), g.rows.size() * 4),
                                               g.n_rows, g.words, g.num_acceptable));
                    py::dict hd;
                    hd["fresh"] = fresh;
                    hd["dropped"] = host.dropped;
                    hd["grammar"] = gr;
                    hd["prev"] = host.prev_tokens;
                    hd["order"] = host.order;
#if defined(FPR_ND_API_VERSION) && FPR_ND_API_VERSION >= 326
                    hd["lazy"] = host.on_ctx_lazy;   // a lazy step calls on_ctx only for the check mode's shadow
#endif
                    (*embp).attr("on_ctx")(in.step, plan_tuple(*in.plan), bt, hd);
                    st = Status::OK;
                  } catch (std::exception& e) {
                    std::fprintf(stderr, "core2b mixed on_ctx %lld: %s\n", (long long)in.step, e.what());
                  }
                }))
              return Status::INTERNAL;
            return st;
          };
          mreplay = [ex, embp, pl](const std::vector<core2b::LazyAdmission>& adm) -> Status {
            Status st = Status::INTERNAL;
            if (!ex->call([&] {
                  try {
                    py::list ents;
                    for (const auto& a : adm) {
                      py::list ids, fr;
                      for (int64_t i = 0; i < a.n; ++i) ids.append(a.ids[i]);
                      for (const auto& x : pl->get(a.rid)) fr.append(py::bytes(x));
                      ents.append(py::make_tuple(a.rid, a.serial, ids,
                                                 a.has_seed ? py::object(py::int_(a.seed)) : py::object(py::none()),
                                                 a.structured, fr));
                    }
                    (*embp).attr("replay")(ents);
                    st = Status::OK;
                  } catch (std::exception& e) {
                    std::fprintf(stderr, "core2b lazy replay: %s\n", e.what());
                  }
                }))
              return Status::INTERNAL;
            return st;
          };
          mguard = [ex, embp](const std::string& key, std::string* why) -> bool {
            bool ok = false;
            ex->call([&] {
              try {
                py::tuple r = (*embp).attr("h1_guard_ok")(key).cast<py::tuple>();
                ok = r[0].cast<bool>();
                *why = r[1].cast<std::string>();
              } catch (std::exception& e) {
                *why = std::string("guard raised: ") + e.what();
              }
            });
            return ok;
          };
          if (nv.contains("mixed_exec")) {           // core2b::run_mixed when every op is native
            wc.mixed_exec = true;
            if (nv.contains("mixed_async")) wc.mixed_async = nv["mixed_async"].cast<bool>();   // L4 (default off)
            msnap.reset(new fpr::p0::MixedSnap(mixed_snap_of(nv["mixed_exec"].cast<py::dict>())));
            mensure = [ex, embp](int64_t RA, int64_t RT, const std::map<std::string, int64_t>& stats,
                                 std::string* why) -> Status {
              Status st = Status::BAD_ARG;
              ex->call([&] {
                try {
                  py::dict sd;
                  for (const auto& kv : stats) sd[py::str(kv.first)] = kv.second;
                  py::tuple r = (*embp).attr("ensure_mixed_views")(RA, RT, sd).cast<py::tuple>();
                  if (r[0].cast<bool>()) st = Status::OK;
                  else *why = r[1].cast<std::string>();
                } catch (std::exception& e) {
                  *why = std::string("ensure_mixed_views raised: ") + e.what();
                }
              });
              return st;
            };
          }
          mpost = [ex, embp]() -> Status {
            Status st = Status::INTERNAL;
            ex->call([&] {
              try {
                (*embp).attr("mixed_post")();
                st = Status::OK;
              } catch (std::exception& e) {
                std::fprintf(stderr, "core2b mixed post: %s\n", e.what());
              }
            });
            return st;
          };
        }
        oprobe = [ex, embp](int64_t* changed, std::string* what) -> Status {   // H2: the publication + what it changed
          Status st = Status::INTERNAL;
          ex->call([&] {
            try {
              py::tuple r = (*embp).attr("post_probe")().cast<py::tuple>();
              *changed = r[0].cast<int64_t>();
              *what = r[1].cast<std::string>();
              st = Status::OK;
            } catch (std::exception& e) {
              std::fprintf(stderr, "core2b post probe: %s\n", e.what());
            }
          });
          return st;
        };
        post = [ex, embp]() -> Status {   // exporter.ensure_bucket (new buckets) + refresh_graphs, at a step boundary
          Status st = Status::INTERNAL;
          ex->call([&] {
            try {
              (*embp).attr("post")();
              st = Status::OK;
            } catch (py::error_already_set& e) {
              std::fprintf(stderr, "core2b post-embedded: %s\n", e.what());
            }
          });
          return st;
        };
      } else {
        throw std::runtime_error("fpr_core2b: native kind " + kind + " (fake | nd | embed_all)");
      }
      nw_ = new core2b::NativeWorker(ndp_, wc, emb, fr, post);
      {
        std::shared_ptr<PyExec> exr = ex_;
        nw_->set_py_epoch([exr]() { return exr->runs(); });
      }
      if (mbody) nw_->set_mixed(mbody, mpost, mguard, hov);
      if (monctx) nw_->set_mixed_onctx(monctx);
      if (monctx && oprobe) nw_->set_onctx_probe(oprobe);
      if (mreplay && wc.onctx_lazy) nw_->set_mixed_replay(mreplay);   // unset = today's eager on_ctx
      // today's behaviour plus the executor's refusal table
      if (msnap) {
        nw_->set_mixed_exec(*msnap, mensure, nullptr, nullptr);
        core2b::SpanEngineFlags sf;   // _issue_span's switches + the loaded driver
        std::string dso;
        {
          py::dict me = nv["mixed_exec"].cast<py::dict>();
          if (me.contains("span_flags")) {
            py::dict f = me["span_flags"].cast<py::dict>();
            sf.l2pf = f["l2pf"].cast<bool>(); sf.nancheck = f["nancheck"].cast<bool>();
            sf.prof_pieces = f["prof_pieces"].cast<bool>(); sf.tp_on = f["tp_on"].cast<bool>();
            sf.mega_on = f["mega_on"].cast<bool>(); sf.router_pack3 = f["router_pack3"].cast<bool>();
            sf.agrs_on = f["agrs_on"].cast<bool>(); sf.agrs_ag32 = f["agrs_ag32"].cast<bool>();
            sf.agrs_agce = f["agrs_agce"].cast<bool>(); sf.agrs_min_r = f["agrs_min_r"].cast<int64_t>();
          }
          if (me.contains("driver_so")) dso = me["driver_so"].cast<std::string>();
        }
        // no MoeVariantFn yet -> SET_STEP / ISSUE / JOIN_MIX refuse (logged)
        nw_->set_mixed_driver(dso, nullptr, sf);
      }
      wk.reset(nw_);
    } else {
      wk.reset(new PyWorker(worker, pl_, ex_, cfg.contains("check_k1") ? cfg["check_k1"].cast<int32_t>() : 0));
    }
    auto core = make_core(c, std::unique_ptr<IScheduler>(new BorrowedScheduler(rs)), bb, std::move(wk),
                          core::kv_facade_of([kv]() -> const IKVManager* { return kv; }),
                          pycfg::hasher(c.lifecycle.page_tokens), &err);
    if (!core) throw std::runtime_error("fpr_core2b: make_core: " + err);
    // payloads: the 2b-α0 Python worker decodes SamplingParams from them (drops at finish); native: the embedded step
    // reads them for admissions (SeqInit sampling), dropped by the frees hook at finish
    core_ = std::make_unique<TapCore>(std::move(core), pl_, mmb);
  }
  ~Core2b() {
    if (ex_) ex_->stop();
    {
      py::gil_scoped_release nogil;
      if (core_) core_->stop();
    }
    core_.reset();
    nd_.reset();
    mm_ = py::object();
    emb_ = py::object();
    nd_owner_ = py::object();
  }
  ICore* get() { return core_.get(); }
  std::string fatal() const { return core_->fatal(); }
  std::string native_stats() const { return ndp_ ? ndp_->stats_json() : std::string(); }
  // the shared "previous AG bucket" with Python's E._agrs_st_last (core2b_proc._Embedded._agce_pull / push)
  std::string take_last_seg() { return nw_ ? nw_->take_last_seg() : std::string(); }
  void set_last_seg(const std::string& seg) { if (nw_) nw_->set_last_seg(seg); }
  py::dict worker_stats() const {     // NativeWorker routing + image admission counters
    py::dict d;
    d["lc_ghosts"] = fpr::core::lifecycle_ghosts();   // ghost fix: always present, gate bar 0
    d["lc_pending_dropped"] = fpr::core::lifecycle_pending_dropped();
    if (nw_) {
      d["native"] = nw_->n_native();
      d["embedded"] = nw_->n_embedded();
      d["embedded_mm"] = nw_->n_embedded_mm();
      d["mixed_native"] = nw_->n_mixed_native();
      d["mixed_exec"] = nw_->n_mixed_exec();
      d["mixed_async"] = nw_->n_mixed_async();
      py::dict tm;                    // L0: {name: (count, mean us, max us)}
      for (const auto& kv : nw_->timers())
        tm[py::str(kv.first)] = py::make_tuple(kv.second.n, kv.second.n ? kv.second.us / (double)kv.second.n : 0.0, kv.second.max_us);
      d["timers"] = tm;
      py::dict rf;
      for (const auto& e : nw_->mixed_refusals()) rf[py::str(e.first)] = e.second;
      d["mixed_refusals"] = rf;
      py::dict drf;                   // decode-side embedded steps by class
      for (const auto& e : nw_->decode_refusals()) drf[py::str(e.first)] = e.second;
      d["decode_refusals"] = drf;
    }
    if (mmb_) {
      d["mm_admitted"] = mmb_->admitted.load();
      d["mm_rejected"] = mmb_->rejected.load();
      d["mm_free_posts"] = mmb_->posts.load();
      std::lock_guard<std::mutex> g(mmb_->mu);
      d["mm_live"] = (int64_t)mmb_->rids.size();
    }
    return d;
  }
  bool has_work() const { return core_->has_work(); }
  void set_h1_epoch(uint64_t addr) {   // H3: mixed_cxx_head.epoch_addr() -> the NativeWorker
    if (nw_) nw_->set_h1_epoch(addr);
  }
  // the loop thread; its Python work then runs only in serve_python() on the caller's thread
  void start() {
    ex_->set_inline(false);
    core_->start();
  }
  int serve_python(double timeout_s) { return ex_->serve(timeout_s); }
  void stop() {
    ex_->stop();                        // queued Python jobs are cancelled; the loop sees the worker fail and stops
    py::gil_scoped_release nogil;
    core_->stop();
  }
  bool step_once() {
    py::gil_scoped_release nogil;
    return core_->step_once();
  }
  void input(int type, py::bytes payload) {
    std::string p = payload, err;
    if (core_->input((uint8_t)type, p, &err) != Status::OK) throw std::runtime_error("fpr_core2b input: " + err);
  }
  py::list pop_outputs(double timeout_s) {   // [(client_index, [frames])]
    std::vector<WireOut> b;
    {
      py::gil_scoped_release nogil;
      core_->pop_outputs(&b, timeout_s);
    }
    py::list out;
    for (const auto& w : b) {
      py::list fr;
      for (const auto& f : w.frames) fr.append(py::bytes(f));
      out.append(py::make_tuple(w.client_index, fr));
    }
    return out;
  }

 private:
  py::object owner_;                  // keeps the rank scheduler (and its KV manager) alive
  std::shared_ptr<Payloads> pl_;
  std::shared_ptr<PyExec> ex_;
  py::object mm_;                     // core2b_proc._MM (images) or None
  std::shared_ptr<MmBridge> mmb_;
  std::unique_ptr<nd::IEngine2b> nd_; // the native device when owned (fake; declared before core_: outlives the loop)
  nd::IEngine2b* ndp_ = nullptr;      // the native device
  py::object nd_owner_, emb_;         // nd: the Python fpr_nd2.Engine2b + core2b_proc._Embedded
  core2b::NativeWorker* nw_ = nullptr;   // owned by core_ (the native worker, for stats)
  std::unique_ptr<ICore> core_;
};

class PyServer {
 public:
  void start(std::vector<std::string> inputs, std::vector<std::string> outputs, py::bytes identity, py::bytes ready,
             int poll_ms) {
    core2b::ServerConfig c;
    c.inputs = std::move(inputs);
    c.outputs = std::move(outputs);
    c.identity = identity;
    c.ready_payload = ready;
    c.poll_ms = poll_ms;
    std::string err;
    bool ok;
    {
      py::gil_scoped_release nogil;
      ok = s_.start(c, &err);
    }
    if (!ok) throw std::runtime_error("fpr_core2b.Server.start: " + err);
  }
  void attach(Core2b& c) { s_.attach(c.get()); }
  void dead() {
    py::gil_scoped_release nogil;
    s_.dead();
  }
  void stop() {
    py::gil_scoped_release nogil;
    s_.stop();
  }
  py::dict stats() const {
    const auto& t = s_.stats();
    py::dict d;
    d["in_msgs"] = t.in_msgs.load();
    d["in_bad"] = t.in_bad.load();
    d["in_buffered"] = t.in_buffered.load();
    d["out_msgs"] = t.out_msgs.load();
    d["out_frames"] = t.out_frames.load();
    d["out_bytes"] = t.out_bytes.load();
    d["out_dropped"] = t.out_dropped.load();
    d["last_error"] = s_.last_error();
    return d;
  }

 private:
  core2b::Server s_;
};

}  // namespace

PYBIND11_MODULE(fpr_core2b, m) {
  m.doc() = "engine process: V1 socket IO + the C++ ICore loop";
  py::class_<Core2b>(m, "Core2b")
      .def(py::init<py::dict, py::capsule, py::capsule, py::object, py::object, py::list, py::object, int64_t>())
      .def("start", &Core2b::start)
      .def("serve_python", &Core2b::serve_python)
      .def("stop", &Core2b::stop)
      .def("step_once", &Core2b::step_once)
      .def("fatal", &Core2b::fatal)
      .def("native_stats", &Core2b::native_stats)
      .def("take_last_seg", &Core2b::take_last_seg)
      .def("set_last_seg", &Core2b::set_last_seg)
      .def("worker_stats", &Core2b::worker_stats)
      .def("set_h1_epoch", &Core2b::set_h1_epoch)
      .def("has_work", &Core2b::has_work)
      .def("input", &Core2b::input)
      .def("pop_outputs", &Core2b::pop_outputs);
  // zero-copy views of the embedded step's page lists / window tables: fpr_kvtables.Pages / Window's API
  py::class_<PagesView>(m, "Pages")
      .def("__len__", [](const PagesView& p) { return p.v->size(); })
      .def("__getitem__", [](const PagesView& p, int64_t i) { return (*p.v)[(size_t)norm_index(i, (int64_t)p.v->size())]; })
      .def("__getitem__", [](const PagesView& p, py::slice s) {
        size_t start, stop, step, len;
        if (!s.compute(p.v->size(), &start, &stop, &step, &len)) throw py::error_already_set();
        py::list l;
        for (size_t i = 0; i < len; ++i) l.append((*p.v)[start + i * step]);
        return l;
      })
      .def("__iter__", [](const PagesView& p) { return py::iter(py::cast(*p.v)); })
      .def("__contains__", [](const PagesView& p, int64_t x) { return std::find(p.v->begin(), p.v->end(), x) != p.v->end(); })
      .def("__eq__", [](const PagesView& p, py::object o) {
        if (py::isinstance<PagesView>(o)) return *p.v == *o.cast<PagesView&>().v;
        try {
          return *p.v == o.cast<std::vector<int64_t>>();
        } catch (...) {
          return false;
        }
      })
      .def("index", [](const PagesView& p, int64_t x) {
        auto it = std::find(p.v->begin(), p.v->end(), x);
        if (it == p.v->end()) throw py::value_error(std::to_string(x) + " is not in list");
        return (int64_t)(it - p.v->begin());
      })
      .def("count", [](const PagesView& p, int64_t x) { return (int64_t)std::count(p.v->begin(), p.v->end(), x); })
      .def("__reversed__", [](const PagesView& p) { std::vector<int64_t> r(p.v->rbegin(), p.v->rend()); return py::iter(py::cast(r)); })
      .def("copy", [](const PagesView& p) { return *p.v; })
      .def("__bool__", [](const PagesView& p) { return !p.v->empty(); })
      .def("tolist", [](const PagesView& p) { return *p.v; })
      .def("ptr", [](const PagesView& p) { return (uintptr_t)p.v.get(); })
      .def("use_count", [](const PagesView& p) { return (int64_t)p.v.use_count(); })
      .def("__repr__", [](const PagesView& p) { return "Pages(" + std::to_string(p.v->size()) + ")"; })
      .def("__reduce__", [](const PagesView& p) {
        return py::make_tuple(py::module_::import("builtins").attr("list"), py::make_tuple(py::cast(*p.v)));
      });
  py::class_<WindowView>(m, "Window")
      .def("__len__", [](const WindowView& w) { return w.w->size(); })
      .def("__contains__", [](const WindowView& w, py::object b) { return py::isinstance<py::int_>(b) && w.w->contains(b.cast<int64_t>()); })
      .def("__getitem__", [](const WindowView& w, int64_t b) {
        int64_t i = w.w->find(b);
        if (i < 0) throw py::key_error(std::to_string(b));
        return w.w->items[(size_t)i].second;
      })
      .def("get", [](const WindowView& w, int64_t b, py::object d) -> py::object {
        int64_t i = w.w->find(b);
        return i < 0 ? d : py::object(py::int_(w.w->items[(size_t)i].second));
      }, py::arg("b"), py::arg("default") = py::none())
      .def("keys", [](const WindowView& w) { std::vector<int64_t> k; for (auto& e : w.w->items) k.push_back(e.first); return k; })
      .def("values", [](const WindowView& w) { std::vector<int64_t> k; for (auto& e : w.w->items) k.push_back(e.second); return k; })
      .def("items", [](const WindowView& w) { return w.w->items; })
      .def("__iter__", [](const WindowView& w) { std::vector<int64_t> k; for (auto& e : w.w->items) k.push_back(e.first); return py::iter(py::cast(k)); })
      .def("__bool__", [](const WindowView& w) { return !w.w->empty(); })
      .def("copy", [](const WindowView& w) { py::dict d; for (auto& e : w.w->items) d[py::int_(e.first)] = e.second; return d; })
      .def("__eq__", [](const WindowView& w, py::object o) {
        if (py::isinstance<WindowView>(o)) return w.w->items == o.cast<WindowView&>().w->items;
        if (!py::isinstance<py::dict>(o)) return false;
        py::dict d = o.cast<py::dict>();
        if (d.size() != w.w->size()) return false;
        for (auto& e : w.w->items) {
          py::int_ k(e.first);
          if (!d.contains(k) || d[k].cast<int64_t>() != e.second) return false;
        }
        return true;
      })
      .def("ptr", [](const WindowView& w) { return (uintptr_t)w.w.get(); })
      .def("use_count", [](const WindowView& w) { return (int64_t)w.w.use_count(); })
      .def("__repr__", [](const WindowView& w) { return "Window(" + std::to_string(w.w->size()) + ")"; })
      .def("__reduce__", [](const WindowView& w) {
        py::list it;
        for (auto& e : w.w->items) it.append(py::make_tuple(e.first, e.second));
        return py::make_tuple(py::module_::import("builtins").attr("dict"), py::make_tuple(it));
      });
  py::class_<PyServer>(m, "Server")
      .def(py::init<>())
      .def("start", &PyServer::start, py::arg("inputs"), py::arg("outputs"), py::arg("identity"), py::arg("ready"),
           py::arg("poll_ms") = 50)
      .def("attach", &PyServer::attach, py::keep_alive<1, 2>())
      .def("dead", &PyServer::dead)
      .def("stop", &PyServer::stop)
      .def("stats", &PyServer::stats);
}
