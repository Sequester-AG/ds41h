// native_worker.cc -- see native_worker.h.
#include <sys/stat.h>
#include <cstdlib>
#include "native_worker.h"

#include "../p0/moe_decide.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include <cstdio>
#include <cstdlib>

namespace fpr {
namespace core2b {

namespace {
using Clk = std::chrono::steady_clock;
double us_since(Clk::time_point t0) { return std::chrono::duration<double, std::micro>(Clk::now() - t0).count(); }
struct NwHandle {                     // native: nd's handle; embedded: the step's results (it ran synchronously)
  nd::Handle* nd = nullptr;
  bool native = true;
  bool deferred = false;              // L4: results come from drain_mixed_ (the step's sync + finish not run yet)
  std::vector<StepResult> results;
};
}  // namespace

bool has_prefill(const std::vector<StepBatch>& b) {
  for (const auto& sb : b)
    for (const auto& se : sb.seqs)
      if (se.kind != "decode") return true;
  return false;
}

bool has_image_rows(const std::vector<StepBatch>& b) {
  for (const auto& sb : b)
    for (const auto& se : sb.seqs)
      if (!se.images.empty() && se.kind != "decode" && se.start >= 0 && se.n_rows > 0) return true;
  return false;
}

NativeWorker::NativeWorker(nd::IEngine2b* nd, const NativeWorkerConfig& cfg, EmbeddedFn embedded, MmFrees frees,
                           PostEmbedded post)
    : nd_(nd), cfg_(cfg), embedded_(std::move(embedded)), frees_(std::move(frees)), post_(std::move(post)) {
  tb_.sources = cfg_.sources;
  tb_.stub_indexer = cfg_.stub_indexer;
  tb_.full_block_tables = cfg_.full_block_tables;
  // TableBuilder.tokens_of is Python's per-process str hash: the worker always passes row_tokens (StepInput), so a
  // lookup here means a step without them, which is an error, not a silent stand-in
  tb_.tokens_of = [](const Rid&, int64_t) -> int64_t { return -1; };
}

NativeWorker::~NativeWorker() {
  if (pend_.c == nullptr) return;     // L4: never leave nd mid-step (the loop waits every handle: this is the safety net)
  try {
    pend_.h = nullptr;                // its handle may be gone: the results go to a scratch vector
    if (drain_mixed_() != Status::OK) std::fprintf(stderr, "core2b NativeWorker: L4 drain at exit: %s\n", err_.c_str());
  } catch (...) {
  }
}

Status NativeWorker::dispatch(const StepInput& in, StepHandle** h) {
  Status st;
  try {
    st = dispatch_(in, h);
  } catch (std::exception& e) {
    err_ = std::string("exception: ") + e.what();
    st = Status::INTERNAL;
  } catch (...) {
    err_ = "unknown exception";
    st = Status::INTERNAL;
  }
  if (st != Status::OK) std::fprintf(stderr, "core2b NativeWorker dispatch step %lld: %s\n", (long long)in.step, err_.c_str());
  return st;
}

Status NativeWorker::wait(StepHandle* h, std::vector<StepResult>* out) {
  Status st;
  try {
    st = wait_(h, out);
  } catch (std::exception& e) {
    err_ = std::string("exception: ") + e.what();
    st = Status::INTERNAL;
  } catch (...) {
    err_ = "unknown exception";
    st = Status::INTERNAL;
  }
  if (st != Status::OK) std::fprintf(stderr, "core2b NativeWorker wait: %s\n", err_.c_str());
  return st;
}

Status NativeWorker::dispatch_(const StepInput& in, StepHandle** h) {
  // L4: a deferred native mixed step completes first (its sync + finish, results onto its handle): everything below
  // (Python's frees, nd's async_ok / begin / dispatch / embedded_begin) then sees nd with no open step, as without L4
  if (pend_.c != nullptr && drain_mixed_() != Status::OK) return Status::INTERNAL;
  if (frees_ && (!in.finished.empty() || !in.preempted.empty())) {
    frees_(in.finished, in.preempted);
    ++own_py_;
  }
  // worker.build_compact(tb, sb, k1, row_tokens) per rank (Worker.step, compact path)
  std::vector<CompactTables> cts(in.batches.size());
  std::vector<int64_t> rt;
  for (size_t r = 0; r < in.batches.size(); ++r) {
    const std::vector<int64_t>* rtp = nullptr;
    if (r < in.row_tokens.size() && !in.row_tokens[r].empty()) {
      rt.assign(in.row_tokens[r].begin(), in.row_tokens[r].end());
      rtp = &rt;
    }
    if (tb_.build_compact(in.batches[r], cfg_.k1, rtp, &cts[r], &err_) != Status::OK) {
      std::fprintf(stderr, "core2b NativeWorker: build_compact rank %zu: %s\n", r, err_.c_str());
      return Status::INTERNAL;
    }
  }
  // StepContext
  nd::StepHost host;
  for (const auto& q : in.new_reqs) {
    nd::NewSeq s;
    s.rid = q.rid;
    s.serial = ++serial_;             // per admission: a re-admitted request is a new engine-side sequence
    s.prompt = q.prompt;
    s.output = q.output;
    s.sampling = q.sampling;
    if (cfg_.draw_seeds && !s.sampling.has_seed) {   // the same value reaches nd and the embedded step's SeqInit
      s.sampling.seed = (int64_t)seeds_();
      s.sampling.has_seed = true;
    }
    s.structured = q.structured;
    host.fresh.push_back(std::move(s));
  }
  host.dropped = in.finished;
  host.dropped.insert(host.dropped.end(), in.preempted.begin(), in.preempted.end());
  for (const auto& g : in.grammar) {
    nd::GrammarRows x;
    x.rid = g.rid;
    x.rows = g.rows;
    x.n_rows = g.n_rows;
    x.words = g.words;
    x.num_acceptable = g.num_acceptable;
    host.grammar.push_back(std::move(x));
  }
  host.prev_tokens = in.prev_tokens;
  host.order = in.order;
  host.num_spec = in.num_spec;
#if defined(FPR_ND_API_VERSION) && FPR_ND_API_VERSION >= 326
  if (lazy_on_()) {                   // on_ctx lazy: nd accepts admissions without Python's on_ctx
    host.on_ctx_lazy = true;
    for (const auto& rid : host.dropped) lazy_q_.erase(rid);   // admitted + dropped before a sync: never replayed (J drops it)
  }
#endif
  auto hh = std::make_unique<NwHandle>();
  bool mm = has_image_rows(in.batches);   // images: always the embedded engine (design §5a)
  // a native mixed step (header), else the routing below (a mixed step goes embedded)
  if (!mm && cfg_.mixed_native && mixed_body_ && mixed_post_ && h_ && has_prefill(in.batches)) {
    nd::IMixed2b* mx = nd_->mixed();
    std::string mwhy;
    if (mx == nullptr) {
      mwhy = "nd has no IMixed2b";
    } else if (mx->can_head(in.batches, cts, host, &mwhy) && h1_guard_ok_(mx, in.batches, host, &mwhy)) {
      const auto t0 = Clk::now();
      const uint64_t pe0 = py_epoch_();
      const uint64_t bep0 = fpr_eh_buckets_epoch(h_);
      step_transparent_ = 0;
      Status s = mixed_step_(mx, in, cts, host, &hh->results);
      tick_("mixed.step", us_since(t0));
      const uint64_t pe1 = py_epoch_();
      const bool py_free = pe1 == pe0;   // the step ran no Python at all (a native step, after L3a's lazy push)
      // H2: its only Python was transparent on_ctx jobs (no other job, buckets_epoch unchanged): nothing to publish
      const bool state_only = !py_free && onctx_skip_ && step_transparent_ > 0 && pe1 - pe0 == step_transparent_ &&
                              fpr_eh_buckets_epoch(h_) == bep0;
      if (s == Status::OK) {
        ++n_mixed_;
        hh->native = false;           // completed: wait() returns hh->results
        if (pend_.c != nullptr) {     // L4: deferred by the SYNC handler: the results arrive at the drain
          hh->deferred = true;
          pend_.h = hh.get();
          ++n_async_;
          if (!py_free && drain_mixed_() != Status::OK) return Status::INTERNAL;   // (the handler checked: unreachable)
        }
        const auto tp = Clk::now();
        if (py_free) {                // nothing Python created this step: nothing to publish (L3)
          tick_("mixed.publish skipped", 0.0);
        } else if (state_only && !(onctx_check_ && onctx_probe_)) {
          tick_("mixed.publish skipped (state-only)", 0.0);
        } else if (state_only) {      // H2 check: publish anyway, count what it changed (must be nothing)
          const uint64_t pp0 = py_epoch_();
          const uint64_t bp0 = fpr_eh_buckets_epoch(h_);
          int64_t changed = 0;
          std::string what;
          ++own_py_;
          if (onctx_probe_(&changed, &what) != Status::OK) {
            err_ = "post-mixed publication (H2 probe) failed";
            return Status::INTERNAL;
          }
          tick_("mixed.publish probe (Python)", us_since(tp));
          if (fpr_eh_buckets_epoch(h_) != bp0) {
            ++changed;
            what += " buckets_epoch";
          }
          if (changed != 0) {
            tick_("h2 check CHANGED", 0.0);
            if (++n_check_changed_ <= 20)
              std::fprintf(stderr, "core2b H2 check: a state-only step's publication changed %lld item(s):%s\n",
                           (long long)changed, what.c_str());
          } else {
            tick_("h2 check ok", 0.0);
            if (ens_valid_ && ens_py_ == pp0) ens_py_ = py_epoch_();   // as the skip would have left it
          }
        } else if (post_) {
          const uint64_t pp0 = py_epoch_();
          ++own_py_;
          if (post_() != Status::OK) {  // graphs / buckets the Python halves created (H1 / T captures)
            err_ = "post-mixed publication failed";
            return Status::INTERNAL;
          }
          tick_("mixed.publish (Python)", us_since(tp));
          // the publication leaves the mixed views as they were unless it created / evicted (buckets_epoch moved): the
          // ensure cache stays valid; the H1 guard cache does not (the exporter's scan may retire keys)
          if (ens_valid_ && ens_py_ == pp0 && fpr_eh_buckets_epoch(h_) == ens_ep_) ens_py_ = py_epoch_();
        }
        *h = reinterpret_cast<StepHandle*>(hh.release());
        return Status::OK;
      }
      if (s != Status::BAD_ARG) return Status::INTERNAL;   // fatal: nd is mid-step
      mwhy = "begin: " + err_;        // refused before any device op
    }
    note_refusal(mwhy);
  }
  std::string why;
  bool lazy_dec = false;              // on_ctx lazy: this native decode step's admissions are queued
  bool onctx = false;                 // this native decode step's admissions go through Python's on_ctx first
#if defined(FPR_ND_API_VERSION) && FPR_ND_API_VERSION >= 324
  // a decode step that admits requests can run natively once Python's hooks.on_ctx has
  // created their state. Ask nd with the flag set; only if it accepts does the on_ctx job run (an embedded step runs
  // on_ctx itself, so it must not have run here: exactly once either way). Drop-only steps already go native.
#if FPR_ND_API_VERSION >= 326
  if (!mm && lazy_on_() && !host.fresh.empty()) {   // on_ctx lazy: no Python, the admissions are queued
    why = nd_->async_ok(in.batches, cts, host);
    lazy_dec = why.empty();
  } else
#endif
  if (!mm && mixed_onctx_ && !host.fresh.empty()) {
    host.on_ctx_done = true;
    why = nd_->async_ok(in.batches, cts, host);
    if (why.empty()) {
      onctx = true;
    } else {
      host.on_ctx_done = false;
    }
  } else
#endif
    why = mm ? std::string("mm") : nd_->async_ok(in.batches, cts, host);
  if (onctx) {                        // (why == ""): Python's on_ctx for the admissions, then nd's native dispatch
    const auto t0 = Clk::now();
    const Status s = onctx_call_(in, host);
    tick_("decode.on_ctx (Python)", us_since(t0));
    if (s != Status::OK) {
      err_ = "decode on_ctx failed (see 'core2b mixed on_ctx' above)";
      return Status::INTERNAL;
    }
  }
  if (lazy_dec) {
    lazy_queue_(host);
    if (cfg_.onctx_lazy_check && mixed_onctx_) {   // check mode: the eager on_ctx into Python's SHADOW (core2b_proc)
      ++own_py_;
      if (mixed_onctx_(in, host) != Status::OK) {
        err_ = "decode on_ctx (lazy check shadow) failed";
        return Status::INTERNAL;
      }
    }
  }
  if (why.empty()) {
    ++n_native_;
    const auto t0 = Clk::now();
    if (nd_->dispatch(in.batches, cts, host, &hh->nd, &err_) != Status::OK) return Status::INTERNAL;
    tick_("decode.dispatch", us_since(t0));
  } else {                            // the embedded Python engine: drain + push_state, the step, pull_state
    ++n_embedded_;
    if (mm) ++n_mm_;
    note_decode_refusal(why);         // the reason by class (the census had none on this route)
    hh->native = false;
    if (!embedded_) {
      err_ = "step needs the embedded engine (" + why + ") but the worker has none";
      return Status::INTERNAL;
    }
    auto t0 = Clk::now();
    ++pybody_;   // P1: an embedded step's Python body may change view sources
    flush_touches_("embedded");   // the hits' LRU / _busy touches before this body's own bucket() calls
    if (flush_lazy_(&err_) != Status::OK) return Status::INTERNAL;   // Python learns the queued admissions first
    if (nd_->embedded_begin(host, &err_) != Status::OK) return Status::INTERNAL;
    tick_("embedded.begin (push_state)", us_since(t0));
    t0 = Clk::now();
    ++own_py_;
    Status st = embedded_(in, cts, host, &hh->results);
    tick_(mm ? "embedded.step mm (Python)" : "embedded.step (Python)", us_since(t0));
    t0 = Clk::now();
    if (nd_->embedded_end(&err_) != Status::OK) return Status::INTERNAL;
    tick_("embedded.end (pull_state)", us_since(t0));
    if (st != Status::OK) {
      err_ = "the embedded engine step failed (see 'core2b embedded step' above)";
      return Status::INTERNAL;
    }
    ++own_py_;
    if (post_ && post_() != Status::OK) {
      err_ = "post-embedded publication failed";
      return Status::INTERNAL;
    }
  }
  *h = reinterpret_cast<StepHandle*>(hh.release());   // StepHandle is opaque: each worker casts its own type through it
  return Status::OK;
}

Status NativeWorker::mixed_step_(nd::IMixed2b* mx, const StepInput& in, const std::vector<CompactTables>& cts,
                                 const nd::StepHost& host, std::vector<StepResult>* out) {
  nd::IMixed2b::Ctx* c = nullptr;
  std::string e;
  auto tb0 = Clk::now();
  pushed_c_ = nullptr;
  step_pe0_ = py_epoch_();             // L4: a step that runs Python from here on keeps its sync
  defer_sync_ = false;
  Status s = mx->begin(in.batches, cts, host, &c, &e);
  tick_("mixed.begin (incl. push_state)", us_since(tb0));
  if (s == Status::BAD_ARG) {         // refused, nothing done
    err_ = e;
    return Status::BAD_ARG;
  }
  if (s != Status::OK || c == nullptr) {
    err_ = "beta0a begin: " + e;
    return Status::INTERNAL;
  }
  pc_groups_.clear();
  if (cfg_.mixed_exec && snap_ok_) {
    bool copies = false;
    for (const StepBatch& sb : in.batches) copies = copies || !sb.copies.empty();
    if (copies) {
      std::vector<CopyOp> ops;
      dedup_copies(in.batches, &ops);
      page_copy_batches(ops, &pc_groups_);
    }
  }
  if (cfg_.mixed_exec && snap_ok_) {
    bool native = false;
    const auto te = Clk::now();
    cur_in_ = &in;
    cur_host_ = &host;
    const Status xs = exec_step_(mx, c, out, &native);
    cur_in_ = nullptr;
    cur_host_ = nullptr;
    if (xs != Status::OK) return Status::INTERNAL;
    tick_(native ? "b0b.step (native)" : "b0b.refused", us_since(te));
    if (native) {
      ++n_exec_;
      return Status::OK;
    }
  }
  const std::vector<StepBatch>& rb = mx->resolved(c);
  MixedKeys k;
  if (push_now_(mx, c, &e) != Status::OK) {   // the Python body reads hist / prev-of / sets
    err_ = "beta0a push_now: " + e;
    return Status::INTERNAL;
  }
  tb0 = Clk::now();
  ++own_py_;
  ++pybody_;
  flush_touches_("beta0a");   // the hits' LRU / _busy touches before this body's own bucket() calls
  const Status bs = mixed_body_(in, cts, host, rb, mx->tables(c), &k);
  tick_("b0a.body (Python)", us_since(tb0));
  if (bs != Status::OK) {
    err_ = "beta0a: the Python mixed body failed (see 'core2b mixed body' above)";
    return Status::INTERNAL;
  }
  if (mx->prepare_head(c, &e) != Status::OK) {
    err_ = "beta0a prepare_head: " + e;
    return Status::INTERNAL;
  }
  for (int r = 0; r < (int)rb.size(); ++r) {
    FprEhMixedBucketRank t;
    std::memset(&t, 0, sizeof(t));
    t.struct_size = (int32_t)sizeof(t);
    if (h_->fill_mixed == nullptr || h_->fill_mixed(h_->ctx, k.r_t, FPR_EH_MIXED_T, k.key_t.c_str(), r, &t) != 0) {
      err_ = "beta0a fill_mixed(" + k.key_t + ", rank " + std::to_string(r) + "): " +
             (h_->last_error ? std::string(h_->last_error(h_->ctx)) : std::string());
      return Status::INTERNAL;
    }
    if (!fpr_eh_mixed_current(h_, &t)) {
      err_ = "beta0a: bucket T view " + k.key_t + " stale (generation) right after ensure_mixed";
      return Status::INTERNAL;
    }
    StepBatch tb;
    if (tail_batch(rb[r], cfg_.tail_bound, SLIDING_WINDOW, &tb, &e) != Status::OK) {
      err_ = "beta0a tail_batch rank " + std::to_string(r) + ": " + e;
      return Status::INTERNAL;
    }
    if (mx->h1_launch(c, r, t, tb, &e) != Status::OK) {
      err_ = "beta0a h1_launch rank " + std::to_string(r) + ": " + e;
      return Status::INTERNAL;
    }
  }
  tb0 = Clk::now();
  ++own_py_;
  const Status ps = mixed_post_();
  tick_("b0a.post (Python)", us_since(tb0));
  if (ps != Status::OK) {
    err_ = "beta0a: the Python mixed post failed (see 'core2b mixed post' above)";
    return Status::INTERNAL;
  }
  if (mx->finish(c, out, &e) != Status::OK) {
    err_ = "beta0a finish: " + e;
    return Status::INTERNAL;
  }
  return Status::OK;
}

const FprEhMixedBucketRank* NativeWorker::view_(char role, int r) const {
  const std::vector<FprEhMixedBucketRank>& v = role == 'A' ? views_a_ : views_t_;
  return r >= 0 && r < (int)v.size() ? &v[(size_t)r] : nullptr;
}

void NativeWorker::set_mixed_exec(const p0::MixedSnap& snap, MixedEnsureFn ensure, MixedPrevFn prev, MixedSyncFn sync) {
  using K = p0::MixedOpKind;
  snap_ = snap;
  snap_ok_ = true;
  ensure_ = std::move(ensure);
  prev_ = std::move(prev);
  sync_ = std::move(sync);
  // the page copies: checked in phase 1a (nothing issued), run first in phase 2 (after begin's drain)
  // the handover's layout must be the one this worker was compiled against: the struct grows by appended slices, and a
  // shim / exporter built from an older engine_handover.h publishes a shorter struct (nd checks it against ITS header at
  // init, not against core2b's). A mismatch refuses every native step
  issue_.add_prepare_plan("handover", [this](const p0::MixedPlan&, const MixedStepIn&) -> std::string {
    if (h_ == nullptr) return "no handover";
    if (h_->struct_size != (int32_t)sizeof(FprEngineHandover) || h_->version != FPR_EH_VERSION)
      return "struct_size " + std::to_string(h_->struct_size) + " / version " + std::to_string(h_->version) + " != core2b's " +
             std::to_string(sizeof(FprEngineHandover)) + " / " + std::to_string(FPR_EH_VERSION);
    return "";
  });
  issue_.add_prepare_plan("copies", [this](const p0::MixedPlan&, const MixedStepIn&) -> std::string {
    std::string w;
    if (!pc_groups_.empty() && !cur_mx_->page_copies_ok(pc_groups_, &w)) return w.empty() ? "page copies refused" : w;
    return "";
  });
  // a step that admits / drops requests needs Python's hooks.on_ctx (eng.hist / samp_ctx / prev_seq): refuse it natively
  // when no callback is wired
  issue_.add_prepare_plan("on_ctx", [this](const p0::MixedPlan&, const MixedStepIn&) -> std::string {
    if (cur_host_ != nullptr && (!cur_host_->fresh.empty() || !cur_host_->dropped.empty()) && !mixed_onctx_ && !lazy_on_())
      return "admission / drop step without an on_ctx callback";
    return "";
  });
  issue_.set_begin([this](const p0::MixedPlan&, std::string* err) -> Status {
    // phase 2's first work (every refusal point is behind us): D13._fallback's order drain -> push -> on_ctx -> step, i.e.
    // nd's push (lazy: now), then Python's on_ctx for this step's admissions / drops, exactly once
    if (lazy_on_() && cur_host_ != nullptr && cur_in_ != nullptr) {   // on_ctx lazy: queue, no push, no Python
      lazy_queue_(*cur_host_);
      if (cfg_.onctx_lazy_check && mixed_onctx_ && (!cur_host_->fresh.empty() || !cur_host_->dropped.empty())) {
        ++own_py_;                    // check mode: the eager on_ctx into Python's SHADOW (no push: the shadow needs none)
        if (mixed_onctx_(*cur_in_, *cur_host_) != Status::OK) {
          *err = "on_ctx (lazy check shadow) failed";
          return Status::INTERNAL;
        }
      }
    } else if (cur_host_ != nullptr && cur_in_ != nullptr && (!cur_host_->fresh.empty() || !cur_host_->dropped.empty())) {
      if (push_now_(cur_mx_, cur_c_, err) != Status::OK) return Status::INTERNAL;
      const auto t0 = Clk::now();
      const Status s = onctx_call_(*cur_in_, *cur_host_);
      tick_("b0b.on_ctx (Python)", us_since(t0));
      if (s != Status::OK) {
        *err = "on_ctx failed (see 'core2b mixed on_ctx' above)";
        return Status::INTERNAL;
      }
    }
    if (pc_groups_.empty()) return Status::OK;
    return cur_mx_->page_copies(cur_c_, pc_groups_, err) == Status::OK ? Status::OK : Status::INTERNAL;
  });

  // the step's views (plan only: before tb_tail moves): ensure_mixed at the step boundary, then fill_mixed per role / rank
  issue_.add_prepare_plan("views", [this](const p0::MixedPlan& P, const MixedStepIn&) -> std::string {
    if (h_ == nullptr || h_->fill_mixed == nullptr) return "no handover fill_mixed";
    if (!ensure_) return "no ensure_mixed";
    std::string w;
    // L3b: skip the Python ensure when nothing can have changed since the last one: the same R_A / R_T, no Python job
    // since (bucket creation / eviction / exporter refreshes all run in Python) and the exporter's buckets_epoch unchanged.
    // fill_mixed + the generation check below still run every step (a stale view refuses)
    const int64_t RT0 = P.T.empty() ? 0 : P.RT;
    const uint64_t ep = fpr_eh_buckets_epoch(h_);
    if (ens_native_ < 0) {   // P1: PHYS_CXX_ENS_NATIVE (needs both onebk flags: every prod key a pinned view)
      const char* en = std::getenv("PHYS_CXX_ENS_NATIVE");
      const char* ec = std::getenv("PHYS_CXX_ENS_CHECK");
      const char* o1 = std::getenv("ENG_MIX_ONE_BUCKET");
      const char* o2 = std::getenv("ENG_MIX_ONE_BUCKET_T");
      const bool on = en != nullptr && std::strcmp(en, "1") == 0;
      const bool onebk = o1 != nullptr && o2 != nullptr && std::strcmp(o1, "1") == 0 && std::strcmp(o2, "1") == 0;
      ens_native_ = (on && onebk) ? ((ec != nullptr && std::strcmp(ec, "1") == 0) ? 2 : 1) : 0;
      std::fprintf(stderr, "core2b P1: native ensure hit %s%s (PHYS_CXX_ENS_NATIVE %s, onebk A+T %s)\n",
                   ens_native_ ? "on" : "off", ens_native_ == 2 ? ", check on" : "", on ? "1" : "0", onebk ? "on" : "off");
    }
    bool p1_hit = false, p1_check = false;
    std::vector<std::vector<FprEhMixedBucketRank>> p1_views;   // (check mode) the cached-hit views, for the re-compare
    if (!(ens_valid_ && P.RA == ens_ra_ && RT0 == ens_rt_ && py_epoch_() == ens_py_ && ep == ens_ep_) && ens_native_ > 0) {
      const char* why = nullptr;
      for (char role : {'A', 'T'}) {
        if (role == 'T' && RT0 == 0) continue;
        const int64_t R = role == 'A' ? P.RA : RT0;
        const std::string key = std::to_string(R) + "|prod|" + role;
        auto it = ens_cache_.find(key);
        if (it == ens_cache_.end()) { why = "nocache"; break; }
        if (it->second.body != pybody_) { why = "body"; break; }
        if (it->second.bep != ep) { why = "bep"; break; }
        std::vector<FprEhMixedBucketRank> vv;
        std::string w2;
        if (!ens_fill_(R, role, &vv, &w2)) { why = "fill"; break; }
        bool same = vv.size() == it->second.sig.size();
        for (size_t r = 0; same && r < vv.size(); ++r) same = view_sig_(vv[r]) == it->second.sig[r];
        if (!same) { why = "sig"; break; }
        p1_views.push_back(std::move(vv));
      }
      // the touch word is (R_A << 24) | R_T: refuse the hit (Python ensure instead) for keys outside 24 / 39 bits, never alias
      if (why == nullptr && (RT0 < 0 || RT0 >= (int64_t(1) << 24) || P.RA < 0 || P.RA >= (int64_t(1) << 39))) {
        why = "touch24";
        if (++n_touch_range_ <= 5)
          std::fprintf(stderr, "core2b P1c: touch key out of range (RA %lld, RT %lld): native hit refused\n", (long long)P.RA, (long long)RT0);
      }
      if (why == nullptr) {
        if (ens_native_ == 2) p1_check = true;          // (check: Python replays this step's touch itself)
        else { p1_hit = true; touch_log_.emplace_back(P.RA, RT0); }
      } else {
        tick_((std::string("c128a ens miss ") + why).c_str(), 0.0);
      }
    }
    if (ens_valid_ && P.RA == ens_ra_ && RT0 == ens_rt_ && py_epoch_() == ens_py_ && ep == ens_ep_) {
      tick_("b0b.ensure skipped", 0.0);
    } else if (p1_hit) {
      tick_("b0b.ensure native hit", 0.0);   // (D5 stats + the touch log ride the next Python ensure)
      // bound the log: a long all-native stretch would otherwise replay O(stretch) touches in one later
      // Python call (a host stall at a random step).
      // this step's views are filled below, after it.
      if (touch_cap_ == 0) {   // (PHYS_CXX_TOUCH_CAP: tests only; default kTouchCap)
        const char* tc = std::getenv("PHYS_CXX_TOUCH_CAP");
        const long v = tc != nullptr ? std::strtol(tc, nullptr, 10) : 0;
        touch_cap_ = v > 0 ? (size_t)v : kTouchCap;
      }
      if (touch_log_.size() >= touch_cap_) flush_touches_("cap");
    } else {
      ens_valid_ = false;
      if (push_now_(cur_mx_, cur_c_, &w) != Status::OK) return "push_now: " + w;   // (before any Python of the step)
      const auto te = Clk::now();
      const uint64_t pe0 = py_epoch_();
      ++own_py_;
      std::map<std::string, int64_t> st_ = stats_pending_;   // P1: the touch log (replayed first, in order) + check flag
      for (size_t i = 0; i < touch_log_.size(); ++i) {
        char k_[32];
        std::snprintf(k_, sizeof k_, "__touch_%09zu", i);
        st_[k_] = (touch_log_[i].first << 24) | touch_log_[i].second;
      }
      if (p1_check) st_["__check"] = 1;
      const Status es = ensure_(P.RA, RT0, st_, &w);
      tick_(p1_check ? "b0b.ensure (Python, P1 check)" : "b0b.ensure (Python)", us_since(te));
      if (es != Status::OK) return "ensure_mixed: " + w;
      touch_log_.clear();
      stats_pending_.clear();           // Python added them to Engine.stats (D5)
      {   // P1: cache every key of this step at its post-ensure export (+ check mode's compare)
        const uint64_t ep2 = fpr_eh_buckets_epoch(h_);
        size_t vi = 0;
        for (char role : {'A', 'T'}) {
          if (role == 'T' && RT0 == 0) continue;
          const int64_t R = role == 'A' ? P.RA : RT0;
          const std::string key = std::to_string(R) + "|prod|" + role;
          std::vector<FprEhMixedBucketRank> vv;
          std::string w2;
          if (!ens_fill_(R, role, &vv, &w2)) { ens_cache_.erase(key); ++vi; continue; }
          EnsCacheE e;
          e.body = pybody_;
          e.bep = ep2;
          for (const auto& v : vv) e.sig.push_back(view_sig_(v));
          if (p1_check && vi < p1_views.size()) {
            bool same = p1_views[vi].size() == vv.size() && ep2 == ep;
            for (size_t r = 0; same && r < vv.size(); ++r) same = view_sig_(p1_views[vi][r]) == e.sig[r];
            if (same) tick_("ens check ok", 0.0);
            else {
              tick_("ens check CHANGED", 0.0);
              if (++n_ens_changed_ <= 20)
                std::fprintf(stderr, "core2b ENS check: %s export changed under a would-be native hit (bep %llu -> %llu)\n",
                             key.c_str(), (unsigned long long)ep, (unsigned long long)ep2);
            }
          }
          ens_cache_[key] = std::move(e);
          ++vi;
        }
      }
      ens_valid_ = true;
      ens_ra_ = P.RA;
      ens_rt_ = RT0;
      ens_py_ = py_epoch_();            // after the ensure's own job
      ens_ep_ = fpr_eh_buckets_epoch(h_);
      // the ensure's own job keeps the guard's passed keys valid unless it created / evicted (buckets_epoch moved)
      if (guard_epoch_ == pe0 && guard_bep_ == ens_ep_) guard_epoch_ = ens_py_;
    }
    const int W = snap_.W;
    for (char role : {'A', 'T'}) {
      std::vector<FprEhMixedBucketRank>& vv = role == 'A' ? views_a_ : views_t_;
      const int64_t R = role == 'A' ? P.RA : P.RT;
      vv.assign(role == 'T' && P.T.empty() ? 0 : (size_t)W, FprEhMixedBucketRank());
      const std::string key = std::to_string(R) + "|prod|" + role;
      for (size_t r = 0; r < vv.size(); ++r) {
        FprEhMixedBucketRank& v = vv[r];
        std::memset(&v, 0, sizeof(v));
        v.struct_size = (int32_t)sizeof(v);
        if (h_->fill_mixed(h_->ctx, R, role == 'A' ? FPR_EH_MIXED_A : FPR_EH_MIXED_T, key.c_str(), (int32_t)r, &v) != 0)
          return "fill_mixed(" + key + ", rank " + std::to_string(r) + ")";
        if (!fpr_eh_mixed_current(h_, &v)) return "view " + key + " stale";
      }
    }
    return "";
  });
  // Phase 1a: the views' key tables (handover tkeys; NULL until the
  // exporter has them, e.g. si20 before Python's first l20 step creates it), tail_ix's room, the stage buildable
  libs_.set(h_);
  issue_.add_prepare_plan("t_stage", [this](const p0::MixedPlan& P, const MixedStepIn&) -> std::string {
    bool any = false;
    for (const p0::MixedOp& o : P.ops) any = any || o.kind == K::T_STAGE;
    if (!any) return "";
    std::string w;
    if (tstage_fn_ == nullptr) {
      tstage_fn_ = libs_.fn("tail_gather", "tail_gather.t_stage", (int64_t)sizeof(TStageArgs), &w);
      if (tstage_fn_ == nullptr) return "" + w;
      tstage_args_ = reinterpret_cast<int (*)(const void*, void*)>(libs_.sym("tail_gather", "fpr_t_stage_args", &w));
    }
    for (const p0::MixedOp& o : P.ops) {
      if (o.kind != K::T_STAGE) continue;
      const int r = o.rank;
      const FprEhMixedBucketRank* a = view_('A', r);
      const FprEhMixedBucketRank* t = view_('T', r);
      if (a == nullptr || t == nullptr || r >= (int)P.T.size() || r >= (int)P.nT.size()) return "no view";
      if (a->tkeys == nullptr || t->tkeys == nullptr) return "key tables not exported";
      if (a->tkeys_bytes != (int64_t)sizeof(TgKeys) || t->tkeys_bytes != (int64_t)sizeof(TgKeys)) return "key table size";
      const int64_t n = P.nT[(size_t)r];
      const p0::TailStage& g = P.T[(size_t)r].stage;
      if ((int64_t)g.ix.size() != n) return "ix size";
      const FprEhTensor& tix = h_->ranks[r].tail_ix;
      if (n > 0 && (tix.data == nullptr || tix.shape[0] < n)) return "tail_ix not allocated / too small";
      TStageArgs x{};
      x.ix = n ? static_cast<const int64_t*>(tix.data) : nullptr;
      x.n = n;
      x.l20 = g.l20 ? 1 : 0;
      x.clear = snap_.clear_tail_padding ? 1 : 0;
      x.A = static_cast<const TgKeys*>(a->tkeys);
      x.T = static_cast<const TgKeys*>(t->tkeys);
      if (tstage_args_ != nullptr) {      // FPR's own builder (host only, pure): its refusal = the launch's
        alignas(16) unsigned char out[4096];
        if (tstage_args_(&x, out) != 0) return "fpr_t_stage_args refused (rank " + std::to_string(r) + ")";
      } else {
        const std::string e = t_stage_check(*x.A, *x.T, n, g.l20, snap_.clear_tail_padding);
        if (!e.empty()) return "" + e + " (rank " + std::to_string(r) + ")";
      }
    }
    return "";
  });
  issue_.set(K::T_STAGE, [this](const p0::MixedOp& o, const p0::MixedPlan& P, std::string* err) -> Status {
    const int r = o.rank;
    const p0::TailStage& g = P.T[(size_t)r].stage;
    const int64_t n = P.nT[(size_t)r];
    if (n > 0 && cur_mx_->set_tail_ix(cur_c_, r, g.ix, err) != Status::OK) return Status::INTERNAL;
    TStageArgs x{};
    x.ix = n ? static_cast<const int64_t*>(h_->ranks[r].tail_ix.data) : nullptr;
    x.n = n;
    x.l20 = g.l20 ? 1 : 0;
    x.clear = snap_.clear_tail_padding ? 1 : 0;
    x.A = static_cast<const TgKeys*>(view_('A', r)->tkeys);
    x.T = static_cast<const TgKeys*>(view_('T', r)->tkeys);
    return cur_mx_->cabi(cur_c_, r, tstage_fn_, &x, err) == Status::OK ? Status::OK : Status::INTERNAL;
  });
  // NEW_SCHEDS: Engine._new_scheds' fresh FlashMLA plans =
  // fpr_attn_reset_plans(rank, the view's plan slots) on every rank; host only (the driver's per-rank plan state: no device
  // op, no nd), between spans. Phase 1a: the symbol and every view's plan_slots (handover slice 5b, int64) exported
  issue_.add_prepare_plan("new_scheds", [this](const p0::MixedPlan& P, const MixedStepIn&) -> std::string {
    bool any = false;
    for (const p0::MixedOp& o : P.ops) any = any || o.kind == K::NEW_SCHEDS;
    if (!any) return "";
    if (reset_plans_ == nullptr) {
      std::string w;
      reset_plans_ = reinterpret_cast<int (*)(int, const int64_t*, int)>(libs_.sym("mixed_cxx", "fpr_attn_reset_plans", &w));
      if (reset_plans_ == nullptr) return "fpr_attn_reset_plans: " + (w.empty() ? std::string("not exported") : w);
    }
    for (const p0::MixedOp& o : P.ops) {
      if (o.kind != K::NEW_SCHEDS) continue;
      for (int r = 0; r < snap_.W; ++r) {
        const FprEhMixedBucketRank* v = view_(o.role, r);
        if (v == nullptr || v->plan_slots == nullptr || v->n_plan_slots < 0)
          return std::string("plan slots not exported (") + o.role + " rank " + std::to_string(r) + ")";
      }
    }
    return "";
  });
  issue_.set(K::NEW_SCHEDS, [this](const p0::MixedOp& o, const p0::MixedPlan&, std::string* err) -> Status {
    for (int r = 0; r < snap_.W; ++r) {
      const FprEhMixedBucketRank* v = view_(o.role, r);
      const int rc = reset_plans_(r, v->plan_slots, v->n_plan_slots);
      if (rc != 0) {
        *err = "fpr_attn_reset_plans rank " + std::to_string(r) + " rc " + std::to_string(rc);
        return Status::INTERNAL;
      }
    }
    return Status::OK;
  });
  issue_.set(K::UPLOAD, [this](const p0::MixedOp& o, const p0::MixedPlan& P, std::string* err) {
    const FprEhMixedBucketRank* v = view_(o.role, o.rank);
    const std::vector<p0::BucketRank>& br = o.role == 'A' ? P.A : P.T;
    if (v == nullptr || o.rank >= (int)br.size()) { *err = "no view"; return Status::INTERNAL; }
    return cur_mx_->upload(cur_c_, o.rank, *v, br[(size_t)o.rank].up, err);
  });
  issue_.set(K::SET_VIDX, [this](const p0::MixedOp& o, const p0::MixedPlan& P, std::string* err) {
    for (int r = 0; r < snap_.W; ++r) {
      const FprEhMixedBucketRank* v = view_(o.role, r);
      if (v == nullptr) { *err = "no view"; return Status::INTERNAL; }
      if (cur_mx_->set_vidx(cur_c_, r, *v, o.role == 'A' ? P.vA : P.vT, err) != Status::OK) return Status::INTERNAL;
    }
    return Status::OK;
  });
  issue_.set(K::PREPARE_HEAD, [this](const p0::MixedOp&, const p0::MixedPlan&, std::string* err) {
    return cur_mx_->prepare_head(cur_c_, err);
  });
  issue_.set(K::HEAD_EAGER, [this](const p0::MixedOp&, const p0::MixedPlan& P, std::string* err) {
    for (int r = 0; r < snap_.W; ++r) {
      const FprEhMixedBucketRank* v = view_('T', r);
      if (v == nullptr || r >= (int)P.T.size()) { *err = "no T view"; return Status::INTERNAL; }
      if (cur_mx_->h1_launch(cur_c_, r, *v, P.T[(size_t)r].tb, err) != Status::OK) return Status::INTERNAL;
    }
    return Status::OK;
  });
  // here: the head results come from nd's finish after the step; the memory counters are native
  issue_.set(K::HEAD_RESULT, [](const p0::MixedOp&, const p0::MixedPlan&, std::string*) { return Status::OK; });
  issue_.set(K::MEM_AFTER_STEP, [](const p0::MixedOp&, const p0::MixedPlan&, std::string*) { return Status::OK; });
  own_gen_hr_ = issue_.gen(K::HEAD_RESULT), own_gen_mem_ = issue_.gen(K::MEM_AFTER_STEP);
  issue_.set(K::SYNC, [this](const p0::MixedOp& o, const p0::MixedPlan& P, std::string* err) {
    // L4: the step's LAST sync with only host no-ops after it and no Python since before begin: deferred to the drain
    if (cfg_.mixed_async && !sync_ && py_epoch_() == step_pe0_ && sync_deferrable_(o, P)) {
      defer_sync_ = true;
      return Status::OK;
    }
    return sync_ ? sync_(cur_mx_, cur_c_, err) : cur_mx_->sync(cur_c_, err);
  });
  // the owners' op files (core2b/owner_<name>.cpp, mixed_owner.h): their hooks run after the ones above (views first)
  MixedOwnerRegistry reg(&issue_, [this]() {
    MixedOwnerCtx x;
    x.mx = cur_mx_;
    x.c = cur_c_;
    x.h = h_;
    x.snap = &snap_;
    x.view = [this](char role, int r) { return view_(role, r); };
    x.libs = &libs_;
    return x;
  });
  owners_ = register_mixed_owners(reg);
  if (!owners_.empty()) std::fprintf(stderr, "core2b beta0b: mixed owners: %s\n", owners_.c_str());
}

Status NativeWorker::exec_step_(nd::IMixed2b* mx, nd::IMixed2b::Ctx* c, std::vector<StepResult>* out, bool* native) {
  *native = false;
  MixedStepIn in;
  in.snap = &snap_;
  in.rb = &mx->resolved(c);
  in.tables = &mx->tables(c);
  if (prev_)
    in.prev = [this, mx, c](const Rid& rid, p0::PrevSeq* o) { return prev_(mx, c, rid, o); };
  else                                // the prev-of map Hooks._device_tokens_v2 reads at this step
    in.prev = [mx, c](const Rid& rid, p0::PrevSeq* o) {
      int32_t rank = -1;
      int64_t seq = -1;
      if (!mx->prev_of(c, rid, &rank, &seq)) return false;
      o->rank = rank;
      o->seq_index = seq;
      return true;
    };
  in.tb_tail = h_ != nullptr ? reinterpret_cast<ITables*>(h_->tb_tail) : nullptr;
  in.mixed_steps_before = mixed_steps_;
  // Only when tb_tail's creator (fpr_kvtables.so) has the same ITables
  // vtable: the virtuals are appended LAST, so an
  // older creator's vtable ends before them. Unwired (or save_state() == nullptr): a phase-1b refusal is FATAL (run_mixed)
#if defined(FPR_TABLES_ABI) && FPR_TABLES_ABI >= 2
  if (in.tb_tail != nullptr && h_->tables_abi == FPR_TABLES_ABI) {
    ITables* t = in.tb_tail;
    in.tb_save = [t]() { return t->save_state(); };
    in.tb_restore = [t](const std::shared_ptr<const void>& s) { t->restore_state(s); };
  }
#endif
  // PHYS_CXX_MIXED_EXEC_OFF_FILE: while the file exists, refuse natively
  {
    static const char* off_file = std::getenv("PHYS_CXX_MIXED_EXEC_OFF_FILE");
    struct stat sb_;
    if (off_file != nullptr && *off_file && ::stat(off_file, &sb_) == 0) {
      note_refusal("beta0b: toggle off (PHYS_CXX_MIXED_EXEC_OFF_FILE)");
      return Status::OK;
    }
  }
  p0::MixedPlan plan;
  std::string why;
  MixedExec ex;
  cur_mx_ = mx;
  cur_c_ = c;
  // NEVER hold ex_mu_ across run_mixed: its plan hooks call mensure, which waits
  // for Python's thread to serve the ensure job, while Python's thread may be blocked on ex_mu_ in timers() /
  // mixed_exec_stats() (core2b_proc's PHYS_CXX_IO_LOG worker_stats). Only this (core) thread runs run_mixed: it accumulates
  // into a local copy and publishes it + a snapshot of the issue timers under the lock afterwards.
  MixedExecStats st_local;
  {
    std::lock_guard<std::mutex> g(ex_mu_);
    st_local = ex_stats_;
  }
  ex = run_mixed(in, &issue_, &plan, &why, &st_local);
  {
    std::lock_guard<std::mutex> g(ex_mu_);
    ex_stats_ = st_local;
    issue_times_ = issue_.times();
  }
  cur_mx_ = nullptr;
  cur_c_ = nullptr;
  if (ex == MixedExec::REFUSED) {
    note_refusal("beta0b: " + why);
    return Status::OK;
  }
  if (ex == MixedExec::FATAL) {
    err_ = "beta0b " + why;
    return Status::INTERNAL;
  }
  std::string e;
  if (defer_sync_) {                  // L4: the SYNC handler skipped the step's last sync: finish at the drain
    defer_sync_ = false;
    pend_.mx = mx, pend_.c = c, pend_.h = nullptr;   // dispatch_ attaches the step's handle
  } else if (mx->finish(c, out, &e) != Status::OK) {
    err_ = "beta0b finish: " + e;
    return Status::INTERNAL;
  }
  ++mixed_steps_;
  for (const auto& kv : plan.stats) stats_pending_[kv.first] += kv.second;   // D5: to Engine.stats at the next ensure
  *native = true;
  return Status::OK;
}

void NativeWorker::set_mixed_driver(const std::string& driver_so, MoeVariantFn moe, const SpanEngineFlags& flags) {
  using K = p0::MixedOpKind;
  std::string e;
  if (!drv_.load(driver_so, &e)) {
    std::fprintf(stderr, "core2b beta0b: driver issue side off (%s): SET_STEP / ISSUE / JOIN_MIX refuse\n", e.c_str());
    return;
  }
  moe_ = std::move(moe);
  span_flags_ = flags;
  // phase 1 (plan only, after the views): every segment's program + moe variants; a refusal here leaves nothing issued
  issue_.add_prepare_plan("span", [this](const p0::MixedPlan& P, const MixedStepIn&) -> std::string {
    spans_.clear();
    variants_.clear();
    const int W = snap_.W;
    int32_t kinds[8], kc[8];            // owner kind ids; <= 0 = not registered (a zeroed handover field is 0)
    for (int q = 0; q < 8; ++q) kinds[q] = h_->span_kinds[q] > 0 ? h_->span_kinds[q] : -1;
    for (int q = 0; q < 8; ++q) kc[q] = h_->span_kinds_cp[q] > 0 ? h_->span_kinds_cp[q] : -1;
    SpanKinds k;
    k.embed = kinds[0]; k.engram_bd = kinds[1]; k.attn_a = kinds[2]; k.idx = kinds[3]; k.attn_b = kinds[4]; k.moe = kinds[5];
    k.attn_bi = kc[0]; k.attn_bo = kc[1]; k.idx_c = kc[2]; k.idx_s = kc[3];                  // slice 6 (CP steps)
    k.cp_x1a_send = kc[4]; k.cp_x1a_recv = kc[5]; k.cp_x2_send = kc[6]; k.cp_x2_recv = kc[7];
    std::string seg_prev;
    { std::lock_guard<std::mutex> g(seg_mu_); seg_prev = last_seg_; }   // shared with Python's E._agrs_st_last
    for (const p0::MixedOp& o : P.ops) {
      if (o.kind == K::ISSUE_MIXED || o.kind == K::L20_HEAD || o.kind == K::L20_TAIL) return "segment kind not native";
      if (o.kind != K::ISSUE) continue;
      const bool T = o.role == 'T';
      const int64_t R = T ? P.RT : P.RA;
      const p0::VidxOut& V = T ? P.vT : P.vA;
      SpanSegment sg;
      for (int32_t L = o.lay0; o.nlay && L <= o.lay1; ++L) sg.layers.push_back(L);
      sg.final_ = o.final_; sg.embed = o.embed; sg.R = R; sg.cp_on = P.cp;
      sg.l2pf = span_flags_.l2pf; sg.nancheck = span_flags_.nancheck; sg.prof_pieces = span_flags_.prof_pieces;
      sg.dbg_sync = snap_.dbg_sync; sg.tp_on = span_flags_.tp_on; sg.mega_on = span_flags_.mega_on;
      sg.router_pack3 = span_flags_.router_pack3; sg.agrs_on = span_flags_.agrs_on; sg.agrs_ag32 = span_flags_.agrs_ag32;
      sg.agrs_agce = span_flags_.agrs_agce; sg.agrs_min_r = span_flags_.agrs_min_r;
      sg.index_layers = snap_.index_layers; sg.idx_on = snap_.idx;
      const std::string seg = std::string(1, o.role) + "|" + std::to_string(R);
      std::vector<SpanRankHandles> h((size_t)W);
      std::vector<int32_t> var((size_t)W);
      sg.ce.assign((size_t)W, 0); sg.rsf.assign((size_t)W, 0); sg.fl_rsf.assign((size_t)W, 0); sg.pf.assign((size_t)W, 0);
      const std::vector<p0::BucketRank>& BR = T ? P.T : P.A;
      for (int r = 0; r < W; ++r) {
        const FprEhMixedBucketRank* v = view_(o.role, r);
        if (v == nullptr || !v->span.built) return "span table not built (" + seg + ")";
        const FprEhSpanTable& t = v->span;
        if (t.n_layers != (int32_t)sg.layers.size()) return "span table layers (" + seg + ")";
        h[(size_t)r].embed = t.embed; h[(size_t)r].final_ = t.final_;
        for (int q = 0; q < t.n_layers; ++q) {
          const FprEhSpanLayer& x = t.layers[q];
          if (x.layer != sg.layers[(size_t)q]) return "span table layer order (" + seg + ")";   // handles pair by index
          if (x.engram_bd >= 0 && std::find(sg.engram_layers.begin(), sg.engram_layers.end(), x.layer) == sg.engram_layers.end())
            sg.engram_layers.push_back(x.layer);
          h[(size_t)r].layers.push_back(SpanLayerHandles{x.layer, x.bd0, x.engram_bd, x.attn_a, x.idx, x.attn_b, x.bd1,
                                                         x.router, x.router_ce, x.ag, x.moe, x.rs, x.rs_fl, x.moe_rsf,
                                                         x.attn_b_pf});
        }
        sg.engram_on = !sg.engram_layers.empty();
        sg.pf[(size_t)r] = r < (int)BR.size() && BR[(size_t)r].has_pf_ab;   // ws["pf_ab"]: attn_b_pf (-1 -> refusal)
        if (moe_) {
          var[(size_t)r] = moe_(o.role, R, V.vcount, V.ndec, r);
        } else {
          const p0::MoeSnap* ms = h_->moe_snap_bytes == (int64_t)sizeof(p0::MoeSnap)
                                      ? static_cast<const p0::MoeSnap*>(h_->moe_snap) : nullptr;
          if (ms == nullptr || ms->version != p0::kMoeSnapVersion) return "moe decide unavailable (handover moe_snap)";
          if (v->moe_bucket == nullptr || v->moe_bucket_bytes != (int64_t)sizeof(p0::MoeBucket))
            return "moe bucket not exported (" + seg + ")";
          const p0::MoeBucket& mb = *static_cast<const p0::MoeBucket*>(v->moe_bucket);
          var[(size_t)r] = p0::moe_decide(*ms, mb, V.vcount, V.ndec);
          if (var[(size_t)r] >= 0 && p0::moe_bucket_refuses(mb, var[(size_t)r])) return "moe: fork variant on an A8sh layer";
        }
        if (var[(size_t)r] < 0) return "moe variant -1 (Python moe)";
        sg.rsf[(size_t)r] = var[(size_t)r] == p0::MOE_PROD_RSF;
        sg.fl_rsf[(size_t)r] = t.fl_rsf;
        sg.ce[(size_t)r] = span_flags_.agrs_on && span_flags_.agrs_agce && R >= span_flags_.agrs_min_r && seg_prev == seg;
      }
      std::vector<std::vector<SpanEntry>> prog;
      const std::string w = build_span(sg, h, k, &prog);
      if (!w.empty()) return w;
      spans_.push_back(std::move(prog));
      variants_.push_back(std::move(var));
      seg_prev = seg;
    }
    span_i_ = sp_i_ = 0;
    return "";
  });
  issue_.set(K::SET_STEP, [this](const p0::MixedOp& o, const p0::MixedPlan& P, std::string* err) {
    if (sp_i_ >= variants_.size()) { *err = "no segment for set_step"; return Status::INTERNAL; }
    const p0::VidxOut& V = o.role == 'T' ? P.vT : P.vA;
    for (int r = 0; r < snap_.W; ++r) {
      const DrvStepParams sp = step_params(o.role == 'T', V.vcount, variants_[sp_i_][(size_t)r]);
      if (drv_.set_step_params(r, sp) != 0) { *err = "set_step_params rank " + std::to_string(r); return Status::INTERNAL; }
    }
    ++sp_i_;
    return Status::OK;
  });
  issue_.set(K::ISSUE, [this](const p0::MixedOp& o, const p0::MixedPlan& P, std::string* err) {
    if (span_i_ >= spans_.size()) { *err = "no program for issue"; return Status::INTERNAL; }
    std::vector<int> rc;
    drv_.run_span(spans_[span_i_], &rc);
    for (size_t r = 0; r < rc.size(); ++r)
      if (rc[r] != 0) { *err = "span rank " + std::to_string(r) + ": " + drv_.last_error((int)r); return Status::INTERNAL; }
    {
      std::lock_guard<std::mutex> g(seg_mu_);
      last_seg_ = std::string(1, o.role) + "|" + std::to_string(o.role == 'T' ? P.RT : P.RA);
      seg_native_ = true;                                // the next Python step takes it as E._agrs_st_last
    }
    ++span_i_;
    return Status::OK;
  });
  issue_.set(K::JOIN_MIX, [this](const p0::MixedOp& o, const p0::MixedPlan&, std::string* err) {
    if (drv_.join(o.rank, h_->ranks[o.rank].step_stream) != 0) { *err = "join rank " + std::to_string(o.rank); return Status::INTERNAL; }
    return Status::OK;
  });
}

void NativeWorker::set_mixed(MixedBodyFn body, MixedPostFn post, MixedGuardFn guard, const FprEngineHandover* h) {
  mixed_body_ = std::move(body);
  mixed_post_ = std::move(post);
  mixed_guard_ = std::move(guard);
  h_ = h;
  // H3: the 'handover' prepare plan refuses every native mixed step on a struct mismatch, silently mid-serve.
  // Say it once, loudly, at wiring time
  if (h_ != nullptr && (h_->struct_size != (int32_t)sizeof(FprEngineHandover) || h_->version != FPR_EH_VERSION)) {
    std::fprintf(stderr, "core2b NativeWorker: ERROR: engine handover struct_size %d / version %d != core2b's %zu / %d: "
                 "EVERY native mixed step will refuse ('handover'): nd / exporter and core2b were built from different headers\n",
                 (int)h_->struct_size, (int)h_->version, sizeof(FprEngineHandover), (int)FPR_EH_VERSION);
    note_refusal("handover mismatch at init");
  }
#if defined(FPR_ND_API_VERSION)
  // the nd API guard: IMixed2b's vtable / call contracts must be the ones compiled here (nd and core2b build separately)
  const std::string js = nd_ ? nd_->stats_json() : std::string();
  const size_t at = js.find("\"nd_api\"");
  long long v = -1;
  if (at != std::string::npos) {
    const size_t c = js.find(':', at);
    if (c != std::string::npos) v = std::strtoll(js.c_str() + c + 1, nullptr, 10);
  }
  if (v != FPR_ND_API_VERSION) {
    std::fprintf(stderr, "core2b NativeWorker: nd_api %lld != core2b's %d: native mixed steps OFF\n", v, FPR_ND_API_VERSION);
    cfg_.mixed_native = false;
    cfg_.mixed_exec = false;
    return;
  }
#endif
#if defined(FPR_ND_LAZY_PUSH)
  if (nd_ && nd_->mixed()) {
    nd_->mixed()->set_lazy_push(true);
    lazy_push_ = true;
  }
#endif
}

Status NativeWorker::push_now_(nd::IMixed2b* mx, nd::IMixed2b::Ctx* c, std::string* err) {
#if defined(FPR_ND_LAZY_PUSH)
  if (!lazy_push_ || mx == nullptr || c == nullptr || c == pushed_c_) return Status::OK;   // once per step
  if (flush_lazy_(err) != Status::OK) return Status::INTERNAL;   // the queued admissions before J
  const auto t0 = Clk::now();
  const Status s = mx->push_now(c, err);
  tick_("mixed.push_now (Python)", us_since(t0));
  if (s == Status::OK) pushed_c_ = c;
  return s;
#else
  (void)mx, (void)c, (void)err;
  return Status::OK;                  // eager push: nd's begin pushed already
#endif
}

void NativeWorker::lazy_queue_(const nd::StepHost& host) {
  for (const auto& f : host.fresh) {
    LazyAdmission a;
    a.rid = f.rid;
    a.serial = f.serial;
    a.seed = f.sampling.seed;
    a.has_seed = f.sampling.has_seed;
    a.structured = f.structured;
    lazy_q_[f.rid] = a;
  }
  if (!host.fresh.empty()) tick_("lazy.queued (admissions; value = count)", (double)host.fresh.size());
}

Status NativeWorker::flush_lazy_(std::string* err) {   // on_ctx lazy: one replay of every queued admission
  if (lazy_q_.empty()) return Status::OK;
#if defined(FPR_ND_API_VERSION) && FPR_ND_API_VERSION >= 326
  std::vector<LzRid> rids;
  rids.reserve(lazy_q_.size());
  for (const auto& kv : lazy_q_) rids.push_back(kv.first);
  std::vector<nd::ReplayEntry> ents;
  nd_->replay_view(rids, &ents);
  std::map<LzRid, const nd::ReplayEntry*> by;
  for (const auto& e : ents) by[e.rid] = &e;
  std::vector<LazyAdmission> adm;
  adm.reserve(rids.size());
  for (const auto& rid : rids) {
    auto it = by.find(rid);
    LazyAdmission a = lazy_q_[rid];
    if (it == by.end() || !it->second->known) {   // nd does not know a live queued rid: a bookkeeping bug, loud, never silent
      if (err) *err = "on_ctx lazy: nd's replay_view does not know queued rid " + std::string(rid);
      return Status::INTERNAL;
    }
    if (a.has_seed && it->second->seed != a.seed) {
      if (err) *err = "on_ctx lazy: seed mismatch nd vs worker for rid " + std::string(rid);
      return Status::INTERNAL;
    }
    a.ids = it->second->ids;
    a.n = it->second->n;
    adm.push_back(a);
  }
  const auto t0 = Clk::now();
  const uint64_t re0 = py_epoch_();   // H3: FPR's lazy replay only writes per-request host state, like on_ctx (H2)
  const uint64_t rb0 = fpr_eh_buckets_epoch(h_);
  ++own_py_;
  const Status s = mixed_replay_(adm);
  tick_("lazy.replay (Python)", us_since(t0));
  {
    const uint64_t re1 = py_epoch_();
    const uint64_t alone = 1 + (py_epoch_fn_ ? 1 : 0);
    if (s == Status::OK && onctx_skip_ && re1 - re0 == alone && fpr_eh_buckets_epoch(h_) == rb0) {
      step_transparent_ += alone;
      if (ens_valid_ && ens_py_ == re0) ens_py_ = re1;
      tick_("h2 replay transparent", 0.0);
    } else {
      tick_("h2 replay opaque", 0.0);
    }
  }
  if (s != Status::OK) {
    if (err) *err = "on_ctx lazy replay failed (see 'core2b lazy replay' above)";
    return Status::INTERNAL;
  }
  lazy_q_.clear();
  return Status::OK;
#else
  if (err) *err = "on_ctx lazy needs nd API >= 326";
  return Status::INTERNAL;
#endif
}

void NativeWorker::tick_(const char* k, double us) {
  std::lock_guard<std::mutex> g(tm_mu_);
  timers_[k].add(us);
}

std::map<std::string, StepTimer> NativeWorker::timers() const {
  std::map<std::string, StepTimer> t;
  {
    std::lock_guard<std::mutex> g(tm_mu_);
    t = timers_;
  }
  std::lock_guard<std::mutex> g(ex_mu_);
  for (const auto& kv : issue_times_) t[kv.first] = kv.second;   // the snapshot (issue_ itself is the core thread's)
  for (const auto& kv : ex_stats_.times) t[kv.first] = kv.second;
  return t;
}

MixedExecStats NativeWorker::mixed_exec_stats() const {
  std::lock_guard<std::mutex> g(ex_mu_);
  return ex_stats_;
}

bool NativeWorker::h1_guard_ok_(nd::IMixed2b* mx, const std::vector<StepBatch>& b, const nd::StepHost& host,
                               std::string* why) {
  if (!mixed_guard_) {                // no guard wired: never replay a captured H1 graph unchecked
    *why = "H1 guard missing";
    return false;
  }
#if defined(FPR_ND_API_VERSION) && FPR_ND_API_VERSION >= 325
  std::string key = mx->h1_key_for(b, host);   // "...|bm" on a step with sampled grammar masks
#else
  (void)host;
  std::string key = mx->h1_key(b);
#endif
  if (key.empty()) {
    *why = "H1 key unknown";
    return false;
  }
  std::string gw;
  // L3c: a key that passed while no Python has run since cannot have gone stale (the guarded objects change only in
  // Python; the exporter retires stale keys at its Python boundaries)
  // (+ the exporter's buckets_epoch: a bucket created / evicted since may hold what a captured head reads)
  const uint64_t bep = fpr_eh_buckets_epoch(h_);
  // H3: verified == epoch and the buckets unchanged since that clean full scan -> the scan's
  // result still holds (can_head already required the key's H1 graph published): no Python call. Else today's path below
  if (h1_ep_ != nullptr) {
    const uint64_t v = __atomic_load_n(&h1_ep_[1], __ATOMIC_ACQUIRE);
    const uint64_t e = __atomic_load_n(&h1_ep_[0], __ATOMIC_ACQUIRE);
    const uint64_t vb = __atomic_load_n(&h1_ep_[2], __ATOMIC_ACQUIRE);
    if (v != 0 && e == v && vb == bep) {
      if (!h1_verify_) {
        tick_("mixed.h1_guard native (epoch)", 0.0);
        return true;
      }
      std::string gw2;                  // VERIFY: the Python guard anyway, with the Python path's own bookkeeping
      const uint64_t pv0 = py_epoch_();
      ++own_py_;
      const bool ok2 = mixed_guard_(key, &gw2);
      const uint64_t pv1 = py_epoch_();
      if (ens_valid_ && ens_py_ == pv0) ens_py_ = pv1;
      guard_epoch_ = pv1;
      guard_bep_ = bep;
      tick_(ok2 ? "mixed.h1_guard epoch verify ok" : "mixed.h1_guard epoch WOULD-SKIP-BUT-STALE", 0.0);
      if (!ok2) {
        *why = "H1 stale: " + (gw2.empty() ? key : gw2);
        return false;
      }
      guard_keys_.insert(key);
      return true;
    }
  }
  if (py_epoch_() != guard_epoch_ || bep != guard_bep_) guard_keys_.clear();
  if (guard_keys_.count(key)) {
    tick_("mixed.h1_guard skipped", 0.0);
    return true;
  }
  const auto tg = Clk::now();
  const uint64_t pe0 = py_epoch_();
  ++own_py_;
  const bool gok = mixed_guard_(key, &gw);
  tick_("mixed.h1_guard (Python)", us_since(tg));
  const uint64_t pe1 = py_epoch_();
  if (ens_valid_ && ens_py_ == pe0) ens_py_ = pe1;   // the guard's own job leaves the mixed views as they were
  guard_epoch_ = pe1;
  guard_bep_ = bep;
  if (gok) guard_keys_.insert(key);
  if (!gok) {
    *why = "H1 stale: " + (gw.empty() ? key : gw);
    return false;
  }
  return true;
}

void NativeWorker::note_refusal(const std::string& why) {
  std::string k;
  const std::string b0 = "beta0b: ";
  if (why.compare(0, b0.size(), b0) == 0) {
    const std::string r = why.substr(b0.size());
    k = "beta0b " + r.substr(0, std::min(r.find("("), (size_t)56));
    while (!k.empty() && (k.back() == ' ' || k.back() == ':')) k.pop_back();
    std::lock_guard<std::mutex> g(refusals_mu_);
    ++refusals_[k];
    return;
  }
  k = why.substr(0, std::min(why.find_first_of(":(0123456789"), (size_t)48));
  while (!k.empty() && k.back() == ' ') k.pop_back();
  std::lock_guard<std::mutex> g(refusals_mu_);
  ++refusals_[k.empty() ? why.substr(0, 48) : k];
}

// the decode route's embedded steps: note_refusal's class rule (text up to the first ':' / '(' / digit, at
// most 48 chars); each class's first full text once to stderr, so a census has the raw reason
void NativeWorker::note_decode_refusal(const std::string& why) {
  std::string k = why.substr(0, std::min(why.find_first_of(":(0123456789"), (size_t)48));
  while (!k.empty() && k.back() == ' ') k.pop_back();
  if (k.empty()) k = why.empty() ? std::string("(empty reason)") : why.substr(0, 48);
  bool first = false;
  {
    std::lock_guard<std::mutex> g(refusals_mu_);
    first = ++decode_refusals_[k] == 1;
  }
  if (first) std::fprintf(stderr, "core2b decode refusal (first): %s\n", why.c_str());
}

std::map<std::string, int64_t> NativeWorker::decode_refusals() const {
  std::lock_guard<std::mutex> g(refusals_mu_);
  return decode_refusals_;
}

// H3 the H1 epoch block; PHYS_CXX_H1_EPOCH_VERIFY read here
void NativeWorker::set_h1_epoch(uint64_t addr) {
  const uint64_t* p = reinterpret_cast<const uint64_t*>(addr);
  h1_ep_ = (p != nullptr && __atomic_load_n(&p[3], __ATOMIC_ACQUIRE) == 0x48314550ull) ? p : nullptr;
  const char* v = std::getenv("PHYS_CXX_H1_EPOCH_VERIFY");
  h1_verify_ = v != nullptr && std::strcmp(v, "1") == 0;
  std::fprintf(stderr, "core2b H3: H1 epoch fast path %s%s\n", h1_ep_ ? "on" : "off (no block / magic)",
               h1_ep_ && h1_verify_ ? ", VERIFY on" : "");
}

std::map<std::string, int64_t> NativeWorker::mixed_refusals() const {
  std::lock_guard<std::mutex> g(refusals_mu_);
  return refusals_;
}

// L4: o is the plan's last device-ordered point: every later op is host-only (NdMixedIssue::host_only) or the worker's
// own no-op HEAD_RESULT / MEM_AFTER_STEP (no owner replaced them). CP steps (CP_DONE after the sync) stay synchronous.
bool NativeWorker::sync_deferrable_(const p0::MixedOp& o, const p0::MixedPlan& P) const {
  using K = p0::MixedOpKind;
  const p0::MixedOp* b = P.ops.data();
  const p0::MixedOp* e = b + P.ops.size();
  if (&o < b || &o >= e) return false;   // not the plan's own op (a copy): keep the sync
  if (issue_.gen(K::HEAD_RESULT) != own_gen_hr_ || issue_.gen(K::MEM_AFTER_STEP) != own_gen_mem_) return false;
  for (const p0::MixedOp* q = &o + 1; q < e; ++q)
    if (!NdMixedIssue::host_only(q->kind) && q->kind != K::HEAD_RESULT && q->kind != K::MEM_AFTER_STEP) return false;
  return true;
}

// L4: the deferred step's sync + finish (the calls exec_step_ would have made then), results onto its handle
// ---------------------------------------------------------------- P1
// the exported view as nd consumes it: the struct bytes + the arrays it points to (pointer lists / byte sizes / plan slots)
// + the tail key table bytes (rewritten in place only with a generation bump, but compared anyway: cheap)
std::string NativeWorker::view_sig_(const FprEhMixedBucketRank& v) const {
  std::string s(reinterpret_cast<const char*>(&v), sizeof(v));
  if (v.n_ptrs > 0 && v.ptrs != nullptr) s.append(reinterpret_cast<const char*>(v.ptrs), sizeof(void*) * (size_t)v.n_ptrs);
  if (v.n_ptrs > 0 && v.bytes != nullptr) s.append(reinterpret_cast<const char*>(v.bytes), sizeof(int64_t) * (size_t)v.n_ptrs);
  if (v.n_plan_slots > 0 && v.plan_slots != nullptr)
    s.append(reinterpret_cast<const char*>(v.plan_slots), sizeof(int64_t) * (size_t)v.n_plan_slots);
  if (v.tkeys != nullptr && v.tkeys_bytes > 0 && v.tkeys_bytes <= 65536)
    s.append(reinterpret_cast<const char*>(v.tkeys), (size_t)v.tkeys_bytes);
  return s;
}

// replay the native hits' pending touches (+ D5 stats) before a Python body, through ensure_ in replay-only mode (R_A -1)
void NativeWorker::flush_touches_(const char* where) {
  if (touch_log_.empty() || !ensure_) return;
  std::map<std::string, int64_t> st_ = stats_pending_;
  for (size_t i = 0; i < touch_log_.size(); ++i) {
    char k_[32];
    std::snprintf(k_, sizeof k_, "__touch_%09zu", i);
    st_[k_] = (touch_log_[i].first << 24) | touch_log_[i].second;
  }
  std::string w;
  const auto te = Clk::now();
  ++own_py_;
  const Status es = ensure_(-1, 0, st_, &w);
  tick_((std::string("b0b.touch replay before ") + where + " (Python)").c_str(), us_since(te));
  if (es == Status::OK) {
    touch_log_.clear();
    stats_pending_.clear();
  } else {
    tick_("P1b touch log pending at a body", 0.0);
    std::fprintf(stderr, "core2b P1b: touch replay before %s failed (%s): the log stays for the next Python ensure\n", where, w.c_str());
  }
}

bool NativeWorker::ens_fill_(int64_t R, char role, std::vector<FprEhMixedBucketRank>* out, std::string* why) {
  const std::string key = std::to_string(R) + "|prod|" + role;
  out->assign((size_t)snap_.W, FprEhMixedBucketRank());
  for (int r = 0; r < snap_.W; ++r) {
    FprEhMixedBucketRank& v = (*out)[(size_t)r];
    std::memset(&v, 0, sizeof(v));
    v.struct_size = (int32_t)sizeof(v);
    if (h_->fill_mixed(h_->ctx, R, role == 'A' ? FPR_EH_MIXED_A : FPR_EH_MIXED_T, key.c_str(), (int32_t)r, &v) != 0) {
      *why = "fill_mixed";
      return false;
    }
    if (!fpr_eh_mixed_current(h_, &v)) { *why = "stale"; return false; }
  }
  return true;
}

// ---------------------------------------------------------------- H2
void NativeWorker::set_mixed_onctx(MixedOnCtxFn f) {
  mixed_onctx_ = std::move(f);
  const char* sk = std::getenv("PHYS_CXX_ONCTX_SKIP");
  const char* ck = std::getenv("PHYS_CXX_ONCTX_CHECK");
  onctx_skip_ = !(sk != nullptr && std::strcmp(sk, "0") == 0);
  onctx_check_ = ck != nullptr && std::strcmp(ck, "1") == 0;
  std::fprintf(stderr, "core2b H2: on_ctx state-only skip %s, check %s\n", onctx_skip_ ? "on" : "off", onctx_check_ ? "on" : "off");
}

Status NativeWorker::onctx_call_(const StepInput& in, const nd::StepHost& host) {
  const uint64_t e0 = py_epoch_();
  const uint64_t b0 = fpr_eh_buckets_epoch(h_);
  ++own_py_;
  const Status s = mixed_onctx_(in, host);
  const uint64_t e1 = py_epoch_();
  // the on_ctx job alone: NativeWorker's own count + (with the PyExec epoch) its one job; anything more = another job ran
  const uint64_t alone = 1 + (py_epoch_fn_ ? 1 : 0);
  if (s == Status::OK && onctx_skip_ && e1 - e0 == alone && fpr_eh_buckets_epoch(h_) == b0) {
    step_transparent_ += alone;
    if (ens_valid_ && ens_py_ == e0) ens_py_ = e1;
    tick_("h2 on_ctx transparent", 0.0);
  } else {
    tick_("h2 on_ctx opaque", 0.0);
  }
  return s;
}

Status NativeWorker::drain_mixed_() {
  const PendingMixed p = pend_;
  pend_ = PendingMixed();
  if (p.c == nullptr) return Status::OK;
  NwHandle* hh = static_cast<NwHandle*>(p.h);
  std::vector<StepResult> scratch;
  std::string e;
  const auto t0 = Clk::now();
  if (p.mx->sync(p.c, &e) != Status::OK) {
    err_ = "L4 deferred sync: " + e;
    return Status::INTERNAL;
  }
  tick_("mixed.drain sync (L4)", us_since(t0));
  if (p.mx->finish(p.c, hh != nullptr ? &hh->results : &scratch, &e) != Status::OK) {
    err_ = "beta0b finish (L4 deferred): " + e;
    return Status::INTERNAL;
  }
  if (hh != nullptr) hh->deferred = false;
  tick_("mixed.drain (L4)", us_since(t0));
  return Status::OK;
}

Status NativeWorker::wait_(StepHandle* h, std::vector<StepResult>* out) {
  std::unique_ptr<NwHandle> own(reinterpret_cast<NwHandle*>(h));
  if (own->deferred) {                // L4: no dispatch since: its sync + finish run now
    if (pend_.h != own.get()) {
      err_ = "L4: waiting a deferred mixed step that is not the pending one";
      return Status::INTERNAL;
    }
    if (drain_mixed_() != Status::OK) return Status::INTERNAL;
  }
  if (!own->native) {
    *out = std::move(own->results);
    return Status::OK;
  }
  return nd_->wait(own->nd, out, &err_) == Status::OK ? Status::OK : Status::INTERNAL;
}

}  // namespace core2b
}  // namespace fpr
