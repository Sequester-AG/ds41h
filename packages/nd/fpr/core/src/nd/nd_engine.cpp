// nd_engine.cpp -- NativeDecode ND-2 / ND-3: nd::Engine (nd_engine.h). Each block names the
// Python it replaces (engine/d13.py D13.dispatch, engine/d13s.py Steady, engine.py hooks / prev_of, kernels/head/hsd.py).
// Status: v0 DRAFT, CPU-compiled only; gate = the in-engine harness arms ENG_D13_NATIVE=2 vs 0.
#include "nd_engine.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "../../../engine/mixed_cxx/fpr_cabi.h"
#include "../../../kernels/d13s/hsdprep_cabi.h"
#include "../../../kernels/pagecopy/pagecopy_cabi.h"
#include "../../../kernels/engram/engram_cabi.h"   // (v3.26) "engram.ring_copy"
#include "../../../kernels/rowexp/rowexp_cabi.h"
#include "../../tests/json_min.h"

namespace fpr {
namespace nd {
namespace {

bool ck(cudaError_t e, const char* what, std::string* err) {
  if (e == cudaSuccess) return true;
  if (err) *err = std::string("nd_engine: ") + what + ": " + cudaGetErrorString(e);
  return false;
}
std::string join_key(std::initializer_list<std::string> parts) {   // nd_export._key_text: "|".join(str(x))
  std::string s;
  for (const auto& p : parts) {
    if (!s.empty()) s += "|";
    s += p;
  }
  return s;
}
std::string py_bool(bool b) { return b ? "True" : "False"; }
template <class T>
T* tp(const FprEhTensor& t) { return reinterpret_cast<T*>(t.data); }

}  // namespace

Engine::Engine() = default;

Engine::~Engine() {   // nd's own pinned result sets / events / hsdprep error words (never the engine's)
  if (!ext_ready_ || h_ == nullptr) return;
  for (auto& set : res_)
    for (int r = 0; r < h_->n_ranks; ++r) {
      if (set.ns[r]) cudaFreeHost(set.ns[r]);
      if (set.smp[r]) cudaFreeHost(set.smp[r]);
      if (set.ev[r]) cudaEventDestroy(set.ev[r]);
    }
  mixed_free();
  for (int r = 0; r < h_->n_ranks; ++r) {
    if (prep_err_[r]) cudaFree(prep_err_[r]);
    for (int i = 0; i < 2; ++i) {
      if (dt_h_[r][i]) cudaFreeHost(dt_h_[r][i]);
      if (dt_ev_[r][i]) cudaEventDestroy(dt_ev_[r][i]);
    }
    if (mig_src_ev_[r]) cudaEventDestroy(mig_src_ev_[r]);
    if (mig_done_ev_[r]) cudaEventDestroy(mig_done_ev_[r]);
  }
}

// ------------------------------------------------------------------------------------------ init
Status Engine::init(const FprEngineHandover* h, RunPython run_python, std::string* err) {
  if (h == nullptr || h->version != FPR_EH_VERSION || h->struct_size != (int32_t)sizeof(FprEngineHandover)) {
    if (err) *err = "nd_engine: handover version / size mismatch";
    return Status::BAD_ARG;
  }
  h_ = h;
  run_python_ = std::move(run_python);
  const char* cj = h->config_json;   // v0.4
  if (cj == nullptr) {
    if (err) *err = "nd_engine: handover without config_json (v0.4)";
    return Status::BAD_ARG;
  }
  try {
    const jmin::J c = jmin::parse(std::string(cj));
    auto b = [&](const char* k, bool d) { const jmin::J* x = c.get(k); return x ? x->truthy() : d; };
    auto i = [&](const char* k, int64_t d) { const jmin::J* x = c.get(k); return x ? x->integer() : d; };
    cfg_.gt = b("gt", true); cfg_.owner = b("owner", true); cfg_.empty = b("empty", true);
    cfg_.adv_slice = b("adv_slice", true); cfg_.d13s = b("d13s", true); cfg_.tfuse = b("tfuse", false);
    cfg_.check = b("check", false); cfg_.k1 = i("k1", 4); cfg_.k = i("k", 3); cfg_.delta_cap = i("delta_cap", 65536);
    cfg_.c1d = b("c1d", false); cfg_.c1d_smax = i("c1d_smax", 0); cfg_.c1_mega = b("c1_mega", false);
    cfg_.c1_runs = b("c1_runs", false); cfg_.c1_ab = b("c1_ab", false); cfg_.c1_e0 = b("c1_e0", false);   // (v3.22)
    cfg_.c1_nrm = i("c1_nrm", 4);
    cfg_.head_bm = b("head_bm", false);
    cfg_.peer_map = b("peer_map", false);
    cfg_.decode_copies = b("decode_copies", false);   // (v3.27, P4: an older exporter = refused as before)
    cfg_.decode_mig = b("decode_mig", false);         // (v3.27, P4: migrated device tokens; default refused)
    cfg_.mixed_max_rows = i("mixed_max_rows", 0);
    if (const jmin::J* x = c.get("buckets")) for (const auto& e : x->a) cfg_.buckets.push_back(e.integer());
    cfg_.d13.graphs = b("graphs", true); cfg_.d13.largest_bucket = i("largest_bucket", 320);
    cfg_.d13.graph_max_ctx = i("graph_max_ctx", 0); cfg_.d13.k1 = cfg_.k1; cfg_.d13.delta_cap = cfg_.delta_cap;
    cfg_.lng.graph_max_ctx = cfg_.d13.graph_max_ctx;   // (v3.30) long-context decode: engine.step's long-variant rule
    cfg_.lng.k1 = cfg_.k1;
    if (const jmin::J* x = c.get("long_level_ctx")) for (const auto& e : x->a) cfg_.lng.level_ctx.push_back(e.integer());
    cfg_.lng.long_graphs = b("long_graphs", false); cfg_.lng.long_any_rows = b("long_any_rows", false);
    cfg_.lng.split_hsd = b("split_hsd", false); cfg_.lng.dbg_sync = b("dbg_sync", false); cfg_.lng.hsd_on = true;
    cfg_.long_async = b("long_async", false);   // (v3.33, ND_LONG)
    cfg_.lng.c1_lane = b("long_c1_lane", false);   // (v3.34) D13's lane / variable rows: long steps sync
    cfg_.d13.fb_reuse = b("fb_reuse", true);
    if (const jmin::J* x = c.get("scratch"))
      for (size_t r = 0; r < x->a.size() && r < (size_t)NW; ++r) cfg_.d13.scratch[r] = x->a[r].integer();
    cfg_.hsd.k = cfg_.k; cfg_.hsd.max_seqs = i("hsd_max_seqs", 80); cfg_.hsd.nt_max = i("hsd_nt_max", 0);
    cfg_.hsd.pool = b("hsd_pool", true); cfg_.hsd.window = i("hsd_window", 128); cfg_.hsd.ring = i("hsd_ring", 0);
    if (const jmin::J* x = c.get("hsd_seq_buckets")) for (const auto& e : x->a) cfg_.hsd.seq_buckets.push_back(e.integer());
    if (const jmin::J* x = c.get("hsd_row_buckets")) for (const auto& e : x->a) cfg_.hsd.row_buckets.push_back(e.integer());
    if (const jmin::J* x = c.get("ext")) {   // ND-2: the engine's extension builds (their C tables)
      if (const jmin::J* y = x->get("rowexp")) cfg_.ext_rowexp = y->s;
      if (const jmin::J* y = x->get("hsdprep")) cfg_.ext_hsdprep = y->s;
      if (const jmin::J* y = x->get("pagecopy")) cfg_.ext_pagecopy = y->s;   // (v3.10, optional)
    }
    cfg_.gate = b("gate", false); cfg_.nancheck = b("nancheck", false); cfg_.vrows = b("vrows", false);
    cfg_.res_owner = b("res_owner", true); cfg_.err_every = std::max<int64_t>(1, i("err_every", 1));
  } catch (const std::exception& e) {
    if (err) *err = std::string("nd_engine: config_json: ") + e.what();
    return Status::BAD_ARG;
  }
  host_.reset(new d13::D13Host(cfg_.d13, cfg_.hsd));
  // (v0.7) the engine's D13.tb as a shared C++ ITables (NULL: ENG_CXX_TB off -> nd's private builder, as before; init
  // already refused any other handover version / size)
  if (h->tb_d13 != nullptr) {
    host_->set_tables(static_cast<ITables*>(h->tb_d13));
    stats_["tables_shared"] = 1;
  }
  for (int i = 0; i < 2; ++i)   // HsdPrep writes straight into the engine's pinned blob sets (as the 2a binding)
    for (int r = 0; r < h_->n_ranks; ++r)
      host_->hsd().bind(i, r, reinterpret_cast<uint8_t*>(h_->ranks[r].hsd_hblob.h[i].ptr));
  if (h->hsd_bm_h.data != nullptr)   // HSD._bm_h: _build's mask staging = the engine's own pinned buffer
    host_->hsd().bind_bm(static_cast<int32_t*>(h->hsd_bm_h.data));
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ requests (hooks.on_ctx)
void Engine::observe(const StepHost& host, bool native) {
  observe_requests(host, &reqs_, &prev_, native ? &native_drops_ : nullptr);
}

bool Engine::prev_of(const Rid& rid, int64_t* rank, int64_t* index) const {   // engine.prev_of, PREV_STEP_CHECK on
  auto it = prev_.find(rid);
  if (it == prev_.end() || it->second.serial != step_serial_ - 1) return false;
  *rank = it->second.rank;
  *index = it->second.index;
  return true;
}

void Engine::set_prev_seq(const std::vector<StepBatch>& b) {   // D13._set_prev_seq (HSD on)
  for (size_t r = 0; r < b.size(); ++r)
    for (size_t si = 0; si < b[r].seqs.size(); ++si) prev_[b[r].seqs[si].rid] = PrevOf{(int64_t)r, (int64_t)si, step_serial_};
}

// ------------------------------------------------------------------------------------------ handover access
bool Engine::bucket_view(int64_t R, int r, FprEhBucketRank* v, std::string* err) {
  if (h_->fill_bucket(h_->ctx, R, r, v) != 0) {
    if (err) *err = std::string("nd_engine: fill_bucket: ") + h_->last_error(h_->ctx);
    return false;
  }
  if (!fpr_eh_current(h_, v)) {
    if (err) *err = "nd_engine: bucket view stale (generation)";
    return false;
  }
  return true;
}

const FprEhGraph* Engine::graph(const std::string& key, int r, int piece) {
  FprEhGraphTable t;
  h_->graphs(h_->ctx, &t);
  if (t.version != gversion_) {   // re-index the published entries [0, count)
    gindex_.clear();
    for (int64_t i = 0; i < t.count; ++i) {
      const FprEhGraph& g = t.entries[i];
      gindex_[std::string(g.key) + "#" + std::to_string(g.rank) + "#" + std::to_string(g.piece)].push_back(&g);
    }
    gversion_ = t.version;
  }
  auto it = gindex_.find(key + "#" + std::to_string(r) + "#" + std::to_string(piece));
  if (it == gindex_.end()) return nullptr;
  const FprEhGraph* g = it->second.back();   // the newest entry supersedes (F)
  // v0.5: gen_slot -1 = a bucket-independent entry; (v3.1) H1 entries carry one slot per key (bumped on a guard retire)
  if (g->gen_slot >= 0 && __atomic_load_n(&h_->gens[g->gen_slot], __ATOMIC_ACQUIRE) != g->generation) return nullptr;
  return g;
}

std::string Engine::l_key(int64_t R, int32_t mega) const {
  return mega < 0 ? join_key({std::to_string(R), "LA"}) : join_key({std::to_string(R), "LA", "M", std::to_string(mega)});
}

int32_t Engine::mega_owner(const std::vector<StepBatch>& b, int64_t R) const {   // (v3.22) engine.c1m_owner, AB off
  if (!cfg_.c1_mega || !cfg_.c1_runs || R != cfg_.c1_nrm) return -1;
  return c1m_owner_of(b, cfg_.c1_nrm);
}

bool Engine::bm_ready_() const {   // the bitmask variant of graph H can run natively
  if (!cfg_.head_bm || h_ == nullptr) return false;
  const FprEhTensor& bh = h_->hsd_bm_h;
  if (bh.data == nullptr || bh.ndim != 2 || bh.dtype != FPR_EH_I32 || bh.shape[1] != d13::HsdPrep::BM_WORDS ||
      bh.stride[0] != bh.shape[1])
    return false;
  for (int r = 0; r < h_->n_ranks; ++r) {
    const FprEhTensor& bd = h_->hsd_bm_d[r];
    if (bd.data == nullptr || bd.ndim != 2 || bd.dtype != FPR_EH_I32 || bd.shape[0] < bh.shape[0] ||
        bd.shape[1] != bh.shape[1] || bd.stride[0] != bd.shape[1])
      return false;
  }
  return true;
}

std::string Engine::h_key(int64_t R, const d13::HsdStep& c, bool bm) const {   // (R, "H", (S_pad, D_pad, NT) + gvar)
  std::string k = join_key({std::to_string(R), "H", std::to_string(c.S_pad), std::to_string(c.D_pad), std::to_string(c.NT)});
  // hsdt_c1d's gvar (HSDT._prepare): c1d on; with SMAX: S_pad <= SMAX and NT <= S_pad; then D_pad == 1 and exactly one rank
  // with drafting rows -> (("c1d", owner),)
  bool on = cfg_.c1d;
  if (on && cfg_.c1d_smax > 0 && (c.S_pad > cfg_.c1d_smax || c.NT > c.S_pad)) on = false;
  if (on && c.D_pad == 1) {
    int owner = -1, n_own = 0;
    for (int r = 0; r < (int)c.dseqs.size(); ++r)
      if (!c.dseqs[r].empty()) {
        owner = r;
        ++n_own;
      }
    if (n_own == 1) k += "|c1d|" + std::to_string(owner);
  }
  if (bm) k += "|bm";   // Engine._hkey's bitmask variant (R, "H", k + ("bm",)), FPR's ENG_HEAD_BM_GRAPH)
  return k;
}

bool Engine::hsd_key_of(const std::vector<StepBatch>& b, d13::HsdStep* k, std::string* why) const {
  auto bucket = [](int64_t n, const std::vector<int64_t>& bs, int64_t* out) {   // hsd._bucket (raises past the last)
    for (int64_t x : bs)
      if (n <= x) {
        *out = x;
        return true;
      }
    return false;
  };
  int64_t ms = 1, md = 1, mt = 1;
  k->dseqs.assign(b.size(), {});
  for (size_t r = 0; r < b.size(); ++r) {
    const auto& srl = b[r].sample_rows;
    int64_t nd = 0;
    int32_t prev = -1;
    for (size_t i = 0; i < srl.size(); ++i) {
      if (i && srl[i] <= srl[i - 1]) {
        *why = "HSD sample rows not ascending (per-element prepare)";
        return false;
      }
      const int32_t ss = (srl[i] >= 0 && srl[i] < (int32_t)b[r].seq_slot.size()) ? b[r].seq_slot[srl[i]] : -1;
      if (i && ss < prev) {
        *why = "a sequence's HSD rows not contiguous (per-element prepare)";
        return false;
      }
      if (i == 0 || ss != prev) ++nd;
      prev = ss;
    }
    for (int64_t j = 0; j < nd; ++j) k->dseqs[r].push_back({j, j});   // (only the per-rank counts matter for the key)
    ms = std::max<int64_t>(ms, (int64_t)srl.size());
    md = std::max<int64_t>(md, nd);
    mt = std::max<int64_t>(mt, (int64_t)b[r].tail_rows.size());
  }
  if (!bucket(ms, cfg_.hsd.row_buckets, &k->S_pad) || !bucket(md, cfg_.hsd.seq_buckets, &k->D_pad) ||
      !bucket(mt, host_->hsd().tail_buckets(k->S_pad), &k->NT)) {
    *why = "HSD rows beyond its largest bucket";
    return false;
  }
  return true;
}

// ------------------------------------------------------------------------------------------ routing (D13._async_ok +)
// ------------------------------------------------------------------------------------------ (v3.21, L0) loop counters
namespace {
struct ScopeUs {   // host us of one nd call into nd_tick(hook) at scope exit (every return path)
  Engine* e;
  const char* k;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  ~ScopeUs() { e->nd_tick(k, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()); }
};
std::string reason_key(const std::string& w) {   // the refusal histogram's key: digits -> '#', JSON-safe, <= 80 chars
  std::string k = "refuse:";
  for (char ch : w) {
    if (k.size() >= 87) break;
    if (ch >= '0' && ch <= '9') ch = '#';
    else if (ch == '"' || ch == '\\' || (unsigned char)ch < 0x20) ch = '_';
    if (ch == '#' && !k.empty() && k.back() == '#') continue;
    k.push_back(ch);
  }
  return k;
}
}  // namespace

// ------------------------------------------------------------------------------------------ (v3.38) ND_EVPROF
bool Engine::evp_init_() {
  if (evp_on_ >= 0) return evp_on_ == 1;
  const char* v = std::getenv("ND_EVPROF");
  evp_on_ = (v != nullptr && std::string(v) == "1" && h_ != nullptr) ? 1 : 0;
  if (evp_on_ != 1) return false;
  for (auto& set : evp_)
    for (int r = 0; r < h_->n_ranks; ++r) {
      const c10::cuda::CUDAGuard dg((c10::DeviceIndex)h_->ranks[r].device);
      for (auto& e : set.e[r])
        if (cudaEventCreate(&e) != cudaSuccess) {   // (timing events; any failure: profiling off, serving unaffected)
          cudaGetLastError();
          evp_on_ = 0;
          stats_["evp_init_failed"] = 1;
          return false;
        }
    }
  stats_["evp_on"] = 1;
  return true;
}

void Engine::evp_rec_(int k) {
  if (evp_on_ != 1 || evp_cur_ < 0) return;
  for (int r = 0; r < h_->n_ranks; ++r) {
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)h_->ranks[r].device);
    if (cudaEventRecord(evp_[evp_cur_].e[r][k], (cudaStream_t)h_->ranks[r].step_stream) != cudaSuccess) {
      cudaGetLastError();
      stats_["evp_record_failed"]++;
      evp_cur_ = -1;   // (this step is not harvested)
      return;
    }
  }
}

void Engine::evp_harvest_() {
  static const char* SEG[5] = {"pre", "L", "head", "H", "adv"};
  for (;;) {   // in seq order: the oldest recorded, unharvested set
    int best = -1;
    for (int i = 0; i < (int)evp_.size(); ++i)
      if (evp_[i].rec && !evp_[i].harv && (best < 0 || evp_[i].seq < evp_[best].seq)) best = i;
    if (best < 0) return;
    EvpSet& s = evp_[best];
    for (int r = 0; r < h_->n_ranks; ++r) {
      const cudaError_t q = cudaEventQuery(s.e[r][5]);
      if (q == cudaErrorNotReady) return;   // (GPU still on it: the next dispatch)
      if (q != cudaSuccess) {
        cudaGetLastError();
        s.harv = true;
        stats_["evp_query_failed"]++;
        goto next;
      }
    }
    {
      std::array<int64_t, 6> mx{};   // per segment (0..4) + total (5): max over ranks
      int64_t gap_mx = -1, gap0 = -1;
      const bool chain = evp_prev_ >= 0 && evp_[evp_prev_].harv && evp_[evp_prev_].seq == s.seq - 1;
      for (int r = 0; r < h_->n_ranks; ++r) {
        const c10::cuda::CUDAGuard dg((c10::DeviceIndex)h_->ranks[r].device);
        float ms = 0.f;
        for (int k = 0; k <= 5; ++k) {
          const int a = k < 5 ? k : 0, b = k < 5 ? k + 1 : 5;
          if (cudaEventElapsedTime(&ms, s.e[r][a], s.e[r][b]) != cudaSuccess) {
            cudaGetLastError();
            ms = 0.f;
          }
          const int64_t ns = (int64_t)(ms * 1e6f);
          if (r == 0) stats_[std::string("evp_") + (k < 5 ? SEG[k] : "step") + "_ns_r0"] += ns;
          mx[k] = std::max(mx[k], ns);
        }
        if (chain && cudaEventElapsedTime(&ms, evp_[evp_prev_].e[r][5], s.e[r][0]) == cudaSuccess) {
          const int64_t ns = (int64_t)(ms * 1e6f);
          if (r == 0) gap0 = ns;
          gap_mx = std::max(gap_mx, ns);
        } else {
          cudaGetLastError();
        }
      }
      for (int k = 0; k <= 5; ++k) stats_[std::string("evp_") + (k < 5 ? SEG[k] : "step") + "_ns_maxr"] += mx[k];
      stats_["evp_n"]++;
      if (gap0 >= 0) {
        stats_["evp_gap_ns_r0"] += gap0;
        stats_["evp_gap_ns_maxr"] += gap_mx;
        stats_["evp_gap_n"]++;
        if (gap0 > 50000) stats_["evp_gap_gt50us_r0"]++;
        if (gap0 > 200000) stats_["evp_gap_gt200us_r0"]++;
        int64_t& gm = stats_["evp_gap_max_ns_r0"];
        gm = std::max(gm, gap0);
      }
      s.harv = true;
      evp_prev_ = best;
      // per-step means (us) over the
      // window, rank 0 and max-over-ranks, + the idle gap (the shutdown stats print is not reliable under the harness)
      {
        static int64_t every = 0;
        if (every == 0) {
          const char* ev = std::getenv("ND_EVPROF_EVERY");
          every = ev != nullptr ? std::max<int64_t>(16, std::atoll(ev)) : 512;
        }
        if (stats_["evp_n"] % every == 0) {
          static std::map<std::string, int64_t> last;
          static const char* K[] = {"evp_pre_ns_r0", "evp_L_ns_r0", "evp_head_ns_r0", "evp_H_ns_r0", "evp_adv_ns_r0",
                                    "evp_step_ns_r0", "evp_step_ns_maxr", "evp_gap_ns_r0", "evp_gap_ns_maxr", "evp_n",
                                    "evp_gap_n", "evp_gap_gt50us_r0", "evp_gap_gt200us_r0", "evp_mega", "evp_steady"};
          std::map<std::string, int64_t> d;
          for (const char* k : K) d[k] = stats_[k] - last[k], last[k] = stats_[k];
          const double n = std::max<int64_t>(1, d["evp_n"]), ng = std::max<int64_t>(1, d["evp_gap_n"]);
          std::fprintf(stderr,
                       "nd EVPROF window %lld steps (mega %lld, steady %lld): us/step r0 pre %.1f L %.1f head %.1f H %.1f adv %.1f "
                       "| step r0 %.1f maxr %.1f | gap r0 %.1f maxr %.1f over %lld (>50us %lld, >200us %lld) | total %lld\n",
                       (long long)d["evp_n"], (long long)d["evp_mega"], (long long)d["evp_steady"], d["evp_pre_ns_r0"] / n / 1e3,
                       d["evp_L_ns_r0"] / n / 1e3, d["evp_head_ns_r0"] / n / 1e3, d["evp_H_ns_r0"] / n / 1e3,
                       d["evp_adv_ns_r0"] / n / 1e3, d["evp_step_ns_r0"] / n / 1e3, d["evp_step_ns_maxr"] / n / 1e3,
                       d["evp_gap_ns_r0"] / ng / 1e3, d["evp_gap_ns_maxr"] / ng / 1e3, (long long)d["evp_gap_n"],
                       (long long)d["evp_gap_gt50us_r0"], (long long)d["evp_gap_gt200us_r0"], (long long)stats_["evp_n"]);
          std::fflush(stderr);
        }
      }
    }
  next:;
  }
}

void Engine::nd_tick(const char* hook, double us) {
  const std::string k = std::string("nd_") + hook;
  stats_[k + "_us"] += (int64_t)us;
  stats_[k + "_n"] += 1;
  int64_t& m = stats_[k + "_max_us"];
  m = std::max<int64_t>(m, (int64_t)us);
}

std::string Engine::async_ok(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host) {
  ScopeUs t{this, "async_ok"};
  std::string w = async_ok_(b, ct, host);
  if (w.empty()) stats_["async_ok_native"]++;
  else stats_[reason_key(w)]++;
  return w;
}

std::string Engine::async_ok_(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host) {
  if (cfg_.check) return "a *_CHECK mode (compares against Python)";
  if (!cfg_.gt) return "ENG_D13_GT off (ND covers the graph-T path only)";
  if (cfg_.nancheck) return "ENG_NANCHECK (Python's checks)";
  if (cfg_.vrows) return "ENG_D13_VROWS (variable rows: not ported to nd yet)";
  if (!cfg_.hsd.pool) return "HSD without the window pool (per-element prepare)";
  std::string why = host_->async_ok(b, ct);
  if (why == "page copies" && cfg_.decode_copies) {   // (v3.27, P4) Worker.copies natively at dispatch (Python: embedded)
    for (const auto& sb : b)
      if (!sb.poison.empty()) return "page copies with POISON pages (debug: Python's Worker.copies)";
    std::vector<StepBatch> b2(b);   // the rest of D13Host's checks on the step without its copies (rare: copy steps only)
    for (auto& x : b2) x.copies.clear();
    why = host_->async_ok(b2, ct);
    if (!why.empty()) return why;
    std::vector<PageCopyBatch> g;
    copy_groups_(b, &g);
    std::string w;
    if (!page_copies_ok(g, &w)) return w.empty() ? std::string("page copies") : w;   // "page copies: <reason>"
  }
  if (why == "context beyond the decode graphs' indexer bound" && !cfg_.lng.level_ctx.empty()) {
    // (v3.30, long-context decode slice 1) the class per level: which long variant Python runs it on (its sync long
    // graphs; D13 has no async long variant yet, so nd has none to replay) or the eager mixed path
    const int lvl = long_level_of(b, ct, cfg_.lng);
    if (lvl > 0 && cfg_.long_async) {
      why.clear();   // (v3.33, ND_LONG) the async long variant: the checks below run on its key Rk (long_lvl_)
    } else if (lvl > 0) {
      why = "long-context decode lvl " + std::to_string(lvl) + " (ND_LONG off: Python's long variant)";
      stats_["long_ctx_steps_lvl" + std::to_string(lvl)]++;
    } else if (lvl == kLongEdge) {   // (v3.34, = D13._long_level) the level its device-resolved starts leave open: sync
      why = "long level edge (device-resolved start)";
      stats_["long_ctx_steps_edge"]++;
    } else if (lvl == kLongLane) {
      why = "context beyond the decode graphs' indexer bound (C1 lane / variable rows: sync)";
      stats_["long_ctx_steps_lane"]++;
    } else {
      why = "context beyond the decode graphs' indexer bound (no long variant: Python's eager mixed path)";
      stats_["long_ctx_steps_eager"]++;
    }
  }
  if (!why.empty()) return why;
  bool bm = false;   // grammar rows: the step's masks (Hooks._bitmasks) -> graph H's "|bm" variant
  if (!host.grammar.empty()) {
    std::vector<std::vector<int32_t>> bst;
    std::vector<d13::HsdBitmask> bmk;
    const int rc = grammar_bitmasks(b, host.grammar, &bst, &bmk);
    if (rc < 0) return "grammar rows (malformed: Python raises)";
    bm = rc > 0;
    if (bm && !bm_ready_()) return "grammar rows (no bitmask graph H: ENG_HEAD_BM_GRAPH off / HSD masks not exported)";
  }
  if (!host.fresh.empty() && !host.on_ctx_done && !host.on_ctx_lazy)
    return "admissions (Python's on_ctx creates the request state)";
  for (size_t r = 0; r < b.size(); ++r)   // device tokens across devices (Python: one cross-device copy): embedded
    for (const auto& se : b[r].seqs) {
      auto it = prev_.find(se.rid);   // (async_ok runs before this step's serial bump: "previous step" = step_serial_)
      if (se.kind == "decode" && it != prev_.end() && it->second.serial == step_serial_ && it->second.rank != (int64_t)r) {
        bool host_ids = true;
        for (int64_t x = se.row0; x < se.row0 + se.n_rows && x < (int64_t)ct[r].tokens.size() && x < b[r].n_rows; ++x)
          host_ids = host_ids && ct[r].tokens[x] >= 0;
        if (!host_ids && !cfg_.decode_mig) return "migrated sequence (cross-device next ids)";
        if (!host_ids) {   // (v3.27, P4) the cross-device copy natively (device_tokens): devtok_check's range rules here
          const FprEhRankState& r0s = h_->ranks[it->second.rank];
          if (it->second.rank < 0 || it->second.rank >= h_->n_ranks || se.n_rows <= 0 || se.n_rows > r0s.next_ids.shape[1] ||
              it->second.index < 0 || it->second.index >= r0s.next_ids.shape[0])
            return "migrated sequence (cross-device next ids out of range)";
        }
      }
    }
  int64_t n = 0;
  for (const auto& sb : b) n = std::max<int64_t>(n, sb.n_rows);
  int64_t R = 0;
  for (int64_t x : cfg_.buckets)
    if (n <= x) {
      R = x;
      break;
    }
  if (R == 0) return "rows > largest bucket";
  // (v3.22, L1) C1 steps on c1mega: D13.dispatch's key (R, "LA", "M", owner); the rest of the async path is unchanged.
  // Engram's prepare (generic, or c1mega's e0_prepare under C1M_IX + C1IDX_E0) runs in p_embed INSIDE the captured graph L
  // nd only uploads its inputs (ND-1), the same for both. Every c1mega buffer is
  // persistent and baked into the launch blob over the owner's exported bucket state; epochs are device-side: replay only.
  // ENG_D13_VROWS (the NR-variant key) is refused above (cfg vrows).
  const int lvl_ = long_lvl_(b, ct);           // (v3.33) 0 unless ND_LONG and a long-context step
  const int64_t Rk = long_key(R, lvl_);        // the bucket KEY: graphs / views; R stays the rows
  // c1mega runs lvl 0 only (c1idx_host LMAX = 262,144): owner_of returns None for a long state
  // (s["lvl"] checked; s["R"] != NRM): a long C1 step keys (Rk, "LA"), never the mega key
  const int32_t mo = lvl_ > 0 ? -1 : mega_owner(b, R);
  if (mo >= 0 && cfg_.c1_ab) return "C1 mega with ENG_C1_MEGA_AB (per-arm graph keys)";
  // the bucket key in the not-captured refusals, inside parentheses: core2b's decode_refusals class key
  // (cut at the first ':' / '(' / digit) is unchanged; its once-per-class stderr line shows the full text (Rk tells the long
  // variants apart, mega the C1 owner)
  const std::string lk = " (Rk " + std::to_string(Rk) + (lvl_ > 0 ? ", long lvl " + std::to_string(lvl_) : std::string()) +
                         ", mega " + std::to_string(mo) + ")";
  for (int r = 0; r < h_->n_ranks; ++r)
    if (graph(l_key(Rk, mo), r, 0) == nullptr)
      return std::string(mo >= 0 ? "C1 mega graph L not captured / published" : "graph L not captured / published") + lk;
  // the bucket ensured (its views: ensure_bucket) and graph T of each rank's has-sequences case published: pre() needs
  // both, and must never find one missing after rank 0's work is already enqueued
  for (int r = 0; r < h_->n_ranks && r < (int)b.size(); ++r) {
    FprEhBucketRank v;
    std::string e_;
    const std::string tk = "; Rk " + std::to_string(Rk) + (lvl_ > 0 ? ", long lvl " + std::to_string(lvl_) : std::string()) +
                           ", rank " + std::to_string(r) + ", seqs " + (b[r].seqs.empty() ? "0" : "1") + ")";
    if (!bucket_view(Rk, r, &v, &e_)) return std::string("bucket not ensured / stale view (ensure_bucket") + tk;
    if (graph(join_key({"T", std::to_string(Rk), std::to_string(r), py_bool(!b[r].seqs.empty())}), r, 0) == nullptr)
      return std::string("graph T not captured / published (ensure_bucket") + tk;
  }
  // graph H of this step's HSD key must be published too (a new key = an embedded step: Python runs HSD eagerly and
  // captures it when the step is extracted, i.e. in pull_state's D13.drain; the next refresh_graphs publishes it)
  d13::HsdStep hk;
  std::string why_h;
  if (!hsd_key_of(b, &hk, &why_h)) return why_h;
  const std::string hks = h_key(Rk, hk, bm);
  for (int r = 0; r < h_->n_ranks; ++r)
    if (graph(hks, r, 0) == nullptr) return "graph H not captured / published: " + hks;
  return "";
}

int Engine::long_lvl_(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct) const {   // (v3.33)
  if (!cfg_.long_async || cfg_.lng.level_ctx.empty()) return 0;
  const int lvl = long_level_of(b, ct, cfg_.lng);
  return lvl > 0 ? lvl : 0;   // (lvl -1 never reaches here: async_ok refused it)
}

// ------------------------------------------------------------------------------------------ dispatch (D13.dispatch async)
Status Engine::dispatch(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host,
                        Handle** out, std::string* err) {
  ScopeUs t_us_{this, "dispatch"};   // (v3.21, L0)
  if (!ext_ready_) {
    Status s0 = ext_init(err);
    if (s0 != Status::OK) return s0;
  }
  if (evp_init_()) {   // (v3.38) ND_EVPROF: harvest finished steps, open this step's set, E0
    evp_harvest_();
    const int i = (int)((evp_seq_ + 1) % (int64_t)evp_.size());
    if (evp_[i].rec && !evp_[i].harv) {
      evp_cur_ = -1;                           // (8 steps unfinished: skip this one)
      stats_["evp_skipped"]++;
    } else {
      evp_cur_ = i;
      evp_[i].seq = ++evp_seq_;
      evp_[i].rec = evp_[i].harv = false;
      evp_rec_(0);
    }
  }
  Rid clash;
  if (!new_dropped_disjoint(host, &clash)) {
    if (err) *err = "nd_engine: a context drops and admits the same request: " + clash;
    return Status::BAD_ARG;
  }
  bool copy_step = false;   // (v3.27, P4) Worker.copies first, as the embedded step (copies, then Engine.step's prologue)
  for (const auto& sb : b) copy_step = copy_step || !sb.copies.empty();
  if (copy_step) {
    if (!cfg_.decode_copies) {
      if (err) *err = "nd_engine: a decode step with page copies (async_ok should have refused: ND_DECODE_COPIES off)";
      return Status::BAD_ARG;
    }
    std::vector<PageCopyBatch> g;
    copy_groups_(b, &g);
    const Status cs = issue_page_copies_(nullptr, g, err);   // refused (BAD_ARG) before any device op
    if (cs != Status::OK) return cs;
    stats_["decode_copy_steps"]++;
  }
  observe(host, !host.on_ctx_done);   // on_ctx ran in Python: its drops are applied there, none queued
  if (host.on_ctx_done && (!host.fresh.empty() || !host.dropped.empty())) stats_["decode_on_ctx_steps"]++;
  int64_t n = 0;
  for (const auto& sb : b) n = std::max<int64_t>(n, sb.n_rows);
  int64_t R = 0;
  for (int64_t x : cfg_.buckets)
    if (n <= x) {
      R = x;
      break;
    }
  for (const auto& sb : b)   // engine.bump_serial: one step with sequences
    if (!sb.seqs.empty()) {
      ++step_serial_;
      break;
    }
  const int lvl = long_lvl_(b, ct);         // (v3.33, ND_LONG) = async_ok's (same batches / tables)
  const int64_t Rk = long_key(R, lvl);       // the bucket KEY (== R below lvl 1); R stays the rows
  if (lvl > 0) stats_["long_ctx_native_lvl" + std::to_string(lvl)]++;
  std::unique_ptr<Handle> hd(new Handle());
  hd->R = Rk;                    // (is_steady compares keys)
  hd->mega = lvl > 0 ? -1 : mega_owner(b, R);   // (v3.22) = async_ok's (same batches); (v3.34b) never on a long variant
  if (hd->mega >= 0) stats_["c1_mega_steps"]++;
  hd->step = b.empty() ? 0 : b[0].step;
  hd->batches = b;
  Status s = pre(Rk, R, b, ct, err);   // D13._pre
  if (s != Status::OK) return s;
  evp_rec_(1);
  for (int piece = 0;; ++piece) {   // eng.replay(lk): ENG_LAUNCH_SPLIT pieces piece-major (piece p on every rank first)
    bool any = false;
    for (int r = 0; r < h_->n_ranks; ++r) {
      const FprEhGraph* g = graph(l_key(Rk, hd->mega), r, piece);
      if (g == nullptr) continue;
      any = true;
      const c10::cuda::CUDAGuard dg((c10::DeviceIndex)h_->ranks[r].device);
      if (!ck(cudaGraphLaunch((cudaGraphExec_t)g->exec, (cudaStream_t)h_->ranks[r].step_stream), "graph L", err))
        return Status::INTERNAL;
    }
    if (!any) break;
  }
  evp_rec_(2);
  // d13s.Steady.live and is_steady (after graph L, as Python): no wait for N, HSD prepared on the device
  hd->grammar = host.grammar;   // this step's grammar rows (masks, acceptable, the drafts readback set)
  // (v3.27) a copy / migrated step takes the general path (Python runs it embedded: synchronous, never Steady)
  const bool steady = cfg_.d13s && hsdprep_ != nullptr && have_cur_ && !copy_step && !mig_step_ && is_steady(b, Rk, host);
  s = steady ? steady_head(hd.get(), b, Rk, err) : head(hd.get(), b, ct, Rk, err);   // (bucket views / graph H: keys)
  if (s != Status::OK) return s;
  evp_rec_(3);
  s = tail(hd.get(), b, Rk, err);   // _set_prev_seq, graph H, _advance, _record
  if (s != Status::OK) return s;
  evp_rec_(5);
  if (evp_cur_ >= 0) {
    evp_[evp_cur_].rec = true;
    stats_[steady ? "evp_steady" : "evp_head"]++;
    if (hd->mega >= 0) stats_["evp_mega"]++;
    evp_cur_ = -1;
  }
  Handle* raw = hd.get();
  inflight_ = raw;
  last_prep_ = raw;
  stats_[steady ? "steady" : "head"]++;
  keep(std::move(hd));
  *out = raw;
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ D13._pre (ND-1 per rank)
Status Engine::pre(int64_t Rk, int64_t R, const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct,
                   std::string* err) {   // (v3.33) Rk = the bucket key (views, set indices, graph T), R = its rows
  d2h_i_ ^= 1;
  dl_i_ ^= 1;
  mig_step_ = false;   // (v3.27, P4) migrated decode sequences: mark every rank's step stream BEFORE this step's work
  for (int r = 0; r < h_->n_ranks && r < (int)b.size() && !mig_step_ && cfg_.decode_mig; ++r)
    for (const auto& se : b[r].seqs) {
      int64_t r0 = -1, si0 = -1;
      if (migrated_(r, se, b[r], ct[r], &r0, &si0)) {
        mig_step_ = true;
        break;
      }
    }
  if (mig_step_) {
    if (!cfg_.decode_mig) {
      if (err) *err = "nd_engine: a migrated decode sequence (async_ok should have refused: ND_DECODE_MIG off)";
      return Status::BAD_ARG;
    }
    for (int r = 0; r < h_->n_ranks; ++r) {
      const c10::cuda::CUDAGuard dg((c10::DeviceIndex)h_->ranks[r].device);
      if (!ck(cudaEventRecord(mig_src_ev_[r], (cudaStream_t)h_->ranks[r].step_stream), "migrated src event", err))
        return Status::INTERNAL;
    }
    stats_["decode_mig_steps"]++;
  }
  for (int r = 0; r < h_->n_ranks; ++r) {
    FprEhBucketRank v;
    if (!bucket_view(Rk, r, &v, err)) return Status::INTERNAL;
    const FprEhRankState& rs = h_->ranks[r];
    auto& iu = in_up_i_[{Rk, r}];
    PreFull p;
    p.c.rank = r;
    p.c.device = rs.device;
    p.c.stream = (cudaStream_t)rs.step_stream;
    p.c.R = R;
    p.c.ph = tp<int64_t>(rs.d2h_pos[d2h_i_]);
    p.c.oh = tp<int64_t>(rs.d2h_own[d2h_i_]);
    p.c.pos_d = tp<const int64_t>(v.pos);
    p.c.own_d = tp<const int64_t>(v.own);
    p.c.pos_ev = (cudaEvent_t)v.pos_ev[d2h_i_];
    p.hsd_on = true;
    p.dl_rows_h = reinterpret_cast<int64_t*>(rs.dl_rows.h[dl_i_].ptr);
    p.dl_cols_h = reinterpret_cast<int64_t*>(rs.dl_cols.h[dl_i_].ptr);
    p.dl_pages_h = reinterpret_cast<int32_t*>(rs.dl_pages.h[dl_i_].ptr);
    p.dl_ev_wait = p.dl_ev = (cudaEvent_t)rs.dl_rows.ev[dl_i_];
    p.dl_rows_d = tp<int64_t>(rs.dl_rows_d);
    p.dl_cols_d = tp<int64_t>(rs.dl_cols_d);
    p.dl_pages_d = tp<int32_t>(rs.dl_pages_d);
    p.delta_cap = rs.delta_cap;
    p.n_direct = &stats_["decode_delta_direct"];   // (v3.28) _deltas' direct branch (above DELTA_CAP)
    p.dbt = tp<int32_t>(rs.rx_dbt);
    p.dbt_rows = rs.rx_dbt.shape[0];
    p.dbt_cols = rs.rx_dbt.shape[1];
    p.dbt_stride0 = rs.rx_dbt.stride[0];
    p.in_h = reinterpret_cast<uint8_t*>(v.in.h[iu[0]].ptr);
    p.in_ev_wait = p.in_ev = (cudaEvent_t)v.in.ev[iu[0]];
    p.in_d = v.in_d.ptr;
    p.in_bytes = v.in_d.bytes;
    p.in_off_tok = v.in_off_tok;
    p.in_off_valid = v.in_off_valid;
    p.in_off_samp = v.in_off_samp;
    p.up_h = reinterpret_cast<uint8_t*>(v.up.h[iu[1]].ptr);
    p.up_ev_wait = p.up_ev = (cudaEvent_t)v.up.ev[iu[1]];
    p.up_d = v.up_d.ptr;
    p.up_bytes = v.up_d.bytes;
    p.up_off_valid = v.up_off_valid;
    p.up_off_pslot = v.up_off_pslot;
    p.up_off_rs = v.up_off_rs;
    p.up_B = v.up_B;
    p.pslot_fill = rs.idx_dbt.shape[0] - 1;
    p.rslot_d = v.idx_rslot.data;   // (owner signature: the base copy is harmless when no owner step follows)
    p.rbase_d = v.own_rbase.data;
    p.rslot_bytes = v.idx_rslot.shape[0] * 8;
    const uint8_t rxi = rx_hi_[r];
    p.rx_h32 = reinterpret_cast<int32_t*>(rs.rx_h32.h[rxi].ptr);
    p.rx_h64 = reinterpret_cast<int64_t*>(rs.rx_h64.h[rxi].ptr);
    p.rx_ev_wait = p.rx_ev = (cudaEvent_t)rs.rx_h32.ev[rxi];
    p.rx_b32 = reinterpret_cast<int32_t*>(rs.rx_b32_d.ptr);
    p.rx_b64 = reinterpret_cast<int64_t*>(rs.rx_b64_d.ptr);
    p.S_max = rs.rx_S_max;
    p.ops_max = rs.rx_ops_max;
    p.R_max = rs.rx_R_max;
    p.nb_h = reinterpret_cast<int64_t*>(rs.nb.h[d2h_i_].ptr);
    p.nb_ev_wait = p.nb_ev = (cudaEvent_t)rs.nb.ev[d2h_i_];
    p.nb_d = reinterpret_cast<int64_t*>(rs.nb_d.ptr);
    const FprEhGraph* gt = graph(join_key({"T", std::to_string(Rk), std::to_string(r), py_bool(!b[r].seqs.empty())}), r, 0);
    if (gt == nullptr) {
      if (err) *err = "nd_engine: graph T not published for bucket " + std::to_string(Rk);
      return Status::BAD_ARG;
    }
    p.gt_exec = (cudaGraphExec_t)gt->exec;
    p.engram = rs.engram_pk_d.ptr != nullptr;
    if (p.engram) {
      const uint8_t ei = eg_i_[r];
      p.eg_stage_h = reinterpret_cast<int32_t*>(rs.engram_stage.h[ei].ptr);
      p.eg_ev = (cudaEvent_t)rs.engram_stage.ev[ei];
      p.eg_pk_d = reinterpret_cast<int32_t*>(rs.engram_pk_d.ptr);
      p.eg_pk_n = rs.engram_pk_d.bytes / 4;
      p.eg_off_row0 = rs.engram_pk_off[1]; p.eg_off_nrows = rs.engram_pk_off[2]; p.eg_off_start = rs.engram_pk_off[3];
      p.eg_off_slot = rs.engram_pk_off[4]; p.eg_off_look = rs.engram_pk_off[5]; p.eg_off_counts = rs.engram_pk_off[6];
      p.eg_max_rows = rs.engram_pk_off[1];                 // _pk_off["row0"] = max_rows (ids section)
      p.eg_max_seqs = (rs.engram_pk_off[2] - rs.engram_pk_off[1]);   // nrows - row0 = S
      p.eg_ring_w = 8;
    }
    // the Python bookkeeping of the sets this call uses (toggles as Python's)
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    Status s = pre_full(p, b[r], ct[r], *host_, err);
    if (s != Status::OK) return s;
    s = device_tokens(r, b[r], ct[r], v, err);   // (Engine._upload_v2's tail; after pre_full's ops: value-identical order)
    if (s != Status::OK) return s;
    iu[0] ^= 1;
    iu[1] ^= 1;
    rx_hi_[r] ^= 1;
    if (p.engram) eg_i_[r] ^= 1;
  }
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ ND-2: C tables + nd's buffers
namespace {
using CFn = int (*)(const void*);
CFn find_fn(const std::string& so, const char* sym, const char* name, int64_t args_size, std::string* err) {
  if (so.empty()) {
    if (err) *err = std::string("nd_engine: no extension path for ") + name + " (config_json \"ext\")";
    return nullptr;
  }
  void* lib = dlopen(so.c_str(), RTLD_NOW | RTLD_NOLOAD);   // the engine's own loaded build
  if (lib == nullptr) lib = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    const char* e = dlerror();
    if (err) *err = std::string("nd_engine: dlopen ") + so + ": " + (e ? e : "?");
    return nullptr;
  }
  auto tf = reinterpret_cast<FprTableFn>(dlsym(lib, sym));
  const FprExtTable* t = tf ? tf() : nullptr;
  if (t == nullptr || t->abi_version != FPR_CABI_VERSION) {
    if (err) *err = std::string("nd_engine: ") + sym + " missing or ABI version mismatch in " + so;
    return nullptr;
  }
  for (int i = 0; i < t->n_fns; ++i)
    if (std::strcmp(t->fns[i].name, name) == 0) {
      if (t->fns[i].args_size != args_size) {
        if (err) *err = std::string("nd_engine: ") + name + " Args size " + std::to_string(t->fns[i].args_size) +
                        " != " + std::to_string(args_size) + " (header drift)";
        return nullptr;
      }
      return t->fns[i].fn;
    }
  if (err) *err = std::string("nd_engine: ") + name + " not in " + sym;
  return nullptr;
}
}  // namespace

Status Engine::ext_init(std::string* err) {
  if (ext_ready_) return Status::OK;
  rx_advance_ = find_fn(cfg_.ext_rowexp, "fpr_rowexp_table", "rowexp.advance", (int64_t)sizeof(FprRowexpAdvanceArgs), err);
  if (rx_advance_ == nullptr) return Status::BAD_ARG;
  {
    std::string e2;
    rx_ring_ops_ = find_fn(cfg_.ext_rowexp, "fpr_rowexp_table", "rowexp.ring_ops", (int64_t)sizeof(FprRowexpRingOpsArgs), &e2);
    if (rx_ring_ops_ != nullptr)
      rx_expand_ = find_fn(cfg_.ext_rowexp, "fpr_rowexp_table", "rowexp.expand", (int64_t)sizeof(FprRowexpExpandArgs), &e2);
    if (rx_ring_ops_ == nullptr || rx_expand_ == nullptr) {
      rx_ring_ops_ = rx_expand_ = nullptr;
      mixed_off_ = "rowexp without ring_ops / expand: " + e2;
    }
  }
  stats_["mixed_off"] = 0;
  if (!cfg_.ext_pagecopy.empty()) {   // (v3.10, optional: missing / without the C ABI -> native page copies refused)
    std::string e3;
    pc_copy_ = find_fn(cfg_.ext_pagecopy, "fpr_pagecopy_table", "pagecopy.copy", (int64_t)sizeof(FprPagecopyArgs), &e3);
    stats_["pagecopy_native"] = pc_copy_ != nullptr;
  }
#ifdef FPR_ENGRAM_RING_COPY_MAX
  if (h_->engram_lib != nullptr && h_->engram_lib[0] != 0) {   // the state pool's Engram ring follow
    std::string e4;
    eg_ring_copy_ = find_fn(h_->engram_lib, "fpr_engram_table", "engram.ring_copy", (int64_t)sizeof(FprEngramRingCopyArgs), &e4);
  }
#endif
  stats_["engram_ring_copy_native"] = eg_ring_copy_ != nullptr;
  for (int d = 0; d < h_->n_ranks; ++d)   // (v3.26) P2P capability per ordered rank pair (the ring copy reads the source)
    for (int s2 = 0; s2 < h_->n_ranks; ++s2) {
      int can = 0;
      if (d == s2) can = 1;
      else if (cudaDeviceCanAccessPeer(&can, h_->ranks[d].device, h_->ranks[s2].device) != cudaSuccess) can = 0;
      peer_ok_[(size_t)d][(size_t)s2] = can != 0;
    }
  if (cfg_.d13s) {
    hsdprep_ = find_fn(cfg_.ext_hsdprep, "fpr_hsdprep_table", "hsdprep.hsdprep", (int64_t)sizeof(FprHsdprepArgs), err);
    if (hsdprep_ == nullptr) return Status::BAD_ARG;
  }
  const int64_t MR = (int64_t)NW * (cfg_.hsd.max_seqs + 1), K1 = cfg_.k + 1;
  for (int r = 0; r < h_->n_ranks; ++r) {
    const FprEhRankState& rs = h_->ranks[r];
    if (rs.sampled.ndim != 2 || rs.sampled.shape[1] != K1 || rs.num_sampled.dtype != FPR_EH_I32 ||
        rs.sampled.dtype != FPR_EH_I64 || rs.num_sampled.stride[0] != 1) {
      if (err) *err = "nd_engine: num_sampled / sampled views are not int32 [MR] / int64 [MR, k + 1]";
      return Status::BAD_ARG;
    }
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    for (auto& set : res_) {
      if (!ck(cudaHostAlloc((void**)&set.ns[r], MR * 4, cudaHostAllocPortable), "result set ns", err) ||
          !ck(cudaHostAlloc((void**)&set.smp[r], MR * K1 * 8, cudaHostAllocPortable), "result set sampled", err) ||
          !ck(cudaEventCreateWithFlags(&set.ev[r], cudaEventDisableTiming), "result set event", err))
        return Status::INTERNAL;
    }
    if (rs.dt_d.data == nullptr || rs.dt_d.ndim != 2 || rs.dt_d.shape[0] != 2 || rs.dt_d.dtype != FPR_EH_I64 ||
        rs.next_ids.data == nullptr || rs.next_ids.ndim != 2 || rs.next_ids.dtype != FPR_EH_I64 ||
        rs.next_ids.stride[0] != rs.next_ids.shape[1]) {
      if (err) *err = "nd_engine: dt_d [2, nmax] int64 / next_ids contiguous int64 missing (nd_export creates Hooks._dt_d)";
      return Status::BAD_ARG;
    }
    dt_nmax_ = rs.dt_d.shape[1];
    for (int i = 0; i < 2; ++i)
      if (!ck(cudaHostAlloc((void**)&dt_h_[r][i], 2 * dt_nmax_ * 8, cudaHostAllocPortable), "dt staging", err) ||
          !ck(cudaEventCreateWithFlags(&dt_ev_[r][i], cudaEventDisableTiming), "dt event", err))
        return Status::INTERNAL;
    if (!ck(cudaEventCreateWithFlags(&mig_src_ev_[r], cudaEventDisableTiming), "migrated src event", err) ||   // (v3.27)
        !ck(cudaEventCreateWithFlags(&mig_done_ev_[r], cudaEventDisableTiming), "migrated done event", err))
      return Status::INTERNAL;
    if (!ck(cudaMalloc((void**)&prep_err_[r], 2 * sizeof(int32_t)), "hsdprep err", err) ||
        !ck(cudaMemset(prep_err_[r], 0, 2 * sizeof(int32_t)), "hsdprep err", err))
      return Status::INTERNAL;
  }
  if (mixed_off_.empty()) {
    std::string e2;
    if (mixed_ext_init(&e2) != Status::OK) {
      mixed_off_ = "mixed staging: " + e2;
      mixed_free();
      mx_start_h_.fill(nullptr);
      mx_acc_h_.fill(nullptr);
      mx_acc_d_.fill(nullptr);
      mx_idx_d_.fill(nullptr);
      for (auto& a : mx_idx_h_) a.fill(nullptr);
      for (auto& a : mx_idx_ev_) a.fill(nullptr);
    }
  }
  if (!mixed_off_.empty()) stats_["mixed_off"] = 1;   // (the reason: can_head's why / begin's error)
  ext_ready_ = true;
  return Status::OK;
}

void Engine::keep(std::unique_ptr<Handle> h) {   // free the oldest CONSUMED handles beyond KEEP (no other reference left)
  handles_.push_back(std::move(h));
  while (handles_.size() > KEEP) {
    Handle* o = handles_.front().get();
    if (!o->consumed || o == inflight_ || o == last_prep_) break;
    for (auto& set : res_)
      if (set.owner == o) set.owner = nullptr;
    for (auto& eo : ev_owner_)
      if (eo == o) eo = nullptr;
    handles_.erase(handles_.begin());
  }
}

// ------------------------------------------------------------------------------------------ Hooks._device_tokens_v2
Status Engine::device_tokens(int r, const StepBatch& sb, const CompactTables& ct, const FprEhBucketRank& v,
                             std::string* err) {
  // decode rows of a sequence that ran in the immediately previous step on THIS rank read their input ids from that step's
  // hsd.next_ids row (index si0 * (k + 1) + j); host-given ids win; migrated sequences are refused by async_ok
  const FprEhRankState& rs = h_->ranks[r];
  const int64_t k1 = rs.next_ids.shape[1];
  std::vector<int64_t> rows, src;
  std::vector<DevtokMig> mig;                   // (v3.27, P4) Python: s["tok"][row0:row0 + nr].copy_(next_ids[r0][si0, :nr])
  std::array<bool, NW> from{};
  for (const auto& se : sb.seqs) {
    if (se.kind != "decode") continue;
    int64_t r0 = -1, si0 = -1;
    if (!prev_of(se.rid, &r0, &si0)) continue;
    bool host_ids = true;                       // min(tl_[row0:row0 + nr]) >= 0
    for (int64_t x = se.row0; x < se.row0 + se.n_rows && x < (int64_t)ct.tokens.size() && x < sb.n_rows; ++x)
      host_ids = host_ids && ct.tokens[x] >= 0;
    if (host_ids) continue;
    if (r0 != r) {
      if (!cfg_.decode_mig || !mig_step_ || r0 < 0 || r0 >= h_->n_ranks) {
        if (err) *err = "nd_engine: device tokens of a migrated sequence (async_ok should have refused): " + se.rid;
        return Status::INTERNAL;
      }
      mig.push_back(DevtokMig{(int64_t)se.row0, (int64_t)se.n_rows, (int32_t)r0, si0});
      from[(size_t)r0] = true;
      continue;
    }
    for (int64_t j = 0; j < se.n_rows; ++j) {
      rows.push_back(se.row0 + j);
      src.push_back(si0 * k1 + j);
    }
  }
  if (mig.empty()) return devtok_issue(r, reinterpret_cast<int64_t*>(v.tok.data), v.tok.shape[0], rows, src, {}, err);
  // (v3.27) RAW: rank r's step stream waits for each source rank's step N (its next_ids rows), marked at pre()'s start
  const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
  cudaStream_t ss = (cudaStream_t)rs.step_stream;
  for (int r0 = 0; r0 < h_->n_ranks; ++r0)
    if (from[(size_t)r0] && !ck(cudaStreamWaitEvent(ss, mig_src_ev_[(size_t)r0], 0), "migrated src wait", err))
      return Status::INTERNAL;
  const Status s = devtok_issue(r, reinterpret_cast<int64_t*>(v.tok.data), v.tok.shape[0], rows, src, mig, err);
  if (s != Status::OK) return s;
  // WAR: each source rank's later work (graph L / H of this step: they rewrite next_ids) waits for these copies
  {
    const c10::cuda::CUDAGuard dg2((c10::DeviceIndex)rs.device);
    if (!ck(cudaEventRecord(mig_done_ev_[(size_t)r], ss), "migrated done event", err)) return Status::INTERNAL;
  }
  for (int r0 = 0; r0 < h_->n_ranks; ++r0) {
    if (!from[(size_t)r0]) continue;
    const c10::cuda::CUDAGuard dg0((c10::DeviceIndex)h_->ranks[r0].device);
    if (!ck(cudaStreamWaitEvent((cudaStream_t)h_->ranks[r0].step_stream, mig_done_ev_[(size_t)r], 0), "migrated done wait", err))
      return Status::INTERNAL;
  }
  stats_["decode_mig_seqs"] += (int64_t)mig.size();
  return Status::OK;
}

bool Engine::migrated_(int r, const SeqEntry& se, const StepBatch& sb, const CompactTables& ct, int64_t* r0,
                       int64_t* si0) const {   // (v3.27, P4) device_tokens' migrated case, as a predicate
  if (se.kind != "decode") return false;
  if (!prev_of(se.rid, r0, si0) || *r0 == r) return false;
  bool host_ids = true;
  for (int64_t x = se.row0; x < se.row0 + se.n_rows && x < (int64_t)ct.tokens.size() && x < sb.n_rows; ++x)
    host_ids = host_ids && ct.tokens[x] >= 0;
  return !host_ids;
}

void Engine::copy_groups_(const std::vector<StepBatch>& b, std::vector<PageCopyBatch>* g) {   // (v3.27) Worker.copies' groups
  g->clear();
  std::vector<CopyOp> ops;
  dedup_copies(b, &ops);
  page_copy_batches(ops, g);
}

Status Engine::devtok_check(int r, int64_t tok_len, const std::vector<int64_t>& rows, const std::vector<int64_t>& src,
                            const std::vector<DevtokMig>& mig, std::string* err) const {
  const FprEhRankState& rs = h_->ranks[r];
  const int64_t k1 = rs.next_ids.shape[1];
  const int64_t m = (int64_t)rows.size();
  if ((int64_t)src.size() != m) {
    if (err) *err = "nd_engine: device-token rows / src sizes differ";
    return Status::BAD_ARG;
  }
  if (m > dt_nmax_) {
    if (err) *err = "nd_engine: more device-token rows than Hooks._dt_d holds";
    return Status::BAD_ARG;
  }
  for (size_t i = 0; i < rows.size(); ++i)
    if (rows[i] < 0 || rows[i] >= tok_len || src[i] < 0 || src[i] >= rs.next_ids.shape[0] * k1) {
      if (err) *err = "nd_engine: device-token pair out of range";
      return Status::BAD_ARG;
    }
  for (const DevtokMig& g : mig)
    if (g.rank < 0 || g.rank >= h_->n_ranks || g.row0 < 0 || g.n <= 0 || g.row0 + g.n > tok_len || g.n > k1 ||
        g.seq < 0 || g.seq >= h_->ranks[g.rank].next_ids.shape[0]) {
      if (err) *err = "nd_engine: migrated device-token copy out of range";
      return Status::BAD_ARG;
    }
  return Status::OK;
}

Status Engine::devtok_issue(int r, int64_t* tok_p, int64_t tok_len, const std::vector<int64_t>& rows,
                            const std::vector<int64_t>& src, const std::vector<DevtokMig>& mig, std::string* err) {
  const FprEhRankState& rs = h_->ranks[r];
  const int64_t k1 = rs.next_ids.shape[1];
  const int64_t m = (int64_t)rows.size();
  if (devtok_check(r, tok_len, rows, src, mig, err) != Status::OK) return Status::INTERNAL;
  if (!m && mig.empty()) return Status::OK;
  const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
  cudaStream_t ss = (cudaStream_t)rs.step_stream;
  for (const DevtokMig& g : mig) {   // migrated: s["tok"][row0:row0 + n].copy_(next_ids[r0][si0, :n].to(dev)), in order
    const FprEhRankState& r0 = h_->ranks[g.rank];
    const int64_t* srcp = reinterpret_cast<const int64_t*>(r0.next_ids.data) + g.seq * r0.next_ids.shape[1];
    if (!ck(cudaMemcpyPeerAsync(tok_p + g.row0, rs.device, srcp, r0.device, g.n * 8, ss), "devtok migrated copy", err))
      return Status::INTERNAL;
  }
  if (!m) return Status::OK;
  const int i = dt_i_[r];
  if (!ck(cudaEventSynchronize(dt_ev_[r][i]), "dt set", err)) return Status::INTERNAL;   // its copy two steps ago
  int64_t* hb = dt_h_[r][i];
  std::copy(rows.begin(), rows.end(), hb);
  std::copy(src.begin(), src.end(), hb + dt_nmax_);
  if (!ck(cudaMemcpy2DAsync(rs.dt_d.data, dt_nmax_ * 8, hb, dt_nmax_ * 8, m * 8, 2, cudaMemcpyHostToDevice, ss), "dt H2D", err) ||
      !ck(cudaEventRecord(dt_ev_[r][i], ss), "dt event", err))
    return Status::INTERNAL;
  dt_i_[r] = (uint8_t)(i ^ 1);
  // s["tok"].index_copy_(0, d[0, :n], hsd.next_ids[r].view(-1).index_select(0, d[1, :n])) on the step stream
  const c10::Device dev(c10::DeviceType::CUDA, (c10::DeviceIndex)rs.device);
  const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(ss, (c10::DeviceIndex)rs.device));
  const auto i64 = at::TensorOptions().dtype(at::kLong).device(dev);
  at::Tensor tok = at::for_blob(tok_p, {tok_len}).options(i64).target_device(dev).make_tensor();
  at::Tensor d0 = at::for_blob(rs.dt_d.data, {m}).options(i64).target_device(dev).make_tensor();
  at::Tensor d1 = at::for_blob(reinterpret_cast<int64_t*>(rs.dt_d.data) + dt_nmax_, {m}).options(i64).target_device(dev)
                      .make_tensor();
  at::Tensor nid = at::for_blob(rs.next_ids.data, {rs.next_ids.shape[0] * k1}).options(i64).target_device(dev).make_tensor();
  tok.index_copy_(0, d0, at::index_select(nid, 0, d1));
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ the steady path (d13s.py)
bool Engine::is_steady(const std::vector<StepBatch>& b, int64_t R, const StepHost& host) const {
  if (inflight_ == nullptr || inflight_ != last_prep_ || inflight_->R != R || inflight_->done || inflight_->deferred) return false;
  if (!host.grammar.empty()) return false;   // (v3.25) Steady.is_steady: H.bm_step or H.grammar -> not steady
  for (const auto& sb : b)                   // (v3.25) ... any structured request (Hooks.structured) -> not steady
    for (const auto& se : sb.seqs) {
      auto it = reqs_.find(se.rid);
      if (it != reqs_.end() && it->second.structured) return false;
    }
  for (const auto& sb : b)
    for (const auto& se : sb.seqs)
      if (se.kind != "decode" || se.n_rows != 4) return false;
  auto sig = [](const std::vector<StepBatch>& x) {   // Steady.signature
    std::ostringstream o;
    for (const auto& sb : x) {
      o << "[";
      for (const auto& se : sb.seqs) o << se.rid << "," << se.state_slot << "," << se.kind << "," << se.n_rows << "," << se.row0 << ";";
      o << "]";
    }
    return o.str();
  };
  return sig(b) == sig(inflight_->batches);
}

namespace {
const d13::HsdPrep::Field* field(const d13::HsdPrep& p, const char* name) {
  for (const auto& f : p.layout())
    if (f.name == name) return &f;
  return nullptr;
}
}  // namespace

Status Engine::defer_extract(Handle* nh, std::string* err) {   // Steady.defer_extract(N)
  ResSet& set = res_[res_i_];
  if (set.owner != nullptr && !set.owner->done) {   // (never in the 2b loop: a step is waited before dispatch(N + 2))
    Status s0 = extract(set.owner, err);
    if (s0 != Status::OK) return s0;
  }
  set.owner = nh;
  nh->res_set = res_i_;
  res_i_ = (res_i_ + 1) % NRING;
  const d13::HsdStep& c = nh->cur;   // (hsd.cur = N's step record here)
  const int64_t n = c.n_req, K1 = cfg_.k + 1;
  nh->act.clear();
  for (int r = 0; r < h_->n_ranks; ++r)
    if (r < (int)c.dseqs.size() && !c.dseqs[r].empty()) nh->act.push_back(r);
  for (int r : nh->act) {
    const FprEhRankState& rs = h_->ranks[r];
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    cudaStream_t xs = (cudaStream_t)rs.xstream, ss = (cudaStream_t)rs.step_stream;
    if (!ck(cudaStreamWaitEvent(xs, nh->ev[r], 0), "defer: xstream wait", err) ||
        !ck(cudaMemcpyAsync(set.ns[r], rs.num_sampled.data, n * 4, cudaMemcpyDeviceToHost, xs), "defer: num_sampled", err) ||
        !ck(cudaMemcpy2DAsync(set.smp[r], K1 * 8, rs.sampled.data, rs.sampled.stride[0] * 8, K1 * 8, n,
                              cudaMemcpyDeviceToHost, xs), "defer: sampled", err) ||
        !ck(cudaEventRecord(set.ev[r], xs), "defer: record", err) ||
        !ck(cudaStreamWaitEvent(ss, set.ev[r], 0), "defer: step stream wait", err))   // graph H of N + 1 rewrites them
      return Status::INTERNAL;
  }
  nh->deferred = true;
  stats_["deferred"]++;
  return Status::OK;
}

Status Engine::steady_head(Handle* hd, const std::vector<StepBatch>& b, int64_t R, std::string* err) {
  Status s = defer_extract(inflight_, err);
  if (s != Status::OK) return s;
  // Steady.prepare_device: cur = N's record with N + 1's step numbers; hsdprep per rank over the live device blob
  d13::HsdStep c = cur_;
  c.step.assign(b.size(), 0);
  for (size_t r = 0; r < b.size(); ++r) c.step[r] = b[r].step;
  cur_ = c;
  hd->cur = c;
  const d13::HsdPrep& hp = host_->hsd();
  const char* names[] = {"cs", "ce", "g_dslot", "g_p0", "pos", "wbase", "wpages", "lslot", "tpos", "tslot"};
  const d13::HsdPrep::Field* f[10];
  for (int i = 0; i < 10; ++i)
    if ((f[i] = field(hp, names[i])) == nullptr) {
      if (err) *err = std::string("nd_engine: HSD blob field ") + names[i] + " missing";
      return Status::INTERNAL;
    }
  for (int r = 0; r < h_->n_ranks; ++r) {
    const FprEhRankState& rs = h_->ranks[r];
    FprEhBucketRank v;
    if (!bucket_view(R, r, &v, err)) return Status::INTERNAL;
    uint8_t* blob = reinterpret_cast<uint8_t*>(rs.hsd_dblob.ptr);
    FprHsdprepArgs a{};
    a.c = FprCall{(cudaStream_t)rs.step_stream, rs.device, 0};
    a.cs = reinterpret_cast<const int32_t*>(blob + f[0]->off);
    a.ce = reinterpret_cast<const int32_t*>(blob + f[1]->off);
    a.g_dslot = reinterpret_cast<const int64_t*>(blob + f[2]->off);
    a.g_p0 = reinterpret_cast<int64_t*>(blob + f[3]->off);
    a.pos = reinterpret_cast<int64_t*>(blob + f[4]->off);
    a.num_sampled = tp<const int32_t>(rs.num_sampled);
    a.WD = (int32_t)(NW * c.D_pad);
    a.q0 = (int32_t)(r * c.D_pad);
    a.D_pad = (int32_t)c.D_pad;
    a.wbase = reinterpret_cast<int64_t*>(blob + f[5]->off);
    a.wpages = reinterpret_cast<int64_t*>(blob + f[6]->off);
    a.nbw = (int32_t)f[6]->cols;
    a.lslot = reinterpret_cast<const int64_t*>(blob + f[7]->off);
    a.ring = tp<const int32_t>(rs.rx_ring);
    a.window = (int32_t)cfg_.hsd.window;
    a.tpos = reinterpret_cast<int64_t*>(blob + f[8]->off);
    a.tslot = reinterpret_cast<int64_t*>(blob + f[9]->off);
    a.nt = (int32_t)(r < (int)c.n_t.size() ? c.n_t[r] : 0);
    a.rx_pos = tp<const int64_t>(v.pos);
    a.rx_own = tp<const int64_t>(v.own);
    a.err = prep_err_[r];
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    if (hsdprep_(&a) != FPR_OK) {
      if (err) *err = std::string("nd_engine: hsdprep: ") + cudaGetErrorString(cudaPeekAtLastError());
      return Status::INTERNAL;
    }
  }
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ the head (non-steady)
Status Engine::head(Handle* hd, const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, int64_t R,
                    std::string* err) {
  if (inflight_ != nullptr && !inflight_->done) {   // D13.dispatch: N's results first (HSD's result of N is read before
    Status s0 = extract(inflight_, err);           // HSD's prepare of N + 1 overwrites it)
    if (s0 != Status::OK) return s0;
  }
  // D13._positions: this step's expanded positions (the D2H right behind the previous step)
  std::vector<StepBatch> pb(b.size());
  for (int r = 0; r < h_->n_ranks && r < (int)b.size(); ++r) {
    FprEhBucketRank v;
    if (!bucket_view(R, r, &v, err)) return Status::INTERNAL;
    if (!ck(cudaEventSynchronize((cudaEvent_t)v.pos_ev[d2h_i_]), "positions event", err)) return Status::INTERNAL;
    const int64_t n = b[r].n_rows;
    if (!n) {
      pb[r] = b[r];
      continue;
    }
    const FprEhRankState& rs = h_->ranks[r];
    const int64_t* ph = tp<const int64_t>(rs.d2h_pos[d2h_i_]);
    const int64_t* oh = tp<const int64_t>(rs.d2h_own[d2h_i_]);
    host_->positions(b[r], std::vector<int64_t>(ph, ph + n), std::vector<int64_t>(oh, oh + n), &pb[r]);
  }
  // hooks.prepare_head: the sampling states (samp_ctx: drafts_valid = the previous step's device ids or host-given ids)
  std::vector<std::vector<d13::HsdState>> states(pb.size());
  for (size_t r = 0; r < pb.size(); ++r) {
    const auto& tok = ct[r].tokens;
    for (const auto& se : pb[r].seqs) {
      d13::HsdState st;
      auto it = reqs_.find(se.rid);
      int64_t pr = -1, pi = -1;
      const bool dev_ok = prev_of(se.rid, &pr, &pi);
      bool host_ok = false;
      if (!dev_ok && se.kind == "decode" && se.n_rows > 1 && pb[r].n_rows > 0 &&
          se.row0 + se.n_rows <= (int64_t)std::min<size_t>(tok.size(), (size_t)pb[r].n_rows)) {
        host_ok = true;
        for (int64_t x = se.row0; x < se.row0 + se.n_rows; ++x) host_ok = host_ok && tok[x] >= 0;
      }
      if (it != reqs_.end()) {
        st.temperature = it->second.sampling.temperature;
        st.seed = it->second.sampling.seed;
        st.ident = (uint64_t)it->second.serial;   // the slot's state object identity (samp_ctx[rid] lives per request)
      } else {
        st.temperature = 1.0;
        st.seed = 0;
        st.ident = 0;
      }
      st.drafts_valid = dev_ok || host_ok;
      const GrammarRows* g = grammar_of(hd->grammar, se.rid);   // (v3.25) num_acceptable_drafts = int(grammar[rid][1])
      st.has_num_acceptable = g != nullptr;
      st.num_acceptable = g != nullptr ? g->num_acceptable : 0;
      states[r].push_back(st);
    }
  }
  // HSD._build: wait for each rank's set (its copy two prepares ago), fill it; HSD._upload: one H2D per rank
  d13::HsdPrep& hp = host_->hsd();
  const int hi = hp.hi();
  for (int r = 0; r < h_->n_ranks; ++r)
    if (!ck(cudaEventSynchronize((cudaEvent_t)h_->ranks[r].hsd_hblob.ev[hi]), "HSD blob set", err)) return Status::INTERNAL;
  d13::HsdStep c;
  std::vector<std::vector<int32_t>> bst;   // Hooks._bitmasks(pb) -> HSD.prepare's bitmasks
  std::vector<d13::HsdBitmask> bmk;
  const int brc = hd->grammar.empty() ? 0 : grammar_bitmasks(pb, hd->grammar, &bst, &bmk);
  if (brc < 0 || (brc > 0 && !bm_ready_())) {
    if (err) *err = "nd_engine: grammar masks on a native step without the bitmask variant (async_ok should have refused)";
    return Status::INTERNAL;
  }
  hd->bm = brc > 0;
  Status s = host_->hsd_prepare(pb, states, hd->bm ? &bmk : nullptr, &c, err);
  if (s != Status::OK) return s;
  if (c.v0) {
    if (err) *err = "nd_engine: HSD rows outside the vectorized layout (async_ok should have refused)";
    return Status::INTERNAL;
  }
  for (int r = 0; r < h_->n_ranks; ++r) {
    const FprEhRankState& rs = h_->ranks[r];
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    cudaStream_t ss = (cudaStream_t)rs.step_stream;
    if (!ck(cudaMemcpyAsync(rs.hsd_dblob.ptr, rs.hsd_hblob.h[hi].ptr, host_->hsd_blob_bytes(), cudaMemcpyHostToDevice, ss),
            "HSD blob H2D", err) ||
        !ck(cudaEventRecord((cudaEvent_t)rs.hsd_hblob.ev[hi], ss), "HSD blob event", err))
      return Status::INTERNAL;
    // HSD._upload's mask part: _bm_d[r][:T].copy_(_bm_h[:T]) on the same stream (every rank gets all T rows).
    // _bm_h is single-buffered as in Python: the previous mask step's copy finished before this prepare (its extraction
    // waited for the step, whose graph H came after the copy)
    if (hd->bm &&
        !ck(cudaMemcpyAsync(h_->hsd_bm_d[r].data, h_->hsd_bm_h.data, (size_t)(c.T * d13::HsdPrep::BM_WORDS * 4),
                            cudaMemcpyHostToDevice, ss), "HSD bitmask H2D", err))
      return Status::INTERNAL;
  }
  if (hd->bm) stats_["bm_steps"]++;
  host_->hsd_uploaded();
  cur_ = c;
  have_cur_ = true;
  hd->cur = c;
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ prev-seq, graph H, advance, record
Status Engine::tail(Handle* hd, const std::vector<StepBatch>& b, int64_t R, std::string* err) {
  set_prev_seq(b);
  const std::string hk = h_key(R, cur_, hd->bm);
  std::vector<const FprEhGraph*> gh(h_->n_ranks, nullptr);
  for (int r = 0; r < h_->n_ranks; ++r)
    if ((gh[r] = graph(hk, r, 0)) == nullptr) {   // (the embedded engine captures every key first: warm-up sweep)
      if (err) *err = "nd_engine: graph H not captured / published: " + hk;
      return Status::INTERNAL;
    }
  for (int r = 0; r < h_->n_ranks; ++r) {
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)h_->ranks[r].device);
    if (!ck(cudaGraphLaunch((cudaGraphExec_t)gh[r]->exec, (cudaStream_t)h_->ranks[r].step_stream), "graph H", err))
      return Status::INTERNAL;
  }
  evp_rec_(4);
  Status s = advance(b, err);
  if (s != Status::OK) return s;
  // _record: the handover's persistent step events, alternating; the slot's previous holder is extracted first
  const int slot = ev_i_;
  ev_i_ ^= 1;
  Handle* old = ev_owner_[slot];
  if (old != nullptr && old != hd && !old->done) {   // (never in the 2b loop)
    s = extract(old, err);
    if (s != Status::OK) return s;
  }
  for (int r = 0; r < h_->n_ranks; ++r) {
    const FprEhRankState& rs = h_->ranks[r];
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    cudaEvent_t e = (cudaEvent_t)rs.step_event[slot];
    if (!ck(cudaEventRecord(e, (cudaStream_t)rs.step_stream), "step event", err)) return Status::INTERNAL;
    hd->ev[r] = e;
  }
  hd->ev_slot = slot;
  ev_owner_[slot] = hd;
  return Status::OK;
}

Status Engine::advance(const std::vector<StepBatch>& b, std::string* err) {   // D13._advance (k_advance per rank)
  qi_ ^= 1;
  const int64_t MR = (int64_t)NW * (cfg_.hsd.max_seqs + 1);
  for (int r = 0; r < h_->n_ranks && r < (int)b.size(); ++r) {
    const int64_t S = (int64_t)b[r].seqs.size();
    if (!S) continue;
    const FprEhRankState& rs = h_->ranks[r];
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    cudaStream_t ss = (cudaStream_t)rs.step_stream;
    const int64_t q0 = r * cur_.D_pad;
    static const std::vector<std::pair<int64_t, int64_t>> none;
    const auto& ds = r < (int)cur_.dseqs.size() ? cur_.dseqs[r] : none;
    bool slice = cfg_.adv_slice && (int64_t)ds.size() == S;
    for (int64_t j = 0; slice && j < S; ++j) slice = ds[j].first == j && ds[j].second == q0 + j;
    const int32_t* acc = nullptr;
    at::Tensor tmp;   // (keeps the gathered counts alive until k_advance is enqueued; stream-ordered reuse after)
    if (slice) {      // ENG_D13_ADV_SLICE: num_sampled[q0 : q0 + S] (int32 already: .to() is a no-op)
      acc = tp<const int32_t>(rs.num_sampled) + q0;
      stats_["adv_slice"]++;
    } else {          // the request index of each sequence -> qidx_d, then num_sampled.index_select (on the step stream)
      if (!ck(cudaEventSynchronize((cudaEvent_t)rs.qidx.ev[qi_]), "qidx set", err)) return Status::INTERNAL;
      int64_t* q = reinterpret_cast<int64_t*>(rs.qidx.h[qi_].ptr);
      std::fill(q, q + S, (int64_t)0);
      for (const auto& p : ds) q[p.first] = p.second;
      if (!ck(cudaMemcpyAsync(rs.qidx_d.ptr, q, S * 8, cudaMemcpyHostToDevice, ss), "qidx H2D", err) ||
          !ck(cudaEventRecord((cudaEvent_t)rs.qidx.ev[qi_], ss), "qidx event", err))
        return Status::INTERNAL;
      const c10::Device dev(c10::DeviceType::CUDA, (c10::DeviceIndex)rs.device);
      const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(ss, (c10::DeviceIndex)rs.device));
      at::Tensor ns_t = at::for_blob(rs.num_sampled.data, {MR}).options(at::TensorOptions().dtype(at::kInt).device(dev))
                            .target_device(dev).make_tensor();
      at::Tensor qd_t = at::for_blob(rs.qidx_d.ptr, {S}).options(at::TensorOptions().dtype(at::kLong).device(dev))
                            .target_device(dev).make_tensor();
      tmp = at::index_select(ns_t, 0, qd_t);
      acc = tmp.data_ptr<int32_t>();
    }
    const int32_t* b32 = reinterpret_cast<const int32_t*>(rs.rx_b32_d.ptr);
    FprRowexpAdvanceArgs a{};
    a.c = FprCall{ss, rs.device, 0};
    a.slot = b32 + rs.rx_o32[0];
    a.kind = b32 + rs.rx_o32[3];
    a.nrows = b32 + rs.rx_o32[2];
    a.start_res = tp<const int64_t>(rs.rx_start);
    a.accepted = acc;
    a.counts = tp<const int32_t>(rs.rx_counts);
    a.nc = tp<int32_t>(rs.rx_nc);
    a.s_max = rs.rx_S_max;
    if (rx_advance_(&a) != FPR_OK) {
      if (err) *err = std::string("nd_engine: rowexp.advance: ") + cudaGetErrorString(cudaPeekAtLastError());
      return Status::INTERNAL;
    }
  }
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ results
void Engine::results_from(Handle* h, const int32_t* const* ns, const int64_t* const* smp, const std::vector<StepBatch>* b) {   // nd_hist.cpp
  results_and_history(h->cur, h_->n_ranks, cfg_.k + 1, ns, smp, &reqs_, &h->res);
  // (v3.39) ND_ACCSTAT: read-only on ns / the step's batches. (jY) mixed_finish's Handle has no batches: it passes them
  if (acc_on_ != 0) acc_count_(h, b != nullptr ? *b : h->batches, ns);
}

void Engine::acc_count_(const Handle* h, const std::vector<StepBatch>& b, const int32_t* const* ns) {
  if (acc_on_ < 0) {
    const char* v = std::getenv("ND_ACCSTAT");
    acc_on_ = (v != nullptr && std::string(v) == "1") ? 1 : 0;
    const char* e = std::getenv("ND_ACCSTAT_EVERY");
    if (e != nullptr) acc_every_ = std::max<int64_t>(64, std::atoll(e));
    if (acc_on_ != 1) return;
  }
  const int K = (int)std::min<int64_t>(cfg_.k, 8);
  const d13::HsdStep& c = h->cur;
  int64_t nreq = 0;
  for (int r = 0; r < h_->n_ranks && r < (int)c.dseqs.size(); ++r) nreq += (int64_t)c.dseqs[r].size();
  if (nreq <= 0) return;
  const int nb = nreq <= 1 ? 0 : nreq <= 8 ? 1 : nreq <= 32 ? 2 : 3;
  int64_t counted = 0, acc_nopos = 0, acc_hist = 0;
  for (int r = 0; r < h_->n_ranks && r < (int)c.dseqs.size(); ++r) {
    if (ns[r] == nullptr) continue;
    for (const auto& sq : c.dseqs[r]) {
      const int64_t si = sq.first, qi = sq.second;
      const int a = std::max(0, std::min<int>(ns[r][qi], K + 1) - 1);   // accepted drafts of this request
      int64_t pos = -1;
      if (r < (int)b.size() && si >= 0 && si < (int64_t)b[r].seqs.size()) {   // (jY) + no batches
        const SeqEntry& se = b[r].seqs[si];
        pos = se.start;
        // (jZ) the request's context = its token history (prompt + output + native extractions; results_and_history above
        // has appended this step's): SeqEntry.start is not the context on native decode steps (core2b plans ahead of the
        // device; steady steps never read positions back)
        const auto it = reqs_.find(se.rid);
        if (it != reqs_.end() && (int64_t)it->second.hist.size() > pos) {
          pos = (int64_t)it->second.hist.size();
          ++acc_hist;
        }
      } else {
        ++acc_nopos;
      }
      const int cb = pos < 0 ? 0 : pos < 8192 ? 0 : pos < 65536 ? 1 : pos < 262144 ? 2 : 3;
      acc_req_[cb][nb]++;
      for (int j = 0; j < a && j < K; ++j) acc_pos_[cb][nb][j]++;
      ++counted;
    }
  }
  stats_["acc_requests"] += counted;
  acc_nopos_ += acc_nopos;
  acc_hist_ += acc_hist;
  if (stats_["acc_requests"] - acc_last_ < acc_every_) return;
  acc_last_ = stats_["acc_requests"];
  static const char* CB[4] = {"<8K", "8-64K", "64-262K", ">=262K"};
  static const char* NB[4] = {"1", "2-8", "9-32", ">32"};
  std::string line = "nd ACCSTAT cumulative requests " + std::to_string(acc_last_) + " (k " + std::to_string(K) +
                     ", no ctx " + std::to_string(acc_nopos_) + ", ctx from hist " + std::to_string(acc_hist_) +
                     "):";   // (jY) no ctx = binned <8K without a position; (jZ) ctx from hist = history > SeqEntry.start
  char buf[160];
  for (int ci = 0; ci < 4; ++ci)
    for (int bi = 0; bi < 4; ++bi) {
      const int64_t n = acc_req_[ci][bi];
      if (n == 0) continue;
      int64_t sum = 0;
      for (int j = 0; j < K; ++j) sum += acc_pos_[ci][bi][j];
      std::snprintf(buf, sizeof(buf), " | ctx %s reqs/step %s: n %lld tok/step %.3f pos", CB[ci], NB[bi], (long long)n,
                    1.0 + (double)sum / (double)n);
      line += buf;
      for (int j = 0; j < K; ++j) {
        std::snprintf(buf, sizeof(buf), " %.3f", (double)acc_pos_[ci][bi][j] / (double)n);
        line += buf;
      }
    }
  std::fprintf(stderr, "%s\n", line.c_str());
  std::fflush(stderr);
}

Status Engine::extract(Handle* h, std::string* err) {
  if (h->done) return Status::OK;
  std::array<const int32_t*, NW> ns{};
  std::array<const int64_t*, NW> smp{};
  if (h->deferred) {   // Steady.extract_deferred: the result copies queued at dispatch(N + 1)
    ResSet& set = res_[h->res_set];
    for (int r : h->act) {
      if (!ck(cudaEventSynchronize(set.ev[r]), "deferred result copies", err)) return Status::INTERNAL;
      ns[r] = set.ns[r];
      smp[r] = set.smp[r];
    }
  } else {             // D13._extract: wait the step; HSD.result's copies on the extraction streams (RES_OWNER: drafting ranks)
    for (int r = 0; r < h_->n_ranks; ++r)
      if (!ck(cudaEventSynchronize(h->ev[r]), "step events", err)) return Status::INTERNAL;
    ResSet& set = res_[NRING];
    const int64_t n = h->cur.n_req, K1 = cfg_.k + 1;
    std::vector<int> act;
    for (int r = 0; r < h_->n_ranks; ++r)
      if (!cfg_.res_owner || (r < (int)h->cur.dseqs.size() && !h->cur.dseqs[r].empty())) act.push_back(r);
    for (int r : act) {
      const FprEhRankState& rs = h_->ranks[r];
      const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
      cudaStream_t xs = (cudaStream_t)rs.xstream;
      if (!ck(cudaStreamWaitEvent(xs, h->ev[r], 0), "extract: xstream wait", err) ||
          !ck(cudaMemcpyAsync(set.ns[r], rs.num_sampled.data, n * 4, cudaMemcpyDeviceToHost, xs), "extract: ns", err) ||
          !ck(cudaMemcpy2DAsync(set.smp[r], K1 * 8, rs.sampled.data, rs.sampled.stride[0] * 8, K1 * 8, n,
                                cudaMemcpyDeviceToHost, xs), "extract: sampled", err) ||
          !ck(cudaEventRecord(set.ev[r], xs), "extract: record", err))
        return Status::INTERNAL;
    }
    for (int r : act) {
      if (!ck(cudaEventSynchronize(set.ev[r]), "extract: copies", err)) return Status::INTERNAL;
      ns[r] = set.ns[r];
      smp[r] = set.smp[r];
    }
  }
  results_from(h, ns.data(), smp.data());
  if (!h->deferred) {   // (v3.25) structured / grammar requests: never deferred (Steady.is_steady refuses them)
    const Status ds = structured_drafts_(h, err);
    if (ds != Status::OK) return ds;
  }
  h->done = true;
  return post_extract(h, err);
}

// Hooks.head_result (engine.py serve D7 / D11): every sequence of the step's batch (seq order si) whose rid is
// structured (Hooks.structured = Req.structured, as of this point: the next step's on_ctx already ran, as in D13) or has
// grammar rows in THIS step (h.grammar) and got tokens -> res.drafts[rid] = next_ids[r][si, 1:k+1]. next_ids still holds
// the step's ids: nd extracts N before graph H of N + 1 (dispatch), at wait / drain otherwise.
Status Engine::structured_drafts_(Handle* h, std::string* err) {
  return drafts_into_(h->batches, h->grammar, &h->res, err);
}

// (v3.26) the shared body: res[r] of the step's batches b (HSD's seq order si = next_ids rows), its grammar rows g
Status Engine::drafts_into_(const std::vector<StepBatch>& b, const std::vector<GrammarRows>& g, std::vector<StepResult>* resv,
                            std::string* err) {
  const int64_t K1 = cfg_.k + 1;
  for (int r = 0; r < h_->n_ranks && r < (int)b.size() && r < (int)resv->size(); ++r) {
    StepResult& res = (*resv)[(size_t)r];
    std::vector<std::pair<int64_t, Rid>> want;
    const auto& seqs = b[(size_t)r].seqs;
    for (size_t si = 0; si < seqs.size(); ++si) {
      const Rid& rid = seqs[si].rid;
      auto it = reqs_.find(rid);
      const bool st = (it != reqs_.end() && it->second.structured) || grammar_of(g, rid) != nullptr;
      if (!st) continue;
      bool tok = false;
      for (const auto& t : res.tokens) tok = tok || t.first == rid;
      if (tok) want.emplace_back((int64_t)si, rid);
    }
    if (want.empty()) continue;
    const FprEhRankState& rs = h_->ranks[r];
    if (rs.next_ids.shape[1] != K1) {
      if (err) *err = "nd_engine: next_ids is not [S_max, k + 1] (structured drafts)";
      return Status::INTERNAL;
    }
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    cudaStream_t xs = (cudaStream_t)rs.xstream;
    std::vector<int64_t> buf(want.size() * (size_t)K1);
    for (size_t j = 0; j < want.size(); ++j) {
      if (want[j].first >= rs.next_ids.shape[0]) {
        if (err) *err = "nd_engine: structured drafts: seq index beyond next_ids rows";
        return Status::INTERNAL;
      }
      const int64_t* src = reinterpret_cast<const int64_t*>(rs.next_ids.data) + want[j].first * K1;
      if (!ck(cudaMemcpyAsync(buf.data() + j * K1, src, (size_t)K1 * 8, cudaMemcpyDeviceToHost, xs), "structured drafts", err))
        return Status::INTERNAL;
    }
    if (!ck(cudaStreamSynchronize(xs), "structured drafts", err)) return Status::INTERNAL;
    for (size_t j = 0; j < want.size(); ++j) {
      std::vector<int32_t> d((size_t)(K1 - 1));
      for (int64_t q = 1; q < K1; ++q) d[(size_t)(q - 1)] = (int32_t)buf[j * K1 + q];
      res.drafts.emplace_back(want[j].second, std::move(d));
    }
    stats_["structured_drafts"] += (int64_t)want.size();
  }
  return Status::OK;
}

Status Engine::post_extract(Handle* h, std::string* err) {   // D13._post_extract
  ++n_extract_;
  if (n_extract_ % cfg_.err_every == 0)
    for (int r = 0; r < h_->n_ranks; ++r) {   // rowexp's cumulative ring-miss counters (a host sync each, as Python)
      const FprEhRankState& rs = h_->ranks[r];
      const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
      int64_t numel = 1;
      for (int d = 0; d < rs.rx_err.ndim; ++d) numel *= rs.rx_err.shape[d];
      std::vector<int32_t> e(numel);
      cudaStream_t xs = (cudaStream_t)rs.xstream;
      if (!ck(cudaMemcpyAsync(e.data(), rs.rx_err.data, numel * 4, cudaMemcpyDeviceToHost, xs), "rowexp err", err) ||
          !ck(cudaStreamSynchronize(xs), "rowexp err", err))
        return Status::INTERNAL;
      int64_t sum = 0;
      for (int32_t x : e) sum += x;
      if (sum != err_seen_[r]) {
        stats_["rowexp_miss"] += sum - err_seen_[r];
        err_seen_[r] = sum;
      }
    }
  if (inflight_ == h) inflight_ = nullptr;
  return Status::OK;
}

Status Engine::wait(Handle* hd, std::vector<StepResult>* out, std::string* err) {
  ScopeUs t_us_{this, "wait"};   // (v3.21, L0)
  if (!hd->done) {
    Status s = extract(hd, err);
    if (s != Status::OK) return s;
  }
  *out = hd->res;
  hd->consumed = true;
  if (inflight_ == hd) inflight_ = nullptr;
  return Status::OK;
}

Status Engine::drain(std::string* err) {
  ScopeUs t_us_{this, "drain"};   // (v3.21, L0)
  if (inflight_ != nullptr && !inflight_->done) {
    Status s = extract(inflight_, err);
    if (s != Status::OK) return s;
  }
  for (int r = 0; r < h_->n_ranks; ++r) {
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)h_->ranks[r].device);
    if (!ck(cudaStreamSynchronize((cudaStream_t)h_->ranks[r].step_stream), "drain step stream", err) ||
        !ck(cudaStreamSynchronize((cudaStream_t)h_->ranks[r].xstream), "drain xstream", err))
      return Status::INTERNAL;
  }
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ embedded steps (push / pull)
Engine::PushScalars Engine::push_scalars_() const {   // (v3.23)
  PushScalars p;
  p.rx_hi = rx_hi_;
  p.eg_i = eg_i_;
  p.eg_gen = eg_gen_;
  p.dl_i = dl_i_;
  p.d2h_i = d2h_i_;
  p.qi = qi_;
  p.hsd_hi = (uint8_t)host_->hsd().hi();
  p.step_serial = step_serial_;
  return p;
}

Status Engine::push_state_(std::string* err, const PushScalars* sc) {
  ScopeUs t_us_{this, "push_state"};   // (v3.21, L0)
  // set handoff (C): nd's alternation indices, prev-of map + serial (I), the natively extracted tokens (J) -> the engine
  std::vector<int64_t> Rs;
  for (const auto& kv : in_up_i_)
    if (Rs.empty() || Rs.back() != kv.first.first) Rs.push_back(kv.first.first);   // (std::map: sorted by (R, rank))
  const int W = h_->n_ranks;
  std::vector<uint8_t> in_i(Rs.size() * W, 0), up_i(Rs.size() * W, 0);
  for (size_t bi = 0; bi < Rs.size(); ++bi)
    for (int r = 0; r < W; ++r) {
      auto it = in_up_i_.find({Rs[bi], r});
      if (it == in_up_i_.end()) continue;
      in_i[bi * W + r] = it->second[0];
      up_i[bi * W + r] = it->second[1];
    }
  std::vector<FprEhPrevOf> prev;
  prev.reserve(prev_.size());
  for (const auto& kv : prev_) prev.push_back(FprEhPrevOf{kv.first.c_str(), kv.second.rank, kv.second.index, kv.second.serial});
  std::vector<FprEhHist> hist;
  push_hist(reqs_, native_drops_, &hist);   // J: pending tokens of live requests, then the native drops (n = -1)
  FprEhSetState st{};
  st.struct_size = (int32_t)sizeof(FprEhSetState);
  st.n_buckets = (int32_t)Rs.size();
  st.bucket_R = Rs.data();
  st.in_i = in_i.data();
  st.up_i = up_i.data();
  const PushScalars ps = sc != nullptr ? *sc : push_scalars_();   // (v3.23: a lazy push sends begin's snapshot)
  for (int r = 0; r < W && r < FPR_EH_MAX_RANKS; ++r) {
    st.rx_hi[r] = ps.rx_hi[r];
    st.engram_i[r] = ps.eg_i[r];
    st.engram_gen[r] = ps.eg_gen[r];
  }
  st.dl_i = ps.dl_i;
  st.d2h_i = ps.d2h_i;
  st.qi = ps.qi;
  st.hsd_hi = ps.hsd_hi;
  st.prev = prev.data();
  st.n_prev = (int64_t)prev.size();
  st.step_serial = ps.step_serial;
  st.hist = hist.data();
  st.n_hist = (int64_t)hist.size();
  int32_t rc = -1;
  if (!run_python_([&] { rc = h_->push_state(h_->ctx, &st); })) {
    if (err) *err = "nd_engine: push_state: the engine's main thread stopped";
    return Status::INTERNAL;
  }
  if (rc != 0) {
    if (err) *err = std::string("nd_engine: push_state: ") + h_->last_error(h_->ctx);
    return Status::INTERNAL;
  }
  for (auto& kv : reqs_) kv.second.hist_pending.clear();
  native_drops_.clear();
  return Status::OK;
}

Status Engine::embedded_begin(const StepHost& host, std::string* err) {
  evp_break_();   // (v3.38) the embedded step's GPU work sits between the native decode steps around it
  ScopeUs t_us_{this, "embedded_begin"};   // (v3.21, L0)
  // D13._fallback's order: drain (step N's results into the history) BEFORE the step's on_ctx (a request dropped and
  // re-admitted in this context must not keep N's tokens; tests/nd_hist_test.cpp "restore in the same context")
  Rid clash;
  if (!new_dropped_disjoint(host, &clash)) {
    if (err) *err = "nd_engine: a context drops and admits the same request: " + clash;
    return Status::BAD_ARG;
  }
  Status s = drain(err);
  if (s != Status::OK) return s;
  observe(host, false);   // (this StepHost reaches Python's on_ctx in the embedded step itself: no native drops)
  s = push_state_(err);
  if (s != Status::OK) return s;
  // D13._fallback's bookkeeping: an embedded step rewrites tables / descriptors, the next native step is not steady
  ++epoch_;
  r_empty_.fill(false);
  for (auto& x : own_last_) x.clear();
  prev_sig_.clear();
  inflight_ = last_prep_ = nullptr;
  stats_["embedded"]++;
  return Status::OK;
}

Status Engine::embedded_end(std::string* err) {
  ScopeUs t_us_{this, "embedded_end"};   // (v3.21, L0)
  FprEhSetState st{};
  st.struct_size = (int32_t)sizeof(FprEhSetState);
  int32_t rc = -1;
  if (!run_python_([&] { rc = h_->pull_state(h_->ctx, &st); })) {
    if (err) *err = "nd_engine: pull_state: the engine's main thread stopped";
    return Status::INTERNAL;
  }
  if (rc != 0) {
    if (err) *err = std::string("nd_engine: pull_state: ") + h_->last_error(h_->ctx);
    return Status::INTERNAL;
  }
  const int W = h_->n_ranks;
  for (int bi = 0; bi < st.n_buckets; ++bi)
    for (int r = 0; r < W; ++r)
      in_up_i_[{st.bucket_R[bi], r}] = {st.in_i[(size_t)bi * W + r], st.up_i[(size_t)bi * W + r]};
  for (int r = 0; r < W && r < FPR_EH_MAX_RANKS; ++r) {
    rx_hi_[r] = st.rx_hi[r];
    eg_i_[r] = st.engram_i[r];
    eg_gen_[r] = st.engram_gen[r];
  }
  dl_i_ = st.dl_i;
  d2h_i_ = st.d2h_i;
  qi_ = st.qi;
  host_->hsd().set_hi(st.hsd_hi);
  prev_.clear();   // (I) the engine's prev-of map after its step: (rank, seq index, producing serial)
  for (int64_t i = 0; i < st.n_prev; ++i)
    prev_[Rid(st.prev[i].rid)] = PrevOf{st.prev[i].rank, st.prev[i].index, st.prev[i].serial};
  step_serial_ = st.step_serial;
  // (v0.5) J deltas: the tokens the embedded step appended to eng.hist -> Req.hist
  const int64_t unknown = pull_hist(st.hist, st.n_hist, &reqs_);
  stats_["pull_hist_entries"] += st.n_hist;
  if (unknown) stats_["pull_hist_unknown_rid"] += unknown;
  // the two sides key HSD's per-slot temperature / seed cache differently (id(obj) vs request serial): re-read all
  const int64_t MR = (int64_t)NW * (cfg_.hsd.max_seqs + 1);
  host_->hsd().restore({}, {}, std::vector<uint64_t>(MR, 0));
  have_cur_ = false;   // hsd.cur is the embedded step's now: nd's next step prepares on the host
  return Status::OK;
}

// (jZr) core2b's on_ctx lazy replay: Req.hist minus its trailing hist_pending (nd_hist.cpp appends every native extraction
// to both; push_state hands hist_pending to Python and clears it), seed, structured. find only: never inserts
void Engine::replay_view(const std::vector<Rid>& rids, std::vector<ReplayEntry>* out) const {
  out->clear();
  out->reserve(rids.size());
  for (const auto& rid : rids) {
    ReplayEntry e;
    e.rid = rid;
    const auto it = reqs_.find(rid);
    if (it != reqs_.end()) {
      const Req& q = it->second;
      const int64_t n = (int64_t)q.hist.size() - (int64_t)q.hist_pending.size();
      if (n >= 0) {
        e.ids = q.hist.data();
        e.n = n;
        e.seed = q.sampling.seed;
        e.structured = q.structured;
        e.known = true;
      }
    }
    out->push_back(e);
  }
}

std::string Engine::stats_json() const {
  std::ostringstream o;
  o << "{\"nd_api\": " << FPR_ND_API_VERSION;
  bool first = false;
  for (const auto& kv : stats_) {
    o << (first ? "" : ", ") << "\"" << kv.first << "\": " << kv.second;
    first = false;
  }
  o << "}";
  return o.str();
}

}  // namespace nd
}  // namespace fpr
