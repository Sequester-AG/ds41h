// nd_mixed.cpp -- nd's IMixed2b (core2b/nd_api.h) for a NATIVE MIXED
// STEP. The contract of engine/d13.py D13._fallback + engine/engine.py Engine.step_mixed's head
//   can_head      every refusal, no device op, no state change
//   begin         observe, drain, push_state, D13._fallback's skip-state resets, D12 deltas, rowexp resolve, _fb_rows,
//                 the full tables, engine.bump_serial
//   [Python body: copies, StepContext, prologue, Engine._mixed_body(prep_head=False)]
//   prepare_head  Hooks.prepare_head(stT, rb): P0 builder + HSD._build + H2D per rank (after T on the step streams)
//   h1_launch     mixed_cxx_head.head_eager per rank
//   [Python: Engine._mixed_post(head_eager=False): sync]
//   finish        head_result + J, prev_seq, rowexp advance + device sync; no pull_state
// Status: BAD_ARG = refused before anything was done (the caller runs the step embedded); INTERNAL = fatal.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <set>
#include <tuple>

#include "../../../engine/mixed_cxx/fpr_cabi.h"   // (v3.11) FprCall for cabi()
#include "../../../kernels/pagecopy/pagecopy_cabi.h"
#include "../../../kernels/engram/engram_cabi.h"   // (v3.26) "engram.ring_copy"
#include "../../../kernels/rowexp/rowexp_cabi.h"
#include "../p0/head_build.h"
#include "nd_engine.h"
#include "../../include/kv.h"   // (v3.10) pool_id / pool_name / POOL_*

namespace fpr {
namespace nd {
namespace {

bool ck(cudaError_t e, const char* what, std::string* err) {
  if (e == cudaSuccess) return true;
  if (err) *err = std::string("nd_mixed: ") + what + ": " + cudaGetErrorString(e);
  return false;
}
template <class T>
T* tp(const FprEhTensor& t) { return reinterpret_cast<T*>(t.data); }
int64_t dsize(int32_t dt) {
  switch (dt) {
    case FPR_EH_U8: return 1;
    case FPR_EH_I32: case FPR_EH_F32: return 4;
    case FPR_EH_I64: return 8;
    case FPR_EH_BF16: case FPR_EH_F16: return 2;
  }
  return 0;
}
int64_t row_bytes(const FprEhTensor& t) {   // bytes of one row (dim 0) of a contiguous tensor
  int64_t n = dsize(t.dtype);
  for (int d = 1; d < t.ndim; ++d) n *= t.shape[d];
  return n;
}
std::string key3(const d13::HsdStep& c) {   // FPR_EH_GRAPH_H1's key text: _key_text((S_pad, D_pad, NT))
  return std::to_string(c.S_pad) + "|" + std::to_string(c.D_pad) + "|" + std::to_string(c.NT);
}
double us_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

// ------------------------------------------------------------------------------------------ staging (ext_init / ~Engine)
Status Engine::mixed_ext_init(std::string* err) {
  for (int r = 0; r < h_->n_ranks; ++r) {
    const FprEhRankState& rs = h_->ranks[r];
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    const int64_t S = std::max<int64_t>(rs.rx_S_max, 1);
    if (!ck(cudaHostAlloc((void**)&mx_start_h_[r], S * 8, cudaHostAllocPortable), "starts staging", err) ||
        !ck(cudaHostAlloc((void**)&mx_acc_h_[r], S * 4, cudaHostAllocPortable), "accepted staging", err) ||
        !ck(cudaMalloc((void**)&mx_acc_d_[r], S * 4), "accepted device", err))
      return Status::INTERNAL;
    const int64_t n = rs.h1_h_own.data != nullptr ? rs.h1_h_own.shape[0] : 0;   // HSD's S_max rows (h_own[r])
    mx_idx_nmax_ = std::max(mx_idx_nmax_, n);
    if (n) {
      for (int i = 0; i < 2; ++i)
        if (!ck(cudaHostAlloc((void**)&mx_idx_h_[r][i], n * 8, cudaHostAllocPortable), "H1 index staging", err) ||
            !ck(cudaEventCreateWithFlags(&mx_idx_ev_[r][i], cudaEventDisableTiming), "H1 index event", err))
          return Status::INTERNAL;
      if (!ck(cudaMalloc((void**)&mx_idx_d_[r], n * 8), "H1 index device", err)) return Status::INTERNAL;
    }
    if (cfg_.mixed_max_rows > 0) {   // (v3.4) IMixed2b::upload / set_vidx / set_tail_ix's own pinned sets (init: never mid-step)
      const int64_t W = h_->n_ranks;
      mxu_cap_ = p0::in_layout(cfg_.mixed_max_rows).bytes;
      mxv_cap_ = 8 * W * cfg_.mixed_max_rows + (4 * W + 7) / 8 * 8 + 4 * W;
      mxx_cap_ = cfg_.mixed_max_rows;
      for (int ro = 0; ro < 2; ++ro)
        for (int i = 0; i < 2; ++i)
          if (!ck(cudaHostAlloc((void**)&mxu_h_[ro][r][i], (size_t)mxu_cap_, cudaHostAllocPortable), "mixed upload set", err) ||
              !ck(cudaEventCreateWithFlags(&mxu_ev_[ro][r][i], cudaEventDisableTiming), "mixed upload event", err) ||
              !ck(cudaHostAlloc((void**)&mxv_h_[ro][r][i], (size_t)mxv_cap_, cudaHostAllocPortable), "mixed vidx set", err) ||
              !ck(cudaEventCreateWithFlags(&mxv_ev_[ro][r][i], cudaEventDisableTiming), "mixed vidx event", err))
            return Status::INTERNAL;
      for (int i = 0; i < 2; ++i)
        if (!ck(cudaHostAlloc((void**)&mxx_h_[r][i], (size_t)(mxx_cap_ * 8), cudaHostAllocPortable), "tail ix set", err) ||
            !ck(cudaEventCreateWithFlags(&mxx_ev_[r][i], cudaEventDisableTiming), "tail ix event", err))
          return Status::INTERNAL;
    }
    if (pc_copy_ != nullptr) {   // (v3.10) native page copies: cap = the largest paged / window pool of any rank
      if (pc_cap_ == 0)
        for (int q = 0; q < h_->n_ranks; ++q)
          for (int i = 0; i < h_->ranks[q].n_pools; ++i)
            if (pool_id(h_->ranks[q].pools[i].name) != POOL_STATE)
              pc_cap_ = std::max<int64_t>(pc_cap_, h_->ranks[q].pools[i].n_pages);
      for (int i = 0; i < 2; ++i)
        if (!ck(cudaHostAlloc((void**)&pch_[r][i], (size_t)(2 * pc_cap_ * 8), cudaHostAllocPortable), "pagecopy set", err) ||
            !ck(cudaEventCreateWithFlags(&pcev_[r][i], cudaEventDisableTiming), "pagecopy event", err))
          return Status::INTERNAL;
      if (!ck(cudaMalloc((void**)&pcd_[r], (size_t)(2 * pc_cap_ * 8)), "pagecopy ids", err) ||
          !ck(cudaEventCreateWithFlags(&pcdone_[r], cudaEventDisableTiming), "pagecopy done event", err))
        return Status::INTERNAL;
      stats_["pagecopy_cap_pages"] = pc_cap_;   // per rank: 2 pinned sets + 1 device buffer
      stats_["pagecopy_pinned_bytes_rank"] = 2 * 2 * pc_cap_ * 8;
      stats_["pagecopy_device_bytes_rank"] = 2 * pc_cap_ * 8;
      if (r == h_->n_ranks - 1)
        std::fprintf(stderr, "nd: pagecopy staging: cap %lld pages -> %lld B pinned + %lld B device per rank (%d ranks)\n",
                     (long long)pc_cap_, (long long)(4 * pc_cap_ * 8), (long long)(2 * pc_cap_ * 8), h_->n_ranks);
    }
  }
  return Status::OK;
}

void Engine::mixed_free() {
  for (int r = 0; r < NW; ++r) {
    if (mx_start_h_[r]) cudaFreeHost(mx_start_h_[r]);
    if (mx_acc_h_[r]) cudaFreeHost(mx_acc_h_[r]);
    if (mx_acc_d_[r]) cudaFree(mx_acc_d_[r]);
    if (mx_idx_d_[r]) cudaFree(mx_idx_d_[r]);
    for (int ro = 0; ro < 2; ++ro)
      for (int i = 0; i < 2; ++i) {
        if (mxu_h_[ro][r][i]) cudaFreeHost(mxu_h_[ro][r][i]);
        if (mxu_ev_[ro][r][i]) cudaEventDestroy(mxu_ev_[ro][r][i]);
        mxu_h_[ro][r][i] = nullptr;
        mxu_ev_[ro][r][i] = nullptr;
        if (mxv_h_[ro][r][i]) cudaFreeHost(mxv_h_[ro][r][i]);
        if (mxv_ev_[ro][r][i]) cudaEventDestroy(mxv_ev_[ro][r][i]);
        mxv_h_[ro][r][i] = nullptr;
        mxv_ev_[ro][r][i] = nullptr;
      }
    for (int i = 0; i < 2; ++i) {
      if (pch_[r][i]) cudaFreeHost(pch_[r][i]);
      if (pcev_[r][i]) cudaEventDestroy(pcev_[r][i]);
      pch_[r][i] = nullptr;
      pcev_[r][i] = nullptr;
    }
    if (pcd_[r]) cudaFree(pcd_[r]);
    if (pcdone_[r]) cudaEventDestroy(pcdone_[r]);
    pcd_[r] = nullptr;
    pcdone_[r] = nullptr;
    for (int i = 0; i < 2; ++i) {
      if (mxx_h_[r][i]) cudaFreeHost(mxx_h_[r][i]);
      if (mxx_ev_[r][i]) cudaEventDestroy(mxx_ev_[r][i]);
      mxx_h_[r][i] = nullptr;
      mxx_ev_[r][i] = nullptr;
    }
    for (int i = 0; i < 2; ++i) {
      if (mx_idx_h_[r][i]) cudaFreeHost(mx_idx_h_[r][i]);
      if (mx_idx_ev_[r][i]) cudaEventDestroy(mx_idx_ev_[r][i]);
    }
  }
}

void Engine::mixed_tick(const char* hook, double us) {
  const std::string k = std::string("mx_") + hook;
  stats_[k + "_us"] += (int64_t)us;
  stats_[k + "_n"] += 1;
  int64_t& m = stats_[k + "_max_us"];
  m = std::max<int64_t>(m, (int64_t)us);
}

// ------------------------------------------------------------------------------------------ (1) begin: D13._fallback's front
Status Engine::begin(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host, Ctx** c,
                     std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  if (c) *c = nullptr;
  evp_break_();   // (v3.38, ND_EVPROF) a native mixed step between two native decode steps: no decode gap across it
  if (!ext_ready_) {
    Status s0 = ext_init(err);
    if (s0 != Status::OK) return s0;
  }
  // ---- refusals (nothing done yet)
  if (mx_ctx_) {
    if (err) *err = "nd_mixed: a native mixed step is already open (begin without finish)";
    return Status::BAD_ARG;
  }
  if (!mixed_off_.empty()) {   // (a missing mixed piece: the step runs embedded, decode unaffected)
    if (err) *err = "nd_mixed: native mixed steps off: " + mixed_off_;
    return Status::BAD_ARG;
  }
  const int W = h_->n_ranks;
  if ((int)b.size() != W || (int)ct.size() != W || c == nullptr) {
    if (err) *err = "nd_mixed: begin wants one batch + one CompactTables per rank (and a context out)";
    return Status::BAD_ARG;
  }
  Rid clash;
  if (!new_dropped_disjoint(host, &clash)) {
    if (err) *err = "nd_mixed: a context drops and admits the same request: " + clash;
    return Status::BAD_ARG;
  }
  std::vector<d13::Deltas> dl(W);   // deltas the staging cannot take (Python: direct index_put), descriptor sizes
  std::vector<d13::Desc> desc(W);
  for (int r = 0; r < W; ++r) {
    const FprEhRankState& rs = h_->ranks[r];
    host_->deltas(r, ct[r], &dl[r]);
    if (dl[r].any && dl[r].nu > 0 && (dl[r].direct || dl[r].nu > rs.delta_cap)) {
      if (err) *err = "nd_mixed: D12 deltas above DELTA_CAP (the embedded step applies them directly)";
      return Status::BAD_ARG;
    }
    d13::D13Host::desc_of_compact(ct[r], &desc[r]);
    if ((int64_t)b[r].seqs.size() > rs.rx_S_max || std::max<int64_t>(b[r].n_rows, 1) > rs.rx_R_max) {
      if (err) *err = "nd_mixed: rowexp sizes (sequences / rows) above its staging";
      return Status::BAD_ARG;
    }
  }
  // ---- committed: from here a non-OK status is fatal. embedded_begin's order: drain (the in-flight step's StepResults
  // STAY on its handle: a later wait() returns them unchanged), observe (Python's on_ctx in the body sees this StepHost:
  // no native drops queued), push_state (the Python body reads hist / prev-of / serial / sets), then D13._fallback's
  // skip-state resets (a synchronous step rewrites tables / descriptors: the next native step is not steady)
  Status s = drain(err);
  if (s != Status::OK) return s == Status::BAD_ARG ? Status::INTERNAL : s;
  observe(host, host.on_ctx_lazy && !host.on_ctx_done);   // (jZr) lazy on_ctx: the body runs no on_ctx: drops queued for J
  pushed_step_ = false;
  if (!lazy_push_) {   // (v3.23, L3a) lazy: push_now(c) before any Python of this step; none -> accumulates
    s = push_state_(err);
    if (s != Status::OK) return Status::INTERNAL;
    pushed_step_ = true;
  } else {
    push_sc_ = push_scalars_();   // (the eager push's values: before this begin's dl / rx flips and serial bump)
    stats_["mixed_push_deferred"]++;
  }
  ++epoch_;
  r_empty_.fill(false);
  for (auto& x : own_last_) x.clear();
  prev_sig_.clear();
  inflight_ = last_prep_ = nullptr;
  std::unique_ptr<Ctx> cx(new Ctx());
  cx->b = b;
  cx->grammar = host.grammar;   // masks / acceptable at prepare_head, the drafts set at finish
  MixedBegin* out = &cx->mb;
  dl_i_ ^= 1;
  std::vector<std::vector<int64_t>> starts(W);
  for (int r = 0; r < W; ++r) {
    const FprEhRankState& rs = h_->ranks[r];
    const StepBatch& sb = b[r];
    const int64_t S = (int64_t)sb.seqs.size();
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    cudaStream_t ss = (cudaStream_t)rs.step_stream;
    const c10::Device dev(c10::DeviceType::CUDA, (c10::DeviceIndex)rs.device);
    // ---- self._deltas(r, ct): pinned set dl_i -> dl_d, dbt.index_put_ (= nd.cpp pre_full's block)
    const d13::Deltas& d = dl[r];
    if (d.any && d.nu > 0) {
      const int64_t nu = d.nu;
      if (!ck(cudaEventSynchronize((cudaEvent_t)rs.dl_rows.ev[dl_i_]), "dl set", err)) return Status::INTERNAL;
      auto* hr = reinterpret_cast<int64_t*>(rs.dl_rows.h[dl_i_].ptr);
      auto* hc = reinterpret_cast<int64_t*>(rs.dl_cols.h[dl_i_].ptr);
      auto* hp = reinterpret_cast<int32_t*>(rs.dl_pages.h[dl_i_].ptr);
      std::memcpy(hr, d.rows.data(), nu * sizeof(int64_t));
      std::memcpy(hc, d.cols.data(), nu * sizeof(int64_t));
      std::memcpy(hp, d.pages.data(), nu * sizeof(int32_t));
      if (!ck(cudaMemcpyAsync(rs.dl_rows_d.data, hr, nu * 8, cudaMemcpyHostToDevice, ss), "dl rows", err) ||
          !ck(cudaMemcpyAsync(rs.dl_cols_d.data, hc, nu * 8, cudaMemcpyHostToDevice, ss), "dl cols", err) ||
          !ck(cudaMemcpyAsync(rs.dl_pages_d.data, hp, nu * 4, cudaMemcpyHostToDevice, ss), "dl pages", err) ||
          !ck(cudaEventRecord((cudaEvent_t)rs.dl_rows.ev[dl_i_], ss), "dl event", err))
        return Status::INTERNAL;
      try {
        const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(ss, (c10::DeviceIndex)rs.device));
        auto mk = [&](void* ptr, at::IntArrayRef sz, at::IntArrayRef strd, at::ScalarType t) {
          return at::for_blob(ptr, sz).strides(strd).target_device(dev).options(at::TensorOptions().dtype(t).device(dev)).make_tensor();
        };
        at::Tensor dbt = mk(rs.rx_dbt.data, {rs.rx_dbt.shape[0], rs.rx_dbt.shape[1]}, {rs.rx_dbt.stride[0], 1}, at::kInt);
        c10::List<std::optional<at::Tensor>> idx;
        idx.push_back(mk(rs.dl_rows_d.data, {nu}, {1}, at::kLong));
        idx.push_back(mk(rs.dl_cols_d.data, {nu}, {1}, at::kLong));
        dbt.index_put_(idx, mk(rs.dl_pages_d.data, {nu}, {1}, at::kInt));
      } catch (const std::exception& e) {
        if (err) *err = std::string("nd_mixed: dbt.index_put_: ") + e.what();
        return Status::INTERNAL;
      }
    }
    // ---- rx.bind_outputs(**rx_own[r]); rx.upload(desc_of_compact(ct)); rx.begin(max(n_rows, 1))
    const uint8_t rxi = rx_hi_[r];
    if (!ck(cudaEventSynchronize((cudaEvent_t)rs.rx_h32.ev[rxi]), "rowexp set", err)) return Status::INTERNAL;
    int64_t n32 = 0;
    auto* h32 = reinterpret_cast<int32_t*>(rs.rx_h32.h[rxi].ptr);
    auto* h64 = reinterpret_cast<int64_t*>(rs.rx_h64.h[rxi].ptr);
    if (d13::DescStaging::pack_into(desc[r], rs.rx_S_max, rs.rx_ops_max, rs.rx_R_max, h32, h64, &n32, err) != Status::OK)
      return Status::INTERNAL;
    if (!ck(cudaMemcpyAsync(rs.rx_b32_d.ptr, h32, n32 * 4, cudaMemcpyHostToDevice, ss), "rowexp b32", err) ||
        !ck(cudaMemcpyAsync(rs.rx_b64_d.ptr, h64, 3 * rs.rx_S_max * 8, cudaMemcpyHostToDevice, ss), "rowexp b64", err) ||
        !ck(cudaEventRecord((cudaEvent_t)rs.rx_h32.ev[rxi], ss), "rowexp event", err))
      return Status::INTERNAL;
    rx_hi_[r] ^= 1;
    const int32_t* b32 = reinterpret_cast<const int32_t*>(rs.rx_b32_d.ptr);
    const int64_t* b64 = reinterpret_cast<const int64_t*>(rs.rx_b64_d.ptr);
    FprRowexpRingOpsArgs ro{};
    ro.c = FprCall{ss, rs.device, 0};
    ro.ring = tp<int32_t>(rs.rx_ring);
    ro.ops = b32 + rs.rx_o32[5];
    ro.counts = b32 + rs.rx_o32[4];
    FprRowexpExpandArgs ex{};
    ex.c = FprCall{ss, rs.device, 0};
    ex.slot = b32 + rs.rx_o32[0];
    ex.row0 = b32 + rs.rx_o32[1];
    ex.nrows = b32 + rs.rx_o32[2];
    ex.kind = b32 + rs.rx_o32[3];
    ex.start = b64;
    ex.replay = b64 + rs.rx_S_max;
    ex.cached = b64 + 2 * rs.rx_S_max;
    ex.counts = b32 + rs.rx_o32[4];
    ex.ring = tp<const int32_t>(rs.rx_ring);
    ex.dbt = tp<const int32_t>(rs.rx_dbt);
    ex.dbt_stride = rs.rx_dbt.stride[0];
    ex.nc = tp<const int32_t>(rs.rx_nc);
    ex.R = std::max<int64_t>(sb.n_rows, 1);
    ex.scratch = rs.rx_scratch_slot;
    ex.pos = tp<int64_t>(rs.rx_own_pos);
    ex.own = tp<int64_t>(rs.rx_own_own);
    ex.si = tp<int32_t>(rs.rx_own_si);
    ex.sl = tp<int32_t>(rs.rx_own_sl);
    ex.paged = tp<int64_t>(rs.rx_own_paged);
    ex.start_o = tp<int64_t>(rs.rx_start);
    ex.seq = tp<int32_t>(rs.rx_seq);
    ex.err = tp<int32_t>(rs.rx_err);
    if (rx_ring_ops_(&ro) != FPR_OK || rx_expand_(&ex) != FPR_OK) {
      if (err) *err = std::string("nd_mixed: rowexp ring_ops / expand: ") + cudaGetErrorString(cudaPeekAtLastError());
      return Status::INTERNAL;
    }
    // ---- starts.append(rx.out["start"][:max(S, 1)].to("cpu", non_blocking=False)[:S].tolist())
    const int64_t ns = std::max<int64_t>(S, 1);
    if (!ck(cudaMemcpyAsync(mx_start_h_[r], rs.rx_start.data, ns * 8, cudaMemcpyDeviceToHost, ss), "starts D2H", err) ||
        !ck(cudaStreamSynchronize(ss), "starts D2H", err))
      return Status::INTERNAL;
    starts[r].assign(mx_start_h_[r], mx_start_h_[r] + S);
  }
  s = host_->fallback_resolve(b, starts, &out->rb, err);   // _fb_rows
  if (s != Status::OK) return Status::INTERNAL;
  s = host_->fallback_tables(out->rb, ct, &out->tables, err);   // D13.tb.build (its D12 delta state)
  if (s != Status::OK) return Status::INTERNAL;
  for (const auto& sb : b)   // engine.bump_serial (Engine.step, after D13's resolve, before prepare_head): one step with
    if (!sb.seqs.empty()) {  // sequences. The Python prologue bumps its pushed copy the same way.
      ++step_serial_;
      break;
    }
  mx_ctx_ = std::move(cx);
  *c = mx_ctx_.get();
  stats_["mixed_steps"]++;
  mixed_tick("begin", us_since(t0));
  return Status::OK;
}

std::string Engine::h1_key(const std::vector<StepBatch>& b) const {   // (v3.1) for the worker's pre-begin guard check
  d13::HsdStep k;
  if (!hsd_key_of(b, &k, nullptr)) return "";
  return key3(k);
}

std::string Engine::h1_key_for(const std::vector<StepBatch>& b, const StepHost& host) const {
  std::string k = h1_key(b);
  if (k.empty() || host.grammar.empty()) return k;
  std::vector<std::vector<int32_t>> bst;
  std::vector<d13::HsdBitmask> bmk;
  return grammar_bitmasks(b, host.grammar, &bst, &bmk) > 0 ? k + "|bm" : k;
}

const std::vector<StepBatch>& Engine::resolved(const Ctx* c) const { return c->mb.rb; }
const std::vector<RankTables>& Engine::tables(const Ctx* c) const { return c->mb.tables; }

// ------------------------------------------------------------------------------------------ (2) can_head (no device op)
bool Engine::can_head(const std::vector<StepBatch>& b, const std::vector<CompactTables>& ct, const StepHost& host,
                      std::string* why) {
  auto no = [&](const std::string& w) {
    if (why) *why = w;
    return false;
  };
  if (!mixed_off_.empty()) return no("native mixed steps off: " + mixed_off_);
  if (!ext_ready_) {   // (v3.40, jX) lazy init here too, as begin() / dispatch() do: a prefill-only start (no pure decode
    // step yet) must not refuse every mixed step. Same thread as dispatch (core2b NativeWorker::dispatch_ runs can_head and
    // nd dispatch sequentially; Python's thread captures only inside run_python jobs the core thread is blocked on).
    if (!ext_fail_.empty()) return no("nd init failed (latched)");   // never retried here: a partial init is not undone
    for (int r = 0; r < h_->n_ranks; ++r) {   // no allocation while this thread's rank stream is capturing
      cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
      const cudaStream_t st = c10::cuda::getCurrentCUDAStream((c10::DeviceIndex)h_->ranks[r].device).stream();
      if (cudaStreamIsCapturing(st, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone)
        return no("nd not initialized (stream capturing)");
    }
    std::string e0;
    if (ext_init(&e0) != Status::OK) {
      ext_fail_ = e0.empty() ? std::string("?") : e0.substr(0, 200);
      stats_["ext_init_failed"] = 1;
      std::fprintf(stderr, "nd: ext_init from can_head failed (native mixed steps off): %s\n", ext_fail_.c_str());
      return no("nd init failed (latched)");
    }
    stats_["ext_init_from_can_head"] = 1;
  }
  if (mx_ctx_) return no("a native mixed step is already open");
  const int W = h_->n_ranks;
  if ((int)b.size() != W || (int)ct.size() != W) return no("one batch + one CompactTables per rank");
  Rid clash;
  if (!new_dropped_disjoint(host, &clash)) return no("a context drops and admits the same request: " + clash);
  // grammar rows: Hooks._bitmasks of the step -> H1's "|bm" variant (sampled masks) or the plain key
  // (rows without a sampled mask: the same run as Python's). Both only with the bitmask variant on (ENG_HEAD_BM_GRAPH +
  // HSD's masks exported: MixedHead admits grammar steps only then)
  bool bm = false;
  if (!host.grammar.empty()) {
    if (!bm_ready_()) return no("grammar rows (no bitmask H1: ENG_HEAD_BM_GRAPH off / HSD masks not exported)");
    std::vector<std::vector<int32_t>> bst;
    std::vector<d13::HsdBitmask> bmk;
    const int rc = grammar_bitmasks(b, host.grammar, &bst, &bmk);
    if (rc < 0) return no("grammar rows (malformed: Python raises)");
    bm = rc > 0;
  }
  for (int r = 0; r < W; ++r) {   // forced / score sequences: Python's GateHead (hsd_on, fill) decides their head
    const CompactTables& t = ct[r];
    for (int64_t x : t.forced)
      if (x != 0) return no("forced sequences (GateHead)");
    for (int64_t x : t.score)
      if (x != 0) return no("score sequences (GateHead)");
  }
  // (v3.26: structured requests are native: finish reads their next drafts back, as head_result)
  d13::HsdStep k;
  if (!hsd_key_of(b, &k, why)) return false;   // rows outside HSD's vectorized layout / no bucket
  const std::string key = key3(k) + (bm ? "|bm" : "");
  for (int r = 0; r < W; ++r) {
    const FprEhGraph* g = graph(key, r, 0);
    if (g == nullptr || g->kind != FPR_EH_GRAPH_H1)
      return no("H1 head graph not published for key " + key + " (the embedded step captures it)");
  }
  if (h_->ranks[0].h1_h_own.data == nullptr) return no("handover without h1_h_own");
  return true;
}

// ------------------------------------------------------------------------------------------ (3) Hooks.prepare_head(stT, rb)
Status Engine::mixed_prepare_head(const std::vector<StepBatch>& b, const std::vector<const std::vector<int64_t>*>& toks,
                                  MixedHead* out, std::string* err, const std::vector<GrammarRows>* g) {
  const auto t0 = std::chrono::steady_clock::now();
  // the P0 builder's snapshot from nd's state: StepHost's sampling per request = samp_ctx (vLLM's dict is not used in 2b);
  // no gate / grammar rows (can_head refuses them)
  p0::HeadSnap snap;
  for (const auto& sb : b)
    for (const auto& se : sb.seqs) {
      if (snap.rids.count(se.rid)) continue;
      p0::HeadRid e;
      auto it = reqs_.find(se.rid);
      if (it != reqs_.end()) {
        e.has_ctx = true;
        e.ctx.temperature = it->second.sampling.temperature;
        e.ctx.seed = it->second.sampling.seed;
      }
      int64_t pr = -1, pi = -1;
      e.dev_prev = prev_of(se.rid, &pr, &pi);
      const GrammarRows* gr = g != nullptr ? grammar_of(*g, se.rid) : nullptr;   // (v3.26) int(grammar[rid][1])
      e.has_grammar = gr != nullptr;
      e.grammar = gr != nullptr ? gr->num_acceptable : 0;
      snap.rids[se.rid] = e;
    }
  std::vector<std::vector<int32_t>> bst;   // Hooks._bitmasks(rb): the caller's masks
  std::vector<d13::HsdBitmask> bmk;
  const int brc = (g == nullptr || g->empty()) ? 0 : grammar_bitmasks(b, *g, &bst, &bmk);
  if (brc < 0 || (brc > 0 && !bm_ready_())) {
    if (err) *err = "nd_mixed: grammar masks without the bitmask variant after begin (mixed_can_head should have refused)";
    return Status::INTERNAL;
  }
  snap.grammar_any = g != nullptr && !g->empty();
  snap.bm_ok = true;   // (masks are this function's: the builder's bitmask refusal is the pre-v3.26 rule)
  const p0::HeadOut ho = p0::build_prepare_head(snap, b, toks);
  if (ho.status != p0::HeadStatus::OK) {   // (after the step's device work: can_head should have refused it)
    if (err) *err = std::string("nd_mixed: prepare_head ") + p0::head_status_name(ho.status) + " after begin";
    return Status::INTERNAL;
  }
  std::vector<std::vector<d13::HsdState>> states(b.size());
  for (size_t r = 0; r < b.size(); ++r)
    for (const auto& se : b[r].seqs) {
      const p0::HeadState& h = ho.states.at(se.rid);
      d13::HsdState x;
      x.temperature = h.v.temperature;
      x.seed = h.v.seed;
      x.drafts_valid = h.v.drafts_valid;
      x.has_num_acceptable = h.v.has_num_acceptable;
      x.num_acceptable = h.v.num_acceptable;
      auto it = reqs_.find(se.rid);   // the slot's state object identity: samp_ctx[rid] lives per request (= head())
      x.ident = (h.src == p0::HeadSrc::CTX && it != reqs_.end()) ? (uint64_t)it->second.serial : 0;
      states[r].push_back(x);
    }
  stats_["mx_resume_seqs"] += ho.n_resume;
  stats_["mx_resume_host_drafts"] += ho.n_host;
  stats_["mx_decode_seqs_seen"] += ho.n_dec;
  // HSD._build into blob set hi (its copy two prepares ago done) + HSD._upload (= head()'s tail)
  d13::HsdPrep& hp = host_->hsd();
  const int hi = hp.hi();
  for (int r = 0; r < h_->n_ranks; ++r)
    if (!ck(cudaEventSynchronize((cudaEvent_t)h_->ranks[r].hsd_hblob.ev[hi]), "HSD blob set", err)) return Status::INTERNAL;
  d13::HsdStep c;
  Status s = host_->hsd_prepare(b, states, brc > 0 ? &bmk : nullptr, &c, err);
  if (s != Status::OK) return Status::INTERNAL;
  if (c.v0) {
    if (err) *err = "nd_mixed: HSD rows outside the vectorized layout (mixed_can_head should have refused)";
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
    if (brc > 0 &&   // (v3.26) HSD._upload's mask part, as head(): _bm_d[r][:T] <- _bm_h[:T] on the step stream
        !ck(cudaMemcpyAsync(h_->hsd_bm_d[r].data, h_->hsd_bm_h.data, (size_t)(c.T * d13::HsdPrep::BM_WORDS * 4),
                            cudaMemcpyHostToDevice, ss), "HSD bitmask H2D", err))
      return Status::INTERNAL;
  }
  if (brc > 0) stats_["mx_bm_steps"]++;
  host_->hsd_uploaded();
  cur_ = c;
  have_cur_ = true;
  out->cur = c;
  out->h1_key = key3(c) + (brc > 0 ? "|bm" : "");
  mixed_tick("prepare_head", us_since(t0));
  return Status::OK;
}

Status Engine::prepare_head(Ctx* c, std::string* err) {
  if (c == nullptr || c != mx_ctx_.get() || c->headed) {
    if (err) *err = "nd_mixed: prepare_head without its open step (or twice)";
    return Status::INTERNAL;
  }
  // T's host ids as engine.py step_mixed builds them: tb_tail.build(tb[r], [tables[r].tokens[i] for i in rb[r].tail_rows]);
  // the P0 builder reads them only when the T table has rows (tab.n)
  const int W = h_->n_ranks;
  std::vector<std::vector<int64_t>> tt(W);
  std::vector<const std::vector<int64_t>*> toks(W, nullptr);
  for (int r = 0; r < W && r < (int)c->mb.rb.size(); ++r) {
    const StepBatch& rb = c->mb.rb[r];
    const RankTables& tab = c->mb.tables[r];
    tt[r].reserve(rb.tail_rows.size());
    for (int32_t i : rb.tail_rows) {
      if (i < 0 || i >= (int64_t)tab.tokens.size()) {
        if (err) *err = "nd_mixed: a tail row outside the full tables' tokens";
        return Status::INTERNAL;
      }
      tt[r].push_back(tab.tokens[i]);
    }
    if (!tt[r].empty()) toks[r] = &tt[r];
  }
  Status s = mixed_prepare_head(c->mb.rb, toks, &c->mh, err, &c->grammar);
  if (s != Status::OK) return Status::INTERNAL;
  c->headed = true;
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ (4) the H1 head of rank r
Status Engine::mixed_h1_launch(const MixedHead& mh, int r, const FprEhMixedBucketRank& t, const StepBatch& tail_b,
                               std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  const FprEhRankState& rs = h_->ranks[r];
  const FprEhGraph* g = graph(mh.h1_key, r, 0);
  if (g == nullptr || g->kind != FPR_EH_GRAPH_H1) {
    if (err) *err = "nd_mixed: H1 graph vanished after mixed_can_head: " + mh.h1_key;
    return Status::INTERNAL;
  }
  const int64_t n_s = r < (int)mh.cur.n_s.size() ? mh.cur.n_s[r] : 0;
  const int64_t n_t = r < (int)mh.cur.n_t.size() ? mh.cur.n_t[r] : 0;
  if ((int64_t)tail_b.sample_rows.size() < n_s || n_s > mx_idx_nmax_ || n_t > t.aux.shape[0] ||
      t.h.dtype != FPR_EH_BF16 || rs.h1_h_own.data == nullptr) {
    if (err) *err = "nd_mixed: H1 layout (sample rows / aux rows / dtypes) is not the eager head's";
    return Status::INTERNAL;
  }
  // (v3.1) mixed_cxx_head._rows' contiguity: T.h [R_T, hidden] row-major, T.aux rows packed (the gather / copy assume it)
  auto packed = [](const FprEhTensor& x) {
    int64_t want = 1;
    for (int d = x.ndim - 1; d >= 0; --d) {
      if (x.shape[d] > 1 && x.stride[d] != want) return false;
      want *= x.shape[d];
    }
    return true;
  };
  if (t.h.ndim != 2 || !packed(t.h) || !packed(t.aux)) {
    if (err) *err = "nd_mixed: H1 sources not contiguous (T.h / T.aux strides)";
    return Status::INTERNAL;
  }
  const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
  cudaStream_t ss = (cudaStream_t)rs.step_stream;
  if (n_s) {   // (a) h_own[:n_s] = T.h[sample_rows[:n_s]]: the index H2D (2 sets), then index_select into h_own
    const int i = mx_idx_i_[r];
    if (!ck(cudaEventSynchronize(mx_idx_ev_[r][i]), "H1 index set", err)) return Status::INTERNAL;
    int64_t* hb = mx_idx_h_[r][i];
    for (int64_t j = 0; j < n_s; ++j) hb[j] = tail_b.sample_rows[j];
    if (!ck(cudaMemcpyAsync(mx_idx_d_[r], hb, n_s * 8, cudaMemcpyHostToDevice, ss), "H1 index H2D", err) ||
        !ck(cudaEventRecord(mx_idx_ev_[r][i], ss), "H1 index event", err))
      return Status::INTERNAL;
    mx_idx_i_[r] = (uint8_t)(i ^ 1);
    try {
      const c10::Device dev(c10::DeviceType::CUDA, (c10::DeviceIndex)rs.device);
      const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(ss, (c10::DeviceIndex)rs.device));
      const auto bf = at::TensorOptions().dtype(at::kBFloat16).device(dev);
      const int64_t hid = t.h.shape[1];
      at::Tensor src = at::for_blob(t.h.data, {t.h.shape[0], hid}).options(bf).target_device(dev).make_tensor();
      at::Tensor dst = at::for_blob(rs.h1_h_own.data, {n_s, hid}).options(bf).target_device(dev).make_tensor();
      at::Tensor ix = at::for_blob(mx_idx_d_[r], {n_s}).options(at::TensorOptions().dtype(at::kLong).device(dev))
                          .target_device(dev).make_tensor();
      at::index_select_out(dst, src, 0, ix);
    } catch (const std::exception& e) {
      if (err) *err = std::string("nd_mixed: H1 gather: ") + e.what();
      return Status::INTERNAL;
    }
  }
  if (n_t && t.aux.data != g->aux_dst &&   // (b) drafter aux[:n_t] = T.aux[:n_t] (skipped when the same buffer)
      !ck(cudaMemcpyAsync(g->aux_dst, t.aux.data, n_t * row_bytes(t.aux), cudaMemcpyDeviceToDevice, ss), "H1 aux", err))
    return Status::INTERNAL;
  if (!ck(cudaGraphLaunch((cudaGraphExec_t)g->exec, ss), "H1 graph", err)) return Status::INTERNAL;   // (c)
  mixed_tick("h1_launch", us_since(t0));
  return Status::OK;
}

Status Engine::h1_launch(Ctx* c, int r, const FprEhMixedBucketRank& t, const StepBatch& tail_b, std::string* err) {
  if (c == nullptr || c != mx_ctx_.get() || !c->headed || r < 0 || r >= h_->n_ranks) {
    if (err) *err = "nd_mixed: h1_launch without prepare_head (or a bad rank)";
    return Status::INTERNAL;
  }
  Status s = mixed_h1_launch(c->mh, r, t, tail_b, err);
  if (s != Status::OK) return Status::INTERNAL;
  ++c->launched;
  return Status::OK;
}

Status Engine::devtok(Ctx* c, int r, const FprEhMixedBucketRank& b, const std::vector<int64_t>& rows,
                      const std::vector<int64_t>& src, const std::vector<std::array<int64_t, 4>>& mig, std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  if (c == nullptr || c != mx_ctx_.get() || r < 0 || r >= h_->n_ranks || b.rank != r) {
    if (err) *err = "nd_mixed: devtok outside an open step (or a bucket view of another rank)";
    return Status::INTERNAL;
  }
  if (!fpr_eh_mixed_current(h_, &b)) {
    if (err) *err = std::string("nd_mixed: devtok on a stale bucket view ") + b.key;
    return Status::INTERNAL;
  }
  if (b.in_d.ptr == nullptr || b.in_off_tok < 0 || b.in_off_tok % 8 || b.in_off_tok + b.R * 8 > b.in_d.bytes) {
    if (err) *err = std::string("nd_mixed: bucket ") + b.key + " has no tok view in in_d";
    return Status::INTERNAL;
  }
  std::vector<DevtokMig> mg;
  mg.reserve(mig.size());
  for (const auto& g : mig) mg.push_back(DevtokMig{g[0], g[1], (int32_t)g[2], g[3]});
  int64_t* tok = reinterpret_cast<int64_t*>(static_cast<char*>(b.in_d.ptr) + b.in_off_tok);
  if (devtok_issue(r, tok, b.R, rows, src, mg, err) != Status::OK) return Status::INTERNAL;
  stats_["mixed_devtok_rows"] += (int64_t)rows.size();
  stats_["mixed_devtok_mig"] += (int64_t)mig.size();
  mixed_tick("devtok", std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  return Status::OK;
}

Status Engine::upload(Ctx* c, int r, const FprEhMixedBucketRank& b, const p0::InputsOut& io, std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  auto bad = [&](const std::string& m) {
    if (err) *err = "nd_mixed: upload: " + m;
    return Status::INTERNAL;
  };
  if (c == nullptr || c != mx_ctx_.get() || r < 0 || r >= h_->n_ranks || b.rank != r)
    return bad("outside an open step (or a bucket view of another rank)");
  if (!fpr_eh_mixed_current(h_, &b)) return bad(std::string("stale bucket view ") + b.key);
  const int ro = b.role == FPR_EH_MIXED_T ? 1 : 0;
  const int64_t nb = (int64_t)io.blob.size();
  if (mxu_cap_ == 0) return bad("no staging (config mixed_max_rows 0)");
  if (nb > mxu_cap_) return bad("blob larger than config mixed_max_rows' layout");
  if (nb != p0::in_layout(b.R).bytes || b.in_d.ptr == nullptr || nb != b.in_d.bytes)
    return bad(std::string("blob / in_d size mismatch for ") + b.key);
  std::vector<DevtokMig> mg;
  for (const p0::MigCopy& g : io.mig) mg.push_back(DevtokMig{g.row0, g.n_rows, g.r0, g.si0});
  if (devtok_check(r, b.R, io.rows, io.src, mg, err) != Status::OK) return Status::INTERNAL;
  if (b.in_off_tok != p0::in_layout(b.R).off[p0::IN_TOK]) return bad("in_off_tok != in_layout's tok offset");
  const FprEhRankState& rs = h_->ranks[r];
  {
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    cudaStream_t ss = (cudaStream_t)rs.step_stream;
    const int i = mxu_i_[ro][r];
    if (!ck(cudaEventSynchronize(mxu_ev_[ro][r][i]), "mixed upload set", err)) return Status::INTERNAL;   // two steps ago
    std::memcpy(mxu_h_[ro][r][i], io.blob.data(), (size_t)nb);
    if (!ck(cudaMemcpyAsync(b.in_d.ptr, mxu_h_[ro][r][i], (size_t)nb, cudaMemcpyHostToDevice, ss), "mixed upload H2D", err) ||
        !ck(cudaEventRecord(mxu_ev_[ro][r][i], ss), "mixed upload event", err))
      return Status::INTERNAL;
    mxu_i_[ro][r] = (uint8_t)(i ^ 1);
  }
  int64_t* tok = reinterpret_cast<int64_t*>(static_cast<char*>(b.in_d.ptr) + b.in_off_tok);
  if (devtok_issue(r, tok, b.R, io.rows, io.src, mg, err) != Status::OK) return Status::INTERNAL;
  stats_["mixed_uploads"] += 1;
  stats_["mixed_upload_bytes"] += nb;
  mixed_tick("upload", std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  return Status::OK;
}

Status Engine::vidx_issue(int r, const FprEhMixedBucketRank& b, const p0::VidxOut& v, std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  auto bad = [&](const std::string& m) {
    if (err) *err = "nd: vidx_issue: " + m;
    return Status::BAD_ARG;
  };
  if (h_ == nullptr || r < 0 || r >= h_->n_ranks || b.rank != r) return bad("a bad rank / a bucket view of another rank");
  if (!fpr_eh_mixed_current(h_, &b)) return bad(std::string("stale bucket view ") + b.key);
  const int64_t W = h_->n_ranks, R = b.R, c8 = (4 * W + 7) / 8 * 8;
  const int64_t off_nv = 8 * W * R, off_nd = off_nv + c8, nb = off_nd + 4 * W;   // (= Engine._vidx_alloc's nbytes)
  if (b.vidx_d.ptr == nullptr) return bad(std::string("no vidx_d (v0.6 not allocated) for ") + b.key);
  if (b.vidx_d.bytes != nb) return bad(std::string("vidx_d bytes != the v0.6 layout for ") + b.key);
  if (mxv_cap_ < nb) return bad("vidx layout larger than config mixed_max_rows'");
  if ((int64_t)v.vidx.size() != v.vcount || v.vcount > W * R || (int64_t)v.nv.size() != W ||
      (v.has_nd && (int64_t)v.nd.size() != W))
    return bad("VidxOut sizes");
  const int ro = b.role == FPR_EH_MIXED_T ? 1 : 0;
  const FprEhRankState& rs = h_->ranks[r];
  const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
  cudaStream_t ss = (cudaStream_t)rs.step_stream;
  const int i = mxv_i_[ro][r];
  if (!ck(cudaEventSynchronize(mxv_ev_[ro][r][i]), "vidx set", err)) return Status::INTERNAL;
  uint8_t* h = mxv_h_[ro][r][i];
  std::memset(h, 0, (size_t)nb);                         // the whole layout is written: deterministic bytes past vcount
  std::memcpy(h, v.vidx.data(), (size_t)v.vcount * 8);
  std::memcpy(h + off_nv, v.nv.data(), (size_t)W * 4);
  if (v.has_nd) std::memcpy(h + off_nd, v.nd.data(), (size_t)W * 4);
  if (!ck(cudaMemcpyAsync(b.vidx_d.ptr, h, (size_t)nb, cudaMemcpyHostToDevice, ss), "vidx H2D", err) ||
      !ck(cudaEventRecord(mxv_ev_[ro][r][i], ss), "vidx event", err))
    return Status::INTERNAL;
  mxv_i_[ro][r] = (uint8_t)(i ^ 1);
  stats_["vidx_issue"] += 1;
  mixed_tick("vidx_issue", std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  return Status::OK;
}

Status Engine::set_vidx(Ctx* c, int r, const FprEhMixedBucketRank& b, const p0::VidxOut& v, std::string* err) {
  if (c == nullptr || c != mx_ctx_.get()) {
    if (err) *err = "nd_mixed: set_vidx outside an open step";
    return Status::INTERNAL;
  }
  return vidx_issue(r, b, v, err) == Status::OK ? Status::OK : Status::INTERNAL;
}

Status Engine::set_tail_ix(Ctx* c, int r, const std::vector<int64_t>& ix, std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  auto bad = [&](const std::string& m) {
    if (err) *err = "nd_mixed: set_tail_ix: " + m;
    return Status::INTERNAL;
  };
  if (c == nullptr || c != mx_ctx_.get() || r < 0 || r >= h_->n_ranks) return bad("outside an open step (or a bad rank)");
  const FprEhRankState& rs = h_->ranks[r];
  const int64_t n = (int64_t)ix.size();
  if (rs.tail_ix.data == nullptr || rs.tail_ix.ndim != 1 || rs.tail_ix.dtype != FPR_EH_I64)
    return bad("no tail_ix (v0.6 not allocated)");
  if (n > rs.tail_ix.shape[0] || n > mxx_cap_) return bad("ix longer than tail_ix / config mixed_max_rows");
  if (!n) return Status::OK;
  const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
  cudaStream_t ss = (cudaStream_t)rs.step_stream;
  const int i = mxx_i_[r];
  if (!ck(cudaEventSynchronize(mxx_ev_[r][i]), "tail ix set", err)) return Status::INTERNAL;
  std::memcpy(mxx_h_[r][i], ix.data(), (size_t)n * 8);
  if (!ck(cudaMemcpyAsync(rs.tail_ix.data, mxx_h_[r][i], (size_t)n * 8, cudaMemcpyHostToDevice, ss), "tail ix H2D", err) ||
      !ck(cudaEventRecord(mxx_ev_[r][i], ss), "tail ix event", err))
    return Status::INTERNAL;
  mxx_i_[r] = (uint8_t)(i ^ 1);
  stats_["mixed_set_tail_ix"] += 1;
  mixed_tick("set_tail_ix", std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  return Status::OK;
}

bool Engine::prev_of(const Ctx* c, const Rid& rid, int32_t* rank, int64_t* seq) const {
  if (c == nullptr || c != mx_ctx_.get()) return false;
  int64_t r = -1, s = -1;
  if (!prev_of(rid, &r, &s)) return false;   // (the engine's own map + serial check)
  *rank = (int32_t)r;
  *seq = s;
  return true;
}

Status Engine::sync(Ctx* c, std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  if (c == nullptr || c != mx_ctx_.get()) {
    if (err) *err = "nd_mixed: sync outside an open step";
    return Status::INTERNAL;
  }
  for (int r = 0; r < h_->n_ranks; ++r) {   // Engine.sync(): torch.cuda.synchronize(d) for d in devs
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)h_->ranks[r].device);
    if (!ck(cudaDeviceSynchronize(), "mixed step sync", err)) return Status::INTERNAL;
  }
  mixed_tick("sync", std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  return Status::OK;
}

bool Engine::page_copies_ok(const std::vector<PageCopyBatch>& groups, std::string* why) const {
  auto no = [&](const std::string& m) {
    if (why) *why = "page copies: " + m;
    return false;
  };
  if (h_ == nullptr || pc_copy_ == nullptr || pc_cap_ == 0) return no("no native pagecopy (ext / C ABI / staging)");
  {  // no (pool, rank, page) is both a source and a destination of the batch, no
     // destination repeats: the groups on different destination streams are then order-free (else Python's copies)
    std::set<std::tuple<int32_t, int32_t, int64_t>> srcs, dsts;
    for (const PageCopyBatch& g : groups)
      for (size_t i = 0; i < g.src_pages.size() && i < g.dst_pages.size(); ++i) {
        srcs.insert(std::make_tuple((int32_t)g.pool, (int32_t)g.src_rank, (int64_t)g.src_pages[i]));
        if (!dsts.insert(std::make_tuple((int32_t)g.pool, (int32_t)g.dst_rank, (int64_t)g.dst_pages[i])).second)
          return no("a destination page repeats in the batch");
      }
    for (const auto& d : dsts)
      if (srcs.count(d)) return no("a page is both a source and a destination of the batch");
  }
  for (const PageCopyBatch& g : groups) {
    if (g.pool == POOL_STATE) {   // + the Engram ring follow (engine.copy_state_slots) natively
      if (eg_ring_copy_ == nullptr) return no("a state-pool group (no engram.ring_copy: the Engram ring follow is Python's)");
      if (!cfg_.peer_map) return no("a state-pool group with ENG_PEER_MAP off (the ring copy's P2P reads)");
      if (g.src_rank >= 0 && g.src_rank < h_->n_ranks && g.dst_rank >= 0 && g.dst_rank < h_->n_ranks) {
        if (!peer_ok_[(size_t)g.dst_rank][(size_t)g.src_rank]) return no("a state-pool group without P2P access");
        const FprEhEngramRank& es = h_->engram[g.src_rank];
        const FprEhEngramRank& ed = h_->engram[g.dst_rank];
        if (es.ring == nullptr || ed.ring == nullptr || es.nring <= 0 || ed.nring <= 0)
          return no("a state-pool group without the exported Engram rings");
        for (size_t i = 0; i < g.src_pages.size() && i < g.dst_pages.size(); ++i)
          if (g.src_pages[i] >= es.nring || g.dst_pages[i] >= ed.nring) return no("a state slot beyond the Engram ring");
      }
    }
    if (g.src_rank < 0 || g.src_rank >= h_->n_ranks || g.dst_rank < 0 || g.dst_rank >= h_->n_ranks) return no("rank");
    const int64_t n = (int64_t)g.src_pages.size();
    if ((int64_t)g.dst_pages.size() != n || n > pc_cap_) return no("group size");
    if (g.page_bytes <= 0 || g.page_bytes % 16) return no("page_bytes");
    for (int side = 0; side < 2; ++side) {
      const FprEhRankState& rs = h_->ranks[side ? g.dst_rank : g.src_rank];
      int found = -1;
      for (int i = 0; i < rs.n_pools; ++i)
        if (pool_id(rs.pools[i].name) == g.pool) found = i;
      if (found < 0 || rs.pools[found].base == nullptr) return no(std::string("pool ") + pool_name(g.pool) + " missing");
      if (rs.pools[found].page_bytes != g.page_bytes) return no("page_bytes != the pool's");
      const std::vector<int64_t>& ids = side ? g.dst_pages : g.src_pages;
      for (int64_t x : ids)
        if (x < 0 || x >= rs.pools[found].n_pages) return no("page id out of range");
    }
  }
  return true;
}

Status Engine::page_copies(Ctx* c, const std::vector<PageCopyBatch>& groups, std::string* err) {
  if (c == nullptr || c != mx_ctx_.get()) {
    if (err) *err = "nd_mixed: page_copies outside an open step";
    return Status::INTERNAL;
  }
  return issue_page_copies_(c, groups, err);
}

// (v3.27) the body of page_copies, shared with the decode dispatch (c == nullptr: P4's page-copy decode steps). Same device
// ops in the same order for both; only the stats prefix and the ring copy's call path (cabi(c) / cabi_raw_) differ
Status Engine::issue_page_copies_(Ctx* c, const std::vector<PageCopyBatch>& groups, std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string px = c != nullptr ? "mixed" : "decode";
  if (!page_copies_ok(groups, err)) return Status::BAD_ARG;
  if (groups.empty()) return Status::OK;
  auto base = [&](int r, int32_t pool) -> void* {
    const FprEhRankState& rs = h_->ranks[r];
    for (int i = 0; i < rs.n_pools; ++i)
      if (pool_id(rs.pools[i].name) == pool) return rs.pools[i].base;
    return nullptr;
  };
  std::array<bool, NW> dst_touched{};
  int64_t pages = 0;
  {  // begin's drain synchronizes step_stream + xstream per rank, not the device: a source page an
     // earlier op on ANOTHER stream of the source rank (or a reader of a destination page on another stream) still
     // touches would race the copy. Copies are rare (migrations), so every touched device (source and destination) is
     // synchronized device-wide before the first copy: strictly stronger than an event per touched source stream.
    std::array<bool, NW> touched{};
    for (const PageCopyBatch& g : groups)
      if (!g.src_pages.empty()) touched[g.src_rank] = touched[g.dst_rank] = true;
    const auto ts = std::chrono::steady_clock::now();
    for (int d = 0; d < h_->n_ranks; ++d) {
      if (!touched[d]) continue;
      const c10::cuda::CUDAGuard dg((c10::DeviceIndex)h_->ranks[d].device);
      if (!ck(cudaDeviceSynchronize(), "pagecopy pre-sync", err)) return Status::INTERNAL;
      stats_[px + "_page_copy_presyncs"]++;
    }
    stats_[px + "_page_copy_presync_us"] +=
        (int64_t)std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ts).count();
  }
  for (const PageCopyBatch& g : groups) {   // Worker.copies' FPR_PAGECOPY loop, first-seen group order
    const int dr = g.dst_rank;
    const FprEhRankState& rs = h_->ranks[dr];
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    cudaStream_t ss = (cudaStream_t)rs.step_stream;
    const int64_t n = (int64_t)g.src_pages.size();
    if (!n) continue;
    const int i = pc_i_[dr];
    if (!ck(cudaEventSynchronize(pcev_[dr][i]), "pagecopy set", err)) return Status::INTERNAL;
    int64_t* h = pch_[dr][i];
    std::memcpy(h, g.src_pages.data(), (size_t)n * 8);
    std::memcpy(h + n, g.dst_pages.data(), (size_t)n * 8);
    if (!ck(cudaMemcpyAsync(pcd_[dr], h, (size_t)(2 * n * 8), cudaMemcpyHostToDevice, ss), "pagecopy ids H2D", err) ||
        !ck(cudaEventRecord(pcev_[dr][i], ss), "pagecopy set event", err))
      return Status::INTERNAL;
    pc_i_[dr] = (uint8_t)(i ^ 1);
    FprPagecopyArgs a{};
    a.call.stream = ss;
    a.call.device = rs.device;
    a.call.rank = dr;
    a.src = base(g.src_rank, g.pool);
    a.dst = base(dr, g.pool);
    a.sp = pcd_[dr];
    a.dp = pcd_[dr] + n;
    a.n = n;
    a.page_bytes = g.page_bytes;
    const int rc = pc_copy_(&a);
    if (rc != 0) {
      if (err) *err = "nd_mixed: pagecopy.copy returned " + std::to_string(rc);
      return Status::INTERNAL;
    }
    dst_touched[dr] = true;
    pages += n;
  }
#ifdef FPR_ENGRAM_RING_COPY_MAX
  // engine.copy_state_slots after the pool copies: per state group, the destination ring's slots get the source's
  // (engram.ring_copy on the destination's step stream, <= 64 pairs per call, Python's op order); page_copies_ok checked
  // the rings, the slots, ENG_PEER_MAP and P2P. Before the done events below: every rank's step stream sees them too
  for (const PageCopyBatch& g : groups) {
    if (g.pool != POOL_STATE || g.src_pages.empty()) continue;
    const FprEhEngramRank& es = h_->engram[g.src_rank];
    const FprEhEngramRank& ed = h_->engram[g.dst_rank];
    for (size_t i0 = 0; i0 < g.src_pages.size(); i0 += FPR_ENGRAM_RING_COPY_MAX) {
      FprEngramRingCopyArgs a{};
      a.src_ring = es.ring;
      a.dst_ring = ed.ring;
      a.src_nring = es.nring;
      a.dst_nring = ed.nring;
      a.src_ready = nullptr;
      a.n = (int32_t)std::min<size_t>(FPR_ENGRAM_RING_COPY_MAX, g.src_pages.size() - i0);
      for (int32_t j = 0; j < a.n; ++j) {
        a.src_slot[j] = (int32_t)g.src_pages[i0 + (size_t)j];
        a.dst_slot[j] = (int32_t)g.dst_pages[i0 + (size_t)j];
      }
      const Status cs = c != nullptr ? cabi(c, g.dst_rank, eg_ring_copy_, &a, err)   // (FprCall = the destination's step stream)
                                     : cabi_raw_(g.dst_rank, eg_ring_copy_, &a, err);
      if (cs != Status::OK) return Status::INTERNAL;
      stats_[px + "_ring_copies"] += a.n;
    }
    dst_touched[(size_t)g.dst_rank] = true;
  }
#endif
  // Worker.copies then synchronizes every touched device: here every rank's step stream waits for each destination's copies
  for (int d = 0; d < h_->n_ranks; ++d) {
    if (!dst_touched[d]) continue;
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)h_->ranks[d].device);
    if (!ck(cudaEventRecord(pcdone_[d], (cudaStream_t)h_->ranks[d].step_stream), "pagecopy done", err)) return Status::INTERNAL;
    for (int r = 0; r < h_->n_ranks; ++r) {
      if (r == d) continue;
      const c10::cuda::CUDAGuard dr_((c10::DeviceIndex)h_->ranks[r].device);
      if (!ck(cudaStreamWaitEvent((cudaStream_t)h_->ranks[r].step_stream, pcdone_[d], 0), "pagecopy cross-rank wait", err))
        return Status::INTERNAL;
    }
  }
  stats_[px + "_page_copy_groups"] += (int64_t)groups.size();
  stats_[px + "_page_copy_pages"] += pages;
  if (c == nullptr) {   // stderr on the 1st and every 64th
    const int64_t k = ++stats_["decode_pc_native_steps"];   // native DECODE copy step (P4)
    if (k == 1 || k % 64 == 0)
      std::fprintf(stderr, "nd: native decode page copies: step %lld, groups %lld, pages %lld (cumulative)\n", (long long)k,
                   (long long)stats_["decode_page_copy_groups"], (long long)stats_["decode_page_copy_pages"]);
  } else {   // (v3.28f) native MIXED copy step, original v3.10b copy code (device pre-sync)
    const int64_t k = ++stats_["mixed_pc_native_steps"];
    if (k == 1 || k % 64 == 0)
      std::fprintf(stderr, "nd: native mixed page copies: step %lld, groups %lld, pages %lld (cumulative, v3.10b)\n", (long long)k,
                   (long long)stats_["mixed_page_copy_groups"], (long long)stats_["mixed_page_copy_pages"]);
  }
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  if (c != nullptr) mixed_tick("page_copies", us);
  else stats_["decode_page_copy_us"] += (int64_t)us;
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ (v3.23, L3a) lazy push
Status Engine::push_now(Ctx* c, std::string* err) {
  if (c == nullptr || c != mx_ctx_.get()) {
    if (err) *err = "nd_mixed: push_now outside an open step";
    return Status::INTERNAL;
  }
  if (pushed_step_) return Status::OK;
  const Status s = push_state_(err, &push_sc_);
  if (s != Status::OK) return Status::INTERNAL;
  pushed_step_ = true;
  stats_["mixed_push_now"]++;
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ (v3.16) Engram lookback
bool Engine::lookback(const Ctx* c, const Rid& rid, int64_t start, int64_t out[3]) const {
  if (c == nullptr || c != mx_ctx_.get() || out == nullptr) return false;
  return lookback_of(reqs_, c->b, rid, start, out);   // (begin observed the step's StepHost: fresh prompts are in reqs_)
}

// ------------------------------------------------------------------------------------------ (v3.11) generic device ops
Status Engine::cabi_raw_(int r, int (*fn)(const void*), void* args, std::string* err) {   // (v3.27) cabi's body, no step
  if (h_ == nullptr || r < 0 || r >= h_->n_ranks || fn == nullptr || args == nullptr) {
    if (err) *err = "nd_engine: cabi_raw: bad rank / fn / args";
    return Status::BAD_ARG;
  }
  const FprEhRankState& rs = h_->ranks[r];
  const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);   // C ABI rule 3: the caller sets the device
  FprCall* call = static_cast<FprCall*>(args);                  // rule 1: every Args struct starts with FprCall
  call->stream = (cudaStream_t)rs.step_stream;
  call->device = rs.device;
  call->rank = r;
  const int rc = fn(args);
  stats_["decode_cabi_calls"]++;
  if (rc != 0) {
    if (err) *err = "nd_engine: cabi_raw rc " + std::to_string(rc);
    return Status::INTERNAL;
  }
  if (!ck(cudaGetLastError(), "cabi launch", err)) return Status::INTERNAL;
  return Status::OK;
}

Status Engine::cabi(Ctx* c, int r, int (*fn)(const void*), void* args, std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  if (c == nullptr || c != mx_ctx_.get()) {
    if (err) *err = "nd_mixed: cabi outside an open step";
    return Status::INTERNAL;
  }
  if (h_ == nullptr || r < 0 || r >= h_->n_ranks || fn == nullptr || args == nullptr) {
    if (err) *err = "nd_mixed: cabi: bad rank / fn / args";
    return Status::BAD_ARG;
  }
  const FprEhRankState& rs = h_->ranks[r];
  const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);   // C ABI rule 3: the caller sets the device
  FprCall* call = static_cast<FprCall*>(args);                  // rule 1: every Args struct starts with FprCall
  call->stream = (cudaStream_t)rs.step_stream;
  call->device = rs.device;
  call->rank = r;
  const int rc = fn(args);
  stats_["mixed_cabi_calls"]++;
  if (rc != 0) {
    if (err) *err = "nd_mixed: cabi rc " + std::to_string(rc);
    return Status::INTERNAL;
  }
  if (!ck(cudaGetLastError(), "cabi launch", err)) return Status::INTERNAL;
  mixed_tick("cabi", std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  return Status::OK;
}

Status Engine::h2d(Ctx* c, int r, const FprEhSets* sets, void* dst, const void* src, int64_t bytes, std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  if (c == nullptr || c != mx_ctx_.get()) {
    if (err) *err = "nd_mixed: h2d outside an open step";
    return Status::INTERNAL;
  }
  auto bad = [&](const std::string& m) {
    if (err) *err = "nd_mixed: h2d: " + m;
    return Status::BAD_ARG;
  };
  if (h_ == nullptr || r < 0 || r >= h_->n_ranks || sets == nullptr || bytes < 0) return bad("bad rank / sets / bytes");
  if (bytes == 0) return Status::OK;
  if (dst == nullptr || src == nullptr) return bad("NULL dst / src");
  const FprEhRankState& rs = h_->ranks[r];
  uint8_t& i = h2d_i_[sets];   // (nd's own alternation per set pair; persistent events: either index is safe)
  if (sets->ev[i] == nullptr || sets->h[i].ptr == nullptr) return bad("the set has no host buffer / event");
  if (bytes > sets->h[i].bytes) {
    if (err) *err = "nd_mixed: h2d: " + std::to_string(bytes) + " B > the set's " + std::to_string(sets->h[i].bytes);
    return Status::INTERNAL;
  }
  cudaPointerAttributes pa{};   // (host / managed / another device's memory refused; "unregistered" passes: some VMM
                                // mappings report it, a host pointer then fails in the copy -> INTERNAL)
  if (cudaPointerGetAttributes(&pa, dst) != cudaSuccess) {
    cudaGetLastError();
    return bad("dst: cudaPointerGetAttributes failed");
  }
  if (pa.type == cudaMemoryTypeHost || pa.type == cudaMemoryTypeManaged ||
      (pa.type == cudaMemoryTypeDevice && pa.device != rs.device))
    return bad("dst is not device memory of rank " + std::to_string(r) + "'s device");
  if (pa.type == cudaMemoryTypeUnregistered) stats_["mixed_h2d_dst_unregistered"]++;
  const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
  cudaStream_t ss = (cudaStream_t)rs.step_stream;
  cudaEvent_t ev = (cudaEvent_t)sets->ev[i];
  if (!ck(cudaEventSynchronize(ev), "h2d set", err)) return Status::INTERNAL;   // the set's copy from two uses ago
  std::memcpy(sets->h[i].ptr, src, (size_t)bytes);
  if (!ck(cudaMemcpyAsync(dst, sets->h[i].ptr, (size_t)bytes, cudaMemcpyHostToDevice, ss), "h2d copy", err) ||
      !ck(cudaEventRecord(ev, ss), "h2d set event", err))
    return Status::INTERNAL;
  i ^= 1;
  stats_["mixed_h2d_calls"]++;
  stats_["mixed_h2d_bytes"] += bytes;
  mixed_tick("h2d", std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  return Status::OK;
}

Status Engine::upload_inputs(int r, p0::InputsSnap snap, void* in_d, int64_t in_d_bytes, void* host, int64_t host_bytes,
                             const RankTables& tab, int64_t* n, int64_t* n_sample, int64_t* n_pairs, std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  if (h_ == nullptr || r < 0 || r >= h_->n_ranks || in_d == nullptr || host == nullptr) {
    if (err) *err = "nd_t: upload_inputs before init / a bad rank / no buffers";
    return Status::BAD_ARG;
  }
  const FprEhRankState& rs = h_->ranks[r];
  snap.rank = r;
  snap.k1 = snap.hsd ? (int32_t)rs.next_ids.shape[1] : 0;
  p0::InputsOut o;
  if (p0::build_inputs(snap, tab, &o, err) != Status::OK) return Status::BAD_ARG;
  const int64_t nb = (int64_t)o.blob.size();
  if (nb != in_d_bytes || nb > host_bytes) {
    if (err) *err = "nd_t: blob " + std::to_string(nb) + " B vs in_d " + std::to_string(in_d_bytes) + " / set " +
                    std::to_string(host_bytes) + " B";
    return Status::BAD_ARG;
  }
  std::vector<DevtokMig> mg;
  for (const p0::MigCopy& g : o.mig) mg.push_back(DevtokMig{g.row0, g.n_rows, g.r0, g.si0});
  if (devtok_check(r, snap.R, o.rows, o.src, mg, err) != Status::OK) return Status::BAD_ARG;   // (before any device op)
  std::memcpy(host, o.blob.data(), (size_t)nb);   // the pinned set (the caller waited for its previous H2D)
  {
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    if (!ck(cudaMemcpyAsync(in_d, host, (size_t)nb, cudaMemcpyHostToDevice, (cudaStream_t)rs.step_stream), "nd_t blob H2D",
            err))
      return Status::INTERNAL;
  }
  int64_t* tok = reinterpret_cast<int64_t*>(static_cast<char*>(in_d) + p0::in_layout(snap.R).off[p0::IN_TOK]);
  if (devtok_issue(r, tok, snap.R, o.rows, o.src, mg, err) != Status::OK) return Status::INTERNAL;
  *n = o.n;
  *n_sample = o.n_sample;
  *n_pairs = (int64_t)o.rows.size();
  stats_["nd_t_uploads"] += 1;
  stats_["nd_t_devtok_rows"] += (int64_t)o.rows.size();
  stats_["nd_t_devtok_mig"] += (int64_t)o.mig.size();
  mixed_tick("nd_t_upload", std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  return Status::OK;
}

// ------------------------------------------------------------------------------------------ (5) after the step's sync
Status Engine::mixed_finish(const MixedHead& mh, const std::vector<StepBatch>& b, const MixedBegin& mb,
                            std::vector<StepResult>* out, std::string* err) {
  const auto t0 = std::chrono::steady_clock::now();
  const int W = h_->n_ranks;
  // head_result per rank: HSD.result's copies (synchronous: the step is synced) + the formula + J history
  Handle tmp;
  tmp.cur = mh.cur;
  ResSet& set = res_[NRING];
  const int64_t n = mh.cur.n_req, K1 = cfg_.k + 1;
  std::array<const int32_t*, NW> ns{};
  std::array<const int64_t*, NW> smp{};
  std::vector<int> act;
  for (int r = 0; r < W; ++r)
    if (!cfg_.res_owner || (r < (int)mh.cur.dseqs.size() && !mh.cur.dseqs[r].empty())) act.push_back(r);
  for (int r : act) {
    const FprEhRankState& rs = h_->ranks[r];
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    cudaStream_t ss = (cudaStream_t)rs.step_stream;
    if (!ck(cudaMemcpyAsync(set.ns[r], rs.num_sampled.data, n * 4, cudaMemcpyDeviceToHost, ss), "result ns", err) ||
        !ck(cudaMemcpy2DAsync(set.smp[r], K1 * 8, rs.sampled.data, rs.sampled.stride[0] * 8, K1 * 8, n,
                              cudaMemcpyDeviceToHost, ss), "result sampled", err) ||
        !ck(cudaStreamSynchronize(ss), "result copies", err))
      return Status::INTERNAL;
    ns[r] = set.ns[r];
    smp[r] = set.smp[r];
  }
  results_from(&tmp, ns.data(), smp.data(), &b);   // (jY) ACCSTAT's context bin reads b[r].seqs[si].start
  *out = tmp.res;
  // head_result's sync-loop record: prev_seq[rid] = (rank, seq index, serial) for the step's batches, CP helpers excluded
  for (int r = 0; r < W && r < (int)b.size(); ++r)
    for (size_t si = 0; si < b[r].seqs.size(); ++si) {
      const SeqEntry& se = b[r].seqs[si];
      if (se.has_cp && se.cp.owner != r) continue;
      prev_[se.rid] = PrevOf{(int64_t)r, (int64_t)si, step_serial_};
    }
  // D13._fallback's tail: accepted per sequence (decode = HSD's count, prefill = n_rows) -> rx.accepted -> rx.end()
  for (int r = 0; r < W && r < (int)mb.rb.size(); ++r) {
    const StepBatch& rb = mb.rb[r];
    const int64_t S = (int64_t)rb.seqs.size();
    if (!S) continue;
    const FprEhRankState& rs = h_->ranks[r];
    const std::vector<std::pair<Rid, int32_t>>& acc = r < (int)out->size() ? (*out)[r].accepted
                                                                          : std::vector<std::pair<Rid, int32_t>>{};
    const std::vector<int64_t> a = host_->fallback_accepted(rb, acc);
    for (int64_t k = 0; k < S; ++k) mx_acc_h_[r][k] = (int32_t)a[k];
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)rs.device);
    cudaStream_t ss = (cudaStream_t)rs.step_stream;
    if (!ck(cudaMemcpyAsync(mx_acc_d_[r], mx_acc_h_[r], S * 4, cudaMemcpyHostToDevice, ss), "accepted H2D", err))
      return Status::INTERNAL;
    const int32_t* b32 = reinterpret_cast<const int32_t*>(rs.rx_b32_d.ptr);
    FprRowexpAdvanceArgs ad{};
    ad.c = FprCall{ss, rs.device, 0};
    ad.slot = b32 + rs.rx_o32[0];
    ad.kind = b32 + rs.rx_o32[3];
    ad.nrows = b32 + rs.rx_o32[2];
    ad.start_res = tp<const int64_t>(rs.rx_start);
    ad.accepted = mx_acc_d_[r];
    ad.counts = tp<const int32_t>(rs.rx_counts);
    ad.nc = tp<int32_t>(rs.rx_nc);
    ad.s_max = rs.rx_S_max;
    if (rx_advance_(&ad) != FPR_OK) {
      if (err) *err = std::string("nd_mixed: rowexp.advance: ") + cudaGetErrorString(cudaPeekAtLastError());
      return Status::INTERNAL;
    }
    if (!ck(cudaDeviceSynchronize(), "fallback device sync", err)) return Status::INTERNAL;   // torch.cuda.synchronize(dev)
  }
  mixed_tick("finish", us_since(t0));
  return Status::OK;
}

Status Engine::finish(Ctx* c, std::vector<StepResult>* out, std::string* err) {
  if (c == nullptr || c != mx_ctx_.get() || !c->headed || c->launched != h_->n_ranks) {
    if (err) *err = "nd_mixed: finish without prepare_head + every rank's h1_launch";
    return Status::INTERNAL;
  }
  std::unique_ptr<Ctx> cx = std::move(mx_ctx_);   // freed on return (whatever the status: a failure is fatal anyway)
  // prev-of on the step's batches as the worker gave them (head_result(r, stT[r], batches[r]): the same seq order as rb)
  Status s = mixed_finish(cx->mh, cx->b, cx->mb, out, err);
  if (s != Status::OK) return Status::INTERNAL;
  // head_result's D7 / D11 drafts of structured / grammar requests (next_ids of this synced step; the
  // worker's batches give head_result's seq order, = rb's)
  s = drafts_into_(cx->b, cx->grammar, out, err);
  if (s != Status::OK) return Status::INTERNAL;
  have_cur_ = false;   // (embedded_end's rule) hsd.cur is the mixed head's: nd's next decode step prepares on the host
  return Status::OK;
}

}  // namespace nd
}  // namespace fpr
