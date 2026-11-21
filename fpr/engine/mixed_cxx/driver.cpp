// driver.cpp -- the C++ mixed-step driver, stage 1b-i: pieces called per (piece, rank, layer)
// from Engine._issue (ENG_MIXED_CXX_PIECES), each issuing exactly the Python piece's launches through the extensions'
// C ABI tables (fpr_cabi.h). Stage 1d moves the layer loop onto 4 threads (fpr_driver_run, mixed_plan.h).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include "fpr_lt.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "pieces.h"
#include "driver_api.h"

namespace {
constexpr int W = 4;
struct Rank {
  cudaEvent_t ev = nullptr;      // main -> side handoff inside a boundary
  cudaEvent_t mix = nullptr;     // the boundary's side-stream mixes done (= Python's s["mix_ev"])
  bool pending = false;          // a mix event the next _join_mix must wait
  int dev = -1;
  cudaEvent_t ev_f = nullptr;    // AGCE: "f ready" (router) / "route rows ready" (ag), main -> the 4 copy streams
  cudaEvent_t ev_r = nullptr;
  cudaEvent_t done[W][2] = {};   // AGCE: this rank's copies to receiver q done, a 2-slot ring by CE count (recorded on
                                 // cs[q], device of this rank)
  int64_t ce_n = 0;              // AGCE: CE all-gathers this rank issued (every rank issues the same ones)
  int64_t seen[W] = {};          // AGCE, receiver side: per sender, the CE count whose done event this rank waited last
  cudaStream_t main = nullptr;   // stage 1d: the rank stream (FPR_K_JOIN / FPR_K_AGCE_JOIN)
  std::string err;               // stage 1d: the last error on this rank's thread
};
Rank g_rank[W];
// AGCE handshake between host threads (stage 1d; never blocks when one thread issues every rank in order, 1b-i):
// g_rec[s][q] = the CE count n whose done event sender s recorded for receiver q (release after the record);
// g_cons[s][q] = the CE count whose done event receiver q has issued its wait on (release after the wait).
// Sender at CE n reuses slot n & 1 only once g_cons >= n - 2; receiver at its own CE count n waits g_rec >= n.
std::atomic<int64_t> g_rec[W][W], g_cons[W][W];
constexpr int FPR_E_HANDSHAKE = -100, FPR_E_EXCEPTION = -101, FPR_E_KIND = -102;
constexpr int FPR_E_DEAD = -103;   // an earlier span failed: every later span is refused (driver_api.h)
// ENG_MIXED_CXX_STAMP=1: per-rank host timing of the spans. issue = the rank's
// entries minus its host waits, wait = spin_until (AGCE ring / receiver barriers), lag = run_span's notify -> the rank thread
// running (the cv wake). One cache line per rank, written only by the thread issuing that rank (no atomics); read at exit.
const bool g_stamp = [] {
  const char* e = std::getenv("ENG_MIXED_CXX_STAMP");
  return e != nullptr && e[0] == '1';
}();
struct alignas(64) Stamp {
  int64_t issue_ns = 0, wait_ns = 0, lag_ns = 0, max_issue_ns = 0, max_wait_ns = 0, max_lag_ns = 0, spans = 0;
};
Stamp g_st[W];
thread_local int64_t t_wait_ns = 0;   // spin_until time of the calling thread since its last reset
inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void stamp_add(int r, int64_t issue, int64_t wait, int64_t lag) {
  Stamp& k = g_st[r];
  k.issue_ns += issue, k.wait_ns += wait, k.lag_ns += lag, ++k.spans;
  k.max_issue_ns = std::max(k.max_issue_ns, issue), k.max_wait_ns = std::max(k.max_wait_ns, wait);
  k.max_lag_ns = std::max(k.max_lag_ns, lag);
}
template <class F>
int spin_until(F ready) {
  auto t0 = std::chrono::steady_clock::now();
  int rc = FPR_OK;
  for (int i = 0; !ready(); ++i) {
    if (i > 64) std::this_thread::yield();
    if ((i & 1023) == 1023 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
      rc = FPR_E_HANDSHAKE;
      break;
    }
  }
  if (g_stamp) t_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
  return rc;
}
std::unordered_map<std::string, const FprExtTable*> g_tab;
std::vector<std::unique_ptr<FprPieceBd>> g_bd;
std::vector<std::unique_ptr<FprPieceRouter>> g_router;
std::vector<std::unique_ptr<FprPieceAg>> g_ag;
std::vector<std::unique_ptr<FprPieceRs>> g_rs;
std::vector<std::unique_ptr<FprPieceFinal>> g_final;
FprFn F_frow = nullptr, F_fmix = nullptr, F_fbmix = nullptr, F_fa = nullptr, F_fbout = nullptr;
FprFn F_top3 = nullptr;                                              // moe_raw
FprFn F_agp2 = nullptr, F_wait = nullptr, F_rsp = nullptr, F_rss = nullptr;   // comm
FprFn F_k1a = nullptr, F_k1b = nullptr;                              // mhc

FprFn find_fn(const FprExtTable* t, const char* name, int64_t args_size) {
  for (int i = 0; i < t->n_fns; ++i)
    if (std::strcmp(t->fns[i].name, name) == 0) {
      TORCH_CHECK(t->fns[i].args_size == args_size, "mixed_cxx: ", name, " args ", t->fns[i].args_size, " != driver's ",
                  args_size, " (header drift)");
      return t->fns[i].fn;
    }
  TORCH_CHECK(false, "mixed_cxx: ", t->ext, " has no ", name);
  return nullptr;
}
inline int ck(cudaError_t e) { return e == cudaSuccess ? FPR_OK : FPR_E_CUDA; }

// Piece structs: slots in a vector, freed slots reused (free list). Adds and frees happen only from the Python thread
// between spans (never while the driver threads run), so a slot never moves or dies under a running thread.
std::unordered_map<const void*, std::vector<int64_t>> g_free;      // per piece vector: its free slots
template <class T>
int64_t add_piece(std::vector<std::unique_ptr<T>>& v, const py::bytes& b, const char* what) {
  std::string s = b;
  TORCH_CHECK(s.size() == sizeof(T), "mixed_cxx: ", what, " is ", s.size(), " bytes, the driver's ", sizeof(T),
              " (ctypes / header drift)");
  auto p = std::make_unique<T>();
  std::memcpy(p.get(), s.data(), s.size());
  TORCH_CHECK(p->version == FPR_PIECES_VERSION, "mixed_cxx: ", what, " pieces version");
  auto& fl = g_free[&v];
  if (!fl.empty()) {
    const int64_t h = fl.back();
    fl.pop_back();
    v[h] = std::move(p);
    return h;
  }
  v.push_back(std::move(p));
  return (int64_t)v.size() - 1;
}
template <class T>
void free_piece(std::vector<std::unique_ptr<T>>& v, int64_t h) {
  if (h < 0 || h >= (int64_t)v.size() || !v[h]) return;
  v[h].reset();
  g_free[&v].push_back(h);
}
template <class T>
int64_t live_pieces(const std::vector<std::unique_ptr<T>>& v) {
  int64_t n = 0;
  for (auto& p : v) n += p != nullptr;
  return n;
}
template <class T>
const T* get_piece(const std::vector<std::unique_ptr<T>>& v, int64_t h, const char* what) {
  TORCH_CHECK(h >= 0 && h < (int64_t)v.size() && v[h], "mixed_cxx: ", what, " handle ", h, " (freed or out of range)");
  return v[h].get();
}
// AGCE receiver side: wait every sender's done event of this rank's current CE all-gather, if not waited yet (sender order;
// idempotent within a layer: the p_moe wrapper and the moe piece may both call it)
int agce_join_c(int r, cudaStream_t stream) {
  Rank& k = g_rank[r];
  const int64_t n = k.ce_n;
  for (int q = 0; q < W; ++q) {
    if (k.seen[q] >= n) continue;
    if (int e = spin_until([&] { return g_rec[q][r].load(std::memory_order_acquire) >= n; })) return e;
    if (int e = ck(cudaStreamWaitEvent(stream, g_rank[q].done[r][n & 1], 0))) return e;
    k.seen[q] = n;
    g_cons[q][r].store(n, std::memory_order_release);
  }
  return FPR_OK;
}
}  // namespace

// ---- driver_api.h (for other owners' piece files)
FprFn fpr_drv_fn(const char* ext, const char* name, int64_t args_size) {
  auto it = g_tab.find(ext);
  return it == g_tab.end() ? nullptr : find_fn(it->second, name, args_size);
}
cudaEvent_t fpr_drv_event(int rank, int which) { return which == 0 ? g_rank[rank].ev : g_rank[rank].mix; }
int fpr_drv_join(int rank, cudaStream_t stream) {
  Rank& k = g_rank[rank];
  if (!k.pending) return FPR_OK;
  k.pending = false;
  return ck(cudaStreamWaitEvent(stream, k.mix, 0));
}
int fpr_drv_device(int rank) { return g_rank[rank].dev; }
int fpr_drv_agce_join(int rank, cudaStream_t stream) { return agce_join_c(rank, stream); }

static FprStepParams g_sp[W];
const FprStepParams* fpr_drv_step_params(int rank) { return &g_sp[rank]; }
extern "C" __attribute__((visibility("default"))) int fpr_drv_set_step_params(int rank, const FprStepParams* sp) {
  if (rank < 0 || rank >= W || sp == nullptr) return FPR_E_ARG;
  std::memcpy(&g_sp[rank], sp, sizeof(FprStepParams));
  return FPR_OK;
}
void set_step_params(int r, py::bytes b) {
  std::string s = b;
  TORCH_CHECK(r >= 0 && r < W && s.size() == sizeof(FprStepParams), "mixed_cxx: FprStepParams is ", s.size(), " bytes, the driver's ",
              sizeof(FprStepParams));
  fpr_drv_set_step_params(r, reinterpret_cast<const FprStepParams*>(s.data()));
}
int64_t sizeof_step_params() { return (int64_t)sizeof(FprStepParams); }
// owners' piece files (weak: absent files simply register nothing)
void fpr_pieces_moe_register(pybind11::module_& m) __attribute__((weak));
void fpr_pieces_attn_register(pybind11::module_& m) __attribute__((weak));
void fpr_pieces_head_register(pybind11::module_& m) __attribute__((weak));
void fpr_pieces_engram_register(pybind11::module_& m) __attribute__((weak));  // FPR (Engram prepare / inject)
void fpr_pieces_c1mega_register(pybind11::module_& m) __attribute__((weak));

// register an extension's table (its Python module must be imported: the .so is looked up with RTLD_NOLOAD)
void register_table(const std::string& lib, const std::string& ext) {
  void* h = dlopen(lib.c_str(), RTLD_NOW | RTLD_NOLOAD);
  TORCH_CHECK(h != nullptr, "mixed_cxx: ", lib, " not loaded: ", dlerror());
  auto f = reinterpret_cast<FprTableFn>(dlsym(h, ("fpr_" + ext + "_table").c_str()));
  TORCH_CHECK(f != nullptr, "mixed_cxx: no fpr_", ext, "_table in ", lib);
  const FprExtTable* t = f();
  TORCH_CHECK(t && t->abi_version == FPR_CABI_VERSION, "mixed_cxx: ", ext, " ABI version mismatch");
  g_tab[ext] = t;
  if (ext == "mhcf") {
    F_frow = find_fn(t, "mhcf.frow", sizeof(FprMhcfFrowArgs));
    F_fmix = find_fn(t, "mhcf.fmix", sizeof(FprMhcfFmixArgs));
    F_fbmix = find_fn(t, "mhcf.fb_mix", sizeof(FprMhcfFbMixArgs));
    F_fa = find_fn(t, "mhcf.fa", sizeof(FprMhcfFaArgs));
    F_fbout = find_fn(t, "mhcf.fb_out", sizeof(FprMhcfFbOutArgs));
  } else if (ext == "moe_raw") {
    F_top3 = find_fn(t, "moe_raw.router_top_pack3", sizeof(FprMoeRawRouterTopPack3Args));
  } else if (ext == "comm") {
    F_agp2 = find_fn(t, "comm.ag_push2", sizeof(FprCommAgPush2Args));
    F_wait = find_fn(t, "comm.wait", sizeof(FprCommWaitArgs));
    F_rsp = find_fn(t, "comm.rs_push", sizeof(FprCommRsPushArgs));
    F_rss = find_fn(t, "comm.rs_sum", sizeof(FprCommRsSumArgs));
  } else if (ext == "mhc") {
    F_k1a = find_fn(t, "mhc.k1a_x", sizeof(FprMhcK1aXArgs));
    F_k1b = find_fn(t, "mhc.k1b_out", sizeof(FprMhcK1bOutArgs));
  }
}

// per rank: the reusable events + every registered extension's init on that device (call with the device current)
void init_rank(int r, int device) {
  TORCH_CHECK(r >= 0 && r < W, "mixed_cxx: rank");
  const c10::cuda::CUDAGuard g((c10::DeviceIndex)device);
  Rank& k = g_rank[r];
  k.dev = device;
  if (!k.ev) C10_CUDA_CHECK(cudaEventCreateWithFlags(&k.ev, cudaEventDisableTiming));
  if (!k.mix) C10_CUDA_CHECK(cudaEventCreateWithFlags(&k.mix, cudaEventDisableTiming));
  if (!k.ev_f) C10_CUDA_CHECK(cudaEventCreateWithFlags(&k.ev_f, cudaEventDisableTiming));
  if (!k.ev_r) C10_CUDA_CHECK(cudaEventCreateWithFlags(&k.ev_r, cudaEventDisableTiming));
  for (int q = 0; q < W; ++q)
    for (int j = 0; j < 2; ++j)
      if (!k.done[q][j]) C10_CUDA_CHECK(cudaEventCreateWithFlags(&k.done[q][j], cudaEventDisableTiming));
  auto it = g_tab.find("mhcf");
  if (it != g_tab.end()) {
    FprMhcfInitArgs a{{nullptr, device, r}};
    TORCH_CHECK(find_fn(it->second, "mhcf.init", sizeof(FprMhcfInitArgs))(&a) == FPR_OK, "mixed_cxx: mhcf.init");
  }
  it = g_tab.find("moe_raw");
  if (it != g_tab.end()) {
    FprMoeRawInitArgs a{{nullptr, device, r}};
    TORCH_CHECK(find_fn(it->second, "moe_raw.init", sizeof(FprMoeRawInitArgs))(&a) == FPR_OK, "mixed_cxx: moe_raw.init");
  }
}

int64_t sizeof_bd() { return (int64_t)sizeof(FprPieceBd); }

int64_t add_bd(py::bytes b) { return add_piece(g_bd, b, "FprPieceBd"); }

// Engine._join_mix for a rank: the current stream waits the pending mix event (the boundary's side-stream work)
void join(int r, int64_t stream) {
  Rank& k = g_rank[r];
  if (k.pending) {
    C10_CUDA_CHECK(cudaStreamWaitEvent((cudaStream_t)stream, k.mix, 0));
    k.pending = false;
  }
}
bool pending(int r) { return g_rank[r].pending; }

// the boundary piece (Engine._split_bd, ENG_MHCF fused path; see pieces.h)
int bd(int r, int64_t h, int chain_input) {
  TORCH_CHECK(F_frow != nullptr, "mixed_cxx: bd needs the mhcf table");
  const FprPieceBd* p = get_piece(g_bd, h, "bd");
  Rank& k = g_rank[r];
  if (k.pending) {                                     // _join_mix(s, pf=False) at the boundary's start
    if (int e = ck(cudaStreamWaitEvent(p->main, k.mix, 0))) return e;
    k.pending = false;
  }
  const int set = (p->which == 0 && p->first_layer && !chain_input) ? 0 : 1;
  int rc;
  if (p->path == 1) {
    if ((rc = F_frow(&p->frow[set]))) return rc;
    if ((rc = ck(cudaEventRecord(k.ev, p->main)))) return rc;
    if ((rc = ck(cudaStreamWaitEvent(p->side, k.ev, 0)))) return rc;
    if ((rc = F_fmix(&p->fmix))) return rc;
    if ((rc = F_fbmix(&p->fb_mix))) return rc;
  } else if (p->path == 2) {
    for (int c = 0; c < p->n_chunks; ++c)
      if ((rc = F_fa(&p->fa[set][c]))) return rc;
    if ((rc = ck(cudaEventRecord(k.ev, p->main)))) return rc;
    for (int c = 0; c < p->n_chunks; ++c)
      if ((rc = F_fbout(&p->fb_out[c]))) return rc;
    if ((rc = ck(cudaStreamWaitEvent(p->side, k.ev, 0)))) return rc;
    for (int c = 0; c < p->n_chunks; ++c)
      if ((rc = F_fbmix(&p->fb_mix_c[c]))) return rc;
  } else {
    return FPR_E_SHAPE;
  }
  if ((rc = ck(cudaEventRecord(k.mix, p->side)))) return rc;
  k.pending = true;
  if (p->which == 0 && p->aux_index >= 0) {            // DSpark aux: s["aux"][:, i].copy_(s["X"].mean(1)) on the main stream
    const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)k.dev));
    auto o = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, k.dev);
    auto X = at::from_blob(p->aux_X, {p->R, p->aux_HC, p->aux_D}, o);
    auto tmp = at::from_blob(p->aux_tmp, {p->R, p->aux_D}, o);
    at::mean_out(tmp, X, at::IntArrayRef{1}, false, c10::nullopt);
    auto dst = at::from_blob(p->aux_dst, {p->R, p->aux_D}, {p->aux_dst_stride0, 1}, o);
    dst.copy_(tmp);
  }
  return FPR_OK;
}

// ---- router / ag / rs / final of prod buckets (pieces.h)
int64_t sizeof_router() { return (int64_t)sizeof(FprPieceRouter); }
int64_t sizeof_ag() { return (int64_t)sizeof(FprPieceAg); }
int64_t sizeof_rs() { return (int64_t)sizeof(FprPieceRs); }
int64_t sizeof_final() { return (int64_t)sizeof(FprPieceFinal); }
int64_t add_router(py::bytes b) { return add_piece(g_router, b, "FprPieceRouter"); }
int64_t add_ag(py::bytes b) { return add_piece(g_ag, b, "FprPieceAg"); }
int64_t add_rs(py::bytes b) { return add_piece(g_rs, b, "FprPieceRs"); }
int64_t add_final(py::bytes b) { return add_piece(g_final, b, "FprPieceFinal"); }
// free a bucket state's handle (mixed_cxx._Cache.__del__, when the state is dropped); unknown / freed handles are ignored
void free_handle(const std::string& kind, int64_t h) {
  if (kind == "bd") free_piece(g_bd, h);
  else if (kind == "router") free_piece(g_router, h);
  else if (kind == "ag") free_piece(g_ag, h);
  else if (kind == "rs") free_piece(g_rs, h);
  else if (kind == "final") free_piece(g_final, h);
}
py::dict live_handles() {
  py::dict d;
  d["bd"] = live_pieces(g_bd), d["router"] = live_pieces(g_router), d["ag"] = live_pieces(g_ag);
  d["rs"] = live_pieces(g_rs), d["final"] = live_pieces(g_final);
  return d;
}

// agrs_engine.p_router (AGCE f copies) + Engine.p_router (prod, router_pack3): see pieces.h
int router(int r, int64_t h, int ce) {
  const FprPieceRouter* p = get_piece(g_router, h, "router");
  TORCH_CHECK(F_top3 != nullptr, "mixed_cxx: router needs the moe_raw table");
  Rank& k = g_rank[r];
  int rc;
  if (ce) {
    if (!p->ce_ok) return FPR_E_ARG;
    if ((rc = ck(cudaEventRecord(k.ev_f, p->main)))) return rc;
    for (int q = 0; q < W; ++q) {
      if ((rc = ck(cudaStreamWaitEvent(p->cs[q], k.ev_f, 0)))) return rc;
      if ((rc = ck(cudaMemcpyAsync(p->gf_dst[q], p->f_src, (size_t)p->nb_f, cudaMemcpyDeviceToDevice, p->cs[q])))) return rc;
    }
  }
  {
    const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)k.dev));
    auto o = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, k.dev);
    auto f = at::from_blob(p->f, {p->R, p->D}, {p->f_st0, p->f_st1}, o);
    auto gw = at::from_blob(p->gw, {p->E, p->D}, {p->gw_st0, p->gw_st1}, o);
    auto lg = at::from_blob(p->lg, {p->R, p->E}, {p->lg_st0, p->lg_st1}, o.dtype(at::kFloat));
    fpr_lt::mm(lg, f, gw);                     // = torch.mm(f, gate_w.t(), out_dtype=torch.float32, out=logits) (fpr_lt; ATen fallback = that)
  }
  return F_top3(&p->top);
}

// agrs_engine.p_ag (AGCE route-row copies + done events) / Engine.p_ag (prod: ag_push2 + wait)
int ag(int r, int64_t h, int ce) {
  const FprPieceAg* p = get_piece(g_ag, h, "ag");
  Rank& k = g_rank[r];
  int rc;
  if (ce) {
    if (!p->ce_ok) return FPR_E_ARG;
    const int64_t n = ++k.ce_n;
    if ((rc = ck(cudaEventRecord(k.ev_r, p->main)))) return rc;
    for (int q = 0; q < W; ++q) {
      if ((rc = ck(cudaStreamWaitEvent(p->cs[q], k.ev_r, 0)))) return rc;
      if ((rc = ck(cudaMemcpyAsync(p->gr_dst[q], p->rr_src, (size_t)p->nb_r, cudaMemcpyDeviceToDevice, p->cs[q])))) return rc;
      if ((rc = spin_until([&] { return g_cons[r][q].load(std::memory_order_acquire) >= n - 2; }))) return rc;
      if ((rc = ck(cudaEventRecord(k.done[q][n & 1], p->cs[q])))) return rc;
      g_rec[r][q].store(n, std::memory_order_release);
    }
    return FPR_OK;
  }
  TORCH_CHECK(F_agp2 != nullptr, "mixed_cxx: ag needs the comm table");
  if ((rc = F_agp2(&p->push))) return rc;
  return F_wait(&p->wait);
}

void agce_join(int r, int64_t stream) {
  C10_CUDA_CHECK(agce_join_c(r, (cudaStream_t)stream) == FPR_OK ? cudaSuccess : cudaErrorUnknown);
}
int64_t agce_pending(int r) {      // senders whose current CE done event this rank has not waited yet (bit mask)
  int64_t m = 0;
  for (int q = 0; q < W; ++q) m |= (g_rank[r].seen[q] < g_rank[r].ce_n) ? (1 << q) : 0;
  return m;
}

// agrs_engine.p_rs / Engine.p_rs (prod)
int rs(int r, int64_t h, int rsf) {
  const FprPieceRs* p = get_piece(g_rs, h, "rs");
  TORCH_CHECK(F_rss != nullptr, "mixed_cxx: rs needs the comm table");
  (void)r;
  if (rsf) return p->rsf_ok ? F_rss(&p->sum_rsf) : FPR_E_ARG;
  if (int rc = F_rsp(&p->push)) return rc;
  return F_rss(&p->sum);
}

// Engine.p_final up to hooks.head_run (prod)
int final_(int r, int64_t h) {
  const FprPieceFinal* p = get_piece(g_final, h, "final");
  Rank& k = g_rank[r];
  int rc;
  if (k.pending) {                                     // _join_mix(s)
    if ((rc = ck(cudaStreamWaitEvent(p->main, k.mix, 0)))) return rc;
    k.pending = false;
  }
  if (p->path == 1) {
    TORCH_CHECK(F_frow != nullptr, "mixed_cxx: final path 1 needs the mhcf table");
    return F_frow(&p->frow);
  }
  if (p->path != 2) return FPR_E_SHAPE;
  TORCH_CHECK(F_k1a != nullptr, "mixed_cxx: final path 2 needs the mhc table");
  for (int c = 0; c < p->n_chunks; ++c) {
    if ((rc = F_k1a(&p->k1a[c]))) return rc;
    if ((rc = F_k1b(&p->k1b[c]))) return rc;
  }
  return FPR_OK;
}

// ---- stage 1d: spans on per-rank threads (pieces.h FprProgEntry)
int fpr_pieces_moe_init_thread(int rank) __attribute__((weak));   // per-(thread, stream) cuBLAS handles first
std::vector<FprPieceFn> g_owner_fn;       // kind FPR_K_OWNER0 + i
std::vector<std::string> g_owner_name;
std::mutex g_owner_mu;

int run_entry(int r, const FprProgEntry& e) {
  switch (e.kind) {
    case FPR_K_BD: return bd(r, e.handle, g_sp[r].chain_input);
    case FPR_K_ROUTER: return router(r, e.handle, e.arg);
    case FPR_K_AG: return ag(r, e.handle, e.arg);
    case FPR_K_AGCE_JOIN: return agce_join_c(r, g_rank[r].main);
    case FPR_K_RS: return rs(r, e.handle, e.arg);
    case FPR_K_FINAL: return final_(r, e.handle);
    case FPR_K_JOIN: return fpr_drv_join(r, g_rank[r].main);
    case FPR_K_WAITEV:                 // handle 0: this rank had no pending Python event (entries stay aligned across ranks)
      return e.handle ? ck(cudaStreamWaitEvent(g_rank[r].main, (cudaEvent_t)e.handle, 0)) : FPR_OK;
    default: {
      const int i = e.kind - FPR_K_OWNER0;
      if (i < 0 || i >= (int)g_owner_fn.size() || g_owner_fn[i] == nullptr) return FPR_E_KIND;
      return g_owner_fn[i](r, e.handle, e.layer);
    }
  }
}

int run_program(int r, const FprProgEntry* p, int64_t n, int64_t* at) {
  for (int64_t i = 0; i < n; ++i) {
    int rc;
    try {
      rc = run_entry(r, p[i]);
    } catch (const std::exception& ex) {
      g_rank[r].err = std::string("kind ") + std::to_string(p[i].kind) + " layer " + std::to_string(p[i].layer) + ": " + ex.what();
      rc = FPR_E_EXCEPTION;
    }
    if (rc != FPR_OK) {
      *at = i;
      if (g_rank[r].err.empty()) g_rank[r].err = "kind " + std::to_string(p[i].kind) + " layer " + std::to_string(p[i].layer) +
                                                 ": FPR status " + std::to_string(rc);
      return rc;
    }
  }
  return FPR_OK;
}

struct Pool {
  std::thread th[W];
  std::mutex mu;
  std::condition_variable go, done;
  uint64_t gen = 0;
  int left = 0;
  const FprProgEntry* prog[W] = {};
  int64_t n[W] = {};
  int rc[W] = {};
  int64_t at[W] = {};
  int64_t t0_ns = 0;   // ENG_MIXED_CXX_STAMP: run_span's notify time
  bool up = false;
  void start() {
    if (up) return;
    up = true;
    for (int r = 0; r < W; ++r)
      th[r] = std::thread([this, r] {
        uint64_t seen = 0;
        bool inited = false;
        for (;;) {
          {
            std::unique_lock<std::mutex> lk(mu);
            go.wait(lk, [&] { return gen != seen; });
            seen = gen;
          }
          int rc_ = FPR_OK;
          int64_t at_ = -1;
          if (n[r] > 0) {
            if (cudaSetDevice(g_rank[r].dev) != cudaSuccess) rc_ = FPR_E_CUDA;
            else if (!inited) {                // once per driver thread, after the device bind: owners' thread init
              inited = true;
              if (fpr_pieces_moe_init_thread) rc_ = fpr_pieces_moe_init_thread(r);
            }
            if (rc_ == FPR_OK) {
              const int64_t ts = g_stamp ? now_ns() : 0;
              t_wait_ns = 0;
              rc_ = run_program(r, prog[r], n[r], &at_);
              if (g_stamp) {
                const int64_t te = now_ns();
                stamp_add(r, te - ts - t_wait_ns, t_wait_ns, ts - t0_ns);
              }
            }
          }
          std::lock_guard<std::mutex> lk(mu);
          rc[r] = rc_;
          at[r] = at_;
          if (--left == 0) done.notify_all();
        }
      });
    for (auto& t : th) t.detach();      // (blocked on `go` at process exit)
  }
};
Pool g_pool;

// one span: per rank, a packed FprProgEntry array; runs the 4 programs on the 4 driver threads, returns when every thread
// has issued its program (no device sync). Returns the per-rank status; error text: last_error(r).
std::atomic<bool> g_dead{false};       // a failed span leaves ranks mid-layer (handshake counters, in-flight collectives):
                                       // every later span is refused (the step's error is fatal for this engine)
std::mutex g_span_mu;                  // (C2) one span at a time, whichever thread issues it (Python or nd's native loop)

// the span body shared by the pybind run_span and the C export: no GIL, no Python. Returns the PRE-FLIGHT
// status (FPR_OK = the span ran and rc[W] holds every rank's status; else nothing ran and rc is untouched)
static int span_pool(const FprProgEntry* const* progs, const int64_t* n, int* rc) {
  std::lock_guard<std::mutex> sl(g_span_mu);
  if (g_dead.load()) return FPR_E_DEAD;
  for (int r = 0; r < W; ++r) {
    if (n[r] < 0 || (n[r] > 0 && progs[r] == nullptr)) return FPR_E_ARG;
    if (g_rank[r].main == nullptr) return FPR_E_INIT;   // set_streams first
  }
  for (int r = 0; r < W; ++r) g_rank[r].err.clear();
  {
    std::unique_lock<std::mutex> lk(g_pool.mu);
    g_pool.start();
    for (int r = 0; r < W; ++r) {
      g_pool.prog[r] = progs[r];
      g_pool.n[r] = n[r];
    }
    g_pool.left = W;
    ++g_pool.gen;
    if (g_stamp) g_pool.t0_ns = now_ns();
    g_pool.go.notify_all();
    g_pool.done.wait(lk, [] { return g_pool.left == 0; });
    for (int r = 0; r < W; ++r) rc[r] = g_pool.rc[r];
  }
  for (int r = 0; r < W; ++r)
    if (rc[r] != FPR_OK) g_dead.store(true);
  return FPR_OK;
}

// the serial reference's body (every entry for every rank before the next entry; ranks shorter than the longest skip)
static int span_serial(const FprProgEntry* const* progs, const int64_t* n, int* rc) {
  std::lock_guard<std::mutex> sl(g_span_mu);
  if (g_dead.load()) return FPR_E_DEAD;
  int64_t nmax = 0;
  for (int r = 0; r < W; ++r) {
    if (n[r] < 0 || (n[r] > 0 && progs[r] == nullptr)) return FPR_E_ARG;
    nmax = std::max(nmax, n[r]);
    g_rank[r].err.clear();
    rc[r] = FPR_OK;
  }
  int64_t iss[W] = {}, wt[W] = {};   // ENG_MIXED_CXX_STAMP: this span's per-rank issue / wait (no lag: one thread)
  for (int64_t i = 0; i < nmax; ++i)
    for (int r = 0; r < W; ++r) {
      if (rc[r] != FPR_OK || n[r] <= i) continue;
      int64_t at = -1;
      const int64_t ts = g_stamp ? now_ns() : 0;
      t_wait_ns = 0;
      rc[r] = run_program(r, progs[r] + i, 1, &at);
      if (g_stamp) {
        iss[r] += now_ns() - ts - t_wait_ns;
        wt[r] += t_wait_ns;
      }
    }
  if (g_stamp)
    for (int r = 0; r < W; ++r) stamp_add(r, iss[r], wt[r], 0);
  for (int r = 0; r < W; ++r)
    if (rc[r] != FPR_OK) g_dead.store(true);
  return FPR_OK;
}

static int first_rc(const int* rc) {
  for (int r = 0; r < W; ++r)
    if (rc[r] != FPR_OK) return rc[r];
  return FPR_OK;
}
extern "C" __attribute__((visibility("default"))) int fpr_drv_run_span(int n_ranks, const FprProgEntry* const* progs,
                                                                   const int64_t* n_entries, int* rc_per_rank) {
  if (n_ranks != W || progs == nullptr || n_entries == nullptr || rc_per_rank == nullptr) return FPR_E_ARG;
  const int st = span_pool(progs, n_entries, rc_per_rank);
  return st != FPR_OK ? st : first_rc(rc_per_rank);
}
extern "C" __attribute__((visibility("default"))) int fpr_drv_run_span_serial(int n_ranks, const FprProgEntry* const* progs,
                                                                          const int64_t* n_entries, int* rc_per_rank) {
  if (n_ranks != W || progs == nullptr || n_entries == nullptr || rc_per_rank == nullptr) return FPR_E_ARG;
  const int st = span_serial(progs, n_entries, rc_per_rank);
  return st != FPR_OK ? st : first_rc(rc_per_rank);
}

static void unpack_progs(const std::vector<py::bytes>& progs, std::string* buf, const FprProgEntry** p, int64_t* n) {
  TORCH_CHECK((int)progs.size() == W, "mixed_cxx: run_span needs ", W, " programs");
  for (int r = 0; r < W; ++r) {
    buf[r] = progs[r];
    TORCH_CHECK(buf[r].size() % sizeof(FprProgEntry) == 0, "mixed_cxx: program bytes");
    p[r] = (const FprProgEntry*)buf[r].data();
    n[r] = (int64_t)(buf[r].size() / sizeof(FprProgEntry));
  }
}
static void span_check(int st) {
  TORCH_CHECK(st != FPR_E_DEAD, "mixed_cxx: an earlier span failed; the driver refuses further spans (restart the engine)");
  TORCH_CHECK(st != FPR_E_INIT, "mixed_cxx: set_streams first");
  TORCH_CHECK(st != FPR_E_ARG, "mixed_cxx: run_span arguments");
}

std::vector<int> run_span(std::vector<py::bytes> progs) {
  std::string buf[W];
  const FprProgEntry* p[W];
  int64_t n[W];
  unpack_progs(progs, buf, p, n);
  int rc[W] = {};
  int st;
  {
    py::gil_scoped_release nogil;
    st = span_pool(p, n, rc);
  }
  span_check(st);
  return std::vector<int>(rc, rc + W);
}
// the same programs issued by the calling thread, rank by rank, entry by entry interleaved as Engine._issue does (every
// entry for every rank before the next entry; programs of equal length): the single-thread reference of run_span
std::vector<int> run_span_serial(std::vector<py::bytes> progs) {
  std::string buf[W];
  const FprProgEntry* p[W];
  int64_t n[W];
  unpack_progs(progs, buf, p, n);
  int rc[W] = {};
  span_check(span_serial(p, n, rc));   // (the calling thread, GIL held: as before)
  return std::vector<int>(rc, rc + W);
}
// the rank's error text of the last span; valid until the next span starts (it clears it)
extern "C" __attribute__((visibility("default"))) const char* fpr_drv_last_error(int rank) {
  return rank >= 0 && rank < W ? g_rank[rank].err.c_str() : "mixed_cxx: bad rank";
}
std::string last_error(int r) { return g_rank[r].err; }
// ENG_MIXED_CXX_STAMP: the per-rank accumulators in microseconds (read when no span runs, e.g. at exit)
py::dict stamps() {
  py::dict d;
  d["on"] = g_stamp;
  for (int r = 0; r < W; ++r) {
    const Stamp& k = g_st[r];
    const std::string s = "_r" + std::to_string(r);
    d[("spans" + s).c_str()] = k.spans;
    d[("issue_us" + s).c_str()] = k.issue_ns / 1000;
    d[("wait_us" + s).c_str()] = k.wait_ns / 1000;
    d[("lag_us" + s).c_str()] = k.lag_ns / 1000;
    d[("max_issue_us" + s).c_str()] = k.max_issue_ns / 1000;
    d[("max_wait_us" + s).c_str()] = k.max_wait_ns / 1000;
    d[("max_lag_us" + s).c_str()] = k.max_lag_ns / 1000;
  }
  return d;
}
int fpr_drv_bd(int rank, int64_t handle) { return bd(rank, handle, g_sp[rank].chain_input); }
int fpr_drv_register_piece(const char* kind, FprPieceFn fn) {
  std::lock_guard<std::mutex> lk(g_owner_mu);
  for (size_t i = 0; i < g_owner_name.size(); ++i)
    if (g_owner_name[i] == kind) {
      g_owner_fn[i] = fn;
      return FPR_K_OWNER0 + (int)i;
    }
  g_owner_name.emplace_back(kind);
  g_owner_fn.push_back(fn);
  return FPR_K_OWNER0 + (int)g_owner_fn.size() - 1;
}
void set_streams(int r, int64_t main) { g_rank[r].main = (cudaStream_t)main; }
// the segment's chain_input for a rank (StepParams): set by the span builder / 1b-i owner pieces from the bucket state
// right before use, so every reader (BD entries, engram_bd) sees the same value as the Python s["chain_input"]
void set_chain_input(int r, int v) { g_sp[r].chain_input = v; }
int64_t sizeof_prog_entry() { return (int64_t)sizeof(FprProgEntry); }
extern "C" __attribute__((visibility("default"))) int32_t fpr_drv_owner_kind(const char* kind) {
  if (kind == nullptr) return -1;
  std::lock_guard<std::mutex> lk(g_owner_mu);
  for (size_t i = 0; i < g_owner_name.size(); ++i)
    if (g_owner_name[i] == kind) return FPR_K_OWNER0 + (int32_t)i;
  return -1;
}
int64_t owner_kind(const std::string& name) { return fpr_drv_owner_kind(name.c_str()); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("register_table", &register_table);
  m.def("init_rank", &init_rank);
  m.def("sizeof_bd", &sizeof_bd);
  m.def("add_bd", &add_bd);
  m.def("join", &join);
  m.def("pending", &pending);
  m.def("bd", &bd);
  m.def("set_step_params", &set_step_params);
  m.def("sizeof_step_params", &sizeof_step_params);
  m.def("sizeof_router", &sizeof_router);
  m.def("sizeof_ag", &sizeof_ag);
  m.def("sizeof_rs", &sizeof_rs);
  m.def("sizeof_final", &sizeof_final);
  m.def("add_router", &add_router);
  m.def("add_ag", &add_ag);
  m.def("add_rs", &add_rs);
  m.def("add_final", &add_final);
  m.def("router", &router);
  m.def("ag", &ag);
  m.def("agce_join", &agce_join);
  m.def("agce_pending", &agce_pending);
  m.def("rs", &rs);
  m.def("final", &final_);
  m.def("run_span", &run_span);
  m.def("free_handle", &free_handle);
  m.def("live_handles", &live_handles);
  m.def("run_span_serial", &run_span_serial);
  m.def("last_error", &last_error);
  m.def("stamps", &stamps, "ENG_MIXED_CXX_STAMP: per-rank span timing (us)");
  m.def("set_streams", &set_streams);
  m.def("set_chain_input", &set_chain_input);
  m.def("dead", [] { return g_dead.load(); });
  m.def("sizeof_prog_entry", &sizeof_prog_entry);
  m.def("owner_kind", &owner_kind);
  if (fpr_pieces_moe_register) fpr_pieces_moe_register(m);
  if (fpr_pieces_attn_register) fpr_pieces_attn_register(m);
  if (fpr_pieces_head_register) fpr_pieces_head_register(m);
  if (fpr_pieces_engram_register) fpr_pieces_engram_register(m);
  if (fpr_pieces_c1mega_register) fpr_pieces_c1mega_register(m);
}

// one fpr_lt warm entry for the whole fpr_mixed_cxx .so (driver.cpp + pieces_*.cpp share one fpr_lt state): mixed_cxx_ltwarm.py
FPR_LT_DEFINE_WARM_ENTRY(fpr_mixed_cxx_lt_warm)
