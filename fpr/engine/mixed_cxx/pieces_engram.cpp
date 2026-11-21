// pieces_engram.cpp -- FPR's embed + Engram pieces (pieces_engram.h). Registered as owner kinds "embed" and "engram_bd"
// (stage 1d spans) plus pybind entries for 1b-i (embed / engram_bd from the Engine hooks). Every launch is the Python piece's:
// engram's C ABI for hash / gather / inject, ATen for the copies / index_select / where / wkv mm (same ops, same streams).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <mutex>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "pieces_engram.h"
#include "driver_api.h"
#include "fpr_lt.h"
#include "../../kernels/attn_dense/w8a16_cabi.h"

namespace {
constexpr int W = 4;
FprFn F_hp = nullptr, F_hash = nullptr, F_gather = nullptr, F_inject = nullptr;
// (v2) engram_w8a16_ext's own table
FprFn F_w8_init = nullptr, F_w8_gemm = nullptr;
bool g_w8_rank[W] = {};               // w8a16.init done on that rank's device (setup, never lazily)
cudaEvent_t g_ev_ws[W] = {};          // per rank: EngramPhys.prepare's est.wait_stream
int g_dev[W] = {-1, -1, -1, -1};

template <class T>
struct Slots {                        // as the driver's: adds / frees only from Python between spans
  std::vector<std::unique_ptr<T>> v;
  std::vector<int64_t> fl;
  int64_t add(std::unique_ptr<T> p) {
    if (!fl.empty()) {
      const int64_t h = fl.back();
      fl.pop_back();
      v[h] = std::move(p);
      return h;
    }
    v.push_back(std::move(p));
    return (int64_t)v.size() - 1;
  }
  const T* get(int64_t h) const { return (h >= 0 && h < (int64_t)v.size()) ? v[h].get() : nullptr; }
  void free(int64_t h) {
    if (h >= 0 && h < (int64_t)v.size() && v[h]) {
      v[h].reset();
      fl.push_back(h);
    }
  }
  int64_t live() const {
    int64_t n = 0;
    for (auto& p : v) n += p != nullptr;
    return n;
  }
};
Slots<FprPieceEmbed> g_embed;
Slots<FprPieceEngramBd> g_ebd;
inline int ck(cudaError_t e) { return e == cudaSuccess ? FPR_OK : FPR_E_CUDA; }

bool tables_ok() { return F_hp && F_hash && F_gather && F_inject; }
void resolve_fns() {
  F_hp = fpr_drv_fn("engram", "engram.hash_params", sizeof(FprEngramHashParamsArgs));
  F_hash = fpr_drv_fn("engram", "engram.hash", sizeof(FprEngramHashArgs));
  F_gather = fpr_drv_fn("engram", "engram.gather", sizeof(FprEngramGatherArgs));
  F_inject = fpr_drv_fn("engram", "engram.inject", sizeof(FprEngramInjectArgs));
}

// (v2) EngramPhys.run_wkv's "w8a16" impl on stream st: for r0 in range(0, T, 128): gemm8(x[r0:r1], y[r0:r1]) (bf16 x [T, K]
// row stride x_ld, y [T, N] row stride ldy; splits 1, no ws / cnt = the pybind call). Shared by prep_run and the self-test.
int w8_wkv(int r, int dev, cudaStream_t st, const void* x, int64_t x_ld, void* y, int64_t ldy, int64_t T, int64_t K, int64_t N,
           const void* packed, int64_t nchunk, int64_t tiles, int64_t rpg, int64_t kxg) {
  if (F_w8_gemm == nullptr || r < 0 || r >= W || !g_w8_rank[r]) return FPR_E_INIT;
  const int64_t es = 2;                                                       // bf16 bytes
  for (int64_t r0 = 0; r0 < T; r0 += 128) {
    const int64_t r1 = std::min<int64_t>(T, r0 + 128);
    FprW8a16GemmArgs a{};
    a.call.stream = st;
    a.call.device = dev;
    a.call.rank = 0;                                                          // = the pybind path's
    a.x = static_cast<const char*>(x) + r0 * x_ld * es;                       // x[r0:r1]
    a.ntok = r1 - r0;
    a.kx = K;
    a.x_ld = x_ld;
    a.packed = static_cast<const uint8_t*>(packed);
    a.nchunk = nchunk;
    a.tiles = tiles;
    a.y = static_cast<char*>(y) + r0 * ldy * es;                              // y[r0:r1]
    a.ldy = ldy;
    a.y_cols = N;
    a.rows_per_group = rpg;
    a.kx_per_group = kxg;
    a.splits = 1;
    if (int rc = F_w8_gemm(&a)) return rc;
  }
  return FPR_OK;
}

// ENG_CXX_EMBED_W8_SELFTEST=1: the first W8A16 prep per (rank, R)
// recomputes each layer's wkv from the same gathered rows into a scratch tensor (w8_wkv, est) and compares it bitwise with what
// the dispatch wrote (wy). With the setup self-test (w8_wkv == Python's gemm8 loop) this checks the real dispatch end to end on
// step data: flag, pointers, strides, slices. One host sync per checked (rank, R, layer); stderr line each.
bool w8_dispatch_selftest_on() {
  static const bool on = [] { const char* e = std::getenv("ENG_CXX_EMBED_W8_SELFTEST"); return e != nullptr && e[0] == '1'; }();
  return on;
}
std::mutex g_w8st_mu;
std::set<std::pair<int, int>> g_w8st_done;
void w8_dispatch_check(int r, int dev, const FprPieceEngramPrep* p, int l) {
  {
    std::lock_guard<std::mutex> lk(g_w8st_mu);
    if (!g_w8st_done.insert({r * 1000000 + l, p->R}).second) return;
  }
  auto ob = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, dev);
  auto y = at::from_blob(p->wy[l], {p->R, p->N}, {p->wy_st0[l], p->wy_st1[l]}, ob);
  auto y2 = at::empty({p->R, p->N}, ob);
  const int rc = w8_wkv(r, dev, p->est, p->wx[l], p->wx_st0[l], y2.data_ptr(), y2.stride(0), p->R, p->K, p->N, p->w8_packed[l],
                        p->w8_nchunk[l], p->w8_tiles[l], p->w8_rpg[l], p->w8_kxg[l]);
  const bool same = rc == FPR_OK && cudaStreamSynchronize(p->est) == cudaSuccess && at::equal(y, y2);
  std::fprintf(stderr, "pieces_engram: W8 DISPATCH SELFTEST rank %d R %d layer %d: %s\n", r, (int)p->R, l,
               rc != FPR_OK ? "w8_wkv failed" : same ? "IDENTICAL" : "DIFFER");
}

int prep_run(int r, const FprPieceEngramPrep* p) {
  int rc;
  if ((rc = ck(cudaEventRecord(g_ev_ws[r], p->main)))) return rc;            // est.wait_stream
  if ((rc = ck(cudaStreamWaitEvent(p->est, g_ev_ws[r], 0)))) return rc;
  const int dev = g_dev[r];
  const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(p->est, (c10::DeviceIndex)dev));
  if (p->resolve) {                                                           // _resolve_starts
    auto oi = at::TensorOptions().dtype(at::kInt).device(at::kCUDA, dev);
    auto b = at::from_blob(p->b_start, {p->S}, oi);
    auto src = at::from_blob(p->src, {p->S}, oi.dtype((c10::ScalarType)p->src_dtype));
    auto neg = b < 0;                                                         // Python's order: (b < 0), then the cast
    auto si = src.to(at::kInt);
    at::where_out(b, neg, si, b);
  }
  if ((rc = F_hash(&p->hash))) return rc;
  auto ob = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, dev);
  for (int l = 0; l < 2; ++l) {
    if ((rc = F_gather(&p->gather[l]))) return rc;
    if (p->w8) {                                                              // (v2) run_wkv "w8a16": 128-row slices
      if ((rc = w8_wkv(r, dev, p->est, p->wx[l], p->wx_st0[l], p->wy[l], p->wy_st0[l], p->R, p->K, p->N, p->w8_packed[l],
                       p->w8_nchunk[l], p->w8_tiles[l], p->w8_rpg[l], p->w8_kxg[l])))
        return rc;
      if (w8_dispatch_selftest_on()) w8_dispatch_check(r, dev, p, l);
      if ((rc = ck(cudaEventRecord(p->ev_l[l], p->est)))) return rc;
      continue;
    }
    auto x = at::from_blob(p->wx[l], {p->R, p->K}, {p->wx_st0[l], p->wx_st1[l]}, ob);
    auto w = at::from_blob(p->ww[l], {p->N, p->K}, {p->ww_st0[l], p->ww_st1[l]}, ob);
    auto y = at::from_blob(p->wy[l], {p->R, p->N}, {p->wy_st0[l], p->wy_st1[l]}, ob);
    if (p->stage_chunks > 0) {                                         // engram_stage.staged_mm (FPR_ENGRAM_STAGE)
      const int64_t C = p->stage_chunks, N = p->N;
      auto buf = at::from_blob(p->stage_buf, {(N + C - 1) / C, p->K}, ob);
      for (int64_t i = 0; i < C; ++i) {
        const int64_t n0 = i * N / C, n1 = (i + 1) * N / C;
        auto b = buf.narrow(0, 0, n1 - n0);
        b.copy_(w.narrow(0, n0, n1 - n0), /*non_blocking=*/true);             // buf[:n].copy_(w[n0:n1], non_blocking=True)
        auto yo = y.narrow(1, n0, n1 - n0);
        fpr_lt::mm(yo, x, b);                                                 // torch.mm(x, b.t(), out=y[:, n0:n1]) (fpr_lt)
      }
    } else {
      fpr_lt::mm(y, x, w);                                                    // torch.mm(x, wbf[l].t(), out=y) (fpr_lt)
    }
    if ((rc = ck(cudaEventRecord(p->ev_l[l], p->est)))) return rc;
  }
  return FPR_OK;
}

int embed_c(int rank, int64_t handle, int /*layer*/) {
  const FprPieceEmbed* p = g_embed.get(handle);
  if (p == nullptr || g_dev[rank] < 0 || (p->has_engram && !tables_ok())) return FPR_E_INIT;
  const int dev = g_dev[rank];
  {
    const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)dev));
    auto o = at::TensorOptions().device(at::kCUDA, dev);
    auto tok = at::from_blob(p->tok, {p->R}, o.dtype((c10::ScalarType)p->tok_dtype));
    if (p->has_engram) {
      auto ids = at::from_blob(p->ids32, {p->R}, o.dtype(at::kInt));
      ids.copy_(tok);                                                         // hooks.ids32[r][:R].copy_(s["tok"])
    }
  }
  if (p->has_engram)
    if (int rc = prep_run(rank, &p->prep)) return rc;
  const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)dev));
  auto o = at::TensorOptions().device(at::kCUDA, dev);
  auto tok = at::from_blob(p->tok, {p->R}, o.dtype((c10::ScalarType)p->tok_dtype));
  auto tab = at::from_blob(p->emb_tab, {p->emb_rows, p->D}, {p->emb_st0, 1}, o.dtype(at::kBFloat16));
  auto tmp = at::from_blob(p->emb_tmp, {p->R, p->D}, o.dtype(at::kBFloat16));
  at::index_select_out(tmp, tab, 0, tok);                                     // torch.index_select(emb, 0, tok)
  auto res0 = at::from_blob(p->res0, {p->R, p->HC, p->D}, {p->res0_st0, p->res0_st1, 1}, o.dtype(at::kBFloat16));
  res0.copy_(tmp.unsqueeze(1).expand({p->R, p->HC, p->D}));
  return FPR_OK;
}

int engram_bd_c(int rank, int64_t handle, int /*layer*/) {
  const FprPieceEngramBd* p = g_ebd.get(handle);
  if (p == nullptr || F_inject == nullptr) return FPR_E_ARG;
  int rc;
  if (p->inject_join && (rc = fpr_drv_join(rank, p->main))) return rc;      // BISECT inject_join: _join_mix(s, pf=False)
  if ((rc = ck(cudaStreamWaitEvent(p->main, p->ev_l, 0)))) return rc;          // inject: main waits ev[l]
  const FprStepParams* sp = fpr_drv_step_params(rank);
  const int set = (p->first_layer && !sp->chain_input) ? 0 : 1;
  if ((rc = F_inject(&p->inj[set]))) return rc;
  return fpr_drv_bd(rank, p->bd);
}

// ---- pybind (Python side: mixed_cxx_engram.py)
void embed_init_rank(int r, int device) {         // the embed piece without Engram needs only the device
  TORCH_CHECK(r >= 0 && r < W, "pieces_engram: rank");
  const c10::cuda::CUDAGuard g((c10::DeviceIndex)device);
  g_dev[r] = device;
  if (!g_ev_ws[r]) C10_CUDA_CHECK(cudaEventCreateWithFlags(&g_ev_ws[r], cudaEventDisableTiming));
}
void engram_init_rank(int r, int device) {
  resolve_fns();
  TORCH_CHECK(tables_ok(), "pieces_engram: the engram table is not registered");
  embed_init_rank(r, device);
}
// (v2) the W8A16 wkv: resolve engram_w8a16_ext's table (its Python module is loaded: RTLD_NOLOAD) and run w8a16.init on
// every rank's device at driver setup (sets the dynamic-smem attribute of every instantiation = loads them: no lazy module
// load on a pool thread later). False (with the reason) = the w8 branch stays off; the v1 paths are unaffected.
std::string engram_w8_init(const std::string& lib) {
  F_w8_init = F_w8_gemm = nullptr;
  void* h = dlopen(lib.c_str(), RTLD_NOW | RTLD_NOLOAD);
  if (h == nullptr) return std::string("not loaded: ") + lib;
  auto f = reinterpret_cast<FprTableFn>(dlsym(h, "fpr_w8a16_table"));
  if (f == nullptr) return "no fpr_w8a16_table in " + lib;
  const FprExtTable* t = f();
  if (t == nullptr || t->abi_version != FPR_CABI_VERSION) return "w8a16 ABI version mismatch";
  FprFn fi = nullptr, fg = nullptr;
  for (int i = 0; i < t->n_fns; ++i) {
    const FprFnEntry& e = t->fns[i];
    if (std::strcmp(e.name, "w8a16.init") == 0 && e.args_size == (int64_t)sizeof(FprW8a16InitArgs)) fi = e.fn;
    if (std::strcmp(e.name, "w8a16.gemm") == 0 && e.args_size == (int64_t)sizeof(FprW8a16GemmArgs)) fg = e.fn;
  }
  if (fi == nullptr || fg == nullptr) return "w8a16.init / w8a16.gemm missing or args size drift";
  F_w8_init = fi, F_w8_gemm = fg;
  return "";
}
int engram_w8_init_rank(int r, int device) {
  TORCH_CHECK(r >= 0 && r < W, "pieces_engram: rank");
  if (F_w8_init == nullptr) return FPR_E_INIT;
  const c10::cuda::CUDAGuard g((c10::DeviceIndex)device);   // c_init checks the current device
  FprW8a16InitArgs a{};
  a.call.stream = nullptr, a.call.device = device, a.call.rank = r;
  const int rc = F_w8_init(&a);
  g_w8_rank[r] = rc == FPR_OK;
  return rc;
}
// (v2) self-test entry (ENG_CXX_EMBED_W8_SELFTEST): w8_wkv on the current stream for tensors x [T, K] / y [T, N] / packed
int engram_w8_wkv_test(int r, torch::Tensor x, torch::Tensor y, torch::Tensor packed, int64_t rpg, int64_t kxg) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && y.dtype() == torch::kBFloat16 && x.dim() == 2 && y.dim() == 2 && x.stride(1) == 1 &&
              y.stride(1) == 1 && x.size(0) == y.size(0) && packed.dim() == 3 && packed.is_contiguous(), "w8_wkv_test args");
  const int dev = x.get_device();
  return w8_wkv(r, dev, at::cuda::getCurrentCUDAStream(dev).stream(), x.data_ptr(), x.stride(0), y.data_ptr(), y.stride(0),
                x.size(0), x.size(1), y.size(1), packed.data_ptr(), packed.size(0), packed.size(1), rpg, kxg);
}
void fix_prep(FprPieceEngramPrep* q) {
  const FprEngramHashParamsArgs a{{nullptr, 0, 0}, q->ctx, &q->hp};
  TORCH_CHECK(F_hp != nullptr && F_hp(&a) == FPR_OK, "pieces_engram: engram.hash_params(ctx ", q->ctx, ")");
  q->hash.hp = &q->hp;
}
int64_t add_embed(py::bytes b) {
  std::string s = b;
  TORCH_CHECK(s.size() == sizeof(FprPieceEmbed), "pieces_engram: FprPieceEmbed is ", s.size(), " bytes, the C++ ",
              sizeof(FprPieceEmbed));
  auto p = std::make_unique<FprPieceEmbed>();
  std::memcpy(p.get(), s.data(), s.size());
  TORCH_CHECK(p->version == FPR_PIECES_VERSION, "pieces_engram: version");
  if (p->has_engram) {
    TORCH_CHECK(p->prep.version == FPR_ENGRAM_PREP_VERSION, "pieces_engram: FprPieceEngramPrep version ", p->prep.version,
                ", the C++ ", FPR_ENGRAM_PREP_VERSION);
    TORCH_CHECK(!p->prep.w8 || F_w8_gemm != nullptr, "pieces_engram: a W8A16 prep without engram_w8_init");
    fix_prep(&p->prep);
  }
  return g_embed.add(std::move(p));
}
int64_t add_engram_bd(py::bytes b) {
  std::string s = b;
  TORCH_CHECK(s.size() == sizeof(FprPieceEngramBd), "pieces_engram: FprPieceEngramBd is ", s.size(), " bytes, the C++ ",
              sizeof(FprPieceEngramBd));
  auto p = std::make_unique<FprPieceEngramBd>();
  std::memcpy(p.get(), s.data(), s.size());
  TORCH_CHECK(p->version == FPR_PIECES_VERSION, "pieces_engram: version");
  return g_ebd.add(std::move(p));
}
void free_engram(const std::string& kind, int64_t h) {
  if (kind == "embed") g_embed.free(h);
  else if (kind == "engram_bd") g_ebd.free(h);
}
py::dict live_engram() {
  py::dict d;
  d["embed"] = g_embed.live(), d["engram_bd"] = g_ebd.live();
  return d;
}
}  // namespace

void fpr_pieces_engram_register(pybind11::module_& m) {
  m.def("engram_init_rank", &engram_init_rank);
  m.def("embed_init_rank", &embed_init_rank);
  m.def("sizeof_embed", [] { return (int64_t)sizeof(FprPieceEmbed); });
  m.def("sizeof_engram_bd", [] { return (int64_t)sizeof(FprPieceEngramBd); });
  m.def("sizeof_engram_parts", [] {
    py::dict d;
    d["prep"] = (int64_t)sizeof(FprPieceEngramPrep), d["hash"] = (int64_t)sizeof(FprEngramHashArgs);
    d["gather"] = (int64_t)sizeof(FprEngramGatherArgs), d["inject"] = (int64_t)sizeof(FprEngramInjectArgs);
    d["hp"] = (int64_t)sizeof(FprEngramHashParams);
    return d;
  });
  m.def("engram_w8_init", &engram_w8_init);
  m.def("engram_w8_init_rank", &engram_w8_init_rank);
  m.def("engram_w8_wkv_test", &engram_w8_wkv_test);
  m.def("prep_version", [] { return (int64_t)FPR_ENGRAM_PREP_VERSION; });
  m.def("add_embed", &add_embed);
  m.def("add_engram_bd", &add_engram_bd);
  m.def("embed", [](int r, int64_t h) { return embed_c(r, h, -1); });
  m.def("engram_bd", [](int r, int64_t h, int L) { return engram_bd_c(r, h, L); });
  m.def("free_engram", &free_engram);
  m.def("live_engram", &live_engram);
  fpr_drv_register_piece("embed", &embed_c);
  fpr_drv_register_piece("engram_bd", &engram_bd_c);
}
