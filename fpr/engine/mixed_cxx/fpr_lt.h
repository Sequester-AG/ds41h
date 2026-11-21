// fpr_lt.h: y[b] = a[b] @ w[b]^T through cublasLtMatmul with a plan cached per (device
// batch, M, N, K, lda, ldw, ldy, batch strides, out dtype) instead of at::mm_out / at::bmm_out.
//
// v2: a CLOSED algo set. warm() (driver init, before any step, on the thread / stream
// that will issue) builds every given shape's plan, launches each DISTINCT algo once on scratch and self-checks every key
// (one sync each, at init). After warm_done, a new key whose heuristic algo is not in the warmed set takes the first warmed
// algo cublasLtMatmulAlgoCheck accepts for it (logged "fpr_lt: unwarmed key", counted), else the ATen path (logged,
// counted): no cuBLASLt kernel module is ever first-loaded by a step through this header.
//
// Safety: every key's FIRST use (in warm(), or later) also runs the ATen op into scratch and compares the bytes (one
// stream sync); a mismatch pins the key to ATen for the process. Outside the fast path (a, w not bf16; y not bf16 / fp32;
// a non-unit inner stride; batched fp32 out; a capturing stream; FPR_LT=0; an Lt error): ATen.
// The stream is NOT in the plan key; the workspace (32 MB) is one per (device, stream), allocated once and kept.
// v3: ONE state per .so, shared by all its translation units: a named namespace, external-linkage inline
// functions (the static State in st() is merged across TUs by the linker) under hidden visibility (no cross-.so merging;
// torch extensions load RTLD_LOCAL anyway). Each .so warms its own state (FPR_LT_DEFINE_WARM_ENTRY(name) exports a C entry
// for that .so). warm() also pre-allocates the 32 MB workspace of every stream it is given (no allocation while serving).
// Thread safe (one mutex).
//   FPR_LT=0        every call takes ATen        FPR_LT_LOG=1    a stderr line per new key
//   FPR_LT_DUMP=<path>  (v4) at exit (and via <name>_dump(path) from FPR_LT_DEFINE_WARM_ENTRY) append every key this .so
//                   planned: "batch M N K lda ldw ldy sa sw sy out_f32 site" (site = the first caller's function name)
#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>
#include <string>

// C ABI shape for warm(): y[b] (M x N, row ld ldy, batch stride sy) = a[b] (M x K, lda, sa) @ w[b]^T (w[b]: N x K, ldw, sw)
typedef struct {
  int64_t batch, M, N, K, lda, ldw, ldy, sa, sw, sy;
  int32_t out_f32, pad_;
} FprLtShape;

#if defined(__CUDACC__)
// (v7) .cu translation units: nvcc's front end (cudafe++, EDG) asserts on this header's host containers under -std=c++20
// A .cu TU gets only this forwarder; the implementation lives in
// the .so's host TU that expands FPR_LT_DEFINE_WARM_ENTRY (every .so using fpr_lt has exactly one: it also defines
// fpr_lt_mm_host, hidden, so the .cu calls share the .so's one state).
__attribute__((visibility("hidden"))) void fpr_lt_mm_host(at::Tensor& y, const at::Tensor& a, const at::Tensor& w, const char* site);
namespace fpr_lt {
inline void mm(at::Tensor& y, const at::Tensor& a, const at::Tensor& w, const char* site = "cu") { fpr_lt_mm_host(y, a, w, site); }
inline void mm_nt(at::Tensor& y, const at::Tensor& a, const at::Tensor& w, const char* site = "cu") { fpr_lt_mm_host(y, a, w, site); }
}  // namespace fpr_lt
#else
#define FPR_LT_SITE __builtin_FUNCTION()
#pragma GCC visibility push(hidden)
namespace fpr_lt {

struct Key {
  int dev;
  int64_t b, M, N, K, lda, ldw, ldy, sa, sw, sy;
  int f32;
  bool operator<(const Key& o) const {   // (v6: no std::tie / tuple compare: cudafe++ under -std=c++20 asserts on it)
    const int64_t x[12] = {dev, b, M, N, K, lda, ldw, ldy, sa, sw, sy, f32};
    const int64_t y[12] = {o.dev, o.b, o.M, o.N, o.K, o.lda, o.ldw, o.ldy, o.sa, o.sw, o.sy, o.f32};
    for (int i = 0; i < 12; ++i)
      if (x[i] != y[i]) return x[i] < y[i];
    return false;
  }
};

struct Plan {
  bool ok = false;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cublasLtMatmulAlgo_t algo{};
  size_t ws = 0;
  const char* site = "?";   // (v4) the first caller's function (__builtin_FUNCTION)
  cublasLtHandle_t h = nullptr;   // (v9) the plan's device's handle (launches read it without the state lock)
  bool busy = false;   // (v10) a thread is building it (others take ATen meanwhile)
  bool rec = false;    // (v10) recorded only (FPR_LT_RECORD_ONLY): no plan
};

struct State {
  std::mutex mu;
  std::map<int, cublasLtHandle_t> hs;   // (v9) one cuBLASLt handle per device, created with that device current
  std::map<Key, Plan> plans;
  std::map<std::pair<int, uintptr_t>, at::Tensor> ws;          // (device, stream) -> 32 MB
  std::map<int, std::vector<cublasLtMatmulAlgo_t>> warmed;     // device -> distinct algos launched at warm()
  std::map<int, bool> warm_done;
  uint64_t hits = 0, keys = 0, pass = 0, fail = 0, fallback = 0, unwarmed = 0, unwarmed_aten = 0;
  int dump_to(const char* path) {   // (v4; caller holds mu or is the destructor)
    FILE* f = path && *path ? std::fopen(path, "a") : nullptr;
    if (f == nullptr) return -1;
    int n = 0;
    for (const auto& kv : plans) {
      const Key& k = kv.first;
      std::fprintf(f, "%lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %d %s\n", (long long)k.b, (long long)k.M,
                   (long long)k.N, (long long)k.K, (long long)k.lda, (long long)k.ldw, (long long)k.ldy, (long long)k.sa,
                   (long long)k.sw, (long long)k.sy, k.f32, kv.second.site ? kv.second.site : "?");
      ++n;
    }
    std::fclose(f);
    return n;
  }
  ~State() {
    if (const char* d = std::getenv("FPR_LT_DUMP")) {
      const int n = dump_to(d);
      std::fprintf(stderr, "fpr_lt: dumped %d keys to %s\n", n, d);
    }
    if (keys || fallback)
      std::fprintf(stderr, "fpr_lt: summary keys %llu self-check pass %llu fail %llu fast %llu fallback %llu unwarmed %llu "
                   "(-> ATen %llu)\n", (unsigned long long)keys, (unsigned long long)pass, (unsigned long long)fail,
                   (unsigned long long)hits, (unsigned long long)fallback, (unsigned long long)unwarmed,
                   (unsigned long long)unwarmed_aten);
  }
};
inline State& st() {
  static State s;
  return s;
}
inline bool env_on(const char* k, bool dflt) {
  const char* v = std::getenv(k);
  return v == nullptr ? dflt : v[0] == '1';
}
inline bool enabled() {
  static const bool on = env_on("FPR_LT", true);
  return on;
}
inline bool logging() {
  static const bool on = env_on("FPR_LT_LOG", false);
  return on;
}
constexpr size_t kWs = 32ull << 20;

inline bool same_algo(const cublasLtMatmulAlgo_t& x, const cublasLtMatmulAlgo_t& y) {
  return std::memcmp(&x, &y, sizeof(x)) == 0;
}

// the ATen reference / fallback for the same op
inline void aten(at::Tensor& y, const at::Tensor& a, const at::Tensor& w) {
  if (a.dim() == 3) {
    at::bmm_out(y, a, w.transpose(1, 2));
  } else if (y.scalar_type() == at::kFloat) {
    at::mm_out(y, a, w.t(), at::kFloat);
  } else {
    at::mm_out(y, a, w.t());
  }
}

inline Key key_of(const at::Tensor& y, const at::Tensor& a, const at::Tensor& w) {
  Key k{};
  k.dev = a.get_device();
  const bool b3 = a.dim() == 3;
  k.b = b3 ? a.size(0) : 1;
  k.M = a.size(b3 ? 1 : 0), k.K = a.size(b3 ? 2 : 1), k.N = w.size(b3 ? 1 : 0);
  k.lda = a.stride(b3 ? 1 : 0), k.ldw = w.stride(b3 ? 1 : 0), k.ldy = y.stride(b3 ? 1 : 0);
  k.sa = b3 ? a.stride(0) : 0, k.sw = b3 ? w.stride(0) : 0, k.sy = b3 ? y.stride(0) : 0;
  k.f32 = y.scalar_type() == at::kFloat;
  return k;
}

inline bool fast_ok(const at::Tensor& y, const at::Tensor& a, const at::Tensor& w) {
  const int d = a.dim();
  if (!((d == 2 || d == 3) && w.dim() == d && y.dim() == d)) return false;
  if (a.scalar_type() != at::kBFloat16 || w.scalar_type() != at::kBFloat16) return false;
  if (!(y.scalar_type() == at::kBFloat16 || (d == 2 && y.scalar_type() == at::kFloat))) return false;
  const int i = d - 1;
  if (a.stride(i) != 1 || w.stride(i) != 1 || y.stride(i) != 1) return false;
  if (a.size(i) != w.size(i) || y.size(i - 1) != a.size(i - 1) || y.size(i) != w.size(i - 1) || a.size(i - 1) <= 0) return false;
  if (d == 3 && (w.size(0) != a.size(0) || y.size(0) != a.size(0))) return false;
  return true;
}

// descriptors + layouts for key k (column-major view: A = w as K x N (ld ldw, op T), B = a as K x M (ld lda),
// C = y as N x M (ld ldy); batched: strided batch offsets sw / sa / sy)
inline bool make_desc(const Key& k, Plan& p) {
  auto ok = [](cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS; };
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  const cudaDataType_t ct = k.f32 ? CUDA_R_32F : CUDA_R_16BF;
  if (!ok(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F))) return false;
  if (!ok(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)))) return false;
  if (!ok(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)))) return false;
  if (!ok(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, k.K, k.N, k.ldw))) return false;
  if (!ok(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, k.K, k.M, k.lda))) return false;
  if (!ok(cublasLtMatrixLayoutCreate(&p.lc, ct, k.N, k.M, k.ldy))) return false;
  if (k.b > 1) {
    const int32_t bc = (int32_t)k.b;
    const cublasLtMatrixLayout_t ls[3] = {p.la, p.lb, p.lc};
    const int64_t ss[3] = {k.sw, k.sa, k.sy};
    for (int q = 0; q < 3; ++q) {
      if (!ok(cublasLtMatrixLayoutSetAttribute(ls[q], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)))) return false;
      if (!ok(cublasLtMatrixLayoutSetAttribute(ls[q], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &ss[q], sizeof(ss[q])))) return false;
    }
  }
  return true;
}

// v9's mm() held the process-wide mutex across a new key's plan build + launch + self-check sync; a
// pool thread's lazy cuBLASLt module load under it waited on device progress that the other ranks' pool threads, blocked on
// the same mutex, never made: DEADLOCK). Now the mutex guards ONLY map lookups / inserts and counters; handle creation,
// workspace allocation, plan builds, launches and self-checks run OUTSIDE it, with a "busy" marker per key (a second thread
// meeting a key under construction takes ATen). After warm_done a NEW key never plans / self-checks on a call path: ATen,
// counted ("unwarmed -> ATen"). FPR_LT_RECORD_ONLY=1: every call runs ATen and only records its key (for FPR_LT_DUMP).

// the device's cuBLASLt handle: lookup under the lock, creation outside it, insert-if-absent
inline cublasLtHandle_t handle_get(State& S, int dev) {
  {
    std::lock_guard<std::mutex> g(S.mu);
    auto it = S.hs.find(dev);
    if (it != S.hs.end()) return it->second;
  }
  cublasLtHandle_t h = nullptr;
  {
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)dev);
    if (cublasLtCreate(&h) != CUBLAS_STATUS_SUCCESS) return nullptr;
  }
  std::lock_guard<std::mutex> g(S.mu);
  auto it = S.hs.find(dev);
  if (it != S.hs.end()) {   // (another thread won: keep its handle)
    cublasLtDestroy(h);
    return it->second;
  }
  S.hs[dev] = h;
  return h;
}

// heuristic algo for p (index 0) -> p.algo / p.ws; false: none (no state access)
inline bool heuristic(Plan& p) {
  cublasLtMatmulPreference_t pref = nullptr;
  if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return false;
  size_t wsb = kWs;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb));
  cublasLtMatmulHeuristicResult_t r{};
  int n = 0;
  const bool ok = cublasLtMatmulAlgoGetHeuristic(p.h, p.op, p.la, p.lb, p.lc, p.lc, pref, 1, &r, &n) == CUBLAS_STATUS_SUCCESS && n > 0;
  cublasLtMatmulPreferenceDestroy(pref);
  if (ok) p.algo = r.algo, p.ws = r.workspaceSize;
  return ok;
}

// the (device, stream) workspace: lookup under the lock, allocation outside it, insert-if-absent
inline void* ws_get(State& S, int dev, cudaStream_t s, const at::TensorOptions& o) {
  const std::pair<int, uintptr_t> key{dev, reinterpret_cast<uintptr_t>(s)};
  {
    std::lock_guard<std::mutex> g(S.mu);
    auto it = S.ws.find(key);
    if (it != S.ws.end()) return it->second.data_ptr();
  }
  at::Tensor t = at::empty({(int64_t)kWs}, o.dtype(at::kByte));
  std::lock_guard<std::mutex> g(S.mu);
  auto it = S.ws.find(key);
  if (it != S.ws.end()) return it->second.data_ptr();   // (another thread won: ours is freed on scope exit, stream-ordered)
  S.ws[key] = t;
  return t.data_ptr();
}

inline bool launch(const Plan& p, at::Tensor& y, const at::Tensor& a, const at::Tensor& w, void* wsp, cudaStream_t s) {
  const float one = 1.f, zero = 0.f;
  return cublasLtMatmul(p.h, p.op, &one, w.data_ptr(), p.la, a.data_ptr(), p.lb, &zero, y.data_ptr(), p.lc, y.data_ptr(),
                        p.lc, &p.algo, wsp, p.ws, s) == CUBLAS_STATUS_SUCCESS;
}

// a key's plan WITHOUT the lock: descriptors, heuristic, one launch + the self-check vs ATen (one stream sync). y holds the
// correct result afterwards either way. The caller stores the plan and the counters under the lock.
inline Plan build_plan(const Key& k, cublasLtHandle_t h, void* wsp, at::Tensor& y, const at::Tensor& a, const at::Tensor& w,
                       cudaStream_t cs) {
  Plan p;
  p.h = h;
  bool built = h != nullptr && wsp != nullptr && make_desc(k, p) && heuristic(p);
  bool same = false;
  if (built) {
    at::Tensor ref = at::empty_like(y);
    aten(ref, a, w);
    built = launch(p, y, a, w, wsp, cs);
    same = built && at::equal(ref, y);   // (syncs the stream: once per key, never under the state lock)
    if (!same) y.copy_(ref);
  } else {
    aten(y, a, w);
  }
  p.ok = same;
  if (logging() || !same)
    std::fprintf(stderr, "fpr_lt: dev %d b %lld M %lld N %lld K %lld ld %lld/%lld/%lld %s: %s\n", k.dev, (long long)k.b,
                 (long long)k.M, (long long)k.N, (long long)k.K, (long long)k.lda, (long long)k.ldw, (long long)k.ldy,
                 k.f32 ? "f32" : "bf16", same ? "self-check PASS" : (built ? "self-check FAIL -> ATen" : "no plan -> ATen"));
  return p;
}

inline void count_plan(State& S, const Plan& p) {   // (S.mu held)
  ++S.keys;
  if (p.ok) ++S.pass; else ++S.fail;
}

inline bool record_only() {
  static const bool on = env_on("FPR_LT_RECORD_ONLY", false);
  return on;
}

inline std::vector<std::string> env_list(const char* k) {
  std::vector<std::string> v;
  const char* e = std::getenv(k);
  if (e == nullptr) return v;
  std::string s(e), cur;
  for (char c : s) {
    if (c == ',') { if (!cur.empty()) v.push_back(cur); cur.clear(); } else cur += c;
  }
  if (!cur.empty()) v.push_back(cur);
  return v;
}
inline bool site_on(const char* site) {
  static const std::vector<std::string> off = env_list("FPR_LT_OFF"), only = env_list("FPR_LT_ONLY");
  const std::string s = site ? site : "?";
  for (const auto& x : off) if (s.find(x) != std::string::npos) return false;
  if (only.empty()) return true;
  for (const auto& x : only) if (s.find(x) != std::string::npos) return true;
  return false;
}
// y[b] = a[b] @ w[b]^T (2-D: y = a @ w^T; y bf16 or fp32 (2-D only); a, w bf16)
inline void mm(at::Tensor& y, const at::Tensor& a, const at::Tensor& w, const char* site = FPR_LT_SITE) {
  if (!enabled() || !site_on(site) || !fast_ok(y, a, w)) {
    if (enabled()) { std::lock_guard<std::mutex> g(st().mu); ++st().fallback; }
    aten(y, a, w);
    return;
  }
  State& S = st();
  if (record_only()) {   // (v10) the recording run: ATen, the key recorded (no plan, no heuristic, no self-check)
    const Key k = key_of(y, a, w);
    {
      std::lock_guard<std::mutex> g(S.mu);
      auto it = S.plans.find(k);
      if (it == S.plans.end()) {
        Plan p;
        p.site = site;
        p.rec = true;
        S.plans.emplace(k, p);
        ++S.keys;
      }
      ++S.fallback;
    }
    aten(y, a, w);
    return;
  }
  const c10::cuda::CUDAGuard dg(a.device());
  const cudaStream_t cs = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(cs, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
    { std::lock_guard<std::mutex> g(S.mu); ++S.fallback; }
    aten(y, a, w);   // (no plan creation / self-check inside a capture)
    return;
  }
  const Key k = key_of(y, a, w);
  Plan p;
  bool fast = false, build = false;
  {
    std::lock_guard<std::mutex> g(S.mu);
    auto it = S.plans.find(k);
    if (it != S.plans.end()) {
      if (it->second.ok && !it->second.busy) {
        p = it->second;
        fast = true;
        ++S.hits;
      } else {
        ++S.fallback;   // (busy: another thread builds it; or a pinned / unwarmed / failed key)
      }
    } else if (S.warm_done[k.dev]) {   // (v10) a NEW key after the warm: ATen for good, never planned on a call path
      Plan q;
      q.site = site;
      S.plans.emplace(k, q);
      ++S.unwarmed;
      ++S.unwarmed_aten;
      ++S.fallback;
      if (S.unwarmed <= 16 || (S.unwarmed & (S.unwarmed - 1)) == 0)
        std::fprintf(stderr, "fpr_lt: unwarmed key -> ATen (%llu): dev %d b %lld M %lld N %lld K %lld site %s\n",
                     (unsigned long long)S.unwarmed, k.dev, (long long)k.b, (long long)k.M, (long long)k.N, (long long)k.K, site);
    } else {   // (no warm on this device: build it here, outside the lock)
      Plan q;
      q.site = site;
      q.busy = true;
      S.plans.emplace(k, q);
      build = true;
    }
  }
  if (fast) {
    void* wsp = ws_get(S, k.dev, cs, a.options());
    if (!launch(p, y, a, w, wsp, cs)) aten(y, a, w);
    return;
  }
  if (!build) {
    aten(y, a, w);
    return;
  }
  Plan q = build_plan(k, handle_get(S, k.dev), ws_get(S, k.dev, cs, a.options()), y, a, w, cs);
  q.site = site;
  std::lock_guard<std::mutex> g(S.mu);
  S.plans[k] = q;
  count_plan(S, q);
}
// (v1 name) bf16 y = a @ w^T
inline void mm_nt(at::Tensor& y, const at::Tensor& a, const at::Tensor& w, const char* site = FPR_LT_SITE) {
  mm(y, a, w, site);
}
inline int dump(const char* path) {   // (v4) append every planned / recorded key of this .so to path; returns the count or -1
  State& S = st();
  std::lock_guard<std::mutex> lk(S.mu);
  return S.dump_to(path);
}

// warm (driver / pool init, or a handle's registration): every shape's plan on scratch tensors with the shape's strides
// (random bf16 inputs), each DISTINCT algo launched and every key self-checked, then warm_done for dev. (v10) the lock is
// held only for the map / counters, never across a build, launch or sync. Returns the number of distinct algos, or -1.
inline int warm(int dev, const FprLtShape* sh, int n, const cudaStream_t* streams = nullptr, int ns = 0) {
  const c10::cuda::CUDAGuard dg((c10::DeviceIndex)dev);
  State& S = st();
  const auto o = at::TensorOptions().dtype(at::kBFloat16).device(at::Device(at::kCUDA, dev));
  const cudaStream_t cs = c10::cuda::getCurrentCUDAStream(dev).stream();
  for (int q = 0; q < ns; ++q) ws_get(S, dev, streams[q], o);   // (every pool stream's workspace now, not while serving)
  const cublasLtHandle_t h = handle_get(S, dev);
  void* wsp = ws_get(S, dev, cs, o);
  // (v9d diag) FPR_LT_WARM_SYNC=1: device-wide sync + error check around each key; =entry (v10b): only at warm() entry (keeps
  // the warm's churn, still names the handle build whose earlier async work faulted)
  static const int wsync = [] { const char* e = std::getenv("FPR_LT_WARM_SYNC"); return !e ? 0 : (std::strcmp(e, "entry") == 0 ? 2 : (e[0] == '1')); }();
  auto chk = [&](const char* when, int i) {
    if (!wsync || (wsync == 2 && i >= 0)) return;
    const cudaError_t e = cudaDeviceSynchronize();
    const cudaError_t l = cudaGetLastError();
    if (e != cudaSuccess || l != cudaSuccess || i < 0)
      std::fprintf(stderr, "fpr_lt: WARM_SYNC dev %d %s shape %d / %d: sync %s, last %s\n", dev, when, i, n, cudaGetErrorString(e),
                   cudaGetErrorString(l));
  };
  chk("ENTRY (before any key)", -1);
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    const FprLtShape& s = sh[i];
    const bool b3 = s.batch > 1;
    auto mk = [&](int64_t rows, int64_t cols, int64_t ld, int64_t bs, at::ScalarType t) {
      const int64_t span = (b3 ? (s.batch - 1) * bs : 0) + (rows - 1) * ld + cols;
      at::Tensor base = (t == at::kFloat ? at::zeros({span}, o.dtype(at::kFloat)) : (at::randn({span}, o.dtype(at::kFloat)) * 0.05).to(t));
      return b3 ? base.as_strided({s.batch, rows, cols}, {bs, ld, 1}) : base.as_strided({rows, cols}, {ld, 1});
    };
    at::Tensor a = mk(s.M, s.K, s.lda, s.sa, at::kBFloat16), w = mk(s.N, s.K, s.ldw, s.sw, at::kBFloat16);
    at::Tensor y = mk(s.M, s.N, s.ldy, s.sy, s.out_f32 ? at::kFloat : at::kBFloat16);
    if (!fast_ok(y, a, w)) { ++bad; continue; }
    const Key k = key_of(y, a, w);
    {
      std::lock_guard<std::mutex> g(S.mu);
      auto it = S.plans.find(k);
      if (it != S.plans.end() && !it->second.rec) continue;   // (planned already; a recorded-only key gets its plan now)
      Plan q;
      q.busy = true;
      S.plans[k] = q;
    }
    chk("BEFORE", i);
    Plan p = build_plan(k, h, wsp, y, a, w, cs);   // (launches the algo: its module is loaded now; self-check syncs here)
    p.site = "warm";
    {
      std::lock_guard<std::mutex> g(S.mu);
      S.plans[k] = p;
      count_plan(S, p);
      if (p.ok) {
        auto& W = S.warmed[dev];
        bool have = false;
        for (auto& g2 : W) have = have || same_algo(g2, p.algo);
        if (!have) W.push_back(p.algo);
      } else {
        ++bad;
      }
    }
    chk("AFTER", i);
  }
  chk("EXIT", n);
  size_t nw;
  {
    std::lock_guard<std::mutex> g(S.mu);
    S.warm_done[dev] = true;
    nw = S.warmed[dev].size();
  }
  std::fprintf(stderr, "fpr_lt: warm dev %d: %d shapes, %zu distinct algos, %d not planned (-> ATen), %d stream workspaces\n",
               dev, n, nw, bad, ns);
  return bad ? -1 : (int)nw;
}


}  // namespace fpr_lt
#pragma GCC visibility pop

#endif  // __CUDACC__

// one exported C warm entry per .so (each .so has its own fpr_lt state): FPR_LT_DEFINE_WARM_ENTRY(fpr_mixed_cxx_lt_warm)
#define FPR_LT_DEFINE_WARM_ENTRY(name)                                                                              \
  extern "C" __attribute__((visibility("default"))) int name(int dev, const FprLtShape* sh, int n,                \
                                                             const cudaStream_t* streams, int ns) {             \
    return fpr_lt::warm(dev, sh, n, streams, ns);                                                                \
  }                                                                                                              \
  extern "C" __attribute__((visibility("default"))) int name##_dump(const char* path) { return fpr_lt::dump(path); } \
  __attribute__((visibility("hidden"))) void fpr_lt_mm_host(at::Tensor& y, const at::Tensor& a, const at::Tensor& w,   \
                                                           const char* site) {                                  \
    fpr_lt::mm(y, a, w, site);                                                                                   \
  }
