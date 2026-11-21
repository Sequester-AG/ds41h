// head_h2.cpp -- the HSD head of a mixed step as a RECORDED OP PROGRAM issued from C++
// phase H2 of the "head" kind.
// Format: op_program.h. Module fpr_mixed_head2, built by mixed_cxx_head2.py (not a pieces_*.cpp yet: the driver build
// is unchanged). Same per-step contract as head_h1.cpp (set_step / head), with the captured graph replaced by a program:
//   1. h_own[r][:n_s] = s["h"][sample_rows[:n_s]]       (index_select_out, as H1)
//   2. drafter b["aux"][:n_t] = s["aux"][:n_t]          (cudaMemcpyAsync, as H1; skipped when the same buffer)
//   3. fpr_op::run(the rank's program of HSDT.run(r, None, None) for this step's HSD key)
//
// Recording (Python thread, mixed_cxx_head2.py, after the key's eager step; no driver thread running): per rank, under a
// throwaway CUDA-graph capture into the engine's graph pool (nothing runs), HSDT.run(r, None, None) with
//   - the rule-9 hooks of every loaded C ABI extension pointed at rec_hook -> CABI ops (stream from the Args' FprCall);
//   - add_cabi: Args bytes from an owner's packer
//
//   - add_boxed: dispatcher ops (TorchDispatchMode), stack built by the op's own schema (createStackForSchema);
//   - add_ev: torch.cuda.Event record / wait.
#include <torch/extension.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "op_program.h"

namespace py = pybind11;

// ---------------------------------------------------------------- op_program run
namespace fpr_op {
int run_range(const Program& p, int64_t b, int64_t e, int64_t* at) {
  int64_t i = b;
  if (b < 0 || e > (int64_t)p.ops.size() || b > e) return FPR_E_ARG;
  try {
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)p.device);
    std::optional<c10::InferenceMode> im;
    std::optional<at::NoGradGuard> ng;
    if (p.grad == INFERENCE)
      im.emplace();
    else if (p.grad == NO_GRAD)
      ng.emplace();
    for (; i < e; ++i) {
      const Op& o = p.ops[i];
      int rc = FPR_OK;
      const cudaStream_t st = p.streams[o.stream];
      switch (o.kind) {
        case CABI:
          rc = o.fn(o.args.data());
          break;
        case TRITON:
          rc = fpr_tr_launch(&o.rec, o.vals.data(), o.grid[0], o.grid[1], o.grid[2], reinterpret_cast<CUstream>(st), 0, 0);
          break;
        case BOXED: {
          const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(st, (c10::DeviceIndex)p.device));
          std::vector<c10::IValue> s = o.stack;
          o.op->callBoxed(&s);
          break;
        }
        case EV_RECORD:
          rc = cudaEventRecord(p.events[o.ev], st) == cudaSuccess ? FPR_OK : FPR_E_CUDA;
          break;
        case EV_WAIT:
          rc = cudaStreamWaitEvent(st, p.events[o.ev], 0) == cudaSuccess ? FPR_OK : FPR_E_CUDA;
          break;
        default:
          rc = FPR_E_ARG;
      }
      if (rc != FPR_OK) {
        if (at) *at = i;
        return rc;
      }
    }
  } catch (...) {
    if (at) *at = i;
    return FPR_E_CUDA;
  }
  return FPR_OK;
}
}  // namespace fpr_op

namespace {
using fpr_op::Op;
using fpr_op::Program;
constexpr int HW = 4;
constexpr int64_t kMax = 4096;
std::unique_ptr<Program> g_prog[kMax];   // fixed slots (no reallocation while a program runs)
std::vector<int64_t> g_free;
int64_t g_next = 0;
std::mutex g_mu;
int64_t g_rec = -1;                      // the program the rule-9 hooks append to (-1: recording off)
std::string g_rec_err;                   // first recording error (an entry missing from its table, ...)
std::atomic<int64_t> g_rec_seq{0};       // hook records so far (the recorder checks that every hooked call made one)
// H2g: a program stream-captured into a CUDA graph by C++ (at::cuda::CUDAGraph into the engine's
// graph pool, thread-local capture mode = Python's H1 capture). Captured graphs are NEVER destroyed (the engine's rule
// for graphs sharing a pool: retired, not freed); failed captures are parked here too.
std::unique_ptr<at::cuda::CUDAGraph> g_graph[kMax];
cudaGraphExec_t g_exec[kMax] = {};
std::vector<std::unique_ptr<at::cuda::CUDAGraph>> g_parked;

Program& prog(int64_t h) {
  TORCH_CHECK(h >= 0 && h < kMax && g_prog[h], "fpr_mixed_head2: bad program handle ", h);
  return *g_prog[h];
}

int stream_index(Program& p, cudaStream_t s) {
  for (size_t i = 0; i < p.streams.size(); ++i)
    if (p.streams[i] == s) return (int)i;
  p.streams.push_back(s);
  return (int)p.streams.size() - 1;
}

const FprExtTable* table_of(const std::string& so, const std::string& ext) {
  void* h = dlopen(so.c_str(), RTLD_NOW | RTLD_NOLOAD);
  TORCH_CHECK(h, "fpr_mixed_head2: ", so, " is not loaded");
  auto f = reinterpret_cast<FprTableFn>(dlsym(h, ("fpr_" + ext + "_table").c_str()));
  TORCH_CHECK(f, "fpr_mixed_head2: no fpr_", ext, "_table in ", so);
  const FprExtTable* t = f();
  TORCH_CHECK(t && t->abi_version == FPR_CABI_VERSION, "fpr_mixed_head2: ", ext, " ABI version");
  return t;
}

const FprFnEntry* entry_of(const FprExtTable* t, const char* name) {
  for (int i = 0; i < t->n_fns; ++i)
    if (!std::strcmp(t->fns[i].name, name)) return &t->fns[i];
  return nullptr;
}

void append_cabi(Program& p, FprFn fn, const char* name, const void* args, int64_t size) {
  Op o;
  o.kind = fpr_op::CABI;
  o.name = name;
  o.fn = fn;
  o.args.assign((size + 7) / 8, 0);
  std::memcpy(o.args.data(), args, size);
  const cudaStream_t s = *reinterpret_cast<const cudaStream_t*>(args);   // every Args starts with FprCall {stream, ...}
  o.stream = stream_index(p, s);
  p.ops.push_back(std::move(o));
}

// ---- rule 9: fpr_<ext>_set_record(hook, user)
// user = the extension's table here.
typedef void (*FprRecordHook)(void* user, const char* name, const void* args, int64_t args_size);
typedef void (*FprSetRecordFn)(FprRecordHook h, void* u);

void rec_hook(void* user, const char* name, const void* args, int64_t size) {
  if (g_rec < 0) return;
  const FprExtTable* t = static_cast<const FprExtTable*>(user);
  const FprFnEntry* e = entry_of(t, name);
  if (!e || e->args_size != size) {
    if (g_rec_err.empty()) g_rec_err = std::string("hook: ") + name + (e ? ": args size differs" : ": not in the table");
    return;
  }
  append_cabi(prog(g_rec), e->fn, name, args, size);
  ++g_rec_seq;
}
std::vector<FprSetRecordFn> g_hooked;

// ---------------------------------------------------------------- per-step staging (= head_h1.cpp's)
struct HeadStep {
  bool valid = false;
  int dev = -1;
  cudaStream_t stream = nullptr;
  void* h_src = nullptr;       // bucket T's s["h"] bf16 [h_rows, hidden]
  int64_t h_rows = 0, hidden = 0;
  const void* idx = nullptr;   // int64 [n_s] device: the tail batch's sample rows (staged by Python on `stream`)
  int64_t n_s = 0;
  void* h_dst = nullptr;       // HSD h_own[r] bf16 [>= n_s, hidden]
  const void* aux_src = nullptr;
  void* aux_dst = nullptr;     // the drafter's b["aux"]
  int64_t n_t = 0, aux_row_bytes = 0;
};
HeadStep g_step[HW];
}  // namespace

namespace {
// the step staging of `rank` for program `handle` (gather + aux copy on the rank's stream)
int stage(int rank, int64_t handle) {
  if (rank < 0 || rank >= HW || handle < 0 || handle >= kMax || !g_prog[handle]) return FPR_E_ARG;
  const HeadStep& s = g_step[rank];
  const Program& p = *g_prog[handle];
  if (!s.valid || p.device != s.dev || p.streams.empty() || p.streams[0] != s.stream) return FPR_E_ARG;
  try {
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)s.dev);
    if (s.n_s > 0) {
      const auto o = at::TensorOptions().device(at::kCUDA, (c10::DeviceIndex)s.dev);
      at::Tensor src = at::from_blob(s.h_src, {s.h_rows, s.hidden}, o.dtype(at::kBFloat16));
      at::Tensor idx = at::from_blob(const_cast<void*>(s.idx), {s.n_s}, o.dtype(at::kLong));
      at::Tensor dst = at::from_blob(s.h_dst, {s.n_s, s.hidden}, o.dtype(at::kBFloat16));
      const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(s.stream, (c10::DeviceIndex)s.dev));
      at::index_select_out(dst, src, 0, idx);
    }
    if (s.n_t > 0 && s.aux_src != s.aux_dst &&
        cudaMemcpyAsync(s.aux_dst, s.aux_src, (size_t)(s.n_t * s.aux_row_bytes), cudaMemcpyDeviceToDevice, s.stream) !=
            cudaSuccess)
      return FPR_E_CUDA;
  } catch (...) {
    return FPR_E_CUDA;
  }
  return FPR_OK;
}

// One thread issuing all W ranks (the Python thread): staging of every rank, then the programs interleaved at their
// stage boundaries (Program::segs), as HSDT.run_all issues them: every rank's collective is launched before any rank's
// next stage, so no GPU spins on a peer whose launches the host has not reached. Different boundary counts: rank by rank.
int launch_graph(int64_t h) {   // the program's captured graph on its main stream (the step's stream: stage checked it)
  return cudaGraphLaunch(g_exec[h], g_prog[h]->streams[0]) == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}

int run_all_ranks(const int64_t* handles) {
  for (int r = 0; r < HW; ++r) {
    const int rc = stage(r, handles[r]);
    if (rc != FPR_OK) return rc;
  }
  bool graphs = true;
  for (int r = 0; r < HW; ++r) graphs = graphs && g_exec[handles[r]] != nullptr;
  if (graphs) {   // one launch per rank (no interleaving needed: each rank's whole head is one queued op)
    for (int r = 0; r < HW; ++r) {
      try {
        const c10::cuda::CUDAGuard dg((c10::DeviceIndex)g_prog[handles[r]]->device);
        const int rc = launch_graph(handles[r]);
        if (rc != FPR_OK) return rc;
      } catch (...) {
        return FPR_E_CUDA;
      }
    }
    return FPR_OK;
  }
  const Program* P[HW];
  for (int r = 0; r < HW; ++r) P[r] = g_prog[handles[r]].get();
  bool same = true;
  for (int r = 1; r < HW; ++r) same = same && P[r]->segs.size() == P[0]->segs.size();
  if (!same) {
    for (int r = 0; r < HW; ++r) {
      const int rc = fpr_op::run(*P[r]);
      if (rc != FPR_OK) return rc;
    }
    return FPR_OK;
  }
  int64_t b[HW] = {0, 0, 0, 0};
  for (size_t k = 0; k <= P[0]->segs.size(); ++k) {
    for (int r = 0; r < HW; ++r) {
      const int64_t e = k < P[r]->segs.size() ? P[r]->segs[k] : (int64_t)P[r]->ops.size();
      const int rc = fpr_op::run_range(*P[r], b[r], e);
      if (rc != FPR_OK) return rc;
      b[r] = e;
    }
  }
  return FPR_OK;
}
}  // namespace

// The piece (FprPieceFn signature of driver_api.h: one call per rank, on the rank's own driver thread): the step
// staging, then the rank's program `handle`.
extern "C" int fpr_head2_c(int rank, int64_t handle, int /*layer*/) {
  const int rc = stage(rank, handle);
  if (rc != FPR_OK) return rc;
  if (g_exec[handle] != nullptr) {
    try {
      const c10::cuda::CUDAGuard dg((c10::DeviceIndex)g_prog[handle]->device);
      return launch_graph(handle);
    } catch (...) {
      return FPR_E_CUDA;
    }
  }
  return fpr_op::run(*g_prog[handle]);
}

namespace {
// head(rank, handle) from ONE thread for ranks 0..W-1 in turn (mixed_cxx_head.MixedHead.head_eager's loop): the first
// W-1 calls only note their handle; the last issues all ranks interleaved (run_all_ranks) and returns its status.
int64_t g_pend[HW] = {-1, -1, -1, -1};
int head(int rank, int64_t handle) {
  if (rank < 0 || rank >= HW) return FPR_E_ARG;
  g_pend[rank] = handle;
  if (rank != HW - 1) return FPR_OK;
  for (int r = 0; r < HW; ++r)
    if (g_pend[r] < 0) return FPR_E_ARG;
  const int rc = run_all_ranks(g_pend);
  for (int r = 0; r < HW; ++r) g_pend[r] = -1;
  return rc;
}

// H2g: capture program h into a CUDA graph (at::cuda::CUDAGraph, pool = the engine's graph pool of this rank, thread-local
// capture on the program's main stream: nothing runs). Throws (and parks the graph) when the capture fails; afterwards
// fpr_head2_c / head() launch the graph instead of issuing the ops.
double capture(int64_t h, uint64_t pool_a, uint64_t pool_b) {
  Program& p = prog(h);
  TORCH_CHECK(g_exec[h] == nullptr, "fpr_mixed_head2: program ", h, " is already captured");
  const c10::cuda::CUDAGuard dg((c10::DeviceIndex)p.device);
  const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(p.streams[0], (c10::DeviceIndex)p.device));
  auto g = std::make_unique<at::cuda::CUDAGraph>();
  const auto t0 = std::chrono::steady_clock::now();
  int rc = FPR_OK;
  int64_t at = -1;
  try {
    g->capture_begin({pool_a, pool_b}, cudaStreamCaptureModeThreadLocal);
    rc = fpr_op::run(p, &at);
    g->capture_end();
  } catch (const std::exception& e) {
    g_parked.push_back(std::move(g));
    TORCH_CHECK(false, "fpr_mixed_head2: capture of program ", h, " (", p.label, ") failed: ", e.what());
  }
  if (rc != FPR_OK) {
    g_parked.push_back(std::move(g));
    TORCH_CHECK(false, "fpr_mixed_head2: capture of program ", h, ": op ", at, " (", at >= 0 ? p.ops[at].name : "", ") rc ", rc);
  }
  g_exec[h] = g->raw_cuda_graph_exec();
  TORCH_CHECK(g_exec[h] != nullptr, "fpr_mixed_head2: capture of program ", h, ": no graph exec");
  g_graph[h] = std::move(g);
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool has_graph(int64_t h) { return h >= 0 && h < kMax && g_exec[h] != nullptr; }

void mark_seg(int64_t h) {
  Program& p = prog(h);
  p.segs.push_back((int64_t)p.ops.size());
}

// ---------------------------------------------------------------- pybind API
int64_t prog_new(int device, int64_t main_stream, int grad, const std::string& label) {
  std::lock_guard<std::mutex> lk(g_mu);
  int64_t h;
  if (!g_free.empty()) {
    h = g_free.back();
    g_free.pop_back();
  } else {
    TORCH_CHECK(g_next < kMax, "fpr_mixed_head2: program table full");
    h = g_next++;
  }
  g_prog[h].reset(new Program());
  g_prog[h]->device = device;
  g_prog[h]->grad = grad;
  g_prog[h]->streams.push_back((cudaStream_t)main_stream);
  g_prog[h]->label = label;
  return h;
}

void prog_free(int64_t h) {
  std::lock_guard<std::mutex> lk(g_mu);
  Program& p = prog(h);
  TORCH_CHECK(g_exec[h] == nullptr, "fpr_mixed_head2: program ", h, " is captured: captured graphs are never freed");
  for (auto e : p.events) cudaEventDestroy(e);
  g_prog[h].reset();
  g_free.push_back(h);
}

// exts: (so path, ext) per extension with a rule-9 hook; every one pointed at rec_hook (recording into `h`)
void rec_begin(int64_t h, const std::vector<std::pair<std::string, std::string>>& exts) {
  TORCH_CHECK(g_rec < 0, "fpr_mixed_head2: a recording is open");
  prog(h);
  g_rec_err.clear();
  g_hooked.clear();
  for (const auto& e : exts) {
    void* lib = dlopen(e.first.c_str(), RTLD_NOW | RTLD_NOLOAD);
    TORCH_CHECK(lib, "fpr_mixed_head2: ", e.first, " is not loaded");
    auto set = reinterpret_cast<FprSetRecordFn>(dlsym(lib, ("fpr_" + e.second + "_set_record").c_str()));
    TORCH_CHECK(set, "fpr_mixed_head2: ", e.second, " has no rule-9 hook (fpr_", e.second, "_set_record)");
    set(&rec_hook, (void*)table_of(e.first, e.second));
    g_hooked.push_back(set);
  }
  g_rec = h;
}

std::string rec_end() {
  for (auto set : g_hooked) set(nullptr, nullptr);
  g_hooked.clear();
  g_rec = -1;
  return g_rec_err;
}

int64_t rec_seq() { return g_rec_seq.load(); }

// (so, ext) -> [(name, args_size)] of the extension's table (the recorder's expected hook names)
std::vector<std::pair<std::string, int64_t>> table_names(const std::string& so, const std::string& ext) {
  const FprExtTable* t = table_of(so, ext);
  std::vector<std::pair<std::string, int64_t>> out;
  for (int i = 0; i < t->n_fns; ++i) out.emplace_back(t->fns[i].name, t->fns[i].args_size);
  return out;
}

int64_t add_cabi(int64_t h, const std::string& so, const std::string& ext, const std::string& name, py::bytes args) {
  const std::string a = args;
  const FprFnEntry* e = entry_of(table_of(so, ext), name.c_str());
  TORCH_CHECK(e, "fpr_mixed_head2: ", name, " not in ", ext, "'s table");
  TORCH_CHECK(e->args_size == (int64_t)a.size(), "fpr_mixed_head2: ", name, " args ", a.size(), " B, table ", e->args_size);
  Program& p = prog(h);
  append_cabi(p, e->fn, name.c_str(), a.data(), (int64_t)a.size());
  return (int64_t)p.ops.size() - 1;
}

// one immediate call of a table entry (one-time inits, e.g. moe_v4.init on the current device)
int call_cabi(const std::string& so, const std::string& ext, const std::string& name, py::bytes args) {
  const std::string a = args;
  const FprFnEntry* e = entry_of(table_of(so, ext), name.c_str());
  TORCH_CHECK(e, "fpr_mixed_head2: ", name, " not in ", ext, "'s table");
  TORCH_CHECK(e->args_size == (int64_t)a.size(), "fpr_mixed_head2: ", name, " args size");
  std::vector<uint64_t> buf((a.size() + 7) / 8, 0);
  std::memcpy(buf.data(), a.data(), a.size());
  return e->fn(buf.data());
}

int64_t add_triton(int64_t h, int64_t stream, py::bytes record, std::vector<uint64_t> vals, int64_t gx, int64_t gy, int64_t gz) {
  const std::string r = record;
  TORCH_CHECK(r.size() == sizeof(FprTrRecord), "fpr_mixed_head2: record size ", r.size(), " != ", sizeof(FprTrRecord));
  Program& p = prog(h);
  Op o;
  o.kind = fpr_op::TRITON;
  std::memcpy(&o.rec, r.data(), sizeof(FprTrRecord));
  TORCH_CHECK(o.rec.gscratch_size == 0 && o.rec.pscratch_size == 0, "fpr_mixed_head2: Triton scratch not supported (",
              o.rec.name, ")");
  TORCH_CHECK(fpr_tr_check(&o.rec, vals.data()) == FPR_OK, "fpr_mixed_head2: Triton values outside the record's class (",
              o.rec.name, ")");
  o.name = o.rec.key;
  o.vals = std::move(vals);
  o.grid[0] = (uint32_t)gx, o.grid[1] = (uint32_t)gy, o.grid[2] = (uint32_t)gz;
  o.stream = stream_index(p, (cudaStream_t)stream);
  p.ops.push_back(std::move(o));
  return (int64_t)p.ops.size() - 1;
}

// a dispatcher op ("aten::mm", "out") with Python args / kwargs, bound by the op's schema (as torch.ops binds them)
int64_t add_boxed(int64_t h, int64_t stream, const std::string& qualname, const std::string& overload, py::tuple args,
                  py::dict kwargs) {
  const c10::OperatorHandle op = c10::Dispatcher::singleton().findSchemaOrThrow(qualname.c_str(), overload.c_str());
  torch::jit::Stack st = torch::jit::createStackForSchema(op.schema(), torch::jit::tuple_slice(std::move(args)),
                                                          py::reinterpret_borrow<py::kwargs>(kwargs), std::nullopt);
  Program& p = prog(h);
  Op o;
  o.kind = fpr_op::BOXED;
  o.name = qualname + (overload.empty() ? "" : "." + overload);
  o.op = op;
  o.stack = std::move(st);
  o.stream = stream_index(p, (cudaStream_t)stream);
  p.ops.push_back(std::move(o));
  return (int64_t)p.ops.size() - 1;
}

int64_t new_event(int64_t h) {
  Program& p = prog(h);
  cudaEvent_t e;
  TORCH_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess, "fpr_mixed_head2: cudaEventCreate");
  p.events.push_back(e);
  return (int64_t)p.events.size() - 1;
}

int64_t add_ev(int64_t h, int64_t stream, int64_t ev, bool wait) {
  Program& p = prog(h);
  TORCH_CHECK(ev >= 0 && ev < (int64_t)p.events.size(), "fpr_mixed_head2: event index");
  Op o;
  o.kind = wait ? fpr_op::EV_WAIT : fpr_op::EV_RECORD;
  o.name = wait ? "ev.wait" : "ev.record";
  o.ev = (int32_t)ev;
  o.stream = stream_index(p, (cudaStream_t)stream);
  p.ops.push_back(std::move(o));
  return (int64_t)p.ops.size() - 1;
}

void keep(int64_t h, at::Tensor t) { prog(h).keep.push_back(t); }

py::list ops_of(int64_t h) {
  py::list l;
  const Program& p = prog(h);
  for (const auto& o : p.ops) l.append(py::make_tuple(o.kind, o.stream, o.name));
  return l;
}

int64_t n_streams(int64_t h) { return (int64_t)prog(h).streams.size(); }

// the raw words of op i: a CABI op's Args (8-byte words), a Triton op's values (the recorder's pointer checks)
std::vector<uint64_t> op_words(int64_t h, int64_t i) {
  const Program& p = prog(h);
  TORCH_CHECK(i >= 0 && i < (int64_t)p.ops.size(), "fpr_mixed_head2: op index");
  const Op& o = p.ops[i];
  return o.kind == fpr_op::CABI ? o.args : o.kind == fpr_op::TRITON ? o.vals : std::vector<uint64_t>();
}

void set_step(int rank, int dev, int64_t stream, int64_t h_src, int64_t h_rows, int64_t hidden, int64_t idx, int64_t n_s,
              int64_t h_dst, int64_t aux_src, int64_t aux_dst, int64_t n_t, int64_t aux_row_bytes) {
  TORCH_CHECK(rank >= 0 && rank < HW, "fpr_mixed_head2: rank");
  HeadStep& s = g_step[rank];
  s.dev = dev;
  s.stream = (cudaStream_t)stream;
  s.h_src = (void*)h_src;
  s.h_rows = h_rows;
  s.hidden = hidden;
  s.idx = (const void*)idx;
  s.n_s = n_s;
  s.h_dst = (void*)h_dst;
  s.aux_src = (const void*)aux_src;
  s.aux_dst = (void*)aux_dst;
  s.n_t = n_t;
  s.aux_row_bytes = aux_row_bytes;
  s.valid = true;
}

void clear_step(int rank) {
  TORCH_CHECK(rank >= 0 && rank < HW, "fpr_mixed_head2: rank");
  g_step[rank] = HeadStep();
}

py::tuple run_py(int64_t h) {
  const Program& p = prog(h);
  int64_t at = -1;
  const int rc = fpr_op::run(p, &at);
  return py::make_tuple(rc, at, at >= 0 && at < (int64_t)p.ops.size() ? p.ops[at].name : std::string());
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prog_new", &prog_new, "(device, main stream, grad mode 0 inference / 1 no_grad / 2 grad, label) -> handle");
  m.def("prog_free", &prog_free);
  m.def("rec_begin", &rec_begin);
  m.def("rec_end", &rec_end, "-> the first hook error ('' = none)");
  m.def("rec_seq", &rec_seq);
  m.def("table_names", &table_names);
  m.def("add_cabi", &add_cabi);
  m.def("call_cabi", &call_cabi);
  m.def("add_triton", &add_triton);
  m.def("add_boxed", &add_boxed);
  m.def("new_event", &new_event);
  m.def("add_ev", &add_ev);
  m.def("keep", &keep);
  m.def("ops", &ops_of);
  m.def("n_streams", &n_streams);
  m.def("op_words", &op_words);
  m.def("set_step", &set_step, "this step's head inputs of a rank (pointers / counts; = head_h1's)");
  m.def("clear_step", &clear_step);
  m.def("head", &head, "ranks 0..3 in turn from one thread: the last call stages every rank and issues the 4 programs "
                       "interleaved at their stage boundaries -> FPR status (0 = OK)");
  m.def("capture", &capture, "H2g: program h -> a CUDA graph in pool (a, b) (the engine's gpool[r]); returns the capture ms");
  m.def("has_graph", &has_graph);
  m.def("mark_seg", &mark_seg, "a stage boundary at the program's current op count (the recorder, at each _steps yield)");
  m.def("segs", [](int64_t h) { return prog(h).segs; });
  m.def("run", &run_py, "issue program h alone -> (status, failing op index, its name)");
  m.def("record_size", []() { return (int64_t)sizeof(FprTrRecord); });
}
