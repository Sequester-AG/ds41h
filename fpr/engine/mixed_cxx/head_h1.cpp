// head_h1.cpp -- the HSD head of a mixed step issued from C++: phase H1 of the "head" kind
// Module fpr_mixed_head, built by mixed_cxx_head.py (NOT a
// pieces_*.cpp: the driver build does not pick it up; H2 moves fpr_head_c into pieces_head.cpp and registers it).
//
// What Hooks.head_eager(stT, batches) + HSDT.run_all do per rank, as 3 issues on the rank's engine stream:
//   1. h_own[r][:n_s] = s["h"][sample_rows[:n_s]]       (Python: s["h"].index_select(0, ix) then HSDT._steps' copy:
//                                                        a gather, bit-identical by construction)
//   2. drafter b["aux"][:n_t] = s["aux"][:n_t]          (HSDT._steps' copy; skipped when it is the same buffer)
//   3. the rank's captured graph of HSDT.run(r, None, None) for this step's HSD key (= graph H's content; captured by
//      mixed_cxx_head.py into the engine's graph pool after the key's eager step)
// Rules: no Python, no host sync, no allocation (ATen index_select_out into the caller's buffer), errors as FPR codes.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <mutex>
#include <string>
#include <vector>

namespace {
constexpr int HW = 4;
// one handle per H1 key per rank: 16384 = 4096 keys x 4 ranks (the served full grid
// is 968 keys = 3,872 handles; a STALE recapture takes a new slot while the retired graph stays live, so slots only grow).
// A full table is a TORCH_CHECK error (Python exception at capture), never an out-of-bounds write.
constexpr int64_t kMaxGraphs = 16384;
enum { H_OK = 0, H_E_ARG = -1, H_E_CUDA = -5 };   // = fpr_cabi.h FPR_OK / FPR_E_ARG / FPR_E_CUDA

struct HeadGraph {             // a captured per-rank graph; the exec is owned by Python's torch.cuda.CUDAGraph object
  cudaGraphExec_t exec = nullptr;
  int dev = -1;
  bool live = false;
};
static_assert(sizeof(HeadGraph) * kMaxGraphs <= (1 << 19), "fpr_mixed_head: graph table above 512 KB");
HeadGraph g_graph[kMaxGraphs];  // fixed slots (no reallocation while a head is issued)
int64_t g_next = 0;
std::vector<int64_t> g_free;
std::mutex g_mu;

struct HeadStep {              // per rank, set by set_step (Python, per step) before head(): what fpr_head_c issues
  bool valid = false;
  int dev = -1;
  cudaStream_t stream = nullptr;
  void* h_src = nullptr;       // bucket T's s["h"] bf16 [h_rows, hidden]
  int64_t h_rows = 0, hidden = 0;
  const void* idx = nullptr;   // int64 [n_s] device: the tail batch's sample rows (staged by Python on `stream`)
  int64_t n_s = 0;
  void* h_dst = nullptr;       // HSD h_own[r] bf16 [>= n_s, hidden]
  const void* aux_src = nullptr;
  void* aux_dst = nullptr;     // the drafter's b["aux"] (shared by every D_pad bucket)
  int64_t n_t = 0, aux_row_bytes = 0;
};
HeadStep g_step[HW];
}  // namespace

// The piece (FprPieceFn signature of driver_api.h: H2 registers it with fpr_drv_register_piece("head", ...)).
extern "C" int fpr_head_c(int rank, int64_t handle, int /*layer*/) {
  if (rank < 0 || rank >= HW || handle < 0 || handle >= kMaxGraphs) return H_E_ARG;
  const HeadStep& s = g_step[rank];
  const HeadGraph& g = g_graph[handle];
  if (!s.valid || !g.live || g.dev != s.dev || s.stream == nullptr) return H_E_ARG;
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
      return H_E_CUDA;
    if (cudaGraphLaunch(g.exec, s.stream) != cudaSuccess) return H_E_CUDA;
  } catch (...) {
    return H_E_CUDA;
  }
  return cudaGetLastError() == cudaSuccess ? H_OK : H_E_CUDA;
}

namespace {
int64_t add_graph(int64_t exec, int dev) {
  std::lock_guard<std::mutex> lk(g_mu);
  int64_t h;
  if (!g_free.empty()) {
    h = g_free.back();
    g_free.pop_back();
  } else {
    TORCH_CHECK(g_next < kMaxGraphs, "fpr_mixed_head: graph table full (", g_next, " of ", kMaxGraphs,
                " handles, none free): more H1 keys x ranks than the table holds, or a STALE-recapture loop");
    h = g_next++;
  }
  g_graph[h].exec = (cudaGraphExec_t)exec;
  g_graph[h].dev = dev;
  g_graph[h].live = true;
  return h;
}

void free_graph(int64_t h) {
  std::lock_guard<std::mutex> lk(g_mu);
  TORCH_CHECK(h >= 0 && h < kMaxGraphs && g_graph[h].live, "fpr_mixed_head: bad graph handle");
  g_graph[h] = HeadGraph();
  g_free.push_back(h);
}

void set_step(int rank, int dev, int64_t stream, int64_t h_src, int64_t h_rows, int64_t hidden, int64_t idx, int64_t n_s,
              int64_t h_dst, int64_t aux_src, int64_t aux_dst, int64_t n_t, int64_t aux_row_bytes) {
  TORCH_CHECK(rank >= 0 && rank < HW, "fpr_mixed_head: rank");
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
  TORCH_CHECK(rank >= 0 && rank < HW, "fpr_mixed_head: rank");
  g_step[rank] = HeadStep();
}

int head(int rank, int64_t handle) { return fpr_head_c(rank, handle, -1); }
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_graph", &add_graph, "a captured graph's cudaGraphExec_t (CUDAGraph.raw_cuda_graph_exec()) -> handle");
  m.def("free_graph", &free_graph);
  m.def("set_step", &set_step, "this step's head inputs of a rank (pointers / counts)");
  m.def("clear_step", &clear_step);
  m.def("head", &head, "gather + aux copy + graph launch on the rank's stream -> FPR status (0 = OK)");
}
