// pieces_c1mega.cpp -- Registered as owner kind "c1mega" (stage 4 spans) plus pybind
// entries for the Python-driven checks (add_c1mega(bytes) -> handle, c1mega(rank, handle, layer) -> status). Every launch is
// the Python hook's: the c1mega C ABI for k_layer, ATen for the aux mean / copy (same ops, same stream).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstring>
#include <memory>
#include <vector>
#include "pieces_c1mega.h"
#include "driver_api.h"

namespace {
FprFn F_run = nullptr;

inline int ck(cudaError_t e) { return e == cudaSuccess ? FPR_OK : FPR_E_CUDA; }

struct Slots {                        // as the driver's: adds / frees only from Python between spans
  std::vector<std::unique_ptr<FprPieceC1mega>> v;
  std::vector<int64_t> fl;
  int64_t add(std::unique_ptr<FprPieceC1mega> p) {
    if (!fl.empty()) {
      const int64_t h = fl.back();
      fl.pop_back();
      v[h] = std::move(p);
      return h;
    }
    v.push_back(std::move(p));
    return (int64_t)v.size() - 1;
  }
  const FprPieceC1mega* get(int64_t h) const { return (h >= 0 && h < (int64_t)v.size()) ? v[h].get() : nullptr; }
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
} g_c1m;

void resolve_fns() {
  if (!F_run) F_run = fpr_drv_fn("c1mega", "c1mega.run", sizeof(FprC1megaRunArgs));
}

}  // namespace

// C1Mega.run(r, s, L, st) for the run starting at `layer` (handle: that run on `rank`); no-op status for a NULL handle
int c1mega_c(int rank, int64_t handle, int layer) {
  const FprPieceC1mega* p = g_c1m.get(handle);
  if (p == nullptr || F_run == nullptr || p->rank != rank || p->layer != layer) return FPR_E_ARG;
  int rc;
  if (p->join && (rc = fpr_drv_join(rank, p->main))) return rc;                         // self.eng._join_mix(s, pf=False)
  for (int i = 0; i < p->n_ev; ++i)                                                     // wait_event(engram ev ...)
    if ((rc = ck(cudaStreamWaitEvent(p->main, p->ev[i], 0)))) return rc;
  if ((rc = F_run(&p->run))) return rc;                                                 // self.X.run(pack, len(run), True)
  if (p->n_aux) {                                                                       // owner: the DSpark aux columns
    const int dev = p->run.c.device;
    const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)dev));
    const auto o = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, dev);
    at::Tensor aux = at::from_blob(p->aux, {p->R, p->A, p->D}, o);
    for (int i = 0; i < p->n_aux; ++i)
      aux.select(1, p->col[i]).copy_(at::from_blob(p->xo[i], {p->R, p->HC, p->D}, o).mean(1));
  }
  return FPR_OK;
}

// ---- pybind (Python side: the package's c1mega_eng.py C1Mega.cxx_pieces)
namespace {
int64_t add_c1mega(py::bytes b) {
  std::string s = b;
  TORCH_CHECK(s.size() == sizeof(FprPieceC1mega), "pieces_c1mega: FprPieceC1mega is ", s.size(), " bytes, the C++ ",
              sizeof(FprPieceC1mega));
  auto p = std::make_unique<FprPieceC1mega>();
  std::memcpy(p.get(), s.data(), s.size());
  TORCH_CHECK(p->version == FPR_PIECES_VERSION, "pieces_c1mega: version");
  TORCH_CHECK(p->n_ev >= 0 && p->n_ev <= FPR_C1M_MAX_EV && p->n_aux >= 0 && p->n_aux <= FPR_C1M_MAX_AUX, "pieces_c1mega: counts");
  TORCH_CHECK(p->run.c.stream == p->main, "pieces_c1mega: run.c.stream != main");
  resolve_fns();
  TORCH_CHECK(F_run != nullptr, "pieces_c1mega: the c1mega table is not registered (register_table(X.__file__, 'c1mega'))");
  return g_c1m.add(std::move(p));
}
}  // namespace

void fpr_pieces_c1mega_register(pybind11::module_& m) {
  m.def("sizeof_c1mega", [] { return (int64_t)sizeof(FprPieceC1mega); });
  m.def("sizeof_c1mega_run", [] { return (int64_t)sizeof(FprC1megaRunArgs); });
  m.def("add_c1mega", &add_c1mega);
  m.def("c1mega", [](int r, int64_t h, int L) { return c1mega_c(r, h, L); });
  m.def("free_c1mega", [](int64_t h) { g_c1m.free(h); });
  m.def("live_c1mega", [] { return g_c1m.live(); });
  fpr_drv_register_piece("c1mega", &c1mega_c);
}
