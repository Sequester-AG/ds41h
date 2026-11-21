// pieces_moe.cpp -- the moe piece of the C++ mixed-step driver. Compiled into FPR's
// mixed_cxx extension (mixed_cxx.py picks up every pieces_*.cpp); driver.cpp calls fpr_pieces_moe_register (weak symbol).
// Python side: mixed_cxx_moe.py (setup / step_params / piece_moe / the static-struct builder).
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "pieces_moe.h"   // (first: pieces.h's FprStepParams, which driver_api.h uses)
#include "driver_api.h"
#include "mixed_plan.h"
#include "fpr_lt.h"   // the pool-thread shared-expert GEMMs through fpr_lt's closed algo set; this .so =
                      // fpr_mixed_cxx, warmed by driver.cpp's FPR_LT_DEFINE_WARM_ENTRY(fpr_mixed_cxx_lt_warm))

namespace {
constexpr int MW = 4;
struct MoeRank {
  cudaEvent_t ev0 = nullptr, ev1 = nullptr;   // fork: shared expert main -> side / side -> main
  int dev = -1;
  cudaStream_t main = nullptr, side = nullptr;  // the rank's streams (thread warm-up)
  void* scratch = nullptr;                      // >= 48 KiB device scratch (Python-owned) for the warm-up mm
  int64_t scratch_bytes = 0;
};
MoeRank g_mr[MW];
// handle table: fixed slots (no reallocation under the driver threads), a free list; add / free from Python (no span
// issuing then: the span builder adds before run(), a bucket state's handles are freed when the state is dropped)
constexpr int64_t kMaxMoe = 1 << 16;
FprPieceMoe* g_slot[kMaxMoe] = {};
std::vector<int64_t> g_free;
int64_t g_next = 0, g_live = 0;
std::mutex g_mu;
FprFn F_layer = nullptr, F_layer_rs = nullptr;                         // agrs_pmws
FprFn F_v4_route2r = nullptr, F_v4_g1 = nullptr, F_v4_g2 = nullptr;     // moe_v4r / moe_v4
FprFn F_r2r = nullptr, F_r2 = nullptr;                                  // route2
FprFn F_swiglu = nullptr, F_prepr5 = nullptr, F_raw_g1 = nullptr, F_raw_g2 = nullptr, F_combine = nullptr;   // moe_raw
inline int ck(cudaError_t e) { return e == cudaSuccess ? FPR_OK : FPR_E_CUDA; }

// ATen views on the piece's buffers (no allocation)
at::Tensor bview(void* ptr, at::IntArrayRef sizes, at::ScalarType dt, int dev) {
  return at::from_blob(ptr, sizes, at::TensorOptions().dtype(dt).device(at::kCUDA, (c10::DeviceIndex)dev));
}

// Engine.p_moe's fork path (see pieces_moe.h); every launch on the stream the Python statement runs on
int fork_moe(const FprPieceMoe* p) {
  if (!p->fork_ok) return FPR_E_SHAPE;
  if (!(F_v4_route2r && F_v4_g1 && F_v4_g2 && F_r2r && F_r2 && F_swiglu && F_prepr5 && F_raw_g1 && F_raw_g2 && F_combine))
    return FPR_E_INIT;
  const int dev = p->device;
  const int64_t Tall = p->Tall, H = p->H, Is = p->Is;
  int rc;
  if ((rc = ck(cudaEventRecord(p->ev0, p->main)))) return rc;                   // ev0.record(cur)
  if ((rc = ck(cudaStreamWaitEvent(p->side, p->ev0, 0)))) return rc;            // side.wait_event(ev0)
  try {
    const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(p->side, (c10::DeviceIndex)dev));
    auto fall = bview(p->fall, {Tall, H}, at::kBFloat16, dev), sw13 = bview(p->sw13, {2 * Is, H}, at::kBFloat16, dev);
    auto gu = bview(p->gu, {Tall, 2 * Is}, at::kBFloat16, dev);
    fpr_lt::mm(gu, fall, sw13);                                                 // torch.mm(fall, sw13.t(), out=gu)
    if ((rc = F_swiglu(&p->swiglu))) return rc;                                 // e.swiglu(gu, hsh)
    auto hsh = bview(p->hsh, {Tall, Is}, at::kBFloat16, dev), sw2 = bview(p->sw2, {H, Is}, at::kBFloat16, dev);
    auto sh = bview(p->sh, {Tall, H}, at::kBFloat16, dev);
    fpr_lt::mm(sh, hsh, sw2);                                                   // torch.mm(hsh, sw2.t(), out=sh)  (fpr_lt)
  } catch (const std::exception&) {
    return FPR_E_CUDA;
  }
  if ((rc = ck(cudaEventRecord(p->ev1, p->side)))) return rc;                   // ev1.record(side)
  if (p->fork_route == 0) {
    if ((rc = F_v4_route2r(&p->v4_route))) return rc;
  } else if (p->fork_route == 1) {
    if ((rc = F_r2r(&p->r2r))) return rc;
  } else {
    try {                                                                       // ids / wts = gr[:, :24] / [:, 24:48].contiguous()
      const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)dev));
      auto gr = bview(p->gr, {Tall, 64}, at::kByte, dev);
      bview(p->ids_buf, {Tall, 24}, at::kByte, dev).copy_(gr.narrow(1, 0, 24));
      bview(p->wts_buf, {Tall, 24}, at::kByte, dev).copy_(gr.narrow(1, 24, 24));
    } catch (const std::exception&) {
      return FPR_E_CUDA;
    }
    if ((rc = F_r2(&p->r2))) return rc;
  }
  if ((rc = F_prepr5(&p->prepr5))) return rc;
  if (p->fork_gemm == 0) {
    if ((rc = F_v4_g1(&p->v4_g1))) return rc;
    if ((rc = F_v4_g2(&p->v4_g2))) return rc;
  } else {
    if ((rc = F_raw_g1(&p->raw_g1))) return rc;
    if ((rc = F_raw_g2(&p->raw_g2))) return rc;
  }
  if ((rc = ck(cudaStreamWaitEvent(p->main, p->ev1, 0)))) return rc;            // cur.wait_event(ev1)
  return F_combine(&p->combine);                                                // e.combine(Y, slotmap, sh, part)
}
}  // namespace

extern "C" int piece_moe(const FprPieceMoe* p, const FprStepParams* sp, int layer) {
  if (p == nullptr || sp == nullptr || p->version != FPR_PIECE_MOE_VERSION || p->layer != layer) return FPR_E_ARG;
  int rc;
  // AGCE: the done events of every sender that copied into this rank since the last join, then any StepParams waits
  if ((rc = fpr_drv_agce_join(p->rank, p->main))) return rc;
  for (int i = 0; i < sp->moe_n_wait && i < 4; ++i)
    if (sp->moe_wait[i] != nullptr && (rc = ck(cudaStreamWaitEvent(p->main, sp->moe_wait[i], 0)))) return rc;
  switch (sp->moe_variant) {
    case FPR_MOE_PROD_RSF: {
      if (!p->rsf_ok || F_layer_rs == nullptr) return FPR_E_SHAPE;
      AgrsPmwsLayerArgs a = p->prod_rs;
      a.vcount = sp->moe_vcount;
      return F_layer_rs(&a);
    }
    case FPR_MOE_PROD_FUSED: {
      if (!p->fused_ok || F_layer == nullptr) return FPR_E_SHAPE;
      AgrsPmwsLayerArgs a = p->prod;
      a.vcount = sp->moe_vcount;
      return F_layer(&a);
    }
    case FPR_MOE_FORK_V4:
    case FPR_MOE_FORK_RAW:
      return fork_moe(p);
    default:
      return FPR_E_SHAPE;
  }
}

namespace {
// per rank: the fork's two events (created on the rank's device; call with it current) and the agrs_pmws functions (when
// its table is registered: mixed_cxx_moe.setup, ENG_AGRS_WS=2). Returns whether the prod variants can run.
bool moe_init_rank(int r, int device) {
  TORCH_CHECK(r >= 0 && r < MW, "pieces_moe: rank");
  int cur = -1;
  C10_CUDA_CHECK(cudaGetDevice(&cur));
  TORCH_CHECK(cur == device, "pieces_moe: moe_init_rank(", r, ", ", device, ") with device ", cur, " current");
  MoeRank& k = g_mr[r];
  k.dev = device;
  if (!k.ev0) C10_CUDA_CHECK(cudaEventCreateWithFlags(&k.ev0, cudaEventDisableTiming));
  if (!k.ev1) C10_CUDA_CHECK(cudaEventCreateWithFlags(&k.ev1, cudaEventDisableTiming));
  F_layer = fpr_drv_fn("agrs_pmws", "agrs_pmws.layer", (int64_t)sizeof(AgrsPmwsLayerArgs));
  F_layer_rs = fpr_drv_fn("agrs_pmws", "agrs_pmws.layer_rs", (int64_t)sizeof(AgrsPmwsLayerArgs));
  return F_layer != nullptr && F_layer_rs != nullptr;
}

// per rank, the fork's tables (registered first: mixed_cxx_moe.setup): resolve the functions and run each table's init on
// this rank's device (current). Returns false when a table is missing (the fork variants then stay Python).
bool moe_init_fork_rank(int r, int device) {
  TORCH_CHECK(r >= 0 && r < MW, "pieces_moe: rank");
  int cur = -1;
  C10_CUDA_CHECK(cudaGetDevice(&cur));
  TORCH_CHECK(cur == device, "pieces_moe: moe_init_fork_rank with device ", cur, " current");
  F_v4_route2r = fpr_drv_fn("moe_v4r", "moe_v4r.route2r", (int64_t)sizeof(MoeV4Route2rArgs));
  F_v4_g1 = fpr_drv_fn("moe_v4", "moe_v4.gemm1r", (int64_t)sizeof(MoeV4Gemm1rArgs));
  F_v4_g2 = fpr_drv_fn("moe_v4", "moe_v4.gemm2r", (int64_t)sizeof(MoeV4Gemm2rArgs));
  F_r2r = fpr_drv_fn("route2", "route2.route2r", (int64_t)sizeof(FprRoute2rArgs));
  F_r2 = fpr_drv_fn("route2", "route2.route2", (int64_t)sizeof(FprRoute2Args));
  F_swiglu = fpr_drv_fn("moe_raw", "moe_raw.swiglu", (int64_t)sizeof(FprMoeRawSwigluArgs));
  F_prepr5 = fpr_drv_fn("moe_raw", "moe_raw.prepr5", (int64_t)sizeof(FprMoeRawPrepr5Args));
  F_raw_g1 = fpr_drv_fn("moe_raw", "moe_raw.gemm1r", (int64_t)sizeof(FprMoeRawGemm1rArgs));
  F_raw_g2 = fpr_drv_fn("moe_raw", "moe_raw.gemm2r", (int64_t)sizeof(FprMoeRawGemm2rArgs));
  F_combine = fpr_drv_fn("moe_raw", "moe_raw.combine", (int64_t)sizeof(FprMoeRawCombineArgs));
  if (!(F_v4_route2r && F_v4_g1 && F_v4_g2 && F_r2r && F_r2 && F_swiglu && F_prepr5 && F_raw_g1 && F_raw_g2 && F_combine)) return false;
  MoeV4InitArgs v{device, 0};
  TORCH_CHECK(fpr_drv_fn("moe_v4", "moe_v4.init", (int64_t)sizeof(MoeV4InitArgs))(&v) == FPR_OK, "pieces_moe: moe_v4.init");
  TORCH_CHECK(fpr_drv_fn("moe_v4r", "moe_v4r.init", (int64_t)sizeof(MoeV4InitArgs))(&v) == FPR_OK, "pieces_moe: moe_v4r.init");
  FprMoeRawInitArgs mi{};
  mi.c.stream = nullptr, mi.c.device = device, mi.c.rank = r;
  TORCH_CHECK(fpr_drv_fn("moe_raw", "moe_raw.init", (int64_t)sizeof(FprMoeRawInitArgs))(&mi) == FPR_OK, "pieces_moe: moe_raw.init");
  FprRoute2InitArgs ri{};
  ri.c.stream = nullptr, ri.c.device = device, ri.c.rank = r;
  TORCH_CHECK(fpr_drv_fn("route2", "route2.init", (int64_t)sizeof(FprRoute2InitArgs))(&ri) == FPR_OK, "pieces_moe: route2.init");
  return true;
}

// per rank: the streams + a small scratch buffer for fpr_pieces_moe_init_thread (Python owns the buffer)
void moe_set_thread_init(int r, int64_t main, int64_t side, int64_t scratch, int64_t bytes) {
  TORCH_CHECK(r >= 0 && r < MW && bytes >= 3 * 64 * 128 * 2, "pieces_moe: moe_set_thread_init");
  MoeRank& k = g_mr[r];
  k.main = (cudaStream_t)main, k.side = (cudaStream_t)side, k.scratch = (void*)scratch, k.scratch_bytes = bytes;
}

// sizes of the C structs the ctypes mirrors build (mixed_cxx_moe.check)
std::vector<int64_t> sizeof_moe_parts() {
  return {(int64_t)sizeof(FprPieceMoe), (int64_t)sizeof(AgrsPmwsLayerArgs), (int64_t)sizeof(MoeV4Route2rArgs),
          (int64_t)sizeof(MoeV4Gemm1rArgs), (int64_t)sizeof(MoeV4Gemm2rArgs), (int64_t)sizeof(FprRoute2rArgs),
          (int64_t)sizeof(FprRoute2Args), (int64_t)sizeof(FprMoeRawSwigluArgs), (int64_t)sizeof(FprMoeRawPrepr5Args),
          (int64_t)sizeof(FprMoeRawGemm1rArgs), (int64_t)sizeof(FprMoeRawGemm2rArgs), (int64_t)sizeof(FprMoeRawCombineArgs)};
}

int64_t sizeof_moe() { return (int64_t)sizeof(FprPieceMoe); }
int64_t sizeof_pmws_args() { return (int64_t)sizeof(AgrsPmwsLayerArgs); }

int64_t add_moe(py::bytes b) {
  std::string s = b;
  TORCH_CHECK((int64_t)s.size() == (int64_t)sizeof(FprPieceMoe), "pieces_moe: FprPieceMoe is ", s.size(), " bytes, the driver's ",
              sizeof(FprPieceMoe), " (ctypes / header drift)");
  auto p = std::make_unique<FprPieceMoe>();
  std::memcpy(p.get(), s.data(), s.size());
  TORCH_CHECK(p->version == FPR_PIECE_MOE_VERSION && p->rank >= 0 && p->rank < MW, "pieces_moe: version / rank");
  p->ev0 = g_mr[p->rank].ev0;
  p->ev1 = g_mr[p->rank].ev1;
  std::lock_guard<std::mutex> lk(g_mu);
  int64_t h;
  if (!g_free.empty()) {
    h = g_free.back();
    g_free.pop_back();
  } else {
    TORCH_CHECK(g_next < kMaxMoe, "pieces_moe: more than ", kMaxMoe, " live moe handles");
    h = g_next++;
  }
  g_slot[h] = p.release();
  ++g_live;
  return h;
}

void free_moe(int64_t h) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (h < 0 || h >= kMaxMoe || g_slot[h] == nullptr) return;
  delete g_slot[h];
  g_slot[h] = nullptr;
  g_free.push_back(h);
  --g_live;
}

// Engine._set_vidx stores a NEW device tensor in s["nv"] (valid rows per rank) at every mixed step, so the prod
// Args' nv pointer frozen at add_moe went stale from the bucket's 2nd mixed step on. mixed_cxx_moe.step_params patches it here between spans (no driver thread reads g_slot).
void moe_set_nv(int64_t h, int64_t nv) {
  std::lock_guard<std::mutex> lk(g_mu);
  TORCH_CHECK(h >= 0 && h < kMaxMoe && g_slot[h] != nullptr, "pieces_moe: moe_set_nv: unknown handle ", h);
  g_slot[h]->prod.nv = reinterpret_cast<const int32_t*>(nv);
  g_slot[h]->prod_rs.nv = reinterpret_cast<const int32_t*>(nv);
}

int64_t live_moe() {
  std::lock_guard<std::mutex> lk(g_mu);
  return g_live;
}

}  // namespace

// stage 1d, once per driver thread after it bound its device: one ATen mm on the rank's main and
// side streams, so torch's per-(thread, stream) cuBLAS handle / workspace exist before any span (the prod layer's shared
// slice runs on main inside agrs_pmws, the fork's on side)
int fpr_pieces_moe_init_thread(int r) {
  if (r < 0 || r >= MW) return FPR_E_ARG;
  MoeRank& k = g_mr[r];
  if (k.scratch == nullptr || k.main == nullptr || k.side == nullptr || k.dev < 0) return FPR_E_INIT;
  try {
    for (cudaStream_t st : {k.main, k.side}) {
      const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(st, (c10::DeviceIndex)k.dev));
      uint8_t* b = static_cast<uint8_t*>(k.scratch);
      auto A = bview(b, {64, 128}, at::kBFloat16, k.dev), B = bview(b + 64 * 128 * 2, {64, 128}, at::kBFloat16, k.dev);
      auto Cm = bview(b + 2 * 64 * 128 * 2, {64, 64}, at::kBFloat16, k.dev);
      at::mm_out(Cm, A, B.t());
    }
  } catch (const std::exception&) {
    return FPR_E_CUDA;
  }
  return ck(cudaGetLastError());
}

namespace {
// stage 1d: the moe piece as the driver threads call it (rank's thread, device set, no GIL)
int moe_c(int r, int64_t h, int layer) {
  if (r < 0 || r >= MW || h < 0 || h >= kMaxMoe) return FPR_E_ARG;
  const FprPieceMoe* p = g_slot[h];
  if (p == nullptr || p->rank != r) return FPR_E_ARG;
  return piece_moe(p, fpr_drv_step_params(r), layer);
}

// 1b-i: the piece called from Python (the hook at the top of Engine.p_moe)
int moe(int r, int64_t h, int layer) {
  TORCH_CHECK(r >= 0 && r < MW && h >= 0 && h < kMaxMoe && g_slot[h] != nullptr, "pieces_moe: rank / handle");
  return moe_c(r, h, layer);
}
}  // namespace

void fpr_pieces_moe_register(pybind11::module_& m) {
  m.def("moe_init_rank", &moe_init_rank, "pieces_moe: per rank (device current): fork events + the agrs_pmws functions -> prod variants available");
  m.def("moe_init_fork_rank", &moe_init_fork_rank, "pieces_moe: per rank (device current): the fork tables' functions + inits");
  m.def("sizeof_moe_parts", &sizeof_moe_parts);
  m.def("sizeof_moe", &sizeof_moe);
  m.def("sizeof_pmws_args", &sizeof_pmws_args);
  m.def("add_moe", &add_moe, "pieces_moe: a static FprPieceMoe (bytes) -> handle");
  m.def("moe", &moe, "pieces_moe: run the moe piece (rank, handle, layer) -> FPR status");
  m.def("free_moe", &free_moe, "pieces_moe: release a handle (its bucket state was dropped)");
  m.def("moe_set_thread_init", &moe_set_thread_init, "pieces_moe: (rank, main stream, side stream, scratch ptr, bytes)");
  m.def("moe_init_thread", &fpr_pieces_moe_init_thread, "pieces_moe: the thread warm-up on the calling thread");
  m.def("live_moe", &live_moe, "pieces_moe: live handles");
  m.def("moe_set_nv", &moe_set_nv, "pieces_moe: (handle, device nv ptr): patch the prod Args' nv (a new tensor every mixed step)");
  fpr_drv_register_piece("moe", &moe_c);   // stage 1d: the span entry the driver threads call
}
