// pm_rs_cabi.cu
//
// Three translation units, as the served builds (agrs_pm_rs: moe_route.cu, moe_prefill.cu, pm_rs.cu), each source
// #included UNCHANGED with HOST code added only (device code, hence SASS, as served; gate tests/sass_gate.py):
//   moe_route_ws.cu   moe_route.cu + agrs_pmws_moe_permute (production's permute with the hist from the caller's ws).
//   moe_prefill_ws.cu moe_prefill.cu + agrs_pmws_set_hybrid (its set_hybrid statics, for agrs_pmws_init).
//   this file         agrs/pm_rs.cu (the RSF kernels k_pm_prep / k_pm_combine / k_pm_combine_push, pm_layer_impl, its
//                     bindings: PYBIND11_MODULE renamed to pm_rs_bind by push_macro, called from this module's init) +
//                     the C ABI + its pybind callers.
// (One combined TU compiled moe_count_kernel and moe_gemm2_bs_kernel<0> to different SASS: not used.)
// pm_core: pm_layer_impl's statements in order with every buffer from the caller's workspace (pm_rs_ws.cu's layout +
// hist; xp and slot share one region, as the caching allocator reuses xp's block for slot), launched on Args.stream:
// torch's current stream is set to it for the call (CUDAStreamGuard over getStreamFromExternal: thread-local), so
// production's launch_pdl and at::mm_out (cuBLAS) use it. No allocation, no host sync, no event, no device change.
// Errors -> FPR_E_* (exceptions caught at the boundary).
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <atomic>
#include <cstdint>

#include "fpr_cabi.h"
#include "pm_rs_cabi.h"
// the shared-expert GEMMs through fpr_lt, which lives in pm_rs_lt.cpp (host TU of this .so: cudafe++ cannot
// compile fpr_lt.h); host code only, device code / SASS unchanged
void agrs_lt_mm(at::Tensor& y, const at::Tensor& a, const at::Tensor& w, const char* site);

#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) static void pm_rs_bind(pybind11::module_& variable)
#include "pm_rs.cu"
#pragma pop_macro("PYBIND11_MODULE")

#if __has_include("agrs_pmws_md5.h")
#include "agrs_pmws_md5.h"   // written by agrs_pmws_cabi.sync_sources into build/pmc_src: md5 of the sources of this build
#endif
#ifndef AGRS_PMWS_SRC_MD5
#define AGRS_PMWS_SRC_MD5 "unset"
#endif

// host shims in the production TUs (moe_route_ws.cu / moe_prefill_ws.cu)
int64_t agrs_pmws_hist_ints(int64_t R, int64_t E);
int agrs_pmws_moe_permute(const int* topk_ids, const float* topk_w, int R, int K, int64_t E, int* meta, float* perm_w, int* hist);
void agrs_pmws_set_hybrid(int64_t split, int64_t engine);

namespace {
constexpr int64_t C_ALIGN = 4096;
constexpr int C_MAXDEV = 32;
constexpr int64_t C_MAX_E = 1024;   // production's MAX_EXPERTS (moe_route.cu) = MAX_E (moe_prefill.cu)
inline int64_t c_up(int64_t v) { return (v + C_ALIGN - 1) / C_ALIGN * C_ALIGN; }

struct CLayout {
  int64_t x = 0, h = 0, xs = 0, ids = 0, tw = 0, pw = 0, meta = 0, hist = 0, bytes = 0;
};

CLayout c_layout(int64_t T, int64_t H, int64_t E, int64_t I) {
  const int64_t RR = T * KTOP;
  CLayout L;
  int64_t o = 0;
  L.x = o;    o = c_up(o + T * H * 2);                              // x [T, H] bf16 (compact gather)
  L.h = o;    o = c_up(o + RR * I * 2);                             // h [RR, I] bf16
  L.xs = o;   o = c_up(o + RR * H * 2);                             // xp, then slot [RR, H] bf16
  L.ids = o;  o = c_up(o + T * KTOP * 4);                           // ids [T, 6] i32
  L.tw = o;   o = c_up(o + T * KTOP * 4);                           // tw [T, 6] f32
  L.pw = o;   o = c_up(o + RR * 4);                                 // perm_w [RR] f32
  L.meta = o; o = c_up(o + ds41h_moe_meta_size(E, RR) * 4);         // meta i32
  L.hist = o; o = c_up(o + agrs_pmws_hist_ints(RR, E) * 4);         // moe_permute's hist [G E] i32
  L.bytes = o;
  return L;
}

// init state: written by agrs_pmws_init only (rule 4: settings made at init, identical for all devices)
std::atomic<uint32_t> g_init_mask{0};
std::atomic<int> g_init_set{0};
int64_t g_init_split = 0, g_init_engine = 0, g_init_swiglu = 0;

at::Tensor view(const void* p, at::IntArrayRef sizes, at::ScalarType dt, int dev) {
  const at::Device d(at::kCUDA, static_cast<c10::DeviceIndex>(dev));
  return at::for_blob(const_cast<void*>(p), sizes).options(at::TensorOptions().dtype(dt).device(d)).target_device(d).make_tensor();
}

int64_t layer_T(const AgrsPmwsLayerArgs* a) { return a->padded ? a->W * a->R : a->vcount; }

int pm_core(const AgrsPmwsLayerArgs* a, bool rs) {
  if (a == nullptr) return FPR_E_ARG;
  const int dev = a->call.device;
  if (dev < 0 || dev >= C_MAXDEV) return FPR_E_ARG;
  if (!(g_init_mask.load() & (1u << dev)) || g_swiglu == nullptr) return FPR_E_INIT;
  const int64_t W = a->W, R = a->R, H = a->H, E = a->E, I = a->I, Is = a->Is;
  if (W < 1 || W > AGRS_PMWS_MAXW || R < 1 || H < 8 || H % 8 || Is < 8 || Is % 8 || E < 1 || I < 1) return FPR_E_ARG;
  if (E > C_MAX_E) return FPR_E_SHAPE;
  const int64_t Tall = W * R, T = layer_T(a);
  if (T < 1 || T > Tall || a->scratch_rows < T) return FPR_E_ARG;
  const int v1 = int(a->variant & 15), v2 = int((a->variant >> 4) & 15);
  if ((a->variant & ~int64_t(0xff)) || !(v1 == 9 || v1 == 10 || v1 == 11 || v1 == 12) || v2 < 11 || v2 > 15) return FPR_E_SHAPE;
  if (!a->fall || !a->gr || !a->nv || !a->w13 || !a->s13 || !a->w2 || !a->s2 || !a->sw13 || !a->sw2 || !a->gu || !a->hsh ||
      !a->sh || !a->ws)
    return FPR_E_ARG;
  RsPeers rp{};
  if (rs) {
    if (a->grid < W || a->grid % W || a->grid > 65535 || a->call.rank < 0 || a->call.rank >= W || a->epoch == nullptr) return FPR_E_ARG;
    for (int q = 0; q < W; ++q) {
      if (a->rs_buf[q] == nullptr || a->rs_flag[q] == nullptr) return FPR_E_ARG;
      rp.buf[q] = reinterpret_cast<__nv_bfloat16*>(a->rs_buf[q]);
      rp.flag[q] = reinterpret_cast<unsigned*>(a->rs_flag[q]);
    }
  } else if (a->part == nullptr) {
    return FPR_E_ARG;
  }
  const CLayout L = c_layout(T, H, E, I);
  if (a->ws_bytes < L.bytes) return FPR_E_WS;
  if (reinterpret_cast<uintptr_t>(a->ws) % 512) return FPR_E_ARG;
  try {
    const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(a->call.stream, static_cast<c10::DeviceIndex>(dev)));
    const cudaStream_t stream = a->call.stream;
    uint8_t* base = static_cast<uint8_t*>(a->ws);
    const int Wi = int(W);
    int* ids = reinterpret_cast<int*>(base + L.ids);
    float* tw = reinterpret_cast<float*>(base + L.tw);
    void* xptr = a->padded ? const_cast<void*>(a->fall) : static_cast<void*>(base + L.x);
    k_pm_prep<<<unsigned(T), 128, 0, stream>>>(reinterpret_cast<const uint4*>(a->fall), a->gr, a->nv, Wi, int(R), int(H / 8),
                                               a->padded ? 0 : 1, a->padded ? nullptr : reinterpret_cast<uint4*>(base + L.x), ids, tw);
    if (cudaGetLastError() != cudaSuccess) return FPR_E_CUDA;
    // shared expert slice (engine p_moe_prod: torch.mm, the fork's swiglu, torch.mm; stock rounding points)
    const at::Tensor x = view(xptr, {T, H}, at::kBFloat16, dev);
    at::Tensor guv = view(a->gu, {T, 2 * Is}, at::kBFloat16, dev), hv = view(a->hsh, {T, Is}, at::kBFloat16, dev),
               shv = view(a->sh, {T, H}, at::kBFloat16, dev);
    const at::Tensor sw13 = view(a->sw13, {2 * Is, H}, at::kBFloat16, dev), sw2 = view(a->sw2, {H, Is}, at::kBFloat16, dev);
    agrs_lt_mm(guv, x, sw13, "agrs_cabi_sh_gu");   // = at::mm_out(guv, x, sw13.t()) through fpr_lt
    g_swiglu(guv.data_ptr(), hv.data_ptr(), int(T), int(Is), stream);
    if (cudaGetLastError() != cudaSuccess) return FPR_E_CUDA;
    agrs_lt_mm(shv, hv, sw2, "agrs_cabi_sh_down");   // = at::mm_out(shv, hv, sw2.t())
    // routed experts (= ds41h_moe_forward's W4A16 path)
    const int64_t RR = T * KTOP;
    int* meta = reinterpret_cast<int*>(base + L.meta);
    float* perm_w = reinterpret_cast<float*>(base + L.pw);
    if (agrs_pmws_moe_permute(ids, tw, int(RR), KTOP, E, meta, perm_w, reinterpret_cast<int*>(base + L.hist)) != 0) return FPR_E_SHAPE;
    const int64_t t0 = 2 * (E + 1) + META_CTR_PM;
    const at::Tensor metat = view(meta, {ds41h_moe_meta_size(E, RR)}, at::kInt, dev), pwt = view(perm_w, {RR}, at::kFloat, dev);
    const at::Tensor w13 = view(a->w13, {E, 2 * I, H / 2}, at::kByte, dev), s13 = view(a->s13, {E, 2 * I, H / 32}, at::kByte, dev);
    const at::Tensor w2 = view(a->w2, {E, H, I / 2}, at::kByte, dev), s2 = view(a->s2, {E, H, I / 32}, at::kByte, dev);
    at::Tensor h = view(base + L.h, {RR, I}, at::kBFloat16, dev);
    {
      at::Tensor xp = view(base + L.xs, {RR, H}, at::kBFloat16, dev);
      ds41h_act_perm16(x, metat.narrow(0, t0 + 2 * RR, RR), KTOP, xp);
      ds41h_moe_gemm(1, xp, pwt, w13, s13, metat, pwt, RR, v1, a->limit, h);
    }
    at::Tensor slot = view(base + L.xs, {RR, H}, at::kBFloat16, dev);   // xp's region (GEMM1 has read it, stream order)
    ds41h_moe_gemm(2, h, pwt, w2, s2, metat, pwt, RR, v2, a->limit, slot);
    if (rs) {
      k_pm_combine_push<<<unsigned(a->grid), 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(slot.data_ptr()), ids, KTOP,
                                                             int(H), int(E), tw, reinterpret_cast<const __nv_bfloat16*>(shv.data_ptr()),
                                                             rp, a->nv, Wi, int(R), a->padded ? 0 : 1, a->call.rank,
                                                             reinterpret_cast<unsigned*>(a->epoch));
    } else {
      const long n = long(Tall) * (H / 8);
      k_pm_combine<<<unsigned((n + 255) / 256), 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(slot.data_ptr()), ids, KTOP,
                                                                 int(H), int(E), tw, reinterpret_cast<const __nv_bfloat16*>(shv.data_ptr()),
                                                                 reinterpret_cast<__nv_bfloat16*>(a->part), a->nv, Wi, int(R),
                                                                 a->padded ? 0 : 1);
    }
    if (cudaGetLastError() != cudaSuccess) return FPR_E_CUDA;
  } catch (const std::exception&) {
    return FPR_E_ARG;   // a production TORCH_CHECK (shape / dtype) or launch_pdl failure
  }
  return FPR_OK;
}
}  // namespace

extern "C" {

int64_t agrs_pmws_ws_bytes_for(int64_t T, int64_t H, int64_t E, int64_t I) {
  if (T < 1 || H < 8 || E < 1 || E > C_MAX_E || I < 1) return FPR_E_ARG;
  return c_layout(T, H, E, I).bytes;
}

int64_t agrs_pmws_layer_ws_bytes(const void* p) {
  const auto* a = static_cast<const AgrsPmwsLayerArgs*>(p);
  if (a == nullptr) return FPR_E_ARG;
  return agrs_pmws_ws_bytes_for(layer_T(a), a->H, a->E, a->I);
}

int agrs_pmws_init(const void* p) {
  const auto* a = static_cast<const AgrsPmwsInitArgs*>(p);
  if (a == nullptr || a->device < 0 || a->device >= C_MAXDEV || a->swiglu_fn == 0 || a->hybrid_split < 0) return FPR_E_ARG;
  int expect = 0;
  if (g_init_set.compare_exchange_strong(expect, 1)) {   // first init in the process: take the settings
    g_init_split = a->hybrid_split;
    g_init_engine = a->hybrid_engine;
    g_init_swiglu = a->swiglu_fn;
    agrs_pmws_set_hybrid(a->hybrid_split, a->hybrid_engine);
    pm_set_swiglu(a->swiglu_fn);
    g_init_set.store(2);
  } else {
    while (g_init_set.load() != 2) {
    }
    if (a->hybrid_split != g_init_split || a->hybrid_engine != g_init_engine || a->swiglu_fn != g_init_swiglu) return FPR_E_ARG;
  }
  g_init_mask.fetch_or(1u << a->device);
  return FPR_OK;
}

int agrs_pmws_init_thread(const void* p) {
  const auto* a = static_cast<const AgrsPmwsThreadArgs*>(p);
  if (a == nullptr || a->call.device < 0 || a->call.device >= C_MAXDEV || a->ws == nullptr) return FPR_E_ARG;
  if (a->ws_bytes < 3 * 64 * 128 * 2) return FPR_E_WS;
  try {
    const int dev = a->call.device;
    const c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(a->call.stream, static_cast<c10::DeviceIndex>(dev)));
    uint8_t* b = static_cast<uint8_t*>(a->ws);
    const at::Tensor A = view(b, {64, 128}, at::kBFloat16, dev), B = view(b + 64 * 128 * 2, {64, 128}, at::kBFloat16, dev);
    at::Tensor C = view(b + 2 * 64 * 128 * 2, {64, 64}, at::kBFloat16, dev);
    at::mm_out(C, A, B.t());   // the layer's mm pattern (x [T, H] @ w [N, H]^T): cuBLAS handle + workspace for this thread / stream
  } catch (const std::exception&) {
    return FPR_E_CUDA;
  }
  return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}

int agrs_pmws_layer(const void* p) { return pm_core(static_cast<const AgrsPmwsLayerArgs*>(p), false); }
int agrs_pmws_layer_rs(const void* p) { return pm_core(static_cast<const AgrsPmwsLayerArgs*>(p), true); }

static const FprFnEntry kFns[] = {
    {"agrs_pmws.init", agrs_pmws_init, nullptr, int64_t(sizeof(AgrsPmwsInitArgs))},
    {"agrs_pmws.init_thread", agrs_pmws_init_thread, nullptr, int64_t(sizeof(AgrsPmwsThreadArgs))},
    {"agrs_pmws.layer", agrs_pmws_layer, agrs_pmws_layer_ws_bytes, int64_t(sizeof(AgrsPmwsLayerArgs))},
    {"agrs_pmws.layer_rs", agrs_pmws_layer_rs, agrs_pmws_layer_ws_bytes, int64_t(sizeof(AgrsPmwsLayerArgs))},
};
static const FprExtTable kTable = {FPR_CABI_VERSION, "agrs_pmws", AGRS_PMWS_SRC_MD5, int(sizeof(kFns) / sizeof(kFns[0])), kFns};

__attribute__((visibility("default"))) const FprExtTable* fpr_agrs_pmws_table(void) { return &kTable; }

}  // extern "C"

// ---- pybind: thin callers of the same C functions (tensors -> Args)
namespace {
void chk(int rc, const char* fn) { TORCH_CHECK(rc == FPR_OK, "agrs_pmws.", fn, ": error ", rc); }

void ck_bf16(const torch::Tensor& t, const char* nm) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dtype() == torch::kBFloat16, nm, ": contiguous CUDA bf16");
}

AgrsPmwsLayerArgs make_args(torch::Tensor fall, torch::Tensor gr, torch::Tensor nv, int64_t R, int64_t vcount, torch::Tensor w13,
                            torch::Tensor s13, torch::Tensor w2, torch::Tensor s2, torch::Tensor sw13, torch::Tensor sw2,
                            torch::Tensor gu, torch::Tensor hsh, torch::Tensor sh, double limit, int64_t variant, bool padded,
                            torch::Tensor ws) {
  ck_bf16(fall, "fall"), ck_bf16(sw13, "sw13"), ck_bf16(sw2, "sw2"), ck_bf16(gu, "gu"), ck_bf16(hsh, "hsh"), ck_bf16(sh, "sh");
  for (const auto* t : {&w13, &s13, &w2, &s2, &gr, &ws})
    TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->dtype() == torch::kUInt8, "u8 operands / ws: contiguous CUDA uint8");
  TORCH_CHECK(nv.is_cuda() && nv.is_contiguous() && nv.dtype() == torch::kInt32, "nv: contiguous CUDA int32");
  TORCH_CHECK(gu.size(0) == hsh.size(0) && hsh.size(0) == sh.size(0), "gu / hsh / sh: same rows");
  AgrsPmwsLayerArgs a{};
  const int dev = int(fall.get_device());
  a.call.stream = c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(dev)).stream();
  a.call.device = dev;
  a.call.rank = 0;
  a.fall = fall.data_ptr();
  a.gr = gr.data_ptr<uint8_t>();
  a.nv = nv.data_ptr<int32_t>();
  a.W = nv.numel(), a.R = R, a.vcount = vcount, a.H = fall.size(1);
  a.w13 = w13.data_ptr<uint8_t>(), a.s13 = s13.data_ptr<uint8_t>(), a.w2 = w2.data_ptr<uint8_t>(), a.s2 = s2.data_ptr<uint8_t>();
  a.E = w13.size(0), a.I = w13.size(1) / 2;
  a.sw13 = sw13.data_ptr(), a.sw2 = sw2.data_ptr(), a.Is = sw2.size(1);
  a.gu = gu.data_ptr(), a.hsh = hsh.data_ptr(), a.sh = sh.data_ptr(), a.scratch_rows = gu.size(0);
  a.limit = limit, a.variant = variant, a.padded = padded ? 1 : 0;
  a.ws = ws.data_ptr(), a.ws_bytes = ws.numel();
  TORCH_CHECK(fall.size(0) == a.W * R, "fall must hold W R rows");
  TORCH_CHECK(gr.dim() == 2 && gr.size(0) >= a.W * R && gr.size(1) == 64, "gr: u8 [>= W R, 64]");
  TORCH_CHECK(w13.dim() == 3 && s13.dim() == 3 && w2.dim() == 3 && s2.dim() == 3 && w2.size(1) == a.H && s2.size(1) == a.H &&
                  w13.size(2) * 2 == a.H && w2.size(2) * 2 == a.I && sw13.size(0) == 2 * a.Is && sw13.size(1) == a.H && sw2.size(0) == a.H,
              "expert / shared-slice shapes");
  return a;
}

void py_layer(torch::Tensor fall, torch::Tensor gr, torch::Tensor nv, int64_t R, int64_t vcount, torch::Tensor w13, torch::Tensor s13,
              torch::Tensor w2, torch::Tensor s2, torch::Tensor sw13, torch::Tensor sw2, torch::Tensor gu, torch::Tensor hsh,
              torch::Tensor sh, torch::Tensor part, double limit, int64_t variant, bool padded, torch::Tensor ws) {
  AgrsPmwsLayerArgs a = make_args(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, limit, variant, padded, ws);
  ck_bf16(part, "part");
  TORCH_CHECK(part.size(0) == a.W * R && part.size(1) == a.H, "part [W R, H]");
  a.part = part.data_ptr();
  chk(agrs_pmws_layer(&a), "layer");
}

void py_layer_rs(torch::Tensor fall, torch::Tensor gr, torch::Tensor nv, int64_t R, int64_t vcount, torch::Tensor w13,
                 torch::Tensor s13, torch::Tensor w2, torch::Tensor s2, torch::Tensor sw13, torch::Tensor sw2, torch::Tensor gu,
                 torch::Tensor hsh, torch::Tensor sh, std::vector<torch::Tensor> bufs, std::vector<torch::Tensor> flags,
                 torch::Tensor epoch, int64_t rank, int64_t grid, double limit, int64_t variant, bool padded, torch::Tensor ws) {
  AgrsPmwsLayerArgs a = make_args(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, limit, variant, padded, ws);
  TORCH_CHECK(int64_t(bufs.size()) == a.W && int64_t(flags.size()) == a.W, "one receive buffer and one flag per rank");
  for (int64_t q = 0; q < a.W; ++q) {
    ck_bf16(bufs[q], "rs buffer");
    TORCH_CHECK(bufs[q].numel() >= a.W * R * a.H, "rs buffer: >= W R H elements");
    TORCH_CHECK(flags[q].is_cuda() && flags[q].dtype() == torch::kInt32, "flag: CUDA int32");
    a.rs_buf[q] = bufs[q].data_ptr();
    a.rs_flag[q] = reinterpret_cast<uint32_t*>(flags[q].data_ptr());
  }
  TORCH_CHECK(epoch.is_cuda() && epoch.dtype() == torch::kInt32 && epoch.device() == fall.device(), "epoch: int32 on fall's device");
  a.epoch = reinterpret_cast<uint32_t*>(epoch.data_ptr());
  a.call.rank = int(rank);
  a.grid = grid;
  chk(agrs_pmws_layer_rs(&a), "layer_rs");
}

void py_init(int64_t device, int64_t hybrid_split, int64_t hybrid_engine, int64_t swiglu_fn) {
  AgrsPmwsInitArgs a{};
  a.device = int32_t(device), a.hybrid_split = hybrid_split, a.hybrid_engine = hybrid_engine, a.swiglu_fn = swiglu_fn;
  chk(agrs_pmws_init(&a), "init");
}

void py_init_thread(torch::Tensor ws) {
  TORCH_CHECK(ws.is_cuda() && ws.dtype() == torch::kUInt8, "ws: CUDA uint8");
  AgrsPmwsThreadArgs a{};
  const int dev = int(ws.get_device());
  a.call.stream = c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(dev)).stream();
  a.call.device = dev;
  a.ws = ws.data_ptr(), a.ws_bytes = ws.numel();
  chk(agrs_pmws_init_thread(&a), "init_thread");
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pm_rs_bind(m);   // pm_rs.cu's module body: production's moe_prefill bindings, set_swiglu, layer_rs, layer (allocating)
  m.def("cabi_layer", &py_layer, "agrs_pmws.layer through the C ABI (caller-owned ws)");
  m.def("cabi_layer_rs", &py_layer_rs, "agrs_pmws.layer_rs through the C ABI (caller-owned ws)");
  m.def("cabi_init", &py_init, "agrs_pmws.init(device, hybrid_split, hybrid_engine, swiglu launcher)");
  m.def("cabi_init_thread", &py_init_thread, "agrs_pmws.init_thread on the current stream (ws >= 48 KiB)");
  m.def("cabi_ws_bytes", &agrs_pmws_ws_bytes_for, "workspace bytes for T call rows, H, E, I");
  m.def("cabi_args_size", []() { return int64_t(sizeof(AgrsPmwsLayerArgs)); });
  m.def("cabi_table_md5", []() { return std::string(fpr_agrs_pmws_table()->src_md5); });
}
