// agrs WS: the production-MoE layer of pm_rs.cu
// (agrs RSF, = kernels/moe_prefill/pm_fused.cu's layer + the RS push fused into the combine) with every per-call buffer
// served from ONE persistent per-GPU workspace instead of the caching allocator.
//
// pm_rs.cu is #included UNCHANGED (its kernels, helpers, pm_layer_impl and bindings are that file's text; only its
// PYBIND11_MODULE is renamed to a plain function, pm_rs_bind, and called from this module's init). This file adds:
//   pm_layer_ws_impl: pm_layer_impl's statements in the same order, where the eight torch::empty calls become views
//     of the workspace:
//       ids [T, 6] i32, tw [T, 6] f32, x [T, H] bf16 (compact rows; fvbuf still wins when given), meta
//       [moe_meta_size(E, RR)] i32, perm_w [RR] f32, h [RR, I] bf16, xp [RR, H] bf16 and slot [RR, H] bf16.
//       RR = 6 T.
//     xp and slot share one region. GEMM1 reads xp and GEMM2 writes slot after it on the same stream; production's
//     allocator does the same, since xp is freed before slot's same-size request, which takes xp's block.
//     The other regions are disjoint. Same kernels, same launch shapes, same operands: the outputs are bitwise those
//     of pm_layer_impl (tests/test_agrs_ws.py).
//     Production's moe_permute still allocates its small hist [cdiv(RR, 2048) * E] i32 (<= 0.1 MB: the small pool).
//   Fallback to pm_layer_impl (allocating, as today; counted in ws_stats, the first of each kind printed once per device):
//     - no workspace on the device;
//     - the stream is capturing (CUDA graphs: keep the allocator's graph-pool semantics);
//     - T above the workspace's row capacity, or H / E / I above its sizes.
//   Stream order: the workspace belongs to the stream of its last call; a call on another stream first records an
//   event on that stream and waits on it (one record + wait per switch; counted).
// Entry points (same arguments as layer / layer_rs): layer_ws, layer_rs_ws; ws_bytes(T, H, E, I) = the layout's
// size; ws_set(buf u8, T, H, E, I) binds a workspace to buf's device (the extension keeps a reference); ws_clear(dev);
// ws_stats() / ws_reset_stats().
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <atomic>
#include <cstdio>
#include <cstdint>

#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) static void pm_rs_bind(pybind11::module_& variable)
#include "pm_rs.cu"
#pragma pop_macro("PYBIND11_MODULE")

namespace {
constexpr int64_t WS_ALIGN = 4096;  // every region starts on a 4 KiB boundary (the caching allocator gives 512 B)
constexpr int WS_MAXDEV = 16;

struct WsLayout {
  int64_t x = 0, h = 0, xs = 0, ids = 0, tw = 0, pw = 0, meta = 0, bytes = 0;
};

inline int64_t ws_up(int64_t v) { return (v + WS_ALIGN - 1) / WS_ALIGN * WS_ALIGN; }

WsLayout ws_layout(int64_t T, int64_t H, int64_t E, int64_t I) {
  const int64_t RR = T * KTOP;
  WsLayout L;
  int64_t o = 0;
  L.x = o;    o = ws_up(o + T * H * 2);                            // x [T, H] bf16
  L.h = o;    o = ws_up(o + RR * I * 2);                           // h [RR, I] bf16
  L.xs = o;   o = ws_up(o + RR * H * 2);                           // xp, then slot [RR, H] bf16
  L.ids = o;  o = ws_up(o + T * KTOP * 4);                         // ids [T, 6] i32
  L.tw = o;   o = ws_up(o + T * KTOP * 4);                         // tw [T, 6] f32
  L.pw = o;   o = ws_up(o + RR * 4);                               // perm_w [RR] f32
  L.meta = o; o = ws_up(o + ds41h_moe_meta_size(E, RR) * 4);       // meta i32
  L.bytes = o;
  return L;
}

struct Ws {
  torch::Tensor buf;  // uint8 [L.bytes] (a reference: the workspace lives as long as it is bound)
  WsLayout L;
  int64_t tcap = 0, H = 0, E = 0, I = 0;
  cudaStream_t last = nullptr;
  bool has_last = false;
  cudaEvent_t ev = nullptr;
  bool warned[4] = {false, false, false, false};
};
Ws g_ws[WS_MAXDEV];

enum { ST_WS = 0, ST_FB_NONE, ST_FB_CAPTURE, ST_FB_SIZE, ST_SWITCH, ST_MAXT, ST_N };
std::atomic<int64_t> g_st[ST_N];
const char* const kStName[ST_N] = {"ws_calls", "fb_no_ws", "fb_capture", "fb_size", "stream_switches", "max_T"};

void ws_bump_max(int64_t T) {
  int64_t cur = g_st[ST_MAXT].load();
  while (T > cur && !g_st[ST_MAXT].compare_exchange_weak(cur, T)) {
  }
}

torch::Tensor ws_view(const Ws& w, int64_t off, std::initializer_list<int64_t> sizes, c10::ScalarType dt) {
  int64_t n = 1;
  for (auto s : sizes) n *= s;
  return w.buf.narrow(0, off, n * int64_t(c10::elementSize(dt))).view(dt).view(sizes);
}

// 0 = usable; else the fallback reason (ST_FB_*), printed once per device and reason.
int ws_usable(int dev, int64_t T, int64_t H, int64_t E, int64_t I, cudaStream_t s) {
  if (dev < 0 || dev >= WS_MAXDEV || !g_ws[dev].buf.defined()) return ST_FB_NONE;
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(s, &cs));
  if (cs != cudaStreamCaptureStatusNone) return ST_FB_CAPTURE;
  const Ws& w = g_ws[dev];
  if (T > w.tcap || H > w.H || E > w.E || I > w.I) return ST_FB_SIZE;
  return 0;
}

void ws_note_fallback(int dev, int why, int64_t T, int64_t H, int64_t E, int64_t I) {
  ++g_st[why];
  if (dev < 0 || dev >= WS_MAXDEV) return;
  Ws& w = g_ws[dev];
  if (why == ST_FB_NONE || w.warned[why]) return;
  w.warned[why] = true;
  std::fprintf(stderr, "agrs_ws: device %d: %s fallback to the allocating layer (T %lld, H %lld, E %lld, I %lld; workspace %lld rows) "
               "[printed once]\n", dev, kStName[why], (long long)T, (long long)H, (long long)E, (long long)I, (long long)w.tcap);
  std::fflush(stderr);
}

void ws_order(Ws& w, cudaStream_t s) {
  if (w.has_last && w.last != s) {
    C10_CUDA_CHECK(cudaEventRecord(w.ev, w.last));
    C10_CUDA_CHECK(cudaStreamWaitEvent(s, w.ev, 0));
    ++g_st[ST_SWITCH];
  }
  w.last = s;
  w.has_last = true;
}
}  // namespace

// pm_layer_impl with the per-call buffers in the workspace (see the file comment); the checks and statements in order.
static void pm_layer_ws_impl(torch::Tensor fall, torch::Tensor gr, torch::Tensor nv, int64_t R, int64_t vcount, torch::Tensor w13,
                             torch::Tensor s13, torch::Tensor w2, torch::Tensor s2, torch::Tensor sw13, torch::Tensor sw2, torch::Tensor gu,
                             torch::Tensor hsh, torch::Tensor sh, torch::Tensor part, double limit, int64_t variant, bool padded,
                             c10::optional<torch::Tensor> fvbuf, const RsPeers* rp, int rank, unsigned* epoch, int grid) {
  {
    const int dev = fall.is_cuda() ? int(fall.get_device()) : -1;
    const int64_t T_ = padded ? int64_t(nv.numel()) * R : vcount;
    const int64_t H_ = fall.dim() == 2 ? fall.size(1) : 0, E_ = w13.size(0), I_ = w13.size(1) / 2;
    int why = ST_FB_NONE;
    if (dev >= 0) {
      const c10::cuda::CUDAGuard g0(fall.device());
      why = ws_usable(dev, T_, H_, E_, I_, c10::cuda::getCurrentCUDAStream().stream());
    }
    if (why != 0) {
      ws_note_fallback(dev, why, T_, H_, E_, I_);
      pm_layer_impl(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, variant, padded, fvbuf, rp, rank,
                    epoch, grid);
      return;
    }
  }
  TORCH_CHECK(g_swiglu != nullptr, "pm_layer: set_swiglu(pm_swiglu launcher) first");
  const c10::cuda::CUDAGuard guard(fall.device());
  const int W = int(nv.numel());
  TORCH_CHECK(W >= 1 && W <= MAXW && nv.is_cuda() && nv.dtype() == torch::kInt32 && nv.is_contiguous(), "pm_layer: nv int32 [W] on device");
  const int64_t Tall = int64_t(W) * R, H = fall.size(1), E = w13.size(0), I = w13.size(1) / 2, Is = sw2.size(1);
  TORCH_CHECK(H % 8 == 0 && Is % 8 == 0 && sw13.size(0) == 2 * Is && sw13.size(1) == H && sw2.size(0) == H, "pm_layer: shapes");
  check_bf16_rows(fall, Tall, H, "fall");
  TORCH_CHECK(fall.size(0) == Tall, "pm_layer: fall must hold W R rows");
  TORCH_CHECK(gr.is_cuda() && gr.is_contiguous() && gr.dtype() == torch::kUInt8 && gr.dim() == 2 && gr.size(0) >= Tall && gr.size(1) == 64,
              "pm_layer: gr u8 [Tall, 64]");
  TORCH_CHECK(part.is_cuda() && part.is_contiguous() && part.dtype() == torch::kBFloat16 && part.size(0) == Tall && part.size(1) == H,
              "pm_layer: part bf16 [Tall, H]");
  const int64_t T = padded ? Tall : vcount;
  TORCH_CHECK(T >= 1 && T <= Tall, "pm_layer: 1 <= T <= Tall");
  check_bf16_rows(gu, T, 2 * Is, "gu");
  check_bf16_rows(hsh, T, Is, "hsh");
  check_bf16_rows(sh, T, H, "sh");
  const int v1 = int(variant & 15), v2 = int((variant >> 4) & 15);
  TORCH_CHECK(!(variant & ~int64_t(0xff)) && (v1 == 9 || v1 == 10 || v1 == 11 || v1 == 12) && v2 >= 11 && v2 <= 15,
              "pm_layer: variant ", variant, " is not a W4A16 act_perm16 variant");
  auto stream = c10::cuda::getCurrentCUDAStream();
  Ws& w = g_ws[fall.get_device()];
  TORCH_CHECK(w.buf.device() == fall.device(), "agrs_ws: workspace on ", w.buf.device(), ", call on ", fall.device());
  ws_order(w, stream.stream());
  ++g_st[ST_WS];
  ws_bump_max(T);
  auto opt = fall.options();
  auto ids = ws_view(w, w.L.ids, {T, KTOP}, torch::kInt32);
  auto tw = ws_view(w, w.L.tw, {T, KTOP}, torch::kFloat32);
  torch::Tensor x = fall;
  if (!padded) {
    if (fvbuf.has_value() && fvbuf->defined()) {
      check_bf16_rows(*fvbuf, T, H, "fvbuf");
      x = fvbuf->narrow(0, 0, T);
    } else {
      x = ws_view(w, w.L.x, {T, H}, torch::kBFloat16);
    }
  }
  k_pm_prep<<<unsigned(T), 128, 0, stream>>>(reinterpret_cast<const uint4*>(fall.data_ptr()), gr.data_ptr<uint8_t>(), nv.data_ptr<int>(), W,
                                             int(R), int(H / 8), padded ? 0 : 1, padded ? nullptr : reinterpret_cast<uint4*>(x.data_ptr()),
                                             ids.data_ptr<int>(), tw.data_ptr<float>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "k_pm_prep launch failed");
  auto guv = gu.narrow(0, 0, T), hv = hsh.narrow(0, 0, T), shv = sh.narrow(0, 0, T);
  at::mm_out(guv, x, sw13.t());
  g_swiglu(guv.data_ptr(), hv.data_ptr(), int(T), int(Is), stream.stream());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "swiglu launch failed");
  at::mm_out(shv, hv, sw2.t());
  const int64_t RR = T * KTOP;
  auto meta = ws_view(w, w.L.meta, {ds41h_moe_meta_size(E, RR)}, torch::kInt32);
  auto perm_w = ws_view(w, w.L.pw, {RR}, torch::kFloat32);
  ds41h_moe_permute(ids, tw, E, meta, perm_w);
  const int64_t t0 = 2 * (E + 1) + META_CTR_PM;
  auto h = ws_view(w, w.L.h, {RR, I}, torch::kBFloat16);
  {
    auto xp = ws_view(w, w.L.xs, {RR, H}, torch::kBFloat16);
    ds41h_act_perm16(x, meta.narrow(0, t0 + 2 * RR, RR), KTOP, xp);
    ds41h_moe_gemm(1, xp, perm_w, w13, s13, meta, perm_w, RR, v1, limit, h);
  }
  auto slot = ws_view(w, w.L.xs, {RR, H}, torch::kBFloat16);  // xp's region (GEMM1 has read it, stream order)
  ds41h_moe_gemm(2, h, perm_w, w2, s2, meta, perm_w, RR, v2, limit, slot);
  if (rp != nullptr) {
    k_pm_combine_push<<<unsigned(grid), 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(slot.data_ptr()), ids.data_ptr<int>(),
                                                         KTOP, int(H), int(E), tw.data_ptr<float>(),
                                                         reinterpret_cast<const __nv_bfloat16*>(shv.data_ptr()), *rp,
                                                         nv.data_ptr<int>(), W, int(R), padded ? 0 : 1, rank, epoch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "k_pm_combine_push launch failed");
    return;
  }
  const long n = long(Tall) * (H / 8);
  k_pm_combine<<<unsigned((n + 255) / 256), 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(slot.data_ptr()), ids.data_ptr<int>(),
                                                             KTOP, int(H), int(E), tw.data_ptr<float>(),
                                                             reinterpret_cast<const __nv_bfloat16*>(shv.data_ptr()),
                                                             reinterpret_cast<__nv_bfloat16*>(part.data_ptr()), nv.data_ptr<int>(), W, int(R),
                                                             padded ? 0 : 1);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "k_pm_combine launch failed");
}

void pm_layer_ws(torch::Tensor fall, torch::Tensor gr, torch::Tensor nv, int64_t R, int64_t vcount, torch::Tensor w13, torch::Tensor s13,
                 torch::Tensor w2, torch::Tensor s2, torch::Tensor sw13, torch::Tensor sw2, torch::Tensor gu, torch::Tensor hsh,
                 torch::Tensor sh, torch::Tensor part, double limit, int64_t variant, bool padded, c10::optional<torch::Tensor> fvbuf) {
  pm_layer_ws_impl(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, variant, padded, fvbuf, nullptr, 0,
                   nullptr, 0);
}

// pm_layer_rs's checks and peer table, verbatim, then the workspace layer.
void pm_layer_rs_ws(torch::Tensor fall, torch::Tensor gr, torch::Tensor nv, int64_t R, int64_t vcount, torch::Tensor w13, torch::Tensor s13,
                    torch::Tensor w2, torch::Tensor s2, torch::Tensor sw13, torch::Tensor sw2, torch::Tensor gu, torch::Tensor hsh,
                    torch::Tensor sh, torch::Tensor part, std::vector<torch::Tensor> bufs, std::vector<torch::Tensor> flags,
                    torch::Tensor epoch, int64_t rank, int64_t grid, double limit, int64_t variant, bool padded,
                    c10::optional<torch::Tensor> fvbuf) {
  const int W = int(nv.numel());
  TORCH_CHECK(int(bufs.size()) == W && int(flags.size()) == W, "layer_rs: one receive buffer and one flag per rank");
  TORCH_CHECK(rank >= 0 && rank < W && grid >= W && grid % W == 0 && grid <= 65535, "layer_rs: rank / grid (a multiple of W)");
  RsPeers rp;
  for (int q = 0; q < W; ++q) {
    TORCH_CHECK(bufs[q].is_cuda() && bufs[q].is_contiguous() && bufs[q].dtype() == torch::kBFloat16 &&
                    bufs[q].numel() >= int64_t(W) * R * fall.size(1),
                "layer_rs: receive buffer ", q, " must be contiguous bf16 with >= W R H elements");
    TORCH_CHECK(flags[q].is_cuda() && flags[q].dtype() == torch::kInt32, "layer_rs: flag ", q, " int32 on device");
    rp.buf[q] = reinterpret_cast<__nv_bfloat16*>(bufs[q].data_ptr());
    rp.flag[q] = reinterpret_cast<unsigned*>(flags[q].data_ptr());
  }
  TORCH_CHECK(epoch.is_cuda() && epoch.dtype() == torch::kInt32 && epoch.device() == fall.device(), "layer_rs: epoch int32 on fall's device");
  pm_layer_ws_impl(fall, gr, nv, R, vcount, w13, s13, w2, s2, sw13, sw2, gu, hsh, sh, part, limit, variant, padded, fvbuf, &rp, int(rank),
                   reinterpret_cast<unsigned*>(epoch.data_ptr()), int(grid));
}

int64_t ws_bytes(int64_t T, int64_t H, int64_t E, int64_t I) {
  TORCH_CHECK(T >= 1 && H >= 8 && E >= 1 && I >= 1, "ws_bytes: T, H, E, I must be positive");
  return ws_layout(T, H, E, I).bytes;
}

void ws_set(torch::Tensor buf, int64_t T, int64_t H, int64_t E, int64_t I) {
  TORCH_CHECK(buf.is_cuda() && buf.dtype() == torch::kUInt8 && buf.dim() == 1 && buf.is_contiguous() && buf.storage_offset() == 0,
              "ws_set: buf must be a whole contiguous CUDA uint8 tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(buf.data_ptr()) % 512 == 0, "ws_set: buf must be 512-B aligned");
  const int dev = int(buf.get_device());
  TORCH_CHECK(dev >= 0 && dev < WS_MAXDEV, "ws_set: device ", dev);
  const WsLayout L = ws_layout(T, H, E, I);
  TORCH_CHECK(buf.numel() >= L.bytes, "ws_set: buf holds ", buf.numel(), " B, the layout needs ", L.bytes);
  const c10::cuda::CUDAGuard guard(buf.device());
  Ws& w = g_ws[dev];
  if (w.ev == nullptr) C10_CUDA_CHECK(cudaEventCreateWithFlags(&w.ev, cudaEventDisableTiming));
  w.buf = buf;
  w.L = L;
  w.tcap = T;
  w.H = H;
  w.E = E;
  w.I = I;
  w.has_last = false;
  for (bool& b : w.warned) b = false;
}

void ws_clear(int64_t dev) {
  TORCH_CHECK(dev >= 0 && dev < WS_MAXDEV, "ws_clear: device ", dev);
  Ws& w = g_ws[dev];
  if (w.buf.defined()) {
    const c10::cuda::CUDAGuard guard(w.buf.device());
    if (w.has_last) C10_CUDA_CHECK(cudaStreamSynchronize(w.last));  // no queued work still uses the buffer
  }
  w.buf = torch::Tensor();
  w.tcap = w.H = w.E = w.I = 0;
  w.has_last = false;
}

std::vector<int64_t> ws_info(int64_t dev) {
  TORCH_CHECK(dev >= 0 && dev < WS_MAXDEV, "ws_info: device ", dev);
  const Ws& w = g_ws[dev];
  return {w.buf.defined() ? w.buf.numel() : 0, w.tcap, w.H, w.E, w.I};
}

std::map<std::string, int64_t> ws_stats() {
  std::map<std::string, int64_t> m;
  for (int i = 0; i < ST_N; ++i) m[kStName[i]] = g_st[i].load();
  return m;
}

void ws_reset_stats() {
  for (auto& a : g_st) a.store(0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pm_rs_bind(m);  // pm_rs.cu's module body: production's moe_prefill bindings, set_swiglu, layer_rs, layer
  m.def("layer_ws", &pm_layer_ws, "layer with the per-call buffers in this device's workspace (fallback: layer)");
  m.def("layer_rs_ws", &pm_layer_rs_ws, "layer_rs with the per-call buffers in this device's workspace (fallback: layer_rs)");
  m.def("ws_bytes", &ws_bytes, "workspace bytes for T call rows (all ranks), hidden H, E experts, I intermediate per rank");
  m.def("ws_set", &ws_set, "bind buf (CUDA uint8, >= ws_bytes) as its device's workspace for T rows, H, E, I");
  m.def("ws_clear", &ws_clear, "unbind a device's workspace (after its stream's queued work)");
  m.def("ws_info", &ws_info, "[bytes, T rows, H, E, I] of a device's workspace (zeros: none)");
  m.def("ws_stats", &ws_stats, "process-wide counters: ws_calls, fb_no_ws, fb_capture, fb_size, stream_switches, max_T");
  m.def("ws_reset_stats", &ws_reset_stats, "zero the counters");
}
