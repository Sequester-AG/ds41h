// pieces_attn.cpp -- the attention pieces of the C++ mixed-step driver. Compiled
// into FPR's mixed_cxx extension (mixed_cxx.py picks up every pieces_*.cpp); driver.cpp calls fpr_pieces_attn_register
// (weak symbol). Python side: mixed_cxx_attn.py.
#include <cstdio>
#include <type_traits>
#include <mutex>
#include <thread>
#include <atomic>
#include "fpr_lt.h"   // cached cublasLt plans (closed algo set after warm) for every GEMM site here
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "pieces_attn.h"   // (first: pieces.h's FprStepParams, which driver_api.h uses)
#include "driver_api.h"

namespace {
// the pointers are this rank's by construction (Python's views of rank r's state). Also one driver query less
// per view.
at::Tensor blob(void* p, at::IntArrayRef sizes, const at::TensorOptions& o) {
  return at::for_blob(p, sizes).options(o).target_device(o.device()).make_tensor();
}
at::Tensor blob(void* p, at::IntArrayRef sizes, at::IntArrayRef strides, const at::TensorOptions& o) {
  return at::for_blob(p, sizes).strides(strides).options(o).target_device(o.device()).make_tensor();
}
constexpr int AW_ = 4;
// FPR_PIECES_THROW=1: the pieces' catch blocks rethrow
// instead of returning FPR_E_CUDA, so the driver's span error (last_error) / the pybind caller carries the exception text.
// Error path only: with the flag off (or no exception) nothing changes.
bool pieces_throw() {
  static const bool t = [] {
    const char* e = std::getenv("FPR_PIECES_THROW");
    return e != nullptr && e[0] == '1';
  }();
  return t;
}
// the same flag for the sub-calls that fail by STATUS (C ABI tables, Triton / capture replays): a labelled
// exception carrying the status, so a bare -5 / -2 / -1 says which sub-call returned it
int thr_rc(int rc, const char* where) {
  if (rc != FPR_OK && pieces_throw())
    throw std::runtime_error(std::string("pieces_attn ") + where + ": FPR status " + std::to_string(rc));
  return rc;
}
std::vector<std::unique_ptr<FprPieceAttnA>> g_attn_a;
// each attn_b handle carries its view's generation (add_attn_b's gen; 0 = unknown): a plan
// slot's owner is (gen, si pointer, R), so a new view whose si block reuses a freed view's address is still another owner.
// Every FprPieceAttnB the pieces run is the first member of one of these (handles come only from add_attn_b).
struct AttnBH {
  FprPieceAttnB b;
  uint64_t gen;
};
static_assert(std::is_standard_layout<AttnBH>::value && offsetof(AttnBH, b) == 0, "AttnBH layout");
std::vector<std::unique_ptr<AttnBH>> g_attn_b;
FprFn F_norm_rope2 = nullptr, F_w8gemm = nullptr, F_ropeq = nullptr, F_actq = nullptr, F_dg_gemm = nullptr, F_dg_einsum = nullptr;
FprFn F_inv_rope = nullptr;    // attn_dense.inv_rope (attn_b's wo = 1 / 2 tails: ENG_WO_W8A16)
FprFn F_rope_q_insert = nullptr;   // attn_dense.rope_q_insert (attn_b's qdq = 1: ENG_QB_DQ)
std::vector<std::unique_ptr<FprKcRecord>> g_dq;   // dequant records (attn_add_dq_record; built between spans only)
int dq_launch(int64_t h, cudaStream_t s) {
  if (h < 0 || h >= (int64_t)g_dq.size() || !g_dq[h]) return thr_rc(FPR_E_ARG, "attn_b dq record handle");
  return thr_rc(fpr_kc_launch(g_dq[h].get(), (CUstream)s), "attn_b dq dequant relaunch");
}
// FPR_ATTN_DQ_PF=1 (default off; read once per process): the combined attn_b's wo = 2 dequants (dq_a, dq_b)
// on a per-device side stream forked from the rank stream at the piece's start. Slots a / b are free there: their previous
// readers (the last wo_a / wo_b GEMMs) are earlier on the rank stream, which the fork event orders before the side stream's
// dequants (mixed_cxx's _dq_handover has already drained Python's pending side-stream dequants on the rank stream and reset
// _Dq's owners before the span). The rank stream joins (waits the side stream's event) before wo_a, so the GEMMs read exactly
// what the inline dequants write; nothing stays pending after the piece, also on error paths. Streams / events are created
// once for every device in attn_add_dq_record (between spans); a capturing rank stream or a second thread on a device falls
// back to the inline dequants.
struct DqPf {
  cudaStream_t s = nullptr;
  cudaEvent_t fork = nullptr, join = nullptr;
  std::atomic<std::thread::id> tid{std::thread::id()};   // the device's driver thread (claimed once, atomically)
};
DqPf g_dqpf[16];
bool dq_pf_on() {
  static const bool v = [] { const char* e = std::getenv("FPR_ATTN_DQ_PF"); return e && e[0] == '1' && e[1] == 0; }();
  return v;
}
void dq_pf_init_all() {   // once, from attn_add_dq_record: called only from the Python main thread between spans (the dq
                          // records are built in mixed_cxx_attn's setup), so the unsynchronised `done` is safe
  static bool done = false;
  if (done || !dq_pf_on()) return;
  done = true;
  int n = 0, cur = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || cudaGetDevice(&cur) != cudaSuccess) return;
  for (int d = 0; d < n && d < 16; ++d) {
    DqPf& q = g_dqpf[d];
    if (cudaSetDevice(d) != cudaSuccess) continue;
    if (cudaStreamCreateWithFlags(&q.s, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&q.fork, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&q.join, cudaEventDisableTiming) != cudaSuccess)
      q.s = nullptr;                     // (this device keeps the inline dequants)
  }
  cudaSetDevice(cur);
}
bool dq_pf_usable(const FprPieceAttnB* p) {
  if (!dq_pf_on() || p->wo != 2 || p->dq_a < 0 || p->dq_b < 0 || p->device < 0 || p->device >= 16) return false;
  DqPf& d = g_dqpf[p->device];
  if (d.s == nullptr) return false;
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(p->main, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) return false;
  const std::thread::id me = std::this_thread::get_id();
  std::thread::id none{};
  if (d.tid.compare_exchange_strong(none, me)) return true;   // first use: this thread owns the device
  return none == me;                     // one driver thread per device (else the inline dequants)
}
int dq_pf_fork(const FprPieceAttnB* p) {
  DqPf& d = g_dqpf[p->device];
  if (cudaEventRecord(d.fork, p->main) != cudaSuccess || cudaStreamWaitEvent(d.s, d.fork, 0) != cudaSuccess)
    return thr_rc(FPR_E_CUDA, "attn_b dq_pf fork");
  // after the wait, ALWAYS re-record join, whatever the launches returned, so a later join wait covers
  // everything this fork put on the side stream; if that record itself fails, drain the side stream before returning
  int rc = dq_launch(p->dq_a, d.s);
  if (rc == FPR_OK) rc = dq_launch(p->dq_b, d.s);
  if (cudaEventRecord(d.join, d.s) != cudaSuccess) {
    cudaStreamSynchronize(d.s);
    return rc != FPR_OK ? rc : thr_rc(FPR_E_CUDA, "attn_b dq_pf join record");
  }
  return rc;
}
int dq_pf_join(const FprPieceAttnB* p) {
  return cudaStreamWaitEvent(p->main, g_dqpf[p->device].join, 0) == cudaSuccess ? FPR_OK
                                                                                : thr_rc(FPR_E_CUDA, "attn_b dq_pf join wait");
}
c10::OperatorHandle* g_fmla_dec = nullptr;   // _flashmla_C::sparse_decode_fwd
std::vector<std::unique_ptr<FprPieceAttnIdx>> g_attn_idx;
FprFn F_prep2 = nullptr, F_bmax = nullptr, F_topk = nullptr, F_cand_logits = nullptr, F_cand_prep = nullptr;
FprAttnIdxStep g_idx_step[4];   // per rank, the segment's idx values (attn_set_idx_step)
FprAttnPfStep g_pf_step[4];     // per rank, the step's attn_prefill.Plan (attn_set_pf_step; stage 1c-1)
FprFn F_gather = nullptr;       // gatherk.gather (the K gather of the prefill attention)
c10::OperatorHandle* g_fmla_pf = nullptr;   // _flashmla_C::sparse_prefill_fwd

// per rank: FlashMLA's per-(segment, plan key) tile-scheduler metadata (Python: the FlashMLASchedMeta of c["sched"]),
// filled by a key's first call in the segment, reused by its later layers. Slots are per (bucket, key) (mixed_cxx_attn
// assigns them), and a segment start resets only its own bucket's slots (1d: another bucket's span may still run)
constexpr int kMaxPlans = 64;
struct AttnRank {
  at::Tensor tsm[kMaxPlans], ns[kMaxPlans];
  bool have[kMaxPlans] = {};
  // the plan's owner: the view's si pointer + R. A mismatch drops the plan (replanned here).
  const void* own_p[kMaxPlans] = {};
  int64_t own_r[kMaxPlans] = {};
  uint64_t own_g[kMaxPlans] = {};
  uint64_t mism = 0;
};
AttnRank g_ar[AW_];
// one lock per rank around g_ar: the Python thread's ensure / segment resets and an nd thread's native
// step plan the same slots; an unlocked at::Tensor copy racing its reset can use a freed TensorImpl
std::mutex g_ar_mu[AW_];

at::ScalarType st_of(int32_t d) {
  switch (d) {
    case FPR_AV_U8: return at::kByte;
    case FPR_AV_I32: return at::kInt;
    case FPR_AV_F32: return at::kFloat;
    case FPR_AV_BF16: return at::kBFloat16;
    case FPR_AV_E4M3: return at::kFloat8_e4m3fn;
    default: return at::kLong;
  }
}
at::Tensor view_t(const FprAttnView& v, int dev) {
  return blob(v.ptr, at::IntArrayRef(v.sizes, v.ndim), at::IntArrayRef(v.strides, v.ndim),
                       at::TensorOptions().dtype(st_of(v.dtype)).device(at::Device(at::kCUDA, dev)));
}
c10::IValue opt_t(const FprAttnView& v, int dev) { return v.ptr ? c10::IValue(view_t(v, dev)) : c10::IValue(); }

at::Tensor bf16_2d(void* p, int64_t rows, int64_t cols, int64_t ld, int dev) {
  return blob(p, {rows, cols}, {ld, 1}, at::TensorOptions().dtype(at::kBFloat16).device(at::Device(at::kCUDA, dev)));
}
// idx_a (Stage B): A-role index layers' compressor + scoring (pieces_attn.h FprPieceIdxA)
std::vector<std::unique_ptr<FprPieceIdxA>> g_idx_a;
FprIdxAStep g_idxa_step[4];     // per rank, the A segment's values (attn_set_idxa_step)
FprFn F_select = nullptr;       // idx2f_topk.select_blocks (layer 20's candidates)
FprFn F_dense8 = nullptr;       // idx2f_fcand8.logits_dense (idx2f.LOGITS "f8")
FprFn F_dense_c = nullptr;      // idx2f_cand.logits_dense (idx2f.LOGITS "cand")
FprFn F_pf4 = nullptr;          // idx2f_pf4.logits_prefill (IDX2F_PF)
}  // namespace

// attn.attn_a, mixed branch: torch.mm(a_rows, W["wqkv_a"].t(), out=ws["qakv"]); kd.norm_rope2(...)
extern "C" int piece_attn_a(const FprPieceAttnA* p, const FprStepParams* sp, int layer) {
  (void)sp;
  if (p == nullptr || p->version != FPR_PIECE_ATTN_VERSION || p->layer != layer || F_norm_rope2 == nullptr) return FPR_E_ARG;
  try {
    const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)p->device));
    at::Tensor a = bf16_2d(p->a_rows, p->R, p->k, p->a_ld, p->device);
    at::Tensor w = bf16_2d(p->wqkv, p->n, p->k, p->k, p->device);
    at::Tensor y = bf16_2d(p->qakv, p->R, p->n, p->n, p->device);
    fpr_lt::mm(y, a, w);
  } catch (const std::exception&) {
    if (pieces_throw()) throw;   // FPR_PIECES_THROW=1: the driver / pybind report e.what()
    return FPR_E_CUDA;
  }
  FprAttnDenseNormRope2Args a = p->nr2;
  a.call.stream = p->main;
  return thr_rc(F_norm_rope2(&a), "attn_a norm_rope2");
}

// attn_prefill.attend for the plan's rows (bucket A, ENG_ATTN_PF), then attn_b_attend's `ws["o"][b:].zero_()`:
// per group of the layer's ratio layout: the compressed gather (unless resident: Plan.comp_src), the window gather, the
// combine (unless this (cw, ratio, isrc) was combined this step: Plan.comb_key), FlashMLA's sparse prefill
static int pf_attend(const FprPieceAttnB* p, FprAttnPfStep* ps) {
  const int d = p->device, ratio = p->ratio;
  int li = -1;
  for (int i = 0; i < ps->nlayouts; ++i)
    if (ps->lay[i].ratio == ratio) li = i;
  if (li < 0) return FPR_E_ARG;
  const FprAttnPfLayout& lay = ps->lay[li];
  const int cwi = ratio ? 0 : 1;
  const int64_t cw = ps->cw[cwi], n = ps->n, lo = ps->row_lo;
  const bool ck_ok = ratio == 0 || p->isrc >= 0;          // ckey = (ratio, isrc) when ratio == 0 or isrc is given
  const bool redo_comb = !ck_ok || !ps->ck_valid[cwi] || ps->ck_ratio[cwi] != ratio || ps->ck_isrc[cwi] != p->isrc;
  const auto i32 = at::TensorOptions().dtype(at::kInt).device(at::Device(at::kCUDA, d));
  const auto bf = at::TensorOptions().dtype(at::kBFloat16).device(at::Device(at::kCUDA, d));
  const auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::Device(at::kCUDA, d));
  int rc = FPR_OK;
  try {
    const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)d));
    if (ratio && redo_comb) {   // to_local: the rows' global top-512 slots -> local compressed positions (pl.loc)
      const int64_t rpp = 64 / ratio;
      at::Tensor ti = blob(const_cast<int32_t*>(static_cast<const int32_t*>(p->pf_ti)) + lo * 512, {n, 512}, i32);
      at::Tensor inv = blob(const_cast<void*>(ps->inv), {ps->n_inv}, i32);
      at::Tensor gl = ti.to(at::kLong).clamp_min(0);
      at::Tensor blk = inv.index({gl.div(rpp, "floor")});
      at::Tensor loc = blk.to(at::kLong) * rpp + gl.remainder(rpp);
      at::Tensor out = blob(ps->loc, {n, 512}, i32);
      at::where_out(out, ti.ge(0).logical_and(blk.ge(0)), loc.to(at::kInt), at::full_like(ti, -1));
    }
    if (ps->ws_ratio != ratio) {        // another ratio's layout overwrote the workspace
      ps->ws_ratio = ratio;
      ps->comp_src = -1;
    }
    const bool redo_comp = ratio != 0 && (p->src < 0 || !lay.disjoint || ps->comp_src != p->src);
    for (int gi = 0; gi < lay.ngroups; ++gi) {
      const FprAttnPfGroup& G = lay.g[gi];
      const int64_t nreq = G.b - G.a, M = G.M, N = G.N;
      FprGatherkGatherArgs ga{};
      ga.call = {p->main, d, p->rank};
      ga.out = G.kv, ga.out_sr = M * 512, ga.out_st = 512, ga.num_reqs = nreq;
      if (redo_comp) {
        ga.max_rows = M, ga.k_cache = static_cast<const uint8_t*>(p->pf_kcomp), ga.page_stride = p->pf_kcomp_ps;
        ga.seq_lens = static_cast<const int32_t*>(ps->seq_lens_r[li]) + G.a, ga.gather_lens = nullptr;
        ga.block_table = static_cast<const int32_t*>(ps->bt) + G.a * ps->ld_bt, ga.ld_bt = ps->ld_bt;
        ga.block_size = 64 / ratio, ga.offset = 0;
        if ((rc = thr_rc(F_gather(&ga), "attn_pf comp gather")) != FPR_OK) return rc;
      }
      ga.max_rows = M - N, ga.k_cache = static_cast<const uint8_t*>(p->pf_kswa), ga.page_stride = p->pf_kswa_ps;
      ga.seq_lens = static_cast<const int32_t*>(ps->seq_lens) + G.a;
      ga.gather_lens = static_cast<const int32_t*>(ps->gather_lens) + G.a;
      ga.block_table = static_cast<const int32_t*>(ps->wbt) + G.a * ps->ld_wbt, ga.ld_bt = ps->ld_wbt;
      ga.block_size = 32, ga.offset = N;
      if ((rc = thr_rc(F_gather(&ga), "attn_pf window gather")) != FPR_OK) return rc;
      if (redo_comb) {
        if (G.rec < 0 || G.rec >= ps->nrecs) return FPR_E_ARG;
        const FprTrRecord* r = &ps->rec[G.rec];
        if ((rc = thr_rc(fpr_tr_check(r, G.vals), "attn_pf combine key check")) != FPR_OK) return rc;
        if ((rc = thr_rc(fpr_tr_launch(r, G.vals, (uint32_t)nreq, (uint32_t)G.gy, 1, (CUstream)p->main, 0, 0),
                         "attn_pf combine launch")) != FPR_OK)
          return rc;
      }
      const int64_t q0 = G.q0, nq = G.q1 - G.q0;
      if (nq <= 0) continue;
      torch::jit::Stack st;
      st.reserve(8);
      st.emplace_back(blob(static_cast<uint16_t*>(p->pf_q) + (lo + q0) * 64 * 512, {nq, 64, 512}, bf));
      st.emplace_back(blob(G.kv, {nreq * M, 1, 512}, bf));
      st.emplace_back(blob(static_cast<int32_t*>(ps->comb[cwi]) + q0 * cw, {nq, 1, cw}, {cw, cw, 1}, i32));
      st.emplace_back(p->scale);
      st.emplace_back(int64_t(512));
      st.emplace_back(blob(const_cast<void*>(p->pf_sink), {64}, f32));
      st.emplace_back(blob(static_cast<int32_t*>(ps->clen[cwi]) + q0, {nq}, i32));
      st.emplace_back(blob(static_cast<uint16_t*>(p->pf_o) + (lo + q0) * 64 * 512, {nq, 64, 512}, bf));
      g_fmla_pf->callBoxed(&st);
    }
    if (ratio) ps->comp_src = (lay.disjoint && p->src >= 0) ? p->src : -1;
    ps->ck_valid[cwi] = ck_ok, ps->ck_ratio[cwi] = ratio, ps->ck_isrc[cwi] = p->isrc;
    if (ps->row_hi < p->R)   // attn_b_attend: ws["o"][b:].zero_()
      blob(static_cast<uint16_t*>(p->pf_o) + ps->row_hi * 64 * 512, {p->R - ps->row_hi, 64 * 512}, bf).zero_();
  } catch (const std::exception&) {
    if (pieces_throw()) throw;   // FPR_PIECES_THROW=1: the driver / pybind report e.what()
    return FPR_E_CUDA;
  }
  return FPR_OK;
}

// attn_b_attend's attention: FlashMLA sparse decode (window + top-512) on the layer's plan; pf (bucket A, ENG_ATTN_PF):
// the decode rows [:row_lo] only, then attn_prefill.attend for the plan's rows and the padding rows zeroed
static int attn_core(const FprPieceAttnB* p) {
  if (p->plan_key < 0 || p->plan_key >= kMaxPlans || g_fmla_dec == nullptr) return FPR_E_ARG;
  // FlashMLA sparse decode (window + top-512), the plan of this layer's key; pf: the decode rows [:row_lo] only
  FprAttnPfStep* ps = p->pf ? &g_pf_step[p->rank] : nullptr;
  if (ps != nullptr && (!ps->valid || F_gather == nullptr || g_fmla_pf == nullptr))
    return thr_rc(FPR_E_ARG, "attn_pf step state not valid / gatherk or sparse_prefill missing");
  const int64_t nrows = ps ? ps->row_lo : p->R;
  if (nrows > 0) try {
    const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)p->device));
    const int d = p->device;
    AttnRank& k = g_ar[p->rank];
    const std::lock_guard<std::mutex> lk(g_ar_mu[p->rank]);   // held across the plan read, the call, the store
    const int pk = p->plan_key;
    auto rows = [&](FprAttnView v) {   // q / si / sl / ti / tl / o [:nrows] (row-major views: the first dim only)
      if (v.ptr != nullptr && v.ndim > 0) v.sizes[0] = std::min<int64_t>(v.sizes[0], nrows);
      return v;
    };
    torch::jit::Stack st;
    st.reserve(13);
    st.emplace_back(view_t(rows(p->fq), d));
    st.emplace_back(view_t(p->fk_swa, d));
    st.emplace_back(view_t(rows(p->fsi), d));
    st.emplace_back(opt_t(rows(p->fsl), d));
    st.emplace_back(opt_t(p->fsink, d));
    const uint64_t gen = reinterpret_cast<const AttnBH*>(p)->gen;
    if (k.have[pk] && (k.own_p[pk] != p->fsi.ptr || k.own_r[pk] != p->R || k.own_g[pk] != gen)) {   // another view's plan
      if (++k.mism <= 16 || (k.mism & (k.mism - 1)) == 0)
        std::fprintf(stderr, "pieces_attn: plan slot %d of rank %d made for another view (R %lld, now R %lld, layer %d): replanned (%llu)\n",
                     pk, p->rank, (long long)k.own_r[pk], (long long)p->R, p->layer, (unsigned long long)k.mism);
      k.have[pk] = false;
      k.tsm[pk] = at::Tensor();
      k.ns[pk] = at::Tensor();
    }
    st.emplace_back(k.have[pk] ? c10::IValue(k.tsm[pk]) : c10::IValue());
    st.emplace_back(k.have[pk] ? c10::IValue(k.ns[pk]) : c10::IValue());
    st.emplace_back(p->comp ? opt_t(p->fk_comp, d) : c10::IValue());
    st.emplace_back(p->comp ? opt_t(rows(p->fti), d) : c10::IValue());
    st.emplace_back(p->comp ? opt_t(rows(p->ftl), d) : c10::IValue());
    st.emplace_back(int64_t(512));
    st.emplace_back(p->scale);
    st.emplace_back(view_t(rows(p->fo), d));
    g_fmla_dec->callBoxed(&st);
    // (out, lse, new_tile_scheduler_metadata, new_num_splits): Python stores the last two in the plan object
    if (!k.have[pk]) {
      k.tsm[pk] = st[2].isNone() ? at::Tensor() : st[2].toTensor();
      k.ns[pk] = st[3].isNone() ? at::Tensor() : st[3].toTensor();
      k.have[pk] = true;
      k.own_p[pk] = p->fsi.ptr;
      k.own_r[pk] = p->R;
      k.own_g[pk] = gen;
    }
  } catch (const std::exception&) {
    if (pieces_throw()) throw;   // FPR_PIECES_THROW=1: the driver / pybind report e.what()
    return FPR_E_CUDA;
  }
  if (ps != nullptr) return pf_attend(p, ps);
  return FPR_OK;
}

// attn.attn_b (attn_b_insert + attn_b_attend), the no-pf DENSE8 mixed branch (pieces_attn.h); split in two halves for the
// CP prefill phases
static int attn_b_check(const FprPieceAttnB* p, int layer) {
  if (p == nullptr || p->version != FPR_PIECE_ATTN_VERSION || p->layer != layer || F_w8gemm == nullptr ||
      g_fmla_dec == nullptr || p->plan_key < 0 || p->plan_key >= kMaxPlans)
    return FPR_E_ARG;
  if (p->wo == 1 || p->wo == 2 ? F_inv_rope == nullptr
                               : (p->wo != 0 || F_ropeq == nullptr || F_actq == nullptr || F_dg_gemm == nullptr ||
                                  F_dg_einsum == nullptr))
    return FPR_E_ARG;
  if (p->qdq && F_rope_q_insert == nullptr) return FPR_E_ARG;
  return FPR_OK;
}
// attn.attn_b_insert: q_b + RoPE(q) + the rows' window-KV insert
static int attn_b_ins(const FprPieceAttnB* p) {
  int rc;
  if (p->qdq) {   // ENG_QB_DQ: attn_b_insert's QB_DQ branch on the exact bf16 in the q slot
    if ((rc = dq_launch(p->dq_q, p->main)) != FPR_OK) return rc;
    try {
      const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)p->device));
      at::Tensor x = bf16_2d(const_cast<void*>(p->qb.x), p->R, p->qb.kx, p->qb.x_ld, p->device);
      at::Tensor w = bf16_2d(p->slot_q, p->q_n, p->q_k, p->q_k, p->device);
      at::Tensor y = bf16_2d(p->qb.y, p->R, p->q_n, p->qb.ldy, p->device);
      fpr_lt::mm(y, x, w);
    } catch (const std::exception&) {
      if (pieces_throw()) throw;
      return FPR_E_CUDA;
    }
    FprAttnDenseRopeQInsertArgs ri = p->rqi;
    ri.call.stream = p->main;
    if ((rc = thr_rc(F_rope_q_insert(&ri), "attn_b qdq rope_q_insert")) != FPR_OK) return rc;
  } else {        // q_b + RoPE(q) + window insert (one w8a16 launch)
    FprW8a16GemmArgs qb = p->qb;
    qb.call.stream = p->main;
    if ((rc = thr_rc(F_w8gemm(&qb), "attn_b q_b w8a16")) != FPR_OK) return rc;
  }
  return FPR_OK;
}
// attn.attn_b_attend: FlashMLA (pf: + attn_prefill.attend) -> inverse RoPE -> wo_a -> wo_b
static int attn_b_att(const FprPieceAttnB* p, bool pf = false) {
  int rc;
  if ((rc = attn_core(p)) != FPR_OK) return rc;
  if (p->wo == 2) {   // ENG_WO_W8A16_DQ: kd.inv_rope(o), then _WoW8's exact-dequant route for wo_a / wo_b
    FprAttnDenseInvRopeArgs ir = p->ir;
    ir.call.stream = p->main;
    if ((rc = thr_rc(F_inv_rope(&ir), "attn_b dq inv_rope")) != FPR_OK) return rc;
    if (pf) {   // FPR_ATTN_DQ_PF: dq_a / dq_b ran on the side stream; the rank stream joins here
      if ((rc = dq_pf_join(p)) != FPR_OK) return rc;
    } else if ((rc = dq_launch(p->dq_a, p->main)) != FPR_OK) {
      return rc;
    }
    const int64_t R = p->R;
    const auto bf = at::TensorOptions().dtype(at::kBFloat16).device(at::Device(at::kCUDA, p->device));
    try {
      const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)p->device));
      // = torch.bmm(o.view(R, 8, 4096).transpose(0, 1), w.view(8, 1024, 4096).transpose(1, 2),
      //             out=z.view(R, 8, 1024).transpose(0, 1))
      at::Tensor a = blob(p->wo_o, {8, R, 4096}, {4096, 32768, 1}, bf);
      at::Tensor wn = blob(p->slot_a, {8, 1024, 4096}, {1024 * 4096, 4096, 1}, bf);   // (= wa transposed: w[b] [1024, 4096])
      at::Tensor zt = blob(p->wo_z, {8, R, 1024}, {1024, p->z_ld, 1}, bf);
      fpr_lt::mm(zt, a, wn);                       // zt[b] = a[b] @ wn[b]^T = at::bmm_out(zt, a, wa)
    } catch (const std::exception&) {
      if (pieces_throw()) throw;
      return FPR_E_CUDA;
    }
    if (!pf && (rc = dq_launch(p->dq_b, p->main)) != FPR_OK) return rc;
    try {
      const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)p->device));
      at::Tensor z = bf16_2d(p->wo_z, R, p->b_k, p->z_ld, p->device);          // = torch.mm(z, w.t(), out=y)
      at::Tensor wb = bf16_2d(p->slot_b, p->b_n, p->b_k, p->b_k, p->device);
      at::Tensor y = bf16_2d(p->wo_y, R, p->b_n, p->y_ld, p->device);
      fpr_lt::mm(y, z, wb);
    } catch (const std::exception&) {
      if (pieces_throw()) throw;
      return FPR_E_CUDA;
    }
    return FPR_OK;
  }
  if (p->wo == 1) {   // ENG_WO_W8A16: kd.inv_rope(o), then _WoW8's w8a16 wo_a / wo_b
    FprAttnDenseInvRopeArgs ir = p->ir;
    ir.call.stream = p->main;
    if ((rc = thr_rc(F_inv_rope(&ir), "attn_b wo8 inv_rope")) != FPR_OK) return rc;
    FprW8a16GemmArgs wa = p->woa8;
    wa.call.stream = p->main;
    if ((rc = thr_rc(F_w8gemm(&wa), "attn_b wo8 wo_a w8a16")) != FPR_OK) return rc;
    FprW8a16GemmArgs wb = p->wob8;
    wb.call.stream = p->main;
    return thr_rc(F_w8gemm(&wb), "attn_b wo8 wo_b w8a16");
  }
  // DENSE8 wo_a / wo_b (attn_act8.wo_a / wo_b: the quant applies the inverse RoPE)
  FprRopeq64InvRopeFp8QuantArgs rq = p->rq;
  rq.call.stream = p->main;
  if ((rc = thr_rc(F_ropeq(&rq), "attn_b DENSE8 ropeq64")) != FPR_OK) return rc;
  FprDgEinsumArgs woa = p->woa;
  woa.call.stream = p->main;
  if ((rc = thr_rc(F_dg_einsum(&woa), "attn_b DENSE8 wo_a einsum")) != FPR_OK) return rc;
  FprFp8LinearActQuant128Args aq = p->aq;
  aq.call.stream = p->main;
  if ((rc = thr_rc(F_actq(&aq), "attn_b DENSE8 act_quant")) != FPR_OK) return rc;
  FprDgGemmArgs wob = p->wob;
  wob.call.stream = p->main;
  return thr_rc(F_dg_gemm(&wob), "attn_b DENSE8 wo_b gemm");
}
extern "C" int piece_attn_b(const FprPieceAttnB* p, const FprStepParams* sp, int layer) {
  (void)sp;
  int rc = attn_b_check(p, layer);
  if (rc != FPR_OK) return rc;
  const bool pf = dq_pf_usable(p);       // FPR_ATTN_DQ_PF=1
  if (pf && (rc = dq_pf_fork(p)) != FPR_OK) {
    dq_pf_join(p);                       // (a partial fork: never leave side-stream slot writes pending)
    return rc;
  }
  if ((rc = attn_b_ins(p)) != FPR_OK || (rc = attn_b_att(p, pf)) != FPR_OK) {
    if (pf) dq_pf_join(p);               // error after the fork: the side stream may still be writing slots a / b
    return rc;
  }
  return FPR_OK;
}

// hooks_idx._score_general, T-role consumer layer (pieces_attn.h FprPieceAttnIdx)
static int idx_scatter_empty(const FprPieceAttnIdx* p, int d) {
  blob(p->ti, {p->R * 512}, at::TensorOptions().dtype(at::kInt).device(at::Device(at::kCUDA, d))).fill_(-1);
  blob(p->tl, {p->R}, at::TensorOptions().dtype(at::kInt).device(at::Device(at::kCUDA, d))).zero_();
  return FPR_OK;
}

extern "C" int piece_attn_idx(const FprPieceAttnIdx* p, FprAttnIdxStep* st, int layer) {
  if (p == nullptr || st == nullptr || p->version != FPR_PIECE_ATTN_VERSION || p->layer != layer || !st->valid ||
      F_prep2 == nullptr || F_bmax == nullptr || F_topk == nullptr || F_cand_logits == nullptr || F_cand_prep == nullptr)
    return FPR_E_ARG;
  const int d = p->device;
  const auto i32 = at::TensorOptions().dtype(at::kInt).device(at::Device(at::kCUDA, d));
  const auto i64 = at::TensorOptions().dtype(at::kLong).device(at::Device(at::kCUDA, d));
  const auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::Device(at::kCUDA, d));
  const auto u8 = at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA, d));
  int rc = FPR_OK;
  try {
    const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)d));
    if (st->idle || st->skip[p->lidx]) return idx_scatter_empty(p, d);
    // 1. projections (F.linear without bias = mm with the weight's transpose)
    at::Tensor wo = bf16_2d(p->wbuf, p->R, 32, 32, d), qo = bf16_2d(p->qbuf, p->R, 4096, 4096, d);
    fpr_lt::mm(wo, bf16_2d(p->x, p->R, p->kx, p->x_ld, d), bf16_2d(p->wp, 32, p->kx, p->kx, d));
    fpr_lt::mm(qo, bf16_2d(p->qr, p->R, p->kq, p->qr_ld, d), bf16_2d(p->wqb, 4096, p->kq, p->kq, d));
    // 2. the CuTe q-quant (captured once on these buffers)
    if ((rc = thr_rc(fpr_kc_launch(&p->qq, (CUstream)p->main), "idx q-quant replay")) != FPR_OK) return rc;
    // 3. the indexer rows' q / weights
    const int64_t IR = st->IR;
    at::Tensor ir = blob(p->ir2row, {IR}, i64);
    at::Tensor q8i = blob(p->q8i, {IR, 32 * 128}, u8), woi = blob(p->woi, {IR, 32}, f32);
    at::index_select_out(q8i, blob(p->q8buf, {p->R, 32 * 128}, u8), 0, ir);
    at::index_select_out(woi, blob(p->wobuf, {p->R, 32}, f32), 0, ir);
  } catch (const std::exception&) {
    if (pieces_throw()) throw;   // FPR_PIECES_THROW=1: the driver / pybind report e.what()
    return FPR_E_CUDA;
  }
  // 4. OwnedIndexer._consumer_ws: chunks of ws.cb pseudo-requests
  // _consumer_ws: one chunk reuses the prep a consumer of this segment computed (OwnedIndexer.prep), except at 24
  const bool reuse = st->single && p->layer != 24 && st->prep_done;
  for (int c = 0; c < st->nchunks; ++c) {
    const FprAttnIdxChunk& k = st->ch[c];
    const int64_t nb = k.b1 - k.b0, rows = 4 * nb, r0 = 4 * k.b0;
    const int32_t* sl = static_cast<const int32_t*>(p->seq_lens1) + r0;
    const int32_t* bt = static_cast<const int32_t*>(p->bt) + k.b0 * p->ld_bt;
    if (!reuse) {
      if (st->use2) {
        FprIdx2fTopkPrep2Args a{};
        a.call = {p->main, d, p->rank};
        a.cand = static_cast<const int32_t*>(p->cand) + r0 * 2048;
        a.ld_cand = 2048, a.lens = sl, a.rows = rows;
        a.ublk = (int32_t*)k.ublk, a.uslot = (int32_t*)k.uslot, a.nunion = (int32_t*)k.nunion, a.clen = (int32_t*)k.clen;
        a.block_table = bt, a.ld_bt = p->ld_bt, a.upk = (int32_t*)k.upk;
        if ((rc = thr_rc(F_prep2(&a), "idx prep2")) != FPR_OK) return rc;
      } else {
        FprIdx2fCandPrepArgs a{};
        a.call = {p->main, d, p->rank};
        a.cand = static_cast<const int32_t*>(p->cand) + r0 * 2048;
        a.ld_cand = 2048, a.rows = rows, a.k = 2048, a.ctx = sl, a.next_n = 4, a.nwords_max = st->nwords;
        a.ublk = (int32_t*)k.ublk, a.uslot = (int32_t*)k.uslot, a.nunion = (int32_t*)k.nunion;
        a.cblk = (int32_t*)k.cblk, a.ld_cblk = k.ld_cblk, a.clen = (int32_t*)k.clen;
        if ((rc = thr_rc(F_cand_prep(&a), "idx cand_prep")) != FPR_OK) return rc;
      }
      if (st->single) st->prep_done = 1;
    }
    FprIdx2fFcand8LogitsArgs lg{};
    lg.call = {p->main, d, p->rank};
    lg.q = static_cast<const uint8_t*>(p->q8i) + r0 * 32 * 128;
    lg.b = nb, lg.next_n = 4;
    lg.kv = static_cast<const uint8_t*>(p->kv), lg.num_pages = p->num_pages, lg.page_stride = p->page_stride;
    lg.weights = static_cast<const float*>(p->woi) + r0 * 32;
    lg.ctx = sl, lg.block_table = bt, lg.ld_bt = p->ld_bt;
    lg.ublk = (const int32_t*)k.ublk, lg.uslot = (const int32_t*)k.uslot, lg.nunion = (const int32_t*)k.nunion;
    lg.ustride = 4 * 2048;
    lg.out = (float*)k.comp, lg.ld_out = 2048 * 8, lg.out_cols = 2048 * 8;
    lg.upk = st->use2 ? (const int32_t*)k.upk : nullptr;
    if ((rc = thr_rc(F_cand_logits(&lg), "idx cand logits")) != FPR_OK) return rc;
    for (int64_t q0 = 0; q0 < rows; q0 += 1024) {   // idx2f.block_max: <= 1,024 rows per call
      const int64_t q1 = std::min<int64_t>(rows, q0 + 1024);
      FprIdx2fTopkBlockMaxArgs bm{};
      bm.call = {p->main, d, p->rank};
      bm.logits = (const float*)k.comp + q0 * 16384, bm.ld_logits = 16384;
      bm.lens = (const int32_t*)k.clen + q0, bm.rows = q1 - q0;
      bm.out = (float*)k.bm + q0 * 2048, bm.ld_out = 2048, bm.max_len = 16384;
      if ((rc = thr_rc(F_bmax(&bm), "idx block_max")) != FPR_OK) return rc;
    }
    FprIdx2fTopkTopkArgs tk{};
    tk.call = {p->main, d, p->rank};
    tk.logits = (const float*)k.comp, tk.ld_logits = 16384, tk.lens = (const int32_t*)k.clen;
    tk.bmax = (const float*)k.bm, tk.ld_bmax = 2048, tk.nblocks = 2048;
    tk.out = static_cast<int32_t*>(p->topk_out) + r0 * 512, tk.ld_out = 512;
    tk.cblk = (const int32_t*)k.cblk, tk.ld_cblk = k.ld_cblk, tk.k = 512, tk.slow = (int32_t*)p->slow, tk.stop = 0;
    tk.rows = rows;
    if ((rc = thr_rc(F_topk(&tk), "idx topk")) != FPR_OK) return rc;
  }
  // 5. global slots / lengths (Triton replay), 6. the scatter into the bucket's ti / tl
  if ((rc = thr_rc(fpr_tr_check(&p->gt, p->gt_vals), "idx global_topk key check")) != FPR_OK) return rc;
  if ((rc = thr_rc(fpr_tr_launch(&p->gt, p->gt_vals, (uint32_t)st->IR, 1, 1, (CUstream)p->main, 0, 0),
                   "idx global_topk launch")) != FPR_OK)
    return rc;
  try {
    const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)d));
    idx_scatter_empty(p, d);
    const int64_t nv = st->nv;
    at::Tensor vir = blob(p->vir, {nv}, i64), vrow = blob(p->vrow, {nv}, i64);
    const int64_t IR = st->IR;
    blob(p->ti, {p->R, 512}, i32).index_copy_(0, vrow, blob(p->gbuf, {IR, 512}, i32).index_select(0, vir));
    blob(p->tl, {p->R}, i32).index_copy_(0, vrow, blob(p->lnbuf, {IR}, i32).index_select(0, vir));
  } catch (const std::exception&) {
    if (pieces_throw()) throw;   // FPR_PIECES_THROW=1: the driver / pybind report e.what()
    return FPR_E_CUDA;
  }
  return FPR_OK;
}

// idx_a (Stage B): hooks_idx._compress_general / _score_general, A-role KV-source layer (pieces_attn.h FprPieceIdxA)
static int idxa_tr(const FprTrRecord* rec, const uint64_t* vals, int nv, int64_t patch_i, uint64_t patch_v, uint32_t grid,
                   cudaStream_t s, const char* what) {
  uint64_t v[FPR_IDXA_MAXV];
  if (nv < 0 || nv > FPR_IDXA_MAXV) return thr_rc(FPR_E_ARG, what);
  std::memcpy(v, vals, sizeof(uint64_t) * (size_t)nv);
  if (patch_i >= 0) {
    if (patch_i >= nv) return thr_rc(FPR_E_ARG, what);
    v[patch_i] = patch_v;
  }
  int rc = thr_rc(fpr_tr_check(rec, v), what);
  if (rc != FPR_OK) return rc;
  return thr_rc(fpr_tr_launch(rec, v, grid, 1, 1, (CUstream)s, 0, 0), what);
}

static int idxa_compress(const FprPieceIdxA* p, const FprIdxAStep* st, int d) {
  if (st->idle) return FPR_OK;                       // (_compress_general: an idle rank compresses nothing)
  const auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::Device(at::kCUDA, d));
  try {
    const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)d));
    // ks = torch.mm(x, comp_w.T, out_dtype=torch.float32)
    at::Tensor ks = blob(p->ks, {p->R, p->ncomp}, f32);
    fpr_lt::mm(ks, bf16_2d(p->x, p->R, p->kx, p->x_ld, d), bf16_2d(p->comp_w, p->ncomp, p->kx, p->kx, d));
  } catch (const std::exception&) {
    if (pieces_throw()) throw;
    return FPR_E_CUDA;
  }
  int rc;
  // fused_save_compress_norm: grid num_reqs + ceil(num_tokens / ratio) (num_reqs 0 at ratio 1)
  const uint32_t g1 = (uint32_t)((p->ratio == 2 ? st->nreq : 0) + (p->R + p->ratio - 1) / p->ratio);
  if ((rc = idxa_tr(&p->fscn, p->fscn_vals, p->fscn_nv, p->fscn_nreq, (uint64_t)st->nreq, g1, p->main,
                    "idx_a fused_save_compress_norm")) != FPR_OK)
    return rc;
  if ((rc = idxa_tr(&p->rqi, p->rqi_vals, p->rqi_nv, -1, 0, (uint32_t)p->R, p->main, "idx_a rope_quant_insert")) != FPR_OK)
    return rc;
  try {
    const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)d));
    at::Tensor kpre = bf16_2d(p->kpre, p->R, 128, 128, d);   // kpre = F.linear(lat, idx_wk)
    fpr_lt::mm(kpre, bf16_2d(p->lat, p->R, 512, 512, d), bf16_2d(p->idx_wk, 128, 512, 512, d));
  } catch (const std::exception&) {
    if (pieces_throw()) throw;
    return FPR_E_CUDA;
  }
  return idxa_tr(&p->iks, p->iks_vals, p->iks_nv, -1, 0, (uint32_t)p->R, p->main, "idx_a indexer_k_norm_rope_store");
}

static int idxa_scatter_empty(const FprPieceIdxA* p, int d) {
  blob(p->ti, {p->R * 512}, at::TensorOptions().dtype(at::kInt).device(at::Device(at::kCUDA, d))).fill_(-1);
  blob(p->tl, {p->R}, at::TensorOptions().dtype(at::kInt).device(at::Device(at::kCUDA, d))).zero_();
  return FPR_OK;
}

// idx2f.OwnedIndexer._dense_sliced's long-row slice body (fast False: rows > LONG_ROW), after the slice's logits lg
// [rows, w]: [layer 20] select_candidates_stock(lg, sl, cand[r0:r1]) = vLLM select_candidate_blocks(lg, None, sl.reshape(-1),
// 2048, 8, out, 1): its _block_scores_kernel -> torch.topk -> _store_candidates_kernel (the Triton kernels replayed from
// the build-time recording, the topk through ATen); stock_topk(lg, sl, out[r0:r1], max_seq_len, production_backend(
// global_rows)) = torch.ops._C.persistent_topk, or cooperative_topk when SparseIndexerTopk("auto") resolves to it (global
// rows <= 64 and slice rows <= 64; stride(0) % 4 == 0 always: w is a multiple of 256); canon_topk when IDX2F_CANON_STOCK.
// The same calls in the same order on the same stream: bit for bit with Python's slice.
static int idxa_stock(const FprPieceIdxA* p, const FprIdxAStep* st, float* lg, int64_t rows, int64_t w, const int32_t* lens,
                      int64_t r0, int d) {
  const auto i32 = at::TensorOptions().dtype(at::kInt).device(at::Device(at::kCUDA, d));
  const auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::Device(at::kCUDA, d));
  const auto u8 = at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA, d));
  if (!p->stock_ok || p->topk_ws == nullptr || rows % 4 != 0) return thr_rc(FPR_E_ARG, "idx_a stock branch not built");
  int rc = FPR_OK;
  try {
    const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)d));
    at::Tensor lgt = blob(lg, {rows, w}, f32);
    at::Tensor sl = blob(const_cast<int32_t*>(lens), {rows / 4, 4}, i32);
    if (p->layer == 20) {
      const int64_t nb = (w + 7) / 8;
      if (p->bs_nv <= 0 || p->bs_nv > 16 || p->sc_nv <= 0 || p->sc_nv > 16) return thr_rc(FPR_E_ARG, "idx_a stock select records");
      at::Tensor scores = at::empty({rows, nb}, f32);   // (= logits.new_empty: the current stream's pool)
      uint64_t v[16];
      std::memcpy(v, p->bs_vals, sizeof(uint64_t) * (size_t)p->bs_nv);
      v[p->bs_i[0]] = (uint64_t)(uintptr_t)lg, v[p->bs_i[1]] = (uint64_t)(uintptr_t)lens;
      v[p->bs_i[2]] = (uint64_t)(uintptr_t)scores.data_ptr(), v[p->bs_i[3]] = (uint64_t)w;
      v[p->bs_i[4]] = (uint64_t)w, v[p->bs_i[5]] = (uint64_t)nb;
      if ((rc = thr_rc(fpr_tr_check(&p->bs, v), "idx_a stock block_scores key check")) != FPR_OK) return rc;
      if ((rc = thr_rc(fpr_tr_launch(&p->bs, v, (uint32_t)rows, (uint32_t)((nb + 127) / 128), 1, (CUstream)p->main, 0, 0),
                       "idx_a stock block_scores launch")) != FPR_OK)
        return rc;
      const auto top = at::topk(scores, std::min<int64_t>(2048, nb), -1, true, true);
      const at::Tensor& tv = std::get<0>(top);
      const at::Tensor& ti = std::get<1>(top);
      std::memcpy(v, p->sc_vals, sizeof(uint64_t) * (size_t)p->sc_nv);
      v[p->sc_i[0]] = (uint64_t)(uintptr_t)tv.data_ptr(), v[p->sc_i[1]] = (uint64_t)(uintptr_t)ti.data_ptr();
      v[p->sc_i[2]] = (uint64_t)(uintptr_t)(static_cast<int32_t*>(p->cand) + r0 * 2048), v[p->sc_i[3]] = (uint64_t)tv.size(1);
      if ((rc = thr_rc(fpr_tr_check(&p->sc, v), "idx_a stock store_candidates key check")) != FPR_OK) return rc;
      if ((rc = thr_rc(fpr_tr_launch(&p->sc, v, (uint32_t)rows, 8, 1, (CUstream)p->main, 0, 0),
                       "idx_a stock store_candidates launch")) != FPR_OK)
        return rc;
    }
    static const c10::OperatorHandle op_p = c10::Dispatcher::singleton().findSchemaOrThrow("_C::persistent_topk", "");
    static const c10::OperatorHandle op_c = c10::Dispatcher::singleton().findSchemaOrThrow("_C::cooperative_topk", "");
    const bool coop = st->global_rows <= 64 && rows <= 64;
    at::Tensor outk = blob(static_cast<int32_t*>(p->topk_out) + r0 * 512, {rows, 512}, i32);
    torch::jit::Stack sk;
    sk.reserve(6);
    sk.emplace_back(lgt);
    sk.emplace_back(sl);
    sk.emplace_back(outk);
    sk.emplace_back(blob(p->topk_ws, {p->topk_ws_n}, u8));
    sk.emplace_back(int64_t(512));
    sk.emplace_back(int64_t(st->max_seq_len));
    (coop ? op_c : op_p).callBoxed(&sk);
    if (p->canon) {                                   // idx2f.canon_topk
      const int32_t big = std::numeric_limits<int32_t>::max();
      at::Tensor key = at::where(outk >= 0, outk, at::full_like(outk, big));
      at::Tensor srt = std::get<0>(key.sort(1));
      outk.copy_(at::where(srt == big, at::full_like(srt, -1), srt));
    }
  } catch (const std::exception&) {
    if (pieces_throw()) throw;
    return FPR_E_CUDA;
  }
  return FPR_OK;
}

static int idxa_score(const FprPieceIdxA* p, const FprIdxAStep* st, int d) {
  const auto i32 = at::TensorOptions().dtype(at::kInt).device(at::Device(at::kCUDA, d));
  const auto i64 = at::TensorOptions().dtype(at::kLong).device(at::Device(at::kCUDA, d));
  const auto f32 = at::TensorOptions().dtype(at::kFloat).device(at::Device(at::kCUDA, d));
  const auto u8 = at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA, d));
  int rc = FPR_OK;
  const int64_t IR = st->IR;
  try {
    const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)d));
    if (st->idle || st->skip[p->lidx]) return idxa_scatter_empty(p, d);
    at::Tensor wo = bf16_2d(p->wbuf, p->R, 32, 32, d), qo = bf16_2d(p->qbuf, p->R, 4096, 4096, d);
    fpr_lt::mm(wo, bf16_2d(p->x, p->R, p->kx, p->x_ld, d), bf16_2d(p->wp, 32, p->kx, p->kx, d));
    fpr_lt::mm(qo, bf16_2d(p->qr, p->R, p->kq, p->qr_ld, d), bf16_2d(p->wqb, 4096, p->kq, p->kq, d));
    if ((rc = thr_rc(fpr_kc_launch(&p->qq, (CUstream)p->main), "idx_a q-quant replay")) != FPR_OK) return rc;
    at::Tensor ir = blob(p->ir2row, {IR}, i64);
    at::Tensor q8i = blob(p->q8i, {IR, 32 * 128}, u8), woi = blob(p->woi, {IR, 32}, f32);
    at::index_select_out(q8i, blob(p->q8buf, {p->R, 32 * 128}, u8), 0, ir);
    at::index_select_out(woi, blob(p->wobuf, {p->R, 32}, f32), 0, ir);
  } catch (const std::exception&) {
    if (pieces_throw()) throw;
    return FPR_E_CUDA;
  }
  // OwnedIndexer._dense_sliced (fast, lean + EagerWorkspace): slices of whole pseudo-requests
  const FprIdxASlice* S = st->sl[p->sidx];
  const int ns = st->nsl[p->sidx];
  const int64_t ml = st->max_len[p->lidx], w = st->width[p->lidx], nb = std::max<int64_t>((ml + 7) / 8, 1);
  const bool stock = st->refuse[p->lidx] == 2;      // rows > LONG_ROW: the stock branch (no block maxima)
  if (ns < 0 || ns > FPR_IDXA_MAX_SLICES) return thr_rc(FPR_E_ARG, "idx_a slices");
  float* lg = static_cast<float*>(p->ws_f32);
  float* bm = static_cast<float*>(p->ws_bm);
  for (int i = 0; i < ns; ++i) {
    const FprIdxASlice& k = S[i];
    const int64_t b = k.b1 - k.b0, r0 = 4 * k.b0, rows = 4 * b;
    if (b <= 0 || b > 1024 || k.b1 > st->B || rows * w > p->ws_f32_n || (!stock && rows * nb > p->ws_bm_n))
      return thr_rc(FPR_E_ARG, "idx_a slice bounds");
    const int32_t* lens = static_cast<const int32_t*>(p->seq_lens) + r0;
    const int32_t* bt = static_cast<const int32_t*>(p->bt) + k.b0 * p->ld_bt;
    const uint8_t* q = static_cast<const uint8_t*>(p->q8i) + r0 * 32 * 128;
    const float* wts = static_cast<const float*>(p->woi) + r0 * 32;
    if (k.gst_off >= 0) {                             // ext_pf().logits_prefill (the shared-key prefill logits)
      if (F_pf4 == nullptr || st->gst == nullptr) return thr_rc(FPR_E_ARG, "idx_a pf4 table");
      FprIdx2fPf4LogitsPrefillArgs a{};
      a.call = {p->main, d, p->rank};
      a.q = q, a.b = b, a.kv = static_cast<const uint8_t*>(p->kv), a.num_pages = p->num_pages;
      a.page_stride = p->page_stride, a.page_tok = p->page_tok, a.weights = wts, a.ctx = lens, a.block_table = bt;
      a.ld_bt = p->ld_bt, a.bt_cols = p->bt_cols, a.gstart = st->gst + k.gst_off, a.g = k.g, a.out = lg, a.ld_out = w, a.abl = 0;
      if ((rc = thr_rc(F_pf4(&a), "idx_a pf4 logits_prefill")) != FPR_OK) return rc;
    } else if (st->logits == 1) {                     // dense_logits: ext_logits() = idx2f_fcand8
      if (F_dense8 == nullptr) return thr_rc(FPR_E_ARG, "idx_a fcand8 table");
      FprIdx2fFcand8LogitsDenseArgs a{};
      a.call = {p->main, d, p->rank};
      a.q = q, a.b = b, a.next_n = 4, a.kv = static_cast<const uint8_t*>(p->kv), a.num_pages = p->num_pages;
      a.page_stride = p->page_stride, a.page_tok = p->page_tok, a.weights = wts, a.ctx = lens, a.block_table = bt;
      a.ld_bt = p->ld_bt, a.bt_cols = p->bt_cols, a.out = lg, a.ld_out = w;
      if ((rc = thr_rc(F_dense8(&a), "idx_a fcand8 logits_dense")) != FPR_OK) return rc;
    } else if (st->logits == 2) {                     // dense_logits: ext_logits() = idx2f_cand
      if (F_dense_c == nullptr) return thr_rc(FPR_E_ARG, "idx_a cand table");
      FprIdx2fCandLogitsDenseArgs a{};
      a.call = {p->main, d, p->rank};
      a.q = q, a.b = b, a.next_n = 4, a.kv = static_cast<const uint8_t*>(p->kv), a.num_pages = p->num_pages;
      a.page_stride = p->page_stride, a.page_tok = p->page_tok, a.weights = wts, a.ctx = lens, a.block_table = bt;
      a.ld_bt = p->ld_bt, a.bt_cols = p->bt_cols, a.out = lg, a.ld_out = w;
      if ((rc = thr_rc(F_dense_c(&a), "idx_a cand logits_dense")) != FPR_OK) return rc;
    } else {
      return thr_rc(FPR_E_ARG, "idx_a logits kind");
    }
    if (stock) {
      if ((rc = idxa_stock(p, st, lg, rows, w, lens, r0, d)) != FPR_OK) return rc;
      continue;
    }
    for (int64_t q0 = 0; q0 < rows; q0 += 1024) {    // idx2f.block_max: <= 1,024 rows per call
      const int64_t q1 = std::min<int64_t>(rows, q0 + 1024);
      FprIdx2fTopkBlockMaxArgs a{};
      a.call = {p->main, d, p->rank};
      a.logits = lg + q0 * w, a.ld_logits = w, a.lens = lens + q0, a.rows = q1 - q0;
      a.out = bm + q0 * nb, a.ld_out = nb, a.max_len = ml;
      if ((rc = thr_rc(F_bmax(&a), "idx_a block_max")) != FPR_OK) return rc;
    }
    if (p->layer == 20) {                             // select_candidates_bmax(bm, nb_lens[r0:r1], cand[r0:r1])
      FprIdx2fTopkSelectBlocksArgs a{};
      a.call = {p->main, d, p->rank};
      a.bmax = bm, a.ld_bmax = nb, a.nblocks = nb, a.nb_lens = static_cast<const int32_t*>(p->nb_lens) + r0;
      a.out = static_cast<int32_t*>(p->cand) + r0 * 2048, a.ld_out = 2048, a.k = 2048, a.rows = rows;
      if ((rc = thr_rc(F_select(&a), "idx_a select_blocks")) != FPR_OK) return rc;
    }
    FprIdx2fTopkTopkArgs a{};                         // exact_topk(lg, lens, bm, out[r0:r1], slow)
    a.call = {p->main, d, p->rank};
    a.logits = lg, a.ld_logits = w, a.lens = lens, a.bmax = bm, a.ld_bmax = nb, a.nblocks = nb;
    a.out = static_cast<int32_t*>(p->topk_out) + r0 * 512, a.ld_out = 512, a.cblk = nullptr, a.ld_cblk = 0, a.k = 512;
    a.slow = static_cast<int32_t*>(p->slow), a.stop = 0, a.rows = rows;
    if ((rc = thr_rc(F_topk(&a), "idx_a topk")) != FPR_OK) return rc;
  }
  if ((rc = thr_rc(fpr_tr_check(&p->gt, p->gt_vals), "idx_a global_topk key check")) != FPR_OK) return rc;
  if ((rc = thr_rc(fpr_tr_launch(&p->gt, p->gt_vals, (uint32_t)IR, 1, 1, (CUstream)p->main, 0, 0),
                   "idx_a global_topk launch")) != FPR_OK)
    return rc;
  try {
    const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(p->main, (c10::DeviceIndex)d));
    idxa_scatter_empty(p, d);
    const int64_t nv = st->nv;
    at::Tensor vir = blob(p->vir, {nv}, i64), vrow = blob(p->vrow, {nv}, i64);
    blob(p->ti, {p->R, 512}, i32).index_copy_(0, vrow, blob(p->gbuf, {IR, 512}, i32).index_select(0, vir));
    blob(p->tl, {p->R}, i32).index_copy_(0, vrow, blob(p->lnbuf, {IR}, i32).index_select(0, vir));
  } catch (const std::exception&) {
    if (pieces_throw()) throw;
    return FPR_E_CUDA;
  }
  return FPR_OK;
}

extern "C" int piece_idx_a(const FprPieceIdxA* p, const FprIdxAStep* st, int layer, int what) {
  if (p == nullptr || st == nullptr || p->version != FPR_PIECE_ATTN_VERSION || p->layer != layer || !st->valid ||
      p->lidx < 0 || p->lidx > 3 || st->refuse[p->lidx] == 1 || (st->refuse[p->lidx] == 2 && !p->stock_ok) ||
      st->refuse[p->lidx] < 0 || st->refuse[p->lidx] > 2 || F_bmax == nullptr || F_topk == nullptr || F_select == nullptr ||
      what < 1 || what > 3)
    return thr_rc(FPR_E_ARG, "idx_a arguments / step");
  int rc = FPR_OK;
  if ((what & 1) && (rc = idxa_compress(p, st, p->device)) != FPR_OK) return rc;
  if (what & 2) rc = idxa_score(p, st, p->device);
  return rc;
}

// pieces_attn.h: the shared fixed-plan FlashMLA sparse decode
extern "C" int fpr_attn_flashmla_decode(const FprAttnFlashMlaDecodeArgs* a) {
  if (a == nullptr || a->q.ptr == nullptr || a->kv.ptr == nullptr || a->indices.ptr == nullptr || a->out.ptr == nullptr ||
      a->tsm.ptr == nullptr || a->ns.ptr == nullptr)
    return FPR_E_ARG;
  try {
    static const c10::OperatorHandle op =
        c10::Dispatcher::singleton().findSchemaOrThrow("_flashmla_C::sparse_decode_fwd", "");   // (thread-safe init)
    const int d = a->call.device;
    const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(a->call.stream, (c10::DeviceIndex)d));
    torch::jit::Stack st;
    st.reserve(13);
    st.emplace_back(view_t(a->q, d));
    st.emplace_back(view_t(a->kv, d));
    st.emplace_back(view_t(a->indices, d));
    st.emplace_back(opt_t(a->topk_length, d));
    st.emplace_back(opt_t(a->attn_sink, d));
    st.emplace_back(view_t(a->tsm, d));
    st.emplace_back(view_t(a->ns, d));
    st.emplace_back(opt_t(a->extra_kv, d));
    st.emplace_back(opt_t(a->extra_indices, d));
    st.emplace_back(opt_t(a->extra_topk_length, d));
    st.emplace_back(a->d_v);
    st.emplace_back(a->sm_scale);
    st.emplace_back(view_t(a->out, d));
    op.callBoxed(&st);
  } catch (const std::exception&) {
    if (pieces_throw()) throw;   // FPR_PIECES_THROW=1: the driver / pybind report e.what()
    return FPR_E_CUDA;
  }
  return FPR_OK;
}

namespace {
// per rank: resolve the attention extensions' functions (their tables registered first: mixed_cxx_attn.setup)
void attn_init_rank(int r, int device) {
  TORCH_CHECK(r >= 0 && r < AW_, "pieces_attn: rank");
  int cur = -1;
  C10_CUDA_CHECK(cudaGetDevice(&cur));
  TORCH_CHECK(cur == device, "pieces_attn: attn_init_rank(", r, ", ", device, ") with device ", cur, " current");
  F_norm_rope2 = fpr_drv_fn("attn_dense", "attn_dense.norm_rope2", (int64_t)sizeof(FprAttnDenseNormRope2Args));
  TORCH_CHECK(F_norm_rope2 != nullptr, "pieces_attn: the attn_dense table is not registered");
  // attn_b's (optional: absent tables leave attn_b on Python)
  F_w8gemm = fpr_drv_fn("w8a16", "w8a16.gemm", (int64_t)sizeof(FprW8a16GemmArgs));
  F_inv_rope = fpr_drv_fn("attn_dense", "attn_dense.inv_rope", (int64_t)sizeof(FprAttnDenseInvRopeArgs));
  F_rope_q_insert = fpr_drv_fn("attn_dense", "attn_dense.rope_q_insert", (int64_t)sizeof(FprAttnDenseRopeQInsertArgs));
  F_ropeq = fpr_drv_fn("ropeq64", "ropeq64.inv_rope_fp8_quant", (int64_t)sizeof(FprRopeq64InvRopeFp8QuantArgs));
  F_actq = fpr_drv_fn("fp8_linear", "fp8_linear.act_quant_128", (int64_t)sizeof(FprFp8LinearActQuant128Args));
  F_dg_gemm = fpr_drv_fn("dg", "dg.fp8_gemm_nt", (int64_t)sizeof(FprDgGemmArgs));
  F_dg_einsum = fpr_drv_fn("dg", "dg.fp8_einsum", (int64_t)sizeof(FprDgEinsumArgs));
  F_prep2 = fpr_drv_fn("idx2f_topk", "idx2f_topk.prep2", (int64_t)sizeof(FprIdx2fTopkPrep2Args));
  F_bmax = fpr_drv_fn("idx2f_topk", "idx2f_topk.block_max", (int64_t)sizeof(FprIdx2fTopkBlockMaxArgs));
  F_topk = fpr_drv_fn("idx2f_topk", "idx2f_topk.topk", (int64_t)sizeof(FprIdx2fTopkTopkArgs));
  F_cand_logits = fpr_drv_fn("idx2f_fcand8", "idx2f_fcand8.logits", (int64_t)sizeof(FprIdx2fFcand8LogitsArgs));
  F_cand_prep = fpr_drv_fn("idx2f_cand", "idx2f_cand.prep", (int64_t)sizeof(FprIdx2fCandPrepArgs));
  F_select = fpr_drv_fn("idx2f_topk", "idx2f_topk.select_blocks", (int64_t)sizeof(FprIdx2fTopkSelectBlocksArgs));
  F_dense8 = fpr_drv_fn("idx2f_fcand8", "idx2f_fcand8.logits_dense", (int64_t)sizeof(FprIdx2fFcand8LogitsDenseArgs));
  F_dense_c = fpr_drv_fn("idx2f_cand", "idx2f_cand.logits_dense", (int64_t)sizeof(FprIdx2fCandLogitsDenseArgs));
  F_pf4 = fpr_drv_fn("idx2f_pf4", "idx2f_pf4.logits_prefill", (int64_t)sizeof(FprIdx2fPf4LogitsPrefillArgs));
  if (F_pf4) {   // (Stage B idx_a; optional: without it every slice takes logits_dense, so pf steps refuse)
    FprFn init = fpr_drv_fn("idx2f_pf4", "idx2f_pf4.init", (int64_t)sizeof(FprIdx2fPf4InitArgs));
    FprIdx2fPf4InitArgs ia{{nullptr, device, r}};
    TORCH_CHECK(init != nullptr && init(&ia) == FPR_OK, "pieces_attn: idx2f_pf4.init");
  }
  for (const char* e : {"idx2f_topk", "idx2f_fcand8", "idx2f_cand"}) {   // their one-time state on this device
    FprFn init = fpr_drv_fn(e, (std::string(e) + ".init").c_str(), (int64_t)sizeof(FprIdx2fTopkInitArgs));
    if (init) {
      FprIdx2fTopkInitArgs ia{{nullptr, device, r}};
      TORCH_CHECK(init(&ia) == FPR_OK, "pieces_attn: ", e, ".init");
    }
  }
  if (g_fmla_dec == nullptr)
    g_fmla_dec = new c10::OperatorHandle(c10::Dispatcher::singleton().findSchemaOrThrow("_flashmla_C::sparse_decode_fwd", ""));
  F_gather = fpr_drv_fn("gatherk", "gatherk.gather", (int64_t)sizeof(FprGatherkGatherArgs));   // (stage 1c-1; optional)
  if (g_fmla_pf == nullptr) {
    auto h = c10::Dispatcher::singleton().findSchema({"_flashmla_C::sparse_prefill_fwd", ""});
    if (h.has_value()) g_fmla_pf = new c10::OperatorHandle(*h);
  }
  if (F_w8gemm) {   // w8a16's one-time state on this device (its SMEM attributes)
    FprW8a16InitArgs ia{{nullptr, device, r}};
    FprFn init = fpr_drv_fn("w8a16", "w8a16.init", (int64_t)sizeof(FprW8a16InitArgs));
    TORCH_CHECK(init != nullptr && init(&ia) == FPR_OK, "pieces_attn: w8a16.init");
  }
}

bool attn_idx_ready() { return F_prep2 && F_bmax && F_topk && F_cand_logits && F_cand_prep; }

int64_t sizeof_attn_idx() { return (int64_t)sizeof(FprPieceAttnIdx); }
std::vector<int64_t> offsets_attn_idx() {   // ctypes cross-check (mixed_cxx_attn._piece_idx_type)
  return {(int64_t)offsetof(FprPieceAttnIdx, qq), (int64_t)offsetof(FprPieceAttnIdx, q8buf), (int64_t)offsetof(FprPieceAttnIdx, gt),
          (int64_t)offsetof(FprPieceAttnIdx, gt_vals), (int64_t)offsetof(FprPieceAttnIdx, vrow)};
}
int64_t sizeof_idx_step() { return (int64_t)sizeof(FprAttnIdxStep); }

// fpr_lt warm at handle registration (the Python thread at ensure / first build, never a pool thread): every
// GEMM key this handle's piece will issue is planned, its algo launched once (cuBLASLt module loaded here) and
// self-checked, and its main stream's workspace allocated, so the pool threads' first steps load / allocate nothing
static FprLtShape lt_sh(int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldw, int64_t ldy, int f32 = 0) {
  return FprLtShape{1, M, N, K, lda, ldw, ldy, 0, 0, 0, f32, 0};
}
static void lt_warm_handle(int dev, cudaStream_t main, std::vector<FprLtShape> v) {
  if (!fpr_lt::enabled()) return;
  // the idx_a ensure-time race goes through fpr_lt but NOT through the idx_a runtime Lt GEMMs
  // (FPR_LT_OFF=idxa still IMAs; FPR_LT=0 clean) -> this registration-time warm is the suspect) FPR_LT_HANDLE_WARM=0 skips it:
  // the init-time ltwarm (mixed_cxx_ltwarm) has planned the recorded keys; an unplanned key falls back to ATen
  // default OFF; FPR_LT_HANDLE_WARM=1 restores it (diag)
  static const bool hw = [] { const char* e = std::getenv("FPR_LT_HANDLE_WARM"); return e && e[0] == '1'; }();
  if (!hw) return;
  std::vector<FprLtShape> u;
  for (const FprLtShape& x : v)
    if (x.M > 0 && x.N > 0 && x.K > 0 && x.lda >= x.K && x.ldw >= x.K && x.ldy >= x.N) u.push_back(x);
  if (u.empty()) return;
  const c10::cuda::CUDAGuard dg((c10::DeviceIndex)dev);
  const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(main, (c10::DeviceIndex)dev));
  cudaStream_t ss[1] = {main};
  fpr_lt::warm(dev, u.data(), (int)u.size(), ss, 1);
}

int64_t add_attn_idx(py::bytes b) {
  std::string s = b;
  TORCH_CHECK((int64_t)s.size() == (int64_t)sizeof(FprPieceAttnIdx), "pieces_attn: FprPieceAttnIdx is ", s.size(),
              " bytes, the driver's ", sizeof(FprPieceAttnIdx), " (ctypes / header drift)");
  auto p = std::make_unique<FprPieceAttnIdx>();
  std::memcpy(p.get(), s.data(), s.size());
  TORCH_CHECK(p->version == FPR_PIECE_ATTN_VERSION && p->rank >= 0 && p->rank < AW_, "pieces_attn: version / rank");
  lt_warm_handle(p->device, (cudaStream_t)p->main, {lt_sh(p->R, 32, p->kx, p->x_ld, p->kx, 32),
                                                     lt_sh(p->R, 4096, p->kq, p->qr_ld, p->kq, 4096)});
  g_attn_idx.push_back(std::move(p));
  return (int64_t)g_attn_idx.size() - 1;
}

// the segment's idx values of rank r (mixed_cxx_attn.step_params; before the segment's span)
void attn_set_idx_step(int r, py::bytes b) {
  std::string s = b;
  TORCH_CHECK(r >= 0 && r < AW_ && s.size() == sizeof(FprAttnIdxStep), "pieces_attn: FprAttnIdxStep is ", s.size(), " bytes, ",
              sizeof(FprAttnIdxStep), " expected");
  std::memcpy(&g_idx_step[r], s.data(), s.size());
}

int attn_idx(int r, int64_t h, int layer) {
  TORCH_CHECK(r >= 0 && r < AW_ && h >= 0 && h < (int64_t)g_attn_idx.size(), "pieces_attn: rank / handle");
  return piece_attn_idx(g_attn_idx[h].get(), &g_idx_step[r], layer);
}
int attn_idx_c(int r, int64_t h, int layer) {
  if (r >= 0 && r < AW_ && h >= FPR_IDXA_HANDLE_BASE) {   // (Stage B) an A-role index layer: compressor + scoring
    const int64_t a = h - FPR_IDXA_HANDLE_BASE;
    if (a >= (int64_t)g_idx_a.size()) return FPR_E_ARG;
    return piece_idx_a(g_idx_a[a].get(), &g_idxa_step[r], layer, 3);
  }
  if (r < 0 || r >= AW_ || h < 0 || h >= (int64_t)g_attn_idx.size()) return FPR_E_ARG;
  return piece_attn_idx(g_attn_idx[h].get(), &g_idx_step[r], layer);
}
// (Stage B) CP prefill: the compressor in P1 (kind idx_c), the scoring in P2 (idx_s); the same A handles
int idx_half(int r, int64_t h, int layer, int what) {
  if (r < 0 || r >= AW_ || h < FPR_IDXA_HANDLE_BASE || h - FPR_IDXA_HANDLE_BASE >= (int64_t)g_idx_a.size()) return FPR_E_ARG;
  return piece_idx_a(g_idx_a[h - FPR_IDXA_HANDLE_BASE].get(), &g_idxa_step[r], layer, what);
}
int idx_c_c(int r, int64_t h, int layer) { return idx_half(r, h, layer, 1); }
int idx_s_c(int r, int64_t h, int layer) { return idx_half(r, h, layer, 2); }
bool idx_a_ready() { return F_bmax && F_topk && F_select && (F_dense8 || F_dense_c); }
bool idx_a_pf4() { return F_pf4 != nullptr; }
int64_t sizeof_idx_a() { return (int64_t)sizeof(FprPieceIdxA); }
std::vector<int64_t> offsets_idx_a() {   // ctypes cross-check (mixed_cxx_attn._piece_idxa_type)
  return {(int64_t)sizeof(FprKcRecord), (int64_t)offsetof(FprPieceIdxA, version), (int64_t)offsetof(FprPieceIdxA, fscn),
          (int64_t)offsetof(FprPieceIdxA, fscn_vals), (int64_t)offsetof(FprPieceIdxA, rqi), (int64_t)offsetof(FprPieceIdxA, iks),
          (int64_t)offsetof(FprPieceIdxA, wp), (int64_t)offsetof(FprPieceIdxA, gt), (int64_t)offsetof(FprPieceIdxA, gt_vals),
          (int64_t)offsetof(FprPieceIdxA, vrow), (int64_t)sizeof(FprTrRecord), (int64_t)offsetof(FprPieceIdxA, pad3),
          (int64_t)offsetof(FprPieceIdxA, bs), (int64_t)offsetof(FprPieceIdxA, sc_vals), (int64_t)offsetof(FprPieceIdxA, bs_i),
          (int64_t)offsetof(FprPieceIdxA, topk_ws)};
}
int64_t sizeof_idxa_step() { return (int64_t)sizeof(FprIdxAStep); }
std::vector<int64_t> offsets_idxa_step() {
  return {(int64_t)offsetof(FprIdxAStep, skip), (int64_t)offsetof(FprIdxAStep, max_len), (int64_t)offsetof(FprIdxAStep, nsl),
          (int64_t)offsetof(FprIdxAStep, sl), (int64_t)offsetof(FprIdxAStep, gst), (int64_t)sizeof(FprIdxASlice),
          (int64_t)offsetof(FprIdxAStep, global_rows), (int64_t)offsetof(FprIdxAStep, max_seq_len)};
}
int64_t add_idx_a(py::bytes b) {
  std::string s = b;
  TORCH_CHECK((int64_t)s.size() == (int64_t)sizeof(FprPieceIdxA), "pieces_attn: FprPieceIdxA is ", s.size(),
              " bytes, the driver's ", sizeof(FprPieceIdxA), " (ctypes / header drift)");
  auto p = std::make_unique<FprPieceIdxA>();
  std::memcpy(p.get(), s.data(), s.size());
  TORCH_CHECK(p->version == FPR_PIECE_ATTN_VERSION && p->rank >= 0 && p->rank < AW_ && p->lidx >= 0 && p->lidx < 4 &&
                  (p->sidx == 0 || p->sidx == 1) && (p->ratio == 1 || p->ratio == 2), "pieces_attn: idx_a version / rank / layer");
  lt_warm_handle(p->device, (cudaStream_t)p->main, {lt_sh(p->R, 32, p->kx, p->x_ld, p->kx, 32),
                                                     lt_sh(p->R, 4096, p->kq, p->qr_ld, p->kq, 4096),
                                                     lt_sh(p->R, p->ncomp, p->kx, p->x_ld, p->kx, p->ncomp, 1),
                                                     lt_sh(p->R, 128, 512, 512, 512, 128)});
  g_idx_a.push_back(std::move(p));
  return FPR_IDXA_HANDLE_BASE + (int64_t)g_idx_a.size() - 1;
}
// the A segment's idx_a values of rank r (mixed_cxx_attn.step_params; before the segment's span)
void attn_set_idxa_step(int r, py::bytes b) {
  std::string s = b;
  TORCH_CHECK(r >= 0 && r < AW_ && s.size() == sizeof(FprIdxAStep), "pieces_attn: FprIdxAStep is ", s.size(), " bytes, ",
              sizeof(FprIdxAStep), " expected");
  std::memcpy(&g_idxa_step[r], s.data(), s.size());
}
void attn_clear_idxa_step(int r) {
  TORCH_CHECK(r >= 0 && r < AW_, "pieces_attn: rank");
  g_idxa_step[r].valid = 0;
}
int attn_idx_a(int r, int64_t h, int layer, int what) {   // (tests / 1b-i) run on the caller's thread
  TORCH_CHECK(r >= 0 && r < AW_ && h >= FPR_IDXA_HANDLE_BASE && h - FPR_IDXA_HANDLE_BASE < (int64_t)g_idx_a.size(),
              "pieces_attn: rank / idx_a handle");
  return piece_idx_a(g_idx_a[h - FPR_IDXA_HANDLE_BASE].get(), &g_idxa_step[r], layer, what);
}

// the step's prefill plan of rank r (mixed_cxx_attn.step_params, bucket A with ENG_ATTN_PF); the reuse state starts over
void reset_pf_reuse(FprAttnPfStep& ps) {
  ps.ws_ratio = ps.comp_src = -1;
  ps.ck_valid[0] = ps.ck_valid[1] = 0;
}
void attn_set_pf_step(int r, py::bytes b) {
  std::string s = b;
  TORCH_CHECK(r >= 0 && r < AW_ && s.size() == sizeof(FprAttnPfStep), "pieces_attn: FprAttnPfStep is ", s.size(), " bytes, ",
              sizeof(FprAttnPfStep), " expected");
  std::memcpy(&g_pf_step[r], s.data(), s.size());
  reset_pf_reuse(g_pf_step[r]);
}
// a Python attn_prefill.attend ran on rank r since the last C++ one (it may have overwritten the workspace / comb)
void attn_pf_invalidate(int r) {
  TORCH_CHECK(r >= 0 && r < AW_, "pieces_attn: rank");
  reset_pf_reuse(g_pf_step[r]);
}
void attn_pf_clear(int r) {
  TORCH_CHECK(r >= 0 && r < AW_, "pieces_attn: rank");
  g_pf_step[r].valid = 0;
}
bool attn_pf_ready() { return F_gather != nullptr && g_fmla_pf != nullptr; }
int64_t sizeof_pf_step() { return (int64_t)sizeof(FprAttnPfStep); }
std::vector<int64_t> offsets_pf_step() {
  return {(int64_t)offsetof(FprAttnPfStep, lay), (int64_t)offsetof(FprAttnPfStep, rec), (int64_t)offsetof(FprAttnPfStep, ws_ratio),
          (int64_t)sizeof(FprAttnPfGroup), (int64_t)sizeof(FprAttnPfLayout)};
}

// the wo = 1 (ENG_WO_W8A16) variant's needs: q_b / wo w8a16, FlashMLA, attn_dense.inv_rope (no DENSE8 tables)
bool attn_b_ready_wo8() { return F_w8gemm && F_inv_rope && g_fmla_dec; }
// the wo = 2 / qdq = 1 variant's extra needs
bool attn_b_ready_dq() { return attn_b_ready_wo8() && F_rope_q_insert; }
int64_t attn_add_dq_record(py::bytes b) {
  std::string s = b;
  TORCH_CHECK((int64_t)s.size() == (int64_t)sizeof(FprKcRecord), "pieces_attn: dq record is ", s.size(), " bytes, FprKcRecord ",
              sizeof(FprKcRecord));
  auto p = std::make_unique<FprKcRecord>();
  std::memcpy(p.get(), s.data(), s.size());
  TORCH_CHECK(p->function != nullptr && p->n_params <= FPR_KC_MAX_PARAMS, "pieces_attn: dq record without a function");
  g_dq.push_back(std::move(p));
  dq_pf_init_all();                      // FPR_ATTN_DQ_PF: side streams / events, once, between spans
  return (int64_t)g_dq.size() - 1;
}
bool attn_b_ready() {
  return F_w8gemm && F_ropeq && F_actq && F_dg_gemm && F_dg_einsum && g_fmla_dec;
}

// a new bucket segment: its FlashMLA plans start over (Python: _new_scheds' fresh FlashMLASchedMeta objects per step)
void attn_reset_plans(int r, std::vector<int64_t> slots) {
  TORCH_CHECK(r >= 0 && r < AW_, "pieces_attn: rank");
  const std::lock_guard<std::mutex> lk(g_ar_mu[r]);
  for (int64_t i : slots) {
    TORCH_CHECK(i >= 0 && i < kMaxPlans, "pieces_attn: plan slot");
    g_ar[r].have[i] = false;
    g_ar[r].tsm[i] = at::Tensor();
    g_ar[r].ns[i] = at::Tensor();
  }
}
int64_t max_plans() { return kMaxPlans; }

int64_t sizeof_attn_b() { return (int64_t)sizeof(FprPieceAttnB); }

int64_t add_attn_b(py::bytes b, int64_t gen) {
  std::string s = b;
  TORCH_CHECK((int64_t)s.size() == (int64_t)sizeof(FprPieceAttnB), "pieces_attn: FprPieceAttnB is ", s.size(),
              " bytes, the driver's ", sizeof(FprPieceAttnB), " (ctypes / header drift)");
  auto p = std::make_unique<AttnBH>();
  std::memcpy(&p->b, s.data(), s.size());
  p->gen = (uint64_t)gen;
  if (gen == 0) {   // a handle without its view's generation: the owner check falls back to (si ptr, R)
    static uint64_t n0 = 0;
    if (++n0 <= 4 || (n0 & (n0 - 1)) == 0)
      std::fprintf(stderr, "pieces_attn: add_attn_b gen 0 (pointer-key owner): %llu handle(s)\n", (unsigned long long)n0);
  }
  TORCH_CHECK(p->b.version == FPR_PIECE_ATTN_VERSION && p->b.rank >= 0 && p->b.rank < AW_, "pieces_attn: version / rank");
  {
    const FprPieceAttnB& q = p->b;
    std::vector<FprLtShape> v = {lt_sh(q.R, q.q_n, q.qb.kx, q.qb.x_ld, q.q_k, q.qb.ldy), lt_sh(q.R, q.b_n, q.b_k, q.z_ld, q.b_k, q.y_ld)};
    if (q.R > 0 && q.z_ld >= 1024) v.push_back(FprLtShape{8, q.R, 1024, 4096, 32768, 4096, q.z_ld, 4096, 1024 * 4096, 1024, 0, 0});
    lt_warm_handle(q.device, (cudaStream_t)q.main, v);
  }
  g_attn_b.push_back(std::move(p));
  return (int64_t)g_attn_b.size() - 1;
}

// test entry: only the attention core of handle h (no q_b, no wo)
int attn_b_core(int r, int64_t h, int layer) {
  TORCH_CHECK(r >= 0 && r < AW_ && h >= 0 && h < (int64_t)g_attn_b.size(), "pieces_attn: rank / handle");
  const FprPieceAttnB* p = &g_attn_b[h]->b;
  TORCH_CHECK(p->layer == layer && p->rank == r, "pieces_attn: handle of another layer / rank");
  return attn_core(p);
}

int attn_b(int r, int64_t h, int layer) {
  TORCH_CHECK(r >= 0 && r < AW_ && h >= 0 && h < (int64_t)g_attn_b.size(), "pieces_attn: rank / handle");
  return piece_attn_b(&g_attn_b[h]->b, fpr_drv_step_params(r), layer);
}

// stage 1d: the same pieces as the driver threads' C functions (fpr_drv_register_piece): rank's thread, device set, no GIL
int attn_a_c(int r, int64_t h, int layer) {
  if (r < 0 || r >= AW_ || h < 0 || h >= (int64_t)g_attn_a.size()) return FPR_E_ARG;
  return piece_attn_a(g_attn_a[h].get(), fpr_drv_step_params(r), layer);
}
int attn_b_c(int r, int64_t h, int layer) {
  if (r < 0 || r >= AW_ || h < 0 || h >= (int64_t)g_attn_b.size()) return FPR_E_ARG;
  return piece_attn_b(&g_attn_b[h]->b, fpr_drv_step_params(r), layer);
}
// CP prefill (mixed_cxx ENG_MIXED_CXX_CP=2): the two halves of an attn_b handle in phases P2 / P3
int attn_bi_c(int r, int64_t h, int layer) {
  if (r < 0 || r >= AW_ || h < 0 || h >= (int64_t)g_attn_b.size()) return FPR_E_ARG;
  const FprPieceAttnB* p = &g_attn_b[h]->b;
  const int rc = attn_b_check(p, layer);
  return rc != FPR_OK ? rc : attn_b_ins(p);
}
int attn_bo_c(int r, int64_t h, int layer) {
  if (r < 0 || r >= AW_ || h < 0 || h >= (int64_t)g_attn_b.size()) return FPR_E_ARG;
  const FprPieceAttnB* p = &g_attn_b[h]->b;
  const int rc = attn_b_check(p, layer);
  return rc != FPR_OK ? rc : attn_b_att(p);
}

int64_t sizeof_attn_a() { return (int64_t)sizeof(FprPieceAttnA); }

int64_t add_attn_a(py::bytes b) {
  std::string s = b;
  TORCH_CHECK((int64_t)s.size() == (int64_t)sizeof(FprPieceAttnA), "pieces_attn: FprPieceAttnA is ", s.size(),
              " bytes, the driver's ", sizeof(FprPieceAttnA), " (ctypes / header drift)");
  auto p = std::make_unique<FprPieceAttnA>();
  std::memcpy(p.get(), s.data(), s.size());
  TORCH_CHECK(p->version == FPR_PIECE_ATTN_VERSION && p->rank >= 0 && p->rank < AW_, "pieces_attn: version / rank");
  lt_warm_handle(p->device, (cudaStream_t)p->main, {lt_sh(p->R, p->n, p->k, p->a_ld, p->k, p->n)});
  g_attn_a.push_back(std::move(p));
  return (int64_t)g_attn_a.size() - 1;
}

int attn_a(int r, int64_t h, int layer) {
  TORCH_CHECK(r >= 0 && r < AW_ && h >= 0 && h < (int64_t)g_attn_a.size(), "pieces_attn: rank / handle");
  return piece_attn_a(g_attn_a[h].get(), fpr_drv_step_params(r), layer);
}

// ---- CP prefill exchanges (pieces_attn.h FprCpMsg; ENG_MIXED_CXX_CP=3)
struct CpState {
  std::vector<FprCpMsg> msg;
  std::vector<at::Tensor> data, buf;          // per message: the gathered rows (sender) and the receive buffer
  std::vector<cudaEvent_t> ev;
  int64_t epoch = 0;
};
CpState g_cp;
std::atomic<int64_t> g_cp_pub[FPR_CP_MAX_MSGS];   // = epoch once the message's event is recorded (release)
cudaStream_t g_cp_stream[AW_] = {};              // the ranks' main streams (attn_cp_step)
inline cudaStream_t g_rank_stream(int r) { return g_cp_stream[r]; }

at::Tensor u8_rows(void* base, int64_t rows, int64_t ld, int dev) {
  return blob(base, {rows, ld}, {ld, 1}, at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA, dev)));
}
at::Tensor i64_vec(const void* p, int64_t n, int dev) {
  return blob(const_cast<void*>(p), {n}, at::TensorOptions().dtype(at::kLong).device(at::Device(at::kCUDA, dev)));
}

int cp_send(int r, int layer, int kind) {
  const int64_t ep = g_cp.epoch;
  for (size_t m = 0; m < g_cp.msg.size(); ++m) {
    const FprCpMsg& M = g_cp.msg[m];
    if (M.src_rank != r || M.layer != layer || M.kind != kind) continue;
    try {
      cudaStream_t rs = g_rank_stream(r);
      const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(rs, (c10::DeviceIndex)M.src_dev));
      // = cp.py: self._pg(r).index_select(0, src)[:, o:o + n].contiguous()   (x2: self._wsec(r)...)
      at::Tensor data = u8_rows(M.src_base, M.src_rows, M.src_ld, M.src_dev)
                            .index_select(0, i64_vec(M.src_idx, M.n, M.src_dev))
                            .slice(1, M.off, M.off + M.len)
                            .contiguous();
      at::Tensor buf;
      {   // the receive buffer from the receiver's CP mail stream's pool (cp.py _send, MAIL_STREAM)
        const c10::cuda::CUDAStreamGuard gm(
            c10::cuda::getStreamFromExternal((cudaStream_t)M.mail_stream, (c10::DeviceIndex)M.dst_dev));
        buf = at::empty({M.n, M.len}, at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA, M.dst_dev)));
      }
      C10_CUDA_CHECK(cudaMemcpyAsync(buf.data_ptr(), data.data_ptr(), (size_t)(M.n * M.len), cudaMemcpyDeviceToDevice, rs));
      C10_CUDA_CHECK(cudaEventRecord(g_cp.ev[m], rs));
      g_cp.data[m] = data;
      g_cp.buf[m] = buf;
      g_cp_pub[m].store(ep, std::memory_order_release);
    } catch (const std::exception&) {
      if (pieces_throw()) throw;
      return FPR_E_CUDA;
    }
  }
  return FPR_OK;
}

int cp_recv(int q, int layer, int kind) {
  const int64_t ep = g_cp.epoch;
  for (size_t m = 0; m < g_cp.msg.size(); ++m) {
    const FprCpMsg& M = g_cp.msg[m];
    if (M.dst_rank != q || M.layer != layer || M.kind != kind) continue;
    const auto t0 = std::chrono::steady_clock::now();
    while (g_cp_pub[m].load(std::memory_order_acquire) != ep) {   // the sender's thread records the event first
      std::this_thread::yield();
      if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30))
        return thr_rc(-100, "cp_recv: the sender never published (30 s)");   // (= driver.cpp FPR_E_HANDSHAKE)
    }
    try {
      cudaStream_t rs = g_rank_stream(q);
      const c10::cuda::CUDAStreamGuard g(c10::cuda::getStreamFromExternal(rs, (c10::DeviceIndex)M.dst_dev));
      C10_CUDA_CHECK(cudaStreamWaitEvent(rs, g_cp.ev[m], 0));           // = cp.py _recv's wait_event
      // = cp.py: self._pg(r)[pages[r][q], o:o + n] = buf   (x2: self._wsec(r)[halo_dst[r], L * WLB:(L + 1) * WLB] = buf)
      u8_rows(M.dst_base, M.dst_rows, M.dst_ld, M.dst_dev)
          .slice(1, M.off, M.off + M.len)
          .index_put_({i64_vec(M.dst_idx, M.n, M.dst_dev)}, g_cp.buf[m]);
    } catch (const std::exception&) {
      if (pieces_throw()) throw;
      return FPR_E_CUDA;
    }
  }
  return FPR_OK;
}
int cp_x1a_send_c(int r, int64_t h, int layer) { (void)h; return cp_send(r, layer, FPR_CP_X1A); }
int cp_x1a_recv_c(int r, int64_t h, int layer) { (void)h; return cp_recv(r, layer, FPR_CP_X1A); }
int cp_x2_send_c(int r, int64_t h, int layer) { (void)h; return cp_send(r, layer, FPR_CP_X2); }
int cp_x2_recv_c(int r, int64_t h, int layer) { (void)h; return cp_recv(r, layer, FPR_CP_X2); }

// Python (between spans): the step's messages; streams[r] = rank r's main stream (the driver's), events created here
// (the body on a byte span: the pybind entry below and the C entry fpr_attn_cp_step, which runs without the GIL)
void cp_step_impl(const char* data, size_t size, const std::vector<int64_t>& streams) {
  struct Span { const char* p; size_t n; size_t size() const { return n; } const char* data() const { return p; } } s{data, size};
  TORCH_CHECK(s.size() % sizeof(FprCpMsg) == 0, "pieces_attn: cp messages are not FprCpMsg[] (", s.size(), " bytes)");
  const size_t n = s.size() / sizeof(FprCpMsg);
  TORCH_CHECK(n <= FPR_CP_MAX_MSGS, "pieces_attn: ", n, " CP messages > ", FPR_CP_MAX_MSGS);
  TORCH_CHECK(g_cp.msg.empty(), "pieces_attn: attn_cp_step before the previous step's attn_cp_done");
  TORCH_CHECK((int)streams.size() == AW_, "pieces_attn: attn_cp_step needs the 4 rank streams");
  for (int r = 0; r < AW_; ++r) g_cp_stream[r] = (cudaStream_t)streams[r];
  g_cp.msg.resize(n);
  std::memcpy(g_cp.msg.data(), s.data(), s.size());
  g_cp.data.assign(n, at::Tensor());
  g_cp.buf.assign(n, at::Tensor());
  g_cp.ev.assign(n, nullptr);
  for (size_t m = 0; m < n; ++m) {
    const FprCpMsg& M = g_cp.msg[m];
    TORCH_CHECK(M.src_rank >= 0 && M.src_rank < AW_ && M.dst_rank >= 0 && M.dst_rank < AW_ && M.n >= 0 && M.len > 0 &&
                    M.off >= 0 && M.off + M.len <= M.src_ld && M.off + M.len <= M.dst_ld && M.mail_stream != nullptr,
                "pieces_attn: CP message ", m, " out of range");
    const c10::cuda::CUDAGuard dg((c10::DeviceIndex)M.src_dev);   // (events live on the sender's device)
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&g_cp.ev[m], cudaEventDisableTiming));
  }
  ++g_cp.epoch;
}
void attn_cp_step(py::bytes b, std::vector<int64_t> streams) {
  std::string s = b;
  cp_step_impl(s.data(), s.size(), streams);
}
// Python, after the step's sync (CPPlan.done): the step's tensors freed, events destroyed
void attn_cp_done() {
  for (auto& e : g_cp.ev)
    if (e != nullptr) C10_CUDA_CHECK(cudaEventDestroy(e));
  g_cp.ev.clear();
  g_cp.data.clear();
  g_cp.buf.clear();
  g_cp.msg.clear();
}
int64_t sizeof_cp_msg() { return (int64_t)sizeof(FprCpMsg); }
}  // namespace

// ---- the setters as C entries for nd's native mixed step (pieces_attn.h): the same state the
// pybind setters write, from C structs built by fpr/core/src/p1 (steps_build / cp_build); a status instead of an
// exception. Called on the nd loop thread between spans (as the pybind ones on the Python thread).
extern "C" __attribute__((visibility("default"))) int fpr_attn_set_idx_step(int r, const FprAttnIdxStep* st) {
  if (r < 0 || r >= AW_ || st == nullptr) return FPR_E_ARG;
  std::memcpy(&g_idx_step[r], st, sizeof(FprAttnIdxStep));
  g_idx_step[r].prep_done = 0;
  return FPR_OK;
}
extern "C" __attribute__((visibility("default"))) int fpr_attn_set_idxa_step(int r, const FprIdxAStep* st) {   // NULL: no idx_a values this segment
  if (r < 0 || r >= AW_) return FPR_E_ARG;
  if (st == nullptr) g_idxa_step[r].valid = 0;
  else std::memcpy(&g_idxa_step[r], st, sizeof(FprIdxAStep));
  return FPR_OK;
}
extern "C" __attribute__((visibility("default"))) int fpr_attn_set_pf_step(int r, const FprAttnPfStep* st) {    // NULL: no prefill plan this step
  if (r < 0 || r >= AW_) return FPR_E_ARG;
  if (st == nullptr) {
    g_pf_step[r].valid = 0;
    return FPR_OK;
  }
  std::memcpy(&g_pf_step[r], st, sizeof(FprAttnPfStep));
  reset_pf_reuse(g_pf_step[r]);
  return FPR_OK;
}
extern "C" __attribute__((visibility("default"))) int fpr_attn_cp_step(const FprCpMsg* msgs, int64_t n, const void* const* rank_streams) {
  if ((n && msgs == nullptr) || rank_streams == nullptr || n < 0) return FPR_E_ARG;
  try {
    std::vector<int64_t> streams(AW_);
    for (int r = 0; r < AW_; ++r) streams[r] = (int64_t)(intptr_t)rank_streams[r];
    cp_step_impl(reinterpret_cast<const char*>(msgs), (size_t)n * sizeof(FprCpMsg), streams);
  } catch (const std::exception&) {
    return FPR_E_ARG;                             // (attn_cp_step's checks: count / ranges / a previous step not done)
  }
  return FPR_OK;
}
// the combine templates' key-class check for core2b (fpr_tr_check is a header inline; core2b has no CUDA headers)
extern "C" __attribute__((visibility("default"))) int fpr_attn_tr_check(const void* record, const uint64_t* vals) {
  return record == nullptr || vals == nullptr ? FPR_E_ARG : fpr_tr_check(static_cast<const FprTrRecord*>(record), vals);
}
extern "C" __attribute__((visibility("default"))) int fpr_attn_cp_max_msgs(void) { return FPR_CP_MAX_MSGS; }
extern "C" __attribute__((visibility("default"))) int fpr_attn_cp_done(void) {
  try {
    attn_cp_done();
  } catch (const std::exception&) {
    return FPR_E_CUDA;
  }
  return FPR_OK;
}

// attn_reset_plans for C callers
// rank r's FlashMLA plan slots start over. Every slot is checked before any is touched (FPR_E_ARG: nothing changed);
// no GIL needed (no Python objects); same effect as the pybind attn_reset_plans.
extern "C" __attribute__((visibility("default"))) int fpr_attn_reset_plans(int rank, const int64_t* slots, int n) {
  if (rank < 0 || rank >= AW_ || n < 0 || (n > 0 && slots == nullptr)) return FPR_E_ARG;
  for (int k = 0; k < n; ++k)
    if (slots[k] < 0 || slots[k] >= kMaxPlans) return FPR_E_ARG;
  try {
    const std::lock_guard<std::mutex> lk(g_ar_mu[rank]);
    for (int k = 0; k < n; ++k) {
      g_ar[rank].have[slots[k]] = false;
      g_ar[rank].tsm[slots[k]] = at::Tensor();
      g_ar[rank].ns[slots[k]] = at::Tensor();
    }
  } catch (const std::exception&) {
    return FPR_E_CUDA;
  }
  return FPR_OK;
}

void fpr_pieces_attn_register(pybind11::module_& m) {
  fpr_drv_register_piece("attn_a", &attn_a_c);   // stage 1d spans (handle_attn_a / handle_attn_b in mixed_cxx_attn.py)
  fpr_drv_register_piece("attn_b", &attn_b_c);
  fpr_drv_register_piece("attn_bi", &attn_bi_c);   // CP prefill phases (ENG_MIXED_CXX_CP=2)
  fpr_drv_register_piece("attn_bo", &attn_bo_c);
  fpr_drv_register_piece("cp_x1a_send", &cp_x1a_send_c);   // CP prefill exchanges (ENG_MIXED_CXX_CP=3)
  fpr_drv_register_piece("cp_x1a_recv", &cp_x1a_recv_c);
  fpr_drv_register_piece("cp_x2_send", &cp_x2_send_c);
  fpr_drv_register_piece("cp_x2_recv", &cp_x2_recv_c);
  m.def("attn_cp_step", &attn_cp_step, "pieces_attn: this CP step's exchange messages (FprCpMsg[] bytes, rank streams)");
  m.def("attn_cp_done", &attn_cp_done, "pieces_attn: the CP step's exchange tensors / events freed (after its sync)");
  m.def("sizeof_cp_msg", &sizeof_cp_msg);
  fpr_drv_register_piece("idx", &attn_idx_c);
  fpr_drv_register_piece("idx_c", &idx_c_c);   // (Stage B) CP prefill: A-role compressor (P1) / scoring (P2)
  fpr_drv_register_piece("idx_s", &idx_s_c);
  m.def("idx_a_ready", &idx_a_ready, "pieces_attn: the idx2f tables the A-role idx piece needs are resolved");
  m.def("idx_a_pf4", &idx_a_pf4, "pieces_attn: idx2f_pf4.logits_prefill resolved (IDX2F_PF slices)");
  m.def("sizeof_idx_a", &sizeof_idx_a);
  m.def("offsets_idx_a", &offsets_idx_a);
  m.def("sizeof_idxa_step", &sizeof_idxa_step);
  m.def("offsets_idxa_step", &offsets_idxa_step);
  m.def("add_idx_a", &add_idx_a, "pieces_attn: a static FprPieceIdxA (bytes) -> handle (>= FPR_IDXA_HANDLE_BASE)");
  m.def("attn_set_idxa_step", &attn_set_idxa_step, "pieces_attn: rank r's FprIdxAStep for the A segment (bytes)");
  m.def("attn_clear_idxa_step", &attn_clear_idxa_step, "pieces_attn: rank r has no idx_a values this segment");
  m.def("attn_idx_a", &attn_idx_a, "pieces_attn: run an idx_a handle (rank, handle, layer, what 1/2/3) -> FPR status");
  m.def("attn_init_rank", &attn_init_rank, "pieces_attn: per rank (device current): resolve the attention tables' functions");
  m.def("sizeof_attn_a", &sizeof_attn_a);
  m.def("add_attn_a", &add_attn_a, "pieces_attn: a static FprPieceAttnA (bytes) -> handle");
  m.def("attn_a", &attn_a, "pieces_attn: run the attn_a piece (rank, handle, layer) -> FPR status");
  m.def("attn_b_ready", &attn_b_ready, "pieces_attn: every table / op attn_b needs is resolved");
  m.def("attn_b_ready_wo8", &attn_b_ready_wo8, "pieces_attn: the ENG_WO_W8A16 attn_b variant's tables / ops are resolved");
  m.def("attn_b_ready_dq", &attn_b_ready_dq, "pieces_attn: + rope_q_insert (the exact-dequant attn_b variant)");
  m.def("attn_add_dq_record", &attn_add_dq_record, "pieces_attn: a w8dq dequant FprKcRecord (bytes) -> handle");
  m.def("attn_idx_ready", &attn_idx_ready, "pieces_attn: the idx2f tables the idx piece needs are resolved");
  m.def("sizeof_attn_idx", &sizeof_attn_idx);
  m.def("offsets_attn_idx", &offsets_attn_idx);
  m.def("sizeof_idx_step", &sizeof_idx_step);
  m.def("add_attn_idx", &add_attn_idx, "pieces_attn: a static FprPieceAttnIdx (bytes) -> handle");
  m.def("attn_set_idx_step", &attn_set_idx_step, "pieces_attn: rank r's FprAttnIdxStep for the segment (bytes)");
  m.def("attn_idx", &attn_idx, "pieces_attn: run the idx piece (rank, handle, layer) -> FPR status");
  m.def("attn_set_pf_step", &attn_set_pf_step, "pieces_attn: rank r's FprAttnPfStep (bytes) for the step's prefill plan");
  m.def("attn_pf_invalidate", &attn_pf_invalidate, "pieces_attn: a Python attend ran on rank r: reuse state starts over");
  m.def("attn_pf_clear", &attn_pf_clear, "pieces_attn: rank r has no prefill plan this step");
  m.def("attn_pf_ready", &attn_pf_ready, "pieces_attn: gatherk + _flashmla_C::sparse_prefill_fwd resolved");
  m.def("sizeof_pf_step", &sizeof_pf_step);
  m.def("offsets_pf_step", &offsets_pf_step);
  m.def("attn_reset_plans", &attn_reset_plans, "pieces_attn: these FlashMLA plan slots of rank r start over (segment start)");
  m.def("max_plans", &max_plans);
  m.def("sizeof_attn_b", &sizeof_attn_b);
  m.def("add_attn_b", &add_attn_b, "pieces_attn: a static FprPieceAttnB (bytes) -> handle; gen: the view's generation",
        py::arg("b"), py::arg("gen") = 0);
  m.def("attn_b", &attn_b, "pieces_attn: run the attn_b piece (rank, handle, layer) -> FPR status");
  m.def("attn_b_core", &attn_b_core, "pieces_attn (test): the attention core of an attn_b handle -> FPR status");
}
