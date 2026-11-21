// D13 steady-state decode, stage 1: HSD's position-dependent per-step tables on the
// device.
//
// In a steady decode step (the same drafting requests in the same order as the previous step, all decode, no grammar),
// HSD's host prepare (kernels/head/hsd.py HSD._build) changes only these fields of its per-rank blob; everything else
// (cs / ce / req_idx / g_dslot / row_req / row_local / tok_slot / tok_j / temp / seeds / rstart / lslot / seq_of_d) is
// the same. This kernel derives them from the previous step's device state, so the host can skip _build + the blob
// upload and the positions D2H:
//   shared section (identical on every rank: every rank holds every request's num_sampled):
//     g_p0[q]  = g_p0[q] + num_sampled[q]              (drafting requests: g_dslot[q] >= 0; = rowexp's k_advance)
//     pos[t]   = g_p0[q] + (t - cs[q])                 (t in [cs[q], ce[q]); decode rows: consecutive positions)
//   rank-specific (q = q0 + j, j < D_pad):
//     wbase[j]     = max(g_p0[q] + 1 - WINDOW, 0) / 32                          (hsd.py: wb)
//     wpages[j][c] = the window page of block wbase + c (c < NBW) of state slot lslot[j], from rowexp's ring:
//                    = window_table.get(block, -1) (the ring holds the window table's blocks; -1 when absent)
//     tpos[i] / tslot[i] = rowexp's pos / own of row i (i < nt: steady decode = every row is a tail row)
// Same integer formulas as the host; ONE CTA (shared-section writes, then a barrier, then the rank-specific reads).
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "hsdprep_cabi.h"
#include "../../engine/mixed_cxx/fpr_cabi_rec.h"

#include <cstdint>

FPR_CABI_RECORD_DECL(hsdprep)
namespace d13s {

constexpr int RB = 512;  // rowexp ring blocks per state slot (kernels/rowexp/rowexp.cu)
constexpr int WT = 32;   // window page tokens

__global__ void __launch_bounds__(512) k_hsdprep(const int* __restrict__ cs, const int* __restrict__ ce,
                                                 const int64_t* __restrict__ g_dslot, int64_t* __restrict__ g_p0,
                                                 int64_t* __restrict__ pos, const int* __restrict__ num_sampled, int WD,
                                                 int q0, int D_pad, int64_t* __restrict__ wbase,
                                                 int64_t* __restrict__ wpages, int nbw, const int64_t* __restrict__ lslot,
                                                 const int2* __restrict__ ring, int window, int64_t* __restrict__ tpos,
                                                 int64_t* __restrict__ tslot, int nt, const int64_t* __restrict__ rx_pos,
                                                 const int64_t* __restrict__ rx_own, int* __restrict__ err) {
  for (int q = threadIdx.x; q < WD; q += blockDim.x) {
    if (g_dslot[q] < 0) continue;
    const int64_t p = g_p0[q] + (int64_t)num_sampled[q];
    g_p0[q] = p;
    const int a = cs[q], b = ce[q];
    for (int t = a; t < b; ++t) pos[t] = p + (int64_t)(t - a);
  }
  __syncthreads();
  for (int x = threadIdx.x; x < D_pad * nbw; x += blockDim.x) {
    const int j = x / nbw, c = x - j * nbw;
    const int q = q0 + j;
    if (g_dslot[q] < 0) continue;  // dummy request: the host's -1 / 0 stay
    const int64_t p0 = g_p0[q];
    const int64_t lo = p0 + 1 - window;
    const int64_t wb = (lo > 0 ? lo : 0) / WT;
    if (c == 0) wbase[j] = wb;
    const int64_t blk = wb + c;
    const int2 e = ring[(size_t)lslot[j] * RB + (int)(blk & (RB - 1))];
    wpages[(size_t)j * nbw + c] = (e.x == (int)blk) ? (int64_t)e.y : (int64_t)-1;
  }
  for (int i = threadIdx.x; i < nt; i += blockDim.x) {
    tpos[i] = rx_pos[i];
    tslot[i] = rx_own[i];
  }
  (void)err;
}

// ---- C ABI (hsdprep_cabi.h; fpr_cabi.h rules): the launch of pybind hsdprep, called by it and by the C++ engine (stage 2b
// NativeDecode, steady steps). Same launch configuration and arguments as before (host code only).
static int hsdprep_core(const FprHsdprepArgs* a) {   // "hsdprep.hsdprep"
  k_hsdprep<<<1, 512, 0, a->c.stream>>>(a->cs, a->ce, a->g_dslot, a->g_p0, a->pos, a->num_sampled, (int)a->WD, (int)a->q0,
                                        (int)a->D_pad, a->wbase, a->wpages, (int)a->nbw, a->lslot,
                                        reinterpret_cast<const int2*>(a->ring), (int)a->window, a->tpos, a->tslot, (int)a->nt,
                                        a->rx_pos, a->rx_own, a->err);
  return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}

void chk(const torch::Tensor& t, at::ScalarType dt, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous(), n, ": contiguous CUDA tensor of the right dtype");
}

void hsdprep(torch::Tensor cs, torch::Tensor ce, torch::Tensor g_dslot, torch::Tensor g_p0, torch::Tensor pos,
             torch::Tensor num_sampled, int64_t q0, int64_t D_pad, torch::Tensor wbase, torch::Tensor wpages,
             torch::Tensor lslot, torch::Tensor ring, int64_t window, torch::Tensor tpos, torch::Tensor tslot, int64_t nt,
             torch::Tensor rx_pos, torch::Tensor rx_own, torch::Tensor err) {
  chk(cs, at::kInt, "cs");
  chk(ce, at::kInt, "ce");
  chk(g_dslot, at::kLong, "g_dslot");
  chk(g_p0, at::kLong, "g_p0");
  chk(pos, at::kLong, "pos");
  chk(num_sampled, at::kInt, "num_sampled");
  chk(wbase, at::kLong, "wbase");
  chk(wpages, at::kLong, "wpages");
  chk(lslot, at::kLong, "lslot");
  chk(ring, at::kInt, "ring");
  chk(tpos, at::kLong, "tpos");
  chk(tslot, at::kLong, "tslot");
  chk(rx_pos, at::kLong, "rx_pos");
  chk(rx_own, at::kLong, "rx_own");
  chk(err, at::kInt, "err");
  const int WD = (int)cs.numel();
  TORCH_CHECK(ce.numel() == WD && g_dslot.numel() == WD && g_p0.numel() == WD && num_sampled.numel() >= WD, "WD sizes");
  TORCH_CHECK(wpages.dim() == 2 && wpages.size(0) >= D_pad && wbase.numel() >= D_pad && lslot.numel() >= D_pad, "D sizes");
  TORCH_CHECK(tpos.numel() >= nt && tslot.numel() >= nt && rx_pos.numel() >= nt && rx_own.numel() >= nt, "tail sizes");
  TORCH_CHECK(ring.dim() == 3 && ring.size(1) == RB && ring.size(2) == 2, "ring [slots, 512, 2]");
  TORCH_CHECK(q0 + D_pad <= WD, "q0 + D_pad <= WD");
  const at::cuda::CUDAGuard g(cs.device());
  FprHsdprepArgs a{};
  a.c = FprCall{at::cuda::getCurrentCUDAStream(), (int)cs.get_device(), 0};
  a.cs = cs.data_ptr<int>(); a.ce = ce.data_ptr<int>(); a.g_dslot = g_dslot.data_ptr<int64_t>(); a.g_p0 = g_p0.data_ptr<int64_t>();
  a.pos = pos.data_ptr<int64_t>(); a.num_sampled = num_sampled.data_ptr<int>(); a.WD = WD; a.q0 = (int32_t)q0;
  a.D_pad = (int32_t)D_pad; a.wbase = wbase.data_ptr<int64_t>(); a.wpages = wpages.data_ptr<int64_t>();
  a.nbw = (int32_t)wpages.size(1); a.lslot = lslot.data_ptr<int64_t>(); a.ring = ring.data_ptr<int>(); a.window = (int32_t)window;
  a.tpos = tpos.data_ptr<int64_t>(); a.tslot = tslot.data_ptr<int64_t>(); a.nt = (int32_t)nt; a.rx_pos = rx_pos.data_ptr<int64_t>();
  a.rx_own = rx_own.data_ptr<int64_t>(); a.err = err.data_ptr<int>();
  FPR_CABI_RECORD("hsdprep.hsdprep", a);
  TORCH_CHECK(hsdprep_core(&a) == FPR_OK, "hsdprep.hsdprep: launch error ", cudaGetErrorString(cudaPeekAtLastError()));
}

}  // namespace d13s

extern "C" {
static int c_hsdprep(const void* a) { return d13s::hsdprep_core((const FprHsdprepArgs*)a); }
static const FprFnEntry k_hsdprep_fns[] = {{"hsdprep.hsdprep", c_hsdprep, nullptr, (int64_t)sizeof(FprHsdprepArgs)}};
static const FprExtTable k_hsdprep_table = {FPR_CABI_VERSION, "hsdprep", "", 1, k_hsdprep_fns};
__attribute__((visibility("default"))) const FprExtTable* fpr_hsdprep_table(void) { return &k_hsdprep_table; }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hsdprep", &d13s::hsdprep, "D13 steady decode: HSD's position-dependent tables on the device (one CTA)");
}
