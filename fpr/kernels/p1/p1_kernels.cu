// p1_kernels.cu -- see p1_cabi.h. Build: p1k.py (torch extension: the C ABI table + pybind test
// entries; nd's pre-step resolves "p1.*" through fpr_drv_fn).
#include <cuda_runtime.h>
#include <stdint.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "p1_cabi.h"

namespace p1k {

__global__ void k_pf_wbt(const int32_t* __restrict__ si, int64_t ld_si, const int32_t* __restrict__ sl,
                         const int64_t* __restrict__ seq, int32_t* __restrict__ wbt, int64_t ld_wbt, int64_t wmax, int64_t wpt,
                         const int32_t* __restrict__ ends, const int32_t* __restrict__ gls, const int32_t* __restrict__ qsl,
                         int64_t row_lo) {
  const int64_t s = blockIdx.x;
  int64_t row0, n, first, g0;
  if (seq != nullptr) {
    row0 = seq[4 * s], n = seq[4 * s + 1], first = seq[4 * s + 2], g0 = seq[4 * s + 3];
  } else {                                           // (v0.6 blob mode)
    row0 = row_lo + qsl[s], n = (int64_t)qsl[s + 1] - qsl[s], first = (int64_t)ends[s] - n, g0 = (int64_t)ends[s] - gls[s];
  }
  int32_t* w = wbt + s * ld_wbt;
  for (int64_t c = threadIdx.x; c < wmax; c += blockDim.x) w[c] = 0;
  __syncthreads();
  const int64_t end = first + n;
  for (int64_t p = g0 + threadIdx.x; p < end; p += blockDim.x) {
    int32_t slot;
    if (p <= first) {
      slot = si[row0 * ld_si + (p - g0)];
    } else {
      const int64_t row = row0 + (p - first);
      slot = si[row * ld_si + (sl[row] - 1)];
    }
    // (positions of one window page map to one window page: equal values, any write order)
    w[p / wpt] = (int32_t)((int64_t)slot / wpt);   // (slots >= 0: Python's floor division)
  }
}

__global__ void k_pf_inv(int32_t* __restrict__ inv, const int32_t* __restrict__ bt, int64_t ld_bt,
                         const int32_t* __restrict__ nb, int64_t n_inv, const int32_t* __restrict__ ends) {
  const int64_t s = blockIdx.y;
  const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t nbs = nb != nullptr ? nb[s] : ((int64_t)ends[s] + 63) / 64;   // (blob mode: Plan's -(-ends // 64))
  if (j >= nbs) return;
  const int64_t pg = bt[s * ld_bt + j];
  if (pg >= 0 && pg < n_inv) inv[pg] = (int32_t)j;   // (Python: index error out of range; the host checks nb / pages)
}

__global__ void k_to_local(const int32_t* __restrict__ ti, const int32_t* __restrict__ inv, int64_t n_inv,
                           int32_t* __restrict__ loc, int64_t total, int64_t rpp) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int32_t g = ti[i];
  const int64_t gl = g < 0 ? 0 : g;                  // g.long().clamp_min(0)
  const int64_t pi = gl / rpp;
  const int32_t iv = pi < n_inv ? inv[pi] : -1;
  loc[i] = (g >= 0 && iv >= 0) ? (int32_t)((int64_t)iv * rpp + gl % rpp) : -1;
}

__global__ void k_carry(const int64_t* __restrict__ ir2row, const uint8_t* __restrict__ valid, const int64_t* __restrict__ ix,
                        const int64_t* __restrict__ row2ir_A, const int32_t* __restrict__ cand_A, int32_t* __restrict__ cand_T) {
  const int64_t i = blockIdx.x;
  const int64_t g = i / 4 * 4;
  int64_t last = -1;                                 // the group's last valid row (amax over valid entries, -1 none)
  for (int j = 0; j < 4; ++j)
    if (valid[g + j]) last = ir2row[g + j] > last ? ir2row[g + j] : last;
  const int64_t fill = valid[i] ? ir2row[i] : last;
  const int64_t src = row2ir_A[ix[fill]];
  const int4* a = reinterpret_cast<const int4*>(cand_A + src * 2048);
  int4* t = reinterpret_cast<int4*>(cand_T + i * 2048);
  for (int c = threadIdx.x; c < 512; c += blockDim.x) t[c] = a[c];
}

int rc_of(cudaError_t e) { return e == cudaSuccess ? FPR_OK : FPR_E_CUDA; }

int c_pf_wbt(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprP1PfWbtArgs*>(p);
  if (a.S < 0 || a.wmax < 0 || a.wmax > a.ld_wbt || a.wpt <= 0 ||
      (a.S && (!a.si || !a.sl || !a.wbt || (!a.seq && (!a.seq_lens || !a.gather_lens || !a.qsl)))))
    return FPR_E_ARG;
  if (!a.S) return FPR_OK;
  k_pf_wbt<<<(unsigned)a.S, 256, 0, (cudaStream_t)a.call.stream>>>(a.si, a.ld_si, a.sl, a.seq, a.wbt, a.ld_wbt, a.wmax, a.wpt,
                                                                    a.seq_lens, a.gather_lens, a.qsl, a.row_lo);
  return rc_of(cudaGetLastError());
}
int c_pf_inv(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprP1PfInvArgs*>(p);
  if (a.n_inv <= 0 || !a.inv || a.S < 0 || a.max_nb < 0 || a.S > 65535 || (a.S && (!a.bt || (!a.nb && !a.seq_lens))))
    return FPR_E_ARG;
  cudaError_t e = cudaMemsetAsync(a.inv, 0xFF, (size_t)a.n_inv * 4, (cudaStream_t)a.call.stream);   // full(-1)
  if (e != cudaSuccess) return FPR_E_CUDA;
  if (!a.S || !a.max_nb) return FPR_OK;
  dim3 grid((unsigned)((a.max_nb + 255) / 256), (unsigned)a.S);
  k_pf_inv<<<grid, 256, 0, (cudaStream_t)a.call.stream>>>(a.inv, a.bt, a.ld_bt, a.nb, a.n_inv, a.seq_lens);
  return rc_of(cudaGetLastError());
}
int c_to_local(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprP1ToLocalArgs*>(p);
  if (a.n < 0 || a.rpp <= 0 || (a.n && (!a.ti || !a.inv || !a.loc))) return FPR_E_ARG;
  const int64_t total = a.n * 512;
  if (!total) return FPR_OK;
  k_to_local<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)a.call.stream>>>(a.ti, a.inv, a.n_inv, a.loc, total, a.rpp);
  return rc_of(cudaGetLastError());
}
int c_carry(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprP1CarryArgs*>(p);
  if (a.IR_T < 0 || a.IR_T % 4 || (a.IR_T && (!a.ir2row_T || !a.valid_T || !a.ix || !a.row2ir_A || !a.cand_A || !a.cand_T)) ||
      (reinterpret_cast<uintptr_t>(a.cand_A) | reinterpret_cast<uintptr_t>(a.cand_T)) % 16)
    return FPR_E_ARG;
  if (!a.IR_T) return FPR_OK;
  k_carry<<<(unsigned)a.IR_T, 128, 0, (cudaStream_t)a.call.stream>>>(a.ir2row_T, a.valid_T, a.ix, a.row2ir_A, a.cand_A, a.cand_T);
  return rc_of(cudaGetLastError());
}

}  // namespace p1k

extern "C" __attribute__((visibility("default"))) const FprExtTable* fpr_p1_table(void) {
  static const FprFnEntry fns[] = {
      {"p1.pf_wbt", p1k::c_pf_wbt, nullptr, static_cast<int64_t>(sizeof(FprP1PfWbtArgs))},
      {"p1.pf_inv", p1k::c_pf_inv, nullptr, static_cast<int64_t>(sizeof(FprP1PfInvArgs))},
      {"p1.to_local", p1k::c_to_local, nullptr, static_cast<int64_t>(sizeof(FprP1ToLocalArgs))},
      {"p1.carry", p1k::c_carry, nullptr, static_cast<int64_t>(sizeof(FprP1CarryArgs))},
  };
  static const FprExtTable t = {FPR_CABI_VERSION, "p1", "new (p1_kernels.cu)", 4, fns};
  return &t;
}

// ---- pybind test entries (p1k_test.py): the same C entries on torch tensors, current stream
namespace {
FprCall call_of(const torch::Tensor& t) {
  FprCall c{};
  c.stream = at::cuda::getCurrentCUDAStream(t.get_device()).stream();
  c.device = t.get_device();
  c.rank = 0;
  return c;
}
void chk(int rc, const char* w) { TORCH_CHECK(rc == FPR_OK, "p1 ", w, ": FPR status ", rc); }
void pf_wbt(torch::Tensor si, torch::Tensor sl, torch::Tensor seq, torch::Tensor wbt, int64_t wmax) {
  FprP1PfWbtArgs a{};
  a.call = call_of(wbt);
  a.si = si.data_ptr<int32_t>(), a.ld_si = si.stride(0), a.sl = sl.data_ptr<int32_t>(), a.seq = seq.data_ptr<int64_t>();
  a.S = seq.size(0), a.wbt = wbt.data_ptr<int32_t>(), a.ld_wbt = wbt.stride(0), a.wmax = wmax, a.wpt = 32;
  chk(p1k::c_pf_wbt(&a), "pf_wbt");
}
void pf_wbt_blob(torch::Tensor si, torch::Tensor sl, torch::Tensor ends, torch::Tensor gls, torch::Tensor qsl, int64_t row_lo,
                 torch::Tensor wbt, int64_t wmax) {
  FprP1PfWbtArgs a{};
  a.call = call_of(wbt);
  a.si = si.data_ptr<int32_t>(), a.ld_si = si.stride(0), a.sl = sl.data_ptr<int32_t>(), a.seq = nullptr;
  a.S = ends.numel(), a.wbt = wbt.data_ptr<int32_t>(), a.ld_wbt = wbt.stride(0), a.wmax = wmax, a.wpt = 32;
  a.seq_lens = ends.data_ptr<int32_t>(), a.gather_lens = gls.data_ptr<int32_t>(), a.qsl = qsl.data_ptr<int32_t>(), a.row_lo = row_lo;
  chk(p1k::c_pf_wbt(&a), "pf_wbt (blob)");
}
void pf_inv_blob(torch::Tensor inv, torch::Tensor bt, torch::Tensor ends, int64_t max_nb) {
  FprP1PfInvArgs a{};
  a.call = call_of(inv);
  a.inv = inv.data_ptr<int32_t>(), a.n_inv = inv.numel(), a.bt = bt.data_ptr<int32_t>(), a.ld_bt = bt.stride(0);
  a.nb = nullptr, a.S = ends.numel(), a.max_nb = max_nb, a.seq_lens = ends.data_ptr<int32_t>();
  chk(p1k::c_pf_inv(&a), "pf_inv (blob)");
}
void pf_inv(torch::Tensor inv, torch::Tensor bt, torch::Tensor nb, int64_t max_nb) {
  FprP1PfInvArgs a{};
  a.call = call_of(inv);
  a.inv = inv.data_ptr<int32_t>(), a.n_inv = inv.numel(), a.bt = bt.data_ptr<int32_t>(), a.ld_bt = bt.stride(0);
  a.nb = nb.data_ptr<int32_t>(), a.S = nb.numel(), a.max_nb = max_nb;
  chk(p1k::c_pf_inv(&a), "pf_inv");
}
void to_local(torch::Tensor ti, torch::Tensor inv, torch::Tensor loc, int64_t rpp) {
  FprP1ToLocalArgs a{};
  a.call = call_of(loc);
  a.ti = ti.data_ptr<int32_t>(), a.inv = inv.data_ptr<int32_t>(), a.n_inv = inv.numel(), a.loc = loc.data_ptr<int32_t>();
  a.n = ti.numel() / 512, a.rpp = rpp;
  chk(p1k::c_to_local(&a), "to_local");
}
void carry(torch::Tensor ir2row_T, torch::Tensor valid_T, torch::Tensor ix, torch::Tensor row2ir_A, torch::Tensor cand_A,
           torch::Tensor cand_T) {
  FprP1CarryArgs a{};
  a.call = call_of(cand_T);
  a.ir2row_T = ir2row_T.data_ptr<int64_t>(), a.valid_T = reinterpret_cast<const uint8_t*>(valid_T.data_ptr());
  a.ix = ix.data_ptr<int64_t>(), a.row2ir_A = row2ir_A.data_ptr<int64_t>(), a.cand_A = cand_A.data_ptr<int32_t>();
  a.cand_T = cand_T.data_ptr<int32_t>(), a.IR_T = ir2row_T.numel();
  chk(p1k::c_carry(&a), "carry");
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pf_wbt", &pf_wbt);
  m.def("pf_inv", &pf_inv);
  m.def("pf_wbt_blob", &pf_wbt_blob);
  m.def("pf_inv_blob", &pf_inv_blob);
  m.def("to_local", &to_local);
  m.def("carry", &carry);
}

// ---- fpr_kwarm(dev): cudaFuncGetAttributes on every __global__ this .so launches (incl. the
// template instances), on device dev (current device restored). Under CUDA_MODULE_LOADING=LAZY that loads the module at
// driver init instead of at the first launch on a pool thread. Host code only: the kernels are unchanged.
extern "C" __attribute__((visibility("default"))) int fpr_kwarm(int dev) {
  int prev = 0;
  if (cudaGetDevice(&prev) != cudaSuccess) return -1;
  if (cudaSetDevice(dev) != cudaSuccess) return -1;
  cudaFuncAttributes fa;
  int bad = 0;
  const void* fns[] = {
      (const void*)&p1k::k_pf_wbt,
      (const void*)&p1k::k_pf_inv,
      (const void*)&p1k::k_to_local,
      (const void*)&p1k::k_carry,
  };
  for (const void* fn : fns) bad += cudaFuncGetAttributes(&fa, fn) != cudaSuccess;
  cudaGetLastError();
  cudaSetDevice(prev);
  return bad ? -1 : (int)(sizeof(fns) / sizeof(fns[0]));   // kernels warmed (p1k)
}
