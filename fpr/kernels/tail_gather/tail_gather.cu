// tail_gather.cu -- FPR_OP_TAIL_GATHER / T_STAGE: the mixed step's
// T stage in ONE launch: every key's tail rows of bucket A into bucket T (was ~13 torch.index_select + copy pairs per rank and
// mixed step) plus T's padding rows (index layers' ti -1 / tl 0, ENG_L20_TAIL's si20 / sl20 from T's si / sl). Byte copies
// of whole rows and 32-bit word fills: bitwise equal to the Python sequence. The pybind entries build the same Args and
// call the same C functions (fpr_cabi.h rule 6). Engine wiring: engine.py Engine._t_stage (ENG_TAIL_GATHER=1 +
// ENG_TAIL_GATHER_KERNEL=1); nd: T_STAGE -> fpr_t_stage_run with the views' FprTgKeys.
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>
#include <cstring>
#include <optional>
#include <string>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "tail_gather_cabi.h"
#include "tail_gather_md5.h"   // TAIL_GATHER_SRC_MD5 (written by tail_gather.py)

namespace {

struct KArgs {   // by value (kernel parameter space, ~1.1 KB): the key and pad tables
  const int64_t* ix;
  int64_t n;
  int32_t n_keys, n_pads;
  const void* src[FPR_TAIL_GATHER_MAX_KEYS];
  void* dst[FPR_TAIL_GATHER_MAX_KEYS];
  int64_t row_bytes[FPR_TAIL_GATHER_MAX_KEYS];
  int64_t src_rows[FPR_TAIL_GATHER_MAX_KEYS];
  const void* pad_src[FPR_TAIL_GATHER_MAX_PADS];
  void* pad_dst[FPR_TAIL_GATHER_MAX_PADS];
  int64_t pad_row_bytes[FPR_TAIL_GATHER_MAX_PADS];
  int64_t pad_rows[FPR_TAIL_GATHER_MAX_PADS];
  int32_t pad_word[FPR_TAIL_GATHER_MAX_PADS];
  int32_t* err;
};

// one row: 16-byte vectors when both rows and the length are 16-byte aligned, else 4-byte words when 4-aligned, else bytes
__device__ __forceinline__ void copy_row(const char* sp, char* dp, int64_t rb) {
  const uintptr_t al = reinterpret_cast<uintptr_t>(sp) | reinterpret_cast<uintptr_t>(dp) | static_cast<uintptr_t>(rb);
  if ((al & 15) == 0) {
    const uint4* s4 = reinterpret_cast<const uint4*>(sp);
    uint4* d4 = reinterpret_cast<uint4*>(dp);
    for (int64_t j = threadIdx.x; j < rb / 16; j += blockDim.x) d4[j] = s4[j];
  } else if ((al & 3) == 0) {
    const uint32_t* s1 = reinterpret_cast<const uint32_t*>(sp);
    uint32_t* d1 = reinterpret_cast<uint32_t*>(dp);
    for (int64_t j = threadIdx.x; j < rb / 4; j += blockDim.x) d1[j] = s1[j];
  } else {
    for (int64_t j = threadIdx.x; j < rb; j += blockDim.x) dp[j] = sp[j];
  }
}

__device__ __forceinline__ void fill_row(char* dp, int64_t rb, uint32_t w) {
  const uintptr_t al = reinterpret_cast<uintptr_t>(dp) | static_cast<uintptr_t>(rb);
  if ((al & 15) == 0) {
    const uint4 v = make_uint4(w, w, w, w);
    uint4* d4 = reinterpret_cast<uint4*>(dp);
    for (int64_t j = threadIdx.x; j < rb / 16; j += blockDim.x) d4[j] = v;
  } else if ((al & 3) == 0) {
    uint32_t* d1 = reinterpret_cast<uint32_t*>(dp);
    for (int64_t j = threadIdx.x; j < rb / 4; j += blockDim.x) d1[j] = w;
  } else {   // (fpr_tail_gather_run rejects a fill whose row is not a multiple of 4 bytes; kept for completeness)
    for (int64_t j = threadIdx.x; j < rb; j += blockDim.x) dp[j] = static_cast<char>((w >> (8 * ((reinterpret_cast<uintptr_t>(dp) + j) & 3))) & 0xff);
  }
}

// grid (rows, n_keys + n_pads): block (i, k) writes row i of key k (gather, i < n) or of pad k - n_keys (n <= i < pad_rows)
__global__ void __launch_bounds__(256) k_tail_gather(const __grid_constant__ KArgs a) {
  const int64_t i = blockIdx.x;
  const int k = blockIdx.y;
  if (k < a.n_keys) {
    if (i >= a.n) return;
    const int64_t s = a.ix[i];
    const int64_t rb = a.row_bytes[k];
    if (s < 0 || s >= a.src_rows[k]) {
      if (threadIdx.x == 0 && a.err != nullptr) atomicAdd(a.err, 1);
      return;
    }
    copy_row(static_cast<const char*>(a.src[k]) + s * rb, static_cast<char*>(a.dst[k]) + i * rb, rb);
    return;
  }
  const int j = k - a.n_keys;
  if (j >= a.n_pads || i < a.n || i >= a.pad_rows[j]) return;
  const int64_t rb = a.pad_row_bytes[j];
  char* dp = static_cast<char*>(a.pad_dst[j]) + i * rb;
  if (a.pad_src[j] != nullptr)
    copy_row(static_cast<const char*>(a.pad_src[j]) + i * rb, dp, rb);
  else
    fill_row(dp, rb, static_cast<uint32_t>(a.pad_word[j]));
}

}  // namespace

extern "C" int fpr_tail_gather_run(const FprTailGatherArgs* a) {
  if (a == nullptr || a->n < 0 || a->n_keys < 0 || a->n_keys > FPR_TAIL_GATHER_MAX_KEYS || a->n_pads < 0 ||
      a->n_pads > FPR_TAIL_GATHER_MAX_PADS || a->n > 2147483647LL)
    return FPR_E_ARG;
  if (a->n > 0 && a->n_keys > 0 && a->ix == nullptr) return FPR_E_ARG;
  KArgs k{};
  k.ix = a->ix, k.n = a->n, k.n_keys = a->n_keys, k.n_pads = a->n_pads, k.err = a->err;
  int64_t rows = a->n_keys > 0 ? a->n : 0;
  for (int j = 0; j < a->n_keys; ++j) {
    if (a->src[j] == nullptr || a->dst[j] == nullptr || a->row_bytes[j] <= 0 || a->src_rows[j] < 0) return FPR_E_ARG;
    k.src[j] = a->src[j], k.dst[j] = a->dst[j], k.row_bytes[j] = a->row_bytes[j], k.src_rows[j] = a->src_rows[j];
  }
  for (int j = 0; j < a->n_pads; ++j) {
    if (a->pad_dst[j] == nullptr || a->pad_row_bytes[j] <= 0 || a->pad_rows[j] < 0 || a->pad_rows[j] > 2147483647LL ||
        (a->pad_src[j] == nullptr && a->pad_row_bytes[j] % 4 != 0))
      return FPR_E_ARG;
    k.pad_src[j] = a->pad_src[j], k.pad_dst[j] = a->pad_dst[j], k.pad_row_bytes[j] = a->pad_row_bytes[j];
    k.pad_rows[j] = a->pad_rows[j], k.pad_word[j] = a->pad_word[j];
    rows = std::max(rows, a->pad_rows[j]);
  }
  const int ny = a->n_keys + a->n_pads;
  if (rows == 0 || ny == 0) return FPR_OK;
  const dim3 grid((unsigned)rows, (unsigned)ny);
  k_tail_gather<<<grid, 256, 0, a->call.stream>>>(k);
  return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}

// Engine._tail_gather's key order (l20: X, a_pre, a_post, a_comb, ws.q, si -> si20, sl -> sl20; else moe_out, xm2, f_post,
// f_comb, f_pre; then ti / tl of each index layer <= 20) and step_mixed's T padding (l20: si20 / sl20 rows [n, R) from T's
// si / sl; clear: ti rows [n, R) = -1, tl = 0)
extern "C" int fpr_t_stage_args(const FprTStageArgs* t, FprTailGatherArgs* out) {
  if (t == nullptr || out == nullptr || t->A == nullptr || t->T == nullptr || t->n < 0) return FPR_E_ARG;
  const FprTgKeys& A = *t->A;
  const FprTgKeys& T = *t->T;
  if (A.version != FPR_TG_KEYS_VERSION || T.version != FPR_TG_KEYS_VERSION || A.n_idx != T.n_idx || A.n_idx < 0 ||
      A.n_idx > FPR_TG_MAX_IDX)
    return FPR_E_ARG;
  *out = FprTailGatherArgs{};
  out->call = t->call, out->ix = t->ix, out->n = t->n, out->err = t->err;
  bool ok = true;
  auto key = [&](int sa, int st) {
    if (A.p[sa] == nullptr || T.p[st] == nullptr || A.row_bytes[sa] != T.row_bytes[st] || A.row_bytes[sa] <= 0 ||
        T.rows[st] < t->n || out->n_keys >= FPR_TAIL_GATHER_MAX_KEYS) {
      ok = false;
      return;
    }
    const int j = out->n_keys++;
    out->src[j] = A.p[sa], out->dst[j] = T.p[st], out->row_bytes[j] = A.row_bytes[sa], out->src_rows[j] = A.rows[sa];
  };
  auto pad = [&](int ss, int sd, bool fill, int32_t word) {
    if (T.p[sd] == nullptr || T.row_bytes[sd] <= 0 || out->n_pads >= FPR_TAIL_GATHER_MAX_PADS ||
        (!fill && (T.p[ss] == nullptr || T.row_bytes[ss] != T.row_bytes[sd] || T.rows[ss] < T.rows[sd]))) {
      ok = false;
      return;
    }
    const int j = out->n_pads++;
    out->pad_src[j] = fill ? nullptr : T.p[ss], out->pad_dst[j] = T.p[sd], out->pad_row_bytes[j] = T.row_bytes[sd];
    out->pad_rows[j] = T.rows[sd], out->pad_word[j] = fill ? word : 0;
  };
  if (t->l20) {
    for (int s : {FPR_TG_X, FPR_TG_A_PRE, FPR_TG_A_POST, FPR_TG_A_COMB, FPR_TG_WS_Q}) key(s, s);
    key(FPR_TG_SI, FPR_TG_SI20);
    key(FPR_TG_SL, FPR_TG_SL20);
  } else {
    for (int s : {FPR_TG_MOE_OUT, FPR_TG_XM2, FPR_TG_F_POST, FPR_TG_F_COMB, FPR_TG_F_PRE}) key(s, s);
  }
  for (int j = 0; j < A.n_idx; ++j) {
    key(FPR_TG_TI0 + j, FPR_TG_TI0 + j);
    key(FPR_TG_TL0 + j, FPR_TG_TL0 + j);
  }
  if (t->l20) {
    pad(FPR_TG_SI, FPR_TG_SI20, false, 0);
    pad(FPR_TG_SL, FPR_TG_SL20, false, 0);
  }
  if (t->clear) {
    for (int j = 0; j < T.n_idx; ++j) {
      pad(-1, FPR_TG_TI0 + j, true, -1);
      pad(-1, FPR_TG_TL0 + j, true, 0);
    }
  }
  return ok ? FPR_OK : FPR_E_ARG;
}

extern "C" int fpr_t_stage_run(const FprTStageArgs* t) {
  FprTailGatherArgs a;
  const int rc = fpr_t_stage_args(t, &a);
  return rc != FPR_OK ? rc : fpr_tail_gather_run(&a);
}

namespace {

int c_run(const void* a) { return fpr_tail_gather_run(static_cast<const FprTailGatherArgs*>(a)); }
int c_t_stage(const void* a) { return fpr_t_stage_run(static_cast<const FprTStageArgs*>(a)); }

// pybind (rule 6): the same Args, the same C function, on the current stream of ix's device. Every key: a contiguous
// row-major src / dst on ix's device with the same bytes per row; dst holds >= n rows. err: optional int32 [1] counter.
void gather(torch::Tensor ix, std::vector<torch::Tensor> srcs, std::vector<torch::Tensor> dsts, std::optional<torch::Tensor> err) {
  TORCH_CHECK(ix.is_cuda() && ix.scalar_type() == torch::kInt64 && ix.dim() == 1 && ix.is_contiguous(), "tail_gather: ix");
  TORCH_CHECK(srcs.size() == dsts.size() && (int64_t)srcs.size() <= FPR_TAIL_GATHER_MAX_KEYS, "tail_gather: keys");
  const c10::cuda::CUDAGuard g(ix.device());
  FprTailGatherArgs a{};
  a.call.stream = at::cuda::getCurrentCUDAStream();
  a.call.device = ix.get_device();
  a.ix = ix.data_ptr<int64_t>();
  a.n = ix.numel();
  a.n_keys = (int32_t)srcs.size();
  for (size_t j = 0; j < srcs.size(); ++j) {
    const auto& s = srcs[j];
    const auto& d = dsts[j];
    TORCH_CHECK(s.is_cuda() && d.is_cuda() && s.get_device() == a.call.device && d.get_device() == a.call.device,
                "tail_gather: key ", j, " device");
    TORCH_CHECK(s.dim() >= 1 && d.dim() >= 1 && s.is_contiguous() && d.is_contiguous() && s.scalar_type() == d.scalar_type(),
                "tail_gather: key ", j, " layout");
    const int64_t rb = s.numel() / std::max<int64_t>(s.size(0), 1) * s.element_size();
    const int64_t rbd = d.numel() / std::max<int64_t>(d.size(0), 1) * d.element_size();
    TORCH_CHECK(rb == rbd && d.size(0) >= a.n, "tail_gather: key ", j, " row bytes / rows");
    a.src[j] = s.data_ptr(), a.dst[j] = d.data_ptr(), a.row_bytes[j] = rb, a.src_rows[j] = s.size(0);
  }
  if (err.has_value()) {
    TORCH_CHECK(err->is_cuda() && err->scalar_type() == torch::kInt32 && err->get_device() == a.call.device, "tail_gather: err");
    a.err = err->data_ptr<int32_t>();
  }
  const int rc = fpr_tail_gather_run(&a);
  TORCH_CHECK(rc == FPR_OK, "tail_gather: rc ", rc);
}

// the T stage from two packed FprTgKeys (tail_gather.py keys_bytes / Engine._tg_keys: the bytes the exporter publishes), on
// the current stream of `device`; ix None only with n == 0
void t_stage(std::optional<torch::Tensor> ix, int64_t n, const std::string& a_keys, const std::string& t_keys, bool l20,
             bool clear, int64_t device, std::optional<torch::Tensor> err) {
  TORCH_CHECK(a_keys.size() == sizeof(FprTgKeys) && t_keys.size() == sizeof(FprTgKeys), "tail_gather: FprTgKeys is ",
              sizeof(FprTgKeys), " bytes, got ", a_keys.size(), " / ", t_keys.size(), " (ctypes / header drift)");
  FprTgKeys A, T;
  std::memcpy(&A, a_keys.data(), sizeof A);
  std::memcpy(&T, t_keys.data(), sizeof T);
  const c10::cuda::CUDAGuard g((c10::DeviceIndex)device);
  FprTStageArgs t{};
  t.call.stream = at::cuda::getCurrentCUDAStream((c10::DeviceIndex)device);
  t.call.device = (int32_t)device;
  if (ix.has_value()) {
    TORCH_CHECK(ix->is_cuda() && ix->get_device() == device && ix->scalar_type() == torch::kInt64 && ix->dim() == 1 &&
                ix->is_contiguous() && ix->numel() >= n, "tail_gather: t_stage ix");
    t.ix = ix->data_ptr<int64_t>();
  } else {
    TORCH_CHECK(n == 0, "tail_gather: t_stage ix None with n ", n);
  }
  t.n = n, t.l20 = l20, t.clear = clear, t.A = &A, t.T = &T;
  if (err.has_value()) {
    TORCH_CHECK(err->is_cuda() && err->scalar_type() == torch::kInt32 && err->get_device() == device, "tail_gather: err");
    t.err = err->data_ptr<int32_t>();
  }
  const int rc = fpr_t_stage_run(&t);
  TORCH_CHECK(rc == FPR_OK, "tail_gather: t_stage rc ", rc);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) const FprExtTable* fpr_tail_gather_table(void) {
  static const FprFnEntry fns[] = {
      {"tail_gather.run", c_run, nullptr, static_cast<int64_t>(sizeof(FprTailGatherArgs))},
      {"tail_gather.t_stage", c_t_stage, nullptr, static_cast<int64_t>(sizeof(FprTStageArgs))},
  };
  static const FprExtTable t = {FPR_CABI_VERSION, "tail_gather", TAIL_GATHER_SRC_MD5, 2, fns};
  return &t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather", &gather, "FPR_OP_TAIL_GATHER: dst[k][i] = src[k][ix[i]] for every key, one launch",
        pybind11::arg("ix"), pybind11::arg("srcs"), pybind11::arg("dsts"), pybind11::arg("err") = pybind11::none());
  m.def("t_stage", &t_stage, "T_STAGE: fpr_t_stage_run on two packed FprTgKeys (gather + padding, one launch)",
        pybind11::arg("ix"), pybind11::arg("n"), pybind11::arg("a_keys"), pybind11::arg("t_keys"), pybind11::arg("l20"),
        pybind11::arg("clear"), pybind11::arg("device"), pybind11::arg("err") = pybind11::none());
  m.def("args_size", []() { return (int64_t)sizeof(FprTailGatherArgs); });
  m.def("tstage_args_size", []() { return (int64_t)sizeof(FprTStageArgs); });
  m.def("keys_size", []() { return (int64_t)sizeof(FprTgKeys); });
}
