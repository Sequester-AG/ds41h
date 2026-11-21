// route2_cabi.cu -- extension moe_v4c_route: moe_v4's route2.cu #included UNCHANGED + the C ABI of moe_v4_cabi.h for route2r
// + thin pybind callers. Device code = the served build's; host code added only.
// route2.cu's PYBIND11_MODULE is renamed (push_macro) to r2_bind and called from this module's init.
//
// moe_v4r_route2r: the served route2r statements (checks, the k_one launch config and arguments) on Args.stream. Served
// host state -> init / arguments: the k_fill / k_one smem attributes (served: g_attr_dev, re-set whenever the device
// changes) and the SM count (g_sms) -> moe_v4r.init per device; g_pdl / g_one_max_t (route2_config) -> Args.pdl / one_max_t.
#include <torch/extension.h>

#include <atomic>

#include "fpr_cabi.h"
#include "moe_v4_cabi.h"

#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) static void r2_bind(pybind11::module_& variable)
#include "route2.cu"
#pragma pop_macro("PYBIND11_MODULE")

#if __has_include("moe_v4c_md5.h")
#include "moe_v4c_md5.h"
#endif
#ifndef MOE_V4C_SRC_MD5
#define MOE_V4C_SRC_MD5 "unset"
#endif

namespace {
constexpr int R2C_MAXDEV = 32;
std::atomic<uint32_t> g_r2c_mask{0};
int g_r2c_sms[R2C_MAXDEV] = {};   // written by moe_v4r_init only
}  // namespace

extern "C" {

int64_t moe_v4r_max_ec(int64_t T) { return T > 0 ? int64_t(max_ec((int)T * TOPK)) : FPR_E_ARG; }

int moe_v4r_init(const void* p) {
  const auto* a = static_cast<const MoeV4InitArgs*>(p);
  if (a == nullptr || a->device < 0 || a->device >= R2C_MAXDEV) return FPR_E_ARG;
  int cur = -1;
  if (cudaGetDevice(&cur) != cudaSuccess || cur != a->device) return FPR_E_ARG;
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a->device) != cudaSuccess || sms <= 0) return FPR_E_CUDA;
  if (cudaFuncSetAttribute(k_fill, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(MAXG * NEXP * sizeof(unsigned short))) !=
          cudaSuccess ||
      cudaFuncSetAttribute(k_one, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(ONE_MAXS * NEXP * sizeof(int))) != cudaSuccess)
    return FPR_E_CUDA;
  g_r2c_sms[a->device] = sms;
  g_r2c_mask.fetch_or(1u << a->device);
  return FPR_OK;
}

int moe_v4r_route2r(const void* p) {
  const auto* a = static_cast<const MoeV4Route2rArgs*>(p);
  if (a == nullptr) return FPR_E_ARG;
  const int dev = a->call.device;
  if (dev < 0 || dev >= R2C_MAXDEV || !(g_r2c_mask.load() & (1u << dev)) || g_r2c_sms[dev] <= 0) return FPR_E_INIT;
  if (a->one_max_t < 0 || int64_t(a->one_max_t) * TOPK > int64_t(ONE_MAXS) * ONE_SEG) return FPR_E_ARG;   // route2_config's check
  if (!(a->T > 0 && a->T <= a->one_max_t) || !a->rows || !a->ecs || !a->t1 || !a->t2 || !a->counts || !a->slotmap || a->grid < 0)
    return FPR_E_ARG;
  const int T = int(a->T), P = T * TOPK, mec = max_ec(P);
  if (a->ecs_bytes < int64_t(mec) * int64_t(sizeof(EC)) || a->t1_bytes < int64_t(mec) * 9 * int64_t(sizeof(Tile)) ||
      a->t2_bytes < int64_t(mec) * 40 * int64_t(sizeof(Tile)) || a->counts_n < 6 || a->slotmap_n < P)
    return FPR_E_WS;   // route2r: buffers
  const int sms = g_r2c_sms[dev];
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  const int sg = a->grid > 0 ? (int)a->grid : sms;
  const int S = (P + ONE_SEG - 1) / ONE_SEG, nE = mec * (NMAX + 49);
  cudaLaunchConfig_t c0 = {};
  c0.gridDim = dim3(std::max(S, std::min((nE + F_THREADS - 1) / F_THREADS, sms)));
  c0.blockDim = dim3(F_THREADS);
  c0.dynamicSmemBytes = (size_t)S * NEXP * sizeof(int);
  c0.stream = a->call.stream;
  c0.attrs = at;
  c0.numAttrs = (a->pdl & 2) ? 1 : 0;
  const int* r32 = (const int*)a->rows;
  if (cudaLaunchKernelEx(&c0, k_one, r32, (const float*)(r32 + 6), (const float*)nullptr, P, S, mec, (EC*)a->ecs, (Tile*)a->t1,
                         (Tile*)a->t2, a->counts, a->slotmap, sg, (int*)nullptr, 16) != cudaSuccess)
    return FPR_E_CUDA;
  return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}

static const FprFnEntry kR2Fns[] = {
    {"moe_v4r.init", moe_v4r_init, nullptr, int64_t(sizeof(MoeV4InitArgs))},
    {"moe_v4r.route2r", moe_v4r_route2r, nullptr, int64_t(sizeof(MoeV4Route2rArgs))},
};
static const FprExtTable kR2Table = {FPR_CABI_VERSION, "moe_v4r", MOE_V4C_SRC_MD5, int(sizeof(kR2Fns) / sizeof(kR2Fns[0])), kR2Fns};

__attribute__((visibility("default"))) const FprExtTable* fpr_moe_v4r_table(void) { return &kR2Table; }

}  // extern "C"

// ---- pybind: thin callers of the same C functions
namespace {
void py_init(int64_t device) {
  MoeV4InitArgs a{};
  a.device = int32_t(device);
  const int rc = moe_v4r_init(&a);
  TORCH_CHECK(rc == FPR_OK, "moe_v4r.init: error ", rc);
}

void py_route2r(torch::Tensor rows, int64_t T, torch::Tensor ecs, torch::Tensor t1, torch::Tensor t2, torch::Tensor counts,
                torch::Tensor slotmap, int64_t grid, int64_t pdl, int64_t one_max_t) {
  TORCH_CHECK(rows.is_cuda() && rows.nbytes() >= (size_t)T * 64, "route2r: rows [T, 64 B]");
  TORCH_CHECK(counts.dtype() == torch::kInt32 && slotmap.dtype() == torch::kInt32, "route2r: counts / slotmap int32");
  MoeV4Route2rArgs a{};
  a.call.stream = at::cuda::getCurrentCUDAStream(rows.get_device()).stream(), a.call.device = int(rows.get_device());
  a.rows = rows.data_ptr(), a.T = T;
  a.ecs = ecs.data_ptr(), a.t1 = t1.data_ptr(), a.t2 = t2.data_ptr(), a.counts = counts.data_ptr<int>(),
  a.slotmap = slotmap.data_ptr<int>();
  a.ecs_bytes = int64_t(ecs.nbytes()), a.t1_bytes = int64_t(t1.nbytes()), a.t2_bytes = int64_t(t2.nbytes());
  a.counts_n = counts.numel(), a.slotmap_n = slotmap.numel();
  a.grid = grid, a.pdl = int32_t(pdl), a.one_max_t = int32_t(one_max_t);
  const int rc = moe_v4r_route2r(&a);
  TORCH_CHECK(rc == FPR_OK, "moe_v4r.route2r: error ", rc);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  r2_bind(m);   // route2.cu's module body (the served bindings)
  m.def("cabi_init", &py_init, "moe_v4r.init on the current device");
  m.def("cabi_route2r", &py_route2r, "moe_v4r.route2r through the C ABI (pdl / one_max_t: route2_config's values)");
  m.def("cabi_max_ec", &moe_v4r_max_ec);
  m.def("cabi_args_sizes", []() { return std::vector<int64_t>{int64_t(sizeof(MoeV4InitArgs)), int64_t(sizeof(MoeV4Route2rArgs))}; });
}
