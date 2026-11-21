// moe_v4_cabi.cu -- extension moe_v4c_api: moe_v4.cu #included UNCHANGED (its own translation unit
// as the served moe_v4_api build; production's defines from v4_cabi.py) + the C ABI of moe_v4_cabi.h for gemm1r / gemm2r
// + thin pybind callers of the same C functions. Device code = the served build's
// only host code is added. moe_v4.cu's PYBIND11_MODULE is renamed (push_macro) to v4_bind and called from this module's
// init, so every served binding is still there.
//
// moe_v4_gemm1r / gemm2r: the served gemm1r / gemm2r statements (Args<MODE> fields, the same tmap2d tensor maps) and
// launch<MODE>'s launch (grid, block 32 + 32 NCW, Lay<MODE>::SMEM, the MS4_PDL launch form) on Args.stream; the smem
// attribute and the SM count come from moe_v4.init (served: set on the first launch / first call).
#include <torch/extension.h>

#include <atomic>

#include "fpr_cabi.h"
#include "moe_v4_cabi.h"

#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) static void v4_bind(pybind11::module_& variable)
#include "moe_v4.cu"
#pragma pop_macro("PYBIND11_MODULE")

#if __has_include("moe_v4c_md5.h")
#include "moe_v4c_md5.h"   // written by v4_cabi.py into the build dir: md5 of the sources of this build
#endif
#ifndef MOE_V4C_SRC_MD5
#define MOE_V4C_SRC_MD5 "unset"
#endif

namespace {
constexpr int V4C_MAXDEV = 32;
std::atomic<uint32_t> g_v4c_mask{0};
int g_v4c_sms[V4C_MAXDEV] = {};   // written by moe_v4_init only

template <int MODE>
int v4c_launch(Args<MODE>& a, int64_t grid, cudaStream_t stream, int sms) {
  if ((MS4_PDL == 1 && MODE == 2) || MS4_PDL == 2) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid > 0 ? (unsigned)grid : (unsigned)sms);
    cfg.blockDim = dim3(32 + 32 * NCW);
    cfg.dynamicSmemBytes = Lay<MODE>::SMEM;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (cudaLaunchKernelEx(&cfg, k_gemm<MODE>, a) != cudaSuccess) return FPR_E_CUDA;
  } else {
    k_gemm<MODE><<<grid > 0 ? (int)grid : sms, 32 + 32 * NCW, Lay<MODE>::SMEM, stream>>>(a);
  }
  return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}

bool v4c_ready(int dev) { return dev >= 0 && dev < V4C_MAXDEV && (g_v4c_mask.load() & (1u << dev)) && g_v4c_sms[dev] > 0; }
}  // namespace

extern "C" {

int moe_v4_init(const void* p) {
  const auto* a = static_cast<const MoeV4InitArgs*>(p);
  if (a == nullptr || a->device < 0 || a->device >= V4C_MAXDEV) return FPR_E_ARG;
  int cur = -1;
  if (cudaGetDevice(&cur) != cudaSuccess || cur != a->device) return FPR_E_ARG;   // the caller made `device` current
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a->device) != cudaSuccess || sms <= 0) return FPR_E_CUDA;
  if (cudaFuncSetAttribute(k_gemm<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, Lay<1>::SMEM) != cudaSuccess ||
      cudaFuncSetAttribute(k_gemm<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, Lay<2>::SMEM) != cudaSuccess)
    return FPR_E_CUDA;
  try {
    (void)tmap_encode();   // the driver entry point, resolved once (served: on the first tensor map)
  } catch (const std::exception&) {
    return FPR_E_CUDA;
  }
  g_v4c_sms[a->device] = sms;
  g_v4c_mask.fetch_or(1u << a->device);
  return FPR_OK;
}

int moe_v4_gemm1r(const void* p) {
  const auto* a = static_cast<const MoeV4Gemm1rArgs*>(p);
  if (a == nullptr) return FPR_E_ARG;
  if (!v4c_ready(a->call.device)) return FPR_E_INIT;
  if (!a->w13 || !a->s13 || a->E < 1 || !a->ecs || !a->tiles || !a->counts || !a->img || !a->himg || a->grid < 0) return FPR_E_ARG;
  try {
    Args<1> g{};
    g.ecs = (const EC*)a->ecs, g.tiles = (const Tile*)a->tiles, g.bimg = (const char*)a->img;
    g.himg = (char*)a->himg, g.nt_dev = const_cast<int*>(a->counts) + 1;
    g.w2 = (const unsigned char*)a->w13;  // GEMM1: raw w13 (only the MS3_ABL 9 timing ablation reads it directly)
    const uint64_t rows = (uint64_t)a->E * 1152;
    void* w13 = const_cast<uint8_t*>(a->w13);
    void* s13 = const_cast<uint8_t*>(a->s13);
    g.tmw = tmap2d(w13, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 2, rows, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    g.tms = tmap2d(s13, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 32, rows, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE);
    g.tmw2 = tmap2d(w13, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 2, rows, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    g.tms2 = tmap2d(s13, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, D / 32, rows, 16, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
    return v4c_launch<1>(g, a->grid, a->call.stream, g_v4c_sms[a->call.device]);
  } catch (const std::exception&) {
    return FPR_E_ARG;   // cuTensorMapEncodeTiled rejected the operands
  }
}

int moe_v4_gemm2r(const void* p) {
  const auto* a = static_cast<const MoeV4Gemm2rArgs*>(p);
  if (a == nullptr) return FPR_E_ARG;
  if (!v4c_ready(a->call.device)) return FPR_E_INIT;
  if (!a->w2 || !a->s2 || a->E < 1 || !a->ecs || !a->tiles || !a->counts || !a->himg || !a->yout || a->grid < 0) return FPR_E_ARG;
  try {
    Args<2> g{};
    g.s2 = (const unsigned char*)a->s2, g.ecs = (const EC*)a->ecs, g.tiles = (const Tile*)a->tiles;
    g.w2 = (const unsigned char*)a->w2;
    g.bimg = (const char*)a->himg, g.nt_dev = const_cast<int*>(a->counts) + 2;
    g.yout = (__nv_bfloat16*)a->yout;
    void* w2 = const_cast<uint8_t*>(a->w2);
    g.tmw = tmap2d(w2, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, IR / 4, (uint64_t)a->E * D, 48, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
    g.tmw2 = tmap2d(w2, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, IR / 4, (uint64_t)a->E * D, 48, 256, CU_TENSOR_MAP_SWIZZLE_NONE);
    return v4c_launch<2>(g, a->grid, a->call.stream, g_v4c_sms[a->call.device]);
  } catch (const std::exception&) {
    return FPR_E_ARG;
  }
}

// the served swiglu / prepr6 / combine statements on Args.stream
int moe_v4_swiglu(const void* p) {
  const auto* a = static_cast<const MoeV4SwigluArgs*>(p);
  if (a == nullptr || !a->gu || !a->h || a->T < 0 || a->I <= 0 || a->I % 8) return FPR_E_ARG;
  if (a->T == 0) return FPR_OK;
  k_swiglu<<<(unsigned)a->T, a->I / 8, 0, a->call.stream>>>((const __nv_bfloat16*)a->gu, (__nv_bfloat16*)a->h, a->I);
  return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}

int moe_v4_prepr6(const void* p) {
  const auto* a = static_cast<const MoeV4Prepr6Args*>(p);
  if (a == nullptr || !a->x || !a->ecs || !a->counts || !a->img || !a->ptok || !a->inv || a->T <= 0 || a->sms <= 0) return FPR_E_ARG;
  k_prescale2<<<a->T, 640, 0, a->call.stream>>>((const __nv_bfloat16*)a->x, a->ptok, a->inv);
  if (cudaGetLastError() != cudaSuccess) return FPR_E_CUDA;
  k_gather5<<<(unsigned)(MSR_G5B * a->sms), 640, 0, a->call.stream>>>((const __nv_bfloat16*)a->x, (EC*)a->ecs, a->ptok, (char*)a->img,
                                                                       const_cast<int*>(a->counts), 1);
  return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}

int moe_v4_combine(const void* p) {
  const auto* a = static_cast<const MoeV4CombineArgs*>(p);
  if (a == nullptr || !a->Y || !a->slotmap || !a->out || a->T < 0) return FPR_E_ARG;
  if (a->T == 0) return FPR_OK;
  k_combine<<<(unsigned)a->T, D / 8, 0, a->call.stream>>>((const __nv_bfloat16*)a->Y, const_cast<int*>(a->slotmap),
                                                           (const __nv_bfloat16*)a->sh, (__nv_bfloat16*)a->out);
  return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}

static const FprFnEntry kV4Fns[] = {
    {"moe_v4.init", moe_v4_init, nullptr, int64_t(sizeof(MoeV4InitArgs))},
    {"moe_v4.gemm1r", moe_v4_gemm1r, nullptr, int64_t(sizeof(MoeV4Gemm1rArgs))},
    {"moe_v4.gemm2r", moe_v4_gemm2r, nullptr, int64_t(sizeof(MoeV4Gemm2rArgs))},
    {"moe_v4.swiglu", moe_v4_swiglu, nullptr, int64_t(sizeof(MoeV4SwigluArgs))},
    {"moe_v4.prepr6", moe_v4_prepr6, nullptr, int64_t(sizeof(MoeV4Prepr6Args))},
    {"moe_v4.combine", moe_v4_combine, nullptr, int64_t(sizeof(MoeV4CombineArgs))},
};
static const FprExtTable kV4Table = {FPR_CABI_VERSION, "moe_v4", MOE_V4C_SRC_MD5, int(sizeof(kV4Fns) / sizeof(kV4Fns[0])), kV4Fns};

__attribute__((visibility("default"))) const FprExtTable* fpr_moe_v4_table(void) { return &kV4Table; }

}  // extern "C"

// ---- pybind: thin callers of the same C functions
namespace {
cudaStream_t cur_stream(const torch::Tensor& t) { return at::cuda::getCurrentCUDAStream(t.get_device()).stream(); }

void py_init(int64_t device) {
  MoeV4InitArgs a{};
  a.device = int32_t(device);
  const int rc = moe_v4_init(&a);
  TORCH_CHECK(rc == FPR_OK, "moe_v4.init: error ", rc);
}

void py_gemm1r(torch::Tensor w13, torch::Tensor s13, torch::Tensor ecs, torch::Tensor tiles, torch::Tensor counts, torch::Tensor img,
               torch::Tensor himg, int64_t grid) {
  TORCH_CHECK(w13.size(1) == 1152 && w13.size(2) == D / 2 && s13.size(2) == D / 32, "gemm1r: raw w13 / s13 shapes");
  MoeV4Gemm1rArgs a{};
  a.call.stream = cur_stream(w13), a.call.device = int(w13.get_device()), a.call.rank = 0;
  a.w13 = (const uint8_t*)w13.data_ptr(), a.s13 = (const uint8_t*)s13.data_ptr(), a.E = w13.size(0);
  a.ecs = ecs.data_ptr(), a.tiles = tiles.data_ptr(), a.counts = counts.data_ptr<int>(), a.img = img.data_ptr();
  a.himg = himg.data_ptr(), a.grid = grid;
  const int rc = moe_v4_gemm1r(&a);
  TORCH_CHECK(rc == FPR_OK, "moe_v4.gemm1r: error ", rc);
}

void py_gemm2r(torch::Tensor w2, torch::Tensor s2, torch::Tensor ecs, torch::Tensor tiles, torch::Tensor counts, torch::Tensor himg,
               torch::Tensor yout, int64_t grid) {
  TORCH_CHECK(w2.size(1) == D && w2.size(2) == IR / 2 && s2.size(2) == IR / 32, "gemm2r: raw w2 / s2 shapes");
  MoeV4Gemm2rArgs a{};
  a.call.stream = cur_stream(w2), a.call.device = int(w2.get_device()), a.call.rank = 0;
  a.w2 = (const uint8_t*)w2.data_ptr(), a.s2 = (const uint8_t*)s2.data_ptr(), a.E = w2.size(0);
  a.ecs = ecs.data_ptr(), a.tiles = tiles.data_ptr(), a.counts = counts.data_ptr<int>(), a.himg = himg.data_ptr();
  a.yout = yout.data_ptr(), a.grid = grid;
  const int rc = moe_v4_gemm2r(&a);
  TORCH_CHECK(rc == FPR_OK, "moe_v4.gemm2r: error ", rc);
}
void py_swiglu(torch::Tensor gu, torch::Tensor h) {
  TORCH_CHECK(gu.size(1) == 2 * h.size(1), "swiglu shapes");
  MoeV4SwigluArgs a{};
  a.call.stream = cur_stream(h), a.call.device = int(h.get_device());
  a.gu = gu.data_ptr(), a.h = h.data_ptr(), a.T = int32_t(h.size(0)), a.I = int32_t(h.size(1));
  const int rc = moe_v4_swiglu(&a);
  TORCH_CHECK(rc == FPR_OK, "moe_v4.swiglu: error ", rc);
}

void py_prepr6(torch::Tensor x, torch::Tensor ecs, torch::Tensor counts, torch::Tensor img, torch::Tensor ptok, torch::Tensor inv,
               int64_t sms) {
  TORCH_CHECK(x.size(1) == D && ptok.numel() >= x.size(0) && inv.numel() >= x.size(0), "prepr6: shapes");
  MoeV4Prepr6Args a{};
  a.call.stream = cur_stream(x), a.call.device = int(x.get_device());
  a.x = x.data_ptr(), a.ecs = ecs.data_ptr(), a.counts = counts.data_ptr<int>(), a.img = img.data_ptr();
  a.ptok = ptok.data_ptr<int>(), a.inv = inv.data_ptr<float>(), a.T = int32_t(x.size(0)), a.sms = int32_t(sms);
  const int rc = moe_v4_prepr6(&a);
  TORCH_CHECK(rc == FPR_OK, "moe_v4.prepr6: error ", rc);
}

void py_combine(torch::Tensor Y, torch::Tensor slotmap, torch::Tensor sh, torch::Tensor out) {
  MoeV4CombineArgs a{};
  a.call.stream = cur_stream(out), a.call.device = int(out.get_device());
  a.Y = Y.data_ptr(), a.slotmap = slotmap.data_ptr<int>(), a.sh = sh.numel() ? sh.data_ptr() : nullptr, a.out = out.data_ptr();
  a.T = int32_t(out.size(0));
  const int rc = moe_v4_combine(&a);
  TORCH_CHECK(rc == FPR_OK, "moe_v4.combine: error ", rc);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  v4_bind(m);   // moe_v4.cu's module body (the served bindings)
  m.def("cabi_swiglu", &py_swiglu, "moe_v4.swiglu through the C ABI");
  m.def("cabi_prepr6", &py_prepr6, "moe_v4.prepr6 through the C ABI");
  m.def("cabi_combine", &py_combine, "moe_v4.combine through the C ABI");
  m.def("cabi_init", &py_init, "moe_v4.init on the current device");
  m.def("cabi_gemm1r", &py_gemm1r, "moe_v4.gemm1r through the C ABI");
  m.def("cabi_gemm2r", &py_gemm2r, "moe_v4.gemm2r through the C ABI");
  m.def("cabi_args_sizes", []() {
    return std::vector<int64_t>{int64_t(sizeof(MoeV4InitArgs)), int64_t(sizeof(MoeV4Gemm1rArgs)), int64_t(sizeof(MoeV4Gemm2rArgs)),
                                int64_t(sizeof(MoeV4SwigluArgs)), int64_t(sizeof(MoeV4Prepr6Args)), int64_t(sizeof(MoeV4CombineArgs))};
  });
}
