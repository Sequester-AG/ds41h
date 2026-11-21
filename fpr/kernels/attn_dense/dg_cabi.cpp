// dg_cabi.cpp -- C ABI wrapper of DeepGEMM's DENSE8 calls. No kernels of its own.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "dg_cabi.h"

namespace fdg {

enum Op { kGemm = 1, kEinsum = 2 };

std::mutex g_mu;                        // the rule-4 exception: DeepGEMM's process-global num_sms (and the warm set)
std::unordered_set<uint64_t> g_warm;    // warmed (op, views without pointers, num_sms, device) keys
std::vector<std::string> g_warm_desc;   // the same, readable (dumpable for stage 2)
bool g_have_ops = false;
c10::OperatorHandle* g_gemm = nullptr;
c10::OperatorHandle* g_einsum = nullptr;
c10::OperatorHandle* g_set_sms = nullptr;
c10::OperatorHandle* g_get_sms = nullptr;

// resolve the dispatcher ops once (pybind side, Python thread: deep_gemm's library is loaded by then)
void resolve_ops() {
  if (g_have_ops) return;
  auto& D = c10::Dispatcher::singleton();
  g_gemm = new c10::OperatorHandle(D.findSchemaOrThrow("deep_gemm::fp8_fp4_gemm_nt", ""));
  g_einsum = new c10::OperatorHandle(D.findSchemaOrThrow("deep_gemm::fp8_einsum", ""));
  g_set_sms = new c10::OperatorHandle(D.findSchemaOrThrow("deep_gemm::set_num_sms", ""));
  g_get_sms = new c10::OperatorHandle(D.findSchemaOrThrow("deep_gemm::get_num_sms", ""));
  g_have_ops = true;
}

at::ScalarType st_of(int32_t dt) {
  switch (dt) {
    case FPR_DG_E4M3: return at::kFloat8_e4m3fn;
    case FPR_DG_F32: return at::kFloat;
    case FPR_DG_BF16: return at::kBFloat16;
    default: return at::kByte;
  }
}

bool view_ok(const FprDgView& v) { return v.ptr != nullptr && v.ndim >= 1 && v.ndim <= 4 && v.dtype >= 1 && v.dtype <= 3; }

at::Tensor tensor_of(const FprDgView& v, int dev) {
  return at::from_blob(v.ptr, at::IntArrayRef(v.sizes, v.ndim), at::IntArrayRef(v.strides, v.ndim),
                       at::TensorOptions().dtype(st_of(v.dtype)).device(at::Device(at::kCUDA, dev)));
}

void fnv(uint64_t& h, int64_t x) {
  for (int i = 0; i < 8; ++i) {
    h ^= static_cast<uint64_t>((x >> (8 * i)) & 0xff);
    h *= 1099511628211ull;
  }
}

uint64_t key_of(int op, const FprDgView* const* v, int nv, int64_t num_sms, int dev, std::string* desc) {
  uint64_t h = 1469598103934665603ull;
  fnv(h, op);
  fnv(h, num_sms);
  fnv(h, dev);
  if (desc) *desc = std::string(op == kGemm ? "fp8_gemm_nt" : "fp8_einsum") + " dev " + std::to_string(dev) + " sms " +
                    std::to_string(num_sms);
  for (int i = 0; i < nv; ++i) {
    fnv(h, v[i]->ndim);
    fnv(h, v[i]->dtype);
    if (desc) *desc += " [";
    for (int j = 0; j < v[i]->ndim; ++j) {
      fnv(h, v[i]->sizes[j]);
      fnv(h, v[i]->strides[j]);
      if (desc) *desc += std::to_string(v[i]->sizes[j]) + ":" + std::to_string(v[i]->strides[j]) + (j + 1 < v[i]->ndim ? "," : "");
    }
    if (desc) *desc += "]";
  }
  return h;
}

int64_t allocs(int dev) {
  const auto s = c10::cuda::CUDACachingAllocator::getDeviceStats(static_cast<c10::DeviceIndex>(dev));
  return s.allocation[0].allocated;   // StatType::AGGREGATE = 0: allocations requested by client code
}

// the one path (pybind warm / call, C ABI call): returns an FPR_* code; *err gets an exception text
int run(int op, const FprCall& call, const FprDgView& a, const FprDgView& sfa, const FprDgView& b, const FprDgView& sfb,
        const FprDgView& d, int64_t num_sms, bool warm, std::string* err) {
  if (!view_ok(a) || !view_ok(sfa) || !view_ok(b) || !view_ok(sfb) || !view_ok(d) || num_sms <= 0) return FPR_E_ARG;
  const FprDgView* vs[5] = {&a, &sfa, &b, &sfb, &d};
  std::string desc;
  const uint64_t key = key_of(op, vs, 5, num_sms, call.device, warm ? &desc : nullptr);
  std::lock_guard<std::mutex> lk(g_mu);
  if (!g_have_ops) return FPR_E_INIT;
  if (!warm && !g_warm.count(key)) return FPR_E_SHAPE;   // never a first-use JIT compile inside a driver step
  try {
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(call.stream, static_cast<c10::DeviceIndex>(call.device)));
    const int dev = call.device;
    at::Tensor ta = tensor_of(a, dev), tsa = tensor_of(sfa, dev), tb = tensor_of(b, dev), tsb = tensor_of(sfb, dev),
               td = tensor_of(d, dev);
    torch::jit::Stack st;
    g_get_sms->callBoxed(&st);
    const int64_t old = st.back().toInt();
    st.clear();
    st.emplace_back(num_sms);
    g_set_sms->callBoxed(&st);
    st.clear();
    const int64_t a0 = allocs(dev);
    if (op == kGemm) {
      // = vLLM fp8_gemm_nt((a, sfa), (b, sfb), d, is_deep_gemm_e8m0_used=False) -> deep_gemm fp8_fp4_gemm_nt
      st = {ta, tsa, tb, tsb, td, c10::IValue(), c10::IValue(), c10::IValue(), c10::IValue(), std::string("nk"), true,
            c10::IValue()};
      g_gemm->callBoxed(&st);
    } else {
      // = vLLM fp8_einsum("bhr,hdr->bhd", (a, sfa), (b, sfb), d, recipe=(1, 128, 128))
      st = {std::string("bhr,hdr->bhd"), ta, tsa, tb, tsb, td, c10::IValue(), std::vector<int64_t>{1, 128, 128}, c10::IValue()};
      g_einsum->callBoxed(&st);
    }
    const int64_t a1 = allocs(dev);
    st.clear();
    st.emplace_back(old);
    g_set_sms->callBoxed(&st);
    if (a1 != a0) return FPR_E_SHAPE;   // the op allocated (a scale-layout transform): not a driver shape; never warmed
    if (warm && g_warm.insert(key).second) g_warm_desc.push_back(desc);
    return FPR_OK;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return FPR_E_CUDA;
  }
}

int c_gemm(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprDgGemmArgs*>(p);
  return run(kGemm, a.call, a.a, a.sfa, a.b, a.sfb, a.d, a.num_sms, false, nullptr);
}

int c_einsum(const void* p) {
  if (p == nullptr) return FPR_E_ARG;
  const auto& a = *static_cast<const FprDgEinsumArgs*>(p);
  return run(kEinsum, a.call, a.a, a.sfa, a.b, a.sfb, a.d, a.num_sms, false, nullptr);
}

// ---- pybind: warm-up, the same calls for gate tests, the warmed set
FprDgView view_of(const at::Tensor& t) {
  FprDgView v{};
  TORCH_CHECK(t.is_cuda() && t.dim() >= 1 && t.dim() <= 4, "dg_cabi: CUDA tensor of rank 1..4");
  v.ptr = t.data_ptr();
  v.ndim = static_cast<int32_t>(t.dim());
  for (int i = 0; i < t.dim(); ++i) {
    v.sizes[i] = t.size(i);
    v.strides[i] = t.stride(i);
  }
  const auto s = t.scalar_type();
  v.dtype = s == at::kFloat8_e4m3fn ? FPR_DG_E4M3 : s == at::kFloat ? FPR_DG_F32 : s == at::kBFloat16 ? FPR_DG_BF16 : 0;
  TORCH_CHECK(v.dtype != 0, "dg_cabi: dtype e4m3 / f32 / bf16");
  return v;
}

int64_t py_run(int op, at::Tensor a, at::Tensor sfa, at::Tensor b, at::Tensor sfb, at::Tensor d, int64_t num_sms, bool warm) {
  resolve_ops();
  FprCall c;
  c.stream = at::cuda::getCurrentCUDAStream(a.get_device());
  c.device = a.get_device();
  c.rank = 0;
  std::string err;
  const int r = run(op, c, view_of(a), view_of(sfa), view_of(b), view_of(sfb), view_of(d), num_sms, warm, &err);
  TORCH_CHECK(r != FPR_E_CUDA, "dg_cabi: DeepGEMM op failed: ", err);
  return r;
}

int64_t warm_gemm(at::Tensor a, at::Tensor sfa, at::Tensor b, at::Tensor sfb, at::Tensor d, int64_t num_sms) {
  return py_run(kGemm, a, sfa, b, sfb, d, num_sms, true);
}
int64_t warm_einsum(at::Tensor a, at::Tensor sfa, at::Tensor b, at::Tensor sfb, at::Tensor d, int64_t num_sms) {
  return py_run(kEinsum, a, sfa, b, sfb, d, num_sms, true);
}
int64_t gemm(at::Tensor a, at::Tensor sfa, at::Tensor b, at::Tensor sfb, at::Tensor d, int64_t num_sms) {
  return py_run(kGemm, a, sfa, b, sfb, d, num_sms, false);
}
int64_t einsum(at::Tensor a, at::Tensor sfa, at::Tensor b, at::Tensor sfb, at::Tensor d, int64_t num_sms) {
  return py_run(kEinsum, a, sfa, b, sfb, d, num_sms, false);
}
std::vector<std::string> warmed() {
  std::lock_guard<std::mutex> lk(g_mu);
  return g_warm_desc;
}

}  // namespace fdg

extern "C" __attribute__((visibility("default"))) const FprExtTable* fpr_dg_table(void) {
  static const FprFnEntry fns[] = {
      {"dg.fp8_gemm_nt", fdg::c_gemm, nullptr, static_cast<int64_t>(sizeof(FprDgGemmArgs))},
      {"dg.fp8_einsum", fdg::c_einsum, nullptr, static_cast<int64_t>(sizeof(FprDgEinsumArgs))},
  };
  static const FprExtTable t = {FPR_CABI_VERSION, "dg", "deep_gemm 2.8.0 (vllm.third_party)", 2, fns};
  return &t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("warm_gemm", &fdg::warm_gemm, "run fp8_gemm_nt (DeepGEMM) at this key and add it to the warmed set (0 = OK, -2 = allocated)");
  m.def("warm_einsum", &fdg::warm_einsum, "run fp8_einsum bhr,hdr->bhd at this key and add it to the warmed set");
  m.def("gemm", &fdg::gemm, "the C path's fp8_gemm_nt (warmed keys only; -2 = not warmed)");
  m.def("einsum", &fdg::einsum, "the C path's fp8_einsum (warmed keys only)");
  m.def("warmed", &fdg::warmed, "the warmed keys (readable; stage 2's AOT list)");
}
