// pm_swiglu: the fork's shared-expert swiglu (kernels/moe_stream/moe_raw.cu k_swiglu, verbatim), built with moe_raw.cu's
// flags (bench.build_extr: no fast math: accurate expf, IEEE division, no flush-to-zero) so pm_fused (production's
// fast-math flags) reproduces the engine's current shared-expert rounding bit for bit. pm_fused calls it through
// launcher() (a C function pointer; one process, same CUDA runtime).
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

// ---- verbatim from kernels/moe_stream/moe_raw.cu ----
// k_swiglu: shared expert middle step on the bf16 [T, 2 I] output of the fused gate|up GEMM -> bf16 h [T, I]
// (stock: up clamp(-10, 10), gate clamp(max 10), silu in fp32, h rounded to bf16). grid T, block I / 8.
__global__ void k_swiglu(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ h, int I) {
  const int t = blockIdx.x, i0 = threadIdx.x * 8;
  const uint4 gv = *(const uint4*)(gu + (size_t)t * 2 * I + i0), uv = *(const uint4*)(gu + (size_t)t * 2 * I + I + i0);
  const __nv_bfloat162* g2 = (const __nv_bfloat162*)&gv;
  const __nv_bfloat162* u2 = (const __nv_bfloat162*)&uv;
  uint4 o;
  __nv_bfloat162* o2 = (__nv_bfloat162*)&o;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 gf = __bfloat1622float2(g2[k]), uf = __bfloat1622float2(u2[k]);
    const float ga = fminf(gf.x, 10.f), gb = fminf(gf.y, 10.f);
    const float ua = fminf(fmaxf(uf.x, -10.f), 10.f), ub = fminf(fmaxf(uf.y, -10.f), 10.f);
    o2[k] = __floats2bfloat162_rn(ga / (1.f + expf(-ga)) * ua, gb / (1.f + expf(-gb)) * ub);
  }
  *(uint4*)(h + (size_t)t * I + i0) = o;
}
// ---- end verbatim ----

extern "C" void mpf_pm_swiglu_launch(const void* gu, void* h, int T, int I, cudaStream_t stream) {
  if (T > 0) k_swiglu<<<(unsigned)T, I / 8, 0, stream>>>((const __nv_bfloat16*)gu, (__nv_bfloat16*)h, I);
}

int64_t launcher() { return reinterpret_cast<int64_t>(&mpf_pm_swiglu_launch); }

void swiglu(torch::Tensor gu, torch::Tensor h) {  // = moe_raw.cu swiglu (tests)
  auto stream = at::cuda::getCurrentCUDAStream();
  const int I = (int)h.size(1);
  TORCH_CHECK(I % 8 == 0 && gu.size(1) == 2 * I, "swiglu shapes");
  mpf_pm_swiglu_launch(gu.data_ptr(), h.data_ptr(), (int)h.size(0), I, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launcher", &launcher, "address of mpf_pm_swiglu_launch(gu, h, T, I, stream)");
  m.def("swiglu", &swiglu);
}
