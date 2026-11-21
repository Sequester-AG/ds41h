// rtag: the DSpark drafter's router + its all-gather in ONE kernel per GPU.
// Replaces, per drafter block, cuBLAS's gate GEMM (bf16 x bf16 -> fp32, split-K + reduce) + drafter.py's _router_kernel
// (or hsdt's dmask variant) + comm.cu's k_ag_push2 of (f rows, routing rows): ~17 us of five launches at C1.
// Same math for the drafter.
// grid = W * NC CTAs (as k_ag_push2): CTA b pushes its share of the f rows to destination s = (rank + 1 + b / NC) % W, then
// computes the gate logits of experts [b * EPC, (b + 1) * EPC); the last CTA (arrival counter) picks top-3 per row, writes
// the 64-B routing rows (ids, -1 padded to 6; weights, 0 padded) locally and into every rank's routing gather buffer, then
// signals every destination once more. Receivers wait for epoch * W * (NC + 1) signals (comm.cu's k_wait): each of the W
// source ranks sends NC f signals + 1 routing signal to every destination.
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace rt {
constexpr int W = 4, D = 5120, NE = 128, TOPK = 3, THR = 256, NWP = THR / 32, MAXR = 4, KIT = (D + 8 * THR - 1) / (8 * THR);
struct Peers { char* f[W]; char* r[W]; unsigned* flag[W]; };

__device__ __forceinline__ void red_release_sys(unsigned* p, unsigned v) {
  asm volatile("red.release.sys.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }

__global__ void __launch_bounds__(THR) k_rtag(const __nv_bfloat16* __restrict__ f, int R, const __nv_bfloat16* __restrict__ gate,
                                              const float* __restrict__ gbias, float* lg, unsigned* cnt, int* rows,
                                              const long long* dslot, int r0, int kq, int rank, int nc, int epc, Peers p,
                                              unsigned* epoch) {
  __shared__ float red[NWP][MAXR * 4];
  __shared__ int last;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, b = blockIdx.x, G = gridDim.x;
  if (b == 0 && t == 0) atomicAdd(epoch, 1u);
  // ---- f rows -> slot `rank` of destination s
  {
    const int s = (rank + 1 + b / nc) % W, bb = b % nc;
    const int n16 = R * D * 2 / 16;
    const uint4* src = (const uint4*)f;
    uint4* dst = (uint4*)(p.f[s] + (size_t)rank * R * D * 2);
    for (int i = bb * THR + t; i < n16; i += nc * THR) dst[i] = src[i];
  }
  // ---- gate logits of experts [b * epc, b * epc + epc) for the R rows: thread t covers elements 8 t + 2048 j; every
  // load of the CTA (f rows from L2, gate rows from HBM) is issued before the first FMA (one latency round, not KIT)
  {
    uint4 fu[KIT][MAXR], gu[KIT][4];
#pragma unroll
    for (int it = 0; it < KIT; ++it) {
      const int k = 8 * t + 8 * THR * it;
#pragma unroll
      for (int r = 0; r < MAXR; ++r)
        fu[it][r] = (k < D && r < R) ? *(const uint4*)(f + (size_t)r * D + k) : make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int e = 0; e < 4; ++e)
        gu[it][e] = (k < D && e < epc) ? __ldg((const uint4*)(gate + (size_t)(b * epc + e) * D + k)) : make_uint4(0, 0, 0, 0);
    }
    float acc[4][MAXR];
#pragma unroll
    for (int e = 0; e < 4; ++e)
#pragma unroll
      for (int r = 0; r < MAXR; ++r) acc[e][r] = 0.f;
#pragma unroll
    for (int it = 0; it < KIT; ++it) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const __nv_bfloat162* hg = (const __nv_bfloat162*)&gu[it][e];
#pragma unroll
        for (int r = 0; r < MAXR; ++r) {
          const __nv_bfloat162* hf = (const __nv_bfloat162*)&fu[it][r];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 x = __bfloat1622float2(hf[j]), y = __bfloat1622float2(hg[j]);
            acc[e][r] = fmaf(x.x, y.x, acc[e][r]);
            acc[e][r] = fmaf(x.y, y.y, acc[e][r]);
          }
        }
      }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e)
#pragma unroll
      for (int r = 0; r < MAXR; ++r) {
        float v = acc[e][r];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[warp][e * MAXR + r] = v;
      }
    __syncthreads();
    if (t < epc * R) {
      const int e = t / R, r = t % R;
      float v = 0.f;
      for (int w = 0; w < NWP; ++w) v += red[w][e * MAXR + r];
      lg[(size_t)r * NE + b * epc + e] = v;
    }
  }
  // ---- arrival; the last CTA routes
  __threadfence();
  __syncthreads();
  if (t == 0) {
    const unsigned a = atomicAdd(cnt, 1u);
    last = a == (unsigned)G - 1;
    if (last) *cnt = 0u;
  }
  __syncthreads();
  // this CTA's f stores -> one signal to its destination
  fence_sys();
  __syncthreads();
  if (t == 0) red_release_sys(p.flag[(rank + 1 + b / nc) % W], 1u);
  if (!last) return;
  __threadfence();
  // one warp per row: = drafter._router_kernel (scores sqrt(softplus(x)), selection on score + bias, weights normalized x 1.5)
  for (int r = warp; r < R; r += NWP) {
    float sc[4], v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float x = __ldcg(lg + (size_t)r * NE + lane + 32 * j);
      const float sp = x > 20.f ? x : log1pf(expf(x));
      sc[j] = sqrtf(sp);
      v[j] = sc[j] + gbias[lane + 32 * j];
    }
    int ids[TOPK];
    float ws[TOPK], sum = 0.f;
#pragma unroll
    for (int q = 0; q < TOPK; ++q) {
      float bv = -INFINITY, bs = 0.f;
      int bi = 0x7fffffff;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (v[j] > bv || (v[j] == bv && lane + 32 * j < bi)) { bv = v[j]; bi = lane + 32 * j; bs = sc[j]; }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, bv, o), s2 = __shfl_xor_sync(0xffffffffu, bs, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, bi, o);
        if (v2 > bv || (v2 == bv && i2 < bi)) { bv = v2; bi = i2; bs = s2; }
      }
      ids[q] = bi;
      ws[q] = bs;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (lane + 32 * j == bi) v[j] = -INFINITY;
    }
#pragma unroll
    for (int q = 0; q < TOPK; ++q) sum += ws[q];
    const bool live = dslot == nullptr || dslot[r0 + r / kq] >= 0;
    if (lane < 16) {   // the 64-B routing row: ids [0, 6), weights [6, 12), words 12..15 untouched (0)
      int word = 0;
      if (lane < 6) word = (lane < TOPK && live) ? ids[lane] : -1;
      else if (lane < 12) {
        const int q = lane - 6;
        word = __float_as_int((q < TOPK && live) ? ws[q] / (sum + 1e-20f) * 1.5f : 0.f);
      }
      if (lane < 12) {
        rows[r * 16 + lane] = word;
#pragma unroll
        for (int s = 0; s < W; ++s) ((int*)(p.r[s] + (size_t)(rank * R + r) * 64))[lane] = word;
      }
    }
  }
  fence_sys();
  __syncthreads();
  if (t < W) red_release_sys(p.flag[t], 1u);   // the routing rows: one more signal to every destination
}
}  // namespace rt

// ---- C ABI (rtag_cabi.h; fpr_cabi.h rules): the launch of pybind rtag, called by it and by the C++ driver
#include "rtag_cabi.h"
#include "../../engine/mixed_cxx/fpr_cabi_rec.h"
FPR_CABI_RECORD_DECL(rtag)

static int rtag_core(const FprRtagArgs* a) {   // "rtag.rtag"
  using namespace rt;
  try {
    const int G = (int)(W * a->nc), epc = (NE + G - 1) / G;
    if (a->R < 0 || a->R > MAXR || a->nc <= 0 || epc > 4 || G * epc != NE) return FPR_E_ARG;
    Peers p;
    for (int s = 0; s < W; ++s) {
      p.f[s] = (char*)a->fbufs[s];
      p.r[s] = (char*)a->rbufs[s];
      p.flag[s] = (unsigned*)a->flags[s];
    }
    k_rtag<<<G, THR, 0, a->c.stream>>>((const __nv_bfloat16*)a->f, (int)a->R, (const __nv_bfloat16*)a->gate, a->gbias, a->lg,
                                       (unsigned*)a->cnt, (int*)a->rows, (const long long*)a->dslot, (int)a->r0, (int)a->kq,
                                       (int)a->c.rank, (int)a->nc, epc, p, (unsigned*)a->epoch);
    return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
  } catch (...) {
    return FPR_E_CUDA;
  }
}

void rtag(torch::Tensor f, torch::Tensor gate, torch::Tensor gbias, torch::Tensor lg, torch::Tensor cnt, torch::Tensor rows,
          torch::Tensor dslot, int64_t r0, int64_t kq, int64_t rank, int64_t nc, std::vector<torch::Tensor> fbufs,
          std::vector<torch::Tensor> rbufs, std::vector<torch::Tensor> flags, torch::Tensor epoch) {
  using namespace rt;
  const c10::cuda::CUDAGuard dg(f.device());
  const int R = (int)f.size(0);
  TORCH_CHECK(f.is_contiguous() && f.size(1) == D && R <= MAXR && f.scalar_type() == torch::kBFloat16, "rtag: f [R <= 4, 5120]");
  TORCH_CHECK(gate.is_contiguous() && gate.size(0) == NE && gate.size(1) == D, "rtag: gate [128, 5120] bf16");
  const int G = (int)(W * nc), epc = (NE + G - 1) / G;
  TORCH_CHECK(epc <= 4 && G * epc == NE, "rtag: W * nc must divide 128 into <= 4 experts per CTA");
  FprRtagArgs ca{};
  ca.c = FprCall{at::cuda::getCurrentCUDAStream(), (int)f.get_device(), (int)rank};
  ca.f = f.data_ptr(); ca.R = R; ca.gate = gate.data_ptr(); ca.gbias = gbias.data_ptr<float>(); ca.lg = lg.data_ptr<float>();
  ca.cnt = cnt.data_ptr<int>(); ca.rows = rows.data_ptr<int>();
  ca.dslot = dslot.numel() ? dslot.data_ptr<int64_t>() : nullptr;
  ca.r0 = (int32_t)r0; ca.kq = (int32_t)kq; ca.nc = (int32_t)nc;
  for (int s = 0; s < W; ++s) {
    ca.fbufs[s] = fbufs[s].data_ptr();
    ca.rbufs[s] = rbufs[s].data_ptr();
    ca.flags[s] = (uint32_t*)flags[s].data_ptr();
  }
  ca.epoch = (uint32_t*)epoch.data_ptr<int>();
  FPR_CABI_RECORD("rtag.rtag", ca);
  TORCH_CHECK(rtag_core(&ca) == FPR_OK, "rtag: launch error");
}

extern "C" {
static int c_rtag(const void* a) { return rtag_core((const FprRtagArgs*)a); }
static const FprFnEntry k_rtag_fns[] = {{"rtag.rtag", c_rtag, nullptr, (int64_t)sizeof(FprRtagArgs)}};
static const FprExtTable k_rtag_table = {FPR_CABI_VERSION, "rtag", "", 1, k_rtag_fns};
__attribute__((visibility("default"))) const FprExtTable* fpr_rtag_table(void) { return &k_rtag_table; }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("rtag", &rtag); }
