// xg: lamport all-gather of small 32-bit word arrays between the 4 GPUs in ONE kernel. Replaces comm.cu's k_ag_push + k_wait pair (data stores, fence.acq_rel.sys, release-add on a flag, then a
// separate wait kernel: ~6-9 us per exchange at C1) for HSD's word exchanges (sampler stats / mass / resample words).
// Every rank stores its n words straight into slot `rank` of every rank's mailbox with st.relaxed.sys; the words carry
// the arrival (sentinel 0xFFFFFFFF, a NaN pattern no fp32 arithmetic produces and no token id / index takes); every rank
// spins on its own mailbox until the W slots are complete and copies them to `out` [W, n] (the gathered buffer the
// consumers read). Mailboxes [2][W][nmax] per rank, double-buffered by call parity; each word is reset to the sentinel right
// after it is read (a peer writes this parity again only two calls later, after the next call, which needs this rank).
// Parity = a per-rank call counter, read by every CTA at start, bumped by the last CTA to finish.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace xg {
constexpr int W = 4, THR = 256;
constexpr unsigned SENT = 0xFFFFFFFFu;
struct Peers { unsigned* box[W]; };

__device__ __forceinline__ void st_sys(unsigned* p, unsigned v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__global__ void __launch_bounds__(THR) k_xg(const unsigned* __restrict__ src, unsigned* __restrict__ out, int n, int nmax,
                                            unsigned* box, Peers p, int rank, unsigned* ctr) {
  __shared__ unsigned par_s;
  __shared__ int last;
  if (threadIdx.x == 0) par_s = *(volatile unsigned*)ctr & 1u;
  __syncthreads();
  const unsigned par = par_s;
  const int i = blockIdx.x * THR + threadIdx.x;
  if (i < n) {
    unsigned v = src[i];
    if (v == SENT) v = 0x7FFFFFFFu;   // a negative full-payload NaN (or two bf16 ones) -> the canonical NaN: still NaN
#pragma unroll
    for (int d = 0; d < W; ++d) st_sys(p.box[(rank + d) % W] + ((size_t)par * W + rank) * nmax + i, v);
#pragma unroll
    for (int s = 0; s < W; ++s) {
      volatile unsigned* q = box + ((size_t)par * W + s) * nmax + i;
      unsigned w;
      while ((w = *q) == SENT) {}
      out[(size_t)s * n + i] = w;
      *q = SENT;   // consumed: back to the sentinel (no peer writes this parity again before our next-but-one call)
    }
  }
  // the last CTA bumps the call counter (every CTA read it at its start)
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(ctr + 1, 1u) == gridDim.x - 1;
    if (last) {
      ctr[1] = 0u;
      atomicAdd(ctr, 1u);
    }
  }
}
}  // namespace xg

// ---- C ABI (xg_cabi.h; fpr_cabi.h rules): the launch of pybind xgather, called by it and by the C++ driver
#include "xg_cabi.h"
#include "../../engine/mixed_cxx/fpr_cabi_rec.h"
FPR_CABI_RECORD_DECL(xg)

static int xgather_core(const FprXgGatherArgs* a) {   // "xg.xgather"
  using namespace xg;
  try {
    if (a->n < 0 || a->n > a->nmax) return FPR_E_ARG;
    Peers p;
    for (int s = 0; s < W; ++s) p.box[s] = (unsigned*)a->peers[s];
    const int G = (a->n + THR - 1) / THR;
    k_xg<<<G > 0 ? G : 1, THR, 0, a->c.stream>>>((const unsigned*)a->src, (unsigned*)a->out, a->n, a->nmax,
                                                  (unsigned*)a->box, p, (int)a->c.rank, (unsigned*)a->ctr);
    return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
  } catch (...) {
    return FPR_E_CUDA;
  }
}

void xgather(torch::Tensor src, torch::Tensor out, torch::Tensor box, std::vector<int64_t> peers, int64_t rank,
             torch::Tensor ctr) {
  using namespace xg;
  const c10::cuda::CUDAGuard dg(src.device());
  const int n = (int)src.numel();
  const int nmax = (int)(box.numel() / (2 * W));
  TORCH_CHECK(src.is_contiguous() && src.element_size() == 4 && out.is_contiguous() && out.numel() >= (int64_t)W * n,
              "xg: src [n] 32-bit contiguous, out [W, n]");
  TORCH_CHECK(n <= nmax && peers.size() == W && ctr.numel() >= 2 && ctr.scalar_type() == torch::kInt32, "xg: box / peers / ctr");
  FprXgGatherArgs ca{{at::cuda::getCurrentCUDAStream(), (int)src.get_device(), (int)rank}, src.data_ptr(), out.data_ptr(), n,
                     nmax, box.data_ptr(), {}, ctr.data_ptr<int>()};
  for (int s = 0; s < W; ++s) ca.peers[s] = (void*)peers[s];
  FPR_CABI_RECORD("xg.xgather", ca);
  TORCH_CHECK(xgather_core(&ca) == FPR_OK, "xg: CUDA launch error");
}

extern "C" {
static int c_xgather(const void* a) { return xgather_core((const FprXgGatherArgs*)a); }
static const FprFnEntry k_xg_fns[] = {{"xg.xgather", c_xgather, nullptr, (int64_t)sizeof(FprXgGatherArgs)}};
static const FprExtTable k_xg_table = {FPR_CABI_VERSION, "xg", "", 1, k_xg_fns};
__attribute__((visibility("default"))) const FprExtTable* fpr_xg_table(void) { return &k_xg_table; }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("xgather", &xgather); }
