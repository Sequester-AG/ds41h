// K5: sequence-parallel collectives by SM-issued peer stores with release/acquire flags
// (single-process 4-GPU harness: peer access, raw peer pointers).
//
//   AG (f || routing): rank r pushes its R own rows (row bytes RB) to slot r of every rank's gather buffer
//                      [W * R, RB] (itself included), then adds 1 to each receiver's flag (red.release.sys);
//                      a receiver waits until its flag reaches epoch * W * NCTA.
//   RS (MoE partials): rank r holds a partial [W * R, H] bf16 for ALL rows; it pushes the R rows owned by rank s into
//                      slot r of rank s's receive buffer [W, R, H]; rank s then sums the W slots in fixed rank order
//                      (fp32, one bf16 rounding) for its own rows (k_rs_sum, or fused into the next mHC boundary).
// Flags are monotonically increasing counters (no reset): the waiter compares with epoch * (number of signals).
// Memory model: data stores (st.global, any proxy) -> fence.acq_rel.sys (each thread) -> __syncthreads -> one thread
// red.release.sys.global.add on the receiver's flag; the waiter spins with ld.acquire.sys and then reads the data.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace spc {
constexpr int W = 4, THR = 512;

__device__ __forceinline__ void red_release_sys(unsigned* p, unsigned v) {
  asm volatile("red.release.sys.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }

struct Peers {
  char* buf[W];        // receive / gather buffer base on each rank
  unsigned* flag[W];   // flag word on each rank (this collective's)
};

#ifndef SPC_UNR
#define SPC_UNR 1  // >1: push loops stage SPC_UNR 16-B loads per thread in registers before their stores
#endif
// grid-stride copy of n16 uint4 (thread offset i0, stride st): with UNR > 1 the loads of UNR iterations are issued before
// their stores (the plain loop pays one load -> remote store round trip per iteration: dst may alias src for the compiler)
template <int UNR>
__device__ __forceinline__ void copy16(uint4* dst, const uint4* __restrict__ src, int n16, int i0, int st) {
  int i = i0;
  if (UNR > 1) {
    for (; i + (UNR - 1) * st < n16; i += UNR * st) {
      uint4 v[UNR];
#pragma unroll
      for (int u = 0; u < UNR; ++u) v[u] = __ldg(src + i + u * st);
#pragma unroll
      for (int u = 0; u < UNR; ++u) dst[i + u * st] = v[u];
    }
  }
  for (; i < n16; i += st) dst[i] = src[i];
}

// AG push: src [R, RB bytes] (16-B multiple) -> buf[s] + (rank * R) * RB on every rank s. grid = W * NCTA: CTA b serves
// destination (rank + 1 + b / NCTA) % W, so the three NVLink peers (and the local copy) are written in parallel; each
// CTA signals only its destination, which therefore counts W * NCTA signals per epoch.
__global__ void __launch_bounds__(THR) k_ag_push(const uint4* __restrict__ src, int R, int RB, int rank, Peers p,
                                                 unsigned* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);  // this rank's count of AG rounds (read by the waits)
  const int n16 = R * RB / 16, nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W, b = blockIdx.x % nc;
  uint4* dst = (uint4*)(p.buf[s] + (size_t)rank * R * RB);
  for (int i = b * THR + threadIdx.x; i < n16; i += nc * THR) dst[i] = src[i];
  fence_sys();
  __syncthreads();
  if (threadIdx.x == 0) red_release_sys(p.flag[s], 1u);
}

// AG push of two row sets in one kernel (f rows and the 64-B routing rows): the same CTAs store both, then ONE release
// signal per CTA on the f flag covers both (one launch, one fence round, one wait instead of two each).
__global__ void __launch_bounds__(THR) k_ag_push2(const uint4* __restrict__ src, int RB, const uint4* __restrict__ src2,
                                                  int RB2, int R, int rank, Peers p, Peers p2, unsigned* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int n16 = R * RB / 16, m16 = R * RB2 / 16, nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W,
            b = blockIdx.x % nc;
  uint4* dst = (uint4*)(p.buf[s] + (size_t)rank * R * RB);
  uint4* dst2 = (uint4*)(p2.buf[s] + (size_t)rank * R * RB2);
  if (SPC_UNR > 1) {
    copy16<SPC_UNR>(dst, src, n16, b * THR + threadIdx.x, nc * THR);
    copy16<1>(dst2, src2, m16, b * THR + threadIdx.x, nc * THR);
  } else {
    for (int i = b * THR + threadIdx.x; i < n16 + m16; i += nc * THR) {
      if (i < n16)
        dst[i] = src[i];
      else
        dst2[i - n16] = src2[i - n16];
    }
  }
  fence_sys();
  __syncthreads();
  if (threadIdx.x == 0) red_release_sys(p.flag[s], 1u);
}

// RS push: part [W * R, H] bf16 (all rows) -> rows [s R, (s+1) R) to buf[s] + rank * R * H * 2 on every rank s;
// grid = W * NCTA, CTA b serves destination (rank + 1 + b / NCTA) % W (as k_ag_push).
__global__ void __launch_bounds__(THR) k_rs_push(const uint4* __restrict__ part, int R, int H, int rank, Peers p,
                                                 unsigned* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int n16 = R * H * 2 / 16, nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W, b = blockIdx.x % nc;
  const uint4* src = part + (size_t)s * n16;
  uint4* dst = (uint4*)(p.buf[s] + (size_t)rank * R * H * 2);
  copy16<SPC_UNR>(dst, src, n16, b * THR + threadIdx.x, nc * THR);
  fence_sys();
  __syncthreads();
  if (threadIdx.x == 0) red_release_sys(p.flag[s], 1u);
}

// wait until *flag >= target (one thread per CTA spins; the CTA then proceeds) -- used at the head of consumers.
// target = (this rank's epoch, bumped by its push kernel earlier in the same stream) * signals per round
__device__ __forceinline__ void wait_flag(const unsigned* flag, const unsigned* epoch, unsigned signals) {
  if (threadIdx.x == 0) {
    const unsigned target = *(volatile const unsigned*)epoch * signals;
    while ((int)(ld_acquire_sys(flag) - target) < 0) __nanosleep(32);
  }
  __syncthreads();
}

__global__ void k_wait(const unsigned* flag, const unsigned* epoch, unsigned signals) { wait_flag(flag, epoch, signals); }

// RS sum: recv [W, R, H] bf16 (slot = source rank) -> out [R, H] bf16 = bf16(sum_{s=0..3} recv[s]) in fixed order.
__global__ void __launch_bounds__(THR) k_rs_sum(const __nv_bfloat162* __restrict__ recv, int R, int H,
                                                const unsigned* flag, const unsigned* epoch, unsigned signals,
                                                __nv_bfloat162* __restrict__ out) {
  wait_flag(flag, epoch, signals);
  const int n2 = R * H / 2;
  for (int i = blockIdx.x * THR + threadIdx.x; i < n2; i += gridDim.x * THR) {
    float2 a = __bfloat1622float2(recv[i]);
#pragma unroll
    for (int s = 1; s < W; ++s) {
      const float2 b = __bfloat1622float2(recv[(size_t)s * n2 + i]);
      a.x += b.x;
      a.y += b.y;
    }
    out[i] = __floats2bfloat162_rn(a.x, a.y);
  }
}
// Fused MoE combine + RS push: row t of this rank's partial (sum of its 6 slot rows of Y in slot order, fp32,
// + the shared-expert addend, one bf16 rounding: the same bits as moe_raw's k_combine) is stored straight into the owner's
// receive slot buf[t / R] + (rank * R + t % R) * H; grid-stride over the rows, then one system fence and one
// red.release.sys per owner per CTA. The owner waits for W * ncta signals per round. grid = ncta CTAs, H / 8 threads.
__global__ void __launch_bounds__(640) k_combine_push(const __nv_bfloat16* __restrict__ Y, const int* __restrict__ slotmap,
                                                      const __nv_bfloat16* __restrict__ sh, int H, int R, int T, int rank,
                                                      Peers p, unsigned* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int d0 = threadIdx.x * 8;
  for (int t = blockIdx.x; t < T; t += gridDim.x) {  // grid-stride over rows: one fence + W signals per CTA, not per row
    const int s = t / R, lr = t % R;
    int sl[6];
#pragma unroll
    for (int k = 0; k < 6; ++k) sl[k] = slotmap[6 * t + k];
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    uint4 v[6];
#pragma unroll
    for (int k = 0; k < 6; ++k) v[k] = sl[k] >= 0 ? *(const uint4*)(Y + (size_t)sl[k] * H + d0) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      const __nv_bfloat162* h = (const __nv_bfloat162*)&v[k];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        acc[2 * i] += f.x, acc[2 * i + 1] += f.y;
      }
    }
    const uint4 sv = sh ? *(const uint4*)(sh + (size_t)t * H + d0) : make_uint4(0, 0, 0, 0);
    const __nv_bfloat162* hs = (const __nv_bfloat162*)&sv;
    uint4 o;
    __nv_bfloat162* ho = (__nv_bfloat162*)&o;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(hs[i]);
      ho[i] = __floats2bfloat162_rn(acc[2 * i] + f.x, acc[2 * i + 1] + f.y);
    }
    *(uint4*)((__nv_bfloat16*)p.buf[s] + ((size_t)rank * R + lr) * H + d0) = o;
  }
  fence_sys();
  __syncthreads();
  if (threadIdx.x < W) red_release_sys(p.flag[threadIdx.x], 1u);  // every CTA signals every owner once
}
}  // namespace spc

static spc::Peers peers(const std::vector<torch::Tensor>& bufs, const std::vector<torch::Tensor>& flags) {
  spc::Peers p;
  for (int s = 0; s < spc::W; ++s) {
    p.buf[s] = (char*)bufs[s].data_ptr();
    p.flag[s] = (unsigned*)flags[s].data_ptr();
  }
  return p;
}

void enable_peers(int64_t n) {
  for (int a = 0; a < n; ++a) {
    cudaSetDevice(a);
    for (int b = 0; b < n; ++b) {
      if (a == b) continue;
      cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) TORCH_CHECK(false, cudaGetErrorString(e));
      cudaGetLastError();
    }
  }
}

void ag_push(torch::Tensor src, int64_t rank, std::vector<torch::Tensor> bufs, std::vector<torch::Tensor> flags,
             torch::Tensor epoch, int64_t ncta) {
  const int R = (int)src.size(0), RB = (int)(src.numel() / src.size(0) * src.element_size());
  TORCH_CHECK(RB % 16 == 0);
  spc::k_ag_push<<<(int)ncta * spc::W, spc::THR, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)src.data_ptr(), R, RB,
                                                                               (int)rank, peers(bufs, flags),
                                                                               (unsigned*)epoch.data_ptr());
}

// f rows src [R, RB] to bufs (flags = the f flags), routing rows src2 [R, RB2] to bufs2 (no flags of their own)
void ag_push2(torch::Tensor src, torch::Tensor src2, int64_t rank, std::vector<torch::Tensor> bufs,
              std::vector<torch::Tensor> flags, std::vector<torch::Tensor> bufs2, torch::Tensor epoch, int64_t ncta) {
  const int R = (int)src.size(0), RB = (int)(src.numel() / src.size(0) * src.element_size());
  const int RB2 = (int)(src2.numel() / src2.size(0) * src2.element_size());
  TORCH_CHECK(RB % 16 == 0 && RB2 % 16 == 0 && src2.size(0) == R, "ag_push2: shapes");
  spc::k_ag_push2<<<(int)ncta * spc::W, spc::THR, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)src.data_ptr(), RB, (const uint4*)src2.data_ptr(), RB2, R, (int)rank, peers(bufs, flags),
      peers(bufs2, flags), (unsigned*)epoch.data_ptr());
}

void rs_push(torch::Tensor part, int64_t rank, std::vector<torch::Tensor> bufs, std::vector<torch::Tensor> flags,
             torch::Tensor epoch, int64_t ncta) {
  const int R = (int)(part.size(0) / spc::W), H = (int)part.size(1);
  spc::k_rs_push<<<(int)ncta * spc::W, spc::THR, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)part.data_ptr(), R, H,
                                                                               (int)rank, peers(bufs, flags),
                                                                               (unsigned*)epoch.data_ptr());
}

void wait(torch::Tensor flag, torch::Tensor epoch, int64_t signals) {
  spc::k_wait<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>((const unsigned*)flag.data_ptr(),
                                                             (const unsigned*)epoch.data_ptr(), (unsigned)signals);
}

void rs_sum(torch::Tensor recv, torch::Tensor flag, torch::Tensor epoch, int64_t signals, torch::Tensor out,
            int64_t ncta) {
  const int R = (int)recv.size(1), H = (int)recv.size(2);
  spc::k_rs_sum<<<(int)ncta, spc::THR, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat162*)recv.data_ptr(), R, H, (const unsigned*)flag.data_ptr(),
      (const unsigned*)epoch.data_ptr(), (unsigned)signals, (__nv_bfloat162*)out.data_ptr());
}

void combine_push(torch::Tensor Y, torch::Tensor slotmap, torch::Tensor sh, int64_t R, int64_t rank,
                  std::vector<torch::Tensor> bufs, std::vector<torch::Tensor> flags, torch::Tensor epoch, int64_t ncta) {
  const int H = (int)Y.size(1), T = (int)(slotmap.numel() / 6);
  spc::k_combine_push<<<(int)ncta, H / 8, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)Y.data_ptr(), slotmap.data_ptr<int>(),
      sh.numel() ? (const __nv_bfloat16*)sh.data_ptr() : nullptr, H, (int)R, T, (int)rank, peers(bufs, flags),
      (unsigned*)epoch.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("combine_push", &combine_push);
  m.def("enable_peers", &enable_peers);
  m.def("ag_push", &ag_push);
  m.def("ag_push2", &ag_push2);
  m.def("rs_push", &rs_push);
  m.def("wait", &wait);
  m.def("rs_sum", &rs_sum);
}
