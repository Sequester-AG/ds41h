// agrs: NVLink push variants for the AG / RS of a full-load mixed step (4 x 4,288 rows of 10 KB), measured
// against the engine's comm.cu kernels (kernels/sp_layer/comm.cu, read-only; same memory model):
//   data stores -> fence.acq_rel.sys -> __syncthreads -> one red.release.sys per CTA on the receiver's flag;
//   the receiver spins with ld.acquire.sys until flag >= epoch * signals.
// k_push: one kernel for both collectives. Destination s (every rank, itself included) gets n16 uint4 from
//   src + s * src_stride16 (AG: stride 0, the rank's own rows; RS: stride n16, the rows owned by s) at slot `rank`.
//   MODE 0 = comm.cu's grid-stride order (CTA b of a destination interleaves with the other CTAs at 16 B per thread);
//   MODE 1 = one contiguous slab per CTA (each CTA streams its own range).
// ce_copy: one cudaMemcpyAsync (copy engine, peer access on) on a given stream.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace agrs {
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
  char* buf[W];
  unsigned* flag[W];
};

template <int UNR>
__device__ __forceinline__ void copy16(uint4* dst, const uint4* __restrict__ src, long n16, long i0, long st) {
  long i = i0;
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

template <int UNR, int MODE>
__global__ void __launch_bounds__(THR) k_push(const uint4* __restrict__ src, long n16, long src_stride16, int rank,
                                              Peers p, unsigned* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W, b = blockIdx.x % nc;
  const uint4* srcs = src + (size_t)s * src_stride16;
  uint4* dst = (uint4*)p.buf[s] + (size_t)rank * n16;
  if (MODE == 0) {
    copy16<UNR>(dst, srcs, n16, (long)b * THR + threadIdx.x, (long)nc * THR);
  } else {
    const long per = (n16 + nc - 1) / nc, a0 = (long)b * per, a1 = min(n16, a0 + per);
    if (a0 < a1) copy16<UNR>(dst + a0, srcs + a0, a1 - a0, threadIdx.x, THR);
  }
  fence_sys();
  __syncthreads();
  if (threadIdx.x == 0) red_release_sys(p.flag[s], 1u);
}

__device__ __forceinline__ void wait_flag(const unsigned* flag, const unsigned* epoch, unsigned signals) {
  if (threadIdx.x == 0) {
    const unsigned target = *(volatile const unsigned*)epoch * signals;
    while ((int)(ld_acquire_sys(flag) - target) < 0) __nanosleep(32);
  }
  __syncthreads();
}
__global__ void k_wait(const unsigned* flag, const unsigned* epoch, unsigned signals) { wait_flag(flag, epoch, signals); }

// comm.cu's k_rs_sum arithmetic (fixed slot order, fp32, one bf16 rounding), without the wait: the sum's own time
__global__ void __launch_bounds__(THR) k_sum(const __nv_bfloat162* __restrict__ recv, long n2, __nv_bfloat162* __restrict__ out) {
  for (long i = (long)blockIdx.x * THR + threadIdx.x; i < n2; i += (long)gridDim.x * THR) {
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

// k_combine_push_dm (agrs, decode RS): comm.cu k_combine_push's per-row statements verbatim (moe_raw k_combine's bits:
// 6 slot rows summed in slot order in fp32, + the shared addend, one bf16 rounding), DESTINATION-MAJOR: CTA group
// g = blockIdx / nc serves owner q = (rank + 1 + g) % W and grid-strides over q's R rows, so the 3 NVLink peers and the
// local slot are written at once (k_combine_push sweeps the owners in row order: one link at a time). Every CTA signals
// every owner once: owners count W * gridDim.x signals per round.
__global__ void __launch_bounds__(640) k_combine_push_dm(const __nv_bfloat16* __restrict__ Y, const int* __restrict__ slotmap,
                                                         const __nv_bfloat16* __restrict__ sh, int H, int R, int rank,
                                                         Peers p, unsigned* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int d0 = threadIdx.x * 8;
  const int nc = gridDim.x / W, s = (rank + 1 + (int)blockIdx.x / nc) % W, b = (int)blockIdx.x % nc;
  for (int lr = ((int)blockIdx.x < nc * W) ? b : R; lr < R; lr += nc) {
    const int t = s * R + lr;
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
  if (threadIdx.x < W) red_release_sys(p.flag[threadIdx.x], 1u);
}
// ---- small-batch one-shot collectives (microbenchmark prototypes, not in the engine) ----
// k_ag1: comm.cu k_ag_push2's stores (f rows + route rows, destination-major CTA groups, one release signal per CTA on the
// f flag), then block 0 waits for this rank's own flag in the SAME kernel (k_ag_push2 + k_wait in one launch).
// nrows (optional, device int32 [W]): push only this rank's first nrows[rank] rows (the valid rows; padding rows of the
// receivers' slots keep old bytes), else all R rows (= k_ag_push2's bytes).
__global__ void __launch_bounds__(THR) k_ag1(const uint4* __restrict__ src, int RB, const uint4* __restrict__ src2, int RB2, int R,
                                             const int* __restrict__ nrows, int rank, Peers p, Peers p2, unsigned* epoch,
                                             unsigned signals) {
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int n = nrows ? min(nrows[rank], R) : R;
  const int n16 = n * RB / 16, m16 = n * RB2 / 16, nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W, b = blockIdx.x % nc;
  uint4* dst = (uint4*)(p.buf[s] + (size_t)rank * R * RB);
  uint4* dst2 = (uint4*)(p2.buf[s] + (size_t)rank * R * RB2);
  for (int i = b * THR + threadIdx.x; i < n16 + m16; i += nc * THR) {
    if (i < n16)
      dst[i] = src[i];
    else
      dst2[i - n16] = src2[i - n16];
  }
  fence_sys();
  __syncthreads();
  if (threadIdx.x == 0) red_release_sys(p.flag[s], 1u);
  if (blockIdx.x == 0) wait_flag(p.flag[rank], epoch, signals);   // block 0 bumped the epoch itself: no cross-CTA race
}

// k_rs1: comm.cu k_rs_push's stores (owner s's rows -> its receive slot `rank`, destination-major, one release signal per
// CTA) and k_rs_sum's arithmetic (fixed slot order 0..3, fp32, one bf16 rounding) in ONE launch. Every CTA bumps the epoch
// once (round = old / gridDim.x: rounds never overlap in a stream), pushes, then waits for its own flag and sums its share
// of this rank's rows. nrows (optional, device [W]): owner s gets only its first nrows[s] rows, this rank sums only its
// nrows[rank] rows (padding rows of out keep old bytes). grid must be resident in one wave (all CTAs push before any waits).
__global__ void __launch_bounds__(THR) k_rs1(const uint4* __restrict__ part, int R, int H, const int* __restrict__ nrows, int rank,
                                             Peers p, unsigned* epoch, unsigned signals, const __nv_bfloat162* __restrict__ recv,
                                             __nv_bfloat162* __restrict__ out) {
  __shared__ unsigned round_;
  if (threadIdx.x == 0) round_ = atomicAdd(epoch, 1u) / gridDim.x;
  const int nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W, b = blockIdx.x % nc;
  const int row16 = H * 2 / 16;
  const int ns = nrows ? min(nrows[s], R) : R;
  const uint4* src = part + (size_t)s * R * row16;
  uint4* dst = (uint4*)(p.buf[s] + (size_t)rank * R * H * 2);
  for (int i = b * THR + threadIdx.x; i < ns * row16; i += nc * THR) dst[i] = src[i];
  fence_sys();
  __syncthreads();
  if (threadIdx.x == 0) {
    red_release_sys(p.flag[s], 1u);
    const unsigned target = (round_ + 1u) * signals;
    while ((int)(ld_acquire_sys(p.flag[rank]) - target) < 0) __nanosleep(32);
  }
  __syncthreads();
  const long n2 = long(nrows ? min(nrows[rank], R) : R) * H / 2, slot2 = long(R) * H / 2;
  for (long i = (long)blockIdx.x * THR + threadIdx.x; i < n2; i += (long)gridDim.x * THR) {
    float2 a = __bfloat1622float2(recv[i]);
#pragma unroll
    for (int q = 1; q < W; ++q) {
      const float2 c = __bfloat1622float2(recv[(size_t)q * slot2 + i]);
      a.x += c.x;
      a.y += c.y;
    }
    out[i] = __floats2bfloat162_rn(a.x, a.y);
  }
}
// ---- LL ("low latency", NCCL's LL protocol) exchanges for tiny messages (for costing a fused small-batch layer) ----
// Every 16-B store carries 8 B of payload and the round's flag twice: {d0, f, d1, f} (st.volatile.v4; 8-B halves are
// single-copy atomic). No fence, no separate release flag: the receiver polls its words until both flags equal the round.
// Round flag: every CTA bumps the kernel's epoch word once, round = old / gridDim.x + 1 (rounds never overlap in a stream).
// LL buffers hold 2x the payload and are never cleared (flags are monotonic; u32 wrap after 4G rounds).
__device__ __forceinline__ void st_ll(uint4* p, unsigned d0, unsigned d1, unsigned f) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(d0), "r"(f), "r"(d1), "r"(f) : "memory");
}
__device__ __forceinline__ uint2 ld_ll(const uint4* p, unsigned f) {
  unsigned a, b, c, d;
  do {
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
  } while (b != f || d != f);
  return make_uint2(a, c);
}
__device__ __forceinline__ unsigned ll_round(unsigned* epoch) {
  __shared__ unsigned r_;
  if (threadIdx.x == 0) r_ = atomicAdd(epoch, 1u) / gridDim.x + 1u;
  __syncthreads();
  return r_;
}
// k_ll_ag: AG of every rank's R rows (n8 = R RB / 8 payload words of 8 B) into every rank's slot `rank` of the LL buffer
// ll[s] ([W][n8] 16-B words), then every CTA unpacks its share of all W slots into out ([W][n8 x 8 B] = gf's layout).
__global__ void __launch_bounds__(THR) k_ll_ag(const uint2* __restrict__ src, long n8, int rank, Peers p, unsigned* epoch,
                                               uint2* __restrict__ out) {
  const unsigned f = ll_round(epoch);
  const int nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W, b = blockIdx.x % nc;
  uint4* dst = (uint4*)p.buf[s] + (size_t)rank * n8;
  for (long i = (long)b * THR + threadIdx.x; i < n8; i += (long)nc * THR) {
    const uint2 v = src[i];
    st_ll(dst + i, v.x, v.y, f);
  }
  const uint4* mine = (const uint4*)p.buf[rank];
  for (long i = (long)blockIdx.x * THR + threadIdx.x; i < W * n8; i += (long)gridDim.x * THR) out[i] = ld_ll(mine + i, f);
}
// k_ll_rs: RS of part [W R, H] bf16: owner s's rows -> its LL slot `rank` ([W][n8] words, n8 = R H / 4), then every CTA
// sums its share of this rank's W slots in k_rs_sum's order (fp32, slots 0..3, one bf16 rounding) into out [R, H].
__global__ void __launch_bounds__(THR) k_ll_rs(const uint2* __restrict__ part, long n8, int rank, Peers p, unsigned* epoch,
                                               __nv_bfloat162* __restrict__ out) {
  const unsigned f = ll_round(epoch);
  const int nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W, b = blockIdx.x % nc;
  uint4* dst = (uint4*)p.buf[s] + (size_t)rank * n8;
  const uint2* src = part + (size_t)s * n8;
  for (long i = (long)b * THR + threadIdx.x; i < n8; i += (long)nc * THR) {
    const uint2 v = src[i];
    st_ll(dst + i, v.x, v.y, f);
  }
  const uint4* mine = (const uint4*)p.buf[rank];
  for (long i = (long)blockIdx.x * THR + threadIdx.x; i < n8; i += (long)gridDim.x * THR) {
    uint2 v = ld_ll(mine + i, f);
    float2 a0 = __bfloat1622float2(*(__nv_bfloat162*)&v.x), a1 = __bfloat1622float2(*(__nv_bfloat162*)&v.y);
#pragma unroll
    for (int q = 1; q < W; ++q) {
      const uint2 u = ld_ll(mine + (size_t)q * n8 + i, f);
      const float2 c0 = __bfloat1622float2(*(const __nv_bfloat162*)&u.x), c1 = __bfloat1622float2(*(const __nv_bfloat162*)&u.y);
      a0.x += c0.x;
      a0.y += c0.y;
      a1.x += c1.x;
      a1.y += c1.y;
    }
    out[2 * i] = __floats2bfloat162_rn(a0.x, a0.y);
    out[2 * i + 1] = __floats2bfloat162_rn(a1.x, a1.y);
  }
}
}  // namespace agrs

static agrs::Peers peers(const std::vector<torch::Tensor>& bufs, const std::vector<torch::Tensor>& flags) {
  agrs::Peers p;
  for (int s = 0; s < agrs::W; ++s) {
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

void push(torch::Tensor src, int64_t n16, int64_t src_stride16, int64_t rank, std::vector<torch::Tensor> bufs,
          std::vector<torch::Tensor> flags, torch::Tensor epoch, int64_t ncta, int64_t unr, int64_t mode) {
  auto st = at::cuda::getCurrentCUDAStream();
  auto p = peers(bufs, flags);
  const uint4* s = (const uint4*)src.data_ptr();
  unsigned* ep = (unsigned*)epoch.data_ptr();
  dim3 g(agrs::W * ncta);
#define L_(U, M) agrs::k_push<U, M><<<g, agrs::THR, 0, st>>>(s, n16, src_stride16, (int)rank, p, ep)
  if (mode == 0) {
    if (unr == 1) L_(1, 0); else if (unr == 4) L_(4, 0); else L_(8, 0);
  } else {
    if (unr == 1) L_(1, 1); else if (unr == 4) L_(4, 1); else L_(8, 1);
  }
#undef L_
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wait(torch::Tensor flag, torch::Tensor epoch, int64_t signals) {
  agrs::k_wait<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>((const unsigned*)flag.data_ptr(),
                                                               (const unsigned*)epoch.data_ptr(), (unsigned)signals);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sum(torch::Tensor recv, int64_t n2, torch::Tensor out, int64_t grid) {
  agrs::k_sum<<<grid, agrs::THR, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat162*)recv.data_ptr(), n2,
                                                                       (__nv_bfloat162*)out.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ce_copy(torch::Tensor dst, int64_t dst_off, torch::Tensor src, int64_t src_off, int64_t nbytes, int64_t stream) {
  C10_CUDA_CHECK(cudaMemcpyAsync((char*)dst.data_ptr() + dst_off, (const char*)src.data_ptr() + src_off, (size_t)nbytes,
                                 cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
}


void combine_push_dm(torch::Tensor Y, torch::Tensor slotmap, torch::Tensor sh, int64_t R, int64_t rank,
                     std::vector<torch::Tensor> bufs, std::vector<torch::Tensor> flags, torch::Tensor epoch, int64_t grid) {
  const int H = (int)Y.size(1);
  TORCH_CHECK(H / 8 <= 640 && grid % agrs::W == 0, "combine_push_dm: H / 8 <= 640 threads, grid a multiple of W");
  agrs::k_combine_push_dm<<<(int)grid, H / 8, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)Y.data_ptr(), slotmap.data_ptr<int>(), sh.numel() ? (const __nv_bfloat16*)sh.data_ptr() : nullptr,
      H, (int)R, (int)rank, peers(bufs, flags), (unsigned*)epoch.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ag1(torch::Tensor src, int64_t RB, torch::Tensor src2, int64_t RB2, int64_t R, c10::optional<torch::Tensor> nrows, int64_t rank,
         std::vector<torch::Tensor> bufs, std::vector<torch::Tensor> flags, std::vector<torch::Tensor> bufs2, torch::Tensor epoch,
         int64_t ncta) {
  auto p2 = peers(bufs2, flags);
  agrs::k_ag1<<<(int)(agrs::W * ncta), agrs::THR, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)src.data_ptr(), (int)RB, (const uint4*)src2.data_ptr(), (int)RB2, (int)R,
      nrows.has_value() ? nrows->data_ptr<int>() : nullptr, (int)rank, peers(bufs, flags), p2, (unsigned*)epoch.data_ptr(),
      (unsigned)(agrs::W * ncta));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rs1(torch::Tensor part, int64_t R, c10::optional<torch::Tensor> nrows, int64_t rank, std::vector<torch::Tensor> bufs,
         std::vector<torch::Tensor> flags, torch::Tensor epoch, torch::Tensor recv, torch::Tensor out, int64_t ncta) {
  const int H = (int)part.size(1);
  agrs::k_rs1<<<(int)(agrs::W * ncta), agrs::THR, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)part.data_ptr(), (int)R, H, nrows.has_value() ? nrows->data_ptr<int>() : nullptr, (int)rank, peers(bufs, flags),
      (unsigned*)epoch.data_ptr(), (unsigned)(agrs::W * ncta), (const __nv_bfloat162*)recv.data_ptr(), (__nv_bfloat162*)out.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// LL exchanges: bufs = every rank's LL buffer (>= W n8 16-B words), flags unused (any W tensors).
void ll_ag(torch::Tensor src, int64_t rank, std::vector<torch::Tensor> bufs, torch::Tensor epoch, torch::Tensor out, int64_t ncta) {
  const long n8 = (long)(src.numel() * src.element_size() / 8);
  agrs::k_ll_ag<<<(int)(agrs::W * ncta), agrs::THR, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint2*)src.data_ptr(), n8, (int)rank, peers(bufs, bufs), (unsigned*)epoch.data_ptr(), (uint2*)out.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void ll_rs(torch::Tensor part, int64_t rank, std::vector<torch::Tensor> bufs, torch::Tensor epoch, torch::Tensor out, int64_t ncta) {
  const long n8 = (long)(part.numel() * part.element_size() / 8 / agrs::W);
  agrs::k_ll_rs<<<(int)(agrs::W * ncta), agrs::THR, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint2*)part.data_ptr(), n8, (int)rank, peers(bufs, bufs), (unsigned*)epoch.data_ptr(), (__nv_bfloat162*)out.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ll_ag", &ll_ag);
  m.def("ll_rs", &ll_rs);
  m.def("ag1", &ag1);
  m.def("rs1", &rs1);
  m.def("enable_peers", &enable_peers);
  m.def("push", &push);
  m.def("wait", &wait);
  m.def("sum", &sum);
  m.def("ce_copy", &ce_copy);
  m.def("combine_push_dm", &combine_push_dm);
}
