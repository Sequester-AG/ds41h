// C1 variants of comm.cu's decode-bucket collectives.
// A decode bucket is padded to R rows per rank (32 at C1, of which 4 are one sequence's rows on its owner, 0 elsewhere);
// comm.cu moves every padded row. These kernels move only the REAL rows, told apart by their 64-B routing row
// (router_top_pack3 writes ids -1 into every padding row; a real row always has 6 experts, id 0 >= 0):
//   AG push2 (f || routing): the routing rows are always pushed (receivers must see the -1 ids of padding rows); an f row
//     only if its routing id 0 >= 0.
//   RS push: partial row t (all W * R gathered rows) only if gathered routing row t is real.
//   RS sum: only this rank's real rows (gathered routing rows rank * R + i).
// Every real row is copied / summed by the same statements as comm.cu (the same bits); a padding row's slot keeps
// whatever it held (finite values of earlier rows: nothing reads a padding row's result for a real row: every kernel of
// the step is row-independent, the MoE gathers routed (real) rows only). Flags / epochs / signal counts unchanged.
#include <cuda_bf16.h>
#include <cmath>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace spc1 {
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

// PDL (ENG_C1_PDL): launched with programmatic stream serialization, the blocks may be scheduled while the previous
// kernel drains; every kernel here waits for it before touching anything (no-op without the launch attribute)
// and lets its own dependents be scheduled at once (they wait for this kernel's completion in their own wait; every CTA of
// this grid is resident by then: the trigger counts only when all CTAs have issued it)
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ bool real_row(const unsigned char* routing, int row) {
  return *(const int*)(routing + (size_t)row * 64) >= 0;
}

// as comm.cu k_ag_push2 (SPC_UNR 1 loop): f rows of padding rows skipped. src2 = this rank's routing rows [R, 64].
__global__ void __launch_bounds__(THR) k_ag_push2v(const uint4* __restrict__ src, int RB, const uint4* __restrict__ src2,
                                                   int RB2, int R, int rank, Peers p, Peers p2, unsigned* epoch) {
  pdl_wait();
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int per = RB / 16, n16 = R * per, m16 = R * RB2 / 16, nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W,
            b = blockIdx.x % nc;
  uint4* dst = (uint4*)(p.buf[s] + (size_t)rank * R * RB);
  uint4* dst2 = (uint4*)(p2.buf[s] + (size_t)rank * R * RB2);
  const unsigned char* rt = (const unsigned char*)src2;
  for (int i = b * THR + threadIdx.x; i < m16; i += nc * THR) dst2[i] = src2[i];
  for (int i = b * THR + threadIdx.x; i < n16; i += nc * THR)
    if (real_row(rt, i / per)) dst[i] = src[i];
  fence_sys();
  __syncthreads();
  if (threadIdx.x == 0) red_release_sys(p.flag[s], 1u);
}

// k_ag_push2v + comm.cu's wait in ONE kernel: every CTA pushes and signals as above; the CTA that finishes last (ticket
// on a monotonic per-rank counter) then waits for this rank's flag (all W * NCTA signals of the round, its own grid's
// included: every other CTA of the grid has pushed before it took the last ticket, so no co-residency is needed).
__global__ void __launch_bounds__(THR) k_ag_push2v_wait(const uint4* __restrict__ src, int RB, const uint4* __restrict__ src2,
                                                        int RB2, int R, int rank, Peers p, Peers p2, unsigned* epoch,
                                                        const unsigned* myflag, unsigned signals, unsigned* ticket) {
  pdl_wait();
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int per = RB / 16, n16 = R * per, m16 = R * RB2 / 16, nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W,
            b = blockIdx.x % nc;
  uint4* dst = (uint4*)(p.buf[s] + (size_t)rank * R * RB);
  uint4* dst2 = (uint4*)(p2.buf[s] + (size_t)rank * R * RB2);
  const unsigned char* rt = (const unsigned char*)src2;
  for (int i = b * THR + threadIdx.x; i < m16; i += nc * THR) dst2[i] = src2[i];
  for (int i = b * THR + threadIdx.x; i < n16; i += nc * THR)
    if (real_row(rt, i / per)) dst[i] = src[i];
  fence_sys();
  __syncthreads();
  __shared__ unsigned last;
  if (threadIdx.x == 0) {
    red_release_sys(p.flag[s], 1u);
    __threadfence();
    last = (atomicAdd(ticket, 1u) % gridDim.x) == gridDim.x - 1;
  }
  __syncthreads();
  if (!last) return;
  if (threadIdx.x == 0) {
    const unsigned target = *(volatile const unsigned*)epoch * signals;
    while ((int)(ld_acquire_sys(myflag) - target) < 0) __nanosleep(32);
  }
}

// ENG_C1_AGRT: moe_raw.cu's router_top_pack3 folded into k_ag_push2v_wait. Each destination's CTA
// group computes the routing rows it pushes (one warp per row, the statements of moe_raw.cu k_router_top with
// MSR_RT_PAR 0: the same bits) straight into the destination's gathered rows; the self group also writes this rank's
// ids / wts / routing rows as router_top_pack3 does. f rows are pushed for the valid rows (= the rows whose routing id 0
// is >= 0 in the unfused path). Flags, epochs, ticket and the last CTA's wait as k_ag_push2v_wait.
constexpr int NEXP = 384;
__global__ void __launch_bounds__(THR) k_ag_rt(const uint4* __restrict__ src, int RB, const float* __restrict__ logits,
                                               const float* __restrict__ bias, float route_scale,
                                               const unsigned char* __restrict__ vmask, const int* __restrict__ ptok_in,
                                               int* __restrict__ ids, float* __restrict__ wts, unsigned* __restrict__ rrow,
                                               int R, int rank, Peers p, Peers p2, unsigned* epoch, const unsigned* myflag,
                                               unsigned signals, unsigned* ticket) {
  pdl_wait();
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int per = RB / 16, n16 = R * per, nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W, b = blockIdx.x % nc;
  constexpr int NWP = THR / 32;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint4* dst = (uint4*)(p.buf[s] + (size_t)rank * R * RB);
  unsigned* dst2 = (unsigned*)(p2.buf[s] + (size_t)rank * R * 64);
  for (int row = b * NWP + warp; row < R; row += nc * NWP) {
    float sc[12], sel[12];
#pragma unroll
    for (int i = 0; i < 12; ++i) {
      const int e = i * 32 + lane;
      const float x = logits[(size_t)row * NEXP + e];
      sc[i] = sqrtf(x > 20.f ? x : log1pf(expf(x)));
      sel[i] = sc[i] + bias[e];
    }
    float pick[6];
    int pid[6];
    unsigned taken = 0;
#pragma unroll
    for (int r = 0; r < 6; ++r) {
      float bv = -INFINITY;
      int bi = 1 << 30;
#pragma unroll
      for (int i = 0; i < 12; ++i)
        if (!((taken >> i) & 1) && sel[i] > bv) bv = sel[i], bi = i * 32 + lane;
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
      }
      pid[r] = bi;
      if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
      float s_r = 0.f;
#pragma unroll
      for (int i = 0; i < 12; ++i)
        if (i * 32 + lane == bi) s_r = sc[i];
      pick[r] = __shfl_sync(0xffffffffu, s_r, bi & 31);
    }
    float sum = 0.f;  // every lane holds all picks: the same sum in the same order as lane 0 of k_router_top
#pragma unroll
    for (int r = 0; r < 6; ++r) sum += pick[r];
    const bool real = vmask[row] != 0;
    unsigned word = 0u;
    int myid = 0;
    float mywt = 0.f;
#pragma unroll
    for (int r = 0; r < 6; ++r) {
      if (lane == r) myid = pid[r], mywt = pick[r] / (sum + 1e-20f) * route_scale;
      if (lane == 6 + r) mywt = pick[r] / (sum + 1e-20f) * route_scale;
    }
    if (!real) word = lane < 6 ? 0xFFFFFFFFu : 0u;
    else if (lane < 6) word = (unsigned)myid;
    else if (lane < 12) word = __float_as_uint(mywt);
    else if (lane == 12) word = (unsigned)ptok_in[row];
    if (lane < 16) {
      dst2[(size_t)row * 16 + lane] = word;
      if (s == rank) rrow[(size_t)row * 16 + lane] = word;
    }
    if (s == rank && lane < 6) {
      ids[row * 6 + lane] = myid;
      wts[row * 6 + lane] = mywt;
    }
  }
  for (int i = b * THR + threadIdx.x; i < n16; i += nc * THR)
    if (vmask[i / per]) dst[i] = src[i];
  fence_sys();
  __syncthreads();
  __shared__ unsigned last;
  if (threadIdx.x == 0) {
    red_release_sys(p.flag[s], 1u);
    __threadfence();
    last = (atomicAdd(ticket, 1u) % gridDim.x) == gridDim.x - 1;
  }
  __syncthreads();
  if (!last) return;
  if (threadIdx.x == 0) {
    const unsigned target = *(volatile const unsigned*)epoch * signals;
    while ((int)(ld_acquire_sys(myflag) - target) < 0) __nanosleep(32);
  }
}

// as comm.cu k_rs_push: rows of padding skipped. gr = this rank's gathered routing rows [W * R, 64].
__global__ void __launch_bounds__(THR) k_rs_pushv(const uint4* __restrict__ part, int R, int H, int rank, Peers p,
                                                  unsigned* epoch, const unsigned char* __restrict__ gr) {
  pdl_wait();
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int per = H * 2 / 16, n16 = R * per, nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W, b = blockIdx.x % nc;
  const uint4* src = part + (size_t)s * n16;
  uint4* dst = (uint4*)(p.buf[s] + (size_t)rank * R * H * 2);
  for (int i = b * THR + threadIdx.x; i < n16; i += nc * THR)
    if (real_row(gr, s * R + i / per)) dst[i] = src[i];
  fence_sys();
  __syncthreads();
  if (threadIdx.x == 0) red_release_sys(p.flag[s], 1u);
}

// ENG_C1_RSC: moe_raw.cu's k_combine folded into k_rs_pushv: each pushed 16-B chunk of a real
// gathered row t is combined here (the 6 slot rows of Y summed in fp32 in slot order, + the shared expert's row, one bf16
// rounding: k_combine's statements, the same bits) and stored straight into the owner's slot; the local `part` copy is
// not written (nothing else reads it in decode buckets).
__global__ void __launch_bounds__(THR) k_rs_cpushv(const __nv_bfloat16* __restrict__ Y, const int* __restrict__ slotmap,
                                                   const __nv_bfloat16* __restrict__ sh, int R, int H, int rank, Peers p,
                                                   unsigned* epoch, const unsigned char* __restrict__ gr) {
  pdl_wait();
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int per = H * 2 / 16, n16 = R * per, nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W, b = blockIdx.x % nc;
  uint4* dst = (uint4*)(p.buf[s] + (size_t)rank * R * H * 2);
  for (int i = b * THR + threadIdx.x; i < n16; i += nc * THR) {
    const int t = s * R + i / per, d0 = (i % per) * 8;
    if (!real_row(gr, t)) continue;
    int sl[6];
#pragma unroll
    for (int k = 0; k < 6; ++k) sl[k] = slotmap[6 * t + k];
    float acc[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) acc[q] = 0.f;
    uint4 v[6];
#pragma unroll
    for (int k = 0; k < 6; ++k) v[k] = sl[k] >= 0 ? *(const uint4*)(Y + (size_t)sl[k] * H + d0) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      const __nv_bfloat162* h = (const __nv_bfloat162*)&v[k];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = __bfloat1622float2(h[q]);
        acc[2 * q] += f.x, acc[2 * q + 1] += f.y;
      }
    }
    const uint4 sv = sh ? *(const uint4*)(sh + (size_t)t * H + d0) : make_uint4(0, 0, 0, 0);
    const __nv_bfloat162* hs = (const __nv_bfloat162*)&sv;
    uint4 o;
    __nv_bfloat162* ho = (__nv_bfloat162*)&o;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 f = __bfloat1622float2(hs[q]);
      ho[q] = __floats2bfloat162_rn(acc[2 * q] + f.x, acc[2 * q + 1] + f.y);
    }
    dst[i] = o;
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

// as comm.cu k_rs_sum (same fp32 order, one bf16 rounding): this rank's real rows only.
__global__ void __launch_bounds__(THR) k_rs_sumv(const __nv_bfloat162* __restrict__ recv, int R, int H,
                                                 const unsigned* flag, const unsigned* epoch, unsigned signals,
                                                 __nv_bfloat162* __restrict__ out, const unsigned char* __restrict__ gr,
                                                 int rank) {
  pdl_wait();
  wait_flag(flag, epoch, signals);
  const int per = H / 2, n2 = R * per;
  for (int i = blockIdx.x * THR + threadIdx.x; i < n2; i += gridDim.x * THR) {
    if (!real_row(gr, rank * R + i / per)) continue;
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

// ---- TP-wo (ENG_C1_TPWO: same math, not bitwise): a few-row decode step's wo_a / wo_b split over the 4 GPUs like
// production's TP4. GPU g owns heads 16 g .. 16 g + 15 = wo_a groups 2 g, 2 g + 1 = z columns 2048 g .. and wo_b's K
// quarter g. k_tp_scatter: every rank pushes its real rows' attention output columns [8192 s, 8192 (s + 1)) to rank s
// (+ its row mask), the last CTA waits for all of this round's pushes into its own buffer. k_tp_rs: rank g pushes the
// fp32 wo_b partial rows of rank s's real rows to rank s; k_tp_rs_sum: the owner sums the 4 partials (fp32, rank
// order) into bf16 y for its real rows.
// inverse RoPE of 4 bf16 pairs (dims 2i, 2i+1, i = i0 .. i0 + 3 of a head's RoPE part), dense.cu rope_pair<true>'s formula
__device__ __forceinline__ uint4 inv_rope4(uint4 v, const float* cs, int i0) {
  __nv_bfloat162* h = (__nv_bfloat162*)&v;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float c = __ldg(cs + i0 + k), sn = __ldg(cs + 32 + i0 + k);
    const float2 f = __bfloat1622float2(h[k]);
    const float a = __fadd_rn(__fmul_rn(f.x, c), __fmul_rn(f.y, sn));
    const float b = __fsub_rn(__fmul_rn(f.y, c), __fmul_rn(f.x, sn));
    h[k] = __halves2bfloat162(__float2bfloat16_rn(a), __float2bfloat16_rn(b));
  }
  return v;
}

// pos / cos_sin (nullptr: none): the inverse RoPE of every head's dims 448..511 applied on the way (instead of
// dense.cu's inv_rope kernel before)
__global__ void __launch_bounds__(THR) k_tp_scatter(const uint4* __restrict__ o, const unsigned char* __restrict__ valid, int R,
                                                    int rank, Peers p, Peers pv, unsigned* epoch, const unsigned* myflag,
                                                    unsigned signals, unsigned* ticket, const long long* __restrict__ pos,
                                                    const float* __restrict__ cos_sin) {
  pdl_wait();
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  const int nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W, b = blockIdx.x % nc;
  constexpr int ROW16 = 32768 * 2 / 16, SL16 = 8192 * 2 / 16;   // uint4 per o row / per head-group slice
  uint4* dst = (uint4*)(p.buf[s] + (size_t)rank * R * 8192 * 2);
  if (b == 0 && threadIdx.x < R) ((unsigned char*)pv.buf[s])[rank * R + threadIdx.x] = valid[threadIdx.x];
  for (int i = b * THR + threadIdx.x; i < R * SL16; i += nc * THR) {
    const int row = i / SL16, c = i % SL16;
    if (!valid[row]) continue;
    uint4 v = o[(size_t)row * ROW16 + s * SL16 + c];
    const int hc = c & 63;                      // uint4 inside the head (64 per 512-dim head); 56..63 = the RoPE dims
    if (pos && hc >= 56) v = inv_rope4(v, cos_sin + __ldg(pos + row) * 64, (hc - 56) * 4);
    dst[i] = v;
  }
  fence_sys();
  __syncthreads();
  __shared__ unsigned last;
  if (threadIdx.x == 0) {
    red_release_sys(p.flag[s], 1u);
    __threadfence();
    last = (atomicAdd(ticket, 1u) % gridDim.x) == gridDim.x - 1;
  }
  __syncthreads();
  if (!last) return;
  if (threadIdx.x == 0) {
    const unsigned target = *(volatile const unsigned*)epoch * signals;
    while ((int)(ld_acquire_sys(myflag) - target) < 0) __nanosleep(32);
  }
}

// part [W * R, 5120] fp32 (all ranks' rows on this rank), vm [W * R] gathered row masks
__global__ void __launch_bounds__(THR) k_tp_rs(const float4* __restrict__ part, const unsigned char* __restrict__ vm, int R,
                                               int rank, Peers p, unsigned* epoch) {
  pdl_wait();
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(epoch, 1u);
  constexpr int ROW4 = 5120 / 4;
  const int nc = gridDim.x / W, s = (rank + 1 + blockIdx.x / nc) % W, b = blockIdx.x % nc;
  const float4* src = part + (size_t)s * R * ROW4;
  float4* dst = (float4*)(p.buf[s] + (size_t)rank * R * 5120 * 4);
  for (int i = b * THR + threadIdx.x; i < R * ROW4; i += nc * THR)
    if (vm[s * R + i / ROW4]) dst[i] = src[i];
  fence_sys();
  __syncthreads();
  if (threadIdx.x == 0) red_release_sys(p.flag[s], 1u);
}

// recv [W, R, 5120] fp32 (slot = source rank) -> y [R, 5120] bf16 for this rank's real rows
__global__ void __launch_bounds__(THR) k_tp_rs_sum(const float2* __restrict__ recv, int R, const unsigned* flag,
                                                   const unsigned* epoch, unsigned signals, __nv_bfloat162* __restrict__ y,
                                                   const unsigned char* __restrict__ valid) {
  pdl_wait();
  wait_flag(flag, epoch, signals);
  constexpr int ROW2 = 5120 / 2;
  const int n2 = R * ROW2;
  for (int i = blockIdx.x * THR + threadIdx.x; i < n2; i += gridDim.x * THR) {
    if (!valid[i / ROW2]) continue;
    float2 a = recv[i];
#pragma unroll
    for (int s = 1; s < W; ++s) {
      const float2 b = recv[(size_t)s * n2 + i];
      a.x += b.x;
      a.y += b.y;
    }
    y[i] = __floats2bfloat162_rn(a.x, a.y);
  }
}
// ---- ENG_C1_RSL: lamport-style TP-wo reduce-scatter in ONE kernel per rank.
// Each rank pushes its fp32 partials of the owners' real rows straight into the owner's receive slots; the owner polls
// each element until it is not the sentinel (-0.0: a sender writes a -0.0 partial as +0.0), sums the 4 sources in rank
// order (its own partial read locally: the order and the values of k_tp_rs + k_tp_rs_sum), writes y and resets its
// slots to the sentinel. No fence, no flag round trip, one launch instead of two. Slots double-buffered by the parity of
// a per-CTA call counter (every rank runs the same calls with the same grid). lbuf[s] = rank s's slots
// [2][W][R][5120] fp32, filled with -0.0 at allocation.
constexpr unsigned SENT = 0x80000000u;
struct LPeers {
  float* buf[W];
};
__device__ __forceinline__ float4 nz4(float4 v) {  // -0.0 -> +0.0 (the sentinel never travels)
  unsigned* u = reinterpret_cast<unsigned*>(&v);
#pragma unroll
  for (int k = 0; k < 4; ++k)
    if (u[k] == SENT) u[k] = 0u;
  return v;
}
__device__ __forceinline__ uint4 ld_vol4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__global__ void __launch_bounds__(THR) k_tp_rsl(const float4* __restrict__ part, const unsigned char* __restrict__ vm, int R,
                                                int rank, LPeers p, unsigned* __restrict__ tick,
                                                __nv_bfloat162* __restrict__ y, const unsigned char* __restrict__ valid) {
  pdl_wait();
  __shared__ unsigned par_s;
  if (threadIdx.x == 0) par_s = tick[blockIdx.x] & 1u;
  __syncthreads();
  const int par = (int)par_s;
  constexpr int ROW4 = 5120 / 4;
  const int nth = gridDim.x * THR, tid = blockIdx.x * THR + threadIdx.x;
  // push: the other ranks' real rows of my partial
  for (int d = 1; d < W; ++d) {
    const int s = (rank + d) % W;
    float4* dst = reinterpret_cast<float4*>(p.buf[s]) + ((size_t)(par * W + rank) * R) * ROW4;
    const float4* src = part + (size_t)s * R * ROW4;
    for (int i = tid; i < R * ROW4; i += nth)
      if (vm[s * R + i / ROW4]) dst[i] = nz4(src[i]);
  }
  // my real rows: poll the 3 peers' slots, sum the 4 sources in rank order, reset the slots
  float4* mine = reinterpret_cast<float4*>(p.buf[rank]) + (size_t)par * W * R * ROW4;
  const float4* own = part + (size_t)rank * R * ROW4;
  for (int i = tid; i < R * ROW4; i += nth) {
    if (!valid[i / ROW4]) continue;
    float4 v[W];
#pragma unroll
    for (int s = 0; s < W; ++s) {
      if (s == rank) {
        v[s] = own[i];
        continue;
      }
      float4* slot = mine + (size_t)s * R * ROW4 + i;
      uint4 u;
      do {
        u = ld_vol4(slot);
      } while (u.x == SENT || u.y == SENT || u.z == SENT || u.w == SENT);
      v[s] = *reinterpret_cast<float4*>(&u);
      *reinterpret_cast<uint4*>(slot) = make_uint4(SENT, SENT, SENT, SENT);
    }
    float4 a = v[0];
#pragma unroll
    for (int s = 1; s < W; ++s) a.x += v[s].x, a.y += v[s].y, a.z += v[s].z, a.w += v[s].w;
    y[2 * i] = __floats2bfloat162_rn(a.x, a.y);
    y[2 * i + 1] = __floats2bfloat162_rn(a.z, a.w);
  }
  __syncthreads();
  if (threadIdx.x == 0) tick[blockIdx.x] += 1u;
}
// ---- ENG_C1_RSL, MoE side: k_rs_cpushv (combine + push) + k_rs_sumv (flag wait + rank-order sum) as ONE lamport
// kernel per rank. The combined bf16 row (k_rs_cpushv's statements, a -0.0 bf16 sent as +0.0) goes straight into the
// owner's slot; the owner combines its own rows locally, polls the 3 peers' 16-B chunks until no bf16 half is the
// sentinel 0x8000, sums the 4 sources in fp32 rank order (k_rs_sumv's statements), writes out and resets its slots.
// lbuf[s] = rank s's slots [2][W][R][H] bf16, filled with -0.0.
__device__ __forceinline__ uint4 combine16(const __nv_bfloat16* __restrict__ Y, const int* __restrict__ slotmap,
                                           const __nv_bfloat16* __restrict__ sh, int t, int H, int d0) {
  int sl[6];
#pragma unroll
  for (int k = 0; k < 6; ++k) sl[k] = slotmap[6 * t + k];
  float acc[8];
#pragma unroll
  for (int q = 0; q < 8; ++q) acc[q] = 0.f;
  uint4 v[6];
#pragma unroll
  for (int k = 0; k < 6; ++k) v[k] = sl[k] >= 0 ? *(const uint4*)(Y + (size_t)sl[k] * H + d0) : make_uint4(0, 0, 0, 0);
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    const __nv_bfloat162* h = (const __nv_bfloat162*)&v[k];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 f = __bfloat1622float2(h[q]);
      acc[2 * q] += f.x, acc[2 * q + 1] += f.y;
    }
  }
  const uint4 sv = sh ? *(const uint4*)(sh + (size_t)t * H + d0) : make_uint4(0, 0, 0, 0);
  const __nv_bfloat162* hs = (const __nv_bfloat162*)&sv;
  uint4 o;
  __nv_bfloat162* ho = (__nv_bfloat162*)&o;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float2 f = __bfloat1622float2(hs[q]);
    ho[q] = __floats2bfloat162_rn(acc[2 * q] + f.x, acc[2 * q + 1] + f.y);
  }
  return o;
}
__device__ __forceinline__ unsigned nzh2(unsigned u) {   // bf16 -0.0 halves -> +0.0
  if ((u & 0xffffu) == 0x8000u) u &= 0xffff0000u;
  if ((u >> 16) == 0x8000u) u &= 0x0000ffffu;
  return u;
}
__device__ __forceinline__ bool has_sent(unsigned u) { return (u & 0xffffu) == 0x8000u || (u >> 16) == 0x8000u; }
struct LPeersB {
  __nv_bfloat16* buf[W];
};
__global__ void __launch_bounds__(THR) k_rs_cpushl(const __nv_bfloat16* __restrict__ Y, const int* __restrict__ slotmap,
                                                   const __nv_bfloat16* __restrict__ sh, int R, int H, int rank, LPeersB p,
                                                   unsigned* __restrict__ tick, __nv_bfloat162* __restrict__ out,
                                                   const unsigned char* __restrict__ gr) {
  pdl_wait();
  __shared__ unsigned par_s;
  if (threadIdx.x == 0) par_s = tick[blockIdx.x] & 1u;
  __syncthreads();
  const int par = (int)par_s, per = H / 8, n16 = R * per;
  const int nth = gridDim.x * THR, tid = blockIdx.x * THR + threadIdx.x;
  for (int d = 1; d < W; ++d) {   // push the other ranks' real rows
    const int s = (rank + d) % W;
    uint4* dst = (uint4*)(p.buf[s] + ((size_t)(par * W + rank) * R) * H);
    for (int i = tid; i < n16; i += nth) {
      const int t = s * R + i / per;
      if (!real_row(gr, t)) continue;
      uint4 o = combine16(Y, slotmap, sh, t, H, (i % per) * 8);
      o.x = nzh2(o.x), o.y = nzh2(o.y), o.z = nzh2(o.z), o.w = nzh2(o.w);
      dst[i] = o;
    }
  }
  uint4* mine = (uint4*)(p.buf[rank] + (size_t)par * W * R * H);
  for (int i = tid; i < n16; i += nth) {   // my real rows
    const int t = rank * R + i / per;
    if (!real_row(gr, t)) continue;
    uint4 v[W];
#pragma unroll
    for (int s = 0; s < W; ++s) {
      if (s == rank) {
        v[s] = combine16(Y, slotmap, sh, t, H, (i % per) * 8);
        continue;
      }
      uint4* slot = mine + (size_t)s * n16 + i;
      uint4 u;
      do {
        u = ld_vol4(slot);
      } while (has_sent(u.x) || has_sent(u.y) || has_sent(u.z) || has_sent(u.w));
      v[s] = u;
      *slot = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {   // k_rs_sumv: fp32, rank order, one bf16 rounding
      float2 a = __bfloat1622float2(((const __nv_bfloat162*)&v[0])[q]);
#pragma unroll
      for (int s = 1; s < W; ++s) {
        const float2 b = __bfloat1622float2(((const __nv_bfloat162*)&v[s])[q]);
        a.x += b.x;
        a.y += b.y;
      }
      out[(size_t)i * 4 + q] = __floats2bfloat162_rn(a.x, a.y);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) tick[blockIdx.x] += 1u;
}
// ---- c1l building block: lamport ALL-reduce of NR fp32 rows [NR, 5120]. Every
// rank pushes its partial rows to every peer's slots [2][W][NR][5120] (sentinel -0.0, a -0.0 partial travels as +0.0),
// then every rank polls the 3 peers, sums the 4 sources in rank order (its own read locally) and writes y bf16: the
// copies on the 4 GPUs are bit-identical (the replicated residual's all-reduce).
__global__ void __launch_bounds__(THR) k_arl(const float4* __restrict__ part, int NR, int rank, LPeers p,
                                             unsigned* __restrict__ tick, __nv_bfloat162* __restrict__ y) {
  pdl_wait();
  __shared__ unsigned par_s;
  if (threadIdx.x == 0) par_s = tick[blockIdx.x] & 1u;
  __syncthreads();
  const int par = (int)par_s;
  constexpr int ROW4 = 5120 / 4;
  const int n4 = NR * ROW4, nth = gridDim.x * THR, tid = blockIdx.x * THR + threadIdx.x;
  for (int d = 1; d < W; ++d) {
    const int s = (rank + d) % W;
    float4* dst = reinterpret_cast<float4*>(p.buf[s]) + (size_t)(par * W + rank) * n4;
    for (int i = tid; i < n4; i += nth) dst[i] = nz4(part[i]);
  }
  float4* mine = reinterpret_cast<float4*>(p.buf[rank]) + (size_t)par * W * n4;
  for (int i = tid; i < n4; i += nth) {
    float4 v[W];
#pragma unroll
    for (int s = 0; s < W; ++s) {
      if (s == rank) {
        v[s] = part[i];
        continue;
      }
      float4* slot = mine + (size_t)s * n4 + i;
      uint4 u;
      do {
        u = ld_vol4(slot);
      } while (u.x == SENT || u.y == SENT || u.z == SENT || u.w == SENT);
      v[s] = *reinterpret_cast<float4*>(&u);
      *reinterpret_cast<uint4*>(slot) = make_uint4(SENT, SENT, SENT, SENT);
    }
    float4 a = v[0];
#pragma unroll
    for (int s = 1; s < W; ++s) a.x += v[s].x, a.y += v[s].y, a.z += v[s].z, a.w += v[s].w;
    y[2 * i] = __floats2bfloat162_rn(a.x, a.y);
    y[2 * i + 1] = __floats2bfloat162_rn(a.z, a.w);
  }
  __syncthreads();
  if (threadIdx.x == 0) tick[blockIdx.x] += 1u;
}
}  // namespace spc1

#include "comm_c1_cabi.h"
#include "../../engine/mixed_cxx/fpr_cabi_rec.h"   // rule 9: optional record hook
FPR_CABI_RECORD_DECL(comm_c1)

// ENG_C1_PDL, read once per process: latched by "comm_c1.init" (C path: before the driver threads launch) or by the first
// pybind launch, as before. Atomic (4 driver threads); -1 = not latched yet: the C cores never read the environment.
static int g_pdl = -1;
static bool pdl_on() {
  int v = __atomic_load_n(&g_pdl, __ATOMIC_ACQUIRE);
  if (v < 0) {
    const char* e = getenv("ENG_C1_PDL");
    v = (e && e[0] == '1') ? 1 : 0;
    __atomic_store_n(&g_pdl, v, __ATOMIC_RELEASE);
  }
  return v == 1;
}
// every launch of this file (C cores and pybind): the caller's stream, PDL attribute = the latched ENG_C1_PDL; no exceptions
template <typename... KArgs, typename... Args>
static int launch_c(cudaStream_t st, void (*k)(KArgs...), int grid, int block, Args... args) {
  const int pdl = __atomic_load_n(&g_pdl, __ATOMIC_ACQUIRE);
  if (pdl < 0) return FPR_E_INIT;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, k, args...);
  const cudaError_t l = cudaGetLastError();   // (also clears a failed launch's error, as C10_CUDA_CHECK did)
  return e == cudaSuccess && l == cudaSuccess ? FPR_OK : FPR_E_CUDA;
}
// pybind side: the FprCall of the current stream (ENG_C1_PDL latched first, as the first launch always did)
static FprCall pyc(const torch::Tensor& t, int64_t rank) {
  (void)pdl_on();
  return FprCall{at::cuda::getCurrentCUDAStream(), (int)t.get_device(), (int)rank};
}
static void chk(int rc, const char* what) { TORCH_CHECK(rc == FPR_OK, what, ": C core error ", rc); }

static spc1::Peers peers_c(void* const* bufs, uint32_t* const* flags) {
  spc1::Peers p;
  for (int s = 0; s < spc1::W; ++s) {
    p.buf[s] = (char*)bufs[s];
    p.flag[s] = (unsigned*)flags[s];
  }
  return p;
}
template <typename A>
static void peers_py(A& ca, const std::vector<torch::Tensor>& bufs, const std::vector<torch::Tensor>& flags) {
  for (int s = 0; s < spc1::W; ++s) ca.bufs[s] = bufs[s].data_ptr(), ca.flags[s] = (uint32_t*)flags[s].data_ptr();
}

static int ag_push2v_core(const FprCommC1AgPush2vArgs* p) {   // C ABI "comm_c1.ag_push2v": the launch of pybind ag_push2v
  try {
    if (p->RB % 16 != 0 || p->RB2 != 64) return FPR_E_ARG;
    return launch_c(p->c.stream, spc1::k_ag_push2v, (int)p->ncta * spc1::W, spc1::THR, (const uint4*)p->src, (int)p->RB,
                    (const uint4*)p->src2, (int)p->RB2, (int)p->R, (int)p->c.rank, peers_c(p->bufs, p->flags),
                    peers_c(p->bufs2, p->flags), (unsigned*)p->epoch);
  } catch (...) {
    return FPR_E_CUDA;
  }
}
void ag_push2v(torch::Tensor src, torch::Tensor src2, int64_t rank, std::vector<torch::Tensor> bufs,
               std::vector<torch::Tensor> flags, std::vector<torch::Tensor> bufs2, torch::Tensor epoch, int64_t ncta) {
  const int R = (int)src.size(0), RB = (int)(src.numel() / src.size(0) * src.element_size());
  const int RB2 = (int)(src2.numel() / src2.size(0) * src2.element_size());
  TORCH_CHECK(RB % 16 == 0 && RB2 == 64 && src2.size(0) == R, "ag_push2v: shapes");
  FprCommC1AgPush2vArgs ca{};
  ca.c = pyc(src, rank);
  ca.src = src.data_ptr(), ca.src2 = src2.data_ptr(), ca.R = R, ca.RB = RB, ca.RB2 = RB2, ca.ncta = (int32_t)ncta;
  peers_py(ca, bufs, flags);
  for (int s = 0; s < spc1::W; ++s) ca.bufs2[s] = bufs2[s].data_ptr();
  ca.epoch = (uint32_t*)epoch.data_ptr();
  FPR_CABI_RECORD("comm_c1.ag_push2v", ca);   // rule 9 (fpr_cabi_rec.h)
  chk(ag_push2v_core(&ca), "comm_c1 ag_push2v");
}

static int ag_push2v_wait_core(const FprCommC1AgPush2vWaitArgs* p) {   // C ABI "comm_c1.ag_push2v_wait"
  try {
    if (p->RB % 16 != 0 || p->RB2 != 64) return FPR_E_ARG;
    return launch_c(p->c.stream, spc1::k_ag_push2v_wait, (int)p->ncta * spc1::W, spc1::THR, (const uint4*)p->src, (int)p->RB,
                    (const uint4*)p->src2, (int)p->RB2, (int)p->R, (int)p->c.rank, peers_c(p->bufs, p->flags),
                    peers_c(p->bufs2, p->flags), (unsigned*)p->epoch, (const unsigned*)p->myflag, (unsigned)p->signals,
                    (unsigned*)p->ticket);
  } catch (...) {
    return FPR_E_CUDA;
  }
}
void ag_push2v_wait(torch::Tensor src, torch::Tensor src2, int64_t rank, std::vector<torch::Tensor> bufs,
                    std::vector<torch::Tensor> flags, std::vector<torch::Tensor> bufs2, torch::Tensor epoch, int64_t ncta,
                    torch::Tensor myflag, int64_t signals, torch::Tensor ticket) {
  const int R = (int)src.size(0), RB = (int)(src.numel() / src.size(0) * src.element_size());
  const int RB2 = (int)(src2.numel() / src2.size(0) * src2.element_size());
  TORCH_CHECK(RB % 16 == 0 && RB2 == 64 && src2.size(0) == R && ticket.numel() >= 1, "ag_push2v_wait: shapes");
  FprCommC1AgPush2vWaitArgs ca{};
  ca.c = pyc(src, rank);
  ca.src = src.data_ptr(), ca.src2 = src2.data_ptr(), ca.R = R, ca.RB = RB, ca.RB2 = RB2, ca.ncta = (int32_t)ncta;
  peers_py(ca, bufs, flags);
  for (int s = 0; s < spc1::W; ++s) ca.bufs2[s] = bufs2[s].data_ptr();
  ca.epoch = (uint32_t*)epoch.data_ptr(), ca.myflag = (const uint32_t*)myflag.data_ptr();
  ca.ticket = (uint32_t*)ticket.data_ptr(), ca.signals = (uint32_t)signals;
  FPR_CABI_RECORD("comm_c1.ag_push2v_wait", ca);   // rule 9 (fpr_cabi_rec.h)
  chk(ag_push2v_wait_core(&ca), "comm_c1 ag_push2v_wait");
}

static int ag_rt_core(const FprCommC1AgRtArgs* p) {   // C ABI "comm_c1.ag_rt"
  try {
    if (p->RB % 16 != 0) return FPR_E_ARG;
    return launch_c(p->c.stream, spc1::k_ag_rt, (int)p->ncta * spc1::W, spc1::THR, (const uint4*)p->src, (int)p->RB,
                    (const float*)p->logits, (const float*)p->bias, (float)p->route_scale, (const unsigned char*)p->valid,
                    (const int*)p->ptok, (int*)p->ids, (float*)p->wts, (unsigned*)p->rrow, (int)p->R, (int)p->c.rank,
                    peers_c(p->bufs, p->flags), peers_c(p->bufs2, p->flags), (unsigned*)p->epoch, (const unsigned*)p->myflag,
                    (unsigned)p->signals, (unsigned*)p->ticket);
  } catch (...) {
    return FPR_E_CUDA;
  }
}
void ag_rt(torch::Tensor src, torch::Tensor logits, torch::Tensor bias, double route_scale, torch::Tensor valid,
           torch::Tensor ptok, torch::Tensor ids, torch::Tensor wts, torch::Tensor rrow, int64_t rank,
           std::vector<torch::Tensor> bufs, std::vector<torch::Tensor> flags, std::vector<torch::Tensor> bufs2,
           torch::Tensor epoch, int64_t ncta, torch::Tensor myflag, int64_t signals, torch::Tensor ticket) {
  const int R = (int)src.size(0), RB = (int)(src.numel() / src.size(0) * src.element_size());
  TORCH_CHECK(RB % 16 == 0 && logits.scalar_type() == torch::kFloat32 && logits.size(0) >= R && logits.size(1) == spc1::NEXP &&
                  logits.is_contiguous() && rrow.numel() >= (int64_t)R * 64 && valid.element_size() == 1 && valid.numel() >= R &&
                  ptok.numel() >= R && ids.numel() >= (int64_t)R * 6 && wts.numel() >= (int64_t)R * 6 && ticket.numel() >= 1,
              "ag_rt: shapes");
  FprCommC1AgRtArgs ca{};
  ca.c = pyc(src, rank);
  ca.src = src.data_ptr(), ca.logits = logits.data_ptr<float>(), ca.bias = bias.data_ptr<float>();
  ca.valid = (const uint8_t*)valid.data_ptr(), ca.ptok = ptok.data_ptr<int>(), ca.ids = ids.data_ptr<int>();
  ca.wts = wts.data_ptr<float>(), ca.rrow = rrow.data_ptr(), ca.route_scale = (float)route_scale;
  ca.R = R, ca.RB = RB, ca.ncta = (int32_t)ncta;
  peers_py(ca, bufs, flags);
  for (int s = 0; s < spc1::W; ++s) ca.bufs2[s] = bufs2[s].data_ptr();
  ca.epoch = (uint32_t*)epoch.data_ptr(), ca.myflag = (const uint32_t*)myflag.data_ptr();
  ca.ticket = (uint32_t*)ticket.data_ptr(), ca.signals = (uint32_t)signals;
  FPR_CABI_RECORD("comm_c1.ag_rt", ca);   // rule 9 (fpr_cabi_rec.h)
  chk(ag_rt_core(&ca), "comm_c1 ag_rt");
}

static int rs_cpushv_core(const FprCommC1RsCpushvArgs* p) {   // C ABI "comm_c1.rs_cpushv"
  try {
    if (p->H % 8 != 0) return FPR_E_ARG;
    return launch_c(p->c.stream, spc1::k_rs_cpushv, (int)p->ncta * spc1::W, spc1::THR, (const __nv_bfloat16*)p->Y,
                    (const int*)p->slotmap, (const __nv_bfloat16*)p->sh, (int)p->R, (int)p->H, (int)p->c.rank,
                    peers_c(p->bufs, p->flags), (unsigned*)p->epoch, (const unsigned char*)p->gr);
  } catch (...) {
    return FPR_E_CUDA;
  }
}
void rs_cpushv(torch::Tensor Y, torch::Tensor slotmap, torch::Tensor sh, int64_t R, int64_t rank, std::vector<torch::Tensor> bufs,
               std::vector<torch::Tensor> flags, torch::Tensor epoch, int64_t ncta, torch::Tensor gr) {
  const int H = (int)Y.size(1);
  TORCH_CHECK(Y.scalar_type() == torch::kBFloat16 && Y.is_contiguous() && H % 8 == 0 && slotmap.numel() >= (int64_t)spc1::W * R * 6 &&
                  gr.numel() == (int64_t)spc1::W * R * 64 && (sh.numel() == 0 || sh.size(1) == H), "rs_cpushv: shapes");
  FprCommC1RsCpushvArgs ca{};
  ca.c = pyc(Y, rank);
  ca.Y = Y.data_ptr(), ca.slotmap = slotmap.data_ptr<int>(), ca.sh = sh.numel() ? sh.data_ptr() : nullptr;
  ca.gr = gr.data_ptr(), ca.R = (int32_t)R, ca.H = H, ca.ncta = (int32_t)ncta;
  peers_py(ca, bufs, flags);
  ca.epoch = (uint32_t*)epoch.data_ptr();
  FPR_CABI_RECORD("comm_c1.rs_cpushv", ca);   // rule 9 (fpr_cabi_rec.h)
  chk(rs_cpushv_core(&ca), "comm_c1 rs_cpushv");
}

static int rs_pushv_core(const FprCommC1RsPushvArgs* p) {   // C ABI "comm_c1.rs_pushv"
  try {
    return launch_c(p->c.stream, spc1::k_rs_pushv, (int)p->ncta * spc1::W, spc1::THR, (const uint4*)p->part, (int)p->R,
                    (int)p->H, (int)p->c.rank, peers_c(p->bufs, p->flags), (unsigned*)p->epoch, (const unsigned char*)p->gr);
  } catch (...) {
    return FPR_E_CUDA;
  }
}
void rs_pushv(torch::Tensor part, int64_t rank, std::vector<torch::Tensor> bufs, std::vector<torch::Tensor> flags,
              torch::Tensor epoch, int64_t ncta, torch::Tensor gr) {
  const int R = (int)(part.size(0) / spc1::W), H = (int)part.size(1);
  TORCH_CHECK(gr.numel() == (int64_t)spc1::W * R * 64, "rs_pushv: gr shape");
  FprCommC1RsPushvArgs ca{};
  ca.c = pyc(part, rank);
  ca.part = part.data_ptr(), ca.gr = gr.data_ptr(), ca.R = R, ca.H = H, ca.ncta = (int32_t)ncta;
  peers_py(ca, bufs, flags);
  ca.epoch = (uint32_t*)epoch.data_ptr();
  FPR_CABI_RECORD("comm_c1.rs_pushv", ca);   // rule 9 (fpr_cabi_rec.h)
  chk(rs_pushv_core(&ca), "comm_c1 rs_pushv");
}

static int rs_sumv_core(const FprCommC1RsSumvArgs* p) {   // C ABI "comm_c1.rs_sumv"
  try {
    return launch_c(p->c.stream, spc1::k_rs_sumv, (int)p->ncta, spc1::THR, (const __nv_bfloat162*)p->recv, (int)p->R, (int)p->H,
                    (const unsigned*)p->flag, (const unsigned*)p->epoch, (unsigned)p->signals, (__nv_bfloat162*)p->out,
                    (const unsigned char*)p->gr, (int)p->c.rank);
  } catch (...) {
    return FPR_E_CUDA;
  }
}
void rs_sumv(torch::Tensor recv, torch::Tensor flag, torch::Tensor epoch, int64_t signals, torch::Tensor out, int64_t ncta,
             torch::Tensor gr, int64_t rank) {
  const int R = (int)recv.size(1), H = (int)recv.size(2);
  TORCH_CHECK(gr.numel() == (int64_t)spc1::W * R * 64, "rs_sumv: gr shape");
  FprCommC1RsSumvArgs ca{};
  ca.c = pyc(recv, rank);
  ca.recv = recv.data_ptr(), ca.flag = (const uint32_t*)flag.data_ptr(), ca.epoch = (const uint32_t*)epoch.data_ptr();
  ca.out = out.data_ptr(), ca.gr = gr.data_ptr(), ca.R = R, ca.H = H, ca.signals = (uint32_t)signals, ca.ncta = (int32_t)ncta;
  FPR_CABI_RECORD("comm_c1.rs_sumv", ca);   // rule 9 (fpr_cabi_rec.h)
  chk(rs_sumv_core(&ca), "comm_c1 rs_sumv");
}

static int tp_scatter_core(const FprCommC1TpScatterArgs* p) {   // C ABI "comm_c1.tp_scatter"
  try {
    if (p->R > spc1::THR) return FPR_E_ARG;   // (the row masks are copied by threads 0 .. R - 1 of one CTA)
    return launch_c(p->c.stream, spc1::k_tp_scatter, (int)p->ncta * spc1::W, spc1::THR, (const uint4*)p->o,
                    (const unsigned char*)p->valid, (int)p->R, (int)p->c.rank, peers_c(p->bufs, p->flags),
                    peers_c(p->vbufs, p->flags), (unsigned*)p->epoch, (const unsigned*)p->myflag, (unsigned)p->signals,
                    (unsigned*)p->ticket, (const long long*)p->pos, (const float*)p->cos_sin);
  } catch (...) {
    return FPR_E_CUDA;
  }
}
void tp_scatter(torch::Tensor o, torch::Tensor valid, int64_t rank, std::vector<torch::Tensor> bufs, std::vector<torch::Tensor> flags,
                std::vector<torch::Tensor> vbufs, torch::Tensor epoch, int64_t ncta, torch::Tensor myflag, int64_t signals,
                torch::Tensor ticket, torch::Tensor pos, torch::Tensor cos_sin) {
  const int R = (int)o.size(0);
  TORCH_CHECK(o.size(1) == 32768 && o.is_contiguous() && valid.numel() >= R && R <= 512, "tp_scatter: shapes");
  FprCommC1TpScatterArgs ca{};
  ca.c = pyc(o, rank);
  ca.o = o.data_ptr(), ca.valid = (const uint8_t*)valid.data_ptr();
  ca.pos = pos.numel() ? (const int64_t*)pos.data_ptr<int64_t>() : nullptr;
  ca.cos_sin = pos.numel() ? cos_sin.data_ptr<float>() : nullptr;
  ca.R = R, ca.ncta = (int32_t)ncta;
  peers_py(ca, bufs, flags);
  for (int s = 0; s < spc1::W; ++s) ca.vbufs[s] = vbufs[s].data_ptr();
  ca.epoch = (uint32_t*)epoch.data_ptr(), ca.myflag = (const uint32_t*)myflag.data_ptr();
  ca.ticket = (uint32_t*)ticket.data_ptr(), ca.signals = (uint32_t)signals;
  FPR_CABI_RECORD("comm_c1.tp_scatter", ca);   // rule 9 (fpr_cabi_rec.h)
  chk(tp_scatter_core(&ca), "comm_c1 tp_scatter");
}

static int tp_rs_core(const FprCommC1TpRsArgs* p) {   // C ABI "comm_c1.tp_rs"
  try {
    return launch_c(p->c.stream, spc1::k_tp_rs, (int)p->ncta * spc1::W, spc1::THR, (const float4*)p->part,
                    (const unsigned char*)p->vm, (int)p->R, (int)p->c.rank, peers_c(p->bufs, p->flags), (unsigned*)p->epoch);
  } catch (...) {
    return FPR_E_CUDA;
  }
}
void tp_rs(torch::Tensor part, torch::Tensor vm, int64_t rank, std::vector<torch::Tensor> bufs, std::vector<torch::Tensor> flags,
           torch::Tensor epoch, int64_t ncta) {
  const int R = (int)(part.size(0) / spc1::W);
  TORCH_CHECK(part.size(1) == 5120 && part.dtype() == torch::kFloat32 && part.is_contiguous() && vm.numel() >= spc1::W * R, "tp_rs");
  FprCommC1TpRsArgs ca{};
  ca.c = pyc(part, rank);
  ca.part = (const float*)part.data_ptr(), ca.vm = (const uint8_t*)vm.data_ptr(), ca.R = R, ca.ncta = (int32_t)ncta;
  peers_py(ca, bufs, flags);
  ca.epoch = (uint32_t*)epoch.data_ptr();
  FPR_CABI_RECORD("comm_c1.tp_rs", ca);   // rule 9 (fpr_cabi_rec.h)
  chk(tp_rs_core(&ca), "comm_c1 tp_rs");
}

static int tp_rs_sum_core(const FprCommC1TpRsSumArgs* p) {   // C ABI "comm_c1.tp_rs_sum"
  try {
    return launch_c(p->c.stream, spc1::k_tp_rs_sum, (int)p->ncta, spc1::THR, (const float2*)p->recv, (int)p->R,
                    (const unsigned*)p->flag, (const unsigned*)p->epoch, (unsigned)p->signals, (__nv_bfloat162*)p->y,
                    (const unsigned char*)p->valid);
  } catch (...) {
    return FPR_E_CUDA;
  }
}
void tp_rs_sum(torch::Tensor recv, torch::Tensor flag, torch::Tensor epoch, int64_t signals, torch::Tensor y, int64_t ncta,
               torch::Tensor valid) {
  const int R = (int)recv.size(1);
  TORCH_CHECK(recv.size(2) == 5120 && recv.dtype() == torch::kFloat32 && y.dtype() == torch::kBFloat16 && y.is_contiguous(), "tp_rs_sum");
  FprCommC1TpRsSumArgs ca{};
  ca.c = pyc(recv, 0);
  ca.recv = (const float*)recv.data_ptr(), ca.flag = (const uint32_t*)flag.data_ptr();
  ca.epoch = (const uint32_t*)epoch.data_ptr(), ca.y = y.data_ptr(), ca.valid = (const uint8_t*)valid.data_ptr();
  ca.R = R, ca.signals = (uint32_t)signals, ca.ncta = (int32_t)ncta;
  FPR_CABI_RECORD("comm_c1.tp_rs_sum", ca);   // rule 9 (fpr_cabi_rec.h)
  chk(tp_rs_sum_core(&ca), "comm_c1 tp_rs_sum");
}

static int tp_rsl_core(const FprCommC1TpRslArgs* p) {   // C ABI "comm_c1.tp_rsl"
  try {
    spc1::LPeers lp;
    for (int s = 0; s < spc1::W; ++s) lp.buf[s] = p->lbufs[s];
    return launch_c(p->c.stream, spc1::k_tp_rsl, (int)p->grid, spc1::THR, (const float4*)p->part, (const unsigned char*)p->vm,
                    (int)p->R, (int)p->c.rank, lp, (unsigned*)p->tick, (__nv_bfloat162*)p->y, (const unsigned char*)p->valid);
  } catch (...) {
    return FPR_E_CUDA;
  }
}
void tp_rsl(torch::Tensor part, torch::Tensor vm, int64_t rank, std::vector<torch::Tensor> lbufs, torch::Tensor tick,
            torch::Tensor y, torch::Tensor valid, int64_t grid) {
  const int R = (int)(part.size(0) / spc1::W);
  TORCH_CHECK(part.size(1) == 5120 && part.dtype() == torch::kFloat32 && part.is_contiguous() && vm.numel() >= spc1::W * R &&
              y.dtype() == torch::kBFloat16 && y.is_contiguous() && lbufs.size() == (size_t)spc1::W &&
              tick.numel() >= grid && lbufs[0].numel() >= (int64_t)2 * spc1::W * R * 5120, "tp_rsl: shapes");
  FprCommC1TpRslArgs ca{};
  ca.c = pyc(part, rank);
  ca.part = (const float*)part.data_ptr(), ca.vm = (const uint8_t*)vm.data_ptr(), ca.y = y.data_ptr();
  ca.valid = (const uint8_t*)valid.data_ptr(), ca.tick = (uint32_t*)tick.data_ptr(), ca.R = R, ca.grid = (int32_t)grid;
  for (int s = 0; s < spc1::W; ++s) ca.lbufs[s] = lbufs[s].data_ptr<float>();
  FPR_CABI_RECORD("comm_c1.tp_rsl", ca);   // rule 9 (fpr_cabi_rec.h)
  chk(tp_rsl_core(&ca), "comm_c1 tp_rsl");
}

static int rs_cpushl_core(const FprCommC1RsCpushlArgs* p) {   // C ABI "comm_c1.rs_cpushl"
  try {
    if (p->H % 8 != 0) return FPR_E_ARG;
    spc1::LPeersB lp;
    for (int s = 0; s < spc1::W; ++s) lp.buf[s] = (__nv_bfloat16*)p->lbufs[s];
    return launch_c(p->c.stream, spc1::k_rs_cpushl, (int)p->grid, spc1::THR, (const __nv_bfloat16*)p->Y, (const int*)p->slotmap,
                    (const __nv_bfloat16*)p->sh, (int)p->R, (int)p->H, (int)p->c.rank, lp, (unsigned*)p->tick,
                    (__nv_bfloat162*)p->out, (const unsigned char*)p->gr);
  } catch (...) {
    return FPR_E_CUDA;
  }
}
void rs_cpushl(torch::Tensor Y, torch::Tensor slotmap, torch::Tensor sh, int64_t R, int64_t rank, std::vector<torch::Tensor> lbufs,
               torch::Tensor tick, torch::Tensor out, torch::Tensor gr, int64_t grid) {
  const int H = (int)Y.size(1);
  TORCH_CHECK(Y.scalar_type() == torch::kBFloat16 && Y.is_contiguous() && H % 8 == 0 && slotmap.numel() >= (int64_t)spc1::W * R * 6 &&
                  gr.numel() == (int64_t)spc1::W * R * 64 && (sh.numel() == 0 || sh.size(1) == H) &&
                  lbufs.size() == (size_t)spc1::W && lbufs[0].numel() >= (int64_t)2 * spc1::W * R * H && tick.numel() >= grid &&
                  out.scalar_type() == torch::kBFloat16 && out.is_contiguous(), "rs_cpushl: shapes");
  FprCommC1RsCpushlArgs ca{};
  ca.c = pyc(Y, rank);
  ca.Y = Y.data_ptr(), ca.slotmap = slotmap.data_ptr<int>(), ca.sh = sh.numel() ? sh.data_ptr() : nullptr;
  ca.out = out.data_ptr(), ca.gr = gr.data_ptr(), ca.tick = (uint32_t*)tick.data_ptr();
  ca.R = (int32_t)R, ca.H = H, ca.grid = (int32_t)grid;
  for (int s = 0; s < spc1::W; ++s) ca.lbufs[s] = lbufs[s].data_ptr();
  FPR_CABI_RECORD("comm_c1.rs_cpushl", ca);   // rule 9 (fpr_cabi_rec.h)
  chk(rs_cpushl_core(&ca), "comm_c1 rs_cpushl");
}

// (not in the C table: the engine does not call it)
void arl(torch::Tensor part, int64_t rank, std::vector<torch::Tensor> lbufs, torch::Tensor tick, torch::Tensor y, int64_t grid) {
  const int NR = (int)part.size(0);
  TORCH_CHECK(part.size(1) == 5120 && part.dtype() == torch::kFloat32 && part.is_contiguous() && y.dtype() == torch::kBFloat16 &&
              y.is_contiguous() && y.size(0) >= NR && lbufs.size() == (size_t)spc1::W &&
              lbufs[0].numel() >= (int64_t)2 * spc1::W * NR * 5120 && tick.numel() >= grid, "arl: shapes");
  spc1::LPeers p;
  for (int s = 0; s < spc1::W; ++s) p.buf[s] = lbufs[s].data_ptr<float>();
  const FprCall c = pyc(part, rank);
  chk(launch_c(c.stream, spc1::k_arl, (int)grid, spc1::THR, (const float4*)part.data_ptr(), NR, (int)rank, p,
               (unsigned*)tick.data_ptr(), (__nv_bfloat162*)y.data_ptr()),
      "comm_c1 arl");
}

// ---- C ABI (fpr_cabi.h): init + table. The only process state is the ENG_C1_PDL latch (no dynamic SMEM, no buffers).
static int init_core(const FprCommC1InitArgs*) {   // "comm_c1.init": latch ENG_C1_PDL before the driver threads launch
  (void)pdl_on();
  return FPR_OK;
}
extern "C" {
static int c_init(const void* a) { return init_core((const FprCommC1InitArgs*)a); }
static int c_ag_push2v(const void* a) { return ag_push2v_core((const FprCommC1AgPush2vArgs*)a); }
static int c_ag_push2v_wait(const void* a) { return ag_push2v_wait_core((const FprCommC1AgPush2vWaitArgs*)a); }
static int c_ag_rt(const void* a) { return ag_rt_core((const FprCommC1AgRtArgs*)a); }
static int c_rs_cpushv(const void* a) { return rs_cpushv_core((const FprCommC1RsCpushvArgs*)a); }
static int c_rs_pushv(const void* a) { return rs_pushv_core((const FprCommC1RsPushvArgs*)a); }
static int c_rs_sumv(const void* a) { return rs_sumv_core((const FprCommC1RsSumvArgs*)a); }
static int c_tp_scatter(const void* a) { return tp_scatter_core((const FprCommC1TpScatterArgs*)a); }
static int c_tp_rs(const void* a) { return tp_rs_core((const FprCommC1TpRsArgs*)a); }
static int c_tp_rs_sum(const void* a) { return tp_rs_sum_core((const FprCommC1TpRsSumArgs*)a); }
static int c_tp_rsl(const void* a) { return tp_rsl_core((const FprCommC1TpRslArgs*)a); }
static int c_rs_cpushl(const void* a) { return rs_cpushl_core((const FprCommC1RsCpushlArgs*)a); }
static const FprFnEntry k_comm_c1_fns[] = {
    {"comm_c1.init", c_init, nullptr, (int64_t)sizeof(FprCommC1InitArgs)},
    {"comm_c1.ag_push2v", c_ag_push2v, nullptr, (int64_t)sizeof(FprCommC1AgPush2vArgs)},
    {"comm_c1.ag_push2v_wait", c_ag_push2v_wait, nullptr, (int64_t)sizeof(FprCommC1AgPush2vWaitArgs)},
    {"comm_c1.ag_rt", c_ag_rt, nullptr, (int64_t)sizeof(FprCommC1AgRtArgs)},
    {"comm_c1.rs_cpushv", c_rs_cpushv, nullptr, (int64_t)sizeof(FprCommC1RsCpushvArgs)},
    {"comm_c1.rs_pushv", c_rs_pushv, nullptr, (int64_t)sizeof(FprCommC1RsPushvArgs)},
    {"comm_c1.rs_sumv", c_rs_sumv, nullptr, (int64_t)sizeof(FprCommC1RsSumvArgs)},
    {"comm_c1.tp_scatter", c_tp_scatter, nullptr, (int64_t)sizeof(FprCommC1TpScatterArgs)},
    {"comm_c1.tp_rs", c_tp_rs, nullptr, (int64_t)sizeof(FprCommC1TpRsArgs)},
    {"comm_c1.tp_rs_sum", c_tp_rs_sum, nullptr, (int64_t)sizeof(FprCommC1TpRsSumArgs)},
    {"comm_c1.tp_rsl", c_tp_rsl, nullptr, (int64_t)sizeof(FprCommC1TpRslArgs)},
    {"comm_c1.rs_cpushl", c_rs_cpushl, nullptr, (int64_t)sizeof(FprCommC1RsCpushlArgs)},
};
static const FprExtTable k_comm_c1_table = {FPR_CABI_VERSION, "comm_c1", "", 12, k_comm_c1_fns};
__attribute__((visibility("default"))) const FprExtTable* fpr_comm_c1_table(void) { return &k_comm_c1_table; }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("arl", &arl);
  m.def("tp_rsl", &tp_rsl);
  m.def("rs_cpushl", &rs_cpushl);
  m.def("tp_scatter", &tp_scatter);
  m.def("tp_rs", &tp_rs);
  m.def("tp_rs_sum", &tp_rs_sum);
  m.def("ag_push2v", &ag_push2v);
  m.def("ag_push2v_wait", &ag_push2v_wait);
  m.def("ag_rt", &ag_rt);
  m.def("rs_cpushv", &rs_cpushv);
  m.def("rs_pushv", &rs_pushv);
  m.def("rs_sumv", &rs_sumv);
}
