// K1: fused mHC sublayer boundary on the rows this rank owns (sequence-parallel layout).
//
// One boundary (ds41h/ref/layer.py decode_layer_ref, VLLM_SM90 numerics), per row r of R owned rows:
//   Xn[j]   = bf16( post[j] * y + sum_i comb[i][j] * Rs[i] )                       hc_post        (j, i < 4)
//   mixes   = (Xn.flat @ fn^T) * rsqrt(mean(Xn.flat^2) + norm_eps)                  hc_mixes       (24 outputs)
//   pre_n   = sigmoid(mixes[0:4] * s0 + b[0:4]) + hc_eps
//   post_n  = 2 sigmoid(mixes[4:8] * s1 + b[4:8])
//   comb_n  = sinkhorn(mixes[8:24] * s2 + b[8:24])   (row softmax + eps, col norm, 19 x (row norm, col norm))
//   in      = bf16( sum_c lag[c] * Xn[c] )                                          hc_collapse with the LAGGED pre
//   out     = bf16( in * rsqrt(mean(in^2) + norm_eps) * w )                         RMSNorm
// Outputs: Xn (the next residual stream), pre_n / post_n / comb_n, out (the sublayer input a or f).
// Stock computes the 24-wide product as a DeepGEMM tf32 GEMM; here it is fp32 FMA (more precise), in a
// fixed order: per 32-column chunk a warp reduction, then a fixed-order sum over the 160 chunks (deterministic).
//
// K1a k_mhc_post_mix: grid = D / 32 chunks, 256 threads (8 warps); lane = column of the chunk, warp w = rows
//      w, w + 8, ...; the 96 fn weights of the lane's column (24 outputs x 4 copies) stay in registers for all rows.
//      Writes Xn and partial[chunk][r][25] = (sum of squares, 24 dots).
// K1b k_mhc_finish: grid = R rows, 256 threads: fixed-order reduction of the partials, split + Sinkhorn (one thread),
//      collapse + RMSNorm (20 elements per thread).
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace mhc {
constexpr int D = 5120, HC = 4, NMIX = 24, NP = 25, CW = 32, NCH = D / CW;  // 160 chunks of 32 columns
#ifndef MHC_THR1
#define MHC_THR1 256
#endif
#ifndef MHC_NODOT
#define MHC_NODOT 0
#endif
#ifndef MHC_V2
#define MHC_V2 1  // 1 (default): K1a v2 (vectorized phase 1, 2 chunks x 64 rows per CTA; same outputs bit for bit at equal MHC_CHAINS)
#endif
#ifndef MHC_PF
#define MHC_PF 0  // 1: K1a v2 prefetches its B fragments (fn hi / lo) into L2 at kernel start
#endif
#ifndef MHC_CHAINS
#define MHC_CHAINS 1  // accumulator chains in K1a's MMA phase (see k_mhc_post_mix)
#endif
#ifndef MHC_ABL
#define MHC_ABL 0  // ablation bits: 1 skip the MMA phase, 2 skip the xn stores
#endif
constexpr int THR = 256, THR1 = MHC_THR1;

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// butterfly reduce-scatter over the warp (fixed order): v[0..63] per lane in; after the steps N = 16, 8, 4, 2, 1 lane l
// holds in v[0..1] the 32-lane sums of the original indices given in k_mhc_post_mix (31 shuffles per 32 values).
template <int N>
__device__ __forceinline__ void bfly(float (&v)[64], int lane) {
  // step with offset N: keep the half selected by (lane & N), add the partner's copy of it
  constexpr int H = N * 2;  // values per lane before this step = 2 * H ... (N = 16: 64 -> 32, 8: 32 -> 16, ...)
#pragma unroll
  for (int q = 0; q < H; ++q) {
    const bool hi = lane & N;
    const float mine = hi ? v[q + H] : v[q], other = hi ? v[q] : v[q + H];
    v[q] = mine + __shfl_xor_sync(~0u, other, N);
  }
}

// K1a (tensor cores): grid = D / 32 column chunks (160), 256 threads (8 warps), all R <= 128 rows per CTA.
//   phase 1: hc_post for rows r = w + 8 i (lane = column), xn stored (bf16) and the bf16-exact values kept as tf32 in
//            shared sx[r][j * 32 + lane] (j = copy; chunk-local k = j * 32 + column); per-row sum of squares (warp sum)
//   phase 2: warp w = rows [16 w, 16 w + 16): 16 k-steps x 3 n-tiles x {hi, lo} mma.m16n8k8 tf32 (3xTF32-style split of
//            fn done once by prep_fn: fn = hi + lo, both tf32; products exact, fp32 accumulation -> ~22-bit dots,
//            vs stock's single tf32 pass). B fragments come pre-ordered from global (L1/L2 hits for the 8 warps).
constexpr int KS = 16, NT = 3, SXP = 132;  // k-steps per chunk, n-tiles (24 = 3 x 8), sx row pitch (conflict-free)
__device__ __forceinline__ void mma_tf32(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(THR) k_mhc_post_mix(
    const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ rs, const float* __restrict__ post,
    const float* __restrict__ comb, const float2* __restrict__ fnb, int R, __nv_bfloat16* __restrict__ xn,
    float* __restrict__ part) {
  __shared__ float sx[128 * SXP];
  __shared__ float spc[128][20];  // per row: post[4], comb[16]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, ch = blockIdx.x, k = ch * CW + lane;
  for (int e = threadIdx.x; e < R * 20; e += THR) {
    const int r = e / 20, q = e % 20;
    spc[r][q] = q < HC ? __ldg(post + r * HC + q) : __ldg(comb + r * 16 + q - HC);
  }
  // all 16 rows' stream loads of this warp in flight before any use (memory-level parallelism)
  __nv_bfloat16 ly[16], lr[16][HC];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int r = warp + 8 * i;
    const int rc = r < R ? r : 0;
    ly[i] = y[(size_t)rc * D + k];
#pragma unroll
    for (int c = 0; c < HC; ++c) lr[i][c] = rs[((size_t)rc * HC + c) * D + k];
  }
  __syncthreads();  // spc ready
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int r = warp + 8 * i;
    float x[HC] = {0.f, 0.f, 0.f, 0.f}, ss = 0.f;
    if (r < R) {
      const float yv = __bfloat162float(ly[i]);
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float a = spc[r][j] * yv;
#pragma unroll
        for (int c = 0; c < HC; ++c) a = fmaf(spc[r][HC + c * HC + j], __bfloat162float(lr[i][c]), a);
        x[j] = bfr(a);
        if (!(MHC_ABL & 2)) xn[((size_t)r * HC + j) * D + k] = __float2bfloat16_rn(x[j]);
        ss = fmaf(x[j], x[j], ss);
      }
    }
#pragma unroll
    for (int j = 0; j < HC; ++j) sx[r * SXP + j * CW + lane] = x[j];
#pragma unroll
    for (int off = 16; off; off >>= 1) ss += __shfl_xor_sync(~0u, ss, off);
    if (lane == 0 && r < R) part[((size_t)ch * R + r) * NP] = ss;
  }
  __syncthreads();
  if (MHC_ABL & 1) return;
  const int g = lane >> 2, t = lane & 3, rb = 16 * warp;
  float d[NT][4];
  const float2* fb = fnb + (size_t)ch * KS * NT * 2 * 32;
#if MHC_CHAINS > 1
  // MHC_CHAINS independent accumulator chains over the k-steps (s % MHC_CHAINS), summed in a fixed order at the end:
  // the 32-deep dependent mma chain per n-tile was the MMA phase's latency (5.4 of 15.4 us)
  float dc[MHC_CHAINS][NT][4];
#pragma unroll
  for (int q = 0; q < MHC_CHAINS; ++q)
#pragma unroll
    for (int n = 0; n < NT; ++n) dc[q][n][0] = dc[q][n][1] = dc[q][n][2] = dc[q][n][3] = 0.f;
#pragma unroll
  for (int s = 0; s < KS; ++s) {
    unsigned a[4];
    const int kk = s * 8;
    a[0] = __float_as_uint(sx[(rb + g) * SXP + kk + t]);
    a[1] = __float_as_uint(sx[(rb + g + 8) * SXP + kk + t]);
    a[2] = __float_as_uint(sx[(rb + g) * SXP + kk + t + 4]);
    a[3] = __float_as_uint(sx[(rb + g + 8) * SXP + kk + t + 4]);
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const float2 bh = __ldg(fb + ((s * NT + n) * 2 + 0) * 32 + lane), bl = __ldg(fb + ((s * NT + n) * 2 + 1) * 32 + lane);
      mma_tf32(dc[s % MHC_CHAINS][n], a, __float_as_uint(bl.x), __float_as_uint(bl.y));  // small part first
      mma_tf32(dc[s % MHC_CHAINS][n], a, __float_as_uint(bh.x), __float_as_uint(bh.y));
    }
  }
#pragma unroll
  for (int n = 0; n < NT; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float acc = dc[0][n][i];
#pragma unroll
      for (int q = 1; q < MHC_CHAINS; ++q) acc += dc[q][n][i];
      d[n][i] = acc;
    }
#else
#pragma unroll
  for (int n = 0; n < NT; ++n) d[n][0] = d[n][1] = d[n][2] = d[n][3] = 0.f;
#pragma unroll 4
  for (int s = 0; s < KS; ++s) {
    unsigned a[4];
    const int kk = s * 8;
    a[0] = __float_as_uint(sx[(rb + g) * SXP + kk + t]);
    a[1] = __float_as_uint(sx[(rb + g + 8) * SXP + kk + t]);
    a[2] = __float_as_uint(sx[(rb + g) * SXP + kk + t + 4]);
    a[3] = __float_as_uint(sx[(rb + g + 8) * SXP + kk + t + 4]);
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const float2 bh = __ldg(fb + ((s * NT + n) * 2 + 0) * 32 + lane), bl = __ldg(fb + ((s * NT + n) * 2 + 1) * 32 + lane);
      mma_tf32(d[n], a, __float_as_uint(bl.x), __float_as_uint(bl.y));  // small part first
      mma_tf32(d[n], a, __float_as_uint(bh.x), __float_as_uint(bh.y));
    }
  }
#endif
#pragma unroll
  for (int n = 0; n < NT; ++n)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = rb + g + 8 * h;
      if (r < R) {
        float* pp = part + ((size_t)ch * R + r) * NP + 1 + 8 * n + 2 * t;
        pp[0] = d[n][2 * h];
        pp[1] = d[n][2 * h + 1];
      }
    }
}

// K1a v2: the same outputs, bit for bit (xn, part), with a vectorized phase 1. CTA (cp, h) = column chunks
// 2 cp, 2 cp + 1 (64 columns: 128-B row pieces instead of 64-B) x rows [64 h, 64 h + 64); thread = (row, 8-column
// group), 16-B loads / stores; the per-chunk sum of squares follows the original butterfly (columns xor 16, 8 across
// lanes, then xor 4, 2, 1 inside the thread: the same tree). Phase 2: warp w = chunk w / 4, rows 16 (w % 4) .., as K1a.
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
struct Fold {  // FOLD: y = bf16(fixed-order fp32 sum of the W = 4 RS receive slots), after the RS flag reached its target
  const __nv_bfloat16* recv;  // [4, R, D] (slot = source rank), written by the peers' RS push
  const unsigned* flag;
  const unsigned* epoch;  // this rank's RS push count (bumped by its own push earlier in the stream)
  unsigned signals;       // signals per round
};
template <bool FOLD>
__global__ void __launch_bounds__(THR) k_mhc_post_mix2(
    const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ rs, const float* __restrict__ post,
    const float* __restrict__ comb, const float2* __restrict__ fnb, int R, __nv_bfloat16* __restrict__ xn,
    float* __restrict__ part, Fold fo = Fold{}) {
  __shared__ float sx[2][64 * SXP];
  __shared__ float spc[64][20];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, cp = blockIdx.x, r0 = 64 * blockIdx.y;
#if MHC_PF
  // the phase-2 B fragments of this CTA's 2 chunks (2 x 24.6 KB, from HBM: 3.9 MB per boundary and layer) into L2 now,
  // under phase 1: one bulk L2 prefetch per 4 KB piece
  if (threadIdx.x < 12) {
    const char* fb0 = (const char*)(fnb + (size_t)(2 * cp) * KS * NT * 2 * 32);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(fb0 + threadIdx.x * 4096), "r"(4096) : "memory");
  }
#endif
  for (int e = threadIdx.x; e < 64 * 20; e += THR) {
    const int rl = e / 20, q = e % 20, r = r0 + rl;
    spc[rl][q] = r < R ? (q < HC ? __ldg(post + r * HC + q) : __ldg(comb + r * 16 + q - HC)) : 0.f;
  }
  const int cg = lane & 7, chl = cg >> 2, col0 = 8 * (cg & 3);  // chunk (0/1) and its first column of this thread
  const int k0 = 64 * cp + 8 * cg;                                // global column of element i = 0
  // two passes of 32 rows: row rl = 32 p + 4 warp + lane / 8
  uint4 ly[2], lr[2][HC];
#pragma unroll
  for (int pp = 0; pp < 2; ++pp) {
    const int rl = 32 * pp + 4 * warp + (lane >> 3), r = r0 + rl, rc = r < R ? r : 0;
    if (!FOLD) ly[pp] = __ldg((const uint4*)(y + (size_t)rc * D + k0));
#pragma unroll
    for (int c = 0; c < HC; ++c) lr[pp][c] = __ldg((const uint4*)(rs + ((size_t)rc * HC + c) * D + k0));
  }
  if (FOLD) {  // = comm.cu k_rs_sum fused: wait for the RS round, then the 4 slots summed in rank order (fp32, one rounding)
    if (threadIdx.x == 0) {
      const unsigned target = *(volatile const unsigned*)fo.epoch * fo.signals;
      while ((int)(ld_acquire_sys(fo.flag) - target) < 0) __nanosleep(32);
    }
    __syncthreads();
    uint4 q[2][4];
#pragma unroll
    for (int pp = 0; pp < 2; ++pp) {
      const int rl = 32 * pp + 4 * warp + (lane >> 3), r = r0 + rl, rc = r < R ? r : 0;
#pragma unroll
      for (int sl = 0; sl < 4; ++sl) q[pp][sl] = __ldcg((const uint4*)(fo.recv + ((size_t)sl * R + rc) * D + k0));
    }
#pragma unroll
    for (int pp = 0; pp < 2; ++pp) {
      __nv_bfloat162* o = (__nv_bfloat162*)&ly[pp];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 a = __bfloat1622float2(((const __nv_bfloat162*)&q[pp][0])[i]);
#pragma unroll
        for (int sl = 1; sl < 4; ++sl) {
          const float2 b = __bfloat1622float2(((const __nv_bfloat162*)&q[pp][sl])[i]);
          a.x += b.x;
          a.y += b.y;
        }
        o[i] = __floats2bfloat162_rn(a.x, a.y);
      }
    }
  }
  __syncthreads();  // spc ready
#pragma unroll
  for (int pp = 0; pp < 2; ++pp) {
    const int rl = 32 * pp + 4 * warp + (lane >> 3), r = r0 + rl;
    float x[HC][8], ss[8];
    const __nv_bfloat16* yv = (const __nv_bfloat16*)&ly[pp];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      ss[i] = 0.f;
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float a = spc[rl][j] * __bfloat162float(yv[i]);
#pragma unroll
        for (int c = 0; c < HC; ++c) a = fmaf(spc[rl][HC + c * HC + j], __bfloat162float(((const __nv_bfloat16*)&lr[pp][c])[i]), a);
        x[j][i] = r < R ? bfr(a) : 0.f;
        ss[i] = fmaf(x[j][i], x[j][i], ss[i]);
      }
    }
    if (r < R && !(MHC_ABL & 2)) {
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        __align__(16) __nv_bfloat16 o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __float2bfloat16_rn(x[j][i]);
        *(uint4*)(xn + ((size_t)r * HC + j) * D + k0) = *(const uint4*)o;
      }
    }
    float* sxr = sx[chl] + rl * SXP;
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      *(float4*)(sxr + j * CW + col0) = make_float4(x[j][0], x[j][1], x[j][2], x[j][3]);
      *(float4*)(sxr + j * CW + col0 + 4) = make_float4(x[j][4], x[j][5], x[j][6], x[j][7]);
    }
    // butterfly over the chunk's 32 columns k = col0 + i: xor 16 (lane ^ 2), xor 8 (lane ^ 1), then xor 4, 2, 1 in-thread
#pragma unroll
    for (int i = 0; i < 8; ++i) ss[i] += __shfl_xor_sync(~0u, ss[i], 2);
#pragma unroll
    for (int i = 0; i < 8; ++i) ss[i] += __shfl_xor_sync(~0u, ss[i], 1);
    float t4[4], t2[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) t4[i] = ss[i] + ss[i + 4];
#pragma unroll
    for (int i = 0; i < 2; ++i) t2[i] = t4[i] + t4[i + 2];
    const float tot = t2[0] + t2[1];
    if ((cg & 3) == 0 && r < R) part[((size_t)(2 * cp + chl) * R + r) * NP] = tot;
  }
  __syncthreads();
  if (MHC_ABL & 1) return;
  const int g = lane >> 2, t = lane & 3, ch = 2 * cp + (warp >> 2), rb = 16 * (warp & 3);
  const float* sxc = sx[warp >> 2];
  float d[NT][4];
  const float2* fb = fnb + (size_t)ch * KS * NT * 2 * 32;
  float dc[MHC_CHAINS][NT][4];
#pragma unroll
  for (int q = 0; q < MHC_CHAINS; ++q)
#pragma unroll
    for (int n = 0; n < NT; ++n) dc[q][n][0] = dc[q][n][1] = dc[q][n][2] = dc[q][n][3] = 0.f;
#pragma unroll
  for (int s = 0; s < KS; ++s) {
    unsigned a[4];
    const int kk = s * 8;
    a[0] = __float_as_uint(sxc[(rb + g) * SXP + kk + t]);
    a[1] = __float_as_uint(sxc[(rb + g + 8) * SXP + kk + t]);
    a[2] = __float_as_uint(sxc[(rb + g) * SXP + kk + t + 4]);
    a[3] = __float_as_uint(sxc[(rb + g + 8) * SXP + kk + t + 4]);
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const float2 bh = __ldg(fb + ((s * NT + n) * 2 + 0) * 32 + lane), bl = __ldg(fb + ((s * NT + n) * 2 + 1) * 32 + lane);
      mma_tf32(dc[s % MHC_CHAINS][n], a, __float_as_uint(bl.x), __float_as_uint(bl.y));
      mma_tf32(dc[s % MHC_CHAINS][n], a, __float_as_uint(bh.x), __float_as_uint(bh.y));
    }
  }
#pragma unroll
  for (int n = 0; n < NT; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float acc = dc[0][n][i];
#pragma unroll
      for (int q = 1; q < MHC_CHAINS; ++q) acc += dc[q][n][i];
      d[n][i] = acc;
    }
#pragma unroll
  for (int n = 0; n < NT; ++n)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = r0 + rb + g + 8 * h;
      if (r < R) {
        float* pp_ = part + ((size_t)ch * R + r) * NP + 1 + 8 * n + 2 * t;
        pp_[0] = d[n][2 * h];
        pp_[1] = d[n][2 * h + 1];
      }
    }
}

__device__ __forceinline__ float sigm(float z) { return 1.f / (1.f + expf(-z)); }
#ifndef MHC_K1B_BULK
#define MHC_K1B_BULK 0  // 1: K1b loads its row's xn (4 x 10 KB) by TMA bulk copies into SMEM (same arithmetic, same order)
#endif
__device__ __forceinline__ uint32_t s_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(s_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_expect(uint64_t* b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(s_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, unsigned parity) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" ::"r"(s_u32(b)),
               "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(s_u32(dst)),
               "l"(src), "r"(bytes), "r"(s_u32(b)) : "memory");
}

struct Push {  // optional: the output row goes straight to slot `rank` of every rank's all-gather buffer (peer stores)
  __nv_bfloat16* buf[4];
  unsigned* flag[4];
  int rank, on;
};
__device__ __forceinline__ void red_release_sys(unsigned* p, unsigned v) {
  asm volatile("red.release.sys.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// MODE 0: the whole K1b. MODE 1 (split): only collapse + RMSNorm + out (+ ptok) -- the sublayer input, on the
// critical path. MODE 2: only the mixes (partial reduction, split, Sinkhorn -> pre / post / comb), which the NEXT boundary
// consumes, so they can run on a side stream. Each part is the MODE 0 code verbatim: the same outputs bit for bit.
template <int MODE>
__global__ void __launch_bounds__(THR) k_mhc_finish(
    const float* __restrict__ part, const __nv_bfloat16* __restrict__ xn, const float* __restrict__ lag,
    const float* __restrict__ scale, const float* __restrict__ base, const float* __restrict__ normw, int R,
    float norm_eps, float hc_eps, int iters, float* __restrict__ pre_o, float* __restrict__ post_o,
    float* __restrict__ comb_o, __nv_bfloat16* __restrict__ out, Push pu, int* __restrict__ ptok_o = nullptr) {
  constexpr bool DO_OUT = MODE != 2, DO_MIX = MODE != 1;
  const int r = blockIdx.x, t = threadIdx.x;
  __shared__ float tot[NP];
  __shared__ float wsum[THR / 32];
  constexpr int PER = D / THR;  // 20
#if MHC_K1B_BULK
  __shared__ __align__(128) __nv_bfloat16 sxn[HC * D];  // this row's 4 copies (40 KB)
  __shared__ uint64_t mbx;
  if (t == 0) {
    mb_init(&mbx, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (t == 0) {
    mb_expect(&mbx, HC * D * 2);
#pragma unroll
    for (int c = 0; c < HC; ++c) bulk_g2s(sxn + c * D, xn + ((size_t)r * HC + c) * D, D * 2, &mbx);
  }
#endif
  // collapse loads first (independent of the mixes): 20 elements x 4 copies per thread
  float lg[HC];
#pragma unroll
  for (int c = 0; c < HC; ++c) lg[c] = DO_OUT ? lag[r * HC + c] : 0.f;
  float v[PER];
  float ss = 0.f;
#if MHC_K1B_BULK
  mb_wait(&mbx, 0);
#endif
#pragma unroll
  for (int e = 0; e < (DO_OUT ? PER : 0); ++e) {
    const int k = e * THR + t;
    float a = 0.f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
#if MHC_K1B_BULK
      const __nv_bfloat16 xv = sxn[c * D + k];
#else
      const __nv_bfloat16 xv = xn[((size_t)r * HC + c) * D + k];
#endif
      a = fmaf(lg[c], __bfloat162float(xv), a);
    }
    v[e] = bfr(a);
    ss = fmaf(v[e], v[e], ss);
  }
  __shared__ float red[10][NP];
  if (DO_MIX && t < 10 * NP) {  // 10 groups x 16 chunks, loads in flight, then a fixed-order combine (deterministic)
    const int gq = t / NP, q = t % NP;
    float pv[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) pv[c] = part[((size_t)(gq * 16 + c) * R + r) * NP + q];
    float a = 0.f;
#pragma unroll
    for (int c = 0; c < 16; ++c) a += pv[c];
    red[gq][q] = a;
  }
  __syncthreads();
  if (DO_MIX && t < NP) {
    float a = 0.f;
    for (int gq = 0; gq < 10; ++gq) a += red[gq][t];
    tot[t] = a;
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) ss += __shfl_xor_sync(~0u, ss, off);
  if ((t & 31) == 0) wsum[t >> 5] = ss;
  __syncthreads();
  if (DO_MIX && t < 32) {  // split + Sinkhorn: lane l < 16 owns c[l / 4][l % 4] (rows / columns summed by shuffles, IEEE divisions)
    const float rsq = rsqrtf(tot[0] / (float)(HC * D) + norm_eps);
    const int l = t & 15, i = l >> 2, j = l & 3;
    if (t < HC) {
      pre_o[r * HC + t] = sigm(tot[1 + t] * rsq * scale[0] + base[t]) + hc_eps;
      post_o[r * HC + t] = 2.f * sigm(tot[1 + HC + t] * rsq * scale[1] + base[HC + t]);
    }
    const float lg = tot[1 + 2 * HC + l] * rsq * scale[2] + base[2 * HC + l];
    // row softmax over j (lanes 4i .. 4i+3): xor 1, 2 stay inside the group of 4
    float mx = fmaxf(lg, __shfl_xor_sync(~0u, lg, 1));
    mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, 2));
    const float ex = expf(lg - mx);
    // fixed summation order j = 0..3 (as the sequential loop), via explicit gathers
    auto rowsum = [&](float v) {
      const int b = t & ~3;
      return ((__shfl_sync(~0u, v, b) + __shfl_sync(~0u, v, b + 1)) + __shfl_sync(~0u, v, b + 2)) + __shfl_sync(~0u, v, b + 3);
    };
    auto colsum = [&](float v) {  // i = 0..3 in order: lanes j, 4 + j, 8 + j, 12 + j (upper half mirrors the lower)
      const int b = (t & 16) | (t & 3);
      return ((__shfl_sync(~0u, v, b) + __shfl_sync(~0u, v, b + 4)) + __shfl_sync(~0u, v, b + 8)) + __shfl_sync(~0u, v, b + 12);
    };
    float c = ex / rowsum(ex) + hc_eps;
    c = c / (colsum(c) + hc_eps);
    for (int it = 1; it < iters; ++it) {
      c = c / (rowsum(c) + hc_eps);
      c = c / (colsum(c) + hc_eps);
    }
    if (t < 16) comb_o[r * 16 + i * HC + j] = c;
  }
  if (!DO_OUT) return;
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < THR / 32; ++i) s += wsum[i];
  const float rr = rsqrtf(s / (float)D + norm_eps);
  if (!pu.on) {
    float mx = 0.f;
#pragma unroll
    for (int e = 0; e < PER; ++e) {
      const int k = e * THR + t;
      const __nv_bfloat16 ob = __float2bfloat16_rn((v[e] * rr) * normw[k]);
      out[(size_t)r * D + k] = ob;
      mx = fmaxf(mx, fabsf(__bfloat162float(ob)));
    }
    if (ptok_o) {  // the MoE's per-token power of two (as msr::k_prescale on this output row): max |out| -> 14 - exponent
#pragma unroll
      for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, off));
      __shared__ float wmx[THR / 32];
      if ((t & 31) == 0) wmx[t >> 5] = mx;
      __syncthreads();
      if (t == 0) {
        float m = wmx[0];
#pragma unroll
        for (int i = 1; i < THR / 32; ++i) m = fmaxf(m, wmx[i]);
        int p = 0;
        if (m > 0.f) {
          int ex;
          frexpf(m, &ex);
          p = 14 - ex;
        }
        ptok_o[r] = p;
      }
    }
    return;
  }
  // push: this row to slot pu.rank of all 4 gather buffers ([4 R rows][D]; the own copy doubles as the router's input)
  __nv_bfloat16 o[PER];
#pragma unroll
  for (int e = 0; e < PER; ++e) o[e] = __float2bfloat16_rn((v[e] * rr) * normw[e * THR + t]);
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    const int s2 = (pu.rank + 1 + d) & 3;  // peers first, own copy last
    __nv_bfloat16* dst = pu.buf[s2] + ((size_t)pu.rank * R + r) * D;
#pragma unroll
    for (int e = 0; e < PER; ++e) dst[e * THR + t] = o[e];
  }
  asm volatile("fence.acq_rel.sys;" ::: "memory");
  __syncthreads();
  if (t < 4) red_release_sys(pu.flag[t], 1u);  // every row signals every rank once: W * R signals per round
}
}  // namespace mhc

// y [R, D] bf16, rs [R, 4, D] bf16, post [R, 4] f32, comb [R, 4, 4] f32, lag [R, 4] f32, fn [24, 4D] f32, scale [3],
// base [24], normw [D] f32 -> xn [R, 4, D] bf16, pre/post_n [R, 4], comb_n [R, 4, 4], out [R, D] bf16; part scratch
// [20 * R * 25] f32.
static bool g_attr = false;
void boundary(torch::Tensor y, torch::Tensor rs, torch::Tensor post, torch::Tensor comb, torch::Tensor lag,
              torch::Tensor fn, torch::Tensor scale, torch::Tensor base, torch::Tensor normw, double norm_eps,
              double hc_eps, int64_t iters, torch::Tensor xn, torch::Tensor pre_o, torch::Tensor post_o,
              torch::Tensor comb_o, torch::Tensor out, torch::Tensor part) {
  const int R = (int)y.size(0);
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(R <= 128, "mhc: at most 128 rows per call");
  if (MHC_V2)
    mhc::k_mhc_post_mix2<false><<<dim3(mhc::NCH / 2, 2), mhc::THR, 0, st>>>(
        (const __nv_bfloat16*)y.data_ptr(), (const __nv_bfloat16*)rs.data_ptr(), post.data_ptr<float>(),
        comb.data_ptr<float>(), (const float2*)fn.data_ptr(), R, (__nv_bfloat16*)xn.data_ptr(), part.data_ptr<float>());
  else
    mhc::k_mhc_post_mix<<<mhc::NCH, mhc::THR, 0, st>>>(
        (const __nv_bfloat16*)y.data_ptr(), (const __nv_bfloat16*)rs.data_ptr(), post.data_ptr<float>(),
        comb.data_ptr<float>(), (const float2*)fn.data_ptr(), R, (__nv_bfloat16*)xn.data_ptr(), part.data_ptr<float>());
  mhc::k_mhc_finish<0><<<R, mhc::THR, 0, st>>>(
      part.data_ptr<float>(), (const __nv_bfloat16*)xn.data_ptr(), lag.data_ptr<float>(), scale.data_ptr<float>(),
      base.data_ptr<float>(), normw.data_ptr<float>(), R, (float)norm_eps, (float)hc_eps, (int)iters,
      pre_o.data_ptr<float>(), post_o.data_ptr<float>(), comb_o.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      mhc::Push{});
}

// boundary + the MoE prescale exponent of every output row (int32 [R]) from K1b (replaces msr::k_prescale for own rows)
void boundary2(torch::Tensor y, torch::Tensor rs, torch::Tensor post, torch::Tensor comb, torch::Tensor lag,
               torch::Tensor fn, torch::Tensor scale, torch::Tensor base, torch::Tensor normw, double norm_eps,
               double hc_eps, int64_t iters, torch::Tensor xn, torch::Tensor pre_o, torch::Tensor post_o,
               torch::Tensor comb_o, torch::Tensor out, torch::Tensor part, torch::Tensor ptok_o) {
  const int R = (int)y.size(0);
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(R <= 128 && ptok_o.scalar_type() == torch::kInt32 && ptok_o.numel() >= R, "mhc boundary2: shapes");
  if (MHC_V2)
    mhc::k_mhc_post_mix2<false><<<dim3(mhc::NCH / 2, 2), mhc::THR, 0, st>>>(
        (const __nv_bfloat16*)y.data_ptr(), (const __nv_bfloat16*)rs.data_ptr(), post.data_ptr<float>(),
        comb.data_ptr<float>(), (const float2*)fn.data_ptr(), R, (__nv_bfloat16*)xn.data_ptr(), part.data_ptr<float>());
  else
    mhc::k_mhc_post_mix<<<mhc::NCH, mhc::THR, 0, st>>>(
        (const __nv_bfloat16*)y.data_ptr(), (const __nv_bfloat16*)rs.data_ptr(), post.data_ptr<float>(),
        comb.data_ptr<float>(), (const float2*)fn.data_ptr(), R, (__nv_bfloat16*)xn.data_ptr(), part.data_ptr<float>());
  mhc::k_mhc_finish<0><<<R, mhc::THR, 0, st>>>(
      part.data_ptr<float>(), (const __nv_bfloat16*)xn.data_ptr(), lag.data_ptr<float>(), scale.data_ptr<float>(),
      base.data_ptr<float>(), normw.data_ptr<float>(), R, (float)norm_eps, (float)hc_eps, (int)iters,
      pre_o.data_ptr<float>(), post_o.data_ptr<float>(), comb_o.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      mhc::Push{}, ptok_o.data_ptr<int>());
}

// the same boundary, but the output row is pushed into every rank's all-gather buffer (slot `rank`, rows of R) with
// release flags (W * R signals per round on each rank); bufs / flags: 4 tensors each (one per rank, peer-accessible)
void boundary_push(torch::Tensor y, torch::Tensor rs, torch::Tensor post, torch::Tensor comb, torch::Tensor lag,
                   torch::Tensor fn, torch::Tensor scale, torch::Tensor base, torch::Tensor normw, double norm_eps,
                   double hc_eps, int64_t iters, torch::Tensor xn, torch::Tensor pre_o, torch::Tensor post_o,
                   torch::Tensor comb_o, torch::Tensor part, int64_t rank, std::vector<torch::Tensor> bufs,
                   std::vector<torch::Tensor> flags) {
  const int R = (int)y.size(0);
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(R <= 128 && bufs.size() == 4 && flags.size() == 4, "mhc push: shapes");
  mhc::Push pu{};
  for (int i = 0; i < 4; ++i) pu.buf[i] = (__nv_bfloat16*)bufs[i].data_ptr(), pu.flag[i] = (unsigned*)flags[i].data_ptr();
  pu.rank = (int)rank, pu.on = 1;
  if (MHC_V2)
    mhc::k_mhc_post_mix2<false><<<dim3(mhc::NCH / 2, 2), mhc::THR, 0, st>>>(
        (const __nv_bfloat16*)y.data_ptr(), (const __nv_bfloat16*)rs.data_ptr(), post.data_ptr<float>(),
        comb.data_ptr<float>(), (const float2*)fn.data_ptr(), R, (__nv_bfloat16*)xn.data_ptr(), part.data_ptr<float>());
  else
    mhc::k_mhc_post_mix<<<mhc::NCH, mhc::THR, 0, st>>>(
        (const __nv_bfloat16*)y.data_ptr(), (const __nv_bfloat16*)rs.data_ptr(), post.data_ptr<float>(),
        comb.data_ptr<float>(), (const float2*)fn.data_ptr(), R, (__nv_bfloat16*)xn.data_ptr(), part.data_ptr<float>());
  mhc::k_mhc_finish<0><<<R, mhc::THR, 0, st>>>(
      part.data_ptr<float>(), (const __nv_bfloat16*)xn.data_ptr(), lag.data_ptr<float>(), scale.data_ptr<float>(),
      base.data_ptr<float>(), normw.data_ptr<float>(), R, (float)norm_eps, (float)hc_eps, (int)iters,
      pre_o.data_ptr<float>(), post_o.data_ptr<float>(), comb_o.data_ptr<float>(), nullptr, pu);
}

// boundary2 with the RS-sum folded into K1a: y = bf16(sum of recv [4, R, D] slots in rank order, fp32) after
// flag >= *epoch * signals (comm.cu's k_rs_sum fused; its output bits). ptok_o may be empty (no prescale exponent).
void boundary_rs(torch::Tensor recv, torch::Tensor flag, torch::Tensor epoch, int64_t signals, torch::Tensor rs,
                 torch::Tensor post, torch::Tensor comb, torch::Tensor lag, torch::Tensor fn, torch::Tensor scale,
                 torch::Tensor base, torch::Tensor normw, double norm_eps, double hc_eps, int64_t iters, torch::Tensor xn,
                 torch::Tensor pre_o, torch::Tensor post_o, torch::Tensor comb_o, torch::Tensor out, torch::Tensor part,
                 torch::Tensor ptok_o) {
  const int R = (int)recv.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(R <= 128 && recv.size(0) == 4 && recv.size(2) == mhc::D, "mhc boundary_rs: recv [4, R, D]");
  mhc::Fold fo{(const __nv_bfloat16*)recv.data_ptr(), (const unsigned*)flag.data_ptr(), (const unsigned*)epoch.data_ptr(),
               (unsigned)signals};
  mhc::k_mhc_post_mix2<true><<<dim3(mhc::NCH / 2, 2), mhc::THR, 0, st>>>(
      nullptr, (const __nv_bfloat16*)rs.data_ptr(), post.data_ptr<float>(), comb.data_ptr<float>(),
      (const float2*)fn.data_ptr(), R, (__nv_bfloat16*)xn.data_ptr(), part.data_ptr<float>(), fo);
  mhc::k_mhc_finish<0><<<R, mhc::THR, 0, st>>>(
      part.data_ptr<float>(), (const __nv_bfloat16*)xn.data_ptr(), lag.data_ptr<float>(), scale.data_ptr<float>(),
      base.data_ptr<float>(), normw.data_ptr<float>(), R, (float)norm_eps, (float)hc_eps, (int)iters,
      pre_o.data_ptr<float>(), post_o.data_ptr<float>(), comb_o.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      mhc::Push{}, ptok_o.numel() ? ptok_o.data_ptr<int>() : nullptr);
}

// split boundary: k1a (optionally RS-folded), then k1b_out on the critical path and k1b_mix on any stream
// after k1a (it reads only part). Together: the same outputs as boundary / boundary2 / boundary_rs, bit for bit.
void k1a(torch::Tensor y, torch::Tensor rs, torch::Tensor post, torch::Tensor comb, torch::Tensor fn, torch::Tensor xn,
         torch::Tensor part) {
  const int R = (int)y.size(0);
  TORCH_CHECK(R <= 128, "mhc k1a: R <= 128");
  mhc::k_mhc_post_mix2<false><<<dim3(mhc::NCH / 2, 2), mhc::THR, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)y.data_ptr(), (const __nv_bfloat16*)rs.data_ptr(), post.data_ptr<float>(),
      comb.data_ptr<float>(), (const float2*)fn.data_ptr(), R, (__nv_bfloat16*)xn.data_ptr(), part.data_ptr<float>());
}
void k1a_rs(torch::Tensor recv, torch::Tensor flag, torch::Tensor epoch, int64_t signals, torch::Tensor rs,
            torch::Tensor post, torch::Tensor comb, torch::Tensor fn, torch::Tensor xn, torch::Tensor part) {
  const int R = (int)recv.size(1);
  TORCH_CHECK(R <= 128 && recv.size(0) == 4 && recv.size(2) == mhc::D, "mhc k1a_rs: recv [4, R, D]");
  mhc::Fold fo{(const __nv_bfloat16*)recv.data_ptr(), (const unsigned*)flag.data_ptr(), (const unsigned*)epoch.data_ptr(),
               (unsigned)signals};
  mhc::k_mhc_post_mix2<true><<<dim3(mhc::NCH / 2, 2), mhc::THR, 0, at::cuda::getCurrentCUDAStream()>>>(
      nullptr, (const __nv_bfloat16*)rs.data_ptr(), post.data_ptr<float>(), comb.data_ptr<float>(),
      (const float2*)fn.data_ptr(), R, (__nv_bfloat16*)xn.data_ptr(), part.data_ptr<float>(), fo);
}
void k1b_out(torch::Tensor xn, torch::Tensor lag, torch::Tensor normw, double norm_eps, torch::Tensor out,
             torch::Tensor ptok_o) {
  const int R = (int)xn.size(0);
  mhc::k_mhc_finish<1><<<R, mhc::THR, 0, at::cuda::getCurrentCUDAStream()>>>(
      nullptr, (const __nv_bfloat16*)xn.data_ptr(), lag.data_ptr<float>(), nullptr, nullptr, normw.data_ptr<float>(), R,
      (float)norm_eps, 0.f, 0, nullptr, nullptr, nullptr, (__nv_bfloat16*)out.data_ptr(), mhc::Push{},
      ptok_o.numel() ? ptok_o.data_ptr<int>() : nullptr);
}
void k1b_mix(torch::Tensor part, int64_t R, torch::Tensor scale, torch::Tensor base, double norm_eps, double hc_eps,
             int64_t iters, torch::Tensor pre_o, torch::Tensor post_o, torch::Tensor comb_o) {
  mhc::k_mhc_finish<2><<<(int)R, mhc::THR, 0, at::cuda::getCurrentCUDAStream()>>>(
      part.data_ptr<float>(), nullptr, nullptr, scale.data_ptr<float>(), base.data_ptr<float>(), nullptr, (int)R,
      (float)norm_eps, (float)hc_eps, (int)iters, pre_o.data_ptr<float>(), post_o.data_ptr<float>(),
      comb_o.data_ptr<float>(), nullptr, mhc::Push{}, nullptr);
}

// fn [24, 4 * D] fp32 -> B fragments [160 chunks][16 k-steps][3 n-tiles][hi, lo][32 lanes] float2 (tf32 bit patterns):
// lane (g, t), reg q: k_local = 8 s + t + 4 q (chunk-local k = 32 copy + column), n = 8 n-tile + g. hi = tf32(fn) (round
// to nearest, ties away, as cvt.rna), lo = tf32(fn - hi).
torch::Tensor prep_fn(torch::Tensor fn) {
  auto f = fn.contiguous().to(torch::kFloat32).view({mhc::NMIX, mhc::HC, mhc::NCH, mhc::CW});  // [n][copy][ch][col]
  auto fk = f.permute({2, 1, 3, 0}).contiguous().view({mhc::NCH, mhc::HC * mhc::CW, mhc::NMIX});  // [ch][k_local][n]
  auto rna = [](torch::Tensor v) {
    auto b = v.view(torch::kInt32);
    return ((b + 0x1000) & ~0x1FFF).view(torch::kFloat32);
  };
  auto hi = rna(fk.clone());
  auto lo = rna((fk - hi).contiguous());
  // gather into [ch][s][nt][part][lane][q]
  auto idx_k = torch::empty({mhc::KS, 32, 2}, torch::kLong), idx_n = torch::empty({mhc::NT, 32}, torch::kLong);
  for (int s = 0; s < mhc::KS; ++s)
    for (int l = 0; l < 32; ++l)
      for (int q = 0; q < 2; ++q) idx_k[s][l][q] = 8 * s + (l & 3) + 4 * q;
  for (int n = 0; n < mhc::NT; ++n)
    for (int l = 0; l < 32; ++l) idx_n[n][l] = 8 * n + (l >> 2);
  auto dev = fn.device();
  auto ik = idx_k.to(dev), in_ = idx_n.to(dev);
  auto out = torch::empty({mhc::NCH, mhc::KS, mhc::NT, 2, 32, 2}, fk.options());
  for (int pidx = 0; pidx < 2; ++pidx) {
    auto src = pidx == 0 ? hi : lo;  // [ch][k][n]
    // out[ch][s][nt][p][l][q] = src[ch][ik[s][l][q]][in_[nt][l]]
    auto kk = ik.view({1, mhc::KS, 1, 32, 2}).expand({mhc::NCH, mhc::KS, mhc::NT, 32, 2});
    auto nn = in_.view({1, 1, mhc::NT, 32, 1}).expand({mhc::NCH, mhc::KS, mhc::NT, 32, 2});
    auto cc = torch::arange(mhc::NCH, ik.options()).view({mhc::NCH, 1, 1, 1, 1}).expand({mhc::NCH, mhc::KS, mhc::NT, 32, 2});
    auto flat = (cc * (mhc::HC * mhc::CW) + kk) * mhc::NMIX + nn;
    out.select(3, pidx).copy_(src.reshape({-1}).index({flat}));
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("boundary", &boundary);
  m.def("boundary2", &boundary2);
  m.def("boundary_push", &boundary_push);
  m.def("boundary_rs", &boundary_rs);
  m.def("k1a", &k1a);
  m.def("k1a_rs", &k1a_rs);
  m.def("k1b_out", &k1b_out);
  m.def("k1b_mix", &k1b_mix);
  m.def("prep_fn", &prep_fn);
}
