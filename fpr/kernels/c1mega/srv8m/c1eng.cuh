// c1eng.cuh: the Engram layer phase for c1mega. Self-contained (namespace c1eg). Called at c1mega's hook 2 (consumers, before the A1
// attention boundary) on ALL 4 GPUs (the residual stream is replicated):
//   kv: the engine's prepared Engram rows of this step (EngramPhys.kv[li] on the owner: [NR][25600] bf16 = key 0..3 |
//       value, complete before the launch); the owner's CTA c pushes its hidden-dimension columns to every peer's
//       CTA c (posted stores + a release flag), peers wait for their own flag (no remote loads);
//   the caller then runs its boundary in identity mode on X1' (the engine's p_attn_bd: y = 0, post = 0, comb = I).
// Same math as kernels/engram k_inject (fp64 partials summed in another order; the one bf16 rounding point is the same).
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace c1eg {
constexpr int NRM = 4, HC = 4, D = 5120, CTH = 256, NCTA = 132, W = 4;
enum { EG_P = 0, EG_N = 4 };

struct EngArgs {
  int li;                       // 0: layer 1, 1: layer 14
  const __nv_bfloat16* kv;      // owner: [NR][kv_stride] bf16 (key 0 | key 1 | key 2 | key 3 | value, D each)
  long long kv_stride;
  const unsigned char* keep;    // owner: [NR] (stock engram_mask: 0 = image token, no injection)
  __nv_bfloat16* kvl[W];        // rank s: landing [NRM][5][D] bf16 (columns of CTA c written by the owner's CTA c)
  unsigned char* keepl[W];      // rank s: landing [NCTA][NRM] (per CTA, written with its columns)
  unsigned* kflag[W];           // rank s: [NCTA] arrival epochs
  const __nv_bfloat16* qw;      // [4][D] Engram q_weight
  const __nv_bfloat16* kw;      // [4][D] Engram k_weight
  double eps, clampv;
  double* part;                 // this GPU: [NCTA][NRM][12]
  __nv_bfloat16* x1;            // this GPU: [NRM][4][D] the injected streams (each CTA writes / reads its columns)
  const __nv_bfloat16* zrow;    // this GPU: [NRM][D] zeros (the identity-mode y)
  unsigned* cnt;                // this GPU: [EG_N]
  unsigned* tick;               // this GPU: [NCTA] per-CTA Engram epoch (the calls of this phase so far)
  // E0 (c1e0.cuh: the wkv projections in-kernel, TP4, at layer 0). li = -1: this struct is layer 0's, e0 its E0Args.
  // Layers 1 / 14 with kvloc set ("local mode"): kv / keep are this GPU's (E0 wrote them), no owner push; the CTA waits
  // for kvcnt[li] >= 528 x e0tick[c] (E0 ran at layer 0 of this launch).
  const void* e0;
  const __nv_bfloat16* kvloc;   // this GPU: [NRM][25600] of this layer (E0Args::kvd[rank][li])
  const unsigned char* keeploc; // this GPU: [NRM] (E0Args::keepl[rank])
  const unsigned* kvcnt;        // this GPU: [2] (E0Args::kvcnt[rank])
  const unsigned* e0tick;       // this GPU: [NCTA] (E0Args::tick)
  long long* tim;
};

__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, %0;" ::"n"(CTH) : "memory"); }
__device__ __forceinline__ void st_rel_sys(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acq_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// waits poll with RELAXED loads and acquire ONCE: an acquire load compiles to LDG.STRONG + CCTL.IVALL (the whole
// SM's L1 invalidated per poll), which in situ stalls the other warps' LSU traffic (smem included) on the same SM
__device__ __forceinline__ unsigned ld_rlx_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_rlx(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fence_acq_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_acq() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void red_rel(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

#ifdef C1EG_TIM
__device__ __forceinline__ void egt_stamp(const EngArgs& e, int c, int tid, int k) {
  if (tid == 0 && e.tim != nullptr) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    e.tim[c * 32 + k] = (long long)t;
  }
}
#define EGT(k) egt_stamp(e, c, tid, k)
#else
#define EGT(k)
#endif
#ifndef C1EG_V1
// The injection for this CTA's columns col0 .. col0 + ncols - 1 (16-column tiles; ncols <= 48). post / comb: the
// boundary's mixes per row (shared memory, c1mega's S.post / S.comb); y = xin [NR][D], rs = rsin [NR][4][D]. On return
// e.x1 holds X1' for the columns and the caller switches its boundary to identity mode. red: >= 512 doubles of shared
// scratch, gs: >= 16 doubles. All 256 consumer threads.
// v2: the same arithmetic, summation orders and memory traffic per row as v1 (bitwise), with
//   - the per-row warp reductions only in the warps that hold columns (tid < ncols: w < nw). A warp without columns has
//     12 partials of exactly +0.0, and the CTA partial s (from +0.0, never -0.0: an exact cancellation rounds to +0.0)
//     is unchanged by adding +0.0, so skipping those warps keeps the bits;
//   - no barriers inside the row loop (red indexed by row);
//   - the 132 CTA partials of a sum loaded in batches of 12 independent loads by one thread per (row, sum), then added
//     in CTA order from 0.0 (v1: 16 threads, 3 sums each, one dependent L2 round trip per CTA);
//   - the counter polled relaxed, acquired once; the epoch read by every thread (no broadcast barriers).
// (A first v2 preloaded every row's inputs into registers: +272 B of stack for the noinline glue's saved registers, i.e.
// +69 MB per GPU of local-memory reservation at 2048 threads x 132 SMs. Kept out.)
__device__ void inject(const EngArgs& e, int rank, int owner, int NR, int c, int col0, int ncols, const __nv_bfloat16* xin,
                       const __nv_bfloat16* rsin, const float (*post)[HC], const float (*comb)[16], double* red, double* gs,
                       int tid) {
  const int lane = tid & 31, w = tid >> 5;
  // (thread 0 wrote e.tick[c] at the end of the previous call, barriers ago; it writes it again only after the last one)
  const unsigned ecall = e.tick[c] + 1u;
  // ---- the kv columns: the owner pushes, peers wait
  const __nv_bfloat16* kvb;   // [n][5][D] view: element (n, j, col) at kvb[n * ks + j * D + col]
  long long ks;
  const unsigned char* kp;
  if (e.kvloc != nullptr) {   // local mode: E0's kv on this GPU
    kvb = e.kvloc, ks = 5LL * D, kp = e.keeploc;
    if (tid == 0) {
      while ((int)(ld_rlx_sys(e.kvcnt + e.li) - 528u * e.e0tick[c]) < 0) {
      }
      fence_acq_sys();
    }
    cbar();
    EGT(6);
  } else if (rank == owner) {
    kvb = e.kv, ks = e.kv_stride, kp = e.keep;
    const int nch = ncols / 8;   // 16-B chunks per (row, part)
    for (int i = tid; i < NR * 5 * nch; i += CTH) {
      const int n = i / (5 * nch), j = (i / nch) % 5, q = i % nch;
      const uint4 v = __ldcg(reinterpret_cast<const uint4*>(e.kv + n * e.kv_stride + j * D + col0) + q);
#pragma unroll
      for (int s = 0; s < W; ++s)
        if (s != rank) *(reinterpret_cast<uint4*>(e.kvl[s] + ((size_t)n * 5 + j) * D + col0) + q) = v;
    }
    if (tid < NR)
#pragma unroll
      for (int s = 0; s < W; ++s)
        if (s != rank) e.keepl[s][c * NRM + tid] = e.keep[tid];
    __threadfence_system();
    cbar();
    if (tid < W && tid != rank) st_rel_sys(e.kflag[tid] + c, ecall);
  } else {
    kvb = e.kvl[rank], ks = 5LL * D, kp = e.keepl[rank] + c * NRM;
    if (tid == 0) {
      while ((int)(ld_acq_sys(e.kflag[rank] + c) - ecall) < 0) {
      }
    }
    cbar();
  }
  // ---- X1 = hc_post on the columns, fp64 partial sums per row (row n: threads tid < ncols, one column each). As v1 (the
  // same loads per row, X1 stored and re-read for X1': v1's register profile, so the noinline glue's frame stays small),
  // without the per-row barriers (red indexed by row) and with the reductions only in the warps holding columns.
  const int nw = (ncols + 31) >> 5;
  for (int n = 0; n < NR; ++n) {
    double acc[12];
#pragma unroll
    for (int k = 0; k < 12; ++k) acc[k] = 0.0;
    if (tid < ncols) {
      const int col = col0 + tid;
      const float yv = __bfloat162float(xin[(size_t)n * D + col]);
      float rv[HC];
#pragma unroll
      for (int cc = 0; cc < HC; ++cc) rv[cc] = __bfloat162float(rsin[((size_t)n * HC + cc) * D + col]);
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float a = post[n][j] * yv;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) a = fmaf(comb[n][cc * HC + j], rv[cc], a);
        const __nv_bfloat16 xb = __float2bfloat16_rn(a);
        e.x1[((size_t)n * HC + j) * D + col] = xb;
        const double x = (double)__bfloat162float(xb);
        const double k = (double)__bfloat162float(kvb[n * ks + (long long)j * D + col]);
        const double qv = (double)__bfloat162float(e.qw[j * D + col]), wv = (double)__bfloat162float(e.kw[j * D + col]);
        acc[3 * j] = fma(x, x, acc[3 * j]);
        acc[3 * j + 1] = fma(k, k, acc[3 * j + 1]);
        acc[3 * j + 2] = fma(x * qv * wv, k, acc[3 * j + 2]);
      }
    }
    if (w < nw) {   // (warp-uniform)
#pragma unroll
      for (int k = 0; k < 12; ++k) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[k] += __shfl_xor_sync(0xffffffffu, acc[k], o);
      }
      if (lane == 0)
#pragma unroll
        for (int k = 0; k < 12; ++k) red[(n * 8 + w) * 12 + k] = acc[k];
    }
  }
  cbar();
  if (tid < NR * 12) {   // this CTA's partial of (row n, sum k): the warps' sums in warp order
    const int n = tid / 12, k = tid % 12;
    double s = 0.0;
    for (int ww = 0; ww < nw; ++ww) s += red[(n * 8 + ww) * 12 + k];
    e.part[((size_t)c * NRM + n) * 12 + k] = s;
  }
  // identical on all CTAs and GPUs)
  cbar();
  EGT(7);
  if (tid == 0) {
    __threadfence();
    red_rel(e.cnt + EG_P, 1u);
    while ((int)(ld_rlx(e.cnt + EG_P) - (unsigned)NCTA * ecall) < 0) {
    }
    fence_acq();
  }
  cbar();
  EGT(15);
  if (tid < NR * 12) {   // sum (n, k) over the 132 CTAs in CTA order, from 0.0 (batches of 12: 24 regs)
    constexpr int B = 12;
    static_assert(NCTA % B == 0, "batches");
    const double* p = e.part + tid;   // (n * 12 + k = tid)
    double s = 0.0;
#pragma unroll 1
    for (int c0 = 0; c0 < NCTA; c0 += B) {
      double v[B];
#pragma unroll
      for (int i = 0; i < B; ++i) v[i] = __ldcg(p + (size_t)(c0 + i) * NRM * 12);
#pragma unroll
      for (int i = 0; i < B; ++i) s += v[i];
    }
    red[tid] = s;
  }
  cbar();
  if (tid < NR * HC) {
    const int n = tid >> 2, j = tid & 3;
    const double hh = red[n * 12 + 3 * j], kk = red[n * 12 + 3 * j + 1], hk = red[n * 12 + 3 * j + 2];
    const double hr = 1.0 / sqrt(hh / D + e.eps), kr = 1.0 / sqrt(kk / D + e.eps);
    const double dot = hk * hr * kr / sqrt((double)D);
    double gi = sqrt(fmax(fabs(dot), e.clampv));
    if (dot < 0.0) gi = -gi;
    double g = 1.0 / (1.0 + exp(-gi));
    if (!kp[n]) g = 0.0;
    gs[tid] = g;
  }
  cbar();
  EGT(31);
  // ---- X1' = bf16(x + g value) on the columns (each thread rewrites the columns it wrote above)
  for (int n = 0; n < NR; ++n)
    if (tid < ncols) {
      const int col = col0 + tid;
      const double v = (double)__bfloat162float(kvb[n * ks + 4LL * D + col]);
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        __nv_bfloat16* xp = e.x1 + ((size_t)n * HC + j) * D + col;
        *xp = __double2bfloat16(fma(gs[n * HC + j], v, (double)__bfloat162float(*xp)));
      }
    }
  cbar();
  if (tid == 0) e.tick[c] = ecall;
}
#else   // C1EG_V1 (timeline twins only): srv8x's inject, with the same sub-stamps
// The injection for this CTA's columns col0 .. col0 + ncols - 1 (16-column tiles; ncols <= 48). post / comb: the
// boundary's mixes per row (shared memory, c1mega's S.post / S.comb); y = xin [NR][D], rs = rsin [NR][4][D]. On return
// e.x1 holds X1' for the columns and the caller switches its boundary to identity mode. red: >= 8 x 12 doubles of shared
// scratch, gs: >= 16 doubles. All 256 consumer threads.
__device__ void inject(const EngArgs& e, int rank, int owner, int NR, int c, int col0, int ncols, const __nv_bfloat16* xin,
                       const __nv_bfloat16* rsin, const float (*post)[HC], const float (*comb)[16], double* red, double* gs,
                       int tid) {
  const int lane = tid & 31, w = tid >> 5;
  if (tid == 0) reinterpret_cast<unsigned*>(gs)[0] = e.tick[c] + 1u;   // (no static shared memory: c1mega has none to spare)
  cbar();
  const unsigned ecall = reinterpret_cast<const unsigned*>(gs)[0];
  cbar();
  // ---- the kv columns: the owner pushes, peers wait
  const __nv_bfloat16* kvb;   // [n][5][D] view: element (n, j, col) at kvb[n * ks + j * D + col]
  long long ks;
  const unsigned char* kp;
  if (e.kvloc != nullptr) {   // local mode: E0's kv on this GPU
    kvb = e.kvloc, ks = 5LL * D, kp = e.keeploc;
    if (tid == 0) {
      while ((int)(ld_acq_sys(e.kvcnt + e.li) - 528u * e.e0tick[c]) < 0) {   // (all-wait: acquire polls, v13)
      }
    }
    cbar();
    EGT(6);
  } else if (rank == owner) {
    kvb = e.kv, ks = e.kv_stride, kp = e.keep;
    const int nch = ncols / 8;   // 16-B chunks per (row, part)
    for (int i = tid; i < NR * 5 * nch; i += CTH) {
      const int n = i / (5 * nch), j = (i / nch) % 5, q = i % nch;
      const uint4 v = __ldcg(reinterpret_cast<const uint4*>(e.kv + n * e.kv_stride + j * D + col0) + q);
#pragma unroll
      for (int s = 0; s < W; ++s)
        if (s != rank) *(reinterpret_cast<uint4*>(e.kvl[s] + ((size_t)n * 5 + j) * D + col0) + q) = v;
    }
    if (tid < NR)
#pragma unroll
      for (int s = 0; s < W; ++s)
        if (s != rank) e.keepl[s][c * NRM + tid] = e.keep[tid];
    __threadfence_system();
    cbar();
    if (tid < W && tid != rank) st_rel_sys(e.kflag[tid] + c, ecall);
  } else {
    kvb = e.kvl[rank], ks = 5LL * D, kp = e.keepl[rank] + c * NRM;
    if (tid == 0) {
      while ((int)(ld_acq_sys(e.kflag[rank] + c) - ecall) < 0) {
      }
    }
    cbar();
  }
  // ---- X1 = hc_post on the columns, fp64 partial sums per row (row n: threads tid < ncols, one column each)
  for (int n = 0; n < NR; ++n) {
    double acc[12];
#pragma unroll
    for (int k = 0; k < 12; ++k) acc[k] = 0.0;
    if (tid < ncols) {
      const int col = col0 + tid;
      const float yv = __bfloat162float(xin[(size_t)n * D + col]);
      float rv[HC];
#pragma unroll
      for (int cc = 0; cc < HC; ++cc) rv[cc] = __bfloat162float(rsin[((size_t)n * HC + cc) * D + col]);
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float a = post[n][j] * yv;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) a = fmaf(comb[n][cc * HC + j], rv[cc], a);
        const __nv_bfloat16 xb = __float2bfloat16_rn(a);
        e.x1[((size_t)n * HC + j) * D + col] = xb;
        const double x = (double)__bfloat162float(xb);
        const double k = (double)__bfloat162float(kvb[n * ks + (long long)j * D + col]);
        const double qv = (double)__bfloat162float(e.qw[j * D + col]), wv = (double)__bfloat162float(e.kw[j * D + col]);
        acc[3 * j] = fma(x, x, acc[3 * j]);
        acc[3 * j + 1] = fma(k, k, acc[3 * j + 1]);
        acc[3 * j + 2] = fma(x * qv * wv, k, acc[3 * j + 2]);
      }
    }
#pragma unroll
    for (int k = 0; k < 12; ++k) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc[k] += __shfl_xor_sync(0xffffffffu, acc[k], o);
    }
    if (lane == 0)
#pragma unroll
      for (int k = 0; k < 12; ++k) red[w * 12 + k] = acc[k];
    cbar();
    if (tid < 12) {
      double s = 0.0;
      for (int ww = 0; ww < CTH / 32; ++ww) s += red[ww * 12 + tid];
      e.part[((size_t)c * NRM + n) * 12 + tid] = s;
    }
    cbar();
  }
  // identical on all CTAs and GPUs)
  cbar();
  EGT(7);
  if (tid == 0) {
    __threadfence();
    red_rel(e.cnt + EG_P, 1u);
    while ((int)(ld_acq(e.cnt + EG_P) - (unsigned)NCTA * ecall) < 0) {
    }
  }
  cbar();
  EGT(15);
  if (tid < NR * HC) {
    const int n = tid >> 2, j = tid & 3;
    double hh = 0.0, kk = 0.0, hk = 0.0;
    for (int cc = 0; cc < NCTA; ++cc) {
      const double* p = e.part + ((size_t)cc * NRM + n) * 12 + 3 * j;
      hh += __ldcg(p), kk += __ldcg(p + 1), hk += __ldcg(p + 2);
    }
    const double hr = 1.0 / sqrt(hh / D + e.eps), kr = 1.0 / sqrt(kk / D + e.eps);
    const double dot = hk * hr * kr / sqrt((double)D);
    double gi = sqrt(fmax(fabs(dot), e.clampv));
    if (dot < 0.0) gi = -gi;
    double g = 1.0 / (1.0 + exp(-gi));
    if (!kp[n]) g = 0.0;
    gs[tid] = g;
  }
  cbar();
  EGT(31);
  // ---- X1' = bf16(x + g value) on the columns (each thread rewrites the columns it wrote above)
  for (int n = 0; n < NR; ++n)
    if (tid < ncols) {
      const int col = col0 + tid;
      const double v = (double)__bfloat162float(kvb[n * ks + 4LL * D + col]);
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        __nv_bfloat16* xp = e.x1 + ((size_t)n * HC + j) * D + col;
        *xp = __double2bfloat16(fma(gs[n * HC + j], v, (double)__bfloat162float(*xp)));
      }
    }
  cbar();
  if (tid == 0) e.tick[c] = ecall;
}
#endif   // C1EG_V1
}  // namespace c1eg
