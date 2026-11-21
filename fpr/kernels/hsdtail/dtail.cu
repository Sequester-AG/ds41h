// dtail: the C1 draft tail of HSD in ONE persistent kernel per GPU.
// Replaces, for drafter buckets of <= 16 draft rows (C1: 4 requests x k = 3), kernels/head's
//   DraftHead e4m3 (head.py _rowq_kernel + torch._scaled_mm rowwise)  and  the k Markov draws
//   (sampler.py draft3_local + the "d{i}" all-gather + draft_combine, and the bias GEMM emb @ w2^T of every next slot).
// Same math (the drafter may approximate; verification stays exact): per-row e4m3 activations with amax / 448 scales,
// the e4m3 head with its per-vocab-row fp32 scales, fp32 accumulation (another order than cuBLASLt's), one bf16 rounding;
// Markov: bias = bf16(emb . w2) (fp32 accumulation, another order), logits = bf16(base + bias) (the cache gets them),
// x = logits / T + Gumbel noise (the same pre-computed noise), first max (lowest global token id on ties), lse of
// logits / T; the winner's w1 row is the next slot's Markov embedding.
//
// Layout: CTA c (132, one per SM, all co-resident) owns n8 vocab tiles [c * NT / G, (c + 1) * NT / G) of this rank's slice.
//   Phase 0: every CTA quantizes the R <= 16 draft rows itself (L2 reads) into SMEM (rows padded to 5184 B: conflict-free).
//   Phase 1: each warp streams whole n8 tiles (8 vocab rows x 5120 B) straight from HBM (16-B loads, 8 chunks of 64 k in
//            flight) into mma.sync m16n8k32 e4m3 (k permuted identically in A and B: a thread's 16 contiguous bytes feed two
//            MMAs); base = bf16(acc * xs[row] * ws[v]) stays in SMEM (and is written to base_out for checks).
//   Markov step i: per (request q, own vocab v): bias (i > 0) from emb_i (SMEM) and w2 (L2-prefetched), logits, cache,
//            argmax / (m, s) -> CTA partial -> grid barrier -> CTA q merges the 132 partials and pushes its rank's 5 words
//            to every rank's exchange slot (lamport: the words carry the arrival, sentinel 0xFFFFFFFF) -> every CTA waits for
//            the 4 ranks' words, picks the winner (lowest rank on ties = lowest token id), loads w1[winner] for slot i + 1;
//            CTA 0 writes draft_tokens / token_logit / lse.
//   Exchange buffers are double-buffered by launch parity (from the barrier counter); CTA 0 resets the other parity at start.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace dt {
constexpr int K = 5120, RW = 256, G = 132, THR = 256, NW = THR / 32;
constexpr int APAD = K + 64;              // SMEM row stride of the quantized activations (bank-conflict-free 16-B reads)
constexpr int MAXR = 12, MAXQ = 4, MAXS = 4, MAXV = 256;   // rows, requests, Markov slots, vocab rows per CTA
constexpr int NWRD = 8;                   // words per exchange entry (5 used)
constexpr unsigned SENT = 0xFFFFFFFFu;

struct Peers { float* x[4]; };

struct Args {
  const __nv_bfloat16* hd; int R;                 // [R, K] draft hidden (rank-major requests, k rows each)
  const unsigned char* wq; const float* ws; int loc, vstart;   // e4m3 head slice [loc, K], scales [loc]
  const __nv_bfloat16* w2; const __nv_bfloat16* w1;           // [loc, RW] slice, [V, RW] replicated
  const __nv_bfloat16* bias0; long long bias0_rs;              // [n, loc] slot-0 bias (side stream)
  const float* noise; long long nz_s0, nz_s1;                  // noise[i * s0 + q * s1 + v]
  const int* idx; long long idx_s0, idx_s1;                    // request-state index of (q, i); -1: none
  const float* temp;
  __nv_bfloat16* cache; long long c_s0, c_s1;                  // cache[req * s0 + i * s1 + v]
  long long* tokens; long long tok_s0;                         // draft_tokens[q * s0 + i]
  float* tlogit; float* lse; long long vp_s0; int sink;
  __nv_bfloat16* base_out;                                     // [R, loc] (checks; may be null)
  float* part;                                                 // [MAXQ][G][5] scratch
  unsigned long long* bar;                                     // [1] monotonic arrival counter (64-bit: never wraps)
  float* xloc; Peers xp; int rank, n, k;                       // exchange: [2][MAXS][MAXQ][4][NWRD] on every rank
  unsigned long long* dbg;                                     // [G][16] %globaltimer stamps (null: off)
  int markov;                                                  // 0: head only (base_out), no Markov steps / exchanges
  const unsigned char* xq; const float* xsg;                   // pre-quantized rows [R, K] e4m3 + scales [R] (null: quantize)
};

__device__ __forceinline__ void st_sys(float* p, float v) {
  asm volatile("st.relaxed.sys.global.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_sys_u(const float* p) {
  unsigned v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void mma_e4m3(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ unsigned s_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
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

__device__ __forceinline__ void grid_barrier(unsigned long long* bar, unsigned long long target) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(bar, 1ull);
    unsigned long long v;
    do {
      asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(bar) : "memory");
    } while (v < target);
  }
  __syncthreads();
}

// (value, index) max with the lowest index on ties; NaN never wins
__device__ __forceinline__ void amax_merge(float& v, int& i, float v2, int i2) {
  if (v2 > v || (v2 == v && i2 < i)) { v = v2; i = i2; }
}

#define STAMP(i)                                                                                         \
  do {                                                                                                   \
    if (A.dbg && threadIdx.x == 0) {                                                                     \
      unsigned long long t_;                                                                             \
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));                                            \
      A.dbg[blockIdx.x * 16 + (i)] = t_;                                                                  \
    }                                                                                                    \
  } while (0)

__global__ void __launch_bounds__(THR, 1) k_dtail(Args A) {
  extern __shared__ __align__(16) unsigned char sm[];
  unsigned char* sa = sm;                                         // [MAXR][APAD] e4m3
  float* sbase = (float*)(sm + MAXR * APAD);                      // [MAXR][MAXV] base logits (bf16 values)
  float* sxs = sbase + MAXR * MAXV;                               // [MAXR] activation scales
  float* semb = sxs + MAXR;                                       // [MAXQ][RW] Markov embedding (fp32 of bf16)
  float* sred = semb + MAXQ * RW;                                 // [NW][MAXR] reductions
  float* sbias = sred + NW * MAXR;                                // [MAXQ][MAXV] Markov bias of the own vocab rows
  __nv_bfloat16* sw2 = (__nv_bfloat16*)(sbias + MAXQ * MAXV);     // [MAXV][RW] this CTA's w2 rows (bulk copy at start)
  __shared__ __align__(8) uint64_t w2bar;
  __shared__ float s_v[NW][MAXQ], s_l[NW][MAXQ], s_m[NW][MAXQ], s_s[NW][MAXQ];
  __shared__ int s_i[NW][MAXQ];
  __shared__ int s_tok[MAXQ];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c = blockIdx.x;
  const int R = A.R, n = A.n, k = A.k;
  const int ntile = A.loc / 8;
  const int t0 = (int)((long long)ntile * c / G), t1 = (int)((long long)ntile * (c + 1) / G);
  const int v0 = t0 * 8, nv = (t1 - t0) * 8;                     // own vocab rows [v0, v0 + nv)
  // ---- launch epoch (parity) from the barrier counter: k barriers per launch, G CTAs
  STAMP(0);
  const unsigned long long b0 = *(volatile unsigned long long*)A.bar;
  const unsigned long long epoch = b0 / (unsigned long long)((k > 0 ? k : 1) * G);
  const int par = (int)(epoch & 1);
  const unsigned long long bbase = epoch * (unsigned long long)((k > 0 ? k : 1) * G);
  float* xcur = A.xloc + (size_t)par * MAXS * MAXQ * 4 * NWRD;
  if (c == 0 && A.markov) {   // reset the other parity (read by the previous launch, complete)
    float* xo = A.xloc + (size_t)(par ^ 1) * MAXS * MAXQ * 4 * NWRD;
    for (int e = tid; e < MAXS * MAXQ * 4 * NWRD; e += THR) ((unsigned*)xo)[e] = SENT;
  }
  // this CTA's w2 rows into SMEM (read by the bias GEMV of Markov steps 1 .. k-1), overlapping phases 0 / 1
  if (tid == 0) {
    mb_init(&w2bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (A.xq != nullptr) {   // ---- phase 0': the rows quantized by head.py's _rowq_kernel (same bits as DraftHead e4m3)
    for (int e = tid * 16; e < MAXR * K; e += THR * 16) {
      const int r = e / K, kk = e % K;
      const uint4 u = r < R ? *(const uint4*)(A.xq + (size_t)r * K + kk) : make_uint4(0, 0, 0, 0);
      *(uint4*)(sa + (size_t)r * APAD + kk) = u;
    }
    if (tid < MAXR) sxs[tid] = tid < R ? A.xsg[tid] : 0.f;
    __syncthreads();
  } else
  // ---- phase 0: per-row amax scale, e4m3 rows into SMEM (= head.py _rowq_kernel: s = max(amax, 1e-12) / 448,
  // e4m3(clamp(x / s, +-448)), round to nearest even, saturating)
  {
    float mx[MAXR];
#pragma unroll
    for (int r = 0; r < MAXR; ++r) mx[r] = 0.f;
    for (int e = tid * 8; e < K; e += THR * 8) {
#pragma unroll
      for (int r = 0; r < MAXR; ++r) {
        if (r < R) {
          const uint4 u = *(const uint4*)(A.hd + (size_t)r * K + e);
          const __nv_bfloat162* h = (const __nv_bfloat162*)&u;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(h[j]);
            mx[r] = fmaxf(mx[r], fmaxf(fabsf(f.x), fabsf(f.y)));
          }
        }
      }
    }
#pragma unroll
    for (int r = 0; r < MAXR; ++r) {
#pragma unroll
      for (int o = 16; o; o >>= 1) mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], o));
      if (lane == 0) sred[warp * MAXR + r] = mx[r];
    }
    __syncthreads();
    if (tid < MAXR) {
      float m = 0.f;
      for (int w = 0; w < NW; ++w) m = fmaxf(m, sred[w * MAXR + tid]);
      sxs[tid] = fmaxf(m, 1e-12f) / 448.f;
    }
    __syncthreads();
    for (int e = tid * 8; e < K; e += THR * 8) {
      for (int r = 0; r < MAXR; ++r) {
        unsigned w0 = 0, w1 = 0;
        if (r < R) {
          const uint4 u = *(const uint4*)(A.hd + (size_t)r * K + e);
          const __nv_bfloat162* h = (const __nv_bfloat162*)&u;
          const float is = 1.f / sxs[r];   // x * (1 / s): the drafter's quantization (rowq divides; same math class)
          unsigned short q[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float2 f = __bfloat1622float2(h[j]);
            f.x = fminf(fmaxf(f.x * is, -448.f), 448.f);
            f.y = fminf(fmaxf(f.y * is, -448.f), 448.f);
            q[j] = (unsigned short)__nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
          }
          w0 = (unsigned)q[0] | ((unsigned)q[1] << 16);
          w1 = (unsigned)q[2] | ((unsigned)q[3] << 16);
        }
        *(uint2*)(sa + (size_t)r * APAD + e) = make_uint2(w0, w1);
      }
    }
    __syncthreads();
  }
  // ---- phase 1: base logits of the own vocab tiles (warp w: tiles t0 + w, t0 + w + NW, ...), software-pipelined: the
  // next group of U k64 chunks (U x 16 B per thread) is in flight while the current one feeds the MMAs
  STAMP(1);
  {
    const int g = lane >> 2, t = lane & 3;
    const bool hi = g + 8 < R;
    const unsigned char* arow0 = sa + (size_t)g * APAD + 16 * t;
    const unsigned char* arow1 = sa + (size_t)(g + 8) * APAD + 16 * t;
    constexpr int U = 8, GPT = K / (64 * U);   // 10 groups per tile
    const int nt_w = (t1 - t0 - warp + NW - 1) / NW;
    const int ng = nt_w > 0 ? nt_w * GPT : 0;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    uint4 b0[U], b1[U];
    auto gptr = [&](int j) {
      const int tile = t0 + warp + NW * (j / GPT);
      return A.wq + (size_t)(tile * 8 + g) * K + 16 * t + (size_t)(j % GPT) * 64 * U;
    };
    auto load = [&](uint4 (&bb)[U], int j) {
      const unsigned char* p = gptr(j);
#pragma unroll
      for (int u = 0; u < U; ++u) bb[u] = ld_stream(p + 64 * u);
    };
    auto compute = [&](const uint4 (&bb)[U], int j) {
      const int ko = (j % GPT) * 64 * U;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const uint4 a = *(const uint4*)(arow0 + ko + 64 * u);
        const uint4 a8 = hi ? *(const uint4*)(arow1 + ko + 64 * u) : make_uint4(0, 0, 0, 0);
        mma_e4m3(acc, a.x, a8.x, a.y, a8.y, bb[u].x, bb[u].y);
        mma_e4m3(acc, a.z, a8.z, a.w, a8.w, bb[u].z, bb[u].w);
      }
      if (j % GPT == GPT - 1) {   // tile done: c0, c1 = row g, vocab 2t, 2t + 1; c2, c3 = row g + 8
        const int tile = t0 + warp + NW * (j / GPT);
        const int vl = (tile - t0) * 8 + 2 * t;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = g + 8 * h;
          if (row < R) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int v = tile * 8 + 2 * t + e;
              const float y = bfr(acc[2 * h + e] * sxs[row] * A.ws[v]);
              sbase[row * MAXV + vl + e] = y;
              if (A.base_out) A.base_out[(size_t)row * A.loc + v] = __float2bfloat16_rn(y);
            }
          }
        }
        acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
      }
    };
    if (ng > 0) load(b0, 0);
    for (int j = 0; j < ng; j += 2) {
      if (j + 1 < ng) load(b1, j + 1);
      compute(b0, j);
      if (j + 1 < ng) {
        if (j + 2 < ng) load(b0, j + 2);
        compute(b1, j + 1);
      }
    }
    __syncthreads();
  }
  STAMP(2);
  if (!A.markov) return;
  // this CTA's w2 rows into SMEM now (after the head's stream; Markov step 0 does not read them)
  if (k > 1 && tid == 0) {
    const char* p = (const char*)(A.w2 + (size_t)v0 * RW);
    const unsigned bytes = (unsigned)nv * RW * 2;
    mb_expect(&w2bar, bytes);
    for (unsigned o = 0; o < bytes; o += 16384) bulk_g2s((char*)sw2 + o, p + o, min(16384u, bytes - o), &w2bar);
  }
  // per request q: the request-state index and temperature of every Markov slot, once (no dependent global loads per q)
  __shared__ int s_req[MAXS][MAXQ];
  __shared__ float s_tmp[MAXS][MAXQ];
  if (tid < n * k) {
    const int q = tid / k, i_ = tid % k;
    const int req = A.idx[q * A.idx_s0 + i_ * A.idx_s1];
    s_req[i_][q] = req;
    s_tmp[i_][q] = req >= 0 ? A.temp[req] : 0.f;
  }
  __syncthreads();
  // ---- Markov steps
  for (int i = 0; i < k; ++i) {
    // own vocab row v = v0 + tid (one per thread), every request q
    const int v = v0 + tid;
    const bool own = tid < nv;
    float bias[MAXQ];
    if (i == 0) {
#pragma unroll
      for (int q = 0; q < MAXQ; ++q)
        bias[q] = (own && q < n) ? __bfloat162float(A.bias0[(size_t)q * A.bias0_rs + v]) : 0.f;
    } else {
      // bias[q][v] = bf16(emb_q . w2[v]): warp w takes vocab rows w, w + NW, ...; lane l holds elements 8 l .. 8 l + 7
      // (one coalesced 16-B load per row per lane), 8 rows' loads in flight, warp-shuffle sums
      float ev[MAXQ][8];
#pragma unroll
      for (int q = 0; q < MAXQ; ++q)
#pragma unroll
        for (int j = 0; j < 8; ++j) ev[q][j] = q < n ? semb[q * RW + 8 * lane + j] : 0.f;
      if (i == 1) mb_wait(&w2bar, 0);   // the w2 rows landed (once per launch)
      constexpr int BR = 8;
      for (int vb = warp; vb < nv; vb += NW * BR) {
        uint4 u[BR];
#pragma unroll
        for (int x = 0; x < BR; ++x) {
          const int vl = vb + NW * x;
          u[x] = vl < nv ? *(const uint4*)(sw2 + (size_t)vl * RW + 8 * lane) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int x = 0; x < BR; ++x) {
          const int vl = vb + NW * x;
          const __nv_bfloat162* h = (const __nv_bfloat162*)&u[x];
          float acc[MAXQ];
#pragma unroll
          for (int q = 0; q < MAXQ; ++q) {
            acc[q] = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const float2 f = __bfloat1622float2(h[j]);
              acc[q] = fmaf(ev[q][2 * j], f.x, acc[q]);
              acc[q] = fmaf(ev[q][2 * j + 1], f.y, acc[q]);
            }
#pragma unroll
            for (int o = 16; o; o >>= 1) acc[q] += __shfl_xor_sync(0xffffffffu, acc[q], o);
          }
          if (lane == 0 && vl < nv) {
#pragma unroll
            for (int q = 0; q < MAXQ; ++q) sbias[q * MAXV + vl] = bfr(acc[q]);
          }
        }
      }
      __syncthreads();
#pragma unroll
      for (int q = 0; q < MAXQ; ++q) bias[q] = own ? sbias[q * MAXV + tid] : 0.f;
    }
    // logits, cache, noised value, stats (per q); warp then CTA reductions
#pragma unroll
    for (int q = 0; q < MAXQ; ++q) {
      if (q >= n) break;
      const int req = s_req[i][q];
      const float temp = s_tmp[i][q];
      const float tt = temp > 0.f ? temp : 1.f;
      float val = -INFINITY, lg = -INFINITY, xs_ = -INFINITY;
      int tok = 0x7fffffff;
      if (own) {
        lg = bfr(sbase[(q * k + i) * MAXV + tid] + bias[q]);
        if (req >= 0) A.cache[(size_t)req * A.c_s0 + (size_t)i * A.c_s1 + v] = __float2bfloat16_rn(lg);
        float x = lg;
        if (temp != 0.f) {
          if (temp != 1.f) x = x / temp;
          x = x + A.noise[(size_t)i * A.nz_s0 + (size_t)q * A.nz_s1 + v];
        }
        val = x;
        tok = A.vstart + v;
        xs_ = tt != 1.f ? lg / tt : lg;
      }
      // argmax (value, token) + the logit at the winner; max of logits / T
      float bv = val, bl = lg, bm = xs_;
      int bt = tok;
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, bv, o), l2 = __shfl_xor_sync(0xffffffffu, bl, o);
        const int t2 = __shfl_xor_sync(0xffffffffu, bt, o);
        if (v2 > bv || (v2 == bv && t2 < bt)) { bv = v2; bt = t2; bl = l2; }
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, o));
      }
      if (lane == 0) { s_v[warp][q] = bv; s_i[warp][q] = bt; s_l[warp][q] = bl; s_m[warp][q] = bm; }
    }
    __syncthreads();
    // CTA max of logits / T per q, then sum exp
#pragma unroll
    for (int q = 0; q < MAXQ; ++q) {
      if (q >= n) break;
      float m = -INFINITY;
      for (int w = 0; w < NW; ++w) m = fmaxf(m, s_m[w][q]);
      float e = 0.f;
      if (own && m > -INFINITY) {
        const float temp = s_tmp[i][q];
        const float tt = temp > 0.f ? temp : 1.f;
        const float lg = bfr(sbase[(q * k + i) * MAXV + tid] + bias[q]);
        e = expf((tt != 1.f ? lg / tt : lg) - m);
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
      if (lane == 0) s_s[warp][q] = e;
    }
    __syncthreads();
    if (tid < n) {
      const int q = tid;
      float bv = -INFINITY, bl = -INFINITY, m = -INFINITY, s = 0.f;
      int bt = 0x7fffffff;
      for (int w = 0; w < NW; ++w) {
        if (s_v[w][q] > bv || (s_v[w][q] == bv && s_i[w][q] < bt)) { bv = s_v[w][q]; bt = s_i[w][q]; bl = s_l[w][q]; }
        m = fmaxf(m, s_m[w][q]);
      }
      for (int w = 0; w < NW; ++w) s += s_s[w][q];
      float* p = A.part + ((size_t)q * G + c) * 5;
      p[0] = bv; p[1] = __int_as_float(bt); p[2] = bl; p[3] = m; p[4] = m > -INFINITY ? s : 0.f;
    }
    STAMP(3 + 4 * i);
    grid_barrier(A.bar, bbase + (unsigned long long)((i + 1) * G));
    STAMP(4 + 4 * i);
    // CTA q merges the G partials of request q -> its rank's 5 words to every rank
    if (c < n) {
      const int q = c;
      float bv = -INFINITY, bl = -INFINITY, top = -INFINITY;
      int bt = 0x7fffffff;
      for (int cc = lane; cc < G; cc += 32) {
        const float* p = A.part + ((size_t)q * G + cc) * 5;
        const float v2 = __ldcg(p), l2 = __ldcg(p + 2), m2 = __ldcg(p + 3);
        const int t2 = __float_as_int(__ldcg(p + 1));
        if (v2 > bv || (v2 == bv && t2 < bt)) { bv = v2; bt = t2; bl = l2; }
        top = fmaxf(top, m2);
      }
      if (warp == 0) {
#pragma unroll
        for (int o = 16; o; o >>= 1) {
          const float v2 = __shfl_xor_sync(0xffffffffu, bv, o), l2 = __shfl_xor_sync(0xffffffffu, bl, o);
          const int t2 = __shfl_xor_sync(0xffffffffu, bt, o);
          if (v2 > bv || (v2 == bv && t2 < bt)) { bv = v2; bt = t2; bl = l2; }
          top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, o));
        }
        float tot = 0.f;
        for (int cc = lane; cc < G; cc += 32) {
          const float* p = A.part + ((size_t)q * G + cc) * 5;
          const float m2 = __ldcg(p + 3), s2 = __ldcg(p + 4);
          if (m2 > -INFINITY) tot += s2 * expf(m2 - top);
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
        if (lane < 4) {   // lane d pushes to rank d: slot [par][i][q][rank]
          float* dst = A.xp.x[lane] + (((size_t)par * MAXS + i) * MAXQ + q) * 4 * NWRD + (size_t)A.rank * NWRD;
          st_sys(dst + 0, bv);
          st_sys(dst + 1, __int_as_float(bt));
          st_sys(dst + 2, bl);
          st_sys(dst + 3, top);
          st_sys(dst + 4, tot);
        }
      }
    }
    // every CTA: wait for the 4 ranks' words of every request (each lane polls its words in one pass, volatile loads in
    // flight together), then pick the winners
    if (warp == 0) {
      const volatile unsigned* xs = (const volatile unsigned*)(xcur + (size_t)i * MAXQ * 4 * NWRD);
      const int nw = n * 20;
      bool ok;
      do {
        ok = true;
#pragma unroll
        for (int x = 0; x < (MAXQ * 20 + 31) / 32; ++x) {
          const int e = lane + 32 * x;
          if (e < nw) {
            const int q = e / 20, s2 = (e / 5) % 4, w = e % 5;
            if (xs[((size_t)q * 4 + s2) * NWRD + w] == SENT) ok = false;
          }
        }
        if (!__all_sync(0xffffffffu, ok)) __nanosleep(40); else break;
      } while (true);
      STAMP(5 + 4 * i);
      if (lane < n) {
        const int q = lane;
        float bv = -INFINITY, bl = -INFINITY, top = -INFINITY;
        int bt = 0x7fffffff;
        float vv[4], ll[4], mv[4], sv[4];
        int tt[4];
#pragma unroll
        for (int s2 = 0; s2 < 4; ++s2) {
          const volatile unsigned* p = xs + ((size_t)q * 4 + s2) * NWRD;
          vv[s2] = __int_as_float(p[0]); tt[s2] = (int)p[1]; ll[s2] = __int_as_float(p[2]);
          mv[s2] = __int_as_float(p[3]); sv[s2] = __int_as_float(p[4]);
        }
#pragma unroll
        for (int s2 = 0; s2 < 4; ++s2) {   // rank order: the lowest rank wins ties (= the lowest token id)
          if (vv[s2] > bv) { bv = vv[s2]; bt = tt[s2]; bl = ll[s2]; }
          top = fmaxf(top, mv[s2]);
        }
        float tot = 0.f;
#pragma unroll
        for (int s2 = 0; s2 < 4; ++s2) tot += mv[s2] > -INFINITY ? sv[s2] * expf(mv[s2] - top) : 0.f;
        s_tok[q] = bt;
        if (c == 0) {
          const int req = s_req[i][q];
          const int slot = req >= 0 ? req : A.sink;
          A.tokens[(size_t)q * A.tok_s0 + i] = (long long)bt;
          A.tlogit[(size_t)slot * A.vp_s0 + i] = bl;
          A.lse[(size_t)slot * A.vp_s0 + i] = top + logf(tot);
        }
      }
    }
    __syncthreads();
    if (i + 1 < k) {   // the winners' Markov embeddings for slot i + 1
      for (int e = tid; e < n * RW; e += THR) {
        const int q = e / RW;
        semb[e] = __bfloat162float(A.w1[(size_t)s_tok[q] * RW + (e % RW)]);
      }
      __syncthreads();
    }
    STAMP(6 + 4 * i);
  }
}
}  // namespace dt

static size_t smem_bytes(bool markov) {   // head only: the A operand + base logits + scales (L1 keeps the rest)
  using namespace dt;
  return (size_t)MAXR * APAD + ((size_t)MAXR * MAXV + MAXR + MAXQ * RW + NW * MAXR + MAXQ * MAXV) * 4 +
         (markov ? (size_t)MAXV * RW * 2 : 0);
}

// ---- C ABI (dtail_cabi.h; fpr_cabi.h rules): the launch of pybind dtail, called by it and by the C++ driver
#include <atomic>
#include "dtail_cabi.h"
#include "../../engine/mixed_cxx/fpr_cabi_rec.h"
FPR_CABI_RECORD_DECL(dtail)

static std::atomic<unsigned> g_attr_mask{0};   // devices whose k_dtail max-dynamic-SMEM attribute is set

static int dtail_core(const FprDtailArgs* p) {   // "dtail.dtail"
  using namespace dt;
  try {
    if (p->R != p->n * p->k || p->R > MAXR || p->n > MAXQ || p->k > MAXS || p->loc % 8 != 0 ||
        (p->loc / 8 + G - 1) / G * 8 > MAXV || (!p->markov && !p->base_out) || p->c.device < 0 || p->c.device >= 32)
      return FPR_E_ARG;
    Args a{};
    a.hd = (const __nv_bfloat16*)p->hd; a.R = p->R;
    a.wq = (const unsigned char*)p->wq; a.ws = p->ws; a.loc = p->loc; a.vstart = p->vstart;
    a.w2 = (const __nv_bfloat16*)p->w2; a.w1 = (const __nv_bfloat16*)p->w1;
    a.bias0 = (const __nv_bfloat16*)p->bias0; a.bias0_rs = p->bias0_rs;
    a.noise = p->noise; a.nz_s0 = p->nz_s0; a.nz_s1 = p->nz_s1;
    a.idx = p->idx; a.idx_s0 = p->idx_s0; a.idx_s1 = p->idx_s1;
    a.temp = p->temp;
    a.cache = (__nv_bfloat16*)p->cache; a.c_s0 = p->c_s0; a.c_s1 = p->c_s1;
    a.tokens = (long long*)p->tokens; a.tok_s0 = p->tok_s0;
    a.tlogit = p->tlogit; a.lse = p->lse; a.vp_s0 = p->vp_s0; a.sink = p->sink;
    a.base_out = (__nv_bfloat16*)p->base_out;
    a.part = p->part; a.bar = (unsigned long long*)p->bar;
    a.xloc = p->xloc;
    for (int i = 0; i < 4; ++i) a.xp.x[i] = p->xpeers[i];
    a.rank = (int)p->c.rank; a.n = p->n; a.k = p->k;
    a.dbg = (unsigned long long*)p->dbg;
    a.markov = p->markov;
    a.xq = (const unsigned char*)p->xq; a.xsg = p->xsg;
    const unsigned bit = 1u << p->c.device;
    if (!(g_attr_mask.load(std::memory_order_acquire) & bit)) {   // per device, once (idempotent if two threads race)
      if (cudaFuncSetAttribute(k_dtail, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes(true)) != cudaSuccess)
        return FPR_E_CUDA;
      g_attr_mask.fetch_or(bit, std::memory_order_acq_rel);
    }
    k_dtail<<<G, THR, smem_bytes(p->markov != 0), p->c.stream>>>(a);
    return cudaGetLastError() == cudaSuccess ? FPR_OK : FPR_E_CUDA;
  } catch (...) {
    return FPR_E_CUDA;
  }
}

void dtail(torch::Tensor hd, torch::Tensor wq, torch::Tensor ws, torch::Tensor w2, torch::Tensor w1, torch::Tensor bias0,
           torch::Tensor noise, torch::Tensor idx, torch::Tensor temp, torch::Tensor cache, torch::Tensor tokens,
           torch::Tensor tlogit, torch::Tensor lse, int64_t sink, int64_t vstart, torch::Tensor base_out, torch::Tensor part,
           torch::Tensor bar, torch::Tensor xloc, std::vector<int64_t> xpeers, int64_t rank, int64_t n, int64_t k,
           torch::Tensor dbg, bool markov, torch::Tensor xq, torch::Tensor xsg) {
  using namespace dt;
  const c10::cuda::CUDAGuard dg(hd.device());
  const int R = (int)hd.size(0);
  TORCH_CHECK(hd.scalar_type() == torch::kBFloat16 && hd.is_contiguous() && hd.size(1) == K && R == n * k && R <= MAXR,
              "dtail: hd [n * k <= 12, 5120] bf16");
  TORCH_CHECK(n <= MAXQ && k <= MAXS && xpeers.size() == 4, "dtail: n <= 4, k <= 4, 4 peers");
  TORCH_CHECK(wq.is_contiguous() && wq.size(1) == K && wq.element_size() == 1, "dtail: wq [loc, 5120] e4m3");
  const int loc = (int)wq.size(0);
  TORCH_CHECK(loc % 8 == 0 && (loc / 8 + G - 1) / G * 8 <= MAXV, "dtail: vocab slice");
  TORCH_CHECK(ws.numel() == loc && ws.scalar_type() == torch::kFloat32, "dtail: ws");
  TORCH_CHECK(w2.is_contiguous() && w2.size(0) == loc && w2.size(1) == RW && w1.is_contiguous() && w1.size(1) == RW, "dtail: w1 / w2");
  TORCH_CHECK(bias0.stride(1) == 1 && noise.stride(2) == 1 && cache.stride(2) == 1 && idx.scalar_type() == torch::kInt32, "dtail: strides");
  TORCH_CHECK(part.numel() >= (int64_t)MAXQ * G * 5 && xloc.numel() >= (int64_t)2 * MAXS * MAXQ * 4 * NWRD &&
                  bar.scalar_type() == torch::kInt64, "dtail: scratch (bar int64)");
  TORCH_CHECK(!xq.numel() || (xq.is_contiguous() && xq.size(0) >= R && xq.size(1) == K && xsg.numel() >= R), "dtail: xq / xs");
  TORCH_CHECK(markov || base_out.numel(), "dtail: head-only mode needs base_out");
  FprDtailArgs ca{};
  ca.c = FprCall{at::cuda::getCurrentCUDAStream(), (int)hd.get_device(), (int)rank};
  ca.hd = hd.data_ptr(); ca.R = R;
  ca.wq = wq.data_ptr(); ca.ws = ws.data_ptr<float>(); ca.loc = loc; ca.vstart = (int)vstart;
  ca.w2 = w2.data_ptr(); ca.w1 = w1.data_ptr();
  ca.bias0 = bias0.data_ptr(); ca.bias0_rs = bias0.stride(0);
  ca.noise = noise.data_ptr<float>(); ca.nz_s0 = noise.stride(0); ca.nz_s1 = noise.stride(1);
  ca.idx = idx.data_ptr<int>(); ca.idx_s0 = idx.stride(0); ca.idx_s1 = idx.stride(1);
  ca.temp = temp.data_ptr<float>();
  ca.cache = cache.data_ptr(); ca.c_s0 = cache.stride(0); ca.c_s1 = cache.stride(1);
  ca.tokens = tokens.data_ptr<int64_t>(); ca.tok_s0 = tokens.stride(0);
  ca.tlogit = tlogit.data_ptr<float>(); ca.lse = lse.data_ptr<float>(); ca.vp_s0 = lse.stride(0); ca.sink = (int)sink;
  ca.base_out = base_out.numel() ? base_out.data_ptr() : nullptr;
  ca.part = part.data_ptr<float>(); ca.bar = (uint64_t*)bar.data_ptr<int64_t>();
  ca.xloc = xloc.data_ptr<float>();
  for (int i = 0; i < 4; ++i) ca.xpeers[i] = (float*)xpeers[i];
  ca.n = (int)n; ca.k = (int)k;
  ca.dbg = dbg.numel() >= (int64_t)G * 16 ? (uint64_t*)dbg.data_ptr<int64_t>() : nullptr;
  ca.markov = markov ? 1 : 0;
  ca.xq = xq.numel() ? xq.data_ptr() : nullptr;
  ca.xsg = xq.numel() ? xsg.data_ptr<float>() : nullptr;
  FPR_CABI_RECORD("dtail.dtail", ca);
  TORCH_CHECK(dtail_core(&ca) == FPR_OK, "dtail: launch error");
}

extern "C" {
static int c_dtail(const void* a) { return dtail_core((const FprDtailArgs*)a); }
static const FprFnEntry k_dtail_fns[] = {{"dtail.dtail", c_dtail, nullptr, (int64_t)sizeof(FprDtailArgs)}};
static const FprExtTable k_dtail_table = {FPR_CABI_VERSION, "dtail", "", 1, k_dtail_fns};
__attribute__((visibility("default"))) const FprExtTable* fpr_dtail_table(void) { return &k_dtail_table; }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("dtail", &dtail); }
