// mHC boundary, fused streaming version. Same outputs as
// kernels/sp_layer/mhc.cu (k1a + k1b_out + k1b_mix, default flags), bit for bit, with fewer bytes per row:
//
//   mhc.cu (engine path)  k1a: y 2D + rs 8D in, xn 8D + part 16,000 out; k1b_out: xn 8D in, out 2D; k1b_mix: part 16,000
//                         = 175,552 B per row with post / comb / lag (+ the tf32 fn fragments from L2: 3.9 MB per 64-row slab)
//   here                  fa: y 2D + rs 8D in, xn 8D + in 2D + gpart 1,000 out; fb_out: in 2D in, out 2D; fb_mix: gpart
//                         = 125,076 B per row (floor of a boundary: y + rs in, xn + out = 102,400)
//
// fa (= k_fa2, one kernel per boundary): grid (80 column slices of 64 columns, 3 row splits) = 240 CTAs, 2 per SM
// (109 KB SMEM), cluster of 8 slices = one 16-chunk group of the mix reduction. 320 threads in roles:
//   - warp 8, TMA producer: per 32-row sub-tile one 2D box of y [32 x 64], one of rs viewed as [4R, D] ([128 x 64]) and
//     three 1D bulk copies (post, comb, lag rows) into a 2-stage ring; the CTA's fp32 fn slice (24.6 KB) once.
//   - warps 0-3, X (phase 1, k1a v2's arithmetic): xn = bf16(post * y + comb^T rs) per copy, stored; the collapse
//     in = bf16(sum_c lag[c] xn[c]) (k1b_out's arithmetic; lag = the previous boundary's pre) stored as bf16 (exactly
//     k1b_out's v[e]); the per-chunk sum of squares (k1a v2's butterfly); x as bf16 into a k-permuted MMA tile.
//   - warps 4-7, M: k1a's mix MMAs (3xTF32 split of fn, lo then hi per k-step, one chain per n-tile, = MHC_CHAINS 1); B
//     fragments rebuilt from the fp32 fn in SMEM (hi = rna(f), lo = rna(f - hi): prep_fn's bits).
//   - warp 9, Rd: X and M push each row's chunk partials into the row's reducer CTA (rank row / 4) with st.async
//     (byte-counted on its mbarrier, no fences); Rd folds the group's 16 chunk partials per row in chunk order (k1b's
//     `a = 0; a += part[16 g + c]`) from its own SMEM -> gpart, then frees the gather buffer to the 8 writers.
// fa1 (the first version: all 8 warps through phase 1 -> MMA -> DSMEM fold in series) is kept for A/B: 2x slower.
// fb (row kernel, k_mhc_finish's code): out = bf16(in * rsqrt(mean(in^2) + eps) * w) with k1b's summation order (thread t:
// columns t + 256 e), ptok; mixes: tot = sum over the 10 group partials in order (k1b's second level), split, Sinkhorn.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace mhcf {
constexpr int D = 5120, HC = 4, NP = 25, CW = 32, NCH = D / CW, NG = 10, GP = NG * NP;  // gpart: [R][10][25]
constexpr int SUB = 32, COLS = 64, NSL = D / COLS, CL = 8, KS = 16, NT = 3;
constexpr int NCW = 8, THR = (NCW + 1) * 32;
__host__ __device__ constexpr size_t in_off(int R) { return ((size_t)R * GP + 63) & ~(size_t)63; }  // floats; 256-B aligned
#ifndef MHCF_NST
#define MHCF_NST 2
#endif
#ifndef MHCF_NOMMA
#define MHCF_NOMMA 0  // timing ablation: skip the MMA phase
#endif
#ifndef MHCF_PROF
#define MHCF_PROF 0  // 1: warp 0 lane 0 of every CTA sums clock64 deltas per phase into g_prof[cta][8]
#endif
#if MHCF_PROF
__device__ unsigned long long g_prof[2048][8];
#define PT(i) unsigned long long _t##i = clock64()
#define PACC(k, a, b) if (tid == 0) g_prof[blockIdx.y * gridDim.x + blockIdx.x][k] += _t##b - _t##a
#define PADD(cond, k, a, b) if (lane == 0 && (cond)) g_prof[blockIdx.y * gridDim.x + blockIdx.x][k] += _t##b - _t##a
#else
#define PADD(cond, k, a, b)
#define PT(i)
#define PACC(k, a, b)
#endif
#ifndef MHCF_MMA_VOLATILE
#define MHCF_MMA_VOLATILE 0
#endif
#ifndef MHCF_ABL
#define MHCF_ABL 0  // timing ablations: 1 no MMA, 2 no cluster hand-shake / reduction, 4 no xn / in stores, 8 stream only
#endif
constexpr int NST = MHCF_NST;
constexpr int SXP = 136;  // bf16 per row of the MMA tile (k-permuted: k, k + 4 of each 8-group adjacent)
// one stage (bytes, 128-B aligned pieces)
constexpr int ST_Y = 0, ST_RS = SUB * COLS * 2, ST_POST = ST_RS + SUB * HC * COLS * 2, ST_COMB = ST_POST + SUB * 16,
              ST_LAG = ST_COMB + SUB * 64, ST_BYTES = ST_LAG + SUB * 16;
constexpr int FN_FLOATS = 2 * KS * NT * 32 * 2;  // this CTA's 2 chunks of fp32 fn in fragment order
constexpr int SM_STAGE = 0, SM_SX = NST * ST_BYTES, SM_PB = SM_SX + 2 * SUB * SXP * 2,
              SM_FN = SM_PB + 2 * 2 * SUB * NP * 4, SM_BAR = SM_FN + FN_FLOATS * 4, SM_TOTAL = SM_BAR + 128;
static_assert(ST_BYTES % 128 == 0 && SM_SX % 128 == 0 && SM_PB % 16 == 0 && SM_FN % 128 == 0, "smem alignment");

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ uint32_t s_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(s_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_expect(uint64_t* b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(s_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(s_u32(b)) : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, unsigned parity) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" ::"r"(s_u32(b)),
               "r"(parity) : "memory");
}
__device__ __forceinline__ void mb_wait_cl(uint64_t* b, unsigned parity) {  // acquire at cluster scope (remote writers)
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" ::"r"(
                   s_u32(b)),
               "r"(parity) : "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, unsigned rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ void mb_arrive_remote_relaxed(uint64_t* b, unsigned rank) {
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" ::"r"(mapa(s_u32(b), rank)) : "memory");
}
__device__ __forceinline__ void mb_arrive_remote(uint64_t* b, unsigned rank) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(mapa(s_u32(b), rank)) : "memory");
}
__device__ __forceinline__ void st_cl(uint32_t a, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(a), "f"(v) : "memory");
}
__device__ __forceinline__ void st_cl2(uint32_t a, float v0, float v1) {
  asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};" ::"r"(a), "f"(v0), "f"(v1) : "memory");
}
// async remote store into a peer CTA's SMEM; completion counted in bytes on that CTA's mbarrier (no release fence)
__device__ __forceinline__ void st_async(uint32_t a, float v, uint32_t mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(a), "f"(v), "r"(mb) : "memory");
}
__device__ __forceinline__ void st_async2(uint32_t a, float v0, float v1, uint32_t mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];" ::"r"(a), "f"(v0), "f"(v1),
               "r"(mb) : "memory");
}
__device__ __forceinline__ float ld_cl(uint32_t a) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ unsigned cl_rank() {
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(s_u32(dst)),
               "l"(src), "r"(bytes), "r"(s_u32(b)) : "memory");
}
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, int c0, int c1, uint64_t* b) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
                   s_u32(dst)),
               "l"(m), "r"(c0), "r"(c1), "r"(s_u32(b)) : "memory");
}
__device__ __forceinline__ void bar_named(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mma_tf32(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  // not volatile: a pure function of its operands; lets the scheduler hoist the next k-step's fragment
  // loads / B rebuild above this k-step's dependent lo / hi pair (the chain order through d is unchanged)
#if MHCF_MMA_VOLATILE
  asm volatile(
#else
  asm(
#endif
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned rna(unsigned b) { return (b + 0x1000u) & ~0x1FFFu; }  // prep_fn's tf32 rounding

struct Args {
  const float *post, *comb, *lag;
  const float* fnf;  // [160 ch][16 s][3 nt][32 lanes][2] fp32 (prep_f)
  __nv_bfloat16 *xn, *inb;
  float* gpart;
  int R, nsub_total, nrs;
};

// MIX: the mixes' MMAs + cluster group reduction (needs the 8-CTA cluster); !MIX: xn + in only (the head's boundary).
template <bool MIX>
__device__ __forceinline__ void fa_body(const CUtensorMap* tmy, const CUtensorMap* tmrs, const Args& A) {
  extern __shared__ __align__(1024) unsigned char sm[];
  uint64_t* bar = (uint64_t*)(sm + SM_BAR);
  uint64_t *full = bar, *empty = bar + NST, *pfull = bar + 2 * NST, *pempty = bar + 2 * NST + 2, *fnbar = bar + 2 * NST + 4;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, cp = blockIdx.x, R = A.R;
  const int s0 = (int)((long long)A.nsub_total * blockIdx.y / A.nrs), s1 = (int)((long long)A.nsub_total * (blockIdx.y + 1) / A.nrs);
  const int nsub = s1 - s0;
  const unsigned crank = MIX ? cl_rank() : 0u;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) mb_init(&full[s], 1), mb_init(&empty[s], NCW);
    for (int b = 0; b < 2; ++b) mb_init(&pfull[b], CL), mb_init(&pempty[b], CL);
    mb_init(fnbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (MIX) cluster_sync();
  else __syncthreads();
  float* fns = (float*)(sm + SM_FN);
  if (warp == NCW) {  // ---------------- producer
    if (lane == 0) {
      if (MIX) {
        mb_expect(fnbar, FN_FLOATS * 4);
        bulk_g2s(fns, A.fnf + (size_t)(2 * cp) * (FN_FLOATS / 2), FN_FLOATS * 4, fnbar);
      }
      for (int n = 0; n < nsub; ++n) {
        const int s = n % NST, r0 = (s0 + n) * SUB, nr = min(SUB, R - r0);
        if (n >= NST) mb_wait(&empty[s], ((n / NST) - 1) & 1);
        unsigned char* st = sm + SM_STAGE + s * ST_BYTES;
        mb_expect(&full[s], ST_POST + nr * 96);
        tma2d(st + ST_Y, tmy, COLS * cp, r0, &full[s]);
        tma2d(st + ST_RS, tmrs, COLS * cp, HC * r0, &full[s]);
        bulk_g2s(st + ST_POST, A.post + (size_t)r0 * HC, nr * 16, &full[s]);
        bulk_g2s(st + ST_COMB, A.comb + (size_t)r0 * 16, nr * 64, &full[s]);
        bulk_g2s(st + ST_LAG, A.lag + (size_t)r0 * HC, nr * 16, &full[s]);
      }
    }
    __syncwarp();
  } else {  // ---------------- consumers (256 threads)
    __nv_bfloat16* sx = (__nv_bfloat16*)(sm + SM_SX);   // [2 chunks][32 rows][SXP]
    float* pb = (float*)(sm + SM_PB);                   // [2 bufs][2 chunks][32 rows][25]
    const int rl = 4 * warp + (lane >> 3), cg = lane & 7, chl = cg >> 2, col0 = 8 * (cg & 3);
    const int k0 = COLS * cp + 8 * cg;
    if (MIX) mb_wait(fnbar, 0);
    for (int n = 0; n < nsub; ++n) {
      const int s = n % NST, r0 = (s0 + n) * SUB, r = r0 + rl, b = n & 1;
      const unsigned char* st = sm + SM_STAGE + s * ST_BYTES;
      PT(0);
      mb_wait(&full[s], (n / NST) & 1);
      PT(1);
      PACC(0, 0, 1);
      const uint4 yv4 = *(const uint4*)(st + ST_Y + rl * 128 + cg * 16);
      uint4 rv4[HC];
#pragma unroll
      for (int c = 0; c < HC; ++c) rv4[c] = *(const uint4*)(st + ST_RS + (rl * HC + c) * 128 + cg * 16);
      float pst[HC], cmb[16], lg[HC];
      {
        const float4 p4 = *(const float4*)(st + ST_POST + rl * 16), l4 = *(const float4*)(st + ST_LAG + rl * 16);
        pst[0] = p4.x, pst[1] = p4.y, pst[2] = p4.z, pst[3] = p4.w;
        lg[0] = l4.x, lg[1] = l4.y, lg[2] = l4.z, lg[3] = l4.w;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float4 c4 = *(const float4*)(st + ST_COMB + rl * 64 + q * 16);
          cmb[4 * q] = c4.x, cmb[4 * q + 1] = c4.y, cmb[4 * q + 2] = c4.z, cmb[4 * q + 3] = c4.w;
        }
      }
      __syncwarp();
      if (lane == 0) mb_arrive(&empty[s]);  // the stage is in registers
      if (MHCF_ABL & 8) {
        const unsigned acc = yv4.x ^ rv4[0].y ^ rv4[1].z ^ rv4[2].w ^ rv4[3].x ^ __float_as_uint(pst[1] + cmb[5] + lg[2]);
        if (acc == 0x9e3779b9u) A.gpart[tid] = 1.f;
        continue;
      }
      const __nv_bfloat16* yv = (const __nv_bfloat16*)&yv4;
      // hb = bf16(a) once (k1a's bfr); x = its fp32 value (0 for rows past R, as k1a) for ss, the collapse and the tile
      const bool rv = r < R;
      __align__(16) __nv_bfloat16 hb[HC][8];
      float x[HC][8], ss[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        ss[i] = 0.f;
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          float a = pst[j] * __bfloat162float(yv[i]);
#pragma unroll
          for (int c = 0; c < HC; ++c) a = fmaf(cmb[c * HC + j], __bfloat162float(((const __nv_bfloat16*)&rv4[c])[i]), a);
          hb[j][i] = rv ? __float2bfloat16_rn(a) : __float2bfloat16_rn(0.f);
          x[j][i] = __bfloat162float(hb[j][i]);
          ss[i] = fmaf(x[j][i], x[j][i], ss[i]);
        }
      }
      if (rv && !(MHCF_ABL & 4)) {
#pragma unroll
        for (int j = 0; j < HC; ++j) *(uint4*)(A.xn + ((size_t)r * HC + j) * D + k0) = *(const uint4*)hb[j];
        __align__(16) __nv_bfloat16 o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {  // k1b_out's collapse: a = 0; a = fmaf(lag[c], xn[c], a) for c = 0..3; bf16
          float a = 0.f;
#pragma unroll
          for (int c = 0; c < HC; ++c) a = fmaf(lg[c], x[c][i], a);
          o[i] = __float2bfloat16_rn(a);
        }
        *(uint4*)(A.inb + (size_t)r * D + k0) = *(const uint4*)o;
      }
      if (!MIX) continue;
      if (n >= 2 && !(MHCF_ABL & 2)) mb_wait_cl(&pempty[b], ((n >> 1) - 1) & 1);  // every reducer is done with this buffer's last use
      {
        __align__(16) __nv_bfloat16 o[8];
        __nv_bfloat16* sxr = sx + (chl * SUB + rl) * SXP;
#pragma unroll
        for (int j = 0; j < HC; ++j) {
#pragma unroll
          for (int m = 0; m < 4; ++m) o[2 * m] = hb[j][m], o[2 * m + 1] = hb[j][m + 4];
          *(uint4*)(sxr + j * CW + col0) = *(const uint4*)o;
        }
      }
      // butterfly over the chunk's 32 columns (k1a v2): xor 16 (lane ^ 2), xor 8 (lane ^ 1), then xor 4, 2, 1 in-thread
#pragma unroll
      for (int i = 0; i < 8; ++i) ss[i] += __shfl_xor_sync(~0u, ss[i], 2);
#pragma unroll
      for (int i = 0; i < 8; ++i) ss[i] += __shfl_xor_sync(~0u, ss[i], 1);
      {
        float t4[4], t2[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) t4[i] = ss[i] + ss[i + 4];
#pragma unroll
        for (int i = 0; i < 2; ++i) t2[i] = t4[i] + t4[i + 2];
        if ((cg & 3) == 0) pb[((b * 2 + chl) * SUB + rl) * NP] = t2[0] + t2[1];
      }
      PT(2);
      bar_named(1, NCW * 32);  // MMA tile complete
      PT(3);
      PACC(1, 1, 2);
      PACC(2, 2, 3);
      if (!MHCF_NOMMA && !(MHCF_ABL & 1)) {
        // warp -> (chunk, m-tile) pair p = warp >> 1, n-tiles {0, 1} (even warp) or {2} (odd warp)
        const int p = warp >> 1, pc = p >> 1, rb = 16 * (p & 1), g = lane >> 2, t = lane & 3, nb = (warp & 1) ? 2 : 0;
        const int nn = (warp & 1) ? 1 : 2;
        const __nv_bfloat16* sxc = sx + pc * SUB * SXP;
        const float2* fc = (const float2*)(fns + pc * (FN_FLOATS / 2));
        float d[2][4];
#pragma unroll
        for (int q = 0; q < 2; ++q) d[q][0] = d[q][1] = d[q][2] = d[q][3] = 0.f;
#pragma unroll
        for (int s2 = 0; s2 < KS; ++s2) {
          const int kk = 8 * s2;
          const unsigned w0 = *(const unsigned*)(sxc + (rb + g) * SXP + kk + 2 * t);
          const unsigned w1 = *(const unsigned*)(sxc + (rb + g + 8) * SXP + kk + 2 * t);
          const unsigned a[4] = {w0 << 16, w1 << 16, w0 & 0xFFFF0000u, w1 & 0xFFFF0000u};
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            if (q < nn) {
              const float2 f = fc[(s2 * NT + nb + q) * 32 + lane];
              const unsigned hx = rna(__float_as_uint(f.x)), hy = rna(__float_as_uint(f.y));
              const unsigned lx = rna(__float_as_uint(__fsub_rn(f.x, __uint_as_float(hx))));
              const unsigned ly = rna(__float_as_uint(__fsub_rn(f.y, __uint_as_float(hy))));
              mma_tf32(d[q], a, lx, ly);  // small part first (k1a)
              mma_tf32(d[q], a, hx, hy);
            }
          }
        }
#pragma unroll
        for (int q = 0; q < 2; ++q)
          if (q < nn)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              float* pp = pb + ((b * 2 + pc) * SUB + rb + g + 8 * h) * NP + 1 + 8 * (nb + q) + 2 * t;
              pp[0] = d[q][2 * h];
              pp[1] = d[q][2 * h + 1];
            }
      }
      PT(4);
      bar_named(1, NCW * 32);  // this CTA's partials of sub-tile n complete
      PT(5);
      PACC(3, 3, 4);
      PACC(4, 4, 5);
      if (MHCF_ABL & 2) continue;
      if (tid == 0) {
        asm volatile("fence.acq_rel.cluster;" ::: "memory");
#pragma unroll
        for (int q = 0; q < CL; ++q) mb_arrive_remote(&pfull[b], q);
      }
      if (tid < 4 * NP) {  // reducer: rows 4 crank .. + 3 of the sub-tile, one (row, q) per thread
        const int ro = 4 * (int)crank + tid / NP, q = tid % NP;
        mb_wait_cl(&pfull[b], (n >> 1) & 1);
        const uint32_t base = s_u32(pb + ((b * 2) * SUB + ro) * NP + q);
        float v[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) v[c] = ld_cl(mapa(base + (c & 1) * SUB * NP * 4, c >> 1));
        float a = 0.f;
#pragma unroll
        for (int c = 0; c < 16; ++c) a += v[c];
        if (r0 + ro < R) A.gpart[(size_t)(r0 + ro) * GP + (cp / CL) * NP + q] = a;
      }
      if (warp < 4) {
        bar_named(2, 128);
        PT(6);
        PACC(5, 5, 6);
        PACC(6, 0, 6);
        if (tid == 0)
#pragma unroll
          for (int q = 0; q < CL; ++q) mb_arrive_remote(&pempty[b], q);
      }
    }
  }
  if (MIX) cluster_sync();  // no CTA leaves while a peer may still read its partials or signal its barriers
}

__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(THR, 2)
    k_fa_mix(const __grid_constant__ CUtensorMap tmy, const __grid_constant__ CUtensorMap tmrs, const Args A) {
  fa_body<true>(&tmy, &tmrs, A);
}
__global__ void __launch_bounds__(THR, 2)
    k_fa_x(const __grid_constant__ CUtensorMap tmy, const __grid_constant__ CUtensorMap tmrs, const Args A) {
  fa_body<false>(&tmy, &tmrs, A);
}

__device__ __forceinline__ float sigm(float z) { return 1.f / (1.f + expf(-z)); }
// k_mhc_finish (mhc.cu) with the collapse taken from `in` (fa) and the first level of the mix reduction from gpart (fa).
// MODE 0: both, 1: out (+ ptok) only, 2: mixes only.
template <int MODE>
__global__ void __launch_bounds__(256) k_fb(const float* __restrict__ gpart, const __nv_bfloat16* __restrict__ inb,
                                            const float* __restrict__ scale, const float* __restrict__ base,
                                            const float* __restrict__ normw, int R, float norm_eps, float hc_eps, int iters,
                                            float* __restrict__ pre_o, float* __restrict__ post_o,
                                            float* __restrict__ comb_o, __nv_bfloat16* __restrict__ out,
                                            int* __restrict__ ptok_o) {
  constexpr bool DO_OUT = MODE != 2, DO_MIX = MODE != 1;
  constexpr int THRB = 256, PER = D / THRB;
  const int r = blockIdx.x, t = threadIdx.x;
  __shared__ float tot[NP];
  __shared__ float wsum[THRB / 32];
  float v[PER];
  float ss = 0.f;
#pragma unroll
  for (int e = 0; e < (DO_OUT ? PER : 0); ++e) {
    v[e] = __bfloat162float(inb[(size_t)r * D + e * THRB + t]);
    ss = fmaf(v[e], v[e], ss);
  }
  if (DO_MIX && t < NP) {
    float pv[NG];
#pragma unroll
    for (int gq = 0; gq < NG; ++gq) pv[gq] = gpart[(size_t)r * GP + gq * NP + t];
    float a = 0.f;
#pragma unroll
    for (int gq = 0; gq < NG; ++gq) a += pv[gq];
    tot[t] = a;
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) ss += __shfl_xor_sync(~0u, ss, off);
  if ((t & 31) == 0) wsum[t >> 5] = ss;
  __syncthreads();
  if (DO_MIX && t < 32) {
    const float rsq = rsqrtf(tot[0] / (float)(HC * D) + norm_eps);
    const int l = t & 15, i = l >> 2, j = l & 3;
    if (t < HC) {
      pre_o[r * HC + t] = sigm(tot[1 + t] * rsq * scale[0] + base[t]) + hc_eps;
      post_o[r * HC + t] = 2.f * sigm(tot[1 + HC + t] * rsq * scale[1] + base[HC + t]);
    }
    const float lg = tot[1 + 2 * HC + l] * rsq * scale[2] + base[2 * HC + l];
    float mx = fmaxf(lg, __shfl_xor_sync(~0u, lg, 1));
    mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, 2));
    const float ex = expf(lg - mx);
    auto rowsum = [&](float v) {
      const int b = t & ~3;
      return ((__shfl_sync(~0u, v, b) + __shfl_sync(~0u, v, b + 1)) + __shfl_sync(~0u, v, b + 2)) + __shfl_sync(~0u, v, b + 3);
    };
    auto colsum = [&](float v) {
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
  for (int i = 0; i < THRB / 32; ++i) s += wsum[i];
  const float rr = rsqrtf(s / (float)D + norm_eps);
  float mx = 0.f;
#pragma unroll
  for (int e = 0; e < PER; ++e) {
    const int k = e * THRB + t;
    const __nv_bfloat16 ob = __float2bfloat16_rn((v[e] * rr) * normw[k]);
    out[(size_t)r * D + k] = ob;
    mx = fmaxf(mx, fabsf(__bfloat162float(ob)));
  }
  if (ptok_o) {
#pragma unroll
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, off));
    __shared__ float wmx[THRB / 32];
    if ((t & 31) == 0) wmx[t >> 5] = mx;
    __syncthreads();
    if (t == 0) {
      float m = wmx[0];
#pragma unroll
      for (int i = 1; i < THRB / 32; ++i) m = fmaxf(m, wmx[i]);
      int p = 0;
      if (m > 0.f) {
        int ex;
        frexpf(m, &ex);
        p = 14 - ex;
      }
      ptok_o[r] = p;
    }
  }
}

// fb_out with 4 columns per thread (8-B accesses, 64 threads per row): thread p plays k_fb's virtual threads 4p .. 4p + 3
// (virtual thread t sums columns t + 256 e in order). k_fb's warp butterfly (xor 16, 8, 4, 2, 1 over a virtual warp of 32)
// becomes lane xor 4, 2, 1 inside each 8-lane group (virtual xor 16, 8, 4), then virtual xor 2 and xor 1 in-thread; the
// 8 virtual warp sums are added in order. Same bits as k_fb<1>.
__global__ void __launch_bounds__(64) k_fbo4(const __nv_bfloat16* __restrict__ inb, const float* __restrict__ normw, int R,
                                             float norm_eps, __nv_bfloat16* __restrict__ out, int* __restrict__ ptok_o) {
  constexpr int PER = D / 256;
  const int r = blockIdx.x, p = threadIdx.x;
  __shared__ float wsum[8], wmx[2];
  float v[PER][4], s4[4] = {0.f, 0.f, 0.f, 0.f};
  const __nv_bfloat16* src = inb + (size_t)r * D + 4 * p;
  uint2 w[PER];
#pragma unroll
  for (int e = 0; e < PER; ++e) w[e] = *(const uint2*)(src + e * 256);
#pragma unroll
  for (int e = 0; e < PER; ++e) {
    const __nv_bfloat16* h = (const __nv_bfloat16*)&w[e];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      v[e][q] = __bfloat162float(h[q]);
      s4[q] = fmaf(v[e][q], v[e][q], s4[q]);
    }
  }
#pragma unroll
  for (int off = 4; off; off >>= 1)
#pragma unroll
    for (int q = 0; q < 4; ++q) s4[q] += __shfl_xor_sync(~0u, s4[q], off);
  const float t0 = s4[0] + s4[2], t1 = s4[1] + s4[3];
  if ((p & 7) == 0) wsum[p >> 3] = t0 + t1;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += wsum[i];
  const float rr = rsqrtf(s / (float)D + norm_eps);
  float mx = 0.f;
  __nv_bfloat16* dst = out + (size_t)r * D + 4 * p;
#pragma unroll
  for (int e = 0; e < PER; ++e) {
    const float4 nw = *(const float4*)(normw + e * 256 + 4 * p);
    const float nq[4] = {nw.x, nw.y, nw.z, nw.w};
    __align__(8) __nv_bfloat16 o[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      o[q] = __float2bfloat16_rn((v[e][q] * rr) * nq[q]);
      mx = fmaxf(mx, fabsf(__bfloat162float(o[q])));
    }
    *(uint2*)(dst + e * 256) = *(const uint2*)o;
  }
  if (ptok_o) {
#pragma unroll
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, off));
    if ((p & 31) == 0) wmx[p >> 5] = mx;
    __syncthreads();
    if (p == 0) {
      const float m = fmaxf(wmx[0], wmx[1]);
      int pe = 0;
      if (m > 0.f) {
        int ex;
        frexpf(m, &ex);
        pe = 14 - ex;
      }
      ptok_o[r] = pe;
    }
  }
}
// ------------------------------------------------------------------------------------------------------------- frow
// frow: the small-R path (decode buckets, the tail bucket, the head). One CTA per row, 256 threads:
// two TMA bulk copies bring the row's y (10 KB) and rs (4 copies, 40 KB) into SMEM; phase 1 (k1a's arithmetic) writes
// xn in place over rs; the collapse + RMSNorm + out (+ ptok) in-CTA with k1b_out's exact thread map and summation order
// (thread t: columns t + 256 e, e = 0..19; warp butterfly; 8 warp sums in order); xn and out leave by bulk stores.
// The mixes are not computed here: the caller runs mhc.cu's k1a_mix (from xn) + k1b_mix on the side stream (mhc.cu's
// MODE split: the same pre / post / comb bits).
__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, unsigned bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(s_u32(src)), "r"(bytes) : "memory");
}
constexpr int FR_SMEM = 5 * D * 2 + 64;
__global__ void __launch_bounds__(256) k_frow(const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ rs,
                                              const float* __restrict__ post, const float* __restrict__ comb,
                                              const float* __restrict__ lag, const float* __restrict__ normw, int R,
                                              float norm_eps, __nv_bfloat16* __restrict__ xn, __nv_bfloat16* __restrict__ out,
                                              int* __restrict__ ptok_o) {
  constexpr int THRB = 256, PER = D / THRB;
  extern __shared__ __align__(128) unsigned char sm[];
  __nv_bfloat16* srs = (__nv_bfloat16*)sm;  // [4][D]: rs, then xn in place
  __nv_bfloat16* sy = srs + HC * D;         // [D]: y, then out in place
  uint64_t* mb = (uint64_t*)(sm + 5 * D * 2);
  __shared__ float wsum[THRB / 32], wmx[THRB / 32];
  const int r = blockIdx.x, t = threadIdx.x;
  if (t == 0) {
    mb_init(mb, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (t == 0) {
    mb_expect(mb, 5 * D * 2);
    bulk_g2s(srs, rs + (size_t)r * HC * D, HC * D * 2, mb);
    bulk_g2s(sy, y + (size_t)r * D, D * 2, mb);
  }
  float pst[HC], cmb[16], lg[HC];
  {
    const float4 p4 = *(const float4*)(post + (size_t)r * HC), l4 = *(const float4*)(lag + (size_t)r * HC);
    pst[0] = p4.x, pst[1] = p4.y, pst[2] = p4.z, pst[3] = p4.w;
    lg[0] = l4.x, lg[1] = l4.y, lg[2] = l4.z, lg[3] = l4.w;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float4 c4 = *(const float4*)(comb + (size_t)r * 16 + 4 * q);
      cmb[4 * q] = c4.x, cmb[4 * q + 1] = c4.y, cmb[4 * q + 2] = c4.z, cmb[4 * q + 3] = c4.w;
    }
  }
  mb_wait(mb, 0);
  float v[PER], ss = 0.f;
#pragma unroll
  for (int e = 0; e < PER; ++e) {
    const int k = e * THRB + t;
    const float yv = __bfloat162float(sy[k]);
    float rv[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) rv[c] = __bfloat162float(srs[c * D + k]);
    float x[HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      float a = pst[j] * yv;
#pragma unroll
      for (int c = 0; c < HC; ++c) a = fmaf(cmb[c * HC + j], rv[c], a);
      const __nv_bfloat16 hb = __float2bfloat16_rn(a);
      x[j] = __bfloat162float(hb);
      srs[j * D + k] = hb;  // xn in place (this thread read rs[0..3][k] above)
    }
    float ai = 0.f;  // k1b_out's collapse
#pragma unroll
    for (int c = 0; c < HC; ++c) ai = fmaf(lg[c], x[c], ai);
    v[e] = bfr(ai);
    ss = fmaf(v[e], v[e], ss);
  }
  __syncthreads();  // xn row complete in SMEM
  if (t == 0) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    bulk_s2g(xn + (size_t)r * HC * D, srs, HC * D * 2);
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) ss += __shfl_xor_sync(~0u, ss, off);
  if ((t & 31) == 0) wsum[t >> 5] = ss;
  __syncthreads();
  float s2 = 0.f;
#pragma unroll
  for (int i = 0; i < THRB / 32; ++i) s2 += wsum[i];
  const float rr = rsqrtf(s2 / (float)D + norm_eps);
  float mx = 0.f;
#pragma unroll
  for (int e = 0; e < PER; ++e) {
    const int k = e * THRB + t;
    const __nv_bfloat16 ob = __float2bfloat16_rn((v[e] * rr) * normw[k]);
    sy[k] = ob;  // out in place over y (y is no longer read)
    mx = fmaxf(mx, fabsf(__bfloat162float(ob)));
  }
  __syncthreads();  // out row complete in SMEM
  if (t == 0) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    bulk_s2g(out + (size_t)r * D, sy, D * 2);
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  }
  if (ptok_o) {  // k1b_out's MoE prescale exponent
#pragma unroll
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, off));
    if ((t & 31) == 0) wmx[t >> 5] = mx;
    __syncthreads();
    if (t == 0) {
      float m = wmx[0];
#pragma unroll
      for (int i = 1; i < THRB / 32; ++i) m = fmaxf(m, wmx[i]);
      int p = 0;
      if (m > 0.f) {
        int ex;
        frexpf(m, &ex);
        p = 14 - ex;
      }
      ptok_o[r] = p;
    }
  }
  if (t == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");  // SMEM stays valid until both stores read it
}

// fb_mix, one warp per row (8 rows per CTA): k_fb<2>'s per-row code (the 10-group sum by lanes t < 25 in order, split,
// Sinkhorn on the warp) with tot[] passed by shuffles instead of SMEM; k_fb<2> ran 1 row per 256-thread CTA (1 busy warp)
__global__ void __launch_bounds__(256) k_fbm8(const float* __restrict__ gpart, const float* __restrict__ scale,
                                              const float* __restrict__ base, int R, float norm_eps, float hc_eps, int iters,
                                              float* __restrict__ pre_o, float* __restrict__ post_o, float* __restrict__ comb_o) {
  const int t = threadIdx.x & 31, r = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (r >= R) return;
  float tv = 0.f;
  if (t < NP) {
    float pv[NG];
#pragma unroll
    for (int gq = 0; gq < NG; ++gq) pv[gq] = gpart[(size_t)r * GP + gq * NP + t];
    float a = 0.f;
#pragma unroll
    for (int gq = 0; gq < NG; ++gq) a += pv[gq];
    tv = a;
  }
  const float rsq = rsqrtf(__shfl_sync(~0u, tv, 0) / (float)(HC * D) + norm_eps);
  const int l = t & 15, i = l >> 2, j = l & 3;
  const float t_pre = __shfl_sync(~0u, tv, 1 + (t & 3)), t_post = __shfl_sync(~0u, tv, 1 + HC + (t & 3));
  const float t_c = __shfl_sync(~0u, tv, 1 + 2 * HC + l);
  if (t < HC) {
    pre_o[r * HC + t] = sigm(t_pre * rsq * scale[0] + base[t]) + hc_eps;
    post_o[r * HC + t] = 2.f * sigm(t_post * rsq * scale[1] + base[HC + t]);
  }
  const float lg = t_c * rsq * scale[2] + base[2 * HC + l];
  float mx = fmaxf(lg, __shfl_xor_sync(~0u, lg, 1));
  mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, 2));
  const float ex = expf(lg - mx);
  auto rowsum = [&](float v) {
    const int b = t & ~3;
    return ((__shfl_sync(~0u, v, b) + __shfl_sync(~0u, v, b + 1)) + __shfl_sync(~0u, v, b + 2)) + __shfl_sync(~0u, v, b + 3);
  };
  auto colsum = [&](float v) {
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

// fb_out, frow-style: the row's in (10 KB) by one bulk copy into SMEM, k_fb<1>'s arithmetic and order (thread t: columns
// t + 256 e), out written to SMEM in place and stored by one bulk copy
constexpr int FBO_SMEM = D * 2 + 64;
__global__ void __launch_bounds__(256) k_fbo_row(const __nv_bfloat16* __restrict__ inb, const float* __restrict__ normw,
                                                 int R, float norm_eps, __nv_bfloat16* __restrict__ out,
                                                 int* __restrict__ ptok_o) {
  constexpr int THRB = 256, PER = D / THRB;
  extern __shared__ __align__(128) unsigned char sm[];
  __nv_bfloat16* si = (__nv_bfloat16*)sm;
  uint64_t* mb = (uint64_t*)(sm + D * 2);
  __shared__ float wsum[THRB / 32], wmx[THRB / 32];
  const int r = blockIdx.x, t = threadIdx.x;
  if (t == 0) {
    mb_init(mb, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (t == 0) {
    mb_expect(mb, D * 2);
    bulk_g2s(si, inb + (size_t)r * D, D * 2, mb);
  }
  mb_wait(mb, 0);
  float v[PER], ss = 0.f;
#pragma unroll
  for (int e = 0; e < PER; ++e) {
    v[e] = __bfloat162float(si[e * THRB + t]);
    ss = fmaf(v[e], v[e], ss);
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) ss += __shfl_xor_sync(~0u, ss, off);
  if ((t & 31) == 0) wsum[t >> 5] = ss;
  __syncthreads();
  float s2 = 0.f;
#pragma unroll
  for (int i = 0; i < THRB / 32; ++i) s2 += wsum[i];
  const float rr = rsqrtf(s2 / (float)D + norm_eps);
  float mx = 0.f;
#pragma unroll
  for (int e = 0; e < PER; ++e) {
    const int k = e * THRB + t;
    const __nv_bfloat16 ob = __float2bfloat16_rn((v[e] * rr) * normw[k]);
    si[k] = ob;
    mx = fmaxf(mx, fabsf(__bfloat162float(ob)));
  }
  __syncthreads();
  if (t == 0) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    bulk_s2g(out + (size_t)r * D, si, D * 2);
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  }
  if (ptok_o) {
#pragma unroll
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, off));
    if ((t & 31) == 0) wmx[t >> 5] = mx;
    __syncthreads();
    if (t == 0) {
      float m = wmx[0];
#pragma unroll
      for (int i = 1; i < THRB / 32; ++i) m = fmaxf(m, wmx[i]);
      int p = 0;
      if (m > 0.f) {
        int ex;
        frexpf(m, &ex);
        p = 14 - ex;
      }
      ptok_o[r] = p;
    }
  }
  if (t == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
}  // namespace mhcf

// ------------------------------------------------------------------------------------------------------------------ host
static PFN_cuTensorMapEncodeTiled_v12000 g_encode = nullptr;
static CUtensorMap tmap(const void* p, int rows, int box_rows) {
  if (!g_encode) {
    cudaDriverEntryPointQueryResult q;
    void* fn = nullptr;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q) == cudaSuccess && fn,
                "mhcf: cuTensorMapEncodeTiled");
    g_encode = (PFN_cuTensorMapEncodeTiled_v12000)fn;
  }
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)mhcf::D, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)mhcf::D * 2};
  cuuint32_t box[2] = {(cuuint32_t)mhcf::COLS, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1, 1};
  CUresult r = g_encode(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), dims, strides, box, es,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "mhcf: tensor map encode failed ", (int)r);
  return m;
}
// cudaFuncSetAttribute is per device: one bit per (kernel, device). A process-wide flag would leave GPUs 1-3
// without the 109 KB dynamic-SMEM opt-in -> "invalid argument" at the first launch there.
static unsigned long long g_attr_done[7] = {0, 0, 0, 0, 0, 0, 0};
template <typename K>
static void smem_attr(K kern, int bytes, int which) {
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  if (!(g_attr_done[which] >> dev & 1ull)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    g_attr_done[which] |= 1ull << dev;
  }
}
static int g_nrs = 0;
static void same_dev(const torch::Tensor& a, std::initializer_list<torch::Tensor> ts) {
  for (const auto& t : ts) TORCH_CHECK(t.is_cuda() && t.device() == a.device(), "mhcf: all tensors on one device");
}  // 0: auto (3 row splits = 240 CTAs on 132 SMs x 2)
static void check_rows(const torch::Tensor& t, int R, int inner, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.size(0) >= R && t.stride(-1) == 1 && t.numel() / t.size(0) == inner &&
                  t.stride(0) == inner && ((uintptr_t)t.data_ptr() % 16) == 0,
              "mhcf: ", what, " must be [R, ...] rows-contiguous, 16-B aligned");
}
// scratch: >= R * (250 + 2560) fp32 (gpart [R][250] fp32, then in [R][D] bf16); the engine's part_mhc (160 R 25) fits.
static void launch_fa(bool mix, torch::Tensor y, torch::Tensor rs, torch::Tensor post, torch::Tensor comb, torch::Tensor lag,
                      const float* fnf, torch::Tensor xn, torch::Tensor scratch, int64_t nrs) {
  using namespace mhcf;
  const c10::cuda::CUDAGuard dg(y.device());
  same_dev(y, {rs, post, comb, lag, xn, scratch});
  const int R = (int)y.size(0);
  TORCH_CHECK(R >= 1 && rs.size(0) == R && xn.size(0) >= R, "mhcf: rows");
  check_rows(y, R, D, "y");
  check_rows(rs, R, HC * D, "rs");
  check_rows(xn, R, HC * D, "xn");
  check_rows(post, R, HC, "post");
  check_rows(comb, R, 16, "comb");
  check_rows(lag, R, HC, "lag");
  TORCH_CHECK(y.scalar_type() == torch::kBFloat16 && rs.scalar_type() == torch::kBFloat16 && post.scalar_type() == torch::kFloat32 &&
                  comb.scalar_type() == torch::kFloat32 && lag.scalar_type() == torch::kFloat32,
              "mhcf: dtypes");
  TORCH_CHECK(scratch.scalar_type() == torch::kFloat32 && scratch.numel() >= (int64_t)(in_off(R) + (size_t)R * D / 2), "mhcf: scratch");
  smem_attr(k_fa_mix, SM_TOTAL, 0);
  smem_attr(k_fa_x, SM_TOTAL, 1);
  const int ns = (R + SUB - 1) / SUB;
  int k = (int)(nrs > 0 ? nrs : (g_nrs > 0 ? g_nrs : 3));
  k = std::max(1, std::min(k, ns));
  Args a{post.data_ptr<float>(), comb.data_ptr<float>(), lag.data_ptr<float>(), fnf, (__nv_bfloat16*)xn.data_ptr(),
         (__nv_bfloat16*)(scratch.data_ptr<float>() + in_off(R)), scratch.data_ptr<float>(), R, ns, k};
  const CUtensorMap my = tmap(y.data_ptr(), R, SUB), mr = tmap(rs.data_ptr(), HC * R, HC * SUB);
  auto st = at::cuda::getCurrentCUDAStream();
  if (mix) k_fa_mix<<<dim3(NSL, k), THR, SM_TOTAL, st>>>(my, mr, a);
  else k_fa_x<<<dim3(NSL, k), THR, SM_TOTAL, st>>>(my, mr, a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void fa(torch::Tensor y, torch::Tensor rs, torch::Tensor post, torch::Tensor comb, torch::Tensor lag, torch::Tensor fnf,
        torch::Tensor xn, torch::Tensor scratch, int64_t nrs) {
  TORCH_CHECK(fnf.scalar_type() == torch::kFloat32 && fnf.numel() == (int64_t)mhcf::NCH * mhcf::FN_FLOATS / 2, "mhcf: fnf (prep_f)");
  launch_fa(true, y, rs, post, comb, lag, fnf.data_ptr<float>(), xn, scratch, nrs);
}
void fa_x(torch::Tensor y, torch::Tensor rs, torch::Tensor post, torch::Tensor comb, torch::Tensor lag, torch::Tensor xn,
          torch::Tensor scratch, int64_t nrs) {
  launch_fa(false, y, rs, post, comb, lag, nullptr, xn, scratch, nrs);
}
// ---------------------------------------------------------------------------------------------------------------- fa2
// fa2: fa with the work split by role, so the MMA chains and the cluster hand-shake leave the phase-1
// critical path (fa's profile: per 32-row sub-tile 1.9 K cycles phase 1, 3.5 K MMA, 6.5 K cluster, all in series).
//   warps 0-3 X: phase 1 (2 passes of 16 rows), xn / in stores, the MMA tile (single buffer) and the ss partial
//   warps 4-7 M: one (chunk, m-tile) pair each, all 3 n-tiles; the 24 partials -> pb[n % NPB]
//   warp 8: TMA producer;  warp 9 Rd: signals pb ready to the cluster, folds sub-tile n - 1's group sums (DSMEM) while
//   the others run sub-tile n, writes gpart, frees the buffers.
// The arithmetic of every value is fa's (k1a v2's), so the outputs are the same bits.
namespace mhcf {
constexpr int NX = 4, NM = 4, W_PROD = 8, W_RD = 9, THR2 = 320, NPB = 3;
constexpr int GQ = 26;  // gather row: 26 floats, q at q + 1 (the q = 1 + 8 n + 2 t pairs of the MMA land 8-B aligned)
constexpr int PBB = CL * 4 * 2 * GQ * 4;  // one gather buffer: 8 source CTAs x 4 rows x 2 chunks x 26 fp32 (6.5 KB)
__device__ __forceinline__ int gb_idx(int b, int src, int lrow, int chl, int q) { return (((b * CL + src) * 4 + lrow) * 2 + chl) * GQ + q + 1; }
constexpr int S2_STAGE = 0, S2_SX = NST * ST_BYTES, S2_PB = S2_SX + 2 * SUB * SXP * 2, S2_FN = S2_PB + NPB * PBB,
              S2_BAR = S2_FN + FN_FLOATS * 4, S2_TOTAL = S2_BAR + 256;
static_assert(S2_FN % 128 == 0, "fn smem alignment");

__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(THR2, 2)
    k_fa2(const __grid_constant__ CUtensorMap tmy, const __grid_constant__ CUtensorMap tmrs, const Args A) {
  extern __shared__ __align__(1024) unsigned char sm[];
  uint64_t* bar = (uint64_t*)(sm + S2_BAR);
  uint64_t *full = bar, *empty = bar + NST, *sxf = bar + 2 * NST, *sxe = sxf + 1, *ploc = sxe + 1, *pfull = ploc + NPB,
           *pempty = pfull + NPB, *fnbar = pempty + NPB;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, cp = blockIdx.x, R = A.R;
  const int s0 = (int)((long long)A.nsub_total * blockIdx.y / A.nrs), s1 = (int)((long long)A.nsub_total * (blockIdx.y + 1) / A.nrs);
  const int nsub = s1 - s0;
  const unsigned crank = cl_rank();
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) mb_init(&full[s], 1), mb_init(&empty[s], NX);
    mb_init(sxf, NX), mb_init(sxe, NM);
    for (int b = 0; b < NPB; ++b) mb_init(&ploc[b], NM), mb_init(&pfull[b], 1), mb_init(&pempty[b], CL);
    mb_init(fnbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster_sync();
  float* fns = (float*)(sm + S2_FN);
  __nv_bfloat16* sx = (__nv_bfloat16*)(sm + S2_SX);  // [2 chunks][32 rows][SXP]
  float* gbs = (float*)(sm + S2_PB);                 // gather: [NPB][8 src][4 rows][2 chunks][25]; rows 4 crank .. + 3
  if (warp == W_PROD) {  // ---------------- producer
    if (lane == 0) {
      mb_expect(fnbar, FN_FLOATS * 4);
      bulk_g2s(fns, A.fnf + (size_t)(2 * cp) * (FN_FLOATS / 2), FN_FLOATS * 4, fnbar);
      for (int n = 0; n < nsub; ++n) {
        const int s = n % NST, r0 = (s0 + n) * SUB, nr = min(SUB, R - r0);
        if (n >= NST) mb_wait(&empty[s], ((n / NST) - 1) & 1);
        unsigned char* st = sm + S2_STAGE + s * ST_BYTES;
        mb_expect(&full[s], ST_POST + nr * 96);
        tma2d(st + ST_Y, &tmy, COLS * cp, r0, &full[s]);
        tma2d(st + ST_RS, &tmrs, COLS * cp, HC * r0, &full[s]);
        bulk_g2s(st + ST_POST, A.post + (size_t)r0 * HC, nr * 16, &full[s]);
        bulk_g2s(st + ST_COMB, A.comb + (size_t)r0 * 16, nr * 64, &full[s]);
        bulk_g2s(st + ST_LAG, A.lag + (size_t)r0 * HC, nr * 16, &full[s]);
      }
    }
    __syncwarp();
  } else if (warp < NX) {  // ---------------- X: phase 1
    const int cg = lane & 7, chl = cg >> 2, col0 = 8 * (cg & 3), k0 = COLS * cp + 8 * cg;
    for (int n = 0; n < nsub; ++n) {
      const int s = n % NST, r0 = (s0 + n) * SUB, b = n % NPB;
      const unsigned char* st = sm + S2_STAGE + s * ST_BYTES;
      PT(0);
      mb_wait(&full[s], (n / NST) & 1);
      PT(1);
      PACC(0, 0, 1);
      uint4 tw[2][HC];  // both passes' k-permuted MMA tile rows (bf16), written after M frees the tile
      float tot[2];
#pragma unroll
      for (int pp = 0; pp < 2; ++pp) {
        const int rl = 16 * pp + 4 * warp + (lane >> 3), r = r0 + rl;
        const bool rv = r < R;
        const uint4 yv4 = *(const uint4*)(st + ST_Y + rl * 128 + cg * 16);
        uint4 rv4[HC];
#pragma unroll
        for (int c = 0; c < HC; ++c) rv4[c] = *(const uint4*)(st + ST_RS + (rl * HC + c) * 128 + cg * 16);
        float pst[HC], cmb[16], lg[HC];
        {  // rows past R: y / rs are TMA zero fill; zero the (stale) post / comb / lag too, so every x is +0 (k1a's 0.f)
          const float4 p4 = *(const float4*)(st + ST_POST + rl * 16), l4 = *(const float4*)(st + ST_LAG + rl * 16);
          pst[0] = rv ? p4.x : 0.f, pst[1] = rv ? p4.y : 0.f, pst[2] = rv ? p4.z : 0.f, pst[3] = rv ? p4.w : 0.f;
          lg[0] = l4.x, lg[1] = l4.y, lg[2] = l4.z, lg[3] = l4.w;
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float4 c4 = *(const float4*)(st + ST_COMB + rl * 64 + q * 16);
            cmb[4 * q] = rv ? c4.x : 0.f, cmb[4 * q + 1] = rv ? c4.y : 0.f, cmb[4 * q + 2] = rv ? c4.z : 0.f,
                    cmb[4 * q + 3] = rv ? c4.w : 0.f;
          }
        }
        if (pp == 1) {
          __syncwarp();
          if (lane == 0) mb_arrive(&empty[s]);  // both passes of the stage are in registers
        }
        const __nv_bfloat16* yv = (const __nv_bfloat16*)&yv4;
        __align__(16) __nv_bfloat162 hb2[HC][4];  // bf16(a) of columns (2 m, 2 m + 1)
        float x[HC][8], ss[8];
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          float a[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            a[i] = pst[j] * __bfloat162float(yv[i]);
#pragma unroll
            for (int c = 0; c < HC; ++c) a[i] = fmaf(cmb[c * HC + j], __bfloat162float(((const __nv_bfloat16*)&rv4[c])[i]), a[i]);
          }
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            hb2[j][m] = __floats2bfloat162_rn(a[2 * m], a[2 * m + 1]);
            const float2 f2 = __bfloat1622float2(hb2[j][m]);
            x[j][2 * m] = f2.x, x[j][2 * m + 1] = f2.y;
          }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          ss[i] = 0.f;
#pragma unroll
          for (int j = 0; j < HC; ++j) ss[i] = fmaf(x[j][i], x[j][i], ss[i]);
        }
        if (rv) {
#pragma unroll
          for (int j = 0; j < HC; ++j) *(uint4*)(A.xn + ((size_t)r * HC + j) * D + k0) = *(const uint4*)hb2[j];
          __align__(16) __nv_bfloat162 o2[4];
#pragma unroll
          for (int m = 0; m < 4; ++m) {  // k1b_out's collapse: a = 0; a = fmaf(lag[c], xn[c], a) for c = 0..3; bf16
            float a0 = 0.f, a1 = 0.f;
#pragma unroll
            for (int c = 0; c < HC; ++c) a0 = fmaf(lg[c], x[c][2 * m], a0), a1 = fmaf(lg[c], x[c][2 * m + 1], a1);
            o2[m] = __floats2bfloat162_rn(a0, a1);
          }
          *(uint4*)(A.inb + (size_t)r * D + k0) = *(const uint4*)o2;
        }
#pragma unroll
        for (int j = 0; j < HC; ++j) {  // tile word m = (col m, col m + 4) of the 8-group
          const unsigned* h = (const unsigned*)hb2[j];
          tw[pp][j].x = __byte_perm(h[0], h[2], 0x5410);  // (c0, c4)
          tw[pp][j].y = __byte_perm(h[0], h[2], 0x7632);  // (c1, c5)
          tw[pp][j].z = __byte_perm(h[1], h[3], 0x5410);  // (c2, c6)
          tw[pp][j].w = __byte_perm(h[1], h[3], 0x7632);  // (c3, c7)
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) ss[i] += __shfl_xor_sync(~0u, ss[i], 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) ss[i] += __shfl_xor_sync(~0u, ss[i], 1);
        float t4[4], t2[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) t4[i] = ss[i] + ss[i + 4];
#pragma unroll
        for (int i = 0; i < 2; ++i) t2[i] = t4[i] + t4[i + 2];
        tot[pp] = t2[0] + t2[1];
      }
      PT(2);
      PACC(1, 1, 2);
      if (n >= 1) mb_wait(sxe, (n - 1) & 1);                      // M is done with the tile
      PT(3);
      PACC(2, 2, 3);
      if (n >= NPB) mb_wait_cl(&pempty[b], ((n / NPB) - 1) & 1);  // every reducer is done with gb[b]
      PT(7);
      PACC(7, 3, 7);
#pragma unroll
      for (int pp = 0; pp < 2; ++pp) {
        const int rl = 16 * pp + 4 * warp + (lane >> 3);
        __nv_bfloat16* sxr = sx + (chl * SUB + rl) * SXP;
#pragma unroll
        for (int j = 0; j < HC; ++j) *(uint4*)(sxr + j * CW + col0) = tw[pp][j];
        if ((cg & 3) == 0) {  // the row's reducer is CTA rank rl / 4; 8 such arrivals per source and reducer
          const unsigned dst = (unsigned)(rl >> 2);
          st_async(mapa(s_u32(gbs + gb_idx(b, (int)crank, rl & 3, chl, 0)), dst), tot[pp], mapa(s_u32(&pfull[b]), dst));
        }
      }
      __syncwarp();
      if (lane == 0) mb_arrive(sxf);  // this warp's rows of the tile and their ss partials are in SMEM
    }
  } else if (warp < NX + NM) {  // ---------------- M: MMA
    const int p = warp - NX, pc = p >> 1, rb = 16 * (p & 1), g = lane >> 2, t = lane & 3;
    const __nv_bfloat16* sxc = sx + pc * SUB * SXP;
    const float2* fc = (const float2*)(fns + pc * (FN_FLOATS / 2));
    mb_wait(fnbar, 0);
    for (int n = 0; n < nsub; ++n) {
      const int b = n % NPB;
      PT(0);
      mb_wait(sxf, n & 1);
      PT(1);
      PADD(false, 3, 0, 1);
      float d[NT][4];
#pragma unroll
      for (int q = 0; q < NT; ++q) d[q][0] = d[q][1] = d[q][2] = d[q][3] = 0.f;
#pragma unroll
      for (int s2 = 0; s2 < KS; ++s2) {
        const int kk = 8 * s2;
        const unsigned w0 = *(const unsigned*)(sxc + (rb + g) * SXP + kk + 2 * t);
        const unsigned w1 = *(const unsigned*)(sxc + (rb + g + 8) * SXP + kk + 2 * t);
        const unsigned a[4] = {w0 << 16, w1 << 16, w0 & 0xFFFF0000u, w1 & 0xFFFF0000u};
#pragma unroll
        for (int q = 0; q < NT; ++q) {
          const float2 f = fc[(s2 * NT + q) * 32 + lane];
          const unsigned hx = rna(__float_as_uint(f.x)), hy = rna(__float_as_uint(f.y));
          const unsigned lx = rna(__float_as_uint(__fsub_rn(f.x, __uint_as_float(hx))));
          const unsigned ly = rna(__float_as_uint(__fsub_rn(f.y, __uint_as_float(hy))));
          mma_tf32(d[q], a, lx, ly);
          mma_tf32(d[q], a, hx, hy);
        }
      }
      __syncwarp();
      if (lane == 0) mb_arrive(sxe);  // the tile's A fragments are consumed
      PT(2);
      PADD(false, 4, 1, 2);
#pragma unroll
      for (int h = 0; h < 2; ++h) {  // rows rb + g + 8 h -> reducer rank row / 4; st.async, bytes counted on the reducer's pfull
        const int row = rb + g + 8 * h;
        const unsigned dst = (unsigned)(row >> 2);
        const uint32_t mb = mapa(s_u32(&pfull[b]), dst);
#pragma unroll
        for (int q = 0; q < NT; ++q)
          st_async2(mapa(s_u32(gbs + gb_idx(b, (int)crank, row & 3, pc, 1 + 8 * q + 2 * t)), dst), d[q][2 * h], d[q][2 * h + 1], mb);
      }
    }
  } else {  // ---------------- Rd (warp 9): fold this CTA's 4 rows of every sub-tile from its gather buffer
    constexpr unsigned GB_BYTES = CL * 4 * 2 * NP * 4;  // 8 sources x 4 rows x 2 chunks x 25 fp32 per sub-tile
    for (int m = 0; m < nsub; ++m) {
      const int b = m % NPB, r0 = (s0 + m) * SUB;
      if (lane == 0) mb_expect(&pfull[b], GB_BYTES);  // peers' st.async may already have landed (tx-count goes negative)
      PT(0);
      mb_wait_cl(&pfull[b], (m / NPB) & 1);
      PT(1);
      PADD(true, 5, 0, 1);
      float a4[4];
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        const int o = min(it * 32 + lane, 4 * NP - 1), lrow = o / NP, q = o % NP;
        float v[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) v[c] = gbs[gb_idx(b, c >> 1, lrow, c & 1, q)];
        float a = 0.f;
#pragma unroll
        for (int c = 0; c < 16; ++c) a += v[c];
        a4[it] = a;
      }
      PT(5);
      PADD(true, 6, 1, 5);
      __syncwarp();
      // free the buffer to the 8 writers. relaxed: a release arrive at cluster scope cost 3.3 K cycles per sub-tile (it
      // waits for this thread's earlier global stores); the reads it orders have been consumed by the adds above
      if (lane < CL) mb_arrive_remote_relaxed(&pempty[b], lane);
      PT(6);
      PADD(true, 4, 5, 6);
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        const int o = it * 32 + lane, lrow = o / NP, q = o % NP, r = r0 + 4 * (int)crank + lrow;
        if (o < 4 * NP && r < R) A.gpart[(size_t)r * GP + (cp / CL) * NP + q] = a4[it];
      }
      PT(2);
      PADD(true, 3, 6, 2);
    }
  }
  cluster_sync();
}
}  // namespace mhcf

void fa2(torch::Tensor y, torch::Tensor rs, torch::Tensor post, torch::Tensor comb, torch::Tensor lag, torch::Tensor fnf,
         torch::Tensor xn, torch::Tensor scratch, int64_t nrs) {
  using namespace mhcf;
  const c10::cuda::CUDAGuard dg(y.device());
  same_dev(y, {rs, post, comb, lag, fnf, xn, scratch});
  const int R = (int)y.size(0);
  TORCH_CHECK(fnf.scalar_type() == torch::kFloat32 && fnf.numel() == (int64_t)NCH * FN_FLOATS / 2, "mhcf: fnf (prep_f)");
  TORCH_CHECK(R >= 1 && rs.size(0) == R && xn.size(0) >= R, "mhcf: rows");
  check_rows(y, R, D, "y");
  check_rows(rs, R, HC * D, "rs");
  check_rows(xn, R, HC * D, "xn");
  check_rows(post, R, HC, "post");
  check_rows(comb, R, 16, "comb");
  check_rows(lag, R, HC, "lag");
  TORCH_CHECK(y.scalar_type() == torch::kBFloat16 && rs.scalar_type() == torch::kBFloat16 && post.scalar_type() == torch::kFloat32 &&
                  comb.scalar_type() == torch::kFloat32 && lag.scalar_type() == torch::kFloat32,
              "mhcf: dtypes");
  TORCH_CHECK(scratch.scalar_type() == torch::kFloat32 && scratch.numel() >= (int64_t)(in_off(R) + (size_t)R * D / 2), "mhcf: scratch");
  smem_attr(k_fa2, S2_TOTAL, 2);
  const int ns = (R + SUB - 1) / SUB;
  int k = (int)(nrs > 0 ? nrs : (g_nrs > 0 ? g_nrs : 3));
  k = std::max(1, std::min(k, ns));
  Args a{post.data_ptr<float>(), comb.data_ptr<float>(), lag.data_ptr<float>(), fnf.data_ptr<float>(), (__nv_bfloat16*)xn.data_ptr(),
         (__nv_bfloat16*)(scratch.data_ptr<float>() + in_off(R)), scratch.data_ptr<float>(), R, ns, k};
  const CUtensorMap my = tmap(y.data_ptr(), R, SUB), mr = tmap(rs.data_ptr(), HC * R, HC * SUB);
  k_fa2<<<dim3(NSL, k), THR2, S2_TOTAL, at::cuda::getCurrentCUDAStream()>>>(my, mr, a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
int64_t smem2_bytes() { return mhcf::S2_TOTAL; }
// ------------------------------------------------------------------------------------------------------------- fmix
// fmix: the mixes' part of the boundary from xn (the frow split: frow on the main stream, fmix +
// fb_mix on the side). fa2's M / Rd machinery with phase 1 replaced by reading the bf16 xn tile: per 32-row sub-tile one
// TMA box of xn ([4R, D] view, 128 x 64); 2 X warps: per-chunk sum of squares (k1a v2's order) + the k-permuted MMA tile;
// 2 groups of 4 M warps on alternating sub-tiles (double-buffered tile: the 32-deep mma chain of one sub-tile overlaps
// the next one's); Rd folds the 16-chunk groups as in fa2 -> gpart. Same partials, same bits as fa2 / k1a.
namespace mhcf {
#ifndef MHCF_FX_NX
#define MHCF_FX_NX 2
#endif
#ifndef MHCF_FX_UNR
#define MHCF_FX_UNR 1  // X passes interleaved by the compiler (1: one pass at a time)
#endif
constexpr int FX_UNR = MHCF_FX_UNR;
constexpr int FX_NX = MHCF_FX_NX, FX_MW0 = FX_NX, FX_NMG = 2, FX_W_PROD = FX_MW0 + 4 * FX_NMG, FX_W_RD = FX_W_PROD + 1,
              FX_THR = (FX_W_RD + 1) * 32;  // 12 warps
constexpr int FX_ST = SUB * HC * COLS * 2;  // 16 KB: the xn box
constexpr int FX_STAGE = 0, FX_SX = NST * FX_ST, FX_SXB = 2 * SUB * SXP * 2, FX_PB = FX_SX + 2 * FX_SXB,
              FX_FN = FX_PB + NPB * PBB, FX_BAR = FX_FN + FN_FLOATS * 4, FX_TOTAL = FX_BAR + 256;
static_assert(FX_FN % 128 == 0 && FX_PB % 16 == 0, "fmix smem alignment");

__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(FX_THR, 2)
    k_fmix(const __grid_constant__ CUtensorMap tmx, const Args A) {
  extern __shared__ __align__(128) unsigned char sm[];
  uint64_t* bar = (uint64_t*)(sm + FX_BAR);
  uint64_t *full = bar, *empty = bar + NST, *sxf = bar + 2 * NST, *sxe = sxf + 2, *pfull = sxe + 2, *pempty = pfull + NPB,
           *fnbar = pempty + NPB;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, cp = blockIdx.x, R = A.R;
  const int s0 = (int)((long long)A.nsub_total * blockIdx.y / A.nrs), s1 = (int)((long long)A.nsub_total * (blockIdx.y + 1) / A.nrs);
  const int nsub = s1 - s0;
  const unsigned crank = cl_rank();
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) mb_init(&full[s], 1), mb_init(&empty[s], FX_NX);
    for (int q = 0; q < 2; ++q) mb_init(&sxf[q], FX_NX), mb_init(&sxe[q], 4);
    for (int b = 0; b < NPB; ++b) mb_init(&pfull[b], 1), mb_init(&pempty[b], CL);
    mb_init(fnbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster_sync();
  float* fns = (float*)(sm + FX_FN);
  float* gbs = (float*)(sm + FX_PB);
  if (warp == FX_W_PROD) {  // ---------------- producer
    if (lane == 0) {
      mb_expect(fnbar, FN_FLOATS * 4);
      bulk_g2s(fns, A.fnf + (size_t)(2 * cp) * (FN_FLOATS / 2), FN_FLOATS * 4, fnbar);
      for (int n = 0; n < nsub; ++n) {
        const int s = n % NST, r0 = (s0 + n) * SUB;
        if (n >= NST) mb_wait(&empty[s], ((n / NST) - 1) & 1);
        mb_expect(&full[s], FX_ST);
        tma2d(sm + FX_STAGE + s * FX_ST, &tmx, COLS * cp, HC * r0, &full[s]);
      }
    }
    __syncwarp();
  } else if (warp < FX_NX) {  // ---------------- X: ss partials + MMA tile from the xn box
    const int cg = lane & 7, chl = cg >> 2, col0 = 8 * (cg & 3);
    for (int n = 0; n < nsub; ++n) {
      const int s = n % NST, b = n % NPB, q2 = n & 1;
      const unsigned char* st = sm + FX_STAGE + s * FX_ST;
      __nv_bfloat16* sx = (__nv_bfloat16*)(sm + FX_SX + q2 * FX_SXB);
      PT(0);
      mb_wait(&full[s], (n / NST) & 1);
      PT(1);
      if (n >= 2) mb_wait(&sxe[q2], ((n >> 1) - 1) & 1);           // group q2 is done with this tile buffer
      PT(2);
      if (n >= NPB) mb_wait_cl(&pempty[b], ((n / NPB) - 1) & 1);    // every reducer is done with gb[b]
      PT(3);
      PACC(0, 0, 1);
      PACC(1, 1, 2);
      PACC(2, 2, 3);
#pragma unroll FX_UNR
      for (int it = 0; it < SUB / (4 * FX_NX); ++it) {  // passes of 4 FX_NX rows (4 per warp)
        const int rl = 4 * FX_NX * it + 4 * warp + (lane >> 3);
        uint4 xv[HC];
#pragma unroll
        for (int j = 0; j < HC; ++j) xv[j] = *(const uint4*)(st + (rl * HC + j) * 128 + cg * 16);
        float ss[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) ss[i] = 0.f;
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          const __nv_bfloat162* h2 = (const __nv_bfloat162*)&xv[j];
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            const float2 f2 = __bfloat1622float2(h2[m]);
            ss[2 * m] = fmaf(f2.x, f2.x, ss[2 * m]);
            ss[2 * m + 1] = fmaf(f2.y, f2.y, ss[2 * m + 1]);
          }
        }
        __nv_bfloat16* sxr = sx + (chl * SUB + rl) * SXP;
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          const unsigned* h = (const unsigned*)&xv[j];
          uint4 w;
          w.x = __byte_perm(h[0], h[2], 0x5410), w.y = __byte_perm(h[0], h[2], 0x7632);
          w.z = __byte_perm(h[1], h[3], 0x5410), w.w = __byte_perm(h[1], h[3], 0x7632);
          *(uint4*)(sxr + j * CW + col0) = w;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) ss[i] += __shfl_xor_sync(~0u, ss[i], 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) ss[i] += __shfl_xor_sync(~0u, ss[i], 1);
        float t4[4], t2[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) t4[i] = ss[i] + ss[i + 4];
#pragma unroll
        for (int i = 0; i < 2; ++i) t2[i] = t4[i] + t4[i + 2];
        if ((cg & 3) == 0) {
          const unsigned dst = (unsigned)(rl >> 2);
          st_async(mapa(s_u32(gbs + gb_idx(b, (int)crank, rl & 3, chl, 0)), dst), t2[0] + t2[1], mapa(s_u32(&pfull[b]), dst));
        }
      }
      __syncwarp();
      PT(4);
      PACC(3, 3, 4);
      if (lane == 0) {
        mb_arrive(&empty[s]);
        mb_arrive(&sxf[q2]);
      }
    }
  } else if (warp < FX_W_PROD) {  // ---------------- M: group grp takes sub-tiles n = grp, grp + 2, ...
    const int grp = (warp - FX_MW0) >> 2, p = (warp - FX_MW0) & 3, pc = p >> 1, rb = 16 * (p & 1), g = lane >> 2,
              t = lane & 3;
    const float2* fc = (const float2*)(fns + pc * (FN_FLOATS / 2));
    mb_wait(fnbar, 0);
    for (int n = grp; n < nsub; n += 2) {
      const int b = n % NPB;
      const __nv_bfloat16* sxc = (const __nv_bfloat16*)(sm + FX_SX + grp * FX_SXB) + pc * SUB * SXP;
      PT(0);
      mb_wait(&sxf[grp], (n >> 1) & 1);
      PT(1);
      PADD(warp == FX_MW0, 4, 0, 1);
      float d[NT][4];
#pragma unroll
      for (int q = 0; q < NT; ++q) d[q][0] = d[q][1] = d[q][2] = d[q][3] = 0.f;
#pragma unroll
      for (int s2 = 0; s2 < KS; ++s2) {
        const int kk = 8 * s2;
        const unsigned w0 = *(const unsigned*)(sxc + (rb + g) * SXP + kk + 2 * t);
        const unsigned w1 = *(const unsigned*)(sxc + (rb + g + 8) * SXP + kk + 2 * t);
        const unsigned a[4] = {w0 << 16, w1 << 16, w0 & 0xFFFF0000u, w1 & 0xFFFF0000u};
#pragma unroll
        for (int q = 0; q < NT; ++q) {
          const float2 f = fc[(s2 * NT + q) * 32 + lane];
          const unsigned hx = rna(__float_as_uint(f.x)), hy = rna(__float_as_uint(f.y));
          const unsigned lx = rna(__float_as_uint(__fsub_rn(f.x, __uint_as_float(hx))));
          const unsigned ly = rna(__float_as_uint(__fsub_rn(f.y, __uint_as_float(hy))));
          mma_tf32(d[q], a, lx, ly);
          mma_tf32(d[q], a, hx, hy);
        }
      }
      __syncwarp();
      PT(2);
      PADD(warp == FX_MW0, 5, 1, 2);
      if (lane == 0) mb_arrive(&sxe[grp]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = rb + g + 8 * h;
        const unsigned dst = (unsigned)(row >> 2);
        const uint32_t mb = mapa(s_u32(&pfull[b]), dst);
#pragma unroll
        for (int q = 0; q < NT; ++q)
          st_async2(mapa(s_u32(gbs + gb_idx(b, (int)crank, row & 3, pc, 1 + 8 * q + 2 * t)), dst), d[q][2 * h], d[q][2 * h + 1], mb);
      }
    }
  } else {  // ---------------- Rd: fold (fa2's)
    constexpr unsigned GB_BYTES = CL * 4 * 2 * NP * 4;
    for (int m = 0; m < nsub; ++m) {
      const int b = m % NPB, r0 = (s0 + m) * SUB;
      if (lane == 0) mb_expect(&pfull[b], GB_BYTES);
      mb_wait_cl(&pfull[b], (m / NPB) & 1);
      float a4[4];
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        const int o = min(it * 32 + lane, 4 * NP - 1), lrow = o / NP, q = o % NP;
        float v[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) v[c] = gbs[gb_idx(b, c >> 1, lrow, c & 1, q)];
        float a = 0.f;
#pragma unroll
        for (int c = 0; c < 16; ++c) a += v[c];
        a4[it] = a;
      }
      __syncwarp();
      if (lane < CL) mb_arrive_remote_relaxed(&pempty[b], lane);
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        const int o = it * 32 + lane, lrow = o / NP, q = o % NP, r = r0 + 4 * (int)crank + lrow;
        if (o < 4 * NP && r < R) A.gpart[(size_t)r * GP + (cp / CL) * NP + q] = a4[it];
      }
    }
  }
  cluster_sync();
}
}  // namespace mhcf

// fmix: xn [R, 4, D] -> gpart (scratch[0 : R * 250]); then fb_mix(scratch, R, ...) gives pre / post / comb
void fmix(torch::Tensor xn, int64_t Rr, torch::Tensor fnf, torch::Tensor scratch, int64_t nrs) {
  using namespace mhcf;
  const c10::cuda::CUDAGuard dg(xn.device());
  same_dev(xn, {fnf, scratch});
  const int R = (int)Rr;
  TORCH_CHECK(R >= 1 && xn.size(0) >= R, "mhcf fmix: rows");
  check_rows(xn, R, HC * D, "xn");
  TORCH_CHECK(xn.scalar_type() == torch::kBFloat16, "mhcf fmix: xn bf16");
  TORCH_CHECK(fnf.scalar_type() == torch::kFloat32 && fnf.numel() == (int64_t)NCH * FN_FLOATS / 2, "mhcf: fnf (prep_f)");
  TORCH_CHECK(scratch.scalar_type() == torch::kFloat32 && scratch.numel() >= (int64_t)R * GP && ((uintptr_t)scratch.data_ptr() % 16) == 0,
              "mhcf fmix: scratch");
  smem_attr(k_fmix, FX_TOTAL, 5);
  const int ns = (R + SUB - 1) / SUB;
  int k = (int)(nrs > 0 ? nrs : (g_nrs > 0 ? g_nrs : 3));
  k = std::max(1, std::min(k, ns));
  Args a{nullptr, nullptr, nullptr, fnf.data_ptr<float>(), nullptr, nullptr, scratch.data_ptr<float>(), R, ns, k};
  const CUtensorMap mx = tmap(xn.data_ptr(), HC * R, HC * SUB);
  k_fmix<<<dim3(NSL, k), FX_THR, FX_TOTAL, at::cuda::getCurrentCUDAStream()>>>(mx, a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// ------------------------------------------------------------------------------------------------------------ fmix2
// fmix2: fmix without X warps. xn arrives as 4 TMA 3D boxes per sub-tile (one per copy: [32 rows][64
// cols], 128-B swizzle), the M warps read their tf32 A fragments straight from the boxes (conflict-free) and accumulate
// the per-chunk sum of squares on the way (per column fmaf over the copies in order, then k1a v2's tree: xor 16 / 8 / 4
// in-lane, xor 2 / 1 by shuffles). 2 groups of 4 M warps on alternating sub-tiles, 4-stage ring; Rd as fa2.
namespace mhcf {
constexpr int F2_NMG = 2, F2_W_PROD = 4 * F2_NMG, F2_W_RD = F2_W_PROD + 1, F2_THR = (F2_W_RD + 1) * 32, F2_NST = 4;
constexpr int F2_BOX = SUB * COLS * 2, F2_ST = HC * F2_BOX;  // 4 KB per copy, 16 KB per sub-tile
constexpr int F2_STAGE = 0, F2_PB = F2_NST * F2_ST, F2_FN = F2_PB + NPB * PBB, F2_BAR = F2_FN + FN_FLOATS * 4,
              F2_TOTAL = F2_BAR + 256;
static_assert(F2_FN % 128 == 0, "fmix2 smem alignment");
__device__ __forceinline__ void tma3d(void* dst, const CUtensorMap* m, int c0, int c1, int c2, uint64_t* b) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(
                   s_u32(dst)),
               "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(s_u32(b)) : "memory");
}
__device__ __forceinline__ unsigned lds_u16(const unsigned char* p) {
  unsigned short v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(s_u32(p)));
  return (unsigned)v;
}

__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(F2_THR, 2)
    k_fmix2(const __grid_constant__ CUtensorMap tmx, const Args A) {
  extern __shared__ __align__(1024) unsigned char sm[];
  uint64_t* bar = (uint64_t*)(sm + F2_BAR);
  uint64_t *full = bar, *empty = bar + F2_NST, *pfull = empty + F2_NST, *pempty = pfull + NPB, *fnbar = pempty + NPB;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, cp = blockIdx.x, R = A.R;
  const int s0 = (int)((long long)A.nsub_total * blockIdx.y / A.nrs), s1 = (int)((long long)A.nsub_total * (blockIdx.y + 1) / A.nrs);
  const int nsub = s1 - s0;
  const unsigned crank = cl_rank();
  if (tid == 0) {
    for (int s = 0; s < F2_NST; ++s) mb_init(&full[s], 1), mb_init(&empty[s], 4);
    for (int b = 0; b < NPB; ++b) mb_init(&pfull[b], 1), mb_init(&pempty[b], CL);
    mb_init(fnbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster_sync();
  float* fns = (float*)(sm + F2_FN);
  float* gbs = (float*)(sm + F2_PB);
  if (warp == F2_W_PROD) {  // ---------------- producer
    if (lane == 0) {
      mb_expect(fnbar, FN_FLOATS * 4);
      bulk_g2s(fns, A.fnf + (size_t)(2 * cp) * (FN_FLOATS / 2), FN_FLOATS * 4, fnbar);
      for (int n = 0; n < nsub; ++n) {
        const int s = n % F2_NST, r0 = (s0 + n) * SUB;
        if (n >= F2_NST) mb_wait(&empty[s], ((n / F2_NST) - 1) & 1);
        mb_expect(&full[s], F2_ST);
#pragma unroll
        for (int j = 0; j < HC; ++j) tma3d(sm + F2_STAGE + s * F2_ST + j * F2_BOX, &tmx, COLS * cp, j, r0, &full[s]);
      }
    }
    __syncwarp();
  } else if (warp < F2_W_PROD) {  // ---------------- M: group grp takes sub-tiles n = grp, grp + 2, ...
    const int grp = warp >> 2, p = warp & 3, pc = p >> 1, rb = 16 * (p & 1), g = lane >> 2, t = lane & 3;
    const int rA = rb + g, rB = rb + g + 8;
    const float2* fc = (const float2*)(fns + pc * (FN_FLOATS / 2));
    mb_wait(fnbar, 0);
    for (int n = grp; n < nsub; n += 2) {
      const int s = n % F2_NST, b = n % NPB;
      const unsigned char* st = sm + F2_STAGE + s * F2_ST;
      mb_wait(&full[s], (n / F2_NST) & 1);
      if (n >= NPB) mb_wait_cl(&pempty[b], ((n / NPB) - 1) & 1);  // every reducer is done with gb[b]
      float d[NT][4], sa[4][2], sb[4][2];
#pragma unroll
      for (int q = 0; q < NT; ++q) d[q][0] = d[q][1] = d[q][2] = d[q][3] = 0.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) sa[q][0] = sa[q][1] = sb[q][0] = sb[q][1] = 0.f;
#pragma unroll
      for (int s2 = 0; s2 < KS; ++s2) {
        const int j = s2 >> 2, q4 = s2 & 3, ch = 4 * pc + q4;  // copy, 8-column group of the chunk
        const unsigned char* bx = st + j * F2_BOX;
        // 128-B swizzle: 16-B chunk ^= SMEM address bits [7:9] of the row (robust to the dynamic-SMEM base alignment)
        const unsigned char* ra = bx + rA * 128;
        const unsigned char* rbp = bx + rB * 128;
        const unsigned char* pa = ra + ((ch ^ ((s_u32(ra) >> 7) & 7)) << 4) + 2 * t;
        const unsigned char* pb = rbp + ((ch ^ ((s_u32(rbp) >> 7) & 7)) << 4) + 2 * t;
        const unsigned a[4] = {lds_u16(pa) << 16, lds_u16(pb) << 16, lds_u16(pa + 8) << 16, lds_u16(pb + 8) << 16};
        {  // sum of squares per column, copies in order (k1a: ss = fmaf(x_j, x_j, ss), j = 0..3)
          const float x0 = __uint_as_float(a[0]), x1 = __uint_as_float(a[1]), x2 = __uint_as_float(a[2]), x3 = __uint_as_float(a[3]);
          sa[q4][0] = fmaf(x0, x0, sa[q4][0]);
          sa[q4][1] = fmaf(x2, x2, sa[q4][1]);
          sb[q4][0] = fmaf(x1, x1, sb[q4][0]);
          sb[q4][1] = fmaf(x3, x3, sb[q4][1]);
        }
#pragma unroll
        for (int q = 0; q < NT; ++q) {
          const float2 f = fc[(s2 * NT + q) * 32 + lane];
          const unsigned hx = rna(__float_as_uint(f.x)), hy = rna(__float_as_uint(f.y));
          const unsigned lx = rna(__float_as_uint(__fsub_rn(f.x, __uint_as_float(hx))));
          const unsigned ly = rna(__float_as_uint(__fsub_rn(f.y, __uint_as_float(hy))));
          mma_tf32(d[q], a, lx, ly);
          mma_tf32(d[q], a, hx, hy);
        }
      }
      __syncwarp();
      if (lane == 0) mb_arrive(&empty[s]);
      // the chunk's sum of squares per row: columns c = 8 q + 4 h + t; tree xor 16 (q ^ 2), 8 (q ^ 1), 4 (h), then 2, 1 (t)
      float ta, tb;
      {
        float l16a[2][2], l16b[2][2];
#pragma unroll
        for (int q = 0; q < 2; ++q)
#pragma unroll
          for (int h = 0; h < 2; ++h) l16a[q][h] = sa[q][h] + sa[q + 2][h], l16b[q][h] = sb[q][h] + sb[q + 2][h];
        const float l8a0 = l16a[0][0] + l16a[1][0], l8a1 = l16a[0][1] + l16a[1][1];
        const float l8b0 = l16b[0][0] + l16b[1][0], l8b1 = l16b[0][1] + l16b[1][1];
        ta = l8a0 + l8a1;
        tb = l8b0 + l8b1;
        ta += __shfl_xor_sync(~0u, ta, 2);
        tb += __shfl_xor_sync(~0u, tb, 2);
        ta += __shfl_xor_sync(~0u, ta, 1);
        tb += __shfl_xor_sync(~0u, tb, 1);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = h ? rB : rA;
        const unsigned dst = (unsigned)(row >> 2);
        const uint32_t mb = mapa(s_u32(&pfull[b]), dst);
        if (t == 0) st_async(mapa(s_u32(gbs + gb_idx(b, (int)crank, row & 3, pc, 0)), dst), h ? tb : ta, mb);
#pragma unroll
        for (int q = 0; q < NT; ++q)
          st_async2(mapa(s_u32(gbs + gb_idx(b, (int)crank, row & 3, pc, 1 + 8 * q + 2 * t)), dst), d[q][2 * h], d[q][2 * h + 1], mb);
      }
    }
  } else {  // ---------------- Rd: fold (fa2's)
    constexpr unsigned GB_BYTES = CL * 4 * 2 * NP * 4;
    for (int m = 0; m < nsub; ++m) {
      const int b = m % NPB, r0 = (s0 + m) * SUB;
      if (lane == 0) mb_expect(&pfull[b], GB_BYTES);
      mb_wait_cl(&pfull[b], (m / NPB) & 1);
      float a4[4];
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        const int o = min(it * 32 + lane, 4 * NP - 1), lrow = o / NP, q = o % NP;
        float v[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) v[c] = gbs[gb_idx(b, c >> 1, lrow, c & 1, q)];
        float a = 0.f;
#pragma unroll
        for (int c = 0; c < 16; ++c) a += v[c];
        a4[it] = a;
      }
      __syncwarp();
      if (lane < CL) mb_arrive_remote_relaxed(&pempty[b], lane);
#pragma unroll
      for (int it = 0; it < 4; ++it) {
        const int o = it * 32 + lane, lrow = o / NP, q = o % NP, r = r0 + 4 * (int)crank + lrow;
        if (o < 4 * NP && r < R) A.gpart[(size_t)r * GP + (cp / CL) * NP + q] = a4[it];
      }
    }
  }
  cluster_sync();
}
}  // namespace mhcf

// xn [R, 4, D] as a 3D tensor {D, 4, R}, box {64, 1, 32}, 128-B swizzle
static CUtensorMap tmap3(const void* p, int rows) {
  if (!g_encode) tmap(p, rows, mhcf::SUB);  // resolves the driver entry point
  CUtensorMap m;
  cuuint64_t dims[3] = {(cuuint64_t)mhcf::D, (cuuint64_t)mhcf::HC, (cuuint64_t)rows};
  cuuint64_t strides[2] = {(cuuint64_t)mhcf::D * 2, (cuuint64_t)mhcf::HC * mhcf::D * 2};
  cuuint32_t box[3] = {(cuuint32_t)mhcf::COLS, 1, (cuuint32_t)mhcf::SUB};
  cuuint32_t es[3] = {1, 1, 1};
  CUresult r = g_encode(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(p), dims, strides, box, es,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "mhcf: 3D tensor map encode failed ", (int)r);
  return m;
}
void fmix2(torch::Tensor xn, int64_t Rr, torch::Tensor fnf, torch::Tensor scratch, int64_t nrs) {
  using namespace mhcf;
  const c10::cuda::CUDAGuard dg(xn.device());
  same_dev(xn, {fnf, scratch});
  const int R = (int)Rr;
  TORCH_CHECK(R >= 1 && xn.size(0) >= R, "mhcf fmix2: rows");
  check_rows(xn, R, HC * D, "xn");
  TORCH_CHECK(xn.scalar_type() == torch::kBFloat16, "mhcf fmix2: xn bf16");
  TORCH_CHECK(fnf.scalar_type() == torch::kFloat32 && fnf.numel() == (int64_t)NCH * FN_FLOATS / 2, "mhcf: fnf (prep_f)");
  TORCH_CHECK(scratch.scalar_type() == torch::kFloat32 && scratch.numel() >= (int64_t)R * GP && ((uintptr_t)scratch.data_ptr() % 16) == 0,
              "mhcf fmix2: scratch");
  smem_attr(k_fmix2, F2_TOTAL, 6);
  const int ns = (R + SUB - 1) / SUB;
  int k = (int)(nrs > 0 ? nrs : (g_nrs > 0 ? g_nrs : 3));
  k = std::max(1, std::min(k, ns));
  Args a{nullptr, nullptr, nullptr, fnf.data_ptr<float>(), nullptr, nullptr, scratch.data_ptr<float>(), R, ns, k};
  const CUtensorMap mx = tmap3(xn.data_ptr(), R);
  k_fmix2<<<dim3(NSL, k), F2_THR, F2_TOTAL, at::cuda::getCurrentCUDAStream()>>>(mx, a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// frow: y [R, D], rs [R, 4, D], post / lag [R, 4], comb [R, 16], normw [D] -> xn [R, 4, D], out [R, D], ptok [R] (or empty)
void frow(torch::Tensor y, torch::Tensor rs, torch::Tensor post, torch::Tensor comb, torch::Tensor lag, torch::Tensor normw,
          double norm_eps, torch::Tensor xn, torch::Tensor out, torch::Tensor ptok_o) {
  using namespace mhcf;
  const c10::cuda::CUDAGuard dg(y.device());
  same_dev(y, {rs, post, comb, lag, normw, xn, out});
  const int R = (int)y.size(0);
  TORCH_CHECK(R >= 1 && rs.size(0) == R && xn.size(0) >= R && out.size(0) >= R, "mhcf frow: rows");
  check_rows(y, R, D, "y");
  check_rows(rs, R, HC * D, "rs");
  check_rows(xn, R, HC * D, "xn");
  check_rows(out, R, D, "out");
  check_rows(post, R, HC, "post");
  check_rows(comb, R, 16, "comb");
  check_rows(lag, R, HC, "lag");
  TORCH_CHECK(normw.scalar_type() == torch::kFloat32 && normw.is_contiguous() && normw.numel() == D, "mhcf frow: normw");
  TORCH_CHECK(ptok_o.numel() == 0 || (ptok_o.device() == y.device() && ptok_o.scalar_type() == torch::kInt32 &&
                                      ptok_o.numel() >= R),
              "mhcf frow: ptok");
  smem_attr(k_frow, FR_SMEM, 3);
  k_frow<<<R, 256, FR_SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)y.data_ptr(), (const __nv_bfloat16*)rs.data_ptr(), post.data_ptr<float>(), comb.data_ptr<float>(),
      lag.data_ptr<float>(), normw.data_ptr<float>(), R, (float)norm_eps, (__nv_bfloat16*)xn.data_ptr(),
      (__nv_bfloat16*)out.data_ptr(), ptok_o.numel() ? ptok_o.data_ptr<int>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


static const __nv_bfloat16* in_of(torch::Tensor scratch, int R) {
  return (const __nv_bfloat16*)(scratch.data_ptr<float>() + mhcf::in_off(R));
}
static int g_fbo = 1;  // fb_out kernel: 1 = k_fb<1> (default: 32.0 us at 4,288 rows, 7.7 at 320), 2 = k_fbo_row (32.0 / 8.6), 4 = k_fbo4
static int g_fbm = 8;  // fb_mix kernel: 8 = k_fbm8 (one warp per row, default), 1 = k_fb<2> (one 256-thread CTA per row)
void set_fbm(int64_t k) { g_fbm = (int)k; }
void set_fbo(int64_t k) { g_fbo = (int)k; }
void fb_out(torch::Tensor scratch, int64_t R, torch::Tensor normw, double norm_eps, torch::Tensor out, torch::Tensor ptok_o) {
  const c10::cuda::CUDAGuard dg(scratch.device());
  same_dev(scratch, {normw, out});
  TORCH_CHECK(R >= 1 && (ptok_o.numel() == 0 || (ptok_o.is_cuda() && ptok_o.device() == scratch.device() &&
                                                 ptok_o.scalar_type() == torch::kInt32 && ptok_o.numel() >= R)),
              "mhcf fb_out: ptok int32 [>= R] on the same device, or empty");
  TORCH_CHECK(scratch.numel() >= (int64_t)(mhcf::in_off((int)R) + (size_t)R * mhcf::D / 2) && out.size(0) >= R, "mhcf fb_out: shapes");
  TORCH_CHECK(normw.scalar_type() == torch::kFloat32 && normw.is_contiguous() && ((uintptr_t)normw.data_ptr() % 16) == 0 &&
                  ((uintptr_t)out.data_ptr() % 8) == 0 && out.stride(0) == mhcf::D,
              "mhcf fb_out: normw fp32 contiguous, out rows contiguous");
  if (g_fbo == 2) {
    smem_attr(mhcf::k_fbo_row, mhcf::FBO_SMEM, 4);
    TORCH_CHECK(((uintptr_t)in_of(scratch, (int)R) % 16) == 0 && ((uintptr_t)out.data_ptr() % 16) == 0, "mhcf fb_out: alignment");
    mhcf::k_fbo_row<<<(int)R, 256, mhcf::FBO_SMEM, at::cuda::getCurrentCUDAStream()>>>(
        in_of(scratch, (int)R), normw.data_ptr<float>(), (int)R, (float)norm_eps, (__nv_bfloat16*)out.data_ptr(),
        ptok_o.numel() ? ptok_o.data_ptr<int>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (g_fbo == 4) {
    mhcf::k_fbo4<<<(int)R, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
        in_of(scratch, (int)R), normw.data_ptr<float>(), (int)R, (float)norm_eps, (__nv_bfloat16*)out.data_ptr(),
        ptok_o.numel() ? ptok_o.data_ptr<int>() : nullptr);
    return;
  }
  mhcf::k_fb<1><<<(int)R, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      nullptr, in_of(scratch, (int)R), nullptr, nullptr, normw.data_ptr<float>(), (int)R, (float)norm_eps, 0.f, 0, nullptr,
      nullptr, nullptr, (__nv_bfloat16*)out.data_ptr(), ptok_o.numel() ? ptok_o.data_ptr<int>() : nullptr);
}
void fb_mix(torch::Tensor scratch, int64_t R, torch::Tensor scale, torch::Tensor base, double norm_eps, double hc_eps,
            int64_t iters, torch::Tensor pre_o, torch::Tensor post_o, torch::Tensor comb_o) {
  const c10::cuda::CUDAGuard dg(scratch.device());
  same_dev(scratch, {scale, base, pre_o, post_o, comb_o});
  TORCH_CHECK(R >= 1 && scratch.numel() >= (int64_t)R * mhcf::GP && pre_o.size(0) >= R && post_o.size(0) >= R &&
                  comb_o.size(0) >= R && pre_o.is_contiguous() && post_o.is_contiguous() && comb_o.is_contiguous(),
              "mhcf fb_mix: shapes");
  if (g_fbm == 8) {
    mhcf::k_fbm8<<<((int)R + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        scratch.data_ptr<float>(), scale.data_ptr<float>(), base.data_ptr<float>(), (int)R, (float)norm_eps, (float)hc_eps,
        (int)iters, pre_o.data_ptr<float>(), post_o.data_ptr<float>(), comb_o.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  mhcf::k_fb<2><<<(int)R, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      scratch.data_ptr<float>(), nullptr, scale.data_ptr<float>(), base.data_ptr<float>(), nullptr, (int)R, (float)norm_eps,
      (float)hc_eps, (int)iters, pre_o.data_ptr<float>(), post_o.data_ptr<float>(), comb_o.data_ptr<float>(), nullptr, nullptr);
}
void fb_all(torch::Tensor scratch, int64_t R, torch::Tensor scale, torch::Tensor base, torch::Tensor normw, double norm_eps,
            double hc_eps, int64_t iters, torch::Tensor pre_o, torch::Tensor post_o, torch::Tensor comb_o, torch::Tensor out,
            torch::Tensor ptok_o) {
  const c10::cuda::CUDAGuard dg(scratch.device());
  same_dev(scratch, {scale, base, normw, pre_o, post_o, comb_o, out});
  mhcf::k_fb<0><<<(int)R, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      scratch.data_ptr<float>(), in_of(scratch, (int)R), scale.data_ptr<float>(), base.data_ptr<float>(),
      normw.data_ptr<float>(), (int)R, (float)norm_eps, (float)hc_eps, (int)iters, pre_o.data_ptr<float>(),
      post_o.data_ptr<float>(), comb_o.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      ptok_o.numel() ? ptok_o.data_ptr<int>() : nullptr);
}
// fn [24, 4 D] fp32 (the checkpoint's hc_*_fn, = finish_mhc's *_raw) -> [160 ch][16 s][3 nt][32 lanes][2] fp32: lane (g, t),
// q: chunk-local k = 8 s + t + 4 q (= 32 copy + column), n = 8 nt + g; mhc.cu prep_fn's gather without the hi / lo split.
torch::Tensor prep_f(torch::Tensor fn) {
  using namespace mhcf;
  auto f = fn.contiguous().to(torch::kFloat32).view({24, HC, NCH, CW});
  auto fk = f.permute({2, 1, 3, 0}).contiguous().view({NCH, HC * CW, 24});  // [ch][k_local][n]
  auto dev = fn.device();
  auto s = torch::arange(KS, torch::kLong).view({KS, 1, 1, 1}), nt = torch::arange(NT, torch::kLong).view({1, NT, 1, 1});
  auto l = torch::arange(32, torch::kLong).view({1, 1, 32, 1}), q = torch::arange(2, torch::kLong).view({1, 1, 1, 2});
  auto kk = (8 * s + torch::remainder(l, 4) + 4 * q).expand({KS, NT, 32, 2}), nn = (8 * nt + torch::div(l, 4, "floor")).expand({KS, NT, 32, 2});
  auto idx = (kk * 24 + nn).reshape({-1}).to(dev);
  return fk.view({NCH, -1}).index_select(1, idx).view({NCH, KS, NT, 32, 2}).contiguous();
}
void set_nrs(int64_t k) { g_nrs = (int)k; }
torch::Tensor prof_take() {  // [2048, 8] int64 cycle sums (MHCF_PROF builds), then zeroed
  auto t = torch::zeros({2048, 8}, torch::kInt64);
#if MHCF_PROF
  cudaMemcpyFromSymbol(t.data_ptr(), mhcf::g_prof, sizeof(mhcf::g_prof));
  static unsigned long long z[2048][8] = {};
  cudaMemcpyToSymbol(mhcf::g_prof, z, sizeof(z));
#endif
  return t;
}
int64_t smem_bytes() { return mhcf::SM_TOTAL; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa", &fa2);   // the boundary's streaming kernel (warp-specialized fa2)
  m.def("fa1", &fa);   // the first version (one role per phase; 2x slower), kept for A/B
  m.def("fa_x", &fa_x);
  m.def("fa2", &fa2);
  m.def("frow", &frow);
  m.def("fmix", &fmix);
  m.def("fmix2", &fmix2);
  m.def("smem2_bytes", &smem2_bytes);
  m.def("fb_out", &fb_out);
  m.def("fb_mix", &fb_mix);
  m.def("fb_all", &fb_all);
  m.def("prep_f", &prep_f);
  m.def("set_nrs", &set_nrs);
  m.def("set_fbo", &set_fbo);
  m.def("set_fbm", &set_fbm);
  m.def("prof_take", &prof_take);
  m.def("smem_bytes", &smem_bytes);
}
