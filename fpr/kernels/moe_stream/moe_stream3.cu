// moe_stream3: v3 of the routed-MoE decode layer (v1/v2 stay in moe_stream.cu).
//
// What changed against v1/v2, and why:
//  1. Chunks with n_pad <= MS3_MMA_MAXN (default 16: 78 % of the expert chunks at T=512) use mma.sync m16n8k16
//     instead of RS-wgmma: same A fragments (the RS-wgmma per-warp A layout is the m16n8k16 one), B fragments via
//     ldmatrix from the same swizzled token image, two accumulator chains. No fence / commit / wait per k64 step, no
//     descriptor arithmetic. Larger chunks keep RS-wgmma (mma.sync's B traffic from SMEM grows with N).
//  2. Scales come pre-biased and interleaved (bench.py make_p): per tile and k64 step one u32 per (warpgroup c, warp w,
//     row group g) = bytes e(b0, R), e(b0, R+8), e(b1, R), e(b1, R+8) with e = (s - 106) << 2 and R = 64c + 16w + g.
//     One LDS.32 per k64 step and one PRMT per multiplier give the f16x2 (2^(s-121), 2^(s-121)) (was 2 LDS.U8 and
//     ~3 ALU per scale).
//  3. Stages carry 32 KB of weights (8 k64 steps; GEMM2: the whole K = 576 tile) in one bulk copy each for A, scales
//     and tokens; an HBM+L2 TMA test streams 4.4 TB/s with 32 KB stages, 2.6-3.2 with 16 KB.
//     Chunks with more tokens than fit (GEMM1 n_pad > 32, GEMM2 > 24) use half-size stages. 3 stages per SM.
// Numerics unchanged: products exact (weights exact in f16, x prescaled to f16 exactly), fp32 accumulation.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

#include "wgmma_rs_gen.cuh"

#ifndef MS3_MMA_MAXN
#define MS3_MMA_MAXN 32  // n_pad <= this: mma.sync path; larger: RS-wgmma (sweep: 32 best, 16 -3 %, 64 -5 %)
#endif
#ifndef MS3_CHS1
#define MS3_CHS1 8  // GEMM1 k64 steps per stage for n_pad <= 32 (stage-depth sweep)
#endif
#ifndef MS3_S
#define MS3_S 3  // ring stages
#endif
#ifndef MS3_CTAS
#define MS3_CTAS 1  // GEMM CTAs per SM (2: smaller rings, 16 consumer warps per SM, <= 112 registers)
#endif
#ifndef MS3_CHS2
#define MS3_CHS2 9  // GEMM2 k64 steps per stage for n_pad <= 24 (9 = whole tile)
#endif
#ifndef MS3_CHB2
#define MS3_CHB2 3  // GEMM2 k64 steps per stage above 24 tokens
#endif
#ifndef MS3_S2
#define MS3_S2 3  // GEMM2 ring stages
#endif
#ifndef MS3_PSLEEP
#define MS3_PSLEEP 0  // producer: nanoseconds of sleep between empty-barrier polls (0 = spin)
#endif
#ifndef MS3_EPF
#define MS3_EPF 1  // 1: the producer copies the tile's EC (tokens, weights, 2^-p) into the first stage, consumers keep it in
                   //    registers, so the epilogue has no dependent global loads
#endif
#ifndef MS3_RED2
#define MS3_RED2 0  // 1: GEMM2 output as red.global.add.v2.f32 on row pairs (half the requests); measured no gain (L2
                   //    atomic units are per element), so off
#endif
#ifndef MS3_YSTG
#define MS3_YSTG 0  // 1: stage GEMM2's bf16 Y through SMEM (full-sector stores); measured slower
#endif
#ifndef MS3_ABL
#define MS3_ABL 0  // 1: stream only (consumers wait and release; results invalid)
#endif

namespace ms3 {

constexpr int D = 5120, IR = 576, NMAX = 64, NCW = 8, HDR = 1024;  // Hdr at +0, EC copy at +128
template <int MODE>
constexpr int SS = MODE == 1 ? MS3_S : MS3_S2;  // ring stages per GEMM

struct EC {  // one expert chunk (<= 64 tokens); v3 adds inv (2^-p of each token, written by k_gather)
  int e, n_real, n_pad, pad;
  long long img_off, h_off;
  int tok[NMAX];
  float wt[NMAX];
  float inv[NMAX];
};
static_assert(sizeof(EC) == 32 + 3 * 4 * NMAX, "EC layout must match bench.py (ec_inv=True)");
struct Tile {
  int ec, q, e, npad;
  long long boff;
};
static_assert(sizeof(Tile) == 24, "Tile layout must match bench.py");
struct Hdr {
  int tile, ch, npad, nch;
};

template <int MODE>
struct Cfg;
template <>
struct Cfg<1> {  // GEMM1: K = 5120 = 80 k64 steps; 8 per stage (n_pad <= 32) or 4
  static constexpr int KB = 80, QPE = 9, NSMALL = 32, CHS = MS3_CHS1, CHB = MS3_CHS1 / 2;
};
template <>
struct Cfg<2> {  // GEMM2: K = 576 = 9 k64 steps; the whole tile per stage (n_pad <= 24) or 3
  static constexpr int KB = 9, QPE = 40, NSMALL = 24, CHS = MS3_CHS2, CHB = MS3_CHB2;
};
template <int MODE>
struct Lay {
  using C = Cfg<MODE>;
  static constexpr int B_MAX = (C::CHS * C::NSMALL > C::CHB * NMAX ? C::CHS * C::NSMALL : C::CHB * NMAX) * 128;
  static constexpr int A_MAX = C::CHS * 4096, S_MAX = C::CHS * 256;
  static constexpr int OFF_A = B_MAX, OFF_S = B_MAX + A_MAX, OFF_H = OFF_S + S_MAX;
  static constexpr int STAGE = ((OFF_H + HDR) + 1023) / 1024 * 1024;
  static constexpr int STG = (MODE == 2 && MS3_YSTG) ? NCW * 16 * NMAX * 2 : 0;  // bf16 Y staging [token][16 rows]
  static constexpr int SMEM = 1024 + SS<MODE> * STAGE + 1024 + STG;
};
static_assert(Lay<1>::SMEM <= 227 * 1024 / MS3_CTAS - 1024 * (MS3_CTAS - 1) && Lay<2>::SMEM <= 227 * 1024 / MS3_CTAS - 1024 * (MS3_CTAS - 1),
              "ring must fit in SMEM (per CTA)");

// ------------------------------------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t a, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, unsigned b) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(b) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W_%=;\n}\n" ::"r"(
                   a),
               "r"(parity)
               : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t a, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(a), "r"(parity) : "memory");
  return ok;
}
__device__ __forceinline__ void mbar_wait_sleep(uint32_t a, uint32_t parity, unsigned ns) {
  while (!mbar_try(a, parity)) __nanosleep(ns);
}
__device__ __forceinline__ uint64_t pol_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t pol_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, unsigned bytes, uint32_t mbar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(
          dst),
      "l"(src), "r"(bytes), "r"(mbar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void tma_row(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t mbar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
      "l"(tm), "r"(c0), "r"(c1), "r"(mbar), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_arrive(uint32_t mbar) {  // arrives when this thread's cp.asyncs land
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(mbar) : "memory");
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ unsigned lds32(uint32_t a) {
  unsigned v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], uint32_t a) {
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}
__device__ __forceinline__ void mma16816(float* d, const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void red_v2(float* p, float x, float y) {  // p 8-byte aligned
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(p), "f"(x), "f"(y) : "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {  // K-major, 128-B rows, 128B swizzle, SBO 1024
  return ((uint64_t(saddr) & 0x3FFFFull) >> 4) | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 62);
}
__device__ __forceinline__ unsigned fp4_lo_e4m3(unsigned w) { return ((w << 4) & 0x80808080u) | ((w << 2) & 0x1c1c1c1cu); }
__device__ __forceinline__ unsigned fp4_hi_e4m3(unsigned w) { return (w & 0x80808080u) | ((w >> 2) & 0x1c1c1c1cu); }
__device__ __forceinline__ unsigned cvt_e4m3x2(unsigned v16) {
  unsigned h;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)v16));
  return h;
}
__device__ __forceinline__ unsigned hmul2(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ unsigned swz(int n, int b) {
  return (n >> 3) * 1024 + (n & 7) * 128 + ((((b >> 4) ^ (n & 7)) & 7) << 4) + (b & 15);
}
#ifndef MS3_CVT2
#define MS3_CVT2 1  // 1: both e4m3 pairs of a register in one asm block (mov.b32 {l, h}): no shift for the high half
#endif
// e4m3x4 register -> two f16x2 (low pair, high pair)
__device__ __forceinline__ void cvt_e4m3x4(unsigned v, unsigned& lo2, unsigned& hi2) {
  if (MS3_CVT2) {
    asm("{\n.reg .b16 l, h;\nmov.b32 {l, h}, %2;\ncvt.rn.f16x2.e4m3x2 %0, l;\ncvt.rn.f16x2.e4m3x2 %1, h;\n}"
        : "=r"(lo2), "=r"(hi2) : "r"(v));
  } else {
    lo2 = cvt_e4m3x2(v & 0xffff);
    hi2 = cvt_e4m3x2(v >> 16);
  }
}
// one word (8 FP4 codes of rows g / g+8, k 2t..2t+1, 2t+8..2t+9) -> the m16n8k16 / RS-wgmma A fragment
__device__ __forceinline__ void dequant_word(unsigned (&a)[4], unsigned w, unsigned mg, unsigned mg8) {
  const unsigned lo = fp4_lo_e4m3(w), hi = fp4_hi_e4m3(w);
  unsigned l0, l1, h0, h1;
  cvt_e4m3x4(lo, l0, l1);
  cvt_e4m3x4(hi, h0, h1);
  a[0] = hmul2(l0, mg);   // row g,   k 2t..2t+1
  a[2] = hmul2(l1, mg);   // row g,   k 2t+8..
  a[1] = hmul2(h0, mg8);  // row g+8, k 2t..
  a[3] = hmul2(h1, mg8);  // row g+8, k 2t+8..
}

// ------------------------------------------------------------------------------------------------ small kernels
__global__ void k_prescale(const __nv_bfloat16* __restrict__ x, int* __restrict__ ptok, float* __restrict__ inv) {
  const __nv_bfloat16* r = x + (size_t)blockIdx.x * D;
  float m = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) m = fmaxf(m, fabsf(__bfloat162float(r[i])));
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  __shared__ float wm[32];
  if (threadIdx.x % 32 == 0) wm[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < blockDim.x / 32; ++i) m = fmaxf(m, wm[i]);
    m = fmaxf(m, wm[0]);
    int p = 0;
    if (m > 0.f) {
      int e;
      frexpf(m, &e);
      p = 14 - e;
    }
    ptok[blockIdx.x] = p;
    inv[blockIdx.x] = exp2f((float)-p);
  }
}
// k_prescale16: per-token power of two p (as k_prescale) and the prescaled f16 copy x16 [T, D] (exact for elements
// >= 2^-28 of the token max) that GEMM1 gathers rows from with TMA (no per-expert token images).
__global__ void k_prescale16(const __nv_bfloat16* __restrict__ x, int* __restrict__ ptok, float* __restrict__ inv,
                             __half* __restrict__ x16) {
  const __nv_bfloat16* r = x + (size_t)blockIdx.x * D;
  __shared__ float wm[32];
  __shared__ float sc_s;
  float m = 0.f;
  for (int i = threadIdx.x * 8; i < D; i += blockDim.x * 8) {
    const uint4 v = *(const uint4*)(r + i);
    const __nv_bfloat162* h = (const __nv_bfloat162*)&v;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 f = __bfloat1622float2(h[k]);
      m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  if (threadIdx.x % 32 == 0) wm[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < blockDim.x / 32; ++i) m = fmaxf(m, wm[i]);
    m = fmaxf(m, wm[0]);
    int p = 0;
    if (m > 0.f) {
      int e;
      frexpf(m, &e);
      p = 14 - e;
    }
    ptok[blockIdx.x] = p;
    inv[blockIdx.x] = exp2f((float)-p);
    sc_s = exp2f((float)p);
  }
  __syncthreads();
  const float sc = sc_s;
  for (int i = threadIdx.x * 8; i < D; i += blockDim.x * 8) {
    const uint4 v = *(const uint4*)(r + i);
    const __nv_bfloat162* h = (const __nv_bfloat162*)&v;
    uint4 o;
    __half2* o2 = (__half2*)&o;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 f = __bfloat1622float2(h[k]);
      o2[k] = __floats2half2_rn(f.x * sc, f.y * sc);
    }
    *(uint4*)(x16 + (size_t)blockIdx.x * D + i) = o;
  }
}
__global__ void k_gather(const __nv_bfloat16* __restrict__ x, EC* __restrict__ ecs, const int* __restrict__ ptok,
                         char* __restrict__ img, const int* __restrict__ nec_dev = nullptr) {
  if (nec_dev && (int)blockIdx.x >= *nec_dev) return;
  EC& ec = ecs[blockIdx.x];
  for (int n = blockIdx.y; n < ec.n_pad; n += gridDim.y) {
  int tok = ec.tok[n], slab = threadIdx.x >> 3, cc = threadIdx.x & 7;
  if (threadIdx.x == 0) ec.inv[n] = tok >= 0 ? exp2f((float)-ptok[tok]) : 0.f;
  uint4 v = make_uint4(0, 0, 0, 0);
  if (tok >= 0) {
    float sc = exp2f((float)ptok[tok]);
    const __nv_bfloat16* src = x + (size_t)tok * D + slab * 64 + cc * 8;
    __half h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = __float2half_rn(__bfloat162float(src[i]) * sc);
    v = *(uint4*)h;
  }
  char* dst = img + ec.img_off + (size_t)slab * ec.n_pad * 128 + swz(n, cc * 16);
  *(uint4*)dst = v;
  }
}

// ------------------------------------------------------------------------------------------------ GEMM kernels
template <int MODE>
struct Args {
  const unsigned* A;        // repacked words (moe_stream layout)
  const unsigned char* Sc;  // v3 scales: [e][q][kb][64 slots][4 B]
  const EC* ecs;
  const Tile* tiles;
  int ntiles;
  const char* bimg;  // GEMM1: x image; GEMM2: H image
  char* himg;        // GEMM1 output
  const float* inv;
  float* out;
  const int* nt_dev = nullptr;     // device tile count (device routing); nullptr = ntiles
  __nv_bfloat16* yout = nullptr;   // GEMM2: bf16 output per (EC slot) [slots][D], plain stores (EC.pad = slot base)
  int gather = 0;                  // GEMM1 token rows from x16 instead of per-EC images: 1 = TMA tensor row loads
                                   // (correct, 3.4x slower), 2 = cp.async 16 B by the producer warp (default with device routing)
  int zrow = 0;                    // x16 row of zeros for padding slots (= T)
  alignas(64) CUtensorMap tmx;     // x16 [T + 1, D] f16, box 64 x 1, SWIZZLE_128B
};

template <int MODE, int NT>
__device__ __forceinline__ void run_tile(const Args<MODE>& a, char* ring, uint64_t* full, uint64_t* empty, long& it,
                                         int tile, int c, int t) {
  using C = Cfg<MODE>;
  using Ly = Lay<MODE>;
  constexpr int N = NT * 8;
  constexpr int CH = N <= C::NSMALL ? C::CHS : C::CHB;
  constexpr int NCH = C::KB / CH;
  constexpr bool MMA = N <= MS3_MMA_MAXN;
  static_assert(C::KB % CH == 0, "stages must tile K");
  const int w = t / 32, lane = t % 32, g = lane / 4, tq = lane % 4, R = 64 * c + 16 * w + g;
  const int slot = 32 * c + 8 * w + g;
  // epilogue metadata: from the EC copy in the tile's first stage (MS3_EPF) or, as before, from global at the end
  int etok[NT][2];
  float ewt[NT][2], einv[NT][2];
  long long eh_off = 0;
  int eslot = 0, enr = 0;
  if (MS3_EPF) {
    const EC* e0 = (const EC*)(ring + (size_t)(it % SS<MODE>) * Ly::STAGE + Ly::OFF_H + 128);
#pragma unroll
    for (int nb = 0; nb < NT; ++nb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nb + 2 * tq + hh;
        etok[nb][hh] = e0->tok[n];
        ewt[nb][hh] = e0->wt[n];
        einv[nb][hh] = e0->inv[n];
      }
    eh_off = e0->h_off;
    eslot = e0->pad;
    enr = e0->n_real;
  }
  float acc[2][N / 2];
#pragma unroll
  for (int q = 0; q < 2; ++q)
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[q][i] = 0.f;
  for (int ch = 0; ch < NCH; ++ch) {
    const int s = it % SS<MODE>;
    if (ch > 0) mbar_wait(smem_u32(full + s), (it / SS<MODE>) & 1);
    char* st = ring + (size_t)s * Ly::STAGE;
    const uint32_t aB = smem_u32(st);
    if (MS3_ABL != 1) {
      if constexpr (MMA) {
#pragma unroll
        for (int kb = 0; kb < CH; ++kb) {
          const unsigned sc = *(const unsigned*)(st + Ly::OFF_S + kb * 256 + slot * 4);
          const unsigned m0 = __byte_perm(sc, 0, 0x0404), m08 = __byte_perm(sc, 0, 0x1414);
          const unsigned m1 = __byte_perm(sc, 0, 0x2424), m18 = __byte_perm(sc, 0, 0x3434);
          const uint4 wv = *(const uint4*)(st + Ly::OFF_A + kb * 4096 + (c * 128 + t) * 16);
          const unsigned ws[4] = {wv.x, wv.y, wv.z, wv.w};
          unsigned bf[NT][2][4];
          const uint32_t slab = aB + kb * N * 128;
#pragma unroll
          for (int nb = 0; nb < NT; ++nb)
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
              ldsm_x4(bf[nb][kk], slab + swz(nb * 8 + (lane & 7), (kk * 4 + (lane >> 3)) * 16));
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            unsigned fr[4];
            dequant_word(fr, ws[j], j < 2 ? m0 : m1, j < 2 ? m08 : m18);
#pragma unroll
            for (int nb = 0; nb < NT; ++nb)
              mma16816(&acc[j & 1][nb * 4], fr, bf[nb][j / 2][(j & 1) * 2], bf[nb][j / 2][(j & 1) * 2 + 1]);
          }
        }
      } else {
        unsigned A[2][4][4];
#pragma unroll
        for (int kb = 0; kb < CH; ++kb) {
          const int set = kb & 1;
          if (kb >= 2) wg_wait<1>();  // the group that read A[set] (kb - 2) is done
          const unsigned sc = *(const unsigned*)(st + Ly::OFF_S + kb * 256 + slot * 4);
          const unsigned m0 = __byte_perm(sc, 0, 0x0404), m08 = __byte_perm(sc, 0, 0x1414);
          const unsigned m1 = __byte_perm(sc, 0, 0x2424), m18 = __byte_perm(sc, 0, 0x3434);
          const uint4 wv = *(const uint4*)(st + Ly::OFF_A + kb * 4096 + (c * 128 + t) * 16);
          const unsigned ws[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
          for (int j = 0; j < 4; ++j) dequant_word(A[set][j], ws[j], j < 2 ? m0 : m1, j < 2 ? m08 : m18);
          if (MODE == 1 && kb == 0 && a.gather == 2) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          wg_fence();
          const uint64_t db = desc_sw128(aB + kb * N * 128);
#pragma unroll
          for (int j = 0; j < 4; ++j) WgRS<N>::mma(acc[0], A[set][j], db + 2 * j, 1);
          wg_commit();
        }
        wg_wait<0>();
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_u32(empty + s));
    ++it;
  }
  if (MMA) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) acc[0][i] += acc[1][i];
  }
  // ---- epilogue. acc[0][4 nb + {0,1,2,3}] = (row g, col 8nb+2tq), (g, +1), (g+8, 2tq), (g+8, +1)
  const Tile tl = a.tiles[tile];
  if (!MS3_EPF) {
    const EC& ec = a.ecs[tl.ec];
#pragma unroll
    for (int nb = 0; nb < NT; ++nb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nb + 2 * tq + hh;
        etok[nb][hh] = ec.tok[n];
        ewt[nb][hh] = ec.wt[n];
        einv[nb][hh] = etok[nb][hh] >= 0 ? a.inv[etok[nb][hh]] : 0.f;
      }
    eh_off = ec.h_off;
    eslot = ec.pad;
    enr = ec.n_real;
  }
  if (MODE == 1) {
    const int il = 32 * c + 8 * w + g;
    char* hs = a.himg + eh_off + (size_t)tl.q * N * 128;
#pragma unroll
    for (int nb = 0; nb < NT; ++nb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nb + 2 * tq + hh;
        const int tok = etok[nb][hh];
        float h = 0.f;
        if (tok >= 0) {
          const float iv = einv[nb][hh];
          float gt = fminf(acc[0][4 * nb + hh] * iv, 10.f);
          float up = fminf(fmaxf(acc[0][4 * nb + 2 + hh] * iv, -10.f), 10.f);
          h = gt / (1.f + expf(-gt)) * up * ewt[nb][hh] * 256.f;
        }
        *(__half*)(hs + swz(n, il * 2)) = __float2half_rn(h);
      }
  } else {
    const int r0 = 128 * tl.q + R;
    if (a.yout) {
      // Device routing: bf16 expert output per (token, slot) at stock's rounding point, no atomics, deterministic. Lanes
      // g / g^1 swap one value so each writes a 4-byte (row pair) store: even lane token n0 = 8nb + 2tq, odd n0 + 1.
      // (Staging through SMEM for full-sector stores measured 206 vs 199 us: not worth it, MS3_YSTG=1 keeps it.)
      if (MS3_YSTG) {
        __nv_bfloat16* stg = (__nv_bfloat16*)(ring + SS<MODE> * Ly::STAGE + 1024) + (c * 4 + w) * 16 * NMAX;
#pragma unroll
        for (int nb = 0; nb < NT; ++nb)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int n = 8 * nb + 2 * tq + hh;
            stg[n * 16 + g] = __float2bfloat16_rn(acc[0][4 * nb + hh] * (1.f / 256.f));
            stg[n * 16 + g + 8] = __float2bfloat16_rn(acc[0][4 * nb + 2 + hh] * (1.f / 256.f));
          }
        __syncwarp();
        __nv_bfloat16* yb = a.yout + (size_t)eslot * D + 128 * tl.q + 64 * c + 16 * w;
        for (int ch = lane; ch < 2 * enr; ch += 32) {
          const int n = ch >> 1, hf = ch & 1;
          *(uint4*)(yb + (size_t)n * D + 8 * hf) = *(const uint4*)(stg + n * 16 + 8 * hf);
        }
        __syncwarp();
      } else {
        const bool odd = g & 1;
        const int rb = 128 * tl.q + (R & ~1);
#pragma unroll
        for (int nb = 0; nb < NT; ++nb)
#pragma unroll
          for (int rs = 0; rs < 2; ++rs) {
            const float v0 = acc[0][4 * nb + 2 * rs], v1 = acc[0][4 * nb + 2 * rs + 1];
            const float got = __shfl_xor_sync(0xffffffffu, odd ? v0 : v1, 4);
            const int n = 8 * nb + 2 * tq + (odd ? 1 : 0);
            if (etok[nb][odd ? 1 : 0] >= 0)
              *(__nv_bfloat162*)(a.yout + (size_t)(eslot + n) * D + rb + 8 * rs) =
                  __floats2bfloat162_rn((odd ? got : v0) * (1.f / 256.f), (odd ? v1 : got) * (1.f / 256.f));
          }
      }
    } else if (MS3_RED2) {
      // rows R and R^1 live in lanes g and g^1 (same tq). The even lane takes token n0 = 8nb + 2tq, the odd lane token
      // n0 + 1; each sends the value the other needs and issues one 8-byte red per row set (R, R+8).
      const bool odd = g & 1;
      const int rb = 128 * tl.q + (R & ~1);
#pragma unroll
      for (int nb = 0; nb < NT; ++nb)
#pragma unroll
        for (int rs = 0; rs < 2; ++rs) {  // rs 0: rows R / R^1, rs 1: rows R+8 / (R+8)^1
          const float v0 = acc[0][4 * nb + 2 * rs], v1 = acc[0][4 * nb + 2 * rs + 1];  // (row, n0), (row, n0 + 1)
          const float got = __shfl_xor_sync(0xffffffffu, odd ? v0 : v1, 4);
          const int tok = etok[nb][odd ? 1 : 0];
          if (tok >= 0 && MS3_ABL != 2)
            red_v2(a.out + (size_t)tok * D + rb + 8 * rs, (odd ? got : v0) * (1.f / 256.f), (odd ? v1 : got) * (1.f / 256.f));
        }
    } else {
#pragma unroll
      for (int nb = 0; nb < NT; ++nb)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int tok = etok[nb][hh];
          if (tok >= 0 && MS3_ABL != 2) {  // MS3_ABL 2: GEMM2 without its output atomics (timing only)
            atomicAdd(a.out + (size_t)tok * D + r0, acc[0][4 * nb + hh] * (1.f / 256.f));
            atomicAdd(a.out + (size_t)tok * D + r0 + 8, acc[0][4 * nb + 2 + hh] * (1.f / 256.f));
          }
        }
    }
  }
}

template <int MODE>
__global__ void __launch_bounds__(32 + 32 * NCW, MS3_CTAS) k_gemm(const __grid_constant__ Args<MODE> a) {
  using C = Cfg<MODE>;
  using Ly = Lay<MODE>;
  extern __shared__ __align__(1024) char smem_raw[];
  // align without a uintptr_t round trip, so the compiler keeps the shared address space (LDS, not generic LD)
  char* ring = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  uint64_t* full = (uint64_t*)(ring + SS<MODE> * Ly::STAGE);
  uint64_t* empty = full + SS<MODE>;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    const unsigned fa = (MODE == 1 && a.gather == 2) ? 33u : 1u;  // + 32 cp.async arrivals per stage
    for (int s = 0; s < SS<MODE>; ++s) mbar_init(smem_u32(full + s), fa), mbar_init(smem_u32(empty + s), NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == NCW) {  // ---------------------------------------------------------------- producer (last warp)
    // lane 0 does the ring control and the bulk copies; with a.gather all 32 lanes also issue the token-row TMA loads
    if (lane != 0 && !(MODE == 1 && a.gather)) return;
    const uint64_t pw = pol_first(), pb = pol_last();
    long it = 0;
    const int ntiles = a.nt_dev ? *a.nt_dev : a.ntiles;
    int tile = blockIdx.x;
    Tile tl = tile < ntiles ? a.tiles[tile] : Tile{0, 0, 0, 0, 0};
    const bool gat = MODE == 1 && a.gather;
    int tokA = 0, tokB = 0;  // this lane's token slots lane, lane + 32 of the current tile (gather)
    if (gat && tile < ntiles) tokA = a.ecs[tl.ec].tok[lane], tokB = a.ecs[tl.ec].tok[lane + 32];
    for (;;) {
      const bool end = tile >= ntiles;
      const int ntile = tile + gridDim.x;
      const Tile nx = ntile < ntiles ? a.tiles[ntile] : Tile{0, 0, 0, 0, 0};
      int nA = 0, nB = 0;
      if (gat && ntile < ntiles) nA = a.ecs[nx.ec].tok[lane], nB = a.ecs[nx.ec].tok[lane + 32];
      const int npad = end ? 0 : tl.npad;
      if (!end && (npad < 8 || npad > NMAX || (npad & 7))) __trap();
      const int CH = npad <= C::NSMALL ? C::CHS : C::CHB;
      const int nch = end ? 1 : C::KB / CH;
      for (int ch = 0; ch < nch; ++ch, ++it) {
        const int s = it % SS<MODE>;
        if (it >= SS<MODE>) {
          if (MS3_PSLEEP) mbar_wait_sleep(smem_u32(empty + s), ((it / SS<MODE>) + 1) & 1, MS3_PSLEEP);
          else mbar_wait(smem_u32(empty + s), ((it / SS<MODE>) + 1) & 1);
        }
        char* st = ring + (size_t)s * Ly::STAGE;
        const uint32_t fb = smem_u32(full + s), dst = smem_u32(st);
        if (lane == 0) {
          Hdr* h = (Hdr*)(st + Ly::OFF_H);
          h->tile = end ? -1 : tile;
          h->ch = ch;
          h->npad = npad;
          h->nch = nch;
        }
        if (end) {
          if (lane == 0) mbar_arrive(fb);
          if (gat && a.gather == 2) mbar_arrive(fb);  // the 32 cp.async arrivals the barrier expects
          break;
        }
        const size_t blk = (size_t)tl.e * C::QPE + tl.q, k0 = blk * C::KB + (size_t)ch * CH;
        const unsigned bA = CH * 4096, bS = CH * 256, bB = CH * npad * 128;
        const unsigned bE = (MS3_EPF && ch == 0) ? (unsigned)sizeof(EC) : 0u;
        if (lane == 0) {
          mbar_expect_tx(fb, bA + bS + (gat && a.gather == 2 ? 0u : bB) + bE);
          if (bE) bulk(dst + Ly::OFF_H + 128, a.ecs + tl.ec, bE, fb, pb);
          bulk(dst + Ly::OFF_A, (const char*)a.A + k0 * 4096, bA, fb, pw);
          bulk(dst + Ly::OFF_S, a.Sc + k0 * 256, bS, fb, pw);
          if (!gat) bulk(dst, a.bimg + tl.boff + (size_t)ch * CH * npad * 128, bB, fb, pb);
        }
        if (gat) {  // token rows: slab kb, row n <- x16[tok][64 (ch CH + kb) .. +64], swizzled by the TMA unit
          __syncwarp();
          const int nops = CH * npad;
          for (int j0 = 0; j0 < nops; j0 += 32) {
            const int j = j0 + lane, kb = j / npad, n = j - kb * npad;
            const int ta = __shfl_sync(0xffffffffu, tokA, n & 31), tb = __shfl_sync(0xffffffffu, tokB, n & 31);
            const int tok = n < 32 ? ta : tb;
            if (j < nops) {
              const int row = tok >= 0 ? tok : a.zrow;
              if (a.gather == 1) {
                tma_row(dst + (unsigned)((kb * npad + n) * 128), &a.tmx, 64 * (ch * CH + kb), row, fb, pb);
              } else {  // 8 x 16 B into the 128B-swizzled slot of (slab kb, row n)
                const char* src = (const char*)a.bimg + ((size_t)row * D + 64 * (ch * CH + kb)) * 2;
                const uint32_t slab = dst + (unsigned)(kb * npad * 128);
#pragma unroll
                for (int cc = 0; cc < 8; ++cc) cp_async16(slab + swz(n, cc * 16), src + cc * 16);
              }
            }
          }
          if (a.gather == 2) cp_async_arrive(fb);
          __syncwarp();
        }
      }
      if (end) break;
      tile = ntile;
      tl = nx;
      tokA = nA, tokB = nB;
    }
    return;
  }
  // ------------------------------------------------------------------ consumers (warps 0..7 = warpgroups 0, 1)
  const int ct = threadIdx.x, c = ct / 128, t = ct % 128;
  long it = 0;
  for (;;) {
    const int s = it % SS<MODE>;
    mbar_wait(smem_u32(full + s), (it / SS<MODE>) & 1);
    const Hdr h = *(const Hdr*)(ring + (size_t)s * Ly::STAGE + Ly::OFF_H);
    if (h.tile < 0) break;
    switch (h.npad / 8) {
      case 1: run_tile<MODE, 1>(a, ring, full, empty, it, h.tile, c, t); break;
      case 2: run_tile<MODE, 2>(a, ring, full, empty, it, h.tile, c, t); break;
      case 3: run_tile<MODE, 3>(a, ring, full, empty, it, h.tile, c, t); break;
      case 4: run_tile<MODE, 4>(a, ring, full, empty, it, h.tile, c, t); break;
      case 5: run_tile<MODE, 5>(a, ring, full, empty, it, h.tile, c, t); break;
      case 6: run_tile<MODE, 6>(a, ring, full, empty, it, h.tile, c, t); break;
      case 7: run_tile<MODE, 7>(a, ring, full, empty, it, h.tile, c, t); break;
      default: run_tile<MODE, 8>(a, ring, full, empty, it, h.tile, c, t); break;
    }
  }
}


// ================================================================================================ routing tables
// k_route: device routing tables from the router's top-6 (ids [T, 6] int32, w [T, 6] fp32), one CTA, deterministic.
//   1. stable radix sort of the P = 6T (token, slot) pairs by expert (CUB): sorted position i = the pair's Y slot;
//   2. per-expert counts and segments, experts split into chunks of <= 64 tokens (EC records, in expert order);
//   3. EC prefix sums of n_pad -> token-image (80 slabs) and H-image (9 slabs) offsets;
//   4. tiles (GEMM1: 9 per EC, GEMM2: 40) in snake order: heaviest n_pad first, every other round of `grid` reversed;
//   5. slotmap[p] = sorted position (for the combine), counts = {n_ec, ntiles1, ntiles2, img bytes, h bytes, P}.
constexpr int RT_THREADS = 1024, RT_ITEMS = 6, RT_MAXP = RT_THREADS * RT_ITEMS, NEXP = 384;
constexpr int RT_MAXEC = NEXP + RT_MAXP / NMAX;  // 480: sum over experts of ceil(c_e / 64)
using RtSort = cub::BlockRadixSort<int, RT_THREADS, RT_ITEMS, int>;
using RtScan = cub::BlockScan<int, RT_THREADS>;
struct RtSmem {
  union {
    typename RtSort::TempStorage sort;
    typename RtScan::TempStorage scan;
  } t;
  int sv[RT_MAXP];  // sorted pair index p = 6 t + k
  int cnt[NEXP], seg[NEXP], ecb[NEXP];
  int ec_e[RT_MAXEC], ec_c[RT_MAXEC], ec_np[RT_MAXEC], ec_pre[RT_MAXEC], ec_pos[RT_MAXEC];
  int n_ec;
};

// global scratch written by k_route for k_route_fill (int32 offsets)
constexpr int RS_SV = 0, RS_EE = RS_SV + RT_MAXP, RS_EK = RS_EE + RT_MAXEC, RS_ENP = RS_EK + RT_MAXEC,
              RS_EPRE = RS_ENP + RT_MAXEC, RS_EPOS = RS_EPRE + RT_MAXEC, RS_CNT = RS_EPOS + RT_MAXEC, RS_SEG = RS_CNT + NEXP,
              RS_NEC = RS_SEG + NEXP, RS_INTS = RS_NEC + 32;

__global__ void __launch_bounds__(RT_THREADS, 1) k_route(const int* __restrict__ ids, int T, int* __restrict__ rts,
                                                         int* __restrict__ counts, int* __restrict__ slotmap) {
  extern __shared__ __align__(16) char rt_raw[];
  RtSmem& sm = *reinterpret_cast<RtSmem*>(rt_raw);
  const int tid = threadIdx.x, P = 6 * T;
  int key[RT_ITEMS], val[RT_ITEMS];
#pragma unroll
  for (int r = 0; r < RT_ITEMS; ++r) {
    const int p = tid * RT_ITEMS + r;
    key[r] = p < P ? ids[p] : 1023;  // padding sorts last
    val[r] = p;
  }
  for (int e = tid; e < NEXP; e += RT_THREADS) sm.cnt[e] = 0;
  RtSort(sm.t.sort).Sort(key, val, 0, 10);  // stable: within an expert the pairs stay in (token, slot) order
  __syncthreads();
#pragma unroll
  for (int r = 0; r < RT_ITEMS; ++r) {
    const int i = tid * RT_ITEMS + r;  // blocked arrangement
    if (i < P) {
      sm.sv[i] = val[r];
      rts[RS_SV + i] = val[r];
      slotmap[val[r]] = i;
      atomicAdd(&sm.cnt[key[r]], 1);  // the counts do not depend on the atomic order
    }
  }
  __syncthreads();
  {  // segments and EC bases: exclusive scans over the experts
    const int c = tid < NEXP ? sm.cnt[tid] : 0, nc = (c + NMAX - 1) / NMAX;
    int sc, snc, tot;
    RtScan(sm.t.scan).ExclusiveSum(c, sc);
    __syncthreads();
    RtScan(sm.t.scan).ExclusiveSum(nc, snc, tot);
    if (tid < NEXP) sm.seg[tid] = sc, sm.ecb[tid] = snc;
    if (tid == 0) sm.n_ec = tot;
    __syncthreads();
    if (tid < NEXP)
      for (int k = 0; k < nc; ++k) {
        const int x = snc + k, nr = min(NMAX, c - NMAX * k);
        sm.ec_e[x] = tid, sm.ec_c[x] = k, sm.ec_np[x] = (nr + 7) & ~7;
      }
  }
  __syncthreads();
  const int n_ec = sm.n_ec;
  {  // image offsets: exclusive scan of n_pad over the ECs
    const int np = tid < n_ec ? sm.ec_np[tid] : 0;
    int pre, tot;
    RtScan(sm.t.scan).ExclusiveSum(np, pre, tot);
    if (tid < n_ec) sm.ec_pre[tid] = pre;
    if (tid == 0) counts[3] = tot * 80 * 128, counts[4] = tot * 9 * 128;
  }
  __syncthreads();
  if (tid < n_ec) {  // stable rank by n_pad descending (8 buckets)
    const int np = sm.ec_np[tid];
    int before = 0;
    for (int x = 0; x < n_ec; ++x) {
      const int o = sm.ec_np[x];
      before += (o > np) || (o == np && x < tid);
    }
    sm.ec_pos[tid] = before;
  }
  __syncthreads();
  // hand the small per-EC tables to k_route_fill (many CTAs: one SM writing ~0.5 MB of ECs and tiles took ~30 us)
  for (int x = tid; x < n_ec; x += RT_THREADS) {
    rts[RS_EE + x] = sm.ec_e[x], rts[RS_EK + x] = sm.ec_c[x], rts[RS_ENP + x] = sm.ec_np[x];
    rts[RS_EPRE + x] = sm.ec_pre[x], rts[RS_EPOS + x] = sm.ec_pos[x];
  }
  for (int e = tid; e < NEXP; e += RT_THREADS) rts[RS_CNT + e] = sm.cnt[e], rts[RS_SEG + e] = sm.seg[e];
  if (tid == 0) {
    rts[RS_NEC] = n_ec;
    counts[0] = n_ec, counts[1] = 9 * n_ec, counts[2] = 40 * n_ec, counts[5] = P;
  }
}

// k_route_fill: EC records (64 token slots each) and the snake-ordered GEMM1 / GEMM2 tiles, grid-stride over all CTAs.
__global__ void __launch_bounds__(256) k_route_fill(const int* __restrict__ rts, const float* __restrict__ w,
                                                    EC* __restrict__ ecs, Tile* __restrict__ t1, Tile* __restrict__ t2,
                                                    int grid, const float* __restrict__ inv) {
  const int n_ec = rts[RS_NEC], stride = gridDim.x * blockDim.x;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_ec * (NMAX + 49); j += stride) {
    if (j < n_ec * NMAX) {  // EC token slots
      const int x = j / NMAX, n = j % NMAX, e = rts[RS_EE + x], k = rts[RS_EK + x];
      const int nr = min(NMAX, rts[RS_CNT + e] - NMAX * k), base = rts[RS_SEG + e] + NMAX * k;
      EC& ec = ecs[x];
      const int p = n < nr ? rts[RS_SV + base + n] : -1;
      ec.tok[n] = p >= 0 ? p / 6 : -1;
      ec.wt[n] = p >= 0 ? w[p] : 0.f;
      ec.inv[n] = (inv && p >= 0) ? inv[p / 6] : 0.f;
      if (n == 0) {
        const long long pre = rts[RS_EPRE + x];
        ec.e = e, ec.n_real = nr, ec.n_pad = rts[RS_ENP + x], ec.pad = base;
        ec.img_off = pre * 80 * 128, ec.h_off = pre * 9 * 128;
      }
    } else {  // tiles: 9 GEMM1 + 40 GEMM2 per EC, heaviest first, alternate rounds reversed
      const int jj = j - n_ec * NMAX, x = jj / 49, q = jj % 49, e = rts[RS_EE + x], np = rts[RS_ENP + x];
      const long long pre = rts[RS_EPRE + x];
      const int pos0 = rts[RS_EPOS + x];
      if (q < 9) {
        const int u = pos0 * 9 + q, n1 = 9 * n_ec, r = u / grid, pos = u % grid, L = min(grid, n1 - r * grid);
        t1[r * grid + ((r & 1) ? L - 1 - pos : pos)] = Tile{x, q, e, np, pre * 80 * 128};
      } else {
        const int qq = q - 9, u = pos0 * 40 + qq, n2 = 40 * n_ec, r = u / grid, pos = u % grid,
                  L = min(grid, n2 - r * grid);
        t2[r * grid + ((r & 1) ? L - 1 - pos : pos)] = Tile{x, qq, e, np, pre * 9 * 128};
      }
    }
  }
}

// k_router_top: router epilogue on the fp32 gate logits [T, 384] (bf16 x bf16 GEMM, fp32 accumulate: exact products,
// as ref gate_ref): scores = sqrt(softplus(logit)) (torch softplus: threshold 20), top-6 of scores + bias (ties: lower
// expert id), weights = scores[sel] / (sum + 1e-20) * route_scale. One warp per token; ids in descending order.
__global__ void k_router_top(const float* __restrict__ logits, const float* __restrict__ bias, int T,
                             int* __restrict__ ids, float* __restrict__ wts, float route_scale) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= T) return;
  float sc[12], sel[12];
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    const int e = i * 32 + lane;
    const float x = logits[(size_t)warp * NEXP + e];
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
      if (!((taken >> i) & 1) && sel[i] > bv) bv = sel[i], bi = i * 32 + lane;  // within a lane: ascending ids
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
  if (lane == 0) {
    float sum = 0.f;
#pragma unroll
    for (int r = 0; r < 6; ++r) sum += pick[r];
#pragma unroll
    for (int r = 0; r < 6; ++r) {
      ids[warp * 6 + r] = pid[r];
      wts[warp * 6 + r] = pick[r] / (sum + 1e-20f) * route_scale;
    }
  }
}

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

// k_combine: per token, the 6 bf16 expert outputs (slot order k = 0..5, fp32 sum) + the bf16 shared-expert output
// -> bf16 rank partial (stock: y = bf16(routed + shared)). grid T, block 640 x 8 elements.
__global__ void k_combine(const __nv_bfloat16* __restrict__ Y, const int* __restrict__ slotmap,
                          const __nv_bfloat16* __restrict__ sh, __nv_bfloat16* __restrict__ out) {
  const int t = blockIdx.x, d0 = threadIdx.x * 8;
  int sl[6];
#pragma unroll
  for (int k = 0; k < 6; ++k) sl[k] = slotmap[6 * t + k];
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  uint4 v[6];
#pragma unroll
  for (int k = 0; k < 6; ++k) v[k] = *(const uint4*)(Y + (size_t)sl[k] * D + d0);  // all loads before the sums
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    const __nv_bfloat162* h = (const __nv_bfloat162*)&v[k];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(h[i]);
      acc[2 * i] += f.x, acc[2 * i + 1] += f.y;
    }
  }
  const uint4 sv = *(const uint4*)(sh + (size_t)t * D + d0);
  const __nv_bfloat162* hs = (const __nv_bfloat162*)&sv;
  uint4 o;
  __nv_bfloat162* ho = (__nv_bfloat162*)&o;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(hs[i]);
    ho[i] = __floats2bfloat162_rn(acc[2 * i] + f.x, acc[2 * i + 1] + f.y);
  }
  *(uint4*)(out + (size_t)t * D + d0) = o;
}

}  // namespace ms3

// ------------------------------------------------------------------------------------------------ bindings
using namespace ms3;
static int g_sms = 0;

void prep(torch::Tensor x, torch::Tensor ecs, int64_t n_ec, torch::Tensor img, torch::Tensor ptok, torch::Tensor inv) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int T = x.size(0);
  k_prescale<<<T, 256, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), ptok.data_ptr<int>(), inv.data_ptr<float>());
  k_gather<<<dim3(n_ec, NMAX), 640, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (EC*)ecs.data_ptr(),
                                                ptok.data_ptr<int>(), (char*)img.data_ptr());
}

template <int MODE>
void gemm(torch::Tensor A, torch::Tensor Sc, torch::Tensor ecs, torch::Tensor tiles, torch::Tensor counter,
          torch::Tensor bimg, torch::Tensor himg, torch::Tensor inv, torch::Tensor out, int64_t grid) {
  (void)counter;  // static tile schedule (kept for the moe_stream call signature)
  auto stream = at::cuda::getCurrentCUDAStream();
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, A.get_device());
  Args<MODE> a{(const unsigned*)A.data_ptr(), (const unsigned char*)Sc.data_ptr(), (const EC*)ecs.data_ptr(),
               (const Tile*)tiles.data_ptr(), (int)tiles.size(0), (const char*)bimg.data_ptr(),
               MODE == 1 ? (char*)himg.data_ptr() : nullptr, inv.data_ptr<float>(),
               MODE == 2 ? out.data_ptr<float>() : nullptr};
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(k_gemm<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, Lay<MODE>::SMEM);
    attr = true;
  }
  k_gemm<MODE><<<grid > 0 ? (int)grid : g_sms * MS3_CTAS, 32 + 32 * NCW, Lay<MODE>::SMEM, stream>>>(a);
}

// Device-routing entry points (bf16 Y, combine); the moe_stream-compatible ones above stay for bench.py.
void route(torch::Tensor ids, torch::Tensor w, int64_t T, torch::Tensor ecs, torch::Tensor t1, torch::Tensor t2,
           torch::Tensor counts, torch::Tensor slotmap, torch::Tensor rts, int64_t grid, torch::Tensor inv) {
  TORCH_CHECK(T * 6 <= RT_MAXP, "route: T too large");
  TORCH_CHECK(rts.numel() >= RS_INTS, "route: scratch too small");
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(k_route, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(RtSmem));
    attr = true;
  }
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, ids.get_device());
  k_route<<<1, RT_THREADS, sizeof(RtSmem), stream>>>(ids.data_ptr<int>(), (int)T, rts.data_ptr<int>(), counts.data_ptr<int>(),
                                                     slotmap.data_ptr<int>());
  k_route_fill<<<2 * g_sms, 256, 0, stream>>>(rts.data_ptr<int>(), w.data_ptr<float>(), (EC*)ecs.data_ptr(), (Tile*)t1.data_ptr(),
                                              (Tile*)t2.data_ptr(), grid > 0 ? (int)grid : g_sms,
                                              inv.numel() ? inv.data_ptr<float>() : nullptr);
}
void swiglu(torch::Tensor gu, torch::Tensor h) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int I = (int)h.size(1);
  TORCH_CHECK(I % 8 == 0 && gu.size(1) == 2 * I, "swiglu shapes");
  k_swiglu<<<(unsigned)h.size(0), I / 8, 0, stream>>>((const __nv_bfloat16*)gu.data_ptr(), (__nv_bfloat16*)h.data_ptr(), I);
}
int64_t route_scratch_ints() { return RS_INTS; }
void prescale16(torch::Tensor x, torch::Tensor ptok, torch::Tensor inv, torch::Tensor x16) {
  auto stream = at::cuda::getCurrentCUDAStream();
  k_prescale16<<<(unsigned)x.size(0), 256, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), ptok.data_ptr<int>(),
                                                         inv.data_ptr<float>(), (__half*)x16.data_ptr());
}
static PFN_cuTensorMapEncodeTiled_v12000 tmap_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&fn, 12000, cudaEnableDefault, &q) ==
                        cudaSuccess && fn,
                "cuTensorMapEncodeTiled not available");
  }
  return fn;
}
// GEMM1 with TMA token-row gathers from x16 [rows, D] f16 (rows = T + 1; row T zeros for padding slots)
void gemm1g(torch::Tensor A, torch::Tensor Sc, torch::Tensor ecs, torch::Tensor tiles, torch::Tensor counts,
            torch::Tensor x16, torch::Tensor himg, torch::Tensor inv, int64_t grid, int64_t mode) {
  auto stream = at::cuda::getCurrentCUDAStream();
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, A.get_device());
  Args<1> a{(const unsigned*)A.data_ptr(), (const unsigned char*)Sc.data_ptr(), (const EC*)ecs.data_ptr(),
            (const Tile*)tiles.data_ptr(), 0, (const char*)x16.data_ptr(), (char*)himg.data_ptr(), inv.data_ptr<float>(),
            nullptr, counts.data_ptr<int>() + 1, nullptr};
  a.gather = (int)mode;
  a.zrow = (int)x16.size(0) - 1;
  static void* cached_ptr = nullptr;
  static int64_t cached_rows = -1;
  static CUtensorMap cached;
  if (cached_ptr != x16.data_ptr() || cached_rows != x16.size(0)) {
    cuuint64_t dims[2] = {(cuuint64_t)D, (cuuint64_t)x16.size(0)};
    cuuint64_t strides[1] = {(cuuint64_t)D * 2};
    cuuint32_t box[2] = {64, 1}, es[2] = {1, 1};
    CUresult r = tmap_encode()(&cached, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, x16.data_ptr(), dims, strides, box, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    cached_ptr = x16.data_ptr(), cached_rows = x16.size(0);
  }
  a.tmx = cached;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(k_gemm<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, Lay<1>::SMEM);
    attr = true;
  }
  k_gemm<1><<<grid > 0 ? (int)grid : g_sms * MS3_CTAS, 32 + 32 * NCW, Lay<1>::SMEM, stream>>>(a);
}
void router_top(torch::Tensor logits, torch::Tensor bias, torch::Tensor ids, torch::Tensor wts, double route_scale) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int T = (int)logits.size(0);
  TORCH_CHECK(logits.size(1) == NEXP, "router_top: 384 experts");
  k_router_top<<<(T + 7) / 8, 256, 0, stream>>>(logits.data_ptr<float>(), bias.data_ptr<float>(), T, ids.data_ptr<int>(),
                                                wts.data_ptr<float>(), (float)route_scale);
}
void prepx(torch::Tensor x, torch::Tensor ecs, torch::Tensor counts, int64_t max_ec, torch::Tensor img, torch::Tensor ptok,
           torch::Tensor inv) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int T = x.size(0);
  k_prescale<<<T, 256, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), ptok.data_ptr<int>(), inv.data_ptr<float>());
  k_gather<<<dim3((unsigned)max_ec, 8), 640, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (EC*)ecs.data_ptr(),
                                                             ptok.data_ptr<int>(), (char*)img.data_ptr(), counts.data_ptr<int>());
}
template <int MODE>
void gemmx(torch::Tensor A, torch::Tensor Sc, torch::Tensor ecs, torch::Tensor tiles, torch::Tensor counts,
           torch::Tensor bimg, torch::Tensor himg, torch::Tensor inv, torch::Tensor yout, int64_t grid) {
  auto stream = at::cuda::getCurrentCUDAStream();
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, A.get_device());
  Args<MODE> a{(const unsigned*)A.data_ptr(), (const unsigned char*)Sc.data_ptr(), (const EC*)ecs.data_ptr(),
               (const Tile*)tiles.data_ptr(), 0, (const char*)bimg.data_ptr(),
               MODE == 1 ? (char*)himg.data_ptr() : nullptr, inv.data_ptr<float>(), nullptr,
               counts.data_ptr<int>() + MODE, MODE == 2 ? (__nv_bfloat16*)yout.data_ptr() : nullptr};
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(k_gemm<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, Lay<MODE>::SMEM);
    attr = true;
  }
  k_gemm<MODE><<<grid > 0 ? (int)grid : g_sms * MS3_CTAS, 32 + 32 * NCW, Lay<MODE>::SMEM, stream>>>(a);
}
void combine(torch::Tensor Y, torch::Tensor slotmap, torch::Tensor sh, torch::Tensor out) {
  auto stream = at::cuda::getCurrentCUDAStream();
  k_combine<<<(unsigned)out.size(0), D / 8, 0, stream>>>((const __nv_bfloat16*)Y.data_ptr(), slotmap.data_ptr<int>(),
                                                          (const __nv_bfloat16*)sh.data_ptr(), (__nv_bfloat16*)out.data_ptr());
}
int64_t route_max_ec() { return RT_MAXEC; }
int64_t gemm_ctas() { return MS3_CTAS; }

int64_t smem_bytes(int64_t mode) { return mode == 1 ? Lay<1>::SMEM : Lay<2>::SMEM; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prep", &prep);
  m.def("gemm1", &gemm<1>);
  m.def("gemm2", &gemm<2>);
  m.def("smem_bytes", &smem_bytes);
  m.def("route", &route);
  m.def("prepx", &prepx);
  m.def("gemm1x", &gemmx<1>);
  m.def("gemm2x", &gemmx<2>);
  m.def("combine", &combine);
  m.def("route_max_ec", &route_max_ec);
  m.def("gemm_ctas", &gemm_ctas);
  m.def("route_scratch_ints", &route_scratch_ints);
  m.def("swiglu", &swiglu);
  m.def("router_top", &router_top);
  m.def("prescale16", &prescale16);
  m.def("gemm1g", &gemm1g);
}
