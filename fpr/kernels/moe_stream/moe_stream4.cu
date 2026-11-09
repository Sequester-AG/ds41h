// moe_stream4: v4 of the routed-MoE decode layer (v3 in moe_stream3.cu, v1/v2 moe_stream.cu).
//
// Against v3 (measured on the GPU):
//  1. GEMM1 tiles are 384 rows (3 consecutive 128-row q-blocks, MB = 3): each warp keeps 3 m-blocks of 16 rows, so the
//     token fragments (ldmatrix / wgmma B) are loaded once per k64 step for 3 m-blocks and the expert's token image is
//     streamed 3 times instead of 9 (v3's in-kernel stream fell from 3.6 to 3.0 TB/s as the images outgrew L2).
//     Stage = 4 k64 steps x 3 m-blocks = 48 KB of weights (2 steps for chunks above 32 tokens). GEMM2 stays as v3.
//  2. MS4_ACC8 (default 1): 8-token chunks scale on the accumulator: per k32 block two mma.sync into a zero-started
//     temporary, then acc += 2^(s-121) * tmp (exact power of two). No HMUL2 / PRMT per weight.
// Same repacked weights, the same pre-biased scales (bench.py make_p) and EC / Tile structs as v3; Tile.q is the 384-row
// tile index (0..2) for GEMM1. Numerics: products exact, fp32 accumulation (summation order only).
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

#include "wgmma_rs_gen.cuh"

#ifndef MS4_MMA_MAXN
#define MS4_MMA_MAXN 16  // n_pad <= this: mma.sync path; larger: RS-wgmma
#endif
#ifndef MS4_ABL
#define MS4_ABL 0  // 1: stream only (consumers wait and release; results invalid)
#endif
#ifndef MS4_ACC8
#define MS4_ACC8 1  // 1: accumulator-side scales for 8-token chunks
#endif

namespace ms4 {

constexpr int D = 5120, IR = 576, NMAX = 64, NCW = 8, HDR = 128;

struct EC {  // one expert chunk (<= 64 tokens), same as moe_stream.cu
  int e, n_real, n_pad, pad;
  long long img_off, h_off;
  int tok[NMAX];
  float wt[NMAX];
};
static_assert(sizeof(EC) == 32 + 2 * 4 * NMAX, "EC layout must match bench.py");
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
struct Cfg<1> {  // GEMM1: K = 5120 = 80 k64 steps; tile = 3 q-blocks (384 rows); 4 steps per stage (n_pad <= 32) or 2
  static constexpr int KB = 80, QPE = 9, MB = 3, NSMALL = 32, CHS = 4, CHB = 2, S = 3;
};
template <>
struct Cfg<2> {  // GEMM2: K = 576 = 9 k64 steps; tile = 1 q-block (128 rows); whole K per stage (n_pad <= 24) or 3
  static constexpr int KB = 9, QPE = 40, MB = 1, NSMALL = 24, CHS = 9, CHB = 3, S = 3;
};
template <int MODE>
struct Lay {
  using C = Cfg<MODE>;
  static constexpr int B_MAX = (C::CHS * C::NSMALL > C::CHB * NMAX ? C::CHS * C::NSMALL : C::CHB * NMAX) * 128;
  static constexpr int A_MAX = C::MB * C::CHS * 4096, S_MAX = C::MB * C::CHS * 256;
  static constexpr int OFF_A = B_MAX, OFF_S = B_MAX + A_MAX, OFF_H = OFF_S + S_MAX;
  static constexpr int STAGE = ((OFF_H + HDR) + 1023) / 1024 * 1024;
  static constexpr int SMEM = 1024 + C::S * STAGE + 1024;
};
static_assert(Lay<1>::SMEM <= 227 * 1024 && Lay<2>::SMEM <= 227 * 1024, "ring must fit in SMEM");

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
__device__ __forceinline__ void mma16816_z(float* d, const unsigned (&a)[4], unsigned b0, unsigned b1) {  // d = a b
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
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
// one word (8 FP4 codes of rows g / g+8, k 2t..2t+1, 2t+8..2t+9) -> the m16n8k16 / RS-wgmma A fragment
__device__ __forceinline__ void dequant_word(unsigned (&a)[4], unsigned w, unsigned mg, unsigned mg8) {
  const unsigned lo = fp4_lo_e4m3(w), hi = fp4_hi_e4m3(w);
  a[0] = hmul2(cvt_e4m3x2(lo & 0xffff), mg);   // row g,   k 2t..2t+1
  a[2] = hmul2(cvt_e4m3x2(lo >> 16), mg);      // row g,   k 2t+8..
  a[1] = hmul2(cvt_e4m3x2(hi & 0xffff), mg8);  // row g+8, k 2t..
  a[3] = hmul2(cvt_e4m3x2(hi >> 16), mg8);     // row g+8, k 2t+8..
}

// one word without the scale (values e2m1 x 2^-6, exact f16)
__device__ __forceinline__ void dequant_raw(unsigned (&a)[4], unsigned w) {
  const unsigned lo = fp4_lo_e4m3(w), hi = fp4_hi_e4m3(w);
  a[0] = cvt_e4m3x2(lo & 0xffff);
  a[2] = cvt_e4m3x2(lo >> 16);
  a[1] = cvt_e4m3x2(hi & 0xffff);
  a[3] = cvt_e4m3x2(hi >> 16);
}
// pre-biased scale byte e = (s - 106) << 2 -> fp32 2^(s-121): exponent field s + 6 = (e >> 2) + 112
__device__ __forceinline__ float scale_f32(unsigned e) { return __uint_as_float((e << 21) + 0x38000000u); }

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
__global__ void k_gather(const __nv_bfloat16* __restrict__ x, const EC* __restrict__ ecs, const int* __restrict__ ptok,
                         char* __restrict__ img) {
  const EC& ec = ecs[blockIdx.x];
  int n = blockIdx.y;
  if (n >= ec.n_pad) return;
  int tok = ec.tok[n], slab = threadIdx.x >> 3, cc = threadIdx.x & 7;
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

// ------------------------------------------------------------------------------------------------ GEMM kernels
template <int MODE>
struct Args {
  const unsigned* A;        // repacked words (moe_stream layout, 128-row q-blocks)
  const unsigned char* Sc;  // pre-biased scales [e][q][kb][64 slots][4 B] (bench.py make_p)
  const EC* ecs;
  const Tile* tiles;  // GEMM1: q = 384-row tile index 0..2; GEMM2: q = 128-row tile 0..39
  int ntiles;
  const char* bimg;  // GEMM1: x image; GEMM2: H image
  char* himg;        // GEMM1 output
  const float* inv;
  float* out;
};

template <int MODE, int NT>
__device__ __forceinline__ void run_tile(const Args<MODE>& a, char* ring, uint64_t* full, uint64_t* empty, long& it,
                                         int tile, int c, int t) {
  using C = Cfg<MODE>;
  using Ly = Lay<MODE>;
  constexpr int N = NT * 8, MB = C::MB;
  constexpr int CH = N <= C::NSMALL ? C::CHS : C::CHB;
  constexpr int NCH = C::KB / CH;
  constexpr bool MMA = N <= MS4_MMA_MAXN;
  constexpr bool ACC = MMA && MS4_ACC8 && N == 8;
  static_assert(C::KB % CH == 0, "stages must tile K");
  const int w = t / 32, lane = t % 32, g = lane / 4, tq = lane % 4, R = 64 * c + 16 * w + g;
  const int slot = 32 * c + 8 * w + g;
  float acc[MB][2][N / 2];
#pragma unroll
  for (int m = 0; m < MB; ++m)
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
      for (int i = 0; i < N / 2; ++i) acc[m][q][i] = 0.f;
  for (int ch = 0; ch < NCH; ++ch) {
    const int s = it % C::S;
    if (ch > 0) mbar_wait(smem_u32(full + s), (it / C::S) & 1);
    char* st = ring + (size_t)s * Ly::STAGE;
    const uint32_t aB = smem_u32(st);
    if (MS4_ABL != 1) {
      if constexpr (MMA) {
#pragma unroll
        for (int kb = 0; kb < CH; ++kb) {
          unsigned bf[NT][2][4];  // token fragments of this k64 step, shared by the MB m-blocks
          const uint32_t slab = aB + kb * N * 128;
#pragma unroll
          for (int nb = 0; nb < NT; ++nb)
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
              ldsm_x4(bf[nb][kk], slab + swz(nb * 8 + (lane & 7), (kk * 4 + (lane >> 3)) * 16));
#pragma unroll
          for (int m = 0; m < MB; ++m) {
            const unsigned sc = *(const unsigned*)(st + Ly::OFF_S + (m * CH + kb) * 256 + slot * 4);
            const uint4 wv = *(const uint4*)(st + Ly::OFF_A + (m * CH + kb) * 4096 + (c * 128 + t) * 16);
            const unsigned ws[4] = {wv.x, wv.y, wv.z, wv.w};
            if constexpr (ACC) {
#pragma unroll
              for (int b = 0; b < 2; ++b) {  // k32 scale block b = words 2b, 2b+1
                unsigned f0[4], f1[4];
                dequant_raw(f0, ws[2 * b]);
                dequant_raw(f1, ws[2 * b + 1]);
                float tmp[4];
                mma16816_z(tmp, f0, bf[0][b][0], bf[0][b][1]);
                mma16816(tmp, f1, bf[0][b][2], bf[0][b][3]);
                const float sg = scale_f32((sc >> (16 * b)) & 0xffu), sg8 = scale_f32((sc >> (16 * b + 8)) & 0xffu);
                acc[m][b][0] = fmaf(sg, tmp[0], acc[m][b][0]);
                acc[m][b][1] = fmaf(sg, tmp[1], acc[m][b][1]);
                acc[m][b][2] = fmaf(sg8, tmp[2], acc[m][b][2]);
                acc[m][b][3] = fmaf(sg8, tmp[3], acc[m][b][3]);
              }
            } else {
              const unsigned m0 = __byte_perm(sc, 0, 0x0404), m08 = __byte_perm(sc, 0, 0x1414);
              const unsigned m1 = __byte_perm(sc, 0, 0x2424), m18 = __byte_perm(sc, 0, 0x3434);
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                unsigned fr[4];
                dequant_word(fr, ws[j], j < 2 ? m0 : m1, j < 2 ? m08 : m18);
#pragma unroll
                for (int nb = 0; nb < NT; ++nb)
                  mma16816(&acc[m][j & 1][nb * 4], fr, bf[nb][j / 2][(j & 1) * 2], bf[nb][j / 2][(j & 1) * 2 + 1]);
              }
            }
          }
        }
      } else {
        unsigned A[2][4][4];
#pragma unroll
        for (int kb = 0; kb < CH; ++kb) {
          const uint64_t db = desc_sw128(aB + kb * N * 128);
#pragma unroll
          for (int m = 0; m < MB; ++m) {
            const int gi = kb * MB + m, set = gi & 1;  // one wgmma group per (k64 step, m-block)
            if (gi >= 2) wg_wait<1>();               // the group that read A[set] (gi - 2) is done
            const unsigned sc = *(const unsigned*)(st + Ly::OFF_S + (m * CH + kb) * 256 + slot * 4);
            const unsigned m0 = __byte_perm(sc, 0, 0x0404), m08 = __byte_perm(sc, 0, 0x1414);
            const unsigned m1 = __byte_perm(sc, 0, 0x2424), m18 = __byte_perm(sc, 0, 0x3434);
            const uint4 wv = *(const uint4*)(st + Ly::OFF_A + (m * CH + kb) * 4096 + (c * 128 + t) * 16);
            const unsigned ws[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) dequant_word(A[set][j], ws[j], j < 2 ? m0 : m1, j < 2 ? m08 : m18);
            wg_fence();
#pragma unroll
            for (int j = 0; j < 4; ++j) WgRS<N>::mma(acc[m][0], A[set][j], db + 2 * j, 1);
            wg_commit();
          }
        }
        wg_wait<0>();
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_u32(empty + s));
    ++it;
  }
  // ---- epilogue per m-block. acc[m][*][4 nb + {0,1,2,3}] = (row g, col 8nb+2tq), (g, +1), (g+8, 2tq), (g+8, +1)
  const Tile tl = a.tiles[tile];
  const EC& ec = a.ecs[tl.ec];
#pragma unroll
  for (int m = 0; m < MB; ++m) {
    const int q = tl.q * MB + m;  // 128-row q-block within the expert
    float v[N / 2];
#pragma unroll
    for (int i = 0; i < N / 2; ++i) v[i] = acc[m][0][i] + acc[m][1][i];
    if (MODE == 1) {
      const int il = 32 * c + 8 * w + g;
      char* hs = a.himg + ec.h_off + (size_t)q * N * 128;
#pragma unroll
      for (int nb = 0; nb < NT; ++nb)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int n = 8 * nb + 2 * tq + hh;
          const int tok = ec.tok[n];
          float h = 0.f;
          if (tok >= 0) {
            const float iv = a.inv[tok];
            float gt = fminf(v[4 * nb + hh] * iv, 10.f);
            float up = fminf(fmaxf(v[4 * nb + 2 + hh] * iv, -10.f), 10.f);
            h = gt / (1.f + expf(-gt)) * up * ec.wt[n] * 256.f;
          }
          *(__half*)(hs + swz(n, il * 2)) = __float2half_rn(h);
        }
    } else {
      const int r0 = 128 * q + R;
#pragma unroll
      for (int nb = 0; nb < NT; ++nb)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int n = 8 * nb + 2 * tq + hh;
          const int tok = ec.tok[n];
          if (tok >= 0) {
            atomicAdd(a.out + (size_t)tok * D + r0, v[4 * nb + hh] * (1.f / 256.f));
            atomicAdd(a.out + (size_t)tok * D + r0 + 8, v[4 * nb + 2 + hh] * (1.f / 256.f));
          }
        }
    }
  }
}

template <int MODE>
__global__ void __launch_bounds__(32 + 32 * NCW, 1) k_gemm(Args<MODE> a) {
  using C = Cfg<MODE>;
  using Ly = Lay<MODE>;
  extern __shared__ __align__(1024) char smem_raw[];
  char* ring = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);  // keeps the shared address space
  uint64_t* full = (uint64_t*)(ring + C::S * Ly::STAGE);
  uint64_t* empty = full + C::S;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int s = 0; s < C::S; ++s) mbar_init(smem_u32(full + s), 1), mbar_init(smem_u32(empty + s), NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == NCW) {  // ---------------------------------------------------------------- producer (last warp)
    if (lane != 0) return;
    const uint64_t pw = pol_first(), pb = pol_last();
    long it = 0;
    int tile = blockIdx.x;
    Tile tl = tile < a.ntiles ? a.tiles[tile] : Tile{0, 0, 0, 0, 0};
    for (;;) {
      const bool end = tile >= a.ntiles;
      const int ntile = tile + gridDim.x;
      const Tile nx = ntile < a.ntiles ? a.tiles[ntile] : Tile{0, 0, 0, 0, 0};
      const int npad = end ? 0 : tl.npad;
      if (!end && (npad < 8 || npad > NMAX || (npad & 7))) __trap();
      const int CH = npad <= C::NSMALL ? C::CHS : C::CHB;
      const int nch = end ? 1 : C::KB / CH;
      for (int ch = 0; ch < nch; ++ch, ++it) {
        const int s = it % C::S;
        if (it >= C::S) mbar_wait(smem_u32(empty + s), ((it / C::S) + 1) & 1);
        char* st = ring + (size_t)s * Ly::STAGE;
        Hdr* h = (Hdr*)(st + Ly::OFF_H);
        h->tile = end ? -1 : tile;
        h->ch = ch;
        h->npad = npad;
        h->nch = nch;
        if (end) {
          mbar_arrive(smem_u32(full + s));
          break;
        }
        const unsigned bA = CH * 4096, bS = CH * 256, bB = CH * npad * 128;
        const uint32_t fb = smem_u32(full + s), dst = smem_u32(st);
        mbar_expect_tx(fb, C::MB * (bA + bS) + bB);
#pragma unroll
        for (int m = 0; m < C::MB; ++m) {
          const size_t blk = (size_t)tl.e * C::QPE + (size_t)tl.q * C::MB + m, k0 = blk * C::KB + (size_t)ch * CH;
          bulk(dst + Ly::OFF_A + m * bA, (const char*)a.A + k0 * 4096, bA, fb, pw);
          bulk(dst + Ly::OFF_S + m * bS, a.Sc + k0 * 256, bS, fb, pw);
        }
        bulk(dst, a.bimg + tl.boff + (size_t)ch * CH * npad * 128, bB, fb, pb);
      }
      if (end) break;
      tile = ntile;
      tl = nx;
    }
    return;
  }
  // ------------------------------------------------------------------ consumers (warps 0..7 = warpgroups 0, 1)
  const int ct = threadIdx.x, c = ct / 128, t = ct % 128;
  long it = 0;
  for (;;) {
    const int s = it % C::S;
    mbar_wait(smem_u32(full + s), (it / C::S) & 1);
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

}  // namespace ms4

// ------------------------------------------------------------------------------------------------ bindings
using namespace ms4;
static int g_sms = 0;

void prep(torch::Tensor x, torch::Tensor ecs, int64_t n_ec, torch::Tensor img, torch::Tensor ptok, torch::Tensor inv) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int T = x.size(0);
  k_prescale<<<T, 256, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), ptok.data_ptr<int>(), inv.data_ptr<float>());
  k_gather<<<dim3(n_ec, NMAX), 640, 0, stream>>>((const __nv_bfloat16*)x.data_ptr(), (const EC*)ecs.data_ptr(),
                                                ptok.data_ptr<int>(), (char*)img.data_ptr());
}

template <int MODE>
void gemm(torch::Tensor A, torch::Tensor Sc, torch::Tensor ecs, torch::Tensor tiles, torch::Tensor counter,
          torch::Tensor bimg, torch::Tensor himg, torch::Tensor inv, torch::Tensor out, int64_t grid) {
  (void)counter;
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
  k_gemm<MODE><<<grid > 0 ? (int)grid : g_sms, 32 + 32 * NCW, Lay<MODE>::SMEM, stream>>>(a);
}

int64_t smem_bytes(int64_t mode) { return mode == 1 ? Lay<1>::SMEM : Lay<2>::SMEM; }
int64_t tiles_per_expert(int64_t mode) { return mode == 1 ? Cfg<1>::QPE / Cfg<1>::MB : Cfg<2>::QPE / Cfg<2>::MB; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prep", &prep);
  m.def("gemm1", &gemm<1>);
  m.def("gemm2", &gemm<2>);
  m.def("smem_bytes", &smem_bytes);
  m.def("tiles_per_expert", &tiles_per_expert);
}
