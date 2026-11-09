// moe_stream: prototype of the routed-MoE decode layer at the HBM floor.
//
// One TP rank, one layer. Weights are repacked offline (repack.py) so that the bytes a TMA bulk copy lands in SMEM
// are exactly each consumer thread's wgmma A fragment (64 weight rows x k16 per warpgroup), dequantized once per
// weight per step. Tokens routed to an expert are the N side (8..64, padded to 8) and come from a per-expert,
// pre-swizzled f16 image (k_gather) so each stage is one contiguous bulk copy.
//
//   k_prescale  per-token power-of-two scale p so that max|x| * 2^p <= 2^14 (bf16 -> f16 is then exact for every
//               element >= 2^-28 max; smaller elements round, contributing < 2^-28 relative)
//   k_gather    x [T, 5120] bf16 -> per-expert-chunk B image [80 slabs][n_pad][128 B, 128B-swizzled] f16
//   k_gemm<1>   tiles (expert chunk, q): 64 intermediates = 64 gate + 64 up rows interleaved, K = 5120;
//               epilogue h = silu(min(g, 10)) * clamp(u, -10, 10) * w_route * 2^8 -> f16 H image [9 slabs][n_pad][128]
//   k_gemm<2>   tiles (expert chunk, rt): 128 output rows, K = 576; epilogue out[token][row] += acc * 2^-8 (fp32 atomics)
//
// CTA: warps 0..7 = two consumer warpgroups (rows 0..63 / 64..127 of the tile; wgmma needs warpgroups aligned to
// warp 4k); warp 8 lane 0 = producer (atomic tile queue, TMA bulk copies into an S-stage ring with full/empty
// mbarriers, L2 evict-first for weights).
// Weight value = e2m1 * 2^(s-127): FP4 -> e4m3 bit trick (x 2^-6) -> cvt e4m3x2 -> f16x2, HMUL2 by 2^(s-121).
// Exact for scale bytes 107..136 (measured range 118..124 on layers 3, 21, 38); bench.py checks the range at load.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

#include "wgmma_rs_gen.cuh"

namespace ms {

#ifndef MS_LAZY
#define MS_LAZY 0  // 1: release ring slots after the next stage's kb = 1 wait (v2: GEMM1 -16 %, off)
#endif
#ifndef MS_ABL
#define MS_ABL 0  // timing ablations (results invalid): 1 stream only, 2 no wgmma, 3 no dequant
#endif
constexpr int D = 5120, IR = 576, NMAX = 64, S = 4, NCW = 8;
constexpr int HDR = 128;  // per-stage header bytes

struct EC {  // one expert chunk (<= 64 tokens)
  int e, n_real, n_pad, pad;
  long long img_off, h_off;
  int tok[NMAX];  // -1 = padding
  float wt[NMAX];
};
static_assert(sizeof(EC) == 32 + 2 * 4 * NMAX, "EC layout must match bench.py");
struct Tile {  // everything the producer needs, so it never chases EC pointers (built by bench.py build_tiles)
  int ec, q, e, npad;
  long long boff;  // GEMM1: EC img_off, GEMM2: EC h_off
};
static_assert(sizeof(Tile) == 24, "Tile layout must match bench.py");
struct Hdr {
  int tile, ch, npad, pad;
};

template <int MODE>
struct Cfg;
template <>
struct Cfg<1> {  // GEMM1: K = 5120 = 80 k64 blocks, 4 per stage
  static constexpr int KB = 80, CH = 4, NCH = 20, QPE = 9;
};
template <>
struct Cfg<2> {  // GEMM2: K = 576 = 9 k64 blocks, 3 per stage
  static constexpr int KB = 9, CH = 3, NCH = 3, QPE = 40;
};
template <int MODE>
struct Lay {
  static constexpr int B_MAX = Cfg<MODE>::CH * NMAX * 128;
  static constexpr int A_B = Cfg<MODE>::CH * 4096;
  static constexpr int S_B = Cfg<MODE>::CH * 2 * 128;
  static constexpr int OFF_A = B_MAX, OFF_S = B_MAX + A_B, OFF_H = B_MAX + A_B + S_B;
  static constexpr int STAGE = ((OFF_H + HDR) + 1023) / 1024 * 1024;
  static constexpr int SMEM = 1024 + S * STAGE + 1024;  // align slack + ring + barriers
};

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
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
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
// FP4 -> e4m3 (x 2^-6, exact) and e4m3x2 -> f16x2 (exact); same as ds41h sm90.cuh
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
// f16x2 (2^(s-121), 2^(s-121)): f16 exponent field s - 121 + 15
__device__ __forceinline__ unsigned scale_mul(unsigned s) {
  unsigned f = (s - 106u) << 10;
  return f | (f << 16);
}
// byte offset of (row n, byte b within the 128-byte K row) in a 128B-swizzled K-major slab
__device__ __forceinline__ unsigned swz(int n, int b) {
  return (n >> 3) * 1024 + (n & 7) * 128 + ((((b >> 4) ^ (n & 7)) & 7) << 4) + (b & 15);
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
      frexpf(m, &e);  // m = f * 2^e, f in [0.5, 1)  ->  m * 2^(14 - e) < 2^14
      p = 14 - e;
    }
    ptok[blockIdx.x] = p;
    inv[blockIdx.x] = exp2f((float)-p);
  }
}

// grid (n_ec, NMAX), block 640 = 80 slabs x 8 16-byte chunks
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
  const unsigned* A;        // repacked words
  const unsigned char* Sc;  // repacked scales
  const EC* ecs;
  const Tile* tiles;
  int ntiles;
  int* counter;
  const char* bimg;  // GEMM1: x image; GEMM2: H image
  char* himg;        // GEMM1 output
  const float* inv;  // per-token 2^-p
  float* out;        // GEMM2 output [T][D] fp32
};

template <int MODE, int NT>
__device__ __forceinline__ void run_tile(const Args<MODE>& a, char* ring, uint64_t* full, uint64_t* empty, long& it,
                                         int tile, int c, int t) {
  using C = Cfg<MODE>;
  using Ly = Lay<MODE>;
  constexpr int N = NT * 8;
  // release a stage's slot lazily (after the next stage's kb = 1 wait) when A's register double buffer lines up
  // across stages (CH even) and the ablation keeps the wgmma waits
  constexpr bool LAZY = MS_LAZY && (C::CH % 2 == 0) && (MS_ABL == 0 || MS_ABL == 3);
  const int w = t / 32, lane = t % 32, g = lane / 4, tq = lane % 4, R = 64 * c + 16 * w + g;
  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  unsigned A[2][4][4];
  for (int ch = 0; ch < C::NCH; ++ch) {
    const int s = it % S;
    if (ch > 0) mbar_wait(smem_u32(full + s), (it / S) & 1);
    char* st = ring + (size_t)s * Ly::STAGE;
    const uint32_t aB = smem_u32(st), aA = aB + Ly::OFF_A;
    const unsigned char* sc = (const unsigned char*)st + Ly::OFF_S;
#pragma unroll
    for (int kb = 0; kb < C::CH; ++kb) {
      const int set = kb & 1;
      if (MS_ABL == 1) continue;  // ablation: stream only
      // the group that read A[set] (kb - 2, possibly in the previous stage) is done
      if (kb >= 2 || (LAZY && ch > 0)) wg_wait<1>();
      if (LAZY && ch > 0 && kb == 1) {  // every group of the previous stage is done: release its slot
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_u32(empty + (it - 1) % S));
      }
      uint4 wv = lds128(aA + (kb * 2 + c) * 2048 + t * 16);
      unsigned ws[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
      for (int j = 0; j < 4 && MS_ABL == 3; ++j)  // ablation: no dequant, raw words as A
        A[set][j][0] = A[set][j][1] = A[set][j][2] = A[set][j][3] = ws[j];
#pragma unroll
      for (int j = 0; j < 4 && MS_ABL != 3; ++j) {
        const int b = 2 * kb + j / 2;
        const unsigned sg = sc[b * 128 + R], sg8 = sc[b * 128 + R + 8];
        const unsigned mg = scale_mul(sg), mg8 = scale_mul(sg8);
        const unsigned lo = fp4_lo_e4m3(ws[j]), hi = fp4_hi_e4m3(ws[j]);
        A[set][j][0] = hmul2(cvt_e4m3x2(lo & 0xffff), mg);   // (n0, n2) = row g,   k 2t..2t+1
        A[set][j][2] = hmul2(cvt_e4m3x2(lo >> 16), mg);      // (n4, n6) = row g,   k 2t+8..
        A[set][j][1] = hmul2(cvt_e4m3x2(hi & 0xffff), mg8);  // (n1, n3) = row g+8, k 2t..
        A[set][j][3] = hmul2(cvt_e4m3x2(hi >> 16), mg8);     // (n5, n7) = row g+8, k 2t+8..
      }
      if (MS_ABL == 2) {  // ablation: dequant only, keep the values alive without wgmma
#pragma unroll
        for (int j = 0; j < 4; ++j)
          asm volatile("" ::"r"(A[set][j][0]), "r"(A[set][j][1]), "r"(A[set][j][2]), "r"(A[set][j][3]));
        continue;
      }
      wg_fence();
      const uint64_t db = desc_sw128(aB + kb * N * 128);
#pragma unroll
      for (int j = 0; j < 4; ++j) WgRS<N>::mma(acc, A[set][j], db + 2 * j, 1);
      wg_commit();
    }
    if (!LAZY || ch == C::NCH - 1) {  // drain only at the end of the tile (the epilogue needs acc anyway)
      wg_wait<0>();
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(empty + s));
    }
    ++it;
  }
  // ---- epilogue. acc[4 nb + {0,1,2,3}] = (row g, col 8nb+2tq), (g, +1), (g+8, 2tq), (g+8, +1)
  const Tile tl = a.tiles[tile];
  const EC& ec = a.ecs[tl.ec];
  if (MODE == 1) {
    const int il = 32 * c + 8 * w + g;  // intermediate within the tile = K index within H slab q
    char* hs = a.himg + ec.h_off + (size_t)tl.q * N * 128;
#pragma unroll
    for (int nb = 0; nb < NT; ++nb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nb + 2 * tq + hh;
        const int tok = ec.tok[n];
        float h = 0.f;
        if (tok >= 0) {
          const float iv = a.inv[tok];
          float gt = fminf(acc[4 * nb + hh] * iv, 10.f);
          float up = fminf(fmaxf(acc[4 * nb + 2 + hh] * iv, -10.f), 10.f);
          h = gt / (1.f + expf(-gt)) * up * ec.wt[n] * 256.f;
        }
        *(__half*)(hs + swz(n, il * 2)) = __float2half_rn(h);
      }
  } else {
    const int r0 = 128 * tl.q + R;
#pragma unroll
    for (int nb = 0; nb < NT; ++nb)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nb + 2 * tq + hh;
        const int tok = ec.tok[n];
        if (tok >= 0) {
          atomicAdd(a.out + (size_t)tok * D + r0, acc[4 * nb + hh] * (1.f / 256.f));
          atomicAdd(a.out + (size_t)tok * D + r0 + 8, acc[4 * nb + 2 + hh] * (1.f / 256.f));
        }
      }
  }
}

template <int MODE>
__global__ void __launch_bounds__(32 + 32 * NCW, 1) k_gemm(Args<MODE> a) {
  using C = Cfg<MODE>;
  using Ly = Lay<MODE>;
  extern __shared__ __align__(1024) char smem_raw[];
  char* ring = (char*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  uint64_t* full = (uint64_t*)(ring + S * Ly::STAGE);
  uint64_t* empty = full + S;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int s = 0; s < S; ++s) mbar_init(smem_u32(full + s), 1), mbar_init(smem_u32(empty + s), NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == NCW) {  // ---------------------------------------------- producer (last warp: warpgroups 0..1 stay aligned)
    if (lane != 0) return;
    const uint64_t pw = pol_first(), pb = pol_last();
    long it = 0;
    // static round-robin over tiles (all tiles of one GEMM stream the same weight bytes), descriptor of the next tile
    // loaded one tile ahead so no dependent global load sits between two tiles' copies
    int tile = blockIdx.x;
    Tile tl = tile < a.ntiles ? a.tiles[tile] : Tile{0, 0, 0, 0, 0};
    for (;;) {
      const bool end = tile >= a.ntiles;
      const int ntile = tile + gridDim.x;
      const Tile nx = ntile < a.ntiles ? a.tiles[ntile] : Tile{0, 0, 0, 0, 0};
      const int npad = end ? 0 : tl.npad;
      if (!end && (npad < 8 || npad > NMAX || (npad & 7))) __trap();
      const int nch = end ? 1 : C::NCH;
      for (int ch = 0; ch < nch; ++ch, ++it) {
        const int s = it % S;
        if (it >= S) mbar_wait(smem_u32(empty + s), ((it / S) + 1) & 1);
        char* st = ring + (size_t)s * Ly::STAGE;
        Hdr* h = (Hdr*)(st + Ly::OFF_H);
        h->tile = end ? -1 : tile;
        h->ch = ch;
        h->npad = npad;
        if (end) {
          mbar_arrive(smem_u32(full + s));
          break;
        }
        const size_t blk = (size_t)tl.e * C::QPE + tl.q;
        const char* srcA = (const char*)a.A + (blk * C::KB + (size_t)ch * C::CH) * 4096;
        const char* srcS = (const char*)a.Sc + (blk * (2 * C::KB) + (size_t)ch * 2 * C::CH) * 128;
        const char* srcB = a.bimg + tl.boff + (size_t)ch * C::CH * npad * 128;
        const unsigned bB = C::CH * npad * 128;
        mbar_expect_tx(smem_u32(full + s), Ly::A_B + Ly::S_B + bB);
        const uint32_t fb = smem_u32(full + s), dst = smem_u32(st);
        for (unsigned o = 0; o < (unsigned)Ly::A_B; o += 8192) bulk(dst + Ly::OFF_A + o, srcA + o, 8192 < Ly::A_B - o ? 8192 : Ly::A_B - o, fb, pw);
        bulk(dst + Ly::OFF_S, srcS, Ly::S_B, fb, pw);
        for (unsigned o = 0; o < bB; o += 8192) bulk(dst + o, srcB + o, 8192 < bB - o ? 8192 : bB - o, fb, pb);
      }
      if (end) break;
      tile = ntile;
      tl = nx;
    }
    return;
  }
  // ------------------------------------------------------------------ consumers
  const int ct = threadIdx.x, c = ct / 128, t = ct % 128;
  long it = 0;
  for (;;) {
    const int s = it % S;
    mbar_wait(smem_u32(full + s), (it / S) & 1);
    const Hdr h = *(const Hdr*)(ring + (size_t)s * Ly::STAGE + Ly::OFF_H);
    if (h.tile < 0) break;
    const int nt = h.npad / 8;
    switch (nt) {
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

}  // namespace ms

// ------------------------------------------------------------------------------------------------ bindings
using namespace ms;
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
  auto stream = at::cuda::getCurrentCUDAStream();
  if (!g_sms) cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, A.get_device());
  Args<MODE> a{(const unsigned*)A.data_ptr(), (const unsigned char*)Sc.data_ptr(), (const EC*)ecs.data_ptr(),
               (const Tile*)tiles.data_ptr(), (int)tiles.size(0), counter.data_ptr<int>(), (const char*)bimg.data_ptr(),
               MODE == 1 ? (char*)himg.data_ptr() : nullptr, inv.data_ptr<float>(),
               MODE == 2 ? out.data_ptr<float>() : nullptr};
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(k_gemm<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, Lay<MODE>::SMEM);
    attr = true;
  }
  cudaMemsetAsync(counter.data_ptr<int>(), 0, sizeof(int), stream);
  k_gemm<MODE><<<grid > 0 ? (int)grid : g_sms, 32 + 32 * NCW, Lay<MODE>::SMEM, stream>>>(a);
}

int64_t smem_bytes(int64_t mode) { return mode == 1 ? Lay<1>::SMEM : Lay<2>::SMEM; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prep", &prep);
  m.def("gemm1", &gemm<1>);
  m.def("gemm2", &gemm<2>);
  m.def("smem_bytes", &smem_bytes);
}
