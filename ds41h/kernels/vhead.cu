// ds41h vhead: LOSSLESS-compressed bf16 verification lm_head GEMV (sm90a).
//
// The target's lm_head shard (vocab-parallel, [32320, 5120] bf16 per TP rank = 331 MB) is streamed once per decode step
// (~95 us in situ at 350 W, HBM bound). Its bf16 exponents are concentrated: E in [112, 127] covers 99.984 % of the
// weights (DeepSeek-V4.1-Flash head.weight, every TP shard). Each weight is stored as
//   * one byte s|m (sign + 7 mantissa bits, unchanged) and
//   * a 4-bit code c = E - 112 (E = 0x70 | c),
// i.e. 12 bits instead of 16 (248 MB per rank, -25 % HBM bytes). Weights with E outside [112, 127] ("escapes": zeros,
// subnormals, |w| < 2^-15, ~26K per rank) are stored as code 0 / byte 0 (decodes to +2^-15) plus an exact correction
// entry (k, row, true bf16) per tile; the kernel adds w_true * x - 2^-15 * x (exact fp32 products) into the fp32
// accumulator before the bf16 rounding. Decoding reproduces every bf16 weight bit for bit (test_vhead: unpack == W), the
// products are exact and the sums are fp32 (HMMA, like cuBLAS) -> only the fp32 summation order differs from stock.
//
// Fragment layout (as dvh::head_bf16_kernel in drafter.cu): warp = 16-row tile; per k32 block kb, lane (g = lane/4,
// t = lane%4) owns rows g / g+8 at k = 32 kb + 8 t + [0, 8); the same k permutation is applied to x (B operand, n = token),
// so mma.m16n8k16 sums the right products. Storage per (tile, kb, lane): 16 s|m bytes (S.x/S.y: row g, S.z/S.w: row g+8),
// and per (tile, kb pair, lane) 4 code words (x/y: rows g / g+8 of the even kb, z/w: of the odd kb). A code word holds the 8
// codes of one row's 8 weights so that pair p (weights 2p, 2p+1 = one bf16x2 register) decodes as
//   a = prmt(S, sign-replicating selector)            -> [rep(s1) | s1m1 | rep(s0) | s0m0]
//   u = (rotl(C, r_p) & 0x07800780) | 0x38003800       -> exponent fields (0x70 | c) << 7 of both halves
//   R = (a & 0x807F807F) | u                           -> two bf16
// with r_p = 0, 4, 28, 8 (the encoder stores rotr(c_lo << 7 | c_hi << 23, r_p)).
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "common.cuh"
#include "host.cuh"

namespace vhc {
constexpr int K = 5120, KB = K / 32, KB2 = KB / 2, MAXM = 8;
constexpr int E0 = 112;          // code 0 exponent
constexpr int ESC_CAP = 4096;    // escape entries per tile (pack scratch)

__device__ __forceinline__ unsigned prmt(unsigned a, unsigned sel) {
  unsigned d;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(d) : "r"(a), "r"(sel));
  return d;
}
__device__ __forceinline__ unsigned rotl(unsigned x, int r) { return __funnelshift_l(x, x, r); }
__device__ __forceinline__ unsigned dec(unsigned s, unsigned c, unsigned sel) {
  const unsigned a = prmt(s, sel);
  const unsigned u = (c & 0x07800780u) | 0x38003800u;
  return (a & 0x807F807Fu) | u;
}
constexpr unsigned SEL01 = 0x9180u, SEL23 = 0xB3A2u;

// 8 bf16x2 A-fragment registers of one kb from S (16 s|m bytes) and the two code words (row g, row g+8)
__device__ __forceinline__ void decode_kb(const uint4 s, unsigned c0, unsigned c1, uint4& va, uint4& vb) {
  va.x = dec(s.x, c0, SEL01);
  va.y = dec(s.x, rotl(c0, 4), SEL23);
  va.z = dec(s.y, rotl(c0, 28), SEL01);
  va.w = dec(s.y, rotl(c0, 8), SEL23);
  vb.x = dec(s.z, c1, SEL01);
  vb.y = dec(s.z, rotl(c1, 4), SEL23);
  vb.z = dec(s.w, rotl(c1, 28), SEL01);
  vb.w = dec(s.w, rotl(c1, 8), SEL23);
}

__device__ __forceinline__ void hmma_bf16_acc(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ------------------------------------------------------------------------------------------------ pack (one-time)
struct PackArgs {
  const unsigned short* w;  // bf16 bits [n, ldw]
  long ldw;
  uint4* sm;                // [tiles][KB][32]
  uint2* code;              // [tiles][KB2][32][2] (uint4 viewed as 2 x uint2: even kb, odd kb)
  int* esc_count;           // [tiles]
  uint2* esc_scratch;       // [tiles][ESC_CAP]
};

constexpr int PT = 256;  // pack threads: thread i handles items i, i + 256, ... of the tile's KB * 32 (kb, lane) items

__device__ __forceinline__ void pack_item(const PackArgs& a, int tile, int item, uint4& s, unsigned& c0, unsigned& c1, unsigned& escmask) {
  const int kb = item >> 5, lane = item & 31, g = lane >> 2, t = lane & 3;
  unsigned sb[16], cd[16];
  escmask = 0u;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const unsigned short* row = a.w + long(tile * 16 + g + 8 * h) * a.ldw + kb * 32 + 8 * t;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const unsigned v = row[j];
      const unsigned e = (v >> 7) & 0xFFu;
      const bool ok = e >= unsigned(E0) && e <= unsigned(E0 + 15);
      sb[8 * h + j] = ok ? (((v >> 8) & 0x80u) | (v & 0x7Fu)) : 0u;
      cd[8 * h + j] = ok ? (e - E0) : 0u;
      if (!ok) escmask |= 1u << (8 * h + j);
    }
  }
  s.x = sb[0] | sb[1] << 8 | sb[2] << 16 | sb[3] << 24;
  s.y = sb[4] | sb[5] << 8 | sb[6] << 16 | sb[7] << 24;
  s.z = sb[8] | sb[9] << 8 | sb[10] << 16 | sb[11] << 24;
  s.w = sb[12] | sb[13] << 8 | sb[14] << 16 | sb[15] << 24;
  unsigned c[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const unsigned* q = cd + 8 * h;
    const unsigned t0 = q[0] << 7 | q[1] << 23, t1 = q[2] << 7 | q[3] << 23, t2 = q[4] << 7 | q[5] << 23, t3 = q[6] << 7 | q[7] << 23;
    c[h] = t0 | __funnelshift_r(t1, t1, 4) | __funnelshift_r(t2, t2, 28) | __funnelshift_r(t3, t3, 8);
  }
  c0 = c[0], c1 = c[1];
}

__global__ void __launch_bounds__(PT) pack_kernel(const PackArgs a) {
  const int tile = blockIdx.x, tid = threadIdx.x;
  constexpr int ITEMS = KB * 32 / PT;  // 20
  __shared__ int warp_tot[PT / 32];
  __shared__ int base_sh[PT / 32];
  int mine = 0;
  for (int it = 0; it < ITEMS; ++it) {
    const int item = tid + it * PT;
    uint4 s;
    unsigned c0, c1, em;
    pack_item(a, tile, item, s, c0, c1, em);
    const int kb = item >> 5, lane = item & 31;
    a.sm[(long(tile) * KB + kb) * 32 + lane] = s;
    a.code[((long(tile) * KB2 + (kb >> 1)) * 32 + lane) * 2 + (kb & 1)] = make_uint2(c0, c1);
    mine += __popc(em);
  }
  // deterministic compaction of the escapes: thread order, then item order
  const int lane = tid & 31, warp = tid >> 5;
  int incl = mine;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) warp_tot[warp] = incl;
  __syncthreads();
  if (tid == 0) {
    int acc = 0;
    for (int w = 0; w < PT / 32; ++w) base_sh[w] = acc, acc += warp_tot[w];
    a.esc_count[tile] = acc;
  }
  __syncthreads();
  int pos = base_sh[warp] + incl - mine;
  for (int it = 0; it < ITEMS && mine > 0; ++it) {
    const int item = tid + it * PT;
    const int kb = item >> 5, ln = item & 31, g = ln >> 2, t = ln & 3;
#pragma unroll 1
    for (int h = 0; h < 2; ++h) {
      const int r = g + 8 * h;
      const unsigned short* row = a.w + long(tile * 16 + r) * a.ldw + kb * 32 + 8 * t;
      for (int j = 0; j < 8; ++j) {
        const unsigned v = row[j];
        const unsigned e = (v >> 7) & 0xFFu;
        if (!(e >= unsigned(E0) && e <= unsigned(E0 + 15))) {
          if (pos < ESC_CAP) a.esc_scratch[long(tile) * ESC_CAP + pos] = make_uint2(unsigned(kb * 32 + 8 * t + j) | unsigned(r) << 16, v);
          ++pos;
        }
      }
    }
  }
}

// test helper: decode the packed form back to bf16 bits [n, K] (escapes NOT patched: they read +2^-15)
__global__ void unpack_kernel(const uint4* sm, const uint4* code, unsigned* out, long ldo_words) {
  const int tile = blockIdx.x;
  for (int item = threadIdx.x; item < KB * 32; item += blockDim.x) {
    const int kb = item >> 5, lane = item & 31, g = lane >> 2, t = lane & 3;
    const uint4 s = sm[(long(tile) * KB + kb) * 32 + lane];
    const uint4 cw = code[(long(tile) * KB2 + (kb >> 1)) * 32 + lane];
    const unsigned c0 = (kb & 1) ? cw.z : cw.x, c1 = (kb & 1) ? cw.w : cw.y;
    uint4 va, vb;
    decode_kb(s, c0, c1, va, vb);
    unsigned* ra = out + long(tile * 16 + g) * ldo_words + (kb * 32 + 8 * t) / 2;
    unsigned* rb = out + long(tile * 16 + g + 8) * ldo_words + (kb * 32 + 8 * t) / 2;
    ra[0] = va.x, ra[1] = va.y, ra[2] = va.z, ra[3] = va.w;
    rb[0] = vb.x, rb[1] = vb.y, rb[2] = vb.z, rb[3] = vb.w;
  }
}

// ------------------------------------------------------------------------------------------------ GEMV
struct Args {
  const __nv_bfloat16* x;
  long ldx;
  const long long* idx;  // [m] row indices of x, or null
  const uint4* sm;
  const uint4* code;
  const int* esc_off;    // [tiles + 1]
  const uint2* esc;
  __nv_bfloat16* y;
  long ldy;
  int m, n;
};

template <int T, int CK>
__global__ void __launch_bounds__(T, 1) gemv_kernel(const Args a) {
  constexpr int W = T / 32;
  extern __shared__ __align__(16) unsigned char smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);  // [KB][32]: lane (gg, tt) = x[gg][32 kb + 8 tt .. +8] (bf16)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int tiles = a.n / 16;
  const int tile = blockIdx.x * W + warp;
  const bool live = tile < tiles;
  const int tl = live ? tile : 0;
  const uint4* sp = a.sm + long(tl) * KB * 32 + lane;
  const uint4* cp = a.code + long(tl) * KB2 * 32 + lane;
  static_assert(CK % 2 == 0 && KB % (2 * CK) == 0, "chunking");
  uint4 s0[CK], s1[CK], c0[CK / 2], c1[CK / 2];
#pragma unroll
  for (int q = 0; q < CK; ++q) s0[q] = ld_stream16(sp + q * 32);
#pragma unroll
  for (int q = 0; q < CK / 2; ++q) c0[q] = ld_stream16(cp + q * 32);
#pragma unroll
  for (int q = 0; q < CK; ++q) s1[q] = ld_stream16(sp + (CK + q) * 32);
#pragma unroll
  for (int q = 0; q < CK / 2; ++q) c1[q] = ld_stream16(cp + (CK / 2 + q) * 32);
  pdl_wait();
  for (int i = tid; i < KB * 32; i += T) {
    const int kb = i >> 5, l = i & 31, gg = l >> 2, tt = l & 3;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (gg < a.m) {
      const long row = a.idx != nullptr ? long(a.idx[gg]) : long(gg);
      v = __ldg(reinterpret_cast<const uint4*>(a.x + row * a.ldx + kb * 32 + 8 * tt));
    }
    xs[i] = v;
  }
  __syncthreads();
  if (live) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto step = [&](const uint4 s, unsigned cw0, unsigned cw1, int kb) {
      uint4 va, vb;
      decode_kb(s, cw0, cw1, va, vb);
      const uint4 xb = xs[kb * 32 + lane];
      hmma_bf16_acc(acc, va.x, vb.x, va.y, vb.y, xb.x, xb.y);
      hmma_bf16_acc(acc, va.z, vb.z, va.w, vb.w, xb.z, xb.w);
    };
    auto chunk = [&](const uint4 (&s)[CK], const uint4 (&c)[CK / 2], int kb) {
#pragma unroll
      for (int q = 0; q < CK / 2; ++q) {
        step(s[2 * q], c[q].x, c[q].y, kb + 2 * q);
        step(s[2 * q + 1], c[q].z, c[q].w, kb + 2 * q + 1);
      }
    };
#pragma unroll 1
    for (int kb = 0; kb < KB; kb += 2 * CK) {  // both chunks preloaded at entry
      chunk(s0, c0, kb);
      if (kb + 2 * CK < KB) {
#pragma unroll
        for (int q = 0; q < CK; ++q) s0[q] = ld_stream16(sp + (kb + 2 * CK + q) * 32);
#pragma unroll
        for (int q = 0; q < CK / 2; ++q) c0[q] = ld_stream16(cp + ((kb + 2 * CK) / 2 + q) * 32);
      }
      chunk(s1, c1, kb + CK);
      if (kb + 3 * CK < KB) {
#pragma unroll
        for (int q = 0; q < CK; ++q) s1[q] = ld_stream16(sp + (kb + 3 * CK + q) * 32);
#pragma unroll
        for (int q = 0; q < CK / 2; ++q) c1[q] = ld_stream16(cp + ((kb + 3 * CK) / 2 + q) * 32);
      }
    }
    // escapes of this tile: + w_true * x - 2^-15 * x (the decoded placeholder), exact fp32 products
    const int e0 = __ldg(a.esc_off + tile), e1 = __ldg(a.esc_off + tile + 1);
    const __nv_bfloat16* xsh = reinterpret_cast<const __nv_bfloat16*>(xs);
    for (int e = e0; e < e1; ++e) {
      const uint2 en = __ldg(a.esc + e);
      const int k = int(en.x & 0xFFFFu), r = int(en.x >> 16);
      if ((r & 7) != g) continue;
      const float wt = __uint_as_float(en.y << 16);
      const int kb = k >> 5, tt = (k >> 3) & 3, j = k & 7;
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int tok = 2 * t + c;
        const float xv = __bfloat162float(xsh[(kb * 32 + tok * 4 + tt) * 8 + j]);
        float& d = acc[(r >> 3) * 2 + c];
        d = __fmaf_rn(wt, xv, d);
        d = __fmaf_rn(-3.0517578125e-05f, xv, d);  // -2^-15
      }
    }
    const int r0 = tile * 16;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int tok = 2 * t + c;
      if (tok < a.m) {
        a.y[long(tok) * a.ldy + r0 + g] = __float2bfloat16_rn(acc[c]);
        a.y[long(tok) * a.ldy + r0 + g + 8] = __float2bfloat16_rn(acc[2 + c]);
      }
    }
  }
  pdl_launch_dependents();
}

}  // namespace vhc

// ------------------------------------------------------------------------------------------------ plain bf16 row-range GEMV
// DS41H_VERIFY_HEAD=3 (exact, stock precision): y[t, n] = bf16(sum_k W[n, k] x[t, k]) for t < m <= 8, W the bf16 lm_head shard
// [N, 5120] row-major (as loaded, no copy), x = hidden_states[idx] (gather fused). fp32 accumulation of the exact bf16
// products (HMMA m16n8k16, like cuBLAS), ONE bf16 rounding per logit; only the fp32 summation order differs from stock.
// Deterministic: the work split and the reduction order are static (bitwise repeatable run to run).
//
// Why a new kernel: the head is a pure HBM stream (331 MB per rank); cuBLAS nvjet and the fragment-layout GEMVs (dvh / vhc)
// read 64-128 B per row per request from 2000-32000 concurrent row streams, i.e. poor DRAM row-buffer locality (3.2-3.5 TB/s
// in situ vs 3.9-4.05 for a contiguous stream). Here each SM owns a contiguous row range [s N / G, (s+1) N / G) (244-245 rows
// = 2.5 MB, balanced to one row) and streams it as 16-row tiles x SK-wide k stages: warp w copies row w of the stage with
// cp.async.cg 16 B (512 B contiguous per warp instruction, SK * 2 contiguous bytes per row per stage) into an XOR-swizzled smem
// ring of D stages; then every warp takes its fixed k16 steps of the stage (ldmatrix.x4 + HMMA) with its B fragments (x) held
// in registers for the whole kernel. Per tile, the 16 warp partials are summed in warp order by warps 0-3 (double-buffered
// smem), rounded once and stored. Rows past the range are zero-filled (cp.async src-size 0) and never stored.
namespace vrh {
constexpr int K = 5120, MAXM = 8, T = 512, NW = T / 32;

struct Args {
  const __nv_bfloat16* x;
  long ldx;
  const long long* idx;  // [m] row indices of x, or null
  const __nv_bfloat16* w;
  long ldw;
  __nv_bfloat16* y;
  long ldy;
  int m, n;
};

__device__ __forceinline__ void cp16z(unsigned dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void ldsm_x4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

template <int SK, int D>
struct Cfg {
  static constexpr int SPT = K / SK;     // stages per 16-row tile
  static constexpr int KW = SK / 256;    // k16 steps per warp per stage (16 warps x KW x 16 = SK)
  static constexpr int ROWB = SK * 2;    // bytes of one row per stage
  static constexpr int STB = 16 * ROWB;  // bytes per stage
  static constexpr int CPL = ROWB / 512; // cp.async per lane per stage (warp w copies row w)
  static constexpr int RED = 2 * NW * 128 * 4;
  static constexpr int SMEM = D * STB + RED;
  static_assert(K % SK == 0 && SK % 256 == 0 && D >= 2, "vrh config");
};

template <int SK, int D>
__global__ void __launch_bounds__(T, 1) rows_kernel(const Args a) {
  using C = Cfg<SK, D>;
  extern __shared__ __align__(128) unsigned char smem[];
  float* red = reinterpret_cast<float*>(smem + D * C::STB);  // [2][NW][32 lanes][4]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, s = blockIdx.x;
  const int r0 = int((long(s) * a.n) / G), r1 = int((long(s + 1) * a.n) / G);
  const int tiles = (r1 - r0 + 15) >> 4;
  const int S = tiles * C::SPT;
  const unsigned sb = smem_u32(smem);
  const int wsw = warp & 7;
  // producer side: this thread's part of stage j = row `warp` of the stage, CPL x 16 B (512 B contiguous per warp instruction)
  auto issue = [&](int j) {
    if (j < S) {
      const int tile = j / C::SPT, ks = j - tile * C::SPT;
      const int row = r0 + tile * 16 + warp;
      const bool ok = row < r1;
      const __nv_bfloat16* src = a.w + long(ok ? row : r0) * a.ldw + ks * SK;
      const unsigned dst = sb + unsigned((j % D) * C::STB + warp * C::ROWB);
#pragma unroll
      for (int u = 0; u < C::CPL; ++u) {
        const int c = u * 32 + lane;  // logical 16-byte chunk of this row segment
        cp16z(dst + unsigned((c ^ wsw) << 4), src + c * 8, ok ? 16 : 0);
      }
    }
    cp_async_commit();  // possibly empty: keeps the group count aligned
  };
#pragma unroll
  for (int j = 0; j < D - 1; ++j) issue(j);  // weights do not depend on the previous kernel: in flight across the PDL wait
  pdl_wait();
  // B fragments of the warp's fixed k16 steps (token g = lane / 4, k pair 2t / 2t + 8), for the whole kernel
  const int g = lane >> 2, t = lane & 3;
  unsigned b0[C::SPT * C::KW], b1[C::SPT * C::KW];
  {
    const bool live = g < a.m;
    const long xr = live ? (a.idx != nullptr ? long(a.idx[g]) : long(g)) : 0;
    const unsigned* xp = reinterpret_cast<const unsigned*>(a.x + xr * a.ldx);
#pragma unroll
    for (int st = 0; st < C::SPT; ++st)
#pragma unroll
      for (int u = 0; u < C::KW; ++u) {
        const int k0 = st * SK + (warp * C::KW + u) * 16;
        b0[st * C::KW + u] = live ? __ldg(xp + (k0 >> 1) + t) : 0u;
        b1[st * C::KW + u] = live ? __ldg(xp + (k0 >> 1) + 4 + t) : 0u;
      }
  }
  // ldmatrix.x4 lane roles: matrix q = lane / 8 (0: rows 0-7 k 0-7, 1: rows 8-15 k 0-7, 2: rows 0-7 k 8-15, 3: rows 8-15 k 8-15)
  const int q = lane >> 3, arow = (lane & 7) + 8 * (q & 1), ahalf = q >> 1;
  const unsigned arow_off = unsigned(arow * C::ROWB);
  const int asw = arow & 7;
  // reducer (warps 0-3): thread rq sums element rq of the 16 warp partials (lane rq / 4, register rq % 4)
  auto reduce_store = [&](int tile) {
    const int rq = tid;  // < 128
    const float* rp = red + ((tile & 1) * NW) * 128 + rq;
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < NW; ++w) v += rp[w * 128];
    const int sl = rq >> 2, e = rq & 3, sg = sl >> 2, st = sl & 3;
    const int tok = 2 * st + (e & 1), row = r0 + tile * 16 + sg + 8 * (e >> 1);
    if (tok < a.m && row < r1) a.y[long(tok) * a.ldy + row] = __float2bfloat16_rn(v);
  };
  int j = 0;
  for (int tile = 0; tile < tiles; ++tile) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int ks = 0; ks < C::SPT; ++ks, ++j) {
      cp_async_wait<D - 2>();
      __syncthreads();  // stage j landed for every thread; every warp is done with stage j - 1 (its slot is refilled below)
      if (ks == 0 && tile > 0 && warp < 4) reduce_store(tile - 1);
      issue(j + D - 1);
      const unsigned base = sb + unsigned((j % D) * C::STB) + arow_off;
#pragma unroll
      for (int u = 0; u < C::KW; ++u) {
        const int c = 2 * (warp * C::KW + u) + ahalf;
        unsigned a0, a1, a2, a3;
        ldsm_x4(base + unsigned((c ^ asw) << 4), a0, a1, a2, a3);
        vhc::hmma_bf16_acc(acc, a0, a1, a2, a3, b0[ks * C::KW + u], b1[ks * C::KW + u]);
      }
    }
    *reinterpret_cast<float4*>(red + ((tile & 1) * NW + warp) * 128 + lane * 4) = make_float4(acc[0], acc[1], acc[2], acc[3]);
  }
  cp_async_wait<0>();
  __syncthreads();
  if (tiles > 0 && warp < 4) reduce_store(tiles - 1);
  pdl_launch_dependents();
}
}  // namespace vrh

// ------------------------------------------------------------------------------------------------ host
std::vector<torch::Tensor> pack(torch::Tensor w) {
  using namespace vhc;
  TORCH_CHECK(w.is_cuda() && w.dim() == 2 && w.scalar_type() == at::kBFloat16 && w.stride(1) == 1 && w.size(1) == K && w.size(0) % 16 == 0,
              "vhead pack: w [n % 16 == 0, 5120] bf16 (unit inner stride)");
  const int tiles = int(w.size(0) / 16);
  const c10::cuda::CUDAGuard guard(w.device());
  auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(w.device());
  auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(w.device());
  torch::Tensor sm = torch::empty({tiles, KB, 32, 16}, u8);
  torch::Tensor code = torch::empty({tiles, KB2, 32, 16}, u8);
  torch::Tensor cnt = torch::zeros({tiles}, i32);
  torch::Tensor scratch = torch::empty({long(tiles) * ESC_CAP, 2}, i32);
  PackArgs a{};
  a.w = reinterpret_cast<const unsigned short*>(w.data_ptr());
  a.ldw = w.stride(0);
  a.sm = reinterpret_cast<uint4*>(sm.data_ptr());
  a.code = reinterpret_cast<uint2*>(code.data_ptr());
  a.esc_count = cnt.data_ptr<int>();
  a.esc_scratch = reinterpret_cast<uint2*>(scratch.data_ptr());
  pack_kernel<<<tiles, PT, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int mx = cnt.max().item<int>();
  TORCH_CHECK(mx <= ESC_CAP, "vhead pack: ", mx, " escapes in one tile > capacity ", ESC_CAP);
  torch::Tensor off = torch::zeros({tiles + 1}, i32);
  off.slice(0, 1).copy_(torch::cumsum(cnt, 0, torch::kInt32));
  // compact: entries of tile i = scratch[i * CAP : i * CAP + cnt[i]]
  torch::Tensor ar = torch::arange(ESC_CAP, i32).unsqueeze(0).expand({tiles, ESC_CAP});
  torch::Tensor keep = ar < cnt.unsqueeze(1);
  torch::Tensor esc = scratch.view({tiles, ESC_CAP, 2}).index({keep}).contiguous();  // row-major order = tile, then pos
  return {sm, code, off, esc};
}

torch::Tensor unpack(torch::Tensor sm, torch::Tensor code) {
  using namespace vhc;
  const int tiles = int(sm.size(0));
  const c10::cuda::CUDAGuard guard(sm.device());
  torch::Tensor out = torch::empty({long(tiles) * 16, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(sm.device()));
  unpack_kernel<<<tiles, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint4*>(sm.data_ptr()),
                                                                     reinterpret_cast<const uint4*>(code.data_ptr()),
                                                                     reinterpret_cast<unsigned*>(out.data_ptr()), K / 2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <int T, int CK>
void launch_gemv(const vhc::Args& a, int tiles, bool pdl) {
  using namespace vhc;
  constexpr int smem = KB * 32 * 16;
  opt_in_smem<gemv_kernel<T, CK>>(smem);
  const int grid = (tiles + T / 32 - 1) / (T / 32);
  if (pdl) {
    launch_pdl("ds41h vhead gemv", gemv_kernel<T, CK>, dim3(grid), dim3(T), smem, a);
  } else {
    gemv_kernel<T, CK><<<grid, T, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void gemv(torch::Tensor x, std::optional<torch::Tensor> idx, torch::Tensor sm, torch::Tensor code, torch::Tensor esc_off,
          torch::Tensor esc, torch::Tensor y, bool pdl, int64_t variant) {
  using namespace vhc;
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.size(1) == K && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "vhead gemv: x [*, 5120] bf16, 16-byte aligned rows");
  const int tiles = int(sm.size(0));
  TORCH_CHECK(sm.is_contiguous() && code.is_contiguous() && sm.numel() == long(tiles) * KB * 32 * 16 && code.numel() == long(tiles) * KB2 * 32 * 16,
              "vhead gemv: packed tensors");
  TORCH_CHECK(esc_off.scalar_type() == at::kInt && esc_off.numel() == tiles + 1 && esc.scalar_type() == at::kInt && esc.is_contiguous(),
              "vhead gemv: escapes");
  const int m = int(y.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM && y.scalar_type() == at::kBFloat16 && y.size(1) == long(tiles) * 16 && y.stride(1) == 1, "vhead gemv: y [m <= 8, n]");
  Args a{};
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ldx = x.stride(0);
  if (idx.has_value()) {
    TORCH_CHECK(idx->scalar_type() == at::kLong && idx->is_contiguous() && idx->numel() >= m, "vhead gemv: idx int64 [m]");
    a.idx = reinterpret_cast<const long long*>(idx->data_ptr<int64_t>());
  } else {
    TORCH_CHECK(x.size(0) >= m, "vhead gemv: x rows");
  }
  a.sm = reinterpret_cast<const uint4*>(sm.data_ptr());
  a.code = reinterpret_cast<const uint4*>(code.data_ptr());
  a.esc_off = esc_off.data_ptr<int>();
  a.esc = reinterpret_cast<const uint2*>(esc.data_ptr());
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  a.ldy = y.stride(0);
  a.m = m;
  a.n = tiles * 16;
  const c10::cuda::CUDAGuard guard(x.device());
  if (variant == 1) launch_gemv<512, 2>(a, tiles, pdl);
  else if (variant == 2) launch_gemv<256, 8>(a, tiles, pdl);
  else if (variant == 3) launch_gemv<256, 4>(a, tiles, pdl);
  else launch_gemv<512, 4>(a, tiles, pdl);
}

template <int SK, int D>
void launch_rows(const vrh::Args& a, int ctas, bool pdl) {
  using C = vrh::Cfg<SK, D>;
  opt_in_smem<vrh::rows_kernel<SK, D>>(C::SMEM);
  if (pdl) {
    launch_pdl("ds41h vhead rows", vrh::rows_kernel<SK, D>, dim3(ctas), dim3(vrh::T), C::SMEM, a);
  } else {
    vrh::rows_kernel<SK, D><<<ctas, vrh::T, C::SMEM, c10::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

static int sm_count(int dev) {
  static int cached[64] = {0};
  if (dev >= 0 && dev < 64 && cached[dev] > 0) return cached[dev];
  int n = 0;
  TORCH_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && n > 0);
  if (dev >= 0 && dev < 64) cached[dev] = n;
  return n;
}

// y[m, n] (bf16) = x[idx] (or x[:m]) @ w^T, w bf16 [n, 5120] (row stride ldw), m <= 8. ctas = 0: one CTA per SM.
void gemv_rows(torch::Tensor x, std::optional<torch::Tensor> idx, torch::Tensor w, torch::Tensor y, bool pdl, int64_t variant,
               int64_t ctas) {
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.size(1) == vrh::K && x.stride(0) % 2 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 4 == 0,
              "vhead rows: x [*, 5120] bf16, even row stride");
  TORCH_CHECK(w.dim() == 2 && w.scalar_type() == at::kBFloat16 && w.stride(1) == 1 && w.size(1) == vrh::K && w.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "vhead rows: w [n, 5120] bf16, 16-byte aligned rows");
  const int m = int(y.size(0));
  TORCH_CHECK(m >= 1 && m <= vrh::MAXM && y.scalar_type() == at::kBFloat16 && y.size(1) == w.size(0) && y.stride(1) == 1,
              "vhead rows: y [m <= 8, n] bf16");
  TORCH_CHECK(x.device() == w.device() && y.device() == w.device(), "vhead rows: devices");
  vrh::Args a{};
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ldx = x.stride(0);
  if (idx.has_value()) {
    TORCH_CHECK(idx->scalar_type() == at::kLong && idx->is_contiguous() && idx->numel() >= m && idx->device() == x.device(),
                "vhead rows: idx int64 [m]");
    a.idx = reinterpret_cast<const long long*>(idx->data_ptr<int64_t>());
  } else {
    TORCH_CHECK(x.size(0) >= m, "vhead rows: x rows");
  }
  a.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  a.ldw = w.stride(0);
  a.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  a.ldy = y.stride(0);
  a.m = m;
  a.n = int(w.size(0));
  const c10::cuda::CUDAGuard guard(x.device());
  const int g = ctas > 0 ? int(ctas) : sm_count(x.device().index());
  switch (variant) {
    case 1: launch_rows<1024, 5>(a, g, pdl); break;
    case 2: launch_rows<512, 10>(a, g, pdl); break;
    case 3: launch_rows<1280, 5>(a, g, pdl); break;
    case 4: launch_rows<512, 12>(a, g, pdl); break;
    case 5: launch_rows<1280, 4>(a, g, pdl); break;
    default: launch_rows<1024, 6>(a, g, pdl); break;
  }
}

int64_t preload() {
  int n = 0;
  auto touch = [&](const void* f) {
    cudaFuncAttributes attr;
    if (cudaFuncGetAttributes(&attr, f) == cudaSuccess) ++n;
  };
  touch(reinterpret_cast<const void*>(vhc::gemv_kernel<512, 4>));
  touch(reinterpret_cast<const void*>(vhc::gemv_kernel<512, 2>));
  touch(reinterpret_cast<const void*>(vhc::gemv_kernel<256, 8>));
  touch(reinterpret_cast<const void*>(vhc::gemv_kernel<256, 4>));
  touch(reinterpret_cast<const void*>(vrh::rows_kernel<1024, 6>));
  touch(reinterpret_cast<const void*>(vrh::rows_kernel<1024, 5>));
  touch(reinterpret_cast<const void*>(vrh::rows_kernel<512, 10>));
  touch(reinterpret_cast<const void*>(vrh::rows_kernel<1280, 5>));
  touch(reinterpret_cast<const void*>(vrh::rows_kernel<512, 12>));
  touch(reinterpret_cast<const void*>(vrh::rows_kernel<1280, 4>));
  return n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack", &pack, "lossless 12-bit packing of a bf16 [n, 5120] head (s|m bytes + 4-bit exponent codes + escapes)");
  m.def("unpack", &unpack, "decode the packed head back to bf16 (escapes read +2^-15; tests)");
  m.def("gemv", &gemv, "packed bf16 head GEMV, M <= 8, fp32 accumulation, optional row gather");
  m.def("gemv_rows", &gemv_rows, "plain bf16 head GEMV (row range per SM, contiguous row bursts), M <= 8, fp32 accumulation, optional row gather");
  m.def("preload", &preload);
}
