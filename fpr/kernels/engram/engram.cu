// Engram on owned rows (layers 1 and 14 of DeepSeek-V4.1-Flash), sequence-parallel layout: every rank runs
// Engram for the rows of the sequences it owns, with all 24 hash columns, so no all-gather of looked-up rows and no
// all-gather of the wkv output (production: both, ~5.6 + 23 MB per Engram layer at T = 456).
//
// Pieces:
//   k_hash    n-gram hash ids of both Engram layers for every row; semantics = stock vLLM 0.32 `_hash_ids_kernel`
//             (common/engram.py:291-423, V2 runner path: history = in-batch rows, then the per-sequence lookback ids, no
//             slot cache), bit-exact by construction (integer math). Lookback ids come from the host (prefill chunks) or
//             from a per-sequence device ring of the last 16 (position, token) pairs keyed by the sequence's state slot
//             (decode), which the same kernel updates with the last <= 8 rows of every sequence.
//   k_gather  24 records per row and layer from the production host tables (ds41h engram_host packed layout: 272-B records =
//             256 B e4m3 + 8 B ue8m0 + 8 B pad, one shared pinned copy per layer in /dev/shm, zero-copy over PCIe), dequantized
//             EXACTLY like stock `_engram_lookup_kernel`: bf16_rn(fp32(e4m3) * fp32_bits(scale << 23)), no ftz (pinned PTX,
//             the same instruction chain as ds41h/kernels/engram_host.cu), stock's per-column ownership rule (an id outside
//             [lo[c], hi[c]) -> zeros).
//   k_inject  the gated injection (stock `_fused_engram_post_wkv_kernel`, engram.py:1188-1277) on the full hc stream, per row:
//             optional fused hc_post first (the fork's K1a phase-1 formula bit for bit: x = bf16(post[j] y + sum_c comb[c][j]
//             Rs[c]) as an fp32 FMA chain), then gate = sigmoid(signed sqrt(max(|dot|, clamp))), dot = <x, q k key> rms(x)
//             rms(key) / sqrt(D), out = bf16(x + gate value). Reductions and the scalar tail in fp64 (bf16 products are exact
//             in fp64), one rounding to bf16 at stock's rounding point -> closer to the fp32 reference than stock's fp32
//             tree sums + approximate rsqrt / exp (same rounding points, more precise math).
//   The wkv GEMM in between (rows [R, 6144] bf16 x wkv^T -> kv [R, 25600] bf16, fp32 accumulation, one rounding) is the
//   fork's W8A16 kernel (kernels/attn_dense/w8a16.cu, R <= 128 per call) or cuBLAS on the exact bf16 dequant (prefill).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace eg {

constexpr int NL = 2;        // Engram layers (1, 14)
constexpr int NG = 4;        // max n-gram size (shifts 0..3)
constexpr int NHD = 8;       // hash heads per n-gram order
constexpr int NC = 24;       // hash columns per layer = (NG - 1) * NHD
constexpr int DIM = 256;     // engram head dim
constexpr int SD = 8;        // ue8m0 scales per record
constexpr int D = 5120;      // hidden
constexpr int HC = 4;        // hc copies
constexpr int KVW = (HC + 1) * D;
constexpr int RING = 16;     // token-history ring entries per state slot (power of two)
constexpr int RING_W = 8;    // rows per sequence written to the ring per step (last <= 8)
constexpr int DEAD_TOKEN = 129264;  // IMAGE_SENTINEL_BASE_ID (common/mm_preprocess.py:51): image-span tokens

struct HashParams {
  long long mult[NL][NG];
  long long prime[NL][NC];
  long long off[NL][NC];
  int pad_id;    // token_map[engram_pad_token_id]
  int vocab;     // token_map size
};

// ------------------------------------------------------------------------------------------------ k_hash
// rows t < T: ids / pos / seq (index into the step's sequence list, -1 = padding row). Sequences s: row0, nrows, start
// (position of the first row), slot (state slot = ring row, -1 = none), look[s][0..2] = raw token ids at positions start-1,
// start-2, start-3 (-1 = unknown -> pad, as stock's fallback; -2 = read the device ring).
// Out: hash [T, NL, NC] int32 (-1 for padding rows), keep [T] u8 (stock engram_mask = token is not an image sentinel).
// err[0] += ring misses (entry for another position), err[1] += token ids outside the token map, err[2] += ring reads of a
// sequence with more than RING_W rows (would race with its own ring writes; the host must pass the lookback then).
// One thread per (row, head): it rebuilds the row's 4 n-gram values (cheap: 4 token-map loads) and computes the 6 ids of
// its head (2 layers x 3 orders): 6 int64 mods instead of 48 serial ones per thread. Head-0 threads count errors, write
// keep and the ring (ring reads only touch positions < start, ring writes only positions >= start of the same sequence:
// disjoint ring entries for <= RING_W rows, so the order between threads does not matter).
__global__ void __launch_bounds__(256) k_hash(const int* __restrict__ ids, const int* __restrict__ counts,
                                              const int* __restrict__ s_row0,
                                              const int* __restrict__ s_nrows, const int* __restrict__ s_start,
                                              const int* __restrict__ s_slot, const int* __restrict__ s_look,
                                              const int* __restrict__ tmap, int2* __restrict__ ring, int T,
                                              const __grid_constant__ HashParams hp, int* __restrict__ hash,
                                              unsigned char* __restrict__ keep, int* __restrict__ err) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = g / NHD, hd = g % NHD;
  if (t >= T) return;
  const bool lead = hd == 0;
  int* h = hash + (size_t)t * NL * NC;
  const int S = counts[0], n = counts[1];
  if (t >= n || S == 0) {  // padding row (the bucket's rows beyond this step's)
#pragma unroll
    for (int l = 0; l < NL; ++l)
#pragma unroll
      for (int o = 0; o < NG - 1; ++o) h[l * NC + o * NHD + hd] = -1;
    if (lead) keep[t] = 0;
    return;
  }
  // the row's sequence = the last one with row0 <= t (sequences in row order); its position = start + (t - row0)
  int lo_s = 0, hi_s = S - 1;
  while (lo_s < hi_s) {
    const int mid = (lo_s + hi_s + 1) >> 1;
    if (__ldg(s_row0 + mid) <= t) lo_s = mid; else hi_s = mid - 1;
  }
  const int s = lo_s;
  const int start = s_start[s], nrows = s_nrows[s], row0 = s_row0[s], slot = s_slot[s];
  const int p = start + (t - row0);
  const int tok0 = ids[t];
  long long roll[NL] = {0, 0};
  bool blocked = false;
#pragma unroll
  for (int sh = 0; sh < NG; ++sh) {
    const int lp = p - sh;
    long long src;
    if (lp >= start) {  // in batch: the row sh before (rows of a sequence are contiguous and position-consecutive)
      const int tk = ids[t - sh];
      if (tk < 0 || tk >= hp.vocab) {
        if (lead) atomicAdd(err + 1, 1);
        src = hp.pad_id;
      } else {
        src = tk == DEAD_TOKEN ? -1 : tmap[tk];
      }
    } else {
      const int col = start - 1 - lp;
      int wt = -1;
      if (col >= 0 && col < NG - 1) {
        wt = s_look[s * (NG - 1) + col];
        if (wt == -2) {
          wt = -1;
          if (lp >= 0) {
            if (nrows > RING_W) {
              if (lead) atomicAdd(err + 2, 1);
            } else if (slot >= 0) {
              const int2 e = ring[(size_t)slot * RING + (lp & (RING - 1))];
              if (e.x == lp) wt = e.y; else if (lead) atomicAdd(err, 1);
            }
          }
        }
      }
      if (wt >= hp.vocab) {
        if (lead) atomicAdd(err + 1, 1);
        wt = -1;
      }
      // stock: known = in_window & (window_token >= 0); unknown -> fallback = pad_id (not blocked: V2 has no slot cache)
      src = wt >= 0 ? (wt == DEAD_TOKEN ? -1 : tmap[wt]) : hp.pad_id;
    }
    blocked |= (lp < 0) || (src == -1);
    const long long v = blocked ? (long long)hp.pad_id : src;
#pragma unroll
    for (int l = 0; l < NL; ++l) {
      roll[l] ^= v * hp.mult[l][sh];
      if (sh > 0) {
        const int c = (sh - 1) * NHD + hd;
        h[l * NC + c] = (int)(roll[l] % hp.prime[l][c] + hp.off[l][c]);
      }
    }
  }
  if (lead) {
    keep[t] = tok0 != DEAD_TOKEN;
    if (slot >= 0 && t - row0 >= nrows - RING_W) ring[(size_t)slot * RING + (p & (RING - 1))] = make_int2(p, tok0);
  }
}

// ------------------------------------------------------------------------------------------------ k_gather
struct GatherParams {
  const uint8_t* tab[NL];   // device-visible address of record 0 of each layer's packed table
  __nv_bfloat16* out[NL];   // [T, NC, DIM] contiguous (row stride NC * DIM)
  int lidx[NL];             // hash-layer index of the selected layers (0 = layer 1, 1 = layer 14)
  int nl;                   // number of selected layers (1 or 2)
  long long lo[NL][NC], hi[NL][NC];
  const int* hash;          // [T, NL, NC]
  int T;
  int rows;                 // T * nl * NC
};

__device__ __forceinline__ uint32_t dequant2(uint32_t pair16, float scale) {  // = ds41h engram_host.cu dequant2
  uint32_t h2;
  asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(h2) : "r"(pair16));
  float f0, f1;
  asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %2; cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }" : "=f"(f0), "=f"(f1) : "r"(h2));
  asm("mul.rn.f32 %0, %0, %1;" : "+f"(f0) : "f"(scale));
  asm("mul.rn.f32 %0, %0, %1;" : "+f"(f1) : "f"(scale));
  uint32_t b2;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b2) : "f"(f1), "f"(f0));
  return b2;
}

__device__ __forceinline__ uint4 ld_nc16(const uint8_t* p, uint64_t pol) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}

// 16 lanes per record: lane l loads value bytes [16 l, 16 l + 16), lane 0 also the 8 scale bytes (16-B aligned: record
// offset 272 i + 256); the scale word is shuffled within the 16-lane group. U records in flight per group, grid-stride.
// Row order r = (t * nl + layer) * NC + c: the 48 records of a token are adjacent (one id fetch stream).
template <int U>
__global__ void __launch_bounds__(512) k_gather(const __grid_constant__ GatherParams p) {
  const int lane = threadIdx.x & 15;
  const int grp = (blockIdx.x * blockDim.x + threadIdx.x) >> 4;
  const int ngrp = (gridDim.x * blockDim.x) >> 4;
  const unsigned gmask = 0xffffu << (threadIdx.x & 16);
  uint64_t pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  for (int base = 0; base < p.rows; base += ngrp * U) {
    long long idx[U];
    int li[U];
    bool own[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int r = base + u * ngrp + grp;
      own[u] = false;
      idx[u] = 0;
      li[u] = 0;
      if (r < p.rows) {
        const int c = r % NC, tl = r / NC, l = tl % p.nl, t = tl / p.nl;
        const long long i = __ldg(p.hash + ((size_t)t * NL + p.lidx[l]) * NC + c);
        own[u] = i >= p.lo[l][c] && i < p.hi[l][c];
        idx[u] = i;
        li[u] = l;
      }
    }
    uint4 v[U];
    uint2 sc[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      v[u] = make_uint4(0, 0, 0, 0);
      sc[u] = make_uint2(0, 0);
      if (own[u]) {
        const uint8_t* rec = p.tab[li[u]] + idx[u] * 272;
        v[u] = ld_nc16(rec + 16 * lane, pol);
        if (lane == 0) {
          const uint4 s4 = ld_nc16(rec + 256, pol);
          sc[u] = make_uint2(s4.x, s4.y);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int r = base + u * ngrp + grp;
      const uint32_t s0 = __shfl_sync(gmask, sc[u].x, 0, 16), s1 = __shfl_sync(gmask, sc[u].y, 0, 16);
      if (r >= p.rows) continue;
      const int c = r % NC, tl = r / NC, l = tl % p.nl, t = tl / p.nl;
      uint4 o0 = make_uint4(0, 0, 0, 0), o1 = make_uint4(0, 0, 0, 0);
      if (own[u]) {
        const int b = lane >> 1;  // scale byte of this lane's 16 values (32-element blocks)
        const uint32_t sb = ((b < 4 ? s0 : s1) >> (8 * (b & 3))) & 0xffu;
        const float scale = __uint_as_float(sb << 23);
        o0.x = dequant2(v[u].x & 0xffffu, scale);
        o0.y = dequant2(v[u].x >> 16, scale);
        o0.z = dequant2(v[u].y & 0xffffu, scale);
        o0.w = dequant2(v[u].y >> 16, scale);
        o1.x = dequant2(v[u].z & 0xffffu, scale);
        o1.y = dequant2(v[u].z >> 16, scale);
        o1.z = dequant2(v[u].w & 0xffffu, scale);
        o1.w = dequant2(v[u].w >> 16, scale);
      }
      uint4* dst = reinterpret_cast<uint4*>(p.out[l] + ((size_t)t * NC + c) * DIM + 16 * lane);
      dst[0] = o0;
      dst[1] = o1;
    }
  }
}

// ------------------------------------------------------------------------------------------------ k_inject
// One CTA per row, 640 threads x 8 columns. MODE 0: x = x_in [T, 4, D]; MODE 1: x = hc_post(y, rs, post, comb) exactly as
// the fork's K1a (kernels/sp_layer/mhc.cu k_mhc_post_mix2 phase 1). kv row t at kv + t * kv_stride: [key_0 | .. | key_3 |
// value] (bf16). out [T, 4, D] bf16 may alias x_in / rs (every element is read by the thread that writes it, before).
constexpr int IT = 640;
constexpr int IW = IT / 32;

// MODE 2 (RS folded, = the fork's k1a_rs / comm.cu k_rs_sum): y = bf16(fixed-order fp32 sum of the 4 receive slots
// recv[s][t][:] in rank order) once *flag reached *epoch * signals (the peers' reduce-scatter pushes of this round).
struct Fold {
  const __nv_bfloat16* recv;  // [4, R, D]
  const unsigned* flag;
  const unsigned* epoch;
  unsigned signals;
  int R;
};
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ float bfr(float a) { return __bfloat162float(__float2bfloat16_rn(a)); }

template <int MODE>
__global__ void __launch_bounds__(IT) k_inject(const __nv_bfloat16* __restrict__ x_in, const __nv_bfloat16* __restrict__ y,
                                               const __nv_bfloat16* __restrict__ rs, const float* __restrict__ post,
                                               const float* __restrict__ comb, const __nv_bfloat16* __restrict__ kv,
                                               long long kv_stride, const __nv_bfloat16* __restrict__ qw,
                                               const __nv_bfloat16* __restrict__ kw, const unsigned char* __restrict__ keep,
                                               double eps, double clampv, __nv_bfloat16* __restrict__ out,
                                               float* __restrict__ gate_out, const int* __restrict__ kv_idx,
                                               Fold fo) {
  __shared__ double red[IW][12];
  __shared__ double gsm[HC];
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int k0 = 8 * tid;
  // kv row of this row: t, or kv_idx[t] (prefill row dedup: identical 24 ids -> one kv row)
  const __nv_bfloat16* kvr = kv + (size_t)(kv_idx != nullptr ? __ldg(kv_idx + t) : t) * kv_stride;
  uint4 xs[HC];  // x (bf16 x 8) per copy
  if (MODE == 0) {
#pragma unroll
    for (int j = 0; j < HC; ++j) xs[j] = *reinterpret_cast<const uint4*>(x_in + ((size_t)t * HC + j) * D + k0);
  } else {
    uint4 ly;
    if (MODE == 2) {
      __shared__ int go;
      if (tid == 0) {
        const unsigned target = *(volatile const unsigned*)fo.epoch * fo.signals;
        while ((int)(ld_acquire_sys(fo.flag) - target) < 0) __nanosleep(32);
        go = 1;
      }
      __syncthreads();
      uint4 q[4];
#pragma unroll
      for (int sl = 0; sl < 4; ++sl) q[sl] = __ldcg(reinterpret_cast<const uint4*>(fo.recv + ((size_t)sl * fo.R + t) * D + k0));
      __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(&ly);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 a = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&q[0])[i]);
#pragma unroll
        for (int sl = 1; sl < 4; ++sl) {
          const float2 b = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&q[sl])[i]);
          a.x += b.x;
          a.y += b.y;
        }
        o[i] = __floats2bfloat162_rn(a.x, a.y);
      }
      (void)go;
    } else {
      ly = *reinterpret_cast<const uint4*>(y + (size_t)t * D + k0);
    }
    uint4 lr[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) lr[c] = *reinterpret_cast<const uint4*>(rs + ((size_t)t * HC + c) * D + k0);
    float pj[HC], cm[HC][HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) pj[j] = __ldg(post + t * HC + j);
#pragma unroll
    for (int c = 0; c < HC; ++c)
#pragma unroll
      for (int j = 0; j < HC; ++j) cm[c][j] = __ldg(comb + t * 16 + c * HC + j);
    const __nv_bfloat16* yv = reinterpret_cast<const __nv_bfloat16*>(&ly);
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      __nv_bfloat16* xo = reinterpret_cast<__nv_bfloat16*>(&xs[j]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float a = pj[j] * __bfloat162float(yv[i]);
#pragma unroll
        for (int c = 0; c < HC; ++c) a = fmaf(cm[c][j], __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&lr[c])[i]), a);
        xo[i] = __float2bfloat16_rn(a);
      }
    }
  }
  // partial sums (fp64; bf16 x bf16 x bf16 x bf16 products are exact in fp64): |x|^2, |key|^2, <x, q k key> per copy
  double acc[12];
#pragma unroll
  for (int i = 0; i < 12; ++i) acc[i] = 0.0;
#pragma unroll
  for (int j = 0; j < HC; ++j) {
    const uint4 kr = *reinterpret_cast<const uint4*>(kvr + j * D + k0);
    const uint4 qv = __ldg(reinterpret_cast<const uint4*>(qw + j * D + k0));
    const uint4 kv8 = __ldg(reinterpret_cast<const uint4*>(kw + j * D + k0));
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xs[j]);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kr);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(&qv);
    const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&kv8);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const double x = (double)__bfloat162float(xb[i]), k = (double)__bfloat162float(kb[i]);
      acc[3 * j] = fma(x, x, acc[3 * j]);
      acc[3 * j + 1] = fma(k, k, acc[3 * j + 1]);
      acc[3 * j + 2] = fma(x * (double)__bfloat162float(qb[i]) * (double)__bfloat162float(wb[i]), k, acc[3 * j + 2]);
    }
  }
#pragma unroll
  for (int i = 0; i < 12; ++i) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], o);
  }
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 12; ++i) red[warp][i] = acc[i];
  }
  __syncthreads();
  if (tid < HC) {
    double hh = 0.0, kk = 0.0, hk = 0.0;
    for (int w = 0; w < IW; ++w) {  // fixed order: deterministic
      hh += red[w][3 * tid];
      kk += red[w][3 * tid + 1];
      hk += red[w][3 * tid + 2];
    }
    const double hr = 1.0 / sqrt(hh / D + eps), kr = 1.0 / sqrt(kk / D + eps);
    const double dot = hk * hr * kr / sqrt((double)D);
    double gi = sqrt(fmax(fabs(dot), clampv));
    if (dot < 0.0) gi = -gi;
    double g = 1.0 / (1.0 + exp(-gi));
    if (keep != nullptr && !keep[t]) g = 0.0;
    gsm[tid] = g;
    if (gate_out != nullptr) gate_out[t * HC + tid] = (float)g;
  }
  __syncthreads();
  const uint4 vr = *reinterpret_cast<const uint4*>(kvr + HC * D + k0);
  const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vr);
#pragma unroll
  for (int j = 0; j < HC; ++j) {
    const double g = gsm[j];
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xs[j]);
    uint4 o;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      ob[i] = __double2bfloat16(fma(g, (double)__bfloat162float(vb[i]), (double)__bfloat162float(xb[i])));
    *reinterpret_cast<uint4*>(out + ((size_t)t * HC + j) * D + k0) = o;
  }
}

// ------------------------------------------------------------------------------------------------ hc_post only (K1a phase 1)
// x = bf16(post[j] y + sum_c comb[c][j] Rs[c]) for the reference / the unfused path (same arithmetic as k_inject MODE 1).
__global__ void __launch_bounds__(IT) k_hc_post(const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ rs,
                                                const float* __restrict__ post, const float* __restrict__ comb,
                                                __nv_bfloat16* __restrict__ out) {
  const int t = blockIdx.x, k0 = 8 * threadIdx.x;
  const uint4 ly = *reinterpret_cast<const uint4*>(y + (size_t)t * D + k0);
  uint4 lr[HC];
#pragma unroll
  for (int c = 0; c < HC; ++c) lr[c] = *reinterpret_cast<const uint4*>(rs + ((size_t)t * HC + c) * D + k0);
  const __nv_bfloat16* yv = reinterpret_cast<const __nv_bfloat16*>(&ly);
#pragma unroll
  for (int j = 0; j < HC; ++j) {
    const float pj = __ldg(post + t * HC + j);
    uint4 o;
    __nv_bfloat16* xo = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float a = pj * __bfloat162float(yv[i]);
#pragma unroll
      for (int c = 0; c < HC; ++c)
        a = fmaf(__ldg(comb + t * 16 + c * HC + j), __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&lr[c])[i]), a);
      xo[i] = __float2bfloat16_rn(a);
    }
    *reinterpret_cast<uint4*>(out + ((size_t)t * HC + j) * D + k0) = o;
  }
}

// ------------------------------------------------------------------------------------------------ dedup helpers (prefill)
// Expand rows gathered for unique ids back to [T, NC, DIM]: out[(t * NC + c)] = urows[inv[t * nl * NC + l * NC + c]] for
// one selected layer l (inv over the flattened [T, nl, NC] id list). 16 B per thread.
__global__ void k_expand(const __nv_bfloat16* __restrict__ urows, const int* __restrict__ inv, int T, int nl, int l,
                         __nv_bfloat16* __restrict__ out) {
  const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;  // 16-B chunk index
  const long long total = (long long)T * NC * (DIM / 8);
  if (e >= total) return;
  const long long rc = e / (DIM / 8);
  const int q = (int)(e % (DIM / 8));
  const int t = (int)(rc / NC), c = (int)(rc % NC);
  const int u = inv[((size_t)t * nl + l) * NC + c];
  reinterpret_cast<uint4*>(out + rc * DIM)[q] = reinterpret_cast<const uint4*>(urows + (size_t)u * DIM)[q];
}

// Gather unique ids (any table): ids [U] int64 global row ids of layer `tab`, out [U, DIM] (ownership: id < 0 -> zeros).
template <int UN>
__global__ void __launch_bounds__(512) k_gather_ids(const uint8_t* __restrict__ tab, const long long* __restrict__ ids,
                                                    int n, __nv_bfloat16* __restrict__ out) {
  const int lane = threadIdx.x & 15;
  const int grp = (blockIdx.x * blockDim.x + threadIdx.x) >> 4;
  const int ngrp = (gridDim.x * blockDim.x) >> 4;
  const unsigned gmask = 0xffffu << (threadIdx.x & 16);
  uint64_t pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  for (int base = 0; base < n; base += ngrp * UN) {
    long long idx[UN];
    uint4 v[UN];
    uint2 sc[UN];
#pragma unroll
    for (int u = 0; u < UN; ++u) {
      const int r = base + u * ngrp + grp;
      idx[u] = r < n ? __ldg(ids + r) : -1;
      v[u] = make_uint4(0, 0, 0, 0);
      sc[u] = make_uint2(0, 0);
      if (idx[u] >= 0) {
        const uint8_t* rec = tab + idx[u] * 272;
        v[u] = ld_nc16(rec + 16 * lane, pol);
        if (lane == 0) {
          const uint4 s4 = ld_nc16(rec + 256, pol);
          sc[u] = make_uint2(s4.x, s4.y);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < UN; ++u) {
      const int r = base + u * ngrp + grp;
      const uint32_t s0 = __shfl_sync(gmask, sc[u].x, 0, 16), s1 = __shfl_sync(gmask, sc[u].y, 0, 16);
      if (r >= n) continue;
      uint4 o0 = make_uint4(0, 0, 0, 0), o1 = make_uint4(0, 0, 0, 0);
      if (idx[u] >= 0) {
        const int b = lane >> 1;
        const uint32_t sb = ((b < 4 ? s0 : s1) >> (8 * (b & 3))) & 0xffu;
        const float scale = __uint_as_float(sb << 23);
        o0.x = dequant2(v[u].x & 0xffffu, scale);
        o0.y = dequant2(v[u].x >> 16, scale);
        o0.z = dequant2(v[u].y & 0xffffu, scale);
        o0.w = dequant2(v[u].y >> 16, scale);
        o1.x = dequant2(v[u].z & 0xffffu, scale);
        o1.y = dequant2(v[u].z >> 16, scale);
        o1.z = dequant2(v[u].w & 0xffffu, scale);
        o1.w = dequant2(v[u].w >> 16, scale);
      }
      uint4* dst = reinterpret_cast<uint4*>(out + (size_t)r * DIM + 16 * lane);
      dst[0] = o0;
      dst[1] = o1;
    }
  }
}

// ------------------------------------------------------------------------------------------------ PCIe probe
// Read n records of 272 B at the given ids (no dequant, XOR-reduced so nothing is optimized away): pure host-read rate.
__global__ void __launch_bounds__(512) k_probe(const uint8_t* __restrict__ tab, const long long* __restrict__ ids, int n,
                                               unsigned* __restrict__ sink) {
  const int lane = threadIdx.x & 15;
  const int grp = (blockIdx.x * blockDim.x + threadIdx.x) >> 4;
  const int ngrp = (gridDim.x * blockDim.x) >> 4;
  uint64_t pol;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  unsigned x = 0;
  for (int r = grp; r < n; r += ngrp) {
    const uint4 v = ld_nc16(tab + __ldg(ids + r) * 272 + 16 * lane, pol);
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (x == 0x9e3779b9u) sink[0] = x;
}

}  // namespace eg

// ================================================================================================== bindings
namespace {

eg::HashParams make_hash_params(torch::Tensor mult, torch::Tensor prime, torch::Tensor off, int64_t pad_id, int64_t vocab) {
  TORCH_CHECK(mult.dtype() == torch::kInt64 && mult.numel() == eg::NL * eg::NG, "mult [2, 4] int64 (cpu)");
  TORCH_CHECK(prime.dtype() == torch::kInt64 && prime.numel() == eg::NL * eg::NC, "prime [2, 24] int64 (cpu)");
  TORCH_CHECK(off.dtype() == torch::kInt64 && off.numel() == eg::NL * eg::NC, "off [2, 24] int64 (cpu)");
  auto m = mult.contiguous().cpu(), p = prime.contiguous().cpu(), o = off.contiguous().cpu();
  eg::HashParams hp{};
  for (int l = 0; l < eg::NL; ++l) {
    for (int s = 0; s < eg::NG; ++s) hp.mult[l][s] = m.data_ptr<int64_t>()[l * eg::NG + s];
    for (int c = 0; c < eg::NC; ++c) {
      hp.prime[l][c] = p.data_ptr<int64_t>()[l * eg::NC + c];
      hp.off[l][c] = o.data_ptr<int64_t>()[l * eg::NC + c];
    }
  }
  hp.pad_id = static_cast<int>(pad_id);
  hp.vocab = static_cast<int>(vocab);
  return hp;
}

struct HashCtx {  // host-side constant parameters (built once)
  eg::HashParams hp;
};
std::vector<HashCtx> g_hash;

int64_t hash_setup(torch::Tensor mult, torch::Tensor prime, torch::Tensor off, int64_t pad_id, int64_t vocab) {
  g_hash.push_back({make_hash_params(mult, prime, off, pad_id, vocab)});
  return static_cast<int64_t>(g_hash.size() - 1);
}

void chk_i32(const torch::Tensor& t, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kInt32 && t.is_contiguous(), n, ": contiguous int32 CUDA");
}

// all inputs int32 CUDA: ids [>= T]; counts [>= 2] = (sequences, real rows) in device memory; s_row0/s_nrows/s_start/
// s_slot [S]; s_look [S, 3]; tmap [V]; ring int32 [slots, RING, 2]; out hash [>= T, 2, 24] int32, keep [>= T] uint8,
// err [3] int32. Rows of a sequence are contiguous and position-consecutive (StepBatch order), so a row's sequence and
// position come from the per-sequence arrays: no per-row host data.
void hash(int64_t ctx, torch::Tensor ids, torch::Tensor counts, torch::Tensor s_row0, torch::Tensor s_nrows,
          torch::Tensor s_start, torch::Tensor s_slot, torch::Tensor s_look, torch::Tensor tmap, torch::Tensor ring, int64_t T,
          torch::Tensor out, torch::Tensor keep, torch::Tensor err) {
  const c10::cuda::CUDAGuard guard_(ids.device());
  TORCH_CHECK(ctx >= 0 && ctx < (int64_t)g_hash.size(), "hash ctx");
  for (auto* p : {&ids, &counts, &s_row0, &s_nrows, &s_start, &s_slot, &s_look, &tmap, &ring, &out, &err}) chk_i32(*p, "hash arg");
  TORCH_CHECK(keep.is_cuda() && keep.dtype() == torch::kUInt8, "keep uint8");
  TORCH_CHECK(ring.dim() == 3 && ring.size(1) == eg::RING && ring.size(2) == 2, "ring [slots, 16, 2]");
  TORCH_CHECK(out.numel() >= T * eg::NL * eg::NC && ids.numel() >= T && keep.numel() >= T, "sizes");
  if (T == 0) return;
  auto st = c10::cuda::getCurrentCUDAStream(ids.device().index()).stream();
  eg::k_hash<<<(T * eg::NHD + 255) / 256, 256, 0, st>>>(ids.data_ptr<int>(), counts.data_ptr<int>(),
                                              s_row0.data_ptr<int>(), s_nrows.data_ptr<int>(), s_start.data_ptr<int>(),
                                              s_slot.data_ptr<int>(), s_look.data_ptr<int>(), tmap.data_ptr<int>(),
                                              reinterpret_cast<int2*>(ring.data_ptr<int>()), (int)T, g_hash[ctx].hp,
                                              out.data_ptr<int>(), keep.data_ptr<uint8_t>(), err.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// tabs: device addresses of record 0 per selected layer; lidx: hash-layer index per selected layer; outs: bf16 [T, 24, 256];
// lo / hi: [nl][24] ownership ranges. ctas / threads / unroll: launch shape (ctas = 0: enough CTAs for one record per group).
void gather(torch::Tensor hash_ids, int64_t T, std::vector<int64_t> tabs, std::vector<int64_t> lidx, std::vector<torch::Tensor> outs,
            std::vector<int64_t> lo, std::vector<int64_t> hi, int64_t ctas, int64_t threads, int64_t unroll) {
  const c10::cuda::CUDAGuard guard_(hash_ids.device());
  chk_i32(hash_ids, "hash");
  const int nl = static_cast<int>(tabs.size());
  TORCH_CHECK(nl >= 1 && nl <= eg::NL && (int)lidx.size() == nl && (int)outs.size() == nl, "layers");
  TORCH_CHECK((int)lo.size() == nl * eg::NC && (int)hi.size() == nl * eg::NC, "lo/hi");
  TORCH_CHECK(threads >= 32 && threads <= 512 && threads % 32 == 0, "threads");
  if (T == 0) return;
  eg::GatherParams p{};
  for (int l = 0; l < nl; ++l) {
    TORCH_CHECK(outs[l].is_cuda() && outs[l].dtype() == torch::kBFloat16 && outs[l].is_contiguous() &&
                    outs[l].numel() >= T * eg::NC * eg::DIM, "out bf16 [T, 24, 256]");
    TORCH_CHECK(tabs[l] % 16 == 0, "table base 16-B aligned");
    p.tab[l] = reinterpret_cast<const uint8_t*>(tabs[l]);
    p.out[l] = reinterpret_cast<__nv_bfloat16*>(outs[l].data_ptr());
    p.lidx[l] = static_cast<int>(lidx[l]);
    for (int c = 0; c < eg::NC; ++c) {
      p.lo[l][c] = lo[l * eg::NC + c];
      p.hi[l][c] = hi[l * eg::NC + c];
    }
  }
  p.nl = nl;
  p.hash = hash_ids.data_ptr<int>();
  p.T = static_cast<int>(T);
  p.rows = static_cast<int>(T * nl * eg::NC);
  const int gpc = static_cast<int>(threads) / 16;
  const int need = (p.rows + gpc * (int)unroll - 1) / (gpc * (int)unroll);
  const int grid = ctas > 0 ? std::max(1, std::min<int>((int)ctas, need)) : std::max(1, need);
  auto st = c10::cuda::getCurrentCUDAStream(hash_ids.device().index()).stream();
  switch (unroll) {
    case 1: eg::k_gather<1><<<grid, threads, 0, st>>>(p); break;
    case 2: eg::k_gather<2><<<grid, threads, 0, st>>>(p); break;
    case 4: eg::k_gather<4><<<grid, threads, 0, st>>>(p); break;
    case 8: eg::k_gather<8><<<grid, threads, 0, st>>>(p); break;
    default: TORCH_CHECK(false, "unroll 1/2/4/8");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_ids(int64_t tab, torch::Tensor ids, torch::Tensor out, int64_t ctas, int64_t threads, int64_t unroll) {
  const c10::cuda::CUDAGuard guard_(ids.device());
  TORCH_CHECK(ids.is_cuda() && ids.dtype() == torch::kInt64 && ids.is_contiguous(), "ids int64");
  TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.numel() >= ids.numel() * eg::DIM, "out");
  const int n = static_cast<int>(ids.numel());
  if (n == 0) return;
  const int gpc = static_cast<int>(threads) / 16;
  const int need = (n + gpc * (int)unroll - 1) / (gpc * (int)unroll);
  const int grid = ctas > 0 ? std::max(1, std::min<int>((int)ctas, need)) : std::max(1, need);
  auto st = c10::cuda::getCurrentCUDAStream(ids.device().index()).stream();
  auto tp = reinterpret_cast<const uint8_t*>(tab);
  auto o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  switch (unroll) {
    case 1: eg::k_gather_ids<1><<<grid, threads, 0, st>>>(tp, reinterpret_cast<const long long*>(ids.data_ptr<int64_t>()), n, o); break;
    case 2: eg::k_gather_ids<2><<<grid, threads, 0, st>>>(tp, reinterpret_cast<const long long*>(ids.data_ptr<int64_t>()), n, o); break;
    case 4: eg::k_gather_ids<4><<<grid, threads, 0, st>>>(tp, reinterpret_cast<const long long*>(ids.data_ptr<int64_t>()), n, o); break;
    case 8: eg::k_gather_ids<8><<<grid, threads, 0, st>>>(tp, reinterpret_cast<const long long*>(ids.data_ptr<int64_t>()), n, o); break;
    default: TORCH_CHECK(false, "unroll 1/2/4/8");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void expand(torch::Tensor urows, torch::Tensor inv, int64_t T, int64_t nl, int64_t l, torch::Tensor out) {
  const c10::cuda::CUDAGuard guard_(inv.device());
  chk_i32(inv, "inv");
  if (T == 0) return;
  const long long total = T * eg::NC * (eg::DIM / 8);
  auto st = c10::cuda::getCurrentCUDAStream(inv.device().index()).stream();
  eg::k_expand<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(urows.data_ptr()),
                                                                  inv.data_ptr<int>(), (int)T, (int)nl, (int)l,
                                                                  reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void probe(int64_t tab, torch::Tensor ids, torch::Tensor sink, int64_t ctas, int64_t threads) {
  const c10::cuda::CUDAGuard guard_(ids.device());
  const int n = static_cast<int>(ids.numel());
  auto st = c10::cuda::getCurrentCUDAStream(ids.device().index()).stream();
  eg::k_probe<<<(int)ctas, (int)threads, 0, st>>>(reinterpret_cast<const uint8_t*>(tab), reinterpret_cast<const long long*>(ids.data_ptr<int64_t>()), n,
                                                  reinterpret_cast<unsigned*>(sink.data_ptr<int>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x_in: bf16 [T, 4, 5120] (mode 0) or y [T, 5120] + rs [T, 4, 5120] + post [T, 4] + comb [T, 4, 4] fp32 (mode 1);
// kv bf16 [T, >= 25600] (row stride kv.stride(0)); qw / kw bf16 [4, 5120]; keep uint8 [T] or empty; out bf16 [T, 4, 5120];
// gate_out fp32 [T, 4] or empty.
void inject(int64_t T, c10::optional<torch::Tensor> x_in, c10::optional<torch::Tensor> y, c10::optional<torch::Tensor> rs,
            c10::optional<torch::Tensor> post, c10::optional<torch::Tensor> comb, torch::Tensor kv, torch::Tensor qw,
            torch::Tensor kw, c10::optional<torch::Tensor> keep, double eps, double clampv, torch::Tensor out,
            c10::optional<torch::Tensor> gate_out, c10::optional<torch::Tensor> kv_idx) {
  const c10::cuda::CUDAGuard guard_(kv.device());
  TORCH_CHECK(kv.dtype() == torch::kBFloat16 && kv.stride(1) == 1 && kv.size(1) >= eg::KVW, "kv bf16 [T, 25600]");
  TORCH_CHECK(qw.dtype() == torch::kBFloat16 && qw.is_contiguous() && qw.numel() == eg::HC * eg::D, "qw");
  TORCH_CHECK(kw.dtype() == torch::kBFloat16 && kw.is_contiguous() && kw.numel() == eg::HC * eg::D, "kw");
  TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous(), "out");
  if (T == 0) return;
  auto st = c10::cuda::getCurrentCUDAStream(kv.device().index()).stream();
  const unsigned char* kp = keep.has_value() && keep->numel() ? keep->data_ptr<uint8_t>() : nullptr;
  float* gp = gate_out.has_value() && gate_out->numel() ? gate_out->data_ptr<float>() : nullptr;
  const int* ki = nullptr;
  if (kv_idx.has_value() && kv_idx->numel()) {
    chk_i32(*kv_idx, "kv_idx");
    ki = kv_idx->data_ptr<int>();
  }
  auto bf = [](const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  auto ob = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if (x_in.has_value() && x_in->numel()) {
    TORCH_CHECK(x_in->dtype() == torch::kBFloat16 && x_in->is_contiguous(), "x_in");
    eg::k_inject<0><<<(int)T, eg::IT, 0, st>>>(bf(*x_in), nullptr, nullptr, nullptr, nullptr, bf(kv), kv.stride(0), bf(qw), bf(kw),
                                              kp, eps, clampv, ob, gp, ki, eg::Fold{});
  } else {
    TORCH_CHECK(y.has_value() && rs.has_value() && post.has_value() && comb.has_value(), "mode 1 needs y, rs, post, comb");
    TORCH_CHECK(y->dtype() == torch::kBFloat16 && y->is_contiguous() && rs->dtype() == torch::kBFloat16 && rs->is_contiguous(), "y/rs");
    TORCH_CHECK(post->dtype() == torch::kFloat32 && post->is_contiguous() && comb->dtype() == torch::kFloat32 && comb->is_contiguous(),
                "post/comb fp32");
    eg::k_inject<1><<<(int)T, eg::IT, 0, st>>>(nullptr, bf(*y), bf(*rs), post->data_ptr<float>(), comb->data_ptr<float>(), bf(kv),
                                              kv.stride(0), bf(qw), bf(kw), kp, eps, clampv, ob, gp, ki, eg::Fold{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// MODE 2: y from the RS receive slots (recv bf16 [4, R, D], flag / epoch uint32 device words, as kernels/sp_layer/mhc.cu
// k1a_rs): the Engram layers' attention boundary can take the folded RS round directly.
void inject_rs(int64_t T, torch::Tensor recv, torch::Tensor flag, torch::Tensor epoch, int64_t signals, torch::Tensor rs,
               torch::Tensor post, torch::Tensor comb, torch::Tensor kv, torch::Tensor qw, torch::Tensor kw,
               c10::optional<torch::Tensor> keep, double eps, double clampv, torch::Tensor out,
               c10::optional<torch::Tensor> gate_out, c10::optional<torch::Tensor> kv_idx) {
  const c10::cuda::CUDAGuard guard_(kv.device());
  TORCH_CHECK(recv.dtype() == torch::kBFloat16 && recv.is_contiguous() && recv.dim() == 3 && recv.size(0) == 4 &&
                  recv.size(2) == eg::D && recv.size(1) >= T, "recv bf16 [4, R, 5120]");
  TORCH_CHECK(kv.dtype() == torch::kBFloat16 && kv.stride(1) == 1 && kv.size(1) >= eg::KVW, "kv");
  if (T == 0) return;
  auto st = c10::cuda::getCurrentCUDAStream(kv.device().index()).stream();
  const unsigned char* kp = keep.has_value() && keep->numel() ? keep->data_ptr<uint8_t>() : nullptr;
  float* gp = gate_out.has_value() && gate_out->numel() ? gate_out->data_ptr<float>() : nullptr;
  const int* ki = kv_idx.has_value() && kv_idx->numel() ? kv_idx->data_ptr<int>() : nullptr;
  auto bf = [](const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  eg::Fold fo{bf(recv), reinterpret_cast<const unsigned*>(flag.data_ptr()), reinterpret_cast<const unsigned*>(epoch.data_ptr()),
              static_cast<unsigned>(signals), static_cast<int>(recv.size(1))};
  eg::k_inject<2><<<(int)T, eg::IT, 0, st>>>(nullptr, nullptr, bf(rs), post.data_ptr<float>(), comb.data_ptr<float>(), bf(kv),
                                            kv.stride(0), bf(qw), bf(kw), kp, eps, clampv,
                                            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), gp, ki, fo);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hc_post(int64_t T, torch::Tensor y, torch::Tensor rs, torch::Tensor post, torch::Tensor comb, torch::Tensor out) {
  const c10::cuda::CUDAGuard guard_(y.device());
  if (T == 0) return;
  auto st = c10::cuda::getCurrentCUDAStream(y.device().index()).stream();
  eg::k_hc_post<<<(int)T, eg::IT, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()),
                                           reinterpret_cast<const __nv_bfloat16*>(rs.data_ptr()), post.data_ptr<float>(),
                                           comb.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<int64_t, std::string, int64_t> host_register(int64_t ptr, int64_t bytes, bool read_only) {
  unsigned flags = cudaHostRegisterMapped | cudaHostRegisterPortable;
  if (read_only) flags |= cudaHostRegisterReadOnly;
  void* hp = reinterpret_cast<void*>(ptr);
  cudaError_t e = cudaHostRegister(hp, static_cast<size_t>(bytes), flags);
  if (e == cudaErrorHostMemoryAlreadyRegistered) {
    (void)cudaGetLastError();
    e = cudaSuccess;
  }
  if (e != cudaSuccess) {
    std::string msg = cudaGetErrorString(e);
    (void)cudaGetLastError();
    return {static_cast<int64_t>(e), msg, 0};
  }
  void* dp = nullptr;
  e = cudaHostGetDevicePointer(&dp, hp, 0);
  if (e != cudaSuccess) {
    std::string msg = cudaGetErrorString(e);
    (void)cudaGetLastError();
    return {static_cast<int64_t>(e), msg, 0};
  }
  return {0, "", reinterpret_cast<int64_t>(dp)};
}

int64_t host_unregister(int64_t ptr) { return static_cast<int64_t>(cudaHostUnregister(reinterpret_cast<void*>(ptr))); }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hash_setup", &hash_setup);
  m.def("hash", &hash);
  m.def("gather", &gather);
  m.def("gather_ids", &gather_ids);
  m.def("expand", &expand);
  m.def("probe", &probe);
  m.def("inject", &inject);
  m.def("inject_rs", &inject_rs);
  m.def("hc_post", &hc_post);
  m.def("host_register", &host_register);
  m.def("host_unregister", &host_unregister);
}
