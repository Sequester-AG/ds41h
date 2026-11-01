// Skinny FP8-weight GEMM for decode (M <= 8 tokens) on sm90a:
//   y[M, N] (bf16) = x[M, K] (bf16) . W[N, K]^T,  W = e4m3 with ue8m0 32x32 block scales (DeepSeek-V4.1).
//
// Adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
// src/brrr/kernels/skinny.cu + fp8_rows.cuh, see LICENSES/BRRR-Apache-2.0.txt). BRRR uses the
// Blackwell block-scaled mma (kind::mxf8f6f4); sm90 has no FP8 mma.sync (it lowers to an F2FP unpack +
// HMMA anyway), so this port runs FP16 HMMA m16n8k16 with a per-k32 post-scale:
//   * weights are the A operand (16 output rows per warp tile), expanded e4m3 -> f16 exactly with
//     cvt.rn.f16x2.e4m3x2 (every e4m3 value is an f16 value);
//   * the <= 8 tokens are the B operand (n8), staged once per CTA in shared memory as f16 scaled by a
//     per-(token, k32) power of two so the block max lands in [2^14, 2^15): exact for every bf16 input
//     whose magnitude is within 2^-31 of its block max (bf16 has 8 significant bits, f16 11), no overflow;
//   * each k32 block is two HMMAs into a zeroed temporary, then acc += tmp * 2^(e_w + e_x) (exact power
//     of two), so products are exact and accumulation is fp32 -- the same numerics class as stock vLLM's
//     W8A16 path on sm90 (Humming, bf16 activations), not the W8A8 block-scaled path BRRR used.
//
// Weight layout ("skinny fragment-major", produced by ds41h/skinny.py:pack): for 16-row tile rt and
// k32 block kb a 512-byte chunk at ((rt * KB + kb) * 512); lane l = 4g + t owns bytes [16l, 16l + 16):
//   bytes 0..7  = W[16rt + g    ][32kb + 8t .. 32kb + 8t + 8)
//   bytes 8..15 = W[16rt + g + 8][32kb + 8t .. 32kb + 8t + 8)
// so one 16-byte ld.global.nc.L1::no_allocate per lane per k32 block, 512 contiguous bytes per warp.
// The k order inside a 32-block is permuted (lane t's 8 bytes feed k = 2t,2t+1 / 2t+8,2t+9 of both
// HMMAs); the activation fragment applies the same permutation, which is legal because the scale is
// constant over the block. Scales: uint8 [N/16, KBP] (KBP = round_up(K/32, 4)), one ue8m0 per
// (16-row tile, k32 block) -- the checkpoint's 32x32 blocks cover whole 16-row tiles.
//
// RS-wgmma path (mma = MMA_WGMMA, env DS41H_PROLOGUE_MMA=wgmma; default MMA_HMMA until validated on GPU):
// the same packed weights feed wgmma.mma_async.m64n8k32.f32.e4m3.e4m3 as the register A operand -- a
// warpgroup = 4 warps = 4 (arbitrary) 16-row tiles that share one k32 block of activations, which are the
// B operand (8 tokens x 32 k, e4m3) read from shared memory through a no-swizzle K-major descriptor.
//   * A fragment (PTX wgmma 8-bit A, per warp): a0 = (row g, k 4t..4t+3), a1 = (row g+8, k 4t..),
//     a2 = (row g, k 16+4t..), a3 = (row g+8, k 16+4t..). The packed lane word is fed as
//     {a0,a1,a2,a3} = {v.x, v.z, v.y, v.w}, i.e. physical k p = 8t + 4h + i is wgmma ("logical") k
//     j = 16h + 4t + i; the B tile is written in logical order (stage_e4m3), so no weight repack.
//   * activations are quantized exactly like the reference act_quant (inference/kernel.py, model.py
//     linear(): per (token, k32) ue8m0 scale 2^ceil(log2(max(amax, 1e-4) / 448)), e4m3 RN satfinite), so
//     this path computes the official W8A8 numerics (stock vLLM sm90 runs W8A16; KL-check before use);
//   * every wgmma starts from zero (scale-d = 0) and is promoted on its own: acc += D_kb * 2^(sw_kb) *
//     2^(sx_tok,kb) (one FFMA per accumulator, factors pre-multiplied per warp in shared memory). Hopper's
//     FP8 tensor-core accumulator is narrower than fp32, so promoting every k32 (32 products) keeps the
//     result at fp32-accumulation quality. Two groups of 2 accumulator sets are in flight.
// Per 512-byte k32 chunk a warp issues (SASS, measured) 1 LDG.128 + 1 QGMMA + 1 LDS.64 + 4 FFMA + ~1
// WARPGROUP.ARRIVE/DEPBAR + ~2.5 uniform descriptor ops ~= 10 instructions (0.020 per weight byte), instead
// of ~30 on the HMMA path (8 F2FP unpacks, 2 HMMA, LDS.128, per-token scale math, 4 FFMA; 0.060 per byte).
//
// This file is also #included by prologue.cu with DS41H_SKINNY_DEVICE_ONLY defined (device code only).
#ifndef DS41H_SKINNY_CU
#define DS41H_SKINNY_CU

#include "common.cuh"
#include "sm90.cuh"

namespace ds41h_skinny {

constexpr int MAXM = 8;         // tokens per call (B operand n8)
constexpr int KW_MAX = 20;      // k32 blocks one warp keeps in registers (20 x 16 B = 80 regs)
constexpr int CHUNK = 512;      // bytes per (16-row tile, k32 block)
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int MAX_K = WARPS * KW_MAX * 32;  // 5120: a CTA covers all of K
constexpr unsigned FULL = 0xffffffffu;

// Shared-memory row pitch (bytes) of the f16 activation tile: 16-byte lane reads of rows g / g+1 must
// fall in opposite 64-byte bank halves (pitch == 64 mod 128) to be conflict free.
__host__ __device__ constexpr int act_pitch(int k) { return 2 * k + (((2 * k) & 127) == 0 ? 64 : 0); }
__host__ __device__ constexpr int round_up(int a, int b) { return (a + b - 1) / b * b; }

// e4m3 bytes 0,1 / 2,3 of w -> f16x2 (exact). The halves are taken with mov.b32 {lo, hi} so ptxas reads the upper half
// in place (F2FP.F16.E4M3.UNPACK_B R, w.H1) instead of emitting a shift per conversion.
__device__ __forceinline__ unsigned f16x2_lo(unsigned w) {
  unsigned r;
  asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, lo;\n}" : "=r"(r) : "r"(w));
  return r;
}
__device__ __forceinline__ unsigned f16x2_hi(unsigned w) {
  unsigned r;
  asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, hi;\n}" : "=r"(r) : "r"(w));
  return r;
}

// m16n8k16 f16 HMMA with a zero accumulator input (C = 0 lets ptxas use RZ).
__device__ __forceinline__ void hmma_f16_zero(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
}

// 2^((eb - 127) + (sw - 127)): weight block scale times the activation staging scale (both biased).
__device__ __forceinline__ float post_scale(unsigned sw, unsigned eb) {
  const int e = int(sw + eb) - 127;
  return __uint_as_float(unsigned(min(max(e, 0), 254)) << 23);
}

// One warp's weights: `cnt` (<= KW_MAX) consecutive k32 blocks of one 16-row tile, prefetched into
// registers. s: lane l holds the 4 scale bytes of blocks kb0 + 4l .. kb0 + 4l + 3.
struct Frag {
  uint4 w[KW_MAX];
  unsigned s;
};

// kb0 must be a multiple of 4 and kb_pitch = round_up(kb_count, 4) (scale words are 4-byte loads).
__device__ __forceinline__ void load_frag(Frag& f, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s, long tile, int kb_count,
                                          int kb_pitch, int kb0, int cnt, int lane) {
  const uint8_t* base = w + ((tile * kb_count + kb0) * 32 + lane) * 16;
#pragma unroll
  for (int i = 0; i < KW_MAX; ++i)
    if (i < cnt) f.w[i] = ld_stream16(base + i * CHUNK);
  f.s = lane * 4 < cnt ? ld_stream4(s + tile * kb_pitch + kb0 + lane * 4) : 0u;
}

// acc[0] = (row g, token 2t), acc[1] = (row g, token 2t+1), acc[2] = (row g+8, 2t), acc[3] = (row g+8, 2t+1)
// (HMMA C fragment, rows relative to the tile). act/xe hold the staged activations; kba0 is the
// activation-tile k32 index matching the first weight block of the fragment.
__device__ __forceinline__ void mma_frag(const Frag& f, const uint8_t* act, int pitch, const uint8_t* xe, int kba0, int cnt, int lane,
                                         float (&acc)[4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* arow = act + g * pitch + t * 16 + kba0 * 64;
  const uint8_t* xrow = xe + kba0 * MAXM + 2 * t;
#pragma unroll
  for (int i = 0; i < KW_MAX; ++i) {
    if (i < cnt) {
      const uint4 b = *reinterpret_cast<const uint4*>(arow + i * 64);  // token g, k = 32kb + 8t .. +8 (f16)
      const uint4 v = f.w[i];
      float d[4];
      // HMMA 1: (k 2t,2t+1) <- bytes 8t+0,1 ; (k 2t+8,2t+9) <- bytes 8t+2,3 (rows g / g+8)
      hmma_f16_zero(d, f16x2_lo(v.x), f16x2_lo(v.z), f16x2_hi(v.x), f16x2_hi(v.z), b.x, b.y);
      // HMMA 2: bytes 8t+4,5 and 8t+6,7
      hmma_f16(d, f16x2_lo(v.y), f16x2_lo(v.w), f16x2_hi(v.y), f16x2_hi(v.w), b.z, b.w);
      const unsigned sw = (__shfl_sync(FULL, f.s, i >> 2) >> (8 * (i & 3))) & 0xffu;
      const unsigned ex = *reinterpret_cast<const unsigned short*>(xrow + i * MAXM);  // tokens 2t, 2t+1
      const float f0 = post_scale(sw, ex & 0xffu), f1 = post_scale(sw, ex >> 8);
      acc[0] = fmaf(d[0], f0, acc[0]);
      acc[1] = fmaf(d[1], f1, acc[1]);
      acc[2] = fmaf(d[2], f0, acc[2]);
      acc[3] = fmaf(d[3], f1, acc[3]);
    }
  }
}

// Store the C fragment into a [16 rows][MAXM tokens] fp32 tile (shared or global).
__device__ __forceinline__ void store_frag_tile(float* tile, const float (&acc)[4], int lane) {
  const int g = lane >> 2, t = lane & 3;
  *reinterpret_cast<float2*>(tile + g * MAXM + 2 * t) = make_float2(acc[0], acc[1]);
  *reinterpret_cast<float2*>(tile + (g + 8) * MAXM + 2 * t) = make_float2(acc[2], acc[3]);
}

// In-place conversion of a staged activation tile: rows tok < m hold raw bf16 (pitch layout), rows
// >= m are zero-filled. xf(tok, c8, f) may transform the 8 values of vector c8 (e.g. an RMSNorm) before
// they are scaled to f16; xe[kb * MAXM + tok] receives the biased exponent byte eb (value = f16 * 2^(eb-127)).
// Requirements: k % 32 == 0; called by all THREADS threads (quad shuffles: the 4 vectors of a k32 block
// sit in one lane quad, and k vectors per warp-strided loop keep whole warps in step).
template <class XForm>
__device__ __forceinline__ void convert_act(uint8_t* act, int pitch, uint8_t* xe, int m, int k, int tid, XForm&& xf) {
  const int v8 = k >> 3;
  for (int v = tid; v < MAXM * v8; v += THREADS) {
    const int tok = v / v8, c8 = v - tok * v8;
    uint4* ptr = reinterpret_cast<uint4*>(act + tok * pitch + c8 * 16);
    float f[8];
    if (tok < m) {
      const uint4 raw = *ptr;
      const unsigned u[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 h = unpack_bf16(u[i]);
        f[2 * i] = h.x;
        f[2 * i + 1] = h.y;
      }
      xf(tok, c8, f);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) f[i] = 0.0f;
    }
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(f[i]));
    amax = quad_max(amax);
    // floor(log2 amax) - 14 = ea - 141; scale by 2^(141 - ea) so the max lands in [2^14, 2^15).
    const int ea = max(int((__float_as_uint(amax) >> 23) & 0xffu), 14);
    const float mul = __uint_as_float(unsigned(268 - ea) << 23);
    *ptr = make_uint4(pack_f16(f[0] * mul, f[1] * mul), pack_f16(f[2] * mul, f[3] * mul), pack_f16(f[4] * mul, f[5] * mul),
                      pack_f16(f[6] * mul, f[7] * mul));
    if ((c8 & 3) == 0) xe[(c8 >> 2) * MAXM + tok] = uint8_t(ea - 14);
  }
}

// convert_act with U independent vectors per thread in flight: all U loads first, then U conversions (the U quad_max
// shuffle pairs and stores interleave instead of one load -> shuffle -> store chain per vector), same values. Requires
// MAXM * k / 8 to be a multiple of 32 (vector validity stays warp-uniform: the shuffles run with whole warps).
template <int U, class XForm>
__device__ __forceinline__ void convert_act_batched(uint8_t* act, int pitch, uint8_t* xe, int m, int k, int tid, XForm&& xf) {
  const int v8 = k >> 3, nv = MAXM * v8;
  for (int base = tid; base < nv; base += U * THREADS) {
    uint4 raw[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int v = base + u * THREADS, tok = v / v8, c8 = v - tok * v8;
      raw[u] = make_uint4(0u, 0u, 0u, 0u);
      if (v < nv && tok < m) raw[u] = *reinterpret_cast<const uint4*>(act + tok * pitch + c8 * 16);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int v = base + u * THREADS;
      if (v < nv) {  // warp-uniform (nv % 32 == 0)
        const int tok = v / v8, c8 = v - tok * v8;
        float f[8];
        if (tok < m) {
          const unsigned w[4] = {raw[u].x, raw[u].y, raw[u].z, raw[u].w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float2 h = unpack_bf16(w[i]);
            f[2 * i] = h.x;
            f[2 * i + 1] = h.y;
          }
          xf(tok, c8, f);
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) f[i] = 0.0f;
        }
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(f[i]));
        amax = quad_max(amax);
        const int ea = max(int((__float_as_uint(amax) >> 23) & 0xffu), 14);
        const float mul = __uint_as_float(unsigned(268 - ea) << 23);
        *reinterpret_cast<uint4*>(act + tok * pitch + c8 * 16) =
            make_uint4(pack_f16(f[0] * mul, f[1] * mul), pack_f16(f[2] * mul, f[3] * mul), pack_f16(f[4] * mul, f[5] * mul),
                       pack_f16(f[6] * mul, f[7] * mul));
        if ((c8 & 3) == 0) xe[(c8 >> 2) * MAXM + tok] = uint8_t(ea - 14);
      }
    }
  }
}

struct Identity {
  __device__ __forceinline__ void operator()(int, int, float (&)[8]) const {}
};

__device__ __forceinline__ void unpack8_bf16(uint4 v, float (&f)[8]) {
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 h = unpack_bf16(u[i]);
    f[2 * i] = h.x;
    f[2 * i + 1] = h.y;
  }
}

// ================================================================ RS-wgmma m64n8k32 e4m3 path
enum : int { MMA_HMMA = 0, MMA_WGMMA = 1 };

// B tile: per k32 block 2 core matrices (logical k 0..15 | 16..31) of 8 token rows x 16 B, padded to 272 B
// so that the 8 quads of a staging warp (8 consecutive blocks of one token) hit distinct banks.
constexpr int WG_KB_PITCH = 272;
constexpr int WG_G = 2;  // accumulator sets (k32 blocks) per commit group; two groups in flight
// (4 sets per group would need 80 A + 32 accumulator registers live at once: ptxas then serializes the
// wgmmas for lack of registers, C7512.)

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
// Generic-proxy st.shared -> async-proxy (wgmma operand) reads: every writer fences, then a barrier.
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void reg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void reg_fence(unsigned& r) { asm volatile("" : "+r"(r)::"memory"); }

// No-swizzle K-major shared-memory matrix descriptor (cute GmmaDescriptor, LayoutType::INTERLEAVE:
// ((8,n),(16B,2)):((16B,SBO),(1,LBO))): start address >> 4 in bits 0..13, LBO = 128 B (the k 16..31 core
// matrix) in bits 16..29, SBO = 256 B (next 8 rows; unused for N = 8) in bits 32..45, swizzle bits 62..63 = 0.
// Kept as two 32-bit halves: the per-block advance (+WG_KB_PITCH/16 in the address field, never carrying
// out of it for smem addresses < 256 KB) is a single 32-bit add on the low word.
__device__ __forceinline__ unsigned wg_desc_lo(const void* smem) { return ((smem_u32(smem) & 0x3FFFFu) >> 4) | ((128u >> 4) << 16); }
constexpr unsigned WG_DESC_HI = 256u >> 4;

// D[64 x 8] = A[64 x 32] (e4m3, this warp's packed 16-byte lane word = rows 16w..16w+15 of A) x B (desc).
__device__ __forceinline__ void wgmma_m64n8k32(float (&d)[4], const uint4& a, unsigned desc_lo) {
  asm volatile(
      "{\n.reg .pred p;\n.reg .b64 desc;\nsetp.ne.b32 p, %10, 0;\nmov.b64 desc, {%8, %9};\n"
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, desc, p, 1, 1;\n}\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a.x), "r"(a.z), "r"(a.y), "r"(a.w), "r"(desc_lo), "r"(WG_DESC_HI), "n"(0));
}

// Weights for the wgmma path: like load_frag, but blocks >= cnt are zero (every warp of a warpgroup issues
// all KW wgmmas, so absent blocks / tiles must contribute exactly 0).
template <int KW>
__device__ __forceinline__ void load_frag_wg(Frag& f, const uint8_t* __restrict__ w, const uint8_t* __restrict__ s, long tile, int kb_count,
                                             int kb_pitch, int kb0, int cnt, int lane) {
  const uint8_t* base = w + ((tile * kb_count + kb0) * 32 + lane) * 16;
#pragma unroll
  for (int i = 0; i < KW; ++i) f.w[i] = i < cnt ? ld_stream16(base + i * CHUNK) : make_uint4(0u, 0u, 0u, 0u);
  f.s = lane * 4 < cnt ? ld_stream4(s + tile * kb_pitch + kb0 + lane * 4) : 0u;
}

// Quantize token rows 0..m-1 of NKB k32 blocks into the B tile (reference act_quant: per (token, k32) ue8m0
// scale, see header) and sx[kb * MAXM + tok] = 2^e; token rows m..7 are zeroed (data and scale).
// src(tok, c8, f, ok) yields the 8 (bf16-valued) floats of vector c8 (physical k 8*c8 .. 8*c8+7) of token
// tok; ok == false marks the padding lanes of the last warp-wide pass (tok clamped to m-1; produce finite
// values, write nothing). Called by threads tid = 0..nthr-1 (nthr % 32 == 0); whole warps iterate together
// (full-mask quad shuffles, no convergence checks); the 4 vectors of a k32 block sit in one lane quad.
// (NKB is a compile-time constant so the index math has no integer divisions.)
template <int NKB, class Src>
__device__ __forceinline__ void stage_e4m3(uint8_t* bt, float* sx, int m, int tid, int nthr, Src&& src) {
  constexpr int V8 = NKB * 4;
  const int n = m * V8, nw = (n + 31) & ~31;
  for (int v = tid; v < nw; v += nthr) {
    const bool ok = v < n;
    const int t0 = v / V8, c8 = v - t0 * V8, tok = min(t0, m - 1);
    float f[8];
    src(tok, c8, f, ok);
    float am = fabsf(f[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) am = fmaxf(am, fabsf(f[i]));
    am = fmaxf(am, __shfl_xor_sync(FULL, am, 1));
    am = fmaxf(am, __shfl_xor_sync(FULL, am, 2));
    const int e = quant_exp(am, AMAX_FLOOR);
    const float inv = pow2(-e);
    const unsigned lo = e4m3x2(f[0] * inv, f[1] * inv) | (e4m3x2(f[2] * inv, f[3] * inv) << 16);
    const unsigned hi = e4m3x2(f[4] * inv, f[5] * inv) | (e4m3x2(f[6] * inv, f[7] * inv) << 16);
    const int kb = c8 >> 2, t = c8 & 3;
    if (ok) {
      uint8_t* dst = bt + kb * WG_KB_PITCH + tok * 16 + 4 * t;  // physical 8t..8t+3 -> logical 4t..4t+3
      *reinterpret_cast<unsigned*>(dst) = lo;
      *reinterpret_cast<unsigned*>(dst + 128) = hi;  // physical 8t+4..8t+7 -> logical 16+4t..
      if (t == 0) sx[kb * MAXM + tok] = pow2(e);
    }
  }
  for (int i = tid; i < MAXM * NKB * 2; i += nthr) {  // i = (kb, half, tok)
    const int tok = i & (MAXM - 1);
    if (tok >= m) *reinterpret_cast<uint4*>(bt + (i >> 4) * WG_KB_PITCH + ((i >> 3) & 1) * 128 + tok * 16) = make_uint4(0u, 0u, 0u, 0u);
  }
  for (int i = tid; i < MAXM * NKB; i += nthr)
    if ((i & (MAXM - 1)) >= m) sx[i] = 0.0f;
}

// x[i] = bf16(bf16(x[i] * r) ... ) helper for the wgmma staging transforms: y = bf16_rn((x * r) * w) for 8
// values with one F2FP pack per pair (same rounding as __float2bfloat16 per element); returns the packed
// bf16x8 and leaves the rounded values in x.
__device__ __forceinline__ uint4 rmsnorm8_bf16(float (&x)[8], float r, const float4& w0, const float4& w1) {
  const float w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
  unsigned pk[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    pk[i] = pack_bf16(__fmul_rn(__fmul_rn(x[2 * i], r), w[2 * i]), __fmul_rn(__fmul_rn(x[2 * i + 1], r), w[2 * i + 1]));
    x[2 * i] = __uint_as_float(pk[i] << 16);
    x[2 * i + 1] = __uint_as_float(pk[i] & 0xffff0000u);
  }
  return make_uint4(pk[0], pk[1], pk[2], pk[3]);
}

// Per-warp promotion factors fac[i * MAXM + n] = 2^(sw_i - 127) * sx[i * MAXM + n] for this warp's blocks
// i < KW (0 for i >= cnt). sword: lane l holds the scale bytes of blocks 4l..4l+3 (Frag::s). sx points at
// the activation scales of the warp's first block. Whole warp; follow with __syncwarp().
template <int KW>
__device__ __forceinline__ void wg_factors(float* fac, const float* sx, unsigned sword, int cnt, int lane) {
  const int n = lane & 7, q = lane >> 3;
#pragma unroll
  for (int j = 0; j < KW / 4; ++j) {
    const unsigned w = __shfl_sync(FULL, sword, j);
    const int i = 4 * j + q;
    const float sw = __uint_as_float(((w >> (8 * q)) & 0xffu) << 23);
    fac[i * MAXM + n] = i < cnt ? sw * sx[i * MAXM + n] : 0.0f;
  }
}

// acc (HMMA C layout: (row g, tok 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1)) += sum_i D_i * fac_i over the KW
// k32 blocks of this warp's fragment; bt = B tile of the first block. All 4 warps of the warpgroup must
// call this together (same bt). Straight-line code: ptxas serializes wgmma pipelines that cross branches.
template <int KW>
__device__ __forceinline__ void wg_mma(Frag& f, const uint8_t* bt, const float* fac, int lane, float (&acc)[4]) {
  constexpr int NG = KW / WG_G;
  static_assert(KW % WG_G == 0 && NG >= 1, "KW must be a multiple of the group size");
  const unsigned desc0 = wg_desc_lo(bt);
  const float* fp = fac + 2 * (lane & 3);
  float d[2][WG_G][4];
#pragma unroll
  for (int g = 0; g <= NG; ++g) {
    if (g < NG) {
      wgmma_fence();  // orders the previous promotion's reads of d[g & 1] (and the weight loads) before the wgmmas
#pragma unroll
      for (int s = 0; s < WG_G; ++s) wgmma_m64n8k32(d[g & 1][s], f.w[g * WG_G + s], desc0 + unsigned((g * WG_G + s) * (WG_KB_PITCH >> 4)));
      wgmma_commit();
    }
    if (g >= 1) {
      if (g < NG)
        wgmma_wait<1>();
      else
        wgmma_wait<0>();
      const int gp = g - 1;
#pragma unroll
      for (int s = 0; s < WG_G; ++s) {
        float(&dd)[4] = d[gp & 1][s];
        reg_fence(dd[0]), reg_fence(dd[1]), reg_fence(dd[2]), reg_fence(dd[3]);
        uint4& a = f.w[gp * WG_G + s];  // A registers stay live until their wgmma retired
        reg_fence(a.x), reg_fence(a.y), reg_fence(a.z), reg_fence(a.w);
        const float2 fc = *reinterpret_cast<const float2*>(fp + (gp * WG_G + s) * MAXM);
        acc[0] = fmaf(dd[0], fc.x, acc[0]);
        acc[1] = fmaf(dd[1], fc.y, acc[1]);
        acc[2] = fmaf(dd[2], fc.x, acc[2]);
        acc[3] = fmaf(dd[3], fc.y, acc[3]);
      }
    }
  }
}

#ifndef DS41H_SKINNY_DEVICE_ONLY

struct Args {
  const __nv_bfloat16* x;  // [m, k], row stride x_ld elements
  long x_ld;
  const uint8_t* w;  // packed weights
  const uint8_t* s;  // [n/16, kbp]
  __nv_bfloat16* y;  // [m, n] contiguous
  int m, n, k, kw;   // kw: k32 blocks per k-part (multiple of 4)
};

__host__ __device__ constexpr int smem_bytes(int k) { return MAXM * act_pitch(k) + round_up((k / 32) * MAXM, 16) + WARPS * 16 * MAXM * 4; }

// CTA = TPC 16-row tiles x KPC k-parts (TPC * KPC = 8 warps); the k-parts are reduced in shared memory.
template <int KPC>
__global__ void __launch_bounds__(THREADS, 2) skinny_kernel(const Args a) {
  constexpr int TPC = WARPS / KPC;
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kb = a.k >> 5, kbp = round_up(kb, 4), tiles = a.n >> 4, pitch = act_pitch(a.k);
  uint8_t* act = smem;
  uint8_t* xe = smem + MAXM * pitch;
  float* red = reinterpret_cast<float*>(xe + round_up(kb * MAXM, 16));

  const int tile = blockIdx.x * TPC + warp / KPC, part = warp % KPC;
  const int kb0 = part * a.kw, cnt = min(a.kw, kb - kb0);
  const bool active = tile < tiles && cnt > 0;
  Frag f;
  if (active) load_frag(f, a.w, a.s, tile, kb, kbp, kb0, cnt, lane);  // weights are immutable: before the PDL wait
  pdl_launch_dependents();
  pdl_wait();

  const int v8 = a.k >> 3;
  for (int v = tid; v < a.m * v8; v += THREADS) {
    const int tok = v / v8, c8 = v - tok * v8;
    cp_async16(act + tok * pitch + c8 * 16, a.x + tok * a.x_ld + c8 * 8);
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
  convert_act(act, pitch, xe, a.m, a.k, tid, Identity{});
  __syncthreads();

  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (active) mma_frag(f, act, pitch, xe, kb0, cnt, lane, acc);
  store_frag_tile(red + warp * 16 * MAXM, acc, lane);
  __syncthreads();
  for (int idx = tid; idx < MAXM * TPC * 16; idx += THREADS) {
    const int tok = idx / (TPC * 16), r = idx - tok * (TPC * 16), tl = r >> 4, row = r & 15;
    const int gt = blockIdx.x * TPC + tl;
    if (tok >= a.m || gt >= tiles) continue;
    float v = 0.0f;
#pragma unroll
    for (int p = 0; p < KPC; ++p) v += red[(tl * KPC + p) * 16 * MAXM + row * MAXM + tok];
    a.y[long(tok) * a.n + gt * 16 + row] = __float2bfloat16(v);
  }
}

// ---------------------------------------------------------------- wgmma kernel
// CTA = 2 warpgroups. KP == 1: two 4-tile quads x one k range (s == 1). KP == 2: one quad, the two
// warpgroups take consecutive k-parts of KW blocks. s > 1 (KP == 2 only): s CTAs of a thread-block cluster
// split K further; their [64 x 8] partials are summed in rank order through distributed shared memory
// (deterministic, no global scratch).
struct WgArgs {
  const __nv_bfloat16* x;
  long x_ld;
  const uint8_t* w;
  const uint8_t* s;
  __nv_bfloat16* y;
  int m, n, k, s_split;
};

__device__ __forceinline__ unsigned cluster_rank() {
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {  // (non-.aligned: callers may arrive from divergent loops)
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;\n" ::: "memory");
}
__device__ __forceinline__ float ld_dsmem(const float* p, unsigned rank) {
  unsigned ra;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(smem_u32(p)), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");
  return v;
}

template <int KW, int KP>
struct WgSmem {
  static constexpr int BT = 0;                                  // B tile, KP*KW blocks
  static constexpr int SX = BT + KP * KW * WG_KB_PITCH;          // [KP*KW][MAXM] f32
  static constexpr int FAC = SX + KP * KW * MAXM * 4;            // [WARPS][KW][MAXM] f32
  static constexpr int RED = FAC + WARPS * KW * MAXM * 4;        // [WARPS][16][MAXM] f32
  static constexpr int PART = RED + WARPS * 16 * MAXM * 4;       // [64][MAXM] f32 (KP == 2)
  static constexpr int RPITCH = KP * KW * 64;                    // raw bf16 activation row (CTA k range)
  static constexpr int RAW = PART + 64 * MAXM * 4;               // [MAXM][RPITCH] bf16
  static constexpr int BYTES = RAW + MAXM * RPITCH;
  static_assert(SX % 16 == 0 && FAC % 16 == 0 && RED % 16 == 0 && PART % 16 == 0 && RAW % 16 == 0);
};

template <int KW, int KP>
__global__ void __launch_bounds__(THREADS, 2) skinny_wgmma_kernel(const WgArgs a) {
  using L = WgSmem<KW, KP>;
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* bt = smem + L::BT;
  float* sx = reinterpret_cast<float*>(smem + L::SX);
  float* fac = reinterpret_cast<float*>(smem + L::FAC) + (threadIdx.x >> 5) * KW * MAXM;
  float* red = reinterpret_cast<float*>(smem + L::RED);
  float* part = reinterpret_cast<float*>(smem + L::PART);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wq = warp & 3;
  // warpgroup index broadcast from lane 0: the compiler then treats it (and the B-tile descriptor derived
  // from it) as warp-uniform and keeps the per-wgmma descriptor arithmetic in the uniform datapath.
  const int wg = __shfl_sync(FULL, tid >> 7, 0);
  const int kb = a.k >> 5, kbp = round_up(kb, 4), tiles = a.n >> 4;
  const int split = a.s_split;
  const int rank = split > 1 ? int(cluster_rank()) : 0;
  const int grp = blockIdx.x / split;
  const int tile = KP == 2 ? grp * 4 + wq : grp * 8 + wg * 4 + wq;
  const int kbc0 = rank * KP * KW;                // first k32 block of this CTA
  const int kb0 = kbc0 + (KP == 2 ? wg * KW : 0);  // first k32 block of this warpgroup
  const int cnt = tile < tiles ? max(0, min(KW, kb - kb0)) : 0;
  Frag f;
  load_frag_wg<KW>(f, a.w, a.s, tile < tiles ? tile : 0, kb, kbp, kb0, cnt, lane);  // weights: before the PDL wait
  pdl_launch_dependents();
  pdl_wait();

  // activations of the CTA's k range: one cp.async round into smem, then -> e4m3 B tile
  uint8_t* raw = smem + L::RAW;
  constexpr int V8 = KP * KW * 4;
  for (int v = tid; v < a.m * V8; v += THREADS) {
    const int tok = v / V8, c8 = v - tok * V8, kv = kbc0 * 4 + c8;  // kv: global vector index (8 k each)
    if (kv * 8 < a.k) cp_async16(raw + tok * L::RPITCH + c8 * 16, a.x + tok * a.x_ld + kv * 8);
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
  stage_e4m3<KP * KW>(bt, sx, a.m, tid, THREADS, [&](int tok, int c8, float (&v)[8], bool) {
    if ((kbc0 * 4 + c8) * 8 < a.k) {
      unpack8_bf16(*reinterpret_cast<const uint4*>(raw + tok * L::RPITCH + c8 * 16), v);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = 0.0f;
    }
  });
  fence_proxy_async_smem();
  __syncthreads();
  wg_factors<KW>(fac, sx + (KP == 2 ? wg * KW : 0) * MAXM, f.s, cnt, lane);
  __syncwarp();
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  wg_mma<KW>(f, bt + (KP == 2 ? wg * KW : 0) * WG_KB_PITCH, fac, lane, acc);
  store_frag_tile(red + warp * 16 * MAXM, acc, lane);
  __syncthreads();

  if constexpr (KP == 1) {
    for (int idx = tid; idx < MAXM * 8 * 16; idx += THREADS) {
      const int tok = idx >> 7, r = idx & 127, tl = r >> 4, row = r & 15, gt = grp * 8 + tl;
      if (tok >= a.m || gt >= tiles) continue;
      a.y[long(tok) * a.n + gt * 16 + row] = __float2bfloat16(red[tl * 16 * MAXM + row * MAXM + tok]);
    }
  } else {
    // warpgroup 0 (k-part 2r) + warpgroup 1 (k-part 2r+1), fixed order
    for (int idx = tid; idx < 64 * MAXM; idx += THREADS) part[idx] = red[idx] + red[4 * 16 * MAXM + idx];
    if (split == 1) {
      __syncthreads();
      for (int idx = tid; idx < MAXM * 64; idx += THREADS) {
        const int tok = idx >> 6, r = idx & 63, gt = grp * 4 + (r >> 4);
        if (tok >= a.m || gt >= tiles) continue;
        a.y[long(tok) * a.n + gt * 16 + (r & 15)] = __float2bfloat16(part[r * MAXM + tok]);
      }
    } else {
      cluster_sync();  // every CTA's partial is complete and visible cluster-wide
      const int rows = 64 / split;
      for (int idx = tid; idx < MAXM * rows; idx += THREADS) {
        const int tok = idx / rows, r = rank * rows + idx % rows, gt = grp * 4 + (r >> 4);
        if (tok >= a.m || gt >= tiles) continue;
        float v = 0.0f;
        for (int q = 0; q < split; ++q) v += ld_dsmem(part + r * MAXM + tok, unsigned(q));
        a.y[long(tok) * a.n + gt * 16 + (r & 15)] = __float2bfloat16(v);
      }
      cluster_sync();  // no CTA exits while its shared memory may still be read
    }
  }
}

#endif  // DS41H_SKINNY_DEVICE_ONLY

}  // namespace ds41h_skinny

#ifndef DS41H_SKINNY_DEVICE_ONLY

#include <c10/cuda/CUDAGuard.h>

#include "host.cuh"

namespace {

using namespace ds41h_skinny;

int kw_of(int kb, int kpc) { return round_up((kb + kpc - 1) / kpc, 4); }

// Smallest k-split per CTA that fits the register budget and still gives >= ~1 CTA per SM; otherwise
// the largest feasible one (more CTAs).
int choose_kpc(int n, int k) {
  const int kb = k / 32, tiles = n / 16;
  int best = -1;
  for (int kpc : {1, 2, 4, 8}) {
    if (kw_of(kb, kpc) > KW_MAX) continue;
    best = kpc;
    if ((tiles + WARPS / kpc - 1) / (WARPS / kpc) >= 120) break;
  }
  return best;
}

template <int KPC>
void launch(const Args& a) {
  constexpr int TPC = WARPS / KPC;
  const int smem = smem_bytes(a.k);
  opt_in_smem<skinny_kernel<KPC>>(smem_bytes(MAX_K));
  const int grid = (a.n / 16 + TPC - 1) / TPC;
  launch_pdl("ds41h skinny gemm", skinny_kernel<KPC>, dim3(grid), dim3(THREADS), size_t(smem), a);
}

// ---------------------------------------------------------------- wgmma configuration
// k-parts P = KP * s of KW blocks (KW a multiple of 4, <= KW_MAX): the fewest parts that still give
// >= WG_MIN_CTAS CTAs (enough SMs pulling weights at the 345 MHz power-capped clock), else the most.
constexpr int WG_MIN_CTAS = 108;
struct WgCfg {
  int kw = -1, kp = 0, s = 0, ctas = 0;
};
WgCfg choose_wg(int n, int k) {
  const int kb = k / 32, tiles = n / 16;
  WgCfg best;
  for (int P : {1, 2, 4, 8, 16}) {
    const int kw = round_up((kb + P - 1) / P, 4);
    if (kw > KW_MAX) continue;
    WgCfg c;
    c.kw = kw;
    c.kp = P == 1 ? 1 : 2;
    c.s = P / c.kp;
    c.ctas = c.kp == 1 ? (tiles + 7) / 8 : (tiles + 3) / 4 * c.s;
    best = c;
    if (c.ctas >= WG_MIN_CTAS) break;
  }
  return best;
}

template <int KW, int KP>
void launch_wg_t(const WgArgs& a, int ctas) {
  constexpr int smem = WgSmem<KW, KP>::BYTES;
  opt_in_smem<skinny_wgmma_kernel<KW, KP>>(smem);
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  int na = 1;
  if (a.s_split > 1) {
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = unsigned(a.s_split);
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;
    na = 2;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(ctas);
  config.blockDim = dim3(THREADS);
  config.dynamicSmemBytes = smem;
  config.stream = c10::cuda::getCurrentCUDAStream();
  config.attrs = attrs;
  config.numAttrs = na;
  const cudaError_t err = cudaLaunchKernelEx(&config, skinny_wgmma_kernel<KW, KP>, a);
  TORCH_CHECK(err == cudaSuccess, "ds41h skinny wgmma launch failed: ", cudaGetErrorString(err));
}

void launch_wg(const WgArgs& a, const WgCfg& c) {
  TORCH_CHECK(c.kw > 0 && (c.kp == 2 || c.s == 1) && c.s <= 8, "skinny wgmma: no feasible split for k=", a.k);
#define DS41H_WG_CASE(KW_)                                    \
  case KW_:                                                   \
    if (c.kp == 1)                                            \
      launch_wg_t<KW_, 1>(a, c.ctas);                         \
    else                                                      \
      launch_wg_t<KW_, 2>(a, c.ctas);                         \
    break;
  switch (c.kw) {
    DS41H_WG_CASE(4)
    DS41H_WG_CASE(8)
    DS41H_WG_CASE(12)
    DS41H_WG_CASE(16)
    DS41H_WG_CASE(20)
    default: TORCH_CHECK(false, "skinny wgmma: unsupported kw ", c.kw);
  }
#undef DS41H_WG_CASE
}

}  // namespace

// x [m, k] bf16 (m <= 8, unit stride along k, 16-byte aligned rows); w/s from ds41h.skinny.pack.
// mma: MMA_HMMA (0, W8A16 numerics) or MMA_WGMMA (1, W8A8 with reference per-32 activation quant).
torch::Tensor gemm(torch::Tensor x, torch::Tensor w, torch::Tensor s, int64_t n, int64_t k, c10::optional<torch::Tensor> out,
                   int64_t mma) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16, "skinny gemm: x must be a 2-D bf16 CUDA tensor");
  const long m = x.size(0);
  TORCH_CHECK(m >= 1 && m <= MAXM, "skinny gemm: 1 <= m <= ", MAXM, ", got ", m);
  TORCH_CHECK(x.size(1) == k && x.stride(1) == 1 && x.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0,
              "skinny gemm: x rows must be unit-stride, 16-byte aligned, width k");
  TORCH_CHECK(n > 0 && n % 16 == 0 && k > 0 && k % 32 == 0 && k <= MAX_K, "skinny gemm: unsupported shape n=", n, " k=", k);
  const int kb = int(k / 32), kbp = round_up(kb, 4);
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kByte && w.numel() == n * k, "skinny gemm: bad packed weight");
  TORCH_CHECK(s.is_cuda() && s.is_contiguous() && s.scalar_type() == at::kByte && s.numel() == (n / 16) * kbp, "skinny gemm: bad scales");
  TORCH_CHECK(w.device() == x.device() && s.device() == x.device(), "skinny gemm: device mismatch");
  const c10::cuda::CUDAGuard guard(x.device());
  torch::Tensor y;
  if (out.has_value()) {
    y = *out;
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kBFloat16 && y.size(0) == m && y.size(1) == n,
                "skinny gemm: bad out tensor");
  } else {
    y = torch::empty({m, n}, x.options());
  }
  if (mma == MMA_WGMMA) {
    const WgCfg c = choose_wg(int(n), int(k));
    const WgArgs wa{data_as<const __nv_bfloat16>(x), x.stride(0), data_as<const uint8_t>(w), data_as<const uint8_t>(s),
                    data_as<__nv_bfloat16>(y), int(m), int(n), int(k), c.s};
    launch_wg(wa, c);
    return y;
  }
  TORCH_CHECK(mma == MMA_HMMA, "skinny gemm: mma must be 0 (hmma) or 1 (wgmma), got ", mma);
  const int kpc = choose_kpc(int(n), int(k));
  TORCH_CHECK(kpc > 0, "skinny gemm: no feasible split for k=", k);
  const Args a{data_as<const __nv_bfloat16>(x), x.stride(0), data_as<const uint8_t>(w), data_as<const uint8_t>(s),
               data_as<__nv_bfloat16>(y), int(m), int(n), int(k), kw_of(kb, kpc)};
  switch (kpc) {
    case 1: launch<1>(a); break;
    case 2: launch<2>(a); break;
    case 4: launch<4>(a); break;
    default: launch<8>(a); break;
  }
  return y;
}

int64_t choose_split(int64_t n, int64_t k) { return choose_kpc(int(n), int(k)); }
// (kw, kp, s, ctas) of the wgmma path for an [n, k] weight (tests / benchmarks).
std::vector<int64_t> choose_split_wg(int64_t n, int64_t k) {
  const WgCfg c = choose_wg(int(n), int(k));
  return {c.kw, c.kp, c.s, c.ctas};
}

template <int KW>
void init_wg() {
  opt_in_smem<skinny_wgmma_kernel<KW, 1>>(WgSmem<KW, 1>::BYTES);
  opt_in_smem<skinny_wgmma_kernel<KW, 2>>(WgSmem<KW, 2>::BYTES);
}

// Set the dynamic-smem attribute of every instantiation now (load time), so no cudaFuncSetAttribute is
// first issued inside a CUDA graph capture.
void init() {
  opt_in_smem<skinny_kernel<1>>(smem_bytes(MAX_K));
  opt_in_smem<skinny_kernel<2>>(smem_bytes(MAX_K));
  opt_in_smem<skinny_kernel<4>>(smem_bytes(MAX_K));
  opt_in_smem<skinny_kernel<8>>(smem_bytes(MAX_K));
  init_wg<4>();
  init_wg<8>();
  init_wg<12>();
  init_wg<16>();
  init_wg<20>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("MAXM") = MAXM;
  m.attr("KW_MAX") = KW_MAX;
  m.attr("MAX_K") = MAX_K;
  m.attr("MMA_HMMA") = int(MMA_HMMA);
  m.attr("MMA_WGMMA") = int(MMA_WGMMA);
  m.def("gemm", &gemm, "skinny FP8 GEMM (m <= 8)", py::arg("x"), py::arg("w"), py::arg("s"), py::arg("n"), py::arg("k"),
        py::arg("out") = py::none(), py::arg("mma") = int64_t(MMA_HMMA));
  m.def("choose_split", &choose_split);
  m.def("choose_split_wg", &choose_split_wg);
  m.def("init", &init);
}

#endif  // DS41H_SKINNY_DEVICE_ONLY
#endif  // DS41H_SKINNY_CU
