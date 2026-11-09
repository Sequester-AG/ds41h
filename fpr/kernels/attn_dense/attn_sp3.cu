// attn_sp3.cu: sparse MQA decode attention, sequence-parallel layout, v3 (ATTN2B_SPA=2).
// One CTA per row (all 64 heads), the row's window + top-k records, sink, fused inverse RoPE; same inputs, outputs and
// numerics as attn_sp.cu (v2), which it replaces when the flag is 2.
//
// Why v3: v2 spends ~7K cycles per 64-key tile because every warp does the gather
// issue (~2.9K, L2 -> SM bound), the dequant (~2.2K) and the score exchange + softmax (~1.4K) one after the other, and
// Q (64 KB in SMEM for SS Q K^T) leaves no room for a second raw slot. Here a third warpgroup holds Q:
//  * WG2 ("QK"): Q dims [0, 16 NQR) as RS A fragments in registers (NQR = 24: 96 registers), the rest in SMEM (SS);
//    computes the full S = Q K^T of each tile (no score exchange), the online softmax (exp2 via ex2.approx, as v2),
//    and writes P (bf16, SW128 A layout) + the per-row rescale factor to SMEM.
//  * WG0 / WG1 ("PV"): own O dims [256 w, 256 w + 256) (128 accumulator registers), rescale O, SS P V from SMEM P, and
//    dequantize tile t + 1 into the other bf16 buffer while P V t runs.
//  * All 12 warps gather (warp per record, 2 tiles ahead) into a 2-slot raw ring; the bf16 RoPE dims go by cp.async
//    straight into the bf16 tile once it is free.
//  * Producer / consumer named barriers per tile parity: E1 (bf16 tile ready: WG0/1 -> WG2), E2 (P ready: WG2 -> WG0/1),
//    E3 (Q K^T done with the tile: WG2 -> WG0/1), E4 (WG2's share of a raw tile landed: WG2 -> WG0/1).
// SMEM ~223 KB; 384 threads at the 168-register cap (setmaxnreg not used).
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

#include "wgmma_bf16_gen.cuh"

#ifndef SPA_PDL
#define SPA_PDL 1
#endif
#ifndef SPA_PROF
#define SPA_PROF 0  // 1: clock64 phase sums per (row, warpgroup) into p.prof [R][3][16]
#endif
#ifndef SPA_REORDER
#define SPA_REORDER 0  // 1: WG0 / WG1 dequantize tile t + 1 BEFORE waiting for P t (its buffer is free since iteration t - 1)
#endif
#ifndef SPA_NQR
#define SPA_NQR 24  // Q K^T k-steps with Q in WG2's registers (the rest SS from SMEM)
#endif

namespace spa {
constexpr int BK = 64, NT = 384, REC = 576, MAXK = 1024;
constexpr int RW = 448;                 // raw slot: e4m3 bytes per record
constexpr int RAW_B = BK * RW + BK * 8;  // + 8-B scale rows = 29,184 B
constexpr int NQS = 32 - SPA_NQR;       // SS k-steps
constexpr int OFF_BF = 0;               // bf16 tiles [buffer 2][half 2] x 32 KB
constexpr int OFF_QS = 131072;          // Q dims [16 NQR, 512) for SS (column blocks of 64 dims, 8 KB each)
constexpr int QS_B = ((NQS * 16 + 63) / 64) * 8192;
constexpr int OFF_RAW = OFF_QS + QS_B;  // 2 raw slots
constexpr int OFF_P = OFF_RAW + 2 * RAW_B;  // 2 x 8 KB P tiles
constexpr int OFF_AL = OFF_P + 16384;   // 2 x 64 rescale factors, + 64 final factors
constexpr int OFF_IDX = OFF_AL + 768;
constexpr int SMEM = OFF_IDX + MAXK * 4 + 1024;
static_assert(SMEM <= 232448, "smem");
static_assert((16 * SPA_NQR) % 64 == 0, "the SS part starts at a column block");
constexpr int MN_LBO = 8192, MN_SBO = 1024;
constexpr float LOG2E = 1.4426950408889634f;
// named barrier ids: E1 1/2, E2 3/4, E3 5/6, E4 7/8 (by tile parity), 9 epilogue
constexpr int E1 = 1, E2 = 3, E3 = 5, E4 = 7;

struct Params {
  const __nv_bfloat16* q;
  __nv_bfloat16* out;
  const uint8_t* swa;
  const uint8_t* comp;
  long long stride;
  const int* si;
  const int* ti;
  int nsi, nti;
  const int* sl;
  const int* tl;
  const float* sink;
  float scale_log2;
  const long long* pos;
  const float* cos_sin;
  unsigned long long* prof;
  int ins;
};

#include "spa_common.cuh"

__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// e4m3 bytes + scale rows of the records k = k0, k0 + kstep, ... of tile t -> raw slot (warp per record: lanes 0..27 the
// 28 chunks, lane 28 the scale row). One cp.async commit group per call.
__device__ __forceinline__ void gather_fp8(const Params& p, const int* sidx, int sl, int n, int t, int k0, int kstep, int lane,
                                           uint32_t raw) {
#pragma unroll 2
  for (int k = k0; k < BK; k += kstep) {
    const int j = min(t * BK + k, n - 1);
    const uint8_t *d, *s;
    rec_addr(p, sidx[j], j < sl, d, s);
    if (lane < 28)
      cp16(raw + k * RW + lane * 16, d + lane * 16);
    else if (lane == 28)
      cp8(raw + BK * RW + k * 8, s);
  }
  cp_commit();
}
// bf16 RoPE dims (8 chunks) of tile t's records -> half tile 1's column block 3 (swizzled); 256 threads (WG0 / WG1),
// two chunks each. One commit group.
__device__ __forceinline__ void gather_rope(const Params& p, const int* sidx, int sl, int n, int t, int tid256, uint32_t bf1) {
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int i = tid256 + 256 * u, k = i >> 3, c = i & 7;
    const int j = min(t * BK + k, n - 1);
    const uint8_t *d, *s;
    rec_addr(p, sidx[j], j < sl, d, s);
    cp16(bf1 + 3 * 8192 + k * 128 + ((c ^ (k & 7)) << 4), d + RW + c * 16);
  }
  cp_commit();
}

// raw slot -> half tile w (64 keys x 256 dims): w 0 = e4m3 dims 0..255 (16 chunks per key), w 1 = dims 256..447 (12)
__device__ __forceinline__ void dequant_tile(uint32_t raw, uint32_t bf, int w, int ltid) {
  if (w == 0) {
    const int c = ltid & 15;
#pragma unroll 4
    for (int jj = 0; jj < 8; ++jj) {
      const int k = (ltid >> 4) + 8 * jj;
      const uint4 v = lds128(raw + k * RW + c * 16);
      const unsigned E = lds_u8(raw + BK * RW + k * 8 + (c >> 2));
      uint4 lo, hi;
      dequant16(v, E, lo, hi);
      const uint32_t a = bf + (c >> 2) * 8192 + k * 128;
      const int q = (c & 3) * 2;
      sts128(a + ((q ^ (k & 7)) << 4), lo);
      sts128(a + (((q + 1) ^ (k & 7)) << 4), hi);
    }
  } else {
#pragma unroll 3
    for (int jj = 0; jj < 6; ++jj) {
      const int i = ltid + 128 * jj, k = i / 12, c = i - 12 * k;
      const uint4 v = lds128(raw + k * RW + 256 + c * 16);
      const unsigned E = lds_u8(raw + BK * RW + k * 8 + 4 + (c >> 2));
      uint4 lo, hi;
      dequant16(v, E, lo, hi);
      const uint32_t a = bf + (c >> 2) * 8192 + k * 128;
      const int q = (c & 3) * 2;
      sts128(a + ((q ^ (k & 7)) << 4), lo);
      sts128(a + (((q + 1) ^ (k & 7)) << 4), hi);
    }
  }
}

__global__ void __launch_bounds__(NT, 1) attn_sp3_kernel(const __grid_constant__ Params p) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t s0 = smem_u32(smem_raw), sb = (s0 + 1023u) & ~1023u;
  int* sidx = reinterpret_cast<int*>(smem_raw + (sb - s0) + OFF_IDX);
  const int r = blockIdx.x, tid = threadIdx.x;
  const int warp12 = __shfl_sync(0xffffffffu, tid >> 5, 0), wg = warp12 >> 2, warp = warp12 & 3, lane = tid & 31;
  const int ltid = tid & 127, g = lane >> 2, t4 = lane & 3;
  long long ck0 = 0;
  unsigned long long gt0 = 0;
  if (SPA_PROF) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt0));
    ck0 = clock64();
  }
  const int sl = __ldg(p.sl + r), n = min(sl + __ldg(p.tl + r), MAXK), ntiles = (n + BK - 1) / BK;
  {
    int v[3];
#pragma unroll
    for (int q = 0; q < 3; ++q) {
      const int j = tid + NT * q;
      v[q] = j < n ? (j < sl ? __ldg(p.si + static_cast<long long>(r) * p.nsi + j) : __ldg(p.ti + static_cast<long long>(r) * p.nti + (j - sl)))
                   : -1;
    }
#pragma unroll
    for (int q = 0; q < 3; ++q)
      if (tid + NT * q < n) sidx[tid + NT * q] = v[q];
  }
  __syncthreads();
  auto bfh = [&](int b, int w) -> uint32_t { return sb + OFF_BF + 65536 * b + 32768 * w; };
  auto raws = [&](int b) -> uint32_t { return sb + OFF_RAW + RAW_B * b; };
  const uint32_t QS = sb + OFF_QS;
  // ---- prologue: tiles 0 and 1 (e4m3 + scales by all 12 warps; RoPE by WG0 / WG1), Q
  const bool early = ntiles > 0 && max(0, sl - p.ins) >= min(2 * BK, sl);  // tiles 0, 1 hold no inserted record
  auto pro_gather = [&]() {
    for (int t = 0; t < 2 && t < ntiles; ++t) {
      gather_fp8(p, sidx, sl, n, t, warp12, 12, lane, raws(t));
      if (wg < 2) gather_rope(p, sidx, sl, n, t, tid, bfh(t, 1));
    }
  };
  if (early) pro_gather();
  if (SPA_PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (!early) pro_gather();
  unsigned qf[SPA_NQR][4];
  if (wg == 2) {
    const int m0 = 16 * warp + g;
    const __nv_bfloat16* qa = p.q + (static_cast<long long>(r) * 64 + m0) * 512 + 2 * t4;
    const __nv_bfloat16* qb = qa + 8 * 512;
#pragma unroll
    for (int s = 0; s < SPA_NQR; ++s) {
      asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(qf[s][0]) : "l"(qa + 16 * s));
      asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(qf[s][1]) : "l"(qb + 16 * s));
      asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(qf[s][2]) : "l"(qa + 16 * s + 8));
      asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(qf[s][3]) : "l"(qb + 16 * s + 8));
    }
    const __nv_bfloat16* qs = p.q + static_cast<long long>(r) * 64 * 512 + 16 * SPA_NQR;
    constexpr int CH = NQS * 2;  // 16-B chunks per head row of the SS part
#pragma unroll 4
    for (int i = ltid; i < 64 * CH; i += 128) {
      const int m = i / CH, ch = i - CH * m, cb = ch >> 3, c8 = ch & 7;
      cp16(QS + cb * 8192 + m * 128 + ((c8 ^ (m & 7)) << 4), qs + m * 512 + ch * 8);
    }
    cp_commit();
  }
  cp_wait<0>();
  __syncthreads();
  if (wg < 2 && ntiles > 0) dequant_tile(raws(0), bfh(0, wg), wg, ltid);
  fence_proxy_async();
  __syncthreads();
  // WG0 / WG1 gather a tile 3 ahead (WG2 2 ahead): their share of tile 2 -> raw slot 0 (tile 0 dequantized)
  if (wg < 2 && ntiles > 2) gather_fp8(p, sidx, sl, n, 2, warp12, 12, lane, raws(0));
  unsigned long long pt[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  (void)pt;
  if (SPA_PROF) pt[8] = clock64() - ck0;

  if (wg == 2) {  // ================================================================ QK warpgroup
    float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
    const int m0 = 16 * warp + g;
    for (int t = 0; t < ntiles; ++t) {
      const int b = t & 1;
      long long tq = SPA_PROF ? clock64() : 0;
      if (t > 0) named_bar(E1 + b, NT);  // tile t complete in buffer b (tile 0: the prologue)
      if (SPA_PROF) { const long long c = clock64(); pt[0] += c - tq; tq = c; }
      float s_[32];
      wg_fence();
#pragma unroll
      for (int s = 0; s < 32; ++s) {
        const uint32_t kb = bfh(b, s >> 4) + ((s & 15) >> 2) * 8192 + (s & 3) * 32;
        if (s < SPA_NQR)
          WgRS<64, 0>::mma(s_, qf[s < SPA_NQR ? s : 0], desc_k(kb), s > 0);
        else
          WgSS<64, 0>::mma(s_, desc_k(QS + ((s - SPA_NQR) >> 2) * 8192 + ((s - SPA_NQR) & 3) * 32), desc_k(kb), 1);
      }
      wg_commit();
      // my share of tile t + 2's e4m3 records -> raw slot b (tile t was dequantized before E1 t)
      if (t + 2 < ntiles) gather_fp8(p, sidx, sl, n, t + 2, warp12, 12, lane, raws(b));
      unsigned long long mask;
      {
        const int j0 = t * BK + lane, j1 = j0 + 32;
        const bool v0 = j0 < n && sidx[j0 < n ? j0 : 0] >= 0, v1 = j1 < n && sidx[j1 < n ? j1 : 0] >= 0;
        mask = (static_cast<unsigned long long>(__ballot_sync(0xffffffffu, v1)) << 32) | __ballot_sync(0xffffffffu, v0);
      }
      if (SPA_PROF) { const long long c = clock64(); pt[1] += c - tq; tq = c; }
      wg_wait<0>();
      bar_arrive(E3 + b, NT);  // done reading buffer b
      if (SPA_PROF) { const long long c = clock64(); pt[2] += c - tq; tq = c; }
      // online softmax of both rows, P -> SMEM (SW128 A layout), rescale factors
      const uint32_t PB = sb + OFF_P + 8192 * b, AL = sb + OFF_AL + 256 * b;
#pragma unroll
      for (int rho = 0; rho < 2; ++rho) {
        if (mask == ~0ull) {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            s_[4 * j + 2 * rho] *= p.scale_log2;
            s_[4 * j + 2 * rho + 1] *= p.scale_log2;
          }
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int c = 8 * j + 2 * t4 + e;
              float& x = s_[4 * j + 2 * rho + e];
              x = ((mask >> c) & 1ull) ? x * p.scale_log2 : -INFINITY;
            }
        }
        float tt[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
          tt[i] = fmaxf(fmaxf(s_[8 * i + 2 * rho], s_[8 * i + 2 * rho + 1]), fmaxf(s_[8 * i + 4 + 2 * rho], s_[8 * i + 5 + 2 * rho]));
        float tmax = fmaxf(fmaxf(tt[0], tt[1]), fmaxf(tt[2], tt[3]));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
        const float mnew = fmaxf(m_run[rho], tmax);
        const float alpha = mnew == -INFINITY ? 1.f : ex2(m_run[rho] - mnew);
        const float ms = mnew == -INFINITY ? 0.f : mnew;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          s_[4 * j + 2 * rho] = ex2(s_[4 * j + 2 * rho] - ms);
          s_[4 * j + 2 * rho + 1] = ex2(s_[4 * j + 2 * rho + 1] - ms);
        }
        float u[8];  // K3 / v2 order: pairs, then a tree over j
#pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = s_[4 * j + 2 * rho] + s_[4 * j + 2 * rho + 1];
#pragma unroll
        for (int i = 0; i < 4; ++i) u[i] = u[2 * i] + u[2 * i + 1];
        l_run[rho] = l_run[rho] * alpha + ((u[0] + u[1]) + (u[2] + u[3]));
        m_run[rho] = mnew;
        const int mr = m0 + 8 * rho;
#pragma unroll
        for (int j = 0; j < 8; ++j)
          asm volatile("st.shared.u32 [%0], %1;" ::"r"(PB + mr * 128 + ((j ^ (mr & 7)) << 4) + 4 * t4),
                       "r"(pack_bf16(s_[4 * j + 2 * rho], s_[4 * j + 2 * rho + 1])) : "memory");
        if (t4 == 0) asm volatile("st.shared.f32 [%0], %1;" ::"r"(AL + 4 * mr), "f"(alpha) : "memory");
      }
      fence_proxy_async();
      bar_arrive(E2 + b, NT);  // P t ready
      if (SPA_PROF) { const long long c = clock64(); pt[3] += c - tq; tq = c; }
      if (t + 2 < ntiles) {  // my share of tile t + 2 must have landed before WG0 / WG1 dequantize it
        cp_wait<0>();
        bar_arrive(E4 + b, NT);
      }
      if (SPA_PROF) { const long long c = clock64(); pt[4] += c - tq; }
    }
    // final factors o / (l + e^(sink - m))
#pragma unroll
    for (int rho = 0; rho < 2; ++rho) {
      float l = l_run[rho];
      l += __shfl_xor_sync(0xffffffffu, l, 1);
      l += __shfl_xor_sync(0xffffffffu, l, 2);
      const float m = m_run[rho];
      const float sink2 = __ldg(p.sink + m0 + 8 * rho) * LOG2E;
      const float mf = fmaxf(m, sink2);
      const float f = m == -INFINITY ? 0.f : ex2(m - mf);
      const float den = l * f + (sink2 == -INFINITY ? 0.f : ex2(sink2 - mf));
      const float so = (den > 0.f && mf < INFINITY) ? f / den : 0.f;
      if (t4 == 0) asm volatile("st.shared.f32 [%0], %1;" ::"r"(sb + OFF_AL + 512 + 4 * (m0 + 8 * rho)), "f"(so) : "memory");
    }
  } else {  // ===================================================================== PV warpgroups 0 / 1
    float o[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) o[i] = 0.f;
    const int m0 = 16 * warp + g;
    // tile t + 1 -> buffer nb: e4m3 shares landed (mine, WG2's via E4, the other P V warpgroup's via barrier 9), then
    // dequant + E1 arrive. Its buffer held tile t - 1 (P V t - 1 waited, Q K^T t - 1 via E3 at the end of t - 1).
    auto dequant_next = [&](int t, int nb) {
      cp_wait<0>();
      if (t + 1 >= 2) named_bar(E4 + nb, NT);
      named_bar(9, 256);
      dequant_tile(raws(nb), bfh(nb, wg), wg, ltid);
      fence_proxy_async();
      bar_arrive(E1 + nb, NT);
    };
    for (int t = 0; t < ntiles; ++t) {
      const int b = t & 1, nb = b ^ 1;
      long long tq = SPA_PROF ? clock64() : 0;
      if (SPA_REORDER && t + 1 < ntiles) {
        dequant_next(t, nb);
        if (SPA_PROF) { const long long c = clock64(); pt[3] += c - tq; tq = c; }
      }
      named_bar(E2 + b, NT);  // P t
      if (SPA_PROF) { const long long c = clock64(); pt[0] += c - tq; tq = c; }
      const uint32_t AL = sb + OFF_AL + 256 * b;
      float a0, a1;
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(a0) : "r"(AL + 4 * m0) : "memory");
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(a1) : "r"(AL + 4 * (m0 + 8)) : "memory");
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        o[4 * j] *= a0;
        o[4 * j + 1] *= a0;
        o[4 * j + 2] *= a1;
        o[4 * j + 3] *= a1;
      }
      wg_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks)
        WgSS<256, 1>::mma(o, desc_k(sb + OFF_P + 8192 * b + ks * 32), desc_mn(bfh(b, wg) + ks * 2048), 1);
      wg_commit();
      if (SPA_PROF) { const long long c = clock64(); pt[1] += c - tq; tq = c; }
      if (t + 1 < ntiles) {  // dequant tile t + 1 into buffer nb while P V t runs (SPA_REORDER: done above)
        if (!SPA_REORDER) {
          dequant_next(t, nb);
          if (SPA_PROF) { const long long c = clock64(); pt[3] += c - tq; tq = c; }
        }
        // my share of tile t + 3 -> raw slot nb (tile t + 1 was dequantized from it)
        if (t + 3 < ntiles) gather_fp8(p, sidx, sl, n, t + 3, warp12, 12, lane, raws(nb));
      }
      if (SPA_PROF) { const long long c = clock64(); pt[4] += c - tq; tq = c; }
      wg_wait<0>();
      named_bar(E3 + b, NT);  // WG2 done with buffer b (and so are both P V warpgroups after this barrier)
      if (SPA_PROF) { const long long c = clock64(); pt[5] += c - tq; tq = c; }
      if (t + 2 < ntiles) gather_rope(p, sidx, sl, n, t + 2, tid, bfh(b, 1));  // tile t + 2's RoPE dims -> buffer b
      if (SPA_PROF) { const long long c = clock64(); pt[6] += c - tq; }
    }
    // ---- epilogue: final factors from WG2, bf16, inverse RoPE (WG1's dims 448..511), staged 16-B stores
    float csv[16];
    const bool csv_ok = p.cos_sin != nullptr;
    if (wg == 1 && csv_ok) {
      const float* cs = p.cos_sin + __ldg(p.pos + r) * 64;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        csv[jj] = __ldg(cs + 4 * jj + t4);
        csv[8 + jj] = __ldg(cs + 32 + 4 * jj + t4);
      }
    }
    // wait for WG2's factors (barrier 10, all 384)
    named_bar(10, NT);
    float so0, so1;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(so0) : "r"(sb + OFF_AL + 512 + 4 * m0) : "memory");
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(so1) : "r"(sb + OFF_AL + 512 + 4 * (m0 + 8)) : "memory");
    const uint32_t ST = sb + OFF_BF;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int d = 256 * wg + 8 * j + 2 * t4;
      float v0 = bf16r(o[4 * j] * so0), v1 = bf16r(o[4 * j + 1] * so0);
      float v2 = bf16r(o[4 * j + 2] * so1), v3 = bf16r(o[4 * j + 3] * so1);
      if (wg == 1 && j >= 24 && csv_ok) {
        const float c = csv[j - 24], sn = csv[8 + j - 24];
        const float a0_ = __fadd_rn(__fmul_rn(v0, c), __fmul_rn(v1, sn)), b0 = __fsub_rn(__fmul_rn(v1, c), __fmul_rn(v0, sn));
        const float a1_ = __fadd_rn(__fmul_rn(v2, c), __fmul_rn(v3, sn)), b1 = __fsub_rn(__fmul_rn(v3, c), __fmul_rn(v2, sn));
        v0 = a0_, v1 = b0, v2 = a1_, v3 = b1;
      }
      const int ch = d >> 3, off = (d & 7) * 2;
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(ST + m0 * 1024 + ((ch ^ (m0 & 7)) << 4) + off), "r"(pack_bf16(v0, v1)) : "memory");
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(ST + (m0 + 8) * 1024 + ((ch ^ ((m0 + 8) & 7)) << 4) + off), "r"(pack_bf16(v2, v3))
                   : "memory");
    }
  }
  if (wg == 2) named_bar(10, NT);  // the final factors are in SMEM
  cp_wait<0>();
  if (SPA_PDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __syncthreads();  // staged o complete
  {
    __nv_bfloat16* orow = p.out + static_cast<long long>(r) * 64 * 512;
    const uint32_t ST = sb + OFF_BF;
    for (int i = tid; i < 64 * 64; i += NT) {
      const int m = i >> 6, ch = i & 63;
      const uint4 v = lds128(ST + m * 1024 + ((ch ^ (m & 7)) << 4));
      *reinterpret_cast<uint4*>(orow + m * 512 + ch * 8) = v;
    }
  }
  if (SPA_PROF && p.prof != nullptr && ltid == 0) {
    pt[10] = gt0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(pt[11]));
#pragma unroll
    for (int i = 0; i < 16; ++i) p.prof[(static_cast<long long>(r) * 3 + wg) * 16 + i] = pt[i];
  }
}
}  // namespace spa

void attn_sp3(torch::Tensor q, torch::Tensor out, torch::Tensor swa, torch::Tensor comp, int64_t stride, torch::Tensor si,
              torch::Tensor ti, torch::Tensor sl, torch::Tensor tl, torch::Tensor sink, double scale,
              c10::optional<torch::Tensor> pos, c10::optional<torch::Tensor> cos_sin, c10::optional<torch::Tensor> prof,
              int64_t ins) {
  const int R = static_cast<int>(sl.numel());
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous() && q.numel() == static_cast<long long>(R) * 64 * 512, "q / out [R, 64, 512]");
  TORCH_CHECK(si.dtype() == torch::kInt32 && ti.dtype() == torch::kInt32 && sl.dtype() == torch::kInt32 &&
                  tl.dtype() == torch::kInt32 && si.is_contiguous() && ti.is_contiguous(),
              "int32 contiguous indices / lengths");
  spa::Params p{};
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.swa = swa.data_ptr<uint8_t>();
  p.comp = comp.data_ptr<uint8_t>();
  p.stride = stride;
  p.si = si.data_ptr<int>();
  p.ti = ti.data_ptr<int>();
  p.nsi = static_cast<int>(si.numel() / R);
  p.nti = static_cast<int>(ti.numel() / R);
  p.sl = sl.data_ptr<int>();
  p.tl = tl.data_ptr<int>();
  p.sink = sink.data_ptr<float>();
  p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  p.pos = pos.has_value() ? reinterpret_cast<const long long*>(pos->data_ptr<int64_t>()) : nullptr;
  p.cos_sin = cos_sin.has_value() ? cos_sin->data_ptr<float>() : nullptr;
  TORCH_CHECK(p.cos_sin == nullptr || p.pos != nullptr, "inverse RoPE needs positions");
  p.prof = prof.has_value() ? reinterpret_cast<unsigned long long*>(prof->data_ptr<int64_t>()) : nullptr;
  p.ins = static_cast<int>(ins);
  auto kern = spa::attn_sp3_kernel;
  static bool init_dev[64] = {};
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  if (!init_dev[cur_dev & 63]) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, spa::SMEM) == cudaSuccess, "smem attr");
    init_dev[cur_dev & 63] = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(R);
  cfg.blockDim = dim3(spa::NT);
  cfg.dynamicSmemBytes = spa::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = SPA_PDL ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, p) == cudaSuccess, "attn_sp3 launch");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_sp3", &attn_sp3, "SP sparse MQA decode attention v3 (QK warpgroup)"); }
