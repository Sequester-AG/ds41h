// attn_seq3.cu: K3 v3 -- warp-specialized sparse MQA decode attention, one CTA per 64 queries.
//
// Same contract and numerics as attn_seq.cu v2 (per-key codes with row masks + absolute record addresses from the
// per-step metadata, 64 query rows per CTA = 4 rows x 16 heads, or 1 row x 64 heads in the sequence-parallel layout),
// restructured after the v2 measurements:
//  * the key gather is TMA bulk copies (one 576-byte data row + one 16-byte scale window per key) into a raw SMEM ring,
//    issued by all 32 lanes of one warp: v2's per-thread LDG gather ran at ~0.9 TB/s with 4 warps per SM (random
//    record LDG bandwidth scales with warps per SM: 1.7 TB/s at 8, 3.5-4.1 at 32-64, gather microbenchmark) and flooded the
//    LSU queues so the softmax of the other warpgroup slowed down 3x;
//  * Q lives in registers of the S warpgroup (RS-wgmma m64n64k16 for Q K^T; SS m64n64 ran at ~1/3 of peak);
//  * three warpgroups: WG0 = S + online softmax (P -> SMEM, alpha), WG1 / WG2 = O for dims 0..255 / 256..511
//    (SS m64n256k16, B = the key tile read MN-major) + raw -> bf16 dequant of later tiles; setmaxnreg 200 / 152 / 152 (launch 168: the two decrements free exactly the 2 x 128 x 16 registers WG0 takes).
// Pipeline (tile i): B1(i) -> WG0: S_{i+1}, softmax_{i+1} | WG1/2: P_i V_i, dequant tile i+2 (raw ring, 2 tiles ahead
// of its TMA) -> B1(i+1).
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "k3_common.cuh"
#include "wgmma_bf16_gen.cuh"

namespace k3v3 {
using namespace k3c;

constexpr int NT3 = 384;
constexpr int RAWB = BK * 576 + BK * 16;  // 37,888: data rows then 16-byte scale windows
constexpr int OFF_K0 = 0, OFF_K1 = 65536, OFF_R0 = 131072, OFF_R1 = OFF_R0 + RAWB, OFF_P0 = OFF_R1 + RAWB,
              OFF_P1 = OFF_P0 + 8192, OFF_ALPHA = OFF_P1 + 8192, OFF_SO = OFF_ALPHA + 512, OFF_HALF = OFF_SO + 256,
              OFF_BAR = OFF_HALF + 128;
constexpr int SMEM3 = OFF_BAR + 16 + 1024;
static_assert(OFF_P0 % 1024 == 0 && OFF_P1 % 1024 == 0, "P tiles must be 1 KB aligned (SW128)");

struct Params {
  const __nv_bfloat16* q;
  __nv_bfloat16* out;
  const int* codes;
  int code_stride;
  const ulonglong2* addr;
  int addr_stride;
  const int* nkeys;
  const float* sink;
  int sink_rows;
  float scale_log2;
  const long long* positions;
  const float* cos_sin;
  int mn_lbo, mn_sbo;
  float* dbg_s;
  unsigned long long* prof;
};
#ifdef K3_PROF
#define KP(it_, slot)                                                                                     \
  do {                                                                                                    \
    if ((tid & 127) == 0 && p.prof != nullptr && (it_) >= 0 && (it_) < 40)                                \
      p.prof[((static_cast<long long>(s) * 40 + (it_)) * 3 + wg) * 8 + (slot)] = (unsigned long long)clock64(); \
  } while (0)
#else
#define KP(it_, slot) \
  do {                \
  } while (0)
#endif

__device__ __forceinline__ void mbar_init(uint32_t a, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, unsigned parity) {
  asm volatile("{\n.reg .pred p;\nW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W_%=;\n}\n" ::"r"(a),
               "r"(parity)
               : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, unsigned bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src),
               "r"(bytes), "r"(mbar)
               : "memory");
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}
__device__ __forceinline__ unsigned lds_u8(uint32_t a) {
  unsigned short v;
  asm volatile("ld.shared.u8 %0, [%1];" : "=h"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ void sts_u8(uint32_t a, unsigned v) {
  asm volatile("st.shared.u8 [%0], %1;" ::"r"(a), "h"(static_cast<unsigned short>(v)) : "memory");
}
__device__ __forceinline__ ulonglong2 ldg_u64x2(const ulonglong2* p) {
  ulonglong2 v;
  asm("ld.global.nc.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }

// Producer warp (warp 4 = first warp of WG1): tile t -> raw buffer b. Lane l handles keys l and l + 32.
__device__ __forceinline__ void tma_tile(const ulonglong2* addr, int t, uint32_t raw, uint32_t half, uint32_t bar, int lane) {
  const ulonglong2 a0 = ldg_u64x2(addr + t * BK + lane), a1 = ldg_u64x2(addr + t * BK + 32 + lane);
  sts_u8(half + lane, static_cast<unsigned>(a0.y & 8ull));
  sts_u8(half + 32 + lane, static_cast<unsigned>(a1.y & 8ull));
  __syncwarp();
  if (lane == 0) mbar_expect_tx(bar, BK * (576 + 16));
  __syncwarp();
  bulk_g2s(raw + lane * 576, reinterpret_cast<const void*>(a0.x), 576, bar);
  bulk_g2s(raw + (lane + 32) * 576, reinterpret_cast<const void*>(a1.x), 576, bar);
  bulk_g2s(raw + BK * 576 + lane * 16, reinterpret_cast<const void*>(a0.y & ~15ull), 16, bar);
  bulk_g2s(raw + BK * 576 + (lane + 32) * 16, reinterpret_cast<const void*>(a1.y & ~15ull), 16, bar);
}

// O warpgroups (256 threads, otid 0..255): raw tile in SMEM -> swizzled bf16 K tile. 4 threads per key, 9 chunks each.
__device__ __forceinline__ void dequant_tile(uint32_t raw, uint32_t half, uint32_t kbase, int otid) {
  const int kk = otid >> 2, c0 = otid & 3;
  const uint32_t src = raw + kk * 576, row = kbase + kk * 128;
  const uint2 scl = lds64(raw + BK * 576 + kk * 16 + lds_u8(half + kk));
#pragma unroll
  for (int j = 0; j < 9; ++j) {
    const int c = c0 + 4 * j;
    const uint4 v = lds128(src + 16 * c);
    if (c < 28) {
      const int sidx = c >> 2;
      const unsigned e = ((sidx < 4 ? scl.x : scl.y) >> (8 * (sidx & 3))) & 0xffu;
      uint4 lo, hi;
      dequant16(v, e, lo, hi);
      const uint32_t a = row + sidx * 8192;
      const int ch = (c & 3) * 2;
      sts128(a + (((ch) ^ (kk & 7)) << 4), lo);
      sts128(a + (((ch + 1) ^ (kk & 7)) << 4), hi);
    } else {
      sts128(row + 7 * 8192 + (((c - 28) ^ (kk & 7)) << 4), v);
    }
  }
}

__device__ __forceinline__ unsigned long long tile_mask(const int* codes, int tile, int lane, int row) {
  const int c0 = __ldg(codes + tile * BK + lane), c1 = __ldg(codes + tile * BK + 32 + lane);
  const unsigned lo = __ballot_sync(0xffffffffu, (c0 >> row) & 1), hi = __ballot_sync(0xffffffffu, (c1 >> row) & 1);
  return (static_cast<unsigned long long>(hi) << 32) | lo;
}

// in-place online softmax of one tile (WG0): sacc -> P (bf16, SMEM), alpha (SMEM)
__device__ __forceinline__ void softmax_tile(float (&sacc)[32], unsigned long long mask, float (&m_run)[2], float (&l_run)[2],
                                             float sl2, uint32_t pbuf, uint32_t alpha_s, int mrow0, int t4) {
#pragma unroll
  for (int rho = 0; rho < 2; ++rho) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int c = 8 * j + 2 * t4 + e;
        float& v = sacc[4 * j + 2 * rho + e];
        v = ((mask >> c) & 1ull) ? v * sl2 : -INFINITY;
      }
    }
    float m4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      m4[i] = fmaxf(fmaxf(sacc[8 * i + 2 * rho], sacc[8 * i + 2 * rho + 1]), fmaxf(sacc[8 * i + 4 + 2 * rho], sacc[8 * i + 5 + 2 * rho]));
    float tmax = fmaxf(fmaxf(m4[0], m4[1]), fmaxf(m4[2], m4[3]));
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
    const float mnew = fmaxf(m_run[rho], tmax);
    const float alpha = mnew == -INFINITY ? 1.f : exp2f(m_run[rho] - mnew);
    const int mr = mrow0 + 8 * rho;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float& x0 = sacc[4 * j + 2 * rho];
      float& x1 = sacc[4 * j + 2 * rho + 1];
      x0 = x0 == -INFINITY ? 0.f : exp2f(x0 - mnew);
      x1 = x1 == -INFINITY ? 0.f : exp2f(x1 - mnew);
      sts32(pbuf + mr * 128 + ((j ^ (mr & 7)) << 4) + 4 * t4, pack_bf16(x0, x1));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
      m4[i] = (sacc[8 * i + 2 * rho] + sacc[8 * i + 2 * rho + 1]) + (sacc[8 * i + 4 + 2 * rho] + sacc[8 * i + 5 + 2 * rho]);
    const float lsum = (m4[0] + m4[1]) + (m4[2] + m4[3]);
    l_run[rho] = l_run[rho] * alpha + lsum;
    m_run[rho] = mnew;
    if (t4 == 0) stsf(alpha_s + 4 * mr, alpha);
  }
}

__global__ void __launch_bounds__(NT3, 1) attn3_kernel(const __grid_constant__ Params p) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t sb = (smem_u32(smem_raw) + 1023u) & ~1023u;
  // warp-uniform role index (as CUTLASS does), so ptxas can allocate registers per setmaxnreg region
  const int wg = __shfl_sync(0xffffffffu, static_cast<int>(threadIdx.x) / 128, 0);
  const int s = blockIdx.x, tid = threadIdx.x, warp = (tid >> 5) & 3, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int nk = __ldg(p.nkeys + s), nt = (nk + BK - 1) / BK;
  const int* codes = p.codes + static_cast<long long>(s) * p.code_stride;
  const ulonglong2* addr = p.addr + static_cast<long long>(s) * p.addr_stride;
  const int mrow0 = 16 * warp + g;
  const uint32_t KB[2] = {sb + OFF_K0, sb + OFF_K1}, RB[2] = {sb + OFF_R0, sb + OFF_R1}, PB[2] = {sb + OFF_P0, sb + OFF_P1};
  const uint32_t HB[2] = {sb + OFF_HALF, sb + OFF_HALF + 64}, BAR[2] = {sb + OFF_BAR, sb + OFF_BAR + 8};

  if (tid == 0) {
    mbar_init(BAR[0], 1);
    mbar_init(BAR[1], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // Q (64 x 512) -> staging in K1 (swizzled), all threads
  const __nv_bfloat16* qs = p.q + static_cast<long long>(s) * 64 * D;
  for (int i = tid; i < 64 * 64; i += NT3) {
    const int m = i >> 6, ch = i & 63, cb = ch >> 3, c8 = ch & 7;
    cp_async16(sb + OFF_K1 + cb * 8192 + m * 128 + ((c8 ^ (m & 7)) << 4), qs + m * D + ch * 8);
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();

  if (wg == 0) {
    // =============================================================== S warpgroup
    asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
    unsigned qf[32][4];
#pragma unroll
    for (int ks = 0; ks < 32; ++ks) {
      const int m = 16 * warp + (lane & 15), d = 16 * ks + 8 * (lane >> 4);
      ldsm_x4(qf[ks], sb + OFF_K1 + (d >> 6) * 8192 + m * 128 + ((((d & 63) >> 3) ^ (m & 7)) << 4));
    }
    bar_arrive(5, NT3);  // Q staging (K1) may be overwritten
    float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
    float sacc[32];
    auto qk = [&](uint32_t kb) {
      wg_fence();
#pragma unroll
      for (int ks = 0; ks < 32; ++ks) WgRSbf<64>::mma(sacc, qf[ks], desc_k(kb + (ks >> 2) * 8192 + (ks & 3) * 32), ks > 0);
      wg_commit();
    };
    if (nt > 0) {
      named_bar(1, NT3);  // tile 0 in K0
      qk(KB[0]);
      const unsigned long long mask = tile_mask(codes, 0, lane, warp);
      wg_wait<0>();
      if (p.dbg_s != nullptr) {
        float* dd = p.dbg_s + static_cast<long long>(s) * 64 * 64;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int c = 8 * j + 2 * t4;
          dd[mrow0 * 64 + c] = sacc[4 * j];
          dd[mrow0 * 64 + c + 1] = sacc[4 * j + 1];
          dd[(mrow0 + 8) * 64 + c] = sacc[4 * j + 2];
          dd[(mrow0 + 8) * 64 + c + 1] = sacc[4 * j + 3];
        }
      }
      softmax_tile(sacc, mask, m_run, l_run, p.scale_log2, PB[0], sb + OFF_ALPHA, mrow0, t4);
      fence_proxy_async();
    }
    for (int it = 0; it < nt; ++it) {
      named_bar(1, NT3);  // P_it out; tile it+1 in K[(it+1)&1]
      KP(it, 0);
      if (it + 1 < nt) {
        const int nb = (it + 1) & 1;
        qk(KB[nb]);
        const unsigned long long mask = tile_mask(codes, it + 1, lane, warp);
        wg_wait<0>();
        KP(it, 1);
        softmax_tile(sacc, mask, m_run, l_run, p.scale_log2, PB[nb], sb + OFF_ALPHA + 256 * nb, mrow0, t4);
        fence_proxy_async();
        KP(it, 2);
      }
    }
#pragma unroll
    for (int rho = 0; rho < 2; ++rho) {
      float l = l_run[rho];
      l += __shfl_xor_sync(0xffffffffu, l, 1);
      l += __shfl_xor_sync(0xffffffffu, l, 2);
      const float m = m_run[rho];
      const float sink2 = __ldg(p.sink + (mrow0 + 8 * rho) % p.sink_rows) * LOG2E;
      const float mf = fmaxf(m, sink2);
      const float f = m == -INFINITY ? 0.f : exp2f(m - mf);
      const float den = l * f + (sink2 == -INFINITY ? 0.f : exp2f(sink2 - mf));
      const float so = (den > 0.f && mf < INFINITY) ? f / den : 0.f;
      if (t4 == 0) stsf(sb + OFF_SO + 4 * (mrow0 + 8 * rho), so);
    }
    named_bar(4, NT3);
  } else {
    // =============================================================== O warpgroups (+ producer warp 4)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 152;");
    const int otid = tid - 128, half = wg - 1;
    const bool prod = (tid >> 5) == 4;
    if (nt > 0) {
      if (prod) {
        tma_tile(addr, 0, RB[0], HB[0], BAR[0], lane);
        if (nt > 1) tma_tile(addr, 1, RB[1], HB[1], BAR[1], lane);
      }
      mbar_wait(BAR[0], 0);
      dequant_tile(RB[0], HB[0], KB[0], otid);
      fence_proxy_async();
      named_bar(2, 256);  // raw 0 read by everyone
      if (prod && nt > 2) tma_tile(addr, 2, RB[0], HB[0], BAR[0], lane);
      named_bar(1, NT3);  // K0 ready
    }
    named_bar(5, NT3);  // WG0 has Q in registers: K1 is free
    if (nt > 1) {
      mbar_wait(BAR[1], 0);
      dequant_tile(RB[1], HB[1], KB[1], otid);
      fence_proxy_async();
      named_bar(2, 256);
      if (prod && nt > 3) tma_tile(addr, 3, RB[1], HB[1], BAR[1], lane);
    }
    float o[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) o[i] = 0.f;
    for (int it = 0; it < nt; ++it) {
      const int cur = it & 1;
      named_bar(1, NT3);  // P_it, alpha_it out (and tile it+1 dequantized by us before)
      KP(it, 0);
      const uint32_t al = sb + OFF_ALPHA + 256 * cur;
      const float a0 = ldsf(al + 4 * mrow0), a1 = ldsf(al + 4 * (mrow0 + 8));
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
        WgSS<256, 1>::mma(o, desc_k(PB[cur] + ks * 32), desc_mn(KB[cur] + 4 * half * 8192 + ks * 2048, p.mn_lbo, p.mn_sbo), 1);
      wg_commit();
      wg_wait<0>();
      KP(it, 1);
      if (it + 2 < nt) {
        named_bar(3, 256);  // both O warpgroups done reading K[cur]
        mbar_wait(BAR[cur], ((it + 2) >> 1) & 1);
        dequant_tile(RB[cur], HB[cur], KB[cur], otid);
        fence_proxy_async();
        named_bar(2, 256);  // raw[cur] read by everyone
        if (prod && it + 4 < nt) tma_tile(addr, it + 4, RB[cur], HB[cur], BAR[cur], lane);
      }
      KP(it, 2);
    }
    named_bar(4, NT3);
    const float so0 = ldsf(sb + OFF_SO + 4 * mrow0), so1 = ldsf(sb + OFF_SO + 4 * (mrow0 + 8));
    const long long trow = 4LL * s + warp;
    __nv_bfloat16* out0 = p.out + (trow * 16 + g) * D + 256 * half;
    __nv_bfloat16* out1 = out0 + 8 * D;
    const float* cs = nullptr;
    if (p.cos_sin != nullptr && half == 1) cs = p.cos_sin + __ldg(p.positions + trow) * 64;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int d = 8 * j + 2 * t4;
      float v0 = bf16r(o[4 * j] * so0), v1 = bf16r(o[4 * j + 1] * so0);
      float v2 = bf16r(o[4 * j + 2] * so1), v3 = bf16r(o[4 * j + 3] * so1);
      if (cs != nullptr && j >= 24) {
        const int i = 4 * (j - 24) + t4;
        const float c = __ldg(cs + i), sn = __ldg(cs + 32 + i);
        const float x0 = __fadd_rn(__fmul_rn(v0, c), __fmul_rn(v1, sn)), y0 = __fsub_rn(__fmul_rn(v1, c), __fmul_rn(v0, sn));
        const float x1 = __fadd_rn(__fmul_rn(v2, c), __fmul_rn(v3, sn)), y1 = __fsub_rn(__fmul_rn(v3, c), __fmul_rn(v2, sn));
        v0 = x0, v1 = y0, v2 = x1, v3 = y1;
      }
      *reinterpret_cast<unsigned*>(out0 + d) = pack_bf16(v0, v1);
      *reinterpret_cast<unsigned*>(out1 + d) = pack_bf16(v2, v3);
    }
  }
}

}  // namespace k3v3

void attn3(torch::Tensor q, torch::Tensor out, torch::Tensor codes, torch::Tensor addr, torch::Tensor nkeys, torch::Tensor sink,
           double scale, c10::optional<torch::Tensor> positions, c10::optional<torch::Tensor> cos_sin, int64_t mn_lbo, int64_t mn_sbo,
           c10::optional<torch::Tensor> dbg_s, c10::optional<torch::Tensor> prof) {
  const int nseq = static_cast<int>(nkeys.size(0));
  TORCH_CHECK(q.dtype() == torch::kBFloat16 && q.is_contiguous() && q.size(0) == 4 * nseq && q.size(1) == 16 && q.size(2) == 512, "attn3: q");
  TORCH_CHECK(out.is_contiguous() && out.sizes() == q.sizes(), "attn3: out");
  TORCH_CHECK(addr.dtype() == torch::kInt64 && addr.dim() == 3 && addr.size(0) == nseq && addr.size(2) == 2 && addr.size(1) % 64 == 0,
              "attn3: addr");
  k3v3::Params p{};
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.codes = codes.data_ptr<int>();
  p.code_stride = static_cast<int>(codes.stride(0));
  p.addr = reinterpret_cast<const ulonglong2*>(addr.data_ptr<int64_t>());
  p.addr_stride = static_cast<int>(addr.size(1));
  p.nkeys = nkeys.data_ptr<int>();
  p.sink = sink.data_ptr<float>();
  p.sink_rows = static_cast<int>(sink.numel());
  TORCH_CHECK(p.sink_rows == 16 || p.sink_rows == 64, "attn3: sink has 16 or 64 entries");
  p.scale_log2 = static_cast<float>(scale) * k3c::LOG2E;
  p.positions = positions.has_value() ? reinterpret_cast<const long long*>(positions->data_ptr<int64_t>()) : nullptr;
  p.cos_sin = cos_sin.has_value() ? cos_sin->data_ptr<float>() : nullptr;
  TORCH_CHECK(p.cos_sin == nullptr || p.positions != nullptr, "attn3: inverse RoPE needs positions");
  p.mn_lbo = static_cast<int>(mn_lbo);
  p.mn_sbo = static_cast<int>(mn_sbo);
  p.dbg_s = dbg_s.has_value() ? dbg_s->data_ptr<float>() : nullptr;
  p.prof = prof.has_value() ? reinterpret_cast<unsigned long long*>(prof->data_ptr<int64_t>()) : nullptr;
  static bool attr_dev[64] = {};  // per device (one process may launch on several GPUs)
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  if (!attr_dev[cur_dev & 63]) {
    TORCH_CHECK(cudaFuncSetAttribute(k3v3::attn3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, k3v3::SMEM3) == cudaSuccess,
                "attn3 smem attr");
    attr_dev[cur_dev & 63] = true;
  }
  k3v3::attn3_kernel<<<nseq, k3v3::NT3, k3v3::SMEM3, at::cuda::getCurrentCUDAStream()>>>(p);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "attn3 launch");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn3", &attn3, "K3 v3: warp-specialized sparse MQA decode attention"); }
