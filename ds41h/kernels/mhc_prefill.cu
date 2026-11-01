// ds41h 'prefill' component, part 3: one fused kernel per mHC sublayer boundary for prefill (sm_90a).
//
// Replaces stock vLLM 0.32.0's unfused prefill path of mhc_fused_post_pre_delayed_tilelang
// (model_executor/kernels/mhc/tilelang.py:269-497, taken for num_tokens > 32 on sm90):
//   1. mhc_post_tilelang            residual_cur[t,o,:] = bf16(post[t,o] * x[t,:] + sum_i comb[t,i,o] * residual[t,i,:])
//   2. tf32_hc_prenorm_gemm (DeepGEMM) mixes[t,:] = residual_cur[t] (bf16, 20480) @ fn^T (fp32 [24, 20480], TF32 math),
//                                      sqrsum[t] = sum residual_cur[t]^2
//   3. mhc_pre_big_fuse_with_norm   rms -> pre/post (sigmoid) + comb (softmax + 20 Sinkhorn iterations), and the
//                                   delayed collapse u = bf16(sum_i pre_in[t,i] * residual_cur[t,i,:]) followed by
//                                   layer_input = bf16(u * rsqrt(mean(u^2) + eps) * norm_w)
// Stock streams residual_cur through HBM three times (~41 ms per 8192-token chunk for 41 boundaries at TP4).
// The collapse uses the PREVIOUS sublayer's pre-mix (pre_in), so it does not depend on this boundary's GEMV:
// everything is one pass over x [T, 5120] + residual [T, 4, 5120] (bf16) that writes residual_cur and the
// normalized layer input once.
//
// Work split (cluster of CL = 8 CTAs per BM = 64-token tile; CTA rank r owns hidden columns [640 r, 640 r + 640)):
//   * 8 warps per CTA = 4 m16 token tiles x 2 column halves (320 columns, 10 steps of 32 columns each).
//     Lane (g, t) of a warp holds tokens g and g + 8 of its tile and 8 consecutive columns 8t..8t+7 of a step,
//     loads them with 16-byte loads, computes the 4 post-mapped streams in fp32 (stock operation order: post * x,
//     then one FFMA per input stream), rounds them to bf16 (stored to residual_out) and uses the bf16 words
//     directly as the A fragments of mma.m16n8k16 (rows = tokens, k = stream x column; with the 8 columns of a
//     lane split as k {2t, 2t+1, 2t+8, 2t+9} of two consecutive k16 blocks the B fragment of a lane is 8
//     consecutive fn values of one row, so fn needs no k permutation, only a lane-major packing).
//   * GEMV on tensor cores: fn (fp32) = hi + lo with hi = bf16(fn), lo = bf16(fn - hi) (~16 mantissa bits,
//     ~2^-17 relative per product vs TF32's 2^-11 in stock), fp32 accumulation; packed once per weight
//     (ds41h/mhc_prefill.py pack_fn) and staged through shared memory with a 3-stage cp.async pipeline
//     (12 KB per step per column half, read by the 4 token-tile warps).
//   * The sum of squares for the RMS uses the bf16 stream values (stock: DeepGEMM squares the bf16 input).
//   * The collapse u (bf16, as stock rounds it before the RMSNorm) of the CTA's 64 tokens x 640 columns stays in
//     shared memory (84 KB). Per-token partials (24 mixes, sum residual_cur^2, sum u^2) are reduced over the two
//     column halves in shared memory and over the 8 CTAs of the cluster through distributed shared memory
//     (fixed rank order, deterministic). Every CTA then normalizes its own column slice of u and writes it;
//     CTA r additionally finishes tokens [8r, 8r + 8) of the tile: rms, pre / post sigmoids and the
//     lane-parallel Sinkhorn (same code and operation order as ds41h/kernels/prologue.cu sinkhorn_job).
//   HBM traffic per token: 10 KB (x) + 40 KB (residual) read, 40 KB (residual_out) + 10 KB (layer input) written;
//   fn: 1.97 MB per 64 tokens from L2.
// Mode PRE (template POST = false): no x / post / comb: residual is used as is (stock mhc_pre_delayed_tilelang
// with a carried pre-mix, the Engram-layer case); residual_out is not written.
// AUX: also writes aux = bf16(mean of the 4 bf16 streams) (stock capture_aux / write_aux).
#include <cooperative_groups.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.cuh"
#include "host.cuh"
#include "sm90.cuh"

namespace cg = cooperative_groups;

namespace ds41h_mhc {

constexpr int HC = 4, HID = 5120, MIX = HC * (HC + 2), HCD = HC * HID;
constexpr int NT = MIX / 8;                     // 3 n8 tiles of mixes
constexpr int CL = 8;                           // CTAs per cluster
constexpr int HS = HID / CL;                    // 640 columns per CTA
constexpr int BM = 64;                          // tokens per cluster tile
constexpr int MT = BM / 16;                     // 4 m16 tiles
constexpr int HALVES = 2;
constexpr int WARPS = MT * HALVES;              // 8
constexpr int THREADS = WARPS * 32;             // 256
constexpr int HW = HS / HALVES;                 // 320 columns per warp
constexpr int SH = 32;                          // columns per step (4 lanes x 8)
constexpr int STEPS = HW / SH;                  // 10
constexpr int GSTEPS = HID / SH;                // 160
constexpr int FN_BLK = 512;                     // one (step, stream, n-tile, hi/lo) fragment block: 32 lanes x 16 B
constexpr int FN_STEP = HC * NT * 2 * FN_BLK;   // 12288 B per step
constexpr int FN_PIECES = HALVES * FN_STEP / 16;  // 16-byte cp.async pieces per step (both halves)
constexpr int STAGES = 3;
constexpr int U_PITCH = HS * 2 + 64;            // 1344 B: rows g and g + 1 fall into different bank halves
constexpr int PST = 28;                         // partial row (floats): 24 mixes, sum r^2, sum u^2
constexpr int NPART = MIX + 2;
constexpr int CST = 24;                         // coefficient row (floats): post[4], comb[16] (i-major), pre[4]
constexpr int TOK_PER_CTA = BM / CL;            // 8 tokens finished (Sinkhorn) per CTA = one per warp
static_assert(HS % (HALVES * SH) == 0 && FN_PIECES % THREADS == 0 && TOK_PER_CTA == WARPS && GSTEPS == CL * HALVES * STEPS);

constexpr int SM_U = 0;
constexpr int SM_FN = SM_U + BM * U_PITCH;                    // 86016
constexpr int SM_PART = SM_FN + STAGES * HALVES * FN_STEP;    // 159744
constexpr int SM_PC = SM_PART + HALVES * BM * PST * 4;        // 174080
constexpr int SM_COEF = SM_PC + BM * PST * 4;                 // 181248
constexpr int SM_RN = SM_COEF + BM * CST * 4;                 // 187392
constexpr int SMEM = SM_RN + BM * 4;                          // 187648
static_assert(SM_FN % 128 == 0 && SM_PART % 16 == 0 && SM_COEF % 16 == 0 && SMEM <= 227 * 1024);

struct Params {
  const __nv_bfloat16* x;      // [T, HID]          sublayer output (POST)
  const __nv_bfloat16* res;    // [T, HC, HID]      residual streams in
  const float* post;           // [T, HC]           post mix (POST)
  const float* comb;           // [T, HC, HC]       comb[t][i][o] (POST)
  const float* pre_in;         // [T, HC] or null   carried pre-mix (null: one-hot stream 0)
  const uint8_t* fnp;          // packed fn (see pack layout in mhc_prefill.py)
  const float* hc_scale;       // [3]
  const float* hc_base;        // [24]
  const __nv_bfloat16* norm_w; // [HID]
  __nv_bfloat16* res_out;      // [T, HC, HID] (POST)
  __nv_bfloat16* layer_in;     // [T, HID]
  __nv_bfloat16* aux;          // [T, HID] (AUX)
  float* post_out;             // [T, HC]
  float* comb_out;             // [T, HC * HC]
  float* pre_out;              // [T, HC]
  int T;
  int iters;
  float rms_eps, hc_pre_eps, sk_eps, post_mult, norm_eps;
};

__device__ __forceinline__ unsigned long long evict_first() {
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ unsigned long long evict_last() {
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ uint4 ld_once(const void* p, unsigned long long pol) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ void st_global16(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void st_global16_if(void* p, uint4 v, bool pred) {
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %5, 0;\n @q st.global.v4.u32 [%0], {%1, %2, %3, %4};\n}" ::"l"(p), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w), "r"(unsigned(pred))
               : "memory");
}
__device__ __forceinline__ void unpack8(uint4 v, float (&f)[8]) {
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(u[i] << 16);
    f[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
  }
}
// Stock math (tilelang, no fast math): sigmoid = 1 / (1 + expf(-x)) with IEEE division and the accurate expf. This file
// is compiled WITHOUT --use_fast_math (mhc_prefill.py ext()), so expf is the libdevice function, not __expf.
__device__ __forceinline__ float sigmoid_(float x) { return __fdiv_rn(1.0f, 1.0f + expf(-x)); }

// Per-thread state of the main loop (all pointers advance with the step; offsets inside a step are immediates).
struct LaneState {
  const uint4* xq[2];        // x rows (uint4 = 8 bf16), at column hbase
  const uint4* rq[2];        // residual rows (stream i at + i * HID / 8)
  uint4* oq[2];              // residual_out rows (stream i at + i * HID / 8)
  uint4* aq[2];              // aux rows
  const uint8_t* fsrc;       // fn pieces of this thread for step 0 (+ s * FN_STEP)
  uint32_t fdst;             // smem address of this thread's fn pieces in stage 0 (+ stage * HALVES * FN_STEP)
  uint32_t usm;              // smem address of u[lr0][half * HW + 8 t]
  uint32_t fs;               // smem address of the fn fragments of this warp's half / lane in stage 0
  uint32_t cf[2];            // smem addresses of the coefficient rows lr0, lr0 + 8
  bool ok[2];
};

// fn fragments of step `s` (both column halves) -> stage buffer `stage`: 6 x 16 B per thread.
// piece = tid + 256 k: half = k / 3, q = tid + 256 (k % 3) (FN_STEP / 16 = 768 = 3 x 256).
__device__ __forceinline__ void issue_fn(const LaneState& L, int s, int stage, unsigned long long pol) {
  const uint8_t* src = L.fsrc + long(s) * FN_STEP;
  const uint32_t dst = L.fdst + stage * (HALVES * FN_STEP);
#pragma unroll
  for (int k = 0; k < FN_PIECES / THREADS; ++k) {
    const int hf = k / 3, q = (k % 3) * THREADS * 16;
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst + hf * FN_STEP + q),
                 "l"(src + long(hf) * STEPS * FN_STEP + q), "l"(pol)
                 : "memory");
  }
}

template <bool POST>
__device__ __forceinline__ void load_step(const LaneState& L, int s, uint4 (&x)[2], uint4 (&r)[2][HC], unsigned long long pol) {
  const int o4 = s * (SH / 8);
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    if constexpr (POST) x[rr] = ld_once(L.xq[rr] + o4, pol);
#pragma unroll
    for (int i = 0; i < HC; ++i) r[rr][i] = ld_once(L.rq[rr] + i * (HID / 8) + o4, pol);
  }
}

__device__ __forceinline__ float4 lds_f4(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint4 lds_u4(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ void sts_u4(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// One 32-column step: post map (POST), residual_out / aux stores, sum of squares, collapse into smem, GEMV mma.
template <bool POST, bool AUX>
__device__ __forceinline__ void do_step(const LaneState& L, int s, const uint4 (&cx)[2], const uint4 (&cr)[2][HC], float (&acc)[NT][4],
                                        float (&sq)[2], float (&usq)[2], unsigned long long pol_keep) {
  const int stage = s % STAGES;
  cp_async_wait<STAGES - 2>();
  __syncthreads();  // stage s visible to all; everyone is done with step s - 1, whose buffer is refilled below
  if (s + STAGES - 1 < STEPS) issue_fn(L, s + STAGES - 1, (s + STAGES - 1) % STAGES, pol_keep);
  cp_async_commit();
  const int o4 = s * (SH / 8);

  uint4 pk[2][HC];
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    if constexpr (POST) {
      float xf[8], rf[HC][8];
      unpack8(cx[rr], xf);
#pragma unroll
      for (int i = 0; i < HC; ++i) unpack8(cr[rr][i], rf[i]);
      const float4 po = lds_f4(L.cf[rr]);
      const float4 cm0 = lds_f4(L.cf[rr] + 16), cm1 = lds_f4(L.cf[rr] + 32), cm2 = lds_f4(L.cf[rr] + 48), cm3 = lds_f4(L.cf[rr] + 64);
      const float pov[HC] = {po.x, po.y, po.z, po.w};
      const float cmv[HC][HC] = {{cm0.x, cm0.y, cm0.z, cm0.w}, {cm1.x, cm1.y, cm1.z, cm1.w},
                                 {cm2.x, cm2.y, cm2.z, cm2.w}, {cm3.x, cm3.y, cm3.z, cm3.w}};
#pragma unroll
      for (int o = 0; o < HC; ++o) {
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float a = __fmul_rn(pov[o], xf[e]);  // stock mhc_post_tilelang: post * x, then one FFMA per input stream
#pragma unroll
          for (int i = 0; i < HC; ++i) a = fmaf(cmv[i][o], rf[i][e], a);
          v[e] = a;
        }
        pk[rr][o] = pack8_bf16(v);
      }
    } else {
#pragma unroll
      for (int o = 0; o < HC; ++o) pk[rr][o] = cr[rr][o];
    }
    const float4 pr = lds_f4(L.cf[rr] + 80);
    const float prv[HC] = {pr.x, pr.y, pr.z, pr.w};
    float u[8], ax[8];
#pragma unroll
    for (int o = 0; o < HC; ++o) {
      float vr[8];
      unpack8(pk[rr][o], vr);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        sq[rr] = fmaf(vr[e], vr[e], sq[rr]);
        u[e] = o == 0 ? __fmul_rn(prv[0], vr[e]) : fmaf(prv[o], vr[e], u[e]);
        if constexpr (AUX) ax[e] = o == 0 ? vr[e] : __fadd_rn(ax[e], vr[e]);
      }
      if constexpr (POST) st_global16_if(L.oq[rr] + o * (HID / 8) + o4, pk[rr][o], L.ok[rr]);
    }
    const uint4 ub = pack8_bf16(u);
    sts_u4(L.usm + rr * 8 * U_PITCH + s * SH * 2, ub);
    float uf[8];
    unpack8(ub, uf);
#pragma unroll
    for (int e = 0; e < 8; ++e) usq[rr] = fmaf(uf[e], uf[e], usq[rr]);
    if constexpr (AUX) {
#pragma unroll
      for (int e = 0; e < 8; ++e) ax[e] = __fmul_rn(ax[e], 0.25f);
      st_global16_if(L.aq[rr] + o4, pack8_bf16(ax), L.ok[rr]);
    }
  }

  // GEMV: per stream, the 6 B fragments (3 n-tiles x hi/lo, two k16 blocks each) are loaded once and each A quad
  // (k16 block kb of rows g / g + 8) feeds 6 consecutive HMMAs, so it is assembled once.
  const uint32_t fs = L.fs + stage * (HALVES * FN_STEP);
#pragma unroll
  for (int o = 0; o < HC; ++o) {
    uint4 b[NT][2];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
      for (int hl = 0; hl < 2; ++hl) b[j][hl] = lds_u4(fs + ((o * NT + j) * 2 + hl) * FN_BLK);
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
      for (int hl = 0; hl < 2; ++hl) hmma_bf16(acc[j], pk[0][o].x, pk[1][o].x, pk[0][o].y, pk[1][o].y, b[j][hl].x, b[j][hl].y);
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
      for (int hl = 0; hl < 2; ++hl) hmma_bf16(acc[j], pk[0][o].z, pk[1][o].z, pk[0][o].w, pk[1][o].w, b[j][hl].z, b[j][hl].w);
    }
  }
}

template <bool POST, bool AUX>
__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(THREADS, 1) mhc_boundary_kernel(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = int(cluster.block_rank());
  const int tile = blockIdx.x / CL;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int mt = warp % MT, half = warp / MT;
  const int T = p.T;
  const int tok0 = tile * BM;
  const unsigned long long pol_stream = evict_first(), pol_keep = evict_last();
  const uint32_t sbase = smem_u32(smem);

  float* coef = reinterpret_cast<float*>(smem + SM_COEF);
  float* rn = reinterpret_cast<float*>(smem + SM_RN);

  const int lr0 = mt * 16 + g;
  const int hbase = rank * HS + half * HW + 8 * t;
  LaneState L;
  L.fsrc = p.fnp + long(rank) * (HALVES * STEPS) * FN_STEP + threadIdx.x * 16;
  L.fdst = sbase + SM_FN + threadIdx.x * 16;
  L.fs = sbase + SM_FN + half * FN_STEP + lane * 16;
  L.usm = sbase + SM_U + lr0 * U_PITCH + (half * HW + 8 * t) * 2;
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int lr = lr0 + 8 * rr;
    const long row = min(tok0 + lr, T - 1);
    L.ok[rr] = tok0 + lr < T;
    L.cf[rr] = sbase + SM_COEF + lr * CST * 4;
    L.xq[rr] = POST ? reinterpret_cast<const uint4*>(p.x + row * HID + hbase) : nullptr;
    L.aq[rr] = AUX ? reinterpret_cast<uint4*>(p.aux + row * HID + hbase) : nullptr;
    L.rq[rr] = reinterpret_cast<const uint4*>(p.res + row * HCD + hbase);
    L.oq[rr] = POST ? reinterpret_cast<uint4*>(p.res_out + row * HCD + hbase) : nullptr;
  }

  // fn prologue (a weight: independent of upstream kernels)
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    issue_fn(L, s, s, pol_keep);
    cp_async_commit();
  }

  // per-token coefficients of the tile
  for (int idx = threadIdx.x; idx < BM * CST; idx += THREADS) {
    const int lr = idx / CST, c = idx % CST;
    const long tr = min(tok0 + lr, T - 1);
    float v;
    if (c < HC) {
      v = POST ? __ldg(p.post + tr * HC + c) : 0.0f;
    } else if (c < HC + HC * HC) {
      v = POST ? __ldg(p.comb + tr * HC * HC + (c - HC)) : 0.0f;
    } else {
      const int j = c - HC - HC * HC;
      v = p.pre_in != nullptr ? __ldg(p.pre_in + tr * HC + j) : (j == 0 ? 1.0f : 0.0f);
    }
    coef[idx] = v;
  }

  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
  float sq[2] = {0.0f, 0.0f}, usq[2] = {0.0f, 0.0f};
  uint8_t* usm = smem + SM_U;

  static_assert(STEPS % 2 == 0, "the main loop is unrolled by two");
  uint4 xa[2], ra[2][HC], xb[2], rb[2][HC];
  load_step<POST>(L, 0, xa, ra, pol_stream);
#pragma unroll 1
  for (int s = 0; s < STEPS; s += 2) {
    load_step<POST>(L, s + 1, xb, rb, pol_stream);
    do_step<POST, AUX>(L, s, xa, ra, acc, sq, usq, pol_keep);
    if (s + 2 < STEPS) load_step<POST>(L, s + 2, xa, ra, pol_stream);
    do_step<POST, AUX>(L, s + 1, xb, rb, acc, sq, usq, pol_keep);
  }
  cp_async_wait_all();

  // ---------------------------------------------------------------- per-token partials
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    sq[rr] += __shfl_xor_sync(0xffffffffu, sq[rr], 1);
    sq[rr] += __shfl_xor_sync(0xffffffffu, sq[rr], 2);
    usq[rr] += __shfl_xor_sync(0xffffffffu, usq[rr], 1);
    usq[rr] += __shfl_xor_sync(0xffffffffu, usq[rr], 2);
  }
  {
    float* part = reinterpret_cast<float*>(smem + SM_PART) + half * BM * PST;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      part[lr0 * PST + 8 * j + 2 * t] = acc[j][0];
      part[lr0 * PST + 8 * j + 2 * t + 1] = acc[j][1];
      part[(lr0 + 8) * PST + 8 * j + 2 * t] = acc[j][2];
      part[(lr0 + 8) * PST + 8 * j + 2 * t + 1] = acc[j][3];
    }
    if (t == 0) {
      part[lr0 * PST + MIX] = sq[0];
      part[lr0 * PST + MIX + 1] = usq[0];
      part[(lr0 + 8) * PST + MIX] = sq[1];
      part[(lr0 + 8) * PST + MIX + 1] = usq[1];
    }
  }
  __syncthreads();
  float* pc = reinterpret_cast<float*>(smem + SM_PC);
  {
    const float* p0 = reinterpret_cast<const float*>(smem + SM_PART);
    const float* p1 = p0 + BM * PST;
    for (int idx = threadIdx.x; idx < BM * NPART; idx += THREADS) {
      const int lr = idx / NPART, c = idx % NPART;
      pc[lr * PST + c] = p0[lr * PST + c] + p1[lr * PST + c];
    }
  }
  cluster.sync();  // partials of all 8 CTAs visible cluster-wide

  if (threadIdx.x < BM) {
    float s2 = 0.0f;
#pragma unroll
    for (int r = 0; r < CL; ++r) s2 += cluster.map_shared_rank(pc, r)[threadIdx.x * PST + MIX + 1];
    rn[threadIdx.x] = rsqrtf(__fdiv_rn(s2, float(HID)) + p.norm_eps);
  }
  {
    // Sinkhorn of token rank * 8 + warp (stock pre-norm epilogue, tilelang_kernels.py:350-412)
    const int lr = rank * TOK_PER_CTA + warp;
    const long tr = tok0 + lr;
    float tot = 0.0f;
    if (lane < NPART - 1) {
#pragma unroll
      for (int r = 0; r < CL; ++r) tot += cluster.map_shared_rank(pc, r)[lr * PST + lane];
    }
    const float sqv = __shfl_sync(0xffffffffu, tot, MIX);
    const float rms = rsqrtf(__fdiv_rn(sqv, float(HCD)) + p.rms_eps);
    const float mixv = __fmul_rn(tot, rms);
    if (tr < T) {
      if (lane < HC) {
        p.pre_out[tr * HC + lane] = sigmoid_(fmaf(mixv, __ldg(p.hc_scale + 0), __ldg(p.hc_base + lane))) + p.hc_pre_eps;
      } else if (lane < 2 * HC) {
        p.post_out[tr * HC + lane - HC] = sigmoid_(fmaf(mixv, __ldg(p.hc_scale + 1), __ldg(p.hc_base + lane))) * p.post_mult;
      }
    }
    // comb logits on lanes 16..31 (lane 16 + 4j + k = comb[j][k]; row = xor 1, 2; column = xor 4, 8)
    const float cmv = __shfl_sync(0xffffffffu, mixv, (lane - 8) & 31);
    const int ci = lane & 15;
    const float x = fmaf(cmv, __ldg(p.hc_scale + 2), __ldg(p.hc_base + 2 * HC + ci));
    float mx = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float e = expf(x - mx);
    float rsum = e + __shfl_xor_sync(0xffffffffu, e, 1);
    rsum += __shfl_xor_sync(0xffffffffu, rsum, 2);
    float c = __fdiv_rn(e, rsum) + p.sk_eps;
    float csum = c + __shfl_xor_sync(0xffffffffu, c, 4);
    csum += __shfl_xor_sync(0xffffffffu, csum, 8);
    c = __fdiv_rn(c, csum + p.sk_eps);
    for (int it = 1; it < p.iters; ++it) {
      rsum = c + __shfl_xor_sync(0xffffffffu, c, 1);
      rsum += __shfl_xor_sync(0xffffffffu, rsum, 2);
      c = __fdiv_rn(c, rsum + p.sk_eps);
      csum = c + __shfl_xor_sync(0xffffffffu, c, 4);
      csum += __shfl_xor_sync(0xffffffffu, csum, 8);
      c = __fdiv_rn(c, csum + p.sk_eps);
    }
    if (lane >= 16 && tr < T) p.comb_out[tr * HC * HC + ci] = c;
  }
  __syncthreads();  // rn visible

  // ---------------------------------------------------------------- normalized layer input (own column slice)
  for (int idx = threadIdx.x; idx < BM * (HS / 8); idx += THREADS) {
    const int lr = idx / (HS / 8), c8 = idx % (HS / 8);
    const long tr = tok0 + lr;
    if (tr >= T) continue;
    float uf[8], wf[8], o[8];
    unpack8(*reinterpret_cast<const uint4*>(usm + lr * U_PITCH + c8 * 16), uf);
    unpack8(__ldg(reinterpret_cast<const uint4*>(p.norm_w + rank * HS + c8 * 8)), wf);
    const float r = rn[lr];
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = __fmul_rn(__fmul_rn(uf[e], r), wf[e]);
    st_global16(p.layer_in + tr * HID + rank * HS + c8 * 8, pack8_bf16(o));
  }
  cluster.sync();  // no CTA leaves while others may still read its partials
}

template <bool POST, bool AUX>
void launch(const Params& p, int tiles) {
  auto kernel = mhc_boundary_kernel<POST, AUX>;
  opt_in_smem<mhc_boundary_kernel<POST, AUX>>(SMEM);
  kernel<<<dim3(tiles * CL), dim3(THREADS), SMEM, c10::cuda::getCurrentCUDAStream()>>>(p);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "ds41h mhc_boundary launch failed: ", cudaGetErrorString(err));
}

}  // namespace ds41h_mhc

// =====================================================================================================================
// v2 (default): same math and cluster tile as v1, restructured for the 200 W / 345 MHz regime where v1 is issue- and
// latency-bound (1 CTA per SM, a __syncthreads per 32-column step, a serial warp-shuffle Sinkhorn with the SM idle,
// ~120 register MOVs per step to build mma A fragments):
//   * transposed GEMV: mixes^T = fn48 (48 rows = 24 mixes hi + 24 mixes lo) x r^T, i.e. fn is the A operand (3 m16
//     tiles, pre-packed per lane: ds41h/mhc_prefill.py pack_fn(impl=2)) and the post-mapped bf16 words are the B
//     operand. A lane's B pair of a k16 block is two ADJACENT words of the 16-byte vector it stores to residual_out
//     (b0 = cols 8t..8t+1, b1 = cols 8t+2..8t+3 of its token), so the same registers feed st.global.v4 and the mma
//     without any register shuffling; rows g / g + 8 of an m16 tile hold the hi and lo halves of mix g, g + 8, g + 16
//     in the same lane, summed in the epilogue.
//   * 2 CTAs per SM (<= 128 registers, ~100 KB smem): one CTA's epilogue overlaps the other's streaming.
//   * fn steps (12 KB per 32 columns, contiguous) arrive by cp.async.bulk into a 3-stage ring per column half with
//     full/empty mbarriers: one producer lane per half, no CTA-wide barrier in the main loop.
//   * the unnormalized collapse u (bf16) goes to layer_input itself (L2 evict_last) instead of 80 KB of smem; after
//     the cluster reduction each CTA re-reads its slice from L2, scales it and writes the final values.
//   * Sinkhorn: one thread per token (register 4x4 matrix, reciprocal-multiply normalizations) in one warp while the
//     other warps normalize.
namespace ds41h_mhc2 {
using ds41h_mhc::Params;
using ds41h_mhc::HC;
using ds41h_mhc::HID;
using ds41h_mhc::MIX;
using ds41h_mhc::HCD;
using ds41h_mhc::NT;

constexpr int CL = 8, HS = HID / CL, BM = 64, MT = BM / 16, HALVES = 2, WARPS = MT * HALVES, THREADS = WARPS * 32;
constexpr int HW = HS / HALVES, SH = 32, STEPS = HW / SH, GSTEPS = HID / SH;
constexpr int FN_BLK = 512, FN_STEP = HC * NT * 2 * FN_BLK;  // 12288 B: one 32-column step of the packed fn
constexpr int STAGES = 3;
constexpr int PST = 28, NPART = MIX + 2, CST = 24, TOK_PER_CTA = BM / CL;
static_assert(TOK_PER_CTA <= 32 && GSTEPS == CL * HALVES * STEPS && STEPS >= STAGES);

constexpr int SM_FN = 0;
constexpr int SM_BAR = SM_FN + HALVES * STAGES * FN_STEP;  // 73728: full[2][3], empty[2][3]
constexpr int SM_COEF = SM_BAR + 128;
constexpr int SM_PART = SM_COEF + BM * CST * 4;
constexpr int SM_PC = SM_PART + HALVES * BM * PST * 4;
constexpr int SM_RN = SM_PC + BM * PST * 4;
constexpr int SMEM = SM_RN + BM * 4;  // 101760 B: two CTAs per SM
static_assert(SM_COEF % 16 == 0 && SM_PART % 16 == 0 && SM_PC % 16 == 0 && 2 * (SMEM + 1024) <= 228 * 1024);

__device__ __forceinline__ void mbar_init_u32(uint32_t bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive_u32(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait_u32(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}" ::"r"(bar), "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, unsigned long long pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar), "l"(pol)
               : "memory");
}
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ void st_global16_hint_if(void* ptr, uint4 v, unsigned long long pol, bool pred) {
  asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %6, 0;\n @q st.global.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;\n}" ::"l"(ptr),
               "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol), "r"(unsigned(pred))
               : "memory");
}
__device__ __forceinline__ uint4 ld_cg16(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr) : "memory");
  return v;
}

// Stock pre-norm epilogue for one token (tilelang_kernels.py mhc_pre_big_fuse_with_norm: rms, pre / post sigmoids,
// softmax + Sinkhorn), one thread per token. tot = 24 mixes + sum of squares (cluster totals).
__device__ __forceinline__ void finish_token(const Params& p, const float (&tot)[MIX + 1], long tr) {
  const float rms = rsqrtf(__fdiv_rn(tot[MIX], float(HCD)) + p.rms_eps);
  const float sc0 = __ldg(p.hc_scale + 0), sc1 = __ldg(p.hc_scale + 1), sc2 = __ldg(p.hc_scale + 2);
  float4 pre4, post4;
  float* prev = reinterpret_cast<float*>(&pre4);
  float* postv = reinterpret_cast<float*>(&post4);
#pragma unroll
  for (int j = 0; j < HC; ++j) {
    prev[j] = ds41h_mhc::sigmoid_(fmaf(__fmul_rn(tot[j], rms), sc0, __ldg(p.hc_base + j))) + p.hc_pre_eps;
    postv[j] = ds41h_mhc::sigmoid_(fmaf(__fmul_rn(tot[HC + j], rms), sc1, __ldg(p.hc_base + HC + j))) * p.post_mult;
  }
  float c[HC][HC];
#pragma unroll
  for (int j = 0; j < HC; ++j) {
#pragma unroll
    for (int k = 0; k < HC; ++k) c[j][k] = fmaf(__fmul_rn(tot[2 * HC + j * HC + k], rms), sc2, __ldg(p.hc_base + 2 * HC + j * HC + k));
  }
  const float eps = p.sk_eps;
#pragma unroll
  for (int j = 0; j < HC; ++j) {  // softmax over k, + eps
    const float mx = fmaxf(fmaxf(c[j][0], c[j][1]), fmaxf(c[j][2], c[j][3]));
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < HC; ++k) {
      c[j][k] = expf(c[j][k] - mx);
      sum += c[j][k];
    }
#pragma unroll
    for (int k = 0; k < HC; ++k) c[j][k] = __fdiv_rn(c[j][k], sum) + eps;  // stock: comb / row_sum + eps (exact division)
  }
  auto col_norm = [&]() {
#pragma unroll
    for (int k = 0; k < HC; ++k) {
      const float d = ((c[0][k] + c[1][k]) + c[2][k]) + c[3][k] + eps;
#pragma unroll
      for (int j = 0; j < HC; ++j) c[j][k] = __fdiv_rn(c[j][k], d);
    }
  };
  auto row_norm = [&]() {
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      const float d = ((c[j][0] + c[j][1]) + c[j][2]) + c[j][3] + eps;
#pragma unroll
      for (int k = 0; k < HC; ++k) c[j][k] = __fdiv_rn(c[j][k], d);
    }
  };
  col_norm();
  for (int it = 1; it < p.iters; ++it) {
    row_norm();
    col_norm();
  }
  *reinterpret_cast<float4*>(p.pre_out + tr * HC) = pre4;
  *reinterpret_cast<float4*>(p.post_out + tr * HC) = post4;
  float4* co = reinterpret_cast<float4*>(p.comb_out + tr * HC * HC);
#pragma unroll
  for (int j = 0; j < HC; ++j) co[j] = make_float4(c[j][0], c[j][1], c[j][2], c[j][3]);
}

template <bool POST, bool AUX>
#ifndef MHC_V2_MIN_BLOCKS
#define MHC_V2_MIN_BLOCKS 2
#endif
__global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(THREADS, MHC_V2_MIN_BLOCKS) mhc_boundary_v2(const Params p) {
  using ds41h_mhc::evict_first;
  using ds41h_mhc::evict_last;
  using ds41h_mhc::ld_once;
  using ds41h_mhc::lds_f4;
  using ds41h_mhc::lds_u4;
  using ds41h_mhc::st_global16;
  using ds41h_mhc::unpack8;
  extern __shared__ __align__(128) uint8_t smem[];
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = int(cluster.block_rank());
  const int tile = blockIdx.x / CL;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int mt = warp % MT, half = warp / MT;
  const int T = p.T;
  const int tok0 = tile * BM;
  const unsigned long long pol_stream = evict_first(), pol_keep = evict_last();
  const uint32_t sbase = smem_u32(smem);
  const uint32_t bar_full = sbase + SM_BAR + half * STAGES * 8;
  const uint32_t bar_empty = sbase + SM_BAR + (HALVES + half) * STAGES * 8;
  const uint32_t fn_ring = sbase + SM_FN + half * STAGES * FN_STEP;
  const uint8_t* fn_src = p.fnp + long(rank * HALVES + half) * STEPS * FN_STEP;
  const bool producer = mt == 0 && lane == 0;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < HALVES * STAGES; ++i) {
      mbar_init_u32(sbase + SM_BAR + i * 8, 1);                              // full: producer arrive + tx bytes
      mbar_init_u32(sbase + SM_BAR + (HALVES * STAGES + i) * 8, MT);         // empty: one arrive per consumer warp
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  float* coef = reinterpret_cast<float*>(smem + SM_COEF);
  for (int idx = threadIdx.x; idx < BM * CST; idx += THREADS) {
    const int lr = idx / CST, c = idx % CST;
    const long tr = min(tok0 + lr, T - 1);
    float v;
    if (c < HC) {
      v = POST ? __ldg(p.post + tr * HC + c) : 0.0f;
    } else if (c < HC + HC * HC) {
      v = POST ? __ldg(p.comb + tr * HC * HC + (c - HC)) : 0.0f;
    } else {
      const int j = c - HC - HC * HC;
      v = p.pre_in != nullptr ? __ldg(p.pre_in + tr * HC + j) : (j == 0 ? 1.0f : 0.0f);
    }
    coef[idx] = v;
  }
  __syncthreads();  // barriers initialized, coefficients visible
  if (producer) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_expect_tx(bar_full + s * 8, FN_STEP);
      bulk_g2s(fn_ring + s * FN_STEP, fn_src + long(s) * FN_STEP, FN_STEP, bar_full + s * 8, pol_keep);
    }
  }

  const int lr0 = mt * 16 + g;
  const int hbase = rank * HS + half * HW + 8 * t;
  // 32-bit element offsets (T < 2^31 / 20480, checked on the host); bases stay in the parameter bank.
  int xo[2], ro[2];
  uint32_t cf[2];
  bool ok[2];
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    const int lr = lr0 + 8 * rr;
    const int row = min(tok0 + lr, T - 1);
    ok[rr] = tok0 + lr < T;
    cf[rr] = sbase + SM_COEF + lr * CST * 4;
    xo[rr] = row * HID + hbase;
    ro[rr] = row * HCD + hbase;
  }
  auto xq = [&](int rr) { return reinterpret_cast<const uint4*>(p.x + xo[rr]); };
  auto rq = [&](int rr) { return reinterpret_cast<const uint4*>(p.res + ro[rr]); };
  auto oq = [&](int rr) { return reinterpret_cast<uint4*>(p.res_out + ro[rr]); };
  auto uq = [&](int rr) { return reinterpret_cast<uint4*>(p.layer_in + xo[rr]); };
  auto aq = [&](int rr) { return reinterpret_cast<uint4*>(p.aux + xo[rr]); };

  // acc[m-tile][n-tile = row rr][4]: C[m = 48-row mix index][n = token]; lane (g, t) holds rows g, g + 8 of each
  // m-tile for tokens 2t, 2t + 1 of the n-tile (this warp's tokens lr0 - g + {2t, 2t + 1} (+ 8)).
  float acc[3][2][4];
#pragma unroll
  for (int m = 0; m < 3; ++m) {
#pragma unroll
    for (int n = 0; n < 2; ++n) acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.0f;
  }
  float usq[2] = {0.0f, 0.0f}, rsq[2] = {0.0f, 0.0f};

  uint4 xd[2], rd[2][HC];
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    if constexpr (POST) xd[rr] = ld_once(xq(rr), pol_stream);
#pragma unroll
    for (int i = 0; i < HC; ++i) rd[rr][i] = ld_once(rq(rr) + i * (HID / 8), pol_stream);
  }

#pragma unroll 1
  for (int s = 0; s < STEPS; ++s) {
    const int stage = s % STAGES;
    const uint32_t par = (s / STAGES) & 1;
    const int o4 = s * (SH / 8);
    uint4 pk[2][HC];
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
#ifdef MHC_EXP_NO_POST
      if constexpr (false) {
#else
      if constexpr (POST) {
#endif
        const float4 po = lds_f4(cf[rr]);
        const float pov[HC] = {po.x, po.y, po.z, po.w};
        float cmv[HC][HC];
#pragma unroll
        for (int i = 0; i < HC; ++i) {
          const float4 cm = lds_f4(cf[rr] + 16 + 16 * i);
          cmv[i][0] = cm.x, cmv[i][1] = cm.y, cmv[i][2] = cm.z, cmv[i][3] = cm.w;
        }
        const unsigned xw[4] = {xd[rr].x, xd[rr].y, xd[rr].z, xd[rr].w};
        unsigned rw[HC][4];
#pragma unroll
        for (int i = 0; i < HC; ++i) rw[i][0] = rd[rr][i].x, rw[i][1] = rd[rr][i].y, rw[i][2] = rd[rr][i].z, rw[i][3] = rd[rr][i].w;
        unsigned ow[HC][4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {  // one bf16x2 word (2 columns) at a time keeps the live set small
          const float2 xf = unpack_bf16(xw[w]);
          float v[HC][2];
#pragma unroll
          for (int o = 0; o < HC; ++o) {  // stock: post * x, then one FFMA per input stream (in order)
            v[o][0] = __fmul_rn(pov[o], xf.x);
            v[o][1] = __fmul_rn(pov[o], xf.y);
          }
#pragma unroll
          for (int i = 0; i < HC; ++i) {
            const float2 rf = unpack_bf16(rw[i][w]);
#pragma unroll
            for (int o = 0; o < HC; ++o) {
              v[o][0] = fmaf(cmv[i][o], rf.x, v[o][0]);
              v[o][1] = fmaf(cmv[i][o], rf.y, v[o][1]);
            }
          }
#pragma unroll
          for (int o = 0; o < HC; ++o) ow[o][w] = pack_bf16(v[o][0], v[o][1]);
        }
#pragma unroll
        for (int o = 0; o < HC; ++o) {
          pk[rr][o] = make_uint4(ow[o][0], ow[o][1], ow[o][2], ow[o][3]);
          ds41h_mhc::st_global16_if(oq(rr) + o * (HID / 8) + o4, pk[rr][o], ok[rr]);
        }
      } else {
#pragma unroll
        for (int o = 0; o < HC; ++o) pk[rr][o] = rd[rr][o];
#ifdef MHC_EXP_NO_POST
        if constexpr (POST) {
#pragma unroll
          for (int o = 0; o < HC; ++o) ds41h_mhc::st_global16_if(oq(rr) + o * (HID / 8) + o4, pk[rr][o], ok[rr]);
        }
#endif
      }
      // delayed collapse u = bf16(sum_o pre_o * bf16(v_o)) (+ aux = bf16(mean_o bf16(v_o)))
#ifndef MHC_EXP_NO_COLLAPSE
      const float4 pr = lds_f4(cf[rr] + 80);
      const float prv[HC] = {pr.x, pr.y, pr.z, pr.w};
      float u[8], ax[8];
#pragma unroll
      for (int o = 0; o < HC; ++o) {
        float vr[8];
        unpack8(pk[rr][o], vr);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          rsq[rr] = fmaf(vr[e], vr[e], rsq[rr]);  // sum of squares of the rounded streams (stock: bf16 GEMM input)
          u[e] = o == 0 ? __fmul_rn(prv[0], vr[e]) : fmaf(prv[o], vr[e], u[e]);
          if constexpr (AUX) ax[e] = o == 0 ? vr[e] : __fadd_rn(ax[e], vr[e]);
        }
      }
      const uint4 ub = pack8_bf16(u);
      st_global16_hint_if(uq(rr) + o4, ub, pol_keep, ok[rr]);
      float uf[8];
      unpack8(ub, uf);
#pragma unroll
      for (int e = 0; e < 8; ++e) usq[rr] = fmaf(uf[e], uf[e], usq[rr]);
      if constexpr (AUX) {
#pragma unroll
        for (int e = 0; e < 8; ++e) ax[e] = __fmul_rn(ax[e], 0.25f);
        ds41h_mhc::st_global16_if(aq(rr) + o4, pack8_bf16(ax), ok[rr]);
      }
#endif
    }

    // tensor pipe: mixes^T += fn48 (A, smem) x r^T (B = this lane's post-mapped words, no shuffling)
    mbar_wait_u32(bar_full + stage * 8, par);
    const uint32_t fs = fn_ring + stage * FN_STEP + lane * 16;
#ifndef MHC_EXP_NO_GEMV
#pragma unroll
    for (int o = 0; o < HC; ++o) {
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
#pragma unroll
        for (int m = 0; m < 3; ++m) {
          const uint4 af = lds_u4(fs + ((o * 3 + m) * 2 + kb) * FN_BLK);
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            const unsigned b0 = kb == 0 ? pk[n][o].x : pk[n][o].z;
            const unsigned b1 = kb == 0 ? pk[n][o].y : pk[n][o].w;
            hmma_bf16(acc[m][n], af.x, af.y, af.z, af.w, b0, b1);
          }
        }
      }
    }
#else
    (void)fs;
#endif
    if (s + 1 < STEPS) {  // next step's loads (issued after the GEMV: register budget of 2 CTAs per SM)
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        if constexpr (POST) xd[rr] = ld_once(xq(rr) + o4 + SH / 8, pol_stream);
#pragma unroll
        for (int i = 0; i < HC; ++i) rd[rr][i] = ld_once(rq(rr) + i * (HID / 8) + o4 + SH / 8, pol_stream);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive_u32(bar_empty + stage * 8);
    if (producer && s + STAGES < STEPS) {
      mbar_wait_u32(bar_empty + stage * 8, par);
      mbar_expect_tx(bar_full + stage * 8, FN_STEP);
      bulk_g2s(fn_ring + stage * FN_STEP, fn_src + long(s + STAGES) * FN_STEP, FN_STEP, bar_full + stage * 8, pol_keep);
    }
  }

  // ---------------------------------------------------------------- per-token partials (24 mixes, sum r^2, sum u^2)
#pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    usq[rr] += __shfl_xor_sync(0xffffffffu, usq[rr], 1);
    usq[rr] += __shfl_xor_sync(0xffffffffu, usq[rr], 2);
    rsq[rr] += __shfl_xor_sync(0xffffffffu, rsq[rr], 1);
    rsq[rr] += __shfl_xor_sync(0xffffffffu, rsq[rr], 2);
  }
  {
    float* part = reinterpret_cast<float*>(smem + SM_PART) + half * BM * PST;
    // 48 rows: m-tile 0 = hi mixes 0..15, m-tile 1 = hi 16..23 | lo 0..7, m-tile 2 = lo 8..23 -> lane g holds hi and lo
    // of mixes g, g + 8, g + 16:   mix g      = mt0.row g      + mt1.row g + 8
    //                              mix g + 8  = mt0.row g + 8  + mt2.row g
    //                              mix g + 16 = mt1.row g      + mt2.row g + 8
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int tok = mt * 16 + n * 8 + 2 * t + c;
        part[tok * PST + g] = acc[0][n][c] + acc[1][n][2 + c];
        part[tok * PST + g + 8] = acc[0][n][2 + c] + acc[2][n][c];
        part[tok * PST + g + 16] = acc[1][n][c] + acc[2][n][2 + c];
      }
    }
    if (t == 0) {
      part[lr0 * PST + MIX] = rsq[0];
      part[(lr0 + 8) * PST + MIX] = rsq[1];
      part[lr0 * PST + MIX + 1] = usq[0];
      part[(lr0 + 8) * PST + MIX + 1] = usq[1];
    }
  }
  __syncthreads();
  float* pc = reinterpret_cast<float*>(smem + SM_PC);
  {
    const float* p0 = reinterpret_cast<const float*>(smem + SM_PART);
    const float* p1 = p0 + BM * PST;
    for (int idx = threadIdx.x; idx < BM * NPART; idx += THREADS) {
      const int lr = idx / NPART, c = idx % NPART;
      pc[lr * PST + c] = p0[lr * PST + c] + p1[lr * PST + c];
    }
  }
  cluster_arrive();
  cluster_wait();  // partials of all 8 CTAs visible cluster-wide

  float* rn = reinterpret_cast<float*>(smem + SM_RN);
  if (threadIdx.x < BM) {
    float s2 = 0.0f;
#pragma unroll
    for (int r = 0; r < CL; ++r) s2 += cluster.map_shared_rank(pc, r)[threadIdx.x * PST + MIX + 1];
    rn[threadIdx.x] = rsqrtf(__fdiv_rn(s2, float(HID)) + p.norm_eps);
  }
  const bool sk = warp == WARPS - 1 && lane < TOK_PER_CTA;
  const int sk_lr = rank * TOK_PER_CTA + lane;
  float tot[MIX + 1];
  if (sk) {
#pragma unroll
    for (int c = 0; c <= MIX; ++c) tot[c] = 0.0f;
#pragma unroll
    for (int r = 0; r < CL; ++r) {
      const float* q = cluster.map_shared_rank(pc, r) + sk_lr * PST;
#pragma unroll
      for (int c = 0; c <= MIX; ++c) tot[c] += q[c];
    }
  }
  cluster_arrive();  // done reading the other CTAs' partials (waited for before exit)
  __syncthreads();   // rn visible
  if (sk && tok0 + sk_lr < T) finish_token(p, tot, tok0 + sk_lr);

  // ---------------------------------------------------------------- normalized layer input (own column slice, from L2)
  for (int idx = threadIdx.x; idx < BM * (HS / 8); idx += THREADS) {
    const int lr = idx / (HS / 8), c8 = idx % (HS / 8);
    const long tr = tok0 + lr;
    if (tr >= T) continue;
    __nv_bfloat16* dst = p.layer_in + tr * HID + rank * HS + c8 * 8;
    float uf[8], wf[8], o[8];
    unpack8(ld_cg16(dst), uf);
    unpack8(__ldg(reinterpret_cast<const uint4*>(p.norm_w + rank * HS + c8 * 8)), wf);
    const float r = rn[lr];
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = __fmul_rn(__fmul_rn(uf[e], r), wf[e]);
    st_global16(dst, pack8_bf16(o));
  }
  cluster_wait();
}

template <bool POST, bool AUX>
void launch(const Params& p, int tiles) {
  auto kernel = mhc_boundary_v2<POST, AUX>;
  opt_in_smem<mhc_boundary_v2<POST, AUX>>(SMEM);
  kernel<<<dim3(tiles * CL), dim3(THREADS), SMEM, c10::cuda::getCurrentCUDAStream()>>>(p);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "ds41h mhc_boundary v2 launch failed: ", cudaGetErrorString(err));
}

}  // namespace ds41h_mhc2

// =====================================================================================================================
// v3: persistent clusters of 16 CTAs (non-portable cluster size), CTA r owns columns [320 r, 320 r + 320) of every
// token for the whole kernel, so its slice of the packed fn (hi + lo, 48 mma rows x 4 streams x 320 columns) lives in
// REGISTERS (each of the 10 compute warps holds its 32 columns: 96 registers per lane) -- no fn traffic per token.
// Tokens stream through in stages of 8 (one mma n-tile): per stage and token the CTA bulk-copies its contiguous 640 B
// slice of x and of each residual stream (cp.async.bulk, full/empty mbarrier ring), i.e. long DRAM bursts instead of
// v1/v2's 64 B pieces per row. Per-token partials (24 mixes, sum r^2, sum u^2) are reduced over the 10 warps by the
// service warp (warp 9, also the producer) and sent through distributed shared memory to the stage's leader CTA
// (stage q -> CTA q % 16), whose warps 0..7 finish one token each (rms, sigmoids, Sinkhorn: stock op order) and push
// the RMSNorm factor back to all 16 CTAs (remote mbarrier arrive). Each CTA keeps its unnormalized u slice for D
// stages in a smem ring and writes the final layer input once the factor arrives. All waits are on stages that are
// already complete or earlier in every CTA's program order (see the comments at each wait), so the pipeline cannot
// deadlock; DS41H_MHC_DEBUG builds trap after ~2^28 spins.
namespace ds41h_mhc3 {
using ds41h_mhc::Params;
using ds41h_mhc::HC;
using ds41h_mhc::HID;
using ds41h_mhc::MIX;
using ds41h_mhc::HCD;

constexpr int CL = 16, HS = HID / CL, WARPS = HS / 32, THREADS = WARPS * 32;  // 320 columns, 10 warps
#ifndef MHC3_D
#define MHC3_D 6
#endif
#ifndef MHC3_NR
#define MHC3_NR 8
#endif
#ifndef MHC3_NS
#define MHC3_NS 3
#endif
constexpr int TOK = 8, NS = MHC3_NS, NR = MHC3_NR, D = MHC3_D, NSCR = 2;
constexpr int NPART = MIX + 2;            // 24 mixes, sum r^2, sum u^2
constexpr int SRC = HC + 1;               // x + 4 residual streams
constexpr int ROWB = HS * 2;              // 640 B per (token, source)
constexpr int TOKB = SRC * ROWB + 64;     // token pitch in a stage (pad: conflict-free 16 B lane loads)
constexpr int CO_POST = 0, CO_COMB = TOK * HC * 4, CO_PRE = CO_COMB + TOK * HC * HC * 4, CO_BYTES = CO_PRE + TOK * HC * 4;
constexpr int STAGE = TOK * TOKB + CO_BYTES;
constexpr int UTOKB = ROWB + 64;          // u ring token pitch
constexpr int USLOT = TOK * UTOKB;
static_assert(D < NR && NS >= 2 && WARPS == 10 && TOK == 8);

constexpr int SM_STAGE = 0;
constexpr int SM_U = SM_STAGE + NS * STAGE;
constexpr int SM_SCR = SM_U + NR * USLOT;                          // [NSCR][WARPS][TOK][NPART] f32
constexpr int SM_RED = SM_SCR + NSCR * WARPS * TOK * NPART * 4;    // [CL][TOK][NPART] f32 (as a stage leader)
constexpr int SM_RN = SM_RED + CL * TOK * NPART * 4;               // [NR][TOK] f32
constexpr int SM_BAR = SM_RN + NR * TOK * 4;
constexpr int B_FULL = 0, B_EMPTY = NS, B_DONE = 2 * NS, B_SFREE = 2 * NS + NSCR, B_RED = 2 * NS + 2 * NSCR, B_RN = B_RED + 1;
constexpr int NBAR = B_RN + NR;
constexpr int SMEM = SM_BAR + NBAR * 8;
static_assert(SM_U % 16 == 0 && SM_SCR % 16 == 0 && SM_RED % 16 == 0 && SM_RN % 16 == 0 && SM_BAR % 8 == 0 && SMEM <= 227 * 1024);

#ifdef DS41H_MHC_DEBUG
#define MHC3_SPIN_GUARD(n) if (++(n) > (1u << 28)) __trap();
#else
#define MHC3_SPIN_GUARD(n)
#endif

__device__ __forceinline__ uint32_t bar_addr(uint32_t sbase, int i) { return sbase + SM_BAR + i * 8; }
__device__ __forceinline__ bool try_wait_cl(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void wait_cl(uint32_t bar, uint32_t parity) {
  unsigned n = 0;
  while (!try_wait_cl(bar, parity)) { MHC3_SPIN_GUARD(n) }
}
__device__ __forceinline__ void arrive_local(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.release.cta.shared::cta.b64 st, [%0];\n}" ::"r"(bar) : "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void arrive_remote(uint32_t cluster_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(cluster_addr) : "memory");
}
__device__ __forceinline__ void st_cluster_f32(uint32_t cluster_addr, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(cluster_addr), "f"(v) : "memory");
}
__device__ __forceinline__ void expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, unsigned long long pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar), "l"(pol)
               : "memory");
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

struct Ctx {
  int rank, cluster_id, nclusters, nblocks, T;
  uint32_t sbase;
};

__device__ __forceinline__ int stage_tok0(const Ctx& c, int q) { return (c.cluster_id + q * c.nclusters) * TOK; }

// Producer (warp 9): bulk-copy stage q's data (and coefficients) into ring slot q % NS. One copy per lane.
template <bool POST>
__device__ __forceinline__ void issue_stage(const Params& p, const Ctx& c, int q, int lane, unsigned long long pol) {
  const int tok0 = stage_tok0(c, q);
  const int ntok = min(TOK, c.T - tok0);
  const int slot = q % NS;
  const uint32_t st = c.sbase + SM_STAGE + slot * STAGE;
  const uint32_t full = bar_addr(c.sbase, B_FULL + slot);
  constexpr int NSRC_LD = POST ? SRC : HC;
  uint32_t bytes = uint32_t(ntok) * NSRC_LD * ROWB;
  if constexpr (POST) bytes += uint32_t(ntok) * (HC + HC * HC) * 4;
  if (p.pre_in != nullptr) bytes += uint32_t(ntok) * HC * 4;
  if (lane == 0) expect_tx(full, bytes);
  __syncwarp();
  // lanes 0..39 (two rounds): (token, source) row copies; lanes of round 2 also do the coefficient copies
  for (int i = lane; i < TOK * SRC + 3; i += 32) {
    if (i < TOK * SRC) {
      const int t = i / SRC, j = i % SRC;
      if (t >= ntok || (!POST && j == 0)) continue;
      const long tok = tok0 + t;
      const __nv_bfloat16* src = j == 0 ? p.x + tok * HID + c.rank * HS : p.res + tok * HCD + (j - 1) * HID + c.rank * HS;
      bulk_g2s(st + t * TOKB + j * ROWB, src, ROWB, full, pol);
    } else {
      const int k = i - TOK * SRC;
      if (k == 0 && POST) bulk_g2s(st + TOK * TOKB + CO_POST, p.post + long(tok0) * HC, ntok * HC * 4, full, pol);
      if (k == 1 && POST) bulk_g2s(st + TOK * TOKB + CO_COMB, p.comb + long(tok0) * HC * HC, ntok * HC * HC * 4, full, pol);
      if (k == 2 && p.pre_in != nullptr) bulk_g2s(st + TOK * TOKB + CO_PRE, p.pre_in + long(tok0) * HC, ntok * HC * 4, full, pol);
    }
  }
}

// Leader: finish token t of stage q (all 16 CTAs' partials are in this CTA's SM_RED), push rn to every CTA.
// Called by warp t (0..7) of the leader CTA, all lanes.
__device__ __forceinline__ void leader_finish(const Params& p, const Ctx& c, int q, int t, int lane, uint8_t* smem) {
  const float* red = reinterpret_cast<const float*>(smem + SM_RED);
  // lanes 0..25: column sums over the 16 CTAs (fixed order: deterministic)
  float v = 0.0f;
  if (lane < NPART) {
#pragma unroll
    for (int r = 0; r < CL; ++r) v += red[(r * TOK + t) * NPART + lane];
  }
  const int tok0 = stage_tok0(c, q);
  const long tr = tok0 + t;
  float tot[MIX + 1];
#pragma unroll
  for (int k = 0; k <= MIX; ++k) tot[k] = __shfl_sync(0xffffffffu, v, k);
  const float usq = __shfl_sync(0xffffffffu, v, MIX + 1);
  const float rn = rsqrtf(__fdiv_rn(usq, float(HID)) + p.norm_eps);
  if (lane == 0 && tr < c.T) ds41h_mhc2::finish_token(p, tot, tr);
  // rn -> every CTA's slot (lane r < 16 serves CTA r), then one arrive per CTA (8 leader warps -> count 8)
  if (lane < CL) {
    const uint32_t slot = c.sbase + SM_RN + ((q % NR) * TOK + t) * 4;
    st_cluster_f32(mapa(slot, lane), rn);
    arrive_remote(mapa(bar_addr(c.sbase, B_RN + q % NR), lane));
  }
}

template <bool POST, bool AUX>
__global__ void __launch_bounds__(THREADS, 1) mhc_boundary_v3(const Params p, const uint4* __restrict__ fnp, int nclusters) {
  extern __shared__ __align__(128) uint8_t smem[];
  Ctx c;
  {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    c.rank = int(r);
    uint32_t cid;
    asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(cid));
    c.cluster_id = int(cid);
  }
  c.nclusters = nclusters;
  c.T = p.T;
  c.nblocks = (p.T + TOK - 1) / TOK;
  c.sbase = smem_u32(smem);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int nst = c.cluster_id < c.nblocks ? (c.nblocks - c.cluster_id + c.nclusters - 1) / c.nclusters : 0;
  const unsigned long long pol_stream = ds41h_mhc::evict_first();

  if (threadIdx.x == 0) {
    for (int i = 0; i < NS; ++i) {
      mbar_init(reinterpret_cast<unsigned long long*>(smem + SM_BAR) + B_FULL + i, 1);
      mbar_init(reinterpret_cast<unsigned long long*>(smem + SM_BAR) + B_EMPTY + i, WARPS);
    }
    for (int i = 0; i < NSCR; ++i) {
      mbar_init(reinterpret_cast<unsigned long long*>(smem + SM_BAR) + B_DONE + i, WARPS);
      mbar_init(reinterpret_cast<unsigned long long*>(smem + SM_BAR) + B_SFREE + i, 1);
    }
    mbar_init(reinterpret_cast<unsigned long long*>(smem + SM_BAR) + B_RED, CL);
    for (int i = 0; i < NR; ++i) mbar_init(reinterpret_cast<unsigned long long*>(smem + SM_BAR) + B_RN + i, TOK);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // fn fragments of this warp's 32 columns (global step rank * 10 + warp of the impl-2 packing), resident in registers
  uint32_t fa[HC][2][3][4];
  {
    const uint4* src = fnp + (long(c.rank * WARPS + warp) * (HC * 3 * 2) * 32) + lane;
#pragma unroll
    for (int o = 0; o < HC; ++o) {
#pragma unroll
      for (int m = 0; m < 3; ++m) {
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 v = __ldg(src + ((o * 3 + m) * 2 + kb) * 32);
          fa[o][kb][m][0] = v.x, fa[o][kb][m][1] = v.y, fa[o][kb][m][2] = v.z, fa[o][kb][m][3] = v.w;
        }
      }
    }
  }
  // norm weight of this lane's 8 columns
  float nw[8];
  ds41h_mhc::unpack8(__ldg(reinterpret_cast<const uint4*>(p.norm_w + c.rank * HS + warp * 32 + 8 * t)), nw);
  cluster_sync_all();  // barriers initialized cluster-wide before any remote arrive
  if (warp == WARPS - 1) {
    for (int q = 0; q < min(NS, nst); ++q) issue_stage<POST>(p, c, q, lane, pol_stream);
  }

  const int col = c.rank * HS + warp * 32 + 8 * t;  // this lane's first column
  int lead_next = c.rank;                           // next stage this CTA leads (warps 0..7 finish one token each)
  for (int q = 0; q < nst + D; ++q) {
    if (q < nst) {
      const int slot = q % NS;
      const int tok0 = stage_tok0(c, q);
      const bool ok = tok0 + g < c.T;
      // ---------------------------------------------------------------- compute stage q (token g of the n-tile)
      wait_cl(bar_addr(c.sbase, B_FULL + slot), (q / NS) & 1);
      const uint8_t* stp = smem + SM_STAGE + slot * STAGE;
      const uint8_t* tp = stp + g * TOKB + (warp * 32 + 8 * t) * 2;
      const float* co = reinterpret_cast<const float*>(stp + TOK * TOKB);
      uint4 pk[HC];
      if constexpr (POST) {
        const uint4 xv = *reinterpret_cast<const uint4*>(tp);
        uint4 rv[HC];
#pragma unroll
        for (int i = 0; i < HC; ++i) rv[i] = *reinterpret_cast<const uint4*>(tp + (i + 1) * ROWB);
        const float4 po = *reinterpret_cast<const float4*>(co + (CO_POST / 4) + g * HC);
        const float pov[HC] = {po.x, po.y, po.z, po.w};
        float cmv[HC][HC];
#pragma unroll
        for (int i = 0; i < HC; ++i) {
          const float4 cm = *reinterpret_cast<const float4*>(co + (CO_COMB / 4) + g * HC * HC + 4 * i);
          cmv[i][0] = cm.x, cmv[i][1] = cm.y, cmv[i][2] = cm.z, cmv[i][3] = cm.w;
        }
        const unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
        unsigned ow[HC][4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const float2 xf = unpack_bf16(xw[w]);
          float v[HC][2];
#pragma unroll
          for (int o = 0; o < HC; ++o) v[o][0] = __fmul_rn(pov[o], xf.x), v[o][1] = __fmul_rn(pov[o], xf.y);
#pragma unroll
          for (int i = 0; i < HC; ++i) {
            const unsigned rwi = w == 0 ? rv[i].x : w == 1 ? rv[i].y : w == 2 ? rv[i].z : rv[i].w;
            const float2 rf = unpack_bf16(rwi);
#pragma unroll
            for (int o = 0; o < HC; ++o) v[o][0] = fmaf(cmv[i][o], rf.x, v[o][0]), v[o][1] = fmaf(cmv[i][o], rf.y, v[o][1]);
          }
#pragma unroll
          for (int o = 0; o < HC; ++o) ow[o][w] = pack_bf16(v[o][0], v[o][1]);
        }
#pragma unroll
        for (int o = 0; o < HC; ++o) {
          pk[o] = make_uint4(ow[o][0], ow[o][1], ow[o][2], ow[o][3]);
          ds41h_mhc::st_global16_if(p.res_out + long(min(tok0 + g, c.T - 1)) * HCD + o * HID + col, pk[o], ok);
        }
      } else {
#pragma unroll
        for (int o = 0; o < HC; ++o) pk[o] = *reinterpret_cast<const uint4*>(tp + (o + 1) * ROWB);
      }
      float4 pr = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
      if (p.pre_in != nullptr) pr = *reinterpret_cast<const float4*>(co + (CO_PRE / 4) + g * HC);
      __syncwarp();
      if (lane == 0) arrive_local(bar_addr(c.sbase, B_EMPTY + slot));  // stage data consumed
      // collapse u = bf16(sum_o pre_o * bf16(v_o)), sum r^2, aux
      const float prv[HC] = {pr.x, pr.y, pr.z, pr.w};
      float u[8], ax[8], rsq = 0.0f;
#pragma unroll
      for (int o = 0; o < HC; ++o) {
        float vr[8];
        ds41h_mhc::unpack8(pk[o], vr);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          rsq = fmaf(vr[e], vr[e], rsq);
          u[e] = o == 0 ? __fmul_rn(prv[0], vr[e]) : fmaf(prv[o], vr[e], u[e]);
          if constexpr (AUX) ax[e] = o == 0 ? vr[e] : __fadd_rn(ax[e], vr[e]);
        }
      }
      const uint4 ub = pack8_bf16(u);
      *reinterpret_cast<uint4*>(smem + SM_U + (q % NR) * USLOT + g * UTOKB + (warp * 32 + 8 * t) * 2) = ub;
      float uf[8], usq = 0.0f;
      ds41h_mhc::unpack8(ub, uf);
#pragma unroll
      for (int e = 0; e < 8; ++e) usq = fmaf(uf[e], uf[e], usq);
      if constexpr (AUX) {
#pragma unroll
        for (int e = 0; e < 8; ++e) ax[e] = __fmul_rn(ax[e], 0.25f);
        ds41h_mhc::st_global16_if(p.aux + long(min(tok0 + g, c.T - 1)) * HID + col, pack8_bf16(ax), ok);
      }
      // GEMV partials: mixes^T[48 rows][8 tokens] += fn48 (registers) x r^T (this lane's words)
      float acc[3][4];
#pragma unroll
      for (int m = 0; m < 3; ++m) acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0.0f;
#pragma unroll
      for (int o = 0; o < HC; ++o) {
#pragma unroll
        for (int m = 0; m < 3; ++m) {
          hmma_bf16(acc[m], fa[o][0][m][0], fa[o][0][m][1], fa[o][0][m][2], fa[o][0][m][3], pk[o].x, pk[o].y);
          hmma_bf16(acc[m], fa[o][1][m][0], fa[o][1][m][1], fa[o][1][m][2], fa[o][1][m][3], pk[o].z, pk[o].w);
        }
      }
      rsq += __shfl_xor_sync(0xffffffffu, rsq, 1);
      rsq += __shfl_xor_sync(0xffffffffu, rsq, 2);
      usq += __shfl_xor_sync(0xffffffffu, usq, 1);
      usq += __shfl_xor_sync(0xffffffffu, usq, 2);
      // partials -> scratch slot q % NSCR (free once the service warp has read stage q - NSCR)
      const int ss = q % NSCR;
      if (q >= NSCR) wait_cl(bar_addr(c.sbase, B_SFREE + ss), ((q / NSCR) - 1) & 1);
      float* scr = reinterpret_cast<float*>(smem + SM_SCR) + (ss * WARPS + warp) * TOK * NPART;
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int tok = 2 * t + cc;  // C column of this lane: tokens 2t, 2t + 1 of the n-tile
        scr[tok * NPART + g] = acc[0][cc] + acc[1][2 + cc];
        scr[tok * NPART + g + 8] = acc[0][2 + cc] + acc[2][cc];
        scr[tok * NPART + g + 16] = acc[1][cc] + acc[2][2 + cc];
      }
      if (t == 0) {
        scr[g * NPART + MIX] = rsq;
        scr[g * NPART + MIX + 1] = usq;
      }
      __syncwarp();
      if (lane == 0) arrive_local(bar_addr(c.sbase, B_DONE + ss));

      // ---------------------------------------------------------------- service warp: reduce, send, produce
      if (warp == WARPS - 1) {
        // all 10 warps wrote stage q's partials (they only wait on earlier-stage events before arriving)
        wait_cl(bar_addr(c.sbase, B_DONE + ss), (q / NSCR) & 1);
        const float* s0 = reinterpret_cast<const float*>(smem + SM_SCR) + ss * WARPS * TOK * NPART;
        const int leader = q % CL;
        for (int i = lane; i < TOK * NPART; i += 32) {
          float a = 0.0f;
#pragma unroll
          for (int w = 0; w < WARPS; ++w) a += s0[w * TOK * NPART + i];
          st_cluster_f32(mapa(c.sbase + SM_RED + (c.rank * TOK * NPART + i) * 4, leader), a);
        }
        __syncwarp();
        if (lane == 0) {
          arrive_local(bar_addr(c.sbase, B_SFREE + ss));
          arrive_remote(mapa(bar_addr(c.sbase, B_RED), leader));  // release.cluster: orders the partial stores above
        }
        // refill: stage q's ring slot once all warps consumed it
        if (q + NS < nst) {
          wait_cl(bar_addr(c.sbase, B_EMPTY + slot), (q / NS) & 1);
          issue_stage<POST>(p, c, q + NS, lane, pol_stream);
        }
      }
      // ---------------------------------------------------------------- leader duty (non-blocking poll)
      if (warp < TOK && lead_next <= q && lead_next < nst && try_wait_cl(bar_addr(c.sbase, B_RED), (lead_next / CL) & 1)) {
        leader_finish(p, c, lead_next, warp, lane, smem);
        lead_next += CL;
      }
    }
    // ---------------------------------------------------------------- normalize stage q - D
    const int qn = q - D;
    if (qn >= 0 && qn < nst) {
      if (warp < TOK && lead_next == qn) {  // this CTA leads qn and has not finished it: block (all partials of qn
        wait_cl(bar_addr(c.sbase, B_RED), (qn / CL) & 1);  // were sent at the end of each CTA's stage qn < q)
        leader_finish(p, c, qn, warp, lane, smem);
        lead_next += CL;
      }
      wait_cl(bar_addr(c.sbase, B_RN + qn % NR), (qn / NR) & 1);
      const int tok0 = stage_tok0(c, qn);
      const float rn = *reinterpret_cast<const float*>(smem + SM_RN + ((qn % NR) * TOK + g) * 4);
      float uf[8], o8[8];
      ds41h_mhc::unpack8(*reinterpret_cast<const uint4*>(smem + SM_U + (qn % NR) * USLOT + g * UTOKB + (warp * 32 + 8 * t) * 2), uf);
#pragma unroll
      for (int e = 0; e < 8; ++e) o8[e] = __fmul_rn(__fmul_rn(uf[e], rn), nw[e]);
      ds41h_mhc::st_global16_if(p.layer_in + long(min(tok0 + g, c.T - 1)) * HID + col, pack8_bf16(o8), tok0 + g < c.T);
    }
  }
  cluster_sync_all();  // nobody exits while others may still write into its shared memory
}

template <bool POST, bool AUX>
void launch(const Params& p, const void* fnp) {
  auto kernel = mhc_boundary_v3<POST, AUX>;
  static int max_clusters = -1;
  static bool init = false;
  if (!init) {
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess);
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CL * 8);
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = SMEM;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CL, attr[0].val.clusterDim.y = 1, attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int n = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveClusters(&n, (void*)kernel, &cfg) == cudaSuccess, "cluster occupancy query failed");
    max_clusters = n;
    init = true;
  }
  TORCH_CHECK(max_clusters >= 1, "ds41h mhc v3: no 16-CTA cluster fits on this GPU");
  const int nblocks = (p.T + TOK - 1) / TOK;
  const int ncl = std::min(max_clusters, nblocks);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CL * ncl);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CL, attr[0].val.clusterDim.y = 1, attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, p, reinterpret_cast<const uint4*>(fnp), ncl);
  TORCH_CHECK(err == cudaSuccess, "ds41h mhc_boundary v3 launch failed: ", cudaGetErrorString(err));
}

}  // namespace ds41h_mhc3

// =====================================================================================================================
// v4: v3's streaming core without any cross-CTA synchronization, plus a small finishing kernel.
//   kernel A (persistent, 1 CTA per SM, no clusters): CTA = (column slice c of 16 x 320 columns, token group gi);
//     the slice's packed fn lives in registers (96 per lane, 10 warps x 32 columns); 8-token stages are bulk-copied as
//     contiguous 640 B row segments (3-stage full/empty mbarrier ring, producer = warp 9). Per stage: post map ->
//     residual_out, collapse u (bf16, UNNORMALIZED) -> layer_input, sum r^2 / sum u^2 / the 24-mix GEMV partials
//     (fn registers x the lane's post-mapped words) reduced over the 10 warps -> partials[token][slice][26] (fp32,
//     fixed layout: deterministic). No CTA ever waits for another one.
//   kernel B (one warp per token): sums the 16 slice partials in fixed order, rms / sigmoids / Sinkhorn (stock op
//     order, ds41h_mhc2::finish_token), and scales its layer_input row in place: u * rsqrt(mean u^2 + eps) * w.
// Extra traffic vs v3: 20 KB per token (u written + read + rewritten) + 1.7 KB partials, for zero coupling.
namespace ds41h_mhc4 {
using ds41h_mhc::Params;
using ds41h_mhc::HC;
using ds41h_mhc::HID;
using ds41h_mhc::MIX;
using ds41h_mhc::HCD;
using ds41h_mhc3::TOK;
using ds41h_mhc3::SRC;
using ds41h_mhc3::ROWB;
using ds41h_mhc3::TOKB;
using ds41h_mhc3::CO_POST;
using ds41h_mhc3::CO_COMB;
using ds41h_mhc3::CO_PRE;
using ds41h_mhc3::CO_BYTES;
using ds41h_mhc3::STAGE;
using ds41h_mhc3::NPART;

constexpr int NSL = 16, HS = HID / NSL, WARPS = HS / 32, THREADS = WARPS * 32;  // 320 columns, 10 warps
#ifndef MHC4_NS
#define MHC4_NS 4
#endif
constexpr int NS = MHC4_NS, NSCR = 2;
constexpr int SM_STAGE = 0;
constexpr int SM_SCR = SM_STAGE + NS * STAGE;                   // [NSCR][WARPS][TOK][NPART]
constexpr int SM_BAR = SM_SCR + NSCR * WARPS * TOK * NPART * 4;
constexpr int B_FULL = 0, B_EMPTY = NS, B_DONE = 2 * NS, B_SFREE = 2 * NS + NSCR, NBAR = 2 * NS + 2 * NSCR;
constexpr int SMEM = SM_BAR + NBAR * 8;
static_assert(SM_SCR % 16 == 0 && SM_BAR % 8 == 0 && SMEM <= 227 * 1024);

struct Ctx {
  int slice, group, ngroups, T;
  uint32_t sbase;
};
__device__ __forceinline__ int stage_tok0(const Ctx& c, int q) { return (c.group + q * c.ngroups) * TOK; }
__device__ __forceinline__ uint32_t bar_addr(uint32_t sbase, int i) { return sbase + SM_BAR + i * 8; }

template <bool POST>
__device__ __forceinline__ void issue_stage(const Params& p, const Ctx& c, int q, int lane, unsigned long long pol) {
  const int tok0 = stage_tok0(c, q);
  const int ntok = min(TOK, c.T - tok0);
  const int slot = q % NS;
  const uint32_t st = c.sbase + SM_STAGE + slot * STAGE;
  const uint32_t full = bar_addr(c.sbase, B_FULL + slot);
  constexpr int NSRC_LD = POST ? SRC : HC;
  uint32_t bytes = uint32_t(ntok) * NSRC_LD * ROWB;
  if constexpr (POST) bytes += uint32_t(ntok) * (HC + HC * HC) * 4;
  if (p.pre_in != nullptr) bytes += uint32_t(ntok) * HC * 4;
  if (lane == 0) ds41h_mhc3::expect_tx(full, bytes);
  __syncwarp();
  for (int i = lane; i < TOK * SRC + 3; i += 32) {
    if (i < TOK * SRC) {
      const int t = i / SRC, j = i % SRC;
      if (t >= ntok || (!POST && j == 0)) continue;
      const long tok = tok0 + t;
      const __nv_bfloat16* src = j == 0 ? p.x + tok * HID + c.slice * HS : p.res + tok * HCD + (j - 1) * HID + c.slice * HS;
      ds41h_mhc3::bulk_g2s(st + t * TOKB + j * ROWB, src, ROWB, full, pol);
    } else {
      const int k = i - TOK * SRC;
      if (k == 0 && POST) ds41h_mhc3::bulk_g2s(st + TOK * TOKB + CO_POST, p.post + long(tok0) * HC, ntok * HC * 4, full, pol);
      if (k == 1 && POST) ds41h_mhc3::bulk_g2s(st + TOK * TOKB + CO_COMB, p.comb + long(tok0) * HC * HC, ntok * HC * HC * 4, full, pol);
      if (k == 2 && p.pre_in != nullptr) ds41h_mhc3::bulk_g2s(st + TOK * TOKB + CO_PRE, p.pre_in + long(tok0) * HC, ntok * HC * 4, full, pol);
    }
  }
}

template <bool POST, bool AUX>
__global__ void __launch_bounds__(THREADS, 1) mhc_main_v4(const Params p, const uint4* __restrict__ fnp, float* __restrict__ part,
                                                          int ngroups) {
  using ds41h_mhc3::wait_cl;
  using ds41h_mhc3::arrive_local;
  extern __shared__ __align__(128) uint8_t smem[];
  Ctx c;
  c.slice = blockIdx.x % NSL;
  c.group = blockIdx.x / NSL;
  c.ngroups = ngroups;
  c.T = p.T;
  c.sbase = smem_u32(smem);
  const int nblocks = (p.T + TOK - 1) / TOK;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int nst = c.group < nblocks ? (nblocks - c.group + ngroups - 1) / ngroups : 0;
  const unsigned long long pol_stream = ds41h_mhc::evict_first();

  if (threadIdx.x == 0) {
    auto* bars = reinterpret_cast<unsigned long long*>(smem + SM_BAR);
    for (int i = 0; i < NS; ++i) mbar_init(bars + B_FULL + i, 1), mbar_init(bars + B_EMPTY + i, WARPS);
    for (int i = 0; i < NSCR; ++i) mbar_init(bars + B_DONE + i, WARPS), mbar_init(bars + B_SFREE + i, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  uint32_t fa[HC][2][3][4];
  {
    const uint4* src = fnp + (long(c.slice * WARPS + warp) * (HC * 3 * 2) * 32) + lane;
#pragma unroll
    for (int o = 0; o < HC; ++o) {
#pragma unroll
      for (int m = 0; m < 3; ++m) {
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const uint4 v = __ldg(src + ((o * 3 + m) * 2 + kb) * 32);
          fa[o][kb][m][0] = v.x, fa[o][kb][m][1] = v.y, fa[o][kb][m][2] = v.z, fa[o][kb][m][3] = v.w;
        }
      }
    }
  }
  __syncthreads();
  if (warp == WARPS - 1) {
    for (int q = 0; q < min(NS, nst); ++q) issue_stage<POST>(p, c, q, lane, pol_stream);
  }
  const int col = c.slice * HS + warp * 32 + 8 * t;
  for (int q = 0; q < nst; ++q) {
    const int slot = q % NS;
    const int tok0 = stage_tok0(c, q);
    const bool ok = tok0 + g < c.T;
    const long row = min(tok0 + g, c.T - 1);
    wait_cl(bar_addr(c.sbase, B_FULL + slot), (q / NS) & 1);
    const uint8_t* stp = smem + SM_STAGE + slot * STAGE;
    const uint8_t* tp = stp + g * TOKB + (warp * 32 + 8 * t) * 2;
    const float* co = reinterpret_cast<const float*>(stp + TOK * TOKB);
    uint4 pk[HC];
    if constexpr (POST) {
      const uint4 xv = *reinterpret_cast<const uint4*>(tp);
      uint4 rv[HC];
#pragma unroll
      for (int i = 0; i < HC; ++i) rv[i] = *reinterpret_cast<const uint4*>(tp + (i + 1) * ROWB);
      const float4 po = *reinterpret_cast<const float4*>(co + (CO_POST / 4) + g * HC);
      const float pov[HC] = {po.x, po.y, po.z, po.w};
      float cmv[HC][HC];
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        const float4 cm = *reinterpret_cast<const float4*>(co + (CO_COMB / 4) + g * HC * HC + 4 * i);
        cmv[i][0] = cm.x, cmv[i][1] = cm.y, cmv[i][2] = cm.z, cmv[i][3] = cm.w;
      }
      const unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
      unsigned ow[HC][4];
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const float2 xf = unpack_bf16(xw[w]);
        float v[HC][2];
#pragma unroll
        for (int o = 0; o < HC; ++o) v[o][0] = __fmul_rn(pov[o], xf.x), v[o][1] = __fmul_rn(pov[o], xf.y);
#pragma unroll
        for (int i = 0; i < HC; ++i) {
          const unsigned rwi = w == 0 ? rv[i].x : w == 1 ? rv[i].y : w == 2 ? rv[i].z : rv[i].w;
          const float2 rf = unpack_bf16(rwi);
#pragma unroll
          for (int o = 0; o < HC; ++o) v[o][0] = fmaf(cmv[i][o], rf.x, v[o][0]), v[o][1] = fmaf(cmv[i][o], rf.y, v[o][1]);
        }
#pragma unroll
        for (int o = 0; o < HC; ++o) ow[o][w] = pack_bf16(v[o][0], v[o][1]);
      }
#pragma unroll
      for (int o = 0; o < HC; ++o) {
        pk[o] = make_uint4(ow[o][0], ow[o][1], ow[o][2], ow[o][3]);
        ds41h_mhc::st_global16_if(p.res_out + row * HCD + o * HID + col, pk[o], ok);
      }
    } else {
#pragma unroll
      for (int o = 0; o < HC; ++o) pk[o] = *reinterpret_cast<const uint4*>(tp + (o + 1) * ROWB);
    }
    float4 pr = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    if (p.pre_in != nullptr) pr = *reinterpret_cast<const float4*>(co + (CO_PRE / 4) + g * HC);
    __syncwarp();
    if (lane == 0) arrive_local(bar_addr(c.sbase, B_EMPTY + slot));
    const float prv[HC] = {pr.x, pr.y, pr.z, pr.w};
    float u[8], ax[8], rsq = 0.0f;
#pragma unroll
    for (int o = 0; o < HC; ++o) {
      float vr[8];
      ds41h_mhc::unpack8(pk[o], vr);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        rsq = fmaf(vr[e], vr[e], rsq);
        u[e] = o == 0 ? __fmul_rn(prv[0], vr[e]) : fmaf(prv[o], vr[e], u[e]);
        if constexpr (AUX) ax[e] = o == 0 ? vr[e] : __fadd_rn(ax[e], vr[e]);
      }
    }
    const uint4 ub = pack8_bf16(u);
    ds41h_mhc::st_global16_if(p.layer_in + row * HID + col, ub, ok);  // unnormalized; kernel B scales it in place
    float uf[8], usq = 0.0f;
    ds41h_mhc::unpack8(ub, uf);
#pragma unroll
    for (int e = 0; e < 8; ++e) usq = fmaf(uf[e], uf[e], usq);
    if constexpr (AUX) {
#pragma unroll
      for (int e = 0; e < 8; ++e) ax[e] = __fmul_rn(ax[e], 0.25f);
      ds41h_mhc::st_global16_if(p.aux + row * HID + col, pack8_bf16(ax), ok);
    }
    float acc[3][4];
#pragma unroll
    for (int m = 0; m < 3; ++m) acc[m][0] = acc[m][1] = acc[m][2] = acc[m][3] = 0.0f;
#pragma unroll
    for (int o = 0; o < HC; ++o) {
#pragma unroll
      for (int m = 0; m < 3; ++m) {
        hmma_bf16(acc[m], fa[o][0][m][0], fa[o][0][m][1], fa[o][0][m][2], fa[o][0][m][3], pk[o].x, pk[o].y);
        hmma_bf16(acc[m], fa[o][1][m][0], fa[o][1][m][1], fa[o][1][m][2], fa[o][1][m][3], pk[o].z, pk[o].w);
      }
    }
    rsq += __shfl_xor_sync(0xffffffffu, rsq, 1);
    rsq += __shfl_xor_sync(0xffffffffu, rsq, 2);
    usq += __shfl_xor_sync(0xffffffffu, usq, 1);
    usq += __shfl_xor_sync(0xffffffffu, usq, 2);
    const int ss = q % NSCR;
    if (q >= NSCR) wait_cl(bar_addr(c.sbase, B_SFREE + ss), ((q / NSCR) - 1) & 1);
    float* scr = reinterpret_cast<float*>(smem + SM_SCR) + (ss * WARPS + warp) * TOK * NPART;
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int tok = 2 * t + cc;
      scr[tok * NPART + g] = acc[0][cc] + acc[1][2 + cc];
      scr[tok * NPART + g + 8] = acc[0][2 + cc] + acc[2][cc];
      scr[tok * NPART + g + 16] = acc[1][cc] + acc[2][2 + cc];
    }
    if (t == 0) {
      scr[g * NPART + MIX] = rsq;
      scr[g * NPART + MIX + 1] = usq;
    }
    __syncwarp();
    if (lane == 0) arrive_local(bar_addr(c.sbase, B_DONE + ss));
    if (warp == WARPS - 1) {  // service: reduce the 10 warps' partials -> global, refill the ring
      wait_cl(bar_addr(c.sbase, B_DONE + ss), (q / NSCR) & 1);
      const float* s0 = reinterpret_cast<const float*>(smem + SM_SCR) + ss * WARPS * TOK * NPART;
      const int ntok = min(TOK, c.T - tok0);
      for (int i = lane; i < TOK * NPART; i += 32) {
        float a = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) a += s0[w * TOK * NPART + i];
        const int tk = i / NPART, k = i % NPART;
        if (tk < ntok) part[(long(tok0 + tk) * NSL + c.slice) * NPART + k] = a;
      }
      __syncwarp();
      if (lane == 0) arrive_local(bar_addr(c.sbase, B_SFREE + ss));
      if (q + NS < nst) {
        wait_cl(bar_addr(c.sbase, B_EMPTY + slot), (q / NS) & 1);
        issue_stage<POST>(p, c, q + NS, lane, pol_stream);
      }
    }
  }
}

// Kernel B: one warp per token.
__global__ void __launch_bounds__(256) mhc_finish_v4(const Params p, const float* __restrict__ part) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long tr = long(blockIdx.x) * 8 + warp;
  if (tr >= p.T) return;
  float v = 0.0f;
  if (lane < NPART) {
    const float* pp = part + tr * NSL * NPART + lane;
#pragma unroll
    for (int s = 0; s < NSL; ++s) v += pp[s * NPART];
  }
  float tot[MIX + 1];
#pragma unroll
  for (int k = 0; k <= MIX; ++k) tot[k] = __shfl_sync(0xffffffffu, v, k);
  const float usq = __shfl_sync(0xffffffffu, v, MIX + 1);
  const float rn = rsqrtf(__fdiv_rn(usq, float(HID)) + p.norm_eps);
  if (lane == 0) ds41h_mhc2::finish_token(p, tot, tr);
  uint4* row = reinterpret_cast<uint4*>(p.layer_in + tr * HID);
  const uint4* w8 = reinterpret_cast<const uint4*>(p.norm_w);
#pragma unroll 4
  for (int i = lane; i < HID / 8; i += 32) {
    float uf[8], wf[8], o[8];
    ds41h_mhc::unpack8(row[i], uf);
    ds41h_mhc::unpack8(__ldg(w8 + i), wf);
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = __fmul_rn(__fmul_rn(uf[e], rn), wf[e]);
    row[i] = pack8_bf16(o);
  }
}

template <bool POST, bool AUX>
void launch(const Params& p, const void* fnp, float* part) {
  auto kernel = mhc_main_v4<POST, AUX>;
  static int nsm = 0;
  if (nsm == 0) {
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess);
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  }
  const int nblocks = (p.T + TOK - 1) / TOK;
  const int ngroups = std::max(1, std::min(nsm / NSL, nblocks));
  auto stream = c10::cuda::getCurrentCUDAStream();
  kernel<<<dim3(NSL * ngroups), dim3(THREADS), SMEM, stream>>>(p, reinterpret_cast<const uint4*>(fnp), part, ngroups);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "ds41h mhc v4 main launch failed: ", cudaGetErrorString(err));
  mhc_finish_v4<<<dim3((p.T + 7) / 8), dim3(256), 0, stream>>>(p, part);
  err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "ds41h mhc v4 finish launch failed: ", cudaGetErrorString(err));
}

}  // namespace ds41h_mhc4

using namespace ds41h_mhc;

static void check_bf16(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous(), name, " must be a contiguous CUDA bf16 tensor");
}
static void check_f32(const torch::Tensor& t, const char* name, long numel) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous() && t.numel() == numel, name,
              " must be a contiguous CUDA float32 tensor with ", numel, " elements");
}

// One mHC boundary. x / post / comb given -> POST mode (returns the post-mapped residual), else PRE mode (residual
// used as is and returned unchanged). Returns {residual_out, post [T,4], comb [T,4,4], layer_input [T,5120],
// pre [T,4], aux [T,5120] or [0,5120]}.
std::vector<torch::Tensor> mhc_boundary(const c10::optional<torch::Tensor>& x, const torch::Tensor& residual,
                                        const c10::optional<torch::Tensor>& post, const c10::optional<torch::Tensor>& comb,
                                        const c10::optional<torch::Tensor>& pre_in, const torch::Tensor& fn_packed,
                                        const torch::Tensor& hc_scale, const torch::Tensor& hc_base, const torch::Tensor& norm_w,
                                        double rms_eps, double hc_pre_eps, double sk_eps, double post_mult, int64_t iters,
                                        double norm_eps, bool aux, int64_t impl) {
  check_bf16(residual, "residual");
  TORCH_CHECK(residual.dim() == 3 && residual.size(1) == HC && residual.size(2) == HID, "residual must be [T, 4, 5120]");
  const long T = residual.size(0);
  TORCH_CHECK(T < (1L << 31) / HCD, "too many tokens");  // 32-bit element offsets in the v2 kernel
  const bool mode_post = x.has_value();
  TORCH_CHECK(mode_post == post.has_value() && mode_post == comb.has_value(), "x, post and comb go together");
  const c10::cuda::CUDAGuard guard(residual.device());
  TORCH_CHECK(fn_packed.is_cuda() && fn_packed.is_contiguous() && fn_packed.nbytes() == size_t(GSTEPS) * FN_STEP,
              "fn_packed must be the packed hc fn (", GSTEPS * FN_STEP, " bytes)");
  check_f32(hc_scale, "hc_scale", 3);
  check_f32(hc_base, "hc_base", MIX);
  check_bf16(norm_w, "norm_w");
  TORCH_CHECK(norm_w.numel() == HID, "norm_w must have 5120 elements");
  Params p{};
  if (mode_post) {
    check_bf16(*x, "x");
    TORCH_CHECK(x->dim() == 2 && x->size(0) == T && x->size(1) == HID, "x must be [T, 5120]");
    check_f32(*post, "post", T * HC);
    check_f32(*comb, "comb", T * HC * HC);
    p.x = reinterpret_cast<const __nv_bfloat16*>(x->data_ptr());
    p.post = post->data_ptr<float>();
    p.comb = comb->data_ptr<float>();
  }
  if (pre_in.has_value()) {
    check_f32(*pre_in, "pre_in", T * HC);
    p.pre_in = pre_in->data_ptr<float>();
  }
  auto opts = residual.options();
  torch::Tensor res_out = mode_post ? torch::empty_like(residual) : residual;
  torch::Tensor post_out = torch::empty({T, HC}, opts.dtype(torch::kFloat32));
  torch::Tensor comb_out = torch::empty({T, HC, HC}, opts.dtype(torch::kFloat32));
  torch::Tensor pre_out = torch::empty({T, HC}, opts.dtype(torch::kFloat32));
  torch::Tensor layer_in = torch::empty({T, HID}, opts);
  torch::Tensor aux_out = torch::empty({aux ? T : 0, HID}, opts);
  if (T == 0) return {res_out, post_out, comb_out, layer_in, pre_out, aux_out};
  p.res = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
  p.fnp = reinterpret_cast<const uint8_t*>(fn_packed.data_ptr());
  p.hc_scale = hc_scale.data_ptr<float>();
  p.hc_base = hc_base.data_ptr<float>();
  p.norm_w = reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr());
  p.res_out = reinterpret_cast<__nv_bfloat16*>(res_out.data_ptr());
  p.layer_in = reinterpret_cast<__nv_bfloat16*>(layer_in.data_ptr());
  p.aux = aux ? reinterpret_cast<__nv_bfloat16*>(aux_out.data_ptr()) : nullptr;
  p.post_out = post_out.data_ptr<float>();
  p.comb_out = comb_out.data_ptr<float>();
  p.pre_out = pre_out.data_ptr<float>();
  p.T = int(T);
  p.iters = int(iters);
  p.rms_eps = float(rms_eps);
  p.hc_pre_eps = float(hc_pre_eps);
  p.sk_eps = float(sk_eps);
  p.post_mult = float(post_mult);
  p.norm_eps = float(norm_eps);
  TORCH_CHECK(p.iters >= 1, "sinkhorn iterations must be >= 1");
  TORCH_CHECK(impl >= 1 && impl <= 4, "impl must be 1..4");
  if (impl == 4) {  // fn in the impl-2 packing; slice partials in a scratch tensor
    namespace v4 = ds41h_mhc4;
    torch::Tensor part = torch::empty({T, v4::NSL, ds41h_mhc3::NPART}, opts.dtype(torch::kFloat32));
    if (mode_post) {
      if (aux) v4::launch<true, true>(p, fn_packed.data_ptr(), part.data_ptr<float>());
      else v4::launch<true, false>(p, fn_packed.data_ptr(), part.data_ptr<float>());
    } else {
      if (aux) v4::launch<false, true>(p, fn_packed.data_ptr(), part.data_ptr<float>());
      else v4::launch<false, false>(p, fn_packed.data_ptr(), part.data_ptr<float>());
    }
    return {res_out, post_out, comb_out, layer_in, pre_out, aux_out};
  }
  if (impl == 3) {  // fn in the impl-2 packing
    namespace v3 = ds41h_mhc3;
    if (mode_post) {
      if (aux) v3::launch<true, true>(p, fn_packed.data_ptr()); else v3::launch<true, false>(p, fn_packed.data_ptr());
    } else {
      if (aux) v3::launch<false, true>(p, fn_packed.data_ptr()); else v3::launch<false, false>(p, fn_packed.data_ptr());
    }
    return {res_out, post_out, comb_out, layer_in, pre_out, aux_out};
  }
  if (impl == 2) {
    namespace v2 = ds41h_mhc2;
    const int tiles = int((T + v2::BM - 1) / v2::BM);
    if (mode_post) {
      if (aux) v2::launch<true, true>(p, tiles); else v2::launch<true, false>(p, tiles);
    } else {
      if (aux) v2::launch<false, true>(p, tiles); else v2::launch<false, false>(p, tiles);
    }
    return {res_out, post_out, comb_out, layer_in, pre_out, aux_out};
  }
  const int tiles = int((T + BM - 1) / BM);
  if (mode_post) {
    if (aux) launch<true, true>(p, tiles); else launch<true, false>(p, tiles);
  } else {
    if (aux) launch<false, true>(p, tiles); else launch<false, false>(p, tiles);
  }
  return {res_out, post_out, comb_out, layer_in, pre_out, aux_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mhc_boundary", &mhc_boundary, "fused mHC post + pre + RMSNorm for prefill (sm_90a)", pybind11::arg("x"),
        pybind11::arg("residual"), pybind11::arg("post"), pybind11::arg("comb"), pybind11::arg("pre_in"), pybind11::arg("fn_packed"),
        pybind11::arg("hc_scale"), pybind11::arg("hc_base"), pybind11::arg("norm_w"), pybind11::arg("rms_eps"),
        pybind11::arg("hc_pre_eps"), pybind11::arg("sk_eps"), pybind11::arg("post_mult"), pybind11::arg("iters"),
        pybind11::arg("norm_eps"), pybind11::arg("aux"), pybind11::arg("impl") = 2);
  m.attr("BM") = BM;
  m.attr("CLUSTER") = CL;
  m.attr("FN_PACKED_BYTES") = GSTEPS * FN_STEP;
}
