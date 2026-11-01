// ds41h dmhc: the DSpark DRAFTER layers' FFN-side mHC (stock mhc_fused_post_pre_delayed_tilelang = mhc_fused_tilelang +
// mhc_pre_big_fuse_with_norm_tilelang, 2 launches, ~13 us in situ at 350 W) as ONE launch for <= 8 tokens.
// Drafter-only (approximations allowed); the math follows the stock kernels step by step (vLLM 0.32.0
// model_executor/kernels/mhc/tilelang_kernels.py), only the fp32 summation order of the 24 pre-norm mixes / the squared
// sums differs (40 hidden slices summed in slice order instead of the stock split-k tiles):
//   post:      new_r[j] = post[j] * x[h] + sum_k comb[k][j] * res[k][h]          (fp32; residual_out = bf16(new_r))
//   pre-norm:  mix[n] = sum_{j,h} fn[n][j][h] * new_r[j][h],  sqr = sum new_r^2   (unrounded new_r, as stock)
//   coeffs:    rms = rsqrt(sqr / (4 H) + rms_eps); m = mix * rms
//              pre_out[j] = sigmoid(m[j] s0 + b[j]) + pre_eps; post_out[j] = sigmoid(m[4+j] s1 + b[4+j]) * post_mult
//              comb = Sinkhorn(m[8 + 4j + k] s2 + b[...]) (row softmax + eps, column norm, (iters - 1) x row / column)
//   collapse:  c[h] = bf16(sum_j pre_in[j] * bf16(new_r[j][h]))   (delayed: the CARRIED pre-mix, i.e. the attention's)
//   norm:      layer_input[h] = bf16(c[h] * rsqrt(sum c^2 / H + norm_eps) * w[h])
// One kernel: CTA c owns hidden slice [128 c, 128 c + 128) for all tokens (warp t = token t, lane = 4 hidden values); per-CTA
// partials (24 mixes, sqr, sum c^2 per token) go to a scratch buffer, the last CTA (atomic ticket) reduces them in CTA order,
// computes the coefficients and the per-token norm factor, then releases every CTA (epoch flag; all 40 CTAs are co-resident),
// which normalizes its own slice from registers. Graph safe: counters self-reset, the epoch lives in device memory.
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "common.cuh"
#include "host.cuh"

namespace dmhc {
constexpr int H = 5120, HC = 4, NMIX = HC * (HC + 2), MAXM = 8, T = 256, SL = 128, G = H / SL, NP = NMIX + 2;
static_assert(SL == 32 * 4, "lane owns 4 hidden values");

struct Args {
  const __nv_bfloat16* x;      // [m, H]
  const __nv_bfloat16* res;    // [m, HC, H]
  const float* post;           // [m, HC] (stock [m, HC, 1])
  const float* comb;           // [m, HC, HC]
  const float* fn;             // [NMIX, HC, H] fp32
  const float* hc_scale;       // [3]
  const float* hc_base;        // [NMIX]
  const float* pre_in;         // [m, HC] carried pre-mix
  const __nv_bfloat16* norm_w; // [H]
  __nv_bfloat16* res_out;      // [m, HC, H]
  float* post_out;             // [m, HC]
  float* comb_out;             // [m, HC, HC]
  __nv_bfloat16* layer_input;  // [m, H]
  float* pre_out;              // [m, HC]
  float* part;                 // [G][MAXM][NP] scratch
  float* nrm;                  // [MAXM] per-token norm factor (scratch)
  unsigned* ctr;               // [0] arrival ticket, [1] epoch, [2] router ticket
  int m, sk_iters;
  float rms_eps, pre_eps, sk_eps, post_mult, norm_eps;
  // optional fused router (ROUTE): gate logits of layer_input (fp32), dsv4 sqrtsoftplus + bias top-k (drouter semantics)
  const __nv_bfloat16* gate_w;  // [E, H] bf16
  const float* bias;            // [E]
  const float* bias_vl;         // [E] or null (image-sentinel tokens)
  const int* ids_in;            // [m] int32 token ids or null
  long image_lo;
  float* rpart;                 // [G][MAXM][E] router partial logits (scratch)
  int* out_ids;                 // [m, k]
  float* out_w;                 // [m, k]
  float rscale;
  int renorm, k, E;
};
constexpr int E_MAX = 128;

__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

constexpr int SMEM = NMIX * HC * SL * 4;  // 48 KB fn slice, shared by the 8 token warps (dynamic, opt-in)
constexpr int SMEM_R = SMEM + E_MAX * SL * 2 + MAXM * SL * 2;  // + gate slice [E][SL] bf16 + normalized x slice [8][SL] bf16

__device__ __noinline__ void coeffs(const Args& a, const float* red, int tt) {
      const float rms = rsqrtf(red[NMIX] / float(HC * H) + a.rms_eps);
      float mm[NMIX];
#pragma unroll
      for (int n = 0; n < NMIX; ++n) mm[n] = red[n] * rms;
      const float s0 = a.hc_scale[0], s1 = a.hc_scale[1], s2 = a.hc_scale[2];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        a.pre_out[tt * HC + j] = sigm(mm[j] * s0 + a.hc_base[j]) + a.pre_eps;
        a.post_out[tt * HC + j] = sigm(mm[j + HC] * s1 + a.hc_base[j + HC]) * a.post_mult;
      }
      float cmx[HC][HC], rs[HC], cs[HC];
#pragma unroll
      for (int j = 0; j < HC; ++j)
#pragma unroll
        for (int k = 0; k < HC; ++k) cmx[j][k] = mm[j * HC + k + 2 * HC] * s2 + a.hc_base[j * HC + k + 2 * HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float mx = cmx[j][0];
#pragma unroll
        for (int k = 1; k < HC; ++k) mx = fmaxf(mx, cmx[j][k]);
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < HC; ++k) cmx[j][k] = expf(cmx[j][k] - mx), s += cmx[j][k];
#pragma unroll
        for (int k = 0; k < HC; ++k) cmx[j][k] = cmx[j][k] / s + a.sk_eps;
      }
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        cs[k] = 0.f;
#pragma unroll
        for (int j = 0; j < HC; ++j) cs[k] += cmx[j][k];
      }
#pragma unroll
      for (int j = 0; j < HC; ++j)
#pragma unroll
        for (int k = 0; k < HC; ++k) cmx[j][k] = cmx[j][k] / (cs[k] + a.sk_eps);
      for (int it = 1; it < a.sk_iters; ++it) {
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          rs[j] = 0.f;
#pragma unroll
          for (int k = 0; k < HC; ++k) rs[j] += cmx[j][k];
        }
#pragma unroll
        for (int j = 0; j < HC; ++j)
#pragma unroll
          for (int k = 0; k < HC; ++k) cmx[j][k] = cmx[j][k] / (rs[j] + a.sk_eps);
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          cs[k] = 0.f;
#pragma unroll
          for (int j = 0; j < HC; ++j) cs[k] += cmx[j][k];
        }
#pragma unroll
        for (int j = 0; j < HC; ++j)
#pragma unroll
          for (int k = 0; k < HC; ++k) cmx[j][k] = cmx[j][k] / (cs[k] + a.sk_eps);
      }
#pragma unroll
      for (int j = 0; j < HC; ++j)
#pragma unroll
        for (int k = 0; k < HC; ++k) a.comb_out[(tt * HC + j) * HC + k] = cmx[j][k];
      if (a.nrm != nullptr) a.nrm[tt] = rsqrtf(red[NMIX + 1] / float(H) + a.norm_eps);
    }

template <bool ROUTE>
__global__ void __launch_bounds__(T, 1) mhc_kernel(const Args a) {
  extern __shared__ __align__(16) float fsm[];
  float (*fns)[HC][SL] = reinterpret_cast<float (*)[HC][SL]>(fsm);
  __nv_bfloat16* gsm = reinterpret_cast<__nv_bfloat16*>(fsm + NMIX * HC * SL);  // [E][SL]
  __nv_bfloat16* xsm = gsm + E_MAX * SL;                                       // [MAXM][SL]
  __shared__ int last;
  const int tid = threadIdx.x, lane = tid & 31, t = tid >> 5, c = blockIdx.x;
  const int h0 = c * SL;
  const unsigned e0 = ld_volatile(a.ctr + 1);
  for (int i = tid; i < NMIX * HC * SL / 4; i += T) {  // weights first (no upstream dependency)
    const int n = i / (HC * SL / 4), r = i % (HC * SL / 4), j = r / (SL / 4), q = r % (SL / 4);
    reinterpret_cast<float4*>(&fns[n][j][0])[q] = __ldg(reinterpret_cast<const float4*>(a.fn + (long(n) * HC + j) * H + h0) + q);
  }
  if (ROUTE) {
    for (int i = tid; i < a.E * SL / 8; i += T) {
      const int e = i / (SL / 8), q = i % (SL / 8);
      reinterpret_cast<uint4*>(gsm + e * SL)[q] = __ldg(reinterpret_cast<const uint4*>(a.gate_w + long(e) * H + h0) + q);
    }
  }
  pdl_wait();
  __syncthreads();
  const bool live = t < a.m;
  float cv[4] = {0.f, 0.f, 0.f, 0.f};  // collapse (bf16-rounded values) of my 4 hidden values
  if (live) {
    float pm[HC], cm[HC][HC], pin[HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) pm[j] = a.post[t * HC + j], pin[j] = a.pre_in[t * HC + j];
#pragma unroll
    for (int k = 0; k < HC; ++k)
#pragma unroll
      for (int j = 0; j < HC; ++j) cm[k][j] = a.comb[(t * HC + k) * HC + j];
    const int h = h0 + 4 * lane;
    float xv[4], rv[HC][4];
    {
      const uint2 u = *reinterpret_cast<const uint2*>(a.x + long(t) * H + h);
      const float2 f0 = unpack_bf16(u.x), f1 = unpack_bf16(u.y);
      xv[0] = f0.x, xv[1] = f0.y, xv[2] = f1.x, xv[3] = f1.y;
    }
#pragma unroll
    for (int k = 0; k < HC; ++k) {
      const uint2 u = *reinterpret_cast<const uint2*>(a.res + (long(t) * HC + k) * H + h);
      const float2 f0 = unpack_bf16(u.x), f1 = unpack_bf16(u.y);
      rv[k][0] = f0.x, rv[k][1] = f0.y, rv[k][2] = f1.x, rv[k][3] = f1.y;
    }
    float sqr = 0.f, mix[NMIX], css = 0.f;
#pragma unroll
    for (int n = 0; n < NMIX; ++n) mix[n] = 0.f;
#pragma unroll 1
    for (int e = 0; e < 4; ++e) {
      float nr[HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float v = pm[j] * xv[e];
#pragma unroll
        for (int k = 0; k < HC; ++k) v = __fmaf_rn(cm[k][j], rv[k][e], v);
        nr[j] = v;
        sqr = __fmaf_rn(v, v, sqr);
      }
#pragma unroll
      for (int n = 0; n < NMIX; ++n)
#pragma unroll
        for (int j = 0; j < HC; ++j) mix[n] = __fmaf_rn(fns[n][j][4 * lane + e], nr[j], mix[n]);
      float o = 0.f;
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        const __nv_bfloat16 b = __float2bfloat16_rn(nr[j]);
        a.res_out[(long(t) * HC + j) * H + h + e] = b;
        o = __fmaf_rn(pin[j], __bfloat162float(b), o);
      }
      cv[e] = __bfloat162float(__float2bfloat16_rn(o));
      css = __fmaf_rn(cv[e], cv[e], css);
    }
    // warp sums -> per-CTA partials
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      sqr += __shfl_xor_sync(0xffffffffu, sqr, o);
      css += __shfl_xor_sync(0xffffffffu, css, o);
#pragma unroll
      for (int n = 0; n < NMIX; ++n) mix[n] += __shfl_xor_sync(0xffffffffu, mix[n], o);
    }
    if (lane == 0) {
      float* p = a.part + (long(c) * MAXM + t) * NP;
#pragma unroll
      for (int n = 0; n < NMIX; ++n) p[n] = mix[n];
      p[NMIX] = sqr;
      p[NMIX + 1] = css;
    }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    last = atomicAdd(a.ctr, 1u) == unsigned(G - 1);
  }
  __syncthreads();
  if (last) {
    __threadfence();
    __shared__ float red[MAXM][NP];
    for (int i = tid; i < a.m * NP; i += T) {  // fixed CTA order
      const int tt = i / NP, n = i % NP;
      float v = 0.f;
      for (int cc = 0; cc < G; ++cc) v += __ldcg(a.part + (long(cc) * MAXM + tt) * NP + n);
      red[tt][n] = v;
    }
    __syncthreads();
    if (tid < a.m) coeffs(a, red[tid], tid);
    __syncthreads();
    if (tid == 0) {
      a.ctr[0] = 0u;
      __threadfence();
      atomicExch(a.ctr + 1, e0 + 1u);
    }
  }
  // every CTA: wait for the coefficients, then normalize its own slice
  if (tid == 0) {
    unsigned spins = 0;
    while (ld_volatile(a.ctr + 1) == e0) {
      if (++spins > 32u) __nanosleep(64);
    }
    __threadfence();
  }
  __syncthreads();
  if (live) {
    const float r = __ldcg(a.nrm + t);
    const int h = h0 + 4 * lane;
    const uint2 wu = *reinterpret_cast<const uint2*>(a.norm_w + h);
    const float2 w0 = unpack_bf16(wu.x), w1 = unpack_bf16(wu.y);
    uint2 u;
    u.x = pack_bf16(cv[0] * r * w0.x, cv[1] * r * w0.y);
    u.y = pack_bf16(cv[2] * r * w1.x, cv[3] * r * w1.y);
    *reinterpret_cast<uint2*>(a.layer_input + long(t) * H + h) = u;
    if (ROUTE) *reinterpret_cast<uint2*>(xsm + t * SL + 4 * lane) = u;
  }
  if (ROUTE) {
    __syncthreads();
    // partial logits of this hidden slice: thread (expert e = tid / 2, half) over 64 hidden values, all tokens
    const int e = tid >> 1, half = tid & 1;
    float acc[MAXM];
#pragma unroll
    for (int tt = 0; tt < MAXM; ++tt) acc[tt] = 0.f;
    if (e < a.E) {
      const __nv_bfloat16* gr = gsm + e * SL + 64 * half;
#pragma unroll 4
      for (int q = 0; q < 64; q += 2) {
        const float2 g2 = unpack_bf16(*reinterpret_cast<const unsigned*>(gr + q));
#pragma unroll
        for (int tt = 0; tt < MAXM; ++tt) {
          if (tt < a.m) {
            const float2 x2 = unpack_bf16(*reinterpret_cast<const unsigned*>(xsm + tt * SL + 64 * half + q));
            acc[tt] = __fmaf_rn(x2.x, g2.x, acc[tt]);
            acc[tt] = __fmaf_rn(x2.y, g2.y, acc[tt]);
          }
        }
      }
    }
#pragma unroll
    for (int tt = 0; tt < MAXM; ++tt) acc[tt] += __shfl_xor_sync(0xffffffffu, acc[tt], 1);
    if (half == 0 && e < a.E) {
#pragma unroll
      for (int tt = 0; tt < MAXM; ++tt)
        if (tt < a.m) a.rpart[(long(c) * MAXM + tt) * a.E + e] = acc[tt];
    }
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      last = atomicAdd(a.ctr + 2, 1u) == unsigned(G - 1);
    }
    __syncthreads();
    if (last) {
      __threadfence();
      __shared__ float lg[MAXM][E_MAX];
      for (int i = tid; i < a.m * a.E; i += T) {  // fixed CTA order
        const int tt = i / a.E, ee = i % a.E;
        float v = 0.f;
        for (int cc = 0; cc < G; ++cc) v += __ldcg(a.rpart + (long(cc) * MAXM + tt) * a.E + ee);
        lg[tt][ee] = v;
      }
      __syncthreads();
      if (tid == 0) a.ctr[2] = 0u;
      // top-k: warp tt = token tt (drouter::router_kernel semantics)
      if (t < a.m) {
        auto score = [](float v) { return sqrtf(v > 20.f ? v : logf(1.f + expf(v))); };
        bool image = false;
        if (a.bias_vl != nullptr && a.ids_in != nullptr && a.image_lo > 0) {
          const long id = long(a.ids_in[t]);
          image = id >= a.image_lo && id < a.image_lo + 5;
        }
        const float* b = image ? a.bias_vl : a.bias;
        constexpr int VPL = E_MAX / 32;
        float cur[VPL];
#pragma unroll
        for (int v = 0; v < VPL; ++v) {
          const int ex = v * 32 + lane;
          if (ex < a.E) {
            const float cc = score(lg[t][ex]) + (b != nullptr ? b[ex] : 0.f);
            cur[v] = (cc == cc) ? cc : -1e30f;
          } else {
            cur[v] = -INFINITY;
          }
        }
        float sel_w = 0.f, sum = 0.f;
        int sel_id = 0;
        for (int kk = 0; kk < a.k; ++kk) {
          float best = -INFINITY;
          int best_e = 0x7fffffff;
#pragma unroll
          for (int v = 0; v < VPL; ++v) {
            const int ex = v * 32 + lane;
            if (cur[v] > best || (cur[v] == best && ex < best_e)) best = cur[v], best_e = ex;
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oe = __shfl_xor_sync(0xffffffffu, best_e, o);
            if (ob > best || (ob == best && oe < best_e)) best = ob, best_e = oe;
          }
#pragma unroll
          for (int v = 0; v < VPL; ++v)
            if (v * 32 + lane == best_e) cur[v] = -INFINITY;
          const float wk = score(lg[t][best_e]);
          sum += wk;
          if (lane == kk) sel_w = wk, sel_id = best_e;
        }
        const float mul = a.renorm ? a.rscale / (sum > 0.f ? sum : 1.f) : a.rscale;
        if (lane < a.k) {
          a.out_w[t * a.k + lane] = sel_w * mul;
          a.out_ids[t * a.k + lane] = sel_id;
        }
      }
    }
  }
  pdl_launch_dependents();
}

// ------------------------------------------------------------------------------------------------ cluster variant
// DS41H_DRAFTER_MHC_CL=1 (with DS41H_DRAFTER_MHC=2): the same FFN-side mHC + router as ONE thread-block cluster of 16 CTAs
// (hidden slice of 320 per CTA), all cross-CTA reductions through distributed shared memory and two hardware cluster
// barriers instead of global partials + atomic tickets + a serial last-CTA reduction (the v1 kernel above: 19 us standalone).
// Critical-path order: phase A (every CTA: post-map -> residual_out, collapse c with the CARRIED pre-mix, partial mixes / sqr /
// sum c^2, partial router logits sum_h gate[e][h] * c[h] * w[h]); cluster barrier; phase B (every CTA: norm factors
// r = rsqrt(sum c^2 / H + eps) and its layer_input slice bf16(c * w * r); rank i reduces the logits of experts [8 i, 8 i + 8)
// into rank 0; the last rank reduces the 24 mixes + sqr and runs the Sinkhorn coefficients); cluster barrier; rank 0 top-k.
// Drafter-only approximations vs stock: router logits from the unrounded c * w * r (stock: the bf16 layer_input) and the
// layer_input rounding order (c * w) * r; fp32 summation orders differ (fixed: deterministic). Weights (fn, gate, norm) are
// prefetched into smem before the PDL wait. Graph safe (no global counters).
namespace cg = cooperative_groups;
constexpr int CL = 16, SLC = H / CL, VPL2 = SLC / 32, GPAD = SLC + 2, T2 = 256, EPR = E_MAX / CL;
static_assert(SLC % 32 == 0 && SLC % 4 == 0 && E_MAX % CL == 0 && T2 / 32 == MAXM, "cluster layout");
constexpr int O_FN = 0;                                // [NMIX][HC][SLC] f32
constexpr int O_G = O_FN + NMIX * HC * SLC * 4;        // [E_MAX][GPAD] bf16 (padded rows: conflict-free per-expert reads)
constexpr int O_C = O_G + E_MAX * GPAD * 2;            // [MAXM][SLC] f32: c * w
constexpr int O_W = O_C + MAXM * SLC * 4;              // [SLC] f32 norm weight slice
constexpr int O_P = O_W + SLC * 4;                     // [MAXM][NP] f32: mixes, sqr, sum c^2 (read by other ranks)
constexpr int O_RP = O_P + MAXM * NP * 4;              // [2][MAXM][E_MAX] f32 router partial halves (read by other ranks)
constexpr int O_LG = O_RP + 2 * MAXM * E_MAX * 4;      // rank 0: [MAXM][E_MAX] logits; last rank: [MAXM][NP] reduced mixes
constexpr int O_R = O_LG + MAXM * E_MAX * 4;           // [MAXM] norm factors
constexpr int O_TMP = O_R + MAXM * 4;                  // [MAXM][CL] gathered sum c^2
constexpr int SMEM_CL = O_TMP + MAXM * CL * 4;
static_assert(SMEM_CL <= 232448, "cluster smem");

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

template <bool ROUTE>
__global__ void __launch_bounds__(T2, 1) cl_kernel(const Args a) {
  cg::cluster_group cluster = cg::this_cluster();
  extern __shared__ __align__(16) unsigned char sm[];
  float* fns = reinterpret_cast<float*>(sm + O_FN);
  unsigned* gws = reinterpret_cast<unsigned*>(sm + O_G);
  float* cws = reinterpret_cast<float*>(sm + O_C);
  float* wsm = reinterpret_cast<float*>(sm + O_W);
  float* part = reinterpret_cast<float*>(sm + O_P);
  float* rp = reinterpret_cast<float*>(sm + O_RP);
  float* lg = reinterpret_cast<float*>(sm + O_LG);
  float* rr = reinterpret_cast<float*>(sm + O_R);
  float* tmp = reinterpret_cast<float*>(sm + O_TMP);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rank = int(cluster.block_rank());
  const int h0 = rank * SLC;
  // constants first (independent of the upstream kernels): fn / gate / norm-weight slices into smem
  for (int i = tid; i < NMIX * HC * (SLC / 4); i += T2) {
    const int row = i / (SLC / 4), q = i % (SLC / 4);  // row = n * HC + j of fn [NMIX, HC * H]
    cp_async16(fns + row * SLC + 4 * q, a.fn + long(row) * H + h0 + 4 * q);
  }
  if (ROUTE) {
    for (int i = tid; i < a.E * (SLC / 2); i += T2) {
      const int e = i / (SLC / 2), q = i % (SLC / 2);
      cp_async4(gws + e * (GPAD / 2) + q, a.gate_w + long(e) * H + h0 + 2 * q);
    }
  }
  cp_async_commit();
  for (int i = tid; i < SLC; i += T2) wsm[i] = bf2f(a.norm_w[h0 + i]);
  pdl_wait();
  cp_async_wait_all();
  __syncthreads();
  // ---- phase A: warp t = token t; lane owns hidden h = lane + 32 i of the slice
  const int t = warp;
  if (t < a.m) {
    float pm[HC], pin[HC], cm[HC][HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) pm[j] = a.post[t * HC + j], pin[j] = a.pre_in[t * HC + j];
#pragma unroll
    for (int k = 0; k < HC; ++k)
#pragma unroll
      for (int j = 0; j < HC; ++j) cm[k][j] = a.comb[(t * HC + k) * HC + j];
    float xv[VPL2], rv[HC][VPL2];
#pragma unroll
    for (int i = 0; i < VPL2; ++i) {
      const int hg = h0 + lane + 32 * i;
      xv[i] = bf2f(a.x[long(t) * H + hg]);
#pragma unroll
      for (int k = 0; k < HC; ++k) rv[k][i] = bf2f(a.res[(long(t) * HC + k) * H + hg]);
    }
    float mixv[NMIX], sqr = 0.f, css = 0.f;
#pragma unroll
    for (int n = 0; n < NMIX; ++n) mixv[n] = 0.f;
#pragma unroll
    for (int i = 0; i < VPL2; ++i) {
      const int h = lane + 32 * i, hg = h0 + h;
      float nr[HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float v = pm[j] * xv[i];
#pragma unroll
        for (int k = 0; k < HC; ++k) v = __fmaf_rn(cm[k][j], rv[k][i], v);
        nr[j] = v;
        sqr = __fmaf_rn(v, v, sqr);
      }
#pragma unroll
      for (int n = 0; n < NMIX; ++n)
#pragma unroll
        for (int j = 0; j < HC; ++j) mixv[n] = __fmaf_rn(fns[(n * HC + j) * SLC + h], nr[j], mixv[n]);
      float o = 0.f;
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        const __nv_bfloat16 b = __float2bfloat16_rn(nr[j]);
        a.res_out[(long(t) * HC + j) * H + hg] = b;
        o = __fmaf_rn(pin[j], bf2f(b), o);
      }
      const float c = bf2f(__float2bfloat16_rn(o));
      css = __fmaf_rn(c, c, css);
      cws[t * SLC + h] = c * wsm[h];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sqr += __shfl_xor_sync(0xffffffffu, sqr, off);
      css += __shfl_xor_sync(0xffffffffu, css, off);
#pragma unroll
      for (int n = 0; n < NMIX; ++n) mixv[n] += __shfl_xor_sync(0xffffffffu, mixv[n], off);
    }
    if (lane == 0) {
#pragma unroll
      for (int n = 0; n < NMIX; ++n) part[t * NP + n] = mixv[n];
      part[t * NP + NMIX] = sqr;
      part[t * NP + NMIX + 1] = css;
    }
  } else {
#pragma unroll
    for (int i = 0; i < VPL2; ++i) cws[t * SLC + lane + 32 * i] = 0.f;  // padded tokens contribute nothing
  }
  if (ROUTE) {
    __syncthreads();
    // partial router logits of this slice: thread (expert e = lane + 32 (warp % 4), half = warp / 4) over 160 hidden values
    const int e = lane + 32 * (warp & 3), half = warp >> 2;
    float acc[MAXM];
#pragma unroll
    for (int tt = 0; tt < MAXM; ++tt) acc[tt] = 0.f;
    if (e < a.E) {
      const unsigned* gr = gws + e * (GPAD / 2) + half * (SLC / 4);
      const float* cb = cws + half * (SLC / 2);
#pragma unroll 4
      for (int q = 0; q < SLC / 4; ++q) {
        const float2 g2 = unpack_bf16(gr[q]);
#pragma unroll
        for (int tt = 0; tt < MAXM; ++tt) {
          acc[tt] = __fmaf_rn(g2.x, cb[tt * SLC + 2 * q], acc[tt]);
          acc[tt] = __fmaf_rn(g2.y, cb[tt * SLC + 2 * q + 1], acc[tt]);
        }
      }
    }
#pragma unroll
    for (int tt = 0; tt < MAXM; ++tt) rp[(half * MAXM + tt) * E_MAX + e] = acc[tt];
  }
  cluster.sync();  // every rank's part / rp complete and visible cluster-wide
  // ---- phase B
  if (tid < MAXM * CL) {
    const int tt = tid / CL, c = tid % CL;
    const float* rpart = cluster.map_shared_rank(part, c);
    tmp[tt * CL + c] = tt < a.m ? rpart[tt * NP + NMIX + 1] : 0.f;
  }
  __syncthreads();
  if (tid < MAXM) {
    float sacc = 0.f;
#pragma unroll
    for (int c = 0; c < CL; ++c) sacc += tmp[tid * CL + c];
    rr[tid] = rsqrtf(sacc / float(H) + a.norm_eps);
  }
  __syncthreads();
  for (int i = tid; i < a.m * SLC; i += T2) {
    const int tt = i / SLC, h = i % SLC;
    a.layer_input[long(tt) * H + h0 + h] = __float2bfloat16_rn(cws[tt * SLC + h] * rr[tt]);
  }
  if (ROUTE && tid < MAXM * EPR) {
    const int tt = tid / EPR, e = rank * EPR + tid % EPR;
    if (tt < a.m && e < a.E) {
      float sacc = 0.f;
#pragma unroll
      for (int c = 0; c < CL; ++c) {
        const float* r = cluster.map_shared_rank(rp, c);
        sacc += r[tt * E_MAX + e] + r[(MAXM + tt) * E_MAX + e];
      }
      float* lg0 = cluster.map_shared_rank(lg, 0);
      lg0[tt * E_MAX + e] = sacc * rr[tt];
    }
  }
  if (rank == CL - 1) {  // mixes + Sinkhorn (the last rank's LG region is free: only rank 0 receives logits)
    float* red = lg;
    for (int i = tid; i < a.m * NP; i += T2) {
      const int tt = i / NP, n = i % NP;
      float sacc = 0.f;
#pragma unroll
      for (int c = 0; c < CL; ++c) sacc += cluster.map_shared_rank(part, c)[tt * NP + n];
      red[tt * NP + n] = sacc;
    }
    __syncthreads();
    if (tid < a.m) coeffs(a, red + tid * NP, tid);
  }
  cluster.sync();  // logits in rank 0; no rank reads remote smem after this point
  if (ROUTE && rank == 0 && t < a.m) {
    auto score = [](float v) { return sqrtf(v > 20.f ? v : logf(1.f + expf(v))); };
    bool image = false;
    if (a.bias_vl != nullptr && a.ids_in != nullptr && a.image_lo > 0) {
      const long id = long(a.ids_in[t]);
      image = id >= a.image_lo && id < a.image_lo + 5;
    }
    const float* b = image ? a.bias_vl : a.bias;
    constexpr int VPL = E_MAX / 32;
    float cur[VPL];
#pragma unroll
    for (int v = 0; v < VPL; ++v) {
      const int ex = v * 32 + lane;
      if (ex < a.E) {
        const float cc = score(lg[t * E_MAX + ex]) + (b != nullptr ? b[ex] : 0.f);
        cur[v] = (cc == cc) ? cc : -1e30f;
      } else {
        cur[v] = -INFINITY;
      }
    }
    float sel_w = 0.f, sum = 0.f;
    int sel_id = 0;
    for (int kk = 0; kk < a.k; ++kk) {
      float best = -INFINITY;
      int best_e = 0x7fffffff;
#pragma unroll
      for (int v = 0; v < VPL; ++v) {
        const int ex = v * 32 + lane;
        if (cur[v] > best || (cur[v] == best && ex < best_e)) best = cur[v], best_e = ex;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oe = __shfl_xor_sync(0xffffffffu, best_e, o);
        if (ob > best || (ob == best && oe < best_e)) best = ob, best_e = oe;
      }
#pragma unroll
      for (int v = 0; v < VPL; ++v)
        if (v * 32 + lane == best_e) cur[v] = -INFINITY;
      const float wk = score(lg[t * E_MAX + best_e]);
      sum += wk;
      if (lane == kk) sel_w = wk, sel_id = best_e;
    }
    const float mul = a.renorm ? a.rscale / (sum > 0.f ? sum : 1.f) : a.rscale;
    if (lane < a.k) {
      a.out_w[t * a.k + lane] = sel_w * mul;
      a.out_ids[t * a.k + lane] = sel_id;
    }
  }
  pdl_launch_dependents();
}
}  // namespace dmhc

// ================================================================== drafter final mHC post + collapse + RMSNorm (FINAL_FUSE)
// Stock end of the drafter forward + head input (3 launches): mhc_post_tilelang new_r[j][h] = bf16(post[j] x[h] +
// sum_i comb[i][j] res[i][h]) (fp32: post * x, then FMAs in input-stream order) -> hc_collapse_triton c[h] =
// bf16(sum_j pre[j] new_r[j][h]) -> vLLM rms_norm y[h] = bf16(bf16(c[h] rsqrt(sum c^2 / H + eps)) w[h]). One CTA per token
// (640 threads x 8 hidden values); writes c (the forward's stock output) and y (the normed head input). Same per-element
// math and roundings; only the fp32 order of the sum of squares differs from vLLM's block reduction (drafter-only).
namespace dfin {
constexpr int H = 5120, HC = 4, T = 640, V = 8, NW = T / 32, MAXM = 8;
static_assert(T * V == H, "8 hidden values per thread");

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__global__ void __launch_bounds__(T) fin_kernel(const __nv_bfloat16* __restrict__ x, long ldx, const __nv_bfloat16* __restrict__ res,
                                                const float* __restrict__ post, const float* __restrict__ comb,
                                                const float* __restrict__ pre, const __nv_bfloat16* __restrict__ w, float eps,
                                                __nv_bfloat16* __restrict__ c_out, __nv_bfloat16* __restrict__ y_out) {
  __shared__ float red[NW];
  const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, h0 = tid * V;
  const uint4 wv = __ldg(reinterpret_cast<const uint4*>(w + h0));  // the norm weight does not depend on the upstream kernel
  pdl_wait();
  float cm[HC][HC], pm[HC], qm[HC];
#pragma unroll
  for (int i = 0; i < HC; ++i) {
    pm[i] = post[t * HC + i];
    qm[i] = pre[t * HC + i];
#pragma unroll
    for (int j = 0; j < HC; ++j) cm[i][j] = comb[(t * HC + i) * HC + j];
  }
  const uint4 xv = *reinterpret_cast<const uint4*>(x + long(t) * ldx + h0);
  uint4 rv[HC];
#pragma unroll
  for (int i = 0; i < HC; ++i) rv[i] = *reinterpret_cast<const uint4*>(res + (long(t) * HC + i) * H + h0);
  float xf[V], rf[HC][V], wf[V];
  {
    const unsigned u[4] = {xv.x, xv.y, xv.z, xv.w}, uw[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 f = unpack_bf16(u[q]), g = unpack_bf16(uw[q]);
      xf[2 * q] = f.x, xf[2 * q + 1] = f.y, wf[2 * q] = g.x, wf[2 * q + 1] = g.y;
    }
#pragma unroll
    for (int i = 0; i < HC; ++i) {
      const unsigned r[4] = {rv[i].x, rv[i].y, rv[i].z, rv[i].w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = unpack_bf16(r[q]);
        rf[i][2 * q] = f.x, rf[i][2 * q + 1] = f.y;
      }
    }
  }
  float cc[V], ss = 0.0f;
#pragma unroll
  for (int e = 0; e < V; ++e) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      float nr = pm[j] * xf[e];
#pragma unroll
      for (int i = 0; i < HC; ++i) nr = __fmaf_rn(cm[i][j], rf[i][e], nr);
      acc = __fmaf_rn(qm[j], bf16r(nr), acc);
    }
    cc[e] = bf16r(acc);
    ss = __fmaf_rn(cc[e], cc[e], ss);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int q = 0; q < NW; ++q) tot += red[q];
  const float sc = rsqrtf(tot / float(H) + eps);
  float yv[V];
#pragma unroll
  for (int e = 0; e < V; ++e) yv[e] = bf16r(cc[e] * sc) * wf[e];
  *reinterpret_cast<uint4*>(c_out + long(t) * H + h0) =
      make_uint4(pack_bf16(cc[0], cc[1]), pack_bf16(cc[2], cc[3]), pack_bf16(cc[4], cc[5]), pack_bf16(cc[6], cc[7]));
  *reinterpret_cast<uint4*>(y_out + long(t) * H + h0) =
      make_uint4(pack_bf16(yv[0], yv[1]), pack_bf16(yv[2], yv[3]), pack_bf16(yv[4], yv[5]), pack_bf16(yv[6], yv[7]));
  pdl_launch_dependents();
}
}  // namespace dfin

// x [m, H] bf16 (row stride ldx), res [m, 4, H] bf16, post [m, 4(, 1)] / comb [m, 4, 4] / pre [m, 4] f32, w [H] bf16 ->
// c_out = stock collapse(mhc_post(...)) and y_out = rms_norm(c_out) (both [m, H] bf16, contiguous)
void final_fuse(torch::Tensor x, torch::Tensor res, torch::Tensor post, torch::Tensor comb, torch::Tensor pre, torch::Tensor w,
                double eps, torch::Tensor c_out, torch::Tensor y_out, bool pdl) {
  using namespace dfin;
  const int m = int(x.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM, "final_fuse: 1..8 tokens");
  auto bfc = [](const torch::Tensor& t) { return t.scalar_type() == at::kBFloat16 && t.is_contiguous(); };
  auto f32 = [](const torch::Tensor& t) { return t.scalar_type() == at::kFloat && t.is_contiguous(); };
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.size(1) == H && x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "final_fuse: x [m, 5120] bf16, 16-byte aligned rows");
  TORCH_CHECK(bfc(res) && res.numel() == long(m) * HC * H && reinterpret_cast<uintptr_t>(res.data_ptr()) % 16 == 0, "final_fuse: residual [m, 4, 5120]");
  TORCH_CHECK(f32(post) && post.numel() == m * HC && f32(comb) && comb.numel() == m * HC * HC && f32(pre) && pre.numel() == m * HC,
              "final_fuse: mixes (contiguous fp32)");
  TORCH_CHECK(bfc(w) && w.numel() == H && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "final_fuse: norm weight");
  TORCH_CHECK(bfc(c_out) && c_out.numel() == long(m) * H && bfc(y_out) && y_out.numel() == long(m) * H &&
                  reinterpret_cast<uintptr_t>(c_out.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(y_out.data_ptr()) % 16 == 0,
              "final_fuse: outputs");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* rp = reinterpret_cast<const __nv_bfloat16*>(res.data_ptr());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  auto* cp = reinterpret_cast<__nv_bfloat16*>(c_out.data_ptr());
  auto* yp = reinterpret_cast<__nv_bfloat16*>(y_out.data_ptr());
  const float* pp = post.data_ptr<float>();
  const float* cbp = comb.data_ptr<float>();
  const float* qp = pre.data_ptr<float>();
  const long ldx = x.stride(0);
  const float e = float(eps);
  if (pdl) {
    launch_pdl("ds41h final_fuse", fin_kernel, dim3(m), dim3(T), 0, xp, ldx, rp, pp, cbp, qp, wp, e, cp, yp);
  } else {
    fin_kernel<<<m, T, 0, c10::cuda::getCurrentCUDAStream()>>>(xp, ldx, rp, pp, cbp, qp, wp, e, cp, yp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

int64_t scratch_floats() { return int64_t(dmhc::G) * dmhc::MAXM * dmhc::NP + dmhc::MAXM; }

static int g_mhc_variant = 0;  // 0: v1 kernel (global partials), 1: cluster kernel (DS41H_DRAFTER_MHC_CL)
void mhc_set_variant(int64_t v) { g_mhc_variant = int(v); }

template <bool ROUTE>
static void launch_cl(const dmhc::Args& a, bool pdl) {
  using namespace dmhc;
  static const bool attr = [] {
    TORCH_CHECK(cudaFuncSetAttribute(cl_kernel<ROUTE>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess,
                "dmhc cluster: non-portable cluster size not allowed");
    TORCH_CHECK(cudaFuncSetAttribute(cl_kernel<ROUTE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_CL) == cudaSuccess,
                "dmhc cluster: smem opt-in");
    return true;
  }();
  (void)attr;
  cudaLaunchAttribute at[2];
  int na = 0;
  at[na].id = cudaLaunchAttributeClusterDimension;
  at[na].val.clusterDim.x = CL;
  at[na].val.clusterDim.y = 1;
  at[na].val.clusterDim.z = 1;
  ++na;
  if (pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(CL);
  cfg.blockDim = dim3(T2);
  cfg.dynamicSmemBytes = SMEM_CL;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cfg.attrs = at;
  cfg.numAttrs = na;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, cl_kernel<ROUTE>, a);
  TORCH_CHECK(err == cudaSuccess, "dmhc cluster launch failed: ", cudaGetErrorString(err));
}

// max active clusters of the cluster kernel on this device (0 = cannot run: no GPC with 16 free SMs for this config)
int64_t cl_max_active(bool route) {
  using namespace dmhc;
  auto k = route ? cl_kernel<true> : cl_kernel<false>;
  TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess);
  TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_CL) == cudaSuccess);
  cudaLaunchAttribute at{};
  at.id = cudaLaunchAttributeClusterDimension;
  at.val.clusterDim.x = CL;
  at.val.clusterDim.y = 1;
  at.val.clusterDim.z = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(CL);
  cfg.blockDim = dim3(T2);
  cfg.dynamicSmemBytes = SMEM_CL;
  cfg.attrs = &at;
  cfg.numAttrs = 1;
  int n = 0;
  if (cudaOccupancyMaxActiveClusters(&n, k, &cfg) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return n;
}

// x [m, H] bf16, res [m, 4, H] bf16, post [m, 4(, 1)] f32, comb [m, 4, 4] f32, fn [24, 4H] f32, hc_scale [3], hc_base [24],
// pre_in [m, 4] f32, norm_w [H] bf16 -> res_out, post_out, comb_out, layer_input, pre_out (preallocated).
void mhc_ffn(torch::Tensor x, torch::Tensor res, torch::Tensor post, torch::Tensor comb, torch::Tensor fn, torch::Tensor hc_scale,
             torch::Tensor hc_base, torch::Tensor pre_in, torch::Tensor norm_w, torch::Tensor res_out, torch::Tensor post_out,
             torch::Tensor comb_out, torch::Tensor layer_input, torch::Tensor pre_out, torch::Tensor scratch, torch::Tensor ctr,
             double rms_eps, double pre_eps, double sk_eps, double post_mult, int64_t sk_iters, double norm_eps, bool pdl,
             std::optional<torch::Tensor> gate_w, std::optional<torch::Tensor> bias, std::optional<torch::Tensor> bias_vl,
             std::optional<torch::Tensor> ids_in, int64_t image_lo, std::optional<torch::Tensor> rpart, std::optional<torch::Tensor> out_ids,
             std::optional<torch::Tensor> out_w, double rscale, bool renorm, int64_t topk) {
  using namespace dmhc;
  const int m = int(x.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM, "dmhc: 1..8 tokens");
  auto bf = [](const torch::Tensor& t) { return t.scalar_type() == at::kBFloat16 && t.is_contiguous(); };
  auto f32 = [](const torch::Tensor& t) { return t.scalar_type() == at::kFloat && t.is_contiguous(); };
  TORCH_CHECK(bf(x) && x.numel() == long(m) * H && bf(res) && res.numel() == long(m) * HC * H, "dmhc: x / residual");
  TORCH_CHECK(f32(post) && post.numel() == m * HC && f32(comb) && comb.numel() == m * HC * HC && f32(pre_in) && pre_in.numel() == m * HC, "dmhc: mixes");
  TORCH_CHECK(f32(fn) && fn.numel() == long(NMIX) * HC * H && f32(hc_scale) && hc_scale.numel() == 3 && f32(hc_base) && hc_base.numel() == NMIX, "dmhc: hc params");
  TORCH_CHECK(bf(norm_w) && norm_w.numel() == H, "dmhc: norm weight");
  TORCH_CHECK(bf(res_out) && res_out.numel() == long(m) * HC * H && f32(post_out) && post_out.numel() == m * HC && f32(comb_out) &&
                  comb_out.numel() == m * HC * HC && bf(layer_input) && layer_input.numel() == long(m) * H && f32(pre_out) && pre_out.numel() == m * HC,
              "dmhc: outputs");
  TORCH_CHECK(g_mhc_variant == 1 || (f32(scratch) && scratch.numel() >= scratch_floats() && ctr.scalar_type() == at::kInt && ctr.numel() >= 3),
              "dmhc: scratch");
  for (const auto* t : {&x, &res, &res_out, &layer_input, &norm_w}) TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 8 == 0, "dmhc: alignment");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(fn.data_ptr()) % 16 == 0, "dmhc: fn alignment");
  Args a{};
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.res = reinterpret_cast<const __nv_bfloat16*>(res.data_ptr());
  a.post = post.data_ptr<float>();
  a.comb = comb.data_ptr<float>();
  a.fn = fn.data_ptr<float>();
  a.hc_scale = hc_scale.data_ptr<float>();
  a.hc_base = hc_base.data_ptr<float>();
  a.pre_in = pre_in.data_ptr<float>();
  a.norm_w = reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr());
  a.res_out = reinterpret_cast<__nv_bfloat16*>(res_out.data_ptr());
  a.post_out = post_out.data_ptr<float>();
  a.comb_out = comb_out.data_ptr<float>();
  a.layer_input = reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr());
  a.pre_out = pre_out.data_ptr<float>();
  a.part = scratch.data_ptr<float>();
  a.nrm = a.part + long(G) * MAXM * NP;
  a.ctr = reinterpret_cast<unsigned*>(ctr.data_ptr<int>());
  a.m = m;
  a.sk_iters = int(sk_iters);
  a.rms_eps = float(rms_eps), a.pre_eps = float(pre_eps), a.sk_eps = float(sk_eps), a.post_mult = float(post_mult), a.norm_eps = float(norm_eps);
  const bool route = gate_w.has_value();
  if (route) {
    TORCH_CHECK(bf(*gate_w) && gate_w->dim() == 2 && gate_w->size(1) == H && gate_w->size(0) <= E_MAX, "dmhc: gate weight [E <= 128, H] bf16");
    TORCH_CHECK(bias.has_value() && f32(*bias) && bias->numel() == gate_w->size(0), "dmhc: gate bias");
    TORCH_CHECK(g_mhc_variant == 1 || (rpart.has_value() && f32(*rpart) && rpart->numel() >= long(G) * MAXM * gate_w->size(0)),
                "dmhc: router scratch");
    TORCH_CHECK(out_ids.has_value() && out_ids->scalar_type() == at::kInt && out_ids->is_contiguous() && out_ids->numel() == m * topk, "dmhc: ids out");
    TORCH_CHECK(out_w.has_value() && f32(*out_w) && out_w->numel() == m * topk && topk >= 1 && topk <= 32, "dmhc: weights out");
    a.gate_w = reinterpret_cast<const __nv_bfloat16*>(gate_w->data_ptr());
    a.bias = bias->data_ptr<float>();
    if (bias_vl.has_value()) {
      TORCH_CHECK(f32(*bias_vl) && bias_vl->numel() == gate_w->size(0), "dmhc: bias_vl");
      a.bias_vl = bias_vl->data_ptr<float>();
    }
    if (ids_in.has_value()) {
      TORCH_CHECK(ids_in->scalar_type() == at::kInt && ids_in->is_contiguous() && ids_in->numel() >= m, "dmhc: ids int32");
      a.ids_in = ids_in->data_ptr<int>();
    }
    a.image_lo = image_lo;
    a.rpart = rpart.has_value() ? rpart->data_ptr<float>() : nullptr;
    a.out_ids = out_ids->data_ptr<int>();
    a.out_w = out_w->data_ptr<float>();
    a.rscale = float(rscale);
    a.renorm = renorm ? 1 : 0;
    a.k = int(topk);
    a.E = int(gate_w->size(0));
  }
  const c10::cuda::CUDAGuard guard(x.device());
  if (g_mhc_variant == 1) {
    a.part = nullptr;
    a.nrm = nullptr;
    a.ctr = nullptr;
    if (route) launch_cl<true>(a, pdl);
    else launch_cl<false>(a, pdl);
    return;
  }
  if (route) {
    opt_in_smem<mhc_kernel<true>>(SMEM_R);
    if (pdl) {
      launch_pdl("ds41h dmhc+router", mhc_kernel<true>, dim3(G), dim3(T), SMEM_R, a);
    } else {
      mhc_kernel<true><<<G, T, SMEM_R, c10::cuda::getCurrentCUDAStream()>>>(a);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return;
  }
  opt_in_smem<mhc_kernel<false>>(SMEM);
  if (pdl) {
    launch_pdl("ds41h dmhc", mhc_kernel<false>, dim3(G), dim3(T), SMEM, a);
  } else {
    mhc_kernel<false><<<G, T, SMEM, c10::cuda::getCurrentCUDAStream()>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

int64_t rpart_floats() { return int64_t(dmhc::G) * dmhc::MAXM * dmhc::E_MAX; }

int64_t preload() {
  cudaFuncAttributes attr;
  int n = 0;
  n += cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(dmhc::mhc_kernel<false>)) == cudaSuccess;
  n += cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(dmhc::mhc_kernel<true>)) == cudaSuccess;
  n += cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(dmhc::cl_kernel<false>)) == cudaSuccess;
  n += cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(dmhc::cl_kernel<true>)) == cudaSuccess;
  n += cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(dfin::fin_kernel)) == cudaSuccess;
  return n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mhc_ffn", &mhc_ffn, "drafter FFN-side mHC (post + delayed pre + collapse + RMSNorm), <= 8 tokens, one launch");
  m.def("scratch_floats", &scratch_floats);
  m.def("rpart_floats", &rpart_floats);
  m.def("preload", &preload);
  m.def("mhc_set_variant", &mhc_set_variant, "0: v1 kernel, 1: 16-CTA cluster kernel (DSMEM reductions)");
  m.def("cl_max_active", &cl_max_active, "cudaOccupancyMaxActiveClusters of the cluster kernel");
  m.def("final_fuse", &final_fuse, "drafter final mhc_post + hc collapse + RMSNorm (<= 8 tokens, one launch): c_out, y_out");
}
