// ds41h MoE data-movement kernels for the W4A16 / W4A8 prefill paths (sm_90a):
//   act_perm16    x bf16 -> routed-row order, W4A16 logical K order (W4A16 GEMM1 input; exact copy)
//   moe_topk      fused sqrtsoftplus + bias top-k routing (stock dsv4_topk semantics)
//   moe_permute   token -> expert permutation (two-pass counting sort over many CTAs)
//   act_quant     x bf16 -> e4m3 (per-token scale) in the GEMM "fragment order" (see below)
//   h_quant       SwiGLU output bf16 -> e4m3 (per-row scale) in fragment order for GEMM2
//   moe_combine   sum of the top-k expert rows of each token (fp32 accumulate) -> bf16
//
// Fragment order. The grouped GEMM (moe_prefill.cu) feeds dequantized FP4 weights to RS-wgmma as
// the A operand. Lane quad index t (= lane & 3) of every warp owns the contiguous physical K range
// [t*KB/4, (t+1)*KB/4) of each K block of KB elements (KB = 128, or 64 for a 64-wide tail block),
// so one 16-byte (8-byte for KB=64) smem load gives it all FP4 values it needs for the block and a
// single ue8m0 scale. wgmma instruction s of the block reads, at fragment column f = 16h + 4t + i
// (h in {0,1}, i in 0..3), the activation of physical k
//      p = t*(KB/4) + 8s + 4h + i
// so activations must be stored at logical column L = 32s + 16h + 4t + i. The dot product over a
// K block is permutation invariant, so this is exact. act_quant/h_quant write that order.
#include "common.cuh"
#include "host.cuh"

#include <algorithm>

namespace {

constexpr int MAX_EXPERTS = 1024;

// ------------------------------------------------------------------------------------------------
// moe_topk: one warp per token. Matches vllm/model_executor/layers/fused_moe/router/dsv4_topk.py
// (_dsv4_topk_kernel): w = sqrt(softplus(logit)) with softplus = logit > 20 ? logit : log(1+e^x),
// choice = w + bias (bias_vl for image sentinel tokens), NaN -> -1e30, top-k by value with ties to
// the lowest expert id, weights = w_sel * scale / sum(w_sel) (sum > 0) when renormalize.
template <int VPL>  // values per lane = ceil(E / 32)
__global__ void __launch_bounds__(128) moe_topk_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                       const float* __restrict__ bias_vl, const int64_t* __restrict__ input_ids,
                                                       long image_lo, float* __restrict__ out_w, int* __restrict__ out_ids, int T, int E,
                                                       int K, float scale, int renormalize) {
  pdl_wait();
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= T) return;
  bool image = false;
  if (bias_vl != nullptr && input_ids != nullptr && image_lo > 0) {
    const long id = input_ids[row];
    image = id >= image_lo && id < image_lo + 5;
  }
  const float* b = image ? bias_vl : bias;
  const float* lrow = logits + size_t(row) * E;
  auto score = [](float x) { return sqrtf(x > 20.f ? x : logf(1.f + expf(x))); };  // sqrt(softplus)
  float cur[VPL];
#pragma unroll
  for (int v = 0; v < VPL; ++v) {
    const int e = v * 32 + lane;
    if (e < E) {
      const float c = score(lrow[e]) + (b != nullptr ? b[e] : 0.f);
      cur[v] = (c == c) ? c : -1e30f;
    } else {
      cur[v] = -INFINITY;
    }
  }
  float sel_w = 0.f, sum = 0.f;
  int sel_id = 0;
  for (int k = 0; k < K; ++k) {
    float best = -INFINITY;
    int best_e = 0x7fffffff;
#pragma unroll
    for (int v = 0; v < VPL; ++v) {
      const int e = v * 32 + lane;
      if (cur[v] > best || (cur[v] == best && e < best_e)) best = cur[v], best_e = e;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oe = __shfl_xor_sync(0xffffffffu, best_e, o);
      if (ob > best || (ob == best && oe < best_e)) best = ob, best_e = oe;
    }
    // owner lane retires the expert; the unbiased score is recomputed from the logit (bit-identical)
#pragma unroll
    for (int v = 0; v < VPL; ++v)
      if (v * 32 + lane == best_e) cur[v] = -INFINITY;
    const float wk = score(lrow[best_e]);
    sum += wk;
    if (lane == k) sel_w = wk, sel_id = best_e;
  }
  const float mul = renormalize ? scale / (sum > 0.f ? sum : 1.f) : scale;
  if (lane < K) {
    out_w[size_t(row) * K + lane] = sel_w * mul;
    out_ids[size_t(row) * K + lane] = sel_id;
  }
}

// ------------------------------------------------------------------------------------------------
// moe_permute: two kernels over G = ceil(R / PERM_CHUNK) CTAs (each owns a contiguous chunk of the
// R = T*K routed rows). moe_count writes per-CTA expert histograms hist[G][E]; moe_scatter (every
// CTA) sums the histograms of all / preceding CTAs per expert, scans them over experts and
// scatters its chunk. Inputs topk_ids/topk_w [R] (row-major [T, K]). Outputs (int32 unless noted),
// meta = counts | off | ctr | perm_tok | perm_slot | pos_of:
//   counts[E+1]           counts per expert (counts[E] = number of valid routed rows)
//   off [E+1]             exclusive prefix of counts (row offsets in expert order)
//   ctr [META_CTR]        zeroed (persistent-GEMM tile counters, see moe_prefill.cu)
//   perm_tok[R]           token of routed row `pos`        (rows >= off[E] untouched)
//   perm_slot[R]          t*K + k of routed row `pos`
//   pos_of[R]             inverse: routed row of slot t*K + k (-1 for dropped ids)
//   perm_w[R] (f32)       routing weight of routed row `pos`
// Ids outside [0, E) (expert-parallel masking) are dropped. The order of rows inside an expert is
// whatever the shared-memory atomics produce; every routed row is computed independently and the
// combine sums in slot order, so outputs do not depend on it.
constexpr int PERM_CHUNK = 2048;
constexpr int META_CTR = 8;

__global__ void __launch_bounds__(512) moe_count_kernel(const int* __restrict__ ids, int R, int E, int* __restrict__ hist, int* __restrict__ pos_of) {
  __shared__ int s_cnt[MAX_EXPERTS];
  pdl_wait();
  const int tid = threadIdx.x;
  for (int e = tid; e < E; e += blockDim.x) s_cnt[e] = 0;
  __syncthreads();
  const int lo = blockIdx.x * PERM_CHUNK, hi = min(R, lo + PERM_CHUNK);
  for (int i = lo + tid; i < hi; i += blockDim.x) {
    const int e = ids[i];
    if (unsigned(e) < unsigned(E))
      atomicAdd(&s_cnt[e], 1);
    else
      pos_of[i] = -1;
  }
  __syncthreads();
  for (int e = tid; e < E; e += blockDim.x) hist[blockIdx.x * E + e] = s_cnt[e];
}

__global__ void __launch_bounds__(1024) moe_scatter_kernel(const int* __restrict__ ids, const float* __restrict__ wts, int R, int K, int E,
                                                           int G, const int* __restrict__ hist, int* __restrict__ counts,
                                                           int* __restrict__ off, int* __restrict__ ctr, int* __restrict__ perm_tok,
                                                           int* __restrict__ perm_slot, float* __restrict__ perm_w, int* __restrict__ pos_of) {
  __shared__ int s_fill[MAX_EXPERTS];
  __shared__ int s_wsum[32];
  pdl_wait();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // per expert (one per thread): total count and the count of the CTAs before this one
  int c = 0, before = 0;
  if (tid < E) {
    for (int g = 0; g < G; ++g) {
      const int h = hist[g * E + tid];
      before += g < int(blockIdx.x) ? h : 0;
      c += h;
    }
  }
  // exclusive scan of c over experts (E <= 1024 = blockDim)
  int ic = c;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int a = __shfl_up_sync(0xffffffffu, ic, o);
    if (lane >= o) ic += a;
  }
  if (lane == 31) s_wsum[warp] = ic;
  __syncthreads();
  if (warp == 0) {
    int a = s_wsum[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int x = __shfl_up_sync(0xffffffffu, a, o);
      if (lane >= o) a += x;
    }
    s_wsum[lane] = a;  // inclusive warp totals
  }
  __syncthreads();
  const int ex_c = (warp ? s_wsum[warp - 1] : 0) + ic - c;
  if (tid < E) {
    s_fill[tid] = ex_c + before;
    if (blockIdx.x == 0) {
      counts[tid] = c;
      off[tid] = ex_c;
    }
  }
  if (blockIdx.x == 0) {
    if (tid == E - 1) {
      off[E] = ex_c + c;
      counts[E] = ex_c + c;
    }
    if (tid < META_CTR) ctr[tid] = 0;
  }
  __syncthreads();
  const int lo = blockIdx.x * PERM_CHUNK, hi = min(R, lo + PERM_CHUNK);
  for (int i = lo + tid; i < hi; i += blockDim.x) {
    const int e = ids[i];
    if (unsigned(e) < unsigned(E)) {
      const int pos = atomicAdd(&s_fill[e], 1);
      perm_tok[pos] = i / K;
      perm_slot[pos] = i;
      pos_of[i] = pos;
      perm_w[pos] = wts[i];
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Quantization helpers. Per-row dynamic scale: s = amax / 448 (s = 1 for an all-zero row),
// q = e4m3_rn_satfinite(x / s). The GEMM multiplies the accumulator by s.
__device__ __forceinline__ float row_scale(float amax) { return amax > 0.f ? amax * (1.f / 448.f) : 1.f; }
__device__ __forceinline__ float row_inv_scale(float amax) { return amax > 0.f ? 448.f / amax : 1.f; }
__device__ __forceinline__ unsigned e4m3x4(float a, float b, float c, float d) { return e4m3x2(a, b) | (e4m3x2(c, d) << 16); }

// Logical byte offset (inside a K block) of the 4 consecutive physical elements [4l, 4l+4) for a
// 128-wide block (l in 0..31) and for a 64-wide block (l in 0..15). See the header comment.
__device__ __forceinline__ int frag_col128(int l) { return 32 * ((l >> 1) & 3) + 16 * (l & 1) + 4 * (l >> 3); }
__device__ __forceinline__ int frag_col64(int l) { return 32 * ((l >> 1) & 1) + 16 * (l & 1) + 4 * (l >> 2); }

// act_quant: one CTA (4 warps) per token row of H bf16 values (H % 128 == 0, H <= 128 * 4 * 16).
// Writes xq [T, H] e4m3 in fragment order and xs [T] f32.
template <int BPW>  // 128-blocks per warp (H / 512)
__global__ void __launch_bounds__(128) act_quant_kernel(const __nv_bfloat16* __restrict__ x, int H, uint8_t* __restrict__ xq,
                                                        float* __restrict__ xs) {
  __shared__ float s_max[4];
  pdl_wait();
  const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const __nv_bfloat16* xr = x + size_t(row) * H;
  float v[BPW][4];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < BPW; ++j) {
    const int blk = warp * BPW + j;
    const uint2 raw = *reinterpret_cast<const uint2*>(xr + blk * 128 + lane * 4);
    const float2 a = unpack_bf16(raw.x), b = unpack_bf16(raw.y);
    v[j][0] = a.x, v[j][1] = a.y, v[j][2] = b.x, v[j][3] = b.y;
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y))));
  }
  amax = warp_max(amax);
  if (lane == 0) s_max[warp] = amax;
  __syncthreads();
  amax = fmaxf(fmaxf(s_max[0], s_max[1]), fmaxf(s_max[2], s_max[3]));
  const float inv = row_inv_scale(amax);
  uint8_t* qr = xq + size_t(row) * H;
#pragma unroll
  for (int j = 0; j < BPW; ++j) {
    const int blk = warp * BPW + j;
    *reinterpret_cast<unsigned*>(qr + blk * 128 + frag_col128(lane)) =
        e4m3x4(v[j][0] * inv, v[j][1] * inv, v[j][2] * inv, v[j][3] * inv);
  }
  if (threadIdx.x == 0) xs[row] = row_scale(amax);
}

// act_quant_perm: act_quant whose quantized row of token t is written to the routed rows of t
// (xq_p[pos_of[t*K + k]], k < K, skipping dropped ids): the activations in routed-row (expert) order,
// so the grouped GEMM can load contiguous row ranges with TMA instead of gathering token rows.
template <int BPW>
__global__ void __launch_bounds__(128) act_quant_perm_kernel(const __nv_bfloat16* __restrict__ x, int H, int K, const int* __restrict__ pos_of,
                                                             uint8_t* __restrict__ xq, float* __restrict__ xs) {
  __shared__ float s_max[4];
  pdl_wait();
  const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const __nv_bfloat16* xr = x + size_t(row) * H;
  float v[BPW][4];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < BPW; ++j) {
    const int blk = warp * BPW + j;
    const uint2 raw = *reinterpret_cast<const uint2*>(xr + blk * 128 + lane * 4);
    const float2 a = unpack_bf16(raw.x), b = unpack_bf16(raw.y);
    v[j][0] = a.x, v[j][1] = a.y, v[j][2] = b.x, v[j][3] = b.y;
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y))));
  }
  amax = warp_max(amax);
  if (lane == 0) s_max[warp] = amax;
  __syncthreads();
  amax = fmaxf(fmaxf(s_max[0], s_max[1]), fmaxf(s_max[2], s_max[3]));
  const float inv = row_inv_scale(amax);
  unsigned q[BPW];
#pragma unroll
  for (int j = 0; j < BPW; ++j) q[j] = e4m3x4(v[j][0] * inv, v[j][1] * inv, v[j][2] * inv, v[j][3] * inv);
  for (int k = 0; k < K; ++k) {
    const int pos = pos_of[size_t(row) * K + k];
    if (pos < 0) continue;
    uint8_t* qr = xq + size_t(pos) * H;
#pragma unroll
    for (int j = 0; j < BPW; ++j) *reinterpret_cast<unsigned*>(qr + (warp * BPW + j) * 128 + frag_col128(lane)) = q[j];
  }
  if (threadIdx.x == 0) xs[row] = row_scale(amax);
}

// act_perm16 (W4A16 path): x bf16 [T, H] -> xp bf16 [R, H], token t's row copied to its routed rows
// pos_of[t*K + k] (dropped ids skipped), columns of every 64-wide block in the W4A16 logical K order of
// moe_prefill.cu (logical column L holds physical column a16_phys(L)). Values are copied, never rounded.
// One CTA (128 threads) per token: the row is staged in shared memory (physical order), then every thread
// assembles 16-byte logical chunks and stores them to each routed row.
__device__ __forceinline__ int a16_phys_r(int L) {  // == a16_phys (moe_prefill.cu); tests check both orders
  return 16 * ((L >> 1) & 3) + 8 * (L >> 5) + ((2 * ((L >> 4) & 1) + ((L >> 3) & 1) + 4 * (L & 1) + ((L >> 1) & 3)) & 7);
}
__global__ void __launch_bounds__(128) act_perm16_kernel(const __nv_bfloat16* __restrict__ x, int H, int K, const int* __restrict__ pos_of,
                                                        __nv_bfloat16* __restrict__ xp) {
  __shared__ __align__(16) unsigned short s_x[8192];
  __shared__ int s_pos[32];
  pdl_wait();
  const int row = blockIdx.x, tid = threadIdx.x;
  const uint4* xr = reinterpret_cast<const uint4*>(x + size_t(row) * H);
  const int nch = H / 8;
  for (int c = tid; c < nch; c += 128) reinterpret_cast<uint4*>(s_x)[c] = __ldg(xr + c);
  if (tid < K) s_pos[tid] = pos_of ? pos_of[size_t(row) * K + tid] : row;  // (no pos_of: same row -- act_log16)
  __syncthreads();
  for (int c = tid; c < nch; c += 128) {
    // logical chunk c: block b = c >> 3, logical columns 8 (c & 7) .. + 7
    const int base = 64 * (c >> 3), L0 = 8 * (c & 7);
    unsigned u[4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
      u[q] = unsigned(s_x[base + a16_phys_r(L0 + 2 * q)]) | (unsigned(s_x[base + a16_phys_r(L0 + 2 * q + 1)]) << 16);
    const uint4 v = make_uint4(u[0], u[1], u[2], u[3]);
    for (int k = 0; k < K; ++k) {
      const int pos = s_pos[k];
      if (pos >= 0) *reinterpret_cast<uint4*>(xp + size_t(pos) * H + 8 * c) = v;
    }
  }
}

// h_quant: one warp per routed row of I bf16 values (I % 64 == 0, I <= 1280), grid-stride over rows,
// 16-byte loads. Rows >= *nrows are skipped (nrows = off[E] on device, so the launch can use the
// host-side upper bound R). Writes hq [R, I] e4m3 in fragment order (128-blocks, then one 64-block
// if I % 128 == 64) and the row scales: hs[row], or hs[hs_idx[row]] (slot order t*K + k, so that
// moe_combine applies them; GEMM2 then needs no per-token scale in its epilogue).
__global__ void __launch_bounds__(256) h_quant_kernel(const __nv_bfloat16* __restrict__ h, int I, const int* __restrict__ nrows,
                                                      int R, uint8_t* __restrict__ hq, float* __restrict__ hs,
                                                      const int* __restrict__ hs_idx) {
  pdl_wait();
  const int lane = threadIdx.x & 31;
  const int n = min(R, *nrows);
  const int nchunk = I / 8, nfull = I / 128;  // 16-byte chunks per row, full 128-blocks
  for (int row = blockIdx.x * 8 + (threadIdx.x >> 5); row < n; row += gridDim.x * 8) {
    const uint4* hr = reinterpret_cast<const uint4*>(h + size_t(row) * I);
    uint4 raw[5];  // chunk lane + 32 j
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
      const int chk = lane + 32 * j;
      raw[j] = chk < nchunk ? __ldg(hr + chk) : make_uint4(0u, 0u, 0u, 0u);
      const unsigned u[4] = {raw[j].x, raw[j].y, raw[j].z, raw[j].w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = unpack_bf16(u[q]);
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
    amax = warp_max(amax);
    const float inv = row_inv_scale(amax);
    uint8_t* qr = hq + size_t(row) * I;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
      const int chk = lane + 32 * j;
      if (chk < nchunk) {
        const float2 f0 = unpack_bf16(raw[j].x), f1 = unpack_bf16(raw[j].y), f2 = unpack_bf16(raw[j].z), f3 = unpack_bf16(raw[j].w);
        const unsigned q0 = e4m3x4(f0.x * inv, f0.y * inv, f1.x * inv, f1.y * inv);  // physical [8c, 8c+4)
        const unsigned q1 = e4m3x4(f2.x * inv, f2.y * inv, f3.x * inv, f3.y * inv);  // physical [8c+4, 8c+8)
        int base, l, c0, c1;
        if (chk < 16 * nfull) {
          base = (chk >> 4) * 128, l = 2 * (chk & 15);
          c0 = frag_col128(l), c1 = frag_col128(l + 1);
        } else {
          base = nfull * 128, l = 2 * (chk - 16 * nfull);
          c0 = frag_col64(l), c1 = frag_col64(l + 1);
        }
        *reinterpret_cast<unsigned*>(qr + base + c0) = q0;
        *reinterpret_cast<unsigned*>(qr + base + c1) = q1;
      }
    }
    if (lane == 0) hs[hs_idx ? hs_idx[row] : row] = row_scale(amax);
  }
}

// moe_combine: out[t, :] = (addend[t, :] +) sum_k slot[t*K + k, :] (* sscale[t*K + k] if given) over valid expert ids,
// fp32 accumulation in k order (deterministic), one bf16 rounding. One thread per 8 columns. `addend` (optional, bf16
// [T, H]) folds the shared-expert output in: stock rounds the routed sum to bf16 and adds the bf16 shared output with a
// second rounding; one rounding of the fp32 sum is at least as precise.
__global__ void __launch_bounds__(256) moe_combine_kernel(const __nv_bfloat16* __restrict__ slot, const int* __restrict__ ids, int T,
                                                          int K, int H, int E, const float* __restrict__ sscale,
                                                          const __nv_bfloat16* __restrict__ addend, __nv_bfloat16* __restrict__ out) {
  pdl_wait();
  const int per_row = H / 8;
  const long idx = long(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= long(T) * per_row) return;
  const int t = int(idx / per_row), c = int(idx % per_row) * 8;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (addend) add8_bf16(acc, *reinterpret_cast<const uint4*>(addend + size_t(t) * H + c));
  for (int k = 0; k < K; ++k) {
    const int e = ids[size_t(t) * K + k];
    if (unsigned(e) >= unsigned(E)) continue;
    const uint4 v = *reinterpret_cast<const uint4*>(slot + (size_t(t) * K + k) * H + c);
    if (sscale) {
      const float sc = sscale[size_t(t) * K + k];
      const unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = unpack_bf16(u[i]);
        acc[2 * i] = fmaf(f.x, sc, acc[2 * i]);
        acc[2 * i + 1] = fmaf(f.y, sc, acc[2 * i + 1]);
      }
    } else {
      add8_bf16(acc, v);
    }
  }
  *reinterpret_cast<uint4*>(out + size_t(t) * H + c) = pack8_bf16(acc);
}

}  // namespace

// ================================================================================================ host
void ds41h_moe_topk(torch::Tensor logits, torch::Tensor bias, c10::optional<torch::Tensor> bias_vl, c10::optional<torch::Tensor> input_ids,
                    int64_t image_lo, double scale, bool renormalize, torch::Tensor out_w, torch::Tensor out_ids) {
  TORCH_CHECK(logits.dtype() == torch::kFloat32 && logits.dim() == 2);
  const int T = logits.size(0), E = logits.size(1), K = out_ids.size(1);
  TORCH_CHECK(E <= 512 && K <= 32 && K <= E, "moe_topk: E <= 512, K <= 32");
  TORCH_CHECK(bias.dtype() == torch::kFloat32 && bias.numel() == E);
  TORCH_CHECK(out_w.dtype() == torch::kFloat32 && out_ids.dtype() == torch::kInt32 && out_w.size(0) == T && out_ids.size(0) == T);
  const float* bvl = nullptr;
  const int64_t* iid = nullptr;
  if (bias_vl.has_value() && bias_vl->defined()) {
    TORCH_CHECK(input_ids.has_value() && input_ids->dtype() == torch::kInt64 && input_ids->numel() >= T, "bias_vl needs int64 input_ids");
    TORCH_CHECK(bias_vl->dtype() == torch::kFloat32 && bias_vl->numel() == E);
    bvl = cptr<float>(*bias_vl);
    iid = cptr<int64_t>(*input_ids);
  }
  if (T == 0) return;
  const dim3 grid(cdiv(T, 4));
  const int vpl = cdiv(E, 32);
#define DS41H_TOPK(V)                                                                                                                 \
  launch_pdl("moe_topk", moe_topk_kernel<V>, grid, dim3(128), 0, cptr<float>(logits), cptr<float>(bias), bvl, iid, long(image_lo), \
             mptr<float>(out_w), mptr<int>(out_ids), T, E, K, float(scale), int(renormalize))
  if (vpl <= 4)
    DS41H_TOPK(4);
  else if (vpl <= 8)
    DS41H_TOPK(8);
  else if (vpl <= 12)
    DS41H_TOPK(12);
  else
    DS41H_TOPK(16);
#undef DS41H_TOPK
}

int64_t ds41h_moe_meta_size(int64_t E, int64_t R) { return 2 * (E + 1) + META_CTR + 3 * R; }

// meta: int32 [2*(E+1) + META_CTR + 3*R]: counts[E+1] | off[E+1] | ctr[META_CTR] | perm_tok[R] | perm_slot[R] | pos_of[R];
// perm_w f32 [R]
void ds41h_moe_permute(torch::Tensor topk_ids, torch::Tensor topk_w, int64_t E, torch::Tensor meta, torch::Tensor perm_w) {
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32 && topk_w.dtype() == torch::kFloat32 && topk_ids.dim() == 2);
  TORCH_CHECK(topk_ids.sizes() == topk_w.sizes() && topk_ids.is_contiguous() && topk_w.is_contiguous());
  TORCH_CHECK(E >= 1 && E <= MAX_EXPERTS);
  const int R = topk_ids.numel(), K = topk_ids.size(1);
  TORCH_CHECK(meta.dtype() == torch::kInt32 && meta.numel() >= ds41h_moe_meta_size(E, R));
  TORCH_CHECK(perm_w.dtype() == torch::kFloat32 && perm_w.numel() >= R);
  const int G = R > 0 ? cdiv(R, PERM_CHUNK) : 1;
  auto hist = torch::empty({long(G) * E}, meta.options());
  int* m = mptr<int>(meta);
  int* pos_of = m + 2 * (E + 1) + META_CTR + 2 * R;
  launch_pdl("moe_count", moe_count_kernel, dim3(G), dim3(512), 0, cptr<int>(topk_ids), R, int(E), mptr<int>(hist), pos_of);
  launch_pdl("moe_scatter", moe_scatter_kernel, dim3(G), dim3(1024), 0, cptr<int>(topk_ids), cptr<float>(topk_w), R, K, int(E), G,
             cptr<int>(hist), m, m + (E + 1), m + 2 * (E + 1), m + 2 * (E + 1) + META_CTR, m + 2 * (E + 1) + META_CTR + R,
             mptr<float>(perm_w), pos_of);
}

void ds41h_act_quant(torch::Tensor x, torch::Tensor xq, torch::Tensor xs) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2);
  const int T = x.size(0), H = x.size(1);
  TORCH_CHECK(H % 512 == 0 && H <= 8192, "act_quant: H % 512 == 0, H <= 8192");
  TORCH_CHECK(xq.dtype() == torch::kUInt8 && xq.numel() >= long(T) * H && xs.dtype() == torch::kFloat32 && xs.numel() >= T);
  if (T == 0) return;
  const int bpw = H / 512;
#define DS41H_AQ(B) \
  launch_pdl("act_quant", act_quant_kernel<B>, dim3(T), dim3(128), 0, cptr<__nv_bfloat16>(x), H, mptr<uint8_t>(xq), mptr<float>(xs))
  switch (bpw) {
    case 1: DS41H_AQ(1); break;
    case 2: DS41H_AQ(2); break;
    case 4: DS41H_AQ(4); break;
    case 8: DS41H_AQ(8); break;
    case 10: DS41H_AQ(10); break;
    case 14: DS41H_AQ(14); break;
    case 16: DS41H_AQ(16); break;
    default: TORCH_CHECK(false, "act_quant: unsupported H=", H);
  }
#undef DS41H_AQ
}

// xq_p [R, H] (R = T*K, rows in routed order), xs [T] per token; pos_of = the moe_permute meta slice
void ds41h_act_quant_perm(torch::Tensor x, torch::Tensor pos_of, int64_t K, torch::Tensor xq, torch::Tensor xs) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous());
  const int T = x.size(0), H = x.size(1);
  TORCH_CHECK(H % 512 == 0 && H <= 8192, "act_quant_perm: H % 512 == 0, H <= 8192");
  TORCH_CHECK(K >= 1 && pos_of.dtype() == torch::kInt32 && pos_of.numel() >= long(T) * K);
  TORCH_CHECK(xq.dtype() == torch::kUInt8 && xq.numel() >= long(T) * K * H && xs.dtype() == torch::kFloat32 && xs.numel() >= T);
  if (T == 0) return;
  const int bpw = H / 512;
#define DS41H_AQP(B)                                                                                                                   \
  launch_pdl("act_quant_perm", act_quant_perm_kernel<B>, dim3(T), dim3(128), 0, cptr<__nv_bfloat16>(x), H, int(K), cptr<int>(pos_of), \
             mptr<uint8_t>(xq), mptr<float>(xs))
  switch (bpw) {
    case 1: DS41H_AQP(1); break;
    case 2: DS41H_AQP(2); break;
    case 4: DS41H_AQP(4); break;
    case 8: DS41H_AQP(8); break;
    case 10: DS41H_AQP(10); break;
    case 14: DS41H_AQP(14); break;
    case 16: DS41H_AQP(16); break;
    default: TORCH_CHECK(false, "act_quant_perm: unsupported H=", H);
  }
#undef DS41H_AQP
}

// xp [R, H] bf16 (R = T*K, routed-row order, W4A16 logical K order); pos_of = the moe_permute meta slice
void ds41h_act_perm16(torch::Tensor x, torch::Tensor pos_of, int64_t K, torch::Tensor xp) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous());
  const int T = x.size(0), H = x.size(1);
  TORCH_CHECK(H % 64 == 0 && H <= 8192, "act_perm16: H % 64 == 0, H <= 8192");
  TORCH_CHECK(K >= 1 && K <= 32 && pos_of.dtype() == torch::kInt32 && pos_of.numel() >= long(T) * K);
  TORCH_CHECK(xp.dtype() == torch::kBFloat16 && xp.is_contiguous() && xp.numel() >= long(T) * K * H);
  if (T == 0) return;
  launch_pdl("act_perm16", act_perm16_kernel, dim3(T), dim3(128), 0, cptr<__nv_bfloat16>(x), H, int(K), cptr<int>(pos_of),
             mptr<__nv_bfloat16>(xp));
}

// xl [T, H] bf16: x with the W4A16 logical K order (same rows) -- input of the x-gather GEMM1 variants (13 / 14)
void ds41h_act_log16(torch::Tensor x, torch::Tensor xl) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "act_log16: x bf16 [T, H] contiguous");
  const int T = x.size(0), H = x.size(1);
  TORCH_CHECK(H % 64 == 0 && H <= 8192, "act_log16: H % 64 == 0, H <= 8192");
  TORCH_CHECK(xl.dtype() == torch::kBFloat16 && xl.is_contiguous() && xl.numel() >= long(T) * H, "act_log16: xl bf16 >= [T, H]");
  if (T == 0) return;
  launch_pdl("act_log16", act_perm16_kernel, dim3(T), dim3(128), 0, cptr<__nv_bfloat16>(x), H, 1, static_cast<const int*>(nullptr),
             mptr<__nv_bfloat16>(xl));
}

void ds41h_h_quant(torch::Tensor h, torch::Tensor nrows, torch::Tensor hq, torch::Tensor hs, c10::optional<torch::Tensor> hs_idx) {
  TORCH_CHECK(h.dtype() == torch::kBFloat16 && h.dim() == 2 && h.is_contiguous());
  const int R = h.size(0), I = h.size(1);
  TORCH_CHECK(I % 64 == 0 && I <= 1280, "h_quant: I % 64 == 0, I <= 1280");
  TORCH_CHECK(nrows.dtype() == torch::kInt32 && nrows.numel() >= 1);
  TORCH_CHECK(hq.dtype() == torch::kUInt8 && hq.numel() >= long(R) * I && hs.dtype() == torch::kFloat32 && hs.numel() >= R);
  if (R == 0) return;
  static const int nsm = [] {
    int dev = 0, v = 0;
    TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess);
    TORCH_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess);
    return v;
  }();
  const int grid = std::min(cdiv(R, 8), nsm * 8);
  const int* idx = nullptr;
  if (hs_idx.has_value() && hs_idx->defined()) {
    TORCH_CHECK(hs_idx->dtype() == torch::kInt32 && hs_idx->numel() >= R, "h_quant: hs_idx int32 [R]");
    idx = cptr<int>(*hs_idx);  // slot indices t*K + k < R
  }
  launch_pdl("h_quant", h_quant_kernel, dim3(grid), dim3(256), 0, cptr<__nv_bfloat16>(h), I, cptr<int>(nrows), R, mptr<uint8_t>(hq),
             mptr<float>(hs), idx);
}

void ds41h_moe_combine(torch::Tensor slot, torch::Tensor topk_ids, int64_t E, torch::Tensor out, c10::optional<torch::Tensor> sscale,
                       c10::optional<torch::Tensor> addend) {
  TORCH_CHECK(slot.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16 && topk_ids.dtype() == torch::kInt32);
  const int T = topk_ids.size(0), K = topk_ids.size(1), H = out.size(1);
  TORCH_CHECK(H % 8 == 0 && out.size(0) == T && slot.numel() >= long(T) * K * H);
  const float* sc = nullptr;
  if (sscale.has_value() && sscale->defined()) {
    TORCH_CHECK(sscale->dtype() == torch::kFloat32 && sscale->numel() >= long(T) * K, "moe_combine: sscale f32 [T*K]");
    sc = cptr<float>(*sscale);
  }
  const __nv_bfloat16* ad = nullptr;
  if (addend.has_value() && addend->defined()) {
    TORCH_CHECK(addend->dtype() == torch::kBFloat16 && addend->is_contiguous() && addend->dim() == 2 && addend->size(0) == T &&
                    addend->size(1) == H, "moe_combine: addend bf16 contiguous [T, H]");
    ad = cptr<__nv_bfloat16>(*addend);
  }
  if (T == 0) return;
  const long n = long(T) * (H / 8);
  launch_pdl("moe_combine", moe_combine_kernel, dim3(unsigned((n + 255) / 256)), dim3(256), 0, cptr<__nv_bfloat16>(slot), cptr<int>(topk_ids),
             T, K, H, int(E), sc, ad, mptr<__nv_bfloat16>(out));
}
