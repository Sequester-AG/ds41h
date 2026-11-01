// ds41h 'prefill' component kernels (sm_90a).
//
// LOSSLESS (default): dequant_fp8_bf16 (exact bf16 value of a 32x32-block FP8 weight, for the W8A16 prefill linears)
// and inv_rope_bf16_ (stock bf16 inverse RoPE of the attention output, in place, for the bf16 wo_a GEMM); see the
// section near the end of this file.
//
// LOSSY opt-in (DS41H_PREFILL_LINEAR=1 / DS41H_PREFILL_WOA=1): block-FP8 activation quantization for the W8A8 dense
// linears (act_quant_128) and inverse RoPE + FP8 quant for the FP8 wo_a einsum (inv_rope_fp8_quant):
//
// Producer for DeepGEMM's sm90 `fp8_gemm_nt` 1D2D kernel (vendored in vLLM 0.32.0 as
// vllm.third_party.deep_gemm): activations e4m3 [M, KP] (K-major, KP = K rounded up to 128, the
// padding columns are written as +0) with one fp32 scale per (row, 128-wide K block), stored
// MN-major and TMA-aligned exactly like vLLM's `per_token_group_quant_fp8(column_major_scales=True,
// tma_aligned_scales=True)`: element (m, kb) at s[kb * MA + m], MA = round_up(M, 4).
//
// Numerics: scale = 2^ceil(log2(max(amax, 1e-4) / 448)) (UE8M0 = the reference act_quant rounding of
// inference/kernel.py with scale_fmt="ue8m0", here per 128 instead of per 32) or amax / 448 (fp32).
// With a power-of-two scale the bf16 multiply x * 2^-e is exact, so the kernel scales in bf16x2 and
// converts with cvt.rn.satfinite.e4m3x2.f32 (round to nearest even, saturating) -- bit-identical to
// fp32 x / scale -> e4m3 (checked by tests/test_prefill.py::test_act_quant_exact).
//
// Work split: one CTA per row, 8 threads per 128-element block, 16 bf16 (2 x 16-byte loads) per thread,
// one 16-byte store of e4m3 per thread; the block amax is a 3-step xor shuffle inside the 8-lane group.
// Helpers (quant_exp, pow2, e4m3x2) come from common.cuh (portions adapted from BRRR, Apache-2.0).
#include <cuda_bf16.h>

#include "common.cuh"
#include "host.cuh"

namespace {

constexpr int QB = 128;          // quantization block along K (DeepGEMM 1D2D granularity)
constexpr int EPT = 16;          // bf16 elements per thread
constexpr int TPB = QB / EPT;    // threads per block = 8 (an aligned 8-lane group of one warp)
constexpr int MAX_THREADS = 1024;  // KP <= 16384

__device__ __forceinline__ __nv_bfloat162 as_bf162(unsigned u) { return *reinterpret_cast<__nv_bfloat162*>(&u); }
__device__ __forceinline__ unsigned as_u32(__nv_bfloat162 v) { return *reinterpret_cast<unsigned*>(&v); }

// Two e4m3 bytes of a bf16x2 word that is already scaled into the e4m3 range.
__device__ __forceinline__ unsigned e4m3x2_of_bf16x2(unsigned w) {
  return e4m3x2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

template <bool UE8M0>
__global__ void __launch_bounds__(MAX_THREADS) act_quant_128_kernel(const __nv_bfloat16* __restrict__ x, long ldx, int K, int KP,
                                                                    uint8_t* __restrict__ q, float* __restrict__ s, int MA) {
  // One CTA per row (blockIdx.x), 8 threads per 128-wide block: no integer division anywhere.
  const int row = blockIdx.x;
  const int kb = threadIdx.x / TPB;
  const int sub = threadIdx.x % TPB;
  // Threads of the last warp beyond KP/128 still run the (full-mask) shuffles; their 8-lane groups are
  // aligned, so they only exchange with each other, and they store nothing.
  const bool valid = kb < KP / QB;
  const int col = kb * QB + sub * EPT;

  uint4 v0 = make_uint4(0, 0, 0, 0), v1 = make_uint4(0, 0, 0, 0);
  if (valid && col < K) {  // K % 16 == 0 (host check): a thread is either fully inside K or fully padding
    const uint4* p = reinterpret_cast<const uint4*>(x + long(row) * ldx + col);
    v0 = __ldg(p);
    v1 = __ldg(p + 1);
  }
  unsigned w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};

  // block amax (bf16 max is exact)
  __nv_bfloat162 m2 = __habs2(as_bf162(w[0]));
#pragma unroll
  for (int i = 1; i < 8; ++i) m2 = __hmax2(m2, __habs2(as_bf162(w[i])));
  float amax = fmaxf(__low2float(m2), __high2float(m2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));

  unsigned o[4];
  float scale;
  if constexpr (UE8M0) {
    const int e = quant_exp(amax, AMAX_FLOOR);  // 2^e >= max(amax, floor) / 448, e in [-22, 120]
    scale = pow2(e);
    const float inv = pow2(-e);
    const __nv_bfloat162 inv2 = __floats2bfloat162_rn(inv, inv);  // exact: a power of two
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const unsigned a = as_u32(__hmul2(as_bf162(w[2 * i]), inv2));
      const unsigned b = as_u32(__hmul2(as_bf162(w[2 * i + 1]), inv2));
      o[i] = e4m3x2_of_bf16x2(a) | (e4m3x2_of_bf16x2(b) << 16);
    }
  } else {
    const float a = fmaxf(amax, AMAX_FLOOR);
    scale = a * (1.0f / 448.0f);
    const float inv = 448.0f / a;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f0 = unpack_bf16(w[2 * i]), f1 = unpack_bf16(w[2 * i + 1]);
      o[i] = e4m3x2(f0.x * inv, f0.y * inv) | (e4m3x2(f1.x * inv, f1.y * inv) << 16);
    }
  }
  if (valid) {
    *reinterpret_cast<uint4*>(q + long(row) * KP + col) = make_uint4(o[0], o[1], o[2], o[3]);
    if (sub == 0) s[long(kb) * MA + row] = scale;
  }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Inverse RoPE + block FP8 quantization of the attention output for the FP8 wo_a einsum: a faster drop-in for vLLM
// 0.32.0's Triton fused_inv_rope_fp8_quant (models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py, SM90 branch:
// fp32 scales, tma_aligned_scales=False), same semantics: per (token, head) x = o (fp32); dims >= nope_dim are rotated
// back pairwise (even: x * cos + x[d ^ 1] * sin, odd: x * cos - x[d ^ 1] * sin, cos / sin = cache[pos][(d - nope) / 2]
// / [half + (d - nope) / 2]); per 128 block scale = 2^ceil(log2(max(amax, 1e-10) / 448)); q = e4m3(clamp(x / scale)).
// Layout: out [G, T, HPG * 512] e4m3 (group-major), scales [G, T, HPG * 4] fp32 at g * 4 * HPG * TA + t + c * TA
// (TA = T rounded up to 4; rows t in [T, TA) get zero scales). One warp per (token, head), lane l = dims 16l..16l+15.
template <int HPG>
__global__ void __launch_bounds__(HPG * 2 * 32) inv_rope_fp8_quant_kernel(const __nv_bfloat16* __restrict__ o, long o_stride_t,
                                                                           long o_stride_h, const int64_t* __restrict__ positions,
                                                                           const float* __restrict__ cache, long cache_stride, int T,
                                                                           int TA, uint8_t* __restrict__ out, float* __restrict__ scales) {
  constexpr int NOPE = 448, HALF = 32, D = 512, CH = D / 128;
  const int t = blockIdx.x;
  const int head = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = head / HPG, hg = head % HPG;
  const int sub = lane & 7, chunk = lane >> 3;
  float* srow = scales + long(g) * (HPG * CH) * TA + t + long(hg * CH + chunk) * TA;
  if (t >= T) {
    if (sub == 0) *srow = 0.0f;
    return;
  }
  const uint4* src = reinterpret_cast<const uint4*>(o + long(t) * o_stride_t + long(head) * o_stride_h + 16 * lane);
  const uint4 v0 = __ldg(src), v1 = __ldg(src + 1);
  float x[16];
  {
    const unsigned w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float2 f = unpack_bf16(w[i]);
      x[2 * i] = f.x;
      x[2 * i + 1] = f.y;
    }
  }
  if (16 * lane >= NOPE) {  // lanes 28..31: dims 448..511, pairs (2k, 2k + 1) inside the lane
    const long pos = positions[t];
    const int c0 = (16 * lane - NOPE) / 2;  // first cos / sin index of this lane (8 of them)
    const float4* cs = reinterpret_cast<const float4*>(cache + pos * cache_stride + c0);
    const float4* sn = reinterpret_cast<const float4*>(cache + pos * cache_stride + HALF + c0);
    const float4 ca = __ldg(cs), cb = __ldg(cs + 1), sa = __ldg(sn), sb = __ldg(sn + 1);
    const float cv[8] = {ca.x, ca.y, ca.z, ca.w, cb.x, cb.y, cb.z, cb.w};
    const float sv[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float e = x[2 * k], od = x[2 * k + 1];
      x[2 * k] = fmaf(od, sv[k], e * cv[k]);
      x[2 * k + 1] = fmaf(-e, sv[k], od * cv[k]);
    }
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
  const int e = quant_exp(amax, 1e-10f);  // 2^e = 2^ceil(log2(max(amax, 1e-10) / 448))
  const float inv = pow2(-e);
  unsigned q[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    q[i] = e4m3x2(x[4 * i] * inv, x[4 * i + 1] * inv) | (e4m3x2(x[4 * i + 2] * inv, x[4 * i + 3] * inv) << 16);
  uint8_t* dst = out + (long(g) * T + t) * (HPG * D) + hg * D + 16 * lane;
  *reinterpret_cast<uint4*>(dst) = make_uint4(q[0], q[1], q[2], q[3]);
  if (sub == 0) *srow = pow2(e);
}

// o bf16 [T, G * HPG, 512] (unit stride on the last dim, 16-byte aligned rows), positions int64 [T],
// cache fp32 [max_pos, 64] -> (out e4m3 [T, G, HPG * 512] view of [G, T, ...], scales fp32 [T, G, HPG * 4] view)
std::vector<torch::Tensor> inv_rope_fp8_quant(torch::Tensor o, torch::Tensor positions, torch::Tensor cache, int64_t n_groups) {
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == torch::kBFloat16 && o.dim() == 3 && o.size(2) == 512 && o.stride(2) == 1,
              "o must be bf16 [T, H, 512]");
  TORCH_CHECK(o.stride(1) % 8 == 0 && o.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0, "o alignment");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 && positions.numel() >= o.size(0), "positions");
  TORCH_CHECK(cache.is_cuda() && cache.scalar_type() == torch::kFloat32 && cache.dim() == 2 && cache.size(1) == 64 && cache.stride(1) == 1 &&
                  cache.stride(0) % 4 == 0, "cos_sin_cache must be fp32 [max_pos, 64]");
  const long T = o.size(0), H = o.size(1);
  TORCH_CHECK(H % n_groups == 0 && H / n_groups == 8, "this kernel is built for 8 heads per wo_a group");
  const long G = n_groups, TA = (T + 3) / 4 * 4;
  auto out = torch::empty({G, T, 8 * 512}, o.options().dtype(torch::kFloat8_e4m3fn));
  auto sbuf = torch::empty({G * 32 * TA}, o.options().dtype(torch::kFloat32));
  auto sc = sbuf.as_strided({G, T, 32}, {32 * TA, 1, TA});
  if (T > 0) {
    const dim3 grid(static_cast<unsigned>(TA), static_cast<unsigned>(1));
    // one CTA per token: G * 8 warps (H <= 32)
    TORCH_CHECK(H <= 32, "at most 32 heads");
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (G == 2) {
      inv_rope_fp8_quant_kernel<8><<<grid, dim3(2 * 8 * 32), 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()), o.stride(0), o.stride(1), positions.data_ptr<int64_t>(),
          cache.data_ptr<float>(), cache.stride(0), int(T), int(TA), reinterpret_cast<uint8_t*>(out.data_ptr()), sbuf.data_ptr<float>());
    } else {
      TORCH_CHECK(false, "only 2 wo_a groups per rank (TP4) are instantiated");
    }
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "inv_rope_fp8_quant launch failed: ", cudaGetErrorString(err));
  }
  return {out.transpose(0, 1), sc.transpose(0, 1)};
}

// x bf16 [M, K] (unit stride along K, row stride % 8 == 0, 16-byte aligned) ->
//   q  float8_e4m3fn [M, KP] (contiguous, columns [K, KP) = +0)
//   s  float32 [M, KP/128] view with strides (1, MA) of a [KP/128 * MA] buffer (MN-major, TMA aligned)
std::vector<torch::Tensor> act_quant_128(torch::Tensor x, int64_t kp, bool ue8m0) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2, "x must be a 2-D CUDA bf16 tensor");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "x rows must be unit-stride, 16-byte aligned with a row stride multiple of 8");
  const long M = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16");
  TORCH_CHECK(kp % QB == 0 && kp >= K && kp - K < QB, "kp must be K rounded up to a multiple of 128");
  TORCH_CHECK(M < (1L << 31) / 4 && kp < (1L << 30), "shape too large");
  const long MA = (M + 3) / 4 * 4;
  auto q = torch::empty({M, kp}, x.options().dtype(torch::kFloat8_e4m3fn));
  auto sbuf = torch::empty({(kp / QB) * MA}, x.options().dtype(torch::kFloat32));
  auto s = sbuf.as_strided({M, kp / QB}, {1, MA});
  if (M == 0) return {q, s};
  TORCH_CHECK(kp <= MAX_THREADS / TPB * QB, "kp too large for one CTA per row");
  const int threads = int((kp / QB * TPB + 31) / 32 * 32);
  const dim3 grid(static_cast<unsigned>(M));
  auto stream = c10::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* qp = reinterpret_cast<uint8_t*>(q.data_ptr());
  if (ue8m0)
    act_quant_128_kernel<true><<<grid, threads, 0, stream>>>(xp, x.stride(0), int(K), int(kp), qp, sbuf.data_ptr<float>(), int(MA));
  else
    act_quant_128_kernel<false><<<grid, threads, 0, stream>>>(xp, x.stride(0), int(K), int(kp), qp, sbuf.data_ptr<float>(), int(MA));
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "act_quant_128 launch failed: ", cudaGetErrorString(err));
  return {q, s};
}

// ---------------------------------------------------------------------------------------------------------------
// LOSSLESS W8A16 support (default prefill path, see ds41h/linear.py "LINEAR16").
//
// dequant_fp8_bf16: exact bf16 value of a checkpoint FP8 weight, w e4m3 [N, K] with one ue8m0 byte S per 32x32 block
// (s [ceil(N/32), K/32]): value = e4m3 * 2^(S - 127). e4m3 -> f16 (cvt.rn.f16x2.e4m3x2: every e4m3 value is a normal
// f16 or zero) -> f32 (exact) -> * 2^(S - 127) (a power of two: exact while the product stays a normal f32, i.e.
// 10 <= S <= 246, checked on the host at capture time) -> bf16 (cvt.rn: exact, <= 4 significant bits inside bf16's
// exponent range). So bf16 GEMM(x, dequant(w)) with fp32 accumulation is exactly W8A16 math up to summation order.
// One thread per 16 consecutive K elements (16 B in, 32 B out; inside one 32-block), grid (ceil(K/16/128), N).
__global__ void __launch_bounds__(128) dequant_fp8_bf16_kernel(const uint8_t* __restrict__ w, long ldw, const uint8_t* __restrict__ s,
                                                                long lds, int K, __nv_bfloat16* __restrict__ out, long ldo) {
  const int c = blockIdx.x * 128 + threadIdx.x;  // 16-element chunk index along K
  const int n = blockIdx.y;
  if (c * 16 >= K) return;
  const uint4 v = __ldg(reinterpret_cast<const uint4*>(w + long(n) * ldw + c * 16));
  const float f = e8m0_to_float(__ldg(s + long(n >> 5) * lds + (c >> 1)));
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
  unsigned o[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>((u[i] >> (16 * h)) & 0xffffu), __NV_E4M3);
      const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&hr));
      o[2 * i + h] = pack_bf16(x.x * f, x.y * f);
    }
  }
  uint4* dst = reinterpret_cast<uint4*>(out + long(n) * ldo + c * 16);
  dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
  dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

// w uint8/e4m3 [N, K] (unit stride along K, row stride % 16 == 0, 16-byte aligned), s uint8 [ceil(N/32), K/32]
// (row stride arbitrary) -> out bf16 [N, K] (allocated if not given; unit stride along K, row stride % 8 == 0).
torch::Tensor dequant_fp8_bf16(torch::Tensor w, torch::Tensor s, c10::optional<torch::Tensor> out_) {
  TORCH_CHECK(w.is_cuda() && w.element_size() == 1 && w.dim() == 2 && w.stride(1) == 1, "w must be a 2-D CUDA 1-byte tensor");
  TORCH_CHECK(w.stride(0) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "w rows must be 16-byte aligned");
  const long N = w.size(0), K = w.size(1);
  TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32");
  TORCH_CHECK(s.is_cuda() && s.element_size() == 1 && s.dim() == 2 && s.size(0) == (N + 31) / 32 && s.size(1) == K / 32 &&
                  s.stride(1) == 1, "s must be uint8 [ceil(N/32), K/32]");
  torch::Tensor out = out_.has_value() ? *out_ : torch::empty({N, K}, w.options().dtype(torch::kBFloat16));
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.size(0) == N && out.size(1) == K &&
                  out.stride(1) == 1 && out.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "out must be bf16 [N, K] with 16-byte aligned rows");
  if (N == 0 || K == 0) return out;
  TORCH_CHECK(N < 65536, "N too large for grid.y");
  const dim3 grid(static_cast<unsigned>((K / 16 + 127) / 128), static_cast<unsigned>(N));
  dequant_fp8_bf16_kernel<<<grid, 128, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(w.data_ptr()), w.stride(0), reinterpret_cast<const uint8_t*>(s.data_ptr()), s.stride(0), int(K),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0));
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "dequant_fp8_bf16 launch failed: ", cudaGetErrorString(err));
  return out;
}

// inv_rope_bf16_: the stock BF16 wo_a input (vLLM's Triton fused_inv_rope_fp8_quant with quantize=False: x fp32, dims
// >= 448 rotated back pairwise, even: x * cos + x[d ^ 1] * sin, odd: x * cos - x[d ^ 1] * sin, rounded to bf16) IN PLACE
// on the attention output: the 448 no-RoPE dims of each head are copied unchanged by stock, so only the 64 RoPE dims
// are read and written (1/8 of stock's traffic, no [G, T, 4096] copy). The caller passes the attention output, which
// nothing reads after the o-projection. One thread per 4 (even, odd) pairs, 8 threads per head, one CTA per token.
__global__ void __launch_bounds__(256) inv_rope_bf16_kernel(__nv_bfloat16* __restrict__ o, long o_stride_t, long o_stride_h, int H,
                                                            const int64_t* __restrict__ positions, const float* __restrict__ cache,
                                                            long cache_stride) {
  constexpr int NOPE = 448, HALF = 32;
  const int t = blockIdx.x;
  const int head = threadIdx.x >> 3, j = threadIdx.x & 7;  // pairs 4j .. 4j + 3 of this head's 32
  if (head >= H) return;
  uint4* p = reinterpret_cast<uint4*>(o + long(t) * o_stride_t + long(head) * o_stride_h + NOPE + 8 * j);
  const uint4 v = *p;
  const long pos = positions[t];
  const float4 cv = __ldg(reinterpret_cast<const float4*>(cache + pos * cache_stride + 4 * j));
  const float4 sv = __ldg(reinterpret_cast<const float4*>(cache + pos * cache_stride + HALF + 4 * j));
  const unsigned u[4] = {v.x, v.y, v.z, v.w};
  const float c[4] = {cv.x, cv.y, cv.z, cv.w}, sn[4] = {sv.x, sv.y, sv.z, sv.w};
  unsigned r[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 x = unpack_bf16(u[k]);  // (even, odd) pair k
    // Triton contracts `x * cos + partner * sin` as fma(x, cos, partner * sin) (first product fused).
    const float ev = __fmaf_rn(x.x, c[k], __fmul_rn(x.y, sn[k]));
    const float od = __fmaf_rn(x.y, c[k], -__fmul_rn(x.x, sn[k]));
    r[k] = pack_bf16(ev, od);
  }
  *p = make_uint4(r[0], r[1], r[2], r[3]);
}

// o bf16 [T, H, 512] (unit stride on the last dim, 16-byte aligned head rows), positions int64 [>= T],
// cache fp32 [max_pos, 64] (cos | sin) -> o with the RoPE dims rotated back (in place).
void inv_rope_bf16_(torch::Tensor o, torch::Tensor positions, torch::Tensor cache) {
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == torch::kBFloat16 && o.dim() == 3 && o.size(2) == 512 && o.stride(2) == 1,
              "o must be bf16 [T, H, 512]");
  TORCH_CHECK(o.stride(1) % 8 == 0 && o.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0, "o alignment");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 && positions.numel() >= o.size(0) &&
                  (positions.dim() == 1 && positions.stride(0) == 1), "positions must be contiguous int64 [>= T]");
  TORCH_CHECK(cache.is_cuda() && cache.scalar_type() == torch::kFloat32 && cache.dim() == 2 && cache.size(1) == 64 && cache.stride(1) == 1 &&
                  cache.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(cache.data_ptr()) % 16 == 0,
              "cos_sin_cache must be fp32 [max_pos, 64]");
  const long T = o.size(0), H = o.size(1);
  TORCH_CHECK(H >= 1 && H <= 32, "1..32 heads");
  if (T == 0) return;
  inv_rope_bf16_kernel<<<dim3(static_cast<unsigned>(T)), dim3(static_cast<unsigned>(H * 8)), 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(o.data_ptr()), o.stride(0), o.stride(1), int(H), positions.data_ptr<int64_t>(), cache.data_ptr<float>(),
      cache.stride(0));
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "inv_rope_bf16_ launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant_fp8_bf16", &dequant_fp8_bf16, "exact bf16 value of an e4m3 [N,K] weight with 32x32 ue8m0 block scales",
        pybind11::arg("w"), pybind11::arg("s"), pybind11::arg("out") = pybind11::none());
  m.def("inv_rope_bf16_", &inv_rope_bf16_, "in-place inverse RoPE of the RoPE dims of the attention output (stock bf16 wo_a input)",
        pybind11::arg("o"), pybind11::arg("positions"), pybind11::arg("cache"));
  m.def("inv_rope_fp8_quant", &inv_rope_fp8_quant, "inverse RoPE + 1x128 FP8 quant of the attention output (wo_a input)",
        pybind11::arg("o"), pybind11::arg("positions"), pybind11::arg("cache"), pybind11::arg("n_groups"));
  m.def("act_quant_128", &act_quant_128, "bf16 [M,K] -> (e4m3 [M,KP], fp32 MN-major scales [M,KP/128]) per 1x128 block",
        pybind11::arg("x"), pybind11::arg("kp"), pybind11::arg("ue8m0") = true);
}
