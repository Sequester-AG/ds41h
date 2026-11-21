// indexer2f: topk4 = idx2f_topk.cu's exact top-K (FROZEN served source, its
// algorithm unchanged) in SMALLER CTAs, so several rows run per SM.
//
// Why: the served topk_kernel runs one 1,024-thread CTA with 197 KB of static shared memory per row -> 1 CTA per SM, and
// a row is ~7 radix passes whose cost is barrier latency over 32 warps (~14 us per row at 16-24 K positions; 2,048
// prefill rows = ~220 us per index layer, 3.3x the logits read). Here:
//   * TPB threads per row (template: 256 / 512 / 1,024): per-warp histograms, scans and barriers over TPB / 32 warps;
//   * dynamic shared memory sized by the call's widest row (the block-key array holds max(nb_max, 2 kCMax) keys: the
//     candidates alias it exactly as in v3), so a 64 K-position call needs ~105 KB at 256 threads -> 2 rows per SM.
// Output contract and arithmetic: identical to topk_kernel (the EXACT top-K under (key desc, lowest position on ties),
// ascending position order, n <= K rows 0 .. n-1 then -1, cblk position map, the same kCMax overflow path streamed over
// the row, the same `slow` counter). The result is a pure function of the row, so it is bitwise the served output; the
// test checks that on random / tie-heavy / NaN / inf / short rows.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cub/block/block_scan.cuh>

namespace idx2f_topk4 {

constexpr int kBlock = 8;
constexpr int kCMax = 8192;
constexpr int kBins = 256;
constexpr int kMaxBlocks = 32768;   // rows up to 262,144 positions

__device__ __forceinline__ uint32_t ukey(float f) {
  const uint32_t b = __float_as_uint(f);
  if ((b & 0x7fffffffu) > 0x7f800000u) return 0xffffffffu;  // NaN: highest
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

template <int TPB>
struct Sm {
  using Scan = cub::BlockScan<int, TPB>;
  static constexpr int kWarps = TPB / 32;
  uint32_t* a;      // [na] block keys, then the candidates: key [0, kCMax), pos [kCMax, 2 kCMax)
  int* sblk;        // [kCMax]
  int* whist;       // [kWarps * kBins]
  int* hist;        // [kBins]
  int* misc;        // [4]
  typename Scan::TempStorage* scan;
  static __host__ __device__ size_t bytes(int na) {
    size_t b = size_t(na) * 4 + size_t(kCMax) * 4 + size_t(kWarps) * kBins * 4 + kBins * 4 + 16;
    b = (b + 15) & ~size_t(15);
    return b + sizeof(typename Scan::TempStorage);
  }
  __device__ Sm(uint8_t* raw, int na) {
    a = reinterpret_cast<uint32_t*>(raw);
    sblk = reinterpret_cast<int*>(a + na);
    whist = sblk + kCMax;
    hist = whist + kWarps * kBins;
    misc = hist + kBins;
    size_t off = size_t(reinterpret_cast<uint8_t*>(misc + 4) - raw);
    off = (off + 15) & ~size_t(15);
    scan = reinterpret_cast<typename Scan::TempStorage*>(raw + off);
  }
};

template <int TPB, typename Get>
__device__ __forceinline__ void radix_pass(Sm<TPB>& sm, int N, Get get, uint32_t prefix, uint32_t pmask, int sh, int need) {
  using Scan = typename Sm<TPB>::Scan;
  constexpr int kWarps = Sm<TPB>::kWarps;
  const int tid = threadIdx.x, w = tid >> 5;
  for (int i = tid; i < kWarps * kBins; i += TPB) sm.whist[i] = 0;
  __syncthreads();
  int* wh = sm.whist + w * kBins;
  for (int i = tid; i < N; i += TPB) {
    const uint32_t u = get(i);
    if ((u & pmask) == prefix) atomicAdd(&wh[(u >> sh) & 255u], 1);
  }
  __syncthreads();
  for (int b = tid; b < kBins; b += TPB) {
    int c = 0;
#pragma unroll 8
    for (int k = 0; k < kWarps; ++k) c += sm.whist[k * kBins + b];
    sm.hist[b] = c;
  }
  __syncthreads();
  // suffix scan over the 256 bins (descending): item t < 128 owns bins 255 - 2t, 254 - 2t; TPB >= 128
  static_assert(TPB >= kBins / 2, "TPB >= 128");
  int c0 = 0, c1 = 0, hb = 0;
  if (tid < kBins / 2) hb = kBins - 1 - 2 * tid, c0 = sm.hist[hb], c1 = sm.hist[hb - 1];
  int incl, tot;
  Scan(*sm.scan).InclusiveSum(c0 + c1, incl, tot);
  const int excl = incl - c0 - c1;
  if (tid < kBins / 2 && excl < need && need <= incl) {
    if (excl + c0 >= need) sm.misc[0] = hb, sm.misc[1] = excl;
    else sm.misc[0] = hb - 1, sm.misc[1] = excl + c0;
  }
  __syncthreads();
}

template <int TPB, typename Get>
__device__ uint32_t kth_key(Sm<TPB>& sm, int N, int K, Get get, int& gt, int npass = 4) {
  uint32_t prefix = 0, pmask = 0;
  int need = K, above_total = 0;
  for (int pass = 0; pass < npass; ++pass) {
    const int sh = 24 - 8 * pass;
    radix_pass<TPB>(sm, N, get, prefix, pmask, sh, need);
    const int d = sm.misc[0], above = sm.misc[1];
    __syncthreads();
    prefix |= uint32_t(d) << sh;
    pmask |= 255u << sh;
    need -= above;
    above_total += above;
  }
  gt = above_total;
  return prefix;
}

template <int TPB, typename Get, typename Emit>
__device__ void emit_sel(Sm<TPB>& sm, int N, uint32_t kth, int need_eq, Get get, Emit val, int* out) {
  using Scan = typename Sm<TPB>::Scan;
  const int chunk = (N + TPB - 1) / TPB;
  const int i0 = min(N, int(threadIdx.x) * chunk), i1 = min(N, i0 + chunk);
  int g = 0, e = 0;
  for (int i = i0; i < i1; ++i) {
    const uint32_t u = get(i);
    g += u > kth;
    e += u == kth;
  }
  int gb, eb, tot;
  Scan(*sm.scan).ExclusiveSum(g, gb, tot);
  __syncthreads();
  Scan(*sm.scan).ExclusiveSum(e, eb, tot);
  for (int i = i0; i < i1; ++i) {
    const uint32_t u = get(i);
    if (u > kth) {
      out[gb + min(eb, need_eq)] = val(i);
      ++gb;
    } else if (u == kth) {
      if (eb < need_eq) out[gb + eb] = val(i);
      ++eb;
    }
  }
}

struct Args {
  const float* lg;
  long ld;
  const int* lens;
  const float* bmax;
  long ldb;
  int* out;
  long ldo;
  const int* cblk;
  long ldc;
  int K;
  int* slow;
  int na;   // block-key array length (>= the widest row's blocks and >= 2 kCMax)
};

template <int TPB>
__global__ void __launch_bounds__(TPB) topk4_kernel(const Args a) {
  using Scan = typename Sm<TPB>::Scan;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  Sm<TPB> sm(smem_raw, a.na);
  const int row = blockIdx.x, tid = threadIdx.x;
  const int n = __ldg(a.lens + row);
  const int K = a.K;
  int* out = a.out + row * a.ldo;
  const int* cblk = a.cblk ? a.cblk + row * a.ldc : nullptr;
  auto posmap = [&](int p) { return cblk ? ((__ldg(cblk + (p >> 3)) << 3) | (p & 7)) : p; };
  if (n <= K) {
    for (int i = tid; i < K; i += TPB) out[i] = i < n ? posmap(i) : -1;
    return;
  }
  const float* lrow = a.lg + row * a.ld;
  const float* brow = a.bmax + row * a.ldb;
  const int nb = (n + kBlock - 1) / kBlock;

  for (int i = tid; i < nb; i += TPB) sm.a[i] = ukey(__ldg(brow + i));
  __syncthreads();
  uint32_t tau = 0;
  if (nb > K) {
    int gt_unused;
    tau = kth_key<TPB>(sm, nb, K, [&](int i) { return sm.a[i]; }, gt_unused, 3);
  }
  bool overflow = false;
  {
    const int chunk = (nb + TPB - 1) / TPB;
    const int b0 = min(nb, tid * chunk), b1 = min(nb, b0 + chunk);
    int cnt = 0;
    for (int b = b0; b < b1; ++b) cnt += sm.a[b] >= tau;
    int off, S;
    Scan(*sm.scan).ExclusiveSum(cnt, off, S);
    __syncthreads();
    if (S > kCMax) {
      overflow = true;
    } else {
      for (int b = b0; b < b1; ++b)
        if (sm.a[b] >= tau) sm.sblk[off++] = b;
    }
    __syncthreads();
    if (!overflow) sm.misc[2] = S;
    __syncthreads();
  }
  int C = 0;
  uint32_t* ckey = sm.a;
  int* cpos = reinterpret_cast<int*>(sm.a + kCMax);
  if (!overflow) {
    const int S = sm.misc[2];
    for (int r0 = 0; r0 < S; r0 += TPB) {
      const int i = r0 + tid;
      uint32_t u[8];
      int b = 0, m = 0, c = 0;
      if (i < S) {
        b = sm.sblk[i];
        m = min(kBlock, n - kBlock * b);
        const float4* p = reinterpret_cast<const float4*>(lrow + long(kBlock) * b);
        const float4 x0 = __ldg(p), x1 = __ldg(p + 1);
        const float v[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          u[j] = ukey(v[j]);
          c += (j < m) && u[j] >= tau;
        }
      }
      int off, tot;
      Scan(*sm.scan).ExclusiveSum(c, off, tot);
      if (C + tot > kCMax) {
        overflow = true;
        break;
      }
      if (i < S) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if ((j < m) && u[j] >= tau) ckey[C + off] = u[j], cpos[C + off] = kBlock * b + j, ++off;
      }
      C += tot;
      __syncthreads();
    }
  }
  __syncthreads();
  if (!overflow) {
    auto get = [&](int i) { return ckey[i]; };
    int gt;
    const uint32_t kth = kth_key<TPB>(sm, C, K, get, gt);
    emit_sel<TPB>(sm, C, kth, K - gt, get, [&](int i) { return posmap(cpos[i]); }, out);
  } else {
    if (tid == 0) atomicAdd(a.slow, 1);
    auto get = [&](int i) { return ukey(__ldg(lrow + i)); };
    int gt;
    const uint32_t kth = kth_key<TPB>(sm, n, K, get, gt);
    emit_sel<TPB>(sm, n, kth, K - gt, get, [&](int i) { return posmap(i); }, out);
  }
}

template <int TPB>
void launch(const Args& k, int rows, cudaStream_t st, int dev) {
  static bool set[64] = {false};
  const size_t smem = Sm<TPB>::bytes(k.na);
  TORCH_CHECK(smem <= 232448, "topk4: rows too wide for shared memory at TPB ", TPB);
  if (!set[dev & 63]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(topk4_kernel<TPB>, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448));
    set[dev & 63] = true;
  }
  topk4_kernel<TPB><<<rows, TPB, smem, st>>>(k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace idx2f_topk4

static void check_rows(const at::Tensor& t, const char* name, at::ScalarType ty) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == ty && t.dim() == 2 && t.stride(1) == 1, name, ": 2-D, unit inner stride");
}

// = idx2f_topk.topk (same arguments, minus `stop`) + tpb; nb_max (0 = bmax.size(1)) bounds the rows' blocks
void topk4(at::Tensor logits, at::Tensor lens, at::Tensor bmax, int64_t k, at::Tensor out, std::optional<at::Tensor> cblk,
           at::Tensor slow, int64_t tpb, int64_t nb_max) {
  using namespace idx2f_topk4;
  check_rows(logits, "logits", at::kFloat);
  check_rows(bmax, "bmax", at::kFloat);
  check_rows(out, "out", at::kInt);
  TORCH_CHECK(lens.is_cuda() && lens.scalar_type() == at::kInt && lens.is_contiguous(), "lens");
  TORCH_CHECK(slow.is_cuda() && slow.scalar_type() == at::kInt && slow.numel() >= 1, "slow counter");
  const int rows = int(logits.size(0));
  TORCH_CHECK(lens.numel() >= rows && bmax.size(0) >= rows && out.size(0) >= rows && out.size(1) >= k && k > 0 && k <= kCMax,
              "shapes");
  TORCH_CHECK(bmax.size(1) <= kMaxBlocks, "rows up to 262,144 positions");
  TORCH_CHECK(logits.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0, "logits rows 16-B aligned");
  if (cblk) check_rows(*cblk, "cblk", at::kInt);
  if (!rows) return;
  const c10::cuda::CUDAGuard guard(logits.device());
  const int nbm = nb_max > 0 ? int(nb_max) : int(bmax.size(1));
  Args a{logits.data_ptr<float>(), logits.stride(0), lens.data_ptr<int>(), bmax.data_ptr<float>(), bmax.stride(0),
         out.data_ptr<int>(), out.stride(0), cblk ? cblk->data_ptr<int>() : nullptr, cblk ? cblk->stride(0) : 0, int(k),
         slow.data_ptr<int>(), std::max(nbm, 2 * kCMax)};
  const cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const int dev = logits.get_device();
  if (tpb == 256) launch<256>(a, rows, st, dev);
  else if (tpb == 512) launch<512>(a, rows, st, dev);
  else if (tpb == 1024) launch<1024>(a, rows, st, dev);
  else TORCH_CHECK(false, "tpb in {256, 512, 1024}");
}

int64_t smem_bytes(int64_t tpb, int64_t nb_max) {
  using namespace idx2f_topk4;
  const int na = std::max(int(nb_max), 2 * kCMax);
  return tpb == 256 ? Sm<256>::bytes(na) : tpb == 512 ? Sm<512>::bytes(na) : Sm<1024>::bytes(na);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk4", &topk4, "exact top-K (idx2f_topk.topk's contract) in TPB-thread CTAs", py::arg("logits"), py::arg("lens"),
        py::arg("bmax"), py::arg("k"), py::arg("out"), py::arg("cblk"), py::arg("slow"), py::arg("tpb") = 256,
        py::arg("nb_max") = 0);
  m.def("smem_bytes", &smem_bytes);
}
