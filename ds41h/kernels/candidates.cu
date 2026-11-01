// ds41h CANDIDATES: radix-select kernels for the DeepSeek-V4.1 two-level sparse-attention indexer.
//
// Radix select (4 passes x 8 bits over an order-preserving u32 key) adapted from BRRR
// (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0, src/brrr/kernels/candidates.cu).
// See LICENSES/BRRR-Apache-2.0.txt.
//
// Kernels (all CUDA-graph safe: no allocation, no host sync, all bounds read on device):
//
//  select_blocks  == stock vllm/model_executor/kernels/attention/dsa/candidate_blocks.py:141-189
//                    (_block_scores_kernel + scores.topk + _store_candidates_kernel) fused into one launch.
//                    Block score = NaN-propagating max of the block's in-bounds logits (stock :37-45),
//                    the newest block is pinned to +inf (stock :46-50), top min(k, nblocks) blocks by
//                    torch.topk key order (NaN > +inf > ... > -inf), ties -> lowest block index (the order
//                    torch's radix gatherTopK fills ties in), then -inf entries are written as -1 (stock :69-73).
//                    Output order: ascending block index (stock: descending score). The only consumer,
//                    apply_candidate_mask, is order-insensitive (it sets per-block flags, stock :92-105).
//  mask_candidates == stock apply_candidate_mask (candidate_blocks.py:192-235: _candidate_flags_kernel +
//                    _mask_candidates_kernel) fused; block flags are rebuilt in shared memory per CTA, so no
//                    [rows, nblocks+1] flags tensor is allocated.
//  topk_rows      == top-k over each row's [0, row_end) (stock "torch" indexer top-k semantics,
//                    layers/indexer_topk.py:359-371), output in ascending position order, -1 padded; optional
//                    x8 expand (BRRR select_sorted EXPAND) for block-granular candidates.
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <torch/extension.h>

#include "common.cuh"

namespace {

// order(-inf): keys strictly above this are finite, +inf or NaN.
constexpr unsigned ORDER_NEG_INF = 0x007fffffu;  // ~0xff800000
constexpr unsigned ORDER_POS_INF = 0xff800000u;  // 0x7f800000 | sign bit

// torch.topk key order (aten TopKTypeConfig<float>::convert): all NaNs map to the maximum key.
// Pure integer math on the bits: --use_fast_math (ftz) must not flush subnormals or alter NaN tests,
// so scores are compared and max-reduced as keys, never as floats.
__device__ __forceinline__ unsigned topk_key_bits(unsigned bits) {
  return (bits & 0x7fffffffu) > 0x7f800000u ? 0xffffffffu : ((bits & 0x80000000u) ? ~bits : bits | 0x80000000u);
}
__device__ __forceinline__ unsigned topk_key(float v) { return topk_key_bits(__float_as_uint(v)); }

// ------------------------------------------------------------------------------------------------
// Block-wide helpers. THREADS is a multiple of 32; all threads of the CTA must call these.
// ------------------------------------------------------------------------------------------------
template <int THREADS>
struct Radix {
  static constexpr int WARPS = THREADS / 32;
  using Scan = cub::BlockScan<int, THREADS>;
  using Reduce = cub::BlockReduce<int, THREADS>;
  struct Smem {
    unsigned whist[WARPS][256];  // per-warp histograms (atomic contention only inside a warp)
    unsigned hist[256];
    int warp_total[8];
    unsigned chosen[4];  // bin, remaining after the bin, count of the bin, broadcast scratch
    union {
      typename Scan::TempStorage scan;
      typename Reduce::TempStorage reduce;
    };
    int table[2][THREADS];  // per (item, warp) exclusive offsets for index-ordered ranking
  };

  static __device__ __forceinline__ int block_sum(Smem& s, int v, int tid) {
    const int total = Reduce(s.reduce).Sum(v);
    if (tid == 0) s.chosen[3] = unsigned(total);
    __syncthreads();
    const int r = int(s.chosen[3]);
    __syncthreads();
    return r;
  }

  // Pick the bin (scanning from bin 255 down) where the descending cumulative count reaches `remaining`.
  static __device__ __forceinline__ void pick_bin(Smem& s, int remaining, int tid) {
    int count = 0, inclusive = 0;
    if (tid < 256) {
      count = int(s.hist[255 - tid]);
      inclusive = count;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int up = __shfl_up_sync(0xffffffffu, inclusive, o);
        if ((tid & 31) >= o) inclusive += up;
      }
      if ((tid & 31) == 31) s.warp_total[tid >> 5] = inclusive;
    }
    __syncthreads();
    if (tid < 256) {
      for (int w = 0; w < (tid >> 5); ++w) inclusive += s.warp_total[w];
      const int before = inclusive - count;
      if (before < remaining && inclusive >= remaining) {
        s.chosen[0] = unsigned(255 - tid);
        s.chosen[1] = unsigned(remaining - before);
        s.chosen[2] = unsigned(count);
      }
    }
    __syncthreads();
  }

  // Key of the k-th largest candidate (1 <= k <= #candidates). visit(f) calls f(key) for every candidate
  // key owned by the calling thread. *take_eq = how many keys equal to the threshold are selected,
  // *eq_total = how many candidates equal the threshold.
  template <class Visit>
  static __device__ __forceinline__ unsigned select(Smem& s, int k, int tid, int* take_eq, int* eq_total, Visit visit) {
    const int warp = tid >> 5;
    unsigned prefix = 0, mask = 0;
    int remaining = k, last_count = 0;
#pragma unroll 1
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int i = tid; i < WARPS * 256; i += THREADS) (&s.whist[0][0])[i] = 0;
      __syncthreads();
      visit([&](unsigned key) {
        if ((key & mask) == prefix) atomicAdd(&s.whist[warp][(key >> shift) & 255u], 1u);
      });
      __syncthreads();
      for (int b = tid; b < 256; b += THREADS) {
        unsigned total = 0;
#pragma unroll 8
        for (int w = 0; w < WARPS; ++w) total += s.whist[w][b];
        s.hist[b] = total;
      }
      __syncthreads();
      pick_bin(s, remaining, tid);
      prefix |= s.chosen[0] << shift;
      mask |= 255u << shift;
      remaining = int(s.chosen[1]);
      last_count = int(s.chosen[2]);
      __syncthreads();
    }
    *take_eq = remaining;
    *eq_total = last_count;
    return prefix;
  }

  // Index-ordered ranking over items j = i*THREADS + tid, i < per (per*WARPS <= THREADS). flag(i, j)
  // must be warp-uniform in control flow (it may itself use ballots). Fills s.table[slot] with the
  // exclusive offset of each (i, warp) group and returns the number of flagged items. The rank of a
  // flagged item is then s.table[slot][i*WARPS + warp] + popc(ballot(flag) & lanemask_lt).
  template <class Flag>
  static __device__ __forceinline__ int prepare_ranks(Smem& s, int slot, int per, int tid, Flag flag) {
    const int warp = tid >> 5, lane = tid & 31;
    for (int i = 0; i < per; ++i) {
      const unsigned b = __ballot_sync(0xffffffffu, flag(i, i * THREADS + tid));
      if (lane == 0) s.table[slot][i * WARPS + warp] = __popc(b);
    }
    __syncthreads();
    const int n = per * WARPS;
    const int mine = tid < n ? s.table[slot][tid] : 0;
    int excl, total;
    Scan(s.scan).ExclusiveSum(mine, excl, total);
    __syncthreads();
    if (tid < n) s.table[slot][tid] = excl;
    __syncthreads();
    return total;
  }
};

__device__ __forceinline__ unsigned lanemask_lt() {
  unsigned m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
}

// ------------------------------------------------------------------------------------------------
// select_blocks: one CTA per row; block keys live in dynamic shared memory ([nblocks] u32).
// ------------------------------------------------------------------------------------------------
constexpr int MAX_SELECT_BLOCKS = 32768;  // keys <= 128 KB smem; table rows (per*WARPS) <= THREADS

struct BlocksArgs {
  const float* logits;
  long stride_row, stride_col;
  const int* starts;  // nullable (absent starts mean zero)
  long stride_start;
  const int* ends;
  long stride_end;
  int row_repeat;
  int width, nblocks, block_size;
  int k;      // topk_blocks (number of output columns written)
  int k_eff;  // min(topk_blocks, nblocks)
  int* out;
  long out_row, out_col;
};

template <int THREADS, int BS>
__global__ void __launch_bounds__(THREADS) select_blocks_kernel(const BlocksArgs a) {
  using R = Radix<THREADS>;
  constexpr int WARPS = R::WARPS;
  __shared__ typename R::Smem s;
  extern __shared__ unsigned keys[];
  const int tid = threadIdx.x, warp = tid >> 5;
  const long row = blockIdx.x;
  const long brow = row / a.row_repeat;  // decode rows share bounds in groups of row_repeat (stock :33-34)
  const int start = a.starts ? a.starts[brow * a.stride_start] : 0;
  const int end = a.ends[brow * a.stride_end];
  const int bs = BS > 0 ? BS : a.block_size;
  const int limit = min(end, a.width);  // in-bounds columns: start <= col < limit (stock :39-42)
  const int pinned = end > start ? (end - start - 1) / bs : -1;
  int nscan = limit > start ? min(a.nblocks, cdiv(limit - start, bs)) : 0;
  if (pinned >= 0 && pinned < a.nblocks) nscan = max(nscan, pinned + 1);
  const float* lrow = a.logits + row * a.stride_row;

  int above = 0;
  for (int j = tid; j < a.nblocks; j += THREADS) {
    unsigned key = 0u;  // blocks past nscan score -inf in stock; 0 sorts below -inf and is never kept
    if (j < nscan) {
      // Block score = NaN-propagating max of the in-bounds logits (stock :37-45). As keys, NaN is the
      // maximum, so propagation is the plain unsigned max; a block with no in-bounds column is -inf.
      unsigned kmax = ORDER_NEG_INF;
      const int c0 = start + j * bs;
      if (BS == 8 && a.stride_col == 1 && c0 + 8 <= limit && ((reinterpret_cast<uintptr_t>(lrow + c0) & 15u) == 0)) {
        const uint4 v0 = __ldg(reinterpret_cast<const uint4*>(lrow + c0));
        const uint4 v1 = __ldg(reinterpret_cast<const uint4*>(lrow + c0) + 1);
        const unsigned v[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
        for (int o = 0; o < 8; ++o) kmax = max(kmax, topk_key_bits(v[o]));
      } else {
        for (int o = 0; o < bs; ++o) {
          const int col = c0 + o;
          if (col < limit) kmax = max(kmax, topk_key(__ldg(lrow + long(col) * a.stride_col)));
        }
      }
      key = j == pinned ? ORDER_POS_INF : kmax;  // the pin overrides NaN too (stock :46-50)
    }
    keys[j] = key;
    above += key > ORDER_NEG_INF;
  }
  const int total_above = R::block_sum(s, above, tid);  // also publishes keys[] to the CTA

  // torch.topk(scores, k_eff) followed by "-inf -> -1" keeps exactly the non -inf blocks when there are
  // at most k_eff of them; otherwise the top k_eff keys (all > -inf), ties to the lowest block index.
  unsigned threshold = ORDER_NEG_INF;  // keep key > threshold, plus take_eq keys == threshold
  int take_eq = 0, eq_total = 0;
  bool need_rank = false;
  if (total_above > a.k_eff) {
    threshold = R::select(s, a.k_eff, tid, &take_eq, &eq_total, [&](auto count) {
      for (int j = tid; j < a.nblocks; j += THREADS) {
        const unsigned key = keys[j];
        if (key > ORDER_NEG_INF) count(key);
      }
    });
    need_rank = take_eq < eq_total;  // CTA-uniform
  }
  const int per = cdiv(a.nblocks, THREADS);
  if (need_rank)
    R::prepare_ranks(s, 0, per, tid, [&](int, int j) { return j < a.nblocks && keys[j] == threshold; });
  auto keep = [&](int i, int j) {
    const unsigned key = j < a.nblocks ? keys[j] : 0u;
    bool eq = key == threshold && threshold != ORDER_NEG_INF;
    if (need_rank) {
      const unsigned b = __ballot_sync(0xffffffffu, eq);
      eq = eq && s.table[0][i * WARPS + warp] + __popc(b & lanemask_lt()) < take_eq;
    }
    return key > threshold || eq;
  };
  const int kept = R::prepare_ranks(s, 1, per, tid, keep);
  int* out = a.out + row * a.out_row;
  for (int i = 0; i < per; ++i) {
    const int j = i * THREADS + tid;
    const bool f = keep(i, j);
    const unsigned b = __ballot_sync(0xffffffffu, f);
    if (f) out[long(s.table[1][i * WARPS + warp] + __popc(b & lanemask_lt())) * a.out_col] = j;
  }
  for (int c = kept + tid; c < a.k; c += THREADS) out[long(c) * a.out_col] = -1;
}

template <int THREADS, int BS>
void launch_blocks_t(const BlocksArgs& a, int rows, cudaStream_t stream) {
  constexpr auto kernel = select_blocks_kernel<THREADS, BS>;
  static const bool attr = [] {
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SELECT_BLOCKS * 4) == cudaSuccess);
    return true;
  }();
  (void)attr;
  kernel<<<rows, THREADS, size_t(a.nblocks) * 4, stream>>>(a);
}

template <int THREADS>
void launch_blocks(const BlocksArgs& a, int rows, cudaStream_t stream) {
  if (a.block_size == 8) launch_blocks_t<THREADS, 8>(a, rows, stream);
  else launch_blocks_t<THREADS, 0>(a, rows, stream);
}

// ------------------------------------------------------------------------------------------------
// mask_candidates
// ------------------------------------------------------------------------------------------------
constexpr int MASK_THREADS = 512;
constexpr int MASK_COLS_PER_CTA = 8192;
constexpr int MASK_MAX_FLAG_WORDS = 4096;  // nblocks + 1 <= 131072 blocks

struct MaskArgs {
  float* logits;
  long stride_row, stride_col;
  const int* starts;
  long stride_start;
  const int* ends;
  long stride_end;
  int row_repeat;
  int width, nblocks, block_size;
  const int* cand;
  long cand_row, cand_col;
  int num_cand;
  int full;  // 1: no early exit (the deploy_s1 kernel; DS41H_MASK_CHECK reference)
};

__global__ void __launch_bounds__(MASK_THREADS) mask_candidates_kernel(const MaskArgs a) {
  __shared__ unsigned flags[MASK_MAX_FLAG_WORDS];
  const int tid = threadIdx.x;
  const long row = blockIdx.x;
  const long brow = row / a.row_repeat;
  const int start = a.starts ? a.starts[brow * a.stride_start] : 0;
  const int end = a.ends[brow * a.stride_end];
  // A CTA whose column range starts at or after the row's end returns at once. Every
  // consumer of the masked logits reads only [start, end) of a row (decode: the indexer top-k over seq_lens; prefill:
  // topKPerRowPrefill over [ks, ke)), and the paged logits kernel never writes columns >= end either, so the skipped -inf
  // writes (and the per-CTA candidate-flag rebuild) were never read. Columns < end get the identical writes. The grid spans
  // the logits WIDTH (max_model_len for the ratio-1 layers: 655,360 at the endpoint) while contexts are ~65K.
  if (!a.full && int(blockIdx.y) * MASK_COLS_PER_CTA >= end) return;
  const int words = (a.nblocks + 1 + 31) / 32;
  for (int w = tid; w < words; w += MASK_THREADS) flags[w] = 0u;
  __syncthreads();
  // _candidate_flags_kernel (stock :92-105): out-of-width candidates clamp to the edge slot nblocks.
  const int* cand = a.cand + row * a.cand_row;
  for (int c = tid; c < a.num_cand; c += MASK_THREADS) {
    long block = cand[long(c) * a.cand_col];
    if (long(start) + block * a.block_size >= a.width) block = a.nblocks;
    if (block >= 0) atomicOr(&flags[block >> 5], 1u << (block & 31));
  }
  __syncthreads();
  const bool edge = (flags[a.nblocks >> 5] >> (a.nblocks & 31)) & 1u;
  // _mask_candidates_kernel (stock :125-138).
  float* lrow = a.logits + row * a.stride_row;
  const int c_begin = blockIdx.y * MASK_COLS_PER_CTA;
  const int c_end = min(a.width, c_begin + MASK_COLS_PER_CTA);
  for (int col = c_begin + tid; col < c_end; col += MASK_THREADS) {
    const bool valid = col >= start && col < end;
    bool keep = false;
    if (valid) {
      const int block = (col - start) / a.block_size;
      keep = (flags[block >> 5] >> (block & 31)) & 1u;
    }
    keep |= col == a.width - 1 && edge;
    if (!(valid && keep)) lrow[long(col) * a.stride_col] = -INFINITY;
  }
}

// ------------------------------------------------------------------------------------------------
// topk_rows: per-row top-k over [0, row_end), ascending position order (BRRR select_sorted).
// Each thread owns a contiguous chunk, so index order == (thread, offset) order and one scan ranks it.
// Keys are re-read from global memory (L2) every pass; rows are at most a few hundred KB.
// ------------------------------------------------------------------------------------------------
constexpr int TOPK_THREADS = 1024;

struct TopkArgs {
  const float* logits;
  long stride_row;
  int width;
  const int* seq_lens;
  int seq_lens_numel;
  int next_n;
  int k;
  int* out;
  long out_row;
  const int* visible;  // EXPAND only: per-row number of visible expanded positions (nullable -> 8*row_end)
  int* out_lengths;    // EXPAND only
};

template <bool EXPAND>
__global__ void __launch_bounds__(TOPK_THREADS) topk_rows_kernel(const TopkArgs a) {
  using R = Radix<TOPK_THREADS>;
  __shared__ typename R::Smem s;
  const int tid = threadIdx.x;
  const long row = blockIdx.x;
  const float* lrow = a.logits + row * a.stride_row;
  // indexer_topk.py _row_ends (:279-300).
  int n;
  if (a.seq_lens_numel == int(gridDim.x)) {
    n = a.seq_lens[row];
  } else {
    const int b = int(row / a.next_n), j = int(row % a.next_n);
    n = a.seq_lens[b] - a.next_n + 1 + j;
  }
  n = min(max(n, 0), a.width);
  const int vis = EXPAND ? (a.visible ? max(a.visible[row], 0) : n * 8) : 0;
  const int chunk = cdiv(n, TOPK_THREADS), lo = min(n, tid * chunk), hi = min(n, lo + chunk);
  unsigned threshold = 0u;
  bool select_all = n <= a.k;
  int take_eq = 0, eq_total = 0, tie_base = 0;
  bool need_rank = false;
  if (!select_all) {
    threshold = R::select(s, a.k, tid, &take_eq, &eq_total, [&](auto count) {
      for (int i = lo; i < hi; ++i) count(topk_key(lrow[i]));
    });
    need_rank = take_eq < eq_total;
    if (need_rank) {
      int equal = 0;
      for (int i = lo; i < hi; ++i) equal += topk_key(lrow[i]) == threshold;
      int unused;
      typename R::Scan(s.scan).ExclusiveSum(equal, tie_base, unused);
      __syncthreads();
    }
  }
  auto selected = [&](int i, int& eq_seen) {
    if (select_all) return true;
    const unsigned key = topk_key(lrow[i]);
    if (key > threshold) return true;
    if (key != threshold) return false;
    return !need_rank || eq_seen++ < take_eq;
  };
  int kept = 0, covered = 0;
  {
    int eq_seen = tie_base;
    for (int i = lo; i < hi; ++i)
      if (selected(i, eq_seen)) {
        ++kept;
        if constexpr (EXPAND) covered += max(0, min(8, vis - i * 8));
      }
  }
  int offset, count;
  typename R::Scan(s.scan).ExclusiveSum(kept, offset, count);
  __syncthreads();
  int* out = a.out + row * a.out_row;
  {
    int eq_seen = tie_base;
    for (int i = lo; i < hi; ++i)
      if (selected(i, eq_seen)) {
        if constexpr (EXPAND) {
#pragma unroll
          for (int o = 0; o < 8; ++o) out[offset * 8 + o] = i * 8 + o < vis ? i * 8 + o : -1;
        } else {
          out[offset] = i;
        }
        ++offset;
      }
  }
  for (int c = count + tid; c < a.k; c += TOPK_THREADS) {
    if constexpr (EXPAND) {
#pragma unroll
      for (int o = 0; o < 8; ++o) out[c * 8 + o] = -1;
    } else {
      out[c] = -1;
    }
  }
  if constexpr (EXPAND) {
    int total_covered;
    typename R::Scan(s.scan).ExclusiveSum(covered, covered, total_covered);
    if (tid == 0) a.out_lengths[row] = total_covered;
  }
}

void check_bounds(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kInt && t.dim() == 1, name, " must be a 1-D int32 CUDA tensor");
}

}  // namespace

// Max nblocks handled by select_blocks (keys in shared memory) / by mask_candidates (flag bitmap).
int64_t max_select_blocks() { return MAX_SELECT_BLOCKS; }
int64_t max_mask_blocks() { return int64_t(MASK_MAX_FLAG_WORDS) * 32 - 1; }

void select_blocks(torch::Tensor logits, std::optional<torch::Tensor> starts, torch::Tensor ends, int64_t topk_blocks, int64_t block_size,
                   torch::Tensor out, int64_t row_repeat) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.scalar_type() == at::kFloat, "logits must be 2-D fp32 CUDA");
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.scalar_type() == at::kInt && out.size(0) >= logits.size(0) && out.size(1) >= topk_blocks,
              "out must be int32 [>=rows, >=topk_blocks]");
  check_bounds(ends, "ends");
  if (starts) check_bounds(*starts, "starts");
  TORCH_CHECK(block_size > 0 && topk_blocks > 0 && row_repeat > 0);
  const int rows = int(logits.size(0)), width = int(logits.size(1));
  if (rows == 0) return;
  const int nblocks = int(cdiv(width, int(block_size)));
  TORCH_CHECK(nblocks <= max_select_blocks(), "too many blocks for select_blocks");
  const c10::cuda::CUDAGuard guard(logits.device());
  const BlocksArgs a{logits.data_ptr<float>(),
                     logits.stride(0),
                     logits.stride(1),
                     starts ? starts->data_ptr<int>() : nullptr,
                     starts ? starts->stride(0) : 0,
                     ends.data_ptr<int>(),
                     ends.stride(0),
                     int(row_repeat),
                     width,
                     nblocks,
                     int(block_size),
                     int(topk_blocks),
                     int(std::min<int64_t>(topk_blocks, nblocks)),
                     out.data_ptr<int>(),
                     out.stride(0),
                     out.stride(1)};
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (nblocks <= 2048) launch_blocks<256>(a, rows, stream);
  else launch_blocks<1024>(a, rows, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void mask_candidates_impl(torch::Tensor logits, std::optional<torch::Tensor> starts, torch::Tensor ends, torch::Tensor candidates,
                                 int64_t block_size, int64_t row_repeat, int full) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.scalar_type() == at::kFloat, "logits must be 2-D fp32 CUDA");
  TORCH_CHECK(candidates.is_cuda() && candidates.dim() == 2 && candidates.scalar_type() == at::kInt && candidates.size(0) >= logits.size(0),
              "candidates must be int32 [>=rows, K]");
  check_bounds(ends, "ends");
  if (starts) check_bounds(*starts, "starts");
  TORCH_CHECK(block_size > 0 && row_repeat > 0);
  const int rows = int(logits.size(0)), width = int(logits.size(1));
  if (rows == 0 || width == 0) return;
  const int nblocks = int(cdiv(width, int(block_size)));
  TORCH_CHECK(nblocks <= max_mask_blocks(), "too many blocks for mask_candidates");
  const c10::cuda::CUDAGuard guard(logits.device());
  const MaskArgs a{logits.data_ptr<float>(),
                   logits.stride(0),
                   logits.stride(1),
                   starts ? starts->data_ptr<int>() : nullptr,
                   starts ? starts->stride(0) : 0,
                   ends.data_ptr<int>(),
                   ends.stride(0),
                   int(row_repeat),
                   width,
                   nblocks,
                   int(block_size),
                   candidates.data_ptr<int>(),
                   candidates.stride(0),
                   candidates.stride(1),
                   int(candidates.size(1)),
                   full};
  const dim3 grid(rows, cdiv(width, MASK_COLS_PER_CTA));
  mask_candidates_kernel<<<grid, MASK_THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mask_candidates(torch::Tensor logits, std::optional<torch::Tensor> starts, torch::Tensor ends, torch::Tensor candidates, int64_t block_size,
                     int64_t row_repeat) {
  mask_candidates_impl(logits, starts, ends, candidates, block_size, row_repeat, 0);
}

void mask_candidates_full(torch::Tensor logits, std::optional<torch::Tensor> starts, torch::Tensor ends, torch::Tensor candidates,
                          int64_t block_size, int64_t row_repeat) {
  mask_candidates_impl(logits, starts, ends, candidates, block_size, row_repeat, 1);
}

void topk_rows(torch::Tensor logits, torch::Tensor seq_lens, int64_t next_n, int64_t k, torch::Tensor out, std::optional<torch::Tensor> out_lengths,
               std::optional<torch::Tensor> visible) {
  const bool expand = out_lengths.has_value();
  TORCH_CHECK(expand || !visible.has_value(), "visible requires out_lengths (expand mode)");
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.scalar_type() == at::kFloat && logits.stride(1) == 1, "logits must be 2-D fp32, stride(1)==1");
  TORCH_CHECK(seq_lens.is_cuda() && seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous(), "seq_lens must be contiguous int32");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kInt && out.dim() == 2 && out.stride(1) == 1 && out.size(0) >= logits.size(0) &&
                  out.size(1) >= k * (expand ? 8 : 1),
              "out must be int32 [rows, k(*8)] with stride(1)==1");
  TORCH_CHECK(k > 0 && next_n > 0);
  const int rows = int(logits.size(0));
  if (rows == 0) return;
  TORCH_CHECK(seq_lens.numel() == rows || seq_lens.numel() * next_n >= rows, "seq_lens does not cover the rows");
  if (expand) TORCH_CHECK(out_lengths->is_cuda() && out_lengths->scalar_type() == at::kInt && out_lengths->is_contiguous() && out_lengths->numel() >= rows);
  if (visible) TORCH_CHECK(visible->is_cuda() && visible->scalar_type() == at::kInt && visible->is_contiguous() && visible->numel() >= rows);
  const c10::cuda::CUDAGuard guard(logits.device());
  const TopkArgs a{logits.data_ptr<float>(), logits.stride(0), int(logits.size(1)), seq_lens.data_ptr<int>(), int(seq_lens.numel()), int(next_n), int(k),
                   out.data_ptr<int>(), out.stride(0), visible ? visible->data_ptr<int>() : nullptr, expand ? out_lengths->data_ptr<int>() : nullptr};
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (expand) topk_rows_kernel<true><<<rows, TOPK_THREADS, 0, stream>>>(a);
  else topk_rows_kernel<false><<<rows, TOPK_THREADS, 0, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("select_blocks", &select_blocks, "fused block-score + pin + top-k candidate blocks (stock select_candidate_blocks)");
  m.def("mask_candidates", &mask_candidates, "fused candidate flags + mask (stock apply_candidate_mask)");
  m.def("mask_candidates_full", &mask_candidates_full, "mask_candidates without the early exit (CHECK reference)");
  m.def("topk_rows", &topk_rows, "per-row top-k over [0, row_end) in position order (optional x8 expand)");
  m.def("max_select_blocks", &max_select_blocks);
  m.def("max_mask_blocks", &max_mask_blocks);
}
