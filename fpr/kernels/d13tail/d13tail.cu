// d13tail: D13's graph T' (the steady owner's step prep, fpr/engine/d13.py _gto_body / _gt_body + the indexer's
// device_tables_d13) as ONE kernel.
//
// Graph T' on the C1 owner = ~45 tiny torch kernels in a CUDA graph (~67 us on the critical path between graph H and the
// owner's graph L, t-89 trace). The same ops, in the same order, on the same buffers, in one CTA:
//   phase 1  indexer rslot <- rbase; Engram b_start <- sbase; tok[dt[0, j]] <- next_ids_flat[dt[1, j]]   (j < n_tok)
//   phase 2  rowexp k_ring_ops (verbatim: clears, barrier, sets)
//   phase 3  rowexp k_expand (verbatim, one warp per row; ring read with plain loads: written in phase 2)
//   phase 4  per row: own < 0 -> scratch; with sequences: paged = -1 where seq >= 0 and pos // 64 >= nb[seq];
//            the indexer's tables: sl1 = valid ? pos + 1 : 0, sl2 = valid ? (pos + 1) // 2 : 0 (int32),
//            cs1 = valid & paged >= 0 ? paged: -1, cs2 = ...
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>

namespace dt13 {

constexpr int SW = 128;  // sliding window
constexpr int WT = 32;   // window page tokens
constexpr int BT = 64;   // paged page tokens
constexpr int RB = 512;  // ring blocks per state slot

struct Args {
  // phase 1
  long long* rslot; const long long* rbase; int n_rslot;
  int* b_start; const int* sbase; int n_sbase;
  long long* tok; const long long* dt; long long dt_stride; int n_tok; const long long* next_ids;
  // rowexp descriptor (device; sequences in row order)
  const int* slot; const int* row0; const int* nrows; const long long* start; const long long* replay;
  const long long* cached; const int* counts;  // [0] S, [1] rows (real), [2] window ops
  int2* ring; const int* ops; const int* dbt; long long dbt_stride; const int* nc; int R; long long scratch;
  long long* pos; long long* own; int* si; int* sl; long long* paged; long long* start_o; int* seq; int* err;
  // phase 4
  int has_seqs; const long long* nb; const bool* valid; int* sl1; int* sl2; long long* cs1; long long* cs2;
};

__device__ __forceinline__ long long fdiv(long long a, long long b) {   // torch floor division (b > 0 here)
  long long q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}
__device__ __forceinline__ long long fmod_(long long a, long long b) {  // torch remainder (sign of b)
  long long r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) r += b;
  return r;
}

__device__ __forceinline__ int ring_page(const int2* ring, int slot, long long b, int* err) {
  const int2 e = ring[(size_t)slot * RB + (int)(b & (RB - 1))];
  if (e.x != (int)b) {
    atomicAdd(err, 1);
    return -1;
  }
  return e.y;
}

__global__ void __launch_bounds__(1024) k_tail(const Args a) {
  const int tid = threadIdx.x, nth = blockDim.x;
  // ---- phase 1: the in-place inputs reset from the full prep's copies, then the device tokens
  for (int i = tid; i < a.n_rslot; i += nth) a.rslot[i] = a.rbase[i];
  for (int i = tid; i < a.n_sbase; i += nth) a.b_start[i] = a.sbase[i];
  for (int j = tid; j < a.n_tok; j += nth) a.tok[a.dt[j]] = a.next_ids[a.dt[a.dt_stride + j]];
  // ---- phase 2: k_ring_ops (clears first, then sets)
  const int n_ops = a.counts[2];
  for (int i = 0; i < n_ops; ++i) {
    if (a.ops[3 * i + 1] >= 0) continue;
    int2* r = a.ring + (size_t)a.ops[3 * i] * RB;
    for (int j = tid; j < RB; j += nth) r[j] = make_int2(-1, -1);
  }
  __syncthreads();
  for (int i = tid; i < n_ops; i += nth) {
    const int b = a.ops[3 * i + 1];
    if (b >= 0) a.ring[(size_t)a.ops[3 * i] * RB + (b & (RB - 1))] = make_int2(b, a.ops[3 * i + 2]);
  }
  __syncthreads();
  // ---- phase 3: k_expand, one warp per row
  const int lane = tid & 31, nw = nth >> 5;
  const int S = a.counts[0], n = a.counts[1];
  for (int t = tid >> 5; t < a.R; t += nw) {
    int* si = a.si + (size_t)t * SW;
    if (t >= n || S == 0) {  // padding row: engine._upload's scratch pattern
#pragma unroll
      for (int j = 0; j < 4; ++j) si[4 * lane + j] = (lane == 0 && j == 0) ? (int)a.scratch : -1;
      if (lane == 0) {
        a.pos[t] = 0;
        a.own[t] = a.scratch;
        a.sl[t] = 1;
        a.paged[t] = -1;
        a.seq[t] = -1;
      }
      continue;
    }
    int lo_s = 0, hi_s = S - 1;  // the last sequence with row0 <= t
    while (lo_s < hi_s) {
      const int mid = (lo_s + hi_s + 1) >> 1;
      if (a.row0[mid] <= t) lo_s = mid; else hi_s = mid - 1;
    }
    const int s = lo_s;
    const int slot = a.slot[s], i = t - a.row0[s];
    const long long hs = a.start[s];
    const long long start = hs >= 0 ? hs : (long long)a.nc[slot];
    const long long p = start + i;
    const long long rs = a.replay[s];
    const long long lo = rs > p - (SW - 1) ? rs : p - (SW - 1);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const long long idx = lo + 4 * lane + j;
      int v = -1;
      if (idx <= p) {
        const int pg = ring_page(a.ring, slot, idx / WT, a.err);
        v = pg < 0 ? -1 : pg * WT + (int)(idx % WT);
      }
      si[4 * lane + j] = v;
    }
    if (lane == 0) {
      a.pos[t] = p;
      const int pg = ring_page(a.ring, slot, p / WT, a.err + 1);
      a.own[t] = pg < 0 ? -1 : (long long)pg * WT + p % WT;
      a.sl[t] = (int)(p - lo + 1);
      a.paged[t] = p < a.cached[s] ? -1 : (long long)a.dbt[(size_t)slot * a.dbt_stride + p / BT] * BT + p % BT;
      a.seq[t] = s;
      if (i == 0) a.start_o[s] = start;
    }
  }
  __syncthreads();
  // ---- phase 4: the guard's device half + the indexer's device_tables_d13, one thread per row
  for (int t = tid; t < a.R; t += nth) {
    if (a.own[t] < 0) a.own[t] = a.scratch;
    const long long p = a.pos[t];
    long long pg = a.paged[t];
    if (a.has_seqs) {
      const int sq = a.seq[t];
      const long long lim = a.nb[sq < 0 ? 0 : sq];
      if (sq >= 0 && fdiv(p, 64) >= lim) {
        pg = -1;
        a.paged[t] = -1;
      }
    }
    const bool v = a.valid[t];
    const long long p1 = p + 1;
    a.sl1[t] = v ? (int)p1 : 0;
    a.sl2[t] = v ? (int)fdiv(p1, 2) : 0;
    const bool ok = v && pg >= 0;
    a.cs1[t] = ok ? pg : -1;
    a.cs2[t] = ok ? fdiv(pg, 64) * 32 + fdiv(fmod_(pg, 64), 2) : -1;
    a.rslot[t] = v ? a.rslot[t] + fmod_(p, 8) : -1;
  }
}

}  // namespace dt13

namespace {

template <typename T>
T* P(const torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

void chk(const torch::Tensor& t, at::ScalarType dt, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous(), n, ": contiguous CUDA tensor of the right dtype");
}

// tail(rslot, rbase, b_start, sbase, tok, dt, n_tok, next_ids, slot, row0, nrows, start, replay, cached, counts, ring, ops,
//      dbt, nc, R, scratch, pos, own, si, sl, paged, start_o, seq, err, has_seqs, nb, valid, sl1, sl2, cs1, cs2)
void tail(torch::Tensor rslot, torch::Tensor rbase, torch::Tensor b_start, torch::Tensor sbase, torch::Tensor tok,
          torch::Tensor dt, int64_t n_tok, torch::Tensor next_ids, torch::Tensor slot, torch::Tensor row0,
          torch::Tensor nrows, torch::Tensor start, torch::Tensor replay, torch::Tensor cached, torch::Tensor counts,
          torch::Tensor ring, torch::Tensor ops, torch::Tensor dbt, torch::Tensor nc, int64_t R, int64_t scratch,
          torch::Tensor pos, torch::Tensor own, torch::Tensor si, torch::Tensor sl, torch::Tensor paged, torch::Tensor start_o,
          torch::Tensor seq, torch::Tensor err, bool has_seqs, torch::Tensor nb, torch::Tensor valid, torch::Tensor sl1,
          torch::Tensor sl2, torch::Tensor cs1, torch::Tensor cs2) {
  const c10::cuda::CUDAGuard g(ring.device());
  chk(rslot, at::kLong, "rslot"); chk(rbase, at::kLong, "rbase"); chk(b_start, at::kInt, "b_start"); chk(sbase, at::kInt, "sbase");
  chk(tok, at::kLong, "tok"); chk(next_ids, at::kLong, "next_ids");
  TORCH_CHECK(dt.is_cuda() && dt.scalar_type() == at::kLong && dt.dim() == 2 && dt.size(0) == 2 && dt.stride(1) == 1, "dt int64 [2, n]");
  TORCH_CHECK(n_tok <= dt.size(1), "n_tok > dt columns");
  chk(slot, at::kInt, "slot"); chk(row0, at::kInt, "row0"); chk(nrows, at::kInt, "nrows"); chk(start, at::kLong, "start");
  chk(replay, at::kLong, "replay"); chk(cached, at::kLong, "cached"); chk(counts, at::kInt, "counts");
  chk(ring, at::kInt, "ring"); chk(ops, at::kInt, "ops"); chk(nc, at::kInt, "nc"); chk(err, at::kInt, "err");
  TORCH_CHECK(ring.dim() == 3 && ring.size(1) == dt13::RB && ring.size(2) == 2, "ring [slots, 512, 2]");
  TORCH_CHECK(dbt.is_cuda() && dbt.scalar_type() == at::kInt && dbt.dim() == 2 && dbt.stride(1) == 1, "dbt int32 [slots, P]");
  chk(pos, at::kLong, "pos"); chk(own, at::kLong, "own"); chk(si, at::kInt, "si"); chk(sl, at::kInt, "sl");
  chk(paged, at::kLong, "paged"); chk(start_o, at::kLong, "start_o"); chk(seq, at::kInt, "seq");
  chk(nb, at::kLong, "nb"); chk(valid, at::kBool, "valid"); chk(sl1, at::kInt, "sl1"); chk(sl2, at::kInt, "sl2");
  chk(cs1, at::kLong, "cs1"); chk(cs2, at::kLong, "cs2");
  TORCH_CHECK(pos.numel() >= R && own.numel() >= R && si.numel() >= R * dt13::SW && sl.numel() >= R && paged.numel() >= R &&
              seq.numel() >= R && valid.numel() >= R && sl1.numel() >= R && sl2.numel() >= R && cs1.numel() >= R &&
              cs2.numel() >= R && rslot.numel() >= R && rbase.numel() >= rslot.numel(), "buffers too small for R rows");
  TORCH_CHECK(sbase.numel() == b_start.numel(), "sbase / b_start sizes differ");
  dt13::Args a{P<long long>(rslot), P<const long long>(rbase), (int)rslot.numel(),
               P<int>(b_start), P<const int>(sbase), (int)sbase.numel(),
               P<long long>(tok), P<const long long>(dt), (long long)dt.stride(0), (int)n_tok, P<const long long>(next_ids),
               P<const int>(slot), P<const int>(row0), P<const int>(nrows), P<const long long>(start), P<const long long>(replay),
               P<const long long>(cached), P<const int>(counts),
               P<int2>(ring), P<const int>(ops), P<const int>(dbt), (long long)dbt.stride(0), P<const int>(nc), (int)R,
               (long long)scratch,
               P<long long>(pos), P<long long>(own), P<int>(si), P<int>(sl), P<long long>(paged), P<long long>(start_o),
               P<int>(seq), P<int>(err),
               has_seqs ? 1 : 0, P<const long long>(nb), P<const bool>(valid), P<int>(sl1), P<int>(sl2), P<long long>(cs1),
               P<long long>(cs2)};
  auto st = c10::cuda::getCurrentCUDAStream(ring.device().index()).stream();
  dt13::k_tail<<<1, 1024, 0, st>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tail", &tail); }
