// D13: device-side row expansion.
//
// With async scheduling the host plans step N+1 before it knows step N's accepted-token counts, so the positions of
// N+1's decode rows (and every table derived from them) are resolved here, on the GPU, from the device's own
// num_computed. Three graph-capturable kernels per rank; every count lives in device memory, so one captured graph
// serves every step of a row bucket:
//   k_advance   after step N's head: nc[slot] = start_N + (decode ? accepted : n_rows)        (serve RankScheduler.update)
//   k_ring_ops  the host's window-page deltas: clears (block -1) first, then (slot, block, page) sets
//   k_expand    step N+1's row tables, = serve TableBuilder.build / build_step_batch for the same StepBatch:
//                 pos = start + i (start = host start, or nc[slot] for decode sequences the host left at -1)
//                 own_slot = window page(pos / 32) * 32 + pos % 32
//                 si[128] = the window slots of positions lo .. pos (lo = max(replay_start, pos - 127)), -1 padded
//                 sl = pos - lo + 1;  paged_slot = pos < cached_upto ? -1 : dbt[slot][pos / 64] * 64 + pos % 64
//                 seq_start[s] = the resolved start (k_advance's input for the next step); seq_slot[row] = s
//               padding rows (row >= rows): the engine's scratch pattern (own = scratch, si[0] = scratch, sl 1, pos 0).
// Window pages per state slot sit in a ring of RB logical 32-token blocks (index block % RB, validated by the block id
// stored with the page); missing pages count in err[0] and give -1.
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>

namespace rx {

constexpr int SW = 128;  // sliding window
constexpr int WT = 32;   // window page tokens
constexpr int BT = 64;   // paged page tokens
constexpr int RB = 512;  // ring blocks per state slot (16K positions of live window)

struct Desc {               // per-step descriptor of one rank (device), sequences in row order
  const int* slot;          // [S] state slot
  const int* row0;          // [S] first row
  const int* nrows;         // [S]
  const int* kind;          // [S] 0 decode, 1 prefill
  const long long* start;   // [S] host start, or -1 = nc[slot]
  const long long* replay;  // [S] replay_start
  const long long* cached;  // [S] cached_upto
  const int* counts;        // [0] S, [1] rows (real), [2] window ops
};

__device__ __forceinline__ int ring_page(const int2* __restrict__ ring, int slot, long long b, int* err) {
  const int2 e = ring[(size_t)slot * RB + (int)(b & (RB - 1))];
  if (e.x != (int)b) {
    atomicAdd(err, 1);
    return -1;
  }
  return e.y;
}

// one thread per sequence of the PREVIOUS step
__global__ void k_advance(const int* __restrict__ slot, const int* __restrict__ kind, const int* __restrict__ nrows,
                          const long long* __restrict__ start_res, const int* __restrict__ accepted,
                          const int* __restrict__ counts, int* __restrict__ nc) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= counts[0]) return;
  nc[slot[s]] = (int)(start_res[s] + (kind[s] == 0 ? accepted[s] : nrows[s]));
}

// one CTA: clears, barrier, sets (ops [n, 3] = slot, block, page; block -1 = clear the slot's ring)
__global__ void __launch_bounds__(1024) k_ring_ops(int2* __restrict__ ring, const int* __restrict__ ops,
                                                   const int* __restrict__ counts) {
  const int n = counts[2];
  for (int i = 0; i < n; ++i) {
    if (ops[3 * i + 1] >= 0) continue;
    int2* r = ring + (size_t)ops[3 * i] * RB;
    for (int j = threadIdx.x; j < RB; j += blockDim.x) r[j] = make_int2(-1, -1);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int b = ops[3 * i + 1];
    if (b >= 0) ring[(size_t)ops[3 * i] * RB + (b & (RB - 1))] = make_int2(b, ops[3 * i + 2]);
  }
}

// one warp per row (R rows = the bucket; 4 si entries per lane)
__global__ void __launch_bounds__(256) k_expand(const Desc d, const int2* __restrict__ ring, const int* __restrict__ dbt,
                                                long long dbt_stride, const int* __restrict__ nc, int R,
                                                long long scratch, long long* __restrict__ pos_o,
                                                long long* __restrict__ own_o, int* __restrict__ si_o,
                                                int* __restrict__ sl_o, long long* __restrict__ paged_o,
                                                long long* __restrict__ start_o, int* __restrict__ seq_o,
                                                int* __restrict__ err) {
  const int t = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (t >= R) return;
  const int S = d.counts[0], n = d.counts[1];
  int* si = si_o + (size_t)t * SW;
  if (t >= n || S == 0) {  // padding row: engine._upload's scratch pattern
#pragma unroll
    for (int j = 0; j < 4; ++j) si[4 * lane + j] = (lane == 0 && j == 0) ? (int)scratch : -1;
    if (lane == 0) {
      pos_o[t] = 0;
      own_o[t] = scratch;
      sl_o[t] = 1;
      paged_o[t] = -1;
      seq_o[t] = -1;
    }
    return;
  }
  int lo_s = 0, hi_s = S - 1;  // the last sequence with row0 <= t
  while (lo_s < hi_s) {
    const int mid = (lo_s + hi_s + 1) >> 1;
    if (__ldg(d.row0 + mid) <= t) lo_s = mid; else hi_s = mid - 1;
  }
  const int s = lo_s;
  const int slot = __ldg(d.slot + s), i = t - __ldg(d.row0 + s);
  const long long hs = __ldg(d.start + s);
  const long long start = hs >= 0 ? hs : (long long)nc[slot];
  const long long p = start + i;
  const long long rs = __ldg(d.replay + s);
  const long long lo = rs > p - (SW - 1) ? rs : p - (SW - 1);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const long long idx = lo + 4 * lane + j;
    int v = -1;
    if (idx <= p) {
      const int pg = ring_page(ring, slot, idx / WT, err);
      v = pg < 0 ? -1 : pg * WT + (int)(idx % WT);
    }
    si[4 * lane + j] = v;
  }
  if (lane == 0) {
    pos_o[t] = p;
    const int pg = ring_page(ring, slot, p / WT, err + 1);
    own_o[t] = pg < 0 ? -1 : (long long)pg * WT + p % WT;
    sl_o[t] = (int)(p - lo + 1);
    paged_o[t] = p < __ldg(d.cached + s) ? -1 : (long long)dbt[(size_t)slot * dbt_stride + p / BT] * BT + p % BT;
    seq_o[t] = s;
    if (i == 0) start_o[s] = start;
  }
}

}  // namespace rx

namespace {

void chk(const torch::Tensor& t, at::ScalarType dt, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous(), n, ": contiguous CUDA tensor of the right dtype");
}

void advance(torch::Tensor slot, torch::Tensor kind, torch::Tensor nrows, torch::Tensor start_res, torch::Tensor accepted,
             torch::Tensor counts, torch::Tensor nc, int64_t s_max) {
  const c10::cuda::CUDAGuard g(nc.device());
  chk(slot, at::kInt, "slot"); chk(kind, at::kInt, "kind"); chk(nrows, at::kInt, "nrows"); chk(start_res, at::kLong, "start_res");
  chk(accepted, at::kInt, "accepted"); chk(counts, at::kInt, "counts"); chk(nc, at::kInt, "nc");
  auto st = c10::cuda::getCurrentCUDAStream(nc.device().index()).stream();
  rx::k_advance<<<(int)((s_max + 127) / 128), 128, 0, st>>>(slot.data_ptr<int>(), kind.data_ptr<int>(), nrows.data_ptr<int>(),
                                                             reinterpret_cast<const long long*>(start_res.data_ptr<int64_t>()),
                                                             accepted.data_ptr<int>(), counts.data_ptr<int>(), nc.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ring_ops(torch::Tensor ring, torch::Tensor ops, torch::Tensor counts) {
  const c10::cuda::CUDAGuard g(ring.device());
  chk(ring, at::kInt, "ring"); chk(ops, at::kInt, "ops"); chk(counts, at::kInt, "counts");
  TORCH_CHECK(ring.dim() == 3 && ring.size(1) == rx::RB && ring.size(2) == 2, "ring [slots, 512, 2]");
  auto st = c10::cuda::getCurrentCUDAStream(ring.device().index()).stream();
  rx::k_ring_ops<<<1, 1024, 0, st>>>(reinterpret_cast<int2*>(ring.data_ptr<int>()), ops.data_ptr<int>(), counts.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void expand(torch::Tensor slot, torch::Tensor row0, torch::Tensor nrows, torch::Tensor kind, torch::Tensor start,
            torch::Tensor replay, torch::Tensor cached, torch::Tensor counts, torch::Tensor ring, torch::Tensor dbt,
            torch::Tensor nc, int64_t R, int64_t scratch, torch::Tensor pos_o, torch::Tensor own_o, torch::Tensor si_o,
            torch::Tensor sl_o, torch::Tensor paged_o, torch::Tensor start_o, torch::Tensor seq_o, torch::Tensor err) {
  const c10::cuda::CUDAGuard g(ring.device());
  chk(slot, at::kInt, "slot"); chk(row0, at::kInt, "row0"); chk(nrows, at::kInt, "nrows"); chk(kind, at::kInt, "kind");
  chk(start, at::kLong, "start"); chk(replay, at::kLong, "replay"); chk(cached, at::kLong, "cached"); chk(counts, at::kInt, "counts");
  chk(ring, at::kInt, "ring"); chk(nc, at::kInt, "nc"); chk(err, at::kInt, "err");
  chk(pos_o, at::kLong, "pos"); chk(own_o, at::kLong, "own"); chk(si_o, at::kInt, "si"); chk(sl_o, at::kInt, "sl");
  chk(paged_o, at::kLong, "paged"); chk(start_o, at::kLong, "start_o"); chk(seq_o, at::kInt, "seq_o");
  TORCH_CHECK(dbt.is_cuda() && dbt.scalar_type() == at::kInt && dbt.dim() == 2 && dbt.stride(1) == 1, "dbt int32 [slots, P]");
  TORCH_CHECK(pos_o.numel() >= R && own_o.numel() >= R && si_o.numel() >= R * rx::SW && sl_o.numel() >= R && paged_o.numel() >= R,
              "outputs too small for R rows");
  if (R == 0) return;
  rx::Desc d{slot.data_ptr<int>(), row0.data_ptr<int>(), nrows.data_ptr<int>(), kind.data_ptr<int>(),
             reinterpret_cast<const long long*>(start.data_ptr<int64_t>()), reinterpret_cast<const long long*>(replay.data_ptr<int64_t>()),
             reinterpret_cast<const long long*>(cached.data_ptr<int64_t>()), counts.data_ptr<int>()};
  auto st = c10::cuda::getCurrentCUDAStream(ring.device().index()).stream();
  rx::k_expand<<<(int)((R * 32 + 255) / 256), 256, 0, st>>>(
      d, reinterpret_cast<const int2*>(ring.data_ptr<int>()), dbt.data_ptr<int>(), dbt.stride(0), nc.data_ptr<int>(), (int)R,
      scratch, reinterpret_cast<long long*>(pos_o.data_ptr<int64_t>()), reinterpret_cast<long long*>(own_o.data_ptr<int64_t>()),
      si_o.data_ptr<int>(), sl_o.data_ptr<int>(), reinterpret_cast<long long*>(paged_o.data_ptr<int64_t>()),
      reinterpret_cast<long long*>(start_o.data_ptr<int64_t>()), seq_o.data_ptr<int>(), err.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("advance", &advance);
  m.def("ring_ops", &ring_ops);
  m.def("expand", &expand);
}
