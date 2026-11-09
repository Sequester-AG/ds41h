// indexer2f: pf4 = pf3 with the 4 rows of a pseudo-request in ONE m64n128k32 chain (was two
// m64n64k32 row pairs: half the WGMMA instructions and 25 % less shared-memory traffic per FLOP); weights from shared
// memory (64 accumulators). pf3 = idx2f_pf.cu with ONE elected empty-stage arrive per math warp (was
// one per thread: 512 shared-memory atomics per key tile).
// PREFILL dense index logits with key tiles shared across rows.
//
// Same arithmetic as idx2f_cand.cu's dense mode (= DeepGEMM sm90_fp8_paged_mqa_logits: per 64-token key tile and row
// pair one m64n64k32 x 4 WGMMA chain, relu * weight, 4-way in-thread sums, * kv scale, 2 xor shuffles), so every logit
// is bit-identical. What changes is who reads a key tile: there, each pseudo-request (4 rows) streamed its sequence's
// keys through L2 itself (8.25 KB per tile per 4 rows = 33 B per score: the pipeline alone ran at ~7 TB/s of L2, 4.5 ps
// per score with no math). Here a CTA task is (group, tile): a group = up to 4 consecutive pseudo-requests of the
// SAME sequence (host-supplied starts, gstart), the producer loads the tile ONCE into a shared stage and the 4 math
// warpgroups score it for their own pseudo-request (16 rows per key tile: 8 B per score). Plus fcand6 / fcand7's
// pipeline fixes: the producer's block-table entries cached in its lanes, 2 coalesced quad-row stores per tile per warp,
// the request's weights in registers.
// q [B, 4, 32, 128] e4m3; kv [pages, P, 1, 132] u8 (P = 64 ratio 1, 32 ratio 2); weights [4B, 32] f32; ctx [4B] i32;
// block_table [B, *] i32 (the members of a group share their sequence's pages: the group's first row is read);
// gstart [G + 1] i32 (group g = pseudo-requests gstart[g] .. gstart[g + 1] - 1, at most 4); out [4B, >= max ctx] f32.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <torch/extension.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace idx2f_pf4 {

constexpr int NN = 4;             // rows per pseudo-request (next_n)
constexpr int kHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kBlock = 8;
constexpr int kTileTok = 64;
constexpr int kBlkPerTile = kTileTok / kBlock;
constexpr int kMathWG = 4;        // = group size: warpgroup g scores member g of the group
constexpr int kMathThreads = kMathWG * 128;
constexpr int kTMAThreads = 128;  // warp 0 of it is the producer; 1-3 exit
constexpr int kThreads = kMathThreads + kTMAThreads;
constexpr int kQStages = 2;
constexpr int kKVStages = 4;
constexpr int kKVTileBytes = kTileTok * kHeadDim;
constexpr int kScTileBytes = kTileTok * 4;
constexpr int kNumTMARegisters = 64;
constexpr int kNumMathRegisters = 104;
constexpr int RPM = 4, NPAIR = 1, N = RPM * kHeads;   // pf4: N = 128
constexpr int QBytes = NN * kHeads * kHeadDim;      // 16 KB per pseudo-request
constexpr int PairQBytes = RPM * kHeads * kHeadDim;
constexpr int WBytes = NN * kHeads * 4;
constexpr int OffQ = 0;                                         // [qstage][member] 16 KB, 1024-aligned
constexpr int OffKV = OffQ + kQStages * kMathWG * QBytes;       // [stage] 8 KB
constexpr int OffW = OffKV + kKVStages * kKVTileBytes;          // [qstage][member] 512 B
constexpr int OffSc = OffW + kQStages * kMathWG * WBytes;       // [stage] 256 B
constexpr int OffBar = OffSc + kKVStages * kScTileBytes;
constexpr int NumBars = 2 * kQStages + 2 * kKVStages;
constexpr int OffTask = OffBar + NumBars * 8;                   // int task_off[G + 1], s_nu[G]
static_assert(OffKV % 1024 == 0 && QBytes % 1024 == 0 && PairQBytes % 1024 == 0, "swizzle alignment");
inline int smem_bytes(int G) { return 1024 + OffTask + (2 * G + 1) * 4; }
constexpr int kMaxG = 1024;

using Barrier = cutlass::arch::ClusterTransactionBarrier;

struct Args {
  const int* gstart;       // [G + 1]
  int G;
  const int* ctx;          // [4B]
  const int* block_table;  // [B, bt_stride]
  long bt_stride;
  int bt_cols;
  const float* weights;    // [4B, 32]
  const uint8_t* kv;
  long page_stride;
  float* out;
  long out_stride;
  int page_tok;
};

struct TaskIter {
  const int* off;
  int g, hi, b;
  __device__ __forceinline__ bool next(int& tb, int& tt) {
    if (g >= hi) return false;
    while (off[b + 1] <= g) ++b;
    tb = b, tt = g - off[b], ++g;
    return true;
  }
};

// ABL (diagnostic only; 0 in production): 1 no WGMMA, 2 no epilogue math, 4 no stores
template <int ABL>
__global__ void __launch_bounds__(kThreads, 1)
    pf_logits_kernel(const __grid_constant__ Args a, const __grid_constant__ CUtensorMap tm_q,
                     const __grid_constant__ CUtensorMap tm_kvp) {
  using WGMMA = typename deep_gemm::mma::sm90::FP8MMASelector<N>::type;
  namespace ptx = deep_gemm::ptx;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int warp_idx = __shfl_sync(0xffffffffu, int(threadIdx.x) / 32, 0);
  const int lane = int(ptx::get_lane_idx());
  int* task_off = reinterpret_cast<int*>(smem + OffTask);
  int* s_nu = task_off + a.G + 1;
  Barrier* bars = reinterpret_cast<Barrier*>(smem + OffBar);
  Barrier* full_q = bars;
  Barrier* empty_q = bars + kQStages;
  Barrier* full_kv = bars + 2 * kQStages;
  Barrier* empty_kv = bars + 2 * kQStages + kKVStages;

  // Task prefix over groups (warp 0): a group's key tiles = ceil(max over its rows of ctx / 64)
  if (warp_idx == 0) {
    const int per = (a.G + 31) / 32, j0 = min(a.G, lane * per), j1 = min(a.G, j0 + per);
    int s = 0;
    for (int j = j0; j < j1; ++j) {
      int mx = 0;
      for (int b = __ldg(a.gstart + j); b < __ldg(a.gstart + j + 1); ++b)
#pragma unroll
        for (int r = 0; r < NN; ++r) mx = max(mx, __ldg(a.ctx + b * NN + r));
      const int nu = (mx + kBlock - 1) / kBlock;
      s_nu[j] = nu;
      s += (nu + kBlkPerTile - 1) / kBlkPerTile;
    }
    int inc = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, inc, o);
      if (lane >= o) inc += y;
    }
    int run = inc - s;
    for (int j = j0; j < j1; ++j) {
      task_off[j] = run;
      run += (s_nu[j] + kBlkPerTile - 1) / kBlkPerTile;
    }
    if (lane == 31) task_off[a.G] = inc;
  }
  if (warp_idx == kMathThreads / 32 && cute::elect_one_sync()) {
    for (int i = 0; i < kQStages; ++i) full_q[i].init(1), empty_q[i].init(kMathThreads);
    for (int i = 0; i < kKVStages; ++i) full_kv[i].init(1), empty_kv[i].init(kMathThreads / 32);   // pf3: per warp
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  const long total = task_off[a.G];
  const int lo = int(total * blockIdx.x / gridDim.x), hi = int(total * (blockIdx.x + 1) / gridDim.x);
  if (lo >= hi) return;
  TaskIter it{task_off, lo, hi, 0};

  if (warp_idx >= kMathThreads / 32) {
    // ------------------------------------------------------------------ producer (one warp)
    cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
    if (warp_idx != kMathThreads / 32) return;
    uint32_t kv_iter = 0, q_iter = 0;
    int cur_j = -1, j, t;
    int cache_j = -1, cache_t = 0, cache_v = 0;
    const int P = a.page_tok, npp = kTileTok / P, per = 32 / npp;
    while (it.next(j, t)) {
      const int g0 = __ldg(a.gstart + j), gsz = __ldg(a.gstart + j + 1) - g0;
      if (j != cur_j) {
        const uint32_t qs = q_iter % kQStages, qph = (q_iter / kQStages) & 1;
        ++q_iter;
        empty_q[qs].wait(qph ^ 1);
        if (cute::elect_one_sync()) {
          for (int m = 0; m < gsz; ++m) {
            uint8_t* sq = smem + OffQ + (qs * kMathWG + m) * QBytes;
            cute::SM90_TMA_LOAD_2D::copy(&tm_q, reinterpret_cast<uint64_t*>(&full_q[qs]),
                                         static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL), sq, 0,
                                         (g0 + m) * NN * kHeads);
            cute::SM90_BULK_COPY_G2S::copy(a.weights + long(g0 + m) * NN * kHeads, reinterpret_cast<uint64_t*>(&full_q[qs]),
                                           smem + OffW + (qs * kMathWG + m) * WBytes, WBytes);
          }
          full_q[qs].arrive_and_expect_tx(gsz * (QBytes + WBytes));
        }
        __syncwarp();
        cur_j = j;
      }
      const int nu = s_nu[j];
      const int tile = t, tok0 = tile * kTileTok;
      if (j != cache_j || tile < cache_t || tile - cache_t >= per) {
        cache_j = j, cache_t = tile;
        const int pj = (cache_t + lane / npp) * npp + lane % npp;
        cache_v = (pj < a.bt_cols) ? __ldg(a.block_table + long(g0) * a.bt_stride + pj) : 0;
      }
      const bool second = lane < npp && (tok0 + lane * P < nu * kBlock);   // else the tile's first page (as stock)
      const int pg = __shfl_sync(0xffffffffu, cache_v, (tile - cache_t) * npp + (second ? lane : 0));
      const uint32_t ks = kv_iter % kKVStages, kph = (kv_iter / kKVStages) & 1;
      ++kv_iter;
      empty_kv[ks].wait(kph ^ 1);
      if (lane < npp) {
        uint8_t* skv = smem + OffKV + ks * kKVTileBytes + lane * P * kHeadDim;
        cute::SM90_TMA_LOAD_3D::copy(&tm_kvp, reinterpret_cast<uint64_t*>(&full_kv[ks]),
                                     static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL), skv, 0, 0, pg);
        uint8_t* ssc = smem + OffSc + ks * kScTileBytes + lane * P * 4;
        cute::SM90_BULK_COPY_G2S::copy(a.kv + long(pg) * a.page_stride + P * kHeadDim,
                                       reinterpret_cast<uint64_t*>(&full_kv[ks]), ssc, P * 4);
      }
      __syncwarp();
      if (lane == 0) full_kv[ks].arrive_and_expect_tx(kTileTok * (kHeadDim + 4));
      __syncwarp();
    }
  } else {
    // ------------------------------------------------------------------ math warpgroups (member wg of each group)
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
    const int wg = warp_idx / 4;
    float accum[WGMMA::kNumAccum];
    float sink = 0.f;
    if constexpr ((ABL & 1) != 0) {
#pragma unroll
      for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i) accum[i] = 0.f;
    }
    const uint32_t sub_warp_offset = (warp_idx % 4) * 16;
    const uint32_t v_0_offset = lane / 4 + 0;
    const uint32_t v_1_offset = lane / 4 + 8;
    uint32_t kv_iter = 0, q_iter = 0, qs = 0;
    int cur_j = -1, mb = -1, nu = 0, j, t;
    int L[NN];
    while (it.next(j, t)) {
      if (j != cur_j) {
        if (cur_j >= 0) empty_q[qs].arrive();
        qs = q_iter % kQStages;
        const uint32_t qph = (q_iter / kQStages) & 1;
        ++q_iter;
        full_q[qs].wait(qph);
        cur_j = j;
        const int g0 = __ldg(a.gstart + j), gsz = __ldg(a.gstart + j + 1) - g0;
        mb = wg < gsz ? g0 + wg : -1;
        nu = 0;
        if (mb >= 0) {
          int mx = 0;
#pragma unroll
          for (int r = 0; r < NN; ++r) L[r] = __ldg(a.ctx + mb * NN + r), mx = max(mx, L[r]);
          nu = (mx + kBlock - 1) / kBlock;
        }
      }
      const int tile = t;
      const uint32_t ks = kv_iter % kKVStages, kph = (kv_iter / kKVStages) & 1;
      ++kv_iter;
      if (tile * kBlkPerTile >= nu) {      // member absent or shorter than this group tile: release the stage
        full_kv[ks].wait(kph);
        __syncwarp();
        if (lane == 0) empty_kv[ks].arrive();
        continue;
      }
      const int u0 = tile * kBlkPerTile + (warp_idx % 4) * 2, u1 = u0 + 1;
      const bool ok0 = u0 < nu, ok1 = u1 < nu;
      const int pos0 = u0 * kBlock + int(v_0_offset), pos1 = u1 * kBlock + int(v_0_offset);
      int slot0[NN], slot1[NN];
#pragma unroll
      for (int r = 0; r < NN; ++r) {
        slot0[r] = (ok0 && u0 * kBlock < L[r]) ? u0 : -1;
        slot1[r] = (ok1 && u1 * kBlock < L[r]) ? u1 : -1;
      }
      full_kv[ks].wait(kph);
      uint8_t* skv = smem + OffKV + ks * kKVTileBytes;
      const float* ssc = reinterpret_cast<const float*>(smem + OffSc + ks * kScTileBytes);
      float scale_kv_0 = 0.f, scale_kv_1 = 0.f;
      float ov0[NN], ov1[NN];
#pragma unroll
      for (int p = 0; p < NPAIR; ++p) {
        uint8_t* sq = smem + OffQ + (qs * kMathWG + wg) * QBytes + p * PairQBytes;
#pragma unroll
        for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i) ptx::warpgroup_fence_operand(accum[i]);
        if constexpr ((ABL & 1) == 0) {
        ptx::warpgroup_arrive();
#pragma unroll
        for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++k) {
          auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
              reinterpret_cast<__nv_fp8_e4m3*>(skv) + k * WGMMA::K, deep_gemm::mma::sm90::to_swizzle_cute_type<kHeadDim>(),
              0, kHeadDim * 8);
          auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
              reinterpret_cast<__nv_fp8_e4m3*>(sq) + k * WGMMA::K, deep_gemm::mma::sm90::to_swizzle_cute_type<kHeadDim>(),
              0, kHeadDim * 8);
          WGMMA::wgmma(desc_a, desc_b, accum, k);
        }
        ptx::warpgroup_commit_batch();
        }
#pragma unroll
        for (uint32_t i = 0; i < WGMMA::kNumAccum; ++i) ptx::warpgroup_fence_operand(accum[i]);
        if (p == 0) {
          scale_kv_0 = ptx::ld_shared(ssc + sub_warp_offset + v_0_offset);
          scale_kv_1 = ptx::ld_shared(ssc + sub_warp_offset + v_1_offset);
        }
        if constexpr ((ABL & 1) == 0) ptx::warpgroup_wait<0>();
        if (p == NPAIR - 1) {   // this warp's reads of the stage are done (the warpgroup's MMAs are waited)
          __syncwarp();
          if (lane == 0) empty_kv[ks].arrive();
        }
        // verbatim DeepGEMM epilogue (idx2f_cand.cu)
        static constexpr uint32_t kNumAccumPerReduce = kHeads / 2;
        static_assert(WGMMA::kNumAccum / kNumAccumPerReduce == RPM, "Invalid accumulation");
#pragma unroll
        for (uint32_t i = 0; i < RPM; ++i) {
          const int r = p * RPM + int(i);
          auto shifted_accum = accum + i * kNumAccumPerReduce;
          if constexpr ((ABL & 2) != 0) {
            ov0[r] = shifted_accum[0] * scale_kv_0, ov1[r] = shifted_accum[2] * scale_kv_1;
            continue;
          }
          float weights[kHeads / 4];
          {
            const float* sw = reinterpret_cast<const float*>(smem + OffW + (qs * kMathWG + wg) * WBytes);
#pragma unroll
            for (uint32_t jj = 0; jj < kHeads / 4; ++jj)
              weights[jj] = ptx::ld_shared(sw + r * kHeads + (jj / 2) * 8 + (jj & 1) + (lane % 4) * 2);
          }
          const auto transform = [&](const uint32_t& jj) { return fmaxf(shifted_accum[jj], 0) * weights[(jj / 4) * 2 + (jj & 1)]; };
          float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
#pragma unroll
          for (uint32_t jj = 1; jj < kHeads / 8; ++jj) {
#pragma unroll
            for (uint32_t k = 0; k < 4; k++) sum[k] += transform(jj * 4 + k);
          }
          float v_0 = (sum[0] + sum[1]) * scale_kv_0;
          float v_1 = (sum[2] + sum[3]) * scale_kv_1;
#pragma unroll
          for (uint32_t jj = 0; jj < 2; ++jj) {
            const auto offset = static_cast<int>(1u << jj);
            v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
            v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
          }
          ov0[r] = v_0, ov1[r] = v_1;
        }
      }
      // quad-row stores: lane r % 4 stores row r (every quad lane holds the row's reduced value)
      const int rr = lane % 4;
      const float w0 = rr == 0 ? ov0[0] : rr == 1 ? ov0[1] : rr == 2 ? ov0[2] : ov0[3];
      const float w1 = rr == 0 ? ov1[0] : rr == 1 ? ov1[1] : rr == 2 ? ov1[2] : ov1[3];
      const int s0 = rr == 0 ? slot0[0] : rr == 1 ? slot0[1] : rr == 2 ? slot0[2] : slot0[3];
      const int s1 = rr == 0 ? slot1[0] : rr == 1 ? slot1[1] : rr == 2 ? slot1[2] : slot1[3];
      const int Lr = rr == 0 ? L[0] : rr == 1 ? L[1] : rr == 2 ? L[2] : L[3];
      if constexpr ((ABL & 4) != 0) {
        sink += w0 + w1 + float(s0 + s1 + Lr);
        continue;
      }
      float* orow = a.out + long(mb * NN + rr) * a.out_stride;
      if (s0 >= 0 && pos0 < Lr) orow[s0 * kBlock + int(v_0_offset)] = w0;
      if (s1 >= 0 && pos1 < Lr) orow[s1 * kBlock + int(v_0_offset)] = w1;
    }
    if constexpr ((ABL & 4) != 0) {
      if (sink == 1234.5f) a.out[0] = sink;
    }
  }
}

// ------------------------------------------------------------------------------------------------ host
PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q{};
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q) == cudaSuccess && ptr,
                "cuTensorMapEncodeTiled not available");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

CUtensorMap encode(const void* base, int rank, const cuuint64_t* dims, const cuuint64_t* strides, const cuuint32_t* box,
                   CUtensorMapSwizzle sw, const char* name) {
  CUtensorMap m;
  cuuint32_t es[5] = {1, 1, 1, 1, 1};
  const CUresult r = encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, const_cast<void*>(base), dims, strides, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(", name, ") failed: ", int(r));
  return m;
}

template <int ABL>
void launch(const Args& a, const CUtensorMap& tm_q, const CUtensorMap& tm_kvp, int G) {
  static bool set_dev[64] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  if (!set_dev[dev]) {
    TORCH_CHECK(cudaFuncSetAttribute(pf_logits_kernel<ABL>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes(kMaxG)) == cudaSuccess,
                "pf smem attribute (", smem_bytes(kMaxG), " B)");
    set_dev[dev] = true;
  }
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  pf_logits_kernel<ABL><<<sms, kThreads, smem_bytes(G), at::cuda::getCurrentCUDAStream()>>>(a, tm_q, tm_kvp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void logits_prefill(at::Tensor q, at::Tensor kv, at::Tensor weights, at::Tensor ctx, at::Tensor block_table, at::Tensor gstart,
                    at::Tensor out, int64_t abl) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kFloat8_e4m3fn && q.dim() == 4 && q.is_contiguous() && q.size(1) == NN &&
                  q.size(2) == kHeads && q.size(3) == kHeadDim, "q: [B, 4, 32, 128] fp8");
  const int B = int(q.size(0));
  const int P = int(kv.size(1));
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == at::kByte && kv.dim() == 4 && (P == 64 || P == 32) && kv.size(2) == 1 &&
                  kv.size(3) == kHeadDim + 4 && kv.stride(0) % 16 == 0 && kv.stride(0) >= P * (kHeadDim + 4) &&
                  reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0, "kv: fused fp8 index-K pages");
  TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == at::kFloat && weights.dim() == 2 && weights.size(1) == kHeads &&
                  weights.stride(0) == kHeads && weights.size(0) >= long(B) * NN &&
                  reinterpret_cast<uintptr_t>(weights.data_ptr()) % 16 == 0, "weights [rows, 32] f32");
  for (auto* p : {&ctx, &block_table, &gstart})
    TORCH_CHECK(p->is_cuda() && p->scalar_type() == at::kInt, "ctx / block_table / gstart int32");
  TORCH_CHECK(ctx.is_contiguous() && ctx.numel() >= long(B) * NN && block_table.dim() == 2 && block_table.stride(1) == 1 &&
                  block_table.size(0) >= B && gstart.is_contiguous() && gstart.dim() == 1, "metadata");
  const int G = int(gstart.numel()) - 1;
  TORCH_CHECK(G >= 0 && G <= kMaxG, "groups <= ", kMaxG);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.dim() == 2 && out.stride(1) == 1 &&
                  out.size(0) >= long(B) * NN && out.stride(0) % 4 == 0, "out [rows, width] f32");
  if (!B || !G) return;
  const c10::cuda::CUDAGuard guard(q.device());
  const cuuint64_t qd[2] = {cuuint64_t(kHeadDim), cuuint64_t(B) * NN * kHeads};
  const cuuint64_t qs[1] = {cuuint64_t(kHeadDim)};
  const cuuint32_t qb[2] = {cuuint32_t(kHeadDim), cuuint32_t(NN * kHeads)};
  const CUtensorMap tm_q = encode(q.data_ptr(), 2, qd, qs, qb, CU_TENSOR_MAP_SWIZZLE_128B, "q");
  const cuuint64_t kd[3] = {cuuint64_t(kHeadDim), cuuint64_t(P), cuuint64_t(kv.size(0))};
  const cuuint64_t kst[2] = {cuuint64_t(kHeadDim), cuuint64_t(kv.stride(0))};
  const cuuint32_t kbp[3] = {cuuint32_t(kHeadDim), cuuint32_t(P), 1};
  const CUtensorMap tm_kvp = encode(kv.data_ptr(), 3, kd, kst, kbp, CU_TENSOR_MAP_SWIZZLE_128B, "kv page");
  Args a{gstart.data_ptr<int>(), G, ctx.data_ptr<int>(), block_table.data_ptr<int>(), block_table.stride(0),
         int(block_table.size(1)), weights.data_ptr<float>(), reinterpret_cast<const uint8_t*>(kv.data_ptr()), kv.stride(0),
         out.data_ptr<float>(), out.stride(0), P};
  switch (int(abl)) {
    case 0: launch<0>(a, tm_q, tm_kvp, G); break;
    case 1: launch<1>(a, tm_q, tm_kvp, G); break;
    case 2: launch<2>(a, tm_q, tm_kvp, G); break;
    case 3: launch<3>(a, tm_q, tm_kvp, G); break;
    case 4: launch<4>(a, tm_q, tm_kvp, G); break;
    case 6: launch<6>(a, tm_q, tm_kvp, G); break;
    case 7: launch<7>(a, tm_q, tm_kvp, G); break;
    default: TORCH_CHECK(false, "abl in {0,1,2,3,4,6,7} (0 = production)");
  }
}

}  // namespace idx2f_pf4

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("logits_prefill", &idx2f_pf4::logits_prefill,
        "dense index logits with key tiles shared by groups of <= 4 pseudo-requests of one sequence (= DeepGEMM arithmetic)");
}
