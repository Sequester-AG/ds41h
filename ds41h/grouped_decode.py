"""ds41h grouped multi-row sparse decode attention (opt-in DS41H_ATTN_GROUPED=1).

Why. Multi-request decode with k=7 runs 8 consecutive positions per request through FlashMLA's SM90 sparse fp8 decode kernel,
one query token per 64-row tile: the 16 local heads (TP4) are padded to 64 (75 % of the QK / PV MMA work is padding) and the
KV entries the 8 positions share are dequantized and multiplied once per position. In real decode (ab2q, 2K context) the
indexer top-512 sets of 4 consecutive rows have a union of ~780 keys (vs 4 x 512) and their 128-key SWA windows a union of
131. Here one 64-row tile = 4 consecutive decode rows x 16 heads, over the UNION of their key sets, with a 4-bit membership
mask per key (a row's score for a key outside its own set is -inf before the softmax): ~2.9x less attention work per row
and no head padding. c64: FlashMLA ~230 us / layer, 13 % of the step at the 350 W cap.

Numerics (lossless class): the same FlashMLA kernel code (legacy/sm90/decode/sparse at e2afa5f) with two edits -- the key
validity byte becomes the group's membership bits and the softmax masks with the row's bit -- so every row attends exactly
to its own keys: bf16 Q.K with fp32 accumulation, fp32 online softmax incl. the attention sink, P rounded to bf16 for PV,
fp32 accumulation, split-KV partials combined in fp32. Only the key order / split boundaries (fp32 summation order) differ
from stock. Grouping is correct for ANY 4 consecutive rows (masks are per row), also across request boundaries or for
padded rows; it is only less efficient there.

Build: like ds41h/prefill_attn.py -- edited copies of FlashMLA's SM90 sparse decode sources (DS41H_FLASHMLA_SRC, default
$DS41H_HOME/src/FlashMLA) + the scheduler / combine kernels + a pybind binding + the union builder, compiled into
DS41H_GDEC_BUILD (default $DS41H_HOME/build/flashmla_gdec); never touches vLLM's _flashmla_C.

Public API
  build() / ext()                         JIT build (nvcc, no GPU needed) / cached
  union(idx [T, W] int32, lens [T] int32 | None, width) -> (uidx [G, width] int32, umask [G, width] uint8, ulen [G] int32)
  decode(q16 [T, 16, 512], kv, uidx, ulen, umask, sink64, scale, out16 [T, 16, 512], extra=(kv, uidx, ulen, umask) | None)
  forward(layer, q, kv_cache, swa_metadata, attn_metadata, swa_only, output) -> bool   drop-in for the stock decode body
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("vllm.ds41h.attn")

ENABLED = os.environ.get("DS41H_ATTN_GROUPED", "0") not in ("0", "", "false", "False")
FLASHMLA_SRC = Path(os.environ.get("DS41H_FLASHMLA_SRC", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "src", "FlashMLA")))
BUILD_DIR = Path(os.environ.get("DS41H_GDEC_BUILD", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "build", "flashmla_gdec")))
EXT_NAME = "ds41h_flashmla_gdec"
GQ = 4  # rows per 64-row tile (x 16 heads)
SWA_WIDTH = 512  # union width of 4 x 128-key SWA windows (multiple of 64)
EXTRA_WIDTH = 2048  # union width of 4 x 512 top-k lists
KDIR = "csrc/cuda_kernels/legacy/sm90/decode/sparse"
FLASHMLA_SOURCES = (
    "csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
    "csrc/cuda_kernels/smxx/decode/combine/combine.cu",
)
FLASHMLA_INCLUDES = ("csrc", "csrc/3rdparty/kerutils/include", "csrc/3rdparty/cutlass/include", "csrc/3rdparty/cutlass/tools/util/include")

_EDITS_CONFIG = (
    ("namespace legacy::sm90::decode::sparse {",
     "namespace legacy::sm90::decode::ds41h_gdec {\n\n"
     "// ds41h: per-key membership bits (bit r = row r of the 4-row group attends to the key; 0 = invalid)\n"
     "struct GroupedDecodeParams : public SparseAttnDecodeParams {\n"
     "    const uint8_t* qmask;        // [b, topk]\n"
     "    const uint8_t* extra_qmask;  // [b, extra_topk]\n"
     "    int stride_qmask_b, stride_extra_qmask_b;\n"
     "};\n"),
    ("    bool is_kv_valid[NUM_K_BUFS][TOPK_BLOCK_SIZE];",
     "    uint8_t is_kv_valid[NUM_K_BUFS][TOPK_BLOCK_SIZE];   // ds41h: membership bits of the key (0 = invalid)"),
    ("devfunc(const SparseAttnDecodeParams &params, const TMAParams &tma_params);",
     "devfunc(const GroupedDecodeParams &params, const TMAParams &tma_params);"),
    ("static void run(const SparseAttnDecodeParams &params);", "static void run(const GroupedDecodeParams &params);"),
)
_EDITS_KERNEL = (
    ('#include "splitkv_mla.h"\n', ""),
    ('#include "components/dequant.h"', '#include "cuda_kernels/legacy/sm90/decode/sparse/components/dequant.h"'),
    ('#include "components/helpers.h"', '#include "cuda_kernels/legacy/sm90/decode/sparse/components/helpers.h"'),
    ('#include "config.h"', '#include "ds41h_gdec_config.h"'),
    ("namespace legacy::sm90::decode::sparse {",
     "namespace legacy::sm90::decode::ds41h_gdec {\nusing namespace ::legacy::sm90::decode::sparse;  // ds41h: FlashMLA's decode helpers (components/)"),
    ("    bool is_kv_valid[],", "    uint8_t is_kv_valid[],"),
    ("            if (!is_kv_valid[(i&1)+(i/2)*8+(idx_in_warpgroup%4)*2])",
     "            if (!(is_kv_valid[(i&1)+(i/2)*8+(idx_in_warpgroup%4)*2] & (1 << (idx_in_warpgroup / 32))))   // ds41h: warp w = row w"),
    ("::devfunc(const SparseAttnDecodeParams &params, const TMAParams &tma_params) {",
     "::devfunc(const GroupedDecodeParams &params, const TMAParams &tma_params) {"),
    ("""                    int2 indices = __ldg((int2*)(indices_base + lane_idx*2));
                    *(char2*)(&plan.is_kv_valid[buf_idx][lane_idx*2]) = {
                        is_index_valid(indices.x, 0),
                        is_index_valid(indices.y, 1)
                    };""",
     """                    int2 indices = __ldg((int2*)(indices_base + lane_idx*2));
                    // ds41h: the key's membership bits in the group (0 for invalid / out-of-length keys)
                    const uint8_t* qmask_base = IS_EXTRA_BLOCK
                        ? params.extra_qmask + batch_idx*params.stride_extra_qmask_b + (block_idx-args.num_orig_kv_blocks)*TOPK_BLOCK_SIZE
                        : params.qmask + batch_idx*params.stride_qmask_b + block_idx*TOPK_BLOCK_SIZE;
                    const uchar2 qm = __ldg((const uchar2*)(qmask_base + lane_idx*2));
                    *(uchar2*)(&plan.is_kv_valid[buf_idx][lane_idx*2]) = make_uchar2(
                        is_index_valid(indices.x, 0) ? qm.x : (unsigned char)0,
                        is_index_valid(indices.y, 1) ? qm.y : (unsigned char)0
                    );"""),
    ("flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const SparseAttnDecodeParams params,",
     "flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const GroupedDecodeParams params,"),
    ("void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &params) {",
     "void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const GroupedDecodeParams &params) {"),
    ("void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params) {",
     "void run_flash_splitkv_mla_fp8_sparse_kernel(const GroupedDecodeParams &params) {"),
)

GDEC_CU = r"""
// ds41h grouped sparse decode: FlashMLA SM90 sparse fp8 decode (e2afa5f, MIT) with 4 decode rows x 16 heads
// per 64-row tile over the union of their key sets (membership masks), + the union builder and a pybind binding.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cub/block/block_radix_sort.cuh>

#include "ds41h_gdec_kernel.cuh"
#include "cuda_kernels/smxx/decode/combine/combine.h"
#include "cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

namespace legacy::sm90::decode::ds41h_gdec {
template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V4, 64>(const GroupedDecodeParams &params);
}

namespace ds41h_gdec_union {  // named (an anonymous namespace makes nvcc's host stubs ambiguous next to the using-directive)
constexpr int UB_THREADS = 256;

// One CTA per group of 4 consecutive rows: the rows' valid entries (j < len, value >= 0) as 32-bit keys value << 2 | row
// (value < 2^30, checked on the host), CUB block radix sort over the used bits, then one output entry per distinct value
// (ascending) with the OR of its rows' bits. Deterministic.
template <int ITEMS>
__global__ void __launch_bounds__(UB_THREADS) gdec_union_kernel(const int* __restrict__ idx, int sidx, const int* __restrict__ len, int T,
                                                                int W, int* __restrict__ uidx, uint8_t* __restrict__ umask,
                                                                int* __restrict__ ulen, int WU, int end_bit) {
  constexpr int NMAX = UB_THREADS * ITEMS;
  using Sort = cub::BlockRadixSort<unsigned, UB_THREADS, ITEMS>;
  __shared__ union {
    typename Sort::TempStorage sort;
    unsigned key[NMAX];
  } sm;
  __shared__ int warp_sum[UB_THREADS / 32];
  const int g = blockIdx.x, tid = threadIdx.x;
  unsigned keys[ITEMS];
#pragma unroll
  for (int e = 0; e < ITEMS; ++e) {
    const int i = tid * ITEMS + e;
    const int r = i / W, j = i - r * W;
    unsigned k = 0xFFFFFFFFu;
    const int row = 4 * g + r;
    if (r < 4 && row < T) {
      const int L = len ? min(__ldg(len + row), W) : W;
      if (j < L) {
        const int v = __ldg(idx + (long)row * sidx + j);
        if (v >= 0) k = ((unsigned)v << 2) | (unsigned)r;
      }
    }
    keys[e] = k;
  }
  Sort(sm.sort).Sort(keys, 0, end_bit);  // the sentinel 0xFFFFFFFF stays above every valid key for end_bit < 32
  __syncthreads();
#pragma unroll
  for (int e = 0; e < ITEMS; ++e) sm.key[tid * ITEMS + e] = keys[e];
  __syncthreads();
  const unsigned* key = sm.key;
  const int base = tid * ITEMS;
  int cnt = 0;
#pragma unroll
  for (int e = 0; e < ITEMS; ++e) {
    const int i = base + e;
    const unsigned k = key[i];
    cnt += k != 0xFFFFFFFFu && (i == 0 || (key[i - 1] >> 2) != (k >> 2));
  }
  int inc = cnt;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, inc, o);
    if ((tid & 31) >= o) inc += y;
  }
  if ((tid & 31) == 31) warp_sum[tid >> 5] = inc;
  __syncthreads();
  int pre = 0, tot = 0;
#pragma unroll
  for (int w = 0; w < UB_THREADS / 32; ++w) pre += w < (tid >> 5) ? warp_sum[w] : 0, tot += warp_sum[w];
  int pos = pre + inc - cnt;
#pragma unroll
  for (int e = 0; e < ITEMS; ++e) {
    const int i = base + e;
    const unsigned k = key[i];
    if (k == 0xFFFFFFFFu || (i > 0 && (key[i - 1] >> 2) == (k >> 2))) continue;
    unsigned m = 0;
    for (int q = i; q < NMAX && q < i + 4; ++q) {
      const unsigned kq = key[q];
      if (kq == 0xFFFFFFFFu || (kq >> 2) != (k >> 2)) break;
      m |= 1u << (kq & 3u);
    }
    if (pos < WU) {
      uidx[(long)g * WU + pos] = (int)(k >> 2);
      umask[(long)g * WU + pos] = (uint8_t)m;
    }
    ++pos;
  }
  if (tid == 0) ulen[g] = min(tot, WU);
}
}  // namespace ds41h_gdec_union
using namespace ds41h_gdec_union;

// (uidx [G, WU] int32, umask [G, WU] uint8, ulen [G] int32) for idx [T, W] (row stride sidx) and lens [T] (optional)
std::vector<at::Tensor> gdec_union(const at::Tensor& idx, const c10::optional<at::Tensor>& lens, int64_t WU, int64_t max_value) {
  TORCH_CHECK(idx.is_cuda() && idx.scalar_type() == at::kInt && idx.dim() == 2 && idx.stride(1) == 1, "union: idx int32 [T, W]");
  const int T = int(idx.size(0)), W = int(idx.size(1));
  TORCH_CHECK(T % 4 == 0 && T > 0, "union: T % 4 == 0");
  TORCH_CHECK(WU % 64 == 0 && WU >= 64, "union: width % 64 == 0");
  const int need = 4 * W;
  TORCH_CHECK(need <= 2048, "union: 4 * W <= 2048");
  if (lens.has_value()) TORCH_CHECK(lens->is_cuda() && lens->scalar_type() == at::kInt && lens->is_contiguous() && lens->numel() >= T, "union: lens");
  const c10::cuda::CUDAGuard guard(idx.device());
  const int G = T / 4;
  auto uidx = at::empty({G, WU}, idx.options());
  auto umask = at::zeros({G, WU}, idx.options().dtype(at::kByte));
  auto ulen = at::empty({G}, idx.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int* lp = lens.has_value() ? lens->data_ptr<int>() : nullptr;
  TORCH_CHECK(max_value > 0 && max_value < (int64_t(1) << 30), "union: values must stay below 2^30 (key = value << 2 | row)");
  int end_bit = 2;
  while ((int64_t(1) << end_bit) < (max_value << 2)) ++end_bit;
  if (need <= 512)
    gdec_union_kernel<2><<<G, UB_THREADS, 0, stream>>>(idx.data_ptr<int>(), int(idx.stride(0)), lp, T, W, uidx.data_ptr<int>(), umask.data_ptr<uint8_t>(), ulen.data_ptr<int>(), int(WU), end_bit);
  else
    gdec_union_kernel<8><<<G, UB_THREADS, 0, stream>>>(idx.data_ptr<int>(), int(idx.stride(0)), lp, T, W, uidx.data_ptr<int>(), umask.data_ptr<uint8_t>(), ulen.data_ptr<int>(), int(WU), end_bit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {uidx, umask, ulen};
}

static int i32(int64_t v) {
  TORCH_CHECK(v >= 0 && v <= INT32_MAX, "stride out of int32 range");
  return int(v);
}

// q [G, 1, 64, 512] bf16 (= 4 rows x 16 heads), kv [nb, bs, 1, 584] (V4 fp8 records), indices [G, 1, W] int32 (W % 64 == 0),
// topk_length [G], attn_sink [64] float, qmask [G, W] uint8 (+ the same for the extra cache), out [G, 1, 64, 512].
// Returns {out, lse, tile_scheduler_metadata, num_splits}.
std::vector<at::Tensor> gdec_fwd(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& indices, const at::Tensor& topk_length,
                                 const at::Tensor& qmask, const c10::optional<at::Tensor>& attn_sink, const c10::optional<at::Tensor>& extra_kv,
                                 const c10::optional<at::Tensor>& extra_indices, const c10::optional<at::Tensor>& extra_topk_length,
                                 const c10::optional<at::Tensor>& extra_qmask, double sm_scale, at::Tensor out,
                                 c10::optional<at::Tensor> tile_md, c10::optional<at::Tensor> num_splits) {
  using namespace legacy::sm90::decode::ds41h_gdec;
  using bf16 = cutlass::bfloat16_t;
  auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 9 && prop->minor == 0, "ds41h grouped decode is built for sm_90a only");
  TORCH_CHECK(q.dim() == 4 && q.size(1) == 1 && q.size(2) == 64 && q.size(3) == 512 && q.scalar_type() == at::kBFloat16 && q.stride(3) == 1,
              "q must be bf16 [G, 1, 64, 512]");
  const int b = int(q.size(0));
  TORCH_CHECK(kv.dim() == 4 && kv.size(2) == 1 && kv.size(3) == 584 && kv.stride(1) == 584, "kv must be [nb, bs, 1, 584] (V4 fp8 records)");
  TORCH_CHECK(indices.dim() == 3 && indices.size(0) == b && indices.size(1) == 1 && indices.scalar_type() == at::kInt && indices.stride(2) == 1,
              "indices [G, 1, W] int32");
  const int topk = int(indices.size(2));
  TORCH_CHECK(topk % 64 == 0, "W % 64 == 0");
  TORCH_CHECK(topk_length.scalar_type() == at::kInt && topk_length.is_contiguous() && topk_length.numel() == b, "topk_length [G]");
  TORCH_CHECK(qmask.scalar_type() == at::kByte && qmask.dim() == 2 && qmask.size(0) == b && qmask.size(1) >= topk && qmask.stride(1) == 1, "qmask [G, W]");
  const bool has_extra = extra_kv.has_value();
  int extra_topk = 0;
  if (has_extra) {
    TORCH_CHECK(extra_indices.has_value() && extra_topk_length.has_value() && extra_qmask.has_value(), "extra cache needs indices / lengths / masks");
    TORCH_CHECK(extra_kv->dim() == 4 && extra_kv->size(2) == 1 && extra_kv->size(3) == 584 && extra_kv->stride(1) == 584, "extra kv");
    TORCH_CHECK(extra_indices->dim() == 3 && extra_indices->size(0) == b && extra_indices->scalar_type() == at::kInt && extra_indices->stride(2) == 1, "extra indices");
    extra_topk = int(extra_indices->size(2));
    TORCH_CHECK(extra_topk % 64 == 0, "extra W % 64 == 0");
    TORCH_CHECK(extra_topk_length->scalar_type() == at::kInt && extra_topk_length->is_contiguous() && extra_topk_length->numel() == b, "extra lengths");
    TORCH_CHECK(extra_qmask->scalar_type() == at::kByte && extra_qmask->dim() == 2 && extra_qmask->size(0) == b && extra_qmask->size(1) >= extra_topk &&
                    extra_qmask->stride(1) == 1, "extra qmask");
  }
  if (attn_sink.has_value()) TORCH_CHECK(attn_sink->scalar_type() == at::kFloat && attn_sink->is_contiguous() && attn_sink->numel() == 64, "sink [64] fp32");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 4 && out.size(0) == b && out.size(1) == 1 && out.size(2) == 64 && out.size(3) == 512 &&
                  out.stride(3) == 1, "out [G, 1, 64, 512]");
  const c10::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int num_sms = prop->multiProcessorCount;
  auto lse = at::empty({b, 1, 64}, q.options().dtype(at::kFloat));

  GroupedDecodeParams p{};
  p.b = b, p.s_q = 1, p.h_q = 64, p.h_kv = 1, p.d_qk = 512, p.d_v = 512;
  p.sm_scale = float(sm_scale), p.sm_scale_div_log2 = float(sm_scale * 1.4426950408889634);
  p.num_blocks = int(kv.size(0)), p.page_block_size = int(kv.size(1)), p.topk = topk;
  p.model_type = ModelType::V4, p.extra_model_type = ModelType::V4;
  p.q = (bf16*)q.data_ptr(), p.kv = (bf16*)kv.data_ptr(), p.indices = indices.data_ptr<int>(), p.topk_length = topk_length.data_ptr<int>();
  p.attn_sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
  p.lse = lse.data_ptr<float>(), p.out = (bf16*)out.data_ptr();
  if (has_extra) {
    p.extra_num_blocks = int(extra_kv->size(0)), p.extra_page_block_size = int(extra_kv->size(1)), p.extra_topk = extra_topk;
    p.extra_kv = (bf16*)extra_kv->data_ptr(), p.extra_indices = extra_indices->data_ptr<int>(), p.extra_topk_length = extra_topk_length->data_ptr<int>();
    p.stride_extra_kv_block = i32(extra_kv->stride(0)), p.stride_extra_kv_row = i32(extra_kv->stride(1));
    p.stride_extra_indices_b = i32(extra_indices->stride(0)), p.stride_extra_indices_s_q = i32(extra_indices->stride(1));
    p.extra_qmask = extra_qmask->data_ptr<uint8_t>(), p.stride_extra_qmask_b = i32(extra_qmask->stride(0));
  }
  p.stride_q_b = i32(q.stride(0)), p.stride_q_s_q = i32(q.stride(1)), p.stride_q_h_q = i32(q.stride(2));
  p.stride_kv_block = i32(kv.stride(0)), p.stride_kv_row = i32(kv.stride(1));
  p.stride_indices_b = i32(indices.stride(0)), p.stride_indices_s_q = i32(indices.stride(1));
  p.stride_lse_b = i32(lse.stride(0)), p.stride_lse_s_q = i32(lse.stride(1));
  p.stride_o_b = i32(out.stride(0)), p.stride_o_s_q = i32(out.stride(1)), p.stride_o_h_q = i32(out.stride(2));
  p.qmask = qmask.data_ptr<uint8_t>(), p.stride_qmask_b = i32(qmask.stride(0));
  p.num_sm = num_sms, p.stream = stream, p.enable_split_kv = true;

  const int num_sm_parts = std::max(num_sms, 1);  // Decode_Sm90_Impl::get_meta(64, 1)
  if (!tile_md.has_value() || !tile_md->defined()) {
    tile_md = at::empty({num_sm_parts, int64_t(sizeof(DecodingSchedMeta) / 4)}, q.options().dtype(at::kInt));
    num_splits = at::empty({b + 1}, q.options().dtype(at::kInt));
    GetDecodeSchedMetaParams sp = {b, 1, 64, 5, num_sm_parts, topk, extra_topk, p.topk_length, has_extra ? p.extra_topk_length : nullptr, nullptr,
                                   (DecodingSchedMeta*)tile_md->data_ptr(), num_splits->data_ptr<int>(), stream};
    smxx::decode::run_get_decoding_sched_meta_kernel(sp);
  }
  TORCH_CHECK(num_splits.has_value() && num_splits->defined() && num_splits->numel() == b + 1, "num_splits");
  p.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_md->data_ptr();
  p.num_splits_ptr = num_splits->data_ptr<int>();
  p.num_sm_parts = num_sm_parts;
  auto lse_accum = at::empty({b + num_sm_parts, 1, 64}, q.options().dtype(at::kFloat));
  auto o_accum = at::empty({b + num_sm_parts, 1, 64, 512}, q.options().dtype(at::kFloat));
  p.lse_accum = lse_accum.data_ptr<float>(), p.o_accum = o_accum.data_ptr<float>();
  p.stride_lse_accum_split = i32(lse_accum.stride(0)), p.stride_lse_accum_s_q = i32(lse_accum.stride(1));
  p.stride_o_accum_split = i32(o_accum.stride(0)), p.stride_o_accum_s_q = i32(o_accum.stride(1)), p.stride_o_accum_h_q = i32(o_accum.stride(2));
  run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V4, 64>(p);
  smxx::decode::run_flash_mla_combine_kernel(p);
  return {out, lse, *tile_md, *num_splits};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("union", &gdec_union, "union of 4 consecutive rows' key lists with membership bits (ds41h)", pybind11::arg("idx"), pybind11::arg("lens"),
        pybind11::arg("width"), pybind11::arg("max_value"));
  m.def("fwd", &gdec_fwd, "grouped sparse decode (4 rows x 16 heads per 64-row tile over the key-set union; FlashMLA SM90 fp8 kernel)",
        pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("indices"), pybind11::arg("topk_length"), pybind11::arg("qmask"),
        pybind11::arg("attn_sink") = pybind11::none(), pybind11::arg("extra_kv") = pybind11::none(), pybind11::arg("extra_indices") = pybind11::none(),
        pybind11::arg("extra_topk_length") = pybind11::none(), pybind11::arg("extra_qmask") = pybind11::none(), pybind11::arg("sm_scale") = 1.0,
        pybind11::arg("out"), pybind11::arg("tile_md") = pybind11::none(), pybind11::arg("num_splits") = pybind11::none());
}
"""


def _write_if_changed(path: Path, text: str) -> None:
    if not path.exists() or path.read_text() != text:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)


def _apply_edits(text: str, edits: tuple, name: str) -> str:
    for old, new in edits:
        if text.count(old) != 1:
            raise RuntimeError(f"ds41h grouped decode: {name} edit anchor found {text.count(old)}x (FlashMLA changed?): {old[:80]!r}")
        text = text.replace(old, new)
    return text


def _sources(src: Path) -> Path:
    base = src / KDIR
    cfg = _apply_edits((base / "config.h").read_text(), _EDITS_CONFIG, "config.h")
    ker = _apply_edits((base / "splitkv_mla.cuh").read_text(), _EDITS_KERNEL, "splitkv_mla.cuh")
    _write_if_changed(BUILD_DIR / "ds41h_gdec_config.h", cfg)
    _write_if_changed(BUILD_DIR / "ds41h_gdec_kernel.cuh", ker)
    out = BUILD_DIR / "ds41h_gdec.cu"
    _write_if_changed(out, GDEC_CU)
    return out


def build(verbose: bool | None = None) -> Any:
    """Build (or reuse) the extension in BUILD_DIR. Needs nvcc, not a GPU."""
    from torch.utils.cpp_extension import load

    from ds41h.cuext import NVCC_FLAGS

    src = FLASHMLA_SRC
    missing = [s for s in FLASHMLA_SOURCES + (f"{KDIR}/splitkv_mla.cuh", f"{KDIR}/config.h") if not (src / s).exists()]
    if missing:
        raise FileNotFoundError(f"FlashMLA sources not found under {src}: {missing} (set DS41H_FLASHMLA_SRC)")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    main = _sources(src)
    cuda_home = os.environ.get("CUDA_HOME", "")
    includes = [str(src / p) for p in FLASHMLA_INCLUDES]
    if cuda_home and (Path(cuda_home) / "include" / "cccl").exists():
        includes.append(str(Path(cuda_home) / "include" / "cccl"))
    return load(
        name=EXT_NAME,
        sources=[str(main)] + [str(src / s) for s in FLASHMLA_SOURCES],
        extra_cflags=["-O3", "-std=c++20", "-DFLASH_MLA_IS_BUILD_ON_CUDA", "-DKERUTILS_IS_BUILD_ON_CUDA"],
        # FlashMLA setup.py nvcc flags (kerutils defines KERUTILS_IS_BUILD_ON_CUDA itself under __CUDACC__)
        extra_cuda_cflags=list(NVCC_FLAGS) + ["-Xptxas=--warn-on-spills", "-DFLASH_MLA_IS_BUILD_ON_CUDA", "-DNDEBUG", "-D_USE_MATH_DEFINES",
                                              "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
                                              "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--expt-extended-lambda", "--diag-suppress=445"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=includes + [str(BUILD_DIR)],
        build_directory=str(BUILD_DIR),
        verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")) if verbose is None else verbose,
    )


@functools.cache
def ext() -> Any:
    return build()


def union(idx, lens, width: int, max_value: int = 1 << 29):
    """(uidx [G, width] int32, umask [G, width] uint8, ulen [G] int32) for rows 4g..4g+3 of idx [T, W] (T % 4 == 0); every value
    must be < max_value (< 2^30; the cache's num_blocks * page_block_size), which also bounds the radix-sort bits."""
    return tuple(ext().union(idx, lens, int(width), int(max_value)))


_sink_cache: dict = {}


def _sink64(sink):
    """sink[:16] repeated for the 4 rows of a 64-row tile (fp32), cached per source tensor. The entry keeps the source tensor
    alive, so its address cannot be handed to another tensor while the entry lives (in one pytest process a
    freed test sink's address was reused by a later case's sink -> stale values; serving sinks are persistent parameters),
    and is refreshed when the source is modified in place (version counter)."""
    key = (sink.data_ptr(), sink.device)
    hit = _sink_cache.get(key)
    if hit is None or hit[1] != sink._version:
        s = sink[:16].float().repeat(GQ).contiguous()
        _sink_cache[key] = (sink, sink._version, s)
        return s
    return hit[2]


def decode(q16, kv, uidx, ulen, umask, sink, scale: float, out16, extra=None, sched=None):
    """Grouped decode of T rows (T % 4 == 0): q16 / out16 [T, 16, 512] contiguous; (uidx, ulen, umask) from union() over the
    rows' main-cache indices; extra = (kv, uidx, ulen, umask) for the second (compressed) cache or None. sched = (tile_md,
    num_splits) of an earlier call with IDENTICAL lengths / widths (FlashMLA's scheduler is a pure function of them) or None
    (computed here). Returns (tile_md, num_splits)."""
    T = q16.shape[0]
    G = T // GQ
    qg, og = q16.view(G, 1, 64, 512), out16.view(G, 1, 64, 512)
    kw = {}
    if extra is not None:
        ekv, eidx, elen, emask = extra
        kw = dict(extra_kv=ekv, extra_indices=eidx.view(G, 1, -1), extra_topk_length=elen, extra_qmask=emask)
    if sched is not None:
        kw.update(tile_md=sched[0], num_splits=sched[1])
    res = ext().fwd(qg, kv, uidx.view(G, 1, -1), ulen, umask, attn_sink=_sink64(sink) if sink is not None else None, sm_scale=float(scale),
                    out=og, **kw)
    return res[2], res[3]


def _forward_cache() -> dict | None:
    """Per-forward dict (vLLM forward context): the SWA union of a KV-cache group is shared by its layers (the SWA metadata is
    built once per step, never rewritten inside the forward). None outside a forward context."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            return None
        return get_forward_context().additional_kwargs.setdefault("ds41h.gdec", {})
    except Exception:  # noqa: BLE001
        return None


def _width(w: int) -> int:
    """Union width for 4 rows of w keys (multiple of 64), 0 if the union builder cannot hold it (4 w > 2048)."""
    n = GQ * int(w)
    return 0 if n > 2048 or n <= 0 else (n + 63) // 64 * 64


def _env_on(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "no", "off")


# Shared per-step work, both bit-identical:
#  S3a DS41H_GDEC_SHARE=1 (default): the layers of one index source (V4.1 config index_source_layer_ids 2, 8, 14, 20, 24, 28,
#      32, 36; consumers reuse the most recent source's top-k buffer and its kv source's cache / block table) see IDENTICAL
#      inputs to compute_global_topk_indices_and_lens, the key-set union and FlashMLA's split-KV scheduler -> computed once
#      per index source and step (38 -> 8 chains of 4 kernels per step) and reused, like stock vLLM shares its planner per
#      layer type. The SWA union and the SWA-only layers' schedule are shared the same way. DS41H_GDEC_SHARE_CHECK=1
#      recomputes everything on every reuse (incl. the attention output with a fresh schedule) and counts differing elements
#      on the GPU (log: "ds41h gdec CHECK").
#  S3b DS41H_GDEC_NOPAD=1 (default): in decode-only steps the grouped path will take (> 8 rows, T % 4 == 0, causal 128-key
#      SWA lists), q and the attention output keep the 16 local heads instead of padded_heads (64): the stock fused
#      qnorm/rope/KV-insert op pads to n_local_heads (== no padding, same kernel) and _alloc_attn_out allocates 16 heads, so
#      the q[:, :16].contiguous() and output[:, :16].copy_() of every layer disappear (dsv41_o_proj reads attn_out[:, :16]
#      either way). Stock FlashMLA fallbacks re-pad (ds41h/attn.py). DS41H_GDEC_NOPAD_CHECK=1 also runs the padded op on a
#      clone of q and counts differing q elements.
SHARE = _env_on("DS41H_GDEC_SHARE", "1")
SHARE_CHECK = _env_on("DS41H_GDEC_SHARE_CHECK", "0")
NOPAD = _env_on("DS41H_GDEC_NOPAD", "1")
NOPAD_CHECK = _env_on("DS41H_GDEC_NOPAD_CHECK", "0")
_check: dict = {}  # device -> int64 counters [union, sched, out, q_nopad, checks, q_checks]
_check_calls = [0]


def _check_counter(device):
    """The device's CHECK counters, or None while a CUDA graph is being captured and they do not exist yet (creating them
    inside a capture would put their zero-fill into the graph: every replay would reset them)."""
    import torch

    c = _check.get(device)
    if c is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        c = torch.zeros(6, dtype=torch.int64, device=device)
        _check[device] = c
    return c


def _check_log(device) -> None:
    """Log the CHECK counters from eager (non-captured) calls only (the read syncs)."""
    import torch

    _check_calls[0] += 1
    if _check_calls[0] % 256 != 1 or torch.cuda.is_current_stream_capturing():
        return
    c = _check_counter(device)
    if c is None:
        return
    v = c.tolist()
    logger.warning("ds41h gdec CHECK: differing elements union %d sched %d out %d q_nopad %d (share checks %d, nopad checks %d)", *v)


def _capturing() -> bool:
    import torch

    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def forward(layer: Any, q, kv_cache, swa_metadata: Any, attn_metadata: Any, swa_only: bool, output) -> bool:
    """Drop-in for the stock _forward_decode body (vLLM 0.32.0 nvidia/flashmla.py:176-246) for > 8 decode rows; False = not taken."""
    import torch

    T = int(swa_metadata.num_decode_tokens)
    if T % GQ or T <= 8 or q.dim() != 3 or q.shape[-1] != 512 or output.dim() != 3:
        return False
    nh = 16
    swa_idx = swa_metadata.decode_swa_indices[:T].reshape(T, -1)
    if swa_idx.shape[1] != int(getattr(layer, "window_size", 128)):
        return False  # causal decode rows only (the DSpark non-causal 192-wide lists are left to stock: tests showed a mismatch)
    sw = _width(swa_idx.shape[1])
    if sw == 0:
        return False
    if not swa_only and _width(layer.topk_indices_buffer.shape[-1]) == 0:
        return False
    q16 = q[:T, :nh]
    if not q16.is_contiguous():
        q16 = q16.contiguous()
    out_direct = output.shape[1] == nh and output[:T].is_contiguous()
    out16 = output[:T] if out_direct else torch.empty((T, nh, 512), dtype=output.dtype, device=output.device)
    swa_cache = layer.swa_cache_layer.kv_cache.unsqueeze(-2)
    cache = _forward_cache()
    cap = _capturing()
    key = ("swa", id(swa_metadata), swa_idx.data_ptr(), T, cap)
    hit = cache.get(key) if cache is not None else None
    if hit is None:
        hit = union(swa_idx, swa_metadata.decode_swa_lens[:T], sw, swa_cache.shape[0] * swa_cache.shape[1])
        if cache is not None:
            cache[key] = hit
    su, sm, sl = hit
    extra = None
    share = SHARE and cache is not None
    if swa_only:
        # schedule of an SWA-only layer = f(SWA union lengths / width) -> shared by all SWA-only layers of the step
        # the schedule is a function of the union LENGTHS only; an SWA union's length per 4-row group is the number of distinct
        # window positions of its rows, the same in every SWA cache group (slot maps are injective per request) -> one schedule
        # for all SWA-only layers of the step (CHECK mode recomputes with the layer's own lengths)
        skey = ("sched_swa", T, cap) if share else None
    else:
        ekv = kv_cache.unsqueeze(-2)
        src = getattr(layer, "index_source_layer_id", None)
        ekey = None
        if share and src is not None:
            # NOT keyed by the SWA metadata: vLLM's hybrid KV manager spreads consecutive layers over several SWA cache groups,
            # each with its own SWA metadata object / buffers (0 hits with id(swa_metadata) in the key). The extra
            # union only reads token_to_req_indices and is_valid_token from it, whose CONTENTS are identical in every group
            # (same requests; slot_mapping >= 0 for the same tokens); CHECK mode recomputes with the layer's own tensors.
            ekey = ("extra", int(src), int(layer.compress_ratio), id(attn_metadata), layer.topk_indices_buffer.data_ptr(),
                    attn_metadata.block_table.data_ptr(), ekv.data_ptr(), int(attn_metadata.block_size), T, cap)
        ehit = cache.get(ekey) if ekey is not None else None
        fresh = None
        if ehit is None or SHARE_CHECK:
            from vllm.models.deepseek_v41.nvidia.flashmla import compute_global_topk_indices_and_lens

            block_size = attn_metadata.block_size // layer.compress_ratio
            glob, lens = compute_global_topk_indices_and_lens(
                layer.topk_indices_buffer[:T], swa_metadata.token_to_req_indices, attn_metadata.block_table[: swa_metadata.num_decodes],
                block_size, swa_metadata.is_valid_token[:T])
            glob = glob.reshape(T, -1)
            ew = _width(glob.shape[1])
            if ew == 0:
                return False
            fresh = union(glob, lens, ew, ekv.shape[0] * ekv.shape[1])
            if ehit is None:
                # (eu, em, el) + the metadata objects (pinned so their ids cannot be reused while the entry lives)
                ehit = (fresh, attn_metadata, swa_metadata)
                if ekey is not None:
                    cache[ekey] = ehit
            else:
                # uidx only where umask != 0: the tail beyond ulen is never written (torch.empty) and never attended
                c = _check_counter(q.device)
                (u0, m0, l0), (u1, m1, l1) = ehit[0], fresh
                if c is not None:
                    c[0] += (m0 != m1).sum() + (l0 != l1).sum() + ((u0 != u1) & (m1 != 0)).sum()
                    c[4] += 1
        eu, em, el = ehit[0]
        extra = (ekv, eu, el, em)
        skey = ("sched", ekey) if ekey is not None else None
    sched = cache.get(skey) if skey is not None else None
    res = decode(q16, swa_cache, su, sl, sm, layer.attn_sink, layer.scale, out16, extra, sched)
    if sched is None:
        if skey is not None:
            cache[skey] = res
    elif SHARE_CHECK:
        # what the unshared path computes for this layer (the freshly built union + a fresh schedule) into scratch, compared with the
        # shared-union / shared-schedule result
        o2 = torch.empty_like(out16)
        extra2 = extra if (swa_only or fresh is None) else (ekv, fresh[0], fresh[2], fresh[1])
        res2 = decode(q16, swa_cache, su, sl, sm, layer.attn_sink, layer.scale, o2, extra2, None)
        c = _check_counter(q.device)
        if c is not None:
            # DecodingSchedMeta: 7 defined ints + int _pad[1] (never set by FlashMLA's scheduler, never read)
            c[1] += (res2[0][:, :7] != sched[0][:, :7]).sum() + (res2[1] != sched[1]).sum()
            c[2] += (o2.view(torch.int16) != out16.view(torch.int16)).sum()
            c[4] += 1
    if SHARE_CHECK or NOPAD_CHECK:
        _check_log(q.device)
    if not out_direct:
        output[:T, :nh].copy_(out16)
    return True


# ------------------------------------------------------------------------------------------------ S3b: unpadded decode buffers
def nopad_decode(layer: Any) -> bool:
    """True iff this forward is a decode-only step that grouped decode takes for this layer (the conditions of forward() that
    depend on the step, evaluated from the forward context): then q / attn_out keep n_local_heads (16). _alloc_attn_out and
    _fused_qnorm_rope_kv_insert both ask, inside the same forward (FULL decode graphs capture both), so they always agree."""
    if not NOPAD:
        return False
    h = int(getattr(layer, "n_local_heads", 0) or 0)
    if h != 16 or h >= int(getattr(layer, "padded_heads", 0) or 0):
        return False
    try:
        from ds41h import attn as _attn
        from ds41h import flags

        if not _attn.GROUPED or _attn._grouped_failed[0]:
            return False
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            return False
        am = get_forward_context().attn_metadata
        if not isinstance(am, dict):
            return False
        md = am.get(layer.swa_cache_layer.prefix)
        if md is None or int(getattr(md, "num_prefills", 1)) != 0:
            return False
        T = int(md.num_decode_tokens)
        # rows the ds41h 16-head decode kernel takes first (attn.forward_decode, T <= DS41H_ATTN_MAX_ROWS) keep today's layout
        lo = max(8, int(flags.MAX_DECODE_TOKENS), int(_attn.MAX_ROWS) if flags.ATTN else 0)
        if T % GQ or T <= lo:
            return False
        idx = md.decode_swa_indices
        if idx is None or idx.numel() == 0 or int(idx[:T].reshape(T, -1).shape[1]) != int(getattr(layer, "window_size", 128)):
            return False
        if int(getattr(layer, "compress_ratio", 0) or 0) > 0:
            buf = getattr(layer, "topk_indices_buffer", None)
            if buf is None or _width(buf.shape[-1]) == 0:
                return False
        return True
    except Exception:  # noqa: BLE001 - never break the forward: padded stock layout
        return False


def install_nopad(cls: Any) -> None:
    """Wrap cls._alloc_attn_out / cls._fused_qnorm_rope_kv_insert (= vLLM's DeepseekV4FlashMLAAttention, possibly already wrapped
    by ds41h/prefill_attn.py NOPAD) for S3b. Idempotent."""
    if not NOPAD or "_ds41h_gdec_prev_alloc_attn_out" in cls.__dict__:
        return
    prev_alloc = cls._alloc_attn_out
    prev_insert = cls._fused_qnorm_rope_kv_insert

    def _alloc_attn_out(self, num_tokens, hidden_states):
        if nopad_decode(self):
            import torch

            return torch.empty((num_tokens, self.n_local_heads, self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        return prev_alloc(self, num_tokens, hidden_states)

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        if not nopad_decode(self):
            return prev_insert(self, q, kv, positions, attn_metadata)
        q_ref = q.clone() if NOPAD_CHECK else None
        saved = self.padded_heads
        self.padded_heads = self.n_local_heads  # the stock op then pads to 16 heads (== no padding), same kernel
        try:
            out = prev_insert(self, q, kv, positions, attn_metadata)
        finally:
            self.padded_heads = saved
        if NOPAD_CHECK:
            import torch

            ref = prev_insert(self, q_ref, kv, positions, attn_metadata)  # padded (64-head) op; re-inserts the same KV bytes
            c = _check_counter(out.device)
            h = out.shape[1]
            if c is not None:
                c[3] += (ref[:, :h].reshape(-1).view(torch.int16) != out.reshape(-1).view(torch.int16)).sum()
                c[5] += 1
        return out

    cls._ds41h_gdec_prev_alloc_attn_out = prev_alloc
    cls._ds41h_gdec_prev_fused_qnorm_rope_kv_insert = prev_insert
    cls._alloc_attn_out = _alloc_attn_out
    cls._fused_qnorm_rope_kv_insert = _fused_qnorm_rope_kv_insert
    logger.info("ds41h gdec: unpadded 16-head q / attn_out for grouped decode steps (S3b%s)", ", CHECK" if NOPAD_CHECK else "")
