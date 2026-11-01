"""ds41h 'prefill' component, part 2: sparse-MLA prefill without 16 -> 64 head padding on sm90.

Stock vLLM 0.32.0 pads the 16 local query heads (TP4) to ``padded_heads = 64``
(models/deepseek_v41/nvidia/flashmla.py ``get_padded_num_q_heads``) for both FlashMLA decode and
prefill, so ``flash_mla_sparse_fwd`` computes 4x the useful QK/PV work in prefill (and the q / output
buffers are 4x larger than needed). vLLM PR #61615 (merged 2026-10-22, after 0.32.0) passes the real
head count to FlashMLA's SM90 small-head (8/16/32) sparse prefill kernel added by
vllm-project/FlashMLA PR #26 (e2afa5f; 1.8-2.25x vs padding at 8/16 heads on H20) -- a 3-line change
on the vLLM side: ``q[..., :H]``, ``attn_sink[:H]``, ``out[..., :H]``.

The FlashMLA extension vendored in vLLM 0.32.0 (vllm/_flashmla_C.abi3.so, built from FlashMLA
5dceadb) does NOT contain the small-head kernel (no "SM90a supports 8, 16, 32" check / SMALL_HEADS
feature in the binary; it rejects h_q not in {64, 128}). So ds41h builds just the SM90 sparse-prefill
part of FlashMLA e2afa5f (csrc/cuda_kernels/legacy/sm90/prefill/sparse, d_qk = 512 instantiations)
plus a pybind11 binding (``BINDING_SRC`` below, a trimmed copy of FlashMLA's
csrc/api/sparse_prefill.cpp SM90 branch) as a separate torch extension: sm_90a only, sources from
$DS41H_HOME/src/FlashMLA (DS41H_FLASHMLA_SRC), output in $DS41H_HOME/build/flashmla_sm90
(DS41H_FLASHMLA_BUILD), never in site-packages, and without registering any torch.ops namespace
(vLLM's _flashmla_C keeps owning ``torch.ops._flashmla_C``). Prebuild with scripts/build_flashmla_sm90.sh.

Patch (scoped to the class + its module, ds41h/patch_prefill_attn.py):
  * ``DeepseekV4FlashMLAAttention._forward_prefill`` is wrapped to publish the running layer, and the
    module-global ``flash_mla_sparse_fwd`` it calls is replaced by ``dispatch_sparse_fwd``: for layers
    with n_local_heads in {8, 16, 32} it runs the small-head kernel on the head slices of the padded
    buffers (strided q / out are supported by the kernel), exactly PR #61615's slicing; otherwise
    (and if the extension is unavailable) it calls the stock function.
  * Optional (DS41H_PREFILL_ATTN_NOPAD=1): in prefill-only steps, ``_alloc_attn_out`` and
    ``_fused_qnorm_rope_kv_insert`` use n_local_heads instead of padded_heads, so q and the attention
    output are [T, 16, 512] instead of [T, 64, 512] (saves ~0.4 GB of HBM writes per layer at 8K
    tokens). Decode / mixed steps keep the padding the FP8 decode kernel requires, and so does every
    forward a CUDA graph records or replays (PIECEWISE / breakable, decoder-replay graphs): only fully
    eager forwards (cudagraph runtime mode NONE, as prefill is under FULL_DECODE_ONLY) drop it.

Switches (need DS41H=1 and DS41H_PREFILL != 0; all default ON -- unit tests + end-to-end KL gate passed;
set =0 to fall back):
  DS41H_PREFILL_ATTN=1          small-head sparse prefill (PR #61615 + FlashMLA #26)
  DS41H_PREFILL_ATTN_NOPAD=1    additionally drop the q / output head padding in prefill-only steps
  DS41H_PREFILL_ATTN_GROUPED=1  4 consecutive queries x 16 heads per CTA over the union of their key sets (needs
                                NOPAD; T % 4 tail and ineligible calls use the small-head kernel)
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from pathlib import Path
from typing import Any

from ds41h import flags

logger = logging.getLogger("vllm.ds41h.prefill")

_ON = flags.ENABLED and flags.PREFILL
ATTN = _ON and flags._flag("DS41H_PREFILL_ATTN", "1")  # default ON (unit tests + end-to-end KL gate)
NOPAD = ATTN and flags._flag("DS41H_PREFILL_ATTN_NOPAD", "1")
# Grouped sparse prefill: GQ = 4 consecutive queries x 16 heads = 64 rows per CTA over the UNION of their key sets
# (FlashMLA's 64-row SM90 kernel with a per-row-group key mask), so each gathered KV row serves up to 4 queries.
# Needs NOPAD (contiguous [T, 16, 512] q / out) and T % 4 == 0; otherwise the small-head kernel runs.
GROUPED = NOPAD and flags._flag("DS41H_PREFILL_ATTN_GROUPED", "1")
GROUPED_MIN_T = int(os.environ.get("DS41H_PREFILL_ATTN_GROUPED_MIN_T", "256"))
GQ = 4
FLASHMLA_SRC = Path(os.environ.get("DS41H_FLASHMLA_SRC", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "src", "FlashMLA")))
BUILD_DIR = Path(os.environ.get("DS41H_FLASHMLA_BUILD", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "build", "flashmla_sm90")))
SMALL_HEADS = (8, 16, 32)
EXT_NAME = "ds41h_flashmla_sm90"

# FlashMLA e2afa5f sources compiled into the extension (SM90 sparse prefill, d_qk = 512 only).
FLASHMLA_SOURCES = (
    "csrc/cuda_kernels/legacy/sm90/prefill/sparse/instantiations/phase1_k512.cu",
    "csrc/cuda_kernels/legacy/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu",
)
FLASHMLA_INCLUDES = (
    "csrc",
    "csrc/3rdparty/kerutils/include",
    "csrc/3rdparty/cutlass/include",
    "csrc/3rdparty/cutlass/tools/util/include",
)

# pybind11 binding: FlashMLA csrc/api/sparse_prefill.cpp (MIT, DeepSeek-AI / vLLM project), SM90 branch only,
# rewritten against the regular torch C++ API. Params are filled field by field (csrc/params.h
# SparseAttnFwdParams at e2afa5f, incl. the stride_out_* fields added by PR #26).
BINDING_SRC = r"""
// ds41h: pybind11 binding of FlashMLA's SM90 sparse prefill (vllm-project/FlashMLA e2afa5f incl. PR #26
// small-head kernel). Adapted from FlashMLA csrc/api/sparse_prefill.cpp (MIT): SM90, d_qk = 512 only.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "params.h"
#include "cuda_kernels/legacy/sm90/prefill/sparse/phase1.h"

namespace {
int as_int(int64_t v) {
  TORCH_CHECK(v >= 0 && v <= INT32_MAX, "stride/size out of int32 range: ", v);
  return static_cast<int>(v);
}
}  // namespace

std::vector<at::Tensor> sparse_prefill_fwd(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& indices, double sm_scale,
                                           int64_t d_v, const c10::optional<at::Tensor>& attn_sink,
                                           const c10::optional<at::Tensor>& topk_length, c10::optional<at::Tensor> out_) {
  auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 9 && prop->minor == 0, "ds41h FlashMLA sparse prefill is built for sm_90a only");
  TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && indices.dim() == 3, "q, kv, indices must be 3-D");
  const int64_t s_q = q.size(0), h_q = q.size(1), d_qk = q.size(2), s_kv = kv.size(0), h_kv = kv.size(1), topk = indices.size(2);
  const bool small = h_q == 8 || h_q == 16 || h_q == 32;
  TORCH_CHECK(small || h_q == 64 || h_q == 128, "unsupported h_q ", h_q, " (8, 16, 32, 64, 128)");
  TORCH_CHECK(h_kv == 1, "sparse prefill requires h_kv == 1, got ", h_kv);
  TORCH_CHECK(d_qk == 512, "this build has d_qk = 512 only, got ", d_qk);
  TORCH_CHECK(d_v == 512, "d_v must be 512, got ", d_v);
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kv.scalar_type() == at::kBFloat16, "q, kv must be bf16");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
  TORCH_CHECK(kv.size(2) == d_qk && indices.size(0) == s_q && indices.size(1) == h_kv, "shape mismatch");
  TORCH_CHECK(q.stride(2) == 1 && kv.stride(2) == 1 && indices.stride(2) == 1, "last dims must be contiguous");
  if (attn_sink.has_value()) {
    TORCH_CHECK(attn_sink->is_cuda() && attn_sink->scalar_type() == at::kFloat && attn_sink->dim() == 1 && attn_sink->size(0) == h_q &&
                    attn_sink->stride(0) == 1, "attn_sink must be contiguous float32 [h_q]");
  }
  if (topk_length.has_value()) {
    TORCH_CHECK(topk_length->is_cuda() && topk_length->scalar_type() == at::kInt && topk_length->dim() == 1 &&
                    topk_length->size(0) == s_q && topk_length->stride(0) == 1, "topk_length must be contiguous int32 [s_q]");
  }
  const c10::cuda::CUDAGuard guard(q.device());
  at::Tensor out = out_.has_value() ? *out_ : at::empty({s_q, h_q, d_v}, q.options());
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 3 && out.size(0) == s_q && out.size(1) == h_q && out.size(2) == d_v &&
                  out.stride(2) == 1, "out must be bf16 [s_q, h_q, 512] with a contiguous last dim");
  if (small) {
    TORCH_CHECK(out.stride(1) >= d_v && out.stride(0) >= (h_q - 1) * out.stride(1) + d_v, "out must not have overlapping rows");
  } else {
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous for 64/128 heads");
  }
  at::Tensor lse = at::empty({s_q, h_q}, q.options().dtype(at::kFloat));
  at::Tensor max_logits = at::empty({s_q, h_q}, q.options().dtype(at::kFloat));
  if (s_q == 0) return {out, max_logits, lse};

  SparseAttnFwdParams p{};
  p.s_q = as_int(s_q); p.s_kv = as_int(s_kv); p.h_q = as_int(h_q); p.h_kv = as_int(h_kv);
  p.d_qk = as_int(d_qk); p.d_v = as_int(d_v); p.topk = as_int(topk);
  p.sm_scale = static_cast<float>(sm_scale);
  p.sm_scale_div_log2 = static_cast<float>(sm_scale * 1.4426950408889634);
  p.q = reinterpret_cast<bf16*>(q.data_ptr());
  p.kv = reinterpret_cast<bf16*>(kv.data_ptr());
  p.indices = indices.data_ptr<int>();
  p.attn_sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
  p.topk_length = topk_length.has_value() ? topk_length->data_ptr<int>() : nullptr;
  p.stride_q_s_q = as_int(q.stride(0)); p.stride_q_h_q = as_int(q.stride(1));
  p.stride_kv_s_kv = as_int(kv.stride(0)); p.stride_kv_h_kv = as_int(kv.stride(1));
  p.stride_indices_s_q = as_int(indices.stride(0)); p.stride_indices_h_kv = as_int(indices.stride(1));
  p.out = reinterpret_cast<bf16*>(out.data_ptr());
  p.max_logits = max_logits.data_ptr<float>();
  p.lse = lse.data_ptr<float>();
  p.stride_out_s_q = as_int(out.stride(0)); p.stride_out_h_q = as_int(out.stride(1));
  p.mode = SparseAttnFwdMode::Prefill;
  p.num_sm = prop->multiProcessorCount;
  p.stream = at::cuda::getCurrentCUDAStream();
  if (p.topk_length != nullptr)
    legacy::sm90::prefill::sparse_fwd::run_fwd_phase1_kernel<512, true>(p);
  else
    legacy::sm90::prefill::sparse_fwd::run_fwd_phase1_kernel<512, false>(p);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "ds41h FlashMLA sparse prefill launch failed: ", cudaGetErrorString(err));
  return {out, max_logits, lse};
}

std::vector<at::Tensor> build_union(const at::Tensor& indices, const c10::optional<at::Tensor>& lens, int64_t s_kv);
std::vector<at::Tensor> sparse_prefill_grouped_fwd(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& uidx, const at::Tensor& umask,
                                                   const at::Tensor& ulen, double sm_scale, const c10::optional<at::Tensor>& attn_sink,
                                                   c10::optional<at::Tensor> out_);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_union", &build_union, "union of 4 consecutive queries' key sets (ds41h)", pybind11::arg("indices"),
        pybind11::arg("lens") = pybind11::none(), pybind11::arg("s_kv"));
  m.def("sparse_prefill_grouped_fwd", &sparse_prefill_grouped_fwd, "grouped sparse prefill, 4 queries x 16 heads (ds41h)",
        pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("uidx"), pybind11::arg("umask"), pybind11::arg("ulen"),
        pybind11::arg("sm_scale"), pybind11::arg("attn_sink") = pybind11::none(), pybind11::arg("out") = pybind11::none());
  m.def("sparse_prefill_fwd", &sparse_prefill_fwd, "FlashMLA SM90 sparse prefill (8/16/32/64/128 heads, d_qk 512)",
        pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("indices"), pybind11::arg("sm_scale"), pybind11::arg("d_v") = 512,
        pybind11::arg("attn_sink") = pybind11::none(), pybind11::arg("topk_length") = pybind11::none(),
        pybind11::arg("out") = pybind11::none());
}
"""

# --------------------------------------------------------------------------------------------- grouped kernel
# The grouped kernel is FlashMLA e2afa5f's SM90 64-row sparse prefill (csrc/cuda_kernels/legacy/sm90/prefill/sparse/
# config.h + phase1.cuh, MIT) with these edits applied at build time (``_grouped_sources``):
#   * namespace legacy::sm90::prefill::ds41h_grouped (no clash with the stock instantiations in the same .so);
#   * params type GroupedParams = SparseAttnFwdParams + qmask (uint8 [s_q, topk], bit j = query j of the group has
#     the key); is_kv_valid becomes that byte (0 for invalid indices) and mask_rP tests bit (row / 16) -- with
#     wgmma's C layout all rows of a warp are one query, so the bit is per warp;
#   * attn_sink is indexed by head = row % 16 (rows are query-major: row = 16 * query + head).
# Rows = super-query s holds queries 4 s .. 4 s + 3 of a contiguous [T, 16, 512] q: h_q = 64, s_q = T / 4.
_GROUPED_EDITS_CONFIG = (
    ("namespace legacy::sm90::prefill::sparse_fwd {", "namespace legacy::sm90::prefill::ds41h_grouped {\n\n"
     "struct GroupedParams : public SparseAttnFwdParams {\n    const uint8_t* qmask;   // [s_q, topk] (stride_indices_s_q)\n};\n"),
    ("    bool is_kv_valid[2][B_TOPK];", "    uint8_t is_kv_valid[2][B_TOPK];   // ds41h: per-query bits of the key (0 = invalid)"),
    ("devfunc(const SparseAttnFwdParams &params, const TMAParams &tma_params);",
     "devfunc(const GroupedParams &params, const TMAParams &tma_params);"),
    ("static void run(const SparseAttnFwdParams &params);", "static void run(const GroupedParams &params);"),
)
_GROUPED_EDITS_PHASE1 = (
    ('#include "config.h"', '#include "ds41h_grouped_config.h"'),
    ('#include "../../helpers.h"', '#include "cuda_kernels/legacy/sm90/helpers.h"'),
    ('#include "small_heads.cuh"', ''),
    ("namespace legacy::sm90::prefill::sparse_fwd {", "namespace legacy::sm90::prefill::ds41h_grouped {"),
    ("::devfunc(const SparseAttnFwdParams &params, const TMAParams &tma_params) {",
     "::devfunc(const GroupedParams &params, const TMAParams &tma_params) {"),
    ("""            plan.bar_is_kv_valid_ready.wait(cur_bar_wait_phase);
            CUTE_UNROLL
            for (int row_idx = 0; row_idx < 2; ++row_idx) {""",
     """            plan.bar_is_kv_valid_ready.wait(cur_bar_wait_phase);
            const int qbit = 1 << (idx_in_warpgroup / 32);   // ds41h: rows 16 w .. 16 w + 15 of warp w = query w
            CUTE_UNROLL
            for (int row_idx = 0; row_idx < 2; ++row_idx) {"""),
    ("if (!plan.is_kv_valid[IS_WG1][col]) rP(i) = -INFINITY;", "if (!(plan.is_kv_valid[IS_WG1][col] & qbit)) rP(i) = -INFINITY;"),
    ("if (!plan.is_kv_valid[IS_WG1][col+1]) rP(i+1) = -INFINITY;",
     "if (!(plan.is_kv_valid[IS_WG1][col+1] & qbit)) rP(i+1) = -INFINITY;"),
    ("params.attn_sink[q_h_idx*B_H + get_AorC_row_idx(i, idx_in_warpgroup)]",
     "params.attn_sink[get_AorC_row_idx(i, idx_in_warpgroup) % 16]"),
    ("        bool is_token_valid[2][NUM_ROWS_PER_GROUP];",
     "        bool is_token_valid[2][NUM_ROWS_PER_GROUP];\n        uint8_t token_qmask[2][NUM_ROWS_PER_GROUP];"),
    ("                    is_token_valid[buf_idx][local_row] = is_cur_token_valid;",
     "                    is_token_valid[buf_idx][local_row] = is_cur_token_valid;\n"
     "                    token_qmask[buf_idx][local_row] = is_cur_token_valid ? __ldg(params.qmask + s_q_idx*params.stride_indices_s_q + offs) : (uint8_t)0;"),
    ("plan.is_kv_valid[buf_idx][local_row*NUM_GROUPS+group_idx] = is_token_valid[buf_idx][local_row];",
     "plan.is_kv_valid[buf_idx][local_row*NUM_GROUPS+group_idx] = token_qmask[buf_idx][local_row];"),
    ("sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params,",
     "sparse_attn_fwd_kernel(__grid_constant__ const GroupedParams params,"),
    ("void KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::run(const SparseAttnFwdParams &params) {",
     "void KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::run(const GroupedParams &params) {"),
)

GROUPED_CU = r"""
// ds41h: grouped sparse prefill (4 queries x 16 heads per CTA over the union of their key sets) + the union builder.
#include <climits>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "ds41h_grouped_phase1.cuh"

namespace legacy::sm90::prefill::ds41h_grouped {
template void KernelTemplate<512, true>::run(const GroupedParams& params);
}

namespace {
constexpr int GQ = 4, UB_THREADS = 256;

// One CTA per group of GQ consecutive queries: per-query bitmaps over [0, s_kv) in shared memory, then the union in
// ascending index order with a GQ-bit membership mask per entry. Deterministic (no order dependence).
__global__ void __launch_bounds__(UB_THREADS) build_union_kernel(const int* __restrict__ idx, int stride_idx, const int* __restrict__ lens,
                                                                  int topk, int s_q, int s_kv, int* __restrict__ uidx,
                                                                  uint8_t* __restrict__ umask, int* __restrict__ ulen, int umax) {
  extern __shared__ uint32_t bm[];  // [GQ][W], W = words of [lo, hi] (<= ceil(s_kv / 32) + 1)
  __shared__ int warp_tot[UB_THREADS / 32];
  __shared__ int s_lo, s_hi;
  const int b = blockIdx.x;
  // pass 1: index range of the group (the bitmap covers only [lo, hi]: one request's rows in a multi-request chunk)
  if (threadIdx.x == 0) s_lo = INT_MAX, s_hi = -1;
  __syncthreads();
  int lo = INT_MAX, hi = -1;
#pragma unroll
  for (int j = 0; j < GQ; ++j) {
    const int q = b * GQ + j;
    if (q >= s_q) break;
    const int len = lens != nullptr ? min(max(lens[q], 0), topk) : topk;
    const int* row = idx + int64_t(q) * stride_idx;
    for (int k = threadIdx.x; k < len; k += UB_THREADS) {
      const int v = __ldg(row + k);
      if (v >= 0 && v < s_kv) lo = min(lo, v), hi = max(hi, v);
    }
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, d));
    hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, d));
  }
  if ((threadIdx.x & 31) == 0) atomicMin(&s_lo, lo), atomicMax(&s_hi, hi);
  __syncthreads();
  const int base_word = s_lo == INT_MAX ? 0 : (s_lo >> 5);
  const int W = s_hi < 0 ? 0 : (s_hi >> 5) - base_word + 1;
  for (int i = threadIdx.x; i < GQ * W; i += UB_THREADS) bm[i] = 0u;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < GQ; ++j) {
    const int q = b * GQ + j;
    if (q >= s_q) break;
    const int len = lens != nullptr ? min(max(lens[q], 0), topk) : topk;
    const int* row = idx + int64_t(q) * stride_idx;
    for (int k = threadIdx.x; k < len; k += UB_THREADS) {
      const int v = __ldg(row + k);
      if (v >= 0 && v < s_kv) atomicOr(&bm[j * W + (v >> 5) - base_word], 1u << (v & 31));
    }
  }
  __syncthreads();
  const int per = (W + UB_THREADS - 1) / UB_THREADS;
  const int w0 = min(threadIdx.x * per, W), w1 = min(w0 + per, W);
  int cnt = 0;
  for (int w = w0; w < w1; ++w) {
    uint32_t o = 0u;
#pragma unroll
    for (int j = 0; j < GQ; ++j) o |= bm[j * W + w];
    cnt += __popc(o);
  }
  // block exclusive scan of cnt
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int incl = cnt;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, d);
    if (lane >= d) incl += v;
  }
  if (lane == 31) warp_tot[warp] = incl;
  __syncthreads();
  int base = 0, total = 0;
#pragma unroll
  for (int i = 0; i < UB_THREADS / 32; ++i) {
    base += i < warp ? warp_tot[i] : 0;
    total += warp_tot[i];
  }
  int pos = base + incl - cnt;
  int* uo = uidx + int64_t(b) * umax;
  uint8_t* mo = umask + int64_t(b) * umax;
  for (int w = w0; w < w1; ++w) {
    uint32_t m[GQ], o = 0u;
#pragma unroll
    for (int j = 0; j < GQ; ++j) {
      m[j] = bm[j * W + w];
      o |= m[j];
    }
    while (o) {
      const int bit = __ffs(o) - 1;
      o &= o - 1u;
      uint8_t mk = 0;
#pragma unroll
      for (int j = 0; j < GQ; ++j) mk |= uint8_t(((m[j] >> bit) & 1u) << j);
      uo[pos] = ((w + base_word) << 5) + bit;
      mo[pos] = mk;
      ++pos;
    }
  }
  if (threadIdx.x == 0) ulen[b] = total;
}
}  // namespace

// indices [s_q, 1, topk] int32 (row stride any), lens [s_q] int32 or None -> (uidx [S, umax] int32, umask [S, umax] u8,
// ulen [S] int32) with S = ceil(s_q / 4), umax = 4 * topk rounded up to 128.
std::vector<at::Tensor> build_union(const at::Tensor& indices, const c10::optional<at::Tensor>& lens, int64_t s_kv) {
  TORCH_CHECK(indices.is_cuda() && indices.scalar_type() == at::kInt && indices.dim() == 3 && indices.size(1) == 1 &&
                  indices.stride(2) == 1, "indices must be int32 [s_q, 1, topk] with a contiguous last dim");
  const int64_t s_q = indices.size(0), topk = indices.size(2);
  TORCH_CHECK(s_kv > 0 && s_kv <= (1 << 18), "s_kv out of range for the union bitmap: ", s_kv);
  if (lens.has_value()) {
    TORCH_CHECK(lens->is_cuda() && lens->scalar_type() == at::kInt && lens->dim() == 1 && lens->size(0) == s_q && lens->stride(0) == 1,
                "lens must be contiguous int32 [s_q]");
  }
  const c10::cuda::CUDAGuard guard(indices.device());
  const int64_t S = (s_q + GQ - 1) / GQ;
  const int64_t umax = (GQ * topk + 127) / 128 * 128;
  auto opts = indices.options();
  at::Tensor uidx = at::empty({S, umax}, opts);
  at::Tensor umask = at::empty({S, umax}, opts.dtype(at::kByte));
  at::Tensor ulen = at::empty({S}, opts);
  if (S == 0) return {uidx, umask, ulen};
  const int W = int((s_kv + 31) / 32) + 1;  // worst case: one group spans [0, s_kv)
  const size_t smem = size_t(GQ) * W * 4;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(build_union_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 140 * 1024) == cudaSuccess);
    attr = true;
  }
  build_union_kernel<<<dim3(unsigned(S)), dim3(UB_THREADS), smem, at::cuda::getCurrentCUDAStream()>>>(
      indices.data_ptr<int>(), int(indices.stride(0)), lens.has_value() ? lens->data_ptr<int>() : nullptr, int(topk), int(s_q), int(s_kv),
      uidx.data_ptr<int>(), umask.data_ptr<uint8_t>(), ulen.data_ptr<int>(), int(umax));
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "ds41h build_union launch failed: ", cudaGetErrorString(err));
  return {uidx, umask, ulen};
}

// q [T, 16, 512] bf16 contiguous (T % 4 == 0), kv [s_kv, 1, 512], (uidx, umask, ulen) from build_union, out [T, 16, 512]
// contiguous. Returns (out, max_logits [T, 16], lse [T, 16]).
std::vector<at::Tensor> sparse_prefill_grouped_fwd(const at::Tensor& q, const at::Tensor& kv, const at::Tensor& uidx, const at::Tensor& umask,
                                                   const at::Tensor& ulen, double sm_scale, const c10::optional<at::Tensor>& attn_sink,
                                                   c10::optional<at::Tensor> out_) {
  auto* prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major == 9 && prop->minor == 0, "ds41h grouped sparse prefill is built for sm_90a only");
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.size(1) == 16 && q.size(2) == 512 && q.is_contiguous(),
              "q must be contiguous bf16 [T, 16, 512]");
  const int64_t T = q.size(0);
  TORCH_CHECK(T % GQ == 0, "T must be a multiple of 4");
  const int64_t S = T / GQ;
  TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == at::kBFloat16 && kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) == 512 && kv.stride(2) == 1,
              "kv must be bf16 [s_kv, 1, 512]");
  TORCH_CHECK(uidx.scalar_type() == at::kInt && uidx.dim() == 2 && uidx.size(0) == S && uidx.is_contiguous() && uidx.size(1) % 128 == 0,
              "uidx shape");
  TORCH_CHECK(umask.scalar_type() == at::kByte && umask.sizes() == uidx.sizes() && umask.is_contiguous(), "umask shape");
  TORCH_CHECK(ulen.scalar_type() == at::kInt && ulen.dim() == 1 && ulen.size(0) == S && ulen.is_contiguous(), "ulen shape");
  if (attn_sink.has_value()) {
    TORCH_CHECK(attn_sink->is_cuda() && attn_sink->scalar_type() == at::kFloat && attn_sink->numel() >= 16 && attn_sink->stride(0) == 1,
                "attn_sink must be float32 [>= 16]");
  }
  const c10::cuda::CUDAGuard guard(q.device());
  at::Tensor out = out_.has_value() ? *out_ : at::empty_like(q);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.sizes() == q.sizes() && out.is_contiguous(), "out must be contiguous like q");
  at::Tensor lse = at::empty({T, 16}, q.options().dtype(at::kFloat));
  at::Tensor max_logits = at::empty({T, 16}, q.options().dtype(at::kFloat));
  if (T == 0) return {out, max_logits, lse};
  legacy::sm90::prefill::ds41h_grouped::GroupedParams p{};
  p.s_q = int(S); p.s_kv = int(kv.size(0)); p.h_q = 64; p.h_kv = 1; p.d_qk = 512; p.d_v = 512; p.topk = int(uidx.size(1));
  p.sm_scale = float(sm_scale);
  p.sm_scale_div_log2 = float(sm_scale * 1.4426950408889634);
  p.q = reinterpret_cast<bf16*>(q.data_ptr());
  p.kv = reinterpret_cast<bf16*>(kv.data_ptr());
  p.indices = uidx.data_ptr<int>();
  p.attn_sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
  p.topk_length = ulen.data_ptr<int>();
  p.stride_q_s_q = 64 * 512; p.stride_q_h_q = 512;
  p.stride_kv_s_kv = int(kv.stride(0)); p.stride_kv_h_kv = int(kv.stride(1));
  p.stride_indices_s_q = int(uidx.stride(0)); p.stride_indices_h_kv = 0;
  p.out = reinterpret_cast<bf16*>(out.data_ptr());
  p.max_logits = max_logits.data_ptr<float>();
  p.lse = lse.data_ptr<float>();
  p.stride_out_s_q = 64 * 512; p.stride_out_h_q = 512;
  p.mode = SparseAttnFwdMode::Prefill;
  p.num_sm = prop->multiProcessorCount;
  p.stream = at::cuda::getCurrentCUDAStream();
  p.qmask = umask.data_ptr<uint8_t>();
  legacy::sm90::prefill::ds41h_grouped::KernelTemplate<512, true>::run(p);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "ds41h grouped sparse prefill launch failed: ", cudaGetErrorString(err));
  return {out, max_logits, lse};
}
"""

NVCC_FLAGS = (
    "-O3",
    "-std=c++20",
    "-DNDEBUG",
    "-D_USE_MATH_DEFINES",
    "-DFLASH_MLA_IS_BUILD_ON_CUDA",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-lineinfo",
    "--diag-suppress=445",
    "-gencode=arch=compute_90a,code=sm_90a",
)


def _write_if_changed(path: Path, text: str) -> None:
    # Atomic (temp file + rename): the 4 TP workers may all reach the first build at once, and a
    # truncate-then-write could hand the baton holder's ninja a partial binding source.
    if not path.exists() or path.read_text() != text:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)


def _apply_edits(text: str, edits: tuple, name: str) -> str:
    for old, new in edits:
        if text.count(old) != 1:
            raise RuntimeError(f"ds41h grouped kernel: {name} edit anchor found {text.count(old)}x (FlashMLA changed?): {old[:80]!r}")
        text = text.replace(old, new)
    return text


def _grouped_sources(src: Path) -> Path:
    """Write the grouped-kernel sources (edited FlashMLA config.h / phase1.cuh + GROUPED_CU) into BUILD_DIR."""
    base = src / "csrc/cuda_kernels/legacy/sm90/prefill/sparse"
    cfg = _apply_edits((base / "config.h").read_text(), _GROUPED_EDITS_CONFIG, "config.h")
    ph1 = (base / "phase1.cuh").read_text()
    cut = ph1.index("template<int D_QK, bool HAVE_TOPK_LENGTH>\nvoid run_fwd_phase1_kernel(")
    ph1 = _apply_edits(ph1[:cut] + "}\n", _GROUPED_EDITS_PHASE1, "phase1.cuh")
    _write_if_changed(BUILD_DIR / "ds41h_grouped_config.h", cfg)
    _write_if_changed(BUILD_DIR / "ds41h_grouped_phase1.cuh", ph1)
    out = BUILD_DIR / "ds41h_grouped.cu"
    _write_if_changed(out, GROUPED_CU)
    return out


def build(verbose: bool | None = None) -> Any:
    """Build (or reuse) the extension in BUILD_DIR. Needs nvcc, not a GPU."""
    from torch.utils.cpp_extension import load

    src = FLASHMLA_SRC
    missing = [s for s in FLASHMLA_SOURCES if not (src / s).exists()]
    if missing:
        raise FileNotFoundError(f"FlashMLA sources not found under {src}: {missing} (set DS41H_FLASHMLA_SRC)")
    if "is_sm90_small_heads_h_q" not in (src / "csrc/params.h").read_text():
        raise RuntimeError(f"{src} predates FlashMLA PR #26 (no small-head prefill); check out e2afa5f or later")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    binding = BUILD_DIR / "ds41h_flashmla_binding.cu"
    _write_if_changed(binding, BINDING_SRC)
    grouped = _grouped_sources(src)
    cuda_home = os.environ.get("CUDA_HOME", "")
    includes = [str(src / p) for p in FLASHMLA_INCLUDES]
    if cuda_home and (Path(cuda_home) / "include" / "cccl").exists():  # cuda/std headers of the CUDA 13 pip toolkit
        includes.append(str(Path(cuda_home) / "include" / "cccl"))
    ptxas = ["-Xptxas=-v,--warn-on-spills"] if os.environ.get("DS41H_PTXAS_VERBOSE") else ["-Xptxas=--warn-on-spills"]
    return load(
        name=EXT_NAME,
        sources=[str(binding), str(grouped)] + [str(src / s) for s in FLASHMLA_SOURCES],
        extra_cflags=["-O3", "-std=c++20", "-DFLASH_MLA_IS_BUILD_ON_CUDA", "-DKERUTILS_IS_BUILD_ON_CUDA"],
        extra_cuda_cflags=list(NVCC_FLAGS) + ptxas,
        extra_ldflags=["-lcuda"],
        extra_include_paths=includes + [str(BUILD_DIR)],
        build_directory=str(BUILD_DIR),
        verbose=bool(os.environ.get("DS41H_BUILD_VERBOSE")) if verbose is None else verbose,
    )


@functools.cache
def ext() -> Any:
    return build()


@functools.cache
def available() -> bool:
    """Extension importable and running on sm90 (checked once; logs why not)."""
    try:
        import torch

        if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)):
            return False
        ext()
        return True
    except Exception:  # noqa: BLE001
        logger.exception("ds41h prefill attn: FlashMLA small-head extension unavailable; stock padded prefill kept")
        return False


def sparse_prefill_fwd(q, kv, indices, sm_scale: float, d_v: int = 512, attn_sink=None, topk_length=None, out=None):
    """(out, max_logits, lse) = FlashMLA SM90 sparse prefill; q [s_q, h_q, 512] bf16 (h_q in 8..128, head
    slices of a padded buffer allowed for 8/16/32), kv [s_kv, 1, 512] bf16, indices [s_q, 1, topk] int32."""
    return ext().sparse_prefill_fwd(q, kv, indices, float(sm_scale), int(d_v), attn_sink, topk_length, out)


def build_union(indices, topk_length, s_kv: int):
    """(uidx [S, umax] int32, umask [S, umax] uint8, ulen [S] int32): union of the key sets of queries 4 s .. 4 s + 3
    (ascending), bit j of umask = query 4 s + j has the key. indices [s_q, 1, topk] int32 (-1 / >= s_kv invalid)."""
    return tuple(ext().build_union(indices, topk_length, int(s_kv)))


def grouped_eligible(q, kv, out) -> bool:
    """Contiguous [T, 16, 512] q / out (NOPAD buffers); T % 4 != 0 is handled by the caller (tail on small heads)."""
    return (
        q.dim() == 3 and tuple(q.shape[1:]) == (16, 512) and q.is_contiguous() and q.shape[0] >= GQ
        and out is not None and out.shape == q.shape and out.is_contiguous() and 0 < kv.shape[0] <= (1 << 18)
    )


def sparse_prefill_grouped(q, kv, indices, sm_scale: float, attn_sink=None, topk_length=None, out=None):
    """Grouped sparse prefill (same results as sparse_prefill_fwd up to accumulation order): q / out contiguous
    [T, 16, 512] with T % 4 == 0. Returns (out, max_logits, lse)."""
    uidx, umask, ulen = build_union(indices, topk_length, kv.shape[0])
    sink = attn_sink[:16].contiguous() if attn_sink is not None else None
    return ext().sparse_prefill_grouped_fwd(q, kv, uidx, umask, ulen, float(sm_scale), sink, out)


def sparse_prefill_ref(q, kv, indices, sm_scale: float, d_v: int = 512, attn_sink=None, topk_length=None):
    """fp32 torch reference with FlashMLA's semantics (flash_mla_interface.py docstring / tests/ref.py):
    invalid indices (< 0 or >= s_kv) and positions >= topk_length are ignored; with attn_sink the output is
    scaled by exp(lse) / (exp(lse) + exp(sink)); lse/max_logits ignore the sink. Returns (out, max_logits, lse)."""
    import torch

    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    ct = torch.float64 if q.dtype == torch.float64 else torch.float32  # fp64 inputs -> fp64 reference
    idx = indices[:, 0, :].long()  # [s_q, topk]
    valid = (idx >= 0) & (idx < s_kv)
    if topk_length is not None:
        valid &= torch.arange(idx.shape[1], device=idx.device)[None, :] < topk_length.long()[:, None]
    k = kv[:, 0, :].to(ct)[idx.clamp(0, s_kv - 1)]  # [s_q, topk, d_qk]
    logits = torch.einsum("shd,std->sht", q.to(ct), k) * sm_scale
    logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
    max_logits = logits.amax(-1)
    lse = torch.logsumexp(logits, -1)
    p = torch.exp(logits - lse[..., None]).nan_to_num(0.0)
    out = torch.einsum("sht,std->shd", p, k[..., :d_v])
    if attn_sink is not None:
        sink = attn_sink.to(ct)[None, :]
        out = out * (1.0 / (1.0 + torch.exp(sink - lse))).nan_to_num(0.0)[..., None]
    return out, max_logits, lse


# ============================================================================================ vLLM glue
_STATE = threading.local()
_LOGGED: set[str] = set()


def _log_once(key: str, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(msg)


def _current_layer() -> Any:
    return getattr(_STATE, "layer", None)


# Debug: DS41H_PREFILL_ATTN_DUMP=<dir> saves (indices, topk_length, s_kv) of the DS41H_PREFILL_ATTN_DUMP_AT-th (default
# 3: past the startup dummy run and the first chunk of the first long prompt) prefill call with >= 4096 queries of
# every layer (rank 0 only, once per layer) for offline analysis of the per-query key sets.
DUMP_DIR = os.environ.get("DS41H_PREFILL_ATTN_DUMP", "")
DUMP_AT = int(os.environ.get("DS41H_PREFILL_ATTN_DUMP_AT", "3"))
_DUMPED: set = set()
_DUMP_SEEN: dict = {}


def _maybe_dump(layer: Any, q, kv, indices, topk_length) -> None:
    import torch

    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
        key = getattr(layer, "prefix", None) or id(layer)
        if key in _DUMPED or q.shape[0] < 4096:
            return
        _DUMP_SEEN[key] = _DUMP_SEEN.get(key, 0) + 1
        if _DUMP_SEEN[key] < DUMP_AT:
            return
        _DUMPED.add(key)
        Path(DUMP_DIR).mkdir(parents=True, exist_ok=True)
        name = str(key).replace(".", "_")
        torch.save({"indices": indices[:, 0].cpu(), "topk_length": None if topk_length is None else topk_length.cpu(),
                    "s_kv": int(kv.shape[0]), "s_q": int(q.shape[0]), "prefix": str(key),
                    "compress_ratio": getattr(layer, "compress_ratio", None)}, Path(DUMP_DIR) / f"{name}.pt")
    except Exception:  # noqa: BLE001
        logger.exception("ds41h prefill attn: index dump failed")


def make_dispatch(stock_fn: Any) -> Any:
    """Replacement for flashmla.py's module-global ``flash_mla_sparse_fwd`` (PR #61615 semantics)."""

    def dispatch_sparse_fwd(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None, out=None):
        import torch

        layer = _current_layer()
        if DUMP_DIR and layer is not None and q.is_cuda and not torch.cuda.is_current_stream_capturing():
            _maybe_dump(layer, q, kv, indices, topk_length)
        h = int(getattr(layer, "n_local_heads", 0) or 0)
        # Never load / build the extension inside a CUDA-graph capture (e.g. a drafter FULL graph); prefill
        # itself is eager, so the first eager prefill loads it.
        loaded = ext.cache_info().currsize > 0
        if not loaded and q.is_cuda and torch.cuda.is_current_stream_capturing():
            h = 0
        if (GROUPED and h == 16 and q.is_cuda and q.shape[0] >= GROUPED_MIN_T and d_v == 512 and grouped_eligible(q, kv, out)
                and available()):
            _log_once("grouped", "ds41h prefill attn: grouped (4 queries x 16 heads, key-set union) sparse prefill active")
            T = q.shape[0]
            T4 = T - T % GQ
            res = sparse_prefill_grouped(q[:T4], kv, indices[:T4], sm_scale, attn_sink,
                                         topk_length[:T4] if topk_length is not None else None, out[:T4])
            if T4 == T:
                return (out, res[1], res[2])
            # remaining 1..3 queries: small-head kernel (lse / max_logits are not used by vLLM's prefill)
            tail = sparse_prefill_fwd(q[T4:], kv, indices[T4:], sm_scale, d_v, attn_sink[:16] if attn_sink is not None else None,
                                      topk_length[T4:] if topk_length is not None else None, out[T4:])
            return (out, torch.cat([res[1], tail[1]]), torch.cat([res[2], tail[2]]))
        if h in SMALL_HEADS and q.is_cuda and q.dim() == 3 and q.shape[1] >= h and available():
            sink = attn_sink[:h] if attn_sink is not None else None
            o = out[:, :h] if out is not None else None
            res = sparse_prefill_fwd(q[:, :h], kv, indices, sm_scale, d_v, sink, topk_length, o)
            return (out if out is not None else res[0], res[1], res[2])
        return stock_fn(q=q, kv=kv, indices=indices, sm_scale=sm_scale, d_v=d_v, attn_sink=attn_sink, topk_length=topk_length, out=out)

    dispatch_sparse_fwd.__wrapped__ = stock_fn  # type: ignore[attr-defined]
    return dispatch_sparse_fwd


def _fully_eager_forward() -> bool:
    """True only for a forward that no CUDA graph records or replays.

    Checked live (not from the cached StepInfo): ``_alloc_attn_out`` runs in a graph segment while
    ``_fused_qnorm_rope_kv_insert`` may run in a breakable-cudagraph eager segment that is re-executed in
    Python at every replay (MRV1's wide ``_prepare_and_attn_eager`` region, the decoder-replay PIECEWISE
    graphs of the trimmed layers). The two must agree on the q / output head count, so NOPAD is only taken
    when the forward context says NONE, the stream is not capturing and no breakable capture is open (its
    eager segments run with stream capture paused). The StepInfo cache is also shared between a capture's
    warmup and capture runs (dataclasses.replace keeps additional_kwargs), so its flags are not enough."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return False
    mode = get_forward_context().cudagraph_runtime_mode
    if getattr(mode, "name", str(mode)) != "NONE":
        return False
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        if BreakableCUDAGraphCapture.is_active():
            return False
    except ImportError:
        pass
    return True


def _nopad_active(layer: Any) -> bool:
    if not NOPAD:
        return False
    h = int(getattr(layer, "n_local_heads", 0) or 0)
    if h not in SMALL_HEADS or h >= int(getattr(layer, "padded_heads", 0) or 0):
        return False
    from ds41h.patch_core import current_step

    st = current_step()
    return (
        st is not None
        and st.has_attn_metadata
        and not st.decode_only
        and st.num_decode_tokens == 0
        and not st.capturing
        and _fully_eager_forward()
        and available()
    )


def install_flashmla_prefill(module: Any) -> None:
    """Patch ``module`` = vllm.models.deepseek_v41.nvidia.flashmla (idempotent). Stock methods are kept on
    the class as ``_ds41h_stock_*``; the stock module function as ``dispatch.__wrapped__``."""
    cls = module.DeepseekV4FlashMLAAttention
    if "_ds41h_stock_forward_prefill" in cls.__dict__:
        return
    stock_fp = cls._forward_prefill

    def _forward_prefill(self, *args, **kwargs):
        prev = getattr(_STATE, "layer", None)
        _STATE.layer = self
        try:
            return stock_fp(self, *args, **kwargs)
        finally:
            _STATE.layer = prev

    _forward_prefill.__wrapped__ = stock_fp  # type: ignore[attr-defined]
    cls._ds41h_stock_forward_prefill = stock_fp
    cls._forward_prefill = _forward_prefill
    module.flash_mla_sparse_fwd = make_dispatch(module.flash_mla_sparse_fwd)

    if NOPAD:
        stock_alloc = cls._alloc_attn_out
        stock_insert = cls._fused_qnorm_rope_kv_insert

        def _alloc_attn_out(self, num_tokens, hidden_states):
            if _nopad_active(self):
                import torch

                return torch.empty((num_tokens, self.n_local_heads, self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
            return stock_alloc(self, num_tokens, hidden_states)

        def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
            if _nopad_active(self):
                saved = self.padded_heads
                self.padded_heads = self.n_local_heads  # the stock op then pads to H (== no padding)
                try:
                    return stock_insert(self, q, kv, positions, attn_metadata)
                finally:
                    self.padded_heads = saved
            return stock_insert(self, q, kv, positions, attn_metadata)

        cls._ds41h_stock_alloc_attn_out = stock_alloc
        cls._ds41h_stock_fused_qnorm_rope_kv_insert = stock_insert
        cls._alloc_attn_out = _alloc_attn_out
        cls._fused_qnorm_rope_kv_insert = _fused_qnorm_rope_kv_insert
    logger.info("ds41h prefill attn: patched %s small-head sparse prefill (nopad=%s)", module.__name__, NOPAD)
