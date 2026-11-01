"""Radix-select kernels for the DeepSeek-V4.1 indexer (kernels/candidates.cu).

Adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
src/brrr/candidates.py); re-hooked to stock vLLM 0.32.0. See LICENSES/BRRR-Apache-2.0.txt.

Drop-in replacements (same signature and output contract as stock):
  * select_candidate_blocks  - vllm/model_executor/kernels/attention/dsa/candidate_blocks.py:141
  * apply_candidate_mask     - candidate_blocks.py:192
  * indexer_topk             - SparseIndexerTopk.forward (vllm/model_executor/layers/indexer_topk.py:302)
Each returns False (doing nothing) when its inputs are outside the supported envelope; the patch wrapper
then calls the stock implementation, so callers never see a behaviour change.

Output-order notes (identical index *sets*, documented order differences):
  * select_candidate_blocks writes the selected block ids in ascending block order, stock writes them in
    descending score order (torch.topk sorted=True). The selected set is identical, including ties
    (lowest index first, torch gatherTopK fill order) and NaN/+inf/-inf handling. The only consumer on the
    CUDA path, apply_candidate_mask, only sets per-block flags (order-insensitive).
  * indexer_topk writes the selected positions in ascending position order followed by -1 padding; the
    consumers (compute_global_topk_indices_and_lens, FlashMLA sparse decode) count/ignore -1 entries and
    are order-insensitive.
"""

from __future__ import annotations

import os
from functools import cache

import torch

from ds41h import cuext


@cache
def ext():
    return cuext.load("ds41h_candidates", "candidates.cu")


def _bounds_ok(t: torch.Tensor | None, rows: int, repeat: int) -> bool:
    if t is None:
        return True
    return t.is_cuda and t.dtype == torch.int32 and t.dim() == 1 and t.shape[0] * repeat >= rows


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
) -> bool:
    """Fused stock select_candidate_blocks. Returns False if unsupported (caller must use stock)."""
    if not (logits.is_cuda and logits.dim() == 2 and logits.dtype == torch.float32):
        return False
    rows, width = logits.shape
    if not (out.is_cuda and out.dtype == torch.int32 and out.dim() == 2 and out.shape[0] >= rows and out.shape[1] >= topk_blocks):
        return False
    if topk_blocks <= 0 or block_size <= 0 or row_repeat <= 0:
        return False
    if not (_bounds_ok(row_ks, rows, row_repeat) and _bounds_ok(row_ke, rows, row_repeat)):
        return False
    if not rows:
        return True
    if not width:
        out.fill_(-1)  # stock :159-161
        return True
    if -(-width // block_size) > ext().max_select_blocks():
        return False
    ext().select_blocks(logits, row_ks, row_ke, int(topk_blocks), int(block_size), out, int(row_repeat))
    return True


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
) -> bool:
    """Fused stock apply_candidate_mask (in place). Returns False if unsupported."""
    if not (logits.is_cuda and logits.dim() == 2 and logits.dtype == torch.float32):
        return False
    rows, width = logits.shape
    if not (candidate_blocks.is_cuda and candidate_blocks.dtype == torch.int32 and candidate_blocks.dim() == 2):
        return False
    if candidate_blocks.shape[0] < rows or block_size <= 0 or row_repeat <= 0:
        return False
    if not (_bounds_ok(row_ks, rows, row_repeat) and _bounds_ok(row_ke, rows, row_repeat)):
        return False
    if not rows or not width:
        return True
    if -(-width // block_size) > ext().max_mask_blocks():
        return False
    ref = logits[: min(rows, 8)].clone() if _MASK_CHECK else None
    ext().mask_candidates(logits, row_ks, row_ke, candidate_blocks, int(block_size), int(row_repeat))
    if ref is not None:
        _mask_check(logits, ref, row_ks, row_ke, candidate_blocks, int(block_size), int(row_repeat))
    return True


# DS41H_MASK_CHECK=1 = in-situ proof of the mask early exit. The first <= 8 rows of every
# call are also masked with the full-width kernel (= deploy_s1) on a copy; differing elements in [start, end) of those rows are
# counted on the GPU (graph-capturable; counters created outside captures only) and logged from eager calls.
_MASK_CHECK = os.environ.get("DS41H_MASK_CHECK", "0") not in ("0", "")
_mask_cnt: dict = {}
_mask_calls = [0]


def _mask_check(logits, ref, row_ks, row_ke, cand, block_size, row_repeat):
    n, width = ref.shape
    ext().mask_candidates_full(ref, row_ks, row_ke, cand, block_size, row_repeat)
    c = _mask_cnt.get(logits.device)
    if c is None:
        if torch.cuda.is_current_stream_capturing():
            return
        c = _mask_cnt[logits.device] = torch.zeros(2, dtype=torch.int64, device=logits.device)
    brow = torch.arange(n, device=logits.device) // row_repeat
    end = row_ke[brow].to(torch.int64)
    start = row_ks[brow].to(torch.int64) if row_ks is not None else torch.zeros_like(end)
    col = torch.arange(width, device=logits.device)[None, :]
    inside = (col >= start[:, None]) & (col < end[:, None])
    c[0] += ((logits[:n].view(torch.int32) != ref.view(torch.int32)) & inside).sum()
    c[1] += 1
    _mask_calls[0] += 1
    if _mask_calls[0] % 64 == 1 and not torch.cuda.is_current_stream_capturing():
        import logging

        v = c.tolist()
        logging.getLogger("vllm.ds41h.candidates").warning("ds41h mask CHECK: differing elements %d in %d checked calls", v[0], v[1])


TOPK_MAX_WIDTH = int(os.environ.get("DS41H_TOPK_MAX_WIDTH", str(1 << 20)))


def indexer_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> bool:
    """Decode top-k over [0, row_end) per row (row_end as in SparseIndexerTopk._row_ends).

    Selection == the stock "torch" backend (indexer_topk.py:359-371): rows with row_end <= k return
    0..row_end-1, otherwise the k largest (torch key order, ties to the lowest position). Output in
    ascending position order, -1 padded. Returns False if unsupported.
    """
    if not (logits.is_cuda and logits.dim() == 2 and logits.dtype == torch.float32 and logits.stride(1) == 1):
        return False
    rows = logits.shape[0]
    if not (seq_lens.is_cuda and seq_lens.dtype == torch.int32 and seq_lens.is_contiguous()):
        return False
    if not (seq_lens.numel() == rows or seq_lens.numel() * next_n >= rows):
        return False
    if not (topk_indices.is_cuda and topk_indices.dtype == torch.int32 and topk_indices.dim() == 2 and topk_indices.stride(1) == 1):
        return False
    if topk_indices.shape[0] < rows or topk_indices.shape[1] < topk_tokens or logits.shape[1] > TOPK_MAX_WIDTH:
        return False
    if rows:
        ext().topk_rows(logits, seq_lens, int(next_n), int(topk_tokens), topk_indices, None, None)
    return True


def topk_sorted(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    k: int,
    out: torch.Tensor,
    out_lengths: torch.Tensor | None = None,
    visible: torch.Tensor | None = None,
) -> None:
    """BRRR select_sorted: top-k per row of [0, row_ends[r]) in position order (-1 padded).

    With `out_lengths` the selection is expanded x8 (block -> 8 positions, masked by `visible[r]`
    positions, default 8 * row_end) into out[r, :8k] and out_lengths[r] = number of covered positions.
    """
    ext().topk_rows(logits, row_ends, 1, int(k), out, out_lengths, visible)
