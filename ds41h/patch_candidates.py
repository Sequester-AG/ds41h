"""CANDIDATES hook: fused radix-select kernels for the DeepSeek-V4.1 two-level indexer.

Patched symbols (stock vLLM 0.32.0):
  * vllm.model_executor.kernels.attention.dsa.candidate_blocks.select_candidate_blocks (:141)
  * vllm.model_executor.kernels.attention.dsa.candidate_blocks.apply_candidate_mask    (:192)
  * vllm.model_executor.layers.sparse_attn_indexer._select_candidate_blocks / _apply_candidate_mask
    (module-level aliases imported at :26-31 and called at :695-709 prefill, :847-866 decode).
  * optional (DS41H_CANDIDATES_TOPK=1, default off): SparseIndexerTopk.forward
    (vllm/model_executor/layers/indexer_topk.py:302), the decode top-512 after the candidate mask.
    Stock picks cooperative_topk on sm90 for <= 64 rows; enable only if tests/test_candidates.py --bench
    shows ours faster for the deployed shapes.
Unsupported inputs fall through to the stock functions. No allocations and no host syncs on the fast
path, so the hooks are safe inside FULL decode CUDA graphs.
"""

from __future__ import annotations

import functools
import logging
import os

from ds41h import flags
from ds41h.patches import register

TOPK_HOOK = os.environ.get("DS41H_CANDIDATES_TOPK", "0") not in ("0", "", "false")


def _wrap_select(stock):
    if getattr(stock, "_ds41h", False):
        return stock

    def select_candidate_blocks(logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat=1):
        from ds41h import candidates

        if not candidates.select_candidate_blocks(logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat):
            stock(logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat)

    select_candidate_blocks._ds41h = True
    select_candidate_blocks._ds41h_stock = stock
    select_candidate_blocks.__doc__ = stock.__doc__
    return select_candidate_blocks


def _wrap_mask(stock):
    if getattr(stock, "_ds41h", False):
        return stock

    def apply_candidate_mask(logits, row_ks, row_ke, candidate_blocks, block_size, row_repeat=1):
        from ds41h import candidates

        if not candidates.apply_candidate_mask(logits, row_ks, row_ke, candidate_blocks, block_size, row_repeat):
            stock(logits, row_ks, row_ke, candidate_blocks, block_size, row_repeat)

    apply_candidate_mask._ds41h = True
    apply_candidate_mask._ds41h_stock = stock
    apply_candidate_mask.__doc__ = stock.__doc__
    return apply_candidate_mask


@functools.cache
def _ext_ok() -> bool:
    """Build/load the extension now (never for the first time inside a CUDA-graph capture).

    A build/load failure must not break the import of the stock module (this runs inside its
    exec_module): log it and leave the stock functions in place. The candidate kernels issue no
    collectives, so a per-rank difference here cannot desynchronize the TP ranks."""
    from ds41h import candidates

    try:
        candidates.ext()
        return True
    except Exception:  # noqa: BLE001
        logging.getLogger("ds41h.candidates").exception("ds41h candidates: extension unavailable; keeping stock kernels")
        return False


def patch_candidate_blocks(module) -> None:
    from ds41h.patch_core import mark_active

    if not _ext_ok():
        return

    module.select_candidate_blocks = _wrap_select(module.select_candidate_blocks)
    module.apply_candidate_mask = _wrap_mask(module.apply_candidate_mask)
    mark_active("candidates", "candidate_blocks.{select_candidate_blocks,apply_candidate_mask}")


def patch_sparse_attn_indexer(module) -> None:
    from ds41h.patch_core import mark_active

    if not _ext_ok():
        return
    # Normally already our wrappers (the module imports them after candidate_blocks was patched);
    # wrapping is idempotent, so this also covers a candidate_blocks imported before the hook.
    module._select_candidate_blocks = _wrap_select(module._select_candidate_blocks)
    module._apply_candidate_mask = _wrap_mask(module._apply_candidate_mask)
    mark_active("candidates", "sparse_attn_indexer.{_select_candidate_blocks,_apply_candidate_mask}")


def patch_indexer_topk(module) -> None:
    from ds41h.patch_core import mark_active

    if not _ext_ok():
        return
    cls = module.SparseIndexerTopk
    base_forward = cls.forward

    def forward(self, logits, seq_lens, next_n, topk_indices, topk_tokens, max_seq_len):
        from ds41h import candidates

        if self._is_cuda and self._backend == "auto" and candidates.indexer_topk(logits, seq_lens, next_n, topk_indices, topk_tokens):
            return None
        return base_forward(self, logits, seq_lens, next_n, topk_indices, topk_tokens, max_seq_len)

    cls.forward = forward
    mark_active("candidates", "SparseIndexerTopk.forward")


if flags.CANDIDATES:
    register("vllm.model_executor.kernels.attention.dsa.candidate_blocks")(patch_candidate_blocks)
    register("vllm.model_executor.layers.sparse_attn_indexer")(patch_sparse_attn_indexer)
    if TOPK_HOOK:
        register("vllm.model_executor.layers.indexer_topk")(patch_indexer_topk)
