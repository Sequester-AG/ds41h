"""Decode-step glue outside the CUDA graphs: fewer eager launches in vLLM V2's input prep.

DS41H_PREP_FUSE (default 0 until gated end-to-end; flags.PREP_FUSE)
-------------------------------------------------------------
Every decode cycle vLLM's V2 runner builds the attention metadata of the next step EAGERLY (58 launches, 162 us span
but only 84 us busy at 350 W: the GPU waits ~1.4 us between back-to-back eager launches; the host is 7-15 ms ahead).
DeepseekSparseSWAMetadataBuilder.build (vllm/v1/attention/backends/mla/sparse_swa.py) runs, per SWA KV-cache group
(8 groups for DeepSeek-V4.1-Flash), three launches for a decode batch:
    torch.ge(slot_mapping, 0, out=is_valid_token)          # vectorized_elementwise<8>
    self.decode_swa_lens[num_decode_tokens:] = 0            # fill of ~16.9K int32
    _COMPUTE_SWA_INDICES_AND_LENS_KERNEL(..., token_offset=0)   # Triton
This patch fuses the three into ONE Triton launch per group (same integer results: is_valid over all slot_mapping
rows, the zero tail of decode_swa_lens, the decode window indices / lens by the stock kernel's exact arithmetic),
-16 launches per cycle (~40 us at 350 W).

How: the stock build() source is checked against a known hash (vLLM 0.32.0), then three statements are rewritten
with an AST transform (everything else in build() stays the stock code): the ge becomes `_ds41h_prep_ge(...)` (which
still runs torch.ge when the batch is not fusable: no decode tokens, non-causal DSpark draft groups), the tail fill
is skipped when fused, and the causal decode kernel call becomes `_ds41h_prep_swa(...)` when fused. Anything
unexpected (hash, statement counts, triton import) -> the patch stays off and the stock build runs.
"""

from __future__ import annotations

import logging

from ds41h import flags
from ds41h.patches import register

logger = logging.getLogger("ds41h")

# sha1 of inspect.getsource(DeepseekSparseSWAMetadataBuilder.build) in vLLM 0.32.0.
_BUILD_SHA1 = "30f3f56c47668fb637b35aeeb9b86d929c06f12b"
_FILL_BLOCK = 1024
_KERNEL = None
STATS = {"fused": 0, "stock": 0}


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["n_slot", "n_dec", "n_lens", "swa_indices_stride", "block_table_stride"])
    def _ds41h_swa_decode_prep(
        slot_ptr, n_slot, is_valid_ptr, lens_ptr, n_dec, n_lens,
        swa_indices_ptr, swa_indices_stride, window_size, index_width,
        query_start_loc_ptr, seq_lens_ptr, token_to_req_indices_ptr,
        block_table_ptr, block_table_stride, block_size, replay_start_ptr,
        FILL_BLOCK: tl.constexpr, TRITON_BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid < n_slot:
            token_idx = pid
            # torch.ge(slot_mapping, 0, out=is_valid_token)
            is_valid = tl.load(slot_ptr + token_idx) >= 0
            tl.store(is_valid_ptr + token_idx, is_valid)
            if token_idx < n_dec:
                # == _compute_swa_indices_and_lens_kernel (HAS_IMAGE=False, token_offset=0), row pid
                if not is_valid:
                    tl.store(lens_ptr + pid, 0)
                    for i in range(0, index_width, TRITON_BLOCK_SIZE):
                        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
                        tl.store(swa_indices_ptr + pid * swa_indices_stride + offset, -1, mask=offset < index_width)
                else:
                    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
                    query_start = tl.load(query_start_loc_ptr + req_idx)
                    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
                    query_len = query_end - query_start
                    seq_len = tl.load(seq_lens_ptr + req_idx)
                    prefix_len = seq_len - query_len
                    pos = prefix_len + token_idx - query_start
                    left = 0
                    right = 0
                    left_add = tl.maximum(left - (window_size - 1), 0)
                    start_pos = tl.maximum(pos - (window_size - 1) - left_add, 0)
                    start_pos = tl.maximum(start_pos, tl.load(replay_start_ptr + req_idx))
                    end_pos = pos + right + 1
                    swa_len = end_pos - start_pos
                    tl.store(lens_ptr + pid, swa_len)
                    for i in range(0, index_width, TRITON_BLOCK_SIZE):
                        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
                        pos_offset = start_pos + offset
                        block_indices = pos_offset // block_size
                        block_numbers = tl.load(block_table_ptr + req_idx * block_table_stride + block_indices,
                                                mask=pos_offset < end_pos)
                        block_offsets = pos_offset % block_size
                        slot_ids = block_numbers * block_size + block_offsets
                        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
                        tl.store(swa_indices_ptr + pid * swa_indices_stride + offset, slot_ids,
                                 mask=offset < index_width)
        else:
            # self.decode_swa_lens[num_decode_tokens:] = 0
            offs = n_dec + (pid - n_slot) * FILL_BLOCK + tl.arange(0, FILL_BLOCK)
            tl.store(lens_ptr + offs, 0, mask=offs < n_lens)

    _KERNEL = _ds41h_swa_decode_prep
    return _KERNEL


def _prep_ge(builder, cam, slot_mapping, is_valid_token, num_decode_tokens) -> bool:
    """Replaces `torch.ge(slot_mapping, 0, out=is_valid_token)`; True = deferred into the fused decode kernel."""
    import torch

    fused = (num_decode_tokens > 0 and cam.causal is True and slot_mapping.is_contiguous()
             and builder.decode_swa_lens.is_contiguous() and builder.decode_swa_indices.stride(-1) == 1)
    if not fused:
        torch.ge(slot_mapping, 0, out=is_valid_token)
        STATS["stock"] += 1
    return fused


def _prep_swa(builder, slot_mapping, is_valid_token, decode_swa_indices, num_decode_tokens, query_start_loc,
              seq_lens, token_to_req_indices, block_table, replay_start) -> None:
    """Replaces the tail fill + the causal decode _COMPUTE_SWA_INDICES_AND_LENS_KERNEL call (+ the deferred ge)."""
    lens = builder.decode_swa_lens
    n_slot = slot_mapping.shape[0]
    n_lens = lens.shape[0]
    n_fill = max(0, -(-(n_lens - num_decode_tokens) // _FILL_BLOCK))
    _get_kernel()[(n_slot + n_fill,)](
        slot_mapping, n_slot, is_valid_token, lens, num_decode_tokens, n_lens,
        decode_swa_indices, decode_swa_indices.stride(0), builder.window_size, decode_swa_indices.shape[-1],
        query_start_loc, seq_lens, token_to_req_indices,
        block_table, block_table.stride(0), builder.block_size, replay_start,
        FILL_BLOCK=_FILL_BLOCK, TRITON_BLOCK_SIZE=1024,
    )
    STATS["fused"] += 1


def _rewrite_build(mod):
    import ast
    import hashlib
    import inspect
    import textwrap

    cls = mod.DeepseekSparseSWAMetadataBuilder
    src = inspect.getsource(cls.build)
    sha = hashlib.sha1(src.encode()).hexdigest()
    if sha != _BUILD_SHA1:
        return None, f"stock build() changed (sha1 {sha[:12]})"
    tree = ast.parse(textwrap.dedent(src))
    counts = {"ge": 0, "fill": 0, "swa": 0}

    class _T(ast.NodeTransformer):
        def visit_Expr(self, node):
            text = ast.unparse(node)
            if text == "torch.ge(slot_mapping, 0, out=is_valid_token)":
                counts["ge"] += 1
                return ast.parse("_ds41h_fused = _ds41h_prep_ge(self, common_attn_metadata, slot_mapping, "
                                 "is_valid_token, num_decode_tokens)").body[0]
            call = node.value
            if (isinstance(call, ast.Call) and ast.unparse(call.func) == "_COMPUTE_SWA_INDICES_AND_LENS_KERNEL"
                    and ast.unparse(call.args[0]) == "decode_swa_indices"
                    and any(k.arg == "token_offset" and ast.unparse(k.value) == "0" for k in call.keywords)):
                counts["swa"] += 1
                new = ast.parse("if _ds41h_fused:\n    _ds41h_prep_swa(self, slot_mapping, is_valid_token, "
                                "decode_swa_indices, num_decode_tokens, query_start_loc, seq_lens, "
                                "token_to_req_indices, block_table, replay_start)\nelse:\n    pass").body[0]
                new.orelse = [node]
                return new
            return node

        def visit_Assign(self, node):
            if ast.unparse(node) == "self.decode_swa_lens[num_decode_tokens:] = 0":
                counts["fill"] += 1
                new = ast.parse("if not _ds41h_fused:\n    pass").body[0]
                new.body = [node]
                return new
            return node

    tree = ast.fix_missing_locations(_T().visit(tree))
    if counts != {"ge": 1, "fill": 1, "swa": 1}:
        return None, f"unexpected statement counts {counts}"
    mod._ds41h_prep_ge = _prep_ge
    mod._ds41h_prep_swa = _prep_swa
    ns: dict = {}
    exec(compile(tree, "<ds41h fused DeepseekSparseSWAMetadataBuilder.build>", "exec"), mod.__dict__, ns)
    build = ns["build"]
    build.__qualname__ = cls.build.__qualname__
    build._ds41h_fused = True
    return build, None


def _patch_sparse_swa(mod) -> None:
    cls = getattr(mod, "DeepseekSparseSWAMetadataBuilder", None)
    if cls is None or getattr(cls.build, "_ds41h_fused", False):
        return
    try:
        import triton  # noqa: F401

        build, why = _rewrite_build(mod)
    except Exception as e:  # noqa: BLE001
        build, why = None, f"{type(e).__name__}: {e}"
    if build is None:
        logger.info("ds41h prep fuse: OFF (%s)", why)
        return
    cls.build = build
    logger.info("ds41h prep fuse: ON (SWA metadata decode: ge + tail fill + indices in one launch per group)")
    try:
        from ds41h.patch_core import mark_active

        mark_active("glue", "DeepseekSparseSWAMetadataBuilder.build(fused decode prep)")
    except Exception:  # noqa: BLE001
        pass


if flags.PREP_FUSE:
    register("vllm.v1.attention.backends.mla.sparse_swa")(_patch_sparse_swa)
