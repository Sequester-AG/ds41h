"""Attention half of one layer, split at the indexer hook.

The same launches, kernels, buffers and order as kernels/attn_dense/attn_half.attn_half (imported read-only: its extension,
workspace and W8A16 objects are reused), cut in two so the indexer / compressor (kernels/indexer2f) can run between the q / kv
norms (they need qr and the attention input) and FlashMLA (which reads their top-512 and the compressed records):

  attn_a(W, ws, a, pos)                 q_a | wkv GEMM -> norm_rope2 (qr, kv)                                   [hook here]
  attn_b(W, c, ws, pos, insert)         q_b + RoPE(q) + window insert -> FlashMLA (window + top-512) -> inverse RoPE
                                        -> wo_a -> wo_b  -> ws["y"]

attn_a + attn_b == attn_half bit for bit (same calls). Differences, both production's:
  * ratio-0 layers (0, 1) call FlashMLA WITHOUT the extra (compressed) cache, as production's flashmla.py does for SWA-only
    layers (extra_k_cache None); attn_half always passes one.
  * c["sched"] may be a fresh FlashMLASchedMeta inside a CUDA-graph capture: the first call of a layer type then captures
    FlashMLA's in-kernel planner, so every replay re-plans for the step's lengths (production: one sched per layer type
    per step, model.py / flashmla.py:212-228). attn_half plans outside the graph instead.
"""
from __future__ import annotations

import os
import sys

import torch

PHYS = os.environ.get("PHYS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(PHYS, "kernels/attn_dense"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("ATTN2B_PDL", "1")
import attn_half as AH  # noqa: E402

SWA_SHIFT = AH.SWA_SHIFT
assert not AH.SPA and not AH.QA8, "engine attn: ATTN2B_SPA / ATTN2B_QA8 paths not supported (attn_half defaults only)"


def workspace(R: int, dev) -> dict:
    return AH._workspace(R, dev)


def _a8_rows(W: dict, ws: dict, which: str):
    """ENG_PF_ACT8_DENSE (lossy opt-in): the bucket-A prefill rows [a, b) when the layer's 8-bit GEMM applies."""
    ab = ws.get("pf_ab")
    if ab is None or W.get("A8") is None or torch.cuda.is_current_stream_capturing():
        return None
    a, b = ab
    return ab if _a8().applies(W["L"], which, b - a) else None


def attn_a(W: dict, ws: dict, a_rows: torch.Tensor, pos: torch.Tensor) -> None:
    kd = AH._ext()
    ab = _a8_rows(W, ws, "qkv")
    if ab is not None:            # prefill rows: production's W8A8 q_a | wkv; the other rows 16-bit as below
        a, b = ab
        R = ws["R"]
        if a > 0:
            torch.mm(a_rows[:a], W["wqkv_a"].t(), out=ws["qakv"][:a])
        _a8().qkv(W["A8"], a_rows[a:b], ws["qakv"][a:b])
        if b < R:
            torch.mm(a_rows[b:], W["wqkv_a"].t(), out=ws["qakv"][b:])
    else:
        torch.mm(a_rows, W["wqkv_a"].t(), out=ws["qakv"])
    kd.norm_rope2(ws["qakv"][:, :1280], W["q_norm"], ws["qr"], ws["qakv"][:, 1280:], W["kv_norm"], ws["kv"], W["eps"],
                  pos, W["cos_sin"])


def attn_b(W: dict, c: dict, ws: dict, pos: torch.Tensor, insert: bool = True, pf=None) -> torch.Tensor:
    """attn_b_insert + attn_b_attend (the same calls; CP prefill runs its window-halo exchange between them)."""
    attn_b_insert(W, c, ws, pos, insert)
    return attn_b_attend(W, c, ws, pos, pf=pf)


QB16 = False     # engine.QB16_MIXED (set per mixed step): q_b of buckets with >= 256 rows on cuBLAS bf16 (bitwise)
QB16D = False    # engine.QB16_DECODE (set per decode step): decode buckets with > 128 rows on the bf16 q_b, captures included


def attn_b_insert(W: dict, c: dict, ws: dict, pos: torch.Tensor, insert: bool = True) -> None:
    """q_b + RoPE(q) + the rows' own window-KV insert."""
    kd = AH._ext()
    R = ws["R"]
    q3 = ws["q"].view(R, 64, 512)
    w8 = W.get("w8")
    full = w8 is not None and "wo_a" in w8
    capt = torch.cuda.is_current_stream_capturing()
    if W.get("wq_b16") is not None and ((QB16 and R >= 256 and not capt) or (QB16D and R > 128 and ws.get("pf_ab") is None)):
        ab = _a8_rows(W, ws, "q_b")
        qf = ws["q"].view(R, 32768)
        if ab is not None:        # prefill rows' q_b on production's W8A8 (the rest bf16); RoPE + insert over all rows
            a, b = ab
            if a > 0:
                torch.mm(ws["qr"][:a], W["wq_b16"].t(), out=qf[:a])
            _a8().q_b(W["A8"], ws["qr"][a:b], qf[a:b])
            if b < R:
                torch.mm(ws["qr"][b:], W["wq_b16"].t(), out=qf[b:])
        else:
            torch.mm(ws["qr"], W["wq_b16"].t(), out=qf)
        kd.rope_q_insert(q3, ws["kv"], pos, W["cos_sin"], c["own_slot"], c["sbuf"], c["stride"], SWA_SHIFT, bool(insert))
    elif w8 is not None and "wq_b" in w8:   # attn_half.w8_qb: the kernel takes <= 128 rows per call (rows independent)
        qb = getattr(AH, "w8_qb", None)
        if qb is not None:
            qb(w8["wq_b"], ws["qr"], ws["q"].view(R, 32768), pos, W["cos_sin"], kv=ws["kv"] if insert else None,
               slot=c["own_slot"] if insert else None, cache=c["sbuf"] if insert else None, page_stride=c["stride"],
               shift=SWA_SHIFT)
        else:
            assert R <= 128, "w8a16 q_b: <= 128 rows per call (attn_half without w8_qb)"
            w8["wq_b"](ws["qr"], ws["q"].view(R, 32768), pos=pos, cs=W["cos_sin"], kv=ws["kv"] if insert else None,
                       slot=c["own_slot"] if insert else None, cache=c["sbuf"] if insert else None,
                       page_stride=c["stride"], shift=SWA_SHIFT)
    else:
        torch.mm(ws["qr"], W["wq_b"].t(), out=ws["q"].view(R, 32768))
        kd.rope_q_insert(q3, ws["kv"], pos, W["cos_sin"], c["own_slot"], c["sbuf"], c["stride"], SWA_SHIFT, bool(insert))


def attn_b_attend(W: dict, c: dict, ws: dict, pos: torch.Tensor, pf=None) -> torch.Tensor:
    """FlashMLA (window + top-512) -> inverse RoPE -> wo_a -> wo_b -> ws["y"].
    pf (ENG_ATTN_PF: kernels/attn_dense/attn_prefill.Plan of the step's prefill rows, bucket A only): rows [:row_lo]
    (decode) on the decode kernel with their own plan (c["sched_dec"]), rows [row_lo, row_hi) through production's prefill
    attention (attn_prefill.attend: gather + combine + flash_mla_sparse_fwd), padding rows zeroed; then the same inverse
    RoPE / wo over all rows."""
    from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache

    kd = AH._ext()
    R = ws["R"]
    w8 = W.get("w8")
    full = w8 is not None and "wo_a" in w8
    comp = c.get("k_comp") is not None
    if pf is not None:
        a, b = pf.row_lo, pf.row_hi
        if a > 0:
            flash_mla_with_kvcache(q=ws["q"][:a], k_cache=c["k_swa"], block_table=None, head_dim_v=512,
                                   tile_scheduler_metadata=c["sched_dec"], cache_seqlens=None, is_fp8_kvcache=True,
                                   indices=c["si"][:a], topk_length=c["sl"][:a], softmax_scale=c["scale"],
                                   attn_sink=W["sink"], extra_k_cache=c["k_comp"] if comp else None,
                                   extra_indices_in_kvcache=c["ti"][:a] if comp else None,
                                   extra_topk_length=c["tl"][:a] if comp else None, out=ws["o"][:a])
        AP = _ap()
        AP.attend(pf, c["ratio"], ws["q"][a:b].view(b - a, 64, 512), ws["o"][a:b].view(b - a, 64, 512), c["k_swa"],
                  c["k_comp"] if comp else None, c["ti"][a:b] if comp else None, c["scale"], W["sink"],
                  src=c.get("src"), isrc=c.get("isrc"))
        if b < R:
            ws["o"][b:].zero_()
    else:
        flash_mla_with_kvcache(q=ws["q"], k_cache=c["k_swa"], block_table=None, head_dim_v=512,
                               tile_scheduler_metadata=c["sched"], cache_seqlens=None, is_fp8_kvcache=True,
                               indices=c["si"], topk_length=c["sl"], softmax_scale=c["scale"], attn_sink=W["sink"],
                               extra_k_cache=c["k_comp"] if comp else None,
                               extra_indices_in_kvcache=c["ti"] if comp else None,
                               extra_topk_length=c["tl"] if comp else None, out=ws["o"])
    kd.inv_rope(ws["o"].view(R, 64, 512), pos, W["cos_sin"])
    o3 = ws["o"].view(R, 64, 512)
    if full:
        w8["wo_a"](ws["o"].view(R, 32768), ws["z"])
        w8["wo_b"](ws["z"], ws["y"])
    else:
        torch.bmm(o3.view(R, 8, 4096).transpose(0, 1), W["wo_aT"], out=ws["z"].view(R, 8, 1024).transpose(0, 1))
        ab = _a8_rows(W, ws, "wo_b")
        if ab is not None:        # prefill rows' wo_b on production's W8A8 (wo_a stays 16-bit for all rows)
            a, b = ab
            if a > 0:
                torch.mm(ws["z"][:a], W["wo_b"].t(), out=ws["y"][:a])
            _a8().wo_b(W["A8"], ws["z"][a:b], ws["y"][a:b])
            if b < R:
                torch.mm(ws["z"][b:], W["wo_b"].t(), out=ws["y"][b:])
        else:
            torch.mm(ws["z"], W["wo_b"].t(), out=ws["y"])
    return ws["y"]


def _ap():
    """kernels/attn_dense/attn_prefill (production's prefill attention for the engine's prefill rows)."""
    import attn_prefill
    return attn_prefill


def _a8():
    """kernels/attn_dense/attn_act8 (production's W8A8 prefill dense path, ENG_PF_ACT8_DENSE)."""
    import attn_act8
    return attn_act8
