#!/usr/bin/env python3
"""ds41h-phys building block: the attention half of one consumer layer in the sequence-parallel layout, one GPU.

A rank owns a contiguous block of whole sequences (4 decode rows each) and runs, for its R rows, all 64 heads:

  a [R, 5120] bf16 (the attention-sublayer input, after attn_norm)
  -> [q_a | kv_proj] = a @ [wq_a; wkv]^T              one bf16 GEMM, exact dequantized weights, fp32 accumulation
  -> q_r = RMSNorm(q_a) * q_norm                       dense.cu norm_rope (the reference's rounding points)
  -> kv  = RoPE(RMSNorm(kv_proj) * kv_norm)            dense.cu norm_rope (+ RoPE of dims 448..511)
  -> q   = q_r @ wq_b^T [R, 64 x 512], RoPE(q) + window KV record insert into the rank's cache   (dense.cu rope_q_insert)
  -> o   = stock FlashMLA sparse decode, one row x 64 heads (window 128 + top-512, sink), as stock calls it
  -> inverse RoPE(o) -> z = per-group wo_a (8 x [1024, 4096], bmm) -> y = z @ wo_b^T [R, 5120]

API (all on one device, current stream, allocation-free after the first call for a given R, CUDA-graph capturable):
  load_attn_weights(layer, device, model_dir=None) -> W
  load_cap_state(step, layer, cap_dir=None) -> state           (files of a local capture directory, CPU)
  rank_rows(state, rank, nranks=4) -> LongTensor of rows       (contiguous whole sequences; 114 -> 29/29/28/28)
  build_rank_cache(state, rows, device) -> cache               (compact window + compressed pages for these rows only,
                                                                remapped indices, insert slots, FlashMLA schedule)
  attn_half(W, cache, a_rows, positions_rows, insert=True) -> y_rows bf16 [R, 5120]

Window KV insert: with insert=True (default, what a real layer does) the rows' own-position window records are rewritten
from this kv path before the attention reads them, so the 4 rows of a sequence see each other's KV computed here (the
captured records of those positions came from production's prologue: 89 % of the values equal). insert=False keeps the
captured records (production's), e.g. to isolate the attention numerics; the q / kv GEMMs still run.

Self-test (python attn_half.py [--rank 0] [--layer 21] [--step 0]): y vs the ds41h reference y on the rank's captured rows
(bf16 ulps; reference = VLLM_SM90 numerics, exact weights, fp32 math, stock cos/sin; attention over the same records,
with the reference's own stored KV at the inserted positions), plus the time of attn_half in a CUDA graph (W / MHz / mJ).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench2b as B  # noqa: E402
import cap_check as C  # noqa: E402

L, Q = B.L, B.Q
SWA_SHIFT = 5  # window cache block 32 (stock)
_EXT = {}


def _ext():
    if "kd" not in _EXT:
        _EXT["k3"], _EXT["kd"] = B.build()
    return _EXT["kd"]


# ATTN2B_QA8=S (default 0 = off): q_a | wkv as one W8A16 GEMM (fp8 weights, exact dequant) with S K-splits instead of the
# cuBLAS bf16 GEMM. Read at import.
QA8 = int(os.environ.get("ATTN2B_QA8", "0"))

# ATTN2B_SPA=1 (default 0): attn_sp.cu (one CTA per row x 64 heads, cp.async record gather, inverse RoPE fused)
# replaces stock FlashMLA (main + combine) and dense.cu's inv_rope; ATTN2B_SPA=2: attn_sp3.cu (v3: a QK warpgroup with
# Q in registers + 2 P V warpgroups; parked); ATTN2B_SPA=4: attn_sp4.cu (v2 + a producer warpgroup). Read at import.
SPA = int(os.environ.get("ATTN2B_SPA", "0"))
# ATTN2B_SPA_PF=MB (default 0): attn_sp also pulls the first MB of the next GEMM's weights (cuBLAS wo_a's bf16 wo_aT, or
# W8A16 wo_a's packed fp8) into L2 while it runs (the attention leaves HBM mostly idle).
SPA_PF = float(os.environ.get("ATTN2B_SPA_PF", "0"))


def _spa():
    if "spa" not in _EXT:
        import attn_sp
        import attn_sp3
        import attn_sp4

        _EXT["spa"] = {2: attn_sp3, 4: attn_sp4}.get(SPA, attn_sp).build()
    return _EXT["spa"]


def _attention(W: dict, cache: dict, ws: dict, positions_rows) -> None:
    """o (rotated back: inverse RoPE on dims 448..511 applied) of the rank's rows into ws["o"]."""
    if SPA == 2:  # v3: QK warpgroup + 2 P V warpgroups
        import attn_sp3

        attn_sp3.attn(_spa(), ws["q"], ws["o"], cache, W["sink"], positions_rows, W["cos_sin"])
        return
    if SPA == 4:  # v4: v2 + a producer warpgroup for the record gather
        import attn_sp4

        attn_sp4.attn(_spa(), ws["q"], ws["o"], cache, W["sink"], positions_rows, W["cos_sin"])
        return
    if SPA:
        import attn_sp

        nxt = None
        if SPA_PF > 0:
            w8 = W.get("w8") or {}
            nxt = w8["wo_a"].packed if "wo_a" in w8 else W.get("wo_aT")
        attn_sp.attn(_spa(), ws["q"], ws["o"], cache, W["sink"], positions_rows, W["cos_sin"], None, nxt,
                     int(SPA_PF * 1e6) if nxt is not None else 0)
        return
    from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache

    flash_mla_with_kvcache(q=ws["q"], k_cache=cache["k_swa"], block_table=None, head_dim_v=512,
                           tile_scheduler_metadata=cache["sched"], cache_seqlens=None, is_fp8_kvcache=True,
                           indices=cache["si"], topk_length=cache["sl"], softmax_scale=cache["scale"], attn_sink=W["sink"],
                           extra_k_cache=cache["k_comp"], extra_indices_in_kvcache=cache["ti"], extra_topk_length=cache["tl"],
                           out=ws["o"])
    _ext().inv_rope(ws["o"].view(cache["R"], 64, 512), positions_rows, W["cos_sin"])


# ------------------------------------------------------------------------------------------------ weights
def load_attn_weights(layer: int, device, model_dir: str | None = None, w8a16: bool | str = False,
                      splits: tuple[int, int, int] = (1, 4, 6)) -> dict:
    """Replicated full-head attention weights of one layer on one GPU (bf16, exactly dequantized from the checkpoint's
    e4m3 + 32x32 ue8m0 blocks), sink, scale and stock's RoPE table. w8a16=True (or "all"): q_b / wo_a / wo_b run in the
    W8A16 kernel (w8a16.cu: fp8 weights, exact bf16 dequant in registers, RoPE + KV insert fused into q_b; the inverse
    RoPE stays in dense.cu before wo_a; split-K counts for q_b, wo_a, wo_b). w8a16="q_b": hybrid, only q_b in W8A16
    (q_b + RoPE + KV insert fused: 25.9 us cold at 116 rows vs cuBLAS 28.7 + 3.7), wo_a / wo_b stay cuBLAS bf16 (the
    split-K W8A16 GEMMs are slower; cuBLAS wo_a writing z directly takes 22 us)."""
    dev = torch.device(device)
    rd = B.CheckpointReader(model_dir or B.MODEL)
    Wf = L.load_layer_weights(rd, layer, rank=None, device=dev, experts="none")
    bf = torch.bfloat16
    W = dict(layer=layer, device=dev, ratio=Wf.compress_ratio,
             wqkv_a=torch.cat([Wf.wq_a, Wf.wkv], 0).to(bf).contiguous(),        # [1280 + 512, 5120]
             q_norm=Wf.q_norm.float().contiguous(), kv_norm=Wf.kv_norm.float().contiguous(),
             wq_b=Wf.wq_b.to(bf).contiguous(),                                   # [32768, 1280]
             wo_aT=Wf.wo_a.to(bf).transpose(1, 2).contiguous(),                  # [8, 4096, 1024]
             wo_b=Wf.wo_b.to(bf).contiguous(),                                   # [5120, 8192]
             sink=Wf.attn_sink.float().contiguous(), scale=512 ** -0.5, eps=1e-20,
             cos_sin=B.stock_cos_sin(Wf.compress_ratio, dev, 0))
    exact = all(bool((W[k].float() == ref).all()) for k, ref in
                (("wq_b", Wf.wq_b), ("wo_b", Wf.wo_b), ("wqkv_a", torch.cat([Wf.wq_a, Wf.wkv], 0))))
    assert exact, "bf16 weights must be exact"
    W["_ref"] = Wf  # fp32 reference weights (self-test only; drop with W.pop('_ref') to save memory)
    if w8a16:
        import w8a16 as W8

        mode = "q_b" if w8a16 == "q_b" else "all"
        assert w8a16 in (True, "all", "q_b"), w8a16
        names = ("wq_b",) if mode == "q_b" else ("wq_b", "wo_a", "wo_b")
        with torch.cuda.device(dev):
            ext = W8.build()
            raw = {n: (rd.get(f"layers.{layer}.attn.{n}.weight").to(dev), rd.get(f"layers.{layer}.attn.{n}.scale").to(dev))
                   for n in names}
            W["w8"] = dict(wq_b=W8.Gemm(ext, *raw["wq_b"], splits[0]))
            if mode == "all":
                W["w8"].update(wo_a=W8.Gemm(ext, *raw["wo_a"], splits[1], rows_per_group=1024, kx_per_group=4096),
                               wo_b=W8.Gemm(ext, *raw["wo_b"], splits[2]))
        W["w8_mode"] = mode
        for k in (("wq_b",) if mode == "q_b" else ("wq_b", "wo_aT", "wo_b")):  # bf16 copies not needed on this path
            W.pop(k)
    if QA8:  # [wq_a; wkv] fp8 rows + their 32 x 32 scale blocks, concatenated as the bf16 copy is
        import w8a16 as W8

        with torch.cuda.device(dev):
            ext = W8.build(extra_defs=("W8_SMAX=16",) if QA8 > 8 else ())  # > 8 splits: the 16-partial reduce build
            w = torch.cat([rd.get(f"layers.{layer}.attn.{n}.weight") for n in ("wq_a", "wkv")], 0).to(dev).contiguous()
            sc = torch.cat([rd.get(f"layers.{layer}.attn.{n}.scale") for n in ("wq_a", "wkv")], 0).to(dev).contiguous()
            W.setdefault("w8", {})["wqkv_a"] = W8.Gemm(ext, w, sc, QA8)
        W.pop("wqkv_a")
    return W


# ------------------------------------------------------------------------------------------------ captured state, rows
def load_cap_state(step: int, layer: int, cap_dir: str | None = None) -> dict:
    if cap_dir:
        C.D = cap_dir.rstrip("/") + "/"
    r0 = C.load(step, layer, 0)
    return dict(step=step, layer=layer, T=r0["T"], r0=r0, rb=C.rebuild(r0))


def rank_rows(state: dict, rank: int, nranks: int = 4) -> torch.Tensor:
    t2r = state["r0"]["attn"]["token_to_req"].long()
    S = int(t2r.max()) + 1
    sizes = [S // nranks + (1 if r < S % nranks else 0) for r in range(nranks)]
    lo, hi = sum(sizes[:rank]), sum(sizes[:rank + 1])
    return ((t2r >= lo) & (t2r < hi)).nonzero().flatten()


def _dedup(idx, rec, ok):
    flat = idx.reshape(-1).long()
    v = flat >= 0
    s, r, o = flat[v], rec.reshape(-1, 584)[v], ok.reshape(-1)[v]
    us, inv = torch.unique(s, return_inverse=True)
    order = torch.argsort(inv * 2 + (~o).long())
    first = order[torch.searchsorted(inv[order], torch.arange(us.numel()))]
    full = torch.full_like(flat, -1)
    full[v] = inv
    return us, r[first].clone(), o[first].clone(), full.view(idx.shape)


def _write_pages(buf, stride, bs, page, pos, urec):
    dev = buf.device
    pb, pos, urec = page.long().to(dev) * stride, pos.long().to(dev), urec.to(dev)
    for i0 in range(0, page.numel(), 1 << 16):
        sl = slice(i0, i0 + (1 << 16))
        d = (pb[sl] + pos[sl] * 576).unsqueeze(1) + torch.arange(576, device=dev)
        sc = (pb[sl] + bs * 576 + pos[sl] * 8).unsqueeze(1) + torch.arange(8, device=dev)
        buf[d.reshape(-1)] = urec[sl, :576].reshape(-1)
        buf[sc.reshape(-1)] = urec[sl, 576:].reshape(-1)


def build_rank_cache(state: dict, rows, device) -> dict:
    """Compact window (block 32) and compressed (block 64) caches holding every record the given rows read, at the
    captured page stride, with the rows' indices remapped (order-preserving) and the own-position insert slots."""
    dev = torch.device(device)
    rows = torch.as_tensor(rows, dtype=torch.long).cpu()
    a, rb = state["r0"]["attn"], state["rb"]
    T = state["T"]
    assert bool((rows[1:] == rows[:-1] + 1).all()), "rows must be one contiguous block"
    t2r = a["token_to_req"].long()
    reqs = t2r[rows].unique()
    assert int((torch.isin(t2r, reqs)).sum()) == rows.numel(), "rows must be whole sequences"
    stride = int(a["swa_cache_stride"][0])
    sbs, cbs = rb["swa_bs"], rb["comp_bs"]
    assert sbs == 1 << SWA_SHIFT
    si = a["swa_indices"].view(T, -1)[rows]
    ti = a["topk_global"].view(T, -1)[rows]
    sl, tl = a["swa_lens"][rows].to(torch.int32), a["topk_lens"][rows].to(torch.int32)
    su, srec, sok, sinv = _dedup(si, rb["swa_rec"][rows], rb["swa_ok"][rows])
    cu, crec, cok, cinv = _dedup(ti, rb["comp_rec"][rows], rb["comp_ok"][rows])
    assert bool(sok.all()) and bool(cok.all()), "capture without full records (use cap2 or later)"
    sb_u, s_page = torch.unique(su // sbs, return_inverse=True)
    cb_u, c_page = torch.unique(cu // cbs, return_inverse=True)
    sbuf = torch.zeros(sb_u.numel() * stride, dtype=torch.uint8, device=dev)
    cbuf = torch.zeros(cb_u.numel() * stride, dtype=torch.uint8, device=dev)
    _write_pages(sbuf, stride, sbs, s_page, su % sbs, srec)
    _write_pages(cbuf, stride, cbs, c_page, cu % cbs, crec)
    s_new, c_new = s_page * sbs + su % sbs, c_page * cbs + cu % cbs
    nsi = torch.where(sinv >= 0, s_new[sinv.clamp(min=0)], torch.full_like(sinv, -1)).to(torch.int32)
    nti = torch.where(cinv >= 0, c_new[cinv.clamp(min=0)], torch.full_like(cinv, -1)).to(torch.int32)
    R = rows.numel()
    own = nsi[torch.arange(R), sl.long() - 1].long()  # the last window key of a row = its own position
    zero = torch.zeros(1, 584, dtype=torch.uint8)
    srow = torch.where((sinv >= 0).unsqueeze(-1), torch.cat([srec, zero])[sinv.clamp(min=0)], zero)
    crow = torch.where((cinv >= 0).unsqueeze(-1), torch.cat([crec, zero])[cinv.clamp(min=0)], zero)
    sval = (si >= 0) & (torch.arange(si.shape[1]) < sl.view(-1, 1))
    tval = (ti >= 0) & (torch.arange(ti.shape[1]) < tl.view(-1, 1))
    from vllm.third_party.flashmla.flash_mla_interface import get_mla_metadata

    return dict(
        rows=rows, R=R, device=dev, stride=stride,
        sbuf=sbuf, cbuf=cbuf,
        k_swa=torch.as_strided(sbuf, (sb_u.numel(), sbs, 1, 584), (stride, 584, 584, 1)),
        k_comp=torch.as_strided(cbuf, (cb_u.numel(), cbs, 1, 584), (stride, 584, 584, 1)),
        si=nsi.to(dev).view(R, 1, -1).contiguous(), sl=sl.to(dev), ti=nti.to(dev).view(R, 1, -1).contiguous(), tl=tl.to(dev),
        own_slot=own.to(dev), own_slot_cpu=own, swa_new=nsi, sinv=sinv,
        rec_rows=torch.cat([srow, crow], 1), valid=torch.cat([sval, tval], 1),
        sched=get_mla_metadata()[0], scale=float(a["scale"]), ws=None,
        ins=int(torch.bincount(t2r[rows] - int(t2r[rows].min())).max()),  # rows per sequence (window records inserted per row)
        pages=dict(swa=int(sb_u.numel()), comp=int(cb_u.numel())),
    )


# ------------------------------------------------------------------------------------------------ FlashMLA schedule
def plan(W: dict, cache: dict) -> dict:
    """Run FlashMLA's tile-scheduler planner for this cache's rows / lengths once (stock: once per step and layer type,
    FlashMLASchedMeta shared by the layers of that type). attn_half only ever reuses the stored plan; call this outside
    graph capture. The plan is a pure function of the lengths / widths (window lens, top-k lens, q rows, heads)."""
    from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache, get_mla_metadata

    R = cache["R"]
    with torch.cuda.device(W["device"]):
        cache["sched"] = get_mla_metadata()[0]
        q = torch.zeros(R, 1, 64, 512, dtype=torch.bfloat16, device=W["device"])
        o = torch.empty_like(q)
        flash_mla_with_kvcache(q=q, k_cache=cache["k_swa"], block_table=None, head_dim_v=512, tile_scheduler_metadata=cache["sched"],
                               cache_seqlens=None, is_fp8_kvcache=True, indices=cache["si"], topk_length=cache["sl"],
                               softmax_scale=cache["scale"], attn_sink=W["sink"], extra_k_cache=cache["k_comp"],
                               extra_indices_in_kvcache=cache["ti"], extra_topk_length=cache["tl"], out=o)
    cache["planned"] = True
    return cache


def share_plan(dst: dict, src: dict) -> dict:
    """Reuse src's FlashMLA plan for dst (another layer of the same type in the same step: same rows, same lengths)."""
    assert src.get("planned"), "plan(src) first"
    assert dst["R"] == src["R"] and torch.equal(dst["sl"], src["sl"]) and torch.equal(dst["tl"], src["tl"]), "lengths differ"
    assert dst["si"].shape == src["si"].shape and dst["ti"].shape == src["ti"].shape
    dst["sched"] = src["sched"]
    dst["planned"] = True
    return dst


# ------------------------------------------------------------------------------------------------ the attention half
W8_MAX_ROWS = 128  # rows per W8A16 CTA (token slice); more rows run as grid z slices in one launch
QB_SLICE = int(os.environ.get("ATTN2B_QB_SLICE", "0"))  # 1: the old per-128-row q_b calls (same bits, more launches)


def w8_qb(g, qr, q2, pos, cs, kv=None, slot=None, cache=None, page_stride=0, shift=SWA_SHIFT) -> None:
    """Fused W8A16 q_b + RoPE(q) (+ window KV insert) for ANY number of rows (decode buckets and prefill chunks).
    One launch: above 128 rows the kernel's grid z covers 128-row token slices. Rows are independent (no split-K for
    q_b), so every row's q (and inserted record) equals a decode-sized call's bit for bit. ATTN2B_QB_SLICE=1 issues
    one call per 128-row slice instead (the previous behaviour).
    g: the W8A16 Gemm (W["w8"]["wq_b"]); qr [R, 1280] / q2 [R, 32768] contiguous; pos [R]; kv [R, 512] and slot [R] or None."""
    R = qr.shape[0]
    if not QB_SLICE:
        g(qr, q2, pos=pos, cs=cs, kv=kv, slot=slot, cache=cache, page_stride=page_stride, shift=shift)
        return
    for s0 in range(0, R, W8_MAX_ROWS):
        s1 = min(R, s0 + W8_MAX_ROWS)
        g(qr[s0:s1], q2[s0:s1], pos=pos[s0:s1], cs=cs, kv=kv[s0:s1] if kv is not None else None,
          slot=slot[s0:s1] if slot is not None else None, cache=cache, page_stride=page_stride, shift=shift)


def _workspace(R: int, dev) -> dict:
    bf = torch.bfloat16
    return dict(R=R, qakv=torch.empty(R, 1792, dtype=bf, device=dev), qr=torch.empty(R, 1280, dtype=bf, device=dev),
                kv=torch.empty(R, 512, dtype=bf, device=dev), q=torch.empty(R, 1, 64, 512, dtype=bf, device=dev),
                o=torch.empty(R, 1, 64, 512, dtype=bf, device=dev), zb=torch.empty(8, R, 1024, dtype=bf, device=dev),
                z=torch.empty(R, 8192, dtype=bf, device=dev), y=torch.empty(R, 5120, dtype=bf, device=dev))


def attn_half(W: dict, cache: dict, a_rows: torch.Tensor, positions_rows: torch.Tensor, insert: bool = True) -> torch.Tensor:
    """y [R, 5120] bf16 for the rank's rows (a_rows [R, 5120] bf16, positions_rows [R] int64, both on W's device).
    Returns a view of a persistent buffer (overwritten by the next call with the same cache)."""
    from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache

    kd = _ext()
    dev = W["device"]
    R = cache["R"]
    assert a_rows.shape == (R, 5120) and a_rows.dtype == torch.bfloat16 and positions_rows.dtype == torch.int64
    if not cache.get("planned") and not SPA:
        plan(W, cache)  # first use outside a plan() call: plan now (never inside a captured graph)
    with torch.cuda.device(dev):
        ws = cache["ws"]
        if ws is None:
            ws = cache["ws"] = _workspace(R, dev)
        w8 = W.get("w8")
        if w8 is not None and "wqkv_a" in w8:
            w8["wqkv_a"](a_rows, ws["qakv"])
        else:
            torch.mm(a_rows, W["wqkv_a"].t(), out=ws["qakv"])
        kd.norm_rope2(ws["qakv"][:, :1280], W["q_norm"], ws["qr"], ws["qakv"][:, 1280:], W["kv_norm"], ws["kv"], W["eps"],
                      positions_rows, W["cos_sin"])  # both norms, one launch
        q3 = ws["q"].view(R, 64, 512)
        full = w8 is not None and "wo_a" in w8
        if w8 is not None and "wq_b" in w8:  # fused W8A16: q_b + RoPE(q) + window KV insert in one kernel (both W8A16 modes)
            w8_qb(w8["wq_b"], ws["qr"], ws["q"].view(R, 32768), positions_rows, W["cos_sin"],
                  kv=ws["kv"] if insert else None, slot=cache["own_slot"] if insert else None,
                  cache=cache["sbuf"] if insert else None, page_stride=cache["stride"], shift=SWA_SHIFT)
        else:
            torch.mm(ws["qr"], W["wq_b"].t(), out=ws["q"].view(R, 32768))
            kd.rope_q_insert(q3, ws["kv"], positions_rows, W["cos_sin"], cache["own_slot"], cache["sbuf"], cache["stride"], SWA_SHIFT,
                             bool(insert))
        _attention(W, cache, ws, positions_rows)  # FlashMLA + dense.cu inverse RoPE, or attn_sp (ATTN2B_SPA=1)
        o3 = ws["o"].view(R, 64, 512)
        if full:  # W8A16 wo_a, wo_b (o already rotated back)
            w8["wo_a"](ws["o"].view(R, 32768), ws["z"])
            w8["wo_b"](ws["z"], ws["y"])
        else:
            # cuBLAS writes z [R, 8 x 1024] directly (strided batch: ldc 8192, batch stride 1024; no copy kernel)
            torch.bmm(o3.view(R, 8, 4096).transpose(0, 1), W["wo_aT"], out=ws["z"].view(R, 8, 1024).transpose(0, 1))
            torch.mm(ws["z"], W["wo_b"].t(), out=ws["y"])
    return ws["y"]


def pieces(W: dict, cache: dict, a_rows, positions_rows, insert: bool = True) -> list:
    """attn_half split into its launches (same calls, same order, same buffers), for timing each in a graph."""
    from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache

    kd = _ext()
    attn_half(W, cache, a_rows, positions_rows, insert)  # workspace + plan
    ws, R, w8 = cache["ws"], cache["R"], W.get("w8")
    full = w8 is not None and "wo_a" in w8
    q3, o3 = ws["q"].view(R, 64, 512), ws["o"].view(R, 64, 512)
    qa = (lambda: w8["wqkv_a"](a_rows, ws["qakv"])) if (w8 is not None and "wqkv_a" in w8) else \
        (lambda: torch.mm(a_rows, W["wqkv_a"].t(), out=ws["qakv"]))
    out = [("q_a|wkv GEMM" + (" (W8A16)" if (w8 is not None and "wqkv_a" in w8) else ""), qa),
           ("q / kv norms (+ kv RoPE), one launch", lambda: kd.norm_rope2(ws["qakv"][:, :1280], W["q_norm"], ws["qr"],
                                                                         ws["qakv"][:, 1280:], W["kv_norm"], ws["kv"], W["eps"],
                                                                         positions_rows, W["cos_sin"]))]
    if w8 is not None and "wq_b" in w8:
        out.append(("q_b + RoPE(q) + KV insert (W8A16)", lambda: w8_qb(
            w8["wq_b"], ws["qr"], ws["q"].view(R, 32768), positions_rows, W["cos_sin"], kv=ws["kv"] if insert else None,
            slot=cache["own_slot"] if insert else None, cache=cache["sbuf"] if insert else None, page_stride=cache["stride"],
            shift=SWA_SHIFT)))
    else:
        out.append(("q_b (bf16 GEMM)", lambda: torch.mm(ws["qr"], W["wq_b"].t(), out=ws["q"].view(R, 32768))))
        out.append(("RoPE(q) + KV insert", lambda: kd.rope_q_insert(q3, ws["kv"], positions_rows, W["cos_sin"], cache["own_slot"],
                                                                     cache["sbuf"], cache["stride"], SWA_SHIFT, bool(insert))))
    if SPA:
        import attn_sp

        out.append(("attn_sp attention + inverse RoPE", lambda: _attention(W, cache, ws, positions_rows)))
    else:
        out.append(("FlashMLA attention", lambda: flash_mla_with_kvcache(
            q=ws["q"], k_cache=cache["k_swa"], block_table=None, head_dim_v=512, tile_scheduler_metadata=cache["sched"],
            cache_seqlens=None, is_fp8_kvcache=True, indices=cache["si"], topk_length=cache["sl"], softmax_scale=cache["scale"],
            attn_sink=W["sink"], extra_k_cache=cache["k_comp"], extra_indices_in_kvcache=cache["ti"], extra_topk_length=cache["tl"],
            out=ws["o"])))
        out.append(("inverse RoPE", lambda: kd.inv_rope(o3, positions_rows, W["cos_sin"])))
    if full:
        out.append(("wo_a (W8A16)", lambda: w8["wo_a"](ws["o"].view(R, 32768), ws["z"])))
        out.append(("wo_b (W8A16)", lambda: w8["wo_b"](ws["z"], ws["y"])))
    else:
        out.append(("wo_a (bmm into z, no copy)", lambda: torch.bmm(o3.view(R, 8, 4096).transpose(0, 1), W["wo_aT"],
                                                                   out=ws["z"].view(R, 8, 1024).transpose(0, 1))))
        out.append(("wo_b (bf16 GEMM)", lambda: torch.mm(ws["z"], W["wo_b"].t(), out=ws["y"])))
    return out


# ------------------------------------------------------------------------------------------------ reference + self-test
def reference(W: dict, cache: dict, a_rows: torch.Tensor, positions_rows: torch.Tensor, insert: bool = True) -> dict:
    """ds41h reference (VLLM_SM90, exact weights, fp32, stock cos/sin) of the same attention half over the same records;
    with insert, the rows' own-position window records are the reference's own stored KV."""
    Wf = W["_ref"]
    cs = W["cos_sin"]
    dev = W["device"]

    def rope_cs_stock(positions, compress_ratio, cfg=L.CFG):
        v = cs[positions.long()]
        return v[..., :32], v[..., 32:]
    L.rope_cos_sin = rope_cs_stock
    n = L.VLLM_SM90
    with L.strict_fp32():
        qkv = L.q_kv_proj_ref(Wf, a_rows.float(), positions_rows, n=n)
        recs = cache["rec_rows"].to(dev)
        kv_rows = Q.v4_unpack(recs[..., :576], recs[..., 576:]).float()
        if insert:  # window keys at this step's inserted slots -> the reference's stored kv of the writing row
            own = cache["own_slot_cpu"]
            win = cache["swa_new"].long()
            hit = (win.unsqueeze(-1) == own.view(1, 1, -1)) & (win.unsqueeze(-1) >= 0)  # [R, 128, R]
            r_i, k_i, src = hit.nonzero(as_tuple=True)
            kv_rows[r_i.to(dev), k_i.to(dev)] = qkv["kv"][src.to(dev)].float()
        o = Q.bf16_round(L.sparse_attn_ref(qkv["q"], kv_rows, Wf.attn_sink, cache["valid"].to(dev), cache["scale"]))
        op = L.o_proj_ref(Wf, o, positions_rows, n=n)
    bf = torch.bfloat16
    return dict(q=qkv["q"].to(bf), o=o.to(bf), o_rot=op["o_rot"].to(bf), z=op["z"].to(bf), y=op["y"].to(bf))


def breakdown(a, W, st, rows, cache, a_rows, pos_rows, insert, meter, dev):
    """Per-launch times of attn_half in CUDA graphs, cold: every graph runs the launch for 3 real layers in turn (their own
    weights and KV caches), as in the step program where other layers run in between; us per layer."""
    layers = [a.layer] + [int(x) for x in a.breakdown.split(",") if x.strip()]
    Ws, Cs, As = [W], [cache], [a_rows]
    for l in layers[1:]:
        Wl = load_attn_weights(l, dev, w8a16=a.w8a16 or False)
        Wl.pop("_ref", None)
        stl = load_cap_state(a.step, l, a.cap)
        Cs.append(build_rank_cache(stl, rows, dev))
        al = next(t for t in stl["r0"]["attn_args"] if isinstance(t, torch.Tensor) and t.dim() == 2)
        As.append(al[rows].to(dev).contiguous())
        Ws.append(Wl)
    plan(Ws[0], Cs[0])
    for c in Cs[1:]:
        share_plan(c, Cs[0])  # same step, same rows / lengths: one plan per step (and layer type)
    P = [pieces(Ws[i], Cs[i], As[i], pos_rows, insert) for i in range(len(layers))]
    n = len(layers)
    tag = {None: " [bf16 cuBLAS]", "all": " [W8A16]", "q_b": " [hybrid: W8A16 q_b, cuBLAS wo_a / wo_b]"}[a.w8a16]
    print(f"  breakdown (CUDA graphs, cold: {n} layers {layers} in turn; us per layer){tag}", flush=True)
    tot = 0.0
    for k in range(len(P[0])):
        name = P[0][k][0]
        fns = [P[i][k][1] for i in range(n)]
        us, g = meter.graph_time(lambda fns=fns: [f() for f in fns], a.reps, 5)
        pw = meter.power(g, 5, 0.8)
        tot += us / n
        print(f"    {name:38s} {us / n:7.1f} us" + (f"  {pw['W']:.0f} W  {pw['mhz']:.0f} MHz  {pw['J_per_call'] * 1e3 / n:.2f} mJ" if pw else ""),
              flush=True)
    us, g = meter.graph_time(lambda: [attn_half(Ws[i], Cs[i], As[i], pos_rows, insert) for i in range(n)], a.reps, 5)
    pw = meter.power(g, 5, 1.0)
    print(f"    {'sum of the pieces':38s} {tot:7.1f} us", flush=True)
    print(f"    {'attn_half, whole (cold)':38s} {us / n:7.1f} us" + (f"  {pw['W']:.0f} W  {pw['mhz']:.0f} MHz  {pw['J_per_call'] * 1e3 / n:.2f} mJ" if pw else ""),
          flush=True)
    # not inside attn_half: the stock global top-k mapping (per layer in stock) and FlashMLA's planner (once per step / type)
    try:
        from vllm.models.deepseek_v41.common.ops.cache_utils import compute_global_topk_indices_and_lens

        ar = st["r0"]["attn"]
        loc = ar["topk_local"][rows].to(dev).contiguous()
        t2r = ar["token_to_req"][rows].to(dev).contiguous()
        bt = ar["block_table"].to(dev)
        iv = ar["is_valid"][rows].to(dev).contiguous()
        us, _ = meter.graph_time(lambda: compute_global_topk_indices_and_lens(loc, t2r, bt, int(ar["comp_block_size"]), iv), a.reps, 10)
        print(f"    {'(global top-k mapping, stock op)':38s} {us:7.1f} us  (not in attn_half: the cache holds global indices)", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"    global top-k mapping: {type(ex).__name__}: {ex}", flush=True)
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(10):
        t0.record()
        plan(Ws[0], Cs[0])
        t1.record()
        torch.cuda.synchronize()
        ts.append(t0.elapsed_time(t1) * 1e3)
    ts.sort()
    print(f"    {'(FlashMLA planner + one call, eager)':38s} {ts[len(ts) // 2]:7.1f} us  (plan(): once per step and layer type, never in the graph)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--cap", default=os.environ.get("CAP_DIR", "cap"))
    ap.add_argument("--no-insert", action="store_true")
    ap.add_argument("--w8a16", nargs="?", const="all", default=None, choices=("all", "q_b"),
                    help="W8A16 dense GEMMs: all (fused q_b / wo_a / wo_b) or q_b (hybrid)")
    ap.add_argument("--breakdown", default="", help="extra layers for the cold rotation, e.g. 20,24 (with --layer 21)")
    ap.add_argument("--reps", type=int, default=40)
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    t0 = time.time()
    _ext()
    W = load_attn_weights(a.layer, dev, w8a16=a.w8a16 or False)
    st = load_cap_state(a.step, a.layer, a.cap)
    rows = rank_rows(st, a.rank)
    cache = build_rank_cache(st, rows, dev)
    r0 = st["r0"]
    a_all = next(t for t in r0["attn_args"] if isinstance(t, torch.Tensor) and t.dim() == 2)
    a_rows = a_all[rows].to(dev).contiguous()
    pos_rows = r0["in"]["positions"][rows].to(dev).contiguous()
    print(f"layer {a.layer} step {a.step} rank {a.rank}: rows {int(rows[0])}..{int(rows[-1])} (R={cache['R']}), pages {cache['pages']}, "
          f"setup {time.time() - t0:.0f} s", flush=True)
    insert = not a.no_insert
    y = attn_half(W, cache, a_rows, pos_rows, insert).clone()
    torch.cuda.synchronize()
    ref = reference(W, cache, a_rows, pos_rows, insert)
    ws = cache["ws"]
    o_name = "o_rot"  # every path rotates o in place (dense.cu inv_rope) before wo_a
    for nm, mine in (("q", ws["q"].view(cache["R"], 64, 512)), (o_name, ws["o"].view(cache["R"], 64, 512)), ("z", ws["z"]), ("y", y)):
        s = B.ulp_stats(mine.contiguous(), ref[nm].contiguous())
        print(f"  {nm} vs reference: exact {s['exact']:.4f} <=1 {s['le1']:.4f} max {s['max']} ulp, row-scaled max {s['row_ulp_max']:.2f} "
              f"p99.9 {s['row_ulp_p999']:.3f}", flush=True)
    # the captured production output of rank 0's 16 heads, NoPE dims 0..447 (untouched by the inverse RoPE)
    o_cap = r0["attn"]["output"][rows].to(dev)
    s = B.ulp_stats(ws["o"].view(cache["R"], 64, 512)[:, :16, :448].contiguous(), o_cap[:, :, :448].contiguous(), row=448)
    print(f"  o heads 0-15 (NoPE dims) vs captured production: exact {s['exact']:.4f} <=1 {s['le1']:.4f} row-scaled max "
          f"{s['row_ulp_max']:.2f}", flush=True)
    y2 = attn_half(W, cache, a_rows, pos_rows, insert)
    torch.cuda.synchronize()
    print(f"  second call bit-identical: {bool(torch.equal(y2, y))}", flush=True)
    meter = B.Meter(dev)
    if a.breakdown:
        breakdown(a, W, st, rows, cache, a_rows, pos_rows, insert, meter, dev)
    us, g = meter.graph_time(lambda: attn_half(W, cache, a_rows, pos_rows, insert), a.reps, 10)
    pw = meter.power(g, 10, 1.5)
    yg = cache["ws"]["y"].clone()
    print(f"  attn_half in a CUDA graph: {us:.1f} us" + (f", {pw['W']:.0f} W, {pw['mhz']:.0f} MHz, {pw['J_per_call'] * 1e3:.2f} mJ" if pw else "")
          + f"; graph output == eager: {bool(torch.equal(yg, y))}", flush=True)


if __name__ == "__main__":
    main()
