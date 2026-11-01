"""GPU tests and micro-benchmark for ds41h 'attn' (sm90a 16-head sparse MLA decode).

    python -m pytest tests/test_attn.py -q          # correctness (needs an sm90 GPU)
    python tests/test_attn.py                       # same tests via the built-in runner (no pytest needed)
    python tests/test_attn.py --bench               # timing for M = 1, 6, 8 rows, 128 SWA + 512 top-k keys

Caches are written in the stock V4 fp8_ds_mla record (584 B/token) with vLLM's own writer
(vllm/models/deepseek_v41/common/ops/cache_utils.py:quantize_and_insert_k_cache) when vLLM is importable,
else with an exact torch replica of it. References:
  * float64 torch attention over the dequantized records (local, always);
  * ds41h.ref sparse attention reference (if present with a compatible signature);
  * stock FlashMLA sparse decode (flash_mla_with_kvcache, exactly as flashmla.py:230-246 calls it);
  * stock DeepseekV4FlashMLAAttention._forward_decode vs ds41h.attn.forward_decode on a stand-in layer.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import sys
import types
import zlib
from pathlib import Path

import torch

try:
    import pytest
except ImportError:  # the serving venv has no pytest: minimal stand-in + built-in runner (see _run_all)
    class _Skipped(Exception):
        pass

    class _Mark:
        @staticmethod
        def parametrize(name, values):
            def deco(fn):
                fn.__dict__.setdefault("_params", []).insert(0, (name, list(values)))
                return fn

            return deco

        @staticmethod
        def skipif(cond, reason=""):
            def deco(fn):
                if cond:
                    fn._skip_reason = reason
                return fn

            return deco

    class _PytestShim:
        mark = _Mark()

        @staticmethod
        def skip(reason=""):
            raise _Skipped(reason)

    pytest = _PytestShim()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HEADS, DIM, NOPE = 16, 512, 448
REC, DATA = 584, 576
SCALE = DIM**-0.5
PAD_HEADS = 64  # stock FlashMLA sm90 decode head padding

_HAS_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)  # kernel is sm_90a-only
pytestmark = pytest.mark.skipif(not _HAS_SM90, reason="needs an sm90 GPU")


def _attn():
    from ds41h import attn

    return attn


# ------------------------------------------------------------------------------------------ caches
def make_cache(num_blocks: int, bs: int, device, extra_page_bytes: int = 0):
    """Raw [num_blocks, page_stride] buffer and its [num_blocks, bs, 584] view. The page stride is rounded to
    the 576-B packing alignment like vLLM (attention.py:522) plus optional extra bytes (pages of other
    layers interleaved in a hybrid block). Unwritten bytes are random garbage (incl. NaN patterns)."""
    stride = (bs * REC + 575) // 576 * 576 + extra_page_bytes
    raw = torch.randint(0, 256, (num_blocks, stride), dtype=torch.uint8, device=device)
    view = raw.as_strided((num_blocks, bs, REC), (stride, REC, 1))
    return raw, view


def _write_k_torch(raw: torch.Tensor, bs: int, k: torch.Tensor, slots: torch.Tensor) -> None:
    """Exact torch replica of quantize_and_insert_k_kernel (cache_utils.py:57-179)."""
    dev = raw.device
    stride = raw.shape[1]
    x = k.float()
    nope = x[:, :NOPE].reshape(-1, 7, 64)
    amax = nope.abs().amax(-1).clamp_min(1e-4)
    e = torch.ceil(torch.log2(amax / 448.0))
    q8 = (nope / torch.exp2(e)[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    q8 = q8.view(torch.uint8).reshape(-1, NOPE)
    sc = (e + 127.0).clamp(0, 255).to(torch.uint8)
    rope = k[:, NOPE:].contiguous().view(torch.uint8)  # [N, 128]
    slots = slots.long()
    base = (slots // bs) * stride
    pos = slots % bs
    flat = raw.view(-1)
    d0 = base + pos * DATA
    flat[d0[:, None] + torch.arange(NOPE, device=dev)] = q8
    flat[d0[:, None] + NOPE + torch.arange(128, device=dev)] = rope
    s0 = base + bs * DATA + pos * 8
    flat[s0[:, None] + torch.arange(7, device=dev)] = sc
    flat[s0 + 7] = 0


def write_k(raw: torch.Tensor, view: torch.Tensor, k: torch.Tensor, slots: torch.Tensor, use_stock: bool = True) -> str:
    """Quantize bf16 rows k [N, 512] into the cache at slots; returns which writer was used."""
    bs = view.shape[1]
    if slots.numel() == 0:
        return "none"
    if use_stock:
        try:
            from vllm.models.deepseek_v41.common.ops.cache_utils import quantize_and_insert_k_cache

            quantize_and_insert_k_cache(k.contiguous(), view, slots.long().contiguous(), block_size=bs)
            return "stock"
        except ImportError:
            pass
    _write_k_torch(raw, bs, k, slots)
    return "torch"


def read_k(raw: torch.Tensor, bs: int, slots: torch.Tensor) -> torch.Tensor:
    """Dequantize records at slots -> float64 [N, 512] (NoPE fp8 * 2^(e-127), RoPE bf16)."""
    dev = raw.device
    stride = raw.shape[1]
    slots = slots.long()
    base = (slots // bs) * stride
    pos = slots % bs
    flat = raw.view(-1)
    d0 = base + pos * DATA
    q8 = flat[d0[:, None] + torch.arange(NOPE, device=dev)].contiguous().view(torch.float8_e4m3fn).double()
    sc = flat[(base + bs * DATA + pos * 8)[:, None] + torch.arange(7, device=dev)].double()
    nope = q8.reshape(-1, 7, 64) * torch.exp2(sc - 127.0)[..., None]
    rope = flat[d0[:, None] + NOPE + torch.arange(128, device=dev)].contiguous().view(torch.bfloat16).double()
    return torch.cat([nope.reshape(-1, NOPE), rope], dim=1)


def random_k(n: int, device, gen: torch.Generator) -> torch.Tensor:
    """bf16 latent rows, per-64-block magnitudes 2^-3..2^0 (distinct UE8M0 scales; q.k * 512^-0.5 then has
    std < 1, so softmax mixes many keys), plus rare 2^3 outlier blocks."""
    k = torch.randn(n, DIM, generator=gen, device=device)
    e = torch.randint(-3, 1, (n, 8), generator=gen, device=device)
    e = torch.where(torch.rand(n, 8, generator=gen, device=device) < 0.002, 3, e)
    blk = torch.exp2(e.float())
    k = (k.view(n, 8, 64) * blk[..., None]).view(n, DIM)
    return k.to(torch.bfloat16)


def poison(raw: torch.Tensor, bs: int, slots: torch.Tensor) -> None:
    """Fill records with NaN e4m3 bytes / NaN bf16 so any read of them shows up in the output."""
    dev = raw.device
    stride = raw.shape[1]
    slots = slots.long()
    base = (slots // bs) * stride
    pos = slots % bs
    flat = raw.view(-1)
    d0 = base + pos * DATA
    flat[d0[:, None] + torch.arange(DATA, device=dev)] = 0x7F
    flat[(base + bs * DATA + pos * 8)[:, None] + torch.arange(8, device=dev)] = 127


# ------------------------------------------------------------------------------------------ metadata
def swa_indices_stock(seq_lens, query_lens, block_table, bs, width, window=128, noncausal=False, replay_start=None):
    """Replica of _compute_swa_indices_and_lens_kernel (sparse_swa.py:1042-1122) or, if noncausal, of the
    DSpark draft kernel (sparse_swa.py:1242-1311). Returns ([T, 1, width] int32, [T] int32, req [T])."""
    dev = block_table.device
    rows, lens, reqs = [], [], []
    for r, (sl, ql) in enumerate(zip(seq_lens, query_lens)):
        prefix = sl - ql
        for t in range(ql):
            pos = prefix + t
            if noncausal:
                start, end = max(prefix - window, 0), sl
            else:
                start = max(pos - (window - 1), 0)
                if replay_start is not None:
                    start = max(start, replay_start[r])
                end = pos + 1
            n = end - start
            p = torch.arange(start, start + width, device=dev)
            slot = block_table[r, (p // bs).clamp(max=block_table.shape[1] - 1)] * bs + p % bs
            rows.append(torch.where(torch.arange(width, device=dev) < n, slot, -1))
            lens.append(n)
            reqs.append(r)
    idx = torch.stack(rows).to(torch.int32).view(-1, 1, width)
    return idx, torch.tensor(lens, dtype=torch.int32, device=dev), torch.tensor(reqs, dtype=torch.int32, device=dev)


def local_to_global(local, req, block_table, bs, is_valid):
    """Replica of compute_global_topk_indices_and_lens (cache_utils.py:832-914)."""
    valid = local >= 0
    blk = block_table[req.long()[:, None], (local.clamp(min=0) // bs).long()]
    glob = torch.where(valid, blk * bs + local % bs, -1).to(torch.int32)
    lens = torch.where(is_valid, valid.sum(-1), 0).to(torch.int32)
    return glob, lens


def make_case(
    seq_lens,
    query_lens,
    *,
    topk=512,
    ratio=1,
    swa_bs=64,
    block_size=64,
    window=128,
    noncausal=False,
    holes=False,
    invalid_rows=(),
    extra_page_bytes=0,
    q_gain=1.0,
    sink="normal",
    seed=0,
    device="cuda",
    use_stock_writer=True,
):
    """A decode batch shaped like stock: requests with query_lens rows each (DSpark verify rows are causal),
    paged SWA + compressed caches with random block tables, indexer-style local top-k (sorted, -1 tail)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    nreq = len(seq_lens)
    swa_blocks_per_req = max(math.ceil(s / swa_bs) for s in seq_lens) + 1
    swa_nb = nreq * swa_blocks_per_req + 3
    swa_raw, swa = make_cache(swa_nb, swa_bs, device, extra_page_bytes)
    perm = torch.randperm(swa_nb, generator=gen, device=device)[: nreq * swa_blocks_per_req]
    swa_bt = perm.view(nreq, swa_blocks_per_req).to(torch.int32)
    width = window
    if noncausal:
        from math import ceil

        width = ceil((window + max(query_lens)) / 64) * 64  # get_dspark_swa_index_width
    swa_idx, swa_lens, req = swa_indices_stock(seq_lens, query_lens, swa_bt, swa_bs, width, window, noncausal)
    T = swa_idx.shape[0]
    is_valid = torch.ones(T, dtype=torch.bool, device=device)
    for r in invalid_rows:  # padded tokens: builder writes len 0 and -1 rows (sparse_swa.py:1065-1075)
        is_valid[r] = False
        swa_lens[r] = 0
        swa_idx[r] = -1
    used = swa_idx[swa_idx >= 0].unique()
    writers = {write_k(swa_raw, swa, random_k(used.numel(), device, gen), used, use_stock_writer)}

    case = dict(swa_raw=swa_raw, swa=swa, swa_bt=swa_bt, swa_idx=swa_idx, swa_lens=swa_lens, req=req, is_valid=is_valid, T=T, writers=writers)
    if topk:
        ebs = block_size // ratio
        ncomp = [s // ratio for s in seq_lens]
        ex_blocks_per_req = max(math.ceil(max(n, 1) / ebs) for n in ncomp) + 1
        ex_nb = nreq * ex_blocks_per_req + 2
        ex_raw, ex = make_cache(ex_nb, ebs, device, extra_page_bytes)
        ex_bt = torch.randperm(ex_nb, generator=gen, device=device)[: nreq * ex_blocks_per_req].view(nreq, -1).to(torch.int32)
        local = torch.full((T, topk), -1, dtype=torch.int32, device=device)
        row = 0
        for r, (sl, ql) in enumerate(zip(seq_lens, query_lens)):
            for t in range(ql):
                pos = sl - ql + t
                n = (pos + 1) // ratio  # visible compressed positions
                take = min(topk, n)
                if take > 0:
                    sel = torch.randperm(n, generator=gen, device=device)[:take].sort().values
                    local[row, :take] = sel.to(torch.int32)
                    if holes and take > 4:  # -1 inside the valid prefix: stock length = count, not last index
                        h = torch.randperm(take, generator=gen, device=device)[: max(1, take // 16)]
                        local[row, h] = -1
                row += 1
        glob, ex_lens = local_to_global(local, req, ex_bt, ebs, is_valid)
        used = glob[(glob >= 0) & (torch.arange(topk, device=device)[None] < ex_lens[:, None])].unique()
        writers.add(write_k(ex_raw, ex, random_k(used.numel(), device, gen), used, use_stock_writer))
        case.update(ex_raw=ex_raw, ex=ex, ex_bt=ex_bt, ex_local=local, ex_glob=glob, ex_lens=ex_lens, ebs=ebs, ratio=ratio, block_size=block_size)
    q = torch.zeros(T, PAD_HEADS, DIM, dtype=torch.bfloat16, device=device)  # stock passes padded q
    q[:, :HEADS] = (torch.randn(T, HEADS, DIM, generator=gen, device=device) * q_gain).to(torch.bfloat16)
    sink_t = torch.full((PAD_HEADS,), float("-inf"), device=device)
    if sink == "normal":
        sink_t[:HEADS] = torch.randn(HEADS, generator=gen, device=device)
    elif sink == "big":
        sink_t[:HEADS] = 6.0 + torch.randn(HEADS, generator=gen, device=device)
    case.update(q=q, sink=sink_t)
    return case


# ------------------------------------------------------------------------------------------ references
def ref_decode(case) -> tuple[torch.Tensor, torch.Tensor]:
    """float64 reference [T, 16, 512] with FlashMLA index/sink semantics, plus sum_j p_j |k_j| / den (the
    contribution magnitude that bounds bf16 rounding of P and of the output)."""
    T = case["T"]
    out = torch.zeros(T, HEADS, DIM, dtype=torch.float64, device=case["q"].device)
    mag = torch.zeros_like(out)
    sink = case["sink"][:HEADS].double()
    W = case["swa_idx"].shape[-1]
    for r in range(T):
        sidx = case["swa_idx"][r, 0]
        keep = (torch.arange(W, device=sidx.device) < case["swa_lens"][r]) & (sidx >= 0)
        keys = [read_k(case["swa_raw"], case["swa"].shape[1], sidx[keep])]
        if "ex" in case:
            gidx = case["ex_glob"][r]
            keep = (torch.arange(gidx.numel(), device=gidx.device) < case["ex_lens"][r]) & (gidx >= 0)
            keys.append(read_k(case["ex_raw"], case["ex"].shape[1], gidx[keep]))
        K = torch.cat(keys)
        qq = case["q"][r, :HEADS].double()
        s = qq @ K.T * SCALE
        m = torch.maximum(s.max(-1).values if K.shape[0] else torch.full_like(sink, float("-inf")), sink)
        m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = torch.exp(s - m_safe[:, None])
        den = p.sum(-1) + torch.exp(sink - m_safe)
        o = (p @ K) / den[:, None]
        a = (p @ K.abs()) / den[:, None]
        out[r] = torch.where(den[:, None] > 0, o, torch.zeros_like(o))
        mag[r] = torch.where(den[:, None] > 0, a, torch.zeros_like(a))
    return out, mag


def run_ds41h(case, mode="local", out=None, **kw) -> torch.Tensor:
    attn = _attn()
    extra = {}
    if "ex" in case:
        if mode == "local":
            extra = dict(extra_cache=case["ex"], extra_indices=case["ex_local"], token_to_req=case["req"], block_table=case["ex_bt"], is_valid=case["is_valid"])
        else:
            extra = dict(extra_cache=case["ex"], extra_indices=case["ex_glob"].view(case["T"], 1, -1), extra_lens=case["ex_lens"])
    return attn.sparse_decode(case["q"], case["swa"], case["swa_idx"], case["swa_lens"], case["sink"], SCALE, out=out, **extra, **kw)


def flashmla_available() -> bool:
    try:
        from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

        return is_flashmla_sparse_supported()[0]
    except Exception:
        return False


def run_flashmla(case) -> torch.Tensor:
    """Stock FlashMLA sparse decode exactly as DeepseekV4FlashMLAAttention._forward_decode calls it."""
    from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata

    T = case["T"]
    out = torch.empty(T, PAD_HEADS, DIM, dtype=torch.bfloat16, device=case["q"].device)
    has_ex = "ex" in case
    flash_mla_with_kvcache(
        q=case["q"].unsqueeze(1),
        k_cache=case["swa"].unsqueeze(-2),
        block_table=None,
        head_dim_v=DIM,
        tile_scheduler_metadata=get_mla_metadata()[0],
        cache_seqlens=None,
        is_fp8_kvcache=True,
        indices=case["swa_idx"],
        topk_length=case["swa_lens"],
        softmax_scale=SCALE,
        attn_sink=case["sink"],
        extra_k_cache=case["ex"].unsqueeze(-2) if has_ex else None,
        extra_indices_in_kvcache=case["ex_glob"].view(T, 1, -1) if has_ex else None,
        extra_topk_length=case["ex_lens"] if has_ex else None,
        out=out.unsqueeze(1),
    )
    return out


def assert_close_ref(out: torch.Tensor, refpair, what: str, atol=1e-3, rtol=1e-2) -> None:
    """|out - ref| <= atol + rtol * sum_j p_j |k_j| / den. Expected error of a bf16-P / fp32-accumulate kernel
    is ~2^-8 of that magnitude (P rounding + output rounding), so rtol = 1e-2 leaves ~2.5x headroom."""
    ref, mag = refpair
    a = out[:, :HEADS].double()
    assert torch.isfinite(a).all(), f"{what}: non-finite output"
    err = (a - ref).abs()
    excess = err - (atol + rtol * mag)
    assert excess.max().item() <= 0, (
        f"{what}: max|err|={err.max().item():.4e}, worst excess {excess.max().item():.3e} "
        f"(max|ref|={ref.abs().max().item():.3e})"
    )


# ------------------------------------------------------------------------------------------ tests
CASES = {
    # name: (seq_lens, query_lens, kwargs)
    "c1_swa_only": ([300], [1], dict(topk=0)),
    "c1_short_swa": ([5], [1], dict(topk=0)),
    "dspark6_r1": ([1500], [6], dict(ratio=1)),
    "dspark6_r2_holes": ([1900], [6], dict(ratio=2, holes=True)),
    "dspark8_short_ctx": ([40], [8], dict(ratio=2)),  # < 512 compressed candidates, < 128 window
    "two_reqs_invalid": ([700, 1200], [4, 4], dict(ratio=1, invalid_rows=(7,))),
    "draft_noncausal": ([400], [5], dict(topk=0, noncausal=True)),  # DSpark drafter: 192-wide window
    "swa_bs32_stride": ([1100], [3], dict(ratio=1, swa_bs=32, extra_page_bytes=1152)),
    "sink_big_qgain": ([900], [2], dict(ratio=1, sink="big", q_gain=3.0)),
    "peaky_q": ([1300], [4], dict(ratio=2, q_gain=8.0)),
    "sink_none": ([900], [2], dict(ratio=2, sink="none")),
    "batch33": ([800] * 11, [3] * 11, dict(ratio=1)),
}


@pytest.mark.parametrize("cluster", [8, 16])
@pytest.mark.parametrize("name", list(CASES))
def test_vs_reference(name, cluster):
    seq, ql, kw = CASES[name]
    case = make_case(seq, ql, seed=zlib.crc32(name.encode()) % 1000, **kw)
    out = run_ds41h(case, cluster=cluster)
    torch.cuda.synchronize()
    assert out.shape == (case["T"], HEADS, DIM)
    ref = ref_decode(case)
    assert_close_ref(out, ref, f"{name}/cl{cluster}")
    for r in case["is_valid"].logical_not().nonzero().flatten().tolist():
        if case["sink"][:HEADS].isfinite().all():
            assert out[r, :HEADS].abs().max().item() == 0.0, "padded token must output 0"


@pytest.mark.parametrize("name", ["dspark6_r1", "dspark6_r2_holes", "two_reqs_invalid", "batch33"])
def test_local_equals_global(name):
    """Fused block-table lookup + count must give bit-identical output to precomputed global indices."""
    seq, ql, kw = CASES[name]
    case = make_case(seq, ql, seed=7, **kw)
    a = run_ds41h(case, mode="local", cluster=8)
    b = run_ds41h(case, mode="global", cluster=8)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


def test_stock_global_indices_match_replica():
    """Our replica of compute_global_topk_indices_and_lens equals the stock Triton kernel."""
    try:
        from vllm.models.deepseek_v41.common.ops import compute_global_topk_indices_and_lens
    except ImportError:
        pytest.skip("vLLM not importable")
    case = make_case([1900, 300], [6, 2], ratio=2, holes=True, invalid_rows=(3,), seed=3)
    glob, lens = compute_global_topk_indices_and_lens(case["ex_local"], case["req"], case["ex_bt"], case["ebs"], case["is_valid"])
    assert torch.equal(lens, case["ex_lens"])
    keep = torch.arange(glob.shape[1], device=glob.device)[None] < lens[:, None]
    assert torch.equal(torch.where(keep, glob, -1), torch.where(keep, case["ex_glob"], -1))


def _unused_slot(cache_view: torch.Tensor, block_table: torch.Tensor) -> int:
    nb, bs = cache_view.shape[0], cache_view.shape[1]
    used = torch.zeros(nb, dtype=torch.bool, device=block_table.device)
    used[block_table.flatten().long()] = True
    free = (~used).nonzero().flatten()
    assert free.numel() > 0
    return int(free[0]) * bs + bs - 1


def test_early_mode_matches():
    """Early mode (indices + non-fresh keys gathered before griddepcontrol.wait) gives bit-identical output, for
    any fresh set (none / some / all of the SWA slots), both index modes, clusters 8 and 16."""
    for name in ("c1_swa_only", "dspark6_r1", "dspark6_r2_holes", "dspark8_short_ctx", "two_reqs_invalid", "draft_noncausal"):
        seq, ql, kw = CASES[name]
        case = make_case(seq, ql, seed=zlib.crc32(name.encode()) % 1000, **kw)
        swa_used = case["swa_idx"][case["swa_idx"] >= 0].unique().to(torch.int64)
        for cl in (8, 16):
            for mode in (("local", "global") if "ex" in case else ("local",)):
                ref = run_ds41h(case, mode=mode, cluster=cl).clone()
                for fresh in (swa_used[:0], swa_used[-6:], swa_used[:64]):
                    out = run_ds41h(case, mode=mode, cluster=cl, fresh=fresh.contiguous()).clone()
                    assert torch.equal(out, ref), f"{name} cl={cl} {mode} fresh={fresh.numel()}: early mode differs"


def test_offtab_matches():
    """DS41H_ATTN_OFFTAB: the writer (offmode 1: resolve + store the key offsets) and a reader (offmode 2: load them instead of
    resolving) give bit-identical outputs to the regular launch, for every index mode case with extra keys, clusters 8 / 16,
    early mode with / without fresh slots (the FRESH_BIT travels in the table)."""
    attn = _attn()
    for name in ("dspark6_r1", "dspark6_r2_holes", "dspark8_short_ctx", "two_reqs_invalid", "swa_bs32_stride", "batch33"):
        seq, ql, kw = CASES[name]
        case = make_case(seq, ql, seed=zlib.crc32(name.encode()) % 1000, **kw)
        swa_used = case["swa_idx"][case["swa_idx"] >= 0].unique().to(torch.int64)
        tab = torch.full((case["T"] * attn.MAX_KEYS * 2 + 64,), -5, dtype=torch.int64, device="cuda")
        for cl in (8, 16):
            for fresh in (None, swa_used[-6:].contiguous()):
                fk = {} if fresh is None else dict(fresh=fresh)
                ref = run_ds41h(case, cluster=cl, **fk).clone()
                w = run_ds41h(case, cluster=cl, offtab=tab, offmode=1, **fk).clone()
                r = run_ds41h(case, cluster=cl, offtab=tab, offmode=2, **fk).clone()
                torch.cuda.synchronize()
                assert torch.equal(w, ref), f"{name} cl={cl} fresh={fresh is not None}: writer differs"
                assert torch.equal(r, ref), f"{name} cl={cl} fresh={fresh is not None}: reader differs"


def test_offtab_glue_pairing():
    """forward_decode with OFFTAB: an index-source layer (indexer set) writes the table, a following layer with the same metadata
    reads it (bit-identical outputs), a layer with other metadata resolves itself, and a stock-fallback writer drops the entry."""
    attn = _attn()
    old = attn.OFFTAB
    attn.OFFTAB = True
    try:
        seq, ql, kw = CASES["dspark6_r2_holes"]
        case = make_case(seq, ql, seed=77, **kw)
        layer, swa_md, attn_md = _stand_in_layer(case, False)
        T = case["T"]
        dev = case["q"].device.index or 0
        attn._offtab_key.pop(dev, None)
        outs = {}
        for role in ("plain", "writer", "reader"):
            layer.indexer = object() if role == "writer" else None
            if role == "plain":
                attn._offtab_key.pop(dev, None)
            out = torch.zeros((T, PAD_HEADS, DIM), dtype=torch.bfloat16, device="cuda")
            assert attn.forward_decode(layer, case["q"], case["ex"], swa_md, attn_md, False, out)
            torch.cuda.synchronize()
            outs[role] = out
            if role == "writer":
                assert dev in attn._offtab_key, "writer did not register the table"
        assert torch.equal(outs["writer"], outs["plain"]) and torch.equal(outs["reader"], outs["plain"])
        # other metadata (a different block table object) -> the reader must not use the table
        layer2, swa_md2, attn_md2 = _stand_in_layer(case, False)
        attn_md2.block_table = case["ex_bt"].clone()
        key_before = attn._offtab_key[dev]
        out = torch.zeros((T, PAD_HEADS, DIM), dtype=torch.bfloat16, device="cuda")
        layer2.indexer = None
        assert attn.forward_decode(layer2, case["q"], case["ex"], swa_md2, attn_md2, False, out)
        torch.cuda.synchronize()
        assert torch.equal(out, outs["plain"]) and attn._offtab_key[dev] == key_before
        # a writer that falls back to stock (ineligible) drops the entry
        layer3, swa_md3, attn_md3 = _stand_in_layer(case, False)
        layer3.indexer = object()
        layer3.n_local_heads = 32  # ineligible
        assert not attn.forward_decode(layer3, case["q"], case["ex"], swa_md3, attn_md3, False, out)
        assert dev not in attn._offtab_key, "stock-fallback writer must drop the table entry"
    finally:
        attn.OFFTAB = old


def test_flag_mode_matches():
    """Flag mode (DS41H_ATTN_FLAG: the wait is aflag[0] >= atarget instead of griddepcontrol.wait) gives bit-identical
    output, early mode or not, clusters 8 and 16, and its last CTA resets the counter to (0, 0)."""
    ctr = torch.zeros(2, dtype=torch.int32, device="cuda")
    for name in ("c1_swa_only", "dspark6_r1", "dspark6_r2_holes", "two_reqs_invalid"):
        seq, ql, kw = CASES[name]
        case = make_case(seq, ql, seed=zlib.crc32(name.encode()) % 1000, **kw)
        swa_used = case["swa_idx"][case["swa_idx"] >= 0].unique().to(torch.int64)
        for cl in (8, 16):
            ref = run_ds41h(case, cluster=cl).clone()
            for fresh in (None, swa_used[-6:].contiguous()):
                ctr.fill_(0)
                ctr[0] = 134  # the producer's signals (G2_CTAS + m) already landed
                out = run_ds41h(case, cluster=cl, fresh=fresh, aflag=ctr, atarget=134).clone()
                torch.cuda.synchronize()
                assert torch.equal(out, ref), f"{name} cl={cl} fresh={fresh is not None}: flag mode differs"
                assert ctr.tolist() == [0, 0], f"{name} cl={cl}: counter not reset ({ctr.tolist()})"


def test_flag_mode_waits_for_producer():
    """The flag-mode attention launched BEFORE the producer's signals (counter below target) completes only after a later
    kernel on another stream raises the counter -- i.e. it really waits on the counter -- and then matches."""
    case = make_case([1500], [6], ratio=1, seed=3)
    ref = run_ds41h(case).clone()
    ctr = torch.zeros(2, dtype=torch.int32, device="cuda")
    ctr[0] = 100
    out = torch.empty_like(ref)
    # load the producer's kernels first: with CUDA lazy loading, the first launch of a kernel can synchronize the device and
    # would then wait for the spinning attention (a test-only hazard: in production the producer is already running)
    torch.cuda._sleep(10)
    ctr[0:1].add_(0)
    torch.cuda.synchronize()
    # two side streams (neither is the legacy default stream, which would serialize with both): the attention spins on s1
    # (its 6 x 16 CTAs are resident) until the producer kernel on s2 raises the counter
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(s1):
        run_ds41h(case, out=out, aflag=ctr, atarget=134, pdl=False)
    with torch.cuda.stream(s2):
        torch.cuda._sleep(1_000_000)
        ctr[0:1].add_(34)  # producer done
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    assert ctr.tolist() == [0, 0]


def test_flag_handoff_glue():
    """forward_decode takes a pending prologue handoff of ITS module (flag mode, same output); a handoff registered by
    another module, or one whose attention falls back to stock, is reset instead (counter (0, 0), regular attention)."""
    from ds41h import prologue

    seq, ql, kw = CASES["dspark6_r2_holes"]
    case = make_case(seq, ql, seed=23, **kw)
    layer, swa_md, attn_md = _stand_in_layer(case, False)
    T = case["T"]
    ref = run_ds41h(case).clone()
    ctr = prologue.handoff_counter("cuda")
    for owner in (layer, object()):
        ctr.fill_(0)
        ctr[0] = prologue.handoff_target(T)
        prologue._handoff[torch.cuda.current_device()] = prologue.Handoff(owner, ctr, prologue.handoff_target(T))
        out = torch.full((T, PAD_HEADS, DIM), 3.0, dtype=torch.bfloat16, device="cuda")
        assert _attn().forward_decode(layer, case["q"], case["ex"], swa_md, attn_md, False, out)
        torch.cuda.synchronize()
        assert not prologue._handoff, "handoff not taken"
        assert ctr.tolist() == [0, 0], f"counter not reset ({ctr.tolist()})"
        assert torch.equal(out[:, :HEADS], ref[:, :HEADS])
    # ineligible layer with a pending handoff: reset, stock fallback (returns False)
    ctr.fill_(0)
    ctr[0] = prologue.handoff_target(T)
    layer.n_local_heads = 32
    prologue._handoff[torch.cuda.current_device()] = prologue.Handoff(layer, ctr, prologue.handoff_target(T))
    out = torch.empty(T, PAD_HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    assert not _attn().forward_decode(layer, case["q"], case["ex"], swa_md, attn_md, False, out)
    torch.cuda.synchronize()
    assert ctr.tolist() == [0, 0] and not prologue._handoff


def test_masked_entries_never_read():
    """Entries at j >= len point at NaN-poisoned records: output must be finite and bit-identical.
    A long request (full 128-key window, 512 top-k with holes) plus a short one (SWA len < 128, top-k len
    < 512), so both index lists have masked tails."""
    case = make_case([1500, 60], [6, 2], ratio=1, holes=True, seed=11)
    T, W = case["T"], case["swa_idx"].shape[-1]
    clean = run_ds41h(case, mode="global", cluster=8).clone()
    bad_swa = _unused_slot(case["swa"], case["swa_bt"])
    poison(case["swa_raw"], case["swa"].shape[1], torch.tensor([bad_swa], device="cuda"))
    bad_ex = _unused_slot(case["ex"], case["ex_bt"])
    poison(case["ex_raw"], case["ex"].shape[1], torch.tensor([bad_ex], device="cuda"))
    for r in range(T):
        n = int(case["swa_lens"][r])
        case["swa_idx"][r, 0, n:] = bad_swa
        case["ex_glob"][r, int(case["ex_lens"][r]):] = bad_ex
    assert int(case["swa_lens"].min()) < W
    out = run_ds41h(case, mode="global", cluster=8)
    torch.cuda.synchronize()
    assert torch.isfinite(out[:, :HEADS].float()).all()
    assert torch.equal(out[:, :HEADS], clean[:, :HEADS])


def test_writer_replica_matches_stock():
    try:
        import vllm.models.deepseek_v41.common.ops.cache_utils  # noqa: F401
    except ImportError:
        pytest.skip("vLLM not importable")
    gen = torch.Generator(device="cuda").manual_seed(5)
    k = random_k(300, "cuda", gen)
    slots = torch.randperm(64 * 32, generator=gen, device="cuda")[:300]
    raw_a, view_a = make_cache(64, 32, "cuda")
    raw_b = raw_a.clone()
    view_b = raw_b.as_strided(view_a.shape, view_a.stride())
    assert write_k(raw_a, view_a, k, slots, use_stock=True) == "stock"
    write_k(raw_b, view_b, k, slots, use_stock=False)
    da, db = read_k(raw_a, 32, slots), read_k(raw_b, 32, slots)
    assert torch.equal(da[:, NOPE:], db[:, NOPE:])  # RoPE part is a bf16 copy
    # NoPE: identical except where amax/448 is an exact power of two and Triton's division rounds up
    # (then the stock writer picks the next UE8M0 exponent); both must be valid quantizations.
    same = (da[:, :NOPE].view(-1, 7, 64) == db[:, :NOPE].view(-1, 7, 64)).all(-1)
    assert same.float().mean().item() >= 0.98
    amax = k.double()[:, :NOPE].abs().view(-1, 7, 64).amax(-1, keepdim=True)
    for d in (da, db):
        assert ((d[:, :NOPE].view(-1, 7, 64) - k.double()[:, :NOPE].view(-1, 7, 64)).abs() <= amax / 8).all()


def test_production_swa_writer_layout():
    """The production SWA writer torch.ops._C.fused_deepseek_v4_kv_rope_insert (attention.py:699, the op the
    stock decode path uses) writes the record the kernel reads. With identity RoPE (cos=1, sin=0) the bytes
    must read back as kv (fp8 error on NoPE, exact RoPE), and the kernel over them must match the reference."""
    try:
        import vllm._C  # noqa: F401  (registers torch.ops._C)

        op = torch.ops._C.fused_deepseek_v4_kv_rope_insert
    except (ImportError, AttributeError, RuntimeError) as e:
        pytest.skip(f"production writer unavailable: {e}")
    gen = torch.Generator(device="cuda").manual_seed(13)
    bs, nb, n = 64, 8, 300
    raw, view = make_cache(nb, bs, "cuda")
    kv = random_k(n, "cuda", gen)
    slots = torch.randperm(nb * bs, generator=gen, device="cuda")[:n].long()
    positions = torch.arange(n, device="cuda", dtype=torch.long)
    cos_sin = torch.cat([torch.ones(n, 32, device="cuda"), torch.zeros(n, 32, device="cuda")], dim=1)
    err = None
    for dt in (torch.float32, torch.bfloat16):
        try:
            op(kv, view, slots, positions, cos_sin.to(dt), bs, None, False)
            err = None
            break
        except RuntimeError as e:  # cos/sin dtype contract differs between builds
            err = e
    if err is not None:
        pytest.skip(f"fused_deepseek_v4_kv_rope_insert call failed: {err}")
    torch.cuda.synchronize()
    back = read_k(raw, bs, slots)
    assert torch.equal(back[:, NOPE:], kv[:, NOPE:].double()), "RoPE dims must be stored as bf16 at bytes 448..575"
    amax = kv.double()[:, :NOPE].abs().view(n, 7, 64).amax(-1, keepdim=True)
    assert ((back[:, :NOPE].view(n, 7, 64) - kv.double()[:, :NOPE].view(n, 7, 64)).abs() <= amax / 8).all()
    # kernel over production-written bytes: 3 rows, 128 keys each (with a short row)
    idx = torch.stack([slots[:128], slots[100:228], slots[172:300]]).to(torch.int32).view(3, 1, 128)
    lens = torch.tensor([128, 128, 77], dtype=torch.int32, device="cuda")
    q = torch.zeros(3, PAD_HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    q[:, :HEADS] = torch.randn(3, HEADS, DIM, generator=gen, device="cuda").to(torch.bfloat16)
    sink = torch.full((PAD_HEADS,), float("-inf"), device="cuda")
    sink[:HEADS] = torch.randn(HEADS, generator=gen, device="cuda")
    case = dict(T=3, q=q, sink=sink, swa_raw=raw, swa=view, swa_idx=idx, swa_lens=lens)
    out = run_ds41h(case)
    torch.cuda.synchronize()
    assert_close_ref(out, ref_decode(case), "production-writer bytes")


@pytest.mark.skipif(not (_HAS_SM90 and flashmla_available()), reason="stock FlashMLA sparse decode unavailable")
@pytest.mark.parametrize("name", ["c1_swa_only", "dspark6_r1", "dspark6_r2_holes", "two_reqs_invalid", "draft_noncausal", "sink_big_qgain", "batch33"])
def test_vs_stock_flashmla(name):
    seq, ql, kw = CASES[name]
    case = make_case(seq, ql, seed=17, **kw)
    ours = run_ds41h(case)
    stock = run_flashmla(case)
    torch.cuda.synchronize()
    ref = ref_decode(case)
    assert_close_ref(stock, ref, f"{name}/stock-flashmla")  # sanity of the harness itself
    assert_close_ref(ours, (stock[:, :HEADS].double(), ref[1]), f"{name}/ds41h-vs-flashmla", rtol=2e-2)


def _stand_in_layer(case, swa_only: bool):
    T = case["T"]
    topk_buf = None
    if not swa_only:
        topk_buf = torch.full((T + 5, case["ex_local"].shape[1]), -7, dtype=torch.int32, device="cuda")
        topk_buf[:T] = case["ex_local"]  # persistent buffer is larger than the step
    layer = types.SimpleNamespace(
        prefix="model.layers.0.attn",
        n_local_heads=HEADS,
        padded_heads=PAD_HEADS,
        compress_ratio=0 if swa_only else case["ratio"],
        topk_indices_buffer=topk_buf,
        swa_cache_layer=types.SimpleNamespace(kv_cache=case["swa"]),
        scale=SCALE,
        attn_sink=case["sink"],
    )
    md_kw = {}
    try:
        from vllm.v1.attention.ops.flashmla import get_mla_metadata

        md_kw = dict(tile_sched_swaonly=get_mla_metadata()[0], tile_sched_c1a=get_mla_metadata()[0], tile_sched_c2a=get_mla_metadata()[0])
    except Exception:
        pass
    tok2req = torch.zeros(T + 9, dtype=torch.int32, device="cuda")
    tok2req[:T] = case["req"]
    valid = torch.zeros(T + 9, dtype=torch.bool, device="cuda")
    valid[:T] = case["is_valid"]
    swa_md = types.SimpleNamespace(
        num_decodes=int(case["req"].max()) + 1,
        num_decode_tokens=T,
        is_valid_token=valid[:T],
        token_to_req_indices=tok2req,
        decode_swa_indices=case["swa_idx"],
        decode_swa_lens=case["swa_lens"],
        block_size=case["swa"].shape[1],
        **md_kw,
    )
    attn_md = None
    if not swa_only:
        attn_md = types.SimpleNamespace(block_table=case["ex_bt"], block_size=case["block_size"])
    return layer, swa_md, attn_md


@pytest.mark.parametrize("name", ["c1_swa_only", "dspark6_r2_holes", "two_reqs_invalid", "draft_noncausal"])
def test_forward_decode_glue(name):
    """ds41h.attn.forward_decode on stock-shaped layer/metadata objects equals the reference, and equals stock
    _forward_decode (FlashMLA) when available."""
    seq, ql, kw = CASES[name]
    case = make_case(seq, ql, seed=23, **kw)
    swa_only = "ex" not in case
    layer, swa_md, attn_md = _stand_in_layer(case, swa_only)
    kv = None if swa_only else case["ex"]
    T = case["T"]
    out = torch.full((T, PAD_HEADS, DIM), 3.0, dtype=torch.bfloat16, device="cuda")
    reason = _attn().decode_ineligible_reason(layer, case["q"], kv, swa_md, attn_md, swa_only)
    assert reason is None, f"stand-in layer judged ineligible: {reason}"
    # forward_decode also returns False when the C++ launcher rejects the call (logged as a WARNING).
    assert _attn().forward_decode(layer, case["q"], kv, swa_md, attn_md, swa_only, out), "launcher rejected the call (see log)"
    torch.cuda.synchronize()
    assert_close_ref(out, ref_decode(case), f"{name}/glue")
    assert (out[:, HEADS:] == 3.0).all(), "padding heads must be left untouched"
    if flashmla_available():
        from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention as cls

        stock = getattr(cls, "_ds41h_stock_forward_decode", cls._forward_decode)
        out_s = torch.empty_like(out)
        stock(layer, case["q"], kv, swa_md, attn_md, swa_only, out_s)
        torch.cuda.synchronize()
        assert_close_ref(out, (out_s[:, :HEADS].double(), ref_decode(case)[1]), f"{name}/glue-vs-stock", rtol=2e-2)


def test_ineligible_falls_back():
    case = make_case([300], [1], topk=0, seed=1)
    layer, swa_md, attn_md = _stand_in_layer(case, True)
    layer.n_local_heads = 32
    out = torch.empty(1, PAD_HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    assert not _attn().forward_decode(layer, case["q"], None, swa_md, None, True, out)


@pytest.mark.parametrize("early", [False, True])
def test_cuda_graph_replay(early):
    """Captured launch follows in-place updates of q / indices / lens (FULL decode graph contract); early mode with
    a captured fresh-slot buffer whose contents also change between replays."""
    case = make_case([1500], [6], ratio=1, seed=31)
    out = torch.empty(case["T"], PAD_HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    fresh = torch.full((6,), -1, dtype=torch.int64, device="cuda")
    kw = dict(fresh=fresh) if early else {}
    run_ds41h(case, out=out, **kw)  # warm-up: builds, sets attributes, caches occupancy outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_ds41h(case, out=out, **kw)
    if early:
        su = case["swa_idx"][case["swa_idx"] >= 0].unique().to(torch.int64)
        fresh.copy_(su[-6:])
    case2 = make_case([1500], [6], ratio=1, seed=32)
    # same shapes; copy new contents into the captured buffers
    for key in ("q", "swa_idx", "swa_lens", "ex_local", "req", "is_valid", "ex_bt", "sink", "swa_raw", "ex_raw"):
        assert case[key].shape == case2[key].shape, key
        case[key].copy_(case2[key])
    g.replay()
    torch.cuda.synchronize()
    eager = run_ds41h(case2)
    torch.cuda.synchronize()
    assert torch.equal(out[:, :HEADS], eager[:, :HEADS])


def test_matches_ds41h_ref():
    """Cross-check with the shared pure-torch reference package when it exposes sparse_attn_ref."""
    try:
        from ds41h.ref.layer import sparse_attn_ref
    except Exception:
        pytest.skip("ds41h.ref.layer.sparse_attn_ref not available")
    case = make_case([1500], [6], ratio=1, seed=41)
    T, W = case["T"], case["swa_idx"].shape[-1]
    rows, masks = [], []
    for r in range(T):
        sidx = case["swa_idx"][r, 0]
        gidx = case["ex_glob"][r]
        ks = read_k(case["swa_raw"], case["swa"].shape[1], sidx.clamp(min=0))
        ke = read_k(case["ex_raw"], case["ex"].shape[1], gidx.clamp(min=0))
        rows.append(torch.cat([ks, ke]))
        masks.append(torch.cat([(torch.arange(W, device="cuda") < case["swa_lens"][r]) & (sidx >= 0),
                                (torch.arange(gidx.numel(), device="cuda") < case["ex_lens"][r]) & (gidx >= 0)]))
    valid = torch.stack(masks)
    kv = torch.where(valid[..., None], torch.stack(rows), 0.0).float()
    q = case["q"][:, :HEADS].float()
    sink = case["sink"][:HEADS].float()
    params = inspect.signature(sparse_attn_ref).parameters
    try:
        ref = sparse_attn_ref(q, kv, sink, valid, **({"scale": SCALE} if "scale" in params else {}))
    except TypeError as e:
        pytest.skip(f"incompatible sparse_attn_ref signature: {e}")
    if isinstance(ref, tuple):
        ref = ref[0]
    out = run_ds41h(case)
    torch.cuda.synchronize()
    ours_ref = ref_decode(case)
    assert_close_ref(ref.double().view(T, HEADS, DIM), ours_ref, "ds41h.ref vs local reference")
    assert_close_ref(out, ours_ref, "ds41h kernel")


# ------------------------------------------------------------------------------------------ benchmark
def _bench_graph(fn, iters: int, reps: int = 20) -> float:
    """Median us per call of fn, with `reps` calls captured back-to-back in one CUDA graph (like the FULL
    decode graph, where launches are queued without host gaps)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    times = []
    for _ in range(max(iters // reps, 5)):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / reps)
    times.sort()
    return times[len(times) // 2]


def bench(rows_list, iters: int, ctx: int, cold_sets: int) -> None:
    """Per-layer decode attention time. 'hot' = the same indices every call (keys L2-resident), 'cold' =
    cycling through `cold_sets` independent batches (~19 MB of caches each) so keys mostly come from HBM.
    'layer' rows time the full per-layer decode attention: ds41h.attn.forward_decode vs stock
    _forward_decode (compute_global_topk_indices_and_lens + FlashMLA, planner amortized like stock)."""
    attn = _attn()
    have_stock = flashmla_available()
    print(f"device: {torch.cuda.get_device_name()}  ctx={ctx}  keys/row = 128 SWA + 512 top-k (ratio-1 layer); "
          f"SWA-only rows use 128 keys. Graph-captured back-to-back calls, median us/call.")
    for M in rows_list:
        sets = [make_case([ctx], [M], ratio=1, seed=100 + i, use_stock_writer=False) for i in range(max(cold_sets, 1))]
        sw_only = make_case([ctx], [M], topk=0, seed=99, use_stock_writer=False)
        c0 = sets[0]
        flops = 4.0 * M * HEADS * 640 * DIM
        uniq = int(torch.unique(c0["swa_idx"][c0["swa_idx"] >= 0]).numel()) + int(torch.unique(c0["ex_glob"][c0["ex_glob"] >= 0]).numel())
        nbytes = uniq * REC + M * HEADS * DIM * 2 * 2
        out = torch.empty(M, PAD_HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        res: dict[str, float] = {}
        for cl in (0, 8, 16):
            tag = f"ds41h kernel cl={cl or 'auto'}"
            res[f"{tag} hot"] = _bench_graph(lambda: run_ds41h(c0, out=out, cluster=cl), iters)
            if cold_sets > 1:
                ring = iter(range(10**9))
                res[f"{tag} cold"] = _bench_graph(lambda: run_ds41h(sets[next(ring) % len(sets)], out=out, cluster=cl), iters, reps=len(sets))
        res["ds41h kernel SWA-only"] = _bench_graph(lambda: run_ds41h(sw_only, out=out), iters)
        layer, swa_md, attn_md = _stand_in_layer(c0, False)
        res["ds41h layer (forward_decode)"] = _bench_graph(lambda: attn.forward_decode(layer, c0["q"], c0["ex"], swa_md, attn_md, False, out), iters)
        if have_stock:
            from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention as cls

            stock = getattr(cls, "_ds41h_stock_forward_decode", cls._forward_decode)
            res["stock layer (_forward_decode)"] = _bench_graph(lambda: stock(layer, c0["q"], c0["ex"], swa_md, attn_md, False, out), iters)
            if cold_sets > 1:
                stand = [_stand_in_layer(c, False) for c in sets]
                ring = iter(range(10**9))

                def stock_cold():
                    i = next(ring) % len(sets)
                    lyr, smd, amd = stand[i]
                    stock(lyr, sets[i]["q"], sets[i]["ex"], smd, amd, False, out)

                res["stock layer cold"] = _bench_graph(stock_cold, iters, reps=len(sets))
            lyr0, smd0, _ = _stand_in_layer(sw_only, True)
            res["stock layer SWA-only"] = _bench_graph(lambda: stock(lyr0, sw_only["q"], None, smd0, None, True, out), iters)
        n16 = attn.ext().max_active_clusters(16, attn.SPREAD)
        print(f"\nM={M}: auto cluster = {16 if M <= n16 else 8} for 640 keys, 8 for SWA-only (max active 16-CTA clusters: {n16}); "
              f"unique key bytes {uniq * REC / 1e6:.2f} MB")
        for k, us in res.items():
            rate = "" if "SWA-only" in k else f"  {nbytes / us / 1e3:7.1f} GB/s  {flops / us / 1e6:6.2f} TFLOPS(16 heads)"
            print(f"  {k:34s} {us:8.2f} us{rate}")


PROF_NAMES = {1: "pdl", 2: "offsets", 3: "keys+q", 4: "scores", 5: "softmax", 6: "PV+send", 7: "recv", 8: "out", 15: "exit"}


def prof(rows_list, clusters, ctx: int, iters: int, local: bool = True, spreads=(True,)) -> None:
    """Phase timeline of one launch inside a graph of 20 back-to-back calls (the 10th is stamped), from a
    -DDS41H_ATTN_PROF build: median / max over CTAs of each stamp relative to the earliest CTA entry."""
    attn = _attn()
    e = cuext_load_prof()
    attn.ext = lambda: e
    NS = int(e.PROF_SLOTS)
    for M in rows_list:
        c0 = make_case([ctx], [M], ratio=1, seed=100, use_stock_writer=False)
        out = torch.empty(M, PAD_HEADS, DIM, dtype=torch.bfloat16, device="cuda")
        for cl, spread in [(c, sp) for c in clusters for sp in spreads]:
            buf = torch.zeros(M * 16 * NS, dtype=torch.int64, device="cuda")
            kw = dict(mode="local" if local else "global", out=out, cluster=cl, spread=spread)
            for _ in range(3):
                run_ds41h(c0, **kw)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(20):
                    e.set_prof(buf if i == 10 else None)
                    run_ds41h(c0, **kw)
            e.set_prof(None)
            acc = []
            for _ in range(iters):
                buf.zero_()
                g.replay()
                torch.cuda.synchronize()
                acc.append(buf.view(-1, NS).cpu().clone())
            times = []
            for _ in range(20):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                b.record()
                b.synchronize()
                times.append(a.elapsed_time(b) * 1e3 / 20)
            st = torch.stack(acc).double()
            valid = st[:, :, 0] > 0
            ncta = int(valid[0].sum())
            st = st[:, valid[0]]
            t0 = st[:, :, 0].min(dim=1, keepdim=True).values
            rel = (st - t0[..., None]) / 1e3
            clk = (st[:, :, 14] - st[:, :, 13]) / (st[:, :, 15] - st[:, :, 0])
            parts = []
            mhz = clk.median().item() * 1e3
            prev = 1
            for slot, name in PROF_NAMES.items():
                v = rel[:, :, slot]
                if slot != 1:  # exact phase duration from clock64 (median over CTAs / replays)
                    d = (st[:, :, 16 + slot] - st[:, :, 16 + prev]) / mhz
                    parts.append(f"{name} {v.median(dim=1).values.median().item():.2f}/{v.max(dim=1).values.median().item():.2f} (+{d.median().item():.2f})")
                    prev = slot
                else:
                    parts.append(f"{name} {v.median(dim=1).values.median().item():.2f}/{v.max(dim=1).values.median().item():.2f}")
            print(f"M={M} cl={cl or 'auto'} spread={int(spread)} ({ncta} CTAs, {'local' if local else 'global'}): {sorted(times)[10]:.2f} us/call, "
                  f"SM clock {clk.median().item() * 1e3:.0f} MHz | entry {rel[:, :, 0].median(dim=1).values.median().item():.2f}/"
                  f"{rel[:, :, 0].max(dim=1).values.median().item():.2f} | " + " | ".join(parts))
            del g


def cuext_load_prof():
    from ds41h import cuext

    return cuext.load("ds41h_sparse_decode_prof", "sparse_decode.cu", extra_cuda_cflags=("-DDS41H_ATTN_PROF",))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--prof", action="store_true", help="phase timeline from a -DDS41H_ATTN_PROF build")
    ap.add_argument("--clusters", default="8,16")
    ap.add_argument("--spreads", default="1,0")
    ap.add_argument("--rows", default="1,6,8")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--ctx", type=int, default=16384, help="context length (positions) of the benchmark request")
    ap.add_argument("--cold-sets", type=int, default=24, help="independent index sets cycled to defeat L2 (1 = hot only)")
    args = ap.parse_args()
    if not _HAS_SM90:
        sys.exit("needs an sm90 GPU")
    if args.prof:
        for local in (True, False):
            prof([int(x) for x in args.rows.split(",")], [int(x) for x in args.clusters.split(",")], args.ctx, 30, local=local,
                 spreads=[bool(int(x)) for x in args.spreads.split(",")])
    elif args.bench:
        bench([int(x) for x in args.rows.split(",")], args.iters, args.ctx, args.cold_sets)
    elif hasattr(pytest, "main"):
        sys.exit(pytest.main([__file__, "-q"]))
    else:
        sys.exit(_run_all())


def _run_all() -> int:
    """Run every test_* function (expanding parametrize grids) without pytest. Returns #failures."""
    import itertools
    import traceback

    passed = failed = skipped = 0
    for name, fn in list(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        grid = getattr(fn, "_params", [])
        combos = itertools.product(*[vals for _, vals in grid]) if grid else [()]
        for combo in combos:
            kwargs = {k: v for (k, _), v in zip(grid, combo)}
            label = f"{name}[{'-'.join(map(str, combo))}]" if combo else name
            reason = getattr(fn, "_skip_reason", None)
            if reason:
                print(f"SKIP {label}: {reason}")
                skipped += 1
                continue
            try:
                fn(**kwargs)
                print(f"PASS {label}")
                passed += 1
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "_Skipped":
                    print(f"SKIP {label}: {e}")
                    skipped += 1
                else:
                    print(f"FAIL {label}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    failed += 1
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    return failed


if __name__ == "__main__":
    main()
