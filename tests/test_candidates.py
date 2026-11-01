"""GPU tests for ds41h CANDIDATES (kernels/candidates.cu, ds41h/candidates.py).

Run (needs a GPU, with DS41H unset so the vLLM functions compared against are stock):
    python -m pytest tests/test_candidates.py -q
    python tests/test_candidates.py            # same tests without pytest
    python tests/test_candidates.py --bench    # micro-benchmark vs stock vLLM

References:
  * torch reference of stock semantics (block scores with NaN propagation + newest-block pin,
    torch.topk(scores, min(k, nblocks)), -inf -> -1), i.e. exactly what stock
    candidate_blocks.select_candidate_blocks computes, compared as index *sets* per row;
  * the stock vLLM functions themselves when vllm is importable (select_candidate_blocks,
    apply_candidate_mask: bitwise-identical masked logits; SparseIndexerTopk "torch" backend).
"""

from __future__ import annotations

import sys

import torch

from common import bench, fmt_bw, require_cuda, run_module_tests

NEG_INF = float("-inf")


def _ext():
    from ds41h import candidates

    return candidates


def _stock_candidate_blocks():
    try:
        from vllm.model_executor.kernels.attention.dsa import candidate_blocks as cb
    except Exception:  # noqa: BLE001
        return None
    sel = getattr(cb.select_candidate_blocks, "_ds41h_stock", cb.select_candidate_blocks)
    mask = getattr(cb.apply_candidate_mask, "_ds41h_stock", cb.apply_candidate_mask)
    if getattr(sel, "_ds41h", False) or getattr(mask, "_ds41h", False):
        return None
    return sel, mask


# ---------------------------------------------------------------------------------------- references
def block_scores_ref(logits, starts, ends, block_size, row_repeat=1):
    """Stock _block_scores_kernel (candidate_blocks.py:14-51) in torch."""
    rows, width = logits.shape
    nblocks = -(-width // block_size)
    dev = logits.device
    r = torch.arange(rows, device=dev) // row_repeat
    start = starts[r].long() if starts is not None else torch.zeros(rows, dtype=torch.long, device=dev)
    end = ends[r].long()
    blocks = torch.arange(nblocks, device=dev)
    off = torch.arange(block_size, device=dev)
    cols = start[:, None, None] + blocks[None, :, None] * block_size + off[None, None, :]
    valid = (cols < end[:, None, None]) & (cols < width)
    vals = torch.gather(logits, 1, cols.clamp(max=width - 1).reshape(rows, -1)).reshape(rows, nblocks, block_size)
    vals = torch.where(valid, vals, torch.full_like(vals, NEG_INF))
    scores = vals.amax(-1)  # propagates NaN like tl.maximum(propagate_nan=ALL)
    pinned = (end > start) & True
    pin_idx = torch.div(end - start - 1, block_size, rounding_mode="floor")
    pin = pinned[:, None] & (blocks[None, :] == pin_idx[:, None])
    return torch.where(pin, torch.full_like(scores, float("inf")), scores)


def select_ref(logits, starts, ends, k, block_size, row_repeat=1):
    scores = block_scores_ref(logits, starts, ends, block_size, row_repeat)
    top = scores.topk(min(k, scores.shape[1]), dim=-1)
    idx = torch.where(top.values != NEG_INF, top.indices, torch.full_like(top.indices, -1))
    return [set(row[row >= 0].tolist()) for row in idx]


def mask_ref(logits, starts, ends, cand, block_size, row_repeat=1):
    """Stock _candidate_flags_kernel + _mask_candidates_kernel (candidate_blocks.py:77-138)."""
    out = logits.clone()
    rows, width = logits.shape
    nblocks = -(-width // block_size)
    for row in range(rows):
        b = row // row_repeat
        start = int(starts[b]) if starts is not None else 0
        end = int(ends[b])
        flags = torch.zeros(nblocks + 1, dtype=torch.bool)
        for c in cand[row].tolist():
            blk = c
            if start + blk * block_size >= width:
                blk = nblocks
            if blk >= 0:
                flags[blk] = True
        cols = torch.arange(width)
        valid = (cols >= start) & (cols < end)
        blk = torch.div(cols - start, block_size, rounding_mode="floor").clamp(0, nblocks)
        keep = torch.where(valid, flags[blk], torch.zeros_like(valid)) | ((cols == width - 1) & flags[nblocks])
        out[row, (~(valid & keep)).to(logits.device)] = NEG_INF
    return out


def topk_rows_ref(logits, row_ends, k):
    """Stock SparseIndexerTopk 'torch' backend (indexer_topk.py:359-371) as sets per row."""
    cols = torch.arange(logits.shape[1], device=logits.device)
    masked = logits.masked_fill(cols[None, :] >= row_ends[:, None].long(), NEG_INF)
    idx = masked.topk(k, dim=-1).indices
    in_range = torch.arange(k, device=logits.device)[None, :] < row_ends[:, None].long()
    idx = torch.where(in_range, idx, torch.full_like(idx, -1))
    return [set(row[row >= 0].tolist()) for row in idx]


# ---------------------------------------------------------------------------------------- inputs
def make_logits(rows, width, kind, gen, device="cuda"):
    if kind == "normal":
        x = torch.randn(rows, width, generator=gen, device=device)
    elif kind == "ties":  # few distinct values -> many equal block maxima
        x = torch.randint(-3, 3, (rows, width), generator=gen, device=device).float()
    elif kind == "zeros":  # every block ties; even rows +0.0, odd rows -0.0 (never mixed inside a row:
        # max(-0.0, +0.0) is order-dependent in torch.amax, so it is no reference for mixed signs)
        x = torch.zeros(rows, width, device=device)
        x[1::2] = -0.0
    elif kind == "special":  # NaN / +-inf sprinkled, -inf regions
        x = torch.randn(rows, width, generator=gen, device=device)
        u = torch.rand(rows, width, generator=gen, device=device)
        x[u < 0.3] = NEG_INF
        x[(u > 0.3) & (u < 0.3005)] = float("nan")
        x[(u > 0.5) & (u < 0.5005)] = float("inf")
    else:
        raise ValueError(kind)
    return x.contiguous()


def check_select_case(rows, width, k, block_size, kind, *, with_starts=False, row_repeat=1, seed=0, stock=None):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = make_logits(rows, width, kind, gen)
    nb = rows // row_repeat if rows % row_repeat == 0 else rows
    if with_starts:
        starts = torch.randint(0, width, (nb,), generator=gen, device="cuda", dtype=torch.int32)
        lens = torch.randint(0, width, (nb,), generator=gen, device="cuda", dtype=torch.int32)
        ends = torch.minimum(starts + lens, torch.full_like(starts, width + 37)).to(torch.int32)  # some end > width
    else:
        starts = None
        ends = torch.randint(0, width + 1, (nb,), generator=gen, device="cuda", dtype=torch.int32)
        ends[0] = width  # full row
        if nb > 1:
            ends[1] = 0  # empty row
    out = torch.full((rows, k), 123456, dtype=torch.int32, device="cuda")
    assert _ext().select_candidate_blocks(logits, starts, ends, k, block_size, out, row_repeat)
    ref = select_ref(logits, starts, ends, k, block_size, row_repeat)
    nblocks = -(-width // block_size)
    for r in range(rows):
        row = out[r].tolist()
        got = [v for v in row if v >= 0]
        assert all(v == -1 for v in row[len(got) :]), f"row {r}: -1 padding must follow the selected ids"
        assert got == sorted(got) and len(set(got)) == len(got), f"row {r}: ids must be ascending and unique"
        assert all(0 <= v < nblocks for v in got)
        assert set(got) == ref[r], f"row {r} ({kind}, w={width}, k={k}): sets differ: {sorted(set(got) ^ ref[r])[:16]}"
    if stock is not None:
        out_s = torch.empty_like(out)
        stock[0](logits, starts, ends, k, block_size, out_s, row_repeat)
        for r in range(rows):
            s = set(v for v in out_s[r].tolist() if v >= 0)
            assert s == ref[r], f"torch reference disagrees with stock vLLM on row {r}"
    return out, logits, starts, ends


# ---------------------------------------------------------------------------------------- tests
def test_select_blocks_decode_shapes():
    require_cuda()
    stock = _stock_candidate_blocks()
    for width, k in [(73728, 2048), (16384, 2048), (4096, 2048), (20000, 512), (100, 2048), (262144, 2048)]:
        for kind in ("normal", "ties", "special", "zeros"):
            check_select_case(6, width, k, 8, kind, seed=width + len(kind), stock=stock)


def test_select_blocks_row_repeat_and_starts():
    require_cuda()
    stock = _stock_candidate_blocks()
    check_select_case(12, 30000, 2048, 8, "ties", row_repeat=6, seed=1, stock=stock)  # spec-decode rows share bounds
    check_select_case(64, 16384, 2048, 8, "normal", with_starts=True, seed=2, stock=stock)  # prefill chunk
    check_select_case(64, 20480, 512, 8, "ties", with_starts=True, seed=3, stock=stock)
    check_select_case(8, 5000, 300, 5, "special", with_starts=True, seed=4, stock=stock)  # generic block size


def test_select_blocks_strided_out_and_empty():
    require_cuda()
    gen = torch.Generator(device="cuda").manual_seed(7)
    logits = make_logits(5, 9000, "normal", gen)
    ends = torch.tensor([9000, 8000, 7, 0, 4500], dtype=torch.int32, device="cuda")
    big = torch.full((8, 4096), 77, dtype=torch.int32, device="cuda")
    view = big[1:6, 100 : 100 + 2048]  # non-contiguous rows
    assert _ext().select_candidate_blocks(logits, None, ends, 2048, 8, view, 1)
    ref = select_ref(logits, None, ends, 2048, 8)
    for r in range(5):
        assert set(v for v in view[r].tolist() if v >= 0) == ref[r]
    assert (big[0] == 77).all() and (big[6:] == 77).all() and (big[1:6, :100] == 77).all()
    empty = torch.empty(3, 0, device="cuda")
    out = torch.zeros(3, 16, dtype=torch.int32, device="cuda")
    assert _ext().select_candidate_blocks(empty, None, ends[:3], 16, 8, out, 1)
    assert (out == -1).all()


def test_mask_matches_stock():
    require_cuda()
    stock = _stock_candidate_blocks()
    gen = torch.Generator(device="cuda").manual_seed(11)
    for rows, width, with_starts, repeat in [(6, 73728, False, 1), (12, 9000, False, 6), (16, 16384, True, 1)]:
        logits = make_logits(rows, width, "normal", gen)
        nb = rows // repeat
        ends = torch.randint(1, width + 1, (nb,), generator=gen, device="cuda", dtype=torch.int32)
        starts = None
        if with_starts:
            starts = torch.randint(0, width // 2, (nb,), generator=gen, device="cuda", dtype=torch.int32)
            ends = torch.maximum(ends, starts + 1)
        nblocks = -(-width // 8)
        cand = torch.randint(-1, nblocks + 3, (rows, 2048), generator=gen, device="cuda", dtype=torch.int32)
        cand[:, :4] = torch.tensor([nblocks - 1, nblocks, -1, 0], dtype=torch.int32)  # edge clamp, padding
        got = logits.clone()
        assert _ext().apply_candidate_mask(got, starts, ends, cand, 8, repeat)
        ref = mask_ref(logits.cpu(), starts.cpu() if starts is not None else None, ends.cpu(), cand.cpu(), 8, repeat)
        assert torch.equal(got.cpu(), ref), f"mask mismatch rows={rows} width={width}"
        if stock is not None:
            st = logits.clone()
            stock[1](st, starts, ends, cand, 8, repeat)
            assert torch.equal(got, st), "mask differs from stock vLLM"


def test_select_then_mask_roundtrip():
    require_cuda()
    out, logits, starts, ends = check_select_case(6, 73728, 2048, 8, "normal", seed=21)
    a = logits.clone()
    assert _ext().apply_candidate_mask(a, starts, ends, out, 8, 1)
    stock = _stock_candidate_blocks()
    if stock is not None:
        out_s = torch.empty_like(out)
        stock[0](logits, starts, ends, 2048, 8, out_s, 1)
        b = logits.clone()
        stock[1](b, starts, ends, out_s, 8, 1)
        assert torch.equal(a, b), "select+mask pipeline must give stock's masked logits bit for bit"


def test_topk_rows():
    require_cuda()
    gen = torch.Generator(device="cuda").manual_seed(5)
    for rows, width, next_n, kind in [(6, 73728, 6, "normal"), (6, 36864, 1, "ties"), (16, 2000, 4, "special"), (4, 600, 1, "normal")]:
        logits = make_logits(rows, width, kind, gen)
        if next_n > 1 and rows % next_n == 0:
            seq = torch.randint(1, width + 1, (rows // next_n,), generator=gen, device="cuda", dtype=torch.int32)
            j = torch.arange(rows, device="cuda") % next_n
            ends = (seq.repeat_interleave(next_n) - next_n + 1 + j).clamp(min=0).to(torch.int32)
            seq_lens = seq.view(-1, 1)
        else:
            seq_lens = torch.randint(0, width + 1, (rows,), generator=gen, device="cuda", dtype=torch.int32)
            seq_lens[0] = 300  # shorter than k
            ends = seq_lens
            next_n = 1
        out = torch.full((rows, 512), 99, dtype=torch.int32, device="cuda")
        assert _ext().indexer_topk(logits, seq_lens.contiguous(), next_n, out, 512)
        ref = topk_rows_ref(logits, ends, 512)
        for r in range(rows):
            row = out[r].tolist()
            got = [v for v in row if v >= 0]
            assert got == sorted(got) and all(v == -1 for v in row[len(got) :])
            assert set(got) == ref[r], f"topk row {r} differs ({kind})"


def test_topk_expand():
    require_cuda()
    gen = torch.Generator(device="cuda").manual_seed(9)
    logits = make_logits(3, 4000, "normal", gen)
    ends = torch.tensor([4000, 100, 2500], dtype=torch.int32, device="cuda")
    vis = torch.tensor([31995, 790, 20000], dtype=torch.int32, device="cuda")
    out = torch.empty(3, 512 * 8, dtype=torch.int32, device="cuda")
    lengths = torch.empty(3, dtype=torch.int32, device="cuda")
    _ext().topk_sorted(logits, ends, 512, out, lengths, vis)
    ref = topk_rows_ref(logits, ends, 512)
    for r in range(3):
        blocks = sorted(ref[r])
        exp = [b * 8 + o if b * 8 + o < int(vis[r]) else -1 for b in blocks for o in range(8)]
        exp += [-1] * (512 * 8 - len(exp))
        assert out[r].tolist() == exp
        assert int(lengths[r]) == sum(1 for v in exp if v >= 0)


def test_cuda_graph_replay():
    require_cuda()
    gen = torch.Generator(device="cuda").manual_seed(13)
    rows, width, k = 6, 32768, 2048
    logits = make_logits(rows, width, "normal", gen)
    ends = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    out = torch.empty(rows, k, dtype=torch.int32, device="cuda")
    masked = logits.clone()
    ext = _ext()
    ext.select_candidate_blocks(logits, None, ends, k, 8, out, 1)  # warm up (module load, smem attribute)
    ext.apply_candidate_mask(masked, None, ends, out, 8, 1)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ext.select_candidate_blocks(logits, None, ends, k, 8, out, 1)
        masked.copy_(logits)
        ext.apply_candidate_mask(masked, None, ends, out, 8, 1)
    for it in range(4):
        logits.copy_(make_logits(rows, width, "ties" if it % 2 else "normal", gen))
        ends.copy_(torch.randint(1, width + 1, (rows,), generator=gen, device="cuda", dtype=torch.int32))
        g.replay()
        torch.cuda.synchronize()
        ref = select_ref(logits, None, ends, k, 8)
        for r in range(rows):
            assert set(v for v in out[r].tolist() if v >= 0) == ref[r]
        exp = mask_ref(logits.cpu(), None, ends.cpu(), out.cpu(), 8)
        assert torch.equal(masked.cpu(), exp)


# ---------------------------------------------------------------------------------------- bench
def run_bench():
    require_cuda()
    ext = _ext()
    stock = _stock_candidate_blocks()
    gen = torch.Generator(device="cuda").manual_seed(0)
    print("\n== select_candidate_blocks (k=2048, block 8) ==")
    for rows, width, ctx, label in [
        (6, 73728, 65536, "decode c1 k=5, 64K ctx"),
        (6, 73728, 16384, "decode c1 k=5, 16K ctx"),
        (96, 73728, 16384, "decode c16 k=5, 16K ctx"),
        (8192, 16384, None, "prefill chunk 8192 x 16K"),
    ]:
        logits = make_logits(rows, width, "normal", gen)
        if ctx is None:
            starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
            ends = torch.arange(1, rows + 1, device="cuda", dtype=torch.int32) * (width // rows)
        else:
            starts, ends = None, torch.full((rows,), ctx, dtype=torch.int32, device="cuda")
        out = torch.empty(rows, 2048, dtype=torch.int32, device="cuda")
        nbytes = rows * width * 4
        t = bench(lambda: ext.select_candidate_blocks(logits, starts, ends, 2048, 8, out, 1))
        print(fmt_bw(f"ds41h  {label}", t, nbytes))
        if stock is not None:
            t = bench(lambda: stock[0](logits, starts, ends, 2048, 8, out, 1))
            print(fmt_bw(f"stock  {label}", t, nbytes))
        masked = logits.clone()
        t = bench(lambda: ext.apply_candidate_mask(masked, starts, ends, out, 8, 1))
        print(fmt_bw(f"ds41h  mask {label}", t, nbytes))
        if stock is not None:
            t = bench(lambda: stock[1](masked, starts, ends, out, 8, 1))
            print(fmt_bw(f"stock  mask {label}", t, nbytes))
    print("\n== decode top-512 (indexer) ==")
    try:
        from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk
    except Exception:  # noqa: BLE001
        SparseIndexerTopk = None
    for rows, width, ctx in [(6, 73728, 65536), (6, 36864, 16384), (96, 73728, 16384)]:
        logits = make_logits(rows, width, "normal", gen)
        seq_lens = torch.full((rows,), ctx, dtype=torch.int32, device="cuda")
        out = torch.empty(rows, 512, dtype=torch.int32, device="cuda")
        t = bench(lambda: ext.indexer_topk(logits, seq_lens, 1, out, 512))
        print(fmt_bw(f"ds41h  topk rows={rows} w={width} ctx={ctx}", t, rows * ctx * 4))
        if SparseIndexerTopk is not None:
            try:
                import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

                ws = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
                t = bench(lambda: torch.ops._C.cooperative_topk(logits, seq_lens.view(-1, 1), out, ws, 512, ctx))
                print(fmt_bw(f"stock  cooperative_topk rows={rows} ctx={ctx}", t, rows * ctx * 4))
            except Exception as e:  # noqa: BLE001
                print(f"stock cooperative_topk unavailable: {e}")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        run_bench()
    else:
        sys.exit(run_module_tests(globals()))
