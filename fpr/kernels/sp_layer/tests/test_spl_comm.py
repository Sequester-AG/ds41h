"""comm.cu, the engine's sequence-parallel collectives, on 4 GPUs: bitwise against torch references.

Every rank owns R rows. The all-gather (AG) gives every rank all 4 R rows; the reduce-scatter (RS) sums each owner's rows
of the 4 ranks' MoE partials. Data moves by SM stores into the peers' memory, then a system fence and one release add
on the receiver's flag per CTA; receivers spin until the flag reaches epoch x signals. The flags are never reset.

Guards, at the row counts the engine runs: the decode buckets, and mixed-step buckets (multiples of 64 rows, which also
covers context-parallel sub-chunks) up to the largest one.
  - AG (ag_push2, the engine's: f rows + 64-B routing rows; ag_push, one row set): every rank's gather buffer holds
    every rank's rows in slot order, byte for byte.
  - RS (rs_push + rs_sum): every owner's receive slot s holds rank s's partial rows, and the output equals torch's
    fixed-order sum (fp32, slots 0..3, one bf16 rounding), bit for bit.
  - combine_push (MoE combine fused with the RS push): the owners' slots hold the combine's bits (6 slot rows summed in
    slot order in fp32, plus the shared-expert addend, one bf16 rounding).
  - The flag protocol: consecutive rounds, and CUDA-graph replays of a decode round, keep exact outputs and counters.
  - peer_copy (the context-parallel exchanges' copy): the bytes arrive intact, ordered only by an event.
Inputs are random; nothing here selects or sorts, so ties cannot change a result.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

W, D, NCTA = 4, 5120, 16           # NCTA: the engine's AG / RS CTAs per destination
RB2 = 64                            # routing row bytes (6 int32 ids, 6 fp32 weights, padding)
DECODE = (32, 64, 112, 120, 128, 192, 224, 240, 256, 320)
MIXED = (576, 1024, 1984, 2048, 3072, 4288, 4416)
ROUNDS = 2


def _gens(devs, seed):
    return [torch.Generator(device=d).manual_seed(seed * 16 + i) for i, d in enumerate(devs)]


def _u8(shape, g, d):
    return torch.randint(0, 256, shape, dtype=torch.uint8, device=d, generator=g)


def rs_ref(slots):
    """comm.cu k_rs_sum's arithmetic: fp32 sum in slot order 0..3, one bf16 rounding."""
    a = slots[0].float()
    for t in slots[1:]:
        a = a + t.float()
    return a.bfloat16()


def combine_ref(Y, slotmap, sh):
    """The MoE combine's arithmetic: per row, the 6 slot rows of Y in slot order (fp32, missing slot = 0), then the
    shared-expert addend, one bf16 rounding."""
    acc = torch.zeros(slotmap.shape[0], Y.shape[1], dtype=torch.float32, device=Y.device)
    for k in range(slotmap.shape[1]):
        sl = slotmap[:, k].long()
        v = Y.index_select(0, sl.clamp_min(0)).float()
        acc = acc + torch.where((sl >= 0)[:, None], v, torch.zeros((), device=Y.device))
    return (acc + sh.float()).bfloat16()


def _slotmap(T, nv, R, g, d):
    """Distinct slot rows for the valid rows (owner q's rows [q R, q R + nv[q])), -1 for padding rows and for ~2 % of
    the valid rows' slots."""
    sm = torch.randperm(6 * T, generator=g, device=d).to(torch.int32).view(T, 6)
    lr = torch.arange(T, device=d) % R
    q = torch.arange(T, device=d) // R
    valid = lr < torch.tensor(nv, device=d)[q]
    drop = torch.rand(T, 6, generator=g, device=d) < 0.02
    return torch.where(valid[:, None] & ~drop, sm, torch.full_like(sm, -1)).contiguous()


@pytest.mark.parametrize("R", DECODE + MIXED)
def test_ag_push2(peers, comm, sync, same, R):
    """AG of the engine's two row sets (f rows of 10,240 B + routing rows of 64 B, one signal per CTA on the f flag)."""
    devs = peers
    gen = _gens(devs, R)
    gf = [torch.full((W * R, 2 * D), 0xA5, dtype=torch.uint8, device=d) for d in devs]
    gr = [torch.full((W * R, RB2), 0x5A, dtype=torch.uint8, device=d) for d in devs]
    fl = [torch.zeros(4, dtype=torch.int32, device=d) for d in devs]
    ep = [torch.zeros(1, dtype=torch.int32, device=d) for d in devs]
    for rnd in range(ROUNDS):
        f = [_u8((R, 2 * D), gen[i], d) for i, d in enumerate(devs)]
        rt = [_u8((R, RB2), gen[i], d) for i, d in enumerate(devs)]
        sync(devs)
        for r, d in enumerate(devs):
            with torch.cuda.device(d):
                comm.ag_push2(f[r], rt[r], r, gf, fl, gr, ep[r], NCTA)
        for r, d in enumerate(devs):
            with torch.cuda.device(d):
                comm.wait(fl[r], ep[r], W * NCTA)
        sync(devs)
        for q, d in enumerate(devs):
            for s in range(W):
                assert same(gf[q][s * R:(s + 1) * R], f[s].to(d)), f"round {rnd}: rank {q} slot {s} f rows differ"
                assert same(gr[q][s * R:(s + 1) * R], rt[s].to(d)), f"round {rnd}: rank {q} slot {s} routing rows differ"
            assert int(fl[q][0]) == (rnd + 1) * W * NCTA and int(ep[q][0]) == rnd + 1


@pytest.mark.parametrize("R", (32, 320, 4288))
def test_ag_push(peers, comm, sync, same, R):
    """AG of one row set (row bytes = f row + routing row)."""
    devs = peers
    RB = 2 * D + RB2
    gen = _gens(devs, 7000 + R)
    g = [torch.full((W * R, RB), 0xC3, dtype=torch.uint8, device=d) for d in devs]
    fl = [torch.zeros(4, dtype=torch.int32, device=d) for d in devs]
    ep = [torch.zeros(1, dtype=torch.int32, device=d) for d in devs]
    for rnd in range(ROUNDS):
        x = [_u8((R, RB), gen[i], d) for i, d in enumerate(devs)]
        sync(devs)
        for r, d in enumerate(devs):
            with torch.cuda.device(d):
                comm.ag_push(x[r], r, g, fl, ep[r], NCTA)
        for r, d in enumerate(devs):
            with torch.cuda.device(d):
                comm.wait(fl[r], ep[r], W * NCTA)
        sync(devs)
        for q, d in enumerate(devs):
            for s in range(W):
                assert same(g[q][s * R:(s + 1) * R], x[s].to(d)), f"round {rnd}: rank {q} slot {s} differs"


@pytest.mark.parametrize("R", DECODE + MIXED)
def test_rs_push_sum(peers, comm, sync, same, R):
    """RS of 4 R-row partials: receive slots == the peers' rows, output == torch's fixed-order fp32 sum, bit for bit."""
    devs = peers
    gen = _gens(devs, 100 + R)
    nan = float("nan")
    rs = [torch.full((W, R, D), nan, dtype=torch.bfloat16, device=d) for d in devs]
    out = [torch.full((R, D), nan, dtype=torch.bfloat16, device=d) for d in devs]
    fl = [torch.zeros(4, dtype=torch.int32, device=d) for d in devs]
    ep = [torch.zeros(1, dtype=torch.int32, device=d) for d in devs]
    for rnd in range(ROUNDS):
        part = [(torch.randn(W * R, D, generator=gen[i], device=d) * (i + 1)).bfloat16() for i, d in enumerate(devs)]
        sync(devs)
        for r, d in enumerate(devs):
            with torch.cuda.device(d):
                comm.rs_push(part[r], r, [x.view(torch.uint8) for x in rs], fl, ep[r], NCTA)
        for r, d in enumerate(devs):
            with torch.cuda.device(d):
                comm.rs_sum(rs[r], fl[r], ep[r], W * NCTA, out[r], 132)
        sync(devs)
        for q, d in enumerate(devs):
            mine = [part[s][q * R:(q + 1) * R].to(d) for s in range(W)]
            for s in range(W):
                assert same(rs[q][s], mine[s]), f"round {rnd}: owner {q} slot {s} differs"
            assert same(out[q], rs_ref(mine)), f"round {rnd}: owner {q} sum differs from the fixed-order reference"
            assert int(fl[q][0]) == (rnd + 1) * W * NCTA and int(ep[q][0]) == rnd + 1


@pytest.mark.parametrize("R", (32, 128, 320, 1024))
def test_combine_push(peers, comm, sync, same, R):
    """Fused MoE combine + RS push: owner q's slot r == the combine of rank r's rows [q R, (q + 1) R)."""
    devs = peers
    T, grid = W * R, 132
    gen = _gens(devs, 200 + R)
    nv = (R, R - 3, R // 2, 1)
    rs = [torch.full((W, R, D), float("nan"), dtype=torch.bfloat16, device=d) for d in devs]
    out = [torch.empty(R, D, dtype=torch.bfloat16, device=d) for d in devs]
    fl = [torch.zeros(4, dtype=torch.int32, device=d) for d in devs]
    ep = [torch.zeros(1, dtype=torch.int32, device=d) for d in devs]
    Y = [(torch.randn(6 * T, D, generator=gen[i], device=d)).bfloat16() for i, d in enumerate(devs)]
    sm = [_slotmap(T, nv, R, gen[i], d) for i, d in enumerate(devs)]
    sh = [(torch.randn(T, D, generator=gen[i], device=d) * 0.3).bfloat16() for i, d in enumerate(devs)]
    sync(devs)
    for r, d in enumerate(devs):
        with torch.cuda.device(d):
            comm.combine_push(Y[r], sm[r], sh[r], R, r, rs, fl, ep[r], grid)
    for r, d in enumerate(devs):
        with torch.cuda.device(d):
            comm.rs_sum(rs[r], fl[r], ep[r], W * grid, out[r], 132)
    sync(devs)
    parts = [combine_ref(Y[r], sm[r], sh[r]) for r in range(W)]
    for q, d in enumerate(devs):
        mine = [parts[s][q * R:(q + 1) * R].to(d) for s in range(W)]
        for s in range(W):
            assert same(rs[q][s], mine[s]), f"owner {q} slot {s}: not the combine's bits"
        assert same(out[q], rs_ref(mine))


@pytest.mark.parametrize("R", (32, 128, 320))
def test_decode_round_graph_replay(peers, comm, sync, same, R):
    """One decode round per rank captured in a CUDA graph (AG + wait, RS push + sum) and replayed: the epoch counters
    advance inside the graphs, so every replay must give exact outputs."""
    devs = peers
    gen = _gens(devs, 300 + R)
    f = [torch.empty(R, 2 * D, dtype=torch.uint8, device=d) for d in devs]
    rt = [torch.empty(R, RB2, dtype=torch.uint8, device=d) for d in devs]
    part = [torch.empty(W * R, D, dtype=torch.bfloat16, device=d) for d in devs]
    gf = [torch.zeros(W * R, 2 * D, dtype=torch.uint8, device=d) for d in devs]
    gr = [torch.zeros(W * R, RB2, dtype=torch.uint8, device=d) for d in devs]
    rs = [torch.zeros(W, R, D, dtype=torch.bfloat16, device=d) for d in devs]
    out = [torch.empty(R, D, dtype=torch.bfloat16, device=d) for d in devs]
    flf = [torch.zeros(4, dtype=torch.int32, device=d) for d in devs]
    epf = [torch.zeros(1, dtype=torch.int32, device=d) for d in devs]
    fls = [torch.zeros(4, dtype=torch.int32, device=d) for d in devs]
    eps = [torch.zeros(1, dtype=torch.int32, device=d) for d in devs]
    streams = [torch.cuda.Stream(device=d) for d in devs]
    graphs = []
    for r, d in enumerate(devs):
        with torch.cuda.device(d):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=streams[r]):
                comm.ag_push2(f[r], rt[r], r, gf, flf, gr, epf[r], NCTA)
                comm.wait(flf[r], epf[r], W * NCTA)
                comm.rs_push(part[r], r, [x.view(torch.uint8) for x in rs], fls, eps[r], NCTA)
                comm.rs_sum(rs[r], fls[r], eps[r], W * NCTA, out[r], 132)
            graphs.append(g)
    for rnd in range(3):
        for i, d in enumerate(devs):
            with torch.cuda.device(d):
                f[i].copy_(_u8((R, 2 * D), gen[i], d))
                rt[i].copy_(_u8((R, RB2), gen[i], d))
                part[i].copy_(torch.randn(W * R, D, generator=gen[i], device=d).bfloat16())
        sync(devs)
        for r, d in enumerate(devs):
            with torch.cuda.device(d):
                graphs[r].replay()
        sync(devs)
        for q, d in enumerate(devs):
            for s in range(W):
                assert same(gf[q][s * R:(s + 1) * R], f[s].to(d)) and same(gr[q][s * R:(s + 1) * R], rt[s].to(d))
            assert same(out[q], rs_ref([part[s][q * R:(q + 1) * R].to(d) for s in range(W)])), f"replay {rnd}: owner {q}"
            assert int(epf[q][0]) == rnd + 1 and int(eps[q][0]) == rnd + 1


SIZES = (((37, 584 * 64), torch.uint8), ((37, 132 * 64), torch.uint8), ((127, 8192), torch.uint8),
         ((1984, D), torch.bfloat16), ((1_000_003,), torch.uint8))


def test_peer_copy(peers, comm, sync, same):
    """peer_copy (one copy-engine copy on the sender's stream), every ordered pair of ranks, section-like shapes and an
    odd byte count; the receiver orders on an event only. A size mismatch is rejected."""
    devs = peers
    g = torch.Generator(device=devs[0]).manual_seed(5)
    for shape, dt in SIZES:
        n = 1
        for v in shape:
            n *= v
        base = torch.randint(0, 256, (n * torch.empty((), dtype=dt).element_size(),), dtype=torch.uint8, device=devs[0],
                             generator=g)
        for a, da in enumerate(devs):
            data = base.to(da).view(dt).view(shape)
            for b, db in enumerate(devs):
                if a == b:
                    continue
                with torch.cuda.device(da):
                    buf = torch.empty(shape, dtype=dt, device=db)
                    comm.peer_copy(buf, data)
                    ev = torch.cuda.Event()
                    ev.record()
                with torch.cuda.device(db):
                    torch.cuda.current_stream().wait_event(ev)
                    sync([db])
                    assert same(buf, data.to(db)), f"{tuple(shape)} {dt}: {a} -> {b} differs"
    with torch.cuda.device(devs[0]):
        with pytest.raises(RuntimeError):
            comm.peer_copy(torch.empty(10, dtype=torch.uint8, device=devs[1]), torch.empty(11, dtype=torch.uint8, device=devs[0]))
