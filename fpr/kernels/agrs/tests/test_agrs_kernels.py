"""agrs kernels on one GPU: the fused combine + RS push of the decode fused path, the production-MoE layer with the RS push
fused into its combine, and the copy-engine copy of the AG fused path. All checks are bitwise.

Guards:
  - The served constants: the fused paths' grids and row thresholds are the ones the serving configuration was certified
    with (they are read from the environment at import; the test skips if any override is set).
  - combine_push_dm (decode RS fused path): every owner's receive slot `rank` holds exactly the decode MoE combine's rows
    (kernels/moe_stream moe_raw combine, comm.cu combine_push, and torch's statement of the arithmetic agree), other
    slots are untouched, every owner's flag gets `grid` signals per call and the epoch one bump.
  - The RS-fused production MoE layer (pm_rs.cu via agrs_pm.FusedRS): its plain layer == kernels/moe_prefill pm_fused's
    layer, and layer_rs stores exactly that layer's partial rows (padding rows included) into every owner's slot, for
    several grids and uneven valid-row layouts, with exact flag / epoch accounting.
  - ce_copy: byte-exact copies at arbitrary byte offsets, nothing outside the target range written.
The four "owners" are four buffers on the one GPU (the kernels address raw pointers); the NVLink path, and the fused paths as
the engine installs them, are test_agrs_levers.py.
"""
from __future__ import annotations

import os

import pytest
import torch

W, D, INTER = 4, 5120, 576
DECODE = (32, 64, 112, 120, 128, 192, 224, 240, 256, 320)


def test_served_constants():
    """Grids and thresholds as served: RSF 528 CTAs, RS sum 528, AG 32 per destination, mixed fused paths from 1,024 rows per
    rank; decode RS fusion 264 CTAs from 128 rows per rank."""
    if any(k.startswith("AGRS_") for k in os.environ):
        pytest.skip("an AGRS_* override is set in the environment")
    import agrs_decode
    import agrs_engine
    assert (agrs_engine.RSF_GRID, agrs_engine.SUM_GRID, agrs_engine.AG_NCTA, agrs_engine.MIN_R) == (528, 528, 32, 1024)
    assert (agrs_decode.GRID, agrs_decode.MIN_R) == (264, 128)


@pytest.mark.gpu
@pytest.mark.parametrize("R,grid", [(r, 264) for r in DECODE] + [(128, 132), (128, 528), (1024, 264)])
def test_combine_push_dm(agrs, comm, moe_raw, dev, sync, same, ref, R, grid):
    T, rank = W * R, 1
    g = torch.Generator(device=dev).manual_seed(R * 7 + grid)
    nv = (4, 0, 0, 0) if R == 32 else (R, R - 1, R // 2, 1)
    Y = torch.randn(6 * T, D, generator=g, device=dev).bfloat16()
    sm = ref.slotmap(T, nv, R, g, dev)
    sh = (torch.randn(T, D, generator=g, device=dev) * 0.3).bfloat16()
    nan = float("nan")
    bufs = [torch.full((W, R, D), nan, dtype=torch.bfloat16, device=dev) for _ in range(W)]
    flags = [torch.zeros(4, dtype=torch.int32, device=dev) for _ in range(W)]
    ep = torch.zeros(1, dtype=torch.int32, device=dev)
    agrs.combine_push_dm(Y, sm, sh, R, rank, bufs, flags, ep, grid)
    part = torch.full((T, D), nan, dtype=torch.bfloat16, device=dev)
    moe_raw.combine(Y, sm, sh, part)                                     # the decode MoE's own combine
    bufs2 = [torch.full((W, R, D), nan, dtype=torch.bfloat16, device=dev) for _ in range(W)]
    flags2 = [torch.zeros(4, dtype=torch.int32, device=dev) for _ in range(W)]
    comm.combine_push(Y, sm, sh, R, rank, bufs2, flags2, torch.zeros(1, dtype=torch.int32, device=dev), 132)
    sync([dev])
    want = ref.combine(Y, sm, sh)
    assert same(part, want), "moe_raw combine differs from the combine's arithmetic"
    for q in range(W):
        assert same(bufs[q][rank], part[q * R:(q + 1) * R]), f"owner {q}: slot {rank} is not the combine's rows"
        assert same(bufs2[q][rank], bufs[q][rank]), f"owner {q}: comm.cu combine_push differs"
        others = torch.cat([bufs[q][:rank], bufs[q][rank + 1:]])
        assert bool(others.isnan().all()), f"owner {q}: other slots were written"
        assert int(flags[q][0]) == grid, f"owner {q}: {int(flags[q][0])} signals, expected {grid}"
    assert int(ep[0]) == 1


@pytest.fixture(scope="module")
def rsf(PM):
    import agrs_pm
    from moe_prefill import pm_fused
    return pm_fused.Fused(PM), agrs_pm.FusedRS(PM)


LAYOUTS = ((1024, (1024, 1000, 1024, 960)), (1152, (1152, 900, 1100, 17)), (2048, (2048, 2047, 1500, 64)),
           (4288, (4260, 4260, 4252, 4252)))


@pytest.mark.gpu
@pytest.mark.parametrize("R,nv", LAYOUTS)
def test_rsf_layer(rsf, experts, dev, sync, same, ref, R, nv):
    base, mine = rsf
    ex = experts(dev, 0)
    T, vcount = W * R, sum(nv)
    g = torch.Generator(device=dev).manual_seed(R)
    fall = (torch.randn(T, D, generator=g, device=dev) * 0.6).bfloat16()
    gr = torch.cat([ref.route_rows(R, nv[q], g, dev) for q in range(W)])
    nvt = torch.tensor(nv, dtype=torch.int32, device=dev)
    gu = torch.empty(T, 2 * INTER, dtype=torch.bfloat16, device=dev)
    hsh = torch.empty(T, INTER, dtype=torch.bfloat16, device=dev)
    sh = torch.empty(T, D, dtype=torch.bfloat16, device=dev)
    args = (fall, gr, nvt, R, vcount, ex["w13"], ex["s13"], ex["w2"], ex["s2"], ex["sw13"], ex["sw2"], gu, hsh, sh)
    nan = float("nan")
    part = torch.full((T, D), nan, dtype=torch.bfloat16, device=dev)
    base.layer(*args, part, padded=False)
    part2 = torch.full((T, D), nan, dtype=torch.bfloat16, device=dev)
    mine.layer(*args, part2, padded=False)
    sync([dev])
    assert not bool(part.isnan().any()), "reference layer left rows unwritten"
    assert same(part2, part), "agrs build of the layer differs from pm_fused"
    rank = 2
    flags = [torch.zeros(4, dtype=torch.int32, device=dev) for _ in range(W)]
    ep = torch.zeros(1, dtype=torch.int32, device=dev)
    fsum = 0
    for calls, G in enumerate((264, 528, 1056), start=1):
        bufs = [torch.full((W, R, D), nan, dtype=torch.bfloat16, device=dev) for _ in range(W)]
        mine.layer_rs(*args, part2, bufs, flags, ep, rank, G, padded=False)
        fsum += G
        sync([dev])
        for q in range(W):
            assert same(bufs[q][rank], part[q * R:(q + 1) * R]), f"grid {G}: owner {q} slot {rank} differs"
            assert int(flags[q][0]) == fsum, f"grid {G}: owner {q} flag {int(flags[q][0])}, expected {fsum}"
        assert int(ep[0]) == calls


@pytest.mark.gpu
def test_ce_copy(agrs, dev, sync, same):
    g = torch.Generator(device=dev).manual_seed(3)
    st = torch.cuda.Stream(device=dev)
    for off, soff, n in ((0, 0, 1 << 20), (5, 3, 999_983), (4096, 64, 4288 * 64), (1, 0, 1)):
        src = torch.randint(0, 256, (soff + n + 7,), dtype=torch.uint8, device=dev, generator=g)
        dst = torch.full((off + n + 11,), 0x11, dtype=torch.uint8, device=dev)
        sync([dev])
        agrs.ce_copy(dst, off, src, soff, n, st.cuda_stream)
        sync([st])
        assert same(dst[off:off + n], src[soff:soff + n])
        assert bool((dst[:off] == 0x11).all()) and bool((dst[off + n:] == 0x11).all())
