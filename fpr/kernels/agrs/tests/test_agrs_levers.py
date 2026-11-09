"""The AG / RS fused paths as the serving engine installs them, on 4 GPUs: per-element equivalence to the engine's comm.cu path.

agrs_engine.install and agrs_decode.install patch the engine's per-layer pieces (p_router, p_ag, p_moe, p_moe_prod,
p_rs). Here they patch a stub engine whose pieces are the engine's default calls verbatim: comm.cu ag_push2 + wait, the
decode MoE's combine (moe_raw) or the production MoE layer (pm_fused), comm.cu rs_push + rs_sum with 16 CTAs per
destination and 132 sum CTAs. Each layer runs twice on identical inputs, fused paths on (the served set) and off, issued as
the engine issues a step (each piece for every rank before the next piece), and the results are compared bit for bit.

Guards (the served set: RSF + SUM528 + AGCE on mixed steps, RSD on decode buckets):
  - Mixed steps (R >= 1,024 rows per rank, valid rows above production's small-call threshold): the gathered f and
    routing rows (AGCE: copy engines from the second layer on; the first layer has no previous layer's state and keeps
    the SM gather), the owners' receive slots (RSF: pushed from inside the production MoE's combine) and the RS output
    (SUM528: 528 sum CTAs) equal the default path's, and the output equals torch's fixed-order sum of the default
    path's partials.
  - The thresholds: below 1,024 rows nothing engages; at production's small-call threshold RSF stays off while SUM528
    and AGCE still engage.
  - Decode buckets: RSD engages from 128 rows per rank exactly; slots and output equal the default path's; the default
    path's partial equals torch's statement of the combine.
  - Decode graphs: a decode round captured per rank (AG, MoE combine, RS) replays with exact outputs, the fused path's flags
    allocated by the eager first step as the engine does.
Routing rows have 6 distinct expert ids per row (sampled without replacement); nothing under test selects, so no ties.
"""
from __future__ import annotations

import types

import pytest
import torch

pytestmark = pytest.mark.gpu

W, D, INTER, NCTA = 4, 5120, 576, 16
DECODE = (32, 64, 112, 120, 128, 192, 224, 240, 256, 320)
SERVED = dict(AGRS_RSF=True, AGRS_SUM528=True, AGRS_AGCE=True, AGRS_AG32=False, AGRS_RSD=True)


def _stub_module():
    E = types.ModuleType("stub_engine")
    for k in SERVED:
        setattr(E, k, False)
    E.PM_FUSED, E.PF_ACT8, E.PM_NT = True, False, "0"

    class Engine:
        """The engine's default calls of one layer's exchange pieces (engine.py p_ag, p_moe, p_moe_prod, p_rs)."""

        def __init__(self, devs, cext, mext=None, PM=None):
            self.devs, self.cext, self.mext, self._pm = devs, cext, mext, PM
            self.cfg = types.SimpleNamespace(ncta=NCTA)
            self.lw = [dict() for _ in devs]
            self._pmf = None

        def prod_moe(self):
            return self._pm

        def pmf(self):
            if self._pmf is None:
                from moe_prefill import pm_fused
                self._pmf = pm_fused.Fused(self.prod_moe())
            return self._pmf

        def p_router(self, r, s, L):
            pass                                    # the routing rows are the test's inputs

        def p_ag(self, r, s, L, st):
            R = s["R"]
            self.cext.ag_push2(s["f"].view(torch.uint8).view(R, D * 2), s["route_row"], r, [x["gf"] for x in st],
                               [x["fl_f"] for x in st], [x["gr"] for x in st], s["ep_f"], self.cfg.ncta)
            self.cext.wait(s["fl_f"], s["ep_f"], W * self.cfg.ncta)

        def p_moe(self, r, s, L):
            if s["moe"] == "prod":
                return self.p_moe_prod(r, s, L)
            self.mext[r].combine(s["Y"], s["slotmap"], s["sh"], s["part"])   # the decode MoE's last call

        def p_moe_prod(self, r, s, L):
            lw = self.lw[r][L]
            fall = s["gf"].view(torch.bfloat16).view(W * s["R"], D)
            self.pmf().layer(fall, s["gr"], s["nv"], s["R"], s["vcount"], lw["w13"], lw["s13"], lw["w2"], lw["s2"],
                             lw["sw13"], lw["sw2"], s["gu"], s["hsh"], s["sh"], s["part"], padded=False)

        def p_rs(self, r, s, L, st):
            self.cext.rs_push(s["part"], r, [x["rs"].view(torch.uint8) for x in st], [x["fl_s"] for x in st], s["ep_s"],
                              self.cfg.ncta)
            self.cext.rs_sum(s["rs"], s["fl_s"], s["ep_s"], W * self.cfg.ncta, s["moe_out"], 132)

    E.Engine = Engine
    return E


@pytest.fixture(scope="module")
def stub(peers):
    import agrs_decode
    import agrs_engine
    E = _stub_module()
    agrs_engine.install(E)                          # the engine's order of installation
    agrs_decode.install(E)
    return types.SimpleNamespace(E=E, eng=agrs_engine, dec=agrs_decode)


def _levers(E, on):
    for k, v in SERVED.items():
        setattr(E, k, v if on else False)


def _layer(eng, st, L):
    """One layer's pieces, each issued for every rank before the next (the engine's issue order)."""
    for name in ("p_router", "p_ag", "p_moe", "p_rs"):
        for r, d in enumerate(eng.devs):
            with torch.cuda.device(d):
                fn = getattr(eng, name)
                if name in ("p_ag", "p_rs"):
                    fn(r, st[r], L, st)
                else:
                    fn(r, st[r], L)


def _state(d, R, moe, nv=None):
    T = W * R

    def e(*shape, dt=torch.bfloat16):
        return torch.empty(shape, dtype=dt, device=d)

    def z(n):
        return torch.zeros(n, dtype=torch.int32, device=d)
    s = dict(R=R, moe=moe, f=e(R, D), route_row=e(R, 64, dt=torch.uint8), gf=e(T, 2 * D, dt=torch.uint8),
             gr=e(T, 64, dt=torch.uint8), fl_f=z(4), ep_f=z(1), sh=e(T, D), part=e(T, D), rs=e(W, R, D), fl_s=z(4),
             ep_s=z(1), moe_out=e(R, D))
    if moe == "prod":
        s.update(nv=torch.tensor(nv, dtype=torch.int32, device=d), vcount=sum(nv), gu=e(T, 2 * INTER), hsh=e(T, INTER))
    else:
        s.update(Y=e(6 * T, D), slotmap=e(T, 6, dt=torch.int32))
    return s


def _poison(st):
    for s in st:
        s["gf"].fill_(0xEE)
        s["gr"].fill_(0xEE)
        for k in ("part", "rs", "moe_out"):
            s[k].fill_(float("nan"))


def _compare(devs, on, off, same, ref, L):
    for q, d in enumerate(devs):
        for k in ("gf", "gr"):
            assert same(on[q][k], off[q][k]), f"layer {L} rank {q}: gathered {k} differs"
        for s in range(W):
            assert same(on[q]["rs"][s], off[q]["rs"][s]), f"layer {L} owner {q}: receive slot {s} differs"
        assert same(on[q]["moe_out"], off[q]["moe_out"]), f"layer {L} owner {q}: RS output differs"
        R = off[q]["R"]
        want = ref.rs([off[s]["part"][q * R:(q + 1) * R].to(d) for s in range(W)])
        assert same(off[q]["moe_out"], want), f"layer {L} owner {q}: default RS output != fixed-order sum"


def _mixed_run(stub, devs, comm, PM, experts, sync, same, ref, R, nv, seed):
    E, A = stub.E, stub.eng
    on, off = E.Engine(devs, comm, PM=PM), E.Engine(devs, comm, PM=PM)
    for eng in (on, off):
        for r, d in enumerate(devs):
            ex = experts(d, r)
            eng.lw[r] = {0: ex, 1: ex}
    st_on = [_state(d, R, "prod", nv) for d in devs]
    st_off = [_state(d, R, "prod", nv) for d in devs]
    gens = [torch.Generator(device=d).manual_seed(seed * 8 + i) for i, d in enumerate(devs)]
    s0 = dict(A.stats)
    for L in (0, 1):
        for r, d in enumerate(devs):
            f = (torch.randn(R, D, generator=gens[r], device=d) * 0.6).bfloat16()
            rt = ref.route_rows(R, nv[r], gens[r], d)
            for st in (st_on, st_off):
                st[r]["f"].copy_(f)
                st[r]["route_row"].copy_(rt)
        _poison(st_on)
        _poison(st_off)
        sync(devs)
        _levers(E, True)
        try:
            _layer(on, st_on, L)
            sync(devs)
        finally:
            _levers(E, False)
        _layer(off, st_off, L)
        sync(devs)
        _compare(devs, st_on, st_off, same, ref, L)
    return {k: A.stats.get(k, 0) - s0.get(k, 0) for k in ("rsf", "sum528", "agce", "ag32")}, st_on


@pytest.mark.parametrize("R,nv", ((1024, (1024, 1000, 1024, 960)), (2048, (2048, 2047, 1500, 64)),
                                  (4288, (4260, 4260, 4252, 4252))))
def test_mixed_levers(stub, peers, comm, PM, experts, sync, same, ref, R, nv):
    """RSF + SUM528 + AGCE on two consecutive layers == the default path, bit for bit."""
    n, st_on = _mixed_run(stub, peers, comm, PM, experts, sync, same, ref, R, nv, R)
    assert n == dict(rsf=2 * W, sum528=2 * W, agce=W, ag32=0), n
    assert all(int(s["ep_f"][0]) == 1 for s in st_on), "the second layer's AG did not run on the copy engines"


def test_mixed_thresholds(stub, peers, comm, PM, experts, sync, same, ref):
    """Below 1,024 rows per rank no fused path engages; at production's small-call threshold (valid rows == SMALL_T) RSF
    stays off, SUM528 and AGCE engage; outputs equal the default path in both cases."""
    n, _ = _mixed_run(stub, peers, comm, PM, experts, sync, same, ref, 960, (960, 900, 960, 7), 1)
    assert n == dict(rsf=0, sum528=0, agce=0, ag32=0), n
    small = int(PM.SMALL_T)
    nv = (small // 4,) * 4
    assert sum(nv) == small
    n, _ = _mixed_run(stub, peers, comm, PM, experts, sync, same, ref, 1024, nv, 2)
    assert n == dict(rsf=0, sum528=2 * W, agce=W, ag32=0), n


def _decode_inputs(st_list, R, nv, gens, devs, ref):
    T = W * R
    for r, d in enumerate(devs):
        g = gens[r]
        vals = dict(f=torch.randn(R, D, generator=g, device=d).bfloat16(),
                    route_row=torch.randint(0, 256, (R, 64), dtype=torch.uint8, device=d, generator=g),
                    Y=torch.randn(6 * T, D, generator=g, device=d).bfloat16(), slotmap=ref.slotmap(T, nv, R, g, d),
                    sh=(torch.randn(T, D, generator=g, device=d) * 0.3).bfloat16())
        for st in st_list:
            for k, v in vals.items():
                st[r][k].copy_(v)


def _decode_nv(R):
    return (4, 0, 0, 0) if R == 32 else (R, R - 1, R // 2, 1)       # 32: one user (4 rows on one rank)


@pytest.mark.parametrize("R", DECODE)
def test_decode_rsd(stub, peers, comm, moe_raw, sync, same, ref, R):
    """RSD on decode buckets (eager): slots and output == the default path; engages exactly from 128 rows per rank."""
    E, devs = stub.E, peers
    on, off = E.Engine(devs, comm, mext=[moe_raw] * W), E.Engine(devs, comm, mext=[moe_raw] * W)
    st_on = [_state(d, R, "raw") for d in devs]
    st_off = [_state(d, R, "raw") for d in devs]
    gens = [torch.Generator(device=d).manual_seed(900 + R * 4 + i) for i, d in enumerate(devs)]
    nv = _decode_nv(R)
    for L in (0, 1):
        _decode_inputs((st_on, st_off), R, nv, gens, devs, ref)
        _poison(st_on)
        _poison(st_off)
        sync(devs)
        _levers(E, True)
        try:
            _layer(on, st_on, L)
            sync(devs)
        finally:
            _levers(E, False)
        _layer(off, st_off, L)
        sync(devs)
        _compare(devs, st_on, st_off, same, ref, L)
        for r in range(W):
            want = ref.combine(st_off[r]["Y"], st_off[r]["slotmap"], st_off[r]["sh"])
            assert same(st_off[r]["part"], want), f"rank {r}: the default combine differs from its arithmetic"
    assert ("agrs_fl_rsd" in st_on[0]) == (R >= 128), "RSD engaged at the wrong bucket size"


@pytest.mark.parametrize("R", (32, 128, 320))
def test_decode_rsd_graphs(stub, peers, comm, moe_raw, sync, same, ref, R):
    """A decode round (AG, MoE combine, RS) captured per rank with the fused paths on, replayed 3 times on new inputs."""
    E, devs = stub.E, peers
    on, off = E.Engine(devs, comm, mext=[moe_raw] * W), E.Engine(devs, comm, mext=[moe_raw] * W)
    st_on = [_state(d, R, "raw") for d in devs]
    st_off = [_state(d, R, "raw") for d in devs]
    gens = [torch.Generator(device=d).manual_seed(1700 + R * 4 + i) for i, d in enumerate(devs)]
    nv = _decode_nv(R)
    streams = [torch.cuda.Stream(device=d) for d in devs]
    _decode_inputs((st_on,), R, nv, gens, devs, ref)
    sync(devs)
    graphs = []
    _levers(E, True)
    try:
        _layer(on, st_on, 0)                        # the bucket's eager first step (allocates the fused path's flags)
        sync(devs)
        for r, d in enumerate(devs):
            with torch.cuda.device(d):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=streams[r]):
                    on.p_router(r, st_on[r], 0)
                    on.p_ag(r, st_on[r], 0, st_on)
                    on.p_moe(r, st_on[r], 0)
                    on.p_rs(r, st_on[r], 0, st_on)
                graphs.append(g)
    finally:
        _levers(E, False)
    for rnd in range(3):
        _decode_inputs((st_on, st_off), R, nv, gens, devs, ref)
        _poison(st_on)
        _poison(st_off)
        sync(devs)
        for r, d in enumerate(devs):
            with torch.cuda.device(d):
                graphs[r].replay()
        sync(devs)
        _layer(off, st_off, 0)
        sync(devs)
        _compare(devs, st_on, st_off, same, ref, f"replay {rnd}")
    assert ("agrs_fl_rsd" in st_on[0]) == (R >= 128)
