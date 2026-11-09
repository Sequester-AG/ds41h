"""agrs: the mixed step's AG / RS options as patches on engine.Engine, behind module flags.

install(E) adds three boolean flags to the engine module (all False = the engine unchanged) and wraps p_ag / p_moe_prod /
p_rs. Each option applies only to buckets with R >= AGRS_MIN_R rows per rank (mixed-step buckets, eager); decode graphs
and the tail / drafter buckets keep the default path. Each option keeps its own flag / epoch words per bucket (comm.cu's
counters assume one signal count per round on a word), allocated on first use on all ranks with a device sync.

  E.AGRS_RSF     RS push fused into the production MoE's combine (pm_rs.cu k_pm_combine_push via agrs_pm.FusedRS;
                 with ENG_PM_NT = 1 / fast: pm2_rs.cu on moe_prefill's narrow-tile kernels via agrs_pmnt.FusedRSNT, so RSF
                 runs the same MoE kernels as the engine's pmf()):
                 when the PM_FUSED path runs (vcount > SMALL_T), the combine stores every row into its owner's RS
                 receive slot; p_rs then only runs comm.cu's k_rs_sum on this path's flag (W * AGRS_RSF_GRID signals).
  E.AGRS_SUM528  k_rs_sum with AGRS_SUM_GRID (528) CTAs instead of 132 (the same per-element arithmetic).
  E.AGRS_AG32    k_ag_push2 with AGRS_AG_NCTA (32) CTAs per destination instead of cfg.ncta (16), on this path's flag.
  E.AGRS_AGCE    the AG by copy engines (no SMs, no spin): p_router first records "f ready" and issues f's copies to the 4
                 gather slots on 4 copy streams per rank (overlapping the router), p_ag then copies the route rows behind
                 them and records one done event per destination; every receiver's p_moe waits on its 4 senders' done
                 events (cross-device stream waits; all sends are issued before any wait). Takes precedence over AG32.

Bit-identity: every option moves the same bytes or computes the same element with the same statements; checked by the
engine A/B (h_md5 / tok_md5 per step) in bar_ab.py.
"""
from __future__ import annotations

import os

import torch

W = 4
D = 5120
RSF_GRID = int(os.environ.get("AGRS_RSF_GRID", "528"))
SUM_GRID = int(os.environ.get("AGRS_SUM_GRID", "528"))
AG_NCTA = int(os.environ.get("AGRS_AG_NCTA", "32"))
MIN_R = int(os.environ.get("AGRS_MIN_R", "1024"))
_frs = {}
_ce = [None]
stats = dict(rsf=0, sum528=0, ag32=0, rsf_nt=0)


def nt_mode(E) -> str:
    """The production-MoE kernel set RSF must match: moe_prefill's ENG_PM_NT ("0" = pm_fused, "1" / "fast" = pmv.FusedV's
    narrow-tile snapshot). Engine module flag E.PM_NT; env AGRS_PMNT overrides it (test
    arms: ENV_AGRS_PMNT, read per call)."""
    m = os.environ.get("AGRS_PMNT")
    if m is None:
        m = str(getattr(E, "PM_NT", "0"))
    return m if m in ("1", "fast") else "0"


def frs(eng, mode: str = "0"):
    """RSF's MoE layer object: agrs_pm.FusedRS (pm_fused's kernels) or agrs_pmnt.FusedRSNT (FusedV's), one per mode."""
    if mode not in _frs:
        if mode == "0":
            import agrs_pm
            _frs[mode] = agrs_pm.FusedRS(eng.prod_moe())
        else:
            import agrs_pmnt
            _frs[mode] = agrs_pmnt.FusedRSNT(eng.prod_moe(), mode)
    return _frs[mode]


def ce_ext():
    if _ce[0] is None:
        import commbench
        _ce[0] = commbench.ext()      # agrs.cu: ce_copy = one cudaMemcpyAsync on a given stream
    return _ce[0]


def _cstreams(eng, r):
    cs = getattr(eng, "_agrs_cs", None)
    if cs is None:
        cs = eng._agrs_cs = [None] * W
    if cs[r] is None:
        cs[r] = [torch.cuda.Stream(device=eng.devs[r]) for _ in range(W)]
    return cs[r]


def _ensure(eng, st, key):
    """Flag word [4] + epoch [1] (int32, zero) named agrs_fl_<key> / agrs_ep_<key> in every rank's bucket state."""
    if f"agrs_fl_{key}" in st[0]:
        return
    for q, s in enumerate(st):
        with torch.cuda.device(eng.devs[q]):
            s[f"agrs_fl_{key}"] = torch.zeros(4, dtype=torch.int32, device=eng.devs[q])
            s[f"agrs_ep_{key}"] = torch.zeros(1, dtype=torch.int32, device=eng.devs[q])
    for d in eng.devs:
        torch.cuda.synchronize(d)    # zeroed and visible before any peer signals them


def install(E) -> None:
    if getattr(E.Engine, "_agrs_installed", False):
        return
    for k in ("AGRS_RSF", "AGRS_SUM528", "AGRS_AG32", "AGRS_AGCE"):   # defaults from the environment (ENG_AGRS_RSF=1 ...)
        if not hasattr(E, k):
            setattr(E, k, os.environ.get(f"ENG_{k}", "0") == "1")
    orig_ag, orig_moe_prod, orig_rs = E.Engine.p_ag, E.Engine.p_moe_prod, E.Engine.p_rs
    orig_router, orig_moe = E.Engine.p_router, E.Engine.p_moe

    def capturing():
        return torch.cuda.is_current_stream_capturing()

    def agce_on(s):
        return E.AGRS_AGCE and s["R"] >= MIN_R and not capturing()

    def p_router(self, r, s, L):
        st = getattr(self, "_agrs_st_last", None)
        if agce_on(s) and st is not None and st[r] is s:
            R = s["R"]
            cur = torch.cuda.current_stream()
            ev = torch.cuda.Event()
            ev.record(cur)                        # f ready (after the ffn boundary)
            cs = _cstreams(self, r)
            nb = R * D * 2
            ce = ce_ext()
            for q in range(W):
                cs[q].wait_event(ev)
                ce.ce_copy(st[q]["gf"], r * nb, s["f"], 0, nb, cs[q].cuda_stream)
            s["agrs_ce_issued"] = True
        return orig_router(self, r, s, L)

    def p_ag(self, r, s, L, st):
        self._agrs_st = st
        self._agrs_st_last = st
        if s.pop("agrs_ce_issued", False):
            R = s["R"]
            cur = torch.cuda.current_stream()
            ev = torch.cuda.Event()
            ev.record(cur)                        # route rows ready
            cs = _cstreams(self, r)
            nb = R * 64
            ce = ce_ext()
            for q in range(W):
                cs[q].wait_event(ev)
                ce.ce_copy(st[q]["gr"], r * nb, s["route_row"], 0, nb, cs[q].cuda_stream)
                d = torch.cuda.Event()
                d.record(cs[q])
                st[q].setdefault("agrs_in", []).append(d)
            stats["agce"] = stats.get("agce", 0) + 1
            return
        if E.AGRS_AG32 and s["R"] >= MIN_R and not capturing():
            _ensure(self, st, "ag")
            R = s["R"]
            self.cext.ag_push2(s["f"].view(torch.uint8).view(R, D * 2), s["route_row"], r, [x["gf"] for x in st],
                               [x["agrs_fl_ag"] for x in st], [x["gr"] for x in st], s["agrs_ep_ag"], AG_NCTA)
            self.cext.wait(s["agrs_fl_ag"], s["agrs_ep_ag"], W * AG_NCTA)
            stats["ag32"] += 1
            return
        return orig_ag(self, r, s, L, st)

    def p_moe(self, r, s, L):
        ins = s.pop("agrs_in", None)
        if ins:
            cur = torch.cuda.current_stream()
            for d in ins:
                cur.wait_event(d)                 # the 4 senders' gf / gr copies into this rank's slots
        return orig_moe(self, r, s, L)

    def p_moe_prod(self, r, s, L):
        st = getattr(self, "_agrs_st", None)
        if (E.AGRS_RSF and E.PM_FUSED and st is not None and st[r] is s and s["R"] >= MIN_R and not capturing()
                and s["vcount"] > self.prod_moe().SMALL_T
                and not (getattr(E, "PF_ACT8", False) and s.get("ndec", -1) >= 0)):   # the 8-bit arm keeps its own layer8 call
            _ensure(self, st, "rsf")
            lw = self.lw[r][L]
            Tall = W * s["R"]
            fall = s["gf"].view(torch.bfloat16).view(Tall, D)
            mode = nt_mode(E)
            stats["rsf_nt"] += mode != "0"
            frs(self, mode).layer_rs(fall, s["gr"], s["nv"], s["R"], s["vcount"], lw["w13"], lw["s13"], lw["w2"], lw["s2"],
                               lw["sw13"], lw["sw2"], s["gu"], s["hsh"], s["sh"], s["part"], [x["rs"] for x in st],
                               [x["agrs_fl_rsf"] for x in st], s["agrs_ep_rsf"], r, RSF_GRID, padded=False)
            s["agrs_rsf_pending"] = True
            stats["rsf"] += 1
            return
        return orig_moe_prod(self, r, s, L)

    def p_rs(self, r, s, L, st):
        grid = SUM_GRID if (E.AGRS_SUM528 and s["R"] >= MIN_R) else 132
        if s.pop("agrs_rsf_pending", False):
            self.cext.rs_sum(s["rs"], s["agrs_fl_rsf"], s["agrs_ep_rsf"], W * RSF_GRID, s["moe_out"], grid)
            stats["sum528"] += grid != 132
            return
        if grid != 132 and not capturing():
            self.cext.rs_push(s["part"], r, [x["rs"].view(torch.uint8) for x in st], [x["fl_s"] for x in st], s["ep_s"],
                              self.cfg.ncta)
            self.cext.rs_sum(s["rs"], s["fl_s"], s["ep_s"], W * self.cfg.ncta, s["moe_out"], grid)
            stats["sum528"] += 1
            return
        return orig_rs(self, r, s, L, st)

    E.Engine.p_ag, E.Engine.p_moe_prod, E.Engine.p_rs = p_ag, p_moe_prod, p_rs
    E.Engine.p_router, E.Engine.p_moe = p_router, p_moe
    E.Engine._agrs_installed = True
