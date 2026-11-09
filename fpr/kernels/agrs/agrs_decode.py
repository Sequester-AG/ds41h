"""agrs: decode RS fused into the fork MoE's combine, behind E.AGRS_RSD (default off).

Decode buckets (moe 'raw', graphs) with R >= AGRS_RSD_MIN_R rows per rank: the fork MoE's last call, e.combine(Y,
slotmap, sh, part), becomes agrs.cu's combine_push_dm (the same per-element statements as comm.cu k_combine_push =
moe_raw k_combine's bits; destination-major CTA groups), which stores every row straight into its owner's RS receive
slot; p_rs then only runs comm.cu's rs_sum on this path's own flag / epoch (W x AGRS_RSD_GRID signals per round).

Capture: the flag / epoch words are allocated in the bucket's first step, which is eager (engine: one eager step per
decode bucket, then capture), so the captured graph replays exactly the eager step's kernels. Under capture nothing
allocates or synchronizes. The drafter (HSD) MoE's combine calls are untouched (the patch is scoped to Engine.p_moe).
"""
from __future__ import annotations

import os
import threading

import torch

W = 4
GRID = int(os.environ.get("AGRS_RSD_GRID", "264"))
MIN_R = int(os.environ.get("AGRS_RSD_MIN_R", "128"))
_ctx = threading.local()
_ag = [None]


def ag_ext():
    if _ag[0] is None:
        import commbench
        _ag[0] = commbench.ext()
    return _ag[0]


def install(E) -> None:
    if getattr(E.Engine, "_agrs_rsd_installed", False):
        return
    if not hasattr(E, "AGRS_RSD"):
        E.AGRS_RSD = os.environ.get("ENG_AGRS_RSD", "0") == "1"
    orig_moe, orig_rs = E.Engine.p_moe, E.Engine.p_rs
    wrapped = set()

    def wrap_ext(e):
        if id(e) in wrapped:
            return
        oc = e.combine

        def combine(Y, slotmap, sh, part):
            c = getattr(_ctx, "c", None)
            if c is None:
                return oc(Y, slotmap, sh, part)
            r, s, st = c
            ag_ext().combine_push_dm(Y, slotmap, sh, s["R"], r, [x["rs"] for x in st],
                                     [x["agrs_fl_rsd"] for x in st], s["agrs_ep_rsd"], GRID)
            s["agrs_rsd_pending"] = True
        e.combine = combine
        wrapped.add(id(e))

    def on(s):     # MIN_R read per call (eager / capture only): test arms switch it with ENV_AGRS_RSD_MIN_R
        return E.AGRS_RSD and s.get("moe") == "raw" and s["R"] >= int(os.environ.get("AGRS_RSD_MIN_R", MIN_R))

    def p_moe(self, r, s, L):
        st = getattr(self, "_agrs_st", None)
        if on(s) and st is not None and st[r] is s:
            if "agrs_fl_rsd" not in st[0]:
                if torch.cuda.is_current_stream_capturing():
                    return orig_moe(self, r, s, L)   # never first-used under capture (the eager step allocates)
                for q, x in enumerate(st):
                    with torch.cuda.device(self.devs[q]):
                        x["agrs_fl_rsd"] = torch.zeros(4, dtype=torch.int32, device=self.devs[q])
                        x["agrs_ep_rsd"] = torch.zeros(1, dtype=torch.int32, device=self.devs[q])
                for d in self.devs:
                    torch.cuda.synchronize(d)
            wrap_ext(self.mext[r])
            _ctx.c = (r, s, st)
            try:
                return orig_moe(self, r, s, L)
            finally:
                _ctx.c = None
        return orig_moe(self, r, s, L)

    def p_rs(self, r, s, L, st):
        if s.pop("agrs_rsd_pending", False):
            self.cext.rs_sum(s["rs"], s["agrs_fl_rsd"], s["agrs_ep_rsd"], W * GRID, s["moe_out"], 132)
            return
        return orig_rs(self, r, s, L, st)

    E.Engine.p_moe, E.Engine.p_rs = p_moe, p_rs
    E.Engine._agrs_rsd_installed = True
