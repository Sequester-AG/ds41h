"""hsdt_c1d: kernels/hsdtail's HSDT + the c1d graph variant for serving.

hsdt.py is live (c1sm5u) and frozen; this module adds only what serving needs around its existing c1d branch (hsdt.py
_steps: the 3 DSpark drafter blocks as ONE c1mega launch of 3 layers per GPU when exactly one rank holds drafting rows,
D_pad = 1, Rd = k; c1d.py's DrafterC1M). That branch's graph content depends on the step's drafting OWNER (the packs
carry the owner's DraftRank buffers and ring pointers), so:

- prepare(): after HSD's prepare, the same eligibility test as _steps; builds the owner's packs here (host H2D + device
  syncs: never inside a capture; once per (owner, DraftRank)) and sets self.cur.gvar = (("c1d", owner),), else ().
- bind_pools(): drops every c1d pack (the drafter rings move to the window pool's layers 40..42).

Engine side
- hsd_key() appends tuple(getattr(cur, "gvar", ())): split_hsd's graph H key (R, "H", key) then differs per owner, so a new
  owner gets one eager step + its own graph H (at most 4 per bucket);
- the ENG_HSDT hook imports os.environ.get("ENG_HSDT_MOD", "hsdt") (this module: ENG_HSDT_MOD=hsdt_c1d).
Needs split_hsd (the default): the one-graph decode path keys HSD by (R, in-graph) and would replay a stale variant.
"""
from __future__ import annotations

import os
import time

import hsdt as _H
from hsdt import parse_switches  # noqa: F401  (the engine hook's API: HSDT_.parse_switches)

W = _H.HS.W
# HSDT_C1D_SMAX (default 0 = no limit): c1d only on PURE-DECODE steps (no prefill chunk tails: NT == S_pad) with
# S_pad <= SMAX. S_pad counts sample ROWS (hsd.py: row bucket = sequence bucket x (k + 1)), so 4 = the C1 bucket at k = 3
# (at most one sequence per rank) -> exactly 4 c1d graph-H variants per server (owner x (S_pad 4, NT 4)), all captured
# by the warm-up owner sweep; every other step (incl. mixed steps, where a chunk's tail rows give NT 132, 260, ... next to
# the lone decoder) runs HSDT's stages, whose graph-H keys do not depend on the owner. SMAX < 4 turns c1d off everywhere.
SMAX = int(os.environ.get("HSDT_C1D_SMAX", "0"))


class HSDT(_H.HSDT):
    """HSDT + the per-step c1d graph variant (module docstring)."""

    def configure(self, **kw):
        super().configure(**kw)
        self._c1d_cfg = bool(self.sw.get("c1d"))   # the configured switch (prepare() may turn it off per step: SMAX)

    def prepare(self, batches, states, bitmasks=None):
        out = super().prepare(batches, states, bitmasks)
        self._c1d_variant()
        return out

    def _c1d_variant(self) -> None:
        c = getattr(self, "cur", None)
        if c is None:
            return
        gv = ()
        on = getattr(self, "_c1d_cfg", bool(self.sw.get("c1d"))) and getattr(self, "c1d", None) is not None
        nt = getattr(c, "NT", None)
        if on and SMAX and (c.S_pad > SMAX or (nt is not None and nt > c.S_pad)):   # NT > S_pad: a mixed step
            on = False
        if on:
            own_ = [x for x in range(W) if len(c.dseqs[x])]
            Dp = c.D_pad
            if Dp == 1 and len(own_) == 1 and self.draft_rank(own_[0], Dp).Rd == self.k:   # = hsdt.py _steps' test
                owner = own_[0]
                self.c1d.packs(owner, [self.draft_rank(x, Dp) for x in range(W)])
                gv = (("c1d", owner),)
                seen = self.__dict__.setdefault("_c1d_seen", set())
                vk = (owner, c.S_pad, getattr(c, "NT", None))   # graph H's key part beyond D_pad (= 1) and gvar
                if vk not in seen:   # = a new graph-H variant (one eager step + capture); warm-up should see all
                    seen.add(vk)
                    print(f"hsdt_c1d: {time.strftime('%H:%M:%S')} first c1d step for owner {owner}, S_pad {c.S_pad}, "
                          f"NT {vk[2]} ({len(seen)} variants)", flush=True)
        if SMAX:   # _steps (hsdt.py) reads sw["c1d"]: off for this step unless it is a c1d step (graph H keyed by gvar)
            self.sw["c1d"] = bool(gv)
        c.gvar = gv

    def bind_pools(self, window_pools):
        super().bind_pools(window_pools)
        for inst in self.__dict__.get("_c1d_insts", {}).values():
            inst._packs.clear()
