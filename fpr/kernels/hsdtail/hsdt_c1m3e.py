"""hsdt_c1m3e: the SERVED hsdt_c1e's HSDT + the c1d3 kinds
(c1d3.DrafterC1M: the drafter on srv8m's layer code). Engine side: ENG_HSDT_MOD=hsdt_c1m3e ENG_HSDT=<the served list with
c1d=s8m_nox2> (fpr/kernels/hsdtail on the path, as every hsdt module). Everything but the drafter kind is hsdt_c1e's (no hsdt_c1f / c1g switches)."""
from __future__ import annotations

import hsdt as _H
import hsdt_c1e as _E

C1D3_KINDS = ("s8m_nox2",)


class HSDT(_E.HSDT):
    def configure(self, c1d=None, **kw):
        if isinstance(c1d, str) and c1d in C1D3_KINDS:
            import c1d3 as _D3
            insts_ = self.__dict__.setdefault("_c1d_insts", {})
            if c1d not in insts_:
                from types import SimpleNamespace as NS_
                d0 = self.ranks0[0]
                _H.mhcf_ext()
                insts_[c1d] = _D3.DrafterC1M(self, self.rd, NS_(norm_eps=d0.eps, hc_eps=d0.hc_eps, hc_iters=d0.iters), kind=c1d)
        super().configure(c1d=c1d, **kw)


def parse_switches(spec):
    items = [x.strip() for x in spec.replace("+", ",").split(",")]
    kind = next((x.split("=", 1)[1] for x in items if x.startswith("c1d=")), None)
    out = _E.parse_switches(",".join(x for x in items if x and not x.startswith("c1d=")))
    if kind is not None:
        assert kind in C1D3_KINDS, f"ENG_HSDT c1d={kind}: not a c1d3 kind {C1D3_KINDS}"
        out["c1d"] = kind
    return out


def __getattr__(name):
    return getattr(_E, name)
