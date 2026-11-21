"""hsdt_c1e: hsdt_c1d's HSDT + ctx3, the drafter's context KV of its 3 blocks batched.

hsdt.py and hsdt_c1d.py are live (c1sm5uwc2) and frozen; this module only adds a switch on top of hsdt_c1d.HSDT.

ctx3: the context KV chain (drafter.py DraftRank.context_kv, and hsdt.py's ctxtp phase 3, DraftRankT.ctp_tail) runs, per
drafter block, wkv GEMM -> (split-K reduce) -> norm_rope -> rope_q_insert, three times in series on the context side
stream. With c1d + ctxtp + dhead (t-56, allucth) that chain is graph H's critical path into the c1d launch (it ends
~30 us after the main stream's x0 glue). ctx3 runs ONE GEMM over the 3 blocks' wkv weights (a [3 x 512, 5,120] bf16 copy
per GPU, 15.7 MB) and then each block's norm_rope (on its column slice: the kernel takes the row stride) and its insert,
unchanged. Same math for the drafter (the batched GEMM may reorder the fp32 sums); the target path is untouched.
Only for context rows <= CTX3_MAX (C1 / small decode steps; larger steps keep the per-block chain).
ctx3f (implies ctx3): the 3 blocks' norm_rope + insert as ONE launch (ctx3.cu: dense.cu's device code verbatim; the q
rows' RoPE, a no-op on the context path's zero dq rows, skipped) -> fpr/kernels/hsdtail/build/ctx3. Bitwise vs ctx3.

Engine side: ENG_HSDT_MOD=hsdt_c1e (engine.py's hook imports HSDT and parse_switches from this module); the switch is
"ctx3" in ENG_HSDT. Everything else = hsdt_c1d (c1d graph variant, HSDT_C1D_SMAX).
"""
from __future__ import annotations

import os

import torch

import hsdt as _H
import hsdt_c1d as _C

DR = _H.DR
HAS_CTX3 = True                    # the bench passes ctx3=False per variant when the module has it
CTX3_MAX = int(os.environ.get("HSDT_CTX3_MAX", "128"))
_W3 = {}                           # device index -> (W3 [3 x dkv, H] bf16, kv3 scratch [CTX3_MAX, 3 x dkv] bf16, out0/1)
HERE = os.path.dirname(os.path.abspath(__file__))
_CTX3 = None


def ctx3_ext():
    global _CTX3
    if _CTX3 is None:
        from torch.utils.cpp_extension import load
        bd = os.path.join(HERE, "build", "ctx3")
        os.makedirs(bd, exist_ok=True)
        _CTX3 = load(name="hsdt_ctx3", sources=[os.path.join(HERE, "ctx3.cu")], build_directory=bd, verbose=False,
                     extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])
    return _CTX3


def parse_switches(spec):
    items = [x.strip() for x in spec.replace("+", ",").split(",")]
    rest = ",".join(x for x in items if x and x not in ("ctx3", "ctx3f"))
    out = _H.parse_switches(rest)
    if "ctx3f" in items:
        out["ctx3"] = "fused"
    elif "ctx3" in items:
        out["ctx3"] = True
    return out


class DraftRankT3(_H.DraftRankT):
    """DraftRankT + ctx3 (self.ctx3_on). Class conversion of HSDT's DraftRanks (adds state only)."""

    def enable_ctx3(self, on=True):
        self.ctx3_on = bool(on)
        self.ctx3_fused = on == "fused"
        if not on:
            return
        if self.ctx3_fused:
            ctx3_ext()
        dv = self.blocks[0]["wqkv_a"].device
        key = dv.index
        if key not in _W3:
            with torch.cuda.device(dv):
                w3 = torch.cat([B["wqkv_a"][1280:] for B in self.blocks], 0).contiguous()
                kv3 = torch.zeros(CTX3_MAX, w3.shape[0], dtype=w3.dtype, device=w3.device)
                o01 = [torch.zeros(CTX3_MAX, w3.shape[0] // 3, dtype=w3.dtype, device=w3.device) for _ in range(2)]
            _W3[key] = (w3, kv3, o01)
        self._w3, self._kv3, self._o01 = _W3[key]
        self._dkv = self._w3.shape[0] // len(self.blocks)

    def _ctx3_kv(self, R):
        """The 3 blocks' context KV from b["main_x"][:R]: one GEMM, then per block norm_rope + insert (as context_kv)."""
        b, kd, D = self.b, self.kd, self._dkv
        kv3 = self._kv3[:R]
        torch.mm(b["main_x"][:R], self._w3.t(), out=kv3)
        if self.ctx3_fused:   # block 2's rows land in kv_ctx (its final content as in the per-block chain)
            _CTX3.norm_insert3(kv3, [B["kv_norm"] for B in self.blocks], [self._o01[0], self._o01[1], b["kv_ctx"]],
                               b["ctx_pos"][:R], self.cos_sin, b["ctx_slot"][:R], list(self.cache), self.page_stride,
                               DR.PAGE_SHIFT, self.eps)
            return
        for i, B in enumerate(self.blocks):
            kd.norm_rope(kv3[:, i * D:(i + 1) * D], B["kv_norm"], b["kv_ctx"][:R], self.eps, b["ctx_pos"][:R], self.cos_sin)
            kd.rope_q_insert(b["dq"][:R], b["kv_ctx"][:R], b["ctx_pos"][:R], self.cos_sin, b["ctx_slot"][:R], self.cache[i],
                             self.page_stride, DR.PAGE_SHIFT, True)

    def context_kv(self, rows=None):
        R = rows or self.R
        if not getattr(self, "ctx3_on", False) or R > CTX3_MAX:
            return super().context_kv(rows)
        b, kd = self.b, self.kd
        torch.mm(b["aux"][:R], self.main_proj.t(), out=b["mx_raw"][:R])   # = drafter.py context_kv's first two ops
        kd.norm_rope(b["mx_raw"][:R], self.main_norm, b["main_x"][:R], self.eps, None, None)
        self._ctx3_kv(R)

    def ctp_tail(self, H_, r, R):
        if not getattr(self, "ctx3_on", False) or R > CTX3_MAX:
            return super().ctp_tail(H_, r, R)
        G = H_.ctp_g   # = hsdt.py DraftRankT.ctp_tail up to the per-block loop
        b, kd = self.b, self.kd
        fl2, ep2 = G.sites["mxp"]
        G.e.wait(fl2[r], ep2[r], DR.W * G.ncta["mxp"])
        rv = H_.ctp_recv[r][: DR.W * DR.W * R].view(DR.W, DR.W, R, _H.CTP_COLS)[:, r]
        b["mx_raw"][:R].view(R, DR.W, _H.CTP_COLS).copy_(rv.transpose(0, 1))
        kd.norm_rope(b["mx_raw"][:R], self.main_norm, b["main_x"][:R], self.eps, None, None)
        self._ctx3_kv(R)


def as_t3(d, on):
    _H.as_t(d)
    if not isinstance(d, DraftRankT3):
        d.__class__ = DraftRankT3
    d.enable_ctx3(on)
    return d


class HSDT(_C.HSDT):
    """hsdt_c1d.HSDT + ctx3 (module docstring)."""

    def configure(self, ctx3=None, **kw):
        super().configure(**kw)
        if ctx3 is not None:
            self._ctx3 = ctx3 if ctx3 == "fused" else bool(ctx3)
            self.sw["ctx3"] = self._ctx3
            for d in list(self.ranks0) + list(self.drk.values()):
                as_t3(d, self._ctx3)

    def draft_rank(self, r, D_pad):
        d = super().draft_rank(r, D_pad)
        if getattr(self, "_ctx3", False) and not getattr(d, "ctx3_on", False):
            as_t3(d, self._ctx3)
        return d
