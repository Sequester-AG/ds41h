"""graph H at C1 = 606-632 us/step, of which the drafter's c1d k_layer is 323 us. c1d2 saved -45 us but was held on the power cap; this moves the drafter
to srv8m.

Kinds (wrappers next to this file, sources c1d_src3/ = a frozen srv8m copy, c1d_src3/SOURCE.md):
  s8m_nox2  c1d_s8m3_nox2.cu: srv8m's levers that fit the drafter's shape (NE 128, TOPK 3, WIN 131, TKA 0, C1M_INS_OLD), fp8 copies
       (C1M_DEP_B16 0, default dense fragments), whole-layer only. NOT applicable at this shape: CML, STF, RT1T, OWNSTG +
       UNIT_EARLY, FN_FRAG2 (see the wrapper's header).
Host differences vs c1d handled here, as srv8m's adapter (kernels/c1mega/srv8m/c1mega_eng.py) does them
  - scratch sizes: scr[3] x2 and fscr[1] x2 (C1M_LBM {value, epoch} partials), scr[4] x4/3 (C1S_RTS2 {logit, score}),
    scr[8] 6 x M x M x 608 fp16, scr[9] = X.cnt_n() counters;
  - Lamport fills: scr[4] = 0x7fc0dead when X.lamport() & 1 (C1M_RTL); the other bits must be 0 for this kind (asserted);
  - patches: patch_bf16 / patch_sh16 / patch_qb8 / patch_wo8 with null pointers (the fp8 copies' paths);
  - builds go to C1D3_BUILD.
pack_layer's signature is unchanged, so c1d.DrafterC1M.packs' argument lists are reused.
"""
from __future__ import annotations

import os

import torch

import c1d as _C

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("C1D3_BUILD", os.path.join(HERE, "build"))
KINDS3 = {"s8m_nox2": (os.path.join(HERE, "c1d_s8m3_nox2.cu"), ())}
for k_, v_ in KINDS3.items():
    _C.KINDS.setdefault(k_, v_)
NRM, D, HC = _C.NRM, _C.D, _C.HC


def ext3(src):
    from torch.utils.cpp_extension import load
    stem = os.path.splitext(os.path.basename(src))[0]
    bd = os.path.join(BUILD, "c1d3_" + stem)
    os.makedirs(bd, exist_ok=True)
    return load(name="c1d3_ext_" + stem, sources=[src], build_directory=bd, verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])


class DrafterC1M(_C.DrafterC1M):
    """c1d.DrafterC1M for the srv8m-based kinds (module docstring)."""

    def __init__(self, hsdt, rd, cfg, log=print, kind="s8m_nox2"):
        assert kind in KINDS3, kind
        self._kind3 = kind
        _ext0 = _C.ext
        _C.ext = lambda defs=(), src=None: ext3(KINDS3[kind][0])   # the private build (c1d.ext would build under hsdtail/build)
        try:
            super().__init__(hsdt, rd, cfg, log=log, kind=kind)
        finally:
            _C.ext = _ext0
        fc = list(self.X.fixed_config())
        assert fc[0] in (-1, 3) and fc[1] in (-1, 0), f"c1d3 {kind}: fixed_config {fc} (want amode 3, bf16 bits 0)"
        lam = int(getattr(self.X, "lamport", lambda: 1 if getattr(self.X, "rtl", lambda: False)() else 0)())
        assert lam & ~1 == 0, f"c1d3 {kind}: lamport bits {lam}: only C1M_RTL (1) is handled for the drafter"
        log(f"c1d3 ({kind}): fixed_config {fc}, lamport {lam}, counters {int(self.X.cnt_n())}, stg {int(self.X.stg_bytes())} B")

    def _buffers(self):
        """c1d's buffers with srv8m's scratch sizes (c1mega_eng.py C1Mega._buffers, M = NRM = 4)."""
        super()._buffers()
        bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
        M = NRM
        lam = int(getattr(self.X, "lamport", lambda: 1 if getattr(self.X, "rtl", lambda: False)() else 0)())
        for r, dev in enumerate(self.devs):
            B = self.B[r]
            with torch.cuda.device(dev):
                z = lambda *sh, dt=f32: torch.zeros(*sh, dtype=dt, device=dev)  # noqa: E731
                B["scr"] = [z(2 * M, 2048, dt=bf), z(M, D, dt=bf), z(132 * M), z(132 * M * 25 * 2), z(4 * M * 384),
                            z(72 * 32 * M), z(M, 576, dt=bf), z(36 * M), z(6 * M * M * 608, dt=torch.float16),
                            z(int(self.X.cnt_n()), dt=i32), z(132, dt=i32)]
                B["fscr"] = [z(132 * M), z(132 * M * 25 * 2), z(M * 4 * 20 * 16 * 512), z(M * 4 * 20 * 32)]
                B["qakv"] = z(2 * M, 1792, dt=bf)
                if lam & 1:   # C1M_RTL: logits polled against 0x7fc0dead
                    B["scr"][4].view(torch.int32).fill_(0x7fc0dead)
            torch.cuda.synchronize(dev)

    def packs(self, owner, dranks):
        """c1d.DrafterC1M.packs + srv8m's extra pointer patches (all null: fp8 copies) on every block's Args."""
        key = (owner, id(dranks[owner]))
        if key in self._packs:
            return self._packs[key]
        X = self.X
        pb0 = X.patch_bf16

        class _X:   # route c1d's calls through srv8m's extra patches (pack_layer / patch_integ / patch_stg unchanged)
            def __getattr__(s, n):
                return getattr(X, n)

            @staticmethod
            def patch_bf16(p_, *a):
                p_ = pb0(p_, 0, 0, 0)
                p_ = X.patch_sh16(p_, 0, 0)
                p_ = X.patch_qb8(p_, 0)
                if hasattr(X, "patch_wo8"):
                    p_ = X.patch_wo8(p_, 0, 0)
                return p_
        self.X = _X()
        try:
            out = super().packs(owner, dranks)
        finally:
            self.X = X
        return out
