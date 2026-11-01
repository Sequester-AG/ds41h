"""Lossless 12-bit verification lm_head. See ds41h/kernels/vhead.cu.

The target's bf16 lm_head shard is repacked once at load (s|m byte + 4-bit exponent code per weight, exact escape list):
248 MB instead of 331 MB per TP rank, decoded bit-exactly inside the GEMV (HMMA, fp32 accumulation). The verification
logits keep stock numerics up to the fp32 summation order (tests/test_vhead.py: unpack == W bitwise; GEMV vs cuBLAS / fp64).

Switches (vp_sampling.verify_logits): DS41H_VERIFY_HEAD=2 uses this kernel for <= 8 verified rows (1 = the older plain bf16
dvh kernel in drafter.cu; 3 = the plain bf16 row-range GEMV vrh in kernels/vhead.cu: no extra memory, contiguous row bursts,
deterministic). DS41H_VHEAD_VARIANT picks the packed launch shape (0 = 512 threads x 4-kb chunks). DS41H_VHEAD_CTL=<json>
(debug / in-situ A/B on one server): {"mode": "stock" | "packed" | "alt"} re-read when the file changes; "alt" alternates
stock and packed per call so one profile trace times both under the same clock / power state.
"""

from __future__ import annotations

import functools
import json
import logging
import os

import torch

logger = logging.getLogger("ds41h.vhead")

VARIANT = int(os.environ.get("DS41H_VHEAD_VARIANT", "0") or 0)
# DS41H_VERIFY_HEAD=3: plain bf16 row-range GEMV (kernels/vhead.cu vrh), launch shape DS41H_VHEAD_ROWS_VARIANT
# (0: 1024-wide k stages x 6, 1: 1024 x 5, 2: 512 x 10, 3: 1280 x 5, 4: 512 x 12, 5: 1280 x 4)
ROWS_VARIANT = int(os.environ.get("DS41H_VHEAD_ROWS_VARIANT", "0") or 0)
CTL = os.environ.get("DS41H_VHEAD_CTL", "")
_CTL = {"mtime": None, "mode": "packed", "n": 0}


@functools.cache
def ext():
    from ds41h import cuext

    mod = cuext.load("ds41h_vhead", "vhead.cu")
    mod.preload()
    return mod


class PackedHead:
    """One rank's packed [n, 5120] bf16 head (n % 16 == 0)."""

    def __init__(self, w: torch.Tensor):
        assert w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[1] == 5120 and w.shape[0] % 16 == 0
        self.n = int(w.shape[0])
        self.sm, self.code, self.off, self.esc = ext().pack(w)
        self.nesc = int(self.esc.shape[0])
        self.bytes = self.sm.numel() + self.code.numel() + self.off.numel() * 4 + self.esc.numel() * 4

    def __call__(self, x: torch.Tensor, idx: torch.Tensor | None, m: int, pdl: bool = True, variant: int | None = None) -> torch.Tensor:
        y = torch.empty(m, self.n, dtype=torch.bfloat16, device=x.device)
        ext().gemv(x, idx, self.sm, self.code, self.off, self.esc, y, pdl, VARIANT if variant is None else int(variant))
        return y

    def unpack(self) -> torch.Tensor:
        """bf16 [n, 5120] with the escapes restored (== the source weight, bit for bit)."""
        w = ext().unpack(self.sm, self.code)
        if self.nesc:
            e = self.esc.long()
            tiles = torch.repeat_interleave(torch.arange(self.off.numel() - 1, device=w.device), (self.off[1:] - self.off[:-1]).long())
            k = e[:, 0] & 0xFFFF
            r = (e[:, 0] >> 16) & 0xF
            bits = (e[:, 1] & 0xFFFF).to(torch.int32)
            wv = w.view(torch.int16)
            wv[tiles * 16 + r, k] = bits.to(torch.int16)
        return w


def ctl_mode() -> str:
    """'packed' (default), 'stock' or 'alt' (DS41H_VHEAD_CTL)."""
    if not CTL:
        return "packed"
    try:
        m = os.stat(CTL).st_mtime_ns
    except OSError:
        return "packed"
    if m != _CTL["mtime"]:
        try:
            with open(CTL) as f:
                _CTL["mode"] = str(json.load(f).get("mode", "packed"))
            _CTL["mtime"] = m
            logger.info("ds41h vhead: ctl mode=%s", _CTL["mode"])
        except (OSError, ValueError):
            return "packed"
    mode = _CTL["mode"]
    if mode == "alt":
        _CTL["n"] += 1
        return "packed" if _CTL["n"] % 2 else "stock"
    return mode


def build_for(target_model: torch.nn.Module) -> PackedHead | None:
    """Pack the target lm_head shard (called at load time, before the KV-cache memory profile)."""
    lm = getattr(target_model, "language_model", None)
    lm = lm if lm is not None and hasattr(lm, "lm_head") else target_model
    head = getattr(lm, "lm_head", None)
    w = getattr(head, "weight", None)
    if not (isinstance(w, torch.Tensor) and w.is_cuda and w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[1] == 5120
            and w.shape[0] % 16 == 0 and w.stride(1) == 1 and getattr(head, "bias", None) is None):
        logger.info("ds41h vhead: lm_head %s not packable; stock verification head", None if w is None else tuple(w.shape))
        return None
    with torch.no_grad():
        p = PackedHead(w.detach())
    torch.cuda.synchronize(w.device)
    lm._ds41h_vhc = p
    logger.info("ds41h vhead: packed lm_head [%d x 5120]: %.1f MB (bf16 %.1f MB), %d escapes", p.n, p.bytes / 1e6,
                w.numel() * 2 / 1e6, p.nesc)
    return p
