#!/usr/bin/env python3
"""ds41h-phys: fused W8A16 dense GEMM (w8a16.cu) -- build, weight repack, call wrappers.

repack(w8 [N, K] e4m3, s8 [N/32, K/32] ue8m0) -> packed uint8 [K/128, N/256, 32896] (chunk-major): per (128-k chunk, 256-row tile)
two 16,448-byte halves, one per consumer warpgroup (rows 128 c .. 128 c + 127): 16 KB of RS-wgmma A fragments (k32 block
kb, 64-row sub-tile m, warp w: 512 B; lane l = 4g + t: 16 B = words [(g, k 2t..2t+1), (g+8, 2t..2t+1)],
[(g, 2t+8..), (g+8, 2t+8..)], the same + 16), then 16 B of scales ([kb][32-row block of the half]) and 48 B of pad.
unpack() inverts it exactly (checked before every use in the bench).
"""
from __future__ import annotations

import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TR, KC = 256, 128
HW = TR // 2 * KC  # fragment bytes per half
HB = HW + 64       # half block: fragments + 16 B scales + pad
CHUNK = 2 * HB


def build(verbose: bool = False, abl: int = 0, prof: bool = False, dq: int | None = None, extra_defs: tuple = ()):
    """abl != 0: a separate timing-ablation build (W8_ABL; 1 no dequant, 2 no wgmma, 4 no x loads; results are wrong).
    prof: clock64 profile build (W8_PROF). dq: dequant variant (W8_DQ; default from env W8A16_DQ, else 1; 0 = cvt path,
    1 = integer bit placement)."""
    from torch.utils.cpp_extension import load

    dq = int(os.environ.get("W8A16_DQ", "1")) if dq is None else int(dq)
    defs = [d for d in os.environ.get("W8A16_DEFS", "").split(",") if d] + list(extra_defs)  # extra -D overrides, e.g. W8_PP=0
    sfx = (f"_abl{abl}" if abl else "") + ("_prof" if prof else "") + (f"_dq{dq}" if dq != 1 else "") + "".join(
        "_" + d.replace("=", "").lower() for d in defs)
    bdir = os.environ.get("W8A16_BUILD", os.path.join(HERE, "build_w8a16")) + sfx
    os.makedirs(bdir, exist_ok=True)
    flags = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-Xptxas=-v", f"-DW8_ABL={abl}",
             f"-DW8_PROF={int(prof)}", f"-DW8_DQ={dq}"] + [f"-D{d}" for d in defs]
    return load(name="w8a16_ext" + sfx, sources=[os.path.join(HERE, "w8a16.cu")], extra_cuda_cflags=flags, extra_ldflags=["-lcuda"],
                extra_include_paths=[HERE], build_directory=bdir, verbose=verbose)


def _maps(device):
    """Fragment byte q of half c -> (byte offset in the block, row in tile 0..255, k in chunk 0..127)."""
    p = torch.arange(2 * HW, device=device)
    c, q = p // HW, p % HW
    kb, m, w, ln, b = q // 4096, (q // 2048) % 2, (q // 512) % 4, (q // 16) % 32, q % 16
    g, t = ln // 4, ln % 4
    word, bi = b // 4, b % 4
    s16, khi = word // 2, word % 2
    row = c * 128 + m * 64 + w * 16 + g + 8 * (bi // 2)
    k = kb * 32 + s16 * 16 + khi * 8 + 2 * t + (bi % 2)
    return c * HB + q, row, k


def _scale_pos(device):
    """Scale byte of (kb, 32-row block rb 0..7) -> byte offset in the block."""
    kb = torch.arange(4, device=device).view(4, 1)
    rb = torch.arange(8, device=device).view(1, 8)
    return ((rb // 4) * HB + HW + kb * 4 + rb % 4).reshape(-1)  # [kb * 8 + rb]


def repack(w8: torch.Tensor, s8: torch.Tensor) -> torch.Tensor:
    N, K = w8.shape
    assert N % TR == 0 and K % KC == 0, (N, K)
    assert s8.shape == (N // 32, K // 32), s8.shape
    dev = w8.device
    wb = w8.contiguous().view(torch.uint8).view(N // TR, TR, K // KC, KC).permute(0, 2, 1, 3)  # [T, C, 256, 128]
    pos, row, k = _maps(dev)
    out = torch.zeros(N // TR, K // KC, CHUNK, dtype=torch.uint8, device=dev)
    out[:, :, pos] = wb[:, :, row, k]
    sb = s8.contiguous().view(torch.uint8).view(N // TR, 8, K // KC, 4).permute(0, 2, 3, 1).reshape(N // TR, K // KC, 32)
    out[:, :, _scale_pos(dev)] = sb
    return out.transpose(0, 1).contiguous()  # chunk-major [K/128, N/256, CHUNK]: the CTAs stream adjacent blocks


def unpack(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed = packed.transpose(0, 1)  # -> [T, C, CHUNK]
    T, C, _ = packed.shape
    pos, row, k = _maps(packed.device)
    w = torch.empty(T, C, TR, KC, dtype=torch.uint8, device=packed.device)
    w[:, :, row, k] = packed[:, :, pos]
    w = w.permute(0, 2, 1, 3).reshape(T * TR, C * KC)
    s = packed[:, :, _scale_pos(packed.device)].reshape(T, C, 4, 8).permute(0, 3, 1, 2).reshape(T * 8, C * 4)
    return w, s


def check_scales(s8: torch.Tensor) -> tuple[int, int]:
    v = s8.contiguous().view(torch.uint8)
    lo, hi = int(v.min()), int(v.max())
    assert 16 <= lo and hi <= 134, f"exact dequant paths need scale bytes in [16, 134], got [{lo}, {hi}]"
    return lo, hi


_WS: dict = {}


def _shared_ws(dev, n: int) -> torch.Tensor:
    key = str(dev)
    if key not in _WS or _WS[key].numel() < n:
        _WS[key] = torch.empty(n, dtype=torch.float32, device=dev)
    return _WS[key]


class Gemm:
    """One packed weight for the fused kernel. rows_per_group / kx_per_group: the grouped wo_a (1024, 4096)."""

    def __init__(self, ext, w8, s8, splits: int, rows_per_group: int | None = None, kx_per_group: int = 0):
        self.ext = ext
        self.N, self.K = w8.shape
        self.scales = check_scales(s8)
        self.packed = repack(w8, s8)
        self.splits = int(splits)
        self.rpg = int(rows_per_group or self.N)
        self.kxg = int(kx_per_group)
        T = self.packed.shape[1]
        dev = self.packed.device
        # split-K workspace (fp32 partials): ONE per device shared by every Gemm (calls run in stream order), so it stays
        # L2-resident between calls (the weights stream evict-first); a private workspace per Gemm / layer goes cold and
        # its partial writes cost ~9 us. Monotonic per-tile arrival counters stay per Gemm.
        self.ws = _shared_ws(dev, max(self.splits, 1) * T * 128 * 260 if self.splits > 1 else 1)  # [T][S][NT][256 + 4]
        self.cnt = torch.zeros(T, dtype=torch.int32, device=dev)

    def __call__(self, x, y, pos=None, cs=None, kv=None, slot=None, cache=None, page_stride=0, shift=0, inv_pos=None, ext=None,
                 prof=None):
        (ext or self.ext).gemm(x, self.packed, y, self.rpg, self.kxg, self.splits, pos, cs, kv, slot, cache, int(page_stride),
                               int(shift), inv_pos, self.ws, self.cnt, prof)
        return y
