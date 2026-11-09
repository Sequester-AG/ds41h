#!/usr/bin/env python3
"""Check a layer capture's KV records (CPU only).

The capture probe (_gather) indexed the cache as cache[slot // bs, slot % bs] with the view's
stride (bs*584, 584, 1). Stock sm90 fp8_ds_mla pages are split: data rows (448 e4m3 NoPE + 64 bf16 RoPE = 576 B) at
page + pos*576, ue8m0 scale rows (7 exponents + pad = 8 B) at page + bs*576 + pos*8. So each captured "record" is the
page slice [pos*584, pos*584 + 584), not the record. Slices of all rows together still cover part of each page; this
script rebuilds what it can and reports per layer / step which rows have every record they read recoverable.
"""
import os
import sys

import torch

# capture directory (CAP_DIR overrides; a re-capture with the fixed probe stores real records and is used as is)
D = os.environ.get("CAP_DIR", "cap/").rstrip("/") + "/"


def page_bytes(idx: torch.Tensor, rec: torch.Tensor, bs: int):
    """idx [N] global slots (-1 = none), rec [N, 584] page slices -> (pages [P], bytes [P, bs*584], known [P, bs*584])."""
    ok = idx >= 0
    i = idx[ok].long()
    c = rec[ok]
    b, p = i // bs, i % bs
    ub, inv = b.unique(return_inverse=True)
    buf = torch.zeros(len(ub), bs * 584, dtype=torch.uint8)
    known = torch.zeros(len(ub), bs * 584, dtype=torch.bool)
    off = p[:, None] * 584 + torch.arange(584)
    buf[inv[:, None], off] = c
    known[inv[:, None], off] = True
    return ub, buf, known


def records(idx: torch.Tensor, bs: int, ub, buf, known):
    """idx [...] global slots -> (records [..., 584] in the stock record order data|scales, recoverable [...] bool)."""
    ok = idx >= 0
    i = idx.clamp(min=0).long()
    b, p = i // bs, i % bs
    pg = torch.searchsorted(ub, b).clamp(max=len(ub) - 1)
    doff = p[..., None] * 576 + torch.arange(576)
    soff = bs * 576 + p[..., None] * 8 + torch.arange(8)
    off = torch.cat([doff, soff], -1)
    rec = buf[pg[..., None], off]
    good = known[pg[..., None], off].all(-1) & (ub[pg] == b)
    rec = torch.where(ok[..., None], rec, torch.zeros((), dtype=torch.uint8))
    return rec, good | ~ok


def load(step: int, layer: int, rank: int = 0):
    return torch.load(f"{D}s{step}_L{layer}_r{rank}.pt", weights_only=False)


def rebuild(r: dict):
    """-> dict(swa_rec [T,128,584], swa_ok [T,128], comp_rec [T,512,584], comp_ok [T,512], swa_bs, comp_bs)."""
    a = r["attn"]
    T = r["T"]
    si = a["swa_indices"].view(T, -1)
    ti = a["topk_global"].view(T, -1)
    sbs = a["swa_cache_shape"][1]
    cbs = a["comp_cache_shape"][1]
    raw_s = a["swa_records"].view(T, -1, 584)
    raw_c = a["comp_records"].view(T, -1, 584)
    # a correct record has pad byte 583 == 0 and plausible ue8m0 exponents; the page slices do not
    looks_split = bool((raw_s[..., 583][si >= 0] != 0).float().mean() > 0.5)
    if not looks_split:
        return dict(swa_rec=raw_s, swa_ok=torch.ones(si.shape, dtype=torch.bool), comp_rec=raw_c,
                    comp_ok=torch.ones(ti.shape, dtype=torch.bool), swa_bs=sbs, comp_bs=cbs, fixed_capture=True)
    sw = records(si, sbs, *page_bytes(si.reshape(-1), raw_s.reshape(-1, 584), sbs))
    cp = records(ti, cbs, *page_bytes(ti.reshape(-1), raw_c.reshape(-1, 584), cbs))
    return dict(swa_rec=sw[0], swa_ok=sw[1], comp_rec=cp[0], comp_ok=cp[1], swa_bs=sbs, comp_bs=cbs, fixed_capture=False)


def main():
    for st in (0, 1):
        for L in (3, 20, 21, 24):
            r = load(st, L)
            a = r["attn"]
            T = r["T"]
            rb = rebuild(r)
            ow = rb["swa_ok"].all(1)
            oc = rb["comp_ok"].all(1)
            sr = rb["swa_rec"][rb["swa_ok"] & (a["swa_indices"].view(T, -1) >= 0)]
            ex = sr[:, 576:583].long()
            pos = r["in"]["positions"].float()
            tl = a["topk_lens"].float()
            print(f"s{st} L{L:2d} T={T} swa_only={a['swa_only']} pos/topk_len med {(pos / tl.clamp(min=1)).median().item():6.1f} "
                  f"topk_len med {int(tl.median())} max {int(tl.max())} swa_len med {int(a['swa_lens'].float().median())} | "
                  f"records recoverable: window {rb['swa_ok'].float().mean().item():.3f} comp {rb['comp_ok'].float().mean().item():.3f} | "
                  f"rows all recoverable: window {ow.float().mean().item():.3f} comp {oc.float().mean().item():.3f} both {(ow & oc).sum().item()} "
                  f"| rebuilt exps {ex.min().item()}-{ex.max().item()} pad!=0 {(sr[:, 583] != 0).sum().item()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
