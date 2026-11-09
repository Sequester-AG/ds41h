#!/usr/bin/env python3
"""Bench: the moe_stream routed-MoE layer on one GPU with real weights of one layer (TP rank 0).

  python3 bench.py --build-only                      # compile the extension (CPU only)
  python3 bench.py --selftest                        # CPU: vectorized repack == reference repack
  python3 bench.py --layer 21 --seqs 128 [--routing routing.pt]   # GPU: correctness, time, bandwidth, energy

Routing: synthetic by default (skewed expert popularity, Gamma(0.47), 4 rows per sequence keeping 60 % of the
previous row's experts), or a saved {'ids': [T, 6] int, 'w': [T, 6] float} file of real routing. Reports per layer:
experts touched, streamed weight bytes, GEMM1/GEMM2/prep time, achieved TB/s, joules (NVML energy counter over a
~2 s loop), max error vs an fp64 reference that applies the kernel's own f16 roundings (x * 2^p, h * 2^8), and the
estimate for production's moe_dec 0x55 at the same shape (2.113 us per touched expert + 0.0909 us per routed row, in-situ
fit at 600 W).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import repack as rp  # noqa: E402

D, I, NE, TOPK, NMAX = 5120, 576, 384, 6, 64
EC_DT = None  # numpy structured dtype, built lazily


# --------------------------------------------------------------------------------------------- vectorized repack
def maps_w13():
    q, kb, c, t, j, n = torch.meshgrid(*(torch.arange(x) for x in (9, 80, 2, 128, 4, 8)), indexing="ij")
    w, lane = t // 32, t % 32
    g, tq = lane // 4, lane % 4
    roff = torch.tensor([0, 8, 0, 8, 0, 8, 0, 8])[n]
    koff = torch.tensor([0, 0, 1, 1, 8, 8, 9, 9])[n]
    i = 64 * q + 32 * c + 8 * w + g
    row = torch.where(roff == 0, i, I + i)
    k = 64 * kb + 16 * j + 2 * tq + (koff % 2) + torch.where(koff >= 8, 8, 0)
    R = torch.arange(128)
    cR, rem = R // 64, R % 64
    wR, gg = rem // 16, rem % 16
    iR = 64 * torch.arange(9).view(9, 1) + 32 * cR + 8 * wR + gg % 8
    srow = torch.where(gg >= 8, I + iR, iR)  # [9, 128]
    return row * D + k, srow


def maps_w2():
    rt, kb, c, t, j, n = torch.meshgrid(*(torch.arange(x) for x in (40, 9, 2, 128, 4, 8)), indexing="ij")
    w, lane = t // 32, t % 32
    g, tq = lane // 4, lane % 4
    roff = torch.tensor([0, 8, 0, 8, 0, 8, 0, 8])[n]
    koff = torch.tensor([0, 0, 1, 1, 8, 8, 9, 9])[n]
    row = 128 * rt + 64 * c + 16 * w + g + roff
    k = 64 * kb + 16 * j + 2 * tq + (koff % 2) + torch.where(koff >= 8, 8, 0)
    srow = 128 * torch.arange(40).view(40, 1) + torch.arange(128)
    return row * I + k, srow


SHIFT = (4 * torch.arange(8)).view(1, 1, 1, 1, 1, 8)


def pack(codes_flat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """codes_flat [R*K] int64 -> words (int64 < 2^32) with the leading dims of idx minus the last (8 nibbles)."""
    return (codes_flat[idx] << SHIFT.to(idx.device)).sum(-1)


def repack_w13_fast(w1, w3, s1, s3, maps):
    idx, srow = maps
    codes = torch.cat([rp.nibbles(w1), rp.nibbles(w3)], 0).flatten()
    words = pack(codes, idx)
    S = torch.cat([s1, s3], 0)  # [1152, 160]
    scales = S[srow].permute(0, 2, 1).contiguous()  # [9, 160, 128]
    return words, scales


def repack_w2_fast(w2, s2, maps):
    idx, srow = maps
    words = pack(rp.nibbles(w2).flatten(), idx)
    scales = s2[srow].permute(0, 2, 1).contiguous()  # [40, 18, 128]
    return words, scales


def selftest():
    gen = torch.Generator().manual_seed(3)
    w1 = torch.randint(0, 256, (I, D // 2), dtype=torch.uint8, generator=gen)
    w3 = torch.randint(0, 256, (I, D // 2), dtype=torch.uint8, generator=gen)
    s1 = torch.randint(110, 130, (I, D // 32), dtype=torch.uint8, generator=gen)
    s3 = torch.randint(110, 130, (I, D // 32), dtype=torch.uint8, generator=gen)
    w2 = torch.randint(0, 256, (D, I // 2), dtype=torch.uint8, generator=gen)
    s2 = torch.randint(110, 130, (D, I // 32), dtype=torch.uint8, generator=gen)
    a, sa = rp.repack_w13(w1, w3, s1, s3)
    b, sb = rp.repack_w2(w2, s2)
    fa, fsa = repack_w13_fast(w1, w3, s1, s3, maps_w13())
    fb, fsb = repack_w2_fast(w2, s2, maps_w2())
    ok = torch.equal(a, fa) and torch.equal(sa, fsa) and torch.equal(b, fb) and torch.equal(sb, fsb)
    print(f"selftest vectorized repack == reference repack: {ok}")
    return ok


# --------------------------------------------------------------------------------------------- checkpoint
class Ckpt:
    def __init__(self, d):
        from safetensors import safe_open
        self.d, self.so = d, safe_open
        self.idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.h = {}

    def get(self, k):
        f = self.idx[k]
        if f not in self.h:
            self.h[f] = self.so(os.path.join(self.d, f), "pt")
        return self.h[f].get_tensor(k).view(torch.uint8)


def load_layer(ck: Ckpt, L: int, dev, cache=None):
    if cache and os.path.exists(cache):
        z = torch.load(cache)
        print(f"layer {L}: repacked weights from {cache}, scale bytes {z['lo']}..{z['hi']}")
        return z["A13"].to(dev), z["S13"].to(dev), z["A2"].to(dev), z["S2"].to(dev), z["raw"]
    m13, m2 = maps_w13(), maps_w2()
    A13 = torch.empty(NE, 9, 80, 2, 128, 4, dtype=torch.int32)
    S13 = torch.empty(NE, 9, 160, 128, dtype=torch.uint8)
    A2 = torch.empty(NE, 40, 9, 2, 128, 4, dtype=torch.int32)
    S2 = torch.empty(NE, 40, 18, 128, dtype=torch.uint8)
    raw = []
    lo, hi = 255, 0
    t0 = time.time()
    for e in range(NE):
        p = f"layers.{L}.ffn.experts.{e}."
        w1, w3, w2 = ck.get(p + "w1.weight")[:I], ck.get(p + "w3.weight")[:I], ck.get(p + "w2.weight")[:, : I // 2]
        s1, s3, s2 = ck.get(p + "w1.scale")[:I], ck.get(p + "w3.scale")[:I], ck.get(p + "w2.scale")[:, : I // 32]
        for s in (s1, s3, s2):
            lo, hi = min(lo, int(s.min())), max(hi, int(s.max()))
        a, sa = repack_w13_fast(w1, w3, s1, s3, m13)
        b, sb = repack_w2_fast(w2, s2, m2)
        A13[e], S13[e], A2[e], S2[e] = rp.to_u32(a), sa, rp.to_u32(b), sb
        raw.append(tuple(z.clone() for z in (w1, w3, w2, s1, s3, s2)))  # clone: views would save whole tensors
    if lo < 107 or hi > 136:
        raise SystemExit(f"scale bytes {lo}..{hi} outside the exact f16 range 107..136: needs a reference exponent")
    print(f"layer {L}: repacked {NE} experts in {time.time() - t0:.0f} s, scale bytes {lo}..{hi}")
    if cache:
        torch.save(dict(A13=A13, S13=S13, A2=A2, S2=S2, raw=raw, lo=lo, hi=hi), cache)
        print(f"saved {cache}")
    if dev is None:
        return None
    return A13.to(dev), S13.to(dev), A2.to(dev), S2.to(dev), raw


def dequant(w, s):
    """FP4 [R, K/2] + ue8m0 [R, K/32] -> float64 [R, K]."""
    codes = rp.nibbles(w)
    v = rp.E2M1.to(w.device)[codes]
    return v * torch.pow(2.0, s.double() - 127).repeat_interleave(32, 1)


# --------------------------------------------------------------------------------------------- routing / tiles
def synth_routing(n_seq, seed=0, r=0.47, keep=0.6):
    torch.manual_seed(seed)  # Gamma / Dirichlet use the global RNG: without this every run routed differently
    g = torch.Generator().manual_seed(seed)
    pop = torch.distributions.Gamma(torch.tensor(r), torch.tensor(1.0)).sample((NE,))
    pop = pop / pop.sum()
    ids = torch.empty(n_seq * 4, TOPK, dtype=torch.long)
    for s in range(n_seq):
        prev = torch.multinomial(pop, TOPK, replacement=False, generator=g)
        ids[4 * s] = prev
        for j in range(1, 4):
            kept = prev[torch.rand(TOPK, generator=g) < keep]
            p = pop.clone()
            p[kept] = 0
            if len(kept) < TOPK:
                new = torch.multinomial(p, TOPK - len(kept), replacement=False, generator=g)
                prev = torch.cat([kept, new])
            else:
                prev = kept
            ids[4 * s + j] = prev
    w = torch.distributions.Dirichlet(torch.ones(TOPK)).sample((n_seq * 4,)) * 1.5
    return ids, w.float()


def build_tiles(ids, w, qpe1=9, order="snake", grid=132, ec_inv=False):
    import numpy as np
    global EC_DT
    EC_DT = np.dtype([("e", "<i4"), ("n_real", "<i4"), ("n_pad", "<i4"), ("pad", "<i4"), ("img_off", "<i8"),
                      ("h_off", "<i8"), ("tok", "<i4", (NMAX,)), ("wt", "<f4", (NMAX,))]
                     + ([("inv", "<f4", (NMAX,))] if ec_inv else []))  # v3: 2^-p per slot, written by k_gather
    T = ids.shape[0]
    per = {}
    for t in range(T):
        for j in range(TOPK):
            per.setdefault(int(ids[t, j]), []).append((t, float(w[t, j])))
    ecs = []
    img = hb = 0
    for e, lst in sorted(per.items(), key=lambda kv: -len(kv[1])):
        for c0 in range(0, len(lst), NMAX):
            ch = lst[c0:c0 + NMAX]
            npad = (len(ch) + 7) // 8 * 8
            rec = np.zeros((), EC_DT)
            rec["e"], rec["n_real"], rec["n_pad"], rec["img_off"], rec["h_off"] = e, len(ch), npad, img, hb
            rec["tok"][:] = -1
            rec["tok"][: len(ch)] = [t for t, _ in ch]
            rec["wt"][: len(ch)] = [x for _, x in ch]
            ecs.append(rec)
            img += 80 * npad * 128
            hb += 9 * npad * 128
    arr = np.stack(ecs)
    TILE_DT = np.dtype([("ec", "<i4"), ("q", "<i4"), ("e", "<i4"), ("npad", "<i4"), ("boff", "<i8")])  # = Tile

    def tiles(qpe, off):
        t = np.zeros(len(ecs) * qpe, TILE_DT)
        for i, r in enumerate(ecs):
            sl = t[i * qpe:(i + 1) * qpe]
            sl["ec"], sl["q"], sl["e"], sl["npad"], sl["boff"] = i, np.arange(qpe), r["e"], r["n_pad"], r[off]
        return t.view(np.int32).reshape(-1, 6)

    t1, t2 = tiles(qpe1, "img_off"), tiles(40, "h_off")  # qpe1 = 3 for v4 (384-row GEMM1 tiles)
    if order == "snake":  # the kernels give tile i to CTA i % grid: heaviest (largest n_pad) first, every other round
        def snake(t):     # reversed, so each CTA's total work is balanced (v4 has only ~6 GEMM1 tiles per CTA)
            idx = np.argsort(-t[:, 3], kind="stable")
            chunks = [idx[i:i + grid] for i in range(0, len(idx), grid)]
            return t[np.concatenate([c if k % 2 == 0 else c[::-1] for k, c in enumerate(chunks)])]
        t1, t2 = snake(t1), snake(t2)
    return arr, t1, t2, img, hb, len(per)


# --------------------------------------------------------------------------------------------- extension
def build_ext():
    from torch.utils.cpp_extension import load
    abl = int(os.environ.get("MS_ABL", "0"))  # timing ablations, see moe_stream.cu (results invalid if > 0)
    bd = os.environ.get("MS_BUILD_DIR")
    if bd and abl:
        bd = f"{bd}_abl{abl}"
        os.makedirs(bd, exist_ok=True)
    return load(name=f"moe_stream_ext{'_abl%d' % abl if abl else ''}", sources=[os.path.join(HERE, "moe_stream.cu")],
                verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
                                   "-Xptxas=-v", "--expt-relaxed-constexpr", f"-DMS_ABL={abl}"],
                build_directory=bd or None)


def build_ext3():
    """moe_stream3.cu (v3). MS3_MMA_MAXN / MS3_ABL env select compile-time variants (own build dir each)."""
    from torch.utils.cpp_extension import load
    mx, abl = int(os.environ.get("MS3_MMA_MAXN", "32")), int(os.environ.get("MS3_ABL", "0"))
    chs, st = int(os.environ.get("MS3_CHS1", "8")), int(os.environ.get("MS3_S", "3"))
    epf, red2 = int(os.environ.get("MS3_EPF", "1")), int(os.environ.get("MS3_RED2", "0"))
    extra = {k: os.environ[k] for k in ("MS3_CTAS", "MS3_CHS2", "MS3_CHB2", "MS3_S2", "MS3_PSLEEP", "MS3_YSTG", "MS3_CVT2") if k in os.environ}
    tag = f"_m{mx}" + (f"_abl{abl}" if abl else "") + ("" if (chs, st) == (8, 3) else f"_c{chs}s{st}") + f"_e{epf}r{red2}"
    tag += "".join(f"_{k[4:].lower()}{v}" for k, v in sorted(extra.items()))
    bd = os.environ.get("MS_BUILD_DIR")
    if bd:
        bd = f"{bd}3{tag}"
        os.makedirs(bd, exist_ok=True)
    return load(name=f"moe_stream3_ext{tag}", sources=[os.path.join(HERE, "moe_stream3.cu")], verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
                                   "-Xptxas=-v", "--expt-relaxed-constexpr", f"-DMS3_MMA_MAXN={mx}",
                                   f"-DMS3_ABL={abl}", f"-DMS3_CHS1={chs}", f"-DMS3_S={st}", f"-DMS3_EPF={epf}", f"-DMS3_RED2={red2}"]
                + [f"-D{k}={v}" for k, v in sorted(extra.items())],
                build_directory=bd or None)


def build_ext4():
    """moe_stream4.cu (v4). MS4_MMA_MAXN / MS4_ABL / MS4_ACC8 env select compile-time variants (own build dir each)."""
    from torch.utils.cpp_extension import load
    mx, abl, acc = (int(os.environ.get(k, d)) for k, d in (("MS4_MMA_MAXN", "16"), ("MS4_ABL", "0"), ("MS4_ACC8", "1")))
    tag = f"_m{mx}_a{acc}" + (f"_abl{abl}" if abl else "")
    bd = os.environ.get("MS_BUILD_DIR")
    if bd:
        bd = f"{bd}4{tag}"
        os.makedirs(bd, exist_ok=True)
    return load(name=f"moe_stream4_ext{tag}", sources=[os.path.join(HERE, "moe_stream4.cu")], verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
                                   "-Xptxas=-v", "--expt-relaxed-constexpr", f"-DMS4_MMA_MAXN={mx}",
                                   f"-DMS4_ABL={abl}", f"-DMS4_ACC8={acc}"],
                build_directory=bd or None)


def build_extr():
    """moe_raw.cu: the v3 kernels on ds41h's raw expert layout (for serving). Same MS3_* knobs as build_ext3."""
    from torch.utils.cpp_extension import load
    keys = ("MS3_MMA_MAXN", "MS3_ABL", "MS3_CHS1", "MS3_S", "MS3_CHS2", "MS3_CHB2", "MS3_S2", "MS3_CVT2", "MSR_GATHER2", "MSR_G2BULK", "MS3_YSTG", "MSR_G2PAIR", "MSR_G1PAIR", "MSR_G1PAIR_CH", "MSR_R8TMA", "MSR_R8NW", "MSR_R8S", "MSR_R8ABL", "MSR_GATHER3", "MSR_G3Y", "MSR_G5U", "MSR_G5B", "MSR_RT_W", "MSR_RT_PAR", "MSR_G5BS", "MSR_G5NOPAD")
    extra = {k: os.environ[k] for k in keys if k in os.environ}
    tag = "".join(f"_{k[4:].lower()}{v}" for k, v in sorted(extra.items()))
    bd = os.environ.get("MS_BUILD_DIR")
    if bd:
        bd = f"{bd}r{tag}"
        os.makedirs(bd, exist_ok=True)
    return load(name=f"moe_raw_ext{tag}", sources=[os.path.join(HERE, "moe_raw.cu")], verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
                                   "-Xptxas=-v", "--expt-relaxed-constexpr"] + [f"-D{k}={v}" for k, v in sorted(extra.items())],
                build_directory=bd or None)


def raw_layout(raw, dev):
    """bench.load_layer raw list (per expert: w1, w3, w2, s1, s3, s2 of rank 0) -> ds41h raw tensors on dev:
    w13 [E, 2I, H/2], s13 [E, 2I, H/32], w2 [E, H, I/2], s2 [E, H, I/32] (uint8, gate rows first)."""
    w13 = torch.stack([torch.cat([r[0], r[1]], 0) for r in raw]).to(dev)
    s13 = torch.stack([torch.cat([r[3], r[4]], 0) for r in raw]).to(dev)
    w2 = torch.stack([r[2] for r in raw]).contiguous().to(dev)
    s2 = torch.stack([r[5] for r in raw]).contiguous().to(dev)
    return w13.contiguous(), s13.contiguous(), w2, s2


P_ROWS = None


def make_p(S):
    """moe_stream scales [..., b = 2*KB, 128 rows] (raw ue8m0 bytes) -> v3 [..., KB, 64 slots, 4 B]:
    slot p = 32c + 8w + g holds e(b0, R), e(b0, R+8), e(b1, R), e(b1, R+8), R = 64c + 16w + g, e = (s - 106) << 2
    (the high byte of the f16 2^(s-121)). Exact for s in 107..136 (checked)."""
    global P_ROWS
    lo, hi = int(S.min()), int(S.max())
    if lo < 107 or hi > 136:
        raise SystemExit(f"v3 scales: bytes {lo}..{hi} outside 107..136")
    if P_ROWS is None:
        p = torch.arange(64)
        P_ROWS = 64 * (p // 32) + 16 * ((p // 8) % 4) + p % 8
    rows = P_ROWS.to(S.device)
    e = ((S.to(torch.int32) - 106) << 2).to(torch.uint8)
    e = e.view(*S.shape[:-2], S.shape[-2] // 2, 2, 128)
    P = torch.stack([e[..., 0, rows], e[..., 0, rows + 8], e[..., 1, rows], e[..., 1, rows + 8]], dim=-1)
    return P.contiguous()


def selftest_p():
    """CPU: make_p is a bijection onto the slots and PRMT(0x0404 / 0x1414 / 0x2424 / 0x3434) gives scale_mul."""
    g = torch.Generator().manual_seed(5)
    S = torch.randint(107, 137, (3, 2, 160, 128), generator=g, dtype=torch.int32).to(torch.uint8)
    P = make_p(S).view(torch.int32)  # [3, 2, 80, 64] little-endian u32
    ok = True
    for c in range(2):
        for w in range(4):
            for gg in range(8):
                slot, R = 32 * c + 8 * w + gg, 64 * c + 16 * w + gg
                for kb in (0, 37, 79):
                    u = int(P[1, 0, kb, slot]) & 0xFFFFFFFF
                    for k, (b, r) in enumerate(((0, R), (0, R + 8), (1, R), (1, R + 8))):
                        byte = (u >> (8 * k)) & 0xFF
                        sel = 0x0404 + 0x1010 * k
                        prmt = (byte << 8) | (byte << 24)  # what __byte_perm(u, 0, sel) returns
                        s = int(S[1, 0, 2 * kb + b, r])
                        f = (s - 106) << 10
                        ok &= prmt == (f | (f << 16))
                        ok &= sel == [0x0404, 0x1414, 0x2424, 0x3434][k]
    print("make_p selftest", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--seqs", type=int, default=128)
    ap.add_argument("--routing", default=None)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--energy-secs", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None, help="repacked-layer cache file (.pt); created if missing")
    ap.add_argument("--prepare-only", action="store_true", help="CPU: repack the layer into --cache and exit")
    ap.add_argument("--v3", action="store_true", help="moe_stream3.cu (mma.sync small chunks, PRMT scales, 32 KB stages)")
    ap.add_argument("--v4", action="store_true", help="moe_stream4.cu (384-row GEMM1 tiles, acc-side scales at N=8)")
    ap.add_argument("--order", default="snake", choices=("snake", "rr"), help="tile order (rr = expert-major, v2/v3 runs)")
    ap.add_argument("--same-img", action="store_true", help="TIMING ONLY: every GEMM1 tile reads the first chunk's x image")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if (selftest() and selftest_p()) else 1)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")  # never query the GPUs at build time
    if a.prepare_only:
        load_layer(Ckpt(a.model), a.layer, None, a.cache)
        return
    ver = "v4" if a.v4 else "v3" if a.v3 else "v2"
    ext = build_ext4() if a.v4 else build_ext3() if a.v3 else build_ext()
    print(f"extension built ({ver}); smem gemm1 {ext.smem_bytes(1)} B, gemm2 {ext.smem_bytes(2)} B")
    if a.build_only:
        return
    dev = torch.device("cuda")
    ck = Ckpt(a.model)
    A13, S13, A2, S2, raw = load_layer(ck, a.layer, dev, a.cache)
    if a.v3 or a.v4:
        S13, S2 = make_p(S13), make_p(S2)
    if a.routing:
        r = torch.load(a.routing)
        ids, w = r["ids"].long(), r["w"].float()
    else:
        ids, w = synth_routing(a.seqs)
    T = ids.shape[0]
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    arr, t1, t2, img_b, h_b, touched = build_tiles(ids, w, qpe1=3 if a.v4 else 9, order=a.order, grid=sms, ec_inv=a.v3)
    import numpy as np
    ecs = torch.from_numpy(np.frombuffer(arr.tobytes(), dtype=np.uint8).copy()).to(dev)
    if a.same_img:  # boff (int64 in columns 4-5) of every GEMM1 tile -> 0: images stay L2-resident (results invalid)
        t1 = t1.copy()
        t1[:, 4:6] = 0
    tiles1, tiles2 = torch.from_numpy(t1).to(dev), torch.from_numpy(t2).to(dev)
    torch.manual_seed(0)
    x = torch.randn(T, D, device=dev).to(torch.bfloat16)
    img = torch.zeros(img_b, dtype=torch.uint8, device=dev)
    himg = torch.zeros(h_b, dtype=torch.uint8, device=dev)
    ptok = torch.zeros(T, dtype=torch.int32, device=dev)
    inv = torch.zeros(T, dtype=torch.float32, device=dev)
    out = torch.zeros(T, D, dtype=torch.float32, device=dev)
    cnt = torch.zeros(1, dtype=torch.int32, device=dev)
    n_ec = len(arr)

    def run():
        out.zero_()
        ext.prep(x, ecs, n_ec, img, ptok, inv)
        ext.gemm1(A13, S13, ecs, tiles1, cnt, img, himg, inv, out, 0)
        ext.gemm2(A2, S2, ecs, tiles2, cnt, himg, himg, inv, out, 0)

    run()
    torch.cuda.synchronize()
    # ---- reference: fp64, with the kernel's own roundings of x (x 2^p -> f16) and h (x 2^8 -> f16)
    p = ptok.double()
    xr = (x.double() * torch.pow(2.0, p).view(-1, 1)).half().double() / torch.pow(2.0, p).view(-1, 1)
    ref = torch.zeros(T, D, dtype=torch.float64, device=dev)
    for rec in arr:
        e, n = int(rec["e"]), int(rec["n_real"])
        tok = torch.as_tensor(rec["tok"][:n], device=dev).long()
        wt = torch.as_tensor(rec["wt"][:n], device=dev).double()
        w1, w3, w2, s1, s3, s2 = (z.to(dev) for z in raw[e])
        gt = (xr[tok] @ dequant(w1, s1).t()).clamp(max=10)
        up = (xr[tok] @ dequant(w3, s3).t()).clamp(-10, 10)
        h = gt * torch.sigmoid(gt) * up * wt.view(-1, 1)
        h = (h * 256).half().double() / 256
        ref.index_add_(0, tok, h @ dequant(w2, s2).t())
    err = (out.double() - ref).abs().max().item()
    scale = ref.abs().max().item()
    # ---- timing
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    tp = t_1 = t_2 = 0.0
    for _ in range(a.reps):
        ev[0].record()
        ext.prep(x, ecs, n_ec, img, ptok, inv)
        ev[1].record()
        ext.gemm1(A13, S13, ecs, tiles1, cnt, img, himg, inv, out, 0)
        ev[2].record()
        ext.gemm2(A2, S2, ecs, tiles2, cnt, himg, himg, inv, out, 0)
        ev[3].record()
        torch.cuda.synchronize()
        tp += ev[0].elapsed_time(ev[1]) / a.reps
        t_1 += ev[1].elapsed_time(ev[2]) / a.reps
        t_2 += ev[2].elapsed_time(ev[3]) / a.reps
    b1 = n_ec * 9 * (80 * 4096 + 160 * 128)
    b2 = n_ec * 40 * (9 * 4096 + 18 * 128)
    # ---- energy
    J = None
    try:
        import pynvml
        pynvml.nvmlInit()
        pr = torch.cuda.get_device_properties(dev)
        bus = f"{pr.pci_domain_id:08x}:{pr.pci_bus_id:02x}:{pr.pci_device_id:02x}.0"
        hdl = pynvml.nvmlDeviceGetHandleByPciBusId(bus)
        e0, t0, n = pynvml.nvmlDeviceGetTotalEnergyConsumption(hdl), time.time(), 0
        while time.time() - t0 < a.energy_secs:
            run()
            n += 1
            if n % 8 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        el = time.time() - t0
        J = (pynvml.nvmlDeviceGetTotalEnergyConsumption(hdl) - e0) / 1000.0
        W, J = J / el, J / n
        # per-GEMM power and SM clock (is one phase capped at 600 W?)
        phase = {}
        for nm, fn in (("gemm1", lambda: ext.gemm1(A13, S13, ecs, tiles1, cnt, img, himg, inv, out, 0)),
                       ("gemm2", lambda: ext.gemm2(A2, S2, ecs, tiles2, cnt, himg, himg, inv, out, 0))):
            torch.cuda.synchronize()
            e0, t0, clk, k = pynvml.nvmlDeviceGetTotalEnergyConsumption(hdl), time.time(), [], 0
            while time.time() - t0 < max(0.5, a.energy_secs / 2):
                for _ in range(16):
                    fn()
                k += 16
                clk.append(pynvml.nvmlDeviceGetClockInfo(hdl, pynvml.NVML_CLOCK_SM))
                torch.cuda.synchronize()
            el = time.time() - t0
            phase[nm] = dict(W=(pynvml.nvmlDeviceGetTotalEnergyConsumption(hdl) - e0) / 1000.0 / el,
                             mhz=sum(clk) / len(clk), mhz_min=min(clk), us=el / k * 1e6)
    except Exception as ex:  # energy is optional
        W = None
        phase = None
        print(f"energy measurement skipped: {ex}")
    base_us = touched * 2.113 + 6 * T * 0.0909
    res = dict(layer=a.layer, T=T, touched=touched, expert_chunks=n_ec, prep_us=tp * 1e3, gemm1_us=t_1 * 1e3,
               gemm2_us=t_2 * 1e3, moe_us=(t_1 + t_2) * 1e3, gemm1_TBps=b1 / (t_1 * 1e-3) / 1e12,
               gemm2_TBps=b2 / (t_2 * 1e-3) / 1e12, total_TBps=(b1 + b2) / ((t_1 + t_2) * 1e-3) / 1e12,
               streamed_GB=(b1 + b2) / 1e9, max_err=err, ref_max=scale, rel_err=err / scale if scale else None,
               W=W, J_per_layer=J, moe_dec_0x55_est_us=base_us, kernel=ver, order=a.order, seed_fixed=True,
               mma_maxn=int(os.environ.get("MS4_MMA_MAXN", "16") if a.v4 else os.environ.get("MS3_MMA_MAXN", "32")) if ver != "v2" else None,
               acc8=int(os.environ.get("MS4_ACC8", "1")) if a.v4 else None,
               abl=int(os.environ.get({"v4": "MS4_ABL", "v3": "MS3_ABL"}.get(ver, "MS_ABL"), "0")),
               stages=os.environ.get("MS3_CHS1", "8") + "x" + os.environ.get("MS3_S", "3") if ver == "v3" else None,
               epf=int(os.environ.get("MS3_EPF", "1")) if ver == "v3" else None, same_img=a.same_img, phase=phase,
               red2=int(os.environ.get("MS3_RED2", "0")) if ver == "v3" else None)
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "a").write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
