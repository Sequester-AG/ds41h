#!/usr/bin/env python3
"""ds41h-phys driver: the attention side of one consumer decoder layer on ONE H200 (TP4 rank 0, 16 heads).

  K2  q_b (bf16 GEMM, weights pre-dequantized from the e4m3 checkpoint: exact) + RoPE(q) + window KV record store
  K3  per-step metadata (key union + row membership, once per step / index group) + per-sequence sparse attention
      (window 128 + top-512, MQA latent 512, sink), optionally with K4's inverse RoPE fused into its epilogue
  K4  inverse RoPE (when not fused) + wo_a (2 local groups, bmm) + wo_b (rank partial)

Synthetic decode step: nseq sequences x 4 rows, contexts around --ctx, KV state in stock vLLM 0.32.0 sm90 fp8_ds_mla
pages (64 tokens, 584 B per token), per-row top-512 sets from a correlated-score generator calibrated to
locally captured probe dumps (union ~1,225 keys per sequence, membership {1: ~740, 2: ~240, 3: ~140, 4: ~100}) or the
dumped sets themselves (--topk real). Every output is checked against the ds41h reference (VLLM_SM90 numerics, fp32
math, stock's fp32 cos/sin cache) in bf16 ulps; stock FlashMLA sparse decode runs on the same state as the numerics and
speed baseline for the attention. Timing by CUDA events, power / SM clock / energy by NVML per op.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.environ.get("DS41H_REF", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, REF)
from ds41h.ref import layer as L  # noqa: E402
from ds41h.ref import quant as Q  # noqa: E402
from ds41h.ref.weights import CheckpointReader  # noqa: E402

MODEL = os.environ.get("DS41H_MODEL_DIR", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))
BS = 64  # paged-cache block size (tokens), stock
PAGE = BS * 584  # page stride in bytes
WINDOW = 128
DUMPS = os.environ.get("ATTN_DUMPS", "dumps")


# ----------------------------------------------------------------------------------------------- build
def build():
    from torch.utils.cpp_extension import load

    bdir = os.environ.get("ATTN2B_BUILD", os.path.join(HERE, "build"))
    os.makedirs(bdir, exist_ok=True)
    flags = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-Xptxas=-v"]
    k3 = load(name="attn_seq_ext", sources=[os.path.join(HERE, "attn_seq.cu")], extra_cuda_cflags=flags,
              build_directory=bdir, verbose=bool(int(os.environ.get("ATTN2B_VERBOSE", "0"))))
    pdl = int(os.environ.get("ATTN2B_PDL", "0"))  # dense.cu with programmatic dependent launch (separate build)
    bdk = bdir + ("_pdl" if pdl else "")
    os.makedirs(bdk, exist_ok=True)
    kd = load(name="attn_dense_ext" + ("_pdl" if pdl else ""), sources=[os.path.join(HERE, "dense.cu")],
              extra_cuda_cflags=flags + [f"-DDENSE_PDL={pdl}"], build_directory=bdk,
              verbose=bool(int(os.environ.get("ATTN2B_VERBOSE", "0"))))
    return k3, kd


def build3(prof=False):
    from torch.utils.cpp_extension import load
    bdir = os.environ.get("ATTN2B_BUILD", os.path.join(HERE, "build")) + ("3p" if prof else "3")
    os.makedirs(bdir, exist_ok=True)
    flags = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-Xptxas=-v"] + (["-DK3_PROF"] if prof else [])
    return load(name="attn_seq3_prof_ext" if prof else "attn_seq3_ext", sources=[os.path.join(HERE, "attn_seq3.cu")],
                extra_cuda_cflags=flags, build_directory=bdir, verbose=bool(int(os.environ.get("ATTN2B_VERBOSE", "0"))))


def build_today():
    """Production's ds41h kernel (ds41h/kernels/sparse_decode.cu, cluster per row) built from the ds41h source tree."""
    from torch.utils.cpp_extension import load
    kdir = os.path.join(os.environ.get("DS41H_SRC", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))), "ds41h", "kernels")
    bdir = os.path.join(HERE, "build_today")
    os.makedirs(bdir, exist_ok=True)
    return load(name="ds41h_sparse_decode_2b", sources=[os.path.join(kdir, "sparse_decode.cu")], extra_cflags=["-O3", "-std=c++20"],
                extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a"],
                extra_include_paths=[kdir], build_directory=bdir, verbose=False)


def build_prof():
    from torch.utils.cpp_extension import load
    bdir = os.environ.get("ATTN2B_BUILD", os.path.join(HERE, "build")) + "_prof"
    os.makedirs(bdir, exist_ok=True)
    flags = ["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-DK3_PROF"]
    return load(name="attn_seq_prof_ext", sources=[os.path.join(HERE, "attn_seq.cu")], extra_cuda_cflags=flags,
                build_directory=bdir, verbose=False)


# ----------------------------------------------------------------------------------------------- numerics helpers
def ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a - b| in bf16 ulps (ordered integer distance), a, b bf16."""
    def ordv(x):
        i = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (ordv(a) - ordv(b)).abs()


def ulp_stats(a: torch.Tensor, b: torch.Tensor, row: int = 512) -> dict:
    """a = candidate, b = reference (bf16). ulps: element-wise bf16 ulp distance (explodes for elements near zero);
    row_ulps: |a - b| in units of the bf16 ulp of the row's max |b| (rows of `row` elements: one head / one token),
    i.e. the error relative to the scale of the vector."""
    d = ulps(a, b).flatten().double()
    n = d.numel()
    af, bf_ = a.float(), b.float()
    rel = ((af - bf_).abs().max() / bf_.abs().max().clamp_min(1e-30)).item()
    rm = bf_.reshape(-1, row).abs().amax(1, keepdim=True).clamp_min(1e-30)
    rulp = torch.exp2(torch.floor(torch.log2(rm)) - 7)
    rd = ((af - bf_).abs().reshape(-1, row) / rulp).flatten()
    q = torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float64, device=d.device)
    dq = torch.quantile(d[torch.randperm(n, device=d.device)[: min(n, 1 << 24)]], q).tolist()
    rq = torch.quantile(rd.double()[torch.randperm(n, device=d.device)[: min(n, 1 << 24)]], q).tolist()
    return dict(n=n, exact=(d == 0).sum().item() / n, le1=(d <= 1).sum().item() / n, le2=(d <= 2).sum().item() / n,
                gt2=int((d > 2).sum().item()), max=int(d.max().item()), mean=d.mean().item(), p50=dq[0], p90=dq[1],
                p99=dq[2], p999=dq[3], rel_max=rel, row_ulp_max=rd.max().item(), row_ulp_mean=rd.mean().item(),
                row_ulp_p99=rq[2], row_ulp_p999=rq[3], nan=int(torch.isnan(af).sum().item()))


def stock_cos_sin(ratio: int, dev, max_pos: int) -> torch.Tensor:
    """Stock vLLM 0.32.0 DeepseekV4ScalingRotaryEmbedding.cos_sin_cache (fp32 [P, 64] = cos | sin), computed the same
    way on the GPU (vLLM builds it under the CUDA default device). Formula copied from
    model_executor/layers/rotary_embedding/{deepseek_scaling_rope.py,common.py}; ratio 0 = plain theta 10000."""
    dim = 64
    if ratio:
        base, factor, orig = 160000.0, 16.0, 65536
    else:
        base, factor, orig = 10000.0, 1.0, max_pos
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float, device=dev) / dim)
    inv_extra = 1.0 / pos_freqs
    inv_inter = 1.0 / (factor * pos_freqs)

    def corr_dim(rot):
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))
    low, high = max(math.floor(corr_dim(32)), 0), min(math.ceil(corr_dim(1)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float, device=dev) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    inv_freq = inv_inter * (1 - mask) + inv_extra * mask
    t = torch.arange(int(orig * factor) if ratio else max_pos, device=dev, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos() * 1.0, freqs.sin() * 1.0), dim=-1).contiguous()


# ----------------------------------------------------------------------------------------------- paged V4 caches
class PagedV4:
    """uint8 [n_pages, PAGE]: token data rows [64 x 576] then scale rows [64 x 8] per page (cache_utils.py)."""

    def __init__(self, n_pages: int, dev):
        self.n = n_pages
        self.buf = torch.zeros(n_pages, PAGE, dtype=torch.uint8, device=dev)
        self.data = self.buf[:, : BS * 576].unflatten(1, (BS, 576))
        self.scl = self.buf[:, BS * 576 :].unflatten(1, (BS, 8))

    def write(self, slots: torch.Tensor, kv: torch.Tensor):
        d, s = Q.v4_pack(kv)
        pg, ps = slots // BS, slots % BS
        self.data[pg, ps] = d
        self.scl[pg, ps] = s

    def gather(self, slots: torch.Tensor) -> torch.Tensor:
        """slots (any shape, >= 0) -> dequantized fp32 [..., 512]."""
        s = slots.reshape(-1).long()
        out = Q.v4_unpack(self.data[s // BS, s % BS], self.scl[s // BS, s % BS])
        return out.reshape(*slots.shape, 512)


def realistic_kv(n: int, w_norm: torch.Tensor, pos: torch.Tensor, ratio: int, gen) -> torch.Tensor:
    """post-RMSNorm, post-RoPE KV rows like stock produces (bf16 values as fp32)."""
    x = torch.randn(n, 512, device=w_norm.device, generator=gen)
    x = L.rmsnorm(x, w_norm, 1e-20, L.VLLM_SM90)
    return Q.bf16_round(L.rope_interleaved(x, pos, ratio))


def synth_topk(n_rows: list[int], gen, beta=0.7, rho=0.3, gam=2.0, tau=3000.0, k=512) -> torch.Tensor:
    """4 rows of one sequence: top-k of correlated scores (shared + AR(1)-across-rows parts + recency bonus).
    Calibrated (n = 62,453) to union 1,225 / membership {1: 743, 2: 240, 3: 143, 4: 99} vs the probe
    dumps' 1,058-1,281 / e.g. {1: 848, 2: 203, 3: 126, 4: 104}."""
    dev = gen.device
    n = max(n_rows)
    z0 = torch.randn(n, device=dev, generator=gen)
    u = torch.randn(n, device=dev, generator=gen)
    rec = gam * torch.exp(-(n - 1 - torch.arange(n, device=dev, dtype=torch.float32)) / tau)
    out = torch.full((4, k), -1, dtype=torch.int32, device=dev)
    for r in range(4):
        if r:
            u = rho * u + math.sqrt(1 - rho * rho) * torch.randn(n, device=dev, generator=gen)
        s = math.sqrt(beta) * z0 + math.sqrt(1 - beta) * u + rec
        nr = n_rows[r]
        kk = min(k, nr)
        out[r, :kk] = torch.topk(s[:nr], kk).indices.to(torch.int32)
    return out


def real_topk_groups():
    """The local probe dumps: 4-row groups (one sequence) of the ratio-1 index layers, n = 62,453."""
    import collections
    import glob
    g = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(DUMPS, "*.pt"))):
        d = torch.load(f, map_location="cpu")
        g[(round(float(d["time"]), 2), d["layer"], d["rows"], d["row"] // 4)][d["row"] % 4] = (d["sel"], d["n"], d["layer"])
    out = []
    for k, v in sorted(g.items()):
        if len(v) == 4 and all((v[r][0] >= 0).sum() == 512 for r in range(4)) and v[0][1] > 60000:
            out.append((torch.stack([v[r][0].to(torch.int32) for r in range(4)]), [v[r][1] for r in range(4)], v[0][2]))
    return out


# ----------------------------------------------------------------------------------------------- measurement
class Meter:
    def __init__(self, dev):
        self.h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            pr = torch.cuda.get_device_properties(dev)
            bus = f"{pr.pci_domain_id:08x}:{pr.pci_bus_id:02x}:{pr.pci_device_id:02x}.0"
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByPciBusId(bus)
        except Exception as ex:  # optional
            print(f"NVML unavailable: {ex}")

    def time(self, fn, reps: int, warm: int = 5) -> float:
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(reps):
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    def graph_time(self, fn, reps: int, inner: int = 20) -> tuple[float, object]:
        """us per call of fn replayed `inner` times inside one CUDA graph (no launch gaps from Python)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                fn()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(reps):
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3 / inner)
        ts.sort()
        return ts[len(ts) // 2], g

    def power(self, graph, inner: int, secs: float) -> dict | None:
        if self.h is None:
            return None
        nv = self.nv
        torch.cuda.synchronize()
        e0, t0, clk, n = nv.nvmlDeviceGetTotalEnergyConsumption(self.h), time.time(), [], 0
        while time.time() - t0 < secs:
            for _ in range(4):
                graph.replay()
            n += 4 * inner
            clk.append(nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_SM))
            torch.cuda.synchronize()
        el = time.time() - t0
        J = (nv.nvmlDeviceGetTotalEnergyConsumption(self.h) - e0) / 1000.0
        return dict(W=J / el, mhz=sum(clk) / len(clk), mhz_min=min(clk), J_per_call=J / n, us_wall=el / n * 1e6)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--seqs", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=60000)
    ap.add_argument("--ctx-jitter", type=int, default=8000)
    ap.add_argument("--topk", choices=["synth", "real"], default="synth")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--energy-secs", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mn-lbo", type=int, default=8192)
    ap.add_argument("--mn-sbo", type=int, default=1024)
    ap.add_argument("--no-flashmla", action="store_true")
    ap.add_argument("--only", default="k2,k3,k4,sp,v3,today", help="subset of k2,k3,k4,sp,v3,today")
    ap.add_argument("--out", default="")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--prof", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    gcpu = torch.Generator().manual_seed(a.seed)
    t_start = time.time()
    k3, kd = build()
    print(f"built extensions in {time.time() - t_start:.0f} s", flush=True)
    S, T = a.seqs, 4 * a.seqs
    res = dict(layer=a.layer, seqs=S, T=T, ctx=a.ctx, ctx_jitter=a.ctx_jitter, topk=a.topk, seed=a.seed,
               gpu=torch.cuda.get_device_name(dev), started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # ---- weights (TP rank 0), dequantized fp32 by the reference loader, bf16 copies must be exact
    rd = CheckpointReader(MODEL)
    W = L.load_layer_weights(rd, a.layer, rank=0, device=dev, experts="none")
    ratio = W.compress_ratio
    assert ratio == 1 or ratio == 2, f"layer {a.layer} is not a compressed (consumer) layer"
    wqb, woa, wob = W.wq_b.contiguous(), W.wo_a.contiguous(), W.wo_b.contiguous()
    wqb16, woa16, wob16 = wqb.to(torch.bfloat16), woa.to(torch.bfloat16), wob.to(torch.bfloat16)
    exact = all(bool((x16.float() == x).all()) for x16, x in ((wqb16, wqb), (woa16, woa), (wob16, wob)))
    sc = {}
    for nm in ("wq_b", "wo_a", "wo_b"):
        s8 = rd.get_tp(f"layers.{a.layer}.attn.{nm}.scale", 0).view(torch.uint8)
        sc[nm] = (int(s8.min()), int(s8.max()))
    res["weights"] = dict(bf16_exact=exact, scale_bytes=sc, wq_b=list(wqb.shape), wo_a=list(woa.shape), wo_b=list(wob.shape))
    print(f"layer {a.layer} ratio {ratio}: weights bf16-exact {exact}, scale bytes {sc}", flush=True)
    woaT16 = woa16.transpose(1, 2).contiguous()  # [G, 4096, 1024]
    sink = W.attn_sink.float().contiguous()

    # ---- RoPE cache (stock) and the reference patched to use it
    cs = stock_cos_sin(ratio, dev, 0)
    cs_ref_native = None

    def rope_cs_stock(positions, compress_ratio, cfg=L.CFG):
        v = cs[positions.long()]
        return v[..., :32], v[..., 32:]

    # ---- sequences, positions
    ctx = (a.ctx + torch.randint(-a.ctx_jitter, a.ctx_jitter + 1, (S,), generator=gcpu)).tolist()
    topk_real = None
    if a.topk == "real":
        topk_real = real_topk_groups()
        assert topk_real, "no real 4-row groups found"
        ctx = [topk_real[s % len(topk_real)][1][0] - 1 + 0 for s in range(S)]  # row 0 sees n = ctx + 1 rows
        res["real_groups"] = [(gr[2], gr[1]) for gr in topk_real]
    pos0 = torch.tensor(ctx, dtype=torch.int64)
    positions = (pos0.view(S, 1) + torch.arange(4).view(1, 4)).reshape(-1).to(dev)
    # native reference cos/sin vs the stock cache on these positions (information only)
    c_nat, s_nat = L.rope_cos_sin(positions, ratio)
    cs_ref_native = dict(cos_max_abs_diff=(c_nat - cs[positions][:, :32]).abs().max().item(),
                         sin_max_abs_diff=(s_nat - cs[positions][:, 32:]).abs().max().item())
    L.rope_cos_sin = rope_cs_stock
    res["rope_cache_vs_ref_native"] = cs_ref_native

    # ---- SWA cache: pages covering [p0 - 127, p0 + 3] per sequence, random physical placement
    n_lp = (max(ctx) + 4) // BS + 1
    swa_bt = torch.full((S, n_lp), 0, dtype=torch.int32)
    need = []
    for s in range(S):
        lo, hi = max(ctx[s] - (WINDOW - 1), 0), ctx[s] + 3
        need += [(s, lp) for lp in range(lo // BS, hi // BS + 1)]
    n_swa = max(4 * len(need), 1024)
    perm = torch.randperm(n_swa, generator=gcpu)[: len(need)]
    for (s, lp), ph in zip(need, perm.tolist()):
        swa_bt[s, lp] = ph
    swa = PagedV4(n_swa, dev)
    swa_bt_d = swa_bt.to(dev)
    # history rows p0 - 127 .. p0 - 1
    hist_pos = torch.cat([torch.arange(max(c - (WINDOW - 1), 0), c) for c in ctx]).to(dev)
    hist_seq = torch.cat([torch.full((c - max(c - (WINDOW - 1), 0),), s) for s, c in enumerate(ctx)]).to(dev)
    hist_slots = swa_bt_d[hist_seq, hist_pos // BS].long() * BS + hist_pos % BS
    swa.write(hist_slots, realistic_kv(hist_pos.numel(), W.kv_norm, hist_pos, ratio, gen))
    slot_mapping = swa_bt_d[torch.arange(S, device=dev).repeat_interleave(4), positions // BS].long() * BS + positions % BS

    # ---- compressed cache: rows [0, (p0 + 4) // ratio) per sequence (the source layer already wrote this step's)
    ncomp = [(c + 4) // ratio for c in ctx]
    n_cp = [(n + BS - 1) // BS for n in ncomp]
    tot = sum(n_cp)
    comp_bt = torch.zeros(S, max(n_cp), dtype=torch.int32)
    perm = torch.randperm(tot, generator=gcpu)
    o = 0
    for s in range(S):
        comp_bt[s, : n_cp[s]] = perm[o : o + n_cp[s]].to(torch.int32)
        o += n_cp[s]
    comp = PagedV4(tot, dev)
    comp_bt_d = comp_bt.to(dev)
    t0 = time.time()
    for s in range(S):
        idx = torch.arange(ncomp[s], device=dev)
        slots = comp_bt_d[s, idx // BS].long() * BS + idx % BS
        comp.write(slots, realistic_kv(ncomp[s], W.kv_norm, idx * ratio, ratio, gen))
    print(f"compressed cache {comp.buf.numel() / 1e9:.2f} GB ({tot} pages) filled in {time.time() - t0:.0f} s", flush=True)
    res["cache"] = dict(comp_GB=comp.buf.numel() / 1e9, comp_pages=tot, swa_pages=n_swa)

    # ---- top-k (local compressed indices per row)
    topk = torch.full((T, 512), -1, dtype=torch.int32, device=dev)
    for s in range(S):
        nr = [(ctx[s] + r + 1) // ratio for r in range(4)]
        if topk_real is not None:
            tk = topk_real[s % len(topk_real)][0].to(dev)
        else:
            tk = synth_topk(nr, gen)
        topk[4 * s : 4 * s + 4] = tk
    # union statistics
    us, hist = [], [0, 0, 0, 0]
    tk_cpu = topk.cpu()
    for s in range(S):
        rows = [set(x for x in tk_cpu[4 * s + r].tolist() if x >= 0) for r in range(4)]
        un = set().union(*rows)
        us.append(len(un))
        for x in un:
            hist[sum(x in rr for rr in rows) - 1] += 1
    res["union"] = dict(mean=sum(us) / S, min=min(us), max=max(us), membership=[h / S for h in hist])
    print(f"top-k union per sequence: mean {sum(us) / S:.0f} min {min(us)} max {max(us)}; membership {[round(h / S) for h in hist]}",
          flush=True)

    # ---- inputs of the layer's attention side (synthetic, realistic magnitudes)
    qr = Q.bf16_round(L.rmsnorm(torch.randn(T, 1280, device=dev, generator=gen), W.q_norm, 1e-20, L.VLLM_SM90)).to(torch.bfloat16)
    kv_raw = realistic_kv(T, W.kv_norm, positions, ratio, gen)
    kv16 = kv_raw.to(torch.bfloat16).contiguous()
    meter = Meter(dev)
    tim = {}

    def measure(name, fn, inner=20):
        us_ev = meter.time(fn, a.reps)
        us_g, g = meter.graph_time(fn, max(10, a.reps // 2), inner)
        pw = meter.power(g, inner, a.energy_secs)
        tim[name] = dict(us_events=us_ev, us_graph=us_g, **(pw or {}))
        print(f"  {name:28s} {us_g:8.2f} us (graph) {us_ev:8.2f} us (events)" +
              (f"  {pw['W']:.0f} W  {pw['mhz']:.0f} MHz  {pw['J_per_call'] * 1e3:.3f} mJ" if pw else ""), flush=True)

    # ================================================================ K2
    q = torch.empty(T, 16, 512, dtype=torch.bfloat16, device=dev)
    q_lin = F.linear(qr, wqb16).view(T, 16, 512)
    q.copy_(q_lin)
    kd.rope_q_insert(q, kv16, positions, cs, slot_mapping, swa.buf, PAGE, 6, True)
    torch.cuda.synchronize()
    with L.strict_fp32():
        q_lin_ref = Q.bf16_round(qr.float() @ wqb.t()).view(T, 16, 512)
        q_ref = Q.bf16_round(L.rope_interleaved(q_lin_ref, positions, ratio)).to(torch.bfloat16)
    chk = dict(q_lin=ulp_stats(q_lin, q_lin_ref.to(torch.bfloat16)), q=ulp_stats(q, q_ref))
    print(f"K2 q row-scaled max {chk['q']['row_ulp_max']:.3f} ulp", flush=True)
    # window KV record bytes at this step's slots vs the reference packer
    d_ref, s_ref = Q.v4_pack(kv_raw)
    sm = slot_mapping
    rec_eq = (swa.data[sm // BS, sm % BS] == d_ref).all(-1) & (swa.scl[sm // BS, sm % BS] == s_ref).all(-1)
    chk["kv_records_bit_exact"] = rec_eq.float().mean().item()
    # RoPE alone (same q_lin input): isolates the epilogue from the GEMM summation order
    q2 = q_lin.clone()
    kd.rope_q_insert(q2, kv16, positions, cs, slot_mapping, swa.buf, PAGE, 6, False)
    chk["rope_only"] = ulp_stats(q2, Q.bf16_round(L.rope_interleaved(q_lin.float(), positions, ratio)).to(torch.bfloat16))
    res["k2_check"] = chk
    print(f"K2 check: q_lin {chk['q_lin']['exact']:.4f} exact max {chk['q_lin']['max']} ulp; q {chk['q']['exact']:.4f} "
          f"max {chk['q']['max']}; RoPE-only max {chk['rope_only']['max']}; KV records bit-exact {chk['kv_records_bit_exact']:.4f}",
          flush=True)
    if "k2" in a.only:
        qtmp = torch.empty_like(q)
        print("K2 timing:")
        measure("k2.q_b_gemm", lambda: torch.matmul(qr, wqb16.t(), out=qtmp.view(T, 8192)))
        measure("k2.rope_q_insert", lambda: kd.rope_q_insert(qtmp, kv16, positions, cs, slot_mapping, swa.buf, PAGE, 6, True))

        def k2_all():
            torch.matmul(qr, wqb16.t(), out=qtmp.view(T, 8192))
            kd.rope_q_insert(qtmp, kv16, positions, cs, slot_mapping, swa.buf, PAGE, 6, True)
        measure("k2.total", k2_all)

    # ================================================================ K3
    KST = 2240
    codes = torch.zeros(S, KST, dtype=torch.int32, device=dev)
    addr = torch.zeros(S, KST, 2, dtype=torch.int64, device=dev)
    nkeys = torch.zeros(S, dtype=torch.int32, device=dev)
    zero_rec = torch.zeros(1024, dtype=torch.uint8, device=dev)

    def run_meta():
        k3.meta(topk, positions, None, swa_bt_d, 6, comp_bt_d, 6, WINDOW, codes, nkeys, swa.buf, PAGE, comp.buf, PAGE, zero_rec, addr)
    run_meta()
    torch.cuda.synchronize()
    nk = nkeys.cpu()
    res["keys_per_seq"] = dict(mean=nk.float().mean().item(), max=int(nk.max()), min=int(nk.min()),
                               tiles_max=int((nk.max() + 63) // 64))
    print(f"K3 key list: {nk.float().mean():.0f} keys per sequence (max {int(nk.max())})", flush=True)
    # metadata check against a CPU rebuild for a few sequences
    meta_ok = True
    for s in range(min(S, 8)):
        c = codes[s, : nk[s]].cpu()
        got = {(int(x) >> 5, (int(x) >> 4) & 1): int(x) & 15 for x in c.tolist()}
        exp = {}
        for r in range(4):
            p = ctx[s] + r
            for ps in range(max(p - (WINDOW - 1), 0), p + 1):
                sl = int(swa_bt[s, ps // BS]) * BS + ps % BS
                exp[(sl, 0)] = exp.get((sl, 0), 0) | (1 << r)
            for ix in tk_cpu[4 * s + r].tolist():
                if ix >= 0:
                    sl = int(comp_bt[s, ix // BS]) * BS + ix % BS
                    exp[(sl, 1)] = exp.get((sl, 1), 0) | (1 << r)
        meta_ok &= got == exp and len(c) == len(exp)
        ad = addr[s, : nk[s]].cpu()
        for i in range(0, int(nk[s]), 97):
            x = int(c[i])
            sl, isc = x >> 5, (x >> 4) & 1
            base = (comp.buf if isc else swa.buf).data_ptr() + (sl // BS) * PAGE
            meta_ok &= int(ad[i, 0]) == base + (sl % BS) * 576 and int(ad[i, 1]) == base + BS * 576 + (sl % BS) * 8
        npad = (int(nk[s]) + 63) // 64 * 64
        meta_ok &= bool((codes[s, nk[s]:npad] == 0).all()) and bool((addr[s, nk[s]:npad] == zero_rec.data_ptr()).all())
    res["meta_check_8seq"] = bool(meta_ok)
    print(f"K3 metadata check (8 sequences vs CPU): {meta_ok}", flush=True)

    out = torch.empty(T, 16, 512, dtype=torch.bfloat16, device=dev)
    out_rot = torch.empty_like(out)
    dbg = torch.zeros(S, 64, 64, dtype=torch.float32, device=dev)
    scale = 512 ** -0.5
    k3.attn(q, out, codes, addr, nkeys, sink, scale, None, None, a.mn_lbo, a.mn_sbo, dbg, None, 0)
    k3.attn(q, out_rot, codes, addr, nkeys, sink, scale, positions, cs, a.mn_lbo, a.mn_sbo, None, None, 0)
    torch.cuda.synchronize()
    # reference: per-row key rows (window 128 + top-512), dequantized from the same pages
    j = torch.arange(WINDOW, device=dev)
    wpos = positions.view(T, 1) - (WINDOW - 1) + j.view(1, WINDOW)
    wval = wpos >= 0
    seq_of = torch.arange(S, device=dev).repeat_interleave(4)
    wsl = swa_bt_d[seq_of.view(T, 1), wpos.clamp_min(0) // BS].long() * BS + wpos.clamp_min(0) % BS
    tval = topk >= 0
    tix = topk.clamp_min(0).long()
    tsl = comp_bt_d[seq_of.view(T, 1), tix // BS].long() * BS + tix % BS
    with L.strict_fp32():
        rows = torch.cat([swa.gather(wsl), comp.gather(tsl)], 1)
        valid = torch.cat([wval, tval], 1)
        rows = rows * valid.unsqueeze(-1)
        o_ref32 = L.sparse_attn_ref(q.float(), rows, sink, valid, scale)
        o_ref = o_ref32.to(torch.bfloat16)
        o_rot_ref = Q.bf16_round(L.rope_interleaved(o_ref.float(), positions, ratio, inverse=True)).to(torch.bfloat16)
        # QK check on tile 0 (raw dot products of the first 64 keys of each sequence's list)
        kl = []
        for s in range(min(S, 4)):
            c = codes[s, :64].cpu().long()
            sl = c >> 5
            isc = ((c >> 4) & 1).bool()
            kr = torch.zeros(64, 512, device=dev)
            if (~isc).any():
                kr[~isc] = swa.gather(sl[~isc].to(dev))
            if isc.any():
                kr[isc] = comp.gather(sl[isc].to(dev))
            kl.append((q[4 * s : 4 * s + 4].float().reshape(64, 512) @ kr.t()))
        s_ref = torch.stack(kl)
    s_err = ((dbg[: len(kl)] - s_ref).abs().max() / s_ref.abs().max()).item()
    chk3 = dict(qk_tile0_rel_err=s_err, o=ulp_stats(out, o_ref), o_rot_fused=ulp_stats(out_rot, o_rot_ref))
    print(f"K3 check: QK tile 0 rel err {s_err:.2e}; o exact {chk3['o']['exact']:.4f} <=1 {chk3['o']['le1']:.4f} "
          f"p99.9 {chk3['o']['p999']:.0f} ulp, row-scaled max {chk3['o']['row_ulp_max']:.2f} p99.9 {chk3['o']['row_ulp_p999']:.3f}; "
          f"fused o_rot exact {chk3['o_rot_fused']['exact']:.4f} row-scaled max {chk3['o_rot_fused']['row_ulp_max']:.2f}",
          flush=True)
    # standalone inverse RoPE kernel on the unfused output
    o_inv = out.clone()
    kd.inv_rope(o_inv, positions, cs)
    chk3["inv_rope_standalone_vs_fused_equal"] = bool((o_inv == out_rot).all())

    # stock FlashMLA sparse decode on the same pages (q padded to 64 heads as stock does)
    fm = None
    if not a.no_flashmla:
        try:
            from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache, get_mla_metadata
            qp = torch.zeros(T, 1, 64, 512, dtype=torch.bfloat16, device=dev)
            qp[:, 0, :16] = q
            sinkp = torch.zeros(64, dtype=torch.float32, device=dev)
            sinkp[:16] = sink
            swa_idx = torch.where(wval, wsl, torch.full_like(wsl, -1)).to(torch.int32).view(T, 1, WINDOW)
            ext_idx = torch.where(tval, tsl, torch.full_like(tsl, -1)).to(torch.int32).view(T, 1, 512)
            swa_len = wval.sum(1).to(torch.int32)
            ext_len = tval.sum(1).to(torch.int32)
            fout = torch.empty(T, 1, 64, 512, dtype=torch.bfloat16, device=dev)
            meta_fm = get_mla_metadata()[0]
            kc0 = swa.buf.view(n_swa, BS, 1, 584)
            kc1 = comp.buf.view(tot, BS, 1, 584)

            def run_fm():
                flash_mla_with_kvcache(q=qp, k_cache=kc0, block_table=None, cache_seqlens=None, head_dim_v=512,
                                       tile_scheduler_metadata=meta_fm, softmax_scale=scale, is_fp8_kvcache=True,
                                       indices=swa_idx, topk_length=swa_len, attn_sink=sinkp, extra_k_cache=kc1,
                                       extra_indices_in_kvcache=ext_idx, extra_topk_length=ext_len, out=fout)
            run_fm()
            torch.cuda.synchronize()
            fm = run_fm
            fo = fout[:, 0, :16].contiguous()
            chk3["stock_flashmla_o"] = ulp_stats(fo, o_ref)
            chk3["cand_vs_stock_o"] = ulp_stats(out, fo)
            sf = chk3["stock_flashmla_o"]
            print(f"stock FlashMLA check: o exact {sf['exact']:.4f} <=1 {sf['le1']:.4f} p99.9 {sf['p999']:.0f} ulp, row-scaled max "
                  f"{sf['row_ulp_max']:.2f} p99.9 {sf['row_ulp_p999']:.3f}; candidate vs stock exact {chk3['cand_vs_stock_o']['exact']:.4f}",
                  flush=True)
            fo_rot = fo.clone()
            kd.inv_rope(fo_rot, positions, cs)
        except Exception as ex:
            print(f"stock FlashMLA failed: {type(ex).__name__}: {ex}", flush=True)
            chk3["stock_flashmla_error"] = f"{type(ex).__name__}: {ex}"
    # production's ds41h kernel (sparse_decode.cu, cluster per row, 16 heads) on the same pages, global-index mode
    sd = None
    if "today" in a.only:
        try:
            sdx = build_today()
            swa_idx2 = torch.where(wval, wsl, torch.full_like(wsl, -1)).to(torch.int32).contiguous()
            ext_idx2 = torch.where(tval, tsl, torch.full_like(tsl, -1)).to(torch.int32).contiguous()
            swa_len2, ext_len2 = wval.sum(1).to(torch.int32), tval.sum(1).to(torch.int32)
            sdo = torch.empty_like(out)
            c0v, c1v = swa.buf.view(n_swa, BS, 584), comp.buf.view(tot, BS, 584)

            def run_sd():  # positional as ds41h/attn.py: cluster 0 = auto, spread 1, pdl 0
                sdx.sparse_decode(q, sdo, c0v, swa_idx2, swa_len2, sink, float(scale), c1v, ext_idx2, ext_len2, None, None, None,
                                  0, True, False, None, None, 0)
            run_sd()
            torch.cuda.synchronize()
            sd = run_sd
            chk3["today_ds41h_o"] = ulp_stats(sdo, o_ref)
            print(f"today ds41h sparse_decode check: o exact {chk3['today_ds41h_o']['exact']:.4f} <=1 "
                  f"{chk3['today_ds41h_o']['le1']:.4f} row-scaled max {chk3['today_ds41h_o']['row_ulp_max']:.2f}", flush=True)
        except Exception as ex:
            print(f"ds41h sparse_decode failed: {type(ex).__name__}: {ex}", flush=True)
            chk3["today_error"] = f"{type(ex).__name__}: {ex}"
    res["k3_check"] = chk3
    if "k3" in a.only:
        print("K3 timing:")
        measure("k3.meta", run_meta)
        measure("k3.attn", lambda: k3.attn(q, out, codes, addr, nkeys, sink, scale, None, None,
                                            a.mn_lbo, a.mn_sbo, None, None, 0))
        measure("k3.attn_fused_invrope", lambda: k3.attn(q, out_rot, codes, addr, nkeys, sink, scale,
                                                         positions, cs, a.mn_lbo, a.mn_sbo, None, None, 0))
        if a.ablate:
            ob = torch.empty_like(out)
            for ab in (1, 2, 3, 4):
                measure(f"k3.attn_abl{ab}", lambda ab=ab: k3.attn(q, ob, codes, addr, nkeys, sink, scale,
                                                                  None, None, a.mn_lbo, a.mn_sbo, None, None, ab))
        if a.prof:
            k3p = build_prof()
            prof = torch.zeros(S * 40 * 2 * 8, dtype=torch.int64, device=dev)
            k3p.attn(q, ob if a.ablate else torch.empty_like(out), codes, addr, nkeys, sink, scale,
                     None, None, a.mn_lbo, a.mn_sbo, None, prof, 0)
            torch.cuda.synchronize()
            pr = prof.view(S, 40, 2, 8).cpu().double()
            nt = int((nk.max() + 63) // 64)
            # v2 stamps per iteration (cycles from WG0's iteration start): WG0 [PV_i + S_i+1 done, softmax_i+1 done],
            # WG1 [PV_i done, gather_i+2 done], bar 1 (WG0)
            prow = []
            for s_ in range(min(S, 4)):
                for it in range(min(nt, 6)):
                    b0 = pr[s_, it, 0, 0]
                    prow.append([s_, it] + [int(v) for v in (pr[s_, it, 0, 1] - b0, pr[s_, it, 0, 2] - b0, pr[s_, it, 1, 1] - b0,
                                                              pr[s_, it, 1, 2] - b0, pr[s_, it, 0, 3] - b0)])
            per_tile = (pr[:, 1:nt - 2, 0, 0].diff(dim=1)).mean().item()
            print("  prof [seq, tile, wg0 pv+qk done, wg0 softmax done, wg1 pv done, wg1 gather done, bar1] cycles:")
            for r in prow:
                print("   ", r)
            print(f"  mean cycles per tile {per_tile:.0f}", flush=True)
            res["k3_prof"] = dict(rows=prow, cycles_per_tile=per_tile)
        if fm is not None:
            measure("k3.stock_flashmla_core", fm)
        if sd is not None:
            measure("k3.today_ds41h_sparse_decode", sd)
        res["k3_bytes"] = dict(union_MB=float(nk.sum()) * 584 / 1e6, per_row_MB=T * 640 * 584 / 1e6)

    # ================================================================ K3 v3 (TMA gather, Q in registers, 3 warpgroups)
    k3v3 = None
    if "v3" in a.only:
        k3v3 = build3()
        out3 = torch.empty_like(out)
        out3r = torch.empty_like(out)
        dbg3 = torch.zeros_like(dbg)
        k3v3.attn3(q, out3, codes, addr, nkeys, sink, scale, None, None, a.mn_lbo, a.mn_sbo, dbg3, None)
        k3v3.attn3(q, out3r, codes, addr, nkeys, sink, scale, positions, cs, a.mn_lbo, a.mn_sbo, None, None)
        torch.cuda.synchronize()
        s3 = ((dbg3[: len(kl)] - s_ref).abs().max() / s_ref.abs().max()).item()
        c3 = dict(qk_tile0_rel_err=s3, o=ulp_stats(out3, o_ref), o_rot_fused=ulp_stats(out3r, o_rot_ref),
                  v3_vs_v2_bit_equal=float((out3 == out).float().mean().item()))
        print(f"K3 v3 check: QK rel err {s3:.2e}; o exact {c3['o']['exact']:.4f} <=1 {c3['o']['le1']:.4f} row-scaled max "
              f"{c3['o']['row_ulp_max']:.2f}; o_rot row-scaled max {c3['o_rot_fused']['row_ulp_max']:.2f}; equal to v2 "
              f"{c3['v3_vs_v2_bit_equal']:.4f}", flush=True)
        res["k3v3_check"] = c3
        measure("k3v3.attn", lambda: k3v3.attn3(q, out3, codes, addr, nkeys, sink, scale, None, None, a.mn_lbo, a.mn_sbo, None, None))
        measure("k3v3.attn_fused_invrope", lambda: k3v3.attn3(q, out3r, codes, addr, nkeys, sink, scale, positions, cs, a.mn_lbo,
                                                             a.mn_sbo, None, None))
        if a.prof:
            k3pp = build3(prof=True)
            prof3 = torch.zeros(S * 40 * 3 * 8, dtype=torch.int64, device=dev)
            k3pp.attn3(q, out3, codes, addr, nkeys, sink, scale, None, None, a.mn_lbo, a.mn_sbo, None, prof3)
            torch.cuda.synchronize()
            pr = prof3.view(S, 40, 3, 8).cpu().double()
            nt = int((nk.max() + 63) // 64)
            print("  v3 prof [seq, tile, WG0 S done, WG0 softmax done, WG1 PV done, WG1 dequant done, WG2 PV done] cycles from WG0 B1:")
            for s_ in range(2):
                for it in range(min(nt, 5)):
                    b0 = pr[s_, it, 0, 0]
                    print("   ", [s_, it] + [int(v) for v in (pr[s_, it, 0, 1] - b0, pr[s_, it, 0, 2] - b0, pr[s_, it, 1, 1] - b0,
                                                              pr[s_, it, 1, 2] - b0, pr[s_, it, 2, 1] - b0)])
            per_tile3 = (pr[:, 1:nt - 2, 0, 0].diff(dim=1)).mean().item()
            print(f"  v3 mean cycles per tile {per_tile3:.0f}", flush=True)
            res["k3v3_prof_cycles_per_tile"] = per_tile3

    # ================================================================ K3 in the sequence-parallel layout (experiment)
    # One CTA per decode row with ALL 64 heads (what a rank would run if attention were sequence-parallel: MQA KV is
    # replicated, so any rank can serve any row): 64 queries share all 640 keys, no masked MACs, no key union.
    if "sp" in a.only:
        nsp = min(T, 128)
        sink64 = rd.get(f"layers.{a.layer}.attn.attn_sink").float().to(dev).contiguous()
        g2 = torch.Generator(device=dev).manual_seed(a.seed + 7)
        qrms = q.float().pow(2).mean().sqrt()
        q_sp = (torch.randn(nsp, 64, 512, device=dev, generator=g2) * qrms).to(torch.bfloat16).contiguous()
        sl_sp = torch.cat([wsl[:nsp], tsl[:nsp]], 1)
        isc_sp = torch.cat([torch.zeros_like(wsl[:nsp]), torch.ones_like(tsl[:nsp])], 1)
        val_sp = torch.cat([wval[:nsp], tval[:nsp]], 1)
        codes_sp = torch.where(val_sp, (sl_sp << 5) | (isc_sp << 4) | 15, torch.zeros_like(sl_sp)).to(torch.int32).contiguous()
        base_sp = torch.where(isc_sp.bool(), torch.full_like(sl_sp, comp.buf.data_ptr()), torch.full_like(sl_sp, swa.buf.data_ptr()))
        base_sp = base_sp + (sl_sp // BS) * PAGE
        addr_sp = torch.stack([base_sp + (sl_sp % BS) * 576, base_sp + BS * 576 + (sl_sp % BS) * 8], -1)
        addr_sp = torch.where(val_sp.unsqueeze(-1), addr_sp, torch.full_like(addr_sp, zero_rec.data_ptr())).contiguous()
        nk_sp = torch.full((nsp,), 640, dtype=torch.int32, device=dev)
        out_sp = torch.empty(4 * nsp, 16, 512, dtype=torch.bfloat16, device=dev)
        qv = q_sp.view(4 * nsp, 16, 512)

        def run_sp():
            k3.attn(qv, out_sp, codes_sp, addr_sp, nk_sp, sink64, scale, None, None, a.mn_lbo, a.mn_sbo, None, None, 0)
        run_sp()
        torch.cuda.synchronize()
        with L.strict_fp32():
            o_sp_ref = L.sparse_attn_ref(q_sp.float(), rows[:nsp], sink64, valid[:nsp], scale).to(torch.bfloat16)
        sp = dict(rows=nsp, o=ulp_stats(out_sp.view(nsp, 64, 512), o_sp_ref))
        print(f"K3-SP check (1 row x 64 heads per CTA): o exact {sp['o']['exact']:.4f} <=1 {sp['o']['le1']:.4f} row-scaled max "
              f"{sp['o']['row_ulp_max']:.2f}", flush=True)
        measure("k3sp.attn_128rows_64heads", run_sp)
        if k3v3 is not None:
            out_sp3 = torch.empty_like(out_sp)

            def run_sp3():
                k3v3.attn3(qv, out_sp3, codes_sp, addr_sp, nk_sp, sink64, scale, None, None, a.mn_lbo, a.mn_sbo, None, None)
            run_sp3()
            torch.cuda.synchronize()
            sp["v3_o"] = ulp_stats(out_sp3.view(nsp, 64, 512), o_sp_ref)
            print(f"K3-SP v3 check: exact {sp['v3_o']['exact']:.4f} <=1 {sp['v3_o']['le1']:.4f} row-scaled max {sp['v3_o']['row_ulp_max']:.2f}",
                  flush=True)
            measure("k3sp.v3_attn_128rows_64heads", run_sp3)
        if fm is not None:
            from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache, get_mla_metadata
            fq = q_sp.view(nsp, 1, 64, 512)
            fo_sp = torch.empty(nsp, 1, 64, 512, dtype=torch.bfloat16, device=dev)
            meta_sp = get_mla_metadata()[0]
            si = swa_idx[:nsp].contiguous()
            ei = ext_idx[:nsp].contiguous()
            sln, eln = swa_len[:nsp].contiguous(), ext_len[:nsp].contiguous()

            def run_fm_sp():
                flash_mla_with_kvcache(q=fq, k_cache=kc0, block_table=None, cache_seqlens=None, head_dim_v=512,
                                       tile_scheduler_metadata=meta_sp, softmax_scale=scale, is_fp8_kvcache=True, indices=si,
                                       topk_length=sln, attn_sink=sink64, extra_k_cache=kc1, extra_indices_in_kvcache=ei,
                                       extra_topk_length=eln, out=fo_sp)
            run_fm_sp()
            torch.cuda.synchronize()
            sp["stock_flashmla_o"] = ulp_stats(fo_sp[:, 0].contiguous(), o_sp_ref)
            print(f"stock FlashMLA SP check: exact {sp['stock_flashmla_o']['exact']:.4f} <=1 {sp['stock_flashmla_o']['le1']:.4f}", flush=True)
            measure("k3sp.stock_flashmla_128rows_64heads", run_fm_sp)
        res["k3_sp"] = sp

    # ================================================================ K4
    o_in = out_rot  # K3 output with the fused inverse RoPE
    z = torch.empty(T, 2048, dtype=torch.bfloat16, device=dev)
    zb = torch.empty(2, T, 1024, dtype=torch.bfloat16, device=dev)
    y = torch.empty(T, 5120, dtype=torch.bfloat16, device=dev)

    o2 = o_in.view(T, 8192)

    def wo_a():
        for gi in range(2):
            torch.mm(o2[:, 4096 * gi : 4096 * (gi + 1)], woaT16[gi], out=z[:, 1024 * gi : 1024 * (gi + 1)])

    def wo_a_bmm_copy():
        torch.bmm(o_in.view(T, 2, 4096).transpose(0, 1), woaT16, out=zb)
        z.view(T, 2, 1024).copy_(zb.transpose(0, 1))

    def k4_gemms():
        wo_a_bmm_copy()
        torch.matmul(z, wob16.t(), out=y)
    k4_gemms()
    torch.cuda.synchronize()
    with L.strict_fp32():
        z_ref = Q.bf16_round(torch.einsum("tgd,grd->tgr", o_in.float().view(T, 2, 4096), woa)).flatten(1)
        y_ref = Q.bf16_round(z_ref @ wob.t())
        y_from_z = Q.bf16_round(z.float() @ wob.t())
    chk4 = dict(z=ulp_stats(z, z_ref.to(torch.bfloat16), 1024), y=ulp_stats(y, y_ref.to(torch.bfloat16), 5120),
                y_given_z=ulp_stats(y, y_from_z.to(torch.bfloat16), 5120))
    # the whole attention side through the reference (o_proj_ref on the reference attention output)
    with L.strict_fp32():
        full = L.o_proj_ref(W, o_ref.float().view(T, 16, 512), positions, n=L.VLLM_SM90)
    chk4["y_vs_ref_chain"] = ulp_stats(y, full["y"].to(torch.bfloat16), 5120)
    if fm is not None:  # stock's chain: FlashMLA o -> inverse RoPE -> reference wo_a / wo_b
        with L.strict_fp32():
            zs = Q.bf16_round(torch.einsum("tgd,grd->tgr", fo_rot.float().view(T, 2, 4096), woa)).flatten(1)
            ys = Q.bf16_round(zs @ wob.t()).to(torch.bfloat16)
        chk4["stock_chain_y_vs_ref_chain"] = ulp_stats(ys, full["y"].to(torch.bfloat16), 5120)
    res["k4_check"] = chk4
    print(f"K4 check: z exact {chk4['z']['exact']:.4f} max {chk4['z']['max']}; y exact {chk4['y']['exact']:.4f} "
          f"max {chk4['y']['max']}; y vs full ref chain <=1 {chk4['y_vs_ref_chain']['le1']:.4f} row-scaled max "
          f"{chk4['y_vs_ref_chain']['row_ulp_max']:.2f}" + (f"; stock chain <=1 {chk4['stock_chain_y_vs_ref_chain']['le1']:.4f} "
          f"row-scaled max {chk4['stock_chain_y_vs_ref_chain']['row_ulp_max']:.2f}" if "stock_chain_y_vs_ref_chain" in chk4 else ""),
          flush=True)
    if "k4" in a.only:
        print("K4 timing:")
        o_tmp = out.clone()
        measure("k4.inv_rope_standalone", lambda: kd.inv_rope(o_tmp, positions, cs))
        measure("k4.wo_a_2mm", wo_a)
        measure("k4.wo_a_bmm_plus_copy", wo_a_bmm_copy)
        measure("k4.wo_a_bmm_ref", lambda: torch.bmm(o_in.view(T, 2, 4096).transpose(0, 1), woaT16, out=zb))
        measure("k4.wo_b_gemm", lambda: torch.matmul(z, wob16.t(), out=y))
        measure("k4.total_gemms", k4_gemms)

    # ================================================================ chained attention side (one graph)
    if set(a.only.split(",")) >= {"k2", "k3", "k4"}:
        qc = torch.empty_like(q)

        def chain():
            torch.matmul(qr, wqb16.t(), out=qc.view(T, 8192))
            kd.rope_q_insert(qc, kv16, positions, cs, slot_mapping, swa.buf, PAGE, 6, True)
            k3.attn(qc, out_rot, codes, addr, nkeys, sink, scale, positions, cs, a.mn_lbo,
                    a.mn_sbo, None, None, 0)
            k4_gemms()
        print("chain (K2 + K3 attn fused + K4 GEMMs; metadata once per step, separate):")
        measure("chain.k2_k3_k4", chain, inner=10)
    res["timing"] = tim
    res["elapsed_s"] = time.time() - t_start
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k in ("k2_check", "k3_check", "k4_check")}, indent=None)[:4000])


if __name__ == "__main__":
    main()
