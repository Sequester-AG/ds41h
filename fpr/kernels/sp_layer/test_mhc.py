#!/usr/bin/env python3
"""K1 fused mHC boundary (mhc.cu) vs ds41h.ref.layer (VLLM_SM90 numerics, strict fp32) on layer weights from the
checkpoint, synthetic stream (or a capture file with --cap). Also a stock-like path (the 24-wide product in tf32, as
DeepGEMM's hc_prenorm GEMM) to show what stock itself differs by. Times R owned rows in a CUDA graph.

  PYTHONPATH=<repo root> python3 test_mhc.py --layer 21 --rows 128 [--sub attn|ffn]
  (ds41h.ref comes from production's ds41h package, ds41h/ at the repo root)
"""
import argparse
import os
import time

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))


def ext():
    defs = [f"-D{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("MHC_")]
    tag = "_".join(d[3:].replace("=", "") for d in defs) or "default"
    bd = os.path.join(HERE, "build_mhc", tag)
    os.makedirs(bd, exist_ok=True)
    return load(name=f"ds41h_phys_mhc_{tag}".replace("-", "_"), sources=[os.path.join(HERE, "mhc.cu")], build_directory=bd,
                verbose=False, extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"] + defs)


def ulps(a, b):
    """|a - b| in bf16 ulps of |b| (element) and of the row max (row-scaled)."""
    a, b = a.float(), b.float()
    d = (a - b).abs()
    ue = torch.exp2(torch.floor(torch.log2(b.abs().clamp_min(2.0 ** -100))) - 7)
    ur = torch.exp2(torch.floor(torch.log2(b.abs().flatten(1).amax(1).clamp_min(2.0 ** -100))) - 7)
    ur = ur.view(-1, *([1] * (b.dim() - 1)))
    e = d / ue
    return dict(exact=float((d == 0).float().mean()), le1=float((e <= 1).float().mean()), max_elem=float(e.max()),
                max_row=float((d / ur).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--sub", default="attn", choices=["attn", "ffn"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))
    a = ap.parse_args()
    from ds41h.ref import layer as L
    from ds41h.ref.weights import default_reader

    dev = torch.device("cuda", 0)
    rd = default_reader(a.model)
    g = lambda n: rd.get(f"layers.{a.layer}.{n}").to(dev).float()  # noqa: E731
    fn, sc, bs = g(f"hc_{a.sub}_fn"), g(f"hc_{a.sub}_scale"), g(f"hc_{a.sub}_base")
    nw = g("attn_norm.weight" if a.sub == "attn" else "ffn_norm.weight")
    cfg = L.ModelConfig.from_hf(rd.config)
    R, D, HC = a.rows, 5120, 4
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    y = (torch.randn(R, D, device=dev, generator=gen) * 0.5).bfloat16()
    rs = (torch.randn(R, HC, D, device=dev, generator=gen) * torch.rand(R, HC, 1, device=dev, generator=gen) * 4).bfloat16()
    post = 2 * torch.sigmoid(torch.randn(R, HC, device=dev, generator=gen))
    comb = L.sinkhorn(torch.randn(R, HC, HC, device=dev, generator=gen) * 2, 20, 1e-6)
    lag = torch.sigmoid(torch.randn(R, HC, device=dev, generator=gen)) + 1e-6
    n = L.VLLM_SM90
    with L.strict_fp32():
        xr = L.hc_post(y, rs, post, comb, n)
        pre_r, post_r, comb_r, mix_r = L.hc_mixes(xr, fn, sc, bs, norm_eps=cfg.norm_eps, hc_eps=cfg.hc_eps, iters=cfg.hc_iters)
        in_r = L.hc_collapse(xr, lag, n)
        out_r = L.rmsnorm(in_r, nw, cfg.norm_eps, n)
    # stock-like: same, but the 24-wide product in tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    xf = xr.flatten(-2)
    mix_s = (xf @ fn.t()) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.norm_eps)
    torch.backends.cuda.matmul.allow_tf32 = False
    pre_s, post_s, comb_s = L.hc_split_sinkhorn(mix_s, sc, bs, HC, cfg.hc_iters, cfg.hc_eps)

    e = ext()
    xn = torch.empty(R, HC, D, dtype=torch.bfloat16, device=dev)
    pre_o, post_o = torch.empty(R, HC, device=dev), torch.empty(R, HC, device=dev)
    comb_o, out = torch.empty(R, HC, HC, device=dev), torch.empty(R, D, dtype=torch.bfloat16, device=dev)
    part = torch.empty(160 * R * 25, device=dev)
    fnp = e.prep_fn(fn)
    run = lambda: e.boundary(y, rs, post, comb, lag, fnp, sc, bs, nw, cfg.norm_eps, cfg.hc_eps, cfg.hc_iters,  # noqa: E731
                             xn, pre_o, post_o, comb_o, out, part)
    run()
    torch.cuda.synchronize()
    rel = lambda u, v: float(((u - v).abs() / v.abs().clamp_min(1e-30)).max())  # noqa: E731
    print(f"layer {a.layer} {a.sub}, R={R}, norm_eps {cfg.norm_eps}, hc_eps {cfg.hc_eps}, iters {cfg.hc_iters}")
    print(f"  Xn (hc_post, bf16)          vs ref: {ulps(xn, xr)}")
    print(f"  pre/post/comb max rel       vs ref: {rel(pre_o, pre_r):.2e} / {rel(post_o, post_r):.2e} / {rel(comb_o, comb_r):.2e}"
          f"   (stock tf32 vs ref: {rel(pre_s, pre_r):.2e} / {rel(post_s, post_r):.2e} / {rel(comb_s, comb_r):.2e})")
    print(f"  out (collapse+RMSNorm, bf16) vs ref: {ulps(out, out_r)}")
    # timing
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        run()
    for _ in range(10):
        gr.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(200):
        gr.replay()
    t1.record()
    torch.cuda.synchronize()
    us = t0.elapsed_time(t1) / 200 * 1000
    mb = (R * D * 2 + R * HC * D * 2 * 3 + R * D * 2) / 1e6
    print(f"  time {us:.1f} us per boundary ({mb:.1f} MB moved, {mb / us:.2f} TB/s effective)")
    ycl = rs.clone()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            run()
            ycl.copy_(rs)
        torch.cuda.synchronize()
    for ev in prof.key_averages():
        if ev.device_type is not None and ("mhc" in ev.key or "copy" in ev.key.lower() or "elementwise" in ev.key.lower()):
            print(f"    {ev.key[:60]:60s} {ev.device_time_total / ev.count:8.1f} us x {ev.count}")


if __name__ == "__main__":
    main()
