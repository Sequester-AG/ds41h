"""Shared harness for the prefill-grade routed MoE (kernels/moe_prefill): weights, inputs, routing, timing, energy,
fp64 reference and error statistics. One GPU, layer 21 rank 0 (real weights from the phys cache).

Inputs: x = stock's captured MoE input rows f of layer 21 (a local capture directory (CAP_DIR), 3 decode steps, ~1.4K
real rows), tiled to T rows. Routing:
  synth : bench.synth_routing (skewed Gamma(0.47) expert popularity, 4-row groups sharing 60 % of their experts)
  cap   : the layer's real router (ref gate: sqrt-softplus fp32, bias top-6, x 1.5) on the tiled real rows
          (the expert histogram of the real decode traffic, scaled by T / 1.4K)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHYS = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PHYS, "kernels", "moe_stream"))
import bench  # noqa: E402

H, I, E, TOPK = 5120, 576, 384, 6
MODEL = os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model")
CACHE = os.environ.get("MOE_LAYER_CACHE", os.path.join(PHYS, "cache", "layer21_r0.pt"))
CAP = os.environ.get("CAP_DIR", "cap")


sys.path.insert(0, os.path.join(PHYS, "tools"))


def _raw_layout_cpu(layer, cache):
    _, _, _, _, raw = bench.load_layer(None, layer, "cpu", cache)
    return tuple(bench.raw_layout(raw, "cpu"))


def load_weights(dev, layer=21, cache=CACHE):
    """raw ds41h layout (w13 [E, 2I, H/2], s13 [E, 2I, H/32], w2 [E, H, I/2], s2 [E, H, I/32]) of rank 0 on dev,
    through the converted-weight cache (tools/wcache.py; built on the CPU: python3 common.py --prebuild)."""
    from wcache import cached

    return cached(f"L{layer}_r0_raw_layout_v1", lambda: _raw_layout_cpu(layer, cache), device=dev)


def load_gate(layer=21, model=MODEL):
    from wcache import cached

    return cached(f"L{layer}_gate_w_b_v1", lambda: _load_gate(layer, model))


def _load_gate(layer, model):
    ck = bench.Ckpt(model)
    p = f"layers.{layer}.ffn.gate."
    ck.get(p + "bias")
    gw = ck.h[ck.idx[p + "weight"]].get_tensor(p + "weight")
    gb = ck.h[ck.idx[p + "bias"]].get_tensor(p + "bias").float()
    return gw, gb


def real_rows(layer=21, steps=(0, 1, 2)):
    fs = []
    for s in steps:
        rec = torch.load(os.path.join(CAP, f"s{s}_L{layer}_r0.pt"), weights_only=False, map_location="cpu")
        fs.append(rec["ffn_args"][0].to(torch.bfloat16))
        del rec
    return torch.cat(fs).contiguous()


def router(f, gate_w, gate_b, topk=TOPK, scale=1.5):
    """ref gate_ref (block.router): sqrt-softplus scores in fp32, bias for the selection only, weights / sum * 1.5."""
    scores = torch.nn.functional.softplus(f.float() @ gate_w.float().t()).sqrt()
    idx = (scores + gate_b).topk(topk, dim=-1).indices
    wts = scores.gather(-1, idx)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * scale
    return idx.int().contiguous(), wts.float().contiguous()


class Inputs:
    """Real rows, gate weights and routed inputs per (T, routing), cached."""

    def __init__(self, dev, layer=21):
        self.dev = dev
        self.f = real_rows(layer).to(dev)
        gw, gb = load_gate(layer)
        self.gw, self.gb = gw.to(dev), gb.to(dev)

    def get(self, T, routing="synth", seed=0):
        reps = (T + self.f.shape[0] - 1) // self.f.shape[0]
        x = self.f.repeat(reps, 1)[:T].contiguous()
        if routing == "cap":
            ids, w = router(x, self.gw, self.gb)
        elif routing == "synth":
            i_, w_ = bench.synth_routing((T + 3) // 4, seed=seed + T)
            ids, w = i_[:T].int().to(self.dev).contiguous(), w_[:T].float().to(self.dev).contiguous()
        else:
            raise ValueError(routing)
        return x, w, ids


def graph_time(fn, iters=20, warm=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000.0, g  # us per call


_nvml = {}


def nvml_handle(dev):
    import pynvml

    if "init" not in _nvml:
        pynvml.nvmlInit()
        _nvml["init"] = True
    pr = torch.cuda.get_device_properties(dev)
    return pynvml.nvmlDeviceGetHandleByPciBusId(f"{pr.pci_domain_id:08x}:{pr.pci_bus_id:02x}:{pr.pci_device_id:02x}.0")


def energy(g, dev, secs=2.0):
    """Replays graph g for ~secs: (W, mJ per call, mean SM MHz)."""
    import pynvml

    hdl = nvml_handle(dev)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, t0, n, clk = pynvml.nvmlDeviceGetTotalEnergyConsumption(hdl), time.time(), 0, []
    while time.time() - t0 < secs:
        for _ in range(4):
            g.replay()
        n += 4
        clk.append(pynvml.nvmlDeviceGetClockInfo(hdl, pynvml.NVML_CLOCK_SM))
        torch.cuda.synchronize()
    el = time.time() - t0
    jt = (pynvml.nvmlDeviceGetTotalEnergyConsumption(hdl) - e0) / 1000.0
    return jt / el, jt / n * 1000.0, sum(clk) / len(clk)


def kprofile(fn, iters=5):
    """Per-kernel device time (us per call) of eager calls of fn."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    out = {}
    for ev in prof.key_averages():
        if ev.device_time_total > 0:
            out[ev.key[:60]] = round(ev.device_time_total / iters, 1)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def dequant64(w, s):
    """FP4 [R, K/2] + ue8m0 [R, K/32] (uint8, on GPU) -> float64 [R, K] exactly."""
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       dtype=torch.float64, device=w.device)
    v = torch.stack([lut[(w & 0x0F).long()], lut[(w >> 4).long()]], dim=-1).flatten(-2)
    return v * torch.pow(2.0, s.double() - 127).repeat_interleave(32, dim=-1)


def ref_rows(T, cap=2048):
    """Tokens the error statistics use: all for T <= cap, else cap evenly spaced ones (keeps the fp64 reference cheap)."""
    return None if T <= cap else torch.linspace(0, T - 1, cap).round().long()


def reference(x, w, ids, w13, s13, w2, s2, limit=10.0, rows=None):
    """fp64 reference of the routed experts (exact weights, no intermediate rounding): [T, H] (or [len(rows), H]) f64."""
    if rows is not None:
        rows = rows.to(x.device)
        x, w, ids = x[rows], w[rows], ids[rows]
    T = x.shape[0]
    xd = x.double()
    out = torch.zeros(T, H, dtype=torch.float64, device=x.device)
    flat = ids.reshape(-1).long()
    flat = torch.where((flat >= 0) & (flat < E), flat, E)  # ids outside [0, E) contribute nothing
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E + 1).tolist()[:E]
    st = 0
    for e, n in enumerate(counts):
        if n == 0:
            continue
        idx = order[st:st + n]
        st += n
        tok = idx // TOPK
        gu = xd[tok] @ dequant64(w13[e], s13[e]).t()
        g, u = gu[:, :I], gu[:, I:]
        h = torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit) * w.reshape(-1)[idx].double()[:, None]
        out.index_add_(0, tok, h @ dequant64(w2[e], s2[e]).t())
    return out


def err_stats(out, ref):
    """bf16 output vs the fp64 reference: bit-identical to bf16(ref), <= 1 ulp, row-scaled ulps (of the row's max)."""
    o, r = out.double(), ref
    rb = r.to(torch.bfloat16).double()
    d = (o - r).abs()
    ulp = torch.pow(2.0, torch.floor(torch.log2(r.abs().clamp_min(2.0 ** -126))) - 7)
    rmax = r.abs().amax(1, keepdim=True).clamp_min(2.0 ** -100)
    rs = d / torch.pow(2.0, torch.floor(torch.log2(rmax)) - 7)
    rsr = rs.amax(1)
    return dict(exact=round(float((o == rb).double().mean()), 5), le1ulp=round(float((d <= ulp).double().mean()), 5),
                rs_max=round(float(rsr.max()), 3), rs_mean=round(float(rs.mean()), 4),
                rs_p999=round(float(torch.quantile(rsr.float(), 0.999)), 3),
                rel_l2=float((o - r).norm() / r.norm()), nonfinite=int((~torch.isfinite(out)).sum()))


def flops(T):
    return T * TOPK * 2 * (H * 2 * I + I * H)


def jdump(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__" and "--prebuild" in sys.argv:  # CUDA_VISIBLE_DEVICES= python3 common.py --prebuild
    from wcache import prebuild

    prebuild("L21_r0_raw_layout_v1", lambda: _raw_layout_cpu(21, CACHE))
    prebuild("L21_gate_w_b_v1", lambda: _load_gate(21, MODEL))
