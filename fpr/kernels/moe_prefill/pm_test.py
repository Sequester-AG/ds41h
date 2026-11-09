#!/usr/bin/env python3
"""pm_fused vs the engine's current production-MoE path (engine.py p_moe_prod, replicated in ref_layer), one GPU, real
layer-21 rank-0 weights (routed + shared slice), real captured MoE input rows (a local capture directory, CAP_DIR) with
the layer's real router, engine bucket layouts (Tall = 4 R rows, each rank's valid rows a prefix of its block).

Checks per configuration: part [Tall, 5120] bit-equal to the current path (compact and padded modes), repeatability,
CUDA-graph capture + replay of the padded mode on new data, the fork swiglu copy == moe_raw's swiglu. Timing: GPU us per
call (eager), host issue us per call (GPU kept busy by a sleep kernel, so only the CPU cost is measured), CUDA kernels
per call (profiler). AGGREGATE OUTPUT ONLY (no row data).

  CUDA_VISIBLE_DEVICES= python3 pm_test.py --prebuild          # CPU: extensions + shared-expert weight cache
  DS41H_PHYS_MOE=0 PYTHONPATH=<repo root>:$PYTHONPATH python3 pm_test.py   # one GPU
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

import common as C
import pm_fused as P

W, D, TOPK, IS = 4, 5120, 6, 576
LAYER = 21
CONFIGS = [  # name, R (rows per rank in the bucket), valid rows per rank
    ("bal1k", 1024, [1024, 1024, 1024, 1024]),        # a 4K-row mixed step, 1K rows per rank: vcount 4,096 = Tall
    ("switch", 768, [700, 650, 720, 690]),             # just above prefill_moe_switch: 2,760 of 3,072
    ("bal4k", 4096, [4096, 4096, 4096, 4096]),         # 16,384
    ("max", 4608, [4416, 4416, 4416, 4416]),           # max_mixed_rows 4,416 per rank: 17,664 of 18,432
    ("skew1", 4352, [4224, 96, 64, 128]),              # one rank prefilling: 4,512 of 17,408
    ("skew2", 3584, [3500, 2048, 900, 3200]),          # 9,648 of 14,336
    ("mid", 1280, [1200, 1100, 300, 1250]),            # 3,850 of 5,120
]


def _shared_cpu(layer=LAYER):
    import block

    ck = C.bench.Ckpt(C.MODEL)
    w1, w3, w2, _, _ = block.load_dense(ck, layer, 0, "cpu")
    return torch.cat([w1, w3], 0).contiguous(), w2.contiguous()


def load_shared(dev, layer=LAYER):
    from wcache import cached

    return cached(f"L{layer}_r0_shared_w13_w2_bf16_v1", lambda: _shared_cpu(layer), device=dev)


def prebuild():
    from wcache import prebuild as wprebuild

    P.build()
    C.bench.build_extr()                 # the engine's fork MoE extension (moe_raw.cu: the current swiglu)
    from ds41h import moe as PM
    PM.ext()                             # production's extension (the current path)
    wprebuild(f"L{LAYER}_r0_shared_w13_w2_bf16_v1", lambda: _shared_cpu(LAYER))
    print("prebuild done", flush=True)


class Layer:
    def __init__(self, dev):
        from ds41h import moe as PM

        self.PM, self.dev = PM, dev
        self.w13, self.s13, self.w2, self.s2 = C.load_weights(dev)
        self.sw13, self.sw2 = load_shared(dev)
        self.mext = C.bench.build_extr()
        self.pmf = P.Fused(PM)
        print(f"production: variant 0x{self.pmf.variant:x}, hybrid {PM.HYBRID}/{PM.HYBRID_ENGINE}, MOE_DEC {PM.MOE_DEC}, "
              f"SMALL_T {PM.SMALL_T}; sw13 {tuple(self.sw13.shape)} sw2 {tuple(self.sw2.shape)}", flush=True)

    def ref(self, b):
        """engine.py p_moe_prod (current path), on bucket b."""
        fall, vi = b["fall"], b["vi"]
        fv = fall.index_select(0, vi)
        gu = torch.mm(fv, self.sw13.t())
        hsh = torch.empty(fv.shape[0], IS, dtype=torch.bfloat16, device=fv.device)
        self.mext.swiglu(gu, hsh)
        sh = torch.mm(hsh, self.sw2.t())
        rows = b["gr"].index_select(0, vi)
        ids = rows[:, :24].contiguous().view(torch.int32).view(-1, TOPK)
        wts = rows[:, 24:48].contiguous().view(torch.float32).view(-1, TOPK)
        y = self.PM.moe_forward_w4a16(fv, wts, ids, self.w13, self.s13, self.w2, self.s2, 10.0, addend=sh)
        b["part"].zero_()
        b["part"].index_copy_(0, vi, y)
        b["ref_sh"] = sh

    def new(self, b, padded, out="part"):
        self.pmf.layer(b["fall"], b["gr"], b["nv"], b["R"], b["vcount"], self.w13, self.s13, self.w2, self.s2, self.sw13, self.sw2,
                       b["gu"], b["hsh"], b["sh"], b[out], padded=padded)


def make_bucket(dev, inp, R, ns, seed, pad="garbage"):
    """Engine bucket state: fall = the all-gathered f rows (bf16 view of gf), gr = route rows (router_top_pack2 layout),
    padding rows: garbage (or NaN) x, padrow route rows (ids -1) -- or garbage route rows (pad='nan': nv must win)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    Tall, V = W * R, sum(ns)
    fall = (torch.randn(Tall, D, generator=g, device=dev) * 3).to(torch.bfloat16)
    gr = torch.zeros(Tall, 64, dtype=torch.uint8, device=dev)
    gr[:, :24] = 0xFF
    if pad == "nan":
        fall.fill_(float("nan"))
        gr = torch.randint(0, 256, (Tall, 64), generator=g, device=dev, dtype=torch.uint8)
    x, w, ids = inp.get(V + 97 * (seed % 7), "cap")
    x, w, ids = x[97 * (seed % 7):], w[97 * (seed % 7):], ids[97 * (seed % 7):]
    gi, gw = gr.view(torch.int32), gr.view(torch.float32)
    o = 0
    for q, n in enumerate(ns):
        fall[q * R:q * R + n] = x[o:o + n]
        gi[q * R:q * R + n, 0:6] = ids[o:o + n].int()
        gw[q * R:q * R + n, 6:12] = w[o:o + n].float()
        o += n
    vi = torch.cat([torch.arange(n, dtype=torch.int64) + q * R for q, n in enumerate(ns)]).to(dev)
    bf = dict(dtype=torch.bfloat16, device=dev)
    return dict(R=R, ns=ns, Tall=Tall, vcount=V, fall=fall.contiguous(), gr=gr, vi=vi,
                nv=torch.tensor(ns, dtype=torch.int32, device=dev),
                part=torch.empty(Tall, D, **bf), out_c=torch.full((Tall, D), float("nan"), **bf),
                out_p=torch.full((Tall, D), float("nan"), **bf), gu=torch.empty(Tall, 2 * IS, **bf),
                hsh=torch.empty(Tall, IS, **bf), sh=torch.empty(Tall, D, **bf))


def cmp(a, b):
    ai, bi = a.view(torch.int16), b.view(torch.int16)
    ne = ai != bi
    n = int(ne.sum())
    return dict(eq=n == 0, diff_el=n, diff_rows=int(ne.any(1).sum()),
                max_int=int((ai.int() - bi.int()).abs().max()) if n else 0)


def gpu_us(fn, iters=8):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return round(a.elapsed_time(b) / iters * 1000.0, 1)


def host_us(fn, n=16):
    """CPU cost of issuing one call: the GPU is held by a sleep kernel, so no launch waits on it."""
    fn()
    torch.cuda.synchronize()
    torch.cuda._sleep(int(1.5e9))
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return round((t1 - t0) / n * 1e6, 1)


def kernels(fn):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    ev = prof.events()
    names = [e.name for e in ev if e.device_type == DeviceType.CUDA]
    launches = [e.name for e in ev if e.device_type == DeviceType.CPU and e.name.startswith("cu") and
                ("Launch" in e.name or "Memset" in e.name or "Memcpy" in e.name)]
    return len(names), len(launches), names


def swiglu_check(L, dev):
    g = torch.Generator(device=dev).manual_seed(5)
    gu = (torch.randn(65536, 2 * IS, generator=g, device=dev) * 6).to(torch.bfloat16)
    gu.view(torch.int16)[::7, ::5] = torch.randint(-32768, 32767, (gu[::7, ::5].shape), generator=g, device=dev,
                                                   dtype=torch.int16)   # every bit pattern class (denormals, inf, nan)
    h1, h2 = torch.empty(65536, IS, dtype=torch.bfloat16, device=dev), torch.empty(65536, IS, dtype=torch.bfloat16, device=dev)
    L.mext.swiglu(gu, h1)
    P.swiglu_ext().swiglu(gu, h2)
    return cmp(h1, h2)


def fp64_check(L, b, n=256):
    """error of the current path and of both pm_fused modes vs the fp64 reference (routed + shared, exact weights, no
    intermediate rounding) on n evenly spaced valid rows."""
    vi = b["vi"]
    p = vi[torch.linspace(0, vi.numel() - 1, n).round().long().to(vi.device)]
    x, rows = b["fall"].index_select(0, p), b["gr"].index_select(0, p)
    ids = rows[:, :24].contiguous().view(torch.int32).view(-1, TOPK)
    w = rows[:, 24:48].contiguous().view(torch.float32).view(-1, TOPK)
    ref = C.reference(x, w, ids, L.w13, L.s13, L.w2, L.s2)
    gu = x.double() @ L.sw13.double().t()
    hh = torch.nn.functional.silu(gu[:, :IS].clamp(max=10.0)) * gu[:, IS:].clamp(-10.0, 10.0)
    ref = ref + hh @ L.sw2.double().t()
    return {k: C.err_stats(b[k].index_select(0, p), ref) for k in ("part", "out_c", "out_p")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebuild", action="store_true")
    ap.add_argument("--configs", default=",".join(c[0] for c in CONFIGS))
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--out", default=os.path.join(C.HERE, "results", "pm_fused.jsonl"))
    a = ap.parse_args()
    if a.prebuild:
        return prebuild()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda", 0)
    t0 = time.time()
    L = Layer(dev)
    inp = C.Inputs(dev)
    sw = swiglu_check(L, dev)
    print(f"setup {time.time() - t0:.0f} s; swiglu copy vs moe_raw: {sw}", flush=True)
    C.jdump(a.out, dict(test="swiglu", **sw))
    want = a.configs.split(",")
    for name, R, ns in [c for c in CONFIGS if c[0] in want]:
        res = dict(test="layer", cfg=name, R=R, ns=ns, Tall=W * R, vcount=sum(ns))
        for pad in ("garbage", "nan") if name in ("skew1", "mid") else ("garbage",):
            b = make_bucket(dev, inp, R, ns, seed=R + len(pad), pad=pad)
            L.ref(b)
            L.new(b, False, "out_c")
            L.new(b, True, "out_p")
            torch.cuda.synchronize()
            c, p = cmp(b["part"], b["out_c"]), cmp(b["part"], b["out_p"])
            sh_p = cmp(b["ref_sh"], b["sh"].index_select(0, b["vi"]))   # padded mode's shared rows vs the current path's
            r2c, r2p = b["out_c"].clone(), b["out_p"].clone()
            L.new(b, False, "out_c")
            L.new(b, True, "out_p")
            torch.cuda.synchronize()
            res[pad] = dict(compact=c, padded=p, padded_shared=sh_p, repeat_compact=cmp(r2c, b["out_c"])["eq"],
                            repeat_padded=cmp(r2p, b["out_p"])["eq"])
            print(f"{name:7s} {pad:7s} Tall {W * R:6d} v {sum(ns):6d}: compact {c} padded {p} (shared {sh_p['eq']}, "
                  f"{sh_p['diff_rows']} rows) repeat {res[pad]['repeat_compact']}/{res[pad]['repeat_padded']}", flush=True)
            if pad == "garbage" and name in ("bal1k", "skew1", "max"):
                res["fp64"] = fp64_check(L, b)
                print(f"        vs fp64 {res['fp64']}", flush=True)
            if pad == "garbage" and not a.no_timing:
                ref_fn = lambda: L.ref(b)            # noqa: E731
                c_fn = lambda: L.new(b, False)       # noqa: E731
                p_fn = lambda: L.new(b, True)        # noqa: E731
                tm = dict(gpu_ref=gpu_us(ref_fn), gpu_compact=gpu_us(c_fn), gpu_padded=gpu_us(p_fn),
                          host_ref=host_us(ref_fn), host_compact=host_us(c_fn), host_padded=host_us(p_fn))
                if name in ("bal1k", "skew1"):
                    for k, fn in (("ref", ref_fn), ("compact", c_fn), ("padded", p_fn)):
                        nk, nl, names = kernels(fn)
                        tm[f"kernels_{k}"], tm[f"launch_calls_{k}"] = nk, nl
                        res[f"kernel_names_{k}"] = names
                res["timing"] = tm
                print(f"        timing {tm}", flush=True)
            del b
            torch.cuda.empty_cache()
        C.jdump(a.out, res)
    # padded mode inside a CUDA graph: capture once per bucket, replay on new data / new valid counts
    for R, sets in ((1024, [[1024] * 4, [1000, 1024, 3, 900], [512, 0, 1024, 1024]]),
                    (4352, [[4224, 96, 64, 128], [4352] * 4, [3000, 4000, 2500, 4100]])):
        st = make_bucket(dev, inp, R, sets[0], seed=11)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            L.new(st, True, "out_p")
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            L.new(st, True, "out_p")
        rows = []
        for i, ns in enumerate(sets):
            b = make_bucket(dev, inp, R, ns, seed=100 + i)
            L.ref(b)
            for k in ("fall", "gr", "nv"):
                st[k].copy_(b[k])
            st["out_p"].fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            rows.append(dict(ns=ns, vs_ref=cmp(b["part"], st["out_p"])))
            del b
        a_, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a_.record()
        for _ in range(8):
            g.replay()
        b_.record()
        torch.cuda.synchronize()
        gr_us = round(a_.elapsed_time(b_) / 8 * 1000.0, 1)
        st["out_p"].fill_(0)
        hr = host_us(lambda: g.replay())
        print(f"graph R {R}: {rows}; replay {gr_us} us GPU, {hr} us host", flush=True)
        C.jdump(a.out, dict(test="graph", R=R, sets=rows, replay_gpu_us=gr_us, replay_host_us=hr))
        del st, g
        torch.cuda.empty_cache()
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
