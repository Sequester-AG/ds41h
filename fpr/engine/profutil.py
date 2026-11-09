"""Profile one engine step: GPU busy time per device, kernels, launch calls, and host time / launch calls per
piece (ENG_PROF_PIECES=1 puts a "piece:<name>" profiler range around every piece of Engine._issue and step_mixed)."""
from __future__ import annotations

import time

import torch


def profile_call(fn, *args, **kw):
    """(fn's result, summary dict) with fn run under torch.profiler (CPU + CUDA activities)."""
    from torch.profiler import ProfilerActivity, profile

    t0 = time.time()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        out = fn(*args, **kw)
    wall = (time.time() - t0) * 1000.0
    evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    busy = {}
    for e in evs:
        busy.setdefault(e.device_index, []).append((e.time_range.start, e.time_range.end))
    per = {}
    for dvi, iv in busy.items():
        iv.sort()
        tot, ce = 0.0, -1.0
        for a0, a1 in iv:
            if a1 > ce:
                tot += a1 - max(a0, ce)
                ce = a1
        per[dvi] = round(tot / 1000, 2)
    cpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CPU]
    nl = sum(1 for e in cpu if "cudaLaunch" in e.name or "cuLaunch" in e.name)

    def launches(ev):
        n, stack = 0, list(ev.cpu_children)
        while stack:
            c = stack.pop()
            n += ("cudaLaunch" in c.name or "cuLaunch" in c.name or "cudaGraphLaunch" in c.name)
            stack.extend(c.cpu_children)
        return n
    pieces = {}
    for e in cpu:
        if e.name.startswith("piece:"):
            p_ = pieces.setdefault(e.name[6:], [0.0, 0, 0])
            p_[0] += (e.time_range.end - e.time_range.start) / 1000.0
            p_[1] += launches(e)
            p_[2] += 1
    summ = dict(wall_ms=round(wall, 1), gpu_busy_ms_per_device=per, kernels=len(evs), cuda_launch_calls=nl,
                categories=categorize([e for e in prof.events() if not e.name.startswith("piece:")]),
                pieces={k: dict(host_ms=round(v[0], 2), launches=v[1], calls=v[2])
                        for k, v in sorted(pieces.items(), key=lambda kv: -kv[1][0])})
    return out, summ


CATS = [  # first match wins: substrings of CUDA kernel names (lower case)
    ("hsd/head+sampler+drafter", ("sampler", "dspark", "drafter", "head_", "verify", "vp_", "markov", "gumbel", "draft")),
    ("indexer (idx2f + compressor)", ("idx2f", "logits_dense", "block_max", "select_kernel", "topk", "cand", "prep2",
                                      "indexer", "compress", "rope_quant", "k_norm_rope", "fused_save", "global_topk")),
    ("engram", ("engram", "hash_k", "gather_k", "inject")),
    ("flashmla planner", ("get_mla_metadata",)),
    ("flashmla attention", ("flash", "sparse_decode", "mla", "splitkv", "combine_kernel")),
    ("moe (route / gather / grouped gemm / combine)", ("route", "gemm1r", "gemm2r", "prepr", "moe", "combine", "swiglu",
                                                       "msr::", "w4a16", "moe_dec", "fused_moe")),
    ("mhc boundaries", ("mhc", "k1a", "k1b", "sinkhorn")),
    ("collectives (ag / rs / waits)", ("ag_push", "rs_push", "rs_sum", "wait_k", "k_wait", "push", "spin")),
    ("dense gemms (cublas / w8a16)", ("gemm", "cutlass", "xmma", "nvjet", "sm90", "w8a16", "cublas", "gemv")),
    ("rope / norms / elementwise", ("rope", "norm", "elementwise", "vectorized", "copy", "fill", "index", "reduce", "cat")),
]


def categorize(prof_events) -> dict:
    """GPU kernel time per category (ms, summed over devices and divided by the device count) + the top kernels.
    record_function ranges ("piece:...", ENG_PROF_PIECES) also appear as CUDA-side annotations: excluded."""
    evs = [e for e in prof_events if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.startswith("piece:")]
    ndev = max(1, len({e.device_index for e in evs}))
    cat, top = {}, {}
    for e in evs:
        n = e.name.lower()
        dur = (e.time_range.end - e.time_range.start) / 1000.0
        c = next((k for k, pats in CATS if any(p in n for p in pats)), "other")
        cat[c] = cat.get(c, 0.0) + dur
        top[e.name[:90]] = top.get(e.name[:90], 0.0) + dur
    return dict(per_device_ms={k: round(v / ndev, 3) for k, v in sorted(cat.items(), key=lambda kv: -kv[1])},
                top_kernels=[(k, round(v / ndev, 3)) for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:30]])


def profile_categorized(fn, *args, **kw):
    """(result, summary with GPU time per kernel category) for one call under torch.profiler."""
    from torch.profiler import ProfilerActivity, profile
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        out = fn(*args, **kw)
    wall = (time.time() - t0) * 1000.0
    summ = categorize(prof.events())
    summ["wall_ms"] = round(wall, 1)
    return out, summ
