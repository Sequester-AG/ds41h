"""Decode-cycle breakdown from vLLM torch-profiler traces (one file per TP rank).

  python scripts/cycle_profile.py TRACE_rank0.json.gz [TRACE_rank1 ...] [--marker REGEX] [--min-layers 20]
                                  [--skip N] [--top 12]

A decode "cycle" = target forward + everything until the next target forward (drafter, sampler, metadata
prep). Target forwards are found as runs of >= --min-layers marker kernels (default: ds41h layer_kernel or the
stock FlashMLA sparse decode kernel, one per layer) separated by > --gap-ms (0.6 ms). For each rank it reports the mean
cycle time, GPU busy/idle per cycle, target span (first to last marker kernel of the run, plus the kernels
before the first marker that belong to the same graph are counted in "pre"), the inter-target window, and
the top kernels of the window. With several ranks it also prints per-rank mean durations of a few kernels
whose duration includes waiting for peers (all-reduce / all-gather), to expose rank skew: the rank that
waits least is the critical path.
"""

import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict

DEFAULT_MARKER = r"layer_kernel|flash_fwd_splitkv_mla_fp8_sparse"
SKEW_KERNELS = ["ll_all_reduce", "ll_all_gather", "allreduce_fusion", "layer_kernel", "prologue_kernel",
                "nvjet_sm90_tst_256x8", "_draft_kernel", "sparse_decode_kernel"]


def load(path):
    with (gzip.open(path) if path.endswith(".gz") else open(path)) as f:
        t = json.load(f)
    ks = [e for e in t["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    ks.sort(key=lambda e: e["ts"])
    return ks


def busy(ks, lo, hi):
    tot, ce = 0.0, lo
    for k in ks:
        s, e = max(k["ts"], lo), min(k["ts"] + k.get("dur", 0), hi)
        if e <= ce:
            continue
        tot += e - max(s, ce)
        ce = e
    return tot


def short(n, w=80):
    n = re.sub(r"\(anonymous namespace\)::", "", n)
    return n if len(n) <= w else n[: w - 3] + "..."


def analyse(path, a):
    ks = load(path)
    mk = re.compile(a.marker)
    idx = [i for i, k in enumerate(ks) if mk.search(k["name"])]
    runs, cur = [], [idx[0]] if idx else []
    for p, q in zip(idx, idx[1:]):
        if ks[q]["ts"] - (ks[p]["ts"] + ks[p].get("dur", 0)) > a.gap_ms * 1000:
            runs.append(cur)
            cur = [q]
        else:
            cur.append(q)
    if cur:
        runs.append(cur)
    runs = [r for r in runs if len(r) >= a.min_layers]
    # keep only runs that are consecutive decode steps (cycle < 100 ms): drops prefill / idle boundaries
    rows, win_k, win_n = [], defaultdict(float), defaultdict(int)
    for r0, r1 in zip(runs, runs[1:]):
        t0, t1 = ks[r0[0]]["ts"], ks[r1[0]]["ts"]
        if t1 - t0 > 100_000:
            continue
        tend = ks[r0[-1]]["ts"] + ks[r0[-1]].get("dur", 0)
        sub = [k for k in ks[r0[0]: r1[0]]]
        b = busy(sub, t0, t1)
        lay = sum(ks[j].get("dur", 0) for j in r0)
        win = [k for k in ks[r0[-1] + 1: r1[0]]]
        rows.append(dict(cycle=(t1 - t0) / 1e3, busy=b / 1e3, idle=(t1 - t0 - b) / 1e3, target=(tend - t0) / 1e3,
                         window=(t1 - tend) / 1e3, marker_sum=lay / 1e3, n=len(r0), nwin=len(win)))
        for k in win:
            win_k[k["name"]] += k.get("dur", 0) / 1e3
            win_n[k["name"]] += 1
    rows = rows[a.skip:]
    skew = defaultdict(list)
    for k in ks:
        for s in SKEW_KERNELS:
            if s in k["name"]:
                skew[s].append(k.get("dur", 0))
                break
    return rows, win_k, win_n, skew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--marker", default=DEFAULT_MARKER)
    ap.add_argument("--min-layers", type=int, default=20)
    ap.add_argument("--gap-ms", type=float, default=0.6, help="marker gap that separates target forwards")
    ap.add_argument("--skip", type=int, default=0, help="drop the first N cycles (boost-clock warmup)")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    skews = []
    for ri, path in enumerate(a.traces):
        rows, win_k, win_n, skew = analyse(path, a)
        skews.append(skew)
        if not rows:
            print(f"rank{ri}: no decode cycles found ({path})")
            continue
        n = len(rows)
        m = {k: statistics.mean(r[k] for r in rows) for k in rows[0]}
        cyc = sorted(r["cycle"] for r in rows)
        print(f"rank{ri}: {n} cycles  cycle mean {m['cycle']:.2f} ms (p10 {cyc[n // 10]:.2f} p50 {cyc[n // 2]:.2f} "
              f"p90 {cyc[(9 * n) // 10]:.2f})  busy {m['busy']:.2f}  idle {m['idle']:.2f}  target {m['target']:.2f} "
              f"(marker sum {m['marker_sum']:.2f}, {m['n']:.0f} markers, {1e3 * m['marker_sum'] / max(m['n'], 1):.0f} us each)  "
              f"inter-target window {m['window']:.2f} ({m['nwin']:.0f} kernels)")
        if ri == 0 or a.top > 0 and len(a.traces) == 1:
            print(f"  inter-target window top kernels (rank{ri}, ms/cycle, launches/cycle):")
            for name, t in sorted(win_k.items(), key=lambda kv: -kv[1])[: a.top]:
                print(f"    {t / n:7.3f}  {win_n[name] / n:5.1f}  {short(name)}")
    if len(skews) > 1:
        print("per-rank median kernel duration (us) [waiting kernels: the smallest is the critical-path rank]:")
        names = [s for s in SKEW_KERNELS if any(s in sk for sk in skews)]
        print("    " + "".join(f"{s[:20]:>22s}" for s in names))
        for ri, sk in enumerate(skews):
            print(f"  r{ri}" + "".join(f"{statistics.median(sk[s]) if sk.get(s) else float('nan'):22.1f}" for s in names))


if __name__ == "__main__":
    main()
