"""Summarize adapter/steptimes.py output (adapter/prof_1u.sh): medians of the per-step host marks for the decode-only
steps, grouped by decode sequences per step, plus the profiler windows (GPU busy per device vs step wall).
  python st_summary.py steps.jsonl [prof_dir]"""
import glob
import json
import statistics as S
import sys


def flat(r, pre=""):
    out = {}
    for k, v in r.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        elif isinstance(v, list) and k == "worker" and len(v) == 4:
            for n, x in zip(("copies", "tables", "ctx", "engine"), v):
                out[f"{pre}worker.{n}"] = x
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[pre + k] = v
    return out


lines = open(sys.argv[1]).read().splitlines()
if "--lines" in sys.argv:                                   # a:b = only lines a..b-1 (one phase of a run)
    a_, b_ = sys.argv[sys.argv.index("--lines") + 1].split(":")
    lines = lines[int(a_ or 0):int(b_) if b_ else None]
recs = [json.loads(ln) for ln in lines if ln.strip()]
recs.sort(key=lambda r: r["t"])
for a, b in zip(recs, recs[1:]):
    a["cycle"] = round((b["t"] - a["t"]) * 1e3, 3)
for r in recs:
    for k0, k1, nm in (("s1", "x0", "pre_exec"), ("x0", "x1", "exec"), ("x1", "m1", "x1_to_m1"), ("m1", "u0", "m1_to_u0"),
                       ("u0", "u1", "update"), ("r1", "m1", "merge")):
        if k0 in r and k1 in r:
            r[nm] = round(r[k1] - r[k0], 3)
groups = {}
for r in recs[:-1]:
    if r.get("pf", 0) == 0 and r.get("dec", 0) > 0 and r.get("copies", 0) == 0:
        groups.setdefault(r["dec"], []).append(r)
for dec, rs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:4]:
    rs = rs[5:] if len(rs) > 20 else rs                       # skip the first steps of a phase (captures)
    keys = {}
    for r in rs:
        for k, v in flat(r).items():
            keys.setdefault(k, []).append(v)
    print(f"== decode-only steps with {dec} sequences: {len(rs)} steps")
    for k in ("cycle", "s1", "pre_exec", "exec", "x1_to_m1", "merge", "m1_to_u0", "update", "u1", "gap", "rows"):
        if k in keys:
            v = keys[k]
            print(f"  {k:28s} p50 {S.median(v):8.3f}  p10 {sorted(v)[len(v) // 10]:8.3f}  p90 {sorted(v)[9 * len(v) // 10]:8.3f}")
    for k in sorted(keys):
        if k.startswith(("worker.", "eng.", "d13.")):
            v = keys[k]
            print(f"  {k:28s} p50 {S.median(v):8.3f}  p90 {sorted(v)[9 * len(v) // 10]:8.3f}")
pdir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
for p in (sorted(glob.glob(pdir + "/prof_*.json")) if pdir else []):
    d = json.load(open(p))
    st = d.get("steps", [])[1:-1]
    print(f"== profiler {p}: {d.get('n_steps')} steps, window {d.get('window_ms')} ms, GPU busy per device {d.get('gpu_busy_ms')}")
    if st:
        w = S.median(x["wall_ms"] for x in st)
        print(f"  per step: wall p50 {w:.3f} ms; GPU busy p50 per device "
              f"{ {k: round(S.median(x['gpu_busy_ms'].get(k, 0) for x in st), 3) for k in st[0]['gpu_busy_ms']} }")
    cat = d.get("categories", {})
    n = max(1, d.get("n_steps") or 1)
    print("  GPU ms per step by category:", {k: round(v / n, 3) for k, v in cat.get("per_device_ms", {}).items()})
    print("  top kernels (ms per step):", [(k[:60], round(v / n, 3)) for k, v in cat.get("top_kernels", [])[:12]])
    print("  host ranges (ms per step):", {k: round(v / n, 3) for k, v in d.get("host_ranges_ms", {}).items()})
