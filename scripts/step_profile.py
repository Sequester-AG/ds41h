"""Per-category GPU time for DECODE steps (or any step) from vLLM torch-profiler traces.

  python scripts/step_profile.py TRACE.json.gz [--decode] [--steps a:b] [--top N]

Reuses prefill_profile.py's trace parsing and step splitting. With --decode it averages over all steps whose
annotation has generation tokens and no context tokens, and reports per-step mean GPU span, busy time, idle gaps
and category totals, plus the top kernels. Categories add ds41h kernel names.
"""

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import prefill_profile as pp  # noqa: E402

DS41H_RULES = [
    ("ds41h decoder", r"layer_kernel|decoder"),
    ("ds41h prologue", r"prologue_kernel|skinny"),
    ("ds41h attn", r"sparse_decode"),
    ("sampling/draft", r"gumbel|sample|reject|markov|draft|vp_|dspark|logits|argmax|softmax"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--steps", default=None, help="slice a:b over the full step list")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    trace = pp.load_trace(Path(a.trace))
    steps = pp.split_steps(trace, "auto", 2.0)
    if a.steps:
        lo, hi = (int(x) if x else None for x in a.steps.split(":"))
        steps = steps[lo:hi]
    if a.decode:
        steps = [s for s in steps if s.kernels and s.ctx_tokens == 0]
    steps = [s for s in steps if s.kernels]
    if not steps:
        sys.exit("no matching steps")
    rules = [(n, re.compile(p, re.I)) for n, p in DS41H_RULES] + pp.compile_rules(None)
    cat = defaultdict(list)
    kern = defaultdict(float)
    nk = defaultdict(int)
    spans, busys = [], []
    for s in steps:
        ks = s.kernels
        t0 = min(k["ts"] for k in ks)
        t1 = max(k["ts"] + k.get("dur", 0) for k in ks)
        spans.append((t1 - t0) / 1000)
        busys.append(pp.busy_union(ks) / 1000)
        per = defaultdict(float)
        for k in ks:
            c = pp.categorize(k["name"], rules)
            per[c] += k.get("dur", 0) / 1000
            kern[k["name"]] += k.get("dur", 0) / 1000 / len(steps)
            nk[k["name"]] += 1
        for c in set(per) | set(cat):
            cat[c].append(per.get(c, 0.0))
    n = len(steps)
    print(f"{n} steps; mean GPU span {statistics.mean(spans):.2f} ms, mean busy {statistics.mean(busys):.2f} ms, "
          f"mean idle {statistics.mean(spans) - statistics.mean(busys):.2f} ms, kernels/step {sum(len(s.kernels) for s in steps) / n:.0f}")
    for c, v in sorted(cat.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {c:16s} {sum(v) / n:8.3f} ms/step")
    print("top kernels (ms/step, launches/step):")
    for name, t in sorted(kern.items(), key=lambda kv: -kv[1])[: a.top]:
        print(f"  {t:8.3f}  {nk[name] / n:6.1f}  {pp.short(name, 90)}")


if __name__ == "__main__":
    main()
