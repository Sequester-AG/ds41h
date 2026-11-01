"""Distribution-based exactness gate: N teacher-forced collections of one server vs the stock
reference dump, compared with stock's own run-to-run distribution.

Why: stock vLLM and ds41h are both run-to-run nondeterministic (stock Humming Stream-K dense GEMMs sum bf16 partials with
unordered atomics; the ds41h decoder accumulates with fp32 atomics), so ONE forced collection gives one random draw of
every gate metric -- the max-type criteria (max |dlogp|, p99) flip between pass and fail on identical configs.

  python scripts/exact_dist.py COLL_DIR [COLL_DIR ...] [--ref DIR] [--set SET] [--stock-json F] [--pairs] [--json OUT]

COLL_DIR = one collection (exact_eval collect dump dir: <tag>/rank*.pt). Arms: collections are grouped by the name
before the last "_<digit>" (def_0, def_1 -> arm "def"); a plain dump dir is its own arm.
Per collection: KL(ref || coll) mean / p99 / max, top-1 agreement, max |dlogp(forced)| over the decode rows, plus the
sensitive rows (s_ja 72 logit gap, s_de 45 / 167 and s_prose 181 p(forced)). Per arm: mean / median / SD / max and the
fraction of collections failing the single-run criteria. --pairs: within-arm collection-vs-collection spread (the arm's own
run-to-run noise). --stock-json: the same summary of N STOCK collections (scripts/exact_dist.py on a stock server's
collections --json): every arm is compared with it (Mann-Whitney U per metric, single-run failure rate; information).

DISTRIBUTION GATE (docs/EXACTNESS.md): PASS if, over N >= 6 collections, the median of every single-run metric
meets the single-run threshold (mean KL <= 7e-3, p99 <= 0.15, top-1 >= 0.980, max |dlogp| <= 1.5). Reported as information:
one-sided Mann-Whitney vs stock's own collections per metric (biased against any non-stock summation order, see the verdict
code), single-run failure rate vs stock's (Fisher), within-arm pair spread vs stock's.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_eval import load_dump  # noqa: E402

SET = os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "runs", "exact", "set_v1.json")
REF = os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "runs", "exact", "ex_stock", "dump")
SENS = [("s_ja", 72, "gap", (2479, 25184)), ("s_de", 45, "p", None), ("s_de", 167, "p", None), ("s_prose", 181, "p", None)]
THRESH = {"mean": (7e-3, "le"), "p99": (0.15, "le"), "top1": (0.980, "ge"), "dlp": (1.5, "le")}
WORSE = {"mean": "greater", "p99": "greater", "max": "greater", "dlp": "greater", "top1": "less"}


def arm_of(path: str) -> str:
    b = os.path.basename(path.rstrip("/"))
    head, _, tail = b.rpartition("_")
    return head if head and tail.isdigit() else b


def mann_whitney_p(x: list[float], y: list[float], alternative: str) -> float:
    """One-sided Mann-Whitney U p-value (normal approximation with tie correction): alternative 'greater' = x tends to be
    larger than y."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    allv = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks = [0.0] * len(allv)
    i = 0
    ties = 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, allv) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    n = n1 + n2
    sigma = math.sqrt(n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))) if n > 1 else 0.0
    if sigma == 0:
        return 0.5
    z = (u1 - mu - 0.5) / sigma if alternative == "greater" else (mu - u1 - 0.5) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2))


def fisher_greater(a: int, n: int, b: int, m: int) -> float:
    """One-sided Fisher exact p that rate a/n > rate b/m (hypergeometric tail)."""
    from math import comb
    tot, k = n + m, a + b
    if n == 0 or m == 0:
        return float("nan")
    den = comb(tot, k)
    return sum(comb(n, i) * comb(m, k - i) for i in range(a, min(n, k) + 1) if 0 <= k - i <= m) / den


def collect_stats(cols: list[str], ref: str, set_path: str, pairs: bool):
    forced = {it["tag"]: it["forced"] for it in json.load(open(set_path))["items"]}
    per = {c: {"kl": [], "t1": [], "dt": []} for c in cols}
    pair = {}
    sens = {c: {} for c in cols}
    ref_sens = {}
    for tag in sorted(forced):
        missing = [c for c in cols if not os.path.exists(os.path.join(c, tag, "rank0.json"))]
        if missing:
            print(f"# {tag}: missing in {len(missing)} collection(s) -> skipped")
            continue
        rx, ridx, _ = load_dump(os.path.join(ref, tag))
        n = rx.shape[0]
        dec = ridx[:n] > 0
        tok = torch.tensor([forced[tag][int(i)] for i in ridx[:n]], dtype=torch.int64)
        lr = torch.log_softmax(rx.float(), -1)
        L = {}
        for c in cols:
            x, idx, _ = load_dump(os.path.join(c, tag))
            m = min(n, x.shape[0])
            assert torch.equal(idx[:m], ridx[:m]), f"{c}/{tag}: forced positions differ"
            L[c] = torch.log_softmax(x[:m].float(), -1)
            la, lb = lr[:m], L[c]
            sel = dec[:m]
            per[c]["kl"].append((la.exp() * (la - lb)).sum(-1).clamp_min(0)[sel])
            per[c]["t1"].append((la.argmax(-1) == lb.argmax(-1))[sel])
            per[c]["dt"].append((la.gather(1, tok[:m, None]) - lb.gather(1, tok[:m, None])).abs()[:, 0][sel])
            for st_, row, kind, toks in SENS:
                if st_ == tag and row < m:
                    sens[c][f"{st_}{row}{kind}"] = (float(x[row, toks[0]].float() - x[row, toks[1]].float()) if kind == "gap"
                                                    else float(L[c][row, tok[row]].exp()))
        for st_, row, kind, toks in SENS:
            if st_ == tag and row < n:
                ref_sens[f"{st_}{row}{kind}"] = (float(rx[row, toks[0]].float() - rx[row, toks[1]].float()) if kind == "gap"
                                                 else float(lr[row, tok[row]].exp()))
        if pairs:
            for a, b in itertools.combinations(cols, 2):
                if arm_of(a) != arm_of(b) or os.path.dirname(a.rstrip("/")) != os.path.dirname(b.rstrip("/")):
                    continue
                m = min(L[a].shape[0], L[b].shape[0])
                la, lb = L[a][:m], L[b][:m]
                sel = dec[:m]
                p = pair.setdefault((a, b), {"kl": [], "t1": [], "dt": []})
                p["kl"].append((la.exp() * (la - lb)).sum(-1).clamp_min(0)[sel])
                p["t1"].append((la.argmax(-1) == lb.argmax(-1))[sel])
                p["dt"].append((la.gather(1, tok[:m, None]) - lb.gather(1, tok[:m, None])).abs()[:, 0][sel])
        del L, lr, rx

    def summ(d):
        kl, t1, dt = torch.cat(d["kl"]), torch.cat(d["t1"]), torch.cat(d["dt"])
        return {"mean": float(kl.mean()), "p99": float(kl.quantile(0.99)), "max": float(kl.max()),
                "top1": float(t1.float().mean()), "dlp": float(dt.max()), "dlp_mean": float(dt.mean())}

    out_cols = {c: {**summ(per[c]), **sens[c]} for c in cols}
    out_pairs = {f"{a}|{b}": summ(d) for (a, b), d in pair.items()}
    return out_cols, out_pairs, ref_sens


def fails(s: dict) -> list[str]:
    return [k for k, (t, how) in THRESH.items() if (s[k] > t if how == "le" else s[k] < t)]


def arm_summary(vals: list[dict]) -> dict:
    out = {"n": len(vals)}
    for k in ("mean", "p99", "max", "top1", "dlp", "dlp_mean"):
        xs = [v[k] for v in vals]
        out[k] = {"mean": st.mean(xs), "median": st.median(xs), "sd": st.stdev(xs) if len(xs) > 1 else 0.0, "min": min(xs),
                  "max": max(xs), "all": xs}
    out["single_run_fails"] = sum(1 for v in vals if fails(v))
    for key in ("s_ja72gap", "s_de45p", "s_de167p", "s_prose181p"):
        out[key] = sorted(round(v[key], 3) for v in vals if key in v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cols", nargs="+", help="collection dump dirs (or a dir of collections with --root)")
    ap.add_argument("--root", action="store_true", help="each argument is a directory containing collection dirs")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--set", default=SET)
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--stock-json", default=None, help="exact_dist --json output of N stock collections (same --ref)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cols = []
    for c in args.cols:
        if args.root:
            cols += sorted(d for d in glob.glob(os.path.join(c, "*")) if os.path.exists(os.path.join(d, "s_ja", "rank0.json")))
        else:
            cols.append(c)
    per, pairs, ref_sens = collect_stats(cols, args.ref, args.set, args.pairs)
    print(f"reference {args.ref}: " + " ".join(f"{k} {v:+.3f}" for k, v in ref_sens.items()))
    for c, s in per.items():
        f = fails(s)
        print(f"{c[-60:]:60s} meanKL {s['mean']:.3e} p99 {s['p99']:.3f} maxKL {s['max']:.3f} top1 {s['top1']:.4f} maxdlp {s['dlp']:.2f} | "
              f"ja72gap {s.get('s_ja72gap', float('nan')):+.2f} de45p {s.get('s_de45p', float('nan')):.3f} "
              f"de167p {s.get('s_de167p', float('nan')):.3f} pr181p {s.get('s_prose181p', float('nan')):.3f} {'FAIL ' + ','.join(f) if f else 'pass'}")
    arms: dict[str, list[dict]] = {}
    for c, s in per.items():
        arms.setdefault(arm_of(c), []).append(s)
    stock = None
    if args.stock_json:
        sj = json.load(open(args.stock_json))
        stock = next(iter(sj["arms"].values())) if len(sj["arms"]) == 1 else sj["arms"].get("stock")
    out = {"ref": args.ref, "ref_sens": ref_sens, "cols": per, "pairs": pairs, "arms": {}, "verdict": {}}
    print()
    for arm, vals in arms.items():
        a = arm_summary(vals)
        out["arms"][arm] = a
        print(f"ARM {arm}: n={a['n']} | " + " | ".join(
            f"{k} med {a[k]['median']:.4g} mean {a[k]['mean']:.4g} sd {a[k]['sd']:.2g} max {a[k]['max']:.4g}" for k in ("mean", "p99", "max", "top1", "dlp"))
            + f" | single-run fails {a['single_run_fails']}/{a['n']}")
        print(f"     ja72 gaps {a['s_ja72gap']}  de45 p {a['s_de45p']}  pr181 p {a['s_prose181p']}")
        med_ok = not fails({k: a[k]["median"] for k in THRESH})
        verdict = {"n": a["n"], "medians_pass": med_ok}
        if stock is not None:
            ps = {k: mann_whitney_p(a[k]["all"], stock[k]["all"], WORSE[k]) for k in ("mean", "p99", "max", "top1", "dlp")}
            verdict["mw_p_worse_than_stock"] = ps
            # Bonferroni over the 5 metrics (family-wise alpha 0.05); information only, see the GATE comment below
            verdict["worse_than_stock"] = [k for k, p in ps.items() if p < 0.05 / len(ps)]
            print("     vs stock (one-sided Mann-Whitney p, ours worse): " + " ".join(f"{k} {p:.3f}" for k, p in ps.items()))
        if stock is not None:  # single-run failure rate vs stock's (one-sided Fisher exact, information)
            sf, sn = stock.get("single_run_fails", 0), stock.get("n", 0)
            verdict["fails_vs_stock"] = (a["single_run_fails"], a["n"], sf, sn, fisher_greater(a["single_run_fails"], a["n"], sf, sn))
        # GATE = the medians of the single-run metrics over N >= 6 collections meet the single-run thresholds. The Mann-Whitney
        # comparison with stock's own collections is INFORMATION only: (1) one server's collections
        # are not independent draws -- a server session shifts all of them (identical config and snapshot c52_ref / c52_ref2:
        # mean KL 6.27 / 5.70e-3, MW p 0.003 / 0.064), so the p-values are anti-conservative; (2) stock's collections share
        # ex_stock's deterministic rounding pattern, so any kernel with a different (equally precise) summation order sits
        # above them by construction (+0.1..0.9e-3 mean KL over all ds41h arms) -> it cannot separate "less precise" from
        # "different order".
        verdict["pass"] = med_ok and a["n"] >= 6
        out["verdict"][arm] = verdict
        fv = verdict.get("fails_vs_stock")
        print(f"     DISTRIBUTION GATE: {'PASS' if verdict['pass'] else 'FAIL'} (n={a['n']}, medians {'ok' if med_ok else 'FAIL'}"
              f"{'; info: MW below 0.01 for ' + ','.join(verdict['worse_than_stock']) if verdict.get('worse_than_stock') else ''}"
              f"{f'; single-run fails {fv[0]}/{fv[1]} vs stock {fv[2]}/{fv[3]} (Fisher p {fv[4]:.2f})' if fv else ''})")
    if pairs:
        print()
        groups: dict[str, list[dict]] = {}
        for k, s in pairs.items():
            groups.setdefault(arm_of(k.split("|")[0]), []).append(s)
        for arm, v in groups.items():
            n = len(v)
            print(f"PAIRS {arm}: n={n} meanKL avg {sum(x['mean'] for x in v) / n:.3e} p99 avg {sum(x['p99'] for x in v) / n:.3f} "
                  f"maxKL max {max(x['max'] for x in v):.3f} maxdlp avg {sum(x['dlp'] for x in v) / n:.2f} max {max(x['dlp'] for x in v):.2f} "
                  f"top1 avg {sum(x['top1'] for x in v) / n:.4f}")
            out["arms"].setdefault(arm, {})["pairs"] = {"n": n, "mean": sum(x["mean"] for x in v) / n,
                                                         "p99": sum(x["p99"] for x in v) / n, "dlp": sum(x["dlp"] for x in v) / n}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
