"""Compare two served runs' PHYS_STEP_SHAPES logs by step kind (where does ASYNC lose to sync
under a 192-user mixed load?). Cycle = the next step's start - this step's start (the worker.step call time).
From fpr/:
  python adapter/tests/shapes_cmp.py A.shapes.jsonl B.shapes.jsonl [--skip 0.1]"""
import json
import statistics as S
import sys


def load(p, skip):
    rs = [json.loads(ln) for ln in open(p) if ln.strip()]
    rs.sort(key=lambda r: r["t"])
    for a, b in zip(rs, rs[1:]):
        a["cyc"] = (b["t"] - a["t"]) * 1e3
    rs = rs[:-1]
    n = len(rs)
    return rs[int(n * skip):int(n * (1 - skip))]


def summ(name, rs):
    tot = sum(r["cyc"] for r in rs)
    dec = [r for r in rs if sum(r["pf"]) == 0 and sum(r["rows"]) > 0]
    mix = [r for r in rs if sum(r["pf"]) > 0]
    emp = [r for r in rs if sum(r["rows"]) == 0]
    cp = [r for r in mix if sum(r.get("cp", [0])) > 0]
    cop = [r for r in rs if r.get("copies", 0)]
    print(f"== {name}: {len(rs)} steps over {tot / 1e3:.1f} s")
    for lab, g in (("decode-only", dec), ("mixed", mix), ("  of which CP", cp), ("empty", emp), ("with copies", cop)):
        if not g:
            print(f"  {lab:14s} 0")
            continue
        t = sum(r["cyc"] for r in g)
        print(f"  {lab:14s} {len(g):6d} steps  {t / 1e3:7.1f} s ({100 * t / tot:4.1f} %)  cycle p50 {S.median(r['cyc'] for r in g):7.2f} ms"
              f"  mean {t / len(g):7.2f}  est p50 {S.median(r['est_ms'] for r in g):7.2f}"
              f"  seqs p50 {S.median(sum(r['dec_seqs']) for r in g):5.0f}  pf rows/step {S.mean(sum(r['pf']) for r in g):7.0f}")
    # decode-only cycle by decode-seq bucket
    bk = {}
    for r in dec:
        s = sum(r["dec_seqs"])
        bk.setdefault(min(s // 32, 7), []).append(r)
    print("  decode-only cycle p50 by decode seqs: " + ", ".join(
        f"{32 * k}-{32 * k + 31}: {S.median(r['cyc'] for r in v):.1f} ms (est {S.median(r['est_ms'] for r in v):.1f}, n {len(v)})"
        for k, v in sorted(bk.items())))
    # the step after a mixed step
    after = [rs[i + 1] for i, r in enumerate(rs[:-1]) if sum(r["pf"]) > 0 and sum(rs[i + 1]["pf"]) == 0]
    if after:
        print(f"  decode step right after a mixed step: cycle p50 {S.median(r['cyc'] for r in after):.2f} ms (n {len(after)})")
    pfrows = sum(sum(r["pf"]) for r in rs)
    drows = sum(sum(r["rows"]) - sum(r["pf"]) for r in rs)
    print(f"  prefill rows {pfrows:,} ({pfrows / tot * 1e3:,.0f}/s), decode rows {drows:,} ({drows / tot * 1e3:,.0f}/s)")
    mt = sum(r["cyc"] for r in mix)
    print(f"  mixed: {len(mix) / tot * 1e3:.2f} steps/s, prefill rows per mixed-step second {pfrows / max(mt, 1e-9) * 1e3:,.0f}")


skip = float(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0.1
for p in [a for a in sys.argv[1:] if a.endswith(".jsonl")]:
    summ(p.split("/")[-2] + "/" + p.split("/")[-1], load(p, skip))
