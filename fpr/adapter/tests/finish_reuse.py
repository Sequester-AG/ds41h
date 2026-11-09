"""Host-side handoffs after finishes, from a served run's step dumps (PHYS_DUMP_STEPS): for every finished / preempted
request, its last batch entry (rank, state slot, uid, n_blocks, pages, window pages, settled position) and the first
later dispatch where another sequence got the same state slot, any of its paged pages, any of its window pages, or its
uid (under D13's async loop: is a finished request's state reused while a step for it may still be in flight?).

From fpr/ with PYTHONPATH=<fpr>:<repo root>:
  python adapter/tests/finish_reuse.py <dump dir> [--last 14]
Distances are in dispatches after the request's last scheduled step (the wasted in-flight step when it finished
in the async loop). Aggregates and ids only; no token ids are printed.
"""
import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from adapter import bridge  # noqa: E402,F401
from adapter.replay import step_files  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps")
    ap.add_argument("--last", type=int, default=14)
    ap.add_argument("--window", type=int, default=40)
    a = ap.parse_args()
    files = step_files([a.dumps])
    steps, init = [], {}
    for f in files:
        with open(f, "rb") as fh:
            ps = pickle.load(fh)["ps"]
        for ri in ps.new_reqs:
            init[ri.rid] = (len(ri.prompt_token_ids), ri.sampling_params.max_tokens)
        ent = {}
        for sb in ps.batches:
            for se in sb.seqs:
                nb = int(se.n_blocks or len(se.block_table))
                ent[se.rid] = dict(rank=sb.rank, slot=int(se.state_slot), uid=int(getattr(se, "uid", -1)),
                                   inc=int(getattr(se, "inc", 0)), kind=se.kind, start=int(se.start),
                                   known=int(getattr(se, "known", -1)), nb=nb, pages=set(se.block_table[:nb]),
                                   win=set(se.window_table.values()), wmax=max(se.window_table) if se.window_table else -1)
        steps.append((ps.step, set(ps.finished) | set(ps.preempted), ent))
    last_seen = {}
    for i, (_, _, ent) in enumerate(steps):
        for rid in ent:
            last_seen[rid] = i
    fins = [(i, rid) for i, (_, f, _) in enumerate(steps) for rid in f]
    print(f"{len(files)} dispatches, {len(fins)} finished / preempted requests")
    rows = []
    for i_f, rid in fins:
        L = last_seen.get(rid)
        if L is None:
            continue
        e0 = steps[L][2][rid]
        reuse = {}
        for j in range(L + 1, min(len(steps), L + a.window)):
            for orid, e in steps[j][2].items():
                if orid == rid:
                    continue
                same_rank = e["rank"] == e0["rank"]
                if same_rank and e["slot"] == e0["slot"] and "slot" not in reuse:
                    reuse["slot"] = (j - L, str(orid)[-6:], e["kind"])
                if same_rank and e["pages"] & e0["pages"] and "page" not in reuse:
                    reuse["page"] = (j - L, str(orid)[-6:], len(e["pages"] & e0["pages"]))
                if same_rank and e["win"] & e0["win"] and "wpage" not in reuse:
                    reuse["wpage"] = (j - L, str(orid)[-6:], len(e["win"] & e0["win"]))
                if e["uid"] == e0["uid"] and "uid" not in reuse:
                    reuse["uid"] = (j - L, str(orid)[-6:])
        pl, mx = init.get(rid, (None, None))
        rows.append(dict(fin=i_f, last=L, rid=str(rid)[-8:], rank=e0["rank"], slot=e0["slot"], uid=e0["uid"],
                         known=e0["known"], prompt=pl, max_tokens=mx, paged_cover=e0["nb"] * 64,
                         window_cover=32 * (e0["wmax"] + 1), reuse=reuse))
    print("fin  last  rid       rank slot uid  known  prompt max   paged<  window< | first reuse: (dispatches after last, by, n)")
    for x in rows[-a.last:]:
        print(f"{x['fin']:4d} {x['last']:5d} {x['rid']} {x['rank']:4d} {x['slot']:4d} {x['uid']:4d} {x['known']:6d} "
              f"{x['prompt']} {x['max_tokens']} {x['paged_cover']:7d} {x['window_cover']:7d} | {x['reuse']}")
    for k in ("slot", "page", "wpage", "uid"):
        d = [x["reuse"][k][0] for x in rows if k in x["reuse"]]
        early = sum(1 for v in d if v <= 2)
        print(f"{k:5s}: reused within the window for {len(d)} of {len(rows)}; within 2 dispatches of the last step: {early}; "
              f"min distance {min(d) if d else '-'}")
    # the host's coverage at the last step vs where the device could be: prompt + max_tokens (+ the k + 1 rows of an
    # in-flight step, + HSD's 2k + 1 drafter window)
    over = [x for x in rows if x["prompt"] is not None and x["max_tokens"] is not None and
            x["prompt"] + x["max_tokens"] + 3 + 7 > min(x["paged_cover"], x["window_cover"])]
    print(f"requests whose final position + in-flight rows + drafter window could pass the host's paged / window "
          f"coverage at their last step: {len(over)} of {len(rows)}")
    for x in over[:8]:
        print("   ", {k: x[k] for k in ("fin", "last", "rid", "known", "prompt", "max_tokens", "paged_cover", "window_cover")})


if __name__ == "__main__":
    main()
