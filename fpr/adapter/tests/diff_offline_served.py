"""Served vs offline worker inputs for the same forced item: does the worker receive the same inputs on both paths?

Runs one synthetic prompt (+ forced continuation) through
  (a) serve's OfflineEngine.force()  (RankScheduler + Worker(backend="engine"), the offline exactness path), and
  (b) the served path: vLLM LLMEngine -> PhysScheduler / PhysExecutor -> the worker wrapper (adapter/stage4_worker.py)
      -> the same Worker,
both on serve's MockEngine (CPU), and diffs per step what the worker receives: chunk boundaries (start, n, last),
positions / paged / window slots, tail and sample rows, cached_upto / replay_start, input tokens, the cost model.

From fpr/ with PYTHONPATH=<fpr>:<repo root>:
  CUDA_VISIBLE_DEVICES= python adapter/tests/diff_offline_served.py --lens 7053,44627 [--cost measured]
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from adapter import bridge  # noqa: E402,F401

MOCK_KW = dict(free_bytes_per_rank=12_000_000_000)
ENGINE_CLS = "shaped_engine.ShapedEngine"     # real engine row geometry + "real" cost rule (no "mock" in the name)


def synth(n: int, seed: int) -> list:
    return [1000 + (seed * 7919 + 104729 * i) % 120000 for i in range(n)]


def entry_view(sb, se, toks=None):
    a, b = se.row0, se.row0 + se.n_rows
    return dict(kind=se.kind, start=se.start, n=se.n_rows, last=bool(se.last_chunk), cached_upto=se.cached_upto,
                replay_start=se.replay_start, tail=[r - a for r in sb.tail_rows if a <= r < b],
                sample=[r - a for r in sb.sample_rows if a <= r < b], positions=list(sb.positions[a:b]),
                paged=list(sb.paged_slot[a:b]), window=list(sb.window_slot[a:b]),
                forced=bool(getattr(se, "forced", False)), tokens=None if toks is None else list(toks[a:b]))


def run_offline(prompt, forced, n_dump, cost):
    import driver as D
    import worker as W

    os.environ.pop("PHYS_COST", None)
    if cost:
        os.environ["PHYS_COST"] = cost
    eng = D.OfflineEngine(backend="engine", cfg=W.WorkerConfig(engine_cls=ENGINE_CLS, engine_kw=MOCK_KW))
    steps = []

    def record(batches):
        for sb in batches:
            for se in sb.seqs:
                v = entry_view(sb, se, None)
                v["tokens"] = [eng._token_at(se.rid, q) if q >= 0 else -1 for q in v["positions"]]
                steps.append(dict(step=sb.step, rank=sb.rank, **v))

    o_run, o_step = eng.worker.run_step, eng.worker.step

    def spy_run(batches):
        record(batches)
        return o_run(batches)

    def spy_step(ps, grammar=None):
        record(ps.batches)
        return o_step(ps, grammar)
    eng.worker.run_step, eng.worker.step = spy_run, spy_step
    eng.force([dict(id="x", prompt_ids=prompt, forced=forced, n_dump=n_dump)])
    cp = eng.sched.cost.p if hasattr(eng.sched.cost, "p") else None
    return steps, cp


def run_served(prompt, forced, n_dump, cost):
    import engine_util as U
    from vllm import SamplingParams

    d = tempfile.mkdtemp(prefix="phys_diff_")
    ff = os.path.join(d, "force.json")
    with open(ff, "w") as f:
        json.dump(dict(tag="x", prompt_ids=prompt, forced=forced, n_dump=n_dump, dump_dir=os.path.join(d, "dump")), f)
    env = dict(PHYS_ADAPTER_WORKER="stage4", PHYS_WORKER_BACKEND="engine", PHYS_ENGINE_CLS=ENGINE_CLS,
               PHYS_ENGINE_KW="free_bytes_per_rank=12000000000", DS41H_FORCE_FILE=ff,
               PHYS_DUMP_STEPS=os.path.join(d, "steps") + ":100000", PHYS_ADAPTER_CHECK="0")
    if cost:
        env["PHYS_COST"] = cost
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        eng = U.make_engine()
        eng.add_request("x", {"prompt_token_ids": prompt}, SamplingParams(max_tokens=n_dump, ignore_eos=True, temperature=0))
        while eng.has_unfinished_requests():
            eng.step()
        core = U.core_of(eng)
        cp = core.scheduler.rs.cost.p if hasattr(core.scheduler.rs.cost, "p") else None
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    steps = []
    sd = os.path.join(d, "steps")
    from adapter import phys_executor as PE
    PE._DUMP["n"] = 0                               # the dump counter is per process
    for fn in sorted(os.listdir(sd), key=lambda x: int(x[4:-4])):
        ps = pickle.load(open(os.path.join(sd, fn), "rb"))["ps"]
        for sb, toks in zip(ps.batches, ps.row_tokens):
            for se in sb.seqs:
                steps.append(dict(step=sb.step, rank=sb.rank, prev=ps.prev_tokens.get(se.rid),
                                  **entry_view(sb, se, toks)))
    shutil.rmtree(d, ignore_errors=True)
    return steps, cp


def summarize(steps):
    pf = [(s["start"], s["n"], s["last"]) for s in steps if s["kind"] == "prefill"]
    dec = [(s["start"], s["n"]) for s in steps if s["kind"] == "decode"]
    return pf, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="7053,44627")
    ap.add_argument("--forced", type=int, default=40)
    ap.add_argument("--cost", default="", help="PHYS_COST for both paths (e.g. measured); default: each path's own")
    a = ap.parse_args()
    for L in [int(x) for x in a.lens.split(",")]:
        prompt = synth(L, L)
        forced = synth(a.forced + 8, L + 1)
        n_dump = a.forced
        off, cp_off = run_offline(prompt, forced, n_dump, a.cost)
        srv, cp_srv = run_served(prompt, forced, n_dump, a.cost)
        pf_o, dec_o = summarize(off)
        pf_s, dec_s = summarize(srv)
        print(f"\n=== prompt {L} tokens, forced {n_dump}")
        print("cost params offline:", cp_off)
        print("cost params served: ", cp_srv)
        print(f"prefill chunks offline ({len(pf_o)}): {pf_o[:12]}{' ...' if len(pf_o) > 12 else ''}")
        print(f"prefill chunks served  ({len(pf_s)}): {pf_s[:12]}{' ...' if len(pf_s) > 12 else ''}")
        print(f"decode groups offline {len(dec_o)}, served {len(dec_s)}; first offline {dec_o[:3]} served {dec_s[:3]}")
        # per-chunk field diffs where the chunk boundaries agree
        oi = {(s["kind"], s["start"], s["n"]): s for s in off}
        nd, nsame = 0, 0
        for s in srv:
            o = oi.get((s["kind"], s["start"], s["n"]))
            if o is None:
                continue
            nsame += 1
            for k in ("last", "cached_upto", "replay_start", "tail", "sample", "positions", "paged", "window", "forced",
                      "tokens"):
                if o[k] != s[k]:
                    nd += 1
                    if k == "tokens":
                        diff = [i for i, (x, y) in enumerate(zip(o[k], s[k])) if x != y]
                        print(f"  DIFF {s['kind']} start {s['start']} {k}: {len(diff)} rows differ, first {diff[:5]} "
                              f"offline {[o[k][i] for i in diff[:3]]} served {[s[k][i] for i in diff[:3]]}")
                    else:
                        print(f"  DIFF {s['kind']} start {s['start']} {k}: offline {str(o[k])[:120]} served {str(s[k])[:120]}")
        print(f"entries with identical (kind, start, n): {nsame}; field differences among them: {nd}")
        pv = [(s["start"], s["prev"]) for s in srv if s["kind"] == "prefill"][:4]
        print("served prev_tokens at prefill chunk starts:", pv, "| prompt[start-3:start]:",
              [(st, prompt[max(0, st - 3):st]) for st, _ in pv])


if __name__ == "__main__":
    main()
