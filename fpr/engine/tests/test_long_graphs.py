"""Decode steps with a context past 262,144 positions replay a CUDA graph, bit for bit equal to the same step issued eagerly.

Bug guarded: the decode graphs' fast indexer handles rows up to 262,144 positions. A single longer sequence used to
send the WHOLE decode step down the eager mixed-step path (about +25 ms per step for every user). The engine now keeps
two more graph variants per bucket (level 1 up to 524,288 positions: the layers that reach past the fast limit use the
stock select / top-k; level 2 up to 655,360: stock everywhere), captured on a bucket's first (eager) step and replayed
after. The stock top-k writes its 512-set in a run-dependent order, so the engine sorts each list (IDX2F_CANON_STOCK):
without that, replay and eager steps summed the attention in different orders.

Workload: one sequence at 300,000 (level 1) or 530,000 (level 2) positions plus 23 at 8-30K (96 decode rows: the
graphed long variants need more than 64 rows in the step), 6 steps, tie-free synthetic KV. Run A replays the captured
long-variant graphs from the second step on; run B retires every captured graph before each step, so every step is
issued eagerly (and captured again). Head inputs and tokens must be identical at every step.
(The eager MIXED-step path is a different set of kernels; it is not compared bit for bit here.)"""
from __future__ import annotations

import pytest

STEPS = 6


@pytest.fixture(scope="module")
def eng():
    import drive
    return drive.engine_once("sync", "")


def _wl(long_ctx: int):
    import drive
    return drive.Workload(ctxs=[long_ctx] + [8_000 + 1_000 * i for i in range(23)], steps=STEPS)


def _retire_all_graphs(k, plan, eng):
    """on_plan hook: before every step after the first, retire (never destroy) every captured graph, so the step runs
    eagerly as a bucket's first step does."""
    if k > 0:
        for key in list(eng.graphs):
            eng._retired.append(eng.graphs.pop(key))


@pytest.mark.gpu(n=4)
@pytest.mark.parametrize("long_ctx", [300_000, 530_000])
def test_long_context_step_is_graphed_and_matches_eager(eng, long_ctx):
    import drive
    from engine import engine as E
    assert E.LONG_GRAPHS
    g = drive.run(eng, _wl(long_ctx))
    e = drive.run(eng, _wl(long_ctx), on_plan=_retire_all_graphs)
    assert len(g.steps) == len(e.steps) == STEPS
    keys = ("long_steps", "eager_steps", "mixed_steps")
    gs, es = ({k: x.stats.get(k, 0) for k in keys} for x in (g, e))
    # A: every step on the long variant, only the first eager (then captured), the rest replays; B: every step eager
    assert gs == dict(long_steps=STEPS, eager_steps=1, mixed_steps=0), gs
    assert es == dict(long_steps=STEPS, eager_steps=STEPS, mixed_steps=0), es
    for x in (g, e):
        assert all(s["nonfinite"] == 0 for s in x.steps)
    diff = [(s["step"], [r for r in range(4) if s["h_rank"][r] != t["h_rank"][r]])
            for s, t in zip(g.steps, e.steps) if s["h_md5"] != t["h_md5"] or s["tok_md5"] != t["tok_md5"]]
    bad = sorted(r for r in g.streams if g.streams[r] != e.streams.get(r))
    assert not diff, f"graph replay and eager steps differ at (step, ranks) {diff}; sequences {bad}"
    assert g.streams == e.streams
