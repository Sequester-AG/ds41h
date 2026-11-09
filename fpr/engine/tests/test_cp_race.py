"""CP prefill under decode load: finite results and run-to-run bitwise repeatability.

Bug guarded: in a CP (context-parallel) prefill step, each rank sends KV sections of its sub-chunk to the later ranks
(engine/cp.py, peer copies on the sender's stream). The receive buffers were allocated with torch.empty on the
receiving device, i.e. from the pool of the receiver's engine stream. The caching allocator may hand out a block that
kernels already queued on that stream are still using, and the sender's copy is not ordered after them. Under load
(two CP groups in one step, frequent eviction of mixed-step buffers) a received section was overwritten or read
half-written: the owning rank's tail rows went NaN and the prompt decoded NaN from then on; without NaN, repeats of
the same workload diverged. The fix allocates receive buffers from a CP-only stream's pool, held until the step ends.

Workload (synthetic, deterministic): 4 seeded decoders, then 56 + 56 prompts of 2,000 tokens (steps 4 and 8) and two
30,000-token prompts (step 10) that prefill with CP while ~116 sequences decode; 56 steps, mixed-step buffer LRU of 2.
Checks: every sampled head input is finite, CP steps with two CP groups occur, and a second run in the same process
reproduces every step's head inputs and tokens bit for bit.
"""
from __future__ import annotations

import pytest

STEPS = 56


@pytest.fixture(scope="module")
def eng():
    import drive
    from engine import engine as E
    E.FREE_CACHE_BYTES = int(0.5e9)          # release idle allocator cache often: more block reuse, as under load
    return drive.engine_once("cp", "upload_v2=1,max_mixed_buckets=2")


def _wl():
    import drive
    return drive.Workload(ctxs=[2_000] * 4, steps=STEPS, arrive={4: [2_000] * 56, 8: [2_000] * 56, 10: [30_000] * 2},
                          arrive_out=140, cp=True, cp_min=2048)


@pytest.fixture(scope="module")
def runs(eng):
    import drive
    return [drive.run(eng, _wl(), mode="sync") for _ in range(2)]


@pytest.mark.gpu(n=4)
def test_cp_steps_happen(runs):
    """The workload exercises the race window: CP steps, some with two CP prompts at once."""
    r = runs[0]
    assert r.stats.get("cp_steps", 0) > 0 and r.sched_stats.get("cp_chunks", 0) > 0
    groups = [s["cp_groups"] for s in r.steps]
    assert max(groups) >= 2, f"no step with two CP prompts at once (CP prompts per step: {groups})"


@pytest.mark.gpu(n=4)
def test_all_head_inputs_finite(runs):
    for i, r in enumerate(runs):
        bad = [(s["step"], s["nonfinite"]) for s in r.steps if s["nonfinite"]]
        assert not bad, f"run {i}: non-finite head inputs at (step, rows) {bad}"


@pytest.mark.gpu(n=4)
def test_repeat_is_bitwise(runs):
    a, b = runs
    assert len(a.steps) == len(b.steps) == STEPS
    first_h = next((s["step"] for s, t in zip(a.steps, b.steps) if s["h_md5"] != t["h_md5"]), None)
    first_t = next((s["step"] for s, t in zip(a.steps, b.steps) if s["tok_md5"] != t["tok_md5"]), None)
    assert first_h is None and first_t is None, f"repeat diverges: head inputs from step {first_h}, tokens from {first_t}"
    assert a.streams == b.streams
