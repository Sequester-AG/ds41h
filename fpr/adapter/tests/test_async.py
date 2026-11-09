"""The vLLM side of D13 on the CPU: async scheduling (PhysAsyncScheduler + PhysExecutor's step thread, vLLM's
step_with_batch_queue with 2 batches in flight) against the synchronous path, with the fake worker emulating the
device-side row expansion (decode positions resolved from its own state; the host's are an upper bound).

  CUDA_VISIBLE_DEVICES= PHYS_MODEL=<model dir> python -m pytest -q adapter/tests/test_async.py

1. the same request mix through a sync and an async engine gives IDENTICAL outputs (token ids, finish reasons, stop
   trimming, grammar, forcing), each equal to the fake model's stream; every step runs the invariant checks;
2. preemption + moves under pressure, async;
3. host time hidden: with a simulated 35 ms GPU step the async loop runs at ~35 ms per step, the sync loop at 35 ms +
   the host time.
4. the D13 dispatch model (PHYS_FAKE_HANDLES=1: worker.step on the engine loop's thread queues N+1 and waits
   for N, returns a StepHandle; prefill / forced steps run synchronously): the same outputs as sync, and host time
   hidden.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_util as U  # noqa: E402
from test_contract import Harness, prompt  # noqa: E402

import pytest  # noqa: E402
from vllm import SamplingParams  # noqa: E402

pytestmark = pytest.mark.vllm      # vLLM on the CPU, fake worker: needs PHYS_MODEL (conftest.py)
from vllm.sampling_params import StructuredOutputsParams  # noqa: E402


class AsyncHarness(Harness):
    def __init__(self, geometry: str, **env):
        old = {k: os.environ.get(k) for k in ("PHYS_ASYNC", *env)}
        os.environ["PHYS_ASYNC"] = "1"
        os.environ.update(env)
        try:
            super().__init__(geometry, async_scheduling=True, scheduler_cls="adapter.phys_scheduler.PhysAsyncScheduler")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        assert self.core.batch_queue is not None and self.core.batch_queue_size == 2
        assert self.sched.async_mode and self.core.model_executor.async_mode


def _save(name: str, d: dict) -> None:
    """Timing results next to the tests' local results dir, when it exists (not in a fresh checkout)."""
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    if os.path.isdir(out):
        with open(os.path.join(out, name), "w") as f:
            json.dump(d, f)


def _mix(h: Harness, tag: str, ff: str | None = None):
    """Requests covering the lifecycle: lengths, stop tokens inside accepted groups, stop strings, grammar, n = 2."""
    reqs = []
    for i in range(24):
        reqs.append((f"{tag}len{i}", prompt(f"len{i}", 50 + (i * 977) % 5000), SamplingParams(max_tokens=1 + (i * 37) % 200)))
    for i in range(10):
        ids = prompt(f"stop{i}", 300 + 40 * i)
        stream = h.fake.expected(ids, 120)
        tok = stream[5 + (i * 13) % 100]
        reqs.append((f"{tag}stop{i}", ids, SamplingParams(max_tokens=150, stop_token_ids=[tok])))
    pat = r"(yes|no)[0-9]{3,40}"
    for i in range(5):
        ids = h.tok.encode(f"Request {i}: answer yes or no, then digits.", add_special_tokens=False)
        reqs.append((f"{tag}so{i}", ids, SamplingParams(max_tokens=60, structured_outputs=StructuredOutputsParams(regex=pat))))
    return reqs


def _key(o):
    return [(list(c.token_ids), c.finish_reason, c.stop_reason) for c in o.outputs]


@pytest.fixture(scope="module")
def sync_h():
    return Harness("kv:2e9")


@pytest.fixture(scope="module")
def async_h():
    return AsyncHarness("kv:2e9")


def test_async_equals_sync(sync_h, async_h):
    rs = sync_h.run(_mix(sync_h, "x"))
    ra = async_h.run(_mix(async_h, "x"))
    assert set(rs) == set(ra)
    diff = [rid for rid in rs if _key(rs[rid]) != _key(ra[rid])]
    assert not diff, (diff[:5], [(_key(rs[r])[0][1], _key(ra[r])[0][1]) for r in diff[:3]])
    for rid, o in ra.items():
        if "so" in rid:
            text = async_h.tok.decode(list(o.outputs[0].token_ids), skip_special_tokens=True)
            assert re.fullmatch(r"(yes|no)[0-9]{3,40}", text), (rid, text)
        else:
            async_h.check_stream(o)
    assert async_h.fake.dev_groups > 0, "the async run never planned ahead of the device"
    async_h.idle()
    sync_h.idle()


def test_async_abort_and_prefix(async_h):
    base = prompt("aconv", 6000)
    o1 = async_h.run([("ac1", base, SamplingParams(max_tokens=200))])["ac1"]
    follow = base + list(o1.outputs[0].token_ids) + prompt("aturn2", 500)
    long = [(f"aab{i}", prompt(f"aab{i}", 9000 + 2000 * i), SamplingParams(max_tokens=300)) for i in range(4)]
    async_h.add(long + [("ac2", follow, SamplingParams(max_tokens=40))])
    done = {}
    for step in range(20000):
        for o in async_h.eng.step():
            if o.finished:
                done[o.request_id] = o
        if step == 3:
            async_h.eng.abort_request(["aab0"])
        if step == 30:
            async_h.eng.abort_request(["aab1"])
        if not async_h.eng.has_unfinished_requests():
            break
    assert done["ac2"].num_cached_tokens >= (len(base) + 200) // 64 * 64 - 64
    async_h.check_stream(done["ac2"], 40)
    for rid in ("aab2", "aab3"):
        async_h.check_stream(done[rid], 300)
    async_h.idle()


def test_async_force_protocol(tmp_path):
    ff = tmp_path / "force.json"
    h = AsyncHarness("kv:2e9", DS41H_FORCE_FILE=str(ff))
    p = prompt("aforced", 2500)
    forced = [2000 + (i * 37) % 40000 for i in range(216)]
    ff.write_text(json.dumps({"tag": "at", "prompt_ids": p, "forced": forced, "n_dump": 200, "dump_dir": str(tmp_path / "d")}))
    done = h.run([("f", p, SamplingParams(max_tokens=200, ignore_eos=True, temperature=0)),
                  ("o", prompt("abystander", 700), SamplingParams(max_tokens=150))])
    assert list(done["f"].outputs[0].token_ids) == forced[:200]
    h.check_stream(done["o"], 150)
    meta = json.loads((tmp_path / "d" / "at" / "rank0.json").read_text())
    assert meta["n"] == 200 and meta["input_mismatch"] == 0 and not meta["partial"], meta
    h.idle()


def test_async_preemption():
    # tiny pools found two serve bugs (fixed: paged allocation beyond the admission reservation; an async
    # preemption livelock); serve's test_no_prefill_deadlock_tight_pools runs this config in both modes
    h = AsyncHarness("small:160,96,12")
    reqs = [(f"ap{i}", prompt(f"p{i}", 2500 + 700 * (i % 5)), SamplingParams(max_tokens=1500)) for i in range(14)]
    done = h.run(reqs, max_steps=8000)
    for rid, ids, sp in reqs:
        h.check_stream(done[rid], 1500)
    st = h.sched.rs.stats
    assert st["preemptions"] > 0, st              # async mode does not move decoding sequences (serve D13)
    h.idle()


def _decode_wall(h: Harness, n: int, steps: int) -> float:
    for i in range(n):
        h.eng.add_request(f"t{i}", {"prompt_token_ids": prompt(f"timing{i}", 4000)},
                          SamplingParams(max_tokens=100000, ignore_eos=True))
    while len(h.sched.rs.running) < n or any(q.phase != "decode" for q in h.sched.rs.running):
        h.eng.step()
    for _ in range(5):
        h.eng.step()
    t0 = time.perf_counter()
    k = 0
    while k < steps:
        h.eng.step()
        k += 1
    wall = (time.perf_counter() - t0) / steps
    h.eng.abort_request([f"t{i}" for i in range(n)])
    while h.eng.has_unfinished_requests():
        h.eng.step()
    return wall


def test_host_time_hidden():
    """Simulated GPU step of 35 ms, 128 decoding sequences: sync = 35 ms + host, async ~= max(35 ms, host)."""
    os.environ["PHYS_FAKE_STEP_MS"] = "35"
    os.environ["PHYS_ADAPTER_CHECK"] = "0"
    try:
        hs = Harness("kv:2e9")
        ha = AsyncHarness("kv:2e9")
    finally:
        os.environ.pop("PHYS_FAKE_STEP_MS", None)
        os.environ["PHYS_ADAPTER_CHECK"] = "1"
    ws = _decode_wall(hs, 128, 60)
    # best of 3: the step thread shares the GIL with the loop thread, so a loaded host stretches single windows
    # (one run in three over the bound at load ~4 on a shared machine; the D13 dispatch model below is not affected)
    wa = min(_decode_wall(ha, 128, 60) for _ in range(3))
    print(f"\n128 decoding sequences, simulated GPU step 35 ms: sync {ws * 1e3:.1f} ms/step, async {wa * 1e3:.1f} ms/step")
    _save("async_timing.json", dict(seqs=128, gpu_ms=35.0, sync_ms=round(ws * 1e3, 2), async_ms=round(wa * 1e3, 2)))
    assert wa < ws, (wa, ws)
    # host time hidden: within 2.5 ms of the simulated GPU step, or (a slower host) at least half of sync's host time gone
    assert wa < max(0.0375, 0.035 + 0.5 * (ws - 0.035)), (wa, ws)


@pytest.fixture(scope="module")
def handle_h():
    return AsyncHarness("kv:2e9", PHYS_FAKE_HANDLES="1")


def test_handles_equal_sync(sync_h, handle_h):
    """D13 dispatch model: worker.step on the engine loop thread, handle.result() merged on the pool thread."""
    ex = handle_h.core.model_executor
    assert ex._main_dispatch and handle_h.fake.handles
    rs = sync_h.run(_mix(sync_h, "h"))
    rh = handle_h.run(_mix(handle_h, "h"))
    assert set(rs) == set(rh)
    diff = [rid for rid in rs if _key(rs[rid]) != _key(rh[rid])]
    assert not diff, (diff[:5], [(_key(rs[r])[0][1], _key(rh[r])[0][1]) for r in diff[:3]])
    assert handle_h.fake.dev_groups > 0 and handle_h.fake.sync_steps > 0
    handle_h.idle()
    sync_h.idle()


def test_handles_force_and_abort(tmp_path):
    ff = tmp_path / "force.json"
    h = AsyncHarness("kv:2e9", PHYS_FAKE_HANDLES="1", DS41H_FORCE_FILE=str(ff))
    p = prompt("hforced", 2500)
    forced = [2000 + (i * 41) % 40000 for i in range(216)]
    ff.write_text(json.dumps({"tag": "ht", "prompt_ids": p, "forced": forced, "n_dump": 200, "dump_dir": str(tmp_path / "d")}))
    h.add([("f", p, SamplingParams(max_tokens=200, ignore_eos=True, temperature=0)),
           ("o", prompt("hbystander", 700), SamplingParams(max_tokens=150)),
           ("a", prompt("habort", 9000), SamplingParams(max_tokens=300))])
    done = {}
    for step in range(20000):
        for o in h.eng.step():
            if o.finished:
                done[o.request_id] = o
        if step == 5:
            h.eng.abort_request(["a"])
        if not h.eng.has_unfinished_requests():
            break
    assert list(done["f"].outputs[0].token_ids) == forced[:200]
    h.check_stream(done["o"], 150)
    meta = json.loads((tmp_path / "d" / "ht" / "rank0.json").read_text())
    assert meta["n"] == 200 and meta["input_mismatch"] == 0 and not meta["partial"], meta
    h.idle()


def test_handles_host_time_hidden():
    """D13 dispatch model, simulated 35 ms GPU step, 128 decoding sequences: ~max(35 ms, host) per step."""
    os.environ["PHYS_FAKE_STEP_MS"] = "35"
    os.environ["PHYS_ADAPTER_CHECK"] = "0"
    try:
        hs = Harness("kv:2e9")
        hh = AsyncHarness("kv:2e9", PHYS_FAKE_HANDLES="1")
    finally:
        os.environ.pop("PHYS_FAKE_STEP_MS", None)
        os.environ["PHYS_ADAPTER_CHECK"] = "1"
    ws = _decode_wall(hs, 128, 60)
    wh = _decode_wall(hh, 128, 60)
    print(f"\n128 decoding sequences, simulated GPU step 35 ms, D13 dispatch model: sync {ws * 1e3:.1f} ms/step, "
          f"dispatch {wh * 1e3:.1f} ms/step")
    _save("async_timing_d13.json", dict(seqs=128, gpu_ms=35.0, sync_ms=round(ws * 1e3, 2), dispatch_ms=round(wh * 1e3, 2)))
    assert wh < ws, (wh, ws)
    assert wh < 0.0375, wh


def _hold_mix(h: Harness, tag: str, n_plain: int = 16, n_so: int = 2):
    """Mostly plain requests + a few structured ones (production-like traffic has a few % strict-tool calls; here ~11 %)."""
    reqs = [(f"{tag}p{i}", prompt(f"{tag}p{i}", 400 + 53 * i), SamplingParams(max_tokens=120)) for i in range(n_plain)]
    pat = r"(yes|no)[0-9]{30,60}"
    for i in range(n_so):
        ids = h.tok.encode(f"Hold {tag} {i}: answer yes or no, then digits.", add_special_tokens=False)
        reqs.append((f"{tag}so{i}", ids, SamplingParams(max_tokens=80, structured_outputs=StructuredOutputsParams(regex=pat))))
    return reqs


def _check_hold_run(h, res, sync_res):
    assert set(res) == set(sync_res)
    diff = [rid for rid in res if _key(res[rid]) != _key(sync_res[rid])]
    assert not diff, (diff[:5], [(_key(sync_res[r])[0][1], _key(res[r])[0][1]) for r in diff[:3]])
    for rid, o in res.items():
        if "so" in rid:
            text = h.tok.decode(list(o.outputs[0].token_ids), skip_special_tokens=True)
            assert re.fullmatch(r"(yes|no)[0-9]{30,60}", text), (rid, text)
        else:
            h.check_stream(o)


@pytest.mark.parametrize("handles", ["0", "1"])
def test_grammar_hold(sync_h, handles):
    """ASYNC + structured outputs (PHYS_ASYNC_GRAMMAR=hold, the default): in-flight structured requests skip a step
    instead of making vLLM defer the step's sampling; their resume step gets host-given ids and grammar-validated
    drafts. Outputs equal sync; almost no step is deferred (at most the first in-flight step of each structured request,
    before its drafts were seen). Control: PHYS_ASYNC_GRAMMAR=defer defers steps."""
    rs = sync_h.run(_hold_mix(sync_h, f"g{handles}"))
    sync_h.idle()
    got = {}
    for mode in ("hold", "defer"):
        h = AsyncHarness("kv:2e9", PHYS_FAKE_HANDLES=handles, PHYS_ASYNC_GRAMMAR=mode)
        st0 = dict(h.sched.phys_stats)
        ra = h.run(_hold_mix(h, f"g{handles}"))
        _check_hold_run(h, ra, rs)
        st = h.sched.phys_stats
        d = {k: st.get(k, 0) - st0.get(k, 0) for k in ("grammar_held", "grammar_host_ids", "grammar_resume_no_ids",
                                                         "deferred_steps", "steps")}
        print(f"\nhandles={handles} {mode}: {d}")
        got[mode] = d
        h.idle()
    # hold: deferrals only at each structured request's first in-flight step and at the tail (structured > the cap)
    # every resume after a hold has host-given ids (a request may also finish in its held step and never resume)
    assert got["hold"]["grammar_held"] > 0 and got["hold"]["grammar_resume_no_ids"] == 0, got
    assert got["hold"]["grammar_host_ids"] >= got["hold"]["grammar_held"] - 2, got
    assert got["defer"]["grammar_held"] == 0 and got["defer"]["deferred_steps"] > 10, got
    assert 4 * got["hold"]["deferred_steps"] < got["defer"]["deferred_steps"], got


def _hold_wall(h: Harness, n_plain: int, n_so: int, steps: int) -> float:
    """Wall per scheduled step while n_plain + n_so sequences decode (the structured ones under a long regex)."""
    for i in range(n_plain):
        h.eng.add_request(f"hp{i}", {"prompt_token_ids": prompt(f"holdt{i}", 2000)},
                          SamplingParams(max_tokens=100000, ignore_eos=True))
    for i in range(n_so):
        ids = h.tok.encode(f"Timing {i}: digits.", add_special_tokens=False)
        h.eng.add_request(f"hs{i}", {"prompt_token_ids": ids},
                          SamplingParams(max_tokens=3000, structured_outputs=StructuredOutputsParams(regex=r"[0-9]{2900}")))
    n = n_plain + n_so
    while len(h.sched.rs.running) < n or any(q.phase != "decode" for q in h.sched.rs.running):
        h.eng.step()
    for _ in range(8):
        h.eng.step()
    s0, t0 = h.sched.phys_stats["steps"], time.perf_counter()
    while h.sched.phys_stats["steps"] - s0 < steps:
        h.eng.step()
    wall = (time.perf_counter() - t0) / (h.sched.phys_stats["steps"] - s0)
    h.eng.abort_request([f"hp{i}" for i in range(n_plain)] + [f"hs{i}" for i in range(n_so)])
    while h.eng.has_unfinished_requests():
        h.eng.step()
    return wall


def test_grammar_hold_time():
    """Simulated 35 ms GPU step, D13 dispatch model, 64 plain + 3 structured decoding sequences: with the hold the loop
    stays at ~35 ms per step; vLLM's deferral serializes it (35 ms + host per step)."""
    os.environ["PHYS_FAKE_STEP_MS"] = "35"
    os.environ["PHYS_ADAPTER_CHECK"] = "0"
    out = {}
    try:
        for mode in ("hold", "defer"):
            h = AsyncHarness("kv:2e9", PHYS_FAKE_HANDLES="1", PHYS_ASYNC_GRAMMAR=mode)
            st0 = dict(h.sched.phys_stats)
            out[mode] = _hold_wall(h, 64, 3, 60)
            out[mode + "_deferred"] = h.sched.phys_stats.get("deferred_steps", 0) - st0.get("deferred_steps", 0)
            h.idle()
    finally:
        os.environ.pop("PHYS_FAKE_STEP_MS", None)
        os.environ["PHYS_ADAPTER_CHECK"] = "1"
    print(f"\n64 + 3 structured decoding, simulated 35 ms step: hold {out['hold'] * 1e3:.1f} ms/step, "
          f"defer {out['defer'] * 1e3:.1f} ms/step ({out})")
    _save("grammar_hold_timing.json", {k: round(v * 1e3, 2) if isinstance(v, float) else v for k, v in out.items()})
    assert out["hold"] < out["defer"], out
    assert out["hold"] < 0.0385, out
