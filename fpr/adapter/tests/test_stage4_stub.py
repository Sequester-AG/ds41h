"""The stage4 worker path (adapter/stage4_worker.py -> serve/worker.py Worker -> its CPU StubBackend, head_stub tokens)
under vLLM 0.32's engine on the CPU: the wrapper's token lookup (TableBuilder.tokens_of <- PhysStep.row_tokens), the
Worker's copies / tables / StepResults, lengths, aborts, preemption, the FORCE hook, with the same per-step invariant
checks as test_contract.py. Token ids are head_stub's (synthetic), so streams are not compared here.

  CUDA_VISIBLE_DEVICES= PHYS_MODEL=<model dir> python -m pytest -q adapter/tests/test_stage4_stub.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_util as U  # noqa: E402
from test_contract import prompt  # noqa: E402

import pytest  # noqa: E402
from vllm import SamplingParams  # noqa: E402

pytestmark = pytest.mark.vllm      # vLLM on the CPU, fake worker: needs PHYS_MODEL (conftest.py)


def _engine(tmp_path, geom_over="", backend="stub", **extra):
    env = dict(PHYS_ADAPTER_WORKER="stage4", PHYS_WORKER_BACKEND=backend, PHYS_GEOM=geom_over,
               DS41H_FORCE_FILE=str(tmp_path / "force.json"), **extra)
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        eng = U.make_engine()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return eng, U.core_of(eng)


def _drain(eng, max_steps=50000, on_step=None):
    done = {}
    for i in range(max_steps):
        for o in eng.step():
            if o.finished:
                done[o.request_id] = o
        if on_step:
            on_step(i)
        if not eng.has_unfinished_requests():
            return done
    raise AssertionError("not drained")


def test_stage4_stub_lifecycle(tmp_path):
    eng, core = _engine(tmp_path, "max_seqs_per_rank=16,max_prefill_tokens_per_rank=64")
    s = core.scheduler
    assert s.geom.max_seqs_per_rank == 16 and s.geom.max_prefill_tokens_per_rank == 64
    reqs = [(f"r{i}", prompt(f"s4_{i}", 100 + 300 * (i % 9)), SamplingParams(max_tokens=20 + 13 * (i % 11), ignore_eos=True))
            for i in range(90)]
    for rid, ids, sp in reqs:
        eng.add_request(rid, {"prompt_token_ids": ids}, sp)
    seen_rows = []

    def on_step(i):
        if i == 5:
            eng.abort_request(["r3", "r4", "r50"])
    orig = core.model_executor.worker.step

    def spy(ps, grammar):
        seen_rows.append(max((b.n_rows for b in ps.batches), default=0))
        return orig(ps, grammar)
    core.model_executor.worker.step = spy
    done = _drain(eng, on_step=on_step)
    for rid, ids, sp in reqs:
        if rid in ("r3", "r4", "r50"):
            continue
        o = done[rid]
        assert len(o.outputs[0].token_ids) == sp.max_tokens and o.outputs[0].finish_reason == "length"
    assert max(seen_rows) <= 16 * 4 + 64, max(seen_rows)
    assert not s.rs.running and not s.rs.kv.seqs and not s.requests
    s.rs.kv.check()

    # FORCE hook through the worker wrapper: forced ids come out, the wrapper feeds them as draft-row tokens
    p = prompt("s4_forced", 700)
    forced = [3000 + i for i in range(80)]
    ff = s.force.path
    with open(ff, "w") as f:
        json.dump({"tag": "s4", "prompt_ids": p, "forced": forced, "n_dump": 64, "dump_dir": str(tmp_path / "dump")}, f)
    eng.add_request("f", {"prompt_token_ids": p}, SamplingParams(max_tokens=64, ignore_eos=True, temperature=0))
    o = _drain(eng)["f"]
    assert list(o.outputs[0].token_ids) == forced[:64]
    meta = json.loads((tmp_path / "dump" / "s4" / "rank0.json").read_text())
    assert meta["input_mismatch"] == 0 and meta["checked_rows"] > 0, meta


def test_stage4_engine_backend_mockengine(tmp_path):
    """The GPU path's code on the CPU: the worker wrapper -> serve's Worker(backend="engine") -> MockEngine (the worker <->
    engine contract without a model, every invariant of the contract asserted per step). Exercises the
    native Worker.step(ps, grammar), StepContext lifecycle (SeqInit / dropped / grammar), block-table deltas, the
    worker's pool sizing from free_bytes and the FORCE hook's dump through serve/force_dump."""
    eng, core = _engine(tmp_path, backend="engine", PHYS_ENGINE_CLS="mock_engine.MockEngine",
                        PHYS_ENGINE_KW="free_bytes_per_rank=12000000000")
    s = core.scheduler
    w = core.model_executor.worker
    assert w.w.backend.__class__.__name__ == "MockEngine" and w.native
    reqs = [(f"m{i}", prompt(f"mock_{i}", 200 + 700 * (i % 6)), SamplingParams(max_tokens=10 + 17 * (i % 9), ignore_eos=True))
            for i in range(60)]
    for rid, ids, sp in reqs:
        eng.add_request(rid, {"prompt_token_ids": ids}, sp)

    def on_step(i):
        if i == 7:
            eng.abort_request(["m5", "m6"])
    done = _drain(eng, on_step=on_step)
    for rid, ids, sp in reqs:
        if rid in ("m5", "m6"):
            continue
        assert len(done[rid].outputs[0].token_ids) == sp.max_tokens, rid
    assert w.w.backend.steps > 0
    assert not s.rs.running and not s.requests
    s.rs.kv.check()
    p = prompt("mock_forced", 900)
    forced = [4000 + i for i in range(70)]
    with open(s.force.path, "w") as f:
        json.dump({"tag": "mk", "prompt_ids": p, "forced": forced, "n_dump": 50, "dump_dir": str(tmp_path / "dump")}, f)
    eng.add_request("f", {"prompt_token_ids": p}, SamplingParams(max_tokens=50, ignore_eos=True, temperature=0))
    o = _drain(eng)["f"]
    assert list(o.outputs[0].token_ids) == forced[:50]
    meta = json.loads((tmp_path / "dump" / "mk" / "rank0.json").read_text())
    assert meta["n"] == 50 and meta["input_mismatch"] == 0 and not meta["partial"], meta


def test_stage4_async_mockengine(tmp_path):
    """Async scheduling through serve's real Worker: Worker(async_mode) -> build_compact -> MockEngine.step_compact (decode
    rows expanded on the 'device'), Worker.signals_prepared / device_positions, the executor's step thread. MockEngine's
    tokens are synthetic: lengths, finish and the invariant checks are what is compared."""
    old = os.environ.get("PHYS_ASYNC")
    os.environ["PHYS_ASYNC"] = "1"
    try:
        env = dict(PHYS_ADAPTER_WORKER="stage4", PHYS_WORKER_BACKEND="engine", PHYS_ENGINE_CLS="mock_engine.MockEngine",
                   PHYS_ENGINE_KW="free_bytes_per_rank=12000000000", DS41H_FORCE_FILE=str(tmp_path / "force.json"))
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            eng = U.make_engine(async_scheduling=True, scheduler_cls="adapter.phys_scheduler.PhysAsyncScheduler")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    finally:
        if old is None:
            os.environ.pop("PHYS_ASYNC", None)
        else:
            os.environ["PHYS_ASYNC"] = old
    core = U.core_of(eng)
    ex = core.model_executor
    assert ex.async_mode and ex._wait_prepared and ex.worker.device_positions and core.batch_queue_size == 2
    reqs = [(f"a{i}", prompt(f"amock_{i}", 300 + 500 * (i % 5)), SamplingParams(max_tokens=20 + 11 * (i % 7), ignore_eos=True))
            for i in range(40)]
    for rid, ids, sp in reqs:
        eng.add_request(rid, {"prompt_token_ids": ids}, sp)

    def on_step(i):
        if i == 9:
            eng.abort_request(["a3"])
    done = _drain(eng, on_step=on_step)
    for rid, ids, sp in reqs:
        if rid != "a3":
            assert len(done[rid].outputs[0].token_ids) == sp.max_tokens, rid
    s = core.scheduler
    assert not s.rs.running and not s.requests
    s.rs.kv.check()


def test_stage4_cost_model_matches_offline(tmp_path):
    """The served scheduler plans with serve's OfflineEngine step costs (with the defaults, the scheduler cut
    768-token prefill chunks where the offline fork cut 4,096, and the engine's numerics depend on the boundaries).
    A real-shaped engine (the real engine's 4,416 rows per step, no "mock" in its class name) gets the engine-measured
    costs; serve's MockEngine keeps the defaults, as in OfflineEngine. adapter/tests/diff_offline_served.py diffs every
    worker input of the two paths."""
    import driver as D
    import worker as W
    from cost_model import ENGINE_MEASURED_2026_11_07, CostParams

    kw = dict(PHYS_ENGINE_KW="free_bytes_per_rank=12000000000")
    _, core = _engine(tmp_path, backend="engine", PHYS_ENGINE_CLS="mock_engine.MockEngine", **kw)
    assert core.scheduler.cost_params == CostParams()
    eng, core = _engine(tmp_path, backend="engine", PHYS_ENGINE_CLS="shaped_engine.ShapedEngine", **kw)
    assert core.scheduler.cost_params == CostParams(**ENGINE_MEASURED_2026_11_07)
    off = D.OfflineEngine(backend="engine", cfg=W.WorkerConfig(engine_cls="shaped_engine.ShapedEngine",
                                                               engine_kw=dict(free_bytes_per_rank=12_000_000_000)))
    assert off.sched.cost.p == core.scheduler.cost_params
    chunks = []
    orig = core.model_executor.worker.step

    def spy(ps, grammar=None):
        chunks.extend((se.start, se.n_rows) for sb in ps.batches for se in sb.seqs if se.kind == "prefill")
        return orig(ps, grammar)
    core.model_executor.worker.step = spy
    eng.add_request("c", {"prompt_token_ids": prompt("chunks", 9000)}, SamplingParams(max_tokens=4, ignore_eos=True))
    _drain(eng)
    assert chunks == [(0, 4096), (4096, 4096), (8192, 9000 - 8192)], chunks


def test_replay_reproduces_served_force_dump(tmp_path, monkeypatch):
    """adapter/replay.py (the decisive served-vs-offline A/B): the served path's worker inputs (PHYS_DUMP_STEPS),
    replayed through a fresh serve Worker without vLLM, give a patch_force dump identical to the served one (logits,
    idx, rows_per_step, input_mismatch), in the served dump's format and tag naming."""
    import torch

    from adapter import phys_executor as PE
    from adapter import replay as R

    steps = tmp_path / "steps"
    monkeypatch.setenv("PHYS_DUMP_STEPS", f"{steps}:100000")
    PE._DUMP["n"] = 0
    kw = dict(PHYS_ENGINE_CLS="mock_engine.MockEngine", PHYS_ENGINE_KW="free_bytes_per_rank=12000000000")
    eng, core = _engine(tmp_path, backend="engine", **kw)
    s = core.scheduler
    p = prompt("replay_forced", 2300)
    forced = [5000 + 7 * i for i in range(90)]
    with open(s.force.path, "w") as f:
        json.dump({"tag": "rp", "prompt_ids": p, "forced": forced, "n_dump": 64, "dump_dir": str(tmp_path / "served")}, f)
    eng.add_request("f", {"prompt_token_ids": p}, SamplingParams(max_tokens=64, ignore_eos=True, temperature=0))
    eng.add_request("o", {"prompt_token_ids": prompt("replay_bystander", 900)},
                    SamplingParams(max_tokens=40, ignore_eos=True))
    done = _drain(eng)
    assert list(done["f"].outputs[0].token_ids) == forced[:64]
    monkeypatch.delenv("PHYS_DUMP_STEPS")
    files = R.step_files([str(steps)])
    assert len(files) == PE._DUMP["n"] > 3
    sz = max(os.path.getsize(f) for f in files)
    assert sz < 2_000_000, sz                     # light copies: no recorded logits inside the step pickles
    w = R.make_worker("mock_engine.MockEngine", dict(free_bytes_per_rank=12_000_000_000), 320, 256)
    out = R.replay(files, w, str(tmp_path / "replayed"))
    assert out["forced"]["rp"]["n"] == 64 and out["forced"]["rp"]["input_mismatch"] == 0, out
    a = torch.load(tmp_path / "served" / "rp" / "rank0.pt")
    b = torch.load(tmp_path / "replayed" / "rp" / "rank0.pt")
    assert set(a) == set(b), (sorted(a), sorted(b))
    for k in a:
        if k in ("seconds", "saved"):
            continue
        va, vb = a[k], b[k]
        same = torch.equal(va, vb) if isinstance(va, torch.Tensor) else va == vb
        assert same, k
    ma = json.loads((tmp_path / "served" / "rp" / "rank0.json").read_text())
    mb = json.loads((tmp_path / "replayed" / "rp" / "rank0.json").read_text())
    for k in ("n", "n_dump", "prompt_len", "rows_per_step", "input_mismatch", "kind", "partial"):
        assert ma[k] == mb[k], (k, ma[k], mb[k])
