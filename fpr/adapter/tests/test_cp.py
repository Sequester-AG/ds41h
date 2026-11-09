"""serve's context-parallel prefill (PHYS_SCHED=cp_prefill=1, docs/CP_PREFILL.md) through the served path on the CPU.

  CUDA_VISIBLE_DEVICES= PHYS_MODEL=<model dir> python -m pytest -q adapter/tests/test_cp.py

A CP chunk is the owner's prefill entry (its sub-chunk, the chunk's last) plus one helper work item per other rank,
keyed ("cp", rid, rank) in the plan, under the prompt's rid in the StepBatch. vLLM sees one request scheduled for the
whole chunk. Checked: exact output streams, the fake worker's per-row prompt-token check on every prefill row (CP
sub-chunks included), the per-step invariants (PHYS_ADAPTER_CHECK), prefix hits, FORCE, and serve's Worker + MockEngine.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_contract import Harness, prompt  # noqa: E402
from test_stage4_stub import _drain, _engine  # noqa: E402

import pytest  # noqa: E402
from vllm import SamplingParams  # noqa: E402

pytestmark = pytest.mark.vllm      # vLLM on the CPU, fake worker: needs PHYS_MODEL (conftest.py)


def _cp_harness(**env):
    env = dict(PHYS_SCHED="cp_prefill=1", **env)
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return Harness("kv:2e9")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_cp_prefill_streams_and_accounting():
    h = _cp_harness()
    assert h.sched.rs.cfg.cp_prefill
    reqs = [(f"long{i}", prompt(f"cplong{i}", 9000 + 7000 * i), SamplingParams(max_tokens=60 + 20 * i)) for i in range(3)]
    reqs += [(f"short{i}", prompt(f"cpshort{i}", 300 + 211 * i), SamplingParams(max_tokens=40)) for i in range(6)]
    done = h.run(reqs)
    for rid, ids, sp in reqs:
        h.check_stream(done[rid], sp.max_tokens)
    st = h.sched.rs.stats
    assert st["cp_chunks"] > 0 and h.sched.phys_stats["cp_chunks"] == st["cp_chunks"], (st, h.sched.phys_stats)
    assert h.sched.phys_stats["cp_helper_rows"] > 0 and h.fake.cp_entries > 0
    for rid, ids, sp in reqs[:3]:
        assert done[rid].num_cached_tokens == 0
    h.idle()


def test_cp_prefill_prefix_hit_and_force(tmp_path):
    ff = tmp_path / "force.json"
    h = _cp_harness(DS41H_FORCE_FILE=str(ff))
    base = prompt("cpconv", 12000)
    o1 = h.run([("c1", base, SamplingParams(max_tokens=50))])["c1"]
    follow = base + list(o1.outputs[0].token_ids) + prompt("cpturn2", 14000)
    p = prompt("cpforced", 20000)
    forced = [3000 + (i * 13) % 50000 for i in range(80)]
    ff.write_text(json.dumps({"tag": "cpf", "prompt_ids": p, "forced": forced, "n_dump": 64, "dump_dir": str(tmp_path / "d")}))
    c0 = h.sched.rs.stats["cp_chunks"]
    done = h.run([("c2", follow, SamplingParams(max_tokens=30)),
                  ("f", p, SamplingParams(max_tokens=64, ignore_eos=True, temperature=0))])
    assert done["c2"].num_cached_tokens >= (len(base) + 50) // 64 * 64 - 64
    h.check_stream(done["c2"], 30)
    assert list(done["f"].outputs[0].token_ids) == forced[:64]
    meta = json.loads((tmp_path / "d" / "cpf" / "rank0.json").read_text())
    assert meta["n"] == 64 and meta["input_mismatch"] == 0 and not meta["partial"], meta
    assert h.sched.rs.stats["cp_chunks"] > c0
    h.idle()


def test_cp_prefill_stage4_mockengine(tmp_path):
    """The GPU path's code: the worker wrapper -> serve's Worker (CP copies, shadow tables) -> MockEngine."""
    # the real engine's row geometry (4,416 rows per rank per step): CP needs room for 4 sub-chunks of >= 128 rows
    # and >= 2 light helper ranks, so 2 long prompts among short ones
    eng, core = _engine(tmp_path, backend="engine", PHYS_ENGINE_CLS="real_shaped_mock.RealShapedMock",
                        PHYS_ENGINE_KW="free_bytes_per_rank=12000000000", PHYS_SCHED="cp_prefill=1")
    s = core.scheduler
    assert s.rs.cfg.cp_prefill
    reqs = [(f"m{i}", prompt(f"cpmock{i}", 12000 + 9000 * i), SamplingParams(max_tokens=20, ignore_eos=True))
            for i in range(2)]
    reqs += [(f"ms{i}", prompt(f"cpmocks{i}", 400), SamplingParams(max_tokens=15, ignore_eos=True)) for i in range(4)]
    for rid, ids, sp in reqs:
        eng.add_request(rid, {"prompt_token_ids": ids}, sp)
    done = _drain(eng)
    for rid, ids, sp in reqs:
        assert len(done[rid].outputs[0].token_ids) == sp.max_tokens, rid
    assert s.rs.stats["cp_chunks"] > 0 and s.rs.stats["cp_copy_pages"] >= 0, s.rs.stats
    assert not s.rs.running and not s.requests
    s.rs.kv.check()
