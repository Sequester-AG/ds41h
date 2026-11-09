"""A vLLM 0.32 LLMEngine (in-process EngineCore) with the fork's PhysScheduler + PhysExecutor and the fake worker, on the
CPU, for the `vllm`-marked tests. The fake worker emits tokens keyed by (prompt, position), so every finished request's
output must equal its expected stream exactly; PHYS_ADAPTER_CHECK=1 runs the scheduler's invariant checks every step.

PHYS_MODEL: a model directory with config.json and the tokenizer (weights are not read: load_format dummy).
"""
from __future__ import annotations

import os
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("PHYS_ADAPTER_WORKER", "fake")
os.environ.setdefault("PHYS_ADAPTER_CHECK", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def engine_args(**over):
    from vllm.engine.arg_utils import EngineArgs

    kw = dict(
        model=os.environ["PHYS_MODEL"], served_model_name=["dsv41-fork"], tokenizer_mode="deepseek_v41",
        tensor_parallel_size=1, max_model_len=655360, max_num_seqs=256, max_num_batched_tokens=17664,
        enable_prefix_caching=True, kv_cache_dtype="fp8_ds_mla", async_scheduling=False,
        speculative_config={"method": "dspark", "num_speculative_tokens": 3, "draft_sample_method": "probabilistic",
                            "rejection_sample_method": "block", "enable_adaptive_verification": False},
        scheduler_cls="adapter.phys_scheduler.PhysScheduler",
        distributed_executor_backend="adapter.phys_executor.PhysExecutor",
        disable_log_stats=True, load_format="dummy",
    )
    kw.update(over)
    return EngineArgs(**kw)


def prompt(tag: str, n: int) -> list:
    """n token ids, unique per tag, deterministic."""
    s = zlib.crc32(tag.encode())
    return [1000 + (s + 7919 * i) % 50000 for i in range(n)]


class Harness:
    """Synchronous scheduling (vLLM's step(), one batch in flight)."""

    def __init__(self, geometry: str = "kv:2e9", env: dict | None = None, **over):
        from vllm.v1.engine.llm_engine import LLMEngine

        os.environ["PHYS_GEOMETRY"] = geometry          # (as the other CPU suites: stays set)
        env = dict(env or {})
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            self.eng = LLMEngine.from_engine_args(engine_args(**over))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.core = self.eng.engine_core.engine_core
        self.sched = self.core.scheduler
        self.ex = self.core.model_executor
        self.fake = self.ex.worker
        self.tok = self.eng.get_tokenizer()

    def add(self, reqs: list) -> None:
        for rid, ids, sp in reqs:
            self.eng.add_request(rid, {"prompt_token_ids": list(ids)}, sp)

    def drain(self, max_steps: int = 20000) -> dict:
        done = {}
        for _ in range(max_steps):
            for o in self.eng.step():
                if o.finished:
                    done[o.request_id] = o
            if not self.eng.has_unfinished_requests():
                break
        assert not self.eng.has_unfinished_requests(), "requests left after max_steps"
        return done

    def run(self, reqs: list, **kw) -> dict:
        self.add(reqs)
        return self.drain(**kw)

    def check_stream(self, out, n_expected: int | None = None) -> list:
        toks = list(out.outputs[0].token_ids)
        if n_expected is not None:
            assert len(toks) == n_expected, (out.request_id, len(toks), n_expected)
        exp = self.fake.expected(out.prompt_token_ids, len(toks))
        assert toks == exp, (out.request_id, [i for i, (a, b) in enumerate(zip(toks, exp)) if a != b][:5])
        return toks

    def idle(self) -> None:
        """Nothing left anywhere: rank scheduler, KV pools (prefix-cache pages aside), vLLM queues."""
        s = self.sched
        assert not s.rs.running and not s.rs.waiting and not s.rs.reqs, (len(s.rs.running), len(s.rs.waiting))
        assert not s.rs.kv.seqs and not s.rs.kv.pinned
        assert not s.running and not s.requests and not getattr(s, "_started", None)
        assert all(p.reserved == 0 for p in s.rs.kv.paged)
        s.rs.kv.check()


class AsyncHarness(Harness):
    """vLLM async scheduling (step_with_batch_queue, 2 batches in flight) with PhysAsyncScheduler."""

    def __init__(self, geometry: str = "kv:2e9", env: dict | None = None, **over):
        super().__init__(geometry, dict(env or {}, PHYS_ASYNC="1"), async_scheduling=True,
                         scheduler_cls="adapter.phys_scheduler.PhysAsyncScheduler", **over)
        assert self.core.batch_queue is not None and self.core.batch_queue_size == 2
        assert self.sched.async_mode and self.ex.async_mode
