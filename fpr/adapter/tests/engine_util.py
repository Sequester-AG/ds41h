"""Helpers for the CPU contract tests: a vLLM 0.32 LLMEngine (in-process EngineCore) with PhysScheduler + PhysExecutor
and the fake worker, configured like the reference endpoint (THROUGHPUT line) where it matters to the engine.

Runs on the CPU: CUDA_VISIBLE_DEVICES= (no GPU is touched), PHYS_ADAPTER_WORKER=fake. PHYS_MODEL: a model directory
with config.json and the tokenizer (weights are not read)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("PHYS_ADAPTER_WORKER", "fake")
os.environ.setdefault("PHYS_ADAPTER_CHECK", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
MODEL = os.environ.get("PHYS_MODEL", "")


def engine_args(**over):
    from vllm.engine.arg_utils import EngineArgs

    kw = dict(
        model=MODEL, served_model_name=["deepseek-v41-flash-h200"], tokenizer_mode="deepseek_v41",
        tensor_parallel_size=int(os.environ.get("PHYS_TP", "1")),
        max_model_len=655360, max_num_seqs=256, max_num_batched_tokens=17664,
        enable_prefix_caching=True, kv_cache_dtype="fp8_ds_mla", async_scheduling=False,
        speculative_config={"method": "dspark", "num_speculative_tokens": 3, "draft_sample_method": "probabilistic",
                            "rejection_sample_method": "block", "enable_adaptive_verification": False},
        scheduler_cls="adapter.phys_scheduler.PhysScheduler",
        distributed_executor_backend="adapter.phys_executor.PhysExecutor",
        disable_log_stats=False, load_format="dummy",
    )
    kw.update(over)
    return EngineArgs(**kw)


def make_engine(**over):
    from vllm.v1.engine.llm_engine import LLMEngine

    return LLMEngine.from_engine_args(engine_args(**over))


def core_of(engine):
    """The in-process EngineCore (VLLM_ENABLE_V1_MULTIPROCESSING=0)."""
    return engine.engine_core.engine_core
