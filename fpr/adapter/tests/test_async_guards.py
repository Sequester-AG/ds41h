"""ASYNC scheduling guards (vLLM 0.32 async scheduling with PhysAsyncScheduler + PhysExecutor), on the CPU with the
fake worker. Marked `vllm` (see conftest.py: needs vLLM and PHYS_MODEL).

1. Structured outputs under ASYNC (adapter/phys_scheduler.py, PHYS_ASYNC_GRAMMAR=hold, the default).
   Guards: vLLM 0.32 defers a step's sampling while any structured-output request has tokens in flight
   (pending_structured_output_tokens), which serializes the async loop for every request in the batch. The hold makes
   an in-flight structured request skip one step instead; its resume step gets host-given token ids and
   grammar-validated drafts. Outputs must equal the synchronous run token for token, every structured output must
   match its grammar, and almost no step may be deferred (control: PHYS_ASYNC_GRAMMAR=defer defers many).
2. The merge of a dispatched step runs on the dispatching thread (adapter/phys_executor.py _LazyFuture,
   PHYS_MERGE_THREAD=lazy, the default).
   Guards: with the D13 dispatch model the step's extraction (handle.result(), which can capture a CUDA graph) ran on
   a separate merge thread, where cuBLAS had no handle for the step streams (CUBLAS_STATUS_NOT_INITIALIZED during
   capture) and could race the loop thread's CUDA calls. Every merge must run on the thread that called worker.step.
  python -m pytest -q adapter/tests/test_async_guards.py
"""
from __future__ import annotations

import os
import re
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.vllm

PAT = r"(yes|no)[0-9]{30,60}"


def _key(o):
    return [(list(c.token_ids), c.finish_reason, c.stop_reason) for c in o.outputs]


def _hold_mix(h, tag: str, n_plain: int = 16, n_so: int = 2):
    """Mostly plain requests + a few structured ones (~11 % of the streams)."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm_harness import prompt

    reqs = [(f"{tag}p{i}", prompt(f"{tag}p{i}", 400 + 53 * i), SamplingParams(max_tokens=120)) for i in range(n_plain)]
    for i in range(n_so):
        ids = h.tok.encode(f"Hold {tag} {i}: answer yes or no, then digits.", add_special_tokens=False)
        reqs.append((f"{tag}so{i}", ids, SamplingParams(max_tokens=80, structured_outputs=StructuredOutputsParams(regex=PAT))))
    return reqs


@pytest.fixture(scope="module")
def sync_h():
    from vllm_harness import Harness

    return Harness()


@pytest.mark.parametrize("handles", ["0", "1"], ids=["step-thread", "dispatch-model"])
def test_grammar_hold(sync_h, handles):
    from vllm_harness import AsyncHarness

    rs = sync_h.run(_hold_mix(sync_h, f"g{handles}"))
    sync_h.idle()
    got = {}
    for mode in ("hold", "defer"):
        h = AsyncHarness(env=dict(PHYS_FAKE_HANDLES=handles, PHYS_ASYNC_GRAMMAR=mode))
        st0 = dict(h.sched.phys_stats)
        ra = h.run(_hold_mix(h, f"g{handles}"))
        assert set(ra) == set(rs)
        diff = [rid for rid in rs if _key(ra[rid]) != _key(rs[rid])]
        assert not diff, (mode, diff[:5])
        for rid, o in ra.items():
            if "so" in rid:
                text = h.tok.decode(list(o.outputs[0].token_ids), skip_special_tokens=True)
                assert re.fullmatch(PAT, text), (mode, rid, text)
            else:
                h.check_stream(o)
        st = h.sched.phys_stats
        got[mode] = {k: st.get(k, 0) - st0.get(k, 0) for k in ("grammar_held", "grammar_host_ids",
                                                               "grammar_resume_no_ids", "deferred_steps")}
        h.idle()
    hold, defer = got["hold"], got["defer"]
    # every resume after a hold has host-given ids (a request may also finish in its held step and never resume)
    assert hold["grammar_held"] > 0 and hold["grammar_resume_no_ids"] == 0, got
    assert hold["grammar_host_ids"] >= hold["grammar_held"] - 2, got
    assert defer["grammar_held"] == 0 and defer["deferred_steps"] > 10, got
    # deferrals only at each structured request's first in-flight step and at the tail
    assert 4 * hold["deferred_steps"] < defer["deferred_steps"], got


def _threads_of(h, reqs):
    """Run reqs; return (threads that called worker.step, threads that merged a dispatched step)."""
    disp, merge = set(), set()
    step0, merge0 = h.fake.step, h.ex._merge_handle

    def step(*a, **k):
        disp.add(threading.get_ident())
        return step0(*a, **k)

    def merge_handle(*a, **k):
        merge.add(threading.get_ident())
        return merge0(*a, **k)

    h.fake.step, h.ex._merge_handle = step, merge_handle
    try:
        done = h.run(reqs)
    finally:
        h.fake.step, h.ex._merge_handle = step0, merge0
    for o in done.values():
        h.check_stream(o)
    return disp, merge


def _plain(tag):
    from vllm import SamplingParams
    from vllm_harness import prompt

    return [(f"{tag}{i}", prompt(f"{tag}{i}", 300 + 97 * i), SamplingParams(max_tokens=60 + 7 * i)) for i in range(12)]


def test_merge_on_dispatch_thread():
    from vllm_harness import AsyncHarness

    h = AsyncHarness(env=dict(PHYS_FAKE_HANDLES="1"))
    assert h.ex._main_dispatch and h.ex._lazy_merge
    disp, merge = _threads_of(h, _plain("lm"))
    assert merge, "no dispatched step was merged (the D13 dispatch model did not run)"
    assert disp == merge == {threading.get_ident()}, (disp, merge)
    h.idle()


def test_merge_thread_control():
    """Control: PHYS_MERGE_THREAD=pool merges on the step thread, so the check above can see a wrong thread."""
    from vllm_harness import AsyncHarness

    h = AsyncHarness(env=dict(PHYS_FAKE_HANDLES="1", PHYS_MERGE_THREAD="pool"))
    assert h.ex._main_dispatch and not h.ex._lazy_merge
    disp, merge = _threads_of(h, _plain("pm"))
    assert disp == {threading.get_ident()} and merge and threading.get_ident() not in merge, (disp, merge)
    h.idle()
