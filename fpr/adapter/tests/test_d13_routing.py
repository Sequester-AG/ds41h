"""ASYNC (D13) step routing: which steps may run pipelined and why the others fall back (engine/d13.py D13._async_ok,
D13.dispatch).

Guards: a step that D13 cannot run with device-resolved positions must take the synchronous fallback, with its reason
recorded (the fallback counter keys on that reason string; the served D13 stats line reports it):
  * page copies planned for the step (copies must land before any kernel reads the pages);
  * more rows on a rank than the largest decode graph bucket;
  * prefill, forced (teacher-forced verify) or score rows;
  * a decode group that is not k + 1 rows in row order (e.g. a sequence capped at max_model_len);
  * a context beyond the decode graphs' indexer bound (decode graphs index at most GRAPH_MAX_CTX positions), including
    a device-resolved start (-1) whose host-known upper bound crosses it;
  * graphs disabled.
A plain decode-only step routes to the pipelined path (empty reason). CPU only: D13 is built without its device state
(object.__new__) and the step's StepBatches / compact tables are synthetic; the fallback is replaced by a recorder.
  python -m pytest -q adapter/tests/test_d13_routing.py
"""
import os
import sys
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from engine import d13 as D  # noqa: E402

K1 = D.K1
BUCKETS = [32, 64, 128, 256]
MAX_CTX = 262_144


def _d13(graphs=True):
    d = object.__new__(D.D13)
    d.eng = NS(cfg=NS(graphs=graphs, buckets=list(BUCKETS)))
    d.E = NS(GRAPH_MAX_CTX=MAX_CTX)
    d.lock = threading.RLock()
    d._pending_caps = []
    d.inflight = None
    d.stats = dict(async_steps=0, fallback_steps=0, first_steps=0)
    d.fallbacks = []
    d._run_pending_caps = lambda: None
    d._fallback = lambda batches, cts, ctx, why: d.fallbacks.append(why) or ("fallback", why)
    return d


def _step(seqs_per_rank=(3, 2, 0, 4), start=4000, known=4000):
    """A decode-only step: every sequence k + 1 rows, host-known starts."""
    batches, cts = [], []
    for S in seqs_per_rank:
        seqs = [NS(known=known, rid=f"r{i}") for i in range(S)]
        batches.append(NS(copies=[], n_rows=K1 * S, seqs=seqs))
        cts.append(NS(kind=np.zeros(S, np.int32), forced=np.zeros(S, np.int32), score=np.zeros(S, np.int32),
                      n_rows=np.full(S, K1, np.int32), row0=(K1 * np.arange(S)).astype(np.int32),
                      start=np.full(S, start, np.int64)))
    return batches, cts


def _why(d, b, c):
    return d._async_ok(b, c)


def test_decode_only_step_is_pipelined():
    d = _d13()
    b, c = _step()
    assert _why(d, b, c) == ""
    b, c = _step(start=-1)                       # device-resolved starts, host-known upper bound well inside
    assert _why(d, b, c) == ""


def test_graphs_off():
    b, c = _step()
    assert _why(_d13(graphs=False), b, c) == "graphs off"


def test_page_copies():
    b, c = _step()
    b[1].copies = [NS(pool="paged", src_rank=0, src_page=3, dst_rank=1, dst_page=9, step=7)]
    assert _why(_d13(), b, c) == "page copies"
    c[1].kind[0] = 1                             # a prefill row on the same rank: copies still decide
    assert _why(_d13(), b, c) == "page copies"


def test_rows_beyond_largest_bucket():
    b, c = _step(seqs_per_rank=(BUCKETS[-1] // K1 + 1, 1, 1, 1))
    assert _why(_d13(), b, c) == "rows > largest bucket"


@pytest.mark.parametrize("field", ["kind", "forced", "score"])
def test_prefill_forced_score_rows(field):
    b, c = _step()
    getattr(c[3], field)[2] = 1
    assert _why(_d13(), b, c) == "prefill / forced / score rows"


def test_decode_layout():
    b, c = _step()
    c[0].n_rows[1] = 2                           # a group shorter than k + 1 (max_model_len cap)
    assert _why(_d13(), b, c) == "decode layout"
    b, c = _step()
    c[3].row0[:] = c[3].row0[::-1]               # groups out of row order
    assert _why(_d13(), b, c) == "decode layout"


def test_context_beyond_graph_bound():
    reason = "context beyond the decode graphs' indexer bound"
    b, c = _step(start=MAX_CTX)
    assert _why(_d13(), b, c) == reason
    b, c = _step(start=MAX_CTX - K1)             # the group ends exactly at the bound: still a graph step
    assert _why(_d13(), b, c) == ""
    b, c = _step(start=MAX_CTX - K1 + 1)
    assert _why(_d13(), b, c) == reason
    # device-resolved start (-1): the host's upper bound known + k + 1 decides
    b, c = _step(start=-1, known=MAX_CTX - 2 * K1)
    assert _why(_d13(), b, c) == ""
    b, c = _step(start=-1, known=MAX_CTX - 2 * K1 + 1)
    assert _why(_d13(), b, c) == reason


def test_empty_ranks_skipped():
    b, c = _step(seqs_per_rank=(0, 0, 0, 2))
    assert _why(_d13(), b, c) == ""


def test_dispatch_routes_every_reason_to_the_fallback():
    """dispatch() hands each non-empty reason to the synchronous fallback, verbatim."""
    cases = []
    b, c = _step()
    b[2].copies = [NS(pool="paged", src_rank=0, src_page=1, dst_rank=2, dst_page=2, step=1)]
    cases.append(("page copies", b, c))
    b, c = _step()
    c[0].forced[0] = 1
    cases.append(("prefill / forced / score rows", b, c))
    b, c = _step(start=MAX_CTX + 5)
    cases.append(("context beyond the decode graphs' indexer bound", b, c))
    d = _d13()
    for why, b, c in cases:
        assert d.dispatch(b, c) == ("fallback", why)
    assert d.fallbacks == [w for w, _, _ in cases]
    d = _d13(graphs=False)
    b, c = _step()
    assert d.dispatch(b, c) == ("fallback", "graphs off")
