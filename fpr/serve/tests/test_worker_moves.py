"""Moves and CP chunks through the worker and MockEngine (CPU, torch for the pool tensors).

A moved sequence gets new pages, window pages and a state slot on its new rank and a new incarnation number. The
worker sends block / window tables as deltas against what the engine already holds for a (rank, slot) and sequence
incarnation; MockEngine rebuilds every table from those deltas and asserts it equals the live tables. These runs check
that every move (prefill steal, decode rebalance, memory relief) and every CP chunk reaches the engine as a complete,
current table set, while kv_content.KVContent checks the data behind the tables."""
from __future__ import annotations

import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from cost_model import CostModel  # noqa: E402
from geometry import Geometry  # noqa: E402
from kv_content import KVContent  # noqa: E402
from scheduler import RankScheduler, Request, SchedConfig  # noqa: E402

FREE = 20_000_000_000          # MockEngine's free bytes per rank


def req(rid, plen, out=20, prefix=None, prefix_tokens=0):
    pb = prefix_tokens // 64

    def hf(i, rid=rid, prefix=prefix, pb=pb):
        return (prefix, i) if prefix is not None and i < pb else (rid, i)
    return Request(rid, plen, out, 0.0, hf)


def worker(kv_bytes, **geom):
    from worker import Worker, WorkerConfig
    cfg = WorkerConfig(engine_cls="mock_engine.MockEngine", reserve_bytes=FREE - kv_bytes)
    return Worker(cfg, Geometry(**geom), "engine", max_rows=512, log=lambda *a: None)


def run(w, s, arrivals, steps, seed):
    from interfaces import build_step_batch
    m = KVContent(s)
    rng, now, done = random.Random(seed), 0.0, []
    for i in range(steps):
        for q in arrivals.get(i, ()):
            s.add_request(q)
        if not (s.running or s.waiting) and not any(k > i for k in arrivals):
            break
        p = s.schedule(now)
        acc = {x.rid: rng.randint(1, x.n) for x in p.work if x.kind == "decode"}
        if p.work or p.copies:
            w.run_step([build_step_batch(p, s.kv, w.geom, r) for r in range(4)])   # MockEngine checks the tables
        m.apply(p, acc)
        now += max(p.est_us / 1e6, 0.05)
        done += s.update(p, acc, now)
        m.check_registered(full=(i % 50 == 0))
    m.check_registered(full=True)
    s.kv.check()
    return m, done


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class TestWorkerMoves(unittest.TestCase):
    def test_moves_reach_the_engine_as_current_tables(self):
        """A mixed workload on tight pools (steals, rebalances, relief moves, prefix copies): every step's tables, as
        the engine rebuilds them from the worker's deltas, equal the live tables, and the data behind them is right."""
        w = worker(150_000_000, window_pool_bytes=60_000_000, state_pool_slots=128)
        s = RankScheduler(w.geom, CostModel(), SchedConfig(rebalance_patience=2, rebalance_every=2))
        rng = random.Random(11)
        arrivals, n = {}, 0
        for st in range(0, 1500, 3):
            if n >= 90:
                break
            ptoks = rng.choice([1_152, 6_016, 12_032])
            plen = ptoks + rng.randrange(200, rng.choice([2_000, 8_000, 20_000]))
            arrivals.setdefault(st, []).append(
                req(f"q{n}", plen, out=rng.choice([5, 40, 600, 2000]), prefix=f"sys{n % 3}", prefix_tokens=ptoks))
            n += 1
        m, done = run(w, s, arrivals, 8_000, 1)
        self.assertFalse(m.errors, m.errors[:4])
        self.assertEqual(len(done), n)
        self.assertGreater(s.stats["migrations"], 0)
        self.assertGreater(w.backend.steps, 100)

    def test_cp_chunks_reach_the_engine(self):
        """The certified scheduler (forced CP for >= 32K tokens left, tiered order) on a workload with big prompts:
        helper shadows, their replica tables and the owner's tables arrive complete at every step, and every request
        finishes."""
        w = worker(2_000_000_000)
        cfg = SchedConfig(mixed_target_tokens=12288, mixed_target_fill=True, mixed_target_wait_s=1.0, cp_prefill=True,
                          cp_min=2048, backlog_drain_wait_s=20.0, mixed_target_full_wait_s=0.0, cp_force_min=32768,
                          cp_force_decode_steps=0, cp_force_room_frac=1.0, cp_force_order="tiered",
                          cp_force_srpt_min_left=131072)
        s = RankScheduler(w.geom, CostModel(), cfg)
        rng = random.Random(7)
        arrivals = {}
        for i in range(24):
            body = rng.choice([800, 6_000, 40_000, 120_000])
            arrivals.setdefault(i * 12, []).append(
                req(f"b{i}", 1_152 + rng.randrange(body // 2, body + 1), out=rng.choice([5, 80]), prefix="sys",
                    prefix_tokens=1_152))
        m, done = run(w, s, arrivals, 6_000, 2)
        self.assertFalse(m.errors, m.errors[:4])
        self.assertEqual(len(done), 24)
        self.assertGreater(s.stats["cp_chunks"], 5)
        self.assertGreater(m.stats["reads_cp"], 0)


if __name__ == "__main__":
    unittest.main()
