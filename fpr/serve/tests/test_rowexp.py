"""D13 (async scheduling, device-side row expansion): the reference expander (serve/rowexp_ref.py) driven by the async
scheduler + TableBuilder.build_compact through MockEngine.step_compact. Needs torch + numpy."""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import numpy  # noqa: F401
    import torch  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

from cost_model import CostModel  # noqa: E402
from geometry import Geometry  # noqa: E402
from scheduler import Request, RankScheduler, SchedConfig  # noqa: E402


def make(n_seed=24, n_new=40, seed=0):
    from driver import seed_decoding
    from worker import Worker, WorkerConfig

    w = Worker(WorkerConfig(engine_cls="mock_engine.MockEngine", reserve_bytes=int(2e9), async_mode=True),
               Geometry(kv_bytes_per_rank=int(8e9)), "engine", max_rows=320)
    s = RankScheduler(w.geom, CostModel(), SchedConfig(async_mode=True, rebalance=False, steal=False))
    rng = random.Random(seed)
    if n_seed:
        qs = seed_decoding(s, [rng.randint(3_000, 40_000) for _ in range(n_seed)], out_tokens=60)
        for q in qs:
            q.max_new_tokens = rng.randint(20, 200)
    for i in range(n_new):
        s.add_request(Request(f"n{i}", rng.randint(100, 5_000), rng.randint(1, 150), 0.0, lambda b, i=i: (f"c{i % 5}", b)))
    return w, s


@unittest.skipUnless(HAVE, "torch + numpy")
class TestRowexp(unittest.TestCase):
    def test_rowexp_equivalence(self):
        """Every async step: the expanded decode rows == what the KV manager's live tables give at the device's true
        positions (own slot, paged slot, every window key), across admissions, prefill -> decode, finishes."""
        from driver import run_offline_async

        w, s = make()
        eng = w.backend
        checked = dict(rows=0, groups=0, steps=0, ahead=0)

        def on_step(p, batches, res):
            checked["steps"] += 1
            for r, rows in enumerate(eng.last_rows):
                ct = eng.last_cts[r]
                for k, rid in enumerate(ct.rids):
                    sk = s.kv.seqs[rid]
                    start = int(rows["seq_start"][k])
                    q = s.reqs[rid]
                    if int(ct.kind[k]) == 0:
                        self.assertGreaterEqual(start, q.num_computed)          # >= the host's settled view
                        if start > q.num_computed:
                            checked["ahead"] += 1
                    r0 = int(ct.row0[k])
                    for i in range(int(ct.n_rows[k])):
                        pos, row = start + i, r0 + i
                        self.assertEqual(int(rows["positions"][row]), pos)
                        self.assertEqual(int(rows["own_slot"][row]), sk.window[pos // 32] * 32 + pos % 32)
                        exp_pg = -1 if pos < sk.cached_tokens else sk.pages[pos // 64] * 64 + pos % 64
                        self.assertEqual(int(rows["paged_slot"][row]), exp_pg)
                        lo = max(sk.replay_start, pos - 127)
                        exp = [sk.window[x // 32] * 32 + x % 32 for x in range(lo, pos + 1)]
                        n = int(rows["sl"][row])
                        self.assertEqual(rows["si"][row, :n].tolist(), exp)
                        checked["rows"] += 1
                    checked["groups"] += 1

        orig = eng.step_compact

        def keep(batches, cts, ctx=None):
            eng.last_cts = cts
            eng.read_blocks = []                      # (rid, window blocks this step reads), per rank
            for r, rows in enumerate(eng.last_rows if False else []):
                pass
            return orig(batches, cts, ctx)

        eng.step_compact = keep

        def on_settled(p_inflight, batches):
            """D13 guarantee: settling step N releases no window page that the in-flight step N+1 reads."""
            for r, rows in enumerate(eng.last_rows):
                ct = eng.last_cts[r]
                for k, rid in enumerate(ct.rids):
                    sk = s.kv.seqs.get(rid)
                    if sk is None:                    # finished in N: its pages are deferred, not released
                        continue
                    start, r0 = int(rows["seq_start"][k]), int(ct.row0[k])
                    for i in range(int(ct.n_rows[k])):
                        pos = start + i
                        for x in range(max(sk.replay_start, pos - 127), pos + 1):
                            self.assertIn(x // 32, sk.window, (rid, pos, x))
                    checked["settled"] += 1

        checked["settled"] = 0
        out = run_offline_async(s, w, 20_000, on_step, on_settled)
        self.assertGreater(checked["settled"], 1_000)
        self.assertEqual(out["done"], 64)
        self.assertGreater(checked["rows"], 2_000)
        self.assertGreater(checked["ahead"], 1_000)                     # the async lag is exercised
        self.assertFalse(s.running or s.waiting)
        s.kv.check()
        for r in range(4):
            self.assertEqual(s.kv.usage(r)["used"], 0)            # every page freed (deferred frees flushed)
        self.assertFalse(s.kv.deferred)

    def test_sync_batches_match_tablebuilder(self):
        """Synchronous StepBatches (all starts known) through build_compact + DeviceState.expand == TableBuilder.build,
        row for row (the bit-exact reference the device row-expansion kernel is tested against)."""
        import numpy as np

        from interfaces import build_step_batch
        from rowexp_ref import DeviceState
        from worker import TableBuilder, build_compact

        s = RankScheduler(Geometry(kv_bytes_per_rank=int(8e9)), CostModel(), SchedConfig(rebalance=False, steal=False))
        rng = random.Random(7)
        for i in range(30):
            s.add_request(Request(f"q{i}", rng.randint(100, 6_000), rng.randint(5, 100), 0.0, lambda b, i=i: (f"z{i % 4}", b)))
        tb_sync, tb_c = TableBuilder(), TableBuilder()
        dev = [DeviceState(s.g.pools()[2].num_pages) for _ in range(4)]
        n = 0
        for _ in range(400):
            if not (s.running or s.waiting):
                break
            p = s.schedule()
            for r in range(4):
                sb = build_step_batch(p, s.kv, s.g, r)
                ref = tb_sync.build(sb)
                ct = build_compact(tb_c, sb, 4)
                dev[r].apply(ct)
                got = dev[r].expand(ct)
                for name in ("positions", "own_slot", "sl", "paged_slot", "seq_slot", "tail_rows", "sample_idx"):
                    self.assertTrue(np.array_equal(np.asarray(got[name]), getattr(ref, name).numpy()), name)
                self.assertTrue(np.array_equal(got["si"], ref.si.numpy()))
                acc = np.asarray([rng.randint(1, se.n_rows) if se.kind == "decode" else se.n_rows for se in sb.seqs])
                dev[r].advance(ct, acc)
                n += sb.n_rows
            s.update(p, {w.rid: 1 for w in p.work if w.kind == "decode"})
        self.assertGreater(n, 5_000)

    def test_async_counts_and_forced(self):
        """Async end to end: exact token counts per request; forced sequences (explicit rows) accept every row."""
        from driver import run_offline_async
        from scheduler import FINISHED

        w, s = make(n_seed=0, n_new=0)
        rng = random.Random(2)
        reqs = []
        for i in range(30):
            q = Request(f"q{i}", rng.randint(200, 6_000), rng.randint(1, 90), 0.0, lambda b, i=i: (f"x{i}", b))
            q.forced = i % 5 == 0
            reqs.append(q)
            s.add_request(q)
        out = run_offline_async(s, w, 20_000)
        self.assertEqual(out["done"], 30)
        self.assertTrue(all(q.phase == FINISHED and q.num_output == q.max_new_tokens for q in reqs))
        s.kv.check()
        self.assertFalse(s.kv.deferred)

    def test_sync_compact_counts_and_forced(self):
        """The synchronous loop with compact tables (WorkerConfig.compact: every start from the host, rows expanded by
        step_compact) end to end: exact token counts; forced sequences accept every row; prefix hits + replay."""
        from driver import OfflineEngine
        from worker import WorkerConfig

        eng = OfflineEngine("engine", cfg=WorkerConfig(engine_cls="mock_engine.MockEngine", reserve_bytes=int(2e9),
                                                       compact=True))
        rng = random.Random(4)
        P = [rng.randrange(1000, 9000) for _ in range(5000)]
        g = eng.generate([dict(id=f"g{i}", prompt_ids=P[: 400 + 900 * i], max_tokens=30 + i, salt="s") for i in range(5)])
        self.assertEqual([len(o["tokens"]) for o in g], [30 + i for i in range(5)])
        f = eng.force([dict(id="f", prompt_ids=P[:3000], forced=P[3000:3040], n_dump=40, salt="s")])
        self.assertEqual(len(f[0]["idx"]), 40)
        self.assertGreater(eng.worker.backend.steps, 20)
        eng.sched.kv.check()


if __name__ == "__main__":
    unittest.main()
