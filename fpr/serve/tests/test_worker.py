"""CPU tests of the worker layer (serve/worker.py, serve/driver.py, serve/weights.py plan). Need torch;
skipped where torch is missing. Run from fpr/ (PYTHONPATH=<fpr>:<repo root>):
python3 -m pytest -q serve/tests/test_worker.py"""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # hosts without torch
    HAVE_TORCH = False

from cost_model import CostModel  # noqa: E402
from geometry import (KV_RECORD_B, PAGE_BYTES, PAGE_OFFSETS, PAGE_TOKENS, RECORDS_PER_PAGE, WINDOW_LAYER_PAGE_B,  # noqa: E402
                      WINDOW_PAGE_BYTES, Geometry, SmallGeometry)
from interfaces import build_step_batch  # noqa: E402
from scheduler import DECODE, RankScheduler, Request, SchedConfig  # noqa: E402
import weights as Wt  # noqa: E402


def G(**kw):
    return Geometry(kv_bytes_per_rank=6_000_000_000, **kw)


class TestPlan(unittest.TestCase):
    def test_weight_plan(self):
        pb = Wt.plan_bytes()
        self.assertAlmostEqual(pb["experts"] / 1e9, 72.2, delta=0.2)      # 40 x 1.80 GB raw FP4 + ue8m0, TP4 slice
        self.assertGreater(pb["total"], 85e9)
        self.assertLess(pb["total"], 95e9)
        keys = [k for k, _ in Wt.entries([21], [0, 1, 2, 3], small_all=True, globals_=True)]
        self.assertIn("L21_r3_raw_experts_v1", keys)
        self.assertIn("cos_sin_c1_v1", keys)
        self.assertEqual(sum(1 for k in keys if k.startswith("mhc_L")), 40)
        self.assertEqual(sum(1 for k in keys if k.endswith("_dense_v1")), 160)
        self.assertEqual(len(keys), len(set(keys)))


@unittest.skipUnless(HAVE_TORCH, "torch")
class TestWorkerCPU(unittest.TestCase):
    def test_size_pools(self):
        from worker import WorkerConfig, size_pools

        g = size_pools(100e9, WorkerConfig(reserve_bytes=8e9), Geometry(), extra_weight_bytes=20e9)
        self.assertEqual(g.kv_bytes_per_rank, int(72e9))
        g = size_pools(100e9, WorkerConfig(reserve_bytes=8e9, max_kv_bytes=5e9), Geometry())
        self.assertEqual(g.kv_bytes_per_rank, int(5e9))
        with self.assertRaises(AssertionError):
            size_pools(9e9, WorkerConfig(reserve_bytes=8e9), Geometry())

    def test_pool_views(self):
        from worker import Pools

        p = Pools(SmallGeometry(paged_pages=10, window_pages=6, state_slots=4), "cpu")
        base = p.paged.data_ptr()
        v = p.comp_view(20)
        self.assertEqual(tuple(v.shape), (10, 64, 1, KV_RECORD_B))
        self.assertEqual(v[3, 5].data_ptr() - base, 3 * PAGE_BYTES + PAGE_OFFSETS["comp20"] + 5 * KV_RECORD_B)
        v2 = p.comp_view(8)
        self.assertEqual(tuple(v2.shape), (10, 32, 1, KV_RECORD_B))
        self.assertEqual(v2[1, 0].data_ptr() - base, PAGE_BYTES + PAGE_OFFSETS["comp8"])
        w = p.window_view(7)
        self.assertEqual(tuple(w.shape), (7, 32, 1, KV_RECORD_B))                     # + the scratch page
        self.assertEqual(w[2, 3].data_ptr() - p.window.data_ptr(), 2 * WINDOW_PAGE_BYTES + 7 * WINDOW_LAYER_PAGE_B + 3 * 584)
        self.assertEqual(p.window_base(7).data_ptr() - p.window.data_ptr(), 7 * WINDOW_LAYER_PAGE_B)

    def _batch(self):
        s = RankScheduler(G(), CostModel(), SchedConfig(rebalance=False))
        s.add_request(Request("a", 5_000, 3, 0.0, lambda b: ("conv", b)))
        rng = random.Random(0)
        while s.running or s.waiting:
            p = s.schedule()
            s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"})
        s.add_request(Request("b", 9_000, 50, 0.0, lambda b: ("conv", b)))
        for i in range(6):
            s.add_request(Request(f"d{i}", 3_000 + 997 * i, 500, 0.0, lambda b, i=i: (f"t{i}", b)))
        return s

    def test_tables_against_kv(self):
        from worker import TableBuilder, comp_slot

        s = self._batch()
        tb = TableBuilder(stub_indexer=True)
        seen_replay = seen_decode = False
        rng = random.Random(1)
        for _ in range(40):
            p = s.schedule()
            for r in range(4):
                sb = build_step_batch(p, s.kv, s.g, r)
                t = tb.build(sb)
                self.assertEqual(t.n, sb.n_rows)
                for se in sb.seqs:
                    for k in range(se.n_rows):
                        row, pos = se.row0 + k, se.start + k
                        win = [x for x in t.si[row].tolist() if x >= 0]
                        self.assertEqual(int(t.sl[row]), len(win))
                        self.assertEqual(win[-1], sb.window_slot[row])          # own position last
                        self.assertEqual(len(win), pos - max(se.replay_start, pos - 127) + 1)
                        self.assertEqual(len(set(win)), len(win))
                        self.assertEqual(win, [se.window_table[w // 32] * 32 + w % 32
                                               for w in range(max(se.replay_start, pos - 127), pos + 1)])
                        c20 = t.ti[20][row].tolist()[: int(t.tl[20][row])]
                        self.assertEqual(len(c20), min(512, pos))
                        self.assertEqual(c20, [comp_slot(se.block_table, 20, j) for j in range(max(0, pos - 512), pos)])
                        if c20:                                                  # newest compressed position = pos - 1
                            bt = se.block_table
                            self.assertEqual(c20[-1], bt[(pos - 1) // 64] * 64 + (pos - 1) % 64)
                        c2 = t.ti[2][row].tolist()[: int(t.tl[2][row])]
                        self.assertEqual(len(c2), min(512, (pos + 1) // 2))
                        if c2:
                            j = (pos + 1) // 2 - 1
                            self.assertEqual(c2[-1], se.block_table[2 * j // 64] * 32 + (2 * j % 64) // 2)
                        if se.rid == "b" and se.start == se.replay_start and se.kind == "prefill":
                            seen_replay = True
                            self.assertGreaterEqual(min(win), 0)
                    seen_decode |= se.kind == "decode"
                dev = t.to_device("cpu", scratch_slot=12345)
                self.assertEqual(tuple(dev["si"].shape), (max(t.n, 1), 1, 128))
                self.assertEqual(tuple(dev["ti20"].shape), (max(t.n, 1), 1, 512))
                if t.n:
                    self.assertTrue(bool((dev["sl"] >= 1).all()))
            s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"})
        self.assertTrue(seen_replay and seen_decode)

    def test_stub_driver_end_to_end(self):
        from driver import run_offline
        from worker import Worker, WorkerConfig

        w = Worker(WorkerConfig(), G(), "stub")
        s = RankScheduler(w.geom, CostModel(), SchedConfig())
        rng = random.Random(2)
        for i in range(32):
            s.add_request(Request(f"r{i}", rng.randint(500, 40_000), rng.randint(10, 300), 0.0,
                                  lambda b, i=i: (f"conv{i % 5}", b)))
        out = run_offline(s, w, 5_000)
        self.assertEqual(out["done"], 32)
        self.assertFalse(s.running or s.waiting)
        s.kv.check()
        self.assertGreater(w.backend.steps, 10)

    def test_offline_engine_api(self):
        from driver import OfflineEngine

        eng = OfflineEngine("stub")
        rng = random.Random(4)
        prompt = [rng.randrange(1000, 9000) for _ in range(900)]
        comp = [rng.randrange(1000, 9000) for _ in range(300)]
        out = eng.generate([dict(id="g1", prompt_ids=prompt, max_tokens=40), dict(id="g2", prompt_ids=prompt[:500], max_tokens=7)])
        self.assertEqual([o["id"] for o in out], ["g1", "g2"])
        self.assertEqual([len(o["tokens"]) for o in out], [40, 7])
        sc = eng.score([dict(id="s1", ids=prompt + comp, start=len(prompt)), dict(id="s2", ids=prompt + comp, start=len(prompt))])
        for o in sc:
            self.assertEqual(o["positions"], list(range(900, 1200)))
            self.assertTrue(all(tk is not None and len(tk) == 20 for tk in o["topk"]))
        # the second identical item may reuse the prompt's cached blocks but must recompute every scored row
        self.assertFalse(eng.sched.running or eng.sched.waiting)
        eng.sched.kv.check()

    def test_engine_contract_with_fake_engine(self):
        """The worker side of the worker <-> engine contract, against a fake Engine on the CPU."""
        import types

        from driver import OfflineEngine
        from interfaces import StepResult
        from worker import WorkerConfig

        calls = dict(bind=0, step=0, rows=0)

        class FakeEngine:
            def __init__(self, devices, model_dir, max_rows):
                self.devices = ["cpu"] * len(devices)
                self.max_rows = max_rows

            def free_bytes(self):
                return [int(20e9)] * len(self.devices)

            def bind_pools(self, pools):
                assert len(pools) == len(self.devices) and pools[0].n_paged > 0
                calls["bind"] += 1

            def step(self, batches, tables, ctx=None):
                calls["step"] += 1
                step_id = batches[0].step
                out = []
                for r, t in enumerate(tables):
                    res = StepResult(r, step_id)
                    if t.n:
                        dev = t.to_device("cpu", scratch_slot=0)
                        assert dev["si"].shape[0] == t.n and dev["seq_bt_upd_pages"].dim() == 1
                        assert int(dev["seq_bt_upd_len"].sum()) == dev["seq_bt_upd_pages"].numel()
                        assert bool((dev["seq_n_rows"].sum() == t.n))
                        calls["rows"] += t.n
                    for k, (rid, row0, nr, kind, last) in enumerate(t.seq_rows):
                        if kind == "decode":
                            res.accepted[rid] = 1 + (step_id % 4 if nr == 4 else 0)
                            res.tokens[rid] = [7] * res.accepted[rid]
                        elif last:
                            res.tokens[rid] = [5]
                    out.append(res)
                return out

        mod = types.ModuleType("engine")
        mod.Engine = FakeEngine
        sys.modules["engine"] = mod
        try:
            eng = OfflineEngine("engine", cfg=WorkerConfig(reserve_bytes=int(2e9), engine_cls="engine.Engine"), max_rows=512)
            out = eng.generate([dict(id=f"x{i}", prompt_ids=list(range(100 + 50 * i)), max_tokens=30) for i in range(6)])
        finally:
            del sys.modules["engine"]
        self.assertEqual(calls["bind"], 1)
        self.assertGreater(calls["step"], 5)
        self.assertTrue(all(len(o["tokens"]) == 30 and o["tokens"][0] == 5 for o in out))
        self.assertEqual(eng.worker.geom.kv_bytes_per_rank, int(18e9))

    def test_force_mode(self):
        """Forced decode: rows 0..n-1 predict forced[0..n-1] (prefill's last row, then 4-row verify steps, all accepted);
        dumps in ds41h's patch_force format; a second item with the same prompt hits the prefix cache."""
        import json
        import tempfile

        from driver import OfflineEngine
        from worker import STUB_VOCAB

        eng = OfflineEngine("stub")
        rng = random.Random(9)
        prompt = [rng.randrange(1000, 9000) for _ in range(700)]
        forced = [rng.randrange(1000, 9000) for _ in range(30)]
        with tempfile.TemporaryDirectory() as td:
            out = eng.force([dict(id="w", prompt_ids=prompt, forced=forced, n_dump=10),
                             dict(id="f", prompt_ids=prompt, forced=forced, n_dump=21, tag="tagf")], dump_dir=td)
            o = out[1]
            self.assertEqual(o["idx"], list(range(21)))
            self.assertEqual(o["positions"], list(range(699, 720)))
            self.assertEqual(tuple(o["logits"].shape), (21, STUB_VOCAB))
            self.assertEqual(o["rows_per_step"][0], 1)                    # the prefill step: 1 row
            self.assertTrue(all(x == 4 for x in o["rows_per_step"][1:-1]))  # then 4-row verify steps
            meta = json.load(open(os.path.join(td, "tagf", "rank0.json")))
            self.assertEqual((meta["kind"], meta["n"], meta["n_dump"], meta["saved"]), ("full", 21, 21, True))
            d = torch.load(os.path.join(td, "tagf", "rank0.pt"), weights_only=False)
            self.assertEqual(d["idx"].tolist(), list(range(21)))
            self.assertTrue(os.path.exists(os.path.join(td, "tagf", "rank3.json")))
        self.assertFalse(eng.sched.running or eng.sched.waiting)
        eng.sched.kv.check()

    def test_cache_salt(self):
        """Items with different salts never share cached blocks; the same salt does (warm -> forced)."""
        from driver import OfflineEngine

        eng = OfflineEngine("stub")
        rng = random.Random(11)
        prompt = [rng.randrange(1000, 9000) for _ in range(1000)]
        eng.generate([dict(id="warm", prompt_ids=prompt, max_tokens=1, salt="S1")])
        eng.force([dict(id="other", prompt_ids=prompt, forced=[5] * 8, n_dump=8, salt="S2")])
        out = {}
        orig = eng.sched.update

        def spy(plan, acc, now=0.0):
            for w in plan.work:
                q = eng.sched.reqs.get(w.rid)
                if q is not None:
                    out[eng.items[w.rid]["key"]] = q.cached_tokens
            return orig(plan, acc, now)

        eng.sched.update = spy
        eng.force([dict(id="same", prompt_ids=prompt, forced=[5] * 8, n_dump=8, salt="S1")])
        eng.force([dict(id="cold", prompt_ids=prompt, forced=[5] * 8, n_dump=8, salt="S3")])
        self.assertEqual(out["same"], 15 * 64)                # 1000 tokens: 15 full blocks hit (cap (P-1)//64)
        self.assertEqual(out["cold"], 0)

    def test_mock_engine_end_to_end(self):
        """MockEngine (the contract with every invariant checked) under the offline driver: generate, score, force."""
        from driver import OfflineEngine
        from worker import WorkerConfig

        eng = OfflineEngine("engine", cfg=WorkerConfig(engine_cls="mock_engine.MockEngine", reserve_bytes=int(2e9)))
        rng = random.Random(5)
        P = [rng.randrange(1000, 9000) for _ in range(1500)]
        g = eng.generate([dict(id=f"g{i}", prompt_ids=P[: 300 + 200 * i], max_tokens=25) for i in range(5)])
        self.assertTrue(all(len(o["tokens"]) == 25 for o in g))
        f = eng.force([dict(id="f", prompt_ids=P, forced=P[:40], n_dump=40)])
        self.assertEqual(tuple(f[0]["logits"].shape), (40, 64))
        sc = eng.score([dict(id="s", ids=P, start=1200)])
        self.assertEqual(len(sc[0]["topk"]), 300)
        self.assertGreater(eng.worker.backend.steps, 10)
        eng.sched.kv.check()

    def test_native_step_protocol(self):
        """Worker.step(ps, grammar): the vLLM adapter's PhysStep (duck-typed here) through MockEngine's lifecycle
        checks."""
        import types

        from worker import Worker, WorkerConfig

        w = Worker(WorkerConfig(engine_cls="mock_engine.MockEngine", reserve_bytes=int(2e9)), G(), "engine", max_rows=512)
        s = RankScheduler(w.geom, CostModel(), SchedConfig())
        for i in range(6):
            s.add_request(Request(f"q{i}", 400 + 100 * i, 30, 0.0, lambda b, i=i: (f"p{i}", b)))
        seen, done = set(), 0
        rng = random.Random(3)
        for _ in range(200):
            p = s.schedule()
            if not p.work:
                break
            batches = [build_step_batch(p, s.kv, w.geom, r) for r in range(4)]
            new = []
            for b in batches:
                for se in b.seqs:
                    if se.rid not in seen:
                        seen.add(se.rid)
                        new.append(types.SimpleNamespace(rid=se.rid, prompt_token_ids=[1] * 10, output_token_ids=[],
                                                         sampling_params=dict(temperature=0.0, seed=1)))
            rows = [[rng.randrange(1000, 9000) if k == 0 or se.kind != "decode" else -1
                     for se in b.seqs for k in range(se.n_rows)] for b in batches]
            ps = types.SimpleNamespace(step=p.step, batches=batches, row_tokens=rows, new_reqs=new, finished=set(),
                                       preempted=set(), prev_tokens={}, order=[])
            res = w.step(ps)
            acc = {}
            for x in res:
                acc.update(x.accepted)
            done += len(s.update(p, acc))
        self.assertEqual(done, 6)
        t = w.tb.build(batches[0], rows[0]) if batches[0].n_rows else None
        if t is not None:
            self.assertEqual(t.tokens.tolist(), rows[0])

    def test_prepared_event_and_flags(self):
        """The vLLM adapter's overlap protocol: Worker.step sets ps.prepared once the host tables + ctx are built,
        BEFORE the engine call (sync and async / step_compact); signals_prepared and device_positions are declared."""
        import threading
        import types

        from worker import StubBackend, Worker, WorkerConfig

        for async_mode in (False, True):
            w = Worker(WorkerConfig(engine_cls="mock_engine.MockEngine", reserve_bytes=int(2e9), async_mode=async_mode),
                       G(), "engine", max_rows=512)
            self.assertTrue(w.signals_prepared)
            self.assertTrue(w.device_positions)                     # MockEngine.step_compact expands on the "device"
            s = RankScheduler(w.geom, CostModel(), SchedConfig(async_mode=async_mode, rebalance=False, steal=False))
            for i in range(4):
                s.add_request(Request(f"q{i}", 300 + 50 * i, 5, 0.0, lambda b, i=i: (f"e{i}", b)))
            seen = set()
            calls = []
            name = "step_compact" if async_mode else "step"
            orig = getattr(w.backend, name)

            def spy(batches, tables, ctx=None, orig=orig):
                calls.append(ps.prepared.is_set())
                return orig(batches, tables, ctx)

            setattr(w.backend, name, spy)
            p = s.schedule()
            batches = [build_step_batch(p, s.kv, w.geom, r) for r in range(4)]
            new = []
            for b in batches:
                for se in b.seqs:
                    if se.rid not in seen:
                        seen.add(se.rid)
                        new.append(types.SimpleNamespace(rid=se.rid, prompt_token_ids=[1] * 10, output_token_ids=[],
                                                         sampling_params=dict(temperature=0.0)))
            ps = types.SimpleNamespace(step=p.step, batches=batches, row_tokens=None, new_reqs=new, finished=set(),
                                       preempted=set(), prev_tokens={}, order=[], prepared=threading.Event())
            w.step(ps)
            self.assertEqual(calls, [True], async_mode)
        self.assertFalse(Worker(WorkerConfig(), Geometry(kv_bytes_per_rank=int(1e9)), "stub").device_positions)
        _ = StubBackend

    def test_too_long_rejected_not_blocking(self):
        """A request no rank can hold is rejected at add_request (it would block the FCFS queue forever, as a
        398K-token item did on small pools); OfflineEngine raises; run_offline(require_done) raises when it stops
        early."""
        from driver import OfflineEngine, run_offline
        from worker import Worker, WorkerConfig

        cfg = WorkerConfig(engine_cls="mock_engine.MockEngine", reserve_bytes=int(1e9), max_kv_bytes=int(2e9))
        eng = OfflineEngine("engine", cfg=cfg)
        cap = eng.sched.max_request_tokens()
        self.assertLess(cap, 398_000)
        with self.assertRaises(ValueError):
            eng.generate([dict(id="long", prompt_ids=[7] * 398_000, max_tokens=4)])
        out = eng.generate([dict(id="ok", prompt_ids=[7] * 3_000, max_tokens=4)])   # the FCFS queue is not blocked
        self.assertEqual(len(out[0]["tokens"]), 4)
        self.assertEqual(eng.sched.stats["rejected_too_long"], 1)
        s = RankScheduler(G(), CostModel(), SchedConfig())
        self.assertFalse(s.add_request(Request("x", s.max_request_tokens(), 1, 0.0, lambda b: ("x", b))))
        self.assertIn("x", s.rejected)
        w = Worker(WorkerConfig(), G(), "stub")
        s = RankScheduler(w.geom, CostModel(), SchedConfig())
        s.add_request(Request("y", 5_000, 50, 0.0, lambda b: ("y", b)))
        with self.assertRaises(RuntimeError):
            run_offline(s, w, max_steps=3, require_done=True)

    def test_deltas_survive_uid_and_slot_reuse(self):
        """A sequence that leaves and a NEW one that gets the same uid AND state slot (FIFO uids, lowest-first slots: at
        full occupancy a preempted / re-admitted request can) must get a full block table and a cleared window ring,
        not a delta against the old sequence's pages. Sync (MockEngine rebuilds every table from the D12 deltas and
        compares with the live KV) and D13 (DeviceState.expand vs the live tables)."""
        import numpy as np

        from interfaces import build_step_batch
        from rowexp_ref import DeviceState
        from worker import TableBuilder, Worker, WorkerConfig, build_compact

        g = Geometry(kv_bytes_per_rank=int(2e9), max_seqs=1)
        w = Worker(WorkerConfig(engine_cls="mock_engine.MockEngine", reserve_bytes=int(1e9)), g, "engine", max_rows=512)
        s = RankScheduler(w.geom, CostModel(), SchedConfig(rebalance=False, steal=False))
        tbc = TableBuilder()
        dev = [DeviceState(w.geom.pools()[2].num_pages) for _ in range(4)]
        rng = random.Random(1)
        s.add_request(Request("a", 300, 400, 0.0, lambda b: ("A", b)))    # short: B's first chunk covers more blocks
        idents = []
        for step in range(120):
            if step == 30:
                s.abort("a")                              # its uid and slot go free ...
                s.add_request(Request("b", 9_500, 400, 0.0, lambda b: ("B", b)))   # ... and B takes both
            p = s.schedule()
            if not p.work:
                continue
            batches = [build_step_batch(p, s.kv, w.geom, r) for r in range(4)]
            w.run_step(batches)                           # MockEngine asserts the delta-rebuilt tables == live tables
            for r, sb in enumerate(batches):
                for se in sb.seqs:
                    idents.append((se.rid, se.uid, sb.rank, se.state_slot))
                ct = build_compact(tbc, sb, 4)
                dev[r].apply(ct)
                got = dev[r].expand(ct)
                for k, se in enumerate(sb.seqs):
                    sk = s.kv.seqs[se.rid]
                    for i in range(se.n_rows):
                        pos, row = int(got["positions"][se.row0 + i]), se.row0 + i
                        exp = -1 if pos < sk.cached_tokens else sk.pages[pos // 64] * 64 + pos % 64
                        self.assertEqual(int(got["paged_slot"][row]), exp, (se.rid, pos))
                acc = np.asarray([rng.randint(1, se.n_rows) if se.kind == "decode" else se.n_rows for se in sb.seqs])
                dev[r].advance(ct, acc)
            s.update(p, {x.rid: 1 for x in p.work if x.kind == "decode"})
        a = {(u, r, sl) for rid, u, r, sl in idents if rid == "a"}
        b = {(u, r, sl) for rid, u, r, sl in idents if rid == "b"}
        self.assertTrue(a & b, "the test needs B on A's uid and slot")

    def test_engine_caps_and_kwargs(self):
        """Worker(backend="engine"): engine kwargs pass through; rows cap = the engine's mixed-step capacity, seats =
        what fits its largest decode bucket."""
        import types

        from worker import Worker, WorkerConfig

        seen = {}

        class E2:
            wants_stub_indexer = True

            def __init__(self, devices, model_dir, max_rows, **kw):
                seen.update(kw, max_rows=max_rows)
                self.devices = ["cpu"] * 4
                self.cfg = types.SimpleNamespace(max_mixed_rows=4416)

            def free_bytes(self):
                return [int(20e9)] * 4

            def bind_pools(self, pools):
                pass

        mod = types.ModuleType("fake_eng2")
        mod.E2 = E2
        sys.modules["fake_eng2"] = mod
        try:
            w = Worker(WorkerConfig(engine_cls="fake_eng2.E2", engine_kw=dict(indexer="stub", head="stub"),
                                    reserve_bytes=int(2e9)), G(), "engine", max_rows=320)
        finally:
            del sys.modules["fake_eng2"]
        self.assertEqual((seen["indexer"], seen["head"], seen["max_rows"]), ("stub", "stub", 320))
        self.assertEqual(w.geom.rows_cap, 4416)
        self.assertEqual(w.geom.max_seqs_per_rank, 80)
        self.assertTrue(w.tb.stub_indexer)

    def test_seed_decoding(self):
        from driver import seed_decoding

        s = RankScheduler(Geometry(kv_bytes_per_rank=20_000_000_000), CostModel(), SchedConfig())
        rng = random.Random(3)
        ctxs = [int(min(194_000, max(14_000, rng.lognormvariate(10.9, 0.6)))) for _ in range(114)]
        qs = seed_decoding(s, ctxs)
        self.assertEqual(len(qs), 114)
        self.assertTrue(all(q.phase == DECODE and q.num_computed == q.prompt_len for q in qs))
        loads = [sum(s._decode_load(q) for q in s.on_rank(r)) for r in range(4)]
        self.assertLess((max(loads) - min(loads)) / (sum(loads) / 4), 0.05, loads)
        self.assertTrue(all(len(s.kv.window_table(q.rid)) <= 6 for q in qs))   # + the drafter block
        s.kv.check()
        p = s.schedule()
        self.assertEqual(sum(1 for w in p.work if w.kind == "decode"), 114)

    def test_synthetic_records_and_fill(self):
        from worker import Pools, TableBuilder, fill_synthetic, synthetic_records

        rec = synthetic_records(1000)
        self.assertEqual(tuple(rec.shape), (1000, 584))
        nope = rec[:, :448].contiguous().view(torch.float8_e4m3fn).float()
        self.assertTrue(bool(torch.isfinite(nope).all()))
        self.assertTrue(bool((rec[:, 576:583] == 127).all()))
        s = self._batch()
        p = s.schedule()
        r = next(w.rank for w in p.work)
        sb = build_step_batch(p, s.kv, s.g, r)
        t = TableBuilder(stub_indexer=True).build(sb)
        pools = Pools(SmallGeometry(paged_pages=max(s.kv.paged[r].spec.num_pages, 1), window_pages=s.kv.window[r].spec.num_pages,
                                    state_slots=4), "cpu")
        pools.paged.zero_()
        pools.window.zero_()
        n = fill_synthetic(pools, t, [0, 5], sources=(20,))
        self.assertGreater(n, 0)
        slot = int(t.ti[20][0][int(t.tl[20][0]) - 1]) if int(t.tl[20][0]) else None
        if slot is not None:
            page, off = divmod(slot, RECORDS_PER_PAGE[20])
            base = page * PAGE_BYTES + PAGE_OFFSETS["comp20"]
            self.assertEqual(int(pools.paged[base + 64 * 576 + off * 8]), 127)       # scale row of that record
        wslot = int(t.si[0][0])
        page, off = divmod(wslot, 32)
        b5 = page * WINDOW_PAGE_BYTES + 5 * WINDOW_LAYER_PAGE_B
        self.assertEqual(int(pools.window[b5 + 32 * 576 + off * 8]), 127)


if __name__ == "__main__":
    unittest.main()
