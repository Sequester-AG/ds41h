"""Regression tests for sequence moves between ranks (serve/kv_manager.py migrate, serve/scheduler.py _move).

A running sequence is moved by three callers: prefill stealing (a not-yet-started prefill goes to an idle rank),
decode rebalancing, and memory relief (a full rank moves its newest sequence away before preempting). Every move must
carry the sequence's whole KV: all paged blocks that hold data (the computed positions and the whole cached prefix),
the live window pages and the per-sequence state slot (compressor state, token ring). Pages registered in the new
rank's prefix cache must hold their data once the step's copies ran. And no copy of a step may read a page that
another copy of the same step fills (copies run in any order before the step's kernels).

The checks run every plan through kv_content.KVContent, a model of what each page holds on the device."""
from __future__ import annotations

import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from cost_model import CostModel  # noqa: E402
from geometry import Geometry, SmallGeometry  # noqa: E402
from kv_content import KVContent, run_checked  # noqa: E402
from kv_manager import OwnedKVManager  # noqa: E402
from scheduler import DECODE, RankScheduler, Request, SchedConfig  # noqa: E402

BT = 64


def req(rid, plen, out=20, prefix=None, prefix_tokens=0, t=0.0):
    """A request whose first `prefix_tokens` tokens are the shared stream `prefix` (block hashes (prefix, i))."""
    pb = prefix_tokens // BT

    def hf(i, rid=rid, prefix=prefix, pb=pb):
        return (prefix, i) if prefix is not None and i < pb else (rid, i)
    return Request(rid, plen, out, t, hf)


def track_moves(s):
    """Record every move with its caller (steal / rebalance / relief) and the source state before the copy."""
    log, cur = [], {"why": None}
    for name, why in (("_steal", "steal"), ("_maybe_rebalance", "rebalance"), ("_allocate_or_relieve", "relief")):
        f = getattr(s, name)

        def wrap(*a, _f=f, _why=why, **k):
            prev, cur["why"] = cur["why"], _why
            try:
                return _f(*a, **k)
            finally:
                cur["why"] = prev
        setattr(s, name, wrap)
    migrate = s.kv.migrate

    def tracked(seq_id, dst, step, num_computed):
        sk = s.kv.seqs[seq_id]
        rec = dict(rid=seq_id, why=cur["why"], step=step, src=sk.rank, dst=dst, computed=num_computed,
                   cached=sk.cached_tokens, registered=sk.num_registered, pages=list(sk.pages),
                   window=dict(sk.window), slot=sk.state_slot)
        ops = migrate(seq_id, dst, step, num_computed)
        if ops is not None and dst != rec["src"]:
            nsk = s.kv.seqs[seq_id]
            rec.update(ops=ops, new_pages=list(nsk.pages), new_window=dict(nsk.window), new_slot=nsk.state_slot)
            log.append(rec)
        return ops
    s.kv.migrate = tracked
    return log


class MoveChecks(unittest.TestCase):
    def assert_whole_kv_copied(self, log):
        """Each move copied every block that holds data, every window page and the state slot, from the old pages to
        the new ones."""
        for m in log:
            paged = {(c.src_page, c.dst_page) for c in m["ops"] if c.pool == "paged"}
            valid = max(m["computed"], m["cached"], m["registered"] * BT)
            for b in range(len(m["pages"])):
                if b * BT < valid:
                    self.assertIn((m["pages"][b], m["new_pages"][b]), paged, (m["why"], m["rid"], b, valid))
            win = {(c.src_page, c.dst_page) for c in m["ops"] if c.pool == "window"}
            for blk, pg in m["window"].items():
                self.assertIn((pg, m["new_window"][blk]), win, (m["why"], m["rid"], blk))
            st = [(c.src_page, c.dst_page) for c in m["ops"] if c.pool == "state"]
            self.assertEqual(st, [(m["slot"], m["new_slot"])], (m["why"], m["rid"]))

    def assert_clean(self, model):
        self.assertFalse(model.errors, model.errors[:5])
        self.assertGreater(model.stats["reads_prefill"], 0)


def steal_scenario(seed=4):
    """Eight decoders spread over the ranks, a 20K prefix cached on one rank, then 12 follow-ups on that prefix. With
    admission-time prefix copies disabled and every rank counted as tied on decode load, the follow-ups queue on the
    prefix's rank (their prefill is cheapest there) and prefill stealing moves not-yet-started ones, with their hit, to
    idle ranks."""
    s = RankScheduler(Geometry(kv_bytes_per_rank=6_000_000_000), CostModel(),
                      SchedConfig(rebalance=False, migrate_min_blocks=10 ** 9, placement_tol=1.0))
    rng = random.Random(seed)
    arrivals = {0: [req(f"bg{i}", 8_000 + 64 * i, out=3000) for i in range(8)],
                30: [req("warm", 20_000, out=2, prefix="P", prefix_tokens=20_000)],
                90: [req(f"f{i}", 20_000 + rng.randrange(2_000, 8_000), out=30, prefix="P", prefix_tokens=20_000)
                     for i in range(12)]}
    return s, arrivals


class TestStealMoves(MoveChecks):
    def test_steal_of_cached_prefill_copies_the_whole_hit(self):
        """Prefill stealing moves follow-ups that have not started (num_computed = replay_start = hit - 128). The
        bounded replay never writes paged KV below the hit, so the move must copy every hit page, including the two
        under the replay rows; copying only below num_computed left those pages registered on the new rank without
        their data, and the moved request and every later hit on that rank read stale KV."""
        s, arrivals = steal_scenario()
        log = track_moves(s)
        model, done = run_checked(s, 1500, random.Random(0), arrivals)
        self.assert_clean(model)
        steals = [m for m in log if m["why"] == "steal"]
        self.assertTrue(any(m["cached"] > m["computed"] for m in steals), steals)   # moved inside the bounded replay
        self.assert_whole_kv_copied(log)
        self.assertEqual(sum(1 for q in done if q.rid.startswith("f")), 12)

    def test_model_catches_the_partial_steal_copy(self):
        """The same scenario with the old migrate behaviour (paged copies only below num_computed) must be reported by
        the content model: a guard that the model would have caught the original bug."""
        s, arrivals = steal_scenario()
        migrate = s.kv.migrate

        def old_migrate(seq_id, dst, step, num_computed):
            src_pages = list(s.kv.seqs[seq_id].pages)
            ops = migrate(seq_id, dst, step, num_computed)
            if not ops:
                return ops
            return [c for c in ops if c.pool != "paged" or src_pages.index(c.src_page) * BT < num_computed]
        s.kv.migrate = old_migrate
        log = track_moves(s)
        model, _ = run_checked(s, 1500, random.Random(0), arrivals)
        self.assertTrue(any(m["why"] == "steal" and m["cached"] > m["computed"] for m in log))
        self.assertTrue(model.errors)
        self.assertTrue(any("registered page without its content" in e[0] or "stale" in e[0] for e in model.errors),
                        model.errors[:3])


class TestRebalanceMoves(MoveChecks):
    def test_rebalanced_decoders_keep_window_and_state(self):
        """Decode rebalancing moves a decoding sequence to the least loaded rank: its next step reads the paged KV of
        all earlier positions, the window of the last 127 positions and its state slot on the new rank, so all three
        must have been copied."""
        s = RankScheduler(Geometry(kv_bytes_per_rank=6_000_000_000), CostModel(),
                          SchedConfig(steal=False, rebalance_patience=2, rebalance_every=2))
        log = track_moves(s)
        rng = random.Random(5)
        # long and short outputs: as the short ones finish, the decode load per rank drifts apart
        arrivals = {i * 3: [req(f"d{i}", rng.randrange(2_000, 40_000), out=rng.choice([30, 60, 1500, 3000]))]
                    for i in range(40)}
        model, done = run_checked(s, 4000, random.Random(1), arrivals)
        self.assert_clean(model)
        moves = [m for m in log if m["why"] == "rebalance"]
        self.assertTrue(moves)
        self.assertGreater(model.stats["reads_decode"], 0)
        self.assert_whole_kv_copied(log)
        self.assertEqual(len(done), 40)


class TestReliefMoves(MoveChecks):
    def test_memory_relief_moves_carry_the_sequence(self):
        """When a rank runs out of pages, the scheduler moves the rank's newest sequence to another rank before it
        preempts anything; the moved sequence (decoding or still prefilling) continues on the new rank from its copied
        pages, window and state."""
        g = SmallGeometry(paged_pages=40, window_pages=64, state_slots=16, reserve_watermark=0.0, max_seqs_per_rank=8)
        s = RankScheduler(g, CostModel(), SchedConfig(placement="round_robin", rebalance=False))
        log = track_moves(s)
        # round robin puts r0, r4, r8, r12 on rank 0; they outgrow it, the other ranks stay nearly empty
        arrivals = {0: [req(f"r{i}", 500 if i % 4 == 0 else 64, out=2000 if i % 4 == 0 else 1) for i in range(16)]}
        model, done = run_checked(s, 3000, random.Random(2), arrivals)
        self.assert_clean(model)
        moves = [m for m in log if m["why"] == "relief"]
        self.assertTrue(moves)
        self.assertEqual(s.stats["preemptions"], 0)
        self.assert_whole_kv_copied(log)
        self.assertEqual(len(done), 16)
        self.assertTrue(all(q.num_output == q.max_new_tokens for q in done))


class TestAllMovePaths(MoveChecks):
    def _fuzz(self, seed, cp):
        """Shared system prefixes, mixed prompt / output sizes, tight pools: steals, rebalances, relief moves and
        cross-rank prefix copies in one run."""
        rng = random.Random(seed)
        s = RankScheduler(Geometry(kv_bytes_per_rank=150_000_000, window_pool_bytes=60_000_000, state_pool_slots=128),
                          CostModel(), SchedConfig(rebalance_patience=2, rebalance_every=2, cp_prefill=cp, cp_min=2048))
        log = track_moves(s)
        arrivals, n = {}, 0
        for step in range(0, 2000, 3):
            if n >= 120:
                break
            ptoks = rng.choice([1_152, 6_016, 12_032])
            plen = ptoks + rng.randrange(200, rng.choice([2_000, 8_000, 24_000]))
            arrivals.setdefault(step, []).append(
                req(f"q{n}", plen, out=rng.choice([5, 40, 600, 2000]), prefix=f"sys{n % 3}", prefix_tokens=ptoks))
            n += 1
        model, done = run_checked(s, 8_000, random.Random(seed + 100), arrivals)
        return s, log, model, done, n

    def test_fuzz_all_move_paths(self):
        """Every move caller under one random workload, every read and registered page checked: the steal fix, the
        relief path and the rebalance path all keep the KV intact, and every request finishes with all its tokens."""
        whys = set()
        for seed, cp in ((11, False), (12, False), (13, True), (14, True)):
            s, log, model, done, n = self._fuzz(seed, cp)
            self.assert_clean(model)
            self.assert_whole_kv_copied(log)
            self.assertEqual(len(done), n, (seed, cp))
            self.assertTrue(all(q.num_output == q.max_new_tokens for q in done))
            whys |= {m["why"] for m in log}
            if cp:
                self.assertGreater(s.stats["cp_chunks"], 0)
            s.kv.check()
        self.assertLessEqual({"steal", "rebalance", "relief"}, whys)


class TestMigrateUnit(unittest.TestCase):
    def test_migrate_copies_pages_window_and_state(self):
        """migrate() of a decoding sequence: one paged copy per computed block, one window copy per live window page,
        exactly one state-slot copy (old slot to new slot), the source pages pinned until release_copies(), and a new
        incarnation number so the worker resends the sequence's tables."""
        kv = OwnedKVManager(Geometry(kv_bytes_per_rank=6_000_000_000))
        h = [("A", i) for i in range(200)]
        kv.admit("a", 0, h, 64 * 200, step=0)
        assert kv.allocate("a", 0, 64 * 200 - 7, 64 * 200 - 7 + 4)
        kv.commit("a", 64 * 200 - 7, lambda i: h[i])
        sk = kv.seqs["a"]
        old_pages, old_win, old_slot, old_inc = list(sk.pages), dict(sk.window), sk.state_slot, sk.inc
        ops = kv.migrate("a", 2, 7, 64 * 200 - 7)
        sk = kv.seqs["a"]
        self.assertEqual(sk.rank, 2)
        paged = [(c.src_page, c.dst_page) for c in ops if c.pool == "paged"]
        self.assertEqual(paged, list(zip(old_pages, sk.pages)))
        win = {(c.src_page, c.dst_page) for c in ops if c.pool == "window"}
        self.assertEqual(win, {(old_win[b], sk.window[b]) for b in old_win})
        st = [c for c in ops if c.pool == "state"]
        self.assertEqual([(c.src_rank, c.src_page, c.dst_rank, c.dst_page) for c in st], [(0, old_slot, 2, sk.state_slot)])
        self.assertGreater(sk.inc, old_inc)
        self.assertTrue(all(kv.paged[0].ref[p] > 0 for p in old_pages))          # pinned until the copies ran
        kv.check()
        kv.release_copies(7)
        kv.check()
        # the full blocks keep serving prefix hits on the new rank
        self.assertEqual(kv.lookup(h)[2], 199)

    def test_same_step_admission_copy_and_moves_read_original_pages(self):
        """A request admitted with a cross-rank prefix copy and moved twice in the same step: if a move copied the
        pages the admission copy is still filling, the moved request would read uninitialised KV. Executed in any order
        (all copies read the state before the step), every final page must hold the prefix."""
        kv = OwnedKVManager(Geometry(kv_bytes_per_rank=6_000_000_000))
        h = [("P", i) for i in range(600)]
        kv.admit("warm", 0, h, 64 * 600 + 1, step=0)
        assert kv.allocate("warm", 0, 64 * 600 + 1)
        kv.commit("warm", 64 * 600 + 1, lambda i: h[i])
        content = {("paged", 0, p): ("P", i) for i, p in enumerate(kv.seqs["warm"].pages[:600])}
        kv.free("warm")
        hb = h + [("B", i) for i in range(20)]
        res = kv.admit("b", 1, hb, 64 * 620, step=5, copy_from=0)
        self.assertGreater(res.copied_blocks, 500)
        ops = list(res.copies) + kv.migrate("b", 2, 5, res.replay_start) + kv.migrate("b", 3, 5, res.replay_start)
        before = dict(content)
        for c in ops:                                       # all reads see the state before the step
            content[(c.pool, c.dst_rank, c.dst_page)] = before.get((c.pool, c.src_rank, c.src_page))
        pages = kv.seqs["b"].pages
        self.assertEqual(kv.seqs["b"].rank, 3)
        for i in range(res.hit_blocks):
            self.assertEqual(content.get(("paged", 3, pages[i])), ("P", i), i)
        kv.release_copies(5)
        kv.check()


class TestModelSelfCheck(unittest.TestCase):
    def test_model_flags_a_missing_copy(self):
        """The content model itself: dropping one paged copy of a move must be reported (a guard against a test
        model that checks nothing)."""
        g = SmallGeometry(paged_pages=40, window_pages=64, state_slots=16, reserve_watermark=0.0, max_seqs_per_rank=8)
        s = RankScheduler(g, CostModel(), SchedConfig(placement="round_robin", rebalance=False))
        for i in range(16):
            s.add_request(req(f"r{i}", 500 if i % 4 == 0 else 64, out=2000 if i % 4 == 0 else 1))
        m = KVContent(s)
        rng, now, dropped = random.Random(2), 0.0, False
        for _ in range(3000):
            if not (s.running or s.waiting):
                break
            p = s.schedule(now)
            if p.migrated and not dropped:
                i = next((k for k, c in enumerate(p.copies) if c.pool == "paged"), None)
                if i is not None:
                    del p.copies[i]
                    dropped = True
            acc = {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}
            m.apply(p, acc)
            now += 0.05
            s.update(p, acc, now)
            m.check_registered(full=True)
            if dropped and m.errors:
                break
        self.assertTrue(dropped)
        self.assertTrue(m.errors)


class TestStateSelfCheck(unittest.TestCase):
    def test_model_flags_a_dropped_state_copy(self):
        """Dropping the state-slot copy of a rebalance move (a decoding sequence) must be reported as a state slot that
        was not carried: the compressor / token-ring state lives in that slot, not in the paged or window pages."""
        s = RankScheduler(Geometry(kv_bytes_per_rank=6_000_000_000), CostModel(),
                          SchedConfig(steal=False, rebalance_patience=2, rebalance_every=2))
        rng = random.Random(5)
        for i in range(40):
            s.add_request(req(f"d{i}", rng.randrange(2_000, 40_000), out=rng.choice([30, 60, 1500, 3000])))
        m = KVContent(s)
        now, dropped = 0.0, False
        for _ in range(4000):
            if not (s.running or s.waiting):
                break
            p = s.schedule(now)
            moved = [rid for rid in p.migrated if s.reqs[rid].phase == DECODE]
            if moved and not dropped:
                p.copies = [c for c in p.copies if c.pool != "state"]
                dropped = True
            acc = {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}
            m.apply(p, acc)
            now += 0.05
            s.update(p, acc, now)
            if dropped and m.errors:
                break
        self.assertTrue(dropped)
        self.assertTrue(any(e[0] == "state slot not carried" for e in m.errors), m.errors[:3])


if __name__ == "__main__":
    unittest.main()
