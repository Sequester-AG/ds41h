"""CPU tests of the sequence-owned KV manager (serve/kv_manager.py).
Run from fpr/ (PYTHONPATH=<fpr>:<repo root>): python3 -m unittest discover serve/tests"""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from geometry import Geometry, SmallGeometry  # noqa: E402
from kv_manager import OwnedKVManager  # noqa: E402

BT = 64


def hashes(tag, n):
    return [(tag, i) for i in range(n)]


def hf(tag):
    return lambda i: (tag, i)


def run_prompt(kv, sid, rank, tag, plen, step=0, copy_from=None, chunk=512):
    """Admit + prefill the whole prompt in chunks + commit; returns the AdmitResult."""
    res = kv.admit(sid, rank, hashes(tag, plen // BT), plen, step, copy_from)
    assert res is not None
    pos = res.replay_start
    while pos < plen:
        n = min(chunk, plen - pos)
        assert kv.allocate(sid, pos, pos + n)
        pos += n
        kv.commit(sid, pos, hf(tag))
        kv.check()
    kv.shrink_reservation(sid)
    return res


class TestKV(unittest.TestCase):
    def test_alloc_grow_free(self):
        kv = OwnedKVManager(SmallGeometry(paged_pages=40, window_pages=32, state_slots=4, reserve_watermark=0.0))
        run_prompt(kv, "a", 1, "A", 1000)
        self.assertEqual(len(kv.block_table("a")), 16)            # ceil(1000 / 64)
        self.assertEqual(kv.usage(1)["used"], 16)
        self.assertEqual(kv.usage(0)["used"], 0)                   # nothing on other ranks
        # decode growth, k + 1 = 4 rows per step
        n = 1000
        for _ in range(100):
            self.assertTrue(kv.allocate("a", n, n + 4))
            n += 3
            kv.commit("a", n, hf("A"))
        kv.check()
        self.assertEqual(len(kv.block_table("a")), -(-(n + 1) // BT))
        kv.free("a")
        kv.check()
        u = kv.usage(1)
        self.assertEqual(u["used"], 0)
        self.assertEqual(u["cached_free"], n // BT)                # full blocks stay cached
        self.assertEqual(u["window_used"], 0)

    def test_hit_cap_and_replay(self):
        kv = OwnedKVManager(SmallGeometry(paged_pages=64, reserve_watermark=0.0))
        run_prompt(kv, "a", 0, "A", 2048)                          # 32 full blocks
        kv.free("a")
        # same prefix, longer prompt: all 32 blocks hit, replay the last 128 tokens for the window
        res = kv.admit("b", 0, hashes("A", 40), 2600, 1)
        self.assertEqual((res.hit_blocks, res.cached_tokens, res.replay_start), (32, 2048, 1920))
        kv.free("b")
        # prompt exactly the cached length: the last token must be computed -> hit capped at 31 blocks
        res = kv.admit("c", 0, hashes("A", 32), 2048, 2)
        self.assertEqual((res.hit_blocks, res.cached_tokens, res.replay_start), (31, 1984, 1856))
        kv.free("c")
        # other ranks see nothing
        self.assertEqual(kv.lookup(hashes("A", 32)), [32, 0, 0, 0])
        kv.check()

    def test_bounded_replay_window_floor(self):
        g = SmallGeometry(paged_pages=64, window_pages=32, reserve_watermark=0.0)
        kv = OwnedKVManager(g)
        run_prompt(kv, "a", 0, "A", 2048)
        kv.free("a")
        res = kv.admit("b", 0, hashes("A", 40), 2600, 1)
        self.assertTrue(kv.allocate("b", res.replay_start, 2600))
        # no window pages below replay_start (1920 // 32 = 60)
        self.assertEqual(min(kv.window_table("b")), res.replay_start // g.window_page_tokens)
        kv.commit("b", 2600, hf("A"))
        # after commit only the next step's window remains: blocks >= (2600 - 127) // 32
        self.assertEqual(min(kv.window_table("b")), (2600 - 127) // 32)
        self.assertLessEqual(len(kv.window_table("b")), 5)
        kv.check()

    def test_lru_order_and_uncached_first(self):
        kv = OwnedKVManager(SmallGeometry(paged_pages=20, reserve_watermark=0.0))
        run_prompt(kv, "a", 0, "A", 10 * BT)
        kv.free("a")
        self.assertEqual(kv.usage(0)["cached_free"], 10)
        run_prompt(kv, "b", 0, "B", 10 * BT)                       # uses the 10 never-hashed pages, evicts nothing
        self.assertEqual(kv.lookup(hashes("A", 10))[0], 10)
        kv.free("b")
        # now 20 cached free pages: A (older) then B. A 4-page allocation evicts A's LAST blocks first.
        run_prompt(kv, "c", 0, "C", 4 * BT)
        self.assertEqual(kv.lookup(hashes("A", 10))[0], 6)
        self.assertEqual(kv.lookup(hashes("B", 10))[0], 10)
        kv.check()

    def test_capacity_all_or_nothing(self):
        kv = OwnedKVManager(SmallGeometry(paged_pages=10, reserve_watermark=0.0))
        self.assertIsNone(kv.admit("big", 0, hashes("X", 11), 11 * BT + 5))
        kv.check()
        self.assertEqual(kv.usage(0)["used"], 0)
        run_prompt(kv, "a", 0, "A", 9 * BT)
        before = (kv.usage(0), list(kv.block_table("a")))
        self.assertFalse(kv.allocate("a", 9 * BT, 9 * BT + 64 + 64))   # needs 2 more pages, 1 left
        self.assertEqual(before, (kv.usage(0), list(kv.block_table("a"))))
        kv.check()

    def test_reservation(self):
        kv = OwnedKVManager(SmallGeometry(paged_pages=20, reserve_watermark=0.0))
        self.assertIsNotNone(kv.admit("a", 0, hashes("A", 12), 12 * BT + 10))   # reserves 13 pages
        self.assertEqual(kv.free_pages(0), 7)
        self.assertIsNone(kv.admit("b", 0, hashes("B", 8), 8 * BT + 1))          # 9 > 7
        self.assertIsNotNone(kv.admit("b", 1, hashes("B", 8), 8 * BT + 1))       # other rank is free
        self.assertTrue(kv.allocate("a", 0, 5 * BT))
        self.assertEqual(kv.usage(0)["reserved"], 8)
        kv.check()

    def test_admit_onto_own_lru_hits(self):
        """Re-admission onto the request's own cached (LRU) pages must not over-reserve: reserved <= num_free after the
        touches (this over-reservation deadlocked prefill under the vLLM adapter)."""
        kv = OwnedKVManager(SmallGeometry(paged_pages=20, reserve_watermark=0.0))
        run_prompt(kv, "a", 0, "A", 12 * BT)            # 12 full pages, registered
        kv.free("a")                                    # all 12 -> LRU (ref 0, cached); 8 never-hashed pages free
        self.assertEqual(kv.free_pages(0), 20)
        # the same prompt + 9 more pages: hits all 12, needs 21 - 12 = 9 new pages. Free looks like 20, but 12 of those
        # are its own LRU hits -> only 8 really free -> must be rejected (the old code admitted it and over-reserved)
        self.assertIsNone(kv.admit("a2", 0, hashes("A", 21), 21 * BT, 1))
        kv.check()
        res = kv.admit("a3", 0, hashes("A", 20), 20 * BT, 1)  # needs 20 - 12 = 8 new pages: fits exactly
        self.assertIsNotNone(res)
        self.assertLessEqual(kv.paged[0].reserved, kv.paged[0].num_free())
        kv.check()
        pos = res.replay_start
        while pos < 20 * BT:                            # and the prefill completes
            n = min(512, 20 * BT - pos)
            self.assertTrue(kv.allocate("a3", pos, pos + n))
            pos += n
            kv.commit("a3", pos, hf("A"))
        kv.check()

    def test_cross_rank_prefix_copy(self):
        kv = OwnedKVManager(SmallGeometry(paged_pages=64, reserve_watermark=0.0))
        run_prompt(kv, "a", 2, "A", 30 * BT)
        kv.free("a")
        res = kv.admit("b", 0, hashes("A", 35), 35 * BT + 7, step=5, copy_from=2)
        self.assertEqual(res.copied_blocks, 30)
        self.assertEqual(res.hit_blocks, 30)
        self.assertEqual(len(res.copies), 30)
        self.assertTrue(all(c.src_rank == 2 and c.dst_rank == 0 and c.step == 5 for c in res.copies))
        # copied pages are registered on rank 0 (the next call of the conversation hits there)
        self.assertEqual(kv.lookup(hashes("A", 30)), [30, 0, 30, 0])
        # source pages pinned until the copy ran
        self.assertEqual(kv.usage(2)["used"], 30)
        kv.check()
        kv.release_copies(5)
        self.assertEqual(kv.usage(2)["used"], 0)
        self.assertEqual(kv.usage(2)["cached_free"], 30)
        kv.check()

    def test_migrate_running(self):
        kv = OwnedKVManager(SmallGeometry(paged_pages=64, window_pages=32, reserve_watermark=0.0))
        run_prompt(kv, "a", 1, "A", 20 * BT + 30)
        n = 20 * BT + 30
        for _ in range(10):
            kv.allocate("a", n, n + 4)
            n += 2
            kv.commit("a", n, hf("A"))
        old = list(kv.block_table("a"))
        ops = kv.migrate("a", 3, step=9, num_computed=n)
        self.assertIsNotNone(ops)
        self.assertEqual(kv.seqs["a"].rank, 3)
        self.assertEqual(len(kv.block_table("a")), len(old))
        self.assertEqual(sum(1 for o in ops if o.pool == "paged"), -(-n // BT))
        self.assertEqual(sum(1 for o in ops if o.pool == "state"), 1)
        self.assertEqual(kv.lookup(hashes("A", n // BT))[3], n // BT)
        kv.check()
        kv.release_copies(9)
        self.assertEqual(kv.usage(1)["used"], 0)
        self.assertEqual(kv.usage(1)["window_used"], 0)
        # the sequence keeps decoding on rank 3
        self.assertTrue(kv.allocate("a", n, n + 4))
        kv.commit("a", n + 4, hf("A"))
        kv.check()

    def test_duplicate_hash_two_writers(self):
        kv = OwnedKVManager(SmallGeometry(paged_pages=64, reserve_watermark=0.0))
        kv.admit("a", 0, hashes("S", 10), 10 * BT + 1)
        kv.admit("b", 0, hashes("S", 10), 10 * BT + 1)                # same prompt, admitted before a committed
        for s in ("a", "b"):
            kv.allocate(s, 0, 10 * BT + 1)
        kv.commit("a", 10 * BT + 1, hf("S"))
        kv.commit("b", 10 * BT + 1, hf("S"))
        kv.check()
        kv.free("a")
        kv.free("b")
        kv.check()
        u = kv.usage(0)
        self.assertEqual(u["cached_free"], 10)                       # one copy cached, b's duplicates went free
        self.assertEqual(u["free"], 64 - 10)

    def test_long_context_655k(self):
        g = Geometry(kv_bytes_per_rank=3_000_000_000)                # 3 GB -> ~13K pages (848K tokens) per rank
        kv = OwnedKVManager(g)
        L = 655_360 - 1_000
        res = kv.admit("L", 0, hashes("L", L // BT), L, 0)
        self.assertIsNotNone(res)
        pos = 0
        while pos < L:
            n = min(4096, L - pos)
            self.assertTrue(kv.allocate("L", pos, pos + n))
            pos += n
            kv.commit("L", pos, hf("L"))
        kv.shrink_reservation("L")
        for _ in range(240):                                          # decode to the max model length
            self.assertTrue(kv.allocate("L", pos, pos + 4))
            pos += 4
            kv.commit("L", pos, hf("L"))
        self.assertEqual(len(kv.block_table("L")), 655_360 // BT)
        self.assertLessEqual(len(kv.window_table("L")), 6)
        kv.check()
        self.assertGreater(kv.paged[0].spec.num_pages * g.page_tokens, 655_360)

    def test_random_stress(self):
        rng = random.Random(7)
        kv = OwnedKVManager(SmallGeometry(paged_pages=120, window_pages=40, state_slots=12, reserve_watermark=0.02))
        live = {}                                                     # sid -> [tag, computed, target]
        step = 0
        for it in range(3000):
            op = rng.random()
            if op < 0.25 and len(live) < 10:
                sid = f"s{it}"
                tag = rng.choice(["P", "Q", "R", sid])                # shared prefixes and private streams
                plen = rng.randint(1, 1500)
                r = rng.randrange(4)
                src = rng.randrange(4) if rng.random() < 0.3 else None
                res = kv.admit(sid, r, hashes(tag, plen // BT), plen, step, src)
                if res is not None:
                    live[sid] = [tag, res.replay_start, plen]
            elif op < 0.75 and live:
                sid = rng.choice(list(live))
                tag, c, tgt = live[sid]
                n = rng.randint(1, 300) if c < tgt else 4
                if kv.allocate(sid, c, c + n):
                    c += n if c < tgt else rng.randint(1, 4)
                    if c >= tgt:
                        kv.shrink_reservation(sid)
                    live[sid][1] = c
                    kv.commit(sid, c, hf(tag))
            elif op < 0.85 and live:
                sid = rng.choice(list(live))
                kv.free(sid)
                del live[sid]
            elif op < 0.92 and live:
                sid = rng.choice(list(live))
                kv.migrate(sid, rng.randrange(4), step, live[sid][1])
            else:
                kv.release_copies(step)
                step += 1
            kv.check()
        for sid in list(live):
            kv.free(sid)
        for s in range(step + 1):
            kv.release_copies(s)
        kv.check()
        for r in range(4):
            self.assertEqual(kv.usage(r)["used"], 0)


if __name__ == "__main__":
    unittest.main()


class TestNoSameStepRecopy(unittest.TestCase):
    def test_copies_never_read_a_page_filled_in_the_same_step(self):
        """A sequence admitted with a cross-rank prefix copy and moved in the same step (the failure guarded: its pages
        were copied before the prefix copy filled them, KL 1.0): the move's copies read the ORIGINAL prefix pages (rank
        0), not the pages the admission copy is still filling (rank 1); a chained move in the same step too. No copy of
        a step reads a page another copy of that step writes."""
        kv = OwnedKVManager(Geometry(kv_bytes_per_rank=6_000_000_000))
        run_prompt(kv, "warm", 0, "P", 64 * 600, step=0)          # the prefix, cached on rank 0
        kv.free("warm")
        orig = {kv.paged[0].cached[("P", i)] for i in range(600)}
        res = kv.admit("b", 1, hashes("P", 600) + hashes("B", 20), 64 * 620, step=5, copy_from=0)
        self.assertGreater(res.copied_blocks, 500)
        ops = kv.migrate("b", 2, 5, res.replay_start)                     # same step as the admission copy
        ops2 = kv.migrate("b", 3, 5, res.replay_start)                    # and once more
        all_ops = list(res.copies) + ops + ops2
        writes = {(c.pool, c.dst_rank, c.dst_page) for c in all_ops}
        reads = {(c.pool, c.src_rank, c.src_page) for c in all_ops}
        self.assertFalse(writes & reads)
        for c in ops + ops2:
            if c.pool == "paged":
                self.assertEqual(c.src_rank, 0)
                self.assertIn(c.src_page, orig)
        # a second admission copying from rank 3's (still filling) prefix pages reads rank 0's too
        res2 = kv.admit("c", 2, hashes("P", 600) + hashes("C", 5), 64 * 605, step=5, copy_from=3)
        if res2 is not None and res2.copies:
            self.assertTrue(all(c.src_rank == 0 for c in res2.copies))
        kv.release_copies(5)
        self.assertNotIn(5, kv.pending)
        kv.check()


class TestStealCopiesTheWholeHit(unittest.TestCase):
    def test_not_started_prefill_move_copies_replay_blocks(self):
        """Prefill stealing moves a not-yet-started prefill with num_computed = replay_start = hit - 128. The bounded
        replay never writes paged KV below cached_tokens, so the move must copy EVERY hit page, the last two (positions
        [replay_start, hit)) included; it used to copy only pages below num_computed and re-register the last two on the
        new rank uncopied (a hit of 22,784 tokens on rank 2 stolen to rank 1 gave KL 7.8e-3 vs 1.7e-3)."""
        for hit in (356, 2, 1):
            kv = OwnedKVManager(Geometry(kv_bytes_per_rank=6_000_000_000))
            run_prompt(kv, "warm", 2, "P", BT * hit + 17, step=0)        # the prefix, cached on rank 2
            kv.free("warm")
            res = kv.admit("fa", 2, hashes("P", hit) + hashes("F", 60), BT * (hit + 60) + 11, step=4)
            self.assertEqual((res.copied_blocks, res.hit_blocks), (0, hit))
            src_pages = list(kv.block_table("fa"))
            ops = kv.migrate("fa", 1, 5, res.replay_start)               # _steal: num_computed == replay_start
            dst_pages = kv.block_table("fa")
            copied = {(c.src_page, c.dst_page) for c in ops if c.pool == "paged" and c.src_rank == 2 and c.dst_rank == 1}
            for i in range(hit):
                self.assertIn((src_pages[i], dst_pages[i]), copied, (hit, i))
            # what the new rank serves from its prefix cache was copied
            self.assertEqual(kv.lookup(hashes("P", hit))[1], hit)
            regs = {kv.paged[1].cached[("P", i)] for i in range(hit)}
            self.assertLessEqual(regs, {d for _, d in copied})
            kv.release_copies(5)
            kv.check()


class TestPoisonOnFree(unittest.TestCase):
    """$PHYS_POISON_ON_FREE (debug): pages that leave an owner for the free list, and prefix-cache pages evicted for a
    new owner, are queued for the worker's 0xFF fill; registered prefix-cache pages are not; =pages skips state slots."""

    def _run(self, mode):
        import random as _r
        from cost_model import CostModel
        from scheduler import DECODE, RankScheduler, Request, SchedConfig

        old = os.environ.get("PHYS_POISON_ON_FREE")
        os.environ["PHYS_POISON_ON_FREE"] = mode
        try:
            s = RankScheduler(Geometry(kv_bytes_per_rank=6_000_000_000, max_tail_rows_per_rank=4608), CostModel(),
                              SchedConfig(rebalance=False, steal=False))
        finally:
            if old is None:
                del os.environ["PHYS_POISON_ON_FREE"]
            else:
                os.environ["PHYS_POISON_ON_FREE"] = old
        s.add_request(Request("a", 5_000, 3, 0.0, lambda i: ("a", i)))
        rng = _r.Random(0)
        for _ in range(500):
            if not (s.running or s.waiting):
                break
            p = s.schedule()
            s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"})
        return s, [op for r in range(4) for op in s.kv.take_poison(r)]

    def test_freed_pages_queued_cached_kept(self):
        s, ops = self._run("1")
        pools = {pool for pool, _ in ops}
        self.assertIn("window", pools)
        self.assertIn("state", pools)
        cached = {p for pl in s.kv.paged for p in pl.lru}
        self.assertTrue(cached)                                     # the prompt's full blocks stay in the prefix cache
        self.assertFalse(cached & {p for pool, p in ops if pool == "paged"})
        self.assertEqual([op for r in range(4) for op in s.kv.take_poison(r)], [])   # taken once

    def test_pages_mode_skips_state(self):
        _, ops = self._run("pages")
        self.assertTrue(ops)
        self.assertNotIn("state", {pool for pool, _ in ops})

    def test_off_by_default(self):
        _, ops = self._run("")
        self.assertEqual(ops, [])
