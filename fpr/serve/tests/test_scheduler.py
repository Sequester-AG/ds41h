"""CPU tests of the rank scheduler (serve/scheduler.py).
Run from fpr/ (PYTHONPATH=<fpr>:<repo root>): python3 -m unittest discover serve/tests"""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cost_model import CostModel  # noqa: E402
from geometry import Geometry, SmallGeometry  # noqa: E402
from scheduler import DECODE, PREFILL, WAITING, RankScheduler, Request, SchedConfig  # noqa: E402


def req(rid, plen, out=50, tag=None, t=0.0):
    tag = tag if tag is not None else rid
    return Request(rid, plen, out, t, lambda i, tag=tag: (tag, i))


def G(**kw):
    """Full-size structure with 6 GB of KV per rank (~39K pages = 2.5M tokens): fast invariant checks."""
    return Geometry(kv_bytes_per_rank=6_000_000_000, **kw)


def until_decoding(s, max_steps=4000):
    """Step until every running sequence decodes and nothing waits."""
    rng, now = random.Random(0), 0.0
    for i in range(max_steps):
        if not s.waiting and s.running and all(q.phase == DECODE for q in s.running):
            return i
        drive(s, 1, rng, now)
    raise AssertionError("prefill did not finish")


def drive(s, steps, rng=None, now=0.0, check_every=20):
    """Run `steps` steps with random acceptance (1 .. k + 1 tokens); returns (finished requests, time)."""
    rng = rng or random.Random(0)
    done = []
    for i in range(steps):
        p = s.schedule(now)
        acc = {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}
        now += p.est_us / 1e6
        done += s.update(p, acc, now)
        if i % check_every == 0 or i == steps - 1:
            s.kv.check()
    return done, now


class TestScheduler(unittest.TestCase):
    def test_decode_rows_and_balance_equal(self):
        s = RankScheduler(G(), CostModel(), SchedConfig(rebalance=False))
        for i in range(30):
            s.add_request(req(f"r{i}", 20_000, out=10_000))
        until_decoding(s)
        p = s.schedule()
        dec = [w for w in p.work if w.kind == "decode"]
        self.assertEqual(len(dec), 30)
        self.assertTrue(all(w.n == 4 for w in dec))
        per = [sum(1 for w in dec if w.rank == r) for r in range(4)]
        self.assertLessEqual(max(per) - min(per), 1, per)

    def test_balance_by_context(self):
        """Mixed contexts: the per-rank decode load (rows + indexer context) stays within 10 % of the mean."""
        rng = random.Random(3)
        s = RankScheduler(G(), CostModel(), SchedConfig(rebalance=False))
        for i in range(24):
            s.add_request(req(f"r{i}", int(rng.choice([15_000, 60_000, 150_000, 300_000]) * rng.uniform(0.8, 1.2)),
                              out=100_000))
        until_decoding(s)
        loads = [0.0] * 4
        for q in s.running:
            loads[q.rank] += s._decode_load(q)
        mean = sum(loads) / 4
        self.assertLess((max(loads) - min(loads)) / mean, 0.10, loads)

    def test_round_robin_is_worse(self):
        """Same mix, round-robin placement: larger idle fraction than balanced (sanity of the cost model + policy)."""
        def idle(mode):
            rng = random.Random(5)
            s = RankScheduler(G(), CostModel(), SchedConfig(placement=mode, rebalance=False))
            for i in range(24):
                s.add_request(req(f"r{i}", int(rng.choice([10_000, 200_000]) * rng.uniform(0.9, 1.1)), out=100_000))
            until_decoding(s)
            return s.schedule().idle_frac
        self.assertLess(idle("balanced"), idle("round_robin"))

    def test_prefix_affinity_and_copy(self):
        s = RankScheduler(G(), CostModel(), SchedConfig(rebalance=False))
        s.add_request(req("a", 50_000, out=10, tag="conv"))
        done, now = drive(s, 200)
        self.assertEqual([q.rid for q in done], ["a"])
        home = s.kv.lookup([("conv", i) for i in range(781)]).index(781)
        # an idle system: the follow-up call goes home (local hit, no copy)
        s.add_request(req("b", 52_000, out=10, tag="conv"))
        p = s.schedule(now)
        qb = s.reqs["b"]
        self.assertEqual(qb.rank, home)
        self.assertEqual(qb.cached_tokens, 781 * 64)
        self.assertEqual(qb.num_computed, 781 * 64 - 128)          # bounded replay of the last 128 hit tokens
        self.assertFalse(p.copies)
        drive(s, 50, now=now)
        # load the home rank heavily, then a third call: placed elsewhere WITH a copy of the cached prefix
        for i in range(12):
            s.add_request(req(f"h{i}", 300_000, out=100_000))
        drive(s, 600)
        s.add_request(req("c", 53_000, out=10, tag="conv"))
        p = s.schedule()
        qc = s.reqs["c"]
        if qc.rank != home:
            self.assertGreater(qc.copied_tokens_total, 0)
            self.assertTrue(any(o.dst_rank == qc.rank for o in p.copies))
        self.assertGreaterEqual(qc.cached_tokens, 52_000 // 64 * 64 - 64)
        s.kv.check()

    def test_prefix_fanout_copies_not_recompute(self):
        """Many calls sharing long cached prefixes at once (fan-out over a few long prompts): copies over the per-step
        budget wait a step, they never fall back to recomputing the prefix (112 calls over 8 prompts once recomputed
        275K)."""
        s = RankScheduler(G(), CostModel(), SchedConfig(rebalance=False, steal=False))
        lens = [90_000, 58_000, 23_000, 3_500]
        for i, n in enumerate(lens):
            s.add_request(req(f"w{i}", n, out=2, tag=f"p{i}"))
        drive(s, 300)
        self.assertFalse(s.running or s.waiting)
        for i, n in enumerate(lens):
            for c in range(12):
                s.add_request(req(f"c{i}_{c}", n, out=40, tag=f"p{i}"))
        done, _ = drive(s, 3000)
        self.assertEqual(len(done), 48)
        recomputed = sum(q.prompt_len - q.hit_tokens_total for q in done)
        self.assertLessEqual(recomputed, 48 * 64)               # only each call's partial last block
        self.assertGreater(s.stats["prefix_copies"], 3)
        self.assertGreater(s.stats["copy_over_budget"], 0)      # the budget bound bites: elsewhere or next step
        s.kv.check()

    def test_mixed_fixed_cost_amortized(self):
        """The real engine's eager mixed steps cost ~41 ms fixed + ~26 us per prefill row. A quota that only fits the
        per-step slack pays the 41 ms every step for a ~256-row chunk (end-to-end: ~1,200 rows per 100 ms step; the
        simulation collapsed to ~170 tok/s). With the fixed cost amortized, each mixed step carries >= ~2x the fixed
        cost in rows, fewer steps carry prefill, and the backlog drains several times faster."""
        from cost_model import ENGINE_MEASURED_1735, CostParams     # (an eager-era fixed cost: ~19 ms per mixed step)
        from driver import seed_decoding

        def run(cfg):
            s = RankScheduler(G(), CostModel(CostParams(**ENGINE_MEASURED_1735)), cfg)
            seed_decoding(s, [40_000] * 48, out_tokens=100_000)
            for i in range(8):
                s.add_request(req(f"p{i}", 60_000, out=5))
            rng, now, mixed_rows, steps = random.Random(0), 0.0, [], 0
            while any(q.phase != DECODE for q in s.running) or s.waiting:
                p = s.schedule(now)
                pf = sum(w.n for w in p.work if w.kind == "prefill")
                if pf:
                    mixed_rows.append(pf)
                now += p.est_us / 1e6
                s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}, now)
                steps += 1
                assert steps < 20_000
            s.kv.check()
            return now, steps, mixed_rows

        # (the starvation slack off in both: this isolates the amortization)
        t_new, n_new, rows_new = run(SchedConfig(rebalance=False, steal=False, starve_wait_s=float("inf")))
        t_old, n_old, rows_old = run(SchedConfig(rebalance=False, steal=False, amortize_mixed=0.0,
                                                 starve_wait_s=float("inf")))
        med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
        self.assertGreater(med(rows_new), 4 * med(rows_old))                # ~1,800 vs ~256 rows per mixed step
        self.assertGreater(min(rows_new[:-8] or rows_new), 1_000)
        self.assertLess(len(rows_new), 0.5 * n_new)                          # fewer steps carry prefill
        self.assertLess(t_new, 0.5 * t_old)                                  # the 480K-token backlog drains >= 2x faster
        _ = rows_old

    def test_mixed_target_tokens(self):
        """Token-target mixed steps: while sequences decode, no prefill runs until the pending prefill reaches the target
        (or a prefill waited mixed_target_wait_s); a step that fires carries at least the target (when available)."""
        from cost_model import ENGINE_MEASURED_2026_11_07, CostParams
        from driver import seed_decoding

        s = RankScheduler(G(max_prefill_tokens_per_rank=8192), CostModel(CostParams(**ENGINE_MEASURED_2026_11_07)),
                          SchedConfig(rebalance=False, steal=False, mixed_target_tokens=8192, mixed_target_wait_s=1.0))
        seed_decoding(s, [40_000] * 48, out_tokens=100_000)
        s.add_request(req("a", 2_000, out=5))
        rng, now, first = random.Random(0), 0.0, None
        while first is None:
            p = s.schedule(now)
            if any(w.kind == "prefill" for w in p.work):
                first = now
            now += p.est_us / 1e6
            s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}, now)
            self.assertLess(now, 5.0)
        self.assertGreaterEqual(first, 1.0)                    # below the target: only the starvation guard fires it
        until_decoding(s)
        for i in range(6):                                     # 12K pending: the next step fires with >= 8,192 rows
            s.add_request(req(f"b{i}", 2_000, out=5))
        p = s.schedule(now)
        pf = sum(w.n for w in p.work if w.kind == "prefill")
        self.assertGreaterEqual(pf, 8192)
        self.assertLess(pf, 8192 + 2_000)
        s.kv.check()

    def test_demand_driven_fill_and_decode_floor(self):
        """mixed_target_fill: a mixed step fires once the pending prefill reaches the threshold and fills to the row
        caps (not to the threshold); under a standing backlog at least mixed_min_decode_steps decode-only steps run
        between two mixed steps."""
        from cost_model import ENGINE_MEASURED_2026_11_07, CostParams
        from driver import seed_decoding

        s = RankScheduler(G(), CostModel(CostParams(**ENGINE_MEASURED_2026_11_07)),
                          SchedConfig(rebalance=False, steal=False, mixed_target_tokens=4096, mixed_target_fill=True,
                                      mixed_min_decode_steps=2))
        seed_decoding(s, [40_000] * 48, out_tokens=100_000)
        for i in range(8):
            s.add_request(req(f"L{i}", 60_000, out=5))           # a standing backlog of 480K tokens
        rng, now, kinds = random.Random(0), 0.0, []
        for _ in range(60):
            p = s.schedule(now)
            pf = sum(w.n for w in p.work if w.kind == "prefill")
            kinds.append(pf)
            now += p.est_us / 1e6
            s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}, now)
        mixed = [i for i, x in enumerate(kinds) if x]
        self.assertGreater(max(kinds), 4 * 4096 - 1)               # filled to the caps, not to the 4K threshold
        self.assertTrue(all(b - a >= 3 for a, b in zip(mixed, mixed[1:])), mixed)   # >= 2 decode-only steps between
        s.kv.check()

    def test_no_copy_reads_a_page_filled_in_the_same_step(self):
        """No CopyOp of a plan reads a page another CopyOp of the same plan writes (the failure guarded: a request
        admitted with a cross-rank prefix copy was stolen / moved in the same step, and its migration copied the pages
        before the prefix copy filled them). Prefix fan-out with steals, rebalance and pressure moves on."""
        rng = random.Random(4)
        s = RankScheduler(Geometry(kv_bytes_per_rank=2_000_000_000), CostModel(), SchedConfig())
        prefixes = [[(f"sys{j}", i) for i in range(20_000)] for j in range(3)]
        n = 0
        for step in range(1500):
            if step % 6 == 0 and n < 120:
                pre = prefixes[n % 3]
                body = [(f"r{n}", i) for i in range(rng.randrange(500, 6000))]
                toks = pre + body
                s.add_request(Request(f"r{n}", len(toks), rng.randrange(5, 40), 0.0, lambda i, t=toks: t[i]))
                n += 1
            p = s.schedule(step * 0.05)
            dst = {(c.pool, c.dst_rank, c.dst_page) for c in p.copies}
            src = {(c.pool, c.src_rank, c.src_page) for c in p.copies}
            self.assertFalse(dst & src, (step, sorted(dst & src)[:4]))
            s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}, step * 0.05)
            if not (s.running or s.waiting) and n >= 120:
                break
        self.assertGreater(s.kv.stats["copies"], 0)
        s.kv.check()

    def test_backlog_drain_relaxes_floor(self):
        """backlog_drain_wait_s: while a long prefill has waited that long for its first chunk, mixed steps may run back
        to back (the decode floor is off); without it, >= mixed_min_decode_steps decode-only steps separate them."""
        from cost_model import ENGINE_MEASURED_2026_11_07, CostParams
        from driver import seed_decoding

        def gaps(drain):
            s = RankScheduler(G(), CostModel(CostParams(**ENGINE_MEASURED_2026_11_07)),
                              SchedConfig(rebalance=False, steal=False, mixed_target_tokens=4096, mixed_target_fill=True,
                                          mixed_min_decode_steps=2, backlog_drain_wait_s=drain))
            seed_decoding(s, [40_000] * 48, out_tokens=100_000)
            for i in range(16):                                   # a standing backlog: 16 x 60K on 4 ranks
                s.add_request(req(f"L{i}", 60_000, out=5))
            rng, now, mixed = random.Random(0), 0.0, []
            for i in range(80):
                p = s.schedule(now)
                if any(w.kind == "prefill" for w in p.work):
                    mixed.append(i)
                now += p.est_us / 1e6 + 0.5                      # time passes: the waiting prefills age past 1 s
                s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}, now)
            return [b - a for a, b in zip(mixed, mixed[1:])]

        self.assertTrue(all(g >= 3 for g in gaps(float("inf"))))
        self.assertIn(1, gaps(1.0))                                # back-to-back mixed steps while draining

    def test_full_chunk_fires(self):
        """mixed_target_full_wait_s: a lone long prompt holds one full chunk on its rank (4,096 < the 12,288 target), so
        without the option each chunk waits for the 1 s guard (~4K tok/s served); with it, a chunk every
        mixed_min_decode_steps + 1 steps, the decode floor kept."""
        from cost_model import ENGINE_MEASURED_2026_11_07, CostParams
        from driver import seed_decoding

        def run(fw):
            s = RankScheduler(G(), CostModel(CostParams(**ENGINE_MEASURED_2026_11_07)),
                              SchedConfig(rebalance=False, steal=False, mixed_target_tokens=12288, mixed_target_fill=True,
                                          mixed_target_wait_s=1.0, mixed_min_decode_steps=2,
                                          mixed_target_full_wait_s=fw))
            seed_decoding(s, [40_000] * 48, out_tokens=100_000)
            s.add_request(req("L", 40_960, out=5))
            rng, now, mixed, i = random.Random(0), 0.0, [], 0
            while s.reqs["L"].first_token_time is None:
                p = s.schedule(now)
                if any(w.kind == "prefill" for w in p.work):
                    mixed.append(i)
                now += p.est_us / 1e6
                s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}, now)
                i += 1
                self.assertLess(now, 60.0)
            s.kv.check()
            return now, [b - a for a, b in zip(mixed, mixed[1:])]

        t_off, _ = run(float("inf"))
        t_on, gaps = run(0.0)
        self.assertGreaterEqual(t_off, 9.0)                    # 10 chunks, 9 of them on the 1 s guard
        self.assertLess(t_on, 0.3 * t_off)
        self.assertTrue(all(g >= 3 for g in gaps), gaps)       # the decode floor still holds

    def test_load_adaptive_controller(self):
        """mixed_adapt_lo / hi: with few decoders (lambda 0) a lone 2K prompt is prefilled in the very next
        step (no 12K threshold, no 1 s wait); with many (lambda 1) the controller is the configured one, plan for plan."""
        from cost_model import ENGINE_MEASURED_2026_11_07, CostParams
        from driver import seed_decoding

        def first_chunk_after(n_dec, adapt):
            kw = dict(rebalance=False, steal=False, mixed_target_tokens=12288, mixed_target_fill=True,
                      mixed_target_wait_s=1.0, mixed_min_decode_steps=2)
            if adapt:
                kw.update(mixed_adapt_lo=40, mixed_adapt_hi=60)
            s = RankScheduler(G(), CostModel(CostParams(**ENGINE_MEASURED_2026_11_07)), SchedConfig(**kw))
            seed_decoding(s, [20_000] * n_dec, out_tokens=100_000)
            rng, now = random.Random(0), 0.0
            for _ in range(5):                                  # settle the decoders
                p = s.schedule(now)
                now += p.est_us / 1e6
                s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}, now)
            s.add_request(req("a", 2_000, out=5))
            t0, plans = now, []
            while s.reqs["a"].first_token_time is None:
                p = s.schedule(now)
                plans.append(sorted((w.rid, w.kind, w.start, w.n) for w in p.work))
                now += p.est_us / 1e6
                s.update(p, {w.rid: 1 for w in p.work if w.kind == "decode"}, now)
                self.assertLess(now - t0, 10.0)
            return s.reqs["a"].first_token_time - t0, plans

        t_low, _ = first_chunk_after(32, adapt=True)            # 32 decoders <= lo: fire at once
        t_low_off, _ = first_chunk_after(32, adapt=False)
        self.assertLess(t_low, 0.3)
        self.assertGreater(t_low_off, 1.0)                      # the configured controller waits for the 1 s guard
        t_hi, plans_hi = first_chunk_after(64, adapt=True)      # 64 >= hi: the configured controller, plan for plan
        t_hi_off, plans_hi_off = first_chunk_after(64, adapt=False)
        self.assertEqual(plans_hi, plans_hi_off)
        self.assertAlmostEqual(t_hi, t_hi_off)

    def test_long_order_fcfs(self):
        """long_order="fcfs": among long prefills the oldest goes first, even when it is larger (srpt: the smaller)."""
        for order, first in (("fcfs", "old"), ("srpt", "new")):
            s = RankScheduler(G(n_ranks=1, max_seqs=32, max_seqs_per_rank=32), CostModel(),
                              SchedConfig(rebalance=False, steal=False, long_order=order, prefill_aging_tok_per_s=0.0))
            s.add_request(req("old", 60_000, t=0.0))
            s.schedule(0.0)                                       # admitted at t = 0
            q = s.reqs["old"]
            s.add_request(req("new", 10_000, t=5.0))
            s.schedule(5.0)
            pend = s._prefill_sorted([x for x in s.prefilling], 0, 6.0)
            self.assertEqual(pend[0].rid, first, (order, [x.rid for x in pend]))
            _ = q

    def test_large_prefill_share(self):
        """Short-first with the anti-starvation share: when short prefills fill a rank's step, the oldest longer prefill
        still gets large_min_rows of it (served load without it: 4-16K TTFT p95 86 s); 0 = pure short-first."""
        from cost_model import ENGINE_MEASURED_2026_11_07, CostParams

        # row-bound (banked, the served default): the share is large_min_rows of the rank's 4,096
        for share, expect in ((1024, 1024), (0, 0)):            # (default off: large_min_rows = 0)
            s = RankScheduler(G(n_ranks=1, max_seqs=32, max_seqs_per_rank=32),
                              CostModel(CostParams(**ENGINE_MEASURED_2026_11_07)),
                              SchedConfig(rebalance=False, steal=False, large_min_rows=share))
            s.add_request(req("big", 20_000))
            s.add_request(req("big2", 9_000))
            for i in range(4):
                s.add_request(req(f"s{i}", 1_500))
            p = s.schedule(0.0)
            rows = {w.rid: w.n for w in p.work if w.kind == "prefill"}
            self.assertEqual(rows.get("big", 0), expect, rows)        # the oldest long one, not the shorter big2
            self.assertNotIn("big2", rows)
            self.assertEqual(sum(rows.values()), s.g.max_prefill_tokens_per_rank)
            s.kv.check()
        # time-bound (slack quota): the share takes at most large_share_frac of the step's prefill time
        s = RankScheduler(G(n_ranks=1, max_seqs=32, max_seqs_per_rank=32), CostModel(),
                          SchedConfig(rebalance=False, steal=False, large_min_rows=1024))
        s.add_request(req("big", 20_000))
        for i in range(4):
            s.add_request(req(f"s{i}", 1_500))
        p = s.schedule(0.0)
        rows = {w.rid: w.n for w in p.work if w.kind == "prefill"}
        self.assertGreater(rows.get("big", 0), 0)
        self.assertGreater(sum(v for k, v in rows.items() if k != "big"), rows["big"])

    def test_starving_long_prefill_raises_slack(self):
        """Anti-starvation by capacity: once a longer prefill has waited starve_wait_s, the step may carry up to
        starve_slack_frac x the decode step of prefill (instead of max_slack_frac)."""
        s = RankScheduler(G(), CostModel(), SchedConfig(rebalance=False, steal=False, starve_wait_s=4.0))
        b = 50_000.0
        self.assertAlmostEqual(s._slack(b, 1e9), 0.8 * b)
        self.assertAlmostEqual(s._slack(b, 1e9, 1.6), 1.6 * b)
        from driver import seed_decoding
        seed_decoding(s, [40_000] * 120, out_tokens=100_000)     # a decode step above the 60 ms / 1.8 floor
        for i in range(4):
            s.add_request(req(f"L{i}", 30_000, out=5, t=0.0))
        p0 = s.schedule(0.0)                                      # admitted at t = 0: not starving yet
        e0 = p0.est_us
        s.update(p0, {w.rid: 1 for w in p0.work if w.kind == "decode"}, 0.05)
        p1 = s.schedule(10.0)                                     # waited 10 s: the larger slack cap (1.2 vs 0.8)
        self.assertGreater(p1.est_us, 1.1 * e0)

    def test_tail_rows_capped(self):
        """Many prefix-hit calls at once (each a 128-row bounded replay = 128 tail rows) must not put more tail rows on
        a rank than the engine's tail bucket (HSD: 1,536): an end-to-end probe, 112 copies of 8 cached prompts, died
        with "3579 exceeds the largest bucket 1536"."""
        from cost_model import ENGINE_MEASURED_2026_11_07, CostParams

        for cp in (CostParams(), CostParams(**ENGINE_MEASURED_2026_11_07)):
            s = RankScheduler(G(), CostModel(cp), SchedConfig(rebalance=False, steal=False))
            for i in range(4):
                s.add_request(req(f"w{i}", 30_000, out=2, tag=f"P{i}"))
            drive(s, 400)
            for i in range(4):
                for c in range(28):
                    s.add_request(req(f"c{i}_{c}", 30_000, out=50, tag=f"P{i}"))
            peak = 0
            for _ in range(3000):
                if not (s.running or s.waiting):
                    break
                p = s.schedule()
                for r in range(4):
                    tail = sum(w.n if w.kind == "decode" else min(w.n, 128) for w in p.work if w.rank == r)
                    peak = max(peak, tail)
                s.update(p, {w.rid: 1 for w in p.work if w.kind == "decode"})
            self.assertLessEqual(peak, s.g.max_tail_rows_per_rank)
            self.assertFalse(s.running or s.waiting)
            s.kv.check()

    def test_prefill_quota_bounds_step(self):
        cfg = SchedConfig(rebalance=False)
        s = RankScheduler(G(), CostModel(), cfg)
        for i in range(60):
            s.add_request(req(f"d{i}", 8_000, out=100_000))
        drive(s, 200)
        base = s.schedule()
        s.update(base, {w.rid: 1 for w in base.work if w.kind == "decode"})
        for i in range(8):
            s.add_request(req(f"big{i}", 200_000, out=10))
        mx = 0.0
        progress = []
        for _ in range(30):
            p = s.schedule()
            dec_only = [w for w in p.work if w.kind == "decode"]
            pf = sum(w.n for w in p.work if w.kind == "prefill")
            progress.append(pf)
            mx = max(mx, p.est_us)
            s.update(p, {w.rid: 1 for w in dec_only})
        self.assertTrue(all(x > 0 for x in progress))               # prefill always progresses
        cap = max(base.est_us * (1 + cfg.max_slack_frac), cfg.prefill_floor_us)
        self.assertLess(mx, cap + 8_000)                            # + one quantum of overshoot
        # every rank with backlog got prefill rows: the water-fill spreads by rank
        p = s.schedule()
        ranks_pf = {w.rank for w in p.work if w.kind == "prefill"}
        ranks_backlog = {q.rank for q in s.running if q.phase == PREFILL}
        self.assertEqual(ranks_pf, ranks_backlog)

    def test_long_prompt_chunked_to_655k(self):
        s = RankScheduler(G(), CostModel())
        L = 655_360 - 200
        s.add_request(req("L", L, out=200))
        done, _ = drive(s, 3000)
        self.assertEqual([q.rid for q in done], ["L"])
        self.assertEqual(done[0].num_output, 200)
        self.assertEqual(s.kv.usage(done[0].rank)["used"], 0)
        self.assertEqual(s.kv.usage(done[0].rank)["cached_free"], (L + 199) // 64)

    def test_uids_stable_and_unique(self):
        g = SmallGeometry(paged_pages=40, window_pages=64, state_slots=16, reserve_watermark=0.0, max_seqs_per_rank=8)
        s = RankScheduler(g, CostModel(), SchedConfig(placement="round_robin", rebalance=False))
        for i in range(16):
            s.add_request(req(f"r{i}", 500 if i % 4 == 0 else 64, out=2000 if i % 4 == 0 else 1))
        seen = {}
        rng = random.Random(0)
        for _ in range(600):
            p = s.schedule()
            uids = [q.uid for q in s.running]
            self.assertEqual(len(uids), len(set(uids)))
            self.assertTrue(all(0 <= u < g.max_seqs for u in uids))
            for w in p.work:
                q = s.reqs[w.rid]
                self.assertEqual(w.uid, q.uid)
                if q.n_preempt == 0:
                    self.assertEqual(seen.setdefault(w.rid, q.uid), q.uid)    # stable across moves
            s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"})
        self.assertGreater(s.stats["migrations"], 0)
        self.assertEqual(len(s.free_uids) + len(s.running), g.max_seqs)

    def test_rows_cap_per_rank(self):
        g = G(max_rows_per_rank=300)
        s = RankScheduler(g, CostModel(), SchedConfig())
        self.assertEqual(s.seats_per_rank(), 75)
        for i in range(400):
            s.add_request(req(f"r{i}", 3_000, out=200))
        rng = random.Random(2)
        for _ in range(200):
            p = s.schedule()
            self.assertTrue(all(p.rows(r) <= 300 for r in range(4)), [p.rows(r) for r in range(4)])
            s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"})

    def test_draft_window_and_low_state_slots(self):
        """Decode steps keep window pages to start + 2k + 1 (the drafter's block); state slots stay compact under churn."""
        s = RankScheduler(G(), CostModel(), SchedConfig(rebalance=False, steal=False))
        rng = random.Random(4)
        for i in range(300):
            s.add_request(req(f"r{i}", rng.randint(200, 3000), out=rng.randint(5, 120)))
        k = s.g.spec_k
        hi_slot, peak = 0, 0
        for _ in range(3000):
            if not (s.running or s.waiting):
                break
            p = s.schedule()
            for w in p.work:
                sk = s.kv.seqs[w.rid]
                if w.kind == "decode":
                    need_hi = (w.start + 2 * k + 1 - 1) // 32
                    self.assertIn(need_hi, sk.window, (w.rid, w.start))
                hi_slot = max(hi_slot, sk.state_slot)
            peak = max(peak, max(len(s.on_rank(r)) for r in range(4)))
            self.assertLess(hi_slot, peak + 2)                   # lowest-first: slots <= peak live per rank (+ pins)
            s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"})
        self.assertLess(hi_slot, s.seats_per_rank())          # never past the seats of a rank
        s.kv.check()

    def test_no_prefill_deadlock_tight_pools(self):
        """Repro from the vLLM adapter: tight pools (160 paged / 96 window pages / 12 state slots per rank), 14 requests
        of 2.5-5.4K prompts x 1,500 tokens. The drafter's extra (k) must only extend the WINDOW allocation; on the paged
        pool it crossed a page boundary past the admission reservation and stalled the last prefill chunk forever."""
        for async_mode, draft in ((False, 3), (True, 3), (False, 0), (True, 0)):
            g = SmallGeometry(paged_pages=160, window_pages=96, state_slots=12, reserve_watermark=0.0,
                              draft_window_tokens=draft)
            s = RankScheduler(g, CostModel(), SchedConfig(async_mode=async_mode, rebalance=False, steal=False))
            rng = random.Random(11)
            for i in range(14):
                s.add_request(req(f"a{i}", rng.randint(2_500, 5_400), out=1_500))
            done, last_progress, prev = 0, 0, None
            pending = None
            for it in range(40_000):
                if not (s.running or s.waiting):
                    break
                p = s.schedule()
                acc = {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}
                if async_mode:
                    if pending is not None:
                        done += len(s.update(*pending))
                    pending = (p, acc)
                else:
                    done += len(s.update(p, acc))
                state = done                              # progress = requests finishing (num_computed can churn)
                if state != prev:
                    last_progress, prev = it, state
                self.assertLess(it - last_progress, 3_000, f"no request finished for 3,000 steps (async={async_mode})")
            if pending is not None:
                done += len(s.update(*pending))
            self.assertEqual(done, 14, f"async={async_mode}")
            s.kv.check()

    def test_caps(self):
        g = G(max_seqs=10, max_seqs_per_rank=3)
        s = RankScheduler(g, CostModel())
        for i in range(20):
            s.add_request(req(f"r{i}", 1000, out=1000))
        drive(s, 5)
        self.assertEqual(len(s.running), 10)
        self.assertTrue(all(len(s.on_rank(r)) <= 3 for r in range(4)))
        self.assertEqual([q.rid for q in s.waiting][:3], ["r10", "r11", "r12"])     # FCFS

    def test_preempt_only_full_rank_and_resume(self):
        """Tiny pools: growth exhausts a rank; the newest sequence OF THAT RANK is moved or preempted (recompute), never
        a sequence elsewhere; preempted requests come back and finish with all their tokens."""
        g = SmallGeometry(paged_pages=60, window_pages=64, state_slots=16, reserve_watermark=0.0,
                          max_seqs_per_rank=8, max_prefill_tokens_per_rank=1024)
        s = RankScheduler(g, CostModel(), SchedConfig(rebalance=False, max_copy_bytes_per_step=0))
        for i in range(12):
            s.add_request(req(f"r{i}", 600, out=900))
        rng = random.Random(1)
        done, now = [], 0.0
        for _ in range(4000):
            if not s.running and not s.waiting:
                break
            p = s.schedule(now)
            for rid in p.preempted:
                self.assertEqual(s.reqs[rid].phase, WAITING)
            acc = {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}
            now += 0.04
            done += s.update(p, acc, now)
            s.kv.check()
        self.assertEqual(sorted(q.rid for q in done), sorted(f"r{i}" for i in range(12)))
        self.assertTrue(all(q.num_output == 900 for q in done))
        self.assertGreater(s.stats["preemptions"], 0)

    def test_pressure_moves_before_preempting(self):
        g = SmallGeometry(paged_pages=40, window_pages=64, state_slots=16, reserve_watermark=0.0, max_seqs_per_rank=8)
        s = RankScheduler(g, CostModel(), SchedConfig(placement="round_robin", rebalance=False))
        # 4 sequences on rank 0 that will outgrow it, nothing elsewhere: round robin puts r0, r4, r8, r12 on rank 0
        for i in range(16):
            s.add_request(req(f"r{i}", 500 if i % 4 == 0 else 64, out=2000 if i % 4 == 0 else 1))
        drive(s, 600)
        self.assertGreater(s.stats["migrations"], 0)
        self.assertEqual(s.stats["preemptions"], 0)


if __name__ == "__main__":
    unittest.main()
