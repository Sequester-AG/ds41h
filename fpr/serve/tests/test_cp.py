"""CP prefill, serve side: chunk formation, shadow sequences, the copies, and the StepBatch entries, through the
scheduler alone and through MockEngine's contract checks."""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cost_model import CostModel  # noqa: E402
from geometry import Geometry  # noqa: E402
from scheduler import DECODE, Request, RankScheduler, SchedConfig  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def G():
    return Geometry(kv_bytes_per_rank=6_000_000_000)


def req(rid, plen, out=20, tag=None):
    tag = tag if tag is not None else rid
    return Request(rid, plen, out, 0.0, lambda i, tag=tag: (tag, i))


class TestCPScheduler(unittest.TestCase):
    def _run(self, s, steps=4000, check=None):
        rng = random.Random(0)
        done = []
        for _ in range(steps):
            if not (s.running or s.waiting):
                break
            p = s.schedule()
            if check:
                check(p)
            done += s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"})
        return done

    def test_lone_long_prompt_goes_cp(self):
        """A lone 60K prompt: every chunk split over the 4 ranks, the owner the highest sub-chunk (>= 128 rows), helpers
        64-aligned and contiguous; before each chunk every helper's replica holds all blocks below the chunk start
        (copies + its earlier sub-chunks + the lower sub-chunks it received in-step); the lowest helper gets the window
        halo; shadows are freed when the prefill ends; the request finishes with all its tokens."""
        s = RankScheduler(G(), CostModel(), SchedConfig(cp_prefill=True, rebalance=False, steal=False))
        s.add_request(req("a", 60_000, out=30))
        have = {}                                # helper rank -> blocks its replica holds
        stats = dict(chunks=0, halo=0)

        def check(p):
            cp_work = [w for w in p.work if w.cp is not None]
            if not cp_work:
                return
            stats["chunks"] += 1
            subs = cp_work[0].cp.subs
            owner = cp_work[0].cp.owner
            self.assertEqual(len(cp_work), 4)
            self.assertEqual(subs[-1][0], owner)
            self.assertGreaterEqual(subs[-1][2], 128)
            start = subs[0][1]
            for (r0, s0, n0), (r1, s1, n1) in zip(subs, subs[1:]):
                self.assertEqual(s0 + n0, s1)
                self.assertEqual(n0 % 64, 0)
            self.assertTrue(all(n >= 128 for _, _, n in subs), subs)        # X2's halo: each from the rank below
            for c in p.copies:                   # the step's copies land before its kernels
                if c.pool == "paged":
                    sk = s.kv.seqs[("cp", "a", c.dst_rank)]
                    b = sk.pages.index(c.dst_page)
                    # X1b: from the rank that computed the block in the previous chunk (it holds it), else the owner
                    src_sk = s.kv.seqs["a"] if c.src_rank == owner else s.kv.seqs[("cp", "a", c.src_rank)]
                    self.assertEqual(src_sk.pages[b], c.src_page)
                    if c.src_rank != owner:
                        self.assertIn(b, have.get(c.src_rank, set()))
                        stats["x1b_peer"] = stats.get("x1b_peer", 0) + 1
                    have.setdefault(c.dst_rank, set()).add(b)
                elif c.pool == "window":
                    self.assertEqual(c.dst_rank, subs[0][0])          # the lowest helper
                    stats["halo"] += 1
            for j, (ri, s0, n0) in enumerate(subs[:-1]):
                self.assertTrue(set(range(start // 64)) <= have.get(ri, set()), (ri, start))
                # after the step: its own sub-chunk + the lower sub-chunks (X1a)
                have.setdefault(ri, set()).update(range(s0 // 64 + n0 // 64))
            for w in cp_work:
                if w.kind == "cp":
                    self.assertEqual(w.cached_upto, 0)
                    self.assertIn(w.rid, s.kv.seqs)

        done = self._run(s, check=check)
        self.assertEqual([q.rid for q in done], ["a"])
        self.assertGreater(stats["chunks"], 3)
        self.assertGreater(stats["halo"], 0)
        self.assertGreater(stats.get("x1b_peer", 0), 0)              # some X1b pages came from a helper
        self.assertGreater(s.stats["cp_copy_pages"], 0)
        self.assertFalse([k for k in s.kv.seqs if isinstance(k, tuple) and k[0] == "cp"])   # shadows freed
        self.assertEqual(len(s.free_shadow_uids), 3 * s.g.max_seqs)
        s.kv.check()
        for r in range(4):
            self.assertEqual(s.kv.usage(r)["used"], s.kv.usage(r)["used"])

    def test_short_prompts_and_busy_ranks_stay_owner_only(self):
        """Prompts below cp_min never go CP; a long prompt goes CP only when >= 2 other ranks are light."""
        s = RankScheduler(G(), CostModel(), SchedConfig(cp_prefill=True, rebalance=False, steal=False))
        for i in range(8):
            s.add_request(req(f"s{i}", 6_000))
        seen_cp = []
        self._run(s, check=lambda p: seen_cp.extend(w for w in p.work if w.cp is not None))
        self.assertFalse(seen_cp)
        s = RankScheduler(G(), CostModel(), SchedConfig(cp_prefill=True, rebalance=False, steal=False))
        for i in range(4):                       # every rank busy with its own long prompt
            s.add_request(req(f"L{i}", 40_000))
        cp_steps = []
        self._run(s, check=lambda p: cp_steps.append(any(w.cp is not None for w in p.work)))
        self.assertLess(sum(cp_steps), 0.5 * len(cp_steps))  # CP mostly once the others' prefill has drained

    def test_short_final_chunk_keeps_128_row_subchunks(self):
        """A prompt whose final CP chunk is short: every sub-chunk keeps >= 128 rows, or the chunk stays owner-only."""
        for plen in (8_192 + 300, 8_192 + 600, 9_000, 12_345, 16_384 + 450):
            s = RankScheduler(G(), CostModel(), SchedConfig(cp_prefill=True, rebalance=False, steal=False))
            s.add_request(req("a", plen, out=2))
            subs_seen = []
            self._run(s, check=lambda p: subs_seen.extend(w.cp.subs for w in p.work if w.cp is not None and w.kind != "cp"))
            self.assertTrue(subs_seen)
            for subs in subs_seen:
                self.assertTrue(all(n >= 128 for _, _, n in subs), (plen, subs))
            s.kv.check()

    def test_preempted_cp_prompt_frees_shadows(self):
        s = RankScheduler(G(), CostModel(), SchedConfig(cp_prefill=True, rebalance=False, steal=False))
        s.add_request(req("a", 60_000))
        for _ in range(3):
            p = s.schedule()
            s.update(p, {})
        q = s.reqs["a"]
        self.assertTrue(q.cp_shadows)
        s.abort("a")
        self.assertFalse([k for k in s.kv.seqs if isinstance(k, tuple) and k[0] == "cp"])
        s.kv.check()

    def test_follow_up_goes_cp_after_its_replay(self):
        """A follow-up with a cached prefix: its first chunk is only the bounded replay (positions up to the cached
        boundary, on its own rank), and the rest goes CP from the next mixed step (cp_replay_first)."""
        s = RankScheduler(G(), CostModel(), SchedConfig(cp_prefill=True, cp_min=2048, rebalance=False, steal=False))
        s.add_request(req("a", 20_000, out=2))
        while s.reqs["a"].phase != DECODE:                    # the first call: prefill 20K, then finish
            p = s.schedule()
            s.update(p, {w.rid: w.n for w in p.work if w.kind == "decode"})
        done = []
        for _ in range(200):
            if not (s.running or s.waiting):
                break
            p = s.schedule()
            done += s.update(p, {w.rid: w.n for w in p.work if w.kind == "decode"})
        q2 = Request("b", 26_000, 2, 0.0, lambda i: ("a", i) if i < 20_000 else ("b", i))   # 6K new on a 20K prefix
        s.add_request(q2)
        seen = []
        for _ in range(400):
            if q2.phase == DECODE or q2.rid not in s.reqs:
                break
            p = s.schedule()
            seen.append([(w.kind, w.start, w.n, w.cp is not None) for w in p.work if w.rid in ("b",) or (w.cp and w.cp.rid == "b")])
            s.update(p, {w.rid: w.n for w in p.work if w.kind == "decode"})
        steps = [x for x in seen if x]
        self.assertTrue(steps)
        first = steps[0]
        self.assertTrue(all(not c for _, _, _, c in first), first)                  # the replay part: owner only
        self.assertEqual(max(st + n for _, st, n, _ in first), q2.cached_tokens)
        self.assertTrue(any(c for x in steps[1:] for _, _, _, c in x), steps[:3])   # then CP
        s.kv.check()

    def test_cp_off_by_default(self):
        s = RankScheduler(G(), CostModel(), SchedConfig(rebalance=False, steal=False))
        s.add_request(req("a", 60_000))
        seen = []
        self._run(s, check=lambda p: seen.extend(w for w in p.work if w.cp is not None or w.kind == "cp"))
        self.assertFalse(seen)


class TestBigPromptCadence(unittest.TestCase):
    def test_forced_cp_back_to_back(self):
        """cp_force_min + cp_force_decode_steps=1: a 120K prompt under 48 decoders and the 12K controller gets every
        chunk as 4-rank CP (helpers busy or not) with at most one decode-only step between two of its chunks; its TTFT
        is under half the base config's."""
        from cost_model import ENGINE_MEASURED_2026_11_07, CostParams
        from driver import seed_decoding

        base = dict(cp_prefill=True, cp_min=2048, rebalance=False, steal=False, mixed_target_tokens=12288,
                    mixed_target_fill=True, mixed_target_wait_s=1.0, backlog_drain_wait_s=20.0)
        out = {}
        for name, kw in (("morning", {}), ("big", dict(mixed_target_full_wait_s=0.0, cp_force_min=32768,
                                                          cp_force_room_frac=1.0, cp_force_decode_steps=1))):
            s = RankScheduler(G(), CostModel(CostParams(**ENGINE_MEASURED_2026_11_07)), SchedConfig(**base, **kw))
            seed_decoding(s, [30_000] * 48, out_tokens=100_000)
            for i in range(3):                                  # each helper has its own long prefill pending
                s.add_request(req(f"o{i}", 40_000, out=5))
            s.add_request(req("big", 120_000, out=5))
            rng, now, steps, cp_steps = random.Random(0), 0.0, [], 0
            while s.reqs.get("big") is None or s.reqs["big"].first_token_time is None:
                p = s.schedule(now)
                mine = [w for w in p.work if w.rid == "big" or (w.cp is not None and w.cp.rid == "big")]
                if mine:
                    steps.append(p.step)
                    cp_steps += any(w.kind == "cp" for w in mine)
                now += p.est_us / 1e6
                s.update(p, {w.rid: rng.randint(1, w.n) for w in p.work if w.kind == "decode"}, now)
                self.assertLess(now, 120.0)
            out[name] = (s.reqs["big"].first_token_time, steps, cp_steps)
        t_m, _, _ = out["morning"]
        t_b, steps, cp_steps = out["big"]
        self.assertGreaterEqual(cp_steps, len(steps) - 2)            # CP all the way (the last < cp_min rows: owner)
        self.assertTrue(all(b - a <= 2 for a, b in zip(steps, steps[1:])), steps)   # <= 1 decode-only step between
        self.assertLess(t_b, 0.5 * t_m, (t_b, t_m))                   # sim costs: 2.5 s vs 6.6 s


def written_pages(s, p):
    """("paged" | "window", rank, page) a plan's prefill / CP work writes: the owner's pages of its chunk (the helpers'
    sub-chunks reach the owner in-step, X1a), each shadow's pages of its own + the lower sub-chunks, window pages too."""
    out = set()
    for w in p.work:
        if w.kind == "decode":
            continue
        sk = s.kv.seqs.get(w.rid)
        if sk is None:
            continue
        lo = w.cp.subs[0][1] if w.cp is not None else w.start
        hi = w.start + w.n
        for b in range(lo // 64, -(-hi // 64)):
            if b < len(sk.pages):
                out.add(("paged", w.rank, sk.pages[b]))
        for b in range(lo // 32, -(-hi // 32)):
            if b in sk.window:
                out.add(("window", w.rank, sk.window[b]))
    return out


def run_lagged(s, max_steps=20_000, on_plan=None, seed=0):
    """D13's order without a worker: schedule N+1, then update N (accepted counts drawn at random)."""
    rng = random.Random(seed)
    pending, done, now = None, [], 0.0
    for _ in range(max_steps):
        if not (s.running or s.waiting or pending):
            break
        p = s.schedule(now) if (s.running or s.waiting) else None
        acc_p = None
        if p is not None:
            if on_plan:
                on_plan(p, pending[0] if pending else None)
            acc_p = {w.rid: (w.n if s.reqs[w.rid].forced else rng.randint(1, w.n))
                     for w in p.work if w.kind == "decode"}
        if pending is not None:
            pp, acc = pending
            now += max(pp.est_us, 1.0) / 1e6
            done += s.update(pp, acc, now)
            pending = None
        if p is not None:
            if p.work:
                pending = (p, acc_p)
            else:
                s.update(p, {}, now)
    return done


class TestCPAsync(unittest.TestCase):
    """CP prefill with async scheduling (D13): progress and the phase advance at schedule time, a CP
    chunk only for a prompt whose previous chunk settled (its copies read that chunk's pages), shadows freed after the
    last chunk settles."""

    def _sched(self, **kw):
        from cost_model import ENGINE_MEASURED_2026_11_07, CostParams
        cfg = dict(cp_prefill=True, cp_min=2048, async_mode=True, rebalance=False, steal=False)
        cfg.update(kw)
        return RankScheduler(G(), CostModel(CostParams(**ENGINE_MEASURED_2026_11_07)), SchedConfig(**cfg))

    def test_lone_long_prompt_goes_cp_async(self):
        s = self._sched()
        s.add_request(req("a", 60_000, out=30))
        st = dict(cp=0)

        def on_plan(p, inflight):
            if any(w.kind == "cp" for w in p.work):
                st["cp"] += 1
        done = run_lagged(s, on_plan=on_plan)
        self.assertEqual([q.rid for q in done], ["a"])
        self.assertEqual(done[0].num_output, 30)
        self.assertGreater(st["cp"], 3)
        self.assertFalse([k for k in s.kv.seqs if isinstance(k, tuple) and k[0] == "cp"])   # shadows freed
        self.assertEqual(len(s.free_shadow_uids), 3 * s.g.max_seqs)
        self.assertFalse(s.kv.deferred)
        s.kv.check()
        for r in range(4):
            self.assertEqual(s.kv.usage(r)["used"], 0)

    def test_async_cp_copies_never_read_inflight_pages(self):
        """Under load (decoders + long prompts + short ones, the drain mode's back-to-back mixed steps and the full-chunk
        rule): no copy of step N+1 reads a page step N (in flight) writes; no CP chunk for a prompt with a chunk in
        flight; exact token counts; everything freed."""
        from driver import seed_decoding

        for kw in (dict(), dict(mixed_target_tokens=12288, mixed_target_fill=True, mixed_target_wait_s=1.0,
                                backlog_drain_wait_s=0.0, mixed_target_full_wait_s=0.0, cp_force_min=32768)):
            s = self._sched(**kw)
            seed_decoding(s, [20_000] * 24, out_tokens=200)
            rng = random.Random(3)
            reqs = [req(f"L{i}", rng.randint(30_000, 140_000), out=rng.randint(5, 40)) for i in range(8)]
            reqs += [req(f"sh{i}", rng.randint(200, 6_000), out=rng.randint(5, 40)) for i in range(30)]
            for q in reqs:
                s.add_request(q)
            st = dict(cp=0, copies=0)

            def on_plan(p, inflight):
                if inflight is None:
                    return
                wr = written_pages(s, inflight)
                for c in p.copies:
                    self.assertNotIn((c.pool, c.src_rank, c.src_page), wr, (p.step, c))
                    st["copies"] += 1
                busy = {w.rid for w in inflight.work if w.kind != "decode"}
                for w in p.work:
                    if w.cp is not None and w.kind == "prefill":
                        self.assertNotIn(w.rid, busy)          # its previous chunk is settled
                        st["cp"] += 1
            done = run_lagged(s, on_plan=on_plan)
            self.assertEqual(len(done), 24 + len(reqs), kw)
            for q in reqs:
                self.assertEqual(q.num_output, q.max_new_tokens)
            self.assertGreater(st["cp"], 10 if kw else 0, kw)        # (default: the light-helper gate, as in sync)
            self.assertGreater(st["copies"], 10 if kw else 0, kw)
            self.assertFalse(s.kv.deferred)
            s.kv.check()
            for r in range(4):
                self.assertEqual(s.kv.usage(r)["used"], 0, kw)

    def test_async_settle_gap(self):
        """cp_async_settle_steps: 2 (default) never gives a prompt CP chunks in consecutive steps under async; 1 (the
        engine's D13 guarantee: prefill / copy steps are synchronous fallbacks on the scheduler's thread) does, back to
        back."""
        from driver import seed_decoding

        for gap, want in ((2, False), (1, True)):
            s = self._sched(mixed_target_tokens=12288, mixed_target_fill=True, mixed_target_wait_s=1.0,
                            mixed_target_full_wait_s=0.0, cp_force_min=32768, cp_force_room_frac=1.0,
                            cp_force_decode_steps=0, cp_async_settle_steps=gap)
            seed_decoding(s, [20_000] * 24, out_tokens=200)
            s.add_request(req("big", 150_000, out=5))
            steps = []

            def on_plan(p, inflight):
                if any(w.kind == "cp" and w.cp.rid == "big" for w in p.work):
                    steps.append(p.step)
            done = run_lagged(s, on_plan=on_plan)
            self.assertEqual(len(done), 25)
            consecutive = any(b - a == 1 for a, b in zip(steps, steps[1:]))
            self.assertEqual(consecutive, want, (gap, steps))
            self.assertFalse(s.kv.deferred)
            s.kv.check()

    def test_forced_cp_takes_busy_helpers(self):
        """cp_force_min: a big prompt goes CP although every helper has its own long prefill pending (the light-helper
        gate refuses it); without the option it does not."""
        for fmin, expect in ((0, False), (32768, True)):
            s = RankScheduler(G(), CostModel(), SchedConfig(cp_prefill=True, cp_min=2048, rebalance=False, steal=False,
                                                         cp_force_min=fmin))
            s.add_request(req("big", 120_000, out=5))
            for i in range(3):
                s.add_request(req(f"o{i}", 60_000, out=5))
            p = s.schedule()
            ranks = {s.reqs[r].rank for r in ("big", "o0", "o1", "o2")}
            if len(ranks) < 4:
                self.skipTest(f"placement put the four prompts on {ranks}")
            got = any(w.cp is not None and w.cp.rid == "big" for w in p.work)
            self.assertEqual(got, expect, (fmin, [(w.rid, w.kind, w.n) for w in p.work]))


@unittest.skipUnless(HAVE_TORCH, "torch")
class TestPoisonWorker(unittest.TestCase):
    def test_freed_pages_hold_0xff(self):
        """$PHYS_POISON_ON_FREE=1 through the worker and MockEngine (CP on): after a prompt finishes, the freed window
        pages and state slots hold 0xFF once the next step ran; a second prompt still completes."""
        from driver import OfflineEngine
        from worker import WorkerConfig, WINDOW_PAGE_BYTES

        old = os.environ.get("PHYS_POISON_ON_FREE")
        os.environ["PHYS_POISON_ON_FREE"] = "1"
        try:
            eng = OfflineEngine("engine", cfg=WorkerConfig(engine_cls="mock_engine.MockEngine", reserve_bytes=int(2e9)),
                                sched_cfg=SchedConfig(cp_prefill=True, cp_min=2048, rebalance=False, steal=False))
        finally:
            if old is None:
                del os.environ["PHYS_POISON_ON_FREE"]
            else:
                os.environ["PHYS_POISON_ON_FREE"] = old
        rng = random.Random(5)
        g1 = eng.generate([dict(id="g1", prompt_ids=[rng.randrange(1000, 9000) for _ in range(20_000)], max_tokens=8)])
        self.assertEqual(len(g1[0]["tokens"]), 8)
        freed = {(r, pool, p) for r in range(4) for pool, p in [(pl, pg) for pl, pg in
                 [(x[0], x[1]) for x in eng.sched.kv.take_poison(r)]]}
        for r in range(4):                                   # put them back for the next step's batches
            for _, pool, p in [x for x in freed if x[0] == r]:
                getattr(eng.sched.kv, pool)[r].poison.append(p)
        g2 = eng.generate([dict(id="g2", prompt_ids=[rng.randrange(1000, 9000) for _ in range(6_000)], max_tokens=4)])
        self.assertEqual(len(g2[0]["tokens"]), 4)
        win = [(r, p) for r, pool, p in freed if pool == "window"]
        self.assertTrue(win)
        hit = 0
        for r, p in win:
            if eng.sched.kv.window[r].ref[p] == 0:           # still free: must hold the poison
                t = eng.worker.pools[r].window[p * WINDOW_PAGE_BYTES:(p + 1) * WINDOW_PAGE_BYTES]
                self.assertTrue(bool((t == 255).all()), (r, p))
                hit += 1
        self.assertGreater(hit, 0)


@unittest.skipUnless(HAVE_TORCH, "torch")
class TestCPWorker(unittest.TestCase):
    def test_cp_through_mock_engine(self):
        """generate + force with CP prefill through the worker and MockEngine (CP checks: subs, owner / helper rows;
        D12 deltas rebuilt per shadow slot; window keys): exact token counts, forced rows at the right positions."""
        from driver import OfflineEngine
        from worker import WorkerConfig

        eng = OfflineEngine("engine", cfg=WorkerConfig(engine_cls="mock_engine.MockEngine", reserve_bytes=int(2e9)),
                            sched_cfg=SchedConfig(cp_prefill=True, rebalance=False, steal=False))
        rng = random.Random(3)
        P = [rng.randrange(1000, 9000) for _ in range(50_000)]
        g = eng.generate([dict(id="g", prompt_ids=P[:30_000], max_tokens=40)])
        self.assertEqual(len(g[0]["tokens"]), 40)
        f = eng.force([dict(id="f", prompt_ids=P, forced=P[:24], n_dump=24)])
        self.assertEqual(f[0]["idx"], list(range(24)))
        self.assertGreater(eng.sched.stats["cp_chunks"], 2)
        eng.sched.kv.check()
        self.assertFalse([k for k in eng.sched.kv.seqs if isinstance(k, tuple) and k[0] == "cp"])


@unittest.skipUnless(HAVE_TORCH, "torch")
class TestCPWorkerAsync(unittest.TestCase):
    def test_cp_through_mock_engine_async(self):
        """CP prefill through the async worker path (build_compact + MockEngine.step_compact: the CP group checks and
        the reference row expansion incl. the shadows' state slots), D13's schedule-N+1-before-settling-N order."""
        from driver import run_offline_async, seed_decoding
        from worker import Worker, WorkerConfig

        w = Worker(WorkerConfig(engine_cls="mock_engine.MockEngine", reserve_bytes=int(2e9), async_mode=True),
                   Geometry(kv_bytes_per_rank=int(8e9)), "engine", max_rows=4096 + 320)
        s = RankScheduler(w.geom, CostModel(), SchedConfig(async_mode=True, rebalance=False, steal=False, cp_prefill=True,
                                                           cp_min=2048, cp_force_min=16384))
        seed_decoding(s, [8_000] * 12, out_tokens=80)
        rng = random.Random(5)
        reqs = [req(f"L{i}", rng.randint(20_000, 70_000), out=rng.randint(5, 30)) for i in range(4)]
        reqs += [req(f"sh{i}", rng.randint(200, 3_000), out=rng.randint(5, 30)) for i in range(12)]
        for q in reqs:
            s.add_request(q)
        out = run_offline_async(s, w, 20_000)
        self.assertEqual(out["done"], 12 + len(reqs))
        self.assertGreater(s.stats["cp_chunks"], 4)
        for q in reqs:
            self.assertEqual(q.num_output, q.max_new_tokens)
        self.assertFalse(s.kv.deferred)
        s.kv.check()


if __name__ == "__main__":
    unittest.main()
