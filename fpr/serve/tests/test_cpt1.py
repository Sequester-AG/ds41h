"""Regression tests for the certified scheduler configuration (CPT1 below, passed to the server as PHYS_SCHED).

CPT1 = the 12K mixed-step target with demand-driven fill, context-parallel (CP) prefill for prompts with >= 2,048
tokens left, and the big-prompt rules:
  * forced CP: a prompt with >= 32,768 tokens left goes CP even when every helper rank is busy with its own prefill;
  * its chunk takes the whole per-rank chunk room (cp_force_room_frac 1.0) and runs back to back
    (cp_force_decode_steps 0); a rank holding a full chunk fires a mixed step at the decode floor (full_wait 0);
  * tiered order: prompts with < 131,072 tokens left go first, oldest first; the huge ones after them, so a 95K
    prompt never waits behind a 240K one, and nothing overtakes a mid-size prompt in the middle of its prefill;
  * backlog drain: once a long prefill has waited 20 s for its first chunk, mixed steps may run back to back.
"""
from __future__ import annotations

import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from cost_model import ENGINE_MEASURED_2026_11_07, CostModel, CostParams  # noqa: E402
from driver import seed_decoding  # noqa: E402
from geometry import Geometry  # noqa: E402
from kv_content import run_checked  # noqa: E402
from scheduler import DECODE, Request, RankScheduler, SchedConfig  # noqa: E402

CPT1 = ("mixed_target_tokens=12288,mixed_target_fill=1,mixed_target_wait_s=1.0,cp_prefill=1,cp_min=2048,"
        "backlog_drain_wait_s=20.0,mixed_target_full_wait_s=0,cp_force_min=32768,cp_force_decode_steps=0,"
        "cp_force_room_frac=1.0,cp_force_order=tiered,cp_force_srpt_min_left=131072")


def parse_sched(spec: str, **over) -> SchedConfig:
    """'key=value,...' into SchedConfig the way the server reads PHYS_SCHED: booleans from 1 / true / yes / on, every
    other value converted to the field's default type; an unknown key raises."""
    cfg = SchedConfig()
    for kv in filter(None, (x.strip() for x in spec.split(","))):
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, v.lower() in ("1", "true", "yes", "on") if isinstance(cur, bool) else type(cur)(v))
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def G():
    return Geometry(kv_bytes_per_rank=6_000_000_000)


def req(rid, plen, out=5, prefix=None, prefix_tokens=0, t=0.0):
    pb = prefix_tokens // 64

    def hf(i, rid=rid, prefix=prefix, pb=pb):
        return (prefix, i) if prefix is not None and i < pb else (rid, i)
    return Request(rid, plen, out, t, hf)


def sched(n_decoders=32, ctx=20_000, **over):
    s = RankScheduler(G(), CostModel(CostParams(**ENGINE_MEASURED_2026_11_07)),
                      parse_sched(CPT1, rebalance=False, steal=False, **over))
    if n_decoders:
        seed_decoding(s, [ctx] * n_decoders, out_tokens=100_000)
    return s


def step(s, now):
    p = s.schedule(now)
    now += p.est_us / 1e6
    s.update(p, {w.rid: 1 for w in p.work if w.kind == "decode"}, now)
    return p, now


def cp_of(p, rid):
    return [w for w in p.work if w.cp is not None and w.cp.rid == rid]


def prefill_of(p, rid):
    return [w for w in p.work if w.rid == rid and w.kind == "prefill"]


class TestConfig(unittest.TestCase):
    def test_certified_fields_parse(self):
        """Every field of the certified PHYS_SCHED string exists in SchedConfig with the certified value (a renamed or
        retyped field would make the server refuse the string or silently run another policy)."""
        cfg = parse_sched(CPT1)
        self.assertEqual((cfg.mixed_target_tokens, cfg.mixed_target_fill, cfg.mixed_target_wait_s), (12288, True, 1.0))
        self.assertEqual((cfg.cp_prefill, cfg.cp_min, cfg.backlog_drain_wait_s), (True, 2048, 20.0))
        self.assertEqual((cfg.mixed_target_full_wait_s, cfg.cp_force_min, cfg.cp_force_decode_steps), (0.0, 32768, 0))
        self.assertEqual((cfg.cp_force_room_frac, cfg.cp_force_order, cfg.cp_force_srpt_min_left),
                         (1.0, "tiered", 131072))
        with self.assertRaises(AttributeError):
            parse_sched("no_such_field=1")


class TestForcedCP(unittest.TestCase):
    def _first_chunk(self, big, **over):
        """Every rank holds a backlog of 24K prefills (below the forced threshold, so they stay on their ranks); then a
        prompt of `big` tokens arrives. Returns whether its first chunk is CP."""
        s = sched(**over)
        for i in range(12):
            s.add_request(req(f"o{i}", 24_000))
        now = 0.0
        for _ in range(2):
            _, now = step(s, now)
        s.add_request(req("x", big))
        for _ in range(400):
            p, now = step(s, now)
            if cp_of(p, "x"):
                return True
            if prefill_of(p, "x"):
                return False
        raise AssertionError("no chunk for x")

    def test_threshold(self):
        """>= 32,768 tokens left: the first chunk is CP although every helper is busy (no light-helper gate); below the
        threshold the prompt needs light helpers and starts on its own rank."""
        self.assertTrue(self._first_chunk(40_000))
        self.assertTrue(self._first_chunk(32_768))
        self.assertFalse(self._first_chunk(30_000))
        self.assertFalse(self._first_chunk(40_000, cp_force_min=0))      # control: without forcing, no CP either

    def test_cached_follow_up_goes_cp_after_its_replay(self):
        """A follow-up with 40K new tokens on a cached 20K prefix: its first chunk is the bounded replay on its own rank
        (positions up to the cached boundary), then forced CP chunks."""
        s = sched(n_decoders=0)
        s.add_request(req("a", 20_000, out=1, prefix="P", prefix_tokens=20_000))
        now = 0.0
        while "a" in s.reqs:
            _, now = step(s, now)
        q = req("b", 60_000, prefix="P", prefix_tokens=20_000)
        s.add_request(q)
        seen = []
        while q.phase != DECODE:
            p, now = step(s, now)
            if prefill_of(p, "b") or cp_of(p, "b"):
                seen.append(p)
            self.assertLess(len(seen), 40)
        first = seen[0]
        self.assertFalse(cp_of(first, "b"))
        self.assertEqual(max(w.start + w.n for w in prefill_of(first, "b")), q.cached_tokens)
        self.assertTrue(all(cp_of(p, "b") for p in seen[1:-1]))


class TestBigPromptRoom(unittest.TestCase):
    def _big_and_short(self, **over):
        """A 240K prompt, a 500-token prompt arriving at step 3: (steps with a CP chunk of the big one, the step of the
        short prompt's first chunk)."""
        s = sched(**over)
        s.add_request(req("big", 240_000))
        now, steps, short_first = 0.0, [], None
        for i in range(400):
            if i == 3:
                s.add_request(req("fu", 500))
            p, now = step(s, now)
            if cp_of(p, "big"):
                steps.append(i)
            if short_first is None and prefill_of(p, "fu"):
                short_first = i
            if s.reqs["big"].phase == DECODE and short_first is not None:
                break
        return steps, short_first

    def test_whole_chunk_room(self):
        """While the big prompt prefills, its chunks take the whole chunk room: the short prompt's first rows come with
        the big prompt's last chunk or after it (cp_force_room_frac 1.0). Control: with room 0.5 they come earlier."""
        steps, short_first = self._big_and_short()
        self.assertGreater(len(steps), 10)
        self.assertGreaterEqual(short_first, steps[-1])
        steps, short_first = self._big_and_short(cp_force_room_frac=0.5)
        self.assertLess(short_first, steps[-1])

    def test_back_to_back(self):
        """The big prompt's chunks run in consecutive steps (cp_force_decode_steps 0: no decode-only step between them).
        Control: with the ordinary cadence at least two decode-only steps separate them."""
        steps, _ = self._big_and_short()
        self.assertTrue(all(b - a == 1 for a, b in zip(steps, steps[1:])), steps)
        steps, _ = self._big_and_short(cp_force_decode_steps=-1)
        self.assertTrue(all(b - a >= 3 for a, b in zip(steps, steps[1:])), steps)

    def test_chunk_rows(self):
        """Each chunk of a lone forced prompt but the last fills every rank's 4,096-row prefill room; only a remainder
        below cp_min is prefilled on the owner alone."""
        s = sched()
        s.add_request(req("big", 100_000))
        now, sizes = 0.0, []
        while s.reqs["big"].phase != DECODE:
            p, now = step(s, now)
            c = cp_of(p, "big")
            if c:
                sizes.append(sum(n for _, _, n in c[0].cp.subs))
            self.assertLess(len(sizes), 20)
        self.assertTrue(all(x == 16384 for x in sizes[:-1]), sizes)
        self.assertGreater(sum(sizes), 100_000 - s.cfg.cp_min)       # the rest (< cp_min rows): owner-only


def overtaken(s, now, a="a", b="b", max_steps=400):
    """Step until `a` finishes its prefill; True if `b` got a CP chunk in a step where `a`, still prefilling, got no
    chunk (b sharing the step of a's last, smaller chunk is not overtaking)."""
    for _ in range(max_steps):
        if s.reqs[a].phase == DECODE:
            return False
        p, now = step(s, now)
        if cp_of(p, b) and not (cp_of(p, a) or prefill_of(p, a)):
            return True
    raise AssertionError("prefill did not finish")


class TestTieredOrder(unittest.TestCase):
    def _arrive(self, first, after, second, **over):
        """`first` runs `after` steps (mid-CP), then `second` arrives: does `second` get a CP chunk before `first`
        finishes?"""
        s = sched(**over)
        s.add_request(req("a", first))
        now = 0.0
        for _ in range(after):
            _, now = step(s, now)
        s.add_request(req("b", second))
        return overtaken(s, now)

    def test_mid_size_prompt_is_never_overtaken(self):
        """A 130K prompt (< 131,072 left) in the middle of its prefill keeps the chunk room: a 40K arrival waits.
        Control: plain shortest-first without the guard lets the 40K prompt overtake."""
        self.assertFalse(self._arrive(130_000, 2, 40_000))
        self.assertTrue(self._arrive(130_000, 2, 40_000, cp_force_order="srpt", cp_force_srpt_min_left=0))

    def test_huge_prompt_yields_to_a_mid_size_one(self):
        """A 240K prompt (>= 131,072 left) yields to a 90K arrival: the smaller big prompt does not wait behind it.
        Control: oldest-first order makes the 90K prompt wait."""
        self.assertTrue(self._arrive(240_000, 2, 90_000))
        self.assertFalse(self._arrive(240_000, 2, 90_000, cp_force_order="fcfs"))

    def test_huge_prompt_below_the_guard_is_not_overtaken(self):
        """Once the huge prompt has fewer than 131,072 tokens left it is in the first tier and, being older, keeps
        its slot against a 90K arrival."""
        s = sched()
        s.add_request(req("a", 200_000))
        now = 0.0
        while s.reqs["a"].prefill_left >= 131_072 or s.reqs["a"].num_computed == 0:
            _, now = step(s, now)
            self.assertLess(now, 30.0)
        s.add_request(req("b", 90_000))
        self.assertFalse(overtaken(s, now))

    def test_oldest_first_within_a_tier(self):
        """Two forced prompts under 131,072 tokens arriving one step apart: the older one gets every chunk until its
        prefill ends, then the younger one starts."""
        s = sched()
        s.add_request(req("a", 60_000))
        now = 0.0
        _, now = step(s, now)
        s.add_request(req("b", 50_000))
        order = []
        while s.reqs["b"].phase != DECODE:
            p, now = step(s, now)
            for rid in ("a", "b"):
                if cp_of(p, rid):
                    order.append(rid)
            self.assertLess(now, 60.0)
        self.assertEqual(order, sorted(order))                      # every "a" chunk before every "b" chunk
        self.assertIn("a", order)
        self.assertIn("b", order)


class TestBacklogDrain(unittest.TestCase):
    def _drain_run(self, **over):
        s = sched(n_decoders=48, ctx=30_000, **over)
        for i in range(64):
            s.add_request(req(f"L{i}", 24_000))
        now, mixed, waited = 0.0, [], False
        for i in range(3000):
            drain = any(q.first_chunk_time is None and q.prefill_left > s.cfg.small_prefill_tokens
                        and q.admit_time is not None and now - q.admit_time >= 20.0 for q in s.prefilling)
            waited = waited or drain
            p, now = step(s, now)
            if any(w.kind != "decode" for w in p.work):
                mixed.append((i, drain))
            if not s.prefilling:
                break
        return waited, mixed

    def test_drain_after_20s(self):
        """A standing backlog of 24K prompts (below the forced threshold) on every rank under 48 decoders: mixed steps
        keep at least 2 decode-only steps between them until a prompt has waited 20 s for its first chunk; from then on
        they may run back to back until the backlog's first chunks are out."""
        waited, mixed = self._drain_run()
        self.assertTrue(waited, "the backlog never waited 20 s")
        gaps_before = [b[0] - a[0] for a, b in zip(mixed, mixed[1:]) if not b[1]]
        gaps_drain = [b[0] - a[0] for a, b in zip(mixed, mixed[1:]) if b[1]]
        self.assertTrue(gaps_before and min(gaps_before) >= 3, gaps_before[:10])
        self.assertIn(1, gaps_drain)
        # control: without the drain the floor holds throughout
        waited, mixed = self._drain_run(backlog_drain_wait_s=float("inf"))
        self.assertTrue(waited)
        self.assertGreaterEqual(min(b[0] - a[0] for a, b in zip(mixed, mixed[1:])), 3)


class TestCpt1EndToEnd(unittest.TestCase):
    def test_content_checked_run(self):
        """A random workload under the certified config, steals and rebalancing on: shared system prefixes, follow-ups,
        big prompts up to 150K (forced CP), every plan executed on the KV content model. Every read finds its data
        (owner pages, helper replicas, window halo, state slot), every registered page holds its block, every request
        finishes with all its tokens, and all CP shadows are released."""
        for seed in (21, 22):
            rng = random.Random(seed)
            s = RankScheduler(G(), CostModel(CostParams(**ENGINE_MEASURED_2026_11_07)),
                              parse_sched(CPT1, rebalance_patience=2, rebalance_every=2))
            arrivals, n = {}, 0
            for st in range(0, 1200, 5):
                if n >= 60:
                    break
                ptoks = rng.choice([1_152, 1_152, 6_016])
                body = rng.choice([500, 3_000, 12_000, 40_000, 90_000, 150_000])
                arrivals.setdefault(st, []).append(
                    req(f"q{n}", ptoks + rng.randrange(body // 2, body + 1), out=rng.choice([5, 60, 400]),
                        prefix=f"sys{n % 2}", prefix_tokens=ptoks))
                n += 1
            model, done = run_checked(s, 20_000, random.Random(seed), arrivals)
            self.assertFalse(model.errors, (seed, model.errors[:4]))
            self.assertEqual(len(done), n, seed)
            self.assertTrue(all(q.num_output == q.max_new_tokens for q in done))
            self.assertGreater(s.stats["cp_chunks"], 10)
            self.assertGreater(model.stats["reads_cp"], 0)
            self.assertFalse([k for k in s.kv.seqs if isinstance(k, tuple) and k[0] == "cp"])
            self.assertEqual(len(s.free_shadow_uids), 3 * s.g.max_seqs)
            s.kv.check()


if __name__ == "__main__":
    unittest.main()
