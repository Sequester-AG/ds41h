"""Regression tests for image prompts in the scheduler (serve/scheduler.py: Request.images, ImageBlockKeys, pinned).

An image fills a range of placeholder positions in the prompt, all with the same token id, so the token stream alone
does not identify the KV of those positions or of any position after them. The scheduler keys every block that ends
after an image's first position by the token-stream key plus the (offset, identifier) of every image before the
block's end, and lists for each prefill work item the images it covers (their encoder outputs are inputs of that step,
the bounded replay below a cache hit included). Image prompts prefill on their owner rank only (no context-parallel
chunks) and are not moved (stolen, rebalanced, relieved) before their prefill completes.

The content checks run every plan through kv_content.KVContent with a `content_fn` per request: what a block's KV
really depends on (its tokens and every image before it), independent of the cache key under test. The model reports
a cache key written with two different contents, so a page shared by two prompts with different images is caught
even before anything reads it."""
from __future__ import annotations

import os
import random
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from cost_model import CostModel  # noqa: E402
from geometry import Geometry, SmallGeometry  # noqa: E402
from kv_content import run_checked  # noqa: E402
from interfaces import build_step_batch, image_rows  # noqa: E402
from scheduler import DECODE, ImageBlockKeys, RankScheduler, Request, SchedConfig  # noqa: E402
from test_cpt1 import cp_of, prefill_of, sched as cpt1_sched, step  # noqa: E402
from test_moves import track_moves  # noqa: E402

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

BT = 64
SYS = 1_152          # a shared text prefix (system prompt), page-aligned
IMG = 576            # placeholder positions of one image


def content(text, images):
    """Ground truth of block i: its token-stream key, and for a block ending after an image's first position also the
    (offset, image) of every image before the block's end."""
    offs = [(o, c) for o, _, c in images]

    def f(i):
        before = tuple((o, c) for o, c in offs if o < (i + 1) * BT)
        return text(i) if not before else ("img", text(i), before)
    return f


def ireq(rid, plen, images=(), out=20, stream="S", shared_tokens=None, t=0.0):
    """A prompt whose first `shared_tokens` tokens (default: all of them) are the token stream `stream` (keys
    (stream, i); placeholders included, so prompts with different images can have equal streams), the rest its own,
    with `images` = ((offset, length, image id), ...)."""
    sb = (plen if shared_tokens is None else shared_tokens) // BT

    def text(i, rid=rid, stream=stream, sb=sb):
        return (stream, i) if i < sb else (rid, i)
    q = Request(rid, plen, out, t, text, images=tuple(images))
    q.content_fn = content(text, tuple(images))
    return q


def strip_image_keys(s):
    """Make `s` key image prompts by their token stream only (the bug the image keys prevent)."""
    add = s.add_request

    def add_text_keys(q):
        ok = add(q)
        if isinstance(q.hash_fn, ImageBlockKeys):
            q.hash_fn = q.hash_fn.base
        return ok
    s.add_request = add_text_keys


def track_phases(s):
    """Wrap migrate (after track_moves): the phase of every moved sequence at the time of its move."""
    log = []
    migrate = s.kv.migrate

    def tracked(seq_id, dst, step_, num_computed):
        q = s.reqs.get(seq_id)
        ops = migrate(seq_id, dst, step_, num_computed)
        if ops is not None and q is not None:
            log.append((seq_id, q.phase, bool(q.images)))
        return ops
    s.kv.migrate = tracked
    return log


class TestImageBlockKeys(unittest.TestCase):
    def setUp(self):
        self.s = RankScheduler(Geometry(kv_bytes_per_rank=6_000_000_000), CostModel(), SchedConfig())

    def keys(self, q, n=None):
        return [q.hash_fn(i) for i in range((n or q.prompt_len) // BT)]

    def test_text_prompt_keys_unchanged(self):
        """A prompt without images keeps its token-stream keys object as is (text prompts behave as before)."""
        q = ireq("t", 4_000)
        base = q.hash_fn
        self.assertTrue(self.s.add_request(q))
        self.assertIs(q.hash_fn, base)

    def test_keys_split_at_the_first_image_block(self):
        """Equal placeholder tokens, different images: equal keys for blocks ending at or before the image's offset
        (a shared system prompt still hits), different keys for the block holding the image's first position and for
        every block after the image."""
        for off in (SYS, SYS + 18):                     # page-aligned and unaligned image starts
            a, b = ireq(f"a{off}", 4_000, [(off, IMG, "img-A")]), ireq(f"b{off}", 4_000, [(off, IMG, "img-B")])
            self.assertTrue(self.s.add_request(a) and self.s.add_request(b))
            ka, kb = self.keys(a), self.keys(b)
            first = off // BT
            self.assertEqual(ka[:first], kb[:first])
            for i in range(first, len(ka)):
                self.assertNotEqual(ka[i], kb[i], (off, i))

    def test_same_images_same_keys(self):
        """Equal token streams with equal images give equal keys: a later prompt with the same image hits the whole
        cached prefix, image blocks included."""
        a, b = ireq("a", 4_000, [(SYS, IMG, "img-A")]), ireq("b", 6_000, [(SYS, IMG, "img-A")])
        self.assertTrue(self.s.add_request(a) and self.s.add_request(b))
        self.assertEqual(self.keys(a), self.keys(b, 4_000))

    def test_image_position_and_order_matter(self):
        """The same image at another offset, or two images in swapped order, give other keys from the first block that
        holds an image position of either prompt."""
        a, b = ireq("a", 4_000, [(SYS, IMG, "img-A")]), ireq("b", 4_000, [(SYS + 64, IMG, "img-A")])
        c = ireq("c", 4_000, [(SYS, IMG, "img-A"), (2_048, IMG, "img-B")])
        d = ireq("d", 4_000, [(SYS, IMG, "img-B"), (2_048, IMG, "img-A")])
        for q in (a, b, c, d):
            self.assertTrue(self.s.add_request(q))
        ka, kb, kc, kd = self.keys(a), self.keys(b), self.keys(c), self.keys(d)
        for i in range(SYS // BT, len(ka)):
            self.assertNotEqual(ka[i], kb[i], i)
            self.assertNotEqual(kc[i], kd[i], i)

    def test_malformed_images_rejected(self):
        """A malformed image list (overlapping spans, a span outside the prompt, zero length, no identifier) is
        rejected by add_request with a reason and never queued: wrong spans would give wrong cache keys."""
        bad = [[(SYS, IMG, "a"), (SYS + 100, IMG, "b")], [(3_900, IMG, "a")], [(SYS, 0, "a")], [(SYS, IMG, "")],
               [(SYS + IMG, 64, "a"), (SYS, IMG, "b")]]
        for i, images in enumerate(bad):
            q = Request(f"bad{i}", 4_000, 5, 0.0, lambda j: ("S", j), images=tuple(images))
            self.assertFalse(self.s.add_request(q), images)
            self.assertIn(f"bad{i}", self.s.rejected)
            self.assertNotIn(q, self.s.waiting)


def image_workload(seed, n=96):
    """Shared text prefixes, then an image (one of four, three sizes) in most prompts, and the same token stream after
    it, so prompts with different images often have equal token streams; mixed output lengths; text-only prompts in
    between."""
    rng = random.Random(seed)
    arrivals = {}
    for k in range(n):
        ptoks = rng.choice([SYS, 6_016])
        plen = ptoks + 1_800 + rng.randrange(0, rng.choice([1_200, 7_000, 18_000]))
        images = ()
        if rng.random() < 0.7:
            images = ((ptoks, rng.choice([256, IMG, 1_024]), f"img{rng.randrange(4)}"),)
            if rng.random() < 0.3:
                images += ((ptoks + 1_536, 256, f"img{rng.randrange(4)}"),)
        arrivals.setdefault(3 * k, []).append(
            ireq(f"q{k}", plen, images, out=rng.choice([5, 40, 600, 2000]), stream=f"sys{k % 2}"))
    return arrivals


def fuzz_sched(cp=False, **over):
    return RankScheduler(Geometry(kv_bytes_per_rank=150_000_000, window_pool_bytes=60_000_000, state_pool_slots=128),
                         CostModel(), SchedConfig(rebalance_patience=2, rebalance_every=2, cp_prefill=cp, cp_min=2048,
                                                  **over))


class TestImagePrefixCache(unittest.TestCase):
    def test_no_kv_shared_across_images(self):
        """Many prompts with equal token streams and different images under tight pools (prefix hits, cross-rank prefix
        copies, steals, rebalances, relief moves): no cache key is ever written with two contents, every read finds
        its own content, every registered page holds its key's content, and prompts with the same image reuse the
        cached image blocks."""
        for seed, cp in ((31, False), (32, True)):
            s = fuzz_sched(cp)
            arrivals = image_workload(seed)
            model, done = run_checked(s, 8_000, random.Random(seed), arrivals)
            self.assertFalse(model.errors, (seed, model.errors[:4]))
            self.assertEqual(len(done), sum(len(v) for v in arrivals.values()), seed)
            self.assertTrue(all(q.num_output == q.max_new_tokens for q in done))
            # image blocks were served from the cache for prompts with the same image (the keys still share)
            reused = [q for q in done if q.images and q.hit_tokens_total > q.images[0][0] + q.images[0][1]]
            self.assertTrue(reused, seed)
            self.assertGreater(len(model.key_content), 0)

    def test_model_catches_text_only_keys(self):
        """The same workload keyed by token stream only must be reported (a guard that the test can see the bug):
        prompts with equal placeholder tokens but different images would share pages."""
        s = fuzz_sched()
        strip_image_keys(s)
        model, _ = run_checked(s, 8_000, random.Random(31), image_workload(31))
        self.assertTrue(any(e[0] in ("one prefix-cache key for two different contents", "stale or missing paged KV")
                            for e in model.errors), model.errors[:3])


class TestImageOwnerOnly(unittest.TestCase):
    def _chunks(self, images, plen=40_000, steps=400):
        """The certified scheduler (forced CP for >= 32,768 tokens left) with 32 decoders, then one long prompt. Returns
        its prefill work items and CP work items over the run."""
        s = cpt1_sched()
        q = ireq("x", plen, images, out=5, stream="X")
        self.assertTrue(s.add_request(q))
        now, pf, cp = 0.0, [], []
        for _ in range(steps):
            p, now = step(s, now)
            pf += prefill_of(p, "x")
            cp += cp_of(p, "x")
            if images:
                self.assertEqual(q.cp_shadows, {})
            if q.phase == DECODE:
                break
        self.assertEqual(q.phase, DECODE)
        return s, q, pf, cp

    def test_long_image_prompt_prefills_owner_only(self):
        """A 40K-token prompt with an image never gets context-parallel chunks or helper shadows: every chunk runs on
        its owner rank (a v1.1 performance limit). The same prompt without the image goes CP (control)."""
        s, q, pf, cp = self._chunks([(SYS, IMG, "img-A")])
        self.assertEqual(cp, [])
        self.assertTrue(pf)
        self.assertTrue(all(w.cp is None and w.rank == pf[0].rank for w in pf))
        self.assertEqual(sum(w.n for w in pf), 40_000)
        _, _, _, cp_text = self._chunks(())
        self.assertTrue(cp_text)

    def test_cached_image_prompt_not_cut_to_its_replay(self):
        """A follow-up on a cached image prefix with >= cp_min new tokens: its first chunk is not cut to the 128 replay
        rows (that cut only prepares a CP chunk, which an image prompt never gets). A text follow-up is cut (control)."""
        for images, cut in (([(SYS, IMG, "img-A")], False), ((), True)):
            s = cpt1_sched()
            warm = ireq("w", 8_000, images, out=2, stream="W")
            self.assertTrue(s.add_request(warm))
            now = 0.0
            while warm.phase != DECODE:
                _, now = step(s, now)
            f = ireq("f", 20_000, images, out=2, stream="W", shared_tokens=8_000)
            self.assertTrue(s.add_request(f))
            first = None
            for _ in range(200):
                p, now = step(s, now)
                w = prefill_of(p, "f") + cp_of(p, "f")
                if w:
                    first = w[0]
                    break
            self.assertIsNotNone(first)
            self.assertEqual(f.cached_tokens, 8_000)             # the whole warm prompt hits (same images)
            self.assertEqual(first.n == f.cached_tokens - f.replay_start, cut, (images, first.n))


class TestImageNoEarlyMoves(unittest.TestCase):
    def test_image_prompts_not_stolen(self):
        """Follow-ups on a cached 20K prefix with an image queue on the prefix's rank, where prefill stealing would
        move not-yet-started ones to idle ranks: no image follow-up is stolen or moved, each prefills where it was
        admitted. The same scenario with text follow-ups steals (control)."""
        for with_image in (True, False):
            s = RankScheduler(Geometry(kv_bytes_per_rank=6_000_000_000), CostModel(),
                              SchedConfig(rebalance=False, migrate_min_blocks=10 ** 9, placement_tol=1.0))
            log = track_moves(s)
            rng = random.Random(4)
            images = ((SYS, IMG, "img-P"),) if with_image else ()
            arrivals = {0: [ireq(f"bg{i}", 8_000 + 64 * i, out=3000, stream=f"bg{i}") for i in range(8)],
                        30: [ireq("warm", 20_000, images, out=2, stream="P")],
                        90: [ireq(f"f{i}", 20_000 + rng.randrange(2_000, 8_000), images, out=30, stream="P",
                                  shared_tokens=20_000) for i in range(12)]}
            model, done = run_checked(s, 1500, random.Random(0), arrivals)
            self.assertFalse(model.errors, model.errors[:3])
            self.assertEqual(sum(1 for q in done if q.rid.startswith("f")), 12)
            steals = [m for m in log if m["why"] == "steal"]
            if with_image:
                self.assertEqual(steals, [])
                self.assertTrue(all(q.n_migrate == 0 for q in done if q.rid.startswith("f")))
            else:
                self.assertTrue(steals)

    def test_no_image_prompt_moves_before_its_prefill_completes(self):
        """Tight pools, mixed image and text prompts: rebalances and relief moves happen, but no image prompt moves
        before it decodes, and decoding image prompts do move, with their KV intact."""
        moved_dec = 0
        whys = set()
        for seed, cp in ((41, False), (42, False), (43, True)):
            s = fuzz_sched(cp)
            log = track_moves(s)
            phases = track_phases(s)
            arrivals = image_workload(seed)
            model, done = run_checked(s, 8_000, random.Random(seed), arrivals)
            self.assertFalse(model.errors, (seed, model.errors[:4]))
            self.assertEqual(len(done), sum(len(v) for v in arrivals.values()), seed)
            self.assertEqual([x for x in phases if x[2] and x[1] != DECODE], [], seed)
            moved_dec += sum(1 for x in phases if x[2] and x[1] == DECODE)
            whys |= {m["why"] for m in log}
        self.assertLessEqual({"rebalance", "relief"}, whys)
        self.assertGreater(moved_dec, 0)

    def test_relief_preempts_an_image_prefill_instead_of_moving_it(self):
        """A rank runs out of pages while its newest sequence, an image prompt, is still prefilling: memory relief
        would move that sequence to another rank; for the image prompt the move is refused and it is preempted
        (recomputed from the start, here on another rank) instead. The same prompt without the image is moved
        (control)."""
        for with_image in (True, False):
            g = SmallGeometry(paged_pages=37, window_pages=64, state_slots=16, reserve_watermark=0.0,
                              max_seqs_per_rank=8, max_prefill_tokens_per_rank=64)
            s = RankScheduler(g, CostModel(), SchedConfig(placement="round_robin", rebalance=False, steal=False))
            log = track_moves(s)
            phases = track_phases(s)
            q = ireq("img", 1_664, ((256, 512, "img-A"),) if with_image else (), out=20, stream="I")
            arrivals = {0: [ireq(f"r{i}", 640, out=600, stream=f"r{i}") for i in range(4)], 12: [q]}
            model, done = run_checked(s, 4000, random.Random(3), arrivals)
            self.assertFalse(model.errors, model.errors[:3])
            self.assertEqual(len(done), 5)
            self.assertEqual(q.num_output, 20)
            early = [m for m in log if m["rid"] == "img" and m["why"] == "relief"]
            if with_image:
                self.assertEqual(early, [])
                self.assertEqual([x for x in phases if x[0] == "img"], [])
                self.assertGreater(s.stats.get("image_moves_refused", 0), 0)
                self.assertGreater(q.n_preempt, 0)
            else:
                self.assertTrue(early)
                self.assertTrue(any(x[0] == "img" and x[1] != DECODE for x in phases))


class TestWorkImages(unittest.TestCase):
    def test_prefill_work_lists_its_images(self):
        """Every prefill work item lists exactly the request's images overlapping its positions (chunks that split an
        image list it in each chunk), decode work lists none, and the chunks of a prompt together list every image
        from its replay start on."""
        s = fuzz_sched()
        seen = {}

        def on_step(i, p):
            for w in p.work:
                q = s.reqs.get(w.rid)
                if q is None or w.kind != "prefill":
                    self.assertEqual(w.images, (), w.kind)
                    continue
                want = tuple(im for im in q.images if im[0] < w.start + w.n and im[0] + im[1] > w.start)
                self.assertEqual(w.images, want, (w.rid, w.start, w.n))
                seen.setdefault(w.rid, set()).update(w.images)
        arrivals = image_workload(51)
        model, done = run_checked(s, 8_000, random.Random(51), arrivals, on_step=on_step)
        self.assertFalse(model.errors, model.errors[:3])
        for q in done:
            if q.n_preempt == 0:
                self.assertEqual(seen.get(q.rid, set()),
                                 {im for im in q.images if im[0] + im[1] > q.replay_start}, q.rid)

    def test_bounded_replay_over_an_image_lists_it(self):
        """A follow-up whose cache hit ends within 128 positions after an image: its first chunk re-runs the last 128
        cached positions (the bounded replay), which include image positions, so the image is an input of that chunk."""
        s = RankScheduler(Geometry(kv_bytes_per_rank=6_000_000_000), CostModel(), SchedConfig(steal=False))
        img = ((SYS, IMG, "img-A"),)
        warm = ireq("w", SYS + IMG + 64, img, out=2, stream="W")         # 1,792 tokens: 28 full blocks
        s.add_request(warm)
        now, first = 0.0, None
        for _ in range(50):
            p = s.schedule(now)
            now += 0.05
            s.update(p, {w.rid: 1 for w in p.work if w.kind == "decode"}, now)
            if warm.phase == "finished":
                break
        f = ireq("f", 6_000, img, out=2, stream="W", shared_tokens=SYS + IMG + 64)
        s.add_request(f)
        for _ in range(50):
            p = s.schedule(now)
            now += 0.05
            w = [x for x in p.work if x.rid == "f" and x.kind == "prefill"]
            s.update(p, {x.rid: 1 for x in p.work if x.kind == "decode"}, now)
            if w:
                first = w[0]
                break
        self.assertIsNotNone(first)
        self.assertLess(first.start, SYS + IMG)            # the replay starts inside the image
        self.assertGreaterEqual(f.cached_tokens, SYS + IMG)
        self.assertEqual(first.images, img)


class TestImagePinOff(unittest.TestCase):
    def test_pin_off_allows_cp_and_moves_with_exact_image_lists(self):
        """SchedConfig.image_pin = False (for an engine that takes image rows on CP helpers and encodes per rank):
        image prompts get CP chunks and may move mid-prefill like text prompts; the KV stays exact, and every CP entry
        (helper and owner) lists the images overlapping its own sub-chunk."""
        cp_img = 0
        for seed in (61, 62):
            s = fuzz_sched(True, image_pin=False)

            def on_step(i, p, s=s):
                nonlocal cp_img
                for w in p.work:
                    if w.cp is None:
                        continue
                    q = s.reqs.get(w.cp.rid)
                    want = tuple(im for im in q.images if im[0] < w.start + w.n and im[0] + im[1] > w.start)
                    self.assertEqual(w.images, want, (w.rid, w.start, w.n))
                    cp_img += bool(want)
            arrivals = image_workload(seed)
            model, done = run_checked(s, 8_000, random.Random(seed), arrivals, on_step=on_step)
            self.assertFalse(model.errors, (seed, model.errors[:4]))
            self.assertEqual(len(done), sum(len(v) for v in arrivals.values()), seed)
        self.assertGreater(cp_img, 0)
        # memory relief moves an image prompt mid-prefill (with the pin it is preempted instead)
        g = SmallGeometry(paged_pages=37, window_pages=64, state_slots=16, reserve_watermark=0.0,
                          max_seqs_per_rank=8, max_prefill_tokens_per_rank=64)
        s = RankScheduler(g, CostModel(), SchedConfig(placement="round_robin", rebalance=False, steal=False,
                                                      image_pin=False))
        phases = track_phases(s)
        q = ireq("img", 1_664, ((256, 512, "img-A"),), out=20, stream="I")
        model, done = run_checked(s, 4000, random.Random(3),
                                  {0: [ireq(f"r{i}", 640, out=600, stream=f"r{i}") for i in range(4)], 12: [q]})
        self.assertFalse(model.errors, model.errors[:3])
        self.assertEqual(len(done), 5)
        self.assertTrue(any(x[0] == "img" and x[1] != DECODE for x in phases))


def worker_and_sched(**cfg):
    from worker import Worker, WorkerConfig

    g = Geometry(kv_bytes_per_rank=6_000_000_000, max_prefill_tokens_per_rank=1024)
    w = Worker(WorkerConfig(engine_cls="mock_engine.MockEngine", reserve_bytes=int(2e9)), g, "engine", max_rows=2048)
    return w, RankScheduler(w.geom, CostModel(), SchedConfig(**cfg))


def gather(batches, keys):
    """A stand-in for the adapter's encode + gather: per rank (idx int32, emb) for interfaces.image_rows, emb row =
    (image number, span row) so the engine side can be checked against the plan."""
    out = []
    for sb in batches:
        rows = image_rows(sb)
        if not rows:
            out.append(None)
            continue
        idx = torch.tensor([r for r, _, _ in rows], dtype=torch.int32)
        emb = torch.tensor([[float(keys[k]), float(lo)] for _, k, lo in rows], dtype=torch.float32)
        out.append((idx, emb))
    return out


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class TestWorkerImageRows(unittest.TestCase):
    def _run(self, reqs, with_mm=True, steps=400):
        """Drive reqs (list of (arrival step, Request)) through Worker.step (the adapter's protocol) on MockEngine,
        handing the gathered image rows over as Worker.step(mm=...). Returns (worker, finished)."""
        w, s = worker_and_sched(steal=False)
        keys = {}
        for _, q in reqs:
            for im in q.images:
                keys.setdefault(im[2], len(keys))
        seen, done, now = set(), [], 0.0
        rng = random.Random(5)
        for i in range(steps):
            for t, q in reqs:
                if t == i:
                    s.add_request(q)
            if not (s.running or s.waiting) and all(t < i for t, _ in reqs):
                break
            p = s.schedule(now)
            if not p.work:
                now += 0.05
                continue
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
                                       preempted=set(p.preempted), prev_tokens={}, order=[])
            res = w.step(ps, mm=gather(batches, keys) if with_mm else None)
            mm_last = getattr(w.backend, "mm_last", {})
            for r, (irows, emb) in list(mm_last.items()):     # the engine got (image, span row) for each image row
                self.assertEqual([(keys[k], lo) for _, k, lo in irows], [(int(a), int(b)) for a, b in emb.tolist()])
            w.backend.mm_last = {}
            acc = {}
            for x in res:
                acc.update(x.accepted)
            now += 0.05
            done += s.update(p, acc, now)
        return w, done

    def test_image_rows_reach_the_engine(self):
        """Image prompts (an image split across two chunks, a cached follow-up whose replay re-runs image positions,
        a text prompt alongside): Worker.step passes the per-rank image rows into StepContext.mm, the engine sees
        exactly the image positions of each step's entries, and every image position after the cache hit gets one
        embedding."""
        img_a = ((900, 1024, "img-A"),)                   # positions 900 .. 1923: split by the 1,024-row chunks
        a = ireq("a", 2_500, img_a, out=4, stream="A")
        b = ireq("b", SYS + IMG + 64, ((SYS, IMG, "img-B"),), out=4, stream="B")    # 28 full blocks
        f = ireq("f", 3_000, ((SYS, IMG, "img-B"),), out=4, stream="B", shared_tokens=SYS + IMG + 64)
        t = ireq("t", 1_500, out=4, stream="T")
        w, done = self._run([(0, a), (0, b), (0, t), (40, f)])
        self.assertEqual(len(done), 4)
        self.assertGreater(f.cached_tokens, SYS + IMG - 128)          # the replay starts inside image B
        self.assertLess(f.replay_start, SYS + IMG)
        want = 1024 + IMG + (SYS + IMG - f.replay_start)              # a's whole image, b's, f's replayed part
        self.assertEqual(w.backend.mm_rows, want)

    def test_missing_image_rows_are_refused(self):
        """A step whose entries have image positions but carries no embeddings is refused by the engine side (the
        rows would otherwise take the placeholder token's embedding)."""
        a = ireq("a", 2_500, ((900, 1024, "img-A"),), out=4, stream="A")
        with self.assertRaises(AssertionError):
            self._run([(0, a)], with_mm=False)


if __name__ == "__main__":
    unittest.main()
