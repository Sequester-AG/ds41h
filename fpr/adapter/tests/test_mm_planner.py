"""The host side of images (adapter/mm.py): content keys, the KV salt of a request's blocks, the image planner and the
worker's encoder cache. Plain CPU tests on synthetic step batches.

Guards: an image span cut by a chunk boundary gets each slice once and is encoded once; a request keeps its image
resident while its prefill still has rows of the span ahead; eviction never drops a key that is needed now or still
referenced; CP helpers encode what their sub-chunk overlaps; the worker's cache mirrors the planner exactly (a drop or
an encode it does not expect fails); the content key, not the client's identifier, decides what is shared.
"""
from __future__ import annotations

import os
import random
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from adapter import mm as MM  # noqa: E402

P = 64


def item(seed: int, n_llm=(2, 3)) -> dict:
    """Processed tensors of a synthetic image: span length n_h * (n_w + 1) + 2."""
    h, w = n_llm
    g = torch.Generator().manual_seed(seed)
    types = [0] + ([1] * w + [2]) * h + [3]
    return dict(patches=torch.randn(h * w * 9, 3, 14, 14, generator=g).to(torch.bfloat16),
                vit_grid=torch.tensor([3 * h, 3 * w]), llm_grid=torch.tensor([h, w]),
                types=torch.tensor(types, dtype=torch.int64))


def image(offset: int, seed: int, n_llm=(2, 3)) -> MM.Image:
    t = item(seed, n_llm)
    return MM.Image(offset, MM.span_length(t), MM.content_key(t), t)


def span_values(t: dict) -> torch.Tensor:
    """The synthetic encoder: span row j of an image = a value derived from its bytes and j."""
    c = int(t["patches"].float().sum().item() * 1000) & 0xFFFF
    return torch.tensor([c * 100_000 + j for j in range(MM.span_length(t))], dtype=torch.int64)


def encode(items):
    return [span_values(it.data) for it in items]


def entry(rid, start, n, row0=0, cp=None, kind="prefill"):
    return SimpleNamespace(rid=rid, kind=kind, start=start, n_rows=n, row0=row0, cp=cp)


def batch(rank, seqs):
    return SimpleNamespace(rank=rank, seqs=seqs)


def apply_all(caches, st):
    return [c.apply(st) for c in caches]


# ------------------------------------------------------------------------------------------------ keys
def test_content_key_is_the_content():
    """Equal processed tensors give equal keys; one changed patch value or a changed grid gives another key."""
    a, b = item(1), item(1)
    assert MM.content_key(a) == MM.content_key(b)
    c = item(1)
    c["patches"][0, 0, 0, 0] += 1
    assert MM.content_key(c) != MM.content_key(a)
    d = item(1)
    d["vit_grid"] = torch.tensor([9, 6])
    assert MM.content_key(d) != MM.content_key(a)
    assert MM.content_key(item(2)) != MM.content_key(a)


def test_images_in_prompt_order():
    im1, im2 = image(100, 1), image(300, 2)
    ri = MM.ReqImages([im2, im1], P)                         # any order in, prompt order inside
    assert [im.offset for im in ri.images] == [100, 300]
    assert ri.overlapping(0, 100) == [] and ri.overlapping(105, 301) == [im1, im2]


# ------------------------------------------------------------------------------------------------ planner
def test_chunk_boundary_inside_a_span():
    """A span cut by a chunk boundary: each chunk gets its slice (lo, n), the image is encoded once, and the request
    keeps it resident between the chunks."""
    im = image(10, 3, (3, 4))                                # 3 * 5 + 2 = 17 positions: [10, 27)
    pl = MM.MMPlanner(1, 1 << 30, 2, P)
    pl.add("a", [im])
    cache = MM.EncoderCache(0, encode)
    st = pl.plan([batch(0, [entry("a", 0, 20)])])
    assert st.encode[0] == [MM.MMItem(im.key, 17, None)] and st.rows[0] == [MM.MMRows(10, 10, im.key, 0)]
    assert pl.refs[0][im.key] == {"a"}
    idx, emb = cache.apply(st)
    assert idx.tolist() == list(range(10, 20)) and emb.tolist() == span_values(im.data)[:10].tolist()
    st = pl.plan([batch(0, [entry("a", 20, 30, row0=5)])])
    assert st.encode[0] == [] and st.rows[0] == [MM.MMRows(5, 7, im.key, 10)]
    assert im.key not in pl.refs[0]                          # the span is done on this rank: released, still resident
    idx, emb = cache.apply(st)
    assert idx.tolist() == list(range(5, 12)) and emb.tolist() == span_values(im.data)[10:].tolist()
    assert pl.stats["encodes"] == 1 and pl.stats["reuses"] == 1
    assert pl.plan([batch(0, [entry("a", 50, 10)])]) is None  # rows past every span: no image work


def test_eviction_respects_need_and_refs():
    """Over the budget, the least recently used unreferenced key goes first; a key needed this step or still held by
    a prefill is never dropped (the budget is soft then)."""
    ims = [image(0, s) for s in range(4)]                    # 10 positions each (2 x 3 grid)
    L = ims[0].length
    nb = L * 2
    pl = MM.MMPlanner(1, 2 * nb, 2, P)
    cache = MM.EncoderCache(0, encode)
    for i, im in enumerate(ims):
        pl.add(f"r{i}", [im])
    apply_all([cache], pl.plan([batch(0, [entry("r0", 0, L)])]))
    apply_all([cache], pl.plan([batch(0, [entry("r1", 0, L)])]))
    st = pl.plan([batch(0, [entry("r2", 0, L)])])
    assert st.drop[0] == [ims[0].key]                        # r0's image: oldest, unreferenced
    apply_all([cache], st)
    st = pl.plan([batch(0, [entry("r3", 0, 4)])])              # r3 holds its image (rows ahead)
    assert st.drop[0] == [ims[1].key]
    apply_all([cache], st)
    st = pl.plan([batch(0, [entry("r2", 0, L)])])              # r2's image is resident: reuse, nothing to drop
    assert st.encode[0] == [] and st.drop[0] == []
    apply_all([cache], st)
    pl.add("r4", [image(0, 7)])
    st = pl.plan([batch(0, [entry("r4", 0, 4)])])              # r3 and r4 both hold theirs: r2's goes
    assert st.drop[0] == [ims[2].key] and pl.used[0] == 2 * nb
    apply_all([cache], st)
    pl.add("r5", [image(0, 8)])
    st = pl.plan([batch(0, [entry("r5", 0, 4)])])              # everything resident is held: soft budget
    assert st.drop[0] == [] and pl.stats["over_cap_steps"] == 1
    apply_all([cache], st)
    pl.release("r3")
    st = pl.plan([batch(0, [entry("r5", 4, L - 4)])])
    assert st.drop[0] == [ims[3].key]
    apply_all([cache], st)
    assert set(cache.store) == set(pl.res[0])
    pl.check()


def test_same_image_two_requests_one_encode():
    """Two requests with the same image content on one rank: one encode, both get the rows; the key is the content,
    so it does not matter what vLLM's identifier said."""
    a, b = image(0, 5), image(30, 5)
    assert a.key == b.key
    pl = MM.MMPlanner(1, 1 << 30, 2, P)
    pl.add("a", [a])
    pl.add("b", [b])
    st = pl.plan([batch(0, [entry("a", 0, a.length, row0=0), entry("b", 30, b.length, row0=a.length)])])
    assert len(st.encode[0]) == 1 and [m.row0 for m in st.rows[0]] == [0, a.length]
    idx, emb = MM.EncoderCache(0, encode).apply(st)
    assert emb.tolist() == span_values(a.data).tolist() * 2


def test_cp_helpers_encode_their_sub_chunk_without_refs():
    """CP: the helper entry carries the prompt's rid and its sub-chunk; it encodes what that sub-chunk overlaps (an
    image straddling two sub-chunks is encoded on both ranks) and holds no ref; the owner holds its ref."""
    im = image(120, 6, (4, 6))                               # 4 * 7 + 2 = 30 positions: [120, 150)
    pl = MM.MMPlanner(2, 1 << 30, 2, P)
    pl.add("q", [im])
    subs = [(0, 0, 128), (1, 128, 128)]
    helper = SimpleNamespace(owner=1, subs=subs, me=0)
    owner = SimpleNamespace(owner=1, subs=subs, me=1)
    st = pl.plan([batch(0, [entry("q", 0, 128, cp=helper)]), batch(1, [entry("q", 128, 128, cp=owner)])])
    assert st.rows[0] == [MM.MMRows(120, 8, im.key, 0)] and st.rows[1] == [MM.MMRows(0, 22, im.key, 8)]
    assert len(st.encode[0]) == 1 and len(st.encode[1]) == 1
    assert im.key not in pl.refs[0] and im.key not in pl.refs[1]      # the span ended in this chunk on the owner
    caches = [MM.EncoderCache(r, encode) for r in range(2)]
    (i0, e0), (i1, e1) = apply_all(caches, st)
    assert e0.tolist() + e1.tolist() == span_values(im.data).tolist()


def test_cache_rejects_a_plan_it_does_not_mirror():
    """The worker's cache fails loudly on a drop or rows of a key it does not hold, and on encoding a resident key."""
    im = image(0, 9)
    c = MM.EncoderCache(0, encode)
    st = MM.MMStep([[MM.MMItem(im.key, im.length, im.data)]], [[MM.MMRows(0, 2, im.key, 0)]], [[]])
    c.apply(st)
    with pytest.raises(AssertionError):
        c.apply(st)                                           # encode of a resident key
    with pytest.raises(AssertionError):
        c.apply(MM.MMStep([[]], [[]], [["img-unknown"]]))
    with pytest.raises(AssertionError):
        c.apply(MM.MMStep([[]], [[MM.MMRows(0, 2, "img-unknown", 0)]], [[]]))


@pytest.mark.parametrize("seed", range(12))
def test_random_schedules_mirror_and_cover(seed):
    """Random requests with random images, chunked at random over 4 ranks, with preemptions (restart from 0, maybe on
    another rank) and a tight budget: every image row of every chunk is covered exactly once with the right span
    values, the worker's cache always holds exactly what the planner models, and the budget holds unless every
    resident key is in use."""
    rng = random.Random(seed)
    R = 4
    pl = MM.MMPlanner(R, rng.choice([40, 200, 1 << 20]), 2, P)
    caches = [MM.EncoderCache(r, encode) for r in range(R)]
    pool = [item(s, (rng.randint(1, 3), rng.randint(1, 4))) for s in range(6)]   # shared images: cross-request reuse
    reqs = {}
    for i in range(10):
        n, ims, pos = rng.randint(40, 400), [], rng.randint(0, 30)
        while True:
            t = rng.choice(pool)
            ln = MM.span_length(t)
            if pos + ln > n:
                break
            ims.append(MM.Image(pos, ln, MM.content_key(t), t))
            pos += ln + rng.randint(0, 40)
        rid = f"q{i}"
        pl.add(rid, ims)
        reqs[rid] = dict(n=n, done=0, rank=rng.randrange(R), ims=ims)
    for step in range(400):
        live = [rid for rid, q in reqs.items() if q["done"] < q["n"]]
        if not live:
            break
        rows_of = [[] for _ in range(R)]
        for rid in rng.sample(live, min(len(live), rng.randint(1, 4))):
            q = reqs[rid]
            if rng.random() < 0.07:                           # preempted: recompute from 0, maybe elsewhere
                pl.release(rid)
                q["done"], q["rank"] = 0, rng.randrange(R)
                continue
            r = q["rank"]
            n = min(q["n"] - q["done"], rng.choice([16, 32, 64, 100]))
            rows_of[r].append((rid, q["done"], n))
            q["done"] += n
        bs = []
        for r in range(R):
            row, seqs = 0, []
            for rid, s0, n in rows_of[r]:
                seqs.append(entry(rid, s0, n, row0=row))
                row += n
            bs.append(batch(r, seqs))
        st = pl.plan(bs)
        for rid, q in reqs.items():
            if q["done"] >= q["n"]:
                pl.forget(rid)                                # finished: its refs go, its keys stay resident
        got = apply_all(caches, st)
        for r in range(R):
            assert set(caches[r].store) == set(pl.res[r]), (step, r)
            want = {}
            for e in bs[r].seqs:
                for im in reqs[e.rid]["ims"]:
                    for p in range(max(e.start, im.offset), min(e.start + e.n_rows, im.offset + im.length)):
                        want[e.row0 + p - e.start] = int(span_values(im.data)[p - im.offset])
            have = {} if got[r] is None else dict(zip(got[r][0].tolist(), got[r][1].tolist()))
            assert have == want, (step, r)
            assert got[r] is None or len(got[r][0]) == len(have), (step, r, "a row covered twice")
            if pl.used[r] > pl.cap:
                assert all(k in pl.refs[r] or any(m.key == k for m in st.rows[r]) for k in pl.res[r]), (step, r)
        pl.check()


def test_stage4_worker_image_hook():
    """The GPU worker's image hook without a GPU: each MMItem goes to the tower's encode_one with vLLM's tensors, and a
    step's image work lands in ps.mm_dev per rank (the caller's PhysStep is left unchanged)."""
    pytest.importorskip("vllm")
    from adapter.stage4_worker import Stage4Worker

    calls = []

    class Tower:
        def encode_one(self, patches, vit_grid, llm_grid, types):
            calls.append((tuple(patches.shape), tuple(int(x) for x in vit_grid), tuple(int(x) for x in llm_grid)))
            return span_values(dict(patches=patches, vit_grid=vit_grid, llm_grid=llm_grid, types=types))

    enc = Stage4Worker._encoder(Tower(), None)
    im = image(0, 4, (2, 2))
    out = enc([MM.MMItem(im.key, im.length, im.data)])
    assert calls == [((36, 3, 14, 14), (6, 6), (2, 2))] and out[0].tolist() == span_values(im.data).tolist()

    w = Stage4Worker.__new__(Stage4Worker)
    w.mm_caches = [MM.EncoderCache(r, enc) for r in range(2)]
    pl = MM.MMPlanner(2, 1 << 30, 2, P)
    pl.add("q", [im])
    st = pl.plan([batch(0, []), batch(1, [entry("q", 0, im.length, row0=3)])])
    ps = SimpleNamespace(mm=st)
    ps2 = w._mm_apply(ps)
    assert not hasattr(ps, "mm_dev") and ps2.mm_dev[0] is None
    idx, emb = ps2.mm_dev[1]
    assert idx.tolist() == list(range(3, 3 + im.length)) and emb.tolist() == span_values(im.data).tolist()
    assert w._mm_apply(SimpleNamespace(mm=None)).mm is None
