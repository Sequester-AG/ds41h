"""Images through the served path on the CPU: vLLM 0.32's own image processing (token prompts with image placeholders
and PIL images), PhysScheduler's image planner, the fake worker's real encoder cache and its KV content model
(PHYS_FAKE_KV=1: every paged slot remembers the token or image row it holds, prefix hits read it back).

Each request's expected stream is keyed by the content it should see: its prompt tokens, with the image positions
replaced by the fake encoder's rows of ITS image (from vLLM's processed tensors of that request). A prefix cache that
shared KV between different images, a missing or misplaced image slice, or a stale cache entry changes the stream or
fails the fake worker's replay check.

Guards: same tokens with different images never share KV; a reused client uuid with a different image does not alias
(vLLM's own block hashes do); the same image hits; spans cut by chunk boundaries and by the bounded replay after a hit;
preemption + recompute; image prompts kept out of CP (serve's image_pin) and, without the pin, CP with images on
helper ranks; images rejected when the worker or PHYS_MM is off.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_harness import Harness, prompt  # noqa: E402

pytestmark = pytest.mark.vllm      # vLLM on the CPU, fake worker: needs PHYS_MODEL (conftest.py)

IMG = 129264
ENV = {"PHYS_MM": "1", "PHYS_FAKE_KV": "1"}


def _harness(geometry: str = "kv:2e9", **env) -> Harness:
    return Harness(geometry, dict(ENV, **env), mm_processor_cache_gb=0)


def picture(seed: int, w: int = 560, h: int = 560):
    import numpy as np
    from PIL import Image

    return Image.fromarray(np.random.RandomState(seed).randint(0, 255, (h, w, 3), dtype=np.uint8))


def ids_with(tag: str, parts: list) -> list:
    """Token ids: ints are text runs of that length, None is one image placeholder."""
    out = []
    for i, p in enumerate(parts):
        out += [IMG] if p is None else prompt(f"{tag}/{i}", p)
    return out


def internal(h, rid: str):
    m = [r for k, r in h.sched.requests.items() if k == rid or k.startswith(rid + "-")]
    assert len(m) == 1, (rid, len(m))
    return m[0]


def truth(req) -> list:
    """The content the fake model must see: the expanded prompt, image spans replaced by the fake encoder's rows."""
    from adapter import mm as MM
    from adapter.fake_worker import fake_span

    c = list(req.prompt_token_ids)
    for f in req.mm_features:
        rows = fake_span(MM.item_tensors(f.data))
        o = f.mm_position.offset
        assert len(rows) == f.mm_position.length and all(t == IMG for t in c[o:o + len(rows)])
        c[o:o + len(rows)] = rows
    return c


def add(h, rid: str, ids: list, pics: list, max_tokens: int = 24, uuids=None) -> dict:
    from vllm import SamplingParams

    p = {"prompt_token_ids": list(ids), "multi_modal_data": {"image": pics}}
    if uuids is not None:
        p["multi_modal_uuids"] = {"image": uuids}
    h.eng.add_request(rid, p, SamplingParams(max_tokens=max_tokens))
    req = internal(h, rid)
    spans = [(f.mm_position.offset, f.mm_position.length) for f in req.mm_features]
    return dict(rid=rid, content=truth(req), spans=spans, n=max_tokens, block_hashes=list(req.block_hashes),
                internal=req.request_id)


def check(h, done: dict, r: dict) -> None:
    from adapter.fake_worker import fake_token, prompt_key

    o = done[r["rid"]]
    toks = list(o.outputs[0].token_ids)
    c = r["content"]
    k = prompt_key(c)
    exp = [fake_token(h.fake.seed, k, len(c) + j, h.fake.vocab) for j in range(r["n"])]
    assert toks == exp, (r["rid"], [i for i, (a, b) in enumerate(zip(toks, exp)) if a != b][:5])


def test_image_streams_exact():
    """One or two images anywhere in the prompt (first position, middle, last before the turn's end), several sizes:
    every stream matches its own image content; each distinct image is encoded once per rank that needs it."""
    h = _harness()
    rs = [add(h, "one", ids_with("one", [300, None, 200]), [picture(1)]),
          add(h, "two", ids_with("two", [100, None, 50, None, 400]), [picture(2, 800, 600), picture(3, 300, 900)]),
          add(h, "first", ids_with("first", [None, 500]), [picture(4)]),
          add(h, "last", ids_with("last", [700, None, 3]), [picture(5, 1200, 400)])]
    done = h.drain()
    for r in rs:
        check(h, done, r)
    span_rows = sum(n for r in rs for _, n in r["spans"])
    assert h.fake.mm_rows == span_rows, (h.fake.mm_rows, span_rows)
    assert h.sched.mm.stats["encodes"] == 5, h.sched.mm.stats
    h.idle()


def test_different_images_never_share_kv():
    """Same token ids, same image size, different pixels: the second request may hit only the text blocks before the
    image, and its stream is its own image's. The same image again hits through the image."""
    h = _harness()
    ids = ids_with("same", [640, None, 900])
    a = add(h, "a", ids, [picture(10)])
    done = h.drain()
    check(h, done, a)
    b = add(h, "b", ids, [picture(11)])
    assert b["block_hashes"][:10] == a["block_hashes"][:10]               # the text prefix
    done = h.drain()
    check(h, done, b)
    off = b["spans"][0][0]
    assert done["b"].num_cached_tokens <= off // 64 * 64, (done["b"].num_cached_tokens, off)
    c = add(h, "c", ids, [picture(10)])
    done = h.drain()
    check(h, done, c)
    end = off + c["spans"][0][1]
    assert done["c"].num_cached_tokens >= end // 64 * 64, (done["c"].num_cached_tokens, end)
    assert h.fake.replay_checked > 0
    h.idle()


def test_reused_uuid_does_not_alias():
    """Two clients send different images under the same uuid. Stock vLLM 0.32 names an image by the client's uuid, so
    its block hashes of the two prompts are equal through the end and only the content salt keeps their KV apart; a
    vLLM patched to always name media by content gives different hashes. Either way the second request hits only the
    text before the image and its stream is its own; its own image again hits through."""
    h = _harness()
    ids = ids_with("uuid", [256, None, 700])
    a = add(h, "a", ids, [picture(20)], uuids=["shared-id"])
    done = h.drain()
    check(h, done, a)
    b = add(h, "b", ids, [picture(21)], uuids=["shared-id"])
    off = b["spans"][0][0]
    assert a["block_hashes"][:off // 64] == b["block_hashes"][:off // 64]   # the text prefix
    # stock vLLM: equal past the image too (the uuid names it); content-named media: different
    done = h.drain()
    check(h, done, b)
    n = b["spans"][0][1]
    assert done["b"].num_cached_tokens <= off // 64 * 64, (done["b"].num_cached_tokens, off)
    b2 = add(h, "b2", ids, [picture(21)], uuids=["shared-id"])
    done = h.drain()
    check(h, done, b2)
    assert done["b2"].num_cached_tokens >= (off + n) // 64 * 64, done["b2"].num_cached_tokens
    h.idle()


def test_span_cut_by_chunks_and_replay():
    """A long prompt whose images straddle prefill chunk boundaries, then a follow-up whose prefix hit ends just after
    an image, so the bounded replay (the hit's last 128 tokens) re-runs image rows over cached pages. Prefill chunks
    of at most 512 rows (PHYS_GEOM) cut every span longer than that."""
    h = _harness(PHYS_GEOM="max_prefill_tokens_per_rank=512")
    a = add(h, "a", ids_with("long", [3000, None, 2500, None, 4000, None, 40]),
            [picture(30, 1400, 1000), picture(31, 900, 900), picture(32)], max_tokens=16)
    done = h.drain()
    check(h, done, a)
    assert h.fake.mm_split > 0, "no image span was cut by a chunk boundary"
    follow = list(internal_ids(a)) + prompt("follow", 300)
    pics = [picture(30, 1400, 1000), picture(31, 900, 900), picture(32)]
    b = add(h, "b", collapse(follow), pics, max_tokens=16)
    assert b["content"][:len(a["content"])] == a["content"]
    r0 = h.fake.replay_checked
    done = h.drain()
    check(h, done, b)
    last_end = a["spans"][-1][0] + a["spans"][-1][1]
    hit = done["b"].num_cached_tokens
    assert hit - 128 < last_end <= hit + 64, (hit, last_end)                # the replay window covers image rows
    assert h.fake.replay_checked - r0 >= 128
    h.idle()


def internal_ids(r: dict) -> list:
    """The expanded prompt of a request record (image positions as the image token)."""
    return [IMG if v >= 1 << 31 else v for v in r["content"]]


def collapse(ids: list) -> list:
    """Expanded ids -> placeholder form (one image token per span), so vLLM expands the images again."""
    out, prev = [], None
    for t in ids:
        if t == IMG and prev == IMG:
            continue
        out.append(t)
        prev = t
    return out


def test_preemption_recompute_with_images():
    """Tiny pools: preemptions recompute prompts with images (their spans are planned again, reused or re-encoded);
    every stream still matches its content."""
    h = _harness("small:160,96,12")
    rs = [add(h, f"p{i}", ids_with(f"p{i}", [1400 + 300 * (i % 4), None, 1200 + 200 * (i % 3)]), [picture(40 + i)],
              max_tokens=1500) for i in range(12)]
    done = h.drain(max_steps=60000)
    for r in rs:
        check(h, done, r)
    st = h.sched.rs.stats
    assert st["preemptions"] > 0, st
    span_rows = sum(n for r in rs for _, n in r["spans"])
    assert h.sched.mm.stats["rows"] > span_rows, (h.sched.mm.stats, span_rows)    # some spans planned again
    h.idle()


@pytest.mark.parametrize("pin", [True, False], ids=["image_pin", "no_pin"])
def test_cp_prefill_with_images(pin):
    """CP prefill with a long image prompt next to a long text prompt. v1.1 (SchedConfig.image_pin, the default): the
    image prompt never goes CP (it prefills on its owner rank, so its images are encoded there only) while the text
    prompt still does. With image_pin off, the image prompt is split over 4 ranks too and the helpers encode the spans
    their sub-chunks overlap (planner and worker cache ready for an engine that takes CP image rows)."""
    from vllm import SamplingParams

    h = _harness(PHYS_SCHED="cp_prefill=1" + ("" if pin else ",image_pin=0"))
    assert h.sched.rs.cfg.cp_prefill and h.sched.rs.cfg.image_pin == pin
    a = add(h, "cp", ids_with("cp", [1500, None, 2600, None, 2600, None, 2600, None, 1200]),
            [picture(50 + i, 900, 700) for i in range(4)], max_tokens=40)
    text = [("cptext", prompt("cp-text", 12000), SamplingParams(max_tokens=30))]
    h.add(text)
    done = h.drain()
    check(h, done, a)
    h.check_stream(done["cptext"], 30)
    assert h.sched.rs.stats["cp_chunks"] > 0 and h.fake.cp_entries > 0
    ranks = sum(1 for c in h.fake.mm_caches if c.stats["encodes"])
    if pin:
        assert ranks == 1, [c.stats for c in h.fake.mm_caches]
    else:
        assert ranks >= 2, [c.stats for c in h.fake.mm_caches]
    h.idle()


def test_images_rejected_when_off():
    """Without PHYS_MM=1 an image request ends with an error and no tokens; text requests run."""
    from vllm import SamplingParams

    h = Harness("kv:2e9", mm_processor_cache_gb=0)
    assert h.sched.mm is None
    h.eng.add_request("img", {"prompt_token_ids": ids_with("off", [100, None, 100]),
                              "multi_modal_data": {"image": [picture(70)]}}, SamplingParams(max_tokens=8))
    done = h.run([("txt", prompt("off-text", 300), SamplingParams(max_tokens=8))])
    assert done["img"].outputs[0].finish_reason not in ("stop", "length") and not done["img"].outputs[0].token_ids
    h.check_stream(done["txt"], 8)
    assert h.sched.phys_stats["rejected"] == 1
    h.idle()


def test_images_async_dispatch():
    """vLLM's async scheduling (PhysAsyncScheduler) on the D13 dispatch model (PHYS_FAKE_HANDLES=1): image prompts
    arriving among decoding text requests give the same exact streams; the image work rides the planned step."""
    from vllm import SamplingParams
    from vllm_harness import AsyncHarness

    h = AsyncHarness("kv:2e9", dict(ENV, PHYS_FAKE_HANDLES="1"), mm_processor_cache_gb=0)
    text = [(f"t{i}", prompt(f"async-text{i}", 500 + 100 * i), SamplingParams(max_tokens=60)) for i in range(4)]
    h.add(text)
    for _ in range(3):
        h.eng.step()
    rs = [add(h, f"i{i}", ids_with(f"async-img{i}", [300, None, 200 + 50 * i]), [picture(80 + i)], max_tokens=40)
          for i in range(3)]
    done = h.drain()
    for rid, _, sp in text:
        h.check_stream(done[rid], 60)
    for r in rs:
        check(h, done, r)
    assert h.fake.mm_rows == sum(n for r in rs for _, n in r["spans"])
    h.idle()
