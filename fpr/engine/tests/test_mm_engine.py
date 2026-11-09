"""Images in the engine's prefill (FPRv1.1), on 4 GPUs, through the worker <-> engine contract.

The worker hands each step StepContext.mm = [per rank (idx int32 [M], emb bf16 [M, 5120])]: the rank's StepBatch rows
inside an image span and their span embeddings (vision tower output: aligner rows at IMAGE, the learned delimiter
vectors at START / NEWLINE / END). The engine (PHYS_MM=1):
  - p_embed: those rows take emb instead of the token-embedding lookup, before the hyper-connection copies;
  - p_router: rows with ids 129264..129268 are routed with the router's bias_vl on every MoE layer (bucket A: layers 0..20
    on all rows; bucket T: layers 21..39 on each chunk's tail rows);
  - Engram already treats 129264 as dead (no n-grams through it, gate 0), as production does.
Here the test plays the worker: it encodes the prompt's images with engine/vision.py's tower on the ranks that need
them and builds ctx.mm from the step's batches (owner chunks and, with CP, the helpers' sub-chunks).

1. test_image_prompt[cp off / on]: 8 decoding sequences plus one 3,000-token prompt with two images (spans at 400 and
   1,800) arriving at step 1. Checked during the run: every span row's embedding is the span row bit for bit; on
   every layer that sees image rows their experts are production's (reference router with bias_vl); head inputs finite;
   the prompt decodes.
2. test_text_is_unchanged: a text workload with the image path enabled equals the same workload with it disabled
   (engine.MM_ON False = the FPRv1 text path), bit for bit, in one process. If they differ, a third disabled run
   tells the known in-process decode nondeterminism (disabled vs disabled differs too) from a real change.
Synthetic inputs only: random patches, hashed token ids.
"""
from __future__ import annotations

import math
import os

import pytest

os.environ.setdefault("PHYS_MM", "1")                 # before the engine module is imported (drive.engine_once)

SPANS = [(400, (36, 48)), (1_800, (20, 29))]           # (offset, ViT patch grid): LLM grids 12 x 16 and 7 x 10
PROMPT, RID = 3_000, "a1_0"


@pytest.fixture(scope="module")
def eng():
    import drive
    return drive.engine_once("sync", "")


def _images():
    import torch
    from engine.vision import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START, ImageInput
    out = []
    for i, (off, (vh, vw)) in enumerate(SPANS):
        lh, lw = math.ceil(vh / 3), math.ceil(vw / 3)
        ty = torch.tensor([IMAGE_START] + ([IMAGE] * lw + [IMAGE_NEW_LINE]) * lh + [IMAGE_END], dtype=torch.int64)
        g = torch.Generator().manual_seed(100 + i)
        pt = (torch.rand(vh * vw, 3, 14, 14, generator=g) * 2 - 1).to(torch.bfloat16)
        out.append(ImageInput(off, pt, (vh, vw), (lh, lw), ty))
    return out


IMAGES = None


def _base(rid):
    """A CP helper's sub-chunk entry carries rid ('cp', <request rid>, i)."""
    return rid[1] if isinstance(rid, tuple) and len(rid) == 3 and rid[0] == "cp" else rid


def _token_fn():
    import drive
    from engine.vision import IMAGE_TOKEN_ID
    spans = [(x.offset, x.offset + x.length) for x in IMAGES]

    def f(rid, pos):
        if _base(rid) == RID and any(a <= pos < b for a, b in spans):
            return IMAGE_TOKEN_ID
        return drive.tok(rid, pos)
    return f


class Worker:
    """The worker's side of the contract: span embeddings per (rank, image), encoded on first need on that rank's
    GPU, and ctx.mm for each step from its batches."""

    def __init__(self, eng):
        self.eng, self.towers, self.spans, self.encoded = eng, {}, {}, 0

    def span(self, r, i):
        import torch
        from engine.vision import VisionTower
        if (r, i) not in self.spans:
            dev = self.eng.devs[r]
            if r not in self.towers:
                self.towers[r] = VisionTower(self.eng.cfg.model, dev)
            x = IMAGES[i]
            with torch.cuda.device(dev), torch.cuda.stream(self.eng.streams[r]):
                self.spans[(r, i)] = self.towers[r].encode_one(x.patches, x.vit_grid, x.llm_grid, x.types)
            self.encoded += 1
        return self.spans[(r, i)]

    def ctx(self, k, batches):
        import torch
        from interfaces import StepContext
        mm, any_ = [], False
        for r, sb in enumerate(batches):
            rows, src = [], []
            for se in sb.seqs:
                if _base(se.rid) != RID or se.kind == "decode" or not se.n_rows:
                    continue
                p0 = int(sb.positions[se.row0])
                for i, x in enumerate(IMAGES):
                    lo, hi = max(p0, x.offset), min(p0 + se.n_rows, x.offset + x.length)
                    if lo < hi:
                        rows.append(torch.arange(se.row0 + lo - p0, se.row0 + hi - p0))
                        src.append(self.span(r, i)[lo - x.offset: hi - x.offset])
            if rows:
                dev = self.eng.devs[r]
                with torch.cuda.device(dev), torch.cuda.stream(self.eng.streams[r]):
                    mm.append((torch.cat(rows).to(torch.int32).to(dev), torch.cat(src)))
                any_ = True
            else:
                mm.append(None)
        c = StepContext()
        c.mm = mm if any_ else None
        return c


def _wl(cp: bool, arrive=True):
    import drive
    # a 2,048-token mixed-step target (below the prompt): the 3,000-token prompt is prefilled in the step it arrives
    return drive.Workload(ctxs=[8_192 + 977 * i for i in range(8)], steps=10, arrive={1: [PROMPT]} if arrive else {},
                          arrive_out=8, cp=cp, cp_min=2048,
                          sched="mixed_target_tokens=2048,mixed_target_fill=1,mixed_target_wait_s=0.0,"
                                "backlog_drain_wait_s=20.0")


@pytest.mark.gpu(n=4)
@pytest.mark.parametrize("cp", [False, True], ids=["cp_off", "cp_on"])
def test_image_prompt(eng, cp):
    import torch
    import drive
    from engine import engine as E
    global IMAGES
    IMAGES = _images()
    assert E.MM_ON
    seen = dict(embed_rows=0, embed_bad=0, route_layers=0, route_rows=0, route_bad=0, cp_rows=0)
    orig_embed, orig_rvl = E.Engine.p_embed, E.Engine._router_vl

    def p_embed(self, r, s):
        orig_embed(self, r, s)
        mm = s.get("mm_emb")
        if mm is not None:
            rows, src = mm
            got = s["res0"].index_select(0, rows)
            seen["embed_rows"] += rows.numel()
            seen["embed_bad"] += int((got != src.unsqueeze(1).expand_as(got)).any(-1).any(-1).sum())
            b = s["batch"]
            seen["cp_rows"] += sum(se.n_rows for se in b.seqs if getattr(se, "cp", None) is not None
                                   and _base(se.rid) == RID)

    def router_vl(self, r, s, L):
        ri = s["mm_rt"]
        sc = torch.nn.functional.softplus(s["logits"].index_select(0, ri)).sqrt()
        want = (sc + self._bias_vl(r, L)).topk(6, dim=-1).indices.sort(dim=-1).values
        orig_rvl(self, r, s, L)
        got = s["ids"].index_select(0, ri).long().sort(dim=-1).values
        seen["route_layers"] += 1
        seen["route_rows"] += ri.numel()
        seen["route_bad"] += int((got != want).any(-1).sum())

    w = Worker(eng)
    E.Engine.p_embed, E.Engine._router_vl = p_embed, router_vl
    try:
        res = drive.run(eng, _wl(cp), token_fn=_token_fn(), ctx_fn=w.ctx)
    finally:
        E.Engine.p_embed, E.Engine._router_vl = orig_embed, orig_rvl
    n_span = sum(x.length for x in IMAGES)
    assert seen["embed_rows"] == n_span and seen["embed_bad"] == 0, seen
    assert seen["route_layers"] >= 21 and seen["route_rows"] >= n_span * 21, seen        # layers 0..20 (+ tail rows)
    assert seen["route_bad"] <= seen["route_rows"] // 1000, seen                          # fp32 near-ties only
    if cp:
        assert res.stats.get("cp_steps", 0) > 0 and seen["cp_rows"] > 0, (res.stats.get("cp_steps"), seen)
    assert all(s["nonfinite"] == 0 for s in res.steps)
    assert len(res.streams.get(RID, [])) >= 2
    assert res.stats.get("mm_rows_plain", 0) == 0


@pytest.mark.gpu(n=4)
def test_text_is_unchanged(eng):
    import drive
    from engine import engine as E

    def go(on):
        old = E.MM_ON
        E.MM_ON = on
        try:
            return drive.run(eng, _wl(cp=True))           # a1_0 is a text prompt here (default token ids)
        finally:
            E.MM_ON = old
    a, b = go(False), go(True)
    same = lambda x, y: (x.streams == y.streams and [s["h_md5"] for s in x.steps] == [s["h_md5"] for s in y.steps])  # noqa: E731
    if same(a, b):
        return
    c = go(False)
    if not same(a, c):
        pytest.xfail("disabled vs disabled differs too (the known in-process decode-step nondeterminism)")
    raise AssertionError("enabling the image path changed a text-only run")
