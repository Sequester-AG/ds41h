"""Images for the phys engine: content keys, the host-side image planner (MMPlanner) and the worker's per-rank
encoder cache (EncoderCache).

vLLM 0.32 expands every image placeholder into a span of positions that all carry IMAGE_TOKEN_ID. The span's
embeddings (aligner rows plus the learned delimiter vectors, engine/vision.py) replace the token embeddings of those
positions at the first layer. vLLM's own encoder scheduling is not used (PhysScheduler sends no
scheduled_encoder_inputs). After the rank scheduler's plan, MMPlanner maps the rows of every prefill entry to the spans
they cover, and keeps per rank a model of the spans resident in the worker's encoder cache. Per step and rank:

  encode[r]  images to encode on rank r before the step's kernels (needed now, not resident)
  rows[r]    MMRows(row0, n, key, lo): rows row0 .. row0 + n - 1 of rank r's StepBatch take span rows lo .. lo + n - 1
  drop[r]    keys to free on rank r (unreferenced, least recently used, over the cache budget)

The rows of a prefill entry are an owner chunk, the replay rows after a prefix hit, a CP helper's sub-chunk or a
recompute after a preemption: all are planned the same way. A request holds a key on a rank while its prefill there
still has rows of the span ahead. A released key stays resident (LRU) until evicted, so a replay or a recompute can
reuse it. The worker's EncoderCache applies drop, encode and gather in that order, and asserts that it mirrors the
planner.

Keys are content keys (blake2b over the processed tensors), never vLLM's identifier alone: stock vLLM 0.32 takes a
client-supplied uuid as the identifier, and two different images under one uuid must share neither an encoder output
nor KV (the rank scheduler salts the KV block keys with these keys: Request.images, serve/scheduler.py). Stock vLLM's processor cache is keyed by the same
uuid, so with stock vLLM serve with --mm-processor-cache-gb 0; a vLLM that names media by content needs neither.
"""
from __future__ import annotations

import bisect
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

IMAGE_TOKEN_ID = 129264          # every position of an image span (the checkpoint's image_token_id)
ITEM_KEYS = ("patches", "vit_grid", "llm_grid", "types")


@dataclass(frozen=True)
class MMItem:
    """One image to encode: its content key, its span length and vLLM's processed tensors (CPU)."""
    key: str
    length: int
    data: Any = field(compare=False, repr=False)     # {"patches", "vit_grid", "llm_grid", "types"} -> torch.Tensor


@dataclass(frozen=True)
class MMRows:
    """Rows row0 .. row0 + n - 1 of a rank's StepBatch take rows lo .. lo + n - 1 of image `key`'s span."""
    row0: int
    n: int
    key: str
    lo: int


@dataclass
class MMStep:
    """The image work of one step, per rank (PhysStep.mm)."""
    encode: list
    rows: list
    drop: list

    def empty(self) -> bool:
        return not any(self.encode) and not any(self.rows) and not any(self.drop)


@dataclass(frozen=True)
class Image:
    """One image of a request: its span [offset, offset + length) in the prompt."""
    offset: int
    length: int
    key: str
    data: Any = field(compare=False, repr=False)


def item_tensors(data) -> dict:
    """vLLM's MultiModalKwargsItem (key -> MultiModalFieldElem) or a plain dict -> {name: tensor}."""
    out = {}
    for k in ITEM_KEYS:
        v = data[k]
        out[k] = getattr(v, "data", v)
    return out


def content_key(t: dict) -> str:
    """blake2b over the item's processed tensors (dtype, shape and bytes of each): equal keys <=> equal encoder input."""
    import torch

    h = hashlib.blake2b(digest_size=16)
    for k in ITEM_KEYS:
        x = t[k].detach().cpu().contiguous()
        h.update(f"{k}:{x.dtype}:{tuple(x.shape)};".encode())
        if x.numel():
            h.update(memoryview(x.view(torch.uint8).numpy()))      # large buffers: hashlib releases the GIL
    return "img-" + h.hexdigest()


def span_length(t: dict) -> int:
    return int(t["types"].numel())


class ReqImages:
    """A request's images in prompt order."""

    def __init__(self, images: list, page_tokens: int):
        self.images = sorted(images, key=lambda im: im.offset)
        self.P = page_tokens
        self.offsets = [im.offset for im in self.images]

    def overlapping(self, s0: int, s1: int):
        """Images whose span overlaps positions [s0, s1)."""
        j = bisect.bisect_left(self.offsets, s1)
        out = []
        for im in self.images[:j]:
            if im.offset + im.length > s0:
                out.append(im)
        return out


class MMPlanner:
    """Host-side planner of the image work (see the module docstring). Deterministic, no device state."""

    def __init__(self, n_ranks: int, cap_bytes: int, row_bytes: int, page_tokens: int):
        self.R, self.cap, self.row_bytes, self.P = n_ranks, int(cap_bytes), int(row_bytes), page_tokens
        self.reqs: dict = {}                                    # rid -> ReqImages
        self.res = [OrderedDict() for _ in range(n_ranks)]      # key -> bytes, least recently used first
        self.refs = [dict() for _ in range(n_ranks)]            # key -> set of rids still prefilling over its span
        self.used = [0] * n_ranks
        self.stats = dict(encodes=0, encode_rows=0, rows=0, drops=0, reuses=0, over_cap_steps=0)

    # -- requests
    def add(self, rid, images: list) -> ReqImages:
        ri = ReqImages(images, self.P)
        self.reqs[rid] = ri
        return ri

    def get(self, rid) -> Optional[ReqImages]:
        return self.reqs.get(rid)

    def release(self, rid) -> None:
        """The request stopped prefilling (preempted, finished, aborted): its keys stay resident but evictable."""
        for refs in self.refs:
            for key in [k for k, s in refs.items() if rid in s]:
                s = refs[key]
                s.discard(rid)
                if not s:
                    del refs[key]

    def forget(self, rid) -> None:
        self.release(rid)
        self.reqs.pop(rid, None)

    # -- per step
    def plan(self, batches: list) -> Optional[MMStep]:
        R = self.R
        enc, rows, drop = [[] for _ in range(R)], [[] for _ in range(R)], [[] for _ in range(R)]
        need = [set() for _ in range(R)]
        for sb in batches:
            r = sb.rank
            for se in sb.seqs:
                if se.kind == "decode" or se.start < 0:
                    continue
                ri = self.reqs.get(se.rid)
                if ri is None:
                    continue
                s0, s1 = se.start, se.start + se.n_rows
                cp = getattr(se, "cp", None)
                owner = cp is None or cp.owner == r           # CP helpers hold no refs (their sub-chunk ends here)
                for im in ri.overlapping(s0, s1):
                    a, b = max(s0, im.offset), min(s1, im.offset + im.length)
                    rows[r].append(MMRows(se.row0 + a - s0, b - a, im.key, a - im.offset))
                    self.stats["rows"] += b - a
                    if im.key not in need[r]:
                        need[r].add(im.key)
                        res = self.res[r]
                        if im.key in res:
                            res.move_to_end(im.key)
                            self.stats["reuses"] += 1
                        else:
                            enc[r].append(MMItem(im.key, im.length, im.data))
                            nb = im.length * self.row_bytes
                            res[im.key] = nb
                            self.used[r] += nb
                            self.stats["encodes"] += 1
                            self.stats["encode_rows"] += im.length
                    if owner:
                        refs = self.refs[r]
                        if im.offset + im.length > s1:
                            refs.setdefault(im.key, set()).add(se.rid)
                        elif im.key in refs:
                            refs[im.key].discard(se.rid)
                            if not refs[im.key]:
                                del refs[im.key]
        for r in range(R):
            if self.used[r] <= self.cap:
                continue
            res, refs = self.res[r], self.refs[r]
            for key in list(res):
                if self.used[r] <= self.cap:
                    break
                if key in need[r] or key in refs:
                    continue
                self.used[r] -= res.pop(key)
                drop[r].append(key)
                self.stats["drops"] += 1
            if self.used[r] > self.cap:
                self.stats["over_cap_steps"] += 1           # everything resident is in use: the budget is soft
        st = MMStep(enc, rows, drop)
        return None if st.empty() else st

    def check(self) -> None:
        for r in range(self.R):
            assert self.used[r] == sum(self.res[r].values()), (r, self.used[r])
            for key, s in self.refs[r].items():
                assert s and key in self.res[r], (r, key, s)


class EncoderCache:
    """One rank's encoder outputs (the worker side of MMPlanner). encode_fn(list[MMItem]) -> one tensor
    [length, hidden] per item, on this rank's device. apply() runs drop, encode and gather for one step and returns
    (idx int32 [M], emb [M, hidden]) for the rank's image rows, or None."""

    def __init__(self, rank: int, encode_fn: Callable, device=None):
        self.rank, self.encode_fn, self.device = rank, encode_fn, device
        self.store: dict = {}
        self.stats = dict(encodes=0, rows=0, drops=0)

    def apply(self, mm: Optional[MMStep]):
        import torch

        if mm is None:
            return None
        r = self.rank
        for key in mm.drop[r]:
            assert key in self.store, (r, key, "drop of a key the cache does not hold")
            del self.store[key]
            self.stats["drops"] += 1
        items = mm.encode[r]
        if items:
            for it in items:
                assert it.key not in self.store, (r, it.key, "encode of a resident key")
            outs = self.encode_fn(items)
            assert len(outs) == len(items)
            for it, o in zip(items, outs):
                assert o.shape[0] == it.length, (r, it.key, tuple(o.shape), it.length)
                self.store[it.key] = o
            self.stats["encodes"] += len(items)
        rows = mm.rows[r]
        if not rows:
            return None
        idx, emb = [], []
        for m in rows:
            span = self.store.get(m.key)
            assert span is not None, (r, m.key, "rows of a key the cache does not hold")
            assert 0 <= m.lo and m.lo + m.n <= span.shape[0], (r, m, tuple(span.shape))
            idx.append(torch.arange(m.row0, m.row0 + m.n, dtype=torch.int32))
            emb.append(span[m.lo:m.lo + m.n])
        self.stats["rows"] += sum(m.n for m in rows)
        idx_t = torch.cat(idx)
        if self.device is not None:
            idx_t = idx_t.to(self.device, non_blocking=True)
        return idx_t, torch.cat(emb)
