"""MockEngine: the worker <-> engine contract (serve/interfaces.py) implemented without a model, with every input
invariant checked. Use it to develop against the contract on the CPU (worker, driver, the vLLM adapter, exactness checks):

    Worker(WorkerConfig(engine_cls="mock_engine.MockEngine"), Geometry(), backend="engine")

It returns what a real engine returns, with stub numbers: accepted tokens drawn from the real-traffic acceptance tail,
synthetic token ids, fake top-k logprobs for score rows, random raw logits (width MockEngine.vocab, kind "full")
for forced rows.
"""
from __future__ import annotations

import random

from geometry import SLIDING_WINDOW
from interfaces import PAD, StepResult
from worker import head_stub


class MockEngine:
    vocab = 64
    device_positions = True          # step_compact expands decode rows "on the device" (rowexp_ref, D13)
    max_tail_rows = 1536             # rows per rank through layers 21..39 + the head (the real engine: HSD's limit)

    def __init__(self, devices, model_dir=None, max_rows: int = 512, free_bytes_per_rank: int = 20_000_000_000, **kw):
        self.devices = ["cpu"] * len(devices)
        self.max_rows = max_rows
        self._free = free_bytes_per_rank
        self.pools = None
        self.steps = 0
        self.rows = 0

    def free_bytes(self) -> list:
        return [self._free] * len(self.devices)

    def bind_pools(self, pools: list) -> None:
        assert len(pools) == len(self.devices)
        self.pools = pools

    def step(self, batches: list, tables: list, ctx=None) -> list:
        assert self.pools is not None, "bind_pools() first"
        assert len(batches) == len(tables) == len(self.devices)
        self.steps += 1
        if ctx is not None:                                    # D11: lifecycle consistency
            scheduled = {rid for t in tables for rid, *_ in t.seq_rows}
            self.live = getattr(self, "live", set())
            for s in ctx.new:
                assert s.rid in scheduled, f"new sequence {s.rid} not scheduled in its first step"
                self.live.add(s.rid)
            self.live -= ctx.dropped
            for rid in scheduled:
                assert rid in self.live, f"{rid} scheduled without a SeqInit"
            for rid, g in ctx.grammar.items():
                assert rid in scheduled
            self._check_mm(batches, ctx)
        self._check_cp(batches)
        out = []
        for r, (sb, t) in enumerate(zip(batches, tables)):
            self._check(sb, t)
            res = head_stub(t, r, sb.step, 4)
            for k, (rid, row0, nr, kind, last) in enumerate(t.seq_rows):
                if int(t.seq["forced"][k]) and rid in res.logits:
                    pos, lg, vs = res.logits[rid]
                    res.logits[rid] = (pos, lg[:, : self.vocab], vs)
            out.append(res)
            self.rows += t.n
        return out

    def _check_mm(self, batches: list, ctx) -> None:
        """Image rows (StepContext.mm): every row whose position lies in an image span of its entry gets exactly one
        embedding row (idx = those rows, in row order) and no other row does; a step without image rows carries none."""
        from interfaces import image_rows

        mm = ctx.mm
        for r, sb in enumerate(batches):
            want = [row for row, _, _ in image_rows(sb)]
            got = mm[r] if mm is not None else None
            if not want:
                assert got is None or int(got[0].shape[0]) == 0, f"rank {r}: image embeddings in a step without image rows"
                continue
            assert got is not None, f"rank {r}: {len(want)} image rows without embeddings"
            idx, emb = got
            assert [int(x) for x in idx.tolist()] == want, f"rank {r}: image rows {idx.tolist()[:4]} != {want[:4]}"
            assert int(emb.shape[0]) == len(want), (r, tuple(emb.shape), len(want))
            self.mm_rows = getattr(self, "mm_rows", 0) + len(want)
            self.mm_last = getattr(self, "mm_last", {})
            self.mm_last[r] = (image_rows(sb), emb)

    def step_compact(self, batches: list, cts: list, ctx=None) -> list:
        """D13: the device side of async scheduling with the numpy reference (rowexp_ref): k_ring_ops + D12 deltas,
        k_expand, stub head results, k_advance."""
        import numpy as np

        from rowexp_ref import DeviceState

        assert self.pools is not None
        if not hasattr(self, "dstate"):
            self.dstate = [DeviceState(p.n_state, 3) for p in self.pools]
        self.steps += 1
        if ctx is not None:
            self._check_mm(batches, ctx)
        self._check_cp(batches)                                # CP under async: the same group checks
        self.last_rows = []
        out = []
        for r, (sb, ct) in enumerate(zip(batches, cts)):
            assert len(sb.tail_rows) <= self.max_tail_rows, (len(sb.tail_rows), self.max_tail_rows)
            st = self.dstate[r]
            st.apply(ct)
            rows = st.expand(ct)
            self.last_rows.append(rows)
            assert len(rows["positions"]) == ct.n == sb.n_rows <= self.max_rows
            res = StepResult(r, sb.step)
            acc = np.zeros(len(ct.slot), dtype=np.int64)
            for k, rid in enumerate(ct.rids):
                n = int(ct.n_rows[k])
                if int(ct.forced[k]):                          # raw logits of the forced rows (as step: D7)
                    r0 = int(ct.row0[k])
                    rr = range(r0, r0 + n) if int(ct.kind[k]) == 0 else ([r0 + n - 1] if int(ct.last_chunk[k]) else [])
                    if rr:
                        import torch

                        g = torch.Generator().manual_seed(abs(hash((str(rid), sb.step))) % (2 ** 31))
                        res.logits[rid] = ([int(rows["positions"][x]) for x in rr], torch.randn(len(rr), self.vocab,
                                                                                                  generator=g), 0)
                if int(ct.kind[k]) == 0:
                    if int(ct.forced[k]):
                        a = n                                  # forced decode: all rows accepted
                        res.tokens[rid] = [7] * n
                    else:
                        rng = random.Random(f"{rid}/{sb.step}")
                        u = rng.random()
                        a = 1 + sum(1 for p in (0.80, 0.60, 0.43)[: n - 1] if u < p)
                        res.tokens[rid] = [rng.randrange(1000, 120_000) for _ in range(a)]
                    res.accepted[rid] = a
                    acc[k] = a
                elif int(ct.last_chunk[k]):
                    res.tokens[rid] = [5]
            st.advance(ct, acc)
            out.append(res)
            self.rows += ct.n
        return out

    def _check_cp(self, batches: list) -> None:
        """CP prefill entries (docs/CP_PREFILL.md 4a): one group = 4 entries on 4 ranks with the same subs, each the
        sub-chunk subs[me]; contiguous, 64-aligned starts; the owner last with >= 128 rows, its tail + sample rows;
        helpers without tail / sample rows."""
        groups: dict = {}
        for sb in batches:
            tail, sample = set(sb.tail_rows), set(sb.sample_rows)
            for se in sb.seqs:
                if se.cp is None:
                    continue
                cp = se.cp
                rank, start, n = cp.subs[cp.me]
                assert (rank, start, n) == (sb.rank, se.start, se.n_rows), (cp.subs, cp.me, sb.rank, se.start, se.n_rows)
                rows = set(range(se.row0, se.row0 + se.n_rows))
                if cp.me == len(cp.subs) - 1:
                    assert sb.rank == cp.owner and se.n_rows >= SLIDING_WINDOW
                    assert set(range(se.row0 + se.n_rows - min(se.n_rows, SLIDING_WINDOW), se.row0 + se.n_rows)) <= tail
                else:
                    assert not (rows & tail) and not (rows & sample), "helper rows run layers 0..20 only"
                    assert se.cached_upto == 0 and se.rid == cp.rid
                groups.setdefault(cp.group, []).append(cp)
        for g, cps in groups.items():
            subs = cps[0].subs
            assert all(c.subs == subs for c in cps) and len(cps) == len(subs), (g, len(cps))
            assert len({r for r, _, _ in subs}) == len(subs)
            for (r0, s0, n0), (r1, s1, n1) in zip(subs, subs[1:]):
                assert s0 + n0 == s1 and s0 % 64 == 0 and n0 % 64 == 0, subs
            assert all(n >= SLIDING_WINDOW for _, _, n in subs), ("every sub-chunk >= 128 rows (X2 halo)", subs)
            assert subs[-1][0] == cps[0].owner

    def _check(self, sb, t) -> None:
        """The invariants an engine may rely on (and the worker guarantees)."""
        assert t.n == sb.n_rows <= self.max_rows, (t.n, self.max_rows)
        assert len(sb.tail_rows) <= self.max_tail_rows, (len(sb.tail_rows), self.max_tail_rows)
        if not t.n:
            return
        assert int(t.seq["n_rows"].sum()) == t.n
        last = t.si.gather(1, (t.sl.long() - 1).clamp(min=0)[:, None]).squeeze(1).long()
        assert bool((last == t.own_slot).all()), "window: own position last"
        assert bool((t.sl >= 1).all()) and bool((t.sl <= SLIDING_WINDOW).all())
        real = t.paged_slot[t.paged_slot != PAD]
        assert real.numel() == real.unique().numel(), "paged write slots unique"
        assert bool((t.seq["uid"] >= 0).all()), "every sequence has a uid"
        assert t.seq["uid"].unique().numel() == t.seq["uid"].numel()
        # D12: rebuild every sequence's block table from the deltas alone, compare with the live KV table
        self.bt = getattr(self, "bt", {})
        upd = {int(k): (int(s0), int(ln)) for k, s0, ln in zip(t.seq["bt_upd_seq"].tolist(), t.seq["bt_upd_start"].tolist(),
                                                              t.seq["bt_upd_len"].tolist())}
        pages = t.seq["bt_upd_pages"].tolist()
        off = 0
        offs = {}
        for k in t.seq["bt_upd_seq"].tolist():
            offs[k] = off
            off += upd[k][1]
        for k, se in enumerate(sb.seqs):
            key = (sb.rank, se.state_slot)
            if int(t.seq["bt_reset"][k]):
                self.bt[key] = []
            if k in upd:
                s0, ln = upd[k]
                assert len(self.bt[key]) == s0, "delta continues where the previous one ended"
                self.bt[key].extend(pages[offs[k]: offs[k] + ln])
            nb = int(t.seq["n_blocks"][k])
            assert self.bt[key][:nb] == list(se.block_table[:nb]), f"{se.rid}: block table from deltas differs"
            assert nb * 64 >= se.start + se.n_rows, "block table covers the rows"
            if se.kind == "decode":
                assert se.n_rows <= 4
        assert set(t.sample_idx.tolist()) <= set(range(t.n)) and set(t.tail_rows.tolist()) <= set(range(t.n))
        _ = random  # (deterministic stub numbers come from head_stub)
