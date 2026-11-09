"""CPU fake worker for the adapter tests: the worker protocol of PhysExecutor without a model.

Protocol (what PhysExecutor calls; the real worker, serve/worker.py, implements the same):
  geom                          Geometry of the pools (registered with the scheduler through adapter/bridge.py)
  step(ps, grammar) -> list     one StepResult per rank (serve/interfaces.py), for PhysStep `ps` (adapter/bridge.py):
                                  accepted[rid]  tokens appended by a decode group (1 .. k + 1)
                                  tokens[rid]    those token ids (decode), or [first token] for a final prefill chunk
                                  logprobs[rid]  optional: list per token of (ids [n + 1], logprobs [n + 1], rank)
                                  drafts[rid]    the next step's k drafts, for structured-output requests only (vLLM
                                                 validates them against the grammar before the next step)
                                grammar: rid -> GrammarRows (bitmask rows per position, number of acceptable drafts)
  warmup(), check_health(), shutdown()

Token semantics (deterministic, position-keyed, so preemption + recompute reproduces the same stream):
  random   output token at position t = hash(prompt, t) in [1000, vocab - 1000) (fake_token); never EOS
  script   PHYS_FAKE_SCRIPT = JSON {marker: reply text}: a prompt whose decoded text contains the marker gets the reply's
           token ids, then EOS. Unmatched prompts fall back to `random`. Used to drive the API layer (reasoning / tool
           parsers, DSML round trip, strict tool grammar) end to end on CPU.
Drafts: the k drafts of a decode group equal the true next tokens with the real-traffic acceptance tail (0.80 / 0.60 /
0.43), otherwise a wrong token; a grammar-masked target picks the first allowed token instead.
The worker checks every StepBatch it gets (row order, slots, sample rows, copies) against the scheduler's contract.

Images (PhysStep.mm, adapter/mm.py): every rank applies the step's drop / encode / gather through the real EncoderCache
with a fake encoder whose span rows derive from the image's processed bytes (fake_span), never from its key. Rows
taking span rows must carry the image token.
KV content model (PHYS_FAKE_KV=1): the fake keeps what each paged-pool slot holds (a token id, or a span row value for
an image row): rows write their paged slot, page copies run before the rows, a CP chunk's sub-chunks are exchanged as in
engine/cp.py (each rank then holds its own and the lower sub-chunks, the owner all of them), accepted decode rows write
theirs. Replay rows after a prefix hit must recompute what the cached page holds, and a request's first token is keyed
by the prompt content read back through its block table (prompt_key over it; equal to the token-only key for text).
A prefix cache that shared KV between different inputs therefore produces a different stream, or fails the replay
check.
"""
from __future__ import annotations

import json
import os
import random
import time
import zlib
from dataclasses import dataclass, field

from adapter import bridge  # noqa: F401
from adapter import mm as MM
from interfaces import StepResult  # noqa: E402  (serve/)

ACCEPT_TAIL = (0.80, 0.60, 0.43)


def _h(*parts) -> int:
    return zlib.crc32("/".join(map(str, parts)).encode())


def prompt_key(prompt_ids) -> int:
    return zlib.crc32(b"".join(int(t).to_bytes(4, "little") for t in prompt_ids))


def fake_token(seed: int, key: int, pos: int, vocab: int) -> int:
    """The random-mode fake model's token at position pos of a request whose prompt has key prompt_key(prompt)."""
    return 1000 + _h(seed, key, pos) % max(1, vocab - 2000)


def fake_span(t: dict) -> list:
    """The fake encoder's span rows of one image (vLLM's processed tensors): one value per span position, derived from
    the image's bytes (>= 2^31, so never a token id)."""
    import torch

    c = 0
    for k in MM.ITEM_KEYS:
        x = t[k].detach().cpu().contiguous()
        c = zlib.crc32(x.view(torch.uint8).numpy().tobytes() if x.numel() else b"", c)
    return [0x80000000 | (_h("img", c, j) & 0x7FFFFFFF) for j in range(int(t["types"].numel()))]


def fake_encode(items) -> list:
    import torch

    return [torch.tensor(fake_span(it.data), dtype=torch.int64) for it in items]


@dataclass
class _Req:
    init: object
    script: list | None = None
    drafts: list = field(default_factory=list)       # drafts for the next decode group (positions start+1 ..)
    key: int = 0
    pos: int | None = None                           # "device state": the real start of the next decode group
    last: int | None = None                          # the last emitted token (row 0 of the next decode group)


class FakeHandle:
    """engine/d13.py StepHandle's contract: result() -> [StepResult per rank] (blocks until the simulated GPU step has
    ended), done(), list-like. Thread-safe (the executor's pool thread waits on it while the main thread dispatches)."""

    def __init__(self, results, ready_at: float):
        self._res, self.ready_at = results, ready_at

    def done(self) -> bool:
        return time.perf_counter() >= self.ready_at

    def result(self) -> list:
        dt = self.ready_at - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        return self._res

    def __iter__(self):
        return iter(self.result())

    def __len__(self):
        return len(self.result())

    def __getitem__(self, i):
        return self.result()[i]


class FakeWorker:
    # emulates the D13 engine: decode positions resolved from its own state (the host's are an upper bound under async
    # scheduling); signals when it consumed the StepBatch (its KV table references) before the simulated GPU time.
    # PHYS_FAKE_HANDLES=1: the dispatch model of engine/d13.py instead: step() queues this step
    # on the (simulated) GPU, waits for the PREVIOUS step and returns a StepHandle; steps with prefill / forced rows
    # run synchronously inside dispatch (their handle comes back done), as in the real engine.
    device_positions = True
    signals_prepared = True
    supports_images = True

    def __init__(self, geom, vocab_size: int, eos_id: int | None, k: int, tokenizer=None, seed: int = 0):
        self.geom, self.vocab, self.eos, self.k, self.tok, self.seed = geom, vocab_size, eos_id, k, tokenizer, seed
        self.reqs: dict = {}
        self.history: dict = {}      # rid -> _Req, kept after the request ends (tests: expected token streams)
        self.steps = 0
        self.rows_total = 0
        self.scripts = {}
        path = os.environ.get("PHYS_FAKE_SCRIPT")
        if path:
            with open(path) as f:
                self.scripts = json.load(f)
        self.log: list = []          # (step, rid, kind, start, n, tokens) for the tests
        self.keep_log = os.environ.get("PHYS_FAKE_LOG", "0") == "1"
        self.step_s = float(os.environ.get("PHYS_FAKE_STEP_MS", "0")) / 1e3   # simulated GPU time per step
        self.dev_groups = 0          # async: decode groups whose positions the host left to the device
        self.handles = os.environ.get("PHYS_FAKE_HANDLES", "0") == "1"
        self.main_thread_dispatch = self.handles      # PhysExecutor: call step() on the engine loop's thread
        self._gpus_idle_at = 0.0                          # handles: when the simulated GPU finishes the last queued step
        self._prev_end = 0.0
        self.sync_steps = 0
        self.cp_entries = 0                           # CP prefill entries seen (owner + helpers)
        self.mm_caches = [MM.EncoderCache(r, fake_encode) for r in range(geom.n_ranks)]
        self.mm_rows = 0                              # rows that took span rows
        self.mm_split = 0                             # image slices that start inside their span (cut by a boundary)
        self.kv_model = os.environ.get("PHYS_FAKE_KV", "0") == "1"
        self.pages: dict = {}                         # (rank, paged page) -> [content of slot 0 .. 63] (KV model)
        self.replay_checked = 0
        self.kv_keys: dict = {}                       # rid -> the content its first token was keyed by (KV model)

    # ------------------------------------------------------------------------------------------ protocol
    def warmup(self) -> None:
        return None

    def check_health(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def step(self, ps, grammar: dict | None) -> list:
        self.steps += 1
        for rid in ps.finished | ps.preempted:
            self.reqs.pop(rid, None)
        for ri in ps.new_reqs:
            self.reqs[ri.rid] = _Req(ri, self._script_for(ri), key=prompt_key(ri.prompt_token_ids))
            self.history[ri.rid] = self.reqs[ri.rid]
        img = self._images(ps)
        if self.kv_model:
            self._kv_rows(ps, img)
        out = []
        busy = self.step_s > 0 and any(sb.n_rows for sb in ps.batches)
        for r, sb in enumerate(ps.batches):
            self._check_batch(ps, r, sb)
            res = StepResult(r, sb.step)
            res.drafts = {}
            toks = ps.row_tokens[r]
            for se in sb.seqs:
                st = self.reqs[se.rid]
                g = grammar.get(se.rid) if grammar else None
                if se.kind == "decode":
                    forced = toks[se.row0 + 1:se.row0 + se.n_rows]
                    if forced and all(t >= 0 for t in forced):      # host-given drafts (FORCE hook) override ours
                        st.drafts = list(forced)
                    start0 = st.pos if se.start < 0 else se.start
                    t_in = toks[se.row0] if toks[se.row0] >= 0 else st.last
                    self._decode(st, se, toks[se.row0], g, res, ps.num_spec.get(se.rid, se.n_rows - 1))
                    if self.kv_model:
                        self._kv_decode(r, se, start0, t_in, res.tokens[se.rid])
                    if getattr(se, "forced", False):
                        self._raw_logits(st, se, res)
                elif se.last_chunk:
                    if getattr(se, "forced", False):
                        self._raw_logits(st, se, res, last_only=True)
                    if self.kv_model:
                        st.key = self._kv_key(r, se, st)
                    t = self._target(st, se.start + se.n_rows, g, 0)
                    st.pos, st.last = se.start + se.n_rows, t
                    res.tokens[se.rid] = [t]
                    self._logprobs(st, res, se.rid, [t])
                    self._new_drafts(st, se.start + se.n_rows, res, se.rid)
                    if self.keep_log:
                        self.log.append((sb.step, se.rid, "prefill_last", se.start, se.n_rows, [t]))
                elif self.keep_log:
                    self.log.append((sb.step, se.rid, "prefill", se.start, se.n_rows, []))
            self.rows_total += sb.n_rows
            out.append(res)
        if self.handles:
            return self._dispatch(ps, out, any(sb.n_rows for sb in ps.batches))
        ev = getattr(ps, "prepared", None)
        if ev is not None:
            ev.set()                                  # host-side tables consumed
        if busy:
            time.sleep(self.step_s)                   # the GPU (releases the GIL: the engine loop plans meanwhile)
        return out

    def _dispatch(self, ps, out, has_rows: bool):
        now = time.perf_counter()
        if not has_rows:
            return out                                # frees only: nothing queued on the GPU
        start = max(now, self._gpus_idle_at)
        end = start + self.step_s
        prev, self._gpus_idle_at, self._prev_end = self._prev_end, end, end
        sync = any(se.kind != "decode" or getattr(se, "forced", False) for sb in ps.batches for se in sb.seqs)
        wait_until = end if sync else prev           # dispatch(N+1) waits for N; sync steps for themselves
        if sync:
            self.sync_steps += 1
        dt = wait_until - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        return FakeHandle(out, end)

    def expected(self, prompt_ids, n_out: int) -> list:
        """The first n_out output tokens of the random-mode fake model for this prompt (no grammar)."""
        k, n = prompt_key(prompt_ids), len(prompt_ids)
        return [fake_token(self.seed, k, n + j, self.vocab) for j in range(n_out)]

    # ------------------------------------------------------------------------------------------ images + KV content
    def _images(self, ps) -> list:
        """Per rank: row -> span row value of this step's image rows (the real EncoderCache, the fake encoder)."""
        out = []
        mm = getattr(ps, "mm", None)
        for r, sb in enumerate(ps.batches):
            got = self.mm_caches[r].apply(mm) if mm is not None else None
            if mm is not None:
                self.mm_split += sum(1 for m in mm.rows[r] if m.lo > 0)
            m = {}
            if got is not None:
                idx, emb = got
                toks = ps.row_tokens[r]
                for i, v in zip(idx.tolist(), emb.tolist()):
                    assert 0 <= i < sb.n_rows, (r, i, sb.n_rows)
                    assert i not in m, (r, i, "row covered by two image slices")
                    assert toks[i] == MM.IMAGE_TOKEN_ID, (r, i, toks[i], "image embedding on a non-image row")
                    m[i] = int(v)
                self.mm_rows += len(m)
            out.append(m)
        return out

    def _page(self, r: int, page: int) -> list:
        p = self.pages.get((r, page))
        if p is None:
            p = self.pages[(r, page)] = [None] * self.geom.page_tokens
        return p

    def _kv_rows(self, ps, img: list) -> None:
        bt = self.geom.page_tokens
        plan = getattr(ps, "plan", None)
        if plan is not None:
            copies = [c for c in plan.copies if c.pool == "paged"]
        else:
            copies = [c for sb in ps.batches for c in sb.copies if c.pool == "paged" and c.dst_rank == sb.rank]
        for c in copies:                                        # page copies run before the step's kernels
            self.pages[(c.dst_rank, c.dst_page)] = list(self._page(c.src_rank, c.src_page))
        groups: dict = {}
        for r, sb in enumerate(ps.batches):
            toks = ps.row_tokens[r]
            for se in sb.seqs:
                if se.kind == "decode" or se.start < 0:
                    continue
                cp = getattr(se, "cp", None)
                g = groups.setdefault(cp.group, ({}, [])) if cp is not None else None
                for i in range(se.n_rows):
                    row, pos = se.row0 + i, se.start + i
                    v = img[r].get(row, toks[row])
                    slot = sb.paged_slot[row]
                    if slot >= 0:
                        self._page(r, slot // bt)[slot % bt] = v
                    elif pos < se.cached_upto:                  # replay row over a cached page: must recompute it
                        have = self._page(r, se.block_table[pos // bt])[pos % bt]
                        assert have is None or have == v, (
                            se.rid, pos, have, v, "replay row differs from the cached page (KV shared between inputs)")
                        self.replay_checked += 1
                    if g is not None:
                        g[0][pos] = v
                if g is not None:
                    g[1].append((r, se))
        for vals, ents in groups.values():                      # engine/cp.py: own + lower sub-chunks, owner all
            for r, se in ents:
                cp = se.cp
                lo = cp.subs[0][1]
                hi = cp.subs[-1][1] + cp.subs[-1][2] if r == cp.owner else cp.subs[cp.me][1] + cp.subs[cp.me][2]
                for pos in range(lo, hi):
                    self._page(r, se.block_table[pos // bt])[pos % bt] = vals[pos]

    def _kv_decode(self, r: int, se, start: int, t_in: int, out: list) -> None:
        bt = self.geom.page_tokens
        ins = [t_in] + list(out[:-1])                           # the accepted rows' input tokens
        for j, v in enumerate(ins):
            pos = start + j
            self._page(r, se.block_table[pos // bt])[pos % bt] = v

    def _kv_key(self, r: int, se, st) -> int:
        """prompt_key over the prompt content this request's block table holds after its last prefill chunk."""
        bt = self.geom.page_tokens
        n = len(st.init.prompt_token_ids)
        got = []
        for pos in range(n):
            v = self._page(r, se.block_table[pos // bt])[pos % bt]
            assert v is not None, (se.rid, pos, "no KV content at a prompt position")
            got.append(v)
        self.kv_keys[se.rid] = got
        return prompt_key(got)

    # ------------------------------------------------------------------------------------------ tokens
    def _script_for(self, ri):
        if not self.scripts or self.tok is None:
            return None
        text = self.tok.decode(ri.prompt_token_ids[-4096:])
        for marker, reply in self.scripts.items():
            if marker in text:
                ids = self.tok.encode(reply, add_special_tokens=False)
                return list(ids) + ([self.eos] if self.eos is not None else [])
        return None

    def _true(self, st: _Req, pos: int) -> int:
        """The 'model' token at position pos (pos >= prompt length)."""
        n_prompt = len(st.init.prompt_token_ids)
        if st.script is not None:
            j = pos - n_prompt
            if 0 <= j < len(st.script):
                return st.script[j]
            return self.eos if self.eos is not None else 0
        return fake_token(self.seed, st.key, pos, self.vocab)

    def _target(self, st: _Req, pos: int, g, i: int) -> int:
        t = self._true(st, pos)
        if g is None:
            return t
        row = g.rows[i]
        if (int(row[t >> 5]) >> (t & 31)) & 1:
            return t
        for w, word in enumerate(row):                      # first allowed token
            word = int(word) & 0xFFFFFFFF
            if word:
                return w * 32 + ((word & -word).bit_length() - 1)
        raise AssertionError(f"{st.init.rid}: the grammar allows no token at row {i}")

    def _new_drafts(self, st: _Req, next_pos: int, res, rid) -> None:
        """Drafts for the decode group that starts at next_pos (its row 0 is the token at next_pos)."""
        rng = random.Random(_h(self.seed, rid, "draft", next_pos))
        u = rng.random()
        n_ok = sum(1 for p in ACCEPT_TAIL[: self.k] if u < p)
        d = []
        for j in range(self.k):
            t = self._true(st, next_pos + 1 + j)
            if j >= n_ok:
                t = 1000 + (t + 7 + j) % max(1, self.vocab - 2000)
            d.append(t)
        st.drafts = d
        if st.init.structured:
            res.drafts[rid] = list(d)

    def _decode(self, st: _Req, se, row0_tok: int, g, res, num_spec: int) -> None:
        if se.start < 0:                                    # async: resolved on the "device" from the fake's own state
            assert st.pos is not None and st.pos >= se.known, (se.rid, st.pos, se.known)
            start = st.pos
            self.dev_groups += 1
        else:
            start = se.start
            assert st.pos is None or st.pos == start, (se.rid, start, st.pos, "host start != device position")
        forced = bool(getattr(se, "forced", False))
        if row0_tok >= 0 and st.last is not None and not forced:
            assert row0_tok == st.last, (se.rid, row0_tok, st.last, "row 0 token != the last emitted one")
        acceptable = num_spec if g is None else min(num_spec, g.acceptable)
        out = []
        if forced:                                            # D8: drafts := the forced rows, every row accepted
            out = list(st.drafts[:num_spec]) + [self._target(st, start + 1 + num_spec, g, num_spec)]
        for i in range(0 if forced else num_spec + 1):
            t = self._target(st, start + 1 + i, g, i)
            out.append(t)
            if i >= acceptable or i >= len(st.drafts) or st.drafts[i] != t:
                break
        res.accepted[se.rid] = len(out)
        res.tokens[se.rid] = out
        st.pos, st.last = start + len(out), out[-1]
        self._logprobs(st, res, se.rid, out)
        self._new_drafts(st, start + len(out), res, se.rid)
        if self.keep_log:
            self.log.append((res.step, se.rid, "decode", start, se.n_rows, out))

    def _raw_logits(self, st: _Req, se, res, last_only: bool = False) -> None:
        """Forced sequences: the raw logits of the rows (full vocab; the fake model's token gets the mass)."""
        import torch

        rows = [se.n_rows - 1] if last_only else list(range(se.n_rows))
        pos = [se.start + i for i in rows]
        t = torch.zeros(len(rows), self.vocab)
        for j, p in enumerate(pos):
            t[j, self._true(st, p + 1)] = 10.0
        res.logits[se.rid] = (pos, t, 0)

    def _logprobs(self, st: _Req, res, rid, toks) -> None:
        n = st.init.sampling_params.logprobs if st.init.sampling_params is not None else None
        if n is None:
            return
        rows = []
        for t in toks:
            alts = [1000 + (t + 13 * (j + 1)) % max(1, self.vocab - 2000) for j in range(max(n, 0))]
            rows.append(([t] + alts, [-0.1] + [-1.0 - j for j in range(len(alts))], 1))
        res.logprobs[rid] = rows

    # ------------------------------------------------------------------------------------------ contract checks
    def _check_batch(self, ps, r: int, sb) -> None:
        assert sb.rank == r, (sb.rank, r)
        assert len(ps.row_tokens[r]) == sb.n_rows == len(sb.positions) == len(sb.seq_slot) == len(sb.paged_slot) \
            == len(sb.window_slot), (r, sb.n_rows)
        seen_prefill = False
        bt, wt = self.geom.page_tokens, self.geom.window_page_tokens
        for slot, se in enumerate(sb.seqs):
            assert se.rid in self.reqs, f"rank {r}: {se.rid} scheduled without worker state (no ReqInit)"
            if se.kind == "decode":
                assert not seen_prefill, "decode rows come before prefill rows"
                drafts = ps.row_tokens[r][se.row0 + 1:se.row0 + se.n_rows]
                assert ps.row_tokens[r][se.row0] >= 0 or self.reqs[se.rid].pos is not None, (se.rid, "row 0 unknown")
                if all(t >= 0 for t in drafts) and drafts and not getattr(se, "forced", False):
                    # host-given ids of a structured request resuming after an ASYNC hold (PhysScheduler): the engine's
                    # own drafts of its last step, and row 0 its last token (the real engine: host ids win)
                    st_ = self.reqs[se.rid]
                    assert self.reqs[se.rid].init.structured, (se.rid, "host-given drafts on a non-structured request")
                    assert drafts == st_.drafts[:len(drafts)], (se.rid, "host-given drafts differ from the engine's")
                    assert st_.last is None or ps.row_tokens[r][se.row0] == st_.last, (se.rid, "host-given row 0")
                else:
                    assert all(t == -1 for t in drafts) or (getattr(se, "forced", False) and all(t >= 0 for t in drafts))
            else:
                seen_prefill = True
                got = ps.row_tokens[r][se.row0:se.row0 + se.n_rows]
                assert all(t >= 0 for t in got)
                ri = self.reqs[se.rid].init                  # prefill rows (CP sub-chunks too) = the sequence's ids
                ids = list(ri.prompt_token_ids) + list(ri.output_token_ids)
                if se.start + se.n_rows <= len(ids):
                    assert got == ids[se.start:se.start + se.n_rows], (se.rid, se.start, se.n_rows, "prefill row tokens")
                if getattr(se, "cp", None) is not None:
                    self.cp_entries += 1
            if se.start < 0:                          # device rows: positions / slots are PAD (D13)
                assert se.kind == "decode" and all(sb.positions[se.row0 + i] == -1 for i in range(se.n_rows))
                continue
            for i in range(se.n_rows):
                row = se.row0 + i
                pos = se.start + i
                assert sb.positions[row] == pos and sb.seq_slot[row] == slot
                exp = -1 if pos < se.cached_upto else se.block_table[pos // bt] * bt + pos % bt
                assert sb.paged_slot[row] == exp, (se.rid, pos, sb.paged_slot[row], exp)
                assert sb.window_slot[row] == se.window_table[pos // wt] * wt + pos % wt
        assert sb.n_rows == sum(se.n_rows for se in sb.seqs)
