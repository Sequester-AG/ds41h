"""Serving worker of the ds41h-phys engine: pools, step tables, and the model backend.

One process drives the 4 GPUs (the fork's collectives, kernels/sp_layer/comm.cu, use peer pointers inside one process).
Per step: the host side (scheduler.py) hands one StepBatch per rank (interfaces.py); the worker
  1. runs the planned rank-to-rank page copies,
  2. turns each StepBatch into device tables (TableBuilder: positions, FlashMLA window / compressed indices, insert slots),
  3. runs the 40-layer step on every rank (backend), and
  4. returns one StepResult per rank.

Backends:
  StubBackend    -- CPU, no model: deterministic acceptance and token ids (driver tests, scheduler-in-the-loop on CPU).
  SPLayerBackend -- GPU: the fork's layer chain as kernels/sp_layer/sp_layer.py runs it (attention mHC boundary ->
                    attn_half -> FFN boundary (prescale) -> router -> one AG push -> routed MoE (route2r, gather5, GEMM1/2)
                    + shared expert on a side stream -> combine pushed to the owners -> RS sum), but CHAINED for real:
                    layer L+1's input is layer L's output (moe_out, xm2, f_post, f_comb, f_pre), layer 0's is the
                    embedding in 4 hyper-connection copies (post 0, comb I, lagged pre = one-hot copy 0, as drafter.py).

STUBS (pieces other modules of the fork provide; each is one function here so it can be swapped):
  indexer_stub   top-512 = the most recent 512 complete compressed positions of the row's own sequence (kernels/indexer2f
                 replaces it); no compressor writes (the compressed records of new positions are not produced).
  head_stub      no LM head / sampler / DSpark (kernels/head replaces it): acceptance drawn from the real-traffic tail,
                 token ids synthetic.
  engram_stub    layers 1 and 14 run without the Engram n-gram embedding (kernels/engram replaces it).
  prefill MoE    prefill rows go through the decode MoE kernels (kernels/moe_prefill replaces it above ~2K rows).
  window-only    layers 0 / 1 (ratio 0) attend window + an empty compressed list (tl = 0).
"""
from __future__ import annotations

import dataclasses
import os
import random
import time
from dataclasses import dataclass, field

from geometry import (KV_RECORD_B, N_LAYERS, PAGE_BYTES, PAGE_OFFSETS, PAGE_TOKENS, RECORDS_PER_PAGE, SLIDING_WINDOW,
                      STATE_SLOT_BYTES, WINDOW_LAYER_PAGE_B, WINDOW_PAGE_BYTES, WINDOW_PAGE_TOKENS, KV_SOURCE_RATIO,
                      Geometry, kv_source_of)
from interfaces import PAD, StepBatch, StepResult, copy_runs

HC, D, TOPK, NE = 4, 5120, 6, 384
ACCEPT_TAIL = (0.80, 0.60, 0.43)          # P(>= j drafts accepted), real traffic at k = 3 (mean 2.83 tokens per step)
STUB_VOCAB = 64                           # the stub's fake "full vocab" width for forced logits (tests only)


@dataclass
class WorkerConfig:
    model: str = os.environ.get("DS41H_MODEL_DIR", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))
    devices: tuple = (0, 1, 2, 3)
    layers: tuple = tuple(range(N_LAYERS))
    weights: str = "reuse:21"              # "all": every layer's own experts + attention; "reuse:L": layer L's for all
                                           # layers (as sp_layer --step-layers; mHC / router / shared are always per layer)
    reserve_bytes: int = 8_000_000_000     # left free after the pools (activations, workspaces, graphs, allocator slack)
    size_as_full_model: bool = True        # reuse mode: size the pools as if all 40 layers' weights were resident
    max_kv_bytes: int = 0                  # > 0: cap the KV pools (tests)
    w8a16: str = "q_b"
    insert: bool = True                    # attention writes the rows' own window KV (real layer)
    ncta: int = 16
    push_ctas: int = 132
    seed: int = 0
    engine_cls: str = "engine.engine.Engine"   # backend="engine": qualname of the engine class;
                                           # "mock_engine.MockEngine" = the contract without a model (CPU)
    engine_kw: dict = field(default_factory=dict)   # extra Engine kwargs (indexer / engram / head / ...)
    async_mode: bool = False               # D13: compact tables (decode rows expanded on the device), engine.step_compact
    compact: bool = False                  # also the synchronous loop sends CompactTables (all starts from the host) to
                                           # engine.step_compact: O(sequences) host tables instead of O(rows x 128)


def size_pools(free_bytes: int, cfg: WorkerConfig, geom: Geometry, extra_weight_bytes: int = 0) -> Geometry:
    """Geometry whose pools fill what is free on the device after the weights (minus reserve, minus the weights that a
    reuse-mode run does not load but the real engine would)."""
    kv = int(free_bytes - cfg.reserve_bytes - extra_weight_bytes)
    if cfg.max_kv_bytes:
        kv = min(kv, cfg.max_kv_bytes)
    assert kv > geom.window_pool_bytes + geom.state_pool_slots * STATE_SLOT_BYTES + 100 * PAGE_BYTES, \
        f"not enough memory for KV pools ({kv / 1e9:.1f} GB)"
    return dataclasses.replace(geom, kv_bytes_per_rank=kv)


# ---------------------------------------------------------------------------------------------- pools
class Pools:
    """The KV tensors of one rank, in the layout of geometry.py. One extra window page at the end is the scratch page
    for padded rows (never handed to a sequence)."""

    def __init__(self, geom: Geometry, device):
        import torch

        p, w, s = geom.pools()
        self.n_paged, self.n_window, self.n_state = p.num_pages, w.num_pages, s.num_pages
        self.device = torch.device(device)
        u8 = torch.uint8
        self.paged = torch.empty(self.n_paged * PAGE_BYTES, dtype=u8, device=self.device)
        self.window = torch.empty((self.n_window + 1) * WINDOW_PAGE_BYTES, dtype=u8, device=self.device)
        self.state = torch.zeros(self.n_state * STATE_SLOT_BYTES, dtype=u8, device=self.device)
        self.scratch_window_page = self.n_window

    @property
    def nbytes(self) -> int:
        return self.paged.numel() + self.window.numel() + self.state.numel()

    def comp_base(self, src: int):
        """1-D view starting at source layer `src`'s compressed records (page stride PAGE_BYTES)."""
        return self.paged[PAGE_OFFSETS[f"comp{src}"]:]

    def comp_view(self, src: int):
        """FlashMLA k_cache view of source `src`'s compressed records: [pages, records per page, 1, 584]."""
        import torch

        return torch.as_strided(self.paged, (self.n_paged, RECORDS_PER_PAGE[src], 1, KV_RECORD_B),
                                (PAGE_BYTES, KV_RECORD_B, KV_RECORD_B, 1), PAGE_OFFSETS[f"comp{src}"])

    def window_base(self, layer: int):
        return self.window[layer * WINDOW_LAYER_PAGE_B:]

    def window_view(self, layer: int):
        import torch

        return torch.as_strided(self.window, (self.n_window + 1, WINDOW_PAGE_TOKENS, 1, KV_RECORD_B),
                                (WINDOW_PAGE_BYTES, KV_RECORD_B, KV_RECORD_B, 1), layer * WINDOW_LAYER_PAGE_B)


# ---------------------------------------------------------------------------------------------- tables
def comp_slot(block_table: list, src: int, j: int) -> int:
    """Slot (k_comp slot space of source `src`) of compressed position j of a sequence."""
    c = KV_SOURCE_RATIO[src]
    tok = j * c
    return block_table[tok // PAGE_TOKENS] * RECORDS_PER_PAGE[src] + (tok % PAGE_TOKENS) // c


def indexer_stub(se, pos: int, src: int, topk: int = 512) -> list:
    """STUB for kernels/indexer2f: the most recent `topk` complete compressed positions before `pos` (ascending)."""
    c = KV_SOURCE_RATIO[src]
    n = (pos + 1) // c if c > 1 else pos           # ratio 1: positions < pos (the row itself is in the window)
    lo = max(0, n - topk)
    return [comp_slot(se.block_table, src, j) for j in range(lo, n)]


@dataclass
class RankTables:
    """Inputs of one rank for one step (CPU tensors; to_device() moves them). Rows in StepBatch order."""
    n: int
    positions: object            # [n] int64
    tokens: object               # [n] int64
    own_slot: object             # [n] int64 window slot of the row's own position
    si: object                   # [n, 128] int32 window slots, ascending, own position last, -1 padded
    sl: object                   # [n] int32
    ti: dict                     # src -> [n, 512] int32 compressed slots (indexer stub), -1 padded
    tl: dict                     # src -> [n] int32
    sample_rows: list
    seq_rows: list               # (rid, row0, n_rows, kind, last_chunk)
    # engine-facing (indexer / compressor / head / per-sequence device state), all CPU tensors:
    seq_slot: object = None      # [n] int32 row -> sequence index in this batch
    paged_slot: object = None    # [n] int64 token-level paged write slot (page*64 + pos%64), -1 = cached / replay row
    tail_rows: object = None     # [m] int64 rows that run layers 21..39
    sample_idx: object = None    # [s] int64 rows whose logits are sampled / returned
    seq: dict = field(default_factory=dict)   # per sequence [S]: state_slot, start, n_rows, kind (0 decode, 1 prefill),
                                              # last_chunk, score, forced, replay_start, cached_upto, n_blocks;
                                              # block_table [S, B]
    targets: object = None       # [n] int64 the token each row predicts (forced / score sequences; -1 unknown)
    tensors: dict = field(default_factory=dict)

    def to_device(self, device, scratch_slot: int):
        import torch

        if self.n == 0:                                   # an idle rank still joins the collectives: one dummy row
            t = dict(positions=torch.zeros(1, dtype=torch.int64), tokens=torch.zeros(1, dtype=torch.int64),
                     own_slot=torch.full((1,), scratch_slot, dtype=torch.int64),
                     si=torch.full((1, 1, SLIDING_WINDOW), -1, dtype=torch.int32), sl=torch.ones(1, dtype=torch.int32))
            t["si"][0, 0, 0] = scratch_slot
            for src in self.ti:
                t[f"ti{src}"] = torch.full((1, 1, 512), -1, dtype=torch.int32)
                t[f"tl{src}"] = torch.zeros(1, dtype=torch.int32)
        else:
            t = dict(positions=self.positions, tokens=self.tokens, own_slot=self.own_slot,
                     si=self.si.view(self.n, 1, SLIDING_WINDOW), sl=self.sl, seq_slot=self.seq_slot,
                     paged_slot=self.paged_slot, tail_rows=self.tail_rows, sample_idx=self.sample_idx,
                     targets=self.targets)
            for src in self.ti:
                t[f"ti{src}"], t[f"tl{src}"] = self.ti[src].view(self.n, 1, 512), self.tl[src]
            for k, v in self.seq.items():
                t[f"seq_{k}"] = v
        self.tensors = {k: v.contiguous().to(device) for k, v in t.items()}
        return self.tensors


class TableBuilder:
    """StepBatch -> RankTables, O(rows) and vectorized per rank (torch, CPU): per sequence only small Python slices of
    the window / compressed pages its rows actually read, then one set of tensor ops for all rows.

    Block tables go out as DELTAS (contract D12): a sequence's full table once per (rank, state slot, uid) -- first
    scheduling, after a rank move or a preemption -- then only the pages appended since; the engine keeps its own
    persistent per-slot device table. full_block_tables=True also materializes seq_block_table [S, maxB] (tests).
    stub_indexer=False skips the stub top-512 tables (an engine with the real indexer ignores them, D9)."""

    def __init__(self, sources=(2, 8, 14, 20), tokens_of=None, stub_indexer: bool = False, full_block_tables: bool = False):
        self.sources = sources
        self.tokens_of = tokens_of or (lambda rid, pos: (hash((str(rid), pos)) % 120_000) + 1000)
        self.stub_indexer = stub_indexer
        self.full_block_tables = full_block_tables
        self.sent = {}                                     # (rank, state_slot) -> (uid, blocks sent)
        self.dev_uid = {}                                  # D13: (rank, state_slot) -> uid initialized on the device
        self.win_sent = {}                                 # D13: (rank, state_slot) -> {window block: page} sent
        self.win_ver = {}                                  # D13: (rank, state_slot) -> SeqKV.wver of that last send
        self.cstate = {}                                   # build_compact: (rank, slot) -> [(uid, inc), blocks, wver, ring]

    def build(self, sb: StepBatch, row_tokens: list | None = None) -> RankTables:
        """row_tokens: the input token id of every row (the adapter's PhysStep.row_tokens: draft rows -1 = the
        engine's device-side drafts, >= 0 = forced); without it tokens come from tokens_of(rid, pos).
        numpy throughout (op overhead ~1 us vs ~5-10 us for small torch ops); torch.from_numpy at the end (no copy)."""
        import numpy as np
        import torch

        n = sb.n_rows
        WT, BT = WINDOW_PAGE_TOKENS, PAGE_TOKENS
        flat_w, rec = [], []
        flat_c = {s: [] for s in self.sources}
        coff = {s: [] for s in self.sources}
        clo = {s: [] for s in self.sources}
        upd, upd_pages = [], []
        tokens = [] if row_tokens is None else None
        stub, sent, rank = self.stub_indexer, self.sent, sb.rank
        for k, se in enumerate(sb.seqs):
            first, nr = se.start, se.n_rows
            last = first + nr - 1
            rs = se.replay_start
            lo_b = max(rs, first - (SLIDING_WINDOW - 1)) // WT
            wtab = se.window_table
            woff = len(flat_w)
            flat_w.extend([wtab.get(b, -1) for b in range(lo_b, last // WT + 1)])
            bt = se.block_table
            nb = se.n_blocks or len(bt)
            if stub:
                for s in self.sources:
                    c = KV_SOURCE_RATIO[s]
                    nc0 = (first + 1) // c if c > 1 else first
                    nc1 = (last + 1) // c if c > 1 else last
                    lo_tok = max(0, nc0 - 512) * c
                    hi_tok = max(0, nc1 - 1) * c
                    coff[s].append(len(flat_c[s]))
                    clo[s].append(lo_tok // BT)
                    flat_c[s].extend(bt[lo_tok // BT: hi_tok // BT + 1])
            key = (rank, se.state_slot)
            uid = se.uid
            ident = (uid, se.inc)                          # not the uid alone: uids and slots recycle (FIFO / lowest)
            prev = sent.get(key)
            reset = prev is None or prev[0] != ident or prev[1] > nb
            s0 = 0 if reset else prev[1]
            if nb > s0 or reset:
                upd.append((k, s0, nb - s0))
                upd_pages.extend(bt[s0:nb])
            sent[key] = (ident, nb)
            rec.append((se.state_slot, uid, first, nr, 0 if se.kind == "decode" else 1, se.last_chunk, se.score, se.forced,
                        rs, se.cached_upto, nb, reset, woff, lo_b))
            if tokens is not None:
                tok_of, rid = self.tokens_of, se.rid
                tokens.extend([tok_of(rid, p) for p in range(first, last + 1)])
        if row_tokens is not None:
            assert len(row_tokens) == n, (len(row_tokens), n)
            tokens = row_tokens
        tok = np.asarray(tokens if n else [0], dtype=np.int64)[:n]
        targets = np.empty(n, dtype=np.int64)
        if n:
            targets[:-1] = tok[1:]
        R = np.asarray(rec, dtype=np.int64).reshape(-1, 14)
        for k, se in enumerate(sb.seqs):                    # the last row of each sequence predicts beyond its rows
            targets[se.row0 + se.n_rows - 1] = (self.tokens_of(se.rid, se.start + se.n_rows) if row_tokens is None
                                                else -1)
        pos = np.asarray(sb.positions, dtype=np.int64)
        seq_of = np.asarray(sb.seq_slot, dtype=np.int64)
        si = np.full((n, SLIDING_WINDOW), -1, dtype=np.int32)
        sl = np.zeros(n, dtype=np.int32)
        ti, tl = {}, {}
        if n:
            fw = np.asarray(flat_w + [-1], dtype=np.int64)
            lo = np.maximum(R[seq_of, 8], pos - (SLIDING_WINDOW - 1))
            idx = lo[:, None] + np.arange(SLIDING_WINDOW, dtype=np.int64)
            ok = idx <= pos[:, None]
            idc = np.minimum(idx, pos[:, None])
            page = fw[R[seq_of, 12][:, None] + idc // WT - R[seq_of, 13][:, None]]
            slots = page * WT + idc % WT
            assert (slots[ok] >= 0).all(), "window page missing"
            si = np.where(ok, slots, -1).astype(np.int32)
            sl = (pos - lo + 1).astype(np.int32)
            ar = np.arange(512, dtype=np.int64)
            for s in (self.sources if stub else ()):
                c, rpp = KV_SOURCE_RATIO[s], RECORDS_PER_PAGE[s]
                fc = np.asarray(flat_c[s] + [0], dtype=np.int64)
                nc = (pos + 1) // c if c > 1 else pos
                lo_c = np.maximum(nc - 512, 0)
                j = lo_c[:, None] + ar
                okc = j < nc[:, None]
                tk = np.where(okc, j * c, 0)
                blk = np.where(okc, tk // BT - np.asarray(clo[s], dtype=np.int64)[seq_of][:, None], 0)
                pg = fc[np.asarray(coff[s], dtype=np.int64)[seq_of][:, None] + blk]
                ti[s] = torch.from_numpy(np.where(okc, pg * rpp + (tk % BT) // c, -1).astype(np.int32))
                tl[s] = torch.from_numpy((nc - lo_c).astype(np.int32))
        elif stub:                                         # an idle rank: empty tables, so to_device pads every source
            for s in self.sources:
                ti[s], tl[s] = torch.zeros((0, 512), dtype=torch.int32), torch.zeros(0, dtype=torch.int32)
        T = lambda a, dt=np.int32: torch.from_numpy(np.ascontiguousarray(a, dtype=dt))  # noqa: E731
        U = np.asarray(upd, dtype=np.int64).reshape(-1, 3)
        Rs = R if len(R) else np.zeros((1, 14), dtype=np.int64)
        seq = dict(state_slot=T(Rs[:, 0]), uid=T(Rs[:, 1]), start=T(Rs[:, 2], np.int64), n_rows=T(Rs[:, 3]),
                   kind=T(Rs[:, 4]), last_chunk=T(Rs[:, 5]), score=T(Rs[:, 6]), forced=T(Rs[:, 7]),
                   replay_start=T(Rs[:, 8], np.int64), cached_upto=T(Rs[:, 9], np.int64), n_blocks=T(Rs[:, 10]),
                   bt_reset=T(Rs[:, 11]), bt_upd_seq=T(U[:, 0]), bt_upd_start=T(U[:, 1]), bt_upd_len=T(U[:, 2]),
                   bt_upd_pages=T(np.asarray(upd_pages, dtype=np.int64)))
        if self.full_block_tables:
            nbm = max([len(se.block_table) for se in sb.seqs] or [1])
            btab = np.full((max(len(sb.seqs), 1), nbm), -1, dtype=np.int32)
            for k, se in enumerate(sb.seqs):
                btab[k, : len(se.block_table)] = se.block_table
            seq["block_table"] = torch.from_numpy(btab)
        return RankTables(n, torch.from_numpy(pos), torch.from_numpy(tok), T(sb.window_slot, np.int64), torch.from_numpy(si),
                          torch.from_numpy(sl), ti, tl, list(sb.sample_rows),
                          [(se.rid, se.row0, se.n_rows, se.kind, se.last_chunk) for se in sb.seqs],
                          seq_slot=T(seq_of), paged_slot=T(sb.paged_slot, np.int64), tail_rows=T(sb.tail_rows, np.int64),
                          sample_idx=T(sb.sample_rows, np.int64), seq=seq, targets=torch.from_numpy(targets))


def _sub_batch(sb: StepBatch, seqs: list) -> StepBatch:
    """A StepBatch holding only `seqs` (rows renumbered from 0, original order)."""
    import dataclasses

    out = StepBatch(sb.rank, sb.step, 0)
    row = 0
    tail, samp = set(sb.tail_rows), set(sb.sample_rows)
    for se in seqs:
        a, b = se.row0, se.row0 + se.n_rows
        k = len(out.seqs)
        ne = dataclasses.replace(se, row0=row, tail_row0=se.tail_row0 - se.row0 + row)
        for name in ("n_blocks", "score", "forced", "uid", "known"):
            setattr(ne, name, getattr(se, name, getattr(ne, name, None)))
        out.seqs.append(ne)
        out.positions.extend(sb.positions[a:b])
        out.seq_slot.extend([k] * (b - a))
        out.paged_slot.extend(sb.paged_slot[a:b])
        out.window_slot.extend(sb.window_slot[a:b])
        out.tail_rows.extend(r - a + row for r in range(a, b) if r in tail)
        out.sample_rows.extend(r - a + row for r in range(a, b) if r in samp)
        row += b - a
    out.n_rows = row
    return out


def build_compact(tb: "TableBuilder", sb: StepBatch, k1: int, row_tokens: list | None = None):
    """D13 (the format kernels/engram reads too): StepBatch -> CompactTables. One record per sequence
    (start = -1: the device's nc[slot]; host-known for prefill chunks, forced decode and the first step of a sequence
    new to its slot), window ops (clears first) and D12 deltas per (rank, slot, uid), host-known per-row data."""
    import numpy as np

    from rowexp_ref import CompactTables

    ct = CompactTables(sb.rank, sb.step, k1)
    clears, sets, btu, btp, rec = [], [], [], [], []
    rank, cst = sb.rank, tb.cstate                     # (rank, slot) -> [ident, blocks sent, wver sent, {block: page}]
    for se in sb.seqs:
        slot = se.state_slot
        key = (rank, slot)
        ident = (se.uid, se.inc)                       # a new incarnation (admit / move), even on a recycled uid + slot
        e = cst.get(key)
        start = se.start
        bt = se.block_table
        nb = se.n_blocks or len(bt)
        if e is None or e[0] != ident:                 # new to this slot: clear its ring, full block table, explicit start
            clears.append((slot, -1, -1))
            e = cst[key] = [ident, -1, -2, {}]
            if start < 0:
                start = se.known
        s0 = e[1]
        if s0 < 0 or s0 > nb:                          # first table, or it shrank: the full table
            btu.append((slot, 0, nb))
            btp.extend(bt[:nb])
            e[1] = nb
        elif nb > s0:
            btu.append((slot, s0, nb - s0))
            btp.extend(bt[s0:nb])
            e[1] = nb
        wv = se.wver
        if wv < 0 or e[2] != wv:                       # the window changed since it was last sent (~1 step in 11)
            ws, wt = e[3], se.window_table
            for blk, pg in wt.items():
                if ws.get(blk) != pg:
                    sets.append((slot, blk, pg))
                    ws[blk] = pg
            if len(ws) > len(wt):
                for blk in [b for b in ws if b not in wt]:
                    del ws[blk]
            e[2] = wv
        rec.append((slot, se.row0, se.n_rows, 0 if se.kind == "decode" else 1, start, se.replay_start,
                    se.cached_upto, se.uid, se.last_chunk, se.score, se.forced))      # bools pack as 0 / 1
    R = np.asarray(rec, dtype=np.int64).reshape(-1, 11)
    (ct.slot, ct.row0, ct.n_rows, ct.kind, ct.start, ct.replay_start, ct.cached_upto, ct.uid, ct.last_chunk, ct.score,
     ct.forced) = (R[:, i] for i in range(11))
    V = np.asarray(clears + sets, dtype=np.int64).reshape(-1, 3)
    ct.win_slot, ct.win_blk, ct.win_page = V[:, 0], V[:, 1], V[:, 2]
    B = np.asarray(btu, dtype=np.int64).reshape(-1, 3)
    ct.bt_upd_slot, ct.bt_upd_start, ct.bt_upd_len = B[:, 0], B[:, 1], B[:, 2]
    ct.bt_upd_pages = np.asarray(btp, dtype=np.int64)
    n = sb.n_rows
    if row_tokens is not None:
        ct.tokens = np.asarray(row_tokens, dtype=np.int64)[:n] if n else np.zeros(0, dtype=np.int64)
    else:                                              # host-known rows only: explicit starts
        tok = []
        for k, se in enumerate(sb.seqs):
            st = int(R[k, 4])
            tok.extend([tb.tokens_of(se.rid, st + i) for i in range(se.n_rows)] if st >= 0 else [-1] * se.n_rows)
        ct.tokens = np.asarray(tok or [0], dtype=np.int64)[:n]
    # targets: row i predicts the token of row i + 1 of the same sequence (rows of a sequence are contiguous)
    tg = np.full(n, -1, dtype=np.int64)
    if n:
        tg[:-1] = ct.tokens[1:]
        tg[ct.row0 + ct.n_rows - 1] = -1
    ct.targets = tg
    ct.tail_rows = np.asarray(sb.tail_rows, dtype=np.int64)
    ct.sample_idx = np.asarray(sb.sample_rows, dtype=np.int64)
    ct.rids = [se.rid for se in sb.seqs]
    return ct


# ---------------------------------------------------------------------------------------------- backends
def head_stub(tables: RankTables, rank: int, step: int, k1: int, seed: int = 0, topk: int = 20) -> StepResult:
    """STUB for kernels/head: per decoding sequence 1 .. k + 1 accepted tokens (real-traffic tail), synthetic ids; for
    score sequences a deterministic fake top-k per row (format of StepResult.logprobs, for plumbing tests only)."""
    res = StepResult(rank, step)
    score = tables.seq.get("score") if tables.seq else None
    forced = tables.seq.get("forced") if tables.seq else None
    for k, (rid, row0, nr, kind, last) in enumerate(tables.seq_rows):
        if forced is not None and int(forced[k]):
            import torch

            rows = range(row0, row0 + nr) if kind == "decode" else ([row0 + nr - 1] if last else [])
            if rows:
                g = torch.Generator().manual_seed(abs(hash((str(rid), step))) % (2 ** 31))
                res.logits[rid] = ([int(tables.positions[x]) for x in rows],
                                   torch.randn(len(rows), STUB_VOCAB, generator=g), 0)
            tg = [int(tables.targets[x]) for x in rows]
            if kind == "decode":
                res.accepted[rid] = nr                       # all forced drafts accepted (+ the bonus = next forced)
                res.tokens[rid] = tg
            elif last:
                res.tokens[rid] = tg[-1:]
            continue
        if score is not None and int(score[k]):
            rng = random.Random(f"{seed}/{rid}/{step}/lp")
            res.logprobs[rid] = [(int(tables.positions[row0 + i]),
                                  [(rng.randrange(1000, 120_000), -0.1 * (j + 1)) for j in range(topk)])
                                 for i in range(nr)]
            if last:
                res.tokens[rid] = [res.logprobs[rid][-1][1][0][0]]
            continue
        if kind == "decode":
            rng = random.Random(f"{seed}/{rid}/{step}")
            u = rng.random()
            a = 1 + sum(1 for p in ACCEPT_TAIL[: nr - 1] if u < p)
            res.accepted[rid] = a
            res.tokens[rid] = [rng.randrange(1000, 120_000) for _ in range(a)]
        elif last:
            res.tokens[rid] = [random.Random(f"{seed}/{rid}/{step}").randrange(1000, 120_000)]
    return res


class StubBackend:
    """CPU backend: no model; checks the tables' invariants and returns head_stub results."""

    def __init__(self, cfg: WorkerConfig, geom: Geometry, k1: int):
        self.cfg, self.geom, self.k1 = cfg, geom, k1
        self.steps = 0

    def run(self, tables: list, step: int) -> list:
        self.steps += 1
        out = []
        for r, t in enumerate(tables):
            if t.n:
                last = t.si.gather(1, (t.sl.long() - 1).clamp(min=0)[:, None]).squeeze(1).long()
                assert bool((last == t.own_slot).all()), "the last window key of a row is its own position"
            out.append(head_stub(t, r, step, self.k1, self.cfg.seed))
        return out


class SPLayerBackend:
    """GPU backend: the fork's sequence-parallel layer chain (kernels/sp_layer), driven by scheduler-built tables."""

    def __init__(self, cfg: WorkerConfig, k1: int, max_rows: int, log=print):
        import sys

        import torch

        import weights as Wt

        Wt._paths()
        for p in (os.path.join(Wt.PHYS, "kernels/sp_layer"),):
            sys.path.insert(0, p)
        os.environ.setdefault("ATTN2B_PDL", "1")
        import attn_half as AH
        import bench
        import test_comm
        import test_mhc
        from ds41h.ref import layer as RL
        from ds41h.ref.weights import default_reader
        from torch.utils.cpp_extension import load as _load

        self.torch, self.AH, self.cfg, self.k1, self.log = torch, AH, cfg, k1, log
        self.W = len(cfg.devices)
        self.devs = [torch.device("cuda", d) for d in cfg.devices]
        self.Rmax = max_rows
        t0 = time.time()
        self.hext, self.cext = test_mhc.ext(), test_comm.ext()
        self.cext.enable_peers(self.W)
        self.r2 = _load(name="route2_ext", sources=[os.path.join(Wt.PHYS, "kernels/moe_stream/route2.cu")], verbose=False,
                        extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr"])
        self.mcfg = RL.ModelConfig.from_hf(default_reader(cfg.model).config)
        log(f"worker: extensions in {time.time() - t0:.1f} s")
        reuse = int(cfg.weights.split(":")[1]) if cfg.weights.startswith("reuse:") else None
        self.reuse = reuse
        self.st, self.layer_w = [], []
        cs = {}
        for r, dev in enumerate(self.devs):
            torch.cuda.set_device(dev)
            mext = bench.build_extr()
            big = {}
            for L in (sorted({reuse}) if reuse is not None else cfg.layers):
                c = Wt.RATIO[L]
                if (c, r) not in cs:      # on the GPU, like vLLM / attn_half (CPU-built tables differ in bits)
                    import bench2b as B2

                    cs[(c, r)] = B2.stock_cos_sin(c, dev, 0)
                AW = Wt.finish_attn(Wt.load(f"attn_L{L}_v1", lambda L=L: Wt.build_attn(L), dev), dev, cs[(c, r)], cfg.w8a16)
                ex = Wt.load(f"L{L}_r{r}_raw_experts_v1", lambda L=L, r=r: Wt.build_raw_experts(L, r), dev)
                big[L] = (AW, ex)
            per = []
            for L in cfg.layers:
                m = Wt.finish_mhc(Wt.load(f"mhc_L{L}_v1", lambda L=L: Wt.build_mhc(L), dev), self.hext)
                dn = Wt.load(f"L{L}_r{r}_dense_v1", lambda L=L, r=r: Wt.build_dense(L, r), dev)
                AW, ex = big[reuse if reuse is not None else L]
                per.append(dict(m, **dn, AW=AW, w13=ex[0], s13=ex[1], w2=ex[2], s2=ex[3]))
            self.layer_w.append(per)
            emb = Wt.load("embed_v1", Wt.build_embed, dev)
            head = Wt.load(f"head_r{r}_v1", lambda r=r: Wt.build_head(r), dev)
            self.st.append(self._buffers(r, dev, mext, emb, head))
        self.streams = [torch.cuda.Stream(device=d) for d in self.devs]
        self.side = [torch.cuda.Stream(device=d) for d in self.devs]
        self.gf = [s["gf"] for s in self.st]
        self.gr = [s["gr"] for s in self.st]
        self.rsb = [s["rs"].view(torch.uint8) for s in self.st]
        self.fl_f = [s["fl_f"] for s in self.st]
        self.fl_s = [s["fl_s"] for s in self.st]
        for d in self.devs:
            torch.cuda.synchronize(d)
        log(f"worker: weights + buffers on {self.W} GPUs in {time.time() - t0:.0f} s (weights={cfg.weights})")

    def free_bytes(self) -> list:
        out = []
        for d in self.devs:
            f, _ = self.torch.cuda.mem_get_info(d)
            out.append(int(f))
        return out

    def _buffers(self, r, dev, mext, emb, head) -> dict:
        torch, R, W = self.torch, self.Rmax, self.W
        bf = torch.bfloat16
        s = dict(dev=dev, mext=mext, emb=emb, head=head)
        for k in ("X", "xm2", "res0"):
            s[k] = torch.zeros(R, HC, D, dtype=bf, device=dev)
        for k in ("a_pre", "a_post", "f_pre", "f_post", "post0", "lag0"):
            s[k] = torch.zeros(R, HC, device=dev)
        for k in ("a_comb", "f_comb"):
            s[k] = torch.zeros(R, HC, HC, device=dev)
        s["comb0"] = torch.eye(HC, device=dev).expand(R, HC, HC).contiguous()
        s["lag0"][:, 0] = 1.0
        s["x0"] = torch.zeros(R, D, dtype=bf, device=dev)
        s["part_mhc"] = torch.empty(160 * R * 25, device=dev)
        s["a"] = torch.zeros(R, D, dtype=bf, device=dev)
        s["f"] = torch.zeros(R, D, dtype=bf, device=dev)
        s["logits"] = torch.empty(R, NE, device=dev)
        s["ids"] = torch.empty(R, TOPK, dtype=torch.int32, device=dev)
        s["wts"] = torch.empty(R, TOPK, device=dev)
        s["route_row"] = torch.zeros(R, 64, dtype=torch.uint8, device=dev)
        s["ptok_own"] = torch.zeros(R, dtype=torch.int32, device=dev)
        s["g5ctr"] = torch.zeros(2, dtype=torch.int32, device=dev)
        s["gf"] = torch.zeros(W * R, D * 2, dtype=torch.uint8, device=dev)
        s["gr"] = torch.zeros(W * R, 64, dtype=torch.uint8, device=dev)
        s["rs"] = torch.zeros(W, R, D, dtype=bf, device=dev)
        for k in ("fl_f", "fl_s"):
            s[k] = torch.zeros(4, dtype=torch.int32, device=dev)
        for k in ("ep_f", "ep_s"):
            s[k] = torch.zeros(1, dtype=torch.int32, device=dev)
        s["moe_out"] = torch.zeros(R, D, dtype=bf, device=dev)
        Tall = W * R
        max_ec = max(int(mext.route_max_ec()), int(self.r2.route2_max_ec(Tall)))
        s.update(max_ec=max_ec, sms=torch.cuda.get_device_properties(dev).multi_processor_count,
                 ecs=torch.zeros(max_ec * (32 + 12 * 64), dtype=torch.uint8, device=dev),
                 t1=torch.zeros(max_ec * 9 * 24, dtype=torch.uint8, device=dev),
                 t2=torch.zeros(max_ec * 40 * 24, dtype=torch.uint8, device=dev),
                 counts=torch.zeros(8, dtype=torch.int32, device=dev),
                 slotmap=torch.zeros(Tall * TOPK, dtype=torch.int32, device=dev),
                 rts=torch.zeros(max(int(mext.route_scratch_ints()), int(self.r2.route2_scratch_ints())), dtype=torch.int32,
                                 device=dev),
                 img=torch.zeros((Tall * TOPK + 7 * max_ec) * 80 * 128, dtype=torch.uint8, device=dev),
                 himg=torch.zeros((Tall * TOPK + 7 * max_ec) * 9 * 128, dtype=torch.uint8, device=dev),
                 Y=torch.zeros(Tall * TOPK, D, dtype=bf, device=dev),
                 gu=torch.empty(Tall, 2 * 576, dtype=bf, device=dev),
                 hsh=torch.empty(Tall, 576, dtype=bf, device=dev),
                 sh=torch.empty(Tall, D, dtype=bf, device=dev))
        return s

    # -- per step
    def prepare(self, r: int, pools: Pools, tab: RankTables) -> None:
        """Per-rank, per-step: device tables, one attn_half cache per KV source type (FlashMLA planned once per type and
        shared by the layers of that type), the layer-0 input (embedding in 4 hc copies)."""
        torch, AH, s = self.torch, self.AH, self.st[r]
        scratch = pools.scratch_window_page * WINDOW_PAGE_TOKENS
        t = tab.to_device(s["dev"], scratch)
        n = max(tab.n, 1)
        s["n"], s["tab"] = tab.n, t
        s["caches"] = {}
        wsn = s.setdefault("ws_by_n", {})
        if n not in wsn:                                 # attn_half workspace for n rows (one, shared by all layers)
            wsn[n] = AH._workspace(n, s["dev"])
        ws = wsn[n]
        with torch.cuda.device(s["dev"]):
            for L in self.cfg.layers:
                lw = self.layer_w[r][L]
                src = kv_source_of(lw["AW"]["layer"]) if self.reuse is not None else kv_source_of(L)
                key = (src, L)
                csrc = src if src is not None else 20
                c = dict(R=n, device=s["dev"], stride=WINDOW_PAGE_BYTES, sbuf=pools.window_base(L), k_swa=pools.window_view(L),
                         k_comp=pools.comp_view(csrc), si=t["si"], sl=t["sl"], ti=t[f"ti{csrc}"],
                         tl=t[f"tl{csrc}"] if src is not None else torch.zeros_like(t["sl"]),
                         own_slot=t["own_slot"], scale=lw["AW"]["scale"], ws=ws, sched=None)
                same = [k for k in s["caches"] if k[0] == src]
                if same:
                    AH.share_plan(c, s["caches"][same[0]])
                else:
                    AH.plan(lw["AW"], c)
                s["caches"][key] = c
            emb = torch.index_select(s["emb"], 0, t["tokens"])
            s["res0"][:n].copy_(emb.unsqueeze(1).expand(n, HC, D))

    # -- the layer as pieces; every piece is issued for ALL ranks before the next piece (as sp_layer's run_all): a hidden
    #    host sync inside a piece (lazy init, allocator) then never waits on a peer push that was not issued yet
    def _inputs(self, r: int, L: int):
        s = self.st[r]
        if L == self.cfg.layers[0]:
            return s["x0"], s["res0"], s["post0"], s["comb0"], s["lag0"]
        return s["moe_out"], s["xm2"], s["f_post"], s["f_comb"], s["f_pre"]

    def p_attn_bd(self, r: int, L: int) -> None:
        s, lw, cfg = self.st[r], self.layer_w[r][L], self.mcfg
        x, res, post, comb, lag = self._inputs(r, L)
        self.hext.boundary(x, res, post, comb, lag, lw["fa"], lw["sa"], lw["ba"], lw["na"], cfg.norm_eps, cfg.hc_eps,
                           cfg.hc_iters, s["X"], s["a_pre"], s["a_post"], s["a_comb"], s["a"], s["part_mhc"])
        # engram_stub: layers 1 and 14 run without the n-gram embedding

    def p_attn(self, r: int, L: int) -> None:
        s, lw = self.st[r], self.layer_w[r][L]
        n = s["n"]
        src = kv_source_of(lw["AW"]["layer"]) if self.reuse is not None else kv_source_of(L)
        if n:
            s["y_cur"] = self.AH.attn_half(lw["AW"], s["caches"][(src, L)], s["a"][:n], s["tab"]["positions"],
                                           insert=self.cfg.insert)
        else:
            s["y_cur"] = s["x0"][:1]

    def p_ffn_bd(self, r: int, L: int) -> None:
        s, lw, cfg = self.st[r], self.layer_w[r][L], self.mcfg
        self.hext.boundary2(s["y_cur"], s["X"], s["a_post"], s["a_comb"], s["a_pre"], lw["ff"], lw["sf"], lw["bf"], lw["nf"],
                            cfg.norm_eps, cfg.hc_eps, cfg.hc_iters, s["xm2"], s["f_pre"], s["f_post"], s["f_comb"], s["f"],
                            s["part_mhc"], s["ptok_own"])

    def p_router(self, r: int, L: int) -> None:
        torch, s, lw = self.torch, self.st[r], self.layer_w[r][L]
        torch.mm(s["f"], lw["gate_w"].t(), out_dtype=torch.float32, out=s["logits"])
        s["mext"].router_top_pack2(s["logits"], lw["gate_b"], s["ids"], s["wts"], 1.5, s["route_row"], s["n"], s["ptok_own"])

    def p_ag(self, r: int, L: int) -> None:
        torch, s, R = self.torch, self.st[r], self.Rmax
        self.cext.ag_push2(s["f"].view(torch.uint8).view(R, D * 2), s["route_row"], r, self.gf, self.fl_f, self.gr, s["ep_f"],
                           self.cfg.ncta)
        self.cext.wait(s["fl_f"], s["ep_f"], self.W * self.cfg.ncta)

    def p_moe(self, r: int, L: int) -> None:
        torch, s, lw, ex = self.torch, self.st[r], self.layer_w[r][L], self.st[r]["mext"]
        Tall = self.W * self.Rmax
        fall = s["gf"].view(torch.bfloat16).view(Tall, D)
        cur = torch.cuda.current_stream()
        ev0 = torch.cuda.Event()
        ev0.record(cur)
        with torch.cuda.stream(self.side[r]):          # shared expert slice on the side stream
            self.side[r].wait_event(ev0)
            torch.mm(fall, lw["sw13"].t(), out=s["gu"])
            ex.swiglu(s["gu"], s["hsh"])
            torch.mm(s["hsh"], lw["sw2"].t(), out=s["sh"])
            ev1 = torch.cuda.Event()
            ev1.record(self.side[r])
        self.r2.route2r(s["gr"], Tall, s["ecs"], s["t1"], s["t2"], s["counts"], s["slotmap"], s["rts"], s["sms"])
        ex.prepr5(fall, s["ecs"], s["counts"], s["img"], s["gr"], s["sms"], s["g5ctr"])
        ex.gemm1r(lw["w13"], lw["s13"], s["ecs"], s["t1"], s["counts"], s["img"], s["himg"], 0)
        ex.gemm2r(lw["w2"], lw["s2"], s["ecs"], s["t2"], s["counts"], s["himg"], s["Y"], 0)
        cur.wait_event(ev1)
        self.cext.combine_push(s["Y"], s["slotmap"][: Tall * TOPK], s["sh"], self.Rmax, r, self.rsb, self.fl_s, s["ep_s"],
                               self.cfg.push_ctas)

    def p_rs(self, r: int, L: int) -> None:
        s = self.st[r]
        self.cext.rs_sum(s["rs"], s["fl_s"], s["ep_s"], self.W * self.cfg.push_ctas, s["moe_out"], 132)

    PIECES = ("p_attn_bd", "p_attn", "p_ffn_bd", "p_router", "p_ag", "p_moe", "p_rs")

    def run_layers(self, trace=None) -> None:
        torch = self.torch
        for L in self.cfg.layers:
            for name in self.PIECES:
                fn = getattr(self, name)
                for r in range(self.W):
                    with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                        fn(r, L)
                if trace:
                    trace(L, name)

    def sync(self) -> None:
        for s_ in self.streams:
            s_.synchronize()

    def capture(self) -> None:
        """One CUDA graph per rank of the whole 40-layer chain on the current tables (prepare() first; FlashMLA plans
        are made outside the capture)."""
        torch = self.torch
        self.graphs = []
        for r in range(self.W):
            with torch.cuda.device(self.devs[r]):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=self.streams[r]):
                    for L in self.cfg.layers:
                        for name in self.PIECES:
                            getattr(self, name)(r, L)
                self.graphs.append(g)
        for d in self.devs:
            torch.cuda.synchronize(d)

    def replay(self, reps: int) -> list:
        """Replay the captured step graphs `reps` times on all ranks; us per step per rank (CUDA events)."""
        torch = self.torch
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(self.W)]
        for r in range(self.W):
            with torch.cuda.device(self.devs[r]):
                ev[r][0].record(torch.cuda.current_stream(self.devs[r]))
        for _ in range(reps):
            for r in range(self.W):
                with torch.cuda.device(self.devs[r]):
                    self.graphs[r].replay()
        for r in range(self.W):
            with torch.cuda.device(self.devs[r]):
                ev[r][1].record(torch.cuda.current_stream(self.devs[r]))
        for d in self.devs:
            torch.cuda.synchronize(d)
        return [ev[r][0].elapsed_time(ev[r][1]) * 1000 / reps for r in range(self.W)]

    def run(self, pools: list, tables: list, step: int) -> list:
        for r in range(self.W):
            self.prepare(r, pools[r], tables[r])
        self.run_layers()
        self.sync()
        return [head_stub(tables[r], r, step, self.k1, self.cfg.seed) for r in range(self.W)]

    def outputs_finite(self) -> list:
        out = []
        for s in self.st:
            n = max(s["n"], 1)
            out.append(bool(self.torch.isfinite(s["moe_out"][:n].float()).all()) and
                       bool(self.torch.isfinite(s["xm2"][:n].float()).all()))
        return out


# ---------------------------------------------------------------------------------------------- worker
class Worker:
    """Pools + tables + backend for the 4 ranks."""

    # Worker.step sets ps.prepared (a threading.Event, when the caller passes one) once the host tables (value copies)
    # and the lifecycle ctx are built, before the engine call. From then on, neither this worker nor the engine reads
    # what the scheduler mutates: only the per-step StepBatch objects and the APPEND-ONLY prefix
    # se.block_table[:se.n_blocks] of the KV manager's page lists (allocate appends, migrate swaps in a new list, free
    # leaves the list as is). Window dicts (popped by commit) are read by the TableBuilder only. So the caller may run
    # scheduler.update / schedule for other steps while the engine runs (the adapter's step thread).
    signals_prepared = True

    @property
    def device_positions(self) -> bool:
        """The backend expands decode rows on the device (D13 step_compact): async scheduling can run on it."""
        return bool(getattr(self.backend, "device_positions", False) and callable(getattr(self.backend, "step_compact", None)))

    def __init__(self, cfg: WorkerConfig, geom: Geometry, backend: str = "stub", max_rows: int = 512, log=print):
        self.cfg, self.log = cfg, log
        self.k1 = geom.rows_per_decode_seq
        self.tb = TableBuilder()                 # stub top-512 tables off: only the interim sp_layer backend uses them
        if backend == "stub":
            self.geom = geom
            self.backend = StubBackend(cfg, geom, self.k1)
            self.pools = None
        elif backend == "engine":
            # the Engine (engine/): it loads the weights; we size and bind
            # the pools; per step it takes the tables and returns StepResults
            import importlib
            import sys

            import weights as Wt

            if Wt.PHYS not in sys.path:
                sys.path.insert(0, Wt.PHYS)
            mod, cls = cfg.engine_cls.rsplit(".", 1)
            Engine = getattr(importlib.import_module(mod), cls)
            self.backend = Engine(devices=cfg.devices, model_dir=cfg.model, max_rows=max_rows, **cfg.engine_kw)
            self.tb.stub_indexer = bool(getattr(self.backend, "wants_stub_indexer", False))
            free = min(self.backend.free_bytes())
            # rows per rank per step: the engine's mixed-step capacity (decode-only steps use its graph buckets <=
            # max_rows; seats stay min(max_seqs_per_rank, ...) so all-decode rows fit the largest bucket)
            ecfg = getattr(self.backend, "cfg", None)
            step_rows = int(getattr(self.backend, "max_step_rows", 0) or getattr(ecfg, "max_mixed_rows", 0) or max_rows)
            seats = min(geom.max_seqs_per_rank, max_rows // geom.rows_per_decode_seq)
            # the engine's tail-row limit (layers 21..39 + head rows per rank per step; HSD's largest tail bucket)
            hsd = getattr(self.backend, "hsd", None)
            tail = int(getattr(self.backend, "max_tail_rows", 0) or getattr(hsd, "max_tail_rows", 0) or
                       geom.max_tail_rows_per_rank)
            self.geom = size_pools(free, cfg, dataclasses.replace(geom, max_rows_per_rank=step_rows,
                                                                  max_seqs_per_rank=seats, max_tail_rows_per_rank=tail))
            self.pools = [Pools(self.geom, d) for d in self.backend.devices]
            self.backend.bind_pools(self.pools)
            p = self.geom.pools()[0]
            log(f"worker[engine]: KV pools {self.geom.kv_bytes_per_rank / 1e9:.1f} GB per rank = {p.num_pages:,} pages")
        else:
            import weights as Wt

            self.backend = SPLayerBackend(cfg, self.k1, max_rows, log)
            self.tb.stub_indexer = True              # the interim chain reads the stub top-512 (D9)
            if self.backend.reuse is not None:
                self.tb.sources = (kv_source_of(self.backend.reuse),)
            free = min(self.backend.free_bytes())
            extra = 0
            if self.backend.reuse is not None and cfg.size_as_full_model:
                pb = Wt.plan_bytes()
                extra = pb["total"] - (Wt.EXPERT_BYTES_PER_LAYER + Wt.ATTN_BYTES_PER_LAYER + Wt.EMBED_BYTES
                                       + Wt.HEAD_BYTES_PER_RANK + len(cfg.layers) * Wt.DENSE_BYTES_PER_LAYER)
            self.geom = size_pools(free, cfg, dataclasses.replace(geom, max_rows_per_rank=max_rows), extra)
            self.pools = [Pools(self.geom, d) for d in self.backend.devs]
            p = self.geom.pools()[0]
            log(f"worker: KV pools {self.geom.kv_bytes_per_rank / 1e9:.1f} GB per rank = {p.num_pages:,} pages "
                f"({p.num_pages * PAGE_TOKENS / 1e6:.1f}M tokens), free before {free / 1e9:.1f} GB, reuse-mode "
                f"weight allowance {extra / 1e9:.1f} GB")

    def poison(self, batches: list) -> int:
        """Debug ($PHYS_POISON_ON_FREE): fill the pages / slots the KV manager freed since the last step with 0xFF, before
        this step's copies and kernels, so a read of a previous owner's leftovers turns into NaN. State slots are also
        passed to backend.poison_state_slots(rank, slots) when the backend has it (engine-side rings, e.g. Engram)."""
        if self.pools is None:
            return 0
        import torch

        size = dict(paged=PAGE_BYTES, window=WINDOW_PAGE_BYTES, state=STATE_SLOT_BYTES)
        n, touched = 0, set()
        hook = getattr(self.backend, "poison_state_slots", None)
        for sb in batches:
            ops = getattr(sb, "poison", None)
            if not ops:
                continue
            pools = self.pools[sb.rank]
            slots = []
            for pool, page in ops:
                b = size[pool]
                t = getattr(pools, pool)
                t[page * b:(page + 1) * b].fill_(255)
                touched.add(t.device)
                if pool == "state":
                    slots.append(page)
                n += 1
            if slots and callable(hook):
                hook(sb.rank, slots)
        for d in touched:
            if d.type == "cuda":
                torch.cuda.synchronize(d)
        return n

    def copies(self, batches: list) -> int:
        """Rank-to-rank page copies of this step (before any kernel)."""
        ops = {}
        for sb in batches:
            for c in sb.copies:
                ops[(c.pool, c.src_rank, c.src_page, c.dst_rank, c.dst_page)] = c
        if self.pools is None or not ops:
            return len(ops)
        import torch

        size = dict(paged=PAGE_BYTES, window=WINDOW_PAGE_BYTES, state=STATE_SLOT_BYTES)
        touched = set()
        for pool, sr, s0, dr, d0, n in copy_runs(list(ops.values())):
            src = getattr(self.pools[sr], pool)
            dst = getattr(self.pools[dr], pool)
            b = size[pool]
            dst[d0 * b:(d0 + n) * b].copy_(src[s0 * b:(s0 + n) * b], non_blocking=True)
            touched |= {src.device, dst.device}
        for d in touched:                  # v0: the step's streams must see the copied pages (copies are rare)
            if d.type == "cuda":
                torch.cuda.synchronize(d)
        # engine-side per-slot state that is not in the pools follows a moved sequence's state slot (Engram's token-history
        # ring; left behind on a rank move, the moved sequence would get a wrong n-gram lookback)
        hook = getattr(self.backend, "copy_state_slots", None)
        st_ops = [(c.src_rank, c.src_page, c.dst_rank, c.dst_page) for c in ops.values() if c.pool == "state"]
        if st_ops and callable(hook):
            hook(st_ops)
        return len(ops)

    def step(self, ps, grammar: dict | None = None, mm: list | None = None) -> list:
        """The adapter's worker protocol (adapter/bridge.py PhysStep; D11): batches + per-row input
        tokens + the request lifecycle (new requests with their sampling params, finished / preempted) + grammar rows.
        `mm`: the step's image rows, per rank (idx int32 [M], emb [M, hidden]) on that rank's device or None (the
        caller encodes and gathers them; interfaces.image_rows lists the rows); passed on as StepContext.mm.
        Returns one StepResult per rank (+ drafts for structured-output requests)."""
        from interfaces import SeqInit, StepContext, sampling_dict

        import time as _t

        batches = ps.batches
        t0 = _t.perf_counter()
        self._shape_log(ps, batches, t0)
        self.poison(batches)
        self.copies(batches)
        t1 = _t.perf_counter()
        compact = self.cfg.async_mode or self.cfg.compact
        tables = None if compact else \
            [self.tb.build(sb, ps.row_tokens[r] if getattr(ps, "row_tokens", None) else None) for r, sb in enumerate(batches)]
        t2 = _t.perf_counter()
        ctx = StepContext(
            new=[SeqInit(q.rid, list(q.prompt_token_ids), list(q.output_token_ids), sampling_dict(q.sampling_params),
                         bool(getattr(q, "structured", False))) for q in getattr(ps, "new_reqs", [])],
            dropped=set(getattr(ps, "finished", set())) | set(getattr(ps, "preempted", set())),
            grammar=dict(grammar or {}), prev_tokens=dict(getattr(ps, "prev_tokens", {})), order=list(getattr(ps, "order", [])),
            mm=self._mm_ctx(mm, batches))
        step = getattr(ps, "step", batches[0].step if batches else 0)
        prepared = getattr(ps, "prepared", None)
        if compact:
            cts = [build_compact(self.tb, sb, self.k1, ps.row_tokens[r] if getattr(ps, "row_tokens", None) else None)
                   for r, sb in enumerate(batches)]
            if prepared is not None:
                prepared.set()
            return self.backend.step_compact(batches, cts, ctx=ctx)
        if prepared is not None:
            prepared.set()
        if isinstance(self.backend, StubBackend):
            return self.backend.run(tables, step)
        if isinstance(self.backend, SPLayerBackend):
            return self.backend.run(self.pools, tables, step)
        t3 = _t.perf_counter()
        res = self.backend.step(batches, tables, ctx=ctx)
        # host split of the last step (ms): page copies, tables (TableBuilder), lifecycle ctx, engine.step
        self.last_ms = (1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2), 1e3 * (_t.perf_counter() - t3))
        return res

    @staticmethod
    def _mm_ctx(mm, batches: list):
        """StepContext.mm from the caller's per-rank image rows: None when no rank has any; shapes checked on the host
        (no device sync)."""
        if mm is None or not any(x is not None for x in mm):
            return None
        assert len(mm) == len(batches), (len(mm), len(batches))
        out = []
        for r, x in enumerate(mm):
            if x is None:
                out.append(None)
                continue
            idx, emb = x
            assert idx.dim() == 1 and emb.dim() == 2 and idx.shape[0] == emb.shape[0], (r, tuple(idx.shape), tuple(emb.shape))
            assert idx.shape[0] <= batches[r].n_rows, (r, idx.shape[0], batches[r].n_rows)
            out.append((idx, emb))
        return out

    def _shape_log(self, ps, batches: list, t0: float) -> None:
        """$PHYS_STEP_SHAPES=<path>: one JSON line per step with row counts only (no tokens, no content): the step's
        start time, the scheduler's estimate, per-rank real rows and prefill rows, decode sequences, copies. The
        imbalance of mixed steps on a served replay is read from it."""
        import json as _json
        import os as _os

        path = _os.environ.get("PHYS_STEP_SHAPES")
        if not path or getattr(self, "_shape_off", False):
            return
        try:                                     # a log, never a reason to fail a step: any error turns it off
            if getattr(self, "_shape_f", None) is None:
                _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
                self._shape_f = open(path, "a", buffering=1)
            rows = [sb.n_rows for sb in batches]
            pf = [sum(se.n_rows for se in sb.seqs if se.kind != "decode") for sb in batches]
            dec = [sum(1 for se in sb.seqs if se.kind == "decode") for sb in batches]
            cpn = [sum(1 for se in sb.seqs if getattr(se, "cp", None) is not None) for sb in batches]
            self._shape_f.write(_json.dumps(dict(t=round(t0, 4), step=getattr(ps, "step", batches[0].step if batches else 0),
                                                 est_ms=round(getattr(ps, "est_us", 0.0) / 1e3, 2), rows=rows, pf=pf,
                                                 dec_seqs=dec, cp=cpn, copies=sum(len(sb.copies) for sb in batches))) + "\n")
        except Exception as e:  # noqa: BLE001
            self._shape_off = True
            self.log(f"worker: PHYS_STEP_SHAPES log off after an error: {e!r}")

    def run_step(self, batches: list) -> list:
        self.poison(batches)
        self.copies(batches)
        if self.cfg.async_mode or self.cfg.compact:  # D13: compact tables, rows expanded on the device
            return self.backend.step_compact(batches, [build_compact(self.tb, sb, self.k1) for sb in batches])
        tables = [self.tb.build(sb) for sb in batches]
        step = batches[0].step if batches else 0
        if isinstance(self.backend, StubBackend):
            return self.backend.run(tables, step)
        if isinstance(self.backend, SPLayerBackend):
            return self.backend.run(self.pools, tables, step)
        return self.backend.step(batches, tables)           # the engine


# ---------------------------------------------------------------------------------------------- synthetic KV (smoke)
def synthetic_records(n: int, gen=None):
    """n valid fp8_ds_mla records [n, 584] uint8: 448 e4m3 NoPE values ~N(0, 0.5), 64 bf16 RoPE values ~N(0, 0.5),
    7 ue8m0 scales = 2^0 (byte 127), 1 pad byte."""
    import torch

    g = gen or torch.Generator().manual_seed(0)
    nope = (torch.randn(n, 448, generator=g) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    rope = (torch.randn(n, 64, generator=g) * 0.5).to(torch.bfloat16).view(torch.uint8)
    sc = torch.full((n, 8), 127, dtype=torch.uint8)
    sc[:, 7] = 0
    return torch.cat([nope, rope, sc], 1).contiguous()


def fill_synthetic(pools: Pools, tables: RankTables, layers, sources=(20,)) -> int:
    """Write synthetic records into every window slot (each layer) and compressed slot (each source) the step reads, so
    the attention runs on finite data. Returns the number of records written."""
    import torch

    import weights as Wt

    Wt._paths()
    import attn_half as AH

    total = 0
    if tables.n:
        ws = torch.unique(tables.si[tables.si >= 0].long())
        rec = synthetic_records(ws.numel())
        for L in layers:
            AH._write_pages(pools.window_base(L), WINDOW_PAGE_BYTES, WINDOW_PAGE_TOKENS, ws // WINDOW_PAGE_TOKENS,
                            ws % WINDOW_PAGE_TOKENS, rec)
            total += ws.numel()
    for src in sources:
        if not tables.n:
            continue
        c = torch.unique(tables.ti[src][tables.ti[src] >= 0].long())
        if not c.numel():
            continue
        bs = RECORDS_PER_PAGE[src]
        AH._write_pages(pools.comp_base(src), PAGE_BYTES, bs, c // bs, c % bs, synthetic_records(c.numel()))
        total += c.numel()
    return total
