#!/usr/bin/env python3
"""ds41h-phys: the engine's PREFILL rows through production's own prefill attention.

Production (vllm deepseek_v41/nvidia/flashmla.py, DeepseekV4FlashMLAAttention._forward_prefill) never runs prefill tokens
through the decode split-kv kernel. Per chunk of requests it
  1. dequantizes each request's compressed region [0, seq_len // ratio) and window region [seq_len - gather_len, seq_len)
     into a bf16 workspace [requests, M, 512] (dequantize_and_gather_k_cache, exact: e4m3 x 2^k and bf16 RoPE),
  2. combines each token's top-512 (LOCAL compressed positions, the indexer's order) and its window positions into one
     index list into that workspace (combine_topk_swa_indices: top-k first, then the window ascending),
  3. attends with FlashMLA's sparse PREFILL kernel (flash_mla_sparse_fwd: bf16 K / V, sink, topk_length).
The engine ran its prefill rows through the decode kernel instead (one query row per tile, fp8 records gathered and
dequantized per row, split-kv + combine). This module is production's path for those rows: the same three ops with the
same arguments, so a prefill row's attention output is production's prefill numerics bit for bit for the same q, records
and top-512 list. Request grouping (which requests share a workspace) only sets memory and launch counts: a token's
result depends on its own index list and the K / V rows it points to, not on its neighbours in the workspace.

Engine API (the engine wires it behind its own flag; nothing here changes attn_half or the decode path):
  pl = Plan(dev, seqs, si, sl, bt)        once per step and rank, before layer 0
      seqs: [(row0, n_rows, first_pos, replay_start)] for the step's PREFILL sequences, in row order, rows contiguous
            (build_step_batch puts decode rows first, then each chunk's rows in order); CP helper sub-chunks are prefill
            sequences of their own (first_pos = the sub-chunk's first position).
      si / sl: the step's window slot lists [R, 1, 128] / [R] (worker tables, device): each sequence's window pages are
            read from them (row0's list + every row's last entry = positions max(replay_start, first - 127) .. end - 1).
      bt: int32 [S, max_blocks] device, each prefill sequence's 64-token page ids (the paged pool; all sources share them).
      n_pool_pages: pages in the rank's paged pool (for the global -> local top-k map).
  attend(pl, ratio, q, out, k_swa, k_comp, ti, scale, sink)    per layer, for the plan's rows only:
      q / out: [n_pf, 64, 512] bf16 views of the step's q / o at rows pl.row_lo .. pl.row_hi (out is written);
      k_swa: the layer's window k_cache view (pools.window_view(L), [pages, 32, 1, 584]); k_comp: the source's
      compressed view (pools.comp_view(src)) or None (ratio 0); ti: the rows' GLOBAL top-512 slots [n_pf, 512] (or
      [n_pf, 1, 512]; the engine's s["ti"][index_source] rows) or None; scale / sink as for the decode kernel.
      src / isrc: the layer's kv source and index source (reuse of the compressed gather and the combine across the
      layers of a source within the step; omit both for production's per-layer calls: same bits either way).
The inverse RoPE, wo_a, wo_b and the decode rows stay as they are. Rows of the bucket that are neither decode nor prefill
(padding) are not touched here.

Measured (pf_step.py, 1 GPU, one 4,096-row chunk at 150K / 196K depth + 192 decode rows, layers 0..20):
the attention core 29.9 -> 18.7 ms per step (-11.2 ms); per layer was 654 / 1,518 / 1,834 us (ratio 0 / 2 / 1) vs
decode rows 80 / 102 / 279 + prefill 334 / 824 / 1,046. Tail bucket (layers 21..39, 128 chunk rows + 192 decode rows):
+0.3 ms with the split, so tail rows stay on the decode kernel. Prefill rows' output == production's per-layer calls,
bitwise (pf_test.py, pf_step.py).
"""
from __future__ import annotations

import os

import torch

# ATTN2B_PF=1 (default 0): the switch the engine reads to route a step's prefill rows here (bucket A, layers 0..20).
# Nothing in this module or attn_half changes behaviour by itself; the engine decides per step.
ENABLED = os.environ.get("ATTN2B_PF", "0") == "1"

WIN = 128
TOPK = 512
PAGE_TOKENS = 64
WPT = 32                      # window page tokens
PREFILL_CHUNK_SIZE = 4        # production's requests per workspace (deepseek_v41/attention.py)
WS_CAP = int(float(os.environ.get("ATTN2B_PF_WS_MB", "1024")) * 2**20)   # bf16 workspace per request group (bytes)
WS_TOTAL = int(float(os.environ.get("ATTN2B_PF_WS_TOTAL_MB", "3072")) * 2**20)   # all groups kept apart (reuse) below this
i32, bf = torch.int32, torch.bfloat16


def _ops():
    from vllm.models.deepseek_v41.common.ops.cache_utils import combine_topk_swa_indices, dequantize_and_gather_k_cache
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    return dequantize_and_gather_k_cache, combine_topk_swa_indices, flash_mla_sparse_fwd


def _view3(k: torch.Tensor) -> torch.Tensor:
    """[pages, rpp, 1, 584] (FlashMLA decode view) -> [pages, rpp, 584] (production's gather view), same bytes."""
    if k.dim() == 3:
        return k
    return torch.as_strided(k, (k.shape[0], k.shape[1], k.shape[3]), (k.stride(0), k.stride(1), k.stride(3)))


class Plan:
    """The step's prefill sequences on one rank (host ints + small device tables), shared by every layer of the step."""

    def __init__(self, dev, seqs: list, si: torch.Tensor, sl: torch.Tensor, bt: torch.Tensor, n_pool_pages: int):
        self.dev = torch.device(dev)
        self.S = len(seqs)
        assert self.S, "no prefill sequences"
        self.row_lo = int(seqs[0][0])
        r = self.row_lo
        firsts, ends, g0s, qsl = [], [], [], [0]
        for row0, n, first, rs in seqs:
            assert int(row0) == r and n > 0, "prefill rows must be contiguous, in sequence order"
            r += int(n)
            firsts.append(int(first))
            ends.append(int(first) + int(n))
            g0s.append(max(int(rs), int(first) - (WIN - 1), 0))
            qsl.append(r - self.row_lo)
        self.row_hi, self.n = r, r - self.row_lo
        self.ends, self.g0s, self.qsl_h = ends, g0s, qsl
        self.gls = [e - g for e, g in zip(ends, g0s)]
        self.seq_lens = torch.tensor(ends, dtype=i32, device=self.dev)
        self.gather_lens = torch.tensor(self.gls, dtype=i32, device=self.dev)
        self.qsl = torch.tensor(qsl, dtype=i32, device=self.dev)
        self.bt = bt[: self.S].to(self.dev, i32).contiguous()
        # window block table [S, end // 32 + 1] from the rows' window lists (blocks outside the region: page 0, never read)
        wmax = max(ends) // WPT + 1
        self.wbt = torch.zeros(self.S, wmax, dtype=i32, device=self.dev)
        si2 = si.view(si.shape[0], -1)
        for s, (row0, n, first, rs) in enumerate(seqs):
            row0, n = int(row0), int(n)
            k0 = int(sl[row0])
            head = si2[row0, :k0]                                                     # positions g0 .. first
            tail = si2[row0 + 1: row0 + n].gather(1, (sl[row0 + 1: row0 + n].long() - 1)[:, None]).squeeze(1)
            slots = torch.cat([head, tail]).long()                                    # positions g0 .. end - 1
            p = torch.arange(self.g0s[s], ends[s], device=self.dev)
            assert slots.numel() == p.numel(), (slots.numel(), p.numel())
            self.wbt[s].index_put_((p // WPT,), (slots // WPT).to(i32))
        # global -> local top-k: block index of every page of the prefill sequences (prefix-shared pages: same index)
        self.inv = torch.full((n_pool_pages + 1,), -1, dtype=i32, device=self.dev)
        for s in range(self.S):
            nb = -(-ends[s] // PAGE_TOKENS)
            self.inv[self.bt[s, :nb].long()] = torch.arange(nb, dtype=i32, device=self.dev)
        self.comb, self.clen, self.comb_key = {}, {}, {}   # per index width (640 / 128): lists, lengths, (ratio, isrc)
        self.loc = torch.empty(self.n, TOPK, dtype=i32, device=self.dev)
        self.groups, self.gws = {}, {}
        self.buf = None
        self.ws_ratio = None      # the ratio whose group layout the workspace holds
        self.comp_src = None      # the kv source whose compressed region is resident in the workspace (None: not valid)
        self.stats = dict(comp_gathers=0, comp_reused=0, combines=0, comb_reused=0)

    def group_plan(self, ratio: int) -> list:
        """[(s_a, s_b, N, M)]: consecutive requests sharing one workspace (<= PREFILL_CHUNK_SIZE, area cap; a request
        alone is never split), N = the group's max compressed count, M = N + the max window gather length."""
        if ratio not in self.groups:
            comp = [e // ratio if ratio else 0 for e in self.ends]
            out, a = [], 0
            while a < self.S:
                N, G, b = comp[a], self.gls[a], a + 1
                while b < self.S and b - a < PREFILL_CHUNK_SIZE:
                    N2, G2 = max(N, comp[b]), max(G, self.gls[b])
                    if (b - a + 1) * (N2 + G2) * 1024 > WS_CAP:
                        break
                    N, G, b = N2, G2, b + 1
                out.append((a, b, N, N + G))
                a = b
            self.groups[ratio] = out
        return self.groups[ratio]

    def group_ws(self, ratio: int) -> tuple[list, bool]:
        """The groups' bf16 workspaces [(b - a), M, 512] for this ratio, and whether they are disjoint (every group
        keeps its compressed region between layers; when the total exceeds WS_TOTAL the groups share one buffer and
        nothing is reused)."""
        if ratio not in self.gws:
            gp = self.group_plan(ratio)
            areas = [(b - a) * M * 512 for a, b, N, M in gp]
            disjoint = len(gp) == 1 or sum(areas) * 2 <= WS_TOTAL
            need = sum(areas) if disjoint else max(areas)
            if self.buf is None or self.buf.numel() < need:
                self.buf = torch.empty(need, dtype=bf, device=self.dev)
                self.gws.clear()
                self.ws_ratio = self.comp_src = None
            views, off = [], 0
            for (a, b, N, M), ar in zip(gp, areas):
                views.append(self.buf[off: off + ar].view(b - a, M, 512))
                off += ar if disjoint else 0
            self.gws[ratio] = (views, disjoint)
        return self.gws[ratio]


def to_local(pl: Plan, ti: torch.Tensor, ratio: int) -> torch.Tensor:
    """The rows' global top-512 slots (page * rpp + offset, -1 = none) -> local compressed positions (block * rpp + offset)."""
    rpp = PAGE_TOKENS // ratio
    g = ti.reshape(pl.n, TOPK)
    gl = g.long().clamp_min(0)
    loc = pl.inv[gl // rpp].long() * rpp + gl % rpp
    torch.where((g >= 0) & (pl.inv[gl // rpp] >= 0), loc.to(i32), torch.full_like(g, -1), out=pl.loc)
    return pl.loc


def attend(pl: Plan, ratio: int, q: torch.Tensor, out: torch.Tensor, k_swa: torch.Tensor, k_comp, ti, scale: float,
           sink: torch.Tensor, topk_local: torch.Tensor | None = None, src: int | None = None,
           isrc: int | None = None) -> torch.Tensor:
    """Production's _forward_prefill for the plan's rows (q / out [n, 64, 512] bf16). Returns out.

    src / isrc (optional, the layer's kv source and index source): within one step the compressed region of a source and
    the index lists of an index source are the same for every layer that reads them (written once, by the source /
    index layer, before its attention), so with them given the compressed gather runs once per source and the
    combine once per (ratio, index source); the window gather and the attention run per layer. Same values, same bits
    (production gathers and combines per layer)."""
    gather, combine, sparse_fwd = _ops()
    assert q.shape == (pl.n, 64, 512) and out.shape == (pl.n, 64, 512)
    k_swa3 = _view3(k_swa)
    top_k = TOPK if ratio else 0           # SWA-only layers: production passes TOP_K = 0 (the lists are never read)
    cw = -(-(top_k + WIN) // 128) * 128
    if cw not in pl.comb:
        pl.comb[cw] = torch.empty(pl.n, cw, dtype=i32, device=pl.dev)
        pl.clen[cw] = torch.empty(pl.n, dtype=i32, device=pl.dev)
    comb, clen = pl.comb[cw], pl.clen[cw]
    ckey = (ratio, isrc) if (ratio == 0 or (isrc is not None and topk_local is None)) else None
    redo_comb = ckey is None or pl.comb_key.get(cw) != ckey
    if ratio:
        assert k_comp is not None
        k_comp3 = _view3(k_comp)
        if redo_comb:
            loc = topk_local if topk_local is not None else to_local(pl, ti, ratio)
    else:
        loc = pl.loc
    views, disjoint = pl.group_ws(ratio)
    if pl.ws_ratio != ratio:               # another ratio's layout overwrote the buffer
        pl.ws_ratio, pl.comp_src = ratio, None
    redo_comp = bool(ratio) and (src is None or not disjoint or pl.comp_src != src)
    for g, (a, b, N, M) in enumerate(pl.group_plan(ratio)):
        kv = views[g]
        if redo_comp:
            gather(kv, k_comp3, seq_lens=pl.seq_lens[a:b] // ratio, gather_lens=None, block_table=pl.bt[a:b],
                   block_size=PAGE_TOKENS // ratio, offset=0)
        gather(kv, k_swa3, seq_lens=pl.seq_lens[a:b], gather_lens=pl.gather_lens[a:b], block_table=pl.wbt[a:b],
               block_size=WPT, offset=N)
        q0, q1 = pl.qsl_h[a], pl.qsl_h[b]
        if redo_comb:
            combine(loc[q0:q1], pl.qsl[a: b + 1], pl.seq_lens[a:b], pl.gather_lens[a:b], WIN, ratio, top_k, M, N,
                    out=(comb[q0:q1], clen[q0:q1]))
        sparse_fwd(q=q[q0:q1], kv=kv.view(-1, 1, 512), indices=comb[q0:q1].unsqueeze(1), sm_scale=scale, attn_sink=sink,
                   topk_length=clen[q0:q1], out=out[q0:q1])
    if ratio:
        pl.stats["comp_gathers" if redo_comp else "comp_reused"] += 1
        pl.comp_src = src if (disjoint and src is not None) else None
    pl.stats["combines" if redo_comb else "comb_reused"] += 1
    pl.comb_key[cw] = ckey
    return out
