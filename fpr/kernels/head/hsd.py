"""HSD: LM head + sampler + DSpark drafter of the 4 ranks behind serve/interfaces.py's HeadSamplerDrafter contract.

One object drives the 4 GPUs (one process, like serve/worker.py). Per step:
    hsd.prepare(batches, states, bitmasks=None)      # host: per-rank StepBatch -> static device tables (outside graphs)
    hsd.run_all(h_rows, aux_rows)                     # eager, stage-interleaved over the ranks (or EMU on one device)
  or, per rank (one CUDA graph per rank and bucket; peer mode only): for r: hsd.run(r, h_rows[r], aux_rows[r])
    res = hsd.result(r)                               # host StepResult (after the step's sync)
    hsd.next_ids[r][:n_seq], hsd.num_emitted[r][:n_seq]   # device, batch.seqs order
The contract (inputs, outputs, what HSD needs from the KV manager) is serve/interfaces.py's HeadSamplerDrafter.

Row layout inside HSD (all ranks): the sample rows of rank r occupy [r * S_pad, r * S_pad + n_s(r)) of the all-gathered
verify rows (S_pad = the bucket of max_r n_s), the draft rows of rank r occupy [r * D_pad, (r + 1) * D_pad) (D_pad = the
drafting-sequence bucket); a request = one sampled sequence (decode: k + 1 rows, finishing prefill: 1 row), requests in
rank-major order. Sampler state (draft distribution q, token logit, lse, temperature, seed) lives at the GLOBAL slot
r * (max_seqs + 1) + SeqEntry.state_slot (stable while the sequence stays on its rank); slot r * (max_seqs + 1) + max_seqs
is rank r's sink (padding).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from operator import itemgetter as _itemgetter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import comm_head  # noqa: E402
import drafter as DR  # noqa: E402
import head as HD  # noqa: E402
import sampler as S  # noqa: E402

V, H, W = 129280, 5120, 4
GLUE = os.environ.get("HSD_GLUE", "1") not in ("0", "")  # fused glue kernels (0: the torch-op version, same values)
NOISE_SIDE = os.environ.get("HSD_NOISE_SIDE", "1") not in ("0", "")  # draft noise + slot-0 bias on a side stream
CTX_SIDE = os.environ.get("HSD_CTX_SIDE", "1") not in ("0", "")  # drafter context KV concurrently with the verification
# prepare() / tables / result(): "1" = vectorized host build into one pinned blob per rank + one H2D copy;
# "0" = the per-element version (validated on 4 GPUs). HSD(prep=...) overrides; default from env.
PREP_DEFAULT = os.environ.get("HSD_PREP", "0")
NBW = 6  # logical 32-token window blocks per drafting sequence covering [Q - 128, Q + k)
BM_WORDS = (V + 31) // 32


def _gather(lst, idx):
    """[lst[i] for i in idx] at C speed (operator.itemgetter; a tuple, also for one index)."""
    if len(idx) == 1:
        return (lst[idx[0]],)
    return _itemgetter(*idx)(lst)


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _bucket(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"{n} exceeds the largest bucket {buckets[-1]}")


@dataclass
class _Step:
    n_s: list          # sample rows per rank
    S_pad: int
    D_pad: int         # drafting sequences per rank (bucket)
    n_req: int         # requests (all ranks)
    T: int             # verify rows (all ranks, padded)
    dseqs: list        # per rank: [(seq index in batch.seqs, request index)]
    rids: list         # per rank: rid per drafting sequence
    n_seq: list        # len(batch.seqs) per rank
    n_t: list          # tail rows per rank
    step: list


class HSD:
    def __init__(self, devs, reader, cfg, window_pools=None, max_seqs=80, k=3, draft_head="e4m3", moe_mode="route2",
                 seq_buckets=(8, 16, 28, 32, 48, 64, 80), mode="peer", ncta=8, max_tail_rows=1024, prep=None,
                 fuse_rs=None, mhc_split=None, mhc2=None):
        """window_pools: per rank an object with window_base(layer) / window_view(layer) (serve/worker.py Pools) whose
        layers 40..42 hold the drafter KV; None: private rings (tests). mode 'peer' (4 GPUs, comm.cu peer stores) or
        'emu' (the 4 ranks on devs[r] that may all be one device, collectives by copies, run_all only)."""
        self.devs = [torch.device(d) for d in devs]
        self.k, self.max_seqs, self.mode, self.ncta = k, max_seqs, mode, ncta
        self.MR = W * (max_seqs + 1)
        self.seq_buckets = tuple(b for b in seq_buckets if b <= max_seqs) + (() if max_seqs in seq_buckets else (max_seqs,))
        self.row_buckets = tuple(b * (k + 1) for b in self.seq_buckets)
        loc = V // W
        self.loc = loc
        headw = reader.get("head.weight")
        w1 = reader.get("mtp.2.markov_head.embed.weight")
        w2 = reader.get("mtp.2.markov_head.head.weight")
        emb = reader.get("embed.weight")
        S_max, D_max = self.row_buckets[-1], self.seq_buckets[-1]
        self.vhead, self.dhead, self.st, self.emb, self.ranks0 = [], [], [], [], []
        self.drk = {}  # (rank, D_pad) -> DraftRank
        for r, dv in enumerate(self.devs):
            with torch.cuda.device(dv):
                wr = headw[r * loc:(r + 1) * loc].to(dv)
                self.vhead.append(HD.VerifyHead(wr, W * S_max))
                self.dhead.append(HD.DraftHead(wr, W * D_max * k, draft_head))
                st = S.RankSampler.create(r, W, w1, w2, self.MR, k, dv)
                st.temp.fill_(1.0)
                self.st.append(st)
                self.emb.append(emb.to(dv))
                pool = None
                if window_pools is not None:
                    from geometry import WINDOW_PAGE_BYTES
                    pool = dict(page_stride=WINDOW_PAGE_BYTES,
                                base=[window_pools[r].window_base(40 + i) for i in range(3)],
                                view=[window_pools[r].window_view(40 + i) for i in range(3)])
                self.ranks0.append(DR.DraftRank(r, dv, reader, cfg, D_max, k, ctx_rows=max(S_max, max_tail_rows),
                                                moe_mode=moe_mode, pool=pool))
        self.pool = window_pools is not None
        # drafter drop-ins (bit-identical, flags): the MoE RS wait + sum folded into the next mHC boundary; the mHC split
        # boundary (mixes on a side stream); moe_mode "v4" = moe_v4's drafter MoE. None: drafter.py's env defaults.
        for d in self.ranks0:
            if fuse_rs is not None:
                d.fuse_rs = bool(fuse_rs)
            if mhc_split is not None:
                d.mhc_split = bool(mhc_split)
            if mhc2 is not None:
                d.mhc2 = bool(mhc2)
        # persistent per-slot ids: [last token, d_0 .. d_{k-1}] of every slot (this rank's view of all slots)
        z = lambda dv, *s, dt=torch.int64: torch.zeros(*s, dtype=dt, device=dv)  # noqa: E731
        self.by_slot = [z(dv, self.MR, k + 1) for dv in self.devs]
        self.next_ids = [z(dv, max_seqs + 1, k + 1) for dv in self.devs]  # row max_seqs = sink of padded entries
        self.num_emitted = [z(dv, max_seqs + 1, dt=torch.int32) for dv in self.devs]
        self.temp_h = torch.ones(self.MR, dtype=torch.float32)
        self.seed_h = torch.zeros(self.MR, dtype=torch.int64)
        self.noise = [torch.zeros(k, W * D_max, loc, device=dv) for dv in self.devs]
        self.h_all = [torch.empty(W * S_max, H, dtype=torch.bfloat16, device=dv) for dv in self.devs]
        self.h_own = [torch.zeros(S_max, H, dtype=torch.bfloat16, device=dv) for dv in self.devs]
        self.hd_all = [torch.empty(W * D_max * k, H, dtype=torch.bfloat16, device=dv) for dv in self.devs]
        self.bm = [None] * W
        self.bm_any = False
        self.gather = comm_head.PeerGather(self.devs, ncta={"h": ncta, "hd": ncta}) if mode == "peer" else None
        self.dcomm = {}
        self.cur = None
        self.tab = [None] * W
        self._tab_cache = {}
        self.side = [torch.cuda.Stream(device=dv) for dv in self.devs]
        self.side_ctx = [torch.cuda.Stream(device=dv) for dv in self.devs]
        self.prep = str(PREP_DEFAULT if prep is None else prep)
        self._res = [None] * W
        self.marks, self.mark_rank = None, 0  # per-stage CUDA events of one rank (bench_hsd1.py); None: off
        if self.prep != "0":
            self._init_tables()

    def bind_pools(self, window_pools):
        """Attach the window pools after construction (= passing window_pools to HSD()): the drafter KV of every bucket
        moves to layers 40..42 of each rank's pool (window_base / window_view, page stride WINDOW_PAGE_BYTES). Call
        before the first prepare() that should use them, outside any graph capture; graphs captured before must be
        recaptured (the cache pointers change)."""
        from geometry import WINDOW_PAGE_BYTES
        for r in range(W):
            base = self.ranks0[r]
            base.page_stride = WINDOW_PAGE_BYTES
            base.cache = [window_pools[r].window_base(40 + i) for i in range(3)]
            base.k_cache = [window_pools[r].window_view(40 + i) for i in range(3)]
        for (r, _), d in self.drk.items():
            b0 = self.ranks0[r]
            d.page_stride, d.cache, d.k_cache = b0.page_stride, b0.cache, b0.k_cache
        self.pool = True

    # ------------------------------------------------------------------ host: StepBatch -> device tables
    def draft_rank(self, r, D_pad):
        key = (r, D_pad)
        if key not in self.drk:
            base = self.ranks0[r]
            if D_pad == base.n:
                self.drk[key] = base
            else:
                pool = dict(page_stride=base.page_stride, base=base.cache, view=base.k_cache) if self.pool else None
                self.drk[key] = DR.DraftRank(r, base.dev, None, _CfgOf(base), D_pad, self.k, ctx_rows=base.b["aux"].shape[0],
                                             moe_mode=base.moe_mode, shared=base, pool=pool)
            d = self.drk[key]
            if d is not base:
                d.fuse_rs, d.mhc_split, d.mhc2 = base.fuse_rs, base.mhc_split, base.mhc2
                if not self.pool:
                    d.cache, d.k_cache, d.page_stride = base.cache, base.k_cache, base.page_stride
                for nm in ("aux", "mx_raw", "main_x", "kvc", "kv_ctx", "dq", "ctx_pos", "ctx_slot"):
                    d.b[nm] = base.b[nm]  # context-KV buffers shared by every bucket (the caller writes aux once)
        return self.drk[key]

    # ------------------------------------------------------------------ persistent tables: one byte blob per rank
    # Every table sits at a fixed offset sized for the largest bucket; a bucket's tables are prefixes. One pinned host
    # blob per rank (double-buffered) -> one H2D copy per rank and step; all buckets' graphs read the same addresses.
    # The first section (SHARED) is identical on every rank (built once, copied); the rest is rank-specific.
    _SPEC = (("cs", np.int32, "WD", 1), ("ce", np.int32, "WD", 1), ("req_idx", np.int32, "WD", 1),
             ("g_dslot", np.int64, "WD", 1), ("g_p0", np.int64, "WD", 1), ("row_req", np.int32, "T", 1),
             ("row_local", np.int32, "T", 1), ("pos", np.int64, "T", 1), ("tok_slot", np.int64, "T", 1),
             ("tok_j", np.int64, "T", 1), ("temp", np.float32, "MR", 1), ("seeds", np.int64, "MR", 1),
             # ---- rank-specific
             ("wbase", np.int64, "D", 1), ("wpages", np.int64, "D", NBW), ("rstart", np.int64, "D", 1),
             ("lslot", np.int64, "D", 1), ("seq_of_d", np.int64, "D", 1), ("tpos", np.int64, "NT", 1),
             ("tslot", np.int64, "NT", 1))
    _SHARED_LAST = "seeds"
    _TDT = {np.dtype(np.int32): torch.int32, np.dtype(np.int64): torch.int64, np.dtype(np.float32): torch.float32}

    def _init_tables(self):
        S_max, D_max = self.row_buckets[-1], self.seq_buckets[-1]
        self._dims = dict(T=W * S_max, WD=W * D_max, D=D_max, NT=self.ranks0[0].b["aux"].shape[0], MR=self.MR)
        off, lay = 0, {}
        for nm, dt, dim, cols in self._SPEC:
            n = self._dims[dim] * cols
            lay[nm] = (off, np.dtype(dt), dim, cols)
            off += -(-n * np.dtype(dt).itemsize // 64) * 64
            if nm == self._SHARED_LAST:
                self._shared_end = off
        self._lay, self._nbytes = lay, off
        pin = torch.cuda.is_available()
        self._hblob = [[torch.zeros(off, dtype=torch.uint8, pin_memory=pin) for _ in range(W)] for _ in range(2)]
        self._hv = [[self._views(self._hblob[i][r], numpy=True) for r in range(W)] for i in range(2)]
        self._hev = [[None] * W for _ in range(2)]
        self._hi = 0
        self._dblob = [torch.zeros(off, dtype=torch.uint8, device=dv) for dv in self.devs]
        self._dv = [self._views(self._dblob[r]) for r in range(W)]
        self._temp_np = np.ones(self.MR, np.float32)
        self._seed_np = np.zeros(self.MR, np.int64)
        self._slot_key = [None] * self.MR  # the state object whose temperature / seed the slot holds
        for i in range(2):
            for r in range(W):
                self._hv[i][r]["temp"][:] = 1.0
        z = lambda dv, n, fill, dt=torch.int64: torch.full((n,), fill, dtype=dt, device=dv)  # noqa: E731
        self._dscratch = [dict(draft_sampled=z(dv, self._dims["T"], -1), ar_k=torch.arange(self.k, device=dv),
                               ar_w=torch.arange(DR.WINDOW + self.k, device=dv)) for dv in self.devs]
        self._bm_d = [None] * W
        self._bm_h = None
        self._res_h = None
        self._res = [None] * W
        for r in range(W):  # the sampler's temperature / seeds = blob views (fixed addresses for every graph)
            with torch.cuda.device(self.devs[r]):
                self._dblob[r].copy_(self._hblob[0][r])
            self.st[r].temp = self._dv[r]["temp"]
            self.st[r].seeds = self._dv[r]["seeds"]

    def _views(self, blob, numpy=False):
        out = {}
        src = blob.numpy() if numpy else blob
        for nm, (off, dt, dim, cols) in self._lay.items():
            n = self._dims[dim] * cols
            if numpy:
                v = src[off:off + n * dt.itemsize].view(dt)
            else:
                v = src[off:off + n * dt.itemsize].view(self._TDT[dt])
            out[nm] = v.reshape(self._dims[dim], cols) if cols > 1 else v
        return out

    def _tables(self, r, key):
        """Rank r's device tables for a bucket key = (S_pad, D_pad, NT_pad): prefixes of the rank's blob views."""
        if self.prep == "0":
            return self._tables_v0(r, key)
        k_ = (r, key)
        if k_ not in self._tab_cache:
            S_pad, D_pad, NT = key
            n_of = dict(T=W * S_pad, WD=W * D_pad, D=D_pad, NT=NT, MR=self.MR)
            v, sc = self._dv[r], self._dscratch[r]
            tb = {nm: v[nm][:n_of[dim]] for nm, (_, _, dim, _) in self._lay.items() if nm not in ("temp", "seeds")}
            tb.update(draft_sampled=sc["draft_sampled"][:W * S_pad], ar_k=sc["ar_k"], ar_w=sc["ar_w"], bm=None, use_bm=False)
            self._tab_cache[k_] = tb
        return self._tab_cache[k_]

    def _tables_v0(self, r, key):
        """Persistent device tables of rank r for a bucket key = (S_pad, D_pad, NT_pad): fixed addresses, so a graph
        captured for the bucket replays with what prepare() copies in."""
        k_ = (r, key)
        if k_ not in self._tab_cache:
            S_pad, D_pad, NT = key
            T, WD = W * S_pad, W * D_pad
            dv = self.devs[r]
            i32, i64 = torch.int32, torch.int64
            z = lambda n, dt, fill=0: torch.full((n,), fill, dtype=dt, device=dv)  # noqa: E731
            tb = dict(cs=z(WD, i32), ce=z(WD, i32), req_idx=z(WD, i32), row_req=z(T, i32), row_local=z(T, i32),
                      pos=z(T, i64), tok_slot=z(T, i64), tok_j=z(T, i64, -1), g_dslot=z(WD, i64, -1), g_p0=z(WD, i64),
                      wbase=z(D_pad, i64), wpages=torch.full((D_pad, NBW), -1, dtype=i64, device=dv), rstart=z(D_pad, i64),
                      lslot=z(D_pad, i64, self.max_seqs), seq_of_d=z(D_pad, i64, self.max_seqs), tpos=z(NT, i64),
                      tslot=z(NT, i64, -1), draft_sampled=z(T, i64, -1), bm=None,
                      ar_k=torch.arange(self.k, device=dv), ar_w=torch.arange(DR.WINDOW + self.k, device=dv))
            self._tab_cache[k_] = tb
        return self._tab_cache[k_]

    def prepare(self, batches, states, bitmasks=None):
        """batches: the 4 ranks' StepBatch; states: dict rid -> sampling state (temperature, seed, drafts_valid,
        optional num_acceptable_drafts); bitmasks: None or per rank a [n_s, BM_WORDS] int32 CPU tensor (sample_rows
        order) or None. Host work (numpy, vectorized over rows) into a pinned blob per rank + one H2D copy per rank,
        outside any graph capture; no host sync."""
        if self.prep == "0":
            return self._prepare_v0(batches, states, bitmasks)
        hb = self._build(batches, states, bitmasks)
        if hb is None:  # rows not in the layout the vectorized build assumes: the per-element version
            return self._prepare_v0(batches, states, bitmasks)
        self._upload(hb)

    def _build(self, batches, states, bitmasks):
        """Host half of prepare: fills host blob set self._hi; returns it (None: fall back to _prepare_v0)."""
        k, MS1, MS = self.k, self.max_seqs + 1, self.max_seqs
        per = []
        for b in batches:
            srl = b.sample_rows
            n = len(srl)
            # pure decode: every row is sampled, in order -> no per-row gathers
            all_rows = n == len(b.positions) and (n == 0 or (srl[0] == 0 and srl[-1] == n - 1))
            sr = np.arange(n, dtype=np.int64) if all_rows else np.asarray(srl, np.int64)
            ss = np.asarray(b.seq_slot if all_rows else _gather(b.seq_slot, srl), np.int64) if n else sr
            if n > 1 and ((not all_rows and np.any(sr[1:] <= sr[:-1])) or np.any(ss[1:] < ss[:-1])):
                return None  # sample rows not ascending or a sequence's rows not contiguous
            first = np.flatnonzero(np.diff(ss, prepend=-1)) if n else sr
            per.append((sr, ss, first, all_rows))
        n_s = [len(x[0]) for x in per]
        S_pad = _bucket(max(max(n_s), 1), self.row_buckets)
        D_pad = _bucket(max(max(len(x[2]) for x in per), 1), self.seq_buckets)
        n_t = [len(b.tail_rows) for b in batches]
        NT = _bucket(max(max(n_t), 1), self.tail_buckets(S_pad))
        key, T, WD = (S_pad, D_pad, NT), W * S_pad, W * D_pad
        hi = self._hi
        for r in range(W):  # the host blob of this set must not be in flight (its copy was two prepares ago)
            ev = self._hev[hi][r]
            if ev is not None:
                ev.synchronize()
        hv = self._hv[hi]
        h0 = hv[0]
        cs, ce, req_idx, g_dslot, g_p0 = h0["cs"], h0["ce"], h0["req_idx"], h0["g_dslot"], h0["g_p0"]
        row_req, row_local, pos, tok_slot, tok_j = h0["row_req"], h0["row_local"], h0["pos"], h0["tok_slot"], h0["tok_j"]
        tnp, snp, skey = self._temp_np, self._seed_np, self._slot_key
        pool = self.pool
        rids, dmap = [None] * WD, []
        for r, b in enumerate(batches):
            sr, ss, first, all_rows = per[r]
            n, nd = len(sr), len(first)
            sink = r * MS1 + MS
            q0, r0 = r * D_pad, r * S_pad
            seqs = b.seqs
            dsi = ss[first].tolist()
            cnt = np.diff(np.append(first, n)) if nd else first
            pos_s = np.asarray(b.positions if all_rows else _gather(b.positions, b.sample_rows), np.int64) if n else sr
            p0 = pos_s[first]
            wb = np.maximum(p0 + 1 - DR.WINDOW, 0) // 32
            wbl, p0l = wb.tolist(), p0.tolist()
            qmax = (p0 + cnt + k - 1).tolist()
            slots, valid, lim, rst, wrow = [0] * nd, [False] * nd, [k] * nd, [0] * nd, []
            # ---- the one per-sequence loop: sampling state, window pages, checks
            for j, si in enumerate(dsi):
                se = seqs[si]
                sl = se.state_slot
                if sl >= MS:
                    raise AssertionError("state_slot >= HSD max_seqs")
                g = r * MS1 + sl
                slots[j] = g
                rid = se.rid
                rids[q0 + j] = rid
                st_ = states[rid]
                valid[j] = se.kind == "decode" and bool(getattr(st_, "drafts_valid", True))
                na = getattr(st_, "num_acceptable_drafts", None)
                if na is not None:
                    lim[j] = int(na)
                if skey[g] is not st_:  # a new state object in this slot: read its temperature / seed (kept alive here)
                    skey[g] = st_
                    tnp[g] = float(getattr(st_, "temperature", 1.0))
                    snp[g] = int(getattr(st_, "seed", 0))
                wg = se.window_table.get
                w0 = wbl[j]
                wrow += (wg(w0, -1), wg(w0 + 1, -1), wg(w0 + 2, -1), wg(w0 + 3, -1), wg(w0 + 4, -1), wg(w0 + 5, -1))
                rs_ = se.replay_start
                rst[j] = rs_
                if pool:
                    lo = rs_ if rs_ > 0 else 0
                    for p_ in (p0l[j], qmax[j]):
                        if p_ >= lo and wg(p_ // 32, -1) < 0:
                            raise AssertionError(f"{rid}: window page for position {p_} missing (allocate the window "
                                                 f"to start + 2k + 1)")
            dmap.append(list(zip(dsi, range(q0, q0 + nd))))
            gs = np.asarray(slots, np.int64)
            # requests (drafting sequences) of rank r; dummies: empty span at the rank's first row, sink slot
            cs[q0:q0 + nd] = r0 + first
            ce[q0:q0 + nd] = r0 + first + cnt
            req_idx[q0:q0 + nd] = gs
            g_dslot[q0:q0 + nd] = gs
            g_p0[q0:q0 + nd] = p0
            cs[q0 + nd:q0 + D_pad] = r0
            ce[q0 + nd:q0 + D_pad] = r0
            req_idx[q0 + nd:q0 + D_pad] = sink
            g_dslot[q0 + nd:q0 + D_pad] = -1
            g_p0[q0 + nd:q0 + D_pad] = 0
            # verify rows of rank r: [r0, r0 + n) real, [r0 + n, r0 + S_pad) pad (sink)
            rq = np.repeat(gs, cnt)
            jj = np.arange(n, dtype=np.int64) - np.repeat(first, cnt)
            ok = (jj == 0) | (np.repeat(np.asarray(valid, dtype=bool), cnt)
                              & (jj <= np.repeat(np.asarray(lim, np.int64), cnt)))
            row_req[r0:r0 + n] = rq
            row_local[r0:r0 + n] = jj
            pos[r0:r0 + n] = pos_s
            tok_slot[r0:r0 + n] = rq
            tok_j[r0:r0 + n] = np.where(ok, jj, -1)
            row_req[r0 + n:r0 + S_pad] = sink
            row_local[r0 + n:r0 + S_pad] = 0
            pos[r0 + n:r0 + S_pad] = 0
            tok_slot[r0 + n:r0 + S_pad] = 0
            tok_j[r0 + n:r0 + S_pad] = -1
            # rank-specific: drafting tables (D_pad) and tail tables (NT)
            h = hv[r]
            wp = h["wpages"]
            if nd:
                wp[:nd] = np.asarray(wrow, np.int64).reshape(nd, NBW)
            wp[nd:D_pad] = -1
            h["wbase"][:nd] = wb
            h["wbase"][nd:D_pad] = 0
            h["rstart"][:nd] = rst
            h["rstart"][nd:D_pad] = 0
            h["lslot"][:nd] = gs - r * MS1
            h["lslot"][nd:D_pad] = MS
            h["seq_of_d"][:nd] = dsi
            h["seq_of_d"][nd:D_pad] = MS
            nt = n_t[r]
            if nt:
                trl = b.tail_rows
                t_all = nt == len(b.positions) and trl[0] == 0 and trl[-1] == nt - 1
                if t_all and all_rows:
                    tp = pos_s
                else:
                    tp = np.asarray(b.positions if t_all else _gather(b.positions, trl), np.int64)
                h["tpos"][:nt] = tp
                if pool:
                    h["tslot"][:nt] = b.window_slot if t_all else _gather(b.window_slot, trl)
                else:  # private ring: state_slot * RING + pos % RING
                    ssl = np.asarray([se.state_slot for se in seqs], np.int64)
                    tss = np.asarray(b.seq_slot if t_all else _gather(b.seq_slot, trl), np.int64)
                    h["tslot"][:nt] = ssl[tss] * DR.RING + tp % DR.RING
            h["tpos"][nt:NT] = 0
            h["tslot"][nt:NT] = -1
        h0["temp"][:] = tnp
        h0["seeds"][:] = snp
        b0 = self._hblob[hi][0].numpy()
        for r in range(1, W):  # the shared section is identical on every rank
            self._hblob[hi][r].numpy()[: self._shared_end] = b0[: self._shared_end]
        self.cur = _Step(n_s, S_pad, D_pad, WD, T, dmap, rids, [len(b.seqs) for b in batches], n_t,
                         [b.step for b in batches])
        self.cur.NT = NT
        has_bm = bitmasks is not None and any(x is not None for x in bitmasks)
        self.bm_any = has_bm
        if has_bm:
            if self._bm_h is None:
                self._bm_h = torch.empty(self._dims["T"], BM_WORDS, dtype=torch.int32, pin_memory=torch.cuda.is_available())
            bm = self._bm_h.numpy()
            for r2, x in enumerate(bitmasks):
                ns_ = 0 if x is None else x.shape[0]
                if ns_:
                    bm[r2 * S_pad: r2 * S_pad + ns_] = x.numpy() if hasattr(x, "numpy") else x
                bm[r2 * S_pad + ns_: (r2 + 1) * S_pad] = -1  # all allowed
        self._res = [None] * W
        return (hi, key, T, has_bm)

    def _upload(self, hb):
        """Device half of prepare: one H2D copy per rank (+ the bitmask rows if any), stream-ordered on each device's
        current stream; the per-bucket table views; first use of a bucket builds its DraftRank / PeerComm."""
        hi, key, T, has_bm = hb
        D_pad = key[1]
        for r in range(W):
            dv = self.devs[r]
            cuda = dv.type == "cuda"
            with torch.cuda.device(dv) if cuda else _nullctx():
                self._dblob[r].copy_(self._hblob[hi][r], non_blocking=cuda)
                tb = self._tables(r, key)
                if has_bm:
                    if self._bm_d[r] is None:
                        self._bm_d[r] = torch.empty(self._dims["T"], BM_WORDS, dtype=torch.int32, device=dv)
                    self._bm_d[r][:T].copy_(self._bm_h[:T], non_blocking=cuda)
                    tb["bm"] = self._bm_d[r][:T]
                tb["use_bm"] = has_bm
                if cuda:
                    ev = self._hev[hi][r]
                    if ev is None:
                        ev = self._hev[hi][r] = torch.cuda.Event()
                    ev.record()
                self.tab[r] = tb
            d = self.draft_rank(r, D_pad)
            if d.sched is None:
                with torch.cuda.device(dv):
                    d.plan()
            if self.mode == "peer" and D_pad not in self.dcomm:
                self.dcomm[D_pad] = DR.PeerComm([self.draft_rank(x, D_pad) for x in range(W)], self.ncta)
        self._hi ^= 1

    def _prepare_v0(self, batches, states, bitmasks=None):
        """The per-element version of prepare (the original): reference for the vectorized one
        (bench_prepare.py checks equal table bytes) and the fallback for row layouts _build does not take."""
        k = self.k
        self._res = [None] * W
        n_s = [len(b.sample_rows) for b in batches]
        S_pad = _bucket(max(max(n_s), 1), self.row_buckets)
        T = W * S_pad
        dseqs = []
        for r, b in enumerate(batches):
            sr = set(b.sample_rows)
            dseqs.append([si for si, se in enumerate(b.seqs) if any(x in sr for x in range(se.row0, se.row0 + se.n_rows))])
        D_pad = _bucket(max(max(len(d) for d in dseqs), 1), self.seq_buckets)
        n_t = [len(b.tail_rows) for b in batches]
        NT = _bucket(max(max(n_t), 1), self.tail_buckets(S_pad))
        key = (S_pad, D_pad, NT)
        WD = W * D_pad
        sink_of = lambda r: r * (self.max_seqs + 1) + self.max_seqs  # noqa: E731
        # ---- host tables (identical on every rank)
        cs = torch.zeros(WD, dtype=torch.int32)
        ce = torch.zeros(WD, dtype=torch.int32)
        req_idx = torch.zeros(WD, dtype=torch.int32)
        row_req = torch.zeros(T, dtype=torch.int32)
        row_local = torch.zeros(T, dtype=torch.int32)
        pos = torch.zeros(T, dtype=torch.int64)
        tok_slot = torch.zeros(T, dtype=torch.int64)
        tok_j = torch.full((T,), -1, dtype=torch.int64)
        g_dslot = torch.full((WD,), -1, dtype=torch.int64)
        g_p0 = torch.zeros(WD, dtype=torch.int64)
        rids, dmap = [None] * WD, [[] for _ in range(W)]
        for r, b in enumerate(batches):
            srow = {row: i for i, row in enumerate(b.sample_rows)}
            row_req[r * S_pad:(r + 1) * S_pad] = sink_of(r)
            for i, row in enumerate(b.sample_rows):
                pos[r * S_pad + i] = b.positions[row]
            for j in range(D_pad):
                q = r * D_pad + j
                if j >= len(dseqs[r]):  # dummy request: empty span at the rank's first row, sink slot
                    cs[q] = ce[q] = r * S_pad
                    req_idx[q] = sink_of(r)
                    continue
                si = dseqs[r][j]
                se = b.seqs[si]
                assert se.state_slot < self.max_seqs, "state_slot >= HSD max_seqs"
                gslot = r * (self.max_seqs + 1) + se.state_slot
                rows = [r * S_pad + srow[x] for x in range(se.row0, se.row0 + se.n_rows) if x in srow]
                cs[q], ce[q], req_idx[q] = rows[0], rows[-1] + 1, gslot
                g_dslot[q], g_p0[q] = gslot, b.positions[b.sample_rows[rows[0] - r * S_pad]]
                rids[q] = se.rid
                dmap[r].append((si, q))
                st_ = states[se.rid]
                valid = bool(getattr(st_, "drafts_valid", True)) and se.kind == "decode"
                nacc = getattr(st_, "num_acceptable_drafts", None)
                lim = k if nacc is None else int(nacc)
                for jj, row in enumerate(rows):
                    row_req[row], row_local[row], tok_slot[row] = gslot, jj, gslot
                    tok_j[row] = jj if (jj == 0 or (valid and jj <= lim)) else -1
                self.temp_h[gslot] = float(getattr(st_, "temperature", 1.0))
                self.seed_h[gslot] = int(getattr(st_, "seed", 0))
        self.cur = _Step(n_s, S_pad, D_pad, WD, T, dmap, rids, [len(b.seqs) for b in batches], n_t,
                         [b.step for b in batches])
        self.cur.NT = NT
        has_bm = bitmasks is not None and any(x is not None for x in bitmasks)
        self.bm_any = has_bm  # True: this step uses the bitmask kernel variant (capture it separately)
        bm_h = None
        if has_bm:
            bm_h = torch.full((T, BM_WORDS), -1, dtype=torch.int32)  # all allowed by default
            for r2, x in enumerate(bitmasks):
                if x is not None:
                    bm_h[r2 * S_pad: r2 * S_pad + x.shape[0]] = x
        # ---- per rank: local drafting tables + copies into the persistent tables
        for r, b in enumerate(batches):
            dv = self.devs[r]
            wbase = torch.zeros(D_pad, dtype=torch.int64)
            wpages = torch.full((D_pad, NBW), -1, dtype=torch.int64)
            rstart = torch.zeros(D_pad, dtype=torch.int64)
            lslot = torch.full((D_pad,), self.max_seqs, dtype=torch.int64)
            seq_of_d = torch.full((D_pad,), self.max_seqs, dtype=torch.int64)
            for j, (si, q) in enumerate(dmap[r]):
                se = b.seqs[si]
                p0 = int(g_p0[q])
                qmin = p0 + 1
                wb = max(qmin - DR.WINDOW, 0) // 32
                wbase[j], rstart[j], seq_of_d[j], lslot[j] = wb, se.replay_start, si, se.state_slot
                for t in range(NBW):
                    wpages[j, t] = se.window_table.get(wb + t, -1)
                if self.pool:
                    qmax = p0 + (int(ce[q]) - int(cs[q])) + k - 1  # last block position at full acceptance
                    for p_ in (qmin - 1, qmax):
                        if p_ >= max(se.replay_start, 0):
                            assert se.window_table.get(p_ // 32, -1) >= 0, (
                                f"{se.rid}: window page for position {p_} missing (allocate the window to start + 2k + 1)")
            tpos = torch.zeros(NT, dtype=torch.int64)
            tslot = torch.full((NT,), -1, dtype=torch.int64)
            tp = [b.positions[x] for x in b.tail_rows]
            tpos[: len(tp)] = torch.tensor(tp or [0], dtype=torch.int64)[: len(tp)]
            if self.pool:
                ts = [b.window_slot[x] for x in b.tail_rows]
            else:  # private ring: state_slot * RING + pos % RING
                ts = [b.seqs[b.seq_slot[x]].state_slot * DR.RING + p_ % DR.RING for x, p_ in zip(b.tail_rows, tp)]
            tslot[: len(ts)] = torch.tensor(ts or [-1], dtype=torch.int64)[: len(ts)]
            with torch.cuda.device(dv):
                tb = self._tables(r, key)
                for nm, v in (("cs", cs), ("ce", ce), ("req_idx", req_idx), ("row_req", row_req), ("row_local", row_local),
                              ("pos", pos), ("tok_slot", tok_slot), ("tok_j", tok_j), ("g_dslot", g_dslot), ("g_p0", g_p0),
                              ("wbase", wbase), ("wpages", wpages), ("rstart", rstart), ("lslot", lslot),
                              ("seq_of_d", seq_of_d), ("tpos", tpos), ("tslot", tslot)):
                    tb[nm].copy_(v, non_blocking=True)
                if has_bm:
                    if tb["bm"] is None:
                        tb["bm"] = torch.empty(T, BM_WORDS, dtype=torch.int32, device=dv)
                    tb["bm"].copy_(bm_h, non_blocking=True)
                tb["use_bm"] = has_bm
                st = self.st[r]
                st.temp.copy_(self.temp_h, non_blocking=True)
                st.seeds.copy_(self.seed_h, non_blocking=True)
                self.tab[r] = tb
            d = self.draft_rank(r, D_pad)
            if d.sched is None:
                with torch.cuda.device(dv):
                    d.plan()
            if self.mode == "peer" and D_pad not in self.dcomm:
                self.dcomm[D_pad] = DR.PeerComm([self.draft_rank(x, D_pad) for x in range(W)], self.ncta)

    def tail_buckets(self, S_pad):
        mt = self.ranks0[0].b["aux"].shape[0]
        return tuple(sorted({min(S_pad + 128 * i, mt) for i in range(0, mt // 128 + 2)}))

    def _glue_bufs(self, r, n):
        key = ("glue", r)
        if key not in self._tab_cache or self._tab_cache[key]["dmi"].shape[0] < n:
            dv, k = self.devs[r], self.k
            nn = W * self.seq_buckets[-1]
            self._tab_cache[key] = dict(dmi=torch.full((nn, k), -1, dtype=torch.int32, device=dv),
                                        dmp=torch.zeros(nn, k, dtype=torch.int64, device=dv),
                                        anc=torch.zeros(nn, dtype=torch.int64, device=dv),
                                        gns=torch.ones(nn, dtype=torch.int64, device=dv))
        return self._tab_cache[key]

    def h_in(self, r):
        """Rank r's input buffer for h_rows ([S_max, 5120] bf16): write the sample rows here and call run(r) without
        tensors to keep a captured graph valid (run copies explicit tensors in)."""
        return self.h_own[r]

    def aux_in(self, r):
        return self.ranks0[r].b["aux"]

    # ------------------------------------------------------------------ device: the step of rank r (generator)
    def _meta(self, r):
        tb, c = self.tab[r], self.cur
        return S.VerifyMeta(draft_sampled=tb["draft_sampled"], pos=tb["pos"], row_req=tb["row_req"],
                            row_local=tb["row_local"], cu=tb["cs"], req_idx=tb["req_idx"], num_reqs=c.n_req,
                            bitmask=tb["bm"] if tb["use_bm"] else None, cu_end=tb["ce"])

    def _steps(self, r, h_rows, aux_rows):
        """Rank r's kernels; yields at every collective ('ag', site, inp, outs) / ('dag', D_pad) / ('drs', D_pad).
        Every size here is the bucket's (fixed for a captured graph)."""
        c, tb, st, k = self.cur, self.tab[r], self.st[r], self.k
        T, n, Dp, NT, S_pad = c.T, c.n_req, c.D_pad, c.NT, c.S_pad
        d = self.draft_rank(r, Dp)
        sts = self.st
        dv = self.devs[r]
        if h_rows is not None and h_rows.data_ptr() != self.h_own[r].data_ptr():
            self.h_own[r][: c.n_s[r]].copy_(h_rows[: c.n_s[r]])
        if aux_rows is not None and aux_rows.data_ptr() != d.b["aux"].data_ptr():
            d.b["aux"][: c.n_t[r]].copy_(aux_rows[: c.n_t[r]].reshape(c.n_t[r], 3 * H))
        if GLUE:
            _pre_glue[(triton.cdiv(max(T, NT), 1024),)](
                tb["draft_sampled"], self.by_slot[r], self.by_slot[r].stride(0), tb["tok_slot"], tb["tok_j"], T,
                d.b["ctx_pos"], d.b["ctx_slot"], tb["tpos"], tb["tslot"], NT, BLOCK=1024)
        else:
            d.b["ctx_pos"][:NT].copy_(tb["tpos"])
            d.b["ctx_slot"][:NT].copy_(tb["tslot"])
            # the verified rows' input ids from the per-slot store (-1: invalid draft / not a draft row)
            ids = self.by_slot[r][tb["tok_slot"], tb["tok_j"].clamp(min=0)]
            tb["draft_sampled"].copy_(torch.where(tb["tok_j"] >= 0, ids, -1))
        # context KV of all tail rows (decode: all k + 1 rows) needs nothing from the verification: on the side stream,
        # overlapping the verify head and the sampler's exchanges; joined before the drafter blocks (pad rows: slot -1)
        cur = torch.cuda.current_stream()
        if CTX_SIDE:
            evc = torch.cuda.Event()
            evc.record(cur)
            self.side_ctx[r].wait_event(evc)
            with torch.cuda.stream(self.side_ctx[r]):
                d.context_kv(NT)
                evc2 = torch.cuda.Event()
                evc2.record(self.side_ctx[r])
        else:
            d.context_kv(NT)
        # ---- verification
        yield ("ag", "h", self.h_own[r][:S_pad], [x[:T] for x in self.h_all])
        lg = self.vhead[r](self.h_all[r][:T])
        self._mark(r, "verify_head")
        m = self._meta(r)
        S.verify3_stats(st, lg, m)
        self._mark(r, "stats")
        yield ("ag", "g1", st.stats[:T], [x.g1(T) for x in sts])
        S.verify3_mass(st, lg, st.g1(T), m)
        yield ("ag", "g2", st.resid[:T], [x.g2(T) for x in sts])
        S.verify2_decide(st, st.g1(T), st.g2(T), m, T)
        self._mark(r, "decide")
        S.verify2_resample(st, lg, st.g1(T), m)
        yield ("ag", "g3", st.out3[:n], [x.g3(n) for x in sts])
        S.verify2_finish(st, st.g3(n), m)
        self._mark(r, "finish")
        # ---- drafter metadata from the result (GPU only); request index == drafting row (rank-major, D_pad each)
        if GLUE:
            g = self._glue_bufs(r, n)
            _post_glue[(n,)](st.num_sampled, st.last, tb["g_dslot"], tb["g_p0"], n, g["dmi"], g["dmp"], g["anc"], r * Dp, Dp,
                             tb["wbase"], tb["wpages"], tb["rstart"], tb["lslot"], d.b["q_pos"], d.b["q_slot"], d.b["idx"],
                             d.b["idx"].stride(0), g["gns"], K=k, KP=triton.next_power_of_2(k), NBW_=NBW, WIN=DR.WINDOW,
                             WBLK=triton.next_power_of_2(DR.WINDOW + k), POOL=self.pool, RING_=DR.RING, MAXS=self.max_seqs,
                             num_warps=4)
            _x0_glue[(Dp, k, triton.cdiv(H, 1024))](d.b["X0"], self.emb[r], g["anc"], r * Dp, DR.NOISE_ID, K=k, HC_=DR.HC,
                                                    D_=H, BLOCK=1024)
            anchors_all = g["anc"][:n]
            dm = S.DraftMeta(idx=g["dmi"][:n], sample_pos=g["dmp"][:n], num_reqs=n)
            gok = None
        else:
            gok = tb["g_dslot"] >= 0
            gns = torch.where(gok, st.num_sampled[:n].to(torch.int64), 1)
            live = gok[r * Dp:(r + 1) * Dp]
            ns = gns[r * Dp:(r + 1) * Dp]
            anchors_all = torch.where(gok, st.last[:n], 0)
            Q = tb["g_p0"][r * Dp:(r + 1) * Dp] + ns
            self._set_block(d, tb, Q, anchors_all[r * Dp:(r + 1) * Dp], r)
            dmi = torch.where(gok[:, None], tb["g_dslot"][:, None], -1).expand(n, k).to(torch.int32).contiguous()
            dmp = ((tb["g_p0"] + gns)[:, None] + 1 + tb["ar_k"][None, :]).contiguous()
            dm = S.DraftMeta(idx=dmi, sample_pos=dmp, num_reqs=n)
        # draft Gumbel noise of all slots on a side stream while the drafter blocks run
        ev = torch.cuda.Event()
        ev.record(cur)
        side = self.side[r] if NOISE_SIDE else cur
        side.wait_event(ev)
        with torch.cuda.stream(side):
            S.draft_noise(st, self.noise[r][:, :n], dm)
            # slot 0's Markov embedding and bias need only the anchors: also off the critical path
            torch.index_select(st.w1, 0, anchors_all, out=st.emb[0, :n])
            torch.mm(st.emb[0, :n], st.w2.t(), out=st.bias[:n])
            ev2 = torch.cuda.Event()
            ev2.record(side)
        self._mark(r, "noise+bias0")
        # ---- drafter blocks
        if CTX_SIDE:
            cur.wait_event(evc2)
        for name, kind, fn in DR.stages([d], None):
            if name == "context_kv":
                continue
            if kind == "local":
                fn(d)
                self._mark(r, name)
            elif kind == "ag":
                yield ("dag", Dp)
            else:
                yield ("drs", Dp)
        yield ("ag", "hd", d.b["hd"], [x[: n * k] for x in self.hd_all])
        b3 = self.dhead[r](self.hd_all[r][: n * k]).view(n, k, self.loc)
        self._mark(r, "draft_head")
        cur.wait_event(ev2)
        for i in range(k):
            S.draft3_local(st, b3[:, i], st.emb[i % 2, :n], dm.idx[:, i], self.noise[r][i, :n], i, 2048, bias_ready=i == 0)
            # one gather buffer per slot (sampler.RankSampler.d_gath): no peer can overwrite slot i while this rank reads it
            yield ("ag", f"d{i}", st.d_out[:n], [x.d_gath(n, i) for x in sts])
            S.draft_combine(st, st.d_gath(n, i), n, i, st.emb[(i + 1) % 2, :n], dm.idx[:, i])
        # ---- outputs: per-slot store (all ranks' drafting sequences), next ids / num emitted of this rank's sequences
        if GLUE:
            self.num_emitted[r].zero_()
            _out_glue[(n,)](self.by_slot[r], self.by_slot[r].stride(0), st.last, st.draft_tokens, st.draft_tokens.stride(0),
                            tb["g_dslot"], g["gns"], n, r * Dp, Dp, tb["seq_of_d"], self.next_ids[r],
                            self.next_ids[r].stride(0), self.num_emitted[r], MAXS1=self.max_seqs + 1, K=k,
                            KP=triton.next_power_of_2(k + 1), num_warps=1)
        else:
            sinks = (torch.arange(n, device=dv) // Dp) * (self.max_seqs + 1) + self.max_seqs
            gsl = torch.where(gok, tb["g_dslot"], sinks)
            self.by_slot[r].index_copy_(0, gsl, torch.cat([st.last[:n][:, None], st.draft_tokens[:n]], 1))
            self.next_ids[r].index_copy_(0, tb["seq_of_d"], self.by_slot[r][gsl[r * Dp:(r + 1) * Dp]])
            self.num_emitted[r].zero_()
            self.num_emitted[r].index_copy_(0, tb["seq_of_d"], torch.where(live, ns, 0).to(torch.int32))

    def _set_block(self, d, tb, Q, anchors, r):
        """Block positions Q .. Q+k-1, their window-pool slots, the ring window indices [Q - 128, Q + k) and the block
        input [anchor, noise x (k-1)] of this rank's drafting sequences."""
        k, Dp = self.k, d.n
        b = d.b
        pos = Q[:, None] + tb["ar_w"][None, :] - DR.WINDOW  # [Dp, 128 + k]
        if self.pool:
            blk = (pos // 32 - tb["wbase"][:, None]).clamp(0, NBW - 1)
            page = torch.gather(tb["wpages"], 1, blk)
            ok = (page >= 0) & (pos >= tb["rstart"][:, None]) & (pos >= 0)
            slot = torch.where(ok, page * 32 + pos % 32, -1)
        else:  # private ring (tests): slot = local state slot * RING + pos % RING
            seq = tb["lslot"][:, None]
            ok = (pos >= tb["rstart"][:, None]) & (pos >= 0) & (seq < self.max_seqs)
            slot = torch.where(ok, seq * DR.RING + pos % DR.RING, -1)
        b["q_pos"].copy_(pos[:, DR.WINDOW:].reshape(-1))
        b["q_slot"].copy_(slot[:, DR.WINDOW:].reshape(-1))
        b["idx"][:, 0, : DR.WINDOW + k].copy_(slot.to(torch.int32).repeat_interleave(k, 0))
        e0 = torch.index_select(self.emb[r], 0, anchors)
        X0 = b["X0"].view(Dp, k, DR.HC, H)
        X0[:, 0].copy_(e0[:, None, :].expand(Dp, DR.HC, H))
        X0[:, 1:].copy_(self.emb[r][DR.NOISE_ID].view(1, 1, 1, H).expand(Dp, k - 1, DR.HC, H))

    # ------------------------------------------------------------------ drivers
    def _mark(self, r, name):
        if self.marks is not None and r == self.mark_rank:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(torch.cuda.current_stream())
            self.marks.append((name, ev))

    def run_all(self, h_rows, aux_rows):
        """All ranks, stage-interleaved (eager). mode 'emu': collectives by copies (one device is fine)."""
        gens = [self._steps(r, h_rows[r], aux_rows[r]) for r in range(W)]
        while True:
            reqs = []
            for r in range(W):
                with torch.cuda.device(self.devs[r]):
                    self._mark(r, "|seg")
                    try:
                        reqs.append(next(gens[r]))
                    except StopIteration:
                        reqs.append(None)
                    q = reqs[-1]
                    self._mark(r, "end" if q is None else (q[1] if q[0] == "ag" else q[0]) + "|")
            if all(x is None for x in reqs):
                return
            kind = reqs[0][0]
            if kind == "ag":
                site = reqs[0][1]
                inps = [x[2] for x in reqs]
                outs = reqs[0][3]
                self._gather(site, inps, outs)
            elif kind == "dag":
                ranks = [self.draft_rank(r, reqs[0][1]) for r in range(W)]
                self._dag(ranks)
            else:
                ranks = [self.draft_rank(r, reqs[0][1]) for r in range(W)]
                self._drs(ranks)

    def _gather(self, site, inps, outs):
        if self.mode == "emu":
            for o in outs:
                for r, x in enumerate(inps):
                    o.reshape(W, -1)[r].copy_(x.reshape(-1))
        else:
            flags, epochs = self.gather._site(site)
            nc = self.gather.ncta.get(site, 1)
            for r in range(W):
                with torch.cuda.device(self.devs[r]):
                    self.gather.e.ag_push(inps[r], r, outs, flags, epochs[r], nc)
            for r in range(W):
                with torch.cuda.device(self.devs[r]):
                    self.gather.e.wait(flags[r], epochs[r], W * nc)

    def _dag(self, ranks):
        if self.mode == "emu":
            DR.EmuComm().ag_f(ranks)
        else:
            comm = self.dcomm[ranks[0].n]
            for r in range(W):
                with torch.cuda.device(self.devs[r]):
                    comm.ag_f_rank(r)

    def _drs(self, ranks):
        if self.mode == "emu":
            DR.EmuComm().rs(ranks)
        else:
            comm = self.dcomm[ranks[0].n]
            for r in range(W):
                with torch.cuda.device(self.devs[r]):
                    comm.rs_rank(r)

    def run(self, r, h_rows, aux_rows):
        """Rank r alone (peer mode): all its kernels on the current stream of device r; capturable per rank."""
        assert self.mode == "peer"
        Dp = self.cur.D_pad
        comm = self.dcomm[Dp]
        for req in self._steps(r, h_rows, aux_rows):
            if req[0] == "ag":
                _, site, inp, outs = req
                flags, epochs = self.gather._site(site)
                nc = self.gather.ncta.get(site, 1)
                self.gather.e.ag_push(inp, r, outs, flags, epochs[r], nc)
                self.gather.e.wait(flags[r], epochs[r], W * nc)
            elif req[0] == "dag":
                comm.ag_f_rank(r)
            else:
                comm.rs_rank(r)

    def result(self, r):
        """Host StepResult of rank r (after the step). The first call of a step copies every rank's sampled ids and
        counts into pinned buffers (stream-ordered after the step) and waits once; the other ranks reuse them."""
        if self.prep == "0":
            return self._result_v0(r)
        import interfaces as IF
        c = self.cur
        n = c.n_req
        if self._res[r] is None:
            if self._res_h is None:
                pin = torch.cuda.is_available()
                self._res_h = [(torch.zeros(self.MR, dtype=torch.int32, pin_memory=pin),
                                torch.zeros(self.MR, self.k + 1, dtype=torch.int64, pin_memory=pin)) for _ in range(W)]
            evs = []
            for r2 in range(W):
                dv, st = self.devs[r2], self.st[r2]
                cuda = dv.type == "cuda"
                with torch.cuda.device(dv) if cuda else _nullctx():
                    self._res_h[r2][0][:n].copy_(st.num_sampled[:n], non_blocking=cuda)
                    self._res_h[r2][1][:n].copy_(st.sampled[:n], non_blocking=cuda)
                    if cuda:
                        ev = torch.cuda.Event()
                        ev.record()
                        evs.append(ev)
            for ev in evs:
                ev.synchronize()
            for r2 in range(W):
                self._res[r2] = (self._res_h[r2][0][:n].tolist(), self._res_h[r2][1][:n].tolist())
        ns, smp = self._res[r]
        res = IF.StepResult(r, c.step[r])
        acc, tok = res.accepted, res.tokens
        for _, qi in c.dseqs[r]:
            rid = c.rids[qi]
            a_ = ns[qi]
            acc[rid] = a_
            tok[rid] = smp[qi][:a_]
        return res

    def _result_v0(self, r):
        """Host StepResult of rank r (one D2H copy; call after the step's sync)."""
        import interfaces as IF
        c = self.cur
        st = self.st[r]
        n = c.n_req
        ns = st.num_sampled[:n].cpu()
        smp = st.sampled[:n].cpu()
        res = IF.StepResult(r, c.step[r])
        for si, qi in c.dseqs[r]:
            rid = c.rids[qi]
            a = int(ns[qi])
            res.accepted[rid] = a
            res.tokens[rid] = smp[qi, :a].tolist()
        return res

    def rank_view(self, r):
        return _RankView(self, r)


# ============================================================================================ fused per-step glue
# The facade's metadata between the pieces as 3 Triton kernels instead of ~40 small torch ops (each ~2-4 us in a graph).
from vllm.triton_utils import tl, triton  # noqa: E402


@triton.jit
def _pre_glue(ds_ptr, by_ptr, by_stride, tok_slot_ptr, tok_j_ptr, T, ctx_pos_ptr, ctx_slot_ptr, tpos_ptr, tslot_ptr, NT,
              BLOCK: tl.constexpr):
    # draft_sampled[row] = by_slot[tok_slot[row], tok_j[row]] (-1 where tok_j < 0); ctx_pos / ctx_slot of the tail rows
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < T
    j = tl.load(tok_j_ptr + o, mask=m, other=-1)
    sl = tl.load(tok_slot_ptr + o, mask=m, other=0)
    v = tl.load(by_ptr + sl * by_stride + tl.maximum(j, 0), mask=m & (j >= 0), other=-1)
    tl.store(ds_ptr + o, tl.where(j >= 0, v, -1), mask=m)
    mt = o < NT
    tl.store(ctx_pos_ptr + o, tl.load(tpos_ptr + o, mask=mt), mask=mt)
    tl.store(ctx_slot_ptr + o, tl.load(tslot_ptr + o, mask=mt), mask=mt)


@triton.jit
def _post_glue(ns_ptr, last_ptr, g_dslot_ptr, g_p0_ptr, n, dmi_ptr, dmp_ptr, anc_ptr, r0, Dp, wbase_ptr, wpages_ptr,
               rstart_ptr, lslot_ptr, q_pos_ptr, q_slot_ptr, idx_ptr, idx_stride, gns_ptr,
               K: tl.constexpr, KP: tl.constexpr, NBW_: tl.constexpr, WIN: tl.constexpr, WBLK: tl.constexpr,
               POOL: tl.constexpr, RING_: tl.constexpr, MAXS: tl.constexpr):
    # one program per drafting row q (all ranks): draft meta (idx, sample_pos), anchors, emitted counts; for this rank's
    # rows (r0 <= q < r0 + Dp) also the block positions / slots and the ring-window indices [Q - 128, Q + K)
    q = tl.program_id(0)
    dslot = tl.load(g_dslot_ptr + q)
    ok = dslot >= 0
    ns = tl.where(ok, tl.load(ns_ptr + q).to(tl.int64), 1)
    p0 = tl.load(g_p0_ptr + q)
    tl.store(gns_ptr + q, ns)
    tl.store(anc_ptr + q, tl.where(ok, tl.load(last_ptr + q), 0))
    kk = tl.arange(0, KP)
    tl.store(dmi_ptr + q * K + kk, tl.where(ok, dslot, -1).to(tl.int32) + kk * 0, mask=kk < K)
    tl.store(dmp_ptr + q * K + kk, p0 + ns + 1 + kk, mask=kk < K)
    j = q - r0
    if (j >= 0) & (j < Dp):
        Q = p0 + ns
        w = tl.arange(0, WBLK)
        pos = Q - WIN + w
        live_w = w < WIN + K
        if POOL:
            blk = tl.minimum(tl.maximum(pos // 32 - tl.load(wbase_ptr + j), 0), NBW_ - 1)
            page = tl.load(wpages_ptr + j * NBW_ + blk, mask=live_w, other=-1)
            good = (page >= 0) & (pos >= tl.load(rstart_ptr + j)) & (pos >= 0)
            slot = tl.where(good, page * 32 + pos % 32, -1)
        else:
            seq = tl.load(lslot_ptr + j)
            good = (pos >= tl.load(rstart_ptr + j)) & (pos >= 0) & (seq < MAXS)
            slot = tl.where(good, seq * RING_ + pos % RING_, -1)
        for t in tl.static_range(K):
            tl.store(idx_ptr + (j * K + t) * idx_stride + w, slot.to(tl.int32), mask=live_w)
        # block positions / slots: entries WIN .. WIN + K - 1 of the window
        tl.store(q_pos_ptr + j * K + (w - WIN), pos, mask=(w >= WIN) & live_w)
        tl.store(q_slot_ptr + j * K + (w - WIN), slot, mask=(w >= WIN) & live_w)


@triton.jit
def _x0_glue(x0_ptr, emb_ptr, anc_ptr, r0, noise_id, K: tl.constexpr, HC_: tl.constexpr, D_: tl.constexpr,
             BLOCK: tl.constexpr):
    # X0 [Dp, K, HC, D]: slot 0 = embedding of the anchor, slots 1..K-1 = embedding of the noise token, 4 hc copies
    j = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1)
    b = tl.program_id(2)
    tok = tl.where(t == 0, tl.load(anc_ptr + r0 + j), noise_id).to(tl.int64)
    o = b * BLOCK + tl.arange(0, BLOCK)
    m = o < D_
    v = tl.load(emb_ptr + tok * D_ + o, mask=m)
    base = ((j * K + t) * HC_) * D_
    for c in tl.static_range(HC_):
        tl.store(x0_ptr + base + c * D_ + o, v, mask=m)


@triton.jit
def _out_glue(by_ptr, by_stride, last_ptr, drafts_ptr, drafts_stride, g_dslot_ptr, gns_ptr, n, r0, Dp, seq_of_d_ptr,
              next_ptr, next_stride, nemit_ptr, MAXS1: tl.constexpr, K: tl.constexpr, KP: tl.constexpr):
    # per drafting row q: by_slot[slot] = [last, drafts]; rows of this rank: next_ids[seq] = that, num_emitted[seq] = ns
    q = tl.program_id(0)
    dslot = tl.load(g_dslot_ptr + q)
    ok = dslot >= 0
    slot = tl.where(ok, dslot, (q // Dp) * MAXS1 + MAXS1 - 1)
    kk = tl.arange(0, KP)
    v = tl.where(kk == 0, tl.load(last_ptr + q), tl.load(drafts_ptr + q * drafts_stride + kk - 1, mask=(kk >= 1) & (kk <= K),
                                                             other=0))
    tl.store(by_ptr + slot * by_stride + kk, v, mask=kk <= K)
    j = q - r0
    if (j >= 0) & (j < Dp):
        si = tl.load(seq_of_d_ptr + j)
        tl.store(next_ptr + si * next_stride + kk, v, mask=kk <= K)
        tl.store(nemit_ptr + si, tl.where(ok, tl.load(gns_ptr + q), 0).to(tl.int32))


def aux_of(xn: torch.Tensor) -> torch.Tensor:
    """Drafter input of one target layer from its block-input residual stream xn [n, 4, 5120] bf16 (the attention
    boundary's Xn of layers 37 / 38 / 39): stock's previous_post.mean(dim=1) (torch mean of bf16 = fp32 sum, one bf16
    rounding) -> [n, 5120] bf16. Stack the three as aux_rows [n, 3, 5120] in tail_rows order."""
    return xn.mean(dim=1)


class FinalHidden:
    """The head input of the own rows after layer 39 (stock: norm(hc_collapse(mhc_post(x, residual, post, comb), pre))):
    one mHC boundary (kernels/sp_layer/mhc.cu, vendored) with lag = layer 39's ffn_pre and normw = norm.weight; the 24
    mixes it also computes are unused. One per device; buffers for up to max_rows rows (the boundary chunks by 128)."""

    def __init__(self, reader, cfg, dev, max_rows):
        import vendor_ext as VE
        self.dev = torch.device(dev)
        with torch.cuda.device(self.dev):
            self.mhc = VE.mhc()
            g = lambda n: reader.get(f"layers.39.{n}").to(self.dev).float()  # noqa: E731
            self.fn, self.sc, self.bs = self.mhc.prep_fn(g("hc_attn_fn")), g("hc_attn_scale"), g("hc_attn_base")
            self.nw = reader.get("norm.weight").to(self.dev).float()
            self.eps, self.hc_eps, self.iters = cfg.norm_eps, cfg.hc_eps, cfg.hc_iters
            e = lambda *s_, dt=torch.float32: torch.empty(*s_, dtype=dt, device=self.dev)  # noqa: E731
            self.xn = e(max_rows, 4, H, dt=torch.bfloat16)
            self.pre, self.post, self.comb = e(max_rows, 4), e(max_rows, 4), e(max_rows, 4, 4)
            self.out = e(max_rows, H, dt=torch.bfloat16)
            self.part = e(160 * 128 * 25)

    def __call__(self, moe_out, x_mid, f_post, f_comb, f_pre):
        """moe_out [n, 5120] bf16 (layer 39's MoE output of the own rows), x_mid [n, 4, 5120] bf16, f_post [n, 4],
        f_comb [n, 4, 4], f_pre [n, 4] (fp32: layer 39's FFN mixes) -> h [n, 5120] bf16 (a view of a persistent buffer)."""
        n = moe_out.shape[0]
        for a in range(0, n, 128):
            b = min(n, a + 128)
            self.mhc.boundary(moe_out[a:b], x_mid[a:b], f_post[a:b], f_comb[a:b], f_pre[a:b], self.fn, self.sc, self.bs,
                              self.nw, self.eps, self.hc_eps, self.iters, self.xn[a:b], self.pre[a:b], self.post[a:b],
                              self.comb[a:b], self.out[a:b], self.part)
        return self.out[:n]


def sink_slot(h, r):
    return r * (h.max_seqs + 1) + h.max_seqs


class _CfgOf:
    """Minimal cfg stand-in for DraftRank(shared=...) (weights come from the shared rank)."""

    def __init__(self, d):
        self.norm_eps, self.hc_eps, self.hc_iters, self.n_mtp_layers = d.eps, d.hc_eps, d.iters, len(d.blocks)


class _RankView:
    """serve/interfaces.py HeadSamplerDrafter for one rank (peer mode; the caller runs all 4 ranks each step)."""

    def __init__(self, hsd, r):
        self.hsd, self.r = hsd, r

    def run(self, h_rows, aux_rows, batch, sampling_state, bitmask, stream):
        with torch.cuda.stream(stream):
            self.hsd.run(self.r, h_rows, aux_rows)
