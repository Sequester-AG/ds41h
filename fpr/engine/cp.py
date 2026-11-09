"""Context-parallel (CP) prefill: one prompt's chunk over the 4 ranks (the exchange code is ported from a standalone
CP prototype).

Contract (serve -> engine): a prefill SeqEntry may carry `cp` = a CPRole with
  group  the prompt's seq_uid; rid = the prompt's rid (its host token history: helpers' Engram lookback ids),
  owner  the owner's rank,
  subs   [(rank, start, n)] in position order, the owner last (n >= 128), starts / ends multiples of 64 except the
         prompt's last row,
  me     this entry's index in subs.
The owner's entry is the prompt's own prefill chunk cut to its sub-chunk (real tables, tail rows, sample row). A
helper's entry belongs to a SHADOW sequence (replica paged pages for blocks 0 .. the chunk's end, scratch window pages for
[start - 128, start + n), a scratch state slot, no tail / sample rows). StepBatch.copies carries the prefix pages, the
previous chunk's pages (X1b) and the lowest helper's window halo. Every exchange gathers with the sender's own tables
and scatters with the receiver's own (no cross-rank page maps in the batch).

Engine (EngineConfig.cp, mixed steps with CP roles): the attention piece of layers 0..20 in 3 phases, each issued for
ALL ranks before the next (so every send is issued before its receive: plain cross-device copies + events):
  P1  attn_a; the compressor of source layers; X1a send: the sub-chunk's comp / idx page sections -> every later rank
  P2  X1a receive (wait + scatter into its own copy); the indexer's scoring; q_b + RoPE + window insert;
      X2 send: the window sections of the 4 blocks before the next sub-chunk (its 127 halo rows) -> the next rank
  P3  X2 receive (wait + scatter); FlashMLA .. wo_b
Layers 21..39 (the owner's tail), mHC, router, AG / MoE (one call) / RS are unchanged.
"""
from __future__ import annotations

import os

import torch

W = 4
WIN = 128
AUDIT = os.environ.get("ENG_CP_AUDIT", "") == "1"   # debug (tests with 0xFF-poisoned pools): which pages / sections each
                                                     # rank lacks before and after a CP step (audit_before / audit_after)
# ENG_CP_MAIL_STREAM=1 (default; 0 = the old allocation, for A/Bs): exchange receive buffers come from a CP-only stream's
# allocator pool on the receiving device (see CPPlan._send).
MAIL_STREAM = os.environ.get("ENG_CP_MAIL_STREAM", "1") == "1"


def _get(o, k):
    return o[k] if isinstance(o, dict) else getattr(o, k)


class CPPlan:
    """The step's CP groups, their per-rank page / window-page index tensors and the exchange mailbox."""

    def __init__(self, eng, batches):
        self.eng = eng
        WK = eng.WK
        self.PO, self.RPP = WK.PAGE_OFFSETS, WK.RECORDS_PER_PAGE
        self.WPB, self.WLB = WK.WINDOW_PAGE_BYTES, WK.WINDOW_LAYER_PAGE_B
        self.groups = {}
        for r, sb in enumerate(batches):
            for se in sb.seqs:
                cp = getattr(se, "cp", None)
                if cp is None:
                    continue
                g = _get(cp, "group")
                gd = self.groups.setdefault(g, dict(subs=[tuple(int(v) for v in x) for x in _get(cp, "subs")], ents={}))
                gd["ents"][r] = se
        self.alias = {}          # helper rid -> the prompt's rid (Engram host history), when they differ
        for g, gd in self.groups.items():
            ranks = [x[0] for x in gd["subs"]]
            assert set(gd["ents"]) == set(ranks), f"CP group {g}: entries on ranks {sorted(gd['ents'])}, subs {ranks}"
            for r, se in gd["ents"].items():
                prid = getattr(se.cp, "rid", None) if not isinstance(se.cp, dict) else se.cp.get("rid")
                if prid is not None and se.rid != prid:
                    self.alias[se.rid] = prid
            devs = eng.devs
            gd["ranks"] = ranks
            # paged pages of every sub-chunk's blocks in every rank's own copy: pages[r][q]
            gd["pages"] = {}
            for r in ranks:
                bt = gd["ents"][r].block_table
                gd["pages"][r] = {}
                for q, st, n in gd["subs"]:
                    blocks = range(st // 64, -(-(st + n) // 64))
                    gd["pages"][r][q] = torch.tensor([bt[b] for b in blocks], dtype=torch.int64).to(devs[r])
            # window halo: the 4 blocks before each later sub-chunk's start (its 127 rows before + 1 unread)
            gd["halo_src"], gd["halo_dst"] = {}, {}
            for i in range(1, len(ranks)):
                p, q = ranks[i - 1], ranks[i]
                st = gd["subs"][i][1]
                hb = [b for b in range((st - WIN) // 32, st // 32) if b >= 0]
                if not hb:
                    continue
                gd["halo_src"][p] = torch.tensor([gd["ents"][p].window_table[b] for b in hb], dtype=torch.int64).to(devs[p])
                gd["halo_dst"][q] = torch.tensor([gd["ents"][q].window_table[b] for b in hb], dtype=torch.int64).to(devs[q])
        self.mail, self.keep = {}, []

    def __bool__(self):
        return bool(self.groups)

    # ---- pool views
    def _pg(self, r):
        p = self.eng.pools[r]
        return p.paged.view(-1, p.paged.numel() // p.n_paged)

    def _wsec(self, r):
        p = self.eng.pools[r]
        return p.window.view(p.n_window + 1, self.WPB)

    def _sections(self, L):
        return [(self.PO[f"comp{L}"], self.RPP[L] * 584), (self.PO[f"idx{L}"], self.RPP[L] * 132)]

    def _mail_stream(self, dev):
        ms = self.eng.__dict__.get("_cp_mail_streams")
        if ms is None:
            ms = self.eng._cp_mail_streams = {str(torch.device(d)): torch.cuda.Stream(device=d) for d in self.eng.devs}
        return ms[str(torch.device(dev))]

    def _send(self, key, data, dst_dev):
        # The receive buffer. Every device's current stream is its engine stream (Engine.__init__), so a plain
        # torch.empty(device=dst) takes a block of the RECEIVER's engine-stream pool: possibly one its host code freed a
        # moment ago (a temporary of its own pieces) while its queued kernels still use it. The caching allocator makes
        # that safe only for a first use ON that stream; this buffer's first use is the copy on the SENDER's stream, which
        # can land before those kernels run and then be overwritten by them (whole or partial X1a messages replaced by
        # the receiver's own data). From a CP-only stream's pool on the receiver instead: nothing ever runs on that
        # stream, and its blocks are freed only in done(), after the step's sync, so a block handed out here has no
        # pending user.
        if MAIL_STREAM:
            cur = torch.cuda.current_stream()
            with torch.cuda.stream(self._mail_stream(dst_dev)):
                buf = torch.empty(data.shape, dtype=data.dtype, device=dst_dev)
            assert torch.cuda.current_stream() == cur, "CP mail: the sender's stream was not restored"
        else:
            buf = torch.empty(data.shape, dtype=data.dtype, device=dst_dev)
        # comm.cu peer_copy: one cudaMemcpyAsync on the sender's step stream. NOT torch's cross-device copy_, which puts
        # a two-way barrier between both devices' current streams (the sender then waits for the receiver's stream to
        # reach the point the single host thread has issued it to: measured CP host time per piece 1.4-3.6x
        # single-owner at equal launch counts). The event below is the only ordering the exchange needs.
        self.eng.cext.peer_copy(buf, data)
        ev = torch.cuda.Event()
        ev.record()
        self.mail[key] = (buf, ev)
        self.keep.append(data)

    def _recv(self, key):
        buf, ev = self.mail.pop(key)
        self.keep.append(buf)                             # alive until the step ends (allocated from another stream)
        torch.cuda.current_stream().wait_event(ev)
        return buf

    # ---- X1a: compressed / index records of a source layer, to every later rank of the group
    def x1a_send(self, r, L):
        for g, gd in self.groups.items():
            rk = gd["ranks"]
            if r not in rk:
                continue
            src = gd["pages"][r][r]
            for q in rk[rk.index(r) + 1:]:
                for k, (o, n) in enumerate(self._sections(L)):
                    self._send(("x1", g, r, q, L, k), self._pg(r).index_select(0, src)[:, o:o + n].contiguous(),
                               self.eng.devs[q])

    def x1a_recv(self, r, L):
        for g, gd in self.groups.items():
            rk = gd["ranks"]
            if r not in rk:
                continue
            for q in rk[:rk.index(r)]:
                for k, (o, n) in enumerate(self._sections(L)):
                    self._pg(r)[gd["pages"][r][q], o:o + n] = self._recv(("x1", g, q, r, L, k))

    # ---- X2: the window sections of layer L's halo blocks, to the next rank of the group
    def x2_send(self, r, L):
        for g, gd in self.groups.items():
            rk = gd["ranks"]
            if r not in rk or r not in gd["halo_src"]:
                continue
            q = rk[rk.index(r) + 1]
            self._send(("x2", g, r, q, L),
                       self._wsec(r).index_select(0, gd["halo_src"][r])[:, L * self.WLB:(L + 1) * self.WLB].contiguous(),
                       self.eng.devs[q])

    def x2_recv(self, r, L):
        for g, gd in self.groups.items():
            rk = gd["ranks"]
            if r not in rk or r not in gd["halo_dst"]:
                continue
            p = rk[rk.index(r) - 1]
            self._wsec(r)[gd["halo_dst"][r], L * self.WLB:(L + 1) * self.WLB] = self._recv(("x2", g, p, r, L))

    def done(self):
        assert not self.mail, f"CP: unconsumed exchanges {list(self.mail)[:4]}"
        self.keep.clear()
        if AUDIT:
            self.audit("after")

    # ---- ENG_CP_AUDIT: with pools filled with 0xFF before the run (fp8 / ue8m0 0xFF = NaN: never valid data), a page
    # section that is >= half 0xFF was never written
    def _missing(self, r, pages, secs):
        if not len(pages):
            return {}
        pg = self._pg(r).index_select(0, pages)
        out = {}
        for name, o, n in secs:
            frac = (pg[:, o:o + n] == 255).float().mean(1)
            bad = (frac > 0.5).nonzero().flatten()
            if bad.numel():
                out[name] = (int(bad.numel()), int(bad[0]))
        return out

    def audit(self, when: str):
        """before: every rank's pages below the chunk start (prefix, earlier chunks: the replica catch-up) and the lowest
        helper's window halo; after: every rank's pages of the chunk up to its own sub-chunk's end (own + X1a)."""
        for d in self.eng.devs:
            torch.cuda.synchronize(d)
        secs = [(f"{k}{L}", self.PO[f"{k}{L}"], self.RPP[L] * (584 if k == "comp" else 132))
                for L in (2, 8, 14, 20) for k in ("comp", "idx")]
        for g, gd in self.groups.items():
            subs = gd["subs"]
            c0 = subs[0][1]
            for j, (r, st, n) in enumerate(subs):
                se = gd["ents"][r]
                bt = se.block_table
                lo_b = (getattr(se, "cached_upto", 0) or 0) // 64 if r == subs[-1][0] else 0
                if when == "before":
                    blocks = range(0, c0 // 64)
                else:
                    blocks = range(c0 // 64, (st + n) // 64)
                pages = torch.tensor([bt[b] for b in blocks if b < len(bt)], dtype=torch.int64, device=self.eng.devs[r])
                miss = self._missing(r, pages, secs)
                if miss:
                    self.eng.log(f"engine CP AUDIT {when}: group {g} rank {r} (sub {j}, start {st}, n {n}, chunk start {c0}) "
                                 f"never-written page sections {{section: (count, first index into blocks {blocks.start}..)}} "
                                 f"{miss}")
                    self.eng.stats["cp_audit_bad"] = self.eng.stats.get("cp_audit_bad", 0) + 1
            halo = [subs[0]] if when == "before" else subs[1:]   # before: the lowest helper's (owner -> helper copy);
            for r0, st0, _ in halo:                              # after: every later sub-chunk's (X2)
                wt = gd["ents"][r0].window_table
                rs_ = getattr(gd["ents"][r0], "replay_start", 0) or 0
                hb = [b for b in range(max(rs_, st0 - (WIN - 1)) // 32, st0 // 32) if b in wt and st0 > rs_]
                if hb:
                    wv = self._wsec(r0).index_select(0, torch.tensor([wt[b] for b in hb], dtype=torch.int64,
                                                                     device=self.eng.devs[r0]))
                    badL = [L for L in range(21) if float((wv[:, L * self.WLB:(L + 1) * self.WLB] == 255).float().mean()) > 0.5]
                    if badL:
                        self.eng.log(f"engine CP AUDIT {when}: group {g} rank {r0} window halo blocks {hb} never written "
                                     f"in layers {badL[:6]}{'...' if len(badL) > 6 else ''}")
                        self.eng.stats["cp_audit_bad"] = self.eng.stats.get("cp_audit_bad", 0) + 1
