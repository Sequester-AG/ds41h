"""Host side of the index / compressor / Engram phases of c1mega for c1mega's engine adapter (c1mega_eng.C1Mega).

C1Idx(eng) owns, per rank: the fp8 copies of the indexer wq_b (+ repacked 32x32 scales; the engine keeps only a bf16
dequant), the per-layer tensor maps, and the phases' scratch (compressor ks, weights_proj, q, logits, block maxima, the
layer-20 candidates, counters, per-CTA epochs). Per C1 step:
  prepare(r, s)                    the per-step device prep the engine's indexer hook did at its first index layer (the
                                   step's block-table row from the persistent per-slot table); before the launch
  ix_struct(st, owner, L)          IxLayer of index layer L (device bytes on the owner's GPU) -> Args::ix (owner only)
  eg_struct(st, owner, L, r)       EngArgs of Engram layer L for rank r (device bytes on rank r's GPU) -> Args::eg
  aout(r)                          the buffer c1mega's A2 must write the normalized attention input to on index layers
The adapter also sets, on index layers: push_topk (mode 3), the layer's comp pages / ti / tl as today (the phases write
the engine's own so["ti"][L] / so["tl"][L]); c1mega runs the indexer, so the engine's Idx2f.run is NOT called for those
layers. Engram: the engine's EngramPhys.prepare (hash, gathers, wkv) must complete before the launch (as today).
E0 (C1IDX_E0=1): the wkv GEMMs move into the launch (layer 0, TP4; c1e0.cuh). Then, per C1 step:
  e0_prepare(owner, T)             hash + gathers only (instead of EngramPhys.prepare), before the launch
  e0_struct(st, owner, r)          (EngArgs, E0Args) device bytes for rank r -> layer 0's Args::eg (keep both alive)
  eg_struct(...)                   layers 1 / 14 then read E0's kv on their own GPU (local mode)
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHYS = os.path.dirname(os.path.dirname(HERE))
W_ = 4
NRM = 4
RATIO = {2: 2, 8: 2, 14: 2, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1}
K_SOURCE = {2: 2, 8: 8, 14: 14, 20: 20, 24: 20, 28: 20, 32: 20, 36: 20}
RING_I = {2: 0, 8: 1, 14: 2}
LMAX = 262_144            # logits row width (the lvl-0 graph bound of the engine's indexer; c1mega runs lvl 0 only)
EPS = 1e-20               # hooks_idx EPS (compressor norm, k_norm)


def ext():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("C1IDX_HOST_BUILD", os.path.join(HERE, "build_c1idx_host"))
    os.makedirs(bd, exist_ok=True)
    return load(name="c1idx_host_ext", sources=[os.path.join(HERE, "c1idx_host.cu")], build_directory=bd,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)


class C1Idx:
    def __init__(self, eng, log=print):
        import geometry as G
        self.eng, self.G, self.log = eng, G, log
        self.X = ext()
        self.H = eng.hooks.idx
        assert self.H is not None, "c1idx: the engine's idx2f indexer hook is required (its weights / views / tables)"
        self.layers = [L for L in RATIO if L in eng.layers]
        self._w8 = {}       # (r, L) -> (wq e4m3 [4096, 1280] uint8, scales [128, 48] uint8)
        self._maps = {}     # (r, L) -> device bytes [5 x 128]
        self._bufs = []
        self._ix = {}       # (key, L) -> device IxLayer bytes
        self._eg = {}       # (key, L, r) -> device EngArgs bytes
        from ds41h.ref.weights import CheckpointReader
        rd = CheckpointReader(os.environ.get("DS41H_MODEL_DIR", os.path.join(os.environ.get("DS41H_HOME", "~/.ds41h"), "model")))
        for L in self.layers:
            pre = f"layers.{L}.attn.indexer.wq_b."
            w8 = rd.get(pre + "weight").view(torch.uint8).contiguous()
            sc = torch.zeros(128, 48, dtype=torch.uint8)
            sc[:, :40] = rd.get(pre + "scale").view(torch.uint8)
            for r, dev in enumerate(eng.devs):
                self._w8[(r, L)] = (w8.to(dev), sc.to(dev))
        f32, bf, i32 = torch.float32, torch.bfloat16, torch.int32
        for r, dev in enumerate(eng.devs):
            z = lambda *sh, dt=f32: torch.zeros(*sh, dtype=dt, device=dev)  # noqa: E731
            self._bufs.append(dict(
                ks=z(NRM, 1024), wbuf=z(NRM, 32, dt=bf), q=z(NRM, 4096, dt=bf), logits=z(NRM, LMAX), bmax=z(NRM, LMAX // 8),
                cand=torch.full((NRM, 2048), -1, dtype=i32, device=dev), cand_n=z(NRM, dt=i32), cnt=z(8, dt=i32),
                itick=z(132, dt=i32), selflag=z(132, dt=i32), rowsel=z(NRM, dt=i32), slow=z(1, dt=i32), aout=z(NRM, 5120, dt=bf),
                dsc=z(self.X.ds_words(), dt=i32),   # v14 distributed selection scratch (histograms, lists; zero)
                # Engram
                kvl=z(NRM, 5, 5120, dt=bf), keepl=z(132 * NRM, dt=torch.uint8), kflag=z(132, dt=i32),
                part=z(132 * NRM * 12, dt=torch.float64), x1=z(NRM, 4, 5120, dt=bf), zrow=z(NRM, 5120, dt=bf),
                ecnt=z(4, dt=i32), etick=z(132, dt=i32)))
        self.added = [sum(t.numel() * t.element_size() for k_, t in b.items()) + sum(
            w.numel() + s.numel() for (r_, L), (w, s) in self._w8.items() if r_ == r) for r, b in enumerate(self._bufs)]
        self.e0_on = False
        self.fold_bt = os.environ.get("C1IDX_FOLD_BT", "1") == "1"
        self.tim_on = os.environ.get("C1IDX_TIM", "0") == "1"   # tests: the glue's sub-stamps per (owner, layer), tim_buf
        # v14: distributed selection for rows with more than max(C1IDX_DTHR, 512) keys (0: v13's one CTA per row) and union
        # tiles for the consumer layers' candidate mode (C1IDX_UNI=0: v13's per-row tiles)
        self.dthr = int(os.environ.get("C1IDX_DTHR", "4096"))
        self.uni = int(os.environ.get("C1IDX_UNI", "1"))
        self._tim = {}
        if os.environ.get("C1IDX_E0", "0") == "1" and getattr(eng.hooks, "engram", None) is not None:
            self._init_e0(rd)

    # ---------------------------------------------------------------- E0: the Engram wkv projections in-kernel (TP4)
    def _init_e0(self, rd=None):
        """E0 streams the engine's OWN fp8 Engram wkv in place: EngramPhys.gemm8[l].packed (W8A16 repack, chunk-major
        [48][100][32,896] on every GPU; GPU r reads row tiles 25 r .. 25 r + 24). Here only E0's landing buffers and
        counters (~0.5 MB per GPU)."""
        f32, bf, u8, i32 = torch.float32, torch.bfloat16, torch.uint8, torch.int32
        self._e0w = []
        for r, dev in enumerate(self.eng.devs):
            e = self.eng.hooks.engram[r]
            g8 = getattr(e, "gemm8", None) or []
            assert len(g8) == 2, "c1idx E0: EngramPhys must hold the W8A16 packed weights (w8_max_rows > 0)"
            pk = [g.packed for g in g8]
            assert all(t.is_cuda and t.dtype == torch.uint8 and tuple(t.shape) == (48, 100, 32896) and t.stride(0) == pk[0].stride(0)
                       for t in pk), [tuple(t.shape) for t in pk]
            self._e0w.append(dict(pk=pk))
            z = lambda *sh, dt=f32: torch.zeros(*sh, dtype=dt, device=dev)  # noqa: E731
            self._bufs[r].update(e0xl=z(2, NRM, 6144, dt=bf), e0keep=z(NRM, dt=u8), e0xcnt=z(1, dt=i32),
                                 e0kv=z(2, NRM, 25600, dt=bf), e0kvcnt=z(2, dt=i32), e0tick=z(132, dt=i32))
            self.added[r] += sum(self._bufs[r][k].numel() * self._bufs[r][k].element_size() for k in
                                 ("e0xl", "e0keep", "e0xcnt", "e0kv", "e0kvcnt", "e0tick"))
        self._e0 = {}
        self.e0_on = True
        self.log(f"c1idx: E0 on (Engram wkv in-kernel, TP4, the engine's packed weights in place): C1Idx "
                 f"{self.added[0] / 1e6:.1f} MB per GPU in all")

    def e0_prepare(self, r: int, T: int, stream=None) -> None:
        """E0 mode: the drop-in for EngramPhys.prepare(T, stream) on a C1 step that c1mega runs (the engine's p_embed):
        hash + both gathers, WITHOUT the wkv GEMMs (E0 computes kv inside the launch); records the same per-layer events
        (ev[l] after gather l), so the adapter's waits stay as they are. Non-C1 steps keep EngramPhys.prepare."""
        e = self.eng.hooks.engram[r]
        e.kv_idx = [None, None]
        st = stream or torch.cuda.current_stream(e.dev)
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(e.dev))
        with torch.cuda.stream(st):
            e._resolve_starts()
            e.run_hash(T)
            for l in range(2):
                e.run_gather(T, (l,))
                e.ev[l].record(st)

    def e0_struct(self, st: list, owner: int, r: int) -> torch.Tensor:
        """Layer 0's EngArgs (li = -1, e0 = E0Args) for rank r: hooks 2 / 1 / 3 of layer 0 run E0 on every GPU."""
        so = st[owner]
        key = (so["R"], owner, so["moe_out"].data_ptr(), r)
        if key in self._e0:
            return self._e0[key]
        eg = self.eng.hooks.engram
        Bs, Wr = self._bufs, self._e0w[r]
        eo = eg[owner]
        own = r == owner
        P = lambda t: t.data_ptr()  # noqa: E731
        ptrs = [P(Wr["pk"][0]), P(Wr["pk"][1]), 0, P(eo.rows[0]) if own else 0, P(eo.rows[1]) if own else 0,
                P(eo.keep) if own else 0]
        ptrs += [P(Bs[s]["e0xl"][li]) for s in range(W_) for li in range(2)]
        ptrs += [P(Bs[s]["e0keep"]) for s in range(W_)] + [P(Bs[s]["e0xcnt"]) for s in range(W_)]
        ptrs += [P(Bs[s]["e0kv"][li]) for s in range(W_) for li in range(2)]
        ptrs += [P(Bs[s]["e0kvcnt"]) for s in range(W_)] + [P(Bs[r]["e0tick"])]
        assert eo.rows[0].is_contiguous() and eo.rows[0].shape[1:] == (24, 256)
        e0b = self.X.pack_e0(ptrs, [so["R"] if so["R"] <= NRM else NRM, owner, Wr["pk"][0].stride(0)]).to(self.eng.devs[r])
        B = Bs[r]
        er = eg[r]
        egp = [0] * 14 + [er.q[0].data_ptr(), er.k[0].data_ptr(), B["part"].data_ptr(), B["x1"].data_ptr(),
                          B["zrow"].data_ptr(), B["ecnt"].data_ptr(), B["etick"].data_ptr(), e0b.data_ptr(), 0, 0, 0, 0]
        b = self.X.pack_eg(egp, [-1, 0], [float(er.eps), float(er.clamp)]).to(self.eng.devs[r])
        self._e0[key] = (b, e0b)
        return self._e0[key]

    def aout(self, r: int) -> torch.Tensor:
        return self._bufs[r]["aout"]

    def maps(self, r: int, L: int) -> torch.Tensor:
        if (r, L) not in self._maps:
            G, W = self.G, self.H.W[r][L]
            src = K_SOURCE[L]
            pool = self.eng.pools[r]
            kp = pool.paged.data_ptr() + G.PAGE_OFFSETS[f"idx{src}"]
            cw = W.get("comp_w")
            with torch.cuda.device(self.eng.devs[r]):
                m = self.X.ix_maps(cw.data_ptr() if cw is not None else 0, cw.shape[0] if cw is not None else 0,
                                   W["idx_wp"].data_ptr(), self._w8[(r, L)][0].data_ptr(), kp, G.RECORDS_PER_PAGE[src],
                                   G.PAGE_BYTES, pool.n_paged)
            self._maps[(r, L)] = m.to(self.eng.devs[r])
        return self._maps[(r, L)]

    def prepare(self, r: int, s: dict) -> None:
        """Before the launch, on rank r's stream: the step's block-table row (Idx2f.run's first-index-layer step). In the
        engine's dev_bt mode this is folded into the kernel (IdxArgs dbt / preq: no kernel here); C1IDX_FOLD_BT=0 restores
        the index_select."""
        d = s.get("idx")
        if d is not None and d.get("dev_bt") and not self.fold_bt:
            torch.index_select(self.H.views[r]["dbt"], 0, d["preq_slot"], out=d["bt"])

    def tim_buf(self, owner: int, L: int) -> torch.Tensor:
        """C1IDX_TIM=1: int64 [132][32] globaltimer stamps of index layer L's phases on the owner (c1hooks_ix.inc front /
        back: 0 hook-3 entry, 1 X1 done, 2 X3 done, 3 hook-3 exit, 4 hook-4 entry, 5 hook-4 exit; x45 8 + k: 0 quant
        start, 6 q staged, 1 quant end, 2 keys ready, 3 logits end, 4 all logits, 5 select end; x2_row 16 + k: 0 entry,
        1 stage 0, 2 ks ready, 3 pooled, 4 latent, 5 kpre end). Zero = not reached. v13: 23 / 24 / 25 X1 chunk 0 / 6 / 9
        received, 26 / 27 X3 chunk 0 / 4 received, 28 the producer's first hook-1 chunk issued, 29 its last X1 chunk, 30 its
        last X3 chunk.
        """
        k = (owner, L)
        if k not in self._tim:
            self._tim[k] = torch.zeros(132, 32, dtype=torch.int64, device=self.eng.devs[owner])
        return self._tim[k]

    def ix_struct(self, st: list, owner: int, L: int) -> torch.Tensor:
        so = st[owner]
        key = (so["R"], owner, so["moe_out"].data_ptr(), L)
        if key in self._ix:
            return self._ix[key]
        G, H, B = self.G, self.H, self._bufs[owner]
        d, v, W = so["idx"], H.views[owner], H.W[owner][L]
        ratio, src = RATIO[L], K_SOURCE[L]
        rpp, PB = G.RECORDS_PER_PAGE[src], G.PAGE_BYTES
        pool = self.eng.pools[owner]
        kind = 2 if L in (2, 8, 14, 20) else 1
        m = self.maps(owner, L)
        P = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
        ring = v.get(f"ring{L}") if kind == 2 and ratio == 2 else None
        ti, tl = so["ti"][L], so["tl"][L]
        fold = bool(d.get("dev_bt")) and self.fold_bt   # bt = dbt[preq_slot[0]] in the kernel (one sequence per C1 step)
        ptrs = [0, 0, P(d["seq_lens"][ratio]), pool.paged.data_ptr() + G.PAGE_OFFSETS[f"idx{src}"], 0 if fold else P(d["bt"]),
                P(B["logits"]), P(B["bmax"]), P(B["cand"]) if L == 20 else 0, P(ti), P(tl), P(B["cnt"]), P(m) + 3 * 128,
                P(B["slow"]), P(B["cand_n"]) if L == 20 else 0, P(B["cand"]) if kind == 1 else 0,
                P(B["cand_n"]) if kind == 1 else 0, P(m) + 4 * 128, P(B["aout"]), P(so["pos"]), P(W["cos_sin"]),
                P(B["ks"]), P(B["wbuf"]), P(B["q"]), P(self._w8[(owner, L)][1]), P(W.get("idx_wk")), P(W.get("comp_norm")),
                P(W.get("idx_k_norm")), P(ring), P(d["rslot"]), P(d["cslot"][ratio]),
                pool.paged.data_ptr() + G.PAGE_OFFSETS[f"comp{src}"], pool.paged.data_ptr() + G.PAGE_OFFSETS[f"idx{src}"],
                P(m), P(B["itick"]), P(B["selflag"]), P(v["dbt"]) if fold else 0, P(d["preq_slot"]) if fold else 0,
                P(self.tim_buf(owner, L)) if self.tim_on else 0, P(B["rowsel"]), P(B["dsc"]) if self.dthr > 0 else 0]
        ntc = (W["comp_w"].shape[0] // 16) if kind == 2 else 0
        ints = [L, ratio, rpp, PB, LMAX, LMAX // 8, ti.stride(0), kind, NRM, ntc,
                ring.stride(0) if ring is not None else 0, PB, 8 if os.environ.get("C1IDX_SEL2", "0") == "1" else 0,
                v["dbt"].stride(0) if fold else 0, self.dthr, self.uni]
        b = self.X.pack_ix(ptrs, ints, [EPS]).to(self.eng.devs[owner])
        self._ix[key] = b
        return b

    def eg_struct(self, st: list, owner: int, L: int, r: int) -> torch.Tensor:
        """Engram layer L (1 or 14) for rank r; the kv rows / keep flags are the owner's EngramPhys buffers."""
        so = st[owner]
        key = (so["R"], owner, so["moe_out"].data_ptr(), L, r)
        if key in self._eg:
            return self._eg[key]
        eg = self.eng.hooks.engram
        li = (1, 14).index(L)
        eo, er = eg[owner], eg[r]
        Bs = self._bufs
        B = Bs[r]
        ptrs = [eo.kv[li].data_ptr(), eo.keep.data_ptr()] + [b["kvl"].data_ptr() for b in Bs] + \
               [b["keepl"].data_ptr() for b in Bs] + [b["kflag"].data_ptr() for b in Bs] + \
               [er.q[li].data_ptr(), er.k[li].data_ptr(), B["part"].data_ptr(), B["x1"].data_ptr(), B["zrow"].data_ptr(),
                B["ecnt"].data_ptr(), B["etick"].data_ptr()]
        if self.e0_on:   # local mode: E0's kv / keep on this GPU (layer 0 of the same launch wrote them)
            ptrs += [0, B["e0kv"][li].data_ptr(), B["e0keep"].data_ptr(), B["e0kvcnt"].data_ptr(), B["e0tick"].data_ptr()]
        b = self.X.pack_eg(ptrs, [li, eo.kv[li].stride(0)], [float(eo.eps), float(eo.clamp)]).to(self.eng.devs[r])
        self._eg[key] = b
        return b
