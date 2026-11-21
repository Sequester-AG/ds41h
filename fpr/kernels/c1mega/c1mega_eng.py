"""ENG_C1_MEGA adapter.

Runs of regular decoder layers as ONE c1mega launch per GPU (c1mega.cu k_layer<true>: the whole layer, TP4 over the
node's 4 GPUs, the residual replicated, owner-mode attention) in decode steps of the 4-row bucket where exactly one rank
holds rows (C1). Index layers (indexer / compressor) and Engram layers stay on the engine's pieces; a run is a maximal
block of consecutive other layers (40 layers: [0] [3..7] [9..13] [15..19] [21..23] [25..27] [29..31] [33..35] [37..39];
graph cuts (ENG_LAUNCH_SPLIT_AT) split runs).

State hand-off (the engine's state stays row-owned: rank `owner` holds the sequence's rows):
  * a run's first layer reads the owner's engine buffers (Engine._inputs: x0 / res0 / post0 / comb0 / lag0 at the first
    layer, else moe_out / xm2 / f_post / f_comb / f_pre) and the owner's per-step tables (pos, own, si / sl, the top-k of
    the layer's index source, the compressed pages of its KV source); the other ranks read them through peer pointers
    after the owner's CTA c set their ready flag c (Args::rdy, release / acquire at system scope);
  * layers inside a run chain through per-rank scratch;
  * the run's last layer writes moe_out, xm2, f_pre, f_post, f_comb into the OWNER's engine buffers (the other ranks
    into scratch), the window KV records of the rows are inserted by the owner's kernel; DSpark aux layers (37..39) keep
    their boundary output and the owner's aux columns are filled after the launch (the engine's X.mean(1)).
The owner cannot overwrite its inputs before the peers read them: every layer's AR #1 needs all 4 GPUs' wo_b partials.

Weights: c1mega's fp8 TP slices per (rank, regular layer), copied from the checkpoint at first use: q_a | wkv (replicated),
q_b rows of the rank's 16 heads, wo_a groups 2r / 2r+1, wo_b's K quarter, the shared expert's quarter, the mHC fn as
per-CTA slices (~55 MB per layer per GPU); the routed experts, gate, norms, sink and RoPE tables are the engine's own.
Same math as the engine (bf16 / fp8 / FP4 exact weights, fp32 accumulation, the engine's rounding points; other
summation order).
"""
from __future__ import annotations

import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
W, D, NRM, HC = 4, 5120, 4, 4
AUX_LAYERS = (37, 38, 39)


def ext():
    """The c1mega extension. Tests: env C1M_SRC = another source file in this directory (a kernel variant) -> its own build
    directory and module name, so variants can be queued side by side. Serving: env C1M_BUILD = the served build root
    (C1M_BUILD=build_srv: fpr/kernels/c1mega/build_srv, prebuilt before the slot starts; relative = to this directory);
    tests never set it (c1m_buildguard refuses), so they build under build/ and never touch the served build."""
    from torch.utils.cpp_extension import load
    src = os.environ.get("C1M_SRC", "c1mega.cu")
    tag = "" if src == "c1mega.cu" else "_" + os.path.splitext(os.path.basename(src))[0]
    bd = os.path.join(HERE, os.environ.get("C1M_BUILD", "build"), "c1m" + tag)
    os.makedirs(bd, exist_ok=True)
    return load(name="c1mega_ext" + tag, sources=[os.path.join(HERE, src)], build_directory=bd, verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])


def fn_slices(fn):
    """hc_*_fn [24, 4 x 5120] fp32 -> per-CTA contiguous slices [132, 96, 48] (the CTA's 32 / 48 columns, zero-padded)"""
    f = fn.reshape(96, D)
    out = torch.zeros(132, 96, 48, dtype=torch.float32)
    for c in range(132):
        ftb, nt = (3 * c, 3) if c < 56 else (168 + 2 * (c - 56), 2)
        out[c, :, :16 * nt] = f[:, 16 * ftb: 16 * (ftb + nt)]
    return out.contiguous()


class C1Mega:
    def __init__(self, eng, index_layers, engram_layers, cut=None, log=print, tim=False, amode=3, b16=None):
        self.eng, self.log, self.tim = eng, log, tim   # tim: per-CTA stamps of each launch's "last" / "first" layer (tests)
        # amode: 1 = the owner attends all 64 heads (q / o exchanged); 2 = every GPU its 16 heads on the owner's KV pages
        # (peer reads: far too slow, kept for reference); 3 = every GPU its 16 heads, the owner pushes each row's selected
        # KV records (window + top-k) into every peer's staging during the layer's front half (posted NVLink stores);
        # b16: of ("wqkv", "wo_a", "wo_b", "sh", "qb8"), the engine's tensors streamed instead of c1mega's fp8 copies ("qb8": q_b
        # from the engine's W8A16 packed fp8 wq_b, no copy; "sh":
        # the shared expert's lw["sw13"] / lw["sw2"]); None: env C1M_B16 (comma list; the deployment config sets it with
        # C1M_SRC = the matching compile-time build, e.g. c1mega_depn.cu for "wqkv,wo_a,wo_b,sh")
        if b16 is None:
            b16 = tuple(x for x in os.environ.get("C1M_B16", "").split(",") if x)
        self.amode, self.b16 = int(amode), tuple(b16)
        assert self.amode in (1, 2, 3) and set(self.b16) <= {"wqkv", "wo_a", "wo_b", "sh", "sh1", "sh2", "qb8"}, (amode, b16)
        self.X = ext()
        if hasattr(self.X, "fixed_config"):   # a deployment build (c1mega_dep*.cu): its compile-time config must match
            fam, fb, _ = self.X.fixed_config()
            bits = (1 if "wqkv" in self.b16 else 0) | (2 if "wo_a" in self.b16 else 0) | (4 if "wo_b" in self.b16 else 0) | (
                8 if "sh" in self.b16 else 0) | (16 if "qb8" in self.b16 else 0)
            assert fam in (-1, self.amode) and fb in (-1, bits), (
                f"c1mega build {os.environ.get('C1M_SRC', 'c1mega.cu')} fixes amode {fam} / bf16 bits {fb}; adapter: amode "
                f"{self.amode}, bf16 {self.b16} (bits {bits})")
        engram_on = eng.hooks.engram is not None
        # C1M_IX=1: the index / compressor and Engram layers run INSIDE the c1mega launches
        # the build must have them (C1M_HAS_PHASES)
        self.ixm = os.environ.get("C1M_IX", "0") == "1"
        self.index_layers, self.engram_layers = tuple(index_layers), tuple(engram_layers) if engram_on else ()
        self.ix = None
        if self.ixm:
            assert getattr(self.X, "has_phases", lambda: False)(), "C1M_IX=1 needs a c1mega build with the layer phases"
            import c1idx_host
            self.ix = c1idx_host.C1Idx(eng, log=log)
        special = set() if self.ixm else set(index_layers) | (set(engram_layers) if engram_on else set())
        Ls = list(eng.layers)
        self.runs = []
        for i, L in enumerate(Ls):
            if L in special:
                continue
            new = not self.runs or Ls[i - 1] != self.runs[-1][-1] or (cut is not None and L == cut)
            if new:
                self.runs.append([L])
            else:
                self.runs[-1].append(L)
        self.runs = [tuple(r) for r in self.runs]
        self.first = {run[0]: k for k, run in enumerate(self.runs)}
        self.member = {L for run in self.runs for L in run}
        self._w = {}          # (r, L) -> (wt list, fw list, tensor maps, the engine's bf16 tensors used)
        self.added = [0] * W  # bytes per GPU of c1mega's own weight copies (the rest are the engine's tensors)
        self._bufs = None     # per rank
        self._packs = {}      # (R, owner, run index) -> [device Args per rank]
        self.stats = dict(launches=0)

    # ---------------------------------------------------------------------------------------------- eligibility
    def owner_of(self, st, batches) -> int | None:
        """The rank holding all rows of a C1 decode step (4-row bucket, exactly one rank with rows), else None."""
        if not self.runs:
            return None
        s0 = st[0]
        if s0["R"] != NRM or s0.get("pf") is not None or s0.get("cp_on") or s0.get("moe") != "raw" or s0.get("lvl", 0):
            return None
        act = [r for r, sb in enumerate(batches) if sb.n_rows > 0]
        if len(act) != 1 or batches[act[0]].n_rows > NRM:
            return None
        return act[0]

    # ---------------------------------------------------------------------------------------------- weights
    def _weights(self):
        if self._w:
            return
        from ds41h.ref.weights import default_reader
        eng = self.eng
        t0 = time.time()
        rd = default_reader(eng.cfg.model)
        for run in self.runs:
            for L in run:
                g_ = lambda n: rd.get(f"layers.{L}.{n}")  # noqa: E731
                u8 = lambda n: g_(n).view(torch.uint8)    # noqa: E731
                woa, woa_s, wob, wob_s = u8("attn.wo_a.weight"), u8("attn.wo_a.scale"), u8("attn.wo_b.weight"), u8("attn.wo_b.scale")
                fn, fsc, fba = g_("hc_ffn_fn").float(), g_("hc_ffn_scale").float(), g_("hc_ffn_base").float()
                afn, asc, aba = g_("hc_attn_fn").float(), g_("hc_attn_scale").float(), g_("hc_attn_base").float()
                w1, w1s = u8("ffn.shared_experts.w1.weight"), u8("ffn.shared_experts.w1.scale")
                w3, w3s = u8("ffn.shared_experts.w3.weight"), u8("ffn.shared_experts.w3.scale")
                w2, w2s = u8("ffn.shared_experts.w2.weight"), u8("ffn.shared_experts.w2.scale")
                wqkv = torch.cat([u8("attn.wq_a.weight"), u8("attn.wkv.weight")], 0)
                wqkv_s = torch.cat([u8("attn.wq_a.scale"), u8("attn.wkv.scale")], 0)
                wqb, wqb_s = u8("attn.wq_b.weight"), u8("attn.wq_b.scale")
                fn_sl, afn_sl = fn_slices(fn), fn_slices(afn)
                for r in range(W):
                    dev = eng.devs[r]
                    lw = eng.lw[r][L]
                    AW = lw["AW"]
                    sw2_s = torch.zeros(160, 32, dtype=torch.uint8)
                    sw2_s[:, :18] = w2s[:, 18 * r: 18 * (r + 1)]
                    wqb_sp = torch.zeros(256, 48, dtype=torch.uint8)
                    wqb_sp[:, :40] = wqb_s[256 * r: 256 * (r + 1)]
                    e8 = torch.zeros(0, dtype=torch.uint8)
                    qa16, oa16, ob16 = "wqkv" in self.b16, "wo_a" in self.b16, "wo_b" in self.b16
                    # "qb8": q_b reads the engine's own W8A16 packed fp8 wq_b (AW["w8"]["wq_b"].packed [10][128][32896], the
                    # same checkpoint bytes and scales) through this rank's [10][32][32896] view: no 0.33 GB copy
                    qbv = None
                    if "qb8" in self.b16:
                        g8 = (AW.get("w8") or {}).get("wq_b")
                        assert g8 is not None, "C1M_B16 qb8: the engine holds no W8A16 wq_b (finish_attn w8a16='q_b')"
                        pk = g8.packed
                        assert tuple(pk.shape) == (10, 128, 32896) and pk.dtype == torch.uint8 and pk.device == torch.device(dev), pk.shape
                        qbv = pk[:, 32 * r: 32 * (r + 1)]
                    s1, s2 = bool({"sh", "sh1"} & set(self.b16)), bool({"sh", "sh2"} & set(self.b16))   # (sh1 / sh2: diagnostics)
                    sh16 = s1 or s2
                    post = [e8 if oa16 else woa[2048 * r: 2048 * (r + 1)], e8 if oa16 else woa_s[64 * r: 64 * (r + 1)],
                            e8 if ob16 else wob[:, 2048 * r: 2048 * (r + 1)], e8 if ob16 else wob_s[:, 64 * r: 64 * (r + 1)], fn_sl, fsc, fba, lw["nf"].float().cpu(), lw["gate_w"], lw["gate_b"],
                            e8 if s1 else torch.cat([w1[576 * r: 576 * (r + 1)], w3[576 * r: 576 * (r + 1)]], 0),
                            e8 if s1 else torch.cat([w1s[18 * r: 18 * (r + 1)], w3s[18 * r: 18 * (r + 1)]], 0),
                            e8 if s2 else w2[:, 576 * r: 576 * (r + 1)],
                            e8 if s2 else sw2_s, lw["w13"], lw["s13"], lw["w2"], lw["s2"]]
                    with torch.cuda.device(dev):
                        wt = [t.to(dev).contiguous() for t in post]
                        self.added[r] += sum(t.numel() * t.element_size() for t in post if t.device.type == "cpu")
                        front = [afn_sl, asc, aba, lw["na"].float().cpu(), e8 if qa16 else wqkv, e8 if qa16 else wqkv_s,
                                 AW["q_norm"], AW["kv_norm"], e8 if qbv is not None else wqb[8192 * r: 8192 * (r + 1)],
                                 e8 if qbv is not None else wqb_sp, AW["sink"], AW["cos_sin"]]
                        fw = [t.to(dev).contiguous() for t in front]
                        self.added[r] += sum(t.numel() * t.element_size() for t in front if t.device.type == "cpu")
                        e16 = torch.zeros(0, dtype=torch.bfloat16, device=dev)
                        b16t = (AW["wqkv_a"] if qa16 else e16, AW["wo_aT"] if oa16 else e16, AW["wo_b"] if ob16 else e16)
                        if sh16:
                            assert lw["sw13"].dtype == torch.bfloat16 and tuple(lw["sw13"].shape) == (1152, 5120), lw["sw13"].shape
                            assert lw["sw2"].dtype == torch.bfloat16 and tuple(lw["sw2"].shape) == (5120, 576), lw["sw2"].shape
                        sht = (lw["sw13"].contiguous() if s1 else e16, lw["sw2"].contiguous() if s2 else e16)
                        tm = self.X.maps3(wt[0], wt[2], fw[4], fw[8] if qbv is None else qbv, wt[10], wt[14], wt[15], wt[4], fw[0], *b16t, *sht).to(dev)
                    self._w[(r, L)] = (wt, fw, tm, b16t, sht, qbv)
        for d in eng.devs:
            torch.cuda.synchronize(d)
        self.log(f"c1mega: weight copies of layers {sorted(self.member)} (bf16 streamed: {self.b16 or 'none'}) in "
                 f"{time.time() - t0:.0f} s: {self.added[0] / 1e6:.1f} MB per GPU ({self.added[0] / len(self.member) / 1e6:.1f} MB "
                 f"per layer), allocated {[round(torch.cuda.memory_allocated(d) / 1e9, 1) for d in eng.devs]} GB")

    # ---------------------------------------------------------------------------------------------- buffers
    def _buffers(self):
        if self._bufs is not None:
            return
        bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
        bufs = []
        for r, dev in enumerate(self.eng.devs):
            with torch.cuda.device(dev):
                z = lambda *sh, dt=f32: torch.zeros(*sh, dtype=dt, device=dev)  # noqa: E731
                outs = lambda: [z(NRM, D, dt=bf), z(NRM, HC, D, dt=bf), z(NRM, D, dt=bf), z(NRM, dt=i32), z(NRM, 4), z(NRM, 4),  # noqa: E731
                                z(NRM, 16), z(NRM, 384), z(NRM, 6, dt=i32), z(NRM, 6), z(NRM, D, dt=bf)]
                B = dict(
                    chain=outs(),                      # inside a run (and the last layer's outputs on the other ranks)
                    side=outs(),                       # the last layer's outputs the engine does not read (owner)
                    Xo={L: z(NRM, HC, D, dt=bf) for L in AUX_LAYERS if L in self.member},
                    Xo0=z(NRM, HC, D, dt=bf), ain=z(NRM, D, dt=bf), apre=z(NRM, 4), apost=z(NRM, 4), acomb=z(NRM, 16),
                    qakv=z(4, 1792, dt=bf), qrout=z(NRM, 1280, dt=bf), kvout=z(NRM, 512, dt=bf), q=z(4, 16, 512, dt=bf),
                    o=z(4, 16, 512, dt=bf),
                    scr=[z(4, 2048, dt=bf), z(4, D, dt=bf), z(132 * 4), z(132 * 4 * 25), z(3 * 4 * 384), z(72 * 32 * 4),
                         z(4, 576, dt=bf), z(36 * 4), z(24 * 4 * 608, dt=torch.float16), z(80, dt=i32), z(132, dt=i32)],
                    fscr=[z(132 * 4), z(132 * 4 * 25), z(4 * 4 * 20 * 16 * 512), z(4 * 4 * 20 * 32)],
                    qall=torch.full((2, 4, 64, 512), -0.0, dtype=bf, device=dev),
                    ogl=torch.full((2, 4, 8192), -0.0, dtype=bf, device=dev),
                    ar1=torch.full((2, 4, 4, D), -0.0, dtype=f32, device=dev),
                    ar2=torch.full((2, 4, 4, D), -0.0, dtype=bf, device=dev),
                    rdy=z(132 + NRM + 1, dt=i32), stg=torch.zeros(self.X.stg_bytes(), dtype=torch.uint8, device=dev), tim=torch.zeros(132 * 192 if self.tim else 0, dtype=torch.int64, device=dev),
                    ti0=torch.full((NRM, 1, 512), -1, dtype=i32, device=dev), tl0=z(NRM, dt=i32))   # layers 0, 1: no top-k
                bufs.append(B)
        self._bufs = bufs

    # ---------------------------------------------------------------------------------------------- argument blocks
    def _pack(self, st, owner: int, k: int):
        import geometry as G
        eng, X, run = self.eng, self.X, self.runs[k]
        bufs = self._bufs
        so = st[owner]
        ar1, ar2 = [b["ar1"] for b in bufs], [b["ar2"] for b in bufs]
        qall, ogl = [b["qall"] for b in bufs], [b["ogl"] for b in bufs]
        rdy = [b["rdy"].data_ptr() for b in bufs]
        stg = [b["stg"].data_ptr() for b in bufs]
        mc = eng.mcfg
        x0, res0, post0, comb0, lag0 = eng._inputs(so, run[0])
        out = []
        for r in range(W):
            B, dev = bufs[r], eng.devs[r]
            ch = B["chain"]
            blobs = []
            for i, L in enumerate(run):
                wt, fw, tm, b16t, sht, qbv = self._w[(r, L)]
                c = so["caches"][L]                  # the owner's tables of this layer
                if i == 0:
                    fin = [x0, res0, post0, comb0.reshape(-1, 16) if comb0.dim() == 3 else comb0, lag0]
                else:
                    fin = [ch[10], ch[1], ch[5], ch[6], ch[4]]
                src = c["src"]
                ti = c["ti"] if c["ti"] is not None else bufs[owner]["ti0"]
                tl = c["tl"] if c["tl"] is not None else bufs[owner]["tl0"]
                fin += [so["pos"], so["own"], so["si"], so["sl"], ti, tl]
                last = i == len(run) - 1
                if last and r == owner:
                    sd = B["side"]
                    outs = [sd[0], so["xm2"], sd[2], sd[3], so["f_pre"], so["f_post"], so["f_comb"].view(-1, 16), sd[7], sd[8],
                            sd[9], so["moe_out"]]
                else:
                    outs = ch
                Xo = B["Xo"].get(L, B["Xo0"])
                fo = [Xo, B["ain"], B["apre"], B["apost"], B["acomb"], B["qakv"], B["qrout"], B["kvout"], B["q"], B["o"]]
                ins = [B["o"], Xo, B["apost"], B["acomb"], B["apre"]]
                pw = eng.pools[owner]
                win = pw.window[L * G.WINDOW_LAYER_PAGE_B:]
                if src is not None:
                    comp, cbs = pw.paged[G.PAGE_OFFSETS[f"comp{src}"]:], G.RECORDS_PER_PAGE[src]
                else:
                    comp, cbs = torch.zeros(0, dtype=torch.uint8), 1
                stamped = (i == len(run) - 1) if self.tim != "first" else (i == 0)
                with torch.cuda.device(dev):
                    b_ = X.pack_layer(r, NRM, ins, wt, outs, B["scr"] + [tm], ar1, ar2, B["tim"] if stamped else B["tim"][:0],
                                      mc.norm_eps, mc.hc_eps,
                                      mc.hc_iters, 1.5, fin, fw, fo, B["fscr"], win, G.WINDOW_PAGE_BYTES,
                                      G.WINDOW_PAGE_TOKENS, comp, G.PAGE_BYTES, cbs, mc.norm_eps, float(c["scale"]),
                                      1 if r == owner else 0, self.amode, owner, qall, ogl)
                    b_ = X.patch_integ(b_, rdy, so["si"].stride(0), ti.stride(0))
                    b_ = X.patch_bf16(b_, *[t.data_ptr() if t.numel() else 0 for t in b16t])
                    b_ = X.patch_sh16(b_, *[t.data_ptr() if t.numel() else 0 for t in sht])
                    b_ = X.patch_qb8(b_, qbv.data_ptr() if qbv is not None else 0)
                    e0l = self.ixm and self.ix.e0_on and L == 0   # E0 (C1IDX_E0=1): Engram wkv GEMVs in layer 0, all ranks
                    if self.ixm and (L in self.index_layers or L in self.engram_layers or e0l):   # C1M_IX: layer phases
                        ixp = egp = aop = 0
                        if L in self.index_layers and r == owner:   # owner-only index / compressor phases
                            ixp, aop = self.ix.ix_struct(st, owner, L).data_ptr(), self.ix.aout(owner).data_ptr()
                        if L in self.engram_layers:                 # Engram inject: every rank
                            egp = self.ix.eg_struct(st, owner, L, r).data_ptr()
                        elif e0l:                                   # (EngArgs, E0Args): C1Idx caches both (kept alive)
                            egp = self.ix.e0_struct(st, owner, r)[0].data_ptr()
                        b_ = X.patch_ext(b_, ixp, egp, aop, 1 if L in self.index_layers else 0)
                    blobs.append(X.patch_stg(b_, stg))
            out.append(torch.cat(blobs).to(dev))
        return out

    # ---------------------------------------------------------------------------------------------- the hook
    def run(self, r: int, s: dict, L: int, st: list) -> None:
        """Engine hook, called for every rank and every layer of a C1 step: launches the run that starts at L (no-op for
        the other layers of a run). Issue order: all ranks' launch of a run before the next piece (Engine._issue's each)."""
        k = self.first.get(L)
        if k is None:
            return
        owner = s["mega"]
        key = (s["R"], owner, k, st[owner]["moe_out"].data_ptr())   # (bucket states are rebuilt by reset_state)
        if key not in self._packs:
            self._weights()
            self._buffers()
            self._packs[key] = self._pack(st, owner, k)
        run = self.runs[k]
        self.eng._join_mix(s, pf=False)   # the previous engine layer's mHC mixes (f_pre / f_post / f_comb: side stream)
        if self.ixm:
            if any(L_ in self.index_layers for L_ in run):
                self.ix.prepare(r, s)          # the step's block-table row (the engine indexer's first-layer prep)
            for L_ in run:                     # Engram: its token-only side work (estream) must be complete
                if L_ in self.engram_layers:
                    torch.cuda.current_stream().wait_event(self.eng.hooks.engram[r].ev[self.engram_layers.index(L_)])
            if self.ix.e0_on and 0 in run and r == owner:   # E0 reads both layers' gathered rows in layer 0
                for ev_ in self.eng.hooks.engram[r].ev[:2]:
                    torch.cuda.current_stream().wait_event(ev_)
        self.X.run(self._packs[key][r], len(run), True)
        self.stats["launches"] += 1
        if r == owner:
            idx = [i for i, L_ in enumerate(run) if L_ in AUX_LAYERS]
            for i in idx:     # DSpark aux = stock previous_post.mean(dim=1) of the block input (the engine's statement)
                L_ = run[i]
                s["aux"][:, AUX_LAYERS.index(L_)].copy_(self._bufs[r]["Xo"][L_].mean(1))
