"""c1d: the 3 DSpark drafter blocks of HSD as c1mega instantiations.

Status: weight preparation (this file) written.
the drafter instance builds with -DC1M_NE=128 -DC1M_TOPK=3 -DC1M_WIN=131
-DC1M_TKA=0 into fpr/kernels/hsdtail/build/c1d_*. Buffers / packing: next.

Weights per (rank, block i), in the order of c1mega.cu's pack_layer
  post (wt, 18): wo_a rows 2048 r ..

(fp8) + scales, shared w2 columns 576 r .. (fp8) + scales (padded [160, 32])
        routed experts w13 / s13 / w2 / s2 (the DraftRank's raw TP slice: the engine's layout)
  front (fw, 12): hc_attn_fn per-CTA slices, hc_attn_scale, hc_attn_base, attn_norm, q_a | wkv (fp8, replicated) + scales,
        q_norm, kv_norm, q_b rows 8192 r .. (16 heads, fp8) + scales (padded [256, 48]), attn sink, the drafter's RoPE table
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
W, D = 4, 5120


def fn_slices(fn):
    """= kernels/c1mega/c1mega_eng.py fn_slices: hc_*_fn [24, 4 x 5120] fp32 -> per-CTA slices [132, 96, 48]."""
    f = fn.reshape(96, D)
    out = torch.zeros(132, 96, 48, dtype=torch.float32)
    for c in range(132):
        ftb, nt = (3 * c, 3) if c < 56 else (168 + 2 * (c - 56), 2)
        out[c, :, :16 * nt] = f[:, 16 * ftb: 16 * (ftb + nt)]
    return out.contiguous()


def block_weights(rd, i, r, B, cos_sin, dev):
    """(post, front) lists of drafter block mtp.i for rank r on device dev. B: the DraftRank's block dict (experts, gate,
    norms, sink); cos_sin: the drafter's RoPE table (DraftRank.cos_sin, fp32 [max_pos, 64])."""
    u8 = torch.uint8
    g = lambda n: rd.get(f"mtp.{i}.{n}")  # noqa: E731
    woa, woa_s = g("attn.wo_a.weight").view(u8), g("attn.wo_a.scale").view(u8)
    wob, wob_s = g("attn.wo_b.weight").view(u8), g("attn.wo_b.scale").view(u8)
    fn, fsc, fba = g("hc_ffn_fn").float(), g("hc_ffn_scale").float(), g("hc_ffn_base").float()
    afn, asc, aba = g("hc_attn_fn").float(), g("hc_attn_scale").float(), g("hc_attn_base").float()
    w1, w1s = g("ffn.shared_experts.w1.weight").view(u8), g("ffn.shared_experts.w1.scale").view(u8)
    w3, w3s = g("ffn.shared_experts.w3.weight").view(u8), g("ffn.shared_experts.w3.scale").view(u8)
    w2, w2s = g("ffn.shared_experts.w2.weight").view(u8), g("ffn.shared_experts.w2.scale").view(u8)
    wqkv = torch.cat([g("attn.wq_a.weight").view(u8), g("attn.wkv.weight").view(u8)], 0)
    wqkv_s = torch.cat([g("attn.wq_a.scale").view(u8), g("attn.wkv.scale").view(u8)], 0)
    wqb, wqb_s = g("attn.wq_b.weight").view(u8), g("attn.wq_b.scale").view(u8)
    sw2_s = torch.zeros(160, 32, dtype=u8)
    sw2_s[:, :18] = w2s[:, 18 * r: 18 * (r + 1)]
    wqb_sp = torch.zeros(256, 48, dtype=u8)
    wqb_sp[:, :40] = wqb_s[256 * r: 256 * (r + 1)]
    post = [woa[2048 * r: 2048 * (r + 1)], woa_s[64 * r: 64 * (r + 1)], wob[:, 2048 * r: 2048 * (r + 1)],
            wob_s[:, 64 * r: 64 * (r + 1)], fn_slices(fn), fsc, fba,
            B["nf"].float(), B["gate_wb"], B["gate_b"],
            torch.cat([w1[576 * r: 576 * (r + 1)], w3[576 * r: 576 * (r + 1)]], 0),
            torch.cat([w1s[18 * r: 18 * (r + 1)], w3s[18 * r: 18 * (r + 1)]], 0), w2[:, 576 * r: 576 * (r + 1)], sw2_s,
            B["w13"], B["s13"], B["w2"], B["s2"]]
    front = [fn_slices(afn), asc, aba, B["na"].float(), wqkv, wqkv_s, B["q_norm"], B["kv_norm"],
             wqb[8192 * r: 8192 * (r + 1)], wqb_sp, B["sink"], cos_sin]
    mv = lambda t: t.to(dev).contiguous() if (t.device.type == "cpu" or t.device != torch.device(dev)) else t.contiguous()  # noqa: E731
    return [mv(t) for t in post], [mv(t) for t in front]


# C1M_WIN = 131 = the ring's keys
DEFS = ("C1M_NE=128", "C1M_TOPK=3", "C1M_WIN=131", "C1M_TKA=0")
C1M_SRC = os.path.join(os.path.dirname(HERE), "c1mega", "c1mega.cu")
# instance kinds: (source, -D defines). "x2b": the C128 expert-path session's x2b expert fragments (c1d_x2b.cu = its
# c1mega_x2bd.cu with C1M_WIN 132: the drafter defines + C1M_MOE_PRODUCE / CONSUME -> c1moe_x2b_*.inc + c1mega.cu; test only)
# "base": c1d_base.cu = c1mega.cu with the drafter knobs and the expert fragments pinned to c1moe_produce / consume.inc
# "c1mega": c1mega.cu with -D
# DEFS only (= x2b fragments now)
KINDS = {"base": (os.path.join(HERE, "c1d_base.cu"), ()), "x2b": (os.path.join(HERE, "c1d_x2b.cu"), ()),
         "x2s": (os.path.join(HERE, "c1d_x2s.cu"), ()),
         "x2ho": (os.path.join(HERE, "c1d_x2ho.cu"), ()),  # pre-x2b fragments + x2h's B-fragment change (test only)
         "c1mega": (C1M_SRC, DEFS)}


def ext(defs=DEFS, src=C1M_SRC):
    """"""
    from torch.utils.cpp_extension import load
    stem = os.path.splitext(os.path.basename(src))[0]
    tag = ("" if stem == "c1mega" else "_" + stem) + "".join("_" + d.lower().replace("=", "") for d in defs)
    bd = os.path.join(HERE, "build", "c1d" + tag)
    os.makedirs(bd, exist_ok=True)
    return load(name="c1d_ext" + tag, sources=[src], build_directory=bd, verbose=False,
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"] + [f"-D{d}" for d in defs])


NRM, HC = 4, 4


class DrafterC1M:
    """HSD's 3 drafter blocks as ONE c1mega launch of 3 layers per GPU (amode 3: every GPU attends its 16 heads, the
    owner pushes the rows' window records into the peers' staging), for steps with exactly one rank holding drafting rows
    and one drafting sequence per rank (C1). Inputs are the OWNER's drafter buffers, read by the peers through peer
    pointers after the owner's ready flags (patch_integ, as the engine integration). Outputs (moe_out, xm2, f_pre / post /
    comb of block 2) are replicated on the 4 GPUs: every GPU runs the final boundary itself (no hd all-gather).
    Packs (device Args) are built up front for every owner, so no host copy ever runs during a step."""

    def __init__(self, hsdt, rd, cfg, log=print, kind="base"):
        self.H, self.rd, self.cfg = hsdt, rd, cfg
        self.tim = os.environ.get("C1D_TIM", "")   # "last" / "first": c1mega's per-phase stamps of block 2 / block 0 (tests)
        self.kind = kind
        src_, defs_ = KINDS[kind]
        self.X = ext(defs_, src_)
        self.devs = hsdt.devs
        t0 = __import__("time").time()
        self.w = {}
        for r, dev in enumerate(self.devs):
            d0 = hsdt.ranks0[r]
            with torch.cuda.device(dev):
                for i, B in enumerate(d0.blocks):
                    wt, fw = block_weights(rd, i, r, B, d0.cos_sin, dev)
                    e16 = torch.zeros(0, dtype=torch.bfloat16, device=dev)
                    tm = self.X.maps2(wt[0], wt[2], fw[4], fw[8], wt[10], wt[14], wt[15], wt[4], fw[0], e16, e16, e16).to(dev)
                    self.w[(r, i)] = (wt, fw, tm)
        self._buffers()
        self._packs = {}
        for d in self.devs:
            torch.cuda.synchronize(d)
        log(f"c1d ({kind}): drafter weights + buffers on {W} GPUs in {__import__('time').time() - t0:.0f} s; "
            f"allocated {[round(torch.cuda.memory_allocated(d) / 1e9, 1) for d in self.devs]} GB")

    def _buffers(self):
        """= kernels/c1mega/c1mega_eng.py C1Mega._buffers (sizes of the main-model instance: upper bounds here)."""
        bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
        self.B = []
        for r, dev in enumerate(self.devs):
            with torch.cuda.device(dev):
                z = lambda *sh, dt=f32: torch.zeros(*sh, dtype=dt, device=dev)  # noqa: E731
                B = dict(
                    chain=[z(NRM, D, dt=bf), z(NRM, HC, D, dt=bf), z(NRM, D, dt=bf), z(NRM, dt=i32), z(NRM, 4), z(NRM, 4),
                           z(NRM, 16), z(NRM, 384), z(NRM, 6, dt=i32), z(NRM, 6), z(NRM, D, dt=bf)],
                    Xo=z(NRM, HC, D, dt=bf), ain=z(NRM, D, dt=bf), apre=z(NRM, 4), apost=z(NRM, 4), acomb=z(NRM, 16),
                    qakv=z(4, 1792, dt=bf), qrout=z(NRM, 1280, dt=bf), kvout=z(NRM, 512, dt=bf), q=z(4, 16, 512, dt=bf),
                    o=z(4, 16, 512, dt=bf),
                    scr=[z(4, 2048, dt=bf), z(4, D, dt=bf), z(132 * 4), z(132 * 4 * 25), z(4 * 384), z(72 * 32 * 4),
                         z(4, 576, dt=bf), z(36 * 4), z(24 * 4 * 608, dt=torch.float16), z(80, dt=i32), z(132, dt=i32)],
                    fscr=[z(132 * 4), z(132 * 4 * 25), z(4 * 4 * 20 * 16 * 512), z(4 * 4 * 20 * 32)],
                    qall=torch.full((2, 4, 64, 512), -0.0, dtype=bf, device=dev),
                    ogl=torch.full((2, 4, 8192), -0.0, dtype=bf, device=dev),
                    ar1=torch.full((2, 4, 4, D), -0.0, dtype=f32, device=dev),
                    ar2=torch.full((2, 4, 4, D), -0.0, dtype=bf, device=dev),
                    rdy=z(132 + NRM + 1, dt=i32), stg=torch.zeros(self.X.stg_bytes(), dtype=torch.uint8, device=dev),
                    tim=torch.zeros(132 * 192 if self.tim else 0, dtype=torch.int64, device=dev),
                    ti0=torch.full((NRM, 1, 512), -1, dtype=i32, device=dev), tl0=z(NRM, dt=i32),
                    xn=z(NRM, HC, D, dt=bf), comp0=torch.zeros(0, dtype=torch.uint8, device=dev))
                self.B.append(B)

    def packs(self, owner, dranks):
        """Device Args of the 3-layer run for every rank, inputs = the owner's DraftRank (dranks[owner], D_pad = 1)."""
        key = (owner, id(dranks[owner]))
        if key in self._packs:
            return self._packs[key]
        X, Bs, mc = self.X, self.B, self.cfg
        do = dranks[owner]
        b = do.b
        Rd = do.Rd
        ar1, ar2 = [x["ar1"] for x in Bs], [x["ar2"] for x in Bs]
        qall, ogl = [x["qall"] for x in Bs], [x["ogl"] for x in Bs]
        rdy = [x["rdy"].data_ptr() for x in Bs]
        stg = [x["stg"].data_ptr() for x in Bs]
        si = b["idx"].view(Rd, -1)
        out = []
        for r, dev in enumerate(self.devs):
            B = Bs[r]
            ch = B["chain"]
            blobs = []
            for i in range(len(do.blocks)):
                wt, fw, tm = self.w[(r, i)]
                if i == 0:   # block 0: y = 0, residual = X0 (anchor / noise embeddings x 4 copies), post 0, comb I, lag one-hot
                    fin = [b["zero_y"], b["X0"], b["zero_post"], b["eye"].view(Rd, 16), b["onehot"]]
                else:
                    fin = [ch[10], ch[1], ch[5], ch[6], ch[4]]
                fin += [b["q_pos"], b["q_slot"], si, b["lens"], B["ti0"], B["tl0"]]
                fo = [B["Xo"], B["ain"], B["apre"], B["apost"], B["acomb"], B["qakv"], B["qrout"], B["kvout"], B["q"], B["o"]]
                ins = [B["o"], B["Xo"], B["apost"], B["acomb"], B["apre"]]
                stamped = self.tim and i == (0 if self.tim == "first" else len(do.blocks) - 1)
                with torch.cuda.device(dev):
                    p_ = X.pack_layer(r, Rd, ins, wt, ch, B["scr"] + [tm], ar1, ar2, B["tim"] if stamped else B["tim"][:0],
                                      mc.norm_eps, mc.hc_eps, mc.hc_iters,
                                      1.5, fin, fw, fo, B["fscr"], do.cache[i], do.page_stride, 32, B["comp0"], 1, 1,
                                      mc.norm_eps, 512 ** -0.5, 1 if r == owner else 0, 3, owner, qall, ogl)
                    p_ = X.patch_integ(p_, rdy, si.stride(0), B["ti0"].stride(0))
                    p_ = X.patch_bf16(p_, 0, 0, 0)
                    blobs.append(X.patch_stg(p_, stg))
            out.append(torch.cat(blobs).to(dev))
        for d in self.devs:
            torch.cuda.synchronize(d)
        self._packs[key] = out
        return out

    def launch(self, r, owner, dranks):
        self.X.run(self.packs(owner, dranks)[r], len(dranks[owner].blocks), True)
        # C1D_SYNC=1 (tests): after the last rank's EAGER launch, wait for all 4 GPUs, so a kernel fault is reported here
        # (c1mega) and not at the next op (HSDT's final boundary); never during a graph capture
        if os.environ.get("C1D_SYNC") == "1" and r == W - 1 and not torch.cuda.is_current_stream_capturing():
            for d in self.devs:
                try:
                    torch.cuda.synchronize(d)
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"c1d: the c1mega run (owner {owner}) failed on GPU {d}: {e}") from e
            print(f"c1d: eager run of owner {owner} completed on all GPUs", flush=True)

    SEQ = (0, 14, 15, 16, 17, 18, 19, 1, 2, 3, 4, 5, 6, 20, 8, 9, 10, 11, 12, 13)
    LAB = {0: "start", 14: "A1 bd", 15: "a ready", 16: "q_a", 17: "q_b", 18: "q ready", 19: "attn", 1: "wo_a",
           2: "z ready", 3: "wo_b", 4: "AR1", 5: "ffn bd", 6: "ss ready", 20: "rnorm", 8: "router (+f image)",
           9: "rt ready", 10: "top-k", 11: "GEMM1", 12: "GEMM2", 13: "end"}

    def stall_table(self, r):
        """"""
        if not self.tim:
            return ""
        t = self.B[r]["tim"].view(132, 192).cpu().double()
        ts, dw, sw, ck = t[:, :48], t[:, 48:96], t[:, 96:144], t[:, 144:192]
        rows = []
        for prev, i in zip(self.SEQ[:-1], self.SEQ[1:]):
            ok = (ts[:, i] > 0) & (ts[:, prev] > 0)
            if not bool(ok.any()):
                continue
            d = (ts[:, i] - ts[:, prev]) / 1000
            w = (dw[:, i] - dw[:, prev]) / 1000
            b = (sw[:, i] - sw[:, prev]) / 1000
            cs = torch.where(ok, ts[:, i], torch.full_like(ts[:, i], -1)).argmax()
            mhz = ((ck[:, i] - ck[:, prev]) / (ts[:, i] - ts[:, prev]).clamp(min=1) * 1000)[ok & (ts[:, i] > ts[:, prev])]
            mz = f" {float(mhz.median()):5.0f} MHz" if len(mhz) else ""
            rows.append(f"  {self.LAB[i]:>22s}: mean {float(d[ok].mean()):5.1f} (data {float(w[ok].mean()):4.1f}, sync "
                        f"{float(b[ok].mean()):4.1f}) | critical CTA {int(cs):3d}: {float(d[cs]):5.1f} (data {float(w[cs]):4.1f}, "
                        f"sync {float(b[cs]):4.1f}){mz}")
        ok0 = ts[:, 0] > 0
        if bool(ok0.any()):
            rows.append(f"  block total (start -> end, max over CTAs): {float((ts[:, 13].max() - ts[ok0, 0].min()) / 1000):.1f} us")
        return "\n".join(rows)

    def outputs(self, r, Rd):
        """Block 2's outputs on rank r (replicated): moe_out, xm2, f_pre, f_post, f_comb [Rd, 16]."""
        ch = self.B[r]["chain"]
        return ch[10][:Rd], ch[1][:Rd], ch[4][:Rd], ch[5][:Rd], ch[6][:Rd]
