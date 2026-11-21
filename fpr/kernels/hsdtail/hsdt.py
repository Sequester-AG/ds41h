"""hsdtail: the C1 HSD tail (verify head, sampler, DSpark drafter, draft head, Markov draws) faster, on top of kernels/head
New file: kernels/head is what the live config builds, so it stays untouched
HSDT subclasses hsd.HSD and DraftRankT drafter.DraftRank. Every change is an instance switch, all default off (= the
base path, the same kernels in the same order); HSDT.configure(...) sets them:

  mhcf     the drafter's mHC boundaries on the engine's fused boundary (kernels/mhc_fuse, the build the engine uses with
           ENG_MHCF=1): frow / frow_rs (collapse + RMSNorm, the RS fold summed inside) on the critical path, fmix + fb_mix
           (the mixes, read only by the NEXT boundary) on the drafter's mix stream. mhcf's contract: bit-identical to
           mhc.cu's boundary / boundary_rs (same adds in the same order).                                     [BITWISE]
  ctx      where the drafter's context KV (main_proj + 3 x wkv / norm / insert, 172 MB of bf16) forks off the verification:
           "start" (base: before the h all-gather: main_proj's GEMM holds every SM, the all-gather push waits ~45 us for
           one), "ag" (after the all-gather: main_proj and the verify head GEMM start together), "head" (after the verify
           head GEMM is issued: measured worse, main_proj starves the sampler kernels of SMs). Same kernels.   [BITWISE]
  par      the sampler's per-row stats / mass kernels (one program per row looping over 16 vocab blocks: ~19-20 us each at
           C1) as (row x block) grids whose last program per row merges the block partials with the same code. [BITWISE,
           checked: the same block reductions at the same num_warps, the same merge over the NB-lane vector]
  dmask    the drafter router writes no routed experts (ids -1, weights 0) for the rows of DUMMY drafting sequences (the
           padding requests of ranks without a drafting sequence: at C1 three of the four ranks), so the TP expert GEMMs
           stream only the real rows' experts. The real rows' outputs are unchanged [BITWISE for real rows: per-row
           expert math and combine order do not depend on the other rows]; dummy rows (never read) differ.
  w8qa     the drafter blocks' q_a | wkv GEMM (18.4 MB of bf16, cuBLAS split-K ~13.5 us at C1) on the fused W8A16 kernel
           from the checkpoint's e4m3 + ue8m0 (9.2 MB; exact dequant, fp32 accumulation; split-K HSDT_W8QA_S, default 8).
           SAME MATH (the drafter only).
  hg       kernels/head's HSD_ATTN_HG (the drafter's attention on the attn_hg kernel, buckets <= HSD_ATTN_HG_MAXR rows)
  w8qb/r2r kernels/head's HSD_W8QB / HSD_R2R module switches, set per variant (the bench loads w8qb weights only with
           HSD_W8QB=1 in the environment at import).
  ctxtp    the drafter's main_proj (context KV input, 157 MB of bf16 per GPU and step) split over the 4 GPUs for context
           buckets of <= 32 rows: all-gather of the aux rows, each GPU computes its 1,280 output columns for every rank's
           rows (39 MB), the slices go back to the row owners; on the context side stream. SAME MATH (cuBLAS's split of
           the sliced GEMM differs); the drafter only.
  rtag     the drafter router + its all-gather as ONE kernel per GPU (rtag.cu: f push, gate GEMV, top-3 by the last CTA,
           routing-row push; the AG stage becomes a wait) for drafter buckets of <= 4 rows per rank. SAME MATH; honours dmask.
  xg       HSD's small all-gathers (sampler words g1 / g2 / g3 / d0..d2; the h / hd rows at C1 sizes) as ONE lamport kernel
           per rank (xg.cu: stores into every rank's mailbox, words carry the arrival, spin, copy out) instead of comm.cu's
           push + fence + flag + wait kernel pair. BITWISE (a copy).
  c1d      the 3 drafter blocks as ONE c1mega launch of 3 layers per GPU, then the final boundary on every GPU (the
           block outputs are replicated: no hd all-gather), for steps where one rank holds the only drafting sequence
           (C1). SAME MATH for the drafter (fp8 TP slices of the exact checkpoint weights, fp32 accumulation, other order).
  dhead    the draft head alone on dtail.cu (head-only mode: per-row e4m3 quantization + the vocab-sharded e4m3 GEMV
           streaming at ~3.3 TB/s, base logits to DraftHead.out) instead of rowq + cuBLASLt's scaled_mm; the Markov draws
           stay HSD's (with xg: lamport exchanges). SAME MATH for the drafter (<= 12 draft rows).
  dtail    draft head (e4m3) + the k Markov draws as ONE persistent kernel per GPU (dtail.cu: in-kernel grid barriers and
           a lamport exchange between the 4 GPUs) for steps of <= 16 draft rows (C1: 4 requests x 3). SAME MATH for the
           drafter (fp32 sums in another order than cuBLASLt / the Triton kernels; the same noise, the same tie rule).
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KH = os.path.join(os.path.dirname(HERE), "head")
PHYS = os.path.dirname(os.path.dirname(HERE))
for _p in (KH, os.path.join(os.path.dirname(HERE), "mhc_fuse")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import drafter as DR  # noqa: E402
import head as HD  # noqa: E402
import hsd as HS  # noqa: E402
import sampler as S  # noqa: E402

from vllm.triton_utils import tl, tldevice, triton  # noqa: E402
from sampler import (_allowed, _block_stats, _div_t, _last_program, _logsumexp, _merge, _nan_to_neg_inf,  # noqa: E402
                     _pick, _ratio_terms)

_MHCF = None


def mhcf_ext():
    """The engine's mhcf build (engine/build/eng_mhcf: prebuilt for the live config's ENG_MHCF=1)."""
    global _MHCF
    if _MHCF is None:
        import mhcf
        _MHCF = mhcf.build(build_directory=os.path.join(PHYS, "engine/build/eng_mhcf"))
    return _MHCF


_RTAG = None
RTAG_NC = int(os.environ.get("HSDT_RTAG_NC", "32"))   # CTAs per destination (grid 4 x NC; 128 experts / grid per CTA)


def rtag_ext():
    global _RTAG
    if _RTAG is None:
        from torch.utils.cpp_extension import load
        bd = os.path.join(HERE, "build", "rtag")
        os.makedirs(bd, exist_ok=True)
        _RTAG = load(name="hsdt_rtag", sources=[os.path.join(HERE, "rtag.cu")], build_directory=bd, verbose=False,
                     extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])
    return _RTAG


_XG = None
XG_SITES = {"g1": 4096, "g2": 4096, "g3": 4096, "d0": 4096, "d1": 4096, "d2": 4096, "h": 16384, "hd": 16384}  # site: max words


def xg_ext():
    global _XG
    if _XG is None:
        from torch.utils.cpp_extension import load
        bd = os.path.join(HERE, "build", "xg")
        os.makedirs(bd, exist_ok=True)
        _XG = load(name="hsdt_xg", sources=[os.path.join(HERE, "xg.cu")], build_directory=bd, verbose=False,
                   extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])
    return _XG


_DTAIL = None
CTP_MAX, CTP_COLS = 32, 1280   # ctxtp: context rows per rank, main_proj output columns per GPU (5,120 / 4)


def dtail_ext():
    global _DTAIL
    if _DTAIL is None:
        from torch.utils.cpp_extension import load
        bd = os.path.join(HERE, "build", "dtail")
        os.makedirs(bd, exist_ok=True)
        _DTAIL = load(name="hsdt_dtail", sources=[os.path.join(HERE, "dtail.cu")], build_directory=bd, verbose=False,
                      extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])
    return _DTAIL


# ============================================================================================ drafter
class DraftRankT(DR.DraftRank):
    """DraftRank + the mhcf boundary switch (self.mhcf_on). Made by class conversion of HSD's DraftRanks (adds state only)."""

    def _t_init(self):
        if getattr(self, "_t_ready", False):
            return
        self._t_ready = True
        self.mhcf_on = False
        self.w8qa_on = False
        self._fnf = {}
        self._dmask = None

    def enable_mhcf(self, rd, on=True):
        self._t_init()
        if on:
            F = mhcf_ext()
            with torch.cuda.device(self.dev):
                for i, B in enumerate(self.blocks):   # blocks are shared by every bucket's DraftRank of this device
                    if "fa_f" not in B:
                        B["fa_f"] = F.prep_f(rd.get(f"mtp.{i}.hc_attn_fn").float().to(self.dev).contiguous())
                        B["ff_f"] = F.prep_f(rd.get(f"mtp.{i}.hc_ffn_fn").float().to(self.dev).contiguous())
                    self._fnf[B["fa"].data_ptr()] = B["fa_f"]
                    self._fnf[B["ff"].data_ptr()] = B["ff_f"]
        self.mhcf_on = bool(on)

    def _mixes(self, xn, n, fn, sc, bs, pre_o, post_o, comb_o):
        """fmix + fb_mix of this boundary on the mix stream (joined by the next boundary's _join_mix)."""
        F = _MHCF
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        with torch.cuda.stream(self.mix_stream):
            self.mix_stream.wait_event(ev)
            F.fmix(xn, n, self._fnf[fn.data_ptr()], self.b["part_mhc"], 0)
            F.fb_mix(self.b["part_mhc"], n, sc, bs, self.eps, self.hc_eps, self.iters, pre_o, post_o, comb_o)
            ev2 = torch.cuda.Event()
            ev2.record(self.mix_stream)
        self.mix_ev = ev2

    def boundary(self, y, rs, post, comb, lag, fn, sc, bs, nw, xn, pre_o, post_o, comb_o, out, mix=True):
        if not getattr(self, "mhcf_on", False) or y.shape[0] > 320:
            return super().boundary(y, rs, post, comb, lag, fn, sc, bs, nw, xn, pre_o, post_o, comb_o, out, mix)
        n = y.shape[0]
        self._join_mix()   # the previous boundary's mixes (post / comb / lag here; its fb_mix read part_mhc)
        _MHCF.frow(y, rs, post, comb, lag, nw, self.eps, xn, out, self.b["ptok0"])
        if mix:
            self._mixes(xn, n, fn, sc, bs, pre_o, post_o, comb_o)

    # ctxtp: the context KV with main_proj split over the 4 GPUs, in three phases on the context side stream. HSDT._steps
    # issues them between its collective yields, so in run_all every rank's push is issued before any rank waits (a host
    # sync inside a rank's segment, e.g. cuBLAS's first workspace on a stream, would otherwise deadlock the peers' waits).
    def ctp_push(self, H_, r, R):
        G = H_.ctp_g
        fl, ep = G.sites["aux"]
        G.e.ag_push(self.b["aux"][:R], r, [x[: DR.W * R] for x in H_.ctp_aux], fl, ep[r], G.ncta["aux"])

    def ctp_mid(self, H_, r, R):
        G = H_.ctp_g
        nc = G.ncta["aux"]
        fl, ep = G.sites["aux"]
        G.e.wait(fl[r], ep[r], DR.W * nc)
        part = H_.ctp_part[r][: DR.W * R]
        torch.mm(H_.ctp_aux[r][: DR.W * R], self.main_proj[r * CTP_COLS:(r + 1) * CTP_COLS].t(), out=part)
        fl2, ep2 = G.sites["mxp"]
        G.e.ag_push(part, r, [x[: DR.W * DR.W * R] for x in H_.ctp_recv], fl2, ep2[r], nc)

    def ctp_tail(self, H_, r, R):
        G = H_.ctp_g
        b, kd = self.b, self.kd
        fl2, ep2 = G.sites["mxp"]
        G.e.wait(fl2[r], ep2[r], DR.W * G.ncta["mxp"])
        rv = H_.ctp_recv[r][: DR.W * DR.W * R].view(DR.W, DR.W, R, CTP_COLS)[:, r]   # [source rank, row, its columns]
        b["mx_raw"][:R].view(R, DR.W, CTP_COLS).copy_(rv.transpose(0, 1))
        kd.norm_rope(b["mx_raw"][:R], self.main_norm, b["main_x"][:R], self.eps, None, None)
        for i, B in enumerate(self.blocks):
            torch.mm(b["main_x"][:R], B["wqkv_a"][1280:].t(), out=b["kvc"][:R])
            kd.norm_rope(b["kvc"][:R], B["kv_norm"], b["kv_ctx"][:R], self.eps, b["ctx_pos"][:R], self.cos_sin)
            kd.rope_q_insert(b["dq"][:R], b["kv_ctx"][:R], b["ctx_pos"][:R], self.cos_sin, b["ctx_slot"][:R], self.cache[i],
                             self.page_stride, DR.PAGE_SHIFT, True)

    def enable_w8qa(self, rd, on=True):
        self._t_init()
        if on:
            ad = os.path.join(os.path.dirname(HERE), "attn_dense")
            if ad not in sys.path:
                sys.path.insert(0, ad)
            import w8a16 as W8
            S_ = int(os.environ.get("HSDT_W8QA_S", "8"))
            with torch.cuda.device(self.dev):
                for i, B in enumerate(self.blocks):
                    if "w8qa" not in B:
                        g = lambda n: rd.get(f"mtp.{i}.attn.{n}").to(self.dev).contiguous()  # noqa: E731
                        w = torch.cat([g("wq_a.weight"), g("wkv.weight")], 0).contiguous()
                        sc = torch.cat([g("wq_a.scale"), g("wkv.scale")], 0).contiguous()
                        B["w8qa"] = W8.Gemm(W8.build(), w, sc, S_)
        self.w8qa_on = bool(on)

    def attention(self, i, insert=True):
        """drafter.DraftRank.attention with the q_a | wkv GEMM on W8A16 when w8qa is on (else the base method)."""
        B = self.blocks[i]
        if not getattr(self, "w8qa_on", False) or B.get("w8qa") is None or self.dense_fp8 or self.Rd > 32:
            return super().attention(i, insert)
        # = drafter.DraftRank.attention (dense_fp8 off) with its first GEMM on W8A16; the module switches read at call time
        from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache
        b, kd, Rd = self.b, self.kd, self.Rd
        NH, HD = DR.NH, DR.HD
        B["w8qa"](b["a"], b["qakv"])
        kd.norm_rope2(b["qakv"][:, :1280], B["q_norm"], b["qr"], b["qakv"][:, 1280:], B["kv_norm"], b["kv"], self.eps,
                      b["q_pos"], self.cos_sin)
        q3 = b["q"].view(Rd, NH, HD)
        if DR.HSD_W8QB and B.get("w8qb") is not None and Rd <= DR.HSD_W8QB_MAXR:
            B["w8qb"](b["qr"], b["q"].view(Rd, NH * HD), pos=b["q_pos"], cs=self.cos_sin, kv=b["kv"] if insert else None,
                      slot=b["q_slot"] if insert else None, cache=self.cache[i] if insert else None,
                      page_stride=self.page_stride, shift=DR.PAGE_SHIFT)
        else:
            if not self._sk(B, "wq_b", b["qr"], b["q"].view(Rd, NH * HD), 32768, 1280):
                torch.mm(b["qr"], B["wq_b"].t(), out=b["q"].view(Rd, NH * HD))
            kd.rope_q_insert(q3, b["kv"], b["q_pos"], self.cos_sin, b["q_slot"], self.cache[i], self.page_stride,
                             DR.PAGE_SHIFT, bool(insert))
        if DR.HSD_ATTN_HG and Rd <= DR.HSD_ATTN_HG_MAXR:
            import importlib
            _EA = importlib.import_module("engine.attn")
            kc = self.k_cache[i]
            _EA.attn_hg().sparse_decode(b["q"].view(Rd, NH, HD), b["o"].view(Rd, NH, HD), kc.select(2, 0) if kc.dim() == 4 else kc,
                                        b["idx"], b["lens"], B["sink"], float(HD ** -0.5), cluster=0, spread=True, pdl=True, hg=4,
                                        skip_empty=True)
        else:
            flash_mla_with_kvcache(q=b["q"], k_cache=self.k_cache[i], block_table=None, cache_seqlens=None, head_dim_v=HD,
                                   tile_scheduler_metadata=self.sched, is_fp8_kvcache=True, indices=b["idx"],
                                   topk_length=b["lens"], softmax_scale=HD ** -0.5, attn_sink=B["sink"], out=b["o"])
        o3 = b["o"].view(Rd, NH, HD)
        kd.inv_rope(o3, b["q_pos"], self.cos_sin)
        if getattr(self, "tpc", None) is not None:   # HSD_TPWO: wo_a / wo_b in the tps / wo_tp / tpr stages
            return
        if not self._sk(B, "wo_a", b["o"].view(Rd, NH * HD), b["z"], 8192, 4096, 1024, 4096):
            torch.bmm(o3.view(Rd, 8, 4096).transpose(0, 1), B["wo_aT"], out=b["z"].view(Rd, 8, 1024).transpose(0, 1))
        if not self._sk(B, "wo_b", b["z"], b["y"], 5120, 8192):
            torch.mm(b["z"], B["wo_b"].t(), out=b["y"])

    def router(self, i):
        rt_ = getattr(self, "_rtag", None)
        if rt_ is not None and self.Rd <= 4:
            H_, r, dsl = rt_
            b, B = self.b, self.blocks[i]
            pc = H_.dcomm[self.n]
            _RTAG.rtag(b["f"], B["gate_wb"], B["gate_b"], H_.rt_lg[r], H_.rt_cnt[r], b["route_row"].view(torch.int32),
                       dsl, r * self.n, self.k, r, RTAG_NC, pc.gf, pc.gr, H_.rt_fl, H_.rt_ep[r])
            return
        if getattr(self, "_dmask", None) is not None:
            return _router_dmask(self, i)
        return super().router(i)

    def boundary_moe(self, post, comb, lag, fn, sc, bs, nw, xn, pre_o, post_o, comb_o, out, mix=True):
        if not getattr(self, "mhcf_on", False) or self.fold is None or self.Rd > 320:
            return super().boundary_moe(post, comb, lag, fn, sc, bs, nw, xn, pre_o, post_o, comb_o, out, mix)
        b = self.b
        flag, epoch, sig = self.fold
        self.fold = None
        self._join_mix()
        # = comm.cu k_rs_sum (slots in rank order, fp32, one bf16 rounding) + the boundary, as mhc.cu's boundary_rs
        _MHCF.frow_rs(b["rs"], flag, epoch, sig, b["Xm"], post, comb, lag, nw, self.eps, xn, out, b["ptok0"])
        if mix:
            self._mixes(xn, self.Rd, fn, sc, bs, pre_o, post_o, comb_o)


@triton.jit
def _router_m_kernel(logits_ptr, bias_ptr, ids_ptr, w_ptr, dslot_ptr, r0, NE_: tl.constexpr, K_: tl.constexpr,
                     ROW_WORDS: tl.constexpr, SCALE: tl.constexpr, KQ: tl.constexpr):
    # = drafter._router_kernel, plus: rows of a dummy drafting sequence (g_dslot < 0) get no routed experts
    r = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, NE_)
    x = tl.load(logits_ptr + r * NE_ + e)
    sp = tl.where(x > 20.0, x, tldevice.log1p(tl.exp(x)))
    sc = tl.sqrt(sp)
    v = sc + tl.load(bias_ptr + e)
    lanes = tl.arange(0, 8)
    ids = tl.full((8,), -1, tl.int32)
    ws = tl.zeros((8,), tl.float32)
    for t in tl.static_range(K_):
        i = tl.argmax(v, axis=0)
        ids = tl.where(lanes == t, i.to(tl.int32), ids)
        ws = tl.where(lanes == t, tl.sum(tl.where(e == i, sc, 0.0), axis=0), ws)
        v = tl.where(e == i, float("-inf"), v)
    ws = ws / (tl.sum(ws, axis=0) + 1e-20) * SCALE
    live = tl.load(dslot_ptr + r0 + r // KQ) >= 0
    ids = tl.where(live, ids, -1)
    ws = tl.where(live, ws, 0.0)
    tl.store(ids_ptr + r * ROW_WORDS + lanes, ids, mask=lanes < 6)
    tl.store(w_ptr + r * ROW_WORDS + 6 + lanes, ws, mask=lanes < 6)


def _router_dmask(d, i):
    b, B = d.b, d.blocks[i]
    lg = torch.mm(b["f"], B["gate_wb"].t(), out_dtype=torch.float32)
    rr = b["route_row"].view(torch.int32)
    dslot, r0 = d._dmask
    _router_m_kernel[(d.Rd,)](lg, B["gate_b"], rr, rr.view(torch.float32), dslot, r0, NE_=DR.NE, K_=DR.TOPK, ROW_WORDS=16,
                              SCALE=1.5, KQ=d.k, num_warps=4)


def parse_switches(spec: str) -> dict:
    """ENG_HSDT spec -> configure() kwargs: comma list of switch names ("mhcf,par,dmask") and name=value ("ctx=ag");
    '+' also separates."""
    out = {}
    for it in spec.replace("+", ",").split(","):
        it = it.strip()
        if not it or it in ("1", "on"):
            continue
        if "=" in it:
            k_, v_ = it.split("=", 1)
            out[k_] = v_ if k_ == "ctx" else v_ not in ("0", "false", "off")
        else:
            out[it] = True
    bad = set(out) - {"mhcf", "ctx", "par", "dmask", "w8qb", "r2r", "dtail", "ctxtp", "hg", "w8qa", "rtag", "xg", "dhead",
                      "c1d"}
    assert not bad, f"ENG_HSDT: unknown switches {sorted(bad)}"
    return out


def as_t(d):
    if not isinstance(d, DraftRankT):
        d.__class__ = DraftRankT
    d._t_init()
    return d


# ============================================================================================ sampler (par)
VB = S.VERIFY_BLOCK


@triton.jit
def _stats_blk_kernel(out_ptr, part_ptr, counts_ptr, logits_ptr, logits_stride, draft_ptr, num_logits, row_req_ptr,
                      temp_ptr, bm_ptr, bm_stride, local_vocab, vocab_start, BLOCK: tl.constexpr, NB: tl.constexpr,
                      WS: tl.constexpr, HAS_MASK: tl.constexpr):
    # program (row, b) = iteration b of sampler._stats_rows_kernel's loop (the same code); the last program of the row
    # merges the NB partials exactly as that kernel's epilogue (its NB-lane vectors ms / ss / ow)
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    temp = tl.load(temp_ptr + tl.load(row_req_ptr + row)).to(tl.float32)
    t = tl.where(temp > 0.0, temp, 1.0)
    dtok = tl.load(draft_ptr + row + 1, mask=row + 1 < num_logits, other=-1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < local_vocab
    ok = _allowed(bm_ptr, bm_stride, row, vocab_start + offs, mask, HAS_MASK)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
    x = _div_t(tl.where(ok, x, float("-inf")), t)
    if temp == 0.0:
        m, i = tl.max(x, axis=0, return_indices=True)
        sv = (vocab_start + b * BLOCK + i).to(tl.float32)
    else:
        m, sv = _block_stats(x)
    d = dtok - vocab_start - b * BLOCK
    dmask = (d >= 0) & (d < BLOCK) & (b * BLOCK + d < local_vocab)
    owned = tl.load(logits_ptr + row * logits_stride + b * BLOCK + d, mask=dmask, other=float("-inf")).to(tl.float32)
    if HAS_MASK:
        dok = _allowed(bm_ptr, bm_stride, row, vocab_start + b * BLOCK + d, dmask, HAS_MASK)
        owned = tl.where(dok, owned, float("-inf"))
    p = part_ptr + (row * NB + b) * 3
    tl.store(p, m)
    tl.store(p + 1, sv)
    tl.store(p + 2, owned / t)
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (row * NB + blocks) * 3
        ms = tl.load(q, volatile=True)
        ss = tl.load(q + 1, volatile=True)
        ow = tl.load(q + 2, volatile=True)
        o = out_ptr + row * WS
        if temp == 0.0:
            clean = _nan_to_neg_inf(ms)
            tl.store(o, tl.max(clean, axis=0))
            tl.store(o + 1, _pick(ss, blocks, tl.argmax(clean, axis=0)))
        else:
            top, total = _merge(ms, ss)
            tl.store(o, top)
            tl.store(o + 1, total)
        tl.store(o + 2, tl.max(ow, axis=0))


@triton.jit
def _mass_blk_kernel(out_ptr, part_ptr, counts_ptr, g1_ptr, logits_ptr, logits_stride, cache_ptr, cache_stride_0,
                     cache_stride_1, vp_logit_ptr, vp_lse_ptr, vp_stride, draft_ptr, num_logits, row_req_ptr, row_local_ptr,
                     temp_ptr, bm_ptr, bm_stride, local_vocab, vocab_start, num_spec, BLOCK: tl.constexpr,
                     NB: tl.constexpr, TP: tl.constexpr, S_: tl.constexpr, WS: tl.constexpr, HAS_MASK: tl.constexpr):
    # program (row, b) = block b of sampler._mass_rows_kernel (prologue recomputed per program: the same values);
    # the last program sums the NB block masses with that kernel's tl.sum over its NB-lane vector
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    req = tl.load(row_req_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    step = tl.load(row_local_ptr + row).to(tl.int64)
    start = row - step
    has_next = row + 1 < num_logits
    next_pos = tl.load(row_local_ptr + row + 1, mask=has_next, other=0)
    not_bonus = has_next & (next_pos != 0)
    rejected = tl.load(draft_ptr + row + 1, mask=has_next, other=-1)
    need = (temp != 0.0) & not_bonus & (rejected >= 0) & (step >= 1) & (step < num_spec)
    mb = 0.0
    if need:
        lanes = tl.arange(0, S_)
        terms, _, _ = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, step, temp, TP,
                                     S_, WS)
        drafted = tl.load(draft_ptr + start + lanes + 1, mask=lanes < step, other=-1)
        log_p = 0.0
        for s in tl.static_range(S_):
            take = (s < step) & (_pick(drafted, lanes, s) >= 0)
            log_p = tl.where(take, tl.minimum(log_p + _pick(terms, lanes, s), 0.0), log_p)
        stats = g1_ptr + (tl.arange(0, TP) * num_logits + row) * WS
        lse = _logsumexp(tl.load(stats), tl.load(stats + 1))
        d_lse = tl.load(vp_lse_ptr + req * vp_stride + step)
        p_tau = tl.exp(log_p)
        offs = b * BLOCK + tl.arange(0, BLOCK)
        mask = offs < local_vocab
        ok = _allowed(bm_ptr, bm_stride, row, vocab_start + offs, mask, HAS_MASK)
        x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
        x = _div_t(tl.where(ok, x, float("-inf")), temp)
        d_lp = _div_t(tl.load(cache_ptr + req * cache_stride_0 + step * cache_stride_1 + offs, mask=mask,
                                other=float("-inf")).to(tl.float32), temp) - d_lse
        mb = tl.sum(tl.maximum(p_tau * tl.exp(x - lse) - tl.exp(d_lp), 0.0), axis=0)
    tl.store(part_ptr + row * NB + b, mb)
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        parts = tl.load(part_ptr + row * NB + blocks, volatile=True)
        tl.store(out_ptr + row * WS, tl.sum(parts, axis=0))


class ParSampler:
    """Per-rank scratch of the (row x block) stats / mass kernels."""

    def __init__(self, st, max_rows):
        dv = st.stats.device
        self.nb = st.vblocks
        self.part_s = torch.zeros(max_rows * self.nb * 3, dtype=torch.float32, device=dv)
        self.part_m = torch.zeros(max_rows * self.nb, dtype=torch.float32, device=dv)
        self.cnt_s = torch.zeros(max_rows, dtype=torch.int32, device=dv)
        self.cnt_m = torch.zeros(max_rows, dtype=torch.int32, device=dv)

    def stats(self, st, logits, m):
        T = logits.shape[0]
        bm = m.bitmask
        _stats_blk_kernel[(T, self.nb)](
            st.stats, self.part_s, self.cnt_s, logits, logits.stride(0), m.draft_sampled, T, m.row_req, st.temp,
            bm if bm is not None else st.stats, bm.stride(0) if bm is not None else 0, st.local, st.start,
            BLOCK=VB, NB=self.nb, WS=S.WS_V, HAS_MASK=bm is not None, num_warps=8)
        return st.stats[:T]

    def mass(self, st, logits, g1, m):
        T = logits.shape[0]
        bm = m.bitmask
        _mass_blk_kernel[(T, self.nb)](
            st.resid, self.part_m, self.cnt_m, g1, logits, logits.stride(0), st.cache, st.cache.stride(0),
            st.cache.stride(1), st.token_logit, st.lse, st.lse.stride(0), m.draft_sampled, T, m.row_req, m.row_local,
            st.temp, bm if bm is not None else st.stats, bm.stride(0) if bm is not None else 0, st.local, st.start,
            st.n_spec, BLOCK=VB, NB=self.nb, TP=st.ranks, S_=triton.next_power_of_2(st.n_spec + 1), WS=S.WS_V,
            HAS_MASK=bm is not None, num_warps=8)
        return st.resid[:T]


# ============================================================================================ HSD
class HSDT(HS.HSD):
    """hsd.HSD + the switches above (configure()). Constructed like HSD; keeps the reader for the mhcf weights."""

    def __init__(self, devs, reader, cfg, *a, **kw):
        super().__init__(devs, reader, cfg, *a, **kw)
        self.rd = reader
        # w8qb / r2r / hg mirror kernels/head drafter.py's module switches (set from the environment at its import)
        self.sw = dict(mhcf=False, ctx="start", par=False, dmask=False, w8qb=DR.HSD_W8QB, r2r=DR.HSD_R2R, orig=False,
                       dtail=False, ctxtp=False, hg=DR.HSD_ATTN_HG, w8qa=False, rtag=False, xg=False, dhead=False,
                       c1d=False)
        self.dt = None
        self.ctp_g = None
        self.rt_fl = None
        self.xg_box = None
        for d in self.ranks0:
            as_t(d)
        self.par = [ParSampler(self.st[r], self.h_all[r].shape[0]) for r in range(HS.W)]
        self._no_dslot = [torch.zeros(0, dtype=torch.int64, device=dv) for dv in self.devs]
        self._no_f32 = [torch.zeros(0, dtype=torch.float32, device=dv) for dv in self.devs]

    def configure(self, mhcf=None, ctx=None, par=None, dmask=None, w8qb=None, r2r=None, orig=None, dtail=None, ctxtp=None,
                  hg=None, w8qa=None, rtag=None, xg=None, dhead=None, c1d=None):
        """Switch changes need a recapture of HSD's graphs (the engine recaptures per arm; the bench per variant).
        w8qb / r2r set kernels/head drafter.py's module switches (all DraftRanks of the process)."""
        if xg is not None:
            self.sw["xg"] = bool(xg)
            if xg and self.xg_box is None:
                xg_ext()
                self.xg_box, self.xg_peers, self.xg_ctr = {}, {}, {}
                for site, nmax in XG_SITES.items():
                    self.xg_box[site] = [torch.full((2 * HS.W * nmax,), -1, dtype=torch.int32, device=dv) for dv in self.devs]
                    self.xg_peers[site] = [x.data_ptr() for x in self.xg_box[site]]
                    self.xg_ctr[site] = [torch.zeros(2, dtype=torch.int32, device=dv) for dv in self.devs]
                for dv in self.devs:
                    torch.cuda.synchronize(dv)
        if rtag is not None:
            self.sw["rtag"] = bool(rtag)
            if rtag and self.rt_fl is None:
                rtag_ext()
                z = lambda n, dv, dt=torch.int32: torch.zeros(n, dtype=dt, device=dv)  # noqa: E731
                self.rt_fl = [z(4, dv) for dv in self.devs]
                self.rt_ep = [z(1, dv) for dv in self.devs]
                self.rt_cnt = [z(1, dv) for dv in self.devs]
                self.rt_lg = [z(4 * DR.NE, dv, torch.float32) for dv in self.devs]
                for dv in self.devs:
                    torch.cuda.synchronize(dv)
        if ctxtp is not None:
            self.sw["ctxtp"] = bool(ctxtp)
            if ctxtp and self.ctp_g is None:
                import comm_head
                self.ctp_g = comm_head.PeerGather(self.devs, ncta={"aux": 8, "mxp": 8}, sites=("aux", "mxp"))
                bf = torch.bfloat16
                self.ctp_aux = [torch.zeros(HS.W * CTP_MAX, 3 * HS.H, dtype=bf, device=dv) for dv in self.devs]
                self.ctp_part = [torch.zeros(HS.W * CTP_MAX, CTP_COLS, dtype=bf, device=dv) for dv in self.devs]
                self.ctp_recv = [torch.zeros(HS.W * HS.W * CTP_MAX, CTP_COLS, dtype=bf, device=dv) for dv in self.devs]
                for dv in self.devs:
                    torch.cuda.synchronize(dv)
        if c1d is not None:
            self.sw["c1d"] = bool(c1d)
            kind_ = c1d if isinstance(c1d, str) else "base"     # True = today's c1mega instance; "x2b": c1d.KINDS (tests)
            insts_ = self.__dict__.setdefault("_c1d_insts", {})
            if c1d and kind_ not in insts_:
                import c1d as C1D
                from types import SimpleNamespace as NS_
                d0 = self.ranks0[0]
                mhcf_ext()
                insts_[kind_] = C1D.DrafterC1M(self, self.rd, NS_(norm_eps=d0.eps, hc_eps=d0.hc_eps, hc_iters=d0.iters),
                                               kind=kind_)
            if c1d:
                self.c1d = insts_[kind_]
            if c1d and all((x, 1) in self.drk for x in range(HS.W)):   # packs for every owner now (H2D + sync: never
                dr1 = [self.drk[(x, 1)] for x in range(HS.W)]           # inside a capture or next to a spinning kernel)
                for o_ in range(HS.W):
                    self.c1d.packs(o_, dr1)
        if dhead is not None:
            self.sw["dhead"] = bool(dhead)
        if dtail is not None:
            self.sw["dtail"] = bool(dtail)
        if (dtail or dhead) and self.dt is None:
            if True:
                dtail_ext()
                self.dt = []
                for dv in self.devs:
                    with torch.cuda.device(dv):
                        self.dt.append(dict(part=torch.zeros(5 * 132 * 5, dtype=torch.float32, device=dv),
                                            bar=torch.zeros(1, dtype=torch.int64, device=dv),
                                            xloc=torch.full((2 * 4 * 5 * 4 * 8,), -1, dtype=torch.int32,
                                                            device=dv).view(torch.float32),
                                            dbg=torch.zeros(132 * 16 if os.environ.get("HSDT_DTAIL_PROF") == "1" else 0,
                                                            dtype=torch.int64, device=dv)))
                self.dt_peers = [x["xloc"].data_ptr() for x in self.dt]
                for dv in self.devs:
                    torch.cuda.synchronize(dv)
        if orig is not None:   # hsd.HSD._steps itself (the reference for this file's copy of it)
            self.sw["orig"] = bool(orig)
        for d in list(self.ranks0) + list(self.drk.values()):
            as_t(d)._dmask = None   # set per step by _steps when dmask / rtag are on
            d._rtag = None
        if dmask is not None:
            self.sw["dmask"] = bool(dmask)
        if w8qb is not None:
            self.sw["w8qb"] = DR.HSD_W8QB = bool(w8qb)
        if r2r is not None:
            self.sw["r2r"] = DR.HSD_R2R = bool(r2r)
        if hg is not None:
            self.sw["hg"] = DR.HSD_ATTN_HG = bool(hg)
        if w8qa is not None:
            self.sw["w8qa"] = bool(w8qa)
            for d in list(self.ranks0) + list(self.drk.values()):
                as_t(d).enable_w8qa(self.rd, bool(w8qa))
        if mhcf is not None:
            self.sw["mhcf"] = bool(mhcf)
            for d in list(self.ranks0) + list(self.drk.values()):
                as_t(d).enable_mhcf(self.rd, bool(mhcf))
        if ctx is not None:
            assert ctx in ("start", "ag", "head")
            self.sw["ctx"] = ctx
        if par is not None:
            self.sw["par"] = bool(par)
        return self

    def draft_rank(self, r, D_pad):
        new = (r, D_pad) not in self.drk
        d = as_t(super().draft_rank(r, D_pad))
        if new and self.sw["mhcf"]:
            d.enable_mhcf(self.rd, True)
        if new and self.sw.get("w8qa"):
            d.enable_w8qa(self.rd, True)
        return d

    def _xg_ok(self, site, inp):
        if not self.sw.get("xg") or site not in XG_SITES or not inp.is_contiguous():
            return False
        return inp.numel() * inp.element_size() // 4 <= XG_SITES[site] and (inp.numel() * inp.element_size()) % 4 == 0

    def _xg(self, site, r, inp, out):
        _XG.xgather(inp.view(-1).view(torch.int32), out.view(-1).view(torch.int32), self.xg_box[site][r], self.xg_peers[site],
                    r, self.xg_ctr[site][r])

    # ------------------------------------------------------------------ drivers (hsd.HSD's + the "nop" yield)
    def run_all(self, h_rows, aux_rows):
        """hsd.HSD.run_all, plus ("nop",): a point every rank reaches before any continues (nothing issued)."""
        gens = [self._steps(r, h_rows[r], aux_rows[r]) for r in range(HS.W)]
        while True:
            reqs = []
            for r in range(HS.W):
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
            if kind == "nop":
                continue
            if kind == "ag" and self._xg_ok(reqs[0][1], reqs[0][2]):
                for r in range(HS.W):   # every rank's kernel launched back to back (each spins for its peers)
                    with torch.cuda.device(self.devs[r]):
                        self._xg(reqs[0][1], r, reqs[r][2], reqs[0][3][r])
            elif kind == "ag":
                self._gather(reqs[0][1], [x[2] for x in reqs], reqs[0][3])
            elif kind == "dag":
                self._dag([self.draft_rank(r, reqs[0][1]) for r in range(HS.W)])
            elif kind in ("tps", "tpr"):
                comm = self.dcomm[self.draft_rank(0, reqs[0][1]).n]
                for r in range(HS.W):
                    with torch.cuda.device(self.devs[r]):
                        (comm.tps_rank if kind == "tps" else comm.tpr_rank)(r)
            elif kind == "drs":
                self._drs([self.draft_rank(r, reqs[0][1]) for r in range(HS.W)])
            else:
                raise ValueError(f"HSDT.run_all: unknown step kind {kind!r}")

    def run(self, r, h_rows, aux_rows):
        """hsd.HSD.run (rank r alone, capturable), plus ("nop",)."""
        assert self.mode == "peer"
        comm = self.dcomm[self.cur.D_pad]
        for req in self._steps(r, h_rows, aux_rows):
            k_ = req[0]
            if k_ == "nop":
                continue
            if k_ == "ag" and self._xg_ok(req[1], req[2]):
                self._xg(req[1], r, req[2], req[3][r])
            elif k_ == "ag":
                _, site, inp, outs = req
                flags, epochs = self.gather._site(site)
                nc = self.gather.ncta.get(site, 1)
                self.gather.e.ag_push(inp, r, outs, flags, epochs[r], nc)
                self.gather.e.wait(flags[r], epochs[r], HS.W * nc)
            elif k_ == "dag":
                comm.ag_f_rank(r)
            elif k_ == "tps":
                comm.tps_rank(r)
            elif k_ == "tpr":
                comm.tpr_rank(r)
            elif k_ == "drs":
                comm.rs_rank(r)
            else:
                raise ValueError(f"HSDT.run: unknown step kind {k_!r}")

    # ------------------------------------------------------------------ device: the step of rank r (generator)
    def _steps(self, r, h_rows, aux_rows):
        """hsd.HSD._steps (GLUE / NOISE_SIDE / CTX_SIDE paths, which the engine uses) with the ctx / par switches."""
        if self.sw["orig"]:
            yield from super()._steps(r, h_rows, aux_rows)
            return
        assert HS.GLUE and HS.CTX_SIDE, "HSDT follows the GLUE + CTX_SIDE path of hsd.py"
        c, tb, st, k = self.cur, self.tab[r], self.st[r], self.k
        T, n, Dp, NT, S_pad = c.T, c.n_req, c.D_pad, c.NT, c.S_pad
        d = self.draft_rank(r, Dp)
        sts = self.st
        if h_rows is not None and h_rows.data_ptr() != self.h_own[r].data_ptr():
            self.h_own[r][: c.n_s[r]].copy_(h_rows[: c.n_s[r]])
        if aux_rows is not None and aux_rows.data_ptr() != d.b["aux"].data_ptr():
            d.b["aux"][: c.n_t[r]].copy_(aux_rows[: c.n_t[r]].reshape(c.n_t[r], 3 * HS.H))
        HS._pre_glue[(triton.cdiv(max(T, NT), 1024),)](
            tb["draft_sampled"], self.by_slot[r], self.by_slot[r].stride(0), tb["tok_slot"], tb["tok_j"], T,
            d.b["ctx_pos"], d.b["ctx_slot"], tb["tpos"], tb["tslot"], NT, BLOCK=1024)
        cur = torch.cuda.current_stream()

        ctp = self.sw["ctxtp"] and NT <= CTP_MAX
        sctx = self.side_ctx[r]

        def fork_ctx():
            evc = torch.cuda.Event()
            evc.record(cur)
            self.side_ctx[r].wait_event(evc)
            with torch.cuda.stream(self.side_ctx[r]):
                d.context_kv(NT)
                evc2 = torch.cuda.Event()
                evc2.record(self.side_ctx[r])
            return evc2

        if ctp:   # phase 1: the aux rows to every rank (the side stream forks here)
            evc = torch.cuda.Event()
            evc.record(cur)
            sctx.wait_event(evc)
            with torch.cuda.stream(sctx):
                d.ctp_push(self, r, NT)
            evc2 = None
        else:
            evc2 = fork_ctx() if self.sw["ctx"] == "start" else None
        # ---- verification
        yield ("ag", "h", self.h_own[r][:S_pad], [x[:T] for x in self.h_all])
        if ctp:   # phase 2 (every rank pushed its aux rows): wait, the own 1,280 columns of every rank's rows, push them back
            with torch.cuda.stream(sctx):
                d.ctp_mid(self, r, NT)
            yield ("nop",)   # every rank's phase 2 issued before any rank issues phase 3 (its wait)
            with torch.cuda.stream(sctx):   # phase 3: assemble main_x rows, the 3 blocks' context KV
                d.ctp_tail(self, r, NT)
                evc2 = torch.cuda.Event()
                evc2.record(sctx)
        elif evc2 is None and self.sw["ctx"] == "ag":
            evc2 = fork_ctx()
        lg = self.vhead[r](self.h_all[r][:T])
        self._mark(r, "verify_head")
        if evc2 is None and not ctp:
            evc2 = fork_ctx()
        m = self._meta(r)
        if self.sw["par"]:
            self.par[r].stats(st, lg, m)
        else:
            S.verify3_stats(st, lg, m)
        self._mark(r, "stats")
        yield ("ag", "g1", st.stats[:T], [x.g1(T) for x in sts])
        if self.sw["par"]:
            self.par[r].mass(st, lg, st.g1(T), m)
        else:
            S.verify3_mass(st, lg, st.g1(T), m)
        yield ("ag", "g2", st.resid[:T], [x.g2(T) for x in sts])
        S.verify2_decide(st, st.g1(T), st.g2(T), m, T)
        self._mark(r, "decide")
        S.verify2_resample(st, lg, st.g1(T), m)
        yield ("ag", "g3", st.out3[:n], [x.g3(n) for x in sts])
        S.verify2_finish(st, st.g3(n), m)
        self._mark(r, "finish")
        # ---- drafter metadata from the result (GPU only)
        g = self._glue_bufs(r, n)
        HS._post_glue[(n,)](st.num_sampled, st.last, tb["g_dslot"], tb["g_p0"], n, g["dmi"], g["dmp"], g["anc"], r * Dp, Dp,
                            tb["wbase"], tb["wpages"], tb["rstart"], tb["lslot"], d.b["q_pos"], d.b["q_slot"], d.b["idx"],
                            d.b["idx"].stride(0), g["gns"], K=k, KP=triton.next_power_of_2(k), NBW_=HS.NBW, WIN=DR.WINDOW,
                            WBLK=triton.next_power_of_2(DR.WINDOW + k), POOL=self.pool, RING_=DR.RING, MAXS=self.max_seqs,
                            num_warps=4)
        HS._x0_glue[(Dp, k, triton.cdiv(HS.H, 1024))](d.b["X0"], self.emb[r], g["anc"], r * Dp, DR.NOISE_ID, K=k, HC_=DR.HC,
                                                      D_=HS.H, BLOCK=1024)
        anchors_all = g["anc"][:n]
        dm = S.DraftMeta(idx=g["dmi"][:n], sample_pos=g["dmp"][:n], num_reqs=n)
        ev = torch.cuda.Event()
        ev.record(cur)
        side = self.side[r] if HS.NOISE_SIDE else cur
        side.wait_event(ev)
        with torch.cuda.stream(side):
            S.draft_noise(st, self.noise[r][:, :n], dm)
            torch.index_select(st.w1, 0, anchors_all, out=st.emb[0, :n])
            torch.mm(st.emb[0, :n], st.w2.t(), out=st.bias[:n])
            ev2 = torch.cuda.Event()
            ev2.record(side)
        self._mark(r, "noise+bias0")
        # ---- drafter blocks
        own_ = [x for x in range(HS.W) if len(c.dseqs[x])] if self.sw.get("c1d") else []
        if self.sw.get("c1d") and Dp == 1 and len(own_) == 1 and d.Rd == k:   # c1d: one c1mega run of the 3 blocks
            owner = own_[0]
            dranks = [self.draft_rank(x, Dp) for x in range(HS.W)]
            cur.wait_event(evc2)   # the context KV of this step in the owner's rings (the kernel reads / inserts after it)
            self.c1d.launch(r, owner, dranks)
            yield ("nop",)         # every rank's launch issued before any rank continues (the run's lamport exchanges)
            y_, xm_, fpre_, fpost_, fcomb_ = self.c1d.outputs(r, d.Rd)
            q0 = owner * Dp * k
            _MHCF.frow(y_, xm_, fpost_, fcomb_, fpre_, d.final_norm, d.eps, self.c1d.B[r]["xn"][:d.Rd],
                       self.hd_all[r][q0:q0 + k], d.b["ptok0"])
            self._mark(r, "c1d")
            hd_done = True
        else:
            hd_done = False
        d._dmask = (tb["g_dslot"], r * Dp) if self.sw["dmask"] else None
        use_rt = self.sw["rtag"] and d.Rd <= 4
        d._rtag = (self, r, tb["g_dslot"] if self.sw["dmask"] else self._no_dslot[r]) if use_rt else None
        if not hd_done:
            cur.wait_event(evc2)
        for name, kind, fn in (() if hd_done else DR.stages([d], None)):
            if name == "context_kv":
                continue
            if kind == "local":
                fn(d)
                self._mark(r, name)
            elif kind == "ag" and use_rt:   # rtag pushed f + routing rows: only the wait (W x (NC + 1) signals per round),
                yield ("nop",)               # after every rank's rtag launch (run_all: no rank waits before all pushed)
                self.dcomm[Dp].e.wait(self.rt_fl[r], self.rt_ep[r], HS.W * (RTAG_NC + 1))   # every source rank: NC + 1
            elif kind == "ag":
                yield ("dag", Dp)
            elif kind in ("tps", "tpr"):
                yield (kind, Dp)
            else:
                yield ("drs", Dp)
        if not hd_done:
            yield ("ag", "hd", d.b["hd"], [x[: n * k] for x in self.hd_all])
        dh = self.dhead[r]
        if self.sw["dtail"] and n * k <= 12 and n <= 4 and k <= 4 and dh.mode == "e4m3":
            cur.wait_event(ev2)   # slot 0's bias and the draft noise (side stream)
            x_ = self.dt[r]
            _DTAIL.dtail(self.hd_all[r][: n * k], dh.wq, dh.ws.view(-1), st.w2, st.w1, st.bias[:n], self.noise[r][:, :n],
                         dm.idx, st.temp, st.cache, st.draft_tokens, st.token_logit, st.lse, st.max_reqs, st.start,
                         dh.out[: n * k], x_["part"], x_["bar"], x_["xloc"], self.dt_peers, r, n, k, x_["dbg"], True,
                         self._no_dslot[r], self._no_f32[r])
            self._mark(r, "dtail")
            yield ("nop",)   # every rank's dtail launched before any rank issues more (its in-kernel exchange)
        else:
            if self.sw["dhead"] and n * k <= 12 and dh.mode == "e4m3":   # head only (same math), Markov below as before
                x_ = self.dt[r]
                xq_, xs_ = HD.quant_rows(self.hd_all[r][: n * k], dh.xq[: n * k], dh.xs[: n * k])   # = DraftHead's rowq
                _DTAIL.dtail(self.hd_all[r][: n * k], dh.wq, dh.ws.view(-1), st.w2, st.w1, st.bias[:n], self.noise[r][:, :n],
                             dm.idx, st.temp, st.cache, st.draft_tokens, st.token_logit, st.lse, st.max_reqs, st.start,
                             dh.out[: n * k], x_["part"], x_["bar"], x_["xloc"], self.dt_peers, r, n, k, x_["dbg"], False,
                             xq_.view(torch.uint8), xs_)
                b3 = dh.out[: n * k].view(n, k, self.loc)
            else:
                b3 = dh(self.hd_all[r][: n * k]).view(n, k, self.loc)
            self._mark(r, "draft_head")
            cur.wait_event(ev2)
            for i in range(k):
                S.draft3_local(st, b3[:, i], st.emb[i % 2, :n], dm.idx[:, i], self.noise[r][i, :n], i, 2048,
                               bias_ready=i == 0)
                yield ("ag", f"d{i}", st.d_out[:n], [x.d_gath(n, i) for x in sts])
                S.draft_combine(st, st.d_gath(n, i), n, i, st.emb[(i + 1) % 2, :n], dm.idx[:, i])
        self.num_emitted[r].zero_()
        HS._out_glue[(n,)](self.by_slot[r], self.by_slot[r].stride(0), st.last, st.draft_tokens, st.draft_tokens.stride(0),
                           tb["g_dslot"], g["gns"], n, r * Dp, Dp, tb["seq_of_d"], self.next_ids[r],
                           self.next_ids[r].stride(0), self.num_emitted[r], MAXS1=self.max_seqs + 1, K=k,
                           KP=triton.next_power_of_2(k + 1), num_warps=1)
