"""DSpark drafter step for the ds41h-phys sequence-parallel step program (kernels/head).

Production's drafter (vLLM DSparkSpeculator, models/deepseek_v41/nvidia/dspark.py; reference ds41h/ref/dspark.py):
  main_x  = main_norm(main_proj(cat(aux_37, aux_38, aux_39)))   aux_l = bf16(mean over the 4 hc copies of block l's input)
  context KV of each draft block = kv_store(RoPE(kv_norm(wkv_i(main_x)))) at the ACCEPTED target positions (ring, window 128)
  block = N = k query tokens per request [new token, noise x (k-1)] (noise 128799) at positions Q .. Q+k-1
          (Q = last accepted position + 1), embedded and repeated to 4 hc copies
  3 DSpark blocks (mtp.0..2): mHC boundary -> attention (ratio 0, theta 10000; every query attends to the 128-token
          context window before Q plus all k block tokens, sink) -> mHC boundary -> MoE (128 experts top-3 + shared)
  head input = norm(hc_collapse(X_out, last ffn_pre)); base logits = head(.) (the target's head, vocab-sharded here);
  sequential Markov sampling (kernels/head/sampler.py)

Layout here (rank r owns whole sequences, like the target layers):
  * main_proj, context KV, mHC, attention (all 64 heads, replicated attention weights), router: on the OWN rows;
  * the ring KV of a sequence lives on its owner only (no KV crosses NVLink);
  * MoE tensor-parallel over ALL rows (rank r: 1/4 of every expert's intermediate dim + 1/4 of the shared expert):
    AG of (f, routing) -> fork MoE kernel (moe_raw.cu, top-3 padded to 6 with id -1) -> RS of the partials.
Kernels reused (snapshot in kernels/head/vendor, built as head_*): mhc.cu boundary, dense.cu (RMSNorm/RoPE, fp8_ds_mla
record insert, inverse RoPE), moe_raw.cu, comm.cu; FlashMLA sparse decode (stock); dense GEMMs cuBLAS bf16 with exactly
dequantized weights (the fork's attention-half baseline).

Exactness: the drafter is allowed to approximate (verification is exact); this one follows the target's numerics class
(exact weights, bf16 activations, fp32 accumulation) and is checked against ds41h.ref.dspark (VLLM_SM90 numerics).
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vendor_ext as VE  # noqa: E402

D, HC, NH, HD = 5120, 4, 64, 512
NOISE_ID = 128799
WINDOW = 128
PAGE_SHIFT = 5
PAGE = 1 << PAGE_SHIFT
REC = 584
RING = 160  # ring slots per sequence (5 pages): window 128 + block + slack, multiple of PAGE
IDX_W = 192  # FlashMLA index width (window 128 + block k, padded to a multiple of 64)
NE, TOPK, TOPK_PAD, I_RANK = 128, 3, 6, 576
W = 4


from vllm.triton_utils import tl, tldevice, triton  # noqa: E402


@triton.jit
def _router_kernel(logits_ptr, bias_ptr, ids_ptr, w_ptr, NE_: tl.constexpr, K_: tl.constexpr, ROW_WORDS: tl.constexpr,
                   SCALE: tl.constexpr):
    # ref gate_ref: scores = sqrt(softplus(logits)) (torch softplus: threshold 20), top-k of scores + bias (selection only),
    # weights = scores[sel] / (sum + 1e-20) * route_scale; written as the 64-B routing row (6 int32 ids, -1 padded;
    # 6 fp32 weights, 0 padded) that comm.cu's AG and moe_raw's route read.
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
    tl.store(ids_ptr + r * ROW_WORDS + lanes, ids, mask=lanes < 6)
    tl.store(w_ptr + r * ROW_WORDS + 6 + lanes, ws, mask=lanes < 6)


def _ref():
    from ds41h.ref import layer as L  # noqa: F401
    return L


def stock_cos_sin(dev, max_pos):
    """Stock ratio-0 RoPE table (theta 10000, no YaRN), fp32 [max_pos, 64] = cos | sin (attn_dense bench2b formula)."""
    dim = 64
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float, device=dev) / dim))
    t = torch.arange(max_pos, dtype=torch.float, device=dev)
    fr = torch.outer(t, inv)
    return torch.cat([fr.cos(), fr.sin()], -1).contiguous()


def load_raw_experts_mtp(rd, i, rank, device, tp=4, n_exp=NE):
    """Raw TP slice of mtp.i's routed experts in the ds41h layout (= sp_layer load_raw_experts with the mtp prefix)."""
    lay = rd.mtp(i)
    u8 = torch.uint8
    out = [[], [], [], []]
    for e in range(n_exp):
        p = f"ffn.experts.{e}."
        out[0].append(torch.cat([lay.tp(p + "w1.weight", rank, tp).view(u8), lay.tp(p + "w3.weight", rank, tp).view(u8)]))
        out[1].append(torch.cat([lay.tp(p + "w1.scale", rank, tp).view(u8), lay.tp(p + "w3.scale", rank, tp).view(u8)]))
        out[2].append(lay.tp(p + "w2.weight", rank, tp).view(u8))
        out[3].append(lay.tp(p + "w2.scale", rank, tp).view(u8))
    return tuple(torch.stack(t).to(device).contiguous() for t in out)


def _wcache():
    """The converted-weight cache (tools/wcache.py), or None outside the fork tree."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))
        from wcache import cached
        return cached
    except ImportError:
        return None


WCACHE_V = "v1"  # bump when a builder's output changes
# RS fold: the MoE reduce-scatter's wait + fixed-order sum inside the next mHC boundary (mhc.cu boundary_rs, comm.cu
# k_rs_sum's bits) instead of a separate k_rs_sum; own draft rows <= 128 only (boundary_rs takes [4, R <= 128, D]).
FUSE_RS = os.environ.get("HSD_FUSE_RS", "0") not in ("0", "")
# mHC split boundary (mhc.cu k1a / k1b_out / k1b_mix, bit for bit the fused one): the mixes for the NEXT boundary on a
# side stream, overlapping the attention / router + MoE; the final boundary's mixes (never read) skipped; <= 128 rows.
MHC_SPLIT = os.environ.get("HSD_MHC_SPLIT", "0") not in ("0", "")
# mhc2: the live sp_layer mhc.cu (any rows per call): one boundary call for > 128 own draft rows (instead of 128-row
# chunks), split / RS fold at any row count; same bits (A/B: bench_hsd1.py --ab).
MHC2 = os.environ.get("HSD_MHC2", "0") not in ("0", "")
_V4 = None


def v4_ext():
    """The kernels/moe_v4 drafter drop-in (v4.py, imported read-only, built into kernels/head/build_v4)."""
    global _V4
    if _V4 is None:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE)))
        from moe_v4 import v4
        _V4 = v4.load(build_root=os.path.join(HERE, "build_v4"))
    return _V4


def block_attn_cpu(rd, cfg, i):
    """Rank-independent part of DSpark block mtp.i (replicated on every rank), built on the CPU: bf16 attention weights
    exactly dequantized from e4m3 + ue8m0, norms, sink, raw mHC fn (prep_fn runs on the device), gate."""
    L = _ref()
    Wf = L.load_layer_weights(rd, cfg.n_layers + i, rank=None, device="cpu", cfg=cfg, mtp=i, experts="none")
    bf = torch.bfloat16
    return dict(
        wqkv_a=torch.cat([Wf.wq_a, Wf.wkv], 0).to(bf).contiguous(), q_norm=Wf.q_norm.float().contiguous(),
        kv_norm=Wf.kv_norm.float().contiguous(), wq_b=Wf.wq_b.to(bf).contiguous(),
        wo_aT=Wf.wo_a.to(bf).transpose(1, 2).contiguous(), wo_b=Wf.wo_b.to(bf).contiguous(),
        sink=Wf.attn_sink.float().contiguous(),
        hc_attn_fn=Wf.hc_attn_fn.float().contiguous(), sa=Wf.hc_attn_scale.float(), ba=Wf.hc_attn_base.float(),
        na=Wf.attn_norm.float(), hc_ffn_fn=Wf.hc_ffn_fn.float().contiguous(), sf=Wf.hc_ffn_scale.float(),
        bff=Wf.hc_ffn_base.float(), nf=Wf.ffn_norm.float(),
        gate_w=Wf.gate_w.float().contiguous(), gate_wb=Wf.gate_w.to(bf).contiguous(), gate_b=Wf.gate_bias.float().contiguous(),
        sh_w1=Wf.sh_w1.to(bf).contiguous(), sh_w3=Wf.sh_w3.to(bf).contiguous(), sh_w2=Wf.sh_w2.to(bf).contiguous(),
    )


def block_ffn_cpu(rd, i, rank):
    """Rank-r part of block mtp.i: the raw TP slice of its 128 routed experts (ds41h layout, byte copies)."""
    w13, s13, w2, s2 = load_raw_experts_mtp(rd, i, rank, "cpu")
    return dict(w13=w13, s13=s13, w2=w2, s2=s2)


def main_proj_cpu(rd):
    from ds41h.ref.weights import dequant_fp8
    mp = dequant_fp8(rd.get("mtp.0.main_proj.weight"), rd.get("mtp.0.main_proj.scale"))
    return dict(main_proj=mp.to(torch.bfloat16).contiguous(), main_norm=rd.get("mtp.0.main_norm.weight").float())


def _get(key, build, dev):
    c = _wcache()
    if c is None:
        return {k: v.to(dev) for k, v in build().items()}
    return c(key, build, device=dev, verbose=False)


def load_block(rd, cfg, i, rank, dev, mhc):
    A = _get(f"dspark_b{i}_attn_bf16_{WCACHE_V}", lambda: block_attn_cpu(rd, cfg, i), dev)
    F = _get(f"dspark_b{i}_r{rank}_tp4_raw_experts_{WCACHE_V}", lambda: block_ffn_cpu(rd, i, rank), dev)
    sl = slice(rank * I_RANK, (rank + 1) * I_RANK)
    B = {k: v for k, v in A.items() if k not in ("hc_attn_fn", "hc_ffn_fn", "sh_w1", "sh_w3", "sh_w2")}
    B["fa"], B["ff"] = mhc.prep_fn(A["hc_attn_fn"]), mhc.prep_fn(A["hc_ffn_fn"])
    B["sw13"] = torch.cat([A["sh_w1"][sl], A["sh_w3"][sl]], 0).contiguous()
    B["sw2"] = A["sh_w2"][:, sl].contiguous()
    B.update(F)
    return B


def prebuild(rd, cfg, ranks=range(4)):
    """Build every cache entry on the CPU (run with CUDA_VISIBLE_DEVICES= )."""
    c = _wcache()
    assert c is not None, "tools/wcache.py not found"
    c(f"dspark_main_proj_bf16_{WCACHE_V}", lambda: main_proj_cpu(rd))
    for i in range(cfg.n_mtp_layers):
        c(f"dspark_b{i}_attn_bf16_{WCACHE_V}", lambda i=i: block_attn_cpu(rd, cfg, i))
        for r in ranks:
            c(f"dspark_b{i}_r{r}_tp4_raw_experts_{WCACHE_V}", lambda i=i, r=r: block_ffn_cpu(rd, i, r))


class DraftRank:
    """Weights and static buffers of one rank's drafter (own sequences n_own, k query tokens each)."""

    def __init__(self, rank, dev, rd, cfg, n_own, k=3, max_pos=262144, ctx_rows=None, moe_mode="route2", shared=None,
                 pool=None):
        """moe_mode: 'base' (moe_raw route + gemm1r / gemm2r), 'route2' (route2.cu tables), 'n8' (route2x + the
        n_pad-8 row-stream kernels for chunks of <= 8 tokens: the drafter has ~8 tokens per expert).
        shared: another DraftRank of the same rank / device whose weights this one reuses (one per graph bucket).
        pool: dict(base=[3 flat uint8 tensors], view=[3 FlashMLA k_cache views], page_stride) = the window pool's
        layers 40..42 (serve/worker.py Pools.window_base / window_view, WINDOW_PAGE_BYTES); None: a private ring."""
        assert moe_mode in ("base", "route2", "n8", "v4")
        self.fold, self.fuse_rs = None, FUSE_RS
        self.mhc_split, self.mix_ev, self.mix_stream = MHC_SPLIT, None, torch.cuda.Stream(device=dev)
        self.mhc2, self._mhc_new = MHC2, None
        self.moe_mode = moe_mode
        self.rank, self.dev, self.n, self.k = rank, torch.device(dev), n_own, k
        self.Rd = n_own * k
        self.R = n_own * (k + 1)  # verified rows of the step (context candidates)
        self.Tall = W * self.Rd
        dv = self.dev
        bf = torch.bfloat16
        with torch.cuda.device(dv):
            self.mhc, self.kd, self.me = VE.mhc(), VE.dense(), VE.moe()
            if shared is not None:
                for nm in ("main_proj", "main_norm", "final_norm", "blocks", "cos_sin"):
                    setattr(self, nm, getattr(shared, nm))
            else:
                MP = _get(f"dspark_main_proj_bf16_{WCACHE_V}", lambda: main_proj_cpu(rd), dv)
                self.main_proj, self.main_norm = MP["main_proj"], MP["main_norm"]
                self.final_norm = rd.get(f"mtp.{cfg.n_mtp_layers - 1}.norm.weight").to(dv).float()
                self.blocks = [load_block(rd, cfg, i, rank, dv, self.mhc) for i in range(cfg.n_mtp_layers)]
                self.cos_sin = stock_cos_sin(dv, max_pos)
            self.eps, self.hc_eps, self.iters = cfg.norm_eps, cfg.hc_eps, cfg.hc_iters
            if pool is not None:  # the window pool's drafter layers (slots = window page * 32 + pos % 32)
                self.page_stride, self.cache, self.k_cache = pool["page_stride"], list(pool["base"]), list(pool["view"])
            else:  # private ring: per block [n_own * RING / PAGE pages] x (PAGE x 584 B), stock fp8_ds_mla page layout
                self.page_stride = PAGE * REC
                npg = n_own * RING // PAGE
                self.cache = [torch.zeros(npg * self.page_stride, dtype=torch.uint8, device=dv) for _ in self.blocks]
                self.k_cache = [torch.as_strided(c, (npg, PAGE, 1, REC), (self.page_stride, REC, REC, 1)) for c in self.cache]
            # step buffers
            Rd, R, Tall = self.Rd, self.R, self.Tall
            e = lambda *s, dt=torch.float32: torch.empty(*s, dtype=dt, device=dv)  # noqa: E731
            cr = max(R, ctx_rows or 0)
            self.b = dict(
                aux=e(cr, 3 * D, dt=bf), mx_raw=e(cr, D, dt=bf), main_x=e(cr, D, dt=bf), kvc=e(cr, 512, dt=bf),
                kv_ctx=e(cr, 512, dt=bf), dq=torch.zeros(cr, 16, HD, dtype=bf, device=dv),
                ctx_pos=torch.zeros(cr, dtype=torch.int64, device=dv), ctx_slot=torch.full((cr,), -1, dtype=torch.int64, device=dv),
                X0=e(Rd, HC, D, dt=bf), X=e(Rd, HC, D, dt=bf), Xm=e(Rd, HC, D, dt=bf), zero_y=torch.zeros(Rd, D, dtype=bf, device=dv),
                zero_post=torch.zeros(Rd, HC, device=dv), eye=torch.eye(HC, device=dv).expand(Rd, HC, HC).contiguous(),
                onehot=torch.nn.functional.one_hot(torch.zeros(Rd, dtype=torch.long, device=dv), HC).float().contiguous(),
                a_pre=e(Rd, HC), a_post=e(Rd, HC), a_comb=e(Rd, HC, HC), f_pre=e(Rd, HC), f_post=e(Rd, HC),
                f_comb=e(Rd, HC, HC), o_pre=e(Rd, HC), o_post=e(Rd, HC), o_comb=e(Rd, HC, HC),
                a=e(Rd, D, dt=bf), f=e(Rd, D, dt=bf), hd=e(Rd, D, dt=bf), part_mhc=e(160 * Rd * 25),
                qakv=e(Rd, 1792, dt=bf), qr=e(Rd, 1280, dt=bf), kv=e(Rd, 512, dt=bf), q=e(Rd, 1, NH, HD, dt=bf),
                o=e(Rd, 1, NH, HD, dt=bf), z=e(Rd, 8192, dt=bf), y=e(Rd, D, dt=bf),
                q_pos=torch.zeros(Rd, dtype=torch.int64, device=dv), q_slot=torch.zeros(Rd, dtype=torch.int64, device=dv),
                idx=torch.full((Rd, 1, IDX_W), -1, dtype=torch.int32, device=dv),
                lens=torch.full((Rd,), WINDOW + k, dtype=torch.int32, device=dv),
                route_row=torch.zeros(Rd, 64, dtype=torch.uint8, device=dv),
                gf=torch.zeros(Tall, D * 2, dtype=torch.uint8, device=dv), gr=torch.zeros(Tall, 64, dtype=torch.uint8, device=dv),
                rs=torch.zeros(W, Rd, D, dtype=bf, device=dv), moe_out=e(Rd, D, dt=bf),
                ids_in=torch.zeros(Rd, dtype=torch.int64, device=dv), gate_lg=e(Rd, NE),
            )
            self.r2 = VE.route2()  # always: moe_mode can be switched on a built rank (one process for all MoE variants)
            max_ec = max(int(self.me.route_max_ec()), int(self.r2.route2_max_ec(Tall)))
            P = Tall * TOPK_PAD
            self.m = dict(
                max_ec=max_ec, sms=torch.cuda.get_device_properties(dv).multi_processor_count,
                ecs=torch.zeros(max_ec * (32 + 12 * 64), dtype=torch.uint8, device=dv),
                t1=torch.zeros(max_ec * 9 * 24, dtype=torch.uint8, device=dv),
                t2=torch.zeros(max_ec * 40 * 24, dtype=torch.uint8, device=dv), counts=torch.zeros(8, dtype=torch.int32, device=dv),
                slotmap=torch.zeros(P, dtype=torch.int32, device=dv),
                rts=torch.zeros(max(int(self.me.route_scratch_ints()), int(self.r2.route2_scratch_ints())),
                                dtype=torch.int32, device=dv),
                ec8=torch.zeros(max_ec, dtype=torch.int32, device=dv),
                img=torch.zeros((P + 7 * max_ec) * 80 * 128, dtype=torch.uint8, device=dv),
                himg=torch.zeros((P + 7 * max_ec) * 9 * 128, dtype=torch.uint8, device=dv),
                ptok=torch.zeros(Tall, dtype=torch.int32, device=dv), inv=torch.zeros(Tall, device=dv),
                Y=torch.zeros(P, D, dtype=bf, device=dv), part=e(Tall, D, dt=bf), gu=e(Tall, 2 * I_RANK, dt=bf),
                hsh=e(Tall, I_RANK, dt=bf), sh=e(Tall, D, dt=bf), empty_f=torch.zeros(0, device=dv),
            )
            self.b["ptok0"] = torch.zeros(0, dtype=torch.int32, device=dv)
            self.sh_stream = None  # v4: the shared expert on a side stream
            self.sched = None

    # ------------------------------------------------------------------ metadata (graph-capturable torch ops)
    def slot(self, seq, pos):
        return seq.to(torch.int64) * RING + pos % RING

    def set_ctx(self, P, ns=None):
        """Context rows of the step: verified rows j = 0..k of own request i at P[i] + j. ns None (default): insert ALL
        rows -- needs nothing from the verification, so context_kv() can run before / concurrently with it: a rejected row
        j >= ns sits at P + j <= Q + k - 1 (Q = P + ns), a block position that every block's attention overwrites with
        the block token's KV before it reads the ring, and the next step re-inserts the accepted positions (same
        attention inputs as inserting only the accepted rows). ns given: only the accepted rows (stock's choice)."""
        b, n, k = self.b, self.n, self.k
        dv = self.dev
        seq = torch.arange(n, device=dv)
        j = torch.arange(k + 1, device=dv)
        cpos = (P[:, None] + j[None, :])
        b["ctx_pos"][: self.R].copy_(cpos.reshape(-1))
        slot = self.slot(seq[:, None], cpos)
        if ns is not None:
            slot = torch.where(j[None, :] < ns[:, None], slot, -1)
        b["ctx_slot"][: self.R].copy_(slot.reshape(-1))

    def set_block(self, P, ns, embed, anchors, noise_emb):
        """Block metadata from the verification result: Q = P + ns, block positions Q .. Q+k-1, ring window indices,
        and the block input [anchor, noise x (k-1)] embedded into 4 hc copies (GPU ops only, graph-capturable)."""
        b, n, k = self.b, self.n, self.k
        dv = self.dev
        seq = torch.arange(n, device=dv)
        Q = P + ns
        t = torch.arange(k, device=dv)
        qpos = Q[:, None] + t[None, :]
        b["q_pos"].copy_(qpos.reshape(-1))
        b["q_slot"].copy_(self.slot(seq[:, None], qpos).reshape(-1))
        w = torch.arange(WINDOW, device=dv)
        wpos = Q[:, None] - WINDOW + w[None, :]
        # contexts shorter than the window: index -1 (FlashMLA's invalid entry), length kept at 128 + k so the scheduler
        # plan stays valid (untested: real traffic contexts are 12K-130K tokens)
        win = torch.where(wpos >= 0, self.slot(seq[:, None], wpos), -1)  # [n, 128]
        blk = self.slot(seq[:, None], qpos)  # [n, k]
        ind = torch.cat([win, blk], 1).to(torch.int32)  # [n, 128 + k]
        b["idx"][:, 0, : WINDOW + k].copy_(ind.repeat_interleave(k, 0))
        e0 = torch.index_select(embed, 0, anchors)  # [n, D]
        X0 = b["X0"].view(n, k, HC, D)
        X0[:, 0].copy_(e0[:, None, :].expand(n, HC, D))
        X0[:, 1:].copy_(noise_emb.view(1, 1, 1, D).expand(n, k - 1, HC, D))

    def set_step(self, P, ns, embed, anchors, noise_emb, ctx_all: bool = True):
        self.set_ctx(P, None if ctx_all else ns)
        self.set_block(P, ns, embed, anchors, noise_emb)

    # ------------------------------------------------------------------ context KV (main_proj, ring insert)
    def context_kv(self, rows=None):
        b, kd = self.b, self.kd
        R = rows or self.R
        torch.mm(b["aux"][:R], self.main_proj.t(), out=b["mx_raw"][:R])
        kd.norm_rope(b["mx_raw"][:R], self.main_norm, b["main_x"][:R], self.eps, None, None)
        for i, B in enumerate(self.blocks):
            torch.mm(b["main_x"][:R], B["wqkv_a"][1280:].t(), out=b["kvc"][:R])
            kd.norm_rope(b["kvc"][:R], B["kv_norm"], b["kv_ctx"][:R], self.eps, b["ctx_pos"][:R], self.cos_sin)
            kd.rope_q_insert(b["dq"][:R], b["kv_ctx"][:R], b["ctx_pos"][:R], self.cos_sin, b["ctx_slot"][:R], self.cache[i],
                             self.page_stride, PAGE_SHIFT, True)

    # ------------------------------------------------------------------ block pieces (own rows)
    def _bx(self):
        """The mHC extension in use: vendor mhc.cu (<= 128 rows per call) or mhc2 (any rows; self.mhc2)."""
        if not self.mhc2:
            return self.mhc
        if self._mhc_new is None:
            self._mhc_new = VE.mhc2()
        return self._mhc_new

    def _join_mix(self):
        if self.mix_ev is not None:  # the previous boundary's mixes (side stream) before anything reads them
            torch.cuda.current_stream().wait_event(self.mix_ev)
            self.mix_ev = None

    def _split_bd(self, k1a_call, n, lag, sc, bs, nw, xn, pre_o, post_o, comb_o, out, mix=True):
        """mhc.cu's split boundary (the same outputs as boundary / boundary_rs bit for bit): K1a (+ the RS fold), then
        K1b-out (collapse + RMSNorm = the sublayer input) on the critical path and K1b-mix (partials -> pre / post / comb,
        read only by the NEXT boundary) on a side stream joined there. mix=False: the mixes are never read (final
        boundary) and are not computed."""
        self._join_mix()
        k1a_call()
        if mix:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
        self._bx().k1b_out(xn, lag, nw, self.eps, out, self.b["ptok0"])
        if mix:
            with torch.cuda.stream(self.mix_stream):
                self.mix_stream.wait_event(ev)
                self._bx().k1b_mix(self.b["part_mhc"], n, sc, bs, self.eps, self.hc_eps, self.iters, pre_o, post_o, comb_o)
                ev2 = torch.cuda.Event()
                ev2.record(self.mix_stream)
            self.mix_ev = ev2

    def boundary(self, y, rs, post, comb, lag, fn, sc, bs, nw, xn, pre_o, post_o, comb_o, out, mix=True):
        # mhc.cu takes <= 128 rows per call (rows are independent): chunk larger draft-row counts
        n = y.shape[0]
        bx = self._bx()
        if self.mhc_split and (n <= 128 or self.mhc2):
            return self._split_bd(lambda: bx.k1a(y, rs, post, comb, fn, xn, self.b["part_mhc"]), n, lag, sc, bs, nw,
                                  xn, pre_o, post_o, comb_o, out, mix)
        self._join_mix()
        if self.mhc2:  # one call for any row count
            bx.boundary(y, rs, post, comb, lag, fn, sc, bs, nw, self.eps, self.hc_eps, self.iters, xn, pre_o, post_o,
                        comb_o, out, self.b["part_mhc"])
            return
        for a in range(0, n, 128):
            e = min(n, a + 128)
            self.mhc.boundary(y[a:e], rs[a:e], post[a:e], comb[a:e], lag[a:e], fn, sc, bs, nw, self.eps, self.hc_eps,
                              self.iters, xn[a:e], pre_o[a:e], post_o[a:e], comb_o[a:e], out[a:e], self.b["part_mhc"])

    def boundary_moe(self, post, comb, lag, fn, sc, bs, nw, xn, pre_o, post_o, comb_o, out, mix=True):
        """The boundary after a MoE block: y = the RS sum of the 4 ranks' partial MoE outputs. With a pending fold (set by
        PeerComm.rs_rank) the boundary waits for the RS round and sums the receive slots itself (boundary_rs)."""
        b = self.b
        if self.fold is None:
            return self.boundary(b["moe_out"], b["Xm"], post, comb, lag, fn, sc, bs, nw, xn, pre_o, post_o, comb_o, out,
                                 mix)
        flag, epoch, sig = self.fold
        self.fold = None
        bx = self._bx()
        if self.mhc_split:
            return self._split_bd(lambda: bx.k1a_rs(b["rs"], flag, epoch, sig, b["Xm"], post, comb, fn, xn,
                                                    b["part_mhc"]), self.Rd, lag, sc, bs, nw, xn, pre_o, post_o,
                                  comb_o, out, mix)
        self._join_mix()
        bx.boundary_rs(b["rs"], flag, epoch, sig, b["Xm"], post, comb, lag, fn, sc, bs, nw, self.eps, self.hc_eps,
                             self.iters, xn, pre_o, post_o, comb_o, out, b["part_mhc"], b["ptok0"])

    def attn_bd(self, i):
        b, B = self.b, self.blocks[i]
        if i == 0:  # stream = the embedding copies (post 0, comb I); lagged pre = one-hot copy 0 (no previous block)
            self.boundary(b["zero_y"], b["X0"], b["zero_post"], b["eye"], b["onehot"], B["fa"], B["sa"], B["ba"], B["na"],
                          b["X"], b["a_pre"], b["a_post"], b["a_comb"], b["a"])
        else:
            self.boundary_moe(b["f_post"], b["f_comb"], b["f_pre"], B["fa"], B["sa"], B["ba"], B["na"], b["X"], b["a_pre"],
                              b["a_post"], b["a_comb"], b["a"])

    def plan(self):
        from vllm.third_party.flashmla.flash_mla_interface import get_mla_metadata
        self.sched = get_mla_metadata()[0]
        self.attention(0, insert=False)  # first call plans (outside any graph capture)

    dense_fp8 = False  # set by enable_dense_fp8()

    def enable_dense_fp8(self, on=True):
        """Drafter dense projections q_a / kv, q_b and wo_b as e4m3 W8A8 GEMMs (torch._scaled_mm, per-row scales on the
        weights and on the activations; wo_a stays a bf16 bmm). LOSSY, the drafter's choice (production's drafter class:
        stock vLLM's FP8 linears quantize the activations too); verification stays exact. The e4m3 copies are made once
        per block (+127 MB per block and device) and shared by DraftRanks with shared=."""
        import head as HD_
        self._q8 = HD_.quant_rows
        for B in self.blocks:
            if "wq_b8" not in B:
                for nm in ("wqkv_a", "wq_b", "wo_b"):
                    w = B[nm].float()
                    s_ = w.abs().amax(-1).clamp_min(1e-12) / 448.0
                    B[nm + "8"] = (w / s_[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
                    B[nm + "8s"] = s_[None, :].contiguous()
                    del w
        if "xq" not in self.b:
            dv = self.dev
            self.b["xq"] = torch.empty(self.Rd * 8192, dtype=torch.float8_e4m3fn, device=dv)  # flat: [Rd, K] views
            self.b["xs"] = torch.empty(self.Rd, dtype=torch.float32, device=dv)
        self.dense_fp8 = on

    def _mm8(self, x, B, nm, out):
        K = x.shape[1]
        q, s_ = self._q8(x, self.b["xq"][: x.shape[0] * K].view(x.shape[0], K), self.b["xs"][: x.shape[0]])
        torch._scaled_mm(q, B[nm + "8"].t(), scale_a=s_[:, None], scale_b=B[nm + "8s"], out_dtype=torch.bfloat16, out=out)

    def attention(self, i, insert=True):
        from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache
        b, B, kd = self.b, self.blocks[i], self.kd
        Rd = self.Rd
        if self.dense_fp8:
            self._mm8(b["a"], B, "wqkv_a", b["qakv"])
        else:
            torch.mm(b["a"], B["wqkv_a"].t(), out=b["qakv"])
        kd.norm_rope2(b["qakv"][:, :1280], B["q_norm"], b["qr"], b["qakv"][:, 1280:], B["kv_norm"], b["kv"], self.eps,
                      b["q_pos"], self.cos_sin)
        if self.dense_fp8:
            self._mm8(b["qr"], B, "wq_b", b["q"].view(Rd, NH * HD))
        else:
            torch.mm(b["qr"], B["wq_b"].t(), out=b["q"].view(Rd, NH * HD))
        q3 = b["q"].view(Rd, NH, HD)
        kd.rope_q_insert(q3, b["kv"], b["q_pos"], self.cos_sin, b["q_slot"], self.cache[i], self.page_stride, PAGE_SHIFT,
                         bool(insert))
        flash_mla_with_kvcache(q=b["q"], k_cache=self.k_cache[i], block_table=None, cache_seqlens=None, head_dim_v=HD,
                               tile_scheduler_metadata=self.sched, is_fp8_kvcache=True, indices=b["idx"],
                               topk_length=b["lens"], softmax_scale=HD ** -0.5, attn_sink=B["sink"], out=b["o"])
        o3 = b["o"].view(Rd, NH, HD)
        kd.inv_rope(o3, b["q_pos"], self.cos_sin)
        torch.bmm(o3.view(Rd, 8, 4096).transpose(0, 1), B["wo_aT"], out=b["z"].view(Rd, 8, 1024).transpose(0, 1))
        if self.dense_fp8:
            self._mm8(b["z"], B, "wo_b", b["y"])
        else:
            torch.mm(b["z"], B["wo_b"].t(), out=b["y"])

    def ffn_bd(self, i):
        b, B = self.b, self.blocks[i]
        self.boundary(b["y"], b["X"], b["a_post"], b["a_comb"], b["a_pre"], B["ff"], B["sf"], B["bff"], B["nf"],
                      b["Xm"], b["f_pre"], b["f_post"], b["f_comb"], b["f"])

    def router(self, i):
        """gate GEMM (bf16 inputs, exact products, fp32 accumulation and output) + one fused kernel (score, top-3,
        weights, routing row). The torch-ops router took 59 us per block, the fp32 sgemm alone 14 us."""
        b, B = self.b, self.blocks[i]
        lg = torch.mm(b["f"], B["gate_wb"].t(), out_dtype=torch.float32)  # tensor-core bf16 products, fp32 accumulation
        rr = b["route_row"].view(torch.int32)
        _router_kernel[(self.Rd,)](lg, B["gate_b"], rr, rr.view(torch.float32), NE_=NE, K_=TOPK, ROW_WORDS=16,
                                   SCALE=1.5, num_warps=4)

    def router_torch(self, i):
        b, B = self.b, self.blocks[i]
        scores = torch.nn.functional.softplus(b["f"].float() @ B["gate_w"].t()).sqrt()
        idx = (scores + B["gate_b"]).topk(TOPK, dim=-1).indices
        wts = scores.gather(-1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * 1.5
        rr = b["route_row"]
        ids6 = torch.cat([idx.int(), torch.full_like(idx.int(), -1)], 1)
        w6 = torch.cat([wts.float(), torch.zeros_like(wts.float())], 1)
        rr[:, :24].copy_(ids6.contiguous().view(torch.uint8))
        rr[:, 24:48].copy_(w6.contiguous().view(torch.uint8))

    def moe(self, i):
        b, B, m, e = self.b, self.blocks[i], self.m, self.me
        Tall = self.Tall
        fall = b["gf"].view(torch.bfloat16).view(Tall, D)
        ids = b["gr"][:, :24].contiguous().view(torch.int32).view(Tall, TOPK_PAD)
        wts = b["gr"][:, 24:48].contiguous().view(torch.float32).view(Tall, TOPK_PAD)
        if self.moe_mode == "v4":
            return self._moe_v4(B, m, fall, ids, wts, Tall)
        torch.mm(fall, B["sw13"].t(), out=m["gu"])
        e.swiglu(m["gu"], m["hsh"])
        torch.mm(m["hsh"], B["sw2"].t(), out=m["sh"])
        if self.moe_mode == "base":
            e.route(ids, wts, Tall, m["ecs"], m["t1"], m["t2"], m["counts"], m["slotmap"], m["rts"], m["sms"], m["empty_f"])
        elif self.moe_mode == "route2":
            self.r2.route2(ids, wts, Tall, m["ecs"], m["t1"], m["t2"], m["counts"], m["slotmap"], m["rts"], m["sms"],
                           m["empty_f"])
        else:
            self.r2.route2x(ids, wts, Tall, m["ecs"], m["t1"], m["t2"], m["counts"], m["slotmap"], m["rts"], m["sms"],
                            m["empty_f"], m["ec8"])
        e.prepr(fall, m["ecs"], m["counts"], m["max_ec"], m["img"], m["ptok"], m["inv"])
        e.gemm1r(B["w13"], B["s13"], m["ecs"], m["t1"], m["counts"], m["img"], m["himg"], 0)
        if self.moe_mode == "n8":
            e.gemm1r8(B["w13"], B["s13"], m["ecs"], m["ec8"], m["counts"], m["img"], m["himg"], 0)
        e.gemm2r(B["w2"], B["s2"], m["ecs"], m["t2"], m["counts"], m["himg"], m["Y"], 0)
        if self.moe_mode == "n8":
            e.gemm2r8(B["w2"], B["s2"], m["ecs"], m["ec8"], m["counts"], m["himg"], m["Y"], 0)
        e.combine(m["Y"], m["slotmap"], m["sh"], m["part"])

    def _moe_v4(self, B, m, fall, ids, wts, Tall):
        """The kernels/moe_v4 drop-in: its route2 (pair items in t2) + prepr6 + its GEMMs,
        the shared expert on a side stream (forked before route2, joined before combine). H, Y bit-identical."""
        e4, r4 = v4_ext()
        if self.sh_stream is None:
            assert int(r4.route2_max_ec(Tall)) <= m["max_ec"] and int(r4.route2_scratch_ints()) <= m["rts"].numel(), \
                "moe_v4 route2 needs larger tables"
            self.sh_stream = torch.cuda.Stream(device=self.dev)
        cur = torch.cuda.current_stream()
        ev = torch.cuda.Event()
        ev.record(cur)
        self.sh_stream.wait_event(ev)
        with torch.cuda.stream(self.sh_stream):
            torch.mm(fall, B["sw13"].t(), out=m["gu"])
            e4.swiglu(m["gu"], m["hsh"])
            torch.mm(m["hsh"], B["sw2"].t(), out=m["sh"])
            ev2 = torch.cuda.Event()
            ev2.record(self.sh_stream)
        r4.route2(ids, wts, Tall, m["ecs"], m["t1"], m["t2"], m["counts"], m["slotmap"], m["rts"], m["sms"], m["empty_f"])
        e4.prepr6(fall, m["ecs"], m["counts"], m["img"], m["ptok"], m["inv"], m["sms"])
        e4.gemm1r(B["w13"], B["s13"], m["ecs"], m["t1"], m["counts"], m["img"], m["himg"], 0)
        e4.gemm2r(B["w2"], B["s2"], m["ecs"], m["t2"], m["counts"], m["himg"], m["Y"], 0)
        cur.wait_event(ev2)
        e4.combine(m["Y"], m["slotmap"], m["sh"], m["part"])

    def final_bd(self):
        b, B = self.b, self.blocks[-1]
        self.boundary_moe(b["f_post"], b["f_comb"], b["f_pre"], B["fa"], B["sa"], B["ba"], self.final_norm, b["X"],
                          b["o_pre"], b["o_post"], b["o_comb"], b["hd"], mix=False)  # o_* are never read


# ====================================================================================== collectives
class EmuComm:
    """Single-device emulation of the AG / RS of the drafter blocks (exactly the fixed-order sums of comm.cu)."""

    def ag_f(self, ranks):
        for r in ranks:
            for s, src in enumerate(ranks):
                Rd = src.Rd
                r.b["gf"][s * Rd:(s + 1) * Rd].copy_(src.b["f"].view(torch.uint8).view(Rd, D * 2))
                r.b["gr"][s * Rd:(s + 1) * Rd].copy_(src.b["route_row"])

    def rs(self, ranks):
        for r_i, r in enumerate(ranks):
            Rd = r.Rd
            acc = sum(src.m["part"][r_i * Rd:(r_i + 1) * Rd].float() for src in ranks)  # fixed order 0..3
            r.b["moe_out"].copy_(acc.to(torch.bfloat16))


class PeerComm:
    """comm.cu peer stores (one process, 4 devices): per-rank push / wait, capturable per device."""

    def __init__(self, ranks, ncta=8):
        import comm_head
        self.e = comm_head.ext()
        self.e.enable_peers(W)
        self.ranks, self.ncta = ranks, ncta
        z = lambda r: torch.zeros(4, dtype=torch.int32, device=r.dev)  # noqa: E731
        self.fl_f = [z(r) for r in ranks]
        self.fl_s = [z(r) for r in ranks]
        self.ep_f = [torch.zeros(1, dtype=torch.int32, device=r.dev) for r in ranks]
        self.ep_s = [torch.zeros(1, dtype=torch.int32, device=r.dev) for r in ranks]
        self.gf = [r.b["gf"] for r in ranks]
        self.gr = [r.b["gr"] for r in ranks]
        self.rsb = [r.b["rs"].view(torch.uint8) for r in ranks]
        for r in ranks:  # the zero-filled flags / epochs must be visible before any peer signals them
            torch.cuda.synchronize(r.dev)

    def ag_f_rank(self, i):
        r = self.ranks[i]
        self.e.ag_push2(r.b["f"].view(torch.uint8).view(r.Rd, D * 2), r.b["route_row"], i, self.gf, self.fl_f, self.gr,
                        self.ep_f[i], self.ncta)
        self.e.wait(self.fl_f[i], self.ep_f[i], W * self.ncta)

    def rs_rank(self, i):
        r = self.ranks[i]
        self.e.rs_push(r.m["part"], i, self.rsb, self.fl_s, self.ep_s[i], self.ncta)
        if r.fuse_rs and (r.Rd <= 128 or r.mhc2):  # the next boundary waits + sums (boundary_rs); no k_rs_sum
            r.fold = (self.fl_s[i], self.ep_s[i], W * self.ncta)
        else:
            self.e.rs_sum(r.b["rs"], self.fl_s[i], self.ep_s[i], W * self.ncta, r.b["moe_out"], 132)


def stages(ranks, comm):
    """The drafter step as a list of (name, kind, fn): kind 'local' -> fn(rank_obj), 'ag' / 'rs' -> collective."""
    st = [("context_kv", "local", lambda d: d.context_kv())]
    for i in range(len(ranks[0].blocks)):
        st += [(f"b{i}.attn_bd", "local", lambda d, i=i: d.attn_bd(i)),
               (f"b{i}.attn", "local", lambda d, i=i: d.attention(i)),
               (f"b{i}.ffn_bd", "local", lambda d, i=i: d.ffn_bd(i)),
               (f"b{i}.router", "local", lambda d, i=i: d.router(i)),
               (f"b{i}.ag", "ag", None),
               (f"b{i}.moe", "local", lambda d, i=i: d.moe(i)),
               (f"b{i}.rs", "rs", None)]
    st.append(("final_bd", "local", lambda d: d.final_bd()))
    return st


def run_emulated(ranks, streams=None):
    """All ranks on one device (or several, synchronously): stage by stage."""
    comm = EmuComm()
    for name, kind, fn in stages(ranks, comm):
        if kind == "local":
            for d in ranks:
                with torch.cuda.device(d.dev):
                    fn(d)
        elif kind == "ag":
            comm.ag_f(ranks)
        else:
            comm.rs(ranks)


def run_rank(ranks, comm: PeerComm, i, only=None):
    """Rank i's whole drafter step (for one CUDA graph per device). only: optional set of stage names."""
    d = ranks[i]
    for name, kind, fn in stages(ranks, comm):
        if only is not None and name.split(".")[-1] not in only and name not in only:
            continue
        if kind == "local":
            fn(d)
        elif kind == "ag":
            comm.ag_f_rank(i)
        else:
            comm.rs_rank(i)
