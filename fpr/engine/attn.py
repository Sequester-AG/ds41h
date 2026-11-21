"""Attention half of one layer, split at the indexer hook.

The same launches, kernels, buffers and order as kernels/attn_dense/attn_half.attn_half (imported read-only: its extension,
workspace and W8A16 objects are reused), cut in two so the indexer / compressor (kernels/indexer2f) can run between the q / kv
norms (they need qr and the attention input) and FlashMLA (which reads their top-512 and the compressed records):

  attn_a(W, ws, a, pos)                 q_a | wkv GEMM -> norm_rope2 (qr, kv)                                   [hook here]
  attn_b(W, c, ws, pos, insert)         q_b + RoPE(q) + window insert -> FlashMLA (window + top-512) -> inverse RoPE
                                        -> wo_a -> wo_b  -> ws["y"]

attn_a + attn_b == attn_half bit for bit (same calls). Differences, both production's:
  * ratio-0 layers (0, 1) call FlashMLA WITHOUT the extra (compressed) cache, as production's flashmla.py does for SWA-only
    layers (extra_k_cache None); attn_half always passes one.
  * c["sched"] may be a fresh FlashMLASchedMeta inside a CUDA-graph capture: the first call of a layer type then captures
    FlashMLA's in-kernel planner, so every replay re-plans for the step's lengths (production: one sched per layer type
    per step, model.py / flashmla.py:212-228). attn_half plans outside the graph instead.
"""
from __future__ import annotations

import os
import sys

import torch

PHYS = os.environ.get("PHYS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(PHYS, "kernels/attn_dense"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("ATTN2B_PDL", "1")
import attn_half as AH  # noqa: E402

SWA_SHIFT = AH.SWA_SHIFT
assert not AH.SPA and not AH.QA8, "engine attn: ATTN2B_SPA / ATTN2B_QA8 paths not supported (attn_half defaults only)"


def workspace(R: int, dev) -> dict:
    return AH._workspace(R, dev)


def _a8_rows(W: dict, ws: dict, which: str):
    """ENG_PF_ACT8_DENSE (lossy opt-in): the bucket-A prefill rows [a, b) when the layer's 8-bit GEMM applies."""
    ab = ws.get("pf_ab")
    if ab is None or W.get("A8") is None or torch.cuda.is_current_stream_capturing():
        return None
    a, b = ab
    return ab if _a8().applies(W["L"], which, b - a) else None


def attn_a(W: dict, ws: dict, a_rows: torch.Tensor, pos: torch.Tensor) -> None:
    kd = AH._ext()
    ab = _a8_rows(W, ws, "qkv")
    d8 = W.get("A8c")
    if d8 is not None and "qkv" in d8 and d8_rows(ws):   # ENG_C128_DENSE8_ADD qkv: every row of the call on the 8-bit arm
        _a8().qkv(d8, a_rows, ws["qakv"])
    elif ab is not None:          # prefill rows: production's W8A8 q_a | wkv; the other rows 16-bit as below
        a, b = ab
        R = ws["R"]
        if a > 0:
            torch.mm(a_rows[:a], W["wqkv_a"].t(), out=ws["qakv"][:a])
        _a8().qkv(W["A8"], a_rows[a:b], ws["qakv"][a:b])
        if b < R:
            torch.mm(a_rows[b:], W["wqkv_a"].t(), out=ws["qakv"][b:])
    elif not (ws.get("pf_ab") is None and skw8_dense(W, "wqkv_a", a_rows, ws["qakv"], ws["R"], valid=ws.get("valid"))):
        torch.mm(a_rows, W["wqkv_a"].t(), out=ws["qakv"])
    kd.norm_rope2(ws["qakv"][:, :1280], W["q_norm"], ws["qr"], ws["qakv"][:, 1280:], W["kv_norm"], ws["kv"], W["eps"],
                  pos, W["cos_sin"])


def attn_b(W: dict, c: dict, ws: dict, pos: torch.Tensor, insert: bool = True, pf=None) -> torch.Tensor:
    """attn_b_insert + attn_b_attend (the same calls; CP prefill runs its window-halo exchange between them)."""
    attn_b_insert(W, c, ws, pos, insert)
    return attn_b_attend(W, c, ws, pos, pf=pf)


# ENG_SKW8_QB: decode buckets of <= 32 rows run q_b on kernels/attn_dense/skw8.cu, then dense.cu rope_q_insert, as the ENG_QB16 paths do.
# ENG_C1_ATTN_HG: TP-wo decode buckets run the attention core on kernels/attn_hg
# (ds41h's 16-head sparse decode kernel with 4 head groups in one launch) instead of FlashMLA: same records, indices,
# lengths, sink and scale; fp32 accumulation, another summation order. ENG_C1_ATTN_HG_CL: cluster size (0 = auto).
C1_ATTN_HG = os.environ.get("ENG_C1_ATTN_HG", "0") == "1"
C1_ATTN_HG_CL = int(os.environ.get("ENG_C1_ATTN_HG_CL", "0"))
C1_ATTN_HG_MAXR = int(os.environ.get("ENG_C1_ATTN_HG_MAXR", "32"))   # buckets of <= this many rows per rank (with a small
                                                                       # bucket, e.g. 8: C1 on hg, fuller steps on FlashMLA)
C1_ATTN_HG_SPREAD = os.environ.get("ENG_C1_ATTN_HG_SPREAD", "1") == "1"   # 1: one CTA per SM (ds41h default)
C1_ATTN_HG_SKIP = os.environ.get("ENG_C1_ATTN_HG_SKIP", "1") == "1"   # padding rows' clusters exit at once (TP-wo scatters
                                                                      # only valid rows)
_HG = []


def attn_hg():
    """kernels/attn_hg/sparse_decode_hg.cu (built into engine/build/attn_hg; ds41h's kernel headers from ds41h/kernels)."""
    if not _HG:
        from torch.utils.cpp_extension import load
        bd = os.path.join(PHYS, "engine/build/attn_hg")
        os.makedirs(bd, exist_ok=True)
        _HG.append(load(name="fpr_attn_hg", sources=[os.path.join(PHYS, "kernels/attn_hg/sparse_decode_hg.cu")],
                        extra_cflags=["-O3", "-std=c++20"],
                        extra_cuda_cflags=["-O3", "-std=c++20", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a",
                                           "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas=-O3"],
                        extra_include_paths=[os.path.join(os.path.dirname(PHYS), "ds41h/kernels")], build_directory=bd,
                        verbose=False))
    return _HG[0]


# ENG_C128A: decode buckets of ENG_C128A_MINR ..
# fp32 accumulation in another order. Every global read after griddepcontrol.wait (ins = -1: the indexer writes ti / tl right before). Window-only
# layers (0, 1) keep FlashMLA. Built only when the flag is on.
C128A = os.environ.get("ENG_C128A", "0") == "1"
# wo_a / wo_b of every call that is
# not TP-wo (decode buckets above ENG_C1_TPWO_MAXR rows, mixed / prefill steps) on production's 8-bit arm
# (kernels/attn_dense/attn_act8: inverse RoPE fused into the 1x128 quant (ropeq64) + DeepGEMM fp8_einsum for wo_a,
# act_quant 1x128 + fp8_gemm_nt for wo_b) from folded e4m3 copies W["A8c"]. The full bf16 wo_aT / wo_b are then not
# loaded (engine._load): C1 reads this rank's quarter W["wo_aT_c1"] / W["wo_b_c1"].
DENSE8 = os.environ.get("ENG_C128_DENSE8", "0") == "1"
# ENG_C128_DENSE8_ADD=qkv,qb,sh: the same 8-bit
# arm (act_quant 1x128 + DeepGEMM fp8_gemm_nt on production's folded e4m3, W["A8c"] keys "qkv" / "q_b"; the engine's
# lw["A8sh"] "sh13" / "sh2" = this rank's shared-expert slice) for q_a | wkv, q_b (+ dense.cu rope_q_insert, = the fused
# W8A16 RoPE + window insert) and the shared expert, in the calls DENSE8 moves for wo: decode buckets above DENSE8_MAXR rows
# (the engine sets it = ENG_C1_TPWO_MAXR) and mixed steps with prefill rows. The bf16 / W8A16 copies stay loaded (C1, small
# buckets, production's fused prefill MoE read them), so each item ADDS memory until its other readers move (S2).
DENSE8_ADD = frozenset(x for x in os.environ.get("ENG_C128_DENSE8_ADD", "").split(",") if x) if DENSE8 else frozenset()
assert DENSE8_ADD <= {"qkv", "qb", "sh"}, f"ENG_C128_DENSE8_ADD: unknown items {sorted(DENSE8_ADD - {'qkv', 'qb', 'sh'})}"
DENSE8_MAXR = 32


def d8_rows(ws: dict) -> bool:
    """ENG_C128_DENSE8_ADD: does this call take the 8-bit arm (mixed step with prefill rows, or a decode bucket > MAXR)?"""
    return ws.get("pf_ab") is not None or ws["R"] > DENSE8_MAXR


C128A_MINR = int(os.environ.get("ENG_C128A_MINR", "112"))
C128A_MAXR = int(os.environ.get("ENG_C128A_MAXR", "256"))
C128A_VAR = os.environ.get("ENG_C128A_VAR", "sp6:SPA_QR=4,SPA_REG_C=216,SPA_REG_P=72")
_C128A = []


def c128a():
    """"""
    if not _C128A:
        p = os.path.join(PHYS, "kernels/c128a")
        if p not in sys.path:
            sys.path.insert(0, p)
        os.environ.setdefault("C128A_BUILD", os.path.join(PHYS, "engine/build/c128a"))
        import c128a as CA
        name, _, defs = C128A_VAR.partition(":")
        _C128A.append((CA, name, CA.build(name, defs)))
    return _C128A[0]


SKW8_QB = os.environ.get("ENG_SKW8_QB", "0") == "1"
_SKW8 = []


SKW8_NST = int(os.environ.get("ENG_SKW8_NST", "6"))
SKW8_NACC = int(os.environ.get("ENG_SKW8_NACC", "1"))   # > 1: SAME MATH, NOT BITWISE (accumulator sets, skw8t.cu SKT_NACC)
SKW8_H = os.environ.get("ENG_SKW8_H", "0") == "1"        # the 128-row-tile deep-ring variant (gemm_h)


SKW8_SRC = "skw8t.cu"   # (skw8t2.cu = skw8t + a chunk stride: views of a bigger packed tensor; bitwise skw8t on contiguous)
_W8TP = {}
_W8TP_CNT = {}


def w8tp():
    """kernels/attn_dense/w8a16tp.cu, the served W8A16 defines (env W8A16_DEFS) + W8_SMAX=16 (wo_a: 8 tiles
    x 16 splits = 128 CTAs)."""
    if "x" not in _W8TP:
        from torch.utils.cpp_extension import load
        user = [d for d in os.environ.get("W8A16_DEFS", "").split(",") if d] + ["W8_SMAX=16"]
        tag = "_".join(d.replace("=", "").lower() for d in user)
        bd = os.path.join(PHYS, f"engine/build/eng_w8a16tp_{tag}")
        os.makedirs(bd, exist_ok=True)
        _W8TP["x"] = load(name=f"eng_w8a16tp_{tag}", sources=[os.path.join(PHYS, "kernels/attn_dense/w8a16tp.cu")],
                          build_directory=bd, extra_ldflags=["-lcuda"], verbose=False,
                          extra_include_paths=[os.path.join(PHYS, "kernels/attn_dense")],
                          extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
                                             "-DW8_ABL=0", "-DW8_PROF=0", "-DW8_DQ=1"] + [f"-D{d}" for d in user])
    return _W8TP["x"]


def _w8tp_ws(dev):
    """per device: TP-wo's OWN split-K fp32 workspace (max(16 splits x 8 tiles (wo_a), 6 x 20 (wo_b)) x 128 x 260 floats =
    17 MB; not w8a16's shared one, whose other users may run on other streams) + one monotonic arrival-counter set per
    TP-wo GEMM (one split count each; both GEMMs run in stream order on the step stream). The engine creates them for every
    device at init (ENG_WO_ONECOPY): never allocated inside a graph capture."""
    k = str(dev)
    if k not in _W8TP_CNT:
        _W8TP_CNT[k] = (torch.empty(max(16 * 8, 6 * 20) * 128 * 260, dtype=torch.float32, device=dev),
                        torch.zeros(64, dtype=torch.int32, device=dev), torch.zeros(64, dtype=torch.int32, device=dev))
    return _W8TP_CNT[k]


def skw8():
    """kernels/attn_dense/skw8t.cu (the TMA-pipelined skinny W8A16; padding rows after the step's last valid row skipped)."""
    if not _SKW8:
        from torch.utils.cpp_extension import load
        tag = f"nst{SKW8_NST}_nacc{SKW8_NACC}" + ("" if SKW8_SRC == "skw8t.cu" else "_" + SKW8_SRC.split(".")[0])
        bd = os.path.join(PHYS, f"engine/build/eng_skw8t_{tag}")
        os.makedirs(bd, exist_ok=True)
        x = load(name=f"eng_skw8t_{tag}", sources=[os.path.join(PHYS, "kernels/attn_dense", SKW8_SRC)],
                 build_directory=bd, extra_ldflags=["-lcuda"], verbose=False,
                 extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                                    f"-DSKT_NST={SKW8_NST}", f"-DSKT_NACC={SKW8_NACC}"])
        _SKW8.append(x.gemm_h if SKW8_H else x.gemm)
        _SKW8.append(x.gemm_vw)
    return _SKW8[0]


def skw8_vw():
    """skw8t gemm whose row mask comes from the previous kernel (griddepcontrol.wait before reading it)."""
    skw8()
    return _SKW8[1]


# ENG_SKW8_DENSE=wqkv_a,wo_a,wo_b (any subset): those GEMMs of decode buckets of <= 32 rows on skw8 from packed fp8 copies
# (engine load: serve/weights.build_attn8), K splits ENG_SKW8_S; the bf16 copies stay for larger buckets and mixed steps.
SKW8_DENSE = tuple(x for x in os.environ.get("ENG_SKW8_DENSE", "").split(",") if x)
SKW8_S = dict(dict(wqkv_a=1, wo_a=1, wo_b=2), **{k: int(v) for k, v in (kv.split(":") for kv in
                                                                       os.environ.get("ENG_SKW8_S", "").split(",") if kv)})
_SKW8_WS = {}


def _skw8_ws(dev):
    """per device: fp32 split-K partials (up to 4 x 32 x 8192) + one arrival counter per 64-row subtile and GEMM name."""
    k = str(dev)
    if k not in _SKW8_WS:
        _SKW8_WS[k] = (torch.empty(8 * 32 * 32768, dtype=torch.float32, device=dev),
                       {n: torch.zeros(512, dtype=torch.int32, device=dev)
                        for n in ("wqkv_a", "wo_a", "wo_b", "q_b", "tp_wo_a", "tp_wo_b")})
    return _SKW8_WS[k]


def skw8_dense(W: dict, name: str, x, y, R: int, rpg: int = 0, kxg: int = 0, valid=None) -> bool:
    """y = x @ W[name]^T on skw8t when enabled for this GEMM and bucket (decode buckets: valid = the step's row mask);
    False: the caller runs its own path."""
    pk = W.get("pk8")
    if pk is None or name not in pk or R > 32 or valid is None:
        return False
    N, K = dict(wqkv_a=(1792, 5120), wo_a=(8192, 4096), wo_b=(5120, 8192))[name]
    ws, cnt = _skw8_ws(x.device)
    skw8()(x, pk[name], y, N, K, rpg, kxg, SKW8_S[name], ws, cnt[name], valid.view(torch.uint8).view(R))
    return True


QB16 = False     # engine.QB16_MIXED (set per mixed step): q_b of buckets with >= 256 rows on cuBLAS bf16 (bitwise)
QB16D = False    # engine.QB16_DECODE (set per decode step): decode buckets with > 128 rows on the bf16 q_b, captures included
QB_DQ = False    # engine.QB_DQ: q_b of >= DQ_MINR rows (not capturing) = the exact bf16 of the packed
DQ_MINR = 384    # fp8 copy (engine._Dq) + cuBLAS + RoPE / insert: the QB16 branch's calls on a dequantized weight (bitwise)


def attn_b_insert(W: dict, c: dict, ws: dict, pos: torch.Tensor, insert: bool = True) -> None:
    """q_b + RoPE(q) + the rows' own window-KV insert."""
    kd = AH._ext()
    R = ws["R"]
    q3 = ws["q"].view(R, 64, 512)
    w8 = W.get("w8")
    full = w8 is not None and "wo_a" in w8
    capt = torch.cuda.is_current_stream_capturing()
    d8 = W.get("A8c")
    if d8 is not None and "q_b" in d8 and d8_rows(ws):   # ENG_C128_DENSE8_ADD qb: 8-bit q_b, then RoPE(q) + window insert
        _a8().q_b(d8, ws["qr"], ws["q"].view(R, 32768))
        kd.rope_q_insert(q3, ws["kv"], pos, W["cos_sin"], c["own_slot"], c["sbuf"], c["stride"], SWA_SHIFT, bool(insert))
    elif W.get("wq_b16") is not None and ((QB16 and R >= 256 and not capt) or (QB16D and R > 128 and ws.get("pf_ab") is None)):
        ab = _a8_rows(W, ws, "q_b")
        qf = ws["q"].view(R, 32768)
        if ab is not None:        # prefill rows' q_b on production's W8A8 (the rest bf16); RoPE + insert over all rows
            a, b = ab
            if a > 0:
                torch.mm(ws["qr"][:a], W["wq_b16"].t(), out=qf[:a])
            _a8().q_b(W["A8"], ws["qr"][a:b], qf[a:b])
            if b < R:
                torch.mm(ws["qr"][b:], W["wq_b16"].t(), out=qf[b:])
        else:
            torch.mm(ws["qr"], W["wq_b16"].t(), out=qf)
        kd.rope_q_insert(q3, ws["kv"], pos, W["cos_sin"], c["own_slot"], c["sbuf"], c["stride"], SWA_SHIFT, bool(insert))
    elif QB_DQ and w8 is not None and "wq_b_dq" in w8 and R >= DQ_MINR and not capt:   # no bf16 q_b copy
        dq = w8["wq_b_dq"]
        torch.mm(ws["qr"], dq.weight().t(), out=ws["q"].view(R, 32768))
        dq.done()
        kd.rope_q_insert(q3, ws["kv"], pos, W["cos_sin"], c["own_slot"], c["sbuf"], c["stride"], SWA_SHIFT, bool(insert))
    elif SKW8_QB and w8 is not None and "wq_b" in w8 and R <= 32 and ws.get("pf_ab") is None and "valid" in ws:
        g = w8["wq_b"]
        wsk, cnt = _skw8_ws(ws["qr"].device)
        skw8()(ws["qr"], g.packed, ws["q"].view(R, 32768), g.N, g.K, 0, 0, SKW8_S.get("q_b", 1), wsk, cnt["q_b"],
               ws["valid"].view(torch.uint8).view(R))
        kd.rope_q_insert(q3, ws["kv"], pos, W["cos_sin"], c["own_slot"], c["sbuf"], c["stride"], SWA_SHIFT, bool(insert))
    elif w8 is not None and "wq_b" in w8:   # attn_half.w8_qb: the kernel takes <= 128 rows per call (rows independent)
        qb = getattr(AH, "w8_qb", None)
        if qb is not None:
            qb(w8["wq_b"], ws["qr"], ws["q"].view(R, 32768), pos, W["cos_sin"], kv=ws["kv"] if insert else None,
               slot=c["own_slot"] if insert else None, cache=c["sbuf"] if insert else None, page_stride=c["stride"],
               shift=SWA_SHIFT)
        else:
            assert R <= 128, "w8a16 q_b: <= 128 rows per call (attn_half without w8_qb)"
            w8["wq_b"](ws["qr"], ws["q"].view(R, 32768), pos=pos, cs=W["cos_sin"], kv=ws["kv"] if insert else None,
                       slot=c["own_slot"] if insert else None, cache=c["sbuf"] if insert else None,
                       page_stride=c["stride"], shift=SWA_SHIFT)
    else:
        torch.mm(ws["qr"], W["wq_b"].t(), out=ws["q"].view(R, 32768))
        kd.rope_q_insert(q3, ws["kv"], pos, W["cos_sin"], c["own_slot"], c["sbuf"], c["stride"], SWA_SHIFT, bool(insert))


def attn_b_attend(W: dict, c: dict, ws: dict, pos: torch.Tensor, pf=None) -> torch.Tensor:
    """FlashMLA (window + top-512) -> inverse RoPE -> wo_a -> wo_b -> ws["y"].
    pf (ENG_ATTN_PF: kernels/attn_dense/attn_prefill.Plan of the step's prefill rows, bucket A only): rows [:row_lo]
    (decode) on the decode kernel with their own plan (c["sched_dec"]), rows [row_lo, row_hi) through production's prefill
    attention (attn_prefill.attend: gather + combine + flash_mla_sparse_fwd), padding rows zeroed; then the same inverse
    RoPE / wo over all rows."""
    from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache

    kd = AH._ext()
    R = ws["R"]
    w8 = W.get("w8")
    full = w8 is not None and "wo_a" in w8
    comp = c.get("k_comp") is not None
    if pf is not None:
        a, b = pf.row_lo, pf.row_hi
        if a > 0:
            flash_mla_with_kvcache(q=ws["q"][:a], k_cache=c["k_swa"], block_table=None, head_dim_v=512,
                                   tile_scheduler_metadata=c["sched_dec"], cache_seqlens=None, is_fp8_kvcache=True,
                                   indices=c["si"][:a], topk_length=c["sl"][:a], softmax_scale=c["scale"],
                                   attn_sink=W["sink"], extra_k_cache=c["k_comp"] if comp else None,
                                   extra_indices_in_kvcache=c["ti"][:a] if comp else None,
                                   extra_topk_length=c["tl"][:a] if comp else None, out=ws["o"][:a])
        AP = _ap()
        AP.attend(pf, c["ratio"], ws["q"][a:b].view(b - a, 64, 512), ws["o"][a:b].view(b - a, 64, 512), c["k_swa"],
                  c["k_comp"] if comp else None, c["ti"][a:b] if comp else None, c["scale"], W["sink"],
                  src=c.get("src"), isrc=c.get("isrc"))
        if b < R:
            ws["o"][b:].zero_()
    elif C128A and comp and ws.get("c128a") and ws.get("pf_ab") is None:   # ENG_C128A: attention + inverse RoPE in one kernel
        CA, name, ext = c128a()
        CA.attn(ext, name, ws["q"], ws["o"], c, W["sink"], pos, W["cos_sin"], ins=-1)
    else:
        flash_mla_with_kvcache(q=ws["q"], k_cache=c["k_swa"], block_table=None, head_dim_v=512,
                               tile_scheduler_metadata=c["sched"], cache_seqlens=None, is_fp8_kvcache=True,
                               indices=c["si"], topk_length=c["sl"], softmax_scale=c["scale"], attn_sink=W["sink"],
                               extra_k_cache=c["k_comp"] if comp else None,
                               extra_indices_in_kvcache=c["ti"] if comp else None,
                               extra_topk_length=c["tl"] if comp else None, out=ws["o"])
    d8 = W.get("A8c")   # ENG_C128_DENSE8 (the layer's folded wo_a / wo_b); its wo_a quant applies the inverse RoPE itself
    if d8 is None and not (C128A and pf is None and comp and ws.get("c128a") and ws.get("pf_ab") is None):
        kd.inv_rope(ws["o"].view(R, 64, 512), pos, W["cos_sin"])
    o3 = ws["o"].view(R, 64, 512)
    if d8 is not None:
        _a8().wo_a(d8, o3, pos, W["cos_sin"], ws["z"].view(R, 8, 1024))
        _a8().wo_b(d8, ws["z"], ws["y"])
    elif full:
        w8["wo_a"](ws["o"].view(R, 32768), ws["z"])
        w8["wo_b"](ws["z"], ws["y"])
    elif pf is None and ws.get("pf_ab") is None and "pk8" in W and R <= 32:
        if not skw8_dense(W, "wo_a", ws["o"].view(R, 32768), ws["z"], R, 1024, 4096, valid=ws.get("valid")):
            torch.bmm(o3.view(R, 8, 4096).transpose(0, 1), W["wo_aT"], out=ws["z"].view(R, 8, 1024).transpose(0, 1))
        if not skw8_dense(W, "wo_b", ws["z"], ws["y"], R, valid=ws.get("valid")):
            torch.mm(ws["z"], W["wo_b"].t(), out=ws["y"])
    else:
        torch.bmm(o3.view(R, 8, 4096).transpose(0, 1), W["wo_aT"], out=ws["z"].view(R, 8, 1024).transpose(0, 1))
        ab = _a8_rows(W, ws, "wo_b")
        if ab is not None:        # prefill rows' wo_b on production's W8A8 (wo_a stays 16-bit for all rows)
            a, b = ab
            if a > 0:
                torch.mm(ws["z"][:a], W["wo_b"].t(), out=ws["y"][:a])
            _a8().wo_b(W["A8"], ws["z"][a:b], ws["y"][a:b])
            if b < R:
                torch.mm(ws["z"][b:], W["wo_b"].t(), out=ws["y"][b:])
        else:
            torch.mm(ws["z"], W["wo_b"].t(), out=ws["y"])
    return ws["y"]


def attn_b_attend_o(W: dict, c: dict, ws: dict, pos: torch.Tensor, inv: bool = True) -> torch.Tensor:
    """Decode buckets, ENG_C1_TPWO: attn_b_attend up to the attention output (FlashMLA + inverse RoPE) -> ws["o"];
    wo_a / wo_b then run split over the 4 GPUs (engine.Engine._tp_wo)."""
    from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache

    kd = AH._ext()
    R = ws["R"]
    comp = c.get("k_comp") is not None
    if C128A and comp and ws.get("c128a"):   # ENG_C128A with ENG_C128A_MINR <= 32 (quality tests at C1 .. C32)
        CA, name, ext = c128a()
        CA.attn(ext, name, ws["q"], ws["o"], c, W["sink"], pos, W["cos_sin"] if inv else None, ins=-1)
        return ws["o"]
    if C1_ATTN_HG and R <= C1_ATTN_HG_MAXR:
        attn_hg().sparse_decode(ws["q"].view(R, 64, 512), ws["o"].view(R, 64, 512), c["k_swa"].select(2, 0), c["si"],
                                c["sl"], W["sink"], float(c["scale"]),
                                extra_cache=c["k_comp"].select(2, 0) if comp else None,
                                extra_indices=c["ti"] if comp else None, extra_lens=c["tl"] if comp else None,
                                cluster=C1_ATTN_HG_CL, spread=C1_ATTN_HG_SPREAD, pdl=True, hg=4,
                                rvalid=ws["valid"].view(torch.uint8).view(-1) if C1_ATTN_HG_SKIP and "valid" in ws else None)
    else:
        flash_mla_with_kvcache(q=ws["q"], k_cache=c["k_swa"], block_table=None, head_dim_v=512,
                               tile_scheduler_metadata=c["sched"], cache_seqlens=None, is_fp8_kvcache=True,
                               indices=c["si"], topk_length=c["sl"], softmax_scale=c["scale"], attn_sink=W["sink"],
                               extra_k_cache=c["k_comp"] if comp else None,
                               extra_indices_in_kvcache=c["ti"] if comp else None,
                               extra_topk_length=c["tl"] if comp else None, out=ws["o"])
    if inv:      # (ENG_C1_TPWO_ROPE: the TP-wo scatter applies it on the way instead)
        kd.inv_rope(ws["o"].view(R, 64, 512), pos, W["cos_sin"])
    return ws["o"]


def _ap():
    """kernels/attn_dense/attn_prefill (production's prefill attention for the engine's prefill rows)."""
    import attn_prefill
    return attn_prefill


def _a8():
    """kernels/attn_dense/attn_act8 (production's W8A8 prefill dense path, ENG_PF_ACT8_DENSE)."""
    import attn_act8
    return attn_act8
