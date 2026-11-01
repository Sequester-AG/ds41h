"""Unit tests + micro-benchmark for the ds41h `moe` component (raw-layout MXFP4 experts: W4A16 default
(lossless vs stock vLLM W4A16), W4A8 opt-in).

  GPU tests (GPUs free):   python -m pytest tests/test_moe.py -x -q
  CPU-only subset (no CUDA ctx):   CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_moe.py -q -k cpu
  Benchmark:                       python tests/test_moe.py --bench [--tokens 8192] [--layer 10] [--rank 0] [--uniform]

Env: DS41H_TEST_LAYER (default 10), DS41H_TEST_RANK (default 0), DS41H_MODEL_DIR (checkpoint, default
$DS41H_HOME/model). Real weights of one layer / TP rank are read from the checkpoint with
ds41h.ref.weights (vLLM-exact TP slicing); the references dequantize with ds41h.ref.weights.dequant_fp4,
independently of ds41h.moe.dequant_fp4 and the CUDA kernels.
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

try:
    import pytest
except ImportError:  # minimal stand-in so `python tests/test_moe.py` works without pytest installed

    class _Skip(Exception):
        pass

    class _Mark:
        @staticmethod
        def skipif(cond, reason=""):
            def deco(fn):
                if cond:
                    fn._skip = reason
                return fn

            return deco

        @staticmethod
        def parametrize(names, values):
            def deco(fn):
                fn._params = getattr(fn, "_params", []) + [([n.strip() for n in names.split(",")], values)]
                return fn

            return deco

    class pytest:  # type: ignore[no-redef]
        mark = _Mark()
        Skip = _Skip

        @staticmethod
        def skip(msg=""):
            raise _Skip(msg)

        @staticmethod
        def main(args):
            import itertools
            import traceback

            sel = os.environ.get("DS41H_TEST_K", "")
            mod = sys.modules[__name__]
            tests = [(n, f) for n, f in vars(mod).items() if n.startswith("test_") and callable(f) and sel in n]
            fails = 0
            for name, fn in tests:
                if getattr(fn, "_skip", None):
                    print(f"SKIP {name}: {fn._skip}")
                    continue
                grids = [[dict(zip(ns, v if isinstance(v, tuple) else (v,))) for v in vals] for ns, vals in getattr(fn, "_params", [])]
                for combo in itertools.product(*grids) if grids else [()]:
                    kw = {k: v for d in combo for k, v in d.items()}
                    try:
                        fn(**kw)
                        print(f"PASS {name} {kw}")
                    except _Skip as e:
                        print(f"SKIP {name} {kw}: {e}")
                    except Exception:
                        fails += 1
                        print(f"FAIL {name} {kw}")
                        traceback.print_exc()
            print(f"{fails} failed")
            return 1 if fails else 0

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ds41h import moe  # noqa: E402
from ds41h.ref.weights import CheckpointReader, default_reader, dequant_fp4 as ref_dequant_fp4  # noqa: E402

if os.environ.get("DS41H_MOE_DEV_KERNELS"):  # dev: build the extension from another kernels dir (moe_route.cu + moe_prefill.cu)
    from ds41h import cuext as _cuext

    @functools.cache
    def _dev_ext():
        from torch.utils.cpp_extension import load as _load

        kd = os.environ["DS41H_MOE_DEV_KERNELS"]
        return _load(name="ds41h_moe_dev_" + os.path.basename(kd.rstrip("/")), sources=[f"{kd}/moe_route.cu", f"{kd}/moe_prefill.cu"], extra_cflags=["-O3", "-std=c++20"],
                     extra_cuda_cflags=_cuext.NVCC_FLAGS, extra_include_paths=[str(_cuext.KERNELS)])

    moe.ext = _dev_ext

LAYER = int(os.environ.get("DS41H_TEST_LAYER", "10"))
RANK = int(os.environ.get("DS41H_TEST_RANK", "0"))
H, I_R, E, TOPK, LIMIT, ROUTE_SCALE = 5120, 576, 384, 6, 10.0, 1.5

HAS_GPU = torch.cuda.is_available()
gpu = pytest.mark.skipif(not HAS_GPU, reason="needs a GPU")
torch.backends.cuda.matmul.allow_tf32 = False  # fp32 references must be real fp32
torch.backends.cudnn.allow_tf32 = False


# ===================================================================================== helpers
def reader() -> CheckpointReader:
    return default_reader(os.environ.get("DS41H_MODEL_DIR"))


@functools.cache
def raw_layer(layer: int = LAYER, rank: int = RANK):
    """(w13, s13, w2, s2) of one layer / rank on the GPU, plus gate weight and bias."""
    w13, s13, w2, s2 = moe.load_raw_experts(layer, rank, "cuda", model_dir=os.environ.get("DS41H_MODEL_DIR"))
    lay = reader().layer(layer)
    gate = lay["ffn.gate.weight"].cuda()
    bias = lay["ffn.gate.bias"].float().cuda()
    return w13, s13, w2, s2, gate, bias


@functools.cache
def dequant_layer(layer: int = LAYER, rank: int = RANK):
    """Dequantized experts via ds41h.ref (independent of ds41h.moe): W13 [E, 2I, H], W2 [E, H, I].
    Stored as bf16, which is exact for e2m1 x ue8m0 values (<= 2 significant bits); use .float()."""
    w13, s13, w2, s2, *_ = raw_layer(layer, rank)
    W13 = torch.stack([ref_dequant_fp4(w13[e], s13[e]).to(torch.bfloat16) for e in range(w13.shape[0])])
    W2 = torch.stack([ref_dequant_fp4(w2[e], s2[e]).to(torch.bfloat16) for e in range(w2.shape[0])])
    assert torch.equal(W13[0].float(), ref_dequant_fp4(w13[0], s13[0]))
    return W13, W2


def per_expert(rows_e: torch.Tensor, fn) -> None:
    """Call fn(e, row_index_tensor) for every expert present in rows_e."""
    order = torch.argsort(rows_e, stable=True)
    counts = torch.bincount(rows_e, minlength=E).tolist()
    start = 0
    for e, c in enumerate(counts):
        if c:
            fn(e, order[start : start + c])
            start += c


def stock_route(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, k: int = TOPK, scale: float = ROUTE_SCALE):
    """vLLM dsv4_topk semantics (fused_moe/router/dsv4_topk.py) on fp32 router logits."""
    logits = x.float() @ gate.float().t()
    w = torch.sqrt(torch.where(logits > 20.0, logits, torch.log(1.0 + torch.exp(logits))))
    choice = w + bias
    choice = torch.where(choice == choice, choice, torch.full_like(choice, -1e30))
    ids = torch.topk(choice, k, dim=-1, sorted=True).indices
    sel = w.gather(1, ids)
    s = sel.sum(-1, keepdim=True)
    sel = sel * scale / torch.where(s > 0, s, torch.ones_like(s))
    return logits, sel.float(), ids.int()


def frag_to_phys(K: int, device=None) -> torch.Tensor:
    """phys[L] = physical K index stored at logical (fragment-order) column L (see moe_route.cu)."""
    idx = []
    base = 0
    while base < K:
        kb = 128 if K - base >= 128 else 64
        for L in range(kb):
            s, h, t, i = L // 32, (L // 16) % 2, (L // 4) % 4, L % 4
            idx.append(base + t * (kb // 4) + 8 * s + 4 * h + i)
        base += kb
    return torch.tensor(idx, dtype=torch.long, device=device)


def unpermute(q: torch.Tensor) -> torch.Tensor:
    """Fragment-order e4m3 bytes [rows, K] -> float values in physical order."""
    K = q.shape[1]
    phys = frag_to_phys(K, q.device)
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    out[:, phys] = q.view(torch.float8_e4m3fn).float()
    return out


def e4m3_exact_rows(T: int, K: int, seed: int = 0) -> torch.Tensor:
    """bf16 rows that the per-row act quant represents exactly (amax = 448 * 2^e_row)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(T, K, generator=g, device="cuda") * 40).clamp(-440, 440).to(torch.float8_e4m3fn).float()
    q[:, 7] = 448.0
    e = torch.randint(-12, -6, (T, 1), generator=g, device="cuda").float()
    return (q * torch.exp2(e)).to(torch.bfloat16)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


META_CTR = 8  # moe_route.cu / moe_prefill.cu


def run_permute(ids: torch.Tensor, w: torch.Tensor, n_exp: int):
    """moe_permute; meta = counts[E+1] | off[E+1] | ctr[8] | perm_tok[R] | perm_slot[R] | pos_of[R]."""
    R = ids.numel()
    assert moe.ext().meta_size(n_exp, R) == 2 * (n_exp + 1) + META_CTR + 3 * R
    meta = torch.full((moe.ext().meta_size(n_exp, R),), -7, dtype=torch.int32, device=ids.device)
    perm_w = torch.empty(R, dtype=torch.float32, device=ids.device)
    moe.ext().moe_permute(ids.int().contiguous(), w.float().contiguous(), n_exp, meta, perm_w)
    E1 = n_exp + 1
    t0 = 2 * E1 + META_CTR
    return SimpleNamespace(
        meta=meta, perm_w=perm_w, counts=meta[:E1], off=meta[E1 : 2 * E1], ctr=meta[2 * E1 : t0],
        tok=meta[t0 : t0 + R], slot=meta[t0 + R : t0 + 2 * R], pos_of=meta[t0 + 2 * R : t0 + 3 * R], R=R,
    )


def expert_of_rows(p) -> torch.Tensor:
    """Expert id of every routed row (pos order) from the offsets."""
    n = int(p.off[-1])
    pos = torch.arange(n, device=p.off.device)
    return torch.searchsorted(p.off[1:].long(), pos, right=True)


# ============================================================================ CPU: weight loading
def _fake_routed_experts(n_local: int, rank: int, tp: int = 4):
    """A RoutedExperts shell with just the state its stock weight_loader reads, using the ds41h method."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    moe_cfg = SimpleNamespace(
        w13_num_shards=2, has_bias=False, tp_rank=rank, tp_size=tp, is_act_and_mul=True, tp_shard_with_padding=False,
        moe_parallel_config=SimpleNamespace(tp_size=tp, use_ep=False, ep_size=1),
    )
    re = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(re)
    re.quant_config = None
    re.moe_config = moe_cfg
    re._fused_shared_expert_quantizer = None
    re._loaded_expert_biases = set()
    re.layer_name = f"model.layers.{LAYER}.ffn.experts"
    re.quant_method = moe.make_raw_mxfp4_moe_method(moe_cfg)
    re.quant_method.create_weights(re, n_local, H, I_R, torch.bfloat16, weight_loader=re.weight_loader)
    return re


def test_cpu_raw_loading_matches_checkpoint():
    """Stock RoutedExperts.weight_loader + ds41h create_weights == checkpoint TP slice ([w1; w3] rows)."""
    rd = reader()
    lay = rd.layer(LAYER)
    experts = [0, 1, 191, 383]
    for rank in (0, 3):
        re = _fake_routed_experts(len(experts), rank)
        local = {g: i for i, g in enumerate(experts)}
        re._map_global_expert_id_to_local_expert_id = lambda g, local=local: local.get(g, -1)
        prefix = re.layer_name
        for g in experts:
            for shard, param in (("w1", "w13"), ("w3", "w13"), ("w2", "w2")):
                for suffix, pname in (("weight", f"{param}_weight"), ("scale", f"{param}_weight_scale")):
                    p = getattr(re, pname)
                    t = lay[f"ffn.experts.{g}.{shard}.{suffix}"]  # exact checkpoint dtype (int8 / e8m0fnu)
                    ok = p.weight_loader(p, t, f"{prefix}.routed_experts.{pname}", shard_id=shard, expert_id=g, return_success=True)
                    assert ok, (g, shard, suffix)
        w13, s13, w2, s2 = moe.load_raw_experts(LAYER, rank, "cpu", experts=experts)
        assert torch.equal(re.w13_weight.data, w13) and torch.equal(re.w13_weight_scale.data, s13)
        assert torch.equal(re.w2_weight.data, w2) and torch.equal(re.w2_weight_scale.data, s2)
        # documented layout: rows [0, 576) = w1 (gate) rows 576r.., rows [576, 1152) = w3 (up); w2 cols 288r..
        g = experts[2]
        u8 = torch.uint8
        assert torch.equal(re.w13_weight.data[2, :I_R], lay[f"ffn.experts.{g}.w1.weight"].view(u8)[rank * I_R : (rank + 1) * I_R])
        assert torch.equal(re.w13_weight.data[2, I_R:], lay[f"ffn.experts.{g}.w3.weight"].view(u8)[rank * I_R : (rank + 1) * I_R])
        assert torch.equal(re.w2_weight.data[2], lay[f"ffn.experts.{g}.w2.weight"].view(u8)[:, rank * 288 : (rank + 1) * 288])
        assert torch.equal(re.w2_weight_scale.data[2], lay[f"ffn.experts.{g}.w2.scale"].view(u8)[:, rank * 18 : (rank + 1) * 18])
        re.quant_method.process_weights_after_loading(_LayerShim(re))
        assert moe.is_raw_moe(re)
        raw = moe.get_raw_experts(re)
        want = (re.w13_weight, re.w13_weight_scale, re.w2_weight, re.w2_weight_scale)
        assert all(a.data_ptr() == b.data_ptr() and a.shape == b.shape for a, b in zip(raw, want))


class _LayerShim:
    """process_weights_after_loading needs .activation/.apply_router_weight_on_input/.layer_name too."""

    def __init__(self, re):
        object.__setattr__(self, "_re", re)

    def __getattr__(self, k):
        if k == "activation":
            return SimpleNamespace(value="silu")
        if k == "apply_router_weight_on_input":
            return False
        return getattr(self._re, k)

    def __setattr__(self, k, v):
        setattr(self._re, k, v)


def test_cpu_patch_swaps_routed_experts_method():
    """patch_prefill wraps DeepseekV4FP8Config.get_quant_method: RoutedExperts -> ds41h raw method,
    dense linears -> stock method; the hook registry carries the patch when the component is on."""
    import importlib

    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase

    from ds41h import patch_prefill, patches

    qc = importlib.import_module(patch_prefill.QUANT_CONFIG_MODULE)
    cls = qc.DeepseekV4FP8Config
    saved = cls.get_quant_method
    base = getattr(saved, "__wrapped__", saved)  # the import hook may already have patched it (DS41H=1)
    try:
        cls.get_quant_method = base
        patch_prefill.install_raw_mxfp4_moe(qc)
        patch_prefill.install_raw_mxfp4_moe(qc)  # idempotent
        assert cls.get_quant_method._ds41h_raw_moe and cls.get_quant_method.__wrapped__ is base
        cfg = cls.from_config({"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32], "fmt": "e4m3", "scale_fmt": "ue8m0"})
        re = RoutedExperts.__new__(RoutedExperts)
        torch.nn.Module.__init__(re)
        re.moe_config = SimpleNamespace(w13_num_shards=2, has_bias=False)
        m = cfg.get_quant_method(re, f"model.layers.{LAYER}.ffn.experts")
        assert type(m).__name__ == "Ds41hRawMxfp4MoEMethod"
        lin = LinearBase.__new__(LinearBase)
        torch.nn.Module.__init__(lin)
        assert type(cfg.get_quant_method(lin, f"model.layers.{LAYER}.attn.wq_b")).__name__ != "Ds41hRawMxfp4MoEMethod"
    finally:
        cls.get_quant_method = saved
    if moe.ENABLED:
        assert patch_prefill.install_raw_mxfp4_moe in patches.PATCHES[patch_prefill.QUANT_CONFIG_MODULE]


# ---- bit-exact emulation of the GEMM consumer data path (moe_prefill.cu + sm90.cuh helpers)
def _prmt(a: int, b: int, sel: int) -> int:
    """PTX prmt.b32 default mode: selector nibble = byte index (3 lsb) + sign-replicate flag (msb)."""
    src = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    out = 0
    for i in range(4):
        n = (sel >> (4 * i)) & 0xF
        v = src[n & 7]
        if n & 8:
            v = 0xFF if v & 0x80 else 0x00
        out |= v << (8 * i)
    return out


def _e4m3_byte(v: float) -> int:
    return int(torch.tensor([v]).to(torch.float8_e4m3fn).view(torch.uint8))


def _lut(d: int) -> tuple[int, int]:  # fp4_e4m3_lut
    if d >= -11:
        k = (12 + d) << 3
        return (k * 0x01010100 + 0x0C080000) & 0xFFFFFFFF, (k * 0x01010101 + 0x1C181410) & 0xFFFFFFFF
    sc = 2.0 ** (6 + d)
    b = [_e4m3_byte(v * sc) for v in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)]
    return b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24, b[4] | b[5] << 8 | b[6] << 16 | b[7] << 24


def _expand(w: int, l0: int, l1: int) -> tuple[int, int]:  # fp4x8_to_e4m3_lut
    m = w & 0x77777777
    w4 = (w << 4) & 0xFFFFFFFF
    lo = (_prmt(w, w4, 0x9D8C) & 0x80808080) | _prmt(l0, l1, m)
    hi = (_prmt(w, w4, 0xBFAE) & 0x80808080) | _prmt(l0, l1, m >> 16)
    return lo, hi


def _dec(word: int) -> list[float]:
    return torch.tensor([(word >> (8 * i)) & 0xFF for i in range(4)], dtype=torch.uint8).view(torch.float8_e4m3fn).float().tolist()


def test_cpu_fp4_expansion_exact():
    """Every (code, sign, relative exponent d) expands to value * 2^(6+d) exactly for d >= -14."""
    fp4 = list(moe.FP4_VALUES)
    g = torch.Generator().manual_seed(1)
    for d in list(range(0, -15, -1)):
        l0, l1 = _lut(d)
        for _ in range(8):
            w = int(torch.randint(0, 2**32, (1,), generator=g, dtype=torch.int64))
            lo, hi = _expand(w, l0, l1)
            got = _dec(lo) + _dec(hi)
            want = [fp4[(w >> (4 * i)) & 0xF] * 2.0 ** (6 + d) for i in range(8)]
            assert got == want, (d, hex(w), got, want)


@pytest.mark.parametrize("mode,kb", [(1, 128), (2, 128), (2, 64)])
def test_cpu_emulated_tile_matches_exact(mode, kb):
    """Emulate one K block of one CTA tile (2 consumer WGs x 4 warps x 32 lanes): staging reads,
    per-row reference exponents, LUT expansion, RS-wgmma fragments (cute ALayout_64x32), logical
    B columns written by act_quant/h_quant, promotion -- and compare with the exact product."""
    g = torch.Generator().manual_seed(mode * 1000 + kb)
    BN = 16
    nbytes = kb // 2
    fp4 = torch.randint(0, 256, (128, nbytes), generator=g, dtype=torch.uint8)  # staging rows (tile-local A rows)
    exps = torch.randint(115, 125, (128, kb // 32), generator=g, dtype=torch.uint8)
    extra = torch.randint(0, 4, (128,), generator=g)  # d = e - e_ref down to -12 (slow LUT path, still exact)
    W = ref_dequant_fp4(fp4, exps)  # [128, kb] exact
    xq = (torch.randn(BN, kb, generator=g) * 30).to(torch.float8_e4m3fn)  # physical order
    phys = frag_to_phys(kb)
    b_logical = xq.float()[:, phys]  # what the B smem tile holds (fragment order)
    ref = W @ xq.float().t()  # [128 staging rows, BN]
    got = torch.full((128, BN), float("nan"))
    for wg in range(2):
        for wq in range(4):
            D = torch.zeros(64, BN)  # this WG's accumulator, warp wq owns rows 16wq..16wq+15
            rows = {}
            frag = {}
            for lane in range(32):
                gg, t = lane >> 2, lane & 3
                sr0 = 32 * wg + 8 * wq + gg if mode == 1 else 64 * wg + 16 * wq + gg
                sr1 = 64 + sr0 if mode == 1 else sr0 + 8
                rows[gg] = (sr0, sr1)
                sub = t if kb == 128 else t >> 1
                for which, sr in ((0, sr0), (1, sr1)):
                    e = int(exps[sr, sub])
                    # per-row reference exponent: the kernel uses the max ue8m0 byte of the whole row
                    # (>= this block's max); emulate a row max up to 3 binades above the block's
                    r = int(exps[sr].max()) + int(extra[sr])
                    l0, l1 = _lut(e - r)
                    words = fp4[sr, (16 if kb == 128 else 8) * t:][: (16 if kb == 128 else 8)].numpy().view("<u4").tolist()
                    for s, wv in enumerate(words):
                        lo, hi = _expand(int(wv), l0, l1)
                        frag[(lane, s, which, 0)] = _dec(lo)  # k 4t..4t+3
                        frag[(lane, s, which, 1)] = _dec(hi)  # k 16+4t..16+4t+3
                    rows.setdefault(("ws", gg, which), 2.0 ** (r - 133))
            for s in range(kb // 32):
                A = torch.zeros(64, 32)
                for lane in range(32):
                    gg, t = lane >> 2, lane & 3
                    for which in (0, 1):
                        m = 16 * wq + gg + 8 * which
                        A[m, 4 * t : 4 * t + 4] = torch.tensor(frag[(lane, s, which, 0)])
                        A[m, 16 + 4 * t : 20 + 4 * t] = torch.tensor(frag[(lane, s, which, 1)])
                D += A @ b_logical[:, 32 * s : 32 * s + 32].t()
            for gg in range(8):
                sr0, sr1 = rows[gg]
                got[sr0] = D[16 * wq + gg] * rows[("ws", gg, 0)]
                got[sr1] = D[16 * wq + gg + 8] * rows[("ws", gg, 1)]
    assert not torch.isnan(got).any()
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-4)


# ---- W4A16 (LD 4) data path emulation: a16_phys order, FP4 -> bf16 expansion, RS-wgmma bf16 fragments
def a16_phys(L: int) -> int:
    """moe_prefill.cu a16_phys: physical K column held at logical column L (0..63) of a 64-wide K block."""
    s, t = L >> 4, (L >> 1) & 3
    n = 2 * (s & 1) + ((L >> 3) & 1) + 4 * (L & 1)
    return 16 * t + 8 * (s >> 1) + ((n + t) & 7)


def a16_phys_index(K: int, device=None) -> torch.Tensor:
    """phys[L] for a whole row of K (multiple of 64) columns."""
    blk = torch.tensor([a16_phys(L) for L in range(64)], dtype=torch.long)
    return (torch.arange(0, K, 64)[:, None] + blk[None, :]).reshape(-1).to(device)


def _bf16_bits_to_float(bits: list[int]) -> list[float]:
    return torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float().tolist()


def _a16_pair_raw(w: int, p: int) -> int:  # fp4_pair_bf16_raw<p>
    ms, ss = 6 - 4 * p, 12 - 4 * p
    mag = ((w << ms) if ms >= 0 else (w >> -ms)) & 0x01C001C0
    sgn = (w << ss) & 0x80008000
    return (mag | sgn) & 0xFFFFFFFF


def _a16_expand(w_rot: int, p: int, S: int) -> list[float]:
    """bf16x2 (low, high) of rotated word pair p times 2^(S-1) (mul.rn.bf16x2, exact)."""
    raw = _a16_pair_raw(w_rot, p)
    lo, hi = _bf16_bits_to_float([raw & 0xFFFF, raw >> 16])
    m = 2.0 ** (S - 1)
    return [lo * m, hi * m]


def test_cpu_a16_order_and_banks():
    """a16_phys is a permutation of every 64-block, and the 8 GEMM1 weight rows a warp reads with one LDS
    (a16_phys of 8 consecutive logical columns, 32-byte rows) cover 4 distinct 8-bank groups twice."""
    p = [a16_phys(L) for L in range(64)]
    assert sorted(p) == list(range(64))
    for c0 in range(0, 64, 8):
        groups = sorted(a16_phys(c) % 4 for c in range(c0, c0 + 8))
        assert groups == [0, 0, 1, 1, 2, 2, 3, 3], (c0, groups)
    assert torch.equal(torch.sort(a16_phys_index(5120)).values, torch.arange(5120))


def test_cpu_a16_expansion_exact():
    """Every e2m1 code (incl. -0, 0.5 subnormal placement) times every scale byte S in [0, 128] expands exactly."""
    fp4 = list(moe.FP4_VALUES)
    for S in list(range(0, 129, 3)) + [117, 126, 127, 128]:
        for w in (0x76543210, 0xFEDCBA98, 0x1F2E3D4C, 0x89ABCDEF):
            for p in range(4):
                got = _a16_expand(w, p, S)
                want = [fp4[(w >> (4 * p)) & 0xF] * 2.0 ** (S - 127), fp4[(w >> (4 * (p + 4))) & 0xF] * 2.0 ** (S - 127)]
                assert got == want, (S, hex(w), p, got, want)


@pytest.mark.parametrize("mode", [1, 2])
def test_cpu_a16_emulated_tile_matches_exact(mode):
    """Emulate one 64-wide K block of a W4A16 tile (2 consumer WGs x 4 warps x 32 lanes): 8-byte FP4 loads of
    rows sr0 / sr1 (MODE 1: a16_phys row choice), rotation by 4t, bf16 expansion with the block scale, RS-wgmma
    bf16 m64nNk16 fragments (R0 = (g, 2t..2t+1), R1 = (g+8, ..), R2 = (g, 2t+8..), R3 = (g+8, 2t+8..)) against the
    B tile in logical order -- the exact product of the physical weights and activations."""
    g = torch.Generator().manual_seed(7 + mode)
    BN = 16
    fp4 = torch.randint(0, 256, (128, 32), generator=g, dtype=torch.uint8)  # A rows (tile-local), 64 K values each
    exps = torch.randint(117, 127, (128, 2), generator=g, dtype=torch.uint8)
    W = ref_dequant_fp4(fp4, exps)  # [128, 64] exact
    xb = (torch.randn(BN, 64, generator=g)).to(torch.bfloat16).float()  # physical order
    phys = a16_phys_index(64)
    b_logical = xb[:, phys]
    ref = W @ xb.t()
    got = {}
    for wg in range(2):
        for wq in range(4):
            D = torch.zeros(64, BN)
            for s in range(4):
                A = torch.zeros(64, 16)
                for lane in range(32):
                    gg, t = lane >> 2, lane & 3
                    c = 32 * wg + 8 * wq + gg
                    sr0 = a16_phys(c) if mode == 1 else 64 * wg + 16 * wq + gg
                    sr1 = 64 + sr0 if mode == 1 else sr0 + 8
                    for which, sr in ((0, sr0), (1, sr1)):
                        words = fp4[sr, 8 * t : 8 * t + 8].numpy().view("<u4").tolist()
                        w = int(words[s >> 1])
                        w_rot = ((w >> (4 * t)) | (w << (32 - 4 * t))) & 0xFFFFFFFF if t else w
                        S = int(exps[sr, t >> 1])
                        r = 16 * wq + gg + 8 * which
                        A[r, 2 * t : 2 * t + 2] = torch.tensor(_a16_expand(w_rot, 2 * (s & 1), S))
                        A[r, 2 * t + 8 : 2 * t + 10] = torch.tensor(_a16_expand(w_rot, 2 * (s & 1) + 1, S))
                D += A @ b_logical[:, 16 * s : 16 * s + 16].t()
            for gg in range(8):
                c = 32 * wg + 8 * wq + gg
                if mode == 1:
                    sr0 = a16_phys(c)
                    got[sr0] = D[16 * wq + gg]
                    got[64 + sr0] = D[16 * wq + gg + 8]
                else:
                    got[64 * wg + 16 * wq + gg] = D[16 * wq + gg]
                    got[64 * wg + 16 * wq + gg + 8] = D[16 * wq + gg + 8]
    assert sorted(got) == list(range(128))
    out = torch.stack([got[r] for r in range(128)])
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_cpu_fragment_order_is_permutation():
    for K in (5120, 576, 128, 64):
        p = frag_to_phys(K)
        assert torch.equal(torch.sort(p).values, torch.arange(K))


def test_cpu_ref_forward_matches_naive():
    """moe_forward_ref (ds41h.moe.dequant_fp4) vs a naive per-token loop using ds41h.ref dequant."""
    g = torch.Generator().manual_seed(0)
    n_exp, T = 3, 5
    w13 = torch.randint(0, 256, (n_exp, 2 * I_R, H // 2), generator=g, dtype=torch.uint8)
    s13 = torch.randint(118, 124, (n_exp, 2 * I_R, H // 32), generator=g, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (n_exp, H, I_R // 2), generator=g, dtype=torch.uint8)
    s2 = torch.randint(118, 124, (n_exp, H, I_R // 32), generator=g, dtype=torch.uint8)
    x = torch.randn(T, H, generator=g) * 0.1
    ids = torch.randint(-1, n_exp, (T, 2), generator=g).int()
    w = torch.rand(T, 2, generator=g)
    y = moe.moe_forward_ref(x, w, ids, w13, s13, w2, s2, LIMIT, compute_dtype=torch.float32)
    ref = torch.zeros(T, H)
    for t in range(T):
        for k in range(2):
            e = int(ids[t, k])
            if e < 0:
                continue
            gu = ref_dequant_fp4(w13[e], s13[e]) @ x[t]
            h = F.silu(gu[:I_R].clamp(max=LIMIT)) * gu[I_R:].clamp(-LIMIT, LIMIT) * w[t, k]
            ref[t] += ref_dequant_fp4(w2[e], s2[e]) @ h
    assert rel_err(y, ref) < 1e-5


# ============================================================================ GPU: stages
@gpu
def test_act_quant_exact():
    T = 257
    # (bf16 * f32 promotes to f32: cast back, act_quant takes bf16 [T, H])
    x = (torch.randn(T, H, device="cuda") * torch.rand(T, 1, device="cuda") * 3).to(torch.bfloat16)
    x[5] = 0
    xq = torch.empty(T, H, dtype=torch.uint8, device="cuda")
    xs = torch.empty(T, dtype=torch.float32, device="cuda")
    moe.ext().act_quant(x, xq, xs)
    amax = x.float().abs().amax(-1)
    inv = torch.where(amax > 0, 448.0 / amax, torch.ones_like(amax))
    q_ref = (x.float() * inv[:, None]).to(torch.float8_e4m3fn).float()
    q = unpermute(xq)
    assert (q == q_ref).float().mean().item() > 0.9999
    torch.testing.assert_close(xs, torch.where(amax > 0, amax / 448.0, torch.ones_like(amax)))
    assert rel_err(q * xs[:, None], x) < 0.04


@gpu
def test_h_quant_exact():
    R, n = 300, 251
    h = (torch.randn(R, I_R, device="cuda") * torch.rand(R, 1, device="cuda")).to(torch.bfloat16)
    hq = torch.full((R, I_R), 0x5A, dtype=torch.uint8, device="cuda")
    hs = torch.full((R,), -1.0, device="cuda")
    moe.ext().h_quant(h, torch.tensor([n], dtype=torch.int32, device="cuda"), hq, hs)
    amax = h.float().abs().amax(-1)
    inv = torch.where(amax > 0, 448.0 / amax, torch.ones_like(amax))
    q_ref = (h.float() * inv[:, None]).to(torch.float8_e4m3fn).float()
    q = unpermute(hq[:n])
    assert (q == q_ref[:n]).float().mean().item() > 0.9999
    assert torch.all(hq[n:] == 0x5A) and torch.all(hs[n:] == -1.0)
    torch.testing.assert_close(hs[:n], amax[:n] / 448.0)


@gpu
@pytest.mark.parametrize("T,n_exp,k", [(1000, 384, 6), (8192, 384, 6), (1, 384, 6), (5000, 128, 3)])
def test_permute(T, n_exp, k):
    ids = torch.randint(0, n_exp, (T, k), device="cuda", dtype=torch.int32)
    ids[0, k - 1] = -1
    ids[T - 1, 0] = n_exp  # out of range -> dropped
    if T > 100:
        ids[50:90] = 5  # a hot expert
    w = torch.rand(T, k, device="cuda")
    p = run_permute(ids, w, n_exp)
    valid = (ids >= 0) & (ids < n_exp)
    cnt = torch.bincount(ids[valid].long(), minlength=n_exp)
    assert torch.equal(p.counts[:n_exp].long(), cnt) and int(p.counts[n_exp]) == int(valid.sum())
    assert torch.equal(p.off.long(), F.pad(cnt.cumsum(0), (1, 0)))
    assert torch.all(p.ctr == 0)
    n = int(valid.sum())
    rows_e = expert_of_rows(p)
    slot = p.slot[:n].long()
    assert torch.equal(ids.reshape(-1)[slot].long(), rows_e)
    assert torch.equal(p.tok[:n].long(), slot // k)
    assert torch.equal(p.perm_w[:n], w.reshape(-1)[slot])
    assert torch.equal(torch.sort(slot).values, torch.nonzero(valid.reshape(-1)).squeeze(1))
    # pos_of is the inverse permutation (slot -> routed row), -1 for dropped ids
    pos_of = p.pos_of.long()
    assert torch.equal(pos_of[slot], torch.arange(n, device=ids.device))
    assert torch.all(pos_of[~valid.reshape(-1)] == -1)


@gpu
@pytest.mark.parametrize("n_exp,k", [(384, 6), (128, 3)])
def test_route_topk(n_exp, k):
    T = 513
    logits = torch.randn(T, n_exp, device="cuda") * 3
    bias = torch.randn(n_exp, device="cuda") * 0.1
    w, ids = moe.route_topk(logits, bias, k, ROUTE_SCALE)
    sp = torch.where(logits > 20.0, logits, torch.log(1.0 + torch.exp(logits))).sqrt()
    ids_ref = torch.topk(sp + bias, k, dim=-1).indices
    sel = sp.gather(1, ids_ref)
    w_ref = sel * ROUTE_SCALE / sel.sum(-1, keepdim=True)
    assert (ids.long() == ids_ref).all(dim=1).float().mean().item() > 0.999
    same = (ids.long() == ids_ref).all(dim=1)
    torch.testing.assert_close(w[same], w_ref[same], rtol=1e-5, atol=1e-6)


GEMM1_VARIANTS = (1, 2, 3, 4, 5, 6, 7, 8)  # moe_prefill.cu dispatch_gemm
GEMM2_VARIANTS = (1, 2, 3, 4, 7, 8)  # 5 (reduce into out[T, H]) is covered by the forward tests


@gpu
@pytest.mark.parametrize("T,variant", [(1, 1), (37, 1), (37, 4), (512, 1), (512, 2), (512, 3), (512, 4), (3000, 1), (3000, 2),
                                       (1, 5), (11, 5), (37, 6), (512, 5), (512, 6), (3000, 5), (8192, 5),
                                       (1, 7), (37, 7), (512, 8), (3000, 7), (8192, 7)])
def test_gemm1_exact_inputs(T, variant):
    """GEMM1 + SwiGLU with exactly representable activations: only accumulation + bf16 error."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    W13, _ = dequant_layer()
    x = e4m3_exact_rows(T, H, seed=T)
    _, tw, ids = stock_route(x, gate, bias)
    p = run_permute(ids, tw, E)
    xq = torch.empty(T, H, dtype=torch.uint8, device="cuda")
    xs = torch.empty(T, dtype=torch.float32, device="cuda")
    moe.ext().act_quant(x, xq, xs)
    assert torch.equal((unpermute(xq) * xs[:, None]).to(torch.bfloat16), x)  # lossless by construction
    if variant in (5, 6):  # TMA-fed variants take the activations in routed-row order
        xq_p = torch.full((p.R, H), 0x7F, dtype=torch.uint8, device="cuda")
        xs_p = torch.empty(T, dtype=torch.float32, device="cuda")
        moe.ext().act_quant_perm(x, p.pos_of, TOPK, xq_p, xs_p)
        n_ = int(p.off[-1])
        assert torch.equal(xq_p[:n_], xq[p.tok[:n_].long()]) and torch.equal(xs_p, xs)
        xq = xq_p
    hbuf = torch.full((p.R, I_R), float("nan"), dtype=torch.bfloat16, device="cuda")
    moe.ext().moe_gemm(1, xq, xs, w13, s13, p.meta, p.perm_w, p.R, variant, LIMIT, hbuf)
    n = int(p.off[-1])
    rows_e = expert_of_rows(p)
    tok = p.tok[:n].long()
    gu = torch.empty(n, 2 * I_R, device="cuda")

    def gemm1_ref(e, r):
        gu[r] = x.float()[tok[r]] @ W13[e].float().t()

    per_expert(rows_e, gemm1_ref)
    g, u = gu.split(I_R, dim=-1)
    h_ref = F.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT) * p.perm_w[:n, None]
    h = hbuf[:n].float()
    assert torch.isfinite(h).all()
    err = rel_err(h, h_ref)
    print(f"GEMM1 T={T} variant={variant}: rel_err {err:.2e}")
    assert err < 5e-3, err
    assert ((h - h_ref).abs().amax(-1) <= 2e-2 * h_ref.abs().amax(-1) + 1e-6).all()
    # the kernel is repeatable (dynamic tile scheduling, self-resetting counter) and bit-deterministic
    hbuf2 = torch.full_like(hbuf, float("nan"))
    moe.ext().moe_gemm(1, xq, xs, w13, s13, p.meta, p.perm_w, p.R, variant, LIMIT, hbuf2)
    assert torch.equal(hbuf2[:n], hbuf[:n])


@gpu
@pytest.mark.parametrize("T,variant", [(1, 1), (100, 1), (100, 2), (700, 1), (3000, 1), (3000, 2), (1, 3), (20, 3), (100, 3), (700, 3), (3000, 3), (8192, 3),
                                       (100, 4), (3000, 4), (1, 7), (20, 7), (700, 7), (3000, 7), (100, 8), (3000, 8), (8192, 7)])
def test_gemm2_exact_inputs(T, variant):
    """GEMM2 on quantized h: reference uses the dequantized hq, so only accumulation/bf16 error remains."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    _, W2 = dequant_layer()
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    p = run_permute(ids, tw, E)
    n = int(p.off[-1])
    h = (torch.randn(p.R, I_R, device="cuda") * torch.rand(p.R, 1, device="cuda")).to(torch.bfloat16)
    hq = torch.empty(p.R, I_R, dtype=torch.uint8, device="cuda")
    hs = torch.empty(p.R, dtype=torch.float32, device="cuda")
    moe.ext().h_quant(h, p.counts[E:], hq, hs)
    slot = torch.full((p.R, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    moe.ext().moe_gemm(2, hq, hs, w2, s2, p.meta, p.perm_w, p.R, variant, LIMIT, slot)
    hv = unpermute(hq[:n]) * hs[:n, None]
    rows_e = expert_of_rows(p)
    y_ref = torch.empty(n, H, device="cuda")

    def gemm2_ref(e, r):
        y_ref[r] = hv[r] @ W2[e].float().t()

    per_expert(rows_e, gemm2_ref)
    y = slot[p.slot[:n].long()].float() * hs[:n, None]  # GEMM2 leaves the per-row scale to moe_combine
    assert torch.isfinite(y).all()
    err = rel_err(y, y_ref)
    print(f"GEMM2 T={T} variant={variant}: rel_err {err:.2e}")
    assert err < 5e-3, err
    # repeatable (dynamic scheduling, self-resetting counters) and bit-deterministic
    slot2 = torch.full_like(slot, float("nan"))
    moe.ext().moe_gemm(2, hq, hs, w2, s2, p.meta, p.perm_w, p.R, variant, LIMIT, slot2)
    rows = p.slot[:n].long()
    assert torch.equal(slot2[rows], slot[rows])


# ---- W4A16 stages
A16_GEMM1 = (9, 10)
A16_GEMM2 = (11, 12)
BF16_ULP = 2.0 ** -8  # relative spacing of bf16 values (one ulp at the bottom of a binade)


def bf16_close(got: torch.Tensor, ref: torch.Tensor, what: str, ulps: float = 1.0) -> None:
    """Elementwise: |got - ref| <= ulps * bf16 ulp(|ref|) + a tiny absolute floor (fp32 summation-order noise).
    got is a bf16 rounding of a value equal to ref up to fp32 accumulation order (rel ~1e-6)."""
    g, r = got.float(), ref.float()
    tol = ulps * BF16_ULP * r.abs() + 1e-5 * r.abs().max().clamp_min(1e-30)
    bad = (g - r).abs() > tol
    assert torch.isfinite(g).all(), what
    assert not bad.any(), f"{what}: {int(bad.sum())} / {g.numel()} elements off by > {ulps} bf16 ulp; max rel {((g - r).abs() / r.abs().clamp_min(1e-30)).max().item():.3e}"


@gpu
def test_act_perm16_exact():
    T, k = 333, TOPK
    x = (torch.randn(T, H, device="cuda") * 3).to(torch.bfloat16)
    ids = torch.randint(0, E, (T, k), device="cuda", dtype=torch.int32)
    ids[3, 2] = -1  # dropped
    p = run_permute(ids, torch.rand(T, k, device="cuda"), E)
    xp = torch.full((p.R, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    moe.ext().act_perm16(x, p.pos_of, k, xp)
    n = int(p.off[-1])
    phys = a16_phys_index(H, "cuda")
    want = x[p.tok[:n].long()][:, phys]
    assert torch.equal(xp[:n].view(torch.int16), want.view(torch.int16))  # bit-exact copy (incl. -0)
    assert torch.isnan(xp[n:].float()).all()  # rows of dropped ids untouched


def _stock_swiglu(gu: torch.Tensor) -> torch.Tensor:
    """vLLM humming _CLAMPED_SILU_IMPL (alpha 1, beta 0) in fp32 on fp32 gate/up."""
    g, u = gu.float().split(I_R, dim=-1)
    g = g.clamp(max=LIMIT)
    u = u.clamp(-LIMIT, LIMIT)
    return g / (1.0 + torch.exp(-g)) * u


@gpu
@pytest.mark.parametrize("T,variant", [(1, 9), (37, 9), (37, 10), (512, 9), (512, 10), (3000, 9), (3000, 10), (8192, 9),
                                       (1, 11), (37, 11), (200, 11), (512, 11), (3000, 11), (8192, 11),
                                       (1, 12), (37, 12), (512, 12), (3000, 12), (8192, 12),
                                       (1, 16), (37, 16), (128, 16), (512, 16), (3000, 16), (1, 17), (37, 17), (128, 17), (700, 17)])
def test_gemm1_w4a16(T, variant):
    """W4A16 GEMM1 + stock SwiGLU on bf16 activations: h = bf16(swiglu(x @ W13^T)) with fp32 accumulation --
    every element within 1 bf16 ulp of the fp32 reference (the only differences: fp32 summation order)."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    W13, _ = dequant_layer()
    x = (torch.randn(T, H, device="cuda") * (0.5 + torch.rand(T, 1, device="cuda"))).to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    p = run_permute(ids, tw, E)
    xp = torch.empty(p.R, H, dtype=torch.bfloat16, device="cuda")
    moe.ext().act_perm16(x, p.pos_of, TOPK, xp)
    hbuf = torch.full((p.R, I_R), float("nan"), dtype=torch.bfloat16, device="cuda")
    moe.ext().moe_gemm(1, xp, p.perm_w, w13, s13, p.meta, p.perm_w, p.R, variant, LIMIT, hbuf)
    n = int(p.off[-1])
    rows_e = expert_of_rows(p)
    tok = p.tok[:n].long()
    gu = torch.empty(n, 2 * I_R, device="cuda")

    def gemm1_ref(e, r):
        gu[r] = x.float()[tok[r]] @ W13[e].float().t()

    per_expert(rows_e, gemm1_ref)
    h_ref = _stock_swiglu(gu)[:, a16_phys_index(I_R, "cuda")]  # h is stored in the logical K order of GEMM2
    h = hbuf[:n]
    print(f"GEMM1 W4A16 T={T} variant={variant}: rel_err {rel_err(h, h_ref):.2e}")
    bf16_close(h, h_ref, f"GEMM1 W4A16 T={T} v{variant}")
    hbuf2 = torch.full_like(hbuf, float("nan"))
    moe.ext().moe_gemm(1, xp, p.perm_w, w13, s13, p.meta, p.perm_w, p.R, variant, LIMIT, hbuf2)
    assert torch.equal(hbuf2[:n], hbuf[:n])  # repeatable, bit-deterministic


@gpu
@pytest.mark.parametrize("T,variant", [(1, 11), (20, 11), (100, 12), (700, 11), (3000, 11), (3000, 12), (8192, 11),
                                       (1, 13), (20, 13), (100, 13), (700, 13), (3000, 13), (8192, 13),
                                       (1, 14), (100, 14), (3000, 14), (8192, 14), (20, 15), (700, 15), (8192, 15),
                                       (1, 16), (20, 16), (128, 16), (700, 16), (3000, 16), (1, 17), (20, 17), (128, 17), (700, 17)])
def test_gemm2_w4a16(T, variant):
    """W4A16 GEMM2: y = bf16(h @ W2^T) per routed row (unscaled, slot order), within 1 bf16 ulp of fp32."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    _, W2 = dequant_layer()
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    p = run_permute(ids, tw, E)
    n = int(p.off[-1])
    h_phys = (torch.randn(p.R, I_R, device="cuda") * torch.rand(p.R, 1, device="cuda")).to(torch.bfloat16)
    h_log = h_phys[:, a16_phys_index(I_R, "cuda")].contiguous()  # GEMM1 writes h in the logical order
    slot = torch.full((p.R, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    moe.ext().moe_gemm(2, h_log, p.perm_w, w2, s2, p.meta, p.perm_w, p.R, variant, LIMIT, slot)
    rows_e = expert_of_rows(p)
    y_ref = torch.empty(n, H, device="cuda")

    def gemm2_ref(e, r):
        y_ref[r] = h_phys[:n][r].float() @ W2[e].float().t()

    per_expert(rows_e, gemm2_ref)
    y = slot[p.slot[:n].long()]
    print(f"GEMM2 W4A16 T={T} variant={variant}: rel_err {rel_err(y, y_ref):.2e}")
    bf16_close(y, y_ref, f"GEMM2 W4A16 T={T} v{variant}")
    slot2 = torch.full_like(slot, float("nan"))
    moe.ext().moe_gemm(2, h_log, p.perm_w, w2, s2, p.meta, p.perm_w, p.R, variant, LIMIT, slot2)
    rows = p.slot[:n].long()
    assert torch.equal(slot2[rows], slot[rows])


def stock_w4a16_emulation(x, tw, ids, W13, W2, round_gu: bool = True):
    """Stock vLLM 0.32.0 W4A16 numerics in torch (fp32 GEMMs on exact weights): gate_up rounded to bf16
    (round_gu; Humming c_dtype), clamped SiLU in fp32, h bf16, y = bf16(h @ W2^T) per routed row,
    out = bf16(sum_k topk_w * y) in fp32."""
    T = x.shape[0]
    acc = torch.zeros(T, H, device=x.device)
    for e in torch.unique(ids).tolist():
        if e < 0 or e >= W13.shape[0]:
            continue
        t, kk = torch.nonzero(ids == e, as_tuple=True)
        gu = x.float()[t] @ W13[e].float().t()
        if round_gu:
            gu = gu.to(torch.bfloat16).float()
        h = _stock_swiglu(gu).to(torch.bfloat16)
        y = (h.float() @ W2[e].float().t()).to(torch.bfloat16)
        acc.index_add_(0, t, y.float() * tw[t, kk, None].float())
    return acc.to(torch.bfloat16)


@gpu
@pytest.mark.parametrize("T,variant", [(1, 0), (5, 0), (128, 0), (1000, 0xCA), (4096, 0), (5, 0xDB), (1000, 0xDB), (4096, 0xDB), (5, 0xEC),
                                       (4096, 0xEC), (1000, 0xFC), (5, 0x100), (128, 0x100), (1000, 0x100)])
def test_forward_w4a16_matches_stock_numerics(T, variant):
    """W4A16 forward vs a torch emulation of stock W4A16 numerics: the only differences are fp32 summation
    order and the SwiGLU input (fp32 accumulator here, bf16-rounded gate/up in stock) -> the difference to
    stock must be of the size of stock's own bf16 rounding noise, and far below the W4A8 error."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    W13, W2 = dequant_layer()
    torch.manual_seed(T)
    x = (torch.randn(T, H, device="cuda") * (0.5 + torch.rand(T, 1, device="cuda"))).to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    y = moe.moe_forward_cuda(x, tw, ids, w13, s13, w2, s2, LIMIT, variant=variant)
    y_same = stock_w4a16_emulation(x, tw, ids, W13, W2, round_gu=False)  # this kernel's rounding points
    y_stock = stock_w4a16_emulation(x, tw, ids, W13, W2, round_gu=True)
    y_exact = torch.zeros(T, H, device="cuda")
    for e in torch.unique(ids).tolist():
        t, kk = torch.nonzero(ids == e, as_tuple=True)
        hh = _stock_swiglu(x.float()[t] @ W13[e].float().t()) * tw[t, kk, None]
        y_exact.index_add_(0, t, hh @ W2[e].float().t())
    e_same, e_stock, e_exact = rel_err(y, y_same), rel_err(y, y_stock), rel_err(y, y_exact)
    e_stock_exact = rel_err(y_stock, y_exact)
    print(f"W4A16 T={T}: rel vs same-rounding emulation {e_same:.2e}, vs stock emulation {e_stock:.2e}, vs fp32 exact {e_exact:.2e} "
          f"(stock emulation vs fp32 exact {e_stock_exact:.2e})")
    assert e_same < 2e-3, e_same  # bf16 rounding flips only (summation order)
    assert e_exact <= 1.2 * e_stock_exact + 1e-4, (e_exact, e_stock_exact)  # at least as precise as stock


@gpu
def test_forward_w4a16_ids_dropped_and_tiny():
    """Ids outside [0, E) contribute nothing; T=1 / all-dropped rows give zeros; matches the ref impl."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    W13, W2 = dequant_layer()
    T = 9
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    ids[0, :] = -1
    ids[1, 2] = E
    ids[2, 0] = -5
    y = moe.moe_forward(x, tw, ids, w13, s13, w2, s2, LIMIT, impl="w4a16")
    assert torch.all(y[0] == 0)
    y_ref = stock_w4a16_emulation(x, tw, ids, W13, W2, round_gu=False)
    assert rel_err(y, y_ref) < 2e-3


@gpu
def test_combine():
    T, k = 333, 6
    ids = torch.randint(0, E, (T, k), device="cuda", dtype=torch.int32)
    ids[0, 1] = -1
    slot = torch.randn(T * k, H, device="cuda").to(torch.bfloat16)
    slot[1] = float("nan")  # invalid slot must be skipped
    out = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
    moe.ext().moe_combine(slot, ids, E, out)
    ref = (slot.float().view(T, k, H) * (ids >= 0)[..., None]).nan_to_num().sum(1)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    sc = torch.rand(T * k, device="cuda") + 0.5  # per-slot scales (GEMM2 output is unscaled)
    sc[1] = float("nan")
    moe.ext().moe_combine(slot, ids, E, out, sc)
    ref = (slot.float().view(T, k, H) * sc.view(T, k, 1) * (ids >= 0)[..., None]).nan_to_num().sum(1)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    # addend (shared-expert output folded in): one fp32 sum, one rounding
    add = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    moe.ext().moe_combine(slot, ids, E, out, sc, add)
    torch.testing.assert_close(out.float(), ref + add.float(), rtol=1e-2, atol=1e-2)


@gpu
@pytest.mark.parametrize("T", [5, 300, 2000, 9000])
def test_forward_hybrid(T):
    """Hybrid remainder tiles (set_hybrid(128): each expert's <= 128-row remainder on moe_gemm16_kernel) == the plain
    0xB9 forward within 1 bf16 ulp (same exact products, fp32 accumulation), and bit-deterministic."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    e = moe.ext()
    e.set_hybrid(0)
    y0 = moe.ext().moe_forward(x, tw.float(), ids.int(), w13, s13, w2, s2, LIMIT, 0)
    try:
        e.set_hybrid(128)
        y1 = e.moe_forward(x, tw.float(), ids.int(), w13, s13, w2, s2, LIMIT, 0)
        y2 = e.moe_forward(x, tw.float(), ids.int(), w13, s13, w2, s2, LIMIT, 0)
    finally:
        e.set_hybrid(0)
        moe._HYBRID_SET[0] = None
    assert torch.equal(y1, y2)
    d = (y1.float() - y0.float()).abs()
    ulp = y0.float().abs().clamp_min(1e-6) * 2.0**-7
    assert torch.all(d <= ulp * 1.01 + 1e-6), float((d / ulp).max())


@gpu
@pytest.mark.parametrize("T", [5, 128, 300, 2000, 9000])
def test_forward_small16(T):
    """Small-batch kernel (0x100: moe_small16_kernel for both GEMMs, mma.sync) and the hybrid with the small16 remainder
    engine (set_hybrid(128, 1)) == the plain 0xB9 forward within 1 bf16 ulp (same exact products, fp32 accumulation),
    bit-deterministic."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    e = moe.ext()
    e.set_hybrid(0, 0)
    y0 = e.moe_forward(x, tw.float(), ids.int(), w13, s13, w2, s2, LIMIT, 0)
    ys = [e.moe_forward(x, tw.float(), ids.int(), w13, s13, w2, s2, LIMIT, 0x100) for _ in range(2)]  # T <= 256: 16-token units
    yf = [e.moe_forward(x, tw.float(), ids.int(), w13, s13, w2, s2, LIMIT, 0x300) for _ in range(2)]  # 64-token units
    try:
        e.set_hybrid(128, 1)
        yh = [e.moe_forward(x, tw.float(), ids.int(), w13, s13, w2, s2, LIMIT, 0) for _ in range(2)]
    finally:
        e.set_hybrid(0, 0)
        moe._HYBRID_SET[0] = None
    for name, (y1, y2) in (("small16", ys), ("small16 NT64", yf), ("hybrid small16", yh)):
        assert torch.equal(y1, y2), name
        d = (y1.float() - y0.float()).abs()
        ulp = y0.float().abs().clamp_min(1e-6) * 2.0**-7
        assert torch.all(d <= ulp * 1.01 + 1e-6), (name, float((d / ulp).max()))


@gpu
@pytest.mark.parametrize("T,variant", [(5, 0), (700, 0), (4096, 0), (700, 0xDB)])
def test_forward_addend(T, variant):
    """moe_forward(addend=s) == bf16(fp32 routed sum + s): within one bf16 rounding of bf16(routed) + s (stock order)."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    add = (torch.randn(T, H, device="cuda") * 0.05).to(torch.bfloat16)
    y = moe.moe_forward_cuda(x, tw, ids, w13, s13, w2, s2, LIMIT, variant)
    ya = moe.moe_forward_cuda(x, tw, ids, w13, s13, w2, s2, LIMIT, variant, addend=add)
    two = (y.float() + add.float()).to(torch.bfloat16).float()
    d = (ya.float() - two).abs()
    ulp = torch.maximum(torch.maximum(two.abs(), y.float().abs()), add.float().abs()).clamp_min(1e-6) * 2.0**-7
    assert torch.all(d <= ulp * 1.51 + 1e-6), float((d / ulp).max())


@gpu
def test_act_log16_exact():
    """act_log16(x) == act_perm16(x, pos_of = arange(T), K = 1): same rows, W4A16 logical K order, bit-exact."""
    T = 333
    x = (torch.randn(T, H, device="cuda") * 3).to(torch.bfloat16)
    xl = torch.full((T, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    moe.ext().act_log16(x, xl)
    want = x[:, a16_phys_index(H, "cuda")]
    assert torch.equal(xl.view(torch.int16), want.view(torch.int16))


@gpu
@pytest.mark.parametrize("T", [1, 5, 37, 300, 2000, 4096, 9000])
def test_forward_xgather(T):
    """x-gather GEMM1 (13 / 14: B rows gathered from act_log16(x) by the loader warp) == the routed-row-copy GEMM1
    (9 / 10) BIT FOR BIT (same products, same wgmma order), and bit-deterministic."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    e = moe.ext()
    for vg, vr in ((0xBD, 0xB9), (0xBE, 0xBA)):
        y0 = e.moe_forward(x, tw.float(), ids.int(), w13, s13, w2, s2, LIMIT, vr)
        y1 = e.moe_forward(x, tw.float(), ids.int(), w13, s13, w2, s2, LIMIT, vg)
        y2 = e.moe_forward(x, tw.float(), ids.int(), w13, s13, w2, s2, LIMIT, vg)
        assert torch.equal(y1, y2), hex(vg)
        assert torch.equal(y0.view(torch.int16), y1.view(torch.int16)), (hex(vg), float((y0.float() - y1.float()).abs().max()))


@gpu
def test_h_quant_slot_order():
    R, n = 300, 251
    h = (torch.randn(R, I_R, device="cuda") * torch.rand(R, 1, device="cuda")).to(torch.bfloat16)
    idx = torch.randperm(R, device="cuda").int()
    hq = torch.empty(R, I_R, dtype=torch.uint8, device="cuda")
    hs = torch.full((R,), -1.0, device="cuda")
    moe.ext().h_quant(h, torch.tensor([n], dtype=torch.int32, device="cuda"), hq, hs, idx)
    amax = h.float().abs().amax(-1)
    torch.testing.assert_close(hs[idx[:n].long()], amax[:n] / 448.0)
    assert torch.all(hs[idx[n:].long()] == -1.0)


# ============================================================================ GPU: end to end
@gpu
@pytest.mark.parametrize("T", [1, 128, 4096])
def test_forward_vs_reference(T):
    """W4A8 routed-expert forward vs fp32 exact-weight reference (ds41h.ref dequant), one layer / rank.

    Expected error: e4m3 rounding (<= 2^-4 relative, RMS ~3.6%) of the GEMM1 input and of h gives
    ~5% relative L2 error on random (incoherent) activations -- the same per-element precision as the
    reference model's own FP8 activation quantization (inference/kernel.py act_quant)."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    W13, W2 = dequant_layer()
    torch.manual_seed(T)
    x = (torch.randn(T, H, device="cuda") * (0.5 + torch.rand(T, 1, device="cuda"))).to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    y = moe.moe_forward(x, tw, ids, w13, s13, w2, s2, LIMIT, impl="w4a8")
    y_ref = torch.zeros(T, H, device="cuda")
    for e in torch.unique(ids).tolist():
        t, kk = torch.nonzero(ids == e, as_tuple=True)
        gu = x.float()[t] @ W13[e].float().t()
        g, u = gu.split(I_R, -1)
        hh = F.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT) * tw[t, kk, None]
        y_ref.index_add_(0, t, hh @ W2[e].float().t())
    err = rel_err(y, y_ref)
    cos = F.cosine_similarity(y.float().flatten(), y_ref.flatten(), dim=0).item()
    print(f"T={T}: rel_err={err:.4f} cos={cos:.6f}")
    assert err < 0.08 and cos > 0.996
    # the torch reference path of ds41h.moe agrees too (bf16 compute = stock W4A16 numerics)
    if T <= 128:
        y_t = moe.moe_forward(x, tw, ids, w13, s13, w2, s2, LIMIT, impl="ref")
        assert rel_err(y_t, y_ref) < 0.01


@gpu
def test_forward_against_ref_package():
    """Optional cross-check against ds41h.ref.layer.moe_ref (routed part, rank-local) if available."""
    try:
        from ds41h.ref import layer as ref_layer
    except ImportError:
        pytest.skip("ds41h.ref.layer not available yet")
    fn = getattr(ref_layer, "routed_experts_ref", None)
    if fn is None:
        pytest.skip("ds41h.ref.layer.routed_experts_ref not available (moe_ref covers the full MoE incl. shared expert)")
    w13, s13, w2, s2, gate, bias = raw_layer()
    x = torch.randn(64, H, device="cuda").to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    y = moe.moe_forward(x, tw, ids, w13, s13, w2, s2, LIMIT)
    y_ref = fn(x, tw, ids, w13, s13, w2, s2, LIMIT)
    assert rel_err(y, y_ref) < 0.08


@gpu
@pytest.mark.parametrize("variant,T", [(0x75, 96), (0x11, 96), (0x95, 96), (0x95, 5), (0, 96), (0x95, 600), (0xB9, 5), (0xCA, 96),
                                       (0xC9, 600), (0xBA, 96), (0, 2000), (0xDB, 5), (0xDB, 96), (0xDB, 600), (0xDB, 2000), (0xBB, 96),
                                       (0xD9, 96), (0xEC, 5), (0xEC, 96), (0xEC, 600), (0xEC, 2000), (0xFC, 96), (0xFC, 2000),
                                       (0xBD, 5), (0xBD, 600), (0xBD, 2000), (0xBE, 96), (0x100, 5), (0x100, 96), (0x100, 600), (0x300, 96)])
def test_forward_cuda_graph_and_determinism(variant, T):
    """CUDA-graph capture/replay with new inputs equals an eager forward. Variants without the bf16
    reduction epilogue (GEMM2 slot rows + moe_combine; all W4A16 variants, 0 = W4A16 default) are
    bit-deterministic; the W4A8 reduction variants (v2 = 5 / 9 / 10) accumulate in bf16 in an arbitrary
    order, so they are compared with a tolerance of a few bf16 ulps of the output magnitude."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    fwd = lambda a, b, c: moe.moe_forward_cuda(a, b, c, w13, s13, w2, s2, LIMIT, variant=variant)
    exact = moe.is_w4a16_variant(variant) or (variant >> 4) not in (5, 9, 10)

    def same(a, b):
        if exact:
            return torch.equal(a, b)
        return bool(torch.isfinite(a).all()) and (a.float() - b.float()).abs().max().item() <= 0.03 * b.float().abs().max().item() + 1e-3

    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    y0 = fwd(x, tw, ids)
    y1 = fwd(x, tw, ids)
    assert same(y0, y1)
    sx, sw, si = x.clone(), tw.clone(), ids.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fwd(sx, sw, si)  # warm up allocations
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sy = fwd(sx, sw, si)
    for seed in range(3):  # several replays with new routing
        x2 = torch.randn(T, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed)).to(torch.bfloat16)
        _, tw2, ids2 = stock_route(x2, gate, bias)
        sx.copy_(x2), sw.copy_(tw2), si.copy_(ids2)
        graph.replay()
        torch.cuda.synchronize()
        assert same(sy, fwd(x2, tw2, ids2))

# ===================================================================================== determinism stress
_STRESS_CFG = {  # name -> (variant passed to moe_forward, hybrid rows, hybrid engine): the DEFAULT prefill MoE paths
    "0x300": (0x300, 0, 0),  # small16, 64-token units (default for <= 512 tokens since 02:30)
    "0x100": (0x100, 0, 0),  # small16, auto 16-token units (gate candidate)
    "0x500": (0x500, 0, 0),  # small16, 16-token units at every size (NT64 races under load)
    "0xd00": (0xD00, 0, 0),  # small16k: 0x500 units, 2 K blocks / stage (64-byte SW64 weight boxes), 3 stages, 4 CTAs / SM
    "0x1500": (0x1500, 0, 0),  # small16k: 2 K blocks / stage, 4 stages, 3 CTAs / SM
    "0x1d00": (0x1D00, 0, 0),  # small16k: 4 K blocks / stage (128-byte SW128 weight boxes), 3 stages, 2 CTAs / SM
    "0x2500": (0x2500, 0, 0),  # small16k: 4 K blocks / stage, 2 stages, 3 CTAs / SM
    "0x4d00": (0x4D00, 0, 0),  # small16k 0xd00 + chunk-fastest unit order (chunks of one expert block on neighbouring CTAs)
    "0x5d00": (0x5D00, 0, 0),  # small16k 0x1d00 + chunk-fastest unit order
    "0x2d00": (0x2D00, 0, 0),  # small16k: 32-token units, 1 K block / stage (32-byte boxes), 6 stages, 3 CTAs / SM
    "0x3500": (0x3500, 0, 0),  # small16k: 32-token units, 2 K blocks / stage, 3 stages, 3 CTAs / SM
    "0x7500": (0x7500, 0, 0),  # small16k 0x3500 + chunk-fastest unit order
    "hyb128": (0, 128, 0),  # 0xB9 + each expert's <= 128-row remainder on moe_gemm16_kernel (default for > 512 tokens)
    "hyb128e1": (0, 128, 1),  # remainder on moe_small16_kernel
    "b9": (0, 0, 0),  # plain 0xB9 (pre-02:30 default)
}


def _side_load(n: int = 3):
    """Queue a few big bf16 GEMMs + a copy on a side stream (SM / HBM contention while the tested kernels run)."""
    s = _side_load.__dict__.setdefault("s", torch.cuda.Stream())
    a = _side_load.__dict__.setdefault("a", torch.randn(6144, 6144, device="cuda").to(torch.bfloat16))
    b = _side_load.__dict__.setdefault("b", torch.empty(64 << 20, dtype=torch.uint8, device="cuda"))
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n):
            torch.mm(a, a)
            b.copy_(b.flip(0))
    return s


@gpu
@pytest.mark.parametrize("T", [38, 65, 128, 300, 1000, 2500])
@pytest.mark.parametrize("cfg", list(_STRESS_CFG))
def test_bit_repeatable_stress(cfg, T):
    """Bit-repeatability of the default prefill MoE paths : identical inputs N times ->
    bit-identical outputs (a) back to back, (b) under a concurrent side-stream GEMM / copy load, (c) interleaved with
    MoE calls of other sizes (stale state / counter reset), (d) CUDA-graph replays interleaved with eager calls."""
    N = int(os.environ.get("DS41H_STRESS_N", "20"))
    w13, s13, w2, s2, gate, bias = raw_layer()
    v, hyb, eng = _STRESS_CFG[cfg]
    e = moe.ext()
    g = torch.Generator(device="cuda").manual_seed(1000 + T)
    x = (torch.randn(T, H, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    tw, ids = tw.float().contiguous(), ids.int().contiguous()
    x2 = (torch.randn(3 * T + 17, H, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    _, tw2, ids2 = stock_route(x2, gate, bias)
    tw2, ids2 = tw2.float().contiguous(), ids2.int().contiguous()
    fwd = lambda a, b, c: e.moe_forward(a, b, c, w13, s13, w2, s2, LIMIT, v)
    bad = {}
    try:
        e.set_hybrid(hyb, eng)
        y0 = fwd(x, tw, ids).clone()
        for mode in ("plain", "side", "interleaved"):
            n = 0
            for i in range(N):
                if mode == "side":
                    s = _side_load()
                if mode == "interleaved":
                    fwd(x2, tw2, ids2)
                    fwd(x[: max(1, T // 3)], tw[: max(1, T // 3)], ids[: max(1, T // 3)])
                y = fwd(x, tw, ids)
                if mode == "side":
                    torch.cuda.current_stream().wait_stream(s)
                n += not torch.equal(y, y0)
            bad[mode] = n
        sx, sw, si = x.clone(), tw.clone(), ids.clone()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            fwd(sx, sw, si)
        torch.cuda.current_stream().wait_stream(st)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sy = fwd(sx, sw, si)
        n = 0
        for i in range(N):
            fwd(x2, tw2, ids2)
            graph.replay()
            n += not torch.equal(sy, y0)
        bad["graph"] = n
    finally:
        e.set_hybrid(0, 0)
        moe._HYBRID_SET[0] = None
    torch.cuda.synchronize()
    assert all(n == 0 for n in bad.values()), (cfg, T, bad)


@gpu
@pytest.mark.parametrize("T", [38, 65, 300])
def test_small_paths_bit_identical_to_b9(T):
    """The small-T / hybrid paths vs the plain 0xB9 forward on real routing: bit-identical? (informational: the
    kernels claim the same products and per-row fp32 order; a difference here is summation order, not a bug).
    Reports the number of differing elements per path."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    e = moe.ext()
    g = torch.Generator(device="cuda").manual_seed(7 + T)
    x = (torch.randn(T, H, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    tw, ids = tw.float().contiguous(), ids.int().contiguous()
    res = {}
    try:
        for cfg, (v, hyb, eng) in _STRESS_CFG.items():
            e.set_hybrid(hyb, eng)
            res[cfg] = e.moe_forward(x, tw, ids, w13, s13, w2, s2, LIMIT, v).clone()
    finally:
        e.set_hybrid(0, 0)
        moe._HYBRID_SET[0] = None
    ref = res["b9"]
    diffs = {k: int((y != ref).sum()) for k, y in res.items()}
    print(f"T={T} elements differing from 0xB9: {diffs}")
    for k, y in res.items():
        d = (y.float() - ref.float()).abs()
        ulp = ref.float().abs().clamp_min(1e-6) * 2.0**-7
        assert torch.all(d <= ulp * 1.01 + 1e-6), (k, float((d / ulp).max()))


@gpu
@pytest.mark.parametrize("T", [1, 5, 16, 38, 65, 128, 300, 512, 1000, 2500])
def test_small16k_bit_identical_to_0x500(T):
    """small16k (2 / 4 K blocks per pipeline stage with 64- / 128-byte swizzled TMA weight boxes, 32-token
    units, chunk-fastest unit order) must
    reproduce small16 0x500 BIT FOR BIT on real routing (same products, same per-K-block fp32 order), including GEMM2's K
    tail (K = 576 = 9 blocks: the last stage holds one block), single-box token tiles (T < 16) and the chunk-fastest unit
    order (+0x4000; experts with > 16 rows at T >= 300)."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    e = moe.ext()
    e.set_hybrid(0, 0)
    g = torch.Generator(device="cuda").manual_seed(31 + T)
    x = (torch.randn(T, H, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    tw, ids = tw.float().contiguous(), ids.int().contiguous()
    ref = e.moe_forward(x, tw, ids, w13, s13, w2, s2, LIMIT, 0x500)
    bad = {}
    for v in (0xD00, 0x1500, 0x1D00, 0x2500, 0x2D00, 0x3500, 0x4D00, 0x5500, 0x5D00, 0x6500, 0x6D00, 0x7500):
        for _ in range(2):
            y = e.moe_forward(x, tw, ids, w13, s13, w2, s2, LIMIT, v)
            n = int((y.view(torch.int16) != ref.view(torch.int16)).sum())
            if n:
                bad[hex(v)] = n
    torch.cuda.synchronize()
    assert not bad, (T, bad)


@gpu
def test_method_apply_matches_forward():
    """The vLLM quant method's apply() runs the same path (with RoutedExperts state)."""
    w13, s13, w2, s2, gate, bias = raw_layer()
    method = moe.make_raw_mxfp4_moe_method(SimpleNamespace(w13_num_shards=2, has_bias=False))
    # the per-layer impl is chosen at load time by the vLLM glue (ds41h_moe_impl); pin the CUDA path here
    layer = SimpleNamespace(w13_weight=w13, w13_weight_scale=s13, w2_weight=w2, w2_weight_scale=s2, swiglu_limit=LIMIT, ds41h_moe_impl="cuda")
    x = torch.randn(40, H, device="cuda").to(torch.bfloat16)
    _, tw, ids = stock_route(x, gate, bias)
    y_apply = method.apply(layer, x, tw, ids, None, None)
    y_fwd = moe.moe_forward(x, tw, ids, w13, s13, w2, s2, LIMIT, impl="cuda")
    if not moe.is_w4a16_variant(moe.default_variant()) and (moe.default_variant() >> 4) in (5, 9, 10):  # W4A8 bf16 reductions
        assert (y_apply.float() - y_fwd.float()).abs().max().item() <= 0.03 * y_fwd.float().abs().max().item() + 1e-3
    else:
        torch.testing.assert_close(y_apply, y_fwd, rtol=0, atol=0)


# ============================================================================ benchmark
def _time(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e-3


def tile_stats(counts: torch.Tensor, bnmax: int) -> tuple[int, float]:
    """(token tiles, fraction of computed token columns that are padding) for 16-granular tiles <= bnmax."""
    c = counts.long()
    tiles = int(((c + bnmax - 1) // bnmax).sum())
    full = c // bnmax
    rem = c - full * bnmax
    computed = int((full * bnmax + (rem + 15) // 16 * 16).sum())
    return tiles, 1.0 - int(c.sum()) / max(computed, 1)


def bench_w4a16(tokens: int = 8192, layer: int = LAYER, rank: int = RANK, uniform: bool = False, variant: int = 0) -> dict:
    """Stage timings of the W4A16 path (act_perm16, GEMM1, GEMM2 -> slot, weighted combine) and the forward."""
    w13, s13, w2, s2, gate, bias = raw_layer(layer, rank)
    n_exp, I = w13.shape[0], w13.shape[1] // 2
    x, tw, ids = bench_routing(tokens, layer, rank, uniform)
    R = tokens * TOPK
    v = variant or moe.W4A16_VARIANT
    v1, v2 = v & 15, (v >> 4) & 15
    p = run_permute(ids, tw, n_exp)
    xp = torch.empty(R, H, dtype=torch.bfloat16, device="cuda")
    h = torch.empty(R, I, dtype=torch.bfloat16, device="cuda")
    slot = torch.empty(R, H, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda")
    ext = moe.ext()
    twf = tw.float().contiguous().view(-1)
    t = {}
    t["permute"] = _time(lambda: ext.moe_permute(ids, tw, n_exp, p.meta, p.perm_w))
    t["act_perm16"] = _time(lambda: ext.act_perm16(x, p.pos_of, TOPK, xp))
    t["GEMM1"] = _time(lambda: ext.moe_gemm(1, xp, p.perm_w, w13, s13, p.meta, p.perm_w, R, v1, LIMIT, h))
    t["GEMM2"] = _time(lambda: ext.moe_gemm(2, h, p.perm_w, w2, s2, p.meta, p.perm_w, R, v2, LIMIT, slot))
    t["combine"] = _time(lambda: ext.moe_combine(slot, ids, n_exp, out, twf))
    t["forward"] = _time(lambda: moe.moe_forward_cuda(x, tw, ids, w13, s13, w2, s2, LIMIT, variant=v))
    f1, f2 = 2.0 * R * (2 * I) * H, 2.0 * R * H * I
    print(f"layer {layer} rank {rank}: T={tokens} W4A16 variant=0x{v:02x}")
    print(f"  permute    {t['permute'] * 1e6:8.1f} us")
    print(f"  act_perm16 {t['act_perm16'] * 1e6:8.1f} us  {(x.numel() * 2 * (1 + TOPK)) / t['act_perm16'] / 1e9:7.1f} GB/s")
    print(f"  GEMM1      {t['GEMM1'] * 1e6:8.1f} us  {f1 / t['GEMM1'] / 1e12:7.1f} TFLOPS")
    print(f"  GEMM2      {t['GEMM2'] * 1e6:8.1f} us  {f2 / t['GEMM2'] / 1e12:7.1f} TFLOPS")
    print(f"  combine    {t['combine'] * 1e6:8.1f} us  {(R * H * 2 + tokens * H * 2) / t['combine'] / 1e9:7.1f} GB/s")
    glue = t["permute"] + t["act_perm16"] + t["combine"]
    print(f"  glue       {glue * 1e6:8.1f} us   kernel sum {(glue + t['GEMM1'] + t['GEMM2']) * 1e6:8.1f} us")
    print(f"  forward    {t['forward'] * 1e6:8.1f} us  {(f1 + f2) / t['forward'] / 1e12:7.1f} TFLOPS end-to-end, {tokens / t['forward']:,.0f} tok/s/layer")
    return t


def bench(tokens: int = 8192, layer: int = LAYER, rank: int = RANK, uniform: bool = False, variant: int = 0) -> dict:
    if moe.is_w4a16_variant(variant):
        return bench_w4a16(tokens, layer, rank, uniform, variant)
    w13, s13, w2, s2, gate, bias = raw_layer(layer, rank)
    n_exp, I = w13.shape[0], w13.shape[1] // 2
    x, tw, ids = bench_routing(tokens, layer, rank, uniform)
    R = tokens * TOPK
    v1, v2 = variant & 15, (variant >> 4) & 15
    p = run_permute(ids, tw, n_exp)
    xq = torch.empty(tokens, H, dtype=torch.uint8, device="cuda")
    xs = torch.empty(tokens, dtype=torch.float32, device="cuda")
    h = torch.empty(R, I, dtype=torch.bfloat16, device="cuda")
    hq = torch.empty(R, I, dtype=torch.uint8, device="cuda")
    hs = torch.empty(R, dtype=torch.float32, device="cuda")
    slot = torch.empty(R, H, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda")
    ext = moe.ext()
    t = {}
    t["permute"] = _time(lambda: ext.moe_permute(ids, tw, n_exp, p.meta, p.perm_w))
    t["act_quant"] = _time(lambda: ext.act_quant(x, xq, xs))
    t["GEMM1"] = _time(lambda: ext.moe_gemm(1, xq, xs, w13, s13, p.meta, p.perm_w, R, v1, LIMIT, h))
    t["h_quant"] = _time(lambda: ext.h_quant(h, p.counts[n_exp:], hq, hs, p.slot))
    t["GEMM2"] = _time(lambda: ext.moe_gemm(2, hq, hs, w2, s2, p.meta, p.perm_w, R, v2, LIMIT, slot))
    t["combine"] = _time(lambda: ext.moe_combine(slot, ids, n_exp, out, hs))
    t["forward"] = _time(lambda: moe.moe_forward_w4a8(x, tw, ids, w13, s13, w2, s2, LIMIT, variant=variant))
    cnt = p.counts[:n_exp]
    f1, f2 = 2.0 * R * (2 * I) * H, 2.0 * R * H * I
    wbytes = w13.numel() + s13.numel() + w2.numel() + s2.numel()
    tiles256, pad256 = tile_stats(cnt, 256)
    tiles128, pad128 = tile_stats(cnt, 128)
    cf = cnt.float()
    print(f"layer {layer} rank {rank}: T={tokens} topk={TOPK} E={n_exp} routing={'uniform' if uniform else 'gate'} variant={variant} "
          f"rows/expert mean={cf.mean():.1f} max={cf.max():.0f} min={cf.min():.0f} empty={int((cnt == 0).sum())}; token tiles "
          f"{tiles256} (BN<=256, {100 * pad256:.1f}% pad) / {tiles128} (BN<=128, {100 * pad128:.1f}% pad)")
    print(f"  permute   {t['permute'] * 1e6:8.1f} us")
    print(f"  act_quant {t['act_quant'] * 1e6:8.1f} us  {(x.numel() * 3) / t['act_quant'] / 1e9:7.1f} GB/s")
    print(f"  GEMM1     {t['GEMM1'] * 1e6:8.1f} us  {f1 / t['GEMM1'] / 1e12:7.1f} TFLOPS")
    print(f"  h_quant   {t['h_quant'] * 1e6:8.1f} us  {(R * I * 3) / t['h_quant'] / 1e9:7.1f} GB/s")
    print(f"  GEMM2     {t['GEMM2'] * 1e6:8.1f} us  {f2 / t['GEMM2'] / 1e12:7.1f} TFLOPS")
    print(f"  combine   {t['combine'] * 1e6:8.1f} us  {(R * H * 2 + tokens * H * 2) / t['combine'] / 1e9:7.1f} GB/s")
    glue = t["permute"] + t["act_quant"] + t["h_quant"] + t["combine"]
    print(f"  glue      {glue * 1e6:8.1f} us   kernel sum {(glue + t['GEMM1'] + t['GEMM2']) * 1e6:8.1f} us")
    print(f"  forward   {t['forward'] * 1e6:8.1f} us  {(f1 + f2) / t['forward'] / 1e12:7.1f} TFLOPS end-to-end, {tokens / t['forward']:,.0f} tok/s/layer; weights {wbytes / 1e9:.2f} GB")
    return t


def accuracy(tokens: int = 2048, layer: int = LAYER, rank: int = RANK, variants=(0xB9, 0xDB, 0x95, 0x75), seed: int = 1) -> None:
    """Forward error of each GEMM variant vs the fp32 exact-weight reference on gate-routed random x."""
    w13, s13, w2, s2, gate, bias = raw_layer(layer, rank)
    W13, W2 = dequant_layer(layer, rank)
    x, tw, ids = bench_routing(tokens, layer, rank, seed=seed)
    y_ref = torch.zeros(tokens, H, device="cuda")
    for e in torch.unique(ids).tolist():
        t, kk = torch.nonzero(ids == e, as_tuple=True)
        gu = x.float()[t] @ W13[e].float().t()
        g, u = gu.split(I_R, -1)
        hh = F.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT) * tw[t, kk, None]
        y_ref.index_add_(0, t, hh @ W2[e].float().t())
    y_bf = moe.moe_forward(x, tw, ids, w13, s13, w2, s2, LIMIT, impl="ref").float()
    print(f"accuracy layer {layer} T={tokens}: torch bf16 W4A16 (stock numerics) rel_err {rel_err(y_bf, y_ref):.4f}")
    y_st = stock_w4a16_emulation(x, tw, ids, W13, W2, round_gu=True).float()
    print(f"  stock W4A16 emulation (bf16 gate/up, bf16 h, bf16 y, fp32 combine): rel_err {rel_err(y_st, y_ref):.4f}")
    for v in variants:
        y = moe.moe_forward_cuda(x, tw, ids, w13, s13, w2, s2, LIMIT, variant=v).float()
        bias_ = ((y * y_ref).sum() / (y_ref * y_ref).sum()).item()
        print(f"  variant 0x{v:02x}: rel_err {rel_err(y, y_ref):.4f}  cos {F.cosine_similarity(y.flatten(), y_ref.flatten(), dim=0).item():.6f}"
              f"  gain {bias_:.5f}  vs bf16 {rel_err(y, y_bf):.4f}")


def bench_routing(tokens: int, layer: int = LAYER, rank: int = RANK, uniform: bool = False, seed: int = 0):
    """(x, topk_weights, topk_ids) of the benchmark: gate routing of seeded random x (same for every impl)."""
    w13, s13, w2, s2, gate, bias = raw_layer(layer, rank)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, H, device="cuda", generator=g).to(torch.bfloat16)
    if uniform:
        ids = torch.argsort(torch.rand(tokens, w13.shape[0], device="cuda", generator=g), dim=-1)[:, :TOPK].int()
        tw = torch.full((tokens, TOPK), ROUTE_SCALE / TOPK, device="cuda")
    else:
        _, tw, ids = stock_route(x, gate, bias)
    return x, tw, ids


def bench_humming(tokens: int = 8192, layer: int = LAYER, rank: int = RANK, uniform: bool = False) -> float:
    """Stock vLLM 0.32.0 W4A16 Humming MoE (the default sm90 DeepSeek-V4.1 TP path: HummingIndexedExperts,
    bf16 activations, MXFP4 weights repacked by convert_to_humming_moe_kernel_format) on the same raw
    layer / rank / routing as bench(). Returns seconds per forward (routed experts only, incl. align + sum)."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingIndexedExperts
    from vllm.model_executor.layers.quantization.utils.humming.moe import (
        convert_to_humming_moe_kernel_format,
        get_humming_moe_quant_config,
    )

    w13, s13, w2, s2, gate, bias = raw_layer(layer, rank)
    n_exp, I = w13.shape[0], w13.shape[1] // 2
    lay = torch.nn.Module()
    for name, t in (("w13_weight", w13), ("w13_weight_scale", s13), ("w2_weight", w2), ("w2_weight_scale", s2)):
        lay.register_parameter(name, torch.nn.Parameter(t.clone(), requires_grad=False))
    lay.moe_config = SimpleNamespace(has_bias=False, num_local_experts=n_exp, activation=MoEActivation.SILU,
                                     intermediate_size_per_partition=I, hidden_dim=H)
    lay.params_dtype = torch.bfloat16
    lay.layer_name = f"model.layers.{layer}.ffn.experts"
    convert_to_humming_moe_kernel_format(lay, quant_config={"quant_method": "mxfp4"}, allow_input_schema_fallback=False)
    qc = get_humming_moe_quant_config(lay, gemm1_clamp_limit=LIMIT)
    moe_cfg = SimpleNamespace(
        num_local_experts=n_exp, num_experts=n_exp, device=torch.device("cuda"), experts_per_token=TOPK, in_dtype=torch.bfloat16,
        moe_parallel_config=SimpleNamespace(use_ep=False), swiglu_limit=LIMIT, swiglu_alpha=None, swiglu_beta=None,
        activation_situ_beta=None, activation_situ_linear_beta=None,
    )
    ex = HummingIndexedExperts(moe_cfg, qc)
    ex.get_global_valid_shape_m = lambda topk_ids: topk_ids.numel()  # no vLLM forward context here
    x, tw, ids = bench_routing(tokens, layer, rank, uniform)
    act = MoEActivation.SILU
    ws1_shape, ws2_shape, out_shape = ex._workspace_shapes(tokens, TOPK, act)
    ws1 = torch.empty(ws1_shape, dtype=torch.bfloat16, device="cuda")
    ws2 = torch.empty(ws2_shape, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(out_shape, dtype=torch.bfloat16, device="cuda")

    def run():
        ex.apply(out, x, lay.w13_weight, lay.w2_weight, tw, ids, act, n_exp, None, None, None, ws1, ws2, None, False)
        return out

    run()
    torch.cuda.synchronize()
    t = _time(run)
    y_ref = moe.moe_forward_w4a16(x, tw, ids, w13, s13, w2, s2, LIMIT)
    y8 = moe.moe_forward_w4a8(x, tw, ids, w13, s13, w2, s2, LIMIT)
    f = 2.0 * tokens * TOPK * 3 * I * H
    print(f"layer {layer} rank {rank}: T={tokens} stock Humming W4A16 (indexed): {t * 1e6:8.1f} us  {f / t / 1e12:6.1f} TFLOPS  "
          f"{tokens / t:,.0f} tok/s/layer   rel diff vs ds41h W4A16 {rel_err(y_ref, out):.5f}, vs ds41h W4A8 {rel_err(y8, out):.4f}")
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--humming", action="store_true", help="also time stock Humming W4A16 on the same layer / routing")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--variant", type=lambda v: int(v, 0), action="append", help="GEMM variants v1 + 16 * v2 (repeatable)")
    ap.add_argument("--only", action="store_true", help="only --tokens (skip 1/128/1024)")
    ap.add_argument("--accuracy", action="store_true", help="forward error of the GEMM variants vs fp32")
    args = ap.parse_args()
    if args.accuracy:
        accuracy(min(args.tokens, 4096), args.layer, args.rank)
    if args.bench or args.humming:
        sizes = [args.tokens] if args.only else sorted({1, 128, 1024, args.tokens})
        for t in sizes:
            if args.bench:
                for v in args.variant or [0]:
                    bench(t, args.layer, args.rank, args.uniform, v)
            if args.humming:
                bench_humming(t, args.layer, args.rank, args.uniform)
    elif not args.accuracy:
        sys.exit(pytest.main([__file__, "-x", "-q"]))
