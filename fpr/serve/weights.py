"""Weights of the serving worker: what each rank holds, the wcache entries (built on the CPU), and the
device-side finishing steps that need CUDA (mHC prep_fn, the W8A16 packing of q_b).

Per rank (sequence-parallel layout, docs/FPR.md):
  replicated : attention projections of every layer (bf16 cuBLAS layout of attn_half + the raw fp8 q_b for the fused
               W8A16 kernel), mHC parameters, the router gate, the token embedding
  TP4 slice  : routed experts in the raw ds41h layout (moe_raw.cu), the shared expert slice, the vocab shard of the LM head,
               the DSpark drafter's 3 blocks (drafter.load_block, kernels/head)

Cache keys (tools/wcache.py; the key changes whenever a builder's output changes):
  attn_L{L}_v1                 replicated, ~295 MB          L{L}_r{r}_raw_experts_v1  per rank, ~1.81 GB
  cos_sin_c{c}_v1              replicated, 268 MB, one per compress ratio -- NOT for exactness runs: a CPU-built table
                               differs in bits from the GPU one vLLM uses; worker.py builds it on the GPU
  L{L}_r{r}_dense_v1           per rank, ~22 MB             mhc_L{L}_v1               replicated, small
  embed_v1                     replicated, 1.32 GB          head_r{r}_v1              per rank, 331 MB
  mtp{i}_r{r}_v1               per rank (drafter block i)

Prebuild (CPU only):
  CUDA_VISIBLE_DEVICES= python3 weights.py --prebuild --layers 21 --ranks 0,1,2,3 [--small-all] [--globals] [--drafter]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

PHYS = os.environ.get("PHYS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.environ.get("DS41H_MODEL_DIR", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model"))
N_LAYERS, N_RANKS, VOCAB, D = 40, 4, 129_280, 5120
RATIO = [0, 0] + [2] * 18 + [1] * 20            # compress_ratios[0:40]
REF = os.environ.get("DS41H_REF", os.path.join(PHYS, "analysis/indexer/ds41h_ref_copy"))

# bytes per rank (for the memory plan and the CPU tests; measured layouts, see the builders)
EXPERT_BYTES_PER_LAYER = 384 * (2 * 576 * 5120 // 2 + 5120 * 576 // 2) + 384 * (2 * 576 * 5120 // 32 + 5120 * 576 // 32)
ATTN_BYTES_PER_LAYER = 2 * (1792 * 5120 + 32768 * 1280 + 8 * 4096 * 1024 + 5120 * 8192) + 32768 * 1280 * 1  # bf16 + fp8 q_b
DENSE_BYTES_PER_LAYER = 2 * (2 * 576 * 5120 + 5120 * 576) + 2 * 384 * 5120 + 4 * 384
EMBED_BYTES = 2 * VOCAB * D
HEAD_BYTES_PER_RANK = 2 * VOCAB // N_RANKS * D


def plan_bytes(n_layers: int = N_LAYERS, drafter: bool = True) -> dict:
    """Weight bytes on ONE rank for the full model (estimate from the layouts; activations / workspaces not included)."""
    out = dict(experts=n_layers * EXPERT_BYTES_PER_LAYER, attention=n_layers * ATTN_BYTES_PER_LAYER,
               dense=n_layers * DENSE_BYTES_PER_LAYER, embed=EMBED_BYTES, head=HEAD_BYTES_PER_RANK,
               drafter=(3 * (128 * EXPERT_BYTES_PER_LAYER // 384 + ATTN_BYTES_PER_LAYER + DENSE_BYTES_PER_LAYER)
                        if drafter else 0))
    out["total"] = sum(out.values())
    return out


def _paths():
    for p in (os.path.join(PHYS, "tools"), os.path.join(PHYS, "kernels/attn_dense"), os.path.join(PHYS, "kernels/moe_stream"),
              os.path.join(PHYS, "kernels/sp_layer"), os.path.join(PHYS, "kernels/head"), REF):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------------------------- CPU builders
def build_raw_experts(layer: int, rank: int, model: str = MODEL, tp: int = N_RANKS):
    """= kernels/sp_layer/sp_layer.py load_raw_experts (raw TP-sliced ds41h layout), on the CPU."""
    _paths()
    import torch
    from ds41h.ref.weights import CheckpointReader

    rd = CheckpointReader(model)
    lay = rd.layer(layer)
    u8 = torch.uint8
    out = [[], [], [], []]
    for e in range(int(rd.config["n_routed_experts"])):
        p = f"ffn.experts.{e}."
        out[0].append(torch.cat([lay.tp(p + "w1.weight", rank, tp).view(u8), lay.tp(p + "w3.weight", rank, tp).view(u8)]))
        out[1].append(torch.cat([lay.tp(p + "w1.scale", rank, tp).view(u8), lay.tp(p + "w3.scale", rank, tp).view(u8)]))
        out[2].append(lay.tp(p + "w2.weight", rank, tp).view(u8))
        out[3].append(lay.tp(p + "w2.scale", rank, tp).view(u8))
    return tuple(torch.stack(t).contiguous() for t in out)


def build_attn(layer: int, model: str = MODEL) -> dict:
    """attn_half.load_attn_weights(layer, 'cpu', w8a16=False) without the fp32 reference copy, plus the raw fp8 q_b
    weight + scale so the fused W8A16 q_b (attn_half's default hybrid) can be packed on the device without the checkpoint."""
    _paths()
    import attn_half as AH
    import bench2b as B

    W = AH.load_attn_weights(layer, "cpu", model, w8a16=False)
    W.pop("_ref", None)
    W.pop("cos_sin")          # [1M, 64] fp32 = 268 MB, the same for every layer of a compress ratio: cos_sin_c{ratio}_v1
    rd = B.CheckpointReader(model)
    W["_raw_wq_b"] = (rd.get(f"layers.{layer}.attn.wq_b.weight"), rd.get(f"layers.{layer}.attn.wq_b.scale"))
    W["device"] = "cpu"
    return W


def build_cos_sin(ratio: int):
    """Stock RoPE table of a compress ratio (attn_half / bench2b.stock_cos_sin), shared by all layers of that ratio."""
    _paths()
    import bench2b as B

    return B.stock_cos_sin(ratio, "cpu", 0)


def build_dense(layer: int, rank: int, model: str = MODEL) -> dict:
    _paths()
    import torch
    import bench
    import block

    w1, w3, w2, gw, gb = block.load_dense(bench.Ckpt(model), layer, rank, "cpu")
    return dict(sw13=torch.cat([w1, w3], 0).contiguous(), sw2=w2, gate_w=gw, gate_b=gb)


def build_mhc(layer: int, model: str = MODEL) -> dict:
    """Raw mHC parameters (prep_fn of the fn matrices runs on the device at load)."""
    _paths()
    from ds41h.ref.weights import default_reader

    rd = default_reader(model)
    g = lambda n: rd.get(f"layers.{layer}.{n}").float()  # noqa: E731
    return dict(fa_raw=g("hc_attn_fn"), sa=g("hc_attn_scale"), ba=g("hc_attn_base"), na=g("attn_norm.weight"),
                ff_raw=g("hc_ffn_fn"), sf=g("hc_ffn_scale"), bf=g("hc_ffn_base"), nf=g("ffn_norm.weight"))


def build_embed(model: str = MODEL):
    _paths()
    from ds41h.ref.weights import default_reader

    import torch
    return default_reader(model).get("embed.weight").to(torch.bfloat16).contiguous()


def build_head(rank: int, model: str = MODEL):
    """Vocab shard of the LM head (kernels/head VerifyHead: logits_r = h_all @ W_r^T)."""
    _paths()
    import torch
    from ds41h.ref.weights import default_reader

    w = default_reader(model).get("head.weight").to(torch.bfloat16)
    v = w.shape[0] // N_RANKS
    return w[rank * v:(rank + 1) * v].contiguous()


def build_drafter_block(i: int, rank: int, model: str = MODEL) -> dict:
    """kernels/head/drafter.py load_block on the CPU; the mHC fn matrices stay raw (prep_fn on the device at load)."""
    _paths()
    import drafter as DR
    from ds41h.ref import layer as RL
    from ds41h.ref.weights import default_reader

    rd = default_reader(model)
    cfg = RL.ModelConfig.from_hf(rd.config)

    class _Raw:
        @staticmethod
        def prep_fn(x):
            return x

    B = DR.load_block(rd, cfg, i, rank, "cpu", _Raw)
    B["fa_raw"], B["ff_raw"] = B.pop("fa"), B.pop("ff")
    return B


def entries(layers, ranks, small_all: bool = False, globals_: bool = False, drafter: bool = False):
    """(key, builder) pairs. layers: layers whose big weights (experts + attention) are built; small_all: mHC + dense of
    all 40 layers (what the reuse mode of the smoke test needs)."""
    out = []
    ratios = sorted({RATIO[L] for L in layers})
    for c in ratios:
        out.append((f"cos_sin_c{c}_v1", lambda c=c: build_cos_sin(c)))
    for L in layers:
        out.append((f"attn_L{L}_v1", lambda L=L: build_attn(L)))
        for r in ranks:
            out.append((f"L{L}_r{r}_raw_experts_v1", lambda L=L, r=r: build_raw_experts(L, r)))
    small = range(N_LAYERS) if small_all else layers
    for L in small:
        out.append((f"mhc_L{L}_v1", lambda L=L: build_mhc(L)))
        for r in ranks:
            out.append((f"L{L}_r{r}_dense_v1", lambda L=L, r=r: build_dense(L, r)))
    if globals_:
        out.append(("embed_v1", build_embed))
        for r in ranks:
            out.append((f"head_r{r}_v1", lambda r=r: build_head(r)))
    if drafter:
        for i in range(3):
            for r in ranks:
                out.append((f"mtp{i}_r{r}_v1", lambda i=i, r=r: build_drafter_block(i, r)))
    return out


# ---------------------------------------------------------------------------------------------- device side
def load(key: str, build, device):
    _paths()
    from wcache import cached

    return cached(key, build, device=device)


def finish_mhc(m: dict, hext) -> dict:
    """mHC parameters on the device: prep_fn of the fn matrices (kernels/sp_layer/mhc.cu)."""
    out = {k: v for k, v in m.items() if not k.endswith("_raw")}
    out["fa"], out["ff"] = hext.prep_fn(m["fa_raw"]), hext.prep_fn(m["ff_raw"])
    return out


def finish_attn(W: dict, device, cos_sin, w8a16: str = "q_b") -> dict:
    """attn_half weights on the device: the fused W8A16 q_b (hybrid, attn_half's default) packed from the raw fp8 copy."""
    _paths()
    import torch

    W = dict(W)
    W["device"] = torch.device(device)
    W["cos_sin"] = cos_sin
    raw = W.pop("_raw_wq_b")
    if w8a16 == "q_b":
        import w8a16 as W8

        with torch.cuda.device(W["device"]):
            W["w8"] = dict(wq_b=W8.Gemm(W8.build(), raw[0].to(device), raw[1].to(device), 1))
        W["w8_mode"] = "q_b"
        W.pop("wq_b")
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prebuild", action="store_true")
    ap.add_argument("--layers", default="21")
    ap.add_argument("--ranks", default="0,1,2,3")
    ap.add_argument("--small-all", action="store_true", help="mHC + dense of all 40 layers")
    ap.add_argument("--globals", action="store_true", help="embedding + vocab-sharded head")
    ap.add_argument("--drafter", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print the per-rank weight bytes of the full model")
    a = ap.parse_args()
    if a.plan:
        print({k: round(v / 1e9, 2) for k, v in plan_bytes().items()})
    if not a.prebuild:
        return
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "prebuild on the CPU: CUDA_VISIBLE_DEVICES= (empty)"
    _paths()
    from wcache import prebuild

    layers = [int(x) for x in a.layers.split(",") if x != ""] if a.layers != "all" else list(range(N_LAYERS))
    ranks = [int(x) for x in a.ranks.split(",")]
    t0 = time.time()
    for key, b in entries(layers, ranks, a.small_all, a.globals, a.drafter):
        t1 = time.time()
        prebuild(key, b)
        print(f"prebuilt {key} in {time.time() - t1:.1f} s", flush=True)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
