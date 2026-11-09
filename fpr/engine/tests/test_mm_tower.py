"""Images (FPRv1.1), one-GPU pieces: the vision tower and image-token routing.

1. engine/vision.py's VisionTower (vLLM's DeepseekV4ViT + DeepseekV4Aligner with the checkpoint's weights):
   - bit for bit vLLM's own DeepseekV41ForCausalLM._process_image_input on the same modules (the serving path);
   - against DeepSeek's reference implementation shipped with the model (inference/vision.py, plain torch, SDPA) on
     the same weights, to the accuracy of the dtype. Measured: bf16 (FlashAttention) relative error ~3 %, mean
     cosine 0.9993-0.9996, worst row 0.976; fp32 relative error 0.5-1 %, mean cosine 0.99996-0.99999. A wrong weight
     mapping or span layout gives ~100 %. Delimiter rows are exactly the learned vectors.
2. Engine._router_vl: image-token rows are routed with the router's bias_vl instead of its correction bias (production's
   Gate: bias = where(image_mask, bias_vl, bias); the bias selects experts, the weights come from the unbiased scores).
   The engine re-runs the router kernel on the compacted image rows; the result must equal a full run of the kernel
   with bias_vl on those rows and leave every other row exactly as the bias run wrote it.
Synthetic inputs only (random patches, random router logits).
"""
from __future__ import annotations

import importlib.util
import math
import os
import types as _types

import pytest


def _span_types(lh: int, lw: int):
    import torch
    from engine.vision import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START
    t = [IMAGE_START] + ([IMAGE] * lw + [IMAGE_NEW_LINE]) * lh + [IMAGE_END]
    return torch.tensor(t, dtype=torch.int64)


def _patches(vh: int, vw: int, seed: int):
    import torch
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(vh * vw, 3, 14, 14, generator=g) * 2 - 1).to(torch.bfloat16)   # the processor's [-1, 1] range


def test_split_and_lengths():
    import torch
    from engine.vision import ImageInput, split_mm_inputs
    grids = [((36, 48), (12, 16)), ((20, 29), (7, 10))]
    pt = torch.cat([_patches(vh, vw, i) for i, ((vh, vw), _) in enumerate(grids)])
    ty = torch.cat([_span_types(lh, lw) for _, (lh, lw) in grids])
    xs = split_mm_inputs([400, 1800], pt, torch.tensor([g[0] for g in grids]), torch.tensor([g[1] for g in grids]), ty)
    assert [x.offset for x in xs] == [400, 1800]
    assert [x.length for x in xs] == [12 * 17 + 2, 7 * 11 + 2]
    assert [x.patches.shape[0] for x in xs] == [36 * 48, 20 * 29]
    assert all(x.types.numel() == x.length for x in xs)
    assert ImageInput(5, embeds=torch.zeros(9, 4)).length == 9


@pytest.fixture(scope="module")
def tower():
    from engine import engine as E
    from engine.vision import VisionTower
    return VisionTower(E.EngineConfig().model, "cuda:0")


def _reference_file(model_dir: str) -> str | None:
    """DeepSeek's reference inference/vision.py: next to the checkpoint, or in $DSV41_INFERENCE_DIR."""
    for d in (os.environ.get("DSV41_INFERENCE_DIR"), os.path.join(model_dir, "inference")):
        if d and os.path.exists(os.path.join(d, "vision.py")):
            return os.path.join(d, "vision.py")
    return None


def _reference(model_dir: str, dev):
    """DeepSeek's reference ViT + Aligner (plain torch) with the same weights."""
    import torch
    from engine.vision import _config, _weights
    f = _reference_file(model_dir)
    if f is None:
        pytest.skip("DeepSeek's inference/vision.py not found (set DSV41_INFERENCE_DIR)")
    spec = importlib.util.spec_from_file_location("ds_ref_vision", f)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    c = _config(model_dir)
    args = _types.SimpleNamespace(dim=c.hidden_size, vision_n_layers=c.vision_n_layers, vision_dim=c.vision_dim,
                                  vision_n_heads=c.vision_n_heads, vision_inter_dim=c.vision_inter_dim,
                                  vision_patch_size=c.vision_patch_size,
                                  vision_downsample_ratio=c.vision_downsample_ratio, vision_rope_theta=c.vision_rope_theta)
    vit, al = ref.ViT(args), ref.Aligner(args)
    w = _weights(model_dir)
    vit.load_state_dict({k[len("vision."):]: v for k, v in w.items() if k.startswith("vision.")})
    al.load_state_dict({k[len("aligner."):]: v for k, v in w.items() if k.startswith("aligner.")})
    return vit.to(dev, torch.bfloat16).eval(), al.to(dev, torch.bfloat16).eval()


TOL = {"bf16": dict(rel=0.05, cos_mean=0.999, cos_min=0.95), "fp32": dict(rel=0.02, cos_mean=0.9999, cos_min=0.995)}


@pytest.fixture(scope="module")
def tower32():
    import torch
    from engine import engine as E
    from engine.vision import VisionTower
    return VisionTower(E.EngineConfig().model, "cuda:0", dtype=torch.float32)


@pytest.mark.gpu(n=1)
@pytest.mark.parametrize("dt", ["bf16", "fp32"])
@pytest.mark.parametrize("vit_grid", [(36, 48), (20, 29)])
def test_tower_matches_reference(tower, tower32, vit_grid, dt):
    import torch
    from engine import engine as E
    from engine.vision import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START
    if dt == "fp32":
        tower = tower32
    dtype = torch.bfloat16 if dt == "bf16" else torch.float32
    vh, vw = vit_grid
    lh, lw = math.ceil(vh / 3), math.ceil(vw / 3)
    ty = _span_types(lh, lw)
    pt = _patches(vh, vw, seed=vh).to(dtype)
    span = tower.encode_one(pt, (vh, vw), (lh, lw), ty)
    assert tuple(span.shape) == (lh * (lw + 1) + 2, 5120) and span.dtype == dtype
    assert bool(torch.isfinite(span.float()).all())
    vit, al = _reference(E.EngineConfig().model, span.device)
    vit, al = vit.to(dtype), al.to(dtype)
    with torch.inference_mode(), torch.device(span.device):     # the reference builds its RoPE tables on the default device
        want = al(vit(pt.to(span.device), vh, vw), vh, vw).float()
    got = span[ty.to(span.device) == IMAGE].float()
    assert got.shape == want.shape == (lh * lw, 5120)
    cos = torch.nn.functional.cosine_similarity(got, want, dim=1)
    rel = (got - want).norm() / want.norm()
    t = TOL[dt]
    assert float(rel) < t["rel"] and float(cos.mean()) > t["cos_mean"] and float(cos.min()) > t["cos_min"], \
        (dt, float(rel), float(cos.mean()), float(cos.min()))
    for role, vec in ((IMAGE_START, tower.delim[IMAGE_START]), (IMAGE_END, tower.delim[IMAGE_END]),
                      (IMAGE_NEW_LINE, tower.delim[IMAGE_NEW_LINE])):
        rows = span[ty.to(span.device) == role]
        assert torch.equal(rows, vec.to(dtype).expand_as(rows))


@pytest.mark.gpu(n=1)
def test_router_vl_rows_equal_a_bias_vl_run():
    import torch
    from engine import engine as E
    import bench                                       # kernels/moe_stream (on sys.path through the engine import)
    dev = torch.device("cuda", 0)
    ext = bench.build_extr()
    T, NE = 300, 384
    g = torch.Generator(device=dev).manual_seed(5)
    logits = torch.randn(T, NE, device=dev, generator=g) * 2
    b_g = torch.randn(NE, device=dev, generator=g) * 0.1
    b_vl = torch.randn(NE, device=dev, generator=g) * 0.1
    ptok = torch.randint(0, 1 << 20, (T,), device=dev, generator=g, dtype=torch.int32)
    valid = torch.ones(T, dtype=torch.bool, device=dev)

    def full(bias):
        o = dict(ids=torch.empty(T, 6, dtype=torch.int32, device=dev), wts=torch.empty(T, 6, device=dev),
                 row=torch.zeros(T, 64, dtype=torch.uint8, device=dev))
        ext.router_top_pack3(logits, bias, o["ids"], o["wts"], 1.5, o["row"], valid, ptok)
        return o

    ref_g, ref_vl = full(b_g), full(b_vl)
    ri = torch.tensor(sorted(set(torch.randint(0, T, (60,), generator=torch.Generator().manual_seed(1)).tolist())),
                      device=dev)
    s = dict(logits=logits, ids=ref_g["ids"].clone(), wts=ref_g["wts"].clone(), route_row=ref_g["row"].clone(),
             ptok_own=ptok, mm_rt=ri)
    stub = _types.SimpleNamespace(mext=[ext], _bias_vl=lambda r, L: b_vl)
    E.Engine._router_vl(stub, 0, s, 0)
    torch.cuda.synchronize(dev)
    m = torch.zeros(T, dtype=torch.bool, device=dev)
    m[ri] = True
    for k, kk in (("ids", "ids"), ("wts", "wts"), ("route_row", "row")):
        assert torch.equal(s[k][m], ref_vl[kk][m]), k
        assert torch.equal(s[k][~m], ref_g[kk][~m]), k
    # production's gate semantics (reference Gate): bias_vl picks the experts of image rows
    sc = torch.nn.functional.softplus(logits[ri]).sqrt()
    want = (sc + b_vl).topk(6, dim=-1).indices.sort(dim=-1).values
    assert torch.equal(s["ids"][ri].long().sort(dim=-1).values, want)


@pytest.mark.gpu(n=1)
def test_tower_equals_vllm_process_image_input(tower):
    """vLLM's own embed path (DeepseekV41ForCausalLM._process_image_input with its _encode_image / _build_image_span)
    on the same modules and inputs gives the tower's spans bit for bit (two images in one batched request)."""
    import torch
    from vllm.models.deepseek_v41.common.vl_cudagraph import DeepseekV4VLEncoderCudaGraphMixin as Mix
    from vllm.models.deepseek_v41.nvidia.vl_model import DeepseekV41ForCausalLM, DeepseekV4VLImagePixelInputs
    grids = [(36, 48), (20, 29)]
    lgs = [(math.ceil(h / 3), math.ceil(w / 3)) for h, w in grids]
    pt = torch.cat([_patches(h, w, seed=h) for h, w in grids]).to(tower.device)
    ty = torch.cat([_span_types(lh, lw) for lh, lw in lgs]).to(tower.device)
    vg, lg = torch.tensor(grids), torch.tensor(lgs)
    stub = _types.SimpleNamespace(vision=tower.vision, aligner=tower.aligner, use_data_parallel=False,
                                  config=_types.SimpleNamespace(vision_patch_size=14),
                                  image_start=torch.nn.Parameter(tower.delim[0], requires_grad=False),
                                  image_newline=torch.nn.Parameter(tower.delim[2], requires_grad=False),
                                  image_end=torch.nn.Parameter(tower.delim[3], requires_grad=False))
    stub._encode_image = _types.MethodType(Mix._encode_image, stub)
    stub._build_image_span = _types.MethodType(Mix._build_image_span, stub)
    inp = DeepseekV4VLImagePixelInputs(patches=pt, vit_grid=vg, llm_grid=lg, types=ty, resolve_bindings={"p": 14})
    with torch.inference_mode():
        want = DeepseekV41ForCausalLM._process_image_input(stub, inp)
    got = tower.encode(pt, vg, lg, ty)
    assert len(got) == len(want) == 2
    for g_, w_ in zip(got, want):
        assert g_.shape == w_.shape and torch.equal(g_, w_)
