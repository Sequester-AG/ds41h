"""DeepSeek-V4.1 vision tower on one GPU, outside vLLM's model runner (images in the fork's prefill).

What it computes is vLLM's DeepseekV41ForCausalLM._process_image_input (vllm/models/deepseek_v41/nvidia/vl_model.py,
non-data-parallel branch) with vLLM's own modules (vllm/models/deepseek_v4/common/vision.py: DeepseekV4ViT +
DeepseekV4Aligner) and the checkpoint's weights (vision.*, aligner.*, image_start / image_end / image_newline):
  per image: aligner(vit(patches)) = one row per IMAGE slot, in reading order; the span = those rows at the IMAGE
  positions and the learned delimiter vectors at IMAGE_START / IMAGE_NEW_LINE / IMAGE_END, in the model dtype (bf16).
  A span has n_llm_h * (n_llm_w + 1) + 2 positions; every one of them carries IMAGE_TOKEN_ID in the prompt's ids.

  tower = VisionTower(model_dir, device)                   # ~0.9 GB of bf16 weights on `device`
  spans = tower.encode(patches, vit_grid, llm_grid, types)  # list of [span_len, 5120] bf16, one per image

The modules are built with vLLM's tensor-parallel group of size 1 (created here if the process has none) and, if the
caller passes it, the serving process's VllmConfig (so encoder attention / custom-op dispatch follow its settings).
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import socket
from types import SimpleNamespace

import torch

IMAGE_TOKEN_ID = 129264
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)      # `types` roles (vLLM common/mm_preprocess.py)


@dataclasses.dataclass
class ImageInput:
    """One image of a request: its span's first prompt position and either the processor outputs (encoded by the
    engine on the request's rank) or a ready span embedding."""
    offset: int
    patches: torch.Tensor | None = None       # [n_vit_h * n_vit_w, 3, 14, 14]
    vit_grid: tuple | None = None             # (n_vit_h, n_vit_w)
    llm_grid: tuple | None = None             # (n_llm_h, n_llm_w)
    types: torch.Tensor | None = None         # [span_len] roles
    embeds: torch.Tensor | None = None        # [span_len, hidden] (instead of the four fields above)

    @property
    def length(self) -> int:
        if self.embeds is not None:
            return int(self.embeds.shape[0])
        h, w = self.llm_grid
        return int(h) * (int(w) + 1) + 2


def split_mm_inputs(offsets, patches, vit_grid, llm_grid, types) -> list[ImageInput]:
    """vLLM's per-request batched processor outputs (all images concatenated) -> one ImageInput per image."""
    out, vo, so = [], 0, 0
    vg = vit_grid.tolist() if torch.is_tensor(vit_grid) else list(vit_grid)
    lg = llm_grid.tolist() if torch.is_tensor(llm_grid) else list(llm_grid)
    for off, (vh, vw), (lh, lw) in zip(offsets, vg, lg, strict=True):
        n, L = int(vh) * int(vw), int(lh) * (int(lw) + 1) + 2
        out.append(ImageInput(int(off), patches[vo:vo + n], (int(vh), int(vw)), (int(lh), int(lw)), types[so:so + L]))
        vo, so = vo + n, so + L
    return out


def _ensure_tp1() -> None:
    """vLLM's linear / attention layers query the tensor-parallel group at construction: create a 1-process group if
    this process has none (the engine runs its own collectives and never uses it)."""
    from vllm.distributed import parallel_state as ps
    if ps.model_parallel_is_initialized():
        return
    import torch.distributed as dist
    if not dist.is_initialized():
        with socket.socket() as s_:
            s_.bind(("127.0.0.1", 0))
            port = s_.getsockname()[1]
        ps.init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                        local_rank=0, backend="gloo")
    ps.initialize_model_parallel(tensor_model_parallel_size=1)


def _config(model_dir: str) -> SimpleNamespace:
    """The tower's dimensions from the checkpoint's config.json (HF layout: text_config.hidden_size + vision_config;
    DeepSeek's inference layout: dim + vision_*)."""
    c = json.load(open(os.path.join(model_dir, "config.json")))
    v = c.get("vision_config")
    if v is not None:
        hid = c.get("hidden_size") or c.get("dim") or (c.get("text_config") or {}).get("hidden_size")
        return SimpleNamespace(hidden_size=hid, vision_n_layers=v["num_hidden_layers"],
                               vision_dim=v["hidden_size"], vision_n_heads=v["num_attention_heads"],
                               vision_inter_dim=v["intermediate_size"], vision_patch_size=v["patch_size"],
                               vision_downsample_ratio=v["downsample_ratio"], vision_rope_theta=v.get("rope_theta", 10000),
                               image_token_id=c["image_token_id"])
    return SimpleNamespace(hidden_size=c["dim"], vision_n_layers=c["vision_n_layers"], vision_dim=c["vision_dim"],
                           vision_n_heads=c["vision_n_heads"], vision_inter_dim=c["vision_inter_dim"],
                           vision_patch_size=c["vision_patch_size"],
                           vision_downsample_ratio=c["vision_downsample_ratio"],
                           vision_rope_theta=c.get("vision_rope_theta", 10000), image_token_id=c["image_token_id"])


def _weights(model_dir: str, prefixes=("vision.", "aligner.", "image_")):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    names = sorted(k for k in idx if k.startswith(prefixes))
    by_file = {}
    for k in names:
        by_file.setdefault(idx[k], []).append(k)
    out = {}
    for f, ks in by_file.items():
        with safe_open(os.path.join(model_dir, f), framework="pt") as fh:
            for k in ks:
                out[k] = fh.get_tensor(k)
    return out


class VisionTower:
    def __init__(self, model_dir: str, device, dtype: torch.dtype = torch.bfloat16, vllm_config=None):
        self.device = torch.device(device)
        self.dtype = dtype
        cfg = _config(model_dir)
        assert cfg.image_token_id == IMAGE_TOKEN_ID, cfg.image_token_id
        self.cfg = cfg
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.config.vllm import get_current_vllm_config_or_none
        from vllm.models.deepseek_v4.common.vision import DeepseekV4Aligner, DeepseekV4ViT
        ctx = contextlib.ExitStack()
        if vllm_config is not None:
            ctx.enter_context(set_current_vllm_config(vllm_config))
        elif get_current_vllm_config_or_none() is None:      # outside a serving process: vLLM's defaults (its tests' way)
            ctx.enter_context(set_current_vllm_config(VllmConfig()))
        old = torch.get_default_dtype()
        with ctx, torch.device(self.device):
            _ensure_tp1()
            torch.set_default_dtype(dtype)                 # vLLM builds a model under its dtype (attention backend dtype)
            try:
                self.vision = DeepseekV4ViT(cfg)
                self.aligner = DeepseekV4Aligner(cfg)
            finally:
                torch.set_default_dtype(old)
            self.vision.to(dtype=dtype)                    # as vl_model.py: the tower and the aligner in the model dtype
            self.aligner.to(dtype=dtype)
        w = _weights(model_dir)
        params = {f"vision.{k}": p for k, p in self.vision.named_parameters()}
        params.update({f"aligner.{k}": p for k, p in self.aligner.named_parameters()})
        missing = sorted(set(params) - set(w))
        assert not missing, f"vision weights missing from the checkpoint: {missing[:4]}"
        with torch.no_grad():
            for k, p in params.items():
                t = w[k]
                assert tuple(t.shape) == tuple(p.shape), (k, tuple(t.shape), tuple(p.shape))
                p.copy_(t.to(device=self.device, dtype=p.dtype))
            # fp32 delimiter parameters (vl_model.py), cast to the model dtype when a span is built
            self.delim = {r: w[n].to(device=self.device, dtype=torch.float32)
                          for r, n in ((IMAGE_START, "image_start"), (IMAGE_NEW_LINE, "image_newline"),
                                       (IMAGE_END, "image_end"))}
        self.vision.eval()
        self.aligner.eval()

    @torch.inference_mode()
    def encode_one(self, patches: torch.Tensor, vit_grid, llm_grid, types: torch.Tensor) -> torch.Tensor:
        """One image -> its span [n_llm_h * (n_llm_w + 1) + 2, hidden] in the model dtype."""
        vh, vw = (int(x) for x in vit_grid)
        lh, lw = (int(x) for x in llm_grid)
        p = patches.to(device=self.device, dtype=self.aligner.w1.weight.dtype)
        assert p.shape[0] == vh * vw, (tuple(p.shape), vh, vw)
        emb = self.aligner(self.vision(p, vh, vw), vh, vw)
        t = types.to(self.device)
        L = lh * (lw + 1) + 2
        assert t.numel() == L and int((t == IMAGE).sum()) == emb.shape[0], (t.numel(), L, tuple(emb.shape))
        span = emb.new_empty(L, emb.shape[-1])
        dt = emb.dtype
        span[t == IMAGE_START] = self.delim[IMAGE_START].to(dt)
        span[t == IMAGE_END] = self.delim[IMAGE_END].to(dt)
        span[t == IMAGE_NEW_LINE] = self.delim[IMAGE_NEW_LINE].to(dt)
        span[t == IMAGE] = emb
        return span

    def encode(self, patches, vit_grid, llm_grid, types) -> list:
        """vLLM's batched layout (all images of a request concatenated) -> one span per image."""
        return [self.encode_one(x.patches, x.vit_grid, x.llm_grid, x.types)
                for x in split_mm_inputs([0] * len(vit_grid), patches, vit_grid, llm_grid, types)]


def default_model_dir() -> str:
    """The engine's model directory (EngineConfig.model's default)."""
    from engine import engine as E
    return E.EngineConfig().model


__all__ = ["IMAGE_TOKEN_ID", "ImageInput", "VisionTower", "split_mm_inputs"]
