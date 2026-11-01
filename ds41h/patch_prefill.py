"""Prefill-path patches owned by the ds41h ``moe`` component.

Installs the raw-layout MXFP4 MoE method (ds41h/moe.py) for DeepSeek-V4.1's routed experts:
``DeepseekV4FP8Config.get_quant_method`` (vllm/models/deepseek_v41/quant_config.py:218-263) returns
``Mxfp4MoEMethod(layer.moe_config)`` for every ``RoutedExperts`` layer of an fp4-expert checkpoint
that is neither skipped nor NVFP4 (lines 239-260). With ds41h enabled exactly those layers get
``Ds41hRawMxfp4MoEMethod`` instead, which keeps the checkpoint layout (no Humming/Marlin repack)
and runs the ds41h W4A8 grouped GEMM. Everything else (dense linears, NVFP4/unquantized experts,
other models) still goes through the stock method. ``DeepseekV4MoE.forward`` / ``MoERunner`` are
untouched: the stock router picks the experts, the runner adds the shared expert and all-reduces.

Enabled when DS41H=1 and DS41H_DECODER or DS41H_PREFILL is on (the decode megakernel needs the raw
layout too); DS41H_MOE=0 turns it off. Patches import torch/vLLM lazily.
"""

from __future__ import annotations

from types import ModuleType

from ds41h import moe
from ds41h.patches import register

QUANT_CONFIG_MODULE = "vllm.models.deepseek_v41.quant_config"


def install_raw_mxfp4_moe(module: ModuleType) -> None:
    """Wrap ``module.DeepseekV4FP8Config.get_quant_method`` (idempotent; original in __wrapped__)."""
    cls = module.DeepseekV4FP8Config
    orig = cls.get_quant_method
    if getattr(orig, "_ds41h_raw_moe", False):
        return

    def get_quant_method(self, layer, prefix):
        # Mirrors the RoutedExperts branch of the stock method (quant_config.py:239-260).
        if (
            isinstance(layer, module.RoutedExperts)
            and self.expert_dtype == "fp4"
            and not module.is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            )
            and not self._is_nvfp4_expert_layer(prefix)
        ):
            return moe.make_raw_mxfp4_moe_method(layer.moe_config)
        return orig(self, layer, prefix)

    get_quant_method._ds41h_raw_moe = True  # type: ignore[attr-defined]
    get_quant_method.__wrapped__ = orig  # type: ignore[attr-defined]
    cls.get_quant_method = get_quant_method


if moe.ENABLED:
    register(QUANT_CONFIG_MODULE)(install_raw_mxfp4_moe)
import ds41h.patch_linear  # noqa
import ds41h.patch_prefill_attn  # noqa
import ds41h.patch_mhc  # noqa

# Prefill sequence parallelism for the full-row layers (ds41h/prefill_sp.py; DS41H_PREFILL_SP, opt-in).
import os as _os
import sys as _sys

from ds41h import prefill_sp as _prefill_sp

_MODEL_MODULE = "vllm.models.deepseek_v41.nvidia.model"
if _prefill_sp.SP:

    @register(_MODEL_MODULE)
    def _patch_sp(module) -> None:
        from ds41h.patch_core import mark_active

        _prefill_sp.install(module)
        mark_active("prefill", f"DeepseekV4Model._run_layers(sequence-parallel full-row layers, T>={_prefill_sp.SP_MIN_T})")

    if _MODEL_MODULE in _sys.modules:
        _patch_sp(_sys.modules[_MODEL_MODULE])


# Determinism probe (debugging only; DS41H_DETPROBE=<dir>, default off): ds41h/detprobe.py
from ds41h import detprobe as _detprobe

if _detprobe.ENABLED:
    register(_MODEL_MODULE)(_detprobe.install)
    if _MODEL_MODULE in _sys.modules:
        _detprobe.install(_sys.modules[_MODEL_MODULE])


# Deterministic stock Humming linears (DS41H_HUMMING_DET=1, default off).
# Stock Humming W8A16 / MXFP8 dense GEMMs use Stream-K: K-slice partials are rounded to bf16 and summed with unordered
# atomics / TMA reduce-adds (humming/include/humming/epilogue/gmem_writer.cuh), i.e. they are run-to-run
# NONDETERMINISTIC (detprobe dp1: layer 0 fused_wqa_wkv and the engram wkv differ between identical forwards) and less
# precise than one fp32 accumulation. With the switch every Humming linear call gets Humming's own batch-invariant
# compute config (vLLM's VLLM_BATCH_INVARIANT setting, here for the Humming linears only): no Stream-K, one fixed K
# partition per output tile -> fp32 accumulation over the whole K, one bf16 rounding: deterministic and at least as
# precise as stock (lossless rule: summation order only; dp2: the whole prefill forward of <= 512-token prompts is then
# bit-reproducible). DS41H_HUMMING_DET: 0 off | 1 every call (incl. the decode graphs' engram wkv) | 2 eager calls only
# (prefill; captured decode graphs keep stock Stream-K). With DS41H_PREFILL_CTL the control key "humming_det" (0/1/2)
# overrides it per eager call (same-server A/B; captured graphs keep the mode of capture time).
from ds41h import flags as _flags
from ds41h import prefill_ctl as _prefill_ctl

HUMMING_DET = int(_os.environ.get("DS41H_HUMMING_DET", "0") or 0) if _flags.ENABLED else 0
_HUMMING_KERNEL_MODULES = tuple(
    f"vllm.model_executor.kernels.linear.{m}.humming" for m in ("mxfp8", "scaled_mm", "mixed_precision", "mxfp4", "nvfp4", "mxfp6")
)
_DET_CFG: dict = {}


def _det_config(cfg):
    """compute config (JSON str / dict / None) -> the same with use_batch_invariant=True (memoized)."""
    import json

    key = cfg if isinstance(cfg, str) or cfg is None else json.dumps(cfg, sort_keys=True)
    out = _DET_CFG.get(key)
    if out is None:
        d = json.loads(cfg) if isinstance(cfg, str) and cfg else (dict(cfg) if isinstance(cfg, dict) else {})
        d["use_batch_invariant"] = True
        out = _DET_CFG[key] = json.dumps(d)
    return out


def _humming_det_mode() -> int:
    return int(_prefill_ctl.get("humming_det", HUMMING_DET)) if _prefill_ctl.PATH else HUMMING_DET


def _patch_humming_det(module) -> None:
    orig = getattr(module, "apply_humming_linear", None)
    if orig is None or getattr(orig, "_ds41h_det", False):
        return

    def apply_humming_linear(layer, x, *args, compute_config=None, **kwargs):
        mode = _humming_det_mode()
        if mode:
            import torch

            if mode == 1 or not torch.cuda.is_current_stream_capturing():
                compute_config = _det_config(compute_config)
        return orig(layer, x, *args, compute_config=compute_config, **kwargs)

    apply_humming_linear._ds41h_det = True  # type: ignore[attr-defined]
    apply_humming_linear.__wrapped__ = orig  # type: ignore[attr-defined]
    module.apply_humming_linear = apply_humming_linear


if HUMMING_DET or (_flags.ENABLED and _prefill_ctl.PATH):
    for _m in _HUMMING_KERNEL_MODULES:
        register(_m)(_patch_humming_det)
        if _m in _sys.modules:
            _patch_humming_det(_sys.modules[_m])


# Deterministic indexer top-k order in prefill (DS41H_TOPK_DET=1, default off).
# The stock prefill indexer top-k (vllm/model_executor/layers/sparse_attn_indexer.py -> ops.top_k_per_row_prefill, rows
# with > index_topk candidates) writes the selected positions in a run-to-run varying order (detprobe dp2: the layer-20
# replay batch of 1K / 3K-token prompts), and the sparse attention accumulates the keys in that order -> fp32 summation
# order differs between identical forwards. With the switch each row's selected positions are sorted ascending (-1 padding
# last) right after the stock kernel: same key set, fixed order (lossless: summation order only). Costs one segmented sort
# of [rows, index_topk] int32 per index layer and prefill chunk. With DS41H_PREFILL_CTL the key "topk_det" overrides it.
TOPK_DET = _flags.ENABLED and _flags._flag("DS41H_TOPK_DET", "0")


class _TopkDetOps:
    """Proxy for the `ops` module of sparse_attn_indexer: top_k_per_row_prefill + ascending row sort, rest delegated."""

    def __init__(self, real) -> None:
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def top_k_per_row_prefill(self, logits, cu_seqlen_ks, cu_seqlen_ke, raw_topk_indices, num_rows, stride0, stride1, topk_tokens):
        self._real.top_k_per_row_prefill(logits, cu_seqlen_ks, cu_seqlen_ke, raw_topk_indices, num_rows, stride0, stride1,
                                         topk_tokens)
        if not (int(_prefill_ctl.get("topk_det", int(TOPK_DET))) if _prefill_ctl.PATH else TOPK_DET):
            return
        import torch

        v = raw_topk_indices[:num_rows]
        if v.numel():
            big = torch.iinfo(v.dtype).max
            k = torch.where(v < 0, torch.full_like(v, big), v).sort(dim=1).values
            v.copy_(torch.where(k == big, torch.full_like(k, -1), k))


def _patch_topk_det(module) -> None:
    ops = getattr(module, "ops", None)
    if ops is None or isinstance(ops, _TopkDetOps):
        return
    module.ops = _TopkDetOps(ops)


_SAI_MODULE = "vllm.model_executor.layers.sparse_attn_indexer"
if TOPK_DET or (_flags.ENABLED and _prefill_ctl.PATH):  # ctl key "topk_det" (0/1) switches it per call
    register(_SAI_MODULE)(_patch_topk_det)
    if _SAI_MODULE in _sys.modules:
        _patch_topk_det(_sys.modules[_SAI_MODULE])


# DSpark context work of an eager prefill step on the replay rows only (ds41h/prefill_ctx.py;
# DS41H_PREFILL_CTX_TRIM=1 or ctl key "ctx_trim", default off): the trimmed rows' drafter inputs are zero (stock replay
# scatter) -> main_proj / the context KV projections run on the replay rows, zeros elsewhere; same main_x and cache bytes.
from ds41h import prefill_ctx as _prefill_ctx

if _prefill_ctx.INSTALL:
    for _mod, _fn in ((_MODEL_MODULE, _prefill_ctx.patch_model),
                      ("vllm.models.deepseek_v41.decoder_replay_layers", _prefill_ctx.patch_replay),
                      ("vllm.models.deepseek_v41.nvidia.dspark", _prefill_ctx.patch_dspark)):
        register(_mod)(_fn)
        if _mod in _sys.modules:
            _fn(_sys.modules[_mod])
