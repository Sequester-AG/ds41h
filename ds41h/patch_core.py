"""Common ds41h runtime plumbing: per-forward step context, TP-group helpers, component registry.

Step context
------------
`current_step()` describes the forward pass that is executing *now* (decode-only or not, token
counts, cudagraph mode). It is derived from vLLM's ForwardContext, which every forward path enters:
  * eager / PIECEWISE target steps: GPUModelRunner.execute_model -> set_forward_context
    (vllm/v1/worker/gpu/model_runner.py:2015-2026),
  * FULL cudagraph capture: ModelCudaGraphManager.capture -> forward_fn -> set_forward_context
    (vllm/v1/worker/gpu/cudagraph_utils.py:733-752), with the capture-time dummy attention metadata,
  * DSpark/DFlash draft forwards: DFlashSpeculator._run_model -> set_forward_context
    (vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:265-280).
FULL replays run no Python, so whatever a kernel decided from the context at capture time is what
replays; the capture context describes exactly the batch shape being captured (uniform decode), so
decisions taken from it stay valid for every replay of that graph. The result is cached in
ForwardContext.additional_kwargs, so it is computed once per forward and dies with the context.

The DeepSeek-V4.1 attention metadata (DeepseekSparseSWAMetadata,
vllm/v1/attention/backends/mla/sparse_swa.py:180-218) carries num_decodes / num_prefills /
num_decode_tokens / num_prefill_tokens with decodes ordered first; any metadata object with those
fields is used, deduplicated by identity (many layers share one object).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from ds41h import flags
from ds41h.patches import register

logger = logging.getLogger("ds41h")
if not logger.handlers and os.environ.get("DS41H_LOG", "1") not in ("0", ""):
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s %(asctime)s [ds41h pid=%(process)d] %(message)s", "%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

_STEP_KEY = "ds41h.step"

COMPONENT_FLAGS = {
    "allreduce": "ALLREDUCE",
    "candidates": "CANDIDATES",
    "attn": "ATTN",
    "prologue": "PROLOGUE",
    "decoder": "DECODER",
    "sampling": "SAMPLING",
    "prefill": "PREFILL",
}

# component -> list of patched symbols actually applied in this process (filled by patch modules).
ACTIVE: dict[str, list[str]] = {}


def mark_active(component: str, symbol: str) -> None:
    """Record that `component` patched `symbol` in this process (shown in the startup log)."""
    ACTIVE.setdefault(component, [])
    if symbol not in ACTIVE[component]:
        ACTIVE[component].append(symbol)


# ---------------------------------------------------------------------------------------------- step context
@dataclass(frozen=True, slots=True)
class StepInfo:
    num_tokens: int  # rows of this forward incl. cudagraph padding (== hidden_states.shape[0])
    decode_only: bool  # no prefill request in this forward (all attention groups agree)
    num_reqs: int | None  # requests (decodes + prefills) per the attention metadata, if known
    num_decode_tokens: int | None  # decode rows (decodes first), if known
    max_decode_query_len: int | None  # 1 + num_spec_tokens for spec-decode verify steps
    cudagraph_mode: str  # "NONE" | "PIECEWISE" | "FULL" (runtime mode of this forward)
    capturing: bool  # a CUDA graph is being captured on the current stream
    has_attn_metadata: bool  # False for profile/dummy runs without metadata

    @property
    def tokens_per_req(self) -> int | None:
        """Uniform tokens per request for decode-only steps (e.g. 1 + k for DSpark verify)."""
        if not self.decode_only or not self.num_reqs:
            return None
        return self.max_decode_query_len


def _metadata_objects(attn_metadata) -> list:
    if attn_metadata is None:
        return []
    groups = attn_metadata if isinstance(attn_metadata, list) else [attn_metadata]
    seen: dict[int, object] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        for md in group.values():
            if id(md) not in seen and hasattr(md, "num_prefills") and hasattr(md, "num_decode_tokens"):
                seen[id(md)] = md
    return list(seen.values())


def _compute_step(ctx) -> StepInfo:
    import torch

    mds = _metadata_objects(ctx.attn_metadata)
    decode_only = bool(mds) and all(int(getattr(md, "num_prefills", 0)) == 0 for md in mds)
    first = mds[0] if mds else None
    num_decodes = getattr(first, "num_decodes", None)
    num_prefills = getattr(first, "num_prefills", None)
    num_reqs = None if num_decodes is None or num_prefills is None else int(num_decodes) + int(num_prefills)
    num_decode_tokens = None if first is None else int(first.num_decode_tokens)
    num_tokens = None
    if ctx.batch_descriptor is not None:
        num_tokens = int(ctx.batch_descriptor.num_tokens)
    elif ctx.is_padding is not None:
        num_tokens = int(ctx.is_padding.shape[0])
    elif first is not None:
        num_tokens = int(first.num_decode_tokens) + int(getattr(first, "num_prefill_tokens", 0))
    mdq = getattr(first, "max_decode_query_len", None)
    mode = getattr(ctx.cudagraph_runtime_mode, "name", str(ctx.cudagraph_runtime_mode))
    return StepInfo(
        num_tokens=-1 if num_tokens is None else num_tokens,
        decode_only=decode_only,
        num_reqs=num_reqs,
        num_decode_tokens=num_decode_tokens,
        max_decode_query_len=None if mdq is None else int(mdq),
        cudagraph_mode=mode,
        capturing=torch.cuda.is_current_stream_capturing(),
        has_attn_metadata=bool(mds),
    )


def current_step() -> StepInfo | None:
    """StepInfo of the forward pass in progress, or None outside a forward context."""
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    ctx = get_forward_context()
    info = ctx.additional_kwargs.get(_STEP_KEY)
    if info is None:
        info = _compute_step(ctx)
        ctx.additional_kwargs[_STEP_KEY] = info
    return info


def is_decode_step(max_tokens: int | None = None) -> bool:
    """True if the current forward is decode-only (and has at most `max_tokens` rows, if given)."""
    info = current_step()
    if info is None or not info.decode_only:
        return False
    return max_tokens is None or 0 <= info.num_tokens <= max_tokens


# ---------------------------------------------------------------------------------------------- TP helpers
def tp_group():
    """vLLM GroupCoordinator of the tensor-parallel group."""
    from vllm.distributed.parallel_state import get_tp_group

    return get_tp_group()


def tp_cpu_group():
    """Gloo process group of the TP ranks (for IPC handle exchange / object collectives)."""
    return tp_group().cpu_group


def tp_device_group():
    """NCCL process group of the TP ranks."""
    return tp_group().device_group


def tp_rank() -> int:
    return tp_group().rank_in_group


def tp_size() -> int:
    return tp_group().world_size


def tp_device():
    """CUDA device of this TP rank."""
    return tp_group().device


# ---------------------------------------------------------------------------------------------- startup log
def enabled_components() -> list[str]:
    return [name for name, flag in COMPONENT_FLAGS.items() if getattr(flags, flag, False)]


def log_components(where: str) -> None:
    on = enabled_components()
    off = [name for name in COMPONENT_FLAGS if name not in on]
    applied = ", ".join(f"{c}[{'; '.join(s)}]" for c, s in sorted(ACTIVE.items())) or "none yet"
    logger.info("ds41h %s: flags on=%s off=%s; applied patches: %s", where, on, off, applied)


_LOGGED_RUNNER = False


def _patch_model_runner(module) -> None:
    runner = module.GPUModelRunner
    base_load = runner.load_model

    def load_model(self, *args, **kwargs):
        global _LOGGED_RUNNER
        result = base_load(self, *args, **kwargs)
        if not _LOGGED_RUNNER:
            _LOGGED_RUNNER = True
            try:
                rank = tp_rank()
            except Exception:  # noqa: BLE001 - logging must never break loading
                rank = "?"
            log_components(f"worker tp_rank={rank} after load_model")
        return result

    runner.load_model = load_model
    mark_active("core", "GPUModelRunner.load_model(log)")


def _install_stack_dump() -> None:
    """`kill -USR1 <pid>` prints every thread's Python stack to stderr (the server log): ptrace (py-spy)
    may not be permitted, and a hung collective otherwise leaves no trace. DS41H_STACKDUMP=0 disables."""
    if os.environ.get("DS41H_STACKDUMP", "1") in ("0", ""):
        return
    try:
        import faulthandler
        import signal

        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except Exception:  # noqa: BLE001 - diagnostics only
        pass


if flags.ENABLED:
    _install_stack_dump()
    register("vllm.v1.worker.gpu.model_runner")(_patch_model_runner)
    if os.environ.get("DS41H_LOG", "1") not in ("0", ""):
        logger.info("ds41h hook installed: components enabled=%s; %s", enabled_components(), flags.numerics_summary())

# DSpark drafter fast paths (prepare inputs, drafter MoE decode, context-KV precompute)
import ds41h.patch_drafter  # noqa: E402,F401

# Teacher-forced decoding debug hook (only with DS41H_FORCE_FILE; numerics-neutral)
import ds41h.patch_force  # noqa: E402,F401
