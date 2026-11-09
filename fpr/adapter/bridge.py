"""Shared plumbing of the vLLM adapter: serve/ import path, the per-step message, the process-local registry.

vLLM 0.32's EngineCore builds the executor first and the scheduler second, both in the EngineCore process
(EngineCore.__init__: executor_class(vllm_config) ... _initialize_kv_caches ... Scheduler(...)). PhysExecutor registers
the engine geometry here; PhysScheduler reads it, so both sides agree on pool sizes without patching vLLM's config
classes. The per-step data travels on the SchedulerOutput itself (`so.phys`, a PhysStep): EngineCore passes the same
object to our own executor's execute_model(), so nothing is serialized.

serve/ uses flat imports (`from geometry import ...`), so its directory goes on sys.path. Its module names (scheduler,
interfaces, worker, geometry, ...) are checked against what is already imported, so a clash fails loudly here.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVE_DIR = os.path.join(ROOT, "serve")
_SERVE_MODULES = ("geometry", "kv_manager", "cost_model", "scheduler", "interfaces", "worker", "weights")


def ensure_serve_path() -> None:
    if SERVE_DIR not in sys.path:
        for m in _SERVE_MODULES:
            mod = sys.modules.get(m)
            if mod is not None and not str(getattr(mod, "__file__", "")).startswith(SERVE_DIR):
                raise ImportError(f"adapter: a foreign top-level module '{m}' ({mod.__file__}) shadows serve/{m}.py")
        sys.path.insert(0, SERVE_DIR)


ensure_serve_path()

from geometry import Geometry, SmallGeometry  # noqa: E402

# ------------------------------------------------------------------------------------------------ registry
_REG: dict[str, Any] = {}


def set_geometry(geom: Geometry) -> None:
    _REG["geometry"] = geom


def set_sampling_caps(caps) -> None:
    """The sampling features the worker's head / sampler implements (None: everything)."""
    _REG["sampling_caps"] = None if caps is None else frozenset(caps)


def get_sampling_caps():
    return _REG.get("sampling_caps")


def set_cost_default(params: dict | None) -> None:
    """The step-cost parameters the scheduler plans with unless $PHYS_COST overrides them: the worker's (serve's
    OfflineEngine rule: the real engine -> cost_model.ENGINE_MEASURED_2026_11_07, stub / mock -> the defaults). The
    prefill chunk boundaries follow from them, and the engine's numerics depend on the boundaries (with the defaults the
    served fork cut 768-token chunks where the offline fork cut 4,096)."""
    _REG["cost_default"] = None if params is None else dict(params)


def get_cost_default():
    return _REG.get("cost_default")


def set_image_support(flag: bool) -> None:
    """True when the worker encodes images and feeds their embeddings to the engine (adapter/mm.py, PhysStep.mm)."""
    _REG["images"] = bool(flag)


def get_image_support() -> bool:
    return bool(_REG.get("images", False))


SAMPLING_FEATURES = ("top_p", "top_k", "min_p", "penalties", "logit_bias", "bad_words", "allowed_token_ids", "logprobs",
                     "prompt_logprobs", "min_tokens")


def sampling_features(sp) -> set:
    """The non-default sampling features a vLLM SamplingParams uses (beyond temperature / seed / stop / max_tokens)."""
    used = set()
    if sp is None:
        return used
    if sp.top_p is not None and sp.top_p < 1.0:
        used.add("top_p")
    if sp.top_k is not None and sp.top_k > 0:
        used.add("top_k")
    if sp.min_p is not None and sp.min_p > 0.0:
        used.add("min_p")
    if (sp.presence_penalty or 0.0) != 0.0 or (sp.frequency_penalty or 0.0) != 0.0 or (sp.repetition_penalty or 1.0) != 1.0:
        used.add("penalties")
    if getattr(sp, "logit_bias", None):
        used.add("logit_bias")
    if getattr(sp, "bad_words", None) or getattr(sp, "bad_words_token_ids", None):
        used.add("bad_words")
    if getattr(sp, "allowed_token_ids", None):
        used.add("allowed_token_ids")
    if sp.logprobs is not None:
        used.add("logprobs")
    if sp.prompt_logprobs is not None:
        used.add("prompt_logprobs")
    if (sp.min_tokens or 0) > 0:
        used.add("min_tokens")
    return used


def get_geometry() -> Geometry:
    """The geometry the executor registered, else one from the environment (CPU tests without PhysExecutor)."""
    g = _REG.get("geometry")
    if g is not None:
        return g
    return geometry_from_env()


def geometry_overrides(g: Geometry) -> Geometry:
    """PHYS_GEOM='field=value,...' overrides Geometry fields (e.g. max_seqs_per_rank=16,max_prefill_tokens_per_rank=64 to
    fit the step program's row buckets while it lacks the big prefill buckets)."""
    import dataclasses

    spec = os.environ.get("PHYS_GEOM", "")
    kw = {}
    for kv in filter(None, (x.strip() for x in spec.split(","))):
        k, v = kv.split("=", 1)
        cur = getattr(g, k)
        kw[k] = type(cur)(float(v)) if isinstance(cur, int) and not isinstance(cur, bool) else type(cur)(v)
    return dataclasses.replace(g, **kw) if kw else g


def geometry_from_env() -> Geometry:
    """PHYS_GEOMETRY=full (default: the H200 pools of geometry.py) | small:<paged>,<window>,<state> (unit tests) |
    kv:<bytes per rank> (full structure, smaller pools)."""
    spec = os.environ.get("PHYS_GEOMETRY", "full")
    if spec == "full":
        return Geometry()
    if spec.startswith("small:"):
        p, w, s = (int(x) for x in spec[6:].split(","))
        return SmallGeometry(paged_pages=p, window_pages=w, state_slots=s)
    if spec.startswith("kv:"):
        return Geometry(kv_bytes_per_rank=int(float(spec[3:])))
    raise ValueError(f"PHYS_GEOMETRY={spec!r}")


# ------------------------------------------------------------------------------------------------ step message
@dataclass
class ReqInit:
    """A request entering the worker (first scheduling, or re-admission after a preemption: recompute)."""
    rid: str
    rank: int
    prompt_token_ids: list            # the prompt
    output_token_ids: list            # outputs so far (non-empty after a preemption: they are recomputed as prefill)
    sampling_params: Any              # vllm.SamplingParams (penalties, seed, temperature, logprobs, ...)
    structured: bool = False          # grammar-constrained (bitmask rows arrive in sample_tokens)


@dataclass
class PhysStep:
    """Everything the worker needs for one engine step (attached to vLLM's SchedulerOutput as `so.phys`).

    batches[r]      the StepBatch of rank r (serve/interfaces.py): rows, slots, block / window tables, copies.
    row_tokens[r]   input token id of every row of rank r: prefill rows and the first row of each decode group carry the
                    real token (host-known); the k draft rows of a decode group carry -1 (the drafts live on the device,
                    written by the previous step's drafter).
    prev_tokens     rid -> the up to 3 token ids before the sequence's first row this step (Engram n-gram context).
    new_reqs        requests whose worker-side state starts this step (sampling params, token history).
    finished        requests that ended (stop, abort, length) since the previous step: drop their state.
    preempted       requests preempted this step (their KV is gone; they come back as new_reqs).
    order           vLLM's request order of this step (ModelRunnerOutput.req_ids).
    num_spec        rid -> draft rows scheduled for the decode group (k, or less at max_model_len).
    mm              adapter/mm.py MMStep (images to encode / rows taking span embeddings / cache drops, per rank), or
                    None: a step without image work.
    """
    step: int
    batches: list
    row_tokens: list
    prev_tokens: dict = field(default_factory=dict)
    new_reqs: list = field(default_factory=list)
    finished: set = field(default_factory=set)
    preempted: set = field(default_factory=set)
    order: list = field(default_factory=list)
    num_spec: dict = field(default_factory=dict)
    est_us: float = 0.0
    plan: Optional[Any] = None
    mm: Optional[Any] = None


class GrammarRows(NamedTuple):
    """The grammar bitmask of one structured-output request for this step: rows[i] masks the logits of position i of
    its decode group (i = 0 .. num_spec); drafts beyond `acceptable` must be rejected (vLLM GrammarOutput
    num_acceptable_drafts: the drafts the grammar did not accept were padded to -1 by the scheduler).
    A NamedTuple: `g.rows` / `g.acceptable`, or `rows, acceptable = g` (serve's StepContext.grammar convention)."""
    rows: Any                          # np.ndarray int32 [num_spec + 1, ceil(vocab / 32)]
    acceptable: int
