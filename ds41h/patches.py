"""Registry of import-time patches, keyed by stock vLLM 0.32.0 module name.

Each component registers its patch functions in its own module under ds41h/patch_*.py and is
listed in COMPONENTS below. Patch functions must import torch/CUDA code lazily (inside the
function body) so that plain interpreter start-up stays cheap. Every component is gated by its
own flag in ds41h.flags so it can be A/B tested in isolation.
"""

from collections import defaultdict
from collections.abc import Callable
from importlib import import_module
from types import ModuleType

type Patch = Callable[[ModuleType], None]

PATCHES: dict[str, list[Patch]] = defaultdict(list)


def register(module_name: str) -> Callable[[Patch], Patch]:
    def deco(fn: Patch) -> Patch:
        PATCHES[module_name].append(fn)
        return fn

    return deco


# Component patch modules; each one calls register(...) at import time if its flag is on.
COMPONENTS = (
    "ds41h.patch_core",
    "ds41h.patch_allreduce",
    "ds41h.patch_candidates",
    "ds41h.patch_attn",
    "ds41h.patch_prologue",
    "ds41h.patch_decoder",
    "ds41h.patch_sampling",
    "ds41h.patch_prefill",
    "ds41h.patch_engram",  # opt-in DS41H_ENGRAM_HOST=1
    "ds41h.patch_serving",  # opt-in DS41H_FAST_TOKENIZE=1 (API-server tokenization)
    "ds41h.patch_glue",  # opt-in DS41H_PREP_FUSE=1 (eager decode input prep)
)

for _name in COMPONENTS:
    try:
        import_module(_name)
    except ModuleNotFoundError as e:  # component not written yet
        if e.name != _name:
            raise
