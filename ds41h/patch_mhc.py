"""ds41h 'prefill' component: fused mHC sublayer boundaries for prefill (imported from ds41h/patch_prefill.py).

Patched stock symbols (vLLM 0.32.0), switch DS41H_PREFILL_MHC (default ON, impl 1),
see ds41h/mhc_prefill.py:
  vllm.models.deepseek_v41.nvidia.model.mhc_shifted_post_pre        (module global, looked up at call time)
  vllm.models.deepseek_v41.nvidia.model.mhc_pre_delayed_tilelang    (module global; Engram-layer pre)
  vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model.finalize_mhc_broadcast_weights (load-time fn packing)
Eligible calls (>= DS41H_PREFILL_MHC_MIN_T rows, plain tensors, no overlap stream / fused all-reduce, not
capturing a CUDA graph) run ds41h/kernels/mhc_prefill.cu; everything else runs the stock function.
"""

import sys

from ds41h import mhc_prefill
from ds41h.patches import register

MODEL_MODULE = "vllm.models.deepseek_v41.nvidia.model"

if mhc_prefill.MHC:

    @register(MODEL_MODULE)
    def patch_mhc(module) -> None:
        from ds41h.patch_core import mark_active

        mhc_prefill.install(module)
        mark_active("prefill", f"mhc_shifted_post_pre/mhc_pre_delayed_tilelang(fused boundary, T>={mhc_prefill.MIN_T})")

    if MODEL_MODULE in sys.modules:  # imported before the hook was installed
        patch_mhc(sys.modules[MODEL_MODULE])
