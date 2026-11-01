"""ds41h 'prefill' component: sparse-MLA prefill without head padding (imported from ds41h/patch_prefill.py).

Patched stock symbols (vLLM 0.32.0), switch DS41H_PREFILL_ATTN (default ON):
  vllm.models.deepseek_v41.nvidia.flashmla.DeepseekV4FlashMLAAttention._forward_prefill
      wrapped to publish the running layer (n_local_heads) to the dispatcher below.
  vllm.models.deepseek_v41.nvidia.flashmla.flash_mla_sparse_fwd   (module global used by _forward_prefill)
      8/16/32 local heads -> FlashMLA e2afa5f SM90 small-head kernel on q[:, :H] / attn_sink[:H] /
      out[:, :H] (vLLM PR #61615 + FlashMLA PR #26, built by ds41h/prefill_attn.py into
      $DS41H_HOME/build/flashmla_sm90); anything else -> stock vendored FlashMLA (padded 64 heads).
  with DS41H_PREFILL_ATTN_NOPAD=1 also DeepseekV4FlashMLAAttention._alloc_attn_out /
      _fused_qnorm_rope_kv_insert: no q / output head padding in prefill-only steps that run fully eager
      (cudagraph runtime mode NONE, no capture open; PIECEWISE / breakable / decoder-replay graphs keep it).
The attn component's _forward_decode patch on the same class is independent of this one.
"""

import sys

from ds41h import prefill_attn
from ds41h.patches import register

FLASHMLA_MODULE = "vllm.models.deepseek_v41.nvidia.flashmla"

if prefill_attn.ATTN:

    @register(FLASHMLA_MODULE)
    def patch_flashmla_prefill(module) -> None:
        from ds41h.patch_core import mark_active

        prefill_attn.install_flashmla_prefill(module)
        mark_active("prefill", "DeepseekV4FlashMLAAttention._forward_prefill(small-head sparse prefill)")
        if prefill_attn.NOPAD:
            mark_active("prefill", "DeepseekV4FlashMLAAttention._alloc_attn_out/_fused_qnorm_rope_kv_insert(no pad)")

    if FLASHMLA_MODULE in sys.modules:  # imported before the hook was installed
        patch_flashmla_prefill(sys.modules[FLASHMLA_MODULE])
