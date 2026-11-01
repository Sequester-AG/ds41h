"""ds41h component 'attn' (flag DS41H_ATTN): sm90 16-head sparse MLA decode.

Patches ``vllm.models.deepseek_v41.nvidia.flashmla.DeepseekV4FlashMLAAttention._forward_decode``;
prefill, q padding and o-projection stay stock. See ds41h/attn.py.
"""

import sys

from ds41h import flags
from ds41h.patches import register

FLASHMLA_MODULE = "vllm.models.deepseek_v41.nvidia.flashmla"

if flags.ATTN:

    @register(FLASHMLA_MODULE)
    def patch_flashmla(module) -> None:
        from ds41h.attn import install_flashmla_patch

        install_flashmla_patch(module)

    # The import hook only sees modules imported after it is installed; patch an already-imported one now.
    if FLASHMLA_MODULE in sys.modules:
        patch_flashmla(sys.modules[FLASHMLA_MODULE])
