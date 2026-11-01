"""ds41h 'prefill' component: dense FP8 linears + wo_a for prefill (imported from ds41h/patch_prefill.py).

Patched stock symbols (vLLM 0.32.0), each behind its own switch (see ds41h/linear.py for details):
  LOSSLESS: DS41H_PREFILL_WOA16=1, DS41H_PREFILL_WQA_SHARD=1 (default ON) / DS41H_PREFILL_LINEAR16=1 (default OFF: stock
  Humming W8A16 is as fast)
      vllm.model_executor.layers.quantization.modelopt.ModelOptLinearMethod.process_weights_after_loading
          LINEAR16: keep a raw copy of the DeepSeek-V4.1 dense FP8 linears (checkpoint e4m3 + 32x32 ue8m0) before
          the stock sm90 kernel (Humming) repacks them; warm DeepGEMM's bf16 GEMM. WOA16: warm the wo_a group GEMMs.
      vllm.model_executor.layers.quantization.modelopt.ModelOptLinearMethod.apply
          LINEAR16: inputs with >= DS41H_PREFILL_LINEAR_MIN_M rows -> exact bf16 dequant + bf16 GEMM (W8A16 math).
      vllm.models.deepseek_v41.nvidia.ops.o_proj.deep_gemm_fp8_o_proj (module global used by dsv41_o_proj)
          WOA16: >= MIN_M tokens with stock's bf16 wo_a -> in-place bf16 inverse RoPE + strided bf16 group GEMMs.
      WQA_SHARD (DS41H_PREFILL_WQA_SHARD=1, default ON, end-to-end): the replicated fused_wqa_wkv of every
          backbone layer runs on ceil(M / tp) rows per rank (stock kernel) + an all-gather along the token dim.
  LOSSY (default OFF; DS41H_LOSSY=1 or explicit): DS41H_PREFILL_LINEAR=1 / DS41H_PREFILL_WOA=1
      ModelOptLinearMethod.process_weights_after_loading / apply as above, but W8A8: 128x128-folded e4m3 copy +
          act_quant_128 + DeepGEMM fp8_gemm_nt (impl deepgemm), or Humming MXFP8 activations (impl humming);
          WOA converts the bmm wo_a to FP8 einsum weights instead of the stock BF16 emulation dequant.
  DS41H_PREFILL_WOA_ROPEQ=1 (default with WOA)
      vllm.models.deepseek_v4.nvidia.ops.o_proj.fused_inv_rope_fp8_quant (module global): >= MIN_M tokens ->
          ds41h CUDA inverse RoPE + 1x128 FP8 quant (fp8_linear.cu), same output layout / numerics.
  DS41H_PREFILL_MOE_FP8ACT=1 (LOSSY, default OFF)
      vllm.models.deepseek_v41.quant_config.DeepseekV4FP8Config.get_quant_method
          port of vLLM PR #61875: the stock Mxfp4MoEMethod (Humming) runs its activations as block FP8.
They require DS41H=1 and DS41H_PREFILL != 0.
"""

import sys

from ds41h import linear
from ds41h.patches import register

MODELOPT_MODULE = "vllm.model_executor.layers.quantization.modelopt"
QUANT_CONFIG_MODULE = "vllm.models.deepseek_v41.quant_config"

if linear.LINEAR or linear.WOA or linear.LINEAR16 or linear.WOA16 or linear.WQA_SHARD:

    @register(MODELOPT_MODULE)
    def patch_modelopt_linear(module) -> None:
        from ds41h.patch_core import mark_active

        linear.install_modelopt(module)
        if linear.LINEAR16:
            mark_active("prefill", f"ModelOptLinearMethod(lossless W8A16: exact bf16 dequant + {linear.GEMM16} bf16 GEMM, M>={linear.MIN_M})")
        if linear.WQA_SHARD:
            mark_active("prefill", f"ModelOptLinearMethod(lossless token-sharded fused_wqa_wkv + all-gather, M>={linear.WQA_SHARD_MIN_M})")
        if linear.LINEAR:
            mark_active("prefill", f"ModelOptLinearMethod(LOSSY W8A8 {linear.IMPL}, M>={linear.MIN_M})")
        if linear.WOA:
            mark_active("prefill", "ModelOptLinearMethod(LOSSY wo_a FP8 einsum)")

    if MODELOPT_MODULE in sys.modules:  # imported before the hook was installed
        patch_modelopt_linear(sys.modules[MODELOPT_MODULE])

O_PROJ_MODULE = "vllm.models.deepseek_v4.nvidia.ops.o_proj"
O_PROJ41_MODULE = "vllm.models.deepseek_v41.nvidia.ops.o_proj"

if linear.WOA16:

    @register(O_PROJ41_MODULE)
    def patch_o_proj16(module) -> None:
        from ds41h.patch_core import mark_active

        linear.install_o_proj16(module)
        mark_active("prefill", f"o_proj.deep_gemm_fp8_o_proj(lossless bf16 wo_a: in-place inverse RoPE + {linear.WOA16_GEMM}, M>={linear.MIN_M})")

    if O_PROJ41_MODULE in sys.modules:
        patch_o_proj16(sys.modules[O_PROJ41_MODULE])

if linear.ROPEQ:

    @register(O_PROJ_MODULE)
    def patch_o_proj(module) -> None:
        from ds41h.patch_core import mark_active

        linear.install_o_proj(module)
        mark_active("prefill", f"o_proj.fused_inv_rope_fp8_quant(ds41h CUDA, M>={linear.MIN_M})")

    if O_PROJ_MODULE in sys.modules:
        patch_o_proj(sys.modules[O_PROJ_MODULE])

if linear.MOE_FP8ACT:

    @register(QUANT_CONFIG_MODULE)
    def patch_moe_fp8_activations(module) -> None:
        from ds41h.patch_core import mark_active

        linear.install_moe_fp8_activations(module)
        mark_active("prefill", "DeepseekV4FP8Config.get_quant_method(MoE block-FP8 activations, #61875)")

    if QUANT_CONFIG_MODULE in sys.modules:
        patch_moe_fp8_activations(sys.modules[QUANT_CONFIG_MODULE])
