"""ds41h component 'prologue' (flag DS41H_PROLOGUE; skinny drop-in additionally DS41H_SKINNY, default on).

Patched stock symbols (vLLM 0.32.0):
  vllm.model_executor.layers.quantization.modelopt.ModelOptLinearMethod.process_weights_after_loading
      keeps a skinny-packed copy of the checkpoint FP8 weight of the DeepSeek-V4.1 decode GEMMs
      (fused_wqa_wkv, wq_b, indexer.wq_b, wo_b, shared_experts.gate_up_proj/down_proj) before the stock
      sm90 kernel (Humming) repacks it, and allocates the prologue scratch (ds41h/skinny.py).
  vllm.model_executor.layers.quantization.modelopt.ModelOptLinearMethod.apply
      routes 2-D bf16 inputs with 1..8 rows of those layers to the skinny sm90a GEMM.
  vllm.v1.worker.gpu.model_runner.GPUModelRunner.load_model
      after loading, ds41h.prologue.prepare_all(model) caches each decoder layer's kernel arguments
      (no allocation or copy can then happen during CUDA graph capture).

The fused prologue itself is not hooked here: the 'decoder' component's DeepseekV4DecoderLayer.forward
patch calls ``ds41h.prologue.run_prologue(...)`` when ``ds41h.prologue.eligible(...)`` (see
ds41h/prologue.py for the exact contract) and the stock mhc_shifted_post_pre + projections otherwise.
"""

import sys

from ds41h import flags
from ds41h.patches import register

MODELOPT_MODULE = "vllm.model_executor.layers.quantization.modelopt"
RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"
ATTENTION_MODULE = "vllm.models.deepseek_v41.attention"

if flags.PROLOGUE:

    @register(MODELOPT_MODULE)
    def patch_modelopt(module) -> None:
        from ds41h import skinny
        from ds41h.patch_core import mark_active

        skinny.install(module)
        mark_active("prologue", "ModelOptLinearMethod.process_weights_after_loading(capture)")
        if skinny.APPLY:
            mark_active("prologue", "ModelOptLinearMethod.apply(skinny M<=8)")

    @register(RUNNER_MODULE)
    def patch_runner(module) -> None:
        """Prepare every decoder layer's prologue arguments right after the weights are loaded (eager,
        before memory profiling and CUDA graph capture)."""
        from ds41h.patch_core import mark_active

        runner = module.GPUModelRunner
        base_load = runner.load_model

        def load_model(self, *args, **kwargs):
            result = base_load(self, *args, **kwargs)
            model = getattr(self, "model", None)
            if model is not None:
                import logging

                from ds41h import prologue

                try:
                    n = prologue.prepare_all(model)
                    logging.getLogger("vllm.ds41h.prologue").info("ds41h prologue: %d decoder layers prepared", n)
                except Exception:  # noqa: BLE001 - layers stay on the stock path
                    logging.getLogger("vllm.ds41h.prologue").exception("ds41h prologue: prepare_all failed")
            return result

        runner.load_model = load_model
        mark_active("prologue", "GPUModelRunner.load_model(prepare layers)")

    import os as _os

    if _os.environ.get("DS41H_PROLOGUE_INDEXER", "0").strip().lower() not in ("0", "", "false", "no", "off"):

        @register(ATTENTION_MODULE)
        def patch_indexer(module) -> None:
            """DS41H_PROLOGUE_INDEXER: DeepseekV4Indexer._wq_b_proj returns the idx_q the ds41h prologue computed for this
            indexer in the current forward (set by the decoder glue as indexer.__dict__["_ds41h_idx_q"], popped here or by
            the glue); otherwise the stock projection."""
            from ds41h.patch_core import mark_active

            cls = module.DeepseekV4Indexer
            if getattr(cls, "_ds41h_wq_b_patched", False):
                return
            stock = cls._wq_b_proj

            def _wq_b_proj(self, qr, qr_scale=None):
                pre = self.__dict__.pop("_ds41h_idx_q", None)
                if isinstance(pre, tuple):  # (qr tensor of the prologue launch, idx_q): only for that very qr (run_prologue)
                    src, pre = pre
                    if src is not qr:
                        pre = None
                if pre is not None and qr_scale is None and getattr(qr, "shape", (None,))[0] == pre.shape[0]:
                    return pre
                return stock(self, qr, qr_scale)

            _wq_b_proj.__wrapped__ = stock  # type: ignore[attr-defined]
            cls._wq_b_proj = _wq_b_proj
            cls._ds41h_wq_b_patched = True
            mark_active("prologue", "DeepseekV4Indexer._wq_b_proj(prologue idx_q)")

        if ATTENTION_MODULE in sys.modules:
            patch_indexer(sys.modules[ATTENTION_MODULE])

        INDEXER_OP_MODULE = "vllm.model_executor.layers.sparse_attn_indexer"
        _fold = (_os.environ.get("DS41H_PROLOGUE_INDEXER_W", "0").strip().lower() not in ("0", "", "false", "no", "off")
                 and _os.environ.get("DS41H_PROLOGUE_INDEXER_FOLD", "0").strip().lower() not in ("0", "", "false", "no", "off"))
        if _fold:

            @register(ATTENTION_MODULE)
            def patch_indexer_fold(module) -> None:
                """DS41H_PROLOGUE_INDEXER_FOLD: DeepseekV4Indexer.forward_q returns the prologue's (q_fp8, None, weights_out) for the
                qr tensor of the launch that computed them (= stock fused_indexer_q_rope_quant, bit-reproduced); otherwise stock."""
                from ds41h.patch_core import mark_active

                cls = module.DeepseekV4Indexer
                if getattr(cls, "_ds41h_fold_patched", False):
                    return
                stock_q = cls.forward_q

                def forward_q(self, qr, qr_scale, indexer_weights, positions, rotary_emb):
                    pend = self.__dict__.pop("_ds41h_fold", None)
                    if pend is not None and pend[0] is qr and qr_scale is None:
                        self.__dict__.pop("_ds41h_idx_q", None)
                        return pend[1], None, pend[2]
                    return stock_q(self, qr, qr_scale, indexer_weights, positions, rotary_emb)

                forward_q.__wrapped__ = stock_q  # type: ignore[attr-defined]
                cls.forward_q = forward_q
                cls._ds41h_fold_patched = True
                mark_active("prologue", "DeepseekV4Indexer.forward_q(prologue q rope/quant fold)")

            @register(INDEXER_OP_MODULE)
            def patch_indexer_op_fold(module) -> None:
                """DS41H_PROLOGUE_INDEXER_FOLD: SparseAttnIndexer.forward_cuda passes skip_topk_buffer_clear=True when q_quant is the
                prologue's q of this forward (the prologue cleared topk_indices_buffer[:m] after its PDL wait)."""
                from ds41h.patch_core import mark_active

                cls = module.SparseAttnIndexer
                if getattr(cls, "_ds41h_fold_patched", False):
                    return
                stock_fc = cls.forward_cuda
                enc = module._encode_layer_name

                def forward_cuda(self, hidden_states, q_quant, k, weights):
                    fq = self.__dict__.pop("_ds41h_fold_q", None)
                    if fq is None or q_quant is not fq:
                        return stock_fc(self, hidden_states, q_quant, k, weights)
                    import torch

                    return torch.ops.vllm.sparse_attn_indexer(  # = stock forward_cuda (vLLM 0.32.0) with skip_topk_buffer_clear=True
                        hidden_states, enc(self.k_cache.prefix), self.k_cache.kv_cache, q_quant, None, k, weights,
                        self.quant_block_size, self.scale_fmt, self.topk_tokens, self.head_dim, self.max_model_len,
                        self.max_total_seq_len, self.topk_indices_buffer, self.skip_k_cache_insert, self.use_pcp,
                        self.pcp_shard_decode_requests, enc(self.dense_mha_metadata_layer_name), self.use_fp4_cache, self.dcp_rank,
                        self.dcp_world_size, self.cp_kv_cache_interleave_size, True, candidate_blocks=self.candidate_blocks,
                        candidate_block_size=self.candidate_block_size, candidate_write=self.candidate_write,
                        topk_backend=self.topk_backend)

                forward_cuda.__wrapped__ = stock_fc  # type: ignore[attr-defined]
                cls.forward_cuda = forward_cuda
                cls._ds41h_fold_patched = True
                mark_active("prologue", "SparseAttnIndexer.forward_cuda(skip clear after the prologue fold)")

            if ATTENTION_MODULE in sys.modules:
                patch_indexer_fold(sys.modules[ATTENTION_MODULE])
            if INDEXER_OP_MODULE in sys.modules:
                patch_indexer_op_fold(sys.modules[INDEXER_OP_MODULE])

    # The import hook only sees modules imported after it is installed; patch already-imported ones now.
    if MODELOPT_MODULE in sys.modules:
        patch_modelopt(sys.modules[MODELOPT_MODULE])
    if RUNNER_MODULE in sys.modules:
        patch_runner(sys.modules[RUNNER_MODULE])
