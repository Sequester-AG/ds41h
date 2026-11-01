"""SAMPLING hook: vocab-parallel DSpark drafting + block verification (ds41h.vp_sampling).

Patched symbols (stock vLLM 0.32.0):
  * vllm.v1.worker.gpu.spec_decode.dspark.speculator.DSparkSpeculator.load_draft_model (:84)
      -> builds the per-rank VPShard (local markov_w2 slice, local draft-logit cache) when eligible.
  * DSparkSpeculator._sample_sequential (:156) -> vp_sampling.draft when the shard exists.
  * vllm.v1.worker.gpu.spec_decode.rejection_sampler.RejectionSampler.__call__ (:407)
      -> all-gathers a vocab-parallel draft-logit cache before running the stock sampler (fallback path).
  * vllm.v1.worker.gpu.sample.sampler.Sampler.add_request (:96)
      -> records per request whether sampling is "plain" (only temperature; no top-k/p, min-p,
         penalties, logit bias / allowed ids / min-tokens stop masking, bad words, thinking budget).
  * vllm.v1.worker.gpu.model_runner.GPUModelRunner.sample (:1592)
      -> vocab-parallel verification when every request in the batch is plain and the batch has drafts,
         no grammar, no logprobs, no batch sharder, block verification without synthetic rates /
         watermarking / adaptive verification; otherwise the stock path (full-vocab logits gather).
  * GPUModelRunner.sample_tokens (:2088) -> stashes this step's aux hidden states so the patched sample() can start the
    DSpark main_proj on a side stream after the verification head GEMV (DS41H_DRAFTER_EARLY_MAIN=1).
Glue fusion in the vocab-parallel sample(): gather_draft_sampled + temperature/seed staging in one launch, run on the side
stream concurrently with the head GEMV (DS41H_SAMPLING_FUSE_GLUE=1, DS41H_SAMPLING_EARLY_GLUE=1); stock
get_num_sampled_and_rejected folded into the rejection kernel. Outputs identical to stock (tests/test_sampling.py).
All eligibility decisions use only configuration and batch metadata identical on every TP rank, so all
ranks take the same path (the vocab-parallel path issues LL collectives).
"""

from __future__ import annotations

import logging

from ds41h import flags
from ds41h.patches import register

logger = logging.getLogger("ds41h.sampling")
# DS41H_SAMPLING_FUSE_GLUE=1: stock gather_draft_sampled + parameter staging in one launch, stock
# get_num_sampled_and_rejected folded into the rejection kernel (2 launches fewer per verify step; same outputs)
FUSE_GLUE = flags._flag("DS41H_SAMPLING_FUSE_GLUE", "1")
# DS41H_DRAFTER_EARLY_MAIN=1: start the DSpark main_proj (+ main_norm) of this step on a side stream right after the
# verification head GEMV, overlapping the latency-bound verification chain (ds41h.patch_drafter.early_main_proj)
EARLY_MAIN = flags._flag("DS41H_DRAFTER_EARLY_MAIN", "1")
# DS41H_SAMPLING_EARLY_GLUE=1: run gather_and_stage on the side stream concurrently with the verification head GEMV
EARLY_GLUE = flags._flag("DS41H_SAMPLING_EARLY_GLUE", "0")
# DS41H_DRAFTER_PRECAT=1: the early main_proj's torch.cat(aux hidden states) is issued on the side stream BEFORE the verification
# head GEMV (it only depends on the target forward), so the side stream starts main_proj right at the head's end event
# (stock order: cat after the head event, ~4 us on the side chain that gates propose). Same values; drafter schedule only.
PRECAT = EARLY_MAIN and flags._flag("DS41H_DRAFTER_PRECAT", "0")
# DS41H_GRAPH_OUT_BATCH=1 (opt-in; target decode-graph tail -> correctness gate): vLLM's FULL decode graph ends with
# store_capture_output's copies of the model outputs into the graph manager's persistent buffers (hidden + 3 aux hidden
# states = 4 serialized copy kernels, ~8 us in situ between the target's final RMSNorm and the verification head). During
# a FULL capture the outputs are written into those same buffers by ONE torch._foreach_copy_ (multi-tensor kernel) and the
# returned tensors alias them, so the 4 stock copies become same-data no-ops (aten copy_ exits early). Same bytes in the
# same buffers; eager / PIECEWISE / ubatched runs and anything that does not match the stock layout stay stock.
GRAPH_OUT_BATCH = flags._flag("DS41H_GRAPH_OUT_BATCH", "0")
GOUT_STATS = {"batched": 0, "skipped": 0}


def patch_speculator(module) -> None:
    from ds41h.patch_core import mark_active

    cls = module.DSparkSpeculator
    base_load = cls.load_draft_model
    base_sample = cls._sample_sequential

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = base_load(self, target_model, target_attn_layer_names)
        self._ds41h_vp = None
        from ds41h import vp_sampling

        self._ds41h_vp = vp_sampling.shard_drafts(self, model)
        if vp_sampling.VERIFY_HEAD_MODE == 2 and self._ds41h_vp is not None:
            try:
                from ds41h import vhead

                vhead.build_for(target_model)  # at load: the packed copy is part of the KV-cache memory profile
            except Exception:  # noqa: BLE001
                import logging

                logging.getLogger("ds41h.sampling").exception("ds41h vhead: packing failed; stock verification head")
        return model

    def _sample_sequential(self, num_reqs, head_hidden):
        if getattr(self, "_ds41h_vp", None) is None:
            return base_sample(self, num_reqs, head_hidden)
        from ds41h import vp_sampling

        return vp_sampling.draft(self, num_reqs, head_hidden)

    cls.load_draft_model = load_draft_model
    cls._sample_sequential = _sample_sequential
    mark_active("sampling", "DSparkSpeculator.{load_draft_model,_sample_sequential}")


def patch_rejection_sampler(module) -> None:
    from ds41h.patch_core import mark_active

    cls = module.RejectionSampler
    base_call = cls.__call__

    def __call__(self, logits, input_batch, draft_logits=None, invalid_drafts=None):
        if draft_logits is not None:
            from ds41h.vp_sampling import LOCAL_DRAFT_LOGITS

            if draft_logits.data_ptr() in LOCAL_DRAFT_LOGITS:
                from vllm.distributed import tensor_model_parallel_all_gather

                draft_logits = tensor_model_parallel_all_gather(draft_logits, dim=-1)
        return base_call(self, logits, input_batch, draft_logits, invalid_drafts)

    cls.__call__ = __call__
    mark_active("sampling", "RejectionSampler.__call__")


def patch_sampler(module) -> None:
    import numpy as np

    from ds41h.patch_core import mark_active

    cls = module.Sampler
    base_add = cls.add_request

    def add_request(self, req_idx, sampling_params):
        base_add(self, req_idx, sampling_params)
        plain = getattr(self, "_ds41h_plain", None)
        if plain is None:
            plain = self._ds41h_plain = np.zeros(self.needs_logits_processing.shape[0], dtype=bool)
        plain[req_idx] = _is_plain(self, req_idx)

    cls.add_request = add_request
    mark_active("sampling", "Sampler.add_request")


def _is_plain(sampler, req_idx: int) -> bool:
    """True if the request's logits processing is at most a temperature (state already written by stock)."""
    from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
    from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
    from vllm.v1.worker.gpu.sample.penalties import PenaltiesState

    procs = sampler.logits_processors
    if len(procs) != 3 or type(procs[0]) is not LogitBiasState or type(procs[1]) is not PenaltiesState or type(procs[2]) is not BadWordsState:
        return False  # custom logits processors
    bias, penalties, bad_words = procs
    if bias.use_logit_bias[req_idx] or penalties.use_penalty[req_idx] or bad_words.num_bad_words.np[req_idx] != 0:
        return False
    budget = sampler.thinking_budget_state
    if budget.enabled and budget.use_thinking_budget[req_idx]:
        return False
    states = sampler.sampling_states
    return bool(states.min_p.np[req_idx] == 0.0 and states.top_k.np[req_idx] == states.vocab_size and states.top_p.np[req_idx] == 1.0)


def patch_model_runner(module) -> None:
    import numpy as np

    from ds41h.patch_core import mark_active

    cls = module.GPUModelRunner
    base_sample = cls.sample

    def vp_eligible(self, input_batch, grammar_output) -> bool:
        spec = getattr(self, "speculator", None)
        shard = getattr(spec, "_ds41h_vp", None)
        if shard is None or grammar_output is not None or self.batch_sharder is not None:
            return False
        if not hasattr(self.model, "compute_logits_local"):
            return False
        if input_batch.num_reqs == 0 or input_batch.num_draft_tokens == 0:
            return False
        rs, sampler = self.rejection_sampler, self.sampler
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

        if rs is None or sampler is None or type(rs) is not RejectionSampler or rs.sampler is not sampler:
            return False
        if not rs.use_block_verification or rs.synthetic_conditional_rates is not None or rs.watermark_key is not None:
            return False
        if rs.enable_adaptive_verification or rs.num_speculative_steps != shard.n_spec:
            return False
        if sampler.compute_nans or sampler.use_fp64_gumbel or sampler.return_sampling_mask:
            return False
        if input_batch.logits_indices.shape[0] > shard.max_logits:
            return False
        idx = input_batch.idx_mapping_np
        plain = getattr(sampler, "_ds41h_plain", None)
        if plain is None or not np.all(plain[idx]):
            return False
        trace = sampler.trace_replay_state
        if trace is not None and np.any(trace.trace_len.np[idx] > 0):
            return False
        from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS

        return sampler.sampling_states.max_num_logprobs(idx) == NO_LOGPROBS

    def sample(self, hidden_states, input_batch, grammar_output):
        if not vp_eligible(self, input_batch, grammar_output):
            return base_sample(self, hidden_states, input_batch, grammar_output)
        from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
        from vllm.v1.worker.gpu.sample.output import SamplerOutput
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import gather_draft_sampled

        from ds41h import vp_sampling

        shard = self.speculator._ds41h_vp
        sampler = self.sampler
        prefill_len = sampler.req_states.prefill_len.gpu
        temperature = sampler.sampling_states.temperature.gpu
        seeds = sampler.sampling_states.seeds.gpu
        early = None
        if FUSE_GLUE and EARLY_GLUE and shard.verify_fused is None:
            # gather + parameter staging (a host-mapped read) do not depend on the logits: run them on the side stream
            # concurrently with the verification head GEMV
            import torch

            from ds41h import patch_drafter

            cur = torch.cuda.current_stream()
            side = patch_drafter.side_stream(cur.device)
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                ds_, pos_ = vp_sampling.gather_and_stage(
                    shard,
                    input_batch.input_ids,
                    input_batch.positions,
                    input_batch.logits_indices,
                    input_batch.expanded_idx_mapping,
                    input_batch.expanded_local_pos,
                    prefill_len,
                    temperature,
                    seeds,
                )
                noise = None
                if vp_sampling.NOISE_EARLY:
                    # residual-resampling Gumbel noise (seed / position / token only), off the critical path
                    noise = vp_sampling.residual_noise(shard, pos_, input_batch.expanded_idx_mapping, input_batch.logits_indices.shape[0])
                ev_g = torch.cuda.Event()
                ev_g.record(side)
            early = (ds_, pos_, ev_g, cur, noise)
        precat = None
        aux0 = getattr(self, "_ds41h_aux", None)
        if PRECAT and isinstance(aux0, (list, tuple)) and len(aux0) > 0 and getattr(self, "pcp_manager", None) is None:
            from ds41h import patch_drafter

            precat = patch_drafter.early_cat(self.speculator, aux0)  # side stream, overlaps the head GEMV
        logits = vp_sampling.verify_logits(self.model, hidden_states, input_batch.logits_indices)
        if vp_sampling.VERIFY_HEAD and getattr(self, "_ds41h_force_installed", False):
            # exactness gate (ds41h/patch_force.py wraps model.compute_logits_local, which the ds41h verify head bypasses):
            # the same capture of the forced requests' rows of this rank's logits shard
            cap = getattr(self, "_ds41h_force_cap", None)
            if cap is not None and "rows" not in cap:
                cap["kind"] = "local"
                cap["width"] = int(logits.shape[-1])
                cap["rows"] = {i: logits[lo:hi].detach().clone() for i, lo, hi in cap["want"]}
        aux = getattr(self, "_ds41h_aux", None)
        if EARLY_MAIN and isinstance(aux, (list, tuple)) and len(aux) > 0 and getattr(self, "pcp_manager", None) is None:
            import torch

            from ds41h import patch_drafter

            ev = torch.cuda.Event()
            ev.record()  # after the head GEMV: the side stream overlaps the verification chain, not the HBM-bound head
            self._ds41h_early = patch_drafter.early_main_proj(self.speculator, aux, after_event=ev, precat=precat)
        fused_path = shard.verify_fused is not None and shard.verify_fused.eligible(logits, input_batch.num_reqs)
        noise = None
        if early is not None:
            draft_sampled, pos, ev_g, cur, noise = early
            cur.wait_event(ev_g)
            draft_sampled.record_stream(cur)
            pos.record_stream(cur)
        elif FUSE_GLUE and not fused_path:
            # Placeholder drafts (before the prompt is fully sampled) become -1, exactly as stock
            # (rejection_sampler.py:418-425); the sampling-parameter staging rides in the same launch.
            draft_sampled, pos = vp_sampling.gather_and_stage(
                shard,
                input_batch.input_ids,
                input_batch.positions,
                input_batch.logits_indices,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                prefill_len,
                temperature,
                seeds,
            )
        else:
            draft_sampled, pos = gather_draft_sampled(
                input_batch.input_ids,
                input_batch.positions,
                input_batch.logits_indices,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                prefill_len,
            )
        # stock get_num_sampled_and_rejected runs over idx_mapping.shape[0] requests: fuse it only when that is num_reqs
        counts = (input_batch.seq_lens, prefill_len) if FUSE_GLUE and not fused_path and input_batch.idx_mapping.shape[0] == input_batch.num_reqs else None
        post = _post_state(self, input_batch) if counts is not None else None
        out = vp_sampling.verify(
            shard,
            logits,
            draft_sampled,
            pos,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            input_batch.num_reqs,
            temperature,
            seeds,
            staged=FUSE_GLUE and not fused_path,
            counts=counts,
            noise=noise,
            post=post,
        )
        if post is not None:
            self._ds41h_post_fused = out[0].data_ptr()  # postprocess_sampled skips the stock post_update for these tokens
        if counts is not None:
            sampled, num_sampled, num_rejected = out
        else:
            sampled, num_sampled = out
            num_sampled, num_rejected = get_num_sampled_and_rejected(
                num_sampled,
                input_batch.seq_lens,
                input_batch.cu_num_logits,
                input_batch.idx_mapping,
                prefill_len,
            )
        output = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        return output, num_sampled, num_rejected

    cls.sample = sample
    mark_active("sampling", "GPUModelRunner.sample")

    if vp_post_fuse():
        import hashlib
        import inspect

        base_post = cls.postprocess_sampled
        got = hashlib.sha1(inspect.getsource(base_post).encode()).hexdigest()[:16]
        if got != _POSTPROCESS_SHA:
            logger.warning("ds41h sampling: postprocess_sampled source %s != %s; DS41H_SAMPLING_POST_FUSE off", got, _POSTPROCESS_SHA)
            _POST["off"] = True
        else:

            def postprocess_sampled(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc=None):
                tag = self.__dict__.pop("_ds41h_post_fused", None)
                if tag is None or tag != sampled_tokens.data_ptr():
                    return base_post(self, idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc)
                # stock postprocess_sampled minus post_update (already applied by the rejection kernel of this step)
                self.model_state.postprocess_state(idx_mapping, num_sampled, self.req_states.num_computed_tokens.gpu)

            postprocess_sampled.__wrapped__ = base_post
            cls.postprocess_sampled = postprocess_sampled
            mark_active("sampling", "GPUModelRunner.postprocess_sampled(post_update fused into the rejection kernel)")

    base_sample_tokens = cls.sample_tokens

    def sample_tokens(self, *args, **kwargs):
        # stash this step's aux hidden states (execute_model_state is cleared inside) for the early main_proj
        st = getattr(self, "execute_model_state", None)
        self._ds41h_aux = getattr(st, "aux_hidden_states", None) if st is not None else None
        try:
            return base_sample_tokens(self, *args, **kwargs)
        finally:
            self._ds41h_aux = None
            spec = getattr(self, "speculator", None)
            if spec is not None and getattr(self, "_ds41h_early", False):
                from ds41h import patch_drafter

                patch_drafter.drop_early_main_proj(spec)  # normally consumed by propose already
            self._ds41h_early = False

    sample_tokens.__wrapped__ = base_sample_tokens  # source-hash guards (patch_tail) see the stock function
    if EARLY_MAIN:
        cls.sample_tokens = sample_tokens
        mark_active("sampling", "GPUModelRunner.sample_tokens(early main_proj)")


# sha1[:16] of inspect.getsource(GPUModelRunner.postprocess_sampled) in vLLM 0.32.0
_POSTPROCESS_SHA = "5b3c5af8986840b1"
_POST: dict = {"off": False}


def graph_out_alias(mgr, out):
    """DS41H_GRAPH_OUT_BATCH: copy a FULL-capture model output (hidden, [aux, ...]) into mgr.hidden_states /
    mgr.aux_hidden_states (the persistent buffers store_capture_output copies into) with one _foreach_copy_ and return
    views aliasing them (same storage offset / sizes / strides as the stock destination slices), or `out` unchanged."""
    import torch

    if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], torch.Tensor) and isinstance(out[1], (list, tuple))):
        return out
    h, aux = out
    buf_h = getattr(mgr, "hidden_states", None)
    buf_a = list(getattr(mgr, "aux_hidden_states", None) or [])
    if buf_h is None or len(aux) == 0 or len(buf_a) != len(aux):
        GOUT_STATS["skipped"] += 1
        return out
    n = h.shape[0]
    srcs, dsts = [h, *aux], [buf_h, *buf_a]
    for x, d in zip(srcs, dsts):
        if not (isinstance(x, torch.Tensor) and isinstance(d, torch.Tensor) and x.dtype == d.dtype and x.device == d.device
                and x.dim() == d.dim() and x.dim() >= 1 and x.shape[0] == n and d.shape[0] >= n and x.shape[1:] == d.shape[1:]
                and x.is_contiguous() and d.is_contiguous() and x.data_ptr() != d.data_ptr()):
            GOUT_STATS["skipped"] += 1
            return out
    views = [d[:n] for d in dsts]
    copied = False
    if len(views) <= 8 and all(v.nbytes % 16 == 0 and v.data_ptr() % 16 == 0 and x.data_ptr() % 16 == 0 for v, x in zip(views, srcs)):
        try:
            from ds41h import patch_meta

            patch_meta.ext().multi_copy(views, srcs)  # one launch, 4 KB per CTA
            copied = True
        except Exception:  # noqa: BLE001 - older meta extension: the multi-tensor-apply copy
            copied = False
    if not copied:
        torch._foreach_copy_(views, srcs)
    GOUT_STATS["batched"] += 1
    return views[0], views[1:]


def patch_graph_out(module) -> None:
    """ModelCudaGraphManager.capture: the target model's forward returns graph_out_alias(...) inside FULL captures."""
    from ds41h.patch_core import mark_active

    cls = getattr(module, "ModelCudaGraphManager", None)
    if cls is None or not hasattr(cls, "capture"):
        logger.info("ds41h graph-out batch: ModelCudaGraphManager.capture not found; stock output copies")
        return
    base_capture = cls.capture

    def capture(self, model, *args, **kwargs):
        if (not kwargs.get("use_aux_hidden_state_outputs", False) or not getattr(self, "is_last_pp_rank", False)
                or getattr(self, "ubatch_runner", None) is not None):
            return base_capture(self, model, *args, **kwargs)
        import torch
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available

        had = "forward" in model.__dict__
        prev = model.__dict__.get("forward")
        base_forward = model.forward
        mgr = self

        def forward(*a, **kw):
            out = base_forward(*a, **kw)
            # FULL capture = forward_fn(CUDAGraphMode.NONE) inside torch.cuda.graph(...); the PIECEWISE path runs with
            # mode PIECEWISE, the warm-up runs uncaptured
            if (torch.cuda.is_current_stream_capturing() and is_forward_context_available()
                    and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE):
                return graph_out_alias(mgr, out)
            return out

        model.forward = forward
        try:
            return base_capture(self, model, *args, **kwargs)
        finally:
            if had:
                model.forward = prev
            else:
                del model.forward
            logger.info("ds41h graph-out batch: %d FULL captures with one batched output copy, %d stock",
                        GOUT_STATS["batched"], GOUT_STATS["skipped"])

    cls.capture = capture
    mark_active("sampling", "ModelCudaGraphManager.capture(one batched output copy, DS41H_GRAPH_OUT_BATCH)")


def vp_post_fuse() -> bool:
    from ds41h import vp_sampling

    return vp_sampling.POST_FUSE and not _POST["off"]


def _post_state(runner, input_batch) -> dict | None:
    """The stock post_update state tensors for this step if the fused update is allowed, else None. Not in a step that
    carries a request of the exactness force hook (it rewrites that request's sampled tokens / counts after sample(); the
    hook sets runner._ds41h_force_cap for exactly those steps, so the other requests of an EXACT=1 server -- the bench,
    GSM8K -- run the fused update and are covered by the gate), prompt logprobs (read num_computed_tokens between sample()
    and postprocess_sampled), PP, or an aux-output connector."""
    if not vp_post_fuse() or getattr(runner, "_ds41h_force_cap", None) is not None:
        return None
    if getattr(runner, "pp_handler", None) is not None or getattr(runner, "aux_output_connector", None) is not None:
        return None
    if not getattr(runner, "is_last_pp_rank", True) or getattr(runner, "pcp_manager", None) is not None:
        return None
    plw = getattr(runner, "prompt_logprobs_worker", None)
    if plw is not None:
        import numpy as np

        if getattr(plw, "token_id_scores", None) or np.any(plw.uses_prompt_logprobs[input_batch.idx_mapping_np]):
            return None
    rs = runner.req_states
    return {
        "num_computed": rs.num_computed_tokens.gpu,
        "last_sampled": rs.last_sampled_tokens,
        "bins": runner.sampler.penalties_state.output_bin_counts,
        "all_token_ids": rs.all_token_ids.gpu,
        "total_len": rs.total_len.gpu,
        "query_start_loc": input_batch.query_start_loc,
    }


if flags.SAMPLING:
    register("vllm.v1.worker.gpu.spec_decode.dspark.speculator")(patch_speculator)
    register("vllm.v1.worker.gpu.spec_decode.rejection_sampler")(patch_rejection_sampler)
    register("vllm.v1.worker.gpu.sample.sampler")(patch_sampler)
    register("vllm.v1.worker.gpu.model_runner")(patch_model_runner)
if GRAPH_OUT_BATCH:
    register("vllm.v1.worker.gpu.cudagraph_utils")(patch_graph_out)
