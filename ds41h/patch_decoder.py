"""Import-time patches for the ds41h decoder layer kernel (flag DS41H_DECODER, see ds41h/decoder.py).

Patched stock vLLM 0.32.0 symbols:
  vllm.models.deepseek_v41.nvidia.flashmla
    DeepseekV4FlashMLAAttention._o_proj      -> returns the raw attention buffer while
                                                ds41h.decoder.raw_attention_output(attn) is active
  vllm.models.deepseek_v41.nvidia.model
    DeepseekV4DecoderLayer.__init__          -> records model dir / config on the layer
    DeepseekV4DecoderLayer.forward           -> ds41h decode path (below)
    DeepseekV41LLMForCausalLM.process_weights_after_loading -> ds41h.decoder.prepare_model
    supports_mhc_all_reduce                  -> False (the kernels need plain tensors between layers;
                                                on sm90 it is already False, model.py:696-701)
  vllm.models.deepseek_v41.nvidia.vl_model
    DeepseekV41ForCausalLM.process_weights_after_loading -> ds41h.decoder.prepare_model(language_model)
      The served architecture is this multimodal wrapper (registry.py:373-376). Its load_weights runs the
      child's process_weights_after_loading (and so prepare_model) BEFORE the per-module quant finalize
      that sets up the ds41h raw experts, and its own process_weights_after_loading -- the only call the
      loader makes after that finalize (model_loader/utils.py:202-204) -- returns early once
      _weights_finalized is set (vl_model.py:319-338). Without this patch prepare_model never sees the
      raw experts and the decoder silently stays off.

Decode forward (decode-only step, M <= max tokens; anything else runs the stock forward):
  attention half: ds41h.decoder.attention_side hook if set; else the ds41h prologue when
    ds41h.decoder.prologue_eligible (flag DS41H_PROLOGUE: ds41h/prologue.py run_prologue, then the
    stock compressor / indexer / sparse attention on its a / qr / kv / q); else stock
    model.py:477-591 verbatim (mhc_stream None, no sequence parallel). It yields the raw attention
    output o (pre wo_a) plus residual, post_mix, res_mix, attn_pre, previous_aux.
  post-attention half: the decoder kernel (flag DS41H_DECODER, model.py:597-625 in one launch), or
    stock _o_proj + mhc_shifted_post_pre + MoE when only the prologue is enabled.
Module globals (mhc_*) are looked up at call time, so other components patching them are honoured.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType

from ds41h import flags
from ds41h.patches import register

# DS41H_PROLOGUE_FIRST=1 (opt-in): the target's FIRST layer (residual None,
# x = the [M, 5120] embedding) also takes the ds41h prologue in model-entry mode (prologue.run_prologue(..., first_layer=True)
# = stock's broadcast + mhc_pre with hc_attn_fn_broadcast; test_prologue_first_layer bit-identical) instead of the stock
# six-kernel entry path (~33 us in the drafter's window trace -> one ~21 us launch). Target numerics: needs the EXACT gate.
PROLOGUE_FIRST = os.environ.get("DS41H_PROLOGUE_FIRST", "0") == "1"

MODEL = "vllm.models.deepseek_v41.nvidia.model"
FLASHMLA = "vllm.models.deepseek_v41.nvidia.flashmla"
VL_MODEL = "vllm.models.deepseek_v41.nvidia.vl_model"


def _patch_flashmla(module: ModuleType) -> None:
    attn_cls = module.DeepseekV4FlashMLAAttention
    if getattr(attn_cls, "_ds41h_decoder_patched", False):
        return
    base_o_proj = attn_cls._o_proj

    def _o_proj(self, attn_out, positions):
        if getattr(self, "_ds41h_raw_o", False):
            return attn_out
        return base_o_proj(self, attn_out, positions)

    attn_cls._o_proj = _o_proj
    attn_cls._ds41h_decoder_patched = True


def _stock_attention_side(module, layer, x, positions, input_ids, pre_mix, post_mix, res_mix, residual, engram_hashes, engram_mask, capture_previous_aux):
    """Stock DeepseekV4DecoderLayer.forward up to the attention output (model.py:477-591) with
    mhc_stream None, no sequence parallel, fuse_mhc_all_reduce False; attention returns raw o."""
    from ds41h import decoder

    previous_aux = None
    mhc_pre = module.mhc_pre_delayed_tilelang
    hc_args = (layer.hc_attn_scale, layer.hc_attn_base, layer.rms_norm_eps, layer.hc_eps, layer.hc_eps, layer.hc_post_alpha, layer.hc_sinkhorn_iters)
    norm = dict(norm_weight=layer.attn_norm.weight, norm_eps=layer.attn_norm.variance_epsilon)
    if residual is None:
        if x.dim() == 2:
            # First layer: embedding broadcast to hc copies, identity pre-mix (model.py:496-514).
            assert layer.hc_attn_fn_broadcast is not None
            residual = x.unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
            post_mix, res_mix, x, attn_pre = mhc_pre(residual, layer.hc_attn_fn_broadcast, *hc_args, x=x, **norm)
        else:
            residual = x
            post_mix, res_mix, x, attn_pre = mhc_pre(residual, layer.hc_attn_fn, *hc_args, pre_mix=pre_mix, **norm)
    elif layer.engram is not None and engram_hashes is not None:
        # Engram layers (1, 14): previous post, Engram injection, pre (model.py:531-561).
        previous_post = module.mhc_post_tilelang(x, residual, post_mix, res_mix)
        if capture_previous_aux:
            previous_aux = previous_post.mean(dim=1)
        residual = layer.engram(previous_post, engram_hashes[:, layer.engram.layer_hash_index], engram_mask)
        post_mix, res_mix, x, attn_pre = mhc_pre(residual, layer.hc_attn_fn, *hc_args, pre_mix=pre_mix, **norm)
    else:
        residual, post_mix, res_mix, x, attn_pre, aux = module.mhc_shifted_post_pre(
            x, residual, post_mix, res_mix, layer.hc_attn_fn, *hc_args, pre_mix=pre_mix, capture_aux=capture_previous_aux, stream=None,
            reduce_results=False, **norm,
        )  # fmt: skip
        if capture_previous_aux:
            previous_aux = aux
    with decoder.raw_attention_output(layer.attn):
        o = layer.attn(positions, x, None)
    return o, residual, post_mix, res_mix, attn_pre, previous_aux


def _side_stream(attn):
    """Aux stream for DS41H_ATTN_SIDE (stock DeepseekV4Attention.aux_stream_list[0]); None = serial (flag off, no aux
    streams on this layer, or a breakable-graph capture, which maybe_execute_in_parallel handles itself)."""
    if not flags.ATTN_SIDE:
        return None
    streams = getattr(attn, "aux_stream_list", None)
    events = getattr(attn, "ln_events", None)
    if not streams or streams[0] is None or not events or len(events) < 2:
        return None
    return streams[0]


def _prologue_attention_side(module, layer, x, positions, input_ids, pre_mix, post_mix, res_mix, residual, engram_hashes, engram_mask, capture_previous_aux,
                             first_layer=False):
    """ds41h prologue (mHC post/pre + norms + q/kv projections + q/kv RoPE + SWA insert) followed by
    the rest of stock DeepseekV4Attention._prepare_and_attn (attention.py:832-914): compressor on
    kv_score = a @ fused_wkv_wgate^T (fp32, :954-960), indexer on (qr, latent, weights_proj(a)),
    compressed-cache insert, then sparse indexer + MLA into the stock-shaped attention buffer."""
    import torch
    import torch.nn.functional as F
    from vllm.forward_context import get_forward_context

    from ds41h import prologue

    attn = layer.attn
    # DS41H_P1A_EXT (prologue-attn, opt-in): the previous layer kernel already produced this layer's P1a (see forward)
    p1a_done = prologue.take_p1a_pending(layer, x.device, x.shape[0]) if prologue.P1A_EXT and not first_layer else None
    out = prologue.run_prologue(layer, x, residual, post_mix, res_mix, pre_mix, positions, get_forward_context().attn_metadata,
                                aux=bool(capture_previous_aux) and not first_layer, p1a_done=p1a_done, first_layer=first_layer)
    residual = out["residual"]
    # Stock capture_aux: bf16 mean over the post-mapped bf16 streams (tilelang_kernels.py:449-458; the
    # overlap-stream path is literally residual.mean(dim=1), ops/mhc.py:260). One reduce kernel (fp32
    # accumulation inside) instead of float() + mean + to(bf16); with DS41H_PROLOGUE_AUX the prologue's P1a jobs
    # emit it (out["aux"], prologue-attn opt-in: no extra kernel).
    previous_aux = (out["aux"] if "aux" in out else residual.mean(dim=1)) if capture_previous_aux and not first_layer else None  # stock: None at entry
    a, qr, kv, q = out["a"], out["qr"], out["kv"], out["q"]
    latent = None
    index_q = index_q_scale = index_weights = None
    # DS41H_PROLOGUE_INDEXER (prologue-attn, opt-in): the prologue already computed indexer.weights_proj(a) (idx_w) and
    # indexer.wq_b(qr) (idx_q, handed to the stock indexer through the patched DeepseekV4Indexer._wq_b_proj).
    idx_w = out.get("idx_w") if attn.indexer is not None else None
    if idx_w is not None:
        attn.indexer.__dict__["_ds41h_idx_q"] = out["idx_q"]
    aux = _side_stream(attn)
    if aux is not None and attn.compressor is not None:
        # DS41H_ATTN_SIDE: stock _prepare_and_attn's stream overlap restored on the ds41h prologue path.
        # Same kernels and dependencies as the serial code below: aux = kv_score GEMM + compressor (latent), main =
        # indexer weights_proj; join; then main = indexer (needs latent) while aux = compressed-cache insert; join
        # before the sparse indexer / attention, which read both caches (stock attention.py:866-904 does the same).
        from vllm.utils.multi_stream_utils import maybe_execute_in_parallel

        comp, idx = attn.compressor, attn.indexer

        def compress():
            kv_score = torch.mm(a, comp.fused_wkv_wgate.weight.T, out_dtype=torch.float32)
            return comp(kv_score, positions)

        weights, latent = maybe_execute_in_parallel(
            (lambda: idx_w if idx_w is not None else idx.weights_proj(a)[0]) if idx is not None else (lambda: None), compress,
            attn.ln_events[0], attn.ln_events[1], aux)
        if idx is not None:
            (index_q, index_q_scale, index_weights), _ = maybe_execute_in_parallel(
                lambda: idx(qr, latent, weights, positions, attn.indexer_rotary_emb, None),
                lambda: comp.insert_cache(latent, positions, attn.rotary_emb),
                attn.ln_events[0], attn.ln_events[1], aux)
        else:
            comp.insert_cache(latent, positions, attn.rotary_emb)
    else:
        if attn.compressor is not None:
            kv_score = torch.mm(a, attn.compressor.fused_wkv_wgate.weight.T, out_dtype=torch.float32)
            latent = attn.compressor(kv_score, positions)
        if attn.indexer is not None:
            weights = idx_w if idx_w is not None else attn.indexer.weights_proj(a)[0]
            index_q, index_q_scale, index_weights = attn.indexer(qr, latent, weights, positions, attn.indexer_rotary_emb, None)
        if attn.compressor is not None:
            attn.compressor.insert_cache(latent, positions, attn.rotary_emb)
    if idx_w is not None:
        attn.indexer.__dict__.pop("_ds41h_idx_q", None)  # not consumed when the stock indexer took its short-context path
    if flags.DECODE_NOPAD and "_ds41h_stock_forward_decode" in type(attn).__dict__:
        # The ds41h sparse decode kernel reads / writes heads 0..15 only and the decoder kernel / stock
        # o_proj read attn_out[:, :n_local_heads]: no zero-padded q / output (DS41H_DECODE_NOPAD). A stock
        # FlashMLA fallback re-pads inside the patched _forward_decode (ds41h/attn.py).
        attn_out = torch.empty(a.shape[0], q.shape[1], q.shape[2], dtype=q.dtype, device=q.device)
    else:
        attn_out = attn._alloc_attn_out(a.shape[0], a)
    if attn_out.shape[1] != q.shape[1]:
        # Stock zero-pads q to padded_heads (attention.py:1058-1078); forward_mqa asserts equal shapes.
        q = F.pad(q, (0, 0, 0, attn_out.shape[1] - q.shape[1]))
    attn._sparse_indexer_and_attn(a, index_q, index_q_scale, index_weights, q, kv, positions, attn_out)
    return attn_out, residual, out["post_mix"], out["res_mix"], out["attn_pre"], previous_aux


# ---------------------------------------------------------------------------------------------- path log
# One-time log (TP rank 0) of which path every decoder layer takes, per distinct per-model signature:
#   "stock" (stock forward), "pro+dec" (ds41h prologue + decoder kernel), "stockpre+dec" (stock mHC /
#   projections + decoder kernel), "pro+stockffn" (prologue + stock post-attention half); "/attn" marks
#   that the layer's decode attention ran the ds41h sparse decode kernel (DS41H_ATTN) in that forward.
# Layers of one forward arrive in increasing layer id; a new forward (or a different row count) flushes.
_PATH_LOG = flags.ENABLED and flags._flag("DS41H_PATH_LOG")
_path_cur: dict = {"key": None, "last": -1, "layers": {}}
_path_seen: set[str] = set()


def _compress(paths: dict[int, str]) -> str:
    out, run = [], None
    for lid in sorted(paths):
        p = paths[lid]
        if run is not None and run[2] == p and run[1] == lid - 1:
            run[1] = lid
        else:
            if run is not None:
                out.append(run)
            run = [lid, lid, p]
    if run is not None:
        out.append(run)
    return " ".join(f"{a}{'' if a == b else f'-{b}'}:{p}" for a, b, p in out)


def _flush_paths() -> None:
    cur = _path_cur
    if cur["key"] is None or not cur["layers"]:
        return
    m, decode_only, capturing = cur["key"]
    sig = f"{'decode' if decode_only else 'mixed/prefill'} {_compress(cur['layers'])}"
    small = decode_only and m <= 8
    key = f"M={m} {sig}" if m <= 16 else sig  # small row counts (c1 verify / draft steps) logged per M
    if key not in _path_seen:
        _path_seen.add(key)
        from ds41h.patch_core import logger

        logger.info("ds41h paths (rank 0, M=%d%s%s): %s", m, ", capture" if capturing else "", "" if small else " [M>8: fused paths not expected]", sig)
    cur["layers"] = {}


def _note_path(layer, positions, path: str) -> None:
    try:
        from ds41h.patch_core import current_step, tp_rank

        if tp_rank() != 0:
            return
        attn = layer.attn
        used = attn.__dict__.pop("_ds41h_attn_used", None)
        if used:
            path += "/attn"
        lid = int(getattr(attn, "layer_id", -1))
        info = current_step()
        key = (int(positions.shape[0]), bool(info is not None and info.decode_only), bool(info is not None and info.capturing))
        cur = _path_cur
        if key != cur["key"] or lid <= cur["last"]:
            _flush_paths()
            cur["key"] = key
        cur["last"] = lid
        cur["layers"][lid] = path
    except Exception:  # noqa: BLE001 - logging must never break a forward
        pass


_P1A_LAYERS: dict = {}  # DS41H_P1A_EXT: decoder layers by prefix (next-layer lookup)


def _p1a_next(layer):
    """The next decoder layer of the same model (prefix '<...>.layers.<i>' -> '<...>.layers.<i+1>') if it can run the fused
    prologue (non-Engram, prepared), else None."""
    from ds41h import prologue

    pre = getattr(layer, "_ds41h_prefix", None)
    if not pre:
        return None
    head, _, idx = pre.rpartition(".")
    if not idx.isdigit():
        return None
    nxt = _P1A_LAYERS.get(f"{head}.{int(idx) + 1}")
    if nxt is None or getattr(nxt, "engram", None) is not None or prologue.prepare_layer(nxt) is None:
        return None
    return nxt


def _patch_model(module: ModuleType) -> None:
    layer_cls = module.DeepseekV4DecoderLayer
    if getattr(layer_cls, "_ds41h_decoder_patched", False):
        return
    base_init = layer_cls.__init__
    base_forward = layer_cls.forward

    def __init__(self, vllm_config, prefix, *args, **kwargs):
        base_init(self, vllm_config, prefix, *args, **kwargs)
        self._ds41h_model_dir = vllm_config.model_config.model
        self._ds41h_cfg = vllm_config.model_config.hf_config
        self._ds41h_prefix = prefix
        _P1A_LAYERS[prefix] = self

    def forward(
        self, x, positions, input_ids, pre_mix=None, post_mix=None, res_mix=None, residual=None, engram_hashes=None, engram_mask=None, *,
        capture_previous_aux=False, mega_gate_metadata=None,
    ):  # fmt: skip
        from ds41h import decoder

        fused = decoder.eligible(self, x, positions)
        pro = decoder.prologue_eligible(self, x, residual, positions)
        first = False
        if PROLOGUE_FIRST and not pro and residual is None and flags.PROLOGUE and getattr(x, "dim", lambda: 0)() == 2:
            from ds41h import prologue

            # model-entry layer: the prologue in first_layer mode (prologue.eligible checks M <= 8, dtype, no Engram / SP)
            first = pro = decoder.decode_step(self, positions, flags.MAX_DECODE_TOKENS) and prologue.eligible(self, x, None, positions, first_layer=True)
        if not (fused or pro):
            out = base_forward(
                self, x, positions, input_ids, pre_mix, post_mix, res_mix, residual, engram_hashes, engram_mask,
                capture_previous_aux=capture_previous_aux, mega_gate_metadata=mega_gate_metadata,
            )  # fmt: skip
            if _PATH_LOG:
                _note_path(self, positions, "stock")
            return out
        args = (self, x, positions, input_ids, pre_mix, post_mix, res_mix, residual, engram_hashes, engram_mask, capture_previous_aux)
        if decoder.attention_side is not None:
            side = decoder.attention_side(*args)
        elif pro:
            side = _prologue_attention_side(module, *args, first_layer=first)
        else:
            side = _stock_attention_side(module, *args)
        o, residual, post_mix, res_mix, attn_pre, previous_aux = side
        if fused:
            import torch

            from ds41h import prologue

            p1a = nxt = None
            if prologue.P1A_EXT and residual.dim() == 3 and 0 < residual.shape[0] <= prologue.MAX_TOKENS:
                # DS41H_P1A_EXT (prologue-attn + decoder XF_P1A, opt-in): this layer kernel also produces the next layer's
                # P1a (post-mapped residual + collapse into the prologue scratch); the next prologue skips it
                nxt = _p1a_next(self)
                if nxt is not None:
                    p1a = (torch.empty_like(residual, memory_format=torch.contiguous_format), *prologue.p1a_buffers(residual.device))
            ffn_kw = {"p1a": p1a} if p1a is not None else {}
            x, residual, post_mix, res_mix, ffn_pre = decoder.ffn(self, o, positions, residual, post_mix, res_mix, attn_pre, **ffn_kw)
            if p1a is not None:
                prologue.set_p1a_pending(residual.device, nxt, p1a[0])
            if _PATH_LOG:
                _note_path(self, positions, ("pro" if pro else "stockpre") + "+dec")
            return x, residual, post_mix, res_mix, ffn_pre, previous_aux
        # Prologue only: stock post-attention half (model.py:591-625 with mhc_stream None).
        x = self.attn._o_proj(o, positions)
        residual, post_mix, res_mix, x, ffn_pre, _ = module.mhc_shifted_post_pre(
            x, residual, post_mix, res_mix, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self.rms_norm_eps, self.hc_eps,
            self.hc_eps, self.hc_post_alpha, self.hc_sinkhorn_iters, pre_mix=attn_pre, norm_weight=self.ffn_norm.weight,
            norm_eps=self.ffn_norm.variance_epsilon, stream=None, reduce_results=False,
        )  # fmt: skip
        x = self.ffn(x, input_ids, mega_gate_metadata)
        if _PATH_LOG:
            _note_path(self, positions, "pro+stockffn")
        return x, residual, post_mix, res_mix, ffn_pre, previous_aux

    layer_cls.__init__ = __init__
    layer_cls.forward = forward
    layer_cls._ds41h_decoder_patched = True

    lm_cls = module.DeepseekV41LLMForCausalLM
    base_pwal = lm_cls.process_weights_after_loading

    def process_weights_after_loading(self) -> None:
        base_pwal(self)
        from ds41h import decoder

        decoder.prepare_model(self)

    lm_cls.process_weights_after_loading = process_weights_after_loading
    # The fused all-reduce + mHC path defers the MoE finalize into the next layer; the ds41h
    # kernels need plain reduced tensors between layers.
    module.supports_mhc_all_reduce = lambda vllm_config: False


def _patch_vl_model(module: ModuleType) -> None:
    """Run prepare_model after the per-module quant finalize (see the module docstring). Collective:
    every TP rank's loader calls the wrapper hook exactly once; prepare_model is idempotent (a
    second call after a successful one only re-binds moved expert tensors)."""
    vl_cls = module.DeepseekV41ForCausalLM
    if getattr(vl_cls, "_ds41h_decoder_patched", False):
        return
    base_pwal = vl_cls.process_weights_after_loading

    def process_weights_after_loading(self) -> None:
        base_pwal(self)
        lm = getattr(self, "language_model", None)
        if lm is not None:
            from ds41h import decoder

            decoder.prepare_model(lm)

    vl_cls.process_weights_after_loading = process_weights_after_loading
    vl_cls._ds41h_decoder_patched = True


# The prologue component (ds41h/prologue.py) relies on this forward patch as its entry point.
if flags.DECODER or flags.PROLOGUE:
    register(FLASHMLA)(_patch_flashmla)
    register(MODEL)(_patch_model)
    register(VL_MODEL)(_patch_vl_model)
    # The import hook only sees modules imported after it is installed; patch already-imported ones.
    if FLASHMLA in sys.modules:
        _patch_flashmla(sys.modules[FLASHMLA])
    if MODEL in sys.modules:
        _patch_model(sys.modules[MODEL])
    if VL_MODEL in sys.modules:
        _patch_vl_model(sys.modules[VL_MODEL])
