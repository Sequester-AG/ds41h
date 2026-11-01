"""DSpark context work of an eager prefill step on the replay rows only .

DS41H_PREFILL_CTX_TRIM (default 1; 0 = stock) or the prefill control key "ctx_trim" = 0/1, read per
step, with DS41H_PREFILL_CTL. Same-server A/B on the certified cert350k7p config (350 W, ab_ctx2): PP c16 16K 41,494 -> 42,032
(+1.3%, 4 pairs, each +1.0..1.8%); TTFT and decode unchanged; drafter cache bytes and target numerics unchanged by construction.

After an eager prefill step with the decoder SWA bounded replay (stock vLLM 0.32.0 decoder_replay_layers.py
DecoderReplayLayers._run, ds41h prefill_sp._replay_sp), every output of the replay layers is ZERO outside the replay rows
(each request's last <= window rows): `out.new_zeros((num_tokens, ...)).index_copy_(0, rows, out)` -- "The trimmed rows'
outputs stay zero; nothing reads them". The DSpark drafter's inputs are such outputs (the target aux hidden states of
layers >= the cut, checked per model), but its propose still projects ALL T rows:
    main_x = main_norm(main_proj(cat(aux)))                            humming 5120 x 15360, ~4.2 ms per 16K rows (350 W)
    per drafter layer: kv = kv_norm(fused_wqa_wkv(main_x)[:, q_lora:])   humming 1792 x 5120, ~0.5 ms per 16K rows, x 3
    _insert_context_kv(kv) for every row (RoPE + quant + cache write)
main_proj and fused_wqa_wkv have no bias and RMSNorm(0) = 0, so the trimmed rows' main_x and context KV are exactly zero
in stock (a later prefix-cache hit in the middle of the prompt reads those zeros today). Here main_proj / main_norm and
fused_wqa_wkv / kv_norm run on the replay rows only, every other row is written as zero, and the insert runs over all
rows as stock -> the same main_x and the same cache bytes (up to the sign of zero; the replay rows' GEMMs run with a
smaller M = possibly another tile / split-K order: drafter-only, allowed). The target model is untouched.
No host sync: the rows are the step's replay-batch row tensor (its length is host-known).
DFlash propose stores the context KV of rows [0, T) eagerly, or -- when the drafter's query step runs as a FULL graph -- the
first num_reqs * (k + 1) rows inside the graph and rows [num_context, T) eagerly: the eager call then gets a SUFFIX slice of
main_x; its replay rows are found by offset (replay rows before the slice go to a dump row, no sync). A pending early
main_proj result of the drafter (decode steps) is used when it matches this step (the drafter's combine runs), and dropped
when stale, exactly as the drafter's combine does.

Validity per step: the target forward clears the note at its start (DeepseekV4Model.forward wrapper); the replay notes
(rows, T) when it runs with a replay batch and every aux hidden-state layer of the model is a replay layer; the drafter
wrappers use it only for inputs with exactly T rows, outside CUDA-graph capture, without a pending early main_proj.
"""

from __future__ import annotations

import logging
import os

from ds41h import flags, prefill_ctl

logger = logging.getLogger("vllm.ds41h.prefill")

CTX_TRIM = flags.ENABLED and flags.PREFILL and flags._flag("DS41H_PREFILL_CTX_TRIM", "1")
MIN_T = int(os.environ.get("DS41H_PREFILL_CTX_TRIM_MIN_T", "512"))
# installed when switched on, or when the runtime control file may switch it on
INSTALL = CTX_TRIM or (flags.ENABLED and flags.PREFILL and bool(prefill_ctl.PATH))

_NOTE: dict = {}  # {"rows": LongTensor [R] (device), "T": int} of the current target forward's replay, or empty
STATS = {"combine": 0, "kv": 0, "skipped": 0}


def enabled() -> bool:
    return bool(int(prefill_ctl.get("ctx_trim", int(CTX_TRIM)))) if prefill_ctl.PATH else CTX_TRIM


def begin_forward() -> None:
    _NOTE.clear()


def _aux_in_replay(model) -> bool:
    """Every aux hidden-state layer of the target model is a replay layer (its rows outside the replay are zero)."""
    ok = model.__dict__.get("_ds41h_ctx_aux_ok")
    if ok is None:
        aux = list(getattr(model, "aux_hidden_state_layers", ()) or ())
        start = getattr(model, "decoder_replay_start", None)
        ok = bool(aux) and start is not None and all(int(i) >= int(start) for i in aux)
        model._ds41h_ctx_aux_ok = ok
    return ok


def note_replay(model, rows, num_tokens: int) -> None:
    """Called by the replay scatter of a target forward (rows = replay_batch.rows, num_tokens = T of the forward)."""
    if not INSTALL:
        return
    if model is not None and _aux_in_replay(model) and 0 < rows.shape[0] < num_tokens:
        _NOTE["rows"], _NOTE["T"] = rows, int(num_tokens)
    else:
        _NOTE.clear()


def rows_for(x):
    """The noted replay rows if the trim applies to the drafter input x ([T, ...]), else None."""
    rows = _NOTE.get("rows")
    if rows is None or x.shape[0] != _NOTE.get("T") or x.shape[0] < MIN_T or not enabled():
        return None
    import torch

    if rows.device != x.device or (x.is_cuda and torch.cuda.is_current_stream_capturing()):
        return None
    return rows


def rows_for_ctx(x):
    """For a context slice x = combine_output[T - n : T] (DFlash propose: rows [0, T) eagerly, or with a FULL drafter graph
    rows [num_context, T) eagerly and the first num_context rows inside the graph): (src, dst) such that
    x.index_select(0, src) are the replay rows inside the slice and dst their positions in an (n + 1)-row buffer whose
    last row is a dump row for replay rows before the slice (dst is None when the slice starts at row 0). No host sync."""
    rows = _NOTE.get("rows")
    T, n = _NOTE.get("T", 0), x.shape[0]
    if rows is None or n > T or n < MIN_T or not enabled():
        return None
    import torch

    if rows.device != x.device or (x.is_cuda and torch.cuda.is_current_stream_capturing()):
        return None
    start = T - n
    if start == 0:
        return rows, None
    rel = rows - start
    return rel.clamp(min=0), torch.where(rel >= 0, rel, torch.full_like(rel, n))


# ------------------------------------------------------------------------------------------------- patches
def _drafter_early_ctx_kv(model, main_x) -> None:
    """DRAFTER fix (mixed-step drafts): with DS41H_DRAFTER_CTXKV_EARLY the captured FULL drafter graph inserts the
    context KV of its first num_context (<= 256) rows from a buffer that the drafter-patched combine_hidden_states fills
    (patch_drafter.fill_ctx_kv). The trimmed combine above bypasses that combine, so in every trimmed eager step (prefill
    rows in the batch, T >= 512) the graph inserted the PREVIOUS step's KV at this step's first context slots (the decode
    requests' newly verified positions): drafts proposed in such steps accepted 1.90 vs 2.88 tokens/step (measured).
    Fill the buffer from this step's trimmed main_x (rows outside the replay are zero -> zero KV = the trimmed
    stock semantics). Drafter cache only; the target model is untouched."""
    import sys

    pd = sys.modules.get("ds41h.patch_drafter")
    if pd is not None and getattr(pd, "CTXKV_EARLY", False):
        pd.fill_ctx_kv(model, main_x)


def patch_model(module) -> None:
    """DeepseekV4Model.forward: clear the replay note at the start of every target forward (idempotent)."""
    cls = module.DeepseekV4Model
    if getattr(cls.forward, "_ds41h_ctx_begin", False):
        return
    base = cls.forward

    def forward(self, *args, **kwargs):
        _NOTE.clear()
        return base(self, *args, **kwargs)

    forward._ds41h_ctx_begin = True  # type: ignore[attr-defined]
    forward.__wrapped__ = base  # type: ignore[attr-defined]
    cls.forward = forward


def patch_replay(module) -> None:
    """DecoderReplayLayers._run (stock replay scatter, non-SP steps): note the replay rows (idempotent)."""
    cls = module.DecoderReplayLayers
    if getattr(cls._run, "_ds41h_ctx_note", False):
        return
    base = cls._run

    def _run(self, hidden_states, *states):
        out = base(self, hidden_states, *states)
        rb = self.replay_batch
        if rb is not None:
            note_replay(getattr(self.run_layers, "__self__", None), rb.rows, int(hidden_states.shape[0]))
        return out

    _run._ds41h_ctx_note = True  # type: ignore[attr-defined]
    _run.__wrapped__ = base  # type: ignore[attr-defined]
    cls._run = _run


def patch_dspark(module) -> None:
    """DSparkDeepseekV4Model.combine_hidden_states / precompute_and_store_context_kv: replay rows only (idempotent).
    Wraps whatever the class has at patch time (stock or the drafter's patched methods); the trimmed path is the
    stock math of vllm/models/deepseek_v41/nvidia/dspark.py on the replay rows."""
    cls = module.DSparkDeepseekV4Model
    if getattr(cls, "_ds41h_ctx_trim", False):
        return
    base_combine = cls.combine_hidden_states
    base_pre = cls.precompute_and_store_context_kv
    insert = module._insert_context_kv

    def combine_hidden_states(self, aux_hidden_states):
        rows = rows_for(aux_hidden_states)
        pend = self.__dict__.get("_ds41h_pending")
        if rows is not None and pend is not None:
            if pend.get("rows") == aux_hidden_states.shape[0] and pend.get("cols") == aux_hidden_states.shape[1]:
                rows = None  # the drafter's early main_proj result of this step applies: its combine consumes it
            else:
                self.__dict__.pop("_ds41h_pending", None)  # stale: the drafter's combine drops it here as well
        if rows is None:
            return base_combine(self, aux_hidden_states)
        sub = self.main_norm(self.main_proj(aux_hidden_states.index_select(0, rows)))
        STATS["combine"] += 1
        if STATS["combine"] in (1, 100, 1000):
            logger.info("ds41h prefill ctx trim: main_proj on %d of %d rows (%d trimmed combines)", rows.shape[0],
                        aux_hidden_states.shape[0], STATS["combine"])
        out = sub.new_zeros((aux_hidden_states.shape[0], *sub.shape[1:])).index_copy_(0, rows, sub)
        _drafter_early_ctx_kv(self, out)
        return out

    def precompute_and_store_context_kv(self, main_x, context_positions, context_slot_mappings=None):
        sel = rows_for_ctx(main_x) if context_slot_mappings is not None else None
        if sel is None:
            return base_pre(self, main_x, context_positions, context_slot_mappings)
        import torch

        src, dst = sel
        with torch.inference_mode():
            n = main_x.shape[0]
            xs = main_x.index_select(0, src)
            for i, layer in enumerate(self.layers):
                slot_mapping = context_slot_mappings[i]
                if slot_mapping is None:
                    continue
                attn = layer.attn
                qr_kv, _ = attn.fused_wqa_wkv(xs)
                kv_r = attn.kv_norm(qr_kv[..., attn.q_lora_rank :])
                if dst is None:
                    kv = kv_r.new_zeros((n, *kv_r.shape[1:])).index_copy_(0, src, kv_r)
                else:  # replay rows before the slice land in the dump row n (stored by the drafter graph instead)
                    kv = kv_r.new_zeros((n + 1, *kv_r.shape[1:])).index_copy_(0, dst, kv_r)[:n]
                insert(attn, kv, context_positions, slot_mapping)
        STATS["kv"] += 1
        if STATS["kv"] in (1, 100, 1000):
            logger.info("ds41h prefill ctx trim: context KV of %d replay rows, slice of %d rows (%d trimmed steps)", src.shape[0], n,
                        STATS["kv"])

    combine_hidden_states.__wrapped__ = base_combine  # type: ignore[attr-defined]
    precompute_and_store_context_kv.__wrapped__ = base_pre  # type: ignore[attr-defined]
    cls.combine_hidden_states = combine_hidden_states
    cls.precompute_and_store_context_kv = precompute_and_store_context_kv
    cls._ds41h_ctx_trim = True
