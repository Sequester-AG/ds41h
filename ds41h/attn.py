"""ds41h 'attn': sm90a sparse MLA decode attention for DeepSeek-V4.1 on stock vLLM 0.32.0.

Replaces the FlashMLA sparse decode call of ``DeepseekV4FlashMLAAttention._forward_decode``
(vllm/models/deepseek_v41/nvidia/flashmla.py:167-246). Stock pads the 16 local query heads to 64 for
FlashMLA's sm90 FP8 decode kernel (flashmla.py:66-74) and runs ``compute_global_topk_indices_and_lens``
plus the FlashMLA planner/split/combine; this path runs one kernel (ds41h/kernels/sparse_decode.cu) that
computes only the 16 real heads, reads the stock V4 ``fp8_ds_mla`` cache records (584 B/token) and the
stock decode metadata, and fuses the top-k block-table lookup.

Prefill, q padding (``get_padded_num_q_heads`` stays 64 so FlashMLA prefill is unchanged) and the
o-projection stay stock: the kernel reads heads 0..15 of the padded q and writes heads 0..15 of the padded
output, and ``dsv41_o_proj`` reads only ``attn_out[:, :n_local_heads]`` (nvidia/ops/o_proj.py:67).

CUDA graphs: the launch allocates nothing, uses only persistent metadata buffers (decode_swa_indices/lens,
topk_indices_buffer, token_to_req_indices, is_valid_token, block_table), runs on the current stream and
has no data-dependent host control flow, so FULL decode graphs capture it.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Any

import torch

from ds41h import cuext, flags

logger = logging.getLogger("vllm.ds41h.attn")  # child of vLLM's logger so its handlers print it

NUM_HEADS = 16  # real query heads per TP rank (64 / TP4)
HEAD_DIM = 512  # latent dim, K == V
V4_RECORD_BYTES = 584  # stock sm90 fp8_ds_mla record (models/deepseek_v41/attention.py:513)
MAX_KEYS = 640  # SWA (128, or 192 for DSpark draft rows) + top-k (512)
MAX_TOPK = 512


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


MAX_ROWS = int(os.environ.get("DS41H_ATTN_MAX_ROWS", "64"))  # decode rows per step handled here
CLUSTER = int(os.environ.get("DS41H_ATTN_CLUSTER", "0"))  # 0 = auto (16 if >256 keys and one wave, else 8)
SPREAD = _env_flag("DS41H_ATTN_SPREAD", "1")  # request >114 KB smem so cluster CTAs get one SM each
# Early mode: resolve the key indices and gather every key that this step does not write (all but the SWA slots
# in swa_metadata.slot_mapping) before griddepcontrol.wait, i.e. overlapped with the predecessor when the kernel
# is PDL-launched right behind the ds41h prologue. Valid because the index buffers / block tables / older KV
# records are produced by kernels that are never PDL-overlapped with this one (see sparse_decode.cu).
# Bit-identical to the regular mode (tests/test_attn.py::test_early_mode_matches). Default ON: validated
# end-to-end (KL) in i11b_allbest / i12_tree_defaults with DS41H_DECODE_NOPAD, where the attention directly follows the prologue
# on most layers (end-to-end-like m=6, v3 prologue: -4..-5 us/layer); DS41H_ATTN_EARLY=0 restores the old schedule.
EARLY = _env_flag("DS41H_ATTN_EARLY", "1")
MAX_FRESH = 64


# DS41H_ATTN_VARIANT=d (opt-in until validated): kernels/sparse_decode_d.cu -- same API / numerics (bit-identical outputs),
# smaller shared memory (<16,16> 115,080 B: unpadded score rows / exchange rows, 40-row K tile with a k8 PV tail) so that
# (a) with DS41H_ATTN_SPREAD_BYTES=116736 an attention CTA can share an SM with ONE prologue CTA (early-mode CTAs start on
# the SMs of exited G1 CTAs and their pre-wait gather overlaps the prologue: -1.9 us/layer end-to-end-like at 350 W, m=6), and
# (b) 16-CTA clusters packed two CTAs per SM are used when the spread ones do not fit in one wave (m = 8 rows at k=7, which
# otherwise falls back to 8-CTA clusters with 80 keys per CTA). DS41H_ATTN_PACK=0 disables (b).
VARIANT = os.environ.get("DS41H_ATTN_VARIANT", "").strip().lower()

# DS41H_ATTN_OFFTAB=1 (opt-in until validated end-to-end; bit-identical): the key offsets of a decode step's rows are resolved once per index
# group. The attention of an index-source layer (layer.indexer is not None: its indexer just rewrote the shared top-k buffer)
# resolves them as usual and stores them in a per-device table (kernel offmode 1); the attentions of the following layers that
# read the same top-k buffer, SWA indices / lens, token metadata, block table, fresh slots and cache page geometry (all compared
# here) load them (offmode 2): one independent 16-B load per key before the old-key gather instead of the index -> block-table
# load chain + valid-count barriers (the attention's pre-wait chain, which follows the prologue's end at k=7). Anything that
# breaks the pairing (an index-source attention on the stock path or rejected by the launcher, other metadata) drops the entry.
OFFTAB = _env_flag("DS41H_ATTN_OFFTAB", "0") and VARIANT != "d"
_offtab: dict[int, torch.Tensor] = {}  # device -> int64 [MAX_ROWS * MAX_KEYS * 2] ({data, scale} offset per (row, key))
_offtab_key: dict[int, tuple] = {}  # device -> resolve key of the table's current contents


TRACE = bool(os.environ.get("DS41H_PA_TRACE_DUMP", "").strip())  # in-situ stamps (see ds41h/prologue.py TRACE_DUMP)

# FlashMLA tile-scheduler plan sharing (opt-in DS41H_TILE_SHARE=1 -- EXPERIMENTAL, NOT VALIDATED:
# the check mode below saw different per-group plans in startup / dummy forwards and no gain in situ (ab1q); keep off; steps with more than
# flags.MAX_DECODE_TOKENS decode rows only, i.e. the stock FlashMLA fallback of multi-request decode -- the <= 8-row ds41h path
# never reaches it). vLLM builds one DeepseekSparseSWAMetadata per SWA KV-cache group (DSv4.1: 8 groups of 5 round-robin
# layers) and each build() creates fresh FlashMLASchedMeta per layer type, so FlashMLA's planner (get_mla_metadata_kernel, one
# CTA: ~48 us at 128 decode rows, ~92 us at 512) runs once per (group, type) = 18x per step instead of once per type (3x), the
# sharing vLLM intends (sparse_swa.py: "shared across all ~60 layers of that type within a decode step"). The plan depends only
# on the shapes (checked by flash_mla_with_kvcache on every reuse) and on topk_length / extra_topk_length, which are the same
# for every group of a type (same rows / seq_lens / top-k counts; vLLM already reuses one plan across the layers of a type
# with different index sources inside a group) -> identical plans, identical outputs. On the first fallback call of a forward,
# every group's metadata gets the first group's still-unplanned FlashMLASchedMeta per type (all objects are fresh per build).
# DS41H_TILE_SHARE=2 = CHECK mode (debug; eager forwards only, e.g. mixed prefill/decode steps): nothing is shared; at the
# next forward the plans every group computed for the previous one are compared per type (torch.equal on
# tile_scheduler_metadata / num_splits); the counts of identical / differing plans are logged for the first 3 and then every
# 25 checked forwards -- the evidence that sharing changes no output.
TILE_SHARE = int(os.environ.get("DS41H_TILE_SHARE", "0") or 0)
_TILE_ATTRS = ("tile_sched_swaonly", "tile_sched_c4a", "tile_sched_c128a", "tile_sched_c1a", "tile_sched_c2a")
_tile_prev: list = []  # CHECK mode: (attr, [FlashMLASchedMeta of every group]) of the previous checked forward
_tile_stats = {"forwards": 0, "same": 0, "diff": 0, "unplanned": 0}
_tile_rows = [0]  # CHECK mode: decode rows of the checked forward (logged)


def _tile_check(mds: list) -> None:
    global _tile_prev
    if torch.cuda.is_current_stream_capturing():
        _tile_prev = []
        return
    for attr, metas in _tile_prev:
        planned = [m for m in metas if getattr(m, "tile_scheduler_metadata", None) is not None]
        _tile_stats["unplanned"] += len(metas) - len(planned)
        for m in planned[1:]:
            ref = planned[0]
            ok = torch.equal(m.tile_scheduler_metadata, ref.tile_scheduler_metadata) and torch.equal(m.num_splits, ref.num_splits)
            _tile_stats["same" if ok else "diff"] += 1
    if _tile_prev:
        _tile_stats["forwards"] += 1
        new_diff = _tile_stats["diff"] != _tile_stats.get("logged_diff", 0)
        if _tile_stats["forwards"] <= 3 or _tile_stats["forwards"] % 25 == 0 or (new_diff and _tile_stats["diff"] <= 200):
            _tile_stats["logged_diff"] = _tile_stats["diff"]
            logger.warning("ds41h attn TILE_SHARE check: %s rows %d (per type: plans of groups 2..n vs group 1 of the same forward)",
                           {k: v for k, v in _tile_stats.items() if k != "logged_diff"}, _tile_rows[0])
    _tile_prev = [(attr, [getattr(md, attr) for md in mds if getattr(md, attr) is not None]) for attr in _TILE_ATTRS]


def _share_tile_sched(swa_metadata: Any) -> None:
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return
    ctx = get_forward_context()
    if ctx.additional_kwargs.get("ds41h.tile_share") is not None:
        return
    ctx.additional_kwargs["ds41h.tile_share"] = True
    if int(getattr(swa_metadata, "num_decode_tokens", 0) or 0) <= flags.MAX_DECODE_TOKENS:
        return
    from ds41h.patch_core import _metadata_objects

    mds = [md for md in _metadata_objects(ctx.attn_metadata) if all(hasattr(md, a) for a in _TILE_ATTRS)]
    if TILE_SHARE == 2:
        _tile_check(mds)
        _tile_rows[0] = int(getattr(swa_metadata, "num_decode_tokens", 0) or 0)
        return
    for attr in _TILE_ATTRS:
        metas = [getattr(md, attr) for md in mds]
        live = [m for m in metas if m is not None]
        if len(live) < 2 or any(getattr(m, "have_initialized", True) for m in live):
            continue  # nothing to share, or a planner already ran for this type in this forward: leave it alone
        for md in mds:
            if getattr(md, attr) is not None:
                setattr(md, attr, live[0])


@functools.cache
def ext() -> Any:
    """The JIT-built extension (sm_90a)."""
    if VARIANT == "d":
        if TRACE:
            return cuext.load("ds41h_sparse_decode_d_prof", "sparse_decode_d.cu", extra_cuda_cflags=("-DDS41H_ATTN_PROF",))
        return cuext.load("ds41h_sparse_decode_d", "sparse_decode_d.cu")
    if TRACE:
        return cuext.load("ds41h_sparse_decode_prof", "sparse_decode.cu", extra_cuda_cflags=("-DDS41H_ATTN_PROF",))
    return cuext.load("ds41h_sparse_decode", "sparse_decode.cu")


def sparse_decode(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
    *,
    out: torch.Tensor | None = None,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
    token_to_req: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    is_valid: torch.Tensor | None = None,
    cluster: int | None = None,
    spread: bool | None = None,
    pdl: bool = True,
    fresh: torch.Tensor | None = None,
    aflag: torch.Tensor | None = None,
    atarget: int = 0,
    offtab: torch.Tensor | None = None,
    offmode: int = 0,
) -> torch.Tensor:
    """Sparse MLA decode for 16 heads over stock V4 fp8_ds_mla paged caches.

    q            [M, >=16, 512] bf16, stride(1) == 512 (stock's padded [M, 64, 512] q works as is).
    swa_cache    [num_blocks, bs, 584] uint8 (page stride = stride(0), any bs).
    swa_indices  [M, W] or [M, 1, W] int32 global slots (block * bs + offset), -1 = none.
    swa_lens     [M] int32; key j is attended iff j < swa_lens[m] and swa_indices[m, j] >= 0.
    sink         [>=16] f32 attention sink (natural-log units); heads 0..15 are used.
    scale        softmax scale (512 ** -0.5).
    extra_cache  optional second (compressed) cache, same record; then extra_indices [M, K] or [M, 1, K]
                 int32 and either
                   global mode: extra_lens [M] int32 (FlashMLA's extra_indices/extra_topk_length), or
                   local mode:  token_to_req [>=M] int32, block_table [R, B] int32, is_valid [>=M] bool;
                                slot = block_table[req, l // bs] * bs + l % bs, length = is_valid ?
                                #(l >= 0) : 0 (fused compute_global_topk_indices_and_lens), K <= 512.
    out          optional [M, >=16, 512] bf16 output (heads 0..15 written); allocated if None.
    cluster      0/None = auto, 8 or 16 CTAs per query row. spread: one CTA per SM. pdl: programmatic
                 dependent launch.
    fresh        early mode: int64 [<= 64] SWA slots written by this step (e.g. swa_metadata.slot_mapping);
                 everything else is read before griddepcontrol.wait (see EARLY).
    aflag        flag mode: the ds41h prologue's handoff counter (prologue.handoff_counter) and atarget its value once the
                 prologue (launched with aflag) stored q and the KV inserts: replaces griddepcontrol.wait (DS41H_ATTN_FLAG).
    offtab       key-offset table (int64 [>= rows * 640 * 2]) and offmode 1 (resolve + store) / 2 (load instead of resolving),
                 see OFFTAB; the caller guarantees that a reader's inputs resolve to the writer's offsets.
    Requires W + K <= 640. Returns out.
    """
    if out is None:
        out = torch.empty(q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    okw = dict(offtab=offtab, offmode=int(offmode)) if offmode else {}
    ext().sparse_decode(
        q,
        out,
        swa_cache,
        swa_indices,
        swa_lens,
        sink,
        float(scale),
        extra_cache,
        extra_indices,
        extra_lens,
        token_to_req,
        block_table,
        is_valid,
        CLUSTER if cluster is None else int(cluster),
        SPREAD if spread is None else bool(spread),
        bool(pdl),
        fresh,
        aflag,
        int(atarget),
        **okw,
    )
    return out


def _cache_layout_ok(t: torch.Tensor) -> bool:
    """The C++ checks of sparse_decode.cu cache_view(): V4 record, contiguous page, 16-B aligned pages."""
    return (
        t.dtype == torch.uint8
        and t.dim() == 3
        and t.shape[-1] == V4_RECORD_BYTES
        and t.stride(2) == 1
        and t.stride(1) == V4_RECORD_BYTES
        and t.stride(0) % 16 == 0
        and t.stride(0) >= t.shape[1] * V4_RECORD_BYTES
        and t.data_ptr() % 16 == 0
    )


@functools.cache
def _is_sm90(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (9, 0)  # the extension is built for sm_90a only


def decode_ineligible_reason(
    layer: Any,
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_metadata: Any,
    attn_metadata: Any,
    swa_only: bool,
) -> str | None:
    """Why ``layer``'s decode step must stay on stock FlashMLA (None if the ds41h kernel applies)."""
    if getattr(layer, "n_local_heads", None) != NUM_HEADS:
        return f"n_local_heads={getattr(layer, 'n_local_heads', None)} (kernel is built for {NUM_HEADS})"
    if q.dtype != torch.bfloat16 or q.dim() != 3 or q.shape[-1] != HEAD_DIM or q.shape[1] < NUM_HEADS:
        return f"q {tuple(q.shape)} {q.dtype}"
    if q.stride(-1) != 1 or q.stride(1) != HEAD_DIM or not q.is_cuda:
        return "q layout"
    if not _is_sm90(q.device.index if q.device.index is not None else torch.cuda.current_device()):
        return "device is not sm90"
    num_tokens = swa_metadata.num_decode_tokens
    if num_tokens > MAX_ROWS:
        return f"{num_tokens} decode rows > DS41H_ATTN_MAX_ROWS={MAX_ROWS}"
    if q.stride(0) % 8 != 0 or q.data_ptr() % 16 != 0:
        return "q alignment"
    swa = layer.swa_cache_layer.kv_cache
    if not _cache_layout_ok(swa):
        return f"SWA cache {tuple(swa.shape)} {swa.dtype} stride {swa.stride()} is not the V4 584-B paged record"
    idx, lens = swa_metadata.decode_swa_indices, swa_metadata.decode_swa_lens
    if idx is None or lens is None:
        return "no decode SWA indices"
    if idx.dtype != torch.int32 or idx.stride(-1) != 1 or lens.dtype != torch.int32 or not lens.is_contiguous():
        return "decode SWA index layout"
    width = idx.shape[-1]
    topk = 0
    if not swa_only:
        topk_buf = getattr(layer, "topk_indices_buffer", None)
        if attn_metadata is None or kv_cache is None or topk_buf is None:
            return "missing compressed metadata"
        if not _cache_layout_ok(kv_cache):
            return f"compressed cache {tuple(kv_cache.shape)} {kv_cache.dtype} stride {kv_cache.stride()} is not the V4 584-B paged record"
        bt = attn_metadata.block_table
        if bt.dtype != torch.int32 or bt.dim() != 2 or bt.stride(1) != 1:
            return "compressed block table layout"
        if kv_cache.shape[1] != attn_metadata.block_size // layer.compress_ratio:
            return "compressed cache page size != block_size // compress_ratio"
        valid, t2r = swa_metadata.is_valid_token, swa_metadata.token_to_req_indices
        if valid is None or t2r is None:
            return "missing token metadata"
        if valid.dtype != torch.bool or not valid.is_contiguous() or t2r.dtype != torch.int32 or not t2r.is_contiguous():
            return "token metadata layout"
        if topk_buf.dtype != torch.int32 or topk_buf.dim() != 2 or topk_buf.stride(-1) != 1:
            return "top-k buffer layout"
        topk = topk_buf.shape[-1]
        if topk > MAX_TOPK:
            return f"top-k {topk} > {MAX_TOPK}"
    if width + topk > MAX_KEYS:
        return f"{width} + {topk} keys > {MAX_KEYS}"
    return None


_logged: set[str] = set()
_active_logged = False


def _log_once(level: int, msg: str, *args: Any) -> None:
    key = msg % args if args else msg
    if key not in _logged:
        _logged.add(key)
        logger.log(level, msg, *args)


def forward_decode(
    layer: Any,
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_metadata: Any,
    attn_metadata: Any,
    swa_only: bool,
    output: torch.Tensor,
) -> bool:
    """Drop-in for the body of stock ``_forward_decode``. Returns False (nothing launched) if ineligible.

    Inputs are exactly the stock ones: q = q[:num_decode_tokens] ([T, padded_heads, 512] bf16), kv_cache =
    the kv-source layer's compressed cache [num_blocks, block_size // ratio, 584] (None for SWA-only),
    output = output[:num_decode_tokens].
    """
    handoff = None
    if "ds41h.prologue" in sys.modules and sys.modules["ds41h.prologue"]._handoff:
        # DS41H_ATTN_FLAG: the ds41h prologue of this layer signals q / KV readiness through a counter (ds41h/prologue.py)
        handoff = sys.modules["ds41h.prologue"].take_handoff(layer, q.device)
    reason = decode_ineligible_reason(layer, q, kv_cache, swa_metadata, attn_metadata, swa_only)
    writer = OFFTAB and not swa_only and getattr(layer, "indexer", None) is not None
    if reason is not None:
        if writer:
            _offtab_key.pop(q.device.index if q.device.index is not None else torch.cuda.current_device(), None)
        if handoff is not None:
            ext().aflag_reset(handoff.counter)  # the stock fallback consumes nothing: reset the prologue's signals
        _log_once(logging.INFO, "ds41h attn: layer %s falls back to stock FlashMLA decode: %s",
                  getattr(layer, "prefix", "?"), reason)
        return False
    num_tokens = swa_metadata.num_decode_tokens
    extra: dict[str, Any] = {}
    if not swa_only:
        # Same tensors stock hands to compute_global_topk_indices_and_lens (flashmla.py:186-194); the
        # block-table rows are indexed by token_to_req_indices, so the [:num_decodes] slice is implied.
        extra = dict(
            extra_cache=kv_cache,
            extra_indices=layer.topk_indices_buffer[:num_tokens],
            token_to_req=swa_metadata.token_to_req_indices,
            block_table=attn_metadata.block_table,
            is_valid=swa_metadata.is_valid_token,
        )
    fresh = None
    if EARLY:
        sm = getattr(swa_metadata, "slot_mapping", None)
        if isinstance(sm, torch.Tensor) and sm.dtype == torch.int64 and sm.is_contiguous() and 0 < sm.numel() <= MAX_FRESH:
            fresh = sm
    okw: dict[str, Any] = {}
    dev_idx = q.device.index if q.device.index is not None else torch.cuda.current_device()
    if OFFTAB and not swa_only:
        key = _offtab_resolve_key(num_tokens, layer.swa_cache_layer.kv_cache, swa_metadata, kv_cache, extra, fresh)
        tab = _offtab_tensor(q.device) if num_tokens <= MAX_ROWS else None
        if tab is None:
            _offtab_key.pop(dev_idx, None)
        elif writer:
            okw = dict(offtab=tab, offmode=1)
            _offtab_key[dev_idx] = key
        elif _offtab_key.get(dev_idx) == key:
            okw = dict(offtab=tab, offmode=2)
    tslot = None
    if TRACE and num_tokens <= 8:
        from ds41h import prologue as _pro

        tslot = _pro.trace_slot("att", q.device, int(getattr(layer, "layer_id", -1)))
        if tslot is not None:
            ext().set_prof(tslot)
    try:
        sparse_decode(
            q,
            layer.swa_cache_layer.kv_cache,
            swa_metadata.decode_swa_indices,
            swa_metadata.decode_swa_lens,
            layer.attn_sink,
            layer.scale,
            out=output,
            fresh=fresh,
            aflag=handoff.counter if handoff is not None else None,
            atarget=handoff.target if handoff is not None else 0,
            **okw,
            **extra,
        )
    except RuntimeError as e:
        if tslot is not None:
            ext().set_prof(None)
        if okw:
            _offtab_key.pop(dev_idx, None)
        if handoff is not None:
            ext().aflag_reset(handoff.counter)
        # The launcher re-validates every tensor (sparse_decode.cu qo_view/cache_view/index_view/...) before
        # touching the GPU and clears any CUDA "last error" it caused, so a check the Python side missed means
        # "ineligible": run stock instead of failing the step. Nothing was launched or written.
        _log_once(logging.WARNING, "ds41h attn: layer %s falls back to stock FlashMLA decode: %s",
                  getattr(layer, "prefix", "?"), str(e).splitlines()[0] if str(e) else type(e).__name__)
        return False
    if tslot is not None:
        ext().set_prof(None)
    layer.__dict__["_ds41h_attn_used"] = True  # read (and cleared) by the per-layer path log in patch_decoder
    global _active_logged
    if not _active_logged:  # hot path: no per-call string formatting
        _active_logged = True
        logger.info("ds41h attn: sm90 16-head sparse decode active (max rows %d, cluster %s)",
                    MAX_ROWS, CLUSTER or "auto")
    return True


def _tkey(t: torch.Tensor | None) -> tuple | None:
    """Identity of a metadata tensor for OFFTAB pairing: storage address, shape, strides, dtype (contents are per-step state
    that every layer of the step reads unchanged, except the top-k buffer, which only index-source layers rewrite)."""
    if t is None:
        return None
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype))


def _cgeo(t: torch.Tensor | None) -> tuple | None:
    """Page geometry of a V4 cache (the offsets are relative to the cache base): blocks, page size, page stride."""
    return None if t is None else (tuple(t.shape), tuple(t.stride()))


def _offtab_resolve_key(rows: int, swa_cache: torch.Tensor, swa_metadata: Any, kv_cache: torch.Tensor | None, extra: dict,
                        fresh: torch.Tensor | None) -> tuple:
    """Everything sparse_decode.cu resolve_offsets reads (plus the launch geometry that maps keys to CTAs)."""
    return (rows, CLUSTER, SPREAD, _cgeo(swa_cache), _tkey(swa_metadata.decode_swa_indices), _tkey(swa_metadata.decode_swa_lens),
            _cgeo(kv_cache), _tkey(extra.get("extra_indices")), _tkey(extra.get("token_to_req")), _tkey(extra.get("block_table")),
            _tkey(extra.get("is_valid")), _tkey(fresh))


def _offtab_tensor(device: torch.device) -> torch.Tensor | None:
    """The device's offset table (allocated on first use outside CUDA-graph capture; None if that would be inside one)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    t = _offtab.get(idx)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            _log_once(logging.INFO, "ds41h attn: OFFTAB table not allocated before graph capture; offsets resolved per layer")
            return None
        t = torch.zeros(MAX_ROWS * MAX_KEYS * 2, dtype=torch.int64, device=device)
        _offtab[idx] = t
    return t


def _debug_compare(stock: Any, layer: Any, args: tuple, output: torch.Tensor) -> None:
    """DS41H_DEBUG_COMPARE=1: rerun stock FlashMLA into a scratch buffer and log the difference (eager only)."""
    if torch.cuda.is_current_stream_capturing():
        return
    ref = torch.empty_like(output)
    stock(layer, *args, ref)
    a, b = output[:, :NUM_HEADS].float(), ref[:, :NUM_HEADS].float()
    diff = (a - b).abs().max().item()
    mag = b.abs().max().item()
    level = logging.WARNING if diff > 2e-2 * max(mag, 1.0) else logging.DEBUG
    logger.log(level, "ds41h attn debug %s: rows=%d max|ds41h-stock|=%.3e max|stock|=%.3e",
               getattr(layer, "prefix", "?"), output.shape[0], diff, mag)


# DS41H_TOPK_OVERLAP=1 (debug, eager forwards only): statistics of how much the indexer top-k sets of
# consecutive decode rows of one request overlap (input to a grouped multi-row decode attention: one KV pass for 4 rows needs
# the union, a shared-key split needs the intersection). Logged every 200 measured layer calls.
TOPK_OVERLAP = os.environ.get("DS41H_TOPK_OVERLAP", "0") not in ("0", "")
_ov = {"calls": 0, "groups": 0, "set": 0.0, "union4": 0.0, "inter4": 0.0, "union8": 0.0, "inter8": 0.0}


def _topk_overlap(layer: Any, swa_metadata: Any) -> None:
    if torch.cuda.is_current_stream_capturing():
        return
    T = int(getattr(swa_metadata, "num_decode_tokens", 0) or 0)
    buf = getattr(layer, "topk_indices_buffer", None)
    t2r = getattr(swa_metadata, "token_to_req_indices", None)
    if T < 16 or T % 8 or buf is None or t2r is None:
        return
    idx = buf[:T].long()
    req = t2r[:T].long()
    k = idx.shape[1]
    for g in (4, 8):
        grp = idx.view(T // g, g * k)
        same = (req.view(T // g, g) == req.view(T // g, g)[:, :1]).all(dim=1)  # groups inside one request only
        v = torch.where(grp >= 0, grp, torch.full_like(grp, -1))
        srt = torch.sort(v, dim=1).values
        valid = srt >= 0
        newv = torch.ones_like(valid)
        newv[:, 1:] = srt[:, 1:] != srt[:, :-1]
        union = (valid & newv).sum(dim=1).float()
        inter = (srt[:, g - 1:] == srt[:, : srt.shape[1] - (g - 1)]) & valid[:, g - 1:]
        inter = inter.sum(dim=1).float()
        n = int(same.sum())
        if n == 0:
            return
        _ov[f"union{g}"] += float(union[same].sum())
        _ov[f"inter{g}"] += float(inter[same].sum())
        if g == 4:
            _ov["groups"] += n
            _ov["set"] += float((idx >= 0).sum(dim=1).float().view(T // 4, 4)[same].mean(dim=1).sum())
    _ov["calls"] += 1
    if _ov["calls"] % 200 == 0:
        n4, n8 = max(1, _ov["groups"]), max(1, _ov["groups"] // 2)
        logger.warning("ds41h attn TOPK_OVERLAP: %d layer calls, %d groups of 4: mean |set| %.1f, union4 %.1f, inter4 %.1f, "
                       "union8 %.1f, inter8 %.1f (rows %d)", _ov["calls"], _ov["groups"], _ov["set"] / n4, _ov["union4"] / n4,
                       _ov["inter4"] / n4, _ov["union8"] / n8, _ov["inter8"] / n8, T)


# DS41H_ATTN_GROUPED=1 (opt-in): > 8 decode rows run ds41h/grouped_decode.py (FlashMLA's SM90
# sparse fp8 decode kernel with 4 rows x 16 heads per 64-row tile over the union of their key sets + per-row membership masks;
# lossless class, see that module). The <= 8-row path never reaches this branch.
GROUPED = os.environ.get("DS41H_ATTN_GROUPED", "0") not in ("0", "")
_grouped_failed = [False]


def _grouped_forward(layer: Any, q, kv_cache, swa_metadata: Any, attn_metadata: Any, swa_only: bool, output) -> bool:
    if _grouped_failed[0] or int(getattr(swa_metadata, "num_decode_tokens", 0) or 0) <= flags.MAX_DECODE_TOKENS:
        return False
    try:
        from ds41h import grouped_decode

        done = grouped_decode.forward(layer, q, kv_cache, swa_metadata, attn_metadata, swa_only, output)
    except Exception:  # noqa: BLE001 - keep serving on the stock kernel
        _grouped_failed[0] = True
        logger.exception("ds41h attn: grouped decode failed; stock FlashMLA decode from now on")
        return False
    if done:
        _log_once(logging.INFO, "ds41h attn: grouped multi-row sparse decode active (4 rows x 16 heads per tile)")
    return done


def install_flashmla_patch(module: Any) -> None:
    """Patch ``module.DeepseekV4FlashMLAAttention._forward_decode`` (module =
    vllm.models.deepseek_v41.nvidia.flashmla). Idempotent; keeps the stock method as
    ``_ds41h_stock_forward_decode``."""
    cls = module.DeepseekV4FlashMLAAttention
    if "_ds41h_stock_forward_decode" in cls.__dict__:
        return
    try:
        ext()  # build/load now (once per machine, file-locked) so the first decode step never JITs
    except Exception:
        logger.exception("ds41h attn: building the sparse decode extension failed; keeping stock FlashMLA decode")
        return
    stock = cls._forward_decode

    def _forward_decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output):
        if forward_decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output):
            if flags.DEBUG_COMPARE:
                _debug_compare(stock, self, (q, kv_cache, swa_metadata, attn_metadata, swa_only), output)
            return None
        if TOPK_OVERLAP and not swa_only:
            _topk_overlap(self, swa_metadata)
        if GROUPED and _grouped_forward(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output):
            return None
        if TILE_SHARE:
            _share_tile_sched(swa_metadata)
        padded = getattr(self, "padded_heads", q.shape[1])
        if q.dim() == 3 and q.shape[1] < padded:
            # Unpadded q / output from the ds41h prologue path (DS41H_DECODE_NOPAD): FlashMLA's FP8 decode
            # kernel needs padded_heads, so pad for it and copy the real heads back.
            import torch.nn.functional as F

            out_p = output.new_empty((output.shape[0], padded, output.shape[2]))
            stock(self, F.pad(q, (0, 0, 0, padded - q.shape[1])), kv_cache, swa_metadata, attn_metadata, swa_only, out_p)
            output.copy_(out_p[:, : output.shape[1]])
            return None
        return stock(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output)

    _forward_decode.__wrapped__ = stock  # type: ignore[attr-defined]
    _forward_decode.__doc__ = stock.__doc__
    cls._ds41h_stock_forward_decode = stock
    cls._forward_decode = _forward_decode
    logger.info("ds41h attn: patched %s.DeepseekV4FlashMLAAttention._forward_decode", module.__name__)
    if GROUPED:
        # 16-head q / attn_out in the decode-only steps grouped decode takes
        from ds41h import grouped_decode

        grouped_decode.install_nopad(cls)
