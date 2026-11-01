"""ds41h prefill sequence parallelism (SP) for the full-row layers of an eager prefill step.

Stock TP4 (vLLM 0.32.0, no EP) runs every per-token part of a DeepSeek-V4.1 layer REPLICATED on all four ranks: both
mHC sublayer boundaries (post-map of the 4 x 5120 residual stream, 24-mix GEMV, Sinkhorn, collapse, RMSNorm; ~2 x 1.25
ms per layer at T=16K / 350 W), the replicated input projections (fused_wqa_wkv 1792 cols, the compressor's wkv|wgate
1024 fp32 cols, the indexer weights 32 cols) and the Engram injection, between two all-reduces of the [T, 5120]
sublayer outputs. With SP each rank keeps the residual stream of only T/4 rows:

    partial o / MoE out [T, H] --reduce-scatter--> rows of this rank [T/4, H] --mHC boundary (T/4 rows)--> layer input
    attention: wqa|wkv, compressor kv|score, indexer weights on the T/4 rows -> all-gather the projections [T, ...]
               -> stock q-norm / wq_b / RoPE / KV insert / indexer / sparse attention / wo_a / wo_b on all T rows
               (TP over heads as before) -> partial [T, H]
    FFN:       layer input [T/4, H] --all-gather--> [T, H] -> stock gate / shared expert / routed MoE (TP) -> partial

A ring all-reduce IS a reduce-scatter + an all-gather, so the communication volume does not grow (it shrinks: the
attention input is gathered as 1792 bf16 + 1024 fp32 + 32 columns instead of 5120 bf16), while the boundaries, the
replicated projections and the Engram post kernel do 1/4 of the work per rank.

Chunks (DS41H_PREFILL_SP_CHUNKS=K, ctl "sp_chunks"; default 1): the rows are split into K consecutive chunks and every
rank owns a quarter of EACH chunk; all SP state is kept per chunk, every collective runs per chunk. With
DS41H_PREFILL_SP_OVERLAP=1 (ctl "sp_overlap") the collectives run on a side stream so chunk k's communication overlaps
chunk k+1's compute: reduce-scatter of chunk k+1 / all-gather of chunk k under the boundary + projections of the other
chunk, wo_b split into K row chunks with the reduce-scatter of chunk k under the wo_b GEMM of chunk k+1.

LOSSLESS: every per-row computation is the stock (or ds41h fused, already gated) kernel on a subset of the rows; only
the GEMM tile shapes (row count) and the NCCL reduction order differ = summation order.

Scope: DeepseekV4Model._run_layers for the full-row layers [0, decoder_replay_start) of an eager step with
>= DS41H_PREFILL_SP_MIN_T rows (not capturing, no stock sequence parallel / mHC overlap / fused mHC all-reduce, no aux
hidden-state layer in the range). Rows of a chunk: L_k = 4 c_k + rem_k; rank r owns [g_k + r c_k, g_k + (r + 1) c_k)
plus the rem_k (<= 3, only the last chunk) tail rows, which every rank holds and computes redundantly (reduced by a small
all-reduce), so no padding copies are needed.

Cut (DS41H_PREFILL_SP_CUT=1, default 1 with SP): instead of all-gathering the residual stream for the stock
replay (layer `cut` writes its KV for every row, then layers >= cut run on each request's last `window` rows), the
replay call (DecoderReplayLayers.__call__) gets the SP state: layer cut's attention boundary + KV projections run on
the local rows (projections all-gathered, KV / compressor / indexer-K writes on all rows as stock write_kv), and the
replay rows' inputs are gathered from their owning ranks (masked rows + one small all-reduce per dtype, exact).

Switch: DS41H_PREFILL_SP (default 1 since the pf350_spv3 gate, 350 W), DS41H_PREFILL_SP_MIN_T (default 2048).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

from ds41h import detprobe as _detprobe, flags, prefill_ctl, prefill_ctx

logger = logging.getLogger("vllm.ds41h.prefill")

_SP_ENV = flags.ENABLED and flags.PREFILL and flags._flag("DS41H_PREFILL_SP", "1")  # default ON since the pf350_spv3 gate
# installed when switched on, or when a runtime control file may switch it on (prefill_ctl key "sp")
SP = _SP_ENV or (flags.ENABLED and flags.PREFILL and bool(prefill_ctl.PATH))
SP_MIN_T = max(64, int(os.environ.get("DS41H_PREFILL_SP_MIN_T", "2048")))
_SP_CUT_ENV = flags._flag("DS41H_PREFILL_SP_CUT", "1")
_CHUNKS_ENV = max(1, int(os.environ.get("DS41H_PREFILL_SP_CHUNKS", "1")))
_OVERLAP_ENV = flags._flag("DS41H_PREFILL_SP_OVERLAP", "0")
_FFN_ENV = flags._flag("DS41H_PREFILL_SP_FFN", "1")
_NORM_LOCAL = flags._flag("DS41H_PREFILL_SP_NORM_LOCAL", "1")  # q / kv RMSNorm on the local rows before the gather  # 1: runner bypass, shared-expert add folded into moe_combine
SP_CUT = SP
_CUT_STATE: list = [None]  # handed from run_layers_sp to the replay call of the same forward
_MEMLOG = os.environ.get("DS41H_PREFILL_SP_MEMLOG", "0") not in ("0", "")
_LOGGED: set[str] = set()
_COMM_STREAM: list = [None]


def _mem(tag: str) -> None:
    """DS41H_PREFILL_SP_MEMLOG=1: allocator state per SP stage (debugging)."""
    if not _MEMLOG:
        return
    import torch

    free, _ = torch.cuda.mem_get_info()
    logger.info("ds41h prefill SP mem %s: alloc %.2f GiB reserved %.2f GiB max_alloc %.2f GiB device free %.2f GiB", tag,
                torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30,
                torch.cuda.max_memory_allocated() / 2**30, free / 2**30)


def _log_once(key: str, msg: str, *args) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(msg, *args)


@contextlib.contextmanager
def _set_attr(obj: Any, name: str, value: Any):
    had = name in obj.__dict__
    old = obj.__dict__.get(name)
    obj.__dict__[name] = value
    try:
        yield
    finally:
        if had:
            obj.__dict__[name] = old
        else:
            obj.__dict__.pop(name, None)


class _Chunk:
    __slots__ = ("g0", "c", "rem", "l0", "n", "lo", "body", "g1")

    def __init__(self, g0, c, rem, l0, rank, world):
        self.g0, self.c, self.rem, self.l0 = g0, c, rem, l0
        self.n = c + rem
        self.lo = g0 + rank * c
        self.body = g0 + world * c
        self.g1 = self.body + rem


class _Rows:
    """Row split of one step into K chunks; in chunk k rank r owns [g_k + r c_k, g_k + (r + 1) c_k) + the tail rows.
    Local SP state is a list of per-chunk tensors [n_k, ...]. Collectives optionally run on a side stream (overlap):
    they wait for the producing work of the current stream and return an event the consumer waits on."""

    def __init__(self, T: int, K: int = 1, overlap: bool = False):
        from vllm.distributed import get_tp_group

        g = get_tp_group()
        self.world = W = g.world_size
        self.rank = g.rank_in_group
        self.comm = g.device_communicator.pynccl_comm
        self.T = T
        q = T // (W * K)
        if q == 0:
            K = 1
            q = T // W
        self.K = K
        self.chunks: list[_Chunk] = []
        g0 = l0 = 0
        for k in range(K):
            L = W * q if k < K - 1 else T - g0
            c = L // W
            ch = _Chunk(g0, c, L - W * c, l0, self.rank, W)
            self.chunks.append(ch)
            g0 += L
            l0 += ch.n
        self.n = l0
        self.overlap = overlap
        self.cs = None
        self._keep: list = []
        if overlap:
            import torch

            if _COMM_STREAM[0] is None:
                _COMM_STREAM[0] = torch.cuda.Stream()
            self.cs = _COMM_STREAM[0]

    # ------------------------------------------------------------------ stream helpers
    @contextlib.contextmanager
    def _on_comm(self, *tensors):
        """Run the enclosed collective on the comm stream after the current stream's pending work; yields the event
        the consumer waits on (None without overlap). Tensors read / written there are recorded on the comm stream."""
        if self.cs is None:
            yield None
            return
        import torch

        main = torch.cuda.current_stream()
        self.cs.wait_stream(main)
        ev = torch.cuda.Event()
        with torch.cuda.stream(self.cs):
            yield ev
            ev.record(self.cs)
        # Keep what the comm stream reads / writes alive until the main stream has waited for `ev` (wait()). NOT
        # record_stream: blocks recorded on a second stream make the caching allocator poll / synchronize events
        # whenever it recycles them (pfsp5 sp2o: 238 cudaEventSynchronize, ~18 ms CPU stalls per layer, 2x slower).
        self._keep.append((ev, tensors))

    def wait(self, ev) -> None:
        if ev is not None:
            import torch

            torch.cuda.current_stream().wait_event(ev)
            # the main stream is now ordered after the collective: its inputs / outputs may be recycled on main
            self._keep = [(e, t) for e, t in self._keep if e is not ev]

    def drain(self) -> None:
        """Main stream after every collective issued so far (end of the SP region)."""
        if self.cs is not None:
            import torch

            torch.cuda.current_stream().wait_stream(self.cs)
            self._keep.clear()

    # ------------------------------------------------------------------ collectives (per chunk)
    def shard(self, x) -> list:
        """Full rows (identical on every rank) -> per-chunk local rows (views when there is no tail)."""
        import torch

        out = []
        for ch in self.chunks:
            a = x[ch.lo : ch.lo + ch.c]
            out.append(a if not ch.rem else torch.cat([a, x[ch.body : ch.g1]]))
        return out

    def reduce_scatter(self, xp, k: int):
        """Partial sums [T, ...] -> (reduced local rows of chunk k [n_k, ...], event)."""
        from vllm.distributed import tensor_model_parallel_all_reduce

        ch = self.chunks[k]
        out = xp.new_empty((ch.n, *xp.shape[1:]))
        with self._on_comm(xp, out) as ev:
            self.comm.reduce_scatter(out[: ch.c], xp[ch.g0 : ch.body])
            if ch.rem:
                out[ch.c :] = tensor_model_parallel_all_reduce(xp[ch.body : ch.g1].contiguous())
        return out, ev

    def all_gather_into(self, out, xs, k: int):
        """Local rows of chunk k [n_k, ...] -> rows [g_k, g_k+1) of the full tensor `out`; returns the event."""
        ch = self.chunks[k]
        xs = xs if xs.is_contiguous() else xs.contiguous()
        with self._on_comm(xs, out) as ev:
            self.comm.all_gather(out[ch.g0 : ch.body], xs[: ch.c])
            if ch.rem:
                out[ch.body : ch.g1] = xs[ch.c :]
        return ev

    def all_gather(self, xs_list) -> Any:
        """Per-chunk local rows -> full rows [T, ...] (synchronous)."""
        x0 = xs_list[0]
        out = x0.new_empty((self.T, *x0.shape[1:]))
        for k, xs in enumerate(xs_list):
            self.wait(self.all_gather_into(out, xs, k))
        return out

    def all_gather_tail(self, t):
        """Every rank's [rem, ...] -> [world, rem, ...] (rank order), synchronous."""
        t = t.contiguous()
        out = t.new_empty((self.world, *t.shape))
        self.comm.all_gather(out, t)
        return out


# =========================================================================================== eligibility
def _eligible(model, layer_ids, hidden_states, positions, residual) -> bool:
    import torch

    if not SP or residual is not None or not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 2:
        return False
    if not int(prefill_ctl.get("sp", int(_SP_ENV))):
        return False
    T = positions.shape[0]
    if T < SP_MIN_T or hidden_states.shape[0] != T or not hidden_states.is_cuda:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    try:
        from vllm.forward_context import get_forward_context, in_piecewise_cudagraph

        if in_piecewise_cudagraph():
            return False
        # Dummy / memory-profile runs (no attention metadata) keep the stock path: SP's peak activation memory is lower
        # than stock's (3.1 vs 4.8 GiB at 16896 rows), but its mix of T/4- and T-row allocations fragments the
        # allocator under the profile's 20 MB split limit (reserved up to +11 GiB), which left too little device memory
        # for FlashInfer's AR workspace right after the profile run (rank 0 OOM -> collective mismatch -> hang).
        if not isinstance(get_forward_context().attn_metadata, dict):
            return False
    except Exception:  # noqa: BLE001
        return False
    if getattr(model, "use_sequence_parallel", False) or getattr(model, "fuse_mhc_all_reduce", False):
        return False
    if getattr(model, "use_native_mega_moe", False):
        return False
    ids = list(layer_ids)
    if not ids or ids[0] != 0 or ids != list(range(ids[0], ids[-1] + 1)):
        return False
    aux = set(getattr(model, "aux_hidden_state_layers", ()) or ())
    if any(i in aux for i in ids):
        return False
    from vllm.distributed import get_tp_group

    g = get_tp_group()
    if g.world_size < 2 or g.device_communicator is None or g.device_communicator.pynccl_comm is None:
        return False
    if g.device_communicator.pynccl_comm.disabled:
        return False
    return True


# =========================================================================================== layer pieces
def _proj_local(attn, x_s, indexer_weights: bool = True):
    """attention.py _run_parallel_input_projections (fused_wqa_wkv, compressor wkv|wgate fp32, indexer weights) on
    local rows, plus _split_qkv_and_norm (per-row q / kv RMSNorm) when it returns plain tensors: then the normalized
    q (1280) / kv (512) are gathered instead of the raw projection (same bytes, the norms run on T/4 rows)."""
    import torch

    wqa = attn.fused_wqa_wkv
    with _set_attr(wqa, "_ds41h_wqa_shard", False):  # rows are already sharded (ds41h/linear.py WQA_SHARD)
        qr_kv = attn._fused_wqa_wkv_gemm(x_s)
    parts = [qr_kv]
    if _NORM_LOCAL:
        qr, qr_scale, kv = attn._split_qkv_and_norm(qr_kv)
        if qr_scale is None and isinstance(qr, torch.Tensor) and isinstance(kv, torch.Tensor):
            parts = [qr, kv]
    kv_score = None
    if attn.compressor is not None:
        kv_score = torch.mm(x_s, attn.compressor.fused_wkv_wgate.weight.T, out_dtype=torch.float32)
    idx_w = None
    if indexer_weights and attn.indexer is not None:
        idx_w, _ = attn.indexer.weights_proj(x_s)
    return parts, kv_score, idx_w


def _projections(attn, x_list, rows: _Rows, indexer_weights: bool = True):
    """Projections of every chunk's local rows, all-gathered into full [T, ...] tensors (chunk k's gather overlaps
    chunk k+1's projections with overlap on). Returns (qr, qr_scale, kv, kv_score, idx_w) as _split_qkv_and_norm +
    _run_parallel_input_projections would on all rows."""
    full = None
    evs = []
    for k, x_s in enumerate(x_list):
        qparts, kv_score, idx_w = _proj_local(attn, x_s, indexer_weights)
        parts = [*qparts, kv_score, idx_w]
        if full is None:
            full = [None if p is None else p.new_empty((rows.T, *p.shape[1:])) for p in parts]
        assert len(parts) == len(full)
        for f, p in zip(full, parts):
            if p is not None:
                evs.append(rows.all_gather_into(f, p, k))
    for ev in evs:
        rows.wait(ev)
    if len(full) == 3:  # raw projection gathered: [qr_kv, kv_score, idx_w] -> split + norm on all rows (stock)
        qr, qr_scale, kv = attn._split_qkv_and_norm(full[0])
        return qr, qr_scale, kv, full[1], full[2]
    assert len(full) == 4  # [qr, kv (normalized on the local rows), kv_score, idx_w]
    return full[0], None, full[1], full[2], full[3]


def _attention(attn, x_list, positions, rows: _Rows):
    """Stock DeepseekV4Attention.forward (attention.py:659-687) with the replicated input projections computed on the
    local rows and all-gathered; wo_b runs per row chunk without its TP all-reduce, each chunk's partial is
    reduce-scattered right away. Returns the per-chunk (reduced local rows, event) list."""
    qr, qr_scale, kv, kv_score, idx_w = _projections(attn, x_list, rows)
    T = rows.T
    H = x_list[0].shape[1]
    # Only .shape[0] / .dtype / .device of hidden_states are read past the projections (attn out buffer, indexer op).
    hs = x_list[0].new_empty((1, H)).expand(T, H)
    attn_out = attn._alloc_attn_out(T, hs)
    attn._prepare_and_attn_fn(hs, qr, kv, qr_scale, kv_score, idx_w, positions, attn_out)
    del qr, kv, kv_score, idx_w
    wo_b = attn.wo_b
    got: list = []

    def wo_b_chunked(z):
        # dsv41_o_proj / ds41h WOA16 end with layer._wo_b_proj(z): row chunks of z through wo_b (TP partial), each
        # chunk reduce-scattered as soon as it exists
        y = None
        for k, ch in enumerate(rows.chunks):
            y = wo_b(z[ch.g0 : ch.g1])
            got.append(_rs_rows(rows, y, k))
        return y

    prev = wo_b.reduce_results
    wo_b.reduce_results = False
    try:
        with _set_attr(attn, "_wo_b_proj", wo_b_chunked):
            attn._o_proj(attn_out, positions)
    finally:
        wo_b.reduce_results = prev
    assert len(got) == rows.K, "wo_b was not reached through layer._wo_b_proj"
    return got


def _rs_rows(rows: _Rows, y, k: int):
    """Reduce-scatter of a chunk-k partial whose rows are [g_k, g_k+1) (y is [L_k, ...], not [T, ...])."""
    from vllm.distributed import tensor_model_parallel_all_reduce

    ch = rows.chunks[k]
    out = y.new_empty((ch.n, *y.shape[1:]))
    with rows._on_comm(y, out) as ev:
        rows.comm.reduce_scatter(out[: ch.c], y[: ch.body - ch.g0])
        if ch.rem:
            out[ch.c :] = tensor_model_parallel_all_reduce(y[ch.body - ch.g0 :].contiguous())
    return out, ev


def _ffn_partial(ffn, x, input_ids):
    """Stock DeepseekV4MoE forward without the runner's final TP all-reduce (moe_runner.py:483-517)."""
    cfg = ffn.experts.moe_config
    prev = cfg.skip_final_all_reduce
    cfg.skip_final_all_reduce = True
    try:
        return ffn(x, input_ids, None)
    finally:
        cfg.skip_final_all_reduce = prev


def _ffn_fused(ffn, x, input_ids):
    """The runner's MoE (moe_runner.py forward / _forward_impl / _apply_quant_method, NO_OVERLAP order, no transforms,
    no routed scale on the output) step by step: stock gate + stock router + stock shared expert (TP partial), then the
    ds41h W4A16 routed experts with the shared output folded into moe_combine (fp32 sum, one rounding instead of the
    runner's bf16 add) and no final all-reduce. None if the layer does not qualify (caller: _ffn_partial)."""
    from ds41h import moe

    r = getattr(ffn, "experts", None)
    if r is None or getattr(r, "gate", None) is None or getattr(r, "_fse_fuse_gate", False):
        return None
    if getattr(r, "routed_input_transform", None) is not None or getattr(r, "routed_output_transform", None) is not None:
        return None
    if float(getattr(r, "routed_scaling_factor", 1.0)) != 1.0 or getattr(r, "_shared_experts", None) is None:
        return None
    re_ = r.routed_experts
    if not getattr(re_, "ds41h_raw", False) or re_.quant_method.is_monolithic:
        return None
    router_logits, _ = r.gate(x)
    tw, ids = r.router.select_experts(hidden_states=x, router_logits=router_logits,
                                      topk_indices_dtype=r._quant_method.topk_indices_dtype, input_ids=input_ids)
    shared = r._shared_experts._layer(x)
    y = moe.forward_layer(re_, x, tw, ids, addend=shared)
    if y is None:
        return None
    return y


def _cols_to_rows(kv_col, rows: _Rows) -> list:
    """Column shard [T, N] of every rank (rank r = global columns [r N, (r + 1) N)) -> per-chunk full rows
    [n_k, world N] of this rank's local rows (body rows by one all-to-all per chunk, the replicated tail rows by a
    small all-gather). Synchronous."""
    import torch.distributed as dist

    from vllm.distributed import get_tp_group

    g = get_tp_group()
    W, N = rows.world, kv_col.shape[1]
    kv_col = kv_col if kv_col.is_contiguous() else kv_col.contiguous()
    outs = []
    for ch in rows.chunks:
        recv = kv_col.new_empty((W, ch.c, N))
        dist.all_to_all_single(recv, kv_col[ch.g0 : ch.body], group=g.device_group)  # rows [g + d c, +c) -> rank d
        out = kv_col.new_empty((ch.n, W * N))
        out[: ch.c].view(ch.c, W, N).copy_(recv.transpose(0, 1))
        if ch.rem:
            tail = rows.all_gather_tail(kv_col[ch.body : ch.g1])  # [W, rem, N]
            out[ch.c :].view(ch.rem, W, N).copy_(tail.transpose(0, 1))
        outs.append(out)
    return outs


def _engram(eng, hidden_list, hash_ids, token_mask, rows: _Rows) -> list:
    """Stock Engram.forward (engram.py:1676-1740, use_sequence_parallel False) for the local rows of every chunk."""
    import torch

    from vllm.models.deepseek_v41.common import engram as em

    T = hash_ids.shape[0]
    wkv = eng.wkv
    if eng._batch is None and getattr(wkv, "gather_output", False) and getattr(wkv, "tp_size", 1) == rows.world:
        # Column-parallel wkv WITHOUT its output all-gather ([T, 6400] per rank), then an all-to-all that hands every
        # rank the full kv of its own rows: 3/4 x [T, 6400] moved per rank instead of all-gathering [T, 25600] (and
        # no transposing copy of the gathered columns). Same per-element GEMM results.
        emb = eng.embed(hash_ids).flatten(-2)
        with _set_attr(wkv, "gather_output", False):
            kv_col = wkv(emb)
        kv_loc = _cols_to_rows(kv_col, rows)
        local = True
    else:
        kv_full = eng.wkv(eng._batch.wkv_input(eng, T)) if eng._batch is not None else eng.wkv(eng.embed(hash_ids).flatten(-2))
        local = False
    assert token_mask is None or token_mask.shape == (T,)
    block_size = None
    outs = []
    for k, (ch, hidden_s) in enumerate(zip(rows.chunks, hidden_list)):
        n, hc_mult, dim = hidden_s.shape
        kv_k = kv_loc[k] if local else None
        if local:
            assert kv_k.shape[1] == (hc_mult + 1) * dim
        assert hc_mult == eng.hc_mult and dim == eng.dim
        output = torch.empty_like(hidden_s)
        block_size = em.triton.next_power_of_2(dim)
        num_warps = 8 if block_size >= 2048 else 4
        parts = [(0, ch.c, 0 if local else ch.lo, ch.lo)]
        if ch.rem:
            parts.append((ch.c, ch.rem, ch.c if local else ch.body, ch.body))
        for start, cnt, src, row0 in parts:
            if cnt <= 0:
                continue
            hs = hidden_s[start : start + cnt]
            out = output[start : start + cnt]
            kv_p = kv_k[src : src + cnt] if local else kv_full[src : src + cnt]
            mask = token_mask[row0 : row0 + cnt] if token_mask is not None else None
            em._fused_engram_post_wkv_kernel[(cnt * hc_mult,)](
                hs, kv_p, eng.q_weight, eng.k_weight, mask if mask is not None else hs, out, cnt,
                hs.stride(0), hs.stride(1), hs.stride(2), kv_p.stride(0), kv_p.stride(1),
                eng.q_weight.stride(0), eng.q_weight.stride(1), eng.k_weight.stride(0), eng.k_weight.stride(1),
                mask.stride(0) if mask is not None else 0, out.stride(0), out.stride(1), out.stride(2),
                eng.eps, eng.clamp_value, DIM=dim, HC_MULT=hc_mult, BLOCK_SIZE=block_size, HAS_MASK=mask is not None,
                num_warps=num_warps,
            )  # fmt: skip
        outs.append(output)
    return outs


def _layer(module, L, rows: _Rows, st: dict, positions, input_ids) -> dict:
    """Stock DeepseekV4DecoderLayer.forward (model.py:455-625; mhc_stream None, no stock SP, no fused mHC all-reduce)
    on the local rows. st: xp (partial [T, H] of the previous FFN, or the full embedding for the first layer) and the
    per-chunk local residual / post / res / pre lists."""
    hc = (L.hc_attn_scale, L.hc_attn_base, L.rms_norm_eps, L.hc_eps, L.hc_eps, L.hc_post_alpha, L.hc_sinkhorn_iters)
    norm = dict(norm_weight=L.attn_norm.weight, norm_eps=L.attn_norm.variance_epsilon)
    K = rows.K
    residual, post, res, x_attn, attn_pre = [None] * K, [None] * K, [None] * K, [None] * K, [None] * K
    if st["residual"] is None:
        # First layer: the embedding broadcast to hc copies, identity pre-mix (model.py:493-514).
        assert L.hc_attn_fn_broadcast is not None
        for k, x_s in enumerate(rows.shard(st["xp"])):
            residual[k] = x_s.unsqueeze(1).expand(-1, L.hc_mult, -1).contiguous()
            post[k], res[k], x_attn[k], attn_pre[k] = module.mhc_pre_delayed_tilelang(
                residual[k], L.hc_attn_fn_broadcast, *hc, x=x_s, **norm)
    elif L.engram is not None and st["engram_hashes"] is not None:
        # Engram layers (1, 14): previous post, Engram injection, pre (model.py:531-561).
        prev_post = []
        for k in range(K):
            x_s, ev = rows.reduce_scatter(st["xp"], k)
            rows.wait(ev)
            prev_post.append(module.mhc_post_tilelang(x_s, st["residual"][k], st["post"][k], st["res"][k]))
        residual = _engram(L.engram, prev_post, st["engram_hashes"][:, L.engram.layer_hash_index], st["engram_mask"], rows)
        del prev_post
        for k in range(K):
            post[k], res[k], x_attn[k], attn_pre[k] = module.mhc_pre_delayed_tilelang(
                residual[k], L.hc_attn_fn, *hc, pre_mix=st["pre"][k], **norm)
    else:
        rs = [rows.reduce_scatter(st["xp"], k) for k in range(K)]  # all issued up front (overlap: comm stream)
        for k, (x_s, ev) in enumerate(rs):
            rows.wait(ev)
            residual[k], post[k], res[k], x_attn[k], attn_pre[k], _ = module.mhc_shifted_post_pre(
                x_s, st["residual"][k], st["post"][k], st["res"][k], L.hc_attn_fn, *hc, pre_mix=st["pre"][k], **norm,
                capture_aux=False, stream=None, reduce_results=False,
            )  # fmt: skip
        del rs
    st["xp"] = None
    o_rs = _attention(L.attn, x_attn, positions, rows)
    del x_attn
    hcf = (L.hc_ffn_scale, L.hc_ffn_base, L.rms_norm_eps, L.hc_eps, L.hc_eps, L.hc_post_alpha, L.hc_sinkhorn_iters)
    ffn_pre = [None] * K
    x_full = None
    evs = []
    for k, (x_s, ev) in enumerate(o_rs):
        rows.wait(ev)
        residual[k], post[k], res[k], x_ffn, ffn_pre[k], _ = module.mhc_shifted_post_pre(
            x_s, residual[k], post[k], res[k], L.hc_ffn_fn, *hcf, pre_mix=attn_pre[k], norm_weight=L.ffn_norm.weight,
            norm_eps=L.ffn_norm.variance_epsilon, stream=None, reduce_results=False,
        )  # fmt: skip
        if x_full is None:
            x_full = x_ffn.new_empty((rows.T, *x_ffn.shape[1:]))
        evs.append(rows.all_gather_into(x_full, x_ffn, k))
    del o_rs
    for ev in evs:
        rows.wait(ev)
    xp = _ffn_fused(L.ffn, x_full, input_ids) if int(prefill_ctl.get("sp_ffn", int(_FFN_ENV))) else None
    if xp is None:
        xp = _ffn_partial(L.ffn, x_full, input_ids)
    st.update(xp=xp, residual=residual, post=post, res=res, pre=ffn_pre)
    return st


def _forward_kv(attn, x_list, positions, rows: _Rows) -> None:
    """Stock DeepseekV4Attention.forward_kv (attention.py:689-715) with the input projections on the local rows."""
    from typing import cast

    import torch

    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    _, _, kv, kv_score, _ = _projections(attn, x_list, rows, indexer_weights=False)
    if isinstance(attn_metadata, dict):
        cache = attn.swa_cache_layer.kv_cache
        torch.ops._C.fused_deepseek_v4_kv_rope_insert(
            kv, cache, cast(Any, attn_metadata[attn.swa_cache_layer.prefix]).slot_mapping, positions,
            attn.rotary_emb.cos_sin_cache, attn.swa_cache_layer.block_size,
            attn._flashinfer_fp8_kv_scale if cache.dtype == torch.float8_e4m3fn else None, attn.kv_mxfp8,
        )  # fmt: skip
    if attn.compressor is not None:
        latent = attn.compressor(kv_score, positions)
        attn.compressor.insert_cache(latent, positions, attn.rotary_emb)
        if attn.indexer is not None and attn.indexer.owns_k:
            attn.indexer._produce_k(latent, positions, attn.indexer_rotary_emb)


def _gather_rows(rows: _Rows, ridx, tensors):
    """Rows `ridx` (global row ids, int64 [R]) of per-chunk local-row tensor lists, assembled on every rank: the owner
    contributes its rows, everyone else zeros, one all-reduce per dtype (x + 0 = x: exact)."""
    import torch

    from vllm.distributed import tensor_model_parallel_all_reduce

    R = ridx.shape[0]
    flat = [None] * len(tensors)
    for k, ch in enumerate(rows.chunks):
        owned = (ridx >= ch.lo) & (ridx < ch.lo + ch.c)
        tail = (ridx >= ch.body) & (ridx < ch.g1)
        take = owned | (tail if rows.rank == 0 else torch.zeros_like(tail))
        local = torch.where(owned, ridx - ch.lo, ridx - ch.body + ch.c)
        local = torch.where(take, local, torch.zeros_like(local))
        for i, tl in enumerate(tensors):
            sel = tl[k].index_select(0, local).reshape(R, -1)
            sel = torch.where(take[:, None], sel, torch.zeros_like(sel))
            flat[i] = sel if flat[i] is None else flat[i] + sel  # rows of other chunks are exact zeros here
    out: list = [None] * len(tensors)
    dts = []
    for f in flat:  # first-appearance order: identical on every rank (a set's order is not)
        if f.dtype not in dts:
            dts.append(f.dtype)
    for dt in dts:
        idx = [i for i, f in enumerate(flat) if f.dtype == dt]
        buf = torch.cat([flat[i] for i in idx], dim=1) if len(idx) > 1 else flat[idx[0]]
        buf = tensor_model_parallel_all_reduce(buf.contiguous())
        parts = buf.split([flat[i].shape[1] for i in idx], dim=1)
        for i, p in zip(idx, parts):
            # column slices of the reduced buffer are strided views: the replay layers' kernels (stock mHC tilelang,
            # attention) require contiguous inputs
            out[i] = p.contiguous().view(R, *tensors[i][0].shape[1:])
    return out


def _replay_sp(drl, cs, positions, input_ids):
    """DecoderReplayLayers.__call__ (decoder_replay_layers.py) for an SP state: layer cut's write_kv on the local rows
    (attention boundary) + all rows (KV writes), then the replay layers on the replay rows gathered from their
    owners. Returns the stock outputs ([T, ...], replay rows filled, the rest zero)."""
    from vllm.forward_context import override_forward_context

    module, model, rows, st = cs["module"], cs["model"], cs["rows"], cs["st"]
    L = model.layers[model.decoder_replay_start]
    hc = (L.hc_attn_scale, L.hc_attn_base, L.rms_norm_eps, L.hc_eps, L.hc_eps, L.hc_post_alpha, L.hc_sinkhorn_iters)
    x_red, x_attn = [], []
    for k in range(rows.K):
        x_s, ev = rows.reduce_scatter(st["xp"], k)
        rows.wait(ev)
        x_red.append(x_s)
        *_, xa, _, _ = module.mhc_shifted_post_pre(
            x_s, st["residual"][k], st["post"][k], st["res"][k], L.hc_attn_fn, *hc, pre_mix=st["pre"][k],
            norm_weight=L.attn_norm.weight, norm_eps=L.attn_norm.variance_epsilon, stream=None, reduce_results=False,
        )  # fmt: skip
        x_attn.append(xa)
    _forward_kv(L.attn, x_attn, positions, rows)
    del x_attn
    rb = drl.replay_batch
    ridx = rb.rows
    x_r, pre_r, post_r, res_r, residual_r = _gather_rows(rows, ridx, (x_red, st["pre"], st["post"], st["res"], st["residual"]))
    T = rows.T
    with override_forward_context(rb.forward_context):
        if rb.run_graph is not None:
            full = []
            for t in (x_r, positions, input_ids, pre_r, post_r, res_r, residual_r):
                if t is None or t.shape[0] == T:
                    full.append(t)
                else:
                    full.append(t.new_empty((T, *t.shape[1:])).index_copy_(0, ridx, t))
            row_outputs = rb.run_graph(ridx, tuple(full))
        else:
            row_outputs = drl.run_layers(
                x_r, positions.index_select(0, ridx), None if input_ids is None else input_ids.index_select(0, ridx),
                pre_r, post_r, res_r, residual_r,
            )  # fmt: skip
    rows.drain()
    res_t = tuple(out.new_zeros((T, *out.shape[1:])).index_copy_(0, ridx, out) for out in row_outputs)
    prefill_ctx.note_replay(model, ridx, T)  # the drafter's context work of this step may skip the zero rows
    _mem("replay done")
    return res_t


def run_layers_sp(module, model, layer_ids, hidden_states, positions, input_ids, engram_hashes, engram_mask):
    """SP version of DeepseekV4Model._run_layers for the full-row layers; returns the stock tuple
    (hidden_states [T, H] reduced, residual [T, 4, H], post_mix, res_mix, pre_mix) on every rank, or hands the SP
    state to the decoder replay of the same forward (cut)."""
    from vllm.distributed import tensor_model_parallel_all_reduce

    T = positions.shape[0]
    K = max(1, int(prefill_ctl.get("sp_chunks", _CHUNKS_ENV)))
    overlap = bool(int(prefill_ctl.get("sp_overlap", int(_OVERLAP_ENV))))
    rows = _Rows(T, K, overlap)
    _log_once(f"sp{rows.K}{int(overlap)}", "ds41h prefill SP: layers %d-%d on 1/%d of the rows per rank (T=%d, %d chunks, overlap %s)",
              layer_ids[0], layer_ids[-1], rows.world, T, rows.K, overlap)
    st = dict(xp=hidden_states, residual=None, post=None, res=None, pre=None, engram_hashes=engram_hashes, engram_mask=engram_mask)
    _mem(f"start T={T}")
    for idx in layer_ids:
        st = _layer(module, model.layers[idx], rows, st, positions, input_ids)
        _mem(f"layer {idx}")
        if _detprobe.ENABLED:  # determinism probe (debugging only)
            _detprobe.record(f"sp.layers.{idx}", "out", (st["xp"], st["residual"], st["post"], st["res"], st["pre"]))
    drl = getattr(model, "decoder_replay_layers", None)
    if (SP_CUT and int(prefill_ctl.get("sp_cut", int(_SP_CUT_ENV))) and drl is not None
            and layer_ids[-1] + 1 == model.decoder_replay_start and drl.replay_batch is not None
            and model.layers[model.decoder_replay_start].engram is None):
        # The replay call of this forward takes the SP state (patched DecoderReplayLayers.__call__).
        rows.drain()
        _CUT_STATE[0] = dict(xp=st["xp"], module=module, model=model, rows=rows, st=st)
        _mem("exit (cut state)")
        return st["xp"], st["residual"], st["post"], st["res"], st["pre"]
    rows.drain()
    x = tensor_model_parallel_all_reduce(st["xp"])
    residual = rows.all_gather(st["residual"])
    post = rows.all_gather(st["post"])
    res = rows.all_gather(st["res"])
    pre = rows.all_gather(st["pre"])
    _mem("exit (full state)")
    return x, residual, post, res, pre


def install(module) -> None:
    """Patch vllm.models.deepseek_v41.nvidia.model.DeepseekV4Model._run_layers (idempotent)."""
    cls = module.DeepseekV4Model
    if getattr(cls, "_ds41h_sp_installed", False):
        return
    base = cls._run_layers

    def _run_layers(self, layer_ids, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual,
                    aux_hidden_by_layer, engram_hashes=None, engram_mask=None):  # fmt: skip
        if _eligible(self, layer_ids, hidden_states, positions, residual):
            return run_layers_sp(module, self, list(layer_ids), hidden_states, positions, input_ids, engram_hashes, engram_mask)
        big = _MEMLOG and residual is None and positions.shape[0] >= SP_MIN_T
        if big:
            _mem(f"base start T={positions.shape[0]}")
        out = base(self, layer_ids, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual,
                   aux_hidden_by_layer, engram_hashes, engram_mask)  # fmt: skip
        if big:
            _mem("base exit")
        return out

    _run_layers.__wrapped__ = base  # type: ignore[attr-defined]
    cls._run_layers = _run_layers
    cls._ds41h_sp_installed = True
    if SP_CUT:
        from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers

        base_call = DecoderReplayLayers.__call__

        def __call__(self, hidden_states, *states):
            cs = _CUT_STATE[0]
            if cs is not None:
                _CUT_STATE[0] = None
                if hidden_states is cs["xp"] and self.replay_batch is not None:
                    positions, input_ids = states[0], states[1]
                    return _replay_sp(self, cs, positions, input_ids)
                logger.warning("ds41h prefill SP: stale cut state dropped")
            return base_call(self, hidden_states, *states)

        __call__.__wrapped__ = base_call  # type: ignore[attr-defined]
        DecoderReplayLayers.__call__ = __call__
    logger.info("ds41h prefill SP: patched %s.DeepseekV4Model._run_layers (min_t=%d)", module.__name__, SP_MIN_T)
