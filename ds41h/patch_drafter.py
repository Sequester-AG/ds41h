"""DRAFTER hook: DSpark drafter fast paths (stock vLLM 0.32.0), flag DS41H_DRAFTER (default on with DS41H=1).

Patched symbols
  * vllm.v1.worker.gpu.spec_decode.dflash.speculator.prepare_dflash_inputs  (DS41H_DRAFTER_PREP)
      same kernel semantics as stock _prepare_dflash_inputs_kernel (dflash/speculator.py:540-735), but the
      graph-safety padding (query_start_loc / seq_lens / sample buffers / query slot mapping up to
      max_num_tokens) is spread over extra programs with 1024-wide blocks instead of the last request's
      program looping with BLOCK_SIZE = next_pow2(query len) (~10 us per call, 3 calls per draft step).
  * DSparkSpeculator.load_draft_model (dspark/speculator.py:84) -> after the stock load, installs
      - the routed-expert decode kernels (ds41h.drafter.MoEDecode) on every drafter layer's ds41h raw MXFP4
        method (instance-level apply wrapper, M <= 8 tokens; else the ds41h prefill W4A8 path)  (DS41H_DRAFTER_MOE)
      - a stacked e4m3 copy of the three layers' wkv rows (skinny format, views of the captured fused_wqa_wkv
        copies) for the context-KV precompute                                                  (DS41H_DRAFTER_CTXKV)
  * vllm.models.deepseek_v41.nvidia.dspark.DSparkDeepseekV4Model.precompute_and_store_context_kv
      -> one skinny GEMM over the stacked wkv rows (instead of three full 1792-row fused_wqa_wkv GEMMs whose
         q_lora half is discarded), then the stock kv_norm + fused RoPE/quant/insert per layer, for <= 8 rows.
  * DSparkSpeculator.propose (wrapper, DS41H_DRAFTER_PREP_BATCH=1 / DS41H_DRAFTER_SKIPCAT=1): the per-draft-KV-group
      prepare_dflash_inputs calls of one propose run as ONE launch (grid z = group); when this step's main_proj already
      ran on the side stream (early_main_proj, called from ds41h.patch_sampling), propose gets it as last_hidden_states
      with aux=None (no torch.cat(aux) + combine).
  * ctx-KV precompute: the three kv_norms in one Triton launch (ctx_kv_norm, DS41H_DRAFTER_CTXKV_NORM3=1).
  * drafter FFN override (install_ffn, DS41H_DRAFTER_FFN=1): stock gate -> ds41h route_topk (or the one-launch router,
      DS41H_DRAFTER_ROUTE1=0) -> MoEDecode with the shared expert folded in -> TP all-reduce (or fused into moe2,
      DS41H_DRAFTER_FFN_AR=0). DS41H_DRAFTER_DEBUG=N logs the first N path decisions per layer and rank.
All decisions use shapes/config only (rank-invariant: the fused-AR variant is a collective); no other collectives.
"""

from __future__ import annotations

import functools
import logging
import os

from ds41h import flags
from ds41h.patches import register

logger = logging.getLogger("ds41h.drafter")

ENABLED = flags.ENABLED and flags._flag("DS41H_DRAFTER")
PREP = ENABLED and flags._flag("DS41H_DRAFTER_PREP")
PREP_BATCH = PREP and flags._flag("DS41H_DRAFTER_PREP_BATCH", "1")  # one prepare launch for all draft KV-cache groups
SKIPCAT = PREP and flags._flag("DS41H_DRAFTER_SKIPCAT", "0")  # early main_proj result -> propose without torch.cat(aux)
MOE = ENABLED and flags._flag("DS41H_DRAFTER_MOE")
FFN = MOE and flags._flag("DS41H_DRAFTER_FFN")  # whole drafter FFN (gate/route + shared + routed) on the ds41h kernels
FFN_AR = FFN and flags._flag("DS41H_DRAFTER_FFN_AR", "0")  # TP all-reduce fused into moe2 (bitwise in test_drafter, but end-to-end/dr3g hung in the vLLM warmup with it on and dr4b passed without it: keep 0)
ROUTER = FFN and flags._flag("DS41H_DRAFTER_ROUTER", "0")  # router GEMV + top-k inside moe13 (else stock gate + ds41h route_topk)
ROUTE1 = FFN and flags._flag("DS41H_DRAFTER_ROUTE1", "0")  # one-launch router kernel (gate GEMV + top-k) before moe13
CTXKV = ENABLED and flags._flag("DS41H_DRAFTER_CTXKV")
# DS41H_DRAFTER_CTXKV_EARLY=1: the context-KV projection + kv_norms run when main_x is produced (early main_proj on the side
# stream during verification, or combine_hidden_states) into a persistent buffer; the captured drafter graph only inserts
# (implies the one-launch norm below)
CTXKV_EARLY = CTXKV and flags._flag("DS41H_DRAFTER_CTXKV_EARLY", "0")
# DS41H_DRAFTER_INS_PAR=1 (with CTXKV_EARLY): in the captured drafter graph the per-layer context-KV inserts (3 independent
# fused_deepseek_v4_kv_rope_insert launches into the 3 layers' caches, 2.5 us each, serialized) run as parallel graph branches,
# joined after the attention-metadata update (with EMBED_EARLY: at its join) -> the insert branch ends ~5 us earlier
INS_PAR = CTXKV_EARLY and flags._flag("DS41H_DRAFTER_INS_PAR", "0")
_INS_JOIN: list = []  # forked insert streams of the capture in flight (joined by _prepare_graph_forward / at the end)
_PGF_DEFER = [False]  # True while the EMBED_EARLY _prepare_graph_forward wrapper runs base_pgf (it joins after the metadata update)
# the drafter layers' kv_norms in one launch (default 0: +0.05 ms in situ on the critical path, r350b A/B)
CTXKV_NORM3 = CTXKV and (flags._flag("DS41H_DRAFTER_CTXKV_NORM3", "0") or CTXKV_EARLY)
CTXKV_BUF_ROWS = 256
# DS41H_DRAFTER_L0PRO=1: the first drafter layer (stock: hc broadcast + full stock mHC pre / projections, ~45 us) enters the
# ds41h prologue like layers 41/42, through an exact virtual previous sublayer: x_prev = embedding, residual = 0,
# post_mix = 1, res_mix = 0 (post -> 4 copies of the embedding, as the stock repeat) and pre_mix = one-hot stream 0 (stock
# pre_mix None = "select residual stream zero at model entry"); identical math, drafter-only
L0PRO = ENABLED and flags._flag("DS41H_DRAFTER_L0PRO", "0")
# DS41H_DRAFTER_FINAL_FUSE=1 (drafter-only, with L0PRO): the end of the drafter forward (stock mhc_post_tilelang + hc_collapse_triton,
# 2 launches) and the draft head's input RMSNorm (1 launch at the start of the draft-sampling graph) as ONE launch
# (kernels/dmhc.cu dfin::fin_kernel). The forward still returns the stock un-normed collapse c; the normed rows y are registered
# on the model (data_ptr / rows of c) and vp_sampling.draft_logits_local uses them when its input is a prefix view of c.
FINAL_FUSE = L0PRO and flags._flag("DS41H_DRAFTER_FINAL_FUSE", "0")
# DS41H_DRAFTER_EMBED_EARLY=1: in the captured drafter graph, the drafter's input embedding (vocab-parallel lookup + TP
# all-reduce) runs on a forked stream concurrently with the context-KV inserts + attention-metadata update
EMBED_EARLY = PREP and flags._flag("DS41H_DRAFTER_EMBED_EARLY", "0")
EMBED_PRIO = EMBED_EARLY and flags._flag("DS41H_DRAFTER_EMBED_PRIO", "1")  # the early-embedding branch on a high-priority stream
# DS41H_DRAFTER_EMBED_EAGER=1 (with EMBED_EARLY and DS41H_DRAFT_TAIL >= 1; drafter-only): the drafter's input embedding (lookup +
# TP all-reduce) is issued EAGERLY on a side stream right before the drafter graph is launched (its input ids come from the eager
# prepare kernel) into a persistent buffer, and the captured graph waits for it through an external event-wait node placed after
# the context-KV inserts / metadata update. In situ the in-graph embedding branch ran after the insert branch on 95 % of ranks x
# windows (dr4b_best) and its all-reduce then gated the first drafter layer (prologue 221 us vs inserts + swa done at 205).
EMBED_EAGER = (EMBED_EARLY and flags._flag("DS41H_DRAFTER_EMBED_EAGER", "0")
               and os.environ.get("DS41H_DRAFT_TAIL", "0").strip() in ("1", "2"))
_EEMB: dict = {}  # eager embedding state: "ev" external event, "buf" [max_tokens, H] persistent embedding, "n" num_reqs -> rows
MAIN = ENABLED and flags._flag("DS41H_DRAFTER_MAIN")  # main_proj GEMV kernel (<= 8 rows)
# DS41H_DRAFTER_MAIN_DST=1: the early main_proj result is copied into the speculator's hidden_states buffer on the side
# stream, so stock propose's self.hidden_states[:n].copy_(hidden_states[:n]) is a same-view no-op (aten copy_ exits early
# for views of the same data) instead of a DtoD on the critical path
MAIN_DST = MAIN and flags._flag("DS41H_DRAFTER_MAIN_DST", "0")
# DS41H_DRAFTER_MAIN_NORM_DST=1 (with MAIN_DST; drafter-only, bitwise): the early main_norm (vLLM rms_norm) writes straight into
# spec.hidden_states[:rows] (same ops.rms_norm kernel and arguments as RMSNorm.forward_cuda) instead of a fresh tensor + a DtoD
# copy: the copy + its gap sat at the end of the side-stream chain that the next propose waits for (dr4b: 182.6-183.6 us).
MAIN_NORM_DST = MAIN_DST and flags._flag("DS41H_DRAFTER_MAIN_NORM_DST", "0")
# drafter-layer o-projection (inverse RoPE + wo_a + wo_b) on the ds41h split-K FP8 GEMV (2 launches instead of 3 + gaps)
OPROJ = ENABLED and flags._flag("DS41H_DRAFTER_OPROJ", "0")
OPROJ_AR = OPROJ and flags._flag("DS41H_DRAFTER_OPROJ_AR", "0")  # TP all-reduce fused into the wo_b launch (moe2 fused-AR protocol)
# DS41H_DRAFTER_MHC=1: the drafter layers' FFN-side mHC (stock mhc_fused + mhc_pre_big_fuse_with_norm, 2 launches) as one
# ds41h launch (kernels/dmhc.cu; drafter-only, same math, mix summation order differs); dispatch by the layer's hc_ffn_fn
MHC = ENABLED and flags._flag("DS41H_DRAFTER_MHC", "0")
# DS41H_DRAFTER_MHC=2: + the router (gate GEMV over the normalized layer input + dsv4 top-k) fused into the same launch; the
# drafter FFN override then takes the routing from the stash instead of running gate + copy + topk (3 launches)
MHC_ROUTE = MHC and os.environ.get("DS41H_DRAFTER_MHC", "0").strip() == "2"
# DS41H_DRAFTER_MHC_CL=1: the FFN-side mHC (+ router with MHC=2) as one 16-CTA thread-block cluster (kernels/dmhc.cu cl_kernel:
# DSMEM reductions + 2 cluster barriers, router logits from c * w * r, Sinkhorn off the router's critical path); drafter-only
MHC_CL = MHC and flags._flag("DS41H_DRAFTER_MHC_CL", "0")
# DS41H_DRAFTER_DEBUG=N: log the first N drafter-FFN path decisions per layer on every rank (rank-divergence hunt)
DEBUG = int(os.environ.get("DS41H_DRAFTER_DEBUG", "0") or 0)

PAD_BLOCK = 1024
_DBG_COUNTS: dict[str, int] = {}


def _debug(layer: str, msg: str) -> None:
    n = _DBG_COUNTS.get(layer, 0)
    if n >= DEBUG:
        return
    _DBG_COUNTS[layer] = n + 1
    try:
        from ds41h.patch_core import current_step, tp_rank

        rank, step = tp_rank(), current_step()
    except Exception:  # noqa: BLE001 - diagnostics only
        rank, step = "?", None
    import torch

    logger.info("ds41h drafter-ffn dbg rank=%s %s #%d %s capturing=%s step=%s", rank, layer, n, msg,
                torch.cuda.is_current_stream_capturing(), step)


# ----------------------------------------------------------------------------------------- prepare inputs
def _prep_kernel():
    from vllm.triton_utils import tl, triton
    from vllm.v1.worker.gpu.cp_utils import cp_local_slot

    @triton.jit
    def _ds41h_prepare_dflash_inputs_kernel(
        out_input_ids_ptr,
        out_query_positions_ptr,
        out_query_start_loc_ptr,
        out_seq_lens_ptr,
        qsm0_ptr,
        qsm1_ptr,
        qsm2_ptr,
        qsm3_ptr,
        out_context_positions_ptr,
        csm0_ptr,
        csm1_ptr,
        csm2_ptr,
        csm3_ptr,
        out_sample_indices_ptr,
        out_sample_pos_ptr,
        out_sample_idx_mapping_ptr,
        out_temperature_ptr,
        out_seeds_ptr,
        target_positions_ptr,
        target_query_start_loc_ptr,
        idx_mapping_ptr,
        last_sampled_ptr,
        next_prefill_tokens_ptr,
        num_sampled_ptr,
        num_rejected_ptr,
        temperature_ptr,
        seeds_ptr,
        bt0_ptr,
        bt1_ptr,
        bt2_ptr,
        bt3_ptr,
        bts0,
        bts1,
        bts2,
        bts3,
        bsz0,
        bsz1,
        bsz2,
        bsz3,
        cp0_ptr,
        cp1_ptr,
        cp2_ptr,
        cp3_ptr,
        cps0,
        cps1,
        cps2,
        cps3,
        btl0,
        btl1,
        btl2,
        btl3,
        parallel_drafting_token_id,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        cp_rank,
        num_reqs,
        SAMPLE_FROM_ANCHOR: tl.constexpr,
        PAD_SLOT_ID: tl.constexpr,
        CP_SIZE: tl.constexpr,
        CP_INTERLEAVE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        PAD: tl.constexpr,
        COPY: tl.constexpr,
        CBLK: tl.constexpr,
    ):
        # Program (req, block, group): group-specific outputs are the query / context slot mappings (block table + block
        # size of that draft KV-cache group); group 0 also writes every shared output (identical for all groups in the
        # stock per-group calls, dflash/speculator.py:421-451), so G groups cost one launch instead of G.
        req_idx = tl.program_id(0)
        block_idx = tl.program_id(1)
        gi = tl.program_id(2)
        if gi == 0:
            out_query_slot_mapping_ptr = qsm0_ptr
            out_context_slot_mapping_ptr = csm0_ptr
            block_table_ptr = bt0_ptr
            block_table_stride = bts0
            block_size = bsz0
            copy_ptr = cp0_ptr
            copy_stride = cps0
            bt_len = btl0
        elif gi == 1:
            out_query_slot_mapping_ptr = qsm1_ptr
            out_context_slot_mapping_ptr = csm1_ptr
            block_table_ptr = bt1_ptr
            block_table_stride = bts1
            block_size = bsz1
            copy_ptr = cp1_ptr
            copy_stride = cps1
            bt_len = btl1
        elif gi == 2:
            out_query_slot_mapping_ptr = qsm2_ptr
            out_context_slot_mapping_ptr = csm2_ptr
            block_table_ptr = bt2_ptr
            block_table_stride = bts2
            block_size = bsz2
            copy_ptr = cp2_ptr
            copy_stride = cps2
            bt_len = btl2
        else:
            out_query_slot_mapping_ptr = qsm3_ptr
            out_context_slot_mapping_ptr = csm3_ptr
            block_table_ptr = bt3_ptr
            block_table_stride = bts3
            block_size = bsz3
            copy_ptr = cp3_ptr
            copy_stride = cps3
            bt_len = btl3
        shared = gi == 0
        if req_idx >= num_reqs:
            # Graph-safety padding (stock: the last request's program, block 0), chunk c of PAD entries.
            if block_idx == 0:
                i = (req_idx - num_reqs) * PAD + tl.arange(0, PAD)
                last_query_end = num_reqs * num_query_per_req
                tl.store(out_query_start_loc_ptr + i, last_query_end, mask=shared & (i >= num_reqs) & (i < max_num_reqs + 1))
                tl.store(out_seq_lens_ptr + i, 0, mask=shared & (i >= num_reqs) & (i < max_num_reqs))
                pad_s = shared & (i >= num_reqs * num_speculative_steps) & (i < max_num_reqs * num_speculative_steps)
                tl.store(out_sample_indices_ptr + i, 0, mask=pad_s)
                tl.store(out_sample_pos_ptr + i, 0, mask=pad_s)
                tl.store(out_sample_idx_mapping_ptr + i, -1, mask=pad_s)
                pad_q = (i >= last_query_end) & (i < max_num_tokens)
                tl.store(out_query_slot_mapping_ptr + i, PAD_SLOT_ID, mask=pad_q)
        else:
            req_state_idx = tl.load(idx_mapping_ptr + req_idx)
            ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
            ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
            num_ctx = ctx_end - ctx_start
            num_rejected = tl.load(num_rejected_ptr + req_idx)
            valid_ctx_end = ctx_end - num_rejected
            num_valid_ctx = valid_ctx_end - ctx_start
            num_sampled = tl.load(num_sampled_ptr + req_idx)
            if num_sampled > 0:
                bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
            else:
                bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)
            last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
            query_base = req_idx * num_query_per_req

            j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            is_ctx = j < num_ctx
            is_valid_ctx = j < num_valid_ctx
            is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)
            query_off = j - num_valid_ctx

            ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
            ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
            ctx_block_num = ctx_pos // (block_size * CP_SIZE)
            ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
            ctx_block_id = tl.load(block_table_ptr + req_idx * block_table_stride + ctx_block_num, mask=is_valid_ctx, other=0).to(tl.int64)
            ctx_resident = is_valid_ctx & (ctx_block_id != 0)
            local_ctx_slot = cp_local_slot(ctx_pos, ctx_block_id, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_SLOT_ID)
            ctx_slot = tl.where(ctx_resident, local_ctx_slot, PAD_SLOT_ID)
            tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=shared & is_ctx)
            tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

            query_pos = last_valid_pos + 1 + query_off
            query_idx = query_base + query_off
            is_bonus = is_query & (query_off == 0)
            input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)
            q_block_num = query_pos // (block_size * CP_SIZE)
            q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
            q_block_id = tl.load(block_table_ptr + req_idx * block_table_stride + q_block_num, mask=is_query, other=0).to(tl.int64)
            q_resident = is_query & (q_block_id != 0)
            local_q_slot = cp_local_slot(query_pos, q_block_id, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_SLOT_ID)
            q_slot = tl.where(q_resident, local_q_slot, PAD_SLOT_ID)
            tl.store(out_input_ids_ptr + query_idx, input_id, mask=shared & is_query)
            clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
            tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=shared & is_query)
            tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

            sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
            is_sample = shared & is_query & (query_off >= sample_off)
            sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
            sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
            tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
            tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
            tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

            if COPY:
                # FULL draft tail (patch_tail.DraftBlockTables): this group's block-table row -> the drafter's private copy
                if block_idx == 0:
                    for c0 in range(0, bt_len, CBLK):
                        cols = c0 + tl.arange(0, CBLK)
                        cm = cols < bt_len
                        row = tl.load(block_table_ptr + req_idx * block_table_stride + cols, mask=cm)
                        tl.store(copy_ptr + req_idx * copy_stride + cols, row, mask=cm)

            if (block_idx == 0) & shared:
                tl.store(out_query_start_loc_ptr + req_idx, query_base)
                tl.store(out_seq_lens_ptr + req_idx, tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len))
                tl.store(out_temperature_ptr + req_state_idx, tl.load(temperature_ptr + req_state_idx))
                tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))

    return _ds41h_prepare_dflash_inputs_kernel


_KERNEL = None
# Batching of the per-draft-KV-group prepare calls of one propose (DS41H_DRAFTER_PREP_BATCH): the patched propose sets
# "groups"; the calls are buffered and the last one launches every group at once.
_BATCH: dict = {"groups": 0, "pending": []}
_GROUP_KEYS = ("query_slot_mapping", "context_slot_mapping", "block_table", "block_size")


def _launch_prepare(calls: list[dict]) -> None:
    global _KERNEL
    from vllm.triton_utils import triton
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    if _KERNEL is None:
        _KERNEL = _prep_kernel()
    a = calls[0]
    G = len(calls)
    assert 1 <= G <= 4
    grp = calls + [calls[-1]] * (4 - G)  # unused slots repeat the last group (never read: grid z = G)
    input_batch = a["input_batch"]
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    num_query_per_req, num_speculative_steps = a["num_query_per_req"], a["num_speculative_steps"]
    max_num_reqs, max_num_tokens = a["max_num_reqs"], a["max_num_tokens"]
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    pad_extent = max(max_num_tokens, max_num_reqs * num_speculative_steps, max_num_reqs + 1)
    num_pad = triton.cdiv(pad_extent, PAD_BLOCK)
    ib = a["input_buffers"]
    # FULL draft tail: a group's block_table may be the drafter's private copy (patch_tail.DraftBlockTables); the kernel then
    # reads the runner's table (identical rows) and copies this step's rows into the private one
    srcs, dsts = [], []
    tail_src = _tail_sources()
    for c in grp:
        bt = c["block_table"]
        real = tail_src.get(bt.data_ptr()) if tail_src else None
        srcs.append(real if real is not None else bt)
        dsts.append(bt)
    copy = tail_src is not None and any(s is not d for s, d in zip(srcs, dsts))
    _KERNEL[(num_reqs + num_pad, num_blocks, G)](
        ib.input_ids,
        ib.positions,
        ib.query_start_loc,
        ib.seq_lens,
        *(c["query_slot_mapping"] for c in grp),
        a["context_positions"],
        *(c["context_slot_mapping"] for c in grp),
        a["sample_indices"],
        a["sample_pos"],
        a["sample_idx_mapping"],
        a["temperature"],
        a["seeds"],
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        a["last_sampled"],
        a["next_prefill_tokens"],
        a["num_sampled"],
        a["num_rejected"],
        a["input_temperature"],
        a["input_seeds"],
        *srcs,
        *(t.stride(0) for t in srcs),
        *(c["block_size"] for c in grp),
        *dsts,
        *(t.stride(0) for t in dsts),
        *(t.shape[1] for t in srcs),
        a["parallel_drafting_token_id"],
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        a["max_model_len"],
        a["cp_rank"],
        num_reqs,
        SAMPLE_FROM_ANCHOR=a["sample_from_anchor"],
        PAD_SLOT_ID=PAD_SLOT_ID,
        CP_SIZE=a["cp_size"],
        CP_INTERLEAVE=a["cp_interleave"],
        BLOCK_SIZE=BLOCK_SIZE,
        PAD=PAD_BLOCK,
        COPY=copy,
        CBLK=1024,
    )
    if copy:
        _tail_synced()


def _tail_sources():
    """{private draft block table data_ptr: runner table} while a FULL-tail propose waits for its sync, else None."""
    import sys

    pt = sys.modules.get("ds41h.patch_tail")
    return pt.pending_sources() if pt is not None and getattr(pt, "FULL", False) else None


def _tail_synced() -> None:
    import sys

    sys.modules["ds41h.patch_tail"].mark_synced()


def fast_prepare_dflash_inputs(
    input_buffers,
    query_slot_mapping,
    context_positions,
    context_slot_mapping,
    sample_indices,
    sample_pos,
    sample_idx_mapping,
    temperature,
    seeds,
    input_batch,
    num_sampled,
    num_rejected,
    last_sampled,
    next_prefill_tokens,
    input_temperature,
    input_seeds,
    block_table,
    block_size,
    cp_rank,
    cp_size,
    cp_interleave,
    parallel_drafting_token_id,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    sample_from_anchor=False,
):
    """Drop-in for stock prepare_dflash_inputs (dflash/speculator.py:733-814); inside a patched propose the per-group
    calls are batched into one launch (shared outputs are identical for every group)."""
    call = dict(
        input_buffers=input_buffers, query_slot_mapping=query_slot_mapping, context_positions=context_positions,
        context_slot_mapping=context_slot_mapping, sample_indices=sample_indices, sample_pos=sample_pos,
        sample_idx_mapping=sample_idx_mapping, temperature=temperature, seeds=seeds, input_batch=input_batch,
        num_sampled=num_sampled, num_rejected=num_rejected, last_sampled=last_sampled, next_prefill_tokens=next_prefill_tokens,
        input_temperature=input_temperature, input_seeds=input_seeds, block_table=block_table, block_size=block_size,
        cp_rank=cp_rank, cp_size=cp_size, cp_interleave=cp_interleave, parallel_drafting_token_id=parallel_drafting_token_id,
        num_query_per_req=num_query_per_req, num_speculative_steps=num_speculative_steps, max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens, max_model_len=max_model_len, sample_from_anchor=sample_from_anchor,
    )
    groups = _BATCH["groups"]
    if groups <= 1:
        _launch_prepare([call])
        return
    pending = _BATCH["pending"]
    if pending:
        first = pending[0]
        same = all(_same(first[k], call[k]) for k in call if k not in _GROUP_KEYS)
        if not same:  # not the per-group loop we batch: flush what we have (stock order) and run this call alone
            _launch_prepare(pending)
            pending.clear()
            _launch_prepare([call])
            return
    pending.append(call)
    if len(pending) >= groups:
        _launch_prepare(pending)
        pending.clear()


def _same(a, b) -> bool:
    import torch

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return a is b
    if isinstance(a, (int, float, bool, str)) or a is None:
        return a == b
    return a is b


def _flush_prepare() -> None:
    pending = _BATCH["pending"]
    if pending:
        _launch_prepare(pending)
        pending.clear()


def patch_dflash(module) -> None:
    from ds41h.patch_core import mark_active

    if PREP and hasattr(module, "prepare_dflash_inputs"):
        module._ds41h_stock_prepare_dflash_inputs = module.prepare_dflash_inputs
        module.prepare_dflash_inputs = fast_prepare_dflash_inputs
        mark_active("drafter", "dflash.speculator.prepare_dflash_inputs")


# ----------------------------------------------------------------------------------------- drafter MoE
def install_moe(model) -> int:
    """Wrap the ds41h raw MXFP4 method of every drafter layer: M <= 8 -> drafter.MoEDecode."""
    import torch

    from ds41h import drafter, moe

    n = 0
    for layer in getattr(model.model, "layers", []):
        try:
            re = moe._routed_experts(layer.ffn)
        except (TypeError, AttributeError):
            continue
        if not getattr(re, "ds41h_raw", False) or not re.w13_weight.is_cuda:
            continue
        qm = re.quant_method
        if getattr(qm, "_ds41h_drafter_moe", False):
            continue
        op = drafter.moe_decode_op(re.w13_weight.device)  # preallocates the h buffer (before graph capture)
        base_apply = qm.apply

        def apply(layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None, _base=base_apply, _op=op):
            if isinstance(x, torch.Tensor) and drafter.MoEDecode.eligible(x, topk_ids):
                return _op(x, topk_weights, topk_ids, layer.w13_weight, layer.w13_weight_scale, layer.w2_weight, layer.w2_weight_scale,
                           getattr(layer, "swiglu_limit", None))
            return _base(layer=layer, x=x, topk_weights=topk_weights, topk_ids=topk_ids, shared_experts=shared_experts,
                         shared_experts_input=shared_experts_input)

        qm.apply = apply
        qm._ds41h_drafter_moe = True
        n += 1
    return n


def _tail_fork() -> None:
    import sys

    pt = sys.modules.get("ds41h.patch_tail")
    if pt is not None and getattr(pt, "LATE_FORK", False):
        pt.mark_fork()


def install_ffn(model) -> int:
    """Replace the drafter layers' DeepseekV4MoE.forward (instance attribute) for <= 8 tokens: stock gate ->
    ds41h route_topk (stock dsv4_topk semantics) -> drafter MoE kernels with the shared expert folded in -> TP
    all-reduce. Anything ineligible runs the stock forward."""
    import torch

    from ds41h import drafter, moe

    n = 0
    ar_comm = None
    if FFN_AR:
        comm = drafter.draft_comm("moe-ar")  # collective: every rank reaches this (rank-invariant config)
        ar_comm = None if comm.disabled else comm
    all_layers = list(getattr(model.model, "layers", []))
    for layer in all_layers:
        ffn = getattr(layer, "ffn", None)
        try:
            re = moe._routed_experts(ffn)
        except (TypeError, AttributeError):
            logger.info("ds41h drafter FFN: %s skipped (no routed experts)", type(ffn).__name__)
            continue
        why = None
        gate = getattr(ffn, "gate", None)
        sh = getattr(ffn, "shared_experts", None)
        sw13 = None if sh is None else sh.gate_up_proj.__dict__.get("_ds41h_skinny")
        sw2 = None if sh is None else sh.down_proj.__dict__.get("_ds41h_skinny")
        if not getattr(re, "ds41h_raw", False) or getattr(ffn, "use_native_mega_moe", False):
            why = "experts not in the ds41h raw layout"
        elif gate is None or getattr(gate, "tid2eid", None) is not None or gate.e_score_correction_bias is None:
            why = "hash routing / no correction bias"
        elif getattr(ffn, "scoring_func", "sqrtsoftplus") != "sqrtsoftplus" or getattr(ffn, "use_sequence_parallel", False):
            why = "scoring / sequence parallel"
        elif sw13 is None or sw2 is None:
            why = "no skinny copies of the shared expert"
        if why is not None:
            logger.info("ds41h drafter FFN: layer %s skipped (%s)", getattr(ffn, "prefix", "?"), why)
            continue
        # vision checkpoints: image-sentinel tokens route with bias_vl (route_topk handles it; the fused router does not)
        bias_vl = getattr(gate, "bias_vl", None)
        image_lo = int(getattr(ffn, "image_sentinel_lo", 0) or 0)
        if bias_vl is None or image_lo == 0:
            bias_vl, image_lo = None, 0
        ffn._ds41h_route_cfg = (bias_vl, image_lo, int(ffn.n_activated_experts), float(ffn.routed_scaling_factor), bool(ffn.renormalize))
        pack = drafter.SharedPack(sw13, sw2)
        op = drafter.moe_decode_op(re.w13_weight.device)
        base_forward = ffn.forward
        topk = int(ffn.n_activated_experts)
        scale = float(ffn.routed_scaling_factor)
        renorm = bool(ffn.renormalize)
        limit = getattr(ffn, "swiglu_limit", None)
        reduce = bool(ffn.reduce_results) and ffn.tp_size > 1
        fused_ar = ar_comm if (reduce and ffn.tp_size == 4) else None

        def forward(hidden_states, input_ids=None, mega_gate_metadata=None, _ffn=ffn, _re=re, _base=base_forward, _pack=pack, _op=op,
                    _topk=topk, _scale=scale, _renorm=renorm, _limit=limit, _reduce=reduce, _bias_vl=bias_vl, _image_lo=image_lo,
                    _ar=fused_ar, _name=str(getattr(ffn, "prefix", "?")), _last=layer is all_layers[-1]):
            if _last:
                _tail_fork()  # FULL-tail late fork point (graph A capture only; ds41h/patch_tail.py LATE_FORK)
            x = hidden_states
            # Every branch below depends on shapes / config only (identical on all TP ranks): the fused-AR variant is a
            # collective, so a rank-divergent choice would deadlock (MoEDecode fixes layout up instead of declining).
            ok = isinstance(x, torch.Tensor) and x.dim() == 2 and drafter.MoEDecode.eligible(x, x[:, :_topk])
            if DEBUG:
                _debug(_name, f"M={tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__} ok={ok} "
                       f"ids={'none' if input_ids is None else tuple(input_ids.shape)} ar={_ar is not None} "
                       f"ptr16={x.data_ptr() % 16 if isinstance(x, torch.Tensor) else -1}")
            if not ok:
                return _base(hidden_states, input_ids, mega_gate_metadata)
            if _bias_vl is not None and input_ids is None:
                return _base(hidden_states, input_ids, mega_gate_metadata)
            pre = _ROUTED.pop(id(_ffn), None) if _ROUTED else None
            if pre is not None and pre[0] == x.data_ptr() and pre[1] == x.shape[0]:
                # routing already computed by the fused mHC + router launch (DS41H_DRAFTER_MHC=2)
                y = _op(x, pre[2], pre[3], _re.w13_weight, _re.w13_weight_scale, _re.w2_weight, _re.w2_weight_scale, _limit,
                        shared=_pack, ar=_ar)
                if _reduce and _ar is None:
                    from vllm.distributed import tensor_model_parallel_all_reduce

                    y = tensor_model_parallel_all_reduce(y)
                return y
            gw = _ffn.gate.weight
            if (ROUTER and _bias_vl is None and gw.dtype == torch.bfloat16 and gw.is_contiguous()
                    and _ffn.gate.e_score_correction_bias.dtype == torch.float32):
                # router GEMV + top-k fused into the expert kernels (moe13)
                y, _, _ = _op.routed(x, gw, _ffn.gate.e_score_correction_bias, _topk, _scale, _renorm, _re.w13_weight,
                                     _re.w13_weight_scale, _re.w2_weight, _re.w2_weight_scale, _limit, shared=_pack, ar=_ar)
            else:
                bias = _ffn.gate.e_score_correction_bias
                b32 = _ffn.__dict__.get("_ds41h_bias32")
                if b32 is None or b32[0] is not bias:
                    # route_topk feeds bias.float().contiguous(): converted once here (outside any capture) instead of a
                    # conversion kernel inside every drafter-layer replay (same fp32 values)
                    b32 = (bias, bias.float().contiguous()) if not torch.cuda.is_current_stream_capturing() else (bias, bias)
                    if b32[1] is not bias:
                        _ffn._ds41h_bias32 = b32
                bias = b32[1]
                if ROUTE1 and drafter.MoEDecode.route_eligible(x, gw, bias, _topk):
                    # gate GEMV + top-k in one launch (stock: cute_dsl gate GEMM + fp32 copy + moe_topk)
                    w, ids = _op.route(x, gw, bias, _topk, _scale, _renorm, bias_vl=_bias_vl,
                                       input_ids=input_ids if _bias_vl is not None else None, image_lo=_image_lo)
                else:
                    logits = _ffn.gate(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    w, ids = moe.route_topk(logits, bias, _topk, _scale, _renorm, bias_vl=_bias_vl,
                                            input_ids=input_ids if _bias_vl is not None else None, image_lo=_image_lo)
                y = _op(x, w, ids, _re.w13_weight, _re.w13_weight_scale, _re.w2_weight, _re.w2_weight_scale, _limit, shared=_pack, ar=_ar)
            if _reduce and _ar is None:
                from vllm.distributed import tensor_model_parallel_all_reduce

                y = tensor_model_parallel_all_reduce(y)
            return y

        ffn.forward = forward
        ffn._ds41h_drafter_ffn = True
        n += 1
    return n


# ----------------------------------------------------------------------------------------- o-projection
def install_oproj(model) -> int:
    """Instance-level _o_proj of every drafter attention: <= 8 rows -> drafter.OProj (+ TP all-reduce like wo_b's
    RowParallelLinear); anything else -> the class _o_proj (stock / ds41h decoder hooks). wo_a / wo_b come from the
    checkpoint (mtp.<i>.attn.wo_a / wo_b, this TP rank's slices) in skinny layout; numerics = stock W8A16 class."""
    import torch

    from ds41h import drafter, skinny
    from ds41h.patch_core import tp_rank, tp_size
    from ds41h.ref.weights import default_reader

    if tp_size() != 4:
        return 0
    rank = tp_rank()
    rd = default_reader()
    ar_comm = None
    if OPROJ_AR:
        comm = drafter.draft_comm("oproj-ar")  # collective: every rank reaches this (rank-invariant config)
        ar_comm = None if comm.disabled else comm
    n = 0
    for i, layer in enumerate(getattr(model.model, "layers", [])):
        attn = getattr(layer, "attn", None)
        if attn is None or getattr(attn, "gemm_rs", None) is not None or getattr(attn, "n_local_groups", 0) != 2:
            continue
        if getattr(attn, "nope_head_dim", 448) != 448 or getattr(attn, "rope_head_dim", 64) != 64 or getattr(attn, "o_lora_rank", 0) != 1024:
            continue
        dev = attn.wo_b.weight.device
        lay = rd.mtp(i)
        wa = skinny.pack(lay.tp("attn.wo_a.weight", rank, 4).to(dev), lay.tp("attn.wo_a.scale", rank, 4).to(dev))
        wb = attn.wo_b.__dict__.get("_ds41h_skinny")
        if wb is None or wb.n != 5120 or wb.k != 2048:
            wb = skinny.pack(lay.tp("attn.wo_b.weight", rank, 4).to(dev), lay.tp("attn.wo_b.scale", rank, 4).to(dev))
        reduce = bool(getattr(attn.wo_b, "reduce_results", True)) and int(getattr(attn.wo_b, "tp_size", 4)) > 1
        op = drafter.OProj(wa, wb, int(attn.n_local_heads), 2, ar=ar_comm if reduce else None)
        cls_o_proj = type(attn)._o_proj

        def o_proj(attn_out, positions, _attn=attn, _op=op, _reduce=reduce, _base=cls_o_proj):
            if getattr(_attn, "_ds41h_raw_o", False) or not isinstance(attn_out, torch.Tensor):
                return _base(_attn, attn_out, positions)
            o = attn_out[:, : _op.n_heads, :]
            if not _op.eligible(o, positions):  # shapes only: identical on every TP rank
                return _base(_attn, attn_out, positions)
            y = _op(o, positions, _attn.rotary_emb.cos_sin_cache)
            if _reduce and _op.ar is None:
                from vllm.distributed import tensor_model_parallel_all_reduce

                y = tensor_model_parallel_all_reduce(y)
            return y

        attn._o_proj = o_proj
        n += 1
    return n


# ----------------------------------------------------------------------------------------- context KV
def build_ctx_wkv(model):
    """Stacked skinny e4m3 copy of the wkv rows of every drafter layer's fused_wqa_wkv, or None."""
    import torch

    from ds41h.skinny import SkinnyWeight

    ws, ss = [], []
    for layer in model.model.layers:
        attn = layer.attn
        sw = attn.fused_wqa_wkv.__dict__.get("_ds41h_skinny")
        q = int(attn.q_lora_rank)
        if sw is None or q % 16 or sw.n != q + 512 or sw.k != 5120:
            return None
        kb = sw.k // 32
        t0 = q // 16
        ws.append(sw.w[t0 * kb * 512 :])
        ss.append(sw.s[t0:])
    return SkinnyWeight(w=torch.cat(ws).contiguous(), s=torch.cat(ss).contiguous(), n=512 * len(ws), k=5120)


_NORM3 = None


def _norm3_kernel():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _ds41h_ctx_kv_norm_kernel(x_ptr, x_stride, w_ptr, out_ptr, out_layer_stride, eps, D: tl.constexpr):
        # vLLM rms_norm rounding: bf16(bf16(x * rsqrt(mean(x^2) + eps)) * w), fp32 statistics
        t = tl.program_id(0)
        layer = tl.program_id(1)
        d = tl.arange(0, D)
        x = tl.load(x_ptr + t * x_stride + layer * D + d).to(tl.float32)
        r = tl.math.rsqrt(tl.sum(x * x, axis=0) / D + eps)
        y = (x * r).to(tl.bfloat16).to(tl.float32) * tl.load(w_ptr + layer * D + d).to(tl.float32)
        tl.store(out_ptr + layer * out_layer_stride + t * D + d, y.to(tl.bfloat16))

    return _ds41h_ctx_kv_norm_kernel


def ctx_kv_norm(kv_all, weights, eps: float, out=None):
    """kv_norm of every drafter layer in one launch: kv_all [T, L*512] bf16, weights [L, 512] bf16 -> [L, T, 512]
    (each layer's slice contiguous, as the stock per-layer RMSNorm output); out: optional [L, >=T, 512] view to write."""
    import torch

    global _NORM3
    if _NORM3 is None:
        _NORM3 = _norm3_kernel()
    T = kv_all.shape[0]
    L = weights.shape[0]
    if out is None:
        out = torch.empty((L, T, 512), dtype=torch.bfloat16, device=kv_all.device)
    assert out.stride(1) == 512 and out.stride(2) == 1
    _NORM3[(T, L)](kv_all, kv_all.stride(0), weights, out, out.stride(0), float(eps), D=512, num_warps=4)
    return out[:, :T]


def fill_ctx_kv(model, main_x) -> bool:
    """DS41H_DRAFTER_CTXKV_EARLY: the context KV (wkv rows of the drafter layers' fused_wqa_wkv + kv_norm) of
    main_x[:CTXKV_BUF_ROWS] into model._ds41h_kvn_buf (the captured drafter graph inserts from it). Same kernels as the
    in-graph path (skinny GEMM + ctx_kv_norm for <= 8 rows; stock projection + kv_norm otherwise). Current stream."""
    buf = model.__dict__.get("_ds41h_kvn_buf")
    if buf is None or main_x.shape[0] == 0:
        return False
    from ds41h import skinny

    rows = min(int(main_x.shape[0]), buf.shape[1])
    x = main_x[:rows]
    sw = model.__dict__.get("_ds41h_ctx_wkv")
    norm3 = model.__dict__.get("_ds41h_ctx_norm")
    if sw is not None and norm3 is not None and skinny.eligible(x, sw):
        ctx_kv_norm(skinny.gemm(x, sw), *norm3, out=buf)
        return True
    for i, layer in enumerate(model.layers):
        attn = layer.attn
        qr_kv, _ = attn.fused_wqa_wkv(x)
        buf[i, :rows].copy_(attn.kv_norm(qr_kv[..., attn.q_lora_rank :]))
    return True


def build_ctx_norm(model):
    """(stacked kv_norm weights [L, 512] bf16, eps) of the drafter layers, or None (different eps / no weight)."""
    import torch

    ws, eps = [], set()
    for layer in model.model.layers:
        n = getattr(layer.attn, "kv_norm", None)
        w = getattr(n, "weight", None)
        e = getattr(n, "variance_epsilon", None)
        if not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16 or w.numel() != 512 or e is None or getattr(n, "has_bias", False):
            return None
        ws.append(w.reshape(512))
        eps.add(float(e))
    if len(eps) != 1:
        return None
    return torch.stack(ws).contiguous(), eps.pop()


_SIDE: dict = {}


def side_stream(dev):
    """The drafter / sampling side stream of this device (early main_proj, early verify glue)."""
    import torch

    st = _SIDE.get(dev)
    if st is None:
        st = _SIDE[dev] = torch.cuda.Stream(device=dev)
    return st


def ins_streams(dev, n: int):
    """n side streams for the parallel context-KV insert branches (DS41H_DRAFTER_INS_PAR)."""
    import torch

    out = []
    for i in range(n):
        key = ("ins", i, str(dev))
        st = _SIDE.get(key)
        if st is None:
            st = _SIDE[key] = torch.cuda.Stream(device=dev)
        out.append(st)
    return out


def join_inserts(cur) -> None:
    """Join the forked insert branches of the capture in flight into `cur`."""
    while _INS_JOIN:
        cur.wait_stream(_INS_JOIN.pop())


def eager_embed_state(model, ids=None):
    """DS41H_DRAFTER_EMBED_EAGER: the persistent embedding buffer + the external event. Created at load time by
    prime_eager_embed (outside any capture); None if it does not exist and a capture is in progress."""
    import torch

    st = _EEMB.get("st")
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        w = getattr(getattr(model, "embed_tokens", None), "weight", None)
        if not (isinstance(w, torch.Tensor) and w.dim() == 2 and w.is_cuda):
            return None
        ev = torch.cuda.Event(external=True)
        # record once NOW: torch creates the CUDA event lazily at its first record, and Event.wait on a not-yet-created event is
        # a no-op -- a wait captured before any record would put NO event-wait node into the graph
        ev.record(torch.cuda.current_stream(w.device))
        st = _EEMB["st"] = {"ev": ev, "n": {}, "calls": 0, "buf": torch.zeros(512, w.shape[1], dtype=w.dtype, device=w.device)}
    return st


def prime_eager_embed(model) -> None:
    """Create the eager-embedding state at load time (outside any capture)."""
    if EMBED_EAGER and model is not None and "st" not in _EEMB:
        if eager_embed_state(model) is not None:
            logger.info("ds41h drafter: eager input embedding (side stream before the graph launch, external event wait)")


def eager_embed(spec, num_reqs: int) -> bool:
    """Before the drafter graph replay of `num_reqs` requests: embedding of the drafter input ids (written by the eager prepare
    kernel on the current stream) on the early-embedding side stream into the persistent buffer, then record the external
    event the graph waits on. False when this graph was captured without the eager embedding (nothing to do)."""
    import torch

    st = _EEMB.get("st")
    if st is None:
        return False
    n = st["n"].get(int(num_reqs))
    if n is None:
        return False
    model = getattr(getattr(spec, "model", None), "model", None)
    base_emb = getattr(type(model), "_ds41h_base_embed", None) if model is not None else None
    if base_emb is None:
        return False
    cur = torch.cuda.current_stream()
    side = embed_stream(cur.device) if EMBED_PRIO else side_stream(cur.device)
    side.wait_stream(cur)  # input ids (prepare kernel) + the previous graph's reads of the buffer are on / joined into cur
    with torch.cuda.stream(side):
        e = base_emb(model, spec.input_buffers.input_ids[:n])
        st["buf"][:n].copy_(e)
        st["ev"].record(side)
    st["calls"] += 1
    return True


def embed_stream(dev):
    """High-priority stream for the captured early-embedding branch (EMBED_EARLY): its kernel nodes carry the stream
    priority, so the graph runs the embedding + all-reduce first instead of after the context-KV insert branch."""
    import torch

    key = ("embed", str(dev))
    st = _SIDE.get(key)
    if st is None:
        lo, hi = torch.cuda.Stream.priority_range()
        st = _SIDE[key] = torch.cuda.Stream(device=dev, priority=hi)
    return st


def early_cat(spec, aux_hidden_states):
    """DS41H_DRAFTER_PRECAT: torch.cat(aux hidden states) on the side stream, forked from the current stream before the
    verification head GEMV is enqueued; early_main_proj(precat=...) then starts main_proj directly after the head event.
    Returns the concatenated tensor (side-stream allocation) or None if early_main_proj would not run."""
    import torch

    model = getattr(getattr(spec, "model", None), "model", None)
    mp = None if model is None else model.__dict__.get("_ds41h_main")
    if mp is None or torch.cuda.is_current_stream_capturing():
        return None
    rows = aux_hidden_states[0].shape[0]
    cols = sum(int(a.shape[-1]) for a in aux_hidden_states)
    if not (0 < rows <= mp.MAX_ROWS and cols == 15360 and all(a.dim() == 2 and a.shape[0] == rows for a in aux_hidden_states)):
        return None
    dev = aux_hidden_states[0].device
    st = side_stream(dev)
    st.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(st):
        x = torch.cat(aux_hidden_states, dim=-1)
    for a in aux_hidden_states:
        a.record_stream(st)
    return x


def early_main_proj(spec, aux_hidden_states, after_event=None, precat=None) -> bool:
    """Run the DSpark main_proj GEMV + main_norm of this step's target aux hidden states on a side stream, overlapping
    the (latency-bound) verification chain on the main stream; the patched combine_hidden_states of the same step
    consumes the result. aux_hidden_states: the list stock propose concatenates (torch.cat(aux, -1)).
    Same kernels and inputs as the in-propose path, so the drafter numerics are unchanged. No collectives.
    Returns True if launched."""
    import torch

    model = getattr(getattr(spec, "model", None), "model", None)
    mp = None if model is None else model.__dict__.get("_ds41h_main")
    if mp is None or not aux_hidden_states or torch.cuda.is_current_stream_capturing():
        return False
    rows = aux_hidden_states[0].shape[0]
    cols = sum(int(a.shape[-1]) for a in aux_hidden_states)
    if not (0 < rows <= mp.MAX_ROWS and cols == 15360 and all(a.dim() == 2 and a.shape[0] == rows for a in aux_hidden_states)):
        return False
    dev = aux_hidden_states[0].device
    st = side_stream(dev)
    main = torch.cuda.current_stream(dev)
    if after_event is not None:
        st.wait_event(after_event)
    else:
        st.wait_stream(main)
    with torch.cuda.stream(st):
        if precat is not None and precat.dim() == 2 and precat.shape[0] == rows and precat.shape[1] == cols:
            x = precat  # concatenated on this stream before the head (early_cat)
        else:
            x = torch.cat(aux_hidden_states, dim=-1)
        y = mp(x)
        hs = getattr(spec, "hidden_states", None) if MAIN_DST else None
        hs_ok = (hs is not None and hs.dim() == 2 and hs.shape[0] >= rows and hs.shape[1] == y.shape[1] and hs.dtype == y.dtype
                 and hs.device == y.device and hs.is_contiguous())
        if MAIN_NORM_DST and hs_ok and _plain_rms(model.main_norm) and y.is_contiguous():
            from vllm import _custom_ops as ops

            out = hs[:rows]  # nothing reads spec.hidden_states between the previous drafter graph and this step's propose
            ops.rms_norm(out, y, model.main_norm.weight.data, model.main_norm.variance_epsilon)
            hs_ok = False  # already in place
        else:
            out = model.main_norm(y)
        if CTXKV_EARLY:
            fill_ctx_kv(model, out)  # context KV off the critical path as well (the drafter graph only inserts)
        if hs_ok:
            dst = hs[:rows]
            dst.copy_(out)  # nothing reads spec.hidden_states between the previous drafter graph and this step's propose
            out = dst
        ev = torch.cuda.Event()
        ev.record(st)
    for a in aux_hidden_states:
        a.record_stream(st)
    model._ds41h_pending = dict(rows=rows, cols=cols, out=out, event=ev, keep=x)
    return True


def _plain_rms(norm) -> bool:
    """vLLM RMSNorm whose forward is ops.rms_norm(out, x, weight, eps) (bf16 weight, no bias / residual variant)."""
    import torch

    w = getattr(norm, "weight", None)
    return (type(norm).__name__ == "RMSNorm" and isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16
            and isinstance(getattr(norm, "variance_epsilon", None), float) and getattr(norm, "has_weight", True)
            and not getattr(norm, "has_bias", False) and getattr(norm, "variance_size_override", None) is None)


def _use_early_main(spec, args):
    """propose(input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states, ...): if this step's
    main_proj already ran on the side stream, hand its result in as last_hidden_states with aux None, so stock propose
    skips torch.cat(aux) + combine_hidden_states (dflash/speculator.py:383-389; nothing else reads either argument)."""
    import torch

    model = getattr(getattr(spec, "model", None), "model", None)
    pending = None if model is None else model.__dict__.get("_ds41h_pending")
    aux = args[4]
    if pending is None or not isinstance(aux, (list, tuple)) or not aux or aux[0].shape[0] != pending["rows"]:
        return args
    model.__dict__.pop("_ds41h_pending", None)
    cur = torch.cuda.current_stream()
    cur.wait_event(pending["event"])
    out = pending["out"]
    out.record_stream(cur)
    args = list(args)
    args[3], args[4] = out, None
    return tuple(args)


def drop_early_main_proj(spec) -> None:
    """Discard an unconsumed early main_proj result (e.g. the step ran no propose)."""
    model = getattr(getattr(spec, "model", None), "model", None)
    if model is not None:
        model.__dict__.pop("_ds41h_pending", None)


# ----------------------------------------------------------------------------------------- drafter final mHC + norm
def final_fuse(model, x, residual, post_mix, res_mix, pre_mix):
    """DS41H_DRAFTER_FINAL_FUSE: c = collapse(mhc_post(x, residual, post_mix, res_mix), pre_mix) and y = model.norm(c) in one
    launch; registers y as model._ds41h_normed = (c.data_ptr(), rows, y) (cleared at the start of every forward; captured pairs
    also in model._ds41h_normed_keep). None (-> stock path) when not eligible."""
    import torch

    from ds41h import drafter

    norm = getattr(model, "norm", None)
    if norm is None or not drafter.HeadFP8.norm_fusable(norm):
        return None
    m = x.shape[0]
    if not (0 < m <= 8 and x.dim() == 2 and x.shape[1] == 5120 and x.dtype == torch.bfloat16 and x.stride(1) == 1
            and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0 and residual.is_contiguous() and residual.dtype == torch.bfloat16
            and residual.shape[0] == m and residual.data_ptr() % 16 == 0):
        return None
    post = post_mix if post_mix.is_contiguous() else post_mix.contiguous()
    comb = res_mix if res_mix.is_contiguous() else res_mix.contiguous()
    pre = pre_mix if pre_mix.is_contiguous() else pre_mix.contiguous()
    if not (post.dtype == comb.dtype == pre.dtype == torch.float32 and post.numel() == 4 * m and comb.numel() == 16 * m
            and pre.numel() == 4 * m):
        return None
    c = torch.empty(m, 5120, dtype=torch.bfloat16, device=x.device)
    y = torch.empty(m, 5120, dtype=torch.bfloat16, device=x.device)
    mhc_ext().final_fuse(x, residual, post, comb, pre, norm.weight, float(norm.variance_epsilon), c, y, drafter.PDL)
    model._ds41h_normed = (c.data_ptr(), m, y)
    if torch.cuda.is_current_stream_capturing():
        # captured graphs (graph pools): keep every captured c -> y pair alive for the graphs' lifetime
        model.__dict__.setdefault("_ds41h_normed_keep", {})[c.data_ptr()] = (m, y)
    return c


# ----------------------------------------------------------------------------------------- drafter FFN-side mHC
_MHC_FNS: dict[int, object] = {}  # data_ptr of a drafter layer's hc_ffn_fn -> that layer
_MHC_STATE: dict[str, tuple] = {}  # device -> (scratch, ctr, router scratch)
_ROUTED: dict[int, tuple] = {}  # id(ffn) -> (layer_input data_ptr, m, weights, ids) from the fused mHC + router launch
_CUR_IDS: list = [None]  # input ids of the drafter layer forward in flight (bias_vl routing of image-sentinel tokens)


@functools.cache
def mhc_ext():
    from ds41h import cuext

    mod = cuext.load("ds41h_dmhc", "dmhc.cu")
    mod.preload()
    if MHC_CL:
        n = int(mod.cl_max_active(bool(MHC_ROUTE)))
        if n > 0:
            mod.mhc_set_variant(1)
            logger.info("ds41h drafter: FFN-side mHC as a 16-CTA cluster kernel (max active clusters %d)", n)
        else:
            logger.warning("ds41h drafter: 16-CTA cluster kernel cannot run on this device (occupancy 0); v1 dmhc kernel")
    return mod


def _mhc_state(dev):
    import torch

    st = _MHC_STATE.get(str(dev))
    if st is None:
        st = _MHC_STATE[str(dev)] = (torch.zeros(int(mhc_ext().scratch_floats()), dtype=torch.float32, device=dev),
                                     torch.zeros(4, dtype=torch.int32, device=dev),
                                     torch.zeros(int(mhc_ext().rpart_floats()), dtype=torch.float32, device=dev))
    return st


def install_mhc(model) -> int:
    """Register the drafter layers' FFN-side mHC parameters (dispatch key: hc_ffn_fn) and build the kernel + scratch now."""
    n = 0
    for layer in getattr(model.model, "layers", []):
        fn = getattr(layer, "hc_ffn_fn", None)
        if fn is None or fn.dtype.itemsize != 4 or fn.dim() != 2 or fn.shape[0] != 24 or fn.shape[1] != 4 * 5120 or not fn.is_contiguous():
            continue
        _MHC_FNS[fn.data_ptr()] = layer
        _mhc_state(fn.device)
        if MHC_ROUTE and "_ds41h_mhc_fwd" not in layer.__dict__:
            base_fwd = layer.forward  # bound (class) forward; record input_ids for the fused router's bias_vl rule

            def fwd(*args, _base=base_fwd, **kwargs):
                _CUR_IDS[0] = args[2] if len(args) > 2 else kwargs.get("input_ids")
                try:
                    return _base(*args, **kwargs)
                finally:
                    _CUR_IDS[0] = None

            layer.forward = fwd
            layer._ds41h_mhc_fwd = True
        n += 1
    if n:
        mhc_ext()
    return n


def _wrap_mhc(module) -> None:
    from ds41h.patch_core import mark_active

    base = module.mhc_shifted_post_pre
    if getattr(base, "_ds41h_mhc", False):
        return

    def mhc_shifted_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                             hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix=None, norm_weight=None,
                             norm_eps=1e-6, capture_aux=False, *, stream=None, reduce_results=False):
        import torch

        if (_MHC_FNS and fn.data_ptr() in _MHC_FNS and stream is None and not reduce_results and not capture_aux
                and isinstance(x, torch.Tensor) and 0 < x.shape[0] <= 8 and pre_mix is not None and norm_weight is not None
                and x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16 and x.is_contiguous() and residual.is_contiguous()
                and post_layer_mix.is_contiguous() and comb_res_mix.is_contiguous() and pre_mix.is_contiguous()):
            m = x.shape[0]
            dev = x.device
            res_out = torch.empty_like(residual)
            post = torch.empty(m, 4, 1, dtype=torch.float32, device=dev)
            comb = torch.empty(m, 4, 4, dtype=torch.float32, device=dev)
            layer_input = torch.empty(m, x.shape[1], dtype=torch.bfloat16, device=dev)
            pre_out = torch.empty(m, 4, dtype=torch.float32, device=dev)
            scratch, ctr, rpart = _mhc_state(dev)
            ffn = getattr(_MHC_FNS[fn.data_ptr()], "ffn", None)
            cfg = getattr(ffn, "_ds41h_route_cfg", None) if MHC_ROUTE else None
            gate = getattr(ffn, "gate", None)
            gw = getattr(gate, "weight", None)
            bias = getattr(gate, "e_score_correction_bias", None)
            rargs = (None, None, None, None, 0, None, None, None, 0.0, False, 0)
            ids_t = None
            if (cfg is not None and isinstance(gw, torch.Tensor) and gw.dtype == torch.bfloat16 and gw.is_contiguous() and gw.dim() == 2
                    and gw.shape[0] <= 128 and isinstance(bias, torch.Tensor) and bias.dtype == torch.float32):
                bias_vl, image_lo, topk, rscale, renorm = cfg
                ids_in = None
                if bias_vl is not None:
                    ids_in = _CUR_IDS[0]
                    if ids_in is None:
                        cfg = None  # stock routing (needs the token ids)
                    elif ids_in.dtype != torch.int32 or not ids_in.is_contiguous():
                        ids_in = ids_in.reshape(-1).to(torch.int32).contiguous()
                if cfg is not None:
                    ids_t = torch.empty(m, topk, dtype=torch.int32, device=dev)
                    w_t = torch.empty(m, topk, dtype=torch.float32, device=dev)
                    bvl = None if bias_vl is None else (bias_vl if bias_vl.dtype == torch.float32 and bias_vl.is_contiguous() else bias_vl.float().contiguous())
                    rargs = (gw, bias, bvl, ids_in, int(image_lo), rpart, ids_t, w_t, rscale, renorm, topk)
            mhc_ext().mhc_ffn(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, pre_mix, norm_weight, res_out, post,
                              comb, layer_input, pre_out, scratch, ctr, float(rms_eps), float(hc_pre_eps), float(hc_sinkhorn_eps),
                              float(hc_post_mult_value), int(sinkhorn_repeat), float(norm_eps), True, *rargs)
            if ids_t is not None:
                _ROUTED[id(ffn)] = (layer_input.data_ptr(), m, rargs[7], ids_t)
            return res_out, post, comb, layer_input, pre_out, x.new_empty(0, x.shape[1])
        return base(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat, pre_mix=pre_mix, norm_weight=norm_weight, norm_eps=norm_eps,
                    capture_aux=capture_aux, stream=stream, reduce_results=reduce_results)

    mhc_shifted_post_pre._ds41h_mhc = True
    mhc_shifted_post_pre.__wrapped__ = base
    module.mhc_shifted_post_pre = mhc_shifted_post_pre
    mark_active("drafter", "deepseek_v41.model.mhc_shifted_post_pre(drafter FFN-side mHC, one launch)")


def patch_dspark_model(module) -> None:
    from ds41h.patch_core import mark_active

    cls = module.DSparkDeepseekV4Model
    if MAIN:
        base_combine = cls.combine_hidden_states

        def combine_hidden_states(self, aux_hidden_states):
            pending = self.__dict__.pop("_ds41h_pending", None)
            if pending is not None and pending["rows"] == aux_hidden_states.shape[0] and pending["cols"] == aux_hidden_states.shape[1]:
                # main_proj + main_norm already ran on the side stream during verification (early_main_proj)
                import torch

                cur = torch.cuda.current_stream()
                cur.wait_event(pending["event"])
                out = pending["out"]
                out.record_stream(cur)
                return out
            mp = self.__dict__.get("_ds41h_main")
            if mp is not None and mp.eligible(aux_hidden_states):
                out = self.main_norm(mp(aux_hidden_states))
            else:
                out = base_combine(self, aux_hidden_states)
            if CTXKV_EARLY:
                fill_ctx_kv(self, out)
            return out

        cls.combine_hidden_states = combine_hidden_states
        mark_active("drafter", "DSparkDeepseekV4Model.combine_hidden_states")
    if not CTXKV:
        return
    base = cls.precompute_and_store_context_kv
    insert = module._insert_context_kv

    def precompute_and_store_context_kv(self, main_x, context_positions, context_slot_mappings=None):
        buf = self.__dict__.get("_ds41h_kvn_buf")
        if buf is not None and main_x.shape[0] <= buf.shape[1] and main_x.shape[0] > 0:
            import torch

            if torch.cuda.is_current_stream_capturing():
                # captured drafter graph: main_x's context KV was produced when main_x was (fill_ctx_kv: early main_proj
                # on the side stream / combine_hidden_states); only the inserts are replayed
                T = main_x.shape[0]
                jobs = [(layer.attn, buf[i, :T], context_slot_mappings[i]) for i, layer in enumerate(self.layers)
                        if context_slot_mappings is not None and context_slot_mappings[i] is not None]
                if INS_PAR and len(jobs) > 1:
                    cur = torch.cuda.current_stream()
                    sts = ins_streams(cur.device, len(jobs) - 1)
                    for (attn, kv, sm), st in zip(jobs[1:], sts):
                        st.wait_stream(cur)  # fork
                        with torch.cuda.stream(st):
                            insert(attn, kv, context_positions, sm)
                    insert(jobs[0][0], jobs[0][1], context_positions, jobs[0][2])
                    _INS_JOIN.extend(sts)
                    if not _PGF_DEFER[0]:
                        join_inserts(cur)  # otherwise joined after the metadata update by _prepare_graph_forward
                    return
                for attn, kv, sm in jobs:
                    insert(attn, kv, context_positions, sm)
                return
        sw = self.__dict__.get("_ds41h_ctx_wkv")
        if sw is None or main_x.shape[0] == 0:
            return base(self, main_x, context_positions, context_slot_mappings)
        from ds41h import skinny

        if not skinny.eligible(main_x, sw):
            return base(self, main_x, context_positions, context_slot_mappings)
        kv_all = skinny.gemm(main_x, sw)  # [T, 512 * layers]
        norm3 = self.__dict__.get("_ds41h_ctx_norm") if CTXKV_NORM3 else None
        kvn = ctx_kv_norm(kv_all, *norm3) if norm3 is not None else None  # [layers, T, 512]: every kv_norm in one launch
        for i, layer in enumerate(self.layers):
            slot_mapping = None if context_slot_mappings is None else context_slot_mappings[i]
            attn = layer.attn
            kv = kvn[i] if kvn is not None else attn.kv_norm(kv_all[:, 512 * i : 512 * (i + 1)])
            if slot_mapping is None:
                continue
            insert(attn, kv, context_positions, slot_mapping)

    cls.precompute_and_store_context_kv = precompute_and_store_context_kv
    mark_active("drafter", "DSparkDeepseekV4Model.precompute_and_store_context_kv")
    if L0PRO:
        base_fwd = cls.forward
        mhc_post = module.mhc_post_tilelang
        collapse = module.hc_collapse_triton

        def forward(self, input_ids, positions, inputs_embeds=None):
            import torch

            self.__dict__.pop("_ds41h_normed", None)  # FINAL_FUSE registration of the previous forward is stale from here on
            if self.use_sequence_parallel or self.use_native_mega_moe or positions.shape[0] == 0 or positions.shape[0] > 8:
                return base_fwd(self, input_ids, positions, inputs_embeds)
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            m, hc = inputs_embeds.shape[0], self.hc_mult
            v = self.__dict__.get("_ds41h_l0v")
            if v is None or v[0].device != inputs_embeds.device:
                dev = inputs_embeds.device
                onehot = torch.zeros(8, hc, dtype=torch.float32, device=dev)
                onehot[:, 0] = 1.0
                v = (torch.zeros(8, hc, inputs_embeds.shape[1], dtype=torch.bfloat16, device=dev),  # residual
                     torch.ones(8, hc, 1, dtype=torch.float32, device=dev),  # post_mix
                     torch.zeros(8, hc, hc, dtype=torch.float32, device=dev),  # res_mix
                     onehot)  # pre_mix
                self._ds41h_l0v = v
            residual, post_mix, res_mix, pre_mix = v[0][:m], v[1][:m], v[2][:m], v[3][:m]
            hidden_states = inputs_embeds
            for layer in self.layers:
                hidden_states, residual, post_mix, res_mix, pre_mix, _ = layer(
                    hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual, mega_gate_metadata=None)
            if FINAL_FUSE:
                c = final_fuse(self, hidden_states, residual, post_mix, res_mix, pre_mix)
                if c is not None:
                    return c
            hidden_states = mhc_post(hidden_states, residual, post_mix, res_mix)
            return collapse(hidden_states, pre_mix)

        cls.forward = forward
        mark_active("drafter", "DSparkDeepseekV4Model.forward(first layer through the prologue)")
    if EMBED_EARLY:
        base_embed = cls.embed_input_ids

        def embed_input_ids(self, input_ids):
            p = self.__dict__.pop("_ds41h_emb", None)
            if p is not None and p[0] == input_ids.data_ptr() and p[1] == input_ids.shape[0]:
                return p[2]  # computed (and joined) in the captured _prepare_graph_forward
            return base_embed(self, input_ids)

        cls.embed_input_ids = embed_input_ids
        cls._ds41h_base_embed = base_embed
        mark_active("drafter", "DSparkDeepseekV4Model.embed_input_ids(early, captured)")


# ----------------------------------------------------------------------------------------- speculator
def patch_speculator(module) -> None:
    from ds41h.patch_core import mark_active

    cls = module.DSparkSpeculator
    base_load = cls.load_draft_model

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = base_load(self, target_model, target_attn_layer_names)
        if EMBED_EAGER:
            try:
                prime_eager_embed(getattr(model, "model", None))
            except Exception:  # noqa: BLE001 - the in-graph early embedding stays
                logger.exception("ds41h drafter: eager embedding unavailable")
        if MOE:
            try:
                n = install_moe(model)
                logger.info("ds41h drafter: routed-expert decode kernels on %d drafter layers", n)
            except Exception:  # noqa: BLE001 - stock/ds41h prefill MoE path stays
                logger.exception("ds41h drafter: MoE decode install failed; drafter MoE stays on the W4A8 path")
        if FFN:
            try:
                n = install_ffn(model)
                logger.info("ds41h drafter: fused drafter FFN (route + shared + routed experts) on %d layers", n)
            except Exception:  # noqa: BLE001
                logger.exception("ds41h drafter: fused FFN install failed; drafter FFN stays on the MoE runner")
        if OPROJ:
            try:
                n = install_oproj(model)
                logger.info("ds41h drafter: o-projection kernels (inverse RoPE + wo_a + wo_b) on %d drafter layers", n)
            except Exception:  # noqa: BLE001
                logger.exception("ds41h drafter: o-projection install failed; stock _o_proj")
        if MHC:
            try:
                n = install_mhc(model)
                logger.info("ds41h drafter: FFN-side mHC kernel on %d drafter layers", n)
            except Exception:  # noqa: BLE001
                _MHC_FNS.clear()
                logger.exception("ds41h drafter: FFN-side mHC kernel unavailable; stock tilelang mHC")
        if MAIN:
            try:
                from ds41h import drafter

                model.model._ds41h_main = drafter.MainProj(model.model.main_norm.weight.device)
                logger.info("ds41h drafter: main_proj GEMV kernel (checkpoint mtp.0.main_proj)")
            except Exception:  # noqa: BLE001
                logger.exception("ds41h drafter: main_proj fast path unavailable")
        if CTXKV:
            try:
                sw = build_ctx_wkv(model)
                if sw is not None:
                    model.model._ds41h_ctx_wkv = sw
                    if CTXKV_NORM3:
                        norm3 = build_ctx_norm(model)
                        if norm3 is not None:
                            import torch

                            # JIT-compile the Triton kernel now (its first real call is inside the dspark graph capture)
                            ctx_kv_norm(torch.zeros(1, 512 * norm3[0].shape[0], dtype=torch.bfloat16, device=norm3[0].device), *norm3)
                            model.model._ds41h_ctx_norm = norm3
                            if CTXKV_EARLY:
                                model.model._ds41h_kvn_buf = torch.zeros(
                                    (norm3[0].shape[0], CTXKV_BUF_ROWS, 512), dtype=torch.bfloat16, device=norm3[0].device)
                                logger.info("ds41h drafter: early context KV (buffer %d rows; graph replays insert only)", CTXKV_BUF_ROWS)
                    logger.info("ds41h drafter: context-KV precompute on stacked wkv rows [%d x %d] e4m3", sw.n, sw.k)
                else:
                    logger.info("ds41h drafter: no skinny fused_wqa_wkv copies; stock context-KV precompute")
            except Exception:  # noqa: BLE001
                logger.exception("ds41h drafter: context-KV fast path unavailable")
        return model

    cls.load_draft_model = load_draft_model
    mark_active("drafter", "DSparkSpeculator.load_draft_model(moe, ctx-kv)")
    if EMBED_EARLY and hasattr(cls, "_prepare_graph_forward"):
        base_pgf = cls._prepare_graph_forward

        def _prepare_graph_forward(self, num_reqs, attn_metadata):
            import torch

            model = getattr(getattr(self, "model", None), "model", None)
            base_emb = getattr(type(model), "_ds41h_base_embed", None) if model is not None else None
            if base_emb is None or not torch.cuda.is_current_stream_capturing():
                return base_pgf(self, num_reqs, attn_metadata)
            n = num_reqs * self.num_query_per_req
            ids = self.input_buffers.input_ids[:n]
            cur = torch.cuda.current_stream()
            if EMBED_EAGER:
                st = eager_embed_state(model, ids)
                if st is not None and n <= st["buf"].shape[0]:
                    _PGF_DEFER[0] = True
                    try:
                        out = base_pgf(self, num_reqs, attn_metadata)  # inserts + metadata update (no embedding in the graph)
                    finally:
                        _PGF_DEFER[0] = False
                    join_inserts(cur)
                    cur.wait_event(st["ev"])  # external event-wait node: the eager side-stream embedding of this step
                    model._ds41h_emb = (ids.data_ptr(), n, st["buf"][:n])
                    st["n"][int(num_reqs)] = n
                    return out
            side = embed_stream(cur.device) if EMBED_PRIO else side_stream(cur.device)
            side.wait_stream(cur)  # fork (inside the capture)
            with torch.cuda.stream(side):
                emb = base_emb(model, ids)
            _PGF_DEFER[0] = True
            try:
                out = base_pgf(self, num_reqs, attn_metadata)  # context-KV precompute + metadata update, concurrently
            finally:
                _PGF_DEFER[0] = False
            join_inserts(cur)  # INS_PAR: the parallel insert branches (no-op otherwise)
            cur.wait_stream(side)  # join before the forward
            model._ds41h_emb = (ids.data_ptr(), n, emb)
            return out

        cls._prepare_graph_forward = _prepare_graph_forward
        mark_active("drafter", "DSparkSpeculator._prepare_graph_forward(early embedding)")
    if PREP and (PREP_BATCH or SKIPCAT):
        base_propose = cls.propose

        def propose(self, *args, **kwargs):
            if SKIPCAT and len(args) >= 5:
                args = _use_early_main(self, args)
            # batch the per-draft-KV-group prepare_dflash_inputs calls of this propose into one launch
            groups = len(getattr(self, "draft_kv_cache_group_ids", ()) or ()) if PREP_BATCH else 0
            _BATCH["groups"] = groups if 1 < groups <= 4 else 0
            _BATCH["pending"].clear()
            try:
                return base_propose(self, *args, **kwargs)
            finally:
                _flush_prepare()  # nothing pending in the normal flow (the last group launches)
                _BATCH["groups"] = 0

        cls.propose = propose
        mark_active("drafter", "DSparkSpeculator.propose(prepare batching)")


if ENABLED:
    register("vllm.v1.worker.gpu.spec_decode.dflash.speculator")(patch_dflash)
    register("vllm.v1.worker.gpu.spec_decode.dspark.speculator")(patch_speculator)
    register("vllm.models.deepseek_v41.nvidia.dspark")(patch_dspark_model)
    if MHC:
        register("vllm.models.deepseek_v41.nvidia.model")(_wrap_mhc)

# batched attention-metadata launches (target SWA groups, drafter-graph SWA updates): own flags, see ds41h/patch_meta.py
import ds41h.patch_meta  # noqa: E402,F401

# draft head + Markov loop on a side stream, overlapping the next step's eager input prep: own flag, see ds41h/patch_tail.py
import ds41h.patch_tail  # noqa: E402,F401
