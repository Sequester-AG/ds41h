"""The StepBatch-driven step program of the FPRv1 engine (one process drives the 4 GPUs).

    eng = Engine(EngineConfig())                 # weights of the REAL 40 layers on 4 GPUs, KV pools, extensions
    res = eng.step(batches, tokens=...)          # batches = [StepBatch per rank] (serve/interfaces.py) -> [StepResult]

The model as the checkpoint has it (text_config):
  * 40 layers, each with its OWN weights: attention (replicated, attn_half layout, fused W8A16 q_b), mHC, router gate,
    shared expert (TP4 slice), routed experts (TP4 slice, the fork's FP4 kernels moe_raw route2r / gather5 / GEMM1 / GEMM2).
  * layer types: compress ratio 0 (layers 0, 1: window only, FlashMLA without the extra cache), 2 (layers 2..19), 1
    (layers 20..39); compressed KV from kv_source_layer_ids 2 / 8 / 14 / 20 (layer L reads source max{s <= L}); top-512 from
    index_source_layer_ids 2 / 8 / 14 / 20 / 24 / 28 / 32 / 36 (layer L uses index layer max{i <= L}).
  * Engram at layers 1 and 14 (kernels/engram EngramPhys, injected before the attention mixes) [hook: cfg.engram].
  * embedding -> 4 hyper-connection copies at layer 0; after layer 39 the final hidden norm(hc_collapse(X_out, ffn_pre_39))
    and the aux rows (mean of the block input of layers 37 / 38 / 39) go to the head / sampler / drafter [hook: cfg.head].
  * indexer + compressor between the q / kv norms and FlashMLA of every index layer [hook: cfg.indexer].

Layout per rank (sequence parallel): rows = the rank's own sequences (StepBatch row order); attention
over all 64 heads on the owner's KV only (pools from serve/worker.py Pools: paged comp/idx pages, window pages, compressor
state, addressed by the StepBatch slots); per layer the f rows + routing rows are all-gathered (peer stores), every rank
runs its TP4 expert slice on all ranks' rows, and the partials are reduce-scattered back to the owners (comm.cu).
The pieces are kernels/sp_layer's (imported read-only) in its measured configuration of the 33.47 ms step:
mHC split boundaries (K1a, K1b-out on the main stream, K1b-mix on the side stream), prescale from K1b, router GEMM +
router_top_pack2, one AG push (ag_push2), shared expert on the side stream, route2r + gather5 + gemm1r / gemm2r, combine,
rs_push + rs_sum (--fuse-rs 0).

CUDA graphs: one graph per rank per row bucket R (rows per rank, padded). Everything that changes per step is a device
buffer of the bucket (positions, tokens, window / compressed slots, lengths, the valid-row mask); FlashMLA's planner runs
inside the graph (a fresh scheduler object per layer type at capture, as production). Padded rows: token 0, position 0,
own window slot = the scratch page, one window key (the scratch slot), no compressed keys, routing ids -1 (masked on the
device after router_top_pack2, so real rows see exactly the routing of an unpadded call).

Hooks (EngineConfig.indexer / engram / head; the placeholders are for tests):
  indexer "stub"  : top-512 = the most recent 512 compressed positions (serve.worker.TableBuilder rule), no compressor
                    writes;  "idx2f": kernels/indexer2f through engine/hooks_idx.py
  engram  "off"   : no Engram injection;                    "phys": kernels/engram EngramPhys
  head    "stub"  : final hidden computed, tokens from serve.worker.head_stub;  "hsd": kernels/head/hsd.py
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import sys
import time
from dataclasses import dataclass, field

import torch

PHYS = os.environ.get("PHYS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(PHYS, "serve"), os.path.join(PHYS, "kernels/sp_layer"), os.path.join(PHYS, "kernels/moe_stream"),
           os.path.join(PHYS, "kernels/attn_dense"), os.path.join(PHYS, "tools"),
           os.environ.get("DS41H_REF", os.path.dirname(PHYS))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("ATTN2B_PDL", "1")
# idx2f's extensions into engine/build/idx2f (entry points that build the Worker directly would otherwise compile
# idx2f into kernels/indexer2f/build)
os.environ.setdefault("IDX2F_BUILD", os.path.join(PHYS, "engine", "build", "idx2f"))
os.environ.setdefault("MOE_V4_BUILD", os.path.join(PHYS, "engine", "build", "moe_v4"))   # never into kernels/moe_v4

import weights as Wt  # noqa: E402  serve/weights.py (cache keys + builders)
from geometry import (INDEX_LAYERS, KV_SOURCE_LAYERS, SLIDING_WINDOW, WINDOW_PAGE_TOKENS, Geometry)  # noqa: E402
from interfaces import StepResult, copy_runs  # noqa: E402

from . import attn as EA  # noqa: E402

N_LAYERS, W, D, HC, TOPK, NE = 40, 4, 5120, 4, 6, 384
RATIO = Wt.RATIO
DEVTOK_V2 = os.environ.get("ENG_DEVTOK", "1") == "1"   # 1: device_tokens with one pinned copy (_device_tokens_v2,
                                                         # same result); 0: the per-sequence path
BISECT = dict(per_layer_plans=True, clear_tail_padding=True, inject_join=True, engram_args="tables", free_cache=True,
              engram_stream=False, evict_empty_cache=True)
# ^ runtime switches of a set of batched-decode fixes (tests bisect batched-vs-alone decode with them)
GRAPH_MAX_CTX = 262_144              # decode graphs' indexer bounds (hooks_idx.MAX_CTX); longer contexts run eagerly
LONG_LEVEL_CTX = (262_144, 524_288, 655_360)   # hooks_idx.LEVEL_CTX: long-row decode-graph variants
LVL_TAG = 1 << 20                    # a long variant's decode bucket key = R + lvl * LVL_TAG (own state, indexer bounds, graphs)
ATTN_PF = os.environ.get("ENG_ATTN_PF", os.environ.get("ATTN2B_PF", "0")) == "1"   # mixed steps, bucket A (layers 0..20):
                              # prefill rows through production's prefill attention (kernels/attn_dense/attn_prefill),
                              # decode rows on the decode kernel with their own FlashMLA plan
PF_ACT8_DENSE_LOAD = os.environ.get("ENG_PF_ACT8_DENSE", "0") == "1"   # TEST ARM (LOSSY, default 0): the
PF_ACT8_DENSE = PF_ACT8_DENSE_LOAD   # mixed steps' prefill rows of layers 0..20 through production's W8A8 dense GEMMs
                                     # (q_a | wkv, q_b [needs ENG_QB16_MIXED], wo_b); weights (+1.87 GB per GPU) loaded only
                                     # with ENG_PF_ACT8_DENSE=1 at start; runtime flag per arm
L20_TAIL = os.environ.get("ENG_L20_TAIL", "0") == "1"   # mixed steps: layer 20 (the last kv source) runs its mHC boundary,
                         # q / kv projections, indexer + compressor and window insert on ALL rows (bucket A), but its
                         # attention core, wo, ffn boundary and MoE on the TAIL rows only (bucket T, which alone consumes
                         # layer 20's output); production runs only fused_wqa_wkv on every prefill row.
                         # The tail rows keep layer 20's FULL window (si / sl gathered from bucket A).
MHCF_LOAD = os.environ.get("ENG_MHCF", "0") == "1"   # kernels/mhc_fuse: the fused mHC boundary
MHCF = MHCF_LOAD                 # (fa + fb_out + fb_mix, bit-identical to k1a + k1b_out + k1b_mix; -3.3 ms per benchmark step);
                                 # weights prepared at load (+3.9 MB / layer) only with ENG_MHCF=1 at start; runtime flag
QB16_LOAD = os.environ.get("ENG_QB16_MIXED", "0") == "1"   # keep the bf16 wq_b of layers 0..20 (+84 MB each)
LAUNCH_SPLIT = os.environ.get("ENG_LAUNCH_SPLIT", "0") == "1"   # split decode graph L in two pieces
LAUNCH_SPLIT_AT = int(os.environ.get("ENG_LAUNCH_SPLIT_AT", "15"))   # (embed + layers < AT | layers >= AT + final + join);
                                            # replay launches piece 0 on all 4 ranks, then piece 1: the ranks start ~0.5 ms
                                            # apart instead of ~1.4 ms (one launch of the 40-layer graph ~0.47 ms CPU), the
                                            # rest of the launches overlap the GPU. Same kernels, same order (+ a side-mix
                                            # join at the cut); AT > the Engram layers (their events are recorded at embed)
RESET_KEEP_GRAPHS = False      # tests only (probing a slow cudaGraphLaunch after reset_state):
RESET_EMPTY_CACHE = True       # keep the old arm's graphs alive / skip reset_state's empty_cache (per arm)
G6 = os.environ.get("ENG_G6", "0") == "1"   # kernels/moe_v4/g6.cu: gather6, a bit-identical drop-in for
                                            # prepr5 in DECODE buckets (image bytes incl. padding rows + ec.inv; -5.5 us per
                                            # layer at T 1,024); built lazily into engine/build/eng_moe_g6 (prebuilt on the CPU)
QB16_DECODE = os.environ.get("ENG_QB16_DECODE", "0") == "1"   # decode buckets with > 128 rows (192..320) on
                                # the bf16 q_b too (W8A16 streams the weights twice above 128 rows; bitwise; -0.45 ms per
                                # decode step at 224 / 240 rows); loads the bf16 wq_b of ALL 40 layers (3.4 GB per GPU);
                                # graphs capture the branch, so the flag is fixed per capture (tests: per arm via reset_state)
QB16_MIXED = QB16_LOAD          # mixed buckets with >= 256 rows: q_b on cuBLAS bf16 + rope_q_insert instead of the fused
                                # W8A16 q_b (bitwise: q and the window-KV records identical; 820 -> 555 us per layer at
                                # 4,288 rows); runtime-switchable (needs the copies loaded: ENG_QB16_MIXED=1 at start)
LONG_GRAPHS = os.environ.get("ENG_LONG_GRAPHS", "1") == "1"   # decode steps with rows past 262,144 positions: the bucket's
                                                              # long variant (graphed) instead of the eager mixed path.
                                                              # Default on (long-context check: eager and graphed arms
                                                              # identical vs production); with IDX2F_CANON_STOCK=1 (set
                                                              # below)
os.environ.setdefault("IDX2F_CANON_STOCK", "1")   # idx2f: the stock persistent top-k's lists in canonical order (ascending,
                                                  # -1 last; read when each OwnedIndexer is built): eager == graph == eager
FREE_CACHE_BYTES = int(float(os.environ.get("ENG_FREE_CACHE_GB", "4")) * 1e9)   # mixed steps: release the allocator's
                                     # idle cache when more than this is cached but unused on a device
V4_MIXED = os.environ.get("ENG_V4_MIXED", "") == "1"   # debug: moe_v4 in mixed buckets too (see EngineConfig.moe_v4)
DUMP_MOE = os.environ.get("ENG_DUMP_MOE", "")         # debug: dump the first mixed step's fork-MoE calls (_dump_moe)
PROF_PIECES = os.environ.get("ENG_PROF_PIECES", "") == "1"   # profiler ranges per piece (host time / launches per piece)
TAIL_BOUND = os.environ.get("ENG_TAIL_BOUND", "1") == "1"   # mixed steps: the tail's layer-21..39 window from the tail start
                                                            # (production's semantics; default on: served == offline
                                                            # bitwise)
POISON = os.environ.get("ENG_POISON", "") == "1"      # debug: fill uninitialized (torch.empty) buffers with NaN / huge ints
PM_FUSED = os.environ.get("ENG_PM_FUSED", "1") == "1"  # production-MoE layers as ONE host call (kernels/moe_prefill/pm_fused,
                                                       # compact rows: bit-identical; default on: 4-GPU A/B bitwise 10 / 10,
                                                       # 88.0 -> 86.9 ms); 0 = the index_select / moe_forward_w4a16 /
                                                       # zero_ / index_copy_ path (~13 host ops per layer per rank)
IMAGE_TOKEN_ID = 129264          # images (FPRv1.1): every image-span position carries this id (vLLM mm_preprocess.py)
IMAGE_ROUTE_IDS = (129264, 129268)  # ids routed with the router's bias_vl (vLLM's image_sentinel_lo .. lo + 4)
MM_ON = os.environ.get("PHYS_MM", "0") == "1"       # images enabled (adapter's switch); off = the text engine, unchanged
MM_CHECK = os.environ.get("ENG_MM_CHECK", "1") == "1"   # image steps: ctx.mm rows are 129264 rows (one small D2H)
PREV_STEP_CHECK = os.environ.get("ENG_PREV_STEP_CHECK", "1") == "1"   # CORRECTNESS (default on):
                                           # prev_seq entries carry the engine step serial that wrote them; device input ids
                                           # (hsd.next_ids rows, the PREVIOUS batch's order) and drafts_valid are used only if
                                           # the sequence ran in the immediately previous engine step. A sequence that missed
                                           # a step (sync rank move: skip_until_step; a skipped allocation) otherwise read the
                                           # row another sequence got in the missed step (found by reading the code):
                                           # one wrong input token per resumed sequence, invisible to teacher forcing. Now it
                                           # resumes on the host's ids (t0) with drafts invalid. 0 = the old behaviour (A/B).
PM_NT = os.environ.get("ENG_PM_NT", "0")   # (default 0) pm_fused calls on kernels/moe_prefill/pmv.FusedV (production's
                                           # moe_prefill.cu kernels + gated patches, a frozen source snapshot): remainder-kernel
                                           # token rows by ONE tile-sized TMA box, and cheaper mbarrier waits. "1" = safe
                                           # (counted spin: gives up after 2^30 polls + device flag instead of a trap;
                                           # RECOMMENDED), "fast" = plain spin (no device watchdog). BIT-IDENTICAL
                                           # (layer + layer8 tests on benchmark / half / CP 4K / 8K / 16K buckets); per layer
                                           # call "1" / "fast": CP 8K -5.5 / -5.5 %, CP 16K -3.4 / -3.2 %, CP 4K -13.3 / -15.0 %,
                                           # benchmark -4.3 / -4.5 %, layer8 (8-bit arm) benchmark -7.1 / -8.3 %
PF_ACT8 = os.environ.get("ENG_PF_ACT8", "0") == "1"   # TEST ARM (opt-in, LOSSY, default 0): pm_fused calls (mixed
                                                       # steps, vcount > SMALL_T) run the PREFILL rows' routed experts on production's
                                                       # W4A8 kernels (e4m3 activations, one dynamic scale per token / routed row,
                                                       # deterministic 0x75); decode rows stay W4A16 (separate sub-call), shared
                                                       # expert 16-bit (kernels/moe_prefill/pm_fused layer8)
PF_ACT8_DEC = int(os.environ.get("ENG_PF_ACT8_DEC", "23"), 0)  # PF_ACT8's decode-row sub-call kernel: a
                                                       # production small16k GEMM variant (23 = forward 0x3500) instead of 0xB9, and the
                                                       # shared expert on a side stream (kernels/moe_prefill/pm2): BIT-IDENTICAL outputs
                                                       # (tested), -220-270 us per layer at 640 decode rows; 0 = pm_fused.layer8


def poison_(*ts) -> None:
    """ENG_POISON=1: NaN for floats, 0x3fffffff for ints, so a read of a never-written element is deterministic (NaN
    output or an out-of-range index) instead of whatever the reused memory held."""
    if not POISON:
        return
    for t in ts:
        if t is None or not isinstance(t, torch.Tensor) or not t.is_cuda or t.numel() == 0:
            continue
        if t.is_floating_point():
            t.fill_(float("nan"))
        elif t.dtype in (torch.int32, torch.int64, torch.int16):
            t.fill_(0x3FFFFFFF if t.dtype != torch.int16 else 0x3FFF)


DBG_SYNC = os.environ.get("ENG_SYNC_DEBUG", "") == "1"   # wait + check every device after each piece (fault finder)
POISON_POOLS = os.environ.get("ENG_POISON_POOLS", "") == "1"   # debug: bind_pools fills the paged + window pools (not the
                                                               # scratch page) with 0xFF (fp8 / bf16 NaN): never-written KV
                                                               # reads turn into NaN (with ENG_NANCHECK / ENG_CP_AUDIT)
NANCHECK = os.environ.get("ENG_NANCHECK", "") == "1"   # debug: eager steps check every rank's block input X after each
                                                       # layer; the first non-finite (step, layer, rank, rows) is logged
MHC_MAX = 128                       # mhc.cu: at most 128 rows per call -> larger buckets run in 128-row chunks
AUX_LAYERS = (37, 38, 39)           # DSpark target layers (aux = bf16 mean over hc of the block input)
ENGRAM_LAYERS = (1, 14)
MODEL = Wt.MODEL


def kv_source_of(L: int):
    return None if L < KV_SOURCE_LAYERS[0] else max(s for s in KV_SOURCE_LAYERS if s <= L)


def index_source_of(L: int):
    return None if L < INDEX_LAYERS[0] else max(s for s in INDEX_LAYERS if s <= L)


@dataclass
class EngineConfig:
    model: str = MODEL
    devices: tuple = (0, 1, 2, 3)
    layers: tuple = tuple(range(N_LAYERS))   # a contiguous run of layers; tests use e.g. (21,) with injected inputs
    buckets: tuple = (32, 64, 112, 120, 128, 192, 224, 240, 256, 320)   # rows per rank (= HSD's row buckets: 4 x its
                                                          # sequence buckets; 120 / 224 / 240: 112 / 224 seqs stop
                                                          # padding to 128 / 256, -0.6 ms GPU each);
                                                          # a decode step uses the smallest >= n
    w8a16: str = "q_b"                       # attn_half's measured default (fused W8A16 q_b + RoPE + insert)
    insert: bool = True                      # the attention writes the rows' own window KV (a real step)
    ncta: int = 16                           # AG / RS CTAs per destination (sp_layer default)
    graphs: bool = True                      # capture one graph per rank and bucket on first use, replay after
    reserve_bytes: int = 6_000_000_000       # left free after pools (activations, workspaces, graphs)
    max_kv_bytes: int = 0                    # > 0: cap the KV pools (tests)
    window_pool_bytes: int = 1_400_000_000
    indexer: str = "stub"                    # "stub" (placeholder) | "idx2f" (kernels/indexer2f OwnedIndexer, engine/hooks_idx.py)
    indexer_mode: str = "fast"               # idx2f mode: "stock" | "fast" | "hybrid"
    engram: str = "off"                      # "off" (placeholder) | "phys" (kernels/engram EngramPhys)
    head: str = "stub"                       # "stub" (placeholder) | "hsd" (kernels/head/hsd.py)
    load_threads: int = 16
    engine_reserve_bytes: int = 10_000_000_000   # free_bytes() leaves this for bucket buffers, graph pools, mixed steps
    max_seqs: int = 80                       # sequences per rank (HSD state slots; geometry.max_seqs_per_rank)
    max_mixed_rows: int = 4096 + 320         # rows per rank of a mixed step (geometry.max_prefill_tokens_per_rank + decode)
    mixed_round: int = 64                    # mixed steps run eagerly on buckets rounded up to this (64: 1,280-row step
                                             # 71.8 -> 67.4 ms without 320 -> 512 padding; was 256)
    draft_head: str = "e4m3"                 # HSD draft head (e4m3 W8A8 default)
    mhc_chunk: int = 1 << 20                 # rows per mHC kernel call: one call per boundary (bitwise on 14/14
                                             # mixed steps, mixed 1,280 rows 81 -> 72 ms) (K1a v2 takes any R:
                                             # 1 << 20 = one call per boundary, bit-identical per sp_layer, -0.5..0.8 ms at
                                             # T 640..1,024 and ~2/3 of a 4K-row mixed step's launches)
    mixed_graphs: bool = False               # piecewise CUDA graphs for mixed steps (fixed-shape pieces captured per
                                             # mixed bucket; indexer, MoE, Engram-inject boundaries and embed stay eager)
    moe_v4: bool = True                      # kernels/moe_v4's GEMMs (+ its route2r pair lists) in DECODE buckets (<= 1,365
                                             # rows, every decode bucket): bit-identical there; NOT in mixed tail buckets
                                             # (the same A/B differed from the first mixed step), so off there
    hsd_fold: bool = False                   # HSD drafter: MoE RS wait + sum folded into the next mHC boundary (bitwise, 4-GPU
                                             # A/B at 256 seqs; above 128 own draft rows it needs hsd_mhc2)
    hsd_mhc2: bool = False                   # HSD drafter: sp_layer's any-rows mhc.cu (no 128-row chunks), -120 us at 256 seqs
    hsd_moe: str = "route2"                  # HSD drafter MoE: "route2" | "v4" (kernels/moe_v4's drop-in, -51 us at 256 seqs;
                                             # A/B on decode steps only so far)
    hsd_prep: str = "1"                      # HSD host prepare: "1" vectorized, one pinned H2D per rank (~0.2 ms at 112
                                             # seqs); "0" the per-element version (~6 ms at 128 seqs)
    upload_v2: bool = True                   # (bitwise on decode steps and 14/14 mixed steps) decode-step
                                             # cut 1: the step inputs (pos / tok / own / si / sl / valid / samp) in
                                             # ONE pinned host blob per bucket (double-buffered) -> one H2D copy per rank
                                             # into one device blob (the bucket's input tensors are its views); was 7
                                             # pin_memory() + copy pairs per rank
    split_hsd: bool = True                   # (bitwise on 10/10 decode steps, accept-all) decode-step cut 2: per bucket two graphs, the 40 layers (+ final) and HSD;
                                             # HSD's host prepare (+ its H2D, ordered on the step stream) runs while the
                                             # layers' graph executes
    moe_v4_mixed: bool = True                # (bitwise on 12/12 steps vs the raw path, 1,280-row mixed step 66.7
                                             # -> 64.6 ms) moe_v4 also in mixed buckets of <= 1,365 rows (all ranks)
    mixed_plans: str = "source"              # mixed steps' FlashMLA plans: "layer" (one per layer) | "source" (one per layer
                                             # type and index source: equal inputs by construction;
                                             # bitwise 11/11, deep-context 1,280-row mixed step 68.2 -> 65.6 ms)
    router_pack3: bool = True                # (bitwise on 13/13 mixed + 10/10 decode steps) moe_raw router_top_pack3
                                             # (padding rows masked in-kernel, -2 launches per layer)
    cp: bool = True                          # (bitwise vs single-owner through all 40 layers at
                                             # 32K / 196K, TTFT 196K 8.88 -> 4.57 s; acts only when serve's cp_prefill emits
                                             # CP roles) context-parallel prefill (engine/cp.py): roles ->
                                             # the attention piece of layers 0..20 in 3 phases with X1a / X2 exchanges
    d13: bool = False                        # D13 async decode (engine/d13.py): Engine.dispatch / step_compact with positions
                                             # resolved on the device (kernels/rowexp); needs upload_v2 + the full pieces
    gate_head: bool = True                   # forced / score sequences: engine/gatehead.py
    prefill_moe_switch: int = 2560           # mixed steps: the fork's decode MoE up to this many rows (all ranks), production's
                                             # above (measured: 2K 0.55 vs 0.69 us / token / layer, 4K equal, 8K+ prod)
    max_mixed_buckets: int = 8               # LRU of eager mixed-step bucket states (A / T roles x sizes) ...
    mixed_bucket_bytes: int = 5_000_000_000  # ... and at most this many bytes of them per rank (a 4,608-row bucket ~2.7 GB)


def _env_kw() -> dict:
    """EngineConfig overrides from $PHYS_ENGINE, e.g. "indexer=idx2f,engram=phys,head=hsd" (serve's worker only passes
    devices / model_dir / max_rows)."""
    out = {}
    for kv in os.environ.get("PHYS_ENGINE", "").split(","):
        if "=" in kv:
            k, v = (x.strip() for x in kv.split("=", 1))
            f = EngineConfig.__dataclass_fields__.get(k)
            assert f is not None, f"PHYS_ENGINE: unknown field {k}"
            out[k] = (v.lower() in ("1", "true", "yes")) if f.type in ("bool", bool) else \
                (int(v) if f.type in ("int", int) else
                 (tuple(int(x) for x in v.split(":")) if f.type in ("tuple", tuple) else v))
    return out


_IN_FIELDS = (("pos", "int64", 1), ("tok", "int64", 1), ("own", "int64", 1), ("samp", "int64", 1),
              ("si", "int32", SLIDING_WINDOW), ("sl", "int32", 1), ("valid", "bool", 1))


def _in_layout(R: int):
    """Byte offsets of the step inputs in one blob (8-byte aligned) and the blob size."""
    import numpy as np
    off, lay = 0, {}
    for nm, dt, cols in _IN_FIELDS:
        lay[nm] = (off, dt, cols)
        off += -(-(R * cols * np.dtype(dt).itemsize) // 8) * 8
    return lay, off


def _in_views(blob, R: int, numpy: bool = False) -> dict:
    """The step inputs as views of a uint8 blob (torch: the bucket's device tensors, si [R, 1, W], valid [R, 1];
    numpy=True: numpy views of a pinned host blob, flat per field)."""
    import numpy as np
    lay, _ = _in_layout(R)
    out = {}
    if numpy:
        b = blob.numpy()
        for nm, (o, dt, cols) in lay.items():
            out[nm] = b[o: o + R * cols * np.dtype(dt).itemsize].view(dt)
        return out
    tdt = dict(int64=torch.int64, int32=torch.int32, bool=torch.bool)
    for nm, (o, dt, cols) in lay.items():
        x = blob[o: o + R * cols * np.dtype(dt).itemsize].view(tdt[dt])
        out[nm] = x.view(R, 1, SLIDING_WINDOW) if nm == "si" else (x.view(R, 1) if nm == "valid" else x)
    return out


def _torch_peer_maps(devs, log=print) -> None:
    """Peer stores (comm.cu AG / RS, HSD's peer mode) write into OTHER devices' torch allocations. With
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (as the served launcher sets it) those are VMM mappings that only
    the owning device may access, unless torch's allocator was told about the peer: comm.cu's own
    cudaDeviceEnablePeerAccess does not tell it (symptom: the first AG faulted, the next cuBLAS call
    reported CUBLAS_STATUS_EXECUTION_FAILED). A torch copy between two devices goes through at::cuda::get_p2p_access,
    which registers the peer with the caching allocator (existing and future expandable segments are mapped for it):
    one tiny copy per ordered device pair. Harmless with the default allocator. ENG_PEER_MAP=0 skips it (tests)."""
    if os.environ.get("ENG_PEER_MAP", "1") != "1":
        log("engine: torch peer mappings SKIPPED (ENG_PEER_MAP=0)")
        return
    for i, di in enumerate(devs):
        for j, dj in enumerate(devs):
            if i != j:
                torch.zeros(1, device=dj).to(di)
    for d in devs:
        torch.cuda.synchronize(d)
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" in conf.replace(" ", ""):
        log(f"engine: PYTORCH_CUDA_ALLOC_CONF={conf}: peers registered with torch's allocator for {len(devs)} devices")


def _cos_sin(ratio: int, dev):
    """Stock vLLM RoPE table of a compress ratio, computed ON THE GPU as vLLM does (CPU bits differ). Ratio 0 (layers 0, 1):
    plain RoPE theta 10000 = deepseek_yarn with factor 1 and original_max = max_position_embeddings (rope.py)."""
    import bench2b as B
    return B.stock_cos_sin(ratio, dev, (1 << 20) if ratio == 0 else 0)


class Engine:
    wants_stub_indexer = False      # class default; instances: True with indexer="stub" (serve D9 / D12)
    supports_images = MM_ON         # images (FPRv1.1, PHYS_MM=1): the engine applies StepContext.mm and bias_vl routing

    def __init__(self, devices=(0, 1, 2, 3), model_dir: str | None = None, max_rows: int = 320, cfg=None, log=print, **kw):
        """serve contract D1: Engine(devices, model_dir, max_rows, **kw) loads the weights (kw: EngineConfig fields, e.g.
        indexer="idx2f", engram="phys", head="hsd"); then free_bytes(), bind_pools(pools), step(batches, tables, ctx)."""
        import bench
        import test_comm
        import test_mhc
        from ds41h.ref import layer as RL
        from ds41h.ref.weights import default_reader
        from torch.utils.cpp_extension import load as _load

        if isinstance(devices, EngineConfig):
            cfg, devices = devices, None
        if cfg is None:
            kw2 = _env_kw()
            kw2.update(kw)
            if devices is not None:
                kw2["devices"] = tuple(d if isinstance(d, int) else torch.device(d).index for d in devices)
            if model_dir:
                kw2["model"] = model_dir
            cfg = EngineConfig(**kw2)
        self.cfg = cfg
        self.max_rows = max_rows
        self.wants_stub_indexer = cfg.indexer == "stub"
        # served (adapter/stage4_worker.py): with HSD the decode rows' input ids (last token + drafts) come from the
        # device (D6), so the host must leave the draft rows at -1; HSD samples with temperature + seed only
        self.device_drafts = cfg.head == "hsd"
        self.sampling_caps = set()
        # serve's Geometry.max_tail_rows_per_rank reads this (decode rows + chunk tails per rank per step): the T bucket
        # and HSD's context rows are sized for a whole mixed bucket
        self.max_tail_rows = -(-cfg.max_mixed_rows // cfg.mixed_round) * cfg.mixed_round
        self.log = log
        self.layers = list(cfg.layers)
        assert self.layers == list(range(self.layers[0], self.layers[-1] + 1)), "layers must be contiguous"
        self.devs = [torch.device("cuda", d) for d in cfg.devices]
        assert len(self.devs) == W
        # Engram's host tables (2 x 97 GiB, production's /dev/shm copy): the page-table populate (~2.5 s, kernel work, no
        # CUDA) runs in a background thread behind the extensions + weight load; Hooks registers them (~10 s on 4 GPUs)
        self.eg_reg = None
        if cfg.engram == "phys" and self.layers[0] <= 1 and self.layers[-1] >= 14:
            sys.path.insert(0, os.path.join(PHYS, "kernels/engram"))
            import engram as EG
            self.eg_reg = EG.open_tables_async()
        t0 = time.time()
        EA.QB16D = QB16_DECODE
        self.mhcf = None
        if MHCF_LOAD:
            p_ = os.path.join(PHYS, "kernels/mhc_fuse")
            if p_ not in sys.path:
                sys.path.insert(0, p_)
            import mhcf
            self.mhcf = mhcf.build(build_directory=os.path.join(PHYS, "engine/build/eng_mhcf"))
        self.hext, self.cext = test_mhc.ext(), test_comm.ext()
        self.cext.enable_peers(W)
        _torch_peer_maps(self.devs, self.log)
        self.r2 = _load(name="route2_ext", sources=[os.path.join(PHYS, "kernels/moe_stream/route2.cu")], verbose=False,
                        extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                                           "--expt-relaxed-constexpr"])
        self.mext = []
        for d in self.devs:
            torch.cuda.set_device(d)
            self.mext.append(bench.build_extr())
        import attn_half as AH
        import w8a16 as W8
        W8.build()
        AH._ext()
        rd = default_reader(cfg.model)
        self.reader = rd
        self.mcfg = RL.ModelConfig.from_hf(rd.config)
        self.norm_w_cpu = rd.get("norm.weight").float()
        log(f"engine: extensions in {time.time() - t0:.1f} s")
        # ---- weights (wcache entries prebuilt on the CPU: engine/prebuild.py), one loader thread per rank
        t1 = time.time()
        self.lw = [None] * W
        self.glob = [None] * W

        # (rank, layer) items in parallel: torch.load from the page cache is CPU-bound per entry (~1-2 GB/s per thread)
        css = []
        for r in range(W):
            with torch.cuda.device(self.devs[r]):
                css.append({c: _cos_sin(c, self.devs[r]) for c in sorted({RATIO[L] for L in self.layers})})
        self.lw = [dict() for _ in range(W)]

        def load_item(rl):
            r, L = rl
            dev = self.devs[r]
            torch.cuda.set_device(dev)      # device-local ops in finish_* (prep_fn, the W8A16 packing) run on dev
            with torch.cuda.device(dev):
                if L is None:
                    emb = Wt.load("embed_v1", Wt.build_embed, dev)
                    self.glob[r] = dict(emb=emb, norm_w=self.norm_w_cpu.to(dev), cos_sin=css[r])
                    return
                ent = Wt.load(f"attn_L{L}_v1", lambda L=L: Wt.build_attn(L), dev)
                qb16 = ent.get("wq_b") if ((QB16_LOAD and L <= 20) or QB16_DECODE) else None   # (finish_attn drops it)
                AW = Wt.finish_attn(ent, dev, css[r][RATIO[L]], cfg.w8a16)
                if qb16 is not None:
                    AW["wq_b16"] = qb16.to(dev)
                AW["L"] = L
                if PF_ACT8_DENSE_LOAD and L <= 20:
                    A8 = EA._a8()
                    AW["A8"] = A8.prepare(Wt.load(f"act8_L{L}_raw_v1", lambda L=L: A8.build_raw(L), dev))
                assert AW["ratio"] == RATIO[L], (L, AW["ratio"])
                ex = Wt.load(f"L{L}_r{r}_raw_experts_v1", lambda L=L: Wt.build_raw_experts(L, r), dev)
                raw_m = Wt.load(f"mhc_L{L}_v1", lambda L=L: Wt.build_mhc(L), dev)
                m = Wt.finish_mhc(raw_m, self.hext)
                if self.mhcf is not None:
                    m["fa_f"], m["ff_f"] = self.mhcf.prep_f(raw_m["fa_raw"]), self.mhcf.prep_f(raw_m["ff_raw"])
                dn = Wt.load(f"L{L}_r{r}_dense_v1", lambda L=L: Wt.build_dense(L, r), dev)
                self.lw[r][L] = dict(m, **dn, AW=AW, w13=ex[0], s13=ex[1], w2=ex[2], s2=ex[3])
                torch.cuda.synchronize(dev)

        items = [(r, None) for r in range(W)] + [(r, L) for L in self.layers for r in range(W)]
        with cf.ThreadPoolExecutor(max(1, cfg.load_threads)) as ex_:
            list(ex_.map(load_item, items))
        log(f"engine: weights of layers {self.layers[0]}..{self.layers[-1]} on {W} GPUs in {time.time() - t1:.0f} s, "
            f"allocated {[round(torch.cuda.memory_allocated(d) / 1e9, 1) for d in self.devs]} GB")
        # ---- KV pools: owned by the worker (serve D2), bound with bind_pools(); tests: make_pools()
        import worker as WK
        self.WK = WK
        self.pools = None
        self.devices = self.devs
        self.tb = WK.TableBuilder(stub_indexer=self.wants_stub_indexer)        # tests without a worker
        self.tb_tail = WK.TableBuilder(stub_indexer=self.wants_stub_indexer)   # mixed steps: the tail rows' tables
        self.streams = [torch.cuda.Stream(device=d) for d in self.devs]
        self.side = [torch.cuda.Stream(device=d) for d in self.devs]
        self.estream = [torch.cuda.Stream(device=d) for d in self.devs]     # Engram's token-only side work
        for d in self.devs:
            torch.cuda.synchronize(d)
        for st_ in self.streams:     # the engine's streams are the devices' current streams: every piece (HSD, Engram,
            torch.cuda.set_stream(st_)  # indexer) that issues on "the current stream" is ordered with the step program
        self.prev_seq = {}           # rid -> (rank, seq index, step serial) of the last step with a head (HSD next_ids rows)
        self.step_serial = 0         # engine steps with sequences so far (step / D13 dispatch / D13 first step): see PREV_STEP_CHECK
        self.bk = {}            # R -> [per-rank bucket state]
        self.bk_bytes = {}      # mixed bucket key -> device bytes per rank
        self._busy = set()      # mixed bucket keys of the current step
        self.graphs = {}        # R -> [CUDAGraph per rank]
        self.mgraphs = {}       # mixed bucket key -> (units, [per unit: [CUDAGraph per rank] or None], kept sched objects)
        self._mm = None         # images: this step's StepContext.mm ([per rank (idx int32 [M], emb bf16 [M, D])] or None)
        self._bvl = {}          # (rank, layer) -> the router's bias_vl [384] fp32 (image-token routing)
        self._retired = []      # replaced / popped CUDAGraphs, kept alive (else Xid 31 faults: torch 2.13 frees a
                                # stream's cuBLAS workspace when ANY graph that reused it is destroyed, while the other
                                # graphs captured on that rank stream still write it); dropped only in reset_state
        self.gkeep = {}         # decode graph key -> the FlashMLA plan objects it was captured with (kept alive: a later
                                # capture on the same bucket state, e.g. D13's (R, "LA") next to the sync path's (R, "L"),
                                # replaces caches[L]["sched"] via _new_scheds; the older graph's in-graph planner still
                                # writes the old objects' metadata -> freed, then unmapped by an empty_cache: Xid 31 WRITE)
        self.hsd_in_graph = {}  # (R, with_hsd) -> HSD bucket key captured inside the layer graph (None: none inside)
        self._last_gkey = {}    # R -> the graph variant of the bucket's last step (replay(R) in tests / benchmarks)
        self.gpool = []         # one graph memory pool per device, shared by the buckets' graphs (replayed one at a time)
        for d in self.devs:
            with torch.cuda.device(d):
                self.gpool.append(torch.cuda.graph_pool_handle())
        self.hist = {}          # rid -> host token ids (prompt + output) for Engram's prefill lookback (ctx.new + results)
        self.hooks = Hooks(self)
        self.hsd = self.hooks.hsd          # (the adapter also looks for backend.hsd)
        self.stats = dict(steps=0, graph_steps=0, eager_steps=0)
        for d in self.devs:
            torch.cuda.synchronize(d)

    # ------------------------------------------------------------------------------------------------ contract (serve D1)
    def free_bytes(self) -> list:
        """Free bytes per rank after weights, minus what buckets / graph pools / mixed steps will take."""
        return [int(torch.cuda.mem_get_info(d)[0]) - self.cfg.engine_reserve_bytes for d in self.devs]

    def bind_pools(self, pools: list) -> None:
        assert len(pools) == W and self.pools is None, "bind_pools once, 4 ranks"
        hs = getattr(self.hooks, "hsd_slots", None)       # slot-indexed hook state must cover every serve state slot
        assert hs is None or all(p.n_state <= hs for p in pools), (
            f"state pool {[p.n_state for p in pools]} slots > HSD's {hs} (Geometry.state_pool_slots)")
        for e in (self.hooks.engram or []):
            assert all(p.n_state <= e.ring.shape[0] for p in pools), "state pool > Engram ring slots"
        self.pools = pools
        for p in pools:   # scratch window page (padded rows) holds zeros: finite padded-row math
            if POISON_POOLS:
                p.paged.fill_(255)
                p.window[: p.scratch_window_page * self.WK.WINDOW_PAGE_BYTES].fill_(255)
            p.window[p.scratch_window_page * self.WK.WINDOW_PAGE_BYTES:].zero_()
        if POISON_POOLS:
            self.log("engine: ENG_POISON_POOLS: paged + window pools filled with 0xFF")
        self.hooks.attach(pools)
        self.sync()

    def make_pools(self, geom: Geometry | None = None, max_kv_bytes: int = 0, reserve_bytes: int = 4_000_000_000):
        """Tests without a worker: size + allocate + bind the pools as serve/worker.py's Worker does. Returns geom."""
        import dataclasses
        WK = self.WK
        g0 = geom or Geometry(window_pool_bytes=self.cfg.window_pool_bytes)
        g0 = dataclasses.replace(g0, max_rows_per_rank=self.max_rows) if hasattr(g0, "max_rows_per_rank") else g0
        wc = WK.WorkerConfig(model=self.cfg.model, devices=self.cfg.devices, reserve_bytes=reserve_bytes,
                             max_kv_bytes=max_kv_bytes or self.cfg.max_kv_bytes)
        self.geom = WK.size_pools(min(self.free_bytes()), wc, g0, 0)
        self.bind_pools([WK.Pools(self.geom, d) for d in self.devs])
        pp = self.geom.pools()[0]
        self.log(f"engine: KV pools {self.geom.kv_bytes_per_rank / 1e9:.1f} GB per rank = {pp.num_pages:,} pages "
                 f"({pp.num_pages * 64 / 1e6:.1f}M tokens)")
        return self.geom

    # ------------------------------------------------------------------------------------------------ buckets
    def bucket_of(self, n: int) -> int:
        for b in self.cfg.buckets:
            if n <= b:
                return b
        raise ValueError(f"{n} rows per rank > largest bucket {self.cfg.buckets[-1]}")

    def bucket(self, R: int, moe: str = "raw", role: str = "") -> list:
        """Per-rank state of a row bucket. moe 'raw': the fork's decode MoE (moe_raw route2r / gather5 / GEMMs, graphs);
        'prod': production's routed MoE (ds41h.moe.moe_forward_w4a16 from the ds41h package) for steps with prefill rows."""
        lvl = 0
        if moe == "raw" and R >= LVL_TAG:            # a long-row variant of decode bucket R % LVL_TAG
            lvl, R = R // LVL_TAG, R % LVL_TAG
        key = (R + lvl * LVL_TAG) if moe == "raw" else (R, moe, role)   # mixed: separate A (all rows) / T (tail rows) states
        if moe != "raw":
            self._busy.add(key)                       # in use by this step: never evicted by the step's next bucket
        if key in self.bk:
            if moe != "raw":
                self.bk[key] = self.bk.pop(key)       # LRU order
            return self.bk[key]
        if moe != "raw":
            mixed = [k for k in self.bk if isinstance(k, tuple) and k not in self._busy]
            per_row = max([self.bk_bytes[k] / k[0] for k in mixed if k in self.bk_bytes] or [650_000])
            need = per_row * R
            evicted = False
            held = lambda: sum(self.bk_bytes.get(k, 0) for k in self.bk if isinstance(k, tuple))  # noqa: E731
            while mixed and (len([k for k in self.bk if isinstance(k, tuple)]) >= self.cfg.max_mixed_buckets or
                             held() + need > self.cfg.mixed_bucket_bytes):
                old = mixed.pop(0)
                mg = self.mgraphs.pop(old, None)      # its piecewise graphs reference the evicted state's buffers
                if mg is not None:                    # retired, not destroyed (captured on the rank streams: see _retired)
                    self._retired.append(mg)
                del self.bk[old]
                self.bk_bytes.pop(old, None)
                evicted = True
            if evicted:
                for d in self.devs:
                    torch.cuda.synchronize(d)
                if BISECT["evict_empty_cache"]:
                    torch.cuda.empty_cache()
                self.stats["mixed_bucket_evictions"] = self.stats.get("mixed_bucket_evictions", 0) + 1
        m0 = torch.cuda.memory_allocated(self.devs[0])
        st = [self._alloc(r, R, moe, lvl) for r in range(W)]
        for d in self.devs:
            torch.cuda.synchronize(d)   # flags / epochs zeroed and visible before any peer signals them
        self.bk[key] = st
        for s_ in st:
            s_["bkey"] = key
        if moe != "raw":
            self.bk_bytes[key] = torch.cuda.memory_allocated(self.devs[0]) - m0     # rank 0's share (ranks are alike)
        return st

    def _alloc(self, r: int, R: int, moe: str = "raw", lvl: int = 0) -> dict:
        dev, bf, f32, i32, i64 = self.devs[r], torch.bfloat16, torch.float32, torch.int32, torch.int64
        Tall = W * R
        e, r2 = self.mext[r], self.r2
        with torch.cuda.device(dev):
            s = dict(R=R, dev=dev, moe=moe, lvl=lvl)
            # step inputs (device buffers, rewritten every step)
            if self.cfg.upload_v2:   # views of one device blob (one H2D copy per rank); same initial values
                s["in_d"] = torch.zeros(_in_layout(R)[1], dtype=torch.uint8, device=dev)
                s.update(_in_views(s["in_d"], R))
                s["si"].fill_(-1)
                s["sl"].fill_(1)
            else:
                s.update(pos=torch.zeros(R, dtype=i64, device=dev), tok=torch.zeros(R, dtype=i64, device=dev),
                         own=torch.zeros(R, dtype=i64, device=dev),
                         si=torch.full((R, 1, SLIDING_WINDOW), -1, dtype=i32, device=dev),
                         sl=torch.ones(R, dtype=i32, device=dev), valid=torch.zeros(R, 1, dtype=torch.bool, device=dev),
                         samp=torch.zeros(R, dtype=i64, device=dev))
            s["ti"] = {il: torch.full((R, 1, 512), -1, dtype=i32, device=dev) for il in INDEX_LAYERS}
            s["tl"] = {il: torch.zeros(R, dtype=i32, device=dev) for il in INDEX_LAYERS}
            # residual stream / mixes
            for k in ("X", "xm2", "res0", "Xf"):
                s[k] = torch.zeros(R, HC, D, dtype=bf, device=dev)
            for k in ("a_pre", "a_post", "f_pre", "f_post", "post0", "lag0"):
                s[k] = torch.zeros(R, HC, dtype=f32, device=dev)
            for k in ("a_comb", "f_comb"):
                s[k] = torch.zeros(R, HC, HC, dtype=f32, device=dev)
            s["comb0"] = torch.eye(HC, device=dev).expand(R, HC, HC).contiguous()
            s["lag0"][:, 0] = 1.0                     # layer 0: hc_collapse with one-hot copy 0 (ds41h.ref hc_collapse None)
            for k in ("x0", "a", "f", "moe_out", "h"):
                s[k] = torch.zeros(R, D, dtype=bf, device=dev)
            s["aux"] = torch.zeros(R, len(AUX_LAYERS), D, dtype=bf, device=dev)
            s["X1"] = torch.zeros(R, HC, D, dtype=bf, device=dev)            # Engram: hc_post + injection
            s["zy"] = torch.zeros(R, D, dtype=bf, device=dev)
            s["zpost"] = torch.zeros(R, HC, dtype=f32, device=dev)
            s["eyec"] = torch.eye(HC, device=dev).expand(R, HC, HC).contiguous()
            s["tok32"] = torch.zeros(R, dtype=i32, device=dev)
            s["part_mhc"] = torch.empty(160 * R * 25, dtype=f32, device=dev)
            s["empty_i"] = torch.zeros(0, dtype=i32, device=dev)
            # router
            s.update(logits=torch.empty(R, NE, dtype=f32, device=dev), ids=torch.empty(R, TOPK, dtype=i32, device=dev),
                     wts=torch.empty(R, TOPK, dtype=f32, device=dev), route_row=torch.zeros(R, 64, dtype=torch.uint8, device=dev),
                     ptok_own=torch.zeros(R, dtype=i32, device=dev))
            pad = torch.zeros(1, 64, dtype=torch.uint8, device=dev)
            pad[0, :24] = 0xFF                          # ids -1, weights 0, prescale 0 (= router_top_pack2's rows >= nvalid)
            s["padrow"] = pad
            # AG / RS (peer buffers; flags created and synchronized before use)
            s.update(gf=torch.zeros(Tall, D * 2, dtype=torch.uint8, device=dev), gr=torch.zeros(Tall, 64, dtype=torch.uint8, device=dev),
                     rs=torch.zeros(W, R, D, dtype=bf, device=dev), fl_f=torch.zeros(4, dtype=i32, device=dev),
                     fl_s=torch.zeros(4, dtype=i32, device=dev), ep_f=torch.zeros(1, dtype=i32, device=dev),
                     ep_s=torch.zeros(1, dtype=i32, device=dev))
            # MoE scratch for 4R rows (sp_layer layout); the moe_raw tables only for decode ('raw') buckets
            s.update(part=torch.empty(Tall, D, dtype=bf, device=dev), gu=torch.empty(Tall, 2 * 576, dtype=bf, device=dev),
                     hsh=torch.empty(Tall, 576, dtype=bf, device=dev), sh=torch.empty(Tall, D, dtype=bf, device=dev))
            if moe == "raw" or Tall <= min(4 * self.cfg.prefill_moe_switch, 8192):   # mixed: the fork's MoE for small steps
                                                                                         # (route2 MAXT 8,192)
                max_ec = max(int(e.route_max_ec()), int(r2.route2_max_ec(Tall)))
                s.update(max_ec=max_ec, sms=torch.cuda.get_device_properties(dev).multi_processor_count,
                         ecs=torch.zeros(max_ec * (32 + 12 * 64), dtype=torch.uint8, device=dev),
                         t1=torch.zeros(max_ec * 9 * 24, dtype=torch.uint8, device=dev),
                         t2=torch.zeros(max_ec * 40 * 24, dtype=torch.uint8, device=dev),
                         counts=torch.zeros(8, dtype=i32, device=dev),
                         slotmap=torch.zeros(Tall * TOPK, dtype=i32, device=dev),
                         rts=torch.zeros(max(int(e.route_scratch_ints()), int(r2.route2_scratch_ints())), dtype=i32, device=dev),
                         img=torch.zeros((Tall * TOPK + 7 * max_ec) * 80 * 128, dtype=torch.uint8, device=dev),
                         himg=torch.zeros((Tall * TOPK + 7 * max_ec) * 9 * 128, dtype=torch.uint8, device=dev),
                         Y=torch.zeros(Tall * TOPK, D, dtype=bf, device=dev),
                         g5ctr=torch.zeros(2, dtype=i32, device=dev), empty_f=torch.zeros(0, dtype=f32, device=dev))
            # attention: one workspace (shared by the layers), one cache dict per layer (views of the pools + tables)
            s["ws"] = EA.workspace(R, dev)
            if POISON:
                self._poison_state(s)
            pools = self.pools[r]
            WK = self.WK
            s["caches"] = {}
            for L in self.layers:
                src, isrc = kv_source_of(L), index_source_of(L)
                c = dict(R=R, device=dev, stride=WK.WINDOW_PAGE_BYTES, sbuf=pools.window_base(L), k_swa=pools.window_view(L),
                         si=s["si"], sl=s["sl"], own_slot=s["own"], scale=self.lw[r][L]["AW"]["scale"], ratio=RATIO[L],
                         k_comp=pools.comp_view(src) if src is not None else None,
                         ti=s["ti"][isrc] if isrc is not None else None, tl=s["tl"][isrc] if isrc is not None else None,
                         sched=None, planned=True, src=src, isrc=isrc)
                s["caches"][L] = c
            s["pinned"] = None
        if self.hooks.idx is not None and moe == "raw":
            self.hooks.idx.alloc(r, s)
        return s

    def _poison_state(self, s: dict) -> None:
        """ENG_POISON: the bucket buffers that every step must fully rewrite before reading (torch.empty ones), on the
        rank's engine stream (ordered with the step's kernels)."""
        r = self.devs.index(s["dev"])
        with torch.cuda.device(s["dev"]), torch.cuda.stream(self.streams[r]):
            poison_(s["part_mhc"], s["logits"], s["ids"], s["wts"], s["part"], s["gu"], s["hsh"], s["sh"],
                    *[v for v in s["ws"].values() if isinstance(v, torch.Tensor)])

    def _new_scheds(self, st: list, per_layer: bool = False) -> None:
        """Fresh FlashMLA scheduler objects per rank. Decode buckets: one per layer type (compress ratio): the first layer
        of a type plans (in-kernel; captured into the graph), the others reuse its plan (production's
        tile_sched_{swaonly, c1a, c2a}); valid because every layer of a type sees the same window / top-k LENGTHS there.
        per_layer (mixed steps, eager): one plan per layer. FlashMLA allows reusing a plan only while topk_length and
        extra_topk_length keep their values, and a mixed step's tail bucket gets layer 20's lengths from bucket A
        (layers 21..23) but layers 24..39's from its own indexer: any difference (e.g. padding rows) on a reused plan is
        undefined behaviour (seen as a device fault)."""
        from vllm.third_party.flashmla.flash_mla_interface import get_mla_metadata
        if per_layer and self.cfg.mixed_plans == "source":
            # one plan per (layer type, index source): the layers of a key read the SAME window lengths (s["sl"]) and
            # the SAME top-k length tensor (s["tl"][index_source_of(L)], written once by its index layer before any of
            # them runs), so the plan's inputs are equal by construction (the fault above came from a plan shared
            # ACROSS index sources); ~15 plans instead of 40 per bucket (the planner was 2.4 ms per 1,280-row mixed step)
            for s in st:
                sch, schd = {}, {}
                for L, c in s["caches"].items():
                    k = (RATIO[L], index_source_of(L), bool(s.get("l20")) and L == 20)   # (ENG_L20_TAIL: its own window)
                    if k not in sch:
                        sch[k] = get_mla_metadata()[0]
                        schd[k] = get_mla_metadata()[0] if s.get("pf") is not None else None
                    c["sched"] = sch[k]
                    c["sched_dec"] = schd[k]      # ENG_ATTN_PF: the decode rows' own plan (fewer rows, same key rule)
            return
        for s in st:
            sch = {c: get_mla_metadata()[0] for c in (0, 1, 2)}
            schd = {c: get_mla_metadata()[0] for c in (0, 1, 2)} if s.get("pf") is not None else None
            for L, c in s["caches"].items():
                c["sched"] = get_mla_metadata()[0] if (per_layer or (s.get("l20") and L == 20)) else sch[RATIO[L]]
                c["sched_dec"] = None if schd is None else (get_mla_metadata()[0] if per_layer else schd[RATIO[L]])

    # ------------------------------------------------------------------------------------------------ host tables
    def tables_for(self, batches: list, tokens: list | None = None) -> list:
        """Tests without a worker: RankTables as the worker builds them (serve.worker.TableBuilder; tokens[r] = row
        tokens, None = synthetic, -1 = the engine's device-side next ids)."""
        return [self.tb.build(sb, tokens[r] if tokens is not None else None) for r, sb in enumerate(batches)]

    def _upload(self, r: int, s: dict, tab) -> None:
        """Pad the rank's tables to the bucket and copy them into the bucket's device buffers (one pinned staging block)."""
        if self.cfg.upload_v2:
            return self._upload_v2(r, s, tab)
        R, n = s["R"], tab.n
        i32, i64 = torch.int32, torch.int64
        scratch = self.pools[r].scratch_window_page * WINDOW_PAGE_TOKENS
        pos = torch.zeros(R, dtype=i64)
        tok = torch.zeros(R, dtype=i64)
        own = torch.full((R,), scratch, dtype=i64)
        si = torch.full((R, SLIDING_WINDOW), -1, dtype=i32)
        si[:, 0] = scratch
        sl = torch.ones(R, dtype=i32)
        valid = torch.zeros(R, 1, dtype=torch.bool)
        samp = torch.zeros(R, dtype=i64)
        if n:
            t_ = tab.tokens.to(torch.int64)
            if bool((t_ < 0).any()):        # decode rows: -1 = the engine's device next ids (HSD); else a fixed id
                t_ = torch.where(t_ < 0, torch.full_like(t_, 1000 if self.hooks.hsd is None else 0), t_)
            pos[:n], tok[:n], own[:n] = tab.positions, t_, tab.own_slot
            si[:n], sl[:n] = tab.si, tab.sl
            valid[:n] = True
            samp[: len(tab.sample_rows)] = torch.tensor(tab.sample_rows, dtype=i64)
        ti, tl = self.hooks.host_topk(r, s, tab)
        host = [pos, tok, own, si.view(R, 1, SLIDING_WINDOW), sl, valid, samp]
        dst = [s["pos"], s["tok"], s["own"], s["si"], s["sl"], s["valid"], s["samp"]]
        for il in INDEX_LAYERS:
            if il in ti:
                host += [ti[il].view(R, 1, 512), tl[il]]
                dst += [s["ti"][il], s["tl"][il]]
        for h, d_ in zip(host, dst):
            d_.copy_(h.pin_memory(), non_blocking=True)
        s["n"] = n
        s["n_sample"] = len(tab.sample_rows) if n else 0
        self.hooks.device_tokens(r, s, tab)

    def _upload_v2(self, r: int, s: dict, tab, dev_pos: bool = False) -> None:
        """_upload with the same device result: the 7 step inputs filled (numpy) into one pinned host blob of the bucket
        (two sets: a set is refilled only after its copy from two steps ago completed) -> ONE H2D copy into s["in_d"]
        (the inputs are its views); the stub indexer's host top-k (idx2f: none) still goes per tensor.
        dev_pos (D13 async decode): pos / own / si / sl are left to rowexp on the device (written after this copy)."""
        import numpy as np
        R, n = s["R"], tab.n
        scratch = self.pools[r].scratch_window_page * WINDOW_PAGE_TOKENS
        if s.get("in_h") is None:
            nb = s["in_d"].numel()
            s["in_h"] = [[torch.zeros(nb, dtype=torch.uint8, pin_memory=True), None] for _ in range(2)]
            s["in_hv"] = [_in_views(x[0], R, numpy=True) for x in s["in_h"]]
            s["in_i"] = 0
        i = s["in_i"]
        hb, ev = s["in_h"][i]
        if ev is not None:
            ev.synchronize()
        v = s["in_hv"][i]
        v["tok"][:] = 0
        si = v["si"].reshape(R, SLIDING_WINDOW)
        if not dev_pos:
            v["pos"][:] = 0
            v["own"][:] = scratch
            si[:] = -1
            si[:, 0] = scratch
            v["sl"][:] = 1
        v["valid"][:] = False
        v["samp"][:] = 0
        if n:
            t_ = tab.tokens.numpy().astype(np.int64, copy=False)
            if (t_ < 0).any():        # decode rows: -1 = the engine's device next ids (HSD); else a fixed id
                t_ = np.where(t_ < 0, 1000 if self.hooks.hsd is None else 0, t_)
            v["tok"][:n] = t_
            if not dev_pos:
                v["pos"][:n] = tab.positions.numpy()
                v["own"][:n] = tab.own_slot.numpy()
                si[:n] = tab.si.numpy().reshape(n, SLIDING_WINDOW)
                v["sl"][:n] = tab.sl.numpy()
            v["valid"][:n] = True
            ns_ = len(tab.sample_rows)
            if ns_:
                v["samp"][:ns_] = tab.sample_rows
        s["in_d"].copy_(hb, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(s["dev"]))
        s["in_h"][i][1] = ev
        s["in_i"] = i ^ 1
        ti, tl = self.hooks.host_topk(r, s, tab)
        for il in INDEX_LAYERS:
            if il in ti:
                s["ti"][il].copy_(ti[il].view(R, 1, 512).pin_memory(), non_blocking=True)
                s["tl"][il].copy_(tl[il].pin_memory(), non_blocking=True)
        s["n"] = n
        s["n_sample"] = len(tab.sample_rows) if n else 0
        self.hooks.device_tokens(r, s, tab)

    # ------------------------------------------------------------------------------------------------ the layer
    def _join_mix(self, s: dict) -> None:
        ev = s.pop("mix_ev", None)
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)

    def _split_bd(self, r, s, k1a_rows, xn, lag, sc, bs, nw, out, ptok, pre, post, comb, fused=None):
        """sp_layer split boundary (--mhc-split 1): K1a, K1b-out (main), K1b-mix (side), in 128-row chunks.
        fused = (x, res, post_in, comb_in, fn_f) with ENG_MHCF: the same boundary as kernels/mhc_fuse fa + fb_out + fb_mix
        (bit-identical; fa computes the collapse itself, so it takes lag too)."""
        cfg = self.mcfg
        self._join_mix(s)
        R = s["R"]
        mc = self.cfg.mhc_chunk
        chunks = [(c0, min(R, c0 + mc)) for c0 in range(0, R, mc)]
        if (MHCF and fused is not None and fused[4] is not None and self.mhcf is not None and R <= 320
                and hasattr(self.mhcf, "frow")):
            # decode-sized buckets + the tail bucket: F.frow = collapse + RMSNorm in one kernel per
            # boundary on the main stream (xn, out, ptok); the mixes from xn on the side stream (F.fmix with fn_f + F.fb_mix):
            # the same bits as k1a + k1b_out + k1b_mix
            F = self.mhcf
            xi, ri, pi, ci, fnf = fused[:5]
            F.frow(xi[:R], ri[:R], pi[:R], ci[:R], lag[:R], nw, cfg.norm_eps, xn[:R], out[:R], ptok[:R] if ptok.numel() else ptok)
            cur = torch.cuda.current_stream()
            ev = torch.cuda.Event()
            ev.record(cur)
            with torch.cuda.stream(self.side[r]):
                self.side[r].wait_event(ev)
                part = s["part_mhc"]
                F.fmix(xn[:R], R, fnf, part, 0)
                F.fb_mix(part, R, sc, bs, cfg.norm_eps, cfg.hc_eps, cfg.hc_iters, pre[:R], post[:R], comb[:R])
                ev2 = torch.cuda.Event()
                ev2.record(self.side[r])
            s["mix_ev"] = ev2
            return
        if MHCF and fused is not None and fused[4] is not None and self.mhcf is not None:
            F = self.mhcf
            xi, ri, pi, ci, fnf = fused[:5]
            parts = []
            for c0, c1 in chunks:
                part = s["part_mhc"][160 * 25 * c0: 160 * 25 * c1]
                F.fa(xi[c0:c1], ri[c0:c1], pi[c0:c1], ci[c0:c1], lag[c0:c1], fnf, xn[c0:c1], part, 0)
                parts.append(part)
            cur = torch.cuda.current_stream()
            ev = torch.cuda.Event()
            ev.record(cur)
            for (c0, c1), part in zip(chunks, parts):
                F.fb_out(part, c1 - c0, nw, cfg.norm_eps, out[c0:c1], ptok[c0:c1] if ptok.numel() else ptok)
            with torch.cuda.stream(self.side[r]):
                self.side[r].wait_event(ev)
                for (c0, c1), part in zip(chunks, parts):
                    F.fb_mix(part, c1 - c0, sc, bs, cfg.norm_eps, cfg.hc_eps, cfg.hc_iters, pre[c0:c1], post[c0:c1],
                             comb[c0:c1])
                ev2 = torch.cuda.Event()
                ev2.record(self.side[r])
            s["mix_ev"] = ev2
            return
        parts = []
        for c0, c1 in chunks:
            part = s["part_mhc"][160 * 25 * c0: 160 * 25 * c1]
            k1a_rows(c0, c1, part)
            parts.append(part)
        cur = torch.cuda.current_stream()
        ev = torch.cuda.Event()
        ev.record(cur)
        for c0, c1 in chunks:
            self.hext.k1b_out(xn[c0:c1], lag[c0:c1], nw, cfg.norm_eps, out[c0:c1], ptok[c0:c1] if ptok.numel() else ptok)
        with torch.cuda.stream(self.side[r]):
            self.side[r].wait_event(ev)
            for (c0, c1), part in zip(chunks, parts):
                self.hext.k1b_mix(part, c1 - c0, sc, bs, cfg.norm_eps, cfg.hc_eps, cfg.hc_iters, pre[c0:c1], post[c0:c1],
                                  comb[c0:c1])
            ev2 = torch.cuda.Event()
            ev2.record(self.side[r])
        s["mix_ev"] = ev2

    def _inputs(self, s: dict, L: int):
        if L == self.layers[0] and not s.get("chain_input"):
            return s["x0"], s["res0"], s["post0"], s["comb0"], s["lag0"]
        return s["moe_out"], s["xm2"], s["f_post"], s["f_comb"], s["f_pre"]

    # ------------------------------------------------------------------------------------------------ images (FPRv1.1)
    @staticmethod
    def _route_img(tok):
        lo, hi = IMAGE_ROUTE_IDS
        return (tok >= lo) & (tok <= hi)

    def _img_rows_A(self, stA: list, batches: list, tables: list) -> None:
        """Bucket A of a mixed step (every row, layers 0..20): s["mm_emb"] = (row index, span rows) from StepContext.mm for
        p_embed, s["mm_rt"] = the rows routed with bias_vl for p_router. Both None on text-only steps, so those run
        exactly the text path. ctx.mm[r] covers every span row of rank r's StepBatch: owner chunks, replay rows, CP
        helper sub-chunks."""
        mm = self._mm
        for s in stA:
            s["mm_emb"] = s["mm_rt"] = None
        if not MM_ON:
            assert not (mm and any(m is not None and m[0].numel() for m in mm)), "StepContext.mm needs PHYS_MM=1"
            return
        for r, s in enumerate(stA):
            sb = batches[r]
            n = sb.n_rows
            tok = torch.as_tensor(tables[r].tokens[:n])
            rt = self._route_img(tok)
            m = mm[r] if mm is not None else None
            if m is not None and m[0].numel():
                idx, emb = m
                dev = self.devs[r]
                assert emb.dtype == torch.bfloat16 and emb.dim() == 2 and emb.shape[1] == D and emb.shape[0] == idx.numel(), \
                    (emb.dtype, tuple(emb.shape), idx.numel())
                assert idx.device == dev and emb.device == dev, (idx.device, emb.device, dev)
                with torch.cuda.device(dev), torch.cuda.stream(self.streams[r]):   # stream order: the worker's writes
                    ii = idx.long()
                    hi = ii.cpu() if MM_CHECK else None
                if MM_CHECK:
                    assert bool((hi >= 0).all() and (hi < n).all()) and bool((tok[hi] == IMAGE_TOKEN_ID).all()), \
                        f"rank {r}: ctx.mm rows outside the step or without token {IMAGE_TOKEN_ID}"
                s["mm_emb"] = (ii, emb)
            if MM_CHECK:     # a 129264 row without an embedding (e.g. the placeholder typed as text) keeps the token's
                k_ = int((tok == IMAGE_TOKEN_ID).sum()) - (0 if s["mm_emb"] is None else s["mm_emb"][0].numel())
                if k_:
                    self.stats["mm_rows_plain"] = self.stats.get("mm_rows_plain", 0) + k_
            if bool(rt.any()):
                s["mm_rt"] = rt.nonzero().flatten().to(self.devs[r])

    def _img_rows_T(self, r: int, sT: dict, tab, tail_rows) -> None:
        """Bucket T of a mixed step (layers 21..39 on the tail rows): its rows routed with bias_vl."""
        sT["mm_emb"] = sT["mm_rt"] = None
        if not MM_ON:
            return
        tt = torch.as_tensor(tab.tokens)[torch.as_tensor(list(tail_rows), dtype=torch.long)] if len(tail_rows) else None
        if tt is not None:
            rt = self._route_img(tt)
            if bool(rt.any()):
                sT["mm_rt"] = rt.nonzero().flatten().to(self.devs[r])

    def _bias_vl(self, r: int, L: int):
        b = self._bvl.get((r, L))
        if b is None:
            b = self._bvl[(r, L)] = self.reader.get(f"layers.{L}.ffn.gate.bias_vl").float().to(self.devs[r])
        return b

    def _router_vl(self, r: int, s: dict, L: int) -> None:
        """Image-token rows: production's gate selects their experts with bias_vl instead of the correction bias (the
        weights come from the unbiased scores either way: vLLM topk_softplus_sqrt with bias_vl / image_sentinel_lo;
        DeepSeek's reference Gate: bias = where(image_mask, bias_vl, bias)). router_top_pack3 computes every row in
        its own warp, so running it on the compacted image rows with bias_vl and scattering ids / weights / routing rows
        back gives exactly a per-row-bias router."""
        ri = s["mm_rt"]
        n = ri.numel()
        lg = s["logits"].index_select(0, ri)
        ids = torch.empty((n,) + tuple(s["ids"].shape[1:]), dtype=s["ids"].dtype, device=lg.device)
        wts = torch.empty((n,) + tuple(s["wts"].shape[1:]), dtype=s["wts"].dtype, device=lg.device)
        rr = s["route_row"].view(torch.uint8).view(-1, 64)
        row = rr.index_select(0, ri)          # the kernel leaves some bytes as they were: start from the rows
        valid = torch.ones(n, dtype=torch.bool, device=lg.device)
        ptok = s["ptok_own"].index_select(0, ri)
        self.mext[r].router_top_pack3(lg, self._bias_vl(r, L), ids, wts, 1.5, row, valid, ptok)
        s["ids"].index_copy_(0, ri, ids)
        s["wts"].index_copy_(0, ri, wts)
        rr.index_copy_(0, ri, row)

    def p_embed(self, r: int, s: dict) -> None:
        if self.hooks.engram is not None:      # token-only Engram work (hash, host-table gathers, wkv) on its own stream
            self.hooks.ids32[r][: s["R"]].copy_(s["tok"])
            self.hooks.engram[r].prepare(s["R"], stream=self.estream[r])
        if s.get("inject_input"):      # tests: the first layer's input stream was set directly (captured data)
            return
        R = s["R"]
        emb = torch.index_select(self.glob[r]["emb"], 0, s["tok"])
        mm = s.get("mm_emb")
        if mm is not None:             # images: span rows take their embedding instead of the token's, before the hc copies
            emb.index_copy_(0, mm[0], mm[1])
        s["res0"].copy_(emb.unsqueeze(1).expand(R, HC, D))

    def p_attn_bd(self, r: int, s: dict, L: int) -> None:
        lw = self.lw[r][L]
        x, res, post, comb, lag = self._inputs(s, L)
        if L in ENGRAM_LAYERS and self.hooks.engram is not None:
            # Engram: X1 = hc_post(prev) + injection in one pass, then the boundary in identity mode on X1
            # (post 0, comb I: K1a's X == X1 bit for bit; mixes / collapse / RMSNorm of the injected stream, as production)
            if BISECT["inject_join"]:
                self._join_mix(s)   # inject reads f_post / f_comb of the side stream's k1b_mix (without this join the
                                # eager production-MoE path had no happens-before edge there)
            self.hooks.engram[r].inject(ENGRAM_LAYERS.index(L), s["R"], y=x, rs=res, post=post, comb=comb, out=s["X1"])
            x, res, post, comb = s["zy"], s["X1"], s["zpost"], s["eyec"]

        def k1(c0, c1, part):
            self.hext.k1a(x[c0:c1], res[c0:c1], post[c0:c1], comb[c0:c1], lw["fa"], s["X"][c0:c1], part)
        self._split_bd(r, s, k1, s["X"], lag, lw["sa"], lw["ba"], lw["na"], s["a"], s["empty_i"], s["a_pre"], s["a_post"],
                       s["a_comb"], fused=(x, res, post, comb, lw.get("fa_f"), lw["fa"]))
        if L in AUX_LAYERS:  # DSpark aux = stock previous_post.mean(dim=1) on the bf16 block input (hsd.aux_of: fp32 sum,
            s["aux"][:, AUX_LAYERS.index(L)].copy_(s["X"].mean(1))   # one bf16 rounding)

    def p_attn_a(self, r: int, s: dict, L: int) -> None:
        EA.attn_a(self.lw[r][L]["AW"], s["ws"], s["a"], s["pos"])

    def p_attn_b(self, r: int, s: dict, L: int) -> None:
        EA.attn_b(self.lw[r][L]["AW"], s["caches"][L], s["ws"], s["pos"], insert=self.cfg.insert, pf=s.get("pf"))

    def p_attn(self, r: int, s: dict, L: int) -> None:
        AW, ws, c = self.lw[r][L]["AW"], s["ws"], s["caches"][L]
        EA.attn_a(AW, ws, s["a"], s["pos"])
        if DBG_SYNC:
            self._dbg_sync(f"attn_a L{L} R{s['R']} {s.get('moe')} (rank {r})", ranks=(r,))
        self.hooks.indexer_run(r, s, L)          # indexer + compressor of index layers (placeholder: nothing on device)
        if DBG_SYNC:
            self._dbg_sync(f"indexer L{L} R{s['R']} {s.get('moe')} (rank {r})", ranks=(r,))
        if DBG_SYNC and s.get("moe") == "prod" and not torch.cuda.is_current_stream_capturing():
            self._dbg_lengths(r, s, L, c)
        EA.attn_b(AW, c, ws, s["pos"], insert=self.cfg.insert, pf=s.get("pf"))

    # -- CP prefill (engine/cp.py): p_attn in 3 phases (same calls; the exchanges between them)
    def _cp_p1(self, r: int, s: dict, L: int) -> None:
        EA.attn_a(self.lw[r][L]["AW"], s["ws"], s["a"], s["pos"])
        if self.hooks.idx is not None and L in INDEX_LAYERS:
            self.hooks.idx._compress_general(r, s, L)
            if L in KV_SOURCE_LAYERS:
                self._cp.x1a_send(r, L)

    def _cp_p2(self, r: int, s: dict, L: int) -> None:
        if self.hooks.idx is not None and L in INDEX_LAYERS:
            if L in KV_SOURCE_LAYERS:
                self._cp.x1a_recv(r, L)
            self.hooks.idx._score_general(r, s, L)
        EA.attn_b_insert(self.lw[r][L]["AW"], s["caches"][L], s["ws"], s["pos"], insert=self.cfg.insert)
        self._cp.x2_send(r, L)

    def _cp_p3(self, r: int, s: dict, L: int) -> None:
        self._cp.x2_recv(r, L)
        EA.attn_b_attend(self.lw[r][L]["AW"], s["caches"][L], s["ws"], s["pos"], pf=s.get("pf"))

    def _dbg_devices(self, st: list, what: str) -> None:
        """ENG_SYNC_DEBUG: every CUDA tensor reachable from rank r's step state (bucket buffers, caches, indexer state incl.
        its OwnedIndexer / CompactK objects, the layers' weights) must live on rank r's device: a rank-k kernel given a
        device-0 pointer reads it silently through peer access."""
        bad = []
        for r, s in enumerate(st):
            dev = s["dev"]
            seen, stack = set(), [("state", s), ("weights", self.lw[r])]
            while stack:
                path, x = stack.pop()
                if id(x) in seen:
                    continue
                seen.add(id(x))
                if isinstance(x, torch.Tensor):
                    if x.is_cuda and x.device != dev:
                        bad.append((r, path, str(x.device), tuple(x.shape)))
                elif isinstance(x, dict):
                    stack.extend((f"{path}.{k}", v) for k, v in x.items() if k not in ("batch", "tab", "dev"))
                elif isinstance(x, (list, tuple)):
                    stack.extend((f"{path}[{i}]", v) for i, v in enumerate(x))
                elif hasattr(x, "__dict__") and not isinstance(x, (type, torch.nn.Module)) and \
                        type(x).__module__.split(".")[0] in ("idx2f", "tables2f", "engram", "hsd", "builtins", "types", "vllm"):
                    stack.extend((f"{path}.{k}", v) for k, v in vars(x).items())
        if bad:
            print(f"ENG_SYNC_DEBUG devices [{what}]: {len(bad)} tensors on another device, e.g. {bad[:6]}", flush=True)
            self.stats.setdefault("dbg_device_mismatch", []).extend(bad[:20])

    def _dbg_lengths(self, r: int, s: dict, L: int, c: dict) -> None:
        """ENG_SYNC_DEBUG, mixed buckets: rows whose FlashMLA lengths differ from the first layer of the same type in this
        bucket and step (where a per-type plan would have been made; FlashMLA allows plan reuse only for equal lengths)."""
        ref = s.setdefault("_dbg_ref", {})
        tl = c["tl"] if c.get("k_comp") is not None else None
        key = RATIO[L]
        cur = (s["sl"].clone(), tl.clone() if tl is not None else None)
        if key not in ref:
            ref[key] = (L, cur)
            return
        L0, (sl0, tl0) = ref[key]
        n = s.get("n", 0)
        R = s["R"]
        dsl = int((cur[0] != sl0).sum())
        dtl = int((cur[1] != tl0).sum()) if tl0 is not None else 0
        if dsl or dtl:
            dt_rows = (cur[1] != tl0).nonzero().flatten() if dtl else torch.zeros(0, dtype=torch.int64)
            pad = int((dt_rows >= n).sum()) if dtl else 0
            ex = [(int(i), int(tl0[i]), int(cur[1][i])) for i in dt_rows[:3].tolist()] if dtl else []
            self.stats.setdefault("dbg_len_mismatch", []).append((r, L, L0, dsl, dtl, pad))
            print(f"ENG_SYNC_DEBUG lengths: rank {r} layer {L} vs {L0} (R {R}, rows {n}): window {dsl}, top-k {dtl} rows "
                  f"differ ({pad} padding); e.g. (row, L{L0}, L{L}) {ex}", flush=True)

    def p_ffn_bd(self, r: int, s: dict, L: int) -> None:
        lw = self.lw[r][L]
        y = s["ws"]["y"]

        def k1(c0, c1, part):
            self.hext.k1a(y[c0:c1], s["X"][c0:c1], s["a_post"][c0:c1], s["a_comb"][c0:c1], lw["ff"], s["xm2"][c0:c1], part)
        self._split_bd(r, s, k1, s["xm2"], s["a_pre"], lw["sf"], lw["bf"], lw["nf"], s["f"], s["ptok_own"], s["f_pre"],
                       s["f_post"], s["f_comb"], fused=(y, s["X"], s["a_post"], s["a_comb"], lw.get("ff_f"), lw["ff"]))

    def p_router(self, r: int, s: dict, L: int) -> None:
        lw = self.lw[r][L]
        torch.mm(s["f"], lw["gate_w"].t(), out_dtype=torch.float32, out=s["logits"])
        if self.cfg.router_pack3:   # the valid-row mask in-kernel (same bytes; -2 launches per layer)
            self.mext[r].router_top_pack3(s["logits"], lw["gate_b"], s["ids"], s["wts"], 1.5, s["route_row"], s["valid"],
                                          s["ptok_own"])
            if s.get("mm_rt") is not None:
                self._router_vl(r, s, L)
            return
        assert s.get("mm_rt") is None, "image-token routing needs router_pack3"
        self.mext[r].router_top_pack2(s["logits"], lw["gate_b"], s["ids"], s["wts"], 1.5, s["route_row"], s["R"], s["ptok_own"])
        # padded rows -> ids -1, prescale 0 (device mask: one graph per bucket serves every row count)
        s["route_row"].copy_(torch.where(s["valid"], s["route_row"], s["padrow"]))

    def p_ag(self, r: int, s: dict, L: int, st: list) -> None:
        R = s["R"]
        self.cext.ag_push2(s["f"].view(torch.uint8).view(R, D * 2), s["route_row"], r, [x["gf"] for x in st],
                           [x["fl_f"] for x in st], [x["gr"] for x in st], s["ep_f"], self.cfg.ncta)
        self.cext.wait(s["fl_f"], s["ep_f"], W * self.cfg.ncta)

    def p_moe(self, r: int, s: dict, L: int) -> None:
        if s["moe"] == "prod" and (s.get("vcount", 1 << 30) > self.cfg.prefill_moe_switch or "ecs" not in s):
            return self.p_moe_prod(r, s, L)
        lw, e = self.lw[r][L], self.mext[r]
        Tall = W * s["R"]
        fall = s["gf"].view(torch.bfloat16).view(Tall, D)
        cur = torch.cuda.current_stream()
        ev0 = torch.cuda.Event()
        ev0.record(cur)
        with torch.cuda.stream(self.side[r]):        # shared expert slice on the side stream (stock rounding points)
            self.side[r].wait_event(ev0)
            torch.mm(fall, lw["sw13"].t(), out=s["gu"])
            e.swiglu(s["gu"], s["hsh"])
            torch.mm(s["hsh"], lw["sw2"].t(), out=s["sh"])
            ev1 = torch.cuda.Event()
            ev1.record(self.side[r])
        gx = e
        if Tall <= 1365 and self.cfg.moe_v4 and (s["moe"] == "raw" or V4_MIXED or self.cfg.moe_v4_mixed):   # moe_v4 drop-in (decode buckets): its route2r
                                               # (pair lists) feeds its GEMMs; prepr5 / combine
            gx, v4r2 = self.moe_v4()           # and every buffer layout are byte-identical to moe_raw + route2
            v4r2.route2r(s["gr"], Tall, s["ecs"], s["t1"], s["t2"], s["counts"], s["slotmap"], s["rts"], s["sms"])
        elif Tall <= 1365:  # route2r: single-kernel path, routing rows read in place (route2.cu k_one limit)
            self.r2.route2r(s["gr"], Tall, s["ecs"], s["t1"], s["t2"], s["counts"], s["slotmap"], s["rts"], s["sms"])
        else:               # larger (mixed-step) buckets: route2's two-kernel path, same tables (sp_layer checks route2r == route2)
            ids = s["gr"][:, :24].contiguous().view(torch.int32).view(Tall, TOPK)
            wts = s["gr"][:, 24:48].contiguous().view(torch.float32).view(Tall, TOPK)
            self.r2.route2(ids, wts, Tall, s["ecs"], s["t1"], s["t2"], s["counts"], s["slotmap"], s["rts"], s["sms"],
                           s["empty_f"])
        if G6 and Tall <= 1365 and s["moe"] == "raw":       # ENG_G6: decode buckets (the range checked bit-identical)
            self.g6().gather6(fall, s["ecs"], s["counts"], s["img"], s["gr"], s["sms"])
        else:
            e.prepr5(fall, s["ecs"], s["counts"], s["img"], s["gr"], s["sms"], s["g5ctr"])
        gx.gemm1r(lw["w13"], lw["s13"], s["ecs"], s["t1"], s["counts"], s["img"], s["himg"], 0)
        gx.gemm2r(lw["w2"], lw["s2"], s["ecs"], s["t2"], s["counts"], s["himg"], s["Y"], 0)
        cur.wait_event(ev1)
        e.combine(s["Y"], s["slotmap"], s["sh"], s["part"])
        if DUMP_MOE and s["moe"] == "prod" and self.stats.get("mixed_steps", 0) == 0:
            self._dump_moe(r, s, L, Tall, fall)

    def g6(self):
        """ENG_G6: kernels/moe_v4's gather6 extension (kernels/moe_v4/g6.cu, NSLOT 8, B 2; per-device SMEM attribute and
        CUDAGuard inside). Built once into engine/build/eng_moe_g6 (prebuilt on the CPU), loaded on first use (a
        bucket's eager first step, before its capture)."""
        x = getattr(self, "_g6", None)
        if x is None:
            from torch.utils.cpp_extension import load
            os.makedirs(os.path.join(PHYS, "engine/build/eng_moe_g6"), exist_ok=True)
            x = self._g6 = load(name="eng_moe_g6s8b2", sources=[os.path.join(PHYS, "kernels/moe_v4/g6.cu")],
                                build_directory=os.path.join(PHYS, "engine/build/eng_moe_g6"), verbose=False,
                                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                                                   "-DG6_NSLOT=8", "-DG6_B=2"])
        return x

    def _dump_moe(self, r: int, s: dict, L: int, Tall: int, fall) -> None:
        """ENG_DUMP_MOE=<dir> (debug, synthetic tests only): the FIRST mixed step's fork-MoE calls (mixed buckets): md5 of
        every call's partial output, and inputs + output of layers 21..24 on every rank (to replay the v4 vs moe_raw
        mixed-bucket difference)."""
        import hashlib
        torch.cuda.current_stream().synchronize()
        part = s["part"][:Tall]
        md5 = hashlib.md5(part.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:12]
        self.stats.setdefault("moe_dump_md5", {})[f"{s['bkey']}|L{L}|r{r}"] = md5
        if 21 <= L <= 24:
            os.makedirs(DUMP_MOE, mode=0o700, exist_ok=True)
            torch.save(dict(gr=s["gr"][:Tall].cpu(), fall=fall.cpu(), part=part.cpu(), Tall=Tall, R=s["R"],
                            vcount=s.get("vcount"), L=L, rank=r, moe_v4=bool(self.cfg.moe_v4), bkey=str(s["bkey"])),
                       os.path.join(DUMP_MOE, f"v4_{int(self.cfg.moe_v4)}_L{L}_r{r}.pt"))

    def p_moe_prod(self, r: int, s: dict, L: int) -> None:
        """Production's routed MoE (ds41h.moe.moe_forward_w4a16, production env: moe_dec 0x55 up to 512 rows,
        W4A16 0xB9 + hybrid above) on the VALID rows of all ranks (the call size picks production's variant), the shared
        expert slice folded into its combine (addend: fp32 sum, one bf16 rounding, as production's prefill SP path), then
        the same RS as the decode path. Eager only (data-dependent sizes).
        PM_FUSED: the same kernels and arithmetic in one host call (pm_fused.layer, compact rows: row gather + route-row
        unpack in one kernel, the shared slice into the bucket's gu / hsh / sh, production's permute / act_perm16 /
        GEMMs, production's combine writing part in place with zero padding rows); bit-identical to the path below."""
        lw = self.lw[r][L]
        Tall = W * s["R"]
        fall = s["gf"].view(torch.bfloat16).view(Tall, D)
        if PM_FUSED and s["vcount"] > self.prod_moe().SMALL_T:
            if PF_ACT8 and s.get("ndec", -1) >= 0:      # decode rows (a prefix of each rank's rows) W4A16, prefill rows W4A8
                n8 = self.stats.get("act8_calls", 0)
                if n8 == 0:                             # once per run (reset_state resets stats): proof that the arm engaged
                    print(f"engine: ENG_PF_ACT8: W4A8 prefill-row MoE on (first call: layer {L} rank {r}, {s['vcount'] - s['ndec']} "
                          f"prefill + {s['ndec']} decode rows; stats act8_calls = layer x rank calls, act8_layer_mask = layers)",
                          flush=True)
                self.stats["act8_calls"] = n8 + 1
                self.stats["act8_layer_mask"] = self.stats.get("act8_layer_mask", 0) | (1 << L)
                (self.pmf8() if PF_ACT8_DEC else self.pmf()).layer8(
                    fall, s["gr"], s["nv"], s["nd"], s["R"], s["vcount"], s["ndec"], lw["w13"], lw["s13"], lw["w2"], lw["s2"],
                    lw["sw13"], lw["sw2"], s["gu"], s["hsh"], s["sh"], s["part"])
                return
            self.pmf().layer(fall, s["gr"], s["nv"], s["R"], s["vcount"], lw["w13"], lw["s13"], lw["w2"], lw["s2"], lw["sw13"],
                             lw["sw2"], s["gu"], s["hsh"], s["sh"], s["part"], padded=False)
            return
        vi = s["vidx"]
        fv = fall.index_select(0, vi)
        gu = torch.mm(fv, lw["sw13"].t())
        hsh = torch.empty(fv.shape[0], 576, dtype=torch.bfloat16, device=fv.device)
        self.mext[r].swiglu(gu, hsh)
        sh = torch.mm(hsh, lw["sw2"].t())
        rows = s["gr"].index_select(0, vi)
        ids = rows[:, :24].contiguous().view(torch.int32).view(-1, TOPK)
        wts = rows[:, 24:48].contiguous().view(torch.float32).view(-1, TOPK)
        y = self.prod_moe().moe_forward_w4a16(fv, wts, ids, lw["w13"], lw["s13"], lw["w2"], lw["s2"], 10.0, addend=sh)
        s["part"].zero_()
        s["part"].index_copy_(0, vi, y)

    def pmf(self):
        """kernels/moe_prefill/pm_fused.Fused (read-only import; built into kernels/moe_prefill/build by pm_fused.build(),
        on the CPU) with production's W4A16 variant and hybrid setting read from ds41h.moe (prod_moe)."""
        if getattr(self, "_pmf", None) is None or getattr(self, "_pmf_nt", "0") != PM_NT:   # PM_NT: runtime flag per arm
            p = os.path.join(PHYS, "kernels")
            if p not in sys.path:
                sys.path.insert(0, p)
            cache = self.__dict__.setdefault("_pmf_by", {})
            if PM_NT not in cache:
                if PM_NT in ("1", "fast"):
                    from moe_prefill import pmv
                    cache[PM_NT] = pmv.FusedV(self.prod_moe(), defs=pmv.NT_SETS[PM_NT], snap=pmv.NT_SNAP)
                else:
                    from moe_prefill import pm_fused
                    cache[PM_NT] = pm_fused.Fused(self.prod_moe())
            self._pmf, self._pmf_nt = cache[PM_NT], PM_NT
        return self._pmf

    def pmf8(self):
        """PF_ACT8_DEC: kernels/moe_prefill/pm2.Fused2 (read-only import; built into kernels/moe_prefill/build/mpf_pm2 by pm2.py
        on the CPU): pm_fused's layer8 with the decode-row sub-call on small16k variant PF_ACT8_DEC and the shared expert
        slice on a side stream (set_overlap). Production's W4A16 variant and hybrid setting, as pmf()."""
        if getattr(self, "_pmf8", None) is None or self._pmf8.vdec != PF_ACT8_DEC:
            p = os.path.join(PHYS, "kernels")
            if p not in sys.path:
                sys.path.insert(0, p)
            from moe_prefill import pm2
            self._pmf8 = pm2.Fused2(self.prod_moe(), vdec=PF_ACT8_DEC)
            self._pmf8.e.set_overlap(True)
        return self._pmf8

    def moe_v4(self):
        """(ext, route2) of kernels/moe_v4/v4.py (read-only import; built into engine/build/moe_v4, MOE_V4_BUILD)."""
        if getattr(self, "_v4", None) is None:
            p = os.path.join(PHYS, "kernels")
            if p not in sys.path:
                sys.path.insert(0, p)
            from moe_v4 import v4
            self._v4 = v4.load()
        return self._v4

    def prod_moe(self):
        """ds41h.moe from production's ds41h package (read-only), next to the reference package ds41h.ref."""
        if getattr(self, "_pm", None) is None:
            for k, v in (("DS41H_MOE_DEC", "1"), ("DS41H_MOE_DEC_CFG", "0x55")):   # production env
                os.environ.setdefault(k, v)
            import ds41h
            dt = os.environ.get("DS41H_DEPLOY", os.path.dirname(PHYS)) + "/ds41h"
            if dt not in list(ds41h.__path__):
                ds41h.__path__.append(dt)
            from ds41h import moe as PM
            from . import prod_moe_ext
            PM.ext = prod_moe_ext.ext     # production's kernels with a per-DEVICE smem opt-in (engine/prod_moe_ext.py)
            # ds41h.moe reads DS41H_MOE_DEC once, at import; in the served process vLLM's ds41h plugin imports it before
            # the engine exists, so the setdefault above comes too late there (the served engine then ran MOE_DEC 0,
            # small calls on W4A16 0x2500 instead of moe_dec 0x55). Its call path reads the module global.
            want = int(os.environ.get("DS41H_MOE_DEC", "1") or 0)
            if PM.MOE_DEC != want:
                self.log(f"engine: ds41h.moe was imported with MOE_DEC={PM.MOE_DEC}; set to {want} (production's env)")
                PM.MOE_DEC = want
            self._pm = PM
        return self._pm

    def p_rs(self, r: int, s: dict, L: int, st: list) -> None:
        self.cext.rs_push(s["part"], r, [x["rs"].view(torch.uint8) for x in st], [x["fl_s"] for x in st], s["ep_s"],
                          self.cfg.ncta)
        self.cext.rs_sum(s["rs"], s["fl_s"], s["ep_s"], W * self.cfg.ncta, s["moe_out"], 132)

    def p_final(self, r: int, s: dict) -> None:
        """Head input: h = norm(hc_collapse(X_out_39, ffn_pre_39)) * norm.weight (mHC boundary with lag = ffn_pre_39)."""
        self._join_mix(s)
        R = s["R"]
        if MHCF and self.mhcf is not None and R <= 320 and hasattr(self.mhcf, "frow"):   # (same bits)
            self.mhcf.frow(s["moe_out"], s["xm2"], s["f_post"], s["f_comb"], s["f_pre"], self.glob[r]["norm_w"],
                           self.mcfg.norm_eps, s["Xf"], s["h"], s["empty_i"])
            self.hooks.head_run(r, s)
            return
        mc = self.cfg.mhc_chunk
        for c0 in range(0, R, mc):
            c1 = min(R, c0 + mc)
            self.hext.k1a_x(s["moe_out"][c0:c1], s["xm2"][c0:c1], s["f_post"][c0:c1], s["f_comb"][c0:c1], s["Xf"][c0:c1])
            self.hext.k1b_out(s["Xf"][c0:c1], s["f_pre"][c0:c1], self.glob[r]["norm_w"], self.mcfg.norm_eps, s["h"][c0:c1],
                              s["empty_i"])
        self.hooks.head_run(r, s)

    # ------------------------------------------------------------------------------------------------ the step
    def _issue(self, st: list, final: bool = True, trace=None, layers=None, embed: bool = True) -> None:
        """Issue the whole step for all ranks, piece by piece (every piece for every rank before the next one: a hidden
        host sync inside a piece never waits on a peer push that was not issued yet; sp_layer run_all)."""
        def each(fn, name="", L=None):
            if PROF_PIECES:
                rf = torch.profiler.record_function(f"piece:{name}")
                rf.__enter__()
            for r in range(W):
                with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                    fn(r, st[r])
            if PROF_PIECES:
                rf.__exit__(None, None, None)
            if DBG_SYNC:
                self._dbg_sync(f"{name} L{L} R{st[0]['R']} {st[0].get('moe')}")
        if embed:
            each(self.p_embed, "embed")
        cp = self._cp if (getattr(self, "_cp", None) and st[0].get("cp_on")) else None
        for L in (self.layers if layers is None else layers):
            each(lambda r, s: self.p_attn_bd(r, s, L), "attn_bd", L)
            if cp is not None and L <= 20:     # CP prefill: every send of a phase issued for all ranks before a receive
                each(lambda r, s: self._cp_p1(r, s, L), "attn P1", L)
                each(lambda r, s: self._cp_p2(r, s, L), "attn P2", L)
                each(lambda r, s: self._cp_p3(r, s, L), "attn P3", L)
            else:
                each(lambda r, s: self.p_attn(r, s, L), "attn", L)
            each(lambda r, s: self.p_ffn_bd(r, s, L), "ffn_bd", L)
            each(lambda r, s: self.p_router(r, s, L), "router", L)
            each(lambda r, s: self.p_ag(r, s, L, st), "ag", L)
            each(lambda r, s: self.p_moe(r, s, L), "moe", L)
            each(lambda r, s: self.p_rs(r, s, L, st), "rs", L)
            if trace:
                trace(L)
            if NANCHECK:
                self._nan_check(st, L)
        if final:
            each(self.p_final, "final")
        each(lambda r, s: self._join_mix(s), "join")

    # ------------------------------------------------------------------------------------------------ mixed graphs
    def _mixed_units(self, layers: tuple, final: bool, embed: bool) -> list:
        """A mixed segment as units: ("g", pieces) = fixed shapes per bucket, captured once and replayed; ("e", pieces) =
        eager every step: embed, the Engram layers' attention boundary (inject waits on Engram's events), the indexer of
        index layers (widths / pseudo-request counts per step), the MoE (fork vs production by the step's valid rows).
        Every graph unit ends with the mHC side stream joined (a capture may not leave a forked stream open)."""
        units, cur = [], []

        def flush():
            if cur:
                units.append(("g", cur + [("join", None)]))
                cur.clear()

        def eager(pc):
            flush()
            units.append(("e", pc))
        if embed:
            eager([("embed", None)])
        for L in layers:
            if L in ENGRAM_LAYERS and self.hooks.engram is not None:
                eager([("attn_bd", L)])
            else:
                cur.append(("attn_bd", L))
            cur.append(("attn_a", L))
            if L in INDEX_LAYERS and self.hooks.idx is not None:
                eager([("idx", L)])
            cur += [("attn_b", L), ("ffn_bd", L), ("router", L), ("ag", L)]
            eager([("moe", L)])
            cur.append(("rs", L))
        if final:
            cur.append(("final", None))
        flush()
        return units

    def _piece(self, name: str, L, st: list):
        return {"embed": lambda r, s: self.p_embed(r, s), "attn_bd": lambda r, s: self.p_attn_bd(r, s, L),
                "attn_a": lambda r, s: self.p_attn_a(r, s, L), "idx": lambda r, s: self.hooks.indexer_run(r, s, L),
                "attn_b": lambda r, s: self.p_attn_b(r, s, L), "ffn_bd": lambda r, s: self.p_ffn_bd(r, s, L),
                "router": lambda r, s: self.p_router(r, s, L), "ag": lambda r, s: self.p_ag(r, s, L, st),
                "moe": lambda r, s: self.p_moe(r, s, L), "rs": lambda r, s: self.p_rs(r, s, L, st),
                "final": lambda r, s: self.p_final(r, s), "join": lambda r, s: self._join_mix(s)}[name]

    def _issue_mixed(self, st: list, layers: list, final: bool, embed: bool) -> None:
        """Piecewise graphs for a mixed segment (cfg.mixed_graphs): the bucket's first step runs every unit eagerly and
        then captures the graph units (one rank at a time, nothing executes during capture); later steps replay them,
        piece by piece over the ranks as _issue does, with the eager units in between. Per-step data reaches the graphs
        only through the bucket's device buffers (positions, tables, masks, top-k): no host value inside a capture."""
        bkey = (st[0]["bkey"], tuple(layers), final, embed)
        ent = self.mgraphs.get(bkey)
        units = ent[0] if ent is not None else self._mixed_units(tuple(layers), final, embed)
        for ui, (kind, pcs) in enumerate(units):
            if kind == "g" and ent is not None:
                for r in range(W):
                    with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                        self._join_mix(st[r])        # a preceding eager boundary's side-stream mixes, before the graph
                        ent[1][ui][r].replay()
                continue
            for name, L in pcs:
                fn = self._piece(name, L, st)
                for r in range(W):
                    with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                        fn(r, st[r])
        if ent is None:                       # capture after this (eager) step
            self.sync()
            self._new_scheds(st, per_layer=True)      # fresh per-layer FlashMLA plans, planned in-kernel in the captures
            keep = [[c["sched"] for c in s_["caches"].values()] for s_ in st]
            gl = []
            for ui, (kind, pcs) in enumerate(units):
                if kind != "g":
                    gl.append(None)
                    continue
                gs = []
                for r in range(W):
                    with torch.cuda.device(self.devs[r]):
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.stream(self.streams[r]):
                            self._join_mix(st[r])   # no event recorded outside the capture may be waited on inside it
                            g.capture_begin(pool=self.gpool[r])
                            try:
                                for name, L in pcs:
                                    self._piece(name, L, st)(r, st[r])
                            finally:
                                g.capture_end()
                        gs.append(g)
                gl.append(gs)
            self.sync()
            self.mgraphs[bkey] = (units, gl, keep)
            self.stats["mixed_captures"] = self.stats.get("mixed_captures", 0) + 1

    def copy_state_slots(self, ops) -> None:
        """serve Worker.copies: engine-side per-slot state follows a state-slot copy (a rank move, kv_manager.migrate).
        Engram's token-history ring (EngramPhys.ring [n_state, 16, 2], read by decode rows' n-gram lookback) is not part
        of the pools; without this a moved sequence's first decode rows read the destination slot's previous owner
        (seen as wrong output for cached prefixes under load). ops = [(src_rank, src_slot, dst_rank,
        dst_slot)], run with the step's other copies, before its kernels."""
        eg = getattr(self.hooks, "engram", None)
        if eg is None or not ops:
            return
        for sr, ss, dr, ds in ops:
            with torch.cuda.device(self.devs[dr]):
                eg[sr].ring_copy(int(ss), int(ds), dst=eg[dr])
        for d in self.devs:
            torch.cuda.synchronize(d)
        self.stats["ring_copies"] = self.stats.get("ring_copies", 0) + len(ops)

    def poison_state_slots(self, rank: int, slots) -> None:
        """serve PHYS_POISON_ON_FREE (debug): engine-side state keyed by freed state slots made invalid. Engram's token
        ring of those slots -> positions -1 (EngramPhys.ring_reset): a later lookback that would have matched a stale
        (position, token) entry of the previous owner becomes a ring miss, counted in EngramPhys.err (ENG_NANCHECK logs
        it). The indexer's per-slot block-table rows are left alone: a stale page id there points at a real page (serve
        poisons the freed pages themselves); an invalid id would fault on reads the kernels mask."""
        eg = getattr(self.hooks, "engram", None)
        sl = [int(x) for x in slots]
        if eg is None or not sl:
            return
        e = eg[rank]
        with torch.cuda.device(self.devs[rank]):
            e.ring.index_fill_(0, torch.tensor(sl, dtype=torch.int64, device=self.devs[rank]), -1)
        self.stats["poisoned_state_slots"] = self.stats.get("poisoned_state_slots", 0) + len(sl)

    def _nan_check(self, st: list, L: int) -> None:
        """ENG_NANCHECK: rows of each rank's block input X of layer L (computed at its attention boundary) that are not
        finite; logs the first occurrence per step with the rows' sequences (CP role, kind, positions)."""
        b0 = st[0].get("batch")
        key = (getattr(b0, "step", None), st[0].get("bkey"))
        if getattr(self, "_nan_seen", None) == key:
            return
        for r in range(W):
            s = st[r]
            n = s["batch"].n_rows if s.get("batch") is not None else 0
            if not n:
                continue
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                bad = (~torch.isfinite(s["X"][:n].float()).view(n, -1).all(1)).nonzero().flatten()
                nb = int(bad.numel())
            if nb:
                sb = s["batch"]
                i = int(bad[0])
                se = sb.seqs[sb.seq_slot[i]]
                cp = getattr(se, "cp", None)
                self.log(f"engine NANCHECK step {key[0]} layer {L} rank {r} (bucket {key[1]}): {nb} / {n} rows non-finite; "
                         f"first row {i} = {se.rid} {se.kind} pos {sb.positions[i] if i < len(sb.positions) else '?'} "
                         f"start {se.start} n {se.n_rows}{' CP me ' + str(cp.me) + ' owner ' + str(cp.owner) if cp else ''}")
                self._nan_seen = key
                self.stats["nan_layers"] = self.stats.get("nan_layers", 0) + 1
                return

    def _dbg_sync(self, what: str, timeout: float = 60.0, ranks=None) -> None:
        """ENG_SYNC_DEBUG=1: wait for every rank's main + side stream after `what`, polling events (a fault on one rank
        must not leave the host blocked in a synchronize on a peer that spins on it), and name the piece that faulted."""
        if torch.cuda.is_current_stream_capturing():
            return
        pend = []
        for r in (range(W) if ranks is None else ranks):
            with torch.cuda.device(self.devs[r]):
                for sx in (self.streams[r], self.side[r], self.estream[r]):
                    e = torch.cuda.Event()
                    e.record(sx)
                    pend.append((r, e))
        t0 = time.time()
        while pend:
            nxt = []
            for r, e in pend:
                try:
                    if not e.query():
                        nxt.append((r, e))
                except Exception as ex:  # noqa: BLE001
                    raise RuntimeError(f"ENG_SYNC_DEBUG: fault after [{what}] on rank {r}: {ex}") from ex
            pend = nxt
            if pend and time.time() - t0 > timeout:
                raise RuntimeError(f"ENG_SYNC_DEBUG: ranks {sorted({r for r, _ in pend})} busy {timeout:.0f} s after [{what}]")
            if pend:
                time.sleep(0.0005)
        self._dbg_last = what

    def _store_graph(self, key, gs) -> None:
        """self.graphs[key] = gs; a graph it replaces is retired (kept alive), never destroyed while others live."""
        old = self.graphs.get(key)
        if old is not None:
            self._retired.append(old)
            self.stats["graphs_retired"] = self.stats.get("graphs_retired", 0) + 1
        self.graphs[key] = gs

    @staticmethod
    def _sched_refs(st: list) -> list:
        return [(c.get("sched"), c.get("sched_dec")) for s in st for c in s["caches"].values()]

    def _capture(self, R: int, gkey=None) -> None:
        """Capture the bucket's graphs: gkey = (R, HSD inside or not); HSD is captured only when its bucket = this one."""
        gkey = gkey or (R, self.hooks.hsd_graph_ok(R))
        with_hsd = gkey[1]
        st = self.bucket(R)
        self._new_scheds(st)
        gs = []
        for r in range(W):    # one rank at a time (nothing executes during capture, so no peer waits block)
            with torch.cuda.device(self.devs[r]):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.gpool[r], stream=self.streams[r]):
                    s = st[r]
                    self.p_embed(r, s)
                    for L in self.layers:
                        self.p_attn_bd(r, s, L)
                        self.p_attn(r, s, L)
                        self.p_ffn_bd(r, s, L)
                        self.p_router(r, s, L)
                        self.p_ag(r, s, L, st)
                        self.p_moe(r, s, L)
                        self.p_rs(r, s, L, st)
                    self.p_final(r, s)
                    self._join_mix(s)
                    if with_hsd:
                        self.hooks.hsd_graph(r, s)
                gs.append(g)
        for d in self.devs:
            torch.cuda.synchronize(d)
        self._store_graph(gkey, gs)
        self.gkeep[gkey] = self._sched_refs(st)
        self.hsd_in_graph[gkey] = self.hooks.hsd_key() if with_hsd else None

    def replay(self, R, with_hsd: bool | None = None) -> None:
        """Replay a bucket's graphs; R = bucket rows (the variant of the last capture / step) or a (R, with_hsd) key."""
        key = R if isinstance(R, tuple) else ((R, with_hsd) if with_hsd is not None else self._last_gkey.get(R, (R, False)))
        gs = self.graphs[key]
        lt = []                           # per launch: host ms of graph.replay() (stats["replay_launch_ms"], diagnostics)
        pc = time.perf_counter
        if isinstance(gs[0], tuple):      # ENG_LAUNCH_SPLIT: piece p on every rank before piece p + 1 on any
            for p_ in range(len(gs[0])):
                for r in range(W):
                    with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                        t_ = pc()
                        gs[r][p_].replay()
                        lt.append(round((pc() - t_) * 1000, 3))
            self.stats["replay_launch_ms"] = lt
            return
        for r in range(W):
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                t_ = pc()
                gs[r].replay()
                lt.append(round((pc() - t_) * 1000, 3))
        self.stats["replay_launch_ms"] = lt

    def sync(self) -> None:
        for d in self.devs:
            torch.cuda.synchronize(d)

    def copies(self, batches: list) -> int:
        """Rank-to-rank page copies of this step, before any kernel (serve/worker.py Worker.copies)."""
        ops = {}
        for sb in batches:
            for c in sb.copies:
                ops[(c.pool, c.src_rank, c.src_page, c.dst_rank, c.dst_page)] = c
        if not ops:
            return 0
        WK = self.WK
        size = dict(paged=WK.PAGE_BYTES, window=WK.WINDOW_PAGE_BYTES, state=WK.STATE_SLOT_BYTES)
        for pool, sr, s0, dr, d0, n in copy_runs(list(ops.values())):
            src, dst, b = getattr(self.pools[sr], pool), getattr(self.pools[dr], pool), size[pool]
            with torch.cuda.device(self.devs[dr]), torch.cuda.stream(self.streams[dr]):
                dst[d0 * b:(d0 + n) * b].copy_(src[s0 * b:(s0 + n) * b], non_blocking=True)
        st_ops = [(c.src_rank, c.src_page, c.dst_rank, c.dst_page) for c in ops.values() if c.pool == "state"]
        if st_ops:                         # as the worker: Engram's per-slot ring follows the moved state slot
            self.copy_state_slots(st_ops)
        return len(ops)

    def prepare(self, batches: list, tables: list, R: int | None = None, copies: bool = False,
                moe: str = "raw", hooks: bool = True, role: str = "") -> list:
        """Host side of a step: bucket, tables (the worker's RankTables) -> device buffers of the bucket."""
        assert self.pools is not None, "bind_pools() first"
        n = max(sb.n_rows for sb in batches)
        assert n <= max(self.max_rows, self.cfg.max_mixed_rows), f"{n} rows per rank > max_rows"
        R = R or self.bucket_of(max(n, 1))
        st = self.bucket(R, moe, role)
        if copies:
            self.copies(batches)
        self._batches = batches
        tm = self.stats.setdefault("host_ms", {})
        t0 = time.perf_counter()
        t_idx = 0.0
        for r, sb in enumerate(batches):
            tab = tables[r]
            assert tab.n == sb.n_rows
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                self._upload(r, st[r], tab)
                if self.hooks.idx is not None:
                    ti = time.perf_counter()
                    if moe == "raw":
                        self.hooks.idx.upload(r, st[r], sb, tab)
                    else:
                        self.hooks.idx.upload_general(r, st[r], sb, tab)
                    t_idx += time.perf_counter() - ti
            st[r]["tab"], st[r]["batch"] = tab, sb
        t1 = time.perf_counter()
        self.hooks.prepare_engram(st, batches)
        t2 = time.perf_counter()
        if hooks:
            self.hooks.prepare_head(st, batches)
        t3 = time.perf_counter()
        for k, v in (("upload", t1 - t0 - t_idx), ("indexer", t_idx), ("engram", t2 - t1), ("head", t3 - t2)):
            tm[k] = round(v * 1000, 2)      # the last step's host times (ms), engine prepare only
        return st

    def step(self, batches: list, tables: list | None = None, ctx=None, copies: bool = False, tokens: list | None = None,
             sampling_state=None, graphs: bool | None = None, R: int | None = None, lookback=None) -> list:
        """serve contract D1: one step on the 4 ranks. batches[r] = rank r's StepBatch, tables[r] = its RankTables (the
        worker's TableBuilder; tests: None -> built here from `tokens`), ctx = StepContext (new sequences with prompt ids
        and sampling params, dropped rids, grammar, prev_tokens). The worker runs the page copies first (D2; tests:
        copies=True). Returns [StepResult per rank] when the results are on the host."""
        t_in = time.perf_counter()
        self.bump_serial(batches)
        use_g = self.cfg.graphs if graphs is None else graphs
        if tables is None:
            tables = self.tables_for(batches, tokens)
        if copies:
            self.copies(batches)
        self.hooks.on_ctx(ctx, sampling_state, lookback)
        self._mm = getattr(ctx, "mm", None) if ctx is not None else None
        ctx_max = max((max(sb.positions) for sb in batches if sb.n_rows), default=0) + 1
        lvl = 0
        if self.hooks.idx is not None and ctx_max > GRAPH_MAX_CTX:
            # a row past the lvl-0 graphs' indexer bound (idx2f fast mode: 262,144 positions). ENG_LONG_GRAPHS: the
            # bucket's long variant (graphed; idx2f's stock select / top-k on the layers past its fast limit, as eager),
            # with split HSD and > 64 rows in all (the stock persistent top-k's capture condition)
            lvl = next((i for i, c in enumerate(LONG_LEVEL_CTX) if ctx_max <= c), -1)
            ok = (LONG_GRAPHS and lvl > 0 and use_g and self.cfg.split_hsd and self.hooks.hsd is not None and not DBG_SYNC
                  and sum(sb.n_rows for sb in batches) > 64)
            if not ok:
                lvl = -1
        if any(se.kind != "decode" for sb in batches for se in sb.seqs) or \
                max(sb.n_rows for sb in batches) > self.cfg.buckets[-1] or lvl < 0:
            # prefill rows, more rows than the largest bucket, or a context beyond the decode graphs' indexer bounds
            # without a long variant (eager steps take idx2f's stock fallback for longer rows)
            return self.step_mixed(batches, tables)
        assert not (self._mm and any(m is not None and m[0].numel() for m in self._mm)), \
            "StepContext.mm on a decode-only step: image rows occur only in prefill rows"
        tq0 = time.perf_counter()
        EA.QB16D = QB16_DECODE
        if lvl:
            R = self.bucket_of(max(max(sb.n_rows for sb in batches), 1)) + lvl * LVL_TAG
            self.stats["long_steps"] = self.stats.get("long_steps", 0) + 1
            return self._step_split(batches, tables, R, t_in, tq0)
        if use_g and self.cfg.split_hsd and self.hooks.hsd is not None and not DBG_SYNC:
            return self._step_split(batches, tables, R, t_in, tq0)
        st = self.prepare(batches, tables, R)
        tq1 = time.perf_counter()
        R = st[0]["R"]
        gkey = (R, self.hooks.hsd_graph_ok(R))      # HSD inside the graph only when active this step with this bucket
        self._last_gkey[R] = gkey
        if use_g and gkey not in self.graphs:
            # first step of this bucket: run it eagerly (the real step: layers + HSD), then capture the bucket's graph
            # for the following steps (capture executes nothing; HSD's rule: capture after one eager step of a bucket)
            self._new_scheds(st)
            self._issue(st)
            self.hooks.head_eager(st)
            self.sync()
            res = [self.hooks.head_result(r, st[r]) for r in range(W)]
            self._capture(R, gkey)
            self.last_st = st
            self.stats["steps"] += 1
            self.stats["eager_steps"] += 1
            return res
        if use_g:
            in_g = gkey[1] and self.hsd_in_graph.get(gkey) == self.hooks.hsd_key()
            if DBG_SYNC:
                self._dbg_sync(f"decode prepare R{R}")
                self._dbg_devices(st, f"decode R{R}")
            evr = []
            for r in range(W):                       # GPU time of the replay (and of any eager head after it) per rank
                with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    evr.append((e0, e1))
            self.replay(gkey)
            if DBG_SYNC:
                self._dbg_sync(f"decode replay {gkey}")
            self.stats["graph_steps"] += 1
            tq2 = time.perf_counter()
            if not in_g:
                self.hooks.head_eager(st)    # HSD not inside this graph variant (no-op on a gate-only step)
            for r in range(W):
                with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                    evr[r][1].record()
            tq3 = time.perf_counter()
            self.last_st = st
            self.sync()
            tq4 = time.perf_counter()
            self.stats["steps"] += 1
            out = [self.hooks.head_result(r, st[r]) for r in range(W)]
            tq5 = time.perf_counter()
            if DBG_SYNC:
                self._dbg_sync(f"decode head_result R{R}")
            ms = lambda a_, b_: round((b_ - a_) * 1000, 3)  # noqa: E731
            self.stats["decode_last"] = dict(R=R, hsd_in_graph=bool(in_g), pre=ms(t_in, tq0), prepare=ms(tq0, tq1),
                                             issue=ms(tq1, tq2),
                                             head_eager_issue=ms(tq2, tq3), sync_wait=ms(tq3, tq4), results=ms(tq4, tq5),
                                             total=ms(t_in, tq5), gpu=round(max(a_.elapsed_time(b_) for a_, b_ in evr), 3))
            return out
        else:
            self._new_scheds(st)
            if DBG_SYNC:
                self._dbg_devices(st, f"eager decode R{R}")
            self._issue(st)
            self.stats["eager_steps"] += 1
            self.hooks.head_eager(st)
        self.last_st = st
        self.sync()
        self.stats["steps"] += 1
        out = [self.hooks.head_result(r, st[r]) for r in range(W)]
        if DBG_SYNC:
            self._dbg_sync(f"decode head_result R{R}")
        return out


    # ------------------------------------------------------------------------------------------------ tests: one engine, many arms
    def bump_serial(self, batches) -> None:
        """One engine step with sequences (PREV_STEP_CHECK): every step entry calls this once, before its uploads."""
        if any(sb.seqs for sb in batches):
            self.step_serial += 1

    def prev_of(self, rid):
        """(rank, seq index) of rid's row in hsd.next_ids if it ran in the immediately previous engine step, else None."""
        p = self.prev_seq.get(rid)
        if p is None:
            return None
        if PREV_STEP_CHECK and (len(p) < 3 or p[2] != self.step_serial - 1):
            return None
        return p[0], p[1]

    def reset_state(self) -> None:
        """Tests (one Engine per test process, A/B arms switched at runtime): drop every per-run state so the
        next run starts like a fresh engine; the caller re-initializes the pools' CONTENTS (synth.init_pools, seeded).
        Dropped / reset: graphs (and the graph pools, which die with their last graph), bucket states (their allocation
        follows config flags), D13 / CP state, host bookkeeping, the table builders' delta state, the hooks' per-slot
        device state (Engram token rings / hashes / dedup; HSD's per-slot store, next ids, emitted counts, sampler
        slot states, blob parity) and the indexer's per-slot block table. A same-arm A / A2 control validates it."""
        self.sync()
        if RESET_KEEP_GRAPHS:      # tests (graph-launch-cost probe): keep the old run's graphs alive (never replayed)
            self._retired.extend([*self.graphs.values(), *self.mgraphs.values()])
        else:
            self._retired.clear()  # everything dies together here (no graph of the old run is replayed again)
        for d in (self.graphs, self.gkeep, self.hsd_in_graph, self._last_gkey, self.mgraphs, self.bk, self.bk_bytes):
            d.clear()
        self._busy = set()
        self._d13_obj = None
        self._cp = None
        self.prev_seq.clear()
        self.hist.clear()
        self.tb = self.WK.TableBuilder(stub_indexer=self.wants_stub_indexer)
        self.tb_tail = self.WK.TableBuilder(stub_indexer=self.wants_stub_indexer)
        import gc
        gc.collect()
        if RESET_EMPTY_CACHE:
            torch.cuda.empty_cache()
        self.gpool = []
        for d in self.devs:
            with torch.cuda.device(d):
                self.gpool.append(torch.cuda.graph_pool_handle())
        H = self.hooks
        H.samp_ctx.clear()
        H.structured.clear()
        H.grammar = {}
        for e in (H.engram or []):
            e.ring.fill_(-1)
            e.hash.fill_(-1)
            e.keep.zero_()
            e.err.zero_()
            e.clear_dedup()
            if hasattr(e, "_start_src"):
                del e._start_src
        hs = H.hsd
        if hs is not None:
            for lst in (hs.by_slot, hs.next_ids, hs.num_emitted):
                for x in lst:
                    x.zero_()
            hs._temp_np[:] = 1.0
            hs._seed_np[:] = 0
            hs._slot_key[:] = [None] * len(hs._slot_key)
            hs._hi = 0
            hs._hev = [[None] * W for _ in range(2)]
            hs.cur = None
            for i in range(2):          # the host blobs' shared sections are rebuilt per prepare; zero for a clean start
                for b_ in hs._hblob[i]:
                    b_.zero_()
            for b_ in hs._dblob:
                b_.zero_()
        if H.idx is not None:
            for v in H.idx.views:
                v["dbt"].zero_()
        self.sync()
        self.stats = dict(steps=0, graph_steps=0, eager_steps=0)

    # ------------------------------------------------------------------------------------------------ D13 (engine/d13.py)
    @property
    def device_positions(self) -> bool:
        """serve's worker / the vLLM adapter: async scheduling can run on this engine (decode positions on the device)."""
        return bool(self.cfg.d13)

    def _d13(self):
        if getattr(self, "_d13_obj", None) is None:
            from engine.d13 import D13
            self._d13_obj = D13(self)
        return self._d13_obj

    def dispatch(self, batches: list, cts: list, ctx=None):
        """D13: queue one step from serve's CompactTables; returns a StepHandle (result() = [StepResult per rank]).
        Dispatching step N+1 waits for step N's results only after N+1 is queued on the GPU."""
        return self._d13().dispatch(batches, cts, ctx)

    def step_compact(self, batches: list, cts: list, ctx=None):
        """serve's worker contract: = dispatch (the handle is list-like)."""
        return self.dispatch(batches, cts, ctx)

    # ------------------------------------------------------------------------------------------------ split decode graphs
    def _hkey(self, R: int):
        """Graph H's key: (R, "H", HSD's bucket key) when HSD runs this step without a grammar bitmask, else None."""
        k = self.hooks.hsd_key()
        return None if k is None or self.hooks.bm_step else (R, "H", k)

    def _prepare_head_on_streams(self, st: list, batches: list) -> None:
        """HSD's host prepare with every device's current stream = the step stream: its one H2D copy per rank is then
        stream-ordered after the work already issued there (graph L) and before graph H (the default stream is not)."""
        import contextlib
        with contextlib.ExitStack() as es:
            for r in range(W):
                es.enter_context(torch.cuda.stream(self.streams[r]))
            self.hooks.prepare_head(st, batches)

    def _capture_split(self, R: int, st: list, layers: bool, hkey, lkey=None) -> None:
        """split_hsd: capture graph L (embed, the layers, final, join) of bucket R and / or graph H (HSD) for hkey."""
        if layers:
            self._new_scheds(st)
            gs = []
            cut = LAUNCH_SPLIT_AT if (LAUNCH_SPLIT and self.layers[0] < LAUNCH_SPLIT_AT <= self.layers[-1]
                                      and (self.hooks.engram is None or max(ENGRAM_LAYERS) < LAUNCH_SPLIT_AT)) else None

            def layer(r, s, L):
                self.p_attn_bd(r, s, L)
                self.p_attn(r, s, L)
                self.p_ffn_bd(r, s, L)
                self.p_router(r, s, L)
                self.p_ag(r, s, L, st)
                self.p_moe(r, s, L)
                self.p_rs(r, s, L, st)
            for r in range(W):   # one rank at a time (nothing executes during capture, so no peer waits block)
                with torch.cuda.device(self.devs[r]):
                    s = st[r]
                    if cut is None:
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, pool=self.gpool[r], stream=self.streams[r]):
                            self.p_embed(r, s)
                            for L in self.layers:
                                layer(r, s, L)
                            self.p_final(r, s)
                            self._join_mix(s)
                        gs.append(g)
                        continue
                    g0, g1 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g0, pool=self.gpool[r], stream=self.streams[r]):
                        self.p_embed(r, s)
                        for L in self.layers:
                            if L < cut:
                                layer(r, s, L)
                        self._join_mix(s)          # the cut: every side branch joins before piece 0 ends
                    with torch.cuda.graph(g1, pool=self.gpool[r], stream=self.streams[r]):
                        for L in self.layers:
                            if L >= cut:
                                layer(r, s, L)
                        self.p_final(r, s)
                        self._join_mix(s)
                    gs.append((g0, g1))
            if cut is not None:
                self.stats["launch_split_captures"] = self.stats.get("launch_split_captures", 0) + 1
            self._store_graph(lkey or (R, "L"), gs)
            self.gkeep[lkey or (R, "L")] = self._sched_refs(st)
        if hkey is not None and hkey in self.graphs:
            # D13's (R, "LA") and the sync path's (R, "L") share eng.bucket(R)'s state, so the existing graph H of this key
            # reads the same buffers: keep it (replacing it destroyed CUDAGraphs -> the step stream's cuBLAS workspace freed
            # under the live graphs). Long-row keys carry the level in R (R + lvl * LVL_TAG).
            hkey = None
            self.stats["h_graphs_kept"] = self.stats.get("h_graphs_kept", 0) + 1
        if hkey is not None:
            gs = []
            for r in range(W):
                with torch.cuda.device(self.devs[r]):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.gpool[r], stream=self.streams[r]):
                        self.hooks.hsd_graph(r, st[r])
                    gs.append(g)
            self._store_graph(hkey, gs)
        for d in self.devs:
            torch.cuda.synchronize(d)

    def _step_split(self, batches: list, tables: list, R, t_in: float, tq0: float) -> list:
        """split_hsd decode step: graph L of bucket R, then HSD's host prepare (overlapped with graph L on the GPU), then
        graph H of (R, HSD key) (eager HSD + capture on a key's first step). Same kernels in the same stream order as the
        one-graph step."""
        Hk = self.hooks
        st = self.prepare(batches, tables, R, hooks=False)
        tq1 = time.perf_counter()
        Rk = st[0]["bkey"]                 # = R, or R + lvl * LVL_TAG for a long-row variant (its own graphs)
        R = st[0]["R"]
        lk = (Rk, "L")
        self._last_gkey[Rk] = lk
        if lk not in self.graphs:
            # first step of this bucket: eager layers + HSD (the real step), then capture graph L and graph H
            self._new_scheds(st)
            self._issue(st)
            self._prepare_head_on_streams(st, batches)
            Hk.head_eager(st)
            self.sync()
            res = [Hk.head_result(r, st[r]) for r in range(W)]
            self._capture_split(Rk, st, True, self._hkey(Rk))
            self.last_st = st
            self.stats["steps"] += 1
            self.stats["eager_steps"] += 1
            return res
        evr = []
        for r in range(W):
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                evr.append((e0, e1))
        self.replay(lk)
        tq2 = time.perf_counter()
        launch_l = self.stats.get("replay_launch_ms")
        self._prepare_head_on_streams(st, batches)       # host work while graph L runs
        tq3 = time.perf_counter()
        hk = self._hkey(Rk)
        cap_h = None
        if hk is not None and hk in self.graphs:
            self.replay(hk)
            self.stats["graph_steps"] += 1
        else:
            Hk.head_eager(st)        # a new HSD key (captured after this step), a grammar step, or HSD off (no-op)
            cap_h = hk
        for r in range(W):
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                evr[r][1].record()
        tq4 = time.perf_counter()
        self.last_st = st
        self.sync()
        tq5 = time.perf_counter()
        self.stats["steps"] += 1
        out = [Hk.head_result(r, st[r]) for r in range(W)]
        tq6 = time.perf_counter()
        if cap_h is not None:
            self._capture_split(Rk, st, False, cap_h)
        ms = lambda a_, b_: round((b_ - a_) * 1000, 3)  # noqa: E731
        self.stats["decode_last"] = dict(R=R, lvl=st[0].get("lvl", 0), split=True, hsd_in_graph=hk is not None and cap_h is None,
                                         pre=ms(t_in, tq0), prepare=ms(tq0, tq1), issue=ms(tq1, tq2),
                                         head_prepare_overlapped=ms(tq2, tq3), issue_h=ms(tq3, tq4),
                                         sync_wait=ms(tq4, tq5), results=ms(tq5, tq6), total=ms(t_in, tq6),
                                         gpu=round(max(a_.elapsed_time(b_) for a_, b_ in evr), 3),
                                         launch_l=launch_l, n_graphs=len(self.graphs))
        return out

    # ------------------------------------------------------------------------------------------------ mixed steps
    def step_mixed(self, batches: list, tables: list) -> list:
        """A step with prefill chunks (and decode rows), eager. Layers first..20 run all rows (bucket A); layers 21..39 run
        StepBatch.tail_rows only (decode rows + the last min(n, 128) rows of each chunk: production's decoder-side bounded
        replay) on bucket T, whose input stream is gathered from A after layer 20; the head runs the sample rows (a subset
        of the tail rows). MoE: production's kernels on every layer of the step (prefill rows present)."""
        cfg = self.cfg
        tw0 = time.perf_counter()
        self._busy = set()
        evs = []
        for r in range(W):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                e0.record()
            evs.append((e0, e1))
        rnd = lambda n: max(cfg.mixed_round, -(-n // cfg.mixed_round) * cfg.mixed_round)  # noqa: E731
        nA = [sb.n_rows for sb in batches]
        assert max(nA) <= cfg.max_mixed_rows, f"{max(nA)} rows per rank > max_mixed_rows {cfg.max_mixed_rows}"
        RA = rnd(max(nA))
        segA = [L for L in self.layers if L <= 20]
        segT = [L for L in self.layers if L > 20]
        if POISON:   # every step: the reused mixed buckets' must-rewrite buffers (before this step's uploads)
            for s_ in self.bucket(RA, "prod", "A"):
                self._poison_state(s_)
        self._cp = None
        if cfg.cp and any(getattr(se, "cp", None) is not None for sb in batches for se in sb.seqs):
            from .cp import CPPlan
            self._cp = CPPlan(self, batches)
            self.stats["cp_steps"] = self.stats.get("cp_steps", 0) + 1
            from . import cp as CPM
            if CPM.AUDIT:                    # debug: the step's copies ran (engine.step copies=True / the worker): audit
                self._cp.audit("before")
            assert not cfg.mixed_graphs, "CP prefill steps are eager (mixed_graphs off)"
        rfA = torch.profiler.record_function("piece:host prepare A") if PROF_PIECES else None
        if rfA is not None:
            rfA.__enter__()
        stA = self.prepare(batches, tables, RA, moe="prod", hooks=False, role="A")
        for s_ in stA:
            s_["cp_on"] = self._cp is not None
        if rfA is not None:
            rfA.__exit__(None, None, None)
        for s_ in stA:
            s_["chain_input"] = False
        self._set_vidx(stA, nA, [dec_prefix(sb) for sb in batches])
        self._img_rows_A(stA, batches, tables)
        EA.QB16 = QB16_MIXED
        if PF_ACT8_DENSE_LOAD:
            EA._a8().set_enabled(PF_ACT8_DENSE)
        self._attn_pf_plans(stA, batches)
        self._new_scheds(stA, per_layer=BISECT["per_layer_plans"])
        for s_ in stA:
            s_.pop("_dbg_ref", None)
        if DBG_SYNC:
            self._dbg_sync(f"mixed prepare A R{RA} rows {nA}")
            self._dbg_devices(stA, f"mixed A R{RA}")
        l20 = (L20_TAIL and not cfg.mixed_graphs and not DBG_SYNC and self._cp is None and 20 in segA and bool(segT)
               and segA[-1] == 20)
        if cfg.mixed_graphs and not DBG_SYNC:
            self._issue_mixed(stA, segA, final=not segT, embed=True)
        elif l20:
            self._issue(stA, final=False, layers=segA[:-1])
            self._l20_head(stA)
            self.stats["l20_tail_steps"] = self.stats.get("l20_tail_steps", 0) + 1
        else:
            self._issue(stA, final=not segT, layers=segA)
        if not segT:
            self.last_st = stA
            self.sync()
            if self._cp is not None:
                self._cp.done()
            out = [self.hooks.head_result(r, stA[r]) for r in range(W)]
            self._cp = None
            return out
        rfT = torch.profiler.record_function("piece:host tail gather + prepare T") if PROF_PIECES else None
        if rfT is not None:
            rfT.__enter__()
        tb = [tail_batch(sb) for sb in batches]
        nT = [b.n_rows for b in tb]
        RT = rnd(max(nT))
        stT = self.bucket(RT, "prod", "T")
        if POISON:
            for s_ in stT:
                self._poison_state(s_)
        for r in range(W):
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                tab = self.tb_tail.build(tb[r], [int(tables[r].tokens[i]) for i in batches[r].tail_rows])
                self._upload(r, stT[r], tab)
                stT[r]["tab"], stT[r]["batch"] = tab, tb[r]
                self._img_rows_T(r, stT[r], tables[r], batches[r].tail_rows)
                if self.hooks.idx is not None:
                    self.hooks.idx.upload_general(r, stT[r], tb[r], tab)
                sA, sT = stA[r], stT[r]
                n = nT[r]
                self._join_mix(sA)
                c20 = sT["caches"].get(20)
                if l20:     # layer 20's state after its window insert; its attention reads the FULL window (A's lists)
                    sT.setdefault("si20", torch.empty_like(sT["si"]))
                    sT.setdefault("sl20", torch.empty_like(sT["sl"]))
                    sT["si20"].copy_(sT["si"])            # padding rows as the T tables have them
                    sT["sl20"].copy_(sT["sl"])
                    c20["si"], c20["sl"] = sT["si20"], sT["sl20"]
                elif c20 is not None:
                    c20["si"], c20["sl"] = sT["si"], sT["sl"]
                sT["l20"] = l20
                sT["ws"]["pf_ab"] = None
                if n:
                    ix = torch.tensor(batches[r].tail_rows, dtype=torch.int64).pin_memory().to(sA["dev"], non_blocking=True)
                    if l20:
                        for k in ("X", "a_pre", "a_post", "a_comb"):
                            sT[k][:n].copy_(sA[k].index_select(0, ix))
                        sT["ws"]["q"][:n].copy_(sA["ws"]["q"].index_select(0, ix))
                        sT["si20"][:n].copy_(sA["si"].index_select(0, ix))
                        sT["sl20"][:n].copy_(sA["sl"].index_select(0, ix))
                    else:
                        for k in ("moe_out", "xm2", "f_post", "f_comb", "f_pre"):
                            sT[k][:n].copy_(sA[k].index_select(0, ix))
                    for il in INDEX_LAYERS:          # index layers <= 20 computed on all rows feed layers 21..23
                        if il <= 20:
                            sT["ti"][il][:n].copy_(sA["ti"][il].index_select(0, ix))
                            sT["tl"][il][:n].copy_(sA["tl"][il].index_select(0, ix))
                    if self.hooks.idx is not None and 20 in segA:
                        self.hooks.idx.carry_candidates(sA, sT, batches[r].tail_rows, ix)
                for il in (INDEX_LAYERS if BISECT["clear_tail_padding"] else ()):   # padding rows: no keys (not stale)
                    if il <= 20:
                        sT["ti"][il][n:].fill_(-1)
                        sT["tl"][il][n:].zero_()
                sT["chain_input"] = True
        self._set_vidx(stT, nT, [dec_prefix(b) for b in tb])
        self.hooks.prepare_head(stT, batches)
        if rfT is not None:
            rfT.__exit__(None, None, None)
        self._new_scheds(stT, per_layer=BISECT["per_layer_plans"])
        for s_ in stT:
            s_.pop("_dbg_ref", None)
        if DBG_SYNC:
            self._dbg_sync(f"mixed tail gather + prepare T R{RT} rows {nT}")
            self._dbg_devices(stT, f"mixed T R{RT}")
        if cfg.mixed_graphs and not DBG_SYNC:
            self._issue_mixed(stT, segT, final=True, embed=False)
        else:
            if l20:
                self._l20_tail(stT)
            self._issue(stT, final=True, layers=segT, embed=False)
        self.hooks.head_eager(stT, batches)
        if DBG_SYNC:
            self._dbg_sync("mixed head")
        self.last_st = stT
        tw1 = time.perf_counter()
        for r in range(W):
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                evs[r][1].record()
        self.sync()
        tw2 = time.perf_counter()
        if self._cp is not None:
            self._cp.done()
        for s_ in stA:                       # ENG_ATTN_PF: the plans' workspaces (~1 KB per gathered record) go now
            s_["pf"] = None
        self._mem_after_step()
        self.stats["mixed_last"] = dict(gpu_ms=round(max(a.elapsed_time(b) for a, b in evs), 2),
                                        issue_ms=round((tw1 - tw0) * 1000, 2), wall_ms=round((tw2 - tw0) * 1000, 2))
        self.stats["steps"] += 1
        self.stats["mixed_steps"] = self.stats.get("mixed_steps", 0) + 1
        out = [self.hooks.head_result(r, stT[r], batches[r]) for r in range(W)]
        self._cp = None
        if DBG_SYNC:
            self._dbg_sync("mixed head_result (gate head / HSD results)")
        return out

    def _each(self, st: list, fn) -> None:
        for r in range(W):
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                fn(r, st[r])

    def _l20_head(self, st: list) -> None:
        """ENG_L20_TAIL, bucket A (all rows): layer 20's attention boundary, q / kv projections, indexer + compressor and
        q_b + window insert (the KV of every row); every piece for every rank before the next (as _issue)."""
        L = 20
        self._each(st, lambda r, s: self.p_attn_bd(r, s, L))
        self._each(st, lambda r, s: EA.attn_a(self.lw[r][L]["AW"], s["ws"], s["a"], s["pos"]))
        self._each(st, lambda r, s: self.hooks.indexer_run(r, s, L))
        self._each(st, lambda r, s: EA.attn_b_insert(self.lw[r][L]["AW"], s["caches"][L], s["ws"], s["pos"],
                                                     insert=self.cfg.insert))
        self._each(st, lambda r, s: self._join_mix(s))

    def _l20_tail(self, st: list) -> None:
        """ENG_L20_TAIL, bucket T (tail rows): layer 20's attention core (full window: caches[20] si / sl = A's rows) + wo,
        then its ffn boundary, router, AG, MoE, RS (the same pieces as _issue)."""
        L = 20
        self._each(st, lambda r, s: EA.attn_b_attend(self.lw[r][L]["AW"], s["caches"][L], s["ws"], s["pos"]))
        self._each(st, lambda r, s: self.p_ffn_bd(r, s, L))
        self._each(st, lambda r, s: self.p_router(r, s, L))
        self._each(st, lambda r, s: self.p_ag(r, s, L, st))
        self._each(st, lambda r, s: self.p_moe(r, s, L))
        self._each(st, lambda r, s: self.p_rs(r, s, L, st))

    def _attn_pf_plans(self, st: list, batches: list) -> None:
        """ENG_ATTN_PF: per rank, kernels/attn_dense/attn_prefill.Plan of the step's prefill sequences (row order; CP
        helper sub-chunks included) on bucket A's device tables (si / sl) and the sequences' page tables -> s["pf"]
        (None: no prefill rows on the rank, or the switch off). Released after the step (_pf_release)."""
        for r, (s, sb) in enumerate(zip(st, batches)):
            s["pf"] = None
            pfs_ = [se for se in sb.seqs if se.kind != "decode"]     # the bucket's prefill rows [a, b) (ENG_PF_ACT8_DENSE)
            s["ws"]["pf_ab"] = (pfs_[0].row0, pfs_[-1].row0 + pfs_[-1].n_rows) if pfs_ else None
        if not ATTN_PF:
            return
        assert not self.cfg.mixed_graphs, "ENG_ATTN_PF: eager mixed steps only"
        AP = EA._ap()
        for r, (s, sb) in enumerate(zip(st, batches)):
            pfs = [se for se in sb.seqs if se.kind != "decode"]
            if not pfs:
                continue
            nbs = [-(-(se.start + se.n_rows) // 64) for se in pfs]
            bt = torch.zeros(len(pfs), max(nbs), dtype=torch.int32)
            for i, (se, nb) in enumerate(zip(pfs, nbs)):
                bt[i, :nb] = torch.as_tensor(se.block_table[:nb], dtype=torch.int32)
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                s["pf"] = AP.Plan(self.devs[r], [(se.row0, se.n_rows, se.start, se.replay_start) for se in pfs],
                                  s["si"], s["sl"], bt.pin_memory().to(self.devs[r], non_blocking=True),
                                  self.pools[r].n_paged)
        self.stats["attn_pf_steps"] = self.stats.get("attn_pf_steps", 0) + 1

    def _mem_after_step(self) -> None:
        """After a (synchronized) mixed step: per-device memory stats of the step (allocated, cached, step peak), and the
        safety net: if more than FREE_CACHE_BYTES sit cached but unused on any device, return them to the driver (the
        caching allocator otherwise hoards blocks of sizes the next steps no longer request, and allocations outside
        torch, such as lazily loaded kernel modules, then fail)."""
        mem, idle = [], 0
        for d in self.devs:
            a_, r_, p_ = torch.cuda.memory_allocated(d), torch.cuda.memory_reserved(d), torch.cuda.max_memory_allocated(d)
            mem.append((round(a_ / 1e9, 2), round(r_ / 1e9, 2), round(p_ / 1e9, 2)))
            idle = max(idle, r_ - a_)
            torch.cuda.reset_peak_memory_stats(d)
        self.stats["mem_last"] = mem          # per device: (allocated, reserved, step peak allocated) GB
        if idle > FREE_CACHE_BYTES and BISECT["free_cache"]:
            torch.cuda.empty_cache()
            self.stats["cache_releases"] = self.stats.get("cache_releases", 0) + 1

    def _set_vidx(self, st: list, ns: list, nds: list | None = None) -> None:
        R = st[0]["R"]
        v = torch.cat([torch.arange(n, dtype=torch.int64) + r * R for r, n in enumerate(ns)])
        nv = torch.tensor(list(ns), dtype=torch.int32)
        act8 = PF_ACT8 and nds is not None and min(nds) >= 0     # (PF_ACT8: decode rows per rank, a prefix of its rows)
        if PF_ACT8 and nds is not None and not act8:
            if not self.stats.get("act8_fallback_steps"):
                print(f"engine: ENG_PF_ACT8: decode rows not a prefix of a rank's rows ({nds}): this step's MoE stays W4A16 "
                      f"(counted in stats act8_fallback_steps)", flush=True)
            self.stats["act8_fallback_steps"] = self.stats.get("act8_fallback_steps", 0) + 1
        nd = torch.tensor(list(nds), dtype=torch.int32) if act8 else None
        for r in range(W):
            st[r]["vidx"] = v.to(self.devs[r], non_blocking=True)
            st[r]["nv"] = nv.to(self.devs[r], non_blocking=True)   # valid rows per rank (pm_fused row map)
            st[r]["nd"] = nd.to(self.devs[r], non_blocking=True) if act8 else None
            st[r]["ndec"] = int(sum(nds)) if act8 else -1
            st[r]["vcount"] = int(v.numel())         # MoE dispatch: fork decode MoE <= prefill_moe_switch rows


def dec_prefix(sb) -> int:
    """PF_ACT8: decode rows of sb if they are a prefix of its rows (serve build_step_batch: decode rows first), else -1."""
    nd = sum(se.n_rows for se in sb.seqs if se.kind == "decode")
    return nd if all(se.row0 + se.n_rows <= nd for se in sb.seqs if se.kind == "decode") else -1


def tail_batch(sb):
    """The StepBatch of sb's tail rows (layers 21..39): rows = sb.tail_rows in order, sequences cut to their tail rows,
    sample rows remapped (every sample row is a tail row)."""
    import dataclasses
    from interfaces import StepBatch
    tb = StepBatch(sb.rank, sb.step, 0)
    tail = list(sb.tail_rows)
    new_of = {row: i for i, row in enumerate(tail)}
    for se in sb.seqs:
        rows = [x for x in range(se.row0, se.row0 + se.n_rows) if x in new_of]
        if not rows:
            continue
        ne = dataclasses.replace(se, row0=new_of[rows[0]], n_rows=len(rows), start=sb.positions[rows[0]])
        if TAIL_BOUND and se.kind != "decode" and not se.score and len(rows) < se.n_rows:
            # production's decoder-side bounded replay (vllm deepseek_v41 model_state._gather_replay_batch_kernel): a
            # trimmed request holds no layer-21..39 window KV below its kept rows, kv_start = max(replay_start,
            # seq_len - window); without it the first 127 tail rows read layer-21..39 window keys of chunk positions
            # that never ran those layers (stale pool bytes)
            ne.replay_start = max(se.replay_start, se.start + se.n_rows - SLIDING_WINDOW)
        ne.tail_row0 = ne.row0
        tb.seqs.append(ne)
        slot = len(tb.seqs) - 1
        for x in rows:
            tb.positions.append(sb.positions[x])
            tb.seq_slot.append(slot)
            tb.paged_slot.append(sb.paged_slot[x])
            tb.window_slot.append(sb.window_slot[x])
            tb.tail_rows.append(new_of[x])
    tb.sample_rows = [new_of[x] for x in sb.sample_rows]
    tb.n_rows = len(tail)
    tb.copies = []
    return tb


class Hooks:
    """The pieces outside the layer program, behind one switch each (EngineConfig.indexer / engram / head). Placeholders
    are clearly marked. Weights load in Engine.__init__ (counted before free_bytes); pool-dependent state attaches at
    bind_pools (attach)."""

    def __init__(self, eng: Engine):
        self.eng = eng
        cfg = eng.cfg
        assert cfg.indexer in ("stub", "idx2f"), f"indexer {cfg.indexer!r}"
        assert cfg.engram in ("off", "phys"), f"engram {cfg.engram!r}"
        assert cfg.head in ("stub", "hsd"), f"head {cfg.head!r}"
        self.engram = None
        self.idx = None
        self.hsd = None
        self.sampling, self.lookback, self.samp_ctx, self.prev_tokens = None, None, {}, {}
        self.structured, self.grammar = set(), {}      # structured-output rids (SeqInit.structured / ctx.grammar)
        self.grammar, self.bm_step = {}, False
        if cfg.engram == "phys":
            sys.path.insert(0, os.path.join(PHYS, "kernels/engram"))
            import engram as EG
            self.EG = EG
            assert eng.layers[0] <= 1 and eng.layers[-1] >= 14, "engram: the run must contain layers 1 and 14"
            t0 = time.time()
            reg = getattr(eng, "eg_reg", None) or EG.open_tables_async()   # started in Engine.__init__ (behind the weights)
            tabs = reg.wait()         # registers here: portable + mapped, one registration serves the 4 devices
            wts = EG.cached_weights(cfg.model)
            Rmax = max(max(cfg.buckets), -(-cfg.max_mixed_rows // cfg.mixed_round) * cfg.mixed_round)
            self.engram, self.ids32 = [], []
            for r, dev in enumerate(eng.devs):
                with torch.cuda.device(dev):
                    e = EG.EngramPhys(dev, max_rows=Rmax, max_seqs=cfg.max_seqs + 32,
                                      n_state_slots=Geometry().state_pool_slots, model_dir=cfg.model, tables=tabs, weights=wts)
                    ids = torch.zeros(Rmax, dtype=torch.int32, device=dev)   # the step's input ids (int32), every bucket
                    e.bind(ids=ids)
                    self.engram.append(e)
                    self.ids32.append(ids)
            eng.log(f"engine: Engram on {W} GPUs in {time.time() - t0:.1f} s (tables: populate {reg.populate_s:.1f} s in the "
                    f"background, waited {reg.waited:.1f} s for it; registration {reg.register_s:.1f} s)")
        if cfg.head == "hsd":
            sys.path.insert(0, os.path.join(PHYS, "kernels/head"))
            import hsd as HSDm
            t0 = time.time()
            # private drafter rings for now; attach() re-points the drafter KV to the window pool's layers 40..42
            assert all(b % 4 == 0 for b in cfg.buckets), "decode buckets = 4 x HSD's sequence buckets (k = 3)"
            # HSD's per-sequence state is indexed by serve's state slot (KVManager's state pool, lowest-first). With CP
            # prefill the helper ranks' shadows hold slots too, so a seated sequence's slot can pass max_seqs (80):
            # "state_slot >= HSD max_seqs". Slots = the state pool's (as Engram's ring);
            # shadows never have sample rows (helper chunks are never a last chunk), so they never reach HSD.
            self.hsd_slots = max(cfg.max_seqs, Geometry().state_pool_slots)
            self.hsd = HSDm.HSD(eng.devs, eng.reader, eng.mcfg, window_pools=None, max_seqs=self.hsd_slots, k=3,
                                draft_head=cfg.draft_head, mode="peer",
                                seq_buckets=tuple(b // 4 for b in cfg.buckets),   # (default buckets: HSD's own default)
                                # tail rows of a mixed step reach the whole bucket (score rows run 21..39 on every
                                # row; 2,048 / 3,579 seen, > 1,536): size HSD's context rows for it
                                max_tail_rows=max(max(cfg.buckets), 1536,
                                                  -(-cfg.max_mixed_rows // cfg.mixed_round) * cfg.mixed_round),
                                prep=cfg.hsd_prep, fuse_rs=cfg.hsd_fold, mhc2=cfg.hsd_mhc2,
                                moe_mode=cfg.hsd_moe)
            eng.log(f"engine: HSD (head + sampler + DSpark drafter) on {W} GPUs in {time.time() - t0:.0f} s")
        self.gate = None
        self.hsd_on = True
        if cfg.gate_head:
            from .gatehead import GateHead
            self.gate = GateHead(eng)
        if cfg.indexer == "idx2f":
            from .hooks_idx import Idx2f
            assert all(L in eng.layers for L in INDEX_LAYERS if L <= eng.layers[-1]) and eng.layers[0] <= 2, \
                "idx2f: the run must start at or before layer 2 (every index layer of the run is computed)"
            self.idx = Idx2f(eng, mode=cfg.indexer_mode)

    def hsd_apply(self, cfg) -> None:
        """Tests (per arm, after reset_state): the HSD drafter's runtime switches hsd_fold / hsd_mhc2 on every DraftRank
        (read per call; graphs are recaptured per arm). hsd_moe is load-time (EngineConfig at init)."""
        hs = getattr(self, "hsd", None)
        if hs is None:
            return
        for d in [*hs.ranks0, *hs.drk.values()]:
            d.fuse_rs, d.mhc2 = bool(cfg.hsd_fold), bool(cfg.hsd_mhc2)

    def attach(self, pools: list) -> None:
        """bind_pools: pool views of the hooks (indexer pages / rings, HSD drafter KV in window layers 40..42)."""
        from geometry import WINDOW_PAGE_BYTES
        if self.idx is not None:
            self.idx.attach(pools)
        if self.engram is not None:
            assert pools[0].n_state <= self.engram[0].ring.shape[0], "Engram ring rows < state pool slots"
        if self.hsd is not None:
            h = self.hsd
            for r, d in enumerate(h.ranks0):
                d.page_stride = WINDOW_PAGE_BYTES
                d.cache = [pools[r].window_base(40 + i) for i in range(3)]
                d.k_cache = [pools[r].window_view(40 + i) for i in range(3)]
            for (r, _), d in h.drk.items():
                d.page_stride, d.cache, d.k_cache = h.ranks0[r].page_stride, h.ranks0[r].cache, h.ranks0[r].k_cache
            h.pool = True

    # -- request lifecycle (serve D11) --------------------------------------------------------------------------------
    def on_ctx(self, ctx, sampling_state=None, lookback=None) -> None:
        eng = self.eng
        self.sampling, self.lookback = sampling_state, lookback
        self.grammar = {}
        if ctx is None:
            return
        from types import SimpleNamespace as NS
        for si in getattr(ctx, "new", []) or []:
            eng.hist[si.rid] = list(si.prompt_ids) + list(si.output_ids)
            sp = si.sampling or {}
            seed = sp.get("seed")
            if seed is None:   # vLLM (v1/worker/gpu/sample/states.py:56): one random int64 per request at admission
                import numpy as np
                seed = int(np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max))
            self.samp_ctx[si.rid] = NS(temperature=float(sp.get("temperature", 1.0)), seed=int(seed),
                                       drafts_valid=False)
            eng.prev_seq.pop(si.rid, None)
            if getattr(si, "structured", False):
                self.structured.add(si.rid)
        for rid in getattr(ctx, "dropped", set()) or set():
            eng.hist.pop(rid, None)
            eng.prev_seq.pop(rid, None)
            self.samp_ctx.pop(rid, None)
            self.structured.discard(rid)
        self.prev_tokens = dict(getattr(ctx, "prev_tokens", {}) or {})
        self.grammar = dict(getattr(ctx, "grammar", {}) or {})      # rid -> (rows int32 [k + 1, ceil(V / 32)], acceptable)

    def _lookback(self, rid, start):
        cp = getattr(self.eng, "_cp", None)
        if cp:                                   # CP helpers' shadow sequences: the prompt's ids (cp.group = its rid)
            rid = cp.alias.get(rid, rid)
        if self.lookback is not None:
            return self.lookback(rid, start)
        h = self.eng.hist.get(rid)
        if h is not None and start <= len(h):
            return [h[start - 1 - c] if start - 1 - c >= 0 else -1 for c in range(3)]
        pt = self.prev_tokens.get(rid)
        if pt:
            pt = list(pt)[-3:][::-1]
            return pt + [-1] * (3 - len(pt))
        import zlib                       # tests without ids: a stable stand-in (hash() is salted per process)
        return [(zlib.crc32(f"{rid}:{start - 1 - c}".encode()) % 120_000) + 1000 for c in range(3)]

    # -- indexer / compressor -------------------------------------------------------------------------------------
    def host_topk(self, r: int, s: dict, tab):
        """PLACEHOLDER (indexer 'stub'): per index layer the top-512 slots on the host = the most recent 512 compressed
        positions of the row's own sequence (TableBuilder rule, its KV source's numbering); padded rows -1 / length 0."""
        R, n = s["R"], tab.n
        ti, tl = {}, {}
        if self.idx is not None:     # computed on the device every step (indexer_run)
            return ti, tl
        for il in INDEX_LAYERS:
            src = kv_source_of(il)
            t = torch.full((R, 512), -1, dtype=torch.int32)
            l_ = torch.zeros(R, dtype=torch.int32)
            if n and src in tab.ti:
                t[:n], l_[:n] = tab.ti[src], tab.tl[src]
            ti[il], tl[il] = t, l_
        return ti, tl

    def indexer_run(self, r: int, s: dict, L: int) -> None:
        """indexer 'idx2f': at L in INDEX_LAYERS, from s['a'] (attention input), s['ws']['qr'], s['pos'] and the pools ->
        s['ti'][L], s['tl'][L]; kv sources also write comp / idx records at the rows' paged slots (hooks_idx.py).
        PLACEHOLDER (indexer 'stub'): nothing on the device, no compressor writes (the pools keep what a test wrote)."""
        if self.idx is not None and L in INDEX_LAYERS:
            self.idx.run(r, s, L)

    # -- Engram + head ----------------------------------------------------------------------------------------------
    def prepare(self, st: list, batches: list) -> None:
        self.prepare_engram(st, batches)
        self.prepare_head(st, batches)

    def prepare_engram(self, st: list, batches: list) -> None:
        if self.engram is not None:
            for r, sb in enumerate(batches):
                s = st[r]
                mk = getattr(self.EG, "step_args_from_tables", None) if BISECT["engram_args"] == "tables" else None
                a = (mk(s["tab"], self._lookback, pad_to=s["R"]) if mk is not None and getattr(s.get("tab"), "seq", None)
                     else self.EG.step_args_from_batch(sb, self._lookback, pad_to=s["R"]))
                e = self.engram[r]             # its hash reads ids32 (bound once): the step's device input ids
                with torch.cuda.device(s["dev"]):
                    if BISECT["engram_stream"]:   # the step's stream: Engram's non-blocking upload is then ordered
                        e.upload(a, stream=self.eng.streams[r])     # before the step (the default stream is not)
                    else:
                        e.upload(a)

    def prepare_head(self, st: list, batches: list) -> None:
        self.hsd_on = self.hsd is not None
        if self.hsd_on and self.gate is not None:   # a step with only gate (forced / score) sequences skips HSD
            self.hsd_on = not self.gate.any_gate([s_["tab"] for s_ in st])[1]
        if self.hsd_on:
            from types import SimpleNamespace as NS
            smp = self.sampling or {}
            states = {}
            n_dec = n_resume = n_host = 0
            for r_, sb in enumerate(batches):
                tab_ = st[r_].get("tab") if r_ < len(st) else None
                tl_ = tab_.tokens.tolist() if (tab_ is not None and getattr(tab_, "n", 0)) else None
                for se in sb.seqs:
                    st_ = smp.get(se.rid) or self.samp_ctx.get(se.rid) or NS(temperature=1.0, seed=0, drafts_valid=False)
                    if smp.get(se.rid) is None:
                        # drafts to verify: HSD's own from the previous engine step (device ids), or drafts the host gave
                        # explicitly (every row id >= 0: vLLM's grammar-validated drafts of a structured request resuming
                        # after an ASYNC skip, forced rows). A sequence resuming with host t0 + -1 drafts has none.
                        dev_ok = self.eng.prev_of(se.rid) is not None
                        host_ok = False
                        if not dev_ok and se.kind == "decode" and se.n_rows > 1 and tl_ is not None \
                                and se.row0 + se.n_rows <= len(tl_):
                            host_ok = min(tl_[se.row0:se.row0 + se.n_rows]) >= 0
                        st_.drafts_valid = dev_ok or host_ok
                        if se.kind == "decode":
                            n_dec += 1
                            n_resume += not dev_ok
                            n_host += host_ok
                    g = self.grammar.get(se.rid)
                    st_.num_acceptable_drafts = None if g is None else int(g[1])
                    states[se.rid] = st_
            es = self.eng.stats                  # (PREV_STEP_CHECK) decode sequences without the previous step's device ids
            es["resume_last"] = (n_resume, n_host, n_dec)
            es["resume_seqs"] = es.get("resume_seqs", 0) + n_resume
            es["resume_host_drafts"] = es.get("resume_host_drafts", 0) + n_host
            es["decode_seqs_seen"] = es.get("decode_seqs_seen", 0) + n_dec
            bms = self._bitmasks(batches) if self.grammar else None
            self.bm_step = bms is not None
            self.hsd.prepare(batches, states, bms)

    def _bitmasks(self, batches):
        """ctx.grammar -> HSD's per-rank [n_s, ceil(V / 32)] int32 masks in sample_rows order (-1 = all allowed; xgrammar
        layout). A sequence's rows[i] masks its i-th sample row (decode: the k + 1 verify rows; final prefill: 1 row)."""
        import numpy as np
        out, anyg = [], False
        for sb in batches:
            sr = list(sb.sample_rows)
            if not sr:
                out.append(None)
                continue
            words = None
            pos_in = {row: i for i, row in enumerate(sr)}
            bm = None
            for se in sb.seqs:
                g = self.grammar.get(se.rid)
                if g is None:
                    continue
                rows = np.asarray(g[0], dtype=np.int32)
                if bm is None:
                    words = rows.shape[-1]
                    bm = np.full((len(sr), words), -1, dtype=np.int32)
                mine = [x for x in range(se.row0, se.row0 + se.n_rows) if x in pos_in]
                for j, x in enumerate(mine[: rows.shape[0]]):
                    bm[pos_in[x]] = rows[j]
            if bm is not None:
                anyg = True
            out.append(None if bm is None else torch.from_numpy(bm))
        return out if anyg else None

    def device_tokens(self, r: int, s: dict, tab) -> None:
        """Decode rows of a sequence that HSD ran last step read their input ids from hsd.next_ids on the device
        ([last emitted, d_0 .. d_{k-1}] per sequence, the previous batch's seq order); others keep the host ids."""
        if self.hsd is None or not tab.n:
            return
        eng = self.eng
        if DEVTOK_V2:
            return self._device_tokens_v2(r, s, tab)
        rows, src = [], []
        for rid, row0, nr, kind, last in tab.seq_rows:
            prv = eng.prev_of(rid)
            if kind != "decode" or prv is None:
                continue
            if bool((tab.tokens[row0:row0 + nr] >= 0).all()):   # forced / host-given ids win (serve D6, D11)
                continue
            r0, si0 = prv
            if r0 != r:            # migrated: one small cross-device copy (outside any graph)
                s["tok"][row0:row0 + nr].copy_(self.hsd.next_ids[r0][si0, :nr].to(s["dev"]))
                continue
            rows += list(range(row0, row0 + nr))
            src += [si0 * 4 + j for j in range(nr)]
        if rows:
            ri = torch.tensor(rows, dtype=torch.int64).to(s["dev"], non_blocking=True)
            si = torch.tensor(src, dtype=torch.int64).to(s["dev"], non_blocking=True)
            s["tok"].index_copy_(0, ri, self.hsd.next_ids[r].view(-1).index_select(0, si))

    def _device_tokens_v2(self, r: int, s: dict, tab) -> None:
        """device_tokens with the same result: one tolist() of the host ids instead of a numpy reduction per sequence,
        and the (row, next_ids index) pairs in one pinned [2, n] buffer -> one H2D copy instead of two pageable ones
        (double-buffered; a set is refilled only after its copy from two steps ago completed)."""
        eng, hsd = self.eng, self.hsd
        tl_ = tab.tokens.tolist()
        k1 = hsd.next_ids[r].shape[1]
        rows, src = [], []
        for rid, row0, nr, kind, last in tab.seq_rows:
            if kind != "decode":
                continue
            prv = eng.prev_of(rid)
            if prv is None or min(tl_[row0:row0 + nr]) >= 0:   # forced / host-given ids win (serve D6, D11)
                continue
            r0, si0 = prv
            if r0 != r:            # migrated: one small cross-device copy (outside any graph)
                s["tok"][row0:row0 + nr].copy_(hsd.next_ids[r0][si0, :nr].to(s["dev"]))
                continue
            rows.extend(range(row0, row0 + nr))
            b = si0 * k1
            src.extend(range(b, b + nr))
        n = len(rows)
        if not n:
            return
        if not hasattr(self, "_dt"):
            nmax = max(4096, n)
            self._dt = [[(torch.zeros(2, nmax, dtype=torch.int64, pin_memory=True), None) for _ in range(2)]
                        for _ in range(len(eng.devs))]
            self._dt_d = [torch.zeros(2, nmax, dtype=torch.int64, device=d) for d in eng.devs]
            self._dt_i = [0] * len(eng.devs)
        i = self._dt_i[r]
        hb, ev = self._dt[r][i]
        if ev is not None:
            ev.synchronize()
        if hb.shape[1] < n:  # grow (rare)
            hb = torch.zeros(2, n, dtype=torch.int64, pin_memory=True)
            self._dt_d[r] = torch.zeros(2, n, dtype=torch.int64, device=s["dev"])
        h = hb.numpy()
        h[0, :n] = rows
        h[1, :n] = src
        d = self._dt_d[r]
        d[:, :n].copy_(hb[:, :n], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(s["dev"]))   # the stream the copy went on (destination device)
        self._dt[r][i] = (hb, ev)
        self._dt_i[r] = i ^ 1
        s["tok"].index_copy_(0, d[0, :n], hsd.next_ids[r].view(-1).index_select(0, d[1, :n]))

    def head_eager(self, st: list, batches: list | None = None) -> None:
        """HSD on the step's last bucket: h_rows = the final hidden of the sample rows (sample_rows order), aux_rows = the
        bucket's tail rows (decode: every row; mixed: the tail bucket IS the tail rows, in batch.tail_rows order)."""
        if not self.hsd_on:
            return
        hs, auxs = [], []
        for r in range(W):
            s = st[r]
            with torch.cuda.device(s["dev"]):
                h = s["h"]
                if batches is not None:    # mixed: the tail bucket's rows; sample rows remapped by tail_batch
                    sr = s["batch"].sample_rows
                    h = s["h"].index_select(0, torch.tensor(sr or [0], dtype=torch.int64).to(s["dev"]))
                hs.append(h)
                auxs.append(s["aux"])
        # eager: HSD's stage-interleaved run_all (every collective issued for all ranks before any rank waits; per-rank
        # run() is for graph capture: a host sync inside one rank's run would deadlock the others' peer waits)
        self.hsd.run_all(hs, auxs)

    def hsd_key(self):
        c = getattr(self.hsd, "cur", None) if self.hsd is not None else None
        return None if c is None or not self.hsd_on else (c.S_pad, c.D_pad, getattr(c, "NT", None))

    def hsd_graph_ok(self, R: int) -> bool:
        """HSD inside the decode graph of bucket R: HSD active this step, its bucket = the engine's (S_pad == R), no
        grammar bitmask variant."""
        k = self.hsd_key()
        return k is not None and k[0] == R and not self.bm_step     # a grammar step uses HSD's bitmask variant: eager

    def hsd_graph(self, r: int, s: dict) -> None:
        """Inside rank r's capture: the bucket's final hidden / aux rows into HSD's input buffers, then HSD's step."""
        R = s["R"]
        self.hsd.h_in(r)[:R].copy_(s["h"])
        self.hsd.aux_in(r)[:R].copy_(s["aux"].view(R, -1))
        self.hsd.run(r, None, None)

    def head_run(self, r: int, s: dict) -> None:
        """In-graph head hook (HSD runs eagerly after the graph today, head_eager)."""
        return

    def head_result(self, r: int, s: dict, batch=None, async_: bool = False) -> StepResult:
        if NANCHECK and self.engram is not None:   # Engram's error counters (ring misses / bad ids / long ring reads)
            ev = self.engram[r].err.tolist()
            seen = self.__dict__.setdefault("_eg_err_seen", {}).get(r, [0, 0, 0])
            if ev != seen:
                sbx = s.get("batch")
                self.eng.log(f"engine NANCHECK step {getattr(sbx, 'step', '?')} rank {r}: Engram errors (ring misses, ids "
                             f"outside the map, ring reads of > 8-row seqs) {seen} -> {ev}")
                self._eg_err_seen[r] = ev
        if NANCHECK:                       # every step type (decode graphs too): the head input of the sample rows
            sb_ = s.get("batch")           # the bucket's own batch (mixed steps: the T bucket's tail batch, whose row
            sr_ = list(getattr(sb_, "sample_rows", []) or [])    # order s["h"] follows; `batch` is bucket A's)
            if sr_:
                hh = s["h"].index_select(0, torch.tensor(sr_, device=s["dev"])).float()
                bad = (~torch.isfinite(hh).all(1)).nonzero().flatten().tolist()
                if bad:
                    rids = [str(sb_.seqs[sb_.seq_slot[sr_[i]]].rid) for i in bad[:4]]
                    self.eng.log(f"engine NANCHECK step {sb_.step} rank {r}: head input non-finite on {len(bad)} / {len(sr_)} "
                                 f"sample rows, e.g. {rids} (kinds {[sb_.seqs[sb_.seq_slot[sr_[i]]].kind for i in bad[:4]]})")
        if self.hsd is not None and self.hsd_on:
            res = self.hsd.result(r)
            seqs = (batch or s["batch"]).seqs
            if not async_:     # (D13 records the next-id rows at dispatch: the next step is dispatched before this)
                for si, se in enumerate(seqs):   # next_ids rows = the HSD batch's seq order
                    cp = getattr(se, "cp", None)
                    if cp is not None and getattr(cp, "owner", r) != r:
                        continue     # CP helper: a shadow under the prompt's rid; only the owner's row is the prompt's
                    self.eng.prev_seq[se.rid] = (r, si, self.eng.step_serial)
            # serve D7 / D11: structured-output sequences return their next k drafts (vLLM grammar-validates them and
            # sends them back as the next step's spec rows + bitmask); without them it can only send -1 drafts and
            # accepts 1 token per step (measured: 46 vs 123 tok/s)
            want = [(si, se.rid) for si, se in enumerate(seqs)
                    if (se.rid in self.structured or se.rid in self.grammar) and se.rid in res.tokens]
            if want:
                nid = self.hsd.next_ids[r]
                k = nid.shape[1] - 1
                rows = nid.index_select(0, torch.tensor([si for si, _ in want], device=nid.device))[:, 1:k + 1].cpu().tolist()
                for (_, rid), d in zip(want, rows):
                    res.drafts[rid] = [int(t) for t in d]
        elif self.hsd is not None:
            res = StepResult(r, s["batch"].step)                    # gate-only step: GateHead.fill fills it
        else:
            res = self.eng.WK.head_stub(s["tab"], r, s["batch"].step, 4)   # PLACEHOLDER (head 'stub')
        if self.gate is not None:
            with torch.cuda.device(s["dev"]):
                self.gate.fill(r, s, s["tab"], res)
        for rid, toks in res.tokens.items():          # host token history (Engram lookback of later prefill chunks)
            if rid in self.eng.hist:
                self.eng.hist[rid].extend(toks)
        return res


def _install_agrs() -> None:
    """The AG / RS options of kernels/agrs/agrs_engine.py (read-only import): wraps p_router /
    p_ag / p_moe / p_moe_prod / p_rs. Module flags AGRS_RSF (RS push fused into production's MoE combine), AGRS_SUM528,
    AGRS_AG32, AGRS_AGCE (AG by copy engines), defaults from ENG_AGRS_* (all 0): with every flag off each wrapper calls
    the original. Only buckets with R >= 1,024 rows per rank (eager mixed steps); decode graphs / capture untouched.
    NOTE: AGRS_RSF takes production MoE calls through its fused-RS layer, so ENG_PF_ACT8's layer8 does not run under it."""
    try:
        p = os.path.join(PHYS, "kernels/agrs")
        if p not in sys.path:
            sys.path.insert(0, p)
        import agrs_engine
        agrs_engine.install(sys.modules[__name__])
    except Exception as ex:  # noqa: BLE001
        print(f"engine: kernels/agrs not installed ({type(ex).__name__}: {ex})", flush=True)
    try:   # decode RS fused into the fork MoE's combine (AGRS_RSD, ENG_AGRS_RSD, default 0; decode buckets with
        # R >= 128 rows per rank; bitwise in a 256-seq engine A/B)
        import agrs_decode
        agrs_decode.install(sys.modules[__name__])
    except Exception as ex:  # noqa: BLE001
        print(f"engine: kernels/agrs agrs_decode not installed ({type(ex).__name__}: {ex})", flush=True)


_install_agrs()
