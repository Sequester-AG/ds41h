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
import threading
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
# DeepGEMM's first use is NOT thread-safe: vLLM's deep_gemm._lazy_init has no lock and calls DeepGEMM's set_pdl, whose
# LazyInit<Runtime> races. The engine initializes DeepGEMM on its own thread before the threaded load and serializes the
# loader threads' prepare calls with this lock.
_A8_LOCK = threading.Lock()

N_LAYERS, W, D, HC, TOPK, NE = 40, 4, 5120, 4, 6, 384
RATIO = Wt.RATIO
DEVTOK_V2 = os.environ.get("ENG_DEVTOK", "1") == "1"   # 1: device_tokens with one pinned copy (_device_tokens_v2,
                                                         # same result); 0: the per-sequence path
BISECT = dict(per_layer_plans=True, clear_tail_padding=True, inject_join=True, engram_args="tables", free_cache=True,
              engram_stream=False, evict_empty_cache=True)
# ^ runtime switches of a set of batched-decode fixes (tests bisect batched-vs-alone decode with them)
if os.environ.get("ENG_EVICT_EMPTY_CACHE", "1") == "0":
    BISECT["evict_empty_cache"] = False   # a mixed-bucket eviction does not empty torch's cache. For expandable_segments:True, where
    # the cache has no split holes left to clear and every release unmaps pages that the next step maps back for the peers
GRAPH_MAX_CTX = 262_144             # decode graphs' indexer bounds (hooks_idx.MAX_CTX); longer contexts run eagerly
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
LONG_ANY_ROWS = os.environ.get("FPR_LONG_ANY_ROWS", "0") == "1"   # long-row decode steps
                     # with <= 64 rows in all also replay the long graphed variant, on the stock PERSISTENT top-k (exact;
                     # production picks the cooperative one below 65 rows, so outputs can differ from production in rare
                     # tie / threshold-bin cases) instead of the eager mixed path (~13 tok/s per request at 270-290K)
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
ROUTE_STATS = os.environ.get("ENG_ROUTE_STATS", "")    # <path>.json gets, per
                                                       # decode bucket, the mean number of distinct routed experts (the
                                                       # route tables' EC count) per layer and step, rewritten every 256
                                                       # steps (one device add per layer inside the graphs)
DUMP_MOE = os.environ.get("ENG_DUMP_MOE", "")         # debug: dump the first mixed step's fork-MoE calls (_dump_moe)
PROF_PIECES = os.environ.get("ENG_PROF_PIECES", "") == "1"   # profiler ranges per piece (host time / launches per piece)
C1_QB_ORDER = os.environ.get("ENG_C1_QB_ORDER", "0") == "1"   # (with ENG_C1_QB_OVERLAP) issue the indexer chain before q_b
C1_QB_OVERLAP = os.environ.get("ENG_C1_QB_OVERLAP", "0") == "1"   # decode buckets, indexer layers
                                                     # the q_b GEMM (+ RoPE + window insert) on a side stream next to the
                                                     # indexer chain (independent; joined before FlashMLA, which reads both)
C1_COMM = os.environ.get("ENG_C1_COMM", "0") == "1"   # the fork MoE's AG / RS move only real rows
                                                     # (kernels/sp_layer/comm_c1.cu: the same statements per real row)
C1_AGW = os.environ.get("ENG_C1_AGW", "0") == "1"   # with ENG_C1_COMM: the AG wait folded into the push kernel
C1_TPWO = os.environ.get("ENG_C1_TPWO", "0") == "1"   # SAME MATH, NOT BITWISE: decode buckets of
                                                     # <= 32 rows per rank run wo_a / wo_b split over the 4 GPUs (TP4 like
                                                     # production: head groups 2g, 2g+1 and wo_b's K quarter g on GPU g,
                                                     # fp32 partials summed on the owner); comm in kernels/sp_layer/comm_c1.cu
C1_TPWO_ROPE = os.environ.get("ENG_C1_TPWO_ROPE", "1") == "1"
C1_TPWO_MAXR = int(os.environ.get("ENG_C1_TPWO_MAXR", "32"))   # TP-wo in decode buckets of <= this many rows per rank   # TP-wo: the inverse RoPE inside the scatter kernel
C1_FOLD = os.environ.get("ENG_C1_FOLD", "0") == "1"   # (with ENG_C1_COMM) the RS sums (MoE; TP-wo) folded into the next
                                                     # mHC boundary (mhcf.cu k_frow_rs: the same adds in the same order)
C1_RSC = os.environ.get("ENG_C1_RSC", "0") == "1"   # (with ENG_C1_COMM) the MoE combine folded into the RS push (comm_c1.cu
                                                     # k_rs_cpushv: k_combine's statements, bitwise)
C1_FMAXR = int(os.environ.get("ENG_C1_FUSE_MAXR", "32"))   # ENG_C1_RSC / ENG_C1_AGRT: decode buckets of <= this many rows
C1_AGRT = os.environ.get("ENG_C1_AGRT", "0") == "1"   # (with ENG_C1_COMM + ENG_C1_AGW) router_top_pack3 folded into the AG
                                                       # push kernel (comm_c1.cu k_ag_rt: the same statements, bitwise)
C1_TPWO_F8 = os.environ.get("ENG_C1_TPWO_F8", "0") == "1"   # (with ENG_C1_TPWO) each GPU's wo_a groups / wo_b K quarter
C1_TPWO_F8_S = tuple(int(x) for x in os.environ.get("ENG_C1_TPWO_F8_S", "4:2").split(":"))   # from packed fp8 slices of the
                                                       # checkpoint's weights on skw8t (exact dequant, fp32 partials); K splits
WO_W8A16 = os.environ.get("ENG_WO_W8A16", "0") == "1"   # R2: ONE resident fp8 copy of wo_a /
                                                       # wo_b. C128 / mixed / prefill run them W8A16 (w8a16.cu: the checkpoint's
                                                       # e4m3 + 32 x 32 scales, exact bf16 dequant, bf16 MMA, fp32 sums =
                                                       # vLLM's W8A16; not bitwise vs cuBLAS on the bf16 copies: summation
                                                       # order); C1 / TP-wo read this rank's bf16 quarter (wo_aT_c1 / wo_b_c1,
                                                       # as DENSE8); the full bf16 copies never reach the GPU (-1.5 GB / GPU)
WO_ONECOPY = os.environ.get("ENG_WO_ONECOPY", "0") == "1"
                                                       # Phase 1b / 1c) with ENG_WO_W8A16 + ENG_C1_TPWO: NO bf16 quarters
                                                       # (wo_aT_c1 / wo_b_c1: -1.51 GB / GPU). TP-wo runs the ENG_C1_TPWO_F8
                                                       # path on THIS RANK'S VIEWS of R2's packed copy (wo_a pk[:, 8r:8r+8],
                                                       # wo_b pk[16r:16r+16]) through w8a16tp.cu; C1
                                                       # must read the packed copy too (c1mega C1M_B16 woa8 / wob8, srv8x)
WO_ONECOPY_S = tuple(int(x) for x in os.environ.get("ENG_WO_ONECOPY_S", "16:6").split(":"))   # TP-wo split-K (wo_a, wo_b)
WO_W8A16_S = tuple(int(x) for x in os.environ.get("ENG_WO_W8A16_S", "4:6").split(":"))   # split-K (wo_a, wo_b) at <= 128 rows
WO_W8A16_DQ = os.environ.get("ENG_WO_W8A16_DQ", "0") == "1"   # with ENG_WO_W8A16, calls of
QB_DQ = os.environ.get("ENG_QB_DQ", "0") == "1"   # prefill q_b (>= ENG_DQ_MINR rows, not capturing) on the same
                                                       # route instead of the fused w8a16 (= the bf16 copy's cuBLAS call + RoPE / insert)
DQ_MINR = int(os.environ.get("ENG_DQ_MINR", "384"))   # > the largest decode bucket (320 rows per rank): captured decode graphs
                                                       # never dequantize
DQ_PF = os.environ.get("ENG_DQ_PF", "1") == "1"   # dequantize the weight that followed this one last time on a side stream
                                                       # right after this one's GEMM (event-ordered, no syncs; bitwise)
C1_G1KS = int(os.environ.get("ENG_C1_G1KS", "0"))   # decode buckets <= 32 rows per rank: moe_v4 GEMM1 n_pad-8 tiles split
                                                       # into this many K parts when they exceed one wave (gemm1rk; same math)
C1_MOEF = os.environ.get("ENG_C1_MOEF", "0") == "1"   # decode buckets <= ENG_C1_MOEF_MAXR rows per rank: moe_v4's fused
C1_MOEF_MAXR = int(os.environ.get("ENG_C1_MOEF_MAXR", "32"))   # GEMM1 -> GEMM2 kernel (gemm12: same tiles, same bytes)
C1_L2PF = set(x for x in os.environ.get("ENG_C1_L2PF", "").split(",") if x)
    # decode buckets of <= ENG_C1_L2PF_MAXR rows per rank: HBM -> L2 prefetch of the next GEMM's weights on a side stream
    # during the latency-bound stage before it (kernels/l2pf/l2pf.cu; no outputs, every consumer's math unchanged).
    # Items: qkv = layer L+1's q_a | wkv, issued at layer L's MoE RS; wo = this rank's TP-wo slices (wo_a groups, wo_b K
    # columns), issued after q_b, before the attention core; gate = the router weight + the shared expert, issued at the
    # FFN boundary. Joined at the end of the step.
C1_L2PF_MAXR = int(os.environ.get("ENG_C1_L2PF_MAXR", "4"))
G6_NOPADW = os.environ.get("ENG_G6_NOPADW", "0") == "1"   # see Engine.g6
MOE_X2 = os.environ.get("ENG_MOE_X2", "0") == "1"
    # off): decode buckets with >= ENG_MOE_X2_MINR rows per rank run the routed GEMM1 as W4A8 (kernels/moe_x2: gather_w8
    # writes an e4m3 act_quant image into the bucket's img buffer, fp8 wgmma GEMM1 on the unchanged raw FP4 weights, moe_v4's
    # H image) and the served gemm2r on it. ENG_MOE_X2_ASC = tok (one ue8m0 scale per token) | 32 (act_quant per 32: the
    # official numerics, kernels/w8a8/act_quant.cuh). moe_x2 is imported / JIT-built only when on (Engine.moe_x2).
MOE_X2_ASC = os.environ.get("ENG_MOE_X2_ASC", "tok")
MOE_X2_MINR = int(os.environ.get("ENG_MOE_X2_MINR", "128"))
MEM_TRACE = os.environ.get("ENG_MEM_TRACE", "0") == "1"   # one log line per mixed step with
    # its composition (prefill rows / sequences / longest prefill context, decode rows, prefill-attention workspace bytes)
    # and its peak allocated memory per GPU: the data for bounding the mixed step's peak (the agent-soak OOMs)
MOE_PRELOAD = os.environ.get("ENG_MOE_PRELOAD", "1") != "0"
    # every routed-MoE extension the configuration can reach is loaded at the end of Engine.__init__ (Engine._preload_moe),
    # not on first use inside a step: torch's JIT load runs a ninja subprocess even when the build is up to date. Same loaders, same cached objects: bitwise by construction.
    # ENG_MOE_PRELOAD=0: first use, as before.
ATTN_PRELOAD = os.environ.get("ENG_ATTN_PRELOAD", "1") != "0"   # the attention /
    # indexer extensions a step would JIT-load on first use, loaded at the end of Engine.__init__ (Engine._preload_attn)
    # through the same cached loaders.
    # Bitwise by construction. ENG_ATTN_PRELOAD=0: first use, as before.
PF_WS_MB = int(os.environ.get("ENG_PF_WS_PREALLOC_MB", "0"))
    # default 0 = allocated per mixed step from the step's prefill content, as before). > 0: one persistent bf16 buffer per
    # GPU of this many MiB, allocated at engine init BEFORE the KV pools are sized, reused by every step's plan (shared
    # layout when the disjoint one does not fit). 1536 = max(ATTN2B_PF_WS_MB 1024, ATTN2B_PF_WS_TOTAL_MB 3072 / 2): every
    # layout attn_prefill can build fits, so the path allocates nothing at runtime (the OOM of the 128-conversation soak).
MIX_POOL_GB = float(os.environ.get("ENG_MIX_POOL_GB", "0") or 0)   # the mixed-step
    # bucket states (A / T roles) live in ONE dedicated torch.cuda.MemPool per GPU of this many GB, reserved at engine
    # init BEFORE the KV pools are sized (one tensor of that size, freed at once: the segment stays in the pool). Buckets
    # are carved from it (no cudaMalloc when R changes) and the LRU cap becomes strict: min(mixed_bucket_bytes, pool -
    # ENG_MIX_POOL_SLACK_GB). Measured soak OOM cause: A + T re-created larger as decode rows grow (0.95 -> 4.55 GB held
    # at 128 seats) in the default pool, with the churn's cached blocks on top. use_on_oom: other allocations may borrow
    # the pool's free blocks as a last resort instead of failing.
MIX_POOL_SLACK_GB = float(os.environ.get("ENG_MIX_POOL_SLACK_GB", "0.4"))
CXX_PIECES = frozenset(x for x in os.environ.get("ENG_MIXED_CXX_PIECES", "").split(",") if x)
    # default empty = off): pieces of mixed buckets issued by the C++ driver (fpr/engine/mixed_cxx: driver.cpp through the
    # extensions' C ABI tables, bitwise by construction): "bd" = p_attn_bd / p_ffn_bd's boundary (ENG_MHCF fused path),
    # "moe" = Engine.p_moe; "router" / "ag" / "rs" /
    # "final" = the served chains of p_router / p_ag / p_rs / p_final (agrs wrappers included), wrapped from outside by
    # mixed_cxx.install() after _install_agrs() (prod buckets, eager; anything else falls through to the chain)
    # "embed" = p_embed (ids32 copy, Engram prepare with the cuBLAS wkv, embedding gather), "engram" = p_attn_bd at the
    # Engram layers (inject + identity-mode boundary): mixed_cxx_engram.py / pieces_engram.cpp
CXX_RUN = bool(CXX_PIECES) and os.environ.get("ENG_MIXED_CXX_RUN", "0") != "0"   # stage 1d
    # Engine._issue of prod buckets as spans of C++ pieces on the driver's 4 per-rank threads (mixed_cxx._issue_span: "1"
    # threads, "serial" = the same spans from the calling thread), Python pieces between spans; needs ENG_MIXED_CXX_PIECES
MIXED_CXX_HEAD = os.environ.get("ENG_MIXED_CXX_HEAD", "0") == "1"   # C1 tail: the HSD head of a
    # mixed step from C++ (engine/mixed_cxx/mixed_cxx_head.py, head_h1.cpp: per rank the sample-row gather, the aux copy
    # and the rank's captured graph of HSDT.run for the step's HSD key; uncaptured keys / grammar steps: the Python head)


def _mxh():
    """ENG_MIXED_CXX_HEAD: engine/mixed_cxx/mixed_cxx_head.py (imported on first use)."""
    p_ = os.path.join(PHYS, "engine/mixed_cxx")
    if p_ not in sys.path:
        sys.path.insert(0, p_)
    import mixed_cxx_head
    return mixed_cxx_head


MIXED_PRESTEP = os.environ.get("ENG_MIXED_PRESTEP", "0") == "1"
    # off; every host sync of a mixed step BEFORE its first
    # layer is issued, the same data and kernels (bitwise): the T bucket looked up (created / evicted, with their all-device
    # syncs) and the fused-RS flags ensured right after prepare A instead of after A's issue; _set_vidx's and head_eager's
    # H2D copies from pinned memory (non-blocking); production prefill attention's Plan uploads non-blocking + its window
    # length read from the host formula instead of the device (kernels/attn_dense/attn_prefill.py). The C++ driver (1b+)
    # then needs nothing host-side during issue.
C128_RSL = os.environ.get("ENG_C128_RSL", "0") == "1"   # decode buckets of
    # ENG_C128_RSL_MINR .. ENG_C128_RSL_MAXR rows per rank (33 .. 256) run the MoE combine + reduce-scatter as ONE lamport
    # kernel per rank: comm_c1.cu's k_rs_cpushl (ENG_C1_RSL's kernel, unchanged; grid-stride, all CTAs resident) with
    # ENG_C128_RSL_GRID CTAs, its own slots / call counters per bucket. Replaces AGRS_RSD's combine_push_dm + rs_sum (flag
    # round trip, two launches). The same combine / rank-order-sum statements (rsl_test: bitwise vs rs_cpushv + rs_sumv).
C128_RTOP = os.environ.get("ENG_C128_RTOP", "0") == "1"   # the router top-k of
    # decode buckets of 33 .. 320 rows on kernels/c128r/rtop.cu (k_router_top's statements verbatim: bit-identical ids /
    # weights / routing rows) launched with ENG_C128_RTOP_W warps per CTA (default 1) instead of moe_raw's 8: 128 rows
    # on 128 SMs instead of 16 (~7.8 us per layer in the C128 trace).
C128_RTOP_W = int(os.environ.get("ENG_C128_RTOP_W", "1"))
STREAM_PRIO = int(os.environ.get("ENG_STREAM_PRIO", "0"))
    # per-GPU streams (self.streams: the layers' critical path) at this CUDA priority (< 0 = higher; torch's range is
    # 0 .. -3); the side streams (mHC mixes, shared expert, Engram) stay at 0. Motive: in the C128 trace the side stream's
    # k_fmix is dispatched first when a boundary's k_frow ends, and the next main-stream GEMM (q_a | wkv, gate) waits ~8 us
    # for SMs, twice per layer (~0.7 ms per step). Scheduling only: the same kernels, the same bits.
C128_AGF = os.environ.get("ENG_C128_AGF", "0") == "1"   # decode buckets of
    # 33 .. 320 rows split the FFN all-gather. The f rows (R x 10 KB) are pushed right after the FFN boundary on the side
    # stream (comm.cu ag_push), under the gate GEMM + top-k; after the top-k only the 64-B routing rows go (ag_push) and
    # one wait covers both (one flag / epoch: two pushes per round). Today's ag_push2v_wait starts after the top-k and
    # carries f too (~24 us to the AG's end in the C128 trace). Same bytes in the same slots.
C128_RSL_MINR = int(os.environ.get("ENG_C128_RSL_MINR", "33"))
C128_RSL_MAXR = int(os.environ.get("ENG_C128_RSL_MAXR", "256"))
C128_RSL_GRID = int(os.environ.get("ENG_C128_RSL_GRID", "264"))
MIX_ONE_BUCKET = os.environ.get("ENG_MIX_ONE_BUCKET", "0") == "1"   # production evicts a
    # mixed bucket every ~1.6 s per slot, LRU thrash over ~32 A sizes, and IMAs track evictions): ONE A bucket at the largest
    # row count (rnd(max_mixed_rows)) is carved once and never evicted; a step of R rows runs on a cached VIEW state over it
    # (Engine._view_state: every tensor a fresh _alloc(r, R, 'prod') would make, with the same shape and, for row-led
    # tensors, the same initial rows), so kernels see exactly a fresh R bucket: no padding compute, no new memory, no
    # eviction, no all-device sync. T buckets are unchanged.
MIX_ONE_BUCKET_T = MIX_ONE_BUCKET and os.environ.get("ENG_MIX_ONE_BUCKET_T", "0") == "1"
# ENG_MIX_ONE_BUCKET): the mixed TAIL buckets too as cached views over ONE pinned max-T bucket (rnd(max_tail_rows) rows), the same
# view rules as A; with max_mixed_rows at the served bound both pinned bases fit the mixed budget: no mixed evictions at all
MIX_VIEW_LAZY_MAX_MB = float(os.environ.get("ENG_MIX_VIEW_LAZY_MAX_MB", "0") or 0)
    # with ENG_MIX_ONE_BUCKET, the bytes of tensors that steps added to the cached view states after their creation (keys
    # outside the creation snapshot) are logged at each new view and in every steplog line


_LAZY_BIG = 1 << 30   # storages above 1 GiB reached from a view are global pools (KV / window), not per-view growth
_LAZY_SKIP_T = (torch.cuda.Stream, torch.cuda.Event, torch.nn.Module, type, type(os))


def _geom_state_slots() -> int:
    """serve's state-pool slots per rank: PHYS_GEOM's state_pool_slots when set (the served geometry the worker binds with
    FPR_WORKER_GEOM), else Geometry()'s default."""
    for kv in os.environ.get("PHYS_GEOM", "").split(","):
        if "=" in kv:
            k, v = (x.strip() for x in kv.split("=", 1))
            if k == "state_pool_slots":
                return int(float(v))
    return Geometry().state_pool_slots


_TOF_SKIP = (str, bytes, int, float, bool) + _LAZY_SKIP_T
_TOF_LEAF = frozenset((int, float, bool, str, bytes, type(None)))   # exact types the walk below never descends into


def _tensors_of(v, depth: int = 3) -> list:
    """the tensors of a state value, up to `depth` (3: s[k] -> dict -> object -> dict) levels into lists / tuples / dicts and plain objects' attributes."""
    out = []
    _tof(v, depth, out)
    return out


def _tof(v, depth: int, out: list) -> None:
    """_tensors_of's walk: appends v's tensors to out"""
    if torch.is_tensor(v):
        out.append(v)
        return
    if depth <= 0 or v is None or isinstance(v, _TOF_SKIP) or type(v).__name__ == "Engine":
        return
    if isinstance(v, (list, tuple)):
        vals = v
    elif isinstance(v, dict):
        vals = v.values()
    elif hasattr(v, "__dict__"):
        vals = vars(v).values()
    else:
        return
    d, leaf, T = depth - 1, _TOF_LEAF, torch.Tensor
    for x in vals:
        if type(x) in leaf:
            continue
        if isinstance(x, T):
            out.append(x)
        else:
            _tof(x, d, out)


MIX_TBL_CAP = os.environ.get("ENG_MIX_TBL_CAP", "0") == "1"   # mixed buckets size the
    # fork MoE's per-slot tables (Y, img, himg) for min(Tall, prefill_moe_switch) rows x TOPK (+ the 7-per-EC padding)
    # instead of Tall x TOPK: a mixed step takes the fork's MoE only when its valid rows (all ranks) are <=
    # prefill_moe_switch, and route2 gives slots to valid (row, expert) pairs only. Memory only (same kernels, same
    # slots); -0.73 GB for a 2,048-row bucket.
# ENG_MIXED_GRAPHS_T: the mixed step's TAIL bucket (layers 21..39 on decode rows + chunk tails) runs as piecewise CUDA graphs
# (_issue_mixed: the indexer of index layers and the MoE stay eager units, everything else replays per bucket). Bucket T
# has no per-step row splits (no ATTN_PF plan, pf_ab None), so its units capture as they are; bucket A (all rows: the
# ATTN_PF prefill attention, production's W8A8 prefill rows) stays eager. Not with CP steps or ENG_L20_TAIL (eager, as
# before). ENG_MIXED_T_ROUND: graphed T buckets are rounded up to this many rows (capped at the engine's max tail rows):
# fewer bucket sizes = fewer captures and less bucket-cache churn (default 128 -> at most 12 T buckets at 1,536).
MIXED_GRAPHS_T = os.environ.get("ENG_MIXED_GRAPHS_T", "0") == "1"
assert not (MIX_ONE_BUCKET_T and MIXED_GRAPHS_T), "ENG_MIX_ONE_BUCKET_T with ENG_MIXED_GRAPHS_T is not gated (T graphs are keyed per bucket)"
ENGRAM_STAGE = int(os.environ.get("FPR_ENGRAM_STAGE", "0") or 0)
MIXED_T_ROUND = max(1, int(os.environ.get("ENG_MIXED_T_ROUND", "128") or 128))
MAX_TAIL_ROWS = int(os.environ.get("ENG_MAX_TAIL_ROWS", "0") or 0)
    # 4,416): cap on the tail rows per rank of a mixed step (layers 21..39 + head; serve's scheduler reads it as
    # Geometry.max_tail_rows_per_rank; serve's own default is 1,536). The T bucket reached 3,584 rows (2.5 GB) in the
    # 128-conversation soak; T at <= 2,048 rows also carries the fork's MoE tables (T2048: 2.57 GB).
MIX_POOL_SPLIT = tuple(float(x) for x in os.environ.get("ENG_MIX_POOL_SPLIT", "").split(",") if x)   # "a_gb,t_gb":
    # instead of one pool, one pool per role (A, T), each holding ONE bucket: a new size first evicts the role's previous
    # bucket, so its pool is entirely free (coalesced) when the next one is carved: no fragmentation, no growth while a
    # bucket fits its pool.
C1_RSL = os.environ.get("ENG_C1_RSL", "0") == "1"   # the TP-wo
    # reduce-scatter + rank-order sum as ONE lamport-style kernel per rank (comm_c1.cu k_tp_rsl: data words carry the
    # arrival, sentinel -0.0; the same order and values as tp_rs + tp_rs_sum except a -0.0 partial travels as +0.0)
C1_MEGA = os.environ.get("ENG_C1_MEGA", "0") == "1"   # decode steps of the
    # 4-row bucket with exactly ONE rank holding rows: every run of regular layers (not index / Engram) as one c1mega launch
    # per GPU (fpr/kernels/c1mega/c1mega_eng.py: whole layers, TP4, owner-mode attention; same math, other summation
    # order); split graph L key (R, "L", "M", owner); everything else unchanged
# ENG_C1_MEGA_AB=<pkgA>,<pkgB>: a within-process served A/B of two c1mega
# packages (dirs under fpr/, e.g. kernels/c1mega/srv5x_ab,kernels/c1mega/srv6_ab; each loaded in isolation with its own
# c1idx_host, its C1M_SRC / C1M_BUILD from the package's ab.json). The ACTIVE arm flips every ENG_C1_MEGA_AB_N C1 steps
# (default 64), at a step boundary before owner_of; ENG_C1_MEGA_AB_START=a|b picks the first arm. Each arm captures its own
# C1 graph L (key + (arm,)); graph H is shared. PHYS_STEP_TIMES rows carry "ab": 0 / 1 in their "d13" / "eng" dict on C1
# steps. Needs ENG_C1_MEGA=1; replaces C1M_PKG.
C1_MEGA_AB = [x.strip() for x in os.environ.get("ENG_C1_MEGA_AB", "").split(",") if x.strip()]
assert not C1_MEGA_AB or (len(C1_MEGA_AB) == 2 and C1_MEGA), "ENG_C1_MEGA_AB needs two packages and ENG_C1_MEGA=1"
C1_MEGA_AB_N = max(1, int(os.environ.get("ENG_C1_MEGA_AB_N", "64") or 64))
C1_MEGA_AB_START = 1 if os.environ.get("ENG_C1_MEGA_AB_START", "a").strip().lower() == "b" else 0
C1_L2PF_GRID = int(os.environ.get("ENG_C1_L2PF_GRID", "32"))
C1_L2PF_CHUNK = int(os.environ.get("ENG_C1_L2PF_CHUNK", "16384"))
C1_RTF = os.environ.get("ENG_C1_RTF", "0") == "1"   # decode buckets <= 32 rows: router GEMM + top-k in one kernel
                                                     # (kernels/moe_stream/rtf.cu: cuBLAS's split-K 3 reproduced)
# FPR_C1_KMAX: k_max > 3 adds the 1 + k_max row C1 bucket.
# Serve lays out a step whose only work is one decoding sequence at 1 + k_max rows (serve/scheduler.py _c1_widen); D13
# runs its head on the lane HSD (ENG_HSDT_MOD=hsdt_c1k, hooks.hsd = the lane only inside that dispatch). Every other
# bucket keeps 4 rows per sequence and the main HSD (k = 3), whose sequence buckets never include the C1 bucket.
C1_KMAX = int(os.environ.get("FPR_C1_KMAX", "3") or 3)
C1_K1L = C1_KMAX + 1 if C1_KMAX > 3 else 0
# ENG_DECODE_PRECREATE. A
# decode bucket's state and graphs are created by the first step that needs its R and never freed, so the memory a process holds depends on
# which R's its traffic happened to reach. On: while some bucket of cfg.buckets has no state, bucket_of() pads a decode
# step (more than max(4, C1_K1L) rows on its fullest rank: never a C1 / C1-lane step) up to the largest missing bucket,
# for ENG_DECODE_PRECREATE_REPEAT steps per bucket (its eager step and graph captures, through the normal path). The
# warm-up then creates every decode bucket and the process holds its worst case from the start. One decision per step
# serial: every bucket_of() call of a step agrees. Padding moves only those steps' rows into a larger bucket (as seat
# imbalance does in serving). Gives up after ENG_DECODE_PRECREATE_MAX_STEPS step serials (logged). Long-row variants
# (ENG_LONG_GRAPHS, key R + lvl * LVL_TAG) stay lazy. HSD's buffers are max-sized at init (kernels/head/hsd.py).
DECODE_PRECREATE = os.environ.get("ENG_DECODE_PRECREATE", "0") == "1"
DECODE_PRECREATE_REPEAT = int(os.environ.get("ENG_DECODE_PRECREATE_REPEAT", "3"))
DECODE_PRECREATE_MAX_STEPS = int(os.environ.get("ENG_DECODE_PRECREATE_MAX_STEPS", "4000"))
# ENG_DECODE_PRECREATE_STALL: the done / GAVE UP decision ran only inside _pc_pick
# i.e. only when a decode step pads (> max(4, C1_K1L) rows). After warm-up a quiet slot sends 4-row C1 steps only, nothing
# calls _pc_pick, and the decision never came: warmup_due() stayed true and nd kept every decode bucket embedded forever.
# Now the native step boundary (native_boundary) also decides once no padding-eligible step has come for STALL step
# serials: "done (stalled)" when every requested bucket exists (its REPEAT count may be short: no traffic can reach it),
# else GAVE UP; the done-point actions (HSD views precreate, MIX precreate, C1 / H1 precapture marks) run as before.
DECODE_PRECREATE_STALL = int(os.environ.get("ENG_DECODE_PRECREATE_STALL", "64"))
# ENG_HSD_VIEWS: HSD's per-bucket drafter scratch (DraftRank b / m) as flat-prefix views of
# its own D_max ranks0[r] (kernels/hsdtail/hsd_views.py; ~1.5 GB per rank). With ENG_DECODE_PRECREATE also every drafter
# bucket is created at PRECREATE's done point (hsd_views.precreate: draft_rank + plan + PeerComm; no graph capture).
HSD_VIEWS = os.environ.get("ENG_HSD_VIEWS", "0") == "1"
# ENG_MIX_PRECREATE
# at PRECREATE's done point every mixed onebk VIEW the cap allows (A: mixed_round .. max_mixed_rows, T: .. max_tail_rows;
# the bases too) is created through bucket(), i.e. with ENG_MIX_PERSIST's buffers and ENG_MIX_ENSURE's handles: no
# first-use view creation (nd: no first-use refusal) in steady state. Views are never evicted (bk_view).
MIX_PRECREATE = os.environ.get("ENG_MIX_PRECREATE", "0") == "1"
# ENG_C1_PRECAPTURE: at PRECREATE's done
# point (after the drafter / mixed views) c1mega's graph L of every C1 owner key (R, kind, "M", owner) + _ab_key() is
# captured without an eager step (kernels/c1mega/c1m_precapture.py: capture only, no synthetic row, so no KV / Engram /
# HSD / D13 state is written; GPUs synced before and after). Owners already captured are skipped; a failed owner keeps
# today's first-step capture (Python first C1 step). ENG_C1_PRECAPTURE_KINDS: "LA" (D13, default) and / or "L".
C1_PRECAPTURE = os.environ.get("ENG_C1_PRECAPTURE", "0") == "1"
C1_PRECAPTURE_KINDS = tuple(x for x in os.environ.get("ENG_C1_PRECAPTURE_KINDS", "LA").split(",") if x)
# ENG_HEAD_BM_GRAPH: a grammar / structured step (hooks.bm_step, HSD's
# bitmask variant) gets its own graph H, key (R, "H", hsd_key() + ("bm",)) = text "R|H|S_pad|D_pad|NT[|gvar]|bm" (nd's
# lookup), captured like any new H key (first use: head_eager + capture; D13: at extraction). Safe to bake: HSD's mask
# rows are persistent (blob path _bm_d[r][:T], all T = W x S_pad rows rewritten every grammar step; _build path the
# per-key tb["bm"]) and num_acceptable is in the blob (lim / tok_j). Never fused into graph L (hsd_graph_ok unchanged).
# Off: grammar steps run head_eager as before.
HEAD_BM_GRAPH = os.environ.get("ENG_HEAD_BM_GRAPH", "0") == "1"
# ENG_MIXED_HEAD_PRECAPTURE: the mixed head's H1 graphs of the keys in ENG_MIXED_HEAD_PRECAPTURE_KEYS captured before first use
# (mixed_cxx_head.precapture: per key a synthetic HSD.prepare + capture, nothing executes). NOT at PRECREATE's done point
# itself: that fires inside D13.dispatch's bucket_of while the previous step is in flight and not yet extracted, and a
# synthetic prepare rewrites HSD's per-step state (cur / _res / tables) that its extraction reads. The done point marks
# it due; it runs at the START of the next Python step_mixed (D13: through _fallback, which drains first), GPUs synced, before
# that step's own prepare_head rewrites HSD's per-step state.
MIXED_HEAD_PRECAPTURE = os.environ.get("ENG_MIXED_HEAD_PRECAPTURE", "0") == "1"
# ENG_NATIVE_BOUNDARY. The
# native loop never calls bucket_of (C++ D13 / nd pick decode buckets), so ENG_DECODE_PRECREATE's per-serial decision
# (_pc_pick) only advances in embedded steps, and np11's H1 precapture waits for a Python step_mixed. Engine.precreate_active() (always answers) lets the exporter tell nd to run decode steps embedded
# until precreate's done point (Python D13.dispatch -> bucket_of -> _pc_pick, exactly as in Python mode). With this flag,
# Engine.native_boundary() (core2b NativeWorker.post: a step boundary, nothing of nd in flight) runs a due H1 precapture
# once D13 has nothing unextracted (else it stays due; the next boundary retries), then resets D13's skip state as
# D13._fallback does (the synthetic HSD.prepare rewrote per-step HSD state). MIX_PRECREATE / C1 precapture still run at the
# done point itself (unchanged). Off: native_boundary() returns at once.
NATIVE_BOUNDARY = os.environ.get("ENG_NATIVE_BOUNDARY", "0") == "1"
# ENG_PRECAP_DEFER: PRECREATE's
# done / GAVE UP point no longer runs the heavy precaptures inline (in the warm-up transient). It queues ONE job (MIX_PRECREATE views,
# the C1 graph-L precapture, the H1 precapture incl. the full grid / decode-H of mixed_cxx_head) that native_boundary() runs at a
# later step boundary (D13 idle), while warmup_due() keeps holding decode embedded:
#   - attempt boundaries: from PRECAP_MIN_B boundaries after done, then every PRECAP_EVERY_B (synchronize + empty_cache + per-GPU
#     cudaMemGetInfo only there: "headroom" = smi-free after empty_cache, i.e. torch slack is ~0 by construction);
#   - run when min-GPU headroom >= PRECAP_MIN_FREE_GB;
#   - deadline (PRECAP_MAX_WAIT_S seconds after done, or PRECAP_MAX_WAIT_B boundaries): run if headroom >= PRECAP_HARD_MIN_GB,
#     else GAVE UP: one loud line, stats precap_defer_gave_up = 1, the hold released WITHOUT the job (first-use captures, as today);
# Never an unbounded hold.
# One final line either way ("ENG_PRECAP_DEFER: precapture done" / "... GAVE UP"): the switch wrapper waits on it before public
# traffic. Before / after lines carry per-GPU headroom, the boundaries / seconds from done and the decode steps run embedded meanwhile.
PRECAP_DEFER = NATIVE_BOUNDARY and os.environ.get("ENG_PRECAP_DEFER", "0") == "1"
PRECAP_MIN_B = int(os.environ.get("ENG_PRECAP_MIN_B", "8"))
PRECAP_EVERY_B = max(1, int(os.environ.get("ENG_PRECAP_EVERY_B", "8")))
PRECAP_MIN_FREE_GB = float(os.environ.get("ENG_PRECAP_MIN_FREE_GB", "6"))
PRECAP_HARD_MIN_GB = float(os.environ.get("ENG_PRECAP_HARD_MIN_GB", "2"))
# at the deadline the job runs only if min free >= the job's ESTIMATED size + PRECAP_MARGIN_GB (and >= HARD_MIN)
# a job started with too little room would OOM mid-serve (EngineDead), worse than GAVE UP. Estimate = ENG_PRECAP_EST_GB if set, else
# H1 keys planned x ENG_PRECAP_H1_MIB_PER_KEY (1.7, hk-03c) + decode-H ENG_PRECAP_DECH_GB (1.2 when ENG_DECODE_HEAD_PRECAPTURE)
# + onebk views planned x ENG_PRECAP_VIEW_MIB (8, c1m-nb-01: +384 MiB / 50) + nd precapture_all ENG_PRECAP_ND_GB (0.5). Logged
# next to the measured before -> after delta.
PRECAP_MARGIN_GB = float(os.environ.get("ENG_PRECAP_MARGIN_GB", "1"))
PRECAP_EST_GB = float(os.environ.get("ENG_PRECAP_EST_GB", "0") or 0)
PRECAP_H1_MIB_PER_KEY = float(os.environ.get("ENG_PRECAP_H1_MIB_PER_KEY", "1.7"))
PRECAP_DECH_GB = float(os.environ.get("ENG_PRECAP_DECH_GB", "1.2"))
PRECAP_VIEW_MIB = float(os.environ.get("ENG_PRECAP_VIEW_MIB", "8"))
PRECAP_ND_GB = float(os.environ.get("ENG_PRECAP_ND_GB", "0.5"))
PRECAP_MAX_WAIT_S = float(os.environ.get("ENG_PRECAP_MAX_WAIT_S", "60"))
PRECAP_MAX_WAIT_B = int(os.environ.get("ENG_PRECAP_MAX_WAIT_B", "512"))
# ENG_DECODE_VIEWS. Every decode bucket state of >=
# ENG_DECODE_VIEWS_MINR rows (lvl 0) takes its per-step scratch as views of ONE per-rank base per key, sized for the largest
# decode bucket (flat prefix view base[:numel].view(shape): contiguous for every shape, as _view_state does for the mixed
# buckets), so the process holds one bucket's scratch instead of the sum.
# Shared: the residual stream / mixes, the router outputs, AG / RS data, MoE scratch and the fork tables' img / himg / Y /
# slotmap (all rewritten every step). Private as before: the input blob and its views, ti / tl, constants (post0, lag0,
# comb0, eyec, padrow, tok32), flags / epochs / Lamport slots and tickets, the fork tables' counters and route tables, the
# attention workspace, the indexer state. A view is zeroed when its bucket is created (= a fresh torch.zeros, stream-
# ordered on the rank stream). Same-bucket consecutive steps already reuse their memory, so the cross-step ordering that
# protects them protects views of different buckets too. Long-row variants (lvl > 0) and buckets < MINR stay private.
DECODE_VIEWS = os.environ.get("ENG_DECODE_VIEWS", "0") == "1"
# ENG_MIX_PERSIST: the mixed step's per-step input tensors become
# persistent buffers nd can hold views of (generation-checked by the exporter): per mixed bucket state and rank ONE device
# blob vidx_d = vidx i64 [W*R] @0 | nv i32 [W] @8WR | nd i32 [W] @8WR + ceil8(4W), filled from one of two pinned sets
# (s["vidx_h"] = [[host, event], [host, event]], s["vidx_i"]; a set is refilled after its copy of two steps ago) with ONE H2D
# on the rank stream; s["vidx"] / ["nv"] / ["nd"] are views of it (same values as the fresh tensors). Per rank: tail_ix
# [max_tail_rows] i64 + its pinned sets (self._tix[r]), and HSD's device-token staging (_dt / _dt_d) allocated at init.
# Readers of vidx / nv / nd run on the rank stream (MoE pieces, the C++ driver's main = rank stream), so the next step's
# H2D (same stream) is ordered after them. FlashMLA plan objects stay fresh per step (FlashMLA plans into new metadata
# tensors when unplanned and reusing a planned object across steps is undefined when lengths change): nd plans per key.
MIX_PERSIST = os.environ.get("ENG_MIX_PERSIST", "0") == "1"
# ENG_TAIL_GATHER: the mixed step's A -> T tail gather reads the rank's persistent
# tail_ix (pinned set + one H2D on the rank stream; no per-step torch.tensor / pin_memory) and writes each index_select
# straight into the T buffer (out=: no temporary + copy). Same values. Implies ENG_MIX_PERSIST's tail_ix.
TAIL_GATHER = os.environ.get("ENG_TAIL_GATHER", "0") == "1"
# ENG_TAIL_GATHER_KERNEL: FPR_OP_TAIL_GATHER (kernels/tail_gather): every key's tail rows
# in ONE launch per 16 keys (a byte copy of whole rows = index_select, bitwise) instead of one index_select per key.
TAIL_GATHER_KERNEL = TAIL_GATHER and os.environ.get("ENG_TAIL_GATHER_KERNEL", "0") == "1"
# ENG_MIX_ENSURE: at every mixed bucket / onebk view creation
# the registered ensure hooks (Engine._ensure_hooks(st): mixed_cxx's driver handle pre-build + the owners' ensure fns) run
# on the new view's 4 states, so every piece handle exists before its first step (nd's ensure_mixed creates buckets
# through the same code: one place for both).
MIX_ENSURE = os.environ.get("ENG_MIX_ENSURE", "0") == "1"
DECODE_VIEWS_MINR = int(os.environ.get("ENG_DECODE_VIEWS_MINR", "64"))
DV_ZERO = ("X", "xm2", "res0", "Xf", "a_pre", "a_post", "f_pre", "f_post", "a_comb", "f_comb", "x0", "a", "f", "moe_out",
           "h", "aux", "X1", "zy", "zpost", "route_row", "ptok_own", "gf", "gr", "rs", "slotmap", "img", "himg", "Y")
FAST_EACH = os.environ.get("ENG_FAST_EACH", "0") == "1"   # _issue's per-rank device + stream switch
    # with torch's C calls instead of the torch.cuda.device / torch.cuda.stream context managers (same state while a
    # piece runs, the same restore after it: device r's previous current stream, then the caller's device). Each
    # context manager pair costs ~30 us of Python (current_stream() -> _get_device_index -> is_available -> getenv);
    # mixed steps do ~1,100 rank-pieces. Also torch.cuda.current_stream() itself (Event.record / wait without a
    # stream, the pieces' own calls): the same Stream object, without the is_available chain.
if FAST_EACH:
    _c_get_dev, _c_set_dev = torch._C._cuda_getDevice, torch._C._cuda_setDevice
    _c_get_stream, _c_set_stream = torch._C._cuda_getCurrentStream, torch._C._cuda_setStream
    _orig_current_stream = torch.cuda.current_stream

    def _fast_current_stream(device=None):
        if device is not None:
            return _orig_current_stream(device)
        torch.cuda._lazy_init()                    # (a flag check once initialized)
        sd = _c_get_stream(_c_get_dev())
        return torch.cuda.Stream(stream_id=sd[0], device_index=sd[1], device_type=sd[2])

    torch.cuda.current_stream = _fast_current_stream
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


def _hostmap(eng) -> None:
    """FPR_HOSTMAP (hosts with GPU-mapped system memory; unset on H200): move lightly read tensors out of HBM, same bytes, same kernels.
    Worker sizes the pools after torch.cuda.empty_cache() (FPR_SIZE_EMPTY_CACHE=1)."""
    want = {x.strip() for x in os.environ.get("FPR_HOSTMAP", "").split(",") if x.strip()}
    if not want:
        return
    import importlib.util
    path = os.environ["FPR_C2C_POOL"]
    spec = importlib.util.spec_from_file_location("c2c_pool", path)
    c2c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c2c)
    moved = []

    def host(t, dev):
        h = c2c.host_copy(t, dev)
        moved.append(t.numel() * t.element_size())
        return h

    hk = eng.hooks
    for r, dev in enumerate(eng.devs):
        with torch.cuda.device(dev):
            if "emb" in want:
                eng.glob[r]["emb"] = host(eng.glob[r]["emb"], dev)
            if "emb_share" in want and hk.hsd is not None:
                assert torch.equal(hk.hsd.emb[r], eng.glob[r]["emb"]), "HSD's embedding differs from the engine's"
                moved.append(hk.hsd.emb[r].numel() * 2)
                hk.hsd.emb[r] = eng.glob[r]["emb"]
            if "gate" in want and hk.gate is not None:
                hk.gate.W[r] = host(hk.gate.W[r], dev)
            if "engram" in want and hk.engram and hk.engram[r].wbf is not None:
                hk.engram[r].wbf = [host(w, dev) for w in hk.engram[r].wbf]
                # FPR_ENGRAM_STAGE=<C>: the cuBLAS branch of Engram's wkv (prefill rows
                # above w8_max_rows) streams the host-resident wbf through ONE persistent HBM buffer in C column chunks
                # instead of
                # cuBLAS re-reading LPDDR5X per M tile. Bitwise; C = 8: 39 MB per GPU
                if ENGRAM_STAGE > 0:
                    spec_s = importlib.util.spec_from_file_location(
                        "engram_stage", os.environ["FPR_ENGRAM_STAGE_PY"])
                    es_ = importlib.util.module_from_spec(spec_s)
                    spec_s.loader.exec_module(es_)
                    if es_.install(hk.engram[r], ENGRAM_STAGE) and r == 0:
                        eng.log(f"engine: FPR_ENGRAM_STAGE={ENGRAM_STAGE}: Engram wkv (rows > w8_max_rows) staged through "
                                f"{hk.engram[r]._stage_buf.numel() * 2 / 1e6:.0f} MB of HBM per GPU")
            if "rope" in want:   # the RoPE tables (per-position row gathers): engine, every layer's attention, indexer
                css = eng.glob[r]["cos_sin"]
                new = {c: host(t, dev) for c, t in css.items()}
                swap = {id(t): new[c] for c, t in css.items()}
                eng.glob[r]["cos_sin"] = new
                for L, lw in eng.lw[r].items():
                    aw = lw.get("AW")
                    if isinstance(aw, dict) and id(aw.get("cos_sin")) in swap:
                        aw["cos_sin"] = swap[id(aw["cos_sin"])]
                if getattr(hk, "idx", None) is not None and hasattr(hk.idx, "W"):
                    for w in hk.idx.W[r].values():
                        if id(w.get("cos_sin")) in swap:
                            w["cos_sin"] = swap[id(w["cos_sin"])]
            torch.cuda.synchronize(dev)
    eng.log(f"engine: FPR_HOSTMAP {sorted(want)}: {sum(moved) / len(eng.devs) / 1e9:.2f} GB per GPU out of HBM")


_DQ_DEV: dict = {}   # ENG_WO_W8A16_DQ / ENG_QB_DQ: str(device) -> {"q" / "a" / "b": _DqSlot}; made at init (main thread)


class _DqSlot:
    """One device's bf16 scratch for one weight kind (q_b / wo_a / wo_b) + its side-stream prefetch state. Host bookkeeping
    in issue order, no device syncs: owner = the _Dq whose exact bf16 the slot holds (or will hold once `ready` fires on the
    side stream); last = the _Dq that read the slot before (the learned call order)."""

    def __init__(self, dev, numel: int, side):
        self.buf = torch.empty(numel, dtype=torch.bfloat16, device=dev)
        self.side, self.owner, self.ready, self.last = side, None, None, None


class _Dq:
    """weight(): the slot's view -- prefetched (the current stream waits on its side-stream
    event) or dequantized now on the current stream (after any pending side write). done(), right after the GEMM's launch:
    with ENG_DQ_PF the weight that followed this one last time is dequantized into the slot on the side stream once the
    current stream has passed the GEMM (an event), so the memory-bound dequant runs under the next compute-bound work. A
    wrong guess only costs a synchronous dequant at the next call (owner check)."""

    def __init__(self, g, dev, kind: str):
        self.g, self.slot = g, _DQ_DEV[str(torch.device(dev))][kind]
        self.N, self.K = g.packed.shape[1] * 256, g.packed.shape[0] * 128
        self.nxt = None

    def _view(self):
        return self.slot.buf[: self.N * self.K].view(self.N, self.K)

    def weight(self):
        import w8dq as W8DQ
        sl, w = self.slot, self._view()
        cur = torch.cuda.current_stream(w.device)
        if sl.last is not None and sl.last is not self:
            sl.last.nxt = self
        sl.last = self
        if sl.ready is not None:   # a side-stream dequant into this slot is pending: order after it
            cur.wait_event(sl.ready)
            sl.ready = None
        if sl.owner is not self:
            W8DQ.ext().dequant_packed(self.g.packed, w)
            sl.owner = self
        return w

    def done(self):
        nx, sl = self.nxt, self.slot
        if not DQ_PF or nx is None or nx is self:
            return
        import w8dq as W8DQ
        dev = sl.buf.device
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(dev))   # after the GEMM's reads of the slot
        with torch.cuda.stream(sl.side):
            sl.side.wait_event(ev)
            W8DQ.ext().dequant_packed(nx.g.packed, nx._view())
            rd = torch.cuda.Event()
            rd.record(sl.side)
        sl.owner, sl.ready = nx, rd


class _WoW8:
    """ENG_WO_W8A16: one packed fp8 weight (w8a16.Gemm) called as attn.py's w8["wo_a"] / w8["wo_b"] (x, y).
    <= 128 rows: w8a16 with the Gemm's split-K (w8a16.cu allows split-K only within one 128-row token slice; ONE S per Gemm:
    its split-K counters are monotonic).
    Other rows: w8a16
    with splits 1. Calls in stream order on the rank's stream (as w8["wq_b"])."""

    def __init__(self, g, grouped: bool = False, dq=None):
        self.g, self.grouped, self.dq = g, grouped, dq

    def __call__(self, x, y):
        g = self.g
        R = x.shape[0]
        if self.dq is not None and R >= DQ_MINR:
            w = self.dq.weight()
            if self.grouped:   # = attn.py's torch.bmm(o3.view(R, 8, 4096).transpose(0, 1), W["wo_aT"], out=z (R, 8, 1024)^T)
                torch.bmm(x.view(R, 8, 4096).transpose(0, 1), w.view(8, 1024, 4096).transpose(1, 2),
                          out=y.view(R, 8, 1024).transpose(0, 1))
            else:              # = attn.py's torch.mm(ws["z"], W["wo_b"].t(), out=ws["y"])
                torch.mm(x, w.t(), out=y)
            self.dq.done()
            return y
        g.ext.gemm(x, g.packed, y, g.rpg, g.kxg, g.splits if R <= 128 else 1, None, None, None, None, None, 0, 0,
                   None, g.ws, g.cnt, None)
        return y


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
        if C1_K1L and C1_K1L not in cfg.buckets:   # FPR_C1_KMAX: the C1 lane bucket (1 + k_max rows, one sequence)
            import dataclasses as _dc
            cfg = _dc.replace(cfg, buckets=tuple(sorted(set(cfg.buckets) | {C1_K1L})))
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
        if MAX_TAIL_ROWS > 0:   # ENG_MAX_TAIL_ROWS
            self.max_tail_rows = min(self.max_tail_rows, -(-MAX_TAIL_ROWS // cfg.mixed_round) * cfg.mixed_round)
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
        if EA.DENSE8:   # ENG_C128_DENSE8: no full bf16 wo_aT / wo_b on the GPU; every reader must have another source
            assert C1_TPWO and not EA.C128A and not EA.SKW8_DENSE and cfg.w8a16 == "q_b", (
                "ENG_C128_DENSE8 needs ENG_C1_TPWO=1 (decode buckets <= ENG_C1_TPWO_MAXR on this rank's bf16 quarter), "
                "w8a16='q_b', ENG_C128A=0 and no ENG_SKW8_DENSE (their paths read the full bf16 wo copies)")
            self.log(f"engine: ENG_C128_DENSE8 on: wo_a / wo_b above {C1_TPWO_MAXR} decode rows and in mixed / prefill steps on "
                     "production's 8-bit arm; bf16 quarter copies for C1")
            from vllm.utils import deep_gemm as _vdg   # (see _A8_LOCK: DeepGEMM's lazy init here, on this thread, once)
            getattr(_vdg, "_lazy_init", lambda: None)()
            for d_ in self.devs:
                with torch.cuda.device(d_):
                    EA._a8().dg_num_sms()
            EA.DENSE8_MAXR = C1_TPWO_MAXR   # (ENG_C128_DENSE8_ADD items switch at the same row count as wo)
            d8add = sorted(getattr(EA, "DENSE8_ADD", ()))
            if d8add:
                self.log(f"engine: ENG_C128_DENSE8_ADD {d8add} on the same arm and rows (copies ADDED: q_a|wkv "
                         "+0.37, q_b +1.68, shared expert +0.36 GB per GPU)")
        if WO_W8A16:   # ENG_WO_W8A16 (R2): no full bf16 wo_aT / wo_b on the GPU; every reader must have another source
            assert not EA.DENSE8 and not EA.C128A and not EA.SKW8_DENSE and cfg.w8a16 == "q_b", (
                "ENG_WO_W8A16 needs ENG_C128_DENSE8=0 (its own wo arm), ENG_C128A=0, no ENG_SKW8_DENSE (their paths read the "
                "full bf16 wo copies) and w8a16='q_b' (the W8A16 extension and W['w8'])")
            self.log(f"engine: ENG_WO_W8A16 on: wo_a / wo_b W8A16 from one fp8 copy (split-K {WO_W8A16_S} at <= 128 rows); "
                     + ("NO bf16 quarters (ENG_WO_ONECOPY): TP-wo on this rank's views of the packed copy (w8a16tp, splits "
                        f"{WO_ONECOPY_S})" if WO_ONECOPY else "bf16 quarter copies for C1 / TP-wo"))
        if WO_ONECOPY:   # ENG_WO_ONECOPY: refuse at startup while any reader of the bf16 quarters remains
            b16_ = set(x for x in os.environ.get("C1M_B16", "").split(",") if x)   # the fallbacks would index AW["wo_aT"],
            bad = []                                                                 # which R2 never loads
            if not WO_W8A16:
                bad.append("ENG_WO_W8A16=0 (the packed copy is R2's)")
            if EA.DENSE8:
                bad.append("ENG_C128_DENSE8=1 (keeps the quarters)")
            if C1_TPWO_F8 or EA.SKW8_DENSE:
                bad.append("ENG_C1_TPWO_F8 / ENG_SKW8_DENSE (their own fp8 copies)")
            if C1_MEGA and b16_ & {"wo_a", "wo_b"}:
                bad.append(f"c1mega with C1M_B16 {sorted(b16_)} (bf16 wo_a / wo_b quarters; C1 needs a package that reads the "
                           "packed copy)")
            if C1_TPWO and C1_TPWO_MAXR * W > 128:
                bad.append(f"ENG_C1_TPWO_MAXR {C1_TPWO_MAXR} (TP-wo rows {C1_TPWO_MAXR * W} > 128: w8a16tp's split-K needs one "
                           "128-row token slice)")
            if C1_MEGA and not {"woa8", "wob8"} <= b16_:
                bad.append(f"c1mega with C1M_B16 {sorted(b16_)} without woa8 / wob8 (it would build its own fp8 wo copies, "
                           "+0.75 GB / GPU; srv8x reads the packed views)")
            if C1_MEGA_AB:
                bad.append("ENG_C1_MEGA_AB (the per-arm C1M_B16 sets are not checked here)")
            assert not bad, "ENG_WO_ONECOPY: bf16 wo-quarter readers remain: " + "; ".join(bad)
            # TP-wo (ENG_C1_TPWO=1) reads this rank's views of the packed copy (AW["pk8tp"], loaded below) on w8a16tp; with
            # ENG_C1_TPWO=0 the decode buckets run R2's full W8A16 (attn.py's w8 path) and need no views
            if C1_TPWO:   # w8a16tp built / loaded and its workspace + counters allocated now, never inside a step / capture
                EA.w8tp()   # (release prebuild: eng_w8a16tp_<W8A16_DEFS>_w8_smax16)
                for d_ in self.devs:
                    with torch.cuda.device(d_):
                        EA._w8tp_ws(d_)
        if (WO_W8A16 and WO_W8A16_DQ) or QB_DQ:   # the extension + per-device slots / side stream, on the main thread
            assert cfg.w8a16 == "q_b", "ENG_WO_W8A16_DQ / ENG_QB_DQ read the W8A16 packed copies (w8a16='q_b')"
            import w8dq as _W8DQ
            _W8DQ.ext()
            for d_ in self.devs:
                side = torch.cuda.Stream(d_)
                sl = {}
                if QB_DQ:
                    sl["q"] = _DqSlot(d_, 32768 * 1280, side)
                if WO_W8A16 and WO_W8A16_DQ:
                    sl["a"], sl["b"] = _DqSlot(d_, 8192 * 4096, side), _DqSlot(d_, 5120 * 8192, side)
                _DQ_DEV[str(torch.device(d_))] = sl
            EA.QB_DQ, EA.DQ_MINR = QB_DQ, DQ_MINR
            mb = sum(v.buf.numel() * 2 for v in _DQ_DEV[str(torch.device(self.devs[0]))].values()) / 2 ** 20
            self.log(f"engine: exact-dequant route at >= {DQ_MINR} rows: wo {bool(WO_W8A16 and WO_W8A16_DQ)} q_b {QB_DQ} "
                     f"prefetch {DQ_PF}; slots {mb:.0f} MiB per GPU (bitwise with the bf16 copies' cuBLAS calls)")
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
        W8.build(prime_devs=[d.index for d in self.devs])   # w8a16 prime (no lazy load in captures)
        AH._ext()
        if MOE_X2:   # ENG_MOE_X2: load / JIT-build kernels/moe_x2 here with the other extensions, never mid-serving (g-60)
            self.moe_x2()
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
                if EA.DENSE8 or WO_W8A16:   # ENG_C128_DENSE8 / ENG_WO_W8A16: the entry from the host; the full bf16 wo_aT / wo_b never reach the GPU
                    ent = dict(Wt.load(f"attn_L{L}_v1", lambda L=L: Wt.build_attn(L), None))
                    wo_aT_h, wo_b_h = ent.pop("wo_aT"), ent.pop("wo_b")
                    ent = {k: (v.to(dev) if isinstance(v, torch.Tensor) and k != "wq_b" else v) for k, v in ent.items()}
                else:
                    ent = Wt.load(f"attn_L{L}_v1", lambda L=L: Wt.build_attn(L), dev)
                qb16 = ent.get("wq_b") if ((QB16_LOAD and L <= 20) or QB16_DECODE) else None   # (finish_attn drops it)
                raw_qb = ent.get("_raw_wq_b") if "qb" in getattr(EA, "DENSE8_ADD", ()) else None   # (host e4m3 + scales; finish_attn pops it)
                AW = Wt.finish_attn(ent, dev, css[r][RATIO[L]], cfg.w8a16)
                if EA.DENSE8:   # this rank's bf16 quarter for C1 + the folded e4m3 wo_a / wo_b (A8c)
                    AW["wo_aT_c1"] = wo_aT_h[2 * r:2 * r + 2].contiguous().to(dev)              # [2][4096][1024]
                    AW["wo_b_c1"] = wo_b_h[:, 2048 * r:2048 * (r + 1)].contiguous().to(dev)     # [5120][2048]
                    del wo_aT_h, wo_b_h
                    e8 = Wt.load(f"attn8_L{L}_v1", lambda L=L: Wt.build_attn8(L), None)   # (host) checkpoint e4m3 + scales
                    src8 = {k: e8[k] for k in ("wo_a", "wo_b")}
                    if "qkv" in getattr(EA, "DENSE8_ADD", ()):   # ENG_C128_DENSE8_ADD: q_a | wkv rows as the bf16 copy concatenates them
                        src8["qkv"] = e8["wqkv_a"]
                    if raw_qb is not None:
                        src8["q_b"] = raw_qb
                    with _A8_LOCK:   # (DeepGEMM / vLLM lazy init: not thread-safe)
                        P8 = EA._a8().prepare({k: tuple(t.to(dev) for t in v) for k, v in src8.items()}, woa=True)
                    woa_ref = P8.pop("_woa_ref", None)   # (plain-layout copy of the folded wo_a: references only)
                    if woa_ref is not None:   # its plain fp32 [64][32] block scales (8 KB) for readers of the bytes (c1mega)
                        P8["wo_a_s128"] = woa_ref[1]
                    del woa_ref
                    P8.pop("_stats", None)
                    AW["A8c"] = P8
                    del e8, src8, raw_qb
                if WO_W8A16:   # ENG_WO_W8A16 (R2): this rank's bf16 quarter for C1 / TP-wo + the W8A16 wo_a / wo_b
                    if not WO_ONECOPY:   # (ENG_WO_ONECOPY: no quarters; views of the packed copy below)
                        AW["wo_aT_c1"] = wo_aT_h[2 * r:2 * r + 2].contiguous().to(dev)              # [2][4096][1024]
                        AW["wo_b_c1"] = wo_b_h[:, 2048 * r:2048 * (r + 1)].contiguous().to(dev)     # [5120][2048]
                    del wo_aT_h, wo_b_h
                    import w8a16 as W8
                    e8 = Wt.load(f"attn8_L{L}_v1", lambda L=L: Wt.build_attn8(L), None)   # (host) checkpoint e4m3 + scales
                    (wa, sa), (wb, sb) = e8["wo_a"], e8["wo_b"]
                    ext8 = AW["w8"]["wq_b"].ext
                    ga_ = W8.Gemm(ext8, wa.to(dev), sa.to(dev), WO_W8A16_S[0], rows_per_group=1024, kx_per_group=4096)
                    gb_ = W8.Gemm(ext8, wb.to(dev), sb.to(dev), WO_W8A16_S[1])
                    dqa_ = _Dq(ga_, dev, "a") if WO_W8A16_DQ else None
                    dqb_ = _Dq(gb_, dev, "b") if WO_W8A16_DQ else None
                    AW["w8"]["wo_a"], AW["w8"]["wo_b"] = _WoW8(ga_, True, dqa_), _WoW8(gb_, False, dqb_)
                    if WO_ONECOPY and C1_TPWO:   # TP-wo's slices = views of the ONE packed copy (= W8.repack of the slices, byte for byte:
                        pa_, pb_ = ga_.packed, gb_.packed   # wo_a tiles 8r .. 8r + 7 (groups 2r, 2r + 1), wo_b K-chunks 16r ..
                        assert tuple(pa_.shape) == (32, 32, W8.CHUNK) and tuple(pb_.shape) == (64, 20, W8.CHUNK), (pa_.shape, pb_.shape)
                        AW["pk8tp"] = dict(wo_a=pa_[:, 8 * r:8 * r + 8], wo_b=pb_[16 * r:16 * r + 16])
                    del e8, wa, sa, wb, sb
                if QB_DQ:   # prefill q_b through the exact dequant of the packed copy (attn.py's QB_DQ branch)
                    AW["w8"]["wq_b_dq"] = _Dq(AW["w8"]["wq_b"], dev, "q")
                if qb16 is not None:
                    AW["wq_b16"] = qb16.to(dev)
                if EA.SKW8_DENSE or (C1_TPWO and C1_TPWO_F8):   # packed fp8 copies for skw8 (decode
                    import w8a16 as W8                         # buckets of <= 32 rows)
                    e8 = Wt.load(f"attn8_L{L}_v1", lambda L=L: Wt.build_attn8(L), None)   # host; only the
                    AW["pk8"] = {k: W8.repack(*(t.to(dev) for t in e8[k])) for k in EA.SKW8_DENSE}   # used ones to the GPU
                    if C1_TPWO and C1_TPWO_F8:   # this GPU's TP-wo slices: wo_a groups 2r, 2r+1; wo_b K columns 2048r ..
                        (wa, sa), (wb, sb) = e8["wo_a"], e8["wo_b"]
                        AW["pk8tp"] = dict(
                            wo_a=W8.repack(wa[2048 * r:2048 * (r + 1)].contiguous().to(dev), sa[64 * r:64 * (r + 1)].contiguous().to(dev)),
                            wo_b=W8.repack(wb[:, 2048 * r:2048 * (r + 1)].contiguous().to(dev), sb[:, 64 * r:64 * (r + 1)].contiguous().to(dev)))
                    del e8
                AW["L"] = L
                if PF_ACT8_DENSE_LOAD and L <= 20:
                    A8 = EA._a8()
                    with _A8_LOCK:   # (DeepGEMM / vLLM lazy init: not thread-safe)
                        AW["A8"] = A8.prepare(Wt.load(f"act8_L{L}_raw_v1", lambda L=L: A8.build_raw(L), dev))
                assert AW["ratio"] == RATIO[L], (L, AW["ratio"])
                ex = Wt.load(f"L{L}_r{r}_raw_experts_v1", lambda L=L: Wt.build_raw_experts(L, r), dev)
                raw_m = Wt.load(f"mhc_L{L}_v1", lambda L=L: Wt.build_mhc(L), dev)
                m = Wt.finish_mhc(raw_m, self.hext)
                if self.mhcf is not None:
                    m["fa_f"], m["ff_f"] = self.mhcf.prep_f(raw_m["fa_raw"]), self.mhcf.prep_f(raw_m["ff_raw"])
                dn = Wt.load(f"L{L}_r{r}_dense_v1", lambda L=L: Wt.build_dense(L, r), dev)
                if "sh" in getattr(EA, "DENSE8_ADD", ()):   # ENG_C128_DENSE8_ADD sh: this rank's shared-expert slice, folded e4m3 (p_moe)
                    d8 = Wt.load(f"L{L}_r{r}_dense8_v1", lambda L=L: Wt.build_dense8(L, r), None)
                    with _A8_LOCK:   # (DeepGEMM / vLLM lazy init: not thread-safe)
                        P8s = EA._a8().prepare({k: tuple(t.to(dev) for t in d8[k]) for k in ("sh13", "sh2")})
                    P8s.pop("_stats", None)
                    dn = dict(dn, A8sh=P8s)
                    del d8
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
        self.streams = [torch.cuda.Stream(device=d, priority=STREAM_PRIO) for d in self.devs]   # ENG_STREAM_PRIO
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
        self.bk_view = {}       # ENG_MIX_ONE_BUCKET: (R, 'prod', 'A') -> [per-rank view states over the max A bucket]
        self.bk_tabs = None     # ENG_MIX_ONE_BUCKET: per rank, the fork-MoE tables shared by the views that need them
        self.bk_keys0 = {}      # ENG_MIX_ONE_BUCKET: view key -> per-rank key snapshots at creation (lazy-bytes accounting)
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
        # ENG_DECODE_PRECREATE: the decode buckets still to create (largest first; None: off or done), this step serial's
        # decision (serial, forced R or None), steps forced per R, and the start (serial, time, GPU 0 allocated bytes)
        self._pc = sorted((b for b in self.cfg.buckets if b != C1_K1L), reverse=True) if DECODE_PRECREATE else None
        self._pc_s, self._pc_f, self._pc_n, self._pc_t0 = None, None, {}, None
        # ENG_DECODE_VIEWS: per rank, key -> the flat base tensor (allocated at the first view, sized for _dv_rmax rows)
        self._dvb = [dict() for _ in self.devs]
        self._dv_rmax = max(b for b in self.cfg.buckets if b != C1_K1L)
        self.gpool = []         # one graph memory pool per device, shared by the buckets' graphs (replayed one at a time)
        for d in self.devs:
            with torch.cuda.device(d):
                self.gpool.append(torch.cuda.graph_pool_handle())
        self.hist = {}          # rid -> host token ids (prompt + output) for Engram's prefill lookback (ctx.new + results)
        self.hooks = Hooks(self)
        self._tix = None
        self._ensure_hooks = []            # ENG_MIX_ENSURE: callables (st) run at mixed bucket / view creation
        if MIX_PERSIST or TAIL_GATHER:
            self._persist_init()
        self.hsd = self.hooks.hsd          # (the adapter also looks for backend.hsd)
        _hostmap(self)
        if C1_MEGA:                        # ENG_C1_MEGA: the adapter and its fp8 TP slices now (before free_bytes() sizes the pools)
            if C1_MEGA_AB:
                self.c1m()                 # ENG_C1_MEGA_AB: both arms built AND weighted in _c1m_load
            else:
                self.c1m()._weights()
        self.pf_ws = None
        if PF_WS_MB > 0 and ATTN_PF:       # ENG_PF_WS_PREALLOC_MB: before free_bytes() sizes the pools
            self.pf_ws = [torch.empty(PF_WS_MB * 2**20 // 2, dtype=torch.bfloat16, device=d) for d in self.devs]
            self.log(f"engine: ENG_PF_WS_PREALLOC_MB {PF_WS_MB}: prefill-attention workspace preallocated on every GPU")
        self.mix_pool = None               # ENG_MIX_POOL_GB / _SPLIT: pool key ("*" one pool, "A" / "T" per role) -> per GPU
        if MIX_POOL_GB > 0 or MIX_POOL_SPLIT:   # before free_bytes() sizes the pools
            sizes = dict(zip(("A", "T"), MIX_POOL_SPLIT)) if MIX_POOL_SPLIT else {"*": MIX_POOL_GB}
            self.mix_pool, self.mix_pool_resv = {}, {}
            for pk_, gb_ in sizes.items():
                ps_ = []
                for d in self.devs:
                    with torch.cuda.device(d):     # a MemPool registers itself on the CURRENT device: create it on d
                        p_ = torch.cuda.MemPool(use_on_oom=True)
                    with torch.cuda.device(d), torch.cuda.use_mem_pool(p_, device=d):
                        t_ = torch.empty(int(gb_ * 1e9), dtype=torch.uint8, device=d)
                        sm_ = [torch.empty(512 * 1024, dtype=torch.uint8, device=d) for _ in range(128)]   # small blocks
                    del t_, sm_
                    ps_.append(p_)
                self.mix_pool[pk_] = ps_
                self.mix_pool_resv[pk_] = [self._pool_bytes(pk_, r)[0] for r in range(W)]
            self.log(f"engine: ENG_MIX_POOL_GB: mixed-bucket pools reserved per GPU (GB) " +
                     ", ".join(f"{k}: {'/'.join(f'{b / 1e9:.2f}' for b in v)}" for k, v in self.mix_pool_resv.items()) +
                     (" (one bucket per role)" if MIX_POOL_SPLIT else f" (strict cap {self._mix_cap() / 1e9:.2f} GB)"))
        if MOE_PRELOAD:                    # ENG_MOE_PRELOAD: the MoE extensions now, never inside a step
            self._preload_moe()
        if ATTN_PRELOAD:                   # ENG_ATTN_PRELOAD: the attention / indexer ones too
            self._preload_attn()
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
    def bucket_of(self, n: int, pad: bool = True) -> int:
        for b in self.cfg.buckets:
            if n <= b:
                break
        else:
            raise ValueError(f"{n} rows per rank > largest bucket {self.cfg.buckets[-1]}")
        if pad and self._pc is not None and n > max(4, C1_K1L):   # (pad=False: long-row variants, debug lookups)
            return self._pc_pick(b)
        return b

    def _mix_precreate(self) -> None:
        """ENG_MIX_PRECREATE: every mixed onebk view (A, and T with ENG_MIX_ONE_BUCKET_T) for R = mixed_round .. the role's
        max, between steps with the GPUs idle (synced before / after). A failure only logs (views stay lazy)."""
        cfg = self.cfg
        rnd = lambda n: max(cfg.mixed_round, -(-n // cfg.mixed_round) * cfg.mixed_round)  # noqa: E731
        roles = [("A", rnd(cfg.max_mixed_rows))] + ([("T", rnd(self.max_tail_rows))] if MIX_ONE_BUCKET_T else [])
        for d_ in self.devs:
            torch.cuda.synchronize(d_)
        t0, a0, n0 = time.perf_counter(), torch.cuda.memory_allocated(self.devs[0]), len(self.bk_view)
        try:
            for role, rmax in roles:
                for R in range(cfg.mixed_round, rmax + 1, cfg.mixed_round):
                    self.bucket(R, "prod", role)
        except Exception as e_:  # noqa: BLE001
            self.log(f"engine: ENG_MIX_PRECREATE FAILED ({type(e_).__name__}: {str(e_)[:200]}): remaining views stay lazy")
        self._busy = set()
        for d_ in self.devs:
            torch.cuda.synchronize(d_)
        self.log(f"engine: ENG_MIX_PRECREATE: {len(self.bk_view) - n0} views created ({', '.join(f'{r} <= {m}' for r, m in roles)}), "
                 f"{len(self.bk_view)} in all; GPU 0 allocated +{(torch.cuda.memory_allocated(self.devs[0]) - a0) / 2**20:.0f} MiB in "
                 f"{time.perf_counter() - t0:.1f} s")
        self.stats["mix_precreate_views"] = len(self.bk_view) - n0

    def _c1_precapture(self) -> None:
        """ENG_C1_PRECAPTURE: c1mega's graph L per C1 owner at PRECREATE's done point. Loaded by file path (kernels/c1mega is the packages' parent: not put on sys.path, no shadowing)."""
        for d_ in self.devs:
            torch.cuda.synchronize(d_)
        res = None
        try:
            import importlib.util
            m_ = sys.modules.get("c1m_precapture")
            if m_ is None:
                sp_ = importlib.util.spec_from_file_location("c1m_precapture",
                                                             os.path.join(PHYS, "kernels/c1mega/c1m_precapture.py"))
                m_ = importlib.util.module_from_spec(sp_)
                sp_.loader.exec_module(m_)
                sys.modules["c1m_precapture"] = m_
            res = m_.precapture(self, kinds=C1_PRECAPTURE_KINDS, log=self.log)
        except Exception as e_:  # noqa: BLE001  (a failed precapture leaves the first-step captures, as before)
            self.log(f"engine: ENG_C1_PRECAPTURE FAILED ({type(e_).__name__}: {str(e_)[:200]}): C1 graphs stay first-step")
        for d_ in self.devs:
            torch.cuda.synchronize(d_)
        if res is not None:
            self.stats["c1_precaptured"] = len(res.get("captured", ()))

    def _h1_precapture(self) -> None:
        """Synced before and after; never raises."""
        self._h1_precap_due = False
        for d_ in self.devs:
            torch.cuda.synchronize(d_)
        try:
            res = _mxh().precapture(self, log=self.log)
            self.stats["h1_precaptured"] = len(res.get("captured", ()))
        except Exception as e_:  # noqa: BLE001  (a failed precapture leaves the first-use captures, as before)
            self.log(f"engine: ENG_MIXED_HEAD_PRECAPTURE FAILED ({type(e_).__name__}: {str(e_)[:200]}): H1 keys stay first-use")
        for d_ in self.devs:
            torch.cuda.synchronize(d_)

    def precreate_active(self) -> bool:
        """ENG_DECODE_PRECREATE has not reached its done point (or give-up) yet. The native loop (nd) runs decode steps
        embedded while this holds, so bucket_of / _pc_pick decide as in Python mode. False when PRECREATE is off or done."""
        return self._pc is not None

    def warmup_due(self) -> bool:
        """the exporter's decode-hold condition: precreate still deciding, or (ENG_NATIVE_BOUNDARY only: native_boundary()
        is what clears it in the native loop) an H1 precapture its done point marked due. Without the flag an H1-due would
        never clear natively and would hold decode embedded forever, so it is not counted."""
        return (self._pc is not None or (NATIVE_BOUNDARY and bool(getattr(self, "_h1_precap_due", False)))
                or getattr(self, "_precap_job", None) is not None)   # (ENG_PRECAP_DEFER: the job pending holds decode too)

    def _d13_idle(self) -> bool:
        """D13 has no step whose results are not extracted yet (the in-flight handle and ENG_D13S's deferred owners)"""
        d = getattr(self, "_d13_obj", None)
        if d is None:
            return True
        h = d.inflight
        if h is not None and not h.done():
            return False
        sd = getattr(d, "steady", None)
        for o in (getattr(sd, "res_owner", None) or ()):
            if o is not None and not o.done():
                return False
        return True

    def _precap_headroom(self) -> list:
        """(ENG_PRECAP_DEFER) per-GPU free GB after synchronize + empty_cache (= smi free: torch slack is ~0 after empty_cache)"""
        out = []
        for d_ in self.devs:
            torch.cuda.synchronize(d_)
        torch.cuda.empty_cache()
        for d_ in self.devs:
            f_, _t = torch.cuda.mem_get_info(d_)
            out.append(f_ / 1e9)
        return out

    def _precap_try(self, job: dict, out: dict) -> None:
        """(ENG_PRECAP_DEFER) one boundary of the pending job: count it; on an attempt boundary measure headroom and run / wait /
        give up (see the flag). Runs with D13 idle (native_boundary's lock + _d13_idle)."""
        job["nb"] += 1
        nb, dt = job["nb"], time.perf_counter() - job["t0"]
        late = dt >= PRECAP_MAX_WAIT_S or nb >= PRECAP_MAX_WAIT_B
        if not late and (nb < PRECAP_MIN_B or (nb - PRECAP_MIN_B) % PRECAP_EVERY_B):
            return
        hr = self._precap_headroom()
        self.stats["precap_attempts"] = self.stats.get("precap_attempts", 0) + 1
        hs = "/".join(f"{x:.2f}" for x in hr)
        emb = self.step_serial - job["s0"]            # step serials during the hold
        est = self._precap_estimate(job)
        need = max(PRECAP_HARD_MIN_GB, est + PRECAP_MARGIN_GB)
        if min(hr) >= max(PRECAP_MIN_FREE_GB, need) or (late and min(hr) >= need):
            self.log(f"engine: ENG_PRECAP_DEFER: running the precapture job ({nb} boundaries / {dt:.1f} s after done, "
                     f"{'deadline' if late else 'headroom'}): headroom GB per GPU before {hs} (smi free after empty_cache); "
                     f"estimated job {est:.2f} GB (+ margin {PRECAP_MARGIN_GB} GB)")
            t1 = time.perf_counter()
            if job["mix"]:
                self._mix_precreate()
            if job["c1"]:
                self._c1_precapture()
            cb = getattr(self, "_precap_extra", None)   # (core2b_proc: nd's precapture_all: decode buckets' graph T / L)
            if cb is not None:
                try:
                    cb()
                    out["ran"].append("precap_nd")
                except Exception as e_:  # noqa: BLE001  (its keys stay first-use; logged)
                    self.log(f"engine: ENG_PRECAP_DEFER: nd precapture FAILED ({type(e_).__name__}: {str(e_)[:200]})")
            self._precap_job = None
            if job["h1"]:
                self._h1_precap_due = True       # native_boundary runs it right after this (same boundary)
            out["ran"].append("precap_job")
            self.stats["precap_done_s"] = round(dt, 1)
            self.stats["precap_done_b"] = nb
            self._precap_after = (t1, hs, nb, dt, emb, job["h1"])
            self._precap_est = est
            if not job["h1"]:
                self._precap_finish_line()
            return
        if late:
            self._precap_job = None
            self.stats["precap_defer_gave_up"] = 1
            self.log(f"engine: ENG_PRECAP_DEFER GAVE UP: min-GPU headroom {min(hr):.2f} GB < {need:.2f} GB (estimated job {est:.2f} + "
                     f"margin {PRECAP_MARGIN_GB}) at the deadline "
                     f"({nb} boundaries / {dt:.1f} s after done; per GPU {hs}): the hold is RELEASED WITHOUT the precapture job "
                     f"(MIX views / C1 / H1 stay first-use captures); {emb} step serials during the hold")
            out["ran"].append("precap_gave_up")
            return

    def _precap_estimate(self, job: dict) -> float:
        """the job's GB per GPU, conservative (see the flags)"""
        if PRECAP_EST_GB > 0:
            return PRECAP_EST_GB
        g = 0.0
        if job.get("h1"):
            nk = int(os.environ.get("ENG_MIXED_HEAD_GRAPHS", "0") or 0) or 2400
            g += nk * PRECAP_H1_MIB_PER_KEY / 1024
            if os.environ.get("ENG_DECODE_HEAD_PRECAPTURE", "0") == "1":
                g += PRECAP_DECH_GB
        if job.get("mix"):
            try:
                rnd = lambda n: max(self.cfg.mixed_round, -(-int(n) // self.cfg.mixed_round) * self.cfg.mixed_round)   # noqa: E731
                nv = rnd(self.cfg.max_mixed_rows) // self.cfg.mixed_round - 1 + rnd(self.max_tail_rows) // self.cfg.mixed_round - 1
            except Exception:  # noqa: BLE001
                nv = 77
            g += max(0, nv - len(getattr(self, "bk_view", {}) or {})) * PRECAP_VIEW_MIB / 1024
        if getattr(self, "_precap_extra", None) is not None:
            g += PRECAP_ND_GB
        return g

    def _precap_finish_line(self) -> None:
        """(ENG_PRECAP_DEFER) the one final line the switch wrapper waits on, with the after-headroom"""
        t1, hs, nb, dt, emb, _h1 = getattr(self, "_precap_after", (time.perf_counter(), "?", 0, 0.0, 0, False))
        ha = "/".join(f"{x:.2f}" for x in self._precap_headroom())
        self.log(f"engine: ENG_PRECAP_DEFER: precapture done in {time.perf_counter() - t1:.1f} s ({nb} boundaries / {dt:.1f} s after "
                 f"the done point; {emb} step serials during the hold): headroom GB per GPU before {hs} -> after {ha}; estimated "
                 f"{getattr(self, '_precap_est', 0.0):.2f} GB")
        self._precap_after = None
        self._w8wide_line("precapture done")

    def _w8wide_line(self, where: str) -> None:
        """every loaded w8a16 extension (attn_dense builds, engram's): per-device
        wide_ok (init status) and the host-side wide launch / fallback counters (counted at launch and at graph capture). Log-only."""
        try:
            mods = [(k, v) for k, v in list(sys.modules.items())
                    if (k.startswith("w8a16_ext") or k.startswith("engram_w8a16_ext")) and hasattr(v, "wide_launches")]
            if mods:
                self.log("engine: w8a16 wide (" + where + "): " + "; ".join(
                    f"{k} ok {[bool(v.wide_ok(int(d.index))) for d in self.devs]} err {[int(v.wide_err(int(d.index))) for d in self.devs]}"
                    f" launches {int(v.wide_launches())} fallbacks {int(v.wide_fallbacks())}" for k, v in mods))
        except Exception as e_:  # noqa: BLE001  (log-only)
            self.log(f"engine: w8a16 wide ({where}): unavailable ({type(e_).__name__}: {e_})")

    def native_boundary(self) -> dict:
        """ENG_NATIVE_BOUNDARY: the native loop's step boundary hook (see the flag). Returns dict(ran, deferred,
        precreate_active). Never raises (_h1_precapture logs its own failure)."""
        out = dict(ran=[], deferred=False, precreate_active=self._pc is not None)
        stall = (self._pc is not None and self._pc_s is not None and DECODE_PRECREATE_STALL > 0
                 and self.step_serial - self._pc_s > DECODE_PRECREATE_STALL)
        job = getattr(self, "_precap_job", None)
        if not NATIVE_BOUNDARY or not (stall or getattr(self, "_h1_precap_due", False) or job is not None):
            return out
        d = getattr(self, "_d13_obj", None)
        if d is not None:
            d.lock.acquire()
        try:
            if not self._d13_idle():
                self.stats["nb_deferred"] = self.stats.get("nb_deferred", 0) + 1
                out["deferred"] = True
                return out
            if stall:                          # (marks _h1_precap_due when MIXED_HEAD_PRECAPTURE: handled just below)
                for d_ in self.devs:
                    torch.cuda.synchronize(d_)
                self._pc_stall_check()
                out["ran"].append("precreate_stalled")
                out["precreate_active"] = self._pc is not None
            job = getattr(self, "_precap_job", None)
            if job is not None:
                self._precap_try(job, out)        # (ENG_PRECAP_DEFER) may run the job (which marks H1 due) or give up
            if not getattr(self, "_h1_precap_due", False) or getattr(self, "_precap_job", None) is not None:
                return out
            self.log(f"engine: ENG_NATIVE_BOUNDARY: H1 precapture at the step boundary (step serial {self.step_serial}, "
                     f"{self.stats.get('nb_deferred', 0)} boundaries deferred)")
            self._h1_precapture()
            if d is not None:                  # as D13._fallback: a synchronous rewrite of HSD's per-step state / tables
                d._epoch += 1
                d._r_empty = [False] * W
                d._own_last = [None] * W
                if d.steady is not None:
                    d.steady.prev_sig = None
            self.stats["nb_precaptures"] = self.stats.get("nb_precaptures", 0) + 1
            out["ran"].append("h1")
            if getattr(self, "_precap_after", None) is not None:   # (ENG_PRECAP_DEFER) the job's H1 part done: the final line
                self._precap_finish_line()
            return out
        finally:
            if d is not None:
                d.lock.release()

    def _pc_pick(self, b: int) -> int:
        """ENG_DECODE_PRECREATE: this step's decode bucket (>= b, its natural bucket). One decision per step serial (D13's
        dispatch asks before its bump_serial: it then shares the previous serial's decision, still one per call)."""
        s = self.step_serial
        if s == self._pc_s:
            return b if self._pc_f is None or self._pc_f < b else self._pc_f
        self._pc_s = s
        if self._pc_t0 is None:
            self._pc_t0 = (s, time.perf_counter(), torch.cuda.memory_allocated(self.devs[0]))
        n_ = self._pc_n
        self._pc = [R for R in self._pc if not (R in self.bk and (n_.get(R, 0) == 0 or n_[R] >= DECODE_PRECREATE_REPEAT))]
        if not self._pc or s - self._pc_t0[0] > DECODE_PRECREATE_MAX_STEPS:
            self._pc_finish(s, "")
            return b
        f = self._pc[0] if self._pc[0] >= b else None    # the largest missing bucket (smaller ones: later steps)
        self._pc_f = f
        if f is None:
            return b
        n_[f] = n_.get(f, 0) + 1
        return f

    def _pc_finish(self, s: int, why: str) -> None:
        """ENG_DECODE_PRECREATE's done / GAVE UP point (from _pc_pick, or from native_boundary when stalled): log, then the
        done-point actions, then _pc = None. Called between steps (GPUs idle or synced by the actions)."""
        n_ = self._pc_n
        if True:
            raw = sorted(k for k in self.bk if not isinstance(k, tuple))
            self.log(f"engine: ENG_DECODE_PRECREATE {'done' if not self._pc else 'GAVE UP (missing ' + str(self._pc) + ')'}{why}: "
                     f"{len(raw)} decode buckets {raw}, {len(self.graphs)} graphs; {sum(n_.values())} padded steps over "
                     f"{s - self._pc_t0[0]} step serials, {time.perf_counter() - self._pc_t0[1]:.1f} s; GPU 0 allocated "
                     f"{self._pc_t0[2] / 1e9:.2f} -> {torch.cuda.memory_allocated(self.devs[0]) / 1e9:.2f} GB")
            self.stats["precreate_steps"] = sum(n_.values())
            if HSD_VIEWS and getattr(self.hooks, "hsd", None) is not None:   # every drafter bucket now (between steps,
                for d_ in self.devs:                                        # GPUs idle: precreate writes the views of the
                    torch.cuda.synchronize(d_)                              # shared base an in-flight step may use)
                try:
                    import hsd_views
                    hsd_views.precreate(self.hooks.hsd, log=self.log)
                except Exception as e_:  # noqa: BLE001  (a failed precreate only leaves the buckets lazy, as before)
                    self.log(f"engine: ENG_HSD_VIEWS precreate FAILED ({type(e_).__name__}: {str(e_)[:200]}): drafter buckets stay lazy")
                for d_ in self.devs:
                    torch.cuda.synchronize(d_)
            if CXX_PIECES and os.environ.get("ENG_MIXED_CXX_EARLY_DRV", "1") == "1":
                for d_ in self.devs:            # the C++ driver (and its owners' setup: mixed_cxx_ltwarm's fpr_lt warm, i.e.
                    torch.cuda.synchronize(d_)  # every cuBLASLt module the pool threads can use) now, GPUs idle, before any
                try:                            # pool thread exists, instead of at the first mixed step mid-serving
                    self._cxx_drv()
                except Exception as e_:  # noqa: BLE001  (the driver is then created lazily, as before)
                    self.log(f"engine: ENG_MIXED_CXX_EARLY_DRV FAILED ({type(e_).__name__}: {str(e_)[:200]}): lazy driver")
                for d_ in self.devs:
                    torch.cuda.synchronize(d_)
            if PRECAP_DEFER:                    # one memory-gated job at a later boundary (see ENG_PRECAP_DEFER)
                self._precap_job = dict(t0=time.perf_counter(), s0=self.step_serial, nb=0, 
                                        mix=bool(MIX_PRECREATE and MIX_ONE_BUCKET), c1=bool(C1_PRECAPTURE), h1=bool(MIXED_HEAD_PRECAPTURE))
                self.log(f"engine: ENG_PRECAP_DEFER: precapture job queued at the done point (step serial {self.step_serial}; mix "
                         f"{self._precap_job['mix']}, c1 {self._precap_job['c1']}, h1 {self._precap_job['h1']}): runs at a step boundary "
                         f"when min-GPU headroom >= {PRECAP_MIN_FREE_GB} GB (deadline {PRECAP_MAX_WAIT_S:.0f} s / {PRECAP_MAX_WAIT_B} "
                         f"boundaries: >= {PRECAP_HARD_MIN_GB} GB or GAVE UP)")
            else:
                if MIX_PRECREATE and MIX_ONE_BUCKET:
                    self._mix_precreate()
                if C1_PRECAPTURE:
                    self._c1_precapture()
                if MIXED_HEAD_PRECAPTURE:
                    self._h1_precap_due = True      # (runs at the next step_mixed's start: see ENG_MIXED_HEAD_PRECAPTURE)
            self._pc = None

    def _pc_stall_check(self) -> None:
        """ENG_DECODE_PRECREATE_STALL (see the flag): at a native step boundary, decide done / GAVE UP once no padding-eligible
        step came for DECODE_PRECREATE_STALL step serials. Buckets that exist count as created even below REPEAT."""
        if self._pc is None or self._pc_s is None or DECODE_PRECREATE_STALL <= 0:
            return
        s = self.step_serial
        if s - self._pc_s <= DECODE_PRECREATE_STALL:
            return
        self._pc = [R for R in self._pc if R not in self.bk]
        self.stats["precreate_stalled"] = self.stats.get("precreate_stalled", 0) + 1
        self._pc_finish(s, f" (stalled: no padding-eligible decode step for {s - self._pc_s} step serials)")

    def bucket(self, R: int, moe: str = "raw", role: str = "") -> list:
        """Per-rank state of a row bucket. moe 'raw': the fork's decode MoE (moe_raw route2r / gather5 / GEMMs, graphs);
        'prod': production's routed MoE (ds41h.moe.moe_forward_w4a16 from the ds41h package) for steps with prefill rows."""
        lvl = 0
        if moe == "raw" and R >= LVL_TAG:            # a long-row variant of decode bucket R % LVL_TAG
            lvl, R = R // LVL_TAG, R % LVL_TAG
        key = (R + lvl * LVL_TAG) if moe == "raw" else (R, moe, role)   # mixed: separate A (all rows) / T (tail rows) states
        if MIX_ONE_BUCKET and moe == "prod" and role == "A":
            Rmx = max(self.cfg.mixed_round, -(-self.cfg.max_mixed_rows // self.cfg.mixed_round) * self.cfg.mixed_round)
            if R < Rmx:
                return self._bucket_view(R, Rmx)
        if MIX_ONE_BUCKET_T and moe == "prod" and role == "T":
            RmxT = max(self.cfg.mixed_round, -(-self.max_tail_rows // self.cfg.mixed_round) * self.cfg.mixed_round)
            if R < RmxT:
                return self._bucket_view(R, RmxT, "T")
        if moe != "raw":
            self._busy.add(key)                       # in use by this step: never evicted by the step's next bucket
        if key in self.bk:
            if moe != "raw":
                self.bk[key] = self.bk.pop(key)       # LRU order
            return self.bk[key]
        if moe != "raw":
            mixed = [k for k in self.bk if isinstance(k, tuple) and k not in self._busy
                     and not (MIX_ONE_BUCKET and k[1] == "prod" and (k[2] == "A" or (MIX_ONE_BUCKET_T and k[2] == "T")))]   # ENG_MIX_ONE_BUCKET(_T): the max A (T) is pinned
            if self.mix_pool is not None:   # the last step's state stays referenced (self.last_st): evicting it frees nothing
                lst = getattr(self, "last_st", None)
                lk = lst[0].get("bkey") if lst else None
                mixed = [k for k in mixed if k != lk]
            if self.mix_pool is not None:   # strict: the same role's largest bucket's bytes per row (fixed overhead included)
                same = [k for k in self.bk_bytes if k[2] == role] or list(self.bk_bytes)
                kb = max(same, key=lambda k: k[0]) if same else None
                need = self.bk_bytes[kb] / kb[0] * R if kb else 1_000_000 * R
            else:
                per_row = max([self.bk_bytes[k] / k[0] for k in mixed if k in self.bk_bytes] or [650_000])
                need = per_row * R
            evicted = False
            held = lambda: sum(self.bk_bytes.get(k, 0) for k in self.bk if isinstance(k, tuple))  # noqa: E731
            if self.mix_pool is not None and MIX_POOL_SPLIT:   # one bucket per role: the role's previous one goes
                mixed = [k for k in mixed if k[2] == role]
                more = lambda: True  # noqa: E731
            else:
                more = lambda: (len([k for k in self.bk if isinstance(k, tuple)]) >= self.cfg.max_mixed_buckets or  # noqa: E731
                                held() + need > self._mix_cap())
            while mixed and more():
                old = mixed.pop(0)
                # its piecewise graphs reference the evicted state's buffers: retired, not destroyed (captured on the rank
                # streams: see _retired). _issue_mixed keys them (bucket key, layers, final, embed), so match on k[0]; a
                # plain pop(old) never matched, and a re-created bucket of the same key REPLAYED the old graphs against
                # the freed (after empty_cache: unmapped) buffers
                for mk_ in [k for k in self.mgraphs if k == old or (isinstance(k, tuple) and k and k[0] == old)]:
                    self._retired.append(self.mgraphs.pop(mk_))
                if self.mix_pool is not None:         # drop the state's tensor references now (reference cycles in the
                    for s_ in self.bk[old]:           # state dicts would hold them until the next gc run): the blocks
                        self._drop_state(s_)          # return to the pool before the next bucket is carved
                del self.bk[old]
                self.bk_bytes.pop(old, None)
                evicted = True
            if evicted:
                for d in self.devs:
                    torch.cuda.synchronize(d)
                if BISECT["evict_empty_cache"]:
                    torch.cuda.empty_cache()
                self.stats["mixed_bucket_evictions"] = self.stats.get("mixed_bucket_evictions", 0) + 1
                # every eviction with its time, so a later Xid 31 / illegal memory
                # access can be lined up with it (logging only)
                self.log(f"engine: mixed-bucket eviction #{self.stats['mixed_bucket_evictions']} at {time.time():.3f} "
                         f"for {key} (empty_cache {BISECT['evict_empty_cache']}); kept {sorted(k for k in self.bk if isinstance(k, tuple))}")
        m0 = torch.cuda.memory_allocated(self.devs[0])
        pk_ = (role if MIX_POOL_SPLIT else "*") if moe != "raw" and self.mix_pool is not None else None
        if pk_ is not None:   # ENG_MIX_POOL_GB / _SPLIT: carved from the reserved pool
            st = []
            for r in range(W):
                with torch.cuda.device(self.devs[r]), torch.cuda.use_mem_pool(self.mix_pool[pk_][r], device=self.devs[r]):
                    st.append(self._alloc(r, R, moe, lvl))
        else:
            st = [self._alloc(r, R, moe, lvl) for r in range(W)]
        for d in self.devs:
            torch.cuda.synchronize(d)   # flags / epochs zeroed and visible before any peer signals them
        self.bk[key] = st
        for s_ in st:
            s_["bkey"] = key
        if moe != "raw":                   # ensure time (ENG_MIX_PERSIST buffers, ENG_MIX_ENSURE hooks)
            self._ensure_new(st, role)
        if moe == "raw":   # one line per new decode bucket state: which R's a process creates, at what cost
            a_ = torch.cuda.memory_allocated(self.devs[0])
            self.log(f"engine: new decode bucket {key}: {(a_ - m0) / 1e9:.3f} GB on GPU 0 (allocated {a_ / 1e9:.2f} GB); "
                     f"{sum(1 for k in self.bk if not isinstance(k, tuple))} decode buckets, {len(self.graphs)} graphs, "
                     f"step serial {self.step_serial}"
                     f"{' (ENG_DECODE_PRECREATE)' if self._pc is not None and key == self._pc_f else ''}")
        if moe != "raw":
            self.bk_bytes[key] = torch.cuda.memory_allocated(self.devs[0]) - m0     # rank 0's share (ranks are alike)
            if pk_ is not None:
                grown = [self._pool_bytes(pk_, r)[0] - self.mix_pool_resv[pk_][r] for r in range(W)]
                if max(grown) > 0:   # the bucket(s) did not fit: the pool took new segments (logged, counted)
                    self.stats["mix_pool_grow"] = self.stats.get("mix_pool_grow", 0) + 1
                    self.log(f"engine: ENG_MIX_POOL_GB: pool grew ({pk_}) by {'/'.join(f'{g / 1e9:.2f}' for g in grown)} GB "
                             f"for bucket {key} ({self.bk_bytes[key] / 1e9:.2f} GB; held "
                             f"{sum(self.bk_bytes.get(k, 0) for k in self.bk if isinstance(k, tuple)) / 1e9:.2f} GB)")
                    self.mix_pool_resv[pk_] = [g + b for g, b in zip(grown, self.mix_pool_resv[pk_])]
        return st

    def _bucket_view(self, R: int, Rmx: int, role: str = "A") -> list:
        """ENG_MIX_ONE_BUCKET: bucket (R, 'prod', 'A') as cached view states over the pinned max A bucket (R < Rmx). The
        views are built once per R (stable objects: per-state caches such as the C++ driver's handles stay valid) and
        rebuilt only if the max bucket was ever re-carved. The fork MoE tables that a fresh R bucket would carry (Tall <=
        min(4 * prefill_moe_switch, 8192)) come from one shared set sized for the largest such R."""
        key = (R, "prod", role)
        self._busy.add(key)
        base = self.bucket(Rmx, "prod", role)
        hit = self.bk_view.get(key)
        if hit is not None and hit[0] is base:
            return hit[1]
        if self._has_fork_tables(W * R, "prod") and self.bk_tabs is None:
            cap = min(4 * self.cfg.prefill_moe_switch, 8192)
            Rt = max(self.cfg.mixed_round, cap // W // self.cfg.mixed_round * self.cfg.mixed_round)
            m0 = torch.cuda.memory_allocated(self.devs[0])
            self.bk_tabs = [self._fork_tables(r, W * Rt, "prod") for r in range(W)]
            for d in self.devs:
                torch.cuda.synchronize(d)   # counters / tickets zeroed and visible before first use
            self.log(f"engine: ENG_MIX_ONE_BUCKET: max A bucket R={Rmx}; fork-MoE tables for R <= {Rt} shared by the views "
                     f"({(torch.cuda.memory_allocated(self.devs[0]) - m0) / 1e9:.2f} GB per GPU)")
        st = [self._view_state(r, base[r], R) for r in range(W)]
        for s_ in st:
            s_["bkey"] = key
        self.bk_view[key] = (base, st)
        self._ensure_new(st, role)         # ensure time, before the key snapshot (none of it counts as lazy)
        self.bk_keys0[key] = [self._view_keys(s_) for s_ in st]
        self.stats["mix_views"] = self.stats.get("mix_views", 0) + 1
        self.log(f"engine: ENG_MIX_ONE_BUCKET: view R={R} {role} over R={Rmx} (#{len(self.bk_view)}); {self._onebk_str(detail=True)}")
        return st

    @staticmethod
    def _view_keys(s: dict) -> tuple:
        """a state's key snapshot: top level, ws, caches[L]"""
        return frozenset(s), frozenset(s["ws"]), {L: frozenset(c) for L, c in s["caches"].items()}

    @staticmethod
    def _state_ptrs(b: dict) -> set:
        """storage pointers of every tensor in a bucket state (top level, ws, caches[L], one level into containers)"""
        out = set()
        vals = list(b.values()) + list(b.get("ws", {}).values())
        for c in b.get("caches", {}).values():
            vals += list(c.values())
        for v in vals:
            for t in _tensors_of(v):
                out.add(t.untyped_storage().data_ptr())
        return out

    def _onebk_lazy(self):
        """ENG_MIX_ONE_BUCKET accounting: tensors steps added to the cached view states after their creation
        (keys outside bk_keys0's snapshot: top level, ws.*, caches.*). Plain buckets freed such tensors on eviction; views
        live forever, so each is growth per A size. Per GPU: bytes of storages owned by exactly one view (growth) and of
        storages several views share (references to common buffers); on GPU 0 the per-key and per-view split of the owned
        ones. Storages of the max A bucket itself are excluded. Returns (own[W], shared[W], {key: B}, {R: B})."""
        own, shared, per_key, per_view = [0] * W, [0] * W, {}, {}
        for r in range(W):
            seen, bptr = {}, {}
            for key, (base, st) in self.bk_view.items():
                snap = self.bk_keys0.get(key)
                if snap is None:
                    continue
                bp = bptr.get(id(base[r]))
                if bp is None:
                    bp = bptr[id(base[r])] = self._state_ptrs(base[r])
                s, (k0, w0, c0) = st[r], snap[r]
                items = [(k, s[k]) for k in s.keys() - k0]
                items += [("ws." + k, s["ws"][k]) for k in s["ws"].keys() - w0]
                for L, c in s["caches"].items():
                    items += [("caches." + k, c[k]) for k in c.keys() - c0.get(L, frozenset())]
                for name, v in items:
                    for t in _tensors_of(v):
                        stg = t.untyped_storage()
                        ptr = stg.data_ptr()
                        if not ptr or ptr in bp or stg.nbytes() > _LAZY_BIG:
                            continue
                        e = seen.setdefault(ptr, [stg.nbytes(), set(), name])
                        e[1].add(key[0])
            for nb, vs, name in seen.values():
                if len(vs) == 1:
                    own[r] += nb
                    if r == 0:
                        per_key[name] = per_key.get(name, 0) + nb
                        R_ = next(iter(vs))
                        per_view[R_] = per_view.get(R_, 0) + nb
                else:
                    shared[r] += nb
        return own, shared, per_key, per_view

    def _onebk_str(self, detail: bool = False) -> str:
        """the accounting as one log fragment (and ENG_MIX_VIEW_LAZY_MAX_MB's assert)"""
        if not (MIX_ONE_BUCKET and self.bk_view):
            return ""
        try:   # diagnostics only: never let the accounting fail a step
            own, shared, pk, pv = self._onebk_lazy()
        except Exception as e:   # noqa: BLE001
            return f"onebk lazy accounting failed: {type(e).__name__}: {str(e)[:80]}"
        mb = 2.0 ** 20
        out = (f"onebk views {len(self.bk_view)} lazy owned MiB/GPU {'/'.join(f'{x / mb:.1f}' for x in own)} "
               f"shared {max(shared) / mb:.1f}")
        if detail:
            ks = ", ".join(f"{k} {v / mb:.1f}" for k, v in sorted(pk.items(), key=lambda kv: -kv[1])) or "none"
            vs = ", ".join(f"R={R_} {v / mb:.1f}" for R_, v in sorted(pv.items(), key=lambda kv: -kv[1])[:6]) or "none"
            out += f" | GPU 0 by key: {ks} | top views: {vs}"
        if MIX_VIEW_LAZY_MAX_MB and len(self.bk_view) >= 4 and max(own) > MIX_VIEW_LAZY_MAX_MB * mb:
            raise AssertionError(f"ENG_MIX_VIEW_LAZY_MAX_MB {MIX_VIEW_LAZY_MAX_MB}: {out}")
        return out

    @staticmethod
    def _drop_state(s: dict) -> None:
        """ENG_MIX_POOL: clear an evicted bucket state's dicts (two levels: caches[L], ws, ti / tl ...)."""
        for v in list(s.values()):
            if isinstance(v, dict):
                for vv in list(v.values()):
                    if isinstance(vv, dict):
                        vv.clear()
                v.clear()
        s.clear()

    def _mix_cap(self) -> float:
        """Mixed-bucket LRU byte cap per rank: mixed_bucket_bytes, or with ENG_MIX_POOL_GB the pool minus a slack."""
        if self.mix_pool is None:
            return self.cfg.mixed_bucket_bytes
        if MIX_POOL_SPLIT:
            return self.cfg.mixed_bucket_bytes
        return min(self.cfg.mixed_bucket_bytes, min(self.mix_pool_resv["*"]) - MIX_POOL_SLACK_GB * 1e9)

    def _pool_bytes(self, pk: str, r: int) -> tuple:
        """ENG_MIX_POOL: (reserved, allocated) bytes of rank r's mixed-bucket pool pk (its segments)."""
        segs = self.mix_pool[pk][r].snapshot(include_traces=False)   # one pool per GPU: its segments are all on it
        return sum(s_["total_size"] for s_ in segs), sum(s_["allocated_size"] for s_ in segs)

    def _alloc(self, r: int, R: int, moe: str = "raw", lvl: int = 0) -> dict:
        dev, bf, f32, i32, i64 = self.devs[r], torch.bfloat16, torch.float32, torch.int32, torch.int64
        Tall = W * R
        e, r2 = self.mext[r], self.r2
        dv = DECODE_VIEWS and moe == "raw" and lvl == 0 and R >= DECODE_VIEWS_MINR and R != C1_K1L

        def Z(key, *shape, dtype):   # ENG_DECODE_VIEWS: a zeroed view of the shared base; else a fresh torch.zeros
            return self._dv_view(r, key, shape, dtype, R) if dv else torch.zeros(*shape, dtype=dtype, device=dev)

        def E(key, *shape, dtype):   # ... of the shared base (zeroed at creation too: deterministic); else torch.empty
            return self._dv_view(r, key, shape, dtype, R) if dv else torch.empty(*shape, dtype=dtype, device=dev)
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
            if C1_K1L and R == C1_K1L and R % 4:   # FPR_C1_KMAX: the C1 lane bucket's indexer runs 4-row pseudo-requests
                RI = -(-R // 4) * 4                 # (hooks_idx: B = 2 at R 6); its top-k outputs hold all RI internal rows
                s["ti_ri"] = {il: torch.full((RI, 1, 512), -1, dtype=i32, device=dev) for il in INDEX_LAYERS}
                s["tl_ri"] = {il: torch.zeros(RI, dtype=i32, device=dev) for il in INDEX_LAYERS}
                s["ti"] = {il: t[:R] for il, t in s["ti_ri"].items()}   # (FlashMLA / the caches read the R real rows:
                s["tl"] = {il: t[:R] for il, t in s["tl_ri"].items()}   # contiguous prefix views of the same tensors)
            # residual stream / mixes
            for k in ("X", "xm2", "res0", "Xf"):
                s[k] = Z(k, R, HC, D, dtype=bf)
            for k in ("a_pre", "a_post", "f_pre", "f_post", "post0", "lag0"):
                s[k] = Z(k, R, HC, dtype=f32) if k in DV_ZERO else torch.zeros(R, HC, dtype=f32, device=dev)
            for k in ("a_comb", "f_comb"):
                s[k] = Z(k, R, HC, HC, dtype=f32)
            s["comb0"] = torch.eye(HC, device=dev).expand(R, HC, HC).contiguous()
            s["lag0"][:, 0] = 1.0                     # layer 0: hc_collapse with one-hot copy 0 (ds41h.ref hc_collapse None)
            for k in ("x0", "a", "f", "moe_out", "h"):
                s[k] = Z(k, R, D, dtype=bf)
            s["aux"] = Z("aux", R, len(AUX_LAYERS), D, dtype=bf)
            s["X1"] = Z("X1", R, HC, D, dtype=bf)            # Engram: hc_post + injection
            s["zy"] = Z("zy", R, D, dtype=bf)
            s["zpost"] = Z("zpost", R, HC, dtype=f32)
            s["eyec"] = torch.eye(HC, device=dev).expand(R, HC, HC).contiguous()
            s["tok32"] = torch.zeros(R, dtype=i32, device=dev)
            s["part_mhc"] = E("part_mhc", 160 * R * 25, dtype=f32)
            s["empty_i"] = torch.zeros(0, dtype=i32, device=dev)
            # router
            s.update(logits=E("logits", R, NE, dtype=f32), ids=E("ids", R, TOPK, dtype=i32),
                     wts=E("wts", R, TOPK, dtype=f32), route_row=Z("route_row", R, 64, dtype=torch.uint8),
                     ptok_own=Z("ptok_own", R, dtype=i32))
            pad = torch.zeros(1, 64, dtype=torch.uint8, device=dev)
            pad[0, :24] = 0xFF                          # ids -1, weights 0, prescale 0 (= router_top_pack2's rows >= nvalid)
            s["padrow"] = pad
            # AG / RS (peer buffers; flags created and synchronized before use)
            s.update(gf=Z("gf", Tall, D * 2, dtype=torch.uint8), gr=Z("gr", Tall, 64, dtype=torch.uint8),
                     rs=Z("rs", W, R, D, dtype=bf), fl_f=torch.zeros(4, dtype=i32, device=dev),
                     fl_s=torch.zeros(4, dtype=i32, device=dev), ep_f=torch.zeros(1, dtype=i32, device=dev),
                     ep_s=torch.zeros(1, dtype=i32, device=dev))
            if C128_AGF and C1_COMM and moe == "raw" and 33 <= R <= 320:   # ENG_C128_AGF flag / epoch (zeroed)
                s.update(agf_fl=torch.zeros(4, dtype=i32, device=dev), agf_ep=torch.zeros(1, dtype=i32, device=dev))
            if C128_RSL and C1_COMM and moe == "raw" and C128_RSL_MINR <= R <= C128_RSL_MAXR:   # ENG_C128_RSL slots
                s.update(rs_lb128=torch.full((2, W, R, D), -0.0, dtype=bf, device=dev),
                         rs_ltk128=torch.zeros(C128_RSL_GRID, dtype=i32, device=dev))
            if C1_RSL and C1_COMM and C1_RSC and moe == "raw" and R <= C1_FMAXR:   # ENG_C1_RSL MoE slots (sentinel -0.0)
                s.update(rs_lb=torch.full((2, W, R, D), -0.0, dtype=bf, device=dev),
                         rs_ltk=torch.zeros(W * self.cfg.ncta, dtype=i32, device=dev))
            # MoE scratch for 4R rows (sp_layer layout); the moe_raw tables only for decode ('raw') buckets
            s.update(part=E("part", Tall, D, dtype=bf), gu=E("gu", Tall, 2 * 576, dtype=bf),
                     hsh=E("hsh", Tall, 576, dtype=bf), sh=E("sh", Tall, D, dtype=bf))
            if self._has_fork_tables(Tall, moe):   # mixed: the fork's MoE for small steps (route2 MAXT 8,192)
                s.update(self._fork_tables(r, Tall, moe, dv_R=R if dv else None))
            # attention: one workspace (shared by the layers), one cache dict per layer (views of the pools + tables)
            s["ws"] = EA.workspace(R, dev)
            if moe == "raw" and R <= 32:     # decode buckets: the step's row mask for skw8t (ENG_SKW8_*: padding rows skipped)
                s["ws"]["valid"] = s["valid"]
            if EA.C128A and moe == "raw" and EA.C128A_MINR <= R <= EA.C128A_MAXR:   # ENG_C128A (engine/attn.py): decode
                s["ws"]["c128a"] = True
                EA.c128a()                                                           # (built here, outside any capture)
            if C1_TPWO and moe == "raw" and R <= C1_TPWO_MAXR:   # TP-wo peer buffers (flags zeroed; synced before first use)
                s.update(tp_og=torch.zeros(Tall, 8192, dtype=bf, device=dev), tp_vm=torch.zeros(Tall, dtype=torch.uint8, device=dev),
                         tp_z=torch.zeros(Tall, 2048, dtype=bf, device=dev), tp_yp=torch.zeros(Tall, D, dtype=f32, device=dev),
                         tp_rs=torch.zeros(W, R, D, dtype=f32, device=dev), tp_fl1=torch.zeros(4, dtype=i32, device=dev),
                         tp_fl2=torch.zeros(4, dtype=i32, device=dev), tp_ep1=torch.zeros(1, dtype=i32, device=dev),
                         tp_ep2=torch.zeros(1, dtype=i32, device=dev), tp_tk=torch.zeros(4, dtype=i32, device=dev))
                if C1_RSL:   # lamport slots [2 parities][W sources][R][D], sentinel -0.0; per-CTA call counters
                    s.update(tp_lb=torch.full((2, W, R, D), -0.0, dtype=f32, device=dev),
                             tp_ltk=torch.zeros(W * self.cfg.ncta, dtype=i32, device=dev))
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
        if MIX_ONE_BUCKET and moe == "prod":   # the keys _view_state must cover (asserted there)
            s["_alloc_keys"] = frozenset(s)
        return s

    def _has_fork_tables(self, Tall: int, moe: str) -> bool:
        """A bucket of Tall rows (all ranks) carries the fork MoE's tables (p_moe takes the fork path only with them)."""
        return moe == "raw" or Tall <= min(4 * self.cfg.prefill_moe_switch, 8192)

    def _fork_sizes(self, r: int, Tall: int, moe: str) -> tuple:
        """(max_ec, nsr) of the fork MoE's tables for a Tall-row bucket (as _alloc has always sized them)."""
        e, r2 = self.mext[r], self.r2
        max_ec = max(int(e.route_max_ec()), int(r2.route2_max_ec(Tall)))
        nsr = min(Tall, self.cfg.prefill_moe_switch) if (MIX_TBL_CAP and moe != "raw") else Tall   # slot rows
        return max_ec, nsr

    def _fork_tables(self, r: int, Tall: int, moe: str, dv_R: int | None = None) -> dict:
        """The fork MoE's per-bucket tables.
        dv_R (ENG_DECODE_VIEWS, a raw bucket of dv_R rows): img / himg / Y / slotmap are views of the shared base, sized
        from _fork_sizes at the largest decode bucket (their sizes are not linear in R: max_ec terms)."""
        dev, bf, f32, i32 = self.devs[r], torch.bfloat16, torch.float32, torch.int32
        e, r2 = self.mext[r], self.r2
        max_ec, nsr = self._fork_sizes(r, Tall, moe)
        big = None
        if dv_R is not None:
            TM = W * self._dv_rmax
            mxm, nsm = self._fork_sizes(r, TM, moe)
            assert mxm >= max_ec and nsm >= nsr, "ENG_DECODE_VIEWS: fork table sizes not monotone in R"
            big = dict(slotmap=self._dv_view(r, "slotmap", (Tall * TOPK,), i32, dv_R, nmax=TM * TOPK),
                       img=self._dv_view(r, "img", ((nsr * TOPK + 7 * max_ec) * 80 * 128,), torch.uint8, dv_R,
                                         nmax=(nsm * TOPK + 7 * mxm) * 80 * 128),
                       himg=self._dv_view(r, "himg", ((nsr * TOPK + 7 * max_ec) * 9 * 128,), torch.uint8, dv_R,
                                          nmax=(nsm * TOPK + 7 * mxm) * 9 * 128),
                       Y=self._dv_view(r, "Y", (nsr * TOPK + (7 * max_ec if nsr < Tall else 0), D), bf, dv_R,
                                       nmax=(nsm * TOPK + (7 * mxm if nsm < TM else 0)) * D))
        with torch.cuda.device(dev):
            t = dict(max_ec=max_ec, sms=torch.cuda.get_device_properties(dev).multi_processor_count,
                     ecs=torch.zeros(max_ec * (32 + 12 * 64), dtype=torch.uint8, device=dev),
                     t1=torch.zeros(max_ec * 9 * 24, dtype=torch.uint8, device=dev),
                     t2=torch.zeros(max_ec * 40 * 24, dtype=torch.uint8, device=dev),
                     counts=torch.zeros(8, dtype=i32, device=dev),
                     slotmap=big["slotmap"] if big is not None else torch.zeros(Tall * TOPK, dtype=i32, device=dev),
                     rts=torch.zeros(max(int(e.route_scratch_ints()), int(r2.route2_scratch_ints())), dtype=i32, device=dev),
                     img=big["img"] if big is not None else torch.zeros((nsr * TOPK + 7 * max_ec) * 80 * 128, dtype=torch.uint8,
                                                                       device=dev),
                     himg=big["himg"] if big is not None else torch.zeros((nsr * TOPK + 7 * max_ec) * 9 * 128, dtype=torch.uint8,
                                                                         device=dev),
                     Y=big["Y"] if big is not None else torch.zeros(nsr * TOPK + (7 * max_ec if nsr < Tall else 0), D, dtype=bf,
                                                                   device=dev),
                     g5ctr=torch.zeros(2, dtype=i32, device=dev), empty_f=torch.zeros(0, dtype=f32, device=dev),
                     g12sync=torch.zeros(max_ec + 1, dtype=i32, device=dev),
                     g1tick=torch.zeros(max_ec * 9 if C1_G1KS > 1 else 1, dtype=i32, device=dev))
            if ROUTE_STATS:
                t.update(rstat=torch.zeros(64, 2, dtype=torch.int64, device=dev))
        return t

    def _dv_view(self, r: int, key: str, shape: tuple, dtype, R: int, nmax: int | None = None):
        """ENG_DECODE_VIEWS: rank r's view of `key` for a bucket of R rows: base[:numel(shape)].view(shape), zeroed (stream-
        ordered on the current = rank stream). The base is allocated at the first request, for _dv_rmax rows: nmax, or
        numel(shape) / R * _dv_rmax (every shared shape but the fork tables' is linear in R)."""
        n = 1
        for x in shape:
            n *= int(x)
        b = self._dvb[r].get(key)
        if b is None:
            if nmax is None:
                assert n % R == 0, f"ENG_DECODE_VIEWS: {key} {shape} is not linear in R {R}"
                nmax = n // R * self._dv_rmax
            with torch.cuda.device(self.devs[r]):
                b = self._dvb[r][key] = torch.zeros(int(nmax), dtype=dtype, device=self.devs[r])
        assert b.dtype == dtype and n <= b.numel(), f"ENG_DECODE_VIEWS: {key} {shape} {dtype} > base {b.numel()} {b.dtype}"
        v = b[:n].view(*[int(x) for x in shape])
        v.zero_()
        return v

    # ENG_MIX_ONE_BUCKET: how each key of a prod bucket state maps onto a view of R rows (every key _alloc makes for a
    # 'prod' bucket must be listed: _view_state asserts it, so a new buffer added to _alloc cannot be silently shared)
    _V_ROW = ("pos", "tok", "own", "samp", "si", "sl", "valid",          # (without upload_v2: separate tensors)
              "X", "xm2", "res0", "Xf", "a_pre", "a_post", "f_pre", "f_post", "post0", "lag0", "a_comb", "f_comb", "comb0",
              "x0", "a", "f", "moe_out", "h", "aux", "X1", "zy", "zpost", "eyec", "tok32",
              "logits", "ids", "wts", "route_row", "ptok_own")          # [R, ...]: prefix rows (alloc-time rows identical)
    _V_TALL = ("gf", "gr", "part", "gu", "hsh", "sh")                     # [W * R, ...]: prefix rows (rank stride R)
    _V_SHARED = ("fl_f", "fl_s", "ep_f", "ep_s", "empty_i", "padrow")    # flags / device epochs / constants: one sequence
    _V_TABS = ("max_ec", "sms", "ecs", "t1", "t2", "counts", "slotmap", "rts", "img", "himg", "Y", "g5ctr", "empty_f",
               "g12sync", "g1tick", "rstat")                             # the fork MoE's tables (shared set, prefixes)

    def _view_state(self, r: int, b: dict, R: int) -> dict:
        """ENG_MIX_ONE_BUCKET: the state _alloc(r, R, 'prod') would return, as views of the max A bucket b (same keys,
        shapes, dtypes; row-led tensors keep their alloc-time rows; scratch with R in a later dim ([W, R, D] rs,
        [8, R, 1024] zb) is the base's first elements re-viewed contiguous: those are fully rewritten before every read;
        flags / device epochs are shared: one continuous use sequence, as one bucket reused across steps)."""
        Rb, Tall, dev = b["R"], W * R, b["dev"]
        assert b["moe"] == "prod" and R <= Rb, (b["moe"], R, Rb)
        s = dict(R=R, dev=dev, moe=b["moe"], lvl=b["lvl"])
        done = {"R", "dev", "moe", "lvl", "bkey", "pinned", "caches", "ws", "ti", "tl", "in_d", "part_mhc", "rs"}

        def flat(t, n, shape):
            assert t.is_contiguous() and t.numel() >= n, (tuple(t.shape), n)
            return t.view(-1)[:n].view(shape)
        if "in_d" in b:   # upload_v2: the R layout's sections inside the base blob (each step's H2D rewrites them all)
            s["in_d"] = b["in_d"][:_in_layout(R)[1]]
            s.update(_in_views(s["in_d"], R))
        for k in self._V_ROW:
            if k in b and k not in s:
                assert b[k].shape[0] == Rb, (k, tuple(b[k].shape))
                s[k] = b[k][:R]
        for k in self._V_TALL:
            assert b[k].shape[0] == W * Rb, (k, tuple(b[k].shape))
            s[k] = b[k][:Tall]
        for k in self._V_SHARED:
            s[k] = b[k]
        s["rs"] = flat(b["rs"], W * R * D, (W, R, D))
        s["part_mhc"] = b["part_mhc"][:160 * R * 25]
        s["ti"] = {il: t[:R] for il, t in b["ti"].items()}
        s["tl"] = {il: t[:R] for il, t in b["tl"].items()}
        # the attention workspace: exactly attn_half._workspace's alloc-time keys (keys a step adds lazily, e.g. pf_ab, are
        # recreated on the view by the same code paths, as on a fresh bucket)
        ws = dict(R=R)
        for k in ("qakv", "qr", "kv", "q", "o", "z", "y"):
            v = b["ws"][k]
            assert v.shape[0] == Rb, (k, tuple(v.shape))
            ws[k] = v[:R]
        ws["zb"] = flat(b["ws"]["zb"], 8 * R * 1024, (8, R, 1024))
        s["ws"] = ws
        if self._has_fork_tables(Tall, "prod"):   # as a fresh R bucket: the fork MoE (p_moe tests "ecs" in s)
            tb = self.bk_tabs[r]
            max_ec, nsr = self._fork_sizes(r, Tall, "prod")
            assert max_ec <= tb["max_ec"], (max_ec, tb["max_ec"])
            ns = nsr * TOPK + 7 * max_ec
            s.update(max_ec=max_ec, sms=tb["sms"], counts=tb["counts"], rts=tb["rts"], g5ctr=tb["g5ctr"],
                     empty_f=tb["empty_f"], ecs=tb["ecs"][:max_ec * (32 + 12 * 64)], t1=tb["t1"][:max_ec * 9 * 24],
                     t2=tb["t2"][:max_ec * 40 * 24], slotmap=tb["slotmap"][:Tall * TOPK], img=tb["img"][:ns * 80 * 128],
                     himg=tb["himg"][:ns * 9 * 128], Y=tb["Y"][:nsr * TOPK + (7 * max_ec if nsr < Tall else 0)],
                     g12sync=tb["g12sync"][:max_ec + 1], g1tick=tb["g1tick"][:max_ec * 9 if C1_G1KS > 1 else 1])
            if "rstat" in tb:
                s["rstat"] = tb["rstat"]
        done |= set(self._V_ROW) | set(self._V_TALL) | set(self._V_SHARED) | set(self._V_TABS)
        missing = set(b.get("_alloc_keys", ())) - done
        assert not missing, f"ENG_MIX_ONE_BUCKET: bucket keys without a view rule: {sorted(missing)}"
        pools = self.pools[r]
        s["caches"] = {}
        for L in self.layers:   # as _alloc, on the view's tensors
            src, isrc = kv_source_of(L), index_source_of(L)
            s["caches"][L] = dict(R=R, device=dev, stride=self.WK.WINDOW_PAGE_BYTES, sbuf=pools.window_base(L),
                                  k_swa=pools.window_view(L), si=s["si"], sl=s["sl"], own_slot=s["own"],
                                  scale=self.lw[r][L]["AW"]["scale"], ratio=RATIO[L],
                                  k_comp=pools.comp_view(src) if src is not None else None,
                                  ti=s["ti"][isrc] if isrc is not None else None, tl=s["tl"][isrc] if isrc is not None else None,
                                  sched=None, planned=True, src=src, isrc=isrc)
        s["pinned"] = None
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

    # ------------------------------------------------------------------------------------------------ ENG_MIX_PERSIST / ENG_TAIL_GATHER
    def _persist_init(self) -> None:
        """Per-rank persistent buffers: tail_ix [max_tail_rows] i64 + two pinned sets each, and HSD's
        device-token staging (_dt / _dt_d) at its fixed size now instead of at first use (never regrown: <= 4,096 rows)."""
        n = int(self.max_tail_rows)
        self._tix = []
        for ri, d in enumerate(self.devs):
            with torch.cuda.device(d):
                self._tix.append(dict(d=torch.zeros(n, dtype=torch.int64, device=d), i=0, gen=1,
                                      h=[[torch.zeros(n, dtype=torch.int64, pin_memory=True), self._set_event(ri)]
                                         for _ in range(2)]))
        hk = self.hooks
        if getattr(hk, "hsd", None) is not None and not hasattr(hk, "_dt"):
            nmax = 4096
            hk._dt = [[(torch.zeros(2, nmax, dtype=torch.int64, pin_memory=True), None) for _ in range(2)] for _ in self.devs]
            hk._dt_d = [torch.zeros(2, nmax, dtype=torch.int64, device=d) for d in self.devs]
            hk._dt_i = [0] * len(self.devs)
        if TAIL_GATHER_KERNEL:   # the extension now, not at the first T stage: the exporter's
            self._tg()           # libs["tail_gather"] exists from init, so native T_STAGE runs from the first mixed step
        self.log(f"engine: ENG_MIX_PERSIST / ENG_TAIL_GATHER: tail_ix [{n}] i64 x 2 pinned sets per rank"
                 f"{'; HSD device-token staging at init (4,096 rows)' if hasattr(hk, '_dt') else ''}")

    def _tail_ix(self, r: int, rows) -> torch.Tensor:
        """Rank r's tail row indices (A rows feeding T, in order) in its persistent tail_ix: one pinned set (refilled
        only after its copy of two steps ago completed) -> one H2D on the current (= rank) stream. Returns tail_ix[:n]."""
        tx = self._tix[r]
        n = len(rows)
        assert n <= tx["d"].numel(), f"ENG_TAIL_GATHER: {n} tail rows > tail_ix {tx['d'].numel()}"
        i = tx["i"]
        hb, ev = tx["h"][i]
        ev.synchronize()                   # (the set's previous H2D; persistent event: FprEhSets rule)
        hb.numpy()[:n] = rows
        d = tx["d"]
        d[:n].copy_(hb[:n], non_blocking=True)
        ev.record(torch.cuda.current_stream(d.device))
        tx["i"] = i ^ 1
        return d[:n]

    def _tail_gather(self, sA: dict, sT: dict, ix: torch.Tensor, n: int, l20: bool) -> None:
        """ENG_TAIL_GATHER (#7, FPR_OP_TAIL_GATHER's Python reference): bucket A's rows ix into bucket T's first n rows,
        index_select written in place (out=), the same keys in the same order as the default path."""
        sel = lambda src, dst: torch.index_select(src, 0, ix, out=dst[:n])  # noqa: E731
        if l20:
            for k in ("X", "a_pre", "a_post", "a_comb"):
                sel(sA[k], sT[k])
            sel(sA["ws"]["q"], sT["ws"]["q"])
            sel(sA["si"], sT["si20"])
            sel(sA["sl"], sT["sl20"])
        else:
            for k in ("moe_out", "xm2", "f_post", "f_comb", "f_pre"):
                sel(sA[k], sT[k])
        for il in INDEX_LAYERS:
            if il <= 20:
                sel(sA["ti"][il], sT["ti"][il])
                sel(sA["tl"][il], sT["tl"][il])

    def _tail_pairs(self, sA: dict, sT: dict, l20: bool) -> list:
        """(src, dst) of every tail key in _tail_gather's order: FPR_OP_TAIL_GATHER's key list."""
        if l20:
            pr = [(sA[k], sT[k]) for k in ("X", "a_pre", "a_post", "a_comb")]
            pr += [(sA["ws"]["q"], sT["ws"]["q"]), (sA["si"], sT["si20"]), (sA["sl"], sT["sl20"])]
        else:
            pr = [(sA[k], sT[k]) for k in ("moe_out", "xm2", "f_post", "f_comb", "f_pre")]
        for il in INDEX_LAYERS:
            if il <= 20:
                pr += [(sA["ti"][il], sT["ti"][il]), (sA["tl"][il], sT["tl"][il])]
        return pr

    def _tg(self):
        """kernels/tail_gather (FPR_OP_TAIL_GATHER), loaded once."""
        m = self.__dict__.get("_tg_mod")
        if m is None:
            d = os.path.join(PHYS, "kernels", "tail_gather")
            if d not in sys.path:
                sys.path.insert(0, d)
            import tail_gather as m
            m.ext()
            self._tg_mod = m
        return m

    def _set_event(self, r: int):
        """A staging set's persistent event (FprEhSets rule), recorded once on rank r's stream now so its cudaEvent_t
        exists (torch creates it at the first record) and a first wait returns at once."""
        ev = torch.cuda.Event()
        with torch.cuda.device(self.devs[r]):
            ev.record(self.streams[r])
        return ev

    def _ensure_new(self, st: list, role: str) -> None:
        """A mixed bucket / onebk view was just created (its 4 states complete, bkey set): ENG_MIX_PERSIST's ensure-time
        buffers (vidx_d + sets; T: si20 / sl20 under ENG_L20_TAIL; the FprTgKeys table), then ENG_MIX_ENSURE's hooks."""
        if MIX_PERSIST:
            for r_, s_ in enumerate(st):
                self._vidx_alloc(r_, s_)
                if role == "T" and L20_TAIL:
                    s_.setdefault("si20", torch.empty_like(s_["si"]))
                    s_.setdefault("sl20", torch.empty_like(s_["sl"]))
                if s_.get("moe") == "prod":
                    self._tg_keys(s_)
        if MIX_ENSURE:
            if not self.__dict__.get("_ensure_inited"):
                self._init_ensure_hooks()
            for h in self._ensure_hooks:
                h(st)
        elif CXX_PIECES and os.environ.get("ENG_MIXED_CXX_PLAN_SLOTS", "1") == "1":
            # the view's FlashMLA plan slot NUMBERS only
            # (s["_cxx_attn_slots"] for nd's handover; no reset, no handle build, no GPU work), so native mixed steps are
            # not refused for missing plan slots while ensure()'s handle pre-build stays off
            import mixed_cxx_attn as MXA
            MXA.plan_slots(self._cxx_drv(), st)

    def _init_ensure_hooks(self) -> None:
        """ENG_MIX_ENSURE, at the first mixed creation: the C++ driver's handle pre-build (mixed_cxx_ensure.ensure_view:
        FPR's pieces + the owners' ensure fns) when ENG_MIXED_CXX_PIECES is on. Other hooks append to _ensure_hooks."""
        self._ensure_inited = True
        if CXX_PIECES:
            drv = self._cxx_drv()
            import mixed_cxx_ensure as MXE
            self._ensure_hooks.insert(0, lambda st: MXE.ensure_view(drv, st))

    def _tg_py(self):
        """kernels/tail_gather's Python module (FprTgKeys / keys_bytes; no extension build)."""
        m = self.__dict__.get("_tg_py_mod")
        if m is None:
            d = os.path.join(PHYS, "kernels", "tail_gather")
            if d not in sys.path:
                sys.path.insert(0, d)
            import tail_gather as m
            self._tg_py_mod = m
        return m

    def _tg_keys(self, s: dict) -> bytes:
        """The view's FprTgKeys: its per-row tail keys in FPR_TG_* order, packed.
        Cached in s["_tg_kb"] with the si20 pointer it saw (si20 / sl20 may appear later: setdefault in an l20 step)."""
        si20 = s.get("si20")
        p20 = si20.data_ptr() if si20 is not None else 0
        c = s.get("_tg_kb")
        if c is not None and c[0] == p20:
            return c[1]
        TG = self._tg_py()
        sl = {TG.X: s["X"], TG.A_PRE: s["a_pre"], TG.A_POST: s["a_post"], TG.A_COMB: s["a_comb"], TG.WS_Q: s["ws"]["q"],
              TG.SI: s["si"], TG.SL: s["sl"], TG.SI20: si20, TG.SL20: s.get("sl20"), TG.MOE_OUT: s["moe_out"],
              TG.XM2: s["xm2"], TG.F_POST: s["f_post"], TG.F_COMB: s["f_comb"], TG.F_PRE: s["f_pre"]}
        idx = [il for il in INDEX_LAYERS if il <= 20]
        for j, il in enumerate(idx):
            sl[TG.TI0 + j], sl[TG.TL0 + j] = s["ti"][il], s["tl"][il]
        b = TG.keys_bytes(s["R"], len(idx), sl)
        s["_tg_kb"] = (p20, b)
        return b

    def _t_stage(self, sA: dict, sT: dict, ix, n: int, l20: bool) -> None:
        """ENG_TAIL_GATHER_KERNEL: bucket T's stage in one launch on the current (= rank) stream: fpr_t_stage_run =
        _tail_gather's keys at ix + the padding rows step_mixed writes (l20: si20 / sl20 from T's si / sl; clear_tail_padding:
        ti -1 / tl 0, rows [n, R)). nd's T_STAGE calls the same C function on the same key tables."""
        self._tg().t_stage(ix, n, self._tg_keys(sA), self._tg_keys(sT), l20, BISECT["clear_tail_padding"],
                           torch.device(sT["dev"]).index)

    def _vidx_alloc(self, r: int, s: dict) -> None:
        """Called at mixed-bucket / view creation (ensure time) and by
        _set_vidx_persist. vidx_gen counts (re)allocations (the exporter's generation)."""
        R = s["R"]
        o_nv = 8 * W * R
        o_nd = o_nv + -(-4 * W // 8) * 8
        nbytes = o_nd + 4 * W
        if s.get("vidx_d") is not None and s["vidx_d"].numel() == nbytes:
            return
        dev = self.devs[r]
        with torch.cuda.device(dev):
            d = torch.zeros(nbytes, dtype=torch.uint8, device=dev)
        s["vidx_d"] = d
        s["vidx_h"] = [[torch.zeros(nbytes, dtype=torch.uint8, pin_memory=True), self._set_event(r)] for _ in range(2)]
        s["vidx_i"] = 0
        s["vidx_gen"] = s.get("vidx_gen", 0) + 1
        s["_vidx_v"] = (d[:o_nv].view(torch.int64), d[o_nv:o_nv + 4 * W].view(torch.int32),
                        d[o_nd:o_nd + 4 * W].view(torch.int32))

    def _set_vidx_persist(self, st: list, ns: list, nds: list | None = None) -> None:
        """ENG_MIX_PERSIST: _set_vidx's values (same act8 rule and fallback count) in each rank's persistent vidx_d blob."""
        import numpy as np
        R = st[0]["R"]
        WR = W * R
        o_nv = 8 * WR
        o_nd = o_nv + -(-4 * W // 8) * 8
        nbytes = o_nd + 4 * W
        v = np.concatenate([np.arange(n, dtype=np.int64) + r * R for r, n in enumerate(ns)])
        vc = int(v.size)
        act8 = PF_ACT8 and nds is not None and min(nds) >= 0
        if PF_ACT8 and nds is not None and not act8:
            if not self.stats.get("act8_fallback_steps"):
                print(f"engine: ENG_PF_ACT8: decode rows not a prefix of a rank's rows ({nds}): this step's MoE stays W4A16 "
                      f"(counted in stats act8_fallback_steps)", flush=True)
            self.stats["act8_fallback_steps"] = self.stats.get("act8_fallback_steps", 0) + 1
        for r in range(W):
            s = st[r]
            self._vidx_alloc(r, s)
            i = s["vidx_i"]
            hb, ev = s["vidx_h"][i]
            ev.synchronize()
            h = hb.numpy()
            h[:8 * vc].view(np.int64)[:] = v
            h[o_nv:o_nv + 4 * W].view(np.int32)[:] = ns
            if act8:
                h[o_nd:o_nd + 4 * W].view(np.int32)[:] = nds
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                s["vidx_d"].copy_(hb, non_blocking=True)
                ev.record(self.streams[r])
            s["vidx_i"] = i ^ 1
            vv, nvv, ndv = s["_vidx_v"]
            s["vidx"] = vv[:vc]
            s["nv"] = nvv
            s["nd"] = ndv if act8 else None
            s["ndec"] = int(sum(nds)) if act8 else -1
            s["vcount"] = vc

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
    def _cxx_drv(self):
        """ENG_MIXED_CXX_PIECES: the C++ mixed-step driver (fpr/engine/mixed_cxx), created on first use."""
        d = getattr(self, "_cxx", None)
        if d is None:
            p_ = os.path.join(PHYS, "engine/mixed_cxx")
            if p_ not in sys.path:
                sys.path.insert(0, p_)
            import mixed_cxx
            d = self._cxx = mixed_cxx.Driver(self, sys.modules[__name__])
            self._cxx_rank = {torch.device(dv): r_ for r_, dv in enumerate(self.devs)}
        return d

    def _join_mix(self, s: dict, pf: bool = True) -> None:
        if CXX_PIECES and getattr(self, "_cxx", None) is not None:   # a C++ boundary's pending side-stream mixes
            self._cxx.join(self._cxx_rank[torch.device(s["dev"])], torch.cuda.current_stream())
        ev = s.pop("mix_ev", None)
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        if pf and C1_L2PF and "pf_evs" in s:   # ENG_C1_L2PF: graph ends / cuts join the prefetch branches (not the
            self._pf_join(s)                    # boundaries' own joins: the main stream never waits for a prefetch)

    # -- ENG_C1_L2PF: weight prefetch into L2 on a side stream (see the flag)
    def l2pfx(self):
        x = getattr(self, "_l2pfx", None)
        if x is None:
            from torch.utils.cpp_extension import load
            bd = os.path.join(PHYS, "engine/build/eng_l2pf")
            os.makedirs(bd, exist_ok=True)
            x = self._l2pfx = load(name="eng_l2pf", sources=[os.path.join(PHYS, "kernels/l2pf/l2pf.cu")], build_directory=bd,
                                   verbose=False, extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])
        return x

    @staticmethod
    def _pf_range(t):
        """(ptr, row_bytes, rows, stride) of a contiguous tensor or a row-strided 2D slice (unit inner stride)."""
        es = t.element_size()
        if t.is_contiguous():
            return (t.data_ptr(), t.numel() * es, 1, 0)
        assert t.dim() == 2 and t.stride(1) == 1, "l2pf: contiguous tensors or row-strided 2D slices only"
        return (t.data_ptr(), t.shape[1] * es, t.shape[0], t.stride(0) * es)

    def _pf_ranges(self, r: int, L: int, item: str) -> list:
        key = (r, L, item)
        c = self.__dict__.setdefault("_pf_cache", {})
        if key not in c:
            lw = self.lw[r][L]
            AW = lw["AW"]
            ts = []
            if item == "qkv":
                ts = [AW.get("wqkv_a")]
            elif item == "wo":
                if "pk8tp" in AW:   # (ENG_WO_ONECOPY: the wo_a view [32][8][CHUNK] as a row-strided [32][8 * CHUNK])
                    ts = [t if t.is_contiguous() else t.as_strided((t.shape[0], t.shape[1] * t.shape[2]), (t.stride(0), 1))
                          for t in (AW["pk8tp"]["wo_a"], AW["pk8tp"]["wo_b"])]
                elif "wo_aT_c1" in AW:   # ENG_C128_DENSE8: this rank's quarter (contiguous)
                    ts = [AW["wo_aT_c1"], AW["wo_b_c1"]]
                elif "wo_aT" in AW and "wo_b" in AW:
                    ts = [AW["wo_aT"][2 * r:2 * r + 2], AW["wo_b"][:, 2048 * r:2048 * (r + 1)]]
            elif item == "gate":
                ts = [lw.get("gate_w"), lw.get("sw13"), lw.get("sw2")]
            c[key] = [self._pf_range(t) for t in ts if isinstance(t, torch.Tensor) and t.is_cuda and t.numel()]
        return c[key]

    def _pf_issue(self, r: int, s: dict, L: int, item: str) -> None:
        if (item not in C1_L2PF or s.get("moe") != "raw" or s["R"] > C1_L2PF_MAXR or L not in self.lw[r]
                or s.get("pf") is not None or s.get("cp_on")):
            return
        rg = self._pf_ranges(r, L, item)
        if not rg:
            return
        pfs = self.__dict__.setdefault("_pf_streams", {})
        side = pfs.get(r)
        if side is None:
            side = pfs[r] = torch.cuda.Stream(device=self.devs[r])
        cur = torch.cuda.current_stream()
        ev = torch.cuda.Event()
        ev.record(cur)
        side.wait_event(ev)
        with torch.cuda.stream(side):
            self.l2pfx().pf([x[0] for x in rg], [x[1] for x in rg], [x[2] for x in rg], [x[3] for x in rg], C1_L2PF_GRID,
                            C1_L2PF_CHUNK)
            ev2 = torch.cuda.Event()
            ev2.record(side)
        s.setdefault("pf_evs", []).append(ev2)

    def _pf_join(self, s: dict) -> None:
        for ev in s.pop("pf_evs", []):
            torch.cuda.current_stream().wait_event(ev)

    def _split_bd(self, r, s, k1a_rows, xn, lag, sc, bs, nw, out, ptok, pre, post, comb, fused=None, fold=None):
        """sp_layer split boundary (--mhc-split 1): K1a, K1b-out, K1b-mix (side), in 128-row chunks.
        fused = (x, res, post_in, comb_in, fn_f) with ENG_MHCF: the same boundary as kernels/mhc_fuse fa + fb_out + fb_mix
        (bit-identical; fa computes the collapse itself, so it takes lag too)."""
        cfg = self.mcfg
        self._join_mix(s, pf=False)
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
            if fold is not None:       # ENG_C1_FOLD: x = the 4-slot RS sum, summed inside the boundary kernel
                F.frow_rs(fold[0], fold[1], fold[2], fold[3], ri[:R], pi[:R], ci[:R], lag[:R], nw, cfg.norm_eps, xn[:R], out[:R],
                          ptok[:R] if ptok.numel() else ptok)
            else:
                F.frow(xi[:R], ri[:R], pi[:R], ci[:R], lag[:R], nw, cfg.norm_eps, xn[:R], out[:R],
                       ptok[:R] if ptok.numel() else ptok)
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
        if "embed" in CXX_PIECES and s.get("moe") == "prod" and not torch.cuda.is_current_stream_capturing() and \
                self._cxx_drv().piece("embed", r, s, None):   # ENG_MIXED_CXX_PIECES: ids32 + Engram prepare + gather in C++
            return
        if self.hooks.engram is not None:      # token-only Engram work (hash, host-table gathers, wkv) on its own stream
            self.hooks.ids32[r][: s["R"]].copy_(s["tok"])
            e0 = self._c1m_e0(s)
            if e0 is not None:   # ENG_C1_MEGA + C1M_IX + C1IDX_E0: c1mega's layer 0 computes the wkv GEMVs: hash + gathers only
                e0.e0_prepare(r, s["R"], stream=self.estream[r])
            else:
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
        if C1_L2PF and L == self.layers[0]:
            self._pf_issue(r, s, L, "qkv")
        if "engram" in CXX_PIECES and L in ENGRAM_LAYERS and self.hooks.engram is not None and s.get("moe") == "prod" \
                and not torch.cuda.is_current_stream_capturing() and self._cxx_drv().piece("engram_bd", r, s, L):
            return                                    # ENG_MIXED_CXX_PIECES: Engram inject + identity boundary in C++
        if "bd" in CXX_PIECES and s.get("moe") == "prod" and not torch.cuda.is_current_stream_capturing() and \
                self._cxx_drv().bd(r, s, L, 0):       # ENG_MIXED_CXX_PIECES: the boundary (+ aux) in C++
            return
        lw = self.lw[r][L]
        x, res, post, comb, lag = self._inputs(s, L)
        if L in ENGRAM_LAYERS and self.hooks.engram is not None:
            # Engram: X1 = hc_post(prev) + injection in one pass, then the boundary in identity mode on X1
            # (post 0, comb I: K1a's X == X1 bit for bit; mixes / collapse / RMSNorm of the injected stream, as production)
            if BISECT["inject_join"]:
                self._join_mix(s, pf=False)   # inject reads f_post / f_comb of the side stream's k1b_mix (without this join the
                                # eager production-MoE path had no happens-before edge there)
            self.hooks.engram[r].inject(ENGRAM_LAYERS.index(L), s["R"], y=x, rs=res, post=post, comb=comb, out=s["X1"])
            x, res, post, comb = s["zy"], s["X1"], s["zpost"], s["eyec"]

        def k1(c0, c1, part):
            self.hext.k1a(x[c0:c1], res[c0:c1], post[c0:c1], comb[c0:c1], lw["fa"], s["X"][c0:c1], part)
        fold = s.pop("fold_x", None) if x is s.get("moe_out") else None
        self._split_bd(r, s, k1, s["X"], lag, lw["sa"], lw["ba"], lw["na"], s["a"], s["empty_i"], s["a_pre"], s["a_post"],
                       s["a_comb"], fused=(x, res, post, comb, lw.get("fa_f"), lw["fa"]), fold=fold)
        if L in AUX_LAYERS:  # DSpark aux = stock previous_post.mean(dim=1) on the bf16 block input (hsd.aux_of: fp32 sum,
            s["aux"][:, AUX_LAYERS.index(L)].copy_(s["X"].mean(1))   # one bf16 rounding)

    def p_attn_a(self, r: int, s: dict, L: int) -> None:
        EA.attn_a(self.lw[r][L]["AW"], s["ws"], s["a"], s["pos"])

    def p_attn_b(self, r: int, s: dict, L: int) -> None:
        EA.attn_b(self.lw[r][L]["AW"], s["caches"][L], s["ws"], s["pos"], insert=self.cfg.insert, pf=s.get("pf"))

    def _tp_on(self, s: dict) -> bool:
        return C1_TPWO and "tp_og" in s and s.get("pf") is None and not s.get("cp_on")

    def _tp_wo1(self, r: int, s: dict, L: int, st: list) -> None:
        """ENG_C1_TPWO piece 1 (end of p_attn): ws["o"] (FlashMLA + inverse RoPE done) -> every rank's head-group slice;
        the kernel's last CTA waits for this rank's slices. Pieces 2 / 3 (p_tpwo2 / p_tpwo3) are issued for every rank
        after this one, so a host sync inside a piece (cuBLAS workspace allocation in an eager step) never waits on a
        GPU that waits on a push not issued yet."""
        R, x, nc = s["R"], self.c1x(), self.cfg.ncta
        cs = self.lw[r][L]["AW"]["cos_sin"]
        pos = s["pos"] if C1_TPWO_ROPE else s["empty_i"]
        x.tp_scatter(s["ws"]["o"].view(R, 32768), s["valid"].view(torch.uint8).view(R), r, [y["tp_og"] for y in st],
                     [y["tp_fl1"] for y in st], [y["tp_vm"] for y in st], s["tp_ep1"], nc, s["tp_fl1"], W * nc, s["tp_tk"],
                     pos, cs)

    def p_tpwo2(self, r: int, s: dict, L: int, st: list) -> None:
        """TP-wo piece 2: z = wo_a (groups 2r, 2r+1) and the fp32 wo_b partial for every rank's rows; push the owners'."""
        if not self._tp_on(s):
            return
        AW, R, x, nc = self.lw[r][L]["AW"], s["R"], self.c1x(), self.cfg.ncta
        T = W * R
        if WO_ONECOPY and "pk8tp" in AW:   # ENG_WO_ONECOPY: w8a16tp on this rank's views of R2's packed copy (all T rows, as
            xt = EA.w8tp()                 # cuBLAS; wo_a -> bf16 z, wo_b -> the fp32 partial)
            ws_, ca_, cb_ = EA._w8tp_ws(s["dev"])
            xt.gemm(s["tp_og"], AW["pk8tp"]["wo_a"], s["tp_z"], 1024, 4096, WO_ONECOPY_S[0], None, None, None, None, None, 0, 0,
                    None, ws_, ca_, None)
            xt.gemm(s["tp_z"], AW["pk8tp"]["wo_b"], s["tp_yp"], 5120, 0, WO_ONECOPY_S[1], None, None, None, None, None, 0, 0,
                    None, ws_, cb_, None)
        elif "pk8tp" in AW:     # ENG_C1_TPWO_F8: the same two GEMMs from fp8 slices (skw8t, rows of tp_vm only)
            ws8, cnt8 = EA._skw8_ws(s["dev"])
            EA.skw8_vw()(s["tp_og"], AW["pk8tp"]["wo_a"], s["tp_z"], 2048, 4096, 1024, 4096, C1_TPWO_F8_S[0], ws8,
                         cnt8["tp_wo_a"], s["tp_vm"])
            EA.skw8_vw()(s["tp_z"], AW["pk8tp"]["wo_b"], s["tp_yp"], 5120, 2048, 0, 0, C1_TPWO_F8_S[1], ws8,
                         cnt8["tp_wo_b"], s["tp_vm"])
        else:   # (ENG_C128_DENSE8: the same bytes as this rank's quarter tensors, contiguous)
            woa_q = AW["wo_aT_c1"] if "wo_aT_c1" in AW else AW["wo_aT"][2 * r:2 * r + 2]
            wob_q = AW["wo_b_c1"] if "wo_b_c1" in AW else AW["wo_b"][:, 2048 * r:2048 * (r + 1)]
            torch.bmm(s["tp_og"].view(T, 2, 4096).transpose(0, 1), woa_q, out=s["tp_z"].view(T, 2, 1024).transpose(0, 1))
            torch.mm(s["tp_z"], wob_q.t(), out_dtype=torch.float32, out=s["tp_yp"])
        if C1_RSL and "tp_lb" in s:   # push + poll + rank-order sum in one kernel (p_tpwo3 then has nothing to do)
            x.tp_rsl(s["tp_yp"], s["tp_vm"], r, [y["tp_lb"] for y in st], s["tp_ltk"], s["ws"]["y"],
                     s["valid"].view(torch.uint8).view(R), W * nc)
            return
        x.tp_rs(s["tp_yp"], s["tp_vm"], r, [y["tp_rs"].view(torch.uint8) for y in st], [y["tp_fl2"] for y in st], s["tp_ep2"], nc)

    def p_tpwo3(self, r: int, s: dict, L: int) -> None:
        """TP-wo piece 3: y (bf16) = the 4 partials of this rank's real rows, summed in rank order."""
        if not self._tp_on(s):
            return
        if C1_L2PF:   # router + shared expert weights during the reduce's wait (the FFN boundary reads them next)
            self._pf_issue(r, s, L, "gate")
        R = s["R"]
        if C1_RSL and "tp_lb" in s:   # ENG_C1_RSL: summed in p_tpwo2's kernel
            return
        if C1_FOLD and MHCF and self.mhcf is not None and self.lw[r][L].get("ff_f") is not None:   # summed in p_ffn_bd's boundary
            s["fold_y"] = (s["tp_rs"], s["tp_fl2"], s["tp_ep2"], W * self.cfg.ncta)
            return
        self.c1x().tp_rs_sum(s["tp_rs"], s["tp_fl2"], s["tp_ep2"], W * self.cfg.ncta, s["ws"]["y"], 132,
                             s["valid"].view(torch.uint8).view(R))

    def p_attn(self, r: int, s: dict, L: int) -> None:
        AW, ws, c = self.lw[r][L]["AW"], s["ws"], s["caches"][L]
        if self._tp_on(s):
            st = self._tp_st
            EA.attn_a(AW, ws, s["a"], s["pos"])
            if C1_QB_OVERLAP and L in INDEX_LAYERS and self.hooks.idx is not None and not DBG_SYNC:
                qs = self.__dict__.setdefault("_qb_side", {})
                side = qs.get(r)
                if side is None:
                    side = qs[r] = torch.cuda.Stream(device=self.devs[r])
                cur = torch.cuda.current_stream()
                ev = torch.cuda.Event()
                ev.record(cur)
                side.wait_event(ev)
                if C1_QB_ORDER:   # ENG_C1_QB_ORDER: the indexer chain issued first (its kernels created before q_b's)
                    self.hooks.indexer_run(r, s, L)
                with torch.cuda.stream(side):
                    EA.attn_b_insert(AW, c, ws, s["pos"], insert=self.cfg.insert)
                    ev2 = torch.cuda.Event()
                    ev2.record(side)
                if not C1_QB_ORDER:
                    self.hooks.indexer_run(r, s, L)
                cur.wait_event(ev2)
            else:
                self.hooks.indexer_run(r, s, L)
                EA.attn_b_insert(AW, c, ws, s["pos"], insert=self.cfg.insert)
            if C1_L2PF:
                self._pf_issue(r, s, L, "wo")
            EA.attn_b_attend_o(AW, c, ws, s["pos"], inv=not C1_TPWO_ROPE)
            self._tp_wo1(r, s, L, st)
            return
        EA.attn_a(AW, ws, s["a"], s["pos"])
        if (C1_QB_OVERLAP and L in INDEX_LAYERS and s["moe"] == "raw" and s.get("pf") is None and not DBG_SYNC
                and self.hooks.idx is not None):
            qs = self.__dict__.setdefault("_qb_side", {})
            side = qs.get(r)
            if side is None:
                side = qs[r] = torch.cuda.Stream(device=self.devs[r])
            cur = torch.cuda.current_stream()
            ev = torch.cuda.Event()
            ev.record(cur)
            side.wait_event(ev)
            with torch.cuda.stream(side):
                EA.attn_b_insert(AW, c, ws, s["pos"], insert=self.cfg.insert)
                ev2 = torch.cuda.Event()
                ev2.record(side)
            self.hooks.indexer_run(r, s, L)
            cur.wait_event(ev2)
            EA.attn_b_attend(AW, c, ws, s["pos"], pf=s.get("pf"))
            return
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
        if C1_L2PF and not self._tp_on(s):
            self._pf_issue(r, s, L, "gate")
        if "bd" in CXX_PIECES and s.get("moe") == "prod" and not torch.cuda.is_current_stream_capturing() and \
                self._cxx_drv().bd(r, s, L, 1):       # ENG_MIXED_CXX_PIECES: the boundary in C++
            return
        lw = self.lw[r][L]
        y = s["ws"]["y"]

        def k1(c0, c1, part):
            self.hext.k1a(y[c0:c1], s["X"][c0:c1], s["a_post"][c0:c1], s["a_comb"][c0:c1], lw["ff"], s["xm2"][c0:c1], part)
        self._split_bd(r, s, k1, s["xm2"], s["a_pre"], lw["sf"], lw["bf"], lw["nf"], s["f"], s["ptok_own"], s["f_pre"],
                       s["f_post"], s["f_comb"], fused=(y, s["X"], s["a_post"], s["a_comb"], lw.get("ff_f"), lw["ff"]),
                       fold=s.pop("fold_y", None))

    def c1m(self):
        """ENG_C1_MEGA: the c1mega adapter (fpr/<C1M_PKG, default kernels/c1mega>/c1mega_eng.py), imported and built on first use.
        ENG_C1_MEGA_AB: the ACTIVE arm's adapter (both built on first use)."""
        ab = getattr(self, "_c1m_ab", None)
        if ab is not None:
            return ab[self._ab_i]
        if C1_MEGA_AB:
            self._c1m_ab = [self._c1m_load(p) for p in C1_MEGA_AB]
            # (no self.stats here: c1m() runs in __init__ before self.stats exists; c1m_owner counts flips at step time)
            self._ab_i, self._ab_n = C1_MEGA_AB_START, 0
            self.log(f"engine: ENG_C1_MEGA_AB arms a={C1_MEGA_AB[0]} b={C1_MEGA_AB[1]}, flip every {C1_MEGA_AB_N} C1 steps, "
                     f"start {'ab'[C1_MEGA_AB_START]}")
            return self._c1m_ab[self._ab_i]
        x = getattr(self, "_c1m", None)
        if x is None:
            p = os.path.join(PHYS, os.environ.get("C1M_PKG", "kernels/c1mega"))   # serving: kernels/c1mega/srv (frozen)
            if p not in sys.path:
                sys.path.insert(0, p)
            import c1mega_eng
            x = self._c1m = c1mega_eng.C1Mega(self, INDEX_LAYERS, ENGRAM_LAYERS,
                                              cut=LAUNCH_SPLIT_AT if LAUNCH_SPLIT else None, log=self.log)
        return x

    def _c1m_load(self, pkg: str):
        """ENG_C1_MEGA_AB: one package's adapter, imported in isolation: its own
        module name, its directory first on sys.path and its own c1idx_host visible while it builds and loads its weights;
        sys.path and the previous c1idx_host module are restored afterwards."""
        import importlib.util
        d = os.path.join(PHYS, pkg)
        f = os.path.join(d, "c1mega_eng.py")
        assert os.path.isfile(f), f"ENG_C1_MEGA_AB: no adapter at {f}"
        name = "c1mega_eng_" + os.path.basename(os.path.normpath(d))
        saved = (list(sys.path), sys.modules.pop("c1idx_host", None))
        try:
            sys.path.insert(0, d)
            spec = importlib.util.spec_from_file_location(name, f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            x = mod.C1Mega(self, INDEX_LAYERS, ENGRAM_LAYERS, cut=LAUNCH_SPLIT_AT if LAUNCH_SPLIT else None, log=self.log)
            x._weights()
            ih = sys.modules.pop("c1idx_host", None)
            if ih is not None and getattr(x, "ix", None) is not None:
                assert os.path.dirname(os.path.realpath(ih.__file__)) == os.path.realpath(d), (pkg, ih.__file__)
        finally:
            sys.path[:] = saved[0]
            if saved[1] is not None:
                sys.modules["c1idx_host"] = saved[1]
        self.log(f"engine: ENG_C1_MEGA_AB adapter {pkg} ({name}) built")
        return x

    def c1m_owner(self, st: list, batches: list):
        """ENG_C1_MEGA: this step's c1mega owner (the active adapter's owner_of; None = not a C1 step). The ONE place that
        decides it (the sync path _step_split and D13's dispatch). ENG_C1_MEGA_AB: the arm flips here, at the step boundary
        BEFORE owner_of, once ENG_C1_MEGA_AB_N C1 steps ran on the current arm, so a step never mixes arms (its run(),
        _c1m_e0 and graph captures all see c1m() = this step's arm)."""
        ab = self._c1m_ab_list()
        if ab is not None and self._ab_n >= C1_MEGA_AB_N:
            self._ab_i ^= 1
            self._ab_n = 0
            self.stats["c1m_ab_flips"] = self.stats.get("c1m_ab_flips", 0) + 1
        mo = self.c1m().owner_of(st, batches)
        if ab is not None and mo is not None:
            self._ab_n += 1
        return mo

    def _c1m_ab_list(self):
        """ENG_C1_MEGA_AB: both arms' adapters (built on first use), else None."""
        if not C1_MEGA_AB:
            return None
        if getattr(self, "_c1m_ab", None) is None:
            self.c1m()
        return self._c1m_ab

    def _ab_key(self) -> tuple:
        """ENG_C1_MEGA_AB: (active arm,) appended to C1 graph-L keys (each arm captures its own graph L); () otherwise."""
        return (self._ab_i,) if getattr(self, "_c1m_ab", None) is not None else ()

    def _tag_mega(self, st: list, mo) -> None:
        """ENG_C1_MEGA: this step's owner (None: the engine's pieces) on every rank's bucket state, for this serial only."""
        for s_ in st:
            s_["mega"], s_["mega_serial"] = mo, self.step_serial

    def _c1m_e0(self, s: dict):
        """ENG_C1_MEGA with the adapter's in-kernel Engram wkv (C1M_IX=1 + C1IDX_E0=1): its C1Idx on this step's c1mega
        steps (e0_prepare: hash + gathers, the same per-layer events; the wkv runs in the launch's layer 0), else None."""
        if not (C1_MEGA and self._mega_on(s)):
            return None
        ix = getattr(self.c1m(), "ix", None)
        return ix if ix is not None and getattr(ix, "e0_on", False) else None

    def _mega_on(self, s: dict) -> bool:
        """ENG_C1_MEGA: this step's layers run on c1mega (set by _step_split / D13 for this step only)."""
        return C1_MEGA and s.get("mega") is not None and s.get("mega_serial") == self.step_serial

    def rtfx(self):
        """ENG_C1_RTF: kernels/moe_stream/rtf.cu, built into engine/build/eng_rtf on first use (an eager step)."""
        x = getattr(self, "_rtfx", None)
        if x is None:
            from torch.utils.cpp_extension import load
            ns = int(os.environ.get("ENG_C1_RTF_NS", "3"))   # 3 = cuBLAS's split (bitwise); more = same math
            bd = os.path.join(PHYS, f"engine/build/eng_rtf_ns{ns}")
            os.makedirs(bd, exist_ok=True)
            x = self._rtfx = load(name=f"eng_rtf_ns{ns}", sources=[os.path.join(PHYS, "kernels/moe_stream/rtf.cu")],
                                  build_directory=bd, extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                                                                         f"-DRTF_NS={ns}"], extra_ldflags=["-lcuda"], verbose=False)
        return x

    def p_router(self, r: int, s: dict, L: int) -> None:
        lw = self.lw[r][L]
        if "agf_fl" in s and s["moe"] == "raw" and not self._agrt(s) and not (C1_RTF and s["R"] <= 32):   # ENG_C128_AGF
            st = self.bk[s["bkey"]]
            R = s["R"]
            cur = torch.cuda.current_stream()
            ev0 = torch.cuda.Event()
            ev0.record(cur)                         # f ready (the FFN boundary is done)
            sd = self.side[r]
            with torch.cuda.stream(sd):
                sd.wait_event(ev0)
                self.cext.ag_push(s["f"].view(torch.uint8).view(R, D * 2), r, [x["gf"] for x in st],
                                  [x["agf_fl"] for x in st], s["agf_ep"], self.cfg.ncta)
                ev1 = torch.cuda.Event()
                ev1.record(sd)
            s["agf_ev"] = ev1
        if (C1_RTF and s["moe"] == "raw" and s["R"] <= 32 and self.cfg.router_pack3 and s.get("mm_rt") is None
                and lw["gate_w"].dtype == torch.bfloat16):
            if "rtf_ws" not in s:      # (the bucket's first step is eager)
                s["rtf_ws"] = torch.zeros(16 * 32 * 384, dtype=torch.float32, device=s["dev"])
                s["rtf_tk"] = torch.zeros(4, dtype=torch.int32, device=s["dev"])
            R = s["R"]
            self.rtfx().router(s["f"], lw["gate_w"], lw["gate_b"], s["ptok_own"], s["valid"].view(torch.uint8).view(R),
                               s["logits"], s["ids"], s["wts"], s["route_row"], s["rtf_ws"], s["rtf_tk"], 1.5)
            return
        torch.mm(s["f"], lw["gate_w"].t(), out_dtype=torch.float32, out=s["logits"])
        if self._agrt(s):           # ENG_C1_AGRT: the top-k runs inside the AG kernel (p_ag)
            return
        if self.cfg.router_pack3:   # the valid-row mask in-kernel (same bytes; -2 launches per layer)
            if C128_RTOP and s["moe"] == "raw" and 33 <= s["R"] <= 320 and s.get("mm_rt") is None:   # ENG_C128_RTOP
                self.rtopx().top_pack3(s["logits"], lw["gate_b"], s["ids"], s["wts"], 1.5, s["route_row"],
                                       s["valid"].view(torch.uint8), s["ptok_own"], C128_RTOP_W)
                return
            self.mext[r].router_top_pack3(s["logits"], lw["gate_b"], s["ids"], s["wts"], 1.5, s["route_row"], s["valid"],
                                          s["ptok_own"])
            if s.get("mm_rt") is not None:
                self._router_vl(r, s, L)
            return
        assert s.get("mm_rt") is None, "image-token routing needs router_pack3"
        self.mext[r].router_top_pack2(s["logits"], lw["gate_b"], s["ids"], s["wts"], 1.5, s["route_row"], s["R"], s["ptok_own"])
        # padded rows -> ids -1, prescale 0 (device mask: one graph per bucket serves every row count)
        s["route_row"].copy_(torch.where(s["valid"], s["route_row"], s["padrow"]))

    def c1x(self):
        """ENG_C1_COMM: kernels/sp_layer/comm_c1.cu, built into engine/build/eng_comm_c1 on first use (an eager step)."""
        x = getattr(self, "_c1x", None)
        if x is None:
            from torch.utils.cpp_extension import load
            bd = os.path.join(PHYS, "engine/build/eng_comm_c1")
            os.makedirs(bd, exist_ok=True)
            x = self._c1x = load(name="eng_comm_c1", sources=[os.path.join(PHYS, "kernels/sp_layer/comm_c1.cu")],
                                 build_directory=bd, verbose=False,
                                 extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])
        return x

    def rtopx(self):
        """ENG_C128_RTOP: kernels/c128r/rtop.cu, built into engine/build/eng_c128_rtop on first use (an eager step)."""
        x = getattr(self, "_rtopx", None)
        if x is None:
            from torch.utils.cpp_extension import load
            bd = os.path.join(PHYS, "engine/build/eng_c128_rtop")
            os.makedirs(bd, exist_ok=True)
            x = self._rtopx = load(name="eng_c128_rtop", sources=[os.path.join(PHYS, "kernels/c128r/rtop.cu")],
                                   build_directory=bd, verbose=False,
                                   extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a"])
        return x

    def _agrt(self, s: dict) -> bool:
        return (C1_AGRT and C1_COMM and C1_AGW and s["moe"] == "raw" and self.cfg.router_pack3 and s.get("mm_rt") is None
                and not C1_RTF and s["R"] <= C1_FMAXR)

    def p_ag(self, r: int, s: dict, L: int, st: list) -> None:
        R = s["R"]
        ev = s.pop("agf_ev", None)
        if ev is not None:   # ENG_C128_AGF: f went in p_router; the routing rows now, then one wait for both pushes
            self.cext.ag_push(s["route_row"], r, [x["gr"] for x in st], [x["agf_fl"] for x in st], s["agf_ep"], self.cfg.ncta)
            torch.cuda.current_stream().wait_event(ev)
            self.cext.wait(s["agf_fl"], s["agf_ep"], W * self.cfg.ncta)
            return
        if C1_COMM and s["moe"] == "raw":
            if C1_AGW:      # push + wait in one kernel (the last CTA waits)
                tk = s.get("c1_tk")
                if tk is None:
                    tk = s["c1_tk"] = torch.zeros(4, dtype=torch.int32, device=s["dev"])   # (first step: eager)
                if self._agrt(s):
                    self.c1x().ag_rt(s["f"].view(torch.uint8).view(R, D * 2), s["logits"], self.lw[r][L]["gate_b"], 1.5,
                                     s["valid"].view(torch.uint8).view(R), s["ptok_own"], s["ids"], s["wts"], s["route_row"], r,
                                     [x["gf"] for x in st], [x["fl_f"] for x in st], [x["gr"] for x in st], s["ep_f"],
                                     self.cfg.ncta, s["fl_f"], W * self.cfg.ncta, tk)
                    return
                self.c1x().ag_push2v_wait(s["f"].view(torch.uint8).view(R, D * 2), s["route_row"], r, [x["gf"] for x in st],
                                          [x["fl_f"] for x in st], [x["gr"] for x in st], s["ep_f"], self.cfg.ncta,
                                          s["fl_f"], W * self.cfg.ncta, tk)
                return
            self.c1x().ag_push2v(s["f"].view(torch.uint8).view(R, D * 2), s["route_row"], r, [x["gf"] for x in st],
                                 [x["fl_f"] for x in st], [x["gr"] for x in st], s["ep_f"], self.cfg.ncta)
            self.cext.wait(s["fl_f"], s["ep_f"], W * self.cfg.ncta)
            return
        self.cext.ag_push2(s["f"].view(torch.uint8).view(R, D * 2), s["route_row"], r, [x["gf"] for x in st],
                           [x["fl_f"] for x in st], [x["gr"] for x in st], s["ep_f"], self.cfg.ncta)
        self.cext.wait(s["fl_f"], s["ep_f"], W * self.cfg.ncta)

    def p_moe(self, r: int, s: dict, L: int) -> None:
        if "moe" in CXX_PIECES and s["moe"] == "prod" and not torch.cuda.is_current_stream_capturing() and \
                self._cxx_drv().piece("moe", r, s, L):
            return                                          # fork); agrs's p_moe wrapper already waited agrs_in
        if s["moe"] == "prod" and (s.get("vcount", 1 << 30) > self.cfg.prefill_moe_switch or "ecs" not in s):
            return self.p_moe_prod(r, s, L)
        lw, e = self.lw[r][L], self.mext[r]
        Tall = W * s["R"]
        fall = s["gf"].view(torch.bfloat16).view(Tall, D)
        cur = torch.cuda.current_stream()
        ev0 = torch.cuda.Event()
        ev0.record(cur)
        p8s = lw.get("A8sh")
        if p8s is not None and not (s["moe"] != "raw" or s["R"] > getattr(EA, "DENSE8_MAXR", 32)):
            p8s = None   # (ENG_C128_DENSE8_ADD sh: decode buckets above DENSE8_MAXR rows and mixed steps only)
        with torch.cuda.stream(self.side[r]):        # shared expert slice on the side stream (stock rounding points)
            self.side[r].wait_event(ev0)
            if p8s is not None:   # 8-bit arm (act_quant 1x128 + fp8_gemm_nt); swiglu's rounding points unchanged
                EA._a8()._gemm(fall, p8s["sh13"], s["gu"])
                e.swiglu(s["gu"], s["hsh"])
                EA._a8()._gemm(s["hsh"], p8s["sh2"], s["sh"])
            else:
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
        if ROUTE_STATS and "rstat" in s and s["moe"] == "raw":   # EC count (= distinct experts while none has > 64 pairs)
            s["rstat"][L, :1].add_(s["counts"][:1])
            s["rstat"][L, 1:].add_(1)
        x2 = MOE_X2 and gx is not e and Tall <= 1365 and s["moe"] == "raw" and s["R"] >= MOE_X2_MINR   # ENG_MOE_X2 (test)
        if x2:
            xm = self.moe_x2()
            if "x2ctr" not in s:   # queue counters (left zeroed by the kernel); per-32 token scales (slots x 160 floats) in
                s["x2ctr"] = torch.zeros(4, dtype=torch.int32, device=s["dev"])   # the img buffer's unused second half
                nsl = s["img"].numel() // 10240
                s["x2asc"] = (s["img"][nsl * 5120: nsl * 5120 + nsl * 640].view(torch.float32) if MOE_X2_ASC == "32"
                              else s["x2ctr"].view(torch.float32))
            xm.gather_w8(fall, s["ecs"], s["counts"], s["img"], s["x2asc"], 0)   # e4m3 image in the img buffer's first half
        elif G6 and Tall <= 1365 and s["moe"] == "raw":       # ENG_G6: decode buckets (the range checked bit-identical)
            self.g6().gather6(fall, s["ecs"], s["counts"], s["img"], s["gr"], s["sms"])
        else:
            e.prepr5(fall, s["ecs"], s["counts"], s["img"], s["gr"], s["sms"], s["g5ctr"])
        if x2:   # ENG_MOE_X2: W4A8 GEMM1 -> moe_v4's f16 H image -> the served gemm2r
            xm.gemm1(lw["w13"], lw["s13"], s["ecs"], s["counts"], s["img"], s["himg"], s["x2ctr"], s["x2asc"], s["x2asc"], 0)
            gx.gemm2r(lw["w2"], lw["s2"], s["ecs"], s["t2"], s["counts"], s["himg"], s["Y"], 0)
        elif C1_MOEF and gx is not e and s["R"] <= C1_MOEF_MAXR and s["moe"] == "raw":
            gx.gemm12(lw["w13"], lw["s13"], lw["w2"], lw["s2"], s["ecs"], s["t1"], s["t2"], s["counts"], s["img"], s["himg"],
                      s["Y"], s["g12sync"])
        else:
            if C1_G1KS > 1 and gx is not e and s["R"] <= 32 and s["moe"] == "raw":   # the GEMM1 parts use Y as fp32 scratch
                gx.gemm1rk(lw["w13"], lw["s13"], s["ecs"], s["t1"], s["counts"], s["img"], s["himg"], 0,
                           s["Y"].view(torch.float32), s["g1tick"], C1_G1KS)
            else:
                gx.gemm1r(lw["w13"], lw["s13"], s["ecs"], s["t1"], s["counts"], s["img"], s["himg"], 0)
            gx.gemm2r(lw["w2"], lw["s2"], s["ecs"], s["t2"], s["counts"], s["himg"], s["Y"], 0)
        cur.wait_event(ev1)
        if not (C1_RSC and C1_COMM and s["moe"] == "raw" and s["R"] <= C1_FMAXR) and "rs_lb128" not in s:
            e.combine(s["Y"], s["slotmap"], s["sh"], s["part"])   # (ENG_C1_RSC / ENG_C128_RSL: combined in p_rs's push)
        if DUMP_MOE and s["moe"] == "prod" and self.stats.get("mixed_steps", 0) == 0:
            self._dump_moe(r, s, L, Tall, fall)

    def g6(self):
        """ENG_G6: kernels/moe_v4's gather6 extension (kernels/moe_v4/g6.cu, NSLOT 8, B 2; per-device SMEM attribute and
        CUDAGuard inside). Built once into engine/build/eng_moe_g6 (prebuilt on the CPU), loaded on first use (a
        bucket's eager first step, before its capture)."""
        x = getattr(self, "_g6", None)
        if x is None:
            from torch.utils.cpp_extension import load
            if G6_NOPADW:   # ENG_G6_NOPADW: the copy g6x.cu, padding slots not written
                os.makedirs(os.path.join(PHYS, "engine/build/eng_moe_g6x"), exist_ok=True)
                x = self._g6 = load(name="eng_moe_g6xs8b2", sources=[os.path.join(PHYS, "kernels/moe_v4/g6x.cu")],
                                    build_directory=os.path.join(PHYS, "engine/build/eng_moe_g6x"), verbose=False,
                                    extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                                                       "-DG6_NSLOT=8", "-DG6_B=2", "-DG6_NOPADW=1"])
                return x
            os.makedirs(os.path.join(PHYS, "engine/build/eng_moe_g6"), exist_ok=True)
            x = self._g6 = load(name="eng_moe_g6s8b2", sources=[os.path.join(PHYS, "kernels/moe_v4/g6.cu")],
                                build_directory=os.path.join(PHYS, "engine/build/eng_moe_g6"), verbose=False,
                                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a",
                                                   "-DG6_NSLOT=8", "-DG6_B=2"])
        return x

    def _dump_route_stats(self) -> None:
        """ENG_ROUTE_STATS: per decode bucket (rank 0's state; the route tables are identical on all ranks), the mean EC
        count per layer and step, and the steps counted."""
        import json
        out = {}
        for key, st in self.bk.items():
            s0 = st[0] if isinstance(st, list) else None
            if not isinstance(key, int) or s0 is None or "rstat" not in s0:
                continue
            a = s0["rstat"].cpu().tolist()
            n = max(x[1] for x in a)
            if n:
                out[str(key)] = dict(steps=n, ec_per_layer=[round(x[0] / x[1], 3) if x[1] else None for x in a[:64]])
        tmp = ROUTE_STATS + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, ROUTE_STATS)

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

    def moe_x2(self):
        """ENG_MOE_X2 (quality test): kernels/moe_x2 (W4A8 GEMM1 + gather_w8 on the raw FP4 layout, X2_PRE=0), imported
        and JIT-built at engine init with the other extensions (engine/build/moe_x2, MOE_X2_BUILD; prebuild it in the slot
        environment). ENG_MOE_X2 unset: never imported, never built."""
        if getattr(self, "_x2", None) is None:
            p = os.path.join(PHYS, "kernels")
            if p not in sys.path:
                sys.path.insert(0, p)
            from moe_x2 import moe_x2 as X2
            d = {"X2_W8": "1", "X2_PRE": "0", "X2_ASC": "1" if MOE_X2_ASC == "32" else "0", "X2_FASTEPI": "1", "X2_SLDS": "1"}
            self._x2 = X2.load(d, build_root=os.environ.get("MOE_X2_BUILD", os.path.join(PHYS, "engine/build/moe_x2")))
        return self._x2

    def moe_v4(self):
        """(ext, route2) of kernels/moe_v4/v4.py (read-only import; built into engine/build/moe_v4, MOE_V4_BUILD)."""
        if getattr(self, "_v4", None) is None:
            p = os.path.join(PHYS, "kernels")
            if p not in sys.path:
                sys.path.insert(0, p)
            from moe_v4 import v4
            self._v4 = v4.load()
        return self._v4

    def _preload_attn(self) -> None:
        """ENG_ATTN_PRELOAD (default on): the attention / indexer extensions this configuration can reach in a step but
        loads lazily, loaded here through the loader its step-path caller uses (each caches its module: the step path gets
        the same one; module load only, nothing allocated on the GPU), under the caller's own conditions:
          attn_hg      attn.attn_hg: attn.attn_b_attend_o, i.e. TP-wo decode buckets (ENG_C1_TPWO) with ENG_C1_ATTN_HG
          idx2f ftopk  idx2f.ext_ftopk (prebuild_ext lists it): OwnedIndexer's fused top-k / select_blocks(from_lens) /
                       fused prep2, reached by the idx2f indexer (cfg.indexer "idx2f")"""
        t0, done = time.time(), []
        if C1_TPWO and getattr(EA, "C1_ATTN_HG", False):
            EA.attn_hg()
            done.append("attn_hg")
        hi = getattr(self.hooks, "idx", None)
        if self.cfg.indexer == "idx2f" and hi is not None and hasattr(getattr(hi, "X", None), "ext_ftopk"):
            hi.X.ext_ftopk()
            done.append("idx2f ftopk")
        self.log(f"engine: ENG_ATTN_PRELOAD: {', '.join(done) or 'nothing to load'} at init ({time.time() - t0:.2f} s)")

    def _preload_moe(self) -> None:
        """ENG_MOE_PRELOAD (default on): every routed-MoE extension this configuration can reach, loaded here at init
        through the loader its step-path caller uses (each caches its module or object, so the step path gets the same
        one; nothing is allocated on the GPU). Without it the first decode / mixed step of a run pays the JIT loads
        (a ninja subprocess each, also when up to date), and a changed source would rebuild mid-serving.
          moe_v4 (api + route)      Engine.moe_v4: p_moe's decode buckets (cfg.moe_v4) and moe_v4-mixed buckets
          g6                        Engine.g6: ENG_G6 decode buckets
          moe_x2                    Engine.moe_x2: ENG_MOE_X2 (already at the extensions step; cached)
          pmf / pmf8                pm_fused (pmv for ENG_PM_NT, pm2 for ENG_PF_ACT8 + PF_ACT8_DEC): p_moe_prod's fused calls
          ds41h_moe                 production's extension (prod_moe_ext): p_moe_prod's unfused calls (ENG_PM_FUSED=0)
          agrs frs                  agrs_engine's RSF layer object, unless agrs_ws seeds it after init (ENG_AGRS_WS, PM_NT 0)
          agrs comm                 commbench (agrs_decode's AGRS_RSD combine push; agrs_engine's AGRS_AGCE copies)
          moe_v4c (api + route)     the C++ moe piece's tables (mixed_cxx_moe.setup at the driver's first use; v4_cabi.load)
          drafter v4                kernels/head/drafter.v4_ext: HSD drafter MoE (cfg.hsd_moe "v4") and the H2 head recorder
        """
        t0, done = time.time(), []
        p = os.path.join(PHYS, "kernels")
        if p not in sys.path:
            sys.path.insert(0, p)
        if self.cfg.moe_v4:
            self.moe_v4()
            done.append("moe_v4")
        if G6:
            self.g6()
            done.append("g6")
        if MOE_X2:
            self.moe_x2()
            done.append("moe_x2")
        if PM_FUSED:
            self.pmf()
            done.append(f"pmf (PM_NT {PM_NT})")
            if PF_ACT8 and PF_ACT8_DEC:
                self.pmf8()
                done.append("pmf8")
        else:
            self.prod_moe().ext()
            done.append("ds41h_moe")
        mod = sys.modules[__name__]
        if getattr(mod, "AGRS_RSF", False) and PM_FUSED:
            import agrs_engine as AE
            m_ = AE.nt_mode(mod)
            if m_ != "0" or os.environ.get("ENG_AGRS_WS", "0") == "0":   # (agrs_ws seeds _frs["0"] right after __init__)
                AE.frs(self, m_)
                done.append(f"agrs frs {m_}")
        if getattr(mod, "AGRS_RSD", False):
            import agrs_decode
            agrs_decode.ag_ext()
            done.append("agrs comm (RSD)")
        if getattr(mod, "AGRS_AGCE", False):   # (commbench.ext is not cached: agrs_engine.ce_ext keeps its own handle)
            import agrs_engine as AE
            AE.ce_ext()
            done.append("agrs comm (AGCE)")
        if "moe" in CXX_PIECES:
            p4 = os.path.join(PHYS, "kernels", "moe_v4")
            if p4 not in sys.path:
                sys.path.insert(0, p4)
            import v4_cabi as V4C
            V4C.load()
            done.append("moe_v4c")
        if getattr(self, "hsd", None) is not None and (self.cfg.hsd_moe == "v4"
                                                       or os.environ.get("ENG_MIXED_HEAD_KIND", "graph") == "prog"):
            dr = next((m for m in list(sys.modules.values()) if m is not None and hasattr(m, "v4_ext")
                       and getattr(m, "__name__", "").endswith("drafter")), None)   # (as mixed_cxx_head2 finds it)
            if dr is not None:
                dr.v4_ext()
                done.append("drafter v4")
        self.log(f"engine: ENG_MOE_PRELOAD: {', '.join(done) or 'nothing'} loaded at init in {time.time() - t0:.2f} s")

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
        if C1_L2PF and L + 1 in self.layers:
            self._pf_issue(r, s, L + 1, "qkv")
        if C1_COMM and s["moe"] == "raw":
            x = self.c1x()
            if "rs_lb128" in s:   # ENG_C128_RSL: combine + push + poll + sum, one kernel per rank
                x.rs_cpushl(s["Y"], s["slotmap"], s["sh"], s["R"], r, [y["rs_lb128"] for y in st], s["rs_ltk128"],
                            s["moe_out"], s["gr"], C128_RSL_GRID)
                return
            if C1_RSL and C1_RSC and s["R"] <= C1_FMAXR and "rs_lb" in s:   # ENG_C1_RSL: combine + push + poll + sum
                x.rs_cpushl(s["Y"], s["slotmap"], s["sh"], s["R"], r, [y["rs_lb"] for y in st], s["rs_ltk"],
                            s["moe_out"], s["gr"], W * self.cfg.ncta)
                return
            if C1_RSC and s["R"] <= C1_FMAXR:
                x.rs_cpushv(s["Y"], s["slotmap"], s["sh"], s["R"], r, [y["rs"].view(torch.uint8) for y in st],
                            [y["fl_s"] for y in st], s["ep_s"], self.cfg.ncta, s["gr"])
            else:
                x.rs_pushv(s["part"], r, [y["rs"].view(torch.uint8) for y in st], [y["fl_s"] for y in st], s["ep_s"],
                           self.cfg.ncta, s["gr"])
            nxt = L + 1
            if (C1_FOLD and MHCF and self.mhcf is not None and nxt in self.layers and nxt not in ENGRAM_LAYERS
                    and s["R"] <= 320 and self.lw[r][nxt].get("fa_f") is not None):
                s["fold_x"] = (s["rs"], s["fl_s"], s["ep_s"], W * self.cfg.ncta)   # summed in layer L+1's boundary
                return
            x.rs_sumv(s["rs"], s["fl_s"], s["ep_s"], W * self.cfg.ncta, s["moe_out"], 132, s["gr"], r)
            return
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
        if CXX_RUN and trace is None and st[0].get("moe") == "prod":
            d_ = self._cxx_drv()                                         # driver's 4 threads; False = not covered
            if sys.modules["mixed_cxx"].issue(d_, st, self.layers if layers is None else layers, final, embed):
                return
        if FAST_EACH:
            fx = [(d.index if isinstance(d, torch.device) else int(d), (sm.stream_id, sm.device_index, sm.device_type))
                  for d, sm in zip(self.devs, self.streams)]
            assert all(dv == sm[1] for dv, sm in fx), "ENG_FAST_EACH: rank stream on another device"

        def each(fn, name="", L=None):
            if PROF_PIECES:
                rf = torch.profiler.record_function(f"piece:{name}")
                rf.__enter__()
            if FAST_EACH:
                d0 = _c_get_dev()
                try:
                    for r in range(W):
                        dv, sm = fx[r]
                        _c_set_dev(dv)
                        prev = _c_get_stream(dv)
                        _c_set_stream(stream_id=sm[0], device_index=sm[1], device_type=sm[2])
                        try:
                            fn(r, st[r])
                        finally:
                            _c_set_stream(stream_id=prev[0], device_index=prev[1], device_type=prev[2])
                finally:
                    _c_set_dev(d0)
            else:
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
        self._tp_st = st                          # ENG_C1_TPWO: p_attn's peers
        for L in (self.layers if layers is None else layers):
            if self._mega_on(st[0]) and L in self.c1m().member:   # ENG_C1_MEGA: the run starting at L (no-op inside one)
                each(lambda r, s: (("c1mega" in CXX_PIECES and not torch.cuda.is_current_stream_capturing()   # stage 4:
                                    and self._cxx_drv().piece("c1mega", r, s, L))
                                   or self.c1m().run(r, s, L, st)), "c1mega", L)
                continue
            each(lambda r, s: self.p_attn_bd(r, s, L), "attn_bd", L)
            if cp is not None and L <= 20:     # CP prefill: every send of a phase issued for all ranks before a receive
                each(lambda r, s: self._cp_p1(r, s, L), "attn P1", L)
                each(lambda r, s: self._cp_p2(r, s, L), "attn P2", L)
                each(lambda r, s: self._cp_p3(r, s, L), "attn P3", L)
            else:
                each(lambda r, s: self.p_attn(r, s, L), "attn", L)
                if self._tp_on(st[0]):
                    each(lambda r, s: self.p_tpwo2(r, s, L, st), "tpwo2", L)
                    each(lambda r, s: self.p_tpwo3(r, s, L), "tpwo3", L)
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
        if C1_L2PF:
            each(lambda r, s: self._pf_join(s), "pf_join")

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
        self._tp_st = st
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
                        self.p_tpwo2(r, s, L, st)
                        self.p_tpwo3(r, s, L)
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
                  and (sum(sb.n_rows for sb in batches) > 64 or LONG_ANY_ROWS))
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
            R = self.bucket_of(max(max(sb.n_rows for sb in batches), 1), pad=False) + lvl * LVL_TAG
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
            if ROUTE_STATS and self.stats["steps"] % 256 == 0:
                self._dump_route_stats()
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
            if ROUTE_STATS and self.stats["steps"] % 256 == 0:
                self._dump_route_stats()
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
        if ROUTE_STATS and self.stats["steps"] % 256 == 0:
            self._dump_route_stats()
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
        """Tests: drop every per-run state so the
        next run starts like a fresh engine; the caller re-initializes the pools' CONTENTS (synth.init_pools, seeded).
        Dropped / reset: graphs (and the graph pools, which die with their last graph), bucket states (their allocation
        follows config flags), D13 / CP state, host bookkeeping, the table builders' delta state, the hooks' per-slot
        device state (Engram token rings / hashes / dedup; HSD's per-slot store, next ids, emitted counts, sampler
        slot states, blob parity) and the indexer's per-slot block table."""
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
        hs = getattr(H.hsd, "main", None) or H.hsd   # (FPR_C1_KMAX: never a lane outside D13 dispatch; be safe)
        H.hsd = hs
        if hs is not None and getattr(hs, "lane", None) is not None:
            hs.lane.reset()                            # FPR_C1_KMAX: the C1 lane HSD (hsdt_c1k) too
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
            if hasattr(H.idx, "mirror_reset"):   # lever (a): the index-key mirror's tables (no-op when off)
                H.idx.mirror_reset()
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
        """Graph H's key: (R, "H", HSD's bucket key) when HSD runs this step without a grammar bitmask; a grammar step:
        (R, "H", key + ("bm",)) with ENG_HEAD_BM_GRAPH, else None (head_eager)."""
        k = self.hooks.hsd_key()
        if k is None:
            return None
        if self.hooks.bm_step:
            return (R, "H", k + ("bm",)) if HEAD_BM_GRAPH else None
        return (R, "H", k)

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
            self._tp_st = st
            gs = []
            cut = LAUNCH_SPLIT_AT if (LAUNCH_SPLIT and self.layers[0] < LAUNCH_SPLIT_AT <= self.layers[-1]
                                      and (self.hooks.engram is None or max(ENGRAM_LAYERS) < LAUNCH_SPLIT_AT)) else None

            def layer(r, s, L):
                if self._mega_on(s) and L in self.c1m().member:   # ENG_C1_MEGA: the run starting at L (no-op inside one)
                    self.c1m().run(r, s, L, st)
                    return
                self.p_attn_bd(r, s, L)
                self.p_attn(r, s, L)
                self.p_tpwo2(r, s, L, st)
                self.p_tpwo3(r, s, L)
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
        mo = self.c1m_owner(st, batches) if C1_MEGA else None   # ENG_C1_MEGA: C1 steps get their own graph L
        self._tag_mega(st, mo)
        lk = (Rk, "L") if mo is None else (Rk, "L", "M", mo) + self._ab_key()   # (ENG_C1_MEGA_AB: + the arm)
        self._last_gkey[Rk] = lk
        if lk not in self.graphs:
            # first step of this bucket: eager layers + HSD (the real step), then capture graph L and graph H
            self._new_scheds(st)
            self._issue(st)
            self._prepare_head_on_streams(st, batches)
            Hk.head_eager(st)
            self.sync()
            res = [Hk.head_result(r, st[r]) for r in range(W)]
            self._capture_split(Rk, st, True, self._hkey(Rk), lkey=lk)
            self.last_st = st
            self.stats["steps"] += 1
            if ROUTE_STATS and self.stats["steps"] % 256 == 0:
                self._dump_route_stats()
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
        if ROUTE_STATS and self.stats["steps"] % 256 == 0:
            self._dump_route_stats()
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
        if mo is not None and self._ab_key():   # ENG_C1_MEGA_AB: this C1 step's arm (PHYS_STEP_TIMES "eng" dict)
            self.stats["decode_last"]["ab"] = self._ab_i
        return out

    # ------------------------------------------------------------------------------------------------ mixed steps
    def step_mixed(self, batches: list, tables: list) -> list:
        """A step with prefill chunks (and decode rows), eager. Layers first..20 run all rows (bucket A); layers 21..39 run
        StepBatch.tail_rows only (decode rows + the last min(n, 128) rows of each chunk: production's decoder-side bounded
        replay) on bucket T, whose input stream is gathered from A after layer 20; the head runs the sample rows (a subset
        of the tail rows). MoE: production's kernels on every layer of the step (prefill rows present)."""
        # step_mixed = _mixed_body (A + tail gather + T issued, no sync) +
        # _mixed_post (the head launch, events, sync, bookkeeping) + the head results, in exactly today's op order. nd's
        # native mixed step calls the two halves itself, with the head's prepare / launch / results native.
        if getattr(self, "_h1_precap_due", False):   # ENG_MIXED_HEAD_PRECAPTURE: Python-driven mixed steps only (D13: after
            self._h1_precapture()                      # _fallback's drain; nd's native mixed calls the halves, not this)
        ctx = self._mixed_body(batches, tables)
        if ctx["out"] is not None:           # no tail segment (layer subsets): bucket A's head ran in the body
            return ctx["out"]
        self._mixed_post(ctx, batches)
        stT = ctx["stT"]
        out = [self.hooks.head_result(r, stT[r], batches[r]) for r in range(W)]
        self._cp = None
        if DBG_SYNC:
            self._dbg_sync("mixed head_result (gate head / HSD results)")
        return out

    def _mixed_body(self, batches: list, tables: list, prep_head: bool = True) -> dict:
        """step_mixed's body: bucket A prepared and issued, the tail gather and bucket T prepared and issued; no sync (the
        work is queued on the rank streams when it returns). prep_head=False: the head's host prepare (Hooks.prepare_head) is
        the caller's (nd's native mixed step: after T's issue, stream-ordered before its H1 head). Returns the context
        _mixed_post needs; ctx["out"] is not None when there is no tail segment and the step already completed."""
        cfg = self.cfg
        if CXX_PIECES and getattr(self, "_cxx", None) is None and os.environ.get("ENG_MIXED_CXX_EARLY_DRV", "1") == "1":
            # the decode-PRECREATE done point did not create the driver (precreate off, or a
            # native loop that never completes it): create it now, before this step issues anything, with every device
            # synced, so mixed_cxx_ltwarm's cuBLASLt module loads never overlap queued cross-rank work
            for d_ in self.devs:
                torch.cuda.synchronize(d_)
            try:
                self._cxx_drv()
            except Exception as e_:  # noqa: BLE001
                self.log(f"engine: ENG_MIXED_CXX_EARLY_DRV (first mixed step) FAILED ({type(e_).__name__}: {str(e_)[:200]})")
            for d_ in self.devs:
                torch.cuda.synchronize(d_)
        tw0 = time.perf_counter()
        self._busy = set()
        if self.mix_pool is not None:   # ENG_MIX_POOL_GB: drop the last step's state (debug dumps only) so an evicted
            self.last_st = None         # bucket's blocks really return to the pool
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
        pre_T = None
        if MIXED_PRESTEP and segT and not cfg.mixed_graphs:   # ENG_MIXED_PRESTEP: the T bucket (and its all-device syncs) now
            tb_ = [tail_batch(sb) for sb in batches]          # (the same lookup order as below: A first, then T)
            nT_ = [b.n_rows for b in tb_]
            RT_ = rnd(max(nT_))
            tgr_ = MIXED_GRAPHS_T and not cfg.mixed_graphs and not DBG_SYNC and self._cp is None and not l20
            if tgr_:
                RT_ = max(RT_, min(-(-max(nT_) // MIXED_T_ROUND) * MIXED_T_ROUND, self.max_tail_rows))
            pre_T = (tb_, nT_, RT_, tgr_, self.bucket(RT_, "prod", "T"))
            if getattr(self, "AGRS_RSF", False) or globals().get("AGRS_RSF", False):
                try:
                    import agrs_engine as AGE_                 # the fused-RS flags (first-use all-device sync) now, not
                    for st_ in (stA, pre_T[4]):                # from inside a rank's p_moe_prod
                        if st_[0]["R"] >= getattr(AGE_, "MIN_R", 1024):
                            AGE_._ensure(self, st_, "rsf")
                except ImportError:
                    pass
        if CXX_PIECES:                                      # ENG_MIXED_CXX_PIECES: bucket A's per-rank StepParams
            self._cxx_drv().set_step(stA)
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
            return dict(out=out)
        rfT = torch.profiler.record_function("piece:host tail gather + prepare T") if PROF_PIECES else None
        if rfT is not None:
            rfT.__enter__()
        if pre_T is not None:                                 # ENG_MIXED_PRESTEP: looked up before A's issue
            tb, nT, RT, tgr, stT = pre_T
        else:
            tb = [tail_batch(sb) for sb in batches]
            nT = [b.n_rows for b in tb]
            RT = rnd(max(nT))
            tgr = MIXED_GRAPHS_T and not cfg.mixed_graphs and not DBG_SYNC and self._cp is None and not l20   # ENG_MIXED_GRAPHS_T
            if tgr:
                RT = max(RT, min(-(-max(nT) // MIXED_T_ROUND) * MIXED_T_ROUND, self.max_tail_rows))
            stT = self.bucket(RT, "prod", "T")
        if POISON:
            for s_ in stT:
                self._poison_state(s_)
        for r in range(W):
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                tab = self.tb_tail.build(tb[r], [int(tables[r].tokens[i]) for i in batches[r].tail_rows])
                self._mixed_upload_T(r, stT[r], tab, tb[r])
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
                    if not TAIL_GATHER_KERNEL:            # (T_STAGE: its pads write rows [n, R) from si / sl)
                        sT["si20"].copy_(sT["si"])        # padding rows as the T tables have them
                        sT["sl20"].copy_(sT["sl"])
                    c20["si"], c20["sl"] = sT["si20"], sT["sl20"]
                elif c20 is not None:
                    c20["si"], c20["sl"] = sT["si"], sT["sl"]
                sT["l20"] = l20
                sT["ws"]["pf_ab"] = None
                if TAIL_GATHER_KERNEL:           # T_STAGE: gather + padding rows, one launch (n 0: padding only)
                    ix = self._tail_ix(r, batches[r].tail_rows) if n else None
                    self._t_stage(sA, sT, ix, n, l20)
                if n:
                    if TAIL_GATHER_KERNEL:
                        pass
                    elif TAIL_GATHER:            # ENG_TAIL_GATHER (#7): persistent ix, gathers written in place
                        ix = self._tail_ix(r, batches[r].tail_rows)
                        self._tail_gather(sA, sT, ix, n, l20)
                    else:
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
                for il in (INDEX_LAYERS if BISECT["clear_tail_padding"] and not TAIL_GATHER_KERNEL else ()):   # padding rows: no keys (not stale)
                    if il <= 20:
                        sT["ti"][il][n:].fill_(-1)
                        sT["tl"][il][n:].zero_()
                sT["chain_input"] = True
        self._set_vidx(stT, nT, [dec_prefix(b) for b in tb])
        if prep_head:
            self.hooks.prepare_head(stT, batches)
        if rfT is not None:
            rfT.__exit__(None, None, None)
        self._new_scheds(stT, per_layer=BISECT["per_layer_plans"])
        for s_ in stT:
            s_.pop("_dbg_ref", None)
        if DBG_SYNC:
            self._dbg_sync(f"mixed tail gather + prepare T R{RT} rows {nT}")
            self._dbg_devices(stT, f"mixed T R{RT}")
        if CXX_PIECES:                                      # ENG_MIXED_CXX_PIECES: bucket T's per-rank StepParams
            self._cxx_drv().set_step(stT)
        if (cfg.mixed_graphs or tgr) and not DBG_SYNC:
            self._issue_mixed(stT, segT, final=True, embed=False)
            if tgr:
                self.stats["mixed_t_graph_steps"] = self.stats.get("mixed_t_graph_steps", 0) + 1
        else:
            if l20:
                self._l20_tail(stT)
            self._issue(stT, final=True, layers=segT, embed=False)
        return dict(out=None, stA=stA, stT=stT, evs=evs, tw0=tw0)

    def _mixed_upload_T(self, r: int, stT_r: dict, tab, tb_r) -> None:
        """bucket T's step-input upload for rank r inside _mixed_body's T loop (on rank r's device
        and stream, right after tb_tail.build): the tail tables into the bucket's buffers + the state's tab / batch.
        nd's native T prepare overrides it; this default is today's code. (_img_rows_T stays at
        the call site: it reads the A tables' tokens at the tail rows and sets the state's image-routing entries.)"""
        self._upload(r, stT_r, tab)
        stT_r["tab"], stT_r["batch"] = tab, tb_r

    def _mixed_post(self, ctx: dict, batches: list, head_eager: bool = True) -> None:
        """step_mixed after _mixed_body: the head launch (head_eager=False: the caller launched it, nd's native H1), the
        step's end events, the sync, the head-key capture (ENG_MIXED_CXX_HEAD), workspace / memory / step statistics."""
        stA, stT, evs, tw0 = ctx["stA"], ctx["stT"], ctx["evs"], ctx["tw0"]
        if head_eager:
            self.hooks.head_eager(stT, batches)
            if DBG_SYNC:
                self._dbg_sync("mixed head")
        self.last_st = stT
        tw1 = time.perf_counter()
        for r in range(W):
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(self.streams[r]):
                evs[r][1].record()
        self.sync()
        if MIXED_CXX_HEAD:                   # ENG_MIXED_CXX_HEAD: capture this step's head key if admitted (nothing runs)
            _mxh().after_sync(self.hooks, stT, batches)
        tw2 = time.perf_counter()
        if self._cp is not None:
            self._cp.done()
        if MEM_TRACE:
            ws_b = [int(s_["pf"].buf.numel()) * 2 if s_.get("pf") is not None and getattr(s_["pf"], "buf", None) is not None
                    else 0 for s_ in stA]
        for s_ in stA:                       # ENG_ATTN_PF: the plans' workspaces (~1 KB per gathered record) go now
            pl_ = s_.get("pf")
            if self.pf_ws is not None and pl_ is not None and hasattr(pl_, "stats_ws"):   # ENG_PF_WS_PREALLOC_MB counters
                for k_, v_ in pl_.stats_ws.items():
                    self.stats["pf_ws_" + k_] = self.stats.get("pf_ws_" + k_, 0) + v_
            s_["pf"] = None
        if self.pf_ws is not None and self.stats.get("mixed_steps", 0) % 200 == 199:
            self.log(f"engine: ENG_PF_WS_PREALLOC_MB fallbacks after {self.stats['mixed_steps'] + 1} mixed steps: shared layout "
                     f"{self.stats.get('pf_ws_shared_fallbacks', 0)}, runtime allocation {self.stats.get('pf_ws_alloc_fallbacks', 0)}")
        self._mem_after_step()
        if MEM_TRACE:
            pf = [se for sb in batches for se in sb.seqs if se.kind != "decode"]
            dec = sum(se.n_rows for sb in batches for se in sb.seqs if se.kind == "decode")
            nb_raw = sum(1 for k in self.bk if isinstance(k, int))
            nb_mix = len(self.bk) - nb_raw
            self.log("memtrace mixed step: prefill rows %d seqs %d max ctx %d, decode rows %d, pf workspace MB %s, step peak "
                     "allocated GB %s, allocated GB %s, reserved GB %s, buckets raw %d mixed %d (mixed held GB %.2f), graphs %d, "
                     "seqs running %d%s" % (sum(se.n_rows for se in pf), len(pf),
                                          max((se.start + se.n_rows for se in pf), default=0), dec,
                                          "/".join(str(b >> 20) for b in ws_b),
                                          "/".join(f"{m[2]:.2f}" for m in self.stats["mem_last"]),
                                          "/".join(f"{m[0]:.2f}" for m in self.stats["mem_last"]),
                                          "/".join(f"{m[1]:.2f}" for m in self.stats["mem_last"]), nb_raw, nb_mix,
                                          sum(self.bk_bytes.values()) / 1e9 if getattr(self, "bk_bytes", None) else 0.0,
                                          len(getattr(self, "graphs", {})),
                                          sum(len(sb.seqs) for sb in batches),
                                          "" if self.mix_pool is None else
                                          ", mix pool resv/alloc GB %s, buckets %s, grow %d" % (
                                              " ".join(f"{k}:{a_ / 1e9:.2f}/{b_ / 1e9:.2f}" for k in self.mix_pool
                                                       for a_, b_ in [self._pool_bytes(k, 0)]),
                                              " ".join(f"{k[2]}{k[0]}:{v / 1e9:.2f}" for k, v in self.bk_bytes.items()),
                                              self.stats.get("mix_pool_grow", 0))))
        self.stats["mixed_last"] = dict(gpu_ms=round(max(a.elapsed_time(b) for a, b in evs), 2),
                                        issue_ms=round((tw1 - tw0) * 1000, 2), wall_ms=round((tw2 - tw0) * 1000, 2))
        self.stats["steps"] += 1
        if ROUTE_STATS and self.stats["steps"] % 256 == 0:
            self._dump_route_stats()
        self.stats["mixed_steps"] = self.stats.get("mixed_steps", 0) + 1
        if not head_eager:                   # the native caller computes the head results itself: step_mixed's reset
            self._cp = None                  # after head_result happens here

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
                                  self.pools[r].n_paged, **({"ws_buf": self.pf_ws[r]} if self.pf_ws else {}))
        self.stats["attn_pf_steps"] = self.stats.get("attn_pf_steps", 0) + 1

    def _mem_after_step(self) -> None:
        """After a (synchronized) mixed step: per-device memory stats of the step (allocated, cached, step peak), and the
        safety net: if more than FREE_CACHE_BYTES sit cached but unused on any device, return them to the driver (the
        caching allocator otherwise hoards blocks of sizes the next steps no longer request, and allocations outside
        torch, such as lazily loaded kernel modules, then fail)."""
        mem, idle = [], 0
        mfree = 0       # ENG_MIX_POOL_GB: the pool's free blocks are reserved on purpose, not idle cache
        if self.mix_pool is not None:
            for k_, v_ in self.mix_pool_resv.items():
                mfree += max(0, min(v_) - sum(b for kk, b in self.bk_bytes.items() if k_ in ("*", kk[2])))
        for d in self.devs:
            a_, r_, p_ = torch.cuda.memory_allocated(d), torch.cuda.memory_reserved(d), torch.cuda.max_memory_allocated(d)
            mem.append((round(a_ / 1e9, 2), round(r_ / 1e9, 2), round(p_ / 1e9, 2)))
            idle = max(idle, r_ - a_ - mfree)
            torch.cuda.reset_peak_memory_stats(d)
        self.stats["mem_last"] = mem          # per device: (allocated, reserved, step peak allocated) GB
        pk = self.stats.setdefault("mem_peak_win", [0.0] * len(mem))   # the step-log window's max step peak per device
        for i_, m_ in enumerate(mem):
            pk[i_] = max(pk[i_], m_[2])
        if idle > FREE_CACHE_BYTES and BISECT["free_cache"]:
            torch.cuda.empty_cache()
            self.stats["cache_releases"] = self.stats.get("cache_releases", 0) + 1
            self.log(f"engine: cache release #{self.stats['cache_releases']} at {time.time():.3f} (idle {idle / 1e9:.2f} GB > "
                     f"ENG_FREE_CACHE_GB)")   # logging only

    def _set_vidx(self, st: list, ns: list, nds: list | None = None) -> None:
        if MIX_PERSIST:
            return self._set_vidx_persist(st, ns, nds)
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
        if MIXED_PRESTEP:                    # pageable -> pinned: a truly asynchronous H2D (pageable can block the host)
            v, nv = v.pin_memory(), nv.pin_memory()
            nd = nd.pin_memory() if nd is not None else None
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
                                      n_state_slots=_geom_state_slots(), model_dir=cfg.model, tables=tabs, weights=wts)
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
            assert all(b % 4 == 0 for b in cfg.buckets if b != C1_K1L), "decode buckets = 4 x HSD's sequence buckets (k = 3)"
            # HSD's per-sequence state is indexed by serve's state slot (KVManager's state pool, lowest-first). With CP
            # prefill the helper ranks' shadows hold slots too, so a seated sequence's slot can pass max_seqs (80):
            # "state_slot >= HSD max_seqs". Slots = the state pool's (as Engram's ring);
            # shadows never have sample rows (helper chunks are never a last chunk), so they never reach HSD.
            self.hsd_slots = max(cfg.max_seqs, _geom_state_slots())
            # ENG_HSDT: kernels/hsdtail's HSDT (a subclass of HSD) with the listed
            # switches, e.g. "mhcf,par,dmask" (hsdt.py docstring; kernels/head stays untouched)
            HSDcls, hsdt_sw = HSDm.HSD, os.environ.get("ENG_HSDT", "")
            if hsdt_sw:
                p_ = os.path.join(PHYS, "kernels/hsdtail")
                if p_ not in sys.path:
                    sys.path.insert(0, p_)
                import importlib
                # ENG_HSDT_MOD (C1 tail, default "hsdt"): the HSDT module, e.g. hsdt_c1d (+ c1d's per-owner graph-H variant)
                HSDT_ = importlib.import_module(os.environ.get("ENG_HSDT_MOD", "hsdt"))
                HSDcls = HSDT_.HSDT
            self.hsd = HSDcls(eng.devs, eng.reader, eng.mcfg, window_pools=None, max_seqs=self.hsd_slots, k=3,
                                draft_head=cfg.draft_head, mode="peer",
                                seq_buckets=tuple(b // 4 for b in cfg.buckets if b != C1_K1L),   # (default buckets: HSD's own default; FPR_C1_KMAX: the C1 bucket runs on the lane)
                                # tail rows of a mixed step reach the whole bucket (score rows run 21..39 on every
                                # row; 2,048 / 3,579 seen, > 1,536): size HSD's context rows for it
                                max_tail_rows=max(max(cfg.buckets), 1536,
                                                  -(-cfg.max_mixed_rows // cfg.mixed_round) * cfg.mixed_round),
                                prep=cfg.hsd_prep, fuse_rs=cfg.hsd_fold, mhc2=cfg.hsd_mhc2,
                                moe_mode=cfg.hsd_moe)
            if hsdt_sw:
                self.hsd.configure(**HSDT_.parse_switches(hsdt_sw))
                eng.log(f"engine: ENG_HSDT={hsdt_sw}: HSDT switches {self.hsd.sw}")
            if HSD_VIEWS:   # ENG_HSD_VIEWS: after the HSD build / configure, before any draft_rank
                p_ = os.path.join(PHYS, "kernels", "hsdtail")
                if p_ not in sys.path:
                    sys.path.insert(0, p_)
                import hsd_views
                hsd_views.install(self.hsd, log=eng.log)
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

    def replay(self, fresh, dropped) -> None:
        """the PER-REQUEST half of on_ctx for admissions / drops that NATIVE steps
        handled without Python (nd's observe_requests is the authoritative copy): replayed in one call before the next Python
        consumer of this state (an embedded / image step, a run_python job). Exactly on_ctx's per-request effects:
          fresh   = [(rid, ids, temperature, seed, structured)]: ids = the request's history as eng.hist would hold it at the
                    sync point BEFORE push_state's pending tokens (nd Req.hist minus hist_pending), temperature = the request's
                    float64 (from its ADD frame, never nd's float32), seed = the drawn / given int64 (NativeWorker draw_seeds);
          dropped = rids that left (a rid admitted AND dropped since the last sync is in neither: the queue cancels it).
        Fresh before dropped, as on_ctx (finished rids never return, so the two sets are disjoint: asserted).
        The per-STEP fields (sampling, lookback, prev_tokens, grammar) are NOT touched: the Python step's own on_ctx sets them."""
        from types import SimpleNamespace as NS
        eng = self.eng
        fresh = list(fresh or ())
        dropped = set(dropped or ())
        clash = dropped.intersection(f[0] for f in fresh)
        assert not clash, f"Hooks.replay: rids both admitted and dropped {sorted(clash)[:4]} (the queue must cancel them)"
        for rid, ids, temperature, seed, structured in fresh:
            eng.hist[rid] = list(ids)
            self.samp_ctx[rid] = NS(temperature=float(temperature), seed=int(seed), drafts_valid=False)
            eng.prev_seq.pop(rid, None)
            if structured:
                self.structured.add(rid)
        for rid in dropped:
            eng.hist.pop(rid, None)
            eng.prev_seq.pop(rid, None)
            self.samp_ctx.pop(rid, None)
            self.structured.discard(rid)

    def per_request_state(self, rids=None) -> dict:
        """the per-request state on_ctx / replay maintain, as plain values (PHYS_CXX_ONCTX_LAZY_CHECK's diff and
        the CPU test): rid -> (hist tuple, temperature, seed, drafts_valid, structured, prev_seq entry or None)"""
        eng = self.eng
        keys = set(eng.hist) | set(self.samp_ctx) | set(self.structured) if rids is None else set(rids)
        out = {}
        for rid in keys:
            sc = self.samp_ctx.get(rid)
            out[rid] = (tuple(eng.hist[rid]) if rid in eng.hist else None,
                        None if sc is None else float(sc.temperature), None if sc is None else int(sc.seed),
                        None if sc is None else bool(sc.drafts_valid), rid in self.structured, eng.prev_seq.get(rid))
        return out

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
        if self.hsd_on and self.gate is not None:
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
        k1 = self.hsd.next_ids[r].shape[1]   # (4; FPR_C1_KMAX: 1 + k_max while hooks.hsd is the C1 lane)
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
            src += [si0 * k1 + j for j in range(nr)]
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
        if batches is not None and MIXED_CXX_HEAD and _mxh().head_eager(self, st, batches):   # (mixed steps only)
            return
        hs, auxs = [], []
        for r in range(W):
            s = st[r]
            with torch.cuda.device(s["dev"]):
                h = s["h"]
                if batches is not None:    # mixed: the tail bucket's rows; sample rows remapped by tail_batch
                    sr = s["batch"].sample_rows
                    ix_ = torch.tensor(sr or [0], dtype=torch.int64)
                    ix_ = ix_.pin_memory().to(s["dev"], non_blocking=True) if MIXED_PRESTEP else ix_.to(s["dev"])
                    h = s["h"].index_select(0, ix_)   # (ENG_MIXED_PRESTEP: no host sync on rank r's stream here)
                hs.append(h)
                auxs.append(s["aux"])
        # eager: HSD's stage-interleaved run_all (every collective issued for all ranks before any rank waits; per-rank
        # run() is for graph capture: a host sync inside one rank's run would deadlock the others' peer waits)
        self.hsd.run_all(hs, auxs)

    def hsd_key(self):
        c = getattr(self.hsd, "cur", None) if self.hsd is not None else None
        # + HSD's per-step graph variant (cur.gvar; kernels/hsdtail/hsdt_c1d.py: the c1d owner; none on HSD / HSDT)
        return None if c is None or not self.hsd_on else (c.S_pad, c.D_pad, getattr(c, "NT", None)) + tuple(getattr(c, "gvar", ()))

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
    # ENG_AGRS_WS=1: one persistent workspace per GPU for the production MoE layer's
    # per-call transients, allocated at the end of Engine.__init__ (before the worker sizes the
    # pools) by kernels/agrs/agrs_ws.py. Outside the try blocks on purpose: a set flag that cannot install fails the start
    # instead of silently serving the allocating path.
    if os.environ.get("ENG_AGRS_WS", "0") != "0":
        p = os.path.join(PHYS, "kernels/agrs")
        if p not in sys.path:
            sys.path.insert(0, p)
        import agrs_ws
        agrs_ws.install(sys.modules[__name__])


_install_agrs()
if CXX_PIECES & {"router", "ag", "rs", "final"}:     # ENG_MIXED_CXX_PIECES (stage 1b-i): outermost, after the agrs wrappers
    _p_ = os.path.join(PHYS, "engine/mixed_cxx")
    if _p_ not in sys.path:
        sys.path.insert(0, _p_)
    import mixed_cxx as _mixed_cxx
    _mixed_cxx.install(sys.modules[__name__])


if os.environ.get("FPR_STEP_LOG"):
    _STEP_LOG_S = float(os.environ["FPR_STEP_LOG"])
    _orig_step = Engine.step

    _LMEM_RESET_GB = float(os.environ.get("FPR_LMEM_RESET_GB", "1.5"))   # 0 = off; see the step log below
    # ENG_STEPLOG_TIER=1: the pipelined step log also gives the
    # largest per-rank row count (rows_rmax, rows_rmin) and, for the window's last step, the decode sequences' paged-pool
    # pages in host memory (ids >= FPR_KV_HBM_PAGES): "tier host/all pages, seqs with any host page" per rank
    _SL_TIER = os.environ.get("ENG_STEPLOG_TIER", "0") == "1"
    _SL_NHBM = int(os.environ.get("FPR_KV_HBM_PAGES", "0") or 0)

    def _tier_str(batches):
        out = []
        for sb in batches:
            hp = ap = hs = 0
            for se in sb.seqs:
                if se.kind != "decode":
                    continue
                bt = se.block_table               # the live list: all of the sequence's paged-pool pages
                h = sum(1 for p in bt if p >= _SL_NHBM)
                hp, ap, hs = hp + h, ap + len(bt), hs + (h > 0)
            out.append(f"{hp}/{ap},{hs}s")
        return "tier " + " ".join(out)
    _PROF_ROWS = int(os.environ.get("FPR_PROF_AT_ROWS", "0"))   # > 0: profile FPR_PROF_STEPS decode steps from the first
    _PROF_STEPS = int(os.environ.get("FPR_PROF_STEPS", "20"))
    _PROF_MIXED = os.environ.get("FPR_PROF_MIXED", "0") == "1"   # one with >= this many rows (after 50 decode steps)
    _PROF_DIR = os.environ.get("FPR_PROF_DIR", "/tmp")
    # FPR_PROF_TRIGGER=<path>: start only once this file exists (instead of after 50 decode steps), e.g. touched by the
    # bench script while its streams decode. FPR_PROF_CPU=0: GPU activity only (less host overhead in the window).
    _PROF_TRIG = os.environ.get("FPR_PROF_TRIGGER", "")
    _PROF_AFTER = int(os.environ.get("FPR_PROF_AFTER", "0"))   # > 0: start after this many decode steps instead of 50
    _PROF_CPU = os.environ.get("FPR_PROF_CPU", "1") == "1"
    # FPR_PROF_WINDOWS=N > 1
    # N windows of FPR_PROF_STEPS steps; window w starts once "<FPR_PROF_TRIGGER>.<w>" exists and is written as
    # fpr_trace_rows<R>_w<w>.json.gz
    _PROF_WINDOWS = int(os.environ.get("FPR_PROF_WINDOWS", "1") or 1)

    def _prof_trig(pr):
        return f"{_PROF_TRIG}.{pr['w']}" if _PROF_WINDOWS > 1 else _PROF_TRIG

    def _prof_tick(self, batches):
        pr = self.__dict__.setdefault("_prof", dict(n=0, p=None, done=False, k=0, w=1))
        if pr["done"] or not _PROF_ROWS:
            return
        rows = sum(sb.n_rows for sb in batches)
        decode = all(se.kind == "decode" for sb in batches for se in sb.seqs)
        if _PROF_MIXED:   # FPR_PROF_MIXED=1: profile mixed (prefill) steps instead, from the first with >= FPR_PROF_AT_ROWS
            decode = not decode
        if pr["p"] is None:
            pr["n"] += decode
            ready = os.path.exists(_prof_trig(pr)) if _PROF_TRIG else pr["n"] > (5 if _PROF_MIXED else _PROF_AFTER or 50)
            if decode and ready and rows >= _PROF_ROWS:
                from torch.profiler import ProfilerActivity, profile
                pr["p"] = profile(activities=([ProfilerActivity.CPU] if _PROF_CPU else []) + [ProfilerActivity.CUDA])
                pr["p"].__enter__()
                pr["rows"] = rows
                self.log(f"profiler: started at decode step {pr['n']} ({rows} rows)")
        else:
            pr["k"] += 1
            if pr["k"] >= _PROF_STEPS:
                for d_ in self.devs:
                    torch.cuda.synchronize(d_)
                pr["p"].__exit__(None, None, None)
                sfx = f"_w{pr['w']}" if _PROF_WINDOWS > 1 else ""
                path = os.path.join(_PROF_DIR, f"fpr_trace_rows{pr['rows']}{sfx}.json.gz")
                pr["p"].export_chrome_trace(path)
                self.log(f"profiler: {pr['k']} steps -> {path}")
                if pr["w"] < _PROF_WINDOWS:            # the next window waits for its own trigger file
                    pr.update(p=None, k=0, w=pr["w"] + 1)
                else:
                    pr["done"] = True

    # FPR_CPROFILE_ROWS=R: cProfile of the engine-core thread (everything it does: vLLM loop, scheduler, worker,
    # engine) for FPR_CPROFILE_S seconds from the first decode step with >= R rows after 50 decode steps; pstats in
    # FPR_PROF_DIR and the top functions in the log. GC pauses of every window are in the step log (gc_ms, gc_n).
    _CPROF_ROWS = int(os.environ.get("FPR_CPROFILE_ROWS", "0"))
    _CPROF_S = float(os.environ.get("FPR_CPROFILE_S", "10"))
    _GC = dict(t=0.0, n=0, t0=None, gen2=0)

    def _gc_cb(phase, info):
        if phase == "start":
            _GC["t0"] = time.perf_counter()
        elif _GC["t0"] is not None:
            _GC["t"] += time.perf_counter() - _GC["t0"]
            _GC["n"] += 1
            _GC["gen2"] += info.get("generation") == 2
            _GC["t0"] = None

    import gc as _gcmod
    _gcmod.callbacks.append(_gc_cb)

    def _cprof_tick(self, batches):
        cp = self.__dict__.setdefault("_cprof", dict(n=0, p=None, t=0.0, done=False))
        if cp["done"] or not _CPROF_ROWS:
            return
        if cp["p"] is None:
            decode = all(se.kind == "decode" for sb in batches for se in sb.seqs)
            if os.environ.get("FPR_CPROFILE_MIXED", "0") == "1":   # profile from a mixed (prefill) step instead
                decode = not decode
            cp["n"] += decode
            if decode and cp["n"] > 5 and sum(sb.n_rows for sb in batches) >= _CPROF_ROWS:
                import cProfile
                cp["p"], cp["t"] = cProfile.Profile(), time.perf_counter()
                cp["rows"] = sum(sb.n_rows for sb in batches)
                cp["steps0"] = self.stats.get("steps", 0)
                cp["p"].enable()
                self.log(f"cprofile: started ({cp['rows']} rows)")
        elif time.perf_counter() - cp["t"] >= _CPROF_S:
            cp["p"].disable()
            cp["done"] = True
            import io
            import pstats
            path = os.path.join(_PROF_DIR, f"fpr_cprofile_rows{cp['rows']}.pstats")
            cp["p"].dump_stats(path)
            n = self.stats.get("steps", 0) - cp["steps0"]
            for key in ("tottime", "cumulative"):
                buf = io.StringIO()
                pstats.Stats(cp["p"], stream=buf).sort_stats(key).print_stats(35)
                self.log(f"cprofile ({n} steps in {_CPROF_S:.0f} s, by {key}):\n" + buf.getvalue()[-9000:])

    def _alloc_delta(self) -> str:
        """Caching-allocator churn since the last window, per GPU: cudaMalloc / cudaFree calls, failed mallocs that made
        the allocator free its cache and retry (each one synchronizes the device), and the engine's empty_cache calls."""
        last = self.__dict__.setdefault("_alloc_last", {})
        out = []
        for d_ in self.devs:
            ms = torch.cuda.memory_stats(d_)
            cur = (ms.get("num_device_alloc", 0), ms.get("num_device_free", 0), ms.get("num_alloc_retries", 0))
            prev = last.get(d_, (0, 0, 0))
            last[d_] = cur
            out.append("/".join(str(c - p) for c, p in zip(cur, prev)))
        rel = self.stats.get("cache_releases", 0)
        drel = rel - last.get("rel", 0)
        last["rel"] = rel
        pk = self.stats.pop("mem_peak_win", None)   # max per-step peak allocated (GB) of the mixed steps since the last line
        pks = (" mixed_step_peak_GB " + "/".join(f"{x:.2f}" for x in pk)) if pk else ""
        return f"malloc/free/retry {' '.join(out)} empty_cache={drel}{pks}"

    _FB_SHORT = {"page copies": "copies", "rows > largest bucket": "rows", "prefill / forced / score rows": "prefill",
                 "decode layout": "layout", "context beyond the decode graphs' indexer bound": "ctx", "graphs off": "nographs"}

    def _fb_why(self) -> str:
        """D13 (ASYNC=1): the synchronous fallback steps since the last window, per reason (D13._async_ok's first failing
        check), decode-only ones apart (they leave the pipeline without prefill), and the fallbacks' mean host phases
        (drain = waiting for the in-flight step, resolve, rows_tables, step = the synchronous Engine.step). Empty without D13;
        tolerates a d13.py without the decode-only counts."""
        d = getattr(self, "_d13_obj", None)
        if d is None:
            return ""
        st = d.stats
        cur = (dict(st.get("fallback_why", {})), dict(st.get("fallback_why_dec", {})), dict(st.get("fb_ms", {})))
        last = self.__dict__.get("_fbw_last") or ({}, {}, {})
        self._fbw_last = cur
        dw = {k: v - last[0].get(k, 0) for k, v in cur[0].items() if v - last[0].get(k, 0)}
        dd = {k: v - last[1].get(k, 0) for k, v in cur[1].items() if v - last[1].get(k, 0)}
        dm = {k: dw[k] - dd.get(k, 0) for k in dw if dw[k] - dd.get(k, 0)}
        n = cur[2].get("n", 0) - last[2].get("n", 0)
        f = lambda m: " ".join(f"{_FB_SHORT.get(k, k)}={v}" for k, v in sorted(m.items())) or "0"  # noqa: E731
        ph = (" ms " + " ".join(f"{k}={(cur[2].get(k, 0.0) - last[2].get(k, 0.0)) / n:.1f}"
                                for k in ("drain", "resolve", "rows_tables", "step"))) if n > 0 else ""
        return f" | fb dec {f(dd)} mix {f(dm)}{ph}"

    def _logged_step(self, batches, *a, **kw):
        _prof_tick(self, batches)
        _cprof_tick(self, batches)
        t0 = time.perf_counter()
        sl = self.__dict__.setdefault("_slog", dict(t_last=None, acc={}, cnt={}, t_win=t0,
                                                    g=self.stats.get("graph_steps", 0), e=self.stats.get("eager_steps", 0)))
        gap = (t0 - sl["t_last"]) * 1e3 if sl["t_last"] is not None else None
        dl0 = self.stats.get("decode_last")
        out = _orig_step(self, batches, *a, **kw)
        t1 = time.perf_counter()
        dl = self.stats.get("decode_last")
        kind = "decode" if dl is not dl0 else "mixed"

        def add(k, v):
            sl["acc"][k] = sl["acc"].get(k, 0.0) + float(v)
            sl["cnt"][k] = sl["cnt"].get(k, 0) + 1
        add(f"{kind}_ms", (t1 - t0) * 1e3)
        add(f"{kind}_rows", sum(sb.n_rows for sb in batches))
        try:   # the window's max state slot id (lowest-first allocation: + 1 = peak in use; diagnostics: never fails a step)
            sl["slot_max"] = max(sl.get("slot_max", -1), max((se.state_slot for sb in batches for se in sb.seqs), default=-1))
        except Exception:   # noqa: BLE001
            pass
        if gap is not None and gap < 5000:
            add(f"gap_before_{kind}_ms", gap)
        if kind == "decode":
            for k in ("prepare", "issue", "sync_wait", "results", "gpu", "total"):
                if k in dl:
                    add(f"d_{k}", dl[k])
        sl["t_last"] = t1
        if t1 - sl["t_win"] >= _STEP_LOG_S:
            g, e = self.stats.get("graph_steps", 0), self.stats.get("eager_steps", 0)
            avg = " ".join(f"{k}={sl['acc'][k] / sl['cnt'][k]:.2f}" for k in sorted(sl["acc"]))
            n = {k[:-3]: sl["cnt"][k] for k in sl["cnt"] if k.endswith("_ms") and not k.startswith("gap")}
            mem = []
            for d_ in self.devs:   # per GPU: device used, torch reserved, the rest (context, modules, local memory, ...)
                f_, t_ = torch.cuda.mem_get_info(d_)
                r_ = torch.cuda.memory_reserved(d_)
                other = (t_ - f_ - r_) / 2**30
                if _LMEM_RESET_GB and other > _LMEM_RESET_GB:
                    # the driver keeps the local-memory reservation of the most demanding kernel it has run (per-thread
                    # stack x every resident thread); re-setting the stack limit drops it back to the default
                    import ctypes
                    with torch.cuda.device(d_):
                        torch.cuda.synchronize(d_)
                        rc = ctypes.CDLL("libcuda.so.1").cuCtxSetLimit(0, ctypes.c_size_t(1024))   # CU_LIMIT_STACK_SIZE
                    f2, _ = torch.cuda.mem_get_info(d_)
                    self.log(f"lmem reset gpu {d_}: rc {rc}, other {other:.2f} -> {(t_ - f2 - r_) / 2**30:.2f} GiB")
                    f_ = f2
                mem.append(f"{(t_ - f_) / 2**30:.1f}/{r_ / 2**30:.1f}/{(t_ - f_ - r_) / 2**30:.1f}")
            self.log(f"steplog {t1 - sl['t_win']:.1f}s: steps {n} graph+{g - sl['g']} eager+{e - sl['e']} | {avg} | "
                     f"GiB used/torch/other {' '.join(mem)} | {_alloc_delta(self)} | "
                     f"gc_ms={_GC['t'] * 1e3:.1f} gc_n={_GC['n']} gc_gen2={_GC['gen2']}{_fb_why(self)}"
                     + (f" | {self._onebk_str()}" if MIX_ONE_BUCKET and self.bk_view else "")
                     + f" | state_slot_max={sl.get('slot_max', -1)}")
            _GC.update(t=0.0, n=0, gen2=0)
            sl.update(acc={}, cnt={}, t_win=t1, g=g, e=e, slot_max=-1)
        return out

    Engine.step = _logged_step
    _orig_step_compact = Engine.step_compact

    # ENG_IDX_DUMP=<dir>: after a pipelined decode
    # step with >= ENG_IDX_DUMP_ROWS rows on its largest rank (default 8), per bucket class (small: <= 32 rows per rank, large:
    # more) at most every ENG_IDX_DUMP_EVERY_S / ENG_IDX_DUMP_EVERY_L seconds (60 / 8) and ENG_IDX_DUMP_N times (6): sync,
    # then save per rank the decode bucket's indexer state as the GPUs left it -- the
    # top-512 record slots and lengths of every index layer (s["ti"] / s["tl"], what FlashMLA reads), the L20 candidates
    # and their per-request union (oi.cand, oi.prep.nunion), the owned requests' paged-pool block tables, row -> request map
    # and compressed lengths -- plus FPR_KV_HBM_PAGES, to <dir>/idxdump_<pid>_<k>.pt.
    _IDX_DUMP = os.environ.get("ENG_IDX_DUMP", "")
    _IDX_DUMP_ROWS = int(os.environ.get("ENG_IDX_DUMP_ROWS", "8"))
    _IDX_DUMP_EVERY = {"s": float(os.environ.get("ENG_IDX_DUMP_EVERY_S", "60")), "l": float(os.environ.get("ENG_IDX_DUMP_EVERY_L", "8"))}
    _IDX_DUMP_N = int(os.environ.get("ENG_IDX_DUMP_N", "6"))

    def _idx_dump(self, batches):
        rmax = max(sb.n_rows for sb in batches)
        if rmax < _IDX_DUMP_ROWS:
            return
        try:
            R = self.bucket_of(rmax, pad=False)   # (debug lookup: no ENG_DECODE_PRECREATE decision)
        except ValueError:
            return
        cls = "s" if R <= 32 else "l"
        st_ = self.__dict__.setdefault("_idxdump", {}).setdefault(cls, dict(n=0, t=0.0))
        if st_["n"] >= _IDX_DUMP_N or time.time() - st_["t"] < _IDX_DUMP_EVERY[cls]:
            return
        sts = self.bk.get(R)
        if not sts:
            return
        for d_ in self.devs:
            torch.cuda.synchronize(d_)
        cpu = lambda t: t.detach().to("cpu", copy=True) if isinstance(t, torch.Tensor) else t  # noqa: E731
        ranks = []
        for r, s_ in enumerate(sts):
            d = s_.get("idx") or {}
            oi = d.get("oi")
            prep = getattr(oi, "prep", None) if oi is not None else None
            ranks.append(dict(
                R=s_.get("R"), pos=cpu(s_.get("pos")), B=d.get("B"),
                ti={L: cpu(t) for L, t in (s_.get("ti") or {}).items()}, tl={L: cpu(t) for L, t in (s_.get("tl") or {}).items()},
                t2r=cpu(d.get("t2r")), bt=cpu(d.get("bt")),
                seq_lens={k: cpu(v) for k, v in (d.get("seq_lens") or {}).items()},
                cand=cpu(getattr(oi, "cand", None)) if oi is not None else None,
                nunion=cpu(getattr(prep, "nunion", None)) if prep is not None else None,
                host=[dict(kind=se.kind, uid=se.uid, start=se.start, known=se.known, cached_upto=se.cached_upto,
                           n_rows=se.n_rows, bt=list(se.block_table)) for se in batches[r].seqs]))
        os.makedirs(_IDX_DUMP, exist_ok=True)
        path = os.path.join(_IDX_DUMP, f"idxdump_{os.getpid()}_{cls}{st_['n']}.pt")
        torch.save(dict(t=time.time(), bucket=R, rows=[sb.n_rows for sb in batches], n_hbm=int(os.environ.get("FPR_KV_HBM_PAGES", "0") or 0),
                        ranks=ranks), path)
        st_.update(n=st_["n"] + 1, t=time.time())
        self.log(f"engine: ENG_IDX_DUMP #{st_['n']} bucket {R} rows {[sb.n_rows for sb in batches]} -> {path}")

    def _logged_step_compact(self, batches, cts, ctx=None):
        """Pipelined decode (ASYNC=1, d13): the dispatch of each step; the window's dispatch time, gap between
        dispatches, and the profiler / cProfile windows as for Engine.step."""
        _prof_tick(self, batches)
        _cprof_tick(self, batches)
        t0 = time.perf_counter()
        sl = self.__dict__.setdefault("_slog", dict(t_last=None, acc={}, cnt={}, t_win=t0,
                                                    g=self.stats.get("graph_steps", 0), e=self.stats.get("eager_steps", 0)))
        gap = (t0 - sl["t_last"]) * 1e3 if sl["t_last"] is not None else None
        out = _orig_step_compact(self, batches, cts, ctx)
        if _IDX_DUMP:
            _idx_dump(self, batches)
        t1 = time.perf_counter()
        kv_ = [("compact_ms", (t1 - t0) * 1e3), ("compact_rows", sum(sb.n_rows for sb in batches))]
        if _SL_TIER:
            kv_ += [("rows_rmax", max(sb.n_rows for sb in batches)), ("rows_rmin", min(sb.n_rows for sb in batches))]
        for k, v in kv_:
            sl["acc"][k] = sl["acc"].get(k, 0.0) + float(v)
            sl["cnt"][k] = sl["cnt"].get(k, 0) + 1
        ht = getattr(out, "t", None)    # d13 host split of this dispatch (prep / launch_l / wait_prev / pos_wait / head)
        if isinstance(ht, dict):        # wait_prev ~ 0: the host is the bottleneck; large: the GPUs are
            for k in ("prep", "launch_l", "wait_prev", "pos_wait", "head"):
                if k in ht:
                    sl["acc"]["h_" + k] = sl["acc"].get("h_" + k, 0.0) + float(ht[k])
                    sl["cnt"]["h_" + k] = sl["cnt"].get("h_" + k, 0) + 1
        if gap is not None and gap < 5000:
            sl["acc"]["gap_before_compact_ms"] = sl["acc"].get("gap_before_compact_ms", 0.0) + gap
            sl["cnt"]["gap_before_compact_ms"] = sl["cnt"].get("gap_before_compact_ms", 0) + 1
        try:   # the window's max state slot id (diagnostics: never fails a step)
            sl["slot_max"] = max(sl.get("slot_max", -1), max((se.state_slot for sb in batches for se in sb.seqs), default=-1))
        except Exception:   # noqa: BLE001
            pass
        sl["t_last"] = t1
        if t1 - sl["t_win"] >= _STEP_LOG_S:
            avg = " ".join(f"{k}={sl['acc'][k] / sl['cnt'][k]:.2f}" for k in sorted(sl["acc"]))
            self.log(f"steplog {t1 - sl['t_win']:.1f}s (pipelined): {avg} | {_alloc_delta(self)} | "
                     f"gc_ms={_GC['t'] * 1e3:.1f} gc_n={_GC['n']}{_fb_why(self)}" + (f" | {_tier_str(batches)}" if _SL_TIER else "")
                     + (f" | {self._onebk_str()}" if MIX_ONE_BUCKET and self.bk_view else "")
                     + f" | state_slot_max={sl.get('slot_max', -1)}")
            _GC.update(t=0.0, n=0, gen2=0)
            sl.update(acc={}, cnt={}, t_win=t1, slot_max=-1)
        return out

    Engine.step_compact = _logged_step_compact
