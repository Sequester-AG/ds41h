"""Engine geometry for the ds41h-phys serving engine: model constants, cache kinds and pool sizes per rank.

Everything here is plain data (stdlib only) so the KV manager, the scheduler and their tests run on any CPU.
Model constants come from the checkpoint's text_config (DeepSeek-V4.1-Flash). Cache byte sizes are vLLM 0.32's own
(verified against kv_cache_interface.py page_size_bytes + _apply_alignment_padding, models/deepseek_v41/attention.py /
compressor.py specs) and match production's page stride (116,928 B), so the fork's kernels read the same page layout
they were validated on.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------------------------- model (text_config)
N_LAYERS = 40
HIDDEN = 5120
N_HEADS = 64
SLIDING_WINDOW = 128
# compress_ratios[0:40]: 0 = window only (layers 0, 1), 2 = layers 2..19, 1 = layers 20..39 (40..42 = drafter)
KV_SOURCE_LAYERS = (2, 8, 14, 20)             # compressed KV is written by these layers and read by the following ones
KV_SOURCE_RATIO = {2: 2, 8: 2, 14: 2, 20: 1}  # compressed positions per token: ctx / ratio
INDEX_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)  # layers that run an indexer (top-512 over their source's positions)
CANDIDATE_LAYERS = (24, 28, 32, 36)            # score only layer 20's candidate_topk_blocks x candidate_block_size
CANDIDATES = 2048 * 8                          # 16,384 candidate positions
LAST_KV_SOURCE = 20                            # prefill: layers 21..39 run on each request's last 128 rows only
INDEX_TOPK = 512
INDEX_HEADS, INDEX_DIM = 32, 128
SPEC_K = 3                                     # DSpark draft tokens per decode step (production THROUGHPUT line)
MAX_MODEL_LEN = 655_360                        # production --max-model-len

KV_RECORD_B = 584       # fp8_ds_mla on sm90: 448 fp8 NoPE + 64 bf16 RoPE + 7 ue8m0 scales + 1 pad (data rows, then scale rows)
INDEX_KEY_B = 132       # indexer K cache: 128 fp8 + 4 B fp32 scale per compressed position
ALIGN = 576             # every per-layer page is padded to a multiple of 576 B (vLLM _apply_alignment_padding)


def _pad(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


# Production's packed 64-token page (one block id addresses the same token block in all 8 per-layer tensors):
PAGE_TOKENS = 64
PAGE_LAYOUT = {  # per-layer page sizes inside the packed page (bytes)
    **{f"comp{s}": _pad(PAGE_TOKENS // KV_SOURCE_RATIO[s] * KV_RECORD_B) for s in KV_SOURCE_LAYERS},   # 19,008 x3 + 37,440
    **{f"idx{s}": _pad(PAGE_TOKENS // KV_SOURCE_RATIO[s] * INDEX_KEY_B) for s in KV_SOURCE_LAYERS},    # 4,608 x3 + 8,640
}
PAGE_BYTES = sum(PAGE_LAYOUT.values())        # 116,928 B = 1,827 B per token
assert PAGE_BYTES == 116_928, PAGE_BYTES
PAGE_OFFSETS = {}                             # byte offset of each per-layer page inside the packed page (our order)
_o = 0
for _k, _v in PAGE_LAYOUT.items():
    PAGE_OFFSETS[_k] = _o
    _o += _v
RECORDS_PER_PAGE = {s: PAGE_TOKENS // KV_SOURCE_RATIO[s] for s in KV_SOURCE_LAYERS}   # 32 / 32 / 32 / 64


def kv_source_of(layer: int):
    """The compressed-KV source a layer attends over (None for the window-only layers 0, 1)."""
    if layer < KV_SOURCE_LAYERS[0]:
        return None
    return max(s for s in KV_SOURCE_LAYERS if s <= layer)

# Sliding window: SlidingWindowMLASpec(block_size=32, sliding_window=128) on all 40 target layers AND the 3 DSpark draft
# layers (the drafter's SWA caches share the target's block tables). Here one window page id = 32 tokens in each of the
# 43 per-layer tensors.
WINDOW_PAGE_TOKENS = 32
N_WINDOW_LAYERS = N_LAYERS + 3
WINDOW_LAYER_PAGE_B = _pad(WINDOW_PAGE_TOKENS * KV_RECORD_B)      # 19,008 B
WINDOW_PAGE_BYTES = WINDOW_LAYER_PAGE_B * N_WINDOW_LAYERS         # 817,344 B
# Compressor running state: CircularBufferSpec(block_size = max(8, next_pow2(k + 2)) = 8, head 1024, fp32) on the
# ratio-2 sources 2 / 8 / 14; one block per request, not prefix-cacheable.
STATE_SLOT_BYTES = 3 * 8 * 1024 * 4                               # 98,304 B


def paged_bytes_per_token() -> float:
    """Bytes per context token of the prefix-cacheable state (one copy, on the owner rank only)."""
    return PAGE_BYTES / PAGE_TOKENS


@dataclass(frozen=True)
class PoolSpec:
    """One kind of KV storage on a rank.

    kind:
      "paged"  -- grows with the context, one page per `tokens_per_page` tokens of a sequence, prefix-cacheable by block
                  hash (compressed KV of the 4 source layers + their index keys: everything a token leaves behind
                  beyond the window).
      "window" -- the 128-token sliding window of the 43 window layers (one page id space); pages that fall out of the
                  window are freed; not prefix-cached (rebuilt by the 128-token bounded replay after a hit).
      "state"  -- a fixed-size per-sequence slot (compressor ring of layers 2 / 8 / 14); not prefix-cached.
    """
    name: str
    kind: str
    tokens_per_page: int
    bytes_per_page: int
    num_pages: int


@dataclass
class Geometry:
    """Pool sizes and limits per rank. Defaults: H200 NVL, ~34 GB of KV per GPU = production's 35.04 GiB (37.6 GB at
    GPU_UTIL 0.91; ~321.8K pages of 116,928 B, replicated on all 4 GPUs) minus the fork's replicated attention weights
    (+3.5 GB per GPU)."""
    n_ranks: int = 4
    spec_k: int = SPEC_K
    page_tokens: int = PAGE_TOKENS
    window: int = SLIDING_WINDOW
    window_page_tokens: int = WINDOW_PAGE_TOKENS
    kv_bytes_per_rank: int = 34_000_000_000
    window_pool_bytes: int = 1_400_000_000     # ~1,700 window pages: 80 seqs x 6 + 2 x 4K-token chunks + margin
    state_pool_slots: int = 512
    max_model_len: int = MAX_MODEL_LEN
    max_seqs: int = 256                        # running sequences, all ranks (production: 128)
    max_seqs_per_rank: int = 80                # decode rows per rank <= 320 (graph shapes of the step program)
    max_prefill_tokens_per_rank: int = 4096    # prefill rows per rank per step (step-program buffer)
    max_rows_per_rank: int = 0                 # all rows of a rank per step (the engine's buffers); 0 = derived:
                                               # (k + 1) x max_seqs_per_rank + max_prefill_tokens_per_rank
    max_tail_rows_per_rank: int = 1536         # rows per rank that run layers 21..39 + the head / drafter in one step
                                               # (decode rows + the last 128 rows of each prefill chunk; all rows of a
                                               # score chunk): HSD's largest tail bucket (engine: max(buckets, 1536))
    replay_tokens: int = SLIDING_WINDOW        # bounded replay after a prefix hit (window + compressor state)
    draft_window_tokens: int = SPEC_K          # DSpark writes block KV at Q .. Q + k - 1 past the verified rows: window
                                               # pages must cover start + 2k + 1 in decode, chunk end + k after a prefill
    reserve_watermark: float = 0.01            # fraction of a rank's pages kept free for decode growth at admission

    @property
    def rows_per_decode_seq(self) -> int:
        return self.spec_k + 1

    @property
    def rows_cap(self) -> int:
        return self.max_rows_per_rank or (self.rows_per_decode_seq * self.max_seqs_per_rank + self.max_prefill_tokens_per_rank)

    def pools(self) -> list[PoolSpec]:
        n_pages = (self.kv_bytes_per_rank - self.window_pool_bytes - self.state_pool_slots * STATE_SLOT_BYTES) // PAGE_BYTES
        return [
            PoolSpec("paged", "paged", self.page_tokens, PAGE_BYTES, int(n_pages)),
            PoolSpec("window", "window", self.window_page_tokens, WINDOW_PAGE_BYTES,
                     int(self.window_pool_bytes // WINDOW_PAGE_BYTES)),
            PoolSpec("state", "state", 0, STATE_SLOT_BYTES, self.state_pool_slots),
        ]


@dataclass
class SmallGeometry(Geometry):
    """Tiny pools for unit tests (same structure, explicit page counts)."""
    paged_pages: int = 64
    window_pages: int = 64
    state_slots: int = 16

    def pools(self) -> list[PoolSpec]:
        return [
            PoolSpec("paged", "paged", self.page_tokens, PAGE_BYTES, self.paged_pages),
            PoolSpec("window", "window", self.window_page_tokens, WINDOW_PAGE_BYTES, self.window_pages),
            PoolSpec("state", "state", 0, STATE_SLOT_BYTES, self.state_slots),
        ]
