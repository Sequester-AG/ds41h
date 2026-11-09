"""Interfaces between the host side (scheduler + KV manager) and the step program's pieces.

Everything is defined per RANK: a rank only ever sees its own sequences (their rows, KV pages, window, compressor state,
sampling state). Cross-rank traffic inside a step is only (a) the per-layer AG of (f, routing) rows and RS of the MoE
partials (kernels/sp_layer/comm.cu), (b) whatever the head / sampler needs (vocab-parallel logits), and (c) page copies
planned by the scheduler, executed BEFORE the step's kernels.

The host builds one StepBatch per rank per step (build_step_batch below, stdlib only and unit-tested); the worker
copies it into the persistent device buffers of the step program (int32 tensors, padded to the captured graph shape).

Row order inside a rank (fixed, the step program and every piece rely on it):
  [ decode rows: k + 1 per decoding sequence, sequence-contiguous, position-ascending ]
  [ prefill rows: per chunk, position-ascending; chunks in scheduler order ]
  [ padding up to the graph's row count (token 0, position 0, seq_slot -1, slots -1) ]
Layers 0..20 run all rows; layers 21..39 run the decode rows + the LAST min(n, 128) rows of every prefill chunk
(production's decoder-side bounded replay); the head / sampler runs the decode rows + the last row of each FINAL chunk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from geometry import SLIDING_WINDOW, Geometry
from kv_manager import CopyOp, OwnedKVManager

PAD = -1


@dataclass
class CPRole:
    """Context-parallel prefill: this entry's part of one CP chunk of a prompt."""
    group: int                 # the prompt's seq_uid (the same on every rank's entry of the chunk)
    rid: object                # the prompt's rid (the engine keys Engram's host token history by it)
    owner: int                 # the prompt's owner rank
    subs: list                 # [(rank, start, n)] in position order, the owner last (n >= 128)
    me: int                    # this entry's index in subs


@dataclass
class SeqEntry:
    """One sequence scheduled on this rank this step (slot = its index in StepBatch.seqs)."""
    rid: object
    kind: str                 # "decode" | "prefill"
    row0: int                 # first row of this sequence in the rank's row order
    n_rows: int               # decode: k + 1; prefill: chunk length
    start: int                # first position computed
    num_computed: int         # positions with final KV before this step (= start)
    cached_upto: int          # positions < cached_upto: paged KV already exists (prefix hit) -> no paged writes
    replay_start: int         # no window KV exists / is read below this position
    last_chunk: bool          # prefill: the chunk's last row produces the first output token
    block_table: list         # paged pool pages of logical blocks 0..ceil((start + n_rows) / 64) - 1
    window_table: dict        # logical 32-token window block -> window pool page
    state_slot: int           # compressor ring slot
    tail_row0: int = 0        # first row that also runs layers 21..39 (prefill: last 128 of the chunk; score: all)
    n_blocks: int = 0         # pages of block_table this step covers (block_table is the live list, a reference)
    score: bool = False       # teacher-forced scoring: every row runs all layers and is sampled for logprobs
    forced: bool = False      # forced decode: drafts are the forced tokens, all accepted, raw logits returned
    uid: int = -1             # global sequence slot, stable across rank moves (scheduler.Request.uid)
    known: int = -1           # async decode (start == -1): the host's settled num_computed (D13 init records)
    wver: int = -1            # SeqKV.wver when the batch was built (-1: unknown -> diff the window table)
    cp: Optional[CPRole] = None   # CP prefill (owner entry: the prompt's chunk cut to its sub-chunk; helper entry: a
                                  # SHADOW sequence of the prompt on this rank: replica pages, scratch window / state)
    images: tuple = ()            # prefill: the prompt's images ((offset, length, key), ...) overlapping this entry's
                                  # positions [start, start + n_rows); their rows take StepContext.mm embeddings
    inc: int = 0              # SeqKV.inc: the delta caches key on (uid, inc), since uids and slots are recycled


@dataclass
class StepBatch:
    rank: int
    step: int
    n_rows: int                                   # real rows (before padding)
    positions: list = field(default_factory=list)     # [n_rows]
    seq_slot: list = field(default_factory=list)      # [n_rows] -> index into seqs
    paged_slot: list = field(default_factory=list)    # [n_rows] token-level page * 64 + pos % 64, or PAD (prefix-cached
                                                      # / replay rows); a cache with compress ratio c stores the record
                                                      # of position pos at page * (64 // c) + (pos % 64) // c
    window_slot: list = field(default_factory=list)   # [n_rows] window page * 32 + pos % 32
    tail_rows: list = field(default_factory=list)     # rows that run layers 21..39 (indices into the row order)
    sample_rows: list = field(default_factory=list)   # rows whose logits are sampled / verified
    seqs: list = field(default_factory=list)          # SeqEntry
    copies: list = field(default_factory=list)        # CopyOp touching this rank (as source or destination)
    poison: list = field(default_factory=list)        # debug: (pool, page) to fill with 0xFF before this step's copies


def build_step_batch(plan, kv: OwnedKVManager, geom: Geometry, rank: int) -> StepBatch:
    """The StepBatch of `rank` for a scheduler StepPlan (after RankScheduler.schedule; KV already allocated)."""
    sb = StepBatch(rank, plan.step, 0)
    work = [w for w in plan.work if w.rank == rank]
    work.sort(key=lambda w: (w.kind != "decode",))           # decode rows first, stable otherwise
    bt, wt = geom.page_tokens, geom.window_page_tokens
    row = 0
    for w in work:
        helper = w.kind == "cp"                             # CP helper: the shadow sequence's tables, the prompt's rid
        sk = kv.seqs[w.rid]
        cp = getattr(w, "cp", None)
        rid = cp.rid if helper else w.rid
        # block / window tables by REFERENCE (no per-step O(context) copy): valid until the next schedule(), which
        # is when the synchronous loop has consumed the batch; n_blocks says how much of the page list this step uses
        se = SeqEntry(rid, "prefill" if helper else w.kind, row, w.n, w.start, w.start, 0 if helper else w.cached_upto,
                      w.replay_start, False if helper else w.last_chunk, sk.pages, sk.window, sk.state_slot)
        se.cp = cp
        dev = w.start < 0                                   # async decode: positions resolved on the device (D13)
        se.n_blocks = len(sk.pages) if dev else -(-(w.start + w.n) // bt)
        if cp is not None:                                  # CP: every entry's table covers the whole chunk
            se.n_blocks = -(-(cp.subs[-1][1] + cp.subs[-1][2]) // bt)
        se.known = int(getattr(w, "known", -1))
        se.images = tuple(getattr(w, "images", ()))
        se.score = bool(getattr(w, "score", False))
        se.forced = bool(getattr(w, "forced", False))
        se.uid = int(getattr(w, "uid", -1))
        se.wver, se.inc = sk.wver, sk.inc
        se.tail_row0 = row if (w.kind == "decode" or se.score) else row + max(0, w.n - SLIDING_WINDOW)
        slot = len(sb.seqs)
        sb.seqs.append(se)
        pages, win, cu = sk.pages, sk.window, w.cached_upto
        sb.seq_slot.extend([slot] * w.n)
        if dev:
            sb.positions.extend([PAD] * w.n)
            sb.paged_slot.extend([PAD] * w.n)
            sb.window_slot.extend([PAD] * w.n)
        else:
            poss = range(w.start, w.start + w.n)
            sb.positions.extend(poss)
            sb.paged_slot.extend([PAD if p < cu else pages[p // bt] * bt + p % bt for p in poss])
            sb.window_slot.extend([win[p // wt] * wt + p % wt for p in poss])
        if helper:                                          # helpers: layers 0..20 only, nothing sampled
            se.tail_row0 = row + w.n
        sb.tail_rows.extend(range(se.tail_row0, row + w.n))
        if helper:
            pass
        elif w.kind == "decode" or se.score:
            sb.sample_rows.extend(range(row, row + w.n))
        elif w.last_chunk:
            sb.sample_rows.append(row + w.n - 1)
        row += w.n
    sb.n_rows = row
    sb.copies = [c for c in plan.copies if rank in (c.src_rank, c.dst_rank)]
    take = getattr(kv, "take_poison", None)
    if take is not None:
        sb.poison = take(rank)
    return sb


# ------------------------------------------------------------------------------------------------ piece contracts
class IndexerCompressor(Protocol):
    """Indexer + compressor (kernels/indexer2f/). Per index layer L in (2, 8, 14, 20, 24, 28, 32, 36), on this rank's
    rows only.

    Source layers (2 / 8 / 14 / 20): compress the rows' KV into the paged pool (records at paged_slot; ratio-2 layers
    through the compressor ring at state_slot; nothing written for PAD rows) and write their index keys. Every index
    layer: score the row's own sequence's compressed positions (layer 20 candidate blocks for 24..36), return the
    top-512 as GLOBAL slots of the paged pool (page * 64 + offset in the layer's compressed numbering), -1 padded, with
    the per-row lengths -- the exact format FlashMLA's sparse decode reads (attn_half: ti / tl).
    Semantics = production's (stock vLLM 0.32 + production's ds41h package), including ties and the candidate masking;
    layers 24..36 run only on StepBatch.tail_rows in prefill (decoder-side bounded replay)."""

    def run(self, layer: int, x_rows, batch: StepBatch, stream) -> tuple: ...


class MoEPrefill(Protocol):
    """Prefill-grade MoE. Routed + shared MoE of one layer over ALL ranks' rows (TP4 slice of every expert):
    f [T, 5120] bf16 after the AG (decode rows of all ranks, then prefill rows), routing [T, 6] ids + weights;
    returns this rank's partial [T, 5120] for the RS. Must accept T up to 4 x (decode rows + max_prefill_tokens_per_rank)
    (~17K) and pick the decode kernel (moe_raw.cu) or the prefill-grade path by T, with production's rounding points
    (bf16 per-expert outputs, fp32 accumulation, fixed combine order)."""

    def run(self, layer: int, f_all, routing_all, n_decode_rows: int, stream): ...


class HeadSamplerDrafter(Protocol):
    """Head, sampler and drafter (kernels/head/). After layer 39, on this rank's sample_rows:
      1. logits (vocab-parallel head or replicated, its choice) -> grammar bitmask (if any) -> sampling with each
         sequence's params (temperature, top-p / top-k / min-p, penalties, seed + position draws as vLLM's gumbel
         sampler) -> DSpark verification of the k drafts ("block" rejection, probabilistic drafts) -> accepted tokens;
      2. the DSpark drafter (3 layers over the target's layers 37 / 38 / 39 hidden states; its window KV lives in the
         window pool's layers 40..42 under the same window_table) proposes the next k drafts;
      3. writes the next step's input ids (last token + k drafts per decoding sequence) into the device buffer and the
         host-visible output: per sequence (num_accepted, tokens[k + 1], optional logprobs) in rank-local seq order.
    The bitmask arrives after the forward (vLLM computes it during the step): the sampler must be able to wait for it
    (only when the batch has structured-output requests)."""

    def run(self, h_rows, aux_rows, batch: StepBatch, sampling_state, bitmask: Optional[object], stream): ...


SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty",
                   "repetition_penalty", "logprobs", "prompt_logprobs", "max_tokens", "min_tokens", "ignore_eos")


def image_rows(sb: StepBatch) -> list:
    """(row, image key, span row) for every row of `sb` whose position lies in an image span of its entry, by row: the
    rows that take StepContext.mm embeddings (span row = position - the image's offset)."""
    out = []
    for se in sb.seqs:
        for off, n, key in se.images:
            for pos in range(max(se.start, off), min(se.start + se.n_rows, off + n)):
                out.append((se.row0 + pos - se.start, key, pos - off))
    out.sort()
    return out


def sampling_dict(sp) -> dict:
    """Plain-value view of a vLLM SamplingParams (or a dict): what the head / sampler needs, engine-agnostic."""
    if sp is None:
        return {}
    if isinstance(sp, dict):
        return {k: sp.get(k) for k in SAMPLING_FIELDS if k in sp}
    return {k: getattr(sp, k) for k in SAMPLING_FIELDS if hasattr(sp, k)}


@dataclass
class SeqInit:
    """A sequence whose engine-side state starts this step (first scheduling, or re-admission after a preemption)."""
    rid: object
    prompt_ids: list
    output_ids: list                                  # non-empty after a preemption (recomputed as prefill)
    sampling: dict                                    # sampling_dict(SamplingParams)
    structured: bool = False                          # grammar-constrained: bitmask rows arrive in StepContext.grammar


@dataclass
class StepContext:
    """Per-step request lifecycle for the engine (optional third argument of Engine.step, contract D11).
    The sequence's uid / state slot come from the same step's tables (seq_uid, seq_state_slot via seq_rows rid)."""
    new: list = field(default_factory=list)           # SeqInit
    dropped: set = field(default_factory=set)         # rids finished or preempted since the previous step: drop state
    grammar: dict = field(default_factory=dict)       # rid -> (rows int32 [num_spec + 1, ceil(V / 32)], acceptable)
    prev_tokens: dict = field(default_factory=dict)   # rid -> up to 3 token ids before its first row (Engram context)
    order: list = field(default_factory=list)         # vLLM's request order of this step (informational)
    mm: Optional[list] = None                         # image rows: per rank (idx int32 [M], emb bf16 [M, hidden]) on
                                                      # that rank's device, or None; idx = rows of the rank's StepBatch
                                                      # whose input is emb (instead of the token embedding). None: no
                                                      # image rows this step. Valid in stream order until the step's
                                                      # kernels read them.


@dataclass
class StepResult:
    """What a rank returns to the host after a step (merged over ranks by the executor into vLLM's ModelRunnerOutput)."""
    rank: int
    step: int
    accepted: dict = field(default_factory=dict)      # rid -> number of tokens appended (1 .. k + 1)
    tokens: dict = field(default_factory=dict)        # rid -> list of the appended token ids
    logprobs: dict = field(default_factory=dict)      # rid -> [(position, [(token id, logprob)] x top-k)] for the rid's
                                                      # sampled rows (score mode: every row; the logprobs of the
                                                      # distribution AFTER that position, i.e. of the token at pos + 1)
    num_nans: dict = field(default_factory=dict)
    logits: dict = field(default_factory=dict)        # forced rows: rid -> (positions, tensor [rows, V or V / tp] fp32,
                                                      # vocab_start); row at position p predicts the token at p + 1
    drafts: dict = field(default_factory=dict)        # structured-output requests only: rid -> the next k drafts (vLLM
                                                      # validates them against the grammar before the next step)


def copy_runs(copies: list[CopyOp]) -> list[tuple]:
    """Group page copies into contiguous runs (pool, src_rank, src_page0, dst_rank, dst_page0, n) for cudaMemcpyPeerAsync
    on the copy engines: fewer, larger transfers."""
    out = []
    for c in sorted(copies, key=lambda c: (c.pool, c.src_rank, c.dst_rank, c.src_page)):
        if out:
            p, sr, s0, dr, d0, n = out[-1]
            if (p, sr, dr) == (c.pool, c.src_rank, c.dst_rank) and c.src_page == s0 + n and c.dst_page == d0 + n:
                out[-1] = (p, sr, s0, dr, d0, n + 1)
                continue
        out.append((c.pool, c.src_rank, c.src_page, c.dst_rank, c.dst_page, 1))
    return out
