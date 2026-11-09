"""Step-time model of the sequence-parallel step program, per rank, for the scheduler and the CPU simulation.

The slowest rank sets every layer: after each layer's attention half the ranks meet at the all-gather (AG) and the
reduce-scatter (RS) of the tensor-parallel MoE. So a step is

    sum over layer groups of  max_r (per-rank attention side of group g)  +  shared side (MoE, collectives)
    + head / sampler / drafter (max over ranks) + prefill-step overhead

Per-rank side: the attention half (replicated fp8 projections + FlashMLA over all 64 heads on the rank's own rows), both
mHC boundaries and the router -- linear in the rank's rows -- plus the indexer, linear in the context of the rank's own
sequences. Shared side: the routed + shared MoE over ALL rows (TP4, balanced by construction) and the two collectives
(sized by the largest rank's rows: buffers are padded to R = max_r rows).

Prefill rows follow production's decoder-side bounded replay (verified in production's endpoint log: layer 20 writes
its KV for every token, and the rest of it and layers 21-39 run on each request's last 128 tokens only): layers 0..20
see every prefill token, layers 21..39 only the chunk's last 128 rows.

Calibration (fork, measured on real traffic at 600 W):
  attention half 155.5-166 us at R = 116 rows (FlashMLA 47 us -> ~0.4 us/row; dense ~93 us, nearly flat in R),
  mHC x2 ~46 us, router ~19 us, AG + RS ~45 us, MoE GEMMs 516-522 us at T 456-512 (408 at 256), MoE glue ~45 us,
  40-layer step 34.02 ms at T = 456 (114 sequences); indexer ~1.7 ms per step with sequence ownership (estimate from
  production's replicated ~170 us/layer at 60K contexts / 4); head + sampler + drafter ~2-3 ms (estimate).
Prefill rows (ESTIMATES in the defaults; the measured terms are the refits below): dense 254 MFLOP per row per layer at
~600 TFLOPS -> 0.42 us + sparse attention ~0.25 us + boundaries ~0.1 us -> ~0.95 us per row per layer, i.e. the same
marginal as a decode row; indexer 8,192 FLOP per scored position per row (fp8, ~1.2 PFLOPS effective); prefill-grade
MoE 0.22 us per row per layer (production's marginal; target 0.16-0.20).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from geometry import CANDIDATES, KV_SOURCE_LAYERS, KV_SOURCE_RATIO, LAST_KV_SOURCE, N_LAYERS, SLIDING_WINDOW

N_A = LAST_KV_SOURCE + 1          # layers 0..20: every prefill row
N_B = N_LAYERS - N_A              # layers 21..39: decode rows + the last 128 rows of each prefill chunk
FULL_POS_PER_CTX = sum(1.0 / KV_SOURCE_RATIO[s] for s in KV_SOURCE_LAYERS)   # 2.5 scored positions per context token
N_CAND_LAYERS = 4
# positions a tail / decode row scores per consumer layer (24..36): min(ctx, ~1.6 x the 16,384 candidates) (measured;
# non-final prompt rows score 0 there)
CAND_SCORED = int(1.6 * CANDIDATES)


@dataclass
class CostParams:
    # per rank, per layer (us)
    a0: float = 130.0             # fixed attention-side cost (dense weight stream, small kernels, mHC / router fixed)
    a_row: float = 0.95           # per row: FlashMLA 0.40 + dense 0.25 + mHC 0.22 + router 0.08
    a_pf_row: float = 0.95        # per PREFILL row of layers 0..20 (rank side; measured eager engine: ~1.6, see below)
    # indexer, per decode sequence (4 rows) per step (us per scored position)
    idx_dec_us_per_pos: float = 0.272e-3
    idx_dec_fixed_us: float = 2.0
    # indexer, per prefill row (us per scored position): 8,192 FLOP / 1.2 PFLOPS
    idx_pf_us_per_pos: float = 6.8e-6
    # shared, per layer (us)
    c0: float = 20.0              # AG + RS fixed
    c_row: float = 0.22           # per row of the LARGEST rank (padded buffers)
    moe0: float = 340.0           # routed GEMMs + glue, fixed (weight stream, power-bound)
    moe_dec_row: float = 0.485    # per decode row (GEMMs 0.43 + glue 0.055)
    moe_pf_row: float = 0.22      # per prefill row (prefill-grade MoE)
    # MoE of a MIXED step (any prefill rows) as a function of the layer's row count T (the measured prefill-MoE curve:
    # 1.15-1.45 us per token per layer at 512, 0.55 at 2K, 0.45 at 4K, 0.32 at 8K, 0.27 at 16K, i.e. the
    # weight stream dominates small steps): MoE(T) = moe_mixed0 + moe_mixed_row * T per layer. 0 = the linear terms above.
    moe_mixed0: float = 0.0
    moe_mixed_row: float = 0.0
    # per step (us)
    head0: float = 1800.0         # head + sampler + DSpark drafter
    head_row: float = 6.0         # per decode row of the largest rank
    pf_overhead: float = 500.0    # extra launch / shape cost of a step that carries prefill rows
    host_us: float = 0.0          # host time per step NOT overlapped with the GPU (the served sync loop: ~3,600; D13: ~0)
    ideal_balance: float = 0.0    # 1: the what-if of every rank at the MEAN load (same total rows): per-rank sides and the
                                  # collectives' row term at the mean rank instead of the busiest
    coll_real_rows: float = 0.0   # 1: variable per-rank rows in the AG / RS (each rank sends only its real rows): the
                                  # collectives' row term follows the MEAN rank's rows, not the busiest rank's (padded)
    eager_ctx_above: int = 0      # > 0: a decode-only step with any row deeper than this runs eager (idx2f's decode
    eager_extra_us: float = 0.0   # graphs: 262,144 positions until widened to 655,360) and costs this much more
                                  # (measured: 47.5 -> ~75 ms while one > 262K call was in flight)
    dec_only_extra_us: float = 0.0  # what-if: extra time of a decode-only step (one engine build ran decode-only steps
                                  # at 75 ms for 133 sequences vs ~46 planned, mixed steps on plan)
    pf_tok_us: float = 0.0        # > 0: the placement-free ceiling model: every prefill token costs this much step time
                                  # machine-wide (all its layer-0..20 costs, collectives included) wherever it sits
    copy_us_per_mb: float = 1.0   # rank-to-rank page copies (copy engines, concurrent): ~12 % of their NVLink time
                                  # charged to the step for contention with the AG / RS (assumption)


# The real engine's mixed steps as measured (eager mixed steps with 112 decode rows per rank and 0-212 prefill rows per
# rank): step = decode-only + 41 ms + ~26 us per prefill row (all ranks; the shared side dominates: adding rows on an
# idle rank costs the same). Without mixed-step graphs the scheduler must know this fixed cost, or it pays it every
# step for a tiny chunk.
# Refit with large chunks (prefill-only steps, real prompts, eager): 4,096 rows on one rank
# 248 ms, 3 ranks x 4,096 rows 353 ms (above 2,560 rows the engine runs production's MoE: ~12.8 us per row shared) and
# small chunks (85 ms at 208 prefill rows next to 112 decode rows per rank): fixed ~34 ms per mixed step,
# rank side ~29 us per prefill row of the owner (a_pf_row 1.35 x 21 layers; incl. ~14 us of host-side indexer prep:
# 57.6 ms per 4K-row step); fits 248 / 356 / 86 / 99 ms against 248 / 353 / 85 / 98-105 measured.
# Refit on the final defaults (lean indexer, idx2f device tables, MoE v4; batched-decode fix): 4,096 rows on one
# rank 239 ms, 3 ranks x 4,096 283 ms (the device tables removed ~70 ms of serialized host prep), small chunks ~85 ms.
# Refit with the measured MoE(T) curve and a 1,280-row mixed step (70.4 ms): mixed overhead 2 ms, owner-rank prefill
# 1.6 us per row per layer, MoE(T) = 0.65 ms + 0.234 us * T per layer
# -> 1 rank x 4,096 rows 221 ms, 3 x 4,096 263 ms, the 1,280-row step 70.5 ms; per prefill row 27 us at 1K rows a step,
# 18 at 4K, 15.6 at 16K (bigger mixed steps amortize the MoE weight stream)
# ENGINE_MEASURED_1735: + a mixed step of 82 decoders + 944 shallow prefill rows (+31.8 ms over decode-only) and the
# indexer made explicit at the measured ~21.5 ps per (prefill row, scored position): overhead 7 ms, owner-rank 1.46 us per
# prefill row per layer + indexer (0.1 us at 2K positions, 3.4 at 64K, 8.1 at 150K)
# Refit with native-CP measurements (real engine, eager, GPU-bound): 1 x 4,096 prefill rows 163 ms and 4 x 4,096
# 243 ms at context 0 (+~0.2 ms per 1K of context), plus the 1,280-row step 70.4 ms: owner-rank prefill 0.92 us per row
# per layer (was 1.46), MoE(T) slope 0.300 us per token per layer (was 0.234), mixed overhead 3 ms (was 7)
# -> 163 / 243 / 70.1 ms, and 186 ms at 100K (measured ~183). Per prefill token: 40 us on one rank, 14.8 us spread over
# 4 x 4,096.
ENGINE_MEASURED_1735 = dict(pf_overhead=7_000.0, moe_pf_row=0.0, a_pf_row=1.46, idx_pf_us_per_pos=21.5e-6,
                            moe_mixed0=650.0, moe_mixed_row=0.234)
# Decode terms fitted to a served replay's decode-only steps (real step gaps incl. the
# sync host time): 39.1 / 40.7 / 42.1 ms at ~70 / 90 / 110 sequences (model 38.4 / 40.4 / 42.4; the defaults planned
# 31.2 / 34.1 / 36.9). Mixed steps already matched (x0.83-1.04).
# Loaded long-context steps (160 decoders, chunks at 150-190K depth, GPU-bound): 4 x 2,048 prefill
# 188.6 ms, 4 x 4,096 306.2 ms, decode-only 49-51 -> mixed overhead 13 ms (was 3), owner-rank prefill 0.94 us per row per
# layer: 188.8 / 306.1 / 49.2 ms (the prefill-only shapes: 170 / 249 vs 163 / 243).
ENGINE_MEASURED_2026_11_07 = dict(pf_overhead=13_000.0, moe_pf_row=0.0, a_pf_row=0.94, idx_pf_us_per_pos=21.5e-6,
                                  moe_mixed0=650.0, moe_mixed_row=0.300,
                                  a0=251.9, a_row=0.6175, moe0=658.75, moe_dec_row=0.3152, head_row=3.9, c_row=0.143,
                                  eager_ctx_above=655_360, eager_extra_us=28_000.0)
# Decode check on a served 192-sequence run (ENG_LONG_GRAPHS=1): decode-only 48.8 / 48.6 / 51.6 / 54.1 ms
# at ~90 / 110 / 130 / 150 sequences (model 47.7 / 49.6 / 51.6 / 53.6; the plan had been 44-47); mixed steps on plan
# overall (336.6 planned vs 324.7 real p50). Decode graphs valid to 655,360 positions (ENG_LONG_GRAPHS). Two later
# engine builds (the loaded long-context step 319.5 -> 297.1 / 289.6 ms wall): the prefill terms x 0.917 / 0.889,
# decode unchanged.
BX_PREFILL_SCALE, B8_PREFILL_SCALE = 0.917, 0.889


def params_from_env(default: dict | None = None) -> "CostParams":
    """CostParams with overrides from $PHYS_COST ("pf_overhead=41000,moe_pf_row=1.24", or "measured" / "graphs"),
    else `default` (a dict of fields) or the dataclass defaults."""
    import os

    spec = os.environ.get("PHYS_COST", "")
    kw = dict(default or {})
    if spec == "graphs":
        kw = {}
    elif spec == "measured":
        kw = dict(ENGINE_MEASURED_2026_11_07)
    elif spec:
        for item in spec.split(","):
            k, v = item.split("=", 1)
            assert k.strip() in CostParams.__dataclass_fields__, f"PHYS_COST: unknown field {k}"
            kw[k.strip()] = float(v)
    return CostParams(**kw)


@dataclass
class RankLoad:
    """What one rank runs in a step."""
    dec_rows: int = 0                         # decode rows (k + 1 per decoding sequence)
    dec_idx_pos: float = 0.0                  # sum over decoding seqs of FULL_POS_PER_CTX * ctx  (layers 2/8/14/20)
    dec_cand_pos: float = 0.0                 # sum over decoding seqs of 4 * min(ctx, ~26,214) (layers 24..36)
    n_dec_seqs: int = 0
    max_dec_ctx: int = 0                      # deepest decoding sequence (decode graphs' indexer bound)
    pf_rows: int = 0                          # prefill rows (layers 0..20)
    pf_tail_rows: int = 0                     # prefill rows that also run layers 21..39 (last 128 of each chunk)
    pf_idx_pos: float = 0.0                   # sum over prefill rows of FULL_POS_PER_CTX * ctx_at_row
    pf_cand_pos: float = 0.0                  # sum over tail rows of 4 * min(ctx, ~26,214)

    def add_decode(self, rows: int, ctx: int) -> None:
        self.dec_rows += rows
        self.n_dec_seqs += 1
        self.max_dec_ctx = max(self.max_dec_ctx, ctx)
        self.dec_idx_pos += FULL_POS_PER_CTX * ctx
        self.dec_cand_pos += N_CAND_LAYERS * min(ctx, CAND_SCORED)

    def add_prefill(self, start: int, n: int) -> None:
        """A chunk of n prefill rows at positions [start, start + n)."""
        self.pf_rows += n
        self.pf_idx_pos += FULL_POS_PER_CTX * n * (start + 0.5 * n)
        tail = min(n, SLIDING_WINDOW)
        self.pf_tail_rows += tail
        self.pf_cand_pos += N_CAND_LAYERS * tail * min(start + n, CAND_SCORED)

    def copy(self) -> "RankLoad":
        return RankLoad(**self.__dict__)


@dataclass
class StepCost:
    total_us: float
    rank_a_us: list = field(default_factory=list)   # per-rank side, layers 0..20 (incl. their indexers)
    rank_b_us: list = field(default_factory=list)   # per-rank side, layers 21..39
    shared_us: float = 0.0
    head_us: float = 0.0

    @property
    def idle_fraction(self) -> float:
        """Rank-time lost waiting for the slowest rank, as a fraction of the step (per-rank side only)."""
        n = len(self.rank_a_us)
        if not n or self.total_us <= 0:
            return 0.0
        ma, mb = max(self.rank_a_us), max(self.rank_b_us)
        lost = sum(ma - a for a in self.rank_a_us) + sum(mb - b for b in self.rank_b_us)
        return lost / (n * self.total_us)


class CostModel:
    def __init__(self, p: CostParams | None = None):
        self.p = p or CostParams()

    def rank_sides(self, ld: RankLoad) -> tuple[float, float]:
        p = self.p
        a = N_A * (p.a0 + p.a_row * ld.dec_rows + p.a_pf_row * ld.pf_rows)
        a += p.idx_dec_us_per_pos * ld.dec_idx_pos + p.idx_dec_fixed_us * ld.n_dec_seqs
        a += p.idx_pf_us_per_pos * ld.pf_idx_pos
        b = N_B * (p.a0 + p.a_row * (ld.dec_rows + ld.pf_tail_rows))
        b += p.idx_dec_us_per_pos * ld.dec_cand_pos + p.idx_pf_us_per_pos * ld.pf_cand_pos
        return a, b

    def layer_sides(self, ld: RankLoad) -> tuple[float, float, float, float]:
        """Per-rank side split by layer kind: (per non-index layer of 0..20, the 4 index layers' indexer total, per layer
        of 21..39, the candidate layers' indexer total)."""
        p = self.p
        base_a = p.a0 + p.a_row * ld.dec_rows + p.a_pf_row * ld.pf_rows
        idx_a = p.idx_dec_us_per_pos * ld.dec_idx_pos + p.idx_dec_fixed_us * ld.n_dec_seqs + p.idx_pf_us_per_pos * ld.pf_idx_pos
        base_b = p.a0 + p.a_row * (ld.dec_rows + ld.pf_tail_rows)
        idx_b = p.idx_dec_us_per_pos * ld.dec_cand_pos + p.idx_pf_us_per_pos * ld.pf_cand_pos
        return base_a, idx_a, base_b, idx_b

    def step(self, loads: list[RankLoad], copy_bytes: int = 0) -> StepCost:
        p = self.p
        sides = [self.rank_sides(ld) for ld in loads]
        ra, rb = [s[0] for s in sides], [s[1] for s in sides]
        if p.ideal_balance:
            ra, rb = [sum(ra) / len(ra)] * len(ra), [sum(rb) / len(rb)] * len(rb)
            att = max(ra) + max(rb)
        else:
            # the ranks meet at every layer (AG / RS): a step pays, per layer, its slowest rank. The indexer runs only on
            # the 4 KV-source layers (0..20) and the 4 candidate layers (21..39), so a deep rank sets those layers only
            # (an AG profile: rows moved by depth lose on the 17 other layers). Sum over layers of the max,
            # not the max of each rank's sum (identical when the ranks are alike).
            ls = [self.layer_sides(ld) for ld in loads]
            ni = len(KV_SOURCE_LAYERS)
            att = (N_A - ni) * max(x[0] for x in ls) + ni * max(x[0] + x[1] / ni for x in ls)
            att += (N_B - N_CAND_LAYERS) * max(x[2] for x in ls) + N_CAND_LAYERS * max(x[2] + x[3] / N_CAND_LAYERS for x in ls)
        dec = sum(ld.dec_rows for ld in loads)
        pf = sum(ld.pf_rows for ld in loads)
        pft = sum(ld.pf_tail_rows for ld in loads)
        rows_a = [ld.dec_rows + (0 if p.pf_tok_us else ld.pf_rows) for ld in loads]
        rows_b = [ld.dec_rows + ld.pf_tail_rows for ld in loads]
        if p.coll_real_rows or p.ideal_balance:
            rmax_a, rmax_b = sum(rows_a) / len(loads), sum(rows_b) / len(loads)
        else:
            rmax_a, rmax_b = max(rows_a), max(rows_b)
        if dec + pf == 0:
            return StepCost(0.0, ra, rb, 0.0, 0.0)
        if pf and p.moe_mixed0 > 0.0:          # mixed step: the measured MoE(T) curve (weight stream + per token)
            shared = N_A * (p.c0 + p.c_row * rmax_a + p.moe_mixed0 + p.moe_mixed_row * (dec + pf))
            shared += N_B * (p.c0 + p.c_row * rmax_b + p.moe_mixed0 + p.moe_mixed_row * (dec + pft))
        else:
            shared = N_A * (p.c0 + p.c_row * rmax_a + p.moe0 + p.moe_dec_row * dec + p.moe_pf_row * pf)
            shared += N_B * (p.c0 + p.c_row * rmax_b + p.moe0 + p.moe_dec_row * dec + p.moe_pf_row * pft)
        head = p.head0 + p.head_row * max(ld.dec_rows for ld in loads) if dec else 0.0
        shared += p.pf_tok_us * pf
        total = att + shared + head + (p.pf_overhead if pf else 0.0) + p.copy_us_per_mb * copy_bytes / 1e6
        total += p.host_us + (p.dec_only_extra_us if not pf else 0.0)
        if not pf and p.eager_ctx_above and max(ld.max_dec_ctx for ld in loads) > p.eager_ctx_above:
            total += p.eager_extra_us
        return StepCost(total, ra, rb, shared, head)

    # -- helpers for the scheduler's placement / quota decisions
    def mixed_fixed_us(self) -> float:
        """Fixed cost a step pays for carrying any prefill rows: the mixed-step overhead plus, with the MoE(T) curve, the
        extra weight stream of the mixed MoE over the decode MoE (per layer moe_mixed0 - moe0)."""
        p = self.p
        extra = N_LAYERS * max(0.0, p.moe_mixed0 - p.moe0) if p.moe_mixed0 > 0.0 else 0.0
        return p.pf_overhead + extra

    def decode_seq_rank_us(self, ctx: int, rows: int) -> float:
        """Per-rank cost a decoding sequence adds (its rows through 40 layers + its indexer)."""
        p = self.p
        return (N_LAYERS * p.a_row * rows + p.idx_dec_us_per_pos * (FULL_POS_PER_CTX * ctx + N_CAND_LAYERS * min(ctx, CAND_SCORED))
                + p.idx_dec_fixed_us)

    def prefill_rank_us(self, start: int, n: int) -> float:
        """Per-rank cost of prefilling positions [start, start + n) of one sequence (one or more chunks)."""
        ld = RankLoad()
        ld.add_prefill(start, n)
        a, b = self.rank_sides(ld)
        return a + b - N_LAYERS * self.p.a0

    def prefill_shared_us(self, n: int) -> float:
        return N_A * self.p.moe_pf_row * n + N_B * self.p.moe_pf_row * min(n, SLIDING_WINDOW) + self.p.pf_tok_us * n


# ---------------------------------------------------------------------------------------------- production reference
@dataclass
class ProdParams:
    """Production (vLLM 0.32 + production's ds41h package, THROUGHPUT k=3, TP4, KV replicated) as ONE logical rank.
    Rough fit, only used to calibrate the synthetic workload against a measured replay of real traffic (3,650 tok/s at
    144 conversations):
      decode step 20.4 ms + 61 us per row (a 128-sequence profile) + context term fitted so that T = 384 rows of real
      traffic gives 65.6-69 ms (a production profile: ~23 ms of long-context attention + replicated indexer for 96
      sequences);
      prefill 15 us per token + 1.6e-5 us per token per context position (cold TTFT 128K 2.22 s, 512K 11.9 s), and a
      mixed step (eager, no graph) costs +34.9 ms."""
    dec0_us: float = 20_400.0
    dec_row_us: float = 61.0
    ctx_us_per_tok: float = 3.0e-3       # per decoding sequence per context token per step
    pf_tok_us: float = 15.0
    pf_ctx_us: float = 1.6e-5
    mixed_us: float = 34_900.0


class ProdCostModel(CostModel):
    """Same interface as CostModel for a single logical rank (n_ranks = 1)."""

    def __init__(self, p: ProdParams | None = None):
        super().__init__(CostParams())
        self.q = p or ProdParams()

    def _ctx(self, ld: RankLoad) -> float:
        return ld.dec_idx_pos / FULL_POS_PER_CTX

    def rank_sides(self, ld: RankLoad) -> tuple[float, float]:
        q = self.q
        a = q.dec_row_us * ld.dec_rows + q.ctx_us_per_tok * self._ctx(ld)
        a += q.pf_tok_us * ld.pf_rows + q.pf_ctx_us * ld.pf_idx_pos / FULL_POS_PER_CTX
        return a, 0.0

    def step(self, loads: list[RankLoad], copy_bytes: int = 0) -> StepCost:
        assert len(loads) == 1
        ld = loads[0]
        if ld.dec_rows + ld.pf_rows == 0:
            return StepCost(0.0, [0.0], [0.0], 0.0, 0.0)
        a, _ = self.rank_sides(ld)
        total = self.q.dec0_us + a + (self.q.mixed_us if ld.pf_rows else 0.0)
        return StepCost(total, [a], [0.0], self.q.dec0_us, 0.0)

    def decode_seq_rank_us(self, ctx: int, rows: int) -> float:
        return self.q.dec_row_us * rows + self.q.ctx_us_per_tok * ctx

    def mixed_fixed_us(self) -> float:
        return 0.0              # production's greedy chunked prefill (the reference keeps its own policy)

    def prefill_rank_us(self, start: int, n: int) -> float:
        return self.q.pf_tok_us * n + self.q.pf_ctx_us * n * (start + 0.5 * n)
