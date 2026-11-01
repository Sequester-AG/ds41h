// ds41h decoder: persistent post-attention layer kernel for decode on sm90a (H200), TP4.
//
// Adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
// src/brrr/kernels/decoder.cu layer_kernel). See LICENSES/BRRR-Apache-2.0.txt.
//
// One launch runs everything stock vLLM 0.32.0 does after the attention core of one
// DeepseekV4DecoderLayer (models/deepseek_v41/nvidia/model.py:591-625) for M <= MAXM tokens:
//   inverse RoPE on o -> wo_a (2 local groups) -> wo_b (K slice) -> NVLink IPC all-reduce #1 ->
//   attention mHC post + FFN mHC pre (hc_ffn_fn GEMV, RMS, Sinkhorn x20) -> RMSNorm(ffn_norm) ->
//   gate (bf16) -> sqrtsoftplus + bias top-6 (*1.5/sum) -> shared expert (FP8) + routed experts
//   (FP4, raw checkpoint layout) -> combine -> all-reduce #2.
//
// Changes vs BRRR (sm120a):
//  * No block-scaled mma on sm90: every 32-wide K block is two FP16 HMMA m16n8k16 into a
//    temporary accumulator, post-scaled by the ue8m0 weight scale (and, for wo_b, the per-token
//    activation exponent). Dense weights: e4m3 -> f16 via cvt.rn.f16x2.e4m3x2. Routed FP4:
//    e2m1 -> e4m3 bit trick (exact, x2^-6) -> f16. Activations are staged as f16 (stock sm90
//    runs these linears W8A16 / W4A16; f16 holds every bf16 value exactly in range).
//  * K is permuted inside every 8-group as P8 = [0,2,4,6,1,3,5,7] so that the FP4 nibble order
//    of the raw checkpoint (low nibble = even k) and the dense e4m3 bytes feed HMMA without any
//    per-MMA shuffles; activations are written in P8 order by the jobs that produce them.
//  * Routed experts read the raw TP-sliced checkpoint layout (w13 [E,1152,2560], s13
//    [E,1152,160], w2 [E,5120,288], s2 [E,5120,18]) through a 16B-chunk XOR-swizzled smem tile
//    and ldmatrix.x4 (no repack, one copy of the weights).
//  * The attention-side mixes come from stock/prologue (no J_AMIX); the FFN mHC follows stock's
//    fused TileLang path (kernels/mhc/tilelang_kernels.py:668-786 + :298-503): the 24-wide GEMV
//    and the RMS run on the fp32 post-mapped stream, the collapse on its bf16 copy.
//  * Runtime solo mode (nranks == 1): the all-reduces are local copies, for single-GPU tests.
//  * Hopper: 132 SMs x PER_SM CTAs, deeper cp.async ring (STAGES), one-shot all-reduce on
//    NVLink (TWO_SHOT_MIN > MAXM by default).
//
// NUMERICS: the default build is LOSSLESS vs stock vLLM 0.32.0 on sm90: every GEMM gets
// 16-bit activations (f16 staging of bf16 values, exact for |v| in [2^-17, 65504]; wo_b's z per 32 block scaled by a
// power of two) with the exact ue8m0 weight scales; only fp32 summation order differs. The coarse m <= 8 path is
// the lossless R1F / R2F HMMA path (DS41H_DECODER_LOSSLESS, see below). The e4m3-activation variants
// (ROUTED_MMA=1, DENSE_MMA=1, LOSSLESS=0 coarse) are LOSSY opt-ins.
//
// RS-wgmma variants (ds41h, compile-time switches, default off = the HMMA paths above; LOSSY):
//  * DS41H_DECODER_ROUTED_MMA=1: routed experts (R1 w13, R2 w2) and DS41H_DECODER_DENSE_MMA=1:
//    wo_a / wo_b / shared expert run wgmma.mma_async m64n{8,16}k32 .f32.e4m3.e4m3 with the
//    weights as the register A operand (one warpgroup = the 4 consumer warps = 64 weight rows,
//    warp w owns rows 16w..16w+15) and the tokens as the N = 8 (16) B operand from smem.
//    Entries / token groups with more than 16 tokens keep the HMMA path.
//  * A fragments need no shuffles: thread (g, t) holds row g / g+8 bytes of k32 columns 4t..4t+3
//    (a0, a1) and 16+4t..16+4t+3 (a2, a3). The FP4 word of row g at k = 8t..8t+7 (ldmatrix, as
//    in the HMMA path) gives a0 = fp4_lo_e4m3 (k 8t+{0,2,4,6}) and a2 = fp4_hi_e4m3 (k 8t+{1,3,
//    5,7}); the P8-staged dense fragment (v.x, v.y, v.z, v.w) gives a0..a3 = (v.x, v.z, v.y,
//    v.w). Both imply the same in-block K order ("wg order") for the activations: physical
//    k = 8t + j lives at B column 16 (j & 1) + 4t (j >> 1) (wg_pos()).
//  * Activations are e4m3 in wg order in scratch, written once by the jobs that produce them:
//    x (FFN input) with a static per-layer power-of-two scale xs >= max|ffn_norm| sqrt(5120)/448
//    (RMSNorm output bound), SwiGLU outputs with 1/4 (|h| <= 100), z (wo_a output) with its
//    per-32 f16 staging exponent + 6; wo_a's input o_rot is quantized per (token, 32) in-job
//    (official act_quant ue8m0 rule). B tiles are K-major, no swizzle: 8 token x 16 B core
//    matrices, LBO = NT x 128 B (K), SBO = 128 B (tokens).
//  * Every k32 wgmma writes its own accumulator set (4 floats per thread per 8 tokens), two
//    commit groups of four k32 blocks in flight; after wgmma.wait_group 1 the older group is
//    promoted into the fp32 totals with its per-(row, k32) weight scale (one FFMA per element).
//    FP4 scales go through a per-warp float table (one ue8m0 word per lane per stage).

#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

#include "host.cuh"
#include "sm90.cuh"

#ifndef DS41H_DECODER_MAXM
#define DS41H_DECODER_MAXM 32
#endif
#ifndef DS41H_DECODER_STAGES
#define DS41H_DECODER_STAGES 6
#endif
#ifndef DS41H_DECODER_PER_SM
#define DS41H_DECODER_PER_SM 2
#endif
#ifndef DS41H_DECODER_TWO_SHOT_MIN
#define DS41H_DECODER_TWO_SHOT_MIN (DS41H_DECODER_MAXM + 1)
#endif
// Routed-expert MMA flavour: 0 = FP16 HMMA m16n8k16 (FP4 -> e4m3 -> f16, activations f16);
// 1 = RS-wgmma m64n8k32 e4m3 (FP4 -> e4m3 in registers, activations e4m3, see header).
#ifndef DS41H_DECODER_ROUTED_MMA
#define DS41H_DECODER_ROUTED_MMA 0
#endif
// Dense FP8 (wo_a, wo_b, shared expert) MMA flavour: 0 = HMMA (e4m3 -> f16), 1 = RS-wgmma e4m3.
#ifndef DS41H_DECODER_DENSE_MMA
#define DS41H_DECODER_DENSE_MMA 0
#endif
static_assert(DS41H_DECODER_ROUTED_MMA == 0 || DS41H_DECODER_ROUTED_MMA == 1, "ROUTED_MMA: 0 = HMMA, 1 = RS-wgmma");
static_assert(DS41H_DECODER_DENSE_MMA == 0 || DS41H_DECODER_DENSE_MMA == 1, "DENSE_MMA: 0 = HMMA, 1 = RS-wgmma");
// Coarse routed path (decoder-v2) for m <= COARSE_M tokens (one 8-token wgmma tile): the FFN input x
// is staged ONCE per CTA as a resident e4m3 B tile (40 KB, union with the f16 staging buffer bt);
// R1 jobs stream 32 gate + 32 up rows of one expert over the full K (or K/2, R1_KP) with RS-wgmma and
// apply SwiGLU in their epilogue (no split-K atomics, no per-job activation staging), writing h for
// all 8 token columns into a per-expert B-tile image; R2 jobs (128 rows x K 576 of one expert) depend
// only on their expert's R1 jobs and are interleaved with later experts' R1 jobs (lag). Default 1
// (validated: solo + 4-GPU tests, end-to-end KL PASS, TG c1 317 vs 266 tok/s); 0 = old R1/R2 jobs.
#ifndef DS41H_DECODER_COARSE
#define DS41H_DECODER_COARSE 1
#endif
// Coarse R1 inner loop: 0 = pairs of stages with a pipeline drain every 16 k32; 2 / 3 = the whole
// job as one straight-line RS-wgmma pipeline with 2 / 3 commit groups in flight.
#ifndef DS41H_DECODER_R1C_NB
#define DS41H_DECODER_R1C_NB 2
#endif
// Fused attention all-reduce (decoder-v2, one-shot mode only): the wo_b job finishing a 64-column row
// block publishes it (bf16, own inbox) and raises that row block's flag in the peers right away; every
// OWNA slice job waits for its row block's peer flags (it is an exchange job: no job is pre-claimed
// while it waits) and reduces its 32 columns from the 4 inboxes (no XA jobs, no yattn round trip).
// 0 = BRRR XA jobs.
#ifndef DS41H_DECODER_FUSED_AR1
#define DS41H_DECODER_FUSED_AR1 1
#endif
// Fused z (decoder-v2): wo_a jobs only release their 512-column z range's counter (no finish_woa);
// every wo_b job quantizes its own z slice straight from the fp32 wo_a sums (finish_woa's numerics),
// the gate jobs clear zacc afterwards. 0 = finish_woa (last K part) + zh / zq / ze round trip.
#ifndef DS41H_DECODER_FUSED_Z
#define DS41H_DECODER_FUSED_Z 1
#endif
// One-wave wo_b (decoder-v2, coarse m <= 8, HMMA dense builds): 160 wo_b jobs of 1024 K instead of
// 320 of 512 K, so they all run in one wave on the 264 CTAs.
#ifndef DS41H_DECODER_WOB1
#define DS41H_DECODER_WOB1 0
#endif
// Coarse R1 stages by TMA (per-layer tensor maps): 2 x [32 rows x 128 B] 128B-swizzled weight boxes +
// 2 x [32 rows x 16 B] scale boxes (the scale bytes of the stage's 2-stage pair) per 9 KB stage,
// issued by one producer lane. 0 = cp.async per lane (16 B weights, 8 B scales).
#ifndef DS41H_DECODER_R1C_TMA
#define DS41H_DECODER_R1C_TMA 1
#endif
// Dense job stages (wo_a, wo_b, gate, shared expert: one contiguous 8 KB staged block each) by one
// cp.async.bulk instead of 512 x 16 B cp.async. Default: on with the coarse path.
#ifndef DS41H_DECODER_DENSE_BULK
#define DS41H_DECODER_DENSE_BULK DS41H_DECODER_COARSE
#endif
// Routing: 1 = hierarchical (the finisher of each 64-expert gate row block takes its top-6, the last of
// them merges 36 candidates); 0 = the last gate job routes all 384 logits (2 fewer loaded round trips).
#ifndef DS41H_DECODER_HIER_ROUTE
#define DS41H_DECODER_HIER_ROUTE 1
#endif
// Lossless coarse path (phase 4, default 1): with COARSE=1 the m <= 8 routed path keeps its coarse job
// structure (TMA-streamed full-K row blocks, R2 jobs lagging their expert's R1 jobs) but every GEMM gets
// 16-bit activations: R1F = 128 rows (64 gate + 64 up) x full K per job, FP4 -> f16 exactly (value *
// 2^-14, integer bit placement) x f16 x (128-k slices streamed with the weights) on HMMA m16n8k16 with
// the exact per-(row, 32 k) ue8m0 scales; R2F = 128 rows of w2 x f16 h; the shared expert takes the HMMA
// fill_q / dense_q path (f16 x / h). 0 = the lossy e4m3 coarse path (resident e4m3 x tile, RS-wgmma
// R1C / R2C, e4m3 shared expert): opt-in only.
#ifndef DS41H_DECODER_LOSSLESS
#define DS41H_DECODER_LOSSLESS 1
#endif
// LC tuning: R2F_PF = R2F h slices streamed by the producer with the weights (into the x-slice
// ring; it checks the expert's R1F completion itself, the consumer skips that dependency wait); LAZY_SIG = R2F
// jobs post their chunk's C_SF arrival lazily (at the next job, after its first stage), so the release fence
// does not stall on the red.adds just issued; R1F_PIPE = software-pipelined LC stage loop (stage s + 1's A / B
// fragments and scales are loaded before stage s's math).
// Defaults 0: the solo bench-multi showed -2..3 us each, but an end-to-end bisect at
// 350 W (4 GPUs, real routing) measured each of them SLOWER in situ (layer_kernel all on 160.0 us,
// R2F_PF+LAZY_SIG off 158.7, R1F_PIPE off 158.6, all off (+GDET=0) 152.0).
#ifndef DS41H_DECODER_R2F_PF
#define DS41H_DECODER_R2F_PF 0
#endif
#ifndef DS41H_DECODER_LAZY_SIG
#define DS41H_DECODER_LAZY_SIG 0
#endif
#ifndef DS41H_DECODER_R1F_PIPE
#define DS41H_DECODER_R1F_PIPE 0
#endif
// Coarse layout: the 80 shared-expert w2 jobs right after the SH1 jobs (they run while routing finishes and the
// first routed weights land) instead of after the first `lag` experts' R1 jobs.
#ifndef DS41H_DECODER_SH2_EARLY
#define DS41H_DECODER_SH2_EARLY 0
#endif
// Producer-local routing (LC, coarse m): no route_block / route_all / routing table round trips. Once every gate
// job has signalled C_G, every CTA's producer warp reads the router sums itself, takes the top-6 per token (the
// same select as route_all) and keeps the expert slots / per-(slot, token) weights in its own shared memory.
// Router sums are double-buffered by launch parity (gate jobs clear the other buffer).
// PROUTE=2: the hierarchical route_block step stays (6 row-block CTAs: per token top-6 of 64 experts),
// but route_all's single-CTA merge + global table + C_ROUTE hop are replaced by every CTA's producer merging the 36
// candidates per token itself (the same select_topk_loop as route_all: identical selection and weights) and building the
// PROUTE smem slot table. The route_block finishers signal C_G (6 arrivals).
#ifndef DS41H_DECODER_PROUTE
#define DS41H_DECODER_PROUTE 0
#endif
// Fast routing (coarse m <= 8): one counter C_G over all gate jobs; the last gate job routes all m
// tokens over the 384 logits with all 4 consumer warps (2 tokens per warp, no route_block hierarchy), builds only what
// the coarse path needs (expert slots in ascending expert order, per-(slot, token) weights, slot count) and releases
// C_ROUTE; the zacc / router-buffer clears follow the arrive (off the routing path). 0 = HIER_ROUTE / route_all.
// FROUTE=2: same, with select_fast (lane-best + REDUX rounds where only the round's winner lane rescans its candidates,
// rolled loop: ~100 instructions per round instead of select_topk_loop's ~570 for 12 candidates per lane).
#ifndef DS41H_DECODER_FROUTE
#define DS41H_DECODER_FROUTE 0
#endif
// WOA RoPE prefetch: before griddepcontrol.wait, the consumers of a CTA whose first job is a wo_a
// job (m <= WOA_SMALL) load the cos / sin rows of the m tokens' positions into shared memory (wf), so the wo_a job only
// waits for the attention output o (one dependent L2 round trip instead of positions -> cos_sin -> RoPE).
#ifndef DS41H_DECODER_WOA_PRE
#define DS41H_DECODER_WOA_PRE 0
#endif
// Polling: POLLF=1 = waits poll with relaxed loads and end with fence.acquire (SASS: CCTL.IVALL,
// the same as the ld.acquire it replaces minus its L2 round trip; the old path re-read the counter with ld.acquire
// twice around the loop). POLL_NS = sleep between polls (ns).
#ifndef DS41H_DECODER_POLLF
#define DS41H_DECODER_POLLF 0
#endif
#ifndef DS41H_DECODER_POLL_NS
#define DS41H_DECODER_POLL_NS 200
#endif
// PDL_EARLY: every CTA executes griddepcontrol.launch_dependents at its start (all CTAs are resident then), so the next
// kernel (the next layer's prologue in the model) is launched right away and its CTAs take SMs as this kernel's CTAs
// exit (its pre-wait work overlaps this kernel's tail). The dependent's griddepcontrol.wait still waits for this whole
// grid (memory included), so its reads of our outputs stay ordered.
#ifndef DS41H_DECODER_PDL_EARLY
#define DS41H_DECODER_PDL_EARLY 0
#endif
// SCL_DIRECT (LC routed jobs): no per-warp float scale tables. Each lane loads the ue8m0 words of its rows g / g+8 of
// both m-tiles with the stage's other operands (R1F: 4 LDS.32 from the scale ring; R2F: 4 SHFL of the job's per-row
// words) and converts the byte at use (2 integer ops per scale). Removes lc_put's LDS -> convert -> STS -> syncwarp ->
// LDS chain at every stage boundary (ABL=4 showed the tables cost ~9 us/layer at m=6, mostly latency).
#ifndef DS41H_DECODER_SCL_DIRECT
#define DS41H_DECODER_SCL_DIRECT 0
#endif
// R1KP_DYN = k (LC, m >= 4): the R1F K split is chosen when the routing is known: full-K jobs (40 stages) while the
// 9 n R1F jobs fit in one wave of CTAs, else k K parts per row block (40 / k stages each; partial sums + SwiGLU by the
// last part, as r1kp > 1). Full-K jobs with 9 n > ctas leave a second wave of ~54 us jobs on a few CTAs whose experts'
// R2F jobs wait for it (4-GPU trace, n = 32: R2 phase +30..40 us). 0 = r1kp from m only (host). 1 = auto (r1kp_auto:
// k in {1, 2, 5} by an estimated makespan; n = 30..35 -> 5, larger n -> 2).
#ifndef DS41H_DECODER_R1KP_DYN
#define DS41H_DECODER_R1KP_DYN 0
#endif
// REL_DEP (LC routed jobs): the stage release (empty-barrier arrive) returns its mbarrier state token, which is folded
// (AND with a runtime zero) into the initial accumulator of the stage's first HMMA, so ptxas cannot sink the release below
// the MMAs (SASS of the default build: the arrive sat after ~95% of the stage's math, i.e. every ring slot was held for
// the consumer's whole math time and slot cycle = memory latency + math -> stream and math serialized).
#ifndef DS41H_DECODER_REL_DEP
#define DS41H_DECODER_REL_DEP 0
#endif
// R1SPLIT (LC coarse m, host r1kp = 1): balanced R1F overflow. With 9 n R1F row blocks > ctas (n >= 30
// experts at 264 CTAs, typical at m = 8 / k = 7) full-K jobs leave a second R1 wave of ~40-stage jobs on a few CTAs, and the
// R2F jobs of those experts (and XF) wait for it (~+15-20 us per layer at n = 30). R1SPLIT keeps full-K jobs for the first
// nf = floor(ctas / 9) experts and splits the row blocks of the remaining n - nf experts into kps K parts (10 / 5 / 2 parts of
// 4 / 8 / 20 stages, the most that still fit in one CTA wave), claimed right after the full-K jobs; partial sums + SwiGLU by
// the last part (the r1kp > 1 path). Claim order [R1 full][R1 parts][SH2][R2 x 40 n][XF]. Lossless (fp32 order only).
#ifndef DS41H_DECODER_R1SPLIT
#define DS41H_DECODER_R1SPLIT 0
#endif
// OWNA_PRE (FUSED_AR1, m <= 8): an OWNA job stages its inputs that do not depend on the all-reduce
// (attention mHC post / comb / pre of the m tokens, the slice's resid0 columns) into bt by cp.async BEFORE it waits for the
// peers' wo_b flags, so after the flags only the inbox loads remain on the critical path (in-situ OWNA tile ~7.4 us: the
// mixes were two dependent L2 round trips after the inbox loads). Same values, same arithmetic (bit-identical).
#ifndef DS41H_DECODER_OWNA_PRE
#define DS41H_DECODER_OWNA_PRE 0
#endif
// RESET_LEAN: the last CTA's end-of-launch counter reset (on the critical path: the next kernel waits for this
// grid) clears only what the coarse m <= 8 path uses ([0, C_R1A + 48 x 18), C_R1D[0, 48), [C_NENT, COUNTERS): ~1.8k of 8.3k words).
#ifndef DS41H_DECODER_RESET_LEAN
#define DS41H_DECODER_RESET_LEAN 0
#endif
// ROUTE1 (coarse m <= 8): one-hop routing. One counter C_G over all gate jobs (arrival BEFORE gate_clears);
// the last arriving gate job's CTA routes all m tokens over the 384 logits itself (route1_nc): no route_block step, no
// C_G hop, no candidate round trip. Per token (one warp per 2 tokens, 12 logits per lane): composite key (ordered key << 32 |
// ~expert: key desc, ties -> lowest expert = route_all's order); T = the 6th largest of the 32 lane maxima (every top-6 key
// is >= T, at most 6 lanes hold keys >= T); the <= 72 candidates >= T are ranked by counting in shared memory; weights
// w = s * (1.5 / sum of the 6 scores in rank order) exactly as route_all. Lean table (route_lean_nc's slots / wtok).
// In-situ k=7 (dec20t): gate GEMM end -> table = 17 us through route_block 4.6 + hop 1.0 + route_all 5.6.
#ifndef DS41H_DECODER_ROUTE1
#define DS41H_DECODER_ROUTE1 0
#endif
// R2P (LC coarse m <= 8): R2F jobs of R2P consecutive 128-row blocks of the same expert (1 = one block, the
// default; 2 / 4). In-situ k=7 trace (dec22t): an R2F job is ~4.3 us of math in ~8.7 us of wall time per job (stage-0 data
// wait 0.9: the next job's stages are claimed / issued only once this one runs and land ~5 us after issue under the routed
// stream; dependency poll 0.45; h-tile bulk copy 0.6; red4 epilogue + C_SF release ~1.0). A multi-block job pays the claim,
// dependency poll and h-tile copy once and streams the next block's stages back to back; each block keeps its own exact
// math and red4 epilogue (lossless: identical per-block arithmetic, fp32 atomic order only), the C_SF arrivals of all its
// blocks are released together at the end. The last R2T experts (in claim order) keep single-block jobs for a fine tail.
#ifndef DS41H_DECODER_R2P
#define DS41H_DECODER_R2P 1
#endif
#ifndef DS41H_DECODER_R2T
#define DS41H_DECODER_R2T 6
#endif
// PFR1 (ROUTE1 + LC coarse m): right after the routing table is released, the router CTA issues L2
// prefetches (TMA prefetch.tensor) of the first PFR1 stages (w13 gate + up boxes) and of scale group 0 of every R1F row
// block of every routed expert. In-situ k=7 (dec22t): table 72.8 us -> producers see it 74.0 -> first R1 issue 75.8 ->
// first stage landed 78.6, p50 80.7 (all CTAs request their first 6 stages at once, ~12 MB from HBM); the HBM is nearly idle
// during routing. Prefetch only (no data or order change: lossless by construction). 0 = off.
#ifndef DS41H_DECODER_PFR1
#define DS41H_DECODER_PFR1 0
#endif
// WOB3 (FUSED_Z + FUSED_AR1, coarse m <= 8): wo_b in ONE wave. Default: 80 row blocks x 4 K parts = 320
// jobs of 4 stages on 264 CTAs -> a second wave (in-situ dec22t critical path: WOB 21.0-31.3 then 32.4-41.5 on the same CTA,
// its OWNA slice ends 53.4 and gates the router). WOB3: 3 K parts of 5 / 5 / 6 stages (k32 [0,20) [20,40) [40,64)) = 240
// jobs, each fits the 6-stage ring (WOB1's 8-stage jobs did not). A part waits for the C_Z ranges it spans; the z fill
// issues all its loads before converting (one L2 round trip instead of one per 128 vectors). Same per-k32 math, fp32 sum
// order of the red.adds only (lossless).
#ifndef DS41H_DECODER_WOB3
#define DS41H_DECODER_WOB3 0
#endif
// HOTDISP (LC): the consumer's dependency check and job dispatch test the routed types (R1C / R2C, ~90% of
// the jobs) with direct branches before the switch (jump tables: BRX + a constant-bank load per level). In-situ k=7 the first
// R1F job of a CTA spends 3.1 us between its dependency check and its first instruction (0.16 us for the CTA's later R1F
// jobs; solo 1.5 us): cold dispatch path under the routed stream. Control flow only.
#ifndef DS41H_DECODER_HOTDISP
#define DS41H_DECODER_HOTDISP 0
#endif
// TAILFREE: no end-of-launch tail. Default: every CTA's exit does a fence + atomicAdd(C_FIN) and the last
// CTA resets the dependency counters and advances the epoch before the grid completes; the next layer's (PDL) prologue
// waits for that (in-situ dec22t: last consumer exit -> reset + epoch done 2.8 us). TAILFREE: two counter sets (epoch
// parity); every CTA clears its slice of the OTHER set (used by the previous launch, complete) right after the PDL wait,
// the epoch is advanced by the XF job of chunk 0 (all CTAs read the epoch before their first job, which XF transitively
// waits for), the producer takes the epoch from shared memory (published by consumer thread 0 before its first ready).
// Control only (no numerics).
#ifndef DS41H_DECODER_TAILFREE
#define DS41H_DECODER_TAILFREE 0
#endif
// LLP (FUSED_AR1, m <= 8): both in-kernel all-reduces as LL pushes. Default (pull): the publisher stores its
// bf16 partial in its own inbox, fences (__threadfence_system) and raises flags in the peers; the consumer polls its flag,
// then loads the peers' partials over NVLink (a full remote round trip on the critical path: in-situ OWNA flags -> inbox
// landed 0.7-2.6 us, XF post 2.3 + peer wait 2.4 + reduce 1.3 us). LLP: the publisher writes 8-byte words {2 bf16, epoch}
// straight into every rank's LL inbox (one NVLink write, no fence, no separate flag); the consumer polls its LOCAL inbox
// until all words carry this launch's epoch and sums the ranks in rank order exactly as reduce4 / owna_tile (bit-identical).
// AR#1: finish_wob -> owna_tile; AR#2: job_xf (push its chunk, then poll + reduce). Exchange protocol only (no numerics).
#ifndef DS41H_DECODER_LLP
#define DS41H_DECODER_LLP 0
#endif
// R1SK (LC coarse m <= 8): stream-K R1F. The n x 9 row blocks x 40 stages (U stage units, expert-major,
// block, K) are cut into J = ceil(U / T) jobs of T = round4(ceil(U / ctas)) stages, i.e. ONE balanced wave whatever n is.
// Default: one 40-stage job per row block -> 9n jobs: n < 29 leaves 264 - 9n CTAs without R1 work for the whole ~45 us R1
// wave (their R2 work waits for it), n >= 30 needs a second wave (R1SPLIT: K parts of the overflow experts). In situ: span =
// 106 + 2.71 n us (traced, n = 20..40). A job's segments are row-block K ranges; a segment covering a whole row block takes
// the full-K path (in-job SwiGLU), otherwise its partial goes to hcacc (red4) and the row block's last segment (arrival
// count = segments covering it) runs finish_r1f -- the K-part path of r1kp > 1 / R1SPLIT. Lossless (fp32 partial order only).
// R1SK=2: T = max(4, ceil(U / ctas)) without rounding to whole scale groups (n = 27: 37 instead of 40 stages; average ~1.3
// stages less for n = 20..37). Scale groups are segment-relative: 4 stages from each segment start (TMA box at the stage's
// own byte offset kc * 4, any kc), a segment's last group may be short (consumer: one extra group step after the stage loop,
// the stage loop itself is unchanged), and before reusing a scale slot the producer waits until the stage that ended the
// slot's previous group was released (short groups break the implicit 3 groups x 4 stages >= ring depth + 1 guarantee).
#ifndef DS41H_DECODER_R1SK
#define DS41H_DECODER_R1SK 0
#endif
// PXF (the moe small16 race): every consumer thread runs fence.proxy.async.shared::cta
// before its warp releases a ring stage. The stages (and the bt x-slice / scale rings released with them) are refilled by TMA /
// bulk copies (async proxy) but read with generic ld.shared / ldmatrix; the PTX memory model orders those reads before the
// producer's next async-proxy write only through a proxy fence (the mbarrier release / acquire covers the generic proxy only).
// No numerics change. Opt-in until its in-situ cost is measured.
#ifndef DS41H_DECODER_PXF
#define DS41H_DECODER_PXF 0
#endif
// TAGX (ROUTE1 + R1SK): the router also writes each slot's expert as a u64 {expert, epoch} (own scratch xtag)
// before releasing C_ROUTE; a producer polls C_ROUTE (lane 0) and the tagged entries (all lanes) in the same
// relaxed loop and finishes with one fence.acquire -- no final ld.acquire round trip (poll_ge) and no separate table read after
// the table is seen (in situ: table released 72.8 -> producers see it 74.0 -> first R1 issue 75.8 us). Control only.
#ifndef DS41H_DECODER_TAGX
#define DS41H_DECODER_TAGX 0
#endif
// GSUM2 (ROUTE1, coarse m <= 8): deterministic router logits without atomics. Default (GDET): every gate job
// adds its 64 x m partial logits as u64 fixed point (red.add.u64: 40 per address), the acq_rel arrival waits for them (in-situ
// gate X2 -> end ~4 us). GSUM2: plain fp32 stores into a per-K-part slot, the last job of each 64-expert block sums its block's
// KP_GATE partials in K-part order (one fp32 order, identical on every rank) into glog, the last of the 6 blocks routes
// (route1_nc reads glog). Lossless (fp32 summation order only; rank-identical like GDET).
#ifndef DS41H_DECODER_GSUM2
#define DS41H_DECODER_GSUM2 0
#endif
// DENSE_TS1: dense GEMM jobs with one 8-token tile (m <= 8) run a specialized stage loop (warp w = row tile w), whose
// executed code is contiguous and ~4x smaller than the generic loop's (which interleaves the unused r = 1..3 tiles).
#ifndef DS41H_DECODER_DENSE_TS1
#define DS41H_DECODER_DENSE_TS1 0
#endif
// R2F_ROLL: the R2F stages 1-4 as a rolled loop (one ~300-instruction body instead of 4 unrolled copies: the job's code is
// fetched once per job per SM, ~24 KB straight-line otherwise -> instruction-fetch bound).
#ifndef DS41H_DECODER_R2F_ROLL
#define DS41H_DECODER_R2F_ROLL 0
#endif
// R1F_LOC (LC R1F jobs): the R1F stage loop as a real call (__noinline__ r1f_loop) on plain locals and symbol-based shared
// addresses. Inlined into the persistent kernel the loop shares the 168 registers with the kernel's live state and ptxas
// spilled loop values into it (SASS: 4 LDL + 1 STL per stage: the scale-group counter and two table bases); a call gets
// the whole budget (the caller's live registers are saved once per job by the ABI). Stage release before the MMAs (as
// REL_DEP). Same math and summation order as the inlined loop (lossless). Needs ABL=0, R1F_PIPE=0, SCL_DIRECT=0.
// R1F_LOC=2: stage pairs with interleaved (independent) HMMAs, same FFMA order (loc_stage2).
#ifndef DS41H_DECODER_R1F_LOC
#define DS41H_DECODER_R1F_LOC 0
#endif
// R2F_LOC: the same for the R2F stages 1-4 (__noinline__ r2f_loop; the inlined loop keeps 2 spill reloads per stage).
// Needs R2F_PF=0 too.
#ifndef DS41H_DECODER_R2F_LOC
#define DS41H_DECODER_R2F_LOC 0
#endif
// PROD_ARR: for TMA / bulk-copy ring stages the producer's lane 0 makes the 32 "cp.async" arrivals of the full barrier
// (count 33 = 32 + lane 0's expect_tx) with ONE mbarrier.arrive of count 32, instead of 32 per-lane
// cp.async.mbarrier.arrive.noinc (the stage has no cp.async): fewer smem atomics per stage in the producer warp
// (ABL=129 trace: a no-op ring stage still costs ~0.45 us of producer + handshake time). cp.async (dense) stages unchanged.
#ifndef DS41H_DECODER_PROD_ARR
#define DS41H_DECODER_PROD_ARR 0
#endif
// DYN_FIRST: every CTA's FIRST job is claimed from a counter (C_FIRST) instead of being job blockIdx.x. In the model the
// layer kernel is PDL-launched while the attention still holds SMs, so some CTAs are placed late; with static first jobs
// those late CTAs own wo_a jobs (every wo_b job waits for all of them), with DYN_FIRST the CTAs that start first take them.
#ifndef DS41H_DECODER_DYN_FIRST
#define DS41H_DECODER_DYN_FIRST 0
#endif
// WARM_ROUTE: before its griddepcontrol.wait (in the model: while the attention still runs), the LAST CTA's consumers run
// route_block + route_all on a private warm scratch (Params::ws: own counters / table / router sums / candidates), so
// the routing code (one-shot per launch, otherwise fetched cold from HBM by whichever CTA routes ~60 us later) is in L2.
// In-situ trace (dec18t): gate GEMM end -> R1 weights landed = 25 us, route_block 4.5 us, route_all select 4 us.
#ifndef DS41H_DECODER_WARM_ROUTE
#define DS41H_DECODER_WARM_ROUTE 0
#endif
// CLEAR_LATE (HIER_ROUTE): a gate job arrives on its row block counter BEFORE clearing zacc / the next launch's router
// buffer (gate_clears), so the arrive's release does not wait for those stores and the last arriver starts route_block
// (and route_all) earlier; the clears follow (they are needed by the next launch only).
#ifndef DS41H_DECODER_CLEAR_LATE
#define DS41H_DECODER_CLEAR_LATE 0
#endif
// ROUTE_LEAN (HIER_ROUTE, coarse m): the last route_block finisher runs route_lean_nc instead of route_all: the same
// candidate merge (select_topk_loop over the 36 row-block candidates: identical selection and weights), then only what the
// coarse path reads (expert slots in ascending order from a 384-bit mask, per-(slot, token) weights, C_NENT) instead of
// route_all's counting sort / route lists / chunk table (in-situ trace: route_all table build ~1.9 us).
#ifndef DS41H_DECODER_ROUTE_LEAN
#define DS41H_DECODER_ROUTE_LEAN 0
#endif
// SMEM_DIET: bt sized for what the build uses (the lossy coarse path's resident e4m3 x tile only when LOSSLESS=0) and the
// R2 scale table ws aliased into wf (no job uses both): 103,424 B smem at STAGES 6 (was 109,568), so STAGES 7 (112,640 B)
// still fits 2 CTAs / SM. Layout only (no math change). Default: on for STAGES >= 7.
#ifndef DS41H_DECODER_SMEM_DIET
#define DS41H_DECODER_SMEM_DIET (DS41H_DECODER_STAGES >= 7)
#endif
// Deterministic router sums: the gate jobs' K-part partials are added as 64-bit fixed point (2^-32 units,
// red.add.u64: exact and order-independent) instead of fp32 red.add, so every rank (and every CTA's producer under
// PROUTE) sees bitwise-identical logits (stock's replicated gate GEMM is identical across ranks; fp32 atomics in
// arrival order are not, which could split a near-tie routing between ranks). One rounding to fp32 at the end.
#ifndef DS41H_DECODER_GDET
#define DS41H_DECODER_GDET 1
#endif
// Coarse mode: the producer issues the shared w1/w3 (SH1) weights only once x is complete (1, decoder-v2), or right
// away (0: they stream during the latency-bound gate phase; SH1 jobs are claimed after every gate job).
#ifndef DS41H_DECODER_SH1_GATE
#define DS41H_DECODER_SH1_GATE 1
#endif
// Ring waits (consumers on full stages, the producer on empty slots) as mbarrier.try_wait with a suspend-time hint
// (ns): the warp sleeps until the phase completes instead of re-polling at the default (short) time limit, so idle
// warps neither burn issue slots / power nor starve the producer warp. 0 = no hint (previous behaviour).
#ifndef DS41H_DECODER_MBAR_SLEEP
#define DS41H_DECODER_MBAR_SLEEP 0
#endif
// Ablation bits (BENCH ONLY: results are wrong when set) for the LC routed jobs (R1F / R2F):
// 1 = consumers only acquire / release the stages (data path + ring handshake), 2 = no FP4 -> f16 conversion (raw
// words into the HMMA), 4 = no per-warp scale tables (constant scales), 8 = no HMMA (conversions XOR-folded),
// 16 = every expert slot streams the weights of slot 0's expert (L2-resident routed weights), 32 = R1F scale groups not
// loaded, 64 = R2F K-tail box not loaded, 128 = no routed stage copies at all (consumer math on stale smem: the in-kernel
// consumer cost with the ring handshake but without the memory stream). 256 = streamed dense GEMM jobs (wo_a / wo_b /
// shared expert; not the gate, so routing stays real) skip their MMAs (stage handshake, fills and epilogues kept).
#ifndef DS41H_DECODER_ABL
#define DS41H_DECODER_ABL 0
#endif
// CTX_INLINE: every device function taking the per-thread Ctx (and Ctx's wait / signal / arrive) is force-inlined, so the
// Ctx (it, sgc, lazy, warp, lane, ...) can live in registers instead of the 472-byte local stack frame (spill reloads of
// c.warp / c.it / c.sgc sat inside the hot stage loops).
#ifndef DS41H_DECODER_CTX_INLINE
#define DS41H_DECODER_CTX_INLINE 0
#endif
#if DS41H_DECODER_CTX_INLINE
#define CTXFN __forceinline__
#else
#define CTXFN
#endif
// Trace mode (ds41h/decoder_trace.py): every CTA records per job the producer claim / issue times
// and the consumer free / data-ready / dependencies-met / end times (%globaltimer), plus the
// exchange sub-phases, into a ring of TRACE_LAUNCHES launches (slot = epoch % TRACE_LAUNCHES).
#ifndef DS41H_DECODER_TRACE
#define DS41H_DECODER_TRACE 0
#endif

#include <type_traits>
#include <utility>

namespace {

constexpr bool ROUTED_WG = DS41H_DECODER_ROUTED_MMA == 1;
constexpr bool DENSE_WG = DS41H_DECODER_DENSE_MMA == 1;
constexpr bool COARSE = DS41H_DECODER_COARSE != 0;
constexpr bool LC = COARSE && DS41H_DECODER_LOSSLESS != 0;  // lossless coarse path (R1F / R2F, HMMA shared expert)
constexpr bool LCQ = COARSE && !LC;                         // lossy e4m3 coarse path (R1C / R2C, opt-in)
constexpr bool R2F_PF = LC && DS41H_DECODER_R2F_PF != 0;
constexpr bool LAZY_SIG = LC && DS41H_DECODER_LAZY_SIG != 0;
constexpr bool R1F_PIPE = LC && DS41H_DECODER_R1F_PIPE != 0;
constexpr bool LC_PS = DS41H_DECODER_R1F_PIPE != 2;  // R1F_PIPE=2: scales read at math time (fewer live registers)
constexpr bool PROUTE = LC && DS41H_DECODER_PROUTE != 0;
constexpr bool PROUTE2 = PROUTE && DS41H_DECODER_PROUTE == 2;  // producers merge route_block's candidates (see route_local_nc)
constexpr bool FROUTE = COARSE && !PROUTE && DS41H_DECODER_FROUTE != 0;
constexpr bool WOA_PRE = DS41H_DECODER_WOA_PRE != 0;
constexpr bool SCL_DIRECT = DS41H_DECODER_SCL_DIRECT != 0;
constexpr bool REL_DEP = DS41H_DECODER_REL_DEP != 0;
constexpr bool R1F_LOC = LC && DS41H_DECODER_R1F_LOC != 0 && (DS41H_DECODER_ABL & 255) == 0 && !R1F_PIPE && !SCL_DIRECT;
constexpr bool R2F_LOC = LC && DS41H_DECODER_R2F_LOC != 0 && (DS41H_DECODER_ABL & 255) == 0 && !R1F_PIPE && !SCL_DIRECT && !R2F_PF;
constexpr bool GDET = DS41H_DECODER_GDET != 0;
constexpr int ABL = DS41H_DECODER_ABL;                      // bench-only ablations of the LC routed jobs (see above)
constexpr int COARSE_M = 8;                                 // coarse routed path for m <= 8 (one 8-token tile)
constexpr bool XQ_ON = ROUTED_WG || DENSE_WG || LCQ;        // write the e4m3 copy of the FFN input x
constexpr int WG_NT = 2;                        // wgmma paths take <= WG_NT x 8 tokens (else HMMA)
constexpr float H_INV = 4.0f;                   // e4m3 code = SwiGLU output * 4 (|h| <= 100 -> <= 400)

// ------------------------------------------------------------------------------- trace layout
constexpr bool TRACE = DS41H_DECODER_TRACE != 0;
constexpr int TRACE_LAUNCHES = 8;  // ring of launches (epoch % 8)
constexpr int TRACE_CAP = 192;     // job records per CTA per launch
constexpr int TRACE_REC = 24;      // u32 words per record (see decoder_trace.py)
constexpr int TRACE_HDR = 20;      // u32 header words per CTA
constexpr int TRACE_CTA = TRACE_HDR + TRACE_CAP * TRACE_REC;
// Record words: 0 type | c << 8 | n << 16, 1 a | b << 16, 2 flags, 3 t_claim (producer got the job
// index), 4 t_issue (producer: stage-0 slot free, copies start), 5 t_route (producer: routing table
// ready, first routed job only), 6 t_free (consumer idle), 7 t_acq (stage 0 landed), 8 t_dep
// (dependencies met), 9 t_end, 10..12 exchange marks (posted, peers' flags seen, reduced),
// 13 ns spun in Ctx::wait, 14 Ctx::wait calls that spun, 15 ns waited for later stages to land.
// Non-exchange jobs use 10..12 as sub-phase marks (see the job functions).
// Times are ns since the CTA's entry (%globaltimer), header 2/3 = entry time lo/hi.
enum : int { TR_TYPE, TR_AB, TR_FLAGS, TR_CLAIM, TR_ISSUE, TR_ROUTE, TR_FREE, TR_ACQ, TR_DEP, TR_END, TR_X0, TR_X1, TR_X2, TR_SPIN, TR_SPINS, TR_SWAIT, TR_G, TR_RB, TR_RBE, TR_RAE,
             TR_S0, TR_S1, TR_S2, TR_S3 };
// TR_S0..TR_S3 (words 20..23): generic sub-phase marks (thread 0) of one-shot code: route_block (gate loads
// landed / select done / candidates stored), route_all (loads landed / select done), OWNA (inbox landed / mixes done / lane sums
// done / stores issued).
// TR_G (word 16): end of a dense GEMM's stage loop (gemm / dense_wg), before its stores. Routing (gate jobs that
// finish a row block / route all): TR_RB (17) route_block start, TR_RBE (18) route_block end, TR_RAE (19)
// route_all done (just before the C_ROUTE release); route_all also overwrites X0 (start) / X1 (top-k) / X2 (table).
enum : unsigned { TF_FINISH = 1u };  // the job ran the finish step of its group (arrive() == last)

[[maybe_unused]] __device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// The wgmma variants need every job that issues wgmma inlined into the kernel: ptxas serializes
// all wgmmas of a called function (C7510 "pipeline crossing function boundary").
#if DS41H_DECODER_ROUTED_MMA || DS41H_DECODER_DENSE_MMA || DS41H_DECODER_COARSE
#define WG_INLINE __forceinline__
#else
#define WG_INLINE
#endif

// ------------------------------------------------------------------------------- model shape
constexpr int HC = 4;
constexpr int MIX = (2 + HC) * HC;  // 24 mixing coefficients
constexpr int HID = 5120;
constexpr int HCD = HC * HID;
constexpr int MAXM = DS41H_DECODER_MAXM;
constexpr int TILE = 8;
constexpr int TOPK = 6;
constexpr int ROUTES = MAXM * TOPK;
constexpr int EXPERTS = 384;
constexpr int SPAN = 32;  // routes per (expert, chunk) entry
constexpr int ENTRIES = EXPERTS + ROUTES / SPAN;
constexpr int HDIM = 512;
constexpr int ROPE = 64;
constexpr int NOPE = HDIM - ROPE;
constexpr int OA = 2048;   // wo_a outputs of this rank: 2 local groups x o_lora_rank 1024
constexpr int OAK = 4096;  // wo_a input per group: 8 heads x 512
constexpr int INTER = 576; // moe_intermediate_size 2304 / TP4
constexpr int SLICES = HID / 32;
constexpr int SH2_LD = 20;  // shared w2 scale columns: 18 padded to 20 (4-byte cp.async)
constexpr float LIMIT = 10.0f;      // swiglu_limit
constexpr int SINKHORN = 20;        // hc_sinkhorn_iters
constexpr float HC_EPS = 1e-6f;     // hc_eps
constexpr float NORM_EPS = 1e-20f;  // rms_norm_eps (mHC RMS and ffn_norm)
constexpr float ROUTE_SCALE = 1.5f; // routed_scaling_factor
// Raw routed-expert layout (bytes): w13 rows are K/2 packed e2m1, w2 rows INTER/2.
constexpr int W13_ROW = HID / 2;
constexpr int S13_ROW = HID / 32;
constexpr int W2_ROW = INTER / 2;
constexpr int S2_ROW = INTER / 32;

// ------------------------------------------------------------------------------ kernel shape
constexpr int GW = 4;            // consumer warps
constexpr int GT = GW * 32;      // consumer threads
constexpr int THREADS = GT + 32; // + 1 producer warp
constexpr int PER_SM = DS41H_DECODER_PER_SM;
constexpr int STAGES = DS41H_DECODER_STAGES;
constexpr int FRAG = 512;
constexpr int FRAGS = 16;
constexpr int SCALE_OFF = FRAGS * FRAG;
constexpr int STAGE = SCALE_OFF + 1024;  // 9 KB, 1024-aligned stages (TMA 128B swizzle atoms)
constexpr bool R1C_TMA = DS41H_DECODER_R1C_TMA != 0;
constexpr bool DENSE_BULK = DS41H_DECODER_DENSE_BULK != 0;
constexpr bool HIER_ROUTE = DS41H_DECODER_HIER_ROUTE != 0;
constexpr int OWNA_ROWS = 16;
constexpr int TWO_SHOT_MIN = DS41H_DECODER_TWO_SHOT_MIN;
constexpr bool FUSED_AR1 = DS41H_DECODER_FUSED_AR1 != 0 && TWO_SHOT_MIN > DS41H_DECODER_MAXM;  // one-shot only
constexpr bool FUSED_Z = DS41H_DECODER_FUSED_Z != 0;

constexpr int CHUNKS = 40;
constexpr int CHUNK_COLS = HID / CHUNKS;
constexpr int CHUNK_VECS = CHUNK_COLS / 8;
constexpr int OTOK = 32;
constexpr int GROUPS = MAXM / OTOK;
constexpr int TILES = MAXM / TILE;
constexpr int MIXW = 32;
constexpr int KPARTS = 10;
constexpr int XNORM_MIN = 16;
constexpr int WOA_BLOCKS = OA / 64;
constexpr int WOB_BLOCKS = HID / 64;
constexpr int GATE_BLOCKS = EXPERTS / 64;
constexpr int SH1_BLOCKS = INTER / 32;
constexpr int KS_WOA = 8;
constexpr int KS_WOB = 16;
// GATE_KS: k16 steps per router-gate job (8: 240 jobs of 2 stages; 16: 120 jobs of 4 stages -> half the
// u64 router-sum atomics and 120 more CTAs free for the shared expert while the gate runs; pair with SH2_EARLY so the shared
// w2 jobs finish inside the routing gap instead of the routed phase). Same products / fixed-point sums (lossless).
#ifndef DS41H_DECODER_GATE_KS
#define DS41H_DECODER_GATE_KS 8
#endif
constexpr int KS_GATE = DS41H_DECODER_GATE_KS;
static_assert(KS_GATE == 8 || KS_GATE == 16, "GATE_KS: 8 or 16");
constexpr int KS_SH1 = 16;
constexpr int KP_WOB = OA / 32 / KS_WOB;
constexpr int KS_WOB1 = 2 * KS_WOB;  // one-wave wo_b K blocks per job
constexpr int KP_GATE = HID / 16 / KS_GATE;
constexpr int KP_SH1 = HID / 32 / KS_SH1;
constexpr int KS_INTER = INTER / 32;
constexpr int INTER_STAGES = cdiv(KS_INTER, 4);
constexpr int R1_BLOCKS = INTER / 64;
constexpr int R1_PARTS = 10;
constexpr int KS_R1 = SLICES / R1_PARTS;
constexpr int R2_BLOCKS = HID / 128;
constexpr int KG_WOA = OAK / 128;
constexpr int KG_WOB = OA / 128;
constexpr int KG_GATE = HID * 2 / 128;
constexpr int KG_SH1 = HID / 128;
constexpr int WOA_SMALL = 16;
constexpr int OWNA_SPANS = 4;
// R2 stage 0 holds the K tail (blocks 16, 17: 32 B/row) and the job's whole scale block.
constexpr int R2_TAIL_BYTES = 128 * 32;
constexpr int R2_SCALE_BYTES = 128 * S2_ROW;
constexpr int WS_LD = 20;  // R2 scale table row pitch (18 used)

// Coarse routed path (m <= COARSE_M).
constexpr int R1C_BLOCKS = INTER / 32;  // R1 jobs (32 gate + 32 up rows) per expert and K part
constexpr int R1C_STAGES = HID / 256;   // full K: 20 stages of 64 rows x 128 B (+ 64 x 8 B scales)
constexpr int XT_BYTES = HID * TILE;    // resident x tile: 8 tokens e4m3 in the wgmma B layout
constexpr int HQI = KS_INTER * 256;     // per-expert h image (18 k32 x 8 tokens x 32 B, B layout)
constexpr int TAB_LD = 20;              // coarse R1 scale table row pitch (16 used, conflict-free float4 reads)
constexpr int TAB_LD1 = 12;             // unrolled coarse R1: per-stage table row pitch (8 used, conflict-free)
constexpr int R1C_NB = DS41H_DECODER_R1C_NB;
constexpr int MAXE = COARSE_M * TOPK;   // distinct experts at m <= COARSE_M

// Lossless coarse path (LC): R1F jobs = 128-row blocks (64 gate + 64 up rows) of an expert's w13 over the
// full K (or K / r1kp) in stages of 128 rows x 64 B (128 k, 4 k32 blocks); R2F jobs = 128 rows of w2 (the R2C
// stages). Inside bt (free during the routed phase above the dense jobs' <= 16 KB B tiles at m <= 8): the
// per-stage f16 x slices [4 k32][8 tokens][64 B] (ring position it % STAGES) and a 3-slot ring of R1F scale
// groups (128 rows x 16 B = the scales of 4 stages, TMA every 4th stage). R2F's f16 h tile [18 k32][8][64 B]
// is bulk-copied by the consumer into bt[0, 9 KB).
constexpr int R1F_BLOCKS = INTER / 64;           // 9 R1F row blocks per expert
constexpr int R1F_STAGES = HID / 128;            // 40 stages of 64 B per row (full K)
constexpr int XSL = 4 * TILE * 64;               // x slice per R1F stage: 2 KB
constexpr int XS_OFF = 16384;                    // x slice ring in bt
constexpr int SCL = 128 * 16;                    // one R1F scale group: 2 KB
constexpr int SC_OFF = XS_OFF + STAGES * XSL;    // 3-slot scale group ring in bt
constexpr int HHI = KS_INTER * TILE * 64;        // f16 h tile of one expert slot (9 KB)
constexpr int XHI = SLICES * TILE * 64;          // f16 x image of 8 token columns (80 KB)
constexpr float FP4_F16 = 16384.0f;              // the direct FP4 -> f16 expansion is value * 2^-14
constexpr int R1X_BLOCKS = LC ? R1F_BLOCKS : R1C_BLOCKS;  // coarse R1 row blocks per expert

// bt: f16 staging (<= OTOK x KS_INTER x 64), resident e4m3 x tile (lossy coarse only), LC x-slice / scale rings above 16 KB.
constexpr bool WS_SEP = ROUTED_WG || LCQ || DS41H_DECODER_SMEM_DIET == 0;  // R2 scale table outside wf (SMEM_DIET)
constexpr int BT_LC_RINGS = 16384 + DS41H_DECODER_STAGES * 4 * 8 * 64 + 3 * 128 * 16;
constexpr int BT_A = OTOK * KS_INTER * 64 > BT_LC_RINGS ? OTOK * KS_INTER * 64 : BT_LC_RINGS;
constexpr int BT = DS41H_DECODER_SMEM_DIET ? ((DS41H_DECODER_COARSE && !DS41H_DECODER_LOSSLESS) && XT_BYTES > BT_A ? XT_BYTES : BT_A)
                                           : (OTOK * KS_INTER * 64 > XT_BYTES ? OTOK * KS_INTER * 64 : XT_BYTES);
// Per-warp ue8m0 -> float tables of the wgmma routed jobs ([stage parity][warp][32 rows x 4]) and
// of the coarse R1 jobs ([warp][16 rows x TAB_LD]).
constexpr int WFT_A = 2 * GW * 128 > GW * 16 * TAB_LD ? 2 * GW * 128 : GW * 16 * TAB_LD;
constexpr int WFT = WFT_A > 2 * GW * 16 * TAB_LD1 ? WFT_A : 2 * GW * 16 * TAB_LD1;
constexpr int BS = OTOK * KS_WOB;         // per-(token, K block) activation scale factors (wo_b)
constexpr int WS = 128 * WS_LD;

static_assert(MAXM % OTOK == 0 && MAXM <= 128, "MAXM must be a multiple of 32, at most 128");
static_assert(SCALE_OFF + 512 <= STAGE && CHUNK_COLS == 128 && FRAGS * FRAG == 8192 && R2_BLOCKS == CHUNKS);
static_assert(OWNA_ROWS * HC * 32 * 4 <= STAGE && (MIX - OWNA_ROWS) * HC * 32 * 4 <= STAGE);
static_assert(R2_TAIL_BYTES + R2_SCALE_BYTES <= SCALE_OFF && R2_SCALE_BYTES % 16 == 0);
static_assert(OTOK * KS_WOA * 64 <= BT && WOA_SMALL * 2 * KS_WOA * 64 <= BT && OTOK * KS_WOB * 64 <= BT && OTOK * KS_SH1 * 64 <= BT &&
              OTOK * KS_R1 * 64 <= BT && OTOK * KS_GATE * 32 <= BT);
static_assert(KS_INTER == 18 && INTER_STAGES == 5 && KS_R1 == 16);
static_assert(STAGES >= 2, "OWNA and the gate hold two stages");
static_assert(KS_INTER * 2 * 2 * 128 <= BT, "wgmma B tile must fit bt");
static_assert(R1C_STAGES % 2 == 0 && HQI <= BT && 2 * 32 * 16 * 2 <= BS * 4, "coarse R1: stage pairs, h image, SwiGLU exchange in bs");
static_assert(!LC || (SC_OFF + 3 * SCL <= BT && HHI <= XS_OFF && R1C_TMA && STAGES <= 9), "lossless coarse: x slice / scale rings in bt, h tile below them, TMA");
static_assert(2 * 32 * 8 <= BS && 2 * GW * 128 <= WFT, "R1F SwiGLU exchange in bs, per-warp scale tables in wf");

__host__ __device__ __forceinline__ int tiles_of(int m) { return cdiv(m, TILE); }
__host__ __device__ __forceinline__ int token_groups(int m) { return cdiv(m, OTOK); }
__host__ __device__ __forceinline__ bool split_norm(int m) { return m > XNORM_MIN; }
__host__ __device__ __forceinline__ int quads(int m) { return cdiv(m, GW); }
__host__ __device__ __forceinline__ int woa_ks(int m) { return m <= WOA_SMALL ? 2 * KS_WOA : KS_WOA; }
// wo_b k32 blocks per job / K parts (one-wave wo_b for the coarse regime, see DS41H_DECODER_WOB1).
__host__ __device__ __forceinline__ int wob_ks(int m) {
  return DS41H_DECODER_WOB1 && !DS41H_DECODER_DENSE_MMA && DS41H_DECODER_COARSE && m <= 8 ? 2 * KS_WOB : KS_WOB;
}
static_assert(!(DS41H_DECODER_WOB3 && DS41H_DECODER_WOB1), "WOB3 replaces WOB1");
// WOB3 (see the knob): K part b of a wo_b row block = stages 5b .. 5b + 4 (+ 1 for b = 2), k32 blocks 20b ..
constexpr bool WOB3 = DS41H_DECODER_WOB3 != 0 && DS41H_DECODER_FUSED_Z != 0 && DS41H_DECODER_DENSE_MMA == 0;
__host__ __device__ __forceinline__ bool wob3_m(int m) { return WOB3 && DS41H_DECODER_COARSE && m <= 8; }
__host__ __device__ __forceinline__ int wob3_st(int b) { return b == 2 ? 6 : 5; }
__host__ __device__ __forceinline__ int wob_kp(int m) { return wob3_m(m) ? 3 : OA / 32 / wob_ks(m); }
static_assert(!WOB3 || FUSED_AR1, "WOB3 needs FUSED_AR1 (plain WOB job range)");

struct Span {
  int first, count, tiles, ts;
};

__host__ __device__ __forceinline__ Span span_of(int first, int count) {
  const int tiles = tiles_of(count);
  return {first, count, tiles, tiles == 1 ? 1 : tiles == 2 ? 2 : 4};
}
__host__ __device__ __forceinline__ Span group_of(int m, int tg) { return span_of(tg * OTOK, m - tg * OTOK < OTOK ? m - tg * OTOK : OTOK); }
__host__ __device__ __forceinline__ Span tile_span(int m, int t) { return span_of(t * TILE, m - t * TILE < TILE ? m - t * TILE : TILE); }
__host__ __device__ __forceinline__ int owna_tiles(int m) { return cdiv(tiles_of(m), OWNA_SPANS); }
__host__ __device__ __forceinline__ int owna_spans(int m) { return cdiv(tiles_of(m), owna_tiles(m)); }

// Job index space, in claim order: [WOA][WOB + lagged XA][OWNA][XNORM][GATE][ROUTE][SH1][MIX]
// [R1 x entries][per chunk: SH2, R2 x entries, lagged XF].
// Coarse mode (m <= COARSE_M, see coarse()): the routed section is instead
// [R1C of experts 0..ln-1][SH2 x 80][per expert e >= ln: R1C(e), R2C(e - ln)][R2C of the last ln][XF x 40]
// with ln = min(lag, n): an expert's R2C jobs trail its R1C jobs by `lag` experts.
struct Layout {
  int m, tg, wob, owna, xnorm, gate, route, sh1, sh2, mix, routed;
  bool coarse;
  int kp, lag, u1;  // coarse: K parts per R1C row block, expert lag of R2C, R1C jobs per expert
  int nf, kps;      // R1SPLIT: experts with full-K R1F jobs, K parts of the others' row blocks (kps = 1: no split)
  int skT, skJ, skU;  // R1SK: stages per job, jobs, stage units (0 = off)
};

__host__ __device__ __forceinline__ bool coarse_m(int m) { return COARSE && m <= COARSE_M; }
constexpr int R1KP_DYN = DS41H_DECODER_R1KP_DYN;
static_assert(R1KP_DYN <= 1 || (40 % R1KP_DYN == 0 && (40 / R1KP_DYN) % 4 == 0),
              "R1KP_DYN must split the 40 R1F stages into parts of whole 4-stage scale groups (2, 5, 10), or be 1 (auto)");

__host__ __device__ __forceinline__ Layout layout(int m, int kp = 1, int lag = 1) {
  Layout l{};
  l.m = m;
  l.tg = token_groups(m);
  l.wob = WOA_BLOCKS * (OAK / 32 / woa_ks(m)) * l.tg;
  l.owna = l.wob + CHUNKS * (2 * wob_kp(m) * l.tg + (FUSED_AR1 ? 0 : 1));
  l.xnorm = l.owna + SLICES * owna_spans(m);
  l.gate = l.xnorm + (split_norm(m) ? KPARTS * tiles_of(m) : 0);
  l.route = l.gate + GATE_BLOCKS * KP_GATE * l.tg;
  l.sh1 = l.route;  // routing runs inside the last gate job (route_all), no ROUTE jobs
  l.sh2 = l.sh1 + SH1_BLOCKS * KP_SH1 * l.tg;
  l.mix = l.sh2 + (DS41H_DECODER_SH2_EARLY && coarse_m(m) ? 2 * CHUNKS : 0);  // SH2_EARLY: [SH2 x 80] before MIX
  l.routed = l.mix + quads(m);
  l.coarse = coarse_m(m);
  l.kp = kp;  // 0 = dynamic (R1KP_DYN): fixed by with_n() once the expert count is known
  l.lag = lag;
  l.u1 = (LC ? R1F_BLOCKS : R1C_BLOCKS) * kp;
  l.nf = 0;
  l.kps = 1;
  l.skT = l.skJ = l.skU = 0;
  return l;
}

// R1KP_DYN: the K split of the R1F jobs from the routed expert count n (same on every CTA: n comes from the table).
// R1KP_DYN=1 (auto): k in {1, 2, 5} minimizing waves(k) x (40 / k + overhead) stage-times, waves = ceil(9 n k / ctas)
// (overhead ~1 stage per full-K job, ~3 per K part: partial red.adds + arrive + the last part's SwiGLU), ties -> smaller k.
__device__ __forceinline__ int r1kp_auto(int n) {
  const int u = (LC ? R1F_BLOCKS : R1C_BLOCKS) * n, ctas = int(gridDim.x);
  int best = 1, bt = (u + ctas - 1) / ctas * (40 + 1);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int k = i ? 5 : 2, t = (u * k + ctas - 1) / ctas * (40 / k + 3);
    if (t < bt) bt = t, best = k;
  }
  return best;
}
constexpr bool R1SPLIT = LC && DS41H_DECODER_R1SPLIT != 0;
constexpr bool R1SK = LC && DS41H_DECODER_R1SK != 0;
constexpr bool R1SKU = R1SK && DS41H_DECODER_R1SK == 2;  // any T (segment-relative scale groups, see R1SK)
constexpr bool PXF = DS41H_DECODER_PXF != 0;
static_assert(DS41H_DECODER_R1SK >= 0 && DS41H_DECODER_R1SK <= 2, "R1SK: 0, 1 (T multiple of 4) or 2 (any T)");
static_assert(!(DS41H_DECODER_R1SK && (DS41H_DECODER_R1SPLIT || DS41H_DECODER_R1F_LOC || DS41H_DECODER_R1F_PIPE || DS41H_DECODER_SCL_DIRECT)),
              "R1SK replaces R1SPLIT; plain R1F stage loop only");
// R1SK: stages per stream-K job for n experts on `ctas` CTAs (>= 4; R1SK=1: multiple of 4 = whole scale groups)
__host__ __device__ __forceinline__ int r1sk_T(int n, int ctas) {
  const int u = n * R1F_BLOCKS * R1F_STAGES, t = (u + ctas - 1) / ctas;
  return DS41H_DECODER_R1SK == 2 ? max(4, t) : max(4, (t + 3) & ~3);
}
__device__ __forceinline__ Layout with_n(Layout l, int n) {
  if (R1SK && l.coarse) {
    l.skU = n * R1F_BLOCKS * R1F_STAGES;
    l.skT = r1sk_T(n, int(gridDim.x));
    l.skJ = (l.skU + l.skT - 1) / l.skT;
    return l;
  }
  if (l.kp == 0) {
    l.kp = (LC ? R1F_BLOCKS : R1C_BLOCKS) * n <= int(gridDim.x) ? 1 : R1KP_DYN == 1 ? r1kp_auto(n) : R1KP_DYN;
    l.u1 = (LC ? R1F_BLOCKS : R1C_BLOCKS) * l.kp;
  }
  if (R1SPLIT && l.coarse && l.kp == 1 && R1F_BLOCKS * n > int(gridDim.x)) {
    l.nf = int(gridDim.x) / R1F_BLOCKS;
    // K parts minimizing the overflow makespan ceil(rem k / ctas) x (40 / k + 2) stage-times (~2 stages of per-part
    // overhead: partial red.adds, arrival, the last part's SwiGLU); k = 1 = the plain second wave.
    const int rem = R1F_BLOCKS * (n - l.nf), ctas = int(gridDim.x);
    int best = 1, bt = cdiv(rem, ctas) * (R1F_STAGES + 2);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int k = i == 0 ? 10 : i == 1 ? 5 : 2, t = cdiv(rem * k, ctas) * (R1F_STAGES / k + 2);
      if (t < bt) bt = t, best = k;
    }
    l.kps = best;
    if (l.kps == 1) l.nf = 0;
  }
  return l;
}

constexpr int R1_JOBS = R1_BLOCKS * R1_PARTS;
constexpr int SH2_JOBS = 2 * CHUNKS;  // per token group (64-row blocks of the shared w2)
__host__ __device__ __forceinline__ int chunk_jobs(const Layout& L, int n) { return 2 * L.tg + n + 1; }
// R2P: R2F jobs of a trailing expert-major section of `ne` experts (first expert e0): experts before the last R2T take
// R2P-block jobs, the last R2T single-block jobs. r2_jobs = job count, r2_job = job r (block count - 1 in Job.c).
constexpr int R2P = (LC && DS41H_DECODER_R2P > 1) ? DS41H_DECODER_R2P : 1;
static_assert(R2P == 1 || R2P == 2 || R2P == 4 || R2P == 5 || R2P == 8, "R2P must divide the 40 R2 row blocks");
static_assert(R2P == 1 || DS41H_DECODER_R2F_PF == 0, "R2P: not with R2F_PF");
static_assert(R2P == 1 || DS41H_DECODER_LAZY_SIG == 0, "R2P: not with LAZY_SIG");
static_assert(DS41H_DECODER_PFR1 == 0 || (DS41H_DECODER_ROUTE1 != 0 && DS41H_DECODER_PFR1 <= 8), "PFR1 needs ROUTE1 (route1_nc issues the prefetches), <= 8 stages");
static_assert(DS41H_DECODER_TAGX == 0 || (DS41H_DECODER_ROUTE1 != 0 && DS41H_DECODER_R1SK != 0), "TAGX needs ROUTE1 (tagged table) and R1SK (smem expert table)");
static_assert(DS41H_DECODER_GSUM2 == 0 || DS41H_DECODER_ROUTE1 != 0, "GSUM2 needs ROUTE1 (route1_nc reads the block sums)");
static_assert(R2P == 1 || DS41H_DECODER_R2T >= 0, "R2T >= 0");
__host__ __device__ __forceinline__ int r2_jobs(int ne) {
  if constexpr (R2P == 1) return ne * CHUNKS;
  const int t = min(DS41H_DECODER_R2T, ne);
  return (ne - t) * (CHUNKS / R2P) + t * CHUNKS;
}
__device__ __forceinline__ int total_jobs(const Layout& L, int n) {
  if (R1SK && L.skJ > 0) return L.routed + L.skJ + (DS41H_DECODER_SH2_EARLY ? 0 : SH2_JOBS) + r2_jobs(n) + CHUNKS;
  if (R1SPLIT && L.kps > 1) return L.routed + R1F_BLOCKS * (L.nf + (n - L.nf) * L.kps) + r2_jobs(n) + (DS41H_DECODER_SH2_EARLY ? 0 : SH2_JOBS) + CHUNKS;
  if (L.coarse) {
    const int ln = min(L.lag, n);
    return L.routed + n * L.u1 + (n - ln) * CHUNKS + r2_jobs(ln) + (DS41H_DECODER_SH2_EARLY ? 0 : SH2_JOBS) + CHUNKS;
  }
  return L.routed + n * R1_JOBS + CHUNKS * chunk_jobs(L, n);
}

enum : int { J_WOA, J_WOB, J_XA, J_OWNA, J_XNORM, J_GATE, J_ROUTE, J_SH1, J_SH2, J_MIX, J_R1, J_R2, J_XF, J_R1C, J_R2C, J_END };

struct Job {
  int type, a, b, c, stages, e, f, n;
#if DS41H_DECODER_TRACE
  unsigned tq, ti, tr;  // producer times (trace mode): claim, stage-0 issue, routing table ready
#endif
};

struct Slot {
  int chunk, item;
};

// Exchange jobs trail their chunk's producers by `lag` chunks so they rarely block (BRRR).
__device__ __forceinline__ int exchange_lag(int per) { return min(CHUNKS, max(1, cdiv(int(gridDim.x), per))); }

__device__ __forceinline__ Slot lagged(int r, int per) {
  const int lag = exchange_lag(per);
  if (r < lag * per) return {r / per, r % per};
  r -= lag * per;
  if (r < (CHUNKS - lag) * (per + 1)) {
    const int k = lag + r / (per + 1), i = r % (per + 1);
    return i < per ? Slot{k, i} : Slot{k - lag, per};
  }
  return {CHUNKS - lag + r - (CHUNKS - lag) * (per + 1), per};
}

__host__ __device__ __forceinline__ Job blocked(int type, int r, int parts, int tg, int stages) { return {type, r / (parts * tg), r / tg % parts, r % tg, stages}; }

// Coarse routed section (see Layout): r = job index relative to L.routed.
__device__ __forceinline__ Job decode_coarse(const Layout& L, int n, int r) {
  const int ln = min(L.lag, n), u1 = L.u1, st = (LC ? R1F_STAGES : R1C_STAGES) / L.kp;
  auto r1c = [&](int e, int i) { return Job{J_R1C, e, i / L.kp, i % L.kp, st}; };
  auto r2c = [&](int e, int i) { return Job{J_R2C, e, i, 0, INTER_STAGES}; };
  // trailing R2 section of ne experts from e0 (R2P: multi-block jobs before the last R2T experts, see r2_jobs)
  auto r2s = [&](int e0, int ne, int i) -> Job {
    if constexpr (R2P > 1) {
      const int t = min(DS41H_DECODER_R2T, ne), big = (ne - t) * (CHUNKS / R2P);
      if (i < big) return Job{J_R2C, e0 + i / (CHUNKS / R2P), i % (CHUNKS / R2P) * R2P, R2P - 1, R2P * INTER_STAGES};
      i -= big;
      return r2c(e0 + ne - t + i / CHUNKS, i % CHUNKS);
    }
    return r2c(e0 + i / CHUNKS, i % CHUNKS);
  };
  if (R1SK && L.skJ > 0) {  // [R1SK jobs: a = job index, b = stages per job][SH2][R2 (r2s)][XF]
    if (r < L.skJ) return Job{J_R1C, r, L.skT, 0, min(L.skT, L.skU - r * L.skT)};
    r -= L.skJ;
    if (!DS41H_DECODER_SH2_EARLY) {
      if (r < SH2_JOBS) return {J_SH2, r, 0, 0, INTER_STAGES};
      r -= SH2_JOBS;
    }
    if (r < r2_jobs(n)) return r2s(0, n, r);
    r -= r2_jobs(n);
    if (r < CHUNKS) return {J_XF, r, 0, 0, 1};
    return {J_END, 0, 0, 0, 1};
  }
  if (R1SPLIT && L.kps > 1) {  // [R1 full: nf experts][R1 K parts: n - nf experts][SH2][R2 x 40 n][XF] (see R1SPLIT)
    const int full = R1F_BLOCKS * L.nf, parts = R1F_BLOCKS * (n - L.nf) * L.kps;
    if (r < full) return Job{J_R1C, r / R1F_BLOCKS, r % R1F_BLOCKS, 0, R1F_STAGES};
    r -= full;
    if (r < parts) {
      const int per = R1F_BLOCKS * L.kps, i = r % per;
      return Job{J_R1C, L.nf + r / per, i / L.kps, i % L.kps, R1F_STAGES / L.kps};
    }
    r -= parts;
    if (!DS41H_DECODER_SH2_EARLY) {
      if (r < SH2_JOBS) return {J_SH2, r, 0, 0, INTER_STAGES};
      r -= SH2_JOBS;
    }
    if (r < r2_jobs(n)) return r2s(0, n, r);
    r -= r2_jobs(n);
    if (r < CHUNKS) return {J_XF, r, 0, 0, 1};
    return {J_END, 0, 0, 0, 1};
  }
  if (r < ln * u1) return r1c(r / u1, r % u1);
  r -= ln * u1;
  if (!DS41H_DECODER_SH2_EARLY) {
    if (r < SH2_JOBS) return {J_SH2, r, 0, 0, INTER_STAGES};
    r -= SH2_JOBS;
  }
  const int per = u1 + CHUNKS;
  if (r < (n - ln) * per) {
    const int e = r / per, i = r % per;
    return i < u1 ? r1c(ln + e, i) : r2c(e, i - u1);
  }
  r -= (n - ln) * per;
  if (r < r2_jobs(ln)) return r2s(n - ln, ln, r);
  r -= r2_jobs(ln);
  if (r < CHUNKS) return {J_XF, r, 0, 0, 1};
  return {J_END, 0, 0, 0, 1};
}

__device__ __forceinline__ Job decode(const Layout& L, int n, int j) {
  if (j < L.wob) return blocked(J_WOA, j, OAK / 32 / woa_ks(L.m), L.tg, woa_ks(L.m) / 4);
  if (FUSED_AR1 && j < L.owna) {
    Job w = blocked(J_WOB, j - L.wob, wob_kp(L.m), L.tg, wob_ks(L.m) / 4);
    if (wob3_m(L.m)) w.stages = wob3_st(w.b);
    return w;
  }
  if (j < L.owna) {
    const int per = 2 * wob_kp(L.m) * L.tg;
    const Slot sl = lagged(j - L.wob, per);
    if (sl.item == per) return {J_XA, sl.chunk, 0, 0, 1};
    const Job w = blocked(J_WOB, sl.item, wob_kp(L.m), L.tg, wob_ks(L.m) / 4);
    return {J_WOB, 2 * sl.chunk + w.a, w.b, w.c, w.stages};
  }
  if (j < L.xnorm) return {J_OWNA, (j - L.owna) % SLICES, (j - L.owna) / SLICES, 0, 2};
  if (j < L.gate) return {J_XNORM, (j - L.xnorm) % KPARTS, (j - L.xnorm) / KPARTS, 0, 1};
  if (j < L.route) return blocked(J_GATE, j - L.gate, KP_GATE, L.tg, KS_GATE / 4);
  if (j < L.sh2) return blocked(J_SH1, j - L.sh1, KP_SH1, L.tg, KS_SH1 / 4);
  if (j < L.mix) return {J_SH2, j - L.sh2, 0, 0, INTER_STAGES};  // SH2_EARLY (coarse m)
  if (j < L.routed) return {J_MIX, j - L.mix, 0, 0, 1};
  int r = j - L.routed;
  if (L.coarse) return decode_coarse(L, n, r);
  if (r < n * R1_JOBS) return {J_R1, r / R1_JOBS, r / R1_PARTS % R1_BLOCKS, r % R1_PARTS, KS_R1 / 4};
  r -= n * R1_JOBS;
  if (r < CHUNKS * chunk_jobs(L, n)) {
    const int shared = 2 * L.tg;
    const Slot sl = lagged(r, shared + n);
    if (sl.item < shared) return {J_SH2, 2 * sl.chunk + sl.item / L.tg, 0, sl.item % L.tg, INTER_STAGES};
    return sl.item < shared + n ? Job{J_R2, sl.item - shared, sl.chunk, 0, INTER_STAGES} : Job{J_XF, sl.chunk, 0, 0, 1};
  }
  return {J_END, 0, 0, 0, 1};
}

// ------------------------------------------------------------------------------ counters
constexpr int C_CLAIM = 0;
constexpr int C_ZA = C_CLAIM + 1;                  // WOA K parts per (row block, group)
constexpr int C_Z = C_ZA + WOA_BLOCKS * GROUPS;    // finished z column blocks per (wo_b K part, group)
constexpr int C_YA = C_Z + KP_WOB * GROUPS;        // WOB K parts per (row block, group)
constexpr int C_YC = C_YA + WOB_BLOCKS * GROUPS;   // published wo_b row blocks per chunk
constexpr int C_YR = C_YC + WOB_BLOCKS;            // reduced attention chunks (C_YC: per chunk, or per row block when FUSED_AR1)
constexpr int C_A = C_YR + CHUNKS;                 // OWNA slices per 8-token tile
constexpr int C_XK = C_A + TILES;                  // normalized x parts per (512-K part, group)
constexpr int C_G = C_XK + KPARTS * GROUPS;        // gate jobs (all groups count into C_G + 0)
constexpr int C_ROUTE = C_G + GROUPS;              // routed tokens
constexpr int C_SA = C_ROUTE + 1;                  // SH1 K parts per (block, group)
constexpr int C_HS = C_SA + SH1_BLOCKS * GROUPS;   // shared activation blocks per group
constexpr int C_SF = C_HS + GROUPS;                // FFN contributions per chunk
constexpr int C_FIN = C_SF + CHUNKS;               // finished CTAs
constexpr int CACHED = C_FIN;
constexpr int C_PUB = C_FIN + 1;                   // exchange arrivals per (which, stage, chunk)
constexpr int C_R1A = C_PUB + 4 * CHUNKS;          // R1 K parts per (entry, block)
constexpr int C_R1D = C_R1A + ENTRIES * (R1_BLOCKS > R1C_BLOCKS ? R1_BLOCKS : R1C_BLOCKS); // finished R1 blocks per entry
constexpr int C_NENT = C_R1D + ENTRIES;
constexpr int C_ECOUNT = C_NENT + 1;
constexpr int C_GB = C_ECOUNT + EXPERTS;           // gate jobs per 64-expert row block (hierarchical routing)
constexpr int C_MX = C_GB + GATE_BLOCKS;           // finished MIX jobs (XF_P1A: the XF jobs read the FFN mHC coefficients)
constexpr int C_FIRST = C_MX + 1;                  // DYN_FIRST: first-job claims
static_assert(!(DS41H_DECODER_DYN_FIRST && DS41H_DECODER_WOA_PRE), "WOA_PRE assumes CTA b's first job is job b");
constexpr int COUNTERS = C_FIRST + 1;
constexpr bool TAILFREE = DS41H_DECODER_TAILFREE != 0;
static_assert(!TAILFREE || (DS41H_DECODER_PROUTE == 0 && DS41H_DECODER_WARM_ROUTE == 0 && DS41H_DECODER_DYN_FIRST == 0 && DS41H_DECODER_R2F_PF == 0),
              "TAILFREE: not with PROUTE / WARM_ROUTE / DYN_FIRST / R2F_PF (counter accesses outside the epoch-aware paths)");

struct Peers {
  __nv_bfloat16* pub[RANKS];
  unsigned* flags[RANKS];
};

constexpr long PUB_SLOT = long(MAXM) * HID;
constexpr long FLAG_BYTES = 16384;
// [exchange flags: (which, stage, src, chunk)][fused all-reduce #1 slice flags: (src, slice, span)]
constexpr int SLICE_FLAGS = 2 * 2 * RANKS * CHUNKS;
static_assert((SLICE_FLAGS + RANKS * SLICES * OWNA_SPANS) * sizeof(unsigned) <= FLAG_BYTES);

struct Table {
  int expert[ENTRIES];
  int chunk[ENTRIES];
  int routes[EXPERTS * MAXM];
};

struct Scratch {
  unsigned* epoch;
  unsigned* counters;
  Table* table;
  float* zacc;           // [MAXM, OA] wo_a accumulators
  __half* zh;            // [MAXM, OA] f16 bf16(z) * 2^-ze, P8 order
  uint8_t* ze;           // [MAXM, OA/32] biased (+127) exponents of zh
  float* yacc;           // [MAXM, HID] wo_b accumulators
  __nv_bfloat16* yattn;  // [MAXM, HID] reduced attention output
  float* mixp;           // [MAXM, SLICES, MIXW] FFN mix partials (24 mixes, sum r^2, sum x^2)
  __nv_bfloat16* xc;     // [MAXM, HID] collapsed FFN input (pre-norm, bf16)
  float* xsq;            // [MAXM, SLICES]
  __nv_bfloat16* xbf;    // [MAXM, HID] normalized FFN input, natural order (gate, split mode)
  __half* xh;            // [MAXM, HID] normalized FFN input, f16 P8 (shared + routed experts)
  float* gacc;           // [2][MAXM, EXPERTS] router logits (GDET: u64 fixed point), buffer = launch epoch & 1 (the other one is cleared)
  float* gpart;          // GSUM2: [KP_GATE][COARSE_M][EXPERTS] per-K-part partial logits + [COARSE_M][EXPERTS] block sums (glog)
  unsigned long long* xtag;  // TAGX: [MAXE] {expert, epoch} per expert slot (own buffer: never written by other paths)
  float* topk_w;         // [ROUTES]
  int* topk_id;          // [ROUTES] (tests)
  float* cand_key;       // [MAXM, GATE_BLOCKS, TOPK] per-row-block top-6 candidates (hierarchical routing)
  float* cand_score;
  int* cand_exp;
  float* sacc;           // [MAXM, 2 * INTER] shared gate/up accumulators
  __half* hs;            // [MAXM, INTER] shared activation, f16 P8
  float* shacc;          // [MAXM, HID] shared down accumulators
  float* hacc;           // [ROUTES, 2 * INTER] routed gate/up accumulators
  __half* hh;            // [ROUTES, INTER] routed activation, f16 P8
  float* yracc;          // [MAXM, HID] routed down accumulators (weighted)
  // e4m3 copies in wg order for the RS-wgmma paths (written only when the variant is on)
  uint8_t* xq;           // [MAXM, HID] x / xs (FFN input, R1 + shared w1/w3)
  uint8_t* hq;           // [ROUTES, INTER] routed activation * H_INV (R2)
  uint8_t* zq;           // [MAXM, OA] z * 2^-(ze - 127 + 6) (wo_b; ze as for zh)
  uint8_t* hsq;          // [MAXM, INTER] shared activation * H_INV (shared w2)
  // coarse routed path (m <= COARSE_M)
  uint8_t* hqi;          // [MAXE, HQI] h * H_INV of every expert slot, all 8 token columns, wgmma B layout
  float* hcacc;          // [MAXE, TILE, 2 * INTER] R1C K-part partial sums (R1_KP = 2)
  float* wtok;           // [MAXE, TILE] routing weight of (expert slot, token), 0 = not routed (route_all)
  uint8_t* xqi;          // [XT_BYTES] e4m3 x / xs of 8 token columns in the resident-x B-tile layout (gate jobs)
  // lossless coarse path (LC)
  uint8_t* xhi;          // [XHI] f16 x of 8 token columns, [k32][token][32 values in P8 order] (gate jobs; R1F x slices)
  uint8_t* hhi;          // [MAXE, HHI] f16 h of every expert slot, same layout over the 576 columns (R1F -> R2F)
  unsigned* trace;       // trace mode: [TRACE_LAUNCHES][ctas][TRACE_CTA]
};

constexpr int PF_MAX = 6;

struct Weights {
  CUtensorMap tm_w13;    // R1C_TMA: 4D {128 B, 1152 rows, 20 K chunks, E}, box {128, 32, 1, 1}, 128B swizzle
  CUtensorMap tm_s13;    // R1C_TMA: 3D {160 B, 1152 rows, E}, box {16, 32, 1}
  CUtensorMap tm_w2;     // R1C_TMA (R2C): 4D {64 B, 5120 rows, 4 K chunks, E}, box {64, 128, 1, 1}, 64B swizzle
  CUtensorMap tm_w2t;    // R1C_TMA (R2C): 3D over w2 + 256 {32 B, 5120 rows, E}, box {32, 128, 1}, 32B swizzle
  CUtensorMap tm_w13f;   // LC (R1F): 4D {64 B, 1152 rows, 40 K chunks, E}, box {64, 64, 1, 1}, 64B swizzle
  CUtensorMap tm_s13f;   // LC (R1F): 3D {160 B, 1152 rows, E}, box {16, 64, 1} (the scales of 4 stages)
  const uint8_t* woa;    // staged e4m3 [OA/64][KG_WOA][16 frags x 512 B] (P8)
  const uint8_t* woa_s;  // ue8m0 [OA/32, OAK/32]
  const uint8_t* wob;    // staged e4m3 [HID/64][KG_WOB] (P8)
  const uint8_t* wob_s;  // ue8m0 [HID/32, OA/32]
  const uint8_t* gate;   // staged bf16 [EXPERTS/64][KG_GATE] (natural order)
  const float* gate_b;   // [EXPERTS] e_score_correction_bias
  const uint8_t* sh13;   // staged e4m3, 32-row interleaved gate/up [2*INTER/64][KG_SH1] (P8)
  const uint8_t* sh13_s; // ue8m0 [2*INTER/32, SLICES], interleaved like the rows
  const uint8_t* sh2;    // staged e4m3 [HID/64][INTER_STAGES] (P8, K padded to 640)
  const uint8_t* sh2_s;  // ue8m0 [HID/32, SH2_LD]
  const uint8_t* w13;    // raw e2m1 [E, 2*INTER, HID/2]
  const uint8_t* s13;    // ue8m0 [E, 2*INTER, HID/32]
  const uint8_t* w2;     // raw e2m1 [E, HID, INTER/2]
  const uint8_t* s2;     // ue8m0 [E, HID, INTER/32]
  long w13_e, s13_e, w2_e, s2_e;  // expert strides (bytes)
  const float* fn;       // hc_ffn_fn [MIX, HCD]
  const float* hc_scale; // [3]
  const float* hc_base;  // [MIX]
  const __nv_bfloat16* norm;  // ffn_norm.weight [HID]
  float xs;              // e4m3 scale of x (power of two >= max|norm| sqrt(HID) / 448), wgmma paths
  // L2 prefetch (evict_last) of up to PF_MAX byte ranges -- the next layer's prologue weights --
  // spread over the OWNA jobs (latency-bound phase, HBM otherwise idle).
  const uint8_t* pf[PF_MAX];
  long pf_len[PF_MAX];
  long pf_total;
  int pf_phase;  // 0: the pf ranges are prefetched during the OWNA jobs; 1: during the coarse R1 jobs (routed phase)
  // Coarse mode: L2 prefetch (evict_normal, above the evict_first routed stream) of the NEXT layer's
  // latency-critical dense weights (wo_a, wo_b, gate, fn), spread over this layer's R1C jobs (the routed
  // phase streams HBM anyway, the next layer's dense chain then reads L2).
  const uint8_t* dpf[PF_MAX];
  long dpf_len[PF_MAX];
  long dpf_total;
};

struct Params {
  Peers peers;
  Scratch s;
  Weights w;
  int rank;
  int nranks;  // 4, or 1 = solo (local all-reduce, tests)
  int m;
  const __nv_bfloat16* o;  // attention output [m, >=16 heads, 512], head stride 512
  long o_stride;
  const long* positions;
  long max_pos;
  const float* cos_sin;  // [max_pos, 64] fp32: cos[32] | sin[32]
  const __nv_bfloat16* resid0;
  const float* attn_post;
  const float* attn_comb;
  const float* attn_pre;
  __nv_bfloat16* y_ffn;
  __nv_bfloat16* resid1;
  float* ffn_post;
  float* ffn_comb;
  float* ffn_pre;
  int r1kp;  // coarse: K parts per R1 row block (1 = full K, 2; LC also 5)
  int lag;   // coarse: R2C jobs of expert e follow the R1C jobs of expert e + lag
  Scratch ws;  // WARM_ROUTE: copy of s with private routing buffers (counters, table, router sums, candidates, top-k, wtok)
  // XF_P1A (optional, nullptr = off): the next layer's prologue phase P1a computed by the XF jobs (see xf_p1a):
  __nv_bfloat16* p1_resid;  // [m, 4, 5120] bf16: post-mapped residual stream (the next layer's residual)
  __nv_bfloat16* p1_xcb;    // [m, 5120] bf16: collapse with ffn_pre
  float* p1_xsq;            // [m, SLICES] fp32: per 32-column slice square sums of the bf16 collapse
};

struct Smem {
  __align__(1024) uint8_t ring[STAGES][STAGE];
  __align__(128) uint8_t bt[BT];   // f16 / e4m3 activation staging; coarse mode: resident x tile
  __align__(128) uint8_t bt2[LCQ ? HQI : 128]; // lossy coarse mode: R2C / SH2 B tiles (bt holds x)
  __align__(16) float wf[WFT];     // per-warp scale tables (routed wgmma / coarse R1); also holds ws unless ROUTED_WG
  __align__(16) float bs[BS];
  __align__(16) uint8_t ws_sep[WS_SEP ? WS : 16];  // separate R2 scale table (routed_wg users: wf = their tables)
  unsigned long long full[STAGES];
  unsigned long long empty[STAGES];
  Job job[STAGES];
  unsigned seen[CACHED];
  float rs[OTOK];
  int routes[SPAN];
  float cand_key[GW * TOPK];
  float cand_score[GW * TOPK];
  int cand_expert[GW * TOPK];
  int flag;
  int last;
  int ready;
  int xres;  // coarse mode: resident x tile loaded into bt
  int woa_pre;  // WOA_PRE: 1 = wf holds the cos / sin rows of the m tokens (first job of this CTA is wo_a, m <= WOA_SMALL)
  // PROUTE (producer-local routing, coarse m): expert slots (ascending expert ids), per-(slot, token) weights
  int nent;  // expert slots of this launch
  unsigned emask[EXPERTS / 32];
  int ebase[EXPERTS / 32];
  int slot_expert[PROUTE ? MAXE : 1];
  int pf_expert[DS41H_DECODER_PFR1 ? MAXE : 1];  // PFR1: routed expert of each slot (route1_nc)
  int sk_expert[DS41H_DECODER_R1SK ? MAXE : 1];  // R1SK: routed expert of each slot (producer copy of the table)
  int rsel_e[PROUTE || FROUTE || DS41H_DECODER_ROUTE_LEAN || DS41H_DECODER_ROUTE1 ? COARSE_M * TOPK : 1];
  float rsel_w[PROUTE || FROUTE || DS41H_DECODER_ROUTE_LEAN || DS41H_DECODER_ROUTE1 ? COARSE_M * TOPK : 1];
  float wtok_s[PROUTE ? MAXE * TILE : 1];
  int nh;    // coarse mode: R2C jobs run by this CTA (hbar parity)
  unsigned epoch;  // TAILFREE: this launch's epoch (consumer thread 0 -> producer)
  unsigned long long xbar;  // coarse mode: resident x tile bulk copy
  unsigned long long hbar;  // coarse mode: R2C h tile bulk copies
#if DS41H_DECODER_TRACE
  unsigned rt_in, rt_cg, rt_done, rt_ep, rt_sel;  // PROUTE: route_local entry / C_G seen / done / epoch read / top-k done
  unsigned rt_ld, rt_sc;                           // PROUTE: ns in gate-sum loads / in scores (sum over token pairs)
  unsigned long long t0;  // CTA entry (%globaltimer)
  unsigned rec[TRACE_REC];
  int nrec;
#endif
};

constexpr unsigned SPIN_NS = DS41H_DECODER_POLL_NS;
// R2 scale table (non-coarse R2 jobs / routed_wg): aliases wf unless the wgmma routed path needs both at once.
__device__ __forceinline__ uint8_t* ws_of(Smem& sm) { return WS_SEP ? sm.ws_sep : reinterpret_cast<uint8_t*>(sm.wf); }
static_assert(WS_SEP || WS <= WFT * 4, "ws must fit in wf");
static_assert(PER_SM != 2 || sizeof(Smem) <= 115712, "2 CTAs / SM need <= 113 KB smem each (STAGES >= 7: SMEM_DIET=1)");
constexpr bool POLLF = DS41H_DECODER_POLLF != 0;
__device__ __forceinline__ void fence_acquire_gpu() { asm volatile("fence.acquire.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_acquire_sys() { asm volatile("fence.acquire.sys;" ::: "memory"); }

__device__ __forceinline__ void ring_wait(unsigned long long* bar, unsigned parity) {
  if constexpr (DS41H_DECODER_MBAR_SLEEP > 0) {
    asm volatile(
        "{\n .reg .pred p;\n LAB_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n @!p bra LAB_WAIT;\n}" ::"r"(smem_u32(bar)),
        "r"(parity), "r"(unsigned(DS41H_DECODER_MBAR_SLEEP))
        : "memory");
  } else {
    mbar_wait(bar, parity);
  }
}

// Polls: an ld.acquire is LDG.STRONG + CCTL.IVALL (invalidates the SM's whole L1 every time), so waiting loops
// poll with relaxed loads and acquire once when the condition holds (the first check is an acquire: no CCTL
// storm while 264 CTAs wait, and the common already-satisfied case costs one load).
__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long ld_relaxed_gpu64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_relaxed_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// Wait until *p >= target (gpu scope); returns the value read with acquire semantics.
__device__ __forceinline__ unsigned poll_ge(const unsigned* p, unsigned target) {
  if constexpr (POLLF) {  // relaxed polls, one acquire fence (no extra round trips)
    unsigned v = ld_relaxed_gpu(p);
    while (v < target) {
      if (SPIN_NS) __nanosleep(SPIN_NS);
      v = ld_relaxed_gpu(p);
    }
    fence_acquire_gpu();
    return v;
  }
  unsigned v = ld_acquire_gpu(p);
  if (v >= target) return v;
  do {
    __nanosleep(SPIN_NS);
  } while (ld_relaxed_gpu(p) < target);
  return ld_acquire_gpu(p);
}

// The kernels' dynamic smem (layer_kernel / routed_bench_kernel place their Smem at its start).
// Addressing bt through the symbol (instead of the Ctx reference) gives ptxas a link-time
// constant, so wgmma descriptors can live in uniform registers.
extern __shared__ __align__(1024) uint8_t ds41h_dsmem[];
__device__ __forceinline__ uint32_t bt_smem_addr() { return smem_u32(ds41h_dsmem) + uint32_t(offsetof(Smem, bt)); }
__device__ __forceinline__ uint32_t bt2_smem_addr() { return smem_u32(ds41h_dsmem) + uint32_t(offsetof(Smem, bt2)); }

// ------------------------------------------------------------------------------ small helpers
__device__ __forceinline__ float bf(float x) { return bf16_round(x); }

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// SiluAndMulWithClamp (stock) / Expert.forward (reference): silu(min(g, 10)) * clamp(u, +-10).
__device__ __forceinline__ float swiglu(float gate, float up) {
  gate = fminf(gate, LIMIT);
  up = fminf(fmaxf(up, -LIMIT), LIMIT);
  return gate / (1.0f + expf(-gate)) * up;
}

__device__ __forceinline__ float f16_clamp(float x) { return fminf(fmaxf(x, -65504.0f), 65504.0f); }

__device__ __forceinline__ unsigned pack_h(float lo, float hi) { return pack_f16(f16_clamp(lo), f16_clamp(hi)); }

// 8 consecutive values (one P8 group) -> f16 in P8 order [0,2,4,6,1,3,5,7].
__device__ __forceinline__ uint4 pack8_p8(const float (&f)[8]) {
  return make_uint4(pack_h(f[0], f[2]), pack_h(f[4], f[6]), pack_h(f[1], f[3]), pack_h(f[5], f[7]));
}

// Position of element c (0..7) of an 8-group in P8 order.
__device__ __forceinline__ int p8_pos(int c) { return (c >> 1) + (c & 1) * 4; }

__device__ __forceinline__ unsigned bf16x2_f16x2(unsigned v) {
  unsigned h;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(f16_clamp(__uint_as_float(v & 0xffff0000u))), "f"(f16_clamp(__uint_as_float(v << 16))));
  return h;
}

// bf16 x8 (natural order) -> f16 x8 in P8 order.
__device__ __forceinline__ uint4 bf16x8_f16_p8(uint4 v) {
  return make_uint4(bf16x2_f16x2(__byte_perm(v.x, v.y, 0x5410)), bf16x2_f16x2(__byte_perm(v.z, v.w, 0x5410)), bf16x2_f16x2(__byte_perm(v.x, v.y, 0x7632)),
                    bf16x2_f16x2(__byte_perm(v.z, v.w, 0x7632)));
}

// Per-32 f16 staging exponent for 8 values per lane in a quad: amax * 2^-e lies in [2^13, 2^14).
__device__ __forceinline__ int f16_block_exp(const float (&x)[8]) {
  float amax = 0.0f;
#pragma unroll
  for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(x[e]));
  amax = quad_max(amax);
  const int be = int((__float_as_uint(amax) >> 23) & 0xffu);
  const int e = be == 0 ? -100 : be - 127 - 13;
  return min(max(e, -100), 100);
}

// Transposing butterfly: afterwards lane l holds the warp-wide sum of v[l]. The per-step width
// is a template argument so every index is static and v stays in registers.
template <int W>
__device__ __forceinline__ void lane_step(float (&v)[32], int lane) {
  const bool hi = lane & W;
#pragma unroll
  for (int i = 0; i < W; ++i) {
    const float keep = hi ? v[i + W] : v[i], send = hi ? v[i] : v[i + W];
    v[i] = keep + __shfl_xor_sync(0xffffffffu, send, W);
  }
}

__device__ __forceinline__ float lane_sums(float (&v)[32], int lane) {
  lane_step<16>(v, lane);
  lane_step<8>(v, lane);
  lane_step<4>(v, lane);
  lane_step<2>(v, lane);
  lane_step<1>(v, lane);
  return v[0];
}

// ------------------------------------------------------------------------------ MMA helpers
__device__ __forceinline__ void hmma16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void bmma16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned cvt_lo(unsigned v) { return e4m3x2_to_f16x2(static_cast<unsigned short>(v & 0xffffu)); }
__device__ __forceinline__ unsigned cvt_hi(unsigned v) { return e4m3x2_to_f16x2(static_cast<unsigned short>(v >> 16)); }

// One 32-wide K block of a 16x8 tile: two HMMA m16n8k16 into a fresh accumulator.
// v.x / v.y: row g e4m3 bytes at P8 positions 0-3 / 4-7 (k = 8t + {0,2,4,6} / {1,3,5,7});
// v.z / v.w: the same for row g + 8. b: token g's 8 f16 activations at the same k, P8 order.
// HMMA slots: first MMA k-slots (2t, 2t+1, 2t+8, 2t+9) = k (8t, 8t+2, 8t+4, 8t+6), second MMA
// = k (8t+1, 8t+3, 8t+5, 8t+7); any in-block permutation is exact if A and B agree.
__device__ __forceinline__ void mma32(float (&d)[4], uint4 v, uint4 b) {
  d[0] = d[1] = d[2] = d[3] = 0.0f;
  hmma16(d, cvt_lo(v.x), cvt_lo(v.z), cvt_hi(v.x), cvt_hi(v.z), b.x, b.y);
  hmma16(d, cvt_lo(v.y), cvt_lo(v.w), cvt_hi(v.y), cvt_hi(v.w), b.z, b.w);
}

// Raw e2m1 words of rows g / g+8 (8 nibbles, low nibble = even k) -> e4m3 x2^-6 in P8 order.
__device__ __forceinline__ uint4 fp4_e4m3(unsigned wg, unsigned wg8) { return make_uint4(fp4_lo_e4m3(wg), fp4_hi_e4m3(wg), fp4_lo_e4m3(wg8), fp4_hi_e4m3(wg8)); }

__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ uint4 ldsm4_u32(uint32_t addr) {
  uint4 r;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(addr) : "memory");
  return r;
}
__device__ __forceinline__ uint4 ldsm4(const void* addr) {
  uint4 r;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(smem_u32(addr)) : "memory");
  return r;
}

// FP4 stage tiles are row-major (64 B or 32 B per row) with the 16-byte chunk index XOR-swizzled
// by row bits, so an 8-row ldmatrix matrix hits 8 distinct 16-byte bank groups.
__device__ __forceinline__ int fp4_off64(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }
__device__ __forceinline__ int fp4_off32(int row, int chunk) { return row * 32 + ((chunk ^ ((row >> 2) & 1)) << 4); }

// ldmatrix.x4 for one 16-row m-tile at r0: returns (row g, K block q), (row g+8, q), (g, q+1),
// (g+8, q+1) words (thread (g, t) gets 32-bit word t of its row, i.e. k = 8t..8t+7).
__device__ __forceinline__ uint4 fp4_words64(const uint8_t* stage, int r0, int q, int lane) {
  const int j = lane >> 3, row = r0 + (lane & 7) + (j & 1) * 8;
  return ldsm4(stage + fp4_off64(row, q + (j >> 1)));
}
__device__ __forceinline__ uint4 fp4_words32(const uint8_t* stage, int r0, int lane) {
  const int j = lane >> 3, row = r0 + (lane & 7) + (j & 1) * 8;
  return ldsm4(stage + fp4_off32(row, j >> 1));
}

// ue8m0 byte q of a packed word -> 2^(byte - 127) as float (0 for byte 0); q is compile-time.
__device__ __forceinline__ float ue8m0_byte(unsigned word, int q) {
  return __uint_as_float((q < 3 ? word << (23 - 8 * q) : word >> 1) & 0x7f800000u);
}

// ------------------------------------------------------------------------------ RS-wgmma helpers
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
// Generic-proxy smem writes (st.shared, cp.async) -> visible to wgmma operand reads (async proxy).
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(unsigned long long* bar, uint32_t bytes);
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, unsigned long long* bar, uint64_t pol);
// Compiler-only register fences: keep accumulator reads after wgmma.wait_group and A registers
// alive (unmodified) until the wgmma that reads them has retired.
__device__ __forceinline__ void rfence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void rfence(unsigned& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void rfence(uint64_t& r) { asm volatile("" : "+l"(r)::"memory"); }

// D[64 x 8NT] = A[64 x 32] (e4m3, registers) * B[8NT x 32]^T (e4m3, smem descriptor). scale-d = 0
// (every k32 block gets a fresh accumulator set); "+f" keeps the promote reads of the previous
// unit in this buffer ordered before the next wgmma.
template <int NT>
__device__ __forceinline__ void wgmma_rs(float (&d)[4 * NT], const unsigned (&a)[4], uint64_t desc);

template <>
__device__ __forceinline__ void wgmma_rs<1>(float (&d)[4], const unsigned (&a)[4], uint64_t desc) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

template <>
__device__ __forceinline__ void wgmma_rs<2>(float (&d)[8], const unsigned (&a)[4], uint64_t desc) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}

// B tile descriptor: K-major, no swizzle; core matrix (16 B K chunk c, 8-token group n) at
// (c * NT + n) * 128 with token rows 16 B apart; LBO (next K chunk) = NT * 128, SBO (next token
// group) = 128. The k32 block kb starts at kb * NT * 256 B: add kb * NT * 16 to the descriptor.
__device__ __forceinline__ uint64_t desc_b(uint32_t smem_addr, int nt) {
  const uint64_t a = smem_addr;
  return ((a & 0x3FFFFull) >> 4) | (uint64_t((nt * 128) >> 4) << 16) | (uint64_t(128 >> 4) << 32);
}
__device__ __forceinline__ int bt_wg_off(int tok, int chunk, int nt) { return (chunk * nt + (tok >> 3)) * 128 + (tok & 7) * 16; }

// wg order inside a 32-wide K block: physical k = 8t + j -> position 16 (j & 1) + 4t + (j >> 1).
__host__ __device__ __forceinline__ int wg_pos(int k) {
  const int t = (k >> 3) & 3, j = k & 7;
  return (k & ~31) | ((j & 1) << 4) | (t << 2) | (j >> 1);
}

// 8 consecutive values of one 8-group (k = 8t..8t+7) times s -> the two wg-order e4m3 words:
// x = elements 0,2,4,6 (position 4t), y = elements 1,3,5,7 (position 16 + 4t).
__device__ __forceinline__ uint2 e4m3_wg8(const float (&x)[8], float s) {
  return make_uint2(e4m3x2(x[0] * s, x[2] * s) | (e4m3x2(x[4] * s, x[6] * s) << 16), e4m3x2(x[1] * s, x[3] * s) | (e4m3x2(x[5] * s, x[7] * s) << 16));
}
// Store them into an e4m3 row (wg order) for the 8-group starting at element k.
__device__ __forceinline__ void store_wg8(uint8_t* row, int k, uint2 v) {
  uint8_t* base = row + (k & ~31) + ((k >> 3) & 3) * 4;
  *reinterpret_cast<unsigned*>(base) = v.x;
  *reinterpret_cast<unsigned*>(base + 16) = v.y;
}
__device__ __forceinline__ uint8_t e4m3_byte(float x) { return uint8_t(e4m3x2(x, 0.0f) & 0xffu); }
// 1 / xs, correctly rounded (exact for the power-of-two xs): under --use_fast_math `1.0f / xs`
// is MUFU.RCP, which is not guaranteed exact, and x / xs feeds an RNE e4m3 rounding whose ties
// (bf16 x has 8 significant bits) the reference (x / xs in fp32) resolves exactly.
__device__ __forceinline__ float xs_inv(const Weights& w) { return __frcp_rn(w.xs); }

// FP4 words of rows g / g+8 (k = 8t..8t+7 of one k32 block) -> wgmma A fragment (e4m3 x 2^-6).
__device__ __forceinline__ void fp4_frag(unsigned wg, unsigned wg8, unsigned (&a)[4]) {
  a[0] = fp4_lo_e4m3(wg);
  a[1] = fp4_lo_e4m3(wg8);
  a[2] = fp4_hi_e4m3(wg);
  a[3] = fp4_hi_e4m3(wg8);
}

// Issue one commit group: NK k32 wgmmas of buffer (a, acc) starting at B-tile block kb0. The A
// fragments, accumulators and descriptors are pinned before wgmma.fence so that no non-wgmma
// instruction defines a wgmma input inside the fence..commit stage (ptxas C7513 / C7514).
template <int NT, int NK, int NKU>
__device__ __forceinline__ void wg_group(unsigned (&a)[NKU][4], float (&acc)[NKU][4 * NT], uint64_t desc, int kb0) {
  uint64_t d[NK];
#pragma unroll
  for (int q = 0; q < NK; ++q) {
    // 32-bit add on the start-address word (never carries: smem addresses < 256 KB)
    d[q] = (desc & 0xFFFFFFFF00000000ull) | uint32_t(uint32_t(desc) + uint32_t((kb0 + q) * NT * 16));
#ifndef DS41H_WG_NODPIN
    rfence(d[q]);
#endif
#pragma unroll
    for (int i = 0; i < 4; ++i) rfence(a[q][i]);
#pragma unroll
    for (int e = 0; e < 4 * NT; ++e) rfence(acc[q][e]);
  }
  wg_fence();
#pragma unroll
  for (int q = 0; q < NK; ++q) wgmma_rs<NT>(acc[q], a[q], d[q]);
  wg_commit();
}

// After the wgmma.wait_group that retired a group: accumulator reads stay below the wait and the
// A fragments stay unmodified up to it.
template <int NT, int NK, int NKU>
__device__ __forceinline__ void wg_retired(unsigned (&a)[NKU][4], float (&acc)[NKU][4 * NT]) {
#pragma unroll
  for (int q = 0; q < NK; ++q) {
#pragma unroll
    for (int e = 0; e < 4 * NT; ++e) rfence(acc[q][e]);
#pragma unroll
    for (int i = 0; i < 4; ++i) rfence(a[q][i]);
  }
}

template <int V>
using IC = std::integral_constant<int, V>;

// Two-buffer RS-wgmma software pipeline over Plan::U units, fully unrolled (ptxas keeps a
// wgmma.wait_group 1 pipeline asynchronous only in straight-line code; a loop back-edge with a
// group in flight serializes every wgmma, C7514). Unit u loads its A fragments into register
// buffer u & 1, is issued as one commit group of Plan::nk(u) k32 wgmmas, and is retired
// (promoted into the fp32 totals) after unit u + 1 has been issued. The callbacks receive
// IC<u>, so every index derived from u is a compile-time constant.
template <class Plan, int u = 0>
struct WgPipe {
  template <class Load, class Issue, class Retire>
  static __device__ __forceinline__ void run(Load& load, Issue& issue, Retire& retire) {
    load(IC<u>{});
    issue(IC<u>{});
    if constexpr (u > 0) {
      wg_wait<1>();
      retire(IC<u - 1>{});
    }
    if constexpr (u + 1 < Plan::U) {
      WgPipe<Plan, u + 1>::run(load, issue, retire);
    } else {
      wg_wait<0>();
      retire(IC<u>{});
    }
  }
};

template <class Plan, class Load, class Issue, class Retire>
__device__ __forceinline__ void wg_pipe(Load&& load, Issue&& issue, Retire&& retire) {
  static_assert(Plan::U >= 2);
  WgPipe<Plan>::run(load, issue, retire);
}

// Unit plan of a routed job (ROUTED_MMA): unit = (stage, 64-row m-tile, k32 blocks q0..q0+nk-1 of
// the stage). nk = 4 for 8 tokens, 2 for 16 (bounds the in-flight accumulators to 32 registers).
// R2 stage 0 (K tail, blocks 16-17) is one 2-block unit per m-tile.
template <bool R2, int NT>
struct RoutedPlan {
  static constexpr int NKU = NT == 1 ? 4 : 2;
  static constexpr int PER = 2 * (4 / NKU);  // units per full stage
  static constexpr int HEAD = R2 ? 2 : 0;
  static constexpr int U = HEAD + (R2 ? INTER_STAGES - 1 : KS_R1 / 4) * PER;
  static constexpr int stage(int u) { return u < HEAD ? 0 : (u - HEAD) / PER + (R2 ? 1 : 0); }
  static constexpr int mt(int u) { return u < HEAD ? u : (u - HEAD) % PER / (4 / NKU); }
  static constexpr int q0(int u) { return u < HEAD ? 0 : (u - HEAD) % (4 / NKU) * NKU; }
  static constexpr int nk(int u) { return u < HEAD ? 2 : NKU; }
  static constexpr bool first(int u) { return u == 0 || stage(u) != stage(u - 1); }
  static constexpr bool last(int u) { return u == U - 1 || stage(u) != stage(u + 1); }
  // first k32 block of the unit in the job's B tile
  static constexpr int kb(int u) { return (R2 ? (stage(u) == 0 ? 16 : 4 * (stage(u) - 1)) : 4 * stage(u)) + q0(u); }
};

// Unit plan of a dense job (DENSE_MMA): stage = 64 rows x 4 k32 blocks; KS = job k32 blocks.
template <int KS, int NT>
struct DensePlan {
  static constexpr int NKU = NT == 1 ? 4 : 2;
  static constexpr int PER = 4 / NKU;
  static constexpr int U = cdiv(KS, NKU);
  static constexpr int stage(int u) { return u / PER; }
  static constexpr int q0(int u) { return u % PER * NKU; }
  static constexpr int nk(int u) { return KS - u * NKU < NKU ? KS - u * NKU : NKU; }
  static constexpr bool first(int u) { return u % PER == 0; }
  static constexpr bool last(int u) { return u == U - 1 || u % PER == PER - 1; }
  static constexpr int kb(int u) { return u * NKU; }
};

// ------------------------------------------------------------------------------ context
struct Ctx {
  const Params& p;
  Smem& sm;
  int warp;
  int lane;
  int tid;
  int it;
  unsigned epoch;
  int sgc = 0;   // LC: R1F scale groups consumed (slot sgc % 3 of the scale ring, see issue_r1f)
  int lazy = 0;  // LAZY_SIG: pending C_SF arrival (chunk + 1) of the previous R2F job, 0 = none

  __device__ __forceinline__ void sync() const { asm volatile("bar.sync 1, %0;" ::"n"(GT) : "memory"); }
  // this launch's dependency counters (TAILFREE: the set of the epoch parity)
  __device__ __forceinline__ unsigned* cs() const { return TAILFREE ? p.s.counters + (epoch & 1u) * COUNTERS : p.s.counters; }

  // Trace helpers (thread 0 owns the CTA's current record; no-ops unless TRACE).
  __device__ __forceinline__ unsigned tnow() const {
#if DS41H_DECODER_TRACE
    return unsigned(gtime() - sm.t0);
#else
    return 0u;
#endif
  }
  __device__ __forceinline__ void tset(int w, unsigned v) {
#if DS41H_DECODER_TRACE
    if (tid == 0) sm.rec[w] = v;
#endif
  }
  __device__ __forceinline__ void tmark(int w) {
    if constexpr (TRACE) tset(w, tnow());
  }
  [[maybe_unused]] __device__ __forceinline__ void tflag(unsigned f) {
#if DS41H_DECODER_TRACE
    if (tid == 0) sm.rec[TR_FLAGS] |= f;
#endif
  }

  __device__ CTXFN void wait(int index, unsigned target) {
    if (tid == 0 && (index >= CACHED || sm.seen[index] < target)) {
      unsigned seen = POLLF ? ld_relaxed_gpu(cs() + index) : ld_acquire_gpu(cs() + index);
#if DS41H_DECODER_TRACE
      const unsigned t = seen < target ? tnow() : 0u;
#endif
      if (POLLF) {
        while (seen < target) {
          if (SPIN_NS) __nanosleep(SPIN_NS);
          seen = ld_relaxed_gpu(cs() + index);
        }
        fence_acquire_gpu();
      } else if (seen < target) {
        seen = poll_ge(cs() + index, target);
      }
#if DS41H_DECODER_TRACE
      if (t) sm.rec[TR_SPIN] += tnow() - t, sm.rec[TR_SPINS] += 1;
#endif
      if (index < CACHED) sm.seen[index] = seen;
    }
    sync();
  }

  // n counters (index0 + i * stride) polled in parallel by threads 0..n-1 (one round trip, not n).
  __device__ CTXFN void wait_n(int index0, int stride, int n, unsigned target) {
    if (tid < n) {
      const int index = index0 + tid * stride;
      if (index >= CACHED || sm.seen[index] < target) {
        const unsigned seen = poll_ge(cs() + index, target);
        if (index < CACHED) sm.seen[index] = seen;
      }
    }
    sync();
  }

  __device__ CTXFN void signal(int index) {
    sync();
    if (tid == 0) red_release(cs() + index, 1u);
  }

  __device__ CTXFN bool arrive(int index, unsigned target) {
    sync();
    if (tid == 0) {
      sm.flag = atom_add_acq_rel(cs() + index, 1u) == target - 1;
#if DS41H_DECODER_TRACE
      if (sm.flag) sm.rec[TR_FLAGS] |= TF_FINISH;
#endif
    }
    sync();
    return sm.flag;
  }

  __device__ __forceinline__ const uint8_t* stage() const { return sm.ring[it % STAGES]; }

  __device__ __forceinline__ void full(int k) const {
#if DS41H_DECODER_TRACE
    if (tid == 0) {
      const unsigned t = tnow();
      ring_wait(&sm.full[k % STAGES], (k / STAGES) & 1);
      sm.rec[TR_SWAIT] += tnow() - t;
      return;
    }
#endif
    ring_wait(&sm.full[k % STAGES], (k / STAGES) & 1);
  }

  __device__ __forceinline__ void acquire() const { full(it); }

  __device__ __forceinline__ void release() {
    if constexpr (PXF) fence_proxy_async();  // generic stage reads before the producer's async-proxy refill (see PXF)
    __syncwarp();
    if (lane == 0) mbar_arrive(&sm.empty[it % STAGES]);
    ++it;
  }

  __device__ __forceinline__ void next() {
    release();
    acquire();
  }

  // REL_DEP: release returning a value (0.0f) that depends on lane 0's arrive (state token AND a runtime zero).
  __device__ __forceinline__ float release_dep() {
    if constexpr (PXF) fence_proxy_async();
    __syncwarp();
    unsigned lo = 0;
    if (lane == 0) {
      unsigned long long st;
      asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(st) : "r"(smem_u32(&sm.empty[it % STAGES])) : "memory");
      lo = unsigned(st);
    }
    ++it;
    unsigned z;
    asm("and.b32 %0, %1, %2;" : "=r"(z) : "r"(lo), "r"(unsigned(p.nranks) >> 16));  // nranks <= 4: always 0 at run time
    return __uint_as_float(z);
  }

  __device__ __forceinline__ const uint8_t* held(int s) const {
    full(it + s);
    return sm.ring[(it + s) % STAGES];
  }

  __device__ __forceinline__ void drop(int stages) {
    for (int s = 0; s < stages; ++s) release();
  }
};

struct Acc8 {
  float4 lo, hi;
};

__device__ __forceinline__ Acc8 load8(const float* src, bool live) {
  const float4* a = reinterpret_cast<const float4*>(src);
  const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  return {live ? __ldcg(a) : zero, live ? __ldcg(a + 1) : zero};
}

__device__ __forceinline__ void unpack8(const Acc8& a, float (&f)[8]) {
  f[0] = a.lo.x, f[1] = a.lo.y, f[2] = a.lo.z, f[3] = a.lo.w, f[4] = a.hi.x, f[5] = a.hi.y, f[6] = a.hi.z, f[7] = a.hi.w;
}

__device__ __forceinline__ void clear8(float* dst) {
  float4* a = reinterpret_cast<float4*>(dst);
  const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  __stcg(a, zero);
  __stcg(a + 1, zero);
}

// ------------------------------------------------------------------------------ all-reduce
// Pull-based chunked all-reduce over CUDA IPC (BRRR): every rank publishes its bf16 partial in
// its own inbox and raises epoch flags in the peers' flag arrays; readers pull with ld.cv.
__device__ __forceinline__ long chunk_off(int i, int k) { return long(i / CHUNK_VECS) * HID + k * CHUNK_COLS + (i % CHUNK_VECS) * 8; }

__device__ __forceinline__ __nv_bfloat16* inbox(const Params& p, int r, int which, int stage, unsigned epoch) {
  return p.peers.pub[r] + ((long(which) * 2 + stage) * 2 + (epoch & 1)) * PUB_SLOT;
}

__device__ __forceinline__ int flag_index(int which, int stage, int src, int g) { return ((which * 2 + stage) * RANKS + src) * CHUNKS + g; }

// LLP inboxes (after the 8 pub slots of each rank's buffer): [which 2][parity 2][src RANKS][token COARSE_M][HID / 2] 8-byte words
// {bf16 pair (low = even column), epoch}. One aligned 8-byte store / load = one transaction (no tearing).
constexpr bool LLP = DS41H_DECODER_LLP != 0;
constexpr long LL_SRC = long(COARSE_M) * (HID / 2);  // words per (which, parity, src)
__device__ __forceinline__ uint2* ll_box(const Params& p, int r, int which, unsigned epoch, int src) {
  return reinterpret_cast<uint2*>(p.peers.pub[r] + 8 * PUB_SLOT) + ((long(which) * 2 + (epoch & 1u)) * RANKS + src) * LL_SRC;
}
__device__ __forceinline__ void ll_put(uint2* dst, unsigned data, unsigned epoch) {
  asm volatile("st.relaxed.sys.global.v2.u32 [%0], {%1, %2};" ::"l"(dst), "r"(data), "r"(epoch) : "memory");
}
__device__ __forceinline__ uint2 ll_get(const uint2* src) {
  uint2 v;
  asm volatile("ld.relaxed.sys.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(src) : "memory");
  return v;
}
// Poll n words until every one carries `epoch` (in place: w[i].x = data).
template <int N>
__device__ __forceinline__ void ll_wait(uint2 (&w)[N], const uint2* const (&src)[N], unsigned epoch) {
#pragma unroll
  for (int i = 0; i < N; ++i) w[i] = ll_get(src[i]);
  for (;;) {
    bool done = true;
#pragma unroll
    for (int i = 0; i < N; ++i) done &= w[i].y == epoch;
    if (done) return;
    __nanosleep(32);
#pragma unroll
    for (int i = 0; i < N; ++i)
      if (w[i].y != epoch) w[i] = ll_get(src[i]);
  }
}

__device__ __forceinline__ unsigned peer_mask(const Params& p) { return p.nranks == 1 ? 0u : ((1u << RANKS) - 1) & ~(1u << p.rank); }

__device__ CTXFN void post(Ctx& c, int which, int stage, int g, unsigned target, unsigned epoch, unsigned mask) {
  if (target > 1) {
    if (!c.arrive(C_PUB + (which * 2 + stage) * CHUNKS + g, target)) return;
  } else {
    c.sync();
  }
  if (c.tid == 0 && mask) {
    __threadfence_system();
    for (int r = 0; r < RANKS; ++r)
      if (mask >> r & 1) st_relaxed_sys(c.p.peers.flags[r] + flag_index(which, stage, c.p.rank, g), epoch);
  }
}

__device__ CTXFN void await(Ctx& c, int which, int stage, int g, unsigned epoch, unsigned mask) {
  if (c.tid < RANKS && (mask >> c.tid & 1)) {
    const unsigned* flag = c.p.peers.flags[c.p.rank] + flag_index(which, stage, c.tid, g);
    if constexpr (POLLF) {
      while (before(ld_relaxed_sys(flag), epoch))
        if (SPIN_NS) __nanosleep(SPIN_NS);
      fence_acquire_sys();
    } else if (before(ld_acquire_sys(flag), epoch)) {
      do {
        __nanosleep(SPIN_NS);
      } while (before(ld_relaxed_sys(flag), epoch));
      (void)ld_acquire_sys(flag);
    }
  }
  c.sync();
}

__device__ __forceinline__ int slice_flag(int src, int s, int span) { return SLICE_FLAGS + (src * SLICES + s) * OWNA_SPANS + span; }

__device__ CTXFN void post_slice(Ctx& c, int s, int span, unsigned epoch, unsigned mask) {
  c.sync();
  if (c.tid == 0 && mask) {
    __threadfence_system();
    for (int r = 0; r < RANKS; ++r)
      if (mask >> r & 1) st_relaxed_sys(c.p.peers.flags[r] + slice_flag(c.p.rank, s, span), epoch);
  }
}

__device__ CTXFN void await_slice(Ctx& c, int s, int span, unsigned epoch, unsigned mask) {
  if (c.tid < RANKS && (mask >> c.tid & 1)) {
    const unsigned* flag = c.p.peers.flags[c.p.rank] + slice_flag(c.tid, s, span);
    if constexpr (POLLF) {
      while (before(ld_relaxed_sys(flag), epoch))
        if (SPIN_NS) __nanosleep(SPIN_NS);
      fence_acquire_sys();
    } else if (before(ld_acquire_sys(flag), epoch)) {
      do {
        __nanosleep(SPIN_NS);
      } while (before(ld_relaxed_sys(flag), epoch));
      (void)ld_acquire_sys(flag);
    }
  }
  c.sync();
}

// Vectors per thread per pass: m <= 16 tokens (256 vectors per chunk) needs one pass with 2.
constexpr int XB = 2;

__device__ CTXFN void reduce4(Ctx& c, int which, int k, unsigned epoch, int nv, __nv_bfloat16* out, __nv_bfloat16* keep) {
  const Params& p = c.p;
  const unsigned live = peer_mask(p) | (1u << p.rank);
  const __nv_bfloat16* src[RANKS];
#pragma unroll
  for (int r = 0; r < RANKS; ++r) src[r] = (live >> r & 1) ? inbox(p, r, which, 0, epoch) : nullptr;
  for (int base = 0; base < nv; base += GT * XB) {
    uint4 v[XB][RANKS];
#pragma unroll
    for (int u = 0; u < XB; ++u) {
      const int i = base + c.tid + u * GT;
#pragma unroll
      for (int r = 0; r < RANKS; ++r)
        v[u][r] = i >= nv || !(live >> r & 1) ? make_uint4(0, 0, 0, 0)
                  : r == p.rank                ? __ldcg(reinterpret_cast<const uint4*>(src[r] + chunk_off(i, k)))
                                               : __ldcv(reinterpret_cast<const uint4*>(src[r] + chunk_off(i, k)));
    }
#pragma unroll
    for (int u = 0; u < XB; ++u) {
      const int i = base + c.tid + u * GT;
      if (i >= nv) continue;
      float acc[8] = {};
#pragma unroll
      for (int r = 0; r < RANKS; ++r) add8_bf16(acc, v[u][r]);
      const uint4 sum = pack8_bf16(acc);
      *reinterpret_cast<uint4*>(out + chunk_off(i, k)) = sum;
      if (keep) *reinterpret_cast<uint4*>(keep + chunk_off(i, k)) = sum;
    }
  }
}

__device__ CTXFN void gather(Ctx& c, const __nv_bfloat16* src, int k, int nv, __nv_bfloat16* out) {
  constexpr int GB = 2 * XB;
  for (int base = 0; base < nv; base += GT * GB) {
    uint4 v[GB];
#pragma unroll
    for (int u = 0; u < GB; ++u) {
      const int i = base + c.tid + u * GT;
      v[u] = i < nv ? __ldcv(reinterpret_cast<const uint4*>(src + chunk_off(i, k))) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < GB; ++u) {
      const int i = base + c.tid + u * GT;
      if (i < nv) *reinterpret_cast<uint4*>(out + chunk_off(i, k)) = v[u];
    }
  }
}

// Reduce chunk k (128 columns x m tokens) of exchange `which` (0: attention, 1: FFN) into out.
__device__ CTXFN void share(Ctx& c, int which, int k, __nv_bfloat16* out) {
  const Params& p = c.p;
  const unsigned epoch = c.epoch;
  const int nv = p.m * CHUNK_VECS;
  const unsigned peers = peer_mask(p);
  const bool one_shot = p.m < TWO_SHOT_MIN || p.nranks == 1;
  // The attention exchange raises one flag after all 40 chunks (it is fully overlapped by the
  // FFN prologue); the FFN exchange flags every chunk so the tail overlaps the routed experts.
  const int per = which == 0 && one_shot ? CHUNKS : 1, g = k / per;
  if (one_shot) {
    post(c, which, 0, g, per, epoch, peers);
    c.tmark(TR_X0);
    await(c, which, 0, g, epoch, peers);
    c.tmark(TR_X1);
    reduce4(c, which, k, epoch, nv, out, nullptr);
    c.tmark(TR_X2);
    return;
  }
  const int owner = k % RANKS;
  post(c, which, 0, g, per, epoch, per == 1 ? 1u << owner : peers);
  if (p.rank != owner) {
    await(c, which, 1, g, epoch, 1u << owner);
    gather(c, inbox(p, owner, which, 1, epoch), k, nv, out);
    return;
  }
  await(c, which, 0, g, epoch, peers);
  reduce4(c, which, k, epoch, nv, out, inbox(p, p.rank, which, 1, epoch));
  post(c, which, 1, g, per == 1 ? 1 : per / RANKS, epoch, peers);
}

// ------------------------------------------------------------------------------ mHC mixes
// Stock mhc_pre_big_fuse_with_norm_tilelang (tilelang_kernels.py:376-427): mix = gemm * rsqrt(
// sumsq / (4*5120) + rms_eps); pre = sigmoid(m*s0 + b) + eps; post = 2 sigmoid(m*s1 + b);
// comb = row softmax + eps, column normalize, then 19 x (row, column) normalizations.
__device__ void mix_finish(int lane, int tok, float mine, const float* scale, const float* base, float* pre, float* post, float* comb) {
  const float sq = __shfl_sync(0xffffffffu, mine, MIX);
  const float rs = rsqrtf(sq / HCD + NORM_EPS);
  const float mix = mine * rs;
  if (lane < HC) {
    pre[tok * HC + lane] = sigmoid(mix * __ldg(scale) + __ldg(base + lane)) + HC_EPS;
  } else if (lane < 2 * HC) {
    post[tok * HC + lane - HC] = 2.0f * sigmoid(mix * __ldg(scale + 1) + __ldg(base + lane));
  }
  const int idx = 2 * HC + (lane & 15);
  float cm = __shfl_sync(0xffffffffu, mix, idx) * __ldg(scale + 2) + __ldg(base + idx);
  cm = expf(cm - quad_max(cm));
  cm = cm / quad_sum(cm) + HC_EPS;
  for (int it = 0; it < SINKHORN; ++it) {
    if (it > 0) cm /= quad_sum(cm) + HC_EPS;
    float col = cm + __shfl_xor_sync(0xffffffffu, cm, 4);
    col += __shfl_xor_sync(0xffffffffu, col, 8);
    cm /= col + HC_EPS;
  }
  if (lane < HC * HC) comb[tok * HC * HC + lane] = cm;
}

// ------------------------------------------------------------------------------ scatter
__device__ __forceinline__ void red4(float* dst, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

// m16n8 C fragment (c0=(g,2t) c1=(g,2t+1) c2=(g+8,2t) c3=(g+8,2t+1)) -> 4 consecutive rows of
// one token per lane: rows quad_row(lane)..+3 of column quad_col(lane).
__device__ __forceinline__ float4 transpose4(const float* a, int lane) {
  const int i = (lane >> 2) & 3;
  const bool p = i & 1, q = i & 2;
  const float r0 = __shfl_xor_sync(0xffffffffu, p ? a[0] : a[1], 4), r1 = __shfl_xor_sync(0xffffffffu, p ? a[2] : a[3], 4);
  const float b0 = p ? r0 : a[0], b1 = p ? a[1] : r0, b2 = p ? r1 : a[2], b3 = p ? a[3] : r1;
  const float w0 = __shfl_xor_sync(0xffffffffu, q ? b0 : b2, 8), w1 = __shfl_xor_sync(0xffffffffu, q ? b1 : b3, 8);
  return q ? make_float4(w0, w1, b2, b3) : make_float4(b0, b1, w0, w1);
}

__device__ __forceinline__ int quad_col(int tt, int lane) { return tt * TILE + 2 * (lane & 3) + ((lane >> 2) & 1); }
__device__ __forceinline__ int quad_row(int lane) { return (lane >> 4) * 4 + ((lane >> 3) & 1) * 8; }

__device__ __forceinline__ void scatter(float* dst, int ld, int row0, const Span& sp, int tt, const float (&a)[4], int lane) {
  const float4 v = transpose4(a, lane);
  const int tok = quad_col(tt, lane);
  if (tok < sp.count) red4(dst + long(sp.first + tok) * ld + row0 + quad_row(lane), v);
}

__device__ __forceinline__ uint4 frag(const uint8_t* stage, int f, int lane) { return reinterpret_cast<const uint4*>(stage + f * FRAG)[lane]; }

// ------------------------------------------------------------------------------ B staging
// f16 B tile: [token tile][K block][8 tokens x 64 B]; lane (g, t) reads 16 B at g*64 + t*16.
__device__ __forceinline__ uint8_t* bt_row(Ctx& c, int tok, int ks, int k) { return c.sm.bt + ((tok >> 3) * ks + k) * 512 + (tok & 7) * 64; }
__device__ __forceinline__ uint4 ld_b(const Ctx& c, int ks, int tt, int k) { return reinterpret_cast<const uint4*>(c.sm.bt + (tt * ks + k) * 512)[c.lane]; }

struct Bq {
  uint4 x;
  float2 f;  // activation scale factors of output columns 2t, 2t+1
};

// Copy `count` f16 P8 activation rows (ks K blocks of 64 B each) into the B tile; with XS also
// the per-(token, K block) exponents as float factors bs[tile][k][8].
template <bool XS, class Row, class Scale>
__device__ __forceinline__ void fill_q(Ctx& c, int count, int ks, Row row, Scale scale) {
  c.sync();
  for (int tok = c.warp; tok < count; tok += GW) {
    const uint8_t* src = row(tok);
    for (int r = c.lane; r < ks * 4; r += 32) cp_async16(bt_row(c, tok, ks, r >> 2) + (r & 3) * 16, src + r * 16);
    if constexpr (XS) {
      if (c.lane < ks) c.sm.bs[((tok >> 3) * ks + c.lane) * 8 + (tok & 7)] = e8m0_to_float(__ldcg(scale(tok) + c.lane));
    }
  }
  cp_async_wait_all();
  c.sync();
}

// wgmma B tile: copy `count` e4m3 activation rows (wg order, ks k32 blocks = 2 x 16 B chunks each)
// into the K-major core-matrix layout of desc_b(bt, nt); with XS also the per-(token, k32)
// factors bs[tile][k][8] (as fill_q).
template <bool XS, class Row, class Scale>
__device__ __forceinline__ void fill_wg(Ctx& c, int count, int ks, int nt, Row row, Scale scale, uint8_t* dst = nullptr) {
  uint8_t* tile = dst ? dst : c.sm.bt;
  c.sync();
  for (int tok = c.warp; tok < count; tok += GW) {
    const uint8_t* src = row(tok);
    for (int ch = c.lane; ch < ks * 2; ch += 32) cp_async16(tile + bt_wg_off(tok, ch, nt), src + ch * 16);
    if constexpr (XS) {
      if (c.lane < ks) c.sm.bs[((tok >> 3) * ks + c.lane) * 8 + (tok & 7)] = scale(tok, c.lane);
    }
  }
  cp_async_wait_all();
  fence_proxy_async();
  c.sync();
}

// ------------------------------------------------------------------------------ GEMM driver
// Warp w handles token tile tt = w % ts and row tiles rt0 + r*rs (BRRR gemm).
template <int A, bool HOLD = false, class Load, class Step>
__device__ __forceinline__ void gemm(Ctx& c, const Span& sp, int ksteps, float (&acc)[GW][A], Load load, Step step) {
  const int rs = GW / sp.ts, rt0 = c.warp / sp.ts, tt = c.warp % sp.ts;
  const bool live = tt < sp.tiles;
#pragma unroll
  for (int r = 0; r < GW; ++r)
#pragma unroll
    for (int e = 0; e < A; ++e) acc[r][e] = 0.0f;
  const int stages = (ksteps + 3) >> 2;
  if (DS41H_DECODER_DENSE_TS1 && sp.ts == 1) {  // one token tile: warp = row tile (rs = GW, rt0 = warp, tt = 0)
#pragma unroll 1
    for (int s = 0; s < stages; ++s) {
      if constexpr (TRACE && !HOLD) {
        if (s >= 1 && s <= 3) c.tmark(TR_RB + s - 1);
      }
      if (!HOLD && s) c.next();
      const uint8_t* stage = HOLD ? c.held(s) : c.stage();
      if constexpr ((ABL & 256) != 0 && !HOLD) {  // ablation: no dense math (the gate keeps its: routing must stay real)
      } else if (s * 4 + 3 < ksteps) {  // full stage (the common case): no per-q bound checks
#pragma unroll
        for (int q = 0; q < 4; ++q) step(acc[0], stage, q, c.warp, load(0, s * 4 + q));
      } else {
#pragma unroll
        for (int q = 0; q < 4; ++q)
          if (s * 4 + q < ksteps) step(acc[0], stage, q, c.warp, load(0, s * 4 + q));
      }
    }
    c.tmark(TR_G);
    if (!HOLD) c.release();
    return;
  }
#pragma unroll 1
  for (int s = 0; s < stages; ++s) {
    if constexpr (TRACE && !HOLD) {  // per-stage marks (words 17..19 = stage 1..3 start) of the streamed dense jobs
      if (s >= 1 && s <= 3) c.tmark(TR_RB + s - 1);
    }
    if (!HOLD && s) c.next();
    const uint8_t* stage = HOLD ? c.held(s) : c.stage();
    if (live && ((ABL & 256) == 0 || HOLD)) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        if (s * 4 + q < ksteps) {
          const auto b = load(tt, s * 4 + q);
#pragma unroll
          for (int r = 0; r < GW; ++r)
            if (r < sp.ts) step(acc[r], stage, q, rt0 + r * rs, b);
        }
      }
    }
  }
  c.tmark(TR_G);
  if (!HOLD) c.release();
}

template <int A, class Store>
__device__ __forceinline__ void stores(const Ctx& c, const Span& sp, const float (&acc)[GW][A], Store store) {
  const int rs = GW / sp.ts, rt0 = c.warp / sp.ts, tt = c.warp % sp.ts;
  if (tt >= sp.tiles) return;
#pragma unroll
  for (int r = 0; r < GW; ++r)
    if (r < sp.ts) store(rt0 + r * rs, tt, acc[r]);
}

// Dense FP8 GEMM over the staged B tile: 32x32-block weight scale x optional activation factor.
template <bool XS, class Store>
__device__ __forceinline__ void dense_q(Ctx& c, const Span& sp, int ks, Store store) {
  const int t = c.lane & 3;
  float acc[GW][4];
  gemm<4>(
      c, sp, ks, acc,
      [&](int tt, int k) {
        Bq b;
        b.x = ld_b(c, ks, tt, k);
        if constexpr (XS) b.f = *reinterpret_cast<const float2*>(c.sm.bs + (tt * ks + k) * 8 + 2 * t);
        else b.f = make_float2(1.0f, 1.0f);
        return b;
      },
      [&](float (&a)[4], const uint8_t* stage, int q, int rt, const Bq& b) {
        float d[4];
        mma32(d, frag(stage, q * 4 + rt, c.lane), b.x);
        const float sw = e8m0_to_float(stage[SCALE_OFF + (rt >> 1) * 4 + q]);
        const float s0 = XS ? sw * b.f.x : sw, s1 = XS ? sw * b.f.y : sw;
        a[0] = fmaf(d[0], s0, a[0]);
        a[1] = fmaf(d[1], s1, a[1]);
        a[2] = fmaf(d[2], s0, a[2]);
        a[3] = fmaf(d[3], s1, a[3]);
      });
  stores(c, sp, acc, store);
}

// Dense FP8 GEMM, RS-wgmma (DENSE_MMA=1): the consumer warpgroup covers the stage's 64 rows
// (warp w = 16-row tile w, its P8 fragments are the A registers) x all 8NT tokens of the wg B
// tile (fill_wg / woa). DensePlan units of 4 (NT=1) or 2 (NT=2) k32 blocks; a stage is released
// as soon as its fragments are in registers. Per k32 and warp (16 rows x 32 B = 512 B): 1 LDS.128,
// 1 wgmma, 4NT FFMA with the 32x32 weight scale (2 integer ops to decode), with XS times the
// per-(token, k32) factors (LDS.64 + 2 FMUL per 8 tokens). `post` scales the totals (static
// activation scale). store(rt, tt, acc[4]) as for dense_q. KS = job k32 blocks (16 or 18).
template <bool XS, int KS, int NT, class Store>
__device__ __forceinline__ void dense_wg(Ctx& c, const Span& sp, float post, Store store, uint32_t bbase = 0) {
  using Plan = DensePlan<KS, NT>;
  constexpr int NKU = Plan::NKU;
  const int t = c.lane & 3;
  const uint64_t desc = desc_b(bbase ? bbase : bt_smem_addr(), NT);
  float tot[4 * NT], acc[2][NKU][4 * NT];
  unsigned A[2][NKU][4];
  unsigned sw[2];  // ue8m0 word (4 k32 blocks) of this warp's 32-row block, per buffer
  const uint8_t* stage = c.stage();
  unsigned sws = 0;
#pragma unroll
  for (int e = 0; e < 4 * NT; ++e) tot[e] = 0.0f;
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int q = 0; q < NKU; ++q)
#pragma unroll
      for (int e = 0; e < 4 * NT; ++e) acc[b][q][e] = 0.0f;
  auto load = [&](auto uc) {
    constexpr int u = decltype(uc)::value, B = u & 1;
    if constexpr (Plan::first(u)) {
      if constexpr (u > 0) c.acquire();
      stage = c.stage();
      sws = *reinterpret_cast<const unsigned*>(stage + SCALE_OFF + (c.warp >> 1) * 4);
    }
#pragma unroll
    for (int q = 0; q < Plan::nk(u); ++q) {
      const uint4 v = frag(stage, (Plan::q0(u) + q) * 4 + c.warp, c.lane);
      A[B][q][0] = v.x;
      A[B][q][1] = v.z;
      A[B][q][2] = v.y;
      A[B][q][3] = v.w;
    }
    sw[B] = sws;
    if constexpr (Plan::last(u)) c.release();
  };
  auto issue = [&](auto uc) {
    constexpr int u = decltype(uc)::value;
    wg_group<NT, Plan::nk(u), NKU>(A[u & 1], acc[u & 1], desc, Plan::kb(u));
  };
  auto retire = [&](auto uc) {
    constexpr int u = decltype(uc)::value, B = u & 1, NK = Plan::nk(u);
    wg_retired<NT, NK, NKU>(A[B], acc[B]);
#pragma unroll
    for (int q = 0; q < NK; ++q) {
      const float fw = ue8m0_byte(sw[B], Plan::q0(u) + q);
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        float s0 = fw, s1 = fw;
        if constexpr (XS) {
          const float2 fx = *reinterpret_cast<const float2*>(c.sm.bs + (j * KS + Plan::kb(u) + q) * 8 + 2 * t);
          s0 *= fx.x;
          s1 *= fx.y;
        }
        tot[4 * j + 0] = fmaf(acc[B][q][4 * j + 0], s0, tot[4 * j + 0]);
        tot[4 * j + 1] = fmaf(acc[B][q][4 * j + 1], s1, tot[4 * j + 1]);
        tot[4 * j + 2] = fmaf(acc[B][q][4 * j + 2], s0, tot[4 * j + 2]);
        tot[4 * j + 3] = fmaf(acc[B][q][4 * j + 3], s1, tot[4 * j + 3]);
      }
    }
  };
  wg_pipe<Plan>(load, issue, retire);
  c.tmark(TR_G);
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const float a[4] = {tot[4 * j] * post, tot[4 * j + 1] * post, tot[4 * j + 2] * post, tot[4 * j + 3] * post};
    store(c.warp, j, a);
  }
}

// Dense GEMM dispatch: with DENSE_MMA=1 and <= WG_NT token tiles, fill(nt) stages the e4m3 B tile
// and dense_wg runs; returns false otherwise (the caller then runs its HMMA fill_q + dense_q).
template <bool XS, int KS, class Fill, class Store>
__device__ __forceinline__ bool dense_wg_dispatch(Ctx& c, const Span& sp, float post, Fill fill, Store store, uint32_t bbase = 0) {
  if constexpr (DENSE_WG) {
    if (sp.tiles <= WG_NT) {
      fill(sp.tiles);
      if (sp.tiles == 1) dense_wg<XS, KS, 1>(c, sp, post, store, bbase);
      else dense_wg<XS, KS, 2>(c, sp, post, store, bbase);
      return true;
    }
  }
  return false;
}

// ------------------------------------------------------------------------------ RoPE
__device__ __forceinline__ float2 rope_cs(const float* row, int pair) { return make_float2(__ldg(row + pair), __ldg(row + ROPE / 2 + pair)); }

// Inverse interleaved RoPE (stock fused_inv_rope_fp8_quant.py:117-131): even x*cos + y*sin,
// odd y*cos - x*sin; rounded to bf16 like stock's bf16 o_proj input.
__device__ __forceinline__ uint4 rope_inv(uint4 v, const float* cs, int d) {
  if (d < NOPE) return v;
  unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float2 r = rope_cs(cs, (d + 2 * q - NOPE) >> 1);
    const float2 x = unpack_bf16(u[q]);
    u[q] = pack_bf16(x.x * r.x + x.y * r.y, x.y * r.x - x.x * r.y);
  }
  return make_uint4(u[0], u[1], u[2], u[3]);
}

__device__ __forceinline__ long clamp_pos(long pos, long max_pos) { return pos < 0 ? 0 : pos >= max_pos ? max_pos - 1 : pos; }

// rope_inv with the token's cos / sin row in shared memory (WOA_PRE).
__device__ __forceinline__ uint4 rope_inv_s(uint4 v, const float* cs, int d) {
  if (d < NOPE) return v;
  unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int pr = (d + 2 * q - NOPE) >> 1;
    const float2 r = make_float2(cs[pr], cs[ROPE / 2 + pr]);
    const float2 x = unpack_bf16(u[q]);
    u[q] = pack_bf16(x.x * r.x + x.y * r.y, x.y * r.x - x.x * r.y);
  }
  return make_uint4(u[0], u[1], u[2], u[3]);
}

// WOA_PRE: cos / sin rows of the m tokens -> wf (before pdl_wait; positions / cos_sin are not written by the previous kernel).
__device__ CTXFN void woa_prefetch(Ctx& c) {
  const Params& p = c.p;
  if (!(WOA_PRE && p.m <= WOA_SMALL && int(blockIdx.x) < layout(p.m, p.r1kp, p.lag).wob)) {
    if (c.tid == 0) c.sm.woa_pre = 0;
    return c.sync();
  }
  for (int i = c.tid; i < p.m * ROPE; i += GT) {
    const long pos = clamp_pos(__ldg(p.positions + i / ROPE), p.max_pos);
    c.sm.wf[i] = __ldg(p.cos_sin + pos * ROPE + i % ROPE);
  }
  if (c.tid == 0) c.sm.woa_pre = 1;
  c.sync();  // wf and the flag visible to every consumer thread
}

// ------------------------------------------------------------------------------ jobs: attention out
// HMMA wo_a (f16 o_rot B tile, dense 32x32 weight scales).
template <int KS, int TOK, int PER>
__device__ __forceinline__ void woa_hmma(Ctx& c, int rb, const Span& sp, int r, int d, const uint4 (&v)[PER], const long (&pos)[PER], bool pre = false) {
  const Params& p = c.p;
  constexpr int VECS = KS * 4;
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int tok = (c.tid + u * GT) / VECS;
    if (tok >= sp.count) continue;
    const uint4 x = pre ? rope_inv_s(v[u], c.sm.wf + (sp.first + tok) * ROPE, d) : rope_inv(v[u], p.cos_sin + pos[u] * ROPE, d);
    *reinterpret_cast<uint4*>(bt_row(c, tok, KS, r >> 2) + (r & 3) * 16) = bf16x8_f16_p8(x);
  }
  c.sync();
  c.tmark(TR_X0);
  float acc[GW][4];
  gemm<4>(
      c, sp, KS, acc, [&](int tt, int k) { return ld_b(c, KS, tt, k); },
      [&](float (&a)[4], const uint8_t* stage, int q, int rt, const uint4& b) {
        float d4[4];
        mma32(d4, frag(stage, q * 4 + rt, c.lane), b);
        const float sw = e8m0_to_float(stage[SCALE_OFF + (rt >> 1) * 4 + q]);
#pragma unroll
        for (int e = 0; e < 4; ++e) a[e] = fmaf(d4[e], sw, a[e]);
      });
  stores(c, sp, acc, [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.zacc, OA, rb * 64 + rt * 16, sp, tt, a, c.lane); });
}

template <int KS, int TOK>
__device__ WG_INLINE void woa_group(Ctx& c, int rb, int kp, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  constexpr int VECS = KS * 4, PER = TOK * VECS / GT;
  static_assert(TOK * KS * 64 <= BT && TOK * VECS % GT == 0);
  const int k0 = kp * KS * 32;  // offset inside the group's 4096 inputs
  // 16 row blocks of 64 per 1024-row group; group g reads heads 8g..8g+7 (stock o_proj.py:89-103).
  const __nv_bfloat16* o = p.o + (rb / 16) * 8 * HDIM + k0;
  const int r = c.tid % VECS, d = (k0 + r * 8) % HDIM;
  const bool pre = WOA_PRE && TOK <= WOA_SMALL && c.sm.woa_pre;
  uint4 v[PER];
  long pos[PER];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int tok = (c.tid + u * GT) / VECS;
    const long t = sp.first + tok;
    v[u] = tok < sp.count ? __ldcg(reinterpret_cast<const uint4*>(o + t * p.o_stride + r * 8)) : make_uint4(0, 0, 0, 0);
    pos[u] = !pre && tok < sp.count && d >= NOPE ? clamp_pos(__ldg(p.positions + t), p.max_pos) : 0;
  }
  c.sync();
  if constexpr (DENSE_WG && TOK <= WG_NT * TILE) {
    // RS-wgmma: o_rot -> e4m3 per (token, 32 K) with the official act_quant ue8m0 rule (amax over
    // the quad that holds the 32 values), factors 2^e in bs, wg-order B tile for N = 8 nt.
    const int nt = sp.tiles;
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int tok = (c.tid + u * GT) / VECS;
      const uint4 x = tok < sp.count ? rope_inv(v[u], p.cos_sin + pos[u] * ROPE, d) : make_uint4(0, 0, 0, 0);
      float f[8] = {};
      add8_bf16(f, x);
      float amax = 0.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(f[e]));
      const int e = quant_exp(quad_max(amax), AMAX_FLOOR);
      if (tok < sp.count) {
        const uint2 w = e4m3_wg8(f, pow2(-e));
        const int kb = r >> 2;
        uint8_t* base = c.sm.bt + (tok & 7) * 16 + (r & 3) * 4;
        *reinterpret_cast<unsigned*>(base + (2 * kb * nt + (tok >> 3)) * 128) = w.x;
        *reinterpret_cast<unsigned*>(base + ((2 * kb + 1) * nt + (tok >> 3)) * 128) = w.y;
        if ((r & 3) == 0) c.sm.bs[((tok >> 3) * KS + kb) * 8 + (tok & 7)] = pow2(e);
      }
    }
    fence_proxy_async();
    c.sync();
    auto store = [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.zacc, OA, rb * 64 + rt * 16, sp, tt, a, c.lane); };
    if (nt == 1) dense_wg<true, KS, 1>(c, sp, 1.0f, store);
    else dense_wg<true, KS, 2>(c, sp, 1.0f, store);
  } else {
    woa_hmma<KS, TOK>(c, rb, sp, r, d, v, pos, pre);
  }
  if (WOA_PRE && c.tid == 0) c.sm.woa_pre = 0;  // wf is reused by later jobs
}

// z = bf16(wo_a output) (stock bmm writes bf16, o_proj.py:76-103) -> f16 * 2^-e per 32 block.
__device__ CTXFN void finish_woa(Ctx& c, int rb, int tg) {
  constexpr int PER = OTOK * 8 / GT;
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  float z[PER][8];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT, tok = sp.first + i / 8;
    float v[8];
    unpack8(load8(p.s.zacc + long(tok) * OA + rb * 64 + i % 8 * 8, i / 8 < sp.count), v);
#pragma unroll
    for (int e = 0; e < 8; ++e) z[u][e] = bf(v[e]);
  }
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT, tok = sp.first + i / 8, col = rb * 64 + i % 8 * 8;
    const int e = f16_block_exp(z[u]);  // quad = one 32-wide block; every lane participates
    if (i / 8 >= sp.count) continue;
    clear8(p.s.zacc + long(tok) * OA + col);
    const float s = pow2(-e);
    float zs[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) zs[k] = z[u][k] * s;
    *reinterpret_cast<uint4*>(p.s.zh + long(tok) * OA + col) = pack8_p8(zs);
    // e4m3 copy for the wgmma wo_b: zs * 2^-6 has amax in [2^7, 2^8) per 32 block.
    if constexpr (DENSE_WG) store_wg8(p.s.zq + long(tok) * OA, col, e4m3_wg8(zs, 1.0f / 64.0f));
    if (i % 4 == 0) p.s.ze[tok * (OA / 32) + col / 32] = uint8_t(e + 127);
  }
  c.signal(C_Z + rb * 2 / KS_WOB * GROUPS + tg);
}

__device__ WG_INLINE void job_woa(Ctx& c, int rb, int kp, int tg, int stages) {
  if (stages * 4 == KS_WOA) woa_group<KS_WOA, OTOK>(c, rb, kp, tg);
  else woa_group<2 * KS_WOA, WOA_SMALL>(c, rb, kp, tg);
  c.tmark(TR_X1);
  if constexpr (FUSED_Z) return c.signal(C_Z + rb * 64 / (wob_ks(c.p.m) * 32) * GROUPS + tg);  // wo_b quantizes z itself
  if (!c.arrive(C_ZA + rb * GROUPS + tg, OAK / 32 / (stages * 4))) return;
  c.tmark(TR_X2);
  finish_woa(c, rb, tg);
}

// FUSED_Z: the wo_b B tile of K part kp straight from the fp32 wo_a sums, with finish_woa's numerics
// (z = bf16(sum), per-32 f16 staging exponent e): nt = 0 -> HMMA tile (f16 z 2^-e, P8, factor 2^e,
// fill_q layout), nt > 0 -> wgmma tile (e4m3 z 2^-(e+6), wg order, factor 2^(e+6), fill_wg layout).
__device__ CTXFN void wob_fill_z(Ctx& c, const Span& sp, int k0, int nt, int ks) {  // k0: first k32 block of the K part
  const Params& p = c.p;
  const int VPT = ks * 4;  // 8-value vectors per token (32 ks columns)
  const int nv = sp.count * VPT;
  constexpr int ZB = WOB3 ? 6 : 1;  // WOB3: the loads of ZB passes are issued before any conversion
  c.sync();
  for (int base0 = 0; base0 < nv; base0 += GT * ZB) {
  Acc8 za[ZB];
#pragma unroll
  for (int u = 0; u < ZB; ++u) {
    const int i = base0 + u * GT + c.tid, tok = i / VPT, v = i % VPT;
    const bool live = i < nv;
    za[u] = load8(p.s.zacc + long(sp.first + (live ? tok : 0)) * OA + k0 * 32 + v * 8, live);
  }
#pragma unroll
  for (int u = 0; u < ZB; ++u) {  // whole quads stay live together (nv % 64 == 0)
    const int base = base0 + u * GT;
    if (base >= nv) break;  // CTA-uniform
    const int i = base + c.tid, tok = i / VPT, v = i % VPT;
    const bool live = i < nv;
    float z[8];
    unpack8(za[u], z);
#pragma unroll
    for (int e = 0; e < 8; ++e) z[e] = bf(z[e]);
    const int e = f16_block_exp(z);  // quad = the 4 vectors of one 32 block
    if (!live) continue;
    const float sc = pow2(-e);
    float zs[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) zs[k] = z[k] * sc;
    const int k32 = v >> 2;
    if (nt == 0) {
      *reinterpret_cast<uint4*>(bt_row(c, tok, ks, k32) + (v & 3) * 16) = pack8_p8(zs);
      if ((v & 3) == 0) c.sm.bs[((tok >> 3) * ks + k32) * 8 + (tok & 7)] = pow2(e);
    } else {
      const uint2 w = e4m3_wg8(zs, 1.0f / 64.0f);
      *reinterpret_cast<unsigned*>(c.sm.bt + bt_wg_off(tok, 2 * k32, nt) + (v & 3) * 4) = w.x;
      *reinterpret_cast<unsigned*>(c.sm.bt + bt_wg_off(tok, 2 * k32 + 1, nt) + (v & 3) * 4) = w.y;
      if ((v & 3) == 0) c.sm.bs[((tok >> 3) * ks + k32) * 8 + (tok & 7)] = pow2(e) * 64.0f;
    }
  }
  }  // ZB passes
  if (nt) fence_proxy_async();
  c.sync();
}

__device__ CTXFN void finish_wob(Ctx& c, int rb, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  __nv_bfloat16* mine = inbox(p, p.rank, 0, 0, c.epoch);
  constexpr int PER = OTOK * 8 / GT;
  Acc8 v[PER];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT;
    v[u] = load8(p.s.yacc + long(sp.first + i / 8) * HID + rb * 64 + (i % 8) * 8, i < sp.count * 8);
  }
#pragma unroll
  const bool llp = LLP && FUSED_AR1 && p.m <= COARSE_M;
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT, tok = sp.first + i / 8, col = rb * 64 + (i % 8) * 8;
    if (i >= sp.count * 8) break;
    clear8(p.s.yacc + long(tok) * HID + col);
    float y[8];
    unpack8(v[u], y);
    const uint4 pk = pack8_bf16(y);
    if (llp) {  // LLP: push the 4 words {bf16 pair, epoch} into every live rank's AR#1 LL inbox (src = this rank)
      const unsigned live = peer_mask(p) | (1u << p.rank);
#pragma unroll
      for (int d = 0; d < RANKS; ++d) {
        if (!(live >> d & 1)) continue;
        uint2* dst = ll_box(p, d, 0, c.epoch, p.rank) + long(tok) * (HID / 2) + col / 2;
        ll_put(dst, pk.x, c.epoch);
        ll_put(dst + 1, pk.y, c.epoch);
        ll_put(dst + 2, pk.z, c.epoch);
        ll_put(dst + 3, pk.w, c.epoch);
      }
    } else {
      *reinterpret_cast<uint4*>(mine + long(tok) * HID + col) = pk;
    }
  }
  if constexpr (FUSED_AR1) {  // flag the published row block in the peers now, then locally
    if (!llp) post_slice(c, rb, tg, c.epoch, peer_mask(p));
    return c.signal(C_YC + rb);
  }
  c.signal(C_YC + rb / 2);
}

__device__ WG_INLINE void job_wob(Ctx& c, int rb, int kp, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  const bool w3 = WOB3 && wob3_m(p.m);
  const int ks = w3 ? 4 * wob3_st(kp) : wob_ks(p.m), k0 = w3 ? 20 * kp : kp * ks;  // K part: k32 blocks k0 .. k0 + ks - 1
  auto store = [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.yacc, HID, rb * 64 + rt * 16, sp, tt, a, c.lane); };
  const bool wg = dense_wg_dispatch<true, KS_WOB>(
      c, sp, 1.0f,
      [&](int nt) {
        if constexpr (FUSED_Z) {
          wob_fill_z(c, sp, kp * KS_WOB, nt, KS_WOB);
          return;
        }
        fill_wg<true>(
            c, sp.count, KS_WOB, nt, [&](int tok) { return p.s.zq + long(sp.first + tok) * OA + kp * KS_WOB * 32; },
            [&](int tok, int k) { return e8m0_to_float(__ldcg(p.s.ze + (sp.first + tok) * (OA / 32) + kp * KS_WOB + k)) * 64.0f; });
      },
      store);
  if (!wg) {
    if constexpr (FUSED_Z) wob_fill_z(c, sp, k0, 0, ks);
    else
      fill_q<true>(
          c, sp.count, ks, [&](int tok) { return reinterpret_cast<const uint8_t*>(p.s.zh + long(sp.first + tok) * OA + kp * ks * 32); },
          [&](int tok) { return p.s.ze + (sp.first + tok) * (OA / 32) + kp * ks; });
    c.tmark(TR_X0);
    dense_q<true>(c, sp, ks, store);
  }
  c.tmark(TR_X1);
  if (c.arrive(C_YA + rb * GROUPS + tg, wob_kp(p.m))) c.tmark(TR_X2), finish_wob(c, rb, tg);
}

__device__ CTXFN void job_xa(Ctx& c, int k) {
  c.release();
  share(c, 0, k, c.p.s.yattn);
  c.signal(C_YR + k);
}

// ------------------------------------------------------------------------------ jobs: FFN mHC
// Attention mHC post (stock mhc_fused_tilelang, tilelang_kernels.py:731-751) for one 32-column
// slice: r[j] = post[j]*y + sum_k comb[k][j]*resid[k] in fp32; resid1 = bf16(r); the 24-wide
// GEMV and sum r^2 use fp32 r; the FFN collapse sum_j attn_pre[j]*bf16(r[j]) is rounded to
// bf16 (tilelang_kernels.py:441-468).
// OWNA_PRE staging in bt (see the knob): co = [tok][24] fp32 (post[4] | pre[4] | comb[4][4]), rr = [tok][4 streams][32] bf16.
constexpr int OP_RR = COARSE_M * 24 * 4;
__device__ CTXFN void owna_pre(Ctx& c, int s) {
  const Params& p = c.p;
  c.sync();  // the previous job's reads of bt are done
  float* co = reinterpret_cast<float*>(c.sm.bt);
  __nv_bfloat16* rr = reinterpret_cast<__nv_bfloat16*>(c.sm.bt + OP_RR);
  for (int i = c.tid; i < p.m * 24; i += GT) {  // 4-byte pieces (the mix tensors may be views)
    const int tok = i / 24, j = i % 24;
    const float* src = j < 4 ? p.attn_post + tok * HC + j : j < 8 ? p.attn_pre + tok * HC + j - 4 : p.attn_comb + tok * HC * HC + j - 8;
    cp_async4(co + i, src);
  }
  for (int i = c.tid; i < p.m * HC * 4; i += GT) {  // 16 B = 8 columns of one (token, stream)
    const int ts = i / 4, q = i % 4;
    cp_async16(rr + ts * 32 + q * 8, p.resid0 + long(ts) * HID + s * 32 + q * 8);
  }
}

template <bool PRE = false>
__device__ CTXFN void owna_tile(Ctx& c, int s, int t8, const float* fn0, const float* fn1) {
  constexpr int PER = TILE / GW;
  const Params& p = c.p;
  const Span sp = tile_span(p.m, t8);
  const int col = s * 32 + c.lane;
  const float* co = reinterpret_cast<const float*>(c.sm.bt);
  const __nv_bfloat16* rr = reinterpret_cast<const __nv_bfloat16*>(c.sm.bt + OP_RR);
  if constexpr (PRE) {
    cp_async_wait_all();
    c.sync();  // the staged mixes / resid0 slice of every thread visible
  }
  float ya[PER], rk[PER][HC];
  const bool llp = LLP && FUSED_AR1 && p.m <= COARSE_M;
  uint2 llw[PER * RANKS];
  if (llp) {  // LLP: this lane's column of the PER tokens from every live rank's pushed words (local inbox), polled together
    const unsigned live = peer_mask(p) | (1u << p.rank);
    const uint2* src[PER * RANKS];
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int i = c.warp + u * GW, tok = sp.first + (i < sp.count ? i : 0);
#pragma unroll
      for (int r = 0; r < RANKS; ++r) src[u * RANKS + r] = ll_box(p, p.rank, 0, c.epoch, (live >> r & 1) ? r : p.rank) + long(tok) * (HID / 2) + col / 2;
    }
    ll_wait(llw, src, c.epoch);
  }
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.warp + u * GW, tok = sp.first + (i < sp.count ? i : 0);
    if (FUSED_AR1 && llp) {  // all-reduce #1 (LLP): fp32 sum of the ranks' bf16 in rank order (as below)
      const unsigned live = peer_mask(p) | (1u << p.rank);
      float acc = 0.0f;
#pragma unroll
      for (int r = 0; r < RANKS; ++r) {
        const unsigned w = llw[u * RANKS + r].x, h = (live >> r & 1) ? ((c.lane & 1) ? (w >> 16) : (w & 0xffffu)) : 0u;
        acc += __uint_as_float(h << 16);
      }
      ya[u] = bf(acc);
    } else if constexpr (FUSED_AR1) {  // all-reduce #1 of this column: fp32 sum of the ranks' bf16 in rank order (reduce4)
      const unsigned live = peer_mask(p) | (1u << p.rank);
      unsigned short v[RANKS];
#pragma unroll
      for (int r = 0; r < RANKS; ++r) {
        const unsigned short* src = reinterpret_cast<const unsigned short*>(inbox(p, (live >> r & 1) ? r : p.rank, 0, 0, c.epoch)) + long(tok) * HID + col;
        v[r] = !(live >> r & 1) ? 0 : r == p.rank ? __ldcg(src) : __ldcv(src);
      }
      float acc = 0.0f;
#pragma unroll
      for (int r = 0; r < RANKS; ++r) acc += __uint_as_float(unsigned(v[r]) << 16);
      ya[u] = bf(acc);
    } else {
      ya[u] = __bfloat162float(__ldcg(p.s.yattn + tok * HID + col));
    }
    if constexpr (TRACE) {
      if (u == 0 && ya[0] != 12345.0f) c.tmark(TR_S0);  // inbox loads landed
    }
#pragma unroll
    for (int k = 0; k < HC; ++k)
      rk[u][k] = PRE ? __bfloat162float(rr[(tok * HC + k) * 32 + c.lane]) : __bfloat162float(__ldg(p.resid0 + (long(tok) * HC + k) * HID + col));
  }
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.warp + u * GW, tok = sp.first + i;
    if (i >= sp.count) break;
    float r[HC];
    float x = 0.0f, sq = 0.0f;
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      float v = (PRE ? co[tok * 24 + j] : __ldg(p.attn_post + tok * HC + j)) * ya[u];
#pragma unroll
      for (int k = 0; k < HC; ++k) v += (PRE ? co[tok * 24 + 8 + k * HC + j] : __ldg(p.attn_comb + tok * HC * HC + k * HC + j)) * rk[u][k];
      r[j] = v;
      const float r16 = bf(v);
      p.resid1[(long(tok) * HC + j) * HID + col] = __float2bfloat16(r16);
      sq += v * v;
      x += (PRE ? co[tok * 24 + 4 + j] : __ldg(p.attn_pre + tok * HC + j)) * r16;
    }
    if constexpr (TRACE) {
      if (u == 0 && x != 12345.0f) c.tmark(TR_S1);  // mixes / resid1 done
    }
    static_assert(MIX + 2 <= MIXW);
    float v[32] = {};
#pragma unroll
    for (int q = 0; q < MIX; ++q)
#pragma unroll
      for (int j = 0; j < HC; ++j) v[q] += (q < OWNA_ROWS ? fn0[(q * HC + j) * 32 + c.lane] : fn1[((q - OWNA_ROWS) * HC + j) * 32 + c.lane]) * r[j];
    x = bf(x);
    p.s.xc[tok * HID + col] = __float2bfloat16(x);
    v[MIX] = sq;
    v[MIX + 1] = x * x;
    const float total = lane_sums(v, c.lane);
    if constexpr (TRACE) {
      if (u == 0 && total != 12345.0f) c.tmark(TR_S2);  // GEMV + lane sums done
    }
    p.s.mixp[(long(tok) * SLICES + s) * MIXW + c.lane] = total;
    if (c.lane == MIX + 1) p.s.xsq[tok * SLICES + s] = total;
  }
  c.tmark(TR_S3);
  c.signal(C_A + t8);
}

__device__ CTXFN void job_owna(Ctx& c, int s, int span) {
  if constexpr (DS41H_DECODER_OWNA_PRE != 0 && FUSED_AR1) {
    if (c.p.m <= COARSE_M) {  // one tile, one span: stage the flag-independent inputs, then wait for the peers
      owna_pre(c, s);
      if (!LLP) await_slice(c, s / 2, 0, c.epoch, peer_mask(c.p));  // LLP: owna_tile polls the pushed words
      const float* fn0 = reinterpret_cast<const float*>(c.stage());
      const float* fn1 = reinterpret_cast<const float*>(c.held(1));
      c.tmark(TR_X0);
      owna_tile<true>(c, s, 0, fn0, fn1);
      c.tmark(TR_X1);
      return c.drop(2);
    }
  }
  if constexpr (FUSED_AR1) {  // all-reduce #1: the peers' flags of this slice's row block, per token group
    const Params& p = c.p;
    const int per = owna_tiles(p.m);
    if (!(LLP && p.m <= COARSE_M))
      for (int g = span * per * TILE / OTOK; g < token_groups(p.m) && g * OTOK < (span + 1) * per * TILE; ++g) await_slice(c, s / 2, g, c.epoch, peer_mask(p));
  }
  const float* fn0 = reinterpret_cast<const float*>(c.stage());
  const float* fn1 = reinterpret_cast<const float*>(c.held(1));
  c.tmark(TR_X0);
  const int per = owna_tiles(c.p.m), last = min(tiles_of(c.p.m), (span + 1) * per);
  for (int t8 = span * per; t8 < last; ++t8) owna_tile(c, s, t8, fn0, fn1);
  c.tmark(TR_X1);
  c.drop(2);
}

// rsqrt(mean(xc^2) + eps) per token of the span (stock tilelang_kernels.py:470-473).
__device__ CTXFN void group_rs(Ctx& c, const Span& sp, float* rs) {
  constexpr int V = SLICES / 4, QMIN = GT / OTOK;
  const int q = sp.count <= GW ? 32 : sp.count <= 2 * GW ? 16 : sp.count <= 4 * GW ? 8 : QMIN;
  const int i = c.tid / q, l = c.tid % q;
  float v = 0.0f;
  if (i < sp.count) {
    const float4* x = reinterpret_cast<const float4*>(c.p.s.xsq + (sp.first + i) * SLICES);
#pragma unroll
    for (int k = 0; k < cdiv(V, QMIN); ++k)
      if (l + k * q < V) {
        const float4 f = __ldcg(x + l + k * q);
        v += (f.x + f.y) + (f.z + f.w);
      }
  }
  for (int o = q / 2; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (l == 0 && i < sp.count) rs[i] = rsqrtf(v / HID + NORM_EPS);
  c.sync();
}

// RMSNorm(ffn_norm) in parallel K parts (m > XNORM_MIN only): x = bf16(w * (xc * rs)).
__device__ CTXFN void job_xnorm(Ctx& c, int kp, int t8) {
  constexpr int PER = TILE * 64 / GT;
  const Params& p = c.p;
  const Span sp = tile_span(p.m, t8);
  c.release();
  const int k0 = kp * 512 + c.tid % 64 * 8;
  uint4 xv[PER];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int row = (c.tid + u * GT) / 64;
    xv[u] = row < sp.count ? __ldcg(reinterpret_cast<const uint4*>(p.s.xc + (sp.first + row) * HID + k0)) : make_uint4(0, 0, 0, 0);
  }
  float nw[8] = {};
  add8_bf16(nw, __ldg(reinterpret_cast<const uint4*>(p.w.norm + k0)));
  float* rs = c.sm.rs;
  group_rs(c, sp, rs);
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int row = (c.tid + u * GT) / 64, tok = sp.first + row;
    if (row >= sp.count) continue;
    float xc[8] = {}, x8[8];
    add8_bf16(xc, xv[u]);
    const float r = rs[row];
#pragma unroll
    for (int e = 0; e < 8; ++e) x8[e] = bf(nw[e] * (xc[e] * r));
    *reinterpret_cast<uint4*>(p.s.xh + tok * HID + k0) = pack8_p8(x8);
    *reinterpret_cast<uint4*>(p.s.xbf + tok * HID + k0) = pack8_bf16(x8);
    if constexpr (XQ_ON) store_wg8(p.s.xq + long(tok) * HID, k0, e4m3_wg8(x8, xs_inv(p.w)));
  }
  c.signal(C_XK + kp * GROUPS + t8 * TILE / OTOK);
}

// Router sums of this launch (buffer epoch & 1; the gate jobs clear the other buffer for the next launch).
__device__ __forceinline__ float* gsum(const Params& p, unsigned epoch) { return p.s.gacc + long(epoch & 1u) * MAXM * EXPERTS; }
__device__ __forceinline__ unsigned long long* gsum64(const Params& p, unsigned epoch) {
  return reinterpret_cast<unsigned long long*>(p.s.gacc) + long(epoch & 1u) * MAXM * EXPERTS;
}
__device__ __forceinline__ unsigned long long f2fx(float v) { return static_cast<unsigned long long>(__float2ll_rn(v * 4294967296.0f)); }
__device__ __forceinline__ float fx2f(unsigned long long u) { return __ll2float_rn(static_cast<long long>(u)) * 2.3283064365386963e-10f; }
// Raw router-sum word (u64 fixed point or fp32 bits) and its value: loads first, conversions after, so a thread's
// loads are all in flight together (a load -> convert -> use chain per logit serialized L2 round trips).
__device__ __forceinline__ unsigned long long graw(const Params& p, unsigned epoch, int idx) {
  if constexpr (GDET) return __ldcg(gsum64(p, epoch) + idx);
  else return __float_as_uint(__ldcg(gsum(p, epoch) + idx));
}
__device__ __forceinline__ float gval(unsigned long long raw) { return GDET ? fx2f(raw) : __uint_as_float(unsigned(raw)); }
// Router logit idx (= token * EXPERTS + expert) of this launch, and its reset.
__device__ __forceinline__ float gread(const Params& p, unsigned epoch, int idx) { return GDET ? fx2f(__ldcg(gsum64(p, epoch) + idx)) : __ldcg(gsum(p, epoch) + idx); }
__device__ __forceinline__ void gzero(const Params& p, unsigned epoch, int idx) {
  if constexpr (GDET) __stcg(gsum64(p, epoch) + idx, 0ull);
  else __stcg(gsum(p, epoch) + idx, 0.0f);
}
// A gate job's partial (m16n8 C fragment) of rows row0.. -> the launch's router sums (scatter / fixed point).
__device__ __forceinline__ void gscatter(const Params& p, unsigned epoch, int row0, const Span& sp, int tt, const float (&a)[4], int lane) {
  if constexpr (GDET) {
    const float4 v = transpose4(a, lane);
    const int tok = quad_col(tt, lane);
    if (tok < sp.count) {
      unsigned long long* d = gsum64(p, epoch) + long(sp.first + tok) * EXPERTS + row0 + quad_row(lane);
      atomicAdd(d, f2fx(v.x));
      atomicAdd(d + 1, f2fx(v.y));
      atomicAdd(d + 2, f2fx(v.z));
      atomicAdd(d + 3, f2fx(v.w));
    }
  } else {
    scatter(gsum(p, epoch), EXPERTS, row0, sp, tt, a, lane);
  }
}

// Router logits (stock GateLinear, fp32 output): bf16 weights x bf16 x, fp32 accumulate.
__device__ CTXFN void gate_group(Ctx& c, int rb, int kp, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  const int k0 = kp * KS_GATE * 16;
  const bool split = split_norm(p.m);
  c.sync();
  if (split) {
    c.wait(C_XK + k0 / 512 * GROUPS + tg, sp.tiles);
    for (int tok = c.warp; tok < sp.count; tok += GW) {
      const uint8_t* src = reinterpret_cast<const uint8_t*>(p.s.xbf + long(sp.first + tok) * HID + k0);
      for (int r = c.lane; r < KS_GATE * 2; r += 32)
        cp_async16(c.sm.bt + ((tok >> 3) * KS_GATE + (r >> 1)) * 256 + (tok & 7) * 32 + (r & 1) * 16, src + r * 16);
    }
    cp_async_wait_all();
    c.sync();
  } else {
    // x / norm loads of this thread's first item are issued before the rs reduction (both L2 round
    // trips overlap; one item per thread for m <= 32).
    uint4 xin0[4], win0[4];
    if (c.tid < sp.count * (KS_GATE / 2)) {
      const int tok = c.tid / (KS_GATE / 2), k = k0 + c.tid % (KS_GATE / 2) * 32;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        xin0[u] = __ldcg(reinterpret_cast<const uint4*>(p.s.xc + (sp.first + tok) * HID + k + u * 8));
        win0[u] = __ldg(reinterpret_cast<const uint4*>(p.w.norm + k + u * 8));
      }
    }
    group_rs(c, sp, c.sm.rs);
    for (int i = c.tid; i < sp.count * (KS_GATE / 2); i += GT) {
      const int tok = i / (KS_GATE / 2), j = i % (KS_GATE / 2), k = k0 + j * 32;
      const long t = sp.first + tok;
      const float r = c.sm.rs[tok];
      uint4 xin[4], win[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        xin[u] = i == c.tid ? xin0[u] : __ldcg(reinterpret_cast<const uint4*>(p.s.xc + t * HID + k + u * 8));
        win[u] = i == c.tid ? win0[u] : __ldg(reinterpret_cast<const uint4*>(p.w.norm + k + u * 8));
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        float xf[8] = {}, nf[8] = {}, x8[8];
        add8_bf16(xf, xin[u]);
        add8_bf16(nf, win[u]);
#pragma unroll
        for (int e = 0; e < 8; ++e) x8[e] = bf(nf[e] * (xf[e] * r));
        // u-th 8 columns of the 32: k16 step 2j + u/2, 16-byte half u%2 (natural bf16 order).
        *reinterpret_cast<uint4*>(c.sm.bt + ((tok >> 3) * KS_GATE + 2 * j + u / 2) * 256 + (tok & 7) * 32 + (u & 1) * 16) = pack8_bf16(x8);
        if (rb == 0) {
          *reinterpret_cast<uint4*>(p.s.xh + t * HID + k + u * 8) = pack8_p8(x8);
          if (LC && coarse_m(p.m)) {  // R1F x image: [k32][token][64 B], the 8-group's 16 bytes (P8)
            const int kk = k + u * 8;
            *reinterpret_cast<uint4*>(p.s.xhi + ((kk >> 5) * TILE + t) * 64 + ((kk >> 3) & 3) * 16) = pack8_p8(x8);
          }
          if constexpr (XQ_ON) {
            const uint2 q = e4m3_wg8(x8, xs_inv(p.w));
            store_wg8(p.s.xq + t * HID, k + u * 8, q);
            if (LCQ && coarse_m(p.m)) {  // resident-x image: chunk 2 kb (+1), token t, word ((k >> 3) & 3)
              const int kk = k + u * 8, kb = kk >> 5, w4 = ((kk >> 3) & 3) * 4;
              *reinterpret_cast<unsigned*>(p.s.xqi + ((2 * kb) * TILE + t) * 16 + w4) = q.x;
              *reinterpret_cast<unsigned*>(p.s.xqi + ((2 * kb + 1) * TILE + t) * 16 + w4) = q.y;
            }
          }
        }
      }
    }
    c.sync();
  }
  c.tmark(TR_X0);
  float acc[GW][4];
  gemm<4, true>(
      c, sp, KS_GATE, acc, [&](int tt, int k) { return reinterpret_cast<const uint2*>(c.sm.bt + (tt * KS_GATE + k) * 256)[c.lane]; },
      [&](float (&a)[4], const uint8_t* stage, int q, int rt, const uint2& b) {
        const uint4 v = frag(stage, q * 4 + rt, c.lane);
        bmma16(a, v.x, v.z, v.y, v.w, b.x, b.y);
      });
  c.tmark(TR_X1);
  if (DS41H_DECODER_GSUM2 != 0 && coarse_m(p.m)) {  // GSUM2: this K part's partial logits, plain stores into slot kp
    float* dst = p.s.gpart + long(kp) * COARSE_M * EXPERTS;
    stores(c, sp, acc, [&](int rt, int tt, const float (&a)[4]) {
      const float4 v = transpose4(a, c.lane);
      const int tok = quad_col(tt, c.lane);
      if (tok < sp.count) __stcg(reinterpret_cast<float4*>(dst + long(sp.first + tok) * EXPERTS + rb * 64 + rt * 16 + quad_row(c.lane)), v);
    });
  } else {
    stores(c, sp, acc, [&](int rt, int tt, const float (&a)[4]) { gscatter(p, c.epoch, rb * 64 + rt * 16, sp, tt, a, c.lane); });
  }
  c.tmark(TR_X2);
  if (!split && rb == 0) {
    if (COARSE && coarse_m(p.m)) asm volatile("fence.proxy.async.global;" ::: "memory");  // xqi / xhi are read by bulk copies
    c.signal(C_XK + k0 / 512 * GROUPS + tg);
  }
}

__device__ __noinline__ void route_all_nc(const Params& p, Smem& sm, int tid, unsigned epoch);
__device__ __forceinline__ void route_all(Ctx& c) { route_all_nc(c.p, c.sm, c.tid, c.epoch); }

__device__ __noinline__ void route_block_nc(const Params& p, Smem& sm, int tid, unsigned epoch, int rb);
__device__ __forceinline__ void route_block(Ctx& c, int rb) { route_block_nc(c.p, c.sm, c.tid, c.epoch, rb); }

__device__ __noinline__ void route_fast_nc(const Params& p, Smem& sm, int tid, unsigned epoch);
__device__ __noinline__ void route_lean_nc(const Params& p, Smem& sm, int tid, unsigned epoch);
__device__ __noinline__ void route1_nc(const Params& p, Smem& sm, int tid, unsigned epoch);

// Hierarchical routing: the gate job finishing a 64-expert row block (all K parts, all token groups)
// computes the block's per-token top-6 candidates; the last of the GATE_BLOCKS block finishers
// merges them and builds the routing table (route_all). Exact: the global top-6 (ties -> lowest
// expert) is the top-6 of the union of the blocks' top-6.
__device__ CTXFN void gate_clears(Ctx& c, int rb, int kp, int tg);

__device__ CTXFN void job_gate(Ctx& c, int rb, int kp, int tg) {
  gate_group(c, rb, kp, tg);
  c.drop(KS_GATE / 4);
  if (DS41H_DECODER_GSUM2 != 0 && DS41H_DECODER_ROUTE1 != 0 && !FROUTE && !PROUTE && coarse_m(c.p.m)) {
    // GSUM2: the last K part of a 64-expert block sums its block in K-part order, the last block routes
    if (c.arrive(C_GB + rb, KP_GATE * token_groups(c.p.m))) {
      c.tmark(TR_RB);
      const Params& p = c.p;
      const float* src = p.s.gpart;
      float* glog = p.s.gpart + long(KP_GATE) * COARSE_M * EXPERTS;
      for (int i = c.tid; i < p.m * 64; i += GT) {  // (token, expert of the block); KP_GATE loads in flight per item
        const long off = long(i >> 6) * EXPERTS + rb * 64 + (i & 63);
        float part[KP_GATE];
#pragma unroll
        for (int k = 0; k < KP_GATE; ++k) part[k] = __ldcg(src + long(k) * COARSE_M * EXPERTS + off);
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < KP_GATE; ++k) sum += part[k];
        __stcg(glog + off, sum);
      }
      c.tmark(TR_RBE);
      if (c.arrive(C_G, GATE_BLOCKS)) route1_nc(c.p, c.sm, c.tid, c.epoch);
    }
    return gate_clears(c, rb, kp, tg);
  }
  if (DS41H_DECODER_ROUTE1 != 0 && !FROUTE && !PROUTE && coarse_m(c.p.m)) {  // ROUTE1: one hop, then the clears
    if (c.arrive(C_G, GATE_BLOCKS * KP_GATE * token_groups(c.p.m))) {
      c.tmark(TR_RB);
      route1_nc(c.p, c.sm, c.tid, c.epoch);
    }
    return gate_clears(c, rb, kp, tg);
  }
  if (FROUTE && coarse_m(c.p.m)) {  // one arrival per gate job; the last one routes, then everyone clears
    if (c.arrive(C_G, GATE_BLOCKS * KP_GATE * token_groups(c.p.m))) {
      c.tmark(TR_RB);
      route_fast_nc(c.p, c.sm, c.tid, c.epoch);
    }
    return gate_clears(c, rb, kp, tg);
  }
  if constexpr (DS41H_DECODER_CLEAR_LATE != 0 && HIER_ROUTE && !PROUTE) {  // arrive first, clear afterwards
    if (c.arrive(C_GB + rb, KP_GATE * token_groups(c.p.m))) {
      c.tmark(TR_RB);
      route_block(c, rb);
      c.tmark(TR_RBE);
      if (c.arrive(C_G, GATE_BLOCKS)) (DS41H_DECODER_ROUTE_LEAN && coarse_m(c.p.m)) ? route_lean_nc(c.p, c.sm, c.tid, c.epoch) : route_all(c);
    }
    return gate_clears(c, rb, kp, tg);
  }
  gate_clears(c, rb, kp, tg);
  if (PROUTE && !PROUTE2 && coarse_m(c.p.m)) return c.signal(C_G);  // every CTA's producer routes (route_local)
  if constexpr (HIER_ROUTE) {
    if (!c.arrive(C_GB + rb, KP_GATE * token_groups(c.p.m))) return;
    c.tmark(TR_RB);
    route_block(c, rb);
    c.tmark(TR_RBE);
    if (PROUTE2 && coarse_m(c.p.m)) return c.signal(C_G);  // candidates published: the producers merge them
    if (c.arrive(C_G, GATE_BLOCKS)) (DS41H_DECODER_ROUTE_LEAN && coarse_m(c.p.m)) ? route_lean_nc(c.p, c.sm, c.tid, c.epoch) : route_all(c);
  } else if (c.arrive(C_G, GATE_BLOCKS * KP_GATE * token_groups(c.p.m))) {
    route_all(c);
  }
}

// zacc (FUSED_Z: every wo_b job has read it) and the other router-sum buffer (the next launch's) cleared in slices.
__device__ CTXFN void gate_clears(Ctx& c, int rb, int kp, int tg) {
  if constexpr (FUSED_Z) {  // every wo_b job has read zacc (gate deps: all OWNA slices <- all wo_b jobs)
    const int tgs = token_groups(c.p.m), jobs = GATE_BLOCKS * KP_GATE * tgs, id = (rb * KP_GATE + kp) * tgs + tg;
    const int n4 = c.p.m * OA / 4, per = cdiv(n4, jobs);
    float4* z = reinterpret_cast<float4*>(c.p.s.zacc);
    for (int i = id * per + c.tid; i < min(n4, (id + 1) * per); i += GT) __stcg(z + i, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  }
  {  // clear the other router-sum buffer (used by the previous launch, needed clean by the next one)
    const int tgs = token_groups(c.p.m), jobs = GATE_BLOCKS * KP_GATE * tgs, id = (rb * KP_GATE + kp) * tgs + tg;
    constexpr int n4 = MAXM * EXPERTS / (GDET ? 2 : 4);  // 16-byte words of the buffer
    const int per = cdiv(n4, jobs);
    float4* g = GDET ? reinterpret_cast<float4*>(gsum64(c.p, c.epoch + 1)) : reinterpret_cast<float4*>(gsum(c.p, c.epoch + 1));
    for (int i = id * per + c.tid; i < min(n4, (id + 1) * per); i += GT) __stcg(g + i, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  }
}

// ------------------------------------------------------------------------------ jobs: routing
template <int PER>
__device__ __forceinline__ void select_topk(const float (&score)[PER], float (&key)[PER], const int (&expert)[PER], int lane, float& chosen_key, float& chosen_score,
                                            int& chosen_expert) {
  for (int k = 0; k < TOPK; ++k) {
    float best = -INFINITY;
    int bi = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (key[i] > best) best = key[i], bi = i;
    int be_local = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (i == bi) be_local = expert[i];
    const unsigned ord = float_order(best);
    const unsigned top = __reduce_max_sync(0xffffffffu, ord);
    const int be = int(__reduce_min_sync(0xffffffffu, ord == top ? unsigned(be_local) : 0xffffffffu));
    const unsigned owner = __ffs(__ballot_sync(0xffffffffu, ord == top && be_local == be)) - 1;
    float sv = 0.0f;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (i == bi) sv = score[i];
    sv = __shfl_sync(0xffffffffu, sv, owner);
    const float kv = __shfl_sync(0xffffffffu, best, owner);
    if (lane == int(owner))
#pragma unroll
      for (int i = 0; i < PER; ++i)
        if (i == bi) key[i] = -INFINITY;
    if (lane == k) chosen_key = kv, chosen_score = sv, chosen_expert = be;
  }
}

// Like select_topk but with explicit (key desc, expert asc) order inside a lane too (candidates
// of the merge are not in expert order).
template <int PER>
__device__ __forceinline__ void select_topk_e(const float (&score)[PER], float (&key)[PER], const int (&expert)[PER], int lane, float& chosen_key, float& chosen_score,
                                              int& chosen_expert) {
  for (int k = 0; k < TOPK; ++k) {
    float best = -INFINITY;
    int bi = 0, bexp = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (key[i] > best || (key[i] == best && expert[i] < bexp)) best = key[i], bi = i, bexp = expert[i];
    const unsigned ord = float_order(best);
    const unsigned top = __reduce_max_sync(0xffffffffu, ord);
    const int be = int(__reduce_min_sync(0xffffffffu, ord == top ? unsigned(bexp) : 0xffffffffu));
    const unsigned owner = __ffs(__ballot_sync(0xffffffffu, ord == top && bexp == be)) - 1;
    float sv = 0.0f;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (i == bi) sv = score[i];
    sv = __shfl_sync(0xffffffffu, sv, owner);
    const float kv = __shfl_sync(0xffffffffu, best, owner);
    if (lane == int(owner))
#pragma unroll
      for (int i = 0; i < PER; ++i)
        if (i == bi) key[i] = -INFINITY;
    if (lane == k) chosen_key = kv, chosen_score = sv, chosen_expert = be;
  }
}

// Two tokens' select_topk_e interleaved: independent warp-reduction chains in one instruction
// stream (the per-round reduce / ballot / shuffle latency dominates at decode clocks).
template <int PER>
__device__ __forceinline__ void select_topk_e2(const float (&score)[2][PER], float (&key)[2][PER], const int (&expert)[2][PER], int lane, float (&ck)[2],
                                               float (&cs)[2], int (&ce)[2]) {
  // Register-only: the best score is tracked during the scan and taken slots are a bitmask (indexing
  // arrays with the runtime argmax compiled to local-memory loads).
  static_assert(PER <= 32);
  unsigned taken[2] = {0u, 0u};
  for (int k = 0; k < TOPK; ++k) {
    float best[2], bsc[2];
    int bi[2], bexp[2], be[2];
    unsigned ord[2], top[2], owner[2];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      best[t] = -INFINITY, bsc[t] = 0.0f, bi[t] = 0, bexp[t] = 0x7fffffff;
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        const bool better = !((taken[t] >> i) & 1u) && (key[t][i] > best[t] || (key[t][i] == best[t] && expert[t][i] < bexp[t]));
        best[t] = better ? key[t][i] : best[t];
        bsc[t] = better ? score[t][i] : bsc[t];
        bexp[t] = better ? expert[t][i] : bexp[t];
        bi[t] = better ? i : bi[t];
      }
      ord[t] = float_order(best[t]);
    }
#pragma unroll
    for (int t = 0; t < 2; ++t) top[t] = __reduce_max_sync(0xffffffffu, ord[t]);
#pragma unroll
    for (int t = 0; t < 2; ++t) be[t] = int(__reduce_min_sync(0xffffffffu, ord[t] == top[t] ? unsigned(bexp[t]) : 0xffffffffu));
#pragma unroll
    for (int t = 0; t < 2; ++t) owner[t] = __ffs(__ballot_sync(0xffffffffu, ord[t] == top[t] && bexp[t] == be[t])) - 1;
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const float sv = __shfl_sync(0xffffffffu, bsc[t], owner[t]);
      const float kv = __shfl_sync(0xffffffffu, best[t], owner[t]);
      if (lane == int(owner[t])) taken[t] |= 1u << bi[t];
      if (lane == k) ck[t] = kv, cs[t] = sv, ce[t] = be[t];
    }
  }
}

// Compact select (i-cache friendly): select_topk_e2's rounds as a runtime loop (#pragma unroll 1), so the ~100-
// instruction round body is fetched once. One-shot straight-line code in this ~0.6 MB kernel runs at instruction-
// fetch speed (the unrolled selects cost ~1 us per round in the producer warp).
template <int PER>
__device__ __forceinline__ void select_topk_loop(const float (&score)[2][PER], const float (&key)[2][PER], const int (&expert)[2][PER], int lane, float (&ck)[2],
                                                 float (&cs)[2], int (&ce)[2]) {
  static_assert(PER <= 32);
  unsigned taken[2] = {0u, 0u};
#pragma unroll 1
  for (int k = 0; k < TOPK; ++k) {
    float best[2], bsc[2];
    int bi[2], bexp[2];
    unsigned ord[2], top[2];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      best[t] = -INFINITY, bsc[t] = 0.0f, bi[t] = 0, bexp[t] = 0x7fffffff;
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        const bool better = !((taken[t] >> i) & 1u) && (key[t][i] > best[t] || (key[t][i] == best[t] && expert[t][i] < bexp[t]));
        best[t] = better ? key[t][i] : best[t];
        bsc[t] = better ? score[t][i] : bsc[t];
        bexp[t] = better ? expert[t][i] : bexp[t];
        bi[t] = better ? i : bi[t];
      }
      ord[t] = float_order(best[t]);
    }
#pragma unroll
    for (int t = 0; t < 2; ++t) top[t] = __reduce_max_sync(0xffffffffu, ord[t]);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int be = int(__reduce_min_sync(0xffffffffu, ord[t] == top[t] ? unsigned(bexp[t]) : 0xffffffffu));
      const unsigned owner = __ffs(__ballot_sync(0xffffffffu, ord[t] == top[t] && bexp[t] == be)) - 1;
      const float sv = __shfl_sync(0xffffffffu, bsc[t], owner);
      const float kv = __shfl_sync(0xffffffffu, best[t], owner);
      if (lane == int(owner)) taken[t] |= 1u << bi[t];
      if (lane == k) ck[t] = kv, cs[t] = sv, ce[t] = be;
    }
  }
}

// select_fast: top-6 of T tokens' candidates (PER per lane; keys with NaN already mapped to -1e30), order key desc, ties ->
// lowest expert (stock rule), result k (0..5) of token t in lane k: ck[t] / cs[t] / ce[t]. Each lane keeps its best
// untaken candidate (ordered key, index); a round = REDUX max over the lanes' bests (+ REDUX min of the experts on an
// exact tie), the owner lane marks it taken and rescans. Rounds are a rolled loop (small code: one-shot, i-cache cold).
template <int T, int PER>
__device__ __forceinline__ void select_fast(const float (&score)[T][PER], const float (&key)[T][PER], const int (&expert)[T][PER], int lane, float (&ck)[T],
                                            float (&cs)[T], int (&ce)[T]) {
  static_assert(PER <= 32);
  unsigned taken[T], bk[T];
  int bi[T];
  auto rescan = [&](int t) {  // best untaken candidate of this lane (strict > keeps the lowest index = lowest expert on ties)
    unsigned b = 0u;
    int j = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const unsigned v = (taken[t] >> i) & 1u ? 0u : float_order(key[t][i]);
      j = v > b ? i : j;
      b = v > b ? v : b;
    }
    bk[t] = b, bi[t] = j;
  };
#pragma unroll
  for (int t = 0; t < T; ++t) taken[t] = 0u, rescan(t);
#pragma unroll 1
  for (int k = 0; k < TOPK; ++k) {
    unsigned top[T], tie[T];
#pragma unroll
    for (int t = 0; t < T; ++t) top[t] = __reduce_max_sync(0xffffffffu, bk[t]);
#pragma unroll
    for (int t = 0; t < T; ++t) tie[t] = __ballot_sync(0xffffffffu, bk[t] == top[t]);
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float sv = 0.0f, kv = 0.0f;
      int ev = 0;
#pragma unroll
      for (int i = 0; i < PER; ++i)
        if (i == bi[t]) sv = score[t][i], kv = key[t][i], ev = expert[t][i];
      int owner = __ffs(tie[t]) - 1;
      if (tie[t] & (tie[t] - 1u)) {  // the max key on several lanes: lowest expert among them
        const unsigned me = __reduce_min_sync(0xffffffffu, bk[t] == top[t] ? unsigned(ev) : 0xffffffffu);
        owner = __ffs(__ballot_sync(0xffffffffu, bk[t] == top[t] && unsigned(ev) == me)) - 1;
      }
      sv = __shfl_sync(0xffffffffu, sv, owner);
      kv = __shfl_sync(0xffffffffu, kv, owner);
      ev = __shfl_sync(0xffffffffu, ev, owner);
      if (lane == k) ck[t] = kv, cs[t] = sv, ce[t] = ev;
      if (lane == owner) {
        taken[t] |= 1u << bi[t];
        rescan(t);
      }
    }
  }
}

// Per-token top-6 of row block rb's 64 experts (stock dsv4_topk scores / keys, see route_all).
// Routing runs in its own (non-inlined) register allocation; the job's Ctx stays in registers (a local
// Ctx here: these functions use no ring stages).
__device__ __noinline__ void route_block_nc(const Params& p, Smem& sm, int tid, unsigned epoch, int rb) {
  Ctx c{p, sm, tid >> 5, tid & 31, tid, 0, epoch};
  c.tmark(TR_X2);
  for (int t0 = c.warp; t0 < p.m; t0 += 2 * GW) {  // tokens t0 and t0 + GW interleaved
    float score[2][2], key[2][2], v[2][2];
    int expert[2][2];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int tok = t0 + t * GW;
        v[t][i] = tok < p.m ? gread(p, epoch, tok * EXPERTS + rb * 64 + c.lane + 32 * i) : 0.0f;
      }
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int tok = t0 + t * GW;
        expert[t][i] = rb * 64 + c.lane + 32 * i;
        if (tok < p.m) gzero(p, epoch, tok * EXPERTS + expert[t][i]);
        score[t][i] = sqrtf(v[t][i] > 20.0f ? v[t][i] : logf(1.0f + expf(v[t][i])));
        key[t][i] = score[t][i] + __ldg(p.w.gate_b + expert[t][i]);
        if (key[t][i] != key[t][i]) key[t][i] = -1e30f;
      }
    if constexpr (TRACE) {
      if (key[0][0] != 12345.0f) c.tmark(TR_S0);  // gate sums landed (data dependency keeps the mark after the loads)
    }
    float kk[2] = {-INFINITY, -INFINITY}, ks[2] = {0.0f, 0.0f};
    int ke[2] = {0, 0};
    select_topk_loop(score, key, expert, c.lane, kk, ks, ke);
    if constexpr (TRACE) {
      if (kk[0] != 12345.0f) c.tmark(TR_S1);
    }
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int tok = t0 + t * GW;
      if (c.lane < TOPK && tok < p.m) {
        const int i = (tok * GATE_BLOCKS + rb) * TOPK + c.lane;
        p.s.cand_key[i] = kk[t];
        p.s.cand_score[i] = ks[t];
        p.s.cand_exp[i] = ke[t];
      }
    }
    c.tmark(TR_S2);
  }
}

// Routing of all m tokens, run by the CTA whose row-block finisher arrives last (no per-token ROUTE
// jobs, no global atomics). Stock dsv4_topk (fused_moe/router/dsv4_topk.py:79-107) per token, one
// warp per token: s = sqrt(softplus(l)) (l > 20 -> l), top-6 of s + bias (NaN -> -1e30, ties ->
// lowest expert), w = s / sum(s) * 1.5 -- here the top-6 of the row blocks' candidates
// (route_block). The expert table is then built in shared memory (counting sort; entries in expert
// order, chunks of SPAN routes) and published with one release of C_ROUTE.
__device__ __noinline__ void route_all_nc(const Params& p, Smem& sm, int tid, unsigned epoch) {
  Ctx c{p, sm, tid >> 5, tid & 31, tid, 0, epoch};
  constexpr int PER = EXPERTS / 32, EPT = EXPERTS / GT;
  static_assert(EXPERTS % GT == 0 && ROUTES * 4 + 2 * EXPERTS * 4 + GW * 4 <= BT);
  int* cnt = reinterpret_cast<int*>(c.sm.bt);  // [EXPERTS] routes per expert
  int* rexp = cnt + EXPERTS;                    // [ROUTES] expert of route r
  int* wsum = rexp + ROUTES;                    // [GW] scan partials
  int* slot_of = wsum + GW;                     // [EXPERTS] first entry of each expert (coarse wtok table)
  c.tmark(TR_X0);
  for (int e = c.tid; e < EXPERTS; e += GT) cnt[e] = 0;
  (void)PER;
  // Candidates per token: HIER_ROUTE: the 36 row-block top-6 entries (2 slots per lane); else all 384
  // router logits (12 per lane: scores / keys computed here, gacc cleared).
  constexpr int NC = GATE_BLOCKS * TOPK;
  static_assert(NC <= 64);
  constexpr int PERR = HIER_ROUTE ? 2 : EXPERTS / 32;
  for (int t0 = c.warp; t0 < p.m; t0 += 2 * GW) {  // tokens t0 and t0 + GW interleaved
    float score[2][PERR], key[2][PERR];
    int expert[2][PERR];
    if constexpr (HIER_ROUTE) {
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < PERR; ++i) {
          const int q = c.lane + 32 * i, tok = t0 + t * GW;
          const bool live = q < NC && tok < p.m;
          key[t][i] = live ? __ldcg(p.s.cand_key + tok * NC + q) : -INFINITY;
          score[t][i] = live ? __ldcg(p.s.cand_score + tok * NC + q) : 0.0f;
          expert[t][i] = live ? __ldcg(p.s.cand_exp + tok * NC + q) : 0x7fffffff;
        }
    } else {
      float v[2][PERR];
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < PERR; ++i) {
          const int tok = t0 + t * GW;
          v[t][i] = tok < p.m ? gread(p, epoch, tok * EXPERTS + c.lane + 32 * i) : 0.0f;
        }
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < PERR; ++i) {
          const int tok = t0 + t * GW;
          expert[t][i] = c.lane + 32 * i;
          if (tok < p.m) gzero(p, epoch, tok * EXPERTS + expert[t][i]);
          score[t][i] = sqrtf(v[t][i] > 20.0f ? v[t][i] : logf(1.0f + expf(v[t][i])));
          key[t][i] = score[t][i] + __ldg(p.w.gate_b + expert[t][i]);
          if (key[t][i] != key[t][i]) key[t][i] = -1e30f;
        }
    }
    if constexpr (TRACE) {
      if (key[0][0] != 12345.0f) c.tmark(TR_S3);  // candidates landed
    }
    float kk[2] = {-INFINITY, -INFINITY}, ks[2] = {0.0f, 0.0f};
    int ke[2] = {0, 0};
    select_topk_loop(score, key, expert, c.lane, kk, ks, ke);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int tok = t0 + t * GW;
      float total = 0.0f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) total += __shfl_sync(0xffffffffu, ks[t], k);
      if (c.lane < TOPK && tok < p.m) {
        const int r = tok * TOPK + c.lane;
        p.s.topk_w[r] = ks[t] * (ROUTE_SCALE / (total > 0.0f ? total : 1.0f));
        p.s.topk_id[r] = ke[t];
        rexp[r] = ke[t];
      }
    }
  }
  c.sync();
  c.tmark(TR_X1);
  Table* tb = p.s.table;
  for (int r = c.tid; r < p.m * TOPK; r += GT) {
    const int e = rexp[r], pos = atomicAdd(cnt + e, 1);
    tb->routes[e * MAXM + pos] = r;
  }
  c.sync();
  // entry index of each expert's first chunk: exclusive scan of cdiv(cnt, SPAN), EPT experts per thread
  int n[EPT], local = 0;
#pragma unroll
  for (int i = 0; i < EPT; ++i) local += n[i] = cdiv(cnt[c.tid * EPT + i], SPAN);
  int incl = local;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_up_sync(0xffffffffu, incl, o);
    if (c.lane >= o) incl += u;
  }
  if (c.lane == 31) wsum[c.warp] = incl;
  c.sync();
  int base = incl - local;
  for (int w = 0; w < c.warp; ++w) base += wsum[w];
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int e = c.tid * EPT + i;
    if (n[i]) c.cs()[C_ECOUNT + e] = unsigned(cnt[e]);
    slot_of[e] = base;
    for (int ch = 0; ch < n[i]; ++ch, ++base) {
      tb->expert[base] = e;
      tb->chunk[base] = ch;
    }
  }
  if (c.tid == GT - 1) c.cs()[C_NENT] = unsigned(base);
  if (coarse_m(p.m)) {  // (slot, token) routing weights for the R2C epilogues (one entry per expert)
    for (int i = c.tid; i < MAXE * TILE; i += GT) p.s.wtok[i] = 0.0f;
    c.sync();
    for (int r = c.tid; r < p.m * TOPK; r += GT) p.s.wtok[slot_of[rexp[r]] * TILE + r / TOPK] = p.s.topk_w[r];
  }
  c.tmark(TR_X2);
  c.sync();  // bar.sync + red.release orders the CTA's table writes (as Ctx::signal)
  c.tmark(TR_RAE);
  // C_ROUTE = m + (entries << 16): the producers get the entry count with the same acquire.
  if (c.tid == 0) red_release(c.cs() + C_ROUTE, unsigned(p.m) + (__ldcg(c.cs() + C_NENT) << 16));
}

// FROUTE: routing of all m <= 8 tokens by the last gate job (see DS41H_DECODER_FROUTE). Warp w routes tokens w and
// w + GW (interleaved select, stock dsv4_topk math as route_all: s = sqrt(softplus(l)) (l > 20 -> l), key = s + bias,
// NaN -> -1e30, top-6 by key with ties -> lowest expert, w = s * (1.5 / sum)); warp 0 then builds the expert slots
// (distinct experts, ascending) and the per-(slot, token) weights. This launch's router buffer was cleared by the
// previous launch's gate jobs, so nothing is zeroed here.
__device__ __noinline__ void route_fast_nc(const Params& p, Smem& sm, int tid, unsigned epoch) {
  Ctx c{p, sm, tid >> 5, tid & 31, tid, 0, epoch};
  constexpr int PER = EXPERTS / 32;
  const int m = p.m;
  if (c.tid < EXPERTS / 32) sm.emask[c.tid] = 0u;
  const int t0 = c.warp, t1 = c.warp + GW;
  {
    unsigned long long raw[2][PER];
    float bias[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      raw[0][i] = t0 < m ? graw(p, epoch, t0 * EXPERTS + c.lane + 32 * i) : 0ull;
      raw[1][i] = t1 < m ? graw(p, epoch, t1 * EXPERTS + c.lane + 32 * i) : 0ull;
      bias[i] = __ldg(p.w.gate_b + c.lane + 32 * i);
    }
    float score[2][PER], key[2][PER];
    int expert[2][PER];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        const float v = gval(raw[t][i]);
        expert[t][i] = c.lane + 32 * i;
        score[t][i] = sqrtf(v > 20.0f ? v : logf(1.0f + expf(v)));
        key[t][i] = score[t][i] + bias[i];
        if (key[t][i] != key[t][i]) key[t][i] = -1e30f;
      }
    float kk[2] = {-INFINITY, -INFINITY}, ks[2] = {0.0f, 0.0f};
    int ke[2] = {0, 0};
    if (t0 < m) {
      if constexpr (DS41H_DECODER_FROUTE == 2) select_fast<2, PER>(score, key, expert, c.lane, kk, ks, ke);
      else select_topk_loop(score, key, expert, c.lane, kk, ks, ke);
    }
    c.sync();  // emask cleared
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int tok = t ? t1 : t0;
      float total = 0.0f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) total += __shfl_sync(0xffffffffu, ks[t], k);
      if (c.lane < TOPK && tok < m) {
        const int r = tok * TOPK + c.lane;
        const float w = ks[t] * (ROUTE_SCALE / (total > 0.0f ? total : 1.0f));
        p.s.topk_w[r] = w;
        p.s.topk_id[r] = ke[t];
        sm.rsel_e[r] = ke[t];
        sm.rsel_w[r] = w;
        atomicOr(sm.emask + (ke[t] >> 5), 1u << (ke[t] & 31));
      }
    }
  }
  c.tmark(TR_RBE);
  c.sync();
  if (c.warp == 0) {  // slots: exclusive prefix of the mask words' popcounts, every set bit -> its slot
    const unsigned word = c.lane < EXPERTS / 32 ? sm.emask[c.lane] : 0u;
    int incl = __popc(word);
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, o);
      if (c.lane >= o) incl += u;
    }
    const int n = __shfl_sync(0xffffffffu, incl, 31);
    if (c.lane < EXPERTS / 32) {
      int slot = incl - __popc(word);
      sm.ebase[c.lane] = slot;
      for (unsigned w = word; w; w &= w - 1) p.s.table->expert[slot++] = 32 * c.lane + (__ffs(w) - 1);
    }
    for (int i = c.lane; i < MAXE * TILE; i += 32) p.s.wtok[i] = 0.0f;
    __syncwarp();
    for (int r = c.lane; r < m * TOPK; r += 32) {
      const int e = sm.rsel_e[r], slot = sm.ebase[e >> 5] + __popc(sm.emask[e >> 5] & ((1u << (e & 31)) - 1u));
      p.s.wtok[slot * TILE + r / TOPK] = sm.rsel_w[r];
    }
    if (c.lane == 0) {
      c.cs()[C_NENT] = unsigned(n);
      sm.nent = n;
    }
  }
  c.sync();  // bar.sync + red.release orders the CTA's table writes (as Ctx::signal)
  c.tmark(TR_RAE);
  if (c.tid == 0) red_release(c.cs() + C_ROUTE, unsigned(m) + (unsigned(sm.nent) << 16));
}

// ROUTE_LEAN: see the knob. Warp w merges the candidates of tokens w, w + GW (route_all's loads / select / weights), then
// warp 0 builds the slots (distinct experts, ascending) and wtok; one release of C_ROUTE (m + slots << 16).
__device__ __noinline__ void route_lean_nc(const Params& p, Smem& sm, int tid, unsigned epoch) {
  Ctx c{p, sm, tid >> 5, tid & 31, tid, 0, epoch};
  constexpr int NC = GATE_BLOCKS * TOPK;
  static_assert(HIER_ROUTE && NC <= 64);
  const int m = p.m;
  c.tmark(TR_X0);
  if (c.tid < EXPERTS / 32) sm.emask[c.tid] = 0u;
  c.sync();
  for (int t0 = c.warp; t0 < m; t0 += 2 * GW) {  // tokens t0 and t0 + GW
    float score[2][2], key[2][2];
    int expert[2][2];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int q = c.lane + 32 * i, tok = t0 + t * GW;
        const bool live = q < NC && tok < m;
        key[t][i] = live ? __ldcg(p.s.cand_key + tok * NC + q) : -INFINITY;
        score[t][i] = live ? __ldcg(p.s.cand_score + tok * NC + q) : 0.0f;
        expert[t][i] = live ? __ldcg(p.s.cand_exp + tok * NC + q) : 0x7fffffff;
      }
    float kk[2] = {-INFINITY, -INFINITY}, ks[2] = {0.0f, 0.0f};
    int ke[2] = {0, 0};
    select_topk_loop(score, key, expert, c.lane, kk, ks, ke);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int tok = t0 + t * GW;
      float total = 0.0f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) total += __shfl_sync(0xffffffffu, ks[t], k);
      if (c.lane < TOPK && tok < m) {
        const int r = tok * TOPK + c.lane;
        const float w = ks[t] * (ROUTE_SCALE / (total > 0.0f ? total : 1.0f));
        p.s.topk_w[r] = w;
        p.s.topk_id[r] = ke[t];
        sm.rsel_e[r] = ke[t];
        sm.rsel_w[r] = w;
        atomicOr(sm.emask + (ke[t] >> 5), 1u << (ke[t] & 31));
      }
    }
  }
  c.tmark(TR_X1);
  c.sync();
  if (c.warp == 0) {  // slots: exclusive prefix of the mask words' popcounts, every set bit -> its slot
    const unsigned word = c.lane < EXPERTS / 32 ? sm.emask[c.lane] : 0u;
    int incl = __popc(word);
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, o);
      if (c.lane >= o) incl += u;
    }
    const int n = __shfl_sync(0xffffffffu, incl, 31);
    if (c.lane < EXPERTS / 32) {
      int slot = incl - __popc(word);
      sm.ebase[c.lane] = slot;
      for (unsigned w = word; w; w &= w - 1) p.s.table->expert[slot++] = 32 * c.lane + (__ffs(w) - 1);
    }
    for (int i = c.lane; i < MAXE * TILE; i += 32) p.s.wtok[i] = 0.0f;
    __syncwarp();
    for (int r = c.lane; r < m * TOPK; r += 32) {
      const int e = sm.rsel_e[r], slot = sm.ebase[e >> 5] + __popc(sm.emask[e >> 5] & ((1u << (e & 31)) - 1u));
      p.s.wtok[slot * TILE + r / TOPK] = sm.rsel_w[r];
    }
    if (c.lane == 0) {
      c.cs()[C_NENT] = unsigned(n);
      sm.nent = n;
    }
  }
  c.tmark(TR_X2);
  c.sync();  // bar.sync + red.release orders the CTA's table writes (as Ctx::signal)
  c.tmark(TR_RAE);
  if (c.tid == 0) red_release(c.cs() + C_ROUTE, unsigned(m) + (unsigned(sm.nent) << 16));
}

// ROUTE1 (see the knob): routing of all m <= 8 tokens in one CTA, one hop after the last gate job.
// PFR1: TMA L2 prefetches (no smem, no barrier) of the first PFR1 stages + scale group 0 of every R1F row block of the
// n routed experts (sm.pf_expert), spread over the router CTA's consumer threads.
__device__ __forceinline__ void tma3d_pf(const CUtensorMap* map, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2)
               : "memory");
}
__device__ __forceinline__ void tma4d_pf(const CUtensorMap* map, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" ::"l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1),
               "r"(c2), "r"(c3)
               : "memory");
}
__device__ __forceinline__ void prefetch_r1f(const Params& p, const Smem& sm, int tid, int n) {
  constexpr int PF = DS41H_DECODER_PFR1, PER = R1F_BLOCKS * (PF + 1);  // per expert: 9 blocks x (PF stages + scale group 0)
  if constexpr (R1SK) {  // the first PF stages (+ their first scale group) of every stream-K job
    const int U = n * R1F_BLOCKS * R1F_STAGES, T = r1sk_T(n, int(gridDim.x)), J = (U + T - 1) / T;
    for (int i = tid; i < J * (PF + 1); i += GT) {
      const int jb = i / (PF + 1), k = i % (PF + 1), u = jb * T + (k < PF ? k : 0);
      if (u >= U) continue;
      const int g = u / R1F_STAGES, e = sm.pf_expert[g / R1F_BLOCKS], b = g % R1F_BLOCKS, kc = u % R1F_STAGES;
      if (k < PF) {
        tma4d_pf(&p.w.tm_w13f, 0, 64 * b, kc, e);
        tma4d_pf(&p.w.tm_w13f, 0, INTER + 64 * b, kc, e);
      } else {
        tma3d_pf(&p.w.tm_s13f, R1SKU ? kc * 4 : (kc >> 2) * 16, 64 * b, e);
        tma3d_pf(&p.w.tm_s13f, R1SKU ? kc * 4 : (kc >> 2) * 16, INTER + 64 * b, e);
      }
    }
    return;
  }
  for (int i = tid; i < n * PER; i += GT) {
    const int e = sm.pf_expert[i / PER], j = i % PER, b = j % R1F_BLOCKS, k = j / R1F_BLOCKS;  // k = PF: scale group 0
    if (k < PF) {
      tma4d_pf(&p.w.tm_w13f, 0, 64 * b, k, e);
      tma4d_pf(&p.w.tm_w13f, 0, INTER + 64 * b, k, e);
    } else {
      tma3d_pf(&p.w.tm_s13f, 0, 64 * b, e);
      tma3d_pf(&p.w.tm_s13f, 0, INTER + 64 * b, e);
    }
  }
}

__device__ __noinline__ void route1_nc(const Params& p, Smem& sm, int tid, unsigned epoch) {
  Ctx c{p, sm, tid >> 5, tid & 31, tid, 0, epoch};
  constexpr int PER = EXPERTS / 32, CMAX = TOPK * PER;  // <= 6 lanes hold keys >= T, 12 each
  static_assert(GW * 2 * CMAX * 12 + EXPERTS * 4 <= BT, "ROUTE1 candidate lists fit bt");
  const int m = p.m, lane = c.lane;
  unsigned long long* cc = reinterpret_cast<unsigned long long*>(c.sm.bt) + c.warp * 2 * CMAX;  // [2 tokens][CMAX] composite keys
  float* cs = reinterpret_cast<float*>(c.sm.bt + GW * 2 * CMAX * 8) + c.warp * 2 * CMAX;       // [2 tokens][CMAX] scores
  c.tmark(TR_X0);
  if (c.tid < EXPERTS / 32) sm.emask[c.tid] = 0u;
  float bias[PER];
  unsigned long long raw[2][PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) bias[i] = __ldg(p.w.gate_b + lane + 32 * i);
#pragma unroll
  for (int t = 0; t < 2; ++t)
#pragma unroll
    for (int i = 0; i < PER; ++i)
      raw[t][i] = c.warp + GW * t >= m ? 0ull
                  : DS41H_DECODER_GSUM2 != 0
                      ? __float_as_uint(__ldcg(p.s.gpart + long(KP_GATE) * COARSE_M * EXPERTS + (c.warp + GW * t) * EXPERTS + lane + 32 * i))
                      : graw(p, epoch, (c.warp + GW * t) * EXPERTS + lane + 32 * i);
  unsigned long long comp[2][PER], lm[2];
  float score[2][PER];
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    lm[t] = 0ull;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const float v = DS41H_DECODER_GSUM2 != 0 ? __uint_as_float(unsigned(raw[t][i])) : gval(raw[t][i]);
      score[t][i] = sqrtf(v > 20.0f ? v : logf(1.0f + expf(v)));
      float key = score[t][i] + bias[i];
      if (key != key) key = -1e30f;
      if (key == 0.0f) key = 0.0f;  // -0 -> +0 (float compare semantics of route_all)
      comp[t][i] = (static_cast<unsigned long long>(float_order(key)) << 32) | (0xffffffffu - unsigned(lane + 32 * i));
      lm[t] = comp[t][i] > lm[t] ? comp[t][i] : lm[t];
    }
  }
  if constexpr (TRACE) {
    if (lm[0] != 12345ull) c.tmark(TR_S0);  // logits landed, keys done
  }
  // T = 6th largest lane maximum (u64 warp max as hi / lo REDUX), both tokens interleaved
  unsigned long long th[2] = {0ull, 0ull};
#pragma unroll 1
  for (int r = 0; r < TOPK; ++r) {
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const unsigned hi = __reduce_max_sync(0xffffffffu, unsigned(lm[t] >> 32));
      const unsigned lo = __reduce_max_sync(0xffffffffu, unsigned(lm[t] >> 32) == hi ? unsigned(lm[t]) : 0u);
      th[t] = (static_cast<unsigned long long>(hi) << 32) | lo;
      if (lm[t] == th[t]) lm[t] = 0ull;
    }
  }
  // candidates >= T -> per-warp shared lists (exclusive warp prefix of the per-lane counts)
  int ncand[2];
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    int cnt = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i) cnt += comp[t][i] >= th[t];
    int incl = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += u;
    }
    ncand[t] = __shfl_sync(0xffffffffu, incl, 31);
    int off = t * CMAX + incl - cnt;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (comp[t][i] >= th[t]) cc[off] = comp[t][i], cs[off] = score[t][i], ++off;
  }
  __syncwarp();
  if constexpr (TRACE) {
    if (ncand[0] != 12345) c.tmark(TR_S1);  // threshold + candidates
  }
  // rank by counting: candidate j's rank = candidates with a larger composite key (all distinct)
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int tok = c.warp + GW * t;
    if (tok >= m) continue;
    for (int j = lane; j < ncand[t]; j += 32) {
      const unsigned long long mine = cc[t * CMAX + j];
      int rank = 0;
#pragma unroll 4
      for (int q = 0; q < ncand[t]; ++q) rank += cc[t * CMAX + q] > mine;
      if (rank < TOPK) sm.rsel_e[tok * TOPK + rank] = int(0xffffffffu - unsigned(mine)), sm.rsel_w[tok * TOPK + rank] = cs[t * CMAX + j];
    }
  }
  __syncwarp();
  c.sync();  // emask cleared (thread < 12) before the atomicOr below
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int tok = c.warp + GW * t;
    if (tok >= m) continue;  // warp-uniform
    const float ks = lane < TOPK ? sm.rsel_w[tok * TOPK + lane] : 0.0f;
    const int ke = lane < TOPK ? sm.rsel_e[tok * TOPK + lane] : 0;
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) total += __shfl_sync(0xffffffffu, ks, k);
    __syncwarp();
    if (lane < TOPK) {
      const int r = tok * TOPK + lane;
      const float w = ks * (ROUTE_SCALE / (total > 0.0f ? total : 1.0f));
      p.s.topk_w[r] = w;
      p.s.topk_id[r] = ke;
      sm.rsel_w[r] = w;
      atomicOr(sm.emask + (ke >> 5), 1u << (ke & 31));
    }
  }
  c.tmark(TR_X1);
  c.sync();
  if (c.warp == 0) {  // slots (distinct experts ascending), wtok, C_NENT -- as route_lean_nc
    const unsigned word = lane < EXPERTS / 32 ? sm.emask[lane] : 0u;
    int incl = __popc(word);
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += u;
    }
    const int n = __shfl_sync(0xffffffffu, incl, 31);
    if (lane < EXPERTS / 32) {
      int slot = incl - __popc(word);
      sm.ebase[lane] = slot;
      for (unsigned w = word; w; w &= w - 1) {
        const int e = 32 * lane + (__ffs(w) - 1);
        if constexpr (DS41H_DECODER_PFR1 != 0) sm.pf_expert[slot] = e;
        if constexpr (DS41H_DECODER_TAGX != 0)
          p.s.xtag[slot] = (static_cast<unsigned long long>(epoch) << 32) | unsigned(e);
        p.s.table->expert[slot++] = e;
      }
    }
    for (int i = lane; i < MAXE * TILE; i += 32) p.s.wtok[i] = 0.0f;
    __syncwarp();
    for (int r = lane; r < m * TOPK; r += 32) {
      const int e = sm.rsel_e[r], slot = sm.ebase[e >> 5] + __popc(sm.emask[e >> 5] & ((1u << (e & 31)) - 1u));
      p.s.wtok[slot * TILE + r / TOPK] = sm.rsel_w[r];
    }
    if (lane == 0) {
      c.cs()[C_NENT] = unsigned(n);
      sm.nent = n;
    }
  }
  c.tmark(TR_X2);
  c.sync();  // bar.sync + red.release orders the CTA's table writes (as Ctx::signal)
  c.tmark(TR_RAE);
  if (c.tid == 0) red_release(c.cs() + C_ROUTE, unsigned(m) + (unsigned(sm.nent) << 16));
  if constexpr (DS41H_DECODER_PFR1 != 0 && LC) prefetch_r1f(p, sm, c.tid, sm.nent);
}

// PROUTE top-6: each lane sorts its 12 candidates (key desc, expert asc = stock's tie rule) with a
// selection network (Batcher odd-even merge for 16 restricted to 12 inputs, pruned to the 6 leading outputs, 38
// comparators, verified with the 0-1 principle), then 6 rounds of warp argmax over the lanes' heads (REDUX max on
// the ordered key, REDUX min on the expert among equal keys); the winning lane pops its head. Branch-free; replaces
// select_topk_e2's 6 x 12-element scans (whose branchy code cost ~1 us per round in the producer warp).
__device__ __forceinline__ void rcex(float (&k)[TOPK * 2], float (&s)[TOPK * 2], int (&e)[TOPK * 2], int a, int b) {
  const bool sw = k[b] > k[a] || (k[b] == k[a] && e[b] < e[a]);
  const float ka = k[a], sa = s[a];
  const int ea = e[a];
  k[a] = sw ? k[b] : ka, s[a] = sw ? s[b] : sa, e[a] = sw ? e[b] : ea;
  k[b] = sw ? ka : k[b], s[b] = sw ? sa : s[b], e[b] = sw ? ea : e[b];
}

template <int T>
__device__ __forceinline__ void top6_sorted(float (&k)[T][TOPK * 2], float (&s)[T][TOPK * 2], int (&e)[T][TOPK * 2], int lane, float (&ck)[T], float (&cs)[T],
                                            int (&ce)[T]) {
  static_assert(EXPERTS / 32 == TOPK * 2, "12 candidates per lane");
  constexpr int NET[38][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {8, 10}, {9, 11}, {1, 2}, {5, 6}, {9, 10}, {0, 4}, {1, 5}, {2, 6}, {3, 7}, {2, 4}, {3, 5}, {1, 2}, {3, 4}, {5, 6}, {9, 10}, {0, 8}, {1, 9}, {2, 10}, {3, 11}, {4, 8}, {5, 9}, {6, 10}, {2, 4}, {3, 5}, {6, 8}, {1, 2}, {3, 4}, {5, 6}};
#pragma unroll
  for (int c = 0; c < 38; ++c)
#pragma unroll
    for (int t = 0; t < T; ++t) rcex(k[t], s[t], e[t], NET[c][0], NET[c][1]);
#pragma unroll
  for (int r = 0; r < TOPK; ++r) {
    // warp argmax of the heads by a 5-step shuffle butterfly on (ordered key, ~expert): the max is unique (experts
    // differ) and every lane ends with it; the owner of expert x is lane x & 31 (expert = lane + 32 i).
    unsigned hk[T], he[T];
    int be[T];
    bool win[T];
#pragma unroll
    for (int t = 0; t < T; ++t) hk[t] = float_order(k[t][0]), he[t] = ~unsigned(e[t][0]);
#pragma unroll
    for (int o = 16; o; o >>= 1)
#pragma unroll
      for (int t = 0; t < T; ++t) {
        const unsigned ok = __shfl_xor_sync(0xffffffffu, hk[t], o), oe = __shfl_xor_sync(0xffffffffu, he[t], o);
        const bool take = ok > hk[t] || (ok == hk[t] && oe > he[t]);
        hk[t] = take ? ok : hk[t];
        he[t] = take ? oe : he[t];
      }
#pragma unroll
    for (int t = 0; t < T; ++t) {
      be[t] = int(~he[t]);
      win[t] = e[t][0] == be[t];
      const int owner = be[t] & 31;
      const float sv = __shfl_sync(0xffffffffu, s[t][0], owner), kv = __shfl_sync(0xffffffffu, k[t][0], owner);
      if (lane == r) ck[t] = kv, cs[t] = sv, ce[t] = be[t];
    }
#pragma unroll
    for (int t = 0; t < T; ++t)  // the winner pops its head (only the leading TOPK entries can still matter)
#pragma unroll
      for (int j = 0; j < TOPK - 1; ++j) {
        k[t][j] = win[t] ? k[t][j + 1] : k[t][j];
        s[t][j] = win[t] ? s[t][j + 1] : s[t][j];
        e[t][j] = win[t] ? e[t][j + 1] : e[t][j];
      }
#pragma unroll
    for (int t = 0; t < T; ++t) k[t][TOPK - 1] = win[t] ? -INFINITY : k[t][TOPK - 1];
  }
}

// PROUTE (LC, coarse m <= 8): routing by every CTA's producer warp. Waits for all gate jobs (C_G), reads the m
// tokens' router sums, takes the top-6 of sqrt(softplus(l)) + bias per token exactly like route_all (ties ->
// lowest expert, NaN keys -> -1e30, w = s * (1.5 / sum)), and builds in shared memory: the expert slots
// (distinct experts in ascending order, slot_expert), the per-(slot, token) weights (wtok_s, 0 = not routed)
// and nent. CTA 0 also writes topk_id / topk_w / C_NENT (tests, trace). Returns the slot count.
__device__ __noinline__ int route_local_nc(const Params& p, Smem& sm, int lane) {
  constexpr int PER = EXPERTS / 32, W = EXPERTS / 32;
  const int m = p.m;
#if DS41H_DECODER_TRACE
  if (lane == 0) sm.rt_in = unsigned(gtime() - sm.t0);
#endif
  if (lane == 0)
    poll_ge(p.s.counters + C_G, unsigned(PROUTE2 ? GATE_BLOCKS : GATE_BLOCKS * KP_GATE));
  __syncwarp();
#if DS41H_DECODER_TRACE
  if (lane == 0) sm.rt_cg = unsigned(gtime() - sm.t0);
#endif
  const unsigned ep = ld_volatile(p.s.epoch) + 1;  // this launch's buffer (read after pdl_wait)
#if DS41H_DECODER_TRACE
  if (lane == 0) sm.rt_ep = unsigned(gtime() - sm.t0);
#endif
  if (lane < W) sm.emask[lane] = 0u;
  for (int i = lane; i < MAXE * TILE; i += 32) sm.wtok_s[i] = 0.0f;
  if constexpr (PROUTE2) {
    __syncwarp();
    constexpr int NC = GATE_BLOCKS * TOPK;
    static_assert(HIER_ROUTE && NC <= 64);
#pragma unroll 1
    for (int t0 = 0; t0 < m; t0 += 2) {  // tokens t0, t0 + 1: route_all's merge (same loads, select, weights)
      float score[2][2], key[2][2];
      int expert[2][2];
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int q = lane + 32 * i, tok = t0 + t;
          const bool live = q < NC && tok < m;
          key[t][i] = live ? __ldcg(p.s.cand_key + tok * NC + q) : -INFINITY;
          score[t][i] = live ? __ldcg(p.s.cand_score + tok * NC + q) : 0.0f;
          expert[t][i] = live ? __ldcg(p.s.cand_exp + tok * NC + q) : 0x7fffffff;
        }
      float kk[2] = {-INFINITY, -INFINITY}, ks[2] = {0.0f, 0.0f};
      int ke[2] = {0, 0};
      select_topk_loop(score, key, expert, lane, kk, ks, ke);
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const int tok = t0 + t;
        float total = 0.0f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) total += __shfl_sync(0xffffffffu, ks[t], k);
        if (lane < TOPK && tok < m) {
          const int r = tok * TOPK + lane;
          sm.rsel_e[r] = ke[t];
          sm.rsel_w[r] = ks[t] * (ROUTE_SCALE / (total > 0.0f ? total : 1.0f));
          atomicOr(sm.emask + (ke[t] >> 5), 1u << (ke[t] & 31));
        }
      }
    }
  } else {
  float bias[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) bias[i] = __ldg(p.w.gate_b + lane + 32 * i);
  __syncwarp();
#pragma unroll 1
  for (int t0 = 0; t0 < m; t0 += 2) {  // tokens t0, t0 + 1 interleaved
    float score[2][PER], key[2][PER];
    int expert[2][PER];
    unsigned long long raw[2][PER];
#if DS41H_DECODER_TRACE
    unsigned long long ta = gtime();
#endif
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int i = 0; i < PER; ++i) raw[t][i] = graw(p, ep, min(t0 + t, m - 1) * EXPERTS + lane + 32 * i);
#if DS41H_DECODER_TRACE
    {
      unsigned long long acc = 0;
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < PER; ++i) acc += raw[t][i];
      asm volatile("" ::"l"(acc));  // the loads have landed
      const unsigned long long tb = gtime();
      if (lane == 0) sm.rt_ld += unsigned(tb - ta);
      ta = tb;
    }
#endif
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        const int tok = t0 + t;
        const float v = tok < m ? gval(raw[t][i]) : 0.0f;
        expert[t][i] = lane + 32 * i;
        score[t][i] = sqrtf(v > 20.0f ? v : logf(1.0f + expf(v)));
        key[t][i] = score[t][i] + bias[i];
        if (key[t][i] != key[t][i]) key[t][i] = -1e30f;
      }
#if DS41H_DECODER_TRACE
    {
      float acc = 0.0f;
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < PER; ++i) acc += key[t][i];
      asm volatile("" ::"f"(acc));
      if (lane == 0) sm.rt_sc += unsigned(gtime() - ta);
    }
#endif
    float kk[2] = {-INFINITY, -INFINITY}, ks[2] = {0.0f, 0.0f};
    int ke[2] = {0, 0};
    select_topk_loop(score, key, expert, lane, kk, ks, ke);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int tok = t0 + t;
      float total = 0.0f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) total += __shfl_sync(0xffffffffu, ks[t], k);
      if (lane < TOPK && tok < m) {
        const int r = tok * TOPK + lane;
        sm.rsel_e[r] = ke[t];
        sm.rsel_w[r] = ks[t] * (ROUTE_SCALE / (total > 0.0f ? total : 1.0f));
        atomicOr(sm.emask + (ke[t] >> 5), 1u << (ke[t] & 31));
      }
    }
  }
  }  // PROUTE2 / full PROUTE
  __syncwarp();
#if DS41H_DECODER_TRACE
  if (lane == 0) sm.rt_sel = unsigned(gtime() - sm.t0);
#endif
  // slots: exclusive prefix of the mask words' popcounts (lanes 0..W-1), then every set bit -> its slot
  const unsigned word = lane < W ? sm.emask[lane] : 0u;
  int incl = __popc(word);
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += u;
  }
  const int n = __shfl_sync(0xffffffffu, incl, 31);
  if (lane < W) {
    int slot = incl - __popc(word);
    sm.ebase[lane] = slot;
    for (unsigned w = word; w; w &= w - 1) sm.slot_expert[slot++] = 32 * lane + (__ffs(w) - 1);
  }
  __syncwarp();
  for (int r = lane; r < m * TOPK; r += 32) {
    const int e = sm.rsel_e[r], slot = sm.ebase[e >> 5] + __popc(sm.emask[e >> 5] & ((1u << (e & 31)) - 1u));
    sm.wtok_s[slot * TILE + r / TOPK] = sm.rsel_w[r];
    if (blockIdx.x == 0) p.s.topk_id[r] = e, p.s.topk_w[r] = sm.rsel_w[r];
  }
  if (lane == 0) {
    sm.nent = n;
    if (blockIdx.x == 0) p.s.counters[C_NENT] = unsigned(n);
#if DS41H_DECODER_TRACE
    sm.rt_done = unsigned(gtime() - sm.t0);
#endif
  }
  __syncwarp();
  return n;
}

// ------------------------------------------------------------------------------ jobs: shared expert
__device__ CTXFN void load_xres(Ctx& c);

// SiluAndMulWithClamp on bf16 gate/up GEMM outputs (stock DeepseekV4MLP, model.py:186-190).
__device__ CTXFN void finish_sh1(Ctx& c, int b, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  constexpr int PER = OTOK / GW;
  const int col = b * 32 + c.lane;
  const int out = b * 32 + (c.lane & ~7) + p8_pos(c.lane & 7);
  float g[PER], u[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int t = c.warp + i * GW;
    const float* h = p.s.sacc + long(sp.first + t) * 2 * INTER + col;
    g[i] = t < sp.count ? __ldcg(h) : 0.0f;
    u[i] = t < sp.count ? __ldcg(h + INTER) : 0.0f;
  }
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int t = c.warp + i * GW, tok = sp.first + t;
    if (t >= sp.count) break;
    float* h = p.s.sacc + long(tok) * 2 * INTER + col;
    __stcg(h, 0.0f);
    __stcg(h + INTER, 0.0f);
    const float a = bf(swiglu(bf(g[i]), bf(u[i])));
    p.s.hs[long(tok) * INTER + out] = __float2half_rn(a);  // |a| <= 100: exact in f16
    if constexpr (DENSE_WG || LCQ) p.s.hsq[long(tok) * INTER + wg_pos(col)] = e4m3_byte(a * H_INV);
  }
  c.signal(C_HS + tg);
}

__device__ WG_INLINE void job_sh1(Ctx& c, int b, int kp, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  auto store = [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.sacc, 2 * INTER, (rt < 2 ? b * 32 : INTER + b * 32 - 32) + rt * 16, sp, tt, a, c.lane); };
  if constexpr (LCQ) {
    if (coarse_m(p.m)) {  // resident x tile (loaded here or by an earlier job of this CTA), RS-wgmma
      load_xres(c);
      dense_wg<false, KS_SH1, 1>(c, sp, p.w.xs, store, bt_smem_addr() + kp * KS_SH1 * 256);
      c.tmark(TR_X1);
      if (c.arrive(C_SA + b * GROUPS + tg, KP_SH1)) c.tmark(TR_X2), finish_sh1(c, b, tg);
      return;
    }
  }
  const bool wg = dense_wg_dispatch<false, KS_SH1>(
      c, sp, p.w.xs,
      [&](int nt) {
        fill_wg<false>(
            c, sp.count, KS_SH1, nt, [&](int tok) { return p.s.xq + long(sp.first + tok) * HID + kp * KS_SH1 * 32; }, [](int, int) { return 0.0f; });
      },
      store);
  if (!wg) {
    fill_q<false>(
        c, sp.count, KS_SH1, [&](int tok) { return reinterpret_cast<const uint8_t*>(p.s.xh + long(sp.first + tok) * HID + kp * KS_SH1 * 32); },
        [](int) -> const uint8_t* { return nullptr; });
    c.tmark(TR_X0);
    dense_q<false>(c, sp, KS_SH1, store);
  }
  c.tmark(TR_X1);
  if (c.arrive(C_SA + b * GROUPS + tg, KP_SH1)) c.tmark(TR_X2), finish_sh1(c, b, tg);
}

__device__ WG_INLINE void job_sh2(Ctx& c, int rb, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  auto store = [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.shacc, HID, rb * 64 + rt * 16, sp, tt, a, c.lane); };
  static_assert(KS_INTER % 4 == 2, "shared w2 wgmma path expects a 2-block K tail");
  if constexpr (LCQ) {
    if (coarse_m(p.m)) {  // bt holds the resident x tile: B tile in bt2, RS-wgmma
      fill_wg<false>(
          c, sp.count, KS_INTER, 1, [&](int tok) { return p.s.hsq + long(sp.first + tok) * INTER; }, [](int, int) { return 0.0f; }, c.sm.bt2);
      dense_wg<false, KS_INTER, 1>(c, sp, 1.0f / H_INV, store, bt2_smem_addr());
      c.signal(C_SF + rb / 2);
      return;
    }
  }
  const bool wg = dense_wg_dispatch<false, KS_INTER>(
      c, sp, 1.0f / H_INV,
      [&](int nt) {
        fill_wg<false>(c, sp.count, KS_INTER, nt, [&](int tok) { return p.s.hsq + long(sp.first + tok) * INTER; }, [](int, int) { return 0.0f; });
      },
      store);
  if (!wg) {
    fill_q<false>(
        c, sp.count, KS_INTER, [&](int tok) { return reinterpret_cast<const uint8_t*>(p.s.hs + long(sp.first + tok) * INTER); },
        [](int) -> const uint8_t* { return nullptr; });
    dense_q<false>(c, sp, KS_INTER, store);
  }
  c.signal(C_SF + rb / 2);
}

__device__ __forceinline__ float mix_reduce(const float* mixp, int tok, int lane) {
  constexpr int ROWS = SLICES / 4, BATCH = 5;
  const float4* src = reinterpret_cast<const float4*>(mixp + long(tok) * SLICES * MIXW) + (lane >> 3) * (MIXW / 4) + (lane & 7);
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
  for (int b = 0; b < ROWS; b += BATCH) {
    float4 v[BATCH];
#pragma unroll
    for (int i = 0; i < BATCH; ++i) v[i] = __ldcg(src + (b + i) * MIXW);
#pragma unroll
    for (int i = 0; i < BATCH; ++i) acc.x += v[i].x, acc.y += v[i].y, acc.z += v[i].z, acc.w += v[i].w;
  }
#pragma unroll
  for (int o = 8; o < 32; o <<= 1) {
    acc.x += __shfl_xor_sync(0xffffffffu, acc.x, o);
    acc.y += __shfl_xor_sync(0xffffffffu, acc.y, o);
    acc.z += __shfl_xor_sync(0xffffffffu, acc.z, o);
    acc.w += __shfl_xor_sync(0xffffffffu, acc.w, o);
  }
  const int from = (lane >> 2) & 7, part = lane & 3;
  const float a0 = __shfl_sync(0xffffffffu, acc.x, from), a1 = __shfl_sync(0xffffffffu, acc.y, from);
  const float a2 = __shfl_sync(0xffffffffu, acc.z, from), a3 = __shfl_sync(0xffffffffu, acc.w, from);
  return part == 0 ? a0 : part == 1 ? a1 : part == 2 ? a2 : a3;
}

__device__ CTXFN void job_mix(Ctx& c, int q) {
  const Params& p = c.p;
  c.release();
  const int tok = q * GW + c.warp;
  if (tok < p.m) mix_finish(c.lane, tok, mix_reduce(p.s.mixp, tok, c.lane), p.w.hc_scale, p.w.hc_base, p.ffn_pre, p.ffn_post, p.ffn_comb);
  if (p.p1_resid != nullptr) c.signal(C_MX);  // XF_P1A reads the coefficients
}

// ------------------------------------------------------------------------------ jobs: routed experts
__device__ __forceinline__ Span entry_from(Ctx& c, const int* routes, int n) {
  c.sync();
  if (c.tid < n) c.sm.routes[c.tid] = __ldcg(routes + c.tid);
  return span_of(0, n);
}

__device__ __forceinline__ Span entry(Ctx& c, const Job& job) { return entry_from(c, c.p.s.table->routes + job.f, job.n); }

// R2 stage 0 carries the job's raw scale block (128 rows x 18 ue8m0, 2-byte aligned rows);
// re-lay it as [128][WS_LD] so every stage reads 4 aligned bytes per row.
__device__ __forceinline__ void r2_scale_table(Ctx& c, const uint8_t* stage) {
  const unsigned short* src = reinterpret_cast<const unsigned short*>(stage + R2_TAIL_BYTES + c.tid * S2_ROW);
  unsigned* dst = reinterpret_cast<unsigned*>(ws_of(c.sm) + c.tid * WS_LD);
#pragma unroll
  for (int w = 0; w < 4; ++w) dst[w] = unsigned(src[2 * w]) | (unsigned(src[2 * w + 1]) << 16);
  dst[4] = src[8];
}

// FP4 x f16 GEMM for one routed entry (<= 32 routes). Each stage: 128 weight rows; warp row tile
// rt = 32 rows = two 16-row m-tiles. R1 (w13): 4 stages x 4 K blocks. R2 (w2): stage 0 = K
// blocks 16-17 (+ scales), stages 1-4 = K blocks 4(s-1)..4(s-1)+3.
template <bool R2, class Store>
__device__ __forceinline__ void routed(Ctx& c, const Span& en, int ks, Store store) {
  const int rs = GW / en.ts, rt0 = c.warp / en.ts, tt = c.warp % en.ts;
  const bool live = tt < en.tiles;
  const int g = c.lane >> 2;
  constexpr int NSTAGES = R2 ? INTER_STAGES : KS_R1 / 4;
  float acc[GW][8];
#pragma unroll
  for (int r = 0; r < GW; ++r)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[r][e] = 0.0f;
#pragma unroll 1
  for (int s = 0; s < NSTAGES; ++s) {
    if (s) c.next();
    const uint8_t* stage = c.stage();
    const bool tail = R2 && s == 0;
    if (tail) {
      r2_scale_table(c, stage);
      c.sync();
    }
    if (!live) continue;
    const int kb = R2 ? (s == 0 ? 16 : (s - 1) * 4) : s * 4;
    const int nq = tail ? 2 : 4;
    uint4 b[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) b[q] = q < nq ? ld_b(c, ks, tt, kb + q) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int r = 0; r < GW; ++r) {
      if (r >= en.ts) break;
      const int rt = rt0 + r * rs;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int r0 = rt * 32 + mt * 16;
        unsigned sg, sg8;
        if constexpr (R2) {
          const int off = s == 0 ? 16 : (s - 1) * 4;
          sg = *reinterpret_cast<const unsigned*>(ws_of(c.sm) + (r0 + g) * WS_LD + off);
          sg8 = *reinterpret_cast<const unsigned*>(ws_of(c.sm) + (r0 + g + 8) * WS_LD + off);
        } else {
          sg = *reinterpret_cast<const unsigned*>(stage + SCALE_OFF + (r0 + g) * 4);
          sg8 = *reinterpret_cast<const unsigned*>(stage + SCALE_OFF + (r0 + g + 8) * 4);
        }
        float* a = acc[r] + mt * 4;
#pragma unroll
        for (int qq = 0; qq < 4; qq += 2) {
          if (qq >= nq) break;
          const uint4 w = tail ? fp4_words32(stage, r0, c.lane) : fp4_words64(stage, r0, qq, c.lane);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int q = qq + h;
            float d[4];
            mma32(d, h ? fp4_e4m3(w.z, w.w) : fp4_e4m3(w.x, w.y), b[q]);
            const float f0 = ue8m0_byte(sg, q), f1 = ue8m0_byte(sg8, q);
            a[0] = fmaf(d[0], f0, a[0]);
            a[1] = fmaf(d[1], f0, a[1]);
            a[2] = fmaf(d[2], f1, a[2]);
            a[3] = fmaf(d[3], f1, a[3]);
          }
        }
      }
    }
  }
  c.release();
  if (!live) return;
#pragma unroll
  for (int r = 0; r < GW; ++r) {
    if (r >= en.ts) break;
    const int rt = rt0 + r * rs;
    // The e4m3 expansion of e2m1 is value * 2^-6 (sm90.cuh): undo it once here.
    float4 lo = transpose4(acc[r], c.lane), hi = transpose4(acc[r] + 4, c.lane);
    lo = make_float4(lo.x * 64.0f, lo.y * 64.0f, lo.z * 64.0f, lo.w * 64.0f);
    hi = make_float4(hi.x * 64.0f, hi.y * 64.0f, hi.z * 64.0f, hi.w * 64.0f);
    const int i = quad_col(tt, c.lane), row = rt * 32 + quad_row(c.lane);
    if (i < en.count) {
      store(i, row, lo);
      store(i, row + 16, hi);
    }
  }
}

// FP4 x e4m3 RS-wgmma GEMM for one routed entry of <= 8 NT routes (ROUTED_MMA=1). Same stage
// layout and job/stage protocol as routed(): R1 = 4 stages of 128 rows x 4 k32; R2 = stage 0 (K
// blocks 16-17 + the job's scale block), stages 1-4 = K blocks 4(s-1)... (RoutedPlan units: stage
// x 64-row m-tile (rows 64 mt + 16 warp ..+15 of this warp) x 4 or 2 k32 blocks). Per k32 and
// warp (16 rows x 32 k = 256 B of FP4): 0.5 ldmatrix.x4, 2 x 7 integer ops (fp4_frag), 1 wgmma,
// 4NT FFMA, fence/commit/wait per unit; scales: one ue8m0 word -> 4 floats per lane per stage
// into a per-warp table (wf_table, in wf), 2 table loads per unit; ~26 SASS per 256 B at NT=1
// (HMMA routed(): ~44). `post` = 64 (bit trick) x activation scale. store(column i of the B tile,
// row, float4 of rows row..row+3) for i < en.count; bbase = B tile smem address (default bt).
__device__ __forceinline__ float* wf_table(const Ctx& c, int parity) { return c.sm.wf + (parity * GW + c.warp) * 128; }

template <bool R2, int NT, class Store>
__device__ __forceinline__ void routed_wg(Ctx& c, const Span& en, float post, Store store, uint32_t bbase = 0) {
  using Plan = RoutedPlan<R2, NT>;
  constexpr int NKU = Plan::NKU;
  const int g = c.lane >> 2;
  const uint64_t desc = desc_b(bbase ? bbase : bt_smem_addr(), NT);
  float tot[2][4 * NT], acc[2][NKU][4 * NT];
  unsigned A[2][NKU][4];
#pragma unroll
  for (int b = 0; b < 2; ++b) {
#pragma unroll
    for (int e = 0; e < 4 * NT; ++e) tot[b][e] = 0.0f;
#pragma unroll
    for (int q = 0; q < NKU; ++q)
#pragma unroll
      for (int e = 0; e < 4 * NT; ++e) acc[b][q][e] = 0.0f;
  }
  const uint8_t* stage = c.stage();
  auto load = [&](auto uc) {
    constexpr int u = decltype(uc)::value, B = u & 1, s = Plan::stage(u);
    if constexpr (Plan::first(u)) {
      if constexpr (u > 0) c.acquire();
      stage = c.stage();
      if constexpr (R2 && s == 0) {
        r2_scale_table(c, stage);
        c.sync();
      }
      // Per-warp float table (double-buffered by stage parity): lane l converts the ue8m0 word of
      // row ((l & 16) << 2) + 16 warp + (l & 15) = (m-tile l >> 4, row l & 15 of this warp).
      const int row = ((c.lane & 16) << 2) + 16 * c.warp + (c.lane & 15);
      unsigned w;
      if constexpr (R2) w = *reinterpret_cast<const unsigned*>(ws_of(c.sm) + row * WS_LD + (s == 0 ? 16 : (s - 1) * 4));
      else w = *reinterpret_cast<const unsigned*>(stage + SCALE_OFF + row * 4);
      __syncwarp();  // stage s-2's reads of this buffer are done (cross-lane WAR)
      reinterpret_cast<float4*>(wf_table(c, s & 1))[c.lane] = make_float4(ue8m0_byte(w, 0), ue8m0_byte(w, 1), ue8m0_byte(w, 2), ue8m0_byte(w, 3));
      __syncwarp();
    }
    const int r0 = Plan::mt(u) * 64 + 16 * c.warp;
    if constexpr (Plan::nk(u) == 2 && R2 && s == 0) {
      const uint4 w = fp4_words32(stage, r0, c.lane);
      fp4_frag(w.x, w.y, A[B][0]);
      fp4_frag(w.z, w.w, A[B][1]);
    } else {
#pragma unroll
      for (int h = 0; h < Plan::nk(u); h += 2) {
        const uint4 w = fp4_words64(stage, r0, Plan::q0(u) + h, c.lane);
        fp4_frag(w.x, w.y, A[B][h]);
        fp4_frag(w.z, w.w, A[B][h + 1]);
      }
    }
    if constexpr (Plan::last(u)) c.release();
  };
  auto issue = [&](auto uc) {
    constexpr int u = decltype(uc)::value;
    wg_group<NT, Plan::nk(u), NKU>(A[u & 1], acc[u & 1], desc, Plan::kb(u));
  };
  auto retire = [&](auto uc) {
    constexpr int u = decltype(uc)::value, B = u & 1, NK = Plan::nk(u), mt = Plan::mt(u), q0 = Plan::q0(u);
    wg_retired<NT, NK, NKU>(A[B], acc[B]);
    const float* tab = wf_table(c, Plan::stage(u) & 1);
    float f0[NK], f1[NK];
    if constexpr (NK == 4) {
      const float4 a = reinterpret_cast<const float4*>(tab)[mt * 16 + g], b = reinterpret_cast<const float4*>(tab)[mt * 16 + g + 8];
      f0[0] = a.x, f0[1] = a.y, f0[2] = a.z, f0[3] = a.w;
      f1[0] = b.x, f1[1] = b.y, f1[2] = b.z, f1[3] = b.w;
    } else {
      const float2 a = *reinterpret_cast<const float2*>(tab + (mt * 16 + g) * 4 + q0), b = *reinterpret_cast<const float2*>(tab + (mt * 16 + g + 8) * 4 + q0);
      f0[0] = a.x, f0[1] = a.y;
      f1[0] = b.x, f1[1] = b.y;
    }
#pragma unroll
    for (int q = 0; q < NK; ++q)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        tot[mt][4 * j + 0] = fmaf(acc[B][q][4 * j + 0], f0[q], tot[mt][4 * j + 0]);
        tot[mt][4 * j + 1] = fmaf(acc[B][q][4 * j + 1], f0[q], tot[mt][4 * j + 1]);
        tot[mt][4 * j + 2] = fmaf(acc[B][q][4 * j + 2], f1[q], tot[mt][4 * j + 2]);
        tot[mt][4 * j + 3] = fmaf(acc[B][q][4 * j + 3], f1[q], tot[mt][4 * j + 3]);
      }
  };
  wg_pipe<Plan>(load, issue, retire);
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      float4 v = transpose4(tot[mt] + 4 * j, c.lane);
      v = make_float4(v.x * post, v.y * post, v.z * post, v.w * post);
      const int i = quad_col(j, c.lane), row = mt * 64 + 16 * c.warp + quad_row(c.lane);
      if (i < en.count) store(i, row, v);
    }
}

// R1 (w13) GEMM of one routed job: stage the entry's x rows and run the HMMA or wgmma loop into
// hacc. Shared by the megakernel (job_r1) and the routed micro-benchmark.
template <bool WG>
__device__ __forceinline__ void r1_gemm(Ctx& c, const Job& job, const Span& en) {
  const Params& p = c.p;
  const int bb = job.b, kp = job.c;
  auto store = [&](int i, int r, float4 v) { red4(p.s.hacc + long(c.sm.routes[i]) * 2 * INTER + (r < 64 ? 0 : INTER - 64) + bb * 64 + r, v); };
  if constexpr (WG) {
    if (en.tiles <= WG_NT) {
      fill_wg<false>(
          c, en.count, KS_R1, en.tiles, [&](int i) { return p.s.xq + long(c.sm.routes[i] / TOPK) * HID + kp * KS_R1 * 32; }, [](int, int) { return 0.0f; });
      c.tmark(TR_X0);
      if (en.tiles == 1) routed_wg<false, 1>(c, en, 64.0f * p.w.xs, store);
      else routed_wg<false, 2>(c, en, 64.0f * p.w.xs, store);
      return;
    }
  }
  fill_q<false>(
      c, en.count, KS_R1, [&](int i) { return reinterpret_cast<const uint8_t*>(p.s.xh + long(c.sm.routes[i] / TOPK) * HID + kp * KS_R1 * 32); },
      [](int) -> const uint8_t* { return nullptr; });
  c.tmark(TR_X0);
  routed<false>(c, en, KS_R1, store);
}

// R2 (w2) GEMM of one routed job into yracc (weighted by the route's top-k weight).
template <bool WG>
__device__ __forceinline__ void r2_gemm(Ctx& c, const Job& job, const Span& en) {
  const Params& p = c.p;
  const int rb = job.b;
  auto store = [&](int i, int r, float4 v) {
    const int route = c.sm.routes[i];
    const float w = __ldcg(p.s.topk_w + route);
    red4(p.s.yracc + long(route / TOPK) * HID + rb * 128 + r, make_float4(w * v.x, w * v.y, w * v.z, w * v.w));
  };
  if constexpr (WG) {
    if (en.tiles <= WG_NT) {
      fill_wg<false>(c, en.count, KS_INTER, en.tiles, [&](int i) { return p.s.hq + long(c.sm.routes[i]) * INTER; }, [](int, int) { return 0.0f; });
      c.tmark(TR_X0);
      if (en.tiles == 1) routed_wg<true, 1>(c, en, 64.0f / H_INV, store);
      else routed_wg<true, 2>(c, en, 64.0f / H_INV, store);
      return;
    }
  }
  fill_q<false>(
      c, en.count, KS_INTER, [&](int i) { return reinterpret_cast<const uint8_t*>(p.s.hh + long(c.sm.routes[i]) * INTER); },
      [](int) -> const uint8_t* { return nullptr; });
  c.tmark(TR_X0);
  routed<true>(c, en, KS_INTER, store);
}

// Routed SiluAndMul on bf16-rounded gate/up (stock Humming W4A16 writes bf16 GEMM outputs).
__device__ CTXFN void finish_r1(Ctx& c, int ei, int bb, const Span& en) {
  const Params& p = c.p;
  for (int item = c.warp; item < en.count * 2; item += GW) {
    const int route = c.sm.routes[item / 2], col = bb * 64 + (item & 1) * 32 + c.lane;
    float* h = p.s.hacc + long(route) * 2 * INTER + col;
    const float g = __ldcg(h), u = __ldcg(h + INTER);
    __stcg(h, 0.0f);
    __stcg(h + INTER, 0.0f);
    const float a = bf(swiglu(bf(g), bf(u)));
    p.s.hh[long(route) * INTER + (col & ~7) + p8_pos(col & 7)] = __float2half_rn(a);
    if constexpr (ROUTED_WG) p.s.hq[long(route) * INTER + wg_pos(col)] = e4m3_byte(a * H_INV);
  }
  c.signal(C_R1D + ei);
}

__device__ WG_INLINE void job_r1(Ctx& c, const Job& job) {
  const int ei = job.a, bb = job.b;
  const Span en = entry(c, job);
  r1_gemm<ROUTED_WG>(c, job, en);
  c.tmark(TR_X1);
  if (c.arrive(C_R1A + ei * R1_BLOCKS + bb, R1_PARTS)) c.tmark(TR_X2), finish_r1(c, ei, bb, en);
}

__device__ WG_INLINE void job_r2(Ctx& c, const Job& job) {
  const Span en = entry(c, job);
  r2_gemm<ROUTED_WG>(c, job, en);
  c.tmark(TR_X1);
  c.signal(C_SF + job.b);
}

// ------------------------------------------------------------------------------ jobs: coarse routed (m <= COARSE_M)
// Resident x tile: the normalized FFN input of all m tokens (e4m3 x / xs in wg order, written by the
// gate jobs) in the wgmma B layout (16-byte chunk cb = 2 kb + ch, token n at cb * 128 + 16 n), staged
// once per CTA into bt (tokens >= m zero). Routing (C_ROUTE) follows every gate job, and the producer
// acquires C_ROUTE before it decodes any routed job, so xq is complete here.
__device__ CTXFN void load_xres(Ctx& c) {
  if (c.sm.xres) return;
  const Params& p = c.p;
  c.sync();  // earlier generic accesses of bt are done
  if (c.tid == 0) {  // one 40 KB bulk copy of the image the gate jobs wrote (token columns >= m: stale, finite, unused)
    fence_proxy_async();
    mbar_expect_tx(&c.sm.xbar, XT_BYTES);
    bulk_g2s(c.sm.bt, p.s.xqi, XT_BYTES, &c.sm.xbar, evict_first_policy());
  }
  mbar_wait(&c.sm.xbar, 0);  // once per launch
  c.sync();
  if (c.tid == 0) c.sm.xres = 1;  // read again only in a later job (after further barriers)
}

// Coarse R1 stage tile: 64 rows (0-31 gate rows 32 b.., 32-63 up rows INTER + 32 b..) x 128 B, the
// 16-byte chunks XOR-swizzled by row & 7 (conflict-free ldmatrix), 64 x 8 B scales at SCALE_OFF.
__device__ __forceinline__ int fp4_off128(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }

// One pair of coarse R1 stages (16 k32 blocks, B blocks kb..kb+15 of the resident x tile) for this
// warp: rows 16 warp .. +15 of the 64-row tile. Pipeline of 4 units of 4 k32 (two commit groups in
// flight, drained at the end of the pair); each stage is released after its last ldmatrix.
__device__ __forceinline__ void r1c_pair(Ctx& c, const uint8_t* s0, const uint8_t* s1, const uint32_t (&uoff)[4], uint64_t xdesc, int kb, float (&tot)[4]) {
  const int g = c.lane >> 2;
  float* tab = c.sm.wf + c.warp * 16 * TAB_LD;
  {
    const int rl = c.lane >> 1, half = c.lane & 1;
    const uint2 v = R1C_TMA ? *reinterpret_cast<const uint2*>(s0 + SCALE_OFF + (16 * c.warp + rl) * 16 + half * 8)
                            : *reinterpret_cast<const uint2*>((half ? s1 : s0) + SCALE_OFF + (16 * c.warp + rl) * 8);
    const float4 a = make_float4(ue8m0_byte(v.x, 0), ue8m0_byte(v.x, 1), ue8m0_byte(v.x, 2), ue8m0_byte(v.x, 3));
    const float4 b = make_float4(ue8m0_byte(v.y, 0), ue8m0_byte(v.y, 1), ue8m0_byte(v.y, 2), ue8m0_byte(v.y, 3));
    __syncwarp();  // the previous pair's table reads are done
    float4* dst = reinterpret_cast<float4*>(tab + rl * TAB_LD + half * 8);
    dst[0] = a;
    dst[1] = b;
    __syncwarp();
  }
  const uint32_t a0 = smem_u32(s0), a1 = smem_u32(s1);
  unsigned A[2][4][4];
  float acc[2][4][4];
#pragma unroll
  for (int b = 0; b < 2; ++b)
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[b][q][e] = 0.0f;
  auto load = [&](int u, int b) {
#pragma unroll
    for (int h = 0; h < 4; h += 2) {
      const uint4 w = ldsm4_u32((u < 2 ? a0 : a1) + uoff[2 * (u & 1) + h / 2]);
      fp4_frag(w.x, w.y, A[b][h]);
      fp4_frag(w.z, w.w, A[b][h + 1]);
    }
  };
  auto retire = [&](int u, int b) {
    wg_retired<1, 4, 4>(A[b], acc[b]);
    const float4 f0 = *reinterpret_cast<const float4*>(tab + g * TAB_LD + 4 * u);
    const float4 f1 = *reinterpret_cast<const float4*>(tab + (g + 8) * TAB_LD + 4 * u);
    const float sg[4] = {f0.x, f0.y, f0.z, f0.w}, su[4] = {f1.x, f1.y, f1.z, f1.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      tot[0] = fmaf(acc[b][q][0], sg[q], tot[0]);
      tot[1] = fmaf(acc[b][q][1], sg[q], tot[1]);
      tot[2] = fmaf(acc[b][q][2], su[q], tot[2]);
      tot[3] = fmaf(acc[b][q][3], su[q], tot[3]);
    }
  };
  load(0, 0);
  wg_group<1, 4, 4>(A[0], acc[0], xdesc, kb + 0);
  load(1, 1);
  c.release();  // s0
  wg_group<1, 4, 4>(A[1], acc[1], xdesc, kb + 4);
  wg_wait<1>();
  retire(0, 0);
  load(2, 0);
  wg_group<1, 4, 4>(A[0], acc[0], xdesc, kb + 8);
  wg_wait<1>();
  retire(1, 1);
  load(3, 1);
  c.release();  // s1
  wg_group<1, 4, 4>(A[1], acc[1], xdesc, kb + 12);
  wg_wait<1>();
  retire(2, 0);
  wg_wait<0>();
  retire(3, 1);
}

// h of 32 columns (k32 block b of the expert's h) x 8 tokens -> the slot's B-tile image (e4m3 h * H_INV).
__device__ __forceinline__ void store_hqi(const Params& p, int slot, int b, int col, int tok, float h) {
  const int pos = wg_pos(col);
  p.s.hqi[long(slot) * HQI + b * 256 + (pos >> 4) * 128 + tok * 16 + (pos & 15)] = e4m3_byte(h * H_INV);
}

// Second K part of a coarse R1 row block (R1_KP = 2): SwiGLU of the summed partials.
__device__ CTXFN void finish_r1c(Ctx& c, int slot, int b) {
  const Params& p = c.p;
  for (int i = c.tid; i < 32 * TILE; i += GT) {
    const int tok = i >> 5, col = i & 31;
    float* h = p.s.hcacc + (long(slot) * TILE + tok) * 2 * INTER + 32 * b + col;
    const float gt = __ldcg(h), up = __ldcg(h + INTER);
    __stcg(h, 0.0f);
    __stcg(h + INTER, 0.0f);
    store_hqi(p, slot, b, col, tok, bf(swiglu(bf(gt), bf(up))));
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");  // hqi is read by R2C bulk copies
  c.signal(C_R1D + slot);
}

// Unrolled coarse R1 (R1C_NB >= 2): the job's NS stages as 2 NS units of 4 k32 in one straight-line
// pipeline with NB commit groups in flight (ptxas keeps wgmma asynchronous only without loop back-
// edges); per-stage scale tables double-buffered by stage parity; each stage is released after its
// last ldmatrix. Stage s holds k32 blocks kb0 + 8 s .. +7.
template <int NS, int NB>
__device__ __forceinline__ void r1c_unrolled(Ctx& c, const uint32_t (&uoff)[4], uint64_t xdesc, int kb0, float (&tot)[4]) {
  constexpr int U = 2 * NS;
  const int g = c.lane >> 2;
  float* tab = c.sm.wf + c.warp * 2 * 16 * TAB_LD1;
  unsigned A[NB][4][4];
  float acc[NB][4][4];
#pragma unroll
  for (int b = 0; b < NB; ++b)
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[b][q][e] = 0.0f;
  uint32_t abase = smem_u32(c.stage());
  auto load = [&](auto uc) {
    constexpr int u = decltype(uc)::value, st = u / 2, q = u % 2, B = u % NB;
    if constexpr (q == 0) {
      if constexpr (st > 0) c.acquire();
      const uint8_t* stage = c.stage();
      abase = smem_u32(stage);
      const int rl = c.lane >> 1, half = c.lane & 1;
      const unsigned w = R1C_TMA ? *reinterpret_cast<const unsigned*>(stage + SCALE_OFF + (16 * c.warp + rl) * 16 + (st & 1) * 8 + half * 4)
                                 : *reinterpret_cast<const unsigned*>(stage + SCALE_OFF + (16 * c.warp + rl) * 8 + half * 4);
      __syncwarp();  // stage st-2's table reads (same parity) are done
      reinterpret_cast<float4*>(tab + (st & 1) * 16 * TAB_LD1 + rl * TAB_LD1 + half * 4)[0] =
          make_float4(ue8m0_byte(w, 0), ue8m0_byte(w, 1), ue8m0_byte(w, 2), ue8m0_byte(w, 3));
      __syncwarp();
    }
#pragma unroll
    for (int h = 0; h < 4; h += 2) {
      const uint4 w = ldsm4_u32(abase + uoff[2 * q + h / 2]);
      fp4_frag(w.x, w.y, A[B][h]);
      fp4_frag(w.z, w.w, A[B][h + 1]);
    }
    if constexpr (q == 1) c.release();
  };
  auto issue = [&](auto uc) {
    constexpr int u = decltype(uc)::value;
    wg_group<1, 4, 4>(A[u % NB], acc[u % NB], xdesc, kb0 + 4 * u);
  };
  auto retire = [&](auto uc) {
    constexpr int u = decltype(uc)::value, st = u / 2, q = u % 2, B = u % NB;
    wg_retired<1, 4, 4>(A[B], acc[B]);
    const float* tb = tab + (st & 1) * 16 * TAB_LD1;
    const float4 f0 = *reinterpret_cast<const float4*>(tb + g * TAB_LD1 + 4 * q);
    const float4 f1 = *reinterpret_cast<const float4*>(tb + (g + 8) * TAB_LD1 + 4 * q);
    const float sg[4] = {f0.x, f0.y, f0.z, f0.w}, su[4] = {f1.x, f1.y, f1.z, f1.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      tot[0] = fmaf(acc[B][k][0], sg[k], tot[0]);
      tot[1] = fmaf(acc[B][k][1], sg[k], tot[1]);
      tot[2] = fmaf(acc[B][k][2], su[k], tot[2]);
      tot[3] = fmaf(acc[B][k][3], su[k], tot[3]);
    }
  };
  [&]<int... I>(std::integer_sequence<int, I...>) {
    (
        [&] {
          load(IC<I>{});
          issue(IC<I>{});
          if constexpr (I >= NB - 1) {
            wg_wait<NB - 1>();
            retire(IC<I - (NB - 1)>{});
          }
        }(),
        ...);
  }(std::make_integer_sequence<int, U>{});
  if constexpr (NB >= 3) {
    wg_wait<1>();
    retire(IC<U - 2>{});
  }
  wg_wait<0>();
  retire(IC<U - 1>{});
}

// Coarse R1 job: expert slot a, row block b (32 gate + 32 up rows), K part c of L.kp (job.stages
// stages of 128 B per row). Warps 0/1 hold gate rows 16 w + g (+8), warps 2/3 the matching up rows.
__device__ WG_INLINE void job_r1c(Ctx& c, const Job& job) {
  const Params& p = c.p;
  load_xres(c);
  c.tmark(TR_X0);
  const int slot = job.a, b = job.b;
  const int j = c.lane >> 3, r = 16 * c.warp + 8 * (j & 1) + (c.lane & 7);
  uint32_t uoff[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) uoff[i] = r * 128 + (((2 * i + (j >> 1)) ^ (r & 7)) << 4);
  const uint64_t xdesc = desc_b(bt_smem_addr(), 1);
  float tot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int kb0 = job.c * job.stages * 8;
  if constexpr (R1C_NB >= 2) {
    if (job.stages == R1C_STAGES) r1c_unrolled<R1C_STAGES, R1C_NB>(c, uoff, xdesc, kb0, tot);
    else r1c_unrolled<R1C_STAGES / 2, R1C_NB>(c, uoff, xdesc, kb0, tot);
  } else {
#pragma unroll 1
    for (int pr = 0; pr < job.stages / 2; ++pr) {
      if (pr) c.acquire();
      const uint8_t* s0 = c.stage();
      const uint8_t* s1 = c.held(1);
      r1c_pair(c, s0, s1, uoff, xdesc, kb0 + 16 * pr, tot);
    }
  }
  c.tmark(TR_X1);
  const float post = 64.0f * p.w.xs;
#pragma unroll
  for (int e = 0; e < 4; ++e) tot[e] *= post;
  if (job.stages != R1C_STAGES) {  // K part: accumulate the partial, the second arrival finishes
    const float4 v = transpose4(tot, c.lane);
    const int tok = quad_col(0, c.lane), row = (c.warp < 2 ? 32 * b + 16 * c.warp : INTER + 32 * b + 16 * (c.warp - 2)) + quad_row(c.lane);
    red4(p.s.hcacc + (long(slot) * TILE + tok) * 2 * INTER + row, v);
    if (c.arrive(C_R1A + slot * R1C_BLOCKS + b, R1C_STAGES / job.stages)) c.tmark(TR_X2), finish_r1c(c, slot, b);
    return;
  }
  float4* xch = reinterpret_cast<float4*>(c.sm.bs);
  if (c.warp >= 2) xch[(c.warp - 2) * 32 + c.lane] = make_float4(tot[0], tot[1], tot[2], tot[3]);
  c.sync();
  if (c.warp < 2) {
    const float4 u = xch[c.warp * 32 + c.lane];
    const float up[4] = {u.x, u.y, u.z, u.w};
    const int g = c.lane >> 2, t = c.lane & 3;
#pragma unroll
    for (int e = 0; e < 4; ++e) store_hqi(p, slot, b, 16 * c.warp + g + (e & 2 ? 8 : 0), 2 * t + (e & 1), bf(swiglu(bf(tot[e]), bf(up[e]))));
    asm volatile("fence.proxy.async.global;" ::: "memory");  // hqi is read by R2C bulk copies
  }
  c.signal(C_R1D + slot);
}

// Coarse R2 job: rows 128 rb .. +127 of expert slot a's w2 (K = 576) against the slot's h image (all
// 8 token columns), scaled per token by its routing weight (0 = not routed: no store), red.add into
// yracc (chunk rb).
__device__ WG_INLINE void job_r2c(Ctx& c, const Job& job) {
  const Params& p = c.p;
  const int slot = job.a, rb = job.b;
  const unsigned par = unsigned(c.sm.nh) & 1u;
  if (c.tid == 0) {  // one bulk copy of the slot's h image (bt2's previous readers finished with their job)
    fence_proxy_async();
    mbar_expect_tx(&c.sm.hbar, HQI);
    bulk_g2s(c.sm.bt2, p.s.hqi + long(slot) * HQI, HQI, &c.sm.hbar, evict_first_policy());
  }
  if (c.tid < TILE) c.sm.rs[c.tid] = __ldcg(p.s.wtok + slot * TILE + c.tid);  // overlaps the h copy
  mbar_wait(&c.sm.hbar, par);
  c.sync();
  if (c.tid == 0) ++c.sm.nh;
  c.tmark(TR_X0);
  auto store = [&](int i, int r, float4 v) {
    const float w = c.sm.rs[i];
    if (w != 0.0f) red4(p.s.yracc + long(i) * HID + rb * 128 + r, make_float4(w * v.x, w * v.y, w * v.z, w * v.w));
  };
  routed_wg<true, 1>(c, span_of(0, TILE), 64.0f / H_INV, store, bt2_smem_addr());
  c.tmark(TR_X1);
  c.signal(C_SF + rb);
}

// ------------------------------------------------------------------------------ jobs: lossless coarse routed (LC, m <= COARSE_M)
// FP4 word of one weight row (8 e2m1 nibbles, low nibble = even k: k = 8t..8t+7 of a k32 block) -> the four
// f16x2 A registers of that k32 block in P8 pairs (k0,k2) (k4,k6) (k1,k3) (k5,k7) (the B order of the f16
// activation images). Exact: the f16 bits (s << 15) | (c << 9) are value(s, c) * 2^-14 for all 16 e2m1 codes
// (code 1 = 0.5 -> the f16 subnormal 2^-15). The even / odd nibbles are first gathered into bytes [s 000 c 0]
// (the f16 high bytes), then one PRMT per register places two of them over zero low bytes.
__device__ __forceinline__ uint4 fp4_f16_p8(unsigned w) {
  // The shifts are multiplies (IMAD / IMAD.HI on the FMA pipe), leaving 4 LOP3 + 4 PRMT on the ALU pipe.
  const unsigned ve = ((w * 16u) & 0x80808080u) | ((w * 2u) & 0x0E0E0E0Eu);           // k0 k2 k4 k6
  const unsigned vo = (w & 0x80808080u) | (__umulhi(w, 0x20000000u) & 0x0E0E0E0Eu);  // k1 k3 k5 k7 (w >> 3)
  return make_uint4(prmt_b32(ve, 0u, 0x1404u), prmt_b32(ve, 0u, 0x3424u), prmt_b32(vo, 0u, 0x1404u), prmt_b32(vo, 0u, 0x3424u));
}

// One k32 block of a 16x8 tile into a fresh accumulator: FP4 words of rows g / g+8 (k = 8t..8t+7) x the
// token-g P8 f16 B fragment (b.x / b.y = k (0,2) / (4,6), b.z / b.w = k (1,3) / (5,7)). Result * 2^-14.
__device__ __forceinline__ void mma32_fp4(float (&d)[4], unsigned wg, unsigned wg8, uint4 b, float d0 = 0.0f) {
  const uint4 a = ABL & 2 ? make_uint4(wg, wg, wg, wg) : fp4_f16_p8(wg), c = ABL & 2 ? make_uint4(wg8, wg8, wg8, wg8) : fp4_f16_p8(wg8);
  if constexpr ((ABL & 8) != 0) {  // ablation: no HMMA (conversions folded so they stay live)
    d[0] = __uint_as_float(a.x ^ c.y), d[1] = __uint_as_float(a.y ^ c.x), d[2] = __uint_as_float(a.z ^ c.w), d[3] = __uint_as_float(a.w ^ c.z ^ b.x);
    return;
  }
  d[0] = d[1] = d[2] = d[3] = d0;
  hmma16(d, a.x, c.x, a.y, c.y, b.x, b.y);
  hmma16(d, a.z, c.z, a.w, c.w, b.z, b.w);
}

// Per-warp float scale table [32 rows][4 k32] of the current stage (double-buffered by stage parity): lane l
// converts the ue8m0 word (4 consecutive k32 scales) of row 32 warp + l of the stage tile.
__device__ __forceinline__ float4* lc_table(const Ctx& c, int parity) { return reinterpret_cast<float4*>(c.sm.wf) + (parity * GW + c.warp) * 32; }
__device__ __forceinline__ void lc_put(const Ctx& c, int parity, unsigned w) {
  if constexpr ((ABL & 4) != 0) return;  // ablation: no scale tables
  float4* tab = lc_table(c, parity);
  // ue8m0 byte q -> float 2^(byte - 127): (byte << 23); left shifts as multiplies (FMA pipe)
  const float f0 = __uint_as_float((w * (1u << 23)) & 0x7f800000u), f1 = __uint_as_float((w * (1u << 15)) & 0x7f800000u);
  const float f2 = __uint_as_float((w * (1u << 7)) & 0x7f800000u), f3 = __uint_as_float(__umulhi(w, 0x80000000u) & 0x7f800000u);
  __syncwarp();  // stage s-2's reads of this buffer are done (cross-lane WAR)
  tab[c.lane] = make_float4(f0, f1, f2, f3);
  __syncwarp();
}

// One stage of an LC job for this warp: rows 32 warp + 16 mt + (g, g+8) (mt = 0, 1) x NQ k32 blocks. A: the
// stage tile (NQ = 4: 128 rows x 64 B, fp4_off64; NQ = 2: the R2 K tail, 128 x 32 B, fp4_off32); B: f16
// [k32][8 tokens][64 B] at bsrc; scales: the warp's float table. lc_load moves everything into registers (the
// ring stage can be released right after), lc_math runs the MMAs: per k32 and m-tile 2 x 11 integer ops
// (fp4_f16_p8), 2 HMMA, 4 FFMA with the exact ue8m0 scale.
struct LcRegs {
  uint4 bq[4], wv[2][2];
  float4 sg[2], s8[2];
  const float4* tab;  // PS = false: the scales are read from this table at math time
  unsigned sw[2][2];  // SCL_DIRECT: ue8m0 words (4 k32 blocks) of rows g / g + 8 of m-tile 0 / 1
  float dz;           // REL_DEP: 0.0f that depends on the stage release (initial accumulator of the stage's MMAs)
};

// PS: also load this thread's scales (else lc_math reads them from the table; R1F_PIPE=2 keeps fewer registers live)
template <int NQ, bool PS = true>
__device__ __forceinline__ void lc_load(const Ctx& c, const uint8_t* st, const uint8_t* bsrc, const float4* tab, LcRegs& r) {
  const int g = c.lane >> 2, t = c.lane & 3;
#pragma unroll
  for (int q = 0; q < NQ; ++q) r.bq[q] = *reinterpret_cast<const uint4*>(bsrc + (q * TILE + g) * 64 + t * 16);
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int h = 0; h < NQ / 2; ++h)
      r.wv[mt][h] = NQ == 4 ? fp4_words64(st, 32 * c.warp + 16 * mt, 2 * h, c.lane) : fp4_words32(st, 32 * c.warp + 16 * mt, c.lane);
  r.tab = tab;
  if constexpr (SCL_DIRECT) {
    // r.sw filled by the caller
  } else if constexpr ((ABL & 4) != 0) {  // ablation: constant scales
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) r.sg[mt] = r.s8[mt] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  } else if constexpr (PS) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) r.sg[mt] = tab[16 * mt + g], r.s8[mt] = tab[16 * mt + g + 8];
  }
}

template <int NQ, bool PS = true>
__device__ __forceinline__ void lc_math(LcRegs& r, float (&tot)[2][4], int lane) {
  if constexpr ((ABL & 1) != 0) return;  // ablation: data path only
  if constexpr (!PS && (ABL & 4) == 0 && !SCL_DIRECT) {
    const int g = lane >> 2;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) r.sg[mt] = r.tab[16 * mt + g], r.s8[mt] = r.tab[16 * mt + g + 8];
  }
#pragma unroll
  for (int q = 0; q < NQ; ++q)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const uint4 w = r.wv[mt][q >> 1];
      float d[4];
      mma32_fp4(d, q & 1 ? w.z : w.x, q & 1 ? w.w : w.y, r.bq[q], REL_DEP ? r.dz : 0.0f);
      float f0, f1;
      if constexpr (SCL_DIRECT) {
        f0 = ue8m0_byte(r.sw[mt][0], NQ == 2 ? q : q), f1 = ue8m0_byte(r.sw[mt][1], q);
      } else {
        f0 = q == 0 ? r.sg[mt].x : q == 1 ? r.sg[mt].y : q == 2 ? r.sg[mt].z : r.sg[mt].w;
        f1 = q == 0 ? r.s8[mt].x : q == 1 ? r.s8[mt].y : q == 2 ? r.s8[mt].z : r.s8[mt].w;
      }
      tot[mt][0] = fmaf(d[0], f0, tot[mt][0]);
      tot[mt][1] = fmaf(d[1], f0, tot[mt][1]);
      tot[mt][2] = fmaf(d[2], f1, tot[mt][2]);
      tot[mt][3] = fmaf(d[3], f1, tot[mt][3]);
    }
}

// Release the stage whose registers were just loaded, keeping the arrive ahead of the math (ptxas otherwise
// sinks it below the MMAs).
__device__ __forceinline__ float lc_release(Ctx& c, float (&tot)[2][4]) {
  if constexpr (REL_DEP) return c.release_dep();
  c.release();
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) rfence(tot[mt][e]);
  return 0.0f;
}

template <int NQ>
__device__ __forceinline__ void lc_stage(Ctx& c, const uint8_t* st, const uint8_t* bsrc, const float4* tab, float (&tot)[2][4]) {
  LcRegs r;
  lc_load<NQ>(c, st, bsrc, tab, r);
  r.dz = lc_release(c, tot);
  lc_math<NQ>(r, tot, c.lane);
}

// LAZY_SIG: post the previous R2F job's chunk arrival now (its red.adds have had a stage's time to complete).
__device__ __forceinline__ void lazy_flush(Ctx& c) {
  if constexpr (LAZY_SIG) {
    if (c.lazy) {
      c.signal(C_SF + c.lazy - 1);
      c.lazy = 0;
    }
  }
}

// h (bf16-rounded SwiGLU output, exact in f16) of intermediate column col (0..575) and token tok of an
// expert slot -> the slot's f16 h tile ([k32][token][32 values in P8 order]).
__device__ __forceinline__ void store_hhi(const Params& p, int slot, int col, int tok, float h) {
  const int pos = 8 * ((col >> 3) & 3) + p8_pos(col & 7);
  reinterpret_cast<__half*>(p.s.hhi + long(slot) * HHI + ((col >> 5) * TILE + tok) * 64)[pos] = __float2half_rn(h);
}

// Last K part of an R1F row block (r1kp = 2 / 5): SwiGLU of the summed partials (64 columns x 8 tokens).
__device__ CTXFN void finish_r1f(Ctx& c, int slot, int b) {
  const Params& p = c.p;
  for (int i = c.tid; i < 64 * TILE; i += GT) {
    const int tok = i >> 6, col = 64 * b + (i & 63);
    float* h = p.s.hcacc + (long(slot) * TILE + tok) * 2 * INTER + col;
    const float gt = __ldcg(h), up = __ldcg(h + INTER);
    __stcg(h, 0.0f);
    __stcg(h + INTER, 0.0f);
    store_hhi(p, slot, col, tok, bf(swiglu(bf(gt), bf(up))));
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");  // hhi is read by R2F bulk copies
  c.signal(C_R1D + slot);
}

// R1F_LOC / R2F_LOC (LC routed stage loops as calls, see the knobs): one full stage (128 rows x 4 k32) of this warp: wait
// for slot rs (unless first), A by ldmatrix (rows 32 warp + 16 mt + ..., a0 / a1 = this lane's k32-pair offsets), B at bsrc
// (+ q * TILE * 64), scales from the warp table (parity par, filled by the caller) -> registers, slot release (the MMAs
// depend on the arrive's state token), then the MMAs. Same math / summation order as lc_load + lc_math (lossless).
__device__ __forceinline__ void loc_stage(Smem& sm, int& rs, unsigned& ph, const uint8_t* bsrc, const float4* tr, int a0, int a1, int lane,
                                          unsigned zmask, float (&tot)[2][4]) {
  const uint8_t* st = sm.ring[rs];
  uint4 wv[2][2], bq[4];
  float4 sg[2], s8[2];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) wv[mt][0] = ldsm4(st + a0 + mt * 1024), wv[mt][1] = ldsm4(st + a1 + mt * 1024);
#pragma unroll
  for (int q = 0; q < 4; ++q) bq[q] = *reinterpret_cast<const uint4*>(bsrc + q * TILE * 64);
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) sg[mt] = tr[16 * mt], s8[mt] = tr[16 * mt + 8];
  if constexpr (PXF) fence_proxy_async();
  __syncwarp();
  unsigned lo = 0;
  if (lane == 0) {
    unsigned long long tok;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(tok) : "r"(smem_u32(&sm.empty[rs])) : "memory");
    lo = unsigned(tok);
  }
  unsigned z;
  asm("and.b32 %0, %1, %2;" : "=r"(z) : "r"(lo), "r"(zmask));
  const float dz = __uint_as_float(z);
  if (++rs == STAGES) rs = 0, ph ^= 1u;
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const uint4 w = wv[mt][q >> 1];
      float d[4];
      mma32_fp4(d, q & 1 ? w.z : w.x, q & 1 ? w.w : w.y, bq[q], dz);
      const float f0 = q == 0 ? sg[mt].x : q == 1 ? sg[mt].y : q == 2 ? sg[mt].z : sg[mt].w;
      const float f1 = q == 0 ? s8[mt].x : q == 1 ? s8[mt].y : q == 2 ? s8[mt].z : s8[mt].w;
      tot[mt][0] = fmaf(d[0], f0, tot[mt][0]);
      tot[mt][1] = fmaf(d[1], f0, tot[mt][1]);
      tot[mt][2] = fmaf(d[2], f1, tot[mt][2]);
      tot[mt][3] = fmaf(d[3], f1, tot[mt][3]);
    }
}

// R1F_LOC=2: two stages per iteration. Both stages' operands are loaded, both slots released, then all 32 HMMAs are
// independent (twice the consumer warp's ILP: the routed stage is latency bound at 2 consumer warps per SM sub-partition)
// while the FFMA accumulation keeps the exact order of the one-stage loop (stage A q 0..3, then stage B): bit-identical.
struct LocOps {
  uint4 wv[2][2], bq[4];
  float4 sg[2], s8[2];
};
__device__ __forceinline__ void loc_load(const uint8_t* st, const uint8_t* bsrc, const float4* tr, int a0, int a1, LocOps& o) {
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) o.wv[mt][0] = ldsm4(st + a0 + mt * 1024), o.wv[mt][1] = ldsm4(st + a1 + mt * 1024);
#pragma unroll
  for (int q = 0; q < 4; ++q) o.bq[q] = *reinterpret_cast<const uint4*>(bsrc + q * TILE * 64);
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) o.sg[mt] = tr[16 * mt], o.s8[mt] = tr[16 * mt + 8];
}
__device__ __forceinline__ void loc_stage2(Smem& sm, int& rs, unsigned& ph, const uint8_t* bxs, const float4* trA, const float4* trB, int a0, int a1,
                                           int lane, unsigned zmask, float (&tot)[2][4]) {
  const int rsB = rs + 1 == STAGES ? 0 : rs + 1;
  LocOps A, B;
  loc_load(sm.ring[rs], bxs + rs * XSL, trA, a0, a1, A);
  loc_load(sm.ring[rsB], bxs + rsB * XSL, trB, a0, a1, B);
  if constexpr (PXF) fence_proxy_async();
  __syncwarp();
  unsigned lo = 0;
  if (lane == 0) {
    unsigned long long ta, tb;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(ta) : "r"(smem_u32(&sm.empty[rs])) : "memory");
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(tb) : "r"(smem_u32(&sm.empty[rsB])) : "memory");
    lo = unsigned(ta) | unsigned(tb);
  }
  unsigned z;
  asm("and.b32 %0, %1, %2;" : "=r"(z) : "r"(lo), "r"(zmask));
  const float dz = __uint_as_float(z);
  if (rsB + 1 == STAGES) ph ^= 1u;
  if (rs + 1 == STAGES) ph ^= 1u;
  rs = rsB + 1 == STAGES ? 0 : rsB + 1;
  float dA[4][2][4], dB[4][2][4];
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const uint4 wa = A.wv[mt][q >> 1], wb = B.wv[mt][q >> 1];
      mma32_fp4(dA[q][mt], q & 1 ? wa.z : wa.x, q & 1 ? wa.w : wa.y, A.bq[q], dz);
      mma32_fp4(dB[q][mt], q & 1 ? wb.z : wb.x, q & 1 ? wb.w : wb.y, B.bq[q], dz);
    }
  auto acc = [&](const LocOps& o, const float (&d)[4][2][4]) {
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const float f0 = q == 0 ? o.sg[mt].x : q == 1 ? o.sg[mt].y : q == 2 ? o.sg[mt].z : o.sg[mt].w;
        const float f1 = q == 0 ? o.s8[mt].x : q == 1 ? o.s8[mt].y : q == 2 ? o.s8[mt].z : o.s8[mt].w;
        tot[mt][0] = fmaf(d[q][mt][0], f0, tot[mt][0]);
        tot[mt][1] = fmaf(d[q][mt][1], f0, tot[mt][1]);
        tot[mt][2] = fmaf(d[q][mt][2], f1, tot[mt][2]);
        tot[mt][3] = fmaf(d[q][mt][3], f1, tot[mt][3]);
      }
  };
  acc(A, dA);
  acc(B, dB);
}

// Wait for ring slot rs (phase ph); TRACE: thread 0 accumulates the wait into TR_SWAIT (as Ctx::full).
__device__ __forceinline__ void loc_wait(Smem& sm, int rs, unsigned ph, int warp, int lane) {
#if DS41H_DECODER_TRACE
  const unsigned long long t0 = (warp | lane) == 0 ? gtime() : 0ull;
  mbar_wait(&sm.full[rs], ph);
  if ((warp | lane) == 0) sm.rec[TR_SWAIT] += unsigned(gtime() - t0);
#else
  ring_wait(&sm.full[rs], ph);
#endif
}

// This lane's ue8m0 word (4 k32 scales of row 32 warp + lane) -> the warp's float table at tw (as lc_put).
__device__ __forceinline__ void loc_put(float4* tw, unsigned w) {
  const float f0 = __uint_as_float((w * (1u << 23)) & 0x7f800000u), f1 = __uint_as_float((w * (1u << 15)) & 0x7f800000u);
  const float f2 = __uint_as_float((w * (1u << 7)) & 0x7f800000u), f3 = __uint_as_float(__umulhi(w, 0x80000000u) & 0x7f800000u);
  __syncwarp();  // the reads of this buffer two stages ago are done (cross-lane WAR)
  *tw = make_float4(f0, f1, f2, f3);
  __syncwarp();
}

// ldmatrix k32-pair offsets of this lane in a 64 B-row stage tile (m-tile 0; m-tile 1: + 1024): row 32 warp + (lane & 7) +
// 8 (j & 1), chunk (2 h + (j >> 1)) ^ ((row >> 1) & 3) (fp4_off64; + 16 rows keeps the swizzle).
__device__ __forceinline__ int loc_a(int warp, int lane, int h) {
  const int j = lane >> 3, row = 32 * warp + (lane & 7) + (j & 1) * 8;
  return row * 64 + (((2 * h + (j >> 1)) ^ ((row >> 1) & 3)) << 4);
}

// R1F_LOC: stages 1.. of an R1F job's K part (stage 0 acquired by the caller). it = the ring counter at stage 0, sgc = the
// scale groups consumed before the job (stages % 4 == 0). Writes the warp's raw sums (x 2^-14) to out[0..7].
__device__ __noinline__ void r1f_loop(float* out, int it, int sgc, int stages, int warp, int lane, unsigned zmask) {
  Smem& sm = *reinterpret_cast<Smem*>(ds41h_dsmem);  // shared-space symbol: LDS / LDSM with immediate offsets
  const int g = lane >> 2, t = lane & 3, a0 = loc_a(warp, lane, 0), a1 = loc_a(warp, lane, 1);
  const uint8_t* const bx = sm.bt + XS_OFF + g * 64 + t * 16;   // + slot * XSL
  const uint8_t* const sw = sm.bt + SC_OFF + (32 * warp + lane) * 16;  // + group slot * SCL + (s & 3) * 4
  float4* const tw = reinterpret_cast<float4*>(sm.wf) + warp * 32 + lane;    // + parity * GW * 32
  const float4* const tr = reinterpret_cast<float4*>(sm.wf) + warp * 32 + g;  // + parity * GW * 32
  int rs = it % STAGES, so = (sgc % 3) * SCL;
  unsigned ph = unsigned(it / STAGES) & 1u;
  float tot[2][4] = {};
  if constexpr (DS41H_DECODER_R1F_LOC == 2) {  // stage pairs (stages % 4 == 0): A = s (table parity 0), B = s + 1 (parity 1)
#pragma unroll 1
    for (int s = 0; s < stages; s += 2) {
      if (s) loc_wait(sm, rs, ph, warp, lane);
      {
        const int rsB = rs + 1 == STAGES ? 0 : rs + 1;
        loc_wait(sm, rsB, rsB == 0 ? ph ^ 1u : ph, warp, lane);
      }
      const unsigned wA = *reinterpret_cast<const unsigned*>(sw + so + (s & 3) * 4), wB = *reinterpret_cast<const unsigned*>(sw + so + ((s + 1) & 3) * 4);
      loc_put(tw, wA);
      loc_put(tw + GW * 32, wB);
      if (((s + 1) & 3) == 3) so = so == 2 * SCL ? 0 : so + SCL;
      loc_stage2(sm, rs, ph, bx, tr, tr + GW * 32, a0, a1, lane, zmask, tot);
    }
  } else {
#pragma unroll 1
    for (int s = 0; s < stages; ++s) {
      if (s) loc_wait(sm, rs, ph, warp, lane);
      const int par = (s & 1) * GW * 32;
      loc_put(tw + par, *reinterpret_cast<const unsigned*>(sw + so + (s & 3) * 4));
      if ((s & 3) == 3) so = so == 2 * SCL ? 0 : so + SCL;
      loc_stage(sm, rs, ph, bx + rs * XSL, tr + par, a0, a1, lane, zmask, tot);
    }
  }
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) out[mt * 4 + e] = tot[mt][e];
}

// R2F_LOC: stages 1-4 of an R2F job (k32 blocks 4 (s - 1) .. of the h tile in bt[0, HHI)); it = the ring counter at
// stage 1, w0..w3 = this lane's scale words of stages 1-4. io: the warp's sums after stage 0 in, final sums out.
__device__ __noinline__ void r2f_loop(float* io, int it, unsigned w0, unsigned w1, unsigned w2, unsigned w3, int warp, int lane, unsigned zmask) {
  Smem& sm = *reinterpret_cast<Smem*>(ds41h_dsmem);
  const int g = lane >> 2, t = lane & 3, a0 = loc_a(warp, lane, 0), a1 = loc_a(warp, lane, 1);
  const uint8_t* const bx = sm.bt + g * 64 + t * 16;  // + 4 (s - 1) * TILE * 64
  float4* const tw = reinterpret_cast<float4*>(sm.wf) + warp * 32 + lane;
  const float4* const tr = reinterpret_cast<float4*>(sm.wf) + warp * 32 + g;
  int rs = it % STAGES;
  unsigned ph = unsigned(it / STAGES) & 1u;
  float tot[2][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) tot[mt][e] = io[mt * 4 + e];
#pragma unroll 1
  for (int s = 1; s < INTER_STAGES; ++s) {
    loc_wait(sm, rs, ph, warp, lane);
    const int par = (s & 1) * GW * 32;
    loc_put(tw + par, s == 1 ? w0 : s == 2 ? w1 : s == 3 ? w2 : w3);
    loc_stage(sm, rs, ph, bx + 4 * (s - 1) * TILE * 64, tr + par, a0, a1, lane, zmask, tot);
  }
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) io[mt * 4 + e] = tot[mt][e];
}

// R1F job: expert slot a, row block b (gate rows 64 b.. = stage tile rows 0-63, up rows INTER + 64 b.. = 64-127),
// K part c of r1kp (job.stages stages of 128 k). Warps 0/1 hold gate rows 32 w + 16 mt + (g, g+8), warps 2/3
// the matching up rows; the SwiGLU pairs meet in bs. Scale group sgc % 3 holds the scales of 4 stages.
// One R1F segment: rows of (slot, row block b), `len` stages of the job's stage stream starting at job stage st0 (the dispatch
// loop acquired job stage 0; a segment starts a scale group). Epilogue: len == R1F_STAGES (full K) -> SwiGLU in this CTA; else
// the partial goes to hcacc and the `target`-th arrival on the row block finishes it (finish_r1f).
__device__ WG_INLINE void r1f_seg(Ctx& c, int slot, int b, int st0, int len, int target) {
  const Params& p = c.p;
  float tot[2][4] = {};
  // Stage s: scale table (parity s & 1) from scale group sgc % 3, A / B / scales into registers, release.
  auto load = [&](int s, LcRegs& r) {
    if (st0 + s) c.acquire();
    const uint8_t* scl = c.sm.bt + SC_OFF + (c.sgc % 3) * SCL;
    if constexpr (SCL_DIRECT) {
      const int g = c.lane >> 2;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const uint8_t* sr = scl + (32 * c.warp + 16 * mt + g) * 16 + (s & 3) * 4;
        r.sw[mt][0] = *reinterpret_cast<const unsigned*>(sr);
        r.sw[mt][1] = *reinterpret_cast<const unsigned*>(sr + 8 * 16);
      }
    } else if constexpr ((ABL & 1) == 0) {
      lc_put(c, s & 1, *reinterpret_cast<const unsigned*>(scl + (32 * c.warp + c.lane) * 16 + (s & 3) * 4));
    }
    if ((s & 3) == 3) ++c.sgc;
    if constexpr ((ABL & 1) == 0) lc_load<4, LC_PS>(c, c.stage(), c.sm.bt + XS_OFF + (c.it % STAGES) * XSL, lc_table(c, s & 1), r);
    r.dz = lc_release(c, tot);
  };
  if constexpr (R1F_LOC) {
    float o[8];
    lazy_flush(c);
    r1f_loop(o, c.it, c.sgc, len, c.warp, c.lane, unsigned(p.nranks) >> 16);  // nranks <= 4: zmask 0 at run time
    c.it += len, c.sgc += len >> 2;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int e = 0; e < 4; ++e) tot[mt][e] = o[mt * 4 + e];
  } else if constexpr (R1F_PIPE) {  // stage s + 1 in registers before stage s's math (len is even)
    LcRegs r0, r1;
    load(0, r0);
#pragma unroll 1
    for (int s = 0; s < len; s += 2) {
      load(s + 1, r1);
      lc_math<4, LC_PS>(r0, tot, c.lane);
      if (s == 0) lazy_flush(c);
      if (s + 2 < len) load(s + 2, r0);
      lc_math<4, LC_PS>(r1, tot, c.lane);
    }
  } else {
#pragma unroll 1
    for (int s = 0; s < len; ++s) {
      LcRegs r;
      load(s, r);
      lc_math<4, LC_PS>(r, tot, c.lane);
      if (s == 0) lazy_flush(c);
    }
  }
  if (R1SKU && (len & 3)) ++c.sgc;  // the segment's short last scale group
  c.tmark(TR_X1);
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) tot[mt][e] *= FP4_F16;
  if (len != R1F_STAGES) {  // K part: accumulate the partial, the last arrival finishes
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const float4 v = transpose4(tot[mt], c.lane);
      const int tok = quad_col(0, c.lane);
      const int row = (c.warp < 2 ? 64 * b + 32 * c.warp : INTER + 64 * b + 32 * (c.warp - 2)) + 16 * mt + quad_row(c.lane);
      red4(p.s.hcacc + (long(slot) * TILE + tok) * 2 * INTER + row, v);
    }
    if (c.arrive(C_R1A + slot * R1C_BLOCKS + b, target)) c.tmark(TR_X2), finish_r1f(c, slot, b);
    return;
  }
  float4* xch = reinterpret_cast<float4*>(c.sm.bs);  // [up warp][mt][lane]
  if (c.warp >= 2)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) xch[((c.warp - 2) * 2 + mt) * 32 + c.lane] = make_float4(tot[mt][0], tot[mt][1], tot[mt][2], tot[mt][3]);
  c.sync();
  if (c.warp < 2) {
    const int g = c.lane >> 2, t = c.lane & 3;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const float4 u = xch[(c.warp * 2 + mt) * 32 + c.lane];
      const float up[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int e = 0; e < 4; ++e)
        store_hhi(p, slot, 64 * b + 32 * c.warp + 16 * mt + g + (e & 2 ? 8 : 0), 2 * t + (e & 1), bf(swiglu(bf(tot[mt][e]), bf(up[e]))));
    }
    asm volatile("fence.proxy.async.global;" ::: "memory");  // hhi is read by R2F bulk copies
  }
  c.signal(C_R1D + slot);
  if (R1SK) c.sync();  // the next segment of this job reuses bs (xch) and the scale tables
}

// R1F job: expert slot a, row block b (gate rows 64 b.. = stage tile rows 0-63, up rows INTER + 64 b.. = 64-127),
// K part c of r1kp (job.stages stages of 128 k). Warps 0/1 hold gate rows 32 w + 16 mt + (g, g+8), warps 2/3
// the matching up rows; the SwiGLU pairs meet in bs. Scale group sgc % 3 holds the scales of 4 stages.
// R1SK: a = stream-K job index, b = stages per job T; the job's stage units [a T, a T + stages) are cut at row-block
// boundaries into segments (the segments covering a row block: ceil-floor count of T multiples inside it, + 1).
__device__ WG_INLINE void job_r1f(Ctx& c, const Job& job) {
  c.tmark(TR_X0);
  if constexpr (R1SK) {
    const int T = job.b, u1 = job.a * T + job.stages;
    for (int u = job.a * T, st = 0; u < u1;) {
      const int g = u / R1F_STAGES, len = min(u1, (g + 1) * R1F_STAGES) - u;
      const int segs = (g * R1F_STAGES + R1F_STAGES - 1) / T - g * R1F_STAGES / T + 1;
      r1f_seg(c, g / R1F_BLOCKS, g % R1F_BLOCKS, st, len, segs);
      u += len, st += len;
    }
  } else {
    r1f_seg(c, job.a, job.b, 0, job.stages, R1F_STAGES / job.stages);
  }
}

// R2F job: rows 128 rb .. +127 of expert slot a's w2 (K = 576, the R2C stages: stage 0 = K tail + the job's
// 128 x 18 scale block, stages 1-4 = 64 B) against the slot's f16 h (all 8 token columns), scaled per token by
// its routing weight (0 = not routed: no store), red.add into yracc. R2F_PF: each stage's h slice (k32 blocks
// 16-17 / 4(s-1)..) arrives with the stage in the x-slice ring (issue_r2f); else one bulk copy of the slot's
// whole h tile into bt[0, HHI) after the dependency wait.
__device__ WG_INLINE void job_r2f(Ctx& c, const Job& job) {
  const Params& p = c.p;
  const int slot = job.a;
  const int nblk = R2P > 1 ? job.c + 1 : 1;  // R2P: row blocks job.b .. job.b + nblk - 1 (5 ring stages each)
  const unsigned par = unsigned(c.sm.nh) & 1u;
  if constexpr (!R2F_PF) {
    c.sync();  // the previous job's generic accesses of bt are done
    if (c.tid == 0) {
      fence_proxy_async();
      mbar_expect_tx(&c.sm.hbar, HHI);
      bulk_g2s(c.sm.bt, p.s.hhi + long(slot) * HHI, HHI, &c.sm.hbar, evict_first_policy());
    }
  }
  // routing weight of token `lane` (0 = not routed)
  const float wt = c.lane >= TILE ? 0.0f : PROUTE ? c.sm.wtok_s[slot * TILE + c.lane] : __ldcg(p.s.wtok + slot * TILE + c.lane);
#pragma unroll 1
  for (int blk = 0; blk < nblk; ++blk) {
  const int rb = job.b + blk;
  if (R2P > 1 && blk) c.acquire();  // stage 0 of block blk (block 0's was acquired by the dispatch loop)
  // This lane's row (32 warp + lane) of the stage-0 scale block: 18 ue8m0 bytes, 2-byte aligned.
  unsigned sw[5] = {};
  if constexpr ((ABL & 1) == 0) {
    const unsigned short* sr = reinterpret_cast<const unsigned short*>(c.stage() + R2_TAIL_BYTES + (32 * c.warp + c.lane) * S2_ROW);
#pragma unroll
    for (int i = 0; i < 4; ++i) sw[i] = unsigned(sr[2 * i]) | (unsigned(sr[2 * i + 1]) << 16);
    sw[4] = sr[8];
  }
  if (!R2F_PF && blk == 0) {
    mbar_wait(&c.sm.hbar, par);
    c.sync();
    if (c.tid == 0) ++c.sm.nh;
  }
  if (blk == 0) c.tmark(TR_X0);
  // B source of the stage at the current ring position (R2F_PF) or of k32 block kb in the h tile
  auto bsrc = [&](int kb) -> const uint8_t* { return R2F_PF ? c.sm.bt + XS_OFF + (c.it % STAGES) * XSL : c.sm.bt + kb * TILE * 64; };
  float tot[2][4] = {};
  LcRegs r0, r1;
  // SCL_DIRECT: rows 32 warp + 16 mt + g (+8) of the stage's scale word, from the lanes that loaded them
  auto rows_sw = [&](unsigned w, LcRegs& r) {
    const int g = c.lane >> 2;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      r.sw[mt][0] = __shfl_sync(0xffffffffu, w, 16 * mt + g);
      r.sw[mt][1] = __shfl_sync(0xffffffffu, w, 16 * mt + g + 8);
    }
  };
  if constexpr ((ABL & 1) == 0) {
    if constexpr (SCL_DIRECT) rows_sw(sw[4], r0);
    else lc_put(c, 0, sw[4]);  // stage 0: k32 blocks 16, 17
    lc_load<2, LC_PS>(c, c.stage(), bsrc(16), lc_table(c, 0), r0);
  }
  r0.dz = lc_release(c, tot);
  auto load = [&](int s, LcRegs& r) {  // stages 1-4: k32 blocks 4 (s - 1) ..
    c.acquire();
    if constexpr ((ABL & 1) == 0) {
      // runtime s (R2F_ROLL): select chain, keeps sw in registers
      const unsigned ws = s == 1 ? sw[0] : s == 2 ? sw[1] : s == 3 ? sw[2] : sw[3];
      if constexpr (SCL_DIRECT) rows_sw(ws, r);
      else lc_put(c, s & 1, ws);
      lc_load<4, LC_PS>(c, c.stage(), bsrc(4 * (s - 1)), lc_table(c, s & 1), r);
    }
    r.dz = lc_release(c, tot);
  };
  if constexpr (R1F_PIPE) {
    load(1, r1);
    lc_math<2, LC_PS>(r0, tot, c.lane);
    lazy_flush(c);
    load(2, r0);
    lc_math<4, LC_PS>(r1, tot, c.lane);
    load(3, r1);
    lc_math<4, LC_PS>(r0, tot, c.lane);
    load(4, r0);
    lc_math<4, LC_PS>(r1, tot, c.lane);
    lc_math<4, LC_PS>(r0, tot, c.lane);
  } else if constexpr (R2F_LOC) {
    lc_math<2, LC_PS>(r0, tot, c.lane);
    lazy_flush(c);
    float io[8];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int e = 0; e < 4; ++e) io[mt * 4 + e] = tot[mt][e];
    r2f_loop(io, c.it, sw[0], sw[1], sw[2], sw[3], c.warp, c.lane, unsigned(p.nranks) >> 16);  // zmask 0 at run time
    c.it += INTER_STAGES - 1;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int e = 0; e < 4; ++e) tot[mt][e] = io[mt * 4 + e];
  } else if constexpr (DS41H_DECODER_R2F_ROLL != 0) {
    lc_math<2, LC_PS>(r0, tot, c.lane);
    lazy_flush(c);
#pragma unroll 1
    for (int s = 1; s < INTER_STAGES; ++s) {
      load(s, r1);
      lc_math<4, LC_PS>(r1, tot, c.lane);
    }
  } else {
    lc_math<2, LC_PS>(r0, tot, c.lane);
    lazy_flush(c);
#pragma unroll
    for (int s = 1; s < INTER_STAGES; ++s) {
      load(s, r1);
      lc_math<4, LC_PS>(r1, tot, c.lane);
    }
  }
  static_assert(INTER_STAGES == 5, "R2F stage schedule");
  if (blk == nblk - 1) c.tmark(TR_X1);
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const float4 v = transpose4(tot[mt], c.lane);
    const int tok = quad_col(0, c.lane), row = 32 * c.warp + 16 * mt + quad_row(c.lane);
    const float w = __shfl_sync(0xffffffffu, wt, tok) * FP4_F16;
    if (w != 0.0f) red4(p.s.yracc + long(tok) * HID + rb * 128 + row, make_float4(w * v.x, w * v.y, w * v.z, w * v.w));
  }
  if constexpr (LAZY_SIG) {
    c.lazy = rb + 1;  // posted by lazy_flush at the next job
  } else if constexpr (R2P == 1) {
    c.signal(C_SF + rb);
  }
  }  // blocks
  if constexpr (R2P > 1) {  // the C_SF arrivals of all blocks of the job, after all their red4s
    c.sync();
    if (c.tid == 0)
      for (int blk = 0; blk < nblk; ++blk) red_release(c.cs() + C_SF + job.b + blk, 1u);
  }
}

// XF_P1A: with p.p1_resid set, each XF job also runs
// the next layer's prologue phase P1a for its chunk's 4 column slices (warp w = slice 4 k + w, lane = column) and all m
// tokens, bit-identical to prologue.cu p1_collapse (MODE 0, no aux capture): new_r[j] = post[j] * y, then
// fma(comb[k][j], resid1[k], .) for k = 0..3 (explicit RN, no contraction), bf16 -> p1_resid; xc = pre[0] * bf16(new_r[0]),
// then fma(pre[j], bf16(new_r[j]), .) -> bf16 p1_xcb; p1_xsq[tok][slice] = warp_sum(xf * xf) (xor butterfly).
// y = this layer's FFN output (y_ffn, the chunk just reduced by this CTA), post / comb / pre = its MIX outputs.
// Staging in bt (free during XF): co = [tok][24] fp32 (post[4] | pre[4] | comb[4][4]), rr = [tok][4 streams][128 columns]
// bf16 resid1 of the chunk. xf_p1a_pre issues them as cp.async (after the MIX acquire) BEFORE the all-reduce, so the
// P1a tail after it only waits for the y loads.
constexpr int P1A_RR = COARSE_M * 24 * 4;  // byte offset of rr in bt (XF_P1A: m <= COARSE_M, checked on the host)
static_assert(P1A_RR + COARSE_M * HC * CHUNK_COLS * 2 <= BT, "XF_P1A staging fits in bt");
__device__ CTXFN void xf_p1a_pre(Ctx& c, int k) {
  const Params& p = c.p;
  c.wait(C_MX, quads(p.m));  // MIX jobs done (acquire, long before in practice)
  float* co = reinterpret_cast<float*>(c.sm.bt);
  __nv_bfloat16* rr = reinterpret_cast<__nv_bfloat16*>(c.sm.bt + P1A_RR);
  for (int i = c.tid; i < p.m * 6; i += GT) {  // 16-byte pieces: post, pre, comb rows 0..3
    const int tok = i / 6, j = i % 6;
    const float* src = j == 0 ? p.ffn_post + tok * HC : j == 1 ? p.ffn_pre + tok * HC : p.ffn_comb + tok * HC * HC + (j - 2) * HC;
    cp_async16(co + tok * 24 + j * 4, src);
  }
  for (int i = c.tid; i < p.m * HC * (CHUNK_COLS / 8); i += GT) {  // 16 columns of one (token, stream) per piece
    const int v = i % (CHUNK_COLS / 8), ts = i / (CHUNK_COLS / 8);
    cp_async16(rr + ts * CHUNK_COLS + v * 8, p.resid1 + long(ts) * HID + k * CHUNK_COLS + v * 8);
  }
}

__device__ CTXFN void xf_p1a(Ctx& c, int k) {
  const Params& p = c.p;
  cp_async_wait_all();
  c.sync();  // staging landed; this CTA's y_ffn chunk writes (share) ordered before the reads below
  const float* co = reinterpret_cast<const float*>(c.sm.bt);
  const __nv_bfloat16* rr = reinterpret_cast<const __nv_bfloat16*>(c.sm.bt + P1A_RR);
  static_assert(CHUNK_COLS / 32 == GW, "one 32-column slice per consumer warp");
  const int s = k * GW + c.warp, cc = c.warp * 32 + c.lane, col = k * CHUNK_COLS + cc;
  constexpr int TU = COARSE_M;  // all y loads of a coarse step in flight together
  for (int t0 = 0; t0 < p.m; t0 += TU) {
    float y[TU];
#pragma unroll
    for (int u = 0; u < TU; ++u) y[u] = __bfloat162float(__ldcg(p.y_ffn + long(t0 + u < p.m ? t0 + u : t0) * HID + col));
#pragma unroll
    for (int u = 0; u < TU; ++u) {
      const int tok = t0 + u;
      if (tok >= p.m) break;  // warp-uniform
      const float* cf = co + tok * 24;
      float rb[HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float t = __fmul_rn(cf[j], y[u]);
#pragma unroll
        for (int kk = 0; kk < HC; ++kk) t = __fmaf_rn(cf[8 + kk * HC + j], __bfloat162float(rr[(tok * HC + kk) * CHUNK_COLS + cc]), t);
        const __nv_bfloat16 b = __float2bfloat16(t);
        p.p1_resid[(long(tok) * HC + j) * HID + col] = b;
        rb[j] = __bfloat162float(b);
      }
      float xc = __fmul_rn(cf[4], rb[0]);
#pragma unroll
      for (int j = 1; j < HC; ++j) xc = __fmaf_rn(cf[4 + j], rb[j], xc);
      const __nv_bfloat16 xb = __float2bfloat16(xc);
      p.p1_xcb[long(tok) * HID + col] = xb;
      const float xf = __bfloat162float(xb);
      const float xs = warp_sum(xf * xf);
      if (c.lane == 0) __stcg(p.p1_xsq + tok * SLICES + s, xs);
    }
  }
  c.sync();  // bt reads done before the next job may stage into bt
}

// Combine bf16(routed) + bf16(shared) and all-reduce chunk k.
__device__ CTXFN void job_xf(Ctx& c, int k) {
  const Params& p = c.p;
  c.release();
  if (p.p1_resid != nullptr) xf_p1a_pre(c, k);  // XF_P1A staging overlaps the combine + all-reduce
  __nv_bfloat16* mine = inbox(p, p.rank, 1, 0, c.epoch);
  if (LLP && p.m <= COARSE_M) {  // LLP all-reduce #2 of chunk k: push this rank's words, poll the local inbox, reduce4's sum order
    const unsigned live = peer_mask(p) | (1u << p.rank);
    const int nv = p.m * CHUNK_VECS, i = c.tid;  // <= 128 vectors: one per thread
    if (i < nv) {
      const long off = chunk_off(i, k);
      const Acc8 rv = load8(p.s.yracc + off, true), sv = load8(p.s.shacc + off, true);
      clear8(p.s.yracc + off);
      clear8(p.s.shacc + off);
      float routed_v[8], shared_v[8], y[8];
      unpack8(rv, routed_v);
      unpack8(sv, shared_v);
#pragma unroll
      for (int e = 0; e < 8; ++e) y[e] = bf(routed_v[e]) + bf(shared_v[e]);
      const uint4 pk = pack8_bf16(y);
      const int tok = i / CHUNK_VECS, col = k * CHUNK_COLS + (i % CHUNK_VECS) * 8;
#pragma unroll
      for (int d = 0; d < RANKS; ++d) {
        if (!(live >> d & 1)) continue;
        uint2* dst = ll_box(p, d, 1, c.epoch, p.rank) + long(tok) * (HID / 2) + col / 2;
        ll_put(dst, pk.x, c.epoch);
        ll_put(dst + 1, pk.y, c.epoch);
        ll_put(dst + 2, pk.z, c.epoch);
        ll_put(dst + 3, pk.w, c.epoch);
      }
      c.tmark(TR_X0);
      const uint2* src[RANKS * 4];
      uint2 w[RANKS * 4];
#pragma unroll
      for (int r = 0; r < RANKS; ++r)
#pragma unroll
        for (int j = 0; j < 4; ++j) src[r * 4 + j] = ll_box(p, p.rank, 1, c.epoch, (live >> r & 1) ? r : p.rank) + long(tok) * (HID / 2) + col / 2 + j;
      ll_wait(w, src, c.epoch);
      c.tmark(TR_X1);
      float acc[8] = {};
#pragma unroll
      for (int r = 0; r < RANKS; ++r)
        add8_bf16(acc, (live >> r & 1) ? make_uint4(w[r * 4].x, w[r * 4 + 1].x, w[r * 4 + 2].x, w[r * 4 + 3].x) : make_uint4(0, 0, 0, 0));
      *reinterpret_cast<uint4*>(p.y_ffn + off) = pack8_bf16(acc);
    }
    c.tmark(TR_X2);
    if (p.p1_resid != nullptr) xf_p1a(c, k);
    if (TAILFREE && k == 0 && c.tid == 0) atomicAdd(p.s.epoch, 1u);
    return;
  }
  constexpr int XU = 2;
  for (int base = 0; base < p.m * CHUNK_VECS; base += GT * XU) {
    Acc8 rv[XU], sv[XU];
#pragma unroll
    for (int u = 0; u < XU; ++u) {
      const int i = base + c.tid + u * GT;
      const bool live = i < p.m * CHUNK_VECS;
      const long off = chunk_off(live ? i : 0, k);
      rv[u] = load8(p.s.yracc + off, live);
      sv[u] = load8(p.s.shacc + off, live);
    }
#pragma unroll
    for (int u = 0; u < XU; ++u) {
      const int i = base + c.tid + u * GT;
      if (i >= p.m * CHUNK_VECS) break;
      const long off = chunk_off(i, k);
      clear8(p.s.yracc + off);
      clear8(p.s.shacc + off);
      float routed_v[8], shared_v[8], y[8];
      unpack8(rv[u], routed_v);
      unpack8(sv[u], shared_v);
#pragma unroll
      for (int e = 0; e < 8; ++e) y[e] = bf(routed_v[e]) + bf(shared_v[e]);
      *reinterpret_cast<uint4*>(mine + off) = pack8_bf16(y);
    }
  }
  share(c, 1, k, p.y_ffn);
  if (p.p1_resid != nullptr) xf_p1a(c, k);
  if (TAILFREE && k == 0 && c.tid == 0) atomicAdd(p.s.epoch, 1u);  // every CTA read this launch's epoch before its first job
}

// ------------------------------------------------------------------------------ scheduling
__device__ CTXFN void deps(Ctx& c, const Job& job) {
  const Params& p = c.p;
  if constexpr (DS41H_DECODER_HOTDISP != 0 && LC) {  // routed types first (see HOTDISP)
    if (job.type == J_R1C) return;
    if (job.type == J_R2C) {
      if constexpr (R2F_PF) return;
      return c.wait(C_R1D + job.a, R1F_BLOCKS);
    }
  }
  switch (job.type) {
    case J_WOB:
      if (WOB3 && wob3_m(p.m)) {  // the 512-column z ranges (C_Z, KS_WOB k32 blocks each) that K part job.b spans
        const int r0 = 20 * job.b / KS_WOB, r1 = (20 * job.b + 4 * wob3_st(job.b) - 1) / KS_WOB;
        return c.wait_n(C_Z + r0 * GROUPS + job.c, GROUPS, r1 - r0 + 1, KS_WOB / 2 * (OAK / 32 / woa_ks(p.m)));
      }
      if constexpr (FUSED_Z) return c.wait(C_Z + job.b * GROUPS + job.c, wob_ks(p.m) / 2 * (OAK / 32 / woa_ks(p.m)));  // every wo_a job of the range
      // finish_woa signals per 512-column range: the job's K part covers wob_ks / 16 of them
      return c.wait_n(C_Z + job.b * (wob_ks(p.m) / KS_WOB) * GROUPS + job.c, GROUPS, wob_ks(p.m) / KS_WOB, KS_WOB / 2);
    case J_XA: return c.wait(C_YC + job.a, 2 * token_groups(p.m));
    case J_OWNA:
      if constexpr (FUSED_AR1) return c.wait(C_YC + job.a / 2, token_groups(p.m));  // own row block published
      return c.wait(C_YR + job.a / (CHUNK_COLS / 32), 1);
    case J_XNORM: return c.wait(C_A + job.b, SLICES);
    case J_GATE:
      if (!split_norm(p.m))
        for (int t = 0; t < tiles_of(p.m); ++t) c.wait(C_A + t, SLICES);
      return;
    case J_SH1: {
      if (LCQ && coarse_m(p.m)) return c.wait_n(C_XK, GROUPS, KPARTS, 512 / (KS_GATE * 16));  // resident x tile: every x part
      const int k0 = job.b * KS_SH1 * 32;
      const unsigned target = split_norm(p.m) ? group_of(p.m, job.c).tiles : 512 / (KS_GATE * 16);
      for (int part = k0 / 512; part < (k0 + KS_SH1 * 32 + 511) / 512; ++part) c.wait(C_XK + part * GROUPS + job.c, target);
      return;
    }
    case J_SH2: return c.wait(C_HS + job.c, SH1_BLOCKS);
    case J_MIX: return c.wait(C_A + job.a * GW / TILE, SLICES);
    // R1 reads xh written before C_G (gate) / C_XK; route_all acquires C_G (last arrival) and the
    // producer acquires C_ROUTE before decoding any routed job, which orders those writes.
    case J_R2: return c.wait(C_R1D + job.a, R1_BLOCKS);
    case J_R2C:
      if constexpr (R2F_PF) return;  // the producer checked C_R1D before streaming the h slices
      return c.wait(C_R1D + job.a, LC ? R1F_BLOCKS : R1C_BLOCKS);
    case J_XF: return c.wait(C_SF + job.a, (PROUTE && coarse_m(p.m) ? unsigned(c.sm.nent) : __ldcg(c.cs() + C_NENT)) + 2 * token_groups(p.m));
    default: return;
  }
}

__device__ WG_INLINE void run_job(Ctx& c, const Job& job) {
  if constexpr (DS41H_DECODER_HOTDISP != 0 && LC) {  // routed types first (see HOTDISP)
    if (job.type == J_R1C) return job_r1f(c, job);
    if (job.type == J_R2C) return job_r2f(c, job);
  }
  switch (job.type) {
    case J_WOA: return job_woa(c, job.a, job.b, job.c, job.stages);
    case J_WOB: return job_wob(c, job.a, job.b, job.c);
    case J_XA: return job_xa(c, job.a);
    case J_OWNA: return job_owna(c, job.a, job.b);
    case J_XNORM: return job_xnorm(c, job.a, job.b);
    case J_GATE: return job_gate(c, job.a, job.b, job.c);
    case J_SH1: return job_sh1(c, job.a, job.b, job.c);
    case J_SH2: return job_sh2(c, job.a, job.c);
    case J_MIX: return job_mix(c, job.a);
    case J_R1: return job_r1(c, job);
    case J_R2: return job_r2(c, job);
    case J_R1C:
      if constexpr (LC && DS41H_DECODER_HOTDISP != 0) return;  // dispatched above (one inlined copy)
      else if constexpr (LC) return job_r1f(c, job);
      else return job_r1c(c, job);
    case J_R2C:
      if constexpr (LC && DS41H_DECODER_HOTDISP != 0) return;
      else if constexpr (LC) return job_r2f(c, job);
      else return job_r2c(c, job);
    default: return job_xf(c, job.a);
  }
}

__device__ int route_table(const Params& p, int lane, const unsigned* cnt) {
  int n = 0;
  if (lane == 0) {
    unsigned v;
    v = poll_ge(cnt + C_ROUTE, unsigned(p.m));
    n = int(v >> 16);  // route_all: C_ROUTE = m + (entries << 16)
  }
  return __shfl_sync(0xffffffffu, n, 0);
}

// TAGX: C_ROUTE (lane 0) and the router's tagged {expert, epoch} entries (every lane: slots lane, lane + 32) polled together with
// relaxed loads; done when C_ROUTE >= m and every slot < n carries this launch's epoch; one fence.acquire; experts -> smem.
__device__ int route_table_tagx(const Params& p, Smem& sm, int lane, const unsigned* cnt, unsigned ep) {
  const unsigned long long* xt = p.s.xtag;
  unsigned long long e0 = 0, e1 = 0;
  int n;
  for (;;) {
    const unsigned v = __shfl_sync(0xffffffffu, lane == 0 ? ld_relaxed_gpu(cnt + C_ROUTE) : 0u, 0);
    e0 = ld_relaxed_gpu64(xt + lane);
    e1 = lane + 32 < MAXE ? ld_relaxed_gpu64(xt + lane + 32) : 0ull;
    if (v >= unsigned(p.m)) {
      n = int(v >> 16);
      const bool ok = (lane >= n || unsigned(e0 >> 32) == ep) && (lane + 32 >= n || unsigned(e1 >> 32) == ep);
      if (__all_sync(0xffffffffu, ok)) break;
    }
    __nanosleep(SPIN_NS);
  }
  fence_acquire_gpu();
  if (lane < n) sm.sk_expert[lane] = int(e0 & 0xffffu);
  if (lane + 32 < n) sm.sk_expert[lane + 32] = int(e1 & 0xffffu);
  __syncwarp();
  return n;
}

// The job read from smem is the same for every consumer thread; a lane-0 shuffle makes the job
// dispatch provably warp-uniform for ptxas (8 SHFL per job).
__device__ __forceinline__ Job uniform_job(const Job& j) {
  Job u;
  u.type = __shfl_sync(0xffffffffu, j.type, 0);
  u.a = __shfl_sync(0xffffffffu, j.a, 0);
  u.b = __shfl_sync(0xffffffffu, j.b, 0);
  u.c = __shfl_sync(0xffffffffu, j.c, 0);
  u.stages = __shfl_sync(0xffffffffu, j.stages, 0);
  u.e = __shfl_sync(0xffffffffu, j.e, 0);
  u.f = __shfl_sync(0xffffffffu, j.f, 0);
  u.n = __shfl_sync(0xffffffffu, j.n, 0);
  return u;
}

// ------------------------------------------------------------------------------ producer
struct Issue {
  uint8_t* stage;
  int lane;
  unsigned long long policy;
  unsigned long long* bar = nullptr;  // DENSE_BULK: the stage's full barrier (lane 0 arrives with expect_tx)

  __device__ __forceinline__ void block(const uint8_t* src) const {
    if (DENSE_BULK && bar) {
      if (lane == 0) {
        mbar_expect_tx(bar, FRAGS * FRAG);
        bulk_g2s(stage, src, FRAGS * FRAG, bar, policy);
      }
      return;
    }
#pragma unroll
    for (int f = 0; f < FRAGS; ++f) cp_async16_hint(stage + f * FRAG + lane * 16, src + f * FRAG + lane * 16, policy);
  }

  // Two 32-row blocks x 4 K blocks of ue8m0 scales (rows `ld` bytes apart).
  __device__ __forceinline__ void scales(const uint8_t* src, long ld) const {
    if (lane < 2) cp_async4(stage + SCALE_OFF + lane * 4, src + lane * ld);
  }
};

__device__ __forceinline__ void issue(const Params& p, const Job job, int s, const Issue is) {
  const Weights& w = p.w;
  switch (job.type) {
    case J_WOA: {
      const int kg = job.b * job.stages + s;
      is.block(w.woa + (long(job.a) * KG_WOA + kg) * FRAGS * FRAG);
      is.scales(w.woa_s + long(job.a * 2) * (OAK / 32) + kg * 4, OAK / 32);
      return;
    }
    case J_WOB: {
      const int kg = (wob3_m(p.m) ? 5 * job.b : job.b * job.stages) + s;
      is.block(w.wob + (long(job.a) * KG_WOB + kg) * FRAGS * FRAG);
      is.scales(w.wob_s + long(job.a * 2) * (OA / 32) + kg * 4, OA / 32);
      return;
    }
    case J_GATE: is.block(w.gate + (long(job.a) * KG_GATE + job.b * (KS_GATE / 4) + s) * FRAGS * FRAG); return;
    case J_SH1: {
      const int kg = job.b * (KS_SH1 / 4) + s;
      is.block(w.sh13 + (long(job.a) * KG_SH1 + kg) * FRAGS * FRAG);
      is.scales(w.sh13_s + long(job.a * 2) * SLICES + kg * 4, SLICES);
      return;
    }
    case J_SH2:
      is.block(w.sh2 + (long(job.a) * INTER_STAGES + s) * FRAGS * FRAG);
      is.scales(w.sh2_s + long(job.a * 2) * SH2_LD + s * 4, SH2_LD);
      return;
    case J_OWNA:
      if (DENSE_BULK && is.bar) {  // 128-byte fn row pieces by bulk copies (lane 0 arrives with expect_tx first)
        const int pieces = (s ? MIX - OWNA_ROWS : OWNA_ROWS) * HC;
        if (is.lane == 0) mbar_expect_tx(is.bar, pieces * 128);
        __syncwarp();
        for (int i = is.lane; i < pieces; i += 32) {
          const int piece = s * OWNA_ROWS * HC + i, r = piece / HC, hc = piece % HC;
          bulk_g2s(is.stage + i * 128, w.fn + long(r) * HCD + hc * HID + job.a * 32, 128, is.bar, is.policy);
        }
        return;
      }
      for (int i = is.lane; i < (s ? MIX - OWNA_ROWS : OWNA_ROWS) * HC * 8; i += 32) {
        const int piece = s * OWNA_ROWS * HC + i / 8, part = i % 8, r = piece / HC, hc = piece % HC;
        cp_async16(is.stage + (i / 8) * 128 + part * 16, w.fn + long(r) * HCD + hc * HID + job.a * 32 + part * 4);
      }
      return;
    case J_R1: {
      // Rows 0-63: gate (w1) rows bb*64.., rows 64-127: up (w3) rows INTER + bb*64..; bytes
      // kp*256 + s*64 .. +63 (K blocks kp*16 + 4s .. +3) of each 2560-byte raw row.
      const uint8_t* base = w.w13 + job.e * w.w13_e + job.c * (KS_R1 * 16) + s * 64;
      const int bb = job.b;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int idx = j * 32 + is.lane, row = idx >> 2, ch = idx & 3;
        const long grow = row < 64 ? bb * 64 + row : INTER + bb * 64 + row - 64;
        cp_async16_hint(is.stage + fp4_off64(row, ch), base + grow * W13_ROW + ch * 16, is.policy);
      }
      const uint8_t* sbase = w.s13 + job.e * w.s13_e + job.c * KS_R1 + s * 4;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int row = j * 32 + is.lane;
        const long grow = row < 64 ? bb * 64 + row : INTER + bb * 64 + row - 64;
        cp_async4(is.stage + SCALE_OFF + row * 4, sbase + grow * S13_ROW);
      }
      return;
    }
    case J_R1C: {
      // 64 rows (0-31: gate rows 32 b.., 32-63: up rows INTER + 32 b..) x 128-byte K chunk
      // kc = c * stages + s of each 2560-byte raw row (fp4_off128 swizzle) + their 8 scale bytes.
      const int kc = job.c * job.stages + s;
      const uint8_t* base = w.w13 + job.e * w.w13_e + kc * 128;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int idx = j * 32 + is.lane, row = idx >> 3, ch = idx & 7;
        const long grow = row < 32 ? 32 * job.b + row : INTER + 32 * job.b + row - 32;
        cp_async16_hint(is.stage + fp4_off128(row, ch), base + grow * W13_ROW + ch * 16, is.policy);
      }
      const uint8_t* sbase = w.s13 + job.e * w.s13_e + kc * 8;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int row = j * 32 + is.lane;
        const long grow = row < 32 ? 32 * job.b + row : INTER + 32 * job.b + row - 32;
        cp_async8(is.stage + SCALE_OFF + row * 8, sbase + grow * S13_ROW);
      }
      return;
    }
    case J_R2C:
    case J_R2: {
      // Rows rb*128 .. +127 of w2 (288 bytes each). Stage 0: K tail bytes 256..287 + the
      // job's 2304-byte scale block; stage s >= 1: bytes (s-1)*64 .. +63.
      const uint8_t* base = w.w2 + job.e * w.w2_e + long(job.b) * 128 * W2_ROW;
      if (s == 0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int idx = j * 32 + is.lane, row = idx >> 1, ch = idx & 1;
          cp_async16_hint(is.stage + fp4_off32(row, ch), base + row * W2_ROW + 256 + ch * 16, is.policy);
        }
        const uint8_t* sbase = w.s2 + job.e * w.s2_e + long(job.b) * 128 * S2_ROW;
        for (int idx = is.lane; idx < R2_SCALE_BYTES / 16; idx += 32) cp_async16_hint(is.stage + R2_TAIL_BYTES + idx * 16, sbase + idx * 16, is.policy);
      } else {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int idx = j * 32 + is.lane, row = idx >> 2, ch = idx & 3;
          cp_async16_hint(is.stage + fp4_off64(row, ch), base + row * W2_ROW + (s - 1) * 64 + ch * 16, is.policy);
        }
      }
      return;
    }
    default: return;
  }
}

// ------------------------------------------------------------------------------ TMA (coarse R1 stages)
__device__ __forceinline__ void mbar_expect_tx(unsigned long long* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma3d(void* dst, const CUtensorMap* map, int c0, int c1, int c2, unsigned long long* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void tma4d(void* dst, const CUtensorMap* map, int c0, int c1, int c2, int c3, unsigned long long* bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5}], [%6], %7;" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_u32(bar)), "l"(pol)
      : "memory");
}

// One coarse R1 stage by TMA (one lane): gate / up rows 32 b.. of expert e, 128-byte K chunk kc ->
// [32 rows][128 B] 128B-swizzled boxes at 0 / 4096 (= fp4_off128 layout), and the 16 scale bytes of
// the stage pair kc & ~1 of the 64 rows at SCALE_OFF (row r at r * 16; the stage's 8 at (kc & 1) * 8).
__device__ __forceinline__ void issue_r1c_tma(const Params& p, const Job& job, int s, uint8_t* st, unsigned long long* bar, uint64_t pol) {
  const int kc = job.c * job.stages + s;
  mbar_expect_tx(bar, 2 * 4096 + 2 * 512);
  tma4d(st, &p.w.tm_w13, 0, 32 * job.b, kc, job.e, bar, pol);
  tma4d(st + 4096, &p.w.tm_w13, 0, INTER + 32 * job.b, kc, job.e, bar, pol);
  tma3d(st + SCALE_OFF, &p.w.tm_s13, (kc >> 1) * 16, 32 * job.b, job.e, bar, pol);
  tma3d(st + SCALE_OFF + 512, &p.w.tm_s13, (kc >> 1) * 16, INTER + 32 * job.b, job.e, bar, pol);
}

__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, unsigned long long* bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(smem_u32(dst)), "l"(src),
               "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
               : "memory");
}

// One coarse R2 stage by TMA (one lane), same smem layout as the cp.async path: stage 0 = K tail
// (bytes 256..287, fp4_off32 = 32B swizzle) + the job's 128 x 18 scale block; stage s >= 1 = bytes
// 64 (s - 1).. of the 128 rows (fp4_off64 = 64B swizzle).
__device__ __forceinline__ void issue_r2c_tma(const Params& p, const Job& job, int s, uint8_t* st, unsigned long long* bar, uint64_t pol) {
  const Weights& w = p.w;
  const int row0 = job.b * 128;
  if (s == 0) {
    mbar_expect_tx(bar, R2_TAIL_BYTES + R2_SCALE_BYTES);
    tma3d(st, &w.tm_w2t, 0, row0, job.e, bar, pol);
    bulk_g2s(st + R2_TAIL_BYTES, w.s2 + job.e * w.s2_e + long(row0) * S2_ROW, R2_SCALE_BYTES, bar, pol);
  } else {
    mbar_expect_tx(bar, 128 * 64);
    tma4d(st, &w.tm_w2, 0, row0, s - 1, job.e, bar, pol);
  }
}

// One R1F stage (one lane, LC): gate / up rows 64 b.. of expert e, 64-byte K chunk kc = c * stages + s ->
// two [64 rows][64 B] 64B-swizzled boxes at 0 / 4096 (= the fp4_off64 layout of the 128-row tile), the f16 x
// slice of kc (k32 blocks 4 kc..+3, 2 KB of the gate jobs' xhi image) -> xs, and on every 4th stage the scale
// group kc / 4 (16 B = 4 stages x 4 k32 per row, gate rows then up rows) -> sc (scale ring slot).
__device__ __forceinline__ void issue_r1f(const Params& p, const Job& job, int s, uint8_t* st, uint8_t* xs, uint8_t* sc, unsigned long long* bar, uint64_t pol,
                                          uint64_t xpol) {
  const int kc = job.c * job.stages + s;
  if constexpr ((ABL & 32) != 0) sc = nullptr;
  mbar_expect_tx(bar, 8192 + XSL + (sc ? SCL : 0));
  tma4d(st, &p.w.tm_w13f, 0, 64 * job.b, kc, job.e, bar, pol);
  tma4d(st + 4096, &p.w.tm_w13f, 0, INTER + 64 * job.b, kc, job.e, bar, pol);
  bulk_g2s(xs, p.s.xhi + long(kc) * XSL, XSL, bar, xpol);
  if (sc) {
    const int sx = R1SKU ? kc * 4 : (kc >> 2) * 16;  // R1SKU: the 4 stages from kc (any kc; past K = OOB zeros)
    tma3d(sc, &p.w.tm_s13f, sx, 64 * job.b, job.e, bar, pol);
    tma3d(sc + SCL / 2, &p.w.tm_s13f, sx, INTER + 64 * job.b, job.e, bar, pol);
  }
}

// One R2F stage (one lane, LC + R2F_PF): the R2C stage (stage 0: K tail + the job's scale block, s >= 1: 64 B of
// the 128 rows) plus the slot's h slice for the stage (stage 0: k32 blocks 16-17 = 1 KB, s >= 1: 4 (s-1).. = 2 KB)
// -> hs (x-slice ring). The h slices are only issued once the expert's R1F blocks are done (hready: checked once
// per job by this lane; the consumer then skips that dependency).
__device__ __forceinline__ void issue_r2f(const Params& p, const Job& job, int s, uint8_t* st, uint8_t* hs, unsigned long long* bar, uint64_t pol, uint64_t hpol,
                                          bool& hready) {
  const Weights& w = p.w;
  const int row0 = job.b * 128;
  if (s == 0) {
    mbar_expect_tx(bar, ((ABL & 64) ? 0 : R2_TAIL_BYTES) + R2_SCALE_BYTES + 2 * TILE * 64);
    if constexpr ((ABL & 64) == 0) tma3d(st, &w.tm_w2t, 0, row0, job.e, bar, pol);
    bulk_g2s(st + R2_TAIL_BYTES, w.s2 + job.e * w.s2_e + long(row0) * S2_ROW, R2_SCALE_BYTES, bar, pol);
  } else {
    mbar_expect_tx(bar, 128 * 64 + 4 * TILE * 64);
    tma4d(st, &w.tm_w2, 0, row0, s - 1, job.e, bar, pol);
  }
  if (!hready) {
    poll_ge(p.s.counters + C_R1D + job.a, unsigned(R1F_BLOCKS));
    asm volatile("fence.proxy.async.global;" ::: "memory");  // hhi written by generic stores, read by bulk copies
    hready = true;
  }
  const int kb = s == 0 ? 16 : 4 * (s - 1);
  bulk_g2s(hs, p.s.hhi + long(job.a) * HHI + kb * TILE * 64, (s == 0 ? 2 : 4) * TILE * 64, bar, hpol);
}

// L2 prefetch of slice i of nsl of the byte ranges ptr/len (one lane issues the bulk prefetches);
// last: evict_last policy, else evict_normal.
__device__ __noinline__ void prefetch_ranges(const uint8_t* const* ptr, const long* len, long total, int i, int nsl, bool last) {
  if (total <= 0 || nsl <= 0) return;
  unsigned long long policy;
  if (last) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
  else asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(policy));
  const long chunk = ((total + nsl - 1) / nsl + 255) & ~255L;
  long lo = long(i) * chunk, hi = min(lo + chunk, total), base = 0;
  for (int r = 0; r < PF_MAX && lo < hi; ++r) {
    const long a = max(lo, base), b = min(hi, base + len[r]);
    for (long o = a; o + 16 <= b; o += 65536) {
      const long n = min(65536L, b - o) & ~15L;
      asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(ptr[r] + (o - base)), "r"(unsigned(n)), "l"(policy) : "memory");
    }
    base += len[r];
  }
}

__device__ __forceinline__ void prefetch_slice(const Weights& w, int i) { prefetch_ranges(w.pf, w.pf_len, w.pf_total, i, SLICES, true); }

__device__ void producer(const Params& p, Smem& sm) {
  const int lane = threadIdx.x & 31;
  Layout L = layout(p.m, p.r1kp, p.lag);
  const unsigned long long policy = evict_first_policy();
  unsigned long long xpol;  // LC: the f16 x image is re-read by every R1F job
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(xpol));
  const volatile int* ready = &sm.ready;
  unsigned* pc = p.s.counters;  // this launch's counter set (TAILFREE: from the consumers' epoch, set at the first claim)
  int n = -1, it = 0, j = int(blockIdx.x), sgp = 0;  // sgp: LC scale groups issued (ring slot sgp % 3)
  [[maybe_unused]] int sge1 = -STAGES, sge2 = -STAGES, sge3 = -STAGES;  // R1SKU: ring counter at the end of groups sgp-1/-2/-3
  [[maybe_unused]] int skn = -1;                      // R1SK: n of the slot -> expert copy in smem
  if constexpr (DS41H_DECODER_DYN_FIRST != 0) {  // first job in start order (see DYN_FIRST)
    unsigned f = 0;
    if (lane == 0) f = atomicAdd(p.s.counters + C_FIRST, 1u);
    j = int(__shfl_sync(0xffffffffu, f, 0));
  }
#if DS41H_DECODER_TRACE
  unsigned tq = 0;  // when this job's index was claimed (0 = initial static assignment)
  auto tnow = [&]() { return unsigned(gtime() - sm.t0); };
#endif
  for (int seq = 0;; ++seq) {
#if DS41H_DECODER_TRACE
    unsigned tr = 0;
    if (n < 0 && j >= L.routed)
      n = PROUTE && L.coarse ? route_local_nc(p, sm, lane)
          : (DS41H_DECODER_TAGX != 0 && R1SK && L.coarse) ? route_table_tagx(p, sm, lane, pc, *reinterpret_cast<const volatile unsigned*>(&sm.epoch))
                                                          : route_table(p, lane, pc),
      tr = tnow(), L = with_n(L, n);
#else
    if (n < 0 && j >= L.routed)
      n = PROUTE && L.coarse ? route_local_nc(p, sm, lane)
          : (DS41H_DECODER_TAGX != 0 && R1SK && L.coarse) ? route_table_tagx(p, sm, lane, pc, *reinterpret_cast<const volatile unsigned*>(&sm.epoch))
                                                          : route_table(p, lane, pc),
      L = with_n(L, n);
#endif
    if (R1SK && L.skJ > 0 && skn != n) {  // R1SK: the expert of every slot into smem once (stream-K jobs span slots)
      if (!(DS41H_DECODER_TAGX != 0 && L.coarse))  // TAGX: route_table_tagx filled sk_expert
        for (int i = lane; i < n; i += 32) sm.sk_expert[i] = __ldcg(p.s.table->expert + i);
      __syncwarp();
      skn = n;
    }
    Job job = n < 0 || j < total_jobs(L, n) ? decode(L, n, j) : Job{J_END, 0, 0, 0, 1};
    if (R1SK && L.skJ > 0 && job.type == J_R1C) {  // a = stream-K job index: experts per stage (sk_expert)
    } else if (job.type == J_R1C || job.type == J_R2C) {  // coarse jobs need only the expert (wtok / all tokens)
      job.e = PROUTE ? sm.slot_expert[job.a] : __ldcg(p.s.table->expert + ((ABL & 16) ? 0 : job.a));
    } else if (job.type == J_R1 || job.type == J_R2) {
      const int e = __ldcg(p.s.table->expert + job.a), ch = __ldcg(p.s.table->chunk + job.a);
      job.e = e;
      job.f = e * MAXM + ch * SPAN;
      job.n = min(SPAN, int(__ldcg(pc + C_ECOUNT + e)) - ch * SPAN);
    }
    bool claimed = false, hready = false;
    unsigned next = 0;
#if DS41H_DECODER_TRACE
    unsigned tq_next = 0;
    job.tq = tq, job.tr = tr, job.ti = 0;
#endif
    // Coarse mode: shared w1/w3 weights are issued only once x is complete (gate row block 0 jobs), so
    // CTAs that pre-claim SH1 jobs do not stream them during the latency-critical wo_b / OWNA phase.
    if (COARSE && DS41H_DECODER_SH1_GATE && job.type == J_SH1 && L.coarse && lane == 0)
      for (int part = 0; part < KPARTS; ++part)
        poll_ge(pc + C_XK + part * GROUPS, unsigned(512 / (KS_GATE * 16)));
    __syncwarp();
    for (int s = 0; s < job.stages; ++s, ++it) {
      const int slot = it % STAGES;
      ring_wait(&sm.empty[slot], ((it / STAGES) & 1) ^ 1);
#if DS41H_DECODER_TRACE
      if (s == 0) job.ti = tnow();
#endif
      const bool bulk = DENSE_BULK && (job.type == J_WOA || job.type == J_WOB || job.type == J_GATE || job.type == J_SH1 || job.type == J_SH2 || job.type == J_OWNA);
      const bool tma = bulk || (R1C_TMA && (job.type == J_R1C || job.type == J_R2C));  // lane 0's expect_tx is its full-barrier arrival
      if (tma && !bulk) {
        if constexpr ((ABL & 128) != 0) {  // ablation: no routed weight / x / scale copies (stale smem), plain arrival
          if (lane == 0) mbar_arrive(&sm.full[slot]);
        } else if (R2F_PF && job.type == J_R2C) {
          if (lane == 0) issue_r2f(p, job, s, sm.ring[slot], sm.bt + XS_OFF + slot * XSL, &sm.full[slot], policy, xpol, hready);
        } else if (LC && job.type == J_R1C) {
          uint8_t* sc;
          if constexpr (R1SKU) {  // (if constexpr: the default build's code stays as it was)
            if (L.skJ > 0) {  // scale groups: 4 stages from each stream-K segment start (any kc, see R1SK)
              const int u = job.a * job.b + s, kc = u % R1F_STAGES, sr = u - max(job.a * job.b, u - kc);  // stage in its segment
              sc = nullptr;
              if ((sr & 3) == 0) {
                // the slot's previous group (sgp - 3) may have ended less than STAGES stages ago (short groups)
                if (sge3 > it - STAGES) ring_wait(&sm.empty[sge3 % STAGES], (sge3 / STAGES) & 1);
                sc = sm.bt + SC_OFF + (sgp++ % 3) * SCL;
              }
              if ((sr & 3) == 3 || kc == R1F_STAGES - 1 || s == job.stages - 1) sge3 = sge2, sge2 = sge1, sge1 = it;
            } else {
              sc = (s & 3) == 0 ? sm.bt + SC_OFF + (sgp++ % 3) * SCL : nullptr;
            }
          } else {
            sc = (s & 3) == 0 ? sm.bt + SC_OFF + (sgp++ % 3) * SCL : nullptr;
          }
          if (R1SK && L.skJ > 0) {  // stage s of stream-K job a: unit a T + s -> (slot, row block, K chunk)
            const int u = job.a * job.b + s, g = u / R1F_STAGES;
            Job js = job;
            js.e = sm.sk_expert[g / R1F_BLOCKS], js.b = g % R1F_BLOCKS, js.c = 0;
            if (lane == 0) issue_r1f(p, js, u % R1F_STAGES, sm.ring[slot], sm.bt + XS_OFF + slot * XSL, sc, &sm.full[slot], policy, xpol);
          } else if (lane == 0) {
            issue_r1f(p, job, s, sm.ring[slot], sm.bt + XS_OFF + slot * XSL, sc, &sm.full[slot], policy, xpol);
          }
        } else if (lane == 0) {
          if (job.type == J_R1C) {
            issue_r1c_tma(p, job, s, sm.ring[slot], &sm.full[slot], policy);
          } else if (R2P > 1) {  // multi-block R2F job: block job.b + s / 5, stage s % 5 of it
            Job jb = job;
            jb.b += s / INTER_STAGES;
            issue_r2c_tma(p, jb, s % INTER_STAGES, sm.ring[slot], &sm.full[slot], policy);
          } else {
            issue_r2c_tma(p, job, s, sm.ring[slot], &sm.full[slot], policy);
          }
        }
      } else {
        issue(p, job, s, Issue{sm.ring[slot], lane, policy, bulk ? &sm.full[slot] : nullptr});
      }
      if (lane == 0 && s == 0) sm.job[slot] = job;
      if (DS41H_DECODER_PROD_ARR && tma) {
        if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], 32;" ::"r"(smem_u32(&sm.full[slot])) : "memory");
      } else {
        cp_async_mbar_arrive(&sm.full[slot]);
      }
      if (lane == 0 && s == 0 && job.type == J_OWNA && job.b == 0 && p.w.pf_phase == 0) prefetch_slice(p.w, job.a);
      if (COARSE && !R1SK && lane == 0 && s == 0 && job.type == J_R1C && job.c == 0) {
        if (p.w.dpf_total > 0) prefetch_ranges(p.w.dpf, p.w.dpf_len, p.w.dpf_total, job.a * R1X_BLOCKS + job.b, n * R1X_BLOCKS, false);
        if (p.w.pf_phase == 1 && p.w.pf_total > 0) prefetch_ranges(p.w.pf, p.w.pf_len, p.w.pf_total, job.a * R1X_BLOCKS + job.b, n * R1X_BLOCKS, true);
      }
      if (lane == 0) {
        if (!tma) mbar_arrive(&sm.full[slot]);
        // Claim the next job only once the consumers have started this one (exchange jobs:
        // finished it), so a CTA never sits on a claimed job behind a blocking wait (BRRR).
        if (!claimed && seq > 0 && job.type != J_END && *ready >= seq) {
          next = atomicAdd(pc + C_CLAIM, 1u);
          claimed = true;
#if DS41H_DECODER_TRACE
          tq_next = tnow();
#endif
        }
      }
    }
    if (job.type == J_END) return;
    // The first job only streams weights; everything after may depend on the previous kernel.
    if (seq == 0) pdl_wait();
    if (lane == 0 && !claimed) {
      while (*ready < seq) __nanosleep(32);
      if (TAILFREE && seq == 0) pc = p.s.counters + (*reinterpret_cast<const volatile unsigned*>(&sm.epoch) & 1u) * COUNTERS;
      next = atomicAdd(pc + C_CLAIM, 1u);
#if DS41H_DECODER_TRACE
      tq_next = tnow();
#endif
    }
    j = int(gridDim.x) + int(__shfl_sync(0xffffffffu, next, 0));
    if (TAILFREE && seq == 0) pc = p.s.counters + (__shfl_sync(0xffffffffu, unsigned(pc == p.s.counters ? 0u : 1u), 0) & 1u) * COUNTERS;
#if DS41H_DECODER_TRACE
    tq = __shfl_sync(0xffffffffu, tq_next, 0);
#endif
  }
}

__global__ void __launch_bounds__(THREADS, PER_SM) layer_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t raw[];
  Smem& sm = *reinterpret_cast<Smem*>(raw);
  for (int i = threadIdx.x; i < CACHED; i += THREADS) sm.seen[i] = 0;
  if (threadIdx.x == 0) {
    sm.ready = -1;
    sm.xres = 0;
    sm.nh = 0;
    sm.epoch = 0;
    mbar_init(&sm.xbar, 1);
    mbar_init(&sm.hbar, 1);
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&sm.full[s], 33);  // 32 cp.async arrivals (noinc) + lane 0
      mbar_init(&sm.empty[s], GW);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if constexpr (DS41H_DECODER_PDL_EARLY != 0) pdl_launch_dependents();
#if DS41H_DECODER_TRACE
  unsigned long long clk0 = 0;
  if (threadIdx.x == 0) {
    sm.t0 = gtime();
    clk0 = clock64();
    sm.nrec = 0;
    sm.rt_in = sm.rt_cg = sm.rt_done = sm.rt_ep = sm.rt_sel = sm.rt_ld = sm.rt_sc = 0;
  }
#endif
  __syncthreads();
  if (threadIdx.x >= GT) {
    producer(p, sm);
  } else {
    const int tid = threadIdx.x;
    Ctx c{p, sm, tid >> 5, tid & 31, tid, 0, 0u};
    if constexpr (WOA_PRE) woa_prefetch(c);
    if constexpr (DS41H_DECODER_WARM_ROUTE != 0 && HIER_ROUTE) {
      if (blockIdx.x == gridDim.x - 1 && coarse_m(p.m)) {  // dry routing on the warm scratch (code -> L2)
        Params q = p;
        q.s = p.ws;
        route_block_nc(q, sm, tid, 0u, 0);
        c.sync();
        if constexpr (DS41H_DECODER_ROUTE_LEAN != 0) route_lean_nc(q, sm, tid, 0u);
        else route_all_nc(q, sm, tid, 0u);
        c.sync();
      }
    }
    pdl_wait();
    c.epoch = ld_volatile(p.s.epoch) + 1;
    if constexpr (TAILFREE || DS41H_DECODER_TAGX != 0) {
      if (tid == 0) {  // before this CTA's first ready (the producer reads it at its first claim)
        *reinterpret_cast<volatile unsigned*>(&sm.epoch) = c.epoch;
        __threadfence_block();
      }
    }
    if constexpr (TAILFREE) {
      // this CTA's slice of the other counter set (the previous launch's: complete; the next launch's: cleared here)
      unsigned* other = p.s.counters + ((c.epoch + 1u) & 1u) * COUNTERS;
      for (int i = int(blockIdx.x) * GT + tid; i < COUNTERS; i += int(gridDim.x) * GT) other[i] = 0u;
    }
    for (int seq = 0;; ++seq) {
#if DS41H_DECODER_TRACE
      const unsigned t_free = c.tnow();
#endif
      c.acquire();
      const Job job = uniform_job(sm.job[c.it % STAGES]);
      if (job.type != J_R1C && job.type != J_R2C) lazy_flush(c);  // LC jobs flush after their first stage
      if (job.type == J_END) break;
#if DS41H_DECODER_TRACE
      if (tid == 0) {
        const Job& pj = sm.job[c.it % STAGES];
        for (int w = 0; w < TRACE_REC; ++w) sm.rec[w] = 0;
        sm.rec[TR_TYPE] = unsigned(job.type) | unsigned(job.c & 0xff) << 8 | unsigned(job.n & 0xffff) << 16;
        sm.rec[TR_AB] = unsigned(job.a & 0xffff) | unsigned(job.b & 0xffff) << 16;
        sm.rec[TR_CLAIM] = pj.tq;
        sm.rec[TR_ISSUE] = pj.ti;
        sm.rec[TR_ROUTE] = pj.tr;
        sm.rec[TR_FREE] = t_free;
        sm.rec[TR_ACQ] = c.tnow();
      }
#endif
      deps(c, job);
      c.tmark(TR_DEP);
      const bool exchange = job.type == J_XA || job.type == J_XF || (FUSED_AR1 && job.type == J_OWNA);
      if (tid == 0 && !exchange) *reinterpret_cast<volatile int*>(&sm.ready) = seq;
      run_job(c, job);
      if (tid == 0 && exchange) *reinterpret_cast<volatile int*>(&sm.ready) = seq;
#if DS41H_DECODER_TRACE
      if (tid == 0) {
        sm.rec[TR_END] = c.tnow();
        const int i = sm.nrec++;
        if (i < TRACE_CAP) {
          unsigned* dst = p.s.trace + (long(c.epoch % TRACE_LAUNCHES) * gridDim.x + blockIdx.x) * TRACE_CTA + TRACE_HDR + i * TRACE_REC;
          for (int w = 0; w < TRACE_REC; ++w) dst[w] = sm.rec[w];
        }
      }
#endif
    }
#if DS41H_DECODER_TRACE
    if (tid == 0) {
      unsigned* hdr = p.s.trace + (long(c.epoch % TRACE_LAUNCHES) * gridDim.x + blockIdx.x) * TRACE_CTA;
      unsigned smid;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      const unsigned long long clk = clock64() - clk0;
      hdr[0] = unsigned(min(sm.nrec, TRACE_CAP));
      hdr[1] = smid;
      hdr[2] = unsigned(sm.t0 & 0xffffffffull);
      hdr[3] = unsigned(sm.t0 >> 32);
      hdr[4] = c.tnow();  // consumer exit
      hdr[5] = blockIdx.x;
      hdr[6] = c.epoch;
      hdr[7] = unsigned(sm.nrec);  // records attempted (> TRACE_CAP: truncated)
      hdr[8] = unsigned(p.m);
      hdr[9] = __ldcg(c.cs() + C_NENT);
      hdr[10] = unsigned(clk & 0xffffffffull);  // SM cycles entry -> consumer exit
      hdr[11] = unsigned(clk >> 32);
      hdr[13] = sm.rt_in;  // PROUTE route_local timeline (0 = not routed by this CTA's producer)
      hdr[14] = sm.rt_cg;
      hdr[15] = sm.rt_done;
      hdr[16] = sm.rt_ep;
      hdr[17] = sm.rt_sel;
      hdr[18] = sm.rt_ld;
      hdr[19] = sm.rt_sc;
      if (blockIdx.x == 0) {  // %globaltimer update granularity
        unsigned long long a = gtime(), b = a, best = ~0ull;
        for (int k = 0; k < 4; ++k) {
          while ((b = gtime()) == a) {
          }
          best = min(best, b - a);
          a = b;
        }
        hdr[12] = unsigned(best);
      }
    }
#endif
  }
  if constexpr (TAILFREE) return;  // counters: other set cleared at the start, epoch advanced by XF chunk 0
  __syncthreads();
  // The last CTA resets the counters and advances the epoch: replay-safe under CUDA graphs.
  if (threadIdx.x == 0) {
    __threadfence();
    sm.last = atomicAdd(p.s.counters + C_FIN, 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (!sm.last) return;
  __threadfence();
  if (DS41H_DECODER_RESET_LEAN != 0 && COARSE && coarse_m(p.m)) {  // only the counters the coarse path uses (see RESET_LEAN)
    constexpr int A = C_R1A + MAXE * R1C_BLOCKS, B = MAXE, C = COUNTERS - C_NENT;
    for (int i = threadIdx.x; i < A + B + C; i += THREADS) p.s.counters[i < A ? i : i < A + B ? C_R1D + i - A : C_NENT + i - A - B] = 0;
  } else {
    for (int i = threadIdx.x; i < COUNTERS; i += THREADS) p.s.counters[i] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
#if DS41H_DECODER_TRACE
    const unsigned ep = ld_volatile(p.s.epoch) + 1;  // this launch's epoch (as the consumers' c.epoch)
#endif
    atomicAdd(p.s.epoch, 1u);
    __threadfence();
#if DS41H_DECODER_TRACE
    // header words 17 / 18 of the last CTA (PROUTE-only otherwise): 0xFFFF flag / counter reset + epoch done (ns since CTA entry)
    unsigned* hdr = p.s.trace + (long(ep % TRACE_LAUNCHES) * gridDim.x + blockIdx.x) * TRACE_CTA;
    hdr[17] = 0xFFFFu;
    hdr[18] = unsigned(gtime() - sm.t0);
#endif
  }
}

// ------------------------------------------------------------------------------ routed micro-benchmark
// Only the routed-expert GEMM jobs (R1 = w13, R2 = w2; no routing, SwiGLU or dependency counters)
// of `slots` (expert, route list) entries, through the same producer issue(), smem ring and
// r1_gemm / r2_gemm as the layer kernel, with the HMMA (WG = false) or RS-wgmma (WG = true) inner
// loops. Inputs p.s.xh / xq (per token = route / TOPK), p.s.hh / hq (per route); outputs red.add
// into p.s.hacc [route, 2 INTER] and p.s.yracc [token, HID] (topk_w weighted) like the layer kernel.
struct Bench {
  Params p;
  const int* slot_expert;  // [slots] expert index into w13 / s13 / w2 / s2
  const int* slot_routes;  // [slots, SPAN] route ids
  const int* slot_count;   // [slots] routes of the slot (1..SPAN)
  int slots;
  int r1;                  // R1 jobs per slot (R1_JOBS or 0)
  int r2;                  // R2 jobs per slot (R2_BLOCKS or 0)
  unsigned* claim;         // job claim counter, zeroed before every launch
};

__device__ __forceinline__ Job bench_job(const Bench& b, int j) {
  const int per = b.r1 + b.r2;
  if (j >= b.slots * per) return {J_END, 0, 0, 0, 1};
  const int slot = j / per, r = j % per;
  Job job = r < b.r1 ? Job{J_R1, slot, r / R1_PARTS, r % R1_PARTS, KS_R1 / 4} : Job{J_R2, slot, r - b.r1, 0, INTER_STAGES};
  job.e = __ldg(b.slot_expert + slot);
  job.f = slot * SPAN;
  job.n = __ldg(b.slot_count + slot);
  return job;
}

__device__ void bench_producer(const Bench& b, Smem& sm) {
  const int lane = threadIdx.x & 31;
  const unsigned long long policy = evict_first_policy();
  int it = 0, j = int(blockIdx.x);
  for (;;) {
    const Job job = bench_job(b, j);
    unsigned next = 0;
    for (int s = 0; s < job.stages; ++s, ++it) {
      const int slot = it % STAGES;
      ring_wait(&sm.empty[slot], ((it / STAGES) & 1) ^ 1);
      issue(b.p, job, s, Issue{sm.ring[slot], lane, policy});
      if (lane == 0 && s == 0) sm.job[slot] = job;
      cp_async_mbar_arrive(&sm.full[slot]);
      if (lane == 0) {
        mbar_arrive(&sm.full[slot]);
        if (s == 0 && job.type != J_END) next = atomicAdd(b.claim, 1u);
      }
    }
    if (job.type == J_END) return;
    j = int(gridDim.x) + int(__shfl_sync(0xffffffffu, next, 0));
  }
}

template <bool WG>
__global__ void __launch_bounds__(THREADS, PER_SM) routed_bench_kernel(const __grid_constant__ Bench b) {
  extern __shared__ __align__(1024) uint8_t raw[];
  Smem& sm = *reinterpret_cast<Smem*>(raw);
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&sm.full[s], 33);
      mbar_init(&sm.empty[s], GW);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x >= GT) {
    bench_producer(b, sm);
    return;
  }
  const int tid = threadIdx.x;
  Ctx c{b.p, sm, tid >> 5, tid & 31, tid, 0, 0u};
  for (;;) {
    c.acquire();
    const Job job = uniform_job(sm.job[c.it % STAGES]);
    if (job.type == J_END) break;
    const Span en = entry_from(c, b.slot_routes + job.f, job.n);
    if (job.type == J_R1) r1_gemm<WG>(c, job, en);
    else r2_gemm<WG>(c, job, en);
  }
}

// ------------------------------------------------------------------------------ host
struct Layer {
  Weights w;
  std::vector<torch::Tensor> keep;
  std::vector<torch::Tensor> pf_keep;
  std::vector<torch::Tensor> dpf_keep;
};

struct Engine {
  Peers peers;
  Scratch s;
  Scratch ws;  // WARM_ROUTE warm scratch
  int rank;
  int nranks;
  int ctas;
  int device;
  int r1kp;  // coarse R1 K parts (0 = auto, 1, 2, LC: 5)
  int lag;   // coarse R2C expert lag (0 = auto)
  std::vector<Layer> layers;
};

std::vector<Engine> engines;

template <class T>
T* mut_ptr(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "expected contiguous CUDA tensor");
  return reinterpret_cast<T*>(t.data_ptr());
}

void check_u8(const torch::Tensor& t, const char* name, std::vector<int64_t> shape) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.element_size() == 1, name, ": expected a contiguous 1-byte CUDA tensor");
  TORCH_CHECK(t.sizes().vec() == shape, name, ": shape ", t.sizes(), " expected ", c10::IntArrayRef(shape));
}

void check_f32(const torch::Tensor& t, const char* name, int64_t numel) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kFloat && t.numel() == numel, name, ": expected contiguous fp32 CUDA tensor with ",
              numel, " elements, got ", t.sizes(), " ", t.scalar_type());
}

PFN_cuTensorMapEncodeTiled_v12000 tmap_encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q{};
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q) == cudaSuccess && ptr,
                "cuTensorMapEncodeTiled not available");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

void tmap_encode(CUtensorMap* map, const void* base, int rank, std::initializer_list<cuuint64_t> dims, std::initializer_list<cuuint64_t> strides,
                 std::initializer_list<cuuint32_t> box, CUtensorMapSwizzle sw, const char* name) {
  cuuint64_t d[5], st[4];
  cuuint32_t bx[5], es[5] = {1, 1, 1, 1, 1};
  int i = 0;
  for (auto v : dims) d[i++] = v;
  i = 0;
  for (auto v : strides) st[i++] = v;
  i = 0;
  for (auto v : box) bx[i++] = v;
  const CUresult r = tmap_encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, const_cast<void*>(base), d, st, bx, es, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled(", name, ") failed: ", int(r));
}

// Raw TP-sliced routed experts (ds41h.moe.get_raw_experts): w13 [E, 1152, 2560], s13 [E, 1152, 160],
// w2 [E, 5120, 288], s2 [E, 5120, 18], 1-byte dtypes, contiguous rows (any expert stride % 16).
void bind_experts(Layer& l, const torch::Tensor& w13, const torch::Tensor& s13, const torch::Tensor& w2, const torch::Tensor& s2) {
  for (const auto* t : {&w13, &s13, &w2, &s2}) {
    TORCH_CHECK(t->is_cuda() && t->element_size() == 1 && t->dim() == 3 && t->size(0) == EXPERTS, "routed expert tensors must be 1-byte CUDA [384, rows, cols]");
    TORCH_CHECK(t->stride(2) == 1 && t->stride(1) == t->size(2) && t->stride(0) % 16 == 0, "routed expert tensors need contiguous rows and a 16B expert stride");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "routed expert tensors must be 16B aligned");
  }
  TORCH_CHECK(w13.size(1) == 2 * INTER && w13.size(2) == W13_ROW, "w13 must be [E, 1152, 2560]");
  TORCH_CHECK(s13.size(1) == 2 * INTER && s13.size(2) == S13_ROW, "s13 must be [E, 1152, 160]");
  TORCH_CHECK(w2.size(1) == HID && w2.size(2) == W2_ROW, "w2 must be [E, 5120, 288]");
  TORCH_CHECK(s2.size(1) == HID && s2.size(2) == S2_ROW, "s2 must be [E, 5120, 18]");
  l.w.w13 = reinterpret_cast<const uint8_t*>(w13.data_ptr());
  l.w.s13 = reinterpret_cast<const uint8_t*>(s13.data_ptr());
  l.w.w2 = reinterpret_cast<const uint8_t*>(w2.data_ptr());
  l.w.s2 = reinterpret_cast<const uint8_t*>(s2.data_ptr());
  l.w.w13_e = w13.stride(0);
  l.w.s13_e = s13.stride(0);
  l.w.w2_e = w2.stride(0);
  l.w.s2_e = s2.stride(0);
  if (R1C_TMA && COARSE) {  // coarse R1 stage boxes (see issue_r1c_tma)
    const cuuint64_t E = w13.size(0);
    tmap_encode(&l.w.tm_w13, w13.data_ptr(), 4, {128, 2 * INTER, W13_ROW / 128, E}, {W13_ROW, 128, cuuint64_t(w13.stride(0))}, {128, 32, 1, 1},
                CU_TENSOR_MAP_SWIZZLE_128B, "w13");
    tmap_encode(&l.w.tm_s13, s13.data_ptr(), 3, {S13_ROW, 2 * INTER, E}, {S13_ROW, cuuint64_t(s13.stride(0))}, {16, 32, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "s13");
    tmap_encode(&l.w.tm_w2, w2.data_ptr(), 4, {64, HID, 4, E}, {W2_ROW, 64, cuuint64_t(w2.stride(0))}, {64, 128, 1, 1}, CU_TENSOR_MAP_SWIZZLE_64B, "w2");
    tmap_encode(&l.w.tm_w2t, static_cast<const uint8_t*>(w2.data_ptr()) + 256, 3, {32, HID, E}, {W2_ROW, cuuint64_t(w2.stride(0))}, {32, 128, 1},
                CU_TENSOR_MAP_SWIZZLE_32B, "w2 tail");
  }
  if (LC) {  // R1F stage boxes and scale groups (see issue_r1f)
    const cuuint64_t E = w13.size(0);
    tmap_encode(&l.w.tm_w13f, w13.data_ptr(), 4, {64, 2 * INTER, W13_ROW / 64, E}, {W13_ROW, 64, cuuint64_t(w13.stride(0))}, {64, 64, 1, 1},
                CU_TENSOR_MAP_SWIZZLE_64B, "w13 f");
    tmap_encode(&l.w.tm_s13f, s13.data_ptr(), 3, {S13_ROW, 2 * INTER, E}, {S13_ROW, cuuint64_t(s13.stride(0))}, {16, 64, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "s13 f");
  }
}

// e4m3 scale of the normalized FFN input x = bf16(w * xc * rsqrt(mean(xc^2) + eps)): |x| <=
// |w| sqrt(HID) (1 + 2^-8), so the power of two >= 1.01 max|w| sqrt(HID) / 448 never saturates.
float x_scale(const torch::Tensor& norm) {
  const float bound = norm.abs().max().item<float>() * sqrtf(float(HID)) * 1.01f;
  return exp2f(ceilf(log2f(fmaxf(bound, 1e-30f) / 448.0f)));
}

}  // namespace

int64_t flag_bytes() { return FLAG_BYTES; }

int64_t inbox_bytes() { return 2L * 2 * 2 * PUB_SLOT * sizeof(__nv_bfloat16) + (LLP ? 2L * 2 * RANKS * LL_SRC * 8 : 0L); }

int64_t create(std::vector<int64_t> buffers, int64_t rank, int64_t nranks) {
  TORCH_CHECK(buffers.size() == RANKS, "need one IPC buffer pointer per rank (0 for absent peers)");
  TORCH_CHECK(nranks == 1 || nranks == RANKS, "nranks must be 1 (solo) or ", RANKS);
  TORCH_CHECK(rank >= 0 && rank < RANKS);
  Engine e{};
  for (int r = 0; r < RANKS; ++r) {
    const bool live = nranks == RANKS || r == rank;
    TORCH_CHECK(!live || buffers[r] != 0, "missing IPC buffer for rank ", r);
    e.peers.flags[r] = live ? reinterpret_cast<unsigned*>(buffers[r]) : nullptr;
    e.peers.pub[r] = live ? reinterpret_cast<__nv_bfloat16*>(buffers[r] + FLAG_BYTES) : nullptr;
  }
  e.rank = int(rank);
  e.nranks = int(nranks);
  TORCH_CHECK(cudaGetDevice(&e.device) == cudaSuccess);
  e.s.epoch = zeroed<unsigned>(1);
  e.s.counters = zeroed<unsigned>(COUNTERS * (TAILFREE ? 2 : 1));
  e.s.table = zeroed<Table>(1);
  e.s.zacc = zeroed<float>(long(MAXM) * OA);
  e.s.zh = zeroed<__half>(long(MAXM) * OA);
  e.s.ze = zeroed<uint8_t>(long(MAXM) * (OA / 32));
  e.s.yacc = zeroed<float>(long(MAXM) * HID);
  e.s.yattn = zeroed<__nv_bfloat16>(long(MAXM) * HID);
  e.s.mixp = zeroed<float>(long(SLICES) * MAXM * MIXW);
  e.s.xc = zeroed<__nv_bfloat16>(long(MAXM) * HID);
  e.s.xsq = zeroed<float>(long(SLICES) * MAXM);
  e.s.xbf = zeroed<__nv_bfloat16>(long(MAXM) * HID);
  e.s.xh = zeroed<__half>(long(MAXM) * HID);
  e.s.gacc = zeroed<float>(2L * 2 * MAXM * EXPERTS);  // room for the u64 (GDET) layout
  e.s.gpart = DS41H_DECODER_GSUM2 ? zeroed<float>(long(KP_GATE + 1) * COARSE_M * EXPERTS) : nullptr;
  e.s.xtag = DS41H_DECODER_TAGX ? zeroed<unsigned long long>(MAXE) : nullptr;
  e.s.topk_w = zeroed<float>(ROUTES);
  e.s.topk_id = zeroed<int>(ROUTES);
  e.s.cand_key = zeroed<float>(long(MAXM) * GATE_BLOCKS * TOPK);
  e.s.cand_score = zeroed<float>(long(MAXM) * GATE_BLOCKS * TOPK);
  e.s.cand_exp = zeroed<int>(long(MAXM) * GATE_BLOCKS * TOPK);
  e.s.sacc = zeroed<float>(long(MAXM) * 2 * INTER);
  e.s.hs = zeroed<__half>(long(MAXM) * INTER);
  e.s.shacc = zeroed<float>(long(MAXM) * HID);
  e.s.hacc = zeroed<float>(long(ROUTES) * 2 * INTER);
  e.s.hh = zeroed<__half>(long(ROUTES) * INTER);
  e.s.yracc = zeroed<float>(long(MAXM) * HID);
  e.s.xq = zeroed<uint8_t>(long(MAXM) * HID);
  e.s.hq = zeroed<uint8_t>(long(ROUTES) * INTER);
  e.s.zq = zeroed<uint8_t>(long(MAXM) * OA);
  e.s.hsq = zeroed<uint8_t>(long(MAXM) * INTER);
  e.s.hqi = zeroed<uint8_t>(long(MAXE) * HQI);
  e.s.hcacc = zeroed<float>(long(MAXE) * TILE * 2 * INTER);
  e.s.wtok = zeroed<float>(long(MAXE) * TILE);
  e.s.xqi = zeroed<uint8_t>(XT_BYTES);
  e.s.xhi = zeroed<uint8_t>(XHI);
  e.s.hhi = zeroed<uint8_t>(long(MAXE) * HHI);
  e.ws = e.s;
  if (DS41H_DECODER_WARM_ROUTE) {  // private routing buffers of the dry routing run (see WARM_ROUTE)
    e.ws.counters = zeroed<unsigned>(COUNTERS);
    e.ws.table = zeroed<Table>(1);
    e.ws.gacc = zeroed<float>(2L * 2 * MAXM * EXPERTS);
    e.ws.topk_w = zeroed<float>(ROUTES);
    e.ws.topk_id = zeroed<int>(ROUTES);
    e.ws.cand_key = zeroed<float>(long(MAXM) * GATE_BLOCKS * TOPK);
    e.ws.cand_score = zeroed<float>(long(MAXM) * GATE_BLOCKS * TOPK);
    e.ws.cand_exp = zeroed<int>(long(MAXM) * GATE_BLOCKS * TOPK);
    e.ws.wtok = zeroed<float>(long(MAXE) * TILE);
  }
  e.r1kp = 0;
  e.lag = 0;
  int sms = 0, per = 0;
  TORCH_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, e.device) == cudaSuccess);
  // Direct call (not host.cuh opt_in_smem): its function-static guard is a process-wide unique
  // symbol, so a second decoder build loaded in the same process (e.g. TRACE=1 next to the default
  // build) would skip the opt-in for its own kernel.
  TORCH_CHECK(cudaFuncSetAttribute(layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))) == cudaSuccess);
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, layer_kernel, THREADS, sizeof(Smem)) == cudaSuccess);
  TORCH_CHECK(per >= PER_SM, "decoder layer kernel fits ", per, " CTAs per SM, needs ", PER_SM, " (smem ", sizeof(Smem), " B)");
  e.ctas = sms * PER_SM;
  e.s.trace = TRACE ? zeroed<unsigned>(long(TRACE_LAUNCHES) * e.ctas * TRACE_CTA) : nullptr;
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  engines.push_back(e);
  return long(engines.size()) - 1;
}

int64_t add_layer(int64_t handle, torch::Tensor woa, torch::Tensor woa_s, torch::Tensor wob, torch::Tensor wob_s, torch::Tensor gate, torch::Tensor gate_b,
                  torch::Tensor sh13, torch::Tensor sh13_s, torch::Tensor sh2, torch::Tensor sh2_s, torch::Tensor w13, torch::Tensor s13, torch::Tensor w2,
                  torch::Tensor s2, torch::Tensor fn, torch::Tensor hc_scale, torch::Tensor hc_base, torch::Tensor norm) {
  auto& eng = engines.at(handle);
  check_u8(woa, "woa", {long(OA) * OAK});
  check_u8(woa_s, "woa_s", {OA / 32, OAK / 32});
  check_u8(wob, "wob", {long(HID) * OA});
  check_u8(wob_s, "wob_s", {HID / 32, OA / 32});
  check_u8(gate, "gate", {long(EXPERTS) * HID * 2});
  check_f32(gate_b, "gate_b", EXPERTS);
  check_u8(sh13, "sh13", {long(2 * INTER) * HID});
  check_u8(sh13_s, "sh13_s", {2 * INTER / 32, SLICES});
  check_u8(sh2, "sh2", {long(HID) * INTER_STAGES * 128});
  check_u8(sh2_s, "sh2_s", {HID / 32, SH2_LD});
  check_f32(fn, "fn", long(MIX) * HCD);
  check_f32(hc_scale, "hc_scale", 3);
  check_f32(hc_base, "hc_base", MIX);
  TORCH_CHECK(norm.is_cuda() && norm.is_contiguous() && norm.scalar_type() == at::kBFloat16 && norm.numel() == HID, "norm must be bf16 [5120]");
  for (const auto* t : {&woa, &woa_s, &wob, &wob_s, &gate, &gate_b, &sh13, &sh13_s, &sh2, &sh2_s, &w13, &s13, &w2, &s2, &fn, &hc_scale, &hc_base, &norm})
    TORCH_CHECK(t->get_device() == eng.device, "layer tensors must live on the engine's device");
  Layer l{};
  l.w.woa = mut_ptr<uint8_t>(woa);
  l.w.woa_s = mut_ptr<uint8_t>(woa_s);
  l.w.wob = mut_ptr<uint8_t>(wob);
  l.w.wob_s = mut_ptr<uint8_t>(wob_s);
  l.w.gate = mut_ptr<uint8_t>(gate);
  l.w.gate_b = mut_ptr<float>(gate_b);
  l.w.sh13 = mut_ptr<uint8_t>(sh13);
  l.w.sh13_s = mut_ptr<uint8_t>(sh13_s);
  l.w.sh2 = mut_ptr<uint8_t>(sh2);
  l.w.sh2_s = mut_ptr<uint8_t>(sh2_s);
  bind_experts(l, w13, s13, w2, s2);
  l.w.fn = mut_ptr<float>(fn);
  l.w.hc_scale = mut_ptr<float>(hc_scale);
  l.w.hc_base = mut_ptr<float>(hc_base);
  l.w.norm = mut_ptr<__nv_bfloat16>(norm);
  l.w.xs = x_scale(norm);
  l.keep = {woa, woa_s, wob, wob_s, gate, gate_b, sh13, sh13_s, sh2, sh2_s, w13, s13, w2, s2, fn, hc_scale, hc_base, norm};
  eng.layers.push_back(std::move(l));
  return long(eng.layers.size()) - 1;
}

// L2 prefetch ranges of a registered layer; empty = none. dense = false: prefetched during the OWNA phase
// (evict_last; e.g. the next layer's prologue weights); dense = true: during the R1C jobs (coarse mode,
// evict_normal; the next layer's wo_a / wo_b / gate / fn).
void set_prefetch(int64_t handle, int64_t layer, std::vector<torch::Tensor> tensors, bool dense) {
  auto& eng = engines.at(handle);
  auto& l = eng.layers.at(layer);
  TORCH_CHECK(int(tensors.size()) <= PF_MAX, "at most ", PF_MAX, " prefetch tensors");
  const uint8_t** ptr = dense ? l.w.dpf : l.w.pf;
  long* len = dense ? l.w.dpf_len : l.w.pf_len;
  long& total = dense ? l.w.dpf_total : l.w.pf_total;
  auto& keep = dense ? l.dpf_keep : l.pf_keep;
  total = 0;
  for (int r = 0; r < PF_MAX; ++r) {
    ptr[r] = nullptr;
    len[r] = 0;
  }
  keep.clear();
  for (size_t r = 0; r < tensors.size(); ++r) {
    const auto& t = tensors[r];
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.get_device() == eng.device, "prefetch tensors: contiguous, on the engine's device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "prefetch tensors must be 16-byte aligned");
    ptr[r] = reinterpret_cast<const uint8_t*>(t.data_ptr());
    len[r] = long(t.numel() * t.element_size()) & ~15L;
    total += len[r];
    keep.push_back(t);
  }
}

// Prefetch phase of a layer's pf ranges (set_prefetch dense=false): 0 = OWNA jobs, 1 = coarse R1 jobs.
void set_prefetch_phase(int64_t handle, int64_t layer, int64_t phase) {
  TORCH_CHECK(phase == 0 || phase == 1, "phase must be 0 (OWNA) or 1 (routed R1 jobs)");
  engines.at(handle).layers.at(layer).w.pf_phase = int(phase);
}

// Re-point a registered layer at (moved / re-created) raw expert tensors.
void update_experts(int64_t handle, int64_t layer, torch::Tensor w13, torch::Tensor s13, torch::Tensor w2, torch::Tensor s2) {
  auto& eng = engines.at(handle);
  auto& l = eng.layers.at(layer);
  for (const auto* t : {&w13, &s13, &w2, &s2}) TORCH_CHECK(t->get_device() == eng.device, "expert tensors must live on the engine's device");
  bind_experts(l, w13, s13, w2, s2);
  l.keep[10] = w13;
  l.keep[11] = s13;
  l.keep[12] = w2;
  l.keep[13] = s2;
}

void run(int64_t handle, int64_t layer, torch::Tensor o, torch::Tensor positions, torch::Tensor cos_sin, torch::Tensor resid0, torch::Tensor attn_post,
         torch::Tensor attn_comb, torch::Tensor attn_pre, torch::Tensor y_ffn, torch::Tensor resid1, torch::Tensor ffn_post, torch::Tensor ffn_comb,
         torch::Tensor ffn_pre, std::optional<torch::Tensor> p1_resid, std::optional<torch::Tensor> p1_xcb, std::optional<torch::Tensor> p1_xsq) {
  const auto& e = engines.at(handle);
  const auto& l = e.layers.at(layer);
  TORCH_CHECK(resid0.dim() == 3 && resid0.size(1) == HC && resid0.size(2) == HID && resid0.scalar_type() == at::kBFloat16 && resid0.is_contiguous(),
              "residual must be contiguous bf16 [m, 4, 5120]");
  const int m = int(resid0.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM, "decoder kernel takes 1..", MAXM, " tokens, got ", m);
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == at::kBFloat16 && o.dim() == 3 && o.size(0) >= m && o.size(1) >= 16 && o.size(2) == HDIM && o.stride(2) == 1 &&
                  o.stride(1) == HDIM && o.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0,
              "attention output must be bf16 [m, >=16, 512] with head stride 512");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == at::kLong && positions.is_contiguous() && positions.numel() >= m, "positions must be int64");
  TORCH_CHECK(cos_sin.is_cuda() && cos_sin.scalar_type() == at::kFloat && cos_sin.is_contiguous() && cos_sin.dim() == 2 && cos_sin.size(1) == ROPE,
              "cos_sin must be fp32 [max_pos, 64]");
  check_f32(attn_post, "post_mix", long(m) * HC);
  check_f32(attn_comb, "res_mix", long(m) * HC * HC);
  check_f32(attn_pre, "attn_pre", long(m) * HC);
  TORCH_CHECK(y_ffn.is_contiguous() && y_ffn.scalar_type() == at::kBFloat16 && y_ffn.numel() == long(m) * HID, "y_ffn must be bf16 [m, 5120]");
  TORCH_CHECK(resid1.is_contiguous() && resid1.scalar_type() == at::kBFloat16 && resid1.numel() == long(m) * HC * HID, "resid1 must be bf16 [m, 4, 5120]");
  check_f32(ffn_post, "ffn_post", long(m) * HC);
  check_f32(ffn_comb, "ffn_comb", long(m) * HC * HC);
  check_f32(ffn_pre, "ffn_pre", long(m) * HC);
  for (const auto* t : {&o, &positions, &cos_sin, &resid0, &attn_post, &attn_comb, &attn_pre, &y_ffn, &resid1, &ffn_post, &ffn_comb, &ffn_pre})
    TORCH_CHECK(t->get_device() == e.device, "decoder inputs must live on the engine's device");
  const c10::cuda::CUDAGuard guard(resid0.device());
  Params p{};
  p.peers = e.peers;
  p.s = e.s;
  p.ws = e.ws;
  p.w = l.w;
  p.rank = e.rank;
  p.nranks = e.nranks;
  p.m = m;
  p.o = reinterpret_cast<const __nv_bfloat16*>(o.data_ptr());
  p.o_stride = o.stride(0);
  p.positions = reinterpret_cast<const long*>(positions.data_ptr());
  p.max_pos = cos_sin.size(0);
  p.cos_sin = reinterpret_cast<const float*>(cos_sin.data_ptr());
  p.resid0 = reinterpret_cast<const __nv_bfloat16*>(resid0.data_ptr());
  p.attn_post = reinterpret_cast<const float*>(attn_post.data_ptr());
  p.attn_comb = reinterpret_cast<const float*>(attn_comb.data_ptr());
  p.attn_pre = reinterpret_cast<const float*>(attn_pre.data_ptr());
  p.y_ffn = mut_ptr<__nv_bfloat16>(y_ffn);
  p.resid1 = mut_ptr<__nv_bfloat16>(resid1);
  p.ffn_post = mut_ptr<float>(ffn_post);
  p.ffn_comb = mut_ptr<float>(ffn_comb);
  p.ffn_pre = mut_ptr<float>(ffn_pre);
  TORCH_CHECK(p1_resid.has_value() == p1_xcb.has_value() && p1_resid.has_value() == p1_xsq.has_value(), "p1_resid / p1_xcb / p1_xsq: all or none");
  if (p1_resid.has_value()) {  // XF_P1A outputs (see xf_p1a)
    TORCH_CHECK(m <= COARSE_M, "p1a (XF_P1A) supports m <= ", COARSE_M, " tokens, got ", m);
    TORCH_CHECK(p1_resid->is_cuda() && p1_resid->get_device() == e.device && p1_resid->scalar_type() == at::kBFloat16 && p1_resid->is_contiguous() &&
                    p1_resid->numel() >= long(m) * HC * HID, "p1_resid must be contiguous bf16 [>= m, 4, 5120]");
    TORCH_CHECK(p1_xcb->is_cuda() && p1_xcb->get_device() == e.device && p1_xcb->scalar_type() == at::kBFloat16 && p1_xcb->is_contiguous() &&
                    p1_xcb->numel() >= long(m) * HID, "p1_xcb must be contiguous bf16 [>= m, 5120]");
    TORCH_CHECK(p1_xsq->is_cuda() && p1_xsq->get_device() == e.device && p1_xsq->scalar_type() == at::kFloat && p1_xsq->is_contiguous() &&
                    p1_xsq->numel() >= long(m) * SLICES, "p1_xsq must be contiguous fp32 [>= m, 160]");
    p.p1_resid = mut_ptr<__nv_bfloat16>(*p1_resid);
    p.p1_xcb = mut_ptr<__nv_bfloat16>(*p1_xcb);
    p.p1_xsq = mut_ptr<float>(*p1_xsq);
  }
  // R1 K parts: r1kp = 0 (default) = auto: LC splits K for small m so the few experts' R1F jobs still fill
  // the grid (m = 1: ~6 experts x 9 row blocks = 54 full-K jobs -> 5 K parts of 8 stages each).
  p.r1kp = e.r1kp > 0 ? e.r1kp : LC && m <= 1 ? 5 : LC && m <= 3 ? 2 : (LC && R1KP_DYN && m <= COARSE_M) ? 0 : 1;
  // An expert's R2C jobs follow its R1C jobs by `lag` experts: about 1.5 grids of R1C jobs, so its
  // R1C jobs have finished when the R2C jobs are claimed.
  p.lag = e.lag > 0 ? e.lag : std::max(1, cdiv(3 * e.ctas / 2, R1X_BLOCKS * std::max(p.r1kp, 1)));
  launch_pdl("ds41h decoder layer kernel", layer_kernel, dim3(e.ctas), dim3(THREADS), sizeof(Smem), p);
}

// Routing of the last launch (tests): ids int32 [MAXM, 6], weights fp32 [MAXM, 6] (device copies).
std::vector<torch::Tensor> debug_topk(int64_t handle) {
  const auto& e = engines.at(handle);
  const auto opts = torch::TensorOptions().device(torch::kCUDA, e.device);
  auto ids = torch::from_blob(e.s.topk_id, {MAXM, TOPK}, opts.dtype(torch::kInt32)).clone();
  auto w = torch::from_blob(e.s.topk_w, {MAXM, TOPK}, opts.dtype(torch::kFloat32)).clone();
  return {ids, w};
}

int64_t layer_ctas(int64_t handle) { return engines.at(handle).ctas; }

// Coarse routed path knobs (m <= COARSE_M): R1 K parts (0 = auto by m, 1 = full K, 2; LC also 5) and the R2
// expert lag (0 = auto).
void set_coarse(int64_t handle, int64_t r1kp, int64_t lag) {
  TORCH_CHECK(r1kp == 0 || r1kp == 1 || r1kp == 2 || (LC && r1kp == 5), "r1kp must be 0 (auto), 1, 2 (or 5 on the lossless path)");
  TORCH_CHECK(lag >= 0, "lag must be >= 0");
  auto& e = engines.at(handle);
  e.r1kp = int(r1kp);
  e.lag = int(lag);
}

// Trace mode: copy of the trace ring, int32 [TRACE_LAUNCHES, ctas, TRACE_CTA] (decoder_trace.py);
// clear = zero it afterwards. Empty tensor in non-trace builds.
torch::Tensor trace_dump(int64_t handle, bool clear) {
  const auto& e = engines.at(handle);
  const auto opts = torch::TensorOptions().device(torch::kCUDA, e.device).dtype(torch::kInt32);
  if (!TRACE) return torch::empty({0}, opts);
  const c10::cuda::CUDAGuard guard(e.device);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  auto out = torch::from_blob(e.s.trace, {TRACE_LAUNCHES, e.ctas, TRACE_CTA}, opts).clone();
  if (clear) TORCH_CHECK(cudaMemset(e.s.trace, 0, sizeof(unsigned) * TRACE_LAUNCHES * e.ctas * TRACE_CTA) == cudaSuccess);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  return out;
}

// e4m3 scale of the FFN input x used by the wgmma paths for a registered layer (tests).
double layer_xs(int64_t handle, int64_t layer) { return engines.at(handle).layers.at(layer).w.xs; }

// Routed-expert GEMM micro-benchmark / unit test (see routed_bench_kernel). wg: 0 = HMMA loops,
// 1 = RS-wgmma loops. mode: 1 = R1 only, 2 = R2 only, 3 = both. Experts: raw layout with any
// expert count E. xh f16 [T, HID] (P8), xq e4m3 [T, HID] (wg order, x / xs), hh f16 [R, INTER]
// (P8), hq e4m3 [R, INTER] (wg order, h * 4), topk_w fp32 [R], hacc fp32 [R, 2 INTER], yracc fp32
// [T, HID] (accumulated into), claim int32 [1] scratch. Launches `reps` times on the current stream.
void bench_routed(int64_t wg, torch::Tensor w13, torch::Tensor s13, torch::Tensor w2, torch::Tensor s2, torch::Tensor slot_expert, torch::Tensor slot_routes,
                  torch::Tensor slot_count, torch::Tensor xh, torch::Tensor xq, torch::Tensor hh, torch::Tensor hq, torch::Tensor topk_w, torch::Tensor hacc,
                  torch::Tensor yracc, torch::Tensor claim, double xs, int64_t mode, int64_t reps) {
  for (const auto* t : {&w13, &s13, &w2, &s2}) {
    TORCH_CHECK(t->is_cuda() && t->element_size() == 1 && t->dim() == 3 && t->stride(2) == 1 && t->stride(1) == t->size(2) && t->stride(0) % 16 == 0,
                "expert tensors: 1-byte CUDA [E, rows, cols] with contiguous rows and a 16B expert stride");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0 && t->size(0) == w13.size(0));
  }
  TORCH_CHECK(w13.size(1) == 2 * INTER && w13.size(2) == W13_ROW && s13.size(1) == 2 * INTER && s13.size(2) == S13_ROW);
  TORCH_CHECK(w2.size(1) == HID && w2.size(2) == W2_ROW && s2.size(1) == HID && s2.size(2) == S2_ROW);
  const int slots = int(slot_expert.numel());
  TORCH_CHECK(slot_expert.scalar_type() == at::kInt && slot_routes.scalar_type() == at::kInt && slot_count.scalar_type() == at::kInt);
  TORCH_CHECK(slot_routes.numel() == long(slots) * SPAN && slot_count.numel() == slots, "slot_routes must be int32 [slots, ", SPAN, "]");
  const long T = xh.size(0), R = hh.size(0);
  TORCH_CHECK(xh.scalar_type() == at::kHalf && xh.dim() == 2 && xh.size(1) == HID && xq.numel() == T * HID && xq.element_size() == 1);
  TORCH_CHECK(hh.scalar_type() == at::kHalf && hh.dim() == 2 && hh.size(1) == INTER && hq.numel() == R * INTER && hq.element_size() == 1);
  TORCH_CHECK(topk_w.scalar_type() == at::kFloat && topk_w.numel() == R && hacc.scalar_type() == at::kFloat && hacc.numel() == R * 2 * INTER);
  TORCH_CHECK(yracc.scalar_type() == at::kFloat && yracc.numel() == T * HID && claim.numel() >= 1 && claim.element_size() == 4);
  TORCH_CHECK(mode >= 1 && mode <= 3 && reps >= 1);
  const c10::cuda::CUDAGuard guard(w13.device());
  Bench b{};
  b.p.m = int(T);
  b.p.nranks = 1;
  b.p.w.w13 = data_as<const uint8_t>(w13);
  b.p.w.s13 = data_as<const uint8_t>(s13);
  b.p.w.w2 = data_as<const uint8_t>(w2);
  b.p.w.s2 = data_as<const uint8_t>(s2);
  b.p.w.w13_e = w13.stride(0);
  b.p.w.s13_e = s13.stride(0);
  b.p.w.w2_e = w2.stride(0);
  b.p.w.s2_e = s2.stride(0);
  b.p.w.xs = float(xs);
  b.p.s.xh = mptr<__half>(xh);
  b.p.s.xq = mptr<uint8_t>(xq);
  b.p.s.hh = mptr<__half>(hh);
  b.p.s.hq = mptr<uint8_t>(hq);
  b.p.s.topk_w = mptr<float>(topk_w);
  b.p.s.hacc = mptr<float>(hacc);
  b.p.s.yracc = mptr<float>(yracc);
  b.slot_expert = cptr<int>(slot_expert);
  b.slot_routes = cptr<int>(slot_routes);
  b.slot_count = cptr<int>(slot_count);
  b.slots = slots;
  b.r1 = mode & 1 ? R1_JOBS : 0;
  b.r2 = mode & 2 ? R2_BLOCKS : 0;
  b.claim = reinterpret_cast<unsigned*>(mptr<int>(claim));
  int dev = 0, sms = 0;
  TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess);
  auto kernel = wg ? routed_bench_kernel<true> : routed_bench_kernel<false>;
  TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))) == cudaSuccess);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  for (int64_t r = 0; r < reps; ++r) {
    TORCH_CHECK(cudaMemsetAsync(b.claim, 0, sizeof(unsigned), stream) == cudaSuccess);
    kernel<<<sms * PER_SM, THREADS, sizeof(Smem), stream>>>(b);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "routed bench launch failed: ", cudaGetErrorString(err));
  }
}

int64_t smem_bytes() { return sizeof(Smem); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("RANKS") = RANKS;
  m.attr("MAXM") = MAXM;
  m.attr("HID") = HID;
  m.attr("HC") = HC;
  m.attr("EXPERTS") = EXPERTS;
  m.attr("INTER") = INTER;
  m.attr("STAGES") = STAGES;
  m.attr("PER_SM") = PER_SM;
  m.attr("TWO_SHOT_MIN") = TWO_SHOT_MIN;
  m.attr("ROUTED_MMA") = DS41H_DECODER_ROUTED_MMA;
  m.attr("DENSE_MMA") = DS41H_DECODER_DENSE_MMA;
  m.attr("SPAN") = SPAN;
  m.attr("TOPK") = TOPK;
  m.attr("TRACE") = DS41H_DECODER_TRACE;
  m.attr("TRACE_LAUNCHES") = TRACE_LAUNCHES;
  m.attr("TRACE_CAP") = TRACE_CAP;
  m.attr("TRACE_REC") = TRACE_REC;
  m.attr("TRACE_HDR") = TRACE_HDR;
  m.attr("CHUNKS") = CHUNKS;
  m.attr("COARSE") = DS41H_DECODER_COARSE;
  m.attr("COARSE_M") = COARSE_M;
  m.attr("LOSSLESS") = DS41H_DECODER_LOSSLESS;
  m.def("flag_bytes", &flag_bytes);
  m.def("inbox_bytes", &inbox_bytes);
  m.def("alloc_ipc", &alloc_ipc);
  m.def("open_ipc", &open_ipc);
  m.def("create", &create, py::arg("buffers"), py::arg("rank"), py::arg("nranks"));
  m.def("add_layer", &add_layer, py::arg("handle"), py::arg("woa"), py::arg("woa_s"), py::arg("wob"), py::arg("wob_s"), py::arg("gate"), py::arg("gate_b"),
        py::arg("sh13"), py::arg("sh13_s"), py::arg("sh2"), py::arg("sh2_s"), py::arg("w13"), py::arg("s13"), py::arg("w2"), py::arg("s2"), py::arg("fn"),
        py::arg("hc_scale"), py::arg("hc_base"), py::arg("norm"));
  m.def("update_experts", &update_experts);
  m.def("set_prefetch", &set_prefetch);
  m.def("set_prefetch_phase", &set_prefetch_phase);
  m.def("run", &run);
  m.def("debug_topk", &debug_topk);
  m.def("layer_ctas", &layer_ctas);
  m.def("set_coarse", &set_coarse, py::arg("handle"), py::arg("r1kp"), py::arg("lag") = 0);
  m.def("trace_dump", &trace_dump, py::arg("handle"), py::arg("clear") = false);
  m.def("layer_xs", &layer_xs);
  m.def("bench_routed", &bench_routed, py::arg("wg"), py::arg("w13"), py::arg("s13"), py::arg("w2"), py::arg("s2"), py::arg("slot_expert"),
        py::arg("slot_routes"), py::arg("slot_count"), py::arg("xh"), py::arg("xq"), py::arg("hh"), py::arg("hq"), py::arg("topk_w"), py::arg("hacc"),
        py::arg("yracc"), py::arg("claim"), py::arg("xs"), py::arg("mode"), py::arg("reps") = 1);
  m.def("smem_bytes", &smem_bytes);
}
