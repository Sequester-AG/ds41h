# FPRv1: First Principles Rewrite v1

FPRv1 is a second engine for the same model, checkpoint and machine. It was written from what the hardware has to do
per step rather than from vLLM's model runner, and lives in [`fpr/`](../fpr). It is served by `scripts/serve_fpr.sh`
through vLLM 0.32.0's unchanged API layer: tokenizer, reasoning and tool-call parsers, the DeepSeek API parity plugin,
prefix caching, DSpark speculative decoding (k=3, probabilistic drafts, block rejection), image input and the
Responses API with stored responses. It reuses ds41h's kernels
for the routed MoE in prefill and for sparse attention, and ds41h's fast tokenizer.

Numerics follow the same policy as ds41h: the checkpoint as shipped, 16-bit activations everywhere. The released
configuration is checked against ds41h with the method of [EXACTNESS.md](EXACTNESS.md) (see Exactness below).

## Design

- **Sequence-owned rows.** Each sequence lives on one GPU: its KV pages, its attention, its indexer and Engram state.
  Nothing is replicated, so the four GPUs together hold about three times the KV of the TP4 layout (measured below).
  Dense layers run on each GPU's own rows. The routed experts are split exactly like ds41h's TP4 slices, byte for
  byte, so each MoE call computes the same products. An all-gather before and a reduce-scatter after each MoE layer
  move the rows; the reduce-scatter is fused into the MoE's combine kernel and the all-gather runs on the copy engines.
- **Mixed steps.** One step carries all decode rows plus prefill chunks up to a 12,288-token target across the four
  GPUs ([`fpr/serve/scheduler.py`](../fpr/serve/scheduler.py)). Decode steps replay CUDA graphs per row bucket,
  including variants for contexts up to 655,360 positions.
- **Context-parallel prefill (CP).** A long prompt's chunks are spread over all four GPUs
  ([`fpr/engine/cp.py`](../fpr/engine/cp.py)); this is mandatory above 32,768 uncached tokens.
- **Fused head.** LM head, sampler and DSpark drafter in one path ([`fpr/kernels/head`](../fpr/kernels/head)).
- **Pipelined dispatch (optional, `ASYNC=1`).** The host prepares step n+1 while the GPUs run step n.
- **Indexer and Engram.** The indexer of [`fpr/kernels/indexer2f`](../fpr/kernels/indexer2f) and Engram tables in host
  RAM ([`fpr/kernels/engram`](../fpr/kernels/engram)).

## Images and the Responses API

vLLM's image processor runs unchanged and turns each image into patches and a span of prompt positions (at most 1,024
per image). FPRv1 encodes the images itself, on the GPU that computes the span's rows: the sequence's own GPU, or a CP
helper GPU for its share of a long prompt. The vision tower ([`fpr/engine/vision.py`](../fpr/engine/vision.py)) is the
model's ViT and aligner and gives the same rows as vLLM's, bit for bit. Those rows replace the token embeddings of the
span at the first layer. Image positions are routed with the router's vision bias on every MoE layer, as in the
reference model, and Engram skips them. The DSpark drafter only proposes text tokens.

The host side ([`fpr/adapter/mm.py`](../fpr/adapter/mm.py)) maps each step's prefill rows to the image spans they
cover, including chunk boundaries inside a span, replays after a prefix-cache hit, CP sub-chunks and recomputes after a
preemption. Each GPU keeps encoded images in an LRU cache (`PHYS_MM_CACHE_MB`, default 512 MB). An image's identity is
a hash of its processed content, never a client-supplied `uuid`. The prefix cache keys the KV of every block that
contains image positions with these hashes, so the same tokens with a different image never share KV. A sequence with
images stays on its GPU until its decode starts.

With images on, the server sets aside 2.6 GB per GPU before it sizes the KV pools: the vision tower's weights, the
encoded-image cache and a 1 GiB margin (`PHYS_MM_MARGIN_MB`) for the workspaces of large mixed steps. The KV cache then
holds 58.8M tokens. `IMAGES=0` serves text only (`--language-model-only`) and keeps the full 64.4M. With images on and
text requests, the engine's results are bit for bit those of the text-only path, in one process.

`RESPONSES_STORE=1` (the default) keeps the results of `/v1/responses` requests made with `store: true`, so they can
be read back with `GET /v1/responses/{id}` and continued with `previous_response_id`. Stock vLLM keeps them in
dictionaries that never shrink; [`fpr/adapter/responses_store.py`](../fpr/adapter/responses_store.py) bounds them as an
LRU of `PHYS_RESPONSES_STORE_MAX` responses (default 4,096). Stored responses live in the server process: a restart
loses them, and an evicted id answers 404 like an unknown one.

## Performance

Measured on 2026-11-08 on one machine: 4× H200 NVL at a 600 W GPU power limit, official checkpoint, 16-bit
activations. ds41h ran its THROUGHPUT configuration (DSpark k=3) the same day with the same tools. ds41h production of 2026-11-08 = the 2026-11-01 release with `DS41H_MOE_SMALL_T=1024` and a grammar-masking change that is not published yet; these benchmarks use no structured output.

**Summary.** Decode with pipelined dispatch, tok/s while all streams run / over the whole run; prefill of one prompt on
a warm server; KV as tokens held at once. The tables below give the details and the synchronous mode.

| | ds41h, production of 2026-11-08 | FPRv1 | FPRv1 vs production |
|---|---|---|---|
| Decode, 1 user, tok/s | 530 / 503 | 321 / 295 | 0.61× / 0.59× |
| Decode, 64 users, tok/s | 6,759 / 5,626 | 8,895 / 6,857 | **1.32× / 1.22×** |
| Decode, 256 users, tok/s | 12,203 / 10,513 | 20,791 / 15,736 | **1.70× / 1.50×** |
| Prefill 8K, K tok/s | 53.1 | 54.4 | 1.02× |
| Prefill 16K | 62.6 | 54.4 | 0.87× |
| Prefill 32K | 60.5 | 60.2 | 1.00× |
| Prefill 64K | 57.3 | 63.4 | **1.11×** |
| Prefill 128K | 56.6 | 64.3 | **1.14×** |
| Prefill 250K | 52.3 (4.78 s) | 61.7 (4.05 s) | **1.18×** |
| KV cache, M tokens | ~21 | 64.4 (58.8 with images on) | **~3.0×** (2.8×) |

**Decode.** `scripts/bench_concurrency.py`: 8K-token prompts, thinking off, 2,048 output tokens per stream, k=3.
Each cell is the aggregate rate while all streams decode / all output tokens over the whole run, in tok/s.

| Users | ds41h, production of 2026-11-08 | FPRv1, synchronous | FPRv1, pipelined (`ASYNC=1`) |
|---|---|---|---|
| 1 | 530 / 503 | 291 / 267 | 321 / 295 |
| 64 | 6,759 / 5,626 | 8,172 / 6,328 | 8,895 / 6,857 |
| 256 | 12,203 / 10,513 | 17,559 / 13,991 | **20,791 / 15,736** |

At 256 users the pipelined engine decodes 1.70× faster than ds41h while all streams run, and 1.50× over the whole run.

**Prefill.** One prompt at a time on a warm server, prompt tokens / time to first token, median of 3 (the bench tool
from [BRRR](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR)). ds41h ran with a 655,360-token maximum
context here so that the 250K prompt fits.

| Prompt | ds41h, production of 2026-11-08 | FPRv1 |
|---|---|---|
| 8K | 53.1K tok/s (0.16 s) | 54.4K tok/s (0.15 s) |
| 16K | 62.6K tok/s | 54.4K tok/s |
| 32K | 60.5K tok/s | 60.2K tok/s |
| 64K | 57.3K tok/s | 63.4K tok/s |
| 128K | 56.6K tok/s | 64.3K tok/s |
| 250K | 52.3K tok/s (4.78 s) | 61.7K tok/s (4.05 s) |

**KV capacity**, as tokens held at once:

| | Tokens | How it was measured |
|---|---|---|
| ds41h, production of 2026-11-08 (TP4, 256 seats, 262,144-token maximum context) | about 21M | 256 streams held 19.9M resident tokens at 95.4 % KV usage without a preemption |
| FPRv1, text only (`IMAGES=0`) | 64.4M | 32 unique 128K prompts held 4.24M resident tokens at 6.58 % pool usage; the pool is 251,692 pages × 64 tokens × 4 GPUs |
| FPRv1 with images on (the default) | 58.8M | the pool is 229,564 pages × 64 tokens × 4 GPUs |

vLLM's start-up log reports ds41h's KV cache as about 7.0M tokens, in a different unit from the measured figure.
In practice both servers reach their 256-seat limit before their KV limit.

## Exactness

The reference is ds41h's own run-to-run variation: the KL divergence between two ds41h runs on the same inputs. FPRv1
passes a group of items when its KL against ds41h stays within 1.2× that floor, in the mean and at p99. The release
was certified as follows:

| Check | Result |
|---|---|
| Offline, 38 items in 3 item orders | pass in all three |
| CP prefill, 3 orders | KL 1.04× (mean) and 0.97× (p99) of the floor; mean NLL change +0.0015, at the method's limit |
| Served, synchronous, 3 runs | teacher-forced pass in every group; draft acceptance at position 0, median 0.798 |
| Served, pipelined, 3 runs | teacher-forced pass in every group; draft acceptance at position 0, median 0.796 |
| Repeated drains: 4× full load, then a full drain, on one server | no fault |
| Images, text bitwise: text items with the image path on, against FPRv1 | identical where FPRv1 itself is reproducible (38 alone, 8 CP); batched within FPRv1's own batched variation |
| Images served, synchronous, 2 runs: 29 image items, teacher-forced, against ds41h with an unsharded vision tower (`--mm-encoder-tp-mode data`) | pass in every image group; all image rows: KL 0.99×, p99 0.98× of the floor, top-1 disagreement 1.91 % (limit 2.15 %) |
| The same runs against ds41h's default, sharded vision tower | all image rows at ds41h's own largest run-to-run difference; 4 of 8 groups miss the per-group margin (see below) |
| Image cache and identity | the same tokens with another image never share KV; a client `uuid` reused on another image gets that image's answer; `uuid`-only references are refused |
| Responses API | store, GET by id, `previous_response_id` chains, image input, unknown id 404 |

Like ds41h, FPRv1 is not bit-for-bit reproducible from one process to the next once requests are batched. Two
fresh processes running the same 38 items batched differ by a KL of about 1e-3 per item (at most 4.6e-3), the same
order as ds41h's own run-to-run variation (about 4e-3). Run alone, 37 of the 38 items are bit-identical across
processes, and with CP all 8 CP items are.

ds41h's own draft acceptance at position 0 is 0.813 ± 0.020 on the same items. The CP NLL change most likely comes from
where the chunk ends fall (KV that later chunks read is cut at each chunk end, as in ds41h's own chunked prefill). The CP
attention itself is bit-identical to a single GPU holding the whole prompt.

On image prompts the vision tower decides the reference. Stock vLLM at TP4 shards the tower over the four GPUs by
default (`--mm-encoder-tp-mode weights`). FPRv1 runs it whole on one GPU, as vLLM does with `--mm-encoder-tp-mode
data`, and against ds41h in that mode it passes in every image group. Against ds41h's default sharded tower, all
image rows together stay at ds41h's own largest run-to-run difference, but four of eight image groups miss the
per-group margin, by 0.03-0.35 percentage points of top-1 disagreement and in one group by 0.0004 of mean NLL.
ds41h's two tower modes differ from each other by the same amount and in the same way: one image group fails the
same check, and the extra differences are spread over the whole generation in both cases. So the difference comes
from the rounding of the sharded tower, not from FPRv1.

## Known limits

- One stream decodes at 0.55-0.61× ds41h's rate. FPRv1's single-stream step is bound by host-side launch cost, and it
  has no counterpart yet to ds41h's fused decoder for steps of up to 8 tokens.
- A single 16K prompt is slower than on ds41h (0.30 s against 0.26 s).
- Images: FPRv1 matches ds41h with an unsharded vision tower. Against ds41h's default sharded tower it differs by the
  same small amount as ds41h's two tower modes differ from each other (see Exactness).
- Exactly four H200 GPUs (`sm_90a`), all driven by one process.
- The same vLLM 0.32.0 dependency as ds41h.

## Running

```bash
source scripts/env.sh
bash scripts/prebuild_fpr.sh                       # on the CPU: CUDA extensions and the converted-weight cache
MODEL=~/.ds41h/model bash scripts/serve_fpr.sh     # synchronous, 224 seats, localhost:8100
ASYNC=1 MAX_SEQS=256 MODEL=~/.ds41h/model bash scripts/serve_fpr.sh   # pipelined dispatch
```

The configuration is [`fpr/config/fprv1.env`](../fpr/config/fprv1.env). `scripts/run_fpr_endpoint.example.sh` shows HTTPS
with API keys.

## Layout

| Path | Contents |
|---|---|
| `fpr/engine/` | The 40-layer step: decode buckets and graphs, mixed steps, CP, pipelined dispatch, the indexer hook, the vision tower |
| `fpr/serve/` | Scheduler, KV manager (pages, prefix cache, moves between GPUs), worker |
| `fpr/adapter/` | vLLM plugins: the scheduler and executor classes, images (host side), the bounded Responses store, warm-up |
| `fpr/kernels/` | CUDA kernels and their wrappers: all-gather / reduce-scatter, attention, MoE, mHC, head, indexer, Engram |
| `fpr/tools/` | The converted-weight cache |
| `fpr/config/` | The serving configuration |
