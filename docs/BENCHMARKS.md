# Benchmarks

All numbers were measured on 2026-11-01 on one machine: 4× H200 NVL, 600 W GPU power limit, official checkpoint,
16-bit activations.

Server settings, with `CAPTURE=1536` at k=5 and `CAPTURE=1024` at k=3:

```bash
THROUGHPUT=1 SPEC_K=5 ENGRAM=host MAXLEN=262144 MAX_SEQS=256 GPU_UTIL=0.91 CAPTURE=1536 scripts/serve_ds41h.sh
```

This gives a GPU KV cache of about 6.6M tokens at k=5 and 7.1M at k=3.

## Decode, 1 to 64 users

Tokens per second, using the bench tool from [BRRR](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR): thinking off,
`ignore_eos`, a fresh prompt for each stream. Values for 64 users are aggregate.

| Prompt | Users | k=3 | k=5 | k=7 |
|---|---|---|---|---|
| 2K | 1 | 514 | 661 | 702 |
| 2K | 4 | 1,104 | 1,385 | 1,461 |
| 2K | 64 | 6,807 | 7,697 | not measured |
| 8K | 1 | 521 | 653 | not measured |
| 8K | 64 | 6,711 | 7,490 | not measured |
| 32K | 1 | not measured | 669 | not measured |

These runs predate the streaming fix below. The bench tool has `continuous_usage_stats` on, so they may understate
what the current code reaches, especially at 64 users.

## Decode, 256 users

Aggregate tokens per second from `scripts/bench_concurrency.py`, with 2,048 output tokens per stream. A run counts only
if all 256 streams return HTTP 200 and all 2,048 tokens. The bench tool from BRRR could not complete 128 or 256 streams
here, so this client measures them instead.

| Prompt | k | Window after the last stream passes 300 tokens | Whole run | Last token |
|---|---|---|---|---|
| 2K | 5 | 12,933 / 12,956 / 13,110 | 10,280 / 10,572 / 10,311 | 51.0 / 49.6 / 50.8 s |
| 8K | 5 | 12,538 | 10,128 | 51.8 s |
| 2K | 3 | 11,902 | 10,016 | 52.3 s |
| 8K | 3 | 11,681 | 9,900 | 53.0 s |

Quote the whole-run figure: the window figure covers only the fast early part of the run. The rate falls during a run,
from about 12.1K to 10.7K to 9.1K tok/s across the early, middle and late thirds at k=5, 8K. The cause is not fully
explained.

The API process used 18–25% of one CPU core during these runs.

## The streaming fix

With `stream_options.continuous_usage_stats` on, stock vLLM 0.32.0 re-decodes and re-parses the entire answer on every
streamed chunk to count reasoning tokens, so the cost grows with output length.
`patches/vllm-0.32.0-reasoning-count.patch` counts incrementally, and `setup/build-venv.sh` applies it.

| Workload | Whole run before | Whole run after | Last token before | Last token after |
|---|---|---|---|---|
| k=5, 2K | ~7,000 tok/s | ~10,400 tok/s | ~74 s | ~50 s |
| k=5, 8K | ~6,800 tok/s | 10,128 tok/s | ~77 s | 51.8 s |
| k=3, 2K | ~5,500 tok/s | 10,016 tok/s | ~95 s | 52.3 s |
| k=3, 8K | ~5,350 tok/s | 9,900 tok/s | ~98 s | 53.0 s |

`tests/test_reason_count.py` checks the fix on the CPU over 41 cases. The final counts matched a full parse in every
run, and the output was byte-identical once tool-call IDs were normalized. Seven chunks differ mid-stream by at most 3
tokens, in a thinking-on path whose behaviour the fix does not otherwise change. The non-streaming thinking-off count
and the Responses API were not changed or checked.

Clients that leave `continuous_usage_stats` off were never affected.

## Prefill

With k=5, using the bench tool from BRRR (prompt tokens divided by time to first token):

| Prompt | Prefill |
|---|---|
| 8K | 42.4K tok/s |
| 16K | 61.8K tok/s |

## Reproducing

The benchmark clients import `brrr.bench`, so install BRRR into a separate environment:

```bash
python -m venv ~/.ds41h/bench && ~/.ds41h/bench/bin/pip install git+https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR
```

Then, with the server running:

```bash
~/.ds41h/bench/bin/bench --port 8100 -s 8192 -c 1 64 -r 1
```

```bash
~/.ds41h/bench/bin/python scripts/bench_concurrency.py --conc 256 --size 8192
```
