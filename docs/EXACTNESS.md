# Exactness: ds41h against stock vLLM 0.32.0

The rule for the default configuration is that the target model stays lossless against stock vLLM 0.32.0 on `sm90`:

- 16-bit (bf16 or fp16) activations go into every target GEMM.
- Weight scales are exact.
- Norm, softmax and routing math are the same as stock.
- Only the float summation order may differ, and being more precise than stock is allowed.

Approximations are allowed only in the DSpark drafter, because rejection sampling keeps the output distribution exact.

## Why the comparison is against stock-vs-stock

Stock vLLM and ds41h are both nondeterministic from run to run, at the bf16 ulp level of the logits (atomics, split-K,
expert reduction order). Two teacher-forced collections on the same stock server already differ by a mean KL of about
4.9e-3. Comparing against zero would therefore fail stock itself, so the gate compares a candidate with the range stock
shows against itself.

## Teacher-forced decode

The harness is `ds41h/patch_force.py` and `scripts/exact_eval.py`. It runs through the real decode and verify path.

`DS41H_FORCE_FILE=<json>` (with `DS41H_FORCE_DUMP=<dir>`) installs instance-level wrappers on the V2 GPUModelRunner after
`load_model`, outside every class-level ds41h patch. For the one request whose prompt token IDs match the force file:

- The drafter's proposals are replaced by the next forced tokens.
- The verify step is forced to accept all of them.
- The target logits of every verified row are copied right after `compute_logits`, or `compute_logits_local` with
  vocab-parallel sampling.

So every step is a real M = 1 + k verify step through the same CUDA graphs and kernels as production, and every server
sees the same token sequence. The hook checks that the input IDs of every decode step equal the forced tokens and that
the API output equals the forced continuation.

The same hook runs on a stock-numerics server: `DS41H=1` with every compute component off (`STOCK=1` in
`scripts/exact_run.sh`).

The prompt set has 6 short chat prompts (code, prose, math, German, Chinese, Japanese; 192 forced positions each) and 3
contexts of about 15K tokens (at most 256 positions each; the few-shot GSM8K prompt yielded 58 because its answer is
short). That gave 1,713 decode rows and 9 prefill rows.

## GSM8K

`scripts/gsm8k_eval.py` runs the first 500 questions of the test split: chat template, thinking off, greedy, at most 512
tokens, concurrency 1. It reports accuracy with a 95% Wilson interval and paired disagreements against stock with an
exact McNemar p-value. GSM8K alone is a weak gate: it could not separate an earlier lossy configuration from stock, while
the teacher-forced KL could.

## Running it

Stock reference:

```bash
STOCK=1 GEN=1 FORCE=1 GSM=1 scripts/exact_run.sh ex_stock SPEC_K=5 MOE=flashinfer_cutlass EXTRA=--enable-expert-parallel GPU_UTIL=0.96 MAXLEN=36864 CAPTURE=64
```

Candidate, in the same shell (`mapfile` keeps each setting as one argument):

```bash
mapfile -t A < <(DSML_RT=0 SPEC_K=5 DRYRUN=1 scripts/serve_ds41h.sh); FORCE=1 FORCE2=1 GSM=1 scripts/exact_run.sh cand "${A[@]}"
```

Analysis, on the CPU, with results under `$DS41H_HOME/runs/exact/`:

```bash
python scripts/exact_eval.py compare "$DS41H_HOME/runs/exact/ex_stock/dump" "$DS41H_HOME/runs/exact/cand/dump" --set "$DS41H_HOME/runs/exact/set_v1.json"
```

```bash
python scripts/gsm8k_eval.py compare "$DS41H_HOME/runs/exact/ex_stock/gsm8k.json" "$DS41H_HOME/runs/exact/cand/gsm8k.json"
```

For a verdict, compare several collections per arm with `scripts/exact_dist.py`. A single run shows a real shift only if
it falls far outside stock's range: 8 forced collections on one stock server failed the single-run criteria in 2 of 8.

The distribution gate passes if, over at least 6 collections, the median of every single-run metric meets its threshold:
mean KL ≤ 7e-3, p99 ≤ 0.15, top-1 ≥ 0.980, max |dlogp| ≤ 1.5.

## Cross-proxy fences

A shared-memory stage filled by TMA or `cp.async.bulk` (async proxy), read with `ld.shared` or `ldmatrix` (generic proxy)
and handed back to the producer needs `fence.proxy.async.shared::cta` between the generic reads and the release. Without
it, the next async write can overtake reads still in flight. The symptom is rare wrong values that appear only under
co-residency: the `small16` MoE kernel returned bad rows in 29 of 1,200 calls under a side-stream load, 0 of 1,200 with
the fence, and never in solo stress. A release that follows only the issue of the loads must be fenced. One that follows
the consumption of the loaded values cannot be overtaken because of the data dependence.

## Reference measurements

Measured at 250 W during development, on an earlier tree of this stack (teacher-forced, full vocabulary, 1,713 rows):

| Comparison | Mean KL | p99 KL | Max KL | Top-1 agreement | Max \|dlogp\| of the forced token |
|---|---|---|---|---|---|
| Stock vs stock, second server | 5.49e-3 | 0.116 | 0.512 | 0.9854 | 0.91 |
| Stock vs stock, same server repeated | 4.89e-3 | 0.092 | 0.461 | 0.9877 | 1.07 |
| ds41h, earlier lossy fast defaults | 1.14e-2 | 0.223 | 1.648 | 0.9766 | 2.38 |
| ds41h, lossless | 5.71e-3 | 0.115 | 0.639 | 0.9842 | 1.08 |

The lossless configuration sits inside the stock-vs-stock range, and the lossy one does not. The lossy defaults are no
longer the default, and `DS41H_LOSSY=1` is the opt-in that restores them.

GSM8K on the same trees:

| Run | Accuracy (95% CI) |
|---|---|
| Stock | 0.974 (0.956–0.985) |
| Stock, second server | 0.976 (0.959–0.986) |
| ds41h, lossless | 0.978 (0.961–0.988) |

Later changes to the default tree were either run through this gate or shown bit-identical by the tests in `tests/` and
by in-situ check switches such as `DS41H_MASK_CHECK=1` and `DS41H_META_CHECK=N`.
