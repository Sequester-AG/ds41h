# ds41h

High-throughput serving of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
on four NVIDIA H200 NVL GPUs, on top of the vLLM 0.32.0 wheel.

ds41h is not a fork of vLLM. It patches vLLM's modules at import time and replaces the hot paths with custom CUDA
kernels: decoder layers, sparse attention, routed MoE, DSpark drafting and sampling, Engram lookups and the NVLink
all-reduce. Each component has its own flag and falls back to the stock path when it is off or a request does not
qualify. Two small source patches to the installed wheel live in `patches/`, and the setup script applies them.

The default configuration keeps stock vLLM's precision: the checkpoint as shipped (MXFP4 experts, FP8 dense weights)
with 16-bit activations. [docs/EXACTNESS.md](docs/EXACTNESS.md) describes how this is checked against stock vLLM.

## Performance

Measured on 2026-11-01 at a 600 W GPU power limit, with TP4 and a 262,144-token maximum context. Thinking was off and
every request ran to its full length. Details and caveats are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Decode, 8K-token prompt | DSpark k=3 | DSpark k=5 |
|---|---|---|
| 1 user | 521 tok/s | 653 tok/s |
| 64 users, total | 6,711 tok/s | 7,490 tok/s |
| 256 users, total over the whole run | 9,900 tok/s | 10,128 tok/s |

Prefill with k=5 reaches 42.4K tok/s at 8K tokens and 61.8K tok/s at 16K tokens.

## FPRv1

FPRv1 (First Principles Rewrite v1, in [`fpr/`](fpr)) is a second engine for the same model and machine, behind the same
vLLM API layer. Each sequence lives on one GPU, the experts are split exactly like ds41h's TP4 slices, long prompts
are prefilled across all four GPUs, and decode can be pipelined. It serves images and the Responses API with stored
responses.

**FPRv1 against ds41h production**, both measured on 2026-11-08 on the same machine (4× H200 NVL, 600 W), ds41h in its THROUGHPUT
configuration. Decode: 8K-token prompts, thinking off, DSpark k=3, 2,048 output tokens per stream, tok/s while all
streams run / over the whole run; FPRv1 with pipelined dispatch. Prefill: one prompt on a warm server, prompt tokens /
time to first token. KV: tokens held at once, measured.

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

ds41h production of 2026-11-08 = the 2026-11-01 release with `DS41H_MOE_SMALL_T=1024` and a grammar-masking change that is not published yet; these benchmarks use no structured output.

Design, the synchronous-mode numbers, the exactness results and the known limits are in [docs/FPR.md](docs/FPR.md).
`scripts/serve_fpr.sh` starts it.

## Requirements

- 4× H200 NVL (`sm_90a`) in one host. The kernels are compiled for this architecture and for exactly four
  tensor-parallel ranks.
- Linux x86-64 and an NVIDIA driver that supports CUDA 13.
- About 480 GB of fast disk for the checkpoint, and about 25 GB for the environment.
- Room for the Engram tables, by mode: about 210 GB on NVMe or about 195 GiB of RAM in `/dev/shm`.
  See [docs/ENGRAM.md](docs/ENGRAM.md).
- [uv](https://docs.astral.sh/uv/), git, patch and a C++ toolchain.

Tested only on one machine: 4× H200 NVL, 1 TB of RAM, driver 595.91.07, Linux 7.0 and glibc 2.43.

## Quick start

Build the environment. This takes no GPU time and installs into `~/.ds41h` (set `DS41H_HOME` to change that):

```bash
bash setup/build-venv.sh
```

Download the checkpoint:

```bash
~/.ds41h/venv/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir ~/.ds41h/model
```

Start the server on `127.0.0.1:8100`. The first start compiles the kernels, which takes about 10 minutes:

```bash
THROUGHPUT=1 ENGRAM=host MAXLEN=262144 MAX_SEQS=256 CAPTURE=1024 GPU_UTIL=0.91 scripts/serve_ds41h.sh
```

`CAPTURE` should be `MAX_SEQS` × (k + 1), here 256 × 4.

Send a request:

```bash
curl -s http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v41-flash-h200","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## Configurations

[scripts/serve_ds41h.sh](scripts/serve_ds41h.sh) documents every option in its header. The main ones are:

| Setting | Effect |
|---|---|
| Default | Latency line: DSpark k=7, for few concurrent streams |
| `THROUGHPUT=1` | Throughput line: k=3 and grouped sparse decode, for many concurrent long-context streams |
| `SPEC_K=<n>` | Another draft length |
| `ENGRAM=nvme`, `host` or `hbm` | Where the Engram n-gram tables live |
| `KVRAM=auto` | Host-RAM KV cache tier, one copy shared by the four ranks |
| `DSML_RT=1` (default) | DeepSeek API parity for tool calls, reasoning round trips and request options |
| `EXTRA="..."` | Further `vllm serve` arguments. `--no-language-model-only` enables images |

[scripts/run_endpoint.example.sh](scripts/run_endpoint.example.sh) is a template for an HTTPS endpoint with API keys.

## Security

Stock vLLM 0.32.0 checks the API key only on paths under `/v1`, `/v2`, `/inference` and `/cohere`. Other routes stay open
even with keys configured, including `/invocations`, which runs the model, and `/scale_elastic_ep`.
`patches/vllm-0.32.0-auth-all-paths.patch` requires the key on every path except `/health`. `setup/build-venv.sh` applies
it, so check that it is in place before you expose a server. With no keys configured, every route is open.

With images on, stock vLLM 0.32.0 also lets a client name an image with its own `uuid` and then uses that `uuid` as
the image's identity in the processor cache, the encoder cache and the prefix-cache keys, which all clients share. A
reused `uuid` therefore hands one client another client's image: a later request with a different image is answered
about the first one, even when the two requests use different `cache_salt` values (measured on this stack).
`patches/vllm-0.32.0-mm-content-identity.patch` makes the identity the hash of the image content, ignores client
`uuid`s, and refuses image parts that carry a `uuid` but no data. Apply it before serving images to more than one client.

## Tests

The CPU tests need the checkpoint on disk:

```bash
source scripts/env.sh
export PYTHONPATH=$PWD:$PYTHONPATH DS41H_MODEL_DIR=~/.ds41h/model
python tests/test_ref_cpu.py
python tests/test_fast_tokenize.py
python tests/test_reason_count.py /tmp/reason_count.json
```

The other files in `tests/` compare each kernel with `ds41h/ref/`, a pure-torch reference built from DeepSeek's official
inference code, or with stock vLLM. They need GPUs and the real weights, and their docstrings give the commands.

## Layout

| Path | Contents |
|---|---|
| `ds41h/` | Import hooks (`hook.py`, `patch_*.py`), flags (`flags.py`), kernel wrappers, pure-torch reference (`ref/`) |
| `ds41h/kernels/` | CUDA kernels, compiled on first use |
| `plugins/` | vLLM plugins: DeepSeek API parity, host-RAM KV tier |
| `runtime/` | `sitecustomize.py`: installs the hooks when `DS41H=1`, maps large checkpoints with `MAP_NORESERVE` |
| `patches/` | Fixes to the installed vLLM wheel, applied by `setup/` |
| `scripts/` | Launchers, kernel prebuild, benchmarks and profilers, exactness tools |
| `setup/` | Pinned package versions and the environment build |
| `docs/` | Design, exactness method, benchmarks, Engram modes, FPRv1 |
| `fpr/` | FPRv1: engine, scheduler and KV manager, vLLM plugins, kernels, configuration ([docs/FPR.md](docs/FPR.md)) |

## Known limits

- It depends on vLLM 0.32.0. Several hooks compare a hash of the stock source and need review on any other version.
- `patches/vllm-0.32.0-reasoning-count.patch` fixes a vLLM streaming cost that grew with output length when
  `continuous_usage_stats` is on. Neither patch has been submitted upstream yet.
- At 256 streams the aggregate decode rate falls during a run (about 12.1K, 10.7K and 9.1K tok/s across the early,
  middle and late thirds at k=5, 8K). This is not fully explained.
- On very short prompts with thinking on and no system prompt, the model can emit garbage tokens when it skips its
  reasoning. DeepSeek's own API does the same. A system prompt or turning thinking off avoids it.
- Comments in the code can cite labels of internal benchmark runs. These runs are not published.

## Acknowledgements

ds41h builds on [BRRR](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR) by Fraser Price, whose approach to
persistent decoder kernels, DSpark sampling and NVLink all-reduce it adapts for `sm_90a` and stock vLLM. It also uses
[vLLM](https://github.com/vllm-project/vllm), [FlashMLA](https://github.com/vllm-project/FlashMLA) and DeepSeek's
reference inference code. See [NOTICE](NOTICE).

## License

Apache License 2.0, see [LICENSE](LICENSE). Third-party notices are in [NOTICE](NOTICE) and [LICENSES/](LICENSES).
The model weights are not part of this repository and have their own license.
