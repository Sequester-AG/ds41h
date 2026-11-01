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
| `docs/` | Design, exactness method, benchmarks, Engram modes |

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
