# ds41h native (4x H200), release R3

Native C/C++ serving of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on one
node with **4x H200** (141 GB each, x86_64, `sm_90a`). The vLLM-based ds41h stack it replaces is at tag `FPRv1`.

- **Engine.** Each sequence lives on one GPU. The experts are split across the four GPUs, and long prompts are
  prefilled across all four. The Engram n-gram tables (~190 GiB) live in host RAM, not in GPU memory.
- **Native loop.** The scheduler, KV manager, block tables, step builders and engine loop are C++. Decode runs on CUDA
  graphs launched by a C++ driver. A C++ frontend replaces vLLM's Python API server, including HTTP/TLS, API keys,
  chat templates and the tool-call parsers. Its deliberate differences are listed in
  [frontend_cxx/DEVIATIONS.md](frontend_cxx/DEVIATIONS.md).
- **Kernels.** Custom sm_90a kernels: a persistent decoder kernel, the sparse-attention indexer, routed-expert GEMMs
  and NVLink all-gather / reduce-scatter.
- **Precision.** No quantization beyond the checkpoint's own formats (MXFP4 experts, FP8 dense weights).

vLLM 0.32.0 still launches the engine process. Python remains for start-up, weight loading, and steps the native loop
does not take, such as image steps.

## Performance

8K prompts, 2,048 output tokens, thinking off, aggregate decode tok/s at *n* concurrent streams (Cn).

### Against stock vLLM at 600 W (estimate)

Stock vLLM 0.32.0 with the DSpark drafter (k=5) was measured on the same host with the GPUs power-capped at ~200 W.
A hardware check in the same session ran GEMMs at ~1/4 and HBM reads at ~1/2 of their 600 W throughput. The 600 W
column scales the stock run by those factors: x1.5-2 for decode (bandwidth- and latency-bound) and x2.5-3.5 for
prefill (mostly GEMMs). ds41h native was measured at 600 W, with DSpark k=3.

| | Stock, measured at ~200 W | Stock, estimated at 600 W | ds41h native, 600 W | Gain at 600 W |
|---|---|---|---|---|
| Decode, 1 stream | 300 tok/s | 450-600 tok/s | 622 tok/s | ~x1.0-1.4 |
| Prefill | 11.2K tok/s | 27-38K tok/s | 52-65K tok/s | ~x1.5-2.4 |
| KV cache | 613K tokens (vLLM's count, 36,864 max context) | same | 67.8M tokens | x35+ |

- **KV cache.** Stock keeps the Engram tables in GPU memory, so its KV cache is small. vLLM's start-up count
  understates tokens held at once by up to ~3x, so x35 is the floor and x110 the raw ratio.
- **Concurrency.** At 64 or more streams stock was not measured. Its KV cache holds roughly 60-180 streams of 10K
  tokens, so 256 streams do not fit.

### Against ds41h-on-vLLM (tag `FPRv1`, measured)

| | ds41h-on-vLLM | ds41h native | Gain |
|---|---|---|---|
| C1 | 530 | 622 | x1.17 |
| C64 | 6,759 | 9,000-9,115 | x1.33-1.35 |
| C128 | - | 14,325 | - |
| C256 | 12,203 | 19,462-19,499 | x1.60 |
| Prefill, 8K-128K prompts | 53-61K tok/s | 52-65K tok/s | x1.0-1.13 |
| First token at 250-290K | 4.78 s (250K) | 4.71 s (290K) | - |
| KV cache, tokens held at once | ~21M | 67.8M | x3.2 |

### R3 on an agentic scan workload (measured)

85 concurrent multi-turn agent conversations on one 4x H200 node: contexts up to 400K tokens, ~2.3K new prompt tokens
and ~45 output tokens per turn on average (most of each prompt hits the prefix cache), 14-minute runs, steady state
from minute 5. R2 and R3 were measured back to back on the same host.

| | R2 | R3 | Gain |
|---|---|---|---|
| Output tok/s | 3,400-3,580 | 3,860-3,960 | ~x1.2-1.25 |
| Decode tok/s per request | 42-44 | 51-52 | ~x1.2 |
| GPU busy (SM activity) | 75-85 % | ~88 % | |
| First token, p50 | 0.37 s | 1.1 s | slower (by design) |

R3 changes, all exact (outputs are unchanged):

- **Long-context steps with page copies stay native.** Batched decode steps past 262K context that also copy prefix
  pages ran in the Python fallback path; they now take the native loop (`packages/nd`, `nd_engine.cpp`).
- **Fewer, fuller mixed steps.** `mixed_min_decode_steps=10` (`fpr/config/h200_native.env`) runs at least 10
  decode-only steps between prefill-carrying steps under load, so each mixed step carries more prefill. This trades
  ~0.6 s of first-token latency for throughput.
- **Start-up under load.** Decode-bucket precreation now finishes after 45 s or 1,024 steps
  (`ENG_DECODE_PRECREATE_DEADLINE_S` / `_SERIALS`) instead of waiting for a quiet moment. A restart under constant
  traffic no longer runs at ~1/3 speed until the traffic stops.

## Requirements

- 4x H200 NVL or SXM in one x86_64 node, NVLink between all four. The kernels are built for exactly four GPUs.
- CUDA 13 driver, g++ >= 13, CMake >= 3.20, [uv](https://docs.astral.sh/uv/), OpenSSL and zlib headers.
- The checkpoint (~480 GB), plus ~195 GiB of host RAM for the Engram tables in `/dev/shm`.

## Build and run

Building runs on the CPU only.

```bash
cp deploy/h200/site.env.example deploy/h200/site.env    # edit the paths, model, TLS and key locations
source deploy/h200/site.env
bash setup/build-venv.sh                                # vLLM 0.32.0 + patches/
bash deploy/h200/build_native.sh                        # frontend, xgrammar, C++ core, native decode, CUDA extensions, weight cache
cp deploy/h200/endpoint_auth.example.yaml $DS41H_HOME/keys/endpoint_auth.yaml   # put real keys in, chmod 600
bash deploy/h200/switch_native.sh                       # start, warm up, health check, watchdog
```

- **Before building.** Put the frontend's third-party sources in `frontend_cxx/third_party/dl/` first: zeromq 4.3.5,
  oniguruma 6.9.10, libjpeg-turbo 3.2.0 and simdjson 3.12.3.
- **`run_native.sh`** starts the server over TLS with API keys. `NATIVE_PRIVATE=1` runs it on 127.0.0.1:8100 without
  TLS for benchmarks. Wait for "warm-up done" before sending load.
- **Rollback.** `rollback_native.sh` restarts the previous configuration. `watchdog.sh` and `mem_guard.sh` call it on
  a fault, and `touch $DS41H_RUN/NO_FALLBACK` pauses them.
- **Required settings.** The served configuration is `fpr/config/h200_native.env` plus `deploy/h200/native.env` (256
  seats). It needs head-graph preload off (`ENG_*_HEAD_PRECAPTURE_PRELOAD=0`) and `PHYS_KEEP_GRAPHS=1`.

## Layout

| Path | Contents |
|---|---|
| `fpr/engine/`, `fpr/kernels/` | Model step, attention, indexer; CUDA kernels and loaders |
| `fpr/core/`, `packages/nd/` | C++ scheduler, KV manager, block tables, engine process, native decode |
| `frontend_cxx/` | C++ API frontend |
| `ds41h/`, `plugins/`, `patches/` | Kernels and hooks shared with ds41h-on-vLLM; vLLM plugins and patches |
| `fpr/config/`, `scripts/`, `deploy/h200/` | Serving configuration, launchers, build, switch / rollback and guards |
| `docs/` | Design, exactness, Engram and FPRv1 notes (from `FPRv1`) |

## Known limits

- Tied to vLLM 0.32.0, CUDA 13 and `sm_90a`, with exactly four GPUs per node. Tested on one 4x H200 NVL node.
- Image steps run on the Python path.
- Flags that are off in the served configuration may refer to kernels that are not included here.
- Some modules with test-like names (`bench2b.py`, `test_comm.py`, `test_mhc.py`, `moe_stream/bench.py`,
  `commbench.py`, `cap_check.py`) are runtime libraries the engine imports.

## License

Apache 2.0, see [LICENSE](LICENSE). Third-party notices are in [NOTICE](NOTICE) and [LICENSES/](LICENSES). The model
weights have their own license.
