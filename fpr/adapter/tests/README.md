# Adapter tests

The adapter serves the fork behind vLLM 0.32's unchanged API server (PhysScheduler, PhysExecutor, the worker
protocol, the FORCE hook). These tests guard it at three levels.

```bash
python -m pytest -q adapter/tests/                                   # plain CPU tests; the rest skip
PHYS_MODEL=/path/to/model python -m pytest -q adapter/tests/         # + vLLM's engine on the CPU (fake worker)
PHYS_SERVER_URL=http://127.0.0.1:8100 python -m pytest -q adapter/tests/test_api_contract.py
```

| Marker | Needs | Skipped when |
|---|---|---|
| none | CPU, torch, numpy | never |
| `vllm` | vLLM 0.32 with the deepseek_v41 tokenizer and parsers; `PHYS_MODEL` = a model directory with `config.json` and the tokenizer (weights are not read) | vLLM or `PHYS_MODEL` missing |
| `gpu` | one CUDA device | no CUDA device |
| `server` | a running fork server: `PHYS_SERVER_URL`; optional `PHYS_API_KEY_FILE` (a file with one client key, never printed), `PHYS_CA_FILE` (https), `PHYS_SERVER_FAKE=1` (fake worker in script mode, exact replies) | `PHYS_SERVER_URL` unset |

## What each file guards

| File | Marker | Guards |
|---|---|---|
| `test_force_order.py` | none | FORCE dumps stay in forced-token order when a forced request moves between ranks mid-decode (the old per-rank grouping misaligned every row after the first move). Fixed patterns, 40 seeded random rank sequences, partial dumps, both dump kinds. |
| `test_d13_routing.py` | none | ASYNC (D13) routing: page copies, rows beyond the largest graph bucket, prefill / forced / score rows, a decode layout other than k + 1 rows in order, contexts beyond the decode graphs' indexer bound (also for device-resolved starts), and graphs off all take the synchronous fallback with their reason. |
| `test_async_guards.py` | `vllm` | Structured outputs under ASYNC: the hold keeps vLLM from deferring sampling for the whole batch; outputs equal the synchronous run and match their grammar. The merge of a dispatched step runs on the dispatching thread (a merge thread lacked cuBLAS handles for graph capture). |
| `test_cublas_workspace.py` | `gpu` | Destroying a CUDA graph must not free the cuBLAS workspace that other graphs captured on the same stream use (torch 2.13; the ASYNC Xid 31 fault). Checks the keep-graphs stopgap, and that the hazard is still present in the installed torch. |
| `test_api_contract.py` | `server` | The OpenAI API contract: chat / reasoning, tool calls, strict tool calls, reasoning effort (each plain and streaming), prefix-cache usage, completions with token ids, 64 parallel chats, API keys on every path but `/health`, re-render exactness (turn 1's prompt + generation is the exact prefix of turn 2's prompt, plain and streaming), the server's parse equal to the model's official parser, strict tool calls valid against their schemas under concurrency; the Responses API store (store, GET, a `previous_response_id` follow-up, and eviction when `PHYS_SERVER_STORE_MAX` gives the server's cap). |
| `test_contract.py` | `vllm` | PhysScheduler + PhysExecutor against vLLM's own engine: exact output streams through MTP acceptance, stop trimming, preemption + recompute, prefix hits, rank moves; per-step invariant checks. |
| `test_async.py` | `vllm` | Async scheduling equals synchronous scheduling (token ids, finish reasons, grammar, forcing); preemption; host time hidden behind a simulated GPU step; the D13 dispatch model. |
| `test_cp.py` | `vllm` | Context-parallel prefill through the served path: exact streams, per-row prompt-token checks on CP sub-chunks, prefix hits, FORCE. |
| `test_stage4_stub.py` | `vllm` | The stage-4 worker path (serve's Worker with its CPU stub backend) under vLLM's engine: copies, tables, lengths, aborts, preemption, FORCE. |
| `test_responses_store.py` | none | adapter/responses_store.py: vLLM's Responses store bounded as a count LRU (responses evicted with their input messages and stream events, reads refresh), the cap from `PHYS_RESPONSES_STORE_MAX`, the class patched once, the pass-through middleware. |
| `test_mm_planner.py` | none | The host side of images (adapter/mm.py): content keys, the KV salt per block, image spans cut by chunk boundaries, encoder-cache residency (refs while a prefill still has span rows ahead, LRU eviction that never drops a key in use), CP helpers, and a randomized check that the worker's cache mirrors the planner and every image row gets its span row exactly once. |
| `test_images.py` | `vllm` | Images end to end through vLLM's own image processing and the fake worker's KV content model (`PHYS_FAKE_KV=1`): exact streams per image content; same tokens with different images never share KV; a reused client uuid with a different image does not alias (vLLM's block hashes alone would); the same image hits; spans cut by chunks and by the bounded replay after a hit; preemption + recompute; CP prefill with images on helper ranks; images rejected when off. |

`vllm_harness.py` and `engine_util.py` build the CPU engine; `api_script.json` holds the fake worker's scripted replies.
