#!/bin/bash
# FPRv1 launcher: the certified configuration (fpr/config/fprv1.env) behind vLLM 0.32.0's unchanged API layer.
# Localhost, port 8100, unless EXTRA overrides it (scripts/run_fpr_endpoint.example.sh shows HTTPS with API keys).
#
#   MODEL=<dir> scripts/serve_fpr.sh             synchronous scheduling, 224 seats (the certified serving configuration)
#   ASYNC=1 MAX_SEQS=256 scripts/serve_fpr.sh    pipelined decode dispatch: the decode-heavy option, also certified
#   FPR_CONFIG=<file>                            another configuration file (same variables as fpr/config/fprv1.env)
#   PORT=<n>  EXTRA="..."                        port; more `vllm serve` arguments (appended, so they win)
#   FPR_WARM=1 (default) | 0                     after /health, send a few synthetic prefill requests, then print
#                                                "warm-up done" (the first mixed step of a process is slower than the rest)
#   FPR_WARM_URL / FPR_WARM_CACERT / FPR_WARM_AUTH   where the warm-up requests go when EXTRA changes host, TLS or keys
#   IMAGES=1 (default) | 0                       image input (vision tower on every GPU); 0 = text only, images rejected
#   PHYS_MM_MARGIN_MB=1024 (default)             with images: memory kept free for the mixed steps' workspaces,
#                                                on top of the vision tower and the encoded-image cache
#   RESPONSES_STORE=1 (default) | 0              keep /v1/responses results for store:true follow-ups (in memory, per
#                                                process, least recently used dropped past PHYS_RESPONSES_STORE_MAX,
#                                                default 4096: fpr/adapter/responses_store.py); 0 = vLLM's default (off)
#   DRYRUN=1                                     print the environment and the vllm command only
#
# The first start builds the CUDA extensions and the converted-weight cache unless scripts/prebuild_fpr.sh ran before.
# Differences to production's launcher (scripts/serve_ds41h.sh), each needed by FPRv1's engine:
#   --scheduler-cls adapter.phys_scheduler.PhysScheduler           the rank scheduler and sequence-owned KV (fpr/serve)
#   --distributed-executor-backend adapter.phys_executor.PhysExecutor   FPRv1's worker inside the EngineCore process
#   --tensor-parallel-size 1                                       vLLM runs no TP workers: the engine drives all four GPUs
#                                                                  from one process
#   --no-async-scheduling (sync) / --async-scheduling (ASYNC=1)    the step batch is position-explicit
#   --load-format dummy                                            the engine loads the checkpoint itself (fpr/tools/wcache.py)
#   IMAGES=0: --language-model-only                                no vision encoder; images are rejected at the API
#   IMAGES=1: PHYS_MM=1                                            the engine encodes images itself (fpr/engine/vision.py)
#             --mm-processor-cache-gb 0 without patches/vllm-0.32.0-mm-content-identity.patch (see README, Security)
#   --max-num-batched-tokens 17664                                 4 GPUs x (4,096 prefill rows + 80 x 4 decode rows)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh"
CFG=${FPR_CONFIG:-$REPO/fpr/config/fprv1.env}
[ -f "$CFG" ] || { echo "serve_fpr: no configuration $CFG" >&2; exit 2; }
# shellcheck disable=SC1090
source "$CFG"
PHYS=${PHYS_ROOT:-$REPO/fpr}   # PHYS_ROOT: the engine tree, if it is not this repository's fpr/
# where scripts/prebuild_fpr.sh put the extensions (IDX2F_BUILD, MOE_V4_BUILD, TORCH_EXTENSIONS_DIR)
# shellcheck disable=SC1091
[ -f "$PHYS/build/prebuilt.env" ] && source "$PHYS/build/prebuilt.env"
MODEL=${MODEL:-${DS41H_MODEL_DIR:-$DS41H_HOME/model}}
[ -d "$MODEL" ] || [ "${DRYRUN:-0}" = 1 ] || { echo "serve_fpr: no checkpoint in $MODEL (set MODEL)" >&2; exit 2; }
export DS41H_MODEL_DIR=$MODEL PHYS_ROOT=$PHYS
export PYTHONPATH=$PHYS:$REPO${PYTHONPATH:+:$PYTHONPATH}
export DS41H=${DS41H:-1} DS41H_FAST_TOKENIZE=${DS41H_FAST_TOKENIZE:-1} TOKENIZERS_PARALLELISM=true
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-7200}
export PHYS_ADAPTER_WORKER=${PHYS_ADAPTER_WORKER:-stage4}
# Production's MoE settings must be in the environment before vLLM starts: with DS41H=1, vLLM's ds41h plugin imports
# ds41h.moe at start-up, and ds41h.moe reads DS41H_MOE_DEC once at import.
export DS41H_MOE_DEC=${DS41H_MOE_DEC:-1} DS41H_MOE_DEC_CFG=${DS41H_MOE_DEC_CFG:-0x55}
# Bounded tail window in mixed steps: the served engine's logits equal the offline engine's bit for bit.
export ENG_TAIL_BOUND=${ENG_TAIL_BOUND:-1}
# Prebuilt indexer kernels (scripts/prebuild_fpr.sh); otherwise they are compiled at the first GPU start.
export IDX2F_BUILD=${IDX2F_BUILD:-$PHYS/engine/build/idx2f}
# Not expandable segments: the engine's all-gather / reduce-scatter and the head's peer mode store straight into other
# GPUs' torch buffers, and with expandable segments torch maps each segment for its owning GPU only.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}
export PHYS_ENGINE="${PHYS_ENGINE:-$FPR_ENGINE}"
sched="$FPR_SCHED"; [ "${FPR_CP:-0}" = 1 ] && sched="${sched:+$sched,}cp_prefill=1"
export PHYS_SCHED="${PHYS_SCHED:-$sched}"
# shellcheck disable=SC2086
[ -n "${FPR_FLAGS:-}" ] && export $FPR_FLAGS
ASYNC=${ASYNC:-${FPR_ASYNC:-0}}
MAX_SEQS=${MAX_SEQS:-$FPR_MAX_SEQS}
SCHED_CLS=adapter.phys_scheduler.PhysScheduler; ASYNC_FLAG=--no-async-scheduling
if [ "$ASYNC" = "1" ]; then
  SCHED_CLS=adapter.phys_scheduler.PhysAsyncScheduler; ASYNC_FLAG=--async-scheduling; export PHYS_ASYNC=1
  # pipelined dispatch in the engine (d13) with the second upload path it needs, unless PHYS_ENGINE names them
  case ",${PHYS_ENGINE}," in *,d13=*) ;; *) PHYS_ENGINE="$PHYS_ENGINE,d13=1" ;; esac
  case ",${PHYS_ENGINE}," in *,upload_v2=*) ;; *) PHYS_ENGINE="$PHYS_ENGINE,upload_v2=1" ;; esac
  export PHYS_ENGINE
fi
PORT=${PORT:-8100}
MM_ARGS=(--language-model-only)
if [ "${IMAGES:-1}" = 1 ]; then
  export PHYS_MM=1
  MM_ARGS=()
  # Stock vLLM 0.32 keys its processor cache by a client-supplied uuid; without the content-identity patch, run it off.
  inputs_py=$(python -c "import vllm.multimodal.processing.inputs as m; print(m.__file__)" 2>/dev/null || true)
  if [ -z "$inputs_py" ] || ! grep -q "references to media by uuid alone are not supported" "$inputs_py"; then
    echo "serve_fpr: vLLM without the mm content-identity patch: --mm-processor-cache-gb 0" >&2
    MM_ARGS=(--mm-processor-cache-gb 0)
  fi
fi
STORE_ARGS=()
if [ "${RESPONSES_STORE:-1}" = 1 ]; then
  # stock vLLM keeps stored responses in dicts that never shrink: bound them (LRU)
  export VLLM_ENABLE_RESPONSES_API_STORE=1 PHYS_RESPONSES_STORE_MAX=${PHYS_RESPONSES_STORE_MAX:-4096}
  STORE_ARGS=(--middleware adapter.responses_store.BoundedStore)
fi
SPEC='{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
ARGS=(
  "$MODEL" --served-model-name deepseek-v41-flash-h200 --host 127.0.0.1 --port "$PORT"
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --tool-parser-plugin "$REPO/plugins/dsml_roundtrip.py"
  --tensor-parallel-size 1 "${MM_ARGS[@]}"
  --load-format dummy --kv-cache-dtype fp8_ds_mla
  --max-model-len 655360 --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens 17664
  --enable-prefix-caching --enable-prompt-tokens-details "$ASYNC_FLAG"
  --speculative-config "$SPEC"
  --scheduler-cls "$SCHED_CLS"
  --distributed-executor-backend adapter.phys_executor.PhysExecutor
  "${STORE_ARGS[@]}"
)
# shellcheck disable=SC2206
[ -n "${EXTRA:-}" ] && ARGS+=($EXTRA)
echo "serve_fpr: config $CFG engine [$PHYS_ENGINE] flags [${FPR_FLAGS:-}] max_seqs $MAX_SEQS sched [$PHYS_SCHED] async $ASYNC images ${IMAGES:-1} responses-store ${RESPONSES_STORE:-1} model $MODEL"
echo "vllm serve ${ARGS[*]}"
[ "${DRYRUN:-0}" = 1 ] && exit 0
cd "$PHYS"
if [ "${FPR_WARM:-1}" != 1 ]; then exec vllm serve "${ARGS[@]}"; fi
vllm serve "${ARGS[@]}" &
svr=$!
trap 'kill -TERM $svr 2>/dev/null; wait $svr; exit 143' TERM INT
URL=${FPR_WARM_URL:-http://127.0.0.1:$PORT}
ca=(); [ -n "${FPR_WARM_CACERT:-}" ] && ca=(--cacert "$FPR_WARM_CACERT")
until [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "${ca[@]}" "$URL/health" 2>/dev/null)" = 200 ]; do
  kill -0 $svr 2>/dev/null || { wait $svr; exit $?; }
  sleep 2
done
wa=(--url "$URL"); [ -n "${FPR_WARM_CACERT:-}" ] && wa+=(--cacert "$FPR_WARM_CACERT")
[ -n "${FPR_WARM_AUTH:-}" ] && wa+=(--auth-file "$FPR_WARM_AUTH")
if wl=$(timeout 300 python "$PHYS/adapter/warm.py" "${wa[@]}" 2>&1); then
  echo "serve_fpr: warm-up done: $wl"
else
  echo "serve_fpr: warm-up FAILED (the server keeps running): $wl" >&2
fi
wait $svr
