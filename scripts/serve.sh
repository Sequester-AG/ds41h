#!/bin/bash
# Base vLLM launcher (127.0.0.1:8100 unless EXTRA overrides it). All knobs via env; prints the command first.
# scripts/serve_ds41h.sh sets the tuned configuration on top of this; use this script directly for stock comparisons.
#   MODEL=<dir>          checkpoint directory (default $DS41H_HOME/model)
#   SPEC_K=0|3|5|7       DSpark draft tokens (0 = off)          default 7
#   MOE=auto|humming|marlin|flashinfer_cutlass|...                default auto (stock picks humming on sm90)
#   HUMMING_W4A8=0|1     Humming MoE with dynamic FP8 activations (VLLM_HUMMING_INPUT_QUANT_CONFIG)
#   DS41H=0|1            enable ds41h hooks (component flags DS41H_* pass through)
#   MAXLEN, BATCH_TOKENS, MAX_SEQS, GPU_UTIL (default 0.96), CAPTURE, LOAD_FORMAT(dummy for kernel A/B), EXTRA
#   PROFILE=1 [PROFILE_DIR=$DS41H_HOME/runs/prof]   torch profiler config (start/stop via /start_profile)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/scripts/env.sh"
# DS41H_SRC: source tree to import ds41h from (default: this repository);
# DS41H_TORCH_EXT_DIR: JIT extension cache override (env.sh pins TORCH_EXTENSIONS_DIR).
export PYTHONPATH=${DS41H_SRC:-$REPO}:${PYTHONPATH}
[ -n "${DS41H_TORCH_EXT_DIR:-}" ] && export TORCH_EXTENSIONS_DIR=$DS41H_TORCH_EXT_DIR
MODEL=${MODEL:-$MODEL_PATH}
[ -f "$MODEL/config.json" ] || { echo "serve: no checkpoint at $MODEL (set MODEL=<dir>)" >&2; exit 2; }
# ds41h kernels load their own weight copies from DS41H_MODEL_DIR: it must be the served checkpoint
export MODEL DS41H_MODEL_DIR=${DS41H_MODEL_DIR:-$MODEL}
SPEC_K=${SPEC_K:-7}
MOE=${MOE:-auto}
MAXLEN=${MAXLEN:-73728}
# 16384 gives +9% prefill at c1 16K / +5% at c16 vs 8192 but needs ~0.1 GiB more activation memory:
# with MAXLEN=36864 the KV check can fail -> pass BATCH_TOKENS=16384 MAXLEN=32768.
BATCH_TOKENS=${BATCH_TOKENS:-8192}
MAX_SEQS=${MAX_SEQS:-16}
GPU_UTIL=${GPU_UTIL:-0.96}  # stock Humming TP MoE (DS41H=0, MOE=auto) pads experts to 128.35 GiB of weights: pass 0.98 for it
CAPTURE=${CAPTURE:-128}
LOAD_FORMAT=${LOAD_FORMAT:-safetensors}
export DS41H=${DS41H:-0}
# ds41h fast tokenize (ds41h/patch_serving.py) uses rayon-parallel encode_batch: env.sh sets TOKENIZERS_PARALLELISM=false
# (stock vLLM does not); allow it for the server when the patch is on (tokenizers disables it in forked children itself).
if [ "$DS41H" = "1" ] && [ "${DS41H_FAST_TOKENIZE:-1}" != "0" ]; then export TOKENIZERS_PARALLELISM=true; fi
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}
# Persist FlashInfer MoE autotune results (EP + flashinfer_cutlass spends ~15 min tuning otherwise).
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-$DS41H_HOME/cache/flashinfer_autotune}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if [ "${HUMMING_W4A8:-0}" = "1" ]; then
  export VLLM_HUMMING_INPUT_QUANT_CONFIG='{"a_dtype":"float8e4m3","group_size":128,"quant_mode":"dynamic_group","scale_dtype":"float32"}'
fi
ARGS=(
  "$MODEL" --served-model-name deepseek-v41-flash-h200 --host 127.0.0.1 --port 8100
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --tensor-parallel-size 4 --language-model-only
  --load-format "$LOAD_FORMAT" --safetensors-load-strategy lazy
  --kv-cache-dtype fp8_ds_mla --engram-config '{"cpu_offload":false,"dp_shared_memory":false}'
  --max-model-len "$MAXLEN" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BATCH_TOKENS"
  --gpu-memory-utilization "$GPU_UTIL"
  --enable-prefix-caching --enable-prompt-tokens-details --async-scheduling
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --max-cudagraph-capture-size "$CAPTURE"
)
[ "$MOE" != "auto" ] && ARGS+=(--moe-backend "$MOE")
if [ "${PROFILE:-0}" = "1" ]; then
  PROFILE_DIR=${PROFILE_DIR:-$DS41H_HOME/runs/prof}
  mkdir -p "$PROFILE_DIR"
  ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true,\"torch_profiler_use_gzip\":true}")
fi
if [ "$SPEC_K" != "0" ]; then
  ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_K,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false}")
fi
# shellcheck disable=SC2206
[ -n "${EXTRA:-}" ] && ARGS+=($EXTRA)
echo "vllm serve ${ARGS[*]}"
exec vllm serve "${ARGS[@]}"
