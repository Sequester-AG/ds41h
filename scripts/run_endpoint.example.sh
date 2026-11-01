#!/bin/bash
# Template: HTTPS endpoint with API keys, images on, host-RAM Engram and KV tier. Copy it, fill in the paths, keep the copy private.
#
# Before exposing it, check that setup/apply-vllm-patches.sh has run: stock vLLM leaves routes outside /v1 open without a key (README.md).
#
# auth.yaml (vLLM reads it through --config; keep it mode 600 and outside the repository):
#   api-key:
#     - <key for client A>
#     - <key for client B>
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TLS=/path/to/tls        # server.key, server.crt
AUTH=/path/to/auth.yaml
export THROUGHPUT=1 ENGRAM=host MAXLEN=655360 MAX_SEQS=128 GPU_UTIL=0.91
export KVRAM=auto KVRAM_MEM_FRACTION=0.935 KVRAM_MEM_GC=0.98
export EXTRA="--host 0.0.0.0 --port 8443 --ssl-keyfile $TLS/server.key --ssl-certfile $TLS/server.crt --config $AUTH --no-language-model-only"
exec bash "$REPO/scripts/serve_ds41h.sh"
