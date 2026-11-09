#!/bin/bash
# Template: FPRv1 behind HTTPS with API keys. Copy it, fill in the paths, keep the copy private.
#
# Before exposing it, check that setup/apply-vllm-patches.sh has run: stock vLLM leaves routes outside /v1 open without a key,
# and keys its image caches by a client-supplied uuid (README.md, Security).
#
# auth.yaml (vLLM reads it through --config; keep it mode 600 and outside the repository):
#   api-key:
#     - <key for client A>
#     - <key for client B>
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TLS=/path/to/tls        # server.key, server.crt, ca.pem
AUTH=/path/to/auth.yaml
export PORT=8443
export EXTRA="--host 0.0.0.0 --port $PORT --ssl-keyfile $TLS/server.key --ssl-certfile $TLS/server.crt --config $AUTH"
export FPR_WARM_URL=https://127.0.0.1:$PORT FPR_WARM_CACERT=$TLS/ca.pem FPR_WARM_AUTH=$AUTH
exec bash "$REPO/scripts/serve_fpr.sh"
