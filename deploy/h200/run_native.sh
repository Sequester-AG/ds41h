#!/bin/bash
# run_native.sh: the public endpoint on the native C/C++ serving stack, 4x H200.
#   0.0.0.0:$PORT over TLS with per-client API keys (AUTH_CONFIG), the C++ frontend (NATIVE_FE=cxx) or vLLM's Python API
#   server (NATIVE_FE=py), the DSML tool parser, EdgeGuard, the warm-up and the native loop switches of native.env.
# Normally started by switch_native.sh; it can also run in the foreground for a first test.
#   NATIVE_ENV=<file>   switches file (default: $DS41H_RUN/native.env, else deploy/h200/native.env)
#   NATIVE_PRIVATE=1    127.0.0.1:8100, plain HTTP, no keys, no EdgeGuard (local benchmarks only)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SITE_ENV:-$HERE/site.env}"
ENVF=${NATIVE_ENV:-$DS41H_RUN/native.env}
[ -f "$ENVF" ] || ENVF=$HERE/native.env
# shellcheck disable=SC1090
source "$ENVF"
[ "${NATIVE_READY:-0}" = 1 ] || { echo "run_native: REFUSED: $ENVF has NATIVE_READY=${NATIVE_READY:-0}"; exit 3; }
NATIVE_FE=${NATIVE_FE:-cxx}
case $NATIVE_FE in
  py) ;;
  cxx) [ -x "${FRONTEND_BIN:-}" ] || { echo "run_native: REFUSED: FRONTEND_BIN '${FRONTEND_BIN:-}' is not an executable"; exit 3; } ;;
  *) echo "run_native: REFUSED: NATIVE_FE '$NATIVE_FE' (py | cxx)"; exit 3 ;;
esac
for kv in ${NATIVE_SW:-}; do export "$kv"; done
NR=${NATIVE_ROOT:-$DS41H_REPO}
cd "$NR" || exit 3
SEATS=${NATIVE_SEATS:-128}
export ENGINE=fpr ASYNC=1 FPR_SEATS=$SEATS MAX_SEQS=$SEATS
export FPR_HSD_CAP_BUCKETS=1 FPR_SIZE_EMPTY_CACHE=1 ENG_FREE_CACHE_GB=4 ENG_FAST_EACH=1 FPR_LONG_ANY_ROWS=1 FPR_STEP_LOG=60
export DS41H_BULK_EXPERTS=0 IMAGES=1 RESPONSES_STORE=1 MAXLEN=655360
export VLLM_MEDIA_URL_ALLOW_REDIRECTS=0
export PYTHONPATH=$NR/runtime
# Required with this release: keep every torch CUDA graph until the process exits.
export PHYS_KEEP_GRAPHS=1
mkdir -p "$DS41H_LOGS"
export PHYS_EC_LOG=$DS41H_LOGS/empty_cache_$(date -u +%Y%m%dT%H%M%SZ).jsonl
if [ "${NATIVE_PRIVATE:-0}" = 1 ]; then
  export PORT=8100 EXTRA="--served-model-name $SERVED_MODEL_NAME" FPR_WARM=1 FPR_WARM_URL=http://127.0.0.1:8100
  unset FPR_WARM_CACERT FPR_WARM_AUTH
else
  [ -f "${AUTH_CONFIG:-}" ] || { echo "run_native: REFUSED: AUTH_CONFIG '${AUTH_CONFIG:-}' (vLLM --config YAML with the API keys) not found"; exit 3; }
  for f in server.key server.crt ca.pem; do [ -f "$TLS_DIR/$f" ] || { echo "run_native: REFUSED: $TLS_DIR/$f missing"; exit 3; }; done
  export PORT=${PORT:-8000}
  export EXTRA="--served-model-name $SERVED_MODEL_NAME --host 0.0.0.0 --port $PORT --ssl-keyfile $TLS_DIR/server.key --ssl-certfile $TLS_DIR/server.crt --config $AUTH_CONFIG --tool-parser-plugin $NR/plugins/dsml_roundtrip.py --middleware adapter.edge_guard.EdgeGuard"
  export FPR_WARM=1 FPR_WARM_URL=https://127.0.0.1:$PORT FPR_WARM_CACERT=$TLS_DIR/ca.pem FPR_WARM_AUTH=$AUTH_CONFIG
fi
# shellcheck disable=SC1090
source "$FPR_CONFIG"
if [ "$NATIVE_FE" = cxx ]; then
  # vLLM's external-frontend seam: Python keeps the sockets and the handshake, the binary serves HTTP / TLS / keys.
  export VLLM_USE_RUST_FRONTEND=1 VLLM_RUST_FRONTEND_PATH=$FRONTEND_BIN
  export FE_TLS_HANDSHAKE_TIMEOUT_S=${FE_TLS_HANDSHAKE_TIMEOUT_S:-60} FE_WEBP_LIB_DIR
  [ -n "${NATIVE_FE_EXTRA:-}" ] && export EXTRA="$EXTRA $NATIVE_FE_EXTRA"
fi
echo "run_native: native stack for H200 (port $PORT), config $FPR_CONFIG, switches $ENVF, $SEATS seats, $(date -u +%FT%TZ)"
echo "run_native: root $NR, frontend $NATIVE_FE"
exec bash scripts/serve_fpr.sh
