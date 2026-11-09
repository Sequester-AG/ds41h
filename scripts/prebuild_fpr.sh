#!/bin/bash
# FPRv1 prebuild on the CPU (no GPU is touched): the CUDA extensions the engine loads, and the converted-weight cache
# (fpr/tools/wcache.py, under $DS41H_HOME/cache/wcache unless PHYS_WCACHE is set). Without it the first server start
# does this work on the GPUs.
#
#   MODEL=<dir> scripts/prebuild_fpr.sh          extensions, then the weight cache
#   PARTS=ext MODEL=<dir> ...                    extensions only (writes fpr/build/prebuilt.env; serve_fpr.sh reads it)
#   PARTS=weights MODEL=<dir> ...                the weight cache only
#   JOBS=<n>                                     weight-cache worker processes (default 4, one quarter of the layers each)
#   FPR_CONFIG=<file>                            the serving configuration whose flags decide what is built
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh"
CFG=${FPR_CONFIG:-$REPO/fpr/config/fprv1.env}
# shellcheck disable=SC1090
source "$CFG"
PHYS=${PHYS_ROOT:-$REPO/fpr}   # PHYS_ROOT: the engine tree, if it is not this repository's fpr/
MODEL=${MODEL:-${DS41H_MODEL_DIR:-$DS41H_HOME/model}}
[ -d "$MODEL" ] || { echo "prebuild_fpr: no checkpoint in $MODEL (set MODEL)" >&2; exit 2; }
export DS41H_MODEL_DIR=$MODEL PHYS_ROOT=$PHYS CUDA_VISIBLE_DEVICES=
export PYTHONPATH=$PHYS:$REPO${PYTHONPATH:+:$PYTHONPATH}
export PHYS_ENGINE="$FPR_ENGINE"
# shellcheck disable=SC2086
[ -n "${FPR_FLAGS:-}" ] && export $FPR_FLAGS
PARTS=${PARTS:-ext,weights}
cd "$PHYS"
if [[ ",$PARTS," == *,ext,* ]]; then
  echo "prebuild_fpr: CUDA extensions"
  python engine/prebuild_ext.py --async
fi
if [[ ",$PARTS," == *,weights,* ]]; then
  JOBS=${JOBS:-4}
  echo "prebuild_fpr: converted-weight cache, $JOBS workers"
  pids=()
  for ((p = 0; p < JOBS; p++)); do
    python engine/prebuild.py --part "$p" --nparts "$JOBS" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  python engine/prebuild.py --idx
fi
echo "prebuild_fpr: done"
