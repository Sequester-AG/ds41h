#!/bin/bash
# Exactness run: serve (scripts/serve.sh) with
# the teacher-forcing hook (ds41h/patch_force.py) -> exact_eval gen / apilp / collect -> GSM8K -> stop the server.
#
#   scripts/exact_run.sh <tag> [VAR=VAL ...]        VAR=VAL pairs go to serve.sh
#
# Knobs (environment of this script):
#   STOCK=1          stock numerics: DS41H=1 with every compute component off (only the force hook is active)
#   SRC=dir          ds41h tree to serve from (default this repository)
#   EXT=dir          torch extension cache for SRC (default $DS41H_HOME/cache/torch_ext_exact_<basename SRC>)
#   SET=file         prompt set (default $DS41H_HOME/runs/exact/set_v1.json); GEN=1 builds it on this server if missing
#   APILP=1          normal-path greedy top-20 logprobs of the set (hook transparency check, exact_eval.py hookcheck)
#   FORCE=1          teacher-forced collection -> <out>/dump/<tag>/rank*.pt; FORCE2=1 repeats it into <out>/dump2 (determinism)
#   GSM=1 GSM_N=500 GSM_C=1 GSM_MAXTOK=512   GSM8K (scripts/gsm8k_eval.py) -> <out>/gsm8k.json
#   GPU_TIMEOUT=3600 READY_TIMEOUT=2400 PREBUILD=1 PREBUILD_ONLY=1 (CPU prebuild + <out>/args.txt, no GPU)
# Results: $DS41H_HOME/runs/exact/<tag>/{server.log,env.txt,progress.log,collect.log,gsm8k.json,gsm8k.log,apilp.json,dump/}
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${DS41H_HOME:-$HOME/.ds41h}
export DS41H_HOME=$ROOT
STOCK_FLAGS=(DS41H=1 DS41H_ALLREDUCE=0 DS41H_CANDIDATES=0 DS41H_ATTN=0 DS41H_PROLOGUE=0 DS41H_DECODER=0 DS41H_SAMPLING=0
             DS41H_PREFILL=0 DS41H_DRAFTER=0 DS41H_MOE=0)

if [ "${1:-}" = "--inner" ]; then
  shift
  TAG=$1; shift
  OUT=$ROOT/runs/exact/$TAG
  source "$REPO/scripts/env.sh"
  export PYTHONPATH=$SRC:$PYTHONPATH TORCH_EXTENSIONS_DIR=$EXT
  cd $SRC
  log() { echo "== $(date +%T) [$TAG] $*" | tee -a $OUT/progress.log; }
  SRV_PGID=
  stop_server() {
    [ -z "$SRV_PGID" ] && return
    log "stopping server (pgid $SRV_PGID)"
    kill -TERM -- -$SRV_PGID 2>/dev/null
    for _ in $(seq 1 45); do kill -0 -- -$SRV_PGID 2>/dev/null || break; sleep 2; done
    kill -KILL -- -$SRV_PGID 2>/dev/null
    sleep 3
    SRV_PGID=
  }
  trap 'stop_server' EXIT
  trap 'log "interrupted"; exit 130' INT TERM
  if curl -sf 127.0.0.1:8100/v1/models >/dev/null 2>&1; then
    log "port 8100 already serving - refusing"; echo failed_port > $OUT/status; exit 97
  fi
  rm -f $OUT/force.json
  log "start server: $*"
  setsid env "$@" DS41H_FORCE_FILE=$OUT/force.json DS41H_FORCE_DUMP=$OUT/dump DS41H_SRC=$SRC DS41H_TORCH_EXT_DIR=$EXT \
    bash $SRC/scripts/serve.sh > $OUT/server.log 2>&1 < /dev/null &
  SRV_PID=$!
  SRV_PGID=$SRV_PID
  t0=$(date +%s); ready=0
  while [ $(( $(date +%s) - t0 )) -lt ${READY_TIMEOUT:-2400} ]; do
    if curl -sf 127.0.0.1:8100/v1/models >/dev/null 2>&1; then ready=1; break; fi
    kill -0 $SRV_PID 2>/dev/null || break
    sleep 5
  done
  if [ $ready != 1 ]; then
    log "server FAILED to start ($(( $(date +%s) - t0 )) s)"; tail -n 60 $OUT/server.log; echo failed_start > $OUT/status; exit 1
  fi
  log "server ready after $(( $(date +%s) - t0 )) s"
  grep -a "ds41h" $OUT/server.log | grep -a -E "force hook installed|hook installed|flags on" | head -8 | tee -a $OUT/progress.log
  PY="env DS41H=0 python"
  SET=${SET:-$ROOT/runs/exact/set_v1.json}
  if [ ! -f "$SET" ] && [ "${GEN:-0}" = "1" ]; then
    timeout 1800 $PY $SRC/scripts/exact_eval.py gen $SET > $OUT/gen.log 2>&1
    log "gen rc=$?: $(tail -n 1 $OUT/gen.log)"
  fi
  if [ "${APILP:-0}" = "1" ] && [ -f "$SET" ]; then
    timeout 1800 $PY $SRC/scripts/exact_eval.py apilp $SET $OUT/apilp.json > $OUT/apilp.log 2>&1
    log "apilp rc=$?"
  fi
  if [ "${FORCE:-1}" = "1" ] && [ -f "$SET" ]; then
    timeout 2400 $PY $SRC/scripts/exact_eval.py collect $SET --force-file $OUT/force.json --dump $OUT/dump > $OUT/collect.log 2>&1
    log "collect rc=$?:"; cat $OUT/collect.log | tee -a $OUT/progress.log
    if [ "${FORCE2:-0}" = "1" ]; then  # second forced collection on the same server: run-to-run determinism
      timeout 2400 $PY $SRC/scripts/exact_eval.py collect $SET --force-file $OUT/force.json --dump $OUT/dump2 > $OUT/collect2.log 2>&1
      log "collect2 rc=$?: $(tail -n 1 $OUT/collect2.log)"
      $PY $SRC/scripts/exact_eval.py compare $OUT/dump $OUT/dump2 --set $SET --json $OUT/repeat.json > $OUT/repeat.txt 2>&1
      log "repeat (same server): $(grep -E '^all ' $OUT/repeat.txt)"
    fi
    if [ "${APILP:-0}" = "1" ] && [ -f $OUT/apilp.json ]; then
      $PY $SRC/scripts/exact_eval.py hookcheck $SET $OUT/apilp.json $OUT/dump > $OUT/hookcheck.log 2>&1
      log "hookcheck: $(tail -n 1 $OUT/hookcheck.log)"
    fi
  fi
  if ! kill -0 $SRV_PID 2>/dev/null; then log "server died"; tail -n 60 $OUT/server.log; echo died > $OUT/status; exit 1; fi
  if [ "${GSM:-1}" = "1" ]; then
    timeout 5400 $PY $SRC/scripts/gsm8k_eval.py run $OUT/gsm8k.json --n ${GSM_N:-500} --concurrency ${GSM_C:-1} \
      --max-tokens ${GSM_MAXTOK:-512} > $OUT/gsm8k.log 2>&1
    log "gsm8k rc=$?: $(tail -n 1 $OUT/gsm8k.log)"
  fi
  curl -s 127.0.0.1:8100/metrics > $OUT/metrics_end.prom
  grep -a "ds41h" $OUT/server.log > $OUT/ds41h.log 2>/dev/null
  if ! kill -0 $SRV_PID 2>/dev/null; then log "server died during the run"; echo died > $OUT/status; exit 1; fi
  echo ok > $OUT/status
  log "done"
  exit 0
fi

# ------------------------------------------------------------------------------------------------ outer
TAG=${1:?usage: exact_run.sh <tag> [VAR=VAL ...]}
shift
export SRC=${SRC:-$REPO}
export EXT=${EXT:-$ROOT/cache/torch_ext_exact_$(basename $SRC)}
OUT=$ROOT/runs/exact/$TAG
mkdir -p $OUT $EXT
rm -f $OUT/status
ARGS=("$@")
[ "${STOCK:-0}" = "1" ] && ARGS=("${STOCK_FLAGS[@]}" "$@")
{ echo "tag=$TAG"; echo "args=${ARGS[*]}"; echo "src=$SRC ext=$EXT"; env | grep -E '^(SET|GEN|APILP|FORCE|GSM|STOCK)' | sort
  (cd $SRC && sha1sum ds41h/kernels/*.cu ds41h/*.py scripts/*.py | sed 's/^/src /'); } > $OUT/env.txt
if [ "${PREBUILD:-1}" = "1" ] && printf '%s\n' "${ARGS[@]}" | grep -q '^DS41H=1$'; then
  echo "== $(date +%T) [$TAG] prebuilding ds41h extensions (CPU only) into $EXT" | tee -a $OUT/progress.log
  (source "$REPO/scripts/env.sh"; export PYTHONPATH=$SRC:$PYTHONPATH TORCH_EXTENSIONS_DIR=$EXT; cd $SRC
   env "${ARGS[@]}" CUDA_VISIBLE_DEVICES='' timeout 3600 python - <<'EOF'
import importlib, importlib.util, time
from ds41h import flags
mods = []
if flags.DECODER: mods.append(("decoder", "extension"))
if flags.PROLOGUE: mods += [("prologue", "ext"), ("skinny", "ext")]
if flags.ATTN: mods.append(("attn", "ext"))
if flags.CANDIDATES: mods.append(("candidates", "ext"))
if flags.ALLREDUCE: mods.append(("allreduce", "ext"))
from ds41h import moe
if moe.ENABLED: mods.append(("moe", "ext"))
from ds41h import patch_drafter
if patch_drafter.ENABLED or flags.SAMPLING: mods.append(("drafter", "ext"))
if flags.PREFILL: mods += [("linear", "ext"), ("prefill_attn", "ext"), ("mhc_prefill", "ext")]
for mod, fn in mods:
    if importlib.util.find_spec(f"ds41h.{mod}") is None:
        continue
    t = time.time()
    try:
        getattr(importlib.import_module(f"ds41h.{mod}"), fn)()
        print(f"prebuild {mod}: ok ({time.time() - t:.0f} s)")
    except Exception as e:  # noqa: BLE001
        print(f"prebuild {mod}: FAILED {type(e).__name__}: {str(e)[:300]}")
EOF
  ) 2>&1 | grep -E "^prebuild|Error" | tee $OUT/prebuild.log | tee -a $OUT/progress.log
fi
if [ "${PREBUILD_ONLY:-0}" = "1" ]; then printf '%s\n' "${ARGS[@]}" > $OUT/args.txt; exit 0; fi
timeout ${GPU_TIMEOUT:-3600} bash $SRC/scripts/exact_run.sh --inner "$TAG" "${ARGS[@]}" 2>&1 | tee $OUT/run.log
rc=${PIPESTATUS[0]}
echo "== $(date +%T) [$TAG] exact_run rc=$rc status=$(cat $OUT/status 2>/dev/null || echo none)"
exit $rc
