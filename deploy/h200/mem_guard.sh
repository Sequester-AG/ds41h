#!/bin/bash
# mem_guard.sh: runtime memory / fault guard for a freshly started native endpoint (started by switch_native.sh with
# the job). Every MG_INT (2) s, until MG_MIN (60) minutes after "warm-up done" (MG_MAX_MIN (120) minutes at most):
#   - memory.free of any GPU < MG_FREE_MIB (2048) once the warm-up is done (the warm-up's own dip is normal)  -> rollback
#   - the endpoint log: OOM / illegal memory access / Xid / EngineDead / warm-up FAILED / import errors       -> rollback
#   - the kernel log since the start: any "NVRM: Xid" line                                                     -> rollback
#   - the job's process group gone                                                                             -> rollback
# Rollback = stop the watchdog first (it would race the rollback), then rollback_native.sh (or MG_ACTION_CMD).
# Logs a status line every minute with the minimum free MiB per GPU. $DS41H_RUN/NO_FALLBACK pauses the actions;
# touch $DS41H_RUN/ABORT_NATIVE to make the guard roll back once. MG_DRY=1: log only.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SITE_ENV:-$HERE/site.env}"
RUN=$DS41H_RUN; ST=$RUN/state; LOG=${MG_LOG:-$DS41H_LOGS/mem_guard.log}; PIDF=$RUN/mem_guard.pid
ACT=${MG_ACTION_CMD:-bash $HERE/rollback_native.sh}
INT=${MG_INT:-2}; MIN=${MG_MIN:-60}; MAXMIN=${MG_MAX_MIN:-120}; FREE_MIB=${MG_FREE_MIB:-2048}
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && [ "$(cat "$PIDF")" != $$ ]; then echo "mem guard already running"; exit 0; fi
echo $$ > "$PIDF"
JOB=$(sed -n 's/^pgid=//p' "$ST"); ELOG=$(sed -n 's/^log=//p' "$ST")
[ -n "$JOB" ] || { log "mem guard: no job in $ST"; rm -f "$PIDF"; exit 0; }
T0=$(date "+%Y-%m-%d %H:%M:%S"); t0=$(date +%s); off=0; tw=""
declare -a mn; for g in 0 1 2 3; do mn[$g]=999999; done
log "mem guard up (pid $$): job $JOB, log $ELOG, every $INT s, free < $FREE_MIB MiB or a fault -> rollback; $MIN min after warm-up${MG_DRY:+ (DRY)}"
mins() { local s=""; for g in 0 1 2 3; do s+="${mn[$g]} "; done; echo "${s% }"; }
rollback() {
  log "ROLLBACK: $1 (min free MiB per GPU $(mins))"
  [ -n "${MG_DRY:-}" ] && { log "DRY: would roll back"; rm -f "$PIDF"; exit 0; }
  w=$(cat "$RUN/watchdog.pid" 2>/dev/null)
  [ -n "$w" ] && ps -o args= -p "$w" 2>/dev/null | grep -q watchdog.sh && kill "$w" && log "watchdog $w stopped"
  rm -f "$PIDF"
  $ACT >> "$LOG" 2>&1; log "rollback rc $?"
  exit 1
}
last_min=$t0
while sleep "$INT"; do
  now=$(date +%s)
  [ "$(sed -n 's/^pgid=//p' "$ST")" = "$JOB" ] || { log "mem guard: state names another job: leaving; min free MiB per GPU $(mins)"; rm -f "$PIDF"; exit 0; }
  why=""; i=0
  while read -r f; do
    f=${f// /}; [ -n "$f" ] || continue
    [ "$f" -lt "${mn[$i]}" ] && mn[$i]=$f
    [ -n "$tw" ] && [ "$f" -lt "$FREE_MIB" ] && why="GPU $i free $f MiB < $FREE_MIB MiB"
    i=$((i + 1))
  done < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
  pgrep -g "$JOB" >/dev/null || why=${why:-"job $JOB gone"}
  [ -e "$RUN/ABORT_NATIVE" ] && { why=${why:-"manual abort ($RUN/ABORT_NATIVE)"}; rm -f "$RUN/ABORT_NATIVE"; }
  sz=$(stat -c %s "$ELOG" 2>/dev/null || echo 0)
  if [ -z "$why" ] && [ "$sz" -gt "$off" ]; then
    chunk=$(tail -c +$((off + 1)) "$ELOG" 2>/dev/null | head -c $((sz - off)))
    e=$(echo "$chunk" | grep -a -m1 -E "illegal memory access|out of memory|OutOfMemoryError|EngineDeadError|NVRM: Xid|CUDA error|warm-up FAILED|ModuleNotFoundError|ImportError|Engine core initialization failed|EngineCore failed to start" | cut -c1-200)
    [ -n "$e" ] && why="endpoint log: $e"
    [ -z "$tw" ] && echo "$chunk" | grep -aq "warm-up done" && { tw=$now; log "warm-up done seen: $MIN min from now"; }
    off=$sz
  fi
  if [ -z "$why" ]; then
    x=$(journalctl -k --since "$T0" --no-pager 2>/dev/null | grep -m1 "NVRM: Xid" | cut -c1-200)
    [ -n "$x" ] && why="kernel: $x"
  fi
  if [ -n "$why" ]; then
    if [ -e "$RUN/NO_FALLBACK" ]; then log "NO_FALLBACK set: not acting on: $why"; else rollback "$why"; fi
  fi
  if [ $((now - last_min)) -ge 60 ]; then
    log "status: min free MiB per GPU $(mins)${tw:+, $(( (now - tw) / 60 )) / $MIN min after warm-up}"; last_min=$now
  fi
  if [ -n "$tw" ] && [ $((now - tw)) -ge $((MIN * 60)) ]; then log "mem guard done: $MIN min clean; min free MiB per GPU $(mins)"; rm -f "$PIDF"; exit 0; fi
  if [ $((now - t0)) -ge $((MAXMIN * 60)) ]; then log "mem guard done: $MAXMIN min cap; min free MiB per GPU $(mins)"; rm -f "$PIDF"; exit 0; fi
done
