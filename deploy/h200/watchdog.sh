#!/bin/bash
# watchdog.sh: auto-rollback for the running native endpoint (started by switch_native.sh after the warm-up).
# Every WD_INTERVAL (30) s while $DS41H_RUN/state names the job it started with, a check fails if:
#   the job's process group is gone; /health does not answer 200 within 10 s; the endpoint log has a new
#   EngineDeadError / "illegal memory access" / "CUDA error" line; the kernel log has a new NVRM Xid 31 for an EngineCore;
#   or the vLLM stat logger shows work pending (Running / Waiting > 0) without prompt or generation progress for
#   WD_STALL_S (300) s (a hang that /health does not see; WD_STALL_S=0 turns the rule off).
# After WD_FAILS (2) failing checks in a row: rollback_native.sh (or WD_FALLBACK_CMD), then exit.
# Leaves when the state file names another job. $DS41H_RUN/NO_FALLBACK pauses it. One instance (pid file).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SITE_ENV:-$HERE/site.env}"
RUN=$DS41H_RUN; ST=$RUN/state; LOG=${WD_LOG:-$DS41H_LOGS/watchdog.log}; PIDF=$RUN/watchdog.pid
URL=${WD_URL:-https://127.0.0.1:$PORT/health}
FB=${WD_FALLBACK_CMD:-bash $HERE/rollback_native.sh}
INT=${WD_INTERVAL:-30}; NFAIL=${WD_FAILS:-2}; STALL_S=${WD_STALL_S:-300}
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
if [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && [ "$(cat "$PIDF")" != $$ ]; then echo "watchdog already running"; exit 0; fi
echo $$ > "$PIDF"
JOB=$(sed -n 's/^pgid=//p' "$ST"); ELOG=$(sed -n 's/^log=//p' "$ST")
[ -n "$JOB" ] || { log "watchdog: no job in $ST"; exit 0; }
T0=$(date "+%Y-%m-%d %H:%M:%S"); T0S=$(date +%s); N0=$(wc -l < "$ELOG" 2>/dev/null || echo 0)
log "watchdog up (pid $$): job $JOB, log $ELOG, $URL every $INT s, fallback after $NFAIL failures, stall rule $STALL_S s"

stall() {   # prints a reason when the stat logger shows pending work without progress for >= STALL_S seconds
  local Y r ok last lt run wt to age
  Y=$(date +%Y)
  r=$(tail -n 20000 "$ELOG" 2>/dev/null | grep -a "Engine 000:" | tail -n 400 |
      sed -nE 's/.*INFO ([0-9]{2})-([0-9]{2}) ([0-9:]{8}) .*Avg prompt throughput: ([0-9.]+) tokens\/s, Avg generation throughput: ([0-9.]+) tokens\/s, Running: ([0-9]+) reqs, Waiting: ([0-9]+) reqs.*/\1-\2 \3 \4 \5 \6 \7/p' |
      awk '{ last = $0; if ($3 + 0 > 0 || $4 + 0 > 0 || ($5 + $6) == 0) ok = $1 " " $2 } END { print ok "|" last }')
  ok=${r%%|*}; last=${r#*|}
  [ -n "$last" ] || return 0
  set -- $last; lt="$1 $2"; run=$5; wt=$6
  [ $((run + wt)) -gt 0 ] || return 0
  if [ -n "$ok" ]; then to=$(date -d "$Y-$ok" +%s 2>/dev/null || echo "$T0S"); else to=$T0S; fi
  age=$(( $(date +%s) - to ))
  [ "$age" -ge "$STALL_S" ] && echo "STALL: no progress for $age s with work pending (newest logger line $lt: Running $run, Waiting $wt)"
  return 0
}

fails=0
while sleep "$INT"; do
  [ -e "$RUN/NO_FALLBACK" ] && { fails=0; continue; }
  [ "$(sed -n 's/^pgid=//p' "$ST")" = "$JOB" ] || { log "watchdog: state names another job: leaving"; break; }
  why=""
  pgrep -g "$JOB" >/dev/null || why="job $JOB gone"
  if [ -z "$why" ]; then
    c=$(curl -s -o /dev/null -m 10 -w '%{http_code}' --cacert "$TLS_DIR/ca.pem" "$URL" 2>/dev/null)
    [ "$c" = 200 ] || why="health $URL -> ${c:-no answer}"
  fi
  if [ -z "$why" ]; then
    e=$(tail -n +$((N0 + 1)) "$ELOG" 2>/dev/null | grep -a -m1 -E "EngineDeadError|illegal memory access|CUDA error" | cut -c1-160)
    [ -n "$e" ] && why="endpoint log: $e"
  fi
  if [ -z "$why" ] && [ "$STALL_S" -gt 0 ]; then why=$(stall); fi
  if [ -z "$why" ]; then
    ep=$(pgrep -f "VLLM::EngineCore" | tr '\n' '|'); ep=${ep%|}
    if [ -n "$ep" ]; then
      x=$(journalctl -k --since "$T0" --no-pager 2>/dev/null | grep -E "NVRM: Xid .*: 31, pid=($ep)," | head -1 | cut -c1-200)
      [ -n "$x" ] && why="kernel: $x"
    fi
  fi
  if [ -z "$why" ]; then fails=0; continue; fi
  fails=$((fails + 1)); log "check failed ($fails / $NFAIL): $why"
  [ $fails -lt $NFAIL ] && continue
  log "FALLBACK: $FB (reason: $why)"
  rm -f "$PIDF"
  $FB >> "$LOG" 2>&1; log "fallback rc=$?"
  exit 0
done
rm -f "$PIDF"
log "watchdog exits"
