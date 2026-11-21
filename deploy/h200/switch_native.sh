#!/bin/bash
# switch_native.sh [NATIVE_ENV]: switch the public endpoint to the native stack described by NATIVE_ENV (default
# deploy/h200/native.env), with an automatic rollback if the new stack does not come up.
#   1. checks: the switches file, the release root, the native modules, the optional SWITCH_BLOCK_UTC window
#   2. pause: $DS41H_RUN/SUPERVISOR_PAUSE (an external supervisor should not restart anything meanwhile); the
#      running watchdog and memory guard are stopped
#   3. the running endpoint (the process group in $DS41H_RUN/state) is stopped; wait until the GPUs are free
#   4. NATIVE_ENV is installed as $DS41H_RUN/native.env (the previous one is kept as native.env.prev) and
#      run_native.sh starts in its own session; the memory guard starts with it
#   5. wait for "warm-up done" and /health 200, then start the watchdog and remove the pause
# If step 5 fails and ROLLBACK=1 (default), the previous native.env is restarted once (rollback_native.sh).
#   NO_ROLLBACK=1   leave a failed start as it is (debugging)
#   FORCE=1         ignore SWITCH_BLOCK_UTC
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SITE_ENV:-$HERE/site.env}"
TE=$(cd "$(dirname "${1:-$HERE/native.env}")" && pwd)/$(basename "${1:-$HERE/native.env}")
RUN=$DS41H_RUN; ST=$RUN/state; mkdir -p "$RUN" "$DS41H_LOGS"
say() { echo "$(date -u +%T) switch_native: $*"; }
health() { curl -s -o /dev/null -w '%{http_code}' --max-time 8 --cacert "$TLS_DIR/ca.pem" "https://127.0.0.1:$PORT/health"; }

# 1. checks
[ -f "$TE" ] || { say "REFUSED: no switches file $TE"; exit 1; }
( set +u; source "$TE"; [ "${NATIVE_READY:-0}" = 1 ] && [ -d "${NATIVE_ROOT:-$DS41H_REPO}/fpr" ] && [ -f "$FPR_CONFIG" ] ) \
  || { say "REFUSED: $TE: NATIVE_READY != 1, or its NATIVE_ROOT / FPR_CONFIG is missing"; exit 1; }
ls "$DS41H_BUILD"/b2a/fpr_core2b*.so "$DS41H_BUILD"/b2a/fpr_sched_py*.so "$DS41H_BUILD"/b2a/fpr_kvtables*.so >/dev/null 2>&1 \
  || { say "REFUSED: native modules missing in $DS41H_BUILD/b2a (deploy/h200/build_native.sh)"; exit 1; }
if [ -n "${SWITCH_BLOCK_UTC:-}" ] && [ "${FORCE:-0}" != 1 ] && [ "${IS_ROLLBACK:-0}" != 1 ]; then
  hm=$(date -u +%H%M); a=${SWITCH_BLOCK_UTC%-*}; b=${SWITCH_BLOCK_UTC#*-}
  if [ "$hm" -ge "$a" ] && [ "$hm" -lt "$b" ]; then say "REFUSED: $(date -u +%H:%M) UTC is inside SWITCH_BLOCK_UTC $SWITCH_BLOCK_UTC"; exit 2; fi
fi

# 2. pause, stop the guards
touch "$RUN/SUPERVISOR_PAUSE"
for pf in "$RUN/watchdog.pid" "$RUN/mem_guard.pid"; do
  w=$(cat "$pf" 2>/dev/null); [ -n "$w" ] && ps -o args= -p "$w" 2>/dev/null | grep -q -E "watchdog\.sh|mem_guard\.sh" && kill "$w" && say "stopped $w ($(basename "$pf"))"
done

# 3. stop the running endpoint
t0=$(date +%s)
old=$(sed -n 's/^pgid=//p' "$ST" 2>/dev/null)
if [ -n "$old" ] && pgrep -g "$old" >/dev/null 2>&1; then
  say "stopping the running endpoint (process group $old)"
  kill -TERM -- "-$old" 2>/dev/null
  for i in $(seq 60); do pgrep -g "$old" >/dev/null || break; sleep 1; done
  pgrep -g "$old" >/dev/null && { say "still alive after 60 s: SIGKILL"; kill -KILL -- "-$old"; sleep 3; }
fi
for i in $(seq 90); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] && break; sleep 1; done
[ "$n" = 0 ] || { say "GPUs not free ($n compute processes): not starting"; exit 1; }
say "stopped ($(( $(date +%s) - t0 )) s)"

# 4. install the switches, start
[ -f "$RUN/native.env" ] && [ "${IS_ROLLBACK:-0}" != 1 ] && cp "$RUN/native.env" "$RUN/native.env.prev"
cp "$TE" "$RUN/native.env.new" && mv "$RUN/native.env.new" "$RUN/native.env"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$DS41H_LOGS/endpoint_${TS}_native.log
setsid nohup bash "$HERE/run_native.sh" > "$LOG" 2>&1 < /dev/null &
pid=$!
printf "pgid=%s\nlog=%s\nenv=%s\nstarted=%s\n" "$pid" "$LOG" "$TE" "$TS" > "$ST.tmp" && mv "$ST.tmp" "$ST"
say "native stack started: pid $pid, log $LOG"
setsid nohup bash "$HERE/mem_guard.sh" > /dev/null 2>&1 < /dev/null &

# 5. warm-up, health, watchdog
ok=0
for i in $(seq 1200); do
  grep -qaE "warm-up (done|FAILED)|Traceback|REFUSED" "$LOG" && break
  kill -0 "$pid" 2>/dev/null || break
  sleep 2
done
if grep -qa "warm-up done" "$LOG"; then
  for i in $(seq 30); do [ "$(health)" = 200 ] && { ok=1; break; }; sleep 5; done
fi
say "window $(( $(date +%s) - t0 )) s"
if [ $ok = 1 ]; then
  setsid nohup bash "$HERE/watchdog.sh" > /dev/null 2>&1 < /dev/null &
  rm -f "$RUN/SUPERVISOR_PAUSE"
  say "LIVE: $(basename "$TE"); watchdog started, pause removed"
  exit 0
fi
say "FAILED: no warm-up / health 200; last log lines:"; tail -20 "$LOG" | cut -c1-200
if [ "${NO_ROLLBACK:-0}" != 1 ] && [ "${IS_ROLLBACK:-0}" != 1 ] && [ -f "$RUN/native.env.prev" ]; then
  say "rolling back to the previous switches"
  exec bash "$HERE/rollback_native.sh"
fi
exit 1
