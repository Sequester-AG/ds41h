#!/bin/bash
# rollback_native.sh [NATIVE_ENV]: restart the endpoint on the previous switches ($DS41H_RUN/native.env.prev, kept by
# switch_native.sh) or on the given file, e.g. the native.env of the previous release root. A safety action: it ignores
# SWITCH_BLOCK_UTC and never rolls back again if this start fails (the endpoint then stays down: check the log).
# Called by switch_native.sh on a failed start, and by watchdog.sh / mem_guard.sh on a fault.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SITE_ENV:-$HERE/site.env}"
TE=${1:-${ROLLBACK_ENV:-$DS41H_RUN/native.env.prev}}
[ -f "$TE" ] || { echo "$(date -u +%T) rollback_native: no rollback switches ($TE)"; exit 1; }
# copy it first: switch_native.sh replaces $DS41H_RUN/native.env.prev
cp "$TE" "$DS41H_RUN/native.env.rollback"
echo "$(date -u +%T) rollback_native: restarting on $TE"
IS_ROLLBACK=1 exec bash "$HERE/switch_native.sh" "$DS41H_RUN/native.env.rollback"
