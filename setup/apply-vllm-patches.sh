#!/usr/bin/env bash
# Apply patches/vllm-*.patch to the installed vLLM 0.32.0 wheel. Idempotent; re-run after any reinstall of vllm.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SITE=${DS41H_HOME:-$HOME/.ds41h}/venv/lib/python3.12/site-packages
for p in "$HERE"/../patches/vllm-*.patch; do
  if (cd "$SITE" && patch --dry-run -R -s -f -p0 < "$p" >/dev/null 2>&1); then
    echo "$(basename "$p"): already applied"
  else
    (cd "$SITE" && patch --no-backup-if-mismatch -p0 < "$p")
  fi
done
