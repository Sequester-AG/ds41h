#!/bin/bash
# CPU-only JIT build of the ds41h extensions, so the first server start does not spend ~10 min compiling.
#   usage: bash scripts/prebuild.sh VAR=VAL ...      (pass the list printed by `DRYRUN=1 scripts/serve_ds41h.sh`)
# Includes the grouped decode extension when DS41H_ATTN_GROUPED is set (into DS41H_GDEC_BUILD).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/scripts/env.sh"
export PYTHONPATH=$REPO:$PYTHONPATH MAX_JOBS=${MAX_JOBS_PB:-4}
cd "$REPO"
env "$@" CUDA_VISIBLE_DEVICES='' nice -n 15 bash -c '[ -n "${DS41H_TORCH_EXT_DIR:-}" ] && export TORCH_EXTENSIONS_DIR=$DS41H_TORCH_EXT_DIR; exec python -' <<'PY'
import importlib, importlib.util, os, time
from ds41h import flags
mods = [("decoder", "extension"), ("prologue", "ext"), ("skinny", "ext"), ("attn", "ext"), ("candidates", "ext"),
        ("allreduce", "ext"), ("moe", "ext"), ("drafter", "ext")]
if flags.PREFILL:
    mods += [("linear", "ext"), ("prefill_attn", "ext"), ("mhc_prefill", "ext")]
if os.environ.get("DS41H_ENGRAM_NVME", "0") not in ("0", "", "false", "False"):
    mods += [("engram_nvme", "ext")]
if os.environ.get("DS41H_ENGRAM_HOST", "0") not in ("0", "", "false", "False"):
    mods += [("engram_host", "ext")]  # Engram tables in pinned host RAM
if os.environ.get("DS41H_META_BATCH", "0") not in ("0", "") or os.environ.get("DS41H_META_DRAFT", "0") not in ("0", ""):
    mods += [("patch_meta", "ext")]
if os.environ.get("DS41H_VERIFY_HEAD", "0") not in ("0", ""):
    mods += [("vhead", "ext")]
mods += [("moe_dec", "ext")]
if os.environ.get("DS41H_ATTN_GROUPED", "0") not in ("0", "", "false", "False"):
    mods += [("grouped_decode", "ext")]  # FlashMLA-based grouped multi-row decode (build dir DS41H_GDEC_BUILD)
for mod, fn in mods:
    if importlib.util.find_spec(f"ds41h.{mod}") is None:
        continue
    t = time.time()
    try:
        getattr(importlib.import_module(f"ds41h.{mod}"), fn)()
        print(f"prebuild {mod}: ok ({time.time() - t:.0f} s)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"prebuild {mod}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
PY
