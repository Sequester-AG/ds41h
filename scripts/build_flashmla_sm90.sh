#!/bin/bash
# Build the ds41h FlashMLA small-head sparse-prefill extension (FlashMLA e2afa5f = vllm-project/FlashMLA
# PR #26, SM90 sparse prefill only, d_qk = 512) for sm_90a. No GPU needed. Output goes to
# $DS41H_FLASHMLA_BUILD (default $DS41H_HOME/build/flashmla_sm90), never into site-packages; vLLM's
# own vllm/_flashmla_C is left untouched. Used by DS41H_PREFILL_ATTN=1 (ds41h/prefill_attn.py).
#   DS41H_FLASHMLA_SRC   FlashMLA checkout (default $DS41H_HOME/src/FlashMLA, must contain PR #26)
#   DS41H_PTXAS_VERBOSE=1  print ptxas register / smem / spill stats
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/scripts/env.sh"
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
export MAX_JOBS=${MAX_JOBS:-2}        # cute/cutlass TUs need several GB of RAM each (keep it low on small hosts)
export DS41H_FLASHMLA_SRC=${DS41H_FLASHMLA_SRC:-$DS41H_HOME/src/FlashMLA}
export DS41H_FLASHMLA_BUILD=${DS41H_FLASHMLA_BUILD:-$DS41H_HOME/build/flashmla_sm90}
git -C "$DS41H_FLASHMLA_SRC" log --oneline -1 || true
mkdir -p "$DS41H_FLASHMLA_BUILD"
python - <<'PY'
import time
t = time.time()
from ds41h import prefill_attn
m = prefill_attn.build(verbose=True)
print(f"built {m.__file__} in {time.time() - t:.0f}s")
PY
