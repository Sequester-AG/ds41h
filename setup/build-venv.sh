#!/usr/bin/env bash
# Build the serving environment under $DS41H_HOME (default ~/.ds41h): CPU only, no GPU use.
#   Needs: uv (https://docs.astral.sh/uv/) on PATH, git, patch, a C++ toolchain. ~25 GB of disk.
#   Result: $DS41H_HOME/venv with vLLM 0.32.0, the CUDA 13 pip toolchain, the toolchain fixes and the vLLM patches of patches/,
#   and $DS41H_HOME/src/FlashMLA at the pinned commit.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DS41H_HOME="${DS41H_HOME:-$HOME/.ds41h}"
export UV_CACHE_DIR=$DS41H_HOME/cache/uv UV_PYTHON_INSTALL_DIR=$DS41H_HOME/python UV_LINK_MODE=copy
VENV=$DS41H_HOME/venv
FLASHMLA_COMMIT=e2afa5fd8c220fbd05fe03fc24c9d5e64c24641b
FLASHINFER_INDEX=https://flashinfer.ai/whl
mkdir -p "$DS41H_HOME"

uv python install 3.12.15
uv venv --python 3.12.15 --prompt ds41h "$VENV"
uv pip install --python "$VENV/bin/python" -r "$HERE/requirements-serving.txt" --override "$HERE/overrides-cuda13.txt"
# FlashInfer AOT wheels. --no-deps: the jit-cache meta wheel requires providers for other GPU architectures, not needed on sm90.
uv pip install --python "$VENV/bin/python" --no-deps --index-url "$FLASHINFER_INDEX" "flashinfer-cubin==0.7.0.post1"
uv pip install --python "$VENV/bin/python" --no-deps --index-url "$FLASHINFER_INDEX/cu130" \
  "flashinfer-jit-cache==0.7.0.post1+cu130" "flashinfer-jit-cache-sm90a==0.7.0.post1+cu130"
uv pip install --python "$VENV/bin/python" -r "$HERE/requirements-devtools.txt" --override "$HERE/overrides-cuda13.txt"
bash "$HERE/apply-toolchain-fixes.sh"
bash "$HERE/apply-vllm-patches.sh"

if [ ! -d "$DS41H_HOME/src/FlashMLA/.git" ]; then
  mkdir -p "$DS41H_HOME/src"
  git clone https://github.com/vllm-project/FlashMLA.git "$DS41H_HOME/src/FlashMLA"
fi
git -C "$DS41H_HOME/src/FlashMLA" checkout --detach "$FLASHMLA_COMMIT"
git -C "$DS41H_HOME/src/FlashMLA" submodule update --init --recursive
echo "environment ready: source scripts/env.sh"
