# Environment for vLLM 0.32.0 / DeepSeek-V4.1-Flash-1017 on 4x H200 NVL (sm_90a).
# Usage: source scripts/env.sh
#   DS41H_HOME   venv, caches, FlashMLA checkout, build output, default model dir (default ~/.ds41h)

DS41H_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DS41H_REPO
export DS41H_HOME="${DS41H_HOME:-$HOME/.ds41h}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-$DS41H_HOME/venv}"
export VIRTUAL_ENV_PROMPT=ds41h

# --- CUDA 13.0 toolkit from pip wheels (nvcc/crt/nvvm 13.0.88, cccl 13.0.85, cudart 13.0.96)
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export CUDA_ROOT="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
case ":$PATH:" in *":$CUDA_HOME/bin:"*) ;; *) export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$DS41H_HOME/bin:$PATH" ;; esac
# libnvrtc-builtins.so.13.0 (Humming NVRTC), libcudart, libnvJitLink, ... live here
case ":${LD_LIBRARY_PATH:-}:" in *":$CUDA_HOME/lib:"*) ;; *) export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;; esac
# link-time search path (pip CUDA has no lib64/ or libcudart.so; see setup/apply-toolchain-fixes.sh)
case ":${LIBRARY_PATH:-}:" in *":$CUDA_HOME/lib:"*) ;; *) export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" ;; esac
export TORCH_CUDA_ARCH_LIST="9.0a"

# --- runtime patch (runtime/sitecustomize.py: MAP_NORESERVE for safetensors, ds41h import hook when DS41H=1)
case ":${PYTHONPATH:-}:" in *":$DS41H_REPO/runtime:"*) ;; *) export PYTHONPATH="$DS41H_REPO/runtime${PYTHONPATH:+:$PYTHONPATH}" ;; esac

# --- caches
export XDG_CACHE_HOME="$DS41H_HOME/cache/xdg"
export UV_CACHE_DIR="$DS41H_HOME/cache/uv"
export UV_PYTHON_INSTALL_DIR="$DS41H_HOME/python"
export UV_LINK_MODE=copy
export PIP_CACHE_DIR="$DS41H_HOME/cache/pip"
export VLLM_CACHE_ROOT="$DS41H_HOME/cache/vllm"
export DG_JIT_CACHE_DIR="$DS41H_HOME/cache/deepgemm"
export TRITON_CACHE_DIR="$DS41H_HOME/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$DS41H_HOME/cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$DS41H_HOME/cache/torch_extensions"
export CUDA_CACHE_PATH="$DS41H_HOME/cache/cuda"
export CUDA_CACHE_MAXSIZE=4294967296
export FLASHINFER_WORKSPACE_BASE="$DS41H_HOME/cache/flashinfer"   # JIT cache -> $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer
export TILELANG_CACHE_DIR="$DS41H_HOME/cache/tilelang"
export HUMMING_CACHE_DIR="$DS41H_HOME/cache/humming/cache"
export HUMMING_TMP_DIR="$DS41H_HOME/cache/humming/tmp"
export QUACK_CACHE_DIR="$DS41H_HOME/cache/quack"
export HF_HOME="${HF_HOME:-$DS41H_HOME/cache/hf}"
export TMPDIR="$DS41H_HOME/tmp"

# --- build parallelism (kernel builds need several GB of RAM per translation unit)
export MAX_JOBS="${MAX_JOBS:-6}"
export NVCC_THREADS=1
export HUMMING_MAX_PARALLEL_BUILD_WORKERS="${HUMMING_MAX_PARALLEL_BUILD_WORKERS:-6}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

# --- serving defaults (override before sourcing if needed)
export MODEL_PATH="${MODEL_PATH:-$DS41H_HOME/model}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$DG_JIT_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$FLASHINFER_WORKSPACE_BASE" \
  "$TILELANG_CACHE_DIR" "$HUMMING_CACHE_DIR" "$HUMMING_TMP_DIR" "$QUACK_CACHE_DIR" "$HF_HOME" "$TMPDIR" 2>/dev/null || true
