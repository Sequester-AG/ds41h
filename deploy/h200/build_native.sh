#!/bin/bash
# build_native.sh: CPU-only build of everything the native stack loads, on the serving host (x86_64, g++ >= 13).
# No GPU is touched; run it at low priority if another server is using the host.
#   STEPS=frontend,xgr,core,nd,ext,weights (default: all, in this order)
#   1. frontend   frontend_cxx's third-party libs (third_party/build.sh; put the source tarballs in third_party/dl/
#                 first: zeromq-4.3.5, onig-6.9.10, libjpeg-turbo-3.2.0 and simdjson 3.12.3's simdjson.h / .cpp), then the
#                 binary (CMake, Release, TLS on)                                  -> frontend_cxx/build/frontend_cxx
#   2. xgr        xgrammar 0.2.8 (the version the venv's wheel is) as a static library     -> $DS41H_BUILD/xgr
#   3. core       fpr_kvtables, fpr_sched_py, fpr_core_py and the engine-process module fpr_core2b -> $DS41H_BUILD/b2a
#   4. nd         native decode (fpr_nd, fpr_nd2; torch's extension loader, CPU)  -> packages/nd/fpr/core/src/nd/build
#   5. ext        the CUDA extensions of the serving configuration (engine/prebuild_ext.py + prebuild_native.py),
#                 next to the kernels (fpr/**/build*)
#   6. weights    the converted-weight cache ($PHYS_WCACHE)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SITE_ENV:-$HERE/site.env}"
R=$DS41H_REPO; NB=$DS41H_BUILD; J=${MAX_JOBS:-8}
CXX=${CXX:-g++}; PY=$VIRTUAL_ENV/bin/python
STEPS=${STEPS:-frontend,xgr,core,nd,ext,weights}
has() { [[ ",$STEPS," == *",$1,"* ]]; }
t0=$(date +%s); say() { echo "$(date +%T) [$(( $(date +%s) - t0 )) s] $*"; }
mkdir -p "$NB/b2a" "$NB/xgr/obj" "$DS41H_LOGS"
cd "$R" || exit 1
# shellcheck disable=SC1091
source scripts/env.sh >/dev/null 2>&1
fail() { say "FAILED: $*"; exit 1; }

if has frontend; then
  ( cd frontend_cxx && bash third_party/build.sh ) > "$DS41H_LOGS/build_fe_third_party.log" 2>&1 || fail "frontend third-party (log $DS41H_LOGS/build_fe_third_party.log)"
  ( cd frontend_cxx && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFE_WITH_TLS=ON && cmake --build build -j "$J" ) \
    > "$DS41H_LOGS/build_fe.log" 2>&1 || fail "frontend (log $DS41H_LOGS/build_fe.log)"
  say "1. frontend: $(ls -la frontend_cxx/build/frontend_cxx | awk '{print $5}') bytes"
fi

XGR=${XGR_SRC:-$NB/xgrammar-0.2.8}
if has xgr; then
  if [ ! -d "$XGR" ]; then
    git clone --depth 1 --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar.git "$XGR" > /dev/null 2>&1 || fail "xgrammar clone"
  fi
  FLAGS="-O3 -std=c++17 -fPIC -Wall -Wno-unused-parameter -DXGRAMMAR_ENABLE_CPPTRACE=0 -I$XGR/include -isystem $XGR/3rdparty/picojson -isystem $XGR/3rdparty/dlpack/include"
  ( cd "$XGR" && find cpp -name '*.cc' -not -path 'cpp/tvm_ffi/*' | sort | while read -r f; do
      echo "$CXX $FLAGS -c $f -o $NB/xgr/obj/$(echo "$f" | tr / _ | sed 's/\.cc$/.o/')"; done | xargs -P "$J" -I{} bash -c '{}' ) \
    > "$DS41H_LOGS/build_xgr.log" 2>&1
  ar rcs "$NB/xgr/libxgrammar.a" "$NB"/xgr/obj/*.o || fail "xgrammar (log $DS41H_LOGS/build_xgr.log)"
  say "2. xgrammar: $(ls "$NB"/xgr/obj/*.o | wc -l) objects"
fi

if has core; then
  C=$R/fpr/core; B=$NB/b2a
  ZI=${ZMQ_INC:-$R/frontend_cxx/third_party/build/zeromq-4.3.5/include}
  ( cd "$C" && make -s -j"$J" -f tests/kvtables_py.mk PY="$PY" CXX="$CXX" B="$B" ) > "$DS41H_LOGS/build_kvtables.log" 2>&1 || fail "fpr_kvtables"
  ( cd "$C/tests" && make -s -j"$J" -f sched.mk pymod PY="$PY" CXX="$CXX" BUILD="$B" ) > "$DS41H_LOGS/build_sched.log" 2>&1 || fail "fpr_sched_py"
  ( cd "$C/tests" && make -s -j"$J" -f core.mk OUT="$B/core" XGR="$XGR" XGRLIB="$NB/xgr/libxgrammar.a" CXX="$CXX" PY="$PY" ZMQ_INC="$ZI" pymod core2b ) \
    > "$DS41H_LOGS/build_core.log" 2>&1 || fail "fpr_core_py / fpr_core2b (log $DS41H_LOGS/build_core.log)"
  cp -p "$B"/core/fpr_core_py*.so "$B"/core/fpr_core2b*.so "$B"/
  "$PY" -c "import sys; sys.path.insert(0, '$B'); import fpr_kvtables, fpr_sched_py, fpr_core_py, fpr_core2b; print('import OK')" || fail "core import"
  say "3. core: $(ls "$B"/*.so | wc -l) modules in $B"
fi

if has nd; then
  ( cd "$R/packages/nd/fpr/core/src/nd" && CUDA_VISIBLE_DEVICES= MAX_JOBS=$J "$PY" -c \
      "import sys; sys.path.insert(0, '.'); import nd_cxx; print('nd', nd_cxx.ext().__file__); print('nd2', nd_cxx.ext2().__file__)" ) \
    > "$DS41H_LOGS/build_nd.log" 2>&1 || fail "nd (log $DS41H_LOGS/build_nd.log)"
  say "4. nd: $(grep -c -E '^nd2? ' "$DS41H_LOGS/build_nd.log") modules"
fi

if has ext || has weights; then
  ENVF=${NATIVE_ENV:-$HERE/native.env}
  # the build sees the server's environment: the configuration's flags, then the native switches (they win)
  ( set -e
    # shellcheck disable=SC1090
    source "$ENVF"
    # shellcheck disable=SC1090
    source "$FPR_CONFIG"
    for kv in ${FPR_FLAGS:-}; do export "$kv"; done
    for kv in ${NATIVE_SW:-}; do export "$kv"; done
    export PHYS_ENGINE="$FPR_ENGINE" PHYS_ROOT=$R/fpr DS41H_DEPLOY=$R CUDA_VISIBLE_DEVICES= MAX_JOBS=$J PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH=$R/fpr:$R:$R/runtime
    export PREBUILD_W8A16_DEFS=${W8A16_DEFS:-}; unset W8A16_DEFS
    cd "$R/fpr"
    if has ext; then
      "$PY" engine/prebuild_ext.py --async --moe-x2 > "$DS41H_LOGS/build_ext.log" 2>&1 || { echo "prebuild_ext failed (log $DS41H_LOGS/build_ext.log)"; exit 1; }
      "$PY" "$HERE/prebuild_native.py" > "$DS41H_LOGS/build_ext_native.log" 2>&1 || { tail -5 "$DS41H_LOGS/build_ext_native.log"; exit 1; }
    fi
    if has weights; then
      export DS41H_MODEL_DIR=$MODEL
      pids=(); for p in 0 1 2 3; do "$PY" engine/prebuild.py --part "$p" --nparts 4 & pids+=($!); done
      for p in "${pids[@]}"; do wait "$p" || exit 1; done
      "$PY" engine/prebuild.py --idx
    fi ) || fail "extensions / weights"
  say "5/6. extensions $(find "$R/fpr" "$R/.torch_ext" -name '*.so' 2>/dev/null | wc -l) .so; weight cache $PHYS_WCACHE"
fi
say "build_native done"
