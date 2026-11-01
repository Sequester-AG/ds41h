#!/usr/bin/env bash
# Idempotent toolchain fixes for the venv (re-run after any reinstall of nvidia-* wheels).
#  1. glibc 2.43: add __THROW to rsqrt/rsqrtf declarations in crt/math_functions.h
#  2. pip CUDA has no unversioned dev symlinks (libcudart.so, ...) and no lib64/ dir:
#     torch.utils.cpp_extension links "-L$CUDA_HOME/lib -lcudart"; FlashInfer JIT links
#     "-L$cuda_home/lib64 -lcudart -lcuda" -> both fail with "cannot find -lcudart".
set -euo pipefail
VENV=${DS41H_HOME:-$HOME/.ds41h}/venv
C=$VENV/lib/python3.12/site-packages/nvidia/cu13
F=$C/include/crt/math_functions.h
ORIG_SHA=decdc28efcfaf0aaf806abc96d7bba9cb84b37c6e83cb82cda59b6aa59916ff8
PATCHED_SHA=26274b7c074dcfc738157b394c1e2524070c0c9d9ceead10c6452c9f7cfdc0f2

sha=$(sha256sum "$F" | cut -d' ' -f1)
if [[ $sha == "$PATCHED_SHA" ]]; then
  echo "math_functions.h: already patched"
elif [[ $sha == "$ORIG_SHA" ]]; then
  sed -i -E 's/\b(rsqrtf?\((double|float) x\));$/\1 __THROW;/' "$F"
  sha=$(sha256sum "$F" | cut -d' ' -f1)
  [[ $sha == "$PATCHED_SHA" ]] || { echo "math_functions.h: unexpected sha $sha after the fix" >&2; exit 1; }
  echo "math_functions.h: patched -> $sha"
else
  echo "math_functions.h: unexpected sha $sha (nvidia-cuda-crt not 13.0.88?)" >&2; exit 1
fi

cd "$C/lib"
for so in libcudart.so.13 libnvrtc.so.13 libcublas.so.13 libcublasLt.so.13 libnvJitLink.so.13 \
          libcurand.so.10 libcusparse.so.12 libcusolver.so.12 libcufft.so.12 libcupti.so.13 \
          libnvfatbin.so.13 libnvblas.so.13; do
  [[ -e $so ]] || continue
  dev=${so%.so.*}.so
  [[ -e $dev ]] || ln -s "$so" "$dev"
done
cd "$C"
[[ -e lib64 ]] || ln -s lib lib64
echo "dev symlinks:"; ls -l "$C/lib64" "$C"/lib/*.so 2>/dev/null | grep -- '->' | awk '{print "  "$(NF-2)" -> "$NF}'
