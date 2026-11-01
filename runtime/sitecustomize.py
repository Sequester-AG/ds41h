"""ds41h runtime patch, on PYTHONPATH through scripts/env.sh.

1. Map large safetensors checkpoints with MAP_NORESERVE.

Why: safetensors (framework="pt") calls
    torch.UntypedStorage.from_file(path, shared=False, nbytes=size)
which creates a *writable private* mapping of the whole file. With
vm.overcommit_memory=0 (heuristic) the kernel can refuse a ~101 GB private
writable mapping (ENOMEM) on a host with less RAM, even though the pages are
file-backed and never written. Adding MAP_NORESERVE keeps identical
copy-on-write semantics but skips the commit-charge check.

Lazy: torch is NOT imported at interpreter start-up. A one-shot meta-path hook
patches torch right after `import torch` finishes, so plain `python`, pip,
build helpers etc. stay fast.
Roots: DS41H_NORESERVE_ROOTS (os.pathsep-separated) replaces the default roots
(MODEL, DS41H_MODEL_DIR, MODEL_PATH, HF_HOME); DS41H_NORESERVE=0 disables the patch.

2. Install the ds41h import-time patcher for stock vLLM when DS41H=1 (inherited by vLLM workers).
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import mmap
import os
import sys
from pathlib import Path

# Linux mmap(2) flag; Python's mmap module may not export the name.
_MAP_NORESERVE = getattr(mmap, 'MAP_NORESERVE', 0x4000)
_MARK = '__ds41h_noreserve__'


def _roots():
    raw = os.environ.get('DS41H_NORESERVE_ROOTS')
    if raw:
        items = raw.split(os.pathsep)
    else:
        items = [os.environ.get(k, '') for k in ('MODEL', 'DS41H_MODEL_DIR', 'MODEL_PATH', 'HF_HOME')]
    out = []
    for item in items:
        if not item:
            continue
        try:
            out.append(Path(item).resolve())
        except (OSError, RuntimeError):
            pass
    return tuple(out)


def _apply(torch):
    storage_cls = getattr(torch, 'UntypedStorage', None)
    if storage_cls is None:
        return
    current = storage_cls.from_file
    if getattr(current, _MARK, False):
        return
    original = current
    roots = _roots()

    def _eligible(filename):
        try:
            supplied = Path(os.fsdecode(filename))
            if supplied.suffix != '.safetensors':
                return None
            # HF snapshot symlinks may resolve to extensionless blobs; the
            # suffix check uses the supplied name, containment the real file.
            resolved = supplied.resolve(strict=True)
            if any(resolved.is_relative_to(r) for r in roots):
                return resolved
        except (OSError, TypeError, ValueError, RuntimeError):
            pass
        return None

    def from_file(filename, shared=False, nbytes=0):
        resolved = None
        if not shared and isinstance(nbytes, int) and nbytes > 0:
            resolved = _eligible(filename)
        if resolved is None:
            return original(filename, shared=shared, nbytes=nbytes)
        with open(resolved, 'rb') as src:
            if nbytes > os.fstat(src.fileno()).st_size:
                return original(filename, shared=shared, nbytes=nbytes)
            mapping = mmap.mmap(
                src.fileno(), nbytes,
                flags=mmap.MAP_PRIVATE | _MAP_NORESERVE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        # The returned storage keeps a reference to the mmap buffer, so the
        # mapping outlives this function and safe_open's context. COW writes
        # never reach the file (MAP_PRIVATE).
        return torch.frombuffer(mapping, dtype=torch.uint8).untyped_storage()

    setattr(from_file, _MARK, True)
    from_file.__wrapped__ = original
    storage_cls.from_file = staticmethod(from_file)


class _TorchPostImportHook(importlib.abc.MetaPathFinder):
    """One-shot finder: wraps torch's loader so _apply runs after import."""

    def find_spec(self, fullname, path=None, target=None):
        # Do not remove ourselves here: importlib.util.find_spec('torch') probes (vLLM does this)
        # call find_spec without executing the module; removal happens after torch executes.
        if fullname != 'torch' or 'torch' in sys.modules:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            return spec
        exec_orig = spec.loader.exec_module

        finder = self

        def exec_module(module):
            exec_orig(module)
            try:
                sys.meta_path.remove(finder)
            except ValueError:
                pass
            try:
                _apply(module)
            except Exception as exc:  # never break `import torch`
                print(f'[ds41h sitecustomize] noreserve patch failed: {exc!r}',
                      file=sys.stderr)

        spec.loader.exec_module = exec_module
        return spec


if sys.platform == 'linux' and os.environ.get('DS41H_NORESERVE', '1') != '0':
    if 'torch' in sys.modules:
        _apply(sys.modules['torch'])
    else:
        sys.meta_path.insert(0, _TorchPostImportHook())


# ds41h: install the import-time patcher for stock vLLM when DS41H=1 (inherited by vLLM workers).
if os.environ.get("DS41H", "0") == "1":
    try:
        import ds41h.hook as _ds41h_hook
        _ds41h_hook.install()
    except Exception as exc:
        print(f"[ds41h sitecustomize] ds41h hook install failed: {exc!r}", file=sys.stderr)
