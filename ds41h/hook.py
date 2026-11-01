"""Import-time patcher: applies ds41h patches to stock vLLM modules as they are imported.

Pattern adapted from BRRR (github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR, Apache-2.0,
src/brrr/hook.py). Installed from runtime/sitecustomize.py when DS41H=1,
so it runs in the API server and in every spawned vLLM worker.
"""

import sys
from collections.abc import Mapping, Sequence
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from types import ModuleType

from ds41h.patches import PATCHES, Patch


class PatchingLoader:
    def __init__(self, inner: Loader, patches: list[Patch]) -> None:
        self.inner = inner
        self.patches = patches

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return self.inner.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.inner.exec_module(module)
        for patch in self.patches:
            patch(module)


class PatchingFinder:
    def __init__(self, patches: Mapping[str, list[Patch]]) -> None:
        self.patches = patches

    def find_spec(self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None) -> ModuleSpec | None:
        patches = self.patches.get(fullname)
        if not patches:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                spec.loader = PatchingLoader(spec.loader, patches)
                return spec
        return None


def install() -> None:
    if not any(isinstance(f, PatchingFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, PatchingFinder(PATCHES))
