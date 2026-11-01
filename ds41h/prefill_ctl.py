"""Runtime control file for prefill experiments: A/B several prefill options on ONE server.

DS41H_PREFILL_CTL=<path to a JSON object>: prefill modules read their switches through get(key, default); the file
is re-read whenever its mtime changes (one os.stat per call). Unset (the default) = every get() returns the default
(the environment switches), i.e. no behaviour change. The experiment driver must only rewrite the file while the
server is idle: every TP rank must see the same value in a given step (collectives depend on it).
Keys used: "sp" (0/1, ds41h/prefill_sp.py), "moe_ab" (comma list of W4A16 variants per layer, ds41h/moe.py).
"""

from __future__ import annotations

import json
import os
from typing import Any

PATH = os.environ.get("DS41H_PREFILL_CTL", "")
_state: dict = {}
_mtime: int | None = None


def get(key: str, default: Any) -> Any:
    global _state, _mtime
    if not PATH:
        return default
    try:
        m = os.stat(PATH).st_mtime_ns
    except OSError:
        return default
    if m != _mtime:
        try:
            with open(PATH) as f:
                _state = json.load(f)
            _mtime = m
        except (OSError, ValueError):
            return default
    return _state.get(key, default)
