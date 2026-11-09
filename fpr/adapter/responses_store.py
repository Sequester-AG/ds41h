"""A bounded Responses API store for vLLM 0.32's API server.

vLLM keeps stored responses (`store: true`, GET /v1/responses/{id}, `previous_response_id`) in three plain dicts on
OpenAIServingResponses that never shrink: response_store (the responses), msg_store (each stored request's input
messages, read back when a follow-up names it) and event_store (the event queues of background streams). Storing only
happens with VLLM_ENABLE_RESPONSES_API_STORE=1.

Loaded in the API server process with `--middleware adapter.responses_store.BoundedStore`: vLLM imports every
--middleware module in build_app, before init_app_state builds the serving objects, so importing this module patches
OpenAIServingResponses first. The three stores become LRU dicts of at most PHYS_RESPONSES_STORE_MAX responses
(default 4096; message lists up to twice that). Reading or writing an entry marks it recently used. Past the cap the
least recently used response goes, together with its stored input messages and stream events, and its id then answers
404 like an unknown id. The middleware itself passes every request through unchanged.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Callable, Optional

DEFAULT_MAX = 4096


class LRUStore(OrderedDict):
    """A dict that keeps at most `cap` entries, evicting the least recently used (read or written) first."""

    def __init__(self, cap: int, on_evict: Optional[Callable] = None):
        super().__init__()
        self.cap, self.on_evict, self.evicted = max(1, int(cap)), on_evict, 0

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.cap:
            old, _ = self.popitem(last=False)
            self.evicted += 1
            if self.on_evict is not None:
                self.on_evict(old)


def store_max() -> int:
    return int(os.environ.get("PHYS_RESPONSES_STORE_MAX", str(DEFAULT_MAX)))


def bound(serving, cap: Optional[int] = None) -> None:
    """Replace one OpenAIServingResponses' stores (empty at construction) with bounded ones."""
    n = cap if cap is not None else store_max()
    msgs, events = LRUStore(2 * n), LRUStore(n)

    def drop(rid):
        msgs.pop(rid, None)
        events.pop(rid, None)

    for name, new in (("response_store", LRUStore(n, on_evict=drop)), ("msg_store", msgs), ("event_store", events)):
        old = getattr(serving, name, None)
        if old:
            for k, v in old.items():
                new[k] = v
        setattr(serving, name, new)


def install(cap: Optional[int] = None) -> bool:
    """Patch OpenAIServingResponses so every instance gets bounded stores. Returns False if it was already patched."""
    from vllm.entrypoints.openai.responses import serving as S

    cls = S.OpenAIServingResponses
    if getattr(cls, "_phys_bounded_store", False):
        return False
    orig = cls.__init__

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        bound(self, cap)

    cls.__init__ = __init__
    cls._phys_bounded_store = True
    return True


class BoundedStore:
    """Pass-through ASGI middleware: its import (vLLM's --middleware) installs the bounded store."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)


install()
