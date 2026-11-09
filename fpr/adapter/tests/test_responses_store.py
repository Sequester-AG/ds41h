"""adapter/responses_store.py: vLLM's Responses API store bounded (count LRU), installed in the API server process
through --middleware. Plain CPU tests; the patch of vLLM's class needs vLLM importable.

Guards: stored responses, their input messages and stream events are evicted together, least recently used first, so
memory stays bounded and a follow-up never finds a response without its messages; reading an entry keeps it; the
middleware passes requests through unchanged. The server-side round trip (store, GET, previous_response_id, eviction)
is in test_api_contract.py.
"""
from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def RS():
    pytest.importorskip("vllm")
    from adapter import responses_store

    return responses_store


def test_lru_order_and_eviction(RS):
    ev = []
    s = RS.LRUStore(3, on_evict=ev.append)
    for k in "abc":
        s[k] = k.upper()
    assert s["a"] == "A" and s.get("b") == "B"           # reads refresh: c is now the oldest
    s["d"] = "D"
    assert ev == ["c"] and list(s) == ["a", "b", "d"] and s.evicted == 1
    assert s.get("c") is None and "c" not in s
    s["a"] = "A2"                                         # an overwrite refreshes too
    s["e"] = "E"
    assert ev == ["c", "b"] and s["a"] == "A2"


def test_follow_up_chain_and_eviction(RS):
    """The serving object's three stores as vLLM uses them: a response and its messages stay together; past the cap
    the oldest response goes with its messages and events; a read keeps one alive."""
    srv = SimpleNamespace(response_store={}, msg_store={}, event_store={})
    RS.bound(srv, 3)
    for i in range(3):
        rid = f"resp_{i}"
        srv.msg_store[rid] = [f"input {i}"]
        srv.response_store[rid] = {"id": rid}
        srv.event_store[rid] = ("events", i)
    assert srv.response_store.get("resp_0")["id"] == "resp_0"      # a follow-up reads turn 1: resp_0 is kept
    assert srv.msg_store.get("resp_0") == ["input 0"]
    srv.msg_store["resp_3"] = ["input 3"]
    srv.response_store["resp_3"] = {"id": "resp_3"}
    assert "resp_1" not in srv.response_store and "resp_1" not in srv.msg_store and "resp_1" not in srv.event_store
    assert "resp_0" in srv.response_store and "resp_0" in srv.msg_store
    for i in range(4, 40):
        srv.msg_store[f"resp_{i}"] = [i]
        srv.response_store[f"resp_{i}"] = {"id": i}
    assert len(srv.response_store) == 3 and len(srv.msg_store) <= 6 and len(srv.event_store) <= 3


def test_cap_from_env(RS, monkeypatch):
    monkeypatch.setenv("PHYS_RESPONSES_STORE_MAX", "5")
    srv = SimpleNamespace(response_store={"keep": 1}, msg_store={}, event_store={})
    RS.bound(srv)
    assert srv.response_store.cap == 5 and srv.msg_store.cap == 10 and srv.response_store["keep"] == 1


def test_installed_on_vllm_class(RS):
    """Importing the module (vLLM's --middleware import) patches OpenAIServingResponses once."""
    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

    assert OpenAIServingResponses._phys_bounded_store is True
    assert RS.install() is False


def test_middleware_passes_through(RS):
    seen = []

    async def app(scope, receive, send):
        seen.append(scope)
        await send({"type": "done"})

    sent = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {}

    asyncio.run(RS.BoundedStore(app)({"type": "http", "path": "/v1/responses"}, receive, send))
    assert seen == [{"type": "http", "path": "/v1/responses"}] and sent == [{"type": "done"}]
