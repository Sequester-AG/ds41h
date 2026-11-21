"""Reject sampling features the phys engine cannot do with a clean 400 BEFORE streaming.

Without this, a request using e.g. min_tokens reaches the engine, PhysScheduler._unsupported() finishes it as
FINISHED_ERROR, and a streaming client already holding the role chunk gets a 500 mid-stream.

Loaded in the API server process with `--middleware adapter.sampling_guard.SamplingGuard` (vLLM imports every
--middleware module in build_app, before the serving objects exist):
  * the import wraps ChatCompletionRequest / CompletionRequest / ResponsesRequest.to_sampling_params: after vLLM has
    built the final SamplingParams (request fields + the server's default_sampling_params), the features it uses
    (adapter.sampling_features: the SAME function the engine's _unsupported applies via bridge) minus the engine's caps
    raise VLLMValidationError -> vLLM's standard 400 ErrorResponse, before any generator exists (stream or not);
  * the caps come from the engine itself, ONE source of truth: the EngineCore utility `phys_sampling_caps`
    (adapter/phys_executor.py adds it in the engine process; it returns sorted(bridge.get_sampling_caps()) or None).
    The middleware fetches them once, on the first GENERATION request (5-s bound, retried after a timeout). An engine without the
    utility (not the phys engine) or caps None = no check here (the engine-side _unsupported stays the backstop).
The middleware itself passes every request through unchanged.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from adapter.sampling_features import sampling_features   # stdlib only (not adapter.bridge: engine-side state)

logger = logging.getLogger("vllm.adapter.sampling_guard")

UTILITY = "phys_sampling_caps"
FETCH_TIMEOUT_S = float(os.environ.get("PHYS_CAPS_FETCH_TIMEOUT_S", "5"))
RETRY_S = 60.0                     # a timed-out fetch is retried on a later request, no sooner than this
_STATE = {"caps": None, "fetched": False, "next_try": 0.0}
_LOCK = asyncio.Lock()

# feature (bridge.sampling_features) -> the request fields that turn it on, in the order the message names them
FIELDS = {
    "top_p": ("top_p",),
    "top_k": ("top_k",),
    "min_p": ("min_p",),
    "penalties": ("presence_penalty", "frequency_penalty", "repetition_penalty"),
    "logit_bias": ("logit_bias",),
    "bad_words": ("bad_words",),
    "allowed_token_ids": ("allowed_token_ids",),
    "logprobs": ("logprobs", "top_logprobs"),
    "prompt_logprobs": ("prompt_logprobs",),
    "min_tokens": ("min_tokens",),
}
_SP_DEFAULT = {"top_p": 1.0, "top_k": 0, "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
               "repetition_penalty": 1.0, "min_tokens": 0}


def _field_of(feature: str, sp) -> str:
    """The request field to name for one missing feature: the first of FIELDS[feature] that is non-default in sp
    (penalties: which penalty), else the first name."""
    names = FIELDS.get(feature, (feature,))
    for n in names:
        v = getattr(sp, n, None)
        if n in _SP_DEFAULT:
            if v is not None and v != _SP_DEFAULT[n] and not (n == "top_k" and v < 0):
                return n
        elif v:
            return n
    return names[0]


def check(sp) -> None:
    """Raise VLLMValidationError when sp uses a sampling feature outside the engine's caps (no-op without caps)."""
    caps = _STATE["caps"]
    if caps is None or sp is None:
        return
    missing = sorted(sampling_features(sp) - caps)
    if not missing:
        return
    from vllm.exceptions import VLLMValidationError
    fields = [_field_of(f, sp) for f in missing]
    raise VLLMValidationError(
        f"{', '.join(fields)} {'is' if len(fields) == 1 else 'are'} not supported by this server's sampler "
        f"(supported: temperature, seed, stop, max_tokens{''.join(', ' + c for c in sorted(caps))})",
        parameter=fields[0])


def _wrap(cls) -> None:
    orig = cls.to_sampling_params
    if getattr(orig, "_phys_guard", False):
        return

    def to_sampling_params(self, *a, **kw):
        sp = orig(self, *a, **kw)
        check(sp)
        return sp

    to_sampling_params._phys_guard = True
    cls.to_sampling_params = to_sampling_params


def install() -> list:
    """Wrap each request class; one that does not exist / has no to_sampling_params in this vLLM is skipped with a
    warning (the middleware import must never break API-server startup)."""
    import importlib
    done = []
    for mod, name in (("vllm.entrypoints.openai.chat_completion.protocol", "ChatCompletionRequest"),
                      ("vllm.entrypoints.openai.completion.protocol", "CompletionRequest"),
                      ("vllm.entrypoints.openai.responses.protocol", "ResponsesRequest")):
        try:
            cls = getattr(importlib.import_module(mod), name)
            if not callable(getattr(cls, "to_sampling_params", None)):
                raise AttributeError(f"{name}.to_sampling_params missing")
            _wrap(cls)
            done.append(name)
        except Exception as e:  # noqa: BLE001
            logger.warning("sampling_guard: %s.%s not wrapped (%s): no API-side check for it", mod, name, str(e)[:160])
    return done


async def fetch_caps(engine_client) -> None:
    """Once: the engine's caps over the utility channel, bounded by FETCH_TIMEOUT_S. No such utility (not the phys
    engine) = no check, for good; a timeout = no check for now, retried on a later request after RETRY_S."""
    if _STATE["fetched"] or time.monotonic() < _STATE["next_try"]:
        return
    async with _LOCK:
        if _STATE["fetched"] or time.monotonic() < _STATE["next_try"]:
            return
        try:
            r = await asyncio.wait_for(engine_client.engine_core.call_utility_async(UTILITY), FETCH_TIMEOUT_S)
            _STATE["caps"] = None if r is None else frozenset(r)
            _STATE["fetched"] = True
            logger.info("sampling_guard: engine sampling caps %s",
                        "all (no check)" if r is None else (sorted(r) or "none beyond temperature / seed / stop / max_tokens"))
        except asyncio.TimeoutError:
            _STATE["next_try"] = time.monotonic() + RETRY_S
            logger.warning("sampling_guard: %s timed out after %.0f s: no API-side check yet, retry in %.0f s",
                           UTILITY, FETCH_TIMEOUT_S, RETRY_S)
        except Exception as e:  # noqa: BLE001  (not the phys engine: the utility does not exist)
            _STATE["caps"] = None
            _STATE["fetched"] = True
            logger.warning("sampling_guard: no %s utility on the engine (%s): no API-side check", UTILITY, str(e)[:160])


def _wants_caps(path: str) -> bool:
    """Only generation routes need the caps: /health, /metrics, /v1/models ... never wait on the utility."""
    return path.startswith(("/v1/chat/completions", "/v1/completions", "/v1/responses"))


class SamplingGuard:
    """Pass-through ASGI middleware: its import installs the to_sampling_params checks; the first generation request
    fetches the engine's caps (before that request is routed). Other routes (the watchdog's /health) never wait."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and not _STATE["fetched"] and _wants_caps(scope.get("path", "")):
            ec = getattr(getattr(scope.get("app"), "state", None), "engine_client", None)
            if ec is not None and getattr(ec, "engine_core", None) is not None:
                await fetch_caps(ec)
        await self.app(scope, receive, send)


INSTALLED = install()
logger.info("sampling_guard: to_sampling_params checks on %s", ", ".join(INSTALLED))
