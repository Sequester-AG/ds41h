"""Markers of the adapter tests and when they run.

  gpu     needs a CUDA device (torch.cuda.is_available()); skipped without one.
  vllm    runs vLLM 0.32's engine on the CPU with the fake worker (no GPU, no weights): needs vLLM with the
          deepseek_v41 tokenizer / parsers installed and PHYS_MODEL = a model directory with config.json and the
          tokenizer files (weights are not read: load_format dummy). Skipped when either is missing.
  server  talks to a running fork server: PHYS_SERVER_URL (e.g. http://127.0.0.1:8100), optional PHYS_API_KEY_FILE
          (a file holding one client key; read, sent as a Bearer token, never printed), optional PHYS_CA_FILE (CA bundle
          for https), PHYS_SERVER_FAKE=1 when the server runs the fake worker in script mode (exact replies).
          Skipped without PHYS_SERVER_URL.
Unmarked tests are plain CPU tests.
"""
from __future__ import annotations

import importlib.util
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device")
    config.addinivalue_line("markers", "vllm: vLLM engine on the CPU with the fake worker (needs vLLM + PHYS_MODEL)")
    config.addinivalue_line("markers", "server: needs a running fork server (PHYS_SERVER_URL)")


def _has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _vllm_reason() -> str:
    if importlib.util.find_spec("vllm") is None:
        return "vLLM is not installed"
    m = os.environ.get("PHYS_MODEL", "")
    if not m or not os.path.isfile(os.path.join(m, "config.json")):
        return "PHYS_MODEL (a model directory with config.json + tokenizer) is not set"
    return ""


def pytest_collection_modifyitems(config, items):
    gpu = _has_gpu()
    vr = _vllm_reason()
    url = os.environ.get("PHYS_SERVER_URL", "")
    for it in items:
        if "gpu" in it.keywords and not gpu:
            it.add_marker(pytest.mark.skip(reason="no CUDA device"))
        if "vllm" in it.keywords and vr:
            it.add_marker(pytest.mark.skip(reason=vr))
        if "server" in it.keywords and not url:
            it.add_marker(pytest.mark.skip(reason="PHYS_SERVER_URL is not set"))
