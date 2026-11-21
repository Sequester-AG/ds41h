"""The non-default sampling features a request uses: imported by
PhysScheduler._unsupported (engine process, via bridge) and adapter/sampling_guard.py (API server process), so both
sides judge a request with the same function. Standard library only (no serve/ path, no torch)."""
from __future__ import annotations


def sampling_features(sp) -> set:
    """The non-default sampling features a vLLM SamplingParams uses (beyond temperature / seed / stop / max_tokens)."""
    used = set()
    if sp is None:
        return used
    if sp.top_p is not None and sp.top_p < 1.0:
        used.add("top_p")
    if sp.top_k is not None and sp.top_k > 0:
        used.add("top_k")
    if sp.min_p is not None and sp.min_p > 0.0:
        used.add("min_p")
    if (sp.presence_penalty or 0.0) != 0.0 or (sp.frequency_penalty or 0.0) != 0.0 or (sp.repetition_penalty or 1.0) != 1.0:
        used.add("penalties")
    if getattr(sp, "logit_bias", None):
        used.add("logit_bias")
    if getattr(sp, "bad_words", None) or getattr(sp, "bad_words_token_ids", None):
        used.add("bad_words")
    if getattr(sp, "allowed_token_ids", None):
        used.add("allowed_token_ids")
    if sp.logprobs is not None:
        used.add("logprobs")
    if sp.prompt_logprobs is not None:
        used.add("prompt_logprobs")
    if (sp.min_tokens or 0) > 0:
        used.add("min_tokens")
    return used
