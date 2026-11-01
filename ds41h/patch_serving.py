"""Serving-path (API server) patches: lossless request-latency work outside the GPU.

DS41H_FAST_TOKENIZE (default 1; flags.FAST_TOKENIZE; =0 -> stock)
---------------------------------------------------------------------
The BRRR bench measures PP = prompt_tokens / TTFT, and TTFT includes the API server's tokenization of the chat
prompt. vLLM's DeepSeek-V4.1 tokenizer (vllm/tokenizers/deepseek_v41.py) encodes the rendered prompt as ONE string
with the Rust HF tokenizer: ~25 ms for the 16,482-token bench prompt (Oniguruma regex pre-tokenizer,
single-threaded), i.e. ~6% of the 16K TTFT at 350 W (426 ms, GPU prefill 405 ms).

This patch encodes long prompts as independent chunks with `encode_batch` (rayon-parallel when
TOKENIZERS_PARALLELISM is not false; serve.sh exports it for DS41H runs) and concatenates the ids:
1.5 ms instead of ~25 ms for the 16K prompt (sequential chunks: ~12 ms).

Exactness (the ids must be bit-identical to the stock single-string encode):
* Chunks are cut only at positions p with text[p-1] in "\\r\\n" and text[p] an ASCII letter.
* For the DeepSeek V3/V4 pre-tokenizer (checked at install, else the patch stays off):
  Sequence[Split(\\p{N}{1,3}, Isolated), Split(CJK run, Isolated), Split(<word/punct/space regex>, Isolated),
  ByteLevel(add_prefix_space=False, use_regex=False)], no normalizer, BPE model:
  - no match of the third regex can cross p: every alternative that consumes a letter needs a preceding
    punctuation/letter/non-newline char (p-1 is a newline), and the newline-run alternatives
    (` ?[\\p{P}\\p{S}]+[\\r\\n]*`, `\\s*[\\r\\n]+`) end greedily at the letter; `\\s+(?!\\S)` is never reached for a
    run ending in a newline because `\\s*[\\r\\n]+` precedes it in the alternation -> the full-text pre-tokens split
    at p, and the chunk pre-tokenization reproduces exactly the full-text pre-tokens on both sides (no lookbehind;
    the only lookahead is unaffected for the same reason);
  - the digit / CJK Split stages only add a boundary at p inside a piece that stage 3 splits at p anyway;
  - BPE merges never cross pre-token boundaries; ByteLevel has no prefix space; no normalizer.
* Added (special) tokens: none may contain \\r / \\n, start with an ASCII letter, or carry lstrip / rstrip /
  single_word (checked at install) -> no added-token match can straddle or absorb across p.
* Verified: 257 texts (bench prompts, 300 random source files, licenses), 4,341 split points, 0 mismatches
  tests/test_fast_tokenize.py; DS41H_FAST_TOKENIZE_VERIFY=N re-checks the first N chunked
  encodes of the process against the stock path at runtime (-1 = all; each check costs the stock encode time).
Chunked paths (everything else is stock): `encode(text, add_special_tokens=False)` without other kwargs (the V4.1
tokenizer's apply_chat_template(tokenize=True)), and `tokenizer(text, add_special_tokens=False[, truncation, max_length])`
(vLLM's renderer, renderers/base.py _tokenize_prompt; truncation that would cut the ids, offsets or other kwargs -> stock),
for a single str >= DS41H_FAST_TOKENIZE_MIN_CHARS (8192) characters. The returned BatchEncoding has the stock layout.
"""

from __future__ import annotations

import json
import logging
import os
import re

from ds41h import flags
from ds41h.patches import register

logger = logging.getLogger("ds41h")

_SAFE = re.compile(r"(?<=[\r\n])(?=[A-Za-z])")
_MIN_CHARS = int(os.environ.get("DS41H_FAST_TOKENIZE_MIN_CHARS", "8192"))
_CHUNK = int(os.environ.get("DS41H_FAST_TOKENIZE_CHUNK", "4096"))
# DS41H_FAST_TOKENIZE_VERIFY=N: compare the first N chunked encodes of this process with the stock encode (N=-1: all;
# 0: none). Each verified call costs the stock encode time; the end-to-end gate uses a small N so warmup / bench calibration
# prompts are checked in-server while the measured bench requests stay fast.
_VERIFY_N = int(os.environ.get("DS41H_FAST_TOKENIZE_VERIFY", "0") or 0)
_VERIFIED = {"ok": 0, "bad": 0}
# kwargs of the renderer's tokenizer(...) call that the chunked __call__ path accepts (anything else -> stock)
_CALL_KEYS = {"truncation", "max_length", "add_special_tokens"}

# The DeepSeek-V3/V4 pre-tokenizer the split rule was proven for (tokenizer.json of DeepSeek-V4.1-Flash-1017).
_DS_SPLITS = (
    r"\p{N}{1,3}",
    "[一-龥぀-ゟ゠-ヿ]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
    "| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
)


def split_chunks(text: str, target: int | None = None) -> list[str]:
    """Cut `text` at safe points (newline followed by an ASCII letter) roughly every `target` characters."""
    target = target or _CHUNK
    out, start = [], 0
    while len(text) - start > 1.5 * target:
        m = _SAFE.search(text, start + target)
        if m is None:
            break
        out.append(text[start:m.start()])
        start = m.start()
    out.append(text[start:])
    return out


def ineligible_reason(rust_tokenizer) -> str | None:
    """None if the chunked encode is provably identical for this Rust tokenizer, else the reason why not."""
    try:
        cfg = json.loads(rust_tokenizer.to_str())
    except Exception as e:  # noqa: BLE001
        return f"cannot serialize tokenizer ({type(e).__name__})"
    norm = cfg.get("normalizer")
    if norm is not None and not (norm.get("type") == "Sequence" and not norm.get("normalizers")):
        return f"normalizer {norm.get('type')}"
    pre = cfg.get("pre_tokenizer") or {}
    seq = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if not seq or len(seq) != 4:
        return "pre_tokenizer is not the 4-stage DeepSeek sequence"
    for stage, pattern in zip(seq[:3], _DS_SPLITS):
        if (stage.get("type") != "Split" or (stage.get("pattern") or {}).get("Regex") != pattern
                or stage.get("behavior") != "Isolated" or stage.get("invert")):
            return f"pre_tokenizer split stage differs: {str(stage)[:120]}"
    bl = seq[3]
    if bl.get("type") != "ByteLevel" or bl.get("add_prefix_space") or bl.get("use_regex"):
        return f"pre_tokenizer ByteLevel stage differs: {bl}"
    if (cfg.get("model") or {}).get("type") != "BPE":
        return "model is not BPE"
    for tok in cfg.get("added_tokens") or []:
        c = tok.get("content", "")
        if "\n" in c or "\r" in c or (c[:1].isascii() and c[:1].isalpha()):
            return f"added token {c!r} may straddle a split point"
        if tok.get("lstrip") or tok.get("rstrip") or tok.get("single_word"):
            return f"added token {c!r} has lstrip/rstrip/single_word"
    return None


def _encode_chunks(rust, chunks) -> list[int]:
    """encode_batch of the chunks with NO truncation / padding: transformers leaves the last call's truncation / padding
    parameters on the shared Rust tokenizer (set_truncation_and_padding), which would otherwise cut every chunk."""
    trunc, pad = rust.truncation, rust.padding
    if trunc is not None:
        rust.no_truncation()
    if pad is not None:
        rust.no_padding()
    try:
        encs = rust.encode_batch(chunks, add_special_tokens=False)
    finally:
        if trunc is not None:
            rust.enable_truncation(**trunc)
        if pad is not None:
            rust.enable_padding(**pad)
    return [i for e in encs for i in e.ids]


def _with_fast_encode(wrapped):
    rust = getattr(wrapped, "_tokenizer", None)
    if rust is None or not hasattr(rust, "encode_batch"):
        logger.info("ds41h fast tokenize: OFF (no Rust tokenizer on %s)", type(wrapped).__name__)
        return wrapped
    reason = ineligible_reason(rust)
    if reason:
        logger.info("ds41h fast tokenize: OFF (%s)", reason)
        return wrapped
    base = wrapped.__class__

    class _FastEncode(base):  # type: ignore[misc, valid-type]
        def encode(self, text, text_pair=None, add_special_tokens=True, **kwargs):
            if (not add_special_tokens and text_pair is None and not kwargs and isinstance(text, str)
                    and len(text) >= _MIN_CHARS):
                chunks = split_chunks(text)
                if len(chunks) > 1:
                    ids = _encode_chunks(self._tokenizer, chunks)
                    if _VERIFY_N < 0 or _VERIFIED["ok"] + _VERIFIED["bad"] < _VERIFY_N:
                        ref = list(super().encode(text, add_special_tokens=False))
                        if ids != ref:
                            _VERIFIED["bad"] += 1
                            logger.error("ds41h fast tokenize verify MISMATCH (%d vs %d ids, %d chars): using the stock "
                                         "ids", len(ids), len(ref), len(text))
                            return ref
                        _VERIFIED["ok"] += 1
                        logger.info("ds41h fast tokenize verify ok #%d: %d ids, %d chars, %d chunks",
                                    _VERIFIED["ok"], len(ids), len(text), len(chunks))
                    return ids
            return super().encode(text, text_pair=text_pair, add_special_tokens=add_special_tokens, **kwargs)

        def __call__(self, text=None, *args, **kwargs):
            # vLLM's renderer tokenizes the rendered chat prompt with tokenizer(prompt, truncation=..., max_length=...,
            # add_special_tokens=False) (renderers/base.py _tokenize_prompt), not encode().
            if (not args and isinstance(text, str) and len(text) >= _MIN_CHARS
                    and set(kwargs) <= _CALL_KEYS and kwargs.get("add_special_tokens", True) is False):
                chunks = split_chunks(text)
                if len(chunks) > 1:
                    ids = _encode_chunks(self._tokenizer, chunks)
                    max_length = kwargs.get("max_length")
                    if not (kwargs.get("truncation") and max_length is not None and len(ids) > max_length):
                        shape = super().__call__("", **kwargs)  # the stock BatchEncoding layout (keys) for these kwargs
                        fill = {"input_ids": ids, "attention_mask": [1] * len(ids), "token_type_ids": [0] * len(ids)}
                        if set(shape.keys()) <= set(fill):
                            for key in shape.keys():
                                shape[key] = fill[key]
                            shape._encodings = None  # no Rust Encoding objects: char/word mapping helpers raise instead of lying
                            if _VERIFY_N < 0 or _VERIFIED["ok"] + _VERIFIED["bad"] < _VERIFY_N:
                                ref = super().__call__(text, **kwargs)
                                if dict(ref) != dict(shape):
                                    _VERIFIED["bad"] += 1
                                    logger.error("ds41h fast tokenize verify MISMATCH in __call__ (%d vs %d ids, %d chars): "
                                                 "using the stock encoding", len(ids), len(ref["input_ids"]), len(text))
                                    return ref
                                _VERIFIED["ok"] += 1
                                logger.info("ds41h fast tokenize verify ok #%d (__call__): %d ids, %d chars, %d chunks",
                                            _VERIFIED["ok"], len(ids), len(text), len(chunks))
                            return shape
            return super().__call__(text, *args, **kwargs)

    _FastEncode.__name__ = base.__name__
    _FastEncode.__qualname__ = base.__qualname__
    wrapped.__class__ = _FastEncode
    logger.info("ds41h fast tokenize: ON (chunked encode_batch for prompts >= %d chars, chunks ~%d chars, "
                "TOKENIZERS_PARALLELISM=%s, verify first %d)", _MIN_CHARS, _CHUNK,
                os.environ.get("TOKENIZERS_PARALLELISM"), _VERIFY_N)
    return wrapped


def _patch_dsv41_tokenizer(mod) -> None:
    orig = mod.get_deepseek_v41_tokenizer
    if getattr(orig, "_ds41h_fast", False):
        return

    def get_deepseek_v41_tokenizer(tokenizer):
        return _with_fast_encode(orig(tokenizer))

    get_deepseek_v41_tokenizer._ds41h_fast = True  # type: ignore[attr-defined]
    get_deepseek_v41_tokenizer.__doc__ = orig.__doc__
    mod.get_deepseek_v41_tokenizer = get_deepseek_v41_tokenizer
    try:
        from ds41h.patch_core import mark_active

        mark_active("serving", "deepseek_v41 tokenizer: chunked parallel encode")
    except Exception:  # noqa: BLE001
        pass


if flags.FAST_TOKENIZE:
    register("vllm.tokenizers.deepseek_v41")(_patch_dsv41_tokenizer)
