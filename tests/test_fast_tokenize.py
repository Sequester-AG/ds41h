"""CPU test: ds41h fast tokenize (ds41h/patch_serving.py) gives bit-identical ids to the stock DeepSeek-V4.1 encode.

  DS41H=1 DS41H_FAST_TOKENIZE=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES= python tests/test_fast_tokenize.py

Goes through the real vLLM tokenizer object (tokenizer_mode deepseek_v41, patched at import by the ds41h hook) and
compares every chunked encode with the stock transformers TokenizersBackend.encode of the same string; also checks the
chat path (apply_chat_template tokenize=True) and prints timings for the bench-sized prompts.
"""

import glob
import os
import random
import secrets
import sys
import sysconfig
import time

os.environ.setdefault("DS41H", "1")
os.environ.setdefault("DS41H_FAST_TOKENIZE", "1")
os.environ.setdefault("DS41H_FAST_TOKENIZE_MIN_CHARS", "64")  # exercise the chunked path on short texts too
MODEL = os.environ.get("DS41H_TOKENIZER_MODEL", os.environ.get("DS41H_MODEL_DIR", os.path.join(os.environ.get("DS41H_HOME", os.path.expanduser("~/.ds41h")), "model")))
SITE = sysconfig.get_paths()["purelib"]


def main() -> int:
    from ds41h import hook

    hook.install()
    from transformers import TokenizersBackend

    from ds41h import patch_serving
    from vllm.tokenizers import get_tokenizer

    tok = get_tokenizer(MODEL, tokenizer_mode="deepseek_v41")
    assert type(tok).encode is not TokenizersBackend.encode, "fast tokenize patch not active"
    assert patch_serving.ineligible_reason(tok._tokenizer) is None
    # the hook may have imported patch_serving at interpreter start (sitecustomize): set the threshold directly
    patch_serving._MIN_CHARS = int(os.environ["DS41H_FAST_TOKENIZE_MIN_CHARS"])
    calls = {"chunked": 0}
    orig_split = patch_serving.split_chunks

    def counting_split(text, target=None):
        out = orig_split(text, target)
        calls["chunked"] += len(out) > 1
        return out

    patch_serving.split_chunks = counting_split

    texts: list[tuple[str, str]] = []
    try:
        from brrr.bench import CODE_SUFFIX, CODE_TASK

        for reps in (8, 60, 240, 480):
            body = f"{secrets.token_hex(8)} " + "\n\n".join([CODE_TASK] * reps) + f"\n\n{CODE_SUFFIX}"
            texts.append((f"bench x{reps}", body))
    except ImportError:
        print("brrr bench not importable: skipping bench prompts")
    random.seed(0)
    files = [f for f in glob.glob(f"{SITE}/vllm/**/*.py", recursive=True) if os.path.getsize(f) > 4000]
    for f in random.sample(files, min(200, len(files))):
        texts.append((f, open(f, errors="replace").read()))
    lic = [f for f in glob.glob(f"{SITE}/**/LICENSE*", recursive=True) if os.path.isfile(f)]
    for f in lic[:30]:
        texts.append((f, open(f, errors="replace").read()))
    edge = [
        "line one\r\nLine two\r\n\r\nthird\rfourth\n" * 50,
        "x = 1\n\n\n   abc\n\t\tdef\n \nghi\n" * 80,
        "数字 123456 和中文\n文本 abc\nДобрый день\nmixed ünïcödé\nemoji 😀👍\nZ" * 60,
        "<｜User｜>inside content\n<｜Assistant｜>\nthink</think>\nend" * 40,
        "!!!\nA!!!\n,,,abc\n...Def\n#include <x>\nint\n" * 70,
        "\n".join(f"word{i} {i * 7919}" for i in range(2000)),
    ]
    texts += [(f"edge{i}", e) for i, e in enumerate(edge)]

    bad = splits = 0
    # transformers leaves the last call's truncation on the shared Rust tokenizer: start from a polluted state
    tok("x" * 4000, truncation=True, max_length=7, add_special_tokens=False)
    for name, text in texts:
        ref = TokenizersBackend.encode(tok, text, add_special_tokens=False)
        for chunk in (64, 512, 4096):
            patch_serving._CHUNK = chunk
            ids = tok.encode(text, add_special_tokens=False)
            splits += len(orig_split(text, chunk)) - 1
            if list(ids) != list(ref):
                bad += 1
                print(f"MISMATCH {name} chunk {chunk}: {len(ids)} vs {len(ref)} ids")
    patch_serving._CHUNK = 4096
    print(f"{len(texts)} texts, {splits} split points, {calls['chunked']} chunked encodes: {bad} mismatches")
    assert calls["chunked"] > len(texts), "chunked path not exercised"

    # renderer path: tokenizer(text, truncation=True, max_length=..., add_special_tokens=False) (renderers/base.py)
    call_bad = 0
    for name, text in texts:
        for kw in (dict(truncation=True, max_length=36865, add_special_tokens=False),
                   dict(truncation=True, max_length=257, add_special_tokens=False),  # cuts -> stock fallback
                   dict(truncation=False, add_special_tokens=False),
                   dict(add_special_tokens=False, return_offsets_mapping=True)):     # offsets -> stock fallback
            got = tok(text, **kw)
            ref = TokenizersBackend.__call__(tok, text, **kw)
            if dict(got) != dict(ref):
                call_bad += 1
                print(f"__call__ MISMATCH {name} {kw}")
    bad += call_bad
    print(f"__call__ path: {len(texts) * 4} calls, {call_bad} mismatches")

    # chat path (what the API server runs) + timings
    for name, body in texts[:4]:
        msgs = [{"role": "user", "content": body}]
        rendered = tok.apply_chat_template(msgs, tokenize=False, thinking=False)
        t0 = time.perf_counter()
        ref = TokenizersBackend.encode(tok, rendered, add_special_tokens=False)
        t1 = time.perf_counter()
        ids = tok.apply_chat_template(msgs, tokenize=True, thinking=False)
        t2 = time.perf_counter()
        ok = list(ids) == list(ref)
        bad += not ok
        print(f"chat {name}: {len(ref)} ids, stock encode {1e3 * (t1 - t0):.1f} ms, chat path (fast) "
              f"{1e3 * (t2 - t1):.1f} ms, identical={ok}")
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
