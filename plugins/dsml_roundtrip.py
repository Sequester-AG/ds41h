"""DeepSeek-V4.1 DeepSeek-API PARITY plugin for vLLM 0.32's serving path (v2.2).

Load in the API server with  --tool-parser-plugin /path/to/dsml_roundtrip.py  (scripts/serve_ds41h.sh: DSML_RT=1, the default). Importing this file
patches vLLM classes; every parser patch is gated on the parser config name "deepseek_v41".

v1 -- assistant-turn ROUND TRIP. The OFFICIAL DSv4.1 encoder (model/encoding/encoding.py) renders an assistant turn as
    <think>{reasoning}</think>{content}\n\n<｜DSML｜ calls>...           (tool_calls_start = "\n\n<｜DSML｜ calls")
and its parser returns content WITHOUT that "\n\n" delimiter and reasoning verbatim. vLLM's engine kept the delimiter in
`content` and stripped trailing whitespace from reasoning, so a client sending the turn back re-rendered different tokens and
lost the prefix cache from the content on. Fix: exactly one trailing "\n\n" leaves the content when the tool block follows
(streaming: trailing newlines are held back until the next chunk decides); reasoning keeps its trailing whitespace.

v2 -- TOOL-CALL ARGUMENTS = the official decoder (parse_tool_calls + decode_dsml_to_arguments), byte for byte:
  * each parameter is buffered until ITS exact end token "</｜DSML｜ parameter" (official _read_until_stop): markup inside a
    value (a parameter start tag, "</｜DSML｜ invoke>", "</｜DSML｜ calls>") stays value text -- vLLM's regex cut the value at
    an embedded parameter tag and invented a key, and its lexer ended the call at an embedded invoke / calls end tag;
  * arguments string = "{" + ", ".join(to_json(name) + ": " + (raw value if string="false" else to_json(value))) + "}"
    (official: the model's raw JSON for non-string values, no re-serialisation);
  * no schema coercion (vLLM turned strings into objects / numbers per the tool schema) and no {"arguments": ...} unwrapping;
  * streaming emits each parameter once it is complete (prefix-stable by construction), the closing brace at the end.
  Non-official forms (no / unknown string attribute) keep vLLM's tolerant reading: JSON if it parses, else the text.

v2 -- REQUEST side (DeepSeek API semantics, docs "Thinking Mode"): reasoning_effort minimal -> low, medium -> high,
  xhigh -> high (low / high / max unchanged; "none" = thinking off, as vLLM already does); the top-level
  thinking={"type": "enabled" | "disabled"} (extra_body) switches thinking mode like the API (explicit chat_template_kwargs win).

v2.1 -- PROMPT: the tools' "strict" field is not rendered into the DSv4.1 system prompt (the DeepSeek API's prompt_tokens
  match the official encoder only without it: 5 tokens per tool). Render-only (vllm.tokenizers.deepseek_v41.encode_messages):
  the request keeps tool.function.strict, so vLLM's strict-tool handling (structural tags) is unchanged.

v2.2 -- OPTION DSML_STRICT (env of the API server; default "vllm" = v2.1 behaviour): "api" sets every tool's strict to None
  before vLLM builds its structural tag -> no grammar-constrained tool calls for strict tools (the DeepSeek API documents strict
  as a beta-endpoint feature). Test hook: a request may carry "dsml_strict": "api" | "vllm" (extra body) to override per request.
"""

from __future__ import annotations

import json
import os

import regex as re

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.logger import init_logger
from vllm.parser import deepseek_v4 as _dv4
from vllm.parser.engine import parser_engine as _pe
from vllm.parser.engine import streaming_parser_engine as _spe
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.tokenizers import deepseek_v41 as _dtok

logger = init_logger("vllm.dsml_roundtrip")

_TOOL_STATES = frozenset({ParserState.TOOL_PREAMBLE, ParserState.TOOL_NAME, ParserState.TOOL_ARGS, ParserState.TOOL_BETWEEN})
_DELIM = "\n\n"
_V41 = "deepseek_v41"
_PE = _pe.ParserEngine
_SPE = _spe.StreamingParserEngine

# official DSML parameter grammar (model/encoding/encoding.py parse_tool_calls)
_PSTART = "<｜DSML｜ parameter"
_PEND = "/｜DSML｜ parameter"  # official tool_parameter_end_token; the value is followed by "<" + this
_PCLOSE = "</｜DSML｜ parameter>"
_PARAM_RE = re.compile(r'^ name="(.*?)" string="(true|false)">(.*?)<$', re.DOTALL)
_LOOSE_RE = re.compile(r'^\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>(.*?)<$', re.DOTALL)  # vLLM tolerance
_TAIL_RE = re.compile(r'^\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>(.*)$', re.DOTALL)
_EFFORT = {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "high", "max": "max"}


def _is_v41(self) -> bool:
    cfg = getattr(self, "parser_engine_config", None)
    return getattr(cfg, "name", "") == _V41


def _to_json(value) -> str:  # official to_json
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return json.dumps(value, ensure_ascii=True)


def _entry(name: str, is_str: str | None, value: str, strict: bool) -> str:
    if is_str == "true":
        return f"{_to_json(name)}: {_to_json(value)}"
    if is_str == "false" and strict:
        return f"{_to_json(name)}: {value}"  # official: the model's raw JSON
    try:  # non-official attribute (missing / other): vLLM's reading, JSON if it parses
        json.loads(value)
        return f"{_to_json(name)}: {value}"
    except (ValueError, TypeError):
        return f"{_to_json(name)}: {_to_json(value)}"


class _Scan:
    """Official parameter loop over vLLM's raw invoke body, incremental over append-only text."""

    __slots__ = ("pos", "open_at", "end_from", "names", "entries", "emitted", "closed")

    def __init__(self) -> None:
        self.pos = 0
        self.open_at = -1
        self.end_from = 0
        self.names: set[str] = set()
        self.entries: list[str] = []
        self.emitted = 0
        self.closed = False

    def advance(self, raw: str) -> None:
        while True:
            if self.open_at < 0:
                s = raw.find(_PSTART, self.pos)
                if s < 0:
                    self.pos = max(self.pos, len(raw) - len(_PSTART) + 1)
                    return
                self.open_at = s
                self.end_from = s + len(_PSTART)
            e = raw.find(_PEND, self.end_from)
            if e < 0:
                self.end_from = max(self.end_from, len(raw) - len(_PEND) + 1)
                return
            content = raw[self.open_at + len(_PSTART):e]
            m = _PARAM_RE.match(content)
            strict = m is not None
            m = m or _LOOSE_RE.match(content)
            if m is not None and m.group(1) not in self.names:  # duplicate names: the first one stays (official: error)
                self.names.add(m.group(1))
                self.entries.append(_entry(m.group(1), m.group(2), m.group(3), strict))
            self.pos = e + len(_PEND)
            self.open_at = -1

    def tail_entry(self, raw: str) -> str | None:
        """A parameter cut off by the end of the output (truncated turns only): vLLM's partial reading."""
        if self.open_at < 0:
            return None
        m = _TAIL_RE.match(raw[self.open_at + len(_PSTART):])
        if m is None or m.group(1) in self.names:
            return None
        value = m.group(3)
        for k in range(len(_PCLOSE) - 1, 0, -1):
            if value.endswith(_PCLOSE[:k]):
                value = value[:-k]
                break
        if m.group(2) == "true":
            return f"{_to_json(m.group(1))}: {_to_json(value)}"
        try:
            json.loads(value)
        except (ValueError, TypeError):
            return None
        return f"{_to_json(m.group(1))}: {value}"


def official_arguments(raw: str) -> str:
    """Non-streaming: the official decode_dsml_to_arguments result for vLLM's raw invoke body."""
    st = _Scan()
    st.advance(raw)
    entries = list(st.entries)
    tail = st.tail_entry(raw)
    if tail is not None:
        entries.append(tail)
    return "{" + ", ".join(entries) + "}"


def param_open(raw: str) -> bool:
    st = _Scan()
    st.advance(raw)
    return st.open_at >= 0


def _with_content(delta, content):
    if delta is None:
        return DeltaMessage(content=content) if content else None
    delta.content = content or None
    if delta.content is None and delta.reasoning is None and not delta.tool_calls:
        return None
    return delta


if not getattr(_PE, "_dsml_rt_patched", False):
    _orig_init = _PE.__init__
    _orig_reset = _PE._reset
    _orig_events_to_delta = _PE._events_to_delta
    _orig_finish_streaming = _PE.finish_streaming
    _orig_strip_content_ws = _PE._strip_content_whitespace
    _orig_fix_arg_types = _PE._fix_arg_types
    _orig_compute_arg_delta = _PE._compute_arg_delta
    _orig_flush_arg_converter = _PE._flush_arg_converter
    _orig_convert_args = _dv4.DeepSeekV4Parser._convert_args

    # ---- v1: content delimiter + verbatim reasoning ------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._rt_nl = ""
        self._rt_done = False
        self._rt_stripped = False
        self._v41_scan = {}
        if _is_v41(self):
            self._strip_trailing_reasoning_ws = False

    def _reset(self, *args, **kwargs):
        _orig_reset(self, *args, **kwargs)
        self._rt_nl = ""
        self._rt_done = False
        self._rt_stripped = False
        self._v41_scan = {}

    def _events_to_delta(self, events, finished=False):
        delta = _orig_events_to_delta(self, events, finished)
        if not _is_v41(self) or getattr(self, "_rt_done", False):
            return delta
        pending = getattr(self, "_rt_nl", "")
        content = (delta.content if delta is not None else None) or ""
        if not pending and not content and self._engine.state not in _TOOL_STATES:
            return delta
        combined = pending + content
        self._rt_nl = ""
        if self._engine.state in _TOOL_STATES:
            # the tool-call block has begun: the trailing "\n\n" was the official delimiter, not content
            self._rt_done = True
            if combined.endswith(_DELIM):
                combined = combined[: -len(_DELIM)]
                self._rt_stripped = True
            if not combined.strip() and self._drop_ws_only_content_before_tools and not self._content_has_nonws:
                combined = ""  # stock rule: whitespace-only content before tools is dropped (not a leftover after text)
        elif not finished and self._content_has_nonws:
            body = combined.rstrip("\n")
            if len(body) < len(combined):
                self._rt_nl = combined[len(body):]
                combined = body
        return _with_content(delta, combined)

    def _strip_content_whitespace(self, content, tools_called):
        # non-streaming (_single_pass_parse): the engine has finished, so the delimiter is removed here, once
        if _is_v41(self) and tools_called and content and not self._rt_stripped and content.endswith(_DELIM):
            content = content[: -len(_DELIM)]
            self._rt_stripped = True
        return _orig_strip_content_ws(self, content, tools_called)

    def finish_streaming(self):
        if _is_v41(self) and getattr(self, "_rt_nl", "") and not self._deferred_content:
            return self._events_to_delta(self._engine.finish(), finished=True)
        return _orig_finish_streaming(self)

    # ---- v2: official tool-call arguments ------------------------------------------------------------------------------
    def _fix_arg_types(self, args_json, func_name):
        if _is_v41(self):
            return args_json  # API parity: no schema coercion
        return _orig_fix_arg_types(self, args_json, func_name)

    def _convert_args(self, raw_args, partial):
        if _is_v41(self):
            return official_arguments(raw_args)  # no {"arguments": ...} unwrapping either
        return _orig_convert_args(self, raw_args, partial)

    def _compute_arg_delta(self, idx, raw_delta):
        if not _is_v41(self):
            return _orig_compute_arg_delta(self, idx, raw_delta)
        if not self._stream_arg_deltas or ">" not in raw_delta:
            return None  # a parameter only completes with its end tag
        slot = self._tool_slots[idx]
        st = self._v41_scan.setdefault(idx, _Scan())
        if st.closed:
            return None
        st.advance(slot.args)
        if st.emitted >= len(st.entries):
            return None
        diff = ("{" if st.emitted == 0 else ", ") + ", ".join(st.entries[st.emitted:])
        st.emitted = len(st.entries)
        slot.streamed_json += diff
        return diff

    def _flush_arg_converter(self, idx):
        if not _is_v41(self):
            return _orig_flush_arg_converter(self, idx)
        slot = self._tool_slots[idx]
        st = self._v41_scan.setdefault(idx, _Scan())
        if st.closed:
            return None  # the engine may end the same call twice (invoke end, then finish()): close it once
        st.advance(slot.args)
        rest = st.entries[st.emitted:]
        tail = st.tail_entry(slot.args)
        if tail is not None:
            rest = rest + [tail]
        if st.emitted == 0:
            diff = "{" + ", ".join(rest) + "}"
        else:
            diff = (", " + ", ".join(rest) if rest else "") + "}"
        st.emitted = len(st.entries) + (tail is not None)
        st.closed = True
        slot.streamed_json += diff
        return diff

    # ---- v2: lexer -- an invoke / calls end tag inside an open parameter value is value text ----------------------------
    _orig_spe_reset = _SPE.reset
    _orig_spe_emit = _SPE._emit_for_state
    _orig_spe_on_terminal = _SPE._on_terminal
    _orig_spe_apply = _SPE._apply_transition

    def _spe_reset(self, *args, **kwargs):
        self._v41_raw = []
        return _orig_spe_reset(self, *args, **kwargs)

    def _spe_emit_for_state(self, text, token_count=0):
        if text and self.state == ParserState.TOOL_ARGS and getattr(self.config, "name", "") == _V41:
            raw = getattr(self, "_v41_raw", None)
            if raw is None:
                raw = self._v41_raw = []
            raw.append(text)
        return _orig_spe_emit(self, text, token_count)

    def _spe_on_terminal(self, terminal, value, token_count=0):
        if (terminal in ("INVOKE_END", "TOOL_END") and self.state == ParserState.TOOL_ARGS
                and getattr(self.config, "name", "") == _V41 and param_open("".join(getattr(self, "_v41_raw", None) or []))):
            return self._emit_for_state(value, token_count)
        return _orig_spe_on_terminal(self, terminal, value, token_count)

    def _spe_apply_transition(self, transition, value, token_count=0):
        events = _orig_spe_apply(self, transition, value, token_count)
        if self.state == ParserState.TOOL_ARGS:
            self._v41_raw = []  # a new invoke body starts
        return events

    # ---- v2: request side -- reasoning_effort / thinking as the DeepSeek API reads them -------------------------------
    _orig_build_chat_params = ChatCompletionRequest.build_chat_params

    def _api_normalize(req) -> None:
        mode = (req.model_extra or {}).get("dsml_strict") or os.environ.get("DSML_STRICT", "vllm")
        if mode == "api":
            for t in req.tools or []:
                fn = getattr(t, "function", None)
                if fn is not None and getattr(fn, "strict", None) is not None:
                    fn.strict = None  # before adjust_request builds the structural tag
        eff = req.reasoning_effort
        if isinstance(eff, str) and eff in _EFFORT and _EFFORT[eff] != eff:
            req.reasoning_effort = _EFFORT[eff]
        thinking = (req.model_extra or {}).get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") in ("enabled", "disabled"):
            kw = dict(req.chat_template_kwargs or {})
            if "thinking" not in kw and "enable_thinking" not in kw:
                on = thinking["type"] == "enabled"
                kw["thinking"] = on
                kw["enable_thinking"] = on
                req.chat_template_kwargs = kw

    def build_chat_params(self, *args, **kwargs):
        _api_normalize(self)
        return _orig_build_chat_params(self, *args, **kwargs)

    # ---- v2.1: prompt -- the tools' "strict" field is not rendered (DeepSeek API prompt_tokens parity) ------------------
    _orig_tok_encode_messages = _dtok.encode_messages

    def _strip_strict(tool):
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and "strict" in fn:
            fn = {k: v for k, v in fn.items() if k != "strict"}
            tool = dict(tool)
            tool["function"] = fn
        return tool

    def _encode_messages_no_strict(messages, *args, **kwargs):
        out = []
        for m in messages:
            if isinstance(m, dict) and m.get("tools"):
                m = dict(m)
                m["tools"] = [_strip_strict(t) for t in m["tools"]]
            out.append(m)
        return _orig_tok_encode_messages(out, *args, **kwargs)

    _PE.__init__ = __init__
    _PE._reset = _reset
    _PE._events_to_delta = _events_to_delta
    _PE.finish_streaming = finish_streaming
    _PE._strip_content_whitespace = _strip_content_whitespace
    _PE._fix_arg_types = _fix_arg_types
    _PE._compute_arg_delta = _compute_arg_delta
    _PE._flush_arg_converter = _flush_arg_converter
    _dv4.DeepSeekV4Parser._convert_args = _convert_args
    _SPE.reset = _spe_reset
    _SPE._emit_for_state = _spe_emit_for_state
    _SPE._on_terminal = _spe_on_terminal
    _SPE._apply_transition = _spe_apply_transition
    ChatCompletionRequest.build_chat_params = build_chat_params
    _dtok.encode_messages = _encode_messages_no_strict
    _PE._dsml_rt_patched = True
    logger.info("dsml_roundtrip v2.2: DeepSeek-V4.1 API parity installed (round trip, official tool arguments, "
                "reasoning_effort / thinking mapping, tools rendered without strict; DSML_STRICT=%s)",
                os.environ.get("DSML_STRICT", "vllm"))
