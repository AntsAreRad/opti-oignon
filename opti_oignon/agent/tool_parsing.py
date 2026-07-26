#!/usr/bin/env python3
"""Tool-block parsing for the agent loop.

Local Ollama models do not reliably emit native function calls, so the agent
must recognise the textual tool-call conventions they actually produce. This
module mirrors the Odysseus ``parse_tool_blocks`` surface (ODYSSEUS_SPEC.md
Section 2.3, Section 5.2): three coexisting call formats, each with its own
compiled regex, plus a normalisation pass into a single parsed representation.

The three formats:

- Fenced code blocks -- a ```fence`` whose body is a JSON object describing a
  tool call. A fence that is not a tool call (ordinary code, arbitrary JSON
  with no tool-name key) is ignored, so this never misfires on plain content.
- Bracketed ``[TOOL_CALL]`` blocks -- ``[TOOL_CALL] {json} [/TOOL_CALL]`` (the
  closing tag is optional), JSON-object payload, nested braces tolerated.
- XML-style ``<invoke>`` / ``<param>`` blocks -- an Anthropic-style call with
  ``<invoke name="...">`` and ``<param name="...">value</param>`` children
  (``<parameter>`` is accepted as an alias).

This module is the parser only. ``dispatch.py`` decides which path a round used
-- native function-calling schemas versus this parser -- and normalises both
into the single ``ToolCall`` representation. The parser stays standalone (no
dependency on ``dispatch``) so it is importlib-isolatable for the runtime
tests; the only optional dependency, ``json_repair``, is guarded.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Module conventions (Theme 3).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Optional tolerant-JSON fallback. ``json_repair`` is stdlib-only and therefore
# importlib-safe, but it is guarded so the parser still loads if it is absent.
try:
    from opti_oignon.json_repair import repair_json as _repair_json

    JSON_REPAIR_AVAILABLE = True
except Exception:  # pragma: no cover - defensive guard
    _repair_json = None
    JSON_REPAIR_AVAILABLE = False


# The three formats this parser recognises.
SUPPORTED_FORMATS = ("fenced", "bracketed", "xml")

# Keys a local model might use to name the tool or carry its arguments. The
# normalisation pass accepts any of these so the dispatch sees one shape.
_NAME_KEYS = ("tool", "name", "tool_name", "action")
_ARG_KEYS = ("arguments", "args", "parameters", "params", "input", "tool_input")


@dataclass
class ParsedToolCall:
    """One tool call recovered from model text.

    ``source`` is the format it was parsed from (one of ``SUPPORTED_FORMATS``);
    ``raw`` is the exact substring matched, kept for observation and audit.
    ``dispatch.py`` converts this into the unified ``ToolCall`` representation.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "source": self.source,
        }


# Compiled patterns

_FENCED_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
_BRACKET_OPEN_RE = re.compile(r"\[TOOL_CALL\]", re.IGNORECASE)
_BRACKET_CLOSE_RE = re.compile(r"\[/TOOL_CALL\]", re.IGNORECASE)
_XML_INVOKE_RE = re.compile(r"<invoke\b([^>]*)>(.*?)</invoke>", re.DOTALL | re.IGNORECASE)
_XML_PARAM_RE = re.compile(
    r"<param(?:eter)?\b([^>]*)>(.*?)</param(?:eter)?>", re.DOTALL | re.IGNORECASE
)
_NAME_ATTR_RE = re.compile(
    r"""name\s*=\s*"([^"]*)"|name\s*=\s*'([^']*)'""", re.IGNORECASE
)


# Low-level helpers


def _loads(text: str) -> Any:
    """Parse JSON, falling back to the tolerant repairer when available."""
    try:
        return json.loads(text)
    except Exception:
        if _repair_json is not None:
            try:
                return _repair_json(text)
            except Exception:
                return None
        return None


def _extract_first_json(text: str, start: int = 0) -> str | None:
    """Return the first balanced ``{...}`` or ``[...]`` span from ``start``.

    String-literal aware, so braces inside JSON strings do not unbalance the
    scan. This is what makes nested-object tool arguments parse correctly.
    """
    n = len(text)
    i = start
    while i < n and text[i] not in "{[":
        i += 1
    if i >= n:
        return None
    open_ch = text[i]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    j = i
    while j < n:
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return text[i : j + 1]
        j += 1
    return None


def _coerce_scalar(value: str) -> Any:
    """Coerce an XML param value: JSON when it parses, else the raw string."""
    try:
        return json.loads(value)
    except Exception:
        return value


def _attr_name(attrs: str) -> str | None:
    """Pull ``name="..."`` (single or double quoted) out of a tag's attrs."""
    m = _NAME_ATTR_RE.search(attrs or "")
    if not m:
        return None
    raw = m.group(1) if m.group(1) is not None else m.group(2)
    return raw.strip() if raw is not None else None


def _build(name: Any, args: Any, source: str, raw: str) -> ParsedToolCall | None:
    """Normalise a (name, args) pair into a ParsedToolCall.

    Arguments are coerced to a dict: a JSON-string payload is parsed; anything
    that is not an object becomes an empty dict (tool arguments are objects).
    Returns ``None`` when there is no usable tool name.
    """
    if name is None:
        return None
    name_s = str(name).strip()
    if not name_s:
        return None
    if isinstance(args, str):
        parsed = _loads(args)
        args = parsed if isinstance(parsed, dict) else {}
    if not isinstance(args, dict):
        args = {}
    return ParsedToolCall(name=name_s, arguments=args, source=source, raw=raw)


def _coerce_call(obj: Any, source: str, raw: str) -> ParsedToolCall | None:
    """Turn a parsed JSON object into a ParsedToolCall, or None if it is not one.

    Recognises a flat ``{"tool"|"name"|...: ..., "arguments"|...: {...}}`` shape
    and the nested OpenAI-style ``{"function": {"name": ..., "arguments": ...}}``.
    """
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return _build(fn.get("name"), fn.get("arguments", {}), source, raw)
    name: Any = None
    for key in _NAME_KEYS:
        candidate = obj.get(key)
        if isinstance(candidate, str) and candidate.strip():
            name = candidate
            break
    if name is None:
        return None
    args: Any = {}
    for key in _ARG_KEYS:
        if key in obj:
            args = obj[key]
            break
    return _build(name, args, source, raw)


# Per-format parsers


def parse_fenced_blocks(text: str) -> list[ParsedToolCall]:
    """Tool calls carried inside fenced code blocks."""
    out: list[ParsedToolCall] = []
    for m in _FENCED_RE.finditer(text or ""):
        span = _extract_first_json(m.group(1))
        if span is None:
            continue
        call = _coerce_call(_loads(span), "fenced", m.group(0))
        if call is not None:
            out.append(call)
    return out


def parse_bracketed_blocks(text: str) -> list[ParsedToolCall]:
    """Tool calls in ``[TOOL_CALL] ... [/TOOL_CALL]`` blocks (close optional)."""
    out: list[ParsedToolCall] = []
    s = text or ""
    for m in _BRACKET_OPEN_RE.finditer(s):
        after = s[m.end() :]
        close = _BRACKET_CLOSE_RE.search(after)
        region = after[: close.start()] if close else after
        span = _extract_first_json(region)
        if span is None:
            continue
        end = m.end() + (close.end() if close else len(region))
        call = _coerce_call(_loads(span), "bracketed", s[m.start() : end])
        if call is not None:
            out.append(call)
    return out


def parse_xml_blocks(text: str) -> list[ParsedToolCall]:
    """Tool calls in XML-style ``<invoke>`` / ``<param>`` blocks."""
    out: list[ParsedToolCall] = []
    for m in _XML_INVOKE_RE.finditer(text or ""):
        name = _attr_name(m.group(1))
        if not name:
            continue
        args: dict[str, Any] = {}
        for pm in _XML_PARAM_RE.finditer(m.group(2)):
            pname = _attr_name(pm.group(1))
            if not pname:
                continue
            args[pname] = _coerce_scalar(pm.group(2).strip())
        out.append(ParsedToolCall(name=name, arguments=args, source="xml", raw=m.group(0)))
    return out


def _dedupe(calls: list[ParsedToolCall]) -> list[ParsedToolCall]:
    """Drop exact duplicates (same name + same arguments), preserving order.

    A message may match more than one pattern (a fenced block whose body also
    contains an ``[TOOL_CALL]`` marker); the first match wins and keeps its
    ``source``.
    """
    seen: set[tuple[str, str]] = set()
    out: list[ParsedToolCall] = []
    for c in calls:
        try:
            key = (c.name, json.dumps(c.arguments, sort_keys=True, default=str))
        except Exception:
            key = (c.name, repr(c.arguments))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def parse_tool_blocks(text: str) -> list[ParsedToolCall]:
    """Parse every recognised tool-call format from model text, in order.

    Fenced first, then bracketed, then XML; exact duplicates are dropped. The
    result is the parser side of the dual dispatch; ``dispatch.py`` normalises
    it together with the native path into the unified ``ToolCall``.
    """
    calls: list[ParsedToolCall] = []
    calls.extend(parse_fenced_blocks(text))
    calls.extend(parse_bracketed_blocks(text))
    calls.extend(parse_xml_blocks(text))
    return _dedupe(calls)


def has_tool_call(text: str) -> bool:
    """Whether the text contains at least one recognisable tool call."""
    return bool(parse_tool_blocks(text))
