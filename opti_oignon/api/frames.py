#!/usr/bin/env python3
"""Frames the sockets send, as a strict parser reads them.

A browser reads each frame with ``JSON.parse``, which refuses what Python's
``json`` writes by default for a number JSON cannot carry (``NaN``,
``Infinity``); and a socket carries its text as UTF-8, which cannot hold a
lone surrogate, the kind a JSON escape decodes into a ``str``. A model's tool
arguments can hold either, and a tool's output the second. The chat client
ended the turn on a frame it could not parse, the agent client dropped the
event, and a frame that could not be sent read, to the chat route, as the
client gone.

Stdlib only.
"""

import json
import math
from typing import Any


def strict_frame(payload: Any) -> Any:
    """The payload as a strict parser reads it: itself, when it already is.

    A number JSON cannot carry comes as its text (``"NaN"``, ``"Infinity"``,
    ``"-Infinity"``), as the tool call approval shows it; a lone surrogate as
    its escape, written out (``"\\ud800"``); a value JSON has no form for, or
    a key that is not text, a number, a boolean or null, as its ``str``.
    Everything else is as it was.
    """
    try:
        json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        return _strict(payload, frozenset())
    return payload


def _strict(value: Any, seen: frozenset[int]) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else json.dumps(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _carried(value)
    if isinstance(value, (dict, list, tuple)):
        if id(value) in seen:
            return _carried(str(value))
        inner = seen | {id(value)}
        if isinstance(value, dict):
            return {_key(key): _strict(item, inner) for key, item in value.items()}
        return [_strict(item, inner) for item in value]
    return _carried(str(value))


def _key(key: Any) -> Any:
    if isinstance(key, float) and not math.isfinite(key):
        return json.dumps(key)
    if key is None or isinstance(key, (bool, int, float)):
        return key
    return _carried(key if isinstance(key, str) else str(key))


def _carried(text: str) -> str:
    """Text UTF-8 can carry: each lone surrogate written as its escape."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return "".join(f"\\u{ord(c):04x}" if "\ud800" <= c <= "\udfff" else c for c in text)
    return text
