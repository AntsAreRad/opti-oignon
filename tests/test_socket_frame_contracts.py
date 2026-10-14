#!/usr/bin/env python3
"""Every frame a socket sends is one a strict parser reads.

A browser reads each frame with ``JSON.parse``. Python's ``json`` writes a
number JSON cannot carry as ``NaN`` or ``Infinity``, which ``JSON.parse``
refuses, and a lone surrogate, the kind a JSON escape in a model's tool
arguments decodes into, as itself, which UTF-8 cannot carry: a server must
send a frame's text as UTF-8. The chat client ended the turn on a frame it
could not parse; a frame that could not be sent read, to the chat route, as
the client gone, and a token frame stopped the turn; the agent stream dropped
an event it could not write, and wrote ``NaN`` where it could. Every frame now
goes through ``api.frames.strict_frame``:

  * SF1 -- a chat frame holding a number JSON cannot carry is read by a strict
    parser, the number as its text and the rest of the frame as it was;
  * SF2 -- a chat frame holding a lone surrogate is sent, and the turn goes on:
    the surrogate comes as its escape, written out;
  * SF3 -- an agent event holding what JSON cannot carry reaches its
    subscribers, read by a strict parser; none is dropped;
  * SF4 -- every frame a socket sends in the package goes through the strict
    frame: a census of the sends, shown able to name one that does not;
  * SF5 -- the strict frame changes only what a strict parser cannot read: a
    frame that is already strict comes back as itself, and on payloads drawn
    at random every other value is as it was.

SF1 and SF2 send through a real starlette socket whose channel carries text
as a server must, as UTF-8. The routes are loaded through the shared
isolation window. Local-only.
"""

import ast
import asyncio
import json
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_sf1_a_chat_frame_holding_a_number_json_cannot_carry_is_read_by_a_strict_parser": 3.0,
    "test_sf2_a_chat_frame_holding_a_lone_surrogate_is_sent_and_the_turn_goes_on": 3.0,
    "test_sf3_an_agent_event_holding_what_json_cannot_carry_reaches_its_subscribers": 3.0,
    "test_sf4_every_frame_a_socket_sends_goes_through_the_strict_frame": 3.0,
    "test_sf5_the_strict_frame_changes_only_what_a_strict_parser_cannot_read": 3.0,
}

_PACKAGE = Path(__file__).resolve().parent.parent / "opti_oignon"
_SENDING = {"api/routes_agent.py", "api/routes_benchmark.py", "api/routes_chat.py", "api/routes_coding.py",
            "api/routes_live_metrics.py"}


def _read(text):
    """The frame as a strict parser reads it: NaN and the infinities refused."""
    def refuse(constant):
        raise ValueError(f"not JSON: {constant}")

    return json.loads(text, parse_constant=refuse)


def _socket():
    """(socket, carried): a starlette socket whose channel carries each frame's text as a server must, as UTF-8."""
    from starlette.websockets import WebSocket

    carried = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        if message["type"] == "websocket.send":
            carried.append(message["text"].encode("utf-8").decode("utf-8"))

    socket = WebSocket({"type": "websocket", "path": "/", "headers": [], "query_string": b""}, receive, send)
    asyncio.run(socket.accept())
    return socket, carried


def _chat():
    deps = types.ModuleType("opti_oignon.api.deps")
    for name in ("ANALYZER_AVAILABLE", "CONVERSATION_AVAILABLE", "EXECUTOR_AVAILABLE", "PRESET_AVAILABLE",
                 "ROUTER_AVAILABLE"):
        setattr(deps, name, False)
    for name in ("analyzer", "conversation_manager", "executor", "preset_manager", "router"):
        setattr(deps, name, None)
    loaded, restore = isolate(targets={"opti_oignon.api.frames": source("api", "frames.py"),
                                       "opti_oignon.api.schemas": source("api", "schemas.py"),
                                       "opti_oignon.api.routes_chat": source("api", "routes_chat.py")},
                              seeded={"opti_oignon.api.deps": deps}, packages=("opti_oignon.api",))
    return loaded["opti_oignon.api.routes_chat"], restore


def _frames():
    loaded, restore = isolate(targets={"opti_oignon.api.frames": source("api", "frames.py")},
                              packages=("opti_oignon.api",))
    return loaded["opti_oignon.api.frames"], restore


# ---------------------------------------------------------------------------
# SF1 -- a number JSON cannot carry, in a chat frame
# ---------------------------------------------------------------------------
def test_sf1_a_chat_frame_holding_a_number_json_cannot_carry_is_read_by_a_strict_parser():
    arguments = {"x": float("nan"), "y": float("inf"), "z": float("-inf"), "n": 1.5, "s": "text", "big": 2 ** 60}
    chat, restore = _chat()
    try:
        socket, carried = _socket()
        alive = asyncio.run(chat._send_token(socket, "tool_call", "", metadata={
            "tool_name": "calculator", "arguments": arguments, "success": True}))
    finally:
        restore()
    assert alive and len(carried) == 1, (alive, carried)
    assert _read(carried[0]) == {"type": "tool_call", "content": "", "metadata": {
        "tool_name": "calculator",
        "arguments": {"x": "NaN", "y": "Infinity", "z": "-Infinity", "n": 1.5, "s": "text", "big": 2 ** 60},
        "success": True}}


# ---------------------------------------------------------------------------
# SF2 -- a lone surrogate, in a chat frame
# ---------------------------------------------------------------------------
def test_sf2_a_chat_frame_holding_a_lone_surrogate_is_sent_and_the_turn_goes_on():
    chat, restore = _chat()
    try:
        socket, carried = _socket()
        sent = [asyncio.run(chat._send_token(socket, "token", "half \ud83d of a pair")),
                asyncio.run(chat._send_token(socket, "tool_call", "", metadata={"result_preview": "out \udcff put"})),
                asyncio.run(chat._send_token(socket, "token", "whole \U0001f600"))]
    finally:
        restore()
    assert sent == [True, True, True], "a frame that is not sent reads as the client gone, and stops the turn"
    frames = [_read(text) for text in carried]
    assert frames[0]["content"] == "half \\ud83d of a pair", frames[0]
    assert frames[1]["metadata"]["result_preview"] == "out \\udcff put", frames[1]
    assert frames[2]["content"] == "whole \U0001f600", "control: a character past the BMP is carried as itself"


# ---------------------------------------------------------------------------
# SF3 -- the agent stream
# ---------------------------------------------------------------------------
def test_sf3_an_agent_event_holding_what_json_cannot_carry_reaches_its_subscribers():
    loaded, restore = isolate(targets={"opti_oignon.api.frames": source("api", "frames.py"),
                                       "opti_oignon.api.routes_agent": source("api", "routes_agent.py")},
                              packages=("opti_oignon.api",))
    try:
        manager = loaded["opti_oignon.api.routes_agent"].AgentRunManager()
        got = []
        manager.subscribe(got.append)
        manager._on_event(SimpleNamespace(kind="tool_call", round=2, data={
            "arguments": {"x": float("nan"), "y": float("-inf")}, "seen": {3}, "out": "a \udcff b"}))
        manager._on_event(SimpleNamespace(kind="done", round=2, data={"answer": "ok"}))
    finally:
        restore()
    assert len(got) == 2, ("no event is dropped", got)
    assert _read(got[0]) == {"kind": "tool_call", "round": 2, "data": {
        "arguments": {"x": "NaN", "y": "-Infinity"}, "seen": "{3}", "out": "a \\udcff b"}}
    assert _read(got[1]) == {"kind": "done", "round": 2, "data": {"answer": "ok"}}


# ---------------------------------------------------------------------------
# SF4 -- the census of the sends
# ---------------------------------------------------------------------------
def _strict_call(node):
    func = node.func if isinstance(node, ast.Call) else None
    return func is not None and (getattr(func, "id", None) or getattr(func, "attr", "")).endswith("strict_frame")


def _unstrict_sends(tree):
    """Each send of a frame that does not go through the strict frame, as (line, why)."""
    found = []
    stream = {node for fn in ast.walk(tree) if isinstance(fn, ast.AsyncFunctionDef) and fn.name == "agent_stream"
              for node in ast.walk(fn)}
    events = [fn for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) and fn.name == "_on_event"]
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr == "send_json" and not (node.args and _strict_call(node.args[0])):
            found.append((node.lineno, "send_json without the strict frame"))
        if node.func.attr in ("send_text", "send_bytes") and node not in stream:
            found.append((node.lineno, f"{node.func.attr} outside the agent stream"))
    for fn in events:
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "dumps"
                    and not (node.args and _strict_call(node.args[0]))):
                found.append((node.lineno, "an agent event written without the strict frame"))
    return found


def _package_sources():
    """Every Python file of the package but its plugins, in order.

    ``__pycache__`` and every directory named ``data`` are not walked: the
    package's data directory holds the maintainer's stores, never a module.
    """
    for folder, dirs, files in os.walk(_PACKAGE):
        top = Path(folder) == _PACKAGE
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", "data") and not (top and d == "plugins"))
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(folder, name)


def test_sf4_every_frame_a_socket_sends_goes_through_the_strict_frame():
    witness = ast.parse("async def f(ws, data):\n    await ws.send_json(data)\n    await ws.send_text('x')\n")
    assert len(_unstrict_sends(witness)) == 2, "control: the census names a send that is not strict"

    sending, unstrict = set(), {}
    for path in _package_sources():
        rel = path.relative_to(_PACKAGE).as_posix()
        text = path.read_text(encoding="utf-8")
        if "send_json" not in text and "send_text" not in text:
            continue
        tree = ast.parse(text)
        if any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") in ("send_json", "send_text")
               for n in ast.walk(tree)):
            sending.add(rel)
        found = _unstrict_sends(tree)
        if found:
            unstrict[rel] = found
    assert sending == _SENDING, ("the modules that send frames", sorted(sending))
    assert unstrict == {}, unstrict


# ---------------------------------------------------------------------------
# SF5 -- only what a strict parser cannot read changes
# ---------------------------------------------------------------------------
_TEXTS = ["", "plain", "h\u00e9", "\U0001f600 past the BMP", "tab\tand\nline", "quote \" and \\ backslash"]


def _drawn(rng, depth):
    """(payload, read): a value, and how a strict parser must read its strict frame."""
    kind = rng.randrange(9 if depth < 3 else 6)
    if kind == 0:
        value = rng.choice([float("nan"), float("inf"), float("-inf")])
        return value, json.dumps(value)
    if kind == 1:
        value = rng.choice([0, -7, 2 ** 53 + 1, 2 ** 70, 1.5, -0.25, 1e300, True, False, None])
        return value, value
    if kind == 2:
        text = rng.choice(_TEXTS)
        return text, text
    if kind == 3:
        lone = chr(rng.randrange(0xD800, 0xE000))
        head, tail = rng.choice(_TEXTS), rng.choice(_TEXTS)
        return head + lone + tail, head + f"\\u{ord(lone):04x}" + tail
    if kind == 4:
        return b"\x00raw", "b'\\x00raw'"
    if kind == 5:
        return {1, 2}, "{1, 2}"
    items = [_drawn(rng, depth + 1) for _ in range(rng.randrange(4))]
    if kind == 6:
        return [value for value, _ in items], [read for _, read in items]
    if kind == 7:
        return tuple(value for value, _ in items), [read for _, read in items]
    keys = rng.sample(["k", "h\u00e9", "\U0001f600", "x\udc80y", 3], len(items))
    shown = ["x\\udc80y" if key == "x\udc80y" else str(key) for key in keys]
    return ({key: value for key, (value, _) in zip(keys, items)},
            {key: read for key, (_, read) in zip(shown, items)})


def test_sf5_the_strict_frame_changes_only_what_a_strict_parser_cannot_read():
    frames, restore = _frames()
    try:
        fine = {"type": "token", "content": "h\u00e9 \U0001f600", "n": [1, 2.5, None, True], "deep": {"k": {"j": "v"}}}
        assert frames.strict_frame(fine) is fine, "a frame already strict comes back as itself"
        rng = random.Random(120)
        for _ in range(3000):
            payload, read = _drawn(rng, 0)
            text = json.dumps(frames.strict_frame(payload), allow_nan=False, ensure_ascii=False)
            text.encode("utf-8")
            assert _read(text) == read, (payload, text)
    finally:
        restore()
