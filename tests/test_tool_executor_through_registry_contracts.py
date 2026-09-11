#!/usr/bin/env python3
"""Contracts that the tool executor asks the registry at every head.

DESELECTED BY NAME in the canonical selection rule, and kept here as the
written specification of a migration that is owed. The tool executor is the
foundation the agentic block builds on, and it reaches the client library
directly at five sites: the native tool decision, the enum-forced decision,
the streamed final answer, its single-shot fallback, and the plain final
answer. None of those requests is admitted, none can be placed, and the
schema channel built for exactly this purpose cannot reach the one head that
needs it most.

Why it is owed rather than done: six sealed suites stand on the module's
direct path with hand-rolled windows that left the finder open, so a request
resolved through the registry from inside them reached the real backend and
tried a real connection -- measured, once, before those suites moved: forty
four contracts, twenty eight of them reading the scripted client. They now
stand on the shared window over the registry bridge, and these four are
selected again.

  * TE1 -- the native decision sends the tool list through the registry and
    reads the calls back off the response.
  * TE2 -- the forced decision sends its enum schema through the registry as
    the schema option -- the channel the previous block built.
  * TE3 -- the streamed final answer streams through the registry's backend.
  * TE4 -- with no backend, each head degrades as it documents and the client
    is never reached.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; the tool registry is the real one with no tools, the backend
registry and the client are stand-ins.
"""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {}}}]


class _RecorderBackend:
    def __init__(self):
        self.calls = []
        self.streams = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        options = kwargs.get("options") or {}
        if options.get("tools"):
            calls = [{"name": "search", "arguments": {"q": "x"}}]
            return SimpleNamespace(
                content="", tool_calls=calls,
                to_dict=lambda: {"message": {"content": "", "tool_calls": [
                    {"function": {"name": "search", "arguments": {"q": "x"}}}
                ]}},
            )
        if options.get("schema"):
            body = json.dumps({"tool_name": "search", "arguments": {"q": "x"}, "reasoning": "r"})
            return SimpleNamespace(content=body, tool_calls=[], to_dict=lambda: {"message": {"content": body}})
        return SimpleNamespace(content="final answer", tool_calls=[], to_dict=lambda: {"message": {"content": "final answer"}})

    def stream(self, **kwargs):
        self.streams.append(kwargs)
        yield SimpleNamespace(content="final ", thinking="", done=False)
        yield SimpleNamespace(content="answer", thinking="", done=True)


class _ForbiddenClient(types.ModuleType):
    def __init__(self):
        super().__init__("ollama")
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("direct client call")


def _load(*, backend):
    registry = SimpleNamespace(active=backend, resolve_backend=lambda model: backend)
    backends = types.ModuleType("opti_oignon.inference_backend")
    backends.get_backend_registry = lambda: registry
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(
        get_model=lambda *a, **k: "stand-in",
        get_temperature=lambda *a, **k: 0.3,
        get=lambda *a, **k: None,
    )
    client = _ForbiddenClient()
    had = "ollama" in sys.modules
    prev = sys.modules.get("ollama")
    sys.modules["ollama"] = client
    loaded, win_restore = isolate(
        targets={
            "opti_oignon.tool_calling": source("tool_calling.py"),
            "opti_oignon.tool_registry": source("tool_registry.py"),
            "opti_oignon.response_hygiene": source("response_hygiene.py"),
            "opti_oignon.structured_output": source("structured_output.py"),
            "opti_oignon.tool_executor": source("tool_executor.py"),
        },
        seeded={
            "opti_oignon.inference_backend": backends,
            "opti_oignon.config": cfg,
        },
        packages=("opti_oignon",),
    )

    def restore():
        win_restore()
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)

    return loaded["opti_oignon.tool_executor"], client, restore


_MESSAGES = [{"role": "user", "content": "find x"}]


# ---------------------------------------------------------------------------
# TE1 -- the native decision goes through the registry
# ---------------------------------------------------------------------------
def test_te1_the_native_decision_sends_tools_through_the_registry():
    backend = _RecorderBackend()
    mod, client, restore = _load(backend=backend)
    try:
        executor = mod.ToolExecutor(default_model="stand-in")
        calls = executor._native_tool_decision(_MESSAGES, "stand-in", _TOOLS)
        assert client.calls == [], "the client was never reached"
        assert len(backend.calls) == 1, "one request went through the registry"
        assert backend.calls[0]["options"].get("tools") == _TOOLS, (
            "the tool list travels as the tools option"
        )
        assert calls == [("search", {"q": "x"})], (
            "and the calls the model made come back parsed"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# TE2 -- the forced decision uses the schema channel
# ---------------------------------------------------------------------------
def test_te2_the_forced_decision_sends_its_schema_through_the_registry():
    backend = _RecorderBackend()
    mod, client, restore = _load(backend=backend)
    try:
        executor = mod.ToolExecutor(default_model="stand-in")
        forced = executor._enum_force_tool(_MESSAGES, "stand-in", ["search"])
        assert client.calls == [], "the client was never reached"
        assert len(backend.calls) == 1
        schema = backend.calls[0]["options"].get("schema")
        assert isinstance(schema, dict) and "search" in json.dumps(schema), (
            "the enum schema travels as the schema option, the channel built "
            "for constrained decoding"
        )
        assert forced == ("search", {"q": "x"}), "and the forced choice comes back"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TE3 -- the streamed final answer goes through the registry
# ---------------------------------------------------------------------------
def test_te3_the_streamed_final_answer_streams_through_the_registry():
    backend = _RecorderBackend()
    mod, client, restore = _load(backend=backend)
    try:
        executor = mod.ToolExecutor(default_model="stand-in")
        out = "".join(executor._stream_final_response(_MESSAGES, "stand-in"))
        assert client.calls == [], "the client was never reached"
        assert len(backend.streams) == 1, "the request was a stream"
        assert out == "final answer", "and the streamed chunks are the answer"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TE4 -- with no backend, each head degrades as documented
# ---------------------------------------------------------------------------
def test_te4_without_a_backend_each_head_degrades_and_never_reaches_the_client():
    mod, client, restore = _load(backend=None)
    try:
        executor = mod.ToolExecutor(default_model="stand-in")
        assert executor._native_tool_decision(_MESSAGES, "stand-in", _TOOLS) == [], (
            "no backend: the native decision has nothing to decide with"
        )
        assert executor._enum_force_tool(_MESSAGES, "stand-in", ["search"]) is None, (
            "no backend: no forced choice"
        )
        streamed = "".join(executor._stream_final_response(_MESSAGES, "stand-in"))
        assert "backend" in streamed.lower(), (
            "no backend: the streamed answer says so instead of inventing one"
        )
        assert client.calls == [], "and at no point was the client reached"
    finally:
        restore()
