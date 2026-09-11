#!/usr/bin/env python3
"""Contracts for the eval harness's seam into the tool executor.

The chat runner used to route a scripted client into the executor by
rebinding a module-level name on the executor module. That binding is gone
with the executor's direct client path; the runner now lays a scripted
backend over the inference registry for the block and takes it away after.
The property is the same one the tool suites hold on the hub: the request
reaches the scripted client, through the registry, and never any other way.

  * CR1 -- inside the block, the executor's final head reaches the scripted
    client through the registry's active backend, as a stream, and the
    executor module carries no client of its own.
  * CR2 -- after the block, the overlay is gone and the previous backend
    serves again.
  * CR3 -- a native decision inside the block forwards the tools and gets
    the scripted calls back.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window over the registry bridge.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

_MSGS = [{"role": "user", "content": "find x"}]
_TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {}}}]


class _Idle:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([{"message": {"content": "idle"}, "done": True}])
        return {"message": {"content": "idle"}}


def _open():
    so = types.ModuleType("opti_oignon.structured_output")
    so.StructuredOutputEngine = object
    so.ToolCallRequest = object
    so.structured_output_engine = None
    so.STRUCTURED_OUTPUT_AVAILABLE = False
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(get_model=lambda *a, **k: "m", get_temperature=lambda *a, **k: 0.3,
                                 get=lambda *a, **k: None, get_user_preference=lambda k, d=None: d)
    cfg.get_model = lambda *a, **k: "m"
    seeded = {"opti_oignon.structured_output": so, "opti_oignon.config": cfg}
    idle = _Idle()
    first = seed_registry(seeded, idle)
    loaded, restore = isolate(
        targets={
            "opti_oignon.tool_calling": source("tool_calling.py"),
            "opti_oignon.tool_registry": source("tool_registry.py"),
            "opti_oignon.response_hygiene": source("response_hygiene.py"),
            "opti_oignon.tool_executor": source("tool_executor.py"),
            "opti_oignon.agent_eval.tasks": source("agent_eval", "tasks.py"),
            "opti_oignon.agent_eval.chat_runner": source("agent_eval", "chat_runner.py"),
        },
        seeded=seeded,
        packages=("opti_oignon", "opti_oignon.agent_eval"),
    )
    cr = loaded["opti_oignon.agent_eval.chat_runner"]
    te = loaded["opti_oignon.tool_executor"]
    registry = seeded["opti_oignon.inference_backend"].get_backend_registry()
    assert cr.FEATURE_AVAILABLE, "harness reports FEATURE_AVAILABLE False"
    return cr, te, registry, first, idle, restore


# ---------------------------------------------------------------------------
# CR1 -- inside the block
# ---------------------------------------------------------------------------
def test_cr1_inside_the_block_the_final_head_reaches_the_scripted_client_through_the_registry():
    cr, te, registry, first, idle, restore = _open()
    try:
        client = cr.ScriptedChatClient([cr.ScriptTurn(content="final answer")])
        with cr.scripted_chat_backend(client):
            assert registry.active is not first, "the overlay is the active backend"
            text = "".join(te.ToolExecutor._stream_final_response(SimpleNamespace(), _MSGS, "m"))
        assert text == "final answer"
        assert len(client.calls) == 1 and client.calls[0]["stream"] is True
        assert client.calls[0]["roles"] == ["user"]
        assert idle.calls == [], "the previous backend saw nothing"
        assert not hasattr(te, "ollama"), "the executor module carries no client of its own"
        assert not hasattr(te, "OLLAMA_AVAILABLE")
    finally:
        restore()


# ---------------------------------------------------------------------------
# CR2 -- after the block
# ---------------------------------------------------------------------------
def test_cr2_after_the_block_the_overlay_is_gone_and_the_previous_backend_serves():
    cr, te, registry, first, idle, restore = _open()
    try:
        client = cr.ScriptedChatClient([cr.ScriptTurn(content="x")])
        with cr.scripted_chat_backend(client):
            overlay = registry.active
        assert registry.active is first and registry.resolve_backend("m") is first
        assert registry.get(overlay.name) is None, "unregistered, not merely deactivated"
        text = "".join(te.ToolExecutor._stream_final_response(SimpleNamespace(), _MSGS, "m"))
        assert text == "idle" and len(idle.calls) == 1, "the previous backend serves again"
        assert client.leftover == 1, "the scripted client was not consumed after the block"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CR3 -- a native decision in the block
# ---------------------------------------------------------------------------
def test_cr3_a_native_decision_in_the_block_forwards_the_tools_and_gets_the_calls_back():
    cr, te, registry, first, idle, restore = _open()
    try:
        client = cr.ScriptedChatClient([cr.ScriptTurn(content="", tool_calls=[{"name": "search", "arguments": {"q": "x"}}])])
        with cr.scripted_chat_backend(client):
            holder = SimpleNamespace()
            calls = te.ToolExecutor._native_tool_decision(holder, _MSGS, "m", _TOOLS)
        assert calls == [("search", {"q": "x"})]
        assert client.calls[0]["tools_param"] is True and client.calls[0]["stream"] is False
        assert holder._last_native_response is not None, "the response is kept for the direct-answer path"
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
