#!/usr/bin/env python3
"""Regression: every non-Executor pipeline must persist its turn.

Background
----------
The published v2.0.1 dropped every agentic turn: ``_save_to_conversation``
called ``add_message(conversation_id=...)`` while the manager method takes
``conv_id``; the resulting ``TypeError`` was swallowed by a DEBUG ``except``,
so conversations were empty on reload and the token counter showed 0.

The fix corrected the keyword, but the *class* of bug remained: every pipeline
that does not run through the Executor must call ``_save_to_conversation``
itself, and nothing enforced it. The persistence audit found three further
gaps:
  * cascading (S69) never persisted -> whole turn dropped.
  * speculative (S70) never persisted -> whole turn dropped.
  * think+tools persisted only the reasoning (via the Executor); the Phase 2
    tool-output block was dropped on reload.

This test would have caught all of them: drive each pipeline and assert the
conversation is non-empty (and, for think+tools, that the tool output is
actually persisted).

Isolation
---------
The pipelines are exercised without Ollama or SQLCipher. ``agentic_executor``
is loaded directly via ``importlib`` under its real package name (so the lazy
``from .conversation import conversation_manager`` resolves), with the package
stubbed so the guarded relative imports degrade to the AVAILABLE=False path.
A fake conversation manager records ``add_message`` calls and exposes
``get_messages``. The per-pipeline AVAILABLE module globals are flipped on and
fake engines are injected, so each pipeline reaches its persistence path.
sys.modules entries are saved and restored around every load.
"""

import importlib.util
import sys
import types
from pathlib import Path

_AE_PATH = Path(__file__).resolve().parent.parent / "opti_oignon" / "agentic_executor.py"

CONV = "conv-regression"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class FakeConvManager:
    """Records add_message calls; mirrors the conv_id/get_messages API."""

    def __init__(self) -> None:
        self._store: dict = {}

    def add_message(self, conv_id=None, role=None, content=None, model=None, metadata=None):
        self._store.setdefault(conv_id, []).append(
            {"role": role, "content": content, "model": model, "metadata": metadata}
        )
        return {"ok": True}

    def get_messages(self, conv_id):
        return list(self._store.get(conv_id, []))


class _Result:
    """Stands in for CascadeResult / SpeculativeResult."""

    def __init__(self, text: str) -> None:
        self.final_response = text
        self.model = "fake-model"
        # Fields touched only by the on_status branch (skipped: on_status=None)
        self.tier_index = 0
        self.tier_name = "fast"
        self.score = 0.9
        self.total_latency_ms = 1.0
        self.draft_accepted = True
        self.iterations = 1
        self.convergence_score = 1.0


class FakeCascade:
    enabled = True

    def cascade(self, query, task_type=None):
        return _Result("cascade answer")


class FakeSpeculative:
    enabled = True

    def generate(self, query, task_type=None):
        return _Result("speculative answer")


class FakeExecutor:
    """Mimics Executor.execute as a streaming generator.

    Records the kwargs it was called with and, like a persist=False call,
    saves nothing itself.
    """

    def __init__(self) -> None:
        self.calls: list = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        yield "reasoning part. "
        yield "more reasoning."


class _ToolResult:
    def __init__(self) -> None:
        self.tool_calls = ["write_file"]
        self.response = "TOOL_OUTPUT_BLOCK"


class FakeToolExec:
    def should_use_tools(self, message, model):
        return True

    def execute_with_tools(self, **kwargs):
        return _ToolResult()


# ---------------------------------------------------------------------------
# Isolated module loading (with save/restore of sys.modules)
# ---------------------------------------------------------------------------
def _install_stubs(fake_cm):
    keys = ("opti_oignon", "opti_oignon.conversation", "opti_oignon.agentic_executor")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []  # guarded relative imports -> AVAILABLE=False
    sys.modules["opti_oignon"] = pkg

    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = fake_cm
    sys.modules["opti_oignon.conversation"] = conv

    spec = importlib.util.spec_from_file_location("opti_oignon.agentic_executor", _AE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.agentic_executor"] = mod
    spec.loader.exec_module(mod)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


def _routing():
    return types.SimpleNamespace(model="m", task_type=None)


def _isolate(agen, mod):
    """Stub AgenticExecutor helpers unrelated to persistence."""
    agen._get_conversation_context = lambda cid: []
    agen.get_tool_history = lambda cid: []
    agen._record_tool_calls = lambda cid, tc: None
    agen._emit_tool_call = lambda tc: None
    return agen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_cascading_pipeline_persists_turn():
    cm = FakeConvManager()
    mod, restore = _install_stubs(cm)
    try:
        mod.CASCADING_INFERENCE_AVAILABLE = True
        agen = mod.AgenticExecutor(cascading_inference=FakeCascade())
        list(agen._execute_cascading_pipeline("q", _routing(), CONV, None))
        msgs = cm.get_messages(CONV)
    finally:
        restore()
    assert len(msgs) == 2, "cascading pipeline must persist user + assistant"
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "cascade answer"
    assert msgs[1]["model"] == "fake-model"


def test_speculative_pipeline_persists_turn():
    cm = FakeConvManager()
    mod, restore = _install_stubs(cm)
    try:
        mod.SPECULATIVE_GENERATION_AVAILABLE = True
        agen = mod.AgenticExecutor(speculative_generator=FakeSpeculative())
        list(agen._execute_speculative_pipeline("q", _routing(), CONV, None))
        msgs = cm.get_messages(CONV)
    finally:
        restore()
    assert len(msgs) == 2, "speculative pipeline must persist user + assistant"
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "speculative answer"
    assert msgs[1]["model"] == "fake-model"


def test_think_tools_pipeline_persists_tool_output():
    cm = FakeConvManager()
    mod, restore = _install_stubs(cm)
    try:
        mod.TOOL_EXECUTOR_AVAILABLE = True
        fake_exec = FakeExecutor()
        agen = mod.AgenticExecutor(executor=fake_exec, tool_executor=FakeToolExec())
        _isolate(agen, mod)
        list(agen._execute_think_tools_pipeline("do a thing", _routing(), CONV, None))
        msgs = cm.get_messages(CONV)
        exec_calls = fake_exec.calls
    finally:
        restore()
    # Phase 1 must suppress the Executor's own save to avoid duplication.
    assert exec_calls and exec_calls[0].get("persist") is False, (
        "think+tools must call Executor.execute(persist=False)"
    )
    assert len(msgs) == 2, "think+tools must persist user + assistant exactly once"
    assert msgs[0]["role"] == "user"
    asst = msgs[1]["content"]
    assert "reasoning" in asst, "reasoning must be persisted"
    assert "TOOL_OUTPUT_BLOCK" in asst, "Phase 2 tool output must be persisted (not dropped)"


def test_tools_pipeline_persists_turn():
    # Guard against regression of the original v2.0.1 fix.
    cm = FakeConvManager()
    mod, restore = _install_stubs(cm)
    try:
        mod.TOOL_EXECUTOR_AVAILABLE = True
        agen = mod.AgenticExecutor(tool_executor=FakeToolExec())
        _isolate(agen, mod)
        list(agen._execute_tools_pipeline("q", _routing(), CONV, None))
        msgs = cm.get_messages(CONV)
    finally:
        restore()
    assert len(msgs) == 2, "tools pipeline must persist user + assistant"
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "TOOL_OUTPUT_BLOCK"


if __name__ == "__main__":
    # Standalone runner (no pytest required): runs every test_* in this module.
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001 - surface harness errors too
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'OK' if failures == 0 else 'FAILED'} - {failures} failure(s)")
    sys.exit(1 if failures else 0)
