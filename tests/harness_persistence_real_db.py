#!/usr/bin/env python3
"""Real-DB persistence harness (model-free).

Validates the v2.0.2 persistence fix against the *real* ConversationManager and
its real storage path (SQLCipher when available, plaintext sqlite otherwise),
without Ollama and without any thinking-capable model. For each pipeline it:

  1. creates a real conversation in a fresh temp database,
  2. drives the pipeline with faked inference engines (so persistence, not model
     quality, is exercised),
  3. reopens the database in a brand-new ConversationManager -- this simulates
     the page reload that exposed the bug,
  4. asserts the turn survived (and, for think+tools, that the Phase 2 tool
     output was persisted, not dropped).

This is the deterministic counterpart to the live-app test: it removes the
model-infrastructure variables (the coder model rejects think=True with a 400;
the MoE thinking model times out on RAM spill) that make think+tools awkward to
exercise through the UI.

Run from the repo root:  python3 tests/harness_persistence_real_db.py
On a machine with SQLCipher + a provisioned key it exercises the encrypted path;
elsewhere it falls back to plaintext sqlite (the persistence fix is orthogonal
to encryption, so the result is equally valid).
"""

import importlib.util
import sys
import types
from pathlib import Path
from tempfile import mkdtemp

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# Defensively stub ollama: the harness fakes all inference, so the real module
# must never be needed, but the conversation import chain is kept clean.
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

from opti_oignon.conversation import ConversationManager  # real class, real storage

_AE_PATH = _REPO / "opti_oignon" / "agentic_executor.py"


# ---------------------------------------------------------------------------
# Faked inference engines (same doubles as the unit test)
# ---------------------------------------------------------------------------
class _Result:
    def __init__(self, text):
        self.final_response = text
        self.model = "fake-model"
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
    def execute(self, **kwargs):
        # think+tools must suppress the Executor's own save
        assert kwargs.get("persist") is False, "expected execute(persist=False)"
        yield "reasoning part. "
        yield "more reasoning."


class _ToolResult:
    def __init__(self):
        self.tool_calls = ["write_file"]
        self.response = "TOOL_OUTPUT_BLOCK"


class FakeToolExec:
    def should_use_tools(self, message, model):
        return True

    def execute_with_tools(self, **kwargs):
        return _ToolResult()


# ---------------------------------------------------------------------------
# Load agentic_executor in isolation, but wire its conversation_manager to a
# *real* ConversationManager instance (so _save_to_conversation hits real I/O).
# ---------------------------------------------------------------------------
def _load_ae_with(real_cm):
    keys = ("opti_oignon", "opti_oignon.conversation", "opti_oignon.agentic_executor")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []  # guarded relative imports -> AVAILABLE=False
    sys.modules["opti_oignon"] = pkg
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = real_cm  # the real instance on the temp DB
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


def _conv_id(conv):
    for attr in ("id", "conversation_id", "uuid", "conv_id"):
        val = getattr(conv, attr, None)
        if val:
            return val
    raise RuntimeError("could not resolve conversation id attribute")


def _routing():
    return types.SimpleNamespace(model="m", task_type=None)


def _run(drive, enable_attr=None, **build_kwargs):
    """Drive one pipeline, then reopen the DB and return persisted messages."""
    db = Path(mkdtemp()) / "conv.db"
    cm = ConversationManager(db_path=db)
    conv = cm.create_conversation(title="harness", model="m")
    cid = _conv_id(conv)

    mod, restore = _load_ae_with(cm)
    try:
        if enable_attr:
            setattr(mod, enable_attr, True)
        agen = mod.AgenticExecutor(**build_kwargs)
        agen._get_conversation_context = lambda c: []
        agen.get_tool_history = lambda c: []
        agen._record_tool_calls = lambda c, t: None
        agen._emit_tool_call = lambda t: None
        drive(agen, cid)
    finally:
        restore()

    # Simulate the reload: a brand-new manager reading the same database file.
    reloaded = ConversationManager(db_path=db)
    return reloaded.get_messages(cid)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_cascading():
    msgs = _run(
        lambda a, c: list(a._execute_cascading_pipeline("q", _routing(), c, None)),
        enable_attr="CASCADING_INFERENCE_AVAILABLE",
        cascading_inference=FakeCascade(),
    )
    assert len(msgs) == 2, f"cascading: expected 2 persisted messages, got {len(msgs)}"
    assert msgs[1].content == "cascade answer"


def check_speculative():
    msgs = _run(
        lambda a, c: list(a._execute_speculative_pipeline("q", _routing(), c, None)),
        enable_attr="SPECULATIVE_GENERATION_AVAILABLE",
        speculative_generator=FakeSpeculative(),
    )
    assert len(msgs) == 2, f"speculative: expected 2 persisted messages, got {len(msgs)}"
    assert msgs[1].content == "speculative answer"


def check_think_tools():
    msgs = _run(
        lambda a, c: list(a._execute_think_tools_pipeline("do a thing", _routing(), c, None)),
        enable_attr="TOOL_EXECUTOR_AVAILABLE",
        executor=FakeExecutor(),
        tool_executor=FakeToolExec(),
    )
    assert len(msgs) == 2, f"think+tools: expected 2 persisted messages, got {len(msgs)}"
    asst = msgs[1].content
    assert "reasoning" in asst, "think+tools: reasoning not persisted"
    assert "TOOL_OUTPUT_BLOCK" in asst, "think+tools: Phase 2 tool output was dropped"


def check_tools():
    msgs = _run(
        lambda a, c: list(a._execute_tools_pipeline("q", _routing(), c, None)),
        enable_attr="TOOL_EXECUTOR_AVAILABLE",
        tool_executor=FakeToolExec(),
    )
    assert len(msgs) == 2, f"tools: expected 2 persisted messages, got {len(msgs)}"
    assert msgs[1].content == "TOOL_OUTPUT_BLOCK"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("check_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'OK - real-DB persistence validated' if failures == 0 else 'FAILED'} "
          f"- {failures} failure(s)")
    sys.exit(1 if failures else 0)
