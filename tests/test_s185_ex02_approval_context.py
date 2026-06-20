"""S185 audit fix EX-02 -- tool-approval gate bound to the execution context.

The Bulbe per-call approval gate used to be a single mutable attribute,
``pre_tool_call_hook``, on the process-wide ``ToolExecutor`` singleton.
routes_chat installed a per-request closure onto that shared attribute before
the generation thread and reset it to ``None`` on completion. Two overlapping
Bulbe sessions (two chat tabs -> two WebSocket connections -> two generation
threads) could therefore:

  - clobber: session B's install overwrites session A's hook, so A's tool calls
    submit approval to B's conversation and emit events into B's stream;
  - drop: when one session finishes it reset the attribute to ``None``, so an
    in-flight session's next ``_execute_tool`` read ``None`` and executed the
    tool with no approval at all, in Bulbe.

The fix threads a per-invocation ``approval_fn`` through
``execute_with_tools -> _execute_tool``; routes_chat passes the request closure
into the executor call rather than mutating the singleton. The legacy
``pre_tool_call_hook`` attribute is kept as a process-wide fallback (EX-01 /
S128 backward compatibility). Resolution precedence in ``_execute_tool``:
explicit ``approval_fn`` > ``pre_tool_call_hook``.

The module is loaded in isolation: ``ollama`` is stubbed and ``opti_oignon`` is
a bare module, so the optional structured-output / tool-registry imports are
absent (their FEATURE flags read False). ``_execute_tool`` is driven against a
fake registry; no LLM and no network are touched.
"""

import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH = _REPO_ROOT / "opti_oignon" / "tool_executor.py"


def _load():
    spec = importlib.util.spec_from_file_location("tool_executor_ex02", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12 dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


te_mod = _load()


# ---------------------------------------------------------------------------
# Minimal fake registry: one always-available tool with a trivial handler.
# ---------------------------------------------------------------------------

class _FakeTool:
    def __init__(self, handler):
        self.enabled = True
        self.handler = handler
        self.parameters = {}


class _FakeRegistry:
    def __init__(self, handler):
        self._tool = _FakeTool(handler)

    def get(self, name):
        return self._tool

    def get_tools_prompt(self):
        # Non-empty so execute_with_tools enters the ReAct loop.
        return "available tools: t"


def _ok_handler(**kwargs):
    return "executed"


def _denies(name, args):
    return False


def _allows(name, args):
    return True


def _raises(name, args):
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Explicit per-invocation approval_fn: deny / allow / fail-secure
# ---------------------------------------------------------------------------

def test_explicit_approval_fn_denies():
    te = te_mod.ToolExecutor(registry=_FakeRegistry(_ok_handler))
    result = te._execute_tool("t", {"q": "x"}, approval_fn=_denies)
    assert result.success is False
    assert "denied" in result.result.lower()


def test_explicit_approval_fn_allows():
    te = te_mod.ToolExecutor(registry=_FakeRegistry(_ok_handler))
    result = te._execute_tool("t", {"q": "x"}, approval_fn=_allows)
    assert result.success is True
    assert result.result == "executed"


def test_explicit_approval_fn_raising_is_failsecure():
    # AP-01: a hook that raises is a denial, not an execution.
    te = te_mod.ToolExecutor(registry=_FakeRegistry(_ok_handler))
    result = te._execute_tool("t", {"q": "x"}, approval_fn=_raises)
    assert result.success is False
    assert (
        "denied" in result.result.lower()
        or "approval error" in result.result.lower()
    )


# ---------------------------------------------------------------------------
# Precedence: explicit approval_fn wins over the legacy singleton attribute
# ---------------------------------------------------------------------------

def test_explicit_fn_overrides_singleton_allow_to_deny():
    te = te_mod.ToolExecutor(registry=_FakeRegistry(_ok_handler))
    te.pre_tool_call_hook = _allows  # legacy global would approve
    result = te._execute_tool("t", {}, approval_fn=_denies)
    assert result.success is False  # explicit deny wins


def test_explicit_fn_overrides_singleton_deny_to_allow():
    te = te_mod.ToolExecutor(registry=_FakeRegistry(_ok_handler))
    te.pre_tool_call_hook = _denies  # legacy global would deny
    result = te._execute_tool("t", {}, approval_fn=_allows)
    assert result.success is True  # explicit allow wins


# ---------------------------------------------------------------------------
# Drop scenario: clearing the singleton must not ungate an explicit gate
# ---------------------------------------------------------------------------

def test_clearing_singleton_does_not_ungate_explicit_gate():
    # The old defect: one session setting the shared attribute to None ungated
    # an in-flight session. With a per-call gate, a None singleton is irrelevant.
    te = te_mod.ToolExecutor(registry=_FakeRegistry(_ok_handler))
    te.pre_tool_call_hook = None
    result = te._execute_tool("t", {}, approval_fn=_denies)
    assert result.success is False


# ---------------------------------------------------------------------------
# Backward compatibility: the legacy singleton is still honored as a fallback
# ---------------------------------------------------------------------------

def test_legacy_singleton_fallback_when_no_explicit_fn():
    te = te_mod.ToolExecutor(registry=_FakeRegistry(_ok_handler))
    te.pre_tool_call_hook = _denies
    result = te._execute_tool("t", {})  # no explicit approval_fn
    assert result.success is False
    te.pre_tool_call_hook = _allows
    result = te._execute_tool("t", {})
    assert result.success is True


# ---------------------------------------------------------------------------
# Core regression: two overlapping sessions do not share the gate
# ---------------------------------------------------------------------------

def test_concurrent_sessions_do_not_share_the_gate():
    """Two overlapping sessions, each gated by its own approval_fn.

    With the old shared-singleton design the gate was a single attribute, so a
    second session's install clobbered the first and a session finishing reset
    it to None. Binding the gate to the call makes each thread's decision
    independent and deterministic even under contention: the denying session
    never executes a tool, the allowing session always does.
    """
    handler_calls = {"n": 0}
    handler_lock = threading.Lock()

    def handler(**kwargs):
        with handler_lock:
            handler_calls["n"] += 1
        return "executed"

    te = te_mod.ToolExecutor(registry=_FakeRegistry(handler))

    iterations = 40
    barrier = threading.Barrier(2)
    results: dict[str, list[bool]] = {"deny": [], "allow": []}

    def slow_deny(name, args):
        time.sleep(0.001)  # widen the overlap window between the two threads
        return False

    def slow_allow(name, args):
        time.sleep(0.001)
        return True

    def run(label, fn):
        local: list[bool] = []
        barrier.wait()
        for _ in range(iterations):
            r = te._execute_tool("t", {}, approval_fn=fn)
            local.append(r.success)
        results[label] = local

    ta = threading.Thread(target=run, args=("deny", slow_deny))
    tb = threading.Thread(target=run, args=("allow", slow_allow))
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    assert results["deny"] == [False] * iterations
    assert results["allow"] == [True] * iterations
    # The handler was reached exactly once per approved call, never for denied.
    assert handler_calls["n"] == iterations


# ---------------------------------------------------------------------------
# Wiring: execute_with_tools threads approval_fn down to _execute_tool
# ---------------------------------------------------------------------------

def test_execute_with_tools_forwards_approval_fn():
    captured: dict[str, object] = {}
    te = te_mod.ToolExecutor(registry=_FakeRegistry(_ok_handler))

    # Drive one tool decision ("t"), then "none" to end the ReAct loop.
    calls = {"n": 0}

    class _Result:
        def __init__(self, data):
            self.success = True
            self.data = data

    class _FakeEngine:
        def generate_structured(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Result(
                    te_mod.ToolDecision(tool_name="t", arguments={}, reasoning="r")
                )
            return _Result(te_mod.ToolDecision(tool_name="none"))

    sentinel = _allows
    orig_execute_tool = te._execute_tool

    def _spy(tool_name, arguments, reasoning="", approval_fn=None):
        captured["approval_fn"] = approval_fn
        return orig_execute_tool(
            tool_name, arguments, reasoning, approval_fn=approval_fn
        )

    te.structured_engine = _FakeEngine()
    te._execute_tool = _spy

    saved_flag = te_mod.STRUCTURED_OUTPUT_AVAILABLE
    te_mod.STRUCTURED_OUTPUT_AVAILABLE = True
    try:
        te.execute_with_tools("hello", model="m", approval_fn=sentinel)
    finally:
        te_mod.STRUCTURED_OUTPUT_AVAILABLE = saved_flag

    assert captured.get("approval_fn") is sentinel
