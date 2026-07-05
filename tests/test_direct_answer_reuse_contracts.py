#!/usr/bin/env python3
"""Contracts for single-pass reuse of the zero-call decision answer.

On the native function-calling path, a capable model that decides to call
no tool usually carries its direct answer in the decision content. Paying
a second generation to reproduce that answer doubles the latency of every
simple turn. The reuse must therefore happen -- and ONLY inside strict
guard rails, because a wrongly reused decision silently replaces the real
final synthesis. These contracts pin the boundary:

  * Contract 1 -- reuse at zero calls: when the native decision returns no
    tool call and carries usable content, that content IS the final
    answer and exactly ONE model generation happens for the whole turn.
  * Contract 2 -- an executed call forbids reuse: as soon as any tool ran
    this turn, the final synthesis over the results is generated as
    before; leftover decision content never leaks into the reply, and the
    captured candidate is consumed either way.
  * Contract 3 -- empty content degrades cleanly: a decision whose content
    is empty after hygiene falls back to the historical generation path,
    byte-for-byte.
  * Contract 4 -- the reuse honors its configuration switch: with the
    preference disabled, the historical two-generation path runs even for
    a perfectly reusable decision.
  * Contract 5 -- hygiene applies exactly once: internal scaffold lines in
    the reused content are stripped from the reply (never surfaced), and
    the turn still costs a single generation.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
contracts: canonical dotted names under a package stub, the real
tool-calling and response-hygiene primitives, and controllable stand-ins
for Ollama, the configuration and the mode probe.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
_OO = _ROOT / "opti_oignon"


def _pydantic_shim():
    mod = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for name in getattr(self.__class__, "__annotations__", {}):
                default = getattr(self.__class__, name, None)
                if isinstance(default, (list, dict)):
                    default = type(default)(default)
                setattr(self, name, default)
            for key, value in kwargs.items():
                setattr(self, key, value)

    mod.BaseModel = BaseModel
    return mod


# ---------------------------------------------------------------------------
# Scripted Ollama stand-in
# ---------------------------------------------------------------------------
class _ScriptedOllama:
    """Plays a fixed sequence of chat responses and records every call."""

    def __init__(self):
        self.script = []
        self.calls = []

    def load(self, *responses):
        self.script = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError(
                "ollama.chat called more times than the script allows"
            )
        return self.script.pop(0)


def _decision(content, calls=()):
    """A native-decision response: content plus optional tool calls."""
    tool_calls = [
        SimpleNamespace(function=SimpleNamespace(name=n, arguments=dict(a)))
        for n, a in calls
    ]
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
    )


def _generation(content):
    """A plain generation response (final answer or salvage candidate)."""
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=[]),
    )


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
_KEYS = (
    "pydantic", "ollama", "opti_oignon", "opti_oignon.tool_calling",
    "opti_oignon.tool_registry", "opti_oignon.structured_output",
    "opti_oignon.response_hygiene", "opti_oignon.tool_executor",
    "opti_oignon.security_mode", "opti_oignon.config",
)


def _load_executor(*, preferences=None):
    """Load the real tool executor chain with scripted seams.

    Returns ``(tool_executor_module, ollama_stub, restore)``.
    """
    saved = {k: sys.modules.get(k) for k in _KEYS}

    try:
        import pydantic  # noqa: F401
    except ImportError:
        sys.modules["pydantic"] = _pydantic_shim()

    ollama_stub = types.ModuleType("ollama")
    scripted = _ScriptedOllama()
    ollama_stub.chat = scripted.chat
    sys.modules["ollama"] = ollama_stub

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    sm = types.ModuleType("opti_oignon.security_mode")
    sm.is_bulbe = lambda: False
    sys.modules["opti_oignon.security_mode"] = sm
    pkg.security_mode = sm

    prefs = dict(preferences or {})
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(
        get_user_preference=lambda key, default=None: prefs.get(key, default),
    )
    cfg.get_model = lambda *a, **k: "scripted-model"
    sys.modules["opti_oignon.config"] = cfg
    pkg.config = cfg

    so = types.ModuleType("opti_oignon.structured_output")
    so.StructuredOutputEngine = object
    so.ToolCallRequest = object
    sys.modules["opti_oignon.structured_output"] = so
    pkg.structured_output = so

    def _real(dotted, path):
        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg.tool_calling = _real(
        "opti_oignon.tool_calling", _OO / "tool_calling.py",
    )
    pkg.response_hygiene = _real(
        "opti_oignon.response_hygiene", _OO / "response_hygiene.py",
    )
    tr = _real("opti_oignon.tool_registry", _OO / "tool_registry.py")
    pkg.tool_registry = tr

    te = _real("opti_oignon.tool_executor", _OO / "tool_executor.py")
    pkg.tool_executor = te

    # Every scripted model is native-capable for these contracts.
    te.model_supports_native_tools = lambda model, capability_lookup=None: True

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return te, scripted, restore


def _make_executor(te):
    """A ToolExecutor over a fresh registry carrying one local echo tool."""
    tr = sys.modules["opti_oignon.tool_registry"]
    registry = tr.ToolRegistry()
    registry.register(tr.ToolDefinition(
        name="echo",
        description="Echo the call back. Local test tool.",
        parameters={},
        handler=lambda **kwargs: "echoed",
    ))
    return te.ToolExecutor(registry=registry)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
def test_c1_zero_call_decision_content_is_the_answer_single_generation():
    te, scripted, restore = _load_executor()
    try:
        executor = _make_executor(te)
        scripted.load(_decision("Direct answer text.", calls=()))
        result = executor.execute_with_tools("hello", model="scripted")
        assert result.response == "Direct answer text.", result.response
        assert result.tool_calls == [], "no tool should have run"
        assert len(scripted.calls) == 1, (
            f"expected a single generation, saw {len(scripted.calls)}"
        )
        assert executor._pending_direct_answer is None, "candidate not consumed"
    finally:
        restore()


def test_c2_an_executed_call_forbids_reuse_and_keeps_the_synthesis():
    te, scripted, restore = _load_executor()
    try:
        executor = _make_executor(te)
        scripted.load(
            _decision("", calls=(("echo", {}),)),
            _decision("Leftover narration.", calls=()),
            _generation("Synthesized final."),
        )
        result = executor.execute_with_tools("do it", model="scripted")
        assert [c.tool_name for c in result.tool_calls] == ["echo"]
        assert result.response == "Synthesized final.", result.response
        assert "Leftover" not in result.response
        assert len(scripted.calls) == 3, len(scripted.calls)
        assert executor._pending_direct_answer is None, "candidate not consumed"
    finally:
        restore()


def test_c3_empty_after_hygiene_degrades_to_the_historical_path():
    te, scripted, restore = _load_executor()
    try:
        executor = _make_executor(te)
        scripted.load(
            _decision("  \n\t ", calls=()),
            _generation("Salvage answer."),
        )
        result = executor.execute_with_tools("hello", model="scripted")
        assert result.response == "Salvage answer.", result.response
        assert result.tool_calls == []
        assert len(scripted.calls) == 2, len(scripted.calls)
        assert executor._pending_direct_answer is None
    finally:
        restore()


def test_c4_the_configuration_switch_disables_the_reuse():
    te, scripted, restore = _load_executor(
        preferences={"tool_direct_answer_reuse": False},
    )
    try:
        executor = _make_executor(te)
        assert hasattr(executor, "_pending_direct_answer"), (
            "the reuse seam does not exist"
        )
        scripted.load(
            _decision("Would be reused.", calls=()),
            _generation("Fresh answer."),
        )
        result = executor.execute_with_tools("hello", model="scripted")
        assert result.response == "Fresh answer.", result.response
        assert len(scripted.calls) == 2, len(scripted.calls)
    finally:
        restore()


def test_c5_scaffold_lines_never_surface_and_the_turn_stays_single_pass():
    te, scripted, restore = _load_executor()
    try:
        executor = _make_executor(te)
        content = "[environment] leftover scaffold\nReal answer."
        scripted.load(_decision(content, calls=()))
        result = executor.execute_with_tools("hello", model="scripted")
        assert result.response == "Real answer.", result.response
        assert "scaffold" not in result.response
        assert len(scripted.calls) == 1, len(scripted.calls)
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
_TESTS = (
    "test_c1_zero_call_decision_content_is_the_answer_single_generation",
    "test_c2_an_executed_call_forbids_reuse_and_keeps_the_synthesis",
    "test_c3_empty_after_hygiene_degrades_to_the_historical_path",
    "test_c4_the_configuration_switch_disables_the_reuse",
    "test_c5_scaffold_lines_never_surface_and_the_turn_stays_single_pass",
)


def _main() -> int:
    passed = 0
    for name in _TESTS:
        try:
            globals()[name]()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
            passed += 1
    print(f"{passed}/{len(_TESTS)} passed")
    return 0 if passed == len(_TESTS) else 1


if __name__ == "__main__":
    sys.exit(_main())
