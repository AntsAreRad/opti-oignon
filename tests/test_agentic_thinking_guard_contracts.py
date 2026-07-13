#!/usr/bin/env python3
"""Thinking-guard contracts: no think-emitting pipeline for a non-thinking model.

The pipeline selector carries a capability guard: when the effective model
cannot accept the native think switch, every think-emitting pipeline
(think, think+tools, reasoning) is steered to its plain counterpart, so
``think=True`` is never sent to a runner that would answer HTTP 400 and
drop the turn. When the gate primitive itself is absent, the pre-guard
default (thinking assumed available) is preserved so nothing regresses.
This suite pins that behavior:

  * TG1 -- a forced think with armed tools selects think+tools only while
    thinking is available, else the plain tools pipeline;
  * TG2 -- a forced think without tools selects think only while thinking
    is available, else the direct pipeline;
  * TG3 -- the auto reasoning route requires the thinking capability; a
    non-thinking model falls through to tools or direct;
  * TG4 -- the auto complex route degrades the same way, with and without
    armed tools;
  * TG5 -- end to end, execute() evaluates the gate on the model and never
    threads think=True to the executor when the capability is absent, and
    threads it when it is;
  * TG6 -- with the gate primitive unimportable, the historical default
    holds: a forced think still selects the think pipeline.

Loads the pipeline module in isolation under a stand-in package; every
``opti_oignon.*`` entry plus the model client entry is snapshotted and
evicted first, and only the stubs a clause needs are seeded. A meta-path
guard refuses any project submodule that was not seeded, so the load
behaves identically whether or not the project is installed (an editable
install resolves submodules by name and would otherwise bypass the
stand-in package). Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the test's
    back -- silently importing live code and reopening real databases. This
    guard sits ahead of every finder and refuses the names that were not
    seeded, so a load behaves identically whether the project is installed
    or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


def _load(seed=None):
    """Load agentic_executor.py under a stand-in package.

    ``seed`` maps short submodule names (e.g. ``"executor"``) to stub
    module objects registered as ``opti_oignon.<name>`` before the load,
    so the module's guarded relative imports resolve to them; everything
    not seeded raises ImportError and the availability flags stay False.
    """
    keys = ["ollama"] + [
        k
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    ]
    saved = {k: sys.modules[k] for k in keys if k in sys.modules}
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules["ollama"] = None  # any client import fails deterministically

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    sys.modules["opti_oignon"] = root
    for name, module in (seed or {}).items():
        full = f"opti_oignon.{name}"
        sys.modules[full] = module
        setattr(root, name, module)

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    def restore():
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        for k in list(sys.modules):
            if k == "opti_oignon" or k.startswith("opti_oignon."):
                del sys.modules[k]
        sys.modules.pop("ollama", None)
        for k, v in saved.items():
            sys.modules[k] = v

    full = "opti_oignon.agentic_executor"
    spec = importlib.util.spec_from_file_location(full, _OO / "agentic_executor.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    root.agentic_executor = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        restore()
        raise

    return mod, restore


class _RecorderExecutor:
    """Executor stand-in that records the kwargs of each execute call."""

    def __init__(self):
        self.calls = []

    def reset(self):
        pass

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        yield "ok"


def _executor_seed(recorder):
    """Stub modules satisfying the executor/router import block."""
    ex = types.ModuleType("opti_oignon.executor")
    ex.Executor = _RecorderExecutor
    ex.executor = recorder
    rt = types.ModuleType("opti_oignon.router")
    rt.RoutingResult = SimpleNamespace
    return {"executor": ex, "router": rt}


def _thinking_seed(verdict):
    """Stub gate primitive answering ``verdict`` for every model."""
    tc = types.ModuleType("opti_oignon.tool_calling")
    tc.model_supports_thinking = lambda model, lookup=None: verdict
    return {"tool_calling": tc}


# ---------------------------------------------------------------------------
# TG1 -- forced think with armed tools: think+tools only while available
# ---------------------------------------------------------------------------
def test_tg1_forced_think_with_tools_degrades_to_plain_tools():
    mod, restore = _load()
    try:
        classification = {"needs_tools": True}
        kept = mod._select_pipeline(
            classification=classification,
            think_override=True,
            web_search_override=None,
            tool_executor_available=True,
            verification_available=False,
            thinking_available=True,
        )
        assert kept == mod.PIPELINE_THINK_TOOLS, (
            f"a thinking-capable model must keep think+tools, got {kept}"
        )
        degraded = mod._select_pipeline(
            classification=classification,
            think_override=True,
            web_search_override=None,
            tool_executor_available=True,
            verification_available=False,
            thinking_available=False,
        )
        assert degraded == mod.PIPELINE_TOOLS, (
            "a non-thinking model with armed tools must degrade the forced "
            f"think to the plain tools pipeline, got {degraded}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# TG2 -- forced think without tools: think only while available
# ---------------------------------------------------------------------------
def test_tg2_forced_think_without_tools_degrades_to_direct():
    mod, restore = _load()
    try:
        classification = {"needs_tools": False}
        kept = mod._select_pipeline(
            classification=classification,
            think_override=True,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=False,
            thinking_available=True,
        )
        assert kept == mod.PIPELINE_THINK, (
            f"a thinking-capable model must keep the think pipeline, got {kept}"
        )
        degraded = mod._select_pipeline(
            classification=classification,
            think_override=True,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=False,
            thinking_available=False,
        )
        assert degraded == mod.PIPELINE_DIRECT, (
            "a non-thinking model without tools must degrade the forced "
            f"think to the direct pipeline, got {degraded}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# TG3 -- the auto reasoning route requires the thinking capability
# ---------------------------------------------------------------------------
def test_tg3_reasoning_route_requires_thinking():
    mod, restore = _load()
    try:
        classification = mod._quick_classify("please decompose the problem")
        assert classification["needs_reasoning"], (
            "fixture must hit the reasoning keywords"
        )
        assert not classification["needs_tools"], (
            "fixture must not hit the tool keywords"
        )
        assert not classification["is_complex"], (
            "fixture must not hit the complexity keywords"
        )
        kept = mod._select_pipeline(
            classification=classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=False,
            reasoning_available=True,
            thinking_available=True,
        )
        assert kept == mod.PIPELINE_REASONING, (
            f"a thinking-capable model must keep reasoning, got {kept}"
        )
        no_think = mod._select_pipeline(
            classification=classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=False,
            reasoning_available=True,
            thinking_available=False,
        )
        assert no_think == mod.PIPELINE_DIRECT, (
            "a non-thinking model must fall through the reasoning route to "
            f"direct, got {no_think}"
        )
        no_think_tools = mod._select_pipeline(
            classification=classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=True,
            verification_available=False,
            reasoning_available=True,
            thinking_available=False,
            capabilities_armed=True,
        )
        assert no_think_tools == mod.PIPELINE_TOOLS, (
            "a non-thinking model with armed capabilities must fall through "
            f"the reasoning route to plain tools, got {no_think_tools}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# TG4 -- the auto complex route degrades the same way
# ---------------------------------------------------------------------------
def test_tg4_complex_route_requires_thinking():
    mod, restore = _load()
    try:
        classification = {"is_complex": True}
        with_tools = mod._select_pipeline(
            classification=dict(classification, needs_tools=True),
            think_override=None,
            web_search_override=None,
            tool_executor_available=True,
            verification_available=False,
            thinking_available=False,
        )
        assert with_tools == mod.PIPELINE_TOOLS, (
            "a complex turn on a non-thinking model with armed tools must "
            f"select plain tools, got {with_tools}"
        )
        without_tools = mod._select_pipeline(
            classification=classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=False,
            thinking_available=False,
        )
        assert without_tools == mod.PIPELINE_DIRECT, (
            "a complex turn on a non-thinking model without tools must "
            f"select direct, got {without_tools}"
        )
        control = mod._select_pipeline(
            classification=classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=False,
            thinking_available=True,
        )
        assert control == mod.PIPELINE_THINK, (
            f"the thinking-capable control must keep think, got {control}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# TG5 -- execute() evaluates the gate on the model and threads its verdict
# ---------------------------------------------------------------------------
def _selector_recorder(mod, seen):
    """Replace the selector with a recorder that always picks direct.

    The selection outcomes themselves belong to the selector clauses; this
    probe pins only that execute() computes the capability verdict and
    threads it into the selection call.
    """
    original = mod._select_pipeline

    def _probe(**kwargs):
        seen.append(kwargs)
        return mod.PIPELINE_DIRECT

    mod._select_pipeline = _probe
    return lambda: setattr(mod, "_select_pipeline", original)


def test_tg5_execute_gates_think_on_the_model_capability():
    recorder = _RecorderExecutor()
    seed = dict(_executor_seed(recorder))
    seed.update(_thinking_seed(False))
    mod, restore = _load(seed=seed)
    try:
        seen = []
        undo = _selector_recorder(mod, seen)
        try:
            ae = mod.AgenticExecutor(executor=recorder, default_model="m")
            routing = SimpleNamespace(model="no-think-model")
            chunks = list(ae.execute("hello there", routing, think=True))
        finally:
            undo()
        assert "ok" in "".join(str(c) for c in chunks)
        assert seen and seen[-1].get("thinking_available") is False, (
            "the gate verdict for a non-thinking model must reach the "
            f"selection call as False, got {seen and seen[-1]}"
        )
        assert recorder.calls and recorder.calls[-1].get("think") is False, (
            "the executor must never receive think=True for a non-thinking "
            f"model, got {recorder.calls and recorder.calls[-1].get('think')}"
        )
    finally:
        restore()

    recorder2 = _RecorderExecutor()
    seed2 = dict(_executor_seed(recorder2))
    seed2.update(_thinking_seed(True))
    mod2, restore2 = _load(seed=seed2)
    try:
        seen2 = []
        undo2 = _selector_recorder(mod2, seen2)
        try:
            ae2 = mod2.AgenticExecutor(executor=recorder2, default_model="m")
            routing2 = SimpleNamespace(model="thinking-model")
            list(ae2.execute("hello there", routing2, think=True))
        finally:
            undo2()
        assert seen2 and seen2[-1].get("thinking_available") is True, (
            "the gate verdict for a thinking-capable model must reach the "
            f"selection call as True, got {seen2 and seen2[-1]}"
        )
    finally:
        restore2()


# ---------------------------------------------------------------------------
# TG6 -- an absent gate primitive preserves the historical default
# ---------------------------------------------------------------------------
def test_tg6_absent_gate_primitive_defaults_to_thinking_available():
    recorder = _RecorderExecutor()
    mod, restore = _load(seed=_executor_seed(recorder))
    try:
        assert mod.THINKING_GATE_AVAILABLE is False, (
            "the gate primitive must be unimportable in this load"
        )
        seen = []
        undo = _selector_recorder(mod, seen)
        try:
            ae = mod.AgenticExecutor(executor=recorder, default_model="m")
            routing = SimpleNamespace(model="any-model")
            list(ae.execute("hello there", routing, think=True))
        finally:
            undo()
        assert seen and seen[-1].get("thinking_available") is True, (
            "with the gate primitive absent the pre-guard default must hold "
            f"(thinking assumed available), got {seen and seen[-1]}"
        )
    finally:
        restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
