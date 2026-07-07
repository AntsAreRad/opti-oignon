#!/usr/bin/env python3
"""Contracts for tool-bound capability enforcement in the agentic executor.

A tool-bound turn must run on a model that can actually call tools. The
executor owns a small guard that either keeps an already-capable model
untouched or re-selects a capable one through the router, and it is wired
into execute() at the point where the tool-bound model is chosen -- and
only for tool-bound turns whose tools are truly reachable. These contracts
pin that behaviour without a live runtime:

  * Contract A1 -- GUARD ENFORCES: for a non-capable model the guard defers
    to the router's tool-capable selection; a router refusal propagates
    unchanged (fail-secure), and a router-provided capable model is
    returned.
  * Contract A2 -- NO PERTURBATION: for an already-capable model the guard
    returns it unchanged and never calls the router. A correctly routed
    tool turn is not disturbed.
  * Contract A3 -- EXECUTE WIRES IT, TOOL-BOUND ONLY: driving execute()
    end to end,
      (i)  a tool-bound turn on a non-capable model, with tools reachable
           and the router refusing, makes execute() refuse explicitly (the
           router error propagates out of the generator); the tools
           pipeline is never reached.
      (ii) a keywordless, non-tool-bound turn on the same non-capable model
           is NOT enforced: the router is never consulted and the direct
           pipeline runs. This is the sentinel proving the guard does not
           leak into ordinary chat.

The capability verdict is the manifest's own predicate, stubbed here so a
model name decides its own verdict; the guard must consult it rather than
reimplement the rule, and must consult the router (not a local table) to
re-select.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
idiom: the executor is loaded alone with every sibling import failing soft,
then the capability predicate and the router are supplied as controllable
stand-ins so the guard's lazy imports resolve against them.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

TOOL_BOUND = "please calculate 17 times 23"      # needs_tools, not web
KEYWORDLESS = "Tell me about the aqueducts of ancient Lyon."  # no heuristic


class _RouterRefusal(RuntimeError):
    """Sentinel standing in for the router's tool-capable refusal."""


class _StubRouter:
    """Router stand-in recording calls; returns a model or refuses."""

    def __init__(self, *, result_model=None, refuse=False):
        self._result_model = result_model
        self._refuse = refuse
        self.calls = []

    def select_tool_capable_model(self, *, model_type, task_type,
                                  priority="balanced"):
        self.calls.append(
            {"model_type": model_type, "task_type": task_type,
             "priority": priority}
        )
        if self._refuse:
            raise _RouterRefusal("no tool-capable model available")
        return self._result_model


_KEYS = (
    "opti_oignon", "opti_oignon.agentic_executor",
    "opti_oignon.capability_manifest", "opti_oignon.router",
)


def _load_agentic_with_stubs(*, capable_names, stub_router):
    """Load the executor alone; supply the capability predicate + router.

    ``capable_names`` is the set the stubbed predicate calls tool-capable;
    ``stub_router`` backs the ``router`` object the guard re-selects
    through. Every other sibling import fails soft, so the manifest system
    is absent (legacy path) and no live runtime is touched. Returns
    ``(ae_module, restore)``.
    """
    saved = {k: sys.modules.get(k) for k in _KEYS}
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.agentic_executor", _OO / "agentic_executor.py",
    )
    ae = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.agentic_executor"] = ae
    spec.loader.exec_module(ae)
    pkg.agentic_executor = ae

    capable = set(capable_names)
    cm = types.ModuleType("opti_oignon.capability_manifest")
    cm.model_tool_capable = lambda name: name in capable
    sys.modules["opti_oignon.capability_manifest"] = cm
    pkg.capability_manifest = cm

    rt = types.ModuleType("opti_oignon.router")
    rt.router = stub_router
    sys.modules["opti_oignon.router"] = rt
    pkg.router = rt

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return ae, restore


def _routing(model):
    return types.SimpleNamespace(
        model=model, model_type="general", task_type="code_python",
    )


# ---------------------------------------------------------------------------
# Contract A1 -- the guard defers to the router for a non-capable model
# ---------------------------------------------------------------------------
def test_a1_guard_enforces_for_non_capable_model():
    # Router refuses -> the refusal propagates unchanged (fail-secure).
    refusing = _StubRouter(refuse=True)
    ae, restore = _load_agentic_with_stubs(
        capable_names=["cap-model"], stub_router=refusing,
    )
    try:
        guard = getattr(ae, "_ensure_tool_capable_model", None)
        assert callable(guard), (
            "the executor must expose _ensure_tool_capable_model as the "
            "tool-bound capability guard"
        )
        raised = None
        try:
            guard("non-capable", _routing("non-capable"))
        except Exception as caught:  # noqa: BLE001 -- assert the type below
            raised = caught
        assert isinstance(raised, _RouterRefusal), (
            "a router refusal for a non-capable model must propagate "
            f"unchanged, got {type(raised).__name__ if raised else None}"
        )
        assert refusing.calls, "the guard must consult the router to re-select"

        # Router returns a capable model -> the guard returns it.
        providing = _StubRouter(result_model="cap-model")
        ae2, restore2 = _load_agentic_with_stubs(
            capable_names=["cap-model"], stub_router=providing,
        )
        try:
            picked = ae2._ensure_tool_capable_model(
                "non-capable", _routing("non-capable"),
            )
            assert picked == "cap-model", (
                f"the guard must return the router's capable model, got {picked}"
            )
            assert providing.calls, "the guard must consult the router"
        finally:
            restore2()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract A2 -- an already-capable model is not perturbed
# ---------------------------------------------------------------------------
def test_a2_guard_leaves_capable_model_untouched():
    router = _StubRouter(result_model="other", refuse=False)
    ae, restore = _load_agentic_with_stubs(
        capable_names=["already-capable"], stub_router=router,
    )
    try:
        picked = ae._ensure_tool_capable_model(
            "already-capable", _routing("already-capable"),
        )
        assert picked == "already-capable", (
            f"a capable model must be returned unchanged, got {picked}"
        )
        assert router.calls == [], (
            "a capable model must NOT trigger a router re-selection; the "
            "guard must short-circuit on the capability verdict"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract A3 -- execute() wires the guard, tool-bound turns only
# ---------------------------------------------------------------------------
def _drive(ae, *, message, model, stub_router):
    """Run execute() to exhaustion with tools reachable; capture outcome.

    Returns ``(chunks, raised)``: the yielded chunks (with the pipelines
    stubbed to markers) and any exception that propagated out.
    """
    inst = ae.AgenticExecutor(
        executor=types.SimpleNamespace(),        # truthy base executor
        tool_executor=types.SimpleNamespace(),   # truthy tool executor
        default_model="fallback-model",
    )
    # Tools are "reachable": pair the truthy tool executor with the module
    # availability flag so tool_executor_available is True this turn.
    ae.TOOL_EXECUTOR_AVAILABLE = True

    # Hermetic legacy baseline so the pipeline choice does not depend on
    # which optional siblings happen to be installed in the running
    # environment: no capability manifest (the manifest-armed tools path is
    # exercised by the wiring suite, not here) and no sandbox registry. The
    # guard under test keys off tool_bound (needs_tools or sandbox), which
    # the message alone sets, so neutralizing these leaves it unaffected --
    # a keywordless turn stays non-tool-bound and takes the direct pipeline.
    ae.CAPABILITY_MANIFEST_AVAILABLE = False
    ae.build_manifest = None
    ae._default_tool_registry = None

    def _stub_tools(*args, **kwargs):
        yield "TOOLS-REACHED"

    def _stub_direct(*args, **kwargs):
        yield "DIRECT-REACHED"

    inst._execute_tools_pipeline = _stub_tools
    inst._execute_direct_pipeline = _stub_direct

    chunks = []
    raised = None
    try:
        for chunk in inst.execute(
            message, _routing(model), use_llm_analysis=False,
        ):
            chunks.append(chunk)
    except Exception as caught:  # noqa: BLE001 -- assert on it in the test
        raised = caught
    return chunks, raised


def test_a3_execute_enforces_tool_bound_only():
    # (i) tool-bound + non-capable + router refuses -> execute() refuses.
    refusing = _StubRouter(refuse=True)
    ae, restore = _load_agentic_with_stubs(
        capable_names=["cap-model"], stub_router=refusing,
    )
    try:
        chunks, raised = _drive(
            ae, message=TOOL_BOUND, model="non-capable", stub_router=refusing,
        )
        assert isinstance(raised, _RouterRefusal), (
            "a tool-bound turn on a non-capable model with tools reachable "
            "must refuse explicitly (router error propagates), got "
            f"{type(raised).__name__ if raised else None} chunks={chunks!r}"
        )
        assert "TOOLS-REACHED" not in chunks, (
            "the refusal must precede tool dispatch; the tools pipeline must "
            "not run on a non-capable model"
        )
        assert refusing.calls, "execute() must consult the router for re-selection"
    finally:
        restore()

    # (ii) keywordless + non-capable -> NOT enforced (sentinel).
    router = _StubRouter(result_model="cap-model", refuse=False)
    ae2, restore2 = _load_agentic_with_stubs(
        capable_names=["cap-model"], stub_router=router,
    )
    try:
        chunks, raised = _drive(
            ae2, message=KEYWORDLESS, model="non-capable", stub_router=router,
        )
        assert raised is None, (
            f"a non-tool-bound turn must not refuse, got {type(raised).__name__}"
        )
        assert "DIRECT-REACHED" in chunks, (
            f"a non-tool-bound turn must run the direct pipeline, got {chunks!r}"
        )
        assert router.calls == [], (
            "a non-tool-bound turn must NOT consult the router; the guard must "
            "not leak into ordinary chat"
        )
    finally:
        restore2()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("A1 guard enforces for non-capable", test_a1_guard_enforces_for_non_capable_model),
        ("A2 guard leaves capable untouched", test_a2_guard_leaves_capable_model_untouched),
        ("A3 execute enforces tool-bound only", test_a3_execute_enforces_tool_bound_only),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
