#!/usr/bin/env python3
"""Contracts for the tool-calling requirement in the model router.

The router owns an optional filter that keeps a tool-bound turn on a model
that can actually call tools. These contracts pin its enforcement, at the
selection surface itself (no live Ollama, no real profiles):

  * Contract R1 -- EXCLUSION: with the requirement engaged, a profiled
    candidate whose capability verdict is negative is skipped, so the pick
    is a tool-capable model even when a non-capable one ranks higher and is
    available.
  * Contract R2 -- FAIL-SECURE: with the requirement engaged, when no
    profiled model is both available and capable and the config fallback is
    itself explicitly non-capable, selection RAISES rather than handing back
    a model that cannot call tools. A silent fallback to a non-capable model
    is the exact regression this forbids.
  * Contract R3 -- NO OVER-CONSTRAINT (sentinel): with the requirement OFF
    (the default), a non-capable top model is returned unchanged. A plain,
    non-tool-bound turn keeps access to models without a tool-calling
    profile; the filter never leaks into ordinary routing.
  * Contract R4 -- PUBLIC ENTRY: the router exposes a real entry point that
    turns the requirement on. It returns a tool-capable model when one
    exists and fails secure when none does. This is the surface a tool-bound
    caller uses; on a tree where the requirement is never raised it is
    absent, so this contract also proves the wiring point exists.

The capability verdict is the manifest's own predicate, stubbed here so a
model name decides its own verdict; the router must consult it rather than
reimplement the rule.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
idiom: a stub package plus stubbed siblings so the router loads without a
live Ollama or the real profile store, and the module-under-test resolves
its lazy capability import against the stubbed copy.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


# ---------------------------------------------------------------------------
# Controllable stubs for the router's siblings
# ---------------------------------------------------------------------------
class _StubProfile:
    """Minimal profile: a name plus a no-op capability list."""

    def __init__(self, name):
        self.name = name

    def has_capability(self, capability):  # vision path only; unused here
        return False


class _StubProfileManager:
    """Profile store stand-in returning a fixed ranked candidate list."""

    def __init__(self, ranked_names):
        self._ranked = list(ranked_names)
        self.count = len(self._ranked)

    def _ensure_loaded(self):
        return None

    def find_best_for_task(self, task_type, speed_tier=None,
                           quality_tier=None, limit=5):
        return [_StubProfile(n) for n in self._ranked[:limit]]

    def build_routing_reason(self, **kwargs):  # route() only; unused here
        return types.SimpleNamespace(to_dict=lambda: {})


class _StubConfig:
    """Config seam stand-in: a single fallback model, no alternatives."""

    def __init__(self, fallback_model):
        self._fallback = fallback_model

    def get_model(self, model_type, priority):
        return self._fallback

    def get_fallback_models(self):
        return []

    def get_temperature(self, key):
        return 0.5

    def get_timeout(self, key):
        return 60


# ---------------------------------------------------------------------------
# Isolated loading of the router with controlled siblings
# ---------------------------------------------------------------------------
_ROUTER_KEYS = (
    "ollama", "opti_oignon", "opti_oignon.analyzer", "opti_oignon.config",
    "opti_oignon.model_profiles", "opti_oignon.capability_manifest",
    "opti_oignon.router",
)


def _load_router(*, ranked_names, capable_names, fallback_model):
    """Load the router alone against controllable capability + profiles.

    ``ranked_names`` are the profiled candidates in preference order,
    ``capable_names`` the set the stub predicate calls tool-capable,
    ``fallback_model`` the name the config seam returns. Every candidate is
    reported available. Returns ``(router_instance, router_module, restore)``.
    """
    saved = {k: sys.modules.get(k) for k in _ROUTER_KEYS}

    ollama_stub = types.ModuleType("ollama")
    ollama_stub.list = lambda: types.SimpleNamespace(models=[])
    sys.modules["ollama"] = ollama_stub

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    analyzer = types.ModuleType("opti_oignon.analyzer")
    analyzer.AnalysisResult = object
    analyzer.TaskType = types.SimpleNamespace(
        PLANNING_DEEP="planning_deep", SIMPLE_QUESTION="simple_question",
    )
    sys.modules["opti_oignon.analyzer"] = analyzer
    pkg.analyzer = analyzer

    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = _StubConfig(fallback_model)
    sys.modules["opti_oignon.config"] = cfg
    pkg.config = cfg

    mp = types.ModuleType("opti_oignon.model_profiles")
    mp.ModelProfile = _StubProfile
    mp.RoutingReason = object
    mp.profile_manager = _StubProfileManager(ranked_names)
    sys.modules["opti_oignon.model_profiles"] = mp
    pkg.model_profiles = mp

    capable = set(capable_names)
    cm = types.ModuleType("opti_oignon.capability_manifest")
    cm.model_tool_capable = lambda name: name in capable
    sys.modules["opti_oignon.capability_manifest"] = cm
    pkg.capability_manifest = cm

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.router", _OO / "router.py",
    )
    rt = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.router"] = rt
    spec.loader.exec_module(rt)
    pkg.router = rt

    instance = rt.ModelRouter()
    # Every profiled candidate plus the config fallback is "pulled".
    available = list(ranked_names) + [fallback_model]
    instance.get_available_models = lambda force_refresh=False: list(available)

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return instance, rt, restore


# ---------------------------------------------------------------------------
# Contract R1 -- exclusion of a non-capable candidate when required
# ---------------------------------------------------------------------------
def test_r1_required_excludes_non_capable_candidate():
    instance, _rt, restore = _load_router(
        ranked_names=["ranks-high-no-tools", "ranks-low-with-tools"],
        capable_names=["ranks-low-with-tools"],
        fallback_model="cfg-model",
    )
    try:
        model, _reason, _alts, profile_used = (
            instance._select_model_with_profiles(
                "general", "code_python", "balanced",
                require_tool_calling=True,
            )
        )
        assert model == "ranks-low-with-tools", (
            f"required turn must skip the non-capable top model, got {model}"
        )
        assert profile_used is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract R2 -- fail-secure: no capable model -> explicit refusal
# ---------------------------------------------------------------------------
def test_r2_required_no_capable_model_refuses():
    instance, rt, restore = _load_router(
        ranked_names=["only-no-tools"],
        capable_names=[],                 # nothing is tool-capable
        fallback_model="cfg-no-tools",    # config fallback is non-capable too
    )
    try:
        exc = getattr(rt, "ToolCapableModelUnavailable", None)
        raised = None
        result = None
        try:
            result = instance._select_model_with_profiles(
                "general", "code_python", "balanced",
                require_tool_calling=True,
            )
        except Exception as caught:  # noqa: BLE001 -- assert the type below
            raised = caught
        assert raised is not None, (
            "required selection with no tool-capable model must RAISE, not "
            f"silently return {result!r}"
        )
        assert exc is not None and isinstance(raised, exc), (
            "the refusal must be ToolCapableModelUnavailable, got "
            f"{type(raised).__name__}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract R3 -- sentinel: the requirement OFF never over-constrains
# ---------------------------------------------------------------------------
def test_r3_not_required_keeps_non_capable_model():
    instance, _rt, restore = _load_router(
        ranked_names=["top-no-tools", "second-with-tools"],
        capable_names=["second-with-tools"],
        fallback_model="cfg-model",
    )
    try:
        model, _reason, _alts, profile_used = (
            instance._select_model_with_profiles(
                "general", "chat", "balanced",
                require_tool_calling=False,
            )
        )
        assert model == "top-no-tools", (
            "a non-tool-bound turn must keep the non-capable top model; the "
            f"filter must not leak into ordinary routing, got {model}"
        )
        assert profile_used is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract R4 -- public entry turns the requirement on
# ---------------------------------------------------------------------------
def test_r4_public_entry_selects_capable_or_refuses():
    # (a) a capable model exists -> the entry returns it.
    instance, rt, restore = _load_router(
        ranked_names=["no-tools", "with-tools"],
        capable_names=["with-tools"],
        fallback_model="cfg-model",
    )
    try:
        entry = getattr(instance, "select_tool_capable_model", None)
        assert callable(entry), (
            "the router must expose select_tool_capable_model as the real "
            "entry point that engages the tool-calling requirement"
        )
        model = entry(model_type="general", task_type="code_python")
        assert model == "with-tools", (
            f"public entry must select the tool-capable model, got {model}"
        )
    finally:
        restore()

    # (b) no capable model -> the entry fails secure (propagates the refusal).
    instance, rt, restore = _load_router(
        ranked_names=["only-no-tools"],
        capable_names=[],
        fallback_model="cfg-no-tools",
    )
    try:
        exc = getattr(rt, "ToolCapableModelUnavailable", None)
        raised = None
        try:
            instance.select_tool_capable_model(
                model_type="general", task_type="code_python",
            )
        except Exception as caught:  # noqa: BLE001 -- assert the type below
            raised = caught
        assert raised is not None, (
            "public entry with no tool-capable model must fail secure (raise)"
        )
        assert exc is not None and isinstance(raised, exc), (
            f"the refusal must be ToolCapableModelUnavailable, got "
            f"{type(raised).__name__}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("R1 required excludes non-capable", test_r1_required_excludes_non_capable_candidate),
        ("R2 required no capable -> refuse", test_r2_required_no_capable_model_refuses),
        ("R3 not required keeps non-capable", test_r3_not_required_keeps_non_capable_model),
        ("R4 public entry capable or refuse", test_r4_public_entry_selects_capable_or_refuses),
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
