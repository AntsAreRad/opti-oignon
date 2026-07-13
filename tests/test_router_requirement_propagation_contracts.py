#!/usr/bin/env python3
"""Contracts for tool-calling requirement propagation across every routing branch.

The router accepts a tool-calling requirement and its profile-ranked path
already enforces it (exclusion, fail-secure refusal, public entry). These
contracts pin the branches the requirement must ALSO traverse, so a posed
requirement is never swallowed between the caller and the effective model
selection:

  * Contract P1 -- FORCED SELECTION: a posed requirement outranks model
    forcing. A forced model whose tool-calling verdict is explicitly
    negative refuses by name (ToolCapableModelUnavailable) instead of
    silently producing a tool-less turn.
  * Contract P2 -- VISION AUTO-ROUTING: with the requirement posed, a
    vision candidate whose verdict is explicitly negative is skipped so a
    capable vision candidate wins; when no vision candidate is capable,
    selection falls through to the requirement-aware standard path
    instead of returning the incapable vision pick.
  * Contract P3 -- CONVENIENCE ENTRY: the module-level route() exposes
    the requirement and forwards it to the instance path, so a caller of
    the convenience function can pose it at all.
  * Contract P4 -- INDETERMINABLE CAPABILITY: with the requirement posed
    and the capability predicate unimportable, selection refuses by name.
    Ordinary routing keeps its decline-to-constrain posture; a POSED
    requirement over an indeterminable capability fails secure instead of
    silently disabling the filter.
  * Contract P5 -- SENTINEL: with the requirement off (the default,
    including the convenience entry's default), the forced branch and the
    vision branch are unchanged: a non-capable pick is returned as
    before. The requirement never leaks into ordinary routing.

The capability verdict is the manifest's own predicate, stubbed here so a
model name decides its own verdict; the router must consult it rather
than reimplement the rule. Loads the router alone under a stand-in
package with controllable siblings; a meta-path guard refuses any project
submodule that was not seeded, so the load behaves identically whether or
not the project is installed. Local-only (the public distribution ships
no tests). Runs under pytest or directly via the __main__ runner.
"""

import importlib.util
import sys
import traceback
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
    ignores the parent path, so a real submodule resolves behind the
    test's back -- silently importing live code. This guard sits ahead of
    every finder and refuses the names that were not seeded, so a load
    behaves identically whether the project is installed or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


# ---------------------------------------------------------------------------
# Controllable stubs for the router's siblings
# ---------------------------------------------------------------------------
class _StubProfile:
    """Minimal profile: a name plus an explicit capability set."""

    def __init__(self, name, capabilities=()):
        self.name = name
        self._capabilities = set(capabilities)

    def has_capability(self, capability):
        return capability in self._capabilities


class _StubProfileManager:
    """Profile store stand-in: ranked task candidates plus vision candidates."""

    def __init__(self, ranked_names, vision_names):
        self._ranked = list(ranked_names)
        self._vision = list(vision_names)
        self.count = len(self._ranked) + len(self._vision)

    def _ensure_loaded(self):
        return None

    def find_best_for_task(self, task_type, speed_tier=None,
                           quality_tier=None, limit=5):
        if task_type == "vision":
            return [
                _StubProfile(n, capabilities=("vision",))
                for n in self._vision[:limit]
            ]
        return [_StubProfile(n) for n in self._ranked[:limit]]

    def build_routing_reason(self, **kwargs):
        return SimpleNamespace(to_dict=lambda: {})


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


def _analysis():
    """AnalysisResult stand-in with the fields route() reads."""
    return SimpleNamespace(
        task_type=SimpleNamespace(value="general_chat"),
        complexity=SimpleNamespace(value="medium"),
        suggested_model_type="general",
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Isolated loading of the router with controlled siblings
# ---------------------------------------------------------------------------
_ROUTER_KEYS = (
    "ollama", "opti_oignon", "opti_oignon.analyzer", "opti_oignon.config",
    "opti_oignon.model_profiles", "opti_oignon.capability_manifest",
    "opti_oignon.router",
)


def _load_router(*, ranked_names, vision_names, capable_names,
                 fallback_model, available=None):
    """Load the router alone against controllable capability + profiles.

    ``ranked_names`` are the profiled task candidates in preference order,
    ``vision_names`` the vision candidates in preference order,
    ``capable_names`` the set the stub predicate calls tool-capable,
    ``fallback_model`` the name the config seam returns, ``available`` the
    installed-model list (defaults to every candidate plus the fallback).
    The predicate records every name it is asked about. Returns
    ``(router_instance, router_module, predicate_calls, restore)``.
    """
    saved = {k: sys.modules.get(k) for k in _ROUTER_KEYS}
    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    ollama_stub = types.ModuleType("ollama")
    ollama_stub.list = lambda: SimpleNamespace(models=[])
    sys.modules["ollama"] = ollama_stub

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    analyzer = types.ModuleType("opti_oignon.analyzer")
    analyzer.AnalysisResult = object
    analyzer.TaskType = SimpleNamespace(
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
    mp.profile_manager = _StubProfileManager(ranked_names, vision_names)
    sys.modules["opti_oignon.model_profiles"] = mp
    pkg.model_profiles = mp

    capable = set(capable_names)
    predicate_calls = []

    def _predicate(name):
        predicate_calls.append(name)
        return name in capable

    cm = types.ModuleType("opti_oignon.capability_manifest")
    cm.model_tool_capable = _predicate
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
    pulled = (
        list(available) if available is not None
        else list(ranked_names) + list(vision_names) + [fallback_model]
    )
    instance.get_available_models = lambda force_refresh=False: list(pulled)

    def restore():
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return instance, rt, predicate_calls, restore


def _refusal_type(rt):
    exc = getattr(rt, "ToolCapableModelUnavailable", None)
    assert exc is not None, "router must expose ToolCapableModelUnavailable"
    return exc


# ---------------------------------------------------------------------------
# Contract P1 -- forced selection under a posed requirement refuses by name
# ---------------------------------------------------------------------------
def test_p1_forced_non_capable_model_refuses_when_required():
    instance, rt, _calls, restore = _load_router(
        ranked_names=["ranked-with-tools"],
        vision_names=[],
        capable_names=["ranked-with-tools"],
        fallback_model="cfg-model",
        available=["forced-no-tools", "ranked-with-tools", "cfg-model"],
    )
    try:
        exc = _refusal_type(rt)
        raised = None
        result = None
        try:
            result = instance.route(
                _analysis(), force_model="forced-no-tools",
                require_tool_calling=True,
            )
        except Exception as caught:  # noqa: BLE001 -- assert the type below
            raised = caught
        assert raised is not None, (
            "a forced model with an explicit negative verdict must refuse "
            "under a posed requirement, not silently return "
            f"{getattr(result, 'model', result)!r}"
        )
        assert isinstance(raised, exc), (
            f"the refusal must be ToolCapableModelUnavailable, got "
            f"{type(raised).__name__}"
        )
        assert "forced-no-tools" in str(raised), (
            "the refusal must name the forced selection, got "
            f"{raised!s}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract P2 -- vision auto-routing skips non-capable picks when required
# ---------------------------------------------------------------------------
def test_p2_vision_skips_non_capable_pick_when_required():
    # (a) a capable vision candidate exists behind the incapable pick.
    instance, _rt, _calls, restore = _load_router(
        ranked_names=["ranked-with-tools"],
        vision_names=["viz-no-tools", "viz-with-tools"],
        capable_names=["viz-with-tools", "ranked-with-tools"],
        fallback_model="cfg-model",
    )
    try:
        result = instance.route(
            _analysis(), images=["imagepayload"],
            require_tool_calling=True,
        )
        assert result.model == "viz-with-tools", (
            "required vision routing must skip the non-capable vision pick "
            f"and select the capable one, got {result.model}"
        )
        assert result.vision_routed is True
    finally:
        restore()

    # (b) no capable vision candidate -> fall through to the
    # requirement-aware standard path, never the incapable vision pick.
    instance, _rt, _calls, restore = _load_router(
        ranked_names=["ranked-with-tools"],
        vision_names=["viz-no-tools"],
        capable_names=["ranked-with-tools"],
        fallback_model="cfg-model",
    )
    try:
        result = instance.route(
            _analysis(), images=["imagepayload"],
            require_tool_calling=True,
        )
        assert result.model == "ranked-with-tools", (
            "with no capable vision candidate the required turn must fall "
            "through to the requirement-aware standard selection, got "
            f"{result.model}"
        )
        assert result.vision_routed is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract P3 -- the module-level convenience exposes and forwards it
# ---------------------------------------------------------------------------
def test_p3_convenience_entry_forwards_the_requirement():
    instance, rt, _calls, restore = _load_router(
        ranked_names=["top-no-tools", "second-with-tools"],
        vision_names=[],
        capable_names=["second-with-tools"],
        fallback_model="cfg-model",
    )
    try:
        rt.router = instance  # the convenience delegates to the singleton
        result = rt.route(_analysis(), require_tool_calling=True)
        assert result.model == "second-with-tools", (
            "the convenience entry must forward the posed requirement so "
            "the non-capable top model is excluded, got "
            f"{result.model}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract P4 -- posed requirement over an indeterminable capability refuses
# ---------------------------------------------------------------------------
def test_p4_required_with_unimportable_predicate_refuses():
    instance, rt, _calls, restore = _load_router(
        ranked_names=["top-no-tools"],
        vision_names=[],
        capable_names=[],
        fallback_model="cfg-model",
    )
    try:
        exc = _refusal_type(rt)
        # Halt the lazy predicate import inside the window: a None entry
        # makes the import system raise before any finder runs.
        pkg = sys.modules["opti_oignon"]
        saved_cm = sys.modules.get("opti_oignon.capability_manifest")
        sys.modules["opti_oignon.capability_manifest"] = None
        try:
            raised = None
            result = None
            try:
                result = instance.route(
                    _analysis(), require_tool_calling=True,
                )
            except Exception as caught:  # noqa: BLE001 -- assert below
                raised = caught
            assert raised is not None, (
                "a posed requirement over an unimportable capability "
                "predicate must refuse, not silently return "
                f"{getattr(result, 'model', result)!r}"
            )
            assert isinstance(raised, exc), (
                f"the refusal must be ToolCapableModelUnavailable, got "
                f"{type(raised).__name__}: {raised!s}"
            )
        finally:
            if saved_cm is None:
                sys.modules.pop("opti_oignon.capability_manifest", None)
            else:
                sys.modules["opti_oignon.capability_manifest"] = saved_cm
            pkg.capability_manifest = saved_cm
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract P5 -- sentinel: the requirement off never over-constrains
# ---------------------------------------------------------------------------
def test_p5_not_required_keeps_forced_and_vision_picks():
    # (a) forced non-capable model is returned unchanged.
    instance, _rt, _calls, restore = _load_router(
        ranked_names=["ranked-with-tools"],
        vision_names=[],
        capable_names=["ranked-with-tools"],
        fallback_model="cfg-model",
        available=["forced-no-tools", "ranked-with-tools", "cfg-model"],
    )
    try:
        result = instance.route(_analysis(), force_model="forced-no-tools")
        assert result.model == "forced-no-tools", (
            "without the requirement a forced non-capable model must be "
            f"returned unchanged, got {result.model}"
        )
    finally:
        restore()

    # (b) the non-capable vision pick is returned unchanged.
    instance, _rt, _calls, restore = _load_router(
        ranked_names=["ranked-with-tools"],
        vision_names=["viz-no-tools", "viz-with-tools"],
        capable_names=["viz-with-tools", "ranked-with-tools"],
        fallback_model="cfg-model",
    )
    try:
        result = instance.route(_analysis(), images=["imagepayload"])
        assert result.model == "viz-no-tools", (
            "without the requirement vision routing must keep its first "
            f"available pick, got {result.model}"
        )
    finally:
        restore()

    # (c) the convenience entry defaults the requirement off.
    instance, rt, _calls, restore = _load_router(
        ranked_names=["top-no-tools", "second-with-tools"],
        vision_names=[],
        capable_names=["second-with-tools"],
        fallback_model="cfg-model",
    )
    try:
        rt.router = instance
        result = rt.route(_analysis())
        assert result.model == "top-no-tools", (
            "the convenience default must keep ordinary routing unchanged, "
            f"got {result.model}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("P1 forced non-capable refuses when required",
         test_p1_forced_non_capable_model_refuses_when_required),
        ("P2 vision skips non-capable pick when required",
         test_p2_vision_skips_non_capable_pick_when_required),
        ("P3 convenience entry forwards the requirement",
         test_p3_convenience_entry_forwards_the_requirement),
        ("P4 required + unimportable predicate refuses",
         test_p4_required_with_unimportable_predicate_refuses),
        ("P5 not required keeps forced and vision picks",
         test_p5_not_required_keeps_forced_and_vision_picks),
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
