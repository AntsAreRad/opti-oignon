#!/usr/bin/env python3
"""Contracts for capability-aware model routing (exposure only).

Two capabilities land here and one invariant is pinned born-green:

  * A PUBLIC tool-calling predicate on the capability manifest, so callers
    other than the manifest builder can ask "can this model call tools?"
    without reaching into a private helper. The verdict is exactly the
    manifest's own: an explicit profile verdict wins in both directions;
    no profile, or no profile system, stays capable (the historical
    fallback protocol drives tools for models without native calling).
  * A routing filter that, WHEN ASKED, skips profiled models whose
    tool-calling verdict is negative -- and only then. The switch is off
    by default, so ordinary routing is byte-identical to today; when every
    profiled candidate is filtered out the router fails open to its config
    selection rather than returning nothing.
  * Per-stage manifests are rebuilt for every stage, never cached across
    stages (a stale manifest would hand one stage another stage's tool
    truth). This already holds; the contract pins it so a future cache
    cannot silently break it. Proven born-green here, and by a directed
    cache mutation in the session log.

No live caller flips the filter on this session: the predicate and the
switch are exposed and pinned, nothing downstream changes its request.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
idiom: canonical dotted names under a package stub, the real module under
test, controllable stand-ins for every sibling it might touch.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
_OO = _ROOT / "opti_oignon"


def _real(dotted, path):
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Clause 1 -- public tool-capability predicate on the manifest
# ---------------------------------------------------------------------------
_MANIFEST_KEYS = (
    "opti_oignon", "opti_oignon.capability_manifest",
    "opti_oignon.model_profiles",
)


def _load_manifest_with_profiles(verdicts):
    """Load capability_manifest alone; profiles resolve through a stand-in.

    ``verdicts`` maps a model name to the ``tool_calling`` attribute the
    stand-in profile exposes (use the sentinel ``_ABSENT`` for a profile
    that carries no verdict, and omit a name entirely for "no profile").
    """
    saved = {k: sys.modules.get(k) for k in _MANIFEST_KEYS}
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    mp = types.ModuleType("opti_oignon.model_profiles")

    class _Profile:
        def __init__(self, verdict):
            self.tool_calling = None if verdict is _ABSENT else verdict

        def has_capability(self, name):
            return False

    def get_profile(model):
        if model in verdicts:
            return _Profile(verdicts[model])
        return None

    mp.get_profile = get_profile
    sys.modules["opti_oignon.model_profiles"] = mp
    pkg.model_profiles = mp

    cm = _real("opti_oignon.capability_manifest", _OO / "capability_manifest.py")
    pkg.capability_manifest = cm

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return cm, restore


_ABSENT = object()


def test_c1_public_predicate_mirrors_the_profile_verdict():
    cm, restore = _load_manifest_with_profiles({
        "neg-model": False,
        "pos-model": True,
        "blank-model": _ABSENT,
    })
    try:
        predicate = getattr(cm, "model_tool_capable", None)
        assert predicate is not None, (
            "capability_manifest exposes no public model_tool_capable"
        )
        assert predicate("neg-model") is False
        assert predicate("pos-model") is True
        # No verdict, and no profile at all, both stay capable.
        assert predicate("blank-model") is True
        assert predicate("unknown-model") is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# Router isolation (clauses 2 and 3)
# ---------------------------------------------------------------------------
_ROUTER_KEYS = (
    "ollama", "opti_oignon", "opti_oignon.analyzer", "opti_oignon.config",
    "opti_oignon.model_profiles", "opti_oignon.router",
    "opti_oignon.capability_manifest",
)


class _FakeProfile:
    def __init__(self, name):
        self.name = name


class _FakePM:
    """Profile manager stand-in: a fixed best-for-task list."""

    def __init__(self, names):
        self._names = list(names)
        self.count = len(self._names)

    def _ensure_loaded(self):
        return None

    def find_best_for_task(self, task_type, speed_tier=None,
                           quality_tier=None, limit=5):
        return [_FakeProfile(n) for n in self._names[:limit]]


def _load_router(*, profile_names, verdicts, available):
    """Load router.py alone with model profiles monkeypatched on.

    ``profile_names`` is the best-for-task order the fake PM returns;
    ``verdicts`` is the tool-capability map the capability_manifest
    stand-in answers with; ``available`` is what the router sees as the
    installed model list.
    """
    saved = {k: sys.modules.get(k) for k in _ROUTER_KEYS}

    ollama_stub = types.ModuleType("ollama")
    ollama_stub.list = lambda: {"models": []}
    sys.modules["ollama"] = ollama_stub

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    analyzer = types.ModuleType("opti_oignon.analyzer")
    analyzer.AnalysisResult = object
    analyzer.TaskType = SimpleNamespace(
        PLANNING_DEEP="planning_deep",
        SIMPLE_QUESTION="simple_question",
    )
    sys.modules["opti_oignon.analyzer"] = analyzer
    pkg.analyzer = analyzer

    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(
        get_temperature=lambda *a, **k: 0.3,
        get_timeout=lambda *a, **k: 30,
    )
    sys.modules["opti_oignon.config"] = cfg
    pkg.config = cfg

    # capability_manifest stand-in: only the public predicate is needed.
    capmod = types.ModuleType("opti_oignon.capability_manifest")
    capmod.model_tool_capable = lambda model: verdicts.get(model, True)
    sys.modules["opti_oignon.capability_manifest"] = capmod
    pkg.capability_manifest = capmod

    # model_profiles must import cleanly; the real selection uses the
    # module globals, which we monkeypatch after load.
    mp = types.ModuleType("opti_oignon.model_profiles")
    mp.ModelProfile = object
    mp.RoutingReason = object
    mp.profile_manager = None
    sys.modules["opti_oignon.model_profiles"] = mp
    pkg.model_profiles = mp

    rt = _real("opti_oignon.router", _OO / "router.py")
    pkg.router = rt

    rt.MODEL_PROFILES_AVAILABLE = True
    rt.profile_manager = _FakePM(profile_names)

    class _FakeRouter(rt.ModelRouter):
        def __init__(self):
            pass

        def get_available_models(self, force_refresh=False):
            return list(available)

        def _select_model(self, model_type, priority):
            return ("config-fallback", "config")

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return rt, _FakeRouter(), restore


def test_c2_filter_skips_the_negative_profile_when_asked():
    rt, router, restore = _load_router(
        profile_names=["neg-model", "cap-model"],
        verdicts={"neg-model": False, "cap-model": True},
        available=["neg-model", "cap-model"],
    )
    try:
        model, reason, _alts, profiled = router._select_model_with_profiles(
            "general", "code_python", "balanced",
            require_tool_calling=True,
        )
        assert model == "cap-model", (model, reason)
        assert profiled is True
    finally:
        restore()


def test_c3_default_keeps_legacy_first_and_all_negative_fails_open():
    # Sub-case A: the switch off preserves the legacy first-available pick.
    rt, router, restore = _load_router(
        profile_names=["neg-model", "cap-model"],
        verdicts={"neg-model": False, "cap-model": True},
        available=["neg-model", "cap-model"],
    )
    try:
        model, _reason, _alts, _profiled = router._select_model_with_profiles(
            "general", "code_python", "balanced",
            require_tool_calling=False,
        )
        assert model == "neg-model", model
    finally:
        restore()

    # Sub-case B: everything negative + required -> config fallback, not empty.
    rt, router, restore = _load_router(
        profile_names=["neg-one", "neg-two"],
        verdicts={"neg-one": False, "neg-two": False},
        available=["neg-one", "neg-two"],
    )
    try:
        model, reason, _alts, profiled = router._select_model_with_profiles(
            "general", "code_python", "balanced",
            require_tool_calling=True,
        )
        assert model == "config-fallback", (model, reason)
        assert profiled is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# Clause 4 -- per-stage manifest freshness (born-green)
# ---------------------------------------------------------------------------
_AGENTIC_KEYS = ("opti_oignon", "opti_oignon.agentic_executor")


def _load_agentic():
    saved = {k: sys.modules.get(k) for k in _AGENTIC_KEYS}
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    ae = _real("opti_oignon.agentic_executor", _OO / "agentic_executor.py")
    pkg.agentic_executor = ae

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return ae, restore


def test_c4_per_stage_manifest_is_rebuilt_never_cached():
    ae, restore = _load_agentic()
    try:
        builds = []

        def _recorder(*, model, web_search_override=None):
            token = SimpleNamespace(
                model=model,
                has_tools=(model != "stage-b-neg"),
            )
            builds.append((model, token))
            return token

        ae.CAPABILITY_MANIFEST_AVAILABLE = True
        ae.build_manifest = _recorder

        first = ae._build_request_manifest("stage-a", None)
        middle = ae._build_request_manifest("stage-b-neg", None)
        third = ae._build_request_manifest("stage-a", None)

        # Three fresh builds, one per call -- no cache short-circuit.
        assert [m for m, _ in builds] == ["stage-a", "stage-b-neg", "stage-a"]
        # A no-tools stage degrades to the legacy None.
        assert middle is None
        # The two same-named stages get DISTINCT manifest objects.
        assert first is not None and third is not None
        assert first is not third, "per-stage manifest was cached"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
_TESTS = (
    "test_c1_public_predicate_mirrors_the_profile_verdict",
    "test_c2_filter_skips_the_negative_profile_when_asked",
    "test_c3_default_keeps_legacy_first_and_all_negative_fails_open",
    "test_c4_per_stage_manifest_is_rebuilt_never_cached",
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
