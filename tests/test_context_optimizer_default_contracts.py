#!/usr/bin/env python3
"""What the orchestrator promises about its shipped default and wiring.

The unified orchestrator only runs when two things hold at once: its
module-level instance exists, and its enabled flag says yes. These
contracts pin both halves. The shipped configuration file enables the
orchestrator, and that is a deliberate product default: it is pinned
here so reverting it is a visible one-line decision, never an accident.
The conservative posture underneath is pinned too -- with no
configuration file at all, the code's own default is off and the caller
keeps its manual pipeline.

The instance half is a lazy accessor: the first call builds the
orchestrator and wires the live collaborators, each behind its own
guarded import, so a missing collaborator degrades that one seam to
nothing instead of taking the accessor down. The accessor memoizes, an
explicit initializer keeps priority over it, and a reset hands the next
caller a fresh build. With every collaborator unreachable the degraded
instance still assembles an honest context: the system prompt first,
the current turn last, the fallback budget shape reported. With the
real budget engine injected, an oversized history triggers the
emergency ladder and the flood never costs the system prompt or the
current turn; a named preset reshapes the zone budgets through the
engine's native override keyword and leaves no residue on the next
plain call. One recorded oddity is pinned as it stands rather than
judged: a per-call preset name that matches nothing falls back to the
balanced ratios silently while the report echoes the requested name --
no production caller passes a per-call preset today, and a change to
that behavior must surface as a red here, not slip through.

Loaded through the shared isolation window; the model runtime is
declared unreachable, and collaborators are absent, seeded stand-ins,
or the real budget engine depending on what each contract exercises.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.context_optimizer"
_ENGINE = "opti_oignon.prompt_optimization"

_CFG = {
    "allocation": {
        "system_ratio": 0.10, "project_ratio": 0.25, "history_ratio": 0.40,
        "user_ratio": 0.10, "reserve_ratio": 0.15, "fingerprint_ratio": 0.025,
    },
    "fallback_context_windows": {},
    "default_context_window": 8192,
    "minimum_budgets": {
        "system": 256, "project": 0, "history": 512, "user": 256, "reserve": 512,
    },
}


def _load(*, seeded=None, with_engine=False):
    """Load the orchestrator alone, with optional collaborator seams."""
    targets = {_TARGET: source("context_optimizer.py")}
    if with_engine:
        targets[_ENGINE] = source("prompt_optimization.py")
    loaded, restore = isolate(
        targets=targets, blocked=("ollama",), seeded=dict(seeded or {}),
    )
    if with_engine:
        return loaded[_TARGET], loaded[_ENGINE], restore
    return loaded[_TARGET], restore


def _module_with(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _flood(n=30, chars=4000):
    return [{"role": "user", "content": "f" * chars} for _ in range(n)]


# ---------------------------------------------------------------------------
# o1 -- the shipped configuration enables the orchestrator
# ---------------------------------------------------------------------------

def test_o1_shipped_configuration_enables_the_orchestrator():
    module, restore = _load()
    try:
        cfg = module._load_config()
        assert cfg.get("enabled") is True
        assert cfg.get("active_preset") == "balanced"
        assert set(cfg.get("priority_presets", {})) == {
            "balanced", "rag_heavy", "history_heavy",
        }
        assert cfg.get("emergency", {}).get("min_recent_messages") == 2
    finally:
        restore()


# ---------------------------------------------------------------------------
# o2 -- with no configuration file the code's own default is off
# ---------------------------------------------------------------------------

def test_o2_missing_configuration_defaults_to_off():
    module, restore = _load()
    try:
        cfg = module._load_config(Path("/tmp/definitely-absent-optimizer.yaml"))
        assert cfg.get("enabled") is False
        assert cfg.get("active_preset") == "balanced"
    finally:
        restore()


# ---------------------------------------------------------------------------
# o3 -- the accessor builds lazily, memoizes, and resets cleanly
# ---------------------------------------------------------------------------

def test_o3_accessor_builds_lazily_memoizes_and_resets():
    module, restore = _load()
    try:
        first = module.get_optimizer()
        assert first is not None
        assert isinstance(first, module.ContextOptimizer)
        assert module.get_optimizer() is first
        # The lazy build read the shipped configuration.
        assert first.enabled is True
        # Every collaborator import was unreachable here, and each seam
        # degraded to nothing on its own.
        assert first._budget_manager is None
        assert first._project_builder is None
        assert first._compressor is None
        assert first._sliding_window is None
        assert first._context_manager is None
        module.reset_optimizer()
        assert module.get_optimizer() is not first
    finally:
        restore()


# ---------------------------------------------------------------------------
# o4 -- the explicit initializer keeps priority over the lazy build
# ---------------------------------------------------------------------------

def test_o4_explicit_initializer_keeps_priority_over_lazy_build():
    module, restore = _load()
    try:
        module.reset_optimizer()
        pinned = module.init_optimizer(config={"enabled": False})
        assert module.get_optimizer() is pinned
        assert module.get_optimizer().enabled is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# o5 -- fully degraded, the assembly still holds its shape
# ---------------------------------------------------------------------------

def test_o5_fully_degraded_assembly_still_holds_its_shape():
    module, restore = _load()
    try:
        module.reset_optimizer()
        optimizer = module.get_optimizer()
        result = optimizer.optimize(
            model="m", system_prompt="S", user_message="U",
            conversation_history=[{"role": "user", "content": "hello there"}],
        )
        assert [m["role"] for m in result.messages] == ["system", "user", "user"]
        assert result.messages[0]["content"] == "S"
        assert result.messages[-1]["content"] == "U"
        assert result.report.total_window == 8192  # the fallback budget shape
        history_zone = next(z for z in result.report.zones if z.zone == "history")
        assert history_zone.strategy == "none"
        assert result.report.overflow is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# o6 -- the lazy build wires the seeded collaborators, seam by seam
# ---------------------------------------------------------------------------

def test_o6_lazy_build_wires_the_seeded_collaborators():
    budget = object()
    builder = object()
    compressor = object()
    window = object()
    manager = object()
    seeds = {
        _ENGINE: _module_with(_ENGINE, prompt_budget_manager=budget),
        "opti_oignon.project_context": _module_with(
            "opti_oignon.project_context", project_context_builder=builder,
        ),
        "opti_oignon.conversation_compressor": _module_with(
            "opti_oignon.conversation_compressor", conversation_compressor=compressor,
        ),
        "opti_oignon.context_window": _module_with(
            "opti_oignon.context_window", sliding_window_manager=window,
        ),
        "opti_oignon.context_manager": _module_with(
            "opti_oignon.context_manager", get_context_manager=lambda: manager,
        ),
    }
    module, restore = _load(seeded=seeds)
    try:
        module.reset_optimizer()
        optimizer = module.get_optimizer()
        assert optimizer._budget_manager is budget
        assert optimizer._project_builder is builder
        assert optimizer._compressor is compressor
        assert optimizer._sliding_window is window
        assert optimizer._context_manager is manager
    finally:
        restore()


# ---------------------------------------------------------------------------
# o7 -- the runtime toggle round-trips and unknown presets are refused
# ---------------------------------------------------------------------------

def test_o7_runtime_toggle_round_trips_and_unknown_presets_refused():
    module, restore = _load()
    try:
        instance = module.init_optimizer(
            config=module._load_config(Path("/tmp/definitely-absent-optimizer.yaml")),
        )
        assert instance.enabled is False
        instance.update_config({"enabled": True})
        assert instance.enabled is True
        instance.update_config({"enabled": False})
        assert instance.enabled is False
        refused = False
        try:
            instance.update_config({"active_preset": "does-not-exist"})
        except ValueError:
            refused = True
        assert refused
    finally:
        restore()


# ---------------------------------------------------------------------------
# o8 -- a flood triggers the emergency ladder, never the fixed segments
# ---------------------------------------------------------------------------

def test_o8_flood_triggers_emergency_never_the_fixed_segments():
    module, engine, restore = _load(with_engine=True)
    try:
        mgr = engine.PromptTokenBudgetManager(config=dict(_CFG))
        optimizer = module.init_optimizer(
            config=module._load_config(Path("/tmp/definitely-absent-optimizer.yaml")),
            budget_manager=mgr,
        )
        result = optimizer.optimize(
            model="m", system_prompt="S", user_message="U",
            conversation_history=_flood(), context_window_override=8192,
        )
        assert result.report.overflow is True
        history_zone = next(z for z in result.report.zones if z.zone == "history")
        assert history_zone.strategy.startswith("no_compressor")
        assert history_zone.strategy.endswith("+emergency")
        assert result.messages[0]["content"] == "S"
        assert result.messages[-1]["content"] == "U"
        assert len(result.messages) - 2 == 6  # what fits under the ceiling
    finally:
        restore()


# ---------------------------------------------------------------------------
# o9 -- a named preset reshapes the budget and leaves no residue
# ---------------------------------------------------------------------------

def test_o9_named_preset_reshapes_budget_and_leaves_no_residue():
    module, engine, restore = _load(with_engine=True)
    try:
        mgr = engine.PromptTokenBudgetManager(config=dict(_CFG))
        optimizer = module.init_optimizer(budget_manager=mgr)
        result = optimizer.optimize(
            model="m", system_prompt="S", user_message="U",
            conversation_history=[{"role": "user", "content": "h" * 400}],
            context_window_override=8192, preset="rag_heavy", project_active=True,
        )
        project_zone = next(z for z in result.report.zones if z.zone == "project")
        assert project_zone.budgeted_tokens == 2867
        assert result.report.preset_used == "rag_heavy"
        # No residue: the next plain calculation is balanced again.
        plain = mgr.calculate_budget(
            model="m", project_active=True, context_window_override=8192,
        )
        assert plain.project_tokens == 2048
    finally:
        restore()


# ---------------------------------------------------------------------------
# o10 -- the unknown-preset fallback is silent and the report echoes it
# ---------------------------------------------------------------------------

def test_o10_unknown_preset_falls_back_silently_and_report_echoes():
    module, engine, restore = _load(with_engine=True)
    try:
        mgr = engine.PromptTokenBudgetManager(config=dict(_CFG))
        optimizer = module.init_optimizer(budget_manager=mgr)
        result = optimizer.optimize(
            model="m", system_prompt="S", user_message="U",
            conversation_history=[{"role": "user", "content": "h" * 400}],
            context_window_override=8192, preset="nope-preset", project_active=True,
        )
        project_zone = next(z for z in result.report.zones if z.zone == "project")
        assert project_zone.budgeted_tokens == 2048  # the balanced figure
        assert result.report.preset_used == "nope-preset"
    finally:
        restore()
