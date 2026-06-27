#!/usr/bin/env python3
"""Tests for the execution-pipeline system (pipelines.py).

Covers the parts that are pure or injectable:

  * ExecutionStep / ExecutionPipeline -- validation rules and dict round-trips.
  * PipelineStore -- the CRUD + YAML persistence layer, including the rule that
    builtin pipelines are read-only (cannot be updated, deleted, or shadowed by
    a custom one). The store takes config_dir / data_dir, so every test runs
    against fresh temp directories.
  * PipelineRunner._evaluate_condition -- the pure step-condition evaluator.

The full PipelineRunner.execute generator (executor-backed, streaming) is not
covered here -- it needs a fake-executor harness and is left for a focused pass.
"""

from opti_oignon.pipelines import (
    ExecutionPipeline,
    ExecutionStep,
    PipelineRunner,
    PipelineStore,
)

_BUILTIN_YAML = (
    "id: builtin-demo\n"
    "name: Builtin Demo\n"
    "description: a builtin pipeline\n"
    "steps:\n"
    "  - step_type: direct\n"
    "    label: Direct\n"
)


def _step(step_type="direct", **over):
    return ExecutionStep(step_type=step_type, **over)


def _pipeline(pid="my-pipeline", *, name="My Pipeline", steps=None, **over):
    return ExecutionPipeline(
        id=pid,
        name=name,
        steps=steps if steps is not None else [_step()],
        **over,
    )


def _store(tmp_path, *, with_builtin=False):
    config_dir = tmp_path / "config"
    if with_builtin:
        config_dir.mkdir()
        (config_dir / "demo.yaml").write_text(_BUILTIN_YAML)
    return PipelineStore(config_dir=config_dir, data_dir=tmp_path / "data")


# ===========================================================================
# ExecutionStep
# ===========================================================================

def test_step_default_label_from_type():
    s = ExecutionStep(step_type="think_tools")
    assert s.label == "Think Tools"          # underscores -> spaces, title-cased


def test_step_validate_ok():
    assert _step("direct", label="Go").validate() == []


def test_step_validate_rejects_unknown_type():
    errors = _step("not_a_type", label="X").validate()
    assert errors  # non-empty


def test_step_validate_rejects_blank_label():
    # An explicit whitespace label is invalid (default-label logic only fills
    # an empty string).
    errors = ExecutionStep(step_type="direct", label="   ").validate()
    assert errors


def test_step_to_dict_minimal_omits_defaults():
    d = _step("direct", label="Go").to_dict()
    assert d == {"step_type": "direct", "label": "Go"}   # defaults omitted


def test_step_to_dict_includes_non_defaults():
    s = _step(
        "tools", label="T", model_override="qwen",
        parameters={"k": 1}, condition="always", pass_previous_output=False,
    )
    d = s.to_dict()
    assert d["model_override"] == "qwen"
    assert d["parameters"] == {"k": 1}
    assert d["condition"] == "always"
    assert d["pass_previous_output"] is False


def test_step_roundtrip():
    s = _step("reasoning", label="R", model_override="m", parameters={"a": 2})
    assert ExecutionStep.from_dict(s.to_dict()).to_dict() == s.to_dict()


# ===========================================================================
# ExecutionPipeline
# ===========================================================================

def test_pipeline_validate_ok():
    assert _pipeline().validate() == []


def test_pipeline_validate_requires_id():
    assert any("ID" in e for e in _pipeline(pid="").validate())


def test_pipeline_validate_rejects_bad_id():
    assert _pipeline(pid="1-bad").validate()        # must start with a letter


def test_pipeline_validate_requires_name():
    assert _pipeline(name="  ").validate()


def test_pipeline_validate_requires_at_least_one_step():
    assert any("etape" in e.lower() for e in _pipeline(steps=[]).validate())


def test_pipeline_validate_aggregates_step_errors():
    errors = _pipeline(steps=[_step("bogus_type", label="X")]).validate()
    assert any(e.startswith("Etape 1:") for e in errors)


def test_pipeline_step_count_and_summary():
    p = _pipeline(steps=[_step("think"), _step("code_verify")])
    assert p.step_count == 2
    assert p.step_types_summary == "think -> code_verify"


def test_pipeline_post_init_converts_dict_steps():
    p = ExecutionPipeline(
        id="p", name="P", steps=[{"step_type": "direct", "label": "D"}],
    )
    assert isinstance(p.steps[0], ExecutionStep)
    assert p.steps[0].step_type == "direct"


def test_pipeline_roundtrip():
    p = _pipeline(steps=[_step("think", label="T"), _step("direct", label="D")])
    p2 = ExecutionPipeline.from_dict(p.id, p.to_dict())
    assert p2.id == p.id
    assert p2.to_dict() == p.to_dict()


# ===========================================================================
# PipelineStore -- CRUD, persistence, builtin protection
# ===========================================================================

def test_store_empty(tmp_path):
    store = _store(tmp_path)
    assert store.list_all() == []


def test_store_create_and_get(tmp_path):
    store = _store(tmp_path)
    assert store.create(_pipeline("p1")) is True
    got = store.get("p1")
    assert got is not None
    assert got.is_builtin is False
    assert [p.id for p in store.list_custom()] == ["p1"]


def test_store_create_rejects_duplicate_id(tmp_path):
    store = _store(tmp_path)
    store.create(_pipeline("dup"))
    assert store.create(_pipeline("dup")) is False


def test_store_create_rejects_invalid_pipeline(tmp_path):
    store = _store(tmp_path)
    assert store.create(_pipeline("bad", steps=[])) is False   # no steps


def test_store_update_custom(tmp_path):
    store = _store(tmp_path)
    store.create(_pipeline("p1", name="Old"))
    created_at = store.get("p1").created_at
    assert store.update("p1", _pipeline("p1", name="New")) is True
    updated = store.get("p1")
    assert updated.name == "New"
    assert updated.created_at == created_at        # preserved across update


def test_store_update_missing_returns_false(tmp_path):
    store = _store(tmp_path)
    assert store.update("ghost", _pipeline("ghost")) is False


def test_store_delete_custom(tmp_path):
    store = _store(tmp_path)
    store.create(_pipeline("p1"))
    assert store.delete("p1") is True
    assert store.get("p1") is None
    assert store.delete("p1") is False             # already gone


def test_store_duplicate(tmp_path):
    store = _store(tmp_path)
    store.create(_pipeline("src", steps=[_step("think"), _step("direct")]))
    dup = store.duplicate("src", "copy")
    assert dup is not None
    assert dup.is_builtin is False
    assert dup.step_count == 2
    assert store.duplicate("src", "copy") is None      # target id exists
    assert store.duplicate("missing", "x") is None     # source missing


def test_store_persistence_reload(tmp_path):
    store = _store(tmp_path)
    store.create(_pipeline("persist", name="Persisted"))
    # A fresh store on the same data_dir reloads the custom pipeline from YAML.
    store2 = PipelineStore(config_dir=tmp_path / "config", data_dir=tmp_path / "data")
    reloaded = store2.get("persist")
    assert reloaded is not None
    assert reloaded.name == "Persisted"


def test_store_get_step_types(tmp_path):
    types = {t["type"] for t in _store(tmp_path).get_step_types()}
    assert "direct" in types
    assert "consensus" in types


# --- builtin protection ---

def test_store_loads_builtin(tmp_path):
    store = _store(tmp_path, with_builtin=True)
    builtin = store.get("builtin-demo")
    assert builtin is not None
    assert builtin.is_builtin is True
    assert [p.id for p in store.list_builtin()] == ["builtin-demo"]


def test_store_cannot_update_builtin(tmp_path):
    store = _store(tmp_path, with_builtin=True)
    assert store.update("builtin-demo", _pipeline("builtin-demo")) is False


def test_store_cannot_delete_builtin(tmp_path):
    store = _store(tmp_path, with_builtin=True)
    assert store.delete("builtin-demo") is False


def test_store_custom_cannot_shadow_builtin(tmp_path):
    store = _store(tmp_path, with_builtin=True)
    # create() refuses an id that already exists (the builtin).
    assert store.create(_pipeline("builtin-demo")) is False


# ===========================================================================
# PipelineRunner._evaluate_condition  (pure)
# ===========================================================================

def _runner():
    return PipelineRunner(agentic_executor=None, smart_router=None)


def test_condition_always_and_empty():
    r = _runner()
    assert r._evaluate_condition("always", "x", "y") is True
    assert r._evaluate_condition("", "x", "y") is True


def test_condition_if_code_detected():
    r = _runner()
    assert r._evaluate_condition("if_code_detected", "def foo(): pass", "") is True
    assert r._evaluate_condition("if_code_detected", "just prose", "") is False


def test_condition_if_code_detected_checks_prev_output():
    r = _runner()
    # The code marker is in prev_output, not current_input.
    assert r._evaluate_condition("if_code_detected", "prose", "```py```") is True


def test_condition_if_long_input():
    r = _runner()
    assert r._evaluate_condition("if_long_input", "a" * 501, "") is True
    assert r._evaluate_condition("if_long_input", "short", "") is False


def test_condition_unrecognized_defaults_true():
    r = _runner()
    assert r._evaluate_condition("if_full_moon", "x", "y") is True
