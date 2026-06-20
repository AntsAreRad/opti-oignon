#!/usr/bin/env python3
"""
S192 F5a tests -- pipelines (PIP-01..PIP-05) plus wiring-gap source pins.

Loads opti_oignon/pipelines.py standalone (stdlib + yaml only; the executor
import is lazy and guarded) and opti_oignon/pipeline_manager.py via the
sys.modules stub + spec_from_file_location + register-before-exec_module
idiom (its top-level `from .config import ...` is satisfied by a stub
package providing DATA_DIR / load_yaml / save_yaml).

Covered:
- PIP-01: pass_previous_output=False runs the step on the original message.
- PIP-02: a failed step does not poison the chain context.
- PIP-03: custom YAML entries cannot overwrite builtin pipelines at load.
- PIP-04: duplicate preserves each step's model override.
- PIP-05: create/update validate (id format, unknown agent), with the
  degraded-mode skip when no agent registry is loaded; route-level source pin.
- PIP-06 / DPL-01: wiring-gap and dict-only source pins (recorded findings).
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "opti_oignon"


# =============================================================================
# Loaders
# =============================================================================

def _load_pipelines_module():
    """Load opti_oignon/pipelines.py standalone."""
    name = "oo_s192_pipelines"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PKG_DIR / "pipelines.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # register before exec (3.13 dataclass idiom)
    spec.loader.exec_module(module)
    return module


def _load_pipeline_manager_module(tmp_path: Path):
    """Load opti_oignon/pipeline_manager.py under a stub package."""
    pkg_name = "oo_s192_pkg"
    mod_name = f"{pkg_name}.pipeline_manager"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    # Stub parent package
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules[pkg_name] = pkg

    # Stub .config with functional YAML helpers
    cfg = types.ModuleType(f"{pkg_name}.config")
    cfg.DATA_DIR = tmp_path

    def load_yaml(path):
        p = Path(path)
        if not p.exists():
            return {}
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    def save_yaml(path, data):
        Path(path).write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True,
                      sort_keys=False),
            encoding="utf-8",
        )
        return True

    cfg.load_yaml = load_yaml
    cfg.save_yaml = save_yaml
    sys.modules[f"{pkg_name}.config"] = cfg

    spec = importlib.util.spec_from_file_location(
        mod_name, PKG_DIR / "pipeline_manager.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # register before exec
    spec.loader.exec_module(module)
    return module


def _bare_manager(pm_mod, tmp_path: Path, agents=None, templates=None):
    """Build a PipelineManager without running __init__ (no fixed-path load)."""
    mgr = object.__new__(pm_mod.PipelineManager)
    mgr._pipelines = {}
    mgr._config_file = tmp_path / "absent_config.yaml"
    mgr._custom_file = tmp_path / "pipelines_custom.yaml"
    mgr._available_agents = list(agents or [])
    mgr._available_templates = list(templates or [])
    return mgr


# =============================================================================
# Fake executor for PipelineRunner tests
# =============================================================================

class FakeExecutor:
    """Records the prompts it receives; can fail on a chosen call index."""

    def __init__(self, fail_on_call: int | None = None):
        self.prompts: list[str] = []
        self.fail_on_call = fail_on_call

    def execute(self, message, routing, **kwargs):
        call_idx = len(self.prompts)
        self.prompts.append(message)
        if self.fail_on_call is not None and call_idx == self.fail_on_call:
            raise RuntimeError("boom")
        yield f"OUT{call_idx}"


def _run_pipeline(mod, executor, steps, message="ORIGINAL"):
    pipeline = mod.ExecutionPipeline(id="t", name="T", steps=steps)
    runner = mod.PipelineRunner(agentic_executor=executor)
    routing = SimpleNamespace(model="m")
    return list(runner.execute(pipeline, message, routing))


# =============================================================================
# PIP-01 -- pass_previous_output semantics
# =============================================================================

class TestPip01PassPreviousOutput:
    def test_false_uses_original_message(self):
        mod = _load_pipelines_module()
        ex = FakeExecutor()
        steps = [
            mod.ExecutionStep(step_type="direct", label="A"),
            mod.ExecutionStep(
                step_type="direct", label="B", pass_previous_output=False
            ),
        ]
        _run_pipeline(mod, ex, steps)
        assert ex.prompts[0] == "ORIGINAL"
        # Pre-fix this was "OUT0" (the bare previous step output).
        assert ex.prompts[1] == "ORIGINAL"

    def test_true_wraps_previous_output_and_original(self):
        mod = _load_pipelines_module()
        ex = FakeExecutor()
        steps = [
            mod.ExecutionStep(step_type="direct", label="A"),
            mod.ExecutionStep(step_type="direct", label="B"),
        ]
        _run_pipeline(mod, ex, steps)
        assert "OUT0" in ex.prompts[1]
        assert "Original question: ORIGINAL" in ex.prompts[1]
        assert "Based on the following previous analysis" in ex.prompts[1]


# =============================================================================
# PIP-02 -- failed step must not poison the chain context
# =============================================================================

class TestPip02FailedStepContext:
    def test_error_text_not_fed_to_next_step(self):
        mod = _load_pipelines_module()
        ex = FakeExecutor(fail_on_call=1)
        steps = [
            mod.ExecutionStep(step_type="direct", label="A"),
            mod.ExecutionStep(step_type="direct", label="B"),
            mod.ExecutionStep(step_type="direct", label="C"),
        ]
        chunks = _run_pipeline(mod, ex, steps)
        # Step C keeps step A's good context, not the error text.
        assert "[ERR]" not in ex.prompts[2]
        assert "OUT0" in ex.prompts[2]
        # The error is still surfaced to the stream and the step_end tuple.
        text = "".join(c for c in chunks if isinstance(c, str))
        assert "[ERR] Step 2 failed" in text
        ends = [c for c in chunks if isinstance(c, tuple) and c[0] == "pipeline_step_end"]
        assert "[ERR]" in ends[1][2]

    def test_failed_first_step_leaves_original_message(self):
        mod = _load_pipelines_module()
        ex = FakeExecutor(fail_on_call=0)
        steps = [
            mod.ExecutionStep(step_type="direct", label="A"),
            mod.ExecutionStep(step_type="direct", label="B"),
        ]
        _run_pipeline(mod, ex, steps)
        # accumulated_output stays empty -> step B falls back to the original.
        assert ex.prompts[1] == "ORIGINAL"


# =============================================================================
# PIP-03 -- custom YAML cannot overwrite a builtin
# =============================================================================

class TestPip03BuiltinShadowing:
    def test_load_custom_skips_builtin_id(self, tmp_path):
        pm = _load_pipeline_manager_module(tmp_path)
        mgr = _bare_manager(pm, tmp_path, agents=["coder"])
        builtin = pm.Pipeline(
            id="debug", name="Debug", is_builtin=True,
            steps=[pm.PipelineStep(name="S", agent="coder")],
        )
        mgr._pipelines["debug"] = builtin
        mgr._custom_file.write_text(yaml.dump({
            "pipelines": {
                "debug": {"name": "Evil", "steps": [{"name": "X", "agent": "coder"}]},
                "mine": {"name": "Mine", "steps": [{"name": "Y", "agent": "coder"}]},
            }
        }), encoding="utf-8")
        mgr._load_custom()
        # Pre-fix: 'debug' was overwritten and downgraded to is_builtin=False.
        assert mgr._pipelines["debug"].is_builtin is True
        assert mgr._pipelines["debug"].name == "Debug"
        assert "mine" in mgr._pipelines
        assert mgr._pipelines["mine"].is_builtin is False


# =============================================================================
# PIP-04 -- duplicate preserves the step model override
# =============================================================================

class TestPip04DuplicateModel:
    def test_model_field_preserved(self, tmp_path):
        pm = _load_pipeline_manager_module(tmp_path)
        mgr = _bare_manager(pm, tmp_path, agents=["coder"])
        src = pm.Pipeline(
            id="src", name="Src", is_builtin=True,
            steps=[pm.PipelineStep(name="S", agent="coder", model="qwen3:32b")],
        )
        mgr._pipelines["src"] = src
        new = mgr.duplicate("src", "copy1")
        assert new is not None
        # Pre-fix: model was dropped (None).
        assert new.steps[0].model == "qwen3:32b"


# =============================================================================
# PIP-05 -- create/update validation
# =============================================================================

class TestPip05Validation:
    def test_create_rejects_invalid_id(self, tmp_path):
        pm = _load_pipeline_manager_module(tmp_path)
        mgr = _bare_manager(pm, tmp_path, agents=["coder"])
        bad = pm.Pipeline(
            id="1 bad id!", name="Bad",
            steps=[pm.PipelineStep(name="S", agent="coder")],
        )
        assert mgr.create(bad) is False
        assert "1 bad id!" not in mgr._pipelines

    def test_create_rejects_unknown_agent(self, tmp_path):
        pm = _load_pipeline_manager_module(tmp_path)
        mgr = _bare_manager(pm, tmp_path, agents=["coder"])
        bad = pm.Pipeline(
            id="ok-id", name="Bad",
            steps=[pm.PipelineStep(name="S", agent="ghost")],
        )
        assert mgr.create(bad) is False

    def test_create_degraded_mode_skips_agent_check(self, tmp_path):
        pm = _load_pipeline_manager_module(tmp_path)
        mgr = _bare_manager(pm, tmp_path, agents=[])  # no registry loaded
        p = pm.Pipeline(
            id="ok-id", name="OK",
            steps=[pm.PipelineStep(name="S", agent="anything")],
        )
        assert mgr.create(p) is True
        assert "ok-id" in mgr._pipelines

    def test_update_rejects_unknown_agent(self, tmp_path):
        pm = _load_pipeline_manager_module(tmp_path)
        mgr = _bare_manager(pm, tmp_path, agents=["coder"])
        ok = pm.Pipeline(
            id="mine", name="Mine",
            steps=[pm.PipelineStep(name="S", agent="coder")],
        )
        assert mgr.create(ok) is True
        bad = pm.Pipeline(
            id="mine", name="Mine2",
            steps=[pm.PipelineStep(name="S", agent="ghost")],
        )
        assert mgr.update("mine", bad) is False
        assert mgr._pipelines["mine"].name == "Mine"

    def test_routes_call_validate_for_write(self):
        # Source pin (heavy import chain): both write routes validate.
        src = (REPO_ROOT / "opti_oignon" / "api" / "routes_pipelines.py").read_text(
            encoding="utf-8"
        )
        assert src.count("pipeline_manager.validate_for_write(") == 2
        assert 'status_code=422, detail="; ".join(errors)' in src


# =============================================================================
# PIP-06 / DPL-01 -- wiring-gap and dict-only source pins (recorded findings)
# =============================================================================

class TestRecordedFindingPins:
    def test_pip06_runner_unwired_pin(self):
        """No API route calls the PipelineRunner (recorded PIP-06).

        When the execution wiring lands, this pin is expected to flip and be
        superseded by the wiring-cycle tests (deselect + re-assert).
        """
        api_dir = REPO_ROOT / "opti_oignon" / "api"
        hits = [
            f.name for f in api_dir.glob("*.py")
            if "get_pipeline_runner" in f.read_text(encoding="utf-8")
        ]
        assert hits == []

    def test_dpl01_dict_only_pin(self):
        """dynamic_planning parses ollama.chat dict-only (recorded DPL-01)."""
        src = (PKG_DIR / "dynamic_planning.py").read_text(encoding="utf-8")
        assert 'chunk.get("message", {}).get("content", "")' in src
        assert 'response.get("message", {}).get("content", "")' in src
