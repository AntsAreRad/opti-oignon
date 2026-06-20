#!/usr/bin/env python3
"""
S216 per-fix suite -- PIP-06 wiring lot.

Covers, against the recorded findings and the arbitrated decisions:

- PIP-06: the S53 seam finished end to end. ChatRequest carries
  exec_pipeline; the frontend sendMessage transmits it; the chat WS
  resolves the pipeline up front (honest refusals on unknown id or missing
  prerequisites), runs the executor-backed PipelineRunner with the S185
  approval gate forwarded per step, relays the step-boundary tuples as
  status (never concatenated), and records the pipeline id in the done
  metadata.
- Runner hardening: inter-step emergency-stop check (the R-04 precursor),
  module-absent fails open (availability posture, S215).
- PIP-07: the three dead raw-requests prompt-generation functions retired
  from pipeline_manager.
- DPL-01: dynamic_planning and the Gradio-era dynamic_pipeline_ui shim
  retired; __init__ exports stripped.
- Supersession (deselect-plus-reassert): the three s186 guards that walk
  _SWEPT_FILES are re-asserted here over the 10-file list (the original
  list minus the retired dynamic_planning.py), reusing the original
  detectors via isolated import; the s192 DPL-01 pin is superseded by the
  absence assertions.

Isolation: importlib.util.spec_from_file_location with sys.modules
pre-seeding where needed; the runner is exercised with an injected fake
executor so no ollama dependency chain loads.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_ROOT / "opti_oignon"
API_DIR = PKG_DIR / "api"
TESTS_DIR = PROJECT_ROOT / "tests"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

ROUTES_CHAT_SRC = (API_DIR / "routes_chat.py").read_text(encoding="utf-8")
SCHEMAS_SRC = (API_DIR / "schemas.py").read_text(encoding="utf-8")
PIPELINES_PATH = PKG_DIR / "pipelines.py"
PIPELINES_SRC = PIPELINES_PATH.read_text(encoding="utf-8")
PIPELINE_MANAGER_SRC = (PKG_DIR / "pipeline_manager.py").read_text(encoding="utf-8")
INIT_SRC = (PKG_DIR / "__init__.py").read_text(encoding="utf-8")
CHAT_TS_SRC = (FRONTEND_DIR / "src/lib/stores/chat.ts").read_text(encoding="utf-8")
PYPROJECT_SRC = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
ROADMAP_SRC = (PROJECT_ROOT / "ROADMAP_POST_AUDIT.md").read_text(encoding="utf-8")


def _load_module(name: str, path: Path):
    """Isolated module load (register-before-exec, the S65+ idiom)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fakes for the runner behaviour tests
# ---------------------------------------------------------------------------

class FakeAgenticExecutor:
    """Records every execute() call and yields configured chunks."""

    def __init__(self, outputs):
        # outputs: list of lists, one inner list per expected step call
        self.outputs = list(outputs)
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        idx = len(self.calls) - 1
        chunks = self.outputs[idx] if idx < len(self.outputs) else []
        yield from chunks


class FakeSmartRouter:
    enabled = False


class FakeEstop:
    """is_stopped() answers from a scripted sequence (last value sticky)."""

    def __init__(self, sequence):
        self._seq = list(sequence)

    def is_stopped(self):
        if len(self._seq) > 1:
            return self._seq.pop(0)
        return self._seq[0]


@pytest.fixture()
def pipelines_mod():
    return _load_module("oo_pipelines_s216", PIPELINES_PATH)


def _two_step_pipeline(mod):
    now = "2026-01-01T00:00:00"
    steps = [
        mod.ExecutionStep(step_type="direct", label="first"),
        mod.ExecutionStep(step_type="direct", label="second"),
    ]
    return mod.ExecutionPipeline(
        id="p1",
        name="P1",
        description="two-step",
        steps=steps,
        created_at=now,
        updated_at=now,
        is_builtin=False,
    )


def _run(mod, pipeline, executor, **kwargs):
    runner = mod.PipelineRunner(
        agentic_executor=executor, smart_router=FakeSmartRouter()
    )
    return list(
        runner.execute(
            pipeline=pipeline, message="hello", routing=object(), **kwargs
        )
    )


# ---------------------------------------------------------------------------
# A. ChatRequest carries exec_pipeline
# ---------------------------------------------------------------------------

class TestChatRequestField:
    def test_source_declares_field(self):
        assert "exec_pipeline: str | None = None" in SCHEMAS_SRC

    def test_model_field_defaults_none(self):
        mod = _load_module("oo_schemas_s216", API_DIR / "schemas.py")
        req = mod.ChatRequest(message="hi")
        assert req.exec_pipeline is None

    def test_model_field_accepts_id(self):
        mod = _load_module("oo_schemas_s216b", API_DIR / "schemas.py")
        req = mod.ChatRequest(message="hi", exec_pipeline="p1")
        assert req.exec_pipeline == "p1"


# ---------------------------------------------------------------------------
# B. routes_chat wiring by source
# ---------------------------------------------------------------------------

class TestRoutesChatWiring:
    def test_conditional_import_block(self):
        assert "from opti_oignon.pipelines import (" in ROUTES_CHAT_SRC
        assert "get_pipeline_runner" in ROUTES_CHAT_SRC
        assert "get_pipeline_store" in ROUTES_CHAT_SRC
        assert "EXEC_PIPELINES_AVAILABLE = True" in ROUTES_CHAT_SRC
        assert "EXEC_PIPELINES_AVAILABLE = False" in ROUTES_CHAT_SRC

    def test_upfront_resolution_refuses_honestly(self):
        assert "Execution pipelines module not available" in ROUTES_CHAT_SRC
        assert "Execution pipelines require the agentic executor" in ROUTES_CHAT_SRC
        assert 'f"Unknown execution pipeline: {request.exec_pipeline}"' in ROUTES_CHAT_SRC

    def test_resolution_happens_before_generation_thread(self):
        resolve_pos = ROUTES_CHAT_SRC.index("_exec_pipeline_obj = None")
        thread_pos = ROUTES_CHAT_SRC.index("def _generate():")
        assert resolve_pos < thread_pos

    def test_pipeline_branch_takes_priority_over_plain_agentic(self):
        branch_pos = ROUTES_CHAT_SRC.index("if _exec_pipeline_obj is not None:")
        agentic_pos = ROUTES_CHAT_SRC.index("elif use_agentic:")
        assert branch_pos < agentic_pos

    def test_runner_invocation_forwards_approval_gate(self):
        start = ROUTES_CHAT_SRC.index("get_pipeline_runner().execute(")
        call = ROUTES_CHAT_SRC[start:start + 700]
        assert "pipeline=_exec_pipeline_obj" in call
        assert "approval_fn=_approval_fn" in call
        assert "on_status=_on_status" in call

    def test_three_tuple_branch_before_two_tuple(self):
        three_pos = ROUTES_CHAT_SRC.index(
            "isinstance(chunk, tuple) and len(chunk) == 3"
        )
        two_pos = ROUTES_CHAT_SRC.index(
            "isinstance(chunk, tuple) and len(chunk) == 2"
        )
        assert three_pos < two_pos

    def test_step_end_relayed_as_status_and_filtered(self):
        start = ROUTES_CHAT_SRC.index(
            "isinstance(chunk, tuple) and len(chunk) == 3"
        )
        block = ROUTES_CHAT_SRC[start:start + 600]
        assert '"pipeline_step_end"' in block
        assert '"status"' in block
        assert "continue" in block

    def test_done_metadata_records_pipeline_id(self):
        assert (
            'done_metadata["exec_pipeline"] = _exec_pipeline_obj.id'
            in ROUTES_CHAT_SRC
        )

    def test_routes_chat_ast_valid(self):
        ast.parse(ROUTES_CHAT_SRC)


# ---------------------------------------------------------------------------
# C. PipelineRunner behaviour (fake executor, no ollama chain)
# ---------------------------------------------------------------------------

class TestRunnerBehaviour:
    def test_happy_two_steps_event_order_and_chaining(self, pipelines_mod):
        mod = pipelines_mod
        execu = FakeAgenticExecutor([["alpha"], ["beta"]])
        mod._resolve_emergency_stop = lambda: None
        out = _run(mod, _two_step_pipeline(mod), execu)
        tags = [c[0] if isinstance(c, tuple) else c for c in out]
        assert tags == [
            "pipeline_step_start", "alpha", "pipeline_step_end",
            "pipeline_step_start", "beta", "pipeline_step_end",
        ]
        assert len(execu.calls) == 2
        # Chaining: step 2 prompt carries step 1 output
        assert "alpha" in execu.calls[1]["message"]
        assert "Based on the following previous analysis" in execu.calls[1]["message"]

    def test_approval_fn_forwarded_to_every_step(self, pipelines_mod):
        mod = pipelines_mod
        execu = FakeAgenticExecutor([["a"], ["b"]])
        mod._resolve_emergency_stop = lambda: None
        sentinel = object()
        _run(mod, _two_step_pipeline(mod), execu, approval_fn=sentinel)
        assert len(execu.calls) == 2
        assert all(c.get("approval_fn") is sentinel for c in execu.calls)

    def test_estop_mid_run_aborts_before_next_step(self, pipelines_mod):
        mod = pipelines_mod
        execu = FakeAgenticExecutor([["a"], ["b"]])
        # First inter-step check passes, second sees the stop. One shared
        # instance: the seam is resolved at every loop iteration.
        shared = FakeEstop([False, True])
        mod._resolve_emergency_stop = lambda: shared
        out = _run(mod, _two_step_pipeline(mod), execu)
        assert len(execu.calls) == 1
        assert any(
            isinstance(c, str) and "emergency stop engaged" in c for c in out
        )
        # The second step never started
        starts = [c for c in out if isinstance(c, tuple) and c[0] == "pipeline_step_start"]
        assert len(starts) == 1

    def test_estop_engaged_from_start_runs_nothing(self, pipelines_mod):
        mod = pipelines_mod
        execu = FakeAgenticExecutor([["a"], ["b"]])
        mod._resolve_emergency_stop = lambda: FakeEstop([True])
        out = _run(mod, _two_step_pipeline(mod), execu)
        assert execu.calls == []
        assert any(
            isinstance(c, str) and "emergency stop engaged" in c for c in out
        )

    def test_estop_module_absent_fails_open(self, pipelines_mod):
        mod = pipelines_mod
        execu = FakeAgenticExecutor([["a"], ["b"]])
        mod._resolve_emergency_stop = lambda: None
        out = _run(mod, _two_step_pipeline(mod), execu)
        assert len(execu.calls) == 2
        assert not any(
            isinstance(c, str) and "emergency stop" in c for c in out
        )

    def test_resolve_seam_exists_in_source(self):
        assert "def _resolve_emergency_stop():" in PIPELINES_SRC
        assert "approval_fn: Callable | None = None" in PIPELINES_SRC
        assert "approval_fn=approval_fn" in PIPELINES_SRC

    def test_pipelines_ast_valid(self):
        ast.parse(PIPELINES_SRC)


# ---------------------------------------------------------------------------
# D. PIP-07 retirement
# ---------------------------------------------------------------------------

class TestPip07Retired:
    def test_dead_functions_absent(self):
        assert "generate_step_prompt" not in PIPELINE_MANAGER_SRC
        assert "_get_prompt_model" not in PIPELINE_MANAGER_SRC

    def test_no_raw_requests_left(self):
        assert "import requests" not in PIPELINE_MANAGER_SRC
        assert "localhost:11434" not in PIPELINE_MANAGER_SRC

    def test_pipeline_manager_ast_valid(self):
        ast.parse(PIPELINE_MANAGER_SRC)


# ---------------------------------------------------------------------------
# E. DPL-01 + shim retirement
# ---------------------------------------------------------------------------

class TestDpl01Retired:
    def test_modules_absent_from_tree(self):
        assert not (PKG_DIR / "dynamic_planning.py").exists()
        assert not (PKG_DIR / "dynamic_pipeline_ui.py").exists()

    def test_init_imports_stripped(self):
        assert "from .dynamic_planning import" not in INIT_SRC
        assert "from .dynamic_pipeline_ui import" not in INIT_SRC

    def test_init_export_names_stripped(self):
        for name in (
            "DYNAMIC_PLANNING_AVAILABLE",
            "DYNAMIC_PIPELINE_AVAILABLE",
            "DynamicPlanningOrchestrator",
            "should_use_dynamic_pipeline",
            "process_with_dynamic_pipeline",
            "plan_dynamic_pipeline",
        ):
            assert name not in INIT_SRC, name

    def test_retirement_documented_in_init(self):
        assert "S216" in INIT_SRC
        assert "retired" in INIT_SRC

    def test_init_ast_valid(self):
        ast.parse(INIT_SRC)


# ---------------------------------------------------------------------------
# F. s186 guards re-asserted over the 10-file list (supersession)
# ---------------------------------------------------------------------------

class TestS186GuardsReasserted:
    """The three deselected s186 tests, re-run via the ORIGINAL detectors
    (isolated import of the original suite) over the corrected swept list."""

    @pytest.fixture()
    def s186(self):
        mod = _load_module(
            "oo_s186_reassert",
            TESTS_DIR / "test_s186_english_inference_core.py",
        )
        expected_removed = "dynamic_planning.py"
        assert expected_removed in mod._SWEPT_FILES
        mod._SWEPT_FILES = [
            f for f in mod._SWEPT_FILES if f != expected_removed
        ]
        assert len(mod._SWEPT_FILES) == 10
        return mod

    def test_no_emoji_holds_on_ten_files(self, s186):
        s186.TestNoEmojiInSweptCore().test_no_emoji()

    def test_pure_ascii_holds_on_ten_files(self, s186):
        s186.TestNoNonAsciiInSweptCore().test_pure_ascii_except_allowed()

    def test_ast_holds_on_ten_files(self, s186):
        s186.TestSweptCoreAstValid().test_ast_valid()


# ---------------------------------------------------------------------------
# G. Supersession pins (pyproject deselects)
# ---------------------------------------------------------------------------

class TestSupersessionPins:
    def test_pip06_unwired_pin_flipped(self):
        """Counter-pin to the superseded s192 unwired pin: the wiring landed.

        Exactly one api file calls get_pipeline_runner, and it is the chat
        route (the S53 seam). The s192 pin documented its own supersession
        ("expected to flip and be superseded by the wiring-cycle tests").
        """
        hits = sorted(
            f.name for f in API_DIR.glob("*.py")
            if "get_pipeline_runner" in f.read_text(encoding="utf-8")
        )
        assert hits == ["routes_chat.py"]

    def test_s192_pip06_pin_deselected(self):
        assert (
            "--deselect=tests/test_s192_f5a_pipelines.py::"
            "TestRecordedFindingPins::test_pip06_runner_unwired_pin"
        ) in PYPROJECT_SRC

    def test_s186_deselects_present(self):
        for tid in (
            "tests/test_s186_english_inference_core.py::TestNoEmojiInSweptCore::test_no_emoji",
            "tests/test_s186_english_inference_core.py::TestNoNonAsciiInSweptCore::test_pure_ascii_except_allowed",
            "tests/test_s186_english_inference_core.py::TestSweptCoreAstValid::test_ast_valid",
        ):
            assert f"--deselect={tid}" in PYPROJECT_SRC, tid

    def test_s192_dpl01_pin_deselected(self):
        assert (
            "--deselect=tests/test_s192_f5a_pipelines.py::"
            "TestRecordedFindingPins::test_dpl01_dict_only_pin"
        ) in PYPROJECT_SRC


# ---------------------------------------------------------------------------
# H. Frontend seam (chat.ts) and roadmap registration
# ---------------------------------------------------------------------------

class TestFrontendSeam:
    def test_options_type_declares_exec_pipeline(self):
        assert "exec_pipeline?: string;" in CHAT_TS_SRC

    def test_request_build_transmits_exec_pipeline(self):
        assert "exec_pipeline: options?.exec_pipeline" in CHAT_TS_SRC


class TestRoadmapRegistration:
    def test_pip06_landed_recorded(self):
        assert "LANDED at S216" in ROADMAP_SRC

    def test_retirements_recorded(self):
        assert ROADMAP_SRC.count("RETIRED at S216") == 2
        assert "REDUCED at S216" in ROADMAP_SRC


# ---------------------------------------------------------------------------
# I. AST validity over every touched Python file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("relpath", [
    "opti_oignon/api/schemas.py",
    "opti_oignon/api/routes_chat.py",
    "opti_oignon/pipelines.py",
    "opti_oignon/pipeline_manager.py",
    "opti_oignon/__init__.py",
])
def test_ast_valid(relpath):
    ast.parse((PROJECT_ROOT / relpath).read_text(encoding="utf-8"))
