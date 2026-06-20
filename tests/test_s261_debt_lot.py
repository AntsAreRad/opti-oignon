#!/usr/bin/env python3
"""S261 -- the logged-debt lot: French-comment normalisation of
agentic_executor.py, the approval_fn threading fix (the S185 EX-02 gate
found dead at the S261 read gate), and the MoE weights-residency
override table in the resource governor (the S259 kv_overrides recipe
mirrored).

Isolation: the established spec_from_file_location idiom with
sys.modules pre-seeding (an ollama stub and an opti_oignon package stub
carrying a real __path__), so the module chain resolves by path without
executing opti_oignon/__init__.py.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_BASE = Path(__file__).resolve().parent.parent
_AX_PATH = _BASE / "opti_oignon" / "agentic_executor.py"
_RG_PATH = _BASE / "opti_oignon" / "resource_governor.py"
_TE_PATH = _BASE / "opti_oignon" / "tool_executor.py"
_YAML_PATH = _BASE / "opti_oignon" / "config" / "resource_governor.yaml"
_DOC_PATH = _BASE / "DEBT_LOT_S261.md"
_VER_PATH = _BASE / "opti_oignon" / "__version__.py"

AX_SRC = _AX_PATH.read_text(encoding="utf-8")
RG_SRC = _RG_PATH.read_text(encoding="utf-8")
TE_SRC = _TE_PATH.read_text(encoding="utf-8")
YAML_SRC = _YAML_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Isolated module loading (the established idiom)
# ---------------------------------------------------------------------------

sys.modules.setdefault("ollama", types.ModuleType("ollama"))

if "opti_oignon" not in sys.modules:
    _pkg = types.ModuleType("opti_oignon")
    _pkg.__path__ = [str(_BASE / "opti_oignon")]
    sys.modules["opti_oignon"] = _pkg


def _load_module(dotted: str, relpath: str):
    existing = sys.modules.get(dotted)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(dotted, str(_BASE / relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


# The documented sweep-order pollution guard (S220): pre-load the
# governor's conditional dependencies by path, reusing whatever a prior
# suite already cached, so standalone and in-sweep resolve the same.
for _dotted, _rel in (
    ("opti_oignon.db_utils", "opti_oignon/db_utils.py"),
    ("opti_oignon.model_warmup", "opti_oignon/model_warmup.py"),
    ("opti_oignon.inference_backend", "opti_oignon/inference_backend.py"),
    ("opti_oignon.speculative_decoding", "opti_oignon/speculative_decoding.py"),
):
    _load_module(_dotted, _rel)

rg = _load_module(
    "opti_oignon.resource_governor", "opti_oignon/resource_governor.py"
)
ax = _load_module(
    "opti_oignon.agentic_executor", "opti_oignon/agentic_executor.py"
)

# ---------------------------------------------------------------------------
# Shared fakes and helpers
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _meminfo_file(tmp_path: Path, available_kb: int = 8 * 1024 * 1024) -> Path:
    p = tmp_path / "meminfo"
    p.write_text(
        "MemTotal:       16384000 kB\n"
        f"MemAvailable:   {available_kb} kB\n"
        "SwapTotal:       2097152 kB\n",
        encoding="utf-8",
    )
    return p


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "resource_governor.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def _gov(tmp_path: Path, yaml_text: str | None = None,
         available_kb: int = 8 * 1024 * 1024):
    cfg = (
        _write_yaml(tmp_path, yaml_text)
        if yaml_text is not None
        else tmp_path / "absent.yaml"
    )
    return rg.ResourceGovernor(
        config_path=cfg,
        db_path=tmp_path / "gov_s261.db",
        warmup=None,
        registry=None,
        clock=_Clock(),
        meminfo_path=_meminfo_file(tmp_path, available_kb),
    )


def _routing(model: str = "qwen3:32b"):
    return SimpleNamespace(
        model=model,
        task_type="general",
        temperature=0.3,
        prompt_variant="standard",
        model_type="general",
        priority_used="primary",
        explanation="test",
        timeout=60,
    )


class _GenExecutor:
    """Minimal base-executor fake: streams two chunks, records nothing."""

    def __init__(self):
        self.last_verification_results: list = []

    def execute(self, **kwargs):
        yield "ok"
        yield " done"


class _CaptureToolExecutor:
    """ToolExecutor fake capturing every execute_with_tools kwargs dict."""

    def __init__(self):
        self.calls: list[dict] = []

    def should_use_tools(self, *args, **kwargs) -> bool:
        return True

    def execute_with_tools(self, **kwargs):
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            response="TOOLS DONE",
            tool_calls=[SimpleNamespace(tool_name="echo", success=True)],
            model="m",
            total_time=0.1,
        )


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


def _param_names(fn: ast.FunctionDef) -> set[str]:
    names = {a.arg for a in fn.args.args}
    names |= {a.arg for a in fn.args.kwonlyargs}
    return names


_AX_TREE = ast.parse(AX_SRC)
_RG_TREE = ast.parse(RG_SRC)


# ---------------------------------------------------------------------------
# 1. French normalisation of agentic_executor.py
# ---------------------------------------------------------------------------

# Every marker below exists in the pristine S260 source ONLY inside
# comments, docstrings, or log literals -- never inside the French
# detection-keyword DATA, which is functional and deliberately kept.
_FRENCH_MARKERS = (
    "Executeur agentique",
    "Mots-cles",
    "Import conditionnel",
    "indisponible",
    "echoue",
    "Erreur pipeline",
    "Erreur generation",
    "Erreur callback",
    "termine en",
    "derniere execution",
    "Dernier result",
    "Sauvegarder",
    "utilisateur",
    "reflexion",
    "Annule l'execution",
    "Impossible de charger",
    "explicite, pipeline=",
    "CONSTANTES DE",
    "ANALYSE DE COMPLEXITE",
    "Proprietes publiques",
    "Utilitaires",
    "Analyse de tache",
    "Selectionner le pipeline",
    "Classifier la",
    "Besoin d'outils",
    "boucle ReAct",
    "si required",
    "if requestede",
    "Necessite au minimum",
    "Indique si",
)


class TestFrenchNormalisation:
    def test_french_markers_absent(self):
        leftovers = [m for m in _FRENCH_MARKERS if m in AX_SRC]
        assert leftovers == [], f"French left in agentic_executor: {leftovers}"

    def test_english_docstrings_present(self):
        assert "Unified agentic executor" in AX_SRC
        assert "PIPELINE CONSTANTS" in AX_SRC
        assert "DETECTION DATA" in AX_SRC
        assert "deliberately kept" in AX_SRC

    def test_french_detection_data_preserved(self):
        for token in (
            '"etape par etape"',
            '"cherche"',
            '"actualite"',
            '"meteo"',
            '"explique"',
            '"lis le fichier"',
        ):
            assert token in AX_SRC, token

    def test_think_equals_think_pin_carried(self):
        assert "think=think" in AX_SRC

    def test_pure_ascii_and_parses(self):
        AX_SRC.encode("ascii")
        ast.parse(AX_SRC)


# ---------------------------------------------------------------------------
# 2. The approval_fn threading fix (S185 EX-02 made live, S261)
# ---------------------------------------------------------------------------


class TestApprovalFnSignatures:
    def test_tools_pipeline_signature_takes_approval_fn(self):
        fn = _method(_AX_TREE, "AgenticExecutor", "_execute_tools_pipeline")
        assert "approval_fn" in _param_names(fn)

    def test_think_tools_signature_takes_approval_fn(self):
        fn = _method(
            _AX_TREE, "AgenticExecutor", "_execute_think_tools_pipeline"
        )
        assert "approval_fn" in _param_names(fn)

    def test_execute_dispatch_passes_approval_fn_to_both(self):
        execute = _method(_AX_TREE, "AgenticExecutor", "execute")
        passed = set()
        for node in ast.walk(execute):
            if isinstance(node, ast.Call) and isinstance(
                node.func, ast.Attribute
            ):
                if node.func.attr in (
                    "_execute_tools_pipeline",
                    "_execute_think_tools_pipeline",
                ):
                    if any(k.arg == "approval_fn" for k in node.keywords):
                        passed.add(node.func.attr)
        assert passed == {
            "_execute_tools_pipeline",
            "_execute_think_tools_pipeline",
        }, passed

    def test_forward_count_is_exactly_four(self):
        # Two dispatch passes plus the two inner execute_with_tools
        # forwards; the pristine tree holds only the two inner ones.
        assert AX_SRC.count("approval_fn=approval_fn") == 4

    def test_execute_docstring_still_names_the_gate(self):
        assert "approval_fn: Optional per-invocation tool-approval gate" in AX_SRC


class TestApprovalFnBehaviour:
    def test_tools_path_forwards_the_sentinel(self, monkeypatch):
        monkeypatch.setattr(ax, "TOOL_EXECUTOR_AVAILABLE", True)
        cap = _CaptureToolExecutor()
        ae = ax.AgenticExecutor(executor=_GenExecutor(), tool_executor=cap)
        sentinel = object()
        list(
            ae.execute(
                "run ls in the workspace",
                _routing(),
                approval_fn=sentinel,
            )
        )
        assert len(cap.calls) == 1, "tools pipeline never reached the executor"
        assert cap.calls[0].get("approval_fn") is sentinel
        assert ae.last_pipeline == ax.PIPELINE_TOOLS

    def test_think_tools_path_forwards_the_sentinel(self, monkeypatch):
        monkeypatch.setattr(ax, "TOOL_EXECUTOR_AVAILABLE", True)
        cap = _CaptureToolExecutor()
        ae = ax.AgenticExecutor(executor=_GenExecutor(), tool_executor=cap)
        sentinel = object()
        list(
            ae.execute(
                "run ls in the workspace",
                _routing(),
                think=True,
                approval_fn=sentinel,
            )
        )
        assert len(cap.calls) == 1, "think+tools phase never ran the tools"
        assert cap.calls[0].get("approval_fn") is sentinel
        assert ae.last_pipeline == ax.PIPELINE_THINK_TOOLS

    def test_direct_helper_defaults_to_none(self, monkeypatch):
        monkeypatch.setattr(ax, "TOOL_EXECUTOR_AVAILABLE", True)
        cap = _CaptureToolExecutor()
        ae = ax.AgenticExecutor(executor=_GenExecutor(), tool_executor=cap)
        list(ae._execute_tools_pipeline("x", _routing(), None, None))
        assert len(cap.calls) == 1, "tools pipeline never reached the executor"
        assert "approval_fn" in cap.calls[0]
        assert cap.calls[0]["approval_fn"] is None

    def test_tools_pipeline_records_tool_calls_again(self, monkeypatch):
        monkeypatch.setattr(ax, "TOOL_EXECUTOR_AVAILABLE", True)
        cap = _CaptureToolExecutor()
        ae = ax.AgenticExecutor(executor=_GenExecutor(), tool_executor=cap)
        chunks = list(ae.execute("run ls in the workspace", _routing()))
        assert len(cap.calls) == 1
        assert "TOOLS DONE" in "".join(c for c in chunks if isinstance(c, str))
        assert len(ae.last_tool_calls) == 1


# ---------------------------------------------------------------------------
# 3. The MoE weights-residency override table (the kv_overrides recipe)
# ---------------------------------------------------------------------------


class TestWeightsCoercer:
    def test_sibling_coercer_exists_and_lowercases(self):
        fn = getattr(rg, "_as_weights_override_map", None)
        assert fn is not None, "_as_weights_override_map missing"
        out = fn({"Foo:Latest": "12.5", "BAR": 3})
        assert out == {"foo:latest": 12.5, "bar": 3.0}

    def test_non_mapping_yields_empty(self):
        fn = getattr(rg, "_as_weights_override_map", None)
        assert fn is not None, "_as_weights_override_map missing"
        assert fn("nope") == {}
        assert fn(17) == {}

    def test_bad_entries_dropped_never_guessed(self):
        fn = getattr(rg, "_as_weights_override_map", None)
        assert fn is not None, "_as_weights_override_map missing"
        out = fn({"a": "x", "b": -1, "c": 0, "d": 2.5})
        assert out == {"d": 2.5}

    def test_none_is_silently_empty(self):
        fn = getattr(rg, "_as_weights_override_map", None)
        assert fn is not None, "_as_weights_override_map missing"
        assert fn(None) == {}


class TestWeightsConfigLoader:
    def test_fields_default_empty(self):
        cfg = rg.GovernorConfig()
        assert hasattr(cfg, "weights_override_models"), "field missing"
        assert hasattr(cfg, "weights_override_families"), "field missing"
        assert cfg.weights_override_models == {}
        assert cfg.weights_override_families == {}

    def test_loader_parses_and_lowercases(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            "weights_overrides:\n"
            "  models:\n"
            '    "Foo:Latest": 12.5\n'
            "  families:\n"
            '    "Qwen": 9\n',
        )
        cfg = rg.load_config(p)
        assert hasattr(cfg, "weights_override_models"), "field missing"
        assert cfg.weights_override_models == {"foo:latest": 12.5}
        assert cfg.weights_override_families == {"qwen": 9.0}

    def test_non_mapping_section_keeps_defaults(self, tmp_path):
        cfg = rg.load_config(_write_yaml(tmp_path, "weights_overrides: 17\n"))
        assert hasattr(cfg, "weights_override_models"), "field missing"
        assert cfg.weights_override_models == {}
        assert cfg.weights_override_families == {}
        assert cfg == rg.GovernorConfig()

    def test_absent_section_is_byte_compatible(self, tmp_path):
        cfg = rg.load_config(_write_yaml(tmp_path, "kv_coefficient: 0.5\n"))
        assert hasattr(cfg, "weights_override_models"), "field missing"
        assert cfg == rg.GovernorConfig()

    def test_kv_tables_stay_orthogonal(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            "weights_overrides:\n  families:\n    moe: 7.0\n",
        )
        cfg = rg.load_config(p)
        assert hasattr(cfg, "weights_override_families"), "field missing"
        assert cfg.weights_override_families == {"moe": 7.0}
        assert cfg.kv_override_models == {}
        assert cfg.kv_override_families == {}
        assert cfg.kv_coefficient == 0.5


class TestWeightsResolver:
    def test_exact_hit_returns_the_override(self, tmp_path):
        gov = _gov(
            tmp_path,
            "weights_overrides:\n  models:\n    'moe-9b:latest': 7.5\n",
        )
        assert hasattr(gov, "resolve_weights_override"), "resolver missing"
        assert gov.resolve_weights_override("moe-9b:latest") == 7.5

    def test_longest_family_substring_wins(self, tmp_path):
        gov = _gov(
            tmp_path,
            "weights_overrides:\n"
            "  families:\n"
            "    qwen: 9.0\n"
            "    qwen3-moe: 4.0\n",
        )
        assert hasattr(gov, "resolve_weights_override"), "resolver missing"
        assert gov.resolve_weights_override("Qwen3-MoE-30B:latest") == 4.0

    def test_exact_beats_family(self, tmp_path):
        gov = _gov(
            tmp_path,
            "weights_overrides:\n"
            "  models:\n"
            "    'qwen3-moe-30b:latest': 5.5\n"
            "  families:\n"
            "    qwen3-moe: 4.0\n",
        )
        assert hasattr(gov, "resolve_weights_override"), "resolver missing"
        assert gov.resolve_weights_override("qwen3-moe-30b:latest") == 5.5

    def test_matching_is_case_insensitive(self, tmp_path):
        gov = _gov(
            tmp_path,
            "weights_overrides:\n  models:\n    'MoE-9B:Latest': 7.5\n",
        )
        assert hasattr(gov, "resolve_weights_override"), "resolver missing"
        assert gov.resolve_weights_override("moe-9b:latest") == 7.5

    def test_unknown_model_is_none(self, tmp_path):
        gov = _gov(
            tmp_path,
            "weights_overrides:\n  families:\n    moe: 7.0\n",
        )
        assert hasattr(gov, "resolve_weights_override"), "resolver missing"
        assert gov.resolve_weights_override("plain-13b:latest") is None

    def test_empty_or_none_model_is_none(self, tmp_path):
        gov = _gov(
            tmp_path,
            "weights_overrides:\n  families:\n    moe: 7.0\n",
        )
        assert hasattr(gov, "resolve_weights_override"), "resolver missing"
        assert gov.resolve_weights_override("") is None
        assert gov.resolve_weights_override(None) is None


class TestAdmitThreading:
    def test_admit_segment_resolves_for_model_and_extra(self):
        admit = _method(_RG_TREE, "ResourceGovernor", "admit")
        first_args = set()
        for node in ast.walk(admit):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "resolve_weights_override"
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                first_args.add(node.args[0].id)
        assert "model" in first_args, first_args
        assert "extra" in first_args, first_args

    def test_override_drives_the_ram_gate(self, tmp_path):
        gov = _gov(
            tmp_path,
            "weights_overrides:\n  families:\n    bigmoe: 50.0\n",
            available_kb=1 * 1024 * 1024,
        )
        d = gov.admit("bigmoe-9x:latest", requested_ctx=4096, caller="chat")
        assert d.admitted is False, "the override never reached the cost"
        assert d.reason == "ram_insufficient"

    def test_unknown_model_prices_exactly_as_today(self, tmp_path):
        gov = _gov(
            tmp_path,
            "weights_overrides:\n  families:\n    bigmoe: 50.0\n",
            available_kb=1 * 1024 * 1024,
        )
        d = gov.admit("plainmodel:latest", requested_ctx=4096, caller="chat")
        assert d.admitted is True

    def test_extra_models_thread_the_override_too(self, tmp_path):
        gov = _gov(
            tmp_path,
            "weights_overrides:\n  families:\n    bigmoe: 50.0\n",
            available_kb=1 * 1024 * 1024,
        )
        d = gov.admit(
            "plainmodel:latest",
            requested_ctx=4096,
            caller="chat",
            extra_models=["bigmoe-draft:latest"],
        )
        assert d.admitted is False, "the extra override never reached the cost"
        assert d.reason == "ram_insufficient"


class TestShippedYaml:
    def test_weights_section_present_in_shipped_yaml(self):
        assert "weights_overrides:" in YAML_SRC

    def test_shipped_yaml_still_mirrors_defaults(self):
        assert rg.load_config(_YAML_PATH) == rg.GovernorConfig()

    def test_resolver_is_none_over_the_shipped_tables(self, tmp_path):
        gov = rg.ResourceGovernor(
            config_path=_YAML_PATH,
            db_path=tmp_path / "gov_shipped.db",
            warmup=None,
            registry=None,
            clock=_Clock(),
            meminfo_path=_meminfo_file(tmp_path),
        )
        assert hasattr(gov, "resolve_weights_override"), "resolver missing"
        assert gov.resolve_weights_override("qwen3:32b") is None
        assert gov.resolve_weights_override("bigmoe-9x:latest") is None


# ---------------------------------------------------------------------------
# 4. The bloc document
# ---------------------------------------------------------------------------


class TestDebtLotDoc:
    def _text(self) -> str:
        assert _DOC_PATH.is_file(), "DEBT_LOT_S261.md missing"
        return _DOC_PATH.read_text(encoding="utf-8")

    def test_doc_exists_with_title(self):
        text = self._text()
        assert "# DEBT_LOT_S261" in text

    def test_doc_names_the_dead_gate_finding(self):
        text = self._text()
        assert (
            "the approval gate the S185 EX-02 fix intended never reached the tool loop"
            in text
        )

    def test_doc_names_the_seven_green_flips(self):
        text = self._text()
        assert "exactly seven pre-existing red nodes flip green" in text

    def test_doc_states_the_fail_secure_answer(self):
        text = self._text()
        assert "an unknown model prices exactly as today" in text

    def test_doc_states_the_resolution_order(self):
        text = self._text()
        assert (
            "exact model, then longest family substring, then the estimator answers"
            in text
        )

    def test_doc_keeps_calibration_host_assured(self):
        text = self._text()
        assert "calibration is host-assured" in text
        assert "## Findings register" in text

    def test_doc_is_pure_ascii(self):
        self._text().encode("ascii")


# ---------------------------------------------------------------------------
# 5. Reassertions (the constrained families' contracts stay live)
# ---------------------------------------------------------------------------


class TestReassertions:
    def test_bloc2_surface_still_defined_once(self):
        for name in (
            "def pressure_state",
            "def admit_or_wait",
            "def evict_model",
            "def _honour_conditional_eviction",
        ):
            assert RG_SRC.count(name) == 1, name
        assert RG_SRC.count("_honour_conditional_eviction(") == 3

    def test_governor_sentinels_hold(self):
        assert rg.checkpoint_before_apply is True
        assert rg.FEATURE_AVAILABLE is True

    def test_governor_source_stays_pure_ascii(self):
        RG_SRC.encode("ascii")

    def test_kv_coercer_path_byte_untouched(self):
        # The kv coercer keeps its own label strings; the weights table
        # arrived as a sibling, never a rewrite.
        assert 'logger.warning(\n                "kv_overrides sub-table is not a mapping; ignored"' in RG_SRC
        assert RG_SRC.count("def _as_kv_override_map") == 1

    def test_s185_tool_executor_contract_carried(self):
        tree = ast.parse(TE_SRC)
        fn = _method(tree, "ToolExecutor", "execute_with_tools")
        assert "approval_fn" in _param_names(fn)


# ---------------------------------------------------------------------------
# 6. Structure
# ---------------------------------------------------------------------------

_SELECTION = (
    "tests/test_background_execution.py",
    "tests/test_batch_reads.py",
    "tests/test_coding_agent.py",
    "tests/test_coding_security_audit.py",
    "tests/test_file_tools.py",
    "tests/test_quick_sandbox.py",
    "tests/test_routes_coding.py",
    "tests/test_s124_security_hardening.py",
    "tests/test_s125_security_hardening_p2.py",
    "tests/test_s138_security_debt.py",
    "tests/test_s140_coverage.py",
    "tests/test_s141_type_annotations.py",
    "tests/test_s175_dispatch.py",
    "tests/test_s175_loop.py",
    "tests/test_s176_tools.py",
    "tests/test_s177_manage_skills.py",
    "tests/test_s177_routes_agent.py",
    "tests/test_s183_a11y_labels.py",
    "tests/test_s183_audit_anchor.py",
    "tests/test_s183_bind_and_approval.py",
    "tests/test_s183_cert_revocation.py",
    "tests/test_s183_keyfile.py",
    "tests/test_s183_mode_failclosed.py",
    "tests/test_s183_packaging.py",
    "tests/test_s183_sandbox_env.py",
    "tests/test_s208_sync_bloc4.py",
    "tests/test_s209_sandbox_bloc0.py",
    "tests/test_s210_sandbox_bloc1.py",
    "tests/test_s211_sandbox_bloc2.py",
    "tests/test_s212_sandbox_bloc3.py",
    "tests/test_s213_sandbox_bloc4.py",
    "tests/test_s214_release.py",
    "tests/test_s215_estop.py",
    "tests/test_s216_pip06.py",
    "tests/test_s217_cleanup.py",
    "tests/test_s218_cleanup2.py",
    "tests/test_s219_ud04_rev2.py",
    "tests/test_s220_bk06.py",
    "tests/test_s221_governor_spec.py",
    "tests/test_s222_agt_spec.py",
    "tests/test_s223_governor_bloc0.py",
    "tests/test_s224_governor_bloc1.py",
    "tests/test_s225_governor_bloc2.py",
    "tests/test_s226_governor_bloc3.py",
    "tests/test_s227_governor_bloc4.py",
    "tests/test_s228_agt_lot1.py",
    "tests/test_s229_agt_lot2.py",
    "tests/test_s230_agt_lot3.py",
    "tests/test_s232_release.py",
    "tests/test_s233_cas7_spec.py",
    "tests/test_s234_remote_inference.py",
    "tests/test_s235_remote_channel_routes.py",
    "tests/test_s235_remote_grants.py",
    "tests/test_s235_remote_streaming.py",
    "tests/test_s236_release.py",
    "tests/test_s237_mobile_spec.py",
    "tests/test_s238_jni_spike_runbook.py",
    "tests/test_s239_pairing_ux_runbook.py",
    "tests/test_s240_chat_client_runbook.py",
    "tests/test_s241_sync_client_runbook.py",
    "tests/test_s242_atrest_consistency.py",
    "tests/test_s243_notes_data_layer.py",
    "tests/test_s244_manage_notes.py",
    "tests/test_s245_notes_route.py",
    "tests/test_s246_note_actions.py",
    "tests/test_s247_note_actions_route.py",
    "tests/test_s248_notes_ui.py",
    "tests/test_s249_notes_attachments_route.py",
    "tests/test_s250_transcription.py",
    "tests/test_s251_caption.py",
    "tests/test_s252_notes_record.py",
    "tests/test_s253_media_ui.py",
    "tests/test_s254_drawing_ui.py",
    "tests/test_s255_release.py",
    "tests/test_s256_mobile_allowed.py",
    "tests/test_s257_notes_publish_glue.py",
    "tests/test_s258_pairing_device_class.py",
    "tests/test_s259_inference_perf.py",
    "tests/test_s260_ui_toggles.py",
    "tests/test_s261_debt_lot.py",
    "tests/test_sandbox_api.py",
    "tests/test_sandbox_manager.py",
    "tests/test_working_memory.py",
)


class TestStructure:
    def test_edited_sources_parse(self):
        ast.parse(AX_SRC)
        ast.parse(RG_SRC)

    def test_edited_and_new_files_pure_ascii(self):
        AX_SRC.encode("ascii")
        RG_SRC.encode("ascii")
        Path(__file__).read_text(encoding="utf-8").encode("ascii")

    def test_selection_literal_self_check(self):
        assert len(_SELECTION) == 83
        assert len(set(_SELECTION)) == 83
        for rel in _SELECTION:
            assert (_BASE / rel).is_file(), rel
        assert "tests/test_s261_debt_lot.py" in _SELECTION

    def test_version_held(self):
        assert '__version__ = "3.12.0"' in _VER_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
