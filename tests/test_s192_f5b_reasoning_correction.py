#!/usr/bin/env python3
"""
S192 F5b tests -- reasoning / self_correction / structured_output
(RSN-01..RSN-04, SCR-01/SCR-02, SOU-01).

Loaders: reasoning.py is loaded under a stub package with the real
json_repair preloaded as its sibling (so the RSN-03 relative import
resolves); self_correction.py and structured_output.py load standalone
(their relative/optional imports are guarded; pydantic is a verification
dep installed with --break-system-packages). ollama is absent from the
container, so the LLM paths are exercised by setting OLLAMA_AVAILABLE and
injecting a fake ollama module attribute -- which also proves the
both-form parsing on object-shaped responses (the pre-fix failure mode).
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "opti_oignon"


# =============================================================================
# Loaders
# =============================================================================

def _load_under_stub_pkg(mod_basename: str, pkg_name: str):
    """Load opti_oignon/<mod_basename>.py as <pkg_name>.<mod_basename>."""
    mod_name = f"{pkg_name}.{mod_basename}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(PKG_DIR)]
        sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(
        mod_name, PKG_DIR / f"{mod_basename}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # register before exec (3.13 idiom)
    spec.loader.exec_module(module)
    return module


def _load_reasoning():
    # Preload the real json_repair as the sibling so the RSN-03 lazy
    # relative import resolves inside the stub package.
    _load_under_stub_pkg("json_repair", "oo_s192_pkgb")
    return _load_under_stub_pkg("reasoning", "oo_s192_pkgb")


def _load_self_correction():
    return _load_under_stub_pkg("self_correction", "oo_s192_pkgb")


def _load_structured_output():
    return _load_under_stub_pkg("structured_output", "oo_s192_pkgb")


# =============================================================================
# Fakes
# =============================================================================

def _object_chat_response(text: str):
    """Object-shaped ollama.chat response (no .get, not subscriptable)."""
    return SimpleNamespace(message=SimpleNamespace(content=text))


def _object_generate_response(text: str):
    """Object-shaped ollama.generate response."""
    return SimpleNamespace(response=text)


class _FakeClient:
    """Records the timeout it was built with; returns object-form replies."""

    instances: list["_FakeClient"] = []

    def __init__(self, timeout=None, **kwargs):
        self.timeout = timeout
        _FakeClient.instances.append(self)

    def chat(self, model, messages, options=None, **kwargs):
        return _object_chat_response("OBJECT-FORM-REPLY")


def _fake_ollama_module(generate_text: str = ""):
    fake = types.ModuleType("fake_ollama")
    fake.Client = _FakeClient
    fake.generate = lambda **kw: _object_generate_response(generate_text)
    fake.chat = lambda **kw: _object_chat_response("module-level")
    return fake


# =============================================================================
# RSN-01 / RSN-02 -- both-form chat parse and enforced timeout
# =============================================================================

class TestRsn01Rsn02CallLlm:
    def test_reply_text_both_forms(self):
        mod = _load_reasoning()
        assert mod._reply_text({"message": {"content": "hi"}}) == "hi"
        assert mod._reply_text(_object_chat_response("hi")) == "hi"
        assert mod._reply_text(None) == ""
        assert mod._reply_text("raw") == "raw"
        assert mod._reply_text({"message": None}) == ""

    def test_call_llm_object_form_and_timeout(self, monkeypatch):
        mod = _load_reasoning()
        monkeypatch.setattr(mod, "OLLAMA_AVAILABLE", True)
        monkeypatch.setattr(mod, "ollama", _fake_ollama_module(), raising=False)
        _FakeClient.instances.clear()

        eng = mod.ReasoningEngine(
            config=mod.ReasoningConfig(timeout_per_step=42)
        )
        # Pre-fix: response.get(...) raised AttributeError on object form
        # and the configured timeout was computed but never wired.
        out = eng._call_llm([{"role": "user", "content": "q"}])
        assert out == "OBJECT-FORM-REPLY"
        assert len(_FakeClient.instances) == 1
        assert _FakeClient.instances[0].timeout == 42

        # Client cache: same timeout reuses, a new timeout builds a second.
        eng._call_llm([{"role": "user", "content": "q"}])
        assert len(_FakeClient.instances) == 1
        eng._call_llm([{"role": "user", "content": "q"}], timeout=7)
        assert len(_FakeClient.instances) == 2
        assert _FakeClient.instances[1].timeout == 7


# =============================================================================
# RSN-03 -- json_repair fallback in the reasoning JSON parse
# =============================================================================

class TestRsn03ParseRepairFallback:
    def test_trailing_prose_repaired(self):
        mod = _load_reasoning()
        eng = mod.ReasoningEngine()
        text = 'Sure! [{"title": "A", "question": "B"}] Hope this helps.'
        # Pre-fix: trailing prose after the array -> JSONDecodeError -> None
        # -> silent single-step fallback.
        parsed = eng._parse_json_response(text)
        assert parsed == [{"title": "A", "question": "B"}]

    def test_single_quotes_repaired(self):
        mod = _load_reasoning()
        eng = mod.ReasoningEngine()
        parsed = eng._parse_json_response("[{'approach': 'X'}]")
        assert parsed == [{"approach": "X"}]

    def test_hopeless_text_returns_none(self):
        mod = _load_reasoning()
        eng = mod.ReasoningEngine()
        assert eng._parse_json_response("no json here at all") is None


# =============================================================================
# RSN-04 -- strategy resolved from config
# =============================================================================

class TestRsn04StrategyResolution:
    def _engine_with_recorders(self, mod, default_strategy):
        eng = mod.ReasoningEngine(
            config=mod.ReasoningConfig(default_strategy=default_strategy)
        )
        calls = []

        def _mk(name):
            def _run(question, model=None, on_step=None, **kw):
                calls.append(name)
                return mod.ReasoningResult(strategy=name, final_answer=name)
            return _run

        eng.decompose_and_solve = _mk("decompose")
        eng.tree_of_thought = _mk("tree_of_thought")
        eng.self_consistency = _mk("self_consistency")
        return eng, calls

    def test_config_field_parsed(self):
        mod = _load_reasoning()
        cfg = mod.ReasoningConfig.from_dict(
            {"reasoning": {"default_strategy": "tree_of_thought"}}
        )
        assert cfg.default_strategy == "tree_of_thought"
        assert mod.ReasoningConfig().default_strategy == "decompose"

    def test_none_resolves_to_config_default(self):
        mod = _load_reasoning()
        eng, calls = self._engine_with_recorders(mod, "self_consistency")
        list(eng.execute_reasoning("q", strategy=None))
        # Pre-fix: the signature default was "decompose" and the executor
        # hardcoded it, so the other strategies were unreachable.
        assert calls == ["self_consistency"]

    def test_explicit_strategy_still_wins(self):
        mod = _load_reasoning()
        eng, calls = self._engine_with_recorders(mod, "self_consistency")
        list(eng.execute_reasoning("q", strategy="tree_of_thought"))
        assert calls == ["tree_of_thought"]

    def test_executor_no_longer_hardcodes_decompose(self):
        # Source pin (heavy import chain): the executor passes strategy=None.
        src = (PKG_DIR / "agentic_executor.py").read_text(encoding="utf-8")
        assert 'strategy="decompose"' not in src
        assert "strategy=None," in src

    def test_yaml_documents_default_strategy(self):
        import yaml
        data = yaml.safe_load(
            (PKG_DIR / "config" / "reasoning.yaml").read_text(encoding="utf-8")
        )
        assert data["reasoning"]["default_strategy"] == "decompose"


# =============================================================================
# SCR-01 -- both-form generate parse on the four self_correction sites
# =============================================================================

class TestScr01GenerateParse:
    def test_generate_text_both_forms(self):
        mod = _load_self_correction()
        assert mod._generate_text({"response": "ok"}) == "ok"
        assert mod._generate_text(_object_generate_response("ok")) == "ok"
        assert mod._generate_text(None) == ""
        assert mod._generate_text("raw") == "raw"

    def test_check_facts_object_form(self, monkeypatch):
        mod = _load_self_correction()
        monkeypatch.setattr(mod, "OLLAMA_AVAILABLE", True)
        monkeypatch.setattr(
            mod, "ollama",
            _fake_ollama_module('{"flags": [], "confidence": 0.9}'),
            raising=False,
        )
        eng = mod.SelfCorrectionEngine()
        res = eng.check_facts("some response text")
        # Pre-fix: result.get(...) raised AttributeError -> swallowed ->
        # the default FactCheckResult(confidence=0.5).
        assert res.confidence == 0.9

    def test_generate_correction_object_form(self, monkeypatch):
        mod = _load_self_correction()
        monkeypatch.setattr(mod, "OLLAMA_AVAILABLE", True)
        corrected_text = "This is the corrected response with substance."
        monkeypatch.setattr(
            mod, "ollama", _fake_ollama_module(corrected_text),
            raising=False,
        )
        eng = mod.SelfCorrectionEngine()
        compliance = mod.ComplianceResult(
            score=0.1,
            instructions_found=["format: json"],
            checks=[mod.InstructionCheck(
                instruction="format: json", satisfied=False,
                explanation="missing",
            )],
            satisfied_count=0,
            total_count=1,
        )
        out = eng._generate_correction(
            "msg", "bad response", compliance, None, None, "m",
        )
        # Pre-fix: AttributeError -> swallowed -> None -> the correction
        # loop exited at iteration 1 without ever correcting.
        assert out == corrected_text


# =============================================================================
# SCR-02 -- LLM-returned scores clamped into [0, 1]
# =============================================================================

class TestScr02Clamp:
    def test_clamp01_unit(self):
        mod = _load_self_correction()
        assert mod._clamp01(5.0, 0.5) == 1.0
        assert mod._clamp01(-2, 0.5) == 0.0
        assert mod._clamp01(0.7, 0.5) == 0.7
        assert mod._clamp01("bogus", 0.5) == 0.5
        assert mod._clamp01(None, 0.3) == 0.3

    def test_quality_check_clamps_out_of_range(self, monkeypatch):
        mod = _load_self_correction()
        monkeypatch.setattr(mod, "OLLAMA_AVAILABLE", True)
        monkeypatch.setattr(
            mod, "ollama",
            _fake_ollama_module(
                '{"completeness": 5.0, "coherence": -2,'
                ' "hallucination_risk": 0.1, "issues": []}'
            ),
            raising=False,
        )
        eng = mod.SelfCorrectionEngine()
        res = eng._llm_quality_check("msg", "resp")
        # Pre-fix: completeness 5.0 leaked into overall_score and passed
        # the thresholds wrongly.
        assert res.completeness_score == 1.0
        assert res.coherence_score == 0.0
        assert 0.0 <= res.overall_score <= 1.0


# =============================================================================
# SOU-01 -- both-form access in structured_output
# =============================================================================

class TestSou01StructuredBothForms:
    def test_message_field_both_forms(self):
        mod = _load_structured_output()
        obj = SimpleNamespace(
            message=SimpleNamespace(content="C", thinking="T")
        )
        assert mod._message_field(obj, "content") == "C"
        assert mod._message_field(obj, "thinking") == "T"
        dct = {"message": {"content": "C2"}}
        assert mod._message_field(dct, "content") == "C2"
        assert mod._message_field(dct, "thinking") == ""
        assert mod._message_field(None, "content") == ""

    def test_generate_structured_dict_form(self, monkeypatch):
        mod = _load_structured_output()
        monkeypatch.setattr(mod, "OLLAMA_AVAILABLE", True)
        fake = types.ModuleType("fake_ollama")
        fake.chat = lambda **kw: {
            "message": {
                "content": (
                    '{"task_type": "explanation", "complexity": "simple"}'
                )
            }
        }
        monkeypatch.setattr(mod, "ollama", fake, raising=False)
        eng = mod.StructuredOutputEngine()
        res = eng.generate_structured(
            messages=[{"role": "user", "content": "q"}],
            schema=mod.TaskAnalysis,
        )
        # Pre-fix: response.message on a dict raised AttributeError ->
        # generic handler -> break, no retry, success=False.
        assert res.success is True
        assert res.data.task_type == "explanation"
        assert res.attempts == 1
