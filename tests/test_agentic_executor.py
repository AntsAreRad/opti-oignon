#!/usr/bin/env python3
"""
TESTS AGENTIC EXECUTOR - OPTI-OIGNON v1.5.0 (S45)
====================================================

Tests pour l'executeur agentique unifie.
Couvre l'analyse de tache, la selection de pipeline,
l'integration des composants, et les fallbacks.

Usage:
    pytest tests/test_agentic_executor.py -v
    pytest tests/test_agentic_executor.py -v -k "quick"
"""

from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# ============================================================
# IMPORT DU MODULE SOUS TEST
# ============================================================
from opti_oignon.agentic_executor import (
    PIPELINE_CODE_VERIFY,
    PIPELINE_DIRECT,
    PIPELINE_THINK,
    PIPELINE_THINK_TOOLS,
    PIPELINE_TOOLS,
    PIPELINE_WEB_SEARCH,
    AgenticExecutor,
    _quick_classify,
    _select_pipeline,
    agentic_executor,
)

# ============================================================
# FIXTURES
# ============================================================

def _make_routing(model="qwen3:32b", task_type="general", temperature=0.3):
    """Cree un objet RoutingResult minimal pour les tests."""
    return SimpleNamespace(
        model=model,
        task_type=task_type,
        temperature=temperature,
        prompt_variant="standard",
        model_type="general",
        priority_used="primary",
        explanation="test",
        timeout=60,
    )


def _make_mock_executor(response_chunks=None, verification_results=None):
    """Cree un mock de l'Executor existant."""
    mock = MagicMock()
    mock.cancel = MagicMock()
    mock.reset = MagicMock()
    mock.last_verification_results = verification_results or []
    mock._last_tool_calls = []

    chunks = response_chunks or ["Hello", " world"]

    def _execute_gen(**kwargs):
        for chunk in chunks:
            yield chunk

    mock.execute = MagicMock(side_effect=_execute_gen)
    return mock


def _make_mock_tool_executor(
    should_use=True,
    tool_calls=None,
    response="Tool result",
):
    """Cree un mock du ToolExecutor."""
    mock = MagicMock()
    mock.should_use_tools = MagicMock(return_value=should_use)

    # Simuler ToolCallResult
    default_calls = tool_calls or []
    result = SimpleNamespace(
        response=response,
        tool_calls=default_calls,
        model="qwen3:32b",
        total_time=1.5,
    )
    mock.execute_with_tools = MagicMock(return_value=result)
    return mock


def _make_mock_structured_engine(analysis_data=None):
    """Cree un mock du StructuredOutputEngine."""
    mock = MagicMock()
    if analysis_data is not None:
        result = SimpleNamespace(success=True, data=analysis_data)
    else:
        result = SimpleNamespace(success=False, data=None)
    mock.generate_structured = MagicMock(return_value=result)
    return mock


def _make_mock_verification_engine(results=None, available=True):
    """Cree un mock du VerificationEngine."""
    mock = MagicMock()
    mock.available = available
    mock.verify_response_code_blocks = MagicMock(return_value=results or [])
    return mock


def _make_tool_call_result(
    name="web_search", success=True, result="search result",
    execution_time=0.5, reasoning="need info",
):
    """Cree un mock de ToolCallResult."""
    return SimpleNamespace(
        tool_name=name,
        arguments={"query": "test"},
        result=result,
        success=success,
        execution_time=execution_time,
        reasoning=reasoning,
    )


def _consume_generator(gen) -> list:
    """Consomme un generateur et retourne la liste des chunks."""
    return list(gen)


# ============================================================
# TESTS: SINGLETON ET INITIALISATION
# ============================================================

class TestSingleton:
    """Tests du singleton et de l'initialisation."""

    def test_singleton_exists(self):
        """Le singleton agentic_executor est cree."""
        assert agentic_executor is not None
        assert isinstance(agentic_executor, AgenticExecutor)

    def test_init_defaults(self):
        """Initialisation avec les composants par defaut."""
        ae = AgenticExecutor()
        assert ae._default_model == "qwen3:32b"
        assert ae._last_tool_calls == []
        assert ae._last_verification_results == []
        assert ae._last_pipeline == PIPELINE_DIRECT

    def test_init_custom_model(self):
        """Initialisation avec un modele personnalise."""
        ae = AgenticExecutor(default_model="nemotron-3-nano:30b")
        assert ae._default_model == "nemotron-3-nano:30b"

    def test_init_custom_components(self):
        """Initialisation avec des composants injectes."""
        mock_exec = MagicMock()
        mock_tool = MagicMock()
        mock_struct = MagicMock()
        mock_verif = MagicMock()

        ae = AgenticExecutor(
            executor=mock_exec,
            tool_executor=mock_tool,
            structured_engine=mock_struct,
            verification_engine=mock_verif,
        )
        assert ae._executor is mock_exec
        assert ae._tool_executor is mock_tool
        assert ae._structured_engine is mock_struct
        assert ae._verification_engine is mock_verif


# ============================================================
# TESTS: CLASSIFICATION HEURISTIQUE
# ============================================================

class TestQuickClassify:
    """Tests de la classification heuristique rapide."""

    def test_simple_question(self):
        """Question simple sans indicateurs speciaux."""
        result = _quick_classify("What is Python?")
        assert not result["needs_tools"]
        assert not result["needs_web"]
        assert not result["is_complex"]

    def test_tool_keywords_search(self):
        """Detection de besoin de recherche."""
        result = _quick_classify("Search for the latest news about AI")
        assert result["needs_tools"]
        assert result["needs_web"]

    def test_tool_keywords_execute(self):
        """Detection de besoin d'execution de code."""
        result = _quick_classify("Run this code for me please")
        assert result["needs_tools"]

    def test_tool_keywords_file(self):
        """Detection de besoin d'acces fichier."""
        result = _quick_classify("Read file config.yaml and show me")
        assert result["needs_tools"]

    def test_code_keywords(self):
        """Detection de question de code."""
        result = _quick_classify("Write a Python function to sort a list")
        assert result["is_code"]

    def test_code_block_detection(self):
        """Detection de blocs de code inline."""
        result = _quick_classify("Fix this:\n```python\ndef foo():\n  pass\n```")
        assert result["is_code"]

    def test_complexity_keywords(self):
        """Detection de question complexe."""
        result = _quick_classify("Explain the pros and cons of microservices architecture")
        assert result["is_complex"]

    def test_long_message_complexity(self):
        """Les messages longs sont consideres complexes."""
        long_msg = " ".join(["word"] * 100)
        result = _quick_classify(long_msg)
        assert result["is_complex"]

    def test_french_keywords(self):
        """Detection de mots-cles en francais."""
        result = _quick_classify("Cherche sur internet les actualites du jour")
        assert result["needs_tools"]
        assert result["needs_web"]

    def test_mixed_indicators(self):
        """Message avec plusieurs indicateurs."""
        result = _quick_classify(
            "Search the web and explain step by step how to optimize Python code"
        )
        assert result["needs_web"]
        assert result["is_code"]
        assert result["is_complex"]


# ============================================================
# TESTS: SELECTION DE PIPELINE
# ============================================================

class TestSelectPipeline:
    """Tests de la selection de pipeline."""

    def test_direct_simple(self):
        """Question simple -> pipeline direct."""
        pipeline = _select_pipeline(
            {"needs_tools": False, "needs_web": False,
             "is_code": False, "is_complex": False},
            think_override=None, web_search_override=None,
            tool_executor_available=True, verification_available=True,
        )
        assert pipeline == PIPELINE_DIRECT

    def test_tools_pipeline(self):
        """Besoin d'outils -> pipeline tools."""
        pipeline = _select_pipeline(
            {"needs_tools": True, "needs_web": False,
             "is_code": False, "is_complex": False},
            think_override=None, web_search_override=None,
            tool_executor_available=True, verification_available=True,
        )
        assert pipeline == PIPELINE_TOOLS

    def test_tools_fallback_when_unavailable(self):
        """Besoin d'outils mais tool_executor indisponible -> direct."""
        pipeline = _select_pipeline(
            {"needs_tools": True, "needs_web": False,
             "is_code": False, "is_complex": False},
            think_override=None, web_search_override=None,
            tool_executor_available=False, verification_available=True,
        )
        assert pipeline == PIPELINE_DIRECT

    def test_web_search_pipeline(self):
        """Besoin de recherche web -> pipeline web_search."""
        pipeline = _select_pipeline(
            {"needs_tools": True, "needs_web": True,
             "is_code": False, "is_complex": False},
            think_override=None, web_search_override=None,
            tool_executor_available=True, verification_available=True,
        )
        assert pipeline == PIPELINE_WEB_SEARCH

    def test_code_verify_pipeline(self):
        """Question de code -> pipeline code_verify."""
        pipeline = _select_pipeline(
            {"needs_tools": False, "needs_web": False,
             "is_code": True, "is_complex": False},
            think_override=None, web_search_override=None,
            tool_executor_available=True, verification_available=True,
        )
        assert pipeline == PIPELINE_CODE_VERIFY

    def test_code_verify_fallback_no_verifier(self):
        """Code sans verifier -> direct."""
        pipeline = _select_pipeline(
            {"needs_tools": False, "needs_web": False,
             "is_code": True, "is_complex": False},
            think_override=None, web_search_override=None,
            tool_executor_available=True, verification_available=False,
        )
        assert pipeline == PIPELINE_DIRECT

    def test_think_pipeline(self):
        """Question complexe -> pipeline think."""
        pipeline = _select_pipeline(
            {"needs_tools": False, "needs_web": False,
             "is_code": False, "is_complex": True},
            think_override=None, web_search_override=None,
            tool_executor_available=True, verification_available=True,
        )
        assert pipeline == PIPELINE_THINK

    def test_think_tools_pipeline(self):
        """Complexe + outils -> pipeline think_tools."""
        pipeline = _select_pipeline(
            {"needs_tools": True, "needs_web": False,
             "is_code": False, "is_complex": True},
            think_override=None, web_search_override=None,
            tool_executor_available=True, verification_available=True,
        )
        assert pipeline == PIPELINE_THINK_TOOLS

    def test_think_override_true(self):
        """think=True force le pipeline think."""
        pipeline = _select_pipeline(
            {"needs_tools": False, "needs_web": False,
             "is_code": False, "is_complex": False},
            think_override=True, web_search_override=None,
            tool_executor_available=True, verification_available=True,
        )
        assert pipeline == PIPELINE_THINK

    def test_think_override_true_with_tools(self):
        """think=True avec besoin d'outils -> think_tools."""
        pipeline = _select_pipeline(
            {"needs_tools": True, "needs_web": False,
             "is_code": False, "is_complex": False},
            think_override=True, web_search_override=None,
            tool_executor_available=True, verification_available=True,
        )
        assert pipeline == PIPELINE_THINK_TOOLS

    def test_web_search_override_true(self):
        """web_search=True force le pipeline web_search."""
        pipeline = _select_pipeline(
            {"needs_tools": False, "needs_web": False,
             "is_code": False, "is_complex": False},
            think_override=None, web_search_override=True,
            tool_executor_available=True, verification_available=True,
        )
        assert pipeline == PIPELINE_WEB_SEARCH

    def test_both_overrides_false(self):
        """think=False et web_search=False -> direct ou code_verify."""
        pipeline = _select_pipeline(
            {"needs_tools": True, "needs_web": True,
             "is_code": False, "is_complex": True},
            think_override=False, web_search_override=False,
            tool_executor_available=True, verification_available=True,
        )
        assert pipeline == PIPELINE_DIRECT

    def test_both_overrides_false_with_code(self):
        """think=False et web_search=False avec code -> code_verify."""
        pipeline = _select_pipeline(
            {"needs_tools": False, "needs_web": False,
             "is_code": True, "is_complex": False},
            think_override=False, web_search_override=False,
            tool_executor_available=True, verification_available=True,
        )
        assert pipeline == PIPELINE_CODE_VERIFY


# ============================================================
# TESTS: PROPRIETES
# ============================================================

class TestProperties:
    """Tests des proprietes de l'AgenticExecutor."""

    def test_available_with_executor(self):
        """available est True quand l'executor est present."""
        ae = AgenticExecutor(executor=MagicMock())
        assert ae.available is True

    def test_available_without_executor(self):
        """available est False quand l'executor est None."""
        ae = AgenticExecutor(executor=None)
        # Forcer _executor a None (le constructeur essaie le defaut)
        ae._executor = None
        assert ae.available is False

    def test_tool_executor_available(self):
        """tool_executor_available reflete la disponibilite."""
        ae = AgenticExecutor(tool_executor=MagicMock())
        assert ae.tool_executor_available is True

    def test_tool_executor_not_available(self):
        """tool_executor_available avec None."""
        ae = AgenticExecutor()
        ae._tool_executor = None
        assert ae.tool_executor_available is False

    def test_verification_available(self):
        """verification_available reflete la disponibilite."""
        mock_verif = MagicMock()
        mock_verif.available = True
        ae = AgenticExecutor(verification_engine=mock_verif)
        assert ae.verification_available is True

    def test_verification_not_available(self):
        """verification_available quand le moteur n'est pas pret."""
        mock_verif = MagicMock()
        mock_verif.available = False
        ae = AgenticExecutor(verification_engine=mock_verif)
        assert ae.verification_available is False

    def test_last_pipeline_default(self):
        """last_pipeline est 'direct' par defaut."""
        ae = AgenticExecutor()
        assert ae.last_pipeline == PIPELINE_DIRECT

    def test_last_tool_calls_empty_initially(self):
        """last_tool_calls est vide initialement."""
        ae = AgenticExecutor()
        assert ae.last_tool_calls == []

    def test_last_verification_results_empty_initially(self):
        """last_verification_results est vide initialement."""
        ae = AgenticExecutor()
        assert ae.last_verification_results == []


# ============================================================
# TESTS: PIPELINE DIRECT
# ============================================================

class TestDirectPipeline:
    """Tests du pipeline direct."""

    def test_simple_query_direct(self):
        """Question simple -> execute via Executor direct."""
        mock_exec = _make_mock_executor(["Hello", " World"])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None  # Pas d'outils

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Hi there", routing)
        )

        assert "Hello" in chunks
        assert " World" in chunks
        assert ae.last_pipeline == PIPELINE_DIRECT
        mock_exec.execute.assert_called_once()

    def test_direct_pipeline_no_executor(self):
        """Sans executor -> message d'erreur."""
        ae = AgenticExecutor(executor=None)
        ae._executor = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Hi", routing)
        )

        assert any("[ERR]" in str(c) for c in chunks)

    def test_direct_with_think_false(self):
        """think=False force le pipeline direct."""
        mock_exec = _make_mock_executor(["Response"])
        ae = AgenticExecutor(executor=mock_exec)

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Explain architecture", routing, think=False, web_search=False)
        )

        assert ae.last_pipeline == PIPELINE_DIRECT

    def test_thinking_chunks_forwarded(self):
        """Les chunks thinking sont transmis."""
        mock_exec = _make_mock_executor([
            ("thinking", "Let me think..."),
            "The answer is 42",
        ])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Complex question", routing, think=True)
        )

        assert ("thinking", "Let me think...") in chunks
        assert "The answer is 42" in chunks
        assert ae.last_pipeline == PIPELINE_THINK


# ============================================================
# TESTS: PIPELINE OUTILS
# ============================================================

class TestToolsPipeline:
    """Tests du pipeline outils."""

    def test_tools_pipeline_activated(self):
        """L'execution de code active le pipeline outils."""
        mock_exec = _make_mock_executor()
        tc = _make_tool_call_result(name="execute_code")
        mock_tool = _make_mock_tool_executor(
            tool_calls=[tc], response="Code output: 42"
        )
        ae = AgenticExecutor(
            executor=mock_exec, tool_executor=mock_tool,
        )

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Execute this code for me", routing)
        )

        assert ae.last_pipeline == PIPELINE_TOOLS
        assert "Code output: 42" in chunks
        assert len(ae.last_tool_calls) == 1
        mock_tool.execute_with_tools.assert_called_once()

    def test_tools_pipeline_callback(self):
        """Le callback on_tool_call est appele pour chaque outil."""
        mock_exec = _make_mock_executor()
        tc1 = _make_tool_call_result(name="read_file")
        tc2 = _make_tool_call_result(name="execute_code")
        mock_tool = _make_mock_tool_executor(
            tool_calls=[tc1, tc2], response="Done"
        )
        ae = AgenticExecutor(
            executor=mock_exec, tool_executor=mock_tool,
        )

        callback_calls = []

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute(
                "Read file config.yaml and run the code", routing,
                on_tool_call=lambda tc: callback_calls.append(tc),
            )
        )

        assert len(callback_calls) == 2
        assert callback_calls[0].tool_name == "read_file"
        assert callback_calls[1].tool_name == "execute_code"

    def test_tools_fallback_to_direct(self):
        """Si tool_executor est None, fallback vers direct."""
        mock_exec = _make_mock_executor(["Fallback response"])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        # Force la classification pour activer tools
        with patch.object(ae, '_classify_message', return_value={
            "needs_tools": True, "needs_web": False,
            "is_code": False, "is_complex": False,
        }):
            chunks = _consume_generator(
                ae.execute("Search for something", routing)
            )

        # Sans tool_executor, le pipeline retombe sur direct
        assert ae.last_pipeline == PIPELINE_DIRECT

    def test_tools_pipeline_error_fallback(self):
        """Erreur dans tool_executor -> fallback vers direct."""
        mock_exec = _make_mock_executor(["Fallback after error"])
        mock_tool = MagicMock()
        mock_tool.should_use_tools = MagicMock(return_value=True)
        mock_tool.execute_with_tools = MagicMock(
            side_effect=Exception("Tool error")
        )
        ae = AgenticExecutor(
            executor=mock_exec, tool_executor=mock_tool,
        )

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Execute this code", routing)
        )

        # Doit avoir un fallback
        assert "Fallback after error" in chunks


# ============================================================
# TESTS: PIPELINE CODE VERIFICATION
# ============================================================

class TestCodeVerifyPipeline:
    """Tests du pipeline de verification de code."""

    def test_code_verify_activated(self):
        """Question de code active la verification."""
        mock_exec = _make_mock_executor(
            ["```python\nprint('hello')\n```"],
            verification_results=[],
        )
        mock_verif = _make_mock_verification_engine(
            results=[SimpleNamespace(
                status="passed", language="python", iterations=1,
                errors_encountered=[], fixes_applied=[],
                execution_output="hello",
            )],
            available=True,
        )
        ae = AgenticExecutor(
            executor=mock_exec, verification_engine=mock_verif,
        )
        ae._tool_executor = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Write Python code to print hello", routing)
        )

        assert ae.last_pipeline == PIPELINE_CODE_VERIFY
        assert len(ae.last_verification_results) == 1
        assert ae.last_verification_results[0].status == "passed"

    def test_code_verify_uses_executor_results(self):
        """Si l'executor a deja verifie, on reutilise ses resultats."""
        existing_vr = SimpleNamespace(
            status="fixed", language="python", iterations=2,
            errors_encountered=["SyntaxError"],
            fixes_applied=["Added missing colon"],
            execution_output="ok",
        )
        mock_exec = _make_mock_executor(
            ["code here"],
            verification_results=[existing_vr],
        )
        mock_verif = _make_mock_verification_engine(available=True)
        ae = AgenticExecutor(
            executor=mock_exec, verification_engine=mock_verif,
        )
        ae._tool_executor = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Fix this Python code", routing)
        )

        # L'executor a deja verifie, on ne re-verifie pas
        assert ae.last_verification_results == [existing_vr]
        mock_verif.verify_response_code_blocks.assert_not_called()


# ============================================================
# TESTS: PIPELINE THINK
# ============================================================

class TestThinkPipeline:
    """Tests du pipeline think."""

    def test_think_auto_detect(self):
        """Question complexe active automatiquement le think mode."""
        mock_exec = _make_mock_executor([
            ("thinking", "Reasoning..."),
            "My analysis is...",
        ])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute(
                "Explain the pros and cons of microservices vs monolith architecture",
                routing,
            )
        )

        assert ae.last_pipeline == PIPELINE_THINK
        assert ("thinking", "Reasoning...") in chunks

    def test_think_forced(self):
        """think=True force le pipeline think."""
        mock_exec = _make_mock_executor(["Response"])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Hello", routing, think=True)
        )

        assert ae.last_pipeline == PIPELINE_THINK
        # Verifie que think=True a ete passe a l'executor
        call_kwargs = mock_exec.execute.call_args
        assert call_kwargs.kwargs.get("think") is True


# ============================================================
# TESTS: PIPELINE WEB SEARCH
# ============================================================

class TestWebSearchPipeline:
    """Tests du pipeline web search."""

    def test_web_search_auto_detect(self):
        """Requete de recherche active le pipeline web search."""
        mock_exec = _make_mock_executor(["Web results..."])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Search for latest news about AI today", routing)
        )

        assert ae.last_pipeline == PIPELINE_WEB_SEARCH

    def test_web_search_forced(self):
        """web_search=True force le pipeline web search."""
        mock_exec = _make_mock_executor(["Results"])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Tell me about cats", routing, web_search=True)
        )

        assert ae.last_pipeline == PIPELINE_WEB_SEARCH
        call_kwargs = mock_exec.execute.call_args
        assert call_kwargs.kwargs.get("web_search") is True


# ============================================================
# TESTS: PIPELINE THINK + TOOLS
# ============================================================

class TestThinkToolsPipeline:
    """Tests du pipeline think+tools combine."""

    def test_think_tools_activated(self):
        """Question complexe + outils -> think_tools."""
        mock_exec = _make_mock_executor([
            ("thinking", "Deep thought..."),
            "Initial analysis",
        ])
        tc = _make_tool_call_result()
        mock_tool = _make_mock_tool_executor(
            should_use=True, tool_calls=[tc], response="Tool enrichment",
        )
        ae = AgenticExecutor(
            executor=mock_exec, tool_executor=mock_tool,
        )

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute(
                "Explain step by step and search for the latest benchmarks",
                routing, think=True,
            )
        )

        assert ae.last_pipeline == PIPELINE_THINK_TOOLS
        assert ("thinking", "Deep thought...") in chunks
        assert len(ae.last_tool_calls) == 1


# ============================================================
# TESTS: ANALYSE DE TACHE VIA LLM
# ============================================================

class TestTaskAnalysis:
    """Tests de l'analyse de tache."""

    def test_analyze_task_without_engine(self):
        """Analyse retourne None sans StructuredOutputEngine."""
        ae = AgenticExecutor()
        ae._structured_engine = None
        result = ae.analyze_task("test message")
        assert result is None

    def test_analyze_task_success(self):
        """Analyse reussie via StructuredOutputEngine."""
        mock_analysis = SimpleNamespace(
            task_type="code_python",
            complexity="moderate",
            requires_tools=["code_execution"],
            requires_thinking=False,
            language="en",
            confidence=0.9,
        )
        mock_engine = _make_mock_structured_engine(mock_analysis)
        ae = AgenticExecutor(structured_engine=mock_engine)
        ae._structured_engine = mock_engine

        result = ae.analyze_task("Write a Python function")
        assert result is not None
        assert result.task_type == "code_python"

    def test_analyze_task_failure_returns_none(self):
        """Analyse echouee retourne None."""
        mock_engine = _make_mock_structured_engine(analysis_data=None)
        ae = AgenticExecutor(structured_engine=mock_engine)
        ae._structured_engine = mock_engine

        result = ae.analyze_task("something")
        assert result is None

    def test_classify_with_llm_analysis(self):
        """Classification avec analyse LLM integree."""
        mock_analysis = SimpleNamespace(
            task_type="code_python",
            complexity="complex",
            requires_tools=["web_search"],
            requires_thinking=True,
            language="en",
            confidence=0.95,
        )
        mock_engine = _make_mock_structured_engine(mock_analysis)
        ae = AgenticExecutor(structured_engine=mock_engine)
        ae._structured_engine = mock_engine

        classification = ae._classify_message(
            "Build a web scraper with error handling",
            use_llm=True,
        )

        assert classification["is_code"] is True
        assert classification["is_complex"] is True
        assert classification["needs_web"] is True
        assert ae.last_task_analysis is not None


# ============================================================
# TESTS: CANCEL
# ============================================================

class TestCancel:
    """Tests du mecanisme d'annulation."""

    def test_cancel_delegates_to_executor(self):
        """cancel() delegue a l'executor de base."""
        mock_exec = MagicMock()
        mock_exec.cancel = MagicMock()
        ae = AgenticExecutor(executor=mock_exec)

        ae.cancel()
        mock_exec.cancel.assert_called_once()

    def test_cancel_without_executor(self):
        """cancel() sans executor ne leve pas d'erreur."""
        ae = AgenticExecutor()
        ae._executor = None
        ae.cancel()  # Ne doit pas lever d'exception


# ============================================================
# TESTS: RESET
# ============================================================

class TestReset:
    """Tests du reset d'etat."""

    def test_reset_clears_state(self):
        """reset() reinitialise tous les etats."""
        ae = AgenticExecutor()
        ae._last_tool_calls = [MagicMock()]
        ae._last_verification_results = [MagicMock()]
        ae._last_pipeline = PIPELINE_TOOLS
        ae._last_task_analysis = MagicMock()
        ae._on_tool_call = MagicMock()

        ae.reset()

        assert ae._last_tool_calls == []
        assert ae._last_verification_results == []
        assert ae._last_pipeline == PIPELINE_DIRECT
        assert ae._last_task_analysis is None
        assert ae._on_tool_call is None


# ============================================================
# TESTS: BACKWARD COMPATIBILITY
# ============================================================

class TestBackwardCompatibility:
    """Tests de retrocompatibilite."""

    def test_executor_has_last_tool_calls(self):
        """L'Executor existant a l'attribut _last_tool_calls (S45)."""
        from opti_oignon.executor import Executor
        exec_instance = Executor()
        assert hasattr(exec_instance, '_last_tool_calls')
        assert exec_instance._last_tool_calls == []
        assert hasattr(exec_instance, 'last_tool_calls')
        assert exec_instance.last_tool_calls == []

    def test_executor_reset_clears_tool_calls(self):
        """reset() de l'Executor efface _last_tool_calls."""
        from opti_oignon.executor import Executor
        exec_instance = Executor()
        exec_instance._last_tool_calls = [MagicMock()]
        exec_instance.reset()
        assert exec_instance._last_tool_calls == []

    def test_agentic_respects_executor_interface(self):
        """L'AgenticExecutor respecte l'interface Generator de l'Executor."""
        mock_exec = _make_mock_executor(["chunk1", "chunk2"])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        gen = ae.execute("test", routing)

        # Doit etre un generateur
        assert hasattr(gen, '__next__')

        chunks = list(gen)
        assert "chunk1" in chunks
        assert "chunk2" in chunks


# ============================================================
# TESTS: GRACEFUL DEGRADATION
# ============================================================

class TestGracefulDegradation:
    """Tests de degradation gracieuse quand des composants sont absents."""

    def test_no_tool_executor_still_works(self):
        """Sans ToolExecutor, les requetes d'outils passent en direct."""
        mock_exec = _make_mock_executor(["Direct response"])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Search for something", routing)
        )

        # Pas de tool_executor, devrait router vers direct ou web_search
        assert any("Direct response" in str(c) or c == "Direct response" for c in chunks)

    def test_no_verification_engine_still_works(self):
        """Sans VerificationEngine, le code n'est pas verifie."""
        mock_exec = _make_mock_executor(["```python\nprint('hi')\n```"])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None
        ae._verification_engine = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Write Python code", routing)
        )

        assert ae.last_verification_results == []

    def test_no_structured_engine_uses_heuristics(self):
        """Sans StructuredOutputEngine, les heuristiques sont utilisees."""
        mock_exec = _make_mock_executor(["Response"])
        ae = AgenticExecutor(executor=mock_exec)
        ae._structured_engine = None
        ae._tool_executor = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Explain something", routing, use_llm_analysis=True)
        )

        # Doit fonctionner meme sans structured engine
        assert "Response" in chunks

    def test_executor_error_handled(self):
        """Erreur dans l'executor -> message d'erreur dans le flux."""
        mock_exec = MagicMock()
        mock_exec.cancel = MagicMock()
        mock_exec.last_verification_results = []

        def _error_gen(**kwargs):
            raise RuntimeError("LLM unavailable")

        mock_exec.execute = MagicMock(side_effect=_error_gen)
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("test", routing)
        )

        assert any("[Error" in str(c) for c in chunks)


# ============================================================
# TESTS: RESULTATS ACCESSIBLES APRES EXECUTION
# ============================================================

class TestResultAccessibility:
    """Tests que les resultats sont accessibles apres execution."""

    def test_tool_calls_accessible(self):
        """Les tool_calls sont accessibles apres execution."""
        mock_exec = _make_mock_executor()
        tc = _make_tool_call_result(name="execute_code")
        mock_tool = _make_mock_tool_executor(
            tool_calls=[tc], response="Result"
        )
        ae = AgenticExecutor(
            executor=mock_exec, tool_executor=mock_tool,
        )

        routing = _make_routing()
        _consume_generator(
            ae.execute("Run this code for me", routing)
        )

        assert len(ae.last_tool_calls) == 1
        assert ae.last_tool_calls[0].tool_name == "execute_code"

    def test_verification_results_accessible(self):
        """Les resultats de verification sont accessibles."""
        vr = SimpleNamespace(
            status="passed", language="python", iterations=1,
            errors_encountered=[], fixes_applied=[],
            execution_output="42",
        )
        mock_exec = _make_mock_executor(
            ["code output"],
            verification_results=[vr],
        )
        mock_verif = _make_mock_verification_engine(available=True)
        ae = AgenticExecutor(
            executor=mock_exec, verification_engine=mock_verif,
        )
        ae._tool_executor = None

        routing = _make_routing()
        _consume_generator(
            ae.execute("Write Python code", routing)
        )

        assert len(ae.last_verification_results) == 1

    def test_pipeline_name_accessible(self):
        """Le nom du pipeline utilise est accessible."""
        mock_exec = _make_mock_executor(["Response"])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        _consume_generator(
            ae.execute("Simple hello", routing)
        )

        assert ae.last_pipeline in (
            PIPELINE_DIRECT, PIPELINE_TOOLS, PIPELINE_CODE_VERIFY,
            PIPELINE_THINK, PIPELINE_WEB_SEARCH, PIPELINE_THINK_TOOLS,
        )

    def test_results_reset_between_executions(self):
        """Les resultats sont reinitialises entre les executions."""
        mock_exec = _make_mock_executor()
        tc = _make_tool_call_result()
        mock_tool = _make_mock_tool_executor(
            tool_calls=[tc], response="Result"
        )
        ae = AgenticExecutor(
            executor=mock_exec, tool_executor=mock_tool,
        )

        routing = _make_routing()

        # Premiere execution avec outils (utiliser mot-cle d'execution, pas recherche)
        _consume_generator(ae.execute("Execute this code", routing))
        assert len(ae.last_tool_calls) == 1

        # Deuxieme execution simple
        ae._tool_executor = None  # Pas d'outils cette fois
        _consume_generator(ae.execute("Hello", routing))
        assert len(ae.last_tool_calls) == 0


# ============================================================
# TESTS: ROUTES_CHAT INTEGRATION
# ============================================================

class TestRoutesChatIntegration:
    """Tests d'integration avec routes_chat.py (verification d'import)."""

    def test_agentic_import_in_routes(self):
        """L'import de l'agentic executor dans routes_chat fonctionne."""
        try:
            from opti_oignon.api import routes_chat
            assert hasattr(routes_chat, 'AGENTIC_EXECUTOR_AVAILABLE')
            assert hasattr(routes_chat, '_agentic_executor')
        except ImportError as e:
            pytest.skip(f"routes_chat import failed (missing dep): {e}")

    def test_agentic_executor_available_flag(self):
        """Le flag AGENTIC_EXECUTOR_AVAILABLE est coherent."""
        try:
            from opti_oignon.api import routes_chat
            if routes_chat.AGENTIC_EXECUTOR_AVAILABLE:
                assert routes_chat._agentic_executor is not None
        except ImportError as e:
            pytest.skip(f"routes_chat import failed (missing dep): {e}")


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
