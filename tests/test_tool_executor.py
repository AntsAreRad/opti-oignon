#!/usr/bin/env python3
"""
Tests pour l'executeur d'outils ReAct (S44).

Couvre:
- Requete simple sans outil
- Requete declenchant web_search
- Requete declenchant execute_code
- Appels multiples sequentiels
- Limite max_tool_calls respectee
- Gestion d'erreur d'execution d'outil
- Outil introuvable
- Boucle ReAct complete avec reponse finale
- Singleton tool_executor
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from opti_oignon.tool_executor import (
    ToolCallResult,
    ToolDecision,
    ToolExecutionResult,
    ToolExecutor,
    tool_executor,
)
from opti_oignon.tool_registry import (
    ToolDefinition,
    ToolParam,
    ToolRegistry,
)

# =============================================================================
# MOCKS
# =============================================================================

class MockStructuredResult:
    """Mock pour StructuredResult."""
    def __init__(self, success=True, data=None):
        self.success = success
        self.data = data
        self.raw_response = ""
        self.attempts = 1
        self.errors = []


class MockStructuredEngine:
    """Mock du StructuredOutputEngine avec reponses configurables."""

    def __init__(self, decisions=None):
        """
        Args:
            decisions: Liste de ToolDecision a retourner dans l'ordre.
                       Chaque appel consomme une decision.
                       None dans la liste = retourner un echec.
        """
        self._decisions = list(decisions or [])
        self._call_count = 0

    def generate_structured(self, messages, schema, model=None,
                            extra_system_prompt="", temperature=0.0,
                            max_retries=2):
        idx = self._call_count
        self._call_count += 1

        if idx < len(self._decisions):
            decision = self._decisions[idx]
            if decision is None:
                return MockStructuredResult(success=False)
            return MockStructuredResult(success=True, data=decision)
        # Par defaut: pas d'outil
        return MockStructuredResult(
            success=True,
            data=ToolDecision(tool_name="none", reasoning="No tool needed"),
        )


def _make_registry_with_tools():
    """Cree un registre avec des outils mock."""
    reg = ToolRegistry()

    def mock_search(query, max_results=5):
        return f"Search results for: {query}"

    def mock_execute(code, language="python", timeout=30):
        return f"Output: {code[:50]}"

    def mock_failing_tool(arg=""):
        raise RuntimeError("Tool exploded")

    reg.register(ToolDefinition(
        name="web_search",
        description="Search the web",
        parameters={
            "query": ToolParam(name="query", type="string",
                               description="Search query", required=True),
            "max_results": ToolParam(name="max_results", type="int",
                                     description="Max results",
                                     required=False, default=5),
        },
        handler=mock_search,
        enabled=True,
    ))

    reg.register(ToolDefinition(
        name="execute_code",
        description="Execute code",
        parameters={
            "code": ToolParam(name="code", type="string",
                              description="Code", required=True),
            "language": ToolParam(name="language", type="string",
                                  description="Language",
                                  required=False, default="python"),
            "timeout": ToolParam(name="timeout", type="int",
                                 description="Timeout",
                                 required=False, default=30),
        },
        handler=mock_execute,
        enabled=True,
    ))

    reg.register(ToolDefinition(
        name="failing_tool",
        description="A tool that always fails",
        parameters={
            "arg": ToolParam(name="arg", type="string",
                             description="Argument",
                             required=False, default=""),
        },
        handler=mock_failing_tool,
        enabled=True,
    ))

    reg.register(ToolDefinition(
        name="disabled_tool",
        description="Disabled tool",
        handler=lambda: "nope",
        enabled=False,
    ))

    return reg


# =============================================================================
# SHOULD_USE_TOOLS
# =============================================================================

class TestShouldUseTools:
    """Tests pour la detection heuristique de besoin d'outils."""

    def test_search_keyword_detected(self):
        """Mot-cle de recherche detecte."""
        reg = _make_registry_with_tools()
        executor = ToolExecutor(registry=reg)
        assert executor.should_use_tools("search for Python tutorials") is True

    def test_code_keyword_detected(self):
        """Mot-cle d'execution de code detecte."""
        reg = _make_registry_with_tools()
        executor = ToolExecutor(registry=reg)
        assert executor.should_use_tools("execute this code for me") is True

    def test_no_keyword_no_tool(self):
        """Pas de mot-cle -> pas d'outil."""
        reg = _make_registry_with_tools()
        executor = ToolExecutor(registry=reg)
        assert executor.should_use_tools("What is the meaning of life?") is False

    def test_no_registry_no_tool(self):
        """Pas de registre -> pas d'outil."""
        executor = ToolExecutor(registry=None)
        assert executor.should_use_tools("search something") is False

    def test_empty_registry_no_tool(self):
        """Registre vide -> pas d'outil."""
        reg = ToolRegistry()
        executor = ToolExecutor(registry=reg)
        assert executor.should_use_tools("search something") is False

    def test_french_keyword_detected(self):
        """Mot-cle francais detecte."""
        reg = _make_registry_with_tools()
        executor = ToolExecutor(registry=reg)
        assert executor.should_use_tools("cherche des infos sur Python") is True


# =============================================================================
# EXECUTE_TOOL (interne)
# =============================================================================

class TestExecuteTool:
    """Tests pour l'execution d'un outil individuel."""

    def test_successful_execution(self):
        """Execution reussie d'un outil."""
        reg = _make_registry_with_tools()
        executor = ToolExecutor(registry=reg)
        result = executor._execute_tool(
            "web_search", {"query": "test"}, "Testing search"
        )
        assert result.success is True
        assert "Search results for: test" in result.result
        assert result.reasoning == "Testing search"
        assert result.execution_time >= 0

    def test_tool_not_found(self):
        """Outil inexistant."""
        reg = _make_registry_with_tools()
        executor = ToolExecutor(registry=reg)
        result = executor._execute_tool("nonexistent", {})
        assert result.success is False
        assert "not found" in result.result.lower()

    def test_disabled_tool(self):
        """Outil desactive."""
        reg = _make_registry_with_tools()
        executor = ToolExecutor(registry=reg)
        result = executor._execute_tool("disabled_tool", {})
        assert result.success is False
        assert "disabled" in result.result.lower()

    def test_tool_handler_error(self):
        """Handler qui leve une exception."""
        reg = _make_registry_with_tools()
        executor = ToolExecutor(registry=reg)
        result = executor._execute_tool("failing_tool", {})
        assert result.success is False
        assert "error" in result.result.lower()

    def test_missing_required_param(self):
        """Parametre requis manquant."""
        reg = _make_registry_with_tools()
        executor = ToolExecutor(registry=reg)
        # web_search requiert "query"
        result = executor._execute_tool("web_search", {})
        assert result.success is False
        assert "missing" in result.result.lower() or "required" in result.result.lower()

    def test_default_params_applied(self):
        """Les valeurs par defaut sont appliquees."""
        reg = _make_registry_with_tools()
        executor = ToolExecutor(registry=reg)
        result = executor._execute_tool(
            "execute_code", {"code": "print('hi')"}
        )
        assert result.success is True
        # language et timeout ont des defaults


# =============================================================================
# EXECUTE_WITH_TOOLS
# =============================================================================

class TestExecuteWithTools:
    """Tests de la boucle ReAct complete."""

    def test_no_tool_needed(self):
        """Le LLM decide qu'aucun outil n'est necessaire."""
        reg = _make_registry_with_tools()
        engine = MockStructuredEngine(decisions=[
            ToolDecision(tool_name="none", reasoning="Direct answer suffices"),
        ])
        executor = ToolExecutor(
            registry=reg,
            structured_engine=engine,
        )

        with patch("opti_oignon.tool_executor.OLLAMA_AVAILABLE", True), \
             patch("opti_oignon.tool_executor.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "Direct response without tools."
            mock_ollama.chat.return_value = mock_response

            result = executor.execute_with_tools("What is 2+2?", model="test")

        assert isinstance(result, ToolExecutionResult)
        assert len(result.tool_calls) == 0
        assert result.response == "Direct response without tools."

    def test_single_tool_call(self):
        """Un seul appel d'outil puis reponse finale."""
        reg = _make_registry_with_tools()
        engine = MockStructuredEngine(decisions=[
            ToolDecision(
                tool_name="web_search",
                arguments={"query": "latest Python version"},
                reasoning="Need current info",
            ),
            ToolDecision(tool_name="none", reasoning="Got the info"),
        ])
        executor = ToolExecutor(
            registry=reg,
            structured_engine=engine,
        )

        with patch("opti_oignon.tool_executor.OLLAMA_AVAILABLE", True), \
             patch("opti_oignon.tool_executor.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "Python 3.13 is the latest."
            mock_ollama.chat.return_value = mock_response

            result = executor.execute_with_tools(
                "What is the latest Python version?", model="test"
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "web_search"
        assert result.tool_calls[0].success is True
        assert result.response == "Python 3.13 is the latest."

    def test_multiple_tool_calls(self):
        """Plusieurs appels d'outils sequentiels."""
        reg = _make_registry_with_tools()
        engine = MockStructuredEngine(decisions=[
            ToolDecision(
                tool_name="web_search",
                arguments={"query": "R tidyverse"},
                reasoning="Search first",
            ),
            ToolDecision(
                tool_name="execute_code",
                arguments={"code": "print(1+1)"},
                reasoning="Then test code",
            ),
            ToolDecision(tool_name="none", reasoning="Done"),
        ])
        executor = ToolExecutor(
            registry=reg,
            structured_engine=engine,
        )

        with patch("opti_oignon.tool_executor.OLLAMA_AVAILABLE", True), \
             patch("opti_oignon.tool_executor.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "Combined answer."
            mock_ollama.chat.return_value = mock_response

            result = executor.execute_with_tools("Help with R", model="test")

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].tool_name == "web_search"
        assert result.tool_calls[1].tool_name == "execute_code"

    def test_max_tool_calls_respected(self):
        """La limite max_tool_calls est respectee."""
        reg = _make_registry_with_tools()
        # Le LLM veut toujours un outil (boucle infinie potentielle)
        engine = MockStructuredEngine(decisions=[
            ToolDecision(
                tool_name="web_search",
                arguments={"query": f"search {i}"},
                reasoning="More info",
            )
            for i in range(10)
        ])
        executor = ToolExecutor(
            registry=reg,
            structured_engine=engine,
            max_tool_calls=3,
        )

        with patch("opti_oignon.tool_executor.OLLAMA_AVAILABLE", True), \
             patch("opti_oignon.tool_executor.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "Final answer."
            mock_ollama.chat.return_value = mock_response

            result = executor.execute_with_tools("Search a lot", model="test")

        # Maximum 3 appels meme si le LLM en demande plus
        assert len(result.tool_calls) <= 3

    def test_tool_error_stops_loop(self):
        """Une erreur d'outil arrete la boucle."""
        reg = _make_registry_with_tools()
        engine = MockStructuredEngine(decisions=[
            ToolDecision(
                tool_name="failing_tool",
                arguments={},
                reasoning="Try this",
            ),
            # Ce deuxieme appel ne devrait pas se produire
            ToolDecision(
                tool_name="web_search",
                arguments={"query": "test"},
                reasoning="After error",
            ),
        ])
        executor = ToolExecutor(
            registry=reg,
            structured_engine=engine,
        )

        with patch("opti_oignon.tool_executor.OLLAMA_AVAILABLE", True), \
             patch("opti_oignon.tool_executor.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "Response after error."
            mock_ollama.chat.return_value = mock_response

            result = executor.execute_with_tools("Do something", model="test")

        # Seulement 1 appel (le failing_tool), le 2eme n'a pas lieu
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].success is False

    def test_structured_engine_failure(self):
        """Echec du StructuredOutputEngine -> reponse directe."""
        reg = _make_registry_with_tools()
        engine = MockStructuredEngine(decisions=[None])  # Echec
        executor = ToolExecutor(
            registry=reg,
            structured_engine=engine,
        )

        with patch("opti_oignon.tool_executor.OLLAMA_AVAILABLE", True), \
             patch("opti_oignon.tool_executor.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "Fallback response."
            mock_ollama.chat.return_value = mock_response

            result = executor.execute_with_tools("test", model="test")

        assert len(result.tool_calls) == 0
        assert result.response == "Fallback response."

    def test_no_registry(self):
        """Pas de registre -> message d'erreur."""
        with patch("opti_oignon.tool_executor._default_registry", None):
            executor = ToolExecutor(
                registry=None,
                structured_engine=MockStructuredEngine(),
            )
            result = executor.execute_with_tools("test")
        assert "not available" in result.response.lower()

    def test_result_timing(self):
        """Le temps total est mesure."""
        reg = _make_registry_with_tools()
        engine = MockStructuredEngine(decisions=[
            ToolDecision(tool_name="none"),
        ])
        executor = ToolExecutor(
            registry=reg,
            structured_engine=engine,
        )

        with patch("opti_oignon.tool_executor.OLLAMA_AVAILABLE", True), \
             patch("opti_oignon.tool_executor.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "ok"
            mock_ollama.chat.return_value = mock_response

            result = executor.execute_with_tools("hi", model="test")

        assert result.total_time >= 0
        assert result.model == "test"


# =============================================================================
# SINGLETON
# =============================================================================

class TestSingleton:
    """Tests du singleton tool_executor."""

    def test_singleton_exists(self):
        """Le singleton est instancie."""
        assert tool_executor is not None
        assert isinstance(tool_executor, ToolExecutor)

    def test_singleton_has_registry(self):
        """Le singleton a un registre."""
        assert tool_executor.registry is not None

    def test_singleton_has_defaults(self):
        """Le singleton a les valeurs par defaut."""
        assert tool_executor.max_tool_calls == 5
        assert tool_executor.default_model == "qwen3:32b"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests des cas limites."""

    def test_empty_message(self):
        """Message vide."""
        reg = _make_registry_with_tools()
        engine = MockStructuredEngine(decisions=[
            ToolDecision(tool_name="none"),
        ])
        executor = ToolExecutor(registry=reg, structured_engine=engine)

        with patch("opti_oignon.tool_executor.OLLAMA_AVAILABLE", True), \
             patch("opti_oignon.tool_executor.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "?"
            mock_ollama.chat.return_value = mock_response

            result = executor.execute_with_tools("", model="test")

        assert result.response is not None

    def test_conversation_context_passed(self):
        """Le contexte de conversation est transmis."""
        reg = _make_registry_with_tools()
        engine = MockStructuredEngine(decisions=[
            ToolDecision(tool_name="none"),
        ])
        executor = ToolExecutor(registry=reg, structured_engine=engine)

        context = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        with patch("opti_oignon.tool_executor.OLLAMA_AVAILABLE", True), \
             patch("opti_oignon.tool_executor.ollama") as mock_ollama:
            mock_response = MagicMock()
            mock_response.message.content = "With context."
            mock_ollama.chat.return_value = mock_response

            result = executor.execute_with_tools(
                "Follow-up", model="test",
                conversation_messages=context,
            )

        # Verifier que ollama.chat a recu les messages de contexte
        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        # Les messages de contexte doivent etre presents
        contents = [m["content"] for m in messages]
        assert any("Previous question" in c for c in contents)

    def test_no_ollama_fallback(self):
        """Sans Ollama, fallback vers les resultats bruts."""
        reg = _make_registry_with_tools()
        engine = MockStructuredEngine(decisions=[
            ToolDecision(
                tool_name="web_search",
                arguments={"query": "test"},
            ),
            ToolDecision(tool_name="none"),
        ])
        executor = ToolExecutor(registry=reg, structured_engine=engine)

        with patch("opti_oignon.tool_executor.OLLAMA_AVAILABLE", False):
            result = executor.execute_with_tools("search test", model="test")

        assert len(result.tool_calls) == 1
        # La reponse contient les resultats bruts
        assert "Search results for: test" in result.response
