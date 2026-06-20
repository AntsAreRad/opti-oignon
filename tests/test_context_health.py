#!/usr/bin/env python3
"""Tests pour la verification du pipeline de contexte (S47 -- Context Pipeline Verification)."""

from dataclasses import dataclass
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_token_budget():
    """Budget tokens simule."""
    budget = MagicMock()
    budget.context_window = 32768
    budget.system_budget = 3276
    budget.history_budget = 19660
    budget.generation_budget = 9831
    budget.total_allocated = 32767
    budget.system_ratio = 0.10
    budget.history_ratio = 0.60
    budget.generation_ratio = 0.30
    budget.available_for_history.return_value = 22937
    return budget


@pytest.fixture
def mock_conversation_messages():
    """Messages de conversation simules."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well! How can I help you today?"},
        {"role": "user", "content": "Tell me about Python decorators."},
        {"role": "assistant", "content": "Python decorators are a powerful feature..." * 10},
    ]


@pytest.fixture
def mock_window_stats():
    """Stats de fenetre glissante simulees."""
    return {
        "strategy": "score_based",
        "kept": 8,
        "dropped": 2,
        "total_tokens": 5000,
        "available_for_input": 22000,
        "context_window": 32768,
        "history_count": 10,
    }


# =============================================================================
# TESTS: Token estimation
# =============================================================================

class TestTokenEstimation:
    """Tests pour l'estimation de tokens."""

    def test_estimate_tokens_empty_string(self):
        """Chaine vide retourne 0."""
        from opti_oignon.api.routes_context import _estimate_tokens
        assert _estimate_tokens("") == 0

    def test_estimate_tokens_none(self):
        """None retourne 0."""
        from opti_oignon.api.routes_context import _estimate_tokens
        assert _estimate_tokens(None) == 0

    def test_estimate_tokens_short_text(self):
        """Texte court estime correctement."""
        from opti_oignon.api.routes_context import _estimate_tokens
        # "Hello" = 5 chars -> ~1 token
        result = _estimate_tokens("Hello")
        assert result >= 1

    def test_estimate_tokens_long_text(self):
        """Texte long estime proportionnellement."""
        from opti_oignon.api.routes_context import _estimate_tokens
        text = "a" * 400  # 400 chars -> ~100 tokens
        result = _estimate_tokens(text)
        assert result == 100

    def test_estimate_tokens_proportional(self):
        """Double texte = double tokens (approximatif)."""
        from opti_oignon.api.routes_context import _estimate_tokens
        short = _estimate_tokens("a" * 100)
        long = _estimate_tokens("a" * 200)
        assert long == 2 * short


# =============================================================================
# TESTS: Budget allocation
# =============================================================================

class TestBudgetAllocation:
    """Tests pour l'allocation du budget."""

    @patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.token_budget_manager")
    def test_budget_allocation_basic(self, mock_tbm, mock_token_budget):
        """Allocation basique avec modele connu."""
        from opti_oignon.api.routes_context import _get_budget_allocation

        mock_tbm.get_budget.return_value = mock_token_budget

        result = _get_budget_allocation("qwen3:32b")

        assert result["system_prompt"] == 3276
        assert result["history"] == 19660
        assert result["reserved_for_response"] == 9831
        assert result["total_allocated"] == 32767
        assert result["context_window"] == 32768

    @patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", False)
    def test_budget_allocation_unavailable(self):
        """Allocation retourne des zeros si module indisponible."""
        from opti_oignon.api.routes_context import _get_budget_allocation

        result = _get_budget_allocation("any_model")

        assert result["system_prompt"] == 0
        assert result["history"] == 0
        assert result["total_allocated"] == 0

    @patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.token_budget_manager")
    def test_budget_allocation_ratios(self, mock_tbm, mock_token_budget):
        """Les ratios sont bien arrondis."""
        from opti_oignon.api.routes_context import _get_budget_allocation

        mock_tbm.get_budget.return_value = mock_token_budget

        result = _get_budget_allocation("qwen3:32b")

        assert result["system_ratio"] == 0.1
        assert result["history_ratio"] == 0.6
        assert result["generation_ratio"] == 0.3

    @patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.token_budget_manager")
    def test_budget_allocation_exception_handling(self, mock_tbm):
        """Exception lors du calcul retourne des zeros."""
        from opti_oignon.api.routes_context import _get_budget_allocation

        mock_tbm.get_budget.side_effect = Exception("Budget error")

        result = _get_budget_allocation("unknown_model")

        assert result["system_prompt"] == 0
        assert result["context_window"] == 0

    @patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.token_budget_manager", None)
    def test_budget_allocation_none_manager(self):
        """Manager None retourne des zeros."""
        from opti_oignon.api.routes_context import _get_budget_allocation

        result = _get_budget_allocation("any_model")
        assert result["system_prompt"] == 0


# =============================================================================
# TESTS: Conversation context
# =============================================================================

class TestConversationContext:
    """Tests pour la recuperation du contexte de conversation."""

    @patch("opti_oignon.api.routes_context.CONVERSATION_AVAILABLE", False)
    def test_no_conversation_manager(self):
        """Pas de gestionnaire de conversation -> resultats vides."""
        from opti_oignon.api.routes_context import _get_conversation_context

        result = _get_conversation_context()

        assert result["conversation_id"] is None
        assert result["messages_count"] == 0
        assert result["estimated_tokens"] == 0

    @patch("opti_oignon.api.routes_context.CONVERSATION_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.conversation_manager")
    @patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", False)
    @patch("opti_oignon.api.routes_context.EXECUTOR_AVAILABLE", False)
    def test_with_conversation_id(self, mock_cm):
        """Conversation specifique retourne les bonnes donnees."""
        from opti_oignon.api.routes_context import _get_conversation_context

        mock_cm.get_messages.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        mock_cm.get_conversation.return_value = {"model": "qwen3:32b"}

        result = _get_conversation_context("test-conv-123")

        assert result["conversation_id"] == "test-conv-123"
        assert result["messages_count"] == 2
        assert result["estimated_tokens"] > 0
        assert result["model"] == "qwen3:32b"

    @patch("opti_oignon.api.routes_context.CONVERSATION_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.conversation_manager")
    @patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", False)
    @patch("opti_oignon.api.routes_context.EXECUTOR_AVAILABLE", False)
    def test_auto_detect_latest_conversation(self, mock_cm):
        """Sans ID, prend la conversation la plus recente."""
        from opti_oignon.api.routes_context import _get_conversation_context

        mock_cm.list_conversations.return_value = [
            {"id": "latest-conv"},
            {"id": "older-conv"},
        ]
        mock_cm.get_messages.return_value = []
        mock_cm.get_conversation.return_value = {"model": "qwen3:32b"}

        result = _get_conversation_context()

        assert result["conversation_id"] == "latest-conv"

    @patch("opti_oignon.api.routes_context.CONVERSATION_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.conversation_manager")
    @patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", False)
    @patch("opti_oignon.api.routes_context.EXECUTOR_AVAILABLE", False)
    def test_empty_conversation_list(self, mock_cm):
        """Liste vide retourne resultats vides."""
        from opti_oignon.api.routes_context import _get_conversation_context

        mock_cm.list_conversations.return_value = []

        result = _get_conversation_context()

        assert result["conversation_id"] is None

    @patch("opti_oignon.api.routes_context.CONVERSATION_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.conversation_manager")
    @patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.token_budget_manager")
    @patch("opti_oignon.api.routes_context.EXECUTOR_AVAILABLE", False)
    def test_usage_percent_calculation(self, mock_tbm, mock_cm, mock_token_budget):
        """Pourcentage d'utilisation calcule correctement."""
        from opti_oignon.api.routes_context import _get_conversation_context

        mock_cm.get_messages.return_value = [
            {"role": "user", "content": "a" * 4000},  # ~1000 tokens
        ]
        mock_cm.get_conversation.return_value = {"model": "qwen3:32b"}

        mock_tbm.get_budget.return_value = mock_token_budget

        result = _get_conversation_context("test-conv")

        assert result["usage_percent"] > 0
        assert isinstance(result["usage_percent"], float)

    @patch("opti_oignon.api.routes_context.CONVERSATION_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.conversation_manager")
    @patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.token_budget_manager")
    @patch("opti_oignon.api.routes_context.EXECUTOR_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.executor")
    def test_trimming_detection_from_executor(self, mock_exec, mock_tbm, mock_cm, mock_token_budget, mock_window_stats):
        """Trimming detecte via les stats executor."""
        from opti_oignon.api.routes_context import _get_conversation_context

        mock_cm.get_messages.return_value = [{"role": "user", "content": "Hello"}]
        mock_cm.get_conversation.return_value = {"model": "qwen3:32b"}
        mock_tbm.get_budget.return_value = mock_token_budget

        # Executor avec stats de trimming
        type(mock_exec).last_window_stats = PropertyMock(return_value=mock_window_stats)

        result = _get_conversation_context("test-conv")

        assert result["trimming_active"] is True
        assert result["last_window_stats"]["dropped"] == 2


# =============================================================================
# TESTS: API Endpoint /api/context/health
# =============================================================================

class TestContextHealthEndpoint:
    """Tests pour l'endpoint GET /api/context/health."""

    def test_endpoint_import(self):
        """Le module routes_context s'importe correctement."""
        from opti_oignon.api.routes_context import router
        assert router is not None

    def test_endpoint_registered(self):
        """L'endpoint est bien enregistre dans le routeur."""
        from opti_oignon.api.routes_context import router

        paths = [r.path for r in router.routes]
        assert "/api/context/health" in paths

    def test_endpoint_response_structure(self):
        """La reponse a la bonne structure."""
        from opti_oignon.api.routes_context import context_health

        with patch("opti_oignon.api.routes_context._get_conversation_context") as mock_conv, \
             patch("opti_oignon.api.routes_context._get_budget_allocation") as mock_budget:

            mock_conv.return_value = {
                "conversation_id": None,
                "model": None,
                "model_context_window": 0,
                "messages_count": 0,
                "estimated_tokens": 0,
                "usage_percent": 0.0,
                "trimming_active": False,
                "last_window_stats": {},
            }
            mock_budget.return_value = {
                "system_prompt": 0, "history": 0,
                "reserved_for_response": 0, "total_allocated": 0,
                "context_window": 0, "system_ratio": 0.0,
                "history_ratio": 0.0, "generation_ratio": 0.0,
            }

            result = context_health()

        assert "status" in result
        assert "context_window_available" in result
        assert "current_conversation" in result
        assert "budget_allocation" in result

    def test_status_healthy(self):
        """Statut 'healthy' quand tout va bien."""
        from opti_oignon.api.routes_context import context_health

        with patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", True), \
             patch("opti_oignon.api.routes_context._get_conversation_context") as mock_conv, \
             patch("opti_oignon.api.routes_context._get_budget_allocation") as mock_budget:

            mock_conv.return_value = {
                "conversation_id": "abc", "model": "qwen3:32b",
                "model_context_window": 32768, "messages_count": 5,
                "estimated_tokens": 2000, "usage_percent": 6.1,
                "trimming_active": False, "last_window_stats": {},
            }
            mock_budget.return_value = {
                "system_prompt": 3276, "history": 19660,
                "reserved_for_response": 9831, "total_allocated": 32767,
                "context_window": 32768, "system_ratio": 0.1,
                "history_ratio": 0.6, "generation_ratio": 0.3,
            }

            result = context_health()

        assert result["status"] == "healthy"

    def test_status_trimming(self):
        """Statut 'trimming' quand le trimming est actif."""
        from opti_oignon.api.routes_context import context_health

        with patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", True), \
             patch("opti_oignon.api.routes_context._get_conversation_context") as mock_conv, \
             patch("opti_oignon.api.routes_context._get_budget_allocation") as mock_budget:

            mock_conv.return_value = {
                "conversation_id": "abc", "model": "qwen3:32b",
                "model_context_window": 32768, "messages_count": 50,
                "estimated_tokens": 28000, "usage_percent": 85.4,
                "trimming_active": True, "last_window_stats": {},
            }
            mock_budget.return_value = {
                "system_prompt": 3276, "history": 19660,
                "reserved_for_response": 9831, "total_allocated": 32767,
                "context_window": 32768, "system_ratio": 0.1,
                "history_ratio": 0.6, "generation_ratio": 0.3,
            }

            result = context_health()

        assert result["status"] == "trimming"

    def test_status_warning_high_usage(self):
        """Statut 'warning' quand l'utilisation depasse 90%."""
        from opti_oignon.api.routes_context import context_health

        with patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", True), \
             patch("opti_oignon.api.routes_context._get_conversation_context") as mock_conv, \
             patch("opti_oignon.api.routes_context._get_budget_allocation") as mock_budget:

            mock_conv.return_value = {
                "conversation_id": "abc", "model": "qwen3:32b",
                "model_context_window": 32768, "messages_count": 100,
                "estimated_tokens": 30000, "usage_percent": 91.5,
                "trimming_active": False, "last_window_stats": {},
            }
            mock_budget.return_value = {
                "system_prompt": 3276, "history": 19660,
                "reserved_for_response": 9831, "total_allocated": 32767,
                "context_window": 32768, "system_ratio": 0.1,
                "history_ratio": 0.6, "generation_ratio": 0.3,
            }

            result = context_health()

        assert result["status"] == "warning"

    def test_status_degraded_no_module(self):
        """Statut 'degraded' quand le module est indisponible."""
        from opti_oignon.api.routes_context import context_health

        with patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", False), \
             patch("opti_oignon.api.routes_context._get_conversation_context") as mock_conv, \
             patch("opti_oignon.api.routes_context._get_budget_allocation") as mock_budget:

            mock_conv.return_value = {
                "conversation_id": None, "model": None,
                "model_context_window": 0, "messages_count": 0,
                "estimated_tokens": 0, "usage_percent": 0.0,
                "trimming_active": False, "last_window_stats": {},
            }
            mock_budget.return_value = {
                "system_prompt": 0, "history": 0,
                "reserved_for_response": 0, "total_allocated": 0,
                "context_window": 0, "system_ratio": 0.0,
                "history_ratio": 0.0, "generation_ratio": 0.0,
            }

            result = context_health()

        assert result["status"] == "degraded"


# =============================================================================
# TESTS: API Endpoint /api/context/budget/{model_name}
# =============================================================================

class TestModelBudgetEndpoint:
    """Tests pour l'endpoint GET /api/context/budget/{model_name}."""

    def test_budget_endpoint_registered(self):
        """L'endpoint budget est bien enregistre."""
        from opti_oignon.api.routes_context import router

        paths = [r.path for r in router.routes]
        assert "/api/context/budget/{model_name}" in paths

    @patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", False)
    def test_budget_unavailable(self):
        """Budget indisponible retourne available=False."""
        from opti_oignon.api.routes_context import get_model_budget

        result = get_model_budget("any_model")

        assert result["available"] is False
        assert result["budget"] is None

    @patch("opti_oignon.api.routes_context.CONTEXT_WINDOW_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.token_budget_manager")
    def test_budget_for_known_model(self, mock_tbm, mock_token_budget):
        """Budget pour un modele connu."""
        from opti_oignon.api.routes_context import get_model_budget

        mock_tbm.get_budget.return_value = mock_token_budget

        result = get_model_budget("qwen3:32b")

        assert result["available"] is True
        assert result["model"] == "qwen3:32b"
        assert result["budget"]["context_window"] == 32768


# =============================================================================
# TESTS: API Endpoint /api/context/stats
# =============================================================================

class TestContextStatsEndpoint:
    """Tests pour l'endpoint GET /api/context/stats."""

    def test_stats_endpoint_registered(self):
        """L'endpoint stats est bien enregistre."""
        from opti_oignon.api.routes_context import router

        paths = [r.path for r in router.routes]
        assert "/api/context/stats" in paths

    @patch("opti_oignon.api.routes_context.EXECUTOR_AVAILABLE", False)
    def test_stats_no_executor(self):
        """Stats indisponibles sans executor."""
        from opti_oignon.api.routes_context import context_stats

        result = context_stats()

        assert result["available"] is False
        assert result["window_stats"] == {}

    @patch("opti_oignon.api.routes_context.EXECUTOR_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.executor")
    def test_stats_with_executor(self, mock_exec, mock_window_stats):
        """Stats disponibles avec executor."""
        from opti_oignon.api.routes_context import context_stats

        type(mock_exec).last_window_stats = PropertyMock(return_value=mock_window_stats)

        result = context_stats()

        assert result["available"] is True
        assert result["window_stats"]["strategy"] == "score_based"
        assert result["window_stats"]["dropped"] == 2

    @patch("opti_oignon.api.routes_context.EXECUTOR_AVAILABLE", True)
    @patch("opti_oignon.api.routes_context.executor")
    def test_stats_empty_window_stats(self, mock_exec):
        """Stats vides quand aucun appel multi-turn."""
        from opti_oignon.api.routes_context import context_stats

        type(mock_exec).last_window_stats = PropertyMock(return_value={})

        result = context_stats()

        assert result["available"] is False
        assert result["window_stats"] == {}


# =============================================================================
# TESTS: Health dashboard integration
# =============================================================================

class TestHealthDashboardIntegration:
    """Tests pour l'integration dans le dashboard de sante."""

    def test_dashboard_schema_has_context_health(self):
        """Le schema HealthDashboard inclut context_health."""
        from opti_oignon.api.schemas import HealthDashboard

        dashboard = HealthDashboard()
        assert hasattr(dashboard, "context_health")
        assert dashboard.context_health is None

    def test_dashboard_schema_version_updated(self):
        """La version du schema est a jour."""
        from opti_oignon.api.schemas import HealthDashboard

        dashboard = HealthDashboard()
        assert dashboard.version == "1.6.6"

    def test_dashboard_context_health_settable(self):
        """context_health peut etre mis a jour."""
        from opti_oignon.api.schemas import HealthDashboard

        dashboard = HealthDashboard()
        dashboard.context_health = {"available": True, "trimming_active": False}

        assert dashboard.context_health["available"] is True


# =============================================================================
# TESTS: Router registration
# =============================================================================

class TestRouterRegistration:
    """Tests pour l'enregistrement du routeur dans l'application."""

    def test_context_router_in_app(self):
        """Le routeur context est enregistre dans l'app FastAPI."""
        from opti_oignon.api.app import app

        routes = [r.path for r in app.routes]
        context_routes = [r for r in routes if "/context" in r]
        assert len(context_routes) > 0

    def test_app_version_updated(self):
        """La version de l'app est mise a jour."""
        from opti_oignon.api.app import app
        assert app.version == "1.6.6"

    def test_health_includes_context_window(self):
        """L'endpoint /api/health inclut context_window dans les modules."""
        from opti_oignon.api.app import health_check

        result = health_check()
        assert "context_window" in result["modules"]


# =============================================================================
# TESTS: Backward compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Tests de non-regression et compatibilite."""

    def test_context_window_module_importable(self):
        """Le module context_window s'importe sans erreur."""
        from opti_oignon.context_window import sliding_window_manager, token_budget_manager
        assert sliding_window_manager is not None
        assert token_budget_manager is not None

    def test_deps_still_exports_all_flags(self):
        """deps.py exporte toujours tous les flags attendus."""
        from opti_oignon.api.deps import (
            ARTIFACT_AVAILABLE,
            BENCHMARK_AVAILABLE,
            CODE_EXECUTOR_AVAILABLE,
            CONFIG_AVAILABLE,
            CONTEXT_MANAGER_AVAILABLE,
            CONTEXT_WINDOW_AVAILABLE,
            CONVERSATION_AVAILABLE,
            EXECUTOR_AVAILABLE,
            MEMORY_AVAILABLE,
            MODEL_WARMUP_AVAILABLE,
            PIPELINE_AVAILABLE,
            PRESET_AVAILABLE,
            PROFILE_AVAILABLE,
            RESPONSE_CACHE_AVAILABLE,
            ROUTER_AVAILABLE,
            SEMANTIC_CACHE_AVAILABLE,
        )
        # Tous les flags doivent etre des booleens
        for flag in [
            CONVERSATION_AVAILABLE, RESPONSE_CACHE_AVAILABLE,
            CONTEXT_WINDOW_AVAILABLE, EXECUTOR_AVAILABLE,
            PROFILE_AVAILABLE, CONTEXT_MANAGER_AVAILABLE,
        ]:
            assert isinstance(flag, bool)

    def test_existing_health_endpoint_still_works(self):
        """L'endpoint /api/health retourne toujours les modules connus."""
        from opti_oignon.api.app import health_check

        result = health_check()
        assert result["status"] == "ok"
        assert "modules" in result
        # Les anciens modules doivent toujours etre presents
        for mod in ["conversation", "presets", "memory", "artifacts",
                     "code_executor", "response_cache", "semantic_cache",
                     "pipelines", "benchmarks", "model_warmup", "model_profiles"]:
            assert mod in result["modules"]

    def test_routes_context_prefix(self):
        """Le prefixe du routeur est /api/context."""
        from opti_oignon.api.routes_context import router
        assert router.prefix == "/api/context"

    def test_token_budget_singleton(self):
        """Le singleton token_budget_manager est accessible."""
        from opti_oignon.context_window import token_budget_manager
        budget = token_budget_manager.get_budget("unknown_model")
        assert budget.context_window > 0

    def test_sliding_window_singleton(self):
        """Le singleton sliding_window_manager est accessible."""
        from opti_oignon.context_window import sliding_window_manager
        stats = sliding_window_manager.get_window_stats(
            [{"role": "user", "content": "test"}],
            "unknown_model",
        )
        assert "message_count" in stats
        assert stats["message_count"] == 1
