"""
API route tests for Conversation Compressor — Opti-Oignon S66.

Tests all endpoints in routes_compression.py using FastAPI TestClient:
  - GET  /api/compression/config
  - PUT  /api/compression/config
  - POST /api/compression/config/reload
  - GET  /api/compression/stats/{conversation_id}
  - POST /api/compression/archive/search/{conversation_id}
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from opti_oignon.api.app import app
from opti_oignon.conversation_compressor import ArchiveSearchResult


client = TestClient(app)


# ============================================================================
# Helpers
# ============================================================================

def _mock_compressor(config_override=None):
    """Build a mock ConversationCompressor with sane defaults."""
    mock = MagicMock()
    mock.get_config.return_value = {
        "enabled": True,
        "strategy": "hybrid",
        "recent_messages_keep": 6,
        "compression_threshold_ratio": 1.0,
        "llm_summary_model": None,
        "llm_summary_max_tokens": 300,
        "llm_summary_temperature": 0.2,
        "llm_summary_timeout": 30,
        "rule_max_facts_per_message": 2,
        "rule_min_message_length": 50,
        "archive_retrieval_top_k": 3,
        "archive_retrieval_min_score": 0.05,
        "archive_retrieval_snippet_length": 300,
        "retrieval_trigger_enabled": True,
        "retrieval_trigger_min_confidence": 0.6,
        **(config_override or {}),
    }
    mock.update_config.return_value = mock.get_config.return_value
    mock.reload_config.return_value = None
    mock.retrieve_from_archive.return_value = []
    return mock


# ============================================================================
# GET /api/compression/config
# ============================================================================

class TestGetConfig:
    def test_returns_200_when_available(self):
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.get("/api/compression/config")
        assert resp.status_code == 200

    def test_returns_correct_fields(self):
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.get("/api/compression/config")
        data = resp.json()
        assert "enabled" in data
        assert "strategy" in data
        assert "recent_messages_keep" in data
        assert "compression_threshold_ratio" in data
        assert "archive_retrieval_top_k" in data
        assert "retrieval_trigger_enabled" in data

    def test_returns_503_when_unavailable(self):
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", False):
            with patch("opti_oignon.api.deps.conversation_compressor", None):
                resp = client.get("/api/compression/config")
        assert resp.status_code == 503

    def test_strategy_value_is_string(self):
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.get("/api/compression/config")
        assert isinstance(resp.json()["strategy"], str)

    def test_enabled_is_bool(self):
        mock_comp = _mock_compressor({"enabled": True})
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.get("/api/compression/config")
        assert resp.json()["enabled"] is True


# ============================================================================
# PUT /api/compression/config
# ============================================================================

class TestUpdateConfig:
    def test_update_strategy_valid(self):
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.put(
                    "/api/compression/config",
                    json={"strategy": "rule"},
                )
        assert resp.status_code == 200
        mock_comp.update_config.assert_called_once()
        call_arg = mock_comp.update_config.call_args[0][0]
        assert call_arg["strategy"] == "rule"

    def test_update_enabled_false(self):
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.put(
                    "/api/compression/config",
                    json={"enabled": False},
                )
        assert resp.status_code == 200
        call_arg = mock_comp.update_config.call_args[0][0]
        assert call_arg["enabled"] is False

    def test_update_recent_messages_keep(self):
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.put(
                    "/api/compression/config",
                    json={"recent_messages_keep": 10},
                )
        assert resp.status_code == 200

    def test_invalid_strategy_returns_422(self):
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.put(
                    "/api/compression/config",
                    json={"strategy": "magic"},
                )
        assert resp.status_code == 422

    def test_empty_body_no_update(self):
        """Empty body should call update_config with empty dict."""
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.put("/api/compression/config", json={})
        assert resp.status_code == 200
        # update_config called with empty dict (all None fields filtered)
        call_arg = mock_comp.update_config.call_args[0][0]
        assert call_arg == {}

    def test_returns_503_when_unavailable(self):
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", False):
            with patch("opti_oignon.api.deps.conversation_compressor", None):
                resp = client.put(
                    "/api/compression/config",
                    json={"strategy": "rule"},
                )
        assert resp.status_code == 503

    def test_recent_messages_keep_min_validation(self):
        """recent_messages_keep must be >= 1."""
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.put(
                    "/api/compression/config",
                    json={"recent_messages_keep": 0},
                )
        assert resp.status_code == 422

    def test_temperature_range_validation(self):
        """Temperature must be 0.0–1.0."""
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.put(
                    "/api/compression/config",
                    json={"llm_summary_temperature": 2.5},
                )
        assert resp.status_code == 422


# ============================================================================
# POST /api/compression/config/reload
# ============================================================================

class TestReloadConfig:
    def test_reload_returns_200(self):
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.post("/api/compression/config/reload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reloaded"

    def test_reload_calls_reload_config(self):
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                client.post("/api/compression/config/reload")
        mock_comp.reload_config.assert_called_once()

    def test_reload_returns_503_when_unavailable(self):
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", False):
            with patch("opti_oignon.api.deps.conversation_compressor", None):
                resp = client.post("/api/compression/config/reload")
        assert resp.status_code == 503


# ============================================================================
# GET /api/compression/stats/{conversation_id}
# ============================================================================

class TestGetStats:
    def test_returns_200_no_compression(self):
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                with patch("opti_oignon.api.deps.EXECUTOR_AVAILABLE", True):
                    mock_executor = MagicMock()
                    mock_executor.last_compression_result = None
                    with patch("opti_oignon.api.deps.executor", mock_executor):
                        resp = client.get("/api/compression/stats/conv-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["last_compression_available"] is False
        assert data["conversation_id"] == "conv-123"

    def test_returns_stats_when_compression_present(self):
        mock_comp = _mock_compressor()
        mock_result = MagicMock()
        mock_result.summary = "Earlier conversation summary:\n- Key fact"
        mock_result.original_count = 12
        mock_result.compressed_count = 8
        mock_result.strategy_used = "hybrid_rule"
        mock_result.tokens_saved = 250
        mock_result.compression_ratio = 0.667

        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                with patch("opti_oignon.api.deps.EXECUTOR_AVAILABLE", True):
                    mock_executor = MagicMock()
                    mock_executor.last_compression_result = mock_result
                    with patch("opti_oignon.api.deps.executor", mock_executor):
                        resp = client.get("/api/compression/stats/conv-abc")

        assert resp.status_code == 200
        data = resp.json()
        assert data["last_compression_available"] is True
        assert data["original_count"] == 12
        assert data["compressed_count"] == 8
        assert data["strategy_used"] == "hybrid_rule"
        assert data["tokens_saved"] == 250

    def test_returns_503_when_compressor_unavailable(self):
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", False):
            with patch("opti_oignon.api.deps.conversation_compressor", None):
                resp = client.get("/api/compression/stats/conv-123")
        assert resp.status_code == 503


# ============================================================================
# POST /api/compression/archive/search/{conversation_id}
# ============================================================================

class TestArchiveSearch:
    def test_returns_200_empty_results(self):
        mock_comp = _mock_compressor()
        mock_comp.retrieve_from_archive.return_value = []
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.post(
                    "/api/compression/archive/search/conv-123",
                    json={"query": "something"},
                )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert data["total_found"] == 0

    def test_returns_results_when_found(self):
        mock_comp = _mock_compressor()
        mock_comp.retrieve_from_archive.return_value = [
            ArchiveSearchResult(0, "user", "Snippet about NMDS", 0.85),
            ArchiveSearchResult(1, "assistant", "NMDS ordination explained", 0.72),
        ]
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.post(
                    "/api/compression/archive/search/conv-123",
                    json={"query": "NMDS ordination"},
                )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_found"] == 2
        assert data["query"] == "NMDS ordination"
        assert data["conversation_id"] == "conv-123"
        assert len(data["results"]) == 2

    def test_result_fields_present(self):
        mock_comp = _mock_compressor()
        mock_comp.retrieve_from_archive.return_value = [
            ArchiveSearchResult(5, "assistant", "Relevant snippet", 0.9, "2024-01-01T00:00:00"),
        ]
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.post(
                    "/api/compression/archive/search/conv-123",
                    json={"query": "test"},
                )
        item = resp.json()["results"][0]
        assert "message_id" in item
        assert "role" in item
        assert "snippet" in item
        assert "score" in item
        assert "timestamp" in item

    def test_min_score_filter_applied(self):
        """Results below min_score should be filtered out."""
        mock_comp = _mock_compressor()
        mock_comp.retrieve_from_archive.return_value = [
            ArchiveSearchResult(0, "user", "High score match", 0.9),
            ArchiveSearchResult(1, "user", "Low score match", 0.03),
        ]
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.post(
                    "/api/compression/archive/search/conv-123",
                    json={"query": "test", "min_score": 0.5},
                )
        data = resp.json()
        assert data["total_found"] == 1
        assert data["results"][0]["score"] >= 0.5

    def test_top_k_passed_to_retriever(self):
        mock_comp = _mock_compressor()
        mock_comp.retrieve_from_archive.return_value = []
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                client.post(
                    "/api/compression/archive/search/conv-123",
                    json={"query": "test", "top_k": 7},
                )
        mock_comp.retrieve_from_archive.assert_called_once_with(
            conversation_id="conv-123",
            query="test",
            top_k=7,
        )

    def test_empty_query_returns_422(self):
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.post(
                    "/api/compression/archive/search/conv-123",
                    json={"query": ""},
                )
        assert resp.status_code == 422

    def test_returns_503_when_unavailable(self):
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", False):
            with patch("opti_oignon.api.deps.conversation_compressor", None):
                resp = client.post(
                    "/api/compression/archive/search/conv-123",
                    json={"query": "test"},
                )
        assert resp.status_code == 503

    def test_retriever_exception_returns_500(self):
        mock_comp = _mock_compressor()
        mock_comp.retrieve_from_archive.side_effect = RuntimeError("DB failure")
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.post(
                    "/api/compression/archive/search/conv-123",
                    json={"query": "test"},
                )
        assert resp.status_code == 500

    def test_top_k_max_validation(self):
        """top_k must be <= 20."""
        mock_comp = _mock_compressor()
        with patch("opti_oignon.api.deps.CONVERSATION_COMPRESSOR_AVAILABLE", True):
            with patch("opti_oignon.api.deps.conversation_compressor", mock_comp):
                resp = client.post(
                    "/api/compression/archive/search/conv-123",
                    json={"query": "test", "top_k": 99},
                )
        assert resp.status_code == 422
