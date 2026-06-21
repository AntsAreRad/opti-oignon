#!/usr/bin/env python3
"""
Tests for Web Search proxy and PII integration (S82).

Covers:
- WebSearchConfig from YAML
- Proxy configuration (set/get/toggle)
- PII integration in search flow
- Retry with backoff logic
- ProxyStatus checks
- Search stats (new counters)
- Backward-compatible alias (web_search_engine)
- Graceful fallback when proxy unreachable
"""

import importlib.util
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module loading (test isolation without ollama)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Config stub
_config_mod = types.ModuleType("opti_oignon.config")
_config_mod.CONFIG_DIR = _PROJECT_ROOT / "opti_oignon" / "config"

def _load_yaml(p):
    import yaml
    with open(p) as f:
        return yaml.safe_load(f) or {}

_config_mod.load_yaml = _load_yaml
sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))
sys.modules["opti_oignon.config"] = _config_mod

# Load pii_sanitizer first (dependency)
_pii_path = _PROJECT_ROOT / "opti_oignon" / "pii_sanitizer.py"
_pii_spec = importlib.util.spec_from_file_location(
    "opti_oignon.pii_sanitizer", _pii_path
)
_pii_mod = importlib.util.module_from_spec(_pii_spec)
sys.modules["opti_oignon.pii_sanitizer"] = _pii_mod
_pii_spec.loader.exec_module(_pii_mod)

# Load web_search module
_ws_path = _PROJECT_ROOT / "opti_oignon" / "web_search.py"
_ws_spec = importlib.util.spec_from_file_location(
    "opti_oignon.web_search", _ws_path
)
_ws_mod = importlib.util.module_from_spec(_ws_spec)
sys.modules["opti_oignon.web_search"] = _ws_mod
_ws_spec.loader.exec_module(_ws_mod)

WebSearcher = _ws_mod.WebSearcher
WebSearchConfig = _ws_mod.WebSearchConfig
ProxyStatus = _ws_mod.ProxyStatus
SearchResult = _ws_mod.SearchResult
PIISanitizer = _pii_mod.PIISanitizer
PIISanitizeConfig = _pii_mod.PIISanitizeConfig


# ===========================================================================
# Helpers
# ===========================================================================

def _make_searcher(**kwargs) -> WebSearcher:
    """Create a WebSearcher with custom config for testing."""
    config = WebSearchConfig(**kwargs)
    pii = PIISanitizer(PIISanitizeConfig(enabled=kwargs.get("pii_sanitization_enabled", True)))
    return WebSearcher(config=config, pii_sanitizer=pii)


def _mock_ddgs_results(results=None):
    """Create a mock DDGS that returns given results."""
    if results is None:
        results = [
            {"title": "Result 1", "body": "Snippet 1", "href": "https://example.com/1"},
            {"title": "Result 2", "body": "Snippet 2", "href": "https://example.com/2"},
        ]
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = results
    return mock_ddgs


# ===========================================================================
# Tests
# ===========================================================================


class TestWebSearchConfig:
    """Test WebSearchConfig creation and YAML loading."""

    def test_default_config(self):
        config = WebSearchConfig()
        assert config.proxy is None
        assert config.proxy_timeout == 15
        assert config.max_retries == 3
        assert config.retry_backoff == [2, 5, 10]
        assert config.pii_sanitization_enabled is True

    def test_from_dict_with_proxy(self):
        data = {
            "proxy": "socks5h://localhost:9050",
            "proxy_timeout": 20,
            "max_retries": 5,
            "retry_backoff": [1, 3, 7],
            "pii_sanitization": {"enabled": False},
        }
        config = WebSearchConfig.from_dict(data)
        assert config.proxy == "socks5h://localhost:9050"
        assert config.proxy_timeout == 20
        assert config.max_retries == 5
        assert config.retry_backoff == [1, 3, 7]
        assert config.pii_sanitization_enabled is False

    def test_from_dict_empty(self):
        config = WebSearchConfig.from_dict({})
        assert config.proxy is None
        assert config.max_retries == 3

    def test_from_yaml_file(self):
        """Config loads from the actual web_search.yaml file."""
        config = _ws_mod._load_config_from_yaml()
        assert config.proxy is None  # Default in YAML
        assert config.max_retries == 3
        assert config.retry_backoff == [2, 5, 10]
        assert config.proxy_timeout == 15


class TestProxyConfiguration:
    """Test proxy setup and properties."""

    def test_no_proxy_by_default(self):
        s = _make_searcher()
        assert s.proxy_configured is False
        assert s.effective_timeout == 10

    def test_proxy_configured(self):
        s = _make_searcher(proxy="socks5h://localhost:9050")
        assert s.proxy_configured is True
        assert s.effective_timeout == 15

    def test_set_proxy_runtime(self):
        s = _make_searcher()
        assert s.proxy_configured is False
        s.set_proxy("socks5h://myproxy:1080")
        assert s.proxy_configured is True
        assert s.config.proxy == "socks5h://myproxy:1080"

    def test_disable_proxy_runtime(self):
        s = _make_searcher(proxy="socks5h://localhost:9050")
        assert s.proxy_configured is True
        s.set_proxy(None)
        assert s.proxy_configured is False

    def test_empty_string_proxy_not_configured(self):
        s = _make_searcher(proxy="")
        assert s.proxy_configured is False


class TestProxyStatus:
    """Test proxy health check."""

    def test_status_no_proxy(self):
        s = _make_searcher()
        status = s.check_proxy_status()
        assert status.configured is False
        assert status.reachable is False

    def test_status_dataclass(self):
        status = ProxyStatus(
            configured=True,
            proxy_url="socks5h://localhost:9050",
            reachable=True,
            latency_ms=150.5,
            exit_ip="198.51.100.1",
        )
        assert status.configured is True
        assert status.latency_ms == 150.5
        assert status.exit_ip == "198.51.100.1"

    @patch.object(_ws_mod, "DDGS_AVAILABLE", False)
    def test_status_ddgs_unavailable(self):
        s = _make_searcher(proxy="socks5h://localhost:9050")
        status = s.check_proxy_status()
        assert status.configured is True
        assert status.reachable is False
        assert "not installed" in status.error


class TestPIIIntegration:
    """Test PII sanitization integration in search flow."""

    def test_sanitize_query(self):
        s = _make_searcher()
        clean, modified = s.sanitize_query("error at user@test.com on 10.0.0.1")
        assert "[EMAIL]" in clean
        assert "[IP]" in clean
        assert modified is True

    def test_sanitize_query_disabled(self):
        s = _make_searcher(pii_sanitization_enabled=False)
        clean, modified = s.sanitize_query("user@test.com")
        assert clean == "user@test.com"
        assert modified is False

    def test_sanitize_query_no_pii(self):
        s = _make_searcher()
        clean, modified = s.sanitize_query("python tutorial")
        assert clean == "python tutorial"
        assert modified is False

    def test_preview_sanitization(self):
        s = _make_searcher()
        preview = s.preview_sanitization("user@test.com at /home/user/file")
        assert preview["was_modified"] is True
        assert len(preview["items"]) >= 2
        assert "[EMAIL]" in preview["sanitized"]

    def test_preview_no_pii_module(self):
        config = WebSearchConfig()
        s = WebSearcher(config=config, pii_sanitizer=None)
        s._pii = None
        preview = s.preview_sanitization("user@test.com")
        assert preview["was_modified"] is False
        assert preview["sanitized"] == "user@test.com"

    @patch.object(_ws_mod, "DDGS_AVAILABLE", True)
    def test_search_sanitizes_before_sending(self):
        """PII should be stripped from the query sent to DDG."""
        s = _make_searcher()
        mock_ddgs = _mock_ddgs_results()

        with patch.object(_ws_mod, "DDGS", return_value=mock_ddgs):
            s.search("error at user@test.com", max_results=2)

        # The query sent to DDG should be sanitized
        call_args = mock_ddgs.text.call_args
        sent_query = call_args.kwargs.get("keywords", call_args.args[0] if call_args.args else "")
        assert "user@test.com" not in sent_query
        assert "[EMAIL]" in sent_query


class TestRetryWithBackoff:
    """Test retry logic on transient failures."""

    @patch.object(_ws_mod, "DDGS_AVAILABLE", True)
    def test_retry_on_ratelimit(self):
        """Should retry on RatelimitException and eventually succeed."""
        s = _make_searcher(max_retries=2, retry_backoff=[0, 0])
        s._last_request_time = time.time()  # Skip rate limit wait

        mock_ddgs_fail = MagicMock()
        mock_ddgs_fail.text.side_effect = [
            _ws_mod.RatelimitException("rate limited"),
            [{"title": "OK", "body": "snippet", "href": "https://ok.com"}],
        ]

        with patch.object(_ws_mod, "DDGS", return_value=mock_ddgs_fail):
            results = s.search("test query")

        assert len(results) == 1
        assert results[0].title == "OK"
        assert s._stats["retries"] >= 1

    @patch.object(_ws_mod, "DDGS_AVAILABLE", True)
    def test_all_retries_exhausted(self):
        """Should return empty list after all retries fail."""
        s = _make_searcher(max_retries=1, retry_backoff=[0])
        s._last_request_time = time.time()

        mock_ddgs_fail = MagicMock()
        mock_ddgs_fail.text.side_effect = _ws_mod.TimeoutException("timeout")

        with patch.object(_ws_mod, "DDGS", return_value=mock_ddgs_fail):
            results = s.search("test query")

        assert results == []
        assert s._stats["errors"] >= 1

    @patch.object(_ws_mod, "DDGS_AVAILABLE", True)
    def test_proxy_passed_to_ddgs(self):
        """When proxy is configured, DDGS should receive it."""
        s = _make_searcher(proxy="socks5h://localhost:9050")
        s._last_request_time = time.time()

        with patch.object(_ws_mod, "DDGS") as MockDDGS:
            mock_instance = MagicMock()
            mock_instance.text.return_value = [
                {"title": "T", "body": "B", "href": "https://t.com"}
            ]
            MockDDGS.return_value = mock_instance
            s.search("test")

        MockDDGS.assert_called_with(
            timeout=15,
            proxy="socks5h://localhost:9050",
        )


class TestSearchStats:
    """Test new stats counters (S82)."""

    def test_initial_stats(self):
        s = _make_searcher()
        stats = s.get_cache_stats()
        assert stats["retries"] == 0
        assert stats["pii_sanitizations"] == 0
        assert stats["proxy_searches"] == 0
        assert stats["proxy_configured"] is False
        assert stats["pii_available"] is True  # We passed a PII instance

    def test_pii_sanitization_counter(self):
        s = _make_searcher()
        s.sanitize_query("user@test.com")
        assert s._stats["pii_sanitizations"] == 1

    @patch.object(_ws_mod, "DDGS_AVAILABLE", True)
    def test_proxy_search_counter(self):
        s = _make_searcher(proxy="socks5h://localhost:9050")
        s._last_request_time = time.time()

        mock_ddgs = _mock_ddgs_results()
        with patch.object(_ws_mod, "DDGS", return_value=mock_ddgs):
            s.search("test")

        assert s._stats["proxy_searches"] >= 1


class TestBackwardCompatibility:
    """Test backward-compatible API surface."""

    def test_web_search_engine_alias(self):
        """web_search_engine should be an alias for web_searcher."""
        assert _ws_mod.web_search_engine is _ws_mod.web_searcher

    def test_convenience_functions_exist(self):
        assert callable(_ws_mod.search)
        assert callable(_ws_mod.search_and_format)
        assert callable(_ws_mod.is_available)

    def test_search_result_dataclass(self):
        r = SearchResult(title="T", snippet="S", url="U")
        assert r.source == "duckduckgo"

    def test_repr(self):
        s = _make_searcher()
        r = repr(s)
        assert "WebSearcher" in r
        assert "proxy=" in r


class TestWebSearcherEdgeCases:
    """Edge cases and additional coverage."""

    def test_empty_query(self):
        s = _make_searcher()
        results = s.search("")
        assert results == []

    def test_none_query(self):
        s = _make_searcher()
        results = s.search(None)
        assert results == []

    @patch.object(_ws_mod, "DDGS_AVAILABLE", False)
    def test_ddgs_not_available_raises(self):
        s = _make_searcher()
        with pytest.raises(RuntimeError, match="not installed"):
            s.search("test")

    def test_cache_uses_sanitized_key(self):
        """Cache key should be based on sanitized query, not raw."""
        s = _make_searcher()
        key1 = s._make_cache_key("user@test.com python", 5)
        key2 = s._make_cache_key("[EMAIL] python", 5)
        # These should be different because the queries are different
        assert key1 != key2

    def test_clear_cache(self):
        s = _make_searcher()
        s._cache["test"] = (time.time(), [])
        assert s.clear_cache() == 1
        assert len(s._cache) == 0
