#!/usr/bin/env python3
"""
Tests for S72 frontend components -- Step 3.

Covers:
- PerformanceDashboard.svelte existence and key elements
- performance.ts API client existence and exports
- types.ts Performance interfaces
- Settings page Performance tab integration
"""

from pathlib import Path

import pytest

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "src"
_LIB = _FRONTEND / "lib"
_API = _LIB / "api"
_PANELS = _LIB / "components" / "panels"
_TYPES = _LIB / "types.ts"
_SETTINGS = _FRONTEND / "routes" / "settings" / "+page.svelte"


class TestPerformanceApiClient:
    """Tests for performance.ts API client."""

    def test_file_exists(self):
        assert (_API / "performance.ts").exists()

    def test_exports_functions(self):
        content = (_API / "performance.ts").read_text(encoding="utf-8")
        assert "getPerformanceSummary" in content
        assert "getLatencyStats" in content
        assert "getDriftResults" in content
        assert "getRecommendations" in content
        assert "getPerformanceHistory" in content
        assert "getThroughput" in content
        assert "getUtilization" in content
        assert "cleanupMetrics" in content

    def test_uses_api_client(self):
        content = (_API / "performance.ts").read_text(encoding="utf-8")
        assert "apiClient" in content

    def test_endpoints_correct(self):
        content = (_API / "performance.ts").read_text(encoding="utf-8")
        assert "/api/performance/summary" in content
        assert "/api/performance/latency" in content
        assert "/api/performance/drift" in content
        assert "/api/performance/recommendations" in content
        assert "/api/performance/history" in content
        assert "/api/performance/throughput" in content
        assert "/api/performance/utilization" in content
        assert "/api/performance/cleanup" in content


class TestPerformanceTypes:
    """Tests for Performance types in types.ts."""

    def test_performance_summary_type(self):
        content = _TYPES.read_text(encoding="utf-8")
        assert "PerformanceSummary" in content

    def test_performance_latency_stats_type(self):
        content = _TYPES.read_text(encoding="utf-8")
        assert "PerformanceLatencyStats" in content

    def test_performance_drift_types(self):
        content = _TYPES.read_text(encoding="utf-8")
        assert "PerformanceDriftEntry" in content
        assert "PerformanceDriftResponse" in content

    def test_performance_recommendation_types(self):
        content = _TYPES.read_text(encoding="utf-8")
        assert "PerformanceRecommendation" in content
        assert "PerformanceRecommendationsResponse" in content

    def test_performance_history_types(self):
        content = _TYPES.read_text(encoding="utf-8")
        assert "PerformanceHistoryRecord" in content
        assert "PerformanceHistoryResponse" in content

    def test_performance_throughput_type(self):
        content = _TYPES.read_text(encoding="utf-8")
        assert "PerformanceThroughput" in content

    def test_performance_utilization_type(self):
        content = _TYPES.read_text(encoding="utf-8")
        assert "PerformanceUtilization" in content


class TestPerformanceDashboard:
    """Tests for PerformanceDashboard.svelte component."""

    def test_file_exists(self):
        assert (_PANELS / "PerformanceDashboard.svelte").exists()

    def test_imports_api_functions(self):
        content = (_PANELS / "PerformanceDashboard.svelte").read_text(encoding="utf-8")
        assert "getPerformanceSummary" in content
        assert "getDriftResults" in content
        assert "getRecommendations" in content

    def test_has_throughput_section(self):
        content = (_PANELS / "PerformanceDashboard.svelte").read_text(encoding="utf-8")
        assert "Tokens In/s" in content
        assert "Tokens Out/s" in content

    def test_has_latency_section(self):
        content = (_PANELS / "PerformanceDashboard.svelte").read_text(encoding="utf-8")
        assert "Latency by Model" in content

    def test_has_utilization_section(self):
        content = (_PANELS / "PerformanceDashboard.svelte").read_text(encoding="utf-8")
        assert "Model Utilization" in content

    def test_has_drift_alerts_section(self):
        content = (_PANELS / "PerformanceDashboard.svelte").read_text(encoding="utf-8")
        assert "Drift Alerts" in content

    def test_has_recommendations_section(self):
        content = (_PANELS / "PerformanceDashboard.svelte").read_text(encoding="utf-8")
        assert "Recommendations" in content

    def test_has_auto_refresh_toggle(self):
        content = (_PANELS / "PerformanceDashboard.svelte").read_text(encoding="utf-8")
        assert "autoRefresh" in content
        assert "toggleAutoRefresh" in content

    def test_has_cleanup_button(self):
        content = (_PANELS / "PerformanceDashboard.svelte").read_text(encoding="utf-8")
        assert "Cleanup Old Records" in content
        assert "handleCleanup" in content


class TestSettingsIntegration:
    """Tests for Performance tab in settings page."""

    def test_imports_performance_dashboard(self):
        content = _SETTINGS.read_text(encoding="utf-8")
        assert "PerformanceDashboard" in content

    def test_has_performance_tab(self):
        content = _SETTINGS.read_text(encoding="utf-8")
        assert "'performance'" in content
        assert "Performance" in content
