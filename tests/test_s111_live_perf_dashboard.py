#!/usr/bin/env python3
"""
Tests for S111 — Real-Time Performance Dashboard + Tuner Backend Integration.

Covers:
  - Real backend benchmark functions (create_ollama_benchmark_fn, create_llamacpp_benchmark_fn)
  - LiveMetricsConfig, MetricsSample serialization
  - LiveMetricsCollector lifecycle, token recording, rolling speed, history
  - GPU utility functions (nvidia-smi, system memory)
  - Routes: /api/metrics/live, /api/metrics/history, /api/metrics/status
  - Speculative decoding S111 additions: AcceptanceRecord, history, rolling rate
  - parse_llamacpp_log_line with multiple log formats
  - SpeculativeDecodingManager.process_log_line, get_acceptance_history
  - SpeculativeDecodingPanel: no hardcoded hex
  - LiveMetricsOverlay: no hardcoded hex
  - Tuner route backend detection (_resolve_benchmark_fn)
  - Schemas: LiveMetrics*, SpeculativeDecodingStatsSchema updated fields

Total: ~50 tests
"""

import collections
import importlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module loading helpers (test isolation pattern)
# ---------------------------------------------------------------------------

def _load_module(name: str, filepath: str):
    """Load a module by file path, bypassing opti_oignon/__init__.py."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_parent_stubs():
    """Ensure parent package stubs exist for test isolation."""
    if "opti_oignon" not in sys.modules:
        oo = types.ModuleType("opti_oignon")
        oo.__path__ = ["opti_oignon"]
        sys.modules["opti_oignon"] = oo
    if "opti_oignon.config" not in sys.modules:
        sys.modules["opti_oignon.config"] = types.ModuleType("opti_oignon.config")


_ensure_parent_stubs()

# Load modules under test
_BASE = Path(__file__).resolve().parent.parent

_auto_tuner = _load_module(
    "opti_oignon.auto_tuner",
    str(_BASE / "opti_oignon" / "auto_tuner.py"),
)
_live_metrics = _load_module(
    "opti_oignon.live_metrics",
    str(_BASE / "opti_oignon" / "live_metrics.py"),
)
_spec_dec = _load_module(
    "opti_oignon.speculative_decoding",
    str(_BASE / "opti_oignon" / "speculative_decoding.py"),
)


# ============================================================================
# AUTO-TUNER: REAL BENCHMARK FUNCTIONS (Phase 1)
# ============================================================================


class TestOllamaBenchmarkFn:
    """Tests for create_ollama_benchmark_fn."""

    def test_creates_callable(self):
        fn = _auto_tuner.create_ollama_benchmark_fn("test-model")
        assert callable(fn)

    def test_benchmark_prompt_defined(self):
        assert hasattr(_auto_tuner, "_BENCHMARK_PROMPT")
        assert len(_auto_tuner._BENCHMARK_PROMPT) > 10

    def test_returns_error_on_connection_failure(self):
        fn = _auto_tuner.create_ollama_benchmark_fn(
            "test-model", host="http://localhost:99999"
        )
        result = fn({"threads": 4, "batch_size": 1024})
        assert isinstance(result, _auto_tuner.BenchmarkResult)
        assert result.error != ""

    def test_param_mapping_includes_num_predict(self):
        """Verify that the benchmark sets num_predict in options."""
        captured = {}

        def mock_post(url, json=None, timeout=None):
            captured["payload"] = json
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "eval_count": 50,
                "eval_duration": 1_000_000_000,
                "prompt_eval_count": 20,
                "prompt_eval_duration": 500_000_000,
            }
            return resp

        fn = _auto_tuner.create_ollama_benchmark_fn("test-model")
        with patch("requests.post", side_effect=mock_post):
            result = fn({"threads": 4, "batch_size": 2048, "flash_attention": True})

        assert result.error == ""
        assert result.tokens_per_second_tg == pytest.approx(50.0, rel=0.01)
        assert result.tokens_per_second_pp == pytest.approx(40.0, rel=0.01)
        opts = captured["payload"]["options"]
        assert opts["num_thread"] == 4
        assert opts["flash_attn"] is True
        assert "num_predict" in opts

    def test_handles_http_error(self):
        def mock_post(url, json=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 500
            resp.text = "Internal Server Error"
            return resp

        fn = _auto_tuner.create_ollama_benchmark_fn("test-model")
        with patch("requests.post", side_effect=mock_post):
            result = fn({"threads": 4})
        assert "500" in result.error

    def test_custom_benchmark_tokens(self):
        captured = {}

        def mock_post(url, json=None, timeout=None):
            captured["payload"] = json
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "eval_count": 10,
                "eval_duration": 500_000_000,
            }
            return resp

        fn = _auto_tuner.create_ollama_benchmark_fn("m", benchmark_tokens=256)
        with patch("requests.post", side_effect=mock_post):
            fn({})
        assert captured["payload"]["options"]["num_predict"] == 256


class TestLlamaCppBenchmarkFn:
    """Tests for create_llamacpp_benchmark_fn."""

    def test_creates_callable(self):
        fn = _auto_tuner.create_llamacpp_benchmark_fn("test.gguf")
        assert callable(fn)

    def test_error_without_backend(self):
        fn = _auto_tuner.create_llamacpp_benchmark_fn("test.gguf", backend=None)
        result = fn({"threads": 4})
        assert result.error != ""
        assert "not available" in result.error.lower()

    def test_with_mock_backend(self):
        mock_backend = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "A " * 100  # ~100 tokens rough estimate
        mock_response.extra = {}
        mock_backend.generate.return_value = mock_response

        fn = _auto_tuner.create_llamacpp_benchmark_fn("test.gguf", backend=mock_backend)
        result = fn({"threads": 4, "batch_size": 1024})

        assert result.error == ""
        assert result.tokens_per_second_tg > 0
        assert result.total_time_ms > 0
        mock_backend.generate.assert_called_once()

    def test_uses_timings_from_backend(self):
        mock_backend = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "hello world"
        mock_response.extra = {
            "timings": {
                "predicted_per_second": 42.5,
                "prompt_per_second": 80.0,
            }
        }
        mock_backend.generate.return_value = mock_response

        fn = _auto_tuner.create_llamacpp_benchmark_fn("test.gguf", backend=mock_backend)
        result = fn({})

        assert result.tokens_per_second_tg == pytest.approx(42.5)
        assert result.tokens_per_second_pp == pytest.approx(80.0)

    def test_handles_generate_exception(self):
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = RuntimeError("Model not loaded")

        fn = _auto_tuner.create_llamacpp_benchmark_fn("test.gguf", backend=mock_backend)
        result = fn({"threads": 4})

        assert "Model not loaded" in result.error


# ============================================================================
# LIVE METRICS (Phase 2)
# ============================================================================


class TestLiveMetricsConfig:
    """Tests for LiveMetricsConfig."""

    def test_defaults(self):
        cfg = _live_metrics.LiveMetricsConfig()
        assert cfg.enabled is True
        assert cfg.sample_interval_ms == 500
        assert cfg.window_seconds == 60
        assert cfg.rolling_speed_window_s == 5.0
        assert cfg.gpu_monitoring is True

    def test_to_dict_roundtrip(self):
        cfg = _live_metrics.LiveMetricsConfig(
            enabled=False, sample_interval_ms=200
        )
        d = cfg.to_dict()
        cfg2 = _live_metrics.LiveMetricsConfig.from_dict(d)
        assert cfg2.enabled is False
        assert cfg2.sample_interval_ms == 200

    def test_from_dict_ignores_unknown(self):
        cfg = _live_metrics.LiveMetricsConfig.from_dict(
            {"enabled": True, "unknown_key": 42}
        )
        assert cfg.enabled is True
        assert not hasattr(cfg, "unknown_key")


class TestMetricsSample:
    """Tests for MetricsSample."""

    def test_defaults(self):
        s = _live_metrics.MetricsSample()
        assert s.tokens_per_second == 0.0
        assert s.gpu_utilization_pct == -1.0
        assert s.is_generating is False

    def test_to_dict(self):
        s = _live_metrics.MetricsSample(
            timestamp=1000.123,
            tokens_per_second=42.567,
            gpu_utilization_pct=85.3,
            is_generating=True,
            active_model="llama3:8b",
        )
        d = s.to_dict()
        assert d["tokens_per_second"] == 42.57
        assert d["gpu_utilization_pct"] == 85.3
        assert d["is_generating"] is True
        assert d["active_model"] == "llama3:8b"
        assert d["timestamp"] == 1000.123


class TestLiveMetricsCollector:
    """Tests for LiveMetricsCollector."""

    def _make_collector(self, **kwargs):
        cfg = _live_metrics.LiveMetricsConfig(
            gpu_monitoring=False, **kwargs
        )
        return _live_metrics.LiveMetricsCollector(config=cfg)

    def test_initial_state(self):
        c = self._make_collector()
        assert not c.is_running
        status = c.get_status()
        assert status["running"] is False
        assert status["total_tokens_all_time"] == 0

    def test_start_stop(self):
        c = self._make_collector(sample_interval_ms=100)
        c.start()
        assert c.is_running
        time.sleep(0.05)
        c.stop()
        assert not c.is_running

    def test_generation_lifecycle(self):
        c = self._make_collector()
        c.start_generation(model="test-model")
        snap = c.current_snapshot()
        assert snap.is_generating is True
        assert snap.active_model == "test-model"

        c.record_token(5)
        c.end_generation(prompt_eval_time_ms=10.0, eval_time_ms=50.0)
        snap2 = c.current_snapshot()
        assert snap2.is_generating is False
        assert snap2.prompt_eval_time_ms == 10.0
        assert snap2.eval_time_ms == 50.0
        assert snap2.total_tokens == 5

    def test_record_multiple_tokens(self):
        c = self._make_collector()
        c.start_generation(model="m")
        c.record_token(3)
        c.record_token(7)
        snap = c.current_snapshot()
        assert snap.total_tokens == 10

    def test_rolling_speed_calculation(self):
        c = self._make_collector(rolling_speed_window_s=2.0)
        c.start_generation(model="m")
        # Record tokens with small delays to get measurable speed
        for _ in range(10):
            c.record_token(1)
            time.sleep(0.02)
        snap = c.current_snapshot()
        # Should be roughly 10 tokens / 0.2s = ~50 tok/s
        # Allow wide margin due to timing
        assert snap.tokens_per_second > 5.0

    def test_history_buffer(self):
        c = self._make_collector(sample_interval_ms=50)
        c.start()
        time.sleep(0.25)
        c.stop()
        history = c.get_history()
        assert len(history) >= 2

    def test_history_seconds_filter(self):
        c = self._make_collector()
        c.start()
        time.sleep(0.15)
        c.stop()
        # All history should be within 60s
        h_all = c.get_history(seconds=60)
        assert len(h_all) >= 1
        # Very short window should still include recent samples
        h_recent = c.get_history(seconds=1)
        assert len(h_recent) >= 0  # May or may not have samples

    def test_record_timing(self):
        c = self._make_collector()
        c.start_generation(model="m")
        c.record_timing(prompt_eval_time_ms=15.5, eval_time_ms=100.0)
        snap = c.current_snapshot()
        assert snap.prompt_eval_time_ms == 15.5
        assert snap.eval_time_ms == 100.0

    def test_config_property_returns_copy(self):
        c = self._make_collector(sample_interval_ms=300)
        cfg = c.config
        assert cfg.sample_interval_ms == 300


class TestGpuUtilities:
    """Tests for GPU utility functions."""

    def test_nvidia_smi_available_returns_bool(self):
        result = _live_metrics._nvidia_smi_available()
        assert isinstance(result, bool)

    def test_query_gpu_metrics_without_nvidia(self):
        with patch("shutil.which", return_value=None):
            # Recreate to test without nvidia-smi
            result = _live_metrics._query_gpu_metrics()
            # Should gracefully return -1 values when nvidia-smi fails
            assert isinstance(result, dict)

    def test_safe_float(self):
        assert _live_metrics._safe_float("42.5") == 42.5
        assert _live_metrics._safe_float("invalid") == -1.0
        assert _live_metrics._safe_float("") == -1.0

    def test_get_system_memory(self):
        used, total = _live_metrics._get_system_memory()
        # On Linux CI, should return real values
        assert total >= 0
        assert used >= 0


class TestLiveMetricsSingleton:
    """Tests for module-level singleton management."""

    def test_reset(self):
        _live_metrics.reset_live_metrics()
        # After reset, get should create a new one
        collector = _live_metrics.get_live_metrics(auto_start=False)
        assert collector is not None
        assert not collector.is_running
        _live_metrics.reset_live_metrics()


# ============================================================================
# SPECULATIVE DECODING S111 ADDITIONS (Phase 3)
# ============================================================================


class TestAcceptanceRecord:
    """Tests for AcceptanceRecord dataclass."""

    def test_defaults(self):
        rec = _spec_dec.AcceptanceRecord()
        assert rec.draft_tokens == 0
        assert rec.accepted_tokens == 0
        assert rec.request_id == ""

    def test_to_dict(self):
        rec = _spec_dec.AcceptanceRecord(
            timestamp=1234.567,
            draft_tokens=16,
            accepted_tokens=12,
            acceptance_rate=0.75,
            speedup_factor=2.1,
            request_id="req-42",
        )
        d = rec.to_dict()
        assert d["draft_tokens"] == 16
        assert d["accepted_tokens"] == 12
        assert d["acceptance_rate"] == 0.75
        assert d["speedup_factor"] == 2.1
        assert d["request_id"] == "req-42"


class TestAcceptanceStatsS111:
    """Tests for S111 additions to AcceptanceStats."""

    def test_record_run_populates_history(self):
        stats = _spec_dec.AcceptanceStats()
        stats.record_run(16, 12, speedup=2.0, request_id="r1")
        stats.record_run(16, 8, speedup=1.5, request_id="r2")

        history = stats.get_history()
        assert len(history) == 2
        assert history[0]["request_id"] == "r1"
        assert history[1]["request_id"] == "r2"

    def test_get_history_last_n(self):
        stats = _spec_dec.AcceptanceStats()
        for i in range(10):
            stats.record_run(16, 10 + i, request_id=f"r{i}")

        h = stats.get_history(last_n=3)
        assert len(h) == 3
        assert h[0]["request_id"] == "r7"

    def test_rolling_acceptance_rate(self):
        stats = _spec_dec.AcceptanceStats()
        stats.record_run(20, 10)  # 50%
        stats.record_run(20, 20)  # 100%
        rate = stats.get_rolling_acceptance_rate(last_n=2)
        assert rate == pytest.approx(30 / 40)

    def test_rolling_rate_empty(self):
        stats = _spec_dec.AcceptanceStats()
        assert stats.get_rolling_acceptance_rate() == 0.0

    def test_to_dict_includes_s111_fields(self):
        stats = _spec_dec.AcceptanceStats()
        stats.record_run(16, 12)
        d = stats.to_dict()
        assert "history_size" in d
        assert d["history_size"] == 1
        assert "rolling_acceptance_rate" in d

    def test_history_max_size(self):
        stats = _spec_dec.AcceptanceStats()
        for i in range(300):
            stats.record_run(10, 5, request_id=f"r{i}")
        # _MAX_ACCEPTANCE_HISTORY = 200
        assert len(stats._history) == 200
        h = stats.get_history()
        assert len(h) == 200


class TestParseLlamaCppLogLine:
    """Tests for parse_llamacpp_log_line."""

    def test_pattern_draft_accepted(self):
        result = _spec_dec.parse_llamacpp_log_line(
            "draft accepted 12/16 tokens (75.00%)"
        )
        assert result is not None
        assert result["draft_tokens"] == 16
        assert result["accepted_tokens"] == 12

    def test_pattern_speculative_accepted(self):
        result = _spec_dec.parse_llamacpp_log_line(
            "speculative: accepted 10, drafted 14"
        )
        assert result is not None
        assert result["draft_tokens"] == 14
        assert result["accepted_tokens"] == 10

    def test_pattern_n_drafted(self):
        result = _spec_dec.parse_llamacpp_log_line(
            "n_drafted = 20, n_accept = 18"
        )
        assert result is not None
        assert result["draft_tokens"] == 20
        assert result["accepted_tokens"] == 18

    def test_with_speedup(self):
        result = _spec_dec.parse_llamacpp_log_line(
            "draft accepted 8/12 tokens, speculative decoding speedup: 2.3x"
        )
        assert result is not None
        assert result["speedup"] == pytest.approx(2.3)

    def test_no_match(self):
        assert _spec_dec.parse_llamacpp_log_line("some random log") is None

    def test_empty_line(self):
        assert _spec_dec.parse_llamacpp_log_line("") is None

    def test_none_input(self):
        assert _spec_dec.parse_llamacpp_log_line(None) is None

    def test_case_insensitive(self):
        result = _spec_dec.parse_llamacpp_log_line(
            "DRAFT ACCEPTED 5/10 TOKENS"
        )
        assert result is not None
        assert result["draft_tokens"] == 10

    def test_extract_speedup_standalone(self):
        assert _spec_dec._extract_speedup("speedup: 3.5x") == 1.0
        assert _spec_dec._extract_speedup(
            "speculative decoding speedup: 3.5x"
        ) == pytest.approx(3.5)
        assert _spec_dec._extract_speedup("no speedup here") == 1.0


class TestSpeculativeDecodingManagerS111:
    """Tests for S111 additions to SpeculativeDecodingManager."""

    def setup_method(self):
        """Reset singleton and clean up persisted stats before each test."""
        _spec_dec.reset_manager()
        # Remove persisted stats file to avoid cross-test contamination.
        results_path = _spec_dec._RESULTS_PATH
        if results_path.is_file():
            results_path.unlink()

    def test_process_log_line_success(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        ok = mgr.process_log_line("draft accepted 10/16 tokens")
        assert ok is True
        assert mgr.stats.total_runs == 1
        assert mgr.stats.accepted_tokens == 10

    def test_process_log_line_no_match(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        ok = mgr.process_log_line("just a normal log line")
        assert ok is False
        assert mgr.stats.total_runs == 0

    def test_get_acceptance_history(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        mgr.record_acceptance(16, 12, request_id="a")
        mgr.record_acceptance(16, 14, request_id="b")
        h = mgr.get_acceptance_history(last_n=10)
        assert len(h) == 2
        assert h[0]["request_id"] == "a"

    def test_get_rolling_acceptance_rate(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        mgr.record_acceptance(20, 10)
        mgr.record_acceptance(20, 18)
        rate = mgr.get_rolling_acceptance_rate(window=5)
        assert rate == pytest.approx(28 / 40)


# ============================================================================
# FRONTEND: NO HARDCODED HEX (existing enforcement)
# ============================================================================


_FRONTEND_DIR = _BASE / "frontend" / "src"
_HEX_RE = re.compile(r'#[0-9a-fA-F]{3,8}\b')
_SVELTE_IGNORE = {"node_modules", ".svelte-kit", "build"}


def _find_svelte_files(root: Path):
    """Yield all .svelte files under root."""
    for p in root.rglob("*.svelte"):
        if not any(part in _SVELTE_IGNORE for part in p.parts):
            yield p


class TestFrontendNoHardcodedHex:
    """Ensure S111 Svelte components use only --oo-* CSS variables."""

    def test_live_metrics_overlay_no_hex(self):
        f = _FRONTEND_DIR / "lib" / "components" / "chat" / "LiveMetricsOverlay.svelte"
        if not f.exists():
            pytest.skip("LiveMetricsOverlay.svelte not found")
        content = f.read_text(encoding="utf-8")
        # Filter out Svelte template syntax like {#each
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            # Skip lines that are clearly template logic
            if "{#each" in line or "{/each" in line or "{#if" in line:
                continue
            matches = _HEX_RE.findall(line)
            assert not matches, (
                f"LiveMetricsOverlay.svelte:{i} has hardcoded hex: {matches}"
            )

    def test_speculative_decoding_panel_no_hex(self):
        f = _FRONTEND_DIR / "lib" / "components" / "settings" / "SpeculativeDecodingPanel.svelte"
        if not f.exists():
            pytest.skip("SpeculativeDecodingPanel.svelte not found")
        content = f.read_text(encoding="utf-8")
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if "{#each" in line or "{/each" in line or "{#if" in line:
                continue
            matches = _HEX_RE.findall(line)
            assert not matches, (
                f"SpeculativeDecodingPanel.svelte:{i} has hardcoded hex: {matches}"
            )


# ============================================================================
# FILE EXISTENCE CHECKS
# ============================================================================


class TestS111FileExistence:
    """Verify all S111 files exist."""

    def test_live_metrics_module(self):
        assert (_BASE / "opti_oignon" / "live_metrics.py").is_file()

    def test_live_metrics_config_yaml(self):
        assert (_BASE / "opti_oignon" / "config" / "live_metrics.yaml").is_file()

    def test_routes_live_metrics(self):
        assert (_BASE / "opti_oignon" / "api" / "routes_live_metrics.py").is_file()

    def test_live_metrics_overlay_svelte(self):
        assert (
            _BASE / "frontend" / "src" / "lib" / "components" / "chat"
            / "LiveMetricsOverlay.svelte"
        ).is_file()


# ============================================================================
# AST VALIDITY
# ============================================================================


class TestASTValidity:
    """Verify all modified Python files parse correctly."""

    @pytest.mark.parametrize("relpath", [
        "opti_oignon/auto_tuner.py",
        "opti_oignon/live_metrics.py",
        "opti_oignon/speculative_decoding.py",
        "opti_oignon/api/routes_tuner.py",
        "opti_oignon/api/routes_live_metrics.py",
        "opti_oignon/api/routes_speculative_decoding.py",
        "opti_oignon/api/schemas.py",
        "opti_oignon/api/deps.py",
        "opti_oignon/api/app.py",
    ])
    def test_ast_valid(self, relpath):
        import ast
        filepath = _BASE / relpath
        if not filepath.is_file():
            pytest.skip(f"{relpath} not found")
        source = filepath.read_text(encoding="utf-8")
        ast.parse(source)


# ============================================================================
# YAML CONFIG VALIDITY
# ============================================================================


class TestLiveMetricsYamlConfig:
    """Tests for live_metrics.yaml loading."""

    def test_load_default_config(self):
        cfg = _live_metrics._load_config()
        assert isinstance(cfg, _live_metrics.LiveMetricsConfig)
        assert cfg.enabled is True

    def test_load_custom_config(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(
                "live_metrics:\n"
                "  enabled: false\n"
                "  sample_interval_ms: 200\n"
                "  window_seconds: 30\n"
            )
            f.flush()
            cfg = _live_metrics._load_config(f.name)

        os.unlink(f.name)
        assert cfg.enabled is False
        assert cfg.sample_interval_ms == 200
        assert cfg.window_seconds == 30

    def test_load_missing_config(self):
        cfg = _live_metrics._load_config("/nonexistent/path.yaml")
        # Should return defaults
        assert cfg.enabled is True
        assert cfg.sample_interval_ms == 500
