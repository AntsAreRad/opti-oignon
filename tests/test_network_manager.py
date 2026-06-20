#!/usr/bin/env python3
"""
Tests for NetworkManager -- S71 Step 1: Connectivity detection & health polling.

Covers:
- YAML config loading and defaults
- Config update and persistence
- check_ollama() with mock
- check_embedding() with mock
- poll_once() online/offline transitions
- Consecutive failure threshold
- Callback registration and firing
- Callback removal
- Background polling start/stop
- Thread safety
- Status snapshot immutability
- Latency tracking
- NetworkStatus dataclass
- Singleton creation
"""

import importlib.util
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Direct module import (bypass __init__.py which requires ollama)
# ---------------------------------------------------------------------------

_mod_path = Path(__file__).resolve().parent.parent / "opti_oignon" / "network_manager.py"

_spec = importlib.util.spec_from_file_location("network_manager_mod", _mod_path)
_mod = importlib.util.module_from_spec(_spec)

# Provide a mock ollama before exec
_mock_ollama = MagicMock()
sys.modules.setdefault("ollama", _mock_ollama)

_spec.loader.exec_module(_mod)

NetworkManager = _mod.NetworkManager
NetworkStatus = _mod.NetworkStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    """Create a temporary network.yaml config."""
    cfg = {
        "enabled": True,
        "poll_interval_seconds": 1,
        "timeout_seconds": 2,
        "max_consecutive_failures": 3,
        "check_embedding": True,
        "embedding_model": "mxbai-embed-large",
        "track_latency": True,
        "latency_warning_ms": 3000,
    }
    if overrides:
        cfg.update(overrides)
    path = tmp_path / "network.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


# ===========================================================================
# TEST CLASSES
# ===========================================================================


class TestNetworkStatus:
    """Tests for NetworkStatus dataclass."""

    def test_default_values(self):
        status = NetworkStatus()
        assert status.online is False
        assert status.ollama_reachable is False
        assert status.embedding_reachable is False
        assert status.last_check == 0.0
        assert status.last_error == ""
        assert status.latency_ms == 0.0
        assert status.consecutive_failures == 0

    def test_to_dict(self):
        status = NetworkStatus(online=True, latency_ms=42.567)
        d = status.to_dict()
        assert d["online"] is True
        assert d["latency_ms"] == 42.57
        assert "consecutive_failures" in d

    def test_custom_values(self):
        status = NetworkStatus(
            online=True,
            ollama_reachable=True,
            embedding_reachable=True,
            last_check=1000.0,
            last_error="",
            latency_ms=15.5,
            consecutive_failures=0,
        )
        assert status.online is True
        assert status.latency_ms == 15.5


class TestNetworkManagerConfig:
    """Tests for config loading, defaults, update, and persistence."""

    def test_default_config_values(self, tmp_path):
        path = tmp_path / "nonexistent.yaml"
        nm = NetworkManager(config_path=path, auto_start=False)
        cfg = nm.get_config()
        assert cfg["enabled"] is True
        assert cfg["poll_interval_seconds"] == 15
        assert cfg["timeout_seconds"] == 5
        assert cfg["max_consecutive_failures"] == 3

    def test_load_from_yaml(self, tmp_path):
        path = _make_config(tmp_path, {"poll_interval_seconds": 30})
        nm = NetworkManager(config_path=path, auto_start=False)
        assert nm.get_config()["poll_interval_seconds"] == 30

    def test_update_config_persists(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        nm.update_config(poll_interval_seconds=60)
        assert nm.get_config()["poll_interval_seconds"] == 60
        # Verify persistence
        with open(path) as f:
            saved = yaml.safe_load(f)
        assert saved["poll_interval_seconds"] == 60

    def test_update_ignores_unknown_keys(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        nm.update_config(unknown_key="value")
        assert "unknown_key" not in nm.get_config()


class TestCheckOllama:
    """Tests for check_ollama() with mocked ollama module."""

    def test_check_ollama_success(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        with patch.object(_mod, "_ollama_module") as mock_ollama:
            mock_ollama.list.return_value = {"models": []}
            with patch.object(_mod, "OLLAMA_AVAILABLE", True):
                assert nm.check_ollama() is True

    def test_check_ollama_failure(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        with patch.object(_mod, "_ollama_module") as mock_ollama:
            mock_ollama.list.side_effect = ConnectionError("refused")
            with patch.object(_mod, "OLLAMA_AVAILABLE", True):
                assert nm.check_ollama() is False

    def test_check_ollama_not_available(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        with patch.object(_mod, "OLLAMA_AVAILABLE", False):
            assert nm.check_ollama() is False


class TestCheckEmbedding:
    """Tests for check_embedding() with mocked ollama module."""

    def test_embedding_found(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        mock_model = MagicMock()
        mock_model.model = "mxbai-embed-large:latest"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        with patch.object(_mod, "_ollama_module") as mock_ollama:
            mock_ollama.list.return_value = mock_response
            with patch.object(_mod, "OLLAMA_AVAILABLE", True):
                assert nm.check_embedding() is True

    def test_embedding_not_found(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        mock_model = MagicMock()
        mock_model.model = "llama3:latest"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        with patch.object(_mod, "_ollama_module") as mock_ollama:
            mock_ollama.list.return_value = mock_response
            with patch.object(_mod, "OLLAMA_AVAILABLE", True):
                assert nm.check_embedding() is False

    def test_embedding_ollama_unavailable(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        with patch.object(_mod, "OLLAMA_AVAILABLE", False):
            assert nm.check_embedding() is False


class TestPollOnce:
    """Tests for poll_once() status transitions and failure counting."""

    def test_poll_once_online(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                result = nm.poll_once()
        assert result.online is True
        assert result.ollama_reachable is True
        assert result.embedding_reachable is True
        assert result.consecutive_failures == 0

    def test_poll_once_offline_after_threshold(self, tmp_path):
        path = _make_config(tmp_path, {"max_consecutive_failures": 2})
        nm = NetworkManager(config_path=path, auto_start=False)
        with patch.object(nm, "check_ollama", return_value=False):
            nm.poll_once()  # failure 1
            result = nm.poll_once()  # failure 2 -> offline
        assert result.online is False
        assert result.consecutive_failures == 2

    def test_poll_once_single_failure_stays_online(self, tmp_path):
        path = _make_config(tmp_path, {"max_consecutive_failures": 3})
        nm = NetworkManager(config_path=path, auto_start=False)
        # First set online
        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                nm.poll_once()
        # One failure -> should remain online
        with patch.object(nm, "check_ollama", return_value=False):
            result = nm.poll_once()
        assert result.online is True  # still online, only 1 failure
        assert result.consecutive_failures == 1

    def test_poll_once_recovery_resets_failures(self, tmp_path):
        path = _make_config(tmp_path, {"max_consecutive_failures": 3})
        nm = NetworkManager(config_path=path, auto_start=False)
        # Two failures
        with patch.object(nm, "check_ollama", return_value=False):
            nm.poll_once()
            nm.poll_once()
        assert nm.status.consecutive_failures == 2
        # Recovery
        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                result = nm.poll_once()
        assert result.consecutive_failures == 0
        assert result.online is True

    def test_poll_once_tracks_latency(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                result = nm.poll_once()
        assert result.latency_ms >= 0


class TestCallbacks:
    """Tests for status change callbacks."""

    def test_callback_fires_on_transition(self, tmp_path):
        path = _make_config(tmp_path, {"max_consecutive_failures": 1})
        nm = NetworkManager(config_path=path, auto_start=False)
        transitions = []

        def on_change(old, new):
            transitions.append((old.online, new.online))

        nm.on_status_change(on_change)

        # Go online
        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                nm.poll_once()

        assert len(transitions) == 1
        assert transitions[0] == (False, True)

    def test_callback_fires_on_offline_transition(self, tmp_path):
        path = _make_config(tmp_path, {"max_consecutive_failures": 1})
        nm = NetworkManager(config_path=path, auto_start=False)
        transitions = []

        def on_change(old, new):
            transitions.append((old.online, new.online))

        nm.on_status_change(on_change)

        # Go online first
        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                nm.poll_once()
        # Go offline
        with patch.object(nm, "check_ollama", return_value=False):
            nm.poll_once()

        assert len(transitions) == 2
        assert transitions[1] == (True, False)

    def test_no_callback_when_no_transition(self, tmp_path):
        path = _make_config(tmp_path, {"max_consecutive_failures": 1})
        nm = NetworkManager(config_path=path, auto_start=False)
        call_count = [0]

        def on_change(old, new):
            call_count[0] += 1

        nm.on_status_change(on_change)

        # Two consecutive online checks
        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                nm.poll_once()
                nm.poll_once()

        assert call_count[0] == 1  # only the initial False->True

    def test_remove_callback(self, tmp_path):
        path = _make_config(tmp_path, {"max_consecutive_failures": 1})
        nm = NetworkManager(config_path=path, auto_start=False)
        call_count = [0]

        def on_change(old, new):
            call_count[0] += 1

        nm.on_status_change(on_change)
        nm.remove_callback(on_change)

        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                nm.poll_once()

        assert call_count[0] == 0

    def test_callback_error_does_not_crash(self, tmp_path):
        path = _make_config(tmp_path, {"max_consecutive_failures": 1})
        nm = NetworkManager(config_path=path, auto_start=False)

        def bad_callback(old, new):
            raise RuntimeError("callback error")

        nm.on_status_change(bad_callback)

        # Should not raise
        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                nm.poll_once()


class TestBackgroundPolling:
    """Tests for background polling thread start/stop."""

    def test_start_and_stop(self, tmp_path):
        path = _make_config(tmp_path, {"poll_interval_seconds": 0.1})
        nm = NetworkManager(config_path=path, auto_start=False)
        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                nm.start()
                assert nm.running is True
                time.sleep(0.3)
                nm.stop()
                assert nm.running is False

    def test_start_idempotent(self, tmp_path):
        path = _make_config(tmp_path, {"poll_interval_seconds": 0.5})
        nm = NetworkManager(config_path=path, auto_start=False)
        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                nm.start()
                thread1 = nm._poll_thread
                nm.start()  # should be no-op
                assert nm._poll_thread is thread1
                nm.stop()

    def test_stop_when_not_started(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        nm.stop()  # should not raise
        assert nm.running is False


class TestStatusSnapshot:
    """Tests for thread-safe status access."""

    def test_status_returns_snapshot(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                nm.poll_once()
        snap = nm.status
        assert snap.online is True
        # Modify internal status; snapshot should be unaffected
        with nm._lock:
            nm._status.online = False
        assert snap.online is True  # snapshot is independent

    def test_is_online_property(self, tmp_path):
        path = _make_config(tmp_path)
        nm = NetworkManager(config_path=path, auto_start=False)
        assert nm.is_online is False
        with patch.object(nm, "check_ollama", return_value=True):
            with patch.object(nm, "check_embedding", return_value=True):
                nm.poll_once()
        assert nm.is_online is True
