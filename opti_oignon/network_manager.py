#!/usr/bin/env python3
"""
NETWORK MANAGER -- Connectivity Detection & Health Polling (S71)
=================================================================

Monitors Ollama availability through background health polling.
Detects online/offline transitions and fires callbacks so that
other components (executor, sync queue, UI) can react.

Thread-safe status updates via threading.Lock. Background polling
runs in a daemon thread that stops gracefully on shutdown.

Configuration loaded from config/network.yaml with sensible defaults.

Author: Leon
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

# Conditional import of ollama
try:
    import ollama as _ollama_module
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    _ollama_module = None


# =============================================================================
# CONSTANTS
# =============================================================================

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "network.yaml"

DEFAULT_POLL_INTERVAL = 15
DEFAULT_TIMEOUT = 5
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_LATENCY_WARNING_MS = 3000


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class NetworkStatus:
    """Current network/Ollama connectivity status.

    Attributes:
        online: Whether Ollama is reachable.
        ollama_reachable: Whether Ollama API responded.
        embedding_reachable: Whether the embedding model is available.
        last_check: Timestamp of last health check (epoch seconds).
        last_error: Description of the last error, or empty string.
        latency_ms: Latency of the last successful Ollama check in ms.
        consecutive_failures: Number of consecutive failed checks.
    """
    online: bool = False
    ollama_reachable: bool = False
    embedding_reachable: bool = False
    last_check: float = 0.0
    last_error: str = ""
    latency_ms: float = 0.0
    consecutive_failures: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "online": self.online,
            "ollama_reachable": self.ollama_reachable,
            "embedding_reachable": self.embedding_reachable,
            "last_check": self.last_check,
            "last_error": self.last_error,
            "latency_ms": round(self.latency_ms, 2),
            "consecutive_failures": self.consecutive_failures,
        }


# =============================================================================
# NETWORK MANAGER
# =============================================================================


class NetworkManager:
    """Monitors Ollama connectivity with background health polling.

    Polls Ollama at a configurable interval and maintains a thread-safe
    NetworkStatus. Fires on_status_change callbacks when the online/offline
    state transitions.

    Args:
        config_path: Path to YAML config file. None uses the default.
        auto_start: If True, start polling immediately on construction.
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        auto_start: bool = False,
    ):
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._config: dict[str, Any] = {}
        self._load_config()

        self._status = NetworkStatus()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._callbacks: list[Callable[[NetworkStatus, NetworkStatus], None]] = []

        if auto_start and self._config.get("enabled", True):
            self.start()

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def _load_config(self) -> None:
        """Load configuration from YAML file with defaults."""
        defaults = {
            "enabled": True,
            "poll_interval_seconds": DEFAULT_POLL_INTERVAL,
            "timeout_seconds": DEFAULT_TIMEOUT,
            "max_consecutive_failures": DEFAULT_MAX_CONSECUTIVE_FAILURES,
            "check_embedding": True,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "track_latency": True,
            "latency_warning_ms": DEFAULT_LATENCY_WARNING_MS,
        }
        try:
            if self._config_path.exists():
                with open(self._config_path, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                defaults.update(loaded)
        except Exception as e:
            logger.warning("Failed to load network config from %s: %s", self._config_path, e)

        self._config = defaults

    def get_config(self) -> dict:
        """Return a copy of the current configuration."""
        return dict(self._config)

    def update_config(self, **kwargs: Any) -> None:
        """Update configuration values and persist to YAML.

        Only known keys are accepted; unknown keys are ignored.
        """
        known_keys = {
            "enabled", "poll_interval_seconds", "timeout_seconds",
            "max_consecutive_failures", "check_embedding", "embedding_model",
            "track_latency", "latency_warning_ms",
        }
        for key, value in kwargs.items():
            if key in known_keys and value is not None:
                self._config[key] = value
        self._save_config()

    def _save_config(self) -> None:
        """Persist current config to YAML."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.warning("Failed to save network config: %s", e)

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    @property
    def is_online(self) -> bool:
        """Whether Ollama is currently reachable."""
        with self._lock:
            return self._status.online

    @property
    def status(self) -> NetworkStatus:
        """Return a snapshot of the current network status."""
        with self._lock:
            return NetworkStatus(
                online=self._status.online,
                ollama_reachable=self._status.ollama_reachable,
                embedding_reachable=self._status.embedding_reachable,
                last_check=self._status.last_check,
                last_error=self._status.last_error,
                latency_ms=self._status.latency_ms,
                consecutive_failures=self._status.consecutive_failures,
            )

    # -----------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------

    def on_status_change(self, callback: Callable[[NetworkStatus, NetworkStatus], None]) -> None:
        """Register a callback for online/offline transitions.

        The callback receives (old_status, new_status) and is called
        only when the online flag changes.

        Args:
            callback: Function to call on status transition.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove a previously registered callback."""
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    def _fire_callbacks(self, old_status: NetworkStatus, new_status: NetworkStatus) -> None:
        """Fire all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(old_status, new_status)
            except Exception as e:
                logger.error("Network status callback error: %s", e)

    # -----------------------------------------------------------------
    # Health checks
    # -----------------------------------------------------------------

    def check_ollama(self) -> bool:
        """Check if Ollama API is reachable.

        Returns:
            True if Ollama responds to a list() call.
        """
        if not OLLAMA_AVAILABLE:
            return False
        try:
            _ollama_module.list()
            return True
        except Exception:
            return False

    def check_embedding(self) -> bool:
        """Check if the embedding model is available.

        Returns:
            True if the embedding model is found in Ollama's model list.
        """
        if not OLLAMA_AVAILABLE:
            return False
        embedding_model = self._config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        try:
            response = _ollama_module.list()
            models = []
            if hasattr(response, "models"):
                models = response.models or []
            elif isinstance(response, dict):
                models = response.get("models", [])
            else:
                models = list(response) if response else []

            for m in models:
                name = getattr(m, "model", "") or (m.get("model", "") if isinstance(m, dict) else "")
                # Match with or without tag
                if name == embedding_model or name.startswith(f"{embedding_model}:"):
                    return True
            return False
        except Exception:
            return False

    def poll_once(self) -> NetworkStatus:
        """Run a single health check cycle and update status.

        This is the core check logic used by both the background thread
        and manual invocations. Thread-safe.

        Returns:
            The new NetworkStatus after the check.
        """
        now = time.time()
        ollama_ok = False
        embedding_ok = False
        latency_ms = 0.0
        error = ""

        # Check Ollama
        start = time.time()
        try:
            ollama_ok = self.check_ollama()
            latency_ms = (time.time() - start) * 1000
        except Exception as e:
            error = str(e)
            latency_ms = (time.time() - start) * 1000

        # Check embedding if configured
        if ollama_ok and self._config.get("check_embedding", True):
            try:
                embedding_ok = self.check_embedding()
            except Exception as e:
                error = f"Embedding check failed: {e}"

        # Determine online state
        max_failures = self._config.get(
            "max_consecutive_failures", DEFAULT_MAX_CONSECUTIVE_FAILURES
        )

        with self._lock:
            old_status = NetworkStatus(
                online=self._status.online,
                ollama_reachable=self._status.ollama_reachable,
                embedding_reachable=self._status.embedding_reachable,
                last_check=self._status.last_check,
                last_error=self._status.last_error,
                latency_ms=self._status.latency_ms,
                consecutive_failures=self._status.consecutive_failures,
            )

            if ollama_ok:
                self._status.consecutive_failures = 0
                self._status.online = True
                self._status.last_error = ""
            else:
                self._status.consecutive_failures += 1
                if not error:
                    error = "Ollama unreachable"
                self._status.last_error = error
                if self._status.consecutive_failures >= max_failures:
                    self._status.online = False

            self._status.ollama_reachable = ollama_ok
            self._status.embedding_reachable = embedding_ok
            self._status.last_check = now
            self._status.latency_ms = latency_ms

            new_status = NetworkStatus(
                online=self._status.online,
                ollama_reachable=self._status.ollama_reachable,
                embedding_reachable=self._status.embedding_reachable,
                last_check=self._status.last_check,
                last_error=self._status.last_error,
                latency_ms=self._status.latency_ms,
                consecutive_failures=self._status.consecutive_failures,
            )

        # Fire callbacks on transition
        if old_status.online != new_status.online:
            transition = "online" if new_status.online else "offline"
            logger.info("Network status transition: %s", transition)
            self._fire_callbacks(old_status, new_status)

        # Latency warning
        warning_ms = self._config.get("latency_warning_ms", DEFAULT_LATENCY_WARNING_MS)
        if ollama_ok and latency_ms > warning_ms:
            logger.warning("Ollama latency %.0fms exceeds threshold %dms", latency_ms, warning_ms)

        return new_status

    # -----------------------------------------------------------------
    # Background polling
    # -----------------------------------------------------------------

    def start(self) -> None:
        """Start background health polling thread.

        If already running, this is a no-op.
        """
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return

        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="network-manager-poll",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info("Network manager polling started (interval=%ds)",
                     self._config.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL))

    def stop(self) -> None:
        """Stop background health polling thread.

        Blocks until the thread exits (with a short timeout).
        """
        self._stop_event.set()
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5)
        self._poll_thread = None
        logger.info("Network manager polling stopped")

    @property
    def running(self) -> bool:
        """Whether the background polling thread is active."""
        return self._poll_thread is not None and self._poll_thread.is_alive()

    def _poll_loop(self) -> None:
        """Background polling loop. Runs until stop_event is set."""
        interval = self._config.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL)
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception as e:
                logger.error("Network poll error: %s", e)
            # Wait for interval, but check stop_event frequently
            self._stop_event.wait(timeout=interval)


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

try:
    network_manager = NetworkManager(auto_start=False)
    # Immediate connectivity check so we start in the correct state
    # instead of defaulting to offline
    try:
        network_manager.poll_once()
    except Exception:
        pass  # Non-fatal: will retry on first request
except Exception as e:
    logger.warning("Failed to create NetworkManager singleton: %s", e)
    network_manager = None  # type: ignore[assignment]
