#!/usr/bin/env python3
"""
MODEL HEALTH MONITOR -- Real-time Ollama Model Availability Tracking (S63)
============================================================================

Monitors Ollama model availability and performance through periodic
health checks. Tracks per-model health status (healthy/degraded/unavailable),
latency, error counts, and consecutive failures.

Integrates with SmartRouter for automatic failover when models degrade.

Configuration is loaded from config/model_health.yaml and can be
overridden via the configure() method.

Author: Leon
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Conditional import of ollama
try:
    import ollama as _ollama_module
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    _ollama_module = None


# Sentinel for distinguishing "not provided" from explicit None
_UNSET = object()


# =============================================================================
# CONSTANTS
# =============================================================================

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "model_health.yaml"

DEFAULT_CHECK_INTERVAL = 60
DEFAULT_DEGRADED_THRESHOLD = 3
DEFAULT_UNAVAILABLE_THRESHOLD = 5
DEFAULT_LATENCY_WARNING_MS = 5000
DEFAULT_MAX_RECORDS = 100


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class ModelStatus(str, Enum):
    """Health status for a monitored model."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class ModelHealthRecord:
    """Health record for a single model.

    Tracks current status, latency, error history, and
    timestamps for monitoring and failover decisions.
    """
    model: str
    status: ModelStatus = ModelStatus.UNKNOWN
    latency_ms: float = 0.0
    last_check: float = 0.0
    last_success: float = 0.0
    error_count: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    check_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "model": self.model,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "last_check": self.last_check,
            "last_success": self.last_success,
            "error_count": self.error_count,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "check_count": self.check_count,
        }

    def reset(self):
        """Reset record to initial unknown state."""
        self.status = ModelStatus.UNKNOWN
        self.latency_ms = 0.0
        self.last_check = 0.0
        self.last_success = 0.0
        self.error_count = 0
        self.consecutive_failures = 0
        self.last_error = ""
        self.check_count = 0


# =============================================================================
# MODEL HEALTH MONITOR
# =============================================================================

class ModelHealthMonitor:
    """Monitors Ollama model availability and performance.

    Runs periodic background health checks against configured models,
    tracking per-model health status. Provides health data for
    SmartRouter failover decisions.

    Usage:
        monitor = ModelHealthMonitor()
        monitor.start()
        record = monitor.get_health("qwen3:32b")
        print(record.status)  # ModelStatus.HEALTHY
        monitor.stop()
    """

    def __init__(
        self,
        enabled: bool = True,
        check_interval: int | None = None,
        degraded_threshold: int | None = None,
        unavailable_threshold: int | None = None,
        latency_warning_ms: int | None = None,
        auto_failover: bool = True,
        max_records: int | None = None,
        config_path: Path | None = None,
        ollama_module: Any = _UNSET,
    ):
        """Initialize the health monitor.

        Args:
            enabled: Whether monitoring is active
            check_interval: Seconds between health checks
            degraded_threshold: Consecutive failures for degraded status
            unavailable_threshold: Consecutive failures for unavailable status
            latency_warning_ms: Latency threshold for warnings
            auto_failover: Whether to enable automatic failover in routing
            max_records: Maximum number of model records to keep
            config_path: Path to YAML config (None = default)
            ollama_module: Ollama module for dependency injection (None = disable, _UNSET = auto)
        """
        # Store constructor values before config load
        self._enabled = enabled
        self._check_interval = check_interval or DEFAULT_CHECK_INTERVAL
        self._degraded_threshold = degraded_threshold or DEFAULT_DEGRADED_THRESHOLD
        self._unavailable_threshold = unavailable_threshold or DEFAULT_UNAVAILABLE_THRESHOLD
        self._latency_warning_ms = latency_warning_ms or DEFAULT_LATENCY_WARNING_MS
        self._auto_failover = auto_failover
        self._max_records = max_records or DEFAULT_MAX_RECORDS
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._ollama = _ollama_module if ollama_module is _UNSET else ollama_module

        # Health records keyed by model name
        self._records: dict[str, ModelHealthRecord] = {}
        self._lock = threading.Lock()

        # Background thread control
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

        # Load YAML config (overrides constructor defaults)
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file if available."""
        if not self._config_path.exists():
            logger.debug("Model health config not found: %s", self._config_path)
            return

        try:
            with open(self._config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Error reading model health config: %s", e)
            return

        config = data.get("model_health", {})
        if not isinstance(config, dict):
            return

        # Apply YAML values
        if "enabled" in config:
            self._enabled = bool(config["enabled"])
        if "check_interval_seconds" in config:
            val = int(config["check_interval_seconds"])
            if val > 0:
                self._check_interval = val
        if "degraded_threshold" in config:
            val = int(config["degraded_threshold"])
            if val > 0:
                self._degraded_threshold = val
        if "unavailable_threshold" in config:
            val = int(config["unavailable_threshold"])
            if val > 0:
                self._unavailable_threshold = val
        if "latency_warning_ms" in config:
            val = int(config["latency_warning_ms"])
            if val > 0:
                self._latency_warning_ms = val
        if "auto_failover" in config:
            self._auto_failover = bool(config["auto_failover"])
        if "max_records" in config:
            val = int(config["max_records"])
            if val > 0:
                self._max_records = val

        logger.info(
            "Model health config loaded: enabled=%s, interval=%ds, "
            "degraded=%d, unavailable=%d, failover=%s",
            self._enabled, self._check_interval,
            self._degraded_threshold, self._unavailable_threshold,
            self._auto_failover,
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether health monitoring is active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def running(self) -> bool:
        """Whether the background check thread is running."""
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def auto_failover(self) -> bool:
        """Whether automatic failover is enabled."""
        return self._auto_failover

    @auto_failover.setter
    def auto_failover(self, value: bool):
        self._auto_failover = value

    @property
    def check_interval(self) -> int:
        """Seconds between health checks."""
        return self._check_interval

    @property
    def degraded_threshold(self) -> int:
        """Consecutive failures before degraded status."""
        return self._degraded_threshold

    @property
    def unavailable_threshold(self) -> int:
        """Consecutive failures before unavailable status."""
        return self._unavailable_threshold

    # -------------------------------------------------------------------------
    # Background thread management
    # -------------------------------------------------------------------------

    def start(self):
        """Start the background health check thread.

        The thread runs as a daemon and checks all known models
        at the configured interval. Safe to call multiple times.
        """
        if self.running:
            logger.debug("Health monitor already running")
            return

        if not self._enabled:
            logger.info("Health monitor disabled, not starting")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._check_loop,
            name="model-health-monitor",
            daemon=True,
        )
        self._running = True
        self._thread.start()
        logger.info("Model health monitor started (interval=%ds)", self._check_interval)

    def stop(self):
        """Stop the background health check thread.

        Signals the thread to stop and waits for it to finish.
        """
        if not self._running:
            return

        self._stop_event.set()
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self._check_interval + 2)
        self._thread = None
        logger.info("Model health monitor stopped")

    def _check_loop(self):
        """Background loop that runs health checks at configured interval."""
        logger.debug("Health check loop started")
        while not self._stop_event.is_set():
            try:
                self.check_all()
            except Exception as e:
                logger.error("Health check loop error: %s", e)

            # Wait for interval or stop signal
            self._stop_event.wait(timeout=self._check_interval)

        logger.debug("Health check loop exited")

    # -------------------------------------------------------------------------
    # Health check logic
    # -------------------------------------------------------------------------

    def check_all(self) -> dict[str, ModelHealthRecord]:
        """Run health checks on all known models.

        Discovers models via ollama.list() and checks each one.

        Returns:
            Dict mapping model name to health record.
        """
        if self._ollama is None:
            logger.debug("Ollama not available, skipping health check")
            return dict(self._records)

        # Discover models
        models = self._discover_models()
        for model_name in models:
            self.check_model(model_name)

        return dict(self._records)

    def check_model(self, model_name: str) -> ModelHealthRecord:
        """Run a health check on a single model.

        Uses ollama.show() to verify model availability and
        measures response latency.

        Args:
            model_name: Name of the Ollama model to check.

        Returns:
            Updated health record for the model.
        """
        with self._lock:
            record = self._records.get(model_name)
            if record is None:
                record = ModelHealthRecord(model=model_name)
                self._records[model_name] = record

        now = time.time()
        record.last_check = now
        record.check_count += 1

        if self._ollama is None:
            # No ollama module available
            record.consecutive_failures += 1
            record.error_count += 1
            record.last_error = "Ollama module not available"
            self._update_status(record)
            return record

        try:
            start = time.monotonic()
            self._ollama.show(model_name)
            elapsed_ms = (time.monotonic() - start) * 1000

            # Success
            record.latency_ms = elapsed_ms
            record.last_success = now
            record.consecutive_failures = 0
            record.last_error = ""

            if elapsed_ms > self._latency_warning_ms:
                logger.warning(
                    "Model %s health check slow: %.0fms (threshold: %dms)",
                    model_name, elapsed_ms, self._latency_warning_ms,
                )

        except Exception as e:
            # Failure
            record.consecutive_failures += 1
            record.error_count += 1
            record.last_error = str(e)
            logger.debug("Health check failed for %s: %s", model_name, e)

        self._update_status(record)
        return record

    def _update_status(self, record: ModelHealthRecord):
        """Update model status based on consecutive failure count."""
        if record.consecutive_failures >= self._unavailable_threshold:
            record.status = ModelStatus.UNAVAILABLE
        elif record.consecutive_failures >= self._degraded_threshold:
            record.status = ModelStatus.DEGRADED
        elif record.consecutive_failures == 0 and record.check_count > 0:
            record.status = ModelStatus.HEALTHY
        # else: remains UNKNOWN or whatever it was

    def _discover_models(self) -> list[str]:
        """Discover available models via ollama.list().

        Returns:
            List of model names.
        """
        if self._ollama is None:
            return []

        try:
            response = self._ollama.list()
            # ollama-python >= 0.4: ListResponse with .models attribute
            if hasattr(response, "models"):
                models = response.models or []
            elif isinstance(response, dict):
                models = response.get("models", [])
            else:
                models = list(response) if response else []

            names = []
            for m in models:
                name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
                if name:
                    names.append(name)
                else:
                    # Fallback to name attribute
                    n = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
                    if n:
                        names.append(n)

            return names

        except Exception as e:
            logger.debug("Failed to discover models: %s", e)
            return list(self._records.keys())

    # -------------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------------

    def get_health(self, model_name: str) -> ModelHealthRecord | None:
        """Get the health record for a specific model.

        Args:
            model_name: Name of the model.

        Returns:
            ModelHealthRecord or None if not tracked.
        """
        return self._records.get(model_name)

    def get_all_health(self) -> dict[str, ModelHealthRecord]:
        """Get all health records.

        Returns:
            Dict mapping model name to health record.
        """
        with self._lock:
            return dict(self._records)

    def get_status(self, model_name: str) -> ModelStatus:
        """Get the current status of a model.

        Args:
            model_name: Name of the model.

        Returns:
            ModelStatus (UNKNOWN if not tracked).
        """
        record = self._records.get(model_name)
        if record is None:
            return ModelStatus.UNKNOWN
        return record.status

    def is_healthy(self, model_name: str) -> bool:
        """Check if a model is healthy.

        Returns True if the model is healthy or unknown (not yet checked).
        Returns False if degraded or unavailable.
        """
        status = self.get_status(model_name)
        return status in (ModelStatus.HEALTHY, ModelStatus.UNKNOWN)

    def is_available(self, model_name: str) -> bool:
        """Check if a model is available for routing.

        Returns True unless the model is explicitly unavailable.
        """
        return self.get_status(model_name) != ModelStatus.UNAVAILABLE

    def get_healthy_models(self) -> list[str]:
        """Get all models with healthy status.

        Returns:
            List of healthy model names.
        """
        with self._lock:
            return [
                name for name, record in self._records.items()
                if record.status == ModelStatus.HEALTHY
            ]

    def get_degraded_models(self) -> list[str]:
        """Get all models with degraded status.

        Returns:
            List of degraded model names.
        """
        with self._lock:
            return [
                name for name, record in self._records.items()
                if record.status == ModelStatus.DEGRADED
            ]

    def get_unavailable_models(self) -> list[str]:
        """Get all models with unavailable status.

        Returns:
            List of unavailable model names.
        """
        with self._lock:
            return [
                name for name, record in self._records.items()
                if record.status == ModelStatus.UNAVAILABLE
            ]

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def configure(
        self,
        enabled: bool | None = None,
        check_interval: int | None = None,
        degraded_threshold: int | None = None,
        unavailable_threshold: int | None = None,
        latency_warning_ms: int | None = None,
        auto_failover: bool | None = None,
    ):
        """Update monitor configuration.

        Args:
            enabled: Enable/disable monitoring
            check_interval: Seconds between checks
            degraded_threshold: Failures for degraded
            unavailable_threshold: Failures for unavailable
            latency_warning_ms: Latency warning threshold
            auto_failover: Enable automatic failover
        """
        if enabled is not None:
            self._enabled = enabled
        if check_interval is not None and check_interval > 0:
            self._check_interval = check_interval
        if degraded_threshold is not None and degraded_threshold > 0:
            self._degraded_threshold = degraded_threshold
        if unavailable_threshold is not None and unavailable_threshold > 0:
            self._unavailable_threshold = unavailable_threshold
        if latency_warning_ms is not None and latency_warning_ms > 0:
            self._latency_warning_ms = latency_warning_ms
        if auto_failover is not None:
            self._auto_failover = auto_failover

    def get_config(self) -> dict[str, Any]:
        """Return current configuration as a dictionary."""
        return {
            "enabled": self._enabled,
            "running": self.running,
            "check_interval_seconds": self._check_interval,
            "degraded_threshold": self._degraded_threshold,
            "unavailable_threshold": self._unavailable_threshold,
            "latency_warning_ms": self._latency_warning_ms,
            "auto_failover": self._auto_failover,
            "max_records": self._max_records,
            "tracked_models": len(self._records),
            "ollama_available": self._ollama is not None,
        }

    def reset(self):
        """Reset all health records."""
        with self._lock:
            self._records.clear()

    def remove_model(self, model_name: str) -> bool:
        """Remove a model from tracking.

        Args:
            model_name: Name of the model to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if model_name in self._records:
                del self._records[model_name]
                return True
            return False

    def to_dict(self) -> dict[str, Any]:
        """Export full state for debugging/API."""
        config = self.get_config()
        config["records"] = {
            name: record.to_dict()
            for name, record in self._records.items()
        }
        config["summary"] = {
            "healthy": len(self.get_healthy_models()),
            "degraded": len(self.get_degraded_models()),
            "unavailable": len(self.get_unavailable_models()),
        }
        return config


# =============================================================================
# SINGLETON
# =============================================================================

model_health_monitor = ModelHealthMonitor()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_health(model_name: str) -> ModelHealthRecord | None:
    """Shortcut to get a model's health record."""
    return model_health_monitor.get_health(model_name)


def is_healthy(model_name: str) -> bool:
    """Shortcut to check if a model is healthy."""
    return model_health_monitor.is_healthy(model_name)


def is_available(model_name: str) -> bool:
    """Shortcut to check if a model is available."""
    return model_health_monitor.is_available(model_name)


def check_all() -> dict[str, ModelHealthRecord]:
    """Shortcut to run health checks on all models."""
    return model_health_monitor.check_all()
