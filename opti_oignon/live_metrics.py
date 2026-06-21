#!/usr/bin/env python3
"""
LIVE PERFORMANCE METRICS -- OPTI-OIGNON S111
=============================================

Real-time performance metrics collection during active inference.
Samples tokens/sec, latency, GPU utilization, and memory usage,
then publishes snapshots for the frontend overlay.

Architecture:
    MetricsSample        -- single point-in-time measurement
    LiveMetricsCollector -- background sampler with rolling window
    get_live_metrics()   -- module-level singleton accessor

GPU utilization is sampled via nvidia-smi (optional, graceful
degradation when no NVIDIA GPU or nvidia-smi is unavailable).

Thread-safe: all public methods use RLock for safe concurrent access.
"""

import collections
import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "live_metrics.yaml"

# Default rolling window: 60 seconds of history at 500ms sampling.
_DEFAULT_WINDOW_SECONDS = 60
_DEFAULT_SAMPLE_INTERVAL_MS = 500
_DEFAULT_ROLLING_SPEED_WINDOW_S = 5.0

# nvidia-smi query for GPU utilization and memory.
_NVIDIA_SMI_QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MetricsSample:
    """Single point-in-time metrics snapshot."""

    timestamp: float = 0.0

    # Inference metrics (updated per-token or per-request).
    tokens_per_second: float = 0.0
    prompt_eval_time_ms: float = 0.0
    eval_time_ms: float = 0.0
    total_tokens: int = 0
    pending_tokens: int = 0

    # System metrics.
    gpu_utilization_pct: float = -1.0  # -1 means unavailable
    gpu_memory_used_mb: float = -1.0
    gpu_memory_total_mb: float = -1.0
    gpu_temperature_c: float = -1.0
    system_memory_used_mb: float = 0.0
    system_memory_total_mb: float = 0.0

    # Inference state.
    is_generating: bool = False
    active_model: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict for API/WS transport."""
        return {
            "timestamp": round(self.timestamp, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "prompt_eval_time_ms": round(self.prompt_eval_time_ms, 2),
            "eval_time_ms": round(self.eval_time_ms, 2),
            "total_tokens": self.total_tokens,
            "pending_tokens": self.pending_tokens,
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 1),
            "gpu_memory_used_mb": round(self.gpu_memory_used_mb, 1),
            "gpu_memory_total_mb": round(self.gpu_memory_total_mb, 1),
            "gpu_temperature_c": round(self.gpu_temperature_c, 1),
            "system_memory_used_mb": round(self.system_memory_used_mb, 1),
            "system_memory_total_mb": round(self.system_memory_total_mb, 1),
            "is_generating": self.is_generating,
            "active_model": self.active_model,
        }


@dataclass
class LiveMetricsConfig:
    """Configuration for the live metrics collector."""

    enabled: bool = True
    sample_interval_ms: int = _DEFAULT_SAMPLE_INTERVAL_MS
    window_seconds: int = _DEFAULT_WINDOW_SECONDS
    rolling_speed_window_s: float = _DEFAULT_ROLLING_SPEED_WINDOW_S
    gpu_monitoring: bool = True

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "enabled": self.enabled,
            "sample_interval_ms": self.sample_interval_ms,
            "window_seconds": self.window_seconds,
            "rolling_speed_window_s": self.rolling_speed_window_s,
            "gpu_monitoring": self.gpu_monitoring,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LiveMetricsConfig":
        """Create from dict, ignoring unknown keys."""
        known = {
            "enabled", "sample_interval_ms", "window_seconds",
            "rolling_speed_window_s", "gpu_monitoring",
        }
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def _nvidia_smi_available() -> bool:
    """Check if nvidia-smi is on PATH."""
    return shutil.which("nvidia-smi") is not None


def _query_gpu_metrics() -> dict:
    """Query GPU metrics via nvidia-smi.

    Returns a dict with keys: gpu_utilization_pct, gpu_memory_used_mb,
    gpu_memory_total_mb, gpu_temperature_c. All values are -1.0 on
    failure.
    """
    result = {
        "gpu_utilization_pct": -1.0,
        "gpu_memory_used_mb": -1.0,
        "gpu_memory_total_mb": -1.0,
        "gpu_temperature_c": -1.0,
    }

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + _NVIDIA_SMI_QUERY,
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return result

        line = proc.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            result["gpu_utilization_pct"] = _safe_float(parts[0])
            result["gpu_memory_used_mb"] = _safe_float(parts[1])
            result["gpu_memory_total_mb"] = _safe_float(parts[2])
            result["gpu_temperature_c"] = _safe_float(parts[3])

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        logger.debug("nvidia-smi query failed: %s", exc)

    return result


def _safe_float(val: str) -> float:
    """Parse a string to float, returning -1.0 on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return -1.0


def _get_system_memory() -> tuple[float, float]:
    """Get system memory usage (used_mb, total_mb).

    Reads from /proc/meminfo on Linux, falls back to psutil if
    available, or returns (0, 0).
    """
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.is_file():
        try:
            text = meminfo_path.read_text(encoding="utf-8")
            mem_total = 0.0
            mem_available = 0.0
            for line in text.splitlines():
                if line.startswith("MemTotal:"):
                    mem_total = float(line.split()[1]) / 1024.0  # kB to MB
                elif line.startswith("MemAvailable:"):
                    mem_available = float(line.split()[1]) / 1024.0
            used = mem_total - mem_available
            return (round(used, 1), round(mem_total, 1))
        except Exception:
            pass

    # Fallback: try psutil.
    try:
        import psutil
        vm = psutil.virtual_memory()
        return (vm.used / (1024 * 1024), vm.total / (1024 * 1024))
    except ImportError:
        pass

    return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Live Metrics Collector
# ---------------------------------------------------------------------------

class LiveMetricsCollector:
    """Collects and stores real-time inference performance metrics.

    Maintains a rolling deque of MetricsSample objects, sampled at
    a configurable interval. Inference code calls record_token() and
    start_generation() / end_generation() to feed data into the
    collector. A background thread periodically samples GPU and memory.

    Thread-safe via RLock.
    """

    def __init__(self, config: LiveMetricsConfig | None = None):
        self._config = config or LiveMetricsConfig()
        self._lock = threading.RLock()

        # Rolling history buffer.
        max_samples = max(
            1,
            int(
                self._config.window_seconds
                * 1000
                / max(self._config.sample_interval_ms, 50)
            ),
        )
        self._history: collections.deque[MetricsSample] = collections.deque(
            maxlen=max_samples,
        )

        # Current generation state.
        self._is_generating = False
        self._active_model = ""
        self._generation_start: float = 0.0
        self._generation_tokens: int = 0
        self._total_tokens_all_time: int = 0
        self._prompt_eval_time_ms: float = 0.0
        self._eval_time_ms: float = 0.0

        # Token timestamps for rolling speed calculation.
        self._token_timestamps: collections.deque[float] = collections.deque(
            maxlen=500,
        )

        # GPU availability check (done once).
        self._gpu_available = (
            self._config.gpu_monitoring and _nvidia_smi_available()
        )

        # Background sampler thread.
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def config(self) -> LiveMetricsConfig:
        """Current configuration (copy)."""
        with self._lock:
            return LiveMetricsConfig.from_dict(self._config.to_dict())

    def start(self) -> None:
        """Start the background metrics sampling thread."""
        with self._lock:
            if self._running:
                return
            self._running = True

        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
            name="live-metrics-sampler",
        )
        self._thread.start()
        logger.info(
            "Live metrics collector started (interval=%dms, window=%ds, gpu=%s)",
            self._config.sample_interval_ms,
            self._config.window_seconds,
            self._gpu_available,
        )

    def stop(self) -> None:
        """Stop the background sampling thread."""
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("Live metrics collector stopped")

    @property
    def is_running(self) -> bool:
        """Whether the sampler is actively running."""
        return self._running

    # -- Inference event hooks --

    def start_generation(self, model: str = "") -> None:
        """Signal that a new generation has started."""
        with self._lock:
            self._is_generating = True
            self._active_model = model
            self._generation_start = time.time()
            self._generation_tokens = 0
            self._prompt_eval_time_ms = 0.0
            self._eval_time_ms = 0.0

    def end_generation(
        self,
        prompt_eval_time_ms: float = 0.0,
        eval_time_ms: float = 0.0,
    ) -> None:
        """Signal that the current generation has ended."""
        with self._lock:
            self._is_generating = False
            self._prompt_eval_time_ms = prompt_eval_time_ms
            self._eval_time_ms = eval_time_ms

    def record_token(self, count: int = 1) -> None:
        """Record that one or more tokens were generated."""
        now = time.time()
        with self._lock:
            self._generation_tokens += count
            self._total_tokens_all_time += count
            for _ in range(count):
                self._token_timestamps.append(now)

    def record_timing(
        self,
        prompt_eval_time_ms: float = 0.0,
        eval_time_ms: float = 0.0,
    ) -> None:
        """Update timing metadata for the current generation."""
        with self._lock:
            if prompt_eval_time_ms > 0:
                self._prompt_eval_time_ms = prompt_eval_time_ms
            if eval_time_ms > 0:
                self._eval_time_ms = eval_time_ms

    # -- Snapshot API --

    def current_snapshot(self) -> MetricsSample:
        """Get the most recent metrics sample."""
        with self._lock:
            if self._history:
                return self._history[-1]
            return self._take_sample()

    def get_history(self, seconds: int | None = None) -> list[dict]:
        """Get metrics history as a list of dicts.

        Args:
            seconds: If provided, return only the last N seconds of
                history. Otherwise return the full buffer.

        Returns:
            List of MetricsSample dicts, oldest first.
        """
        with self._lock:
            samples = list(self._history)

        if seconds is not None and seconds > 0:
            cutoff = time.time() - seconds
            samples = [s for s in samples if s.timestamp >= cutoff]

        return [s.to_dict() for s in samples]

    def get_status(self) -> dict:
        """Get collector status and config."""
        with self._lock:
            return {
                "running": self._running,
                "config": self._config.to_dict(),
                "gpu_available": self._gpu_available,
                "history_size": len(self._history),
                "total_tokens_all_time": self._total_tokens_all_time,
                "is_generating": self._is_generating,
                "active_model": self._active_model,
            }

    # -- Internal --

    def _sample_loop(self) -> None:
        """Background thread: periodically take a metrics sample."""
        interval = max(self._config.sample_interval_ms, 50) / 1000.0
        while self._running:
            try:
                sample = self._take_sample()
                with self._lock:
                    self._history.append(sample)
            except Exception as exc:
                logger.debug("Metrics sample error: %s", exc)
            time.sleep(interval)

    def _should_sample_gpu(self, now: float) -> bool:
        """Whether to spawn nvidia-smi this cycle (S193 LMT-01).

        True while generating or within the rolling-speed window of the last
        recorded token; otherwise the GPU subprocess is skipped and the last
        known values are carried forward by the caller.
        """
        with self._lock:
            if self._is_generating:
                return True
            if self._token_timestamps:
                return (now - self._token_timestamps[-1]) <= self._config.rolling_speed_window_s
        return False

    def _take_sample(self) -> MetricsSample:
        """Take a single metrics sample."""
        now = time.time()
        sample = MetricsSample(timestamp=now)

        with self._lock:
            sample.is_generating = self._is_generating
            sample.active_model = self._active_model
            sample.total_tokens = self._total_tokens_all_time
            sample.pending_tokens = self._generation_tokens
            sample.prompt_eval_time_ms = self._prompt_eval_time_ms
            sample.eval_time_ms = self._eval_time_ms

            # Calculate rolling tokens/sec over the configured window.
            window = self._config.rolling_speed_window_s
            cutoff = now - window
            recent = [
                t for t in self._token_timestamps if t >= cutoff
            ]
            if len(recent) >= 2:
                span = recent[-1] - recent[0]
                if span > 0:
                    sample.tokens_per_second = (len(recent) - 1) / span
            elif len(recent) == 1 and self._is_generating:
                # Single token, estimate from generation start.
                elapsed = now - self._generation_start
                if elapsed > 0:
                    sample.tokens_per_second = 1.0 / elapsed

        # GPU metrics (outside lock to avoid blocking).
        # S193 LMT-01: nvidia-smi is a subprocess spawn. The collector never
        # stops once started (any /api/metrics hit or a telemetry generation
        # event auto-starts it), so querying every interval forked nvidia-smi
        # ~2x/sec forever, even at idle. Query the GPU only while generating
        # or within the rolling window of the last token; otherwise carry the
        # last known values forward (this is an inference overlay).
        if self._gpu_available and self._should_sample_gpu(now):
            gpu = _query_gpu_metrics()
            sample.gpu_utilization_pct = gpu["gpu_utilization_pct"]
            sample.gpu_memory_used_mb = gpu["gpu_memory_used_mb"]
            sample.gpu_memory_total_mb = gpu["gpu_memory_total_mb"]
            sample.gpu_temperature_c = gpu["gpu_temperature_c"]
        elif self._gpu_available:
            with self._lock:
                prev = self._history[-1] if self._history else None
            if prev is not None:
                sample.gpu_utilization_pct = prev.gpu_utilization_pct
                sample.gpu_memory_used_mb = prev.gpu_memory_used_mb
                sample.gpu_memory_total_mb = prev.gpu_memory_total_mb
                sample.gpu_temperature_c = prev.gpu_temperature_c

        # System memory.
        mem_used, mem_total = _get_system_memory()
        sample.system_memory_used_mb = mem_used
        sample.system_memory_total_mb = mem_total

        return sample


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector: LiveMetricsCollector | None = None
_init_lock = threading.Lock()


def get_live_metrics(
    config_path: str | None = None,
    auto_start: bool = True,
) -> LiveMetricsCollector:
    """Get or create the module-level singleton collector.

    Args:
        config_path: Optional path to live_metrics.yaml.
        auto_start: If True, start the background sampler automatically.

    Returns:
        The singleton LiveMetricsCollector instance.
    """
    global _collector
    if _collector is not None:
        return _collector
    with _init_lock:
        if _collector is not None:
            return _collector

        config = _load_config(config_path)
        _collector = LiveMetricsCollector(config=config)

        if auto_start and config.enabled:
            _collector.start()

        return _collector


def reset_live_metrics() -> None:
    """Reset the singleton (for testing)."""
    global _collector
    with _init_lock:
        if _collector is not None:
            _collector.stop()
        _collector = None


def _load_config(config_path: str | None = None) -> LiveMetricsConfig:
    """Load live metrics configuration from YAML."""
    p = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not p.is_file():
        logger.debug("No live_metrics.yaml found at %s, using defaults", p)
        return LiveMetricsConfig()

    try:
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to load live_metrics.yaml: %s", exc)
        return LiveMetricsConfig()

    section = raw.get("live_metrics", {})
    if isinstance(section, dict):
        return LiveMetricsConfig.from_dict(section)
    return LiveMetricsConfig()
