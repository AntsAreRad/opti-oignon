#!/usr/bin/env python3
"""
INFERENCE TELEMETRY PIPELINE -- OPTI-OIGNON S112
==================================================

Central event bus for inference events. Instruments inference at a
single point and fans out to downstream consumers:

    - LiveMetricsCollector  (S111) -- real-time tok/s, GPU, memory
    - PerformanceMonitor    (S72)  -- execution records, drift, latency
    - SpeculativeDecodingManager (S110) -- acceptance rate tracking
    - InferenceProfiler -- per-request time breakdown, off by default

Every consumer is registered here and nowhere else, and each one is gated
by its own toggle in the consumers section of the configuration.

Architecture:
    TelemetryConfig         -- dataclass from YAML
    InferenceEvent          -- typed event emitted by instrumentation
    TelemetryCollector      -- central bus with buffered dispatch
    get_telemetry()         -- module-level singleton accessor

Hooks for backends:
    on_inference_start(model, messages)
    on_token_generated(request_id, count)
    on_inference_end(request_id, model, tokens_in, tokens_out, latency_ms, ...)

Thread-safe: RLock protects all mutable state.

Author: Leon
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "telemetry.yaml"

# Event types.
EVENT_INFERENCE_START = "inference_start"
EVENT_TOKEN_GENERATED = "token_generated"
EVENT_INFERENCE_END = "inference_end"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TelemetryConfig:
    """Configuration loaded from telemetry.yaml."""

    enabled: bool = True
    buffer_max_size: int = 64
    buffer_flush_interval_ms: int = 250
    consumer_live_metrics: bool = True
    consumer_performance_monitor: bool = True
    consumer_speculative_decoding: bool = True
    # The profiler keeps a per-request ring buffer that a REST route serves.
    # Wiring a collector must never widen the collected surface on its own,
    # so this one is inert until it is explicitly armed. Every path that
    # cannot produce a true here -- key absent, section malformed, file
    # unreadable -- lands on this default and leaves the profiler off.
    consumer_inference_profiler: bool = False
    token_tracking_enabled: bool = True
    token_tracking_max_per_request: int = 8192
    debug_logging: bool = False

    def validate(self) -> list[str]:
        """Return validation errors (empty = valid)."""
        errors: list[str] = []
        if self.buffer_max_size < 1:
            errors.append("buffer_max_size must be >= 1")
        if self.buffer_flush_interval_ms < 0:
            errors.append("buffer_flush_interval_ms must be >= 0")
        if self.token_tracking_max_per_request < 1:
            errors.append("token_tracking_max_per_request must be >= 1")
        return errors


@dataclass
class InferenceEvent:
    """A single telemetry event."""

    event_type: str = ""
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    model: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "data": self.data,
        }


@dataclass
class ActiveRequest:
    """Tracks an in-flight inference request."""

    request_id: str = ""
    model: str = ""
    started_at: float = 0.0
    token_count: int = 0
    token_timestamps: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(path: Path | None = None) -> TelemetryConfig:
    """Load telemetry config from YAML, with defaults for missing keys."""
    p = path or _DEFAULT_CONFIG_PATH
    cfg = TelemetryConfig()
    if not p.is_file():
        logger.debug("No telemetry.yaml found, using defaults")
        return cfg

    try:
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to parse telemetry.yaml: %s", exc)
        return cfg

    cfg.enabled = raw.get("enabled", cfg.enabled)
    cfg.debug_logging = raw.get("debug_logging", cfg.debug_logging)

    buf = raw.get("buffer", {})
    if isinstance(buf, dict):
        cfg.buffer_max_size = buf.get("max_size", cfg.buffer_max_size)
        cfg.buffer_flush_interval_ms = buf.get(
            "flush_interval_ms", cfg.buffer_flush_interval_ms
        )

    consumers = raw.get("consumers", {})
    if isinstance(consumers, dict):
        cfg.consumer_live_metrics = consumers.get(
            "live_metrics", cfg.consumer_live_metrics
        )
        cfg.consumer_performance_monitor = consumers.get(
            "performance_monitor", cfg.consumer_performance_monitor
        )
        cfg.consumer_speculative_decoding = consumers.get(
            "speculative_decoding", cfg.consumer_speculative_decoding
        )
        # Strict identity, not truthiness: this toggle is the profiler's only
        # arming path, so a string, a number or a list must not switch a
        # collector on by being merely truthy. Anything that is not the
        # boolean true falls back to the default.
        cfg.consumer_inference_profiler = (
            consumers.get(
                "inference_profiler", cfg.consumer_inference_profiler
            )
            is True
        )

    tt = raw.get("token_tracking", {})
    if isinstance(tt, dict):
        cfg.token_tracking_enabled = tt.get("enabled", cfg.token_tracking_enabled)
        cfg.token_tracking_max_per_request = tt.get(
            "max_tokens_per_request", cfg.token_tracking_max_per_request
        )

    return cfg


# ---------------------------------------------------------------------------
# Consumer type
# ---------------------------------------------------------------------------

# A consumer is any callable that accepts a list of InferenceEvents.
TelemetryConsumer = Callable[[list[InferenceEvent]], None]


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------


class TelemetryCollector:
    """Central telemetry event bus for inference instrumentation.

    Collects events from inference backends and dispatches them to
    registered consumers (LiveMetricsCollector, PerformanceMonitor,
    SpeculativeDecodingManager) either immediately or after buffering.
    """

    def __init__(
        self,
        config: TelemetryConfig | None = None,
        config_path: Path | None = None,
    ) -> None:
        self._config = config or _load_config(config_path)
        self._lock = threading.RLock()

        # Event buffer for batched dispatch.
        self._buffer: list[InferenceEvent] = []
        self._last_flush: float = time.time()

        # Active inference requests.
        self._active_requests: dict[str, ActiveRequest] = {}

        # Registered consumers.
        self._consumers: list[TelemetryConsumer] = []

        # Statistics.
        self._total_events: int = 0
        self._total_requests: int = 0
        self._total_tokens: int = 0

        # Flush thread (if interval > 0).
        self._flush_thread: threading.Thread | None = None
        self._running = False

        # Auto-register built-in consumers.
        if self._config.enabled:
            self._register_builtin_consumers()

    @property
    def config(self) -> TelemetryConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    # ----- Public hooks (called by inference backends) -----

    def on_inference_start(
        self,
        model: str,
        messages: list[dict] | None = None,
        request_id: str | None = None,
    ) -> str:
        """Called when an inference request begins.

        Args:
            model: Model name being used.
            messages: Chat messages (for metadata only, not stored).
            request_id: Optional pre-assigned ID. Generated if not provided.

        Returns:
            The request_id for this inference.
        """
        if not self._config.enabled:
            return request_id or uuid.uuid4().hex[:12]

        rid = request_id or uuid.uuid4().hex[:12]
        now = time.time()

        with self._lock:
            self._active_requests[rid] = ActiveRequest(
                request_id=rid,
                model=model,
                started_at=now,
            )
            self._total_requests += 1

        event = InferenceEvent(
            event_type=EVENT_INFERENCE_START,
            request_id=rid,
            timestamp=now,
            model=model,
            data={
                "message_count": len(messages) if messages else 0,
            },
        )
        self._emit(event)

        if self._config.debug_logging:
            logger.debug("Telemetry: inference_start model=%s rid=%s", model, rid)

        return rid

    def on_token_generated(
        self,
        request_id: str,
        count: int = 1,
    ) -> None:
        """Called when token(s) are generated during streaming.

        Args:
            request_id: The inference request ID.
            count: Number of tokens generated (usually 1).
        """
        if not self._config.enabled:
            return

        now = time.time()

        with self._lock:
            req = self._active_requests.get(request_id)
            if req:
                req.token_count += count
                if (
                    self._config.token_tracking_enabled
                    and len(req.token_timestamps) < self._config.token_tracking_max_per_request
                ):
                    req.token_timestamps.append(now)
            self._total_tokens += count

        event = InferenceEvent(
            event_type=EVENT_TOKEN_GENERATED,
            request_id=request_id,
            timestamp=now,
            model=req.model if req else "",
            data={"count": count},
        )
        self._emit(event)

    def on_inference_end(
        self,
        request_id: str,
        model: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        quality_score: float = 0.0,
        task_type: str = "chat",
        speculative_data: dict | None = None,
    ) -> None:
        """Called when an inference request completes.

        Args:
            request_id: The inference request ID.
            model: Model name (overrides if provided).
            tokens_in: Prompt token count.
            tokens_out: Generated token count.
            latency_ms: Total latency in milliseconds.
            quality_score: Optional quality metric (0-1).
            task_type: Task category for analytics.
            speculative_data: Optional dict with draft_tokens, accepted, speedup.
        """
        if not self._config.enabled:
            return

        now = time.time()

        with self._lock:
            req = self._active_requests.pop(request_id, None)

        # Calculate latency from tracked start if not provided.
        if req and latency_ms <= 0:
            latency_ms = (now - req.started_at) * 1000.0

        effective_model = model or (req.model if req else "unknown")
        effective_tokens_out = tokens_out or (req.token_count if req else 0)

        event = InferenceEvent(
            event_type=EVENT_INFERENCE_END,
            request_id=request_id,
            timestamp=now,
            model=effective_model,
            data={
                "tokens_in": tokens_in,
                "tokens_out": effective_tokens_out,
                "latency_ms": latency_ms,
                "quality_score": quality_score,
                "task_type": task_type,
                "speculative_data": speculative_data,
            },
        )
        self._emit(event)

        if self._config.debug_logging:
            logger.debug(
                "Telemetry: inference_end model=%s rid=%s tokens=%d latency=%.1fms",
                effective_model,
                request_id,
                effective_tokens_out,
                latency_ms,
            )

    # ----- Consumer management -----

    def register_consumer(self, consumer: TelemetryConsumer) -> None:
        """Register a telemetry consumer."""
        with self._lock:
            self._consumers.append(consumer)

    def unregister_consumer(self, consumer: TelemetryConsumer) -> None:
        """Unregister a telemetry consumer."""
        with self._lock:
            self._consumers = [c for c in self._consumers if c is not consumer]

    # ----- Buffer management -----

    def flush(self) -> int:
        """Force-flush the event buffer. Returns number of events dispatched."""
        with self._lock:
            if not self._buffer:
                return 0
            events = list(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()

        self._dispatch(events)
        return len(events)

    def start_flush_thread(self) -> None:
        """Start the background flush thread."""
        if self._running or self._config.buffer_flush_interval_ms <= 0:
            return
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="telemetry-flush"
        )
        self._flush_thread.start()

    def stop_flush_thread(self) -> None:
        """Stop the background flush thread."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=2.0)
            self._flush_thread = None

    # ----- Statistics -----

    def get_stats(self) -> dict[str, Any]:
        """Get telemetry statistics."""
        with self._lock:
            return {
                "enabled": self._config.enabled,
                "total_events": self._total_events,
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "active_requests": len(self._active_requests),
                "buffer_size": len(self._buffer),
                "consumer_count": len(self._consumers),
            }

    # ----- Internal methods -----

    def _emit(self, event: InferenceEvent) -> None:
        """Add event to buffer, flush if needed."""
        with self._lock:
            self._total_events += 1
            self._buffer.append(event)

            # Immediate flush if buffer full or flush interval is 0.
            should_flush = (
                len(self._buffer) >= self._config.buffer_max_size
                or self._config.buffer_flush_interval_ms <= 0
            )

        if should_flush:
            self.flush()

    def _dispatch(self, events: list[InferenceEvent]) -> None:
        """Dispatch events to all registered consumers."""
        with self._lock:
            consumers = list(self._consumers)

        for consumer in consumers:
            try:
                consumer(events)
            except Exception as exc:
                logger.warning(
                    "Telemetry consumer %s failed: %s",
                    getattr(consumer, "__name__", repr(consumer)),
                    exc,
                )

    def _flush_loop(self) -> None:
        """Background thread that periodically flushes the buffer."""
        interval_s = self._config.buffer_flush_interval_ms / 1000.0
        while self._running:
            time.sleep(interval_s)
            if self._running:
                self.flush()

    def _register_builtin_consumers(self) -> None:
        """Register the built-in consumers based on config toggles."""
        if self._config.consumer_live_metrics:
            consumer = _create_live_metrics_consumer()
            if consumer:
                self._consumers.append(consumer)
                logger.debug("Telemetry: registered live_metrics consumer")

        if self._config.consumer_performance_monitor:
            consumer = _create_performance_monitor_consumer()
            if consumer:
                self._consumers.append(consumer)
                logger.debug("Telemetry: registered performance_monitor consumer")

        if self._config.consumer_speculative_decoding:
            consumer = _create_speculative_decoding_consumer()
            if consumer:
                self._consumers.append(consumer)
                logger.debug("Telemetry: registered speculative_decoding consumer")

        if self._config.consumer_inference_profiler:
            consumer = _create_inference_profiler_consumer()
            if consumer:
                self._consumers.append(consumer)
                logger.debug("Telemetry: registered inference_profiler consumer")

    def shutdown(self) -> None:
        """Flush remaining events and stop background thread."""
        self.stop_flush_thread()
        self.flush()
        logger.info("TelemetryCollector shutdown complete")


# ---------------------------------------------------------------------------
# Built-in consumer factories
# ---------------------------------------------------------------------------


def _create_live_metrics_consumer() -> TelemetryConsumer | None:
    """Create a consumer that feeds LiveMetricsCollector."""
    try:
        from opti_oignon.live_metrics import get_live_metrics

        collector = get_live_metrics()
    except Exception:
        return None

    if collector is None:
        return None

    def consumer(events: list[InferenceEvent]) -> None:
        for ev in events:
            if ev.event_type == EVENT_INFERENCE_START:
                collector.start_generation(model=ev.model)
            elif ev.event_type == EVENT_TOKEN_GENERATED:
                collector.record_token(count=ev.data.get("count", 1))
            elif ev.event_type == EVENT_INFERENCE_END:
                latency_ms = ev.data.get("latency_ms", 0.0)
                collector.end_generation(
                    eval_time_ms=latency_ms,
                )

    consumer.__name__ = "live_metrics_consumer"  # type: ignore[attr-defined]
    return consumer


def _create_performance_monitor_consumer() -> TelemetryConsumer | None:
    """Create a consumer that feeds PerformanceMonitor."""
    try:
        from opti_oignon.performance_monitor import performance_monitor
    except Exception:
        return None

    if performance_monitor is None:
        return None

    def consumer(events: list[InferenceEvent]) -> None:
        for ev in events:
            if ev.event_type == EVENT_INFERENCE_END:
                performance_monitor.record_execution(
                    model=ev.model,
                    task_type=ev.data.get("task_type", "chat"),
                    latency_ms=ev.data.get("latency_ms", 0.0),
                    tokens_in=ev.data.get("tokens_in", 0),
                    tokens_out=ev.data.get("tokens_out", 0),
                    quality_score=ev.data.get("quality_score", 0.0),
                    timestamp=ev.timestamp,
                )

    consumer.__name__ = "performance_monitor_consumer"  # type: ignore[attr-defined]
    return consumer


def _create_speculative_decoding_consumer() -> TelemetryConsumer | None:
    """Create a consumer that feeds SpeculativeDecodingManager."""
    try:
        from opti_oignon.speculative_decoding import get_speculative_decoding_manager

        manager = get_speculative_decoding_manager()
    except Exception:
        return None

    if manager is None:
        return None

    def consumer(events: list[InferenceEvent]) -> None:
        for ev in events:
            if ev.event_type == EVENT_INFERENCE_END:
                spec_data = ev.data.get("speculative_data")
                if spec_data and isinstance(spec_data, dict):
                    draft_tokens = spec_data.get("draft_tokens", 0)
                    accepted = spec_data.get("accepted", 0)
                    speedup = spec_data.get("speedup", 1.0)
                    if draft_tokens > 0:
                        manager.record_acceptance(
                            draft_tokens=draft_tokens,
                            accepted=accepted,
                            speedup=speedup,
                            request_id=ev.request_id,
                        )

    consumer.__name__ = "speculative_decoding_consumer"  # type: ignore[attr-defined]
    return consumer


def _create_inference_profiler_consumer() -> TelemetryConsumer | None:
    """Create a consumer that feeds InferenceProfiler.

    The bus is the only party that subscribes the profiler. The profiler
    itself never reaches back here: an accessor that called into the bus
    while the bus was building its consumers would re-enter the singleton
    locks, so the dependency runs one way only, exactly as it does for the
    sibling consumers above.
    """
    try:
        from opti_oignon.inference_profiler import get_profiler

        profiler = get_profiler()
    except Exception:
        return None

    if profiler is None:
        return None

    # The bound method already carries the consumer protocol and its own
    # __name__, which _dispatch uses when reporting a consumer failure.
    return profiler.consume


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector: TelemetryCollector | None = None
_collector_lock = threading.Lock()


def get_telemetry(
    config_path: Path | None = None,
) -> TelemetryCollector:
    """Get or create the singleton TelemetryCollector."""
    global _collector
    if _collector is not None:
        return _collector
    with _collector_lock:
        if _collector is not None:
            return _collector
        _collector = TelemetryCollector(config_path=config_path)
        return _collector


def reset_telemetry() -> None:
    """Reset the singleton (for testing)."""
    global _collector
    with _collector_lock:
        if _collector is not None:
            _collector.shutdown()
        _collector = None
