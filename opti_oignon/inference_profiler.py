#!/usr/bin/env python3
"""
INFERENCE PROFILER -- OPTI-OIGNON S113
========================================

Per-request detailed profiling that hooks into the telemetry pipeline
as a consumer.  Tracks time breakdown (prompt eval, token generation,
overhead) and provides aggregated statistics per model.

Architecture:
    InferenceProfile      -- dataclass for a single request breakdown
    InferenceProfiler     -- consumer + ring-buffer of recent profiles
    get_profiler()        -- module-level singleton accessor

Aggregation: avg / p50 / p95 / p99 by model, computed on demand from
the ring buffer of recent profiles.

Thread-safe: RLock protects all mutable state.

Author: Leon
"""

import collections
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_PROFILES = 500


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class InferenceProfile:
    """Detailed time breakdown for a single inference request."""

    request_id: str = ""
    model: str = ""
    timestamp: float = 0.0

    # Time components (milliseconds).
    total_ms: float = 0.0
    prompt_eval_ms: float = 0.0
    token_gen_ms: float = 0.0
    overhead_ms: float = 0.0

    # Token counts.
    tokens_in: int = 0
    tokens_out: int = 0

    # Derived.
    tok_per_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "timestamp": self.timestamp,
            "total_ms": round(self.total_ms, 2),
            "prompt_eval_ms": round(self.prompt_eval_ms, 2),
            "token_gen_ms": round(self.token_gen_ms, 2),
            "overhead_ms": round(self.overhead_ms, 2),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tok_per_sec": round(self.tok_per_sec, 2),
        }


@dataclass
class _ActiveTrace:
    """Internal: tracks an in-flight request for profiling."""

    request_id: str = ""
    model: str = ""
    start_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0
    token_count: int = 0


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile from a pre-sorted list (0-100)."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


# ---------------------------------------------------------------------------
# InferenceProfiler
# ---------------------------------------------------------------------------


class InferenceProfiler:
    """Per-request inference profiler.

    Implements the telemetry consumer protocol and records detailed time
    breakdowns for each inference request.  Maintains a ring buffer of
    recent profiles and computes per-model aggregated statistics on
    demand.

    The profiler does not subscribe itself.  The telemetry bus registers
    it, and only when its consumer toggle is explicitly armed.
    """

    def __init__(self, max_profiles: int = DEFAULT_MAX_PROFILES) -> None:
        self._lock = threading.RLock()
        self._max_profiles = max(10, max_profiles)

        # Ring buffer of completed profiles.
        self._profiles: collections.deque[InferenceProfile] = collections.deque(
            maxlen=self._max_profiles
        )

        # Active traces (request_id -> _ActiveTrace).
        self._traces: dict[str, _ActiveTrace] = {}

        # Counters.
        self._total_profiled: int = 0

        # Set __name__ on bound method for telemetry dashboard display.
        self.consume.__func__.__name__ = "inference_profiler_consumer"  # type: ignore[attr-defined]

    # ----- Telemetry consumer interface -----

    def consume(self, events: list) -> None:
        """Telemetry consumer callback.

        Accepts a list of InferenceEvent objects from the telemetry
        pipeline and tracks start / token / end events for profiling.
        """
        for ev in events:
            etype = getattr(ev, "event_type", "") or ""
            if etype == "inference_start":
                self._on_start(ev)
            elif etype == "token_generated":
                self._on_token(ev)
            elif etype == "inference_end":
                self._on_end(ev)

    # ----- Event handlers -----

    def _on_start(self, ev: Any) -> None:
        rid = getattr(ev, "request_id", "") or ""
        if not rid:
            return
        with self._lock:
            self._traces[rid] = _ActiveTrace(
                request_id=rid,
                model=getattr(ev, "model", "") or "",
                start_ts=getattr(ev, "timestamp", 0.0) or time.time(),
            )

    def _on_token(self, ev: Any) -> None:
        rid = getattr(ev, "request_id", "") or ""
        if not rid:
            return
        now = getattr(ev, "timestamp", 0.0) or time.time()
        with self._lock:
            trace = self._traces.get(rid)
            if not trace:
                return
            if trace.first_token_ts <= 0:
                trace.first_token_ts = now
            trace.last_token_ts = now
            # S193 PRF-04: null-guard ev.data like _on_end (a None payload
            # would otherwise raise AttributeError inside the consumer loop).
            trace.token_count += (getattr(ev, "data", {}) or {}).get("count", 1)

    def _on_end(self, ev: Any) -> None:
        rid = getattr(ev, "request_id", "") or ""
        if not rid:
            return
        data = getattr(ev, "data", {}) or {}
        end_ts = getattr(ev, "timestamp", 0.0) or time.time()

        with self._lock:
            trace = self._traces.pop(rid, None)

        if trace is None:
            # End event without a matching start -- build a minimal profile.
            total_ms = data.get("latency_ms", 0.0)
            tokens_out = data.get("tokens_out", 0)
            tok_s = (tokens_out / (total_ms / 1000.0)) if total_ms > 0 else 0.0
            profile = InferenceProfile(
                request_id=rid,
                model=getattr(ev, "model", "") or "",
                timestamp=end_ts,
                total_ms=total_ms,
                tokens_in=data.get("tokens_in", 0),
                tokens_out=tokens_out,
                tok_per_sec=tok_s,
            )
        else:
            total_ms = data.get("latency_ms", 0.0)
            if total_ms <= 0 and trace.start_ts > 0:
                total_ms = (end_ts - trace.start_ts) * 1000.0

            # Prompt eval = time from start to first token.
            prompt_eval_ms = 0.0
            if trace.first_token_ts > 0 and trace.start_ts > 0:
                prompt_eval_ms = (trace.first_token_ts - trace.start_ts) * 1000.0

            # Token generation = time from first token to last token.
            token_gen_ms = 0.0
            if trace.first_token_ts > 0 and trace.last_token_ts > 0:
                token_gen_ms = (trace.last_token_ts - trace.first_token_ts) * 1000.0

            # Overhead = total - prompt_eval - token_gen.
            overhead_ms = max(0.0, total_ms - prompt_eval_ms - token_gen_ms)

            tokens_out = data.get("tokens_out", 0) or trace.token_count
            tokens_in = data.get("tokens_in", 0)
            tok_s = (tokens_out / (total_ms / 1000.0)) if total_ms > 0 else 0.0

            profile = InferenceProfile(
                request_id=rid,
                model=trace.model or getattr(ev, "model", ""),
                timestamp=end_ts,
                total_ms=total_ms,
                prompt_eval_ms=prompt_eval_ms,
                token_gen_ms=token_gen_ms,
                overhead_ms=overhead_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tok_per_sec=tok_s,
            )

        with self._lock:
            self._profiles.append(profile)
            self._total_profiled += 1

    # ----- Public API -----

    def get_recent(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the most recent N profiles as dicts."""
        with self._lock:
            items = list(self._profiles)
        # Most recent first.
        items.reverse()
        return [p.to_dict() for p in items[:n]]

    def get_summary(self) -> list[dict[str, Any]]:
        """Return aggregated profiling stats per model."""
        with self._lock:
            items = list(self._profiles)

        if not items:
            return []

        # Group by model.
        by_model: dict[str, list[InferenceProfile]] = {}
        for p in items:
            by_model.setdefault(p.model, []).append(p)

        summaries: list[dict[str, Any]] = []
        for model, profiles in sorted(by_model.items()):
            totals = sorted([p.total_ms for p in profiles])
            n = len(profiles)
            summaries.append({
                "model": model,
                "request_count": n,
                "avg_total_ms": round(sum(totals) / n, 2) if n else 0,
                "p50_total_ms": round(_percentile(totals, 50), 2),
                "p95_total_ms": round(_percentile(totals, 95), 2),
                "p99_total_ms": round(_percentile(totals, 99), 2),
                "avg_prompt_eval_ms": round(
                    sum(p.prompt_eval_ms for p in profiles) / n, 2
                ) if n else 0,
                "avg_token_gen_ms": round(
                    sum(p.token_gen_ms for p in profiles) / n, 2
                ) if n else 0,
                "avg_overhead_ms": round(
                    sum(p.overhead_ms for p in profiles) / n, 2
                ) if n else 0,
                "avg_tok_per_sec": round(
                    sum(p.tok_per_sec for p in profiles) / n, 2
                ) if n else 0,
            })

        return summaries

    @property
    def total_profiled(self) -> int:
        with self._lock:
            return self._total_profiled

    def get_stats(self) -> dict[str, Any]:
        """Quick overview stats."""
        with self._lock:
            return {
                "total_profiled": self._total_profiled,
                "buffer_size": len(self._profiles),
                "buffer_max": self._max_profiles,
                "active_traces": len(self._traces),
            }

    def shutdown(self) -> None:
        """Cleanup."""
        with self._lock:
            self._traces.clear()
        logger.info("InferenceProfiler shutdown complete")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_profiler: InferenceProfiler | None = None
_profiler_lock = threading.Lock()


def get_profiler(
    max_profiles: int = DEFAULT_MAX_PROFILES,
) -> InferenceProfiler:
    """Get or create the singleton InferenceProfiler.

    This is a pure accessor. It does NOT subscribe the profiler to the
    telemetry bus, and it must never be made to: the bus owns the consumer
    registry and consults the configuration toggle before wiring anything.
    Subscribing from here would arm a per-request collector as a side
    effect of merely reading it -- the REST route is the only caller -- and
    would bypass that toggle entirely.
    """
    global _profiler
    if _profiler is not None:
        return _profiler
    with _profiler_lock:
        if _profiler is not None:
            return _profiler
        _profiler = InferenceProfiler(max_profiles=max_profiles)
        return _profiler


def reset_profiler() -> None:
    """Reset the singleton (for testing)."""
    global _profiler
    with _profiler_lock:
        if _profiler is not None:
            _profiler.shutdown()
        _profiler = None
