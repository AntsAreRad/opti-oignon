#!/usr/bin/env python3
"""
SSE / WebSocket backpressure buffer (S159).

Provides a bounded event buffer that sits between a producer (LLM token
generator) and a consumer (WebSocket or SSE client).  When the consumer
falls behind, the buffer drops the oldest events and logs a warning.

Key features:
- Configurable max size (default 100 events)
- Drop-oldest eviction policy with per-client warning counter
- Client slowness detection based on configurable high-water mark
- Graceful disconnect on idle timeout
- Thread-safe for synchronous producers, async-safe for async consumers

Usage::

    buf = BackpressureBuffer(max_size=100, slow_threshold=0.8)
    buf.push({"type": "token", "content": "Hello"})
    event = await buf.pop(timeout=30.0)
"""

import asyncio
import collections
import logging
import threading
import time
from dataclasses import dataclass, field

# Hardcoded, never overridable
checkpoint_before_apply = True

logger = logging.getLogger(__name__)


@dataclass
class BufferStats:
    """Cumulative statistics for a single backpressure buffer."""

    pushed: int = 0
    popped: int = 0
    dropped: int = 0
    slow_warnings: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "pushed": self.pushed,
            "popped": self.popped,
            "dropped": self.dropped,
            "slow_warnings": self.slow_warnings,
            "created_at": self.created_at,
        }


class BackpressureBuffer:
    """Bounded event buffer with drop-oldest eviction.

    Parameters
    ----------
    max_size : int
        Maximum number of events the buffer can hold before eviction.
    slow_threshold : float
        Fraction of ``max_size`` at which the client is considered slow
        (triggers a warning log).  Must be in (0.0, 1.0].
    idle_timeout : float
        Seconds of inactivity on the consumer side before the buffer
        marks the client as timed-out.
    """

    def __init__(
        self,
        max_size: int = 100,
        slow_threshold: float = 0.8,
        idle_timeout: float = 60.0,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if not (0.0 < slow_threshold <= 1.0):
            raise ValueError("slow_threshold must be in (0.0, 1.0]")
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be > 0")

        self._max_size = max_size
        self._slow_threshold = slow_threshold
        self._idle_timeout = idle_timeout
        self._buffer: collections.deque = collections.deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._event = asyncio.Event()
        self._closed = False
        self._last_pop_time = time.time()
        self.stats = BufferStats()

    # -- properties --

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def idle_timeout(self) -> float:
        return self._idle_timeout

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_slow(self) -> bool:
        """True when the buffer fill level exceeds the slow threshold."""
        with self._lock:
            return len(self._buffer) >= int(self._max_size * self._slow_threshold)

    @property
    def is_idle_timed_out(self) -> bool:
        """True when the consumer has not popped for longer than idle_timeout."""
        return (time.time() - self._last_pop_time) > self._idle_timeout

    # -- producer side (sync, called from generator threads) --

    def push(self, event: dict) -> bool:
        """Add an event to the buffer.

        If the buffer is full the oldest event is dropped.
        Returns False if the buffer is closed.
        """
        if self._closed:
            return False

        dropped = False
        with self._lock:
            if len(self._buffer) >= self._max_size:
                self._buffer.popleft()
                self.stats.dropped += 1
                dropped = True
            self._buffer.append(event)
            self.stats.pushed += 1
            fill = len(self._buffer)

        if dropped:
            logger.warning(
                "Backpressure buffer full (%d/%d) -- dropped oldest event "
                "(total dropped: %d)",
                fill,
                self._max_size,
                self.stats.dropped,
            )

        if fill >= int(self._max_size * self._slow_threshold):
            self.stats.slow_warnings += 1
            if self.stats.slow_warnings <= 5 or self.stats.slow_warnings % 50 == 0:
                logger.warning(
                    "Slow client detected: buffer at %d/%d (%.0f%%)",
                    fill,
                    self._max_size,
                    100 * fill / self._max_size,
                )

        # Signal async consumers
        self._event.set()
        return True

    def push_many(self, events: list[dict]) -> int:
        """Push multiple events.  Returns the number actually buffered."""
        count = 0
        for ev in events:
            if self.push(ev):
                count += 1
            else:
                break
        return count

    # -- consumer side (async) --

    async def pop(self, timeout: float | None = None) -> dict | None:
        """Wait for and return the next event.

        Returns None if the buffer is closed or the wait times out.
        The effective timeout is the smaller of the explicit *timeout*
        parameter and the buffer's ``idle_timeout``.
        """
        effective_timeout = self._idle_timeout
        if timeout is not None:
            effective_timeout = min(timeout, self._idle_timeout)

        # Fast path: event already available
        with self._lock:
            if self._buffer:
                self._last_pop_time = time.time()
                self.stats.popped += 1
                return self._buffer.popleft()

        if self._closed:
            return None

        # Wait for signal
        self._event.clear()
        try:
            await asyncio.wait_for(self._event.wait(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            return None

        with self._lock:
            if self._buffer:
                self._last_pop_time = time.time()
                self.stats.popped += 1
                return self._buffer.popleft()
        return None

    async def drain(self, timeout: float = 0.0) -> list[dict]:
        """Pop all currently buffered events at once.

        If the buffer is empty and *timeout* > 0, waits up to *timeout*
        seconds for at least one event before returning.
        """
        with self._lock:
            if self._buffer:
                items = list(self._buffer)
                self._buffer.clear()
                self.stats.popped += len(items)
                self._last_pop_time = time.time()
                return items

        if timeout > 0 and not self._closed:
            self._event.clear()
            try:
                await asyncio.wait_for(self._event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return []
            with self._lock:
                items = list(self._buffer)
                self._buffer.clear()
                self.stats.popped += len(items)
                if items:
                    self._last_pop_time = time.time()
                return items

        return []

    # -- lifecycle --

    def close(self) -> None:
        """Mark the buffer as closed.  Wakes any waiting consumer."""
        self._closed = True
        self._event.set()

    def reset(self) -> None:
        """Clear all events and reset statistics.  Re-opens the buffer."""
        with self._lock:
            self._buffer.clear()
        self._closed = False
        self._last_pop_time = time.time()
        self.stats = BufferStats()

    def get_status(self) -> dict:
        """Return a snapshot of buffer state for diagnostics."""
        with self._lock:
            fill = len(self._buffer)
        return {
            "max_size": self._max_size,
            "current_size": fill,
            "fill_pct": round(100 * fill / self._max_size, 1) if self._max_size else 0,
            "closed": self._closed,
            "is_slow": self.is_slow,
            "idle_timed_out": self.is_idle_timed_out,
            "stats": self.stats.to_dict(),
        }


# -- Module availability flag --
SSE_BACKPRESSURE_AVAILABLE = True
