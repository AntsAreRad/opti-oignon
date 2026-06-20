#!/usr/bin/env python3
"""
Rate Limiter -- Sliding Window Rate Limiting (S156)

Lightweight, thread-safe rate limiter with per-endpoint configuration.
Supports keying by IP address or user ID, with Bulbe-mode awareness
(stricter limits in maximum security mode).

Uses a sliding window algorithm: each key maintains a deque of
timestamps, and requests are allowed only if the count within the
window is below the configured limit.

SA-155-050: Rate limiting for file upload endpoints.
SA-155-051: Rate limiting for user management endpoints.

Usage::

    from opti_oignon.rate_limiter import RateLimiter, rate_limit_check

    limiter = RateLimiter()
    limiter.configure("file_upload", max_requests=10, window_seconds=60)

    # In a route handler:
    allowed, info = limiter.check("file_upload", key="192.168.1.1")
    if not allowed:
        raise HTTPException(status_code=429, detail=info["message"])
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# S145 pattern: hardcoded sentinel for pre-apply verification
checkpoint_before_apply = True


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

@dataclass
class EndpointLimit:
    """Rate limit configuration for a single endpoint group."""

    name: str
    max_requests: int = 10
    window_seconds: int = 60
    # Bulbe mode overrides (stricter)
    bulbe_max_requests: int | None = None
    bulbe_window_seconds: int | None = None

    def effective_limit(self, bulbe: bool = False) -> tuple[int, int]:
        """Return (max_requests, window_seconds) for current mode.

        Args:
            bulbe: Whether Bulbe mode is active.

        Returns:
            Tuple of (max_requests, window_seconds).
        """
        if bulbe:
            mr = self.bulbe_max_requests if self.bulbe_max_requests is not None else max(1, self.max_requests // 2)
            ws = self.bulbe_window_seconds if self.bulbe_window_seconds is not None else self.window_seconds
            return mr, ws
        return self.max_requests, self.window_seconds


# -------------------------------------------------------------------------
# Default endpoint configurations
# -------------------------------------------------------------------------

_DEFAULT_LIMITS: dict[str, EndpointLimit] = {
    "file_upload": EndpointLimit(
        name="file_upload",
        max_requests=10,
        window_seconds=60,
        bulbe_max_requests=5,
    ),
    "user_management": EndpointLimit(
        name="user_management",
        max_requests=5,
        window_seconds=60,
        bulbe_max_requests=2,
    ),
}


# -------------------------------------------------------------------------
# Sliding window rate limiter
# -------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe sliding window rate limiter.

    Tracks request timestamps per (endpoint, key) pair using deques.
    The sliding window evicts expired entries on each check, ensuring
    accurate counts without background cleanup threads.

    Attributes:
        _limits: Configured endpoint limits.
        _windows: Per-(endpoint, key) timestamp deques.
        _lock: Threading lock for concurrent access.
    """

    def __init__(self) -> None:
        self._limits: dict[str, EndpointLimit] = dict(_DEFAULT_LIMITS)
        self._windows: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()
        self._bulbe_cached: bool | None = None
        self._bulbe_check_time: float = 0.0
        logger.info("RateLimiter initialized with %d endpoint configs", len(self._limits))

    def configure(
        self,
        endpoint: str,
        max_requests: int = 10,
        window_seconds: int = 60,
        bulbe_max_requests: int | None = None,
        bulbe_window_seconds: int | None = None,
    ) -> None:
        """Configure or update rate limit for an endpoint group.

        Args:
            endpoint: Endpoint group name (e.g., "file_upload").
            max_requests: Maximum requests allowed per window.
            window_seconds: Sliding window duration in seconds.
            bulbe_max_requests: Override max for Bulbe mode (default: half).
            bulbe_window_seconds: Override window for Bulbe mode.
        """
        self._limits[endpoint] = EndpointLimit(
            name=endpoint,
            max_requests=max_requests,
            window_seconds=window_seconds,
            bulbe_max_requests=bulbe_max_requests,
            bulbe_window_seconds=bulbe_window_seconds,
        )
        logger.debug(
            "Rate limit configured: %s -> %d req / %ds",
            endpoint, max_requests, window_seconds,
        )

    def _is_bulbe(self) -> bool:
        """Check Bulbe mode with caching (re-check every 30s).

        Returns:
            True if Bulbe mode is active.
        """
        now = time.time()
        if self._bulbe_cached is not None and (now - self._bulbe_check_time) < 30.0:
            return self._bulbe_cached
        try:
            from opti_oignon.security_mode import is_bulbe
            self._bulbe_cached = is_bulbe()
        except ImportError:
            self._bulbe_cached = False
        self._bulbe_check_time = now
        return self._bulbe_cached

    def check(
        self,
        endpoint: str,
        key: str,
        now: float | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Check whether a request is allowed under the rate limit.

        Evicts expired timestamps and checks if the count is within
        the configured limit for the endpoint.

        Args:
            endpoint: Endpoint group name.
            key: Rate limit key (IP address, user ID, etc.).
            now: Current timestamp (default: time.time()). Exposed
                 for testing determinism.

        Returns:
            Tuple of (allowed, info_dict).
            info_dict contains: allowed, remaining, limit, window_seconds,
            retry_after (seconds until next allowed request, if denied),
            and message (human-readable).
        """
        if now is None:
            now = time.time()

        limit_config = self._limits.get(endpoint)
        if limit_config is None:
            # No limit configured: allow by default
            return True, {
                "allowed": True,
                "remaining": -1,
                "limit": -1,
                "window_seconds": 0,
                "retry_after": 0,
                "message": "No rate limit configured",
            }

        bulbe = self._is_bulbe()
        max_req, window_sec = limit_config.effective_limit(bulbe)

        bucket_key = (endpoint, key)

        with self._lock:
            dq = self._windows.get(bucket_key)
            if dq is None:
                dq = deque()
                self._windows[bucket_key] = dq

            # Evict expired timestamps
            cutoff = now - window_sec
            while dq and dq[0] <= cutoff:
                dq.popleft()

            current_count = len(dq)

            if current_count < max_req:
                # Allowed: record this request
                dq.append(now)
                remaining = max_req - current_count - 1
                return True, {
                    "allowed": True,
                    "remaining": remaining,
                    "limit": max_req,
                    "window_seconds": window_sec,
                    "retry_after": 0,
                    "message": "OK",
                }
            else:
                # Denied: compute retry_after from oldest entry
                retry_after = round(dq[0] + window_sec - now, 2)
                if retry_after < 0:
                    retry_after = 0.0
                return False, {
                    "allowed": False,
                    "remaining": 0,
                    "limit": max_req,
                    "window_seconds": window_sec,
                    "retry_after": retry_after,
                    "message": (
                        f"Rate limit exceeded: {max_req} requests per "
                        f"{window_sec}s. Retry after {retry_after}s."
                    ),
                }

    def reset(self, endpoint: str | None = None, key: str | None = None) -> int:
        """Reset rate limit state.

        Args:
            endpoint: If provided, only reset this endpoint.
            key: If provided with endpoint, only reset this specific key.

        Returns:
            Number of buckets cleared.
        """
        with self._lock:
            if endpoint is None:
                count = len(self._windows)
                self._windows.clear()
                return count

            if key is not None:
                bucket_key = (endpoint, key)
                if bucket_key in self._windows:
                    del self._windows[bucket_key]
                    return 1
                return 0

            to_remove = [k for k in self._windows if k[0] == endpoint]
            for k in to_remove:
                del self._windows[k]
            return len(to_remove)

    def get_status(self, endpoint: str, key: str, now: float | None = None) -> dict[str, Any]:
        """Get current rate limit status for a key without consuming a request.

        Args:
            endpoint: Endpoint group name.
            key: Rate limit key.
            now: Current timestamp (default: time.time()). Exposed
                 for testing determinism.

        Returns:
            Status dict with current_count, limit, remaining, window_seconds.
        """
        if now is None:
            now = time.time()
        limit_config = self._limits.get(endpoint)
        if limit_config is None:
            return {"configured": False}

        bulbe = self._is_bulbe()
        max_req, window_sec = limit_config.effective_limit(bulbe)

        bucket_key = (endpoint, key)

        with self._lock:
            dq = self._windows.get(bucket_key)
            if dq is None:
                return {
                    "configured": True,
                    "current_count": 0,
                    "limit": max_req,
                    "remaining": max_req,
                    "window_seconds": window_sec,
                    "bulbe_active": bulbe,
                }

            cutoff = now - window_sec
            while dq and dq[0] <= cutoff:
                dq.popleft()

            current = len(dq)
            return {
                "configured": True,
                "current_count": current,
                "limit": max_req,
                "remaining": max(0, max_req - current),
                "window_seconds": window_sec,
                "bulbe_active": bulbe,
            }

    @property
    def configured_endpoints(self) -> list[str]:
        """List all configured endpoint group names."""
        return list(self._limits.keys())

    def cleanup_expired(self) -> int:
        """Remove all expired entries from all buckets.

        Can be called periodically to free memory, though eviction
        also happens on each check() call.

        Returns:
            Number of empty buckets removed.
        """
        now = time.time()
        removed = 0
        with self._lock:
            for bucket_key in list(self._windows.keys()):
                dq = self._windows[bucket_key]
                endpoint = bucket_key[0]
                limit_config = self._limits.get(endpoint)
                if limit_config is None:
                    del self._windows[bucket_key]
                    removed += 1
                    continue
                bulbe = self._is_bulbe()
                _, window_sec = limit_config.effective_limit(bulbe)
                cutoff = now - window_sec
                while dq and dq[0] <= cutoff:
                    dq.popleft()
                if not dq:
                    del self._windows[bucket_key]
                    removed += 1
        return removed


# -------------------------------------------------------------------------
# Module-level singleton
# -------------------------------------------------------------------------

rate_limiter = RateLimiter()


def rate_limit_check(
    endpoint: str,
    key: str,
    now: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Convenience wrapper around the module-level rate limiter.

    Args:
        endpoint: Endpoint group name.
        key: Rate limit key (IP, user ID, etc.).
        now: Optional timestamp override for testing.

    Returns:
        Tuple of (allowed, info_dict).
    """
    return rate_limiter.check(endpoint, key, now=now)
