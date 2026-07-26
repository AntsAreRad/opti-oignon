#!/usr/bin/env python3
"""
RAG connection pool integration.

Wires the ``ConnectionPool`` into RAG SQLite databases,
providing pooled connections for the ingestion job tracker and any
future RAG-side SQLite stores.

Key features:
- Lazy pool creation per database path via ``get_rag_pool()``
- Configurable pool size (default: 3 for RAG workloads)
- Automatic WAL mode and health checks from ``ConnectionPool``
- Context manager for clean checkout/checkin
- Graceful fallback when ``ConnectionPool`` is unavailable

Usage::

    from opti_oignon.rag.pool_integration import get_rag_pool, pooled_connection

    pool = get_rag_pool("/path/to/rag_ingest_jobs.db")
    with pooled_connection(pool) as conn:
        conn.execute("SELECT * FROM jobs")

    # Or one-liner:
    with rag_connection("/path/to/db") as conn:
        conn.execute("INSERT INTO ...")
"""

import contextlib
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Hardcoded, never overridable
checkpoint_before_apply = True

FEATURE_AVAILABLE = True

# -- Constants ---------------------------------------------------------------

DEFAULT_RAG_POOL_SIZE: int = 3
DEFAULT_RAG_POOL_TIMEOUT: float = 10.0

# -- Connection pool availability --------------------------------------------

_POOL_AVAILABLE = False
_ConnectionPool = None

try:
    from opti_oignon.connection_pool import ConnectionPool as _ConnectionPool
    _POOL_AVAILABLE = True
except ImportError:
    logger.debug("ConnectionPool not available; RAG pool integration disabled")


# -- Pool registry -----------------------------------------------------------

_pools: dict[str, Any] = {}
_pools_lock = threading.Lock()


def get_rag_pool(
    db_path: str | Path,
    *,
    pool_size: int = DEFAULT_RAG_POOL_SIZE,
    timeout: float = DEFAULT_RAG_POOL_TIMEOUT,
    health_check: bool = False,
) -> Any:
    """Return a pooled connection manager for a RAG database.

    If ``ConnectionPool`` is available, returns a pool
    instance.  Otherwise returns a ``FallbackPool`` that creates a
    fresh connection each time.

    Pools are cached by resolved path; calling with the same path
    returns the same pool.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    pool_size : int
        Number of connections in the pool (default: 3).
    timeout : float
        Checkout timeout in seconds (default: 10).
    health_check : bool
        Enable PRAGMA integrity_check on checkout (default: False,
        since RAG databases are typically small and fast).

    Returns
    -------
    ConnectionPool or FallbackPool
    """
    key = str(Path(db_path).resolve())
    with _pools_lock:
        if key not in _pools:
            if _POOL_AVAILABLE and _ConnectionPool is not None:
                pool = _ConnectionPool(
                    key,
                    pool_size=pool_size,
                    timeout=timeout,
                    health_check=health_check,
                )
                logger.info(
                    "Created RAG connection pool for %s (size=%d)",
                    key, pool_size,
                )
            else:
                pool = FallbackPool(key, timeout=timeout)
                logger.info(
                    "Created RAG fallback pool for %s (ConnectionPool unavailable)",
                    key,
                )
            _pools[key] = pool
        return _pools[key]


def close_rag_pool(db_path: str | Path) -> bool:
    """Close and remove a RAG pool. Returns True if a pool existed."""
    key = str(Path(db_path).resolve())
    with _pools_lock:
        pool = _pools.pop(key, None)
    if pool is None:
        return False
    try:
        if hasattr(pool, "close"):
            pool.close()
    except Exception as exc:
        logger.debug("Error closing RAG pool %s: %s", key, exc)
    return True


def close_all_rag_pools() -> int:
    """Close all RAG pools. Returns count of pools closed."""
    with _pools_lock:
        keys = list(_pools.keys())
    count = 0
    for key in keys:
        if close_rag_pool(key):
            count += 1
    return count


def list_rag_pools() -> list[str]:
    """Return paths of all active RAG pools."""
    with _pools_lock:
        return list(_pools.keys())


# -- Context managers --------------------------------------------------------

@contextlib.contextmanager
def pooled_connection(pool: Any) -> Generator[sqlite3.Connection, None, None]:
    """Checkout a connection from a pool via context manager.

    Works with both ``ConnectionPool`` and ``FallbackPool``.
    """
    if hasattr(pool, "connection"):
        with pool.connection() as conn:
            yield conn
    elif hasattr(pool, "checkout"):
        conn = pool.checkout()
        try:
            yield conn
        finally:
            pool.checkin(conn)
    else:
        raise TypeError(f"Unsupported pool type: {type(pool)}")


@contextlib.contextmanager
def rag_connection(
    db_path: str | Path,
    **pool_kwargs: Any,
) -> Generator[sqlite3.Connection, None, None]:
    """One-liner: get-or-create a pool and checkout a connection.

    Usage::

        with rag_connection("data/rag/jobs.db") as conn:
            conn.execute("SELECT 1")
    """
    pool = get_rag_pool(db_path, **pool_kwargs)
    with pooled_connection(pool) as conn:
        yield conn


# -- Fallback pool -----------------------------------------------------------

class FallbackPool:
    """Minimal pool substitute when ``ConnectionPool`` is not available.

    Creates a fresh SQLite connection on each checkout and closes it on
    checkin.  Provides the same ``connection()`` context-manager API.
    """

    def __init__(self, db_path: str, *, timeout: float = 5.0) -> None:
        self._db_path = db_path
        self._timeout = timeout
        self._checkout_count = 0
        self._lock = threading.Lock()

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def checkout_count(self) -> int:
        with self._lock:
            return self._checkout_count

    @contextlib.contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create a connection, yield it, then close."""
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            timeout=self._timeout,
        )
        with self._lock:
            self._checkout_count += 1
        try:
            # Enforce WAL for consistency with ConnectionPool
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def close(self) -> None:
        """No-op for fallback pool (connections are per-call)."""
        pass

    def stats(self) -> dict[str, Any]:
        """Return basic stats."""
        return {
            "db_path": self._db_path,
            "type": "fallback",
            "checkout_count": self.checkout_count,
        }


# -- Reset for test isolation ------------------------------------------------

def reset_rag_pools() -> None:
    """Close all pools and clear the registry (for testing)."""
    close_all_rag_pools()
    with _pools_lock:
        _pools.clear()
