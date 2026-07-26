#!/usr/bin/env python3
"""
Thread-safe SQLite/SQLCipher connection pool.

Provides a bounded pool of reusable database connections with health
checks, WAL mode enforcement, and context-manager-based checkout/checkin.

Usage::

    from opti_oignon.connection_pool import ConnectionPool

    pool = ConnectionPool("data/app.db", pool_size=5)
    with pool.connection() as conn:
        conn.execute("SELECT 1")

    # Or via the module-level factory:
    from opti_oignon.connection_pool import get_pool
    pool = get_pool("data/app.db")

The pool integrates with ``db_utils.safe_connect`` when available,
falling back to plain ``sqlite3.connect`` otherwise.
"""

import contextlib
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Hardcoded, never overridable
checkpoint_before_apply = True

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

_SAFE_CONNECT_AVAILABLE = False
_safe_connect_fn = None

try:
    from opti_oignon.db_utils import safe_connect as _safe_connect_fn
    _SAFE_CONNECT_AVAILABLE = True
except ImportError:
    logger.warning(
        "db_utils unavailable: connection_pool falling back to PLAINTEXT "
        "sqlite3. Pooled database connections are NOT encrypted at rest."
    )


def _create_connection(
    db_path: str,
    timeout: float = 5.0,
) -> sqlite3.Connection:
    """Create a new SQLite connection, using SQLCipher when available."""
    if _SAFE_CONNECT_AVAILABLE and _safe_connect_fn is not None:
        conn = _safe_connect_fn(
            db_path,
            check_same_thread=False,
            timeout=timeout,
        )
    else:
        conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=timeout,
        )
    return conn


# ---------------------------------------------------------------------------
# Pool statistics
# ---------------------------------------------------------------------------

@dataclass
class PoolStats:
    """Cumulative statistics for a connection pool."""

    checkouts: int = 0
    checkins: int = 0
    created: int = 0
    failed_health_checks: int = 0
    wait_timeouts: int = 0
    total_wait_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "checkouts": self.checkouts,
            "checkins": self.checkins,
            "created": self.created,
            "failed_health_checks": self.failed_health_checks,
            "wait_timeouts": self.wait_timeouts,
            "avg_wait_ms": round(self.total_wait_ms / max(self.checkouts, 1), 2),
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Connection wrapper
# ---------------------------------------------------------------------------

@dataclass
class PooledConnection:
    """Metadata wrapper around a raw sqlite3.Connection."""

    conn: sqlite3.Connection
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0

    def touch(self) -> None:
        self.last_used_at = time.time()
        self.use_count += 1


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

class ConnectionPool:
    """Thread-safe bounded connection pool for SQLite.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    pool_size : int
        Maximum number of connections in the pool.
    connect_timeout : float
        Timeout in seconds for creating a new connection.
    checkout_timeout : float
        Maximum seconds to wait when all connections are busy.
        Set to 0 for non-blocking (raises immediately).
    health_check : bool
        Run a liveness probe (``SELECT 1``) on every checkout and
        replace dead connections (the previous
        ``PRAGMA integrity_check`` scanned the entire database on
        every checkout). Set to False to skip entirely; integrity
        verification belongs to offline maintenance, not the hot path.
    wal_mode : bool
        Enforce WAL journal mode on every new connection.
    """

    def __init__(
        self,
        db_path: str | Path,
        pool_size: int = 5,
        connect_timeout: float = 5.0,
        checkout_timeout: float = 10.0,
        health_check: bool = True,
        wal_mode: bool = True,
    ) -> None:
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be > 0")

        self._db_path = str(db_path)
        self._pool_size = pool_size
        self._connect_timeout = connect_timeout
        self._checkout_timeout = checkout_timeout
        self._health_check = health_check
        self._wal_mode = wal_mode

        self._pool: queue.Queue[PooledConnection] = queue.Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._total_created = 0
        self._closed = False
        self.stats = PoolStats()

    # -- properties --

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def available(self) -> int:
        """Number of idle connections currently in the pool."""
        return self._pool.qsize()

    @property
    def in_use(self) -> int:
        """Estimated number of connections currently checked out."""
        return self._total_created - self._pool.qsize()

    @property
    def closed(self) -> bool:
        return self._closed

    # -- internal helpers --

    def _new_connection(self, reserved: bool = False) -> PooledConnection:
        """Create a new raw connection and apply WAL mode if configured.

        Args:
            reserved: True when the caller already reserved
                the pool slot (incremented ``_total_created``) under the
                lock; the count is then not incremented again here.
        """
        conn = _create_connection(self._db_path, timeout=self._connect_timeout)
        if self._wal_mode:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except Exception as exc:
                logger.debug("WAL mode enforcement failed: %s", exc)
        with self._lock:
            if not reserved:
                self._total_created += 1
            self.stats.created += 1
        pc = PooledConnection(conn=conn)
        logger.debug(
            "Pool[%s]: created connection #%d",
            self._db_path, self._total_created,
        )
        return pc

    def _check_health(self, pc: PooledConnection) -> bool:
        """Run a health check on a pooled connection.

        Returns True if healthy, False otherwise.
        """
        if not self._health_check:
            return True
        try:
            # Liveness probe, not a full integrity scan.
            result = pc.conn.execute("SELECT 1").fetchone()
            return result is not None and result[0] == 1
        except Exception as exc:
            logger.warning("Health check failed for %s: %s", self._db_path, exc)
            self.stats.failed_health_checks += 1
            return False

    def _discard(self, pc: PooledConnection) -> None:
        """Close and discard a connection (do not return to pool)."""
        try:
            pc.conn.close()
        except Exception:
            pass
        with self._lock:
            self._total_created = max(0, self._total_created - 1)

    # -- public API --

    def checkout(self) -> sqlite3.Connection:
        """Obtain a connection from the pool.

        If the pool is empty and the total created count is below
        ``pool_size``, a new connection is created.  Otherwise, waits
        up to ``checkout_timeout`` seconds for a connection to be
        returned by another thread.

        Raises
        ------
        TimeoutError
            If no connection becomes available within the timeout.
        RuntimeError
            If the pool is closed.
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        start = time.monotonic()

        # Try to get an existing idle connection
        try:
            pc = self._pool.get_nowait()
        except queue.Empty:
            pc = None

        # If none idle, try creating a new one if under limit
        if pc is None:
            # Reserve the slot inside the lock. The previous
            # check-then-act let two threads both pass the bound check and
            # overshoot pool_size under concurrency. On creation failure the
            # reserved slot is released.
            with self._lock:
                can_create = self._total_created < self._pool_size
                if can_create:
                    self._total_created += 1
            if can_create:
                try:
                    pc = self._new_connection(reserved=True)
                except Exception:
                    with self._lock:
                        self._total_created = max(0, self._total_created - 1)
                    raise
            else:
                # Pool is full, wait for a return
                try:
                    pc = self._pool.get(timeout=self._checkout_timeout)
                except queue.Empty:
                    self.stats.wait_timeouts += 1
                    raise TimeoutError(
                        f"Connection pool exhausted ({self._pool_size} connections, "
                        f"waited {self._checkout_timeout}s)"
                    )

        # Health check
        if not self._check_health(pc):
            self._discard(pc)
            # Try once more with a fresh connection
            pc = self._new_connection()
            if not self._check_health(pc):
                self._discard(pc)
                raise RuntimeError("Failed health check on fresh connection")

        pc.touch()
        elapsed_ms = (time.monotonic() - start) * 1000
        self.stats.checkouts += 1
        self.stats.total_wait_ms += elapsed_ms
        return pc.conn

    def checkin(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool.

        If the pool is closed or full, the connection is closed instead.
        """
        if self._closed:
            try:
                conn.close()
            except Exception:
                pass
            return

        # Never return a connection holding an open transaction
        # to the pool -- the next checkout would inherit dirty state and a
        # write lock. rollback() is a no-op on a clean connection; if it
        # fails, the connection is discarded rather than pooled.
        try:
            conn.rollback()
        except Exception as exc:
            logger.debug("Rollback on checkin failed; discarding: %s", exc)
            self._discard(PooledConnection(conn=conn))
            return

        pc = PooledConnection(conn=conn, last_used_at=time.time())
        try:
            self._pool.put_nowait(pc)
            self.stats.checkins += 1
        except queue.Full:
            # Pool somehow overfull -- discard
            self._discard(pc)

    @contextlib.contextmanager
    def connection(self):
        """Context manager for checkout/checkin lifecycle.

        Usage::

            with pool.connection() as conn:
                conn.execute("SELECT 1")
        """
        conn = self.checkout()
        try:
            yield conn
        finally:
            self.checkin(conn)

    def close(self) -> None:
        """Close all idle connections and mark the pool as closed.

        Connections currently checked out will be closed when they are
        returned via ``checkin()``.
        """
        self._closed = True
        while True:
            try:
                pc = self._pool.get_nowait()
                try:
                    pc.conn.close()
                except Exception:
                    pass
            except queue.Empty:
                break
        with self._lock:
            self._total_created = 0
        logger.info("Connection pool closed: %s", self._db_path)

    def get_status(self) -> dict:
        """Return a diagnostic snapshot of pool state."""
        return {
            "db_path": self._db_path,
            "pool_size": self._pool_size,
            "available": self.available,
            "in_use": self.in_use,
            "closed": self._closed,
            "health_check": self._health_check,
            "wal_mode": self._wal_mode,
            "stats": self.stats.to_dict(),
        }


# ---------------------------------------------------------------------------
# Module-level pool registry (singleton per db_path)
# ---------------------------------------------------------------------------

_pools: dict[str, ConnectionPool] = {}
_pools_lock = threading.Lock()


def get_pool(
    db_path: str | Path,
    pool_size: int = 5,
    health_check: bool = True,
    wal_mode: bool = True,
) -> ConnectionPool:
    """Get or create a connection pool for the given database path.

    The first call with a given ``db_path`` creates the pool; subsequent
    calls return the same instance regardless of other parameters.
    """
    key = str(Path(db_path).resolve())
    with _pools_lock:
        if key not in _pools:
            _pools[key] = ConnectionPool(
                db_path=db_path,
                pool_size=pool_size,
                health_check=health_check,
                wal_mode=wal_mode,
            )
            logger.info(
                "Created connection pool for %s (size=%d)",
                db_path, pool_size,
            )
        return _pools[key]


def close_all_pools() -> None:
    """Close all registered connection pools."""
    with _pools_lock:
        for pool in _pools.values():
            pool.close()
        _pools.clear()


def list_pools() -> list[dict]:
    """Return status of all registered pools."""
    with _pools_lock:
        return [pool.get_status() for pool in _pools.values()]


# -- Module availability flag --
CONNECTION_POOL_AVAILABLE = True
