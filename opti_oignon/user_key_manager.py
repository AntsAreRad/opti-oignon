#!/usr/bin/env python3
"""
Per-user encryption key management for Opti-Oignon (S142).

Provides:
- Argon2id-based per-user subkey derivation from user password
- In-memory session key cache (SecureBytes) with automatic wipe on logout
- Salt storage per user in the auth database
- Admin cannot read user data without user's password
- Key lifecycle: derive on login, cache for session, wipe on logout/expiry

Security guarantees:
- Each user's conversations/memories encrypted with their own subkey
- Keys held in SecureBytes (mlock + memset wipe)
- Salt stored alongside user record (not secret, per Argon2 design)
- Session cache indexed by user_id with TTL enforcement
- Thread-safe via threading.Lock

Architecture:
- UserKeyManager: singleton managing all per-user key lifecycle
- UserKeySalt: stored in auth.db user_key_salts table
- Session cache: dict[user_id] -> (SecureBytes, expiry_timestamp)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SecureBytes import (with fallback)
# ---------------------------------------------------------------------------

try:
    from opti_oignon.secure_bytes import SecureBytes, secure_key_from_bytes
    _SECURE_BYTES_AVAILABLE = True
except ImportError:
    _SECURE_BYTES_AVAILABLE = False

    class SecureBytes:  # type: ignore[no-redef]
        """Fallback when secure_bytes module is not available."""

        def __init__(self, data: bytes) -> None:
            self._data = bytes(data)
            self._wiped = False

        def as_bytes(self) -> bytes:
            if self._wiped:
                raise RuntimeError("SecureBytes has been wiped")
            return self._data

        def wipe(self) -> None:
            self._wiped = True
            self._data = b""

        @property
        def is_wiped(self) -> bool:
            return self._wiped

        def __len__(self) -> int:
            return len(self._data)

        def __bool__(self) -> bool:
            return not self._wiped and len(self._data) > 0

        def __repr__(self) -> str:
            return "<SecureBytes [REDACTED]>"

        def __enter__(self) -> "SecureBytes":
            return self

        def __exit__(self, *exc: Any) -> None:
            self.wipe()

    def secure_key_from_bytes(data: bytes) -> SecureBytes:  # type: ignore[misc]
        return SecureBytes(data)


# ---------------------------------------------------------------------------
# Argon2id / PBKDF2 import
# ---------------------------------------------------------------------------

_ARGON2_AVAILABLE = False
try:
    from argon2.low_level import hash_secret_raw, Type as Argon2Type
    _ARGON2_AVAILABLE = True
except ImportError:
    pass

import hashlib

# ---------------------------------------------------------------------------
# DB import
# ---------------------------------------------------------------------------

try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KEY_SIZE = 32
_SALT_LENGTH = 16

# Argon2id parameters (OWASP 2024)
_ARGON2_TIME_COST = 3
_ARGON2_MEMORY_COST = 65536  # 64 MB
_ARGON2_PARALLELISM = 4

# PBKDF2 fallback
_PBKDF2_ITERATIONS = 600_000

# Default session key TTL: 24 hours
_DEFAULT_KEY_TTL = 86400


# ---------------------------------------------------------------------------
# Per-user subkey derivation
# ---------------------------------------------------------------------------


def derive_user_subkey(
    password: str,
    salt: bytes | None = None,
    force_pbkdf2: bool = False,
) -> tuple[bytes, bytes, str]:
    """Derive a per-user encryption subkey from their password.

    Uses Argon2id if available, falls back to PBKDF2-SHA256.

    Args:
        password: User's plaintext password.
        salt: Optional salt (generated if not provided).
        force_pbkdf2: Force PBKDF2 even if Argon2 is available.

    Returns:
        Tuple of (derived_key_32bytes, salt, kdf_name).
    """
    if _ARGON2_AVAILABLE and not force_pbkdf2:
        if salt is None:
            salt = os.urandom(_SALT_LENGTH)
        key = hash_secret_raw(
            secret=password.encode("utf-8"),
            salt=salt,
            time_cost=_ARGON2_TIME_COST,
            memory_cost=_ARGON2_MEMORY_COST,
            parallelism=_ARGON2_PARALLELISM,
            hash_len=_KEY_SIZE,
            type=Argon2Type.ID,
        )
        return key, salt, "argon2id"

    # PBKDF2 fallback
    if salt is None:
        salt = os.urandom(_SALT_LENGTH)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
        dklen=_KEY_SIZE,
    )
    return key, salt, "pbkdf2"


# ---------------------------------------------------------------------------
# Salt storage (SQLite)
# ---------------------------------------------------------------------------


class UserKeySaltStore:
    """Stores per-user encryption key salts in SQLite.

    Table: user_key_salts (user_id TEXT PRIMARY KEY, salt BLOB, kdf TEXT, created_at REAL).
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = _safe_connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_key_salts (
                    user_id TEXT PRIMARY KEY,
                    salt BLOB NOT NULL,
                    kdf TEXT NOT NULL DEFAULT 'argon2id',
                    created_at REAL NOT NULL,
                    rotated_at REAL
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def get_salt(self, user_id: str) -> tuple[bytes, str] | None:
        """Get salt and KDF name for a user.

        Returns:
            Tuple of (salt, kdf_name) or None if not found.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT salt, kdf FROM user_key_salts WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row:
                return bytes(row["salt"]), row["kdf"]
            return None
        finally:
            conn.close()

    def store_salt(self, user_id: str, salt: bytes, kdf: str) -> None:
        """Store or update salt for a user."""
        now = time.time()
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO user_key_salts (user_id, salt, kdf, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       salt = excluded.salt,
                       kdf = excluded.kdf,
                       rotated_at = ?""",
                (user_id, salt, kdf, now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_salt(self, user_id: str) -> bool:
        """Delete salt for a user (used during user data deletion)."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                "DELETE FROM user_key_salts WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()

    def has_salt(self, user_id: str) -> bool:
        """Check if a user has a stored salt."""
        return self.get_salt(user_id) is not None


# ---------------------------------------------------------------------------
# Session key cache
# ---------------------------------------------------------------------------


class _SessionKeyEntry:
    """Holds a cached per-user key with expiry."""

    __slots__ = ("key", "expires_at")

    def __init__(self, key: SecureBytes, expires_at: float) -> None:
        self.key = key
        self.expires_at = expires_at

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def wipe(self) -> None:
        if self.key and not self.key.is_wiped:
            self.key.wipe()


# ---------------------------------------------------------------------------
# UserKeyManager (singleton)
# ---------------------------------------------------------------------------


class UserKeyManager:
    """Manages per-user encryption keys for multi-user isolation.

    Lifecycle:
      1. On user creation: generate salt, store in DB
      2. On login: derive subkey from password + stored salt, cache in memory
      3. During session: retrieve cached key for encrypt/decrypt operations
      4. On logout/expiry: wipe key from memory

    Thread-safe: all cache operations protected by Lock.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        key_ttl: int = _DEFAULT_KEY_TTL,
    ) -> None:
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent / "data" / "auth.db"
        self._salt_store = UserKeySaltStore(db_path)
        self._key_ttl = key_ttl
        self._cache: dict[str, _SessionKeyEntry] = {}
        self._lock = threading.Lock()

    @property
    def salt_store(self) -> UserKeySaltStore:
        """Access the underlying salt store."""
        return self._salt_store

    def initialize_user_key(
        self,
        user_id: str,
        password: str,
    ) -> bool:
        """Initialize encryption key material for a new user.

        Derives a subkey, stores the salt, and caches the key.

        Args:
            user_id: User identifier.
            password: User's plaintext password.

        Returns:
            True if initialization succeeded.
        """
        try:
            raw_key, salt, kdf = derive_user_subkey(password)
            self._salt_store.store_salt(user_id, salt, kdf)
            secure_key = secure_key_from_bytes(raw_key)
            with self._lock:
                # Wipe any existing cached key
                if user_id in self._cache:
                    self._cache[user_id].wipe()
                self._cache[user_id] = _SessionKeyEntry(
                    key=secure_key,
                    expires_at=time.time() + self._key_ttl,
                )
            logger.info(
                "Initialized per-user key for %s (kdf=%s)", user_id, kdf
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize user key for %s: %s", user_id, e)
            return False

    def derive_and_cache(
        self,
        user_id: str,
        password: str,
    ) -> bool:
        """Derive user subkey on login and cache it for the session.

        Uses the stored salt for the user. If no salt exists, initializes
        the user key first.

        Args:
            user_id: User identifier.
            password: User's plaintext password (from login).

        Returns:
            True if derivation and caching succeeded.
        """
        salt_info = self._salt_store.get_salt(user_id)
        if salt_info is None:
            # First login or salt not yet created
            return self.initialize_user_key(user_id, password)

        stored_salt, kdf = salt_info
        try:
            force_pbkdf2 = kdf == "pbkdf2"
            raw_key, _, _ = derive_user_subkey(
                password, salt=stored_salt, force_pbkdf2=force_pbkdf2
            )
            secure_key = secure_key_from_bytes(raw_key)
            with self._lock:
                if user_id in self._cache:
                    self._cache[user_id].wipe()
                self._cache[user_id] = _SessionKeyEntry(
                    key=secure_key,
                    expires_at=time.time() + self._key_ttl,
                )
            logger.debug("Cached per-user key for %s", user_id)
            return True
        except Exception as e:
            logger.error("Failed to derive key for %s: %s", user_id, e)
            return False

    def get_user_key(self, user_id: str) -> SecureBytes | None:
        """Get the cached encryption key for a user.

        Returns None if no key is cached or if the key has expired.
        Expired keys are wiped automatically.
        """
        with self._lock:
            entry = self._cache.get(user_id)
            if entry is None:
                return None
            if entry.is_expired():
                entry.wipe()
                del self._cache[user_id]
                logger.debug("Per-user key expired for %s", user_id)
                return None
            if entry.key.is_wiped:
                del self._cache[user_id]
                return None
            return entry.key

    def get_user_key_bytes(self, user_id: str) -> bytes | None:
        """Get raw key bytes for crypto operations.

        Convenience wrapper around get_user_key().
        """
        key = self.get_user_key(user_id)
        if key is None:
            return None
        return key.as_bytes()

    def wipe_user_key(self, user_id: str) -> bool:
        """Wipe a user's cached key from memory (logout/session expiry).

        Returns True if a key was found and wiped.
        """
        with self._lock:
            entry = self._cache.pop(user_id, None)
            if entry is not None:
                entry.wipe()
                logger.debug("Wiped per-user key for %s", user_id)
                return True
            return False

    def wipe_all(self) -> int:
        """Wipe all cached keys (shutdown)."""
        with self._lock:
            count = len(self._cache)
            for entry in self._cache.values():
                entry.wipe()
            self._cache.clear()
        logger.info("Wiped %d cached per-user keys", count)
        return count

    def is_key_cached(self, user_id: str) -> bool:
        """Check if a valid (non-expired) key is cached for a user."""
        return self.get_user_key(user_id) is not None

    def rotate_user_key(
        self,
        user_id: str,
        new_password: str,
    ) -> tuple[bytes | None, bytes | None]:
        """Rotate a user's encryption key (password change).

        Returns (old_key_bytes, new_key_bytes) so the caller can
        re-encrypt the user's data. Returns (None, None) on failure.
        """
        old_key = self.get_user_key_bytes(user_id)

        try:
            raw_key, salt, kdf = derive_user_subkey(new_password)
            self._salt_store.store_salt(user_id, salt, kdf)
            secure_key = secure_key_from_bytes(raw_key)
            with self._lock:
                if user_id in self._cache:
                    self._cache[user_id].wipe()
                self._cache[user_id] = _SessionKeyEntry(
                    key=secure_key,
                    expires_at=time.time() + self._key_ttl,
                )
            logger.info("Rotated per-user key for %s (kdf=%s)", user_id, kdf)
            return old_key, raw_key
        except Exception as e:
            logger.error("Failed to rotate key for %s: %s", user_id, e)
            return None, None

    def delete_user_keys(self, user_id: str) -> bool:
        """Delete all key material for a user (account deletion).

        Wipes cached key and removes salt from DB.
        """
        self.wipe_user_key(user_id)
        return self._salt_store.delete_salt(user_id)

    def cleanup_expired(self) -> int:
        """Remove expired keys from cache. Returns count of cleaned entries."""
        with self._lock:
            expired = [
                uid for uid, entry in self._cache.items()
                if entry.is_expired()
            ]
            for uid in expired:
                self._cache[uid].wipe()
                del self._cache[uid]
        if expired:
            logger.debug("Cleaned %d expired per-user keys", len(expired))
        return len(expired)

    def get_status(self) -> dict[str, Any]:
        """Get status information about the key manager."""
        with self._lock:
            cached_count = len(self._cache)
            expired_count = sum(
                1 for e in self._cache.values() if e.is_expired()
            )
        return {
            "cached_keys": cached_count,
            "expired_keys": expired_count,
            "key_ttl_seconds": self._key_ttl,
            "argon2_available": _ARGON2_AVAILABLE,
            "secure_bytes_available": _SECURE_BYTES_AVAILABLE,
            "kdf": "argon2id" if _ARGON2_AVAILABLE else "pbkdf2",
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_user_key_manager: UserKeyManager | None = None


def get_user_key_manager() -> UserKeyManager:
    """Get or create the singleton UserKeyManager."""
    global _user_key_manager
    if _user_key_manager is None:
        _user_key_manager = UserKeyManager()
    return _user_key_manager
