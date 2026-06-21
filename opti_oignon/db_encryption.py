#!/usr/bin/env python3
"""
SQLCipher full-database encryption for Opti-Oignon (S126).

Provides a centralized ``get_encrypted_connection()`` factory that replaces
all direct ``sqlite3.connect()`` calls.  When SQLCipher is available, the
factory sets encryption PRAGMAs on every connection.

Mode behaviour:
  - **Daily mode**: encryption optional.  If SQLCipher is installed, DBs are
    encrypted.  If not, falls back to plain sqlite3 with a warning.
  - **Bulbe mode**: encryption required.  Refuses to open a DB without
    SQLCipher installed.

SQLCipher 4.x PRAGMAs applied:
  - ``PRAGMA key = '<derived_hex_key>'``
  - ``PRAGMA cipher_page_size = 4096``
  - ``PRAGMA cipher_hmac_algorithm = HMAC_SHA512``
  - ``PRAGMA kdf_iter = 256000``

Migration: existing unencrypted DBs can be migrated in-place using
``sqlcipher_export`` (ATTACH plain DB, export to encrypted, swap files).

Security derives from the encryption key, not from the presence of
SQLCipher.  An attacker with the key can always read the data
(Kerckhoffs principle).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLCipher availability
# ---------------------------------------------------------------------------

SQLCIPHER_AVAILABLE = False
_sqlcipher_module: Any = None

try:
    # pysqlcipher3 is the most common binding
    import pysqlcipher3.dbapi2 as _sqlcipher_module  # type: ignore[no-redef]
    SQLCIPHER_AVAILABLE = True
    logger.info("SQLCipher available via pysqlcipher3")
except ImportError:
    try:
        # Some distributions ship as sqlcipher3
        import sqlcipher3 as _sqlcipher_module  # type: ignore[no-redef]
        SQLCIPHER_AVAILABLE = True
        logger.info("SQLCipher available via sqlcipher3")
    except ImportError:
        logger.info(
            "SQLCipher not available. Full-DB encryption disabled. "
            "Install pysqlcipher3 or sqlcipher3 for encryption support."
        )


# ---------------------------------------------------------------------------
# SQLCipher exception alignment
# ---------------------------------------------------------------------------
# SQLCipher's bindings raise their own DB-API exception classes, unrelated to
# the standard-library sqlite3 ones. The codebase guards many migrations and
# queries with `except sqlite3.OperationalError` (and siblings), which silently
# stop matching once a connection is SQLCipher-backed. Rebase the SQLCipher
# exception classes onto their sqlite3 equivalents so those handlers keep
# working. Done once at import time; applies to every SQLCipher-backed
# connection in the process.
if SQLCIPHER_AVAILABLE:
    for _dbapi_name in ("sqlcipher3.dbapi2", "pysqlcipher3.dbapi2"):
        try:
            _dbapi = __import__(_dbapi_name, fromlist=["dbapi2"])
        except Exception:
            continue
        for _exc_name in (
            "InterfaceError", "DatabaseError", "DataError", "OperationalError",
            "IntegrityError", "InternalError", "ProgrammingError",
            "NotSupportedError",
        ):
            _sc_exc = getattr(_dbapi, _exc_name, None)
            _sq_exc = getattr(sqlite3, _exc_name, None)
            if (
                _sc_exc is not None
                and _sq_exc is not None
                and not issubclass(_sc_exc, _sq_exc)
            ):
                try:
                    _sc_exc.__bases__ = (_sq_exc,)
                except TypeError:
                    pass


# ---------------------------------------------------------------------------
# Row factory compatible with both sqlite3 and SQLCipher cursors
# ---------------------------------------------------------------------------
# The codebase sets ``conn.row_factory = sqlite3.Row`` in many places, but
# sqlite3.Row rejects SQLCipher cursors ("Row() argument 1 must be
# sqlite3.Cursor"). This drop-in factory mimics sqlite3.Row (integer and
# case-insensitive name indexing, keys(), len(), iteration over values) without
# validating the cursor type, so it works on both backends. When SQLCipher is
# available we alias sqlite3.Row to it, so every existing row_factory assignment
# keeps working under encryption.
class _CompatRow:
    __slots__ = ("_keys", "_values", "_index")

    def __init__(self, cursor: Any, row: Any) -> None:
        self._keys = [d[0] for d in cursor.description]
        self._values = tuple(row)
        self._index: Any = None

    def keys(self) -> list:
        return list(self._keys)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (int, slice)):
            return self._values[key]
        if self._index is None:
            self._index = {k.lower(): i for i, k in enumerate(self._keys)}
        try:
            return self._values[self._index[key.lower()]]
        except (KeyError, AttributeError):
            raise IndexError(f"No item with key {key!r}")

    def __repr__(self) -> str:
        return f"<_CompatRow {dict(zip(self._keys, self._values))!r}>"


if SQLCIPHER_AVAILABLE:
    sqlite3.Row = _CompatRow  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# SQLCipher PRAGMA values (SQLCipher 4.x defaults + hardened)
_CIPHER_PAGE_SIZE = 4096
_CIPHER_HMAC_ALGORITHM = "HMAC_SHA512"
_KDF_ITER = 256000

# TC-02: warn once per process when a connection falls back to plaintext (the
# condition -- no SQLCipher or no key, in a non-enforcing mode -- is stable).
_plaintext_warned = False


# ---------------------------------------------------------------------------
# Key derivation for DB encryption
# ---------------------------------------------------------------------------

def _get_db_encryption_key() -> bytes | None:
    """Derive a DB-specific encryption key from the master key.

    Uses the master encryption key from the keyfile and derives
    a DB-specific subkey via HKDF-like construction (HMAC-SHA256).
    This ensures the DB encryption key is distinct from the
    field-level encryption key and the HMAC signing key.
    """
    try:
        from opti_oignon.encryption import get_encryption_key
        master_key = get_encryption_key()
        if master_key:
            # S129: Extract raw bytes if SecureBytes
            raw_key = master_key.as_bytes() if hasattr(master_key, "as_bytes") else master_key
            # Derive DB subkey: HMAC-SHA256(master_key, "opti-oignon-sqlcipher-v1")
            import hmac
            subkey = hmac.new(
                raw_key,
                b"opti-oignon-sqlcipher-v1",
                hashlib.sha256,
            ).digest()
            return subkey
    except Exception as exc:
        logger.debug("Failed to derive DB encryption key: %s", exc)
    return None


def _key_to_hex_pragma(key: bytes) -> str:
    """Convert a key to the hex format SQLCipher expects.

    SQLCipher accepts: PRAGMA key = "x'<hex>'";
    """
    return f"\"x'{key.hex()}'\""


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def get_encrypted_connection(
    db_path: str | Path,
    *,
    check_same_thread: bool = True,
    timeout: float = 5.0,
    enforce_encryption: bool | None = None,
) -> sqlite3.Connection:
    """Open a SQLite connection with optional SQLCipher encryption.

    This is the single entry point for all DB connections in
    Opti-Oignon.  It replaces direct ``sqlite3.connect()`` calls.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    check_same_thread : bool
        SQLite check_same_thread parameter.
    timeout : float
        Connection timeout in seconds.
    enforce_encryption : bool or None
        If True, raises if SQLCipher not available.
        If None, uses mode-based enforcement (Bulbe = required).

    Returns
    -------
    sqlite3.Connection
        A connection with encryption PRAGMAs applied if possible.

    Raises
    ------
    RuntimeError
        If encryption is required but SQLCipher is not available.
    """
    db_path = str(db_path)

    # Determine whether encryption is required
    if enforce_encryption is None:
        try:
            from opti_oignon.security_mode import is_bulbe
            enforce_encryption = is_bulbe()
        except ImportError:
            enforce_encryption = False

    if enforce_encryption and not SQLCIPHER_AVAILABLE:
        raise RuntimeError(
            f"Bulbe mode requires SQLCipher for database encryption, "
            f"but SQLCipher is not installed. Cannot open {db_path}. "
            f"Install pysqlcipher3 or sqlcipher3."
        )

    key = _get_db_encryption_key()

    if SQLCIPHER_AVAILABLE and key:
        # Use SQLCipher encrypted connection
        conn = _sqlcipher_module.connect(
            db_path,
            check_same_thread=check_same_thread,
            timeout=timeout,
        )
        hex_key = _key_to_hex_pragma(key)
        conn.execute(f"PRAGMA key = {hex_key}")
        conn.execute(f"PRAGMA cipher_page_size = {_CIPHER_PAGE_SIZE}")
        conn.execute(f"PRAGMA cipher_hmac_algorithm = {_CIPHER_HMAC_ALGORITHM}")
        conn.execute(f"PRAGMA kdf_iter = {_KDF_ITER}")

        # Verify the key works by reading a page
        try:
            conn.execute("SELECT count(*) FROM sqlite_master")
        except Exception:
            conn.close()
            raise RuntimeError(
                f"SQLCipher key verification failed for {db_path}. "
                f"The database may not be encrypted or the key is wrong."
            )

        logger.debug("Opened encrypted connection to %s", db_path)
        return conn

    # Fallback: plain sqlite3
    if enforce_encryption:
        raise RuntimeError(
            f"Encryption required but no encryption key available "
            f"for {db_path}."
        )

    conn = sqlite3.connect(
        db_path,
        check_same_thread=check_same_thread,
        timeout=timeout,
    )
    global _plaintext_warned
    if not _plaintext_warned:
        logger.warning(
            "Opening PLAINTEXT database connection to %s: SQLCipher unavailable "
            "or no encryption key. Data at rest is NOT encrypted (Bulbe mode "
            "would refuse). This warning is emitted once per process.",
            db_path,
        )
        _plaintext_warned = True
    return conn


# ---------------------------------------------------------------------------
# Migration: unencrypted -> encrypted
# ---------------------------------------------------------------------------

def migrate_db_to_encrypted(
    db_path: str | Path,
    *,
    backup: bool = True,
) -> dict[str, Any]:
    """Migrate an existing unencrypted DB to SQLCipher encryption.

    Uses the sqlcipher_export extension: attaches the plain DB,
    exports to a new encrypted DB, then swaps files.

    Parameters
    ----------
    db_path : str or Path
        Path to the unencrypted database.
    backup : bool
        If True, keep a .bak copy of the original.

    Returns
    -------
    dict with keys: success, message, backup_path (if applicable)
    """
    db_path = Path(db_path).resolve()

    if not db_path.exists():
        return {"success": False, "message": f"Database not found: {db_path}"}

    if not SQLCIPHER_AVAILABLE:
        return {
            "success": False,
            "message": "SQLCipher not available. Install pysqlcipher3.",
        }

    key = _get_db_encryption_key()
    if not key:
        return {
            "success": False,
            "message": "No encryption key available. Run encryption setup first.",
        }

    # Check if already encrypted
    if is_db_encrypted(db_path):
        return {"success": True, "message": "Database is already encrypted"}

    # Create encrypted copy
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".db", dir=str(db_path.parent)
    )
    os.close(tmp_fd)

    try:
        # Open the new encrypted DB
        hex_key = _key_to_hex_pragma(key)
        enc_conn = _sqlcipher_module.connect(tmp_path)
        enc_conn.execute(f"PRAGMA key = {hex_key}")
        enc_conn.execute(f"PRAGMA cipher_page_size = {_CIPHER_PAGE_SIZE}")
        enc_conn.execute(f"PRAGMA cipher_hmac_algorithm = {_CIPHER_HMAC_ALGORITHM}")
        enc_conn.execute(f"PRAGMA kdf_iter = {_KDF_ITER}")

        # Attach the plain DB and export
        enc_conn.execute(
            f"ATTACH DATABASE '{db_path}' AS plaintext KEY ''"
        )
        enc_conn.execute("SELECT sqlcipher_export('main', 'plaintext')")
        enc_conn.execute("DETACH DATABASE plaintext")
        enc_conn.close()

        # Backup original
        backup_path = None
        if backup:
            backup_path = str(db_path) + ".bak"
            shutil.copy2(str(db_path), backup_path)

        # Swap files
        shutil.move(tmp_path, str(db_path))

        logger.info("Migrated %s to encrypted format", db_path)
        result: dict[str, Any] = {
            "success": True,
            "message": f"Migrated {db_path.name} to encrypted format",
        }
        if backup_path:
            result["backup_path"] = backup_path
        return result

    except Exception as exc:
        # Cleanup temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        logger.error("Migration failed for %s: %s", db_path, exc)
        return {"success": False, "message": f"Migration failed: {exc}"}


def is_db_encrypted(db_path: str | Path) -> bool:
    """Check if a database file is SQLCipher encrypted.

    An encrypted SQLCipher DB will fail to open with plain sqlite3.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return False

    try:
        # Try to open with plain sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("SELECT count(*) FROM sqlite_master")
        conn.close()
        return False  # Plain DB, readable without key
    except Exception:
        return True  # Cannot read without key -> encrypted


def get_db_status(db_path: str | Path) -> dict[str, Any]:
    """Return encryption status for a database file."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {
            "path": str(db_path),
            "exists": False,
            "encrypted": False,
            "size_bytes": 0,
        }

    encrypted = is_db_encrypted(db_path)
    size = db_path.stat().st_size

    return {
        "path": str(db_path),
        "exists": True,
        "encrypted": encrypted,
        "size_bytes": size,
        "sqlcipher_available": SQLCIPHER_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# Bulk status / migration
# ---------------------------------------------------------------------------

def get_all_db_status() -> list[dict[str, Any]]:
    """Return encryption status for all known databases."""
    db_files = list(_DATA_DIR.glob("*.db"))
    return [get_db_status(p) for p in sorted(db_files)]


def migrate_all_databases(*, backup: bool = True) -> dict[str, Any]:
    """Migrate all unencrypted databases to SQLCipher format.

    Returns a summary with per-DB results.
    """
    if not SQLCIPHER_AVAILABLE:
        return {
            "success": False,
            "message": "SQLCipher not available",
            "results": [],
        }

    db_files = list(_DATA_DIR.glob("*.db"))
    results = []
    all_ok = True

    for db_path in sorted(db_files):
        result = migrate_db_to_encrypted(db_path, backup=backup)
        result["db"] = db_path.name
        results.append(result)
        if not result["success"] and "already encrypted" not in result.get("message", ""):
            all_ok = False

    return {
        "success": all_ok,
        "message": f"Processed {len(results)} databases",
        "results": results,
    }


# ---------------------------------------------------------------------------
# Status helper for the security module
# ---------------------------------------------------------------------------

def encryption_status_summary() -> dict[str, Any]:
    """Summary for the security status API."""
    dbs = get_all_db_status()
    total = len(dbs)
    encrypted = sum(1 for d in dbs if d.get("encrypted"))
    return {
        "sqlcipher_available": SQLCIPHER_AVAILABLE,
        "total_databases": total,
        "encrypted_databases": encrypted,
        "unencrypted_databases": total - encrypted,
        "fully_encrypted": total > 0 and encrypted == total,
        "databases": dbs,
    }
