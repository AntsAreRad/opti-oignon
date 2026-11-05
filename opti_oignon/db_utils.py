#!/usr/bin/env python3
"""
Centralized SQLite connection utility.

Provides ``safe_connect()`` as a drop-in replacement for
``sqlite3.connect()`` that routes through ``get_encrypted_connection()``
when SQLCipher is available.  This avoids duplicating the
try/except import pattern in every module.

Usage::

    from opti_oignon.db_utils import safe_connect

    conn = safe_connect("data/my.db", check_same_thread=False)
    # Works exactly like sqlite3.connect but with encryption when available
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Hardcoded, never overridable
checkpoint_before_apply = True

# TC-02: emit the plaintext-fallback warning once per process (the condition is
# process-global -- db_encryption is either importable or not).
_plaintext_fallback_warned = False

# A tree with no db_encryption module stores in plaintext (Bulbe refuses); a db_encryption module that is there and
# does not import (a circular import, a dependency missing, a file half written) refuses every store: what failed,
# else None.
_ENCRYPTION_BROKEN: str | None = None
try:
    from opti_oignon.db_encryption import get_encrypted_connection
    _ENCRYPTION_AVAILABLE = True
except Exception as _exc:  # noqa: BLE001 - every failure but the module's absence refuses below
    _ENCRYPTION_AVAILABLE = False
    get_encrypted_connection = None  # type: ignore[assignment]
    if not (isinstance(_exc, ModuleNotFoundError) and _exc.name in ("opti_oignon", "opti_oignon.db_encryption")):
        _ENCRYPTION_BROKEN = f"{type(_exc).__name__}: {_exc}"


def safe_connect(
    db_path: str | Path,
    *,
    check_same_thread: bool = True,
    timeout: float = 5.0,
    read_only: bool = False,
) -> sqlite3.Connection:
    """Open a SQLite connection, using SQLCipher when available.

    Drop-in replacement for ``sqlite3.connect()`` that transparently
    routes through ``get_encrypted_connection()`` for SQLCipher support.

    Parameters
    ----------
    db_path : str or Path
        Path to the database file.
    check_same_thread : bool
        SQLite check_same_thread parameter.
    timeout : float
        Connection timeout in seconds.
    read_only : bool
        Open the file read-only: nothing is written through the connection,
        and opening or closing it writes no byte of the store's files (no WAL
        checkpoint, no rollback of a hot journal, which is refused instead).

    Returns
    -------
    sqlite3.Connection
    """
    if _ENCRYPTION_AVAILABLE and get_encrypted_connection is not None:
        if read_only:
            return get_encrypted_connection(
                str(db_path),
                check_same_thread=check_same_thread,
                timeout=timeout,
                read_only=True,
            )
        return get_encrypted_connection(
            str(db_path),
            check_same_thread=check_same_thread,
            timeout=timeout,
        )

    # TC-02: db_encryption is unavailable. Do NOT fall back to plaintext
    # silently. A module there that does not import refuses in every mode;
    # a tree without it fails closed in Bulbe (matching
    # get_encrypted_connection, a mode that cannot be read or whose module
    # does not import being Bulbe), and warns loudly once in Daily so a
    # degraded install is not silently writing plaintext while the rest of
    # the code still assumes encryption.
    if _ENCRYPTION_BROKEN is not None:
        raise RuntimeError(
            f"Database encryption module does not import ({_ENCRYPTION_BROKEN}). "
            f"Refusing to open {db_path} in plaintext."
        )
    try:
        from opti_oignon.security_mode import is_bulbe

        bulbe = is_bulbe()
    except ImportError as exc:
        bulbe = not (isinstance(exc, ModuleNotFoundError) and exc.name == "opti_oignon.security_mode")
    except Exception:  # noqa: BLE001 - a mode that cannot be read is Bulbe
        bulbe = True
    if bulbe:
        raise RuntimeError(
            f"Database encryption module unavailable; Bulbe mode requires "
            f"encryption. Refusing to open {db_path} in plaintext."
        )
    global _plaintext_fallback_warned
    if not _plaintext_fallback_warned:
        logger.warning(
            "db_encryption unavailable: opening PLAINTEXT SQLite connections "
            "(e.g. %s). Data at rest is NOT encrypted; install SQLCipher "
            "support to enable encryption.",
            db_path,
        )
        _plaintext_fallback_warned = True
    if read_only:
        uri = Path(db_path).absolute().as_uri() + "?mode=ro"
        if Path(str(db_path) + "-shm").exists():
            uri += "&readonly_shm=1"
        return sqlite3.connect(uri, check_same_thread=check_same_thread, timeout=timeout, uri=True)
    return sqlite3.connect(
        str(db_path),
        check_same_thread=check_same_thread,
        timeout=timeout,
    )


DB_UTILS_AVAILABLE = True
