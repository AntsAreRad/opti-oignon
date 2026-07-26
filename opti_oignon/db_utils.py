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

try:
    from opti_oignon.db_encryption import get_encrypted_connection
    _ENCRYPTION_AVAILABLE = True
except ImportError:
    _ENCRYPTION_AVAILABLE = False
    get_encrypted_connection = None  # type: ignore[assignment]


def safe_connect(
    db_path: str | Path,
    *,
    check_same_thread: bool = True,
    timeout: float = 5.0,
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

    Returns
    -------
    sqlite3.Connection
    """
    if _ENCRYPTION_AVAILABLE and get_encrypted_connection is not None:
        return get_encrypted_connection(
            str(db_path),
            check_same_thread=check_same_thread,
            timeout=timeout,
        )

    # TC-02: db_encryption is unavailable. Do NOT fall back to plaintext
    # silently. Fail closed in Bulbe (matching get_encrypted_connection), and
    # warn loudly once in Daily so a degraded install is not silently writing
    # plaintext while the rest of the code still assumes encryption.
    try:
        from opti_oignon.security_mode import is_bulbe

        bulbe = is_bulbe()
    except Exception:
        bulbe = False
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
    return sqlite3.connect(
        str(db_path),
        check_same_thread=check_same_thread,
        timeout=timeout,
    )


DB_UTILS_AVAILABLE = True
