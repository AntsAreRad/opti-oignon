#!/usr/bin/env python3
"""
SQLCipher full-database encryption for Opti-Oignon.

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

Migration: the stores kept in plaintext -- every ``*.db`` of the
repository's data directory and of the package's -- are encrypted together,
with the platform stopped, by ``python3 -m opti_oignon.db_encryption
encrypt`` (``encrypt_all_stores``): all of them, or none, each encrypted
copy proved before anything moves. ``check`` says the state of each store.
A key made while a store stays in plaintext would leave that store
unreadable, so the server makes none then, and moves no store under its
own open connections.

Security derives from the encryption key, not from the presence of
SQLCipher.  An attacker with the key can always read the data
(Kerckhoffs principle).
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import sqlite3
import stat
import sys
import time
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
            # Extract raw bytes if SecureBytes
            raw_key = master_key.as_bytes() if hasattr(master_key, "as_bytes") else master_key
            return _db_key(raw_key)
    except Exception as exc:
        # A held secret wiped at exit is no key: a store opened without one would be opened in plaintext.
        if type(exc).__name__ == "SecretWiped" and type(exc).__module__ == "opti_oignon.process_secrets":
            raise
        logger.debug("Failed to derive DB encryption key: %s", exc)
    return None


def _db_key(master: bytes) -> bytes:
    """The databases' key: HMAC-SHA256(master key, "opti-oignon-sqlcipher-v1"), apart from every other key."""
    import hmac

    return hmac.new(master, b"opti-oignon-sqlcipher-v1", hashlib.sha256).digest()


def _key_to_hex_pragma(key: bytes) -> str:
    """Convert a key to the hex format SQLCipher expects.

    SQLCipher accepts: PRAGMA key = "x'<hex>'";
    """
    return f"\"x'{key.hex()}'\""


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def _read_only_uri(db_path: str) -> str:
    """The URI that opens ``db_path`` read-only: ``mode=ro``, and its shared index read-only when it exists.

    A read-only connection never checkpoints a WAL file, never deletes one
    and never rolls back a hot journal (it refuses instead); with the index
    read-only too, it writes no byte of any of the store's files.
    """
    uri = Path(db_path).absolute().as_uri() + "?mode=ro"
    if Path(db_path + "-shm").exists():
        uri += "&readonly_shm=1"
    return uri


def get_encrypted_connection(
    db_path: str | Path,
    *,
    check_same_thread: bool = True,
    timeout: float = 5.0,
    enforce_encryption: bool | None = None,
    read_only: bool = False,
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
    read_only : bool
        If True, the file is opened read-only (``_read_only_uri``): the key
        and its PRAGMAs apply as usual, and nothing can be written through
        the connection or by its opening and closing.

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

    options = {"check_same_thread": check_same_thread, "timeout": timeout}
    target = db_path
    if read_only:
        target = _read_only_uri(db_path)
        options["uri"] = True

    if SQLCIPHER_AVAILABLE and key:
        # Use SQLCipher encrypted connection
        conn = _sqlcipher_module.connect(target, **options)
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

    conn = sqlite3.connect(target, **options)
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
    """Return encryption status for every store of both data places (``store_files``)."""
    return [get_db_status(p) for p in store_files()]


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


# ---------------------------------------------------------------------------
# Every store, encrypted offline: all of them, or none
# ---------------------------------------------------------------------------
# The places the platform keeps its SQLite stores: the repository's data
# directory (authentication, the audit chain) and the package's
# (conversations, branches, notes, projects, caches, the governor, the rest).
# None reads them where the package lies (``_places``); a contract sets its own.
_STORE_PLACES: Any = None
# The stores the platform keeps outside the places, one by one. None reads
# them from their configuration (``_lone_stores``): the sandbox's audit log.
_LONE_STORES: Any = None
# Below a place, the folder the platform keeps stores in too: the knowledge
# base's. A plugin's own folder is none: its plugin opens its stores without
# the platform's key, in plaintext, and they are left so.
_STORE_FOLDERS = ("rag",)
# The files SQLite keeps beside a database while it is open or recovering.
_SIDECARS = ("-wal", "-shm", "-journal")
# How every SQLite file no cipher wrote begins.
_PLAIN_HEADER = b"SQLite format 3\x00"
# The database ChromaDB keeps in each of its folders, beside its index: in plaintext, as all it keeps.
_CHROMA_FILE = "chroma.sqlite3"
# The environment variables the platform reads a data directory from. When one is set, the command looks below
# it too for what stays in plaintext there: RAG_HOME moves the knowledge base's ChromaDB.
_FOLLOWED = ("RAG_HOME",)
# What an interrupted run leaves beside a store, named by ``_leftovers``: never said as anything else.
_LEFT_BESIDE = (".db.restoring", ".db.encrypting")
# How many entries the command names in a list before it counts the rest.
_SAID_AT_MOST = 10
# The command that encrypts every store, the platform stopped.
ENCRYPT_COMMAND = "python3 -m opti_oignon.db_encryption encrypt"
# The command that moves ChromaDB's stores into the knowledge base's store, the platform stopped
# (``chroma_migration``), and the folders it knows, each with the namespace of the store its collections go to.
MIGRATE_COMMAND = "python3 -m opti_oignon.db_encryption migrate"
_CHROMA_FOLDERS = {"chroma_v2": "knowledge", "chroma_db": "documents", "project_chroma": "projects"}
# Where the system shows each process's command line and open files.
_PROC = Path("/proc")
# The signals that ask the process to end: a run notes them, and stops at its next safe point, everything put back.
_ENDING = ("SIGTERM", "SIGHUP", "SIGINT")
# Where the sandbox keeps its audit log when its configuration names no place.
_SANDBOX_CONFIG = Path(__file__).resolve().parent / "config" / "sandbox.yaml"
_SANDBOX_DEFAULT = "/tmp/opti-oignon-sandboxes"
_SANDBOX_LOG = "sandbox_audit.db"


class EncryptionRefused(RuntimeError):
    """The offline encryption of the stores refused, by name, and changed nothing."""


class EncryptionStuck(RuntimeError):
    """A run that could not put every store back: where each of their files is, said; the key file and the folders
    of plaintext copies kept."""


class _Ended(BaseException):
    """A signal that asked the process to end, raised at a safe point of the run so it puts everything back."""


def _places() -> tuple:
    """The two places of the stores, or the ones a contract set."""
    package = Path(__file__).resolve().parent
    return (_DATA_DIR, package / "data") if _STORE_PLACES is None else tuple(Path(p) for p in _STORE_PLACES)


def _lone_store_paths() -> list:
    """Where the stores outside the places are, whether they hold anything or not: the sandbox's audit log, where its
    manager opens it -- ``audit_db_path`` joined to ``workspace_base``, each read from ``config/sandbox.yaml`` or
    the manager's default --, when that path is absolute; or the ones a contract set. Nothing of the sandbox is
    read. The manager is never imported here: its import builds the manager, which makes the sandbox's folders."""
    if _LONE_STORES is not None:
        return [Path(p) for p in _LONE_STORES]
    configured = _sandbox_logs()[0]
    return [Path(configured)] if os.path.isabs(configured) else []


def _sandbox_logs() -> tuple:
    """(configured, default): where the sandbox's manager opens its audit log -- ``audit_db_path`` joined to
    ``workspace_base`` as ``config/sandbox.yaml`` gives them --, and where it opens it when a value of its
    configuration does not build: its defaults. A path that is not absolute lies below whatever directory the
    platform starts in."""
    try:
        import yaml

        raw = yaml.safe_load(_SANDBOX_CONFIG.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - a configuration that does not read leaves the sandbox's own defaults
        raw = {}
    raw = raw if isinstance(raw, dict) else {}
    configured = os.path.join(str(raw.get("workspace_base") or _SANDBOX_DEFAULT),
                              str(raw.get("audit_db_path") or _SANDBOX_LOG))
    return configured, os.path.join(_SANDBOX_DEFAULT, _SANDBOX_LOG)


def _sandbox_problems() -> list:
    """What keeps the command from taking the sandbox's audit log, each with what to do: a log configured below
    whatever directory the platform starts in, which the command cannot find; a log at the manager's default
    path while the configuration names another, which the manager opens only when its configuration does not
    build -- taken by neither, as anyone may make that path under /tmp. None when a contract set the stores."""
    if _LONE_STORES is not None:
        return []
    configured, default = _sandbox_logs()
    problems = []
    if not os.path.isabs(configured):
        problems.append(f"the sandbox's audit log is configured below whatever directory the platform starts in "
                        f"({configured}): set an absolute workspace_base in config/sandbox.yaml")
    # A relative log lies below wherever the platform starts, which is not known here: never resolved from here.
    elsewhere = not os.path.isabs(configured) or os.path.realpath(default) != os.path.realpath(configured)
    if elsewhere and _holds_something(Path(default)):
        problems.append(f"{default} holds a sandbox audit log at the manager's default path while its configuration "
                        f"names {configured}: the manager opens it only when its configuration does not build; "
                        + _sandbox_advice(configured))
    return problems


def _sandbox_advice(configured: str) -> str:
    """What to do with a log at the manager's default path, by what the command can see of the configured one:
    there, so this one is stale; not there, so it may move; or not to be looked at -- relative, or its folder not
    searchable --, so nothing is said of it and nothing is moved before looking."""
    folder = os.path.dirname(configured)
    if not os.path.isabs(configured) or (os.path.lexists(folder) and not os.access(folder, os.X_OK)):
        return "where the configured log lies cannot be looked at from here: look at both before moving anything"
    if _holds_something(Path(configured)):
        return "the configured log is there too, so this one is stale: keep what you need of it, then delete it"
    return "move it there, as no log is there yet"


def _unreadable_dirs(places: Any) -> list:
    """The directories the stores are taken from that are there and cannot be read or searched: a place, a store
    folder, the folder of a store outside the places. Their stores could not be taken, so no run is done."""
    dirs = [Path(folder) for folder in _store_dirs(places)]
    if places is None:
        dirs += [path.parent for path in _lone_store_paths() if path.parent not in dirs]
    unreadable = []
    for folder in dirs:
        try:
            if not os.path.lexists(folder):
                continue
            # Its names read, and its files looked at: both, or its stores stay out of sight.
            if not os.access(folder, os.R_OK | os.X_OK):
                unreadable.append(folder)
                continue
            os.listdir(folder)
        except OSError:
            unreadable.append(folder)
    return unreadable


def _lone_stores() -> list:
    """The stores outside the places that hold something (``_lone_store_paths``)."""
    return [path for path in _lone_store_paths() if _holds_something(path)]


def _holds_something(path: Path) -> bool:
    """Whether ``path`` is a symbolic link, or a file that holds something; False when it cannot be looked at
    (``_unreadable_dirs`` says the folder)."""
    try:
        return path.is_symlink() or (path.is_file() and path.stat().st_size > 0)
    except OSError:
        return False


def store_files(places: Any = None) -> list[Path]:
    """Every store of the platform: the ``*.db`` entries that hold something, a symbolic link included, of each
    place and of its store folder (``_STORE_FOLDERS``), place by place, sorted; then the stores outside the places
    (``_lone_stores``) when the places are the platform's own."""
    found: list[Path] = []
    for place in _places() if places is None else places:
        place = Path(place)
        if place.is_dir():
            for pattern in ("*.db", *(f"{folder}/*.db" for folder in _STORE_FOLDERS)):
                found += [path for path in sorted(place.glob(pattern)) if _holds_something(path)]
    if places is None:
        found += [path for path in _lone_stores() if path not in found]
    return found


def other_databases(places: Any = None) -> list[Path]:
    """The ``*.db`` files below the places that hold something and are no store this command takes -- a plugin's
    own, read in plaintext by its plugin; a file a user added to a project --, the folders of plaintext copies
    aside: said, and left as they are."""
    taken = {os.path.realpath(path) for path in store_files(places)}
    beside = {os.path.realpath(folder) for folder in _store_dirs(places)}
    others: list[Path] = []
    seen: set = set()
    for root in _walk_roots(places):
        if os.path.isdir(root):
            for path in sorted(root.rglob("*.db")):
                real = os.path.realpath(path)
                if real not in taken and real not in seen and _holds_something(path) and not _in_kept_folder(
                        Path(real), beside):
                    seen.add(real)
                    others.append(path)
    return others


def _walk_roots(places: Any) -> list:
    """The places, then each store folder that is a symbolic link to a directory: walked from its own top, as
    its stores are taken through it and a walk of the place never follows a link."""
    places = [Path(place) for place in (_places() if places is None else places)]
    return places + [place / folder for place in places for folder in _STORE_FOLDERS
                     if os.path.islink(place / folder) and os.path.isdir(place / folder)]


def _in_kept_folder(path: Path, beside: set) -> bool:
    """Whether ``path``, a real path, lies in a folder of plaintext copies a run kept beside the stores, whose
    real paths are ``beside``."""
    return any(folder.name.startswith("plaintext-") and str(folder.parent) in beside for folder in path.parents)


def plaintext_others(places: Any = None) -> list[Path]:
    """The files below the places that begin as a SQLite file no cipher wrote and are neither a ``*.db`` nor what an
    interrupted run left (``other_databases`` and ``_leftovers`` say those): a vector store's -- ChromaDB keeps its
    texts and vectors in plaintext --, a database under another name. Said, never encrypted here (``_scan``)."""
    return _scan(places)[0]


def unchecked_entries(places: Any = None) -> list:
    """What ``_scan`` could not check below the places, each with why: a symbolic link, never followed, or an entry
    that could not be read."""
    return _scan(places)[1]


def _scan(places: Any) -> tuple:
    """(``plaintext_others``, ``unchecked_entries``), in one walk of the places -- each store folder followed even
    when it is a symbolic link, as the stores are taken through it -- and, when the places are the platform's own,
    of each directory an absolute ``_FOLLOWED`` variable names, where a ``*.db`` is no store and is named too. A
    followed directory that is, by its real path, a place, a store folder or below one is no new root, and a walk
    from one above a place never enters it again. The folders of plaintext copies beside the stores are passed over;
    a folder of that name anywhere else is walked. Only regular files are read; each file is said once; what cannot
    be looked at is said, never raised (``_unreadable_dirs`` says a folder of stores)."""
    own = places is None
    beside = set(_store_dirs(places))
    places = [Path(place) for place in (_places() if own else places)]
    walked = [os.path.realpath(root) for root in _walk_roots(places)]

    def taken(path: Path) -> bool:
        real = os.path.realpath(path)
        return any(real == root or real.startswith(root.rstrip(os.sep) + os.sep) for root in walked)

    roots = [(root, False) for root in _walk_roots(places)]
    roots += [(path, True) for path in (_followed_dirs() if own else []) if not taken(path)]
    reached = {os.path.realpath(root) for root, followed in roots if followed}
    found: list = []
    unchecked: list = []
    for root, followed in roots:
        try:
            os.stat(root)
        except FileNotFoundError:
            continue
        except OSError:
            # A followed directory that cannot be looked at is said; a place, ``_unreadable_dirs`` says.
            unchecked += [(root, "could not be read")] if followed else []
            continue
        if not os.path.isdir(root) or not (followed or os.access(root, os.R_OK | os.X_OK)):
            continue
        for top, dirs, files in os.walk(root, onerror=lambda error: unchecked.append(
                (Path(error.filename), "could not be read"))):
            here = Path(top)
            links = [name for name in sorted(dirs) if os.path.islink(here / name)]
            # A link is said unless what it leads to is walked anyway: a store folder, a place, a followed directory.
            unchecked += [(here / name, "a symbolic link, never followed") for name in links
                          if not (here in places and name in _STORE_FOLDERS) and not taken(here / name)
                          and os.path.realpath(here / name) not in reached]
            # A folder of stores that cannot be read is said by ``_unreadable_dirs``, never twice.
            dirs[:] = sorted(name for name in dirs if name not in links and not (followed and taken(here / name))
                             and not (here in beside and name.startswith("plaintext-"))
                             and not (here / name in beside and not os.access(here / name, os.R_OK | os.X_OK)))
            for name in sorted(files):
                if name.endswith(_LEFT_BESIDE) or (name.endswith(".db") and not followed):
                    continue
                state = _plain_regular(here / name)
                if state is None:
                    unchecked.append((here / name, "a symbolic link, never followed"
                                      if os.path.islink(here / name) else "could not be read"))
                elif state:
                    found.append(here / name)
    return list(dict.fromkeys(found)), list(dict.fromkeys(unchecked))


def _followed_dirs() -> list:
    """The absolute directories a ``_FOLLOWED`` variable moves the platform's data to; a relative one is said by
    ``check``, never walked."""
    return [Path(os.environ[name]) for name in _FOLLOWED
            if os.environ.get(name) and Path(os.environ[name]).is_absolute()]


def _relative_followed() -> list:
    """The ``_FOLLOWED`` variables set to a relative path -- an empty one included, which the platform reads as the
    directory it starts in --, each with its value (an empty one said ``''``): below whatever directory the
    platform starts in, never walked here."""
    return [(name, os.environ[name] or "''") for name in _FOLLOWED
            if name in os.environ and not Path(os.environ[name]).is_absolute()]


def _plain_regular(path: Path) -> Any:
    """Whether ``path`` begins as a SQLite file no cipher wrote: True or False for a regular file read, False for
    what is no regular file nor a link (a pipe is never opened) or is gone, None for a symbolic link -- never
    followed -- or a file that could not be read."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return False
    except OSError:
        return None
    if stat.S_ISLNK(mode):
        return None
    if not stat.S_ISREG(mode):
        return False
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return False
    except OSError:
        return None
    try:
        return os.read(fd, len(_PLAIN_HEADER)) == _PLAIN_HEADER
    except OSError:
        return None
    finally:
        os.close(fd)


def _say_unchecked(unchecked: list) -> None:
    """Say what could not be checked below the places, the first ``_SAID_AT_MOST`` by name."""
    for path, why in unchecked[:_SAID_AT_MOST]:
        print(f"{path}: not checked, {why}")
    if len(unchecked) > _SAID_AT_MOST:
        print(f"{len(unchecked) - _SAID_AT_MOST} more entries below the places not checked")


def _plaintext_said(path: Path) -> str:
    """The line that says a file of ``plaintext_others``."""
    if path.name == _CHROMA_FILE:
        return (f"{path}: a ChromaDB vector store, its texts and vectors in plaintext: ChromaDB has no encryption at "
                "rest, and this command does not encrypt it")
    return f"{path}: a database in plaintext that is no store of the platform: this command does not encrypt it"


def _store_dirs(places: Any) -> list:
    """The directories that hold stores: each place and its store folders, and, when the places are the platform's
    own, the directories of the stores outside them."""
    dirs = []
    for place in _places() if places is None else places:
        place = Path(place)
        dirs += [place, *(place / folder for folder in _STORE_FOLDERS)]
    if places is None:
        dirs += [path.parent for path in _lone_stores()]
    return list(dict.fromkeys(dirs))


def plaintext_folders(places: Any = None) -> list[Path]:
    """The folders of plaintext copies runs kept beside the stores, each holding stores as they were before they
    were encrypted: to delete once the platform reads its stores again and the key is kept safe elsewhere."""
    return [path for folder in _store_dirs(places) for path in sorted(folder.glob("plaintext-*")) if path.is_dir()]


def _same(one: Path, other: Path) -> bool:
    """Whether ``one`` and ``other`` are two names of one regular file -- never through a symbolic link."""
    try:
        first, second = os.lstat(one), os.lstat(other)
    except OSError:
        return False
    return (stat.S_ISREG(first.st_mode) and stat.S_ISREG(second.st_mode)
            and (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino))


def _plain(path: Path) -> bool:
    """Whether ``path`` begins as a SQLite file no cipher wrote."""
    try:
        with open(path, "rb") as handle:
            return handle.read(len(_PLAIN_HEADER)) == _PLAIN_HEADER
    except OSError:
        return False


def plaintext_stores(places: Any = None) -> list[Path]:
    """The stores kept in plaintext: a key made now would leave each of them unreadable."""
    return [path for path in store_files(places) if not path.is_symlink() and _plain(path)]


def _ancestors(proc: Path) -> set:
    """This process's ancestors: its parent, and each one's parent while ``proc`` shows it."""
    found: set = set()
    pid = os.getppid()
    while pid > 1 and pid not in found:
        found.add(pid)
        try:
            status = (proc / str(pid) / "status").read_text(encoding="utf-8", errors="replace")
            pid = int(next(line for line in status.splitlines() if line.startswith("PPid:")).split()[1])
        except (OSError, StopIteration, IndexError, ValueError):
            break
    return found


def store_holders(paths: Any, *, proc: Any = None) -> dict[int, str]:
    """The processes other than this one that run the platform, or hold one of ``paths`` -- or a file SQLite keeps
    beside it -- open: ``{pid: why}``.

    Read from ``proc`` (``/proc``); refused by name (``EncryptionRefused``)
    on a system that shows no open files per process. A process of another
    user cannot be read, and is not counted: the platform runs as its user.
    An ancestor of this process whose command line holds this command --
    the shell or the tool that started it -- is counted only for what it
    holds open.
    """
    proc = Path(_PROC if proc is None else proc)
    if not proc.is_dir():
        raise EncryptionRefused(f"this system shows no open files per process ({proc}): nothing was changed")
    watched = set()
    for path in paths:
        resolved = os.path.realpath(path)
        watched.update({resolved, *(resolved + side for side in _SIDECARS)})
    elders = _ancestors(proc)
    found: dict[int, str] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        why = "" if int(entry.name) in elders and _runs_this_command(entry) else _runs_the_platform(entry)
        try:
            for fd in (entry / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.removesuffix(" (deleted)") in watched:
                    why = f"holds {Path(target).name} open"
                    break
        except OSError:
            pass
        if why:
            found[int(entry.name)] = why
    return found


def _runs_this_command(entry: Path) -> bool:
    """Whether the process at ``entry`` was started with this command in its command line."""
    try:
        return b"db_encryption" in (entry / "cmdline").read_bytes()
    except OSError:
        return False


def _runs_the_platform(entry: Path) -> str:
    """Why the process at ``entry`` runs the platform, read from its command line; "" when it does not."""
    try:
        argv = [part.decode("utf-8", "replace") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
    except OSError:
        return ""
    for part in argv:
        if "opti_oignon" in part or Path(part).name == "oo":
            return f"runs the platform ({Path(part).name})"
    return ""


def _keyed(path: Any, key: bytes) -> Any:
    """A SQLCipher connection to ``path`` under ``key``, with the PRAGMAs every store is opened with."""
    conn = _sqlcipher_module.connect(str(path))
    conn.execute(f"PRAGMA key = {_key_to_hex_pragma(key)}")
    conn.execute(f"PRAGMA cipher_page_size = {_CIPHER_PAGE_SIZE}")
    conn.execute(f"PRAGMA cipher_hmac_algorithm = {_CIPHER_HMAC_ALGORITHM}")
    conn.execute(f"PRAGMA kdf_iter = {_KDF_ITER}")
    return conn


def _opens_under(path: Path, key: Any) -> bool:
    """Whether the store at ``path`` opens under ``key`` and reads whole."""
    conn = None
    try:
        conn = _keyed(path, key)
        return conn.execute("PRAGMA quick_check").fetchall() == [("ok",)]
    except Exception:  # noqa: BLE001 - a store that does not open under the key is no store of this key
        return False
    finally:
        if conn is not None:
            conn.close()


def _shape(conn: Any) -> tuple:
    """What a database holds, as digests: its objects (kind, name, table, SQL); each table that is not virtual by
    its rows' count and the SHA-256 of its rows in order; its user_version, application_id and auto_vacuum."""
    objects = conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall()
    tables = []
    for kind, name, _table, sql in objects:
        text = str(sql or "").upper()
        if kind != "table" or text.startswith("CREATE VIRTUAL TABLE"):
            continue
        order = "" if "WITHOUT ROWID" in text else " ORDER BY rowid"
        digest, count = hashlib.sha256(), 0
        for row in conn.execute('SELECT * FROM "' + name.replace('"', '""') + '"' + order):
            digest.update(repr(tuple(row)).encode("utf-8", "surrogatepass"))
            count += 1
        tables.append((name, count, digest.hexdigest()))
    return (tuple(tuple(row) for row in objects), tuple(tables),
            *(conn.execute(f"PRAGMA {name}").fetchone()[0] for name in ("user_version", "application_id",
                                                                         "auto_vacuum")))


def _encrypted_copy(store: Path, copy: Path, key: bytes) -> None:
    """Write ``copy``: ``store`` encrypted under ``key``, proved to hold what the store holds, with its mode,
    synced to the disk.

    The store's write-ahead log is folded into it first. Raises
    ``EncryptionRefused`` naming the store and what differs; never a word
    of what it holds.
    """
    plain = sqlite3.connect(str(store))
    try:
        plain.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        held = _shape(plain) if plain.execute("PRAGMA integrity_check").fetchall() == [("ok",)] else None
    except sqlite3.DatabaseError:
        held = None
    finally:
        plain.close()
    if held is None:
        raise EncryptionRefused(f"{store.name} does not pass its own integrity check: nothing was changed")
    conn = _keyed(copy, key)
    try:
        conn.execute(f"PRAGMA auto_vacuum = {int(held[-1])}")
        conn.execute("ATTACH DATABASE ? AS plaintext KEY ''", (str(store),))
        conn.execute("SELECT sqlcipher_export('main', 'plaintext')")
        version = conn.execute("PRAGMA plaintext.user_version").fetchone()[0]
        application = conn.execute("PRAGMA plaintext.application_id").fetchone()[0]
        conn.execute("DETACH DATABASE plaintext")
        conn.execute(f"PRAGMA user_version = {int(version)}")
        conn.execute(f"PRAGMA application_id = {int(application)}")
        conn.commit()
    finally:
        conn.close()
    conn = _keyed(copy, key)
    try:
        whole = conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        whole = whole and not conn.execute("PRAGMA cipher_integrity_check").fetchall()
        same = whole and _shape(conn) == held
    finally:
        conn.close()
    if not same:
        raise EncryptionRefused(f"the encrypted copy of {store.name} does not hold what the store holds: nothing was "
                                "changed")
    os.chmod(copy, os.stat(store).st_mode & 0o7777)
    fd = os.open(copy, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _loaded_key() -> Any:
    """The databases' key from the master key that loads -- OPTI_ENCRYPTION_KEY, or the key file --, or None when
    none does. A key file that does not load here is refused by name: it is never replaced."""
    from opti_oignon import encryption

    refused = _environment_key_refused()
    if refused is not None:
        raise refused
    if encryption.get_encryption_key() is None:
        if encryption.keyfile_present():
            raise EncryptionRefused("a key file exists and does not load here (if it is wrapped under a passphrase, "
                                    "set OPTI_KEYFILE_PASSPHRASE); it is never replaced: nothing was changed")
        return None
    key = _get_db_encryption_key()
    if key is None:
        raise EncryptionRefused("the master key that loads gives no database key: nothing was changed")
    return key


def _environment_key_refused() -> Any:
    """The refusal of an OPTI_ENCRYPTION_KEY that is set and is no key -- a base64url-encoded 32-byte key --, or
    None. The platform passes over such a variable to the key file, so the stores would be keyed by another key
    than the one the variable seems to set, and the command would say the key is the variable's."""
    from opti_oignon import encryption

    value = encryption._secret(encryption._ENV_KEY_NAME)
    if not value:
        return None
    try:
        if len(base64.urlsafe_b64decode(value)) == encryption._KEY_SIZE:
            return None
    except (ValueError, TypeError):
        pass
    return EncryptionRefused("OPTI_ENCRYPTION_KEY is set and is no key the platform reads (32 bytes, base64-encoded "
                             "with its padding: 44 characters ending in a single '='): unset it, or set it to the "
                             "key the stores are encrypted with: nothing was changed")


def _key_file_key() -> Any:
    """The databases' key from the key file alone, or None: what the platform opens the stores with when it passes
    over an OPTI_ENCRYPTION_KEY that is no key."""
    from opti_oignon import encryption

    try:
        master, _salt, _kdf = encryption.load_keyfile()
    except Exception:  # noqa: BLE001 - a key file that does not load gives no key
        return None
    raw = master.as_bytes() if hasattr(master, "as_bytes") else master
    return _db_key(raw) if raw and len(raw) == encryption._KEY_SIZE else None


def _write_master(master: bytes, key: bytes, lock_before: Any = None) -> None:
    """Write ``master`` to a new key file -- wrapped under OPTI_KEYFILE_PASSPHRASE when it is set, as setup writes
    one -- and prove it loads to ``key``; refused by name, the file removed and the mode's lockfile put back to
    ``lock_before``, when it does not. A key file or a lockfile that cannot be written refuses, by name."""
    from opti_oignon import encryption

    passphrase = encryption._secret(encryption._ENV_KEYFILE_PASS)
    try:
        encryption.save_keyfile(master, kdf_name="random", passphrase=passphrase, allow_unprotected=passphrase is None)
    except encryption.KeyfileExists:
        raise EncryptionRefused("a key file appeared while this one was made; it is never replaced: nothing was "
                                "changed") from None
    except OSError as exc:
        raise EncryptionRefused(f"{exc}: nothing was changed") from None
    if _get_db_encryption_key() != key:
        try:
            _unmake_key(lock_before, master)
        except OSError as exc:
            raise EncryptionRefused(f"the key does not load once written, and its key file {_key_file()} could not "
                                    f"be removed ({type(exc).__name__}): no store is encrypted under it: delete it"
                                    ) from None
        raise EncryptionRefused("the key does not load once written: nothing was changed")


def _unmake_key(lock_before: Any = None, master: Any = None) -> None:
    """Remove the key file this run made: no store is encrypted under it. Once it is gone, the security mode's
    lockfile, signed under ``master`` as the key file was made (``encryption.save_keyfile``), is put back to
    ``lock_before`` (``_lock_snapshot``) -- unless another maker signed it since; a key file that cannot be removed
    keeps the lockfile signed under it, and raises."""
    from opti_oignon import encryption

    try:
        os.unlink(encryption._DEFAULT_KEYFILE)
    except FileNotFoundError:
        pass
    if lock_before is not None:
        from opti_oignon import security_mode

        try:
            security_mode.put_lock_back(lock_before, master)
        except OSError as exc:
            # Said, never raised over the refusal that undid the key: with no key file, a lockfile signed
            # under the key is verified by nothing.
            logger.error("the security mode's lockfile could not be put back as it was: %s", exc)


def _lock_snapshot() -> Any:
    """The security mode's lockfile as it is now (``security_mode.lock_snapshot``); None where the mode module does
    not import, or keeps no lockfile (a stand-in): no mode to keep."""
    try:
        from opti_oignon import security_mode
    except ImportError:
        return None
    snapshot = getattr(security_mode, "lock_snapshot", None)
    return None if snapshot is None else snapshot()


def _mode_line() -> str:
    """The security mode the platform reads from these files, and why, as ``check`` says it
    (``security_mode.mode_report``)."""
    try:
        from opti_oignon import security_mode

        report = security_mode.mode_report()
    except Exception as exc:  # noqa: BLE001 - a mode that does not read is said, never a crash of check
        return f"security mode: not read here ({type(exc).__name__}: {exc})"
    lock, failure, said = report["lock"], report["failure"], report["yaml_said"]
    # security.yaml holding no plain mode is said wherever the line would not say it.
    held = "" if said in ("daily", "bulbe") else f"; security.yaml says {said}"
    if lock is None:
        source = "security.yaml" if said in ("daily", "bulbe") else f"security.yaml, which says {said}"
        if report["lock_unread"]:
            return f"security mode: {report['mode']}, from {source}: its lock file {report['lock_unread']}, read as none"
        return f"security mode: {report['mode']}, from {source}: no lock file"
    if failure == "lockfile_hmac_failure":
        changed = report["key_changed"]
        return (f"security mode: bulbe, failing secure: its lock file ({lock['MODE'] or 'no mode'}, signed "
                f"{_when(lock['TIMESTAMP'])} by its own account) does not verify under the key that loads (key file "
                f"changed {_when(changed) if changed is not None else 'at a time not read here'}): it was signed under "
                "another key, or written without one; the platform records lockfile_hmac_failure at each start until "
                f"it is signed again under this key, which leaving Bulbe through the downgrade ceremony does{held}")
    if failure == "lockfile_unverifiable":
        return f"security mode: bulbe, failing secure: {report['key']}: its lock file cannot be verified{held}"
    if failure == "security_mode_mismatch":
        return (f"security mode: bulbe, failing secure: security.yaml says {said}, its lock file "
                f"{lock['MODE'] or 'no mode'}")
    if report["verified"]:
        return f"security mode: {report['mode']}; its lock file verifies under the key that loads{held}"
    return f"security mode: {report['mode']}; its lock file is not verified: no key file{held}"


def _when(stamp: Any) -> str:
    """A time in seconds since the epoch, as ``check`` says it (UTC, to the minute); one that is no time, said so."""
    try:
        return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(float(stamp)))
    except (TypeError, ValueError, OverflowError, OSError):
        return f"at a time it does not give ({stamp!r})"


def _key_file() -> Path:
    from opti_oignon import encryption

    return Path(encryption._DEFAULT_KEYFILE)


def _fresh_folder(place: Path, name: str) -> Path:
    """A new folder, 0700, in ``place``: ``name``, or ``name`` with the first number that is free."""
    for n in range(1, 1000):
        folder = place / (name if n == 1 else f"{name}-{n}")
        try:
            folder.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return folder
    raise EncryptionRefused(f"no free folder named {name} in {place}: nothing was changed")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def encrypt_all_stores(*, places: Any = None, proc: Any = None) -> dict[str, Any]:
    """Encrypt every store kept in plaintext, the platform stopped: all of them, or none, even when the run dies.

    Refused by name, nothing changed (``EncryptionRefused``) -- but each
    plaintext store's write-ahead log folded into it: SQLCipher missing;
    another run under way; a process that runs the platform or holds a
    store, or a file SQLite keeps beside one (``store_holders``); a store
    that is a symbolic link; what an interrupted run left that is not
    plainly a copy to make again (``_leftovers``); a key file that does not
    load here; a store that no cipher left in plaintext and the key does
    not open.

    The key is the one that loads, or a new master key held in memory:
    each plaintext store is copied, encrypted under it, beside itself, and
    every copy is proved before anything moves -- its integrity checked,
    its objects, each table's rows in order and their digest, its
    user_version, application_id and auto_vacuum those of the store, its
    mode the store's. Then the holders are read again, a new key is
    written to its key file and proved to load, and each store is switched
    (``_switch``): its plaintext linked into a folder of plaintext copies
    beside it, its encrypted copy taking its name in one rename, so a
    store's name is never missing even when the process is killed.

    SIGTERM, SIGHUP and SIGINT are noted -- and said at once on standard
    error --, never raised where they land: the run stops at its next safe
    point -- before a copy, before the processes are read again, before
    the key is written, before a store is switched -- and puts back every
    store it switched, removing the copies, the folders and the key file
    it wrote; one that lands once the last store's switch has begun lets
    the run finish. A failed step is put back the same way. A store is put
    back before its plaintext link goes, and a store that cannot be put
    back keeps its plaintext, its folder and the key file: the run raises
    ``EncryptionStuck`` naming where each of its files is. A run killed outright leaves every
    store whole, plaintext or encrypted under the key file: the next run
    finishes it, or first names what the killed one left and what to do
    with it (``_leftovers``).

    Returns what was done: the stores encrypted, those already encrypted,
    the folders of plaintext copies this run kept and those earlier runs
    kept (each to delete once the platform reads its stores again, and the
    key kept safe elsewhere), whether the key was made, where it is kept
    and how, the places, the other databases below them, left as they
    are, the directories that could not be synced, and the signal that
    came after the last safe point, if one did.
    """
    import signal

    if not SQLCIPHER_AVAILABLE:
        raise EncryptionRefused("SQLCipher is not installed (pip install sqlcipher3): nothing was changed")
    first = (_places() if places is None else tuple(Path(p) for p in places))[0]
    lock = _alone(first)
    noted: list = []
    previous = {}

    notice = _notice_fd()

    def note(signum, _frame):
        first = not noted
        noted.append(signum)
        if first and notice is not None:
            try:
                os.write(notice, f"signal {signum} noted: the run stops at its next safe point and puts everything "
                                 "back, or finishes if none is left\n".encode())
            except OSError:
                pass

    try:
        for name in _ENDING:
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                if signal.getsignal(signum) is signal.SIG_IGN:
                    continue
                previous[signum] = signal.signal(signum, note)
            except ValueError:
                pass
        return _encrypt(places, proc, noted)
    finally:
        try:
            for signum, handler in previous.items():
                signal.signal(signum, signal.SIG_DFL if handler is None else handler)
        finally:
            try:
                if notice is not None:
                    os.close(notice)
            finally:
                lock.close()


def _notice_fd() -> Any:
    """A descriptor of its own on standard error, for the one notice a signal handler writes; None when the process
    has no standard error -- the number 2 may then be any file it opened."""
    try:
        return os.dup(sys.stderr.fileno()) if sys.stderr is not None else None
    except (OSError, ValueError, AttributeError):
        return None


class _Held:
    """A lock held on a directory, released when closed."""

    def __init__(self, fd: int) -> None:
        self.fd = fd

    def close(self) -> None:
        os.close(self.fd)


def _alone(place: Path) -> Any:
    """Hold the run's lock on ``place`` itself -- no file is made for it --; refused by name when another run
    holds it, or when no lock can be held there."""
    import fcntl

    try:
        fd = os.open(place, os.O_RDONLY)
    except OSError as exc:
        raise EncryptionRefused(f"the run's lock cannot be held on {place} ({type(exc).__name__}): nothing was "
                                "changed") from None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise EncryptionRefused("another encryption of the stores is under way: nothing was changed") from None
    except OSError as exc:
        os.close(fd)
        raise EncryptionRefused(f"the run's lock cannot be held on {place} ({type(exc).__name__}): nothing was "
                                "changed") from None
    return _Held(fd)


def _stop_if_asked(noted: list) -> None:
    """A safe point: when a signal asked the process to end, stop here."""
    if noted:
        raise _Ended(noted[0])


def _switch(store: Path, copy: Path, folder: Path, records: list) -> None:
    """Put ``copy`` in ``store``'s place, ``store`` kept in ``folder``: the store's name is never missing on the
    way -- the plaintext is linked into the folder first, the copy takes the name in one rename. What was done is
    on its record, in ``records``, before and after each step."""
    record = {"store": store, "kept": folder / store.name, "linked": False, "sides": [], "switched": False}
    records.append(record)
    os.link(store, record["kept"])
    record["linked"] = True
    for side in _SIDECARS:
        beside = Path(str(store) + side)
        if os.path.lexists(beside):
            spare = folder / beside.name
            os.link(beside, spare)
            record["sides"].append([beside, spare, False])
            os.unlink(beside)
            record["sides"][-1][2] = True
    os.replace(copy, store)
    record["switched"] = True


def _roll_back(records: list) -> list:
    """Put every store of ``records`` back, the last switched first; where each file is that could not be.

    A switched store is put back by linking its plaintext to a name of its
    own and renaming that over it. Only once the store is that plaintext
    again do the links in its folder go; a store that cannot be put back
    keeps them.
    """
    stuck = []
    for record in reversed(records):
        store, kept = record["store"], record["kept"]
        if record["switched"]:
            try:
                restoring = store.with_name(store.name + ".restoring")
                if os.path.lexists(restoring):
                    os.unlink(restoring)
                os.link(kept, restoring)
                os.replace(restoring, store)
            except OSError as exc:
                stuck.append(f"{store} is encrypted, its plaintext is {kept} ({type(exc).__name__})")
                continue
        if record["linked"]:
            try:
                plain_again = os.path.samefile(store, kept)
            except OSError:
                plain_again = False
            if not plain_again:
                stuck.append(f"{store}: its plaintext is {kept}")
                continue
        for beside, spare, moved in record["sides"]:
            try:
                if moved and not os.path.lexists(beside):
                    os.link(spare, beside)
                if os.path.samefile(beside, spare):
                    os.unlink(spare)
            except OSError as exc:
                stuck.append(f"{beside} is {spare} ({type(exc).__name__})")
        if record["linked"]:
            try:
                os.unlink(kept)
            except FileNotFoundError:
                pass
            except OSError as exc:
                stuck.append(f"{kept} stays ({type(exc).__name__})")
    return stuck


def _leftovers(stores: list, places: Any) -> list:
    """What an interrupted run left that the next one must not pass over, each named with what to do about it: a
    plaintext store half put back, a copy whose store is missing, a key file's temp. Every writer of the key file
    (``encryption._write_keyfile``) puts it in place before the key is used and never replaces it, so a temp that is
    not the key file holds no key in use -- unless the key file is a symbolic link to it; a rotation that renamed a
    temp into place would end this."""
    found = []
    for folder in _store_dirs(places):
        for path in sorted(folder.glob("*.db.restoring")):
            name = path.name[:-len(".restoring")]
            twin = next((sub / name for sub in sorted(folder.glob("plaintext-*")) if _same(sub / name, path)), None)
            found.append(f"{path} (a second name for {twin}, left by a run killed while putting {name} back: delete "
                         "it)" if twin is not None else
                         f"{path} ({name}'s plaintext, left by a run killed while putting {name} back, and in no "
                         f"plaintext-* folder beside it: move it into a new one, as {name})")
        for path in sorted(folder.glob("*.db.encrypting")):
            store = path.with_name(path.name[:-len(".encrypting")])
            if store not in stores:
                found.append(f"{path} (an encrypted copy of {store.name}, which is missing: put {store.name} back, or "
                             "delete the copy if it was removed on purpose)")
    key = _key_file()
    for path in sorted(key.parent.glob("." + key.name + ".tmp-*")):
        if key.is_symlink():
            found.append(f"{path} (a key file's temp, and the key file is a symbolic link: if it leads here, this "
                         "holds the key in use; keep it)")
        else:
            found.append(f"{path} (a second name for the key file, left by a write killed once it was in place: "
                         "delete it)" if _same(path, key) else
                         f"{path} (left by a write of the key file killed before it was in place: nothing was "
                         "encrypted under it; delete it)")
    return found


def _encrypt(places: Any, proc: Any, noted: list) -> dict[str, Any]:
    stores = store_files(places)
    linked = [path.name for path in stores if path.is_symlink()]
    if linked:
        raise EncryptionRefused(f"store(s) {', '.join(linked)} are symbolic links: nothing was changed")
    left = _leftovers(stores, places)
    if left:
        raise EncryptionRefused("an interrupted run left " + "; ".join(left) + "; then run the command again: "
                                "nothing was changed")
    unreadable = _unreadable_dirs(places)
    if unreadable:
        raise EncryptionRefused(", ".join(str(folder) for folder in unreadable) + " cannot be read: the stores in "
                                "it cannot be taken: nothing was changed")
    problems = _sandbox_problems() if places is None else []
    if problems:
        raise EncryptionRefused("; ".join(problems) + "; then run the command again: nothing was changed")
    earlier = plaintext_folders(places)
    alone = [str(path) for path in other_databases(places)]
    others, unchecked = _scan(places)
    held = store_holders(stores, proc=proc)
    if held:
        raise EncryptionRefused("stop every process of the platform first: " + "; ".join(
            f"process {pid} {why}" for pid, why in sorted(held.items())) + ": nothing was changed")
    plain = [path for path in stores if _plain(path)]
    key = _loaded_key()
    if key is None and len(plain) < len(stores):
        raise EncryptionRefused("store(s) " + ", ".join(path.name for path in stores if path not in plain)
                                + " are encrypted under a key that does not load here: nothing was changed")
    unread = [path.name for path in stores if path not in plain and not _opens_under(path, key)]
    if unread:
        raise EncryptionRefused(f"store(s) {', '.join(unread)} do not open under the key that loads here: nothing "
                                "was changed")
    master = None
    if key is None and plain:
        from opti_oignon import encryption

        master = encryption.generate_key()
        key = _db_key(master)
    copies: list = []
    current = None
    try:
        for store in plain:
            _stop_if_asked(noted)
            current = store
            copy = store.with_name(store.name + ".encrypting")
            if os.path.lexists(copy):
                os.unlink(copy)
            copies.append((store, copy))
            _encrypted_copy(store, copy, key)
        current = None
        _stop_if_asked(noted)
        held = store_holders(stores, proc=proc)
        if held:
            raise EncryptionRefused("a process of the platform started meanwhile: " + "; ".join(
                f"process {pid} {why}" for pid, why in sorted(held.items())) + ": nothing was changed")
    except BaseException as exc:
        _drop(copies)  # copies that took no store's place: what it could not remove is harmless
        if isinstance(exc, _Ended):
            raise EncryptionRefused(f"stopped by signal {exc.args[0]} before any store was switched: nothing was "
                                    "changed") from exc
        if isinstance(exc, Exception) and not isinstance(exc, EncryptionRefused):
            named = (f"{current.name} could not be copied encrypted" if current is not None else
                     "the processes could not be read again")
            raise EncryptionRefused(f"{named} ({type(exc).__name__}): nothing was changed") from exc
        raise
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    records: list = []
    folders: dict[Path, Path] = {}
    written = False
    lock_before = None
    try:
        if master is not None:
            _stop_if_asked(noted)
            lock_before = _lock_snapshot()
            _write_master(master, key, lock_before)
            written = True
        for store, copy in copies:
            _stop_if_asked(noted)
            if store.parent not in folders:
                folders[store.parent] = _fresh_folder(store.parent, f"plaintext-{stamp}")
            _switch(store, copy, folders[store.parent], records)
    except BaseException as exc:
        stuck = _roll_back(records)
        stays = _drop(copies)
        if stuck:
            encrypted = any(" is encrypted, " in item for item in stuck)
            raise EncryptionStuck("the stores could not all be put back where they were: " + "; ".join(stuck + stays)
                                  + "; the folders and the key file stay" + (", a store is encrypted under that key"
                                                                            if encrypted else "")) from exc
        for folder in folders.values():
            try:
                folder.rmdir()
            except OSError:
                pass
        changed = "nothing was changed"
        if written:
            try:
                _unmake_key(lock_before, master)
            except OSError as removal:
                changed = "nothing was changed but the key file"
                stays.append(f"the key file {_key_file()} ({type(removal).__name__}), under which no store is "
                             "encrypted")
        note = "".join(f"; {item}: delete it" for item in stays)
        if isinstance(exc, EncryptionRefused):
            if not note:
                raise
            raise EncryptionRefused(f"{exc}{note}") from exc
        if isinstance(exc, _Ended):
            raise EncryptionRefused(f"stopped by signal {exc.args[0]}: every store it switched was put back: "
                                    f"{changed}{note}") from exc
        if isinstance(exc, Exception):
            raise EncryptionRefused(f"a store could not be switched ({type(exc).__name__}): every step was put back: "
                                    f"{changed}{note}") from exc
        raise
    unsynced = []
    for place in {store.parent for store, _copy in copies} | set(folders.values()):
        try:
            _fsync_dir(place)
        except OSError:
            unsynced.append(str(place))
    from opti_oignon import encryption

    environment_key = encryption._secret(encryption._ENV_KEY_NAME)
    passphrase = encryption._secret(encryption._ENV_KEYFILE_PASS)
    return {
        "encrypted": [store.name for store, _copy in copies],
        "already": [path.name for path in stores if path not in plain],
        "plaintext_copies": [str(folder) for folder in folders.values()],
        "earlier_copies": [str(folder) for folder in earlier],
        "key": "made" if master is not None else "existing",
        "key_file": ("environment" if environment_key else "wrapped" if passphrase else "file"),
        "key_path": "" if environment_key else str(_key_file()),
        "places": [str(place) for place in (_places() if places is None else places)],
        "left": alone,
        "plaintext_others": [str(path) for path in others],
        "unchecked": [[str(path), why] for path, why in unchecked],
        "unsynced": unsynced,
        "signal": noted[0] if noted else None,
    }


def _drop(copies: list) -> list:
    """Remove the encrypted copies a run made that took no store's place; what could not be removed."""
    kept = []
    for _store, copy in copies:
        try:
            os.unlink(copy)
        except FileNotFoundError:
            pass
        except OSError as exc:
            kept.append(f"{copy} stays ({type(exc).__name__})")
    return kept


def _in_a_kept_copy(folder: Path, places: Any = None) -> bool:
    """Whether ``folder`` lies in a folder of plaintext copies a run kept: a ``plaintext-*`` folder at or below the
    place, or the followed directory, it lies in -- never one above them, wherever the repository sits."""
    roots = [Path(place) for place in (_places() if places is None else places)]
    roots += _followed_dirs() if places is None else []
    for root in roots:
        if root in Path(folder).parents:
            return any(part.startswith("plaintext-") for part in Path(folder).relative_to(root).parts)
    return False


def _migration_hint(path: Path) -> Any:
    """The line that says what moves the ChromaDB store at ``path`` -- one in a folder ``migrate`` knows, outside a
    folder of plaintext copies --, or None."""
    folder = Path(path).parent
    if Path(path).name != _CHROMA_FILE or folder.name not in _CHROMA_FOLDERS or _in_a_kept_copy(folder):
        return None
    return (f"{folder}: `{MIGRATE_COMMAND}`, the platform stopped, moves its collections into the knowledge base's "
            "store, encrypted with the other stores")


def _say_others(paths: list) -> None:
    """Say each file of ``plaintext_others``, and, for a ChromaDB store ``migrate`` moves, the command that moves it."""
    for path in paths:
        print(_plaintext_said(path))
        line = _migration_hint(path)
        if line is not None:
            print(line)


def _migrate_said() -> int:
    """``migrate``: ChromaDB's stores moved into the knowledge base's store, said; 1 when it is refused, 3 when a
    folder could not be moved once the store had committed."""
    from opti_oignon import chroma_migration

    try:
        done = chroma_migration.migrate_all()
    except EncryptionRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    except EncryptionStuck as exc:
        # What the store took before the run stopped -- written, kept, removed -- is said before where it stopped.
        done = getattr(exc, "done", None)
        if done is not None:
            _said_first(_migration_summary, done)
            if done.get("signal"):
                print(f"signal {done['signal']} came after the commit: the run went on", file=sys.stderr)
        print(f"stopped part way: {exc}", file=sys.stderr)
        return 3
    _said_first(_migration_summary, done)
    if done["signal"]:
        print(f"signal {done['signal']} came after the commit: the run finished", file=sys.stderr)
    return 0


def _said_first(say: Any, done: dict) -> None:
    """``say(done)`` on stdout, flushed before anything follows on stderr, so one log of both reads in order. A
    stdout that cannot be written -- a pipe closed early --, an error or an interrupt while it is said costs only
    what is left of it, said on stderr, never the lines that follow nor the exit code: stdout then points at the
    null device, which the interpreter's last flush cannot fail on."""
    try:
        say(done)
        sys.stdout.flush()
    except (OSError, KeyboardInterrupt) as exc:
        try:
            null = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(null, sys.stdout.fileno())
            finally:
                os.close(null)
        except (OSError, ValueError):
            pass
        try:
            print(f"what this run did is said only up to here ({type(exc).__name__}{': ' if str(exc) else ''}{exc})",
                  file=sys.stderr)
        except OSError:
            pass


def _migration_summary(done: dict) -> None:
    """What a run of ``migrate`` did, said: each folder moved, or committed and left where it is, with what each of
    its collections brought; the memory's facts left with no vector of their text; the store."""
    unmoved = done.get("unmoved", [])
    print("places: " + ", ".join(str(place) for place in _places()))
    if not done["moved"] and not unmoved:
        print("no ChromaDB store to move")
    for moved in done["moved"]:
        print(f"{moved['folder']}: {len(moved['collections'])} collection(s) moved into the knowledge base's store")
        for entry in moved["collections"]:
            print(_collection_said(entry))
        print(f"its folder is kept in {moved['kept_in']}: delete it once the platform reads its knowledge base again")
    for left in unmoved:
        print(f"{left['folder']}: {len(left['collections'])} collection(s) written into the knowledge base's store, "
              "the folder not moved: the next run of migrate moves it")
        for entry in left["collections"]:
            print(_collection_said(entry))
    if done.get("facts_without_vector"):
        print(f"{done['facts_without_vector']} active fact(s) of the memory hold no vector of their text as it reads "
              "now: recall finds them by their words")
    if done.get("facts_uncounted"):
        print(f"the active facts of the memory that hold no vector of their text: not counted, the store did not read "
              f"back ({done['facts_uncounted']}); recall finds such facts by their words")
    for folder in done["unknown"]:
        print(f"{folder}: a ChromaDB store in a folder this command does not know: left as it is")
    if done["moved"] or unmoved:
        state = ("encrypted under the key that loads" if done["encrypted"] else
                 "in plaintext: no key loads, as every store of the platform")
        print(f"the knowledge base's store: {done['target']}, {state}")
    _say_unchecked(unchecked_entries())


def _collection_said(entry: dict) -> str:
    """One collection of a ChromaDB folder, with what the store took of it."""
    left = (f", {entry['left_out']} left out: no active fact of the memory holds them as they read now"
            if entry["left_out"] else "")
    gone = (f", {entry['removed']} removed: the store held them with a text no active fact of the memory holds now"
            if entry["removed"] else "")
    return (f"  {entry['name']}, into the namespace {entry['namespace']}: {entry['written']} item(s) written, "
            f"{entry['kept']} already there kept{left}{gone}")


def main(argv: Any = None) -> int:
    """``python3 -m opti_oignon.db_encryption check|encrypt|migrate``: each store's state, every store encrypted, or
    ChromaDB's stores moved into the knowledge base's store (``chroma_migration``)."""
    import argparse

    parser = argparse.ArgumentParser(prog="python3 -m opti_oignon.db_encryption",
                                     description="Say the state of every store, encrypt them all, or move ChromaDB's "
                                                 "stores into the knowledge base's store, the platform stopped.")
    parser.add_argument("action", choices=("check", "encrypt", "migrate"))
    action = parser.parse_args(argv).action
    for stream in (sys.stdout, sys.stderr):
        # A file name the locale cannot write is said escaped, never a crash after the work is done.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    if action == "check":
        print("places: " + ", ".join(str(place) for place in _places()))
        stores = store_files()
        try:
            key = _loaded_key() if SQLCIPHER_AVAILABLE else None
        except EncryptionRefused as exc:
            # The platform passes over a key variable that is no key: the states are read as it reads them.
            key = _key_file_key() if SQLCIPHER_AVAILABLE and _environment_key_refused() is not None else None
            from opti_oignon import encryption

            passed = ("; meanwhile the platform passes over it to the key file, which loads" if key else
                      "; the key file it passes over to does not load here either (if it is wrapped under a "
                      "passphrase, set OPTI_KEYFILE_PASSPHRASE)" if _environment_key_refused() is not None
                      and encryption.keyfile_present() else "")
            print(f"key: {exc}{passed}")
        for path in stores:
            state = ("a symbolic link" if path.is_symlink() else "plaintext" if _plain(path) else
                     "encrypted" if key is not None and _opens_under(path, key) else "unreadable here")
            print(f"{path}: {state}")
        try:
            held = store_holders(stores)
        except EncryptionRefused as exc:
            print(f"holders: {exc}")
        else:
            for pid, why in sorted(held.items()):
                print(f"process {pid} {why}")
        for folder in _unreadable_dirs(None):
            print(f"{folder}: cannot be read: its stores are not checked, and encrypt refuses until it can")
        for path in other_databases():
            print(f"{path}: no store of the platform, left as it is")
        others, unchecked = _scan(None)
        _say_others(others)
        _say_unchecked(unchecked)
        for problem in _sandbox_problems():
            print(f"sandbox: {problem}; encrypt refuses until then")
        for name, value in _relative_followed():
            print(f"{name} is {value}, below whatever directory the platform starts in: not checked")
        for item in _leftovers(stores, None):
            print(f"left by an interrupted run: {item}")
        for folder in plaintext_folders():
            print(f"plaintext copies kept in {folder}: delete it once the platform reads its stores again and the key "
                  "is kept safe elsewhere")
        print(_mode_line())
        print(f"{len(stores)} store(s), {len(plaintext_stores())} in plaintext; SQLCipher "
              f"{'installed' if SQLCIPHER_AVAILABLE else 'missing'}; a key {'loads' if key else 'does not load'}")
        return 0
    if action == "migrate":
        return _migrate_said()
    try:
        done = encrypt_all_stores()
    except EncryptionRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    except EncryptionStuck as exc:
        print(f"stopped part way: {exc}", file=sys.stderr)
        return 3
    print("places: " + ", ".join(done["places"]))
    print(f"{len(done['encrypted'])} store(s) encrypted, {len(done['already'])} already; key {done['key']} "
          f"({done['key_file']})")
    if done["key_file"] == "environment":
        print("the key is OPTI_ENCRYPTION_KEY: the platform starts with it set, and a copy of it is kept somewhere "
              "safe, off this machine and out of any repository")
    else:
        print(f"key file: {done['key_path']} -- every store opens only with it: keep a copy of it somewhere safe, off "
              "this machine and out of any repository, before deleting the plaintext copies")
        if done["key_file"] == "wrapped":
            print("it is wrapped under OPTI_KEYFILE_PASSPHRASE: the platform starts with it set")
    if done["key"] == "made":
        # The key file is new: the mode the platform reads, said as it reads it now (encryption.save_keyfile).
        print(_mode_line())
    for folder in done["plaintext_copies"]:
        print(f"plaintext copies kept in {folder}: delete it once the platform reads its stores again and the key is "
              "kept safe elsewhere")
    for folder in done["earlier_copies"]:
        print(f"plaintext copies an earlier run kept in {folder}: delete it once the platform reads its stores again "
              "and the key is kept safe elsewhere")
    for path in done["left"]:
        print(f"{path}: no store of the platform, left as it is")
    _say_others([Path(path) for path in done["plaintext_others"]])
    _say_unchecked([(Path(path), why) for path, why in done["unchecked"]])
    for name, value in _relative_followed():
        print(f"{name} is {value}, below whatever directory the platform starts in: not checked")
    for place in done["unsynced"]:
        print(f"{place} could not be synced to the disk: the switch stands; run sync before the next power cut")
    if done["signal"]:
        print(f"signal {done['signal']} came after the last safe point: the run finished", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
