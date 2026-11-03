#!/usr/bin/env python3
"""
Hash-Chain Append-Only Signed Audit Log.

Every security-relevant event is recorded as an append-only entry whose
SHA-512 hash chains back to the previous entry, forming a tamper-evident
ledger.  Any modification to a past row breaks the chain and is detected
by ``verify_chain()``.

Storage: ``data/audit_chain.db`` (SQLCipher when available).

Entry schema::

    id           INTEGER  PRIMARY KEY AUTOINCREMENT
    timestamp    REAL     Unix epoch (time.time())
    event_type   TEXT     e.g. "login_success", "mode_change"
    source       TEXT     Module that generated the event
    action       TEXT     Human description
    severity     TEXT     INFO / WARNING / CRITICAL
    details_json TEXT     JSON-encoded extra payload
    prev_hash    TEXT     Hash of the previous entry (128 hex chars)
    entry_hash   TEXT     SHA-512 of (id||timestamp||...||prev_hash)

Genesis entry (id=1): prev_hash = "0" * 128
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opti_oignon.db_utils import safe_connect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_DB_NAME = "audit_chain.db"
_GENESIS_PREV_HASH = "0" * 128  # 128 hex chars = 64 bytes (SHA-512)

SIGNED_AUDIT_AVAILABLE = True

# ---------------------------------------------------------------------------
# Truncation anchor
# ---------------------------------------------------------------------------
#
# The anchor is keyed on the master encryption key (domain-separated), not on
# the DB path: a path string is not a secret, so a local attacker who truncates
# and recreates the DB at the same path could otherwise recompute a valid MAC.
# A non-secret key id, derived one-way from the same key, lets a check tell a
# genuine tamper (same key id, MAC no longer verifies) from a relocated/fresh
# install or a key change (different key id) without raising a false alarm.
# When no master key is configured (encryption disabled, the default), there is
# no secret to bind to and the anchor degrades to an accidental-corruption
# checksum that is advisory only.
#
# What the check sees is never re-anchored in silence: every verdict but an
# anchor that verifies, a chain grown past it, a first run, and an anchor under
# a key this process does not hold, is written into the chain itself
# (``audit_anchor_<verdict>``), and the chain moves its anchor past that entry
# -- never before. A log line is said by one process; the entry stays.

_ANCHOR_FORMAT_VERSION = "v2"
_ANCHOR_KEY_LABEL = b"opti-oignon-audit-anchor-v1"
_ANCHOR_KEYID_LABEL = b"opti-oignon-audit-anchor-keyid-v1"
_ANCHOR_NOKEY_ID = "nokey"
# An anchor is one short line; one longer than this was written by no writer here.
_ANCHOR_MOST_BYTES = 4096
# The verdicts that say nothing: the anchor verifies, the chain grew past it,
# there is neither an entry nor an anchor (a first run), or the anchor was
# written under a key and none is loaded here (left as it is, for a process
# that holds the key to compare).
_ANCHOR_QUIET = ("ok", "grown", "empty", "key_not_loaded")


def _read_anchor_text(path: Path) -> str | None:
    """The anchor's text, from one read; None when there is none. Raises OSError when it cannot be looked at or
    read: it is no regular file -- a link at its name is one, never followed --, or longer than any anchor written;
    UnicodeDecodeError when its bytes are not UTF-8, which no writer here wrote. Only a regular file is opened,
    without blocking and without following a link -- a FIFO swapped in since answers at once, under the chain's
    lock when an append opened the chain --, and checked again once open."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(st.st_mode):
        raise OSError("the anchor is not a regular file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOCTTY", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("the anchor is not a regular file")
        data = b""
        while len(data) <= _ANCHOR_MOST_BYTES:
            chunk = os.read(fd, _ANCHOR_MOST_BYTES + 1 - len(data))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    if len(data) > _ANCHOR_MOST_BYTES:
        raise OSError(f"the anchor holds more than {_ANCHOR_MOST_BYTES} bytes")
    return data.decode("utf-8")


def _write_anchor_text(path: Path, text: str) -> None:
    """Write the anchor whole: a new file beside it (0600, its name unique), moved in its place. A reader in another
    process never meets half an anchor, and a link at its name is replaced, never written through."""
    temp = path.parent / f".{path.name}.tmp-{os.urandom(4).hex()}"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            view = memoryview(text.encode("utf-8"))
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise


def _anchor_mac(content: str, anchor_key: bytes | None) -> str:
    """Compute the anchor tag for ``content``.

    With a secret-derived ``anchor_key`` this is a keyed HMAC-SHA256 (a path
    attacker cannot forge it). Without a key it is a plain SHA-256 checksum,
    which only detects accidental corruption (an attacker could recompute it).
    """
    if anchor_key is None:
        return hashlib.sha256(content.encode()).hexdigest()
    return hmac.new(anchor_key, content.encode(), hashlib.sha256).hexdigest()

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """Single audit chain entry."""

    id: int
    timestamp: float
    event_type: str
    source: str
    action: str
    severity: str
    details_json: str
    prev_hash: str
    entry_hash: str


# ---------------------------------------------------------------------------
# Hash computation
# ---------------------------------------------------------------------------


def _compute_entry_hash(
    entry_id: int,
    timestamp: float,
    event_type: str,
    source: str,
    action: str,
    severity: str,
    details_json: str,
    prev_hash: str,
) -> str:
    """Compute SHA-512 for an entry by concatenating all fields."""
    payload = (
        f"{entry_id}||{timestamp}||{event_type}||{source}||"
        f"{action}||{severity}||{details_json}||{prev_hash}"
    )
    return hashlib.sha512(payload.encode("utf-8")).hexdigest()


def _entry(event_type: str, source: str, action: str, severity: str, details: dict[str, Any] | None) -> tuple:
    """An entry as the chain writes it: ``(timestamp, event_type, source, action, severity, details_json)``."""
    details_json = json.dumps(details or {}, separators=(",", ":"), default=str)
    return (time.time(), event_type, source, action, severity, details_json)


def _said_not_written(entry: tuple, why: Any) -> None:
    """An entry asked for by the thread writing another, and not written: said as ``chain_log`` says any append
    that fails -- never a debug line."""
    logger.error(
        "AUDIT CHAIN APPEND FAILED -- this event is NOT in the tamper-"
        "evidence chain: %s (event_type=%s, action=%s)",
        why, entry[1], entry[3],
    )


# ---------------------------------------------------------------------------
# SignedAuditLog
# ---------------------------------------------------------------------------


class SignedAuditLog:
    """Append-only hash-chain audit log backed by SQLite.

    Thread-safe.  Uses ``get_encrypted_connection()`` if SQLCipher is
    available, otherwise falls back to plain SQLite.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or (_DATA_DIR / _DB_NAME))
        self._lock = threading.Lock()
        # The thread writing an entry, while it holds the lock, and the
        # entries that thread asked for meanwhile (``append_event``).
        self._holder: int | None = None
        self._queued: list[tuple] = []
        # Both the schema and the integrity check move to the first
        # connection. This log is constructed at module scope, so the check
        # ran in every process that imported the package, including tooling
        # that will never read the log and where its warning reaches nobody.
        # Where the log IS used the check runs immediately before the first
        # caller relies on what it checks, and again at each connection until
        # it concludes: a verdict reached and, when it says something, its
        # entry in the chain. One thread runs it at a time; the others, and
        # the check's own connections, pass it by.
        self._ready = False
        self._check_lock = threading.Lock()
        self._concluded = False
        # The verdict entry whose writing concludes the check; until then the
        # anchor does not move (``_chain``).
        self._unsaid: tuple | None = None
        # False in a process that holds no key, beside an anchor written
        # under one: the anchor is left as it is.
        self._may_anchor = True
        # Whether the check ran under a key: then no anchor is written
        # without one (``_save_anchor``).
        self._keyed_at_check = False
        # The height and tip this process last anchored or compared: the
        # chain each append extends must still hold them (``_chain``).
        self._held: tuple[int, str] | None = None
        # Whether a check that failed was said in the chain already.
        self._failure_said = False
        # What this process's check saw of the anchor (``get_status``).
        self._anchor_seen: dict[str, Any] | None = None

    # -- DB helpers ----------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Open a connection, preferring encrypted if available.

        The first connection makes the schema on itself before it is handed
        out, and only then is the log ready: opening the store asks the
        security mode, and a mode that fails secure records why through
        ``append_event`` -- an entry that must find the table made, whichever
        call opened the store first. A schema that could not be made is made
        at the next call. Then the integrity check, until it concludes.
        """
        if not self._ready:
            directory = os.path.dirname(self._db_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        conn = safe_connect(self._db_path, check_same_thread=False)
        if not self._ready:
            try:
                self._init_db(conn)
            except BaseException:
                conn.close()
                raise
            self._ready = True
        # The check reaches this helper again (verify_chain()): its own
        # connections find the lock held and pass it by, as another thread's
        # do while it runs -- one that waited could hold the chain's lock
        # the check needs to write its verdict.
        if not self._concluded and self._check_lock.acquire(blocking=False):
            try:
                self._check_integrity_on_init()
            finally:
                self._check_lock.release()
        return conn

    def _init_db(self, conn: sqlite3.Connection) -> None:
        """Create the audit_chain table on ``conn`` if it does not exist."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_chain (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    REAL    NOT NULL,
                event_type   TEXT    NOT NULL,
                source       TEXT    NOT NULL DEFAULT '',
                action       TEXT    NOT NULL DEFAULT '',
                severity     TEXT    NOT NULL DEFAULT 'INFO',
                details_json TEXT    NOT NULL DEFAULT '{}',
                prev_hash    TEXT    NOT NULL,
                entry_hash   TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_event_type
            ON audit_chain(event_type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_severity
            ON audit_chain(severity)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp
            ON audit_chain(timestamp)
        """)
        conn.commit()

    def _check_integrity_on_init(self) -> None:
        """Run a quick chain check on startup and warn if broken.

        Audit hardening: also compares the chain with a separate anchor
        file (``_check_anchor``), which sees a chain truncated, replaced by
        a fresh one, or rewritten at the anchored height -- and says, in the
        chain, every anchor it could not check. A walk of the chain that
        fails never keeps the anchor from being compared. A comparison that
        fails is said in the chain once, and runs again at the next
        connection; the anchor waits for it.
        """
        try:
            valid, broken_idx, total = self.verify_chain()
            if total > 0 and not valid:
                logger.warning(
                    "AUDIT CHAIN INTEGRITY BROKEN at entry %s "
                    "(total %d entries). Investigate immediately.",
                    broken_idx, total,
                )
            elif total > 0:
                logger.info(
                    "Audit chain OK: %d entries verified.", total,
                )
        except Exception as exc:
            logger.warning("Audit chain walk failed: %s", exc)
        reason = None
        try:
            # Truncation detection via anchor file
            self._check_anchor()
        except Exception as exc:
            logger.warning("Audit chain init check failed: %s", exc)
            reason = f"the check failed ({type(exc).__name__}: {exc})"
        # Said once the handler is left: the exception, kept, would keep what
        # raised it -- a cursor and its read lock -- and the entry wait on it.
        if reason is not None:
            self._anchor_seen = {"verdict": "not_checked", "reason": reason}
            if not self._failure_said:
                self._failure_said = self._record_anchor_verdict(
                    "not_checked", "WARNING", {"reason": reason}, concludes=False,
                )

    def _get_anchor_path(self) -> Path:
        """Return the path to the chain anchor file."""
        return Path(self._db_path).parent / ".audit_chain_anchor"

    def _anchor_secret(self) -> tuple[bytes | None, str]:
        """Derive the anchor MAC key and a non-secret key id.

        Keyed on the master encryption key via a domain-separated HMAC, the
        same construction used for the SQLCipher subkey. Returns
        ``(None, "nokey")`` when no master key is configured, in which case the
        anchor is a non-cryptographic checksum (advisory only).
        """
        try:
            from opti_oignon.encryption import get_encryption_key
        except Exception:
            return None, _ANCHOR_NOKEY_ID

        try:
            sb = get_encryption_key()
        except Exception:
            sb = None
        if not sb:
            return None, _ANCHOR_NOKEY_ID

        try:
            with sb as master:
                raw = master.as_bytes()
                anchor_key = hmac.new(
                    raw, _ANCHOR_KEY_LABEL, hashlib.sha256,
                ).digest()
                key_id = hmac.new(
                    raw, _ANCHOR_KEYID_LABEL, hashlib.sha256,
                ).hexdigest()[:16]
            return anchor_key, key_id
        except Exception:
            return None, _ANCHOR_NOKEY_ID

    def _key_configured(self) -> bool:
        """Whether a master key is configured here, loaded or not: the key
        file is there, or OPTI_ENCRYPTION_KEY is held (the environment's, or
        taken out of it at start). A held value wiped, or a failure to tell,
        is configured: the anchor is then left as it is."""
        try:
            from opti_oignon.encryption import _ENV_KEY_NAME, _secret, keyfile_present

            return keyfile_present() or bool(_secret(_ENV_KEY_NAME))
        except Exception:
            return True

    def _save_anchor(self, count: int, tip_hash: str) -> None:
        """Save the chain tip to the anchor file for truncation detection.

        Format: ``v2|{key_id}|{count}|{tip}|{mac}``. The MAC is keyed on the
        master encryption key (domain-separated), not on the DB path. Written
        whole (``_write_anchor_text``), 0600. A process whose check ran under
        a key writes none without it: a key that does not load here now
        (wiped, unreadable) never turns the anchor into a checksum.
        """
        anchor_path = self._get_anchor_path()
        try:
            anchor_key, key_id = self._anchor_secret()
            if anchor_key is None and self._keyed_at_check:
                logger.debug("Audit anchor not moved: the key the check ran under does not load here now")
                return
            content = (
                f"{_ANCHOR_FORMAT_VERSION}|{key_id}|{count}|{tip_hash}"
            )
            mac = _anchor_mac(content, anchor_key)
            _write_anchor_text(anchor_path, f"{content}|{mac}")
        except Exception as exc:
            logger.debug("Failed to save audit anchor: %s", exc)

    def _check_anchor(self) -> None:
        """Compare the chain with its anchor, and say what was seen.

        The anchor is read before the chain, and the chain's height and tip
        in one read, so that an entry another writer appends meanwhile reads
        as growth, never as a truncation or a rewrite. Verdicts:

        * ``ok`` -- the anchor verifies and the chain ends at the anchored tip;
          ``grown`` -- the chain holds the anchored tip at the anchored height
          and more after it (re-anchored at the height and tip read);
          ``empty`` -- neither an entry nor an anchor, a first run;
          ``key_not_loaded`` -- the anchor was written under a key and none
          is loaded here: not compared, and left as it is. These say nothing.
        * ``missing`` -- no anchor beside entries; ``unrecognized`` -- no
          anchor written here (legacy, foreign or damaged); ``other_key`` --
          written under another key id (a key change, a relocated install, or
          an anchor made without the key); ``checksum_mismatch`` -- an unkeyed
          anchor whose checksum fails; ``not_checked`` -- the anchor could not
          be looked at or read, or the check failed. A WARNING each.
        * ``tampered`` -- a keyed anchor whose tag fails; ``truncated`` -- the
          chain is shorter than the anchor; ``rewritten`` -- the entry at the
          anchored height is not the anchored tip. CRITICAL under a key;
          without one the anchor is advisory only, and each is a WARNING.

        Every verdict that says something is logged and written into the
        chain (``_record_anchor_verdict``); the check concludes, and the
        anchor moves, only once that entry is in: the evidence outlives the
        anchor it was read from.
        """
        anchor_key, current_key_id = self._anchor_secret()
        keyed = anchor_key is not None
        self._keyed_at_check = keyed
        verdict, seen, tip = self._anchor_verdict(anchor_key, current_key_id)
        self._anchor_seen = {"verdict": verdict, "keyed": keyed, **seen}
        if verdict == "key_not_loaded":
            self._may_anchor = False
        if verdict in ("grown", "empty") and seen["entries"] > 0:
            self._save_anchor(seen["entries"], tip)
        if verdict in _ANCHOR_QUIET:
            self._held = (seen["entries"], tip) if seen["entries"] > 0 else None
            self._concluded = True
            return
        severity = "WARNING"
        if keyed and verdict in ("tampered", "truncated", "rewritten"):
            severity = "CRITICAL"
        self._record_anchor_verdict(verdict, severity, {"keyed": keyed, **seen}, concludes=True)

    def _anchor_verdict(self, anchor_key: bytes | None,
                        current_key_id: str) -> tuple[str, dict[str, Any], str]:
        """The anchor's verdict (``_check_anchor``), the heights it rests on -- the chain's (``entries``) and the
        anchor's (``anchored_entries``), when it gives one --, and the chain's tip, read with its height. The height
        is read before the anchor too: entries appended since are growth, so that no anchor beside a chain empty at
        that first read is a first run."""
        before, _ = self._top()
        try:
            raw = _read_anchor_text(self._get_anchor_path())
        except OSError as exc:
            height, tip = self._top()
            return "not_checked", {"entries": height, "reason": f"{type(exc).__name__}: {exc}"}, tip
        except UnicodeDecodeError:
            height, tip = self._top()
            return "unrecognized", {"entries": height}, tip
        height, tip = self._top()
        if raw is None:
            return ("empty" if before == 0 else "missing"), {"entries": height}, tip
        parts = raw.strip().split("|")
        try:
            version, saved_key_id, saved_count_s, saved_tip, saved_mac = parts
            saved_count = int(saved_count_s)
        except ValueError:
            return "unrecognized", {"entries": height}, tip
        if version != _ANCHOR_FORMAT_VERSION or saved_count < 0:
            return "unrecognized", {"entries": height}, tip
        seen: dict[str, Any] = {"entries": height, "anchored_entries": saved_count}
        if saved_key_id != current_key_id:
            # Not verifiable here: its heights are a claim, said, never compared.
            if anchor_key is None and saved_key_id != _ANCHOR_NOKEY_ID and self._key_configured():
                return "key_not_loaded", {**seen, "anchored_key_id": saved_key_id}, tip
            return "other_key", {**seen, "anchored_key_id": saved_key_id, "key_id": current_key_id}, tip
        content = (
            f"{_ANCHOR_FORMAT_VERSION}|{saved_key_id}|"
            f"{saved_count}|{saved_tip}"
        )
        expected_mac = _anchor_mac(content, anchor_key)
        if not hmac.compare_digest(expected_mac.encode("utf-8"), saved_mac.encode("utf-8")):
            return ("tampered" if anchor_key is not None else "checksum_mismatch"), seen, tip
        if height < saved_count:
            return "truncated", seen, tip
        if saved_count > 0 and self._hash_at(saved_count) != saved_tip:
            return "rewritten", seen, tip
        return ("grown" if height > saved_count else "ok"), seen, tip

    def _record_anchor_verdict(self, verdict: str, severity: str, seen: dict[str, Any], *, concludes: bool) -> bool:
        """Log the anchor's verdict at its severity and write it into the chain as ``audit_anchor_<verdict>``; when
        it ``concludes`` the check, the check concludes once that entry is written. An entry that cannot be written
        is said as not written, as ``chain_log`` says it, and the check runs again at the next connection. Whether
        the entry was written, or queued behind the one this thread is writing."""
        entries, anchored = seen.get("entries"), seen.get("anchored_entries")
        advisory = "" if seen.get("keyed") else " (no master key: advisory only)"
        said = {
            "missing": f"Audit anchor missing beside {entries} entries: a truncation before it was lost cannot be "
                       f"seen; re-anchored",
            "unrecognized": "Audit anchor is not one written here (legacy, foreign or damaged); re-anchored",
            "other_key": "Audit anchor was written under another key (a key change, a relocated install, or an "
                         "anchor made without the key); re-anchored",
            "checksum_mismatch": "Audit anchor checksum mismatch (no master key: advisory only); re-anchored",
            "not_checked": f"Audit anchor not checked ({seen.get('reason')}); the chain was not compared with it",
            "tampered": "AUDIT ANCHOR FILE TAMPERED: its tag fails under the current key. Chain trust is "
                        "compromised",
            "truncated": f"AUDIT CHAIN TRUNCATED: anchor has {anchored} entries but DB only has {entries}. The "
                         f"audit DB may have been replaced{advisory}",
            "rewritten": f"AUDIT CHAIN REWRITTEN: the entry at the anchored height {anchored} is not the anchored "
                         f"tip{advisory}",
        }[verdict]
        logger.log(logging.CRITICAL if severity == "CRITICAL" else logging.WARNING, "%s", said)
        entry = _entry(f"audit_anchor_{verdict}", "signed_audit_log", said, severity, seen)
        if concludes:
            self._unsaid = entry
        try:
            self._append(entry)
        except Exception as exc:  # noqa: BLE001 - said as not written, never dropped
            _said_not_written(entry, exc)
            return False
        return True

    def _top(self) -> tuple[int, str]:
        """The chain's height -- its last entry's id, its number of entries the chain intact -- and that entry's
        hash, in one read ("" for an empty chain)."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT id, entry_hash FROM audit_chain ORDER BY id DESC LIMIT 1").fetchone()
            return (int(row[0]), row[1]) if row else (0, "")
        finally:
            conn.close()

    def _hash_at(self, entry_id: int) -> str | None:
        """The entry_hash of entry ``entry_id``; None when there is none."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT entry_hash FROM audit_chain WHERE id = ?", (entry_id,)).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def _get_tip_hash(self) -> str:
        """Return the entry_hash of the last chain entry."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT entry_hash FROM audit_chain ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else ""
        finally:
            conn.close()

    # -- Core operations -----------------------------------------------------

    def append_event(
        self,
        event_type: str,
        source: str = "",
        action: str = "",
        severity: str = "INFO",
        details: dict[str, Any] | None = None,
    ) -> int | None:
        """Append a new event to the chain.

        Returns the new entry id. The thread already writing an entry is
        never made to wait on itself: opening the store asks the security
        mode, and a mode that fails secure records why -- through here, on
        the same thread. Such an entry is chained right after the one being
        written, in the order asked, and None is returned; if the entry it
        follows is not written, it is said as not written, by name. Another
        thread waits its turn as before.
        """
        return self._append(_entry(event_type, source, action, severity, details))

    def _append(self, entry: tuple) -> int | None:
        """``append_event`` for an entry already made (``_entry``)."""
        if self._holder == threading.get_ident():
            self._queued.append(entry)
            return None
        with self._lock:
            self._holder = threading.get_ident()
            try:
                conn = self._get_conn()
                try:
                    own = self._chain(conn, entry)
                    while self._queued:
                        queued = self._queued.pop(0)
                        try:
                            self._chain(conn, queued)
                        except Exception as exc:
                            _said_not_written(queued, exc)
                    return own
                finally:
                    conn.close()
            finally:
                self._holder = None
                left, self._queued = self._queued, []
                for queued in left:
                    _said_not_written(queued, "the entry it was to follow was not written")

    def _chain(self, conn: sqlite3.Connection, entry: tuple) -> int:
        """Write ``entry`` after the chain's last one, committed, and move the anchor; its id. The anchor moves
        only once the integrity check has concluded -- its verdict reached and, when it says something, written,
        which this entry may be -- and never off an anchor written under a key this process does not hold. A chain
        that no longer holds what this process last anchored or compared -- truncated or rewritten while it ran --
        is said, as the check says it, after the entry, and the anchor waits for that verdict."""
        ts, event_type, source, action, severity, details_json = entry
        try:
            # Fetch last entry hash
            row = conn.execute(
                "SELECT entry_hash FROM audit_chain ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev_hash = row[0] if row else _GENESIS_PREV_HASH

            # We need the id BEFORE computing the hash, but AUTOINCREMENT
            # assigns it on INSERT.  Strategy: insert a placeholder then
            # update, or use a two-step approach.
            # Better: fetch max(id)+1 within the lock.
            max_row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM audit_chain"
            ).fetchone()
            next_id = max_row[0] + 1

            held, gone = (self._held if self._concluded else None), None
            if held is not None:
                at = conn.execute(
                    "SELECT entry_hash FROM audit_chain WHERE id = ?", (held[0],)
                ).fetchone()
                if at is None or at[0] != held[1]:
                    gone = "truncated" if next_id - 1 < held[0] else "rewritten"

            entry_hash = _compute_entry_hash(
                next_id, ts, event_type, source, action,
                severity, details_json, prev_hash,
            )

            conn.execute(
                """INSERT INTO audit_chain
                   (id, timestamp, event_type, source, action,
                    severity, details_json, prev_hash, entry_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (next_id, ts, event_type, source, action,
                 severity, details_json, prev_hash, entry_hash),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        if gone is not None:
            keyed = self._anchor_secret()[0] is not None
            seen = {"keyed": keyed, "entries": next_id - 1, "anchored_entries": held[0]}
            self._held, self._concluded = None, False
            self._anchor_seen = {"verdict": gone, **seen}
            self._record_anchor_verdict(gone, "CRITICAL" if keyed else "WARNING", seen, concludes=True)
        if entry is self._unsaid:
            self._unsaid = None
            self._concluded = True
        # Update anchor for truncation detection
        if self._concluded:
            self._held = (next_id, entry_hash)
            if self._may_anchor:
                try:
                    self._save_anchor(next_id, entry_hash)
                except Exception:
                    pass  # Non-critical, do not fail the append

        return next_id

    def verify_chain(self) -> tuple[bool, int | None, int]:
        """Verify the entire chain from genesis to tip.

        Returns (valid, first_broken_index, total_entries).
        O(n) single pass, no external deps.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT id, timestamp, event_type, source, action, "
                "severity, details_json, prev_hash, entry_hash "
                "FROM audit_chain ORDER BY id ASC"
            )
            try:
                expected_prev = _GENESIS_PREV_HASH
                total = 0

                for row in cursor:
                    total += 1
                    (eid, ts, etype, src, act, sev, djson, phash, ehash) = row

                    # Check prev_hash links to previous entry's hash
                    if phash != expected_prev:
                        return (False, eid, total)

                    # Recompute and compare entry_hash
                    computed = _compute_entry_hash(
                        eid, ts, etype, src, act, sev, djson, phash,
                    )
                    if computed != ehash:
                        return (False, eid, total)

                    expected_prev = ehash

                return (True, None, total)
            finally:
                # Before the connection: a row that raised leaves the cursor
                # holding its statement, and its read lock, past the close.
                cursor.close()
        finally:
            conn.close()

    def get_events(
        self,
        limit: int = 50,
        offset: int = 0,
        event_type: str | None = None,
        severity: str | None = None,
        after: float | None = None,
        before: float | None = None,
    ) -> list[dict[str, Any]]:
        """Query events with optional filters."""
        conn = self._get_conn()
        try:
            clauses: list[str] = []
            params: list[Any] = []

            if event_type:
                clauses.append("event_type = ?")
                params.append(event_type)
            if severity:
                clauses.append("severity = ?")
                params.append(severity)
            if after is not None:
                clauses.append("timestamp >= ?")
                params.append(after)
            if before is not None:
                clauses.append("timestamp <= ?")
                params.append(before)

            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            query = (
                "SELECT id, timestamp, event_type, source, action, "
                "severity, details_json, prev_hash, entry_hash "
                f"FROM audit_chain{where} "
                "ORDER BY id DESC LIMIT ? OFFSET ?"
            )
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [
                {
                    "id": r[0],
                    "timestamp": r[1],
                    "event_type": r[2],
                    "source": r[3],
                    "action": r[4],
                    "severity": r[5],
                    "details": json.loads(r[6]) if r[6] else {},
                    "prev_hash": r[7],
                    "entry_hash": r[8],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_status(self) -> dict[str, Any]:
        """Return chain status: length, last entry, integrity, and what this
        process's check saw of the anchor (``_check_anchor``)."""
        conn = self._get_conn()
        try:
            count_row = conn.execute(
                "SELECT COUNT(*) FROM audit_chain"
            ).fetchone()
            total = count_row[0] if count_row else 0

            last = None
            if total > 0:
                row = conn.execute(
                    "SELECT id, timestamp, event_type, entry_hash "
                    "FROM audit_chain ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    last = {
                        "id": row[0],
                        "timestamp": row[1],
                        "event_type": row[2],
                        "entry_hash": row[3],
                    }

            valid, broken_idx, _ = self.verify_chain()
            seen = self._anchor_seen
            anchor = dict(seen) if seen is not None else {
                "verdict": "not_checked", "reason": "the check has not finished",
            }

            return {
                "total_entries": total,
                "last_entry": last,
                "chain_valid": valid,
                "first_broken_index": broken_idx,
                "anchor": anchor,
            }
        finally:
            conn.close()

    def export_chain_csv(self) -> str:
        """Export the full chain as CSV text."""
        import csv
        import io

        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT id, timestamp, event_type, source, action, "
                "severity, details_json, prev_hash, entry_hash "
                "FROM audit_chain ORDER BY id ASC"
            ).fetchall()

            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow([
                "id", "timestamp", "event_type", "source", "action",
                "severity", "details_json", "prev_hash", "entry_hash",
            ])
            for r in rows:
                writer.writerow(r)
            return buf.getvalue()
        finally:
            conn.close()

    def entry_count(self) -> int:
        """Return the number of entries in the chain."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) FROM audit_chain").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

signed_audit_log: SignedAuditLog | None = None

try:
    signed_audit_log = SignedAuditLog()
except Exception as exc:
    logger.warning("Failed to initialize signed audit log: %s", exc)
    SIGNED_AUDIT_AVAILABLE = False


def chain_log(
    event_type: str,
    source: str = "",
    action: str = "",
    severity: str = "INFO",
    **details: Any,
) -> int | None:
    """Convenience wrapper: log to the chain if available.

    Returns the entry id or None if the chain is unavailable, or if this
    thread is writing an entry already: chained right after it
    (``SignedAuditLog.append_event``).
    """
    if signed_audit_log is None:
        return None
    try:
        return signed_audit_log.append_event(
            event_type=event_type,
            source=source,
            action=action,
            severity=severity,
            details=details if details else None,
        )
    except Exception as exc:
        # NEVER debug. This is the tamper-evidence chain refusing an entry, and
        # no caller reads the return value -- so a debug line is the only trace
        # that the record of a security event was never written at all. An
        # attacker who can make appends fail can make their own traces vanish.
        logger.error(
            "AUDIT CHAIN APPEND FAILED -- this event is NOT in the tamper-"
            "evidence chain: %s (event_type=%s, action=%s)",
            exc, event_type, action,
        )
        return None
