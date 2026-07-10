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

import hashlib
import hmac
import json
import logging
import os
import sqlite3
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

_ANCHOR_FORMAT_VERSION = "v2"
_ANCHOR_KEY_LABEL = b"opti-oignon-audit-anchor-v1"
_ANCHOR_KEYID_LABEL = b"opti-oignon-audit-anchor-keyid-v1"
_ANCHOR_NOKEY_ID = "nokey"


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
        self._init_db()
        self._check_integrity_on_init()

    # -- DB helpers ----------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Open a connection, preferring encrypted if available."""
        return safe_connect(self._db_path, check_same_thread=False)

    def _init_db(self) -> None:
        """Create the audit_chain table if it does not exist."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        conn = self._get_conn()
        try:
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
        finally:
            conn.close()

    def _check_integrity_on_init(self) -> None:
        """Run a quick chain check on startup and warn if broken.

        Audit hardening: also checks for truncation by comparing the
        chain tip against a separate anchor file. If someone deleted
        the DB and recreated it with a fresh genesis, the anchor
        file will detect the discrepancy.
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

            # Truncation detection via anchor file
            self._check_anchor(total)
        except Exception as exc:
            logger.warning("Audit chain init check failed: %s", exc)

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

    def _save_anchor(self, count: int, tip_hash: str) -> None:
        """Save the chain tip to the anchor file for truncation detection.

        Format: ``v2|{key_id}|{count}|{tip}|{mac}``. The MAC is keyed on the
        master encryption key (domain-separated), not on the DB path.
        """
        anchor_path = self._get_anchor_path()
        try:
            anchor_key, key_id = self._anchor_secret()
            content = (
                f"{_ANCHOR_FORMAT_VERSION}|{key_id}|{count}|{tip_hash}"
            )
            mac = _anchor_mac(content, anchor_key)
            anchor_path.write_text(f"{content}|{mac}", encoding="utf-8")
            try:
                os.chmod(anchor_path, 0o600)
            except OSError:
                pass
        except Exception as exc:
            logger.debug("Failed to save audit anchor: %s", exc)

    def _check_anchor(self, current_count: int) -> None:
        """Compare the current chain state against the saved anchor.

        A genuine tamper -- an anchor written under the *same* key whose content
        no longer verifies -- is CRITICAL. A relocated or fresh install, a key
        change, or a legacy/foreign anchor (different or absent key id) is only
        informational and re-anchored. Truncation under an authentic keyed
        anchor is CRITICAL; without a master key the anchor is advisory only and
        never raises CRITICAL.
        """
        anchor_path = self._get_anchor_path()
        if not anchor_path.exists():
            # First run, or the anchor was lost -- create it (informational).
            if current_count > 0:
                self._save_anchor(current_count, self._get_tip_hash())
            return

        try:
            raw_text = anchor_path.read_text(encoding="utf-8").strip()
            parts = raw_text.split("|")
            anchor_key, current_key_id = self._anchor_secret()

            # Legacy (pre-v2) or unrecognized anchor: re-anchor under the
            # current key without alarming. Absorbs a foreign anchor that may
            # have shipped from another environment.
            if len(parts) != 5 or parts[0] != _ANCHOR_FORMAT_VERSION:
                logger.info(
                    "Audit anchor is legacy or unrecognized; re-anchoring "
                    "under the current key."
                )
                if current_count > 0:
                    self._save_anchor(current_count, self._get_tip_hash())
                return

            _, saved_key_id, saved_count_s, saved_tip, saved_mac = parts
            saved_count = int(saved_count_s)
            content = (
                f"{_ANCHOR_FORMAT_VERSION}|{saved_key_id}|"
                f"{saved_count}|{saved_tip}"
            )

            # Different (or absent) secret: relocated install, key rotation, or
            # a keyed<->unkeyed transition. Informational, not a tamper.
            if saved_key_id != current_key_id:
                logger.info(
                    "Audit anchor was written under a different key "
                    "(relocated install or key change); re-anchoring."
                )
                if current_count > 0:
                    self._save_anchor(current_count, self._get_tip_hash())
                return

            # Same key id: verify the content tag.
            expected_mac = _anchor_mac(content, anchor_key)
            if not hmac.compare_digest(expected_mac, saved_mac):
                if anchor_key is not None:
                    # Keyed: content altered by someone without the secret.
                    logger.critical(
                        "AUDIT ANCHOR FILE TAMPERED. Chain trust is "
                        "compromised."
                    )
                else:
                    # Unkeyed: checksum only; a mismatch is accidental
                    # corruption, so re-anchor without alarming.
                    logger.warning(
                        "Audit anchor checksum mismatch (no master key; "
                        "advisory only); re-anchoring."
                    )
                    if current_count > 0:
                        self._save_anchor(
                            current_count, self._get_tip_hash(),
                        )
                return

            # Authentic anchor under the current key: check truncation.
            if current_count < saved_count:
                if anchor_key is not None:
                    logger.critical(
                        "AUDIT CHAIN TRUNCATED: anchor has %d entries but DB "
                        "only has %d. The audit DB may have been replaced.",
                        saved_count, current_count,
                    )
                else:
                    logger.warning(
                        "Audit chain shorter than the anchor (%d < %d); "
                        "advisory only without a master key.",
                        current_count, saved_count,
                    )
            elif current_count > saved_count:
                # Normal growth -- update the anchor.
                self._save_anchor(current_count, self._get_tip_hash())
        except Exception as exc:
            logger.debug("Anchor check failed: %s", exc)

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
    ) -> int:
        """Append a new event to the chain.

        Returns the new entry id.
        """
        details_json = json.dumps(details or {}, separators=(",", ":"), default=str)
        ts = time.time()

        with self._lock:
            conn = self._get_conn()
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

                # Update anchor for truncation detection
                try:
                    self._save_anchor(next_id, entry_hash)
                except Exception:
                    pass  # Non-critical, do not fail the append

                return next_id
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

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
        """Return chain status: length, last entry, integrity."""
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

            return {
                "total_entries": total,
                "last_entry": last,
                "chain_valid": valid,
                "first_broken_index": broken_idx,
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

    Returns the entry id or None if the chain is unavailable.
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
        logger.debug("chain_log failed: %s", exc)
        return None
