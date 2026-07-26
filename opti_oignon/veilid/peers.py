#!/usr/bin/env python3
"""Per-peer registry and watermark store for Veilid sync.

A user's devices sync against each other as explicit, key-addressed peers. This
module is the local record of which peers a device is paired with and how far it
has consumed each one's change feed. It holds two things per peer: the pairing
identity (a stable peer id and the peer's public routing key, plus an optional
human label) and a monotonic watermark (the last peer-feed sequence this device
has applied). The pairing key exchange itself lives in ``pairing``; this stores what pairing
will populate, and the sync engine reads the watermark before a
round and advances it after.

The watermark is monotonic by construction: ``advance_watermark`` writes
``max(current, incoming)`` at the SQL level, so a late or out-of-order answer can
never move a peer's watermark backwards. A peer that is not registered has no
watermark to advance; advancing an unknown peer is a no-op that returns zero,
and the engine refuses to run a round against an unpaired peer.

Each peer also carries the last-seen epoch of its change feed.
The asker stores the epoch a peer's batches advertise and, when a later round
sees a different one (the peer's journal was recreated, sequences restarted),
resets that peer's watermark to 0 for a single full resync. The reset and the
epoch store land in one statement (``reset_for_epoch``) -- the deliberate,
epoch-bound exception to the monotonic advance -- so a crash can never persist
the new epoch while keeping the old watermark, the interleaving that would
silently skip the resync. A NULL epoch means a pre-epoch or freshly-paired
peer; a re-pair preserves the stored epoch like it preserves the watermark.

The store is backed by SQLite in WAL mode under the data directory, with the
project's SQL hygiene: WAL is set outside any transaction right after connecting;
every query is parameterised; and the single table identifier is assembled with
``str.format()`` under a frozenset allowlist, never an f-string, so no
caller-controlled string ever reaches a statement. Registering a peer is an
upsert that refreshes the routing key and label while preserving the watermark
and the original pairing time, so a re-pair (a rotated route) never resets how
far a device has synced.

A peer also carries a pending state. The pairing ceremony
registers a peer PENDING; the entry gates nothing -- the engine refuses to run
a round against it or serve it, and record verification never trusts its
registered key -- until both humans have compared the mutual confirmation code
and confirmed on both devices. The column is additive and nullable, encoded so
the grandfather falls out of the migration by construction: NULL or 0 is
confirmed (every pre-PAIR-02 row reads NULL after ALTER TABLE, so existing
peers are never retroactively locked out), 1 is a fresh pending pairing, and 2
is a peer DEMOTED to pending because a re-pair carried a different signing key
than the stored one (or a first key over a previously unkeyed row): a changed
trust root is a new trust decision. The upsert can only raise the pending
state (demote), never lower it; only an explicit ``confirm_peer`` activates an
entry. ``add_peer``'s default is confirmed -- programmatic registration is an
explicit local trust decision -- and the ceremony path passes ``pending=True``.
The one-row meta table additionally pins this device's own last-generated
pairing material, so the confirmation code recomputes deterministically from
local disk in any mode, independent of the live transport and of route
rotation after the payload was shown.

The registry is a local-disk structure, not a network one, so it is not gated by
the Bulbe boundary: a device may list or manage its peers in any mode. Only
moving records over the wire is Daily-only, and that gate lives in the protocol
envelope and the sync engine, not here.

Kerckhoffs: the registry is open. A peer is addressed by a public routing key the
user holds; there is no secret in the schema, and the security of a sync lives in
the keys and private routes of the transport, not in this table.

The store is a process singleton with a reset hook, and its root is injectable so
tests run against a temporary directory; the data-directory import is lazy and
guarded so the module collects without the backend.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Encrypted DB connections, the same pattern as the
# veilid change feed and sync_queue. The open-registry rationale
# does not stand for the registry's privacy-relevant metadata -- the human labels
# (device names), the device topology, the watermarks, and the SYN-02 local
# device identity -- so the peer registry joins safe_connect and is encrypted at
# rest like the rest of the data layer rather than stored in a plain sqlite3 file
# (the public routing keys are Kerckhoffs material either way). The PeerRecord
# docstring's note that the at-rest posture is the RS-01 lot's is now satisfied
# here; this is that lot.
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    logging.getLogger(__name__).warning(
        "db_utils unavailable: veilid peer registry falling back to PLAINTEXT "
        "sqlite3. The registry (labels, device topology, the SYN-02 local "
        "device identity) is NOT encrypted at rest."
    )
    _safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# The two physical tables: the peer registry and the one-row local-identity meta
# table (SYN-02). Referenced through these constants and validated against the
# allowlist below wherever they are interpolated into a statement.
TABLE_NAME = "veilid_peers"
META_TABLE_NAME = "veilid_local_identity"
DB_FILENAME = "veilid_peers.db"

# The allowlist for the only identifiers ever formatted into SQL (table names).
# This is the sanctioned alternative to an f-string: str.format() under a frozenset.
_TABLES: frozenset[str] = frozenset({TABLE_NAME, META_TABLE_NAME})


def _safe_table(name: str) -> str:
    """Return the table name only if it is in the allowlist, else raise."""
    if name not in _TABLES:
        raise ValueError(f"table identifier not in allowlist: {name!r}")
    return name


_CREATE_TABLE = (
    f"CREATE TABLE IF NOT EXISTS {_safe_table(TABLE_NAME)} ("
    "peer_id TEXT PRIMARY KEY, "
    "routing_key TEXT NOT NULL, "
    "label TEXT NOT NULL DEFAULT '', "
    "watermark INTEGER NOT NULL DEFAULT 0, "
    "added_at TEXT NOT NULL, "
    "updated_at TEXT NOT NULL, "
    "last_epoch TEXT, "
    "signing_pub TEXT, "
    "pending INTEGER, "
    # Per-device remote-inference grants. NULL/1 means
    # remote chat ENABLED (the grandfathered tier-1 default; a fresh insert
    # omits the column, so it is NULL = enabled); 0 means DISABLED. The RAG
    # read-only sub-grant is NULL/0 = off (the conservative default) and 1 = on.
    # Nullable and additive, the SYN-02/PAIR-02 migration shape; preserved on a
    # re-pair (a local trust decision, like the watermark).
    "remote_chat_grant INTEGER, "
    "rag_subgrant INTEGER, "
    # The per-device class marker for phone-class sync
    # filtering. Nullable and additive (the SYN-02 / PAIR-02 / cas 7
    # migration shape): NULL is the grandfathered DESKTOP class -- every
    # pre-N.9 row is a desktop by construction -- and only an explicit
    # ceremony or the control surface writes "phone". Preserved on a
    # re-pair (a local trust decision, like the watermark and the grants).
    "device_class TEXT"
    ")"
)

# The device-class allowlist -- the only values the setter ever
# accepts (NULL clears back to the grandfathered desktop class). The sync
# responder filters NOTE records toward a phone-class peer behind the
# per-item mobile-allowed flag (MOBILE_THREAT_MODEL.md section 3,
# filter-at-serve, decision N9-D1).
DEVICE_CLASS_PHONE = "phone"
DEVICE_CLASS_DESKTOP = "desktop"
DEVICE_CLASSES: frozenset[str] = frozenset(
    {DEVICE_CLASS_PHONE, DEVICE_CLASS_DESKTOP}
)

# Upsert: insert a fresh peer at watermark 0, or refresh the routing key and label
# of an existing one. The DO UPDATE clause deliberately omits watermark, added_at,
# and last_epoch, so a re-pair preserves how far the device has synced, when
# it first paired, and the peer's last-seen feed epoch; only the routing key, the
# label, and updated_at change unconditionally. The signing public key
# refreshes WITH the route when the new pairing carries one, and is
# PRESERVED (COALESCE keeps the stored value over an absent excluded NULL) when
# it does not -- a re-pair via a pre-VL-01 payload can never strip a peer's
# registered key, which would silently downgrade it into the unsigned grace
# path. The pending state can only be RAISED here, never
# lowered -- a re-pair never confirms (only ``confirm_peer`` does) and never
# un-demotes; and a re-pair whose payload carries a signing key DIFFERENT from
# the stored one (or a first key over a previously unkeyed row) demotes the
# peer to pending value 2 (key changed: a new trust root is a new trust
# decision, re-confirmed by both humans). The unqualified column names in the
# CASE read the row's PRE-update values (standard UPDATE semantics), so the
# comparison is against the previously stored key even though signing_pub is
# refreshed in the same statement. A keyless re-pair (excluded NULL) never
# demotes: it carries no trust material.
_UPSERT = (
    f"INSERT INTO {_safe_table(TABLE_NAME)} (peer_id, routing_key, label, watermark, added_at, "
    "updated_at, signing_pub, pending) "
    "VALUES (?, ?, ?, 0, ?, ?, ?, ?) "
    "ON CONFLICT(peer_id) DO UPDATE SET "
    "routing_key = excluded.routing_key, "
    "label = excluded.label, "
    "updated_at = excluded.updated_at, "
    "pending = CASE "
    "WHEN excluded.signing_pub IS NOT NULL "
    "AND (signing_pub IS NULL OR signing_pub != excluded.signing_pub) "
    "THEN 2 ELSE pending END, "
    "signing_pub = COALESCE(excluded.signing_pub, signing_pub)"
)

_SELECT_COLUMNS = (
    "peer_id, routing_key, label, watermark, added_at, updated_at, last_epoch, "
    "signing_pub, pending, remote_chat_grant, rag_subgrant, device_class"
)

_SELECT_ONE = (
    f"SELECT {_SELECT_COLUMNS} FROM {_safe_table(TABLE_NAME)} WHERE peer_id = ?"
)

_SELECT_ALL = (
    f"SELECT {_SELECT_COLUMNS} FROM {_safe_table(TABLE_NAME)} ORDER BY added_at ASC, peer_id ASC"
)

_SELECT_WATERMARK = (
    f"SELECT watermark FROM {_safe_table(TABLE_NAME)} WHERE peer_id = ?"
)

# Monotonic advance: max() here is the SQLite scalar (two arguments), so the
# watermark only ever moves forward, atomically, within the statement.
_ADVANCE = (
    f"UPDATE {_safe_table(TABLE_NAME)} SET watermark = max(watermark, ?), updated_at = ? "
    "WHERE peer_id = ?"
)

# The per-peer last-seen feed epoch. Read it; store it on first
# contact (no reset); and the epoch reset -- the watermark back to 0 and the new
# epoch stored in ONE statement, so the pair is atomic by construction and a
# crash can never persist the new epoch while keeping the old watermark (which
# would silently skip the full resync). The reset is the deliberate, epoch-bound
# exception to the monotonic advance above.
_SELECT_LAST_EPOCH = (
    f"SELECT last_epoch FROM {_safe_table(TABLE_NAME)} WHERE peer_id = ?"
)

_SET_LAST_EPOCH = (
    f"UPDATE {_safe_table(TABLE_NAME)} SET last_epoch = ?, updated_at = ? WHERE peer_id = ?"
)

_RESET_FOR_EPOCH = (
    f"UPDATE {_safe_table(TABLE_NAME)} SET watermark = 0, last_epoch = ?, updated_at = ? "
    "WHERE peer_id = ?"
)

_DELETE_ONE = f"DELETE FROM {_safe_table(TABLE_NAME)} WHERE peer_id = ?"
_SELECT_COUNT = f"SELECT COUNT(*) FROM {_safe_table(TABLE_NAME)}"
_DELETE_ALL = f"DELETE FROM {_safe_table(TABLE_NAME)}"

# Activate a pending peer. The ONLY statement that lowers the
# pending state -- the upsert above can only raise it -- so an entry becomes
# trusted exclusively through the explicit human confirmation. Idempotent on an
# already-confirmed peer.
_CONFIRM = (
    f"UPDATE {_safe_table(TABLE_NAME)} SET pending = NULL, updated_at = ? WHERE peer_id = ?"
)

# The per-device remote-inference grants. Set the remote-chat
# enable bit (1 enabled, 0 disabled) or the RAG read-only sub-grant (1 on, 0 off).
# Neither column appears in the re-pair upsert's DO UPDATE clause, so a grant
# survives a route rotation; only these explicit setters and the control surface
# change it. A revoke is ``set_remote_chat_grant(peer_id, False)`` plus the
# in-memory live-session kill in remote_streaming -- no new revocation primitive.
_SET_REMOTE_CHAT_GRANT = (
    f"UPDATE {_safe_table(TABLE_NAME)} SET remote_chat_grant = ?, updated_at = ? WHERE peer_id = ?"
)

_SET_RAG_SUBGRANT = (
    f"UPDATE {_safe_table(TABLE_NAME)} SET rag_subgrant = ?, updated_at = ? WHERE peer_id = ?"
)

# The device-class marker write (NULL clears it).
_SET_DEVICE_CLASS = (
    f"UPDATE {_safe_table(TABLE_NAME)} SET device_class = ?, updated_at = ? WHERE peer_id = ?"
)

# The one-row local-identity meta table (SYN-02): this installation's stable
# device id, minted once. INSERT OR IGNORE under the CHECK(id = 1) constraint
# keeps the first minted value under concurrency.
_CREATE_META = (
    f"CREATE TABLE IF NOT EXISTS {_safe_table(META_TABLE_NAME)} ("
    "id INTEGER PRIMARY KEY CHECK (id = 1), "
    "device_id TEXT NOT NULL, "
    "created_at TEXT NOT NULL"
    ")"
)

_SELECT_DEVICE_ID = (
    f"SELECT device_id FROM {_safe_table(META_TABLE_NAME)} WHERE id = 1"
)

_INSERT_DEVICE_ID = (
    f"INSERT OR IGNORE INTO {_safe_table(META_TABLE_NAME)} (id, device_id, created_at) VALUES (1, ?, ?)"
)

# This device's own last-generated pairing material, pinned in
# the one-row meta table when the self payload is built. The confirmation code
# is derived from BOTH devices' public material; pinning this side's half at
# generation time keeps the code recomputable from local disk in any mode --
# no live-transport dependency, no drift if the route rotates after the
# payload was displayed (a rotation between the two halves of an exchange
# makes the codes visibly disagree, the documented re-run failure). Public
# material only (Kerckhoffs), like the device id beside it.
_PIN_SELF_MATERIAL = (
    f"UPDATE {_safe_table(META_TABLE_NAME)} SET self_pairing_material = ? WHERE id = 1"
)

_SELECT_SELF_MATERIAL = (
    f"SELECT self_pairing_material FROM {_safe_table(META_TABLE_NAME)} WHERE id = 1"
)


@dataclass(frozen=True)
class PeerRecord:
    """One paired peer: its identity, its public routing key, and its watermark.

    Attributes:
        peer_id: The stable identity of the peer device within this user's set.
        routing_key: The peer's public Veilid routing key; how a private route to
            it is opened. Public by design (Kerckhoffs); the secret is the user's.
        label: An optional human-readable name for the peer; informational.
        watermark: The last peer-feed sequence this device has applied; advanced
            monotonically after a sync round, never regressing.
        added_at: ISO-8601 timestamp of the first pairing; preserved on re-pair.
        updated_at: ISO-8601 timestamp of the last registry write for this peer.
        last_epoch: The last-seen epoch of this peer's change feed,
            or ``None`` for a pre-epoch or freshly-paired peer. Stored
            on first contact; replaced -- atomically with a watermark reset --
            when the peer's journal was recreated. Preserved on re-pair, like
            the watermark.
        signing_pub: The peer's ML-DSA-65 signing PUBLIC key, base64url,
            or ``None`` for a legacy peer whose pairing carried no
            key. Public material, so the registry is an acceptable home per
            Kerckhoffs (unlike the PRIVATE key, which never lands here --
            PEER-01's at-rest posture is the RS-01 lot's, not this one's).
            Refreshed with the route on a re-pair that carries a key;
            preserved when it does not.
        pending: ``True`` while the pairing awaits the PAIR-02 mutual
            confirmation. A pending peer gates nothing: the engine
            refuses to run a round against it or to serve it, and record
            verification never trusts its registered key. ``False`` for a
            confirmed peer -- including every pre-PAIR-02 row, grandfathered
            by the nullable migration.
        key_changed: ``True`` when the peer is pending because a re-pair
            carried a signing key different from the stored one (or a first
            key over a previously unkeyed row): the demotion case, surfaced
            distinctly from a fresh pairing so the human knows the trust root
            changed. Always ``False`` for a confirmed peer.
        remote_chat_enabled: Whether this device's remote-inference (remote
            chat) grant is enabled. ``True`` by default --
            the grandfathered tier-1 stance, and the value a row whose
            ``remote_chat_grant`` is NULL reads; ``False`` only when the user
            explicitly disabled it at the desktop control surface (column 0).
            The remote surface refuses a disabled device.
        rag_subgrant: Whether this device's RAG read-only sub-grant is on.
            ``False`` by default (the conservative default; a
            device can have remote chat without remote RAG). Gates the ``rag``
            scope on a remote request; off means the scope is refused.
        device_class: The peer's device class marker, or ``None``
            for the grandfathered desktop class (every pre-N.9 row, and any
            value outside the allowlist, reads ``None``). ``"phone"`` makes
            the sync responder serve this peer a NOTE record only when the
            note's per-item mobile-allowed flag affirmatively permits it
            (filter-at-serve, decision N9-D1). A local trust decision the
            desktop owns; never changed by a re-pair.
    """

    peer_id: str
    routing_key: str
    label: str = ""
    watermark: int = 0
    added_at: str = ""
    updated_at: str = ""
    last_epoch: str | None = None
    signing_pub: str | None = None
    pending: bool = False
    key_changed: bool = False
    remote_chat_enabled: bool = True
    rag_subgrant: bool = False
    device_class: str | None = None


def _check_peer_id(peer_id: object) -> None:
    if not isinstance(peer_id, str) or not peer_id:
        raise ValueError("peer_id must be a non-empty string")


def _check_routing_key(routing_key: object) -> None:
    if not isinstance(routing_key, str) or not routing_key:
        raise ValueError("routing_key must be a non-empty string")


def _check_watermark(watermark: object) -> int:
    if isinstance(watermark, bool) or not isinstance(watermark, int) or watermark < 0:
        raise ValueError("watermark must be a non-negative integer")
    return int(watermark)


def _check_epoch(epoch: object) -> str:
    if not isinstance(epoch, str) or not epoch:
        raise ValueError("epoch must be a non-empty string")
    return epoch


class PeerStore:
    """A registry of paired peers and their monotonic per-peer watermarks.

    The root is injectable for tests; with no root it resolves under the data
    directory. The connection is created lazily and guarded by a lock, so the
    store is safe to share across threads.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root: Path | None = Path(root) if root is not None else None
        self._db_path: Path | None = None
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _resolve_db_path(self) -> Path:
        if self._root is not None:
            base = self._root
        else:
            from opti_oignon.config import DATA_DIR

            base = Path(DATA_DIR)
        return base / DB_FILENAME

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            self._db_path = self._resolve_db_path()
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = _safe_connect(
                str(self._db_path), check_same_thread=False, timeout=5.0
            )
            # WAL is set outside any transaction, right after connect.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            # Add the
            # last-seen-epoch, signing-public-key, and pending columns to
            # registries that predate them. SQLite has no ADD COLUMN IF NOT
            # EXISTS, so guard with table_info (the AU-06 idiom). Nullable on
            # purpose: NULL means a pre-epoch peer, a pre-VL-01 peer with no
            # registered key, respectively a CONFIRMED peer -- the PAIR-02
            # grandfather falls out of the migration by construction (every
            # pre-existing row reads NULL = confirmed; only new ceremonies
            # write 1 and only a key-change demotion writes 2).
            cols = {
                row[1]
                for row in conn.execute(
                    f"PRAGMA table_info({_safe_table(TABLE_NAME)})"
                ).fetchall()
            }
            if "last_epoch" not in cols:
                conn.execute(
                    f"ALTER TABLE {_safe_table(TABLE_NAME)} ADD COLUMN last_epoch TEXT"
                )
            if "signing_pub" not in cols:
                conn.execute(
                    f"ALTER TABLE {_safe_table(TABLE_NAME)} ADD COLUMN signing_pub TEXT"
                )
            if "pending" not in cols:
                conn.execute(
                    f"ALTER TABLE {_safe_table(TABLE_NAME)} ADD COLUMN pending INTEGER"
                )
            # The per-device remote-inference grant columns,
            # the same additive guarded idiom. Nullable on purpose: NULL means a
            # grandfathered peer -- remote chat ENABLED (the tier-1 default) and
            # the RAG read-only sub-grant OFF (the conservative default).
            if "remote_chat_grant" not in cols:
                conn.execute(
                    f"ALTER TABLE {_safe_table(TABLE_NAME)} ADD COLUMN remote_chat_grant INTEGER"
                )
            if "rag_subgrant" not in cols:
                conn.execute(
                    f"ALTER TABLE {_safe_table(TABLE_NAME)} ADD COLUMN rag_subgrant INTEGER"
                )
            # The per-device class marker, the same additive
            # guarded idiom. Nullable on purpose: NULL is the grandfathered
            # desktop class (every pre-N.9 row is a desktop by construction);
            # only an explicit ceremony or the control surface writes
            # "phone".
            if "device_class" not in cols:
                conn.execute(
                    f"ALTER TABLE {_safe_table(TABLE_NAME)} ADD COLUMN device_class TEXT"
                )
            conn.execute(_CREATE_META)
            # The pinned self pairing material on the meta
            # table, same additive guarded idiom.
            meta_cols = {
                row[1]
                for row in conn.execute(
                    f"PRAGMA table_info({_safe_table(META_TABLE_NAME)})"
                ).fetchall()
            }
            if "self_pairing_material" not in meta_cols:
                conn.execute(
                    f"ALTER TABLE {_safe_table(META_TABLE_NAME)} ADD COLUMN "
                    "self_pairing_material TEXT"
                )
            conn.commit()
            self._connection = conn
        return self._connection

    def local_device_id(self) -> str:
        """This installation's stable device identity, minted once (SYN-02).

        A random per-install identifier (uuid4 hex) created lazily in the
        one-row meta table next to the peer registry. Pairing payloads and
        record provenance need a per-device identity -- every device naming
        itself "local" made pairing peer_ids collide (a later upsert overwrote
        an earlier peer's routing key) and made the reconciler's device
        tie-break meaningless. The registry is where the identity naturally
        lives: it survives a change-feed reset and dies only with the pairing
        registry itself, where re-pairing is required anyway. INSERT OR IGNORE
        keeps the first minted value under concurrency; the value is public
        material (Kerckhoffs), not a secret.
        """
        candidate = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._conn()
            conn.execute(_INSERT_DEVICE_ID, (candidate, now))
            conn.commit()
            row = conn.execute(_SELECT_DEVICE_ID).fetchone()
        return str(row[0]) if row is not None and row[0] else candidate

    @property
    def db_path(self) -> Path:
        if self._db_path is None:
            self._db_path = self._resolve_db_path()
        return self._db_path

    def journal_mode(self) -> str:
        """The active journal mode; expected to be 'wal'."""
        with self._lock:
            conn = self._conn()
            row = conn.execute("PRAGMA journal_mode").fetchone()
        return (row[0] if row else "").lower()

    @staticmethod
    def _row_to_record(row: tuple) -> PeerRecord:
        (
            peer_id,
            routing_key,
            label,
            watermark,
            added_at,
            updated_at,
            last_epoch,
            signing_pub,
            pending_raw,
            remote_chat_grant_raw,
            rag_subgrant_raw,
            device_class_raw,
        ) = row
        # PAIR-02 encoding: NULL/0 confirmed, 1 fresh pending, 2 demoted
        # (signing key changed on re-pair). Read defensively: anything that is
        # not a positive integer is confirmed (the grandfathered shape).
        pending_value = pending_raw if isinstance(pending_raw, int) else 0
        # The grant encoding, read defensively: remote chat is
        # ENABLED unless the column is explicitly 0 (NULL = grandfathered =
        # enabled); the RAG sub-grant is ON only when the column is explicitly 1
        # (NULL/0 = off).
        remote_chat_enabled = not (
            isinstance(remote_chat_grant_raw, int)
            and int(remote_chat_grant_raw) == 0
        )
        rag_subgrant_on = (
            isinstance(rag_subgrant_raw, int) and int(rag_subgrant_raw) == 1
        )
        return PeerRecord(
            peer_id=peer_id,
            routing_key=routing_key,
            label=label or "",
            watermark=int(watermark),
            added_at=added_at or "",
            updated_at=updated_at or "",
            last_epoch=(
                last_epoch if isinstance(last_epoch, str) and last_epoch else None
            ),
            signing_pub=(
                signing_pub
                if isinstance(signing_pub, str) and signing_pub
                else None
            ),
            pending=pending_value > 0,
            key_changed=pending_value == 2,
            remote_chat_enabled=remote_chat_enabled,
            rag_subgrant=rag_subgrant_on,
            # The class encoding, read defensively: only an
            # allowlisted string is a class; NULL and anything else read None
            # (the grandfathered desktop class).
            device_class=(
                device_class_raw
                if isinstance(device_class_raw, str)
                and device_class_raw in DEVICE_CLASSES
                else None
            ),
        )

    def add_peer(
        self,
        peer_id: str,
        routing_key: str,
        *,
        label: str = "",
        signing_pub: str | None = None,
        pending: bool = False,
    ) -> PeerRecord:
        """Register a peer, or refresh an existing one's routing key and label.

        A fresh peer starts at watermark 0. Re-registering an existing peer (a
        re-pair with a rotated route) updates the routing key and label but
        preserves the watermark and the original pairing time, so the device
        never loses track of how far it has synced. The signing public key
        refreshes with the route when ``signing_pub`` is given
        and is preserved when it is ``None`` (a pre-VL-01 payload can never
        strip a registered key). ``pending=True`` registers a
        fresh peer awaiting the mutual confirmation -- the pairing ceremony's
        path; the default ``False`` is the programmatic-registration trust
        decision (and the grandfathered shape). On a re-pair the caller's flag
        is ignored: the upsert can only RAISE the pending state, never lower
        it (only :meth:`confirm_peer` activates an entry), and a re-pair whose
        key DIFFERS from the stored one (or lands a first key over an unkeyed
        row) demotes the peer to the key-changed pending state, logged here
        and surfaced to the human by the pairing panel. Returns the stored
        record.
        """
        _check_peer_id(peer_id)
        _check_routing_key(routing_key)
        if not isinstance(label, str):
            raise ValueError("label must be a string")
        if signing_pub is not None and (
            not isinstance(signing_pub, str) or not signing_pub
        ):
            raise ValueError("signing_pub must be a non-empty string or None")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._conn()
            if signing_pub is not None:
                prev = conn.execute(_SELECT_ONE, (peer_id,)).fetchone()
                if prev is not None:
                    stored = prev[7]
                    if (
                        isinstance(stored, str)
                        and stored
                        and stored != signing_pub
                    ):
                        logger.warning(
                            "peer %s signing key CHANGED on re-pair; the new "
                            "key replaces the stored one and the peer is "
                            "DEMOTED to pending (PAIR-02): both humans must "
                            "re-confirm the pairing before it is trusted "
                            "again. If this re-pair was not yours, treat the "
                            "peer as compromised and reject it.",
                            peer_id,
                        )
                    elif not (isinstance(stored, str) and stored):
                        logger.info(
                            "peer %s acquired its first signing key on "
                            "re-pair; the peer is demoted to pending "
                            "(PAIR-02): a new trust root requires the mutual "
                            "confirmation.",
                            peer_id,
                        )
            conn.execute(
                _UPSERT,
                (
                    peer_id,
                    routing_key,
                    label,
                    now,
                    now,
                    signing_pub,
                    1 if pending else 0,
                ),
            )
            conn.commit()
            row = conn.execute(_SELECT_ONE, (peer_id,)).fetchone()
        return self._row_to_record(row)

    def confirm_peer(self, peer_id: str) -> bool:
        """Activate a pending peer (PAIR-02): the human confirmed the code.

        The only path that lowers the pending state -- the registration upsert
        can only raise it -- so an entry becomes trusted exclusively through
        this explicit confirmation. Idempotent: confirming an
        already-confirmed peer is a harmless no-op that still returns ``True``
        (the row exists); an unknown peer returns ``False``.
        """
        if not isinstance(peer_id, str) or not peer_id:
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._conn()
            cur = conn.execute(_CONFIRM, (now, peer_id))
            conn.commit()
            return cur.rowcount > 0

    def set_remote_chat_grant(self, peer_id: str, enabled: bool) -> bool:
        """Enable or disable a device's remote-inference grant (cas 7 Lot 2).

        Writes the explicit bit (1 enabled, 0 disabled) on the peer row. A
        local trust decision the desktop control surface owns; it is never
        changed by a re-pair (the upsert preserves it). Disabling is the
        durable half of a revoke -- the live half (dropping the device's
        in-flight streaming buffers) is the caller's, via
        ``remote_streaming.kill_sessions_for_device``. Returns ``True`` when a
        row was updated, ``False`` for an unknown peer.
        """
        if not isinstance(peer_id, str) or not peer_id:
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                _SET_REMOTE_CHAT_GRANT, (1 if enabled else 0, now, peer_id)
            )
            conn.commit()
            return cur.rowcount > 0

    def set_rag_subgrant(self, peer_id: str, granted: bool) -> bool:
        """Turn a device's RAG read-only sub-grant on or off (cas 7 Lot 2).

        Writes the explicit bit (1 on, 0 off). Off by default and never changed
        by a re-pair. Gates the ``rag`` scope on a remote request: off means the
        scope is refused, so a device can have remote chat without remote RAG.
        Returns ``True`` when a row was updated, ``False`` for an unknown peer.
        """
        if not isinstance(peer_id, str) or not peer_id:
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                _SET_RAG_SUBGRANT, (1 if granted else 0, now, peer_id)
            )
            conn.commit()
            return cur.rowcount > 0

    def set_device_class(
        self, peer_id: str, device_class: str | None
    ) -> bool:
        """Mark or clear a peer's device class.

        ``"phone"`` marks the peer phone-class: the sync responder then
        serves it a NOTE record only when the note's per-item mobile-allowed
        flag affirmatively permits it (filter-at-serve, decision N9-D1).
        ``None`` clears the marker back to the grandfathered desktop class.
        A local trust decision the desktop owns -- written at the pairing
        ceremony or from the control surface -- and never changed by a
        re-pair (the upsert does not touch the column). Returns ``True``
        when a row was updated, ``False`` for an unknown peer; a value
        outside :data:`DEVICE_CLASSES` raises (a programming error, not a
        trust state).
        """
        if not isinstance(peer_id, str) or not peer_id:
            return False
        if device_class is not None and device_class not in DEVICE_CLASSES:
            raise ValueError(
                f"device_class must be one of {sorted(DEVICE_CLASSES)} or None"
            )
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._conn()
            cur = conn.execute(_SET_DEVICE_CLASS, (device_class, now, peer_id))
            conn.commit()
            return cur.rowcount > 0

    def pin_self_pairing_material(self, material: str) -> None:
        """Pin this device's own last-generated pairing material (PAIR-02).

        Written when the self pairing payload is built, so the confirmation
        code recomputes deterministically from local disk in any mode -- no
        live-transport read, no drift if the route rotates after the payload
        was displayed. Public material only (Kerckhoffs). Ensures the one-row
        meta exists (the same lazy mint as the device identity) and replaces
        the previous pin; regenerating the payload re-pins.
        """
        if not isinstance(material, str) or not material:
            raise ValueError("material must be a non-empty string")
        candidate = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._conn()
            conn.execute(_INSERT_DEVICE_ID, (candidate, now))
            conn.execute(_PIN_SELF_MATERIAL, (material,))
            conn.commit()

    def get_self_pairing_material(self) -> str | None:
        """This device's pinned pairing material, or ``None`` when never pinned."""
        with self._lock:
            conn = self._conn()
            row = conn.execute(_SELECT_SELF_MATERIAL).fetchone()
        if row is None:
            return None
        value = row[0]
        return value if isinstance(value, str) and value else None

    def get_peer(self, peer_id: str) -> PeerRecord | None:
        """Return a peer's record, or ``None`` when it is not registered."""
        if not isinstance(peer_id, str) or not peer_id:
            return None
        with self._lock:
            conn = self._conn()
            row = conn.execute(_SELECT_ONE, (peer_id,)).fetchone()
        return self._row_to_record(row) if row is not None else None

    def has_peer(self, peer_id: str) -> bool:
        """True when a peer is registered."""
        return self.get_peer(peer_id) is not None

    def list_peers(self) -> list[PeerRecord]:
        """All registered peers, ordered by pairing time then peer id."""
        with self._lock:
            conn = self._conn()
            rows = conn.execute(_SELECT_ALL).fetchall()
        return [self._row_to_record(r) for r in rows]

    def remove_peer(self, peer_id: str) -> bool:
        """Remove a peer; returns True when a row was deleted, False otherwise."""
        if not isinstance(peer_id, str) or not peer_id:
            return False
        with self._lock:
            conn = self._conn()
            cur = conn.execute(_DELETE_ONE, (peer_id,))
            conn.commit()
            return cur.rowcount > 0

    def get_watermark(self, peer_id: str) -> int:
        """The peer's watermark, or 0 when the peer is not registered."""
        if not isinstance(peer_id, str) or not peer_id:
            return 0
        with self._lock:
            conn = self._conn()
            row = conn.execute(_SELECT_WATERMARK, (peer_id,)).fetchone()
        return int(row[0]) if row is not None else 0

    def advance_watermark(self, peer_id: str, watermark: int) -> int:
        """Advance a peer's watermark to ``max(current, watermark)``; returns the new value.

        Monotonic at the SQL level: a smaller or equal incoming value leaves the
        stored watermark unchanged, so a late or duplicate answer never regresses
        it. Advancing a peer that is not registered is a no-op that returns 0; the
        sync engine refuses to run a round against an unpaired peer, so this path
        is defensive rather than a registration shortcut.
        """
        _check_peer_id(peer_id)
        w = _check_watermark(watermark)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._conn()
            conn.execute(_ADVANCE, (w, now, peer_id))
            conn.commit()
            row = conn.execute(_SELECT_WATERMARK, (peer_id,)).fetchone()
        return int(row[0]) if row is not None else 0

    def get_last_epoch(self, peer_id: str) -> str | None:
        """The peer's last-seen feed epoch, or ``None`` (CHF-05).

        ``None`` covers a pre-epoch peer (its batches carry no epoch yet), a
        freshly-paired peer (no round consumed), and an unregistered peer --
        all three mean the same to the engine: nothing to compare against.
        """
        if not isinstance(peer_id, str) or not peer_id:
            return None
        with self._lock:
            conn = self._conn()
            row = conn.execute(_SELECT_LAST_EPOCH, (peer_id,)).fetchone()
        if row is None:
            return None
        value = row[0]
        return value if isinstance(value, str) and value else None

    def set_last_epoch(self, peer_id: str, epoch: str) -> bool:
        """Store a peer's feed epoch without touching its watermark (CHF-05).

        The first-contact path: the engine stores the first epoch a peer
        advertises and compares from there; resetting on first contact would
        force a full resync on every fleet upgrade for nothing. Returns True
        when a registered peer was updated; an unknown peer is a no-op
        returning False, like ``advance_watermark`` (defensive, not a
        registration shortcut).
        """
        _check_peer_id(peer_id)
        _check_epoch(epoch)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._conn()
            cur = conn.execute(_SET_LAST_EPOCH, (epoch, now, peer_id))
            conn.commit()
            return cur.rowcount > 0

    def reset_for_epoch(self, peer_id: str, epoch: str) -> bool:
        """Reset a peer's watermark to 0 and store its new epoch, atomically.

        The CHF-05 repair for a recreated peer journal: the watermark goes back
        to 0 (a single full resync; applies are idempotent) and the new epoch
        is stored, in ONE statement under one commit, so a crash can never
        persist the new epoch while keeping the old watermark -- the
        interleaving that would silently skip the resync. The deliberate,
        epoch-bound exception to the monotonic advance: ``advance_watermark``
        can never move a watermark down. Returns True when a registered peer
        was reset; an unknown peer is a no-op returning False.
        """
        _check_peer_id(peer_id)
        _check_epoch(epoch)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._conn()
            cur = conn.execute(_RESET_FOR_EPOCH, (epoch, now, peer_id))
            conn.commit()
            return cur.rowcount > 0

    def count(self) -> int:
        """The number of registered peers."""
        with self._lock:
            conn = self._conn()
            row = conn.execute(_SELECT_COUNT).fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        """Remove every peer (the file and connection are kept)."""
        with self._lock:
            conn = self._conn()
            conn.execute(_DELETE_ALL)
            conn.commit()

    def close(self) -> None:
        """Close the underlying connection, if open."""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                finally:
                    self._connection = None


# Module-level singleton with a reset hook (one store per process, testable).

_store: PeerStore | None = None
_store_lock = threading.Lock()


def get_peer_store(root: Path | str | None = None) -> PeerStore:
    """Return the process peer store, creating it once (with ``root`` if given)."""
    global _store
    with _store_lock:
        if _store is None:
            _store = PeerStore(root=root)
        return _store


def set_peer_store(store: PeerStore | None) -> None:
    """Install a specific store as the process singleton (used by tests)."""
    global _store
    with _store_lock:
        _store = store


def reset_peer_store() -> None:
    """Close and clear the process singleton so the next get creates a fresh one."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None
