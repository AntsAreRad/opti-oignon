#!/usr/bin/env python3
"""Web-free sync engine for Veilid sync (S180 Goal 2, Theme 4).

The engine is the thing that runs a sync round. It sits between the per-device
change feed (the local journal of what this device has changed), the per-peer
store (which peers are paired and how far each has been consumed), and the
transport-agnostic protocol envelope (which turns a watermark into a request, a
request into a batch, and a batch into a reconciled local set). It owns no socket
and no web stack, so the whole round is exercised in isolation with an injected
fake peer; the live Veilid transport that drives the round over the node and
client across a private route lands in S181, with the peer still injectable here.

A round is a pull. The engine resolves the peer's watermark from the store, asks
the peer for the delta since that watermark, reconciles the answer into the local
set (journalling only what changed, so a round is idempotent), and advances the
peer's watermark monotonically. It returns a structured summary: how much was
applied, how many concurrent divergences were retained, how many wire records were
dropped, the watermark before and after, and whether the watermark moved.

Two disciplines hold at the engine seam. Bulbe: every path that acts on records
over the wire calls the binding-layer gate before it acts, so a round refuses
under Bulbe at the binding layer, not by a policy flag -- the same gate the
protocol envelope already enforces, re-asserted here so the engine is an honest
seam in its own right. The pure result-shaping helper stays ungated, because
shaping a summary is not acting on the wire.

Approval: applying a record that introduces executable surface -- a skill -- is a
sensitive action, so it follows the same approval-aware path as the agent's
``manage_skills`` and ``manage_memory`` writes. Each sensitive record in an
incoming batch passes the fail-secure human gate (an injected ``approval_fn``, or
the default manager-backed ``allowlists.request_approval``); a record that is not
approved is deferred rather than applied. Since S207 (SYN-05) a deferred record
is PERSISTED to the per-record deferred ledger (``deferred_ledger.py``) -- the
full wire envelope, so it re-offers without a re-fetch -- and the watermark
ADVANCES past it: a permanently-unapproved record no longer pins the peer's
watermark or re-fetches the whole growing delta. A still-deferred record arriving
again dedups into its ledger entry silently (no re-prompt); the human decides
from the SyncPanel pending-approval list, where an approval re-enters this
engine's verify -> gate -> apply seam against the CURRENT trust state and a
refusal removes the entry without applying. Fail-secure throughout: deferred
means NOT applied. Conversation and memory records are user data, not
executable surface, so they apply without a gate. Under Bulbe the round never runs
at all, so a sensitive apply is refused at the gate by construction; the ledger
itself is local-disk state and its list/approve/refuse are local decisions,
permitted in any mode like pairing management.

Every round and every peer change is recorded in the hash-chain audit log,
best-effort and lazily imported so the engine stays isolatable. The producers that
turn a conversation, a memory entry, or a skill into a record are minimal here
(``record_from_payload`` over an opaque payload); the domain-specific producers
fill in as the syncable stores are wired through, and the panel to pair devices
and control what is shared lands in S182.

Kerckhoffs: the engine is open. Peers are addressed by public routing keys the
user holds; nothing about a round depends on the secrecy of the mechanism.
"""

from __future__ import annotations

import base64
import logging
import sys
import threading
from dataclasses import dataclass
from typing import Any, Mapping

from opti_oignon.veilid.change_feed import ChangeFeed, get_change_feed
from opti_oignon.veilid.deferred_ledger import (
    OFFER_INSERTED,
    OFFER_REPLACED,
    DeferredEntry,
    DeferredLedger,
    get_deferred_ledger,
)
from opti_oignon.veilid.guard import assert_sync_allowed
from opti_oignon.veilid.peers import (
    DEVICE_CLASS_PHONE,
    PeerRecord,
    PeerStore,
    get_peer_store,
)
from opti_oignon.veilid.producers import (
    conversation_record,
    memory_archive_record,
    memory_canonical_record,
    note_record,
    note_update_record,
    skill_record,
)
from opti_oignon.veilid.protocol import (
    Peer,
    RecordBatch,
    apply_local_batch,
    apply_record_batch,
    build_delta_request,
    parse_record_batch,
    respond_to_request,
)
from opti_oignon.veilid.records import (
    RecordKind,
    SyncRecord,
    decode_record,
    new_record,
)

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# The kinds whose application is a sensitive action and must pass the human gate.
# A skill carries executable surface, so adopting one over sync is gated the same
# way an agent's manage_skills write is. Conversation and memory records are user
# data and apply without a gate.
SENSITIVE_KINDS: frozenset[str] = frozenset({RecordKind.SKILL.value})

# S203 (PRT-04): the per-round leg bound. A round loops internally, consuming one
# bounded chunk per leg and threading each chunk's high-water into the next
# request, until the watermark stops advancing (caught up), the peer's answer
# fails to parse, or this many legs have run. (Since S207 a deferral no longer
# ends the loop: the deferred record is persisted to the ledger and the round
# keeps consuming.) The bound is a guard against a misbehaving peer that never
# reports caught-up; a real first sync of a personal-scale dataset converges in
# far fewer legs.
MAX_LEGS = 1024


class PeerNotFound(Exception):
    """A sync round or status read referenced a peer that is not paired."""


class DeferredNotFound(Exception):
    """A ledger action referenced a (kind, record_id) with no pending entry.

    S207 (SYN-05): approving or refusing from the pending-approval list names
    an entry by its key; an unknown key raises this (the route maps it to a
    404). Listing never raises.
    """


class PeerNotConfirmed(Exception):
    """A wire action referenced a peer whose pairing awaits confirmation.

    PAIR-02 (S206): the pairing ceremony registers a peer PENDING; until both
    humans have compared the mutual confirmation code and confirmed on both
    devices, the entry gates nothing -- a round against it and serving it
    refuse with this exception (the route maps it to a 409 with an explicit
    detail), and record verification never trusts its registered key.
    """


# PAIR-02 (S206): the sentinel _verify_records caches for an origin whose
# registry entry is PENDING. Distinct from None (no key at all, the migration
# grace case) on purpose: a pending origin's records are REFUSED outright --
# its registered material is not yet human-confirmed, so it must neither
# verify as trusted nor fall back into the unsigned grace path (which would be
# strictly worse than refusing).
_PENDING_ORIGIN = object()


@dataclass(frozen=True)
class RoundResult:
    """The structured outcome of one sync round.

    Attributes:
        peer_id: The peer the round ran against.
        applied: How many records were newly adopted into the local feed (a key new
            to this device, a content change, or a clock advance); idempotent, so a
            re-run applies 0.
        deferred: How many sensitive records were newly quarantined to the
            deferred ledger this round (inserted, or replacing a stored older
            version) rather than applied (S207, SYN-05). A still-pending
            record arriving again is the silent dedup -- refreshed, not
            counted, not re-prompted. Deferred means NOT applied; the human
            decides from the pending-approval list.
        conflicts: How many concurrent divergences were retained in the conflict
            log (kept, not dropped, for a later merge or human review).
        rejected: How many wire records the batch dropped on decode.
        previous_watermark: The peer's watermark before the round.
        new_watermark: The peer's watermark after the round.
        advanced: Whether the watermark moved forward this round. False when
            nothing new arrived. Since S207 (SYN-05) a deferral no longer
            holds the watermark: the deferred record is persisted to the
            ledger and every consumed chunk advances, so ``advanced`` reports
            persisted movement regardless of deferrals. False when the
            peer vanished mid-round (the advance is a no-op then; SYN-07). After
            an epoch reset (CHF-05) the comparison baseline is the reset (0), not
            the pre-round watermark: a full resync can converge below the stale
            watermark and still be real, persisted progress.
        parsed: False when the peer's answer could not be parsed as a record
            batch (a malformed or garbled reply); the round degraded and held the
            watermark at the last consumed chunk boundary. True for a real
            (possibly empty) batch (SYN-03) -- the status surface records a
            failure, not a clean round.
        legs: How many chunk legs the round ran (PRT-04). 1 for a single-chunk
            round (the common case and every pre-PRT-04 round); more when a delta
            spanned several bounded chunks.
        epoch_reset: True when the peer's feed epoch changed this round (its
            journal was recreated; S204, CHF-05) and the watermark was reset to
            0 -- atomically with the new epoch -- for a full resync over the
            normal leg loop. An internal diagnostic like ``legs``; the API round
            payload is unchanged.
        refused: How many records were refused at the signature seam (S205,
            VL-01): unsigned or invalid-signature records whose ORIGIN device
            has a registered signing key, including a record signed by some
            OTHER device's key (the lookup by origin IS the
            device<->key binding). Refused-and-counted, never applied, never
            an approval prompt; a refusal does NOT hold the watermark (a
            forged record must not be able to pin convergence -- the
            ``rejected`` decode-drop semantics, not the fail-secure
            ``deferred`` quarantine). Refused records do NOT enter the
            deferred ledger (S207 decision): a forgery must never sit one
            click from application, and even a visibility-only row would put
            attacker-controlled content in the panel; the counters here, the
            per-peer status surface, and the audit log are the visibility.
        unverified: How many records were accepted WITHOUT verification.
            Since S208 (the grace flip; 3.7.0) the open-window case is gone --
            an unkeyed-origin record refuses -- so a non-zero count here means
            either this device cannot verify at all (no PQC backend, the
            pre-VL-01 posture, warned) or a test re-opened
            ``signing.ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS`` by monkeypatch.
            Counted so the operator sees an unverifying device honestly.
    """

    peer_id: str
    applied: int = 0
    deferred: int = 0
    conflicts: int = 0
    rejected: int = 0
    previous_watermark: int = 0
    new_watermark: int = 0
    advanced: bool = False
    parsed: bool = True
    legs: int = 1
    epoch_reset: bool = False
    refused: int = 0
    unverified: int = 0


def record_from_payload(
    kind: Any,
    record_id: str,
    payload: Mapping[str, Any],
    *,
    device: str,
    clock: int,
    deleted: bool = False,
    updated_at: str = "",
) -> SyncRecord:
    """Build a sync record from an opaque payload: the minimal producer surface.

    A thin, validating wrapper over :func:`records.new_record`. The domain
    producers that turn a conversation, a memory entry, or a skill into a record
    fill in as those stores are wired through; until then a caller (or a test)
    publishes a record built here. Pure: it computes the content hash and opens
    no socket.
    """
    return new_record(
        kind,
        record_id,
        payload,
        device=device,
        clock=clock,
        deleted=deleted,
        updated_at=updated_at,
    )


def _round_result(
    peer_id: str,
    apply_result: Any,
    *,
    previous_watermark: int,
    new_watermark: int,
    deferred: int,
    advanced: bool,
) -> RoundResult:
    """Shape an apply result and the watermarks into a round summary. Pure."""
    return RoundResult(
        peer_id=peer_id,
        applied=int(getattr(apply_result, "applied", 0)),
        deferred=int(deferred),
        conflicts=len(getattr(apply_result, "conflicts", []) or []),
        rejected=int(getattr(apply_result, "rejected", 0)),
        previous_watermark=int(previous_watermark),
        new_watermark=int(new_watermark),
        advanced=bool(advanced),
    )


def _audit(action: str, **details: Any) -> None:
    """Record a sync event in the hash-chain audit log, best-effort.

    Lazy and guarded so it never raises and the engine stays isolatable, the same
    idiom as the agent's skills audit hook.
    """
    try:
        from opti_oignon.signed_audit_log import chain_log

        chain_log(
            event_type="veilid_sync",
            source="veilid.sync_engine",
            action=action,
            severity="INFO",
            **details,
        )
    except Exception:  # pragma: no cover - audit is best-effort
        logger.debug("sync audit log unavailable", exc_info=True)


def _default_note_gate() -> Any | None:
    """The process-default note gate: the ALREADY-initialised notes store.

    N.9 (S256). Read-only and side-effect free by design: it never
    constructs a store (serving a peer must not seed a database), it only
    consults the module singleton when the application has already opened
    it, through the store's fail-secure ``is_mobile_allowed``. Anything else
    -- the notes module absent, the singleton not initialised, the reader
    missing -- resolves to ``None``, and a ``None`` gate is fail-secure at
    the phone-class filter (decision N9-D2: the record is excluded).
    """
    try:
        from opti_oignon.notes import notes_store as _ns
    except Exception:
        return None
    store = getattr(_ns, "_store", None)
    if store is None:
        return None
    reader = getattr(store, "is_mobile_allowed", None)
    return reader if callable(reader) else None


def _update_sink_for(store_like: Any) -> Any:
    """The canonical note_update apply sink over an update-store-like object.

    S264 (NOTES_CRDT_SPEC.md sections 3 and 5): the ONE recipe that lands a
    received ``note_update`` record through the S263 store's append seam, used
    by the lazy process default and injectable over any object exposing
    ``append_update`` (tests wire a real store at a tmp root). Returns a
    callable ``record -> bool``: True means landed (the record may proceed to
    the feed journal and be served onward), False means REFUSED -- not
    appended, not served, not rendered (section 5). Fail-secure throughout:

    - Attribution: the payload must carry a non-empty string ``note_id``, a
      positive integer ``seq`` (bool rejected), and a base64 ``update_blob_b64``
      that decodes strictly; the record identity must equal ``note_id:seq``
      (a re-coordinated payload cannot be attributed and refuses).
    - The author's order is PRESERVED: the append uses the explicit wire
      ``seq``, never a local re-mint, and ``author_device`` is the record's
      ORIGIN device -- the same identity the signature seam verified --
      never a payload hint.
    - The landing suppresses the store's own publish glue
      (``sync_publish=False``): the engine journals the received record
      verbatim, signature preserved, so re-publishing would re-sign the
      author's update as ours.
    - Any store refusal (unknown or dead parent, indeterminable liveness,
      duplicate seq) or any error refuses; the store logs its reason, the
      engine audits the refusal at the round seam.
    """

    def _sink(record: SyncRecord) -> bool:
        payload = record.payload or {}
        note_id = payload.get("note_id")
        seq = payload.get("seq")
        blob_b64 = payload.get("update_blob_b64")
        if not isinstance(note_id, str) or not note_id:
            return False
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            return False
        if record.record_id != f"{note_id}:{seq}":
            return False
        if not isinstance(blob_b64, str):
            return False
        try:
            blob = base64.b64decode(blob_b64.encode("ascii"), validate=True)
        except Exception:
            return False
        try:
            store_like.append_update(
                note_id,
                blob,
                author_device=record.device,
                seq=seq,
                sync_publish=False,
            )
        except Exception:
            logger.debug(
                "note_update landing refused for %s", record.record_id,
                exc_info=True,
            )
            return False
        return True

    return _sink


def _default_update_sink() -> Any | None:
    """The process-default note_update apply sink, or ``None``.

    The ``_default_note_gate`` idiom (S256): read-only and side-effect free,
    it never constructs a store (applying a round must not seed a database),
    it only consults the ALREADY-initialised update-store singleton. Anything
    else resolves to ``None``, and a ``None`` sink is fail-secure at the
    landing seam (the record is refused: not appended, not served).
    """
    try:
        from opti_oignon.notes import note_updates_store as _nus
    except Exception:
        return None
    store = getattr(_nus, "_store", None)
    if store is None:
        return None
    if not callable(getattr(store, "append_update", None)):
        return None
    return _update_sink_for(store)


def _default_conversation_sink() -> Any | None:
    """The process-default CONVERSATION apply sink, or ``None``.

    The ``_default_update_sink`` idiom (S199, SYN-01): read-only and
    side-effect free. It does NOT import (and thereby construct) the
    conversation manager -- applying a round must not seed a database -- it
    only consults the manager when the application has ALREADY opened the
    conversation module. Anything else resolves to ``None``, fail-secure at the
    landing seam (the record is dropped: not materialised, not served onward).
    The sink is hook-free by construction (``apply_synced_conversation`` never
    re-publishes), so a landed record is journalled verbatim, signature
    preserved.
    """
    mod = sys.modules.get("opti_oignon.conversation")
    if mod is None:
        return None
    manager = getattr(mod, "conversation_manager", None)
    apply = getattr(manager, "apply_synced_conversation", None)
    if not callable(apply):
        return None

    def _sink(record: SyncRecord) -> bool:
        try:
            return bool(
                apply(
                    record.payload,
                    deleted=record.deleted,
                    updated_at=record.updated_at,
                )
            )
        except Exception:
            logger.debug(
                "conversation sink raised for %s",
                getattr(record, "record_id", "?"),
                exc_info=True,
            )
            return False

    return _sink


def _default_note_sink() -> Any | None:
    """The process-default NOTE apply sink, or ``None``.

    The ``_default_note_gate`` idiom (S199, SYN-01): read-only, no seed -- it
    only consults the already-initialised notes store singleton (importing the
    module is cheap; the ``_store`` singleton is None until the application
    opens it). Anything else resolves to ``None``, fail-secure at the landing
    seam. The sink is hook-free (``apply_synced_note`` never re-publishes), so a
    landed record is journalled verbatim, signature preserved. The note id is
    the record id (the note payload does not carry it).
    """
    try:
        from opti_oignon.notes import notes_store as _ns
    except Exception:
        return None
    store = getattr(_ns, "_store", None)
    apply = getattr(store, "apply_synced_note", None)
    if not callable(apply):
        return None

    def _sink(record: SyncRecord) -> bool:
        try:
            return bool(
                apply(
                    record.record_id,
                    record.payload,
                    deleted=record.deleted,
                    updated_at=record.updated_at,
                )
            )
        except Exception:
            logger.debug(
                "note sink raised for %s",
                getattr(record, "record_id", "?"),
                exc_info=True,
            )
            return False

    return _sink


def _default_memory_canonical_sink() -> Any | None:
    """The process-default MEMORY_CANONICAL apply sink, or ``None``.

    The ``_default_note_sink`` idiom (SYN-01, Direction D): read-only, no seed
    -- it only consults the already-initialised canonical-memory store
    singleton (importing the module is cheap; the ``_store`` singleton is None
    until the application opens it). Anything else resolves to ``None``,
    fail-secure at the landing seam. The sink is hook-free
    (``apply_synced_memory_canonical`` never re-publishes), so a landed record
    is journalled verbatim, signature preserved. The fact id is the record id.
    """
    try:
        from opti_oignon.memory import canonical_store as _cs
    except Exception:
        return None
    store = getattr(_cs, "_store", None)
    apply = getattr(store, "apply_synced_memory_canonical", None)
    if not callable(apply):
        return None

    def _sink(record: SyncRecord) -> bool:
        try:
            return bool(
                apply(
                    record.record_id,
                    record.payload,
                    deleted=record.deleted,
                    updated_at=record.updated_at,
                )
            )
        except Exception:
            logger.debug(
                "memory-canonical sink raised for %s",
                getattr(record, "record_id", "?"),
                exc_info=True,
            )
            return False

    return _sink


def _default_update_watermark_gate() -> Any | None:
    """The process-default checkpoint-watermark reader, or ``None``.

    S264 (the spec's section 3 republish contract): the phone-class serve
    filter serves a note's update tail only from the checkpoint watermark
    forward. The reader is the ALREADY-initialised update-store singleton's
    ``get_checkpoint_watermark`` (0 when unset -- no checkpoint means the
    whole tail IS the history); never constructed here, the
    ``_default_note_gate`` idiom. ``None`` (the notes package absent, the
    singleton not initialised) means no local checkpoint state exists at all
    and the filter treats the watermark as 0; a reader that RAISES at serve
    time is indeterminable and the filter drops fail-secure.
    """
    try:
        from opti_oignon.notes import note_updates_store as _nus
    except Exception:
        return None
    store = getattr(_nus, "_store", None)
    if store is None:
        return None
    reader = getattr(store, "get_checkpoint_watermark", None)
    return reader if callable(reader) else None


class SyncEngine:
    """Runs sync rounds against injected peers and manages the paired set.

    The change feed and the peer store are injectable for tests; with neither
    given they resolve to the process singletons. The device id labels this
    device's outbound requests.
    """

    def __init__(
        self,
        *,
        device: str = "local",
        feed: ChangeFeed | None = None,
        store: PeerStore | None = None,
        signer: Any | None = None,
        ledger: DeferredLedger | None = None,
        note_gate: Any | None = None,
        update_sink: Any | None = None,
        conversation_sink: Any | None = None,
        note_sink: Any | None = None,
        memory_canonical_sink: Any | None = None,
        update_watermark_gate: Any | None = None,
    ) -> None:
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string")
        self._device = device
        self._feed = feed
        self._store = store
        # S205 (VL-01): the injectable record signer (the RecordSigner
        # protocol). None resolves to the process default lazily; tests
        # inject a deterministic fake (liboqs is absent in the container).
        self._signer = signer
        # S207 (SYN-05): the injectable deferred ledger. None resolves to the
        # process default lazily, like the feed and the store.
        self._ledger = ledger
        # N.9 (S256): the injectable note gate -- a callable
        # ``note_id -> bool`` the phone-class serve filter consults LIVE
        # (filter-at-serve, decision N9-D1). None resolves lazily to the
        # already-initialised notes store singleton, or to no gate at all,
        # which is fail-secure at the filter (decision N9-D2: excluded).
        self._note_gate = note_gate
        # S264 (NOTES_CRDT_SPEC.md section 3): the injectable note_update
        # apply sink (``record -> bool``) and the injectable checkpoint
        # watermark reader (``note_id -> int``). None resolves lazily to the
        # already-initialised update-store singleton (the note_gate idiom);
        # an unresolvable sink REFUSES the landing fail-secure, an
        # unresolvable watermark reader means no checkpoint state (0), and a
        # reader that raises at serve time drops fail-secure.
        self._update_sink = update_sink
        # S199 (SYN-01): the injectable CONVERSATION apply sink
        # (``record -> bool``). None resolves lazily to the already-loaded
        # conversation manager (the note_gate/update_sink idiom); an
        # unresolvable sink DROPS the record fail-secure at the landing seam.
        self._conversation_sink = conversation_sink
        # S199 (SYN-01): the injectable NOTE apply sink. None resolves lazily
        # to the already-initialised notes store singleton (the note_gate
        # idiom); an unresolvable sink DROPS the record fail-secure.
        self._note_sink = note_sink
        # SYN-01 (Direction D): the injectable MEMORY_CANONICAL apply sink. None
        # resolves lazily to the already-initialised canonical-memory store
        # singleton (the note_sink idiom); an unresolvable sink DROPS the record
        # fail-secure at the landing seam.
        self._memory_canonical_sink = memory_canonical_sink
        self._update_watermark_gate = update_watermark_gate
        # One warning per engine when signing degrades, not one per publish.
        self._warned_unsigned = False

    @property
    def device(self) -> str:
        return self._device

    def _resolve_feed(self) -> ChangeFeed:
        return self._feed if self._feed is not None else get_change_feed()

    def _resolve_store(self) -> PeerStore:
        return self._store if self._store is not None else get_peer_store()

    def _resolve_note_gate(self) -> Any | None:
        # N.9 (S256): the injected gate wins; otherwise the lazy process
        # default (the initialised notes store singleton, or no gate at all
        # -- fail-secure at the filter).
        if self._note_gate is not None:
            return self._note_gate
        return _default_note_gate()

    def _resolve_update_sink(self) -> Any | None:
        # S264: the injected sink wins; otherwise the lazy process default
        # (the initialised update-store singleton, or no sink at all --
        # fail-secure at the landing seam: the record refuses).
        if self._update_sink is not None:
            return self._update_sink
        return _default_update_sink()

    def _resolve_conversation_sink(self) -> Any | None:
        # S199: the injected sink wins; otherwise the lazy process default
        # (the already-loaded conversation manager, or no sink at all --
        # fail-secure at the landing seam: the record drops).
        if self._conversation_sink is not None:
            return self._conversation_sink
        return _default_conversation_sink()

    def _resolve_note_sink(self) -> Any | None:
        # S199: the injected sink wins; otherwise the lazy process default
        # (the already-initialised notes store singleton, or no sink at all --
        # fail-secure at the landing seam: the record drops).
        if self._note_sink is not None:
            return self._note_sink
        return _default_note_sink()

    def _resolve_memory_canonical_sink(self) -> Any | None:
        # SYN-01 (Direction D): the injected sink wins; otherwise the lazy
        # process default (the already-initialised canonical-memory store
        # singleton, or no sink at all -- fail-secure at the landing seam: the
        # record drops).
        if self._memory_canonical_sink is not None:
            return self._memory_canonical_sink
        return _default_memory_canonical_sink()

    def _resolve_update_watermark_gate(self) -> Any | None:
        # S264: the injected reader wins; otherwise the lazy process default
        # (the initialised update-store singleton's checkpoint reader, or
        # None -- no checkpoint state exists, the watermark is 0).
        if self._update_watermark_gate is not None:
            return self._update_watermark_gate
        return _default_update_watermark_gate()

    def _resolve_ledger(self) -> DeferredLedger:
        if self._ledger is not None:
            return self._ledger
        # With an injected store carrying an explicit root (tests, custom
        # deployments), the ledger co-locates with it: the quarantine lives
        # where the trust registry lives, and isolation follows the store's
        # by construction (no cross-test leakage through the process
        # singleton). getattr-defensive (the since_page forward-compat
        # precedent); a store without a root falls through to the process
        # default, which resolves under the data directory like the store.
        root = getattr(self._store, "_root", None) if self._store is not None else None
        if root is not None:
            self._ledger = DeferredLedger(root=root)
            return self._ledger
        return get_deferred_ledger()

    def _resolve_signer(self) -> Any:
        if self._signer is not None:
            return self._signer
        from opti_oignon.veilid.signing import get_record_signer

        return get_record_signer()

    # Local production (local-disk; not gated, not audited -- it is a local edit)

    def publish(self, record: SyncRecord) -> int:
        """Journal a local change so a peer can pull it; returns its sequence.

        Local-disk only, like any local edit, so it is permitted in any mode and
        is not gated by Bulbe; only moving the change over the wire is Daily-only.

        S205 (VL-01, sign-at-publish): a record produced by THIS device is
        signed here, once per local edit, over its canonical bytes -- so the
        journal holds the signature, ``since_page``'s wire budget bounds the
        signed size automatically, and the serve path never re-signs. A record
        originating elsewhere (an applied winner re-journalled by
        ``apply_record_batch``, a test publishing foreign provenance) is
        journalled verbatim: its signature is the ORIGINATOR's business and is
        preserved end to end. When signing is unavailable (no PQC backend, no
        master key), the publish degrades to unsigned with one warning per
        engine -- the honest pre-VL-01 posture -- rather than blocking the
        local edit; key custody and signing stay mode-free local CPU/disk.
        """
        if (
            isinstance(record, SyncRecord)
            and record.device == self._device
            and not record.signature
        ):
            try:
                from opti_oignon.veilid.signing import attach_signature

                record = attach_signature(record, self._resolve_signer())
            except Exception as exc:
                if not self._warned_unsigned:
                    self._warned_unsigned = True
                    logger.warning(
                        "sync: publishing UNSIGNED records (pre-VL-01 "
                        "posture): %s",
                        exc,
                    )
        return self._resolve_feed().record(record)

    def local_records(self) -> list[SyncRecord]:
        """The current latest-per-key snapshot of this device's records."""
        return self._resolve_feed().current_records()

    def current_clock(self, kind: RecordKind | str, record_id: str) -> int:
        """The highest clock this device has journalled for a key (0 if unseen).

        S199 (SYN-01, clock discipline): delegates to the resolved change feed,
        so a domain hook computing ``current_clock(kind, key) + 1`` reads the
        same journal the engine's ``publish_*`` writes -- including an injected
        test feed. Local-disk read; permitted in any mode, not gated.
        """
        return self._resolve_feed().current_clock(kind, record_id)

    def publish_conversation(
        self,
        conversation_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        clock: int,
        deleted: bool = False,
        updated_at: str = "",
    ) -> int:
        """Journal a conversation change for peers to pull; returns its sequence.

        Builds the record through the conversation producer and journals it. Local
        disk only, so it is permitted in any mode and is not gated by Bulbe.
        """
        return self.publish(
            conversation_record(
                conversation_id,
                payload,
                device=self._device,
                clock=clock,
                deleted=deleted,
                updated_at=updated_at,
            )
        )

    def publish_note(
        self,
        note_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        clock: int,
        deleted: bool = False,
        updated_at: str = "",
    ) -> int:
        """Journal a note change for peers to pull; returns its sequence.

        The conversation sibling (N.8): builds the record through the note
        producer and journals it. The note body rides as an opaque CRDT blob in
        the payload, which this engine never interprets. Local disk only, so it is
        permitted in any mode and is not gated by Bulbe; a received note applies
        without the human gate (NOTE is not in ``SENSITIVE_KINDS``), unlike a
        skill.
        """
        return self.publish(
            note_record(
                note_id,
                payload,
                device=self._device,
                clock=clock,
                deleted=deleted,
                updated_at=updated_at,
            )
        )

    def publish_note_update(
        self,
        note_id: str,
        seq: int,
        payload: Mapping[str, Any] | None = None,
        *,
        clock: int,
        updated_at: str = "",
    ) -> int:
        """Journal one opaque note update for peers to pull; returns its sequence.

        The note sibling on the S256 seam (NOTES_CRDT_SPEC.md section 3):
        builds the record through the note_update producer (identity
        ``note_id:seq``, the author's append order preserved) and journals
        it. The update blob rides as opaque payload content this engine never
        interprets. Local disk only, permitted in any mode, not gated by
        Bulbe; a received update applies without the human gate (NOTE_UPDATE
        is not in ``SENSITIVE_KINDS``) but only through the store landing of
        the apply seam, section-5 fail-secure. Structurally no tombstone: the
        producer has no ``deleted`` parameter -- updates leave the world only
        by the local pruning rules of section 4.
        """
        return self.publish(
            note_update_record(
                note_id,
                seq,
                payload,
                device=self._device,
                clock=clock,
                updated_at=updated_at,
            )
        )

    def publish_memory_canonical(
        self,
        fact_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        clock: int,
        deleted: bool = False,
        updated_at: str = "",
    ) -> int:
        """Journal a canonical memory fact for peers to pull; returns its sequence."""
        return self.publish(
            memory_canonical_record(
                fact_id,
                payload,
                device=self._device,
                clock=clock,
                deleted=deleted,
                updated_at=updated_at,
            )
        )

    def publish_memory_archive(
        self,
        entry_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        clock: int,
        deleted: bool = False,
        updated_at: str = "",
    ) -> int:
        """Journal an archive memory entry for peers to pull; returns its sequence."""
        return self.publish(
            memory_archive_record(
                entry_id,
                payload,
                device=self._device,
                clock=clock,
                deleted=deleted,
                updated_at=updated_at,
            )
        )

    def publish_skill(
        self,
        skill_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        clock: int,
        deleted: bool = False,
        updated_at: str = "",
    ) -> int:
        """Journal a skill change for peers to pull; returns its sequence.

        Publishing a skill locally is not gated; applying one received over sync is
        the sensitive action, gated by ``run_round``.
        """
        return self.publish(
            skill_record(
                skill_id,
                payload,
                device=self._device,
                clock=clock,
                deleted=deleted,
                updated_at=updated_at,
            )
        )

    def republish_signed(self) -> int:
        """Re-journal this device's unsigned records WITH signatures (S205).

        The one-time migration the VL-01 rollout prefers over a permanent
        accept-unsigned mode: every record in the current latest-per-key
        snapshot that THIS device originated and that carries no signature is
        re-journalled at the SAME clock with a signature attached --
        tombstones included. The collapse-to-latest semantics make this
        transparent: the re-published row becomes the key's MAX(seq) row at
        an unchanged clock and hash, so ``current_clock`` and the LWW merge
        see nothing new while every peer's next pull receives the signed
        version. Records originated by OTHER devices stay untouched (their
        signatures are their origins' to mint; they converge when those
        origins republish and the rows propagate). Returns how many records
        were re-published. Raises :class:`signing.SigningUnavailable` (or the
        backend's error) when this device cannot sign -- the caller asked for
        a signed set and must know. Local-disk/local-CPU, permitted in any
        mode, never gated.
        """
        from opti_oignon.veilid.signing import attach_signature

        signer = self._resolve_signer()
        feed = self._resolve_feed()
        republished = 0
        for rec in feed.current_records():
            if rec.device != self._device or rec.signature:
                continue
            feed.record(attach_signature(rec, signer))
            republished += 1
        if republished:
            logger.info(
                "sync: republished %d record(s) with signatures (VL-01 "
                "migration)",
                republished,
            )
            _audit("republish_signed", count=republished)
        return republished

    # Peer management (audited; the store stays a pure data layer)

    def register_peer(
        self,
        peer_id: str,
        routing_key: str,
        *,
        label: str = "",
        signing_pub: str | None = None,
        pending: bool = False,
    ) -> PeerRecord:
        """Pair a peer (or refresh its route); audited. Not gated by Bulbe.

        Managing the paired set is a local operation, permitted in any mode; what
        is Daily-only is running a round against a peer. The pairing key exchange
        ceremony is S182; this stores the result. S205 (VL-01): a pairing that
        carries the peer's signing PUBLIC key registers it (the store refreshes
        it with the route, preserves it when absent, and warns on a change);
        the key is what record verification resolves against. S206 (PAIR-02):
        ``pending=True`` registers the peer awaiting the mutual confirmation
        (the ceremony's path; the entry gates nothing until confirmed); on a
        re-pair the store can only RAISE the pending state -- a key change
        demotes -- never lower it; :meth:`confirm_peer` activates.
        """
        rec = self._resolve_store().add_peer(
            peer_id,
            routing_key,
            label=label,
            signing_pub=signing_pub,
            pending=pending,
        )
        _audit(
            "peer_add",
            peer_id=peer_id,
            label=label,
            signing_key=bool(signing_pub),
            pending=bool(getattr(rec, "pending", False)),
        )
        return rec

    def confirm_peer(self, peer_id: str) -> bool:
        """Activate a pending peer (PAIR-02); audited. Not gated by Bulbe.

        The human compared the mutual confirmation code on both devices and
        confirmed this side. The store's confirm is the only path that lowers
        the pending state. Returns ``True`` when the peer exists (idempotent
        on an already-confirmed one), ``False`` for an unknown peer; only an
        actual activation is audited.
        """
        store = self._resolve_store()
        before = store.get_peer(peer_id)
        ok = store.confirm_peer(peer_id)
        if ok and before is not None and getattr(before, "pending", False):
            _audit("pairing_confirm", peer_id=peer_id)
        return ok

    def set_device_class(
        self, peer_id: str, device_class: str | None
    ) -> bool:
        """Mark or clear a paired peer's device class; audited. Not gated by Bulbe.

        N.9 / PAIR-03 (S258): the engine-level mirror of the peer store's
        setter, what the pairing accept seam and the control surface call so
        a class flip lands in the hash-chain audit log like every other
        trust-state change. The store validates against ``DEVICE_CLASSES``
        and raises ``ValueError`` on free text (a programming error;
        callers normalise first); ``None`` clears back to the grandfathered
        NULL. Returns ``True`` when a row was updated, ``False`` for an
        unknown peer; only an EFFECTIVE change is audited (the
        ``confirm_peer`` recipe), so an idempotent re-write is silent.
        """
        store = self._resolve_store()
        before = store.get_peer(peer_id)
        ok = store.set_device_class(peer_id, device_class)
        if (
            ok
            and before is not None
            and getattr(before, "device_class", None) != device_class
        ):
            _audit(
                "peer_device_class",
                peer_id=peer_id,
                device_class=device_class if device_class is not None else "",
            )
        return ok

    def self_signing_pub(self) -> str | None:
        """This device's signing PUBLIC key, base64url, or ``None``.

        What the pairing payload carries (S205) and half of what the PAIR-02
        confirmation code covers. Defensive: a missing backend, a refused mint
        (no master key), or any custody failure degrades to ``None`` -- the
        device pairs as an honest pre-VL-01 peer -- never an exception into
        the pairing surface.
        """
        try:
            signer = self._resolve_signer()
            probe = getattr(signer, "verify_available", None)
            if callable(probe) and not probe():
                return None
            from opti_oignon.veilid.signing import encode_public_key

            raw = signer.public_key()
            return encode_public_key(raw) if raw else None
        except Exception:
            return None

    def unregister_peer(self, peer_id: str) -> bool:
        """Unpair a peer; audited. Returns True when a peer was removed.

        S207 (SYN-05): unpairing cascades the peer's deferred-ledger entries
        away, fail-secure -- a quarantined record from a removed peer must not
        remain one click from application. (A record whose ORIGIN device was
        unpaired but that was deferred from a still-paired serving peer is
        covered instead by approval re-verification refusing against the
        missing key.) The cascade is best-effort and audited; a ledger failure
        never blocks the unpair itself.
        """
        removed = self._resolve_store().remove_peer(peer_id)
        if removed:
            _audit("peer_remove", peer_id=peer_id)
            try:
                purged = self._resolve_ledger().remove_for_peer(peer_id)
            except Exception:  # pragma: no cover - cascade is best-effort
                logger.warning(
                    "deferred-ledger cascade failed for unpaired peer %s",
                    peer_id,
                    exc_info=True,
                )
                purged = 0
            if purged:
                _audit(
                    "sync_deferred_unpair_cascade",
                    peer_id=peer_id,
                    removed=int(purged),
                )
        return removed

    def list_peers(self) -> list[PeerRecord]:
        """All paired peers (a read; not audited)."""
        return self._resolve_store().list_peers()

    def peer_watermark(self, peer_id: str) -> int:
        """A peer's current watermark, or 0 when it is not paired."""
        return self._resolve_store().get_watermark(peer_id)

    # The responder (gated; refuses under Bulbe at the seam; audited)

    def serve_request(self, raw_request: Any, *, peer_id: str = "") -> dict[str, Any]:
        """Answer an inbound delta request from a peer with a batch from the feed.

        The serve half of the exchange, so a paired device both pulls and serves.
        Refuses under Bulbe at the binding-layer gate, re-asserted here at the
        engine seam (the protocol's ``respond_to_request`` gates too, so the refusal
        is enforced, not a handler policy). An unparseable request yields a benign
        empty batch -- high-water 0 and no records, so it can never advance the
        asker (PRT-01) -- via the protocol's defensive responder, never an
        over-send or a crash. PAIR-02 (S206): when the caller supplies the
        asking peer's identity and that peer's registry entry is PENDING, the
        request is refused (:class:`PeerNotConfirmed`) -- an unconfirmed
        pairing gates nothing, serving included. Stated honestly: with an
        EMPTY ``peer_id`` (today's production posture; the inbound identity is
        not authenticated at this seam, the private route is the implicit
        authenticator) there is nothing to check, so the gate acts only where
        an identity is actually supplied. The served answer is
        recorded in the hash-chain audit log. The batch is a JSON-safe wire dict.

        N.9 (S256), filter-at-serve (decision N9-D1): toward a PHONE-CLASS
        asker, NOTE records are dropped from the served batch unless the
        note's per-item mobile-allowed flag affirmatively permits them, via
        a LIVE lookup through the note gate (a journal-time snapshot would
        go stale when the user flips the flag). The journal itself stays
        whole, so every desktop peer keeps the full N.8 note sync. The class
        is keyed on the supplied identity's registry row; fail-secure
        throughout (decision N9-D2): a SUPPLIED identity with NO registry
        row is treated phone-class (an unidentifiable asker is never owed a
        note -- deliberately stricter than the PAIR-02 gate above, which
        refuses serving outright only on a known-pending row, because the
        filter costs one sensitive kind, not the fleet), and an absent or
        unreadable flag means NOT allowed. With an empty ``peer_id`` the
        filter, exactly like the PAIR-02 gate, has nothing to key on; the
        identity binding at this seam is the mobile cycle's host-assured
        half.
        """
        assert_sync_allowed()
        rec: Any | None = None
        if peer_id:
            store = self._resolve_store()
            rec = (
                store.get_peer(peer_id) if hasattr(store, "get_peer") else None
            )
            if rec is not None and getattr(rec, "pending", False):
                raise PeerNotConfirmed(peer_id)
        feed = self._resolve_feed()
        batch = respond_to_request(feed, raw_request, device=self._device)
        notes_filtered = 0
        if peer_id:
            phone_class = rec is None or (
                getattr(rec, "device_class", None) == DEVICE_CLASS_PHONE
            )
            if phone_class:
                batch, notes_filtered = self._filter_notes_for_phone(batch)
        _audit(
            "sync_serve",
            peer_id=peer_id,
            records=len(batch.get("records", []) or []),
            high_water=int(batch.get("high_water", 0)),
            notes_filtered=notes_filtered,
        )
        return batch

    def _filter_notes_for_phone(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Drop NOTE and NOTE_UPDATE records a phone-class peer is not opted into.

        The seam decision (N9-D1) made concrete: the per-record verdict is a
        LIVE lookup through the note gate, and fail-secure throughout
        (N9-D2) -- no resolvable gate, a malformed wire object, a missing or
        non-string id, an unknown note, a raised error, anything not
        affirmatively true EXCLUDES the record. S264 extends the same floor
        to the update kind, keyed on the parent ``note_id`` riding the wire
        payload, plus the republish contract's watermark-forward rule (only
        ``seq`` above the checkpoint watermark is served). The batch's
        high-water is
        deliberately untouched: a filtered record is invisible to the phone,
        not pending for it, so the asker's watermark advances past it; a
        later opt-in reaches the phone through a fresh journal entry (the
        republish contract, NOTES_MOBILE_SYNC_N9_S256.md).
        """
        records = batch.get("records")
        if not isinstance(records, list) or not records:
            return batch, 0
        gate = self._resolve_note_gate()
        wm_gate_resolved = False
        wm_gate: Any | None = None
        kept: list[Any] = []
        dropped = 0
        for wire in records:
            kind = wire.get("kind") if isinstance(wire, dict) else None
            if kind == RecordKind.NOTE_UPDATE.value:
                # S264 (NOTES_CRDT_SPEC.md section 3): the S258 device-class
                # gate is a FLOOR for updates too. The parent note's flag is
                # consulted through the SAME live gate (anything not
                # affirmatively true excludes, N9-D2), keyed on the wire
                # payload's ``note_id`` (absent or malformed: excluded). On
                # top of the floor, the republish contract: only the tail
                # ABOVE the checkpoint watermark is served (``seq > w``); a
                # missing or non-integer seq excludes, an unresolvable
                # watermark reader means no checkpoint state (w = 0, the
                # whole tail IS the history), and a reader that raises is
                # indeterminable -- excluded fail-secure. The batch's
                # high-water stays untouched, exactly as for notes: a
                # filtered update is invisible to the phone, and a later
                # opt-in reaches it through the republished checkpoint plus
                # the surviving tail.
                payload = (
                    wire.get("payload") if isinstance(wire, dict) else None
                )
                note_id = (
                    payload.get("note_id")
                    if isinstance(payload, Mapping)
                    else None
                )
                seq = (
                    payload.get("seq") if isinstance(payload, Mapping) else None
                )
                allowed = False
                if gate is not None and isinstance(note_id, str) and note_id:
                    try:
                        allowed = bool(gate(note_id))
                    except Exception:
                        allowed = False
                if allowed and (
                    isinstance(seq, bool) or not isinstance(seq, int)
                ):
                    allowed = False
                if allowed:
                    if not wm_gate_resolved:
                        wm_gate_resolved = True
                        wm_gate = self._resolve_update_watermark_gate()
                    watermark = 0
                    if wm_gate is not None:
                        try:
                            watermark = int(wm_gate(note_id))
                        except Exception:
                            allowed = False
                    if allowed:
                        allowed = seq > watermark
                if allowed:
                    kept.append(wire)
                else:
                    dropped += 1
                continue
            if kind != RecordKind.NOTE.value:
                kept.append(wire)
                continue
            note_id = wire.get("id")
            allowed = False
            if gate is not None and isinstance(note_id, str) and note_id:
                try:
                    allowed = bool(gate(note_id))
                except Exception:
                    allowed = False
            if allowed:
                kept.append(wire)
            else:
                dropped += 1
        if dropped:
            batch = dict(batch)
            batch["records"] = kept
        return batch, dropped

    # The round (gated; refuses under Bulbe at the seam)

    def _verify_records(
        self, records_in: list[SyncRecord]
    ) -> tuple[list[SyncRecord], int, int]:
        """Partition incoming records by signature verification (S205, VL-01).

        Returns ``(appliable, refused, unverified)``. The key a record is
        checked against is resolved by its ORIGIN (``record.device``): this
        device's own public key for our own records coming back (a resync or
        backstop), else the signing key registered for that device in the
        peer store -- NOT the serving peer's key, because a relayed record (A
        edited, B serves) carries A's signature. That lookup IS the
        device<->key<->provenance binding: a record signed by any OTHER key,
        a re-attributed record, a tampered record, and an unsigned record
        from a keyed origin all fail it and are REFUSED (counted, never
        applied, never prompted for approval, and never holding the
        watermark -- a forgery must not pin convergence; the persisted
        per-record ledger is SYN-05).

        An origin with NO registered key was the migration case: under the
        bounded grace (``ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS``, open
        S205..S207) its records were accepted-and-counted as ``unverified``;
        the window CLOSED at S208 (the Bloc 4 release flip, 3.7.0) and such a
        record now refuses like the rest -- the constant is read at call time,
        so tests re-open the historical behaviour by monkeypatch. PAIR-02
        (S206): an origin whose registry entry is PENDING refuses outright,
        signature or not -- its registered material awaits the human
        confirmation, so it is neither trusted nor admitted to any grace. A
        device that cannot verify at all (no PQC backend) accepts everything
        as ``unverified`` with a warning -- it is wholly pre-VL-01, and
        refusing what it cannot check would partition the fleet, not protect
        it; that posture is a different branch and is unchanged by the flip.

        Verification is reading, ungated in itself; it runs inside the
        already-gated round. Defensive throughout: a malformed stored key
        degrades to "no key", never raises into the round.
        """
        from opti_oignon.veilid.signing import (
            ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS,
            decode_public_key,
            verify_record_signature,
        )

        signer = self._resolve_signer()
        verify_capable = True
        probe = getattr(signer, "verify_available", None)
        if callable(probe):
            try:
                verify_capable = bool(probe())
            except Exception:
                verify_capable = False
        if not verify_capable:
            if records_in:
                logger.warning(
                    "sync: signature verification unavailable (no PQC "
                    "backend); accepting %d record(s) UNVERIFIED (pre-VL-01 "
                    "posture)",
                    len(records_in),
                )
            return list(records_in), 0, len(records_in)

        store = self._resolve_store()
        own_pub: bytes | None = None
        own_pub_resolved = False
        key_cache: dict[str, Any] = {}
        appliable: list[SyncRecord] = []
        refused = 0
        unverified = 0
        for r in records_in:
            origin = r.device
            if origin == self._device:
                if not own_pub_resolved:
                    own_pub_resolved = True
                    try:
                        own_pub = signer.public_key()
                    except Exception:
                        own_pub = None
                key = own_pub
            elif origin in key_cache:
                key = key_cache[origin]
            else:
                peer = store.get_peer(origin)
                if peer is not None and getattr(peer, "pending", False):
                    # PAIR-02 (S206): a PENDING origin refuses outright,
                    # regardless of signature validity -- its registered
                    # material is not yet human-confirmed, so it must neither
                    # verify as trusted nor degrade into the unsigned grace
                    # path (accepting unverified would be strictly worse than
                    # refusing). Relayed records from a pending origin land
                    # here even though a direct round against it never runs.
                    key = _PENDING_ORIGIN
                else:
                    key = decode_public_key(getattr(peer, "signing_pub", None))
                key_cache[origin] = key
            if key is _PENDING_ORIGIN:
                refused += 1
                continue
            if key is None:
                if ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS:
                    appliable.append(r)
                    unverified += 1
                else:
                    refused += 1
                continue
            if not r.signature or not verify_record_signature(r, key, signer):
                refused += 1
                continue
            appliable.append(r)
        if refused:
            logger.warning(
                "sync: refused %d record(s) at the signature seam "
                "(unsigned or invalid against the origin device's "
                "registered key); refused-and-counted, never applied (VL-01)",
                refused,
            )
        return appliable, refused, unverified

    def _gate_records(
        self,
        records_in: list[SyncRecord],
        *,
        peer_id: str,
        approval_fn: Any,
        conversation_id: str,
        approval_manager: Any,
        ledger_reentry: bool = False,
    ) -> tuple[list[SyncRecord], int]:
        """Partition incoming records into appliable and deferred.

        A non-sensitive record applies. A sensitive record (a skill) passes the
        fail-secure human gate; if not approved it is deferred: counted, not
        applied, and PERSISTED to the deferred ledger (S207, SYN-05) with its
        full wire envelope so the human can decide later from the panel and
        the watermark can advance past it. Returns the appliable records and
        the deferred count (newly quarantined this round: ledger inserts and
        replacements).

        Dedup-and-silence: a sensitive record whose key is already pending in
        the ledger never re-prompts -- it is offered to the ledger (which
        replaces, refreshes, or ignores it by the LWW recipe) and counted as
        deferred only when it replaced the stored version with a newer one.
        Staleness, checked FIRST: a sensitive record whose clock is strictly
        below the LOCAL set's ``current_clock`` for its key would lose the
        merge anyway; it is skipped (audited), neither prompted nor ledgered,
        and the same observation sweeps a dead ledger entry for the key (one
        strictly below the local clock -- the wire just proved it
        superseded). An equal clock is NOT skipped: it may be the concurrent
        divergence the human should see.

        ``ledger_reentry`` is the approval path's flag (S207): a record
        re-entering this gate FROM the ledger must not dedup against its own
        entry or be re-persisted -- the panel approval already answered the
        gate -- so the ledger machinery is skipped and only the approval seam
        runs. Never set on the wire path.
        """
        appliable: list[SyncRecord] = []
        deferred = 0
        ledger: DeferredLedger | None = None
        feed: ChangeFeed | None = None
        for r in records_in:
            if r.kind.value not in SENSITIVE_KINDS:
                appliable.append(r)
                continue
            if not ledger_reentry:
                if ledger is None:
                    ledger = self._resolve_ledger()
                if feed is None:
                    feed = self._resolve_feed()
                try:
                    local_clock = feed.current_clock(r.kind, r.record_id)
                except Exception:  # pragma: no cover - defensive read
                    local_clock = 0
                if r.clock < local_clock:
                    # Strictly below what the local set already holds: the
                    # record would lose reconciliation; ledgering it would
                    # offer the human a dead decision. Skipped, audited,
                    # neither prompted nor counted. The same observation
                    # sweeps a dead ENTRY for the key (one strictly below the
                    # local clock) -- the wire just proved it superseded.
                    _audit(
                        "sync_deferred_stale",
                        peer_id=peer_id,
                        kind=r.kind.value,
                        id=r.record_id,
                        device=r.device,
                        clock=int(r.clock),
                        local_clock=int(local_clock),
                    )
                    try:
                        if ledger.purge_below(
                            r.kind.value, r.record_id, local_clock
                        ):
                            _audit(
                                "sync_deferred_superseded",
                                kind=r.kind.value,
                                id=r.record_id,
                                local_clock=int(local_clock),
                            )
                    except Exception:  # pragma: no cover - sweep best-effort
                        logger.debug(
                            "deferred-ledger sweep failed for %s/%s",
                            r.kind.value,
                            r.record_id,
                            exc_info=True,
                        )
                    continue
                if ledger.has(r.kind.value, r.record_id):
                    # Already pending: dedup-and-silence (no re-prompt is the
                    # SYN-05 fix working). The offer arbitrates by the LWW
                    # recipe: a newer version replaces the entry (a fresh
                    # pending decision, counted), a re-arrival refreshes
                    # last_offered_at, an older candidate is ignored.
                    outcome = ledger.offer(r, peer_id=peer_id)
                    if outcome == OFFER_REPLACED:
                        deferred += 1
                    continue
            gate_args = {
                "peer_id": peer_id,
                "kind": r.kind.value,
                "id": r.record_id,
                "device": r.device,
            }
            if self._approve(
                conversation_id=conversation_id,
                label="sync_apply:" + r.kind.value,
                gate_args=gate_args,
                approval_fn=approval_fn,
                approval_manager=approval_manager,
            ):
                appliable.append(r)
            elif ledger_reentry:
                # Defensive: the reentry approval_fn is always-approve by
                # construction; a denial here applies nothing and persists
                # nothing (the entry's fate is the caller's).
                deferred += 1
            else:
                outcome = ledger.offer(r, peer_id=peer_id)  # type: ignore[union-attr]
                if outcome in (OFFER_INSERTED, OFFER_REPLACED):
                    deferred += 1
                _audit(
                    "sync_deferred",
                    peer_id=peer_id,
                    kind=r.kind.value,
                    id=r.record_id,
                    device=r.device,
                    clock=int(r.clock),
                    outcome=outcome,
                )
        return appliable, deferred

    def _land_conversations(
        self, records_in: list[SyncRecord], *, peer_id: str
    ) -> list[SyncRecord]:
        """Materialise verified CONVERSATION records into the conversation store.

        The full-state analogue of :meth:`_land_note_updates` (S199, SYN-01):
        sits AFTER the signature seam and the gate, BEFORE the feed journal. A
        winning CONVERSATION record is written into the conversation store so it
        surfaces in this device's app, then KEPT so it still enters the feed and
        can be served onward (the relay role). The apply is hook-free
        (``apply_synced_conversation`` never re-publishes), so the kept record
        is journalled verbatim by the apply seam, signature preserved -- the
        same posture as the note_update landing. Fail-secure: an unresolvable
        sink (the conversation module not loaded) or an apply that returns False
        DROPS the record -- not materialised, not journalled, not served (a
        record that cannot be persisted must not converge). Other kinds pass
        through untouched.
        """
        kept: list[SyncRecord] = []
        sink: Any | None = None
        sink_resolved = False
        for r in records_in:
            if r.kind.value != RecordKind.CONVERSATION.value:
                kept.append(r)
                continue
            if not sink_resolved:
                sink_resolved = True
                sink = self._resolve_conversation_sink()
            ok = False
            if sink is not None:
                try:
                    ok = bool(sink(r))
                except Exception:
                    logger.debug(
                        "conversation sink raised for %s",
                        r.record_id,
                        exc_info=True,
                    )
                    ok = False
            if ok:
                kept.append(r)
                continue
            logger.warning(
                "sync: refused a conversation at the landing seam "
                "(id=%s origin=%s): not materialised, not served",
                r.record_id,
                r.device,
            )
            _audit(
                "sync_conversation_refused",
                peer_id=peer_id,
                id=r.record_id,
            )
        return kept

    def _land_notes(
        self, records_in: list[SyncRecord], *, peer_id: str
    ) -> list[SyncRecord]:
        """Materialise verified NOTE records into the notes store.

        The full-state analogue of :meth:`_land_conversations` for NOTE
        (S199, SYN-01): existence + metadata land into the notes store (the
        text body converges separately through NOTE_UPDATE, so a stale snapshot
        never regresses it, and ``mobile_allowed`` is never written from the
        wire). The record is KEPT for the feed/relay; a refused or unappliable
        record is DROPPED (fail-secure: a record that cannot be persisted must
        not converge). Other kinds pass through untouched. NOTE and NOTE_UPDATE
        land on independent stores, so their relative order in the batch does
        not affect convergence.
        """
        kept: list[SyncRecord] = []
        sink: Any | None = None
        sink_resolved = False
        for r in records_in:
            if r.kind.value != RecordKind.NOTE.value:
                kept.append(r)
                continue
            if not sink_resolved:
                sink_resolved = True
                sink = self._resolve_note_sink()
            ok = False
            if sink is not None:
                try:
                    ok = bool(sink(r))
                except Exception:
                    logger.debug(
                        "note sink raised for %s",
                        r.record_id,
                        exc_info=True,
                    )
                    ok = False
            if ok:
                kept.append(r)
                continue
            logger.warning(
                "sync: refused a note at the landing seam "
                "(id=%s origin=%s): not materialised, not served",
                r.record_id,
                r.device,
            )
            _audit(
                "sync_note_refused",
                peer_id=peer_id,
                id=r.record_id,
            )
        return kept

    def _land_memory_canonical(
        self, records_in: list[SyncRecord], *, peer_id: str
    ) -> list[SyncRecord]:
        """Materialise verified MEMORY_CANONICAL records into the canonical store.

        The full-state analogue of :meth:`_land_conversations` for canonical
        memory (SYN-01, Direction D): a winning fact lands into the canonical
        store (preserving the device-local ``use_count``; a tombstone is a hard
        delete). The record is KEPT for the feed/relay; a refused or unappliable
        record is DROPPED (fail-secure: a record that cannot be persisted must
        not converge). Other kinds pass through untouched. Canonical memory is
        user data -- it lands ungated, on the same seam as conversation/note.
        """
        kept: list[SyncRecord] = []
        sink: Any | None = None
        sink_resolved = False
        for r in records_in:
            if r.kind.value != RecordKind.MEMORY_CANONICAL.value:
                kept.append(r)
                continue
            if not sink_resolved:
                sink_resolved = True
                sink = self._resolve_memory_canonical_sink()
            ok = False
            if sink is not None:
                try:
                    ok = bool(sink(r))
                except Exception:
                    logger.debug(
                        "memory-canonical sink raised for %s",
                        r.record_id,
                        exc_info=True,
                    )
                    ok = False
            if ok:
                kept.append(r)
                continue
            logger.warning(
                "sync: refused a canonical memory fact at the landing seam "
                "(id=%s origin=%s): not materialised, not served",
                r.record_id,
                r.device,
            )
            _audit(
                "sync_memory_canonical_refused",
                peer_id=peer_id,
                id=r.record_id,
            )
        return kept

    def _land_note_updates(
        self, records_in: list[SyncRecord], *, peer_id: str
    ) -> list[SyncRecord]:
        """Land verified NOTE_UPDATE records through the store's append seam.

        S264 (NOTES_CRDT_SPEC.md sections 3 and 5): sits AFTER the signature
        seam and the gate, BEFORE the feed journal, so a refused update is
        not appended, not served onward (only the feed is served), and not
        rendered -- refused means refused everywhere, loggable, never silent.
        The landing preserves the AUTHOR's explicit ``seq`` (the wire
        coordinates, never a local re-mint) and suppresses the store's own
        publish glue: the kept record is journalled verbatim by the apply
        seam, signature preserved, and re-publishing would re-sign it.
        Fail-secure: an unresolvable sink (the notes package absent, the
        store singleton not initialised) refuses every update -- a record
        that cannot be persisted must not converge. A refusal is logged at
        warning and audited (``sync_note_update_refused``); a benign
        re-delivery refuses at the store's duplicate seam without harm (the
        record already converged through the feed on first arrival). Other
        kinds pass through untouched.
        """
        kept: list[SyncRecord] = []
        sink: Any | None = None
        sink_resolved = False
        for r in records_in:
            if r.kind.value != RecordKind.NOTE_UPDATE.value:
                kept.append(r)
                continue
            if not sink_resolved:
                sink_resolved = True
                sink = self._resolve_update_sink()
            ok = False
            if sink is not None:
                try:
                    ok = bool(sink(r))
                except Exception:
                    logger.debug(
                        "note_update sink raised for %s",
                        r.record_id,
                        exc_info=True,
                    )
                    ok = False
            if ok:
                kept.append(r)
                continue
            logger.warning(
                "sync: refused a note_update at the landing seam "
                "(id=%s origin=%s): not appended, not served, not "
                "rendered (NOTES_CRDT_SPEC.md section 5)",
                r.record_id,
                r.device,
            )
            _audit(
                "sync_note_update_refused",
                peer_id=peer_id,
                id=r.record_id,
                device=r.device,
                clock=int(r.clock),
            )
        return kept

    @staticmethod
    def _approve(
        *,
        conversation_id: str,
        label: str,
        gate_args: dict[str, Any],
        approval_fn: Any,
        approval_manager: Any,
    ) -> bool:
        """Consult the human approval gate, fail-secure.

        Mirrors the agent's manage_skills gate: an injected ``approval_fn`` is
        used if given, else the default manager-backed ``allowlists.request_approval``;
        any error or a missing gate denies (returns False).
        """
        if approval_fn is not None:
            try:
                return bool(approval_fn(conversation_id, label, dict(gate_args)))
            except Exception:
                return False
        try:
            from opti_oignon.agent.allowlists import request_approval

            return request_approval(
                conversation_id, label, dict(gate_args), manager=approval_manager
            )
        except Exception:  # pragma: no cover - fail-secure
            return False

    def run_round(
        self,
        peer_id: str,
        peer: Peer,
        *,
        approval_fn: Any = None,
        conversation_id: str = "",
        approval_manager: Any = None,
    ) -> RoundResult:
        """Run one pull round against a paired peer and apply the answer.

        Refuses under Bulbe at the binding-layer gate. Raises :class:`PeerNotFound`
        when the peer is not paired and :class:`PeerNotConfirmed` when its
        pairing awaits the PAIR-02 mutual confirmation (S206) -- an
        unconfirmed entry gates nothing. Resolves the peer's watermark, asks the peer
        for the delta, gates any sensitive records, reconciles the appliable ones
        into the local set, and advances the watermark monotonically past every
        consumed chunk. S207 (SYN-05): a deferral no longer holds the watermark
        -- the deferred record is persisted to the per-record ledger (full wire
        envelope, so no re-fetch is ever needed) and the round keeps consuming;
        a still-pending record arriving again dedups into its entry silently
        (no re-prompt), and the human approves or refuses from the panel, where
        an approval re-enters this same verify -> gate -> apply seam against
        the CURRENT trust state. Records the round in the audit log and returns
        a structured summary.

        S204 (CHF-05): the peer's batches carry its feed epoch. The first epoch a
        peer advertises is stored (no reset); when a later round sees a different
        one (the peer's journal was recreated, sequences restarted), the watermark
        is reset to 0 atomically with the new epoch and the full set resyncs this
        same round, over the normal bounded leg loop -- never the backstop. A
        pre-epoch peer (no epoch on the wire) leaves the stored epoch untouched
        and falls through to the CHF-01 backstop as before.

        S205 (VL-01): each leg's records pass the signature seam (parse ->
        epoch -> VERIFY -> gate -> apply): verification against the ORIGIN
        device's registered signing key; refusals are counted on the result
        (``refused``), never applied, never prompted, and never hold the
        watermark. ``unverified`` counts records accepted without verification
        (the verify-incapable posture; the historical S205..S207 grace, now
        closed). See :meth:`_verify_records`.
        """
        assert_sync_allowed()
        store = self._resolve_store()
        if not store.has_peer(peer_id):
            raise PeerNotFound(peer_id)
        # PAIR-02 (S206): a pending pairing gates nothing. The check is
        # hasattr/getattr-defensive (the since_page forward-compat precedent)
        # so a store predating the pending state simply skips it.
        rec = store.get_peer(peer_id) if hasattr(store, "get_peer") else None
        if rec is not None and getattr(rec, "pending", False):
            raise PeerNotConfirmed(peer_id)
        feed = self._resolve_feed()
        previous = store.get_watermark(peer_id)

        # S204 (CHF-05): the per-peer last-seen feed epoch, read once and cached
        # locally; writes only on a transition. hasattr-guarded so a store that
        # predates the epoch accessors simply skips the handling (the since_page
        # forward-compat precedent).
        epoch_capable = (
            hasattr(store, "get_last_epoch")
            and hasattr(store, "set_last_epoch")
            and hasattr(store, "reset_for_epoch")
        )
        known_epoch = store.get_last_epoch(peer_id) if epoch_capable else None
        epoch_reset = False

        # PRT-04: a round consumes one bounded chunk per leg, threading each
        # chunk's high-water into the next request, until the cursor stops
        # advancing (caught up), the answer fails to parse, or MAX_LEGS is
        # reached. The leg-local cursor is seeded from the persisted watermark
        # and advanced by each chunk's high-water; the watermark is persisted
        # once, monotonically, at the last fully-consumed chunk boundary.
        # S207 (SYN-05): a deferral neither ends the loop nor holds the
        # boundary -- the deferred record is in the ledger, so every consumed
        # chunk advances. Pagination rides the existing monotonic advance, no
        # new token.
        cursor = previous
        # The boundary we will persist: advanced past every fully-consumed chunk
        # (S207: deferrals included -- the deferred records live in the ledger).
        committed = previous
        total_applied = 0
        total_conflicts = 0
        total_rejected = 0
        total_deferred = 0
        total_refused = 0
        total_unverified = 0
        legs = 0
        parsed = True

        while legs < MAX_LEGS:
            legs += 1
            request = build_delta_request(device=self._device, watermark=cursor)
            raw_batch = peer.fetch(request)
            batch = parse_record_batch(raw_batch)
            if batch is None:
                # A malformed/garbled answer: degrade and hold at the last
                # consumed chunk boundary (SYN-03). No advance for this leg.
                parsed = False
                break

            # S204 (CHF-05): the epoch check sits after parse and BEFORE the
            # gate, so no approval prompt fires for records about to be
            # discarded. A missing/malformed epoch (None) is a pre-epoch peer:
            # the stored epoch stays untouched and CHF-01 remains the floor.
            if epoch_capable:
                answer_epoch = getattr(batch, "epoch", None)
                if isinstance(answer_epoch, str) and answer_epoch:
                    if known_epoch is None:
                        # First contact with an epoch-aware peer: store it, no
                        # reset (resetting here would force a full resync on
                        # every fleet upgrade), and consume the leg normally.
                        store.set_last_epoch(peer_id, answer_epoch)
                        known_epoch = answer_epoch
                    elif answer_epoch != known_epoch:
                        # The peer's journal was recreated. Reset the watermark
                        # to 0 atomically with the new epoch, DISCARD this
                        # leg's answer -- it was fetched at the stale cursor,
                        # whose coverage in the new epoch cannot be trusted (a
                        # low-but-possible stale watermark silently skips the
                        # new journal's first rows, the divergence CHF-01
                        # cannot see) -- and refetch from 0. The full set then
                        # paginates in over the normal leg loop this same
                        # round, never the backstop (cursor 0 is always
                        # valid). The deferred ledger STANDS across the reset
                        # (S207): record clocks are domain state, not journal
                        # sequences, so the resync re-serves the same versions
                        # and they dedup into their entries silently -- the
                        # pending decision survives, never re-prompted.
                        logger.warning(
                            "sync: peer %s feed epoch changed (%s -> %s); its "
                            "journal was recreated. Resetting the watermark "
                            "to 0 for a single full resync (CHF-05).",
                            peer_id,
                            known_epoch,
                            answer_epoch,
                        )
                        store.reset_for_epoch(peer_id, answer_epoch)
                        known_epoch = answer_epoch
                        cursor = 0
                        committed = 0
                        epoch_reset = True
                        continue

            # S205 (VL-01): the signature seam sits after the epoch check (no
            # work on a leg about to be discarded) and BEFORE the approval
            # gate (no prompt for a record about to be refused -- the CHF-05
            # placement precedent). Refusals do not touch the chunk's
            # high-water: a forged record advances past like a rejected
            # decode, never pinning the watermark.
            verified, refused, unverified = self._verify_records(batch.records)
            total_refused += refused
            total_unverified += unverified

            appliable, deferred = self._gate_records(
                verified,
                peer_id=peer_id,
                approval_fn=approval_fn,
                conversation_id=conversation_id,
                approval_manager=approval_manager,
            )
            # S264: land note_update records through the store BEFORE the
            # feed journal (sections 3 and 5); a refused update never enters
            # the feed and is therefore never served onward.
            appliable = self._land_note_updates(appliable, peer_id=peer_id)
            # S199 (SYN-01): materialise CONVERSATION winners into the domain
            # store on the same seam, BEFORE the feed journal -- a refused
            # conversation never enters the feed and is never served onward.
            appliable = self._land_conversations(appliable, peer_id=peer_id)
            # S199 (SYN-01): materialise NOTE existence + metadata into the
            # notes store on the same seam (the body converges via NOTE_UPDATE).
            appliable = self._land_notes(appliable, peer_id=peer_id)
            # SYN-01 (Direction D): materialise MEMORY_CANONICAL winners into
            # the canonical store on the same post-gate seam, BEFORE the feed
            # journal -- a refused fact never enters the feed and is never
            # served onward. Canonical memory is user data: it lands ungated,
            # alongside conversation/note.
            appliable = self._land_memory_canonical(appliable, peer_id=peer_id)
            filtered = RecordBatch(
                device=batch.device,
                high_water=batch.high_water,
                records=appliable,
                rejected=batch.rejected,
                epoch=getattr(batch, "epoch", None),
            )
            apply_result = apply_record_batch(feed, filtered)
            total_applied += int(getattr(apply_result, "applied", 0))
            total_conflicts += len(getattr(apply_result, "conflicts", []) or [])
            total_rejected += int(getattr(apply_result, "rejected", 0))
            total_deferred += deferred

            # S207 (SYN-05) purge-on-apply, the defensive backstop: a
            # sensitive record landing through the seam supersedes any OLDER
            # version of its key sitting in the ledger. In the single-threaded
            # flow the gate's dedup-first order makes this unreachable (a key
            # with a pending entry never reaches appliable) and the gate's own
            # staleness sweep is the working purge; this guards interleavings
            # the loop cannot see (a concurrent panel approval, a parallel
            # round). Strictly-below only; bounded by the sensitive records
            # per chunk.
            for r in appliable:
                if r.kind.value not in SENSITIVE_KINDS:
                    continue
                try:
                    ledger = self._resolve_ledger()
                    cur_clock = feed.current_clock(r.kind, r.record_id)
                    if ledger.purge_below(r.kind.value, r.record_id, cur_clock):
                        _audit(
                            "sync_deferred_superseded",
                            kind=r.kind.value,
                            id=r.record_id,
                            local_clock=int(cur_clock),
                        )
                except Exception:  # pragma: no cover - purge is best-effort
                    logger.debug(
                        "deferred-ledger purge failed for %s/%s",
                        r.kind.value,
                        r.record_id,
                        exc_info=True,
                    )

            next_cursor = int(batch.high_water)
            if next_cursor == cursor:
                # Caught up: an empty/confirming leg reports high-water equal to
                # the cursor, so the cursor stops advancing and the round ends.
                break
            cursor = next_cursor
            committed = next_cursor

        # Persist once, monotonically. SYN-07: advancing an unpaired (vanished)
        # peer is a no-op returning 0, never reported as an advance. After an
        # epoch reset the persisted watermark moved from the reset baseline (0),
        # not from the pre-round value: a resync can converge below the stale
        # watermark and still be real progress; previous_watermark keeps the
        # true pre-round value for honesty.
        new_watermark = store.advance_watermark(peer_id, committed)
        baseline = 0 if epoch_reset else previous
        advanced = new_watermark > baseline

        result = RoundResult(
            peer_id=peer_id,
            applied=total_applied,
            deferred=total_deferred,
            conflicts=total_conflicts,
            rejected=total_rejected,
            previous_watermark=previous,
            new_watermark=new_watermark,
            advanced=advanced,
            parsed=parsed,
            legs=legs,
            epoch_reset=epoch_reset,
            refused=total_refused,
            unverified=total_unverified,
        )
        _audit(
            "sync_round",
            peer_id=peer_id,
            applied=result.applied,
            deferred=result.deferred,
            conflicts=result.conflicts,
            rejected=result.rejected,
            previous_watermark=result.previous_watermark,
            new_watermark=result.new_watermark,
            advanced=result.advanced,
            parsed=result.parsed,
            legs=result.legs,
            epoch_reset=result.epoch_reset,
            refused=result.refused,
            unverified=result.unverified,
        )
        return result

    # The deferred ledger surface (S207, SYN-05). Local-disk decisions on
    # already-fetched, already-quarantined records: permitted in any mode,
    # like pairing management -- only the wire round that FILLS the ledger is
    # Daily-gated. Fail-secure: an approval re-enters the same verify -> gate
    # -> apply seam against the CURRENT trust state; a refusal applies nothing.

    def list_deferred(self) -> list[DeferredEntry]:
        """Every pending-approval entry, oldest first (a read; not audited)."""
        return self._resolve_ledger().list_entries()

    def approve_deferred(self, kind: str, record_id: str) -> dict[str, Any]:
        """Apply a quarantined record through the full seam; audited.

        The panel approval. Loads the entry, re-decodes its stored envelope,
        and re-enters verify -> gate -> apply with the CURRENT registry state:
        a signing key that changed since deferral, an origin demoted to
        pending (PAIR-02), or a now-closed grace REFUSES here -- the entry is
        removed, audited, and nothing applies (a changed trust root is a new
        trust decision, and a quarantined record under a broken one must not
        linger). The gate leg runs with an always-approve function and the
        ``ledger_reentry`` flag: the panel click IS the human approval, so the
        seam keeps its shape without a second prompt and without the record
        dedup-silencing against its own entry. The apply is a benign
        single-record batch (high_water 0, the PRT-01 idiom): watermarks never
        move here. The entry is removed on every terminal outcome.

        Raises :class:`DeferredNotFound` for an unknown key. Returns a
        structured summary: ``approved`` (the record entered the apply),
        ``refused`` (verification said no), ``applied``/``conflicts``/
        ``rejected`` from the reconciler (an approved older-than-local record
        can still lose LWW and apply 0 -- honest, idempotent), and
        ``unverified`` when the grace carried it.
        """
        ledger = self._resolve_ledger()
        entry = ledger.get(kind, record_id)
        if entry is None:
            raise DeferredNotFound(f"{kind}/{record_id}")
        record = decode_record(entry.envelope)
        if record is None:
            # A corrupt quarantine row cannot re-enter the seam; fail-secure
            # it is removed, never applied.
            ledger.remove(kind, record_id)
            _audit(
                "sync_deferred_approve_refused",
                kind=kind,
                id=record_id,
                peer_id=entry.peer_id,
                device=entry.origin_device,
                reason="undecodable",
            )
            return {
                "kind": kind,
                "record_id": record_id,
                "approved": False,
                "refused": True,
                "reason": "undecodable",
                "applied": 0,
                "conflicts": 0,
                "rejected": 0,
                "unverified": 0,
            }
        verified, refused, unverified = self._verify_records([record])
        if refused or not verified:
            ledger.remove(kind, record_id)
            _audit(
                "sync_deferred_approve_refused",
                kind=kind,
                id=record_id,
                peer_id=entry.peer_id,
                device=entry.origin_device,
                reason="verification",
            )
            return {
                "kind": kind,
                "record_id": record_id,
                "approved": False,
                "refused": True,
                "reason": "verification",
                "applied": 0,
                "conflicts": 0,
                "rejected": 0,
                "unverified": 0,
            }
        appliable, _ = self._gate_records(
            verified,
            peer_id=entry.peer_id,
            approval_fn=lambda _conv, _label, _args: True,
            conversation_id="",
            approval_manager=None,
            ledger_reentry=True,
        )
        batch = RecordBatch(
            device=record.device,
            high_water=0,
            records=appliable,
            rejected=0,
            epoch=None,
        )
        # The ungated local-disk apply (S207): the record is already locally
        # held (fetched through the fully-gated round, quarantined since); the
        # human's approval is a local decision permitted in any mode. The wire
        # apply stays apply_record_batch, gated.
        apply_result = apply_local_batch(self._resolve_feed(), batch)
        ledger.remove(kind, record_id)
        applied = int(getattr(apply_result, "applied", 0))
        conflicts = len(getattr(apply_result, "conflicts", []) or [])
        rejected = int(getattr(apply_result, "rejected", 0))
        _audit(
            "sync_deferred_approved",
            kind=kind,
            id=record_id,
            peer_id=entry.peer_id,
            device=entry.origin_device,
            clock=int(entry.clock),
            applied=applied,
            conflicts=conflicts,
            unverified=int(unverified),
        )
        return {
            "kind": kind,
            "record_id": record_id,
            "approved": True,
            "refused": False,
            "reason": "",
            "applied": applied,
            "conflicts": conflicts,
            "rejected": rejected,
            "unverified": int(unverified),
        }

    def refuse_deferred(self, kind: str, record_id: str) -> DeferredEntry:
        """Refuse a quarantined record: remove its entry; audited; nothing applies.

        Raises :class:`DeferredNotFound` for an unknown key. Returns the
        removed entry (its provenance is the audit's and the route's payload).
        """
        ledger = self._resolve_ledger()
        entry = ledger.get(kind, record_id)
        if entry is None:
            raise DeferredNotFound(f"{kind}/{record_id}")
        ledger.remove(kind, record_id)
        _audit(
            "sync_deferred_refused",
            kind=kind,
            id=record_id,
            peer_id=entry.peer_id,
            device=entry.origin_device,
            clock=int(entry.clock),
        )
        return entry


# Module-level singleton with a reset hook (one engine per process, testable).
# SYN-04: creation is guarded by a lock, the same idiom as the change feed, the
# peer store, and the status store singletons.

_engine: SyncEngine | None = None
_engine_lock = threading.Lock()


def _resolve_default_device(store: PeerStore | None) -> str:
    """The persistent per-install device identity, fail-safe to "local".

    SYN-02: a production engine must not label every device "local" -- the
    pairing payload's peer_id and every record's provenance come from the
    engine's device. The identity is minted and persisted by the peer store
    (``local_device_id``); any failure to resolve it falls back to the
    historical "local" rather than blocking the engine.
    """
    try:
        target = store if store is not None else get_peer_store()
        resolved = target.local_device_id()
        if isinstance(resolved, str) and resolved:
            return resolved
    except Exception:  # pragma: no cover - identity resolution is defensive
        logger.debug("device identity resolution failed; using 'local'", exc_info=True)
    return "local"


def get_sync_engine(
    *,
    device: str | None = None,
    feed: ChangeFeed | None = None,
    store: PeerStore | None = None,
    signer: Any | None = None,
    ledger: DeferredLedger | None = None,
) -> SyncEngine:
    """Return the process sync engine, creating it once (with the args if given).

    With no explicit ``device`` the engine takes this installation's persistent
    device identity from the peer store (SYN-02), falling back to "local" only
    when the identity cannot be resolved. The ``signer`` (S205) and the
    ``ledger`` (S207) are injectable like the feed and store; None resolves to
    the process default lazily.
    """
    global _engine
    with _engine_lock:
        if _engine is None:
            resolved = device if device is not None else _resolve_default_device(store)
            _engine = SyncEngine(
                device=resolved, feed=feed, store=store, signer=signer, ledger=ledger
            )
        return _engine


def set_sync_engine(engine: SyncEngine | None) -> None:
    """Install a specific engine as the process singleton (used by tests)."""
    global _engine
    with _engine_lock:
        _engine = engine


def reset_sync_engine() -> None:
    """Clear the process singleton so the next get creates a fresh one."""
    global _engine
    with _engine_lock:
        _engine = None
