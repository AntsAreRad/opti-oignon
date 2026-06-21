#!/usr/bin/env python3
"""Transport-agnostic protocol envelope for Veilid sync (S179 Goal 4, Theme 4).

The protocol over the encoding, the reconciliation, and the change feed, with no
live transport. A sync exchange is a pull: one device asks a peer "what have you
changed since this watermark?", the peer answers with a batch of record versions
and its high-water mark, and the asking device reconciles the batch into its own
set and advances its watermark. Three operations make that exchange, plus two
defensive parsers and a couple of compositions:

- ``build_delta_request`` -- the asking device's outbound request (a watermark).
- ``build_record_batch`` / ``respond_to_request`` -- the peer's outbound answer,
  drawn from its change feed since the requested watermark.
- ``apply_record_batch`` -- the asking device reconciles the answer against its set
  and journals what changed, returning the converged set, the conflict log, the
  new watermark, and how much was applied or rejected.

There is no socket here. The peer is injected, so a full round is exercised with a
fake peer that answers from its own local feed; S180-S181 supply the live route and
its status surface, and S182 the pairing and sharing-control panel. The envelope is
the seam between the pure data layer and the eventual transport.

The Bulbe boundary lives here, at that seam. Every function that would act on
records over the wire -- building a request or a batch, responding, applying an
incoming batch, or running a round -- calls ``guard.assert_sync_allowed`` before it
acts, so sync refuses under Bulbe at the binding layer, the same discipline as the
agent's memory and skills writes, not a policy flag. The pure helpers stay
ungated: parsing an incoming message is reading data, not acting over the wire, and
reconciliation is side-effect-free, so both run in any mode. Incoming peer messages
are treated as data: the parsers never raise, an unparseable message is rejected,
and the record decoder's integrity check still applies before reconciliation.

This module imports only the gate, the encoding, and the reconciler; it operates on
a duck-typed feed, so it is store-agnostic as well as transport-agnostic, and it
collects without the backend.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from opti_oignon.veilid.guard import assert_sync_allowed
from opti_oignon.veilid.reconcile import reconcile
from opti_oignon.veilid.records import (
    SyncRecord,
    decode_records,
    encode_records,
    key_of,
)

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# The protocol wire version and message types. A message that does not carry
# exactly this version and a known type is rejected on parse.
PROTOCOL_VERSION = 1
MSG_DELTA_REQUEST = "delta_request"
MSG_RECORD_BATCH = "record_batch"
# cas 7 (S234): the remote-inference request kind. It rides the same app_call
# transport as the sync delta and is discriminated by this type at
# ``serve_app_call``; the sync responder rejects it on parse (an unknown type to
# ``parse_delta_request``), so the two kinds never collide.
MSG_REMOTE_INFER = "remote_infer"
# cas 7 Lot 2 (S235): the streaming continuation kind (Option A, pull). The
# initial ``remote_infer`` request opens a server-side chunk buffer keyed by the
# (route-authenticated peer, request id) pair; the phone then pulls successive
# chunks with ``remote_infer_cont`` requests, each carrying the request id and a
# cursor, until a terminal done marker. It is discriminated by this type at
# ``serve_app_call`` alongside the other kinds; the sync responder rejects it on
# parse (an unknown type to ``parse_delta_request``), so the three never collide.
MSG_REMOTE_INFER_CONT = "remote_infer_cont"

# S203 (PRT-04): batch bounds. The sender bounds each answer to one chunk; the
# asker walks the journal chunk by chunk by threading each chunk's high-water into
# its next request (the existing monotonic advance, no new token). The receiver
# caps an incoming envelope defensively and REJECTS past the cap (never truncates).
#
# The receiver caps are set well above the sender bound (>= with margin) so a
# compliant sender -- including Bloc 2's per-record ML-DSA-65 signatures (~3.3 KB
# each), sequenced after this lot precisely so signed batches stay bounded -- is
# never rejected, while an unbounded or hostile envelope still cannot exhaust
# memory on parse. Bytes are measured on the serialised wire records, the same
# unit the sender bounds, so the two speak the same units.
SENDER_MAX_RECORDS = 256
SENDER_MAX_BYTES = 1_048_576  # 1 MiB
RECEIVER_MAX_RECORDS = 1024
RECEIVER_MAX_BYTES = 8_388_608  # 8 MiB


class Peer(Protocol):
    """The minimal contract a peer must satisfy: answer a request with a batch.

    A live peer (S180-S181) reaches the remote device over a Veilid private route; a
    fake peer answers from a local feed. The protocol never assumes anything beyond
    this single method, which is what keeps it transport-agnostic.
    """

    def fetch(self, request: dict) -> Any:
        ...


@dataclass(frozen=True)
class DeltaRequest:
    """A parsed delta request: the asking device and its watermark for the peer."""

    device: str
    watermark: int


@dataclass(frozen=True)
class RecordBatch:
    """A parsed batch answer: the records, the peer's high-water, and a reject count.

    Attributes:
        device: The responding device.
        high_water: The peer feed's current maximum sequence; the asking device
            advances its watermark to this once the batch is applied.
        records: The decoded record versions (latest per key from the peer).
        rejected: How many wire records failed to decode and were dropped.
        epoch: The responder feed's epoch (S204, CHF-05), or ``None`` for a
            pre-epoch peer (the field missing or malformed on the wire). The
            asker compares it to the peer's stored last-seen epoch and resets
            its watermark on a change; ``None`` never resets anything and falls
            through to the CHF-01 backstop.
    """

    device: str
    high_water: int
    records: list[SyncRecord] = field(default_factory=list)
    rejected: int = 0
    epoch: str | None = None


@dataclass(frozen=True)
class ApplyResult:
    """The outcome of applying an incoming batch.

    Attributes:
        converged: The converged set after reconciling the batch against the local
            set, one winner per key.
        conflicts: The conflict log from the reconciliation (retained divergences).
        new_watermark: The peer's high-water; advance the per-peer watermark to
            ``max(current, new_watermark)`` so it never regresses.
        applied: How many records were newly adopted into the local feed (a key new
            to us, a content change, or a clock advance), so applying is idempotent.
        rejected: How many wire records the batch dropped on decode.
    """

    converged: list[SyncRecord] = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    new_watermark: int = 0
    applied: int = 0
    rejected: int = 0


def _check_device(device: Any) -> None:
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")


def _check_watermark(watermark: Any) -> None:
    if isinstance(watermark, bool) or not isinstance(watermark, int) or watermark < 0:
        raise ValueError("watermark must be a non-negative integer")


# Outbound, gated: these act on records over the wire and refuse under Bulbe.


def _feed_epoch_of(feed: Any) -> str | None:
    """The feed's epoch read defensively (S204, CHF-05), or ``None``.

    Duck-typed like ``since_page``: a feed that predates ``feed_epoch`` simply
    has no epoch, so the batch omits the field (a pre-epoch sender). A failing
    or malformed read degrades the same way rather than killing the answer --
    the epoch is a repair signal, not a correctness requirement; the CHF-01
    backstop stays the floor.
    """
    reader = getattr(feed, "feed_epoch", None)
    if not callable(reader):
        return None
    try:
        value = reader()
    except Exception:
        logger.debug(
            "feed epoch unreadable; sending a pre-epoch batch", exc_info=True
        )
        return None
    return value if isinstance(value, str) and value else None


def build_delta_request(*, device: str, watermark: int) -> dict[str, Any]:
    """Build the asking device's delta request. Refuses under Bulbe."""
    assert_sync_allowed()
    _check_device(device)
    _check_watermark(watermark)
    return {
        "v": PROTOCOL_VERSION,
        "type": MSG_DELTA_REQUEST,
        "device": device,
        "watermark": watermark,
    }


def build_record_batch(
    feed: Any,
    *,
    device: str,
    watermark: int,
    max_count: int = SENDER_MAX_RECORDS,
    max_bytes: int = SENDER_MAX_BYTES,
) -> dict[str, Any]:
    """Build one bounded batch answer from the local feed. Refuses under Bulbe.

    S203 (PRT-04): the answer is a single chunk, not the whole delta. It reads a
    bounded page via the feed's ``since_page`` (at most ``max_count`` rows, a
    ``max_bytes`` wire budget) and advertises the CHUNK's max sequence as
    ``high_water`` -- not the feed's overall maximum -- so the asker's watermark
    advances chunk by chunk and the round loops while progress is made. A feed
    that predates ``since_page`` degrades to the unbounded ``since`` (the whole
    delta in one message, the feed's high-water), so the bound is additive and
    forward-compatible. The CHF-01 backstop is unchanged in kind: an impossible
    watermark still serves the full current set, now in bounded chunks.

    S204 (CHF-05): the batch carries the feed's epoch -- a property of the feed,
    independent of the delta, so the asker learns it even on a caught-up (empty)
    round. Old readers ignore the field; a pre-epoch feed omits it.
    """
    assert_sync_allowed()
    _check_device(device)
    _check_watermark(watermark)
    if hasattr(feed, "since_page"):
        delta = feed.since_page(
            int(watermark), max_count=max_count, max_bytes=max_bytes
        )
    else:  # pragma: no cover - forward/backward compatibility with a since-only feed
        delta = feed.since(int(watermark))
    batch: dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "type": MSG_RECORD_BATCH,
        "device": device,
        "high_water": int(delta.high_water),
        "records": encode_records(delta.records),
    }
    epoch = _feed_epoch_of(feed)
    if epoch is not None:
        batch["epoch"] = epoch
    return batch


def respond_to_request(
    feed: Any,
    raw_request: Any,
    *,
    device: str,
    max_count: int = SENDER_MAX_RECORDS,
    max_bytes: int = SENDER_MAX_BYTES,
) -> dict[str, Any]:
    """Answer an incoming raw request with a bounded batch. Refuses under Bulbe.

    Defensive on the request: an unparseable request gets a benign empty batch
    (high-water 0, no records) rather than an error or an over-send. The zero
    high-water is deliberate (PRT-01): a defensive answer must never advance the
    asker's watermark -- advertising the feed's real high-water with no records
    would let a request garbled in transit skip every delta up to it -- and 0 is
    a no-op under the peer store's monotonic max() advance. A valid request is
    answered with one bounded chunk (PRT-04); the asker loops for the rest.

    S204 (CHF-05): the benign batch carries the feed's epoch too -- it is the
    responder's true feed identity, and a benign answer can never advance the
    asker (high-water 0 stays a no-op under the monotonic advance), so an epoch
    learned from it is as good as one from a real chunk.
    """
    assert_sync_allowed()
    _check_device(device)
    request = parse_delta_request(raw_request)
    if request is None:
        benign: dict[str, Any] = {
            "v": PROTOCOL_VERSION,
            "type": MSG_RECORD_BATCH,
            "device": device,
            "high_water": 0,
            "records": [],
        }
        epoch = _feed_epoch_of(feed)
        if epoch is not None:
            benign["epoch"] = epoch
        return benign
    return build_record_batch(
        feed,
        device=device,
        watermark=request.watermark,
        max_count=max_count,
        max_bytes=max_bytes,
    )


def apply_record_batch(
    feed: Any,
    batch: Any,
    *,
    local_records: list[SyncRecord] | None = None,
) -> ApplyResult:
    """Reconcile an incoming batch into the local set and journal what changed.

    Refuses under Bulbe: this is the WIRE-side apply, the receiving half of a
    round. Accepts a parsed :class:`RecordBatch` or a raw wire dict
    (parsed defensively; an unparseable batch yields an empty result). The local set
    is taken from ``local_records`` if given, else from the feed's current snapshot.
    A winner is journalled when it is new to us, its content changed, or its clock
    advanced past ours (PRT-02: a same-content winner at a higher clock must still
    be adopted, otherwise the local clock lags and a later local edit, bumped from
    the stale clock, is silently superseded by older content). Applying the same
    batch twice still adopts nothing the second time (equal clocks and hashes).
    """
    assert_sync_allowed()
    return _apply_batch(feed, batch, local_records=local_records)


def apply_local_batch(
    feed: Any,
    batch: Any,
    *,
    local_records: list[SyncRecord] | None = None,
) -> ApplyResult:
    """Apply a batch of ALREADY-LOCALLY-HELD records; ungated by design (S207).

    The same merge core as :func:`apply_record_batch`, without the Bulbe gate.
    The Bulbe boundary gates the WIRE; this function moves nothing over any
    wire -- it reconciles records the device already holds on local disk into
    the local set, journalling local disk to local disk, the same posture as
    the engine's ungated ``publish`` (a local edit). Its one intended caller
    is the deferred-ledger approval seam (SYN-05): the record was fetched in
    Daily through the fully-gated round, quarantined, and the human's later
    approve is a local decision permitted in any mode, like pairing
    management. Records arriving over any wire MUST go through
    :func:`apply_record_batch`; using this on a wire path would bypass the
    mode boundary.
    """
    return _apply_batch(feed, batch, local_records=local_records)


def _apply_batch(
    feed: Any,
    batch: Any,
    *,
    local_records: list[SyncRecord] | None = None,
) -> ApplyResult:
    """The shared merge core: parse defensively, reconcile, journal winners."""
    if isinstance(batch, RecordBatch):
        parsed: RecordBatch | None = batch
    else:
        parsed = parse_record_batch(batch)
    if parsed is None:
        return ApplyResult()
    local = (
        list(local_records) if local_records is not None else feed.current_records()
    )
    merged = reconcile(local, parsed.records)
    local_by_key = {key_of(r): r for r in local}
    applied = 0
    for winner in merged.records:
        current = local_by_key.get(key_of(winner))
        if (
            current is None
            or current.clock < winner.clock
            or current.content_hash != winner.content_hash
        ):
            feed.record(winner)
            applied += 1
    return ApplyResult(
        converged=merged.records,
        conflicts=merged.conflicts,
        new_watermark=int(parsed.high_water),
        applied=applied,
        rejected=parsed.rejected,
    )


def sync_with_peer(
    feed: Any, peer: Peer, *, device: str, watermark: int
) -> ApplyResult:
    """Run one pull round against a peer and apply the answer. Refuses under Bulbe.

    Composes the round: build the request, ask the peer, apply the batch. The peer
    is injected, so this is the full exchange with no live transport.
    """
    assert_sync_allowed()
    request = build_delta_request(device=device, watermark=watermark)
    raw_batch = peer.fetch(request)
    return apply_record_batch(feed, raw_batch)


# Inbound, defensive: parsing is reading data, never acting; ungated, never raises.


def parse_delta_request(obj: Any) -> DeltaRequest | None:
    """Parse an incoming delta request, or return ``None`` on any problem."""
    try:
        if not isinstance(obj, dict):
            return None
        if obj.get("v") != PROTOCOL_VERSION:
            return None
        if obj.get("type") != MSG_DELTA_REQUEST:
            return None
        device = obj.get("device")
        if not isinstance(device, str) or not device:
            return None
        watermark = obj.get("watermark")
        if isinstance(watermark, bool) or not isinstance(watermark, int):
            return None
        if watermark < 0:
            return None
        return DeltaRequest(device=device, watermark=watermark)
    except Exception:
        logger.debug("Rejected an unparseable delta request", exc_info=True)
        return None


def parse_record_batch(
    obj: Any,
    *,
    max_count: int = RECEIVER_MAX_RECORDS,
    max_bytes: int = RECEIVER_MAX_BYTES,
) -> RecordBatch | None:
    """Parse an incoming batch, or return ``None`` on a malformed envelope.

    The envelope is validated strictly; the records inside are decoded defensively
    by the record decoder, which drops and counts any that fail, so a batch with a
    few bad records still yields the good ones.

    S203 (PRT-04): a defensive envelope cap. An envelope carrying more than
    ``max_count`` wire records, or whose serialised records exceed ``max_bytes``,
    is REJECTED (returns ``None``) -- never truncated, so a too-large envelope
    cannot silently drop records or exhaust memory on decode. A rejected batch
    surfaces through the existing parsed-honesty path (SYN-03): the round records
    a malformed answer and holds the watermark. The caps sit well above the
    sender bound, so a compliant sender's chunks are never rejected.
    """
    try:
        if not isinstance(obj, dict):
            return None
        if obj.get("v") != PROTOCOL_VERSION:
            return None
        if obj.get("type") != MSG_RECORD_BATCH:
            return None
        device = obj.get("device")
        if not isinstance(device, str) or not device:
            return None
        high_water = obj.get("high_water")
        if isinstance(high_water, bool) or not isinstance(high_water, int):
            return None
        if high_water < 0:
            return None
        raw_records = obj.get("records")
        if not isinstance(raw_records, list):
            return None
        # The envelope cap: reject (never truncate) past the count or byte bound.
        if len(raw_records) > max_count:
            logger.debug(
                "Rejected a record batch: %d records exceeds the cap %d",
                len(raw_records),
                max_count,
            )
            return None
        wire_bytes = len(
            json.dumps(
                raw_records, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
        if wire_bytes > max_bytes:
            logger.debug(
                "Rejected a record batch: %d wire bytes exceeds the cap %d",
                wire_bytes,
                max_bytes,
            )
            return None
        decoded = decode_records(raw_records)
        # S204 (CHF-05): the epoch is read defensively. A missing, non-string,
        # or empty epoch means a pre-epoch peer (None) -- never a reject; the
        # asker leaves its stored epoch untouched and the CHF-01 backstop
        # remains the floor, exactly as today.
        epoch_raw = obj.get("epoch")
        epoch = epoch_raw if isinstance(epoch_raw, str) and epoch_raw else None
        return RecordBatch(
            device=device,
            high_water=high_water,
            records=decoded.records,
            rejected=decoded.rejected,
            epoch=epoch,
        )
    except Exception:
        logger.debug("Rejected an unparseable record batch", exc_info=True)
        return None
