#!/usr/bin/env python3
"""Veilid sync sub-package.

Optional peer-to-peer sync over Veilid (https://veilid.com): a privacy-first,
end-to-end encrypted overlay that lets a user carry conversations, memory, and
skills across their own devices without a server in the middle.

Sync is a Daily-mode capability. Under Bulbe the node refuses to bind at the
binding layer (see ``guard``). This package builds the foundation -- the node lifecycle
and the binding-layer gate; the async client wrapper, the veilid-server
packaging script, the optional ``[veilid]`` dependency group, and the sync
protocol and UI land in the sibling modules.

The heavy ``veilid`` import is lazy and guarded throughout, so this package
collects without the framework installed.
"""

from __future__ import annotations

FEATURE_AVAILABLE = True
checkpoint_before_apply = True

from opti_oignon.veilid.guard import (  # noqa: E402
    VeilidDisabledInBulbe,
    VeilidError,
    VeilidStateError,
    VeilidTimeout,
    VeilidUnavailable,
    assert_sync_allowed,
    bulbe_disabled,
    current_mode,
    veilid_available,
)
from opti_oignon.veilid.node import (  # noqa: E402
    NodeState,
    VeilidNode,
    get_node,
    reset_node,
    set_node,
)
from opti_oignon.veilid.client import VeilidClient  # noqa: E402
from opti_oignon.veilid.records import (  # noqa: E402
    RECORD_FORMAT_VERSION,
    RECORD_KINDS,
    DecodeResult,
    RecordKind,
    SyncRecord,
    content_hash_for,
    decode_record,
    decode_records,
    encode_record,
    encode_records,
    from_wire_json,
    key_of,
    new_record,
    to_wire_json,
    verify_record_hash,
)
from opti_oignon.veilid.reconcile import (  # noqa: E402
    ConflictEntry,
    MergeResult,
    choose_winner,
    reconcile,
    reconcile_many,
)
from opti_oignon.veilid.change_feed import (  # noqa: E402
    ChangeFeed,
    Delta,
    get_change_feed,
    reset_change_feed,
    set_change_feed,
)
from opti_oignon.veilid.protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    ApplyResult,
    DeltaRequest,
    Peer,
    RecordBatch,
    apply_record_batch,
    build_delta_request,
    build_record_batch,
    parse_delta_request,
    parse_record_batch,
    respond_to_request,
    sync_with_peer,
)
from opti_oignon.veilid.peers import (  # noqa: E402
    PeerRecord,
    PeerStore,
    get_peer_store,
    reset_peer_store,
    set_peer_store,
)
from opti_oignon.veilid.sync_engine import (  # noqa: E402
    SENSITIVE_KINDS,
    PeerNotFound,
    RoundResult,
    SyncEngine,
    get_sync_engine,
    record_from_payload,
    reset_sync_engine,
    set_sync_engine,
)
from opti_oignon.veilid.transport import (  # noqa: E402
    ClientRouteMessenger,
    RouteMessenger,
    VeilidPeer,
    decode_answer,
    resolve_live_peer,
    resolve_self_routing_key,
    serve_app_call,
)
from opti_oignon.veilid.pairing import (  # noqa: E402
    PAIRING_FORMAT_VERSION,
    PAIRING_TYPE,
    ParsedPairing,
    accept_pairing_payload,
    build_pairing_payload,
    decode_pairing_json,
    encode_pairing_json,
    pairing_integrity,
    parse_pairing_payload,
    verify_pairing_payload,
)
from opti_oignon.veilid.producers import (  # noqa: E402
    conversation_record,
    memory_archive_record,
    memory_canonical_record,
    skill_record,
    tombstone_record,
)
from opti_oignon.veilid.sync_status import (  # noqa: E402
    RoundOutcome,
    SyncStatusStore,
    get_sync_status_store,
    reset_sync_status_store,
    set_sync_status_store,
)

__all__ = [
    "FEATURE_AVAILABLE",
    "VeilidDisabledInBulbe",
    "VeilidError",
    "VeilidStateError",
    "VeilidTimeout",
    "VeilidUnavailable",
    "assert_sync_allowed",
    "bulbe_disabled",
    "current_mode",
    "veilid_available",
    "NodeState",
    "VeilidNode",
    "get_node",
    "reset_node",
    "set_node",
    "VeilidClient",
    "RECORD_FORMAT_VERSION",
    "RECORD_KINDS",
    "DecodeResult",
    "RecordKind",
    "SyncRecord",
    "content_hash_for",
    "decode_record",
    "decode_records",
    "encode_record",
    "encode_records",
    "from_wire_json",
    "key_of",
    "new_record",
    "to_wire_json",
    "verify_record_hash",
    "ConflictEntry",
    "MergeResult",
    "choose_winner",
    "reconcile",
    "reconcile_many",
    "ChangeFeed",
    "Delta",
    "get_change_feed",
    "reset_change_feed",
    "set_change_feed",
    "PROTOCOL_VERSION",
    "ApplyResult",
    "DeltaRequest",
    "Peer",
    "RecordBatch",
    "apply_record_batch",
    "build_delta_request",
    "build_record_batch",
    "parse_delta_request",
    "parse_record_batch",
    "respond_to_request",
    "sync_with_peer",
    "PeerRecord",
    "PeerStore",
    "get_peer_store",
    "reset_peer_store",
    "set_peer_store",
    "SENSITIVE_KINDS",
    "PeerNotFound",
    "RoundResult",
    "SyncEngine",
    "get_sync_engine",
    "record_from_payload",
    "reset_sync_engine",
    "set_sync_engine",
    "ClientRouteMessenger",
    "RouteMessenger",
    "VeilidPeer",
    "decode_answer",
    "resolve_live_peer",
    "resolve_self_routing_key",
    "serve_app_call",
    "PAIRING_FORMAT_VERSION",
    "PAIRING_TYPE",
    "ParsedPairing",
    "accept_pairing_payload",
    "build_pairing_payload",
    "decode_pairing_json",
    "encode_pairing_json",
    "pairing_integrity",
    "parse_pairing_payload",
    "verify_pairing_payload",
    "conversation_record",
    "memory_archive_record",
    "memory_canonical_record",
    "skill_record",
    "tombstone_record",
    "RoundOutcome",
    "SyncStatusStore",
    "get_sync_status_store",
    "reset_sync_status_store",
    "set_sync_status_store",
]
