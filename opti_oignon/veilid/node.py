#!/usr/bin/env python3
"""Veilid node lifecycle.

The state machine for a single Veilid node: start (bring the node up and connect
to the running veilid-server), attach (join the network), detach (leave it), and
stop (bring the node down). The node surfaces its state and never leaves a
misleading ``attached`` after a failed transition.

The node drives an injected connector -- the async client wrapper
(opti_oignon/veilid/client.py) in production, a fake in tests -- so
the lifecycle is exercised in isolation without the veilid framework or a live
server. The connector exposes ``connect`` / ``attach`` / ``detach`` /
``shutdown`` and, optionally, ``attachment_state``. The heavy ``veilid`` import
lives in the connector, never here; this module collects without it.

Bulbe boundary: start and attach consult the binding-layer gate
(``guard.assert_sync_allowed``) before any connection is opened or the network
is joined, so a node cannot come up or attach under Bulbe. The gate reads the
live, fail-secure mode. Detach and stop are never gated: a node must always be
able to leave the network and shut down, especially right after a mode change.
Sync is a Daily-only capability.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Any, Callable

from opti_oignon.veilid.guard import (
    VeilidError,
    VeilidStateError,
    VeilidUnavailable,
    assert_sync_allowed,
    bulbe_disabled,
    veilid_available,
)

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True


class NodeState(str, Enum):
    """The node's lifecycle state.

    Stable states are STOPPED, STARTED (running, detached from the network),
    ATTACHED (running and attached), and ERROR. The others are transient,
    held only for the duration of a transition.
    """

    STOPPED = "stopped"
    STARTING = "starting"
    STARTED = "started"
    ATTACHING = "attaching"
    ATTACHED = "attached"
    DETACHING = "detaching"
    STOPPING = "stopping"
    ERROR = "error"


# States in which the node is up (a connection exists).
_RUNNING = frozenset(
    {
        NodeState.STARTED,
        NodeState.ATTACHING,
        NodeState.ATTACHED,
        NodeState.DETACHING,
    }
)


# Audit hook (lazy / guarded): lifecycle events join the hash-chain audit log


def _audit(action: str, **details: Any) -> None:
    """Record a node lifecycle event in the hash-chain audit log, best-effort."""
    try:
        from opti_oignon.signed_audit_log import chain_log

        chain_log(
            event_type="veilid_lifecycle",
            source="veilid.node",
            action=action,
            severity="INFO",
            **details,
        )
    except Exception:  # pragma: no cover - audit is best-effort
        logger.debug("veilid audit log unavailable", exc_info=True)


def _default_connector() -> Any | None:
    """Build the production connector (the Goal 3 async client), lazily and guarded.

    Returns None when the veilid framework or the client wrapper is unavailable,
    so start() refuses cleanly with VeilidUnavailable rather than raising at
    import. When the client wrapper is absent this resolves to None.
    """
    if not veilid_available():
        return None
    try:
        from opti_oignon.veilid.client import VeilidClient
    except Exception:  # pragma: no cover - client lands in Goal 3 / constrained envs
        return None
    try:
        return VeilidClient()
    except Exception:  # pragma: no cover - defensive
        return None


class VeilidNode:
    """A single Veilid node and its lifecycle, driving an injected connector.

    Thread-safe: state transitions are serialised under a lock. A connector may
    be injected directly (tests) or resolved lazily from a factory (production:
    the async client wrapper). All transitions are fail-secure: an underlying
    failure settles the node in a truthful state (ERROR, or back to STARTED for
    a failed attach) and surfaces a typed VeilidError, never an arbitrary one.
    """

    def __init__(
        self,
        *,
        connector: Any = None,
        connector_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._state = NodeState.STOPPED
        self._connector = connector
        self._connector_factory = connector_factory
        self._last_error = ""

    # State surface

    def state(self) -> NodeState:
        with self._lock:
            return self._state

    def is_running(self) -> bool:
        with self._lock:
            return self._state in _RUNNING

    def is_attached(self) -> bool:
        with self._lock:
            return self._state == NodeState.ATTACHED

    def connector(self) -> Any:
        """The node's connector (the async client in production), or None.

        Returned only for a resolved connector; the live-peer resolver uses it to
        drive a request over a private route. Never resolves a connector as a
        side effect: a node that has never started returns None.
        """
        with self._lock:
            return self._connector

    def status(self) -> dict[str, Any]:
        """A snapshot of the node's state plus the binding context.

        Includes the connector's reported attachment state when it exposes one;
        querying it is best-effort and never raises.
        """
        with self._lock:
            state = self._state
            last_error = self._last_error
            connector = self._connector
        attachment = ""
        if connector is not None and hasattr(connector, "attachment_state"):
            try:
                attachment = str(connector.attachment_state() or "")
            except Exception:  # pragma: no cover - connector query is best-effort
                attachment = ""
        return {
            "state": state.value,
            "running": state in _RUNNING,
            "attached": state == NodeState.ATTACHED,
            "attachment": attachment,
            "bulbe_disabled": bulbe_disabled(),
            "veilid_available": veilid_available(),
            "last_error": last_error,
        }

    # Internal helpers

    def _resolve_connector(self) -> Any:
        if self._connector is not None:
            return self._connector
        factory = self._connector_factory or _default_connector
        connector = factory()
        if connector is None:
            raise VeilidUnavailable(
                "no Veilid connector available (the veilid framework or the "
                "client wrapper is not installed)"
            )
        self._connector = connector
        return connector

    def _fail(self, message: str, *, revert: NodeState = NodeState.ERROR) -> None:
        with self._lock:
            self._state = revert
            self._last_error = message
        logger.warning("veilid node: %s", message)

    # Lifecycle transitions

    def start(self) -> dict[str, Any]:
        """Bring the node up and connect. Refused under Bulbe; idempotent when up."""
        assert_sync_allowed()  # binding-layer gate: raises under Bulbe
        with self._lock:
            if self._state in (NodeState.STARTED, NodeState.ATTACHED):
                return self.status()
            if self._state not in (NodeState.STOPPED, NodeState.ERROR):
                raise VeilidStateError(f"cannot start from {self._state.value}")
            self._state = NodeState.STARTING
            self._last_error = ""
        try:
            connector = self._resolve_connector()
            connector.connect()
        except VeilidError:
            self._fail("start failed")
            raise
        except Exception as exc:
            self._fail(f"start failed: {exc!r}")
            raise VeilidError(f"start failed: {exc}") from exc
        with self._lock:
            self._state = NodeState.STARTED
        _audit("start")
        return self.status()

    def attach(self) -> dict[str, Any]:
        """Join the network. Refused under Bulbe; idempotent when already attached."""
        assert_sync_allowed()  # binding-layer gate: raises under Bulbe
        with self._lock:
            if self._state == NodeState.ATTACHED:
                return self.status()
            if self._state != NodeState.STARTED:
                raise VeilidStateError(f"cannot attach from {self._state.value}")
            connector = self._connector
            self._state = NodeState.ATTACHING
        try:
            connector.attach()
        except VeilidError:
            self._fail("attach failed", revert=NodeState.STARTED)
            raise
        except Exception as exc:
            self._fail(f"attach failed: {exc!r}", revert=NodeState.STARTED)
            raise VeilidError(f"attach failed: {exc}") from exc
        with self._lock:
            self._state = NodeState.ATTACHED
        _audit("attach")
        return self.status()

    def detach(self) -> dict[str, Any]:
        """Leave the network, staying up. Never gated; idempotent when detached."""
        with self._lock:
            if self._state == NodeState.STARTED:
                return self.status()
            if self._state != NodeState.ATTACHED:
                raise VeilidStateError(f"cannot detach from {self._state.value}")
            connector = self._connector
            self._state = NodeState.DETACHING
        try:
            connector.detach()
        except VeilidError:
            self._fail("detach failed", revert=NodeState.ATTACHED)
            raise
        except Exception as exc:
            self._fail(f"detach failed: {exc!r}", revert=NodeState.ATTACHED)
            raise VeilidError(f"detach failed: {exc}") from exc
        with self._lock:
            self._state = NodeState.STARTED
        _audit("detach")
        return self.status()

    def stop(self) -> dict[str, Any]:
        """Bring the node down. Never gated; idempotent when already stopped.

        Always settles in STOPPED: even if the connector's shutdown fails, the
        node is recorded down and a typed VeilidError is surfaced for the caller.
        """
        with self._lock:
            if self._state == NodeState.STOPPED:
                return self.status()
            if self._state == NodeState.STOPPING:
                raise VeilidStateError("cannot stop from stopping")
            connector = self._connector
            self._state = NodeState.STOPPING
        error: BaseException | None = None
        if connector is not None:
            try:
                connector.shutdown()
            except Exception as exc:  # the node ends stopped regardless
                error = exc
        with self._lock:
            self._state = NodeState.STOPPED
            self._last_error = "" if error is None else f"stop failed: {error!r}"
        _audit("stop")
        if error is not None:
            logger.warning("veilid node: stop failed: %r", error)
            if isinstance(error, VeilidError):
                raise error
            raise VeilidError(f"stop failed: {error}") from error
        return self.status()


# Module-level singleton (one node per process; reset for tests).
# VLD-01: creation is guarded by a lock, the same idiom as the engine, feed,
# peer-store, and status singletons -- two racing first calls must not build
# two node state machines (and potentially two connectors).

_NODE: VeilidNode | None = None
_NODE_LOCK = threading.Lock()


def get_node() -> VeilidNode:
    global _NODE
    with _NODE_LOCK:
        if _NODE is None:
            _NODE = VeilidNode()
        return _NODE


def set_node(node: VeilidNode | None) -> None:
    global _NODE
    with _NODE_LOCK:
        _NODE = node


def reset_node() -> None:
    global _NODE
    with _NODE_LOCK:
        _NODE = None
