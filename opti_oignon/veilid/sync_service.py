#!/usr/bin/env python3
"""Background sync driver for Veilid (SYN-01 orchestration).

The missing trigger. The domain producers journal a change locally on every
save (conversation, note, canonical memory, skill), but nothing pulls a round,
so a journalled change never reaches a paired device until a human manually
fires ``POST /api/sync/peers/{peer_id}/run``. This service closes that gap: on
a cadence it drives ``SyncEngine.run_round`` against each confirmed peer, so a
local edit converges on the other device without a manual gesture.

Conservative posture, because an auto-driver initiates peer contact on its own:

  * OPT-IN. The driver is inert unless explicitly enabled (``OPTI_SYNC_AUTORUN``
    truthy, default off). Presence of the framework is NOT enough -- contacting
    peers automatically is a deliberate choice the user makes.
  * HARD-STOPPED in Bulbe. When the mode refuses to bind (``guard``), the driver
    never enumerates or contacts a peer. The check runs every pass, so a
    Daily->Bulbe transition silences it on the next tick.
  * CONFIRMED peers only. A pending (pre-PAIR-02) peer is skipped, never
    contacted; the engine would refuse it anyway.
  * NO auto-approval of sensitive records. ``run_round`` is called without an
    approval function, so a received skill (a SENSITIVE_KIND) is deferred to the
    ledger/panel for the human gate; only the non-sensitive kinds apply
    automatically. The driver never widens the trust surface of a round.

Host-bound by construction at the live edge: ``transport.resolve_live_peer``
returns ``None`` when no attached node and client can supply a route (the
framework absent, or no live node), so without a live Veilid node every pass is
a safe no-op that records a transport-unavailable failure and moves on. The
loop logic is unit-tested in-container with injected fakes; the live round is
proven on the maintainer's machine.

Every dependency is injectable (engine, peer store, status store, node, the
live-peer resolver, and the enable/mode gates) and resolves lazily to the
process singleton otherwise, the same idiom as the engine and feed. The module
therefore imports only stdlib at load time -- the veilid chain is pulled inside
the resolvers, never at import -- so it collects without the framework.

This module does NOT start itself. Arming it in the application lifespan is a
separate, deliberate step, so the auto-driver is turned on consciously.

Kerckhoffs: nothing here is secret. The gate is the user's opt-in and the mode,
not obscurity; the routing material the resolver uses is public by design.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Module sentinels (project convention).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# The opt-in environment switch. Truthy values, case-insensitive.
AUTORUN_ENV = "OPTI_SYNC_AUTORUN"
_TRUTHY = {"1", "true", "yes", "on"}

# Defaults. The interval is deliberately unhurried; sync is eventual, not live.
DEFAULT_INTERVAL_SECONDS = 300.0
MIN_INTERVAL_SECONDS = 1.0


def env_autorun_enabled() -> bool:
    """Whether the auto-driver is enabled via the environment (default False)."""
    return os.environ.get(AUTORUN_ENV, "").strip().lower() in _TRUTHY


class SyncService:
    """A background driver that pulls a sync round per confirmed peer.

    The loop is gated on every pass (opt-in AND not Bulbe); when the gate is
    closed the pass is a cheap no-op, so flipping the switch off or entering
    Bulbe silences the driver on the next tick without a restart.
    """

    def __init__(
        self,
        *,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        engine: Any | None = None,
        store: Any | None = None,
        status: Any | None = None,
        node: Any | None = None,
        peer_resolver: Callable[..., Any] | None = None,
        enabled_fn: Callable[[], bool] | None = None,
        bulbe_fn: Callable[[], bool] | None = None,
    ) -> None:
        interval = float(interval_seconds)
        if interval < MIN_INTERVAL_SECONDS:
            raise ValueError(
                f"interval_seconds must be >= {MIN_INTERVAL_SECONDS}"
            )
        self._interval = interval
        self._engine = engine
        self._store = store
        self._status = status
        self._node = node
        self._peer_resolver = peer_resolver
        self._enabled_fn = enabled_fn
        self._bulbe_fn = bulbe_fn
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # Lazy dependency resolution (injection wins; otherwise the singleton).

    def _resolve_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        from opti_oignon.veilid.sync_engine import get_sync_engine

        return get_sync_engine()

    def _resolve_store(self) -> Any:
        if self._store is not None:
            return self._store
        from opti_oignon.veilid.peers import get_peer_store

        return get_peer_store()

    def _resolve_status(self) -> Any:
        if self._status is not None:
            return self._status
        from opti_oignon.veilid.sync_status import get_sync_status_store

        return get_sync_status_store()

    def _resolve_node(self) -> Any:
        if self._node is not None:
            return self._node
        try:
            from opti_oignon.veilid.node import get_node

            return get_node()
        except Exception:  # pragma: no cover - node resolution is defensive
            logger.debug("sync driver: node resolution failed", exc_info=True)
            return None

    def _resolve_live_peer(self, peer_id: str, store: Any, node: Any, device: str) -> Any:
        if self._peer_resolver is not None:
            return self._peer_resolver(
                peer_id, store=store, node=node, device=device
            )
        from opti_oignon.veilid.transport import resolve_live_peer

        return resolve_live_peer(peer_id, store=store, node=node, device=device)

    def _enabled(self) -> bool:
        return self._enabled_fn() if self._enabled_fn is not None else env_autorun_enabled()

    def _bulbe(self) -> bool:
        if self._bulbe_fn is not None:
            return self._bulbe_fn()
        from opti_oignon.veilid.guard import bulbe_disabled

        return bulbe_disabled()

    # One pass.

    def run_once(self) -> int:
        """Drive one round per confirmed peer; return the number that succeeded.

        Gated first: a no-op (0) when the driver is disabled or the mode is
        Bulbe -- no peer is enumerated or contacted. Otherwise each CONFIRMED
        peer is resolved over the live transport and rounded; a peer with no
        live route (transport down) and a peer whose round raises each record a
        failure and the pass continues to the next. Sensitive records are not
        auto-approved (no approval function is passed): they defer to the panel.
        """
        if not self._enabled():
            return 0
        if self._bulbe():
            # Mode refuses to bind: never enumerate or contact a peer.
            return 0

        store = self._resolve_store()
        try:
            peers = store.list_peers()
        except Exception:
            logger.warning("sync driver: listing peers failed", exc_info=True)
            return 0
        if not peers:
            return 0

        engine = self._resolve_engine()
        status = self._resolve_status()
        node = self._resolve_node()
        device = getattr(engine, "device", "") or ""

        succeeded = 0
        for rec in peers:
            if getattr(rec, "pending", False):
                # A pre-PAIR-02 peer gates nothing; skip rather than be refused.
                continue
            peer_id = getattr(rec, "peer_id", "") or ""
            if not peer_id:
                continue
            try:
                live = self._resolve_live_peer(peer_id, store, node, device)
                if live is None:
                    status.record_failure(peer_id, "transport unavailable")
                    continue
                # No approval_fn: a received sensitive record defers to the
                # ledger/panel rather than applying under an automated round.
                result = engine.run_round(peer_id, live)
                status.record_round(result)
                succeeded += 1
            except Exception as exc:  # noqa: BLE001 - one peer must not abort the pass
                reason = str(exc) or exc.__class__.__name__
                try:
                    status.record_failure(peer_id, reason)
                except Exception:  # pragma: no cover - status must never abort the pass
                    logger.debug(
                        "sync driver: recording failure for %s failed",
                        peer_id,
                        exc_info=True,
                    )
                logger.warning(
                    "sync driver: round failed for peer %s: %s", peer_id, reason
                )
        return succeeded

    # Lifecycle.

    def start(self) -> bool:
        """Start the background loop; return False if it is already running.

        Idempotent: a second call while running is a no-op that returns False.
        The thread is a daemon, so it never blocks process exit; ``stop`` is the
        clean shutdown.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="oo-sync-service", daemon=True
            )
            self._thread.start()
            return True

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal the loop to stop and join it (best-effort within ``timeout``)."""
        with self._lock:
            self._stop.set()
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=timeout)

    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def _loop(self) -> None:
        # A modest initial wait so start() returns promptly and the driver does
        # not contact peers the instant the thread spins up (e.g. right after a
        # pairing or at app boot). Interruptible by stop().
        self._stop.wait(min(self._interval, 10.0))
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:  # pragma: no cover - a tick must never kill the loop
                logger.exception("sync driver: tick failed")
            self._stop.wait(self._interval)


# Module-level singleton with a reset hook (one driver per process, testable).
# The same idiom as the engine, feed, peer store, and status store singletons.

_service: SyncService | None = None
_service_lock = threading.Lock()


def get_sync_service() -> SyncService:
    """Return the process sync driver, creating it once."""
    global _service
    with _service_lock:
        if _service is None:
            _service = SyncService()
        return _service


def set_sync_service(service: SyncService | None) -> None:
    """Install a specific driver as the process singleton (used by tests)."""
    global _service
    with _service_lock:
        _service = service


def reset_sync_service() -> None:
    """Stop and clear the process singleton so the next get creates a fresh one."""
    global _service
    with _service_lock:
        if _service is not None:
            try:
                _service.stop()
            except Exception:  # pragma: no cover - reset is defensive
                logger.debug("sync driver: stop during reset failed", exc_info=True)
        _service = None


def arm_if_enabled(
    *,
    factory: Callable[[], SyncService] | None = None,
    enabled_fn: Callable[[], bool] | None = None,
) -> bool:
    """Start the process driver iff the auto-run opt-in is on; else a no-op.

    The arming seam for the application lifespan. Conservative: when the opt-in
    is off (the default) NO thread is started and the function returns False, so
    the driver costs nothing for an install that never opted in. The opt-in is
    the ONLY arming gate; the mode boundary is the driver's own per-pass gate
    (it hard-stops under Bulbe before contacting any peer), so a thread armed
    while in Bulbe stays a safe no-op and resumes if the mode later returns to
    Daily -- no restart needed. Never raises: arming must not break startup, so
    a failure is logged and reported as False. ``factory``/``enabled_fn`` are
    injectable for tests.
    """
    on = (enabled_fn or env_autorun_enabled)()
    if not on:
        return False
    try:
        service = (factory or get_sync_service)()
        started = service.start()
        if started:
            logger.info("veilid sync auto-driver armed (%s on)", AUTORUN_ENV)
        return started
    except Exception:  # noqa: BLE001 - arming must never break application startup
        logger.warning("veilid sync auto-driver: arming failed", exc_info=True)
        return False
