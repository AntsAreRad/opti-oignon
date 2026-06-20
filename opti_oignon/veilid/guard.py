#!/usr/bin/env python3
"""Bulbe binding-layer gate and shared error types for Veilid sync (S178, Theme 4).

Sync is a Daily-mode capability. Under Bulbe the Veilid node refuses to bind or
connect at all: a physical refusal at the layer where a connection would be
opened, not a policy flag a caller can flip. This mirrors network_bind_guard --
the security mode is read live and fail-secure (an indeterminable mode is treated
as Bulbe), and the gate cannot be disabled from configuration.

Why refuse rather than bind to loopback: Veilid is a peer-to-peer overlay, so a
loopback-only node reaches no peers; "bind to loopback" would be a no-op dressed
as a constraint. Refusing to come up is the honest boundary.

This is the foundational module of the sub-package: it carries the error
hierarchy everything else raises, and the lazy presence check for the optional
``veilid`` framework. It imports nothing heavy, so the package collects without
the framework installed.
"""

from __future__ import annotations

import importlib.util
import logging

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True


# Shared error hierarchy (raised across the sub-package)


class VeilidError(RuntimeError):
    """Base for every controlled Veilid sync failure."""


class VeilidDisabledInBulbe(VeilidError):
    """Veilid sync was invoked under Bulbe, where networking is refused."""


class VeilidUnavailable(VeilidError):
    """The veilid framework or a usable connector is not available."""


class VeilidStateError(VeilidError):
    """A lifecycle operation was requested from an invalid state."""


class VeilidTimeout(VeilidError):
    """A Veilid operation did not complete within its timeout."""


# The binding-layer gate


def current_mode() -> str:
    """The live security mode, fail-secure to ``bulbe`` when undeterminable.

    The import is lazy and per-call so the gate always reflects the current mode
    and so this module collects without the backend; any failure to resolve the
    mode is treated as Bulbe (fail-secure), which refuses sync.
    """
    try:
        from opti_oignon.security_mode import get_current_mode

        return get_current_mode()
    except Exception:
        logger.warning(
            "Cannot determine security mode; treating as 'bulbe' (fail-secure)."
        )
        return "bulbe"


def bulbe_disabled() -> bool:
    """True when Veilid sync must be refused for the current mode."""
    return current_mode() == "bulbe"


def assert_sync_allowed() -> None:
    """Raise :class:`VeilidDisabledInBulbe` when sync is not permitted now.

    Called at the binding layer before any connection is opened (node start and
    attach). It is a hard gate: there is no parameter to bypass it; it reads the
    live, fail-secure mode.
    """
    if bulbe_disabled():
        raise VeilidDisabledInBulbe(
            "Veilid sync is disabled in Bulbe mode: peer-to-peer networking is "
            "refused at the binding layer. Switch to Daily mode to sync."
        )


def veilid_available() -> bool:
    """True when the optional ``veilid`` framework is importable.

    Uses ``find_spec`` so it never executes the framework; safe at any time and
    in any environment, including the sandbox where ``veilid`` is absent.
    """
    try:
        return importlib.util.find_spec("veilid") is not None
    except Exception:  # pragma: no cover - find_spec is defensive here
        return False
