#!/usr/bin/env python3
"""Emergency stop (S215): panic control that makes the machine quiet, plus resume.

An availability/safety control, NOT a security boundary. It is explicitly
distinct from the S126/S128 web-search kill switch (a module unload whose
re-enable requires a multi-factor ceremony): resume here needs no ceremony.
Authentication is still required at the route layer (routes_security).

Semantics, as arbitrated at the S215 gate:

- ``stop()`` sets the global "stopped" flag FIRST, so the admission path
  closes before the drain begins and the flag is set even if every later
  step fails (fail-secure on the flag, by construction). It then runs the
  ordered stop steps, each fail-tolerant: a failing step is logged and
  recorded, and the sequence continues -- the machine must end quiet even
  if one primitive errors. The sequence: cancel in-flight generations
  (executor + agentic executor), cancel agent runs, stop the coding
  background run, unload models on every registered inference backend
  (frees VRAM and halts compute; no privileges needed -- stopping a
  systemd-managed Ollama service stays a documented host action), destroy
  all sandbox sessions (the manager is called, never modified), and stop
  the Veilid node (``node.stop`` is never gated, verified F9e). The
  optional ``drop_to_bulbe`` variant escalates the security mode in the
  same gesture (Daily -> Bulbe is the no-ceremony direction). Audit-chained.

- ``resume()`` clears the flag, probes the Ollama-side client
  (``health_check`` on the registry's active backend, reported honestly;
  optional single-model warmup), and restarts the Veilid node only when it
  was running at stop time AND the current mode permits it (``node.start``
  carries the existing fail-secure Bulbe gate; the refusal is caught and
  reported, never overridden). Audit-chained. No ceremony.

- The flag is in-process only: a crash-restart comes back UNSTOPPED by
  design (arbitrated S215 -- persistence would create a lockout-on-boot
  failure mode for an availability control). The audit chain carries the
  history.

The Resource Governor cycle later absorbs this surface as R-04: its
admission gate refuses against the same ``is_stopped()`` flag.

Every collaborator is resolved lazily through a small ``_resolve_*``
function so the orchestration stays import-light and each primitive is
independently fakeable in tests (the proof seams for step order and
per-step fail-tolerance).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global state (in-process only; see the module docstring)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_stopped: bool = False
_stopped_since: float | None = None
_stopped_by: str = ""
_veilid_was_running: bool = False
_last_stop: dict[str, Any] | None = None
_last_resume: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Resolvers (lazy; monkeypatchable proof seams)
# ---------------------------------------------------------------------------

def _resolve_executors() -> list[Any]:
    """The generation executors carrying a cooperative ``cancel()``."""
    found: list[Any] = []
    try:
        from opti_oignon.executor import executor as _ex
        if _ex is not None:
            found.append(_ex)
    except Exception:
        pass
    try:
        from opti_oignon.agentic_executor import agentic_executor as _aex
        if _aex is not None:
            found.append(_aex)
    except Exception:
        pass
    return found


def _resolve_run_manager() -> Any:
    """The Odysseus agent run manager (cooperative ``cancel()``)."""
    try:
        from opti_oignon.api.routes_agent import get_run_manager
        return get_run_manager()
    except Exception:
        return None


def _resolve_coding_run_state() -> Any:
    """The coding agent's background run state (graceful ``stop()``)."""
    try:
        from opti_oignon.api import routes_coding
        return getattr(routes_coding, "_run_state", None)
    except Exception:
        return None


def _resolve_backend_registry() -> Any:
    """The inference backend registry (per-backend ``unload_all``)."""
    try:
        from opti_oignon.inference_backend import get_backend_registry
        return get_backend_registry()
    except Exception:
        return None


def _resolve_sandbox_manager() -> Any:
    """The sandbox manager singleton (``list_sessions`` / ``destroy_sandbox``).

    Called, never modified: the destroy paths shipped with the Sandbox
    Workspace cycle and need no edit for this lot.
    """
    try:
        from opti_oignon import sandbox_manager as _sm
        return getattr(_sm, "sandbox_manager", None)
    except Exception:
        return None


def _resolve_node() -> Any:
    """The Veilid node singleton (``stop`` never gated; ``start`` Bulbe-gated)."""
    try:
        from opti_oignon.veilid.node import get_node
        return get_node()
    except Exception:
        return None


def _resolve_mode_manager() -> Any:
    """The security mode manager (``escalate_to_bulbe``, no-ceremony direction)."""
    try:
        from opti_oignon.security_mode import security_mode_manager
        return security_mode_manager
    except Exception:
        return None


def _resolve_warmup() -> Any:
    """The model warmup helper (optional resume warmup)."""
    try:
        from opti_oignon.model_warmup import ModelWarmup
        return ModelWarmup()
    except Exception:
        return None


def _is_bulbe_refusal(exc: BaseException) -> bool:
    """True when ``exc`` is the Veilid guard's Bulbe refusal.

    Kept as a seam so tests can classify without importing the guard.
    """
    try:
        from opti_oignon.veilid.guard import VeilidDisabledInBulbe
        return isinstance(exc, VeilidDisabledInBulbe)
    except Exception:
        return False


def _chain(action: str, severity: str, **details: Any) -> None:
    """Append the audit-chain row; never raises (the chain is best-effort)."""
    try:
        from opti_oignon.signed_audit_log import chain_log
        chain_log(
            event_type="emergency_stop",
            source="emergency_stop",
            action=action,
            severity=severity,
            **details,
        )
    except Exception as exc:
        logger.debug("emergency stop: audit append failed: %s", exc)


# ---------------------------------------------------------------------------
# Flag surface (the admission path refuses against this; R-04 inherits it)
# ---------------------------------------------------------------------------

def is_stopped() -> bool:
    """True when the emergency stop is engaged."""
    with _lock:
        return _stopped


def status() -> dict[str, Any]:
    """The flag plus the last stop/resume records (for the UI and the API)."""
    with _lock:
        return {
            "stopped": _stopped,
            "since": _stopped_since if _stopped else None,
            "by": _stopped_by if _stopped else "",
            "last_stop": _last_stop,
            "last_resume": _last_resume,
        }


def refusal_payload() -> dict[str, Any]:
    """The uniform honest-refusal body for a stopped system."""
    with _lock:
        since = _stopped_since
    return {
        "error": "emergency_stopped",
        "message": (
            "Emergency stop is engaged: new work is refused until resume."
        ),
        "since": since,
    }


def guard_http() -> None:
    """REST admission guard: raise 503 with the refusal payload when stopped.

    A stopped system answers requests honestly -- refused, not hung.
    """
    if not is_stopped():
        return
    from fastapi import HTTPException
    raise HTTPException(status_code=503, detail=refusal_payload())


def reset_for_tests() -> None:
    """Restore the pristine in-process state (test hook)."""
    global _stopped, _stopped_since, _stopped_by
    global _veilid_was_running, _last_stop, _last_resume
    with _lock:
        _stopped = False
        _stopped_since = None
        _stopped_by = ""
        _veilid_was_running = False
        _last_stop = None
        _last_resume = None


# ---------------------------------------------------------------------------
# Stop steps (each wrapped fail-tolerant by the sequencer)
# ---------------------------------------------------------------------------

def _step_cancel_generations() -> dict[str, Any]:
    cancelled: list[str] = []
    errors: list[str] = []
    for ex in _resolve_executors():
        name = type(ex).__name__
        try:
            ex.cancel()
            cancelled.append(name)
        except Exception as exc:  # one executor failing must not skip the next
            errors.append(f"{name}: {exc}")
    out: dict[str, Any] = {"cancelled": cancelled}
    if errors:
        out["errors"] = errors
    return out


def _step_cancel_agent_runs() -> dict[str, Any]:
    manager = _resolve_run_manager()
    if manager is None:
        return {"skipped": "agent run manager unavailable"}
    result = manager.cancel()
    return result if isinstance(result, dict) else {"cancelled": bool(result)}


def _step_stop_coding_run() -> dict[str, Any]:
    run_state = _resolve_coding_run_state()
    if run_state is None:
        return {"skipped": "coding run state unavailable"}
    # stop() returns falsy when nothing was running: a no-op, not a failure.
    return {"stopping": bool(run_state.stop())}


def _step_unload_models() -> dict[str, Any]:
    registry = _resolve_backend_registry()
    if registry is None:
        return {"skipped": "backend registry unavailable"}
    results: dict[str, Any] = {}
    for backend in registry.backends():
        name = getattr(backend, "name", type(backend).__name__)
        unload = getattr(backend, "unload_all", None)
        if not callable(unload):
            results[name] = "unsupported"
            continue
        try:
            results[name] = unload()
        except Exception as exc:  # one backend failing must not skip the next
            results[name] = f"error: {exc}"
    return {"unloaded": results}


def _step_destroy_sandboxes() -> dict[str, Any]:
    manager = _resolve_sandbox_manager()
    if manager is None:
        return {"skipped": "sandbox manager unavailable"}
    destroyed: list[str] = []
    failed: list[str] = []
    for session in manager.list_sessions():
        session_id = session.get("session_id") if isinstance(session, dict) else None
        if not session_id:
            continue
        try:
            ok = manager.destroy_sandbox(session_id)
            (destroyed if ok else failed).append(session_id)
        except Exception:  # one session failing must not skip the next
            failed.append(session_id)
    out: dict[str, Any] = {"destroyed": destroyed}
    if failed:
        out["failed"] = failed
    return out


def _step_stop_veilid_node() -> dict[str, Any]:
    global _veilid_was_running
    node = _resolve_node()
    if node is None:
        with _lock:
            _veilid_was_running = False
        return {"skipped": "veilid node unavailable"}
    was_running = bool(getattr(node, "is_running", False))
    with _lock:
        _veilid_was_running = was_running
    if not was_running:
        return {"was_running": False}
    node.stop()  # never gated; settles STOPPED even when shutdown errors
    return {"was_running": True, "state": "stopped"}


def _step_drop_to_bulbe(user_id: str) -> dict[str, Any]:
    manager = _resolve_mode_manager()
    if manager is None:
        return {"skipped": "security mode manager unavailable"}
    result = manager.escalate_to_bulbe(user_id or "emergency-stop")
    if isinstance(result, dict) and result.get("success") is False:
        raise RuntimeError(result.get("message", "escalation refused"))
    return result if isinstance(result, dict) else {"success": bool(result)}


# ---------------------------------------------------------------------------
# Resume steps
# ---------------------------------------------------------------------------

def _step_reconnect_ollama(warmup_model: str | None) -> dict[str, Any]:
    registry = _resolve_backend_registry()
    if registry is None:
        return {"skipped": "backend registry unavailable"}
    backend = getattr(registry, "active", None)
    if backend is None:
        return {"healthy": False, "detail": "no active backend"}
    out: dict[str, Any] = {"backend": getattr(backend, "name", "")}
    try:
        out["healthy"] = bool(backend.health_check())
    except Exception as exc:  # an unreachable backend is a finding, not a crash
        out["healthy"] = False
        out["detail"] = f"health check failed: {exc}"
    if warmup_model:
        if not out["healthy"]:
            out["warmup"] = "skipped: backend unhealthy"
        else:
            warmer = _resolve_warmup()
            if warmer is None:
                out["warmup"] = "skipped: warmup helper unavailable"
            else:
                result = warmer.warmup(warmup_model)
                out["warmup"] = {
                    "model": warmup_model,
                    "success": bool(getattr(result, "success", result)),
                }
    return out


def _step_restart_veilid_node(was_running: bool) -> dict[str, Any]:
    if not was_running:
        return {"restarted": False, "reason": "node was not running at stop"}
    node = _resolve_node()
    if node is None:
        return {"restarted": False, "reason": "veilid node unavailable"}
    try:
        node.start()
    except Exception as exc:
        if _is_bulbe_refusal(exc):
            # Daily-only restart: the existing fail-secure gate refused.
            return {
                "restarted": False,
                "reason": "bulbe mode: restart refused by the binding-layer gate",
            }
        raise
    return {"restarted": True}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _run_steps(
    plan: list[tuple[str, Callable[[], Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run the ordered steps, fail-tolerant per step; record every outcome."""
    outcomes: list[dict[str, Any]] = []
    failed: list[str] = []
    for name, fn in plan:
        record: dict[str, Any] = {"step": name, "ok": True}
        try:
            detail = fn()
            if detail is not None:
                record["detail"] = detail
        except Exception as exc:
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            failed.append(name)
            logger.warning("emergency stop: step %s failed: %s", name, exc)
        outcomes.append(record)
    return outcomes, failed


def stop(user_id: str = "", drop_to_bulbe: bool = False) -> dict[str, Any]:
    """Engage the emergency stop. Idempotent: re-engaging re-runs the drain."""
    global _stopped, _stopped_since, _stopped_by, _last_stop
    with _lock:
        already_stopped = _stopped
        _stopped = True  # FIRST: admission closes before the drain begins
        _stopped_since = time.time()
        _stopped_by = user_id or ""

    plan: list[tuple[str, Callable[[], Any]]] = [
        ("cancel_generations", _step_cancel_generations),
        ("cancel_agent_runs", _step_cancel_agent_runs),
        ("stop_coding_run", _step_stop_coding_run),
        ("unload_models", _step_unload_models),
        ("destroy_sandboxes", _step_destroy_sandboxes),
        ("stop_veilid_node", _step_stop_veilid_node),
    ]
    if drop_to_bulbe:
        plan.append(("drop_to_bulbe", lambda: _step_drop_to_bulbe(user_id)))

    steps, failed_steps = _run_steps(plan)

    result: dict[str, Any] = {
        "stopped": True,
        "already_stopped": already_stopped,
        "since": _stopped_since,
        "drop_to_bulbe": drop_to_bulbe,
        "steps": steps,
        "failed_steps": failed_steps,
    }
    with _lock:
        _last_stop = result
    _chain(
        action="stop",
        severity="WARNING",
        user_id=user_id,
        drop_to_bulbe=drop_to_bulbe,
        failed_steps=failed_steps,
        steps=[
            {"step": s["step"], "ok": s["ok"]} for s in steps
        ],
    )
    return result


def resume(user_id: str = "", warmup_model: str | None = None) -> dict[str, Any]:
    """Clear the flag and bring the surfaces back. No ceremony; auth at the route."""
    global _stopped, _stopped_since, _stopped_by, _veilid_was_running, _last_resume
    with _lock:
        was_stopped = _stopped
        veilid_was_running = _veilid_was_running
        _stopped = False
        _stopped_since = None
        _stopped_by = ""

    plan: list[tuple[str, Callable[[], Any]]] = [
        ("reconnect_ollama", lambda: _step_reconnect_ollama(warmup_model)),
        ("restart_veilid_node", lambda: _step_restart_veilid_node(veilid_was_running)),
    ]
    steps, failed_steps = _run_steps(plan)

    result: dict[str, Any] = {
        "stopped": False,
        "was_stopped": was_stopped,
        "steps": steps,
        "failed_steps": failed_steps,
    }
    with _lock:
        _last_resume = result
        _veilid_was_running = False
    _chain(
        action="resume",
        severity="INFO",
        user_id=user_id,
        warmup_model=warmup_model or "",
        failed_steps=failed_steps,
        steps=[
            {"step": s["step"], "ok": s["ok"]} for s in steps
        ],
    )
    return result
