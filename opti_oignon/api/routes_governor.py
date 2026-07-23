#!/usr/bin/env python3
"""Resource Governor API surface (Resource Governor cycle Bloc 4).

Wraps the governor's already-assembled status seat behind a thin, guarded
FastAPI surface, the same shape as routes_sync: web-free payload helpers at
module level (importable and testable where fastapi is absent), and a thin
FastAPI wrapper that maps faults to HTTP codes. The contract (spec Section 9):

- ``GET  /api/governor/status``       -> snapshot, provenance, capacity,
                                          learned ceiling, pressure, queue depth,
                                          the external-Ollama advisory
- ``GET  /api/governor/admissions``   -> the bounded recent-decisions ring
- ``POST /api/governor/evict``        -> per-model eviction (honest boolean,
                                          fail-open semantics surfaced)
- ``GET  /api/governor/config``       -> the live config plus the writable and
                                          read-only key sets
- ``POST /api/governor/config``       -> edit allowlisted scalar keys: validate,
                                          persist to the YAML preserving comments,
                                          reload the singleton, audit the change

The governor is mode-free (spec Section 7): every route behaves identically in
Daily and Bulbe. It is a local resource control with no egress, no secrets, and
no state mutation on user content, so there is no Bulbe gate here.

The status surface assembles one governor call per field with no re-derivation.
Reading the pressure state also evaluates and applies the sustained-pressure
keep_alive policy (the design property), so a status read is not entirely
free of effect; that is intentional and documented at the engine.

POST /config write semantics (the one new design question this bloc answered).
The config file is the single source of truth: a write edits the YAML in place
(targeted line replacement, comments and layout preserved -- a full safe_dump
rewrite would strip the spec caveat comments), then drops the module-level
singleton so the next governor resolution reloads the file. In-flight operations
finish on the config they were constructed with; the response says so. Only
scalar runtime-tunable keys are writable: the rlimits backstop is a
process-wide, once-per-process latch (an API write could not honestly take
effect) and the structured keys (ladder, floors, per-caller queue) stay
YAML-edit-only this bloc. The change rides the existing signed audit chain
(the chain_log idiom, action "config_change"), best-effort and off no hot path.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# The governor module is pure Python and always importable; guarded anyway so a
# constrained build degrades to 503 rather than blocking app startup.
try:
    import opti_oignon.resource_governor as _rg

    _GOVERNOR_OK = True
except Exception:  # pragma: no cover - constrained environments only
    _rg = None  # type: ignore[assignment]
    _GOVERNOR_OK = False


# ---------------------------------------------------------------------------
# Config write surface: the writable allowlist and the read-only key map
# ---------------------------------------------------------------------------

# Dotted keys writable over POST /config. Each maps a GovernorConfig field to
# its YAML location; nested keys live under their section header. The rlimits
# family and the structured keys are deliberately absent (see the module
# docstring and READ_ONLY_KEYS below).
WRITABLE_KEYS: dict[str, dict[str, Any]] = {
    "enabled": {"attr": "enabled", "type": "bool"},
    "total_vram_gb": {"attr": "total_vram_gb", "type": "opt_float"},
    "safety_margin_gb": {"attr": "safety_margin_gb", "type": "float"},
    "snapshot_ttl_s": {"attr": "snapshot_ttl_s", "type": "float"},
    "kv_coefficient": {"attr": "kv_coefficient", "type": "float"},
    "ceiling_floor_gb": {"attr": "ceiling_floor_gb", "type": "float"},
    "decisions_ring_size": {"attr": "decisions_ring_size", "type": "int"},
    "idle_evict_threshold_s": {
        "attr": "idle_evict_threshold_s",
        "type": "float",
    },
    "pressure_keep_alive": {"attr": "pressure_keep_alive", "type": "str"},
    "pressure.soft_threshold": {
        "attr": "pressure_soft_threshold",
        "type": "float",
        "section": "pressure",
        "leaf": "soft_threshold",
    },
    "pressure.hard_threshold": {
        "attr": "pressure_hard_threshold",
        "type": "float",
        "section": "pressure",
        "leaf": "hard_threshold",
    },
    "pressure.sustain_s": {
        "attr": "pressure_sustain_s",
        "type": "float",
        "section": "pressure",
        "leaf": "sustain_s",
    },
    "pressure.refusal_window_s": {
        "attr": "pressure_refusal_window_s",
        "type": "float",
        "section": "pressure",
        "leaf": "refusal_window_s",
    },
    "queue.depth": {
        "attr": "queue_depth",
        "type": "int",
        "section": "queue",
        "leaf": "depth",
    },
    "queue.wait_s": {
        "attr": "queue_wait_s",
        "type": "float",
        "section": "queue",
        "leaf": "wait_s",
    },
    "ollama_limits.max_loaded_models": {
        "attr": "ollama_max_loaded_models",
        "type": "opt_int",
        "section": "ollama_limits",
        "leaf": "max_loaded_models",
    },
    "ollama_limits.num_parallel": {
        "attr": "ollama_num_parallel",
        "type": "opt_int",
        "section": "ollama_limits",
        "leaf": "num_parallel",
    },
    "ollama_limits.max_queue": {
        "attr": "ollama_max_queue",
        "type": "opt_int",
        "section": "ollama_limits",
        "leaf": "max_queue",
    },
    "ollama_limits.spawn_applies": {
        "attr": "ollama_spawn_applies",
        "type": "bool",
        "section": "ollama_limits",
        "leaf": "spawn_applies",
    },
    "ollama_limits.external_advisory": {
        "attr": "ollama_external_advisory",
        "type": "bool",
        "section": "ollama_limits",
        "leaf": "external_advisory",
    },
}

# Keys deliberately not writable over the API, with the honest reason.
READ_ONLY_KEYS: dict[str, str] = {
    "rlimits.enabled": (
        "process-wide setrlimit backstop, latched once per process;"
        " an API write could not honestly take effect -- edit the YAML"
        " and restart"
    ),
    "rlimits.as_gb": "see rlimits.enabled",
    "rlimits.data_gb": "see rlimits.enabled",
    "ctx_ladder": "structured key; edit the YAML directly this release",
    "ctx_floor": "structured key; edit the YAML directly this release",
    "queue.enabled_per_caller": (
        "structured key; edit the YAML directly this release"
    ),
}


# ---------------------------------------------------------------------------
# Web-free payload helpers (importable without fastapi)
# ---------------------------------------------------------------------------


def status_payload(governor: Any) -> dict[str, Any]:
    """The /status body: one governor call per field, no re-derivation.

    Reading ``pressure_state`` applies the sustained-pressure
    keep_alive policy by design; the snapshot already folds the learned
    ceiling into ``capacity_gb`` when lower, so the separate
    ``learned_ceiling_gb`` is the raw learned value for transparency.
    """
    snapshot = governor.get_snapshot_fast()
    try:
        learned_ceiling = governor.store.get_learned_ceiling()
    except Exception:  # pragma: no cover - ceiling read is defensive
        learned_ceiling = None
    return {
        "enabled": bool(governor.config.enabled),
        "snapshot": snapshot.to_dict(),
        "learned_ceiling_gb": learned_ceiling,
        "pressure": governor.pressure_state(),
        "queue_depth": governor.queue_depth,
        "ollama_limits": governor.ollama_limits_advisory(),
    }


def admissions_payload(governor: Any, limit: int = 20) -> dict[str, Any]:
    """The /admissions body: the recent-decisions ring, bounded.

    ``limit`` is clamped to ``[1, decisions_ring_size]`` -- the ring's own
    bound caps paging; there is no token, the ring is the window.
    """
    ring_size = max(1, int(governor.config.decisions_ring_size))
    clamped = max(1, min(int(limit), ring_size))
    decisions = governor.store.recent_decisions(clamped)
    return {
        "admissions": decisions,
        "count": len(decisions),
        "limit": clamped,
        "ring_size": ring_size,
    }


def evict_payload(governor: Any, model: str) -> dict[str, Any]:
    """The /evict body: the public evict_model, fail-open semantics surfaced.

    A False return is not an error: the model was not loaded, no backend
    exposed the unload primitive, or the unload failed -- Ollama's own LRU
    carries the pressure (spec Section 12). The eviction is audit-chained
    by evict_model itself (the async append) with trigger "api".
    """
    evicted = bool(governor.evict_model(model, trigger="api"))
    note = (
        "evicted through the backend unload primitive; the snapshot was"
        " invalidated"
        if evicted
        else "not evicted (model not loaded, no unload primitive, or the"
        " unload failed); Ollama's own LRU carries the pressure"
    )
    return {"evicted": evicted, "model": model, "note": note}


def _config_to_nested(cfg: Any) -> dict[str, Any]:
    """Mirror GovernorConfig back to the YAML section shape (read-only view)."""
    return {
        "enabled": cfg.enabled,
        "total_vram_gb": cfg.total_vram_gb,
        "safety_margin_gb": cfg.safety_margin_gb,
        "snapshot_ttl_s": cfg.snapshot_ttl_s,
        "kv_coefficient": cfg.kv_coefficient,
        "ceiling_floor_gb": cfg.ceiling_floor_gb,
        "decisions_ring_size": cfg.decisions_ring_size,
        "ctx_ladder": list(cfg.ctx_ladder),
        "ctx_floor": dict(cfg.ctx_floor),
        "idle_evict_threshold_s": cfg.idle_evict_threshold_s,
        "pressure": {
            "soft_threshold": cfg.pressure_soft_threshold,
            "hard_threshold": cfg.pressure_hard_threshold,
            "sustain_s": cfg.pressure_sustain_s,
            "refusal_window_s": cfg.pressure_refusal_window_s,
        },
        "pressure_keep_alive": cfg.pressure_keep_alive,
        "queue": {
            "enabled_per_caller": dict(cfg.queue_enabled_per_caller),
            "depth": cfg.queue_depth,
            "wait_s": cfg.queue_wait_s,
        },
        "rlimits": {
            "enabled": cfg.rlimits_enabled,
            "as_gb": cfg.rlimits_as_gb,
            "data_gb": cfg.rlimits_data_gb,
        },
        "ollama_limits": {
            "max_loaded_models": cfg.ollama_max_loaded_models,
            "num_parallel": cfg.ollama_num_parallel,
            "max_queue": cfg.ollama_max_queue,
            "spawn_applies": cfg.ollama_spawn_applies,
            "external_advisory": cfg.ollama_external_advisory,
        },
    }


def config_read_payload(governor: Any) -> dict[str, Any]:
    """The GET /config body: the live config plus the writable/read-only sets."""
    return {
        "config": _config_to_nested(governor.config),
        "writable_keys": sorted(WRITABLE_KEYS.keys()),
        "read_only_keys": dict(READ_ONLY_KEYS),
    }


class ConfigWriteError(Exception):
    """A config-write rejection carrying the HTTP status to map (400/409)."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _coerce_value(spec_type: str, raw: Any, key: str) -> Any:
    """Strict typing for a write value (write what load_config reads back)."""
    if spec_type in ("opt_float", "opt_int") and raw is None:
        return None
    if spec_type == "bool":
        if not isinstance(raw, bool):
            raise ConfigWriteError(400, f"{key} expects a boolean")
        return raw
    if spec_type in ("int", "opt_int"):
        # bool is a subclass of int; reject it explicitly for numeric keys.
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ConfigWriteError(400, f"{key} expects an integer")
        return raw
    if spec_type in ("float", "opt_float"):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ConfigWriteError(400, f"{key} expects a number")
        return float(raw)
    if spec_type == "str":
        if not isinstance(raw, str) or not raw:
            raise ConfigWriteError(400, f"{key} expects a non-empty string")
        return raw
    raise ConfigWriteError(400, f"{key} has an unsupported type")  # pragma: no cover


def _yaml_scalar(value: Any) -> str:
    """Render a scalar the way the shipped YAML writes it."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)  # double-quoted, YAML-valid (e.g. "5m")
    return str(value)


def _set_yaml_scalar(text: str, key: str, value: Any) -> str:
    """Replace one scalar's value in the YAML, preserving comments and layout.

    Top-level keys match ``^key:``; a section.leaf key matches the leaf line
    inside its section block (up to the next top-level key). Raises
    ConfigWriteError(409) when the target line is absent (a hand-edited file).
    """
    spec = WRITABLE_KEYS[key]
    rendered = _yaml_scalar(value)
    lines = text.splitlines(keepends=True)

    section = spec.get("section")
    leaf = spec.get("leaf")
    if section is None:
        pattern = re.compile(r"^(" + re.escape(key) + r"):[^\n]*")
        for i, line in enumerate(lines):
            if pattern.match(line):
                newline = "\n" if line.endswith("\n") else ""
                lines[i] = f"{key}: {rendered}{newline}"
                return "".join(lines)
        raise ConfigWriteError(
            409, f"{key} line not found in the config file; edit the YAML directly"
        )

    # Nested: find the section header, then the leaf before the next top-level
    # key (a non-indented, non-comment, non-blank line).
    header = re.compile(r"^" + re.escape(section) + r":\s*(#.*)?$")
    leaf_pat = re.compile(r"^(\s+)(" + re.escape(leaf) + r"):[^\n]*")
    in_section = False
    for i, line in enumerate(lines):
        if not in_section:
            if header.match(line):
                in_section = True
            continue
        if re.match(r"^\S", line) and not line.startswith("#"):
            break  # left the section without finding the leaf
        m = leaf_pat.match(line)
        if m:
            indent = m.group(1)
            newline = "\n" if line.endswith("\n") else ""
            lines[i] = f"{indent}{leaf}: {rendered}{newline}"
            return "".join(lines)
    raise ConfigWriteError(
        409,
        f"{key} line not found under '{section}:' in the config file;"
        " edit the YAML directly",
    )


def _default_audit(changes: dict[str, dict[str, Any]]) -> None:
    """Best-effort signed-chain append for a config change (the chain_log idiom)."""
    try:
        from opti_oignon.signed_audit_log import chain_log

        chain_log(
            event_type="resource_governor",
            source="api.routes_governor",
            action="config_change",
            severity="INFO",
            changes=changes,
        )
    except Exception as exc:  # pragma: no cover - audit is best-effort
        logger.debug("Config-change audit append failed: %s", exc)


def config_write_payload(
    current_config: Any,
    changes: dict[str, Any],
    config_path: Any,
    reset_fn: Callable[[], None],
    audit_fn: Callable[[dict[str, dict[str, Any]]], None] = _default_audit,
) -> dict[str, Any]:
    """Validate, persist, reload and audit an allowlisted config write.

    All-or-nothing: a single bad key rejects the whole write before any
    file change. After a successful persist the singleton is dropped so the
    next governor resolution reloads the file (the file stays the single
    source of truth); in-flight operations finish on the config they hold.
    Raises ConfigWriteError (mapped to 400/409 at the route).
    """
    if not isinstance(changes, dict) or not changes:
        raise ConfigWriteError(400, "a non-empty object of keys is required")

    typed: dict[str, Any] = {}
    for key, raw in changes.items():
        if key in READ_ONLY_KEYS:
            raise ConfigWriteError(
                400, f"{key} is read-only ({READ_ONLY_KEYS[key]})"
            )
        if key not in WRITABLE_KEYS:
            raise ConfigWriteError(400, f"unknown or unwritable key: {key}")
        typed[key] = _coerce_value(WRITABLE_KEYS[key]["type"], raw, key)

    from pathlib import Path

    p = Path(config_path)
    try:
        text = p.read_text(encoding="utf-8") if p.is_file() else ""
    except Exception as exc:
        raise ConfigWriteError(500, f"could not read the config file: {exc}")
    if not text:
        raise ConfigWriteError(
            409, "config file not found; edit or create the YAML directly"
        )

    applied: dict[str, dict[str, Any]] = {}
    new_text = text
    for key, value in typed.items():
        old = getattr(current_config, WRITABLE_KEYS[key]["attr"])
        new_text = _set_yaml_scalar(new_text, key, value)  # may raise 409
        applied[key] = {"old": old, "new": value}

    try:
        p.write_text(new_text, encoding="utf-8")
    except Exception as exc:
        raise ConfigWriteError(500, f"could not write the config file: {exc}")

    try:
        reset_fn()
    except Exception as exc:  # pragma: no cover - reset is defensive
        logger.debug("Governor singleton reset failed after config write: %s", exc)

    audit_fn(applied)

    return {
        "applied": applied,
        "persisted": True,
        "effective": (
            "next governor access; in-flight operations finish on the"
            " previous config"
        ),
        "notes": [
            "the startup security checklist view is cached for the process"
            " lifetime and refreshes at restart",
            "ollama_limits are read by a future spawner at spawn time and by"
            " the external-server advisory on each status read",
        ],
    }


# ---------------------------------------------------------------------------
# FastAPI surface (guarded; thin wrappers over the helpers)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, Body, Depends, HTTPException, Query

    # Auth parity (the SYN-06 idiom): require authentication on every endpoint,
    # the same per-router dependency routes_sync / routes_security carry. The
    # global deny-by-default AuthMiddleware already covers /api/governor; this
    # is parity, not a gap closure.
    try:
        from .routes_auth import _get_current_user

        _auth_dep = [Depends(_get_current_user)]
    except ImportError:
        _auth_dep = []

    router = APIRouter(
        prefix="/api/governor", tags=["governor"], dependencies=_auth_dep
    )

    def _governor() -> Any:
        if not _GOVERNOR_OK or _rg is None:
            raise HTTPException(
                status_code=503, detail="Resource governor not available"
            )
        try:
            return _rg.get_resource_governor()
        except Exception:  # pragma: no cover - resolution is defensive
            logger.exception("governor resolution failed")
            raise HTTPException(
                status_code=503, detail="Resource governor not available"
            )

    @router.get("/status")
    def governor_status() -> dict[str, Any]:
        """Capacity, provenance, learned ceiling, pressure, queue, advisory."""
        governor = _governor()
        try:
            return status_payload(governor)
        except Exception:  # pragma: no cover - status read is defensive
            logger.exception("governor status failed")
            raise HTTPException(
                status_code=500, detail="Failed to read governor status"
            )

    @router.get("/admissions")
    def governor_admissions(
        limit: int = Query(20, ge=1),
    ) -> dict[str, Any]:
        """The bounded recent-decisions ring (clamped to the ring size)."""
        governor = _governor()
        try:
            return admissions_payload(governor, limit)
        except Exception:  # pragma: no cover - ring read is defensive
            logger.exception("governor admissions failed")
            raise HTTPException(
                status_code=500, detail="Failed to read admissions"
            )

    @router.post("/evict")
    def governor_evict(body: Any = Body(default=None)) -> dict[str, Any]:
        """Evict one model (honest boolean; fail-open semantics surfaced)."""
        governor = _governor()
        data = body if isinstance(body, dict) else {}
        model = str(data.get("model", "") or "")
        if not model:
            raise HTTPException(status_code=400, detail="model is required")
        try:
            return evict_payload(governor, model)
        except Exception:  # pragma: no cover - evict is fail-open at the engine
            logger.exception("governor evict failed")
            raise HTTPException(status_code=500, detail="Failed to evict model")

    @router.get("/config")
    def governor_config_read() -> dict[str, Any]:
        """The live config plus the writable and read-only key sets."""
        governor = _governor()
        try:
            return config_read_payload(governor)
        except Exception:  # pragma: no cover - read is defensive
            logger.exception("governor config read failed")
            raise HTTPException(
                status_code=500, detail="Failed to read governor config"
            )

    @router.post("/config")
    def governor_config_write(body: Any = Body(default=None)) -> dict[str, Any]:
        """Edit allowlisted scalar keys: validate, persist, reload, audit."""
        governor = _governor()
        changes = body if isinstance(body, dict) else None
        try:
            return config_write_payload(
                governor.config,
                changes if changes is not None else {},
                _rg._DEFAULT_CONFIG_PATH,
                _rg.reset_resource_governor,
            )
        except ConfigWriteError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)
        except Exception:  # pragma: no cover - write wrapper is defensive
            logger.exception("governor config write failed")
            raise HTTPException(
                status_code=500, detail="Failed to write governor config"
            )

except Exception:  # pragma: no cover - FastAPI absent (e.g. isolated tests)
    router = None  # type: ignore[assignment]
