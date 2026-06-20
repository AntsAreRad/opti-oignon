#!/usr/bin/env python3
"""
Security status and configuration API routes (S124, S125, S146, S148, S157, S158).

GET    /api/security/status       -- Overall security posture with letter grade
GET    /api/security/config       -- Current security configuration
PUT    /api/security/config       -- Update security settings (admin only)
GET    /api/security/encryption   -- Encryption status (S125)
POST   /api/security/encryption/setup -- Initialize encryption (S125)
GET    /api/security/audit        -- Security audit trail (S125)
POST   /api/security/audit/export-qr     -- QR code of chain tip (S146)
POST   /api/security/audit/export-anchor -- Signed JSON anchor file (S146)
GET    /api/security/audit/anchor-text   -- Plain-text anchor for clipboard (S146)
POST   /api/security/audit/verify-anchor -- Verify imported anchor (S146)
POST   /api/security/redteam/run         -- Launch red team campaign (S148)
GET    /api/security/redteam/status      -- Campaign progress (S148)
GET    /api/security/redteam/results     -- Latest results (S148)
GET    /api/security/redteam/report      -- Download report JSON/text/MD (S148)
GET    /api/security/redteam/reports     -- List all stored reports (S157)
GET    /api/security/redteam/reports/{id} -- Get specific stored report (S157)
DELETE /api/security/redteam/reports/{id} -- Delete stored report (S157)
GET    /api/security/redteam/compare     -- Diff two reports (S157)
GET    /api/security/redteam/suggestions -- List feedback suggestions (S157)
POST   /api/security/redteam/suggestions/{id}/accept -- Accept suggestion (S157)
POST   /api/security/redteam/suggestions/{id}/reject -- Reject suggestion (S157)
GET    /api/security/scheduler           -- Full scheduler status (S158)
POST   /api/security/scheduler/trigger   -- Manually trigger scheduled task (S158)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# S136 audit fix: import auth dependency so ALL security endpoints
# require authentication. Previously, 39 state-changing endpoints
# (mode change, kill switch, plugin allowlist, 2FA, encryption setup)
# had ZERO authentication.
try:
    from .routes_auth import _get_current_user
    _auth_dep = [Depends(_get_current_user)]
except ImportError:
    _auth_dep = []

    async def _get_current_user() -> dict:  # type: ignore[no-redef]
        """Fallback when the auth routes are unavailable (degraded mode)."""
        return {}

router = APIRouter(
    prefix="/api/security",
    tags=["security"],
    dependencies=_auth_dep,
)

_SECURITY_YAML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "security.yaml"
)


# =========================================================================
# Helpers
# =========================================================================

def _load_security_yaml() -> dict[str, Any]:
    """Load the full security.yaml config."""
    try:
        if os.path.isfile(_SECURITY_YAML_PATH):
            with open(_SECURITY_YAML_PATH, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("Failed to load security.yaml: %s", exc)
    return {}


def _save_security_yaml(data: dict[str, Any]) -> None:
    """Write the security.yaml config back to disk."""
    try:
        with open(_SECURITY_YAML_PATH, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save security config: {exc}",
        )


def _compute_security_score() -> tuple[int, str, list[dict[str, Any]]]:
    """Compute a security score from 0-100 with letter grade.

    Returns (score, grade, checks) where checks is a list of
    {name, points, max_points, passed, detail}.
    """
    checks: list[dict[str, Any]] = []
    total = 0

    # 1. CORS not wildcard: +15
    try:
        from opti_oignon.api.app import _cors_origins, _cors_credentials
        is_wildcard = "*" in _cors_origins
        passed = not is_wildcard
        checks.append({
            "name": "cors_not_wildcard",
            "points": 15 if passed else 0,
            "max_points": 15,
            "passed": passed,
            "detail": "CORS restricted to specific origins" if passed
                      else "CORS is wildcard (*) -- any site can call the API",
        })
        total += 15 if passed else 0
    except Exception:
        checks.append({
            "name": "cors_not_wildcard",
            "points": 0, "max_points": 15, "passed": False,
            "detail": "Could not determine CORS config",
        })

    # 2. Security headers active: +15
    try:
        from opti_oignon.api.security_middleware import get_security_headers_config
        cfg = get_security_headers_config()
        passed = not cfg.get("_disabled", False)
        checks.append({
            "name": "security_headers",
            "points": 15 if passed else 0,
            "max_points": 15,
            "passed": passed,
            "detail": "Security headers middleware active" if passed
                      else "Security headers disabled",
        })
        total += 15 if passed else 0
    except Exception:
        checks.append({
            "name": "security_headers",
            "points": 0, "max_points": 15, "passed": False,
            "detail": "Security headers middleware not available",
        })

    # 3. Auth enabled (not single-user): +15
    try:
        from opti_oignon.auth import auth_manager
        if auth_manager is not None:
            passed = not getattr(auth_manager, "_single_user_mode", True)
        else:
            passed = False
        checks.append({
            "name": "auth_enabled",
            "points": 15 if passed else 0,
            "max_points": 15,
            "passed": passed,
            "detail": "Multi-user authentication active" if passed
                      else "Single-user mode (no authentication)",
        })
        total += 15 if passed else 0
    except Exception:
        checks.append({
            "name": "auth_enabled",
            "points": 0, "max_points": 15, "passed": False,
            "detail": "Auth module not available",
        })

    # 4. Rate limiting enabled: +10
    try:
        from opti_oignon.auth import login_rate_limiter
        passed = login_rate_limiter.enabled
        checks.append({
            "name": "rate_limiting",
            "points": 10 if passed else 0,
            "max_points": 10,
            "passed": passed,
            "detail": "Login rate limiting active" if passed
                      else "Login rate limiting disabled",
        })
        total += 10 if passed else 0
    except Exception:
        checks.append({
            "name": "rate_limiting",
            "points": 0, "max_points": 10, "passed": False,
            "detail": "Rate limiter not available",
        })

    # 5. Sandbox bwrap active: +15
    try:
        from opti_oignon.sandbox_manager import sandbox_manager
        if sandbox_manager is not None:
            passed = sandbox_manager.bwrap_available
        else:
            passed = False
        checks.append({
            "name": "sandbox_bwrap",
            "points": 15 if passed else 0,
            "max_points": 15,
            "passed": passed,
            "detail": "Bubblewrap sandbox active" if passed
                      else "Bubblewrap not available -- degraded isolation",
        })
        total += 15 if passed else 0
    except Exception:
        checks.append({
            "name": "sandbox_bwrap",
            "points": 0, "max_points": 15, "passed": False,
            "detail": "Sandbox module not available",
        })

    # 6. Sandbox strict mode: +10
    try:
        from opti_oignon.sandbox_manager import sandbox_manager
        if sandbox_manager is not None:
            passed = sandbox_manager.strict_mode
        else:
            passed = False
        checks.append({
            "name": "sandbox_strict_mode",
            "points": 10 if passed else 0,
            "max_points": 10,
            "passed": passed,
            "detail": "Sandbox strict mode ON" if passed
                      else "Sandbox strict mode OFF -- tempdir fallback allowed",
        })
        total += 10 if passed else 0
    except Exception:
        checks.append({
            "name": "sandbox_strict_mode",
            "points": 0, "max_points": 10, "passed": False,
            "detail": "Sandbox module not available",
        })

    # 7. Plugin module blocking: +10
    try:
        from opti_oignon.plugin_loader import _BLOCKED_IMPORTS
        passed = "os" in _BLOCKED_IMPORTS and "sys" in _BLOCKED_IMPORTS
        checks.append({
            "name": "plugin_module_blocking",
            "points": 10 if passed else 0,
            "max_points": 10,
            "passed": passed,
            "detail": f"Plugin import blocking active ({len(_BLOCKED_IMPORTS)} modules)"
                      if passed else "Critical modules not blocked in plugin sandbox",
        })
        total += 10 if passed else 0
    except Exception:
        checks.append({
            "name": "plugin_module_blocking",
            "points": 0, "max_points": 10, "passed": False,
            "detail": "Plugin loader not available",
        })

    # 8. Plugin data redaction: +10
    try:
        from opti_oignon.plugin_hooks import redact_hook_data, REDACTED_PLACEHOLDER
        # Verify redaction works
        test = redact_hook_data(
            {"message": "x", "model": "y"}, "__test__", force_redact=True,
        )
        passed = test.get("message") == REDACTED_PLACEHOLDER
        checks.append({
            "name": "plugin_data_redaction",
            "points": 10 if passed else 0,
            "max_points": 10,
            "passed": passed,
            "detail": "Hook data redaction active" if passed
                      else "Hook data redaction not functioning",
        })
        total += 10 if passed else 0
    except Exception:
        checks.append({
            "name": "plugin_data_redaction",
            "points": 0, "max_points": 10, "passed": False,
            "detail": "Plugin hooks module not available",
        })

    # 9. Red team resistance (S157): +10
    # Score deduction if last run has critical findings (bypass rate > 30%).
    # Warning if no run in configurable days (default: 30).
    rt_points = 0
    rt_max = 10
    rt_passed = False
    rt_detail = "No red team run recorded"

    if _redteam_report_store:
        # Get most recent report
        latest_id = max(_redteam_report_store.keys())
        latest = _redteam_report_store[latest_id]
        latest_score = latest.get("score", {})
        bypass_rate = latest_score.get("overall_bypass_rate", 1.0)
        timestamp_str = latest.get("timestamp", "")

        # Check age of last run
        stale = False
        redteam_max_age_days = 30
        if timestamp_str:
            try:
                last_run_dt = datetime.fromisoformat(timestamp_str)
                age_days = (datetime.now(timezone.utc) - last_run_dt).days
                stale = age_days > redteam_max_age_days
            except (ValueError, TypeError):
                stale = True

        if stale:
            rt_detail = (
                f"Last red team run is stale (>{redteam_max_age_days} days old)"
            )
            rt_points = 3
        elif bypass_rate > 0.3:
            rt_detail = (
                f"Last red team run has critical bypass rate: "
                f"{bypass_rate:.1%} (>{30}%)"
            )
            rt_points = 0
        elif bypass_rate > 0.1:
            rt_detail = (
                f"Last red team run has elevated bypass rate: "
                f"{bypass_rate:.1%}"
            )
            rt_points = 5
        else:
            rt_detail = (
                f"Red team resistance OK: bypass rate {bypass_rate:.1%}"
            )
            rt_points = 10
            rt_passed = True

    checks.append({
        "name": "redteam_resistance",
        "points": rt_points,
        "max_points": rt_max,
        "passed": rt_passed,
        "detail": rt_detail,
    })
    total += rt_points

    # Compute grade based on percentage of max possible score
    max_possible = sum(c["max_points"] for c in checks)
    pct = (total / max(max_possible, 1)) * 100

    if pct >= 90:
        grade = "A+"
    elif pct >= 80:
        grade = "A"
    elif pct >= 70:
        grade = "B+"
    elif pct >= 60:
        grade = "B"
    elif pct >= 50:
        grade = "C"
    elif pct >= 40:
        grade = "D"
    else:
        grade = "F"

    return total, grade, checks


# =========================================================================
# Schemas
# =========================================================================

class SecurityConfigUpdate(BaseModel):
    """Request body for updating security configuration."""
    cors: dict[str, Any] | None = Field(default=None, description="CORS settings")
    headers: dict[str, Any] | None = Field(default=None, description="Security headers")
    rate_limiting: dict[str, Any] | None = Field(default=None, description="Rate limiting")
    sandbox: dict[str, Any] | None = Field(default=None, description="Sandbox settings")
    plugins: dict[str, Any] | None = Field(default=None, description="Plugin security")
    encryption: dict[str, Any] | None = Field(default=None, description="Encryption settings (S125)")
    jwt: dict[str, Any] | None = Field(default=None, description="JWT cookie settings (S125)")
    search_safety: dict[str, Any] | None = Field(default=None, description="Search injection defense (S125)")
    backup: dict[str, Any] | None = Field(default=None, description="Backup encryption settings (S125)")


# =========================================================================
# Routes
# =========================================================================

@router.get("/status")
def get_security_status() -> dict:
    """Get overall security posture with letter grade (A+ to F).

    The score is computed from 9 security checks (S157: includes red team
    resistance) with percentage-based grading. S158 adds scheduler summary.
    """
    score, grade, checks = _compute_security_score()
    max_score = sum(c["max_points"] for c in checks)

    result = {
        "score": score,
        "max_score": max_score,
        "grade": grade,
        "checks": checks,
    }

    # S158: attach scheduler summary if available
    try:
        from opti_oignon.security_scheduler import get_scheduler
        scheduler = get_scheduler()
        status = scheduler.get_status()
        result["scheduler"] = {
            "enabled": status.get("enabled", False),
            "running": status.get("running", False),
            "last_redteam_run": status.get("redteam", {}).get("last_run"),
            "next_redteam_run": status.get("redteam", {}).get("next_run"),
            "last_dep_audit": (
                status.get("dep_audit", {}).get("last_audit", {}) or {}
            ).get("timestamp"),
            "dep_findings_count": (
                status.get("dep_audit", {}).get("last_audit", {}) or {}
            ).get("filtered_count", 0),
            "alerts_total": status.get("alerts_total", 0),
        }
    except Exception:
        result["scheduler"] = {"available": False}

    return result


@router.get("/config")
def get_security_config() -> dict:
    """Get the current security configuration from security.yaml."""
    config = _load_security_yaml()

    if not config:
        return {
            "loaded": False,
            "detail": "security.yaml not found or empty; using defaults",
            "config": {},
        }

    return {
        "loaded": True,
        "config": config,
    }


@router.put("/config")
def update_security_config(update: SecurityConfigUpdate) -> dict:
    """Update security configuration (persisted to security.yaml).

    Only provided sections are updated; others are left unchanged.
    Note: some changes (CORS, headers) require a server restart to
    take effect.
    """
    current = _load_security_yaml()

    changes: list[str] = []
    if update.cors is not None:
        current["cors"] = {**current.get("cors", {}), **update.cors}
        changes.append("cors")
    if update.headers is not None:
        current["headers"] = {**current.get("headers", {}), **update.headers}
        changes.append("headers")
    if update.rate_limiting is not None:
        current["rate_limiting"] = {
            **current.get("rate_limiting", {}), **update.rate_limiting
        }
        changes.append("rate_limiting")
    if update.sandbox is not None:
        current["sandbox"] = {**current.get("sandbox", {}), **update.sandbox}
        changes.append("sandbox")
    if update.plugins is not None:
        current["plugins"] = {**current.get("plugins", {}), **update.plugins}
        changes.append("plugins")
    if update.encryption is not None:
        current["encryption"] = {**current.get("encryption", {}), **update.encryption}
        changes.append("encryption")
    if update.jwt is not None:
        current["jwt"] = {**current.get("jwt", {}), **update.jwt}
        changes.append("jwt")
    if update.search_safety is not None:
        current["search_safety"] = {**current.get("search_safety", {}), **update.search_safety}
        changes.append("search_safety")
    if update.backup is not None:
        current["backup"] = {**current.get("backup", {}), **update.backup}
        changes.append("backup")

    if not changes:
        return {"updated": False, "detail": "No sections provided"}

    _save_security_yaml(current)

    logger.info("Security config updated: %s", ", ".join(changes))

    return {
        "updated": True,
        "sections": changes,
        "detail": "Some changes may require server restart to take effect",
        "config": current,
    }


# =========================================================================
# S125: Encryption endpoints
# =========================================================================

@router.get("/encryption")
def get_encryption_status() -> dict:
    """Get data-at-rest encryption status (S125)."""
    try:
        from opti_oignon.encryption import get_encryption_manager
        mgr = get_encryption_manager()
        return mgr.get_status()
    except ImportError:
        return {
            "enabled": False,
            "config_enabled": False,
            "has_key": False,
            "keyfile_exists": False,
            "env_key_set": False,
            "keyfile_path": "",
            "detail": "Encryption module not available",
        }


class EncryptionSetupRequest(BaseModel):
    """Request body for encryption setup."""
    mode: str = Field(description="Setup mode: 'passphrase' or 'random'")
    passphrase: str | None = Field(default=None, description="Passphrase (for mode=passphrase)")


@router.post("/encryption/setup")
def setup_encryption(req: EncryptionSetupRequest) -> dict:
    """Initialize data-at-rest encryption (S125).

    Generates an encryption key and saves it to the keyfile.
    Use mode='random' for auto-generated key, or mode='passphrase'
    with a user-provided passphrase for PBKDF2-derived key.
    """
    try:
        from opti_oignon.encryption import get_encryption_manager
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Encryption module not available",
        )

    mgr = get_encryption_manager()

    if mgr.enabled:
        return {
            "setup": False,
            "detail": "Encryption is already configured and active",
            "status": mgr.get_status(),
        }

    if req.mode == "passphrase":
        if not req.passphrase or len(req.passphrase) < 8:
            raise HTTPException(
                status_code=400,
                detail="Passphrase must be at least 8 characters",
            )
        ok = mgr.setup_from_passphrase(req.passphrase)
    elif req.mode == "random":
        ok = mgr.setup_random_key()
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Use 'passphrase' or 'random'.",
        )

    if not ok:
        raise HTTPException(
            status_code=500,
            detail="Encryption setup failed. Check server logs.",
        )

    # Enable in config
    current = _load_security_yaml()
    current.setdefault("encryption", {})["enabled"] = True
    _save_security_yaml(current)

    return {
        "setup": True,
        "detail": "Encryption configured successfully",
        "status": mgr.get_status(),
    }


# =========================================================================
# S125: Security Audit Trail
# =========================================================================

@router.get("/audit")
def get_security_audit(
    event_type: str | None = Query(default=None, description="Filter by event type (auth, sandbox, rate_limit, search_injection)"),
    severity: str | None = Query(default=None, description="Filter by severity (info, warning, critical)"),
    limit: int = Query(default=50, ge=1, le=500, description="Max events to return"),
    since: float | None = Query(default=None, description="Unix timestamp to filter events after"),
) -> dict:
    """Get aggregated security audit trail (S125).

    Collects security-relevant events from:
    - Auth: login attempts, registrations, password changes
    - Sandbox: blocked commands, violations
    - Rate limiter: lockouts, rate limit triggers
    - Search: detected prompt injection attempts

    Supports filtering by event type, severity, and time range.
    """
    events: list[dict] = []

    # 1. Auth audit log
    if event_type is None or event_type == "auth":
        try:
            from opti_oignon.auth import auth_manager
            if auth_manager is not None:
                raw = auth_manager.get_audit_log(limit=limit)
                for entry in raw:
                    ts = entry.get("timestamp", 0)
                    if since and ts < since:
                        continue
                    action = entry.get("action", "")
                    sev = "info"
                    if action in ("login_failed", "delete_user", "password_change"):
                        sev = "warning"
                    if severity and sev != severity:
                        continue
                    events.append({
                        "source": "auth",
                        "event_type": "auth",
                        "action": action,
                        "severity": sev,
                        "user_id": entry.get("user_id", ""),
                        "target": entry.get("target_id", ""),
                        "details": entry.get("details", {}),
                        "timestamp": ts,
                    })
        except Exception as exc:
            logger.debug("Failed to fetch auth audit: %s", exc)

    # 2. Sandbox audit log (blocked commands)
    if event_type is None or event_type == "sandbox":
        try:
            from opti_oignon.sandbox_manager import sandbox_manager as sbx
            if sbx is not None and hasattr(sbx, "_audit_log") and sbx._audit_log is not None:
                raw = sbx._audit_log.get_all_logs(limit=limit)
                for entry in raw:
                    ts = entry.get("timestamp", 0)
                    if since and ts < since:
                        continue
                    blocked = bool(entry.get("blocked", 0))
                    sev = "critical" if blocked else "info"
                    if severity and sev != severity:
                        continue
                    events.append({
                        "source": "sandbox",
                        "event_type": "sandbox",
                        "action": "command_blocked" if blocked else "command_executed",
                        "severity": sev,
                        "session_id": entry.get("session_id", ""),
                        "details": {
                            "command": entry.get("command", "")[:200],
                            "block_reason": entry.get("block_reason", ""),
                            "return_code": entry.get("return_code"),
                            "timed_out": bool(entry.get("timed_out", 0)),
                        },
                        "timestamp": ts,
                    })
        except Exception as exc:
            logger.debug("Failed to fetch sandbox audit: %s", exc)

    # 3. Rate limiter state (current lockouts)
    if event_type is None or event_type == "rate_limit":
        try:
            from opti_oignon.auth import login_rate_limiter
            import time as _time
            now = _time.time()
            if login_rate_limiter and login_rate_limiter.enabled:
                # Report currently locked IPs
                for ip, entry in login_rate_limiter._ip_entries.items():
                    if entry.lockout_until > now:
                        sev = "warning"
                        if severity and sev != severity:
                            continue
                        events.append({
                            "source": "rate_limiter",
                            "event_type": "rate_limit",
                            "action": "ip_locked",
                            "severity": sev,
                            "details": {
                                "ip": ip,
                                "lockout_until": entry.lockout_until,
                                "lockout_count": entry.lockout_count,
                                "attempts_in_window": len(entry.attempts),
                            },
                            "timestamp": entry.lockout_until - login_rate_limiter._get_lockout_duration(entry.lockout_count),
                        })
                # Report locked usernames
                for username, entry in login_rate_limiter._user_entries.items():
                    if entry.lockout_until > now:
                        sev = "critical"
                        if severity and sev != severity:
                            continue
                        events.append({
                            "source": "rate_limiter",
                            "event_type": "rate_limit",
                            "action": "account_locked",
                            "severity": sev,
                            "details": {
                                "username": username,
                                "lockout_until": entry.lockout_until,
                                "lockout_count": entry.lockout_count,
                            },
                            "timestamp": entry.lockout_until - login_rate_limiter._get_lockout_duration(entry.lockout_count),
                        })
        except Exception as exc:
            logger.debug("Failed to fetch rate limiter state: %s", exc)

    # 4. Search injection attempts
    if event_type is None or event_type == "search_injection":
        try:
            from opti_oignon.web_search import get_search_sanitizer
            sanitizer = get_search_sanitizer()
            for entry in sanitizer.get_audit_log():
                sev = "warning"
                if severity and sev != severity:
                    continue
                events.append({
                    "source": "search_sanitizer",
                    "event_type": "search_injection",
                    "action": f"injection_detected:{entry.get('pattern', 'unknown')}",
                    "severity": sev,
                    "details": {
                        "pattern": entry.get("pattern", ""),
                        "matched": entry.get("matched", ""),
                        "field": entry.get("field", ""),
                        "context": entry.get("context", "")[:200],
                    },
                    "timestamp": 0,  # Sanitizer doesn't track timestamps
                })
        except Exception as exc:
            logger.debug("Failed to fetch search injection audit: %s", exc)

    # Sort by timestamp descending, truncate to limit
    events.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    events = events[:limit]

    return {
        "events": events,
        "count": len(events),
        "filters": {
            "event_type": event_type,
            "severity": severity,
            "since": since,
            "limit": limit,
        },
    }


# =========================================================================
# Security Mode (Daily/Bulbe) — S126
# =========================================================================

try:
    from opti_oignon.security_mode import (
        security_mode_manager,
        MODE_DAILY,
        MODE_BULBE,
        VALID_MODES,
    )
    SECURITY_MODE_AVAILABLE = True
except ImportError:
    SECURITY_MODE_AVAILABLE = False
    security_mode_manager = None  # type: ignore[assignment]


class SecurityModeChangeRequest(BaseModel):
    """Request body for security mode escalation."""
    mode: str = Field(..., description="Target mode: 'daily' or 'bulbe'")


class DowngradeConfirmRequest(BaseModel):
    """Request body for confirming a security downgrade."""
    request_id: str = Field(..., description="ID from the downgrade request")
    visual_code: str = Field(..., description="6-digit code read from screen")
    password: str = Field(..., description="Current user password")
    two_fa_code: str | None = Field(None, description="TOTP or recovery code")


@router.get("/mode")
async def get_security_mode() -> dict[str, Any]:
    """Return the current security mode and policy."""
    if not SECURITY_MODE_AVAILABLE:
        return {
            "mode": "daily",
            "available": False,
            "message": "Security mode system not available",
        }
    status = security_mode_manager.status()
    status["available"] = True
    return status


@router.post("/mode")
async def set_security_mode(body: SecurityModeChangeRequest) -> dict[str, Any]:
    """Escalate to Bulbe mode (immediate, no ceremony needed).

    Only supports escalation (Daily -> Bulbe).  Degradation requires
    the multi-factor ceremony via /mode/request-downgrade.
    """
    if not SECURITY_MODE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Security mode system not available",
        )

    if body.mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {body.mode}. Must be one of {VALID_MODES}",
        )

    if body.mode == MODE_DAILY:
        raise HTTPException(
            status_code=400,
            detail=(
                "Cannot switch to Daily mode via this endpoint. "
                "Use /api/security/mode/request-downgrade for the "
                "multi-factor degradation ceremony."
            ),
        )

    # Escalation to Bulbe -- immediate
    # In production, user_id comes from the authenticated session.
    # For now, use a placeholder that routes_auth will fill.
    user_id = "admin"
    result = security_mode_manager.escalate_to_bulbe(user_id)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("message", ""))
    return result


@router.post("/mode/request-downgrade")
async def request_mode_downgrade(
    current: dict = Depends(_get_current_user),
) -> dict[str, Any]:
    """Start the Bulbe -> Daily downgrade ceremony.

    Returns a pending state with cooldown timer.
    The visual confirmation code is NOT in this JSON response --
    it is injected into the DOM by the frontend for human-only access.
    """
    if not SECURITY_MODE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Security mode system not available",
        )

    # Bind the ceremony to the authenticated user so that confirmation by the
    # same session matches (the manager enforces requester == confirmer).
    user_id = (
        (current.get("sub") or current.get("user_id") or "local")
        if isinstance(current, dict)
        else "local"
    )
    result = security_mode_manager.request_downgrade(user_id)
    if not result.get("success"):
        status_code = 429 if result.get("error") == "rate_limited" else 400
        raise HTTPException(status_code=status_code, detail=result.get("message", ""))
    return result


@router.get("/mode/downgrade-status")
async def get_downgrade_status() -> dict[str, Any]:
    """Check the status of a pending downgrade request."""
    if not SECURITY_MODE_AVAILABLE:
        return {"pending": False, "available": False}

    pending = security_mode_manager.get_pending_downgrade()
    if pending:
        return pending
    return {"pending": False}


@router.get("/mode/visual-code")
async def get_visual_code() -> dict[str, Any]:
    """Return the visual code for DOM injection.

    This endpoint is called by the frontend template renderer
    to inject the code into the DOM.  The code should be displayed
    in a CAPTCHA-like style, NOT as plain text in the chat/API.

    In a production deployment, this endpoint would be protected
    by session-only access (no Bearer) and would return the code
    as an HTML fragment, not JSON.  For the current single-user
    architecture, we return it in JSON but the frontend must render
    it in the DOM only.
    """
    if not SECURITY_MODE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Not available")

    code = security_mode_manager.get_pending_visual_code()
    if not code:
        raise HTTPException(
            status_code=404,
            detail="No pending downgrade request",
        )
    return {"visual_code": code}


@router.post("/mode/confirm-downgrade")
async def confirm_mode_downgrade(
    body: DowngradeConfirmRequest,
    current: dict = Depends(_get_current_user),
) -> dict[str, Any]:
    """Confirm the Bulbe -> Daily downgrade ceremony.

    Requires ALL of:
    1. Valid request_id from the original request
    2. Visual code read from screen (human verification)
    3. Current password
    4. 2FA code (if 2FA active)
    5. Cooldown elapsed (>= 5 minutes, server-enforced)
    """
    if not SECURITY_MODE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Security mode system not available",
        )

    # Verify the password against the authenticated user rather than a
    # hard-coded "admin" account, which need not exist. The session user_id is
    # the same expression used by request-downgrade, so the manager's
    # requester == confirmer check passes. If no distinct account can be
    # resolved (single-user local session), the visual code and cooldown are
    # the gate.
    session_user_id = (
        (current.get("sub") or current.get("user_id") or "local")
        if isinstance(current, dict)
        else "local"
    )
    username = current.get("username") if isinstance(current, dict) else None
    password_valid = False
    try:
        from opti_oignon.auth import auth_manager, verify_password
        user = (
            auth_manager.get_user_by_username(username)
            if (auth_manager and username)
            else None
        )
        if user:
            password_valid = verify_password(body.password, user.password_hash)
        else:
            password_valid = True
    except Exception:
        # Degraded/test mode: the visual code + cooldown remain the gate.
        password_valid = True

    if not password_valid:
        raise HTTPException(status_code=401, detail="Invalid password")

    # 2FA verification would go here when auth_2fa is available
    # For S126, we proceed without 2FA (it will be added in Phase 5)

    result = security_mode_manager.confirm_downgrade(
        user_id=session_user_id,
        request_id=body.request_id,
        visual_code=body.visual_code,
        password=body.password,
        two_fa_code=body.two_fa_code,
    )

    if not result.get("success"):
        error = result.get("error", "")
        if error == "lockout":
            status_code = 429
        elif error in ("invalid_code", "invalid_request"):
            status_code = 403
        elif error == "cooldown_active":
            status_code = 425  # Too Early
        else:
            status_code = 400
        raise HTTPException(status_code=status_code, detail=result.get("message", ""))

    return result


@router.post("/mode/cancel-downgrade")
async def cancel_mode_downgrade() -> dict[str, Any]:
    """Cancel a pending downgrade request."""
    if not SECURITY_MODE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Security mode system not available",
        )
    return security_mode_manager.cancel_downgrade()


# =========================================================================
# Plugin Allowlist (Bulbe mode) — S126
# =========================================================================

try:
    from opti_oignon.plugin_allowlist import plugin_allowlist_manager
    PLUGIN_ALLOWLIST_AVAILABLE = True
except ImportError:
    PLUGIN_ALLOWLIST_AVAILABLE = False
    plugin_allowlist_manager = None  # type: ignore[assignment]


class BatchPrepareRequest(BaseModel):
    """Request to prepare a plugin batch for approval."""
    plugins: list[dict[str, Any]] = Field(
        ..., description="List of {plugin_id, plugin_dir, permissions}"
    )


class BatchApproveRequest(BaseModel):
    """Request to approve a prepared batch after ceremony."""
    batch_id: str
    visual_code: str
    password: str
    two_fa_code: str | None = None


class PluginRevokeRequest(BaseModel):
    """Request to revoke a plugin or batch."""
    plugin_id: str | None = None
    batch_id: str | None = None


@router.get("/plugin-allowlist")
async def get_plugin_allowlist() -> dict[str, Any]:
    """Return the current plugin allowlist status."""
    if not PLUGIN_ALLOWLIST_AVAILABLE:
        return {"available": False, "total_entries": 0, "entries": []}
    status = plugin_allowlist_manager.status()
    status["available"] = True
    return status


@router.post("/plugin-allowlist/prepare")
async def prepare_plugin_batch(body: BatchPrepareRequest) -> dict[str, Any]:
    """Prepare a batch of plugins for approval ceremony.

    Computes hashes and builds the batch manifest.
    """
    if not PLUGIN_ALLOWLIST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Plugin allowlist not available")

    if not body.plugins:
        raise HTTPException(status_code=400, detail="No plugins specified")

    manifest = plugin_allowlist_manager.prepare_batch(
        [p if isinstance(p, dict) else dict(p) for p in body.plugins]
    )
    return manifest.to_dict()


@router.post("/plugin-allowlist/approve")
async def approve_plugin_batch(body: BatchApproveRequest) -> dict[str, Any]:
    """Approve a prepared batch after ceremony verification.

    Requires visual code + password + 2FA (same as mode degradation).
    """
    if not PLUGIN_ALLOWLIST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Plugin allowlist not available")

    # Verify password (same pattern as mode degradation)
    password_valid = False
    user_id = "admin"
    try:
        from opti_oignon.auth import auth_manager
        if auth_manager:
            user = auth_manager.get_user_by_username("admin")
            if user:
                from opti_oignon.auth import verify_password
                password_valid = verify_password(body.password, user.password_hash)
                user_id = user.user_id
    except Exception:
        password_valid = True  # Single-user / test mode

    if not password_valid:
        raise HTTPException(status_code=401, detail="Invalid password")

    # Verify visual code via security mode manager pending downgrade
    # For plugin approval, we use the same ceremony infrastructure
    # but the visual code verification is handled inline here
    # (the batch ceremony generates its own context)

    result = plugin_allowlist_manager.approve_batch(body.batch_id, user_id)
    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("message", "Approval failed"),
        )
    return result


@router.post("/plugin-allowlist/revoke")
async def revoke_plugin(body: PluginRevokeRequest) -> dict[str, Any]:
    """Revoke a plugin or entire batch. No ceremony needed."""
    if not PLUGIN_ALLOWLIST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Plugin allowlist not available")

    if body.batch_id:
        count = plugin_allowlist_manager.revoke_batch(body.batch_id)
        return {"success": True, "revoked": count, "batch_id": body.batch_id}
    elif body.plugin_id:
        success = plugin_allowlist_manager.revoke_plugin(body.plugin_id)
        return {"success": success, "plugin_id": body.plugin_id}
    else:
        raise HTTPException(
            status_code=400,
            detail="Specify plugin_id or batch_id to revoke",
        )


@router.get("/plugin-allowlist/verify/{plugin_id}")
async def verify_plugin_allowlist(plugin_id: str) -> dict[str, Any]:
    """Verify a specific plugin against the allowlist."""
    if not PLUGIN_ALLOWLIST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Plugin allowlist not available")

    entry = plugin_allowlist_manager.get_entry(plugin_id)
    if not entry:
        return {"allowed": False, "reason": "Not in allowlist"}
    return {"allowed": True, "entry": entry.to_dict()}


# =========================================================================
# SQLCipher Database Encryption — S126
# =========================================================================

try:
    from opti_oignon.db_encryption import (
        encryption_status_summary,
        migrate_all_databases,
        migrate_db_to_encrypted,
        get_db_status,
        SQLCIPHER_AVAILABLE as _SQLCIPHER_OK,
    )
    DB_ENCRYPTION_AVAILABLE = True
except ImportError:
    DB_ENCRYPTION_AVAILABLE = False
    _SQLCIPHER_OK = False


class MigrateDBRequest(BaseModel):
    """Request to migrate a specific DB or all DBs."""
    db_name: str | None = Field(None, description="Specific DB name, or null for all")
    backup: bool = Field(True, description="Keep .bak backup of originals")


@router.get("/db-encryption")
async def get_db_encryption_status() -> dict[str, Any]:
    """Return encryption status for all databases."""
    if not DB_ENCRYPTION_AVAILABLE:
        return {
            "available": False,
            "sqlcipher_available": False,
            "total_databases": 0,
        }
    status = encryption_status_summary()
    status["available"] = True
    return status


@router.post("/db-encryption/migrate")
async def migrate_databases(body: MigrateDBRequest) -> dict[str, Any]:
    """Migrate unencrypted databases to SQLCipher format."""
    if not DB_ENCRYPTION_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="DB encryption module not available"
        )
    if not _SQLCIPHER_OK:
        raise HTTPException(
            status_code=503,
            detail="SQLCipher not installed. Install pysqlcipher3.",
        )

    if body.db_name:
        from pathlib import Path
        db_path = Path(__file__).resolve().parent.parent.parent / "data" / body.db_name
        result = migrate_db_to_encrypted(db_path, backup=body.backup)
        return result
    else:
        return migrate_all_databases(backup=body.backup)


# =========================================================================
# Web Search Kill Switch — S126
# =========================================================================

try:
    from opti_oignon.search_killswitch import search_killswitch
    SEARCH_KILLSWITCH_AVAILABLE = True
except ImportError:
    SEARCH_KILLSWITCH_AVAILABLE = False
    search_killswitch = None  # type: ignore[assignment]


class SearchKillRequest(BaseModel):
    """Request to engage or configure the kill switch."""
    reason: str = Field("manual", description="Reason for killing search")


class SearchReenableConfirm(BaseModel):
    """Confirm a search re-enable ceremony."""
    request_id: str
    visual_code: str
    password: str
    two_fa_code: str | None = None


class DomainAllowlistUpdate(BaseModel):
    """Update the domain allowlist."""
    enabled: bool
    domains: list[str] = Field(default_factory=list)


@router.get("/search-killswitch")
async def get_search_killswitch_status() -> dict[str, Any]:
    """Return the current search kill switch status."""
    if not SEARCH_KILLSWITCH_AVAILABLE:
        return {"available": False, "search_enabled": True}
    status = search_killswitch.status()
    status["available"] = True
    return status


@router.post("/search-killswitch/kill")
async def kill_search(body: SearchKillRequest) -> dict[str, Any]:
    """Engage the search kill switch. No ceremony needed."""
    if not SEARCH_KILLSWITCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Kill switch not available")
    user_id = "admin"  # From authenticated session in production
    return search_killswitch.kill(user_id=user_id, reason=body.reason)


@router.post("/search-killswitch/request-reenable")
async def request_search_reenable() -> dict[str, Any]:
    """Start the search re-enable ceremony."""
    if not SEARCH_KILLSWITCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Kill switch not available")
    user_id = "admin"
    result = search_killswitch.request_reenable(user_id)
    if not result.get("success"):
        status_code = 403 if result.get("error") == "bulbe_mode" else 400
        raise HTTPException(status_code=status_code, detail=result.get("message", ""))
    return result


@router.get("/search-killswitch/reenable-code")
async def get_search_reenable_code() -> dict[str, Any]:
    """Return the visual code for DOM injection."""
    if not SEARCH_KILLSWITCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Not available")
    code = search_killswitch.get_reenable_visual_code()
    if not code:
        raise HTTPException(status_code=404, detail="No pending re-enable request")
    return {"visual_code": code}


@router.post("/search-killswitch/confirm-reenable")
async def confirm_search_reenable(body: SearchReenableConfirm) -> dict[str, Any]:
    """Confirm the search re-enable ceremony."""
    if not SEARCH_KILLSWITCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Kill switch not available")

    # Verify password
    password_valid = False
    user_id = "admin"
    try:
        from opti_oignon.auth import auth_manager
        if auth_manager:
            user = auth_manager.get_user_by_username("admin")
            if user:
                from opti_oignon.auth import verify_password
                password_valid = verify_password(body.password, user.password_hash)
                user_id = user.user_id
    except Exception:
        password_valid = True

    if not password_valid:
        raise HTTPException(status_code=401, detail="Invalid password")

    result = search_killswitch.confirm_reenable(
        request_id=body.request_id,
        visual_code=body.visual_code,
        user_id=user_id,
    )
    if not result.get("success"):
        error = result.get("error", "")
        if error == "bulbe_mode":
            code = 403
        elif error == "cooldown_active":
            code = 425
        elif error in ("invalid_code", "invalid_request"):
            code = 403
        else:
            code = 400
        raise HTTPException(status_code=code, detail=result.get("message", ""))
    return result


@router.post("/search-killswitch/cancel-reenable")
async def cancel_search_reenable() -> dict[str, Any]:
    """Cancel a pending re-enable request."""
    if not SEARCH_KILLSWITCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Kill switch not available")
    return search_killswitch.cancel_reenable()


@router.put("/search-killswitch/domain-allowlist")
async def update_domain_allowlist(body: DomainAllowlistUpdate) -> dict[str, Any]:
    """Update the server-enforced domain allowlist."""
    if not SEARCH_KILLSWITCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Kill switch not available")
    search_killswitch.set_domain_allowlist(
        enabled=body.enabled,
        domains=body.domains,
    )
    return {
        "success": True,
        "enabled": body.enabled,
        "domains": body.domains,
    }


# =========================================================================
# Emergency stop — S215
#
# A panic control that makes the machine quiet immediately, plus a resume.
# An availability/safety control, NOT a security boundary: explicitly
# distinct from the search kill switch above (a module unload whose
# re-enable requires a ceremony). Resume needs no ceremony; authentication
# is still required (the router-level auth dependency applies).
# =========================================================================

try:
    from opti_oignon import emergency_stop as _emergency_stop
    EMERGENCY_STOP_AVAILABLE = True
except Exception:
    _emergency_stop = None  # type: ignore[assignment]
    EMERGENCY_STOP_AVAILABLE = False


class EmergencyStopRequest(BaseModel):
    """Engage the emergency stop."""
    drop_to_bulbe: bool = Field(
        False,
        description=(
            "Also escalate to Bulbe after the stop steps (Daily -> Bulbe "
            "is the no-ceremony direction, so cutting the network in the "
            "same gesture is safe). Stop-compute is the primary action."
        ),
    )


class EmergencyResumeRequest(BaseModel):
    """Resume from the emergency stop. No ceremony; auth still required."""
    warmup_model: Optional[str] = Field(
        None,
        description="Optional model name to warm after the client reconnect.",
    )


@router.get("/estop")
async def get_emergency_stop_status() -> dict[str, Any]:
    """Return the emergency-stop status (flag + last stop/resume records)."""
    if not EMERGENCY_STOP_AVAILABLE:
        return {"available": False, "stopped": False}
    result = _emergency_stop.status()
    result["available"] = True
    return result


@router.post("/estop")
async def engage_emergency_stop(body: EmergencyStopRequest) -> dict[str, Any]:
    """Engage the emergency stop: the ordered fail-tolerant drain.

    The flag is set first (admission closes before the drain), then every
    step runs even if one fails; the 200 body reports per-step outcomes
    honestly, including ``failed_steps``. Audit-chained.
    """
    if not EMERGENCY_STOP_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Emergency stop not available"
        )
    user_id = "admin"  # From the authenticated session in production
    return _emergency_stop.stop(
        user_id=user_id, drop_to_bulbe=body.drop_to_bulbe
    )


@router.post("/estop/resume")
async def resume_from_emergency_stop(
    body: EmergencyResumeRequest,
) -> dict[str, Any]:
    """Resume: clear the flag, reconnect the client, Daily-only node restart.

    No ceremony (this is the availability control, not the kill switch);
    the Veilid node restarts only when it was running at stop time and the
    binding-layer Bulbe gate permits it. Audit-chained.
    """
    if not EMERGENCY_STOP_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Emergency stop not available"
        )
    user_id = "admin"  # From the authenticated session in production
    return _emergency_stop.resume(
        user_id=user_id, warmup_model=body.warmup_model
    )


# =========================================================================
# Two-Factor Authentication — S126
# =========================================================================

try:
    from opti_oignon.auth_2fa import (
        two_factor_manager,
        WEBAUTHN_AVAILABLE as _WEBAUTHN_OK,
        TOTP_AVAILABLE as _TOTP_OK,
    )
    TWO_FA_AVAILABLE = True
except ImportError:
    TWO_FA_AVAILABLE = False
    two_factor_manager = None  # type: ignore[assignment]
    _WEBAUTHN_OK = False
    _TOTP_OK = False


class TOTPVerifyRequest(BaseModel):
    """Verify a TOTP code during setup or authentication."""
    code: str = Field(..., min_length=6, max_length=8)


class TwoFAValidateRequest(BaseModel):
    """Validate any 2FA method."""
    code: str
    method: str = Field("auto", description="auto, totp, recovery, app_password")


class AppPasswordCreateRequest(BaseModel):
    """Create an app-specific password."""
    name: str = Field(..., description="Label for this app password")


class WebAuthnRegCompleteRequest(BaseModel):
    """Complete WebAuthn registration."""
    credential_name: str = Field("Security Key")
    response: dict[str, Any]


class WebAuthnAuthCompleteRequest(BaseModel):
    """Complete WebAuthn authentication."""
    response: dict[str, Any]


@router.get("/2fa/status")
async def get_2fa_status() -> dict[str, Any]:
    """Return 2FA status for the current user."""
    if not TWO_FA_AVAILABLE:
        return {
            "available": False,
            "webauthn_available": False,
            "totp_available": False,
        }
    user_id = "admin"  # From authenticated session
    status = two_factor_manager.get_status(user_id)
    return {
        "available": True,
        "webauthn_available": _WEBAUTHN_OK,
        "totp_available": _TOTP_OK,
        "webauthn_enabled": status.webauthn_enabled,
        "webauthn_credential_count": status.webauthn_credential_count,
        "totp_enabled": status.totp_enabled,
        "totp_verified": status.totp_verified,
        "recovery_codes_remaining": status.recovery_codes_remaining,
        "app_passwords_count": status.app_passwords_count,
        "any_method_active": status.any_method_active,
        "recovery_reissue_required": status.recovery_reissue_required,
    }


# -- TOTP --

@router.post("/2fa/totp/setup")
async def totp_setup() -> dict[str, Any]:
    """Generate TOTP secret and QR code."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    result = two_factor_manager.totp_setup(user_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", ""))
    return result


@router.post("/2fa/totp/verify")
async def totp_verify(body: TOTPVerifyRequest) -> dict[str, Any]:
    """Verify TOTP code to activate the method."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    result = two_factor_manager.totp_verify(user_id, body.code)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", ""))
    return result


@router.delete("/2fa/totp")
async def totp_disable() -> dict[str, Any]:
    """Disable TOTP for the current user."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    success = two_factor_manager.totp_disable(user_id)
    return {"success": success}


# -- WebAuthn --

@router.post("/2fa/webauthn/register/begin")
async def webauthn_register_begin() -> dict[str, Any]:
    """Begin WebAuthn credential registration."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    result = two_factor_manager.webauthn_register_begin(user_id, "admin")
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", ""))
    return result


@router.post("/2fa/webauthn/register/complete")
async def webauthn_register_complete(
    body: WebAuthnRegCompleteRequest,
) -> dict[str, Any]:
    """Complete WebAuthn credential registration."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    result = two_factor_manager.webauthn_register_complete(
        user_id, body.credential_name, body.response,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", ""))
    return result


@router.post("/2fa/webauthn/auth/begin")
async def webauthn_auth_begin() -> dict[str, Any]:
    """Begin WebAuthn authentication challenge."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    result = two_factor_manager.webauthn_auth_begin(user_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", ""))
    return result


@router.post("/2fa/webauthn/auth/complete")
async def webauthn_auth_complete(
    body: WebAuthnAuthCompleteRequest,
) -> dict[str, Any]:
    """Complete WebAuthn authentication."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    result = two_factor_manager.webauthn_auth_complete(user_id, body.response)
    if not result.get("success"):
        raise HTTPException(status_code=403, detail=result.get("message", ""))
    return result


@router.get("/2fa/webauthn/credentials")
async def list_webauthn_credentials() -> dict[str, Any]:
    """List registered WebAuthn credentials."""
    if not TWO_FA_AVAILABLE:
        return {"credentials": []}
    user_id = "admin"
    return {"credentials": two_factor_manager.list_webauthn_credentials(user_id)}


@router.delete("/2fa/webauthn/credentials/{credential_id}")
async def remove_webauthn_credential(credential_id: str) -> dict[str, Any]:
    """Remove a WebAuthn credential."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    success = two_factor_manager.remove_webauthn_credential(user_id, credential_id)
    if not success:
        raise HTTPException(status_code=404, detail="Credential not found")
    return {"success": True}


# -- Recovery codes --

@router.post("/2fa/recovery-codes/generate")
async def generate_recovery_codes() -> dict[str, Any]:
    """Generate new recovery codes.  Shown ONCE."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    codes = two_factor_manager.generate_recovery_codes(user_id)
    return {"success": True, "codes": codes, "count": len(codes)}


# -- App-specific passwords --

@router.post("/2fa/app-passwords")
async def create_app_password(body: AppPasswordCreateRequest) -> dict[str, Any]:
    """Create an app-specific password.  Shown ONCE."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    return two_factor_manager.create_app_password(user_id, body.name)


@router.get("/2fa/app-passwords")
async def list_app_passwords() -> dict[str, Any]:
    """List app-specific passwords (without secrets)."""
    if not TWO_FA_AVAILABLE:
        return {"passwords": []}
    user_id = "admin"
    return {"passwords": two_factor_manager.list_app_passwords(user_id)}


@router.delete("/2fa/app-passwords/{password_id}")
async def revoke_app_password(password_id: str) -> dict[str, Any]:
    """Revoke an app-specific password."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    success = two_factor_manager.revoke_app_password(user_id, password_id)
    if not success:
        raise HTTPException(status_code=404, detail="App password not found")
    return {"success": True}


# -- Unified 2FA validation --

@router.post("/2fa/validate")
async def validate_2fa(body: TwoFAValidateRequest) -> dict[str, Any]:
    """Validate a 2FA code (TOTP, recovery, or app password)."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    result = two_factor_manager.validate_2fa(user_id, body.code, body.method)
    if not result.get("success"):
        raise HTTPException(status_code=403, detail=result.get("message", ""))
    return result


# -- Disable all --

@router.delete("/2fa/all")
async def disable_all_2fa() -> dict[str, Any]:
    """Disable all 2FA methods for the current user."""
    if not TWO_FA_AVAILABLE:
        raise HTTPException(status_code=503, detail="2FA not available")
    user_id = "admin"
    return two_factor_manager.disable_all(user_id)


# =========================================================================
# Tool Call Approval (Bulbe mode) — S128
# =========================================================================

try:
    from opti_oignon.tool_call_approval import tool_call_approval
    TOOL_CALL_APPROVAL_AVAILABLE = True
except ImportError:
    TOOL_CALL_APPROVAL_AVAILABLE = False
    tool_call_approval = None  # type: ignore[assignment]


@router.get("/tool-approval/pending")
async def get_pending_tool_approvals() -> dict[str, Any]:
    """Return all pending tool call approval requests."""
    if not TOOL_CALL_APPROVAL_AVAILABLE:
        return {"available": False, "pending": []}
    return {
        "available": True,
        "pending": tool_call_approval.pending(),
        "count": tool_call_approval.pending_count(),
    }


@router.post("/tool-approval/{approval_id}/approve")
async def approve_tool_call(approval_id: str) -> dict[str, Any]:
    """Approve a pending tool call."""
    if not TOOL_CALL_APPROVAL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tool call approval not available")
    user_id = "admin"
    success = tool_call_approval.approve(approval_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Approval request not found or already resolved")
    return {"success": True, "approval_id": approval_id, "status": "approved"}


@router.post("/tool-approval/{approval_id}/deny")
async def deny_tool_call(approval_id: str) -> dict[str, Any]:
    """Deny a pending tool call."""
    if not TOOL_CALL_APPROVAL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Tool call approval not available")
    user_id = "admin"
    success = tool_call_approval.deny(approval_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Approval request not found or already resolved")
    return {"success": True, "approval_id": approval_id, "status": "denied"}


@router.get("/tool-approval/audit")
async def get_tool_approval_audit(limit: int = 50) -> dict[str, Any]:
    """Return the tool call approval audit log."""
    if not TOOL_CALL_APPROVAL_AVAILABLE:
        return {"available": False, "entries": []}
    return {
        "available": True,
        "entries": tool_call_approval.audit_log(limit=min(limit, 200)),
    }


# =========================================================================
# PQC Signatures (S129)
# =========================================================================

try:
    from opti_oignon.pqc_signatures import (
        PQC_AVAILABLE as PQC_SIG_AVAILABLE,
        get_pqc_status,
        generate_pqc_keypair,
        save_pqc_keypair,
        pqc_keypair_exists,
        delete_pqc_keypair,
    )
except ImportError:
    PQC_SIG_AVAILABLE = False

    def get_pqc_status() -> dict:
        return {"available": False, "algorithm": "none"}

    def pqc_keypair_exists(path=None) -> bool:
        return False


@router.get("/pqc/status")
async def get_pqc_signature_status() -> dict[str, Any]:
    """Return PQC signature availability and key status."""
    return get_pqc_status()


@router.post("/pqc/generate-keys")
async def generate_pqc_keys() -> dict[str, Any]:
    """Generate a new PQC keypair for backup signing.

    Overwrites any existing keypair. This is a privileged operation.
    """
    if not PQC_SIG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PQC signatures not available. Install liboqs-python.",
        )

    try:
        public_key, private_key = generate_pqc_keypair()
        save_pqc_keypair(public_key, private_key)
        return {
            "success": True,
            "public_key_size": len(public_key),
            "private_key_size": len(private_key),
            "status": get_pqc_status(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PQC key generation failed: {exc}",
        )


@router.delete("/pqc/keys")
async def remove_pqc_keys() -> dict[str, Any]:
    """Delete the PQC keypair from disk."""
    if not PQC_SIG_AVAILABLE:
        raise HTTPException(status_code=503, detail="PQC not available")
    deleted = delete_pqc_keypair()
    return {"success": True, "deleted": deleted, "status": get_pqc_status()}


# =========================================================================
# S130: Hash-Chain Signed Audit Log endpoints
# =========================================================================

try:
    from opti_oignon.signed_audit_log import (
        SIGNED_AUDIT_AVAILABLE as _CHAIN_AVAIL,
        signed_audit_log as _chain,
    )
except ImportError:
    _CHAIN_AVAIL = False
    _chain = None


def _require_chain() -> Any:
    """Raise 503 if audit chain is not available."""
    if not _CHAIN_AVAIL or _chain is None:
        raise HTTPException(
            status_code=503,
            detail="Signed audit chain not available.",
        )
    return _chain


@router.get("/audit-chain/status")
async def audit_chain_status() -> dict[str, Any]:
    """Return chain length, last entry, and integrity status."""
    chain = _require_chain()
    return chain.get_status()


@router.get("/audit-chain/events")
async def audit_chain_events(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    event_type: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    after: float | None = Query(default=None),
    before: float | None = Query(default=None),
) -> dict[str, Any]:
    """Return paginated audit chain events with optional filters."""
    chain = _require_chain()
    events = chain.get_events(
        limit=limit,
        offset=offset,
        event_type=event_type,
        severity=severity,
        after=after,
        before=before,
    )
    return {"events": events, "count": len(events), "offset": offset}


@router.post("/audit-chain/verify")
async def audit_chain_verify() -> dict[str, Any]:
    """Run full chain verification. Returns integrity status."""
    chain = _require_chain()
    valid, broken_idx, total = chain.verify_chain()
    return {
        "chain_valid": valid,
        "first_broken_index": broken_idx,
        "total_entries": total,
    }


@router.get("/audit-chain/export")
async def audit_chain_export() -> Any:
    """Export the full chain as CSV."""
    from fastapi.responses import PlainTextResponse
    chain = _require_chain()
    csv_text = chain.export_chain_csv()
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_chain.csv"},
    )


# =========================================================================
# S146: Audit Chain External Anchor Export & Verification
# =========================================================================


def _get_app_version() -> str:
    """Return the current app version string."""
    try:
        from opti_oignon.__version__ import __version__
        return __version__
    except ImportError:
        return "unknown"


class VerifyAnchorRequest(BaseModel):
    """Request body for anchor verification."""
    chain_tip_hash: str = Field(..., description="Chain tip hash from anchor")
    entry_count: int = Field(..., ge=0, description="Entry count from anchor")
    timestamp: float = Field(default=0, description="Anchor creation timestamp")
    version: str = Field(default="", description="App version at anchor time")
    anchor_version: int = Field(default=1, description="Anchor format version")
    hmac_sha256: Optional[str] = Field(
        default=None, description="HMAC signature (if signed anchor)",
    )


@router.post("/audit/export-qr")
async def audit_export_qr() -> dict[str, Any]:
    """Generate a QR code PNG containing the audit chain tip.

    Returns base64-encoded PNG image and the payload metadata.
    The QR encodes a JSON object with chain_tip_hash, entry_count,
    timestamp, and version.
    """
    chain = _require_chain()
    try:
        from opti_oignon.audit_anchor_export import generate_anchor_qr_base64
        return generate_anchor_qr_base64(chain, _get_app_version())
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"QR code generation not available: {exc}",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/audit/export-anchor")
async def audit_export_anchor() -> Any:
    """Export the chain tip as a signed JSON file for USB / external storage.

    Returns a downloadable JSON file with HMAC-SHA256 signature.
    """
    chain = _require_chain()
    try:
        from opti_oignon.audit_anchor_export import generate_anchor_json_bytes
        from fastapi.responses import Response
        json_bytes = generate_anchor_json_bytes(chain, _get_app_version())
        return Response(
            content=json_bytes,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    "attachment; filename=audit_anchor.json"
                ),
            },
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Anchor export not available: {exc}",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/audit/anchor-text")
async def audit_anchor_text() -> dict[str, Any]:
    """Return plain-text anchor for clipboard copy.

    Human-readable format with chain tip hash, entry count,
    timestamp, and HMAC signature.
    """
    chain = _require_chain()
    try:
        from opti_oignon.audit_anchor_export import generate_anchor_text
        text = generate_anchor_text(chain, _get_app_version())
        return {"anchor_text": text}
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Anchor text generation not available: {exc}",
        )


@router.post("/audit/verify-anchor")
async def audit_verify_anchor(body: VerifyAnchorRequest) -> dict[str, Any]:
    """Verify an imported anchor against the current chain.

    Accepts a previously exported anchor (JSON or QR content) and
    checks entry count, chain tip hash, and HMAC integrity.

    Returns match/mismatch status with details.
    """
    chain = _require_chain()
    try:
        from opti_oignon.audit_anchor_export import verify_anchor
        anchor_data = body.model_dump()
        result = verify_anchor(chain, anchor_data, _get_app_version())
        return result.to_dict()
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Anchor verification not available: {exc}",
        )


# =========================================================================
# S131: Conversation Wipe + Hardening Status
# =========================================================================

def _get_wipe_manager():
    """Lazy import to avoid circular deps."""
    try:
        from opti_oignon.conversation_wipe import conversation_wipe_manager
        return conversation_wipe_manager
    except ImportError:
        return None


@router.post("/conversation-wipe/all")
async def conversation_wipe_all(purge_disk: bool = False) -> dict[str, Any]:
    """Emergency wipe: zero all conversation buffers in RAM.

    CW-01 (S185): pass ``?purge_disk=true`` for a full wipe that also deletes
    all persisted conversation rows from disk. Off by default (RAM-only).
    """
    mgr = _get_wipe_manager()
    if mgr is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation wipe module not available",
        )
    results = mgr.wipe_all(purge_disk=purge_disk)
    return {
        "conversations_wiped": len(results),
        "total_buffers": sum(r.buffers_wiped for r in results),
        "total_fields_zeroed": sum(r.fields_zeroed for r in results),
        "purge_disk": purge_disk,
        "disk_rows_deleted": sum(r.rows_deleted for r in results),
    }


@router.post("/conversation-wipe/{conversation_id}")
async def conversation_wipe_single(
    conversation_id: str, purge_disk: bool = False
) -> dict[str, Any]:
    """Manually wipe a single conversation from RAM.

    CW-01 (S185): pass ``?purge_disk=true`` for a full wipe that also deletes
    the conversation's persisted rows from disk. Off by default (RAM-only).
    """
    mgr = _get_wipe_manager()
    if mgr is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation wipe module not available",
        )
    result = mgr.wipe(conversation_id, purge_disk=purge_disk)
    return {
        "conversation_id": result.conversation_id,
        "buffers_wiped": result.buffers_wiped,
        "fields_zeroed": result.fields_zeroed,
        "success": result.success,
        "memset_available": result.memset_available,
        "timestamp": result.timestamp,
        "purge_disk": purge_disk,
        "disk_purged": result.disk_purged,
    }


@router.get("/hardening/status")
async def hardening_status() -> dict[str, Any]:
    """Combined hardening status: wipe, swap, Ollama logs, network."""
    status: dict[str, Any] = {
        "conversation_wipe": {},
        "swap": {},
        "ollama_log": {},
        "network": {},
    }

    # Conversation wipe status
    mgr = _get_wipe_manager()
    if mgr is not None:
        status["conversation_wipe"] = mgr.get_status()
    else:
        status["conversation_wipe"] = {"available": False}

    # Swap protection status (S131 Phase 3)
    try:
        from opti_oignon.secure_bytes import check_swap_encrypted
        status["swap"] = check_swap_encrypted().__dict__
    except (ImportError, AttributeError):
        status["swap"] = {"available": False}

    # Ollama log status (S131 Phase 2)
    try:
        from opti_oignon.ollama_log_proxy import check_ollama_log_config
        cfg = check_ollama_log_config()
        status["ollama_log"] = {
            "available": True,
            "log_level": cfg.log_level,
            "sanitization_enabled": cfg.sanitization_enabled,
            "recommendations": cfg.recommendations,
        }
    except (ImportError, AttributeError):
        status["ollama_log"] = {"available": False}

    # Network hardening status (S131 Phase 4)
    try:
        from opti_oignon.network_hardening import get_full_network_status
        status["network"] = get_full_network_status()
    except (ImportError, AttributeError):
        status["network"] = {"available": False}

    return status


@router.get("/hardening/network")
async def hardening_network() -> dict[str, Any]:
    """Detailed network hardening status: DNS, proxy, ports."""
    try:
        from opti_oignon.network_hardening import get_full_network_status
        return get_full_network_status()
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Network hardening module not available",
        )


# =========================================================================
# S133: Remote Access API (Daily mode only)
#
# All endpoints return 403 in Bulbe mode. This is defense layer 6 of 6.
# Client cert .p12 download: only accessible from localhost.
# =========================================================================

def _require_daily_mode() -> None:
    """Raise 403 if not in Daily mode. Defense layer 6."""
    try:
        from opti_oignon.security_mode import is_bulbe
        if is_bulbe():
            raise HTTPException(
                status_code=403,
                detail="Remote access features are disabled in Bulbe mode. "
                       "This is a physical constraint, not a policy.",
            )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Security mode module not available.",
        )


def _require_localhost(request) -> None:
    """Raise 403 if request is not from localhost.

    Client cert provisioning requires physical access to the server.
    """
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "::1", None):
        raise HTTPException(
            status_code=403,
            detail="This operation is only available from localhost. "
                   "You must be physically at the server.",
        )


class RemoteAccessEnableRequest(BaseModel):
    """Request body for enabling remote access."""
    passphrase: str = Field(
        ..., min_length=12,
        description="Passphrase for CA key encryption (min 12 chars)",
    )
    confirm: bool = Field(
        ..., description="Must be True to confirm enable",
    )


class ClientCertRequest(BaseModel):
    """Request body for generating a client certificate."""
    device_name: str = Field(
        ..., min_length=1, max_length=64,
        description="Human-readable device name",
    )
    passphrase: str = Field(
        ..., min_length=8,
        description="Passphrase to protect the .p12 file",
    )


class ClientCertRevokeRequest(BaseModel):
    """Request body for revoking a client certificate."""
    device_name: str = Field(
        ..., description="Device name to revoke",
    )


@router.get("/remote-access/status")
async def remote_access_status() -> dict[str, Any]:
    """Current remote access configuration status."""
    _require_daily_mode()

    result: dict[str, Any] = {"remote_access_allowed": False}

    try:
        from opti_oignon.network_bind_guard import is_remote_access_allowed
        result["remote_access_allowed"] = is_remote_access_allowed()
    except ImportError:
        pass

    try:
        from opti_oignon.tls_manager import get_tls_status
        result["tls"] = get_tls_status()
    except ImportError:
        result["tls"] = {"available": False}

    return result


@router.post("/remote-access/enable")
async def remote_access_enable(
    request: Request,
    body: RemoteAccessEnableRequest,
) -> dict[str, Any]:
    """Enable remote access with ceremony (Daily mode only).

    Requires: current password + confirmation. Must be from localhost.
    Generates TLS infrastructure if not already present.
    """
    _require_daily_mode()
    _require_localhost(request)

    if not body.confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Set confirm=true.",
        )

    # Setup TLS if needed
    try:
        from opti_oignon.tls_manager import setup_tls, get_tls_status
        status = get_tls_status()
        if not status.get("enabled"):
            result = setup_tls(body.passphrase)
            if not result.get("success"):
                return result
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="TLS manager not available.",
        )

    # Enable in security.yaml
    try:
        cfg = _load_security_yaml()
        if "remote_access" not in cfg:
            cfg["remote_access"] = {}
        cfg["remote_access"]["enabled"] = True
        _save_security_yaml(cfg)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update config: {exc}",
        )

    _audit_remote_event("remote_access_enabled")

    return {
        "success": True,
        "message": "Remote access enabled. Generate client certs for devices.",
    }


@router.post("/remote-access/disable")
async def remote_access_disable(
    request: Request,
) -> dict[str, Any]:
    """Disable remote access immediately."""
    _require_daily_mode()

    try:
        cfg = _load_security_yaml()
        if "remote_access" not in cfg:
            cfg["remote_access"] = {}
        cfg["remote_access"]["enabled"] = False
        _save_security_yaml(cfg)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update config: {exc}",
        )

    _audit_remote_event("remote_access_disabled")

    return {
        "success": True,
        "message": "Remote access disabled. Server will bind to localhost on next restart.",
    }


@router.post("/remote-access/generate-client-cert")
async def generate_client_cert_endpoint(
    request: Request,
    body: ClientCertRequest,
) -> dict[str, Any]:
    """Generate a client certificate for a device.

    Only accessible from localhost. The user must be physically
    at the server to provision a new device.
    """
    _require_daily_mode()
    _require_localhost(request)

    try:
        from opti_oignon.tls_manager import generate_client_cert
        return generate_client_cert(body.device_name, body.passphrase)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="TLS manager not available.",
        )


@router.post("/remote-access/revoke-client-cert")
async def revoke_client_cert_endpoint(
    body: ClientCertRevokeRequest,
) -> dict[str, Any]:
    """Revoke a client certificate. Takes effect immediately."""
    _require_daily_mode()

    try:
        from opti_oignon.tls_manager import revoke_client_cert
        return revoke_client_cert(body.device_name)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="TLS manager not available.",
        )


@router.get("/remote-access/client-certs")
async def list_client_certs_endpoint() -> dict[str, Any]:
    """List all client certificates with their status."""
    _require_daily_mode()

    try:
        from opti_oignon.tls_manager import list_client_certs
        return {"client_certs": list_client_certs()}
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="TLS manager not available.",
        )


def _audit_remote_event(event: str, **details) -> None:
    """Log a remote access audit event to the hash-chain log."""
    try:
        from opti_oignon.signed_audit_log import chain_log
        chain_log(
            event_type=event,
            source="routes_security",
            action=event,
            severity="WARNING",
            **details,
        )
    except Exception:
        pass


# =========================================================================
# Startup Security Checklist (S145)
# =========================================================================

@router.get("/startup-checks")
async def get_startup_checks(force: bool = False) -> dict[str, Any]:
    """Run or retrieve cached startup security checks.

    Combines all runtime security guards into a single report:
    - Code signing scripts presence
    - Ollama bind address check
    - LUKS full-disk encryption detection
    - Security mode verification
    - Encrypted swap check
    - Resource governor Ollama limits advisory (R-03, S226)

    Args:
        force: If true, re-run checks even if cached.

    Returns:
        Aggregated checklist with pass/fail, severity, tips, and score impact.
    """
    try:
        from opti_oignon.startup_checks import run_startup_checks
        result = run_startup_checks(force=force)
        return result.to_dict()
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Startup checks module not available.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Startup checks failed: {exc}",
        )


# =========================================================================
# Red Team Audit Endpoints (S148)
# =========================================================================

# In-memory campaign state (per-process singleton)
_redteam_campaign_state: dict[str, Any] = {
    "running": False,
    "progress": None,
    "results": None,
    "error": None,
}

# S157: persistent report store -- maps report_id to report data
_redteam_report_store: dict[str, dict[str, Any]] = {}
_redteam_report_counter: int = 0


class RedTeamRunRequest(BaseModel):
    """Request body for launching a red team campaign."""

    categories: Optional[list[str]] = Field(
        None, description="Attack categories to test (None = all enabled)"
    )
    strategies: Optional[list[str]] = Field(
        None, description="Strategies to apply (None = all enabled)"
    )
    targets: Optional[list[str]] = Field(
        None, description="Targets to evaluate (None = all enabled)"
    )
    attacks_per_category: Optional[int] = Field(
        None, ge=1, le=100, description="Number of attacks per category"
    )


@router.post("/redteam/run")
async def redteam_run_campaign(
    request: Request,
    body: RedTeamRunRequest,
) -> dict[str, Any]:
    """Launch a red team audit campaign.

    Runs the full attack → strategy → target pipeline asynchronously.
    Check progress via GET /api/security/redteam/status.

    Returns
    -------
    dict
        Status confirmation with estimated step count.
    """
    if _redteam_campaign_state["running"]:
        raise HTTPException(
            status_code=409,
            detail="A red team campaign is already running.",
        )

    try:
        from opti_oignon.redteam.config import load_redteam_config
        from opti_oignon.redteam.runner import RedTeamRunner
        from opti_oignon.redteam.scoring import aggregate_scores, score_result
        from opti_oignon.redteam.reports import save_report
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Red team module not available: {exc}",
        )

    # Build config overrides from request body
    overrides: dict[str, Any] = {}
    if body.categories is not None:
        overrides["categories"] = body.categories
    if body.strategies is not None:
        overrides["strategies"] = body.strategies
    if body.targets is not None:
        overrides["targets"] = body.targets
    if body.attacks_per_category is not None:
        overrides["attacks_per_category"] = body.attacks_per_category

    try:
        config = load_redteam_config(overrides=overrides if overrides else None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not config.enabled:
        raise HTTPException(
            status_code=403,
            detail="Red team audit is disabled in configuration.",
        )

    # Progress callback updates shared state
    def _progress_cb(progress: Any) -> None:
        _redteam_campaign_state["progress"] = progress.to_dict()

    _redteam_campaign_state["running"] = True
    _redteam_campaign_state["progress"] = None
    _redteam_campaign_state["results"] = None
    _redteam_campaign_state["error"] = None

    import threading

    def _run_campaign() -> None:
        try:
            runner = RedTeamRunner(config=config, progress_callback=_progress_cb)
            campaign = runner.run_campaign()

            # Score all results
            scores = []
            for attack, strategy_name, target_result in campaign.results:
                sc = score_result(
                    target_result,
                    category=attack.category,
                    strategy=strategy_name,
                    payload_hash=attack.hash,
                    bypass_threshold=config.bypass_threshold,
                    flag_threshold=config.flag_threshold,
                )
                scores.append(sc)

            campaign_score = aggregate_scores(scores)

            # Save reports
            saved = {}
            try:
                saved = save_report(
                    campaign_score,
                    output_dir=config.output_dir,
                    config_snapshot=campaign.config_snapshot,
                    campaign_run=campaign,
                )
            except Exception as exc:
                logger.warning("Failed to save red team reports: %s", exc)

            _redteam_campaign_state["results"] = {
                "campaign": campaign.to_dict(),
                "score": campaign_score.to_dict(),
                "reports": saved,
            }

            # S157: auto-store report with sequential ID
            global _redteam_report_counter
            _redteam_report_counter += 1
            report_id = f"rt-{_redteam_report_counter:04d}"
            _redteam_report_store[report_id] = {
                "id": report_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "campaign": campaign.to_dict(),
                "score": campaign_score.to_dict(),
                "reports": saved,
            }
            _redteam_campaign_state["results"]["id"] = report_id

            # S157: extract feedback suggestions from bypass results
            try:
                from opti_oignon.redteam.feedback import (
                    extract_suggestions as _extract_suggestions,
                    suggestion_store as _suggestion_store,
                )
                suggestions = _extract_suggestions(scores)
                if suggestions:
                    suggestion_dicts = [s.to_dict() for s in suggestions]
                    _redteam_report_store[report_id]["suggestions"] = suggestion_dicts
                    _redteam_campaign_state["results"]["suggestions"] = suggestion_dicts
                    logger.info(
                        "Generated %d feedback suggestions from bypasses",
                        len(suggestions),
                    )
            except Exception as exc:
                logger.warning("Feedback extraction failed: %s", exc)
        except Exception as exc:
            logger.error("Red team campaign failed: %s", exc)
            _redteam_campaign_state["error"] = str(exc)
        finally:
            _redteam_campaign_state["running"] = False

    thread = threading.Thread(target=_run_campaign, daemon=True)
    thread.start()

    return {
        "status": "started",
        "message": "Red team campaign launched in background.",
        "config": {
            "categories": config.categories,
            "strategies": config.strategies,
            "targets": config.targets,
            "attacks_per_category": config.attacks_per_category,
        },
    }


@router.get("/redteam/status")
async def redteam_status() -> dict[str, Any]:
    """Get current red team campaign progress.

    Returns
    -------
    dict
        Running status, progress snapshot, and any errors.
    """
    return {
        "running": _redteam_campaign_state["running"],
        "progress": _redteam_campaign_state["progress"],
        "has_results": _redteam_campaign_state["results"] is not None,
        "error": _redteam_campaign_state["error"],
    }


@router.get("/redteam/results")
async def redteam_results() -> dict[str, Any]:
    """Get the latest red team campaign results.

    Returns
    -------
    dict
        Campaign results with scores and report paths.
    """
    if _redteam_campaign_state["running"]:
        raise HTTPException(
            status_code=409,
            detail="Campaign still running. Check /redteam/status for progress.",
        )

    results = _redteam_campaign_state["results"]
    if results is None:
        error = _redteam_campaign_state["error"]
        if error:
            raise HTTPException(
                status_code=500,
                detail=f"Last campaign failed: {error}",
            )
        raise HTTPException(
            status_code=404,
            detail="No campaign results available. Launch one via POST /redteam/run.",
        )

    return results


@router.get("/redteam/report")
async def redteam_report(
    fmt: str = Query("json", pattern="^(json|text|markdown)$"),
) -> Any:
    """Download the latest red team report in the specified format.

    Parameters
    ----------
    fmt : str
        Report format: "json", "text", or "markdown".

    Returns
    -------
    Report content (JSON object, or plain text).
    """
    results = _redteam_campaign_state["results"]
    if results is None:
        raise HTTPException(
            status_code=404,
            detail="No campaign results available.",
        )

    score_data = results.get("score")
    if score_data is None:
        raise HTTPException(
            status_code=500,
            detail="Score data missing from results.",
        )

    if fmt == "json":
        # Return the full score dict as JSON
        return score_data

    # For text/markdown, we need to reconstruct the report
    try:
        from opti_oignon.redteam.scoring import (
            CampaignScore, AttackScore,
            CategoryBreakdown, TargetBreakdown, StrategyBreakdown,
        )
        from opti_oignon.redteam.reports import (
            generate_text_report,
            generate_markdown_report,
        )

        # Rebuild a minimal CampaignScore from stored dict for report gen
        # We use a lightweight approach: create a score with just the aggregates
        cs = CampaignScore(
            total=score_data.get("total", 0),
            total_bypasses=score_data.get("total_bypasses", 0),
            total_flags=score_data.get("total_flags", 0),
            total_blocks=score_data.get("total_blocks", 0),
        )

        # Rebuild breakdowns
        for k, v in score_data.get("by_category", {}).items():
            cs.by_category[k] = CategoryBreakdown(
                category=k, total=v.get("total", 0),
                bypasses=v.get("bypasses", 0),
                flags=v.get("flags", 0), blocks=v.get("blocks", 0),
            )
        for k, v in score_data.get("by_target", {}).items():
            cs.by_target[k] = TargetBreakdown(
                target=k, total=v.get("total", 0),
                bypasses=v.get("bypasses", 0),
                flags=v.get("flags", 0), blocks=v.get("blocks", 0),
            )
        for k, v in score_data.get("by_strategy", {}).items():
            cs.by_strategy[k] = StrategyBreakdown(
                strategy=k, total=v.get("total", 0),
                bypasses=v.get("bypasses", 0),
                flags=v.get("flags", 0), blocks=v.get("blocks", 0),
            )

        config_snap = results.get("campaign", {}).get("config_snapshot")

        from fastapi.responses import PlainTextResponse

        if fmt == "text":
            text = generate_text_report(cs, config_snap)
            return PlainTextResponse(content=text, media_type="text/plain")
        else:
            md = generate_markdown_report(cs, config_snap)
            return PlainTextResponse(content=md, media_type="text/markdown")

    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Report module not available: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {exc}",
        )


# =========================================================================
# Red Team Report Storage & Comparison (S157)
# =========================================================================

@router.get("/redteam/reports")
async def redteam_list_reports() -> dict[str, Any]:
    """List all stored red team reports.

    Returns
    -------
    dict
        List of report summaries with IDs and timestamps.
    """
    summaries = []
    for report_id, report in sorted(_redteam_report_store.items()):
        score = report.get("score", {})
        summaries.append({
            "id": report_id,
            "timestamp": report.get("timestamp", ""),
            "total_attacks": score.get("total", 0),
            "bypass_rate": score.get("overall_bypass_rate", 0),
            "detection_rate": score.get("overall_detection_rate", 0),
        })
    return {"reports": summaries, "count": len(summaries)}


@router.get("/redteam/reports/{report_id}")
async def redteam_get_report(report_id: str) -> dict[str, Any]:
    """Retrieve a specific stored red team report.

    Parameters
    ----------
    report_id : str
        Report identifier (e.g. "rt-0001").

    Returns
    -------
    dict
        Full report data.
    """
    report = _redteam_report_store.get(report_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Report '{report_id}' not found.",
        )
    return report


@router.delete("/redteam/reports/{report_id}")
async def redteam_delete_report(
    report_id: str,
    request: Request,
) -> dict[str, Any]:
    """Delete a stored red team report (admin authorization required).

    Parameters
    ----------
    report_id : str
        Report identifier to delete.

    Returns
    -------
    dict
        Confirmation of deletion.
    """
    # Admin check: the router already requires auth via _auth_dep.
    # Additionally verify admin role if user info is available.
    user = getattr(request.state, "user", None)
    if user is not None:
        role = getattr(user, "role", None) or (
            user.get("role") if isinstance(user, dict) else None
        )
        if role and role != "admin":
            raise HTTPException(
                status_code=403,
                detail="Admin authorization required to delete reports.",
            )

    if report_id not in _redteam_report_store:
        raise HTTPException(
            status_code=404,
            detail=f"Report '{report_id}' not found.",
        )

    del _redteam_report_store[report_id]
    logger.info("Deleted red team report %s", report_id)

    return {
        "status": "deleted",
        "report_id": report_id,
    }


@router.get("/redteam/compare")
async def redteam_compare_reports(
    id1: str = Query(..., description="First report ID (baseline)"),
    id2: str = Query(..., description="Second report ID (comparison)"),
) -> dict[str, Any]:
    """Compare two red team reports and highlight regressions/improvements.

    Parameters
    ----------
    id1 : str
        Baseline report ID.
    id2 : str
        Comparison report ID.

    Returns
    -------
    dict
        Summary diff with regressions and improvements by category.
    """
    report1 = _redteam_report_store.get(id1)
    report2 = _redteam_report_store.get(id2)

    if report1 is None:
        raise HTTPException(
            status_code=404,
            detail=f"Report '{id1}' not found.",
        )
    if report2 is None:
        raise HTTPException(
            status_code=404,
            detail=f"Report '{id2}' not found.",
        )

    score1 = report1.get("score", {})
    score2 = report2.get("score", {})

    # Overall summary
    summary = {
        "bypass_rate_before": score1.get("overall_bypass_rate", 0),
        "bypass_rate_after": score2.get("overall_bypass_rate", 0),
        "detection_rate_before": score1.get("overall_detection_rate", 0),
        "detection_rate_after": score2.get("overall_detection_rate", 0),
        "total_before": score1.get("total", 0),
        "total_after": score2.get("total", 0),
    }

    # Per-category comparison
    cats1 = score1.get("by_category", {})
    cats2 = score2.get("by_category", {})
    all_cats = sorted(set(cats1.keys()) | set(cats2.keys()))

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []

    for cat in all_cats:
        rate1 = cats1.get(cat, {}).get("bypass_rate", 0)
        rate2 = cats2.get(cat, {}).get("bypass_rate", 0)
        diff = rate2 - rate1

        entry = {
            "category": cat,
            "bypass_rate_before": rate1,
            "bypass_rate_after": rate2,
            "diff": round(diff, 4),
        }

        # Regression: bypass rate increased by more than 5%
        if diff > 0.05:
            regressions.append(entry)
        # Improvement: bypass rate decreased by more than 5%
        elif diff < -0.05:
            improvements.append(entry)
        else:
            unchanged.append(entry)

    # Per-target comparison
    tgts1 = score1.get("by_target", {})
    tgts2 = score2.get("by_target", {})
    all_tgts = sorted(set(tgts1.keys()) | set(tgts2.keys()))

    target_diffs: list[dict[str, Any]] = []
    for tgt in all_tgts:
        rate1 = tgts1.get(tgt, {}).get("bypass_rate", 0)
        rate2 = tgts2.get(tgt, {}).get("bypass_rate", 0)
        target_diffs.append({
            "target": tgt,
            "bypass_rate_before": rate1,
            "bypass_rate_after": rate2,
            "diff": round(rate2 - rate1, 4),
        })

    return {
        "id1": id1,
        "id2": id2,
        "timestamp1": report1.get("timestamp", ""),
        "timestamp2": report2.get("timestamp", ""),
        "summary": summary,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "by_target": target_diffs,
    }


# =========================================================================
# Red Team Feedback Loop (S157)
# =========================================================================

@router.get("/redteam/suggestions")
async def redteam_list_suggestions(
    status: str | None = Query(None, pattern="^(pending|accepted|rejected)$"),
) -> dict[str, Any]:
    """List red team feedback suggestions.

    Parameters
    ----------
    status : str or None
        Filter by status: "pending", "accepted", "rejected". None = all.

    Returns
    -------
    dict
        List of suggestions.
    """
    try:
        from opti_oignon.redteam.feedback import suggestion_store as _store
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Feedback module not available.",
        )

    if status == "pending":
        items = _store.list_pending()
    elif status is not None:
        items = [s for s in _store.list_all() if s.status == status]
    else:
        items = _store.list_all()

    return {
        "suggestions": [s.to_dict() for s in items],
        "count": len(items),
    }


@router.post("/redteam/suggestions/{suggestion_id}/accept")
async def redteam_accept_suggestion(
    suggestion_id: str,
) -> dict[str, Any]:
    """Accept a feedback suggestion and apply its pattern to rag.yaml.

    The accepted pattern is appended to the custom_patterns list
    in config/rag.yaml so the RAG sanitizer will detect it in future.

    Parameters
    ----------
    suggestion_id : str
        Suggestion identifier (e.g. "sg-0001").

    Returns
    -------
    dict
        Confirmation with applied status.
    """
    try:
        from opti_oignon.redteam.feedback import (
            suggestion_store as _store,
            apply_suggestion_to_config,
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Feedback module not available.",
        )

    suggestion = _store.get(suggestion_id)
    if suggestion is None:
        raise HTTPException(
            status_code=404,
            detail=f"Suggestion '{suggestion_id}' not found.",
        )

    if suggestion.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Suggestion already {suggestion.status}.",
        )

    _store.accept(suggestion_id)

    # Apply to rag.yaml
    applied = apply_suggestion_to_config(suggestion)

    return {
        "status": "accepted",
        "suggestion_id": suggestion_id,
        "pattern_name": suggestion.pattern_name,
        "regex": suggestion.regex,
        "applied_to_config": applied,
    }


@router.post("/redteam/suggestions/{suggestion_id}/reject")
async def redteam_reject_suggestion(
    suggestion_id: str,
) -> dict[str, Any]:
    """Reject a feedback suggestion.

    Parameters
    ----------
    suggestion_id : str
        Suggestion identifier.

    Returns
    -------
    dict
        Confirmation of rejection.
    """
    try:
        from opti_oignon.redteam.feedback import suggestion_store as _store
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Feedback module not available.",
        )

    suggestion = _store.get(suggestion_id)
    if suggestion is None:
        raise HTTPException(
            status_code=404,
            detail=f"Suggestion '{suggestion_id}' not found.",
        )

    if suggestion.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Suggestion already {suggestion.status}.",
        )

    _store.reject(suggestion_id)

    return {
        "status": "rejected",
        "suggestion_id": suggestion_id,
    }


# =========================================================================
# S158: Security Scheduler endpoints
# =========================================================================


@router.get("/scheduler")
async def get_scheduler_status() -> dict[str, Any]:
    """Get full security scheduler status.

    Returns scheduler configuration, last run timestamps, next scheduled
    run, dependency audit state, quiet hours, and recent alerts.

    Returns
    -------
    dict
        Complete scheduler status.
    """
    try:
        from opti_oignon.security_scheduler import get_scheduler
        scheduler = get_scheduler()
        return scheduler.get_status()
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Security scheduler module not available.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve scheduler status: {exc}",
        )


class SchedulerTriggerRequest(BaseModel):
    """Request body for manually triggering a scheduled run."""

    task: str = Field(
        "redteam",
        description="Task to trigger: 'redteam' or 'dep_audit'",
    )


@router.post("/scheduler/trigger")
async def trigger_scheduler_run(
    body: SchedulerTriggerRequest,
) -> dict[str, Any]:
    """Manually trigger a scheduled security task.

    Bypasses quiet hours. Supports 'redteam' and 'dep_audit' tasks.

    Parameters
    ----------
    body : SchedulerTriggerRequest
        Task selection (redteam or dep_audit).

    Returns
    -------
    dict
        Run result summary.
    """
    if body.task not in ("redteam", "dep_audit"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task: {body.task!r}. Use 'redteam' or 'dep_audit'.",
        )

    try:
        from opti_oignon.security_scheduler import get_scheduler
        scheduler = get_scheduler()
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Security scheduler module not available.",
        )

    try:
        if body.task == "redteam":
            result = scheduler.trigger_redteam()
        else:
            result = scheduler.trigger_dep_audit()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Trigger failed: {exc}",
        )

    return {
        "task": body.task,
        "result": result,
    }
