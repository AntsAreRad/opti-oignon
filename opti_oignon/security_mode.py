#!/usr/bin/env python3
"""
Security Mode System for Opti-Oignon (S126).

Implements the dual-mode Daily/Bulbe architecture:

  - **Daily**: frictionless everyday use with baseline S124/S125 security.
  - **Bulbe**: maximum security -- every layer active, wrapping the
    core like the protective bulb at the heart of an onion.

Mode storage uses dual sources for tamper detection:
  1. security.yaml > security_mode  (human-readable config)
  2. data/.security_mode_lock        (HMAC-SHA512 signed lockfile)

If the two sources disagree, the system fails secure (defaults to Bulbe).

Escalation (Daily -> Bulbe): single authenticated request, immediate.
Degradation (Bulbe -> Daily): 4-factor ceremony with cooling period.

All security derives from keys and human factors, never from code
obscurity (Kerckhoffs principle).
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import logging
import os
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODE_DAILY = "daily"
MODE_BULBE = "bulbe"
VALID_MODES = (MODE_DAILY, MODE_BULBE)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_LOCKFILE_PATH = _DATA_DIR / ".security_mode_lock"
_SECURITY_YAML = Path(__file__).resolve().parent / "config" / "security.yaml"
_DEFAULT_KEYFILE = _DATA_DIR / ".keyfile"

# Downgrade ceremony
DOWNGRADE_COOLDOWN_SECONDS = 300  # 5 minutes
DOWNGRADE_CODE_LENGTH = 6
DOWNGRADE_MAX_ATTEMPTS_PER_HOUR = 3
DOWNGRADE_LOCKOUT_AFTER_FAILURES = 5

# Session timeouts per mode (seconds)
SESSION_TIMEOUT = {
    MODE_DAILY: 3600,   # 60 min
    MODE_BULBE: 900,    # 15 min
}

# Rate limiting per mode (login attempts)
RATE_LIMIT = {
    MODE_DAILY: {"max_attempts": 5, "window_seconds": 300},
    MODE_BULBE: {"max_attempts": 3, "window_seconds": 300},
}

# Cookie SameSite per mode
COOKIE_SAMESITE = {
    MODE_DAILY: "Lax",
    MODE_BULBE: "Strict",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SecurityModeState:
    """Current security mode with metadata."""
    mode: str = MODE_DAILY
    timestamp: float = 0.0
    user_id: str = ""
    hmac_valid: bool = False
    sources_agree: bool = True
    lockfile_exists: bool = False


@dataclass
class DowngradeRequest:
    """Pending downgrade ceremony state."""
    request_id: str = ""
    visual_code: str = ""
    requested_at: float = 0.0
    expires_at: float = 0.0
    user_id: str = ""
    attempts: int = 0
    confirmed: bool = False


@dataclass
class ModePolicy:
    """Per-mode security policy settings."""
    mode: str = MODE_DAILY
    web_search_allowed: bool = True
    db_encryption_required: bool = False
    two_fa_required: bool = False
    plugin_allowlist_required: bool = False
    sandbox_bwrap_required: bool = False
    session_timeout: int = 3600
    backup_encryption_required: bool = False
    cookie_samesite: str = "Lax"
    tool_call_approval_required: bool = False
    rate_limit_max_attempts: int = 5
    rate_limit_window: int = 300
    bearer_auth_allowed: bool = True
    remote_access_allowed: bool = False

    @classmethod
    def for_mode(cls, mode: str) -> ModePolicy:
        """Build the policy for a given mode."""
        if mode == MODE_BULBE:
            return cls(
                mode=MODE_BULBE,
                web_search_allowed=False,
                db_encryption_required=True,
                two_fa_required=True,
                plugin_allowlist_required=True,
                sandbox_bwrap_required=True,
                session_timeout=SESSION_TIMEOUT[MODE_BULBE],
                backup_encryption_required=True,
                cookie_samesite=COOKIE_SAMESITE[MODE_BULBE],
                tool_call_approval_required=True,
                rate_limit_max_attempts=RATE_LIMIT[MODE_BULBE]["max_attempts"],
                rate_limit_window=RATE_LIMIT[MODE_BULBE]["window_seconds"],
                bearer_auth_allowed=False,
                # S133: Hardcoded False in Bulbe. Not from config.
                # This is defense layer 4 of 6.
                remote_access_allowed=False,
            )
        # Daily mode: read remote_access from security.yaml
        remote_enabled = cls._read_remote_access_config()
        return cls(
            mode=MODE_DAILY,
            web_search_allowed=True,
            db_encryption_required=False,
            two_fa_required=False,
            plugin_allowlist_required=False,
            sandbox_bwrap_required=False,
            session_timeout=SESSION_TIMEOUT[MODE_DAILY],
            backup_encryption_required=False,
            cookie_samesite=COOKIE_SAMESITE[MODE_DAILY],
            tool_call_approval_required=False,
            rate_limit_max_attempts=RATE_LIMIT[MODE_DAILY]["max_attempts"],
            rate_limit_window=RATE_LIMIT[MODE_DAILY]["window_seconds"],
            bearer_auth_allowed=True,
            remote_access_allowed=remote_enabled,
        )

    @staticmethod
    def _read_remote_access_config() -> bool:
        """Read remote_access.enabled from security.yaml for Daily mode."""
        try:
            if _SECURITY_YAML.exists():
                with open(_SECURITY_YAML, encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh) or {}
                ra = cfg.get("remote_access", {})
                if isinstance(ra, dict):
                    return bool(ra.get("enabled", False))
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# HMAC helpers
# ---------------------------------------------------------------------------

def _load_signing_key():
    """Load the signing key from the keyfile.

    S129: Returns SecureBytes when available (from load_keyfile()),
    falls back to raw bytes from file if keyfile module is unavailable.
    """
    try:
        from opti_oignon.encryption import load_keyfile
        key, _salt, _kdf = load_keyfile()
        return key  # SecureBytes (S129)
    except Exception:
        pass
    # Direct keyfile read fallback
    try:
        if _DEFAULT_KEYFILE.exists():
            raw = _DEFAULT_KEYFILE.read_bytes()
            # First 32 bytes are the key in our keyfile format
            if len(raw) >= 32:
                return raw[:32]
    except Exception:
        pass
    return None


def _extract_key_bytes(key) -> bytes:
    """Extract raw bytes from a key (SecureBytes or plain bytes).

    S129: Helper for HMAC operations that require raw bytes.
    """
    if hasattr(key, "as_bytes"):
        return key.as_bytes()
    return key


_BACKENDS_YAML = Path(__file__).resolve().parent / "config" / "backends.yaml"


def _default_backend() -> str | None:
    """The backend that would serve a model load, read from backends.yaml.

    A broken or missing read yields None rather than a manufactured verdict: the
    preflight that consults this must not pin a host in Daily on the strength of
    its own inability to read a config file. That matches the stance the rest of
    this surface takes -- a machinery failure is not a security decision.
    """
    try:
        with open(_BACKENDS_YAML, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        backend = cfg.get("default_backend")
        return str(backend) if backend else None
    except Exception as exc:  # noqa: BLE001 - an unreadable config is not a verdict
        logger.error("the default backend could not be read: %s", exc)
        return None


def _compute_lockfile_hmac(
    mode: str, timestamp: float, user_id: str, key
) -> str:
    """Compute HMAC-SHA512 over lockfile fields.

    The signature covers mode||timestamp||user_id so that any
    field change invalidates the HMAC.  Security derives from
    the signing key, not from the format (Kerckhoffs).

    S129: Accepts SecureBytes or raw bytes.
    """
    raw_key = _extract_key_bytes(key)
    message = f"{mode}||{timestamp}||{user_id}".encode()
    return _hmac.new(raw_key, message, hashlib.sha512).hexdigest()


# ---------------------------------------------------------------------------
# Lockfile I/O
# ---------------------------------------------------------------------------

def _write_lockfile(mode: str, user_id: str, key) -> float:
    """Write the HMAC-signed lockfile.  Returns the timestamp used.

    S129: Accepts SecureBytes or raw bytes for key.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    hmac_sig = _compute_lockfile_hmac(mode, ts, user_id, key)
    content = (
        f"MODE:{mode}\n"
        f"TIMESTAMP:{ts}\n"
        f"USER_ID:{user_id}\n"
        f"HMAC:{hmac_sig}\n"
    )
    # The lockfile is made read-only (0400) at the end of this function to
    # deter tampering. A previously written lockfile therefore blocks any
    # rewrite (e.g. the Bulbe -> Daily downgrade, or a re-escalation) with
    # PermissionError, so restore owner write permission before overwriting.
    if _LOCKFILE_PATH.exists():
        try:
            os.chmod(_LOCKFILE_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except OSError:
            pass
    _LOCKFILE_PATH.write_text(content, encoding="utf-8")
    try:
        os.chmod(_LOCKFILE_PATH, stat.S_IRUSR)  # chmod 400
    except OSError:
        pass
    return ts


def _read_lockfile() -> dict[str, str]:
    """Parse the lockfile into a dict of fields.

    Returns empty dict if the lockfile is missing or unreadable.
    """
    try:
        if not _LOCKFILE_PATH.exists():
            return {}
        text = _LOCKFILE_PATH.read_text(encoding="utf-8")
        result: dict[str, str] = {}
        for line in text.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                result[k.strip()] = v.strip()
        return result
    except Exception as exc:
        logger.warning("Failed to read lockfile: %s", exc)
        return {}


def _verify_lockfile(fields: dict[str, str], key) -> bool:
    """Verify the HMAC in the lockfile fields.

    S129: Accepts SecureBytes or raw bytes for key.
    """
    mode = fields.get("MODE", "")
    ts_str = fields.get("TIMESTAMP", "0")
    user_id = fields.get("USER_ID", "")
    stored_hmac = fields.get("HMAC", "")
    if not mode or not stored_hmac:
        return False
    try:
        ts = float(ts_str)
    except ValueError:
        return False
    expected = _compute_lockfile_hmac(mode, ts, user_id, key)
    return _hmac.compare_digest(expected, stored_hmac)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _read_yaml_mode() -> str:
    """Read security_mode from security.yaml, fail-secure on a bad value.

    A recognized mode is returned as-is. Any value outside VALID_MODES --
    an arbitrary or malformed mode string from a hand-edited file -- is
    never propagated: it resolves to the restrictive mode (Bulbe). An
    unreadable or corrupt file (an exception while parsing) likewise
    resolves to Bulbe rather than to the permissive Daily mode. A missing
    file is the documented default of a fresh install and stays Daily.
    """
    try:
        if _SECURITY_YAML.exists():
            with open(_SECURITY_YAML, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            mode = cfg.get("security_mode", MODE_DAILY)
            if mode in VALID_MODES:
                return mode
            logger.warning(
                "Unrecognized security mode %r in YAML; failing secure "
                "to Bulbe.",
                mode,
            )
            return MODE_BULBE
    except Exception as exc:
        logger.warning(
            "Failed to read security mode from YAML: %s; failing secure "
            "to Bulbe.",
            exc,
        )
        return MODE_BULBE
    return MODE_DAILY


def _write_yaml_mode(mode: str) -> None:
    """Update security_mode in security.yaml, preserving other keys."""
    try:
        cfg: dict[str, Any] = {}
        if _SECURITY_YAML.exists():
            with open(_SECURITY_YAML, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        cfg["security_mode"] = mode
        with open(_SECURITY_YAML, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)
    except Exception as exc:
        logger.error("Failed to write security mode to YAML: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

def _audit_log(event: str, severity: str = "INFO", **details: Any) -> None:
    """Log a security audit event.

    Also persists to the security audit DB if available,
    and to the hash-chain signed audit log (S130).
    """
    log_data = {
        "event": event,
        "severity": severity,
        "timestamp": time.time(),
        **details,
    }
    if severity == "CRITICAL":
        logger.critical("SECURITY AUDIT [%s]: %s", event, log_data)
    elif severity == "WARNING":
        logger.warning("SECURITY AUDIT [%s]: %s", event, log_data)
    else:
        logger.info("SECURITY AUDIT [%s]: %s", event, log_data)

    # Persist to audit trail if auth module available
    try:
        from opti_oignon.auth import auth_manager
        if auth_manager and hasattr(auth_manager, "_log_audit_event"):
            auth_manager._log_audit_event(
                event_type=event,
                details=log_data,
            )
    except Exception:
        pass

    # S130: Forward to hash-chain signed audit log
    try:
        from opti_oignon.signed_audit_log import chain_log
        chain_log(
            event_type=event,
            source="security_mode",
            action=event,
            severity=severity,
            **details,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SecurityModeManager
# ---------------------------------------------------------------------------

class SecurityModeManager:
    """Manages the Daily/Bulbe dual-mode security system.

    The manager enforces:
    - Dual-source tamper detection (YAML + HMAC-signed lockfile)
    - Fail-secure: disagreement defaults to Bulbe
    - Instant escalation, ceremony-gated degradation
    """

    def __init__(self) -> None:
        self._pending_downgrade: DowngradeRequest | None = None
        self._downgrade_attempts: list[float] = []  # timestamps
        self._cached_mode: str | None = None
        self._cached_policy: ModePolicy | None = None

    # -- Current state -------------------------------------------------------

    def get_current_mode(self) -> str:
        """Return the current security mode, fail-secure on mismatch."""
        if self._cached_mode is not None:
            return self._cached_mode

        yaml_mode = _read_yaml_mode()
        lockfile = _read_lockfile()
        key = _load_signing_key()

        if not lockfile:
            # No lockfile: trust YAML but create lockfile on next write
            self._cached_mode = yaml_mode
            return yaml_mode

        lockfile_mode = lockfile.get("MODE", "")

        # Verify HMAC if we have a key
        if key:
            if not _verify_lockfile(lockfile, key):
                logger.critical(
                    "Lockfile HMAC verification FAILED -- possible tampering. "
                    "Failing secure to Bulbe mode."
                )
                _audit_log(
                    "lockfile_hmac_failure",
                    severity="CRITICAL",
                    yaml_mode=yaml_mode,
                    lockfile_mode=lockfile_mode,
                )
                self._cached_mode = MODE_BULBE
                return MODE_BULBE

        # Sources must agree
        if yaml_mode != lockfile_mode:
            logger.warning(
                "Security mode mismatch: YAML=%s, lockfile=%s. "
                "Failing secure to Bulbe.",
                yaml_mode, lockfile_mode,
            )
            _audit_log(
                "security_mode_mismatch",
                severity="WARNING",
                yaml_mode=yaml_mode,
                lockfile_mode=lockfile_mode,
            )
            self._cached_mode = MODE_BULBE
            return MODE_BULBE

        self._cached_mode = yaml_mode
        return yaml_mode

    def get_state(self) -> SecurityModeState:
        """Return full state including validation details."""
        yaml_mode = _read_yaml_mode()
        lockfile = _read_lockfile()
        key = _load_signing_key()

        state = SecurityModeState(
            mode=self.get_current_mode(),
            lockfile_exists=bool(lockfile),
        )

        if lockfile:
            state.user_id = lockfile.get("USER_ID", "")
            try:
                state.timestamp = float(lockfile.get("TIMESTAMP", "0"))
            except ValueError:
                state.timestamp = 0.0

            lockfile_mode = lockfile.get("MODE", "")
            state.sources_agree = (yaml_mode == lockfile_mode)

            if key:
                state.hmac_valid = _verify_lockfile(lockfile, key)
            else:
                state.hmac_valid = False

        return state

    def get_policy(self) -> ModePolicy:
        """Get the security policy for the current mode."""
        mode = self.get_current_mode()
        if self._cached_policy and self._cached_policy.mode == mode:
            return self._cached_policy
        self._cached_policy = ModePolicy.for_mode(mode)
        return self._cached_policy

    def is_bulbe(self) -> bool:
        """Check if currently in Bulbe mode."""
        return self.get_current_mode() == MODE_BULBE

    def is_daily(self) -> bool:
        """Check if currently in Daily mode."""
        return self.get_current_mode() == MODE_DAILY

    def invalidate_cache(self) -> None:
        """Force re-read from disk on next access."""
        self._cached_mode = None
        self._cached_policy = None

    # -- Escalation (Daily -> Bulbe) ----------------------------------------

    @staticmethod
    def _fortress_blockers() -> list[str]:
        """What Bulbe would refuse on, asked BEFORE the mode changes.

        Both imports are deferred, and both failures are swallowed on purpose.
        The signing module reads the mode back out of this one, and a module that
        cannot be IMPORTED is a machinery failure, not a verdict. A broken tree
        must not be able to manufacture a refusal here either: it would only pin
        the host in Daily, which buys no security at all.
        """
        blockers: list[str] = []
        try:
            from opti_oignon.pqc_signatures import signing_blockers

            blockers.extend(signing_blockers())
        except Exception as exc:  # noqa: BLE001 - a broken import is not a verdict
            logger.error("the signing readiness could not be determined: %s", exc)
            return []

        try:
            from opti_oignon.model_provenance import (
                SCHEME_PQC,
                manifest_seal_scheme,
            )

            scheme = manifest_seal_scheme()
        except Exception as exc:  # noqa: BLE001 - a broken import is not a verdict
            logger.error("the manifest seal could not be read: %s", exc)
            return blockers

        if scheme is not None and scheme != SCHEME_PQC:
            blockers.append(
                f"the model provenance manifest is sealed with {scheme!r}, "
                f"which a fortress reads as a downgrade and refuses. Every "
                f"model on this host would be rejected."
            )

        blockers.extend(SecurityModeManager._provenance_blockers())
        return blockers

    @staticmethod
    def _provenance_blockers() -> list[str]:
        """The plainest brick of all: a gated backend and NO manifest.

        The downgrade check above fires only when a manifest EXISTS with the
        wrong scheme. It says nothing about the host that has no manifest at
        all -- which is every fresh install. On a fortress the load seam
        enforces, and a model with no pin is refused as unpinned, so a host
        that escalates here loads nothing. This closes that half. The two are
        disjoint by construction: one needs a manifest present, the other needs
        it absent, and neither can hide a failure of the other.

        Scoped to the brick on purpose. A backend whose load seam never calls
        the gate (Ollama, llama-server today) is not bricked by escalation, so
        it earns no blocker even though loading unverified weights in a fortress
        is its own weakness. That weakness is documented as a gap; turning it
        into an escalation refusal would buy no security and only pin the host
        in Daily -- the exact anti-pattern the rest of this preflight avoids.
        """
        backend = _default_backend()
        try:
            from opti_oignon.model_provenance import (
                backend_enforces_provenance,
                load_manifest,
            )
        except Exception as exc:  # noqa: BLE001 - a broken import is not a verdict
            logger.error("provenance readiness could not be determined: %s", exc)
            return []

        if not backend_enforces_provenance(backend):
            return []

        manifest = load_manifest()
        if manifest and manifest.get("entries"):
            return []

        return [
            f"the load backend {backend!r} verifies model provenance and this "
            f"host has no provenance manifest. In Bulbe every model load is "
            f"refused as unpinned. Enrol the models on disk first."
        ]

    def escalate_to_bulbe(
        self, user_id: str, force: bool = False
    ) -> dict[str, Any]:
        """Escalate from Daily to Bulbe mode. Immediate, no ceremony.

        Adding security is NOT always safe, and saying so was the bug. Bulbe
        requires the post-quantum signature -- there it is a property of the
        mode, like the socket bind -- so a host with no signing key becomes, the
        instant it escalates, a host that refuses every model it owns and every
        backup it exports. Nothing crashes; it simply stops working, and the
        operator never asked for that.

        The readiness is checked HERE because this is the one place where
        refusing is free: staying in Daily is the status quo, not a brick. It is
        deliberately NOT checked at the boot. A critical boot check would abort
        the lifespan and take down the very endpoints that mint the key and
        re-seal the manifest -- a check must never remove the exit it is telling
        you to take.

        ``force`` escalates regardless. The emergency stop passes it: a panic
        button that can say no is not a panic button.
        """
        current = self.get_current_mode()
        if current == MODE_BULBE:
            return {
                "success": True,
                "mode": MODE_BULBE,
                "message": "Already in Bulbe mode",
                "changed": False,
            }

        key = _load_signing_key()
        if not key:
            return {
                "success": False,
                "error": "no_signing_key",
                "message": (
                    "Cannot enter Bulbe mode without an encryption key. "
                    "Run encryption setup first."
                ),
            }

        if not force:
            blockers = self._fortress_blockers()
            if blockers:
                _audit_log(
                    "security_mode_escalation_refused",
                    severity="WARNING",
                    user_id=user_id,
                    blockers=blockers,
                )
                return {
                    "success": False,
                    "error": "fortress_not_ready",
                    "message": (
                        "Bulbe requires the post-quantum signature. This host "
                        "cannot produce or verify one yet, so it would refuse "
                        "every model it owns and every backup it exports."
                    ),
                    "blockers": blockers,
                    "remedy": [
                        "POST /api/security/pqc/generate-keys",
                        "POST /api/security/pqc/enroll-models",
                        "POST /api/security/pqc/reseal-manifest",
                    ],
                    "changed": False,
                }

        # Write to both sources atomically
        _write_yaml_mode(MODE_BULBE)
        ts = _write_lockfile(MODE_BULBE, user_id, key)
        self.invalidate_cache()

        _audit_log(
            "security_mode_escalated",
            severity="INFO",
            user_id=user_id,
            from_mode=MODE_DAILY,
            to_mode=MODE_BULBE,
        )

        return {
            "success": True,
            "mode": MODE_BULBE,
            "message": "Escalated to Bulbe mode",
            "changed": True,
            "timestamp": ts,
        }

    # -- Degradation ceremony (Bulbe -> Daily) ------------------------------

    def request_downgrade(self, user_id: str) -> dict[str, Any]:
        """Start the downgrade ceremony.  Returns pending state.

        Step 1 of the multi-factor ceremony:
        - Generates a 6-digit visual code
        - Starts the 5-minute cooldown
        - Code must be displayed in DOM only (not in API JSON)
        """
        current = self.get_current_mode()
        if current == MODE_DAILY:
            return {
                "success": True,
                "pending": False,
                "message": "Already in Daily mode",
            }

        # Rate limit: max attempts per hour
        now = time.time()
        self._downgrade_attempts = [
            t for t in self._downgrade_attempts
            if now - t < 3600
        ]
        if len(self._downgrade_attempts) >= DOWNGRADE_MAX_ATTEMPTS_PER_HOUR:
            _audit_log(
                "downgrade_rate_limited",
                severity="WARNING",
                user_id=user_id,
            )
            return {
                "success": False,
                "error": "rate_limited",
                "message": "Too many downgrade attempts. Try again later.",
            }

        # Generate the visual code
        code = "".join(
            [str(secrets.randbelow(10)) for _ in range(DOWNGRADE_CODE_LENGTH)]
        )

        request_id = secrets.token_urlsafe(16)
        # Grace window after the cooldown during which confirmation is accepted.
        # Kept generous (10 min) because the frontend countdown is a browser
        # timer that gets throttled when the tab is backgrounded; a short window
        # let the request expire before the user could confirm.
        expires_at = now + DOWNGRADE_COOLDOWN_SECONDS + 600

        self._pending_downgrade = DowngradeRequest(
            request_id=request_id,
            visual_code=code,
            requested_at=now,
            expires_at=expires_at,
            user_id=user_id,
            attempts=0,
        )

        _audit_log(
            "downgrade_requested",
            severity="WARNING",
            user_id=user_id,
            request_id=request_id,
        )

        return {
            "success": True,
            "pending": True,
            "request_id": request_id,
            "requested_at": now,
            "cooldown_seconds": DOWNGRADE_COOLDOWN_SECONDS,
            "expires_at": expires_at,
            # NOTE: visual_code is NOT included here.
            # It is injected into the DOM by the frontend route handler,
            # invisible to the LLM token stream.
            # The _visual_code field is available via get_pending_visual_code()
            # for the template renderer only.
        }

    def get_pending_visual_code(self) -> str | None:
        """Return the pending visual code for DOM injection.

        This is called by the HTML template renderer, NOT the JSON API.
        The code appears in the DOM where a human can read it but the
        LLM (which only sees the JSON response) cannot.
        """
        if self._pending_downgrade and not self._pending_downgrade.confirmed:
            return self._pending_downgrade.visual_code
        return None

    def get_pending_downgrade(self) -> dict[str, Any] | None:
        """Return pending downgrade state (without the visual code)."""
        req = self._pending_downgrade
        if not req or req.confirmed:
            return None
        now = time.time()
        if now > req.expires_at:
            self._pending_downgrade = None
            return None
        elapsed = now - req.requested_at
        remaining = max(0, DOWNGRADE_COOLDOWN_SECONDS - elapsed)
        return {
            "pending": True,
            "request_id": req.request_id,
            "requested_at": req.requested_at,
            "cooldown_remaining": remaining,
            "cooldown_complete": remaining <= 0,
            "expires_at": req.expires_at,
            "attempts": req.attempts,
        }

    def cancel_downgrade(self) -> dict[str, Any]:
        """Cancel a pending downgrade request."""
        if self._pending_downgrade:
            _audit_log(
                "downgrade_cancelled",
                severity="INFO",
                user_id=self._pending_downgrade.user_id,
                request_id=self._pending_downgrade.request_id,
            )
            self._pending_downgrade = None
            return {"success": True, "message": "Downgrade cancelled"}
        return {"success": True, "message": "No pending downgrade"}

    def confirm_downgrade(
        self,
        user_id: str,
        request_id: str,
        visual_code: str,
        password: str,
        two_fa_code: str | None = None,
    ) -> dict[str, Any]:
        """Confirm the downgrade ceremony.

        Requires ALL of:
        a. Valid session (checked by caller / route auth)
        b. Matching request_id
        c. Correct visual code (human typed from screen)
        d. Current password (verified by caller)
        e. 2FA code if 2FA is active (verified by caller)
        f. Cooldown elapsed (>= 5 minutes since request)

        The password and 2FA verification are done by the caller
        (API route) before calling this method.  This method only
        checks the ceremony-specific factors.
        """
        req = self._pending_downgrade
        if not req:
            return {
                "success": False,
                "error": "no_pending_request",
                "message": "No pending downgrade request",
            }

        now = time.time()

        # Expired?
        if now > req.expires_at:
            self._pending_downgrade = None
            return {
                "success": False,
                "error": "expired",
                "message": "Downgrade request expired",
            }

        # Track attempts
        req.attempts += 1
        self._downgrade_attempts.append(now)

        # Lockout after too many failures
        if req.attempts > DOWNGRADE_LOCKOUT_AFTER_FAILURES:
            _audit_log(
                "downgrade_lockout",
                severity="CRITICAL",
                user_id=user_id,
                attempts=req.attempts,
            )
            self._pending_downgrade = None
            return {
                "success": False,
                "error": "lockout",
                "message": "Too many failed attempts. Request cancelled.",
            }

        # Check request_id
        if not _hmac.compare_digest(request_id, req.request_id):
            _audit_log(
                "downgrade_invalid_request_id",
                severity="WARNING",
                user_id=user_id,
            )
            return {
                "success": False,
                "error": "invalid_request",
                "message": "Invalid downgrade request",
            }

        # Check user_id matches
        if user_id != req.user_id:
            return {
                "success": False,
                "error": "user_mismatch",
                "message": "Downgrade must be confirmed by the requesting user",
            }

        # Check cooldown (server-side timing)
        elapsed = now - req.requested_at
        if elapsed < DOWNGRADE_COOLDOWN_SECONDS:
            remaining = DOWNGRADE_COOLDOWN_SECONDS - elapsed
            return {
                "success": False,
                "error": "cooldown_active",
                "message": f"Cooldown active. {remaining:.0f}s remaining.",
                "cooldown_remaining": remaining,
            }

        # Check visual code
        if not _hmac.compare_digest(visual_code, req.visual_code):
            _audit_log(
                "downgrade_invalid_code",
                severity="WARNING",
                user_id=user_id,
                attempt=req.attempts,
            )
            return {
                "success": False,
                "error": "invalid_code",
                "message": "Invalid confirmation code",
            }

        # All checks passed -- perform the downgrade
        key = _load_signing_key()
        if not key:
            return {
                "success": False,
                "error": "no_signing_key",
                "message": "Cannot downgrade without signing key",
            }

        _write_yaml_mode(MODE_DAILY)
        ts = _write_lockfile(MODE_DAILY, user_id, key)
        self.invalidate_cache()
        req.confirmed = True
        self._pending_downgrade = None

        _audit_log(
            "security_mode_degraded",
            severity="CRITICAL",
            user_id=user_id,
            from_mode=MODE_BULBE,
            to_mode=MODE_DAILY,
        )

        return {
            "success": True,
            "mode": MODE_DAILY,
            "message": "Downgraded to Daily mode",
            "changed": True,
            "timestamp": ts,
        }

    # -- Status for API / UI ------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return full status dict for the API."""
        state = self.get_state()
        policy = self.get_policy()
        pending = self.get_pending_downgrade()

        return {
            "mode": state.mode,
            "timestamp": state.timestamp,
            "lockfile_exists": state.lockfile_exists,
            "sources_agree": state.sources_agree,
            "hmac_valid": state.hmac_valid,
            "policy": {
                "web_search_allowed": policy.web_search_allowed,
                "db_encryption_required": policy.db_encryption_required,
                "two_fa_required": policy.two_fa_required,
                "plugin_allowlist_required": policy.plugin_allowlist_required,
                "sandbox_bwrap_required": policy.sandbox_bwrap_required,
                "session_timeout": policy.session_timeout,
                "backup_encryption_required": policy.backup_encryption_required,
                "cookie_samesite": policy.cookie_samesite,
                "tool_call_approval_required": policy.tool_call_approval_required,
                "rate_limit_max_attempts": policy.rate_limit_max_attempts,
                "rate_limit_window": policy.rate_limit_window,
                "bearer_auth_allowed": policy.bearer_auth_allowed,
                "remote_access_allowed": policy.remote_access_allowed,
            },
            "pending_downgrade": pending,
        }

    # -- Direct mode set (for initialization / tests) -----------------------

    def initialize_mode(self, mode: str = MODE_DAILY, user_id: str = "system") -> bool:
        """Set mode directly (for first-time setup or tests).

        This bypasses the ceremony and should only be called
        during initial setup or in test fixtures.
        """
        if mode not in VALID_MODES:
            return False
        key = _load_signing_key()
        _write_yaml_mode(mode)
        if key:
            _write_lockfile(mode, user_id, key)
        self.invalidate_cache()
        return True


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

security_mode_manager = SecurityModeManager()


def get_current_mode() -> str:
    """Convenience: return the current security mode."""
    return security_mode_manager.get_current_mode()


def get_policy() -> ModePolicy:
    """Convenience: return the current mode policy."""
    return security_mode_manager.get_policy()


def is_bulbe() -> bool:
    """Convenience: check if in Bulbe mode."""
    return security_mode_manager.is_bulbe()


def is_daily() -> bool:
    """Convenience: check if in Daily mode."""
    return security_mode_manager.is_daily()
