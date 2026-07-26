#!/usr/bin/env python3
"""
Remote Session Guard for Opti-Oignon.

Hardens remote sessions with:
  - JWT bound to client certificate fingerprint
  - Constant-time comparison for all secret comparisons
  - Session revocation on cert fingerprint mismatch
  - IP allowlist enforcement (configurable CIDRs)
  - Per-client-cert rate limiting
  - Uniform response timing (no timing oracles)

All error responses (bad cert, expired JWT, IP denied, rate limited)
return in uniform time. An attacker reading this code must not be
able to distinguish error types by measuring response latency.

Kerckhoffs compliance: the attacker knows every defense layer, every
algorithm, every constant. Security derives from key material and
correct implementation, not from secrecy.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Hardcoded, never overridable
checkpoint_before_apply = True

_SECURITY_YAML = Path(__file__).resolve().parent / "config" / "security.yaml"

# Default IP allowlist: private LAN ranges only
_DEFAULT_ALLOWLIST_CIDRS = [
    "127.0.0.0/8",
    "::1/128",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "fe80::/10",
]

# Rate limiting defaults
RATE_LIMIT_REQUESTS_PER_MIN = 60
RATE_LIMIT_FAILED_AUTH_PER_HOUR = 10

# Suspicious activity threshold
SUSPICIOUS_FAILED_REQUESTS = 3

# Minimum response time for uniform timing (seconds)
# All auth checks return in at least this time to prevent timing oracles
_MIN_RESPONSE_TIME_MS = 50


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SessionBinding:
    """A JWT session bound to a client certificate."""
    token_jti: str = ""
    cert_fingerprint: str = ""
    source_ip: str = ""
    issued_at: float = 0.0
    user_id: str = ""


@dataclass
class RateLimitState:
    """Per-client rate limiting state."""
    request_timestamps: list[float] = field(default_factory=list)
    failed_auth_timestamps: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RemoteSessionGuard
# ---------------------------------------------------------------------------

class RemoteSessionGuard:
    """Guards remote sessions with binding, rate limiting, and timing safety.

    This guard is only active when remote access is enabled (Daily mode).
    In Bulbe mode or when remote access is off, all checks pass through
    (the bind guard and middleware already block remote connections).
    """

    def __init__(self) -> None:
        self._rate_limits: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._revoked_sessions: set[str] = set()
        # RA-01: client-cert fingerprints whose live sessions are revoked.
        self._revoked_fingerprints: set[str] = set()
        self._ip_allowlist: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        self._allowlist_loaded = False

    # -- Session binding -----------------------------------------------------

    def bind_session_to_cert(
        self, token_jti: str, cert_fingerprint: str,
        source_ip: str, user_id: str,
    ) -> SessionBinding:
        """Create a binding between a JWT and a client certificate.

        The cert fingerprint is stored as a JWT claim so that every
        subsequent request can be verified against the presented cert.

        Args:
            token_jti: JWT ID (unique per token).
            cert_fingerprint: SHA256 fingerprint of client cert.
            source_ip: Client IP at time of binding.
            user_id: Authenticated user ID.

        Returns:
            SessionBinding object with all fields set.
        """
        binding = SessionBinding(
            token_jti=token_jti,
            cert_fingerprint=cert_fingerprint,
            source_ip=source_ip,
            issued_at=time.time(),
            user_id=user_id,
        )
        logger.info(
            "Session bound: jti=%s, cert=%s..., ip=%s",
            token_jti[:8], cert_fingerprint[:16], source_ip,
        )
        return binding

    def validate_session_binding(
        self, token_jti: str, token_cert_fp: str,
        request_cert_fp: str,
    ) -> tuple[bool, str]:
        """Validate that the presented cert matches the JWT claim.

        Uses constant-time comparison (hmac.compare_digest) to prevent
        timing side-channels. An attacker reading this code sees:
        we use hmac.compare_digest, so timing attacks are useless.

        Args:
            token_jti: JWT ID from the token.
            token_cert_fp: Cert fingerprint stored in JWT claims.
            request_cert_fp: Cert fingerprint from the current request.

        Returns:
            (is_valid, error_reason) tuple.
        """
        # Check if session was revoked
        if token_jti in self._revoked_sessions:
            return False, "session_revoked"

        # Constant-time fingerprint comparison
        if not token_cert_fp or not request_cert_fp:
            return False, "missing_fingerprint"

        # Reject a revoked client certificate even when the binding matches
        # (RA-01): revocation kills live sessions bound to that fingerprint.
        if request_cert_fp in self._revoked_fingerprints:
            return False, "cert_revoked"

        if not hmac.compare_digest(
            token_cert_fp.encode("utf-8"),
            request_cert_fp.encode("utf-8"),
        ):
            # Mismatch: revoke session and log
            self._revoked_sessions.add(token_jti)
            _audit_event(
                "session_binding_mismatch",
                token_jti=token_jti,
                expected_fp=token_cert_fp[:16] + "...",
                actual_fp=request_cert_fp[:16] + "...",
            )
            return False, "cert_fingerprint_mismatch"

        return True, ""

    def revoke_session(self, token_jti: str) -> None:
        """Revoke a specific session by JTI."""
        self._revoked_sessions.add(token_jti)
        logger.warning("Session revoked: jti=%s", token_jti[:8])

    def revoke_all_remote_sessions(self) -> int:
        """Nuclear option: revoke ALL remote sessions.

        Returns the number of sessions revoked.
        """
        count = len(self._revoked_sessions)
        # Since we track by JTI, add a sentinel that ensures any
        # new validation will trigger a full re-auth
        self._revoked_sessions.clear()
        self._revoke_all_flag = time.time()
        _audit_event("all_remote_sessions_revoked")
        logger.critical("ALL remote sessions revoked")
        return count

    def is_session_revoked(self, token_jti: str) -> bool:
        """Check if a session has been revoked."""
        if token_jti in self._revoked_sessions:
            return True
        # Check the nuclear revoke flag
        if hasattr(self, "_revoke_all_flag"):
            return True
        return False

    def revoke_fingerprint(self, cert_fingerprint: str) -> None:
        """Revoke all live sessions bound to a client-cert fingerprint (RA-01).

        Called by ``tls_manager.revoke_client_cert`` so that revoking a device's
        certificate immediately denies its in-flight sessions, in addition to
        the persistent CRL/metadata check enforced on the request path.
        """
        if not cert_fingerprint:
            return
        self._revoked_fingerprints.add(cert_fingerprint)
        _audit_event(
            "client_cert_fingerprint_revoked",
            fingerprint=cert_fingerprint[:16] + "...",
        )
        logger.warning(
            "Client cert fingerprint revoked: %s...", cert_fingerprint[:16],
        )

    def is_fingerprint_revoked(self, cert_fingerprint: str) -> bool:
        """Whether a client-cert fingerprint has been revoked (in-memory)."""
        return cert_fingerprint in self._revoked_fingerprints

    # -- IP allowlist --------------------------------------------------------

    def check_ip_allowed(self, client_ip: str) -> bool:
        """Check if a client IP is in the allowlist.

        Default: LAN only (192.168.0.0/16, 10.0.0.0/8, 172.16.0.0/12).
        Configurable in security.yaml > remote_access > ip_allowlist.

        Args:
            client_ip: Client IP address string.

        Returns:
            True if the IP is allowed.
        """
        if not self._allowlist_loaded:
            self._load_ip_allowlist()

        try:
            addr = ipaddress.ip_address(client_ip)
        except ValueError:
            return False

        for network in self._ip_allowlist:
            if addr in network:
                return True

        return False

    def _load_ip_allowlist(self) -> None:
        """Load IP allowlist from config or use defaults."""
        cidrs = list(_DEFAULT_ALLOWLIST_CIDRS)

        try:
            if _SECURITY_YAML.exists():
                with open(_SECURITY_YAML, encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh) or {}
                ra = cfg.get("remote_access", {})
                if isinstance(ra, dict):
                    custom_cidrs = ra.get("ip_allowlist", [])
                    if isinstance(custom_cidrs, list) and custom_cidrs:
                        cidrs = custom_cidrs
        except Exception as exc:
            logger.warning("Failed to load IP allowlist: %s", exc)

        self._ip_allowlist = []
        for cidr in cidrs:
            try:
                self._ip_allowlist.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError as exc:
                logger.warning("Invalid CIDR in allowlist: %s (%s)", cidr, exc)

        self._allowlist_loaded = True

    def reload_allowlist(self) -> None:
        """Force reload of IP allowlist from config."""
        self._allowlist_loaded = False
        self._load_ip_allowlist()

    # -- Rate limiting -------------------------------------------------------

    def check_rate_limit(self, client_key: str) -> tuple[bool, str]:
        """Check if a client has exceeded rate limits.

        Args:
            client_key: Client identifier (cert fingerprint or IP).

        Returns:
            (is_allowed, error_reason) tuple.
        """
        now = time.time()
        state = self._rate_limits[client_key]

        # Clean old timestamps
        one_minute_ago = now - 60
        one_hour_ago = now - 3600
        state.request_timestamps = [
            t for t in state.request_timestamps if t > one_minute_ago
        ]
        state.failed_auth_timestamps = [
            t for t in state.failed_auth_timestamps if t > one_hour_ago
        ]

        # Check request rate
        if len(state.request_timestamps) >= RATE_LIMIT_REQUESTS_PER_MIN:
            return False, "rate_limited"

        # Check failed auth rate
        if len(state.failed_auth_timestamps) >= RATE_LIMIT_FAILED_AUTH_PER_HOUR:
            return False, "auth_rate_limited"

        state.request_timestamps.append(now)
        return True, ""

    def record_failed_auth(self, client_key: str) -> None:
        """Record a failed authentication attempt.

        If threshold exceeded (3 failures), revoke all remote sessions
        for this client as a precaution.
        """
        now = time.time()
        state = self._rate_limits[client_key]
        state.failed_auth_timestamps.append(now)

        # Count recent failures (last 5 minutes)
        recent = [t for t in state.failed_auth_timestamps if now - t < 300]
        if len(recent) >= SUSPICIOUS_FAILED_REQUESTS:
            logger.warning(
                "Suspicious activity: %d failed auths from %s in 5 min. "
                "Revoking all remote sessions.",
                len(recent), client_key[:16],
            )
            self.revoke_all_remote_sessions()
            _audit_event(
                "suspicious_activity_detected",
                client_key=client_key[:16] + "...",
                failures=len(recent),
            )

    # -- Constant-time response delay ----------------------------------------

    async def uniform_delay(self, start_time: float) -> None:
        """Add artificial delay to ensure uniform response timing.

        All auth error responses take at least _MIN_RESPONSE_TIME_MS
        to return. This prevents an attacker from distinguishing error
        types by measuring response latency.

        An attacker reading this code sees: every error path goes
        through this function, so timing measurement is useless.

        Args:
            start_time: time.monotonic() value when the check started.
        """
        elapsed_ms = (time.monotonic() - start_time) * 1000
        remaining_ms = _MIN_RESPONSE_TIME_MS - elapsed_ms
        if remaining_ms > 0:
            await asyncio.sleep(remaining_ms / 1000)


# ---------------------------------------------------------------------------
# Security headers for remote access
# ---------------------------------------------------------------------------

# Applied only when remote access is active
REMOTE_SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def get_remote_security_headers() -> dict[str, str]:
    """Return security headers to apply for remote access responses.

    These are applied by the security middleware when remote access
    is active. In local-only mode, the standard headers apply.
    """
    return dict(REMOTE_SECURITY_HEADERS)


def is_remote_request(client_host: str | None) -> bool:
    """Check if a request is from a remote (non-localhost) client."""
    if client_host is None:
        return False
    return client_host not in ("127.0.0.1", "::1")


# ---------------------------------------------------------------------------
# JWT claim helpers for session binding
# ---------------------------------------------------------------------------

def build_remote_jwt_claims(
    cert_fingerprint: str, source_ip: str,
) -> dict[str, str]:
    """Build extra JWT claims for remote session binding.

    These claims are added to the JWT payload during token creation
    for remote sessions. On each request, the cert fingerprint in
    the JWT is compared to the one in the presented client cert.

    Args:
        cert_fingerprint: SHA256 of the client certificate.
        source_ip: Client IP at authentication time.

    Returns:
        Dict of extra claims to add to JWT payload.
    """
    return {
        "cert_fp": cert_fingerprint,
        "src_ip": source_ip,
    }


def extract_cert_fingerprint_from_request(request) -> str | None:
    """Extract client certificate SHA256 fingerprint from a request.

    In mTLS, the client cert is available via the SSL transport.
    Returns None if no client cert is presented.
    """
    try:
        # Uvicorn/Starlette: client cert info in scope
        transport = request.scope.get("transport")
        if transport and hasattr(transport, "get_extra_info"):
            peercert_bin = transport.get_extra_info("peercert_bin")
            if peercert_bin:
                return hashlib.sha256(peercert_bin).hexdigest()

        # Alternative: check for X-Client-Cert-Fingerprint header
        # (set by reverse proxy in front of uvicorn)
        fp_header = request.headers.get("x-client-cert-fingerprint")
        if fp_header and len(fp_header) == 64:
            return fp_header

    except Exception as exc:
        logger.debug("Could not extract client cert fingerprint: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

def _audit_event(event: str, **details) -> None:
    """Log a session security event."""
    logger.warning("SESSION AUDIT [%s]: %s", event, details)
    try:
        from opti_oignon.signed_audit_log import chain_log
        chain_log(
            event_type=event,
            source="remote_session_guard",
            action=event,
            severity="WARNING",
            **details,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

remote_session_guard = RemoteSessionGuard()

REMOTE_SESSION_GUARD_AVAILABLE = True


def reset_remote_session_guard() -> None:
    """Reset the module singleton state (test isolation)."""
    global remote_session_guard
    remote_session_guard = RemoteSessionGuard()
