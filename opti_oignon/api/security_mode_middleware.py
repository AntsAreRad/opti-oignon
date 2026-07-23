#!/usr/bin/env python3
"""
Security Mode Middleware for Opti-Oignon.

Enforces mode-specific restrictions on every HTTP request based on
the current Daily/Bulbe mode.  Sits in the middleware stack alongside
the SecurityHeadersMiddleware.

In **Bulbe** mode:
  - Block search endpoints if kill switch is engaged
  - Enforce cookie-only auth (reject Bearer tokens)
  - Apply stricter rate limits via response headers
  - Set SameSite=Strict on response cookies
  - Block plugin install without allowlist verification

In **Daily** mode:
  - All restrictions relaxed (standard security baseline)

Fail-closed: if the security_mode module is unavailable or the mode cannot
be determined, the middleware applies Bulbe enforcement (the most
restrictive), matching the network bind guard and the Veilid gate.
"""

import logging
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Paths that are always allowed (auth flow, health checks, static)
_ALWAYS_ALLOWED_PREFIXES = (
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/status",
    "/api/auth/refresh",
    "/api/health",
    "/api/security/mode",
    "/docs",
    "/openapi.json",
)

# Search-related path prefixes (blocked in Bulbe when kill switch engaged)
_SEARCH_PREFIXES = (
    "/api/search",
    "/api/web-search",
)

# Plugin install path prefixes (blocked in Bulbe without allowlist)
_PLUGIN_INSTALL_PREFIXES = (
    "/api/plugins/install",
    "/api/plugin-marketplace/install",
)


def _get_security_mode():
    """Load the current security mode.

    Returns (mode_str, policy) or (None, None) if unavailable.
    """
    try:
        from opti_oignon.security_mode import get_current_mode, get_policy
        mode = get_current_mode()
        policy = get_policy()
        return mode, policy
    except Exception:
        return None, None


def _is_kill_switch_engaged() -> bool:
    """Check if the web search kill switch is currently engaged."""
    try:
        from opti_oignon.search_killswitch import search_killswitch
        return search_killswitch.is_killed()
    except Exception:
        return False


def _is_plugin_allowed(request: Request) -> bool:
    """Check if a plugin install request is allowed by the allowlist.

    Extracts plugin_id from the request path or query and verifies
    it against the allowlist manager.
    """
    try:
        from opti_oignon.plugin_allowlist import plugin_allowlist_manager
        # Try to extract plugin_id from query params or path
        plugin_id = request.query_params.get("plugin_id", "")
        if not plugin_id:
            # Attempt to extract from path: /api/plugins/install/{plugin_id}
            parts = request.url.path.strip("/").split("/")
            if len(parts) >= 4:
                plugin_id = parts[3]
        if plugin_id:
            return plugin_allowlist_manager.is_allowed(plugin_id)
        # If we cannot determine the plugin, block in Bulbe
        return False
    except Exception:
        return False


def _reject_if_cert_revoked(request: Request) -> "JSONResponse | None":
    """Reject a request that presents a revoked client certificate (RA-01).

    Local requests present no client certificate, so this is a no-op for them.
    For a request that does present one (remote mTLS), the persistent CRL /
    metadata is consulted with zero grace period; if the revocation status of a
    cert-bearing request cannot be determined, the request is denied (fail
    closed).
    """
    try:
        from opti_oignon.remote_session_guard import (
            extract_cert_fingerprint_from_request,
        )
        fp = extract_cert_fingerprint_from_request(request)
    except Exception:
        return None
    if not fp:
        return None
    try:
        from opti_oignon.tls_manager import is_cert_revoked
        revoked = is_cert_revoked(fp)
    except Exception as exc:
        logger.error(
            "Could not determine revocation for client cert %s...: %s; "
            "denying.",
            fp[:16], exc,
        )
        revoked = True
    if revoked:
        logger.critical(
            "Rejected request from REVOKED client certificate %s... to %s",
            fp[:16], request.url.path,
        )
        return JSONResponse(
            status_code=403,
            content={
                "detail": "Client certificate has been revoked.",
                "mode": "remote",
                "restriction": "cert_revoked",
            },
        )
    return None


class SecurityModeMiddleware(BaseHTTPMiddleware):
    """Enforce Daily/Bulbe mode restrictions on every request.

    Registered in app.py alongside SecurityHeadersMiddleware.
    """

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        logger.info("Security mode middleware registered")

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process request through mode-specific security checks."""
        path = request.url.path

        # RA-01: deny a revoked client certificate before anything else. Local
        # requests carry no client certificate, so this is a no-op for them.
        revoked_response = _reject_if_cert_revoked(request)
        if revoked_response is not None:
            return revoked_response

        # Always allow certain paths (auth, health, mode management)
        for prefix in _ALWAYS_ALLOWED_PREFIXES:
            if path.startswith(prefix):
                return await call_next(request)

        # Load current mode and policy
        mode, policy = _get_security_mode()

        # Fail closed (M-01): an undeterminable mode is treated as Bulbe -- the
        # most restrictive -- matching the network bind guard and the Veilid
        # gate. Previously the request passed through with no enforcement. When
        # no policy object is available the Bulbe checks below fall back to
        # strict defaults (see the getattr() calls).
        if mode is None or policy is None:
            logger.warning(
                "Security mode undeterminable; failing closed to Bulbe "
                "enforcement."
            )
            mode, policy = "bulbe", None

        # --- Daily mode: minimal enforcement ---
        if mode == "daily":
            return await call_next(request)

        # --- Bulbe mode: strict enforcement ---

        # 0. Defense layer 3: Reject non-localhost requests in Bulbe.
        #    Even if binding somehow leaked to a non-local interface,
        #    this middleware blocks any request not from 127.0.0.1 or ::1.
        client_host = request.client.host if request.client else None
        if client_host and client_host not in ("127.0.0.1", "::1"):
            logger.critical(
                "Bulbe: REJECTED request from non-local IP %s to %s. "
                "This should be impossible if bind guard is working.",
                client_host,
                path,
            )
            _audit_non_local_request(client_host, path)
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Access denied. Server is in Bulbe mode "
                              "(localhost only).",
                    "mode": "bulbe",
                    "restriction": "non_local_rejected",
                },
            )

        # 1. Block search endpoints if kill switch is engaged
        for prefix in _SEARCH_PREFIXES:
            if path.startswith(prefix):
                if _is_kill_switch_engaged():
                    logger.warning(
                        "Bulbe: blocked search request (kill switch engaged): %s",
                        path,
                    )
                    return JSONResponse(
                        status_code=403,
                        content={
                            "detail": "Web search is disabled in Bulbe mode "
                                      "(kill switch engaged).",
                            "mode": "bulbe",
                            "restriction": "search_killed",
                        },
                    )

        # 2. Enforce cookie-only auth (reject Bearer tokens)
        if not getattr(policy, "bearer_auth_allowed", False):
            auth_header = request.headers.get("authorization", "")
            if auth_header.lower().startswith("bearer "):
                # Exception: app passwords are allowed for CLI
                # They go through /api/auth/login, already in _ALWAYS_ALLOWED
                logger.warning(
                    "Bulbe: rejected Bearer token auth for %s", path,
                )
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "Bearer token authentication is not allowed "
                                  "in Bulbe mode. Use httpOnly cookie auth.",
                        "mode": "bulbe",
                        "restriction": "bearer_rejected",
                    },
                )

        # 3. Block plugin install without allowlist
        for prefix in _PLUGIN_INSTALL_PREFIXES:
            if path.startswith(prefix) and request.method in ("POST", "PUT"):
                if getattr(policy, "plugin_allowlist_required", True):
                    if not _is_plugin_allowed(request):
                        logger.warning(
                            "Bulbe: blocked plugin install without allowlist: %s",
                            path,
                        )
                        return JSONResponse(
                            status_code=403,
                            content={
                                "detail": "Plugin installation requires allowlist "
                                          "approval in Bulbe mode.",
                                "mode": "bulbe",
                                "restriction": "plugin_not_allowed",
                            },
                        )

        # Process the request
        response = await call_next(request)

        # 4. Set SameSite=Strict on response cookies in Bulbe mode
        if getattr(policy, "cookie_samesite", "Strict").lower() == "strict":
            self._enforce_samesite_strict(response)

        # 5. Add rate limit headers for Bulbe
        if path.startswith("/api/"):
            response.headers["X-RateLimit-Mode"] = "bulbe"
            response.headers["X-RateLimit-MaxAttempts"] = str(
                getattr(policy, "rate_limit_max_attempts", 3)
            )
            response.headers["X-RateLimit-Window"] = str(
                getattr(policy, "rate_limit_window", 300)
            )

        return response

    @staticmethod
    def _enforce_samesite_strict(response: Response) -> None:
        """Rewrite Set-Cookie headers to use SameSite=Strict.

        Scans existing Set-Cookie headers and replaces any
        SameSite=Lax or missing SameSite with SameSite=Strict.
        """
        if not hasattr(response, "headers"):
            return

        raw_headers = response.headers.raw
        updated = []
        for key, value in raw_headers:
            if key.lower() == b"set-cookie":
                cookie_str = value.decode("latin-1", errors="replace")
                # Replace or append SameSite
                if "samesite=" in cookie_str.lower():
                    # Replace existing SameSite value
                    parts = cookie_str.split(";")
                    new_parts = []
                    for part in parts:
                        stripped = part.strip().lower()
                        if stripped.startswith("samesite="):
                            new_parts.append(" SameSite=Strict")
                        else:
                            new_parts.append(part)
                    cookie_str = ";".join(new_parts)
                else:
                    cookie_str += "; SameSite=Strict"
                updated.append((key, cookie_str.encode("latin-1")))
            else:
                updated.append((key, value))

        response.headers.raw[:] = updated


# Module-level availability flag
SECURITY_MODE_MIDDLEWARE_AVAILABLE = True


def _audit_non_local_request(client_host: str, path: str) -> None:
    """Log a critical audit event for non-local request in Bulbe mode."""
    try:
        from opti_oignon.signed_audit_log import chain_log
        chain_log(
            event_type="non_local_request_rejected",
            source="security_mode_middleware",
            action="reject_non_local",
            severity="CRITICAL",
            client_host=client_host,
            path=path,
        )
    except Exception:
        pass
