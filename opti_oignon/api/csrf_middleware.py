#!/usr/bin/env python3
"""
CSRF Validation Middleware for Opti-Oignon (S136 audit fix).

Enforces the double-submit cookie pattern on ALL state-changing HTTP
requests (POST, PUT, DELETE, PATCH).  Previously, ``_validate_csrf()``
was defined in ``routes_auth.py`` but never called -- this middleware
ensures it runs globally via the Starlette middleware stack.

Skip conditions (by design):
  - Non-state-changing methods (GET, HEAD, OPTIONS)
  - Requests using Bearer token (API/CLI clients, not browser)
  - Login / register / health / OpenAPI endpoints
  - WebSocket upgrade requests (handled by WebSocket auth)
  - When cookie_mode is disabled (pure API usage)
  - When csrf_enabled is set to false in security.yaml

Security: constant-time comparison via hmac.compare_digest.
"""

import hmac
import logging
import os
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Hardcoded, never overridable
checkpoint_before_apply = True

_CSRF_COOKIE = "oo_csrf_token"

# Paths exempt from CSRF (pre-auth or read-only endpoints)
_CSRF_EXEMPT_PREFIXES = (
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/status",
    "/api/auth/refresh",
    "/api/health",
    "/api/security/mode",
    "/docs",
    "/openapi.json",
    "/redoc",
)

# Methods that require CSRF validation
_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})


def _is_bulbe() -> bool:
    """Return True when Bulbe (hardened) mode is active.

    Mirrors the check in auth_middleware so the two middlewares agree on
    when authentication-related protections apply.
    """
    try:
        from opti_oignon.security_mode import is_bulbe
        return is_bulbe()
    except Exception:
        return False


def _is_single_user_unauthenticated() -> bool:
    """Return True when the app runs in single-user mode outside Bulbe.

    S171: in single-user mode (non-Bulbe) the auth middleware bypasses
    authentication entirely, so there is no session cookie and no
    cross-site request to forge. CSRF double-submit validation would only
    reject legitimate same-origin POSTs, so it is skipped here -- exactly as
    auth is skipped in auth_middleware. In Bulbe mode, full CSRF enforcement
    is retained regardless of single_user_mode.
    """
    try:
        from opti_oignon.auth import auth_manager as mgr
    except Exception:
        return False
    if mgr is None:
        return False
    return bool(getattr(mgr, "single_user_mode", False)) and not _is_bulbe()


def _load_csrf_config() -> dict[str, Any]:
    """Load CSRF-related config from security.yaml."""
    try:
        import yaml
        sec_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "security.yaml",
        )
        if os.path.isfile(sec_path):
            with open(sec_path, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            jwt_cfg = raw.get("jwt", {})
            return {
                "cookie_mode": jwt_cfg.get("cookie_mode", True),
                "csrf_enabled": jwt_cfg.get("csrf_enabled", True),
            }
    except Exception as exc:
        logger.debug("Failed to load CSRF config: %s", exc)
    return {"cookie_mode": True, "csrf_enabled": True}


class CSRFMiddleware(BaseHTTPMiddleware):
    """Validate CSRF double-submit cookie on all state-changing requests.

    Registered in app.py after SecurityModeMiddleware.
    """

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        cfg = _load_csrf_config()
        self._cookie_mode = cfg["cookie_mode"]
        self._csrf_enabled = cfg["csrf_enabled"]
        logger.info(
            "CSRF middleware registered (cookie_mode=%s, csrf_enabled=%s)",
            self._cookie_mode,
            self._csrf_enabled,
        )

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Validate CSRF token before processing state-changing requests."""
        # S171: skip in single-user mode (non-Bulbe) -- no auth session exists,
        # so there is nothing for CSRF to protect (parity with auth_middleware).
        if _is_single_user_unauthenticated():
            return await call_next(request)

        # Skip if CSRF is disabled or not in cookie mode
        if not self._cookie_mode or not self._csrf_enabled:
            return await call_next(request)

        # Skip non-state-changing methods
        if request.method not in _STATE_CHANGING_METHODS:
            return await call_next(request)

        # Skip exempt paths
        path = request.url.path
        for prefix in _CSRF_EXEMPT_PREFIXES:
            if path.startswith(prefix):
                return await call_next(request)

        # Skip WebSocket upgrades
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        # Skip Bearer token requests (API/CLI clients, not browsers)
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            return await call_next(request)

        # Validate CSRF double-submit cookie
        cookie_token = request.cookies.get(_CSRF_COOKIE, "")
        header_token = request.headers.get("X-CSRF-Token", "")

        if not cookie_token or not header_token:
            logger.warning(
                "CSRF token missing on %s %s (client=%s)",
                request.method,
                path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "CSRF token missing. Include X-CSRF-Token header.",
                },
            )

        # Constant-time comparison to prevent timing oracle
        if not hmac.compare_digest(cookie_token, header_token):
            logger.warning(
                "CSRF token mismatch on %s %s (client=%s)",
                request.method,
                path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF token mismatch."},
            )

        return await call_next(request)


CSRF_MIDDLEWARE_AVAILABLE = True
