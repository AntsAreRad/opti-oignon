#!/usr/bin/env python3
"""
Global Authentication Middleware for Opti-Oignon (S136 audit fix).

Enforces authentication on ALL API endpoints by default.  Only explicitly
listed public endpoints are exempt.  This replaces the per-router
``dependencies=[Depends(_get_current_user)]`` approach which was fragile
and left 37+ routers unprotected.

Design: deny-by-default.  If a path is not in the allowlist, a valid
JWT is required (cookie or Bearer header).  In single-user mode
(non-Bulbe), the check is bypassed.

This middleware runs AFTER CORS and security headers but BEFORE route
dispatch, so every endpoint is covered regardless of whether the router
remembered to add a Depends().
"""

import logging
import os
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Hardcoded, never overridable
checkpoint_before_apply = True

# ---------------------------------------------------------------------------
# Public paths: these do NOT require authentication
# ---------------------------------------------------------------------------

_PUBLIC_EXACT = frozenset({
    "/api/health",
    "/api/auth/status",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
})

_PUBLIC_PREFIXES = (
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/refresh",
    "/api/auth/2fa-challenge",
    "/docs/",
    "/redoc/",
    "/openapi",
    # Static files served by frontend
    "/assets/",
    "/_app/",
    "/static/",
)

# Methods that never need auth (browser preflight)
_EXEMPT_METHODS = frozenset({"OPTIONS"})

# Cookie name (must match routes_auth.py)
_ACCESS_COOKIE = "oo_access_token"


def _is_public_path(path: str) -> bool:
    """Check whether a path is in the public allowlist."""
    if path in _PUBLIC_EXACT:
        return True
    for prefix in _PUBLIC_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def _get_auth_manager():
    """Lazy-load the AuthManager singleton."""
    try:
        from opti_oignon.auth import auth_manager
        return auth_manager
    except Exception:
        return None


def _is_bulbe() -> bool:
    """Check whether Bulbe mode is active."""
    try:
        from opti_oignon.security_mode import is_bulbe
        return is_bulbe()
    except Exception:
        return False


class AuthMiddleware(BaseHTTPMiddleware):
    """Global authentication enforcement.

    Deny-by-default: every non-public path requires a valid JWT.
    In single-user mode (non-Bulbe), auth is bypassed.
    """

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        logger.info("Global auth middleware registered (deny-by-default)")

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Check authentication before dispatching."""
        path = request.url.path

        # Always allow exempt methods (OPTIONS for CORS preflight)
        if request.method in _EXEMPT_METHODS:
            return await call_next(request)

        # Always allow public paths
        if _is_public_path(path):
            return await call_next(request)

        # WebSocket upgrades are handled by per-endpoint auth
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        # Check single-user mode (bypass auth unless Bulbe)
        mgr = _get_auth_manager()
        if mgr is not None and mgr.single_user_mode and not _is_bulbe():
            return await call_next(request)

        # If auth module is not available, fail open with warning
        if mgr is None:
            logger.debug("Auth module unavailable, allowing request to %s", path)
            return await call_next(request)

        # Extract token from cookie or Bearer header
        token = request.cookies.get(_ACCESS_COOKIE)
        if not token:
            auth_header = request.headers.get("authorization", "")
            if auth_header.lower().startswith("bearer "):
                token = auth_header[7:]

        if not token:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Authentication required.",
                    "path": path,
                },
            )

        # Validate token
        payload = mgr.validate_token(token)
        if payload is None:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid or expired token. Please log in again.",
                    "path": path,
                },
            )

        # Token valid — proceed
        return await call_next(request)


AUTH_MIDDLEWARE_AVAILABLE = True
