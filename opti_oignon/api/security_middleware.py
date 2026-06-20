#!/usr/bin/env python3
"""
Security headers middleware for Opti-Oignon (S124).

Adds standard security headers to every HTTP response:
- X-Content-Type-Options
- X-Frame-Options
- Content-Security-Policy
- Referrer-Policy
- Permissions-Policy
- X-XSS-Protection
- Cache-Control (for API responses)
- Strict-Transport-Security (optional, for HTTPS deployments)

Configuration is loaded from config/security.yaml.  If the file is
missing or unreadable, secure defaults are used.
"""

import logging
import os
from typing import Any, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Secure defaults (used when security.yaml is missing or incomplete)
_DEFAULT_HEADERS = {
    "x_frame_options": "DENY",
    "content_security_policy": (
        "default-src 'self'; script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    ),
    "referrer_policy": "strict-origin-when-cross-origin",
    "permissions_policy": "camera=(), microphone=(), geolocation=()",
    "x_content_type_options": "nosniff",
    "x_xss_protection": "1; mode=block",
    "cache_control": "no-store",
    "hsts_enabled": False,
    "hsts_max_age": 31536000,
}


def _load_header_config() -> dict[str, Any]:
    """Load security header configuration from security.yaml.

    Returns a dict with header settings.  Falls back to defaults
    for any missing keys.
    """
    config = dict(_DEFAULT_HEADERS)
    try:
        import yaml
        sec_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "security.yaml"
        )
        if os.path.isfile(sec_path):
            with open(sec_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            headers_cfg = raw.get("headers", {})
            if isinstance(headers_cfg, dict):
                # Only override keys that exist in the yaml
                for key in _DEFAULT_HEADERS:
                    if key in headers_cfg:
                        config[key] = headers_cfg[key]
                # Check enabled flag
                if not headers_cfg.get("enabled", True):
                    config["_disabled"] = True
    except Exception as exc:
        logger.debug("Failed to load security header config: %s", exc)

    return config


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware that injects security headers.

    Must be registered BEFORE CORSMiddleware in the middleware stack
    so that CORS headers are not overwritten.
    """

    def __init__(self, app: Any, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(app)
        self._config = config or _load_header_config()
        self._enabled = not self._config.get("_disabled", False)

        if self._enabled:
            logger.info("Security headers middleware enabled")
        else:
            logger.info("Security headers middleware DISABLED via config")

    @property
    def enabled(self) -> bool:
        """Whether the middleware is active."""
        return self._enabled

    @property
    def config(self) -> dict[str, Any]:
        """Current header configuration."""
        return dict(self._config)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process request and add security headers to response."""
        response = await call_next(request)

        if not self._enabled:
            return response

        cfg = self._config

        # X-Content-Type-Options
        val = cfg.get("x_content_type_options")
        if val:
            response.headers["X-Content-Type-Options"] = str(val)

        # X-Frame-Options
        val = cfg.get("x_frame_options")
        if val:
            response.headers["X-Frame-Options"] = str(val)

        # Content-Security-Policy
        val = cfg.get("content_security_policy")
        if val:
            response.headers["Content-Security-Policy"] = str(val)

        # Referrer-Policy
        val = cfg.get("referrer_policy")
        if val:
            response.headers["Referrer-Policy"] = str(val)

        # Permissions-Policy
        val = cfg.get("permissions_policy")
        if val:
            response.headers["Permissions-Policy"] = str(val)

        # X-XSS-Protection (legacy, still useful for older browsers)
        val = cfg.get("x_xss_protection")
        if val:
            response.headers["X-XSS-Protection"] = str(val)

        # Cache-Control for API responses
        val = cfg.get("cache_control")
        if val and request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = str(val)

        # HSTS (only when explicitly enabled — requires HTTPS)
        if cfg.get("hsts_enabled"):
            max_age = cfg.get("hsts_max_age", 31536000)
            response.headers["Strict-Transport-Security"] = (
                f"max-age={max_age}; includeSubDomains"
            )

        # S133: Apply hardened headers for remote (non-localhost) requests
        self._apply_remote_headers(request, response)

        return response

    @staticmethod
    def _apply_remote_headers(request: Request, response: Response) -> None:
        """Apply extra security headers when request is from a remote client.

        These override the base headers with stricter values for remote
        access (HSTS, no-referrer, X-XSS-Protection=0 for CSP-only).
        """
        try:
            from opti_oignon.remote_session_guard import (
                is_remote_request,
                get_remote_security_headers,
            )
            client_host = request.client.host if request.client else None
            if is_remote_request(client_host):
                for header_name, header_value in get_remote_security_headers().items():
                    response.headers[header_name] = header_value
        except ImportError:
            pass


# =========================================================================
# Module-level helpers
# =========================================================================

SECURITY_HEADERS_AVAILABLE = True


def get_security_headers_config() -> dict[str, Any]:
    """Return the current security headers configuration."""
    return _load_header_config()
