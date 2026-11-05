#!/usr/bin/env python3
"""
Security Mode Middleware for Opti-Oignon.

Enforces mode-specific restrictions on every HTTP request based on
the current Daily/Bulbe mode.  Sits in the middleware stack alongside
the SecurityHeadersMiddleware.

In **Bulbe** mode:
  - Refuse a client that is not this machine's, or a request a proxy
    forwarded, on every path but the health check; close such a client's
    websocket handshake before it is accepted (``__call__``: an HTTP
    middleware never sees a websocket), and its socket opened in Daily at
    its first message once Bulbe began
  - Refuse the routes that only reach the web (the marketplace install, the
    GGUF download, the model pull) before their handler
  - Block search endpoints if kill switch is engaged
  - Enforce cookie-only auth (reject Bearer tokens)
  - Apply stricter rate limits via response headers
  - Set SameSite=Strict on response cookies, the sign-in's included
  - Block plugin install without allowlist verification

The sign-in flow and the schema (``_ALWAYS_ALLOWED_PREFIXES``) are spared
the Bearer refusal alone: the local-only check and the cookie rule hold for
them too.

In **Daily** mode:
  - All restrictions relaxed (standard security baseline)

Fail-closed: if the security_mode module is unavailable or the mode cannot
be determined, the middleware applies Bulbe enforcement (the most
restrictive), matching the network bind guard and the Veilid gate.
"""

import logging
from typing import Any

from starlette._utils import get_route_path
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.websockets import WebSocketDisconnect

logger = logging.getLogger(__name__)

# The client the Host gate hands on for a request a proxy the configuration does not list forwarded
# (``host_gate.UNLISTED_PROXY``; this module is loaded alone too, so the word is held here, and a contract ties the
# two).
_FORWARDED_CLIENT = "unlisted-proxy"

# The clients Bulbe serves, by the socket's address: this machine's own. ``routes_security._from_this_machine``
# reads the same rule for entering Bulbe, and a contract ties the two.
_LOCAL_CLIENTS = ("127.0.0.1", "::1")

# Paths Bulbe takes a Bearer token on (the sign-in flow, the schema: a client holding a token reaches them to learn
# how to sign in), each with every path below it. In Bulbe they are still refused to a client that is not this
# machine's, and their cookies are still made Strict. The security mode's own routes are not among them: in Bulbe
# they are checked as any route.
_ALWAYS_ALLOWED_PREFIXES = (
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/status",
    "/api/auth/refresh",
    "/docs",
    "/openapi.json",
)

# Paths that are always allowed exactly, and nothing below them: the health
# check, not the benchmark routes that live under it. It sets nothing, and it
# answers in every mode whoever reaches it, but a revoked certificate.
_ALWAYS_ALLOWED_EXACT = ("/api/health",)

# Search-related path prefixes (blocked in Bulbe when kill switch engaged)
_SEARCH_PREFIXES = ("/api/search",)

# Plugin install path prefixes (blocked in Bulbe without allowlist)
_PLUGIN_INSTALL_PREFIXES = ("/api/plugins/install",)

# Routes whose only purpose is a request to the web, by method and exact route
# path: refused in Bulbe before their handler, in the web gate's words.
_WEB_ROUTES = (
    ("POST", "/api/plugins/marketplace/install"),
    ("POST", "/api/backends/gguf/download"),
    ("POST", "/api/model-lifecycle/pull"),
)

_WEB_REFUSED_FALLBACK = "Web requests are refused outside Daily mode."


def _matches(path: str, prefix: str) -> bool:
    """Whether ``path`` is ``prefix`` or lies below it, by whole path segments."""
    return path == prefix or path.startswith(prefix + "/")


def _web_refused_text() -> str:
    """The web gate's refusal text for a mode that is not Daily."""
    try:
        from opti_oignon.web_gate import _REFUSALS

        return _REFUSALS["mode"]
    except Exception:
        return _WEB_REFUSED_FALLBACK


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
    """Check if the web search kill switch is currently engaged.

    Fail-closed on a switch that cannot be read: a switch whose own import
    fails, or whose read raises, reads engaged. Only the switch module itself
    being absent reads not engaged, because the web searcher then refuses on
    its own.
    """
    try:
        from opti_oignon.search_killswitch import search_killswitch
    except ModuleNotFoundError as exc:
        if exc.name == "opti_oignon.search_killswitch":
            return False
        return True
    except Exception:
        return True
    try:
        return bool(search_killswitch.is_killed())
    except Exception:
        return True


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
            # Attempt to extract from path: /api/plugins/install/{plugin_id},
            # the path the router reads (the mount prefix removed).
            parts = get_route_path(request.scope).strip("/").split("/")
            if len(parts) >= 4:
                plugin_id = parts[3]
        if plugin_id:
            return plugin_allowlist_manager.is_allowed(plugin_id)
        # If we cannot determine the plugin, block in Bulbe
        return False
    except Exception:
        return False


def _reject_if_cert_revoked(request: Request) -> "JSONResponse | None":
    """Reject a request whose client certificate fingerprint is revoked (RA-01).

    It acts only on a fingerprint a reverse proxy in front of the API puts in
    the ``X-Client-Cert-Fingerprint`` header: uvicorn puts no transport in a
    request's scope, so a request served over mTLS directly names no
    certificate here. Those are checked at their connection, and again while
    it is open, by ``tls_manager.client_cert_gate``. The devices' records are
    consulted with zero grace period; if the revocation status of a named
    fingerprint cannot be determined, the request is denied (fail closed).
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

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Close, before it is accepted, a websocket handshake from a client Bulbe does not serve, in Bulbe or when
        the mode cannot be read, as its HTTP requests are refused: ``dispatch`` never sees a websocket (an HTTP
        middleware hands every other scope straight on). Such a client's socket accepted in Daily is checked again
        at each message it sends or is sent (``_closing_in_bulbe``), and closed at the first one after Bulbe began.
        Every other scope goes on as before."""
        if scope["type"] == "websocket":
            client = scope.get("client")
            client_host = client[0] if client else None
            if client_host and client_host not in _LOCAL_CLIENTS:
                path = get_route_path(scope)
                mode, _policy = _get_security_mode()
                if mode != "daily" and _refused_in_bulbe(client_host, path):
                    await send({"type": "websocket.close", "code": 1008})
                    return
                receive, send = _closing_in_bulbe(client_host, path, receive, send)
        await super().__call__(scope, receive, send)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process request through mode-specific security checks.

        Every decision is made on the path the router matches: the request's
        path with the application's mount prefix removed, compared by whole
        segments.
        """
        path = get_route_path(request.scope)

        # RA-01: deny a revoked client certificate before anything else. Local
        # requests carry no client certificate, so this is a no-op for them.
        revoked_response = _reject_if_cert_revoked(request)
        if revoked_response is not None:
            return revoked_response

        # The health check passes in every mode, whoever asks; every other path
        # is checked below, the sign-in flow's and the security mode's included.
        if path in _ALWAYS_ALLOWED_EXACT:
            return await call_next(request)
        spared = any(_matches(path, prefix) for prefix in _ALWAYS_ALLOWED_PREFIXES)

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
        #    A request a proxy forwarded, which the Host gate hands on as
        #    _FORWARDED_CLIENT, is no local one either: said as what it is.
        client_host = request.client.host if request.client else None
        if _refused_in_bulbe(client_host, path):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Access denied. Server is in Bulbe mode "
                              "(localhost only).",
                    "mode": "bulbe",
                    "restriction": "non_local_rejected",
                },
            )

        # 1. Refuse the web-only routes before their handler: in Bulbe no
        #    request leaves for the web, and these routes make nothing else.
        if (request.method, path) in _WEB_ROUTES:
            logger.warning("Bulbe: refused a web-only route: %s %s", request.method, path)
            return JSONResponse(
                status_code=403,
                content={
                    "detail": _web_refused_text(),
                    "mode": "bulbe",
                    "restriction": "web_refused",
                },
            )

        # 1b. Block search endpoints if kill switch is engaged
        for prefix in _SEARCH_PREFIXES:
            if _matches(path, prefix):
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

        # 2. Enforce cookie-only auth (reject Bearer tokens), but on the sign-in
        #    flow and the schema, which a client holding a token reaches to
        #    learn how to sign in.
        if not spared and not getattr(policy, "bearer_auth_allowed", False):
            auth_header = request.headers.get("authorization", "")
            if auth_header.lower().startswith("bearer "):
                # App passwords make no exception: no client signs in
                # with one yet (docs/security/authentication.md).
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
            if _matches(path, prefix) and request.method in ("POST", "PUT"):
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


def _closing_in_bulbe(client_host: str, path: str, receive: Any, send: Any) -> "tuple[Any, Any]":
    """The ``receive`` and ``send`` of a socket a client Bulbe does not serve opened in Daily, each message it is sent
    or sends checked again: once the mode reads Bulbe, or cannot be read, the socket is closed (1008) and the refusal
    audited, its endpoint told the client went (``WebSocketDisconnect``) at its next send and handed a disconnect at
    its next receive. Its own close after that is dropped: the socket is closed once."""
    closed: list[bool] = []

    def in_bulbe() -> bool:
        mode, _policy = _get_security_mode()
        return mode != "daily"

    async def close() -> None:
        if not closed:
            closed.append(True)
            _refused_in_bulbe(client_host, path)
            await send({"type": "websocket.close", "code": 1008})

    async def checked_receive() -> Any:
        message = await receive()
        if closed or (message.get("type") == "websocket.receive" and in_bulbe()):
            await close()
            return {"type": "websocket.disconnect", "code": 1008}
        return message

    async def checked_send(message: Any) -> None:
        kind = message.get("type")
        if closed and kind == "websocket.close":
            return
        if closed or (kind == "websocket.send" and in_bulbe()):
            await close()
            raise WebSocketDisconnect(code=1008)
        await send(message)

    return checked_receive, checked_send


def _refused_in_bulbe(client_host: "str | None", path: str) -> bool:
    """Whether Bulbe refuses a client at ``client_host`` (another address, or a request a proxy forwarded), the
    refusal logged and audited as what it is; this machine's own clients, and a request with no client, are served."""
    if not client_host or client_host in _LOCAL_CLIENTS:
        return False
    if client_host == _FORWARDED_CLIENT:
        logger.warning(
            "Bulbe: REJECTED a request a proxy forwarded to %s: Bulbe "
            "serves this machine's own clients only.",
            path,
        )
        _audit_non_local_request(client_host, path, forwarded=True)
    else:
        logger.critical(
            "Bulbe: REJECTED request from non-local IP %s to %s. "
            "This should be impossible if bind guard is working.",
            client_host,
            path,
        )
        _audit_non_local_request(client_host, path)
    return True


def _audit_non_local_request(client_host: str, path: str, *, forwarded: bool = False) -> None:
    """Log an audit event for a non-local request in Bulbe mode: critical for a non-local address (the bind guard
    should have made it impossible), a warning for a request a proxy forwarded (any local client may send one)."""
    try:
        from opti_oignon.signed_audit_log import chain_log
        chain_log(
            event_type="forwarded_request_rejected" if forwarded else "non_local_request_rejected",
            source="security_mode_middleware",
            action="reject_non_local",
            severity="WARNING" if forwarded else "CRITICAL",
            client_host=client_host,
            path=path,
        )
    except Exception:
        pass
