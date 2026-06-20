#!/usr/bin/env python3
"""
Authentication API routes (S98, updated S127).

POST   /api/auth/register           -- Create account
POST   /api/auth/login              -- Get JWT token (or 2FA challenge)
POST   /api/auth/login/2fa          -- Complete 2FA challenge (S127)
POST   /api/auth/logout             -- Invalidate session
POST   /api/auth/refresh            -- Refresh token pair
GET    /api/auth/me                 -- Get current user profile
PUT    /api/auth/me                 -- Update profile
PUT    /api/auth/me/password        -- Change password
GET    /api/auth/users              -- List users (admin only)
DELETE /api/auth/users/{user_id}    -- Delete user (admin only)
GET    /api/auth/settings           -- Get per-user settings
PUT    /api/auth/settings           -- Update per-user settings
POST   /api/auth/projects/share     -- Share a project
GET    /api/auth/projects/{id}/members -- List project members
DELETE /api/auth/projects/{project_id}/members/{user_id} -- Remove access
GET    /api/auth/audit              -- Get audit log (admin)
GET    /api/auth/status             -- Auth system status (public)
"""

import logging
import secrets
import threading
import time
from typing import Any

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


# =============================================================================
# SCHEMAS
# =============================================================================

class RegisterRequest(BaseModel):
    """Request body for user registration."""
    username: str = Field(min_length=2, max_length=64, description="Username")
    password: str = Field(min_length=8, description="Password")
    email: str = Field(default="", description="Email address (optional)")


class LoginRequest(BaseModel):
    """Request body for login."""
    username: str = Field(description="Username")
    password: str = Field(description="Password")


class RefreshRequest(BaseModel):
    """Request body for token refresh."""
    refresh_token: str = Field(description="Refresh token")


class LogoutRequest(BaseModel):
    """Request body for logout."""
    refresh_token: str = Field(description="Refresh token to invalidate")


class ProfileUpdateRequest(BaseModel):
    """Request body for profile update."""
    username: str | None = Field(default=None, description="New username")
    email: str | None = Field(default=None, description="New email")
    metadata: dict[str, Any] | None = Field(default=None, description="Metadata to merge")


class PasswordChangeRequest(BaseModel):
    """Request body for password change."""
    current_password: str = Field(description="Current password")
    new_password: str = Field(min_length=8, description="New password")


class SettingsUpdateRequest(BaseModel):
    """Request body for user settings update."""
    theme: str | None = Field(default=None, description="Theme (dark/light)")
    default_model: str | None = Field(default=None, description="Default model name")
    default_preset: str | None = Field(default=None, description="Default preset name")
    sidebar_open: bool | None = Field(default=None, description="Sidebar state")
    language: str | None = Field(default=None, description="UI language")
    preferences: dict[str, Any] | None = Field(default=None, description="Extra preferences")


class ShareProjectRequest(BaseModel):
    """Request body for sharing a project."""
    project_id: str = Field(description="Project UUID")
    username: str = Field(description="Username to share with")
    role: str = Field(default="viewer", description="Role: owner, editor, viewer")


class UserResponse(BaseModel):
    """Public user representation (no password hash)."""
    user_id: str
    username: str
    email: str
    role: str
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = {}


class TokenResponse(BaseModel):
    """JWT token pair response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 0
    user_id: str = ""


class TwoFALoginRequest(BaseModel):
    """Request body for 2FA login step (S127)."""
    challenge_id: str = Field(description="Challenge ID from step-1 login")
    code: str = Field(default="", description="TOTP or recovery code")
    method: str = Field(default="auto", description="2FA method: auto, totp, recovery, webauthn")
    webauthn_response: dict[str, Any] | None = Field(
        default=None, description="WebAuthn assertion response object"
    )


# =============================================================================
# 2FA CHALLENGE STORE (S127)
# =============================================================================

class _ChallengeStore:
    """Server-side store for pending 2FA challenges.

    Each challenge has:
      - challenge_id: random token (URL-safe, 32 bytes)
      - user_id: authenticated user awaiting 2FA
      - created_at: timestamp (expires after CHALLENGE_TTL_SECONDS)
      - attempts: number of failed 2FA attempts
      - locked: True if max attempts exceeded
      - methods: list of available 2FA methods for this user

    Thread-safe via a simple lock.  Periodic cleanup of expired
    entries happens on every create() call.
    """

    CHALLENGE_TTL_SECONDS = 300  # 5 minutes
    MAX_ATTEMPTS = 5

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, user_id: str, methods: list[str]) -> str:
        """Create a new challenge and return its ID."""
        self._cleanup()
        challenge_id = secrets.token_urlsafe(32)
        with self._lock:
            self._store[challenge_id] = {
                "user_id": user_id,
                "created_at": time.time(),
                "attempts": 0,
                "locked": False,
                "methods": methods,
            }
        return challenge_id

    def get(self, challenge_id: str) -> dict[str, Any] | None:
        """Retrieve a challenge if it exists and is not expired."""
        with self._lock:
            entry = self._store.get(challenge_id)
            if entry is None:
                return None
            if time.time() - entry["created_at"] > self.CHALLENGE_TTL_SECONDS:
                del self._store[challenge_id]
                return None
            return dict(entry)

    def record_attempt(self, challenge_id: str) -> bool:
        """Record a failed 2FA attempt.  Returns True if now locked."""
        with self._lock:
            entry = self._store.get(challenge_id)
            if entry is None:
                return True
            entry["attempts"] += 1
            if entry["attempts"] >= self.MAX_ATTEMPTS:
                entry["locked"] = True
            return entry["locked"]

    def consume(self, challenge_id: str) -> None:
        """Remove a challenge after successful 2FA validation."""
        with self._lock:
            self._store.pop(challenge_id, None)

    def _cleanup(self) -> None:
        """Remove expired challenges."""
        now = time.time()
        with self._lock:
            expired = [
                cid for cid, data in self._store.items()
                if now - data["created_at"] > self.CHALLENGE_TTL_SECONDS
            ]
            for cid in expired:
                del self._store[cid]


# Module-level singleton
_challenge_store = _ChallengeStore()


# =============================================================================
# HELPERS
# =============================================================================

def _get_auth_manager():
    """Get the auth manager singleton."""
    from .deps import AUTH_AVAILABLE, auth_manager
    if not AUTH_AVAILABLE or auth_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Authentication module not available",
        )
    return auth_manager


def _get_settings_store():
    """Get the user settings store singleton."""
    from .deps import USER_SETTINGS_AVAILABLE, user_settings_store
    if not USER_SETTINGS_AVAILABLE or user_settings_store is None:
        raise HTTPException(
            status_code=503,
            detail="User settings module not available",
        )
    return user_settings_store


def _get_2fa_manager():
    """Get the 2FA manager singleton (S127).

    Returns None if the auth_2fa module is not available,
    allowing graceful degradation (login works without 2FA).
    """
    try:
        from opti_oignon.auth_2fa import two_factor_manager
        return two_factor_manager
    except Exception:
        return None


def _get_rate_limiter():
    """Get the login rate limiter singleton (S124)."""
    try:
        from opti_oignon.auth import login_rate_limiter
        return login_rate_limiter
    except Exception:
        return None


def _extract_client_ip(request: Request) -> str:
    """Extract client IP from request (S124, S136 audit fix).

    S136 audit fix: X-Forwarded-For is NOT trusted by default because
    it is attacker-controlled.  An attacker can send a different value
    per request to bypass rate limiting entirely.

    X-Forwarded-For is only trusted when:
    1. The direct connection comes from localhost (127.0.0.1 / ::1),
       indicating a local reverse proxy (nginx, caddy, etc.)
    2. The request has a valid X-Forwarded-For header.

    Otherwise, the actual TCP socket peer address is used.
    """
    direct_ip = request.client.host if request.client else "unknown"

    # Only trust X-Forwarded-For from localhost (reverse proxy)
    if direct_ip in ("127.0.0.1", "::1"):
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

    return direct_ip


def _get_security_jwt_config() -> dict:
    """Load JWT cookie configuration from security.yaml (S125)."""
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).parent.parent / "config" / "security.yaml"
    try:
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("jwt", {})
    except Exception:
        pass
    return {"cookie_mode": True}


def _is_cookie_mode() -> bool:
    """Check if httpOnly cookie mode is enabled (S125)."""
    return _get_security_jwt_config().get("cookie_mode", True)


def _is_csrf_enabled() -> bool:
    """Check if CSRF protection is enabled (S125 hardening)."""
    return _get_security_jwt_config().get("csrf_enabled", True)


# S125: Cookie names for httpOnly JWT storage
_ACCESS_COOKIE = "oo_access_token"
_REFRESH_COOKIE = "oo_refresh_token"
_CSRF_COOKIE = "oo_csrf_token"


def _detect_secure_context(request: Request) -> bool:
    """Auto-detect whether we are in an HTTPS context.

    Checks X-Forwarded-Proto (behind reverse proxy), the URL scheme,
    and falls back to the cookie_secure config value.
    Localhost connections are allowed to use HTTP (development).
    """
    # Explicit config override
    jwt_cfg = _get_security_jwt_config()
    cfg_secure = jwt_cfg.get("cookie_secure", True)

    # Behind reverse proxy
    proto = request.headers.get("X-Forwarded-Proto", "")
    if proto.lower() == "https":
        return True

    # Direct HTTPS
    if hasattr(request, "url") and str(request.url).startswith("https"):
        return True

    # Localhost exception: allow HTTP for development
    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return False  # Localhost over HTTP is OK

    return cfg_secure


def _generate_csrf_token() -> str:
    """Generate a cryptographically random CSRF token."""
    import secrets
    return secrets.token_urlsafe(32)


def _is_remote_client(request: Request) -> bool:
    """Check if the request comes from a non-localhost client (S133).

    Used to apply hardened cookie settings for remote sessions.
    """
    try:
        from opti_oignon.remote_session_guard import is_remote_request
        client_host = request.client.host if request.client else None
        return is_remote_request(client_host)
    except ImportError:
        return False


def _set_auth_cookies(
    request: Request,
    response: Response,
    access_token: str,
    refresh_token: str,
    access_max_age: int = 3600,
) -> None:
    """Set httpOnly JWT cookies on a response (S125, hardened).

    Access token: short-lived (default 1h local, 5min remote).
    Refresh token: long-lived (30 days local, 1h remote).
    Both httpOnly (not accessible via JS).
    SameSite: Lax local, Strict remote (S133).
    Secure flag auto-detected from request context.
    CSRF token set as a non-httpOnly cookie (readable by JS for header).
    """
    jwt_cfg = _get_security_jwt_config()
    secure = _detect_secure_context(request)
    samesite = jwt_cfg.get("cookie_samesite", "lax")
    domain = jwt_cfg.get("cookie_domain", "") or None
    path = jwt_cfg.get("cookie_path", "/")

    # S133: Detect remote context and apply hardened settings
    is_remote = _is_remote_client(request)
    if is_remote:
        access_max_age = 300  # 5 minutes for remote sessions
        refresh_max_age = 3600  # 1 hour for remote refresh
        samesite = "strict"  # Always Strict for remote
        secure = True  # Always Secure for remote (TLS required)
    else:
        refresh_max_age = 30 * 24 * 3600  # 30 days for local

    response.set_cookie(
        key=_ACCESS_COOKIE,
        value=access_token,
        max_age=access_max_age,
        httponly=True,
        secure=secure,
        samesite=samesite,
        domain=domain,
        path=path,
    )
    response.set_cookie(
        key=_REFRESH_COOKIE,
        value=refresh_token,
        max_age=refresh_max_age,
        httponly=True,
        secure=secure,
        samesite=samesite,
        domain=domain,
        path=path,
    )

    # S125 hardening: CSRF double-submit cookie
    if _is_csrf_enabled():
        csrf_token = _generate_csrf_token()
        response.set_cookie(
            key=_CSRF_COOKIE,
            value=csrf_token,
            max_age=access_max_age,
            httponly=False,  # Must be readable by JS to send as header
            secure=secure,
            samesite=samesite,
            domain=domain,
            path=path,
        )


def _clear_auth_cookies(response: Response) -> None:
    """Remove httpOnly JWT cookies and CSRF cookie (S125)."""
    jwt_cfg = _get_security_jwt_config()
    domain = jwt_cfg.get("cookie_domain", "") or None
    path = jwt_cfg.get("cookie_path", "/")
    response.delete_cookie(key=_ACCESS_COOKIE, domain=domain, path=path)
    response.delete_cookie(key=_REFRESH_COOKIE, domain=domain, path=path)
    response.delete_cookie(key=_CSRF_COOKIE, domain=domain, path=path)


def _validate_csrf(request: Request) -> None:
    """Validate CSRF double-submit cookie pattern (S125 hardening).

    For state-changing requests (POST, PUT, DELETE), verify that the
    X-CSRF-Token header matches the oo_csrf_token cookie value.
    Skipped for non-cookie auth (Bearer token in header = API/CLI usage).
    """
    if not _is_cookie_mode() or not _is_csrf_enabled():
        return

    # Skip CSRF check if request uses Bearer token (CLI/API client)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return

    # Only check on state-changing methods
    if request.method not in ("POST", "PUT", "DELETE", "PATCH"):
        return

    cookie_token = request.cookies.get(_CSRF_COOKIE, "")
    header_token = request.headers.get("X-CSRF-Token", "")

    if not cookie_token or not header_token:
        raise HTTPException(
            status_code=403,
            detail="CSRF token missing. Include X-CSRF-Token header.",
        )

    # Constant-time comparison
    import hmac as _hmac
    if not _hmac.compare_digest(cookie_token, header_token):
        raise HTTPException(
            status_code=403,
            detail="CSRF token mismatch.",
        )


def _get_current_user(
    request: Request,
    authorization: str | None = Header(default=None),
):
    """Extract and validate the current user (S125: cookie-first, then header).

    In single-user mode, returns a synthetic local user.
    S136 audit fix: Bulbe mode overrides single_user_mode — authentication
    is always required in Bulbe regardless of the single_user_mode setting.
    """
    mgr = _get_auth_manager()

    # S136 audit fix: check Bulbe mode — never bypass auth in Bulbe
    _bulbe_active = False
    try:
        from opti_oignon.security_mode import is_bulbe
        _bulbe_active = is_bulbe()
    except ImportError:
        pass

    # Single-user mode: bypass authentication (except in Bulbe)
    if mgr.single_user_mode and not _bulbe_active:
        return {
            "sub": "local",
            "username": "local",
            "role": "admin",
            "type": "access",
        }

    # S125: Try cookie first if cookie_mode enabled
    token = None
    if _is_cookie_mode():
        token = request.cookies.get(_ACCESS_COOKIE)

    # Fallback: Authorization header (backward compat)
    if not token and authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Use httpOnly cookie or 'Bearer <token>' header.",
        )

    payload = mgr.validate_token(token)
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please log in again.",
        )

    return payload


def _require_admin(current_user: dict = Depends(_get_current_user)):
    """Require the current user to be an admin."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required.",
        )
    return current_user


async def authenticate_websocket(websocket) -> dict | None:
    """Authenticate a WebSocket connection (S136 audit fix).

    Extracts JWT from:
      1. Cookie (oo_access_token) — browser clients
      2. Query parameter (?token=xxx) — CLI/API clients

    S136 audit fix: validates the Origin header to prevent Cross-Site
    WebSocket Hijacking (CSWSH).  A malicious website cannot connect
    to ws://localhost:8000 because the browser sends the attacker's
    Origin, which fails the check.

    Returns the validated payload dict, or None if auth fails.
    In single-user mode (non-Bulbe), returns a synthetic local user.
    """
    # S136 audit fix: validate Origin header (CSWSH prevention)
    origin = websocket.headers.get("origin", "")
    if origin:
        from urllib.parse import urlparse
        parsed = urlparse(origin)
        origin_host = parsed.hostname or ""
        # Allow localhost variants only (or no origin = non-browser client)
        _ALLOWED_WS_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]", ""}
        if origin_host not in _ALLOWED_WS_HOSTS:
            logger.warning(
                "WebSocket CSWSH blocked: origin=%s (host=%s)",
                origin, origin_host,
            )
            return None

    mgr = _get_auth_manager()

    # Check Bulbe mode
    _bulbe_active = False
    try:
        from opti_oignon.security_mode import is_bulbe
        _bulbe_active = is_bulbe()
    except ImportError:
        pass

    # Single-user mode bypass (except in Bulbe)
    if mgr.single_user_mode and not _bulbe_active:
        return {
            "sub": "local",
            "username": "local",
            "role": "admin",
            "type": "access",
        }

    # Try cookie first
    token = websocket.cookies.get(_ACCESS_COOKIE)

    # Fallback: query parameter
    if not token:
        token = websocket.query_params.get("token", "")

    if not token:
        return None

    payload = mgr.validate_token(token)
    return payload


# =============================================================================
# PUBLIC ENDPOINTS (no auth required)
# =============================================================================

@router.get("/status")
def auth_status() -> dict:
    """Get authentication system status."""
    try:
        mgr = _get_auth_manager()
        return {
            "available": True,
            "single_user_mode": mgr.single_user_mode,
            "registration_enabled": mgr.config.get("users", {}).get("allow_registration", True),
            "user_count": mgr.count_users(),
            "cookie_mode": _is_cookie_mode(),
        }
    except HTTPException:
        return {
            "available": False,
            "single_user_mode": True,
            "registration_enabled": False,
            "user_count": 0,
            "cookie_mode": _is_cookie_mode(),
        }


class AuthModeRequest(BaseModel):
    """Request body for toggling authentication mode."""
    single_user_mode: bool = Field(description="True to disable auth, False to enable")


@router.put("/mode")
def set_auth_mode(req: AuthModeRequest) -> dict:
    """Toggle single-user mode (enable/disable authentication).

    S109: Allows users to switch auth on/off from the Settings UI.
    Updates both the in-memory config and the YAML file on disk.
    """
    import yaml
    from pathlib import Path

    mgr = _get_auth_manager()

    # Update in-memory config
    mgr.config["single_user_mode"] = req.single_user_mode

    # Persist to auth.yaml
    config_path = Path(__file__).parent.parent / "config" / "auth.yaml"
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        data["single_user_mode"] = req.single_user_mode
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        logger.info("Auth mode changed: single_user_mode=%s", req.single_user_mode)
    except Exception as e:
        logger.error("Failed to persist auth mode change: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")

    return {
        "single_user_mode": req.single_user_mode,
        "message": "Authentication " + ("disabled" if req.single_user_mode else "enabled"),
    }


@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest, request: Request, response: Response) -> dict:
    """Register a new user account and return tokens.

    S125: Also sets httpOnly cookies when cookie_mode is enabled.
    """
    mgr = _get_auth_manager()

    # Check if registration is allowed
    if not mgr.config.get("users", {}).get("allow_registration", True):
        raise HTTPException(
            status_code=403,
            detail="Registration is disabled. Contact an administrator.",
        )

    user = mgr.create_user(
        username=req.username,
        password=req.password,
        email=req.email,
    )
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Registration failed. Username may already be taken or password too short.",
        )

    # Log audit
    mgr._log_audit(user.user_id, "register", "user", user.user_id)

    tokens = mgr.create_tokens(user)

    # S125: Set httpOnly cookies
    if _is_cookie_mode():
        _set_auth_cookies(
            request,
            response,
            tokens.access_token,
            tokens.refresh_token,
            access_max_age=tokens.expires_in if tokens.expires_in > 0 else 3600,
        )

    return TokenResponse(**tokens.to_dict())


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, request: Request, response: Response) -> dict:
    """Authenticate and get JWT tokens (S127: 2-step 2FA flow).

    Step 1: username + password.
      - If 2FA is NOT active: return JWT tokens immediately.
      - If 2FA IS active: return ``{requires_2fa: true, challenge_id, methods}``
        with NO tokens yet.  Client must call ``POST /login/2fa`` next.

    S124: Rate-limited per IP and per username.
    S125: Sets httpOnly cookies when cookie_mode is enabled.
    """
    mgr = _get_auth_manager()

    # S124: Check rate limit BEFORE attempting authentication
    rate_limiter = _get_rate_limiter()
    client_ip = _extract_client_ip(request)

    if rate_limiter:
        allowed, retry_after = rate_limiter.check_rate_limit(
            client_ip, req.username,
        )
        if not allowed:
            logger.warning(
                "Login rate limited: ip=%s, user=%s, retry_after=%ds",
                client_ip, req.username, retry_after,
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many login attempts. Please try again later."},
                headers={"Retry-After": str(retry_after)},
            )

    user = mgr.authenticate(req.username, req.password)
    if not user:
        # S124: Record failure
        if rate_limiter:
            rate_limiter.record_failure(client_ip, req.username)
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    # S124: Record success (resets username counters)
    if rate_limiter:
        rate_limiter.record_success(client_ip, req.username)

    # S127: Check if 2FA is active for this user
    tfa_mgr = _get_2fa_manager()
    if tfa_mgr is not None:
        try:
            status = tfa_mgr.get_status(user.user_id)
            if status.any_method_active:
                # Build list of available methods
                methods: list[str] = []
                if status.webauthn_enabled:
                    methods.append("webauthn")
                if status.totp_enabled and status.totp_verified:
                    methods.append("totp")
                if status.recovery_codes_remaining > 0:
                    methods.append("recovery")
                if not methods:
                    methods.append("totp")  # Fallback

                challenge_id = _challenge_store.create(
                    user_id=user.user_id, methods=methods,
                )
                logger.info(
                    "2FA challenge issued: user=%s, methods=%s",
                    user.user_id, methods,
                )
                return JSONResponse(
                    status_code=200,
                    content={
                        "requires_2fa": True,
                        "challenge_id": challenge_id,
                        "methods": methods,
                    },
                )
        except Exception as exc:
            logger.warning("2FA status check failed, proceeding without 2FA: %s", exc)

    # No 2FA required: issue tokens immediately
    mgr._log_audit(user.user_id, "login", "user", user.user_id)
    tokens = mgr.create_tokens(user)

    # S125: Set httpOnly cookies
    if _is_cookie_mode():
        _set_auth_cookies(
            request,
            response,
            tokens.access_token,
            tokens.refresh_token,
            access_max_age=tokens.expires_in if tokens.expires_in > 0 else 3600,
        )

    return TokenResponse(**tokens.to_dict())


@router.post("/login/2fa", response_model=TokenResponse)
def login_2fa(req: TwoFALoginRequest, request: Request, response: Response) -> dict:
    """Complete 2FA login (S127 step 2).

    Validates the 2FA code/response against the challenge issued in step 1.
    On success, returns JWT tokens.  On failure, returns 403.

    Rate limited: 5 attempts per challenge, then the challenge is locked
    and the user must restart login from step 1.
    """
    # Retrieve challenge
    challenge = _challenge_store.get(req.challenge_id)
    if challenge is None:
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired 2FA challenge. Please log in again.",
        )

    if challenge["locked"]:
        raise HTTPException(
            status_code=403,
            detail="Too many 2FA attempts. Please log in again.",
        )

    user_id = challenge["user_id"]
    tfa_mgr = _get_2fa_manager()
    if tfa_mgr is None:
        raise HTTPException(
            status_code=503,
            detail="2FA module not available.",
        )

    # Validate 2FA based on method
    success = False
    method_used = req.method

    if req.method == "webauthn" and req.webauthn_response:
        # WebAuthn assertion validation
        try:
            result = tfa_mgr.webauthn_auth_complete(
                user_id, req.webauthn_response,
            )
            success = result.get("success", False)
            method_used = "webauthn"
        except Exception as exc:
            logger.warning("WebAuthn validation error: %s", exc)
            success = False
    else:
        # TOTP, recovery, or auto
        if not req.code:
            raise HTTPException(
                status_code=400,
                detail="A 2FA code is required for this method.",
            )
        result = tfa_mgr.validate_2fa(user_id, req.code, req.method)
        success = result.get("success", False)
        method_used = result.get("method", req.method)

    if not success:
        locked = _challenge_store.record_attempt(req.challenge_id)
        remaining = _ChallengeStore.MAX_ATTEMPTS - (challenge["attempts"] + 1)
        detail = "Invalid 2FA code."
        if locked:
            detail = "Too many 2FA attempts. Please log in again."
        elif remaining > 0:
            detail = f"Invalid 2FA code. {remaining} attempt(s) remaining."
        raise HTTPException(status_code=403, detail=detail)

    # 2FA success: consume challenge and issue tokens
    _challenge_store.consume(req.challenge_id)

    mgr = _get_auth_manager()

    # Retrieve user object for token creation
    user = mgr.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=500,
            detail="User not found after 2FA validation.",
        )

    mgr._log_audit(
        user.user_id, "login_2fa", "user", user.user_id,
        details={"method": method_used},
    )
    tokens = mgr.create_tokens(user)

    # S125: Set httpOnly cookies
    if _is_cookie_mode():
        _set_auth_cookies(
            request,
            response,
            tokens.access_token,
            tokens.refresh_token,
            access_max_age=tokens.expires_in if tokens.expires_in > 0 else 3600,
        )

    return TokenResponse(**tokens.to_dict())


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(
    request: Request,
    response: Response,
    req: RefreshRequest | None = None,
) -> dict:
    """Exchange a refresh token for a new token pair.

    S125: Accepts refresh token from httpOnly cookie or request body.
    """
    mgr = _get_auth_manager()

    # S125: Get refresh token from cookie or body
    rt = None
    if req and req.refresh_token:
        rt = req.refresh_token
    if not rt and _is_cookie_mode():
        rt = request.cookies.get(_REFRESH_COOKIE)
    if not rt:
        raise HTTPException(
            status_code=401,
            detail="Refresh token required.",
        )

    tokens = mgr.refresh_tokens(rt)
    if not tokens:
        # S125: Clear stale cookies on failure
        if _is_cookie_mode():
            _clear_auth_cookies(response)
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired refresh token. Please log in again.",
        )

    # S125: Set new cookies
    if _is_cookie_mode():
        _set_auth_cookies(
            request,
            response,
            tokens.access_token,
            tokens.refresh_token,
            access_max_age=tokens.expires_in if tokens.expires_in > 0 else 3600,
        )

    return TokenResponse(**tokens.to_dict())


@router.post("/logout")
def logout(request: Request, response: Response, req: LogoutRequest | None = None) -> dict:
    """Invalidate a session and clear cookies (S125).

    S125: Accepts refresh token from httpOnly cookie or request body.
    Clears httpOnly cookies on logout.
    """
    mgr = _get_auth_manager()

    # S125: Get refresh token from body or cookie
    rt = None
    if req and req.refresh_token:
        rt = req.refresh_token
    if not rt and _is_cookie_mode():
        rt = request.cookies.get(_REFRESH_COOKIE)

    ok = False
    if rt:
        ok = mgr.logout(rt)

    # S125: Always clear cookies on logout
    if _is_cookie_mode():
        _clear_auth_cookies(response)

    return {"logged_out": ok}


# =============================================================================
# AUTHENTICATED ENDPOINTS
# =============================================================================

@router.get("/me", response_model=UserResponse)
def get_current_profile(current_user: dict = Depends(_get_current_user)) -> dict:
    """Get the current user's profile."""
    mgr = _get_auth_manager()

    if current_user["sub"] == "local":
        return UserResponse(
            user_id="local",
            username="local",
            email="",
            role="admin",
            created_at=0,
            updated_at=0,
        )

    user = mgr.get_user(current_user["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(**user.to_dict())


@router.put("/me", response_model=UserResponse)
def update_profile(
    req: ProfileUpdateRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Update the current user's profile."""
    mgr = _get_auth_manager()

    if current_user["sub"] == "local":
        raise HTTPException(status_code=400, detail="Cannot update local user profile")

    updated = mgr.update_user(
        user_id=current_user["sub"],
        username=req.username,
        email=req.email,
        metadata=req.metadata,
    )
    if not updated:
        raise HTTPException(status_code=400, detail="Update failed (username taken?)")

    return UserResponse(**updated.to_dict())


@router.put("/me/password")
def change_password(
    req: PasswordChangeRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Change the current user's password."""
    mgr = _get_auth_manager()

    if current_user["sub"] == "local":
        raise HTTPException(status_code=400, detail="Cannot change local user password")

    # Verify current password
    user = mgr.authenticate(current_user["username"], req.current_password)
    if not user:
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    ok = mgr.change_password(current_user["sub"], req.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail="Password change failed (too short?)")

    mgr._log_audit(current_user["sub"], "change_password", "user", current_user["sub"])
    return {"changed": True}


# -- User settings -----------------------------------------------------------

@router.get("/settings")
def get_user_settings(current_user: dict = Depends(_get_current_user)) -> dict:
    """Get per-user settings."""
    store = _get_settings_store()
    settings = store.get_settings(current_user["sub"])
    return settings.to_dict()


@router.put("/settings")
def update_user_settings(
    req: SettingsUpdateRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Update per-user settings."""
    store = _get_settings_store()

    kwargs = {}
    if req.theme is not None:
        kwargs["theme"] = req.theme
    if req.default_model is not None:
        kwargs["default_model"] = req.default_model
    if req.default_preset is not None:
        kwargs["default_preset"] = req.default_preset
    if req.sidebar_open is not None:
        kwargs["sidebar_open"] = req.sidebar_open
    if req.language is not None:
        kwargs["language"] = req.language
    if req.preferences is not None:
        kwargs["preferences"] = req.preferences

    settings = store.update_settings(current_user["sub"], **kwargs)
    return settings.to_dict()


# -- Admin endpoints ---------------------------------------------------------

@router.get("/users", response_model=list[UserResponse])
def list_users(
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    admin: dict = Depends(_require_admin),
) -> list:
    """List all users (admin only)."""
    mgr = _get_auth_manager()
    users = mgr.list_users(limit=limit, offset=offset)
    return [UserResponse(**u.to_dict()) for u in users]


@router.delete("/users/{user_id}")
def delete_user(
    user_id: str,
    admin: dict = Depends(_require_admin),
) -> dict:
    """Delete a user account (admin only)."""
    mgr = _get_auth_manager()

    if user_id == admin["sub"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    ok = mgr.delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")

    mgr._log_audit(admin["sub"], "delete_user", "user", user_id)
    return {"deleted": True, "user_id": user_id}


# -- Shared projects ---------------------------------------------------------

@router.post("/projects/share")
def share_project(
    req: ShareProjectRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Share a project with another user."""
    mgr = _get_auth_manager()

    # Find target user by username
    target = mgr.get_user_by_username(req.username)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    result = mgr.share_project(
        project_id=req.project_id,
        user_id=target.user_id,
        role=req.role,
        invited_by=current_user["sub"],
    )
    if not result:
        raise HTTPException(status_code=400, detail="Failed to share project (invalid role?)")

    return result


@router.get("/projects/{project_id}/members")
def list_project_members(
    project_id: str,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """List members of a shared project."""
    mgr = _get_auth_manager()
    members = mgr.list_project_members(project_id)
    return {"project_id": project_id, "members": members}


@router.delete("/projects/{project_id}/members/{user_id}")
def remove_project_member(
    project_id: str,
    user_id: str,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Remove a user's access to a project."""
    mgr = _get_auth_manager()

    # Check caller has owner role (or is admin)
    if current_user.get("role") != "admin":
        if not mgr.check_permission(project_id, current_user["sub"], "owner"):
            raise HTTPException(
                status_code=403,
                detail="Only the project owner can remove members.",
            )

    ok = mgr.remove_project_access(project_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Member not found in project")

    return {"removed": True, "project_id": project_id, "user_id": user_id}


# -- Audit log ---------------------------------------------------------------

@router.get("/audit")
def get_audit_log(
    user_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    admin: dict = Depends(_require_admin),
) -> dict:
    """Get audit log entries (admin only)."""
    mgr = _get_auth_manager()
    entries = mgr.get_audit_log(user_id=user_id, project_id=project_id, limit=limit)
    return {"entries": entries, "count": len(entries)}
