#!/usr/bin/env python3
"""
Authentication API routes.

POST   /api/auth/register           -- Create account (the first, or where registration is open)
POST   /api/auth/login              -- Get JWT token (or 2FA challenge)
POST   /api/auth/login/2fa          -- Complete 2FA challenge
POST   /api/auth/login/2fa/enrol/begin    -- Begin the TOTP of an enrolment challenge (Bulbe, no factor yet)
POST   /api/auth/login/2fa/enrol/complete -- Confirm it: the first session and the recovery codes
POST   /api/auth/logout             -- Invalidate session
POST   /api/auth/refresh            -- Refresh token pair
GET    /api/auth/me                 -- Get current user profile
PUT    /api/auth/me                 -- Update profile
PUT    /api/auth/me/password        -- Change password
GET    /api/auth/users              -- List users (the owner)
POST   /api/auth/users              -- Add an account (the owner, with the owner's password)
DELETE /api/auth/users/{user_id}    -- Delete user (the owner)
GET    /api/auth/settings           -- Get per-user settings
PUT    /api/auth/settings           -- Update per-user settings
POST   /api/auth/projects/share     -- Share a project
GET    /api/auth/projects/{id}/members -- List project members
DELETE /api/auth/projects/{project_id}/members/{user_id} -- Remove access
GET    /api/auth/audit              -- Get audit log (the owner)
GET    /api/auth/status             -- Auth system status (public)
"""

import logging
import secrets
import threading
import time
import unicodedata
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
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


class CreateUserRequest(BaseModel):
    """Request body for an account the platform's owner adds."""
    username: str = Field(min_length=2, max_length=64, description="Username")
    password: str = Field(min_length=8, description="The new account's password")
    email: str = Field(default="", description="Email address (optional)")
    role: str | None = Field(default=None, description="admin, user or viewer; users.default_role when not given")
    owner_password: str = Field(description="The owner's own password")


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


class EnrolledTokenResponse(TokenResponse):
    """The first session of an account that enrolled its second factor at login, and its recovery codes, shown once:
    none when they could not be made (the settings make them)."""
    recovery_codes: list[str] = []


class EnrolBeginRequest(BaseModel):
    """Request body for the first step of an enrolment at login."""
    challenge_id: str = Field(description="Enrolment challenge ID from step-1 login")


class EnrolCompleteRequest(BaseModel):
    """Request body for the last step of an enrolment at login."""
    challenge_id: str = Field(description="Enrolment challenge ID from step-1 login")
    code: str = Field(default="", description="A code of the TOTP the first step began")


class TwoFALoginRequest(BaseModel):
    """Request body for 2FA login step."""
    challenge_id: str = Field(description="Challenge ID from step-1 login")
    code: str = Field(default="", description="TOTP or recovery code")
    method: str = Field(default="auto", description="2FA method: auto, totp, recovery, webauthn")
    webauthn_response: dict[str, Any] | None = Field(
        default=None, description="WebAuthn assertion response object"
    )


# =============================================================================
# 2FA CHALLENGE STORE
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
      - enrol: True for the challenge of an account that enrols its first factor (Bulbe), answered through
        ``/login/2fa/enrol/*`` alone

    Thread-safe via a simple lock.  Periodic cleanup of expired
    entries happens on every create() call.
    """

    CHALLENGE_TTL_SECONDS = 300  # 5 minutes
    MAX_ATTEMPTS = 5

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, user_id: str, methods: list[str], *, webauthn_key: str = "", enrol: bool = False) -> str:
        """Create a new challenge and return its ID; ``webauthn_key`` names the security key's assertion state this
        login began, its own; ``enrol`` marks the challenge of an enrolment."""
        self._cleanup()
        challenge_id = secrets.token_urlsafe(32)
        with self._lock:
            self._store[challenge_id] = {
                "user_id": user_id,
                "created_at": time.time(),
                "attempts": 0,
                "locked": False,
                "methods": methods,
                "webauthn_key": webauthn_key,
                "enrol": enrol,
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

    def take_attempt(self, challenge_id: str) -> int | None:
        """Count a 2FA try before its answer is read, and return the tries counted with it: tries sent together are
        each counted before any is checked, and one past the cap is never checked. None when the challenge is gone or
        expired, 0 when it is locked; the try that reaches the cap locks it."""
        with self._lock:
            entry = self._store.get(challenge_id)
            if entry is None or time.time() - entry["created_at"] > self.CHALLENGE_TTL_SECONDS:
                return None
            if entry["locked"]:
                return 0
            entry["attempts"] += 1
            if entry["attempts"] >= self.MAX_ATTEMPTS:
                entry["locked"] = True
            return entry["attempts"]

    def renew(self, challenge_id: str) -> None:
        """Start a live challenge's time again: an enrolment's five minutes run from its latest begin, the time it
        takes to scan the secret that begin made."""
        with self._lock:
            entry = self._store.get(challenge_id)
            if entry is not None:
                entry["created_at"] = time.time()

    def consume(self, challenge_id: str) -> bool:
        """Remove a challenge after successful 2FA validation; True for the one caller that removed it."""
        with self._lock:
            return self._store.pop(challenge_id, None) is not None

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
    """Get the 2FA manager singleton.

    Returns None if the auth_2fa module is not available: login then refuses, as its second step does.
    """
    try:
        from opti_oignon.auth_2fa import two_factor_manager
        return two_factor_manager
    except Exception:
        return None


_NO_FACTOR_SET_UP = "A second factor cannot be set up here: login refused."

_REGISTRATION_CLOSED = ("Registration is closed: the platform's owner adds accounts (Workshop > Security > "
                        "Authentication mode).")
_REGISTRATION_UNREAD = ("Registration is closed while the auth configuration does not load; fix auth.yaml and restart "
                        "the server.")
_REGISTRATION_FULL = "Registration is closed: the platform holds its users.max_users accounts."

# The roles an account may hold (``auth.VALID_ROLES``).
_ROLES = ("admin", "user", "viewer")


def _registration_open(mgr: Any) -> bool:
    """Whether registration is open past the first account: ``users.allow_registration`` true and the mode Daily. In
    Bulbe, and in a mode that cannot be read, it never is."""
    if mgr.config.get("users", {}).get("allow_registration") is not True:
        return False
    try:
        from opti_oignon.security_mode import is_bulbe
        return not is_bulbe()
    except Exception:  # noqa: BLE001 - a mode that cannot be read opens nothing
        return False


def _registration_closed(mgr: Any) -> str | None:
    """Why ``/register`` refuses now, or None when it would make an account: shut while the auth configuration does
    not load whole; open to the first account of a store that holds none; past it, open only where
    ``_registration_open`` says so and below ``users.max_users``. Raises when the store cannot be read."""
    if getattr(mgr, "config_unread", None):
        return _REGISTRATION_UNREAD
    count = mgr.count_users()
    if count == 0:
        return None
    if not _registration_open(mgr):
        return _REGISTRATION_CLOSED
    cap = mgr.config.get("users", {}).get("max_users", 0)
    if isinstance(cap, int) and not isinstance(cap, bool) and 0 < cap <= count:
        return _REGISTRATION_FULL
    return None


def _acting_account(current_user: dict) -> str:
    """The account a session stands for, as the store says now (``ceremony_factors.session_account``: for the
    single-user session, the owner's when there is one), or the session's own ``sub`` when the store cannot say."""
    sub = current_user.get("sub") or ""
    try:
        from opti_oignon.ceremony_factors import session_account
        account = session_account(current_user)
    except Exception:  # noqa: BLE001 - the record and the check keep the session's name
        return sub
    return account.account_id if account is not None else sub


def _owner_password(act: str, current_user: dict, password: str, request: Request) -> str:
    """The owner's password, asked through the owner's own session as the security ceremonies ask it
    (``ceremony_factors.check_password``): the owner's account id, or a refusal raised with the verdict's status --
    403 for no account, another account's session or a wrong password, 429 for a password being checked or an account
    the login limiter holds back, 503 for one that cannot be checked -- and audited as ``ceremony_password_refused``
    by name, with no secret, where the audit can write."""
    from opti_oignon.ceremony_factors import check_password

    verdict = check_password(current_user, password, client_ip=_extract_client_ip(request))
    if verdict.passed:
        return verdict.account_id
    logger.warning("ceremony_password_refused: %s", verdict.error)
    try:
        from opti_oignon.security_mode import _audit_log
        _audit_log("ceremony_password_refused", severity="WARNING", ceremony=act, error=verdict.error,
                   user_id=current_user.get("sub") or "")
    except Exception:  # noqa: BLE001 - the audit is best effort; the refusal stands
        pass
    raise HTTPException(status_code=verdict.status, detail=verdict.message)


def _registration_enabled(mgr: Any) -> bool:
    """What ``/register`` answers now (``_registration_closed``); a store that cannot be read registers no one."""
    try:
        return _registration_closed(mgr) is None
    except Exception:  # noqa: BLE001 - a store that cannot be read registers no one
        return False


def _enrolment_challenge(tfa_mgr: Any, user: Any) -> JSONResponse:
    """What a password alone gives, in Bulbe, an account with no second factor: no token, the challenge of its
    enrolment -- a TOTP begun and confirmed through ``/login/2fa/enrol/*``, its first session with it --, or 503 when
    no factor can be set up here. Refusing would leave the account no way to enrol: the routes that add a factor
    need a session."""
    if not tfa_mgr.totp_checkable():
        logger.warning("No second factor can be set up here: login refused for user=%s", user.user_id)
        raise HTTPException(status_code=503, detail=_NO_FACTOR_SET_UP)
    challenge_id = _challenge_store.create(user_id=user.user_id, methods=["totp"], enrol=True)
    logger.info("2FA enrolment asked at login: user=%s", user.user_id)
    return JSONResponse(status_code=200, content={
        "requires_2fa_enrolment": True,
        "challenge_id": challenge_id,
        "methods": ["totp"],
    })


# The addresses and the accounts whose login password is being checked: one check at a time for each, refused rather
# than queued, so that tries sent together meet the limiter one after the other -- each failure counted before the
# next try is asked -- and no worker thread waits on another.
_logging_in: set[tuple[str, str]] = set()
_logging_in_lock = threading.Lock()


def _claim_login(client_ip: str, username: str) -> bool:
    keys = {("address", client_ip), ("account", username)}
    with _logging_in_lock:
        if keys & _logging_in:
            return False
        _logging_in.update(keys)
        return True


def _release_login(client_ip: str, username: str) -> None:
    with _logging_in_lock:
        _logging_in.difference_update({("address", client_ip), ("account", username)})


def _get_rate_limiter():
    """Get the login rate limiter singleton."""
    try:
        from opti_oignon.auth import login_rate_limiter
        return login_rate_limiter
    except Exception:
        return None


def _extract_client_ip(request: Request) -> str:
    """Extract client IP from request (audit-hardened).

    Audit fix: X-Forwarded-For is NOT trusted by default because
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
    """Load JWT cookie configuration from security.yaml."""
    from pathlib import Path

    import yaml
    cfg_path = Path(__file__).parent.parent / "config" / "security.yaml"
    try:
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("jwt", {})
    except Exception:
        pass
    return {"cookie_mode": True}


def _is_cookie_mode() -> bool:
    """Check if httpOnly cookie mode is enabled."""
    return _get_security_jwt_config().get("cookie_mode", True)


def _is_csrf_enabled() -> bool:
    """Check if CSRF protection is enabled."""
    return _get_security_jwt_config().get("csrf_enabled", True)


# Cookie names for httpOnly JWT storage
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
    """Check if the request comes from a non-localhost client.

    Used to apply hardened cookie settings for remote sessions.
    A guard that cannot be read takes the client for a remote one: its cookies are the hardened ones.
    """
    try:
        from opti_oignon.remote_session_guard import is_remote_request
        client_host = request.client.host if request.client else None
        return is_remote_request(client_host)
    except Exception:  # noqa: BLE001 - an unknown client gets the hardened cookies
        return True


def _set_auth_cookies(
    request: Request,
    response: Response,
    access_token: str,
    refresh_token: str,
    access_max_age: int = 3600,
) -> None:
    """Set httpOnly JWT cookies on a response (hardened).

    Access token: short-lived (default 1h local, 5min remote).
    Refresh token: long-lived (30 days local, 1h remote).
    Both httpOnly (not accessible via JS).
    SameSite: Lax local, Strict remote.
    Secure flag auto-detected from request context.
    CSRF token set as a non-httpOnly cookie (readable by JS for header).
    """
    jwt_cfg = _get_security_jwt_config()
    secure = _detect_secure_context(request)
    samesite = jwt_cfg.get("cookie_samesite", "lax")
    domain = jwt_cfg.get("cookie_domain", "") or None
    path = jwt_cfg.get("cookie_path", "/")

    # Detect remote context and apply hardened settings
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

    # Hardening: CSRF double-submit cookie
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
    """Remove httpOnly JWT cookies and CSRF cookie."""
    jwt_cfg = _get_security_jwt_config()
    domain = jwt_cfg.get("cookie_domain", "") or None
    path = jwt_cfg.get("cookie_path", "/")
    response.delete_cookie(key=_ACCESS_COOKIE, domain=domain, path=path)
    response.delete_cookie(key=_REFRESH_COOKIE, domain=domain, path=path)
    response.delete_cookie(key=_CSRF_COOKIE, domain=domain, path=path)


def _validate_csrf(request: Request) -> None:
    """Validate CSRF double-submit cookie pattern.

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
    """Extract and validate the current user (cookie-first, then header).

    In single-user mode, returns a synthetic local user.
    Audit fix: Bulbe mode overrides single_user_mode -- authentication
    is always required in Bulbe regardless of the single_user_mode setting.
    """
    mgr = _get_auth_manager()

    # Audit fix: check Bulbe mode -- never bypass auth in Bulbe; a mode that cannot be read is taken for Bulbe.
    try:
        from opti_oignon.security_mode import is_bulbe
        _bulbe_active = is_bulbe()
    except Exception:  # noqa: BLE001 - an unknown mode never bypasses authentication
        _bulbe_active = True

    # Single-user mode: bypass authentication (except in Bulbe)
    if mgr.single_user_mode and not _bulbe_active:
        return {
            "sub": "local",
            "username": "local",
            "role": "admin",
            "type": "access",
        }

    # Try cookie first if cookie_mode enabled
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


def _require_owner(current_user: dict = Depends(_get_current_user)):
    """Require the session to be the platform owner's: the synthetic session of single-user mode, or an account the
    store names an owner now (``AuthManager.owners``: an admin's or, while none is, the first registered) -- never the
    role a token was signed with, which an account deleted or demoted keeps until it expires. 403 for any other
    session and when the owner cannot be told; 503, as everywhere, when the account store is not there."""
    if current_user.get("sub") == "local" and current_user.get("role") == "admin":
        return current_user
    mgr = _get_auth_manager()
    try:
        owner = mgr is not None and mgr.is_owner(current_user.get("sub") or "")
    except Exception:  # noqa: BLE001 - an owner who cannot be told is no owner
        owner = False
    if not owner:
        raise HTTPException(status_code=403, detail="Only the platform's owner may do this.")
    return current_user


async def authenticate_websocket(websocket) -> dict | None:
    """Authenticate a WebSocket connection (audit-hardened).

    Extracts JWT from:
      1. Cookie (oo_access_token) -- browser clients
      2. Query parameter (?token=xxx) -- CLI/API clients

    Cross-Site WebSocket Hijacking (CSWSH): the Host gate
    (``api/host_gate.py``) closes, before any route, a handshake whose
    Origin is not a page of this site -- the request's scheme and the
    Host's name and port, or an origin listed under ``cors.origins`` --,
    so a page of another port or scheme of this machine opens no socket,
    and it marks the handshakes it lets through (``ORIGIN_CHECKED`` in the
    scope), which this check takes as they are: a listed name over TLS, a
    listed origin. A socket the gate did not see is held here to an Origin
    whose host is a loopback name, any port, any scheme.

    Returns the validated payload dict, or None if auth fails.
    In single-user mode (non-Bulbe), returns a synthetic local user.
    """
    # The gate's mark (host_gate.ORIGIN_CHECKED): its rule held. Without it, no Origin at all is a client that is no
    # browser; an opaque one -- "null", from a sandboxed frame or a file -- or one without a loopback host is a page of
    # anywhere, refused.
    scope = getattr(websocket, "scope", None)
    checked = isinstance(scope, dict) and scope.get("opti_oignon.origin_checked") is True
    origin = websocket.headers.get("origin")
    if origin is not None and not checked:
        from urllib.parse import urlparse
        parsed = urlparse(origin)
        origin_host = parsed.hostname or ""
        _ALLOWED_WS_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}
        if origin_host not in _ALLOWED_WS_HOSTS:
            logger.warning(
                "WebSocket CSWSH blocked: origin=%s (host=%s)",
                origin, origin_host,
            )
            return None

    mgr = _get_auth_manager()

    # Check Bulbe mode; a mode that cannot be read is taken for Bulbe.
    try:
        from opti_oignon.security_mode import is_bulbe
        _bulbe_active = is_bulbe()
    except Exception:  # noqa: BLE001 - an unknown mode never bypasses authentication
        _bulbe_active = True

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
            "registration_enabled": _registration_enabled(mgr),
            "user_count": mgr.count_users(),
            "cookie_mode": _is_cookie_mode(),
            # The auth configuration does not load whole: sign-in is asked and registration closed until it does
            "auth_config_unread": bool(getattr(mgr, "config_unread", None)),
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
def set_auth_mode(req: AuthModeRequest, current_user: dict = Depends(_require_owner)) -> dict:
    """Toggle single-user mode (enable/disable authentication).

    Allows the platform's owner to switch auth on/off from the Settings UI (403 for anyone else).
    Updates both the in-memory config and the YAML file on disk, and answers the mode in force, not the one asked:
    with more than one account the store keeps authentication on whatever the flag says.
    """
    import os
    import stat
    import tempfile
    from pathlib import Path

    import yaml

    mgr = _get_auth_manager()
    # A configuration that does not load whole has closed the door: the mode is not changed over it
    if getattr(mgr, "config_unread", None):
        raise HTTPException(
            status_code=409,
            detail="The auth configuration does not load; fix auth.yaml and restart the server first.",
        )

    # Persist to auth.yaml first, whole -- a new file beside it (owner-only, then the file's own rights) renamed
    # over it, through a link at its name -- then the mode in memory: a save that fails changes nothing
    config_path = Path(os.path.realpath(Path(__file__).parent.parent / "config" / "auth.yaml"))
    temp = None
    try:
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError("auth.yaml holds no mapping")
            rights = stat.S_IMODE(os.stat(config_path).st_mode)
        else:
            data = {}
            rights = 0o600

        data["single_user_mode"] = req.single_user_mode
        handle, temp = tempfile.mkstemp(prefix=f".{config_path.name}.", suffix=".tmp", dir=config_path.parent)
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            f.flush()
            os.fchmod(f.fileno(), rights)
            os.fsync(f.fileno())
        os.replace(temp, config_path)
        temp = None

        logger.info("Auth mode changed: single_user_mode=%s", req.single_user_mode)
    except Exception as e:
        if temp is not None:
            try:
                os.unlink(temp)
            except OSError:
                pass
        logger.error("Failed to persist auth mode change: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")

    # Update in-memory config
    mgr.config["single_user_mode"] = req.single_user_mode

    in_force = bool(mgr.single_user_mode)
    message = "Authentication " + ("disabled" if in_force else "enabled")
    if req.single_user_mode and not in_force:
        message += ": it stays on while more than one account exists"
    return {"single_user_mode": in_force, "message": message}


@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest, request: Request, response: Response) -> dict:
    """Register a new user account and return tokens -- in Bulbe, the challenge of its second factor's enrolment
    instead (``login``): its first session comes with its first factor.

    Registration makes the first account of a store that holds none, in any mode; past it, an account only where
    ``users.allow_registration`` is true and the mode is Daily (``_registration_open``), below ``users.max_users`` --
    otherwise the owner adds it (``POST /users``). The store counts and writes in one transaction and reads the mode
    again there (``first_only``), so that two registrations sent together on an empty store make one account, and a
    switch to Bulbe while the password is hashed makes none past the first. Refused 403 while the auth configuration
    does not load whole, the first account included, and 503 when the store cannot be read.

    Also sets httpOnly cookies when cookie_mode is enabled.
    """
    mgr = _get_auth_manager()

    # Who may register, told before the password is hashed
    try:
        closed = _registration_closed(mgr)
    except Exception:  # noqa: BLE001 - a store that cannot be read registers no one
        raise HTTPException(status_code=503, detail="The account store cannot be read: registration refused.")
    if closed:
        raise HTTPException(status_code=403, detail=closed)

    # Whether the new account enrols a second factor first, told before it is made: a two-factor module that cannot
    # be reached, or a factor that cannot be set up where one is asked, refuses with no account left behind.
    tfa_mgr = _get_2fa_manager()
    if tfa_mgr is None:
        raise HTTPException(
            status_code=503,
            detail="The second factor cannot be checked here: registration refused.",
        )
    enrols = tfa_mgr.required_by_mode()
    if enrols and not tfa_mgr.totp_checkable():
        raise HTTPException(
            status_code=503,
            detail="A second factor cannot be set up here: registration refused.",
        )

    try:
        user = mgr.create_user(
            username=req.username,
            password=req.password,
            email=req.email,
            first_only=lambda: not _registration_open(mgr),
        )
    except Exception:  # noqa: BLE001 - a store that cannot be written registers no one
        raise HTTPException(status_code=503, detail="The account store cannot be written: registration refused.")
    if not user:
        # Closed meanwhile -- the first account made, the mode raised to Bulbe, users.max_users reached -- as it would
        # have been told; else the name or the password
        try:
            closed = _registration_closed(mgr)
        except Exception:  # noqa: BLE001 - the refusal stands, in its general words
            closed = None
        if closed:
            raise HTTPException(status_code=403, detail=closed)
        raise HTTPException(
            status_code=400,
            detail="Registration failed: the name may be taken, or the password refused by the rules.",
        )

    # Log audit
    mgr._log_audit(user.user_id, "register", "user", user.user_id)

    if enrols:
        return _enrolment_challenge(tfa_mgr, user)

    tokens = mgr.create_tokens(user)

    # Set httpOnly cookies
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
    """Authenticate and get JWT tokens (2-step 2FA flow).

    Step 1: username + password.
      - If 2FA is NOT active: return JWT tokens immediately.
      - If 2FA IS active: return ``{requires_2fa: true, challenge_id, methods}``
        with NO tokens yet.  Client must call ``POST /login/2fa`` next.

    Rate-limited per IP and per username, one password checked at a time for each: a login that finds one being
    checked for its address or its username is refused (429) before its password is read, and is not counted.
    Sets httpOnly cookies when cookie_mode is enabled.
    """
    mgr = _get_auth_manager()

    # Check rate limit BEFORE attempting authentication
    rate_limiter = _get_rate_limiter()
    client_ip = _extract_client_ip(request)

    if not _claim_login(client_ip, req.username):
        logger.warning("Login refused while another is checked: ip=%s, user=%s", client_ip, req.username)
        return JSONResponse(
            status_code=429,
            content={"detail": "A login for this account or from this address is being checked. Try again."},
            headers={"Retry-After": "1"},
        )
    try:
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
            # Record failure
            if rate_limiter:
                rate_limiter.record_failure(client_ip, req.username)
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password.",
            )

        # Record success (resets username counters)
        if rate_limiter:
            rate_limiter.record_success(client_ip, req.username)
    finally:
        _release_login(client_ip, req.username)

    # Check if 2FA is active for this user. A second factor whose state cannot be read -- the two-factor module
    # unreachable, its store unreadable -- refuses the login: it never lets one through without the factor it may
    # require.
    tfa_mgr = _get_2fa_manager()
    if tfa_mgr is None:
        logger.warning("2FA module unreachable: login refused for user=%s", user.user_id)
        raise HTTPException(
            status_code=503,
            detail="The second factor cannot be checked here: login refused.",
        )
    try:
        status = tfa_mgr.get_status(user.user_id)
    except Exception as exc:  # noqa: BLE001 - an unreadable second factor refuses
        logger.warning("2FA status unreadable (%s): login refused for user=%s", type(exc).__name__, user.user_id)
        raise HTTPException(
            status_code=503,
            detail="The second factor's state cannot be read: login refused.",
        ) from exc
    if status.any_method_active:
        # The methods this account can answer with, among those that can be checked here: a security key only when
        # its assertion can be asked now -- its options travel with this answer --, a TOTP once verified, a recovery
        # code while one remains. App passwords are never a second factor. None that can be checked refuses.
        methods: list[str] = []
        webauthn_options = None
        webauthn_key = ""
        if status.webauthn_enabled:
            # The assertion's state is this login's own: another login of the account never overwrites it.
            webauthn_key = "login:" + secrets.token_urlsafe(16)
            try:
                begun = tfa_mgr.webauthn_auth_begin(user.user_id, key=webauthn_key)
            except Exception as exc:  # noqa: BLE001 - a key that cannot be asked is not offered
                begun = {"success": False, "error": type(exc).__name__}
            if begun.get("success"):
                methods.append("webauthn")
                webauthn_options = begun.get("options")
            else:
                logger.warning("Security key not asked at login (%s): user=%s", begun.get("error"), user.user_id)
        if status.totp_enabled and status.totp_verified and tfa_mgr.totp_checkable():
            methods.append("totp")
        if status.recovery_codes_remaining > 0:
            methods.append("recovery")
        if not methods:
            logger.warning("No second factor of user=%s can be checked here: login refused", user.user_id)
            raise HTTPException(
                status_code=503,
                detail="The second factor cannot be checked here: login refused.",
            )

        challenge_id = _challenge_store.create(
            user_id=user.user_id, methods=methods,
            webauthn_key=webauthn_key if "webauthn" in methods else "",
        )
        logger.info(
            "2FA challenge issued: user=%s, methods=%s",
            user.user_id, methods,
        )
        content: dict[str, Any] = {
            "requires_2fa": True,
            "challenge_id": challenge_id,
            "methods": methods,
        }
        if webauthn_options is not None:
            content["webauthn_options"] = webauthn_options
        return JSONResponse(status_code=200, content=content)

    # No factor active, and one asked all the same (Bulbe, or a mode that cannot be read): the password alone gives
    # no session, the enrolment of a first factor instead.
    if tfa_mgr.is_2fa_required(user.user_id, status=status):
        return _enrolment_challenge(tfa_mgr, user)

    # No 2FA required: issue tokens immediately
    mgr._log_audit(user.user_id, "login", "user", user.user_id)
    tokens = mgr.create_tokens(user)

    # Set httpOnly cookies
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
    """Complete 2FA login (step 2).

    Validates the 2FA code/response against the challenge issued in step 1.
    On success, returns JWT tokens.  On failure, returns 403. A code that
    cannot be checked -- a TOTP secret that is sealed and does not decrypt,
    recovery codes keyed with no key that loads -- is 503 with the reason: none
    is called wrong.

    Rate limited: 5 attempts per challenge, each counted before its answer is
    read, then the challenge is locked and the user must restart login from
    step 1.
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

    # An enrolment is answered through its own routes: no code of a factor the account does not hold yet is tried.
    if challenge.get("enrol"):
        raise HTTPException(
            status_code=403,
            detail="This login enrols a second factor first. Please log in again.",
        )

    user_id = challenge["user_id"]
    tfa_mgr = _get_2fa_manager()
    if tfa_mgr is None:
        raise HTTPException(
            status_code=503,
            detail="2FA module not available.",
        )

    # Only what the first step offered: a method it did not offer -- an app password among them -- is a failed try.
    # An offered method's answer that is not there is refused before any try is counted: nothing is read.
    offered = list(challenge.get("methods") or [])
    not_offered = req.method != "auto" and req.method not in offered
    # TOTP or recovery: the code in its plain form -- the spaces an app shows inside it removed, compatibility forms
    # (fullwidth digits) folded as the TOTP library folds them.
    code = "".join(unicodedata.normalize("NFKC", req.code or "").split())
    if not not_offered and req.method == "webauthn" and not req.webauthn_response:
        raise HTTPException(
            status_code=400,
            detail="A security key's answer is required for this method.",
        )
    if not not_offered and req.method != "webauthn" and not code:
        raise HTTPException(
            status_code=400,
            detail="A 2FA code is required for this method.",
        )

    # The try counted before its answer is read: tries sent together are each counted, and one past the cap is
    # refused unread.
    taken = _challenge_store.take_attempt(req.challenge_id)
    if not taken:
        raise HTTPException(
            status_code=403,
            detail=("Invalid or expired 2FA challenge. Please log in again." if taken is None
                    else "Too many 2FA attempts. Please log in again."),
        )
    locked = taken >= _ChallengeStore.MAX_ATTEMPTS

    if not_offered:
        raise HTTPException(
            status_code=403,
            detail=("Too many 2FA attempts. Please log in again." if locked
                    else "This method was not offered for this login."),
        )

    # Validate 2FA based on method
    success = False
    method_used = req.method

    if req.method == "webauthn":
        # WebAuthn assertion validation, against the options the first step gave
        try:
            result = tfa_mgr.webauthn_auth_complete(
                user_id, req.webauthn_response, key=challenge.get("webauthn_key") or None,
            )
            success = result.get("success", False)
            method_used = "webauthn"
        except Exception as exc:
            logger.warning("WebAuthn validation error: %s", exc)
            success = False
    else:
        # TOTP or recovery, as offered. A code is tried as a recovery code only when it has one's shape (as many
        # hexadecimal characters as the generator makes), whichever method names it, and auto tries any other as a
        # TOTP code -- so that only a well-formed recovery code ever spends one of their few tries.
        if req.method == "auto":
            shaped = "recovery" if tfa_mgr.recovery_shaped(code) else "totp"
            tried = [shaped] if shaped in offered else []
        elif req.method == "recovery" and not tfa_mgr.recovery_shaped(code):
            tried = []
        else:
            tried = [req.method]
        for method in tried:
            try:
                result = tfa_mgr.validate_2fa(user_id, code.lower() if method == "recovery" else code, method)
            except Exception as exc:
                # The field encryption's refusal (encryption.EncryptionUnavailable), asked by its class's name: a
                # TOTP secret that does not decrypt, or keyed recovery codes with no key. The challenge's try is spent.
                if type(exc).__name__ != "EncryptionUnavailable" or type(exc).__module__ != "opti_oignon.encryption":
                    raise
                raise HTTPException(status_code=503, detail=f"2FA code not checked: {exc}") from exc
            if result.get("success", False):
                success = True
                method_used = result.get("method", method)
                break

    if not success:
        remaining = _ChallengeStore.MAX_ATTEMPTS - taken
        detail = "Invalid 2FA code."
        if locked:
            detail = "Too many 2FA attempts. Please log in again."
        elif remaining > 0:
            detail = f"Invalid 2FA code. {remaining} attempt(s) remaining."
        raise HTTPException(status_code=403, detail=detail)

    # 2FA success: consume the challenge and issue tokens -- once: of two answers that pass together, one signs in,
    # and a challenge that expired while the answer was checked has ended as well
    if not _challenge_store.consume(req.challenge_id):
        raise HTTPException(
            status_code=403,
            detail="This login has ended. Please log in again.",
        )

    mgr = _get_auth_manager()

    # Retrieve user object for token creation
    user = mgr.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=403,
            detail="This account no longer exists. Please log in again.",
        )

    mgr._log_audit(
        user.user_id, "login_2fa", "user", user.user_id,
        details={"method": method_used},
    )
    tokens = mgr.create_tokens(user)

    # Set httpOnly cookies
    if _is_cookie_mode():
        _set_auth_cookies(
            request,
            response,
            tokens.access_token,
            tokens.refresh_token,
            access_max_age=tokens.expires_in if tokens.expires_in > 0 else 3600,
        )

    return TokenResponse(**tokens.to_dict())


def _live_enrolment(challenge_id: str) -> tuple[dict[str, Any], Any]:
    """The enrolment challenge and the second-factor manager, while the account still has no factor; or the refusal
    that sends back to the password -- an unknown, expired or locked challenge, one of a second step, one whose
    account has a factor active since (its own other login enrolled one) -- or 503 when the second factor's state
    cannot be read."""
    challenge = _challenge_store.get(challenge_id)
    if challenge is None or not challenge.get("enrol"):
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired enrolment. Please log in again.",
        )
    if challenge["locked"]:
        raise HTTPException(
            status_code=403,
            detail="Too many 2FA attempts. Please log in again.",
        )
    tfa_mgr = _get_2fa_manager()
    if tfa_mgr is None:
        raise HTTPException(
            status_code=503,
            detail="2FA module not available.",
        )
    try:
        status = tfa_mgr.get_status(challenge["user_id"])
    except Exception as exc:  # noqa: BLE001 - an unreadable second factor refuses
        raise HTTPException(
            status_code=503,
            detail="The second factor's state cannot be read: login refused.",
        ) from exc
    if status.any_method_active:
        _factor_since(challenge_id)
    return challenge, tfa_mgr


def _factor_since(challenge_id: str) -> None:
    """An enrolment whose account has a factor active since -- its own other login enrolled one --: the challenge
    is spent and the page sent back to the password."""
    _challenge_store.consume(challenge_id)
    raise HTTPException(
        status_code=403,
        detail="This account has a second factor now. Please log in again.",
    )


@router.post("/login/2fa/enrol/begin")
def login_2fa_enrol_begin(req: EnrolBeginRequest) -> dict:
    """Begin the TOTP of an enrolment challenge: its secret, its provisioning URI and, where it can be drawn, its QR
    code. It may be begun again until it is confirmed; the latest begun is the one the codes are checked against, and
    the enrolment's five minutes run from it. A TOTP confirmed meanwhile is never replaced: the setup refuses in the
    statement that writes, and the page is sent back to log in. 503 when encryption is on and the secret cannot be
    sealed."""
    challenge, tfa_mgr = _live_enrolment(req.challenge_id)
    user = _get_auth_manager().get_user(challenge["user_id"])
    if not user:
        raise HTTPException(
            status_code=403,
            detail="This account no longer exists. Please log in again.",
        )
    try:
        begun = tfa_mgr.totp_setup(user.user_id, label=user.username)
    except Exception as exc:
        # The field encryption's refusal (encryption.EncryptionUnavailable), asked by its class's name.
        if type(exc).__name__ != "EncryptionUnavailable" or type(exc).__module__ != "opti_oignon.encryption":
            raise
        raise HTTPException(status_code=503, detail=f"TOTP not set up: {exc}") from exc
    if begun.get("error") == "already_active":
        _factor_since(req.challenge_id)
    if not begun.get("success"):
        raise HTTPException(status_code=503, detail=_NO_FACTOR_SET_UP)
    _challenge_store.renew(req.challenge_id)
    answer = {"secret": begun["secret"], "uri": begun["uri"]}
    if begun.get("qr_code_base64"):
        answer["qr_code_base64"] = begun["qr_code_base64"]
    return answer


@router.post("/login/2fa/enrol/complete", response_model=EnrolledTokenResponse)
def login_2fa_enrol_complete(req: EnrolCompleteRequest, request: Request, response: Response) -> dict:
    """Confirm the TOTP an enrolment challenge began, with one of its codes: the factor is active, the recovery codes
    are made and shown this once, and the account's first session is given. A wrong code is a try, counted as the
    second step counts them, and counted for the account too, across its logins: past five in five minutes, 429 until
    the window ends. A right code for a secret another login set up again meanwhile activates nothing: the page is
    sent back to log in. Of two right answers sent together, one signs in. A secret that is sealed and does not
    decrypt is 503 with the reason, the try counted."""
    challenge, tfa_mgr = _live_enrolment(req.challenge_id)
    user_id = challenge["user_id"]
    code = "".join(unicodedata.normalize("NFKC", req.code or "").split())
    if not code:
        raise HTTPException(
            status_code=400,
            detail="A 2FA code is required for this method.",
        )
    # The try counted before its code is read, as the second step counts its own; a code that is not there is not
    # one.
    taken = _challenge_store.take_attempt(req.challenge_id)
    if not taken:
        raise HTTPException(
            status_code=403,
            detail=("Invalid or expired enrolment. Please log in again." if taken is None
                    else "Too many 2FA attempts. Please log in again."),
        )
    try:
        verified = tfa_mgr.totp_verify(user_id, code)
    except Exception as exc:
        # The field encryption's refusal (encryption.EncryptionUnavailable), asked by its class's name: the secret is
        # sealed and does not decrypt, so no code was checked. The try stays counted.
        if type(exc).__name__ != "EncryptionUnavailable" or type(exc).__module__ != "opti_oignon.encryption":
            raise
        raise HTTPException(status_code=503, detail=f"TOTP code not checked: {exc}") from exc
    if verified.get("error") == "rate_limited":
        raise HTTPException(
            status_code=429,
            detail="Too many wrong codes for this account. Wait a few minutes, then log in again.",
        )
    if verified.get("error") == "changed":
        _challenge_store.consume(req.challenge_id)
        raise HTTPException(
            status_code=403,
            detail="Another login set this enrolment's secret up again. Please log in again.",
        )
    if not verified.get("success", False):
        remaining = _ChallengeStore.MAX_ATTEMPTS - taken
        detail = "Invalid 2FA code."
        if remaining <= 0:
            detail = "Too many 2FA attempts. Please log in again."
        elif remaining > 0:
            detail = f"Invalid 2FA code. {remaining} attempt(s) remaining."
        raise HTTPException(status_code=403, detail=detail)

    if not _challenge_store.consume(req.challenge_id):
        raise HTTPException(
            status_code=403,
            detail="This login has ended. Please log in again.",
        )

    mgr = _get_auth_manager()
    user = mgr.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=403,
            detail="This account no longer exists. Please log in again.",
        )
    try:
        recovery_codes = tfa_mgr.generate_recovery_codes(user_id)
    except Exception as exc:  # noqa: BLE001 - the factor is active; the codes can be made from the settings
        logger.warning("Recovery codes not made at enrolment (%s): user=%s", type(exc).__name__, user_id)
        recovery_codes = []

    mgr._log_audit(
        user.user_id, "login_2fa_enrolled", "user", user.user_id,
        details={"method": "totp", "recovery_codes": len(recovery_codes)},
    )
    tokens = mgr.create_tokens(user)

    if _is_cookie_mode():
        _set_auth_cookies(
            request,
            response,
            tokens.access_token,
            tokens.refresh_token,
            access_max_age=tokens.expires_in if tokens.expires_in > 0 else 3600,
        )

    return EnrolledTokenResponse(**tokens.to_dict(), recovery_codes=recovery_codes)


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(
    request: Request,
    response: Response,
    req: RefreshRequest | None = None,
) -> dict:
    """Exchange a refresh token for a new token pair.

    Accepts refresh token from httpOnly cookie or request body.
    """
    mgr = _get_auth_manager()

    # Get refresh token from cookie or body
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
        # Clear stale cookies on failure
        if _is_cookie_mode():
            _clear_auth_cookies(response)
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired refresh token. Please log in again.",
        )

    # Set new cookies
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
    """Invalidate a session and clear cookies.

    Accepts refresh token from httpOnly cookie or request body.
    Clears httpOnly cookies on logout.
    """
    mgr = _get_auth_manager()

    # Get refresh token from body or cookie
    rt = None
    if req and req.refresh_token:
        rt = req.refresh_token
    if not rt and _is_cookie_mode():
        rt = request.cookies.get(_REFRESH_COOKIE)

    ok = False
    if rt:
        ok = mgr.logout(rt)

    # Always clear cookies on logout
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
        raise HTTPException(status_code=400, detail="Password change failed: too short, or more than 72 bytes.")

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
    admin: dict = Depends(_require_owner),
) -> list:
    """List all users (the platform's owner, as the store names it now: the one who adds accounts)."""
    mgr = _get_auth_manager()
    users = mgr.list_users(limit=limit, offset=offset)
    return [UserResponse(**u.to_dict()) for u in users]


@router.post("/users", response_model=UserResponse)
def create_account(
    req: CreateUserRequest,
    request: Request,
    current_user: dict = Depends(_get_current_user),
) -> UserResponse:
    """Add an account: the platform's owner's, through the owner's own session, with the owner's password checked as
    the security ceremonies check it (403, 429, 503, audited by name), so that a session alone makes no account.
    Registration's settings do not apply; ``users.max_users`` does. The account gets no session here: it signs in
    itself, and in Bulbe enrols its second factor at its first login -- refused 503, before the account is made,
    where no TOTP can be set up. 400 for a role other than admin, user and viewer, a name taken, or a password the
    rules refuse. An admin's account made here is an owner from then on, and the first registered, while no account
    was an admin's, is no longer one (``AuthManager.owners``)."""
    mgr = _get_auth_manager()
    role = req.role if req.role is not None else mgr.config.get("users", {}).get("default_role", "user")
    if role not in _ROLES:
        raise HTTPException(status_code=400, detail="The role must be admin, user or viewer.")
    owner_id = _owner_password("create_user", current_user, req.owner_password, request)

    tfa_mgr = _get_2fa_manager()
    if tfa_mgr is None:
        raise HTTPException(status_code=503, detail="The second factor cannot be checked here: account not made.")
    if tfa_mgr.required_by_mode() and not tfa_mgr.totp_checkable():
        raise HTTPException(status_code=503, detail="A second factor cannot be set up here: account not made.")

    try:
        user = mgr.create_user(username=req.username, password=req.password, email=req.email, role=role)
    except Exception:  # noqa: BLE001 - a store that cannot be written makes no account
        raise HTTPException(status_code=503, detail="The account store cannot be written: account not made.")
    if not user:
        raise HTTPException(
            status_code=400,
            detail="The account was not made: the name may be taken, the password refused by the rules, or the "
                   "accounts at users.max_users.",
        )
    mgr._log_audit(owner_id, "create_user", "user", user.user_id)
    return UserResponse(**user.to_dict())


@router.delete("/users/{user_id}")
def delete_user(
    user_id: str,
    admin: dict = Depends(_require_owner),
) -> dict:
    """Delete a user account (the platform's owner, as the store names it now: the one who adds accounts). Not the
    owner's own, under its token or the single-user session; audited under the owner's account."""
    mgr = _get_auth_manager()

    actor = _acting_account(admin)
    if user_id in (admin["sub"], actor):
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    ok = mgr.delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")

    # The account's second factors go with it: secrets at rest that no account holds any more.
    tfa_mgr = _get_2fa_manager()
    if tfa_mgr is not None:
        try:
            tfa_mgr.purge_account(user_id)
        except Exception as exc:  # noqa: BLE001 - the account is gone; its factors are inert under its id
            logger.warning("Second factors of deleted user=%s not purged (%s)", user_id, type(exc).__name__)

    mgr._log_audit(actor, "delete_user", "user", user_id)
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
    admin: dict = Depends(_require_owner),
) -> dict:
    """Get audit log entries (the platform's owner, as the store names it now)."""
    mgr = _get_auth_manager()
    entries = mgr.get_audit_log(user_id=user_id, project_id=project_id, limit=limit)
    return {"entries": entries, "count": len(entries)}
