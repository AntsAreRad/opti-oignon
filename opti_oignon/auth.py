#!/usr/bin/env python3
"""
Authentication and user management for Opti-Oignon (S98).

Provides:
- User CRUD with SQLite storage
- Password hashing (bcrypt)
- JWT token generation and validation
- Session management (token refresh, expiry, concurrent session limits)
- Role-based access control helpers
- Single-user mode bypass for backward compatibility

Configuration: config/auth.yaml
Database: data/auth.db (separate SQLite per feature domain pattern)
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from opti_oignon.db_utils import safe_connect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature availability flag
# ---------------------------------------------------------------------------

AUTH_AVAILABLE = True

# S129: SecureBytes for JWT key memory protection
try:
    from opti_oignon.secure_bytes import SecureBytes as _SecureBytes
    _SECURE_BYTES_AVAILABLE = True
except ImportError:
    _SECURE_BYTES_AVAILABLE = False
    _SecureBytes = None

try:
    import bcrypt  # type: ignore
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not installed; password hashing uses fallback PBKDF2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ROLES = ("admin", "user", "viewer")
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "auth.yaml"
DEFAULT_DB_DIR = Path(__file__).parent.parent / "data"

# JWT implementation (minimal, no external dependency required)
# Uses HMAC-SHA512 for signing (S126 PQC upgrade from SHA256)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class User:
    """Represents a user account."""
    user_id: str
    username: str
    email: str
    role: str
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    # password_hash is intentionally excluded from default serialization
    password_hash: str = ""

    def to_dict(self, include_hash: bool = False) -> dict[str, Any]:
        """Serialize to dict, excluding password hash by default."""
        d = {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
        if include_hash:
            d["password_hash"] = self.password_hash
        return d


@dataclass
class Session:
    """Represents an active user session."""
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    refresh_token: str
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuthToken:
    """JWT access + refresh token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 0
    user_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Minimal JWT implementation (S126: HMAC-SHA512 default, backward-compatible)
#
# New tokens are signed with HS512 (256-bit post-quantum security).
# Old HS256 tokens are still accepted during verification for backward
# compatibility.  The algorithm is stored in the JWT header, so
# verification uses the header's alg field automatically.
# ---------------------------------------------------------------------------

import base64

# Map JWT algorithm names to hashlib constructors
_JWT_ALGORITHMS = {
    "HS256": hashlib.sha256,
    "HS512": hashlib.sha512,
}


def _b64url_encode(data: bytes) -> str:
    """Base64url encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    """Base64url decode with padding restoration."""
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def jwt_encode(payload: dict[str, Any], secret: str, algorithm: str = "HS512") -> str:
    """Create a JWT token with HMAC signature.

    Default algorithm is HS512 (HMAC-SHA512) for 256-bit post-quantum
    security.  HS256 is still accepted for backward compatibility.
    """
    if algorithm not in _JWT_ALGORITHMS:
        raise ValueError(f"Unsupported JWT algorithm: {algorithm}")
    hash_fn = _JWT_ALGORITHMS[algorithm]
    header = {"alg": algorithm, "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    message = f"{header_b64}.{payload_b64}"
    signature = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hash_fn,
    ).digest()
    sig_b64 = _b64url_encode(signature)
    return f"{message}.{sig_b64}"


def jwt_decode(token: str, secret: str, algorithm: str = "HS512") -> dict[str, Any] | None:
    """Decode and verify a JWT token. Returns None if invalid or expired.

    S136 audit fix: the algorithm is ALWAYS determined server-side.
    The JWT header's 'alg' field is validated but never trusted to
    downgrade security (prevents algorithm confusion attacks where
    an attacker sends alg:HS256 to reduce from 256-bit to 128-bit).
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts

        # S136 audit fix: validate header alg matches server expectation.
        # Never use attacker-supplied alg for verification.
        try:
            header = json.loads(_b64url_decode(header_b64))
            token_alg = header.get("alg", "")
        except Exception:
            token_alg = ""

        # Reject tokens with mismatched algorithm (prevents downgrade)
        if token_alg and token_alg != algorithm:
            logger.warning(
                "JWT algorithm mismatch: header=%s, expected=%s (rejected)",
                token_alg, algorithm,
            )
            return None

        # Always use server-side algorithm
        if algorithm not in _JWT_ALGORITHMS:
            return None
        hash_fn = _JWT_ALGORITHMS[algorithm]

        # Verify signature
        message = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            hash_fn,
        ).digest()
        actual_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        # Decode payload
        payload = json.loads(_b64url_decode(payload_b64))
        # Check expiry
        exp = payload.get("exp")
        if exp is not None and time.time() > exp:
            return None
        return payload
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(password: str, rounds: int = 12) -> str:
    """Hash a password using bcrypt (preferred) or PBKDF2 fallback."""
    if BCRYPT_AVAILABLE:
        salt = bcrypt.gensalt(rounds=rounds)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
    # Fallback: PBKDF2 with SHA-256
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode(), 100_000)
    return f"pbkdf2:{salt}:{dk.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    if BCRYPT_AVAILABLE and password_hash.startswith("$2"):
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    if password_hash.startswith("pbkdf2:"):
        _, salt, hash_hex = password_hash.split(":", 2)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode(), 100_000)
        return hmac.compare_digest(dk.hex(), hash_hex)
    return False


# ---------------------------------------------------------------------------
# AuthManager
# ---------------------------------------------------------------------------


class AuthManager:
    """Central authentication and user management class.

    Manages user accounts, JWT sessions, and role-based access control.
    Uses a dedicated SQLite database (data/auth.db).
    """

    def __init__(self, config_path: str | Path | None = None, db_path: str | Path | None = None):
        self.config = self._load_config(config_path)
        # RB-02: once a second user is observed, single-user mode latches off
        # for the process lifetime (the safe direction; deleting users back to
        # one must not silently re-enable the authentication bypass).
        self._multi_user_latched = False
        self.db_path = self._resolve_db_path(db_path)
        self._init_db()
        self._ensure_jwt_secret()
        # S136 audit fix: pre-compute a dummy bcrypt hash for timing oracle
        # prevention in authenticate(). This ensures that failed lookups
        # take the same time as successful ones (bcrypt ~200ms).
        self._dummy_hash = hash_password("__timing_oracle_dummy__")
        logger.info("AuthManager initialized (db=%s, single_user=%s)",
                     self.db_path, self.config.get("single_user_mode", True))

    # -- Configuration -------------------------------------------------------

    def _load_config(self, config_path: str | Path | None) -> dict[str, Any]:
        """Load auth configuration from YAML."""
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                return cfg
        except Exception as e:
            logger.warning("Failed to load auth config from %s: %s", path, e)
        # Defaults
        return {
            "jwt": {
                "secret_key": "",
                "access_token_expiry_minutes": 60,
                "refresh_token_expiry_days": 30,
                "algorithm": "HS512",
            },
            "password": {
                "min_length": 8,
                "hash_algorithm": "bcrypt",
                "bcrypt_rounds": 12,
            },
            "users": {
                "allow_registration": True,
                "default_role": "user",
                "max_users": 0,
                "require_email": False,
            },
            "session": {
                "max_sessions": 5,
                "invalidate_on_password_change": True,
            },
            "single_user_mode": True,
            "db_path": "data/auth.db",
        }

    def _resolve_db_path(self, db_path: str | Path | None) -> Path:
        """Resolve database path from argument or config."""
        if db_path:
            return Path(db_path)
        configured = self.config.get("db_path", "data/auth.db")
        # Relative paths are resolved from project root
        p = Path(configured)
        if not p.is_absolute():
            p = Path(__file__).parent.parent / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _ensure_jwt_secret(self):
        """Ensure a JWT secret key exists (generate if empty).

        S129: Wraps the JWT secret in SecureBytes for memory protection.
        S136 audit fix: the plaintext secret is removed from the config
        dict after being wrapped in SecureBytes, so only the mlock'd
        copy remains in memory.
        """
        jwt_cfg = self.config.get("jwt", {})
        if not jwt_cfg.get("secret_key"):
            jwt_cfg["secret_key"] = secrets.token_urlsafe(64)
            self.config.setdefault("jwt", {})["secret_key"] = jwt_cfg["secret_key"]
            logger.info("Generated new JWT secret key")

        # S129: Wrap in SecureBytes for memory protection
        secret_str = jwt_cfg.get("secret_key", "")
        if _SECURE_BYTES_AVAILABLE and secret_str:
            self._jwt_secure_key = _SecureBytes(secret_str.encode("utf-8"))
            # S136 audit fix: remove plaintext from config dict.
            # The secret is now only accessible via SecureBytes.
            jwt_cfg["secret_key"] = "[PROTECTED_BY_SECUREBYTES]"
            self.config.setdefault("jwt", {})["secret_key"] = "[PROTECTED_BY_SECUREBYTES]"
        else:
            self._jwt_secure_key = None

    def _get_jwt_secret(self) -> str:
        """Return the JWT signing secret as a string.

        S129: Reads from SecureBytes if available.
        S136 audit fix: config dict no longer holds the real secret
        (replaced with sentinel), so the fallback only works when
        SecureBytes is not available (i.e. the secret was never wrapped).
        """
        if self._jwt_secure_key is not None and _SECURE_BYTES_AVAILABLE:
            try:
                return self._jwt_secure_key.as_bytes().decode("utf-8")
            except RuntimeError:
                pass  # SecureBytes was wiped, fall through
        # Fallback: only useful when SecureBytes is unavailable
        raw = self.config.get("jwt", {}).get("secret_key", "")
        if raw == "[PROTECTED_BY_SECUREBYTES]":
            # SecureBytes was wiped and config has sentinel — no recovery
            logger.error("JWT secret unavailable: SecureBytes wiped, config has sentinel")
            return ""
        return raw

    # -- Database ------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Get a SQLite connection with WAL mode.

        S136 audit fix: routes through get_encrypted_connection() for
        SQLCipher support when available.
        """
        conn = safe_connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT DEFAULT '',
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    refresh_token TEXT UNIQUE NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS shared_projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'viewer',
                    invited_by TEXT,
                    invite_token TEXT UNIQUE,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    UNIQUE(project_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_type TEXT DEFAULT '',
                    target_id TEXT DEFAULT '',
                    details TEXT DEFAULT '{}',
                    timestamp REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_refresh ON sessions(refresh_token);
                CREATE INDEX IF NOT EXISTS idx_shared_projects_user ON shared_projects(user_id);
                CREATE INDEX IF NOT EXISTS idx_shared_projects_project ON shared_projects(project_id);
                CREATE INDEX IF NOT EXISTS idx_shared_projects_token ON shared_projects(invite_token);
                CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
            """)
            conn.commit()
        finally:
            conn.close()

    # -- User CRUD -----------------------------------------------------------

    def create_user(
        self,
        username: str,
        password: str,
        email: str = "",
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> User | None:
        """Create a new user account.

        Returns the created User or None on failure (duplicate username,
        max users reached, or invalid password).
        """
        # Validate password
        pwd_cfg = self.config.get("password", {})
        min_len = pwd_cfg.get("min_length", 8)
        if len(password) < min_len:
            logger.warning("Password too short (min %d)", min_len)
            return None

        if pwd_cfg.get("require_uppercase") and not any(c.isupper() for c in password):
            logger.warning("Password requires uppercase letter")
            return None

        if pwd_cfg.get("require_digit") and not any(c.isdigit() for c in password):
            logger.warning("Password requires digit")
            return None

        if pwd_cfg.get("require_special") and password.isalnum():
            logger.warning("Password requires special character")
            return None

        # Validate username
        if not username or len(username) < 2 or len(username) > 64:
            logger.warning("Invalid username length")
            return None

        # Check registration allowed
        users_cfg = self.config.get("users", {})
        if not users_cfg.get("allow_registration", True):
            # Only admins can create users when registration is disabled
            pass  # Caller (route) must enforce this

        # Check max users
        max_users = users_cfg.get("max_users", 0)
        if max_users > 0:
            conn = self._get_conn()
            try:
                count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                if count >= max_users:
                    logger.warning("Max users reached (%d)", max_users)
                    return None
            finally:
                conn.close()

        # Check email requirement
        if users_cfg.get("require_email") and not email:
            logger.warning("Email required for registration")
            return None

        # Hash password
        rounds = pwd_cfg.get("bcrypt_rounds", 12)
        pw_hash = hash_password(password, rounds=rounds)

        now = time.time()
        user = User(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            password_hash=pw_hash,
            role=role or users_cfg.get("default_role", "user"),
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO users (user_id, username, email, password_hash, role,
                   created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user.user_id, user.username, user.email, user.password_hash,
                 user.role, user.created_at, user.updated_at,
                 json.dumps(user.metadata)),
            )
            conn.commit()
            logger.info("User created: %s (%s)", user.username, user.user_id)
            return user
        except sqlite3.IntegrityError:
            logger.warning("Duplicate username: %s", username)
            return None
        finally:
            conn.close()

    def get_user(self, user_id: str) -> User | None:
        """Get a user by their ID."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_user(row)
        finally:
            conn.close()

    def get_user_by_username(self, username: str) -> User | None:
        """Get a user by username."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_user(row)
        finally:
            conn.close()

    def update_user(
        self,
        user_id: str,
        username: str | None = None,
        email: str | None = None,
        role: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> User | None:
        """Update user profile fields. Returns updated User or None."""
        user = self.get_user(user_id)
        if not user:
            return None

        updates = []
        params: list[Any] = []

        if username is not None:
            updates.append("username = ?")
            params.append(username)
        if email is not None:
            updates.append("email = ?")
            params.append(email)
        if role is not None and role in VALID_ROLES:
            updates.append("role = ?")
            params.append(role)
        if metadata is not None:
            merged = {**user.metadata, **metadata}
            updates.append("metadata = ?")
            params.append(json.dumps(merged))

        if not updates:
            return user

        now = time.time()
        updates.append("updated_at = ?")
        params.append(now)
        params.append(user_id)

        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE users SET {} WHERE user_id = ?".format(
                    ", ".join(updates)
                ),
                params,
            )
            conn.commit()
            return self.get_user(user_id)
        except sqlite3.IntegrityError:
            logger.warning("Update failed (duplicate username?)")
            return None
        finally:
            conn.close()

    def change_password(self, user_id: str, new_password: str) -> bool:
        """Change a user's password. Optionally invalidates all sessions."""
        pwd_cfg = self.config.get("password", {})
        min_len = pwd_cfg.get("min_length", 8)
        if len(new_password) < min_len:
            return False

        rounds = pwd_cfg.get("bcrypt_rounds", 12)
        pw_hash = hash_password(new_password, rounds=rounds)
        now = time.time()

        conn = self._get_conn()
        try:
            result = conn.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE user_id = ?",
                (pw_hash, now, user_id),
            )
            if result.rowcount == 0:
                return False

            # Invalidate sessions if configured
            session_cfg = self.config.get("session", {})
            if session_cfg.get("invalidate_on_password_change", True):
                conn.execute(
                    "UPDATE sessions SET is_active = 0 WHERE user_id = ?",
                    (user_id,),
                )
            conn.commit()
            return True
        finally:
            conn.close()

    def delete_user(self, user_id: str) -> bool:
        """Delete a user and all their sessions (CASCADE)."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                "DELETE FROM users WHERE user_id = ?", (user_id,)
            )
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()

    def list_users(self, limit: int = 100, offset: int = 0) -> list[User]:
        """List all users (admin endpoint)."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._row_to_user(r) for r in rows]
        finally:
            conn.close()

    def count_users(self) -> int:
        """Count total users."""
        conn = self._get_conn()
        try:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        finally:
            conn.close()

    # -- Authentication ------------------------------------------------------

    def authenticate(self, username: str, password: str) -> User | None:
        """Verify username + password. Returns User on success, None on failure.

        S136 audit fix: performs a dummy password verification even when
        the user does not exist, to prevent timing oracle attacks that
        could enumerate valid usernames (bcrypt takes ~200ms vs instant
        return for non-existent users).
        """
        user = self.get_user_by_username(username)
        if not user:
            # Dummy verify to equalize timing (prevent username enumeration)
            verify_password(password, self._dummy_hash)
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    def create_tokens(self, user: User) -> AuthToken:
        """Create JWT access + refresh tokens for a user."""
        jwt_cfg = self.config.get("jwt", {})
        secret = self._get_jwt_secret()  # S129: SecureBytes-backed
        algorithm = jwt_cfg.get("algorithm", "HS512")
        access_expiry_min = jwt_cfg.get("access_token_expiry_minutes", 60)
        refresh_expiry_days = jwt_cfg.get("refresh_token_expiry_days", 30)

        now = time.time()
        access_exp = now + (access_expiry_min * 60)
        refresh_exp = now + (refresh_expiry_days * 86400)

        # Access token payload
        access_payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role,
            "iat": int(now),
            "exp": int(access_exp),
            "type": "access",
        }
        access_token = jwt_encode(access_payload, secret, algorithm)

        # Refresh token (opaque random string)
        refresh_token = secrets.token_urlsafe(48)

        # Store session
        self._create_session(user.user_id, refresh_token, refresh_exp)

        return AuthToken(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=access_expiry_min * 60,
            user_id=user.user_id,
        )

    def validate_token(self, token: str) -> dict[str, Any] | None:
        """Validate a JWT access token. Returns payload or None."""
        jwt_cfg = self.config.get("jwt", {})
        secret = self._get_jwt_secret()  # S129: SecureBytes-backed
        algorithm = jwt_cfg.get("algorithm", "HS512")
        payload = jwt_decode(token, secret, algorithm)
        if not payload:
            return None
        if payload.get("type") != "access":
            return None
        return payload

    def refresh_tokens(self, refresh_token: str) -> AuthToken | None:
        """Exchange a refresh token for new access + refresh tokens."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT s.*, u.username, u.role FROM sessions s
                   JOIN users u ON s.user_id = u.user_id
                   WHERE s.refresh_token = ? AND s.is_active = 1""",
                (refresh_token,),
            ).fetchone()
            if not row:
                return None

            # Check expiry
            if time.time() > row["expires_at"]:
                conn.execute(
                    "UPDATE sessions SET is_active = 0 WHERE session_id = ?",
                    (row["session_id"],),
                )
                conn.commit()
                return None

            # Invalidate old session
            conn.execute(
                "UPDATE sessions SET is_active = 0 WHERE session_id = ?",
                (row["session_id"],),
            )
            conn.commit()
        finally:
            conn.close()

        # Create new tokens
        user = self.get_user(row["user_id"])
        if not user:
            return None
        return self.create_tokens(user)

    def logout(self, refresh_token: str) -> bool:
        """Invalidate a session by its refresh token."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                "UPDATE sessions SET is_active = 0 WHERE refresh_token = ?",
                (refresh_token,),
            )
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()

    def logout_all(self, user_id: str) -> int:
        """Invalidate all sessions for a user."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                "UPDATE sessions SET is_active = 0 WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
            return result.rowcount
        finally:
            conn.close()

    # -- Shared Projects & RBAC ----------------------------------------------

    def share_project(
        self,
        project_id: str,
        user_id: str,
        role: str = "viewer",
        invited_by: str = "",
    ) -> dict[str, Any] | None:
        """Share a project with a user at the given role level."""
        if role not in ("owner", "editor", "viewer"):
            return None
        now = time.time()
        invite_token = secrets.token_urlsafe(32)

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO shared_projects
                   (project_id, user_id, role, invited_by, invite_token, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (project_id, user_id, role, invited_by, invite_token, now),
            )
            conn.commit()
            self._log_audit(
                user_id=invited_by or user_id,
                action="share_project",
                target_type="project",
                target_id=project_id,
                details={"shared_with": user_id, "role": role},
            )
            return {
                "project_id": project_id,
                "user_id": user_id,
                "role": role,
                "invite_token": invite_token,
            }
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

    def accept_invite(self, invite_token: str, user_id: str) -> dict[str, Any] | None:
        """Accept a project invitation by token."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM shared_projects WHERE invite_token = ?",
                (invite_token,),
            ).fetchone()
            if not row:
                return None
            # Update the user_id to the accepting user (in case link-based)
            conn.execute(
                "UPDATE shared_projects SET user_id = ? WHERE invite_token = ?",
                (user_id, invite_token),
            )
            conn.commit()
            return {
                "project_id": row["project_id"],
                "user_id": user_id,
                "role": row["role"],
            }
        finally:
            conn.close()

    def get_project_role(self, project_id: str, user_id: str) -> str | None:
        """Get a user's role for a project. Returns None if no access."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT role FROM shared_projects WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            ).fetchone()
            return row["role"] if row else None
        finally:
            conn.close()

    def check_permission(
        self,
        project_id: str,
        user_id: str,
        required_role: str = "viewer",
    ) -> bool:
        """Check if a user has at least the required role for a project.

        Role hierarchy: owner > editor > viewer
        """
        role = self.get_project_role(project_id, user_id)
        if role is None:
            return False
        hierarchy = {"owner": 3, "editor": 2, "viewer": 1}
        return hierarchy.get(role, 0) >= hierarchy.get(required_role, 0)

    def list_project_members(self, project_id: str) -> list[dict[str, Any]]:
        """List all users with access to a project."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT sp.*, u.username, u.email FROM shared_projects sp
                   JOIN users u ON sp.user_id = u.user_id
                   WHERE sp.project_id = ? ORDER BY sp.created_at""",
                (project_id,),
            ).fetchall()
            return [
                {
                    "project_id": r["project_id"],
                    "user_id": r["user_id"],
                    "username": r["username"],
                    "email": r["email"],
                    "role": r["role"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def list_user_shared_projects(self, user_id: str) -> list[dict[str, Any]]:
        """List all projects shared with a user."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM shared_projects WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
            return [
                {
                    "project_id": r["project_id"],
                    "role": r["role"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def remove_project_access(self, project_id: str, user_id: str) -> bool:
        """Remove a user's access to a project."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                "DELETE FROM shared_projects WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            )
            conn.commit()
            if result.rowcount > 0:
                self._log_audit(
                    user_id=user_id,
                    action="remove_project_access",
                    target_type="project",
                    target_id=project_id,
                )
            return result.rowcount > 0
        finally:
            conn.close()

    # -- Audit Log -----------------------------------------------------------

    def _log_audit(
        self,
        user_id: str,
        action: str,
        target_type: str = "",
        target_id: str = "",
        details: dict[str, Any] | None = None,
    ):
        """Record an action in the audit log.

        S130: Also forwards to the hash-chain signed audit log.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO audit_log (user_id, action, target_type, target_id,
                   details, timestamp) VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, action, target_type, target_id,
                 json.dumps(details or {}), time.time()),
            )
            conn.commit()
        except Exception as e:
            logger.debug("Audit log failed: %s", e)
        finally:
            conn.close()

        # S130: Forward to hash-chain signed audit log
        try:
            from opti_oignon.signed_audit_log import chain_log
            chain_log(
                event_type=f"auth_{action}",
                source="auth",
                action=action,
                severity="WARNING" if "fail" in action.lower() else "INFO",
                user_id=user_id,
                target_type=target_type,
                target_id=target_id,
                **(details or {}),
            )
        except Exception:
            pass

    def get_audit_log(
        self,
        user_id: str | None = None,
        project_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve audit log entries."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM audit_log"
            params: list[Any] = []
            conditions = []

            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            if project_id:
                conditions.append("target_id = ?")
                params.append(project_id)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [
                {
                    "id": r["id"],
                    "user_id": r["user_id"],
                    "action": r["action"],
                    "target_type": r["target_type"],
                    "target_id": r["target_id"],
                    "details": json.loads(r["details"]) if r["details"] else {},
                    "timestamp": r["timestamp"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    # -- Single-user mode ----------------------------------------------------

    @property
    def single_user_mode(self) -> bool:
        """Whether single-user mode (no auth required) is effectively active.

        Single-user mode is a convenience bypass for a loopback, single-account
        install: the auth middleware and the RBAC dependency skip authentication
        when it is on and Bulbe is not active. RB-02: it is derived, not static,
        so an operator who provisions multiple users but leaves the default flag
        on does not silently keep an authentication bypass. Rules, in order:

        - an explicit ``single_user_mode: false`` in config always wins;
        - it auto-disables once a second user exists (``count_users() > 1``),
          latching off for the process lifetime;
        - it fails safe: an undeterminable user count is treated as multi-user
          (authentication required).
        """
        # Explicit opt-out always wins.
        if not self.config.get("single_user_mode", True):
            return False
        # Multi-user already observed: stay off (one-way latch).
        if self._multi_user_latched:
            return False
        # Auto-disable once a second user exists; fail safe on any error.
        try:
            if self.count_users() > 1:
                self._multi_user_latched = True
                logger.warning(
                    "single_user_mode auto-disabled: more than one user "
                    "exists; authentication is now required (RB-02)"
                )
                return False
        except Exception:
            return False
        return True

    # -- Internal helpers ----------------------------------------------------

    def _row_to_user(self, row: sqlite3.Row) -> User:
        """Convert a database row to a User dataclass."""
        meta = {}
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            pass
        return User(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            role=row["role"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=meta,
        )

    def _create_session(self, user_id: str, refresh_token: str, expires_at: float):
        """Create a new session record in the database."""
        # Enforce max sessions
        session_cfg = self.config.get("session", {})
        max_sessions = session_cfg.get("max_sessions", 5)

        conn = self._get_conn()
        try:
            if max_sessions > 0:
                # Count active sessions
                count = conn.execute(
                    "SELECT COUNT(*) FROM sessions WHERE user_id = ? AND is_active = 1",
                    (user_id,),
                ).fetchone()[0]
                if count >= max_sessions:
                    # Deactivate oldest session
                    conn.execute(
                        """UPDATE sessions SET is_active = 0
                           WHERE session_id = (
                               SELECT session_id FROM sessions
                               WHERE user_id = ? AND is_active = 1
                               ORDER BY created_at ASC LIMIT 1
                           )""",
                        (user_id,),
                    )

            session_id = str(uuid.uuid4())
            now = time.time()
            conn.execute(
                """INSERT INTO sessions (session_id, user_id, refresh_token,
                   created_at, expires_at, is_active, metadata)
                   VALUES (?, ?, ?, ?, ?, 1, '{}')""",
                (session_id, user_id, refresh_token, now, expires_at),
            )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# S124: Login rate limiting
# ---------------------------------------------------------------------------

@dataclass
class _RateLimitEntry:
    """Track failed login attempts for an IP or username."""

    attempts: list[float] = field(default_factory=list)
    lockout_until: float = 0.0
    lockout_count: int = 0


class LoginRateLimiter:
    """In-memory rate limiter for login endpoints (S124).

    Tracks failed attempts per IP address and per username with:
    - Sliding window: max N attempts per time window
    - Progressive lockout: base_seconds * 2^(lockout_count-1), capped
    - Per-username account lock after threshold failures

    Configuration loaded from security.yaml > rate_limiting section.

    Thread-safe: a single lock guards the per-IP and per-username maps and
    their attempt lists across the check, record, and status calls (AU-01).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or self._load_config()
        self.enabled: bool = cfg.get("enabled", True)
        self.max_attempts: int = cfg.get("login_max_attempts", 5)
        self.window_seconds: int = cfg.get("login_window_seconds", 300)
        self.lockout_base: int = cfg.get("lockout_base_seconds", 60)
        self.lockout_max: int = cfg.get("lockout_max_seconds", 3600)
        self.account_lock_threshold: int = cfg.get("account_lock_threshold", 10)
        self.account_lock_duration: int = cfg.get("account_lock_duration_seconds", 900)

        # Per-IP tracking
        self._ip_entries: dict[str, _RateLimitEntry] = {}
        # Per-username tracking
        self._user_entries: dict[str, _RateLimitEntry] = {}
        # AU-01: a single lock guards both maps and their attempt lists. The
        # critical sections are short, so one lock is sufficient.
        self._lock = threading.Lock()

    @staticmethod
    def _load_config() -> dict[str, Any]:
        """Load rate limiting config from security.yaml."""
        try:
            sec_path = Path(__file__).parent / "config" / "security.yaml"
            if sec_path.is_file():
                with open(sec_path, "r", encoding="utf-8") as fh:
                    raw = yaml.safe_load(fh) or {}
                return raw.get("rate_limiting", {})
        except Exception:
            pass
        return {}

    def _clean_window(self, entry: _RateLimitEntry, now: float) -> None:
        """Remove attempts outside the sliding window."""
        cutoff = now - self.window_seconds
        entry.attempts = [t for t in entry.attempts if t > cutoff]

    def _get_lockout_duration(self, lockout_count: int) -> int:
        """Calculate lockout duration with exponential backoff."""
        if lockout_count <= 0:
            return self.lockout_base
        duration = self.lockout_base * (2 ** (lockout_count - 1))
        return min(duration, self.lockout_max)

    def check_rate_limit(
        self, ip: str, username: str
    ) -> tuple[bool, int]:
        """Check if a login attempt is allowed.

        Parameters
        ----------
        ip : str
            Client IP address.
        username : str
            Username being attempted.

        Returns
        -------
        tuple[bool, int]
            (allowed, retry_after_seconds).
            If allowed is False, retry_after_seconds indicates when to retry.
        """
        if not self.enabled:
            return True, 0

        now = time.time()

        with self._lock:
            # Check per-IP
            ip_entry = self._ip_entries.get(ip)
            if ip_entry:
                # Check lockout
                if ip_entry.lockout_until > now:
                    retry = int(ip_entry.lockout_until - now) + 1
                    return False, retry

                self._clean_window(ip_entry, now)

                if len(ip_entry.attempts) >= self.max_attempts:
                    # Trigger lockout
                    ip_entry.lockout_count += 1
                    duration = self._get_lockout_duration(ip_entry.lockout_count)
                    ip_entry.lockout_until = now + duration
                    logger.warning(
                        "Rate limit lockout for IP %s: %ds (count=%d)",
                        ip, duration, ip_entry.lockout_count,
                    )
                    return False, duration

            # Check per-username account lock
            user_entry = self._user_entries.get(username)
            if user_entry:
                if user_entry.lockout_until > now:
                    retry = int(user_entry.lockout_until - now) + 1
                    return False, retry

                self._clean_window(user_entry, now)

                if len(user_entry.attempts) >= self.account_lock_threshold:
                    user_entry.lockout_until = now + self.account_lock_duration
                    logger.warning(
                        "Account locked for user '%s': %ds",
                        username, self.account_lock_duration,
                    )
                    return False, self.account_lock_duration

            return True, 0

    def record_failure(self, ip: str, username: str) -> None:
        """Record a failed login attempt for both IP and username."""
        if not self.enabled:
            return

        now = time.time()

        with self._lock:
            # Record for IP
            if ip not in self._ip_entries:
                self._ip_entries[ip] = _RateLimitEntry()
            self._ip_entries[ip].attempts.append(now)

            # Record for username
            if username not in self._user_entries:
                self._user_entries[username] = _RateLimitEntry()
            self._user_entries[username].attempts.append(now)

    def record_success(self, ip: str, username: str) -> None:
        """Record a successful login — resets counters for that username."""
        if not self.enabled:
            return

        with self._lock:
            # Reset username counters (successful login proves ownership)
            if username in self._user_entries:
                del self._user_entries[username]

            # For IP, just clean old entries but do NOT reset lockout count
            # (prevents attackers from resetting by using a valid credential)
            if ip in self._ip_entries:
                self._clean_window(self._ip_entries[ip], time.time())

    def get_status(self, ip: str | None = None, username: str | None = None) -> dict[str, Any]:
        """Get rate limit status for debugging/monitoring."""
        now = time.time()
        status: dict[str, Any] = {"enabled": self.enabled}

        with self._lock:
            if ip and ip in self._ip_entries:
                entry = self._ip_entries[ip]
                self._clean_window(entry, now)
                status["ip"] = {
                    "attempts_in_window": len(entry.attempts),
                    "locked_until": entry.lockout_until if entry.lockout_until > now else None,
                    "lockout_count": entry.lockout_count,
                }

            if username and username in self._user_entries:
                entry = self._user_entries[username]
                self._clean_window(entry, now)
                status["username"] = {
                    "attempts_in_window": len(entry.attempts),
                    "locked_until": entry.lockout_until if entry.lockout_until > now else None,
                }

        return status


# Module-level rate limiter singleton
login_rate_limiter = LoginRateLimiter()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

try:
    auth_manager = AuthManager()
except Exception as e:
    logger.error("Failed to initialize AuthManager: %s", e)
    auth_manager = None  # type: ignore
    AUTH_AVAILABLE = False
