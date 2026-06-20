#!/usr/bin/env python3
"""
Two-Factor Authentication for Opti-Oignon (S126).

Supports three 2FA methods (in priority order):

  1. **WebAuthn/FIDO2** — hardware keys (YubiKey 5, Google Titan),
     platform authenticators (Apple Passkeys, Windows Hello)
  2. **TOTP** — time-based one-time passwords (Google Authenticator,
     Authy) as backup when hardware key unavailable
  3. **Recovery codes** — 10 one-time codes shown once at setup,
     AES-256-GCM encrypted at rest

Mode behaviour:
  - **Daily**: 2FA optional (user can enable if desired)
  - **Bulbe**: 2FA required for login, mode degradation, plugin
    approval, and search re-enable

App-specific passwords allow CLI tools to skip interactive 2FA
while maintaining auditability.  They are revocable and logged.

Dependencies:
  - python-fido2 (WebAuthn/FIDO2)
  - pyotp (TOTP)
  - qrcode (TOTP setup QR)
  All optional: features degrade gracefully if missing.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as _hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from opti_oignon.db_utils import safe_connect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature availability
# ---------------------------------------------------------------------------

WEBAUTHN_AVAILABLE = False
try:
    from fido2.server import Fido2Server  # type: ignore[import]
    from fido2.webauthn import (  # type: ignore[import]
        PublicKeyCredentialRpEntity,
        PublicKeyCredentialUserEntity,
        AttestedCredentialData,
        AuthenticatorData,
    )
    WEBAUTHN_AVAILABLE = True
except ImportError:
    logger.info("python-fido2 not installed. WebAuthn disabled.")

TOTP_AVAILABLE = False
try:
    import pyotp  # type: ignore[import]
    TOTP_AVAILABLE = True
except ImportError:
    logger.info("pyotp not installed. TOTP disabled.")

QRCODE_AVAILABLE = False
try:
    import qrcode  # type: ignore[import]
    import io as _io
    QRCODE_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_2FA_DB_PATH = _DATA_DIR / "auth_2fa.db"

RECOVERY_CODE_COUNT = 10
RECOVERY_CODE_LENGTH = 16  # S136: 16 hex chars = 64 bits (was 8 = 32 bits)
APP_PASSWORD_LENGTH = 32

# Rate limiting
MAX_RECOVERY_ATTEMPTS_PER_HOUR = 3
MAX_TOTP_ATTEMPTS_PER_WINDOW = 5
TOTP_WINDOW_SECONDS = 300

# WebAuthn defaults
RP_ID = "localhost"
RP_NAME = "Opti-Oignon"
ORIGIN = "http://localhost:5173"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WebAuthnCredential:
    """Stored WebAuthn credential."""
    credential_id: str  # base64url
    public_key: str     # base64url serialized
    user_id: str
    name: str           # user-friendly name ("YubiKey 5C")
    created_at: float
    last_used: float = 0.0
    sign_count: int = 0


@dataclass
class TOTPConfig:
    """TOTP configuration for a user."""
    user_id: str
    secret_encrypted: str  # AES-256-GCM encrypted
    issuer: str = RP_NAME
    algorithm: str = "SHA1"  # RFC 6238 default
    digits: int = 6
    period: int = 30
    verified: bool = False
    created_at: float = 0.0


@dataclass
class AppPassword:
    """App-specific password for CLI use (skip interactive 2FA)."""
    password_id: str
    user_id: str
    name: str           # user-friendly label
    password_hash: str
    created_at: float
    last_used: float = 0.0
    revoked: bool = False


@dataclass
class TwoFAStatus:
    """2FA status for a user."""
    user_id: str
    webauthn_enabled: bool = False
    webauthn_credential_count: int = 0
    totp_enabled: bool = False
    totp_verified: bool = False
    recovery_codes_remaining: int = 0
    app_passwords_count: int = 0
    any_method_active: bool = False
    recovery_reissue_required: bool = False


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _get_2fa_conn() -> sqlite3.Connection:
    """Get a connection to the 2FA database.

    S136 audit fix: routes through get_encrypted_connection() for
    SQLCipher support when available.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = safe_connect(str(_2FA_DB_PATH), timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_2fa_db() -> None:
    """Initialize the 2FA database schema."""
    conn = _get_2fa_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS webauthn_credentials (
                credential_id TEXT PRIMARY KEY,
                public_key TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT 'Security Key',
                created_at REAL NOT NULL,
                last_used REAL DEFAULT 0,
                sign_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS totp_config (
                user_id TEXT PRIMARY KEY,
                secret_encrypted TEXT NOT NULL,
                issuer TEXT DEFAULT 'Opti-Oignon',
                algorithm TEXT DEFAULT 'SHA1',
                digits INTEGER DEFAULT 6,
                period INTEGER DEFAULT 30,
                verified INTEGER DEFAULT 0,
                last_step INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recovery_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                used INTEGER DEFAULT 0,
                used_at REAL DEFAULT 0,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS app_passwords (
                password_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_used REAL DEFAULT 0,
                revoked INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_webauthn_user
                ON webauthn_credentials(user_id);
            CREATE INDEX IF NOT EXISTS idx_recovery_user
                ON recovery_codes(user_id);
            CREATE INDEX IF NOT EXISTS idx_app_pw_user
                ON app_passwords(user_id);
        """)
        # AU-06: add the TOTP replay-protection column to databases that predate
        # it. SQLite has no ADD COLUMN IF NOT EXISTS, so guard with table_info.
        totp_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(totp_config)").fetchall()
        }
        if "last_step" not in totp_cols:
            conn.execute(
                "ALTER TABLE totp_config ADD COLUMN last_step INTEGER DEFAULT 0"
            )
        conn.commit()
    finally:
        conn.close()


# Initialize on import
_init_2fa_db()


# ---------------------------------------------------------------------------
# Encryption helpers (for TOTP secret storage)
# ---------------------------------------------------------------------------

def _encrypt_secret(plaintext: str) -> str:
    """Encrypt a TOTP secret using the project encryption module."""
    try:
        from opti_oignon.encryption import EncryptionManager
        mgr = EncryptionManager()
        if mgr.enabled and mgr.has_key:
            return mgr.encrypt(plaintext)
    except Exception:
        pass
    # Fallback: base64 (not secure, but allows operation without encryption)
    return "B64:" + base64.b64encode(plaintext.encode()).decode()


def _decrypt_secret(ciphertext: str) -> str:
    """Decrypt a stored TOTP secret."""
    if ciphertext.startswith("B64:"):
        return base64.b64decode(ciphertext[4:]).decode()
    try:
        from opti_oignon.encryption import EncryptionManager
        mgr = EncryptionManager()
        if mgr.enabled and mgr.has_key:
            return mgr.decrypt(ciphertext)
    except Exception:
        pass
    return ciphertext


# AU-03: recovery codes and app passwords are HMAC'd under a subkey derived from
# the master encryption key on its own HKDF domain (Kerckhoffs-clean: the master
# key is the only secret), versioned with a "v2:" prefix. The pre-AU-03 scheme
# (the public-constant or keyfile HMAC, or plain SHA-256) is retained only to
# verify and migrate existing hashes. The domain string is distinct from the
# SQLCipher subkey, the learned-router MAC, and the audit anchor.
_CODE_HMAC_INFO = b"oo-2fa-code-hmac-v2"
_V2_PREFIX = "v2:"


def _derive_2fa_code_key() -> Optional[bytes]:
    """Derive the 2FA code-hashing subkey off the master key (HMAC-SHA256).

    Domain-separated from the SQLCipher subkey, the learned-router MAC and the
    audit anchor (distinct info string). Returns None when no master key is
    configured, in which case hashing falls back to the legacy scheme and the
    at-rest protection rests on SQLCipher.
    """
    try:
        from opti_oignon.encryption import get_encryption_key
        sb = get_encryption_key()
    except Exception:
        return None
    if not sb:
        return None
    try:
        raw = sb.as_bytes() if hasattr(sb, "as_bytes") else sb
        return _hmac.new(raw, _CODE_HMAC_INFO, hashlib.sha256).digest()
    except Exception:
        return None


def _hash_code_v2(code: str, subkey: bytes) -> str:
    """HMAC a code under the master-key-derived subkey (new scheme)."""
    return _V2_PREFIX + _hmac.new(
        subkey, code.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _hash_code_legacy(code: str) -> str:
    """Reproduce the pre-AU-03 hash of a code (for verifying / migrating).

    Keyed HMAC with the legacy server key when available, else plain SHA-256.
    """
    server_key = _get_hash_server_key()
    if server_key:
        return _hmac.new(
            server_key.encode("utf-8"), code.encode("utf-8"), hashlib.sha256
        ).hexdigest()
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _hash_code(code: str) -> str:
    """Hash a recovery code or app password for storage.

    AU-03: prefers a v2 hash keyed off the master encryption key on its own HKDF
    domain (the master key is the only secret, so it is Kerckhoffs-clean). Falls
    back to the legacy scheme when no master key is configured, in which case the
    at-rest protection rests on SQLCipher.
    """
    subkey = _derive_2fa_code_key()
    if subkey is not None:
        return _hash_code_v2(code, subkey)
    return _hash_code_legacy(code)


def _verify_code(code: str, stored: str) -> tuple[bool, bool]:
    """Check a code against a stored hash of either scheme (constant-time).

    Returns ``(matched, is_legacy)``: ``is_legacy`` is True when the match was
    against a pre-AU-03 hash, so the caller can re-issue (recovery codes) or
    rehash-on-use (app passwords).
    """
    if stored.startswith(_V2_PREFIX):
        subkey = _derive_2fa_code_key()
        if subkey is None:
            return False, False  # v2 hash but no key: cannot verify (fail safe)
        return _hmac.compare_digest(_hash_code_v2(code, subkey), stored), False
    # Legacy: try the keyed HMAC and the plain SHA-256 fallback.
    server_key = _get_hash_server_key()
    if server_key and _hmac.compare_digest(
        _hmac.new(
            server_key.encode("utf-8"), code.encode("utf-8"), hashlib.sha256
        ).hexdigest(),
        stored,
    ):
        return True, True
    if _hmac.compare_digest(
        hashlib.sha256(code.encode("utf-8")).hexdigest(), stored
    ):
        return True, True
    return False, False


def _get_hash_server_key() -> str:
    """Get a server-side key for HMAC hashing of recovery codes.

    Tries to load from the encryption keyfile, then from the JWT secret.
    Returns empty string if neither is available.
    """
    try:
        from opti_oignon.encryption import get_encryption_status
        status = get_encryption_status()
        if status.get("key_available"):
            return "oo-recovery-hmac-key-" + status.get("kdf", "default")
    except Exception:
        pass
    try:
        key_path = _DATA_DIR / ".keyfile"
        if key_path.exists():
            return key_path.read_text(encoding="utf-8").strip()[:32]
    except Exception:
        pass
    return ""


def _totp_matched_step(totp: Any, code: str, now: float) -> Optional[int]:
    """Return the time-step a TOTP code matches within the +/-1 window, or None.

    Used for replay protection (AU-06): the caller records the last consumed
    step and rejects a code whose step is at or below it. The accept decision
    itself is still made by ``totp.verify(code, valid_window=1)``; this only
    identifies which of the three accepted steps matched. The step is a property
    of the (secret, code) pair, so it is stable regardless of small clock drift
    as long as the code is still inside the verify window.
    """
    period = int(getattr(totp, "interval", 30) or 30)
    current = int(now // period)
    for offset in (-1, 0, 1):
        step = current + offset
        try:
            candidate = str(totp.at(step * period))
        except Exception:
            continue
        if _hmac.compare_digest(candidate, str(code)):
            return step
    return None


# ---------------------------------------------------------------------------
# TwoFactorAuthManager
# ---------------------------------------------------------------------------

class TwoFactorAuthManager:
    """Manages all 2FA methods for Opti-Oignon.

    Supports WebAuthn/FIDO2, TOTP, recovery codes, and app-specific
    passwords.  Each method can be enabled independently.
    """

    def __init__(self) -> None:
        self._webauthn_challenges: dict[str, Any] = {}
        self._totp_attempts: dict[str, list[float]] = {}  # user_id -> timestamps
        self._recovery_attempts: dict[str, list[float]] = {}

    # -- Status ---------------------------------------------------------------

    def get_status(self, user_id: str) -> TwoFAStatus:
        """Get the 2FA status for a user."""
        conn = _get_2fa_conn()
        try:
            # WebAuthn
            row = conn.execute(
                "SELECT COUNT(*) as c FROM webauthn_credentials WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            webauthn_count = row["c"] if row else 0

            # TOTP
            totp_row = conn.execute(
                "SELECT verified FROM totp_config WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            totp_enabled = totp_row is not None
            totp_verified = bool(totp_row["verified"]) if totp_row else False

            # Recovery codes (and AU-03 re-issue flag, computed from the stored
            # hash scheme: any unused code lacking the v2 prefix is a pre-AU-03
            # hash that should be re-issued).
            recovery_rows = conn.execute(
                "SELECT code_hash FROM recovery_codes "
                "WHERE user_id = ? AND used = 0",
                (user_id,),
            ).fetchall()
            recovery_count = len(recovery_rows)
            recovery_reissue = any(
                not r["code_hash"].startswith(_V2_PREFIX) for r in recovery_rows
            )

            # App passwords
            app_row = conn.execute(
                "SELECT COUNT(*) as c FROM app_passwords "
                "WHERE user_id = ? AND revoked = 0",
                (user_id,),
            ).fetchone()
            app_count = app_row["c"] if app_row else 0

            any_active = (webauthn_count > 0) or (totp_enabled and totp_verified)

            return TwoFAStatus(
                user_id=user_id,
                webauthn_enabled=webauthn_count > 0,
                webauthn_credential_count=webauthn_count,
                totp_enabled=totp_enabled,
                totp_verified=totp_verified,
                recovery_codes_remaining=recovery_count,
                app_passwords_count=app_count,
                any_method_active=any_active,
                recovery_reissue_required=recovery_reissue,
            )
        finally:
            conn.close()

    def is_2fa_required(self, user_id: str) -> bool:
        """Check if 2FA is required for this user based on mode."""
        try:
            from opti_oignon.security_mode import is_bulbe
            if is_bulbe():
                return True
        except ImportError:
            pass
        # In Daily mode, 2FA is required only if user has enabled it
        status = self.get_status(user_id)
        return status.any_method_active

    # -- WebAuthn/FIDO2 -------------------------------------------------------

    def webauthn_register_begin(
        self, user_id: str, username: str
    ) -> dict[str, Any]:
        """Begin WebAuthn credential registration.

        Returns the options to pass to navigator.credentials.create().
        """
        if not WEBAUTHN_AVAILABLE:
            return {"success": False, "error": "webauthn_unavailable",
                    "message": "python-fido2 not installed"}

        rp = PublicKeyCredentialRpEntity(RP_ID, RP_NAME)
        user_entity = PublicKeyCredentialUserEntity(
            id=user_id.encode("utf-8"),
            name=username,
            display_name=username,
        )

        # Get existing credentials to exclude
        conn = _get_2fa_conn()
        try:
            rows = conn.execute(
                "SELECT credential_id, public_key FROM webauthn_credentials "
                "WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()

        existing_creds = []
        for row in rows:
            try:
                cred_data = base64.urlsafe_b64decode(row["public_key"] + "==")
                existing_creds.append(AttestedCredentialData(cred_data))
            except Exception:
                pass

        server = Fido2Server(rp)
        registration_data, state = server.register_begin(
            user_entity,
            credentials=existing_creds,
        )

        # Store challenge in memory (per-user, short-lived)
        self._webauthn_challenges[user_id] = {
            "state": state,
            "type": "register",
            "timestamp": time.time(),
        }

        # Serialize for JSON transport
        options = dict(registration_data)
        return {
            "success": True,
            "options": _serialize_webauthn_options(options),
        }

    def webauthn_register_complete(
        self, user_id: str, credential_name: str, response_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Complete WebAuthn credential registration."""
        if not WEBAUTHN_AVAILABLE:
            return {"success": False, "error": "webauthn_unavailable"}

        challenge = self._webauthn_challenges.pop(user_id, None)
        if not challenge or challenge.get("type") != "register":
            return {"success": False, "error": "no_challenge",
                    "message": "No pending registration challenge"}

        # Timeout: 5 minutes
        if time.time() - challenge["timestamp"] > 300:
            return {"success": False, "error": "expired",
                    "message": "Registration challenge expired"}

        try:
            rp = PublicKeyCredentialRpEntity(RP_ID, RP_NAME)
            server = Fido2Server(rp)
            auth_data = server.register_complete(
                challenge["state"],
                response_data,
            )

            # Store credential
            cred_id_b64 = base64.urlsafe_b64encode(
                auth_data.credential_data.credential_id
            ).decode().rstrip("=")
            pub_key_b64 = base64.urlsafe_b64encode(
                bytes(auth_data.credential_data)
            ).decode().rstrip("=")

            conn = _get_2fa_conn()
            try:
                conn.execute(
                    """INSERT INTO webauthn_credentials
                       (credential_id, public_key, user_id, name, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (cred_id_b64, pub_key_b64, user_id,
                     credential_name or "Security Key", time.time()),
                )
                conn.commit()
            finally:
                conn.close()

            return {
                "success": True,
                "credential_id": cred_id_b64,
                "name": credential_name,
            }
        except Exception as exc:
            logger.warning("WebAuthn registration failed: %s", exc)
            return {"success": False, "error": "registration_failed",
                    "message": str(exc)}

    def webauthn_auth_begin(self, user_id: str) -> dict[str, Any]:
        """Begin WebAuthn authentication challenge."""
        if not WEBAUTHN_AVAILABLE:
            return {"success": False, "error": "webauthn_unavailable"}

        conn = _get_2fa_conn()
        try:
            rows = conn.execute(
                "SELECT credential_id, public_key FROM webauthn_credentials "
                "WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {"success": False, "error": "no_credentials",
                    "message": "No WebAuthn credentials registered"}

        credentials = []
        for row in rows:
            try:
                cred_data = base64.urlsafe_b64decode(row["public_key"] + "==")
                credentials.append(AttestedCredentialData(cred_data))
            except Exception:
                pass

        if not credentials:
            return {"success": False, "error": "invalid_credentials"}

        rp = PublicKeyCredentialRpEntity(RP_ID, RP_NAME)
        server = Fido2Server(rp)
        auth_data, state = server.authenticate_begin(credentials)

        self._webauthn_challenges[user_id] = {
            "state": state,
            "type": "auth",
            "timestamp": time.time(),
            "credentials": credentials,
        }

        return {
            "success": True,
            "options": _serialize_webauthn_options(dict(auth_data)),
        }

    def webauthn_auth_complete(
        self, user_id: str, response_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Complete WebAuthn authentication."""
        if not WEBAUTHN_AVAILABLE:
            return {"success": False, "error": "webauthn_unavailable"}

        challenge = self._webauthn_challenges.pop(user_id, None)
        if not challenge or challenge.get("type") != "auth":
            return {"success": False, "error": "no_challenge"}

        if time.time() - challenge["timestamp"] > 300:
            return {"success": False, "error": "expired"}

        try:
            rp = PublicKeyCredentialRpEntity(RP_ID, RP_NAME)
            server = Fido2Server(rp)
            server.authenticate_complete(
                challenge["state"],
                challenge["credentials"],
                response_data,
            )

            # Update last_used
            conn = _get_2fa_conn()
            try:
                # Update based on matching credential
                conn.execute(
                    "UPDATE webauthn_credentials SET last_used = ? "
                    "WHERE user_id = ?",
                    (time.time(), user_id),
                )
                conn.commit()
            finally:
                conn.close()

            return {"success": True, "method": "webauthn"}
        except Exception as exc:
            logger.warning("WebAuthn authentication failed: %s", exc)
            return {"success": False, "error": "auth_failed",
                    "message": str(exc)}

    def list_webauthn_credentials(self, user_id: str) -> list[dict[str, Any]]:
        """List all WebAuthn credentials for a user."""
        conn = _get_2fa_conn()
        try:
            rows = conn.execute(
                "SELECT credential_id, name, created_at, last_used, sign_count "
                "FROM webauthn_credentials WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def remove_webauthn_credential(
        self, user_id: str, credential_id: str
    ) -> bool:
        """Remove a WebAuthn credential."""
        conn = _get_2fa_conn()
        try:
            cur = conn.execute(
                "DELETE FROM webauthn_credentials "
                "WHERE credential_id = ? AND user_id = ?",
                (credential_id, user_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    # -- TOTP -----------------------------------------------------------------

    def totp_setup(self, user_id: str) -> dict[str, Any]:
        """Generate a new TOTP secret for a user.

        Returns the secret and provisioning URI.  The secret must
        be verified with totp_verify() before it becomes active.
        """
        if not TOTP_AVAILABLE:
            return {"success": False, "error": "totp_unavailable",
                    "message": "pyotp not installed"}

        secret = pyotp.random_base32()
        encrypted = _encrypt_secret(secret)

        conn = _get_2fa_conn()
        try:
            # Upsert: replace existing unverified config
            conn.execute(
                """INSERT OR REPLACE INTO totp_config
                   (user_id, secret_encrypted, verified, created_at)
                   VALUES (?, ?, 0, ?)""",
                (user_id, encrypted, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

        totp = pyotp.TOTP(secret)
        uri = totp.provisioning_uri(name=user_id, issuer_name=RP_NAME)

        result: dict[str, Any] = {
            "success": True,
            "secret": secret,
            "uri": uri,
        }

        # Generate QR code if available
        if QRCODE_AVAILABLE:
            try:
                qr = qrcode.QRCode(box_size=6, border=2)
                qr.add_data(uri)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                buf = _io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                result["qr_code_base64"] = f"data:image/png;base64,{b64}"
            except Exception:
                pass

        return result

    def totp_verify(self, user_id: str, code: str) -> dict[str, Any]:
        """Verify a TOTP code and activate TOTP for the user.

        Called during setup to confirm the user has correctly
        configured their authenticator app.
        """
        if not TOTP_AVAILABLE:
            return {"success": False, "error": "totp_unavailable"}

        conn = _get_2fa_conn()
        try:
            row = conn.execute(
                "SELECT secret_encrypted FROM totp_config WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if not row:
                return {"success": False, "error": "not_configured",
                        "message": "TOTP not set up for this user"}

            secret = _decrypt_secret(row["secret_encrypted"])
            totp = pyotp.TOTP(secret)

            if totp.verify(code, valid_window=1):
                conn.execute(
                    "UPDATE totp_config SET verified = 1 WHERE user_id = ?",
                    (user_id,),
                )
                conn.commit()
                return {"success": True, "message": "TOTP verified and activated"}
            else:
                return {"success": False, "error": "invalid_code",
                        "message": "Invalid TOTP code"}
        finally:
            conn.close()

    def totp_validate(self, user_id: str, code: str) -> bool:
        """Validate a TOTP code during authentication.

        Rate limited: max 5 attempts per 5 minutes. AU-06: a code is single-use
        within its validity window -- the consumed time-step is recorded per
        user and a code at or below the stored step is rejected, so a captured
        live code cannot be replayed during the +/-1 step window.
        """
        if not TOTP_AVAILABLE:
            return False

        # Rate limiting
        now = time.time()
        attempts = self._totp_attempts.get(user_id, [])
        attempts = [t for t in attempts if now - t < TOTP_WINDOW_SECONDS]
        if len(attempts) >= MAX_TOTP_ATTEMPTS_PER_WINDOW:
            logger.warning("TOTP rate limit exceeded for user %s", user_id)
            return False
        attempts.append(now)
        self._totp_attempts[user_id] = attempts

        conn = _get_2fa_conn()
        try:
            row = conn.execute(
                "SELECT secret_encrypted, verified, last_step FROM totp_config "
                "WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if not row or not row["verified"]:
                return False

            secret = _decrypt_secret(row["secret_encrypted"])
            totp = pyotp.TOTP(secret)
            if not totp.verify(code, valid_window=1):
                return False

            # AU-06: reject replay of a code within its validity window.
            matched_step = _totp_matched_step(totp, code, now)
            last_step = row["last_step"] or 0
            if matched_step is not None and matched_step <= last_step:
                logger.warning("TOTP replay rejected for user %s", user_id)
                return False
            if matched_step is not None:
                conn.execute(
                    "UPDATE totp_config SET last_step = ? WHERE user_id = ?",
                    (matched_step, user_id),
                )
                conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def totp_disable(self, user_id: str) -> bool:
        """Disable TOTP for a user."""
        conn = _get_2fa_conn()
        try:
            cur = conn.execute(
                "DELETE FROM totp_config WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    # -- Recovery codes -------------------------------------------------------

    def generate_recovery_codes(self, user_id: str) -> list[str]:
        """Generate new recovery codes.  Replaces any existing codes.

        The plaintext codes are returned ONCE and never stored.
        Only hashes are persisted.
        """
        conn = _get_2fa_conn()
        try:
            # Delete existing codes
            conn.execute(
                "DELETE FROM recovery_codes WHERE user_id = ?",
                (user_id,),
            )

            codes = []
            now = time.time()
            for _ in range(RECOVERY_CODE_COUNT):
                code = secrets.token_hex(RECOVERY_CODE_LENGTH // 2)
                code_hash = _hash_code(code)
                conn.execute(
                    """INSERT INTO recovery_codes
                       (user_id, code_hash, created_at) VALUES (?, ?, ?)""",
                    (user_id, code_hash, now),
                )
                codes.append(code)

            conn.commit()
            return codes
        finally:
            conn.close()

    def validate_recovery_code(self, user_id: str, code: str) -> bool:
        """Validate and consume a recovery code.  One-time use.

        Rate limited: 3 attempts per hour. AU-03: matches both the v2
        (master-key-derived) and the legacy hash schemes; a match against a
        legacy hash is still accepted, and the user's remaining recovery codes
        are reported as needing re-issue via recovery_reissue_required().
        """
        now = time.time()
        attempts = self._recovery_attempts.get(user_id, [])
        attempts = [t for t in attempts if now - t < 3600]
        if len(attempts) >= MAX_RECOVERY_ATTEMPTS_PER_HOUR:
            logger.warning(
                "Recovery code rate limit exceeded for user %s", user_id
            )
            return False
        attempts.append(now)
        self._recovery_attempts[user_id] = attempts

        conn = _get_2fa_conn()
        try:
            rows = conn.execute(
                "SELECT id, code_hash FROM recovery_codes "
                "WHERE user_id = ? AND used = 0",
                (user_id,),
            ).fetchall()
            matched_id = None
            for r in rows:
                ok, _ = _verify_code(code, r["code_hash"])
                if ok:
                    matched_id = r["id"]
                    break
            if matched_id is None:
                return False

            conn.execute(
                "UPDATE recovery_codes SET used = 1, used_at = ? WHERE id = ?",
                (now, matched_id),
            )
            conn.commit()

            # Audit
            try:
                from opti_oignon.security_mode import _audit_log
                _audit_log(
                    "recovery_code_used",
                    severity="WARNING",
                    user_id=user_id,
                )
            except Exception:
                pass

            return True
        finally:
            conn.close()

    def recovery_reissue_required(self, user_id: str) -> bool:
        """Whether the user has unused recovery codes under the legacy scheme.

        AU-03: pre-AU-03 recovery codes were HMAC'd with a public constant and
        cannot be transparently re-hashed (the plaintext is not stored). This
        returns True when any unused recovery code lacks the v2 scheme prefix, so
        the user can be prompted to re-issue (regenerate). It becomes False once
        regenerated, since new codes are keyed under the master-key scheme.
        """
        conn = _get_2fa_conn()
        try:
            rows = conn.execute(
                "SELECT code_hash FROM recovery_codes "
                "WHERE user_id = ? AND used = 0",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()
        return any(not r["code_hash"].startswith(_V2_PREFIX) for r in rows)

    # -- App-specific passwords -----------------------------------------------

    def create_app_password(
        self, user_id: str, name: str
    ) -> dict[str, Any]:
        """Create an app-specific password for CLI use.

        Returns the plaintext password ONCE.
        """
        password = secrets.token_urlsafe(APP_PASSWORD_LENGTH)
        pw_hash = _hash_code(password)
        pw_id = secrets.token_urlsafe(8)

        conn = _get_2fa_conn()
        try:
            conn.execute(
                """INSERT INTO app_passwords
                   (password_id, user_id, name, password_hash, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (pw_id, user_id, name, pw_hash, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

        return {
            "success": True,
            "password_id": pw_id,
            "name": name,
            "password": password,  # shown once
        }

    def validate_app_password(self, user_id: str, password: str) -> bool:
        """Validate an app-specific password.

        AU-03: matches both the v2 (master-key-derived) and the legacy hash
        schemes. A legacy hash is transparently rehashed to v2 on a successful
        match (the plaintext is available at validation time), so app passwords
        upgrade in place as they are used.
        """
        conn = _get_2fa_conn()
        try:
            rows = conn.execute(
                "SELECT password_id, password_hash FROM app_passwords "
                "WHERE user_id = ? AND revoked = 0",
                (user_id,),
            ).fetchall()
            matched_id = None
            is_legacy = False
            for r in rows:
                ok, legacy = _verify_code(password, r["password_hash"])
                if ok:
                    matched_id = r["password_id"]
                    is_legacy = legacy
                    break
            if matched_id is None:
                return False

            if is_legacy:
                # Rehash-on-use: upgrade the stored hash to the v2 scheme.
                conn.execute(
                    "UPDATE app_passwords SET last_used = ?, password_hash = ? "
                    "WHERE password_id = ?",
                    (time.time(), _hash_code(password), matched_id),
                )
            else:
                conn.execute(
                    "UPDATE app_passwords SET last_used = ? "
                    "WHERE password_id = ?",
                    (time.time(), matched_id),
                )
            conn.commit()
            return True
        finally:
            conn.close()

    def revoke_app_password(self, user_id: str, password_id: str) -> bool:
        """Revoke an app-specific password."""
        conn = _get_2fa_conn()
        try:
            cur = conn.execute(
                "UPDATE app_passwords SET revoked = 1 "
                "WHERE password_id = ? AND user_id = ?",
                (password_id, user_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def list_app_passwords(self, user_id: str) -> list[dict[str, Any]]:
        """List app-specific passwords (without the actual passwords)."""
        conn = _get_2fa_conn()
        try:
            rows = conn.execute(
                "SELECT password_id, name, created_at, last_used, revoked "
                "FROM app_passwords WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -- Unified validation ---------------------------------------------------

    def validate_2fa(
        self, user_id: str, code: str, method: str = "auto"
    ) -> dict[str, Any]:
        """Validate a 2FA code using the appropriate method.

        method:
          - "auto": try TOTP, then recovery code
          - "totp": TOTP only
          - "recovery": recovery code only
          - "app_password": app-specific password
        """
        if method == "auto" or method == "totp":
            if self.totp_validate(user_id, code):
                return {"success": True, "method": "totp"}

        if method == "auto" or method == "recovery":
            if self.validate_recovery_code(user_id, code):
                return {"success": True, "method": "recovery"}

        if method == "app_password":
            if self.validate_app_password(user_id, code):
                return {"success": True, "method": "app_password"}

        return {"success": False, "error": "invalid_code",
                "message": "Invalid 2FA code"}

    # -- Cleanup ---------------------------------------------------------------

    def disable_all(self, user_id: str) -> dict[str, Any]:
        """Disable all 2FA methods for a user."""
        conn = _get_2fa_conn()
        try:
            conn.execute(
                "DELETE FROM webauthn_credentials WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM totp_config WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM recovery_codes WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "UPDATE app_passwords SET revoked = 1 WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
            return {"success": True, "message": "All 2FA methods disabled"}
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# WebAuthn serialization helpers
# ---------------------------------------------------------------------------

def _serialize_webauthn_options(options: dict[str, Any]) -> dict[str, Any]:
    """Serialize WebAuthn options for JSON transport.

    Converts bytes to base64url strings.
    """
    import copy
    result = copy.deepcopy(options)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, bytes):
            return base64.urlsafe_b64encode(obj).decode().rstrip("=")
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    return _convert(result)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

two_factor_manager = TwoFactorAuthManager()
