#!/usr/bin/env python3
"""
Two-Factor Authentication for Opti-Oignon.

Supports three 2FA methods (in priority order):

  1. **WebAuthn/FIDO2** -- hardware keys (YubiKey 5, Google Titan),
     platform authenticators (Apple Passkeys, Windows Hello)
  2. **TOTP** -- time-based one-time passwords (Google Authenticator,
     Authy) as backup when hardware key unavailable
  3. **Recovery codes** -- 10 one-time codes shown once at setup,
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
import logging
import secrets
import sqlite3
import string
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opti_oignon.db_utils import safe_connect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature availability
# ---------------------------------------------------------------------------

WEBAUTHN_AVAILABLE = False
try:
    from fido2.server import Fido2Server  # type: ignore[import]
    from fido2.webauthn import (  # type: ignore[import]
        AttestedCredentialData,
        AuthenticatorData,  # noqa: F401
        PublicKeyCredentialRpEntity,
        PublicKeyCredentialUserEntity,
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
    import io as _io

    import qrcode  # type: ignore[import]
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
RECOVERY_CODE_LENGTH = 16  # 16 hex chars = 64 bits (was 8 = 32 bits)
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

# Whether the schema has been built in this process. The cost is paid once,
# at the first connection, rather than at import: a module that builds a
# database merely by being imported charges every caller for a capability
# most of them never use, and does it at a moment where a refusal -- an
# operator who requires encrypted storage, say -- cannot be handled by
# anyone.
_SCHEMA_READY = False


def _get_2fa_conn() -> sqlite3.Connection:
    """Get a connection to the 2FA database.

    Audit fix: routes through get_encrypted_connection() for
    SQLCipher support when available.
    """
    global _SCHEMA_READY

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = safe_connect(str(_2FA_DB_PATH), timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    if not _SCHEMA_READY:
        # Set before the call, not after: the initialiser asks for its own
        # connection and would otherwise recur through this branch.
        _SCHEMA_READY = True
        _init_2fa_db()
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
        _retire_unbound(conn)
    finally:
        conn.close()


# The identities no account holds. Up to block 176, every second factor enrolled from the settings was stored under
# the literal "admin", which no login ever read and which any session could write or overwrite: no account can
# vouch for those rows. They are retired -- deleted once, their counts kept in ``retired_factors`` and audited --
# and never handed to an account.
_UNBOUND_IDS = ("admin",)


def _retire_unbound(conn: Any) -> None:
    """Delete, in one transaction, the factors stored under an identity no account holds, and keep their counts.

    Run at the first connection of each process, after the schema; the next process finds nothing, records nothing
    and audits nothing. Best effort: a store that refuses -- read-only, locked -- keeps the rows, which nothing reads
    any more, and the connection still serves, so a retirement never turns into a refused login; the refusal is
    audited. Its record's table is its own, made here, so that the schema every connection needs writes nothing new
    on an old store. It never decrypts a secret, and its audit names counts alone.
    """
    counts = {"totp": 0, "webauthn": 0, "recovery_codes": 0, "app_passwords": 0}
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS retired_factors ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, retired_at REAL NOT NULL, totp INTEGER NOT NULL, "
            "webauthn INTEGER NOT NULL, recovery_codes INTEGER NOT NULL, app_passwords INTEGER NOT NULL)"
        )
        for unbound in _UNBOUND_IDS:
            counts["totp"] += conn.execute(
                "SELECT COUNT(*) FROM totp_config WHERE user_id = ?", (unbound,)).fetchone()[0]
            counts["webauthn"] += conn.execute(
                "SELECT COUNT(*) FROM webauthn_credentials WHERE user_id = ?", (unbound,)).fetchone()[0]
            counts["recovery_codes"] += conn.execute(
                "SELECT COUNT(*) FROM recovery_codes WHERE user_id = ?", (unbound,)).fetchone()[0]
            counts["app_passwords"] += conn.execute(
                "SELECT COUNT(*) FROM app_passwords WHERE user_id = ?", (unbound,)).fetchone()[0]
            conn.execute("DELETE FROM totp_config WHERE user_id = ?", (unbound,))
            conn.execute("DELETE FROM webauthn_credentials WHERE user_id = ?", (unbound,))
            conn.execute("DELETE FROM recovery_codes WHERE user_id = ?", (unbound,))
            conn.execute("DELETE FROM app_passwords WHERE user_id = ?", (unbound,))
        if sum(counts.values()):
            conn.execute(
                "INSERT INTO retired_factors (retired_at, totp, webauthn, recovery_codes, app_passwords) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time(), counts["totp"], counts["webauthn"], counts["recovery_codes"],
                 counts["app_passwords"]),
            )
        conn.commit()
    except Exception as exc:  # noqa: BLE001 - a retirement that cannot be written keeps rows nothing reads
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001 - nothing to undo when the transaction never opened
            pass
        logger.warning("Unbound second factors not retired (%s): the rows stay, unread", type(exc).__name__)
        try:
            from opti_oignon.security_mode import _audit_log
            _audit_log("two_factor_unbound_not_retired", severity="WARNING", error=type(exc).__name__)
        except Exception:  # noqa: BLE001 - the audit is best effort
            pass
        return
    if not sum(counts.values()):
        return
    logger.warning("Retired the second factors of no account: %s", counts)
    try:
        from opti_oignon.security_mode import _audit_log
        _audit_log("two_factor_unbound_retired", severity="WARNING", **counts)
    except Exception:  # noqa: BLE001 - the audit is best effort; the retirement stands
        pass



# ---------------------------------------------------------------------------
# Encryption helpers (for TOTP secret storage)
# ---------------------------------------------------------------------------

def _encrypt_secret(plaintext: str) -> str:
    """Encrypt a TOTP secret using the project encryption module: sealed when
    encryption is on, base64 (``B64:``, no secret) only when it is off. When it
    is on and no key loads, or security.yaml cannot say, or the cipher fails,
    raises ``EncryptionUnavailable`` with the reason, and nothing is stored."""
    from opti_oignon.encryption import EncryptionManager
    try:
        mgr = EncryptionManager()
    except Exception as exc:
        from opti_oignon.encryption import EncryptionUnavailable
        raise EncryptionUnavailable(
            f"the field encryption cannot be built ({type(exc).__name__}): nothing is stored") from exc
    # A manager with no refusing state is on or off.
    refusing = getattr(mgr, "refusing", None)
    if refusing:
        from opti_oignon.encryption import EncryptionUnavailable
        raise EncryptionUnavailable(refusing)
    if mgr.enabled:
        return mgr.encrypt(plaintext)
    return "B64:" + base64.b64encode(plaintext.encode()).decode()


# The prefixes of a sealed field, encryption's _ENCRYPTED_PREFIX_V2 and _V1 (a
# contract holds them equal): a stored secret that carries neither, nor "B64:",
# was stored before encryption.
_SEALED_PREFIXES = ("ENC2:", "ENC:")


def _decrypt_secret(ciphertext: str) -> str:
    """Decrypt a stored TOTP secret: with the key whenever one loads, also while
    the manager refuses to seal (a security.yaml that cannot say). A sealed
    secret that does not decrypt (no key loads, or the cipher fails) raises
    ``EncryptionUnavailable`` with the reason: it is never handed on as the
    secret, which no code would match. A row stored before encryption, neither
    sealed nor ``B64:``, is the secret itself."""
    if ciphertext.startswith("B64:"):
        return base64.b64decode(ciphertext[4:]).decode()
    if not ciphertext.startswith(_SEALED_PREFIXES):
        return ciphertext
    from opti_oignon.encryption import EncryptionManager, get_encryption_key
    try:
        mgr = EncryptionManager()
        if not mgr.has_key:
            # A manager built with the switch off loads no key; a key that loads still reads what it sealed.
            key = get_encryption_key()
            if key is not None:
                mgr = EncryptionManager(key=key, enabled=False)
        has_key = mgr.has_key
        plain = mgr.decrypt(ciphertext) if has_key else ciphertext
    except Exception as exc:  # noqa: BLE001 - said below as the reason no code can be checked
        reason = f"the TOTP secret does not decrypt ({type(exc).__name__}): no TOTP code can be checked"
    else:
        if not has_key:
            reason = "the TOTP secret is sealed and no key loads: no TOTP code can be checked until one does"
        elif plain.startswith(_SEALED_PREFIXES):
            # The manager hands a value it cannot open back as it is.
            reason = "the TOTP secret does not decrypt with the key that loads: no TOTP code can be checked"
        else:
            return plain
    from opti_oignon.encryption import EncryptionUnavailable
    raise EncryptionUnavailable(reason)


# AU-03: recovery codes and app passwords are HMAC'd under a subkey derived from
# the master encryption key on its own HKDF domain (Kerckhoffs-clean: the master
# key is the only secret), versioned with a "v2:" prefix. With no key that loads
# they are hashed with scrypt under a salt of their own ("s1:"), which needs no
# secret and makes a search of a 64-bit recovery code cost 2^64 scrypt runs. The
# pre-AU-03 scheme (the public-constant or keyfile HMAC, or plain SHA-256) is
# never written: it is retained only to verify and migrate existing hashes. The
# domain string is distinct from the SQLCipher subkey, the learned-router MAC,
# and the audit anchor.
_CODE_HMAC_INFO = b"oo-2fa-code-hmac-v2"
_V2_PREFIX = "v2:"
_S1_PREFIX = "s1:"
_S1_N, _S1_R, _S1_P = 2 ** 14, 8, 1
_S1_SALT_BYTES = 16
_S1_HASH_BYTES = 32


def _derive_2fa_code_key() -> bytes | None:
    """Derive the 2FA code-hashing subkey off the master key (HMAC-SHA256).

    Domain-separated from the SQLCipher subkey, the learned-router MAC and the
    audit anchor (distinct info string). Returns None when no master key loads,
    in which case codes are hashed with scrypt under a salt of their own (s1).
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


def _scrypt_code(code: str, salt: bytes) -> bytes:
    """The s1 digest of a code under a salt."""
    return hashlib.scrypt(code.encode("utf-8"), salt=salt, n=_S1_N, r=_S1_R, p=_S1_P, dklen=_S1_HASH_BYTES)


def _hash_code_s1(code: str) -> str:
    """Hash a code with scrypt under a fresh salt (the scheme with no key): ``s1:<salt hex>:<digest hex>``."""
    salt = secrets.token_bytes(_S1_SALT_BYTES)
    return f"{_S1_PREFIX}{salt.hex()}:{_scrypt_code(code, salt).hex()}"


def _verify_code_s1(code: str, stored: str) -> bool:
    """Whether a code matches an s1 hash; one that is not well formed matches nothing."""
    salt_hex, sep, digest_hex = stored[len(_S1_PREFIX):].partition(":")
    if (not sep or len(salt_hex) != 2 * _S1_SALT_BYTES or len(digest_hex) != 2 * _S1_HASH_BYTES
            or not all(c in string.hexdigits for c in salt_hex + digest_hex)):
        return False
    return _hmac.compare_digest(_scrypt_code(code, bytes.fromhex(salt_hex)), bytes.fromhex(digest_hex))


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


def _code_hasher():
    """The hash of the scheme in force, the key looked up once: a set of codes made with it is of one scheme.

    AU-03: prefers a v2 hash keyed off the master encryption key on its own HKDF
    domain (the master key is the only secret, so it is Kerckhoffs-clean). With
    no key that loads, an s1 hash: scrypt under a salt of its own, which needs
    no secret. Never the legacy scheme.
    """
    subkey = _derive_2fa_code_key()
    if subkey is not None:
        return lambda code: _hash_code_v2(code, subkey)
    return _hash_code_s1


def _hash_code(code: str) -> str:
    """Hash a recovery code or app password for storage, in the scheme in force (``_code_hasher``)."""
    return _code_hasher()(code)


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
    if stored.startswith(_S1_PREFIX):
        return _verify_code_s1(code, stored), False
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


def _totp_matched_step(totp: Any, code: str, now: float) -> int | None:
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
        # Every pending key ceremony -- a registration under the user id, a login under its own key --, written and
        # read under the lock: a login's sweep walks them while other requests add and claim theirs.
        self._webauthn_challenges: dict[str, Any] = {}
        self._challenges_lock = threading.Lock()
        self._totp_attempts: dict[str, list[float]] = {}  # user_id -> timestamps
        self._recovery_attempts: dict[str, list[float]] = {}
        # The two windows above, each read, checked and written back under this lock: tries sent together are each
        # counted before any code is read.
        self._attempts_lock = threading.Lock()
        # The wrong codes sent to confirm a TOTP, per account: the enrolment at login makes a new challenge at each
        # login, so its tries are counted here too, across them.
        self._totp_confirm_failures: dict[str, list[float]] = {}
        self._confirm_lock = threading.Lock()

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
                not r["code_hash"].startswith((_V2_PREFIX, _S1_PREFIX)) for r in recovery_rows
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

    def retired_unbound(self) -> dict[str, Any] | None:
        """The latest retirement of factors no account held: when, and how many of each; None when there was none,
        or when the record cannot be read -- it informs, it never refuses."""
        try:
            conn = _get_2fa_conn()
        except Exception:  # noqa: BLE001 - a store that cannot be opened has no record to tell
            return None
        try:
            row = conn.execute(
                "SELECT retired_at, totp, webauthn, recovery_codes, app_passwords "
                "FROM retired_factors ORDER BY id DESC LIMIT 1"
            ).fetchone()
        except sqlite3.Error:
            return None
        finally:
            conn.close()
        if row is None:
            return None
        return {"at": row["retired_at"], "totp": row["totp"], "webauthn": row["webauthn"],
                "recovery_codes": row["recovery_codes"], "app_passwords": row["app_passwords"]}

    @staticmethod
    def totp_checkable() -> bool:
        """Whether a TOTP code can be checked here: the library that computes them is installed."""
        return TOTP_AVAILABLE

    @staticmethod
    def recovery_shaped(code: str) -> bool:
        """Whether a code has a recovery code's shape: as many hexadecimal characters as the generator makes."""
        return len(code) == RECOVERY_CODE_LENGTH and all(c in string.hexdigits for c in code)

    @staticmethod
    def required_by_mode() -> bool:
        """Whether the mode asks a second factor of every account: in Bulbe, and when the mode cannot be read."""
        try:
            from opti_oignon.security_mode import is_bulbe
            return bool(is_bulbe())
        except Exception:  # noqa: BLE001 - an unreadable mode is read as Bulbe, where the factor is asked
            return True

    def is_2fa_required(self, user_id: str, status: TwoFAStatus | None = None) -> bool:
        """Whether the login asks this account a second factor: every account in Bulbe (and when the mode cannot be
        read), in Daily one with a factor active -- read from ``status`` when the caller holds it."""
        if self.required_by_mode():
            return True
        return (status if status is not None else self.get_status(user_id)).any_method_active

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

        rp = PublicKeyCredentialRpEntity(id=RP_ID, name=RP_NAME)
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
        with self._challenges_lock:
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

        with self._challenges_lock:
            challenge = self._webauthn_challenges.pop(user_id, None)
        if not challenge or challenge.get("type") != "register":
            return {"success": False, "error": "no_challenge",
                    "message": "No pending registration challenge"}

        # Timeout: 5 minutes
        if time.time() - challenge["timestamp"] > 300:
            return {"success": False, "error": "expired",
                    "message": "Registration challenge expired"}

        try:
            rp = PublicKeyCredentialRpEntity(id=RP_ID, name=RP_NAME)
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

    def webauthn_auth_begin(self, user_id: str, *, key: str | None = None) -> dict[str, Any]:
        """Begin WebAuthn authentication challenge. The state is kept under ``key`` -- one login's own, so that
        another login of the same account, or a pending registration, never overwrites it -- or under the user id."""
        if not WEBAUTHN_AVAILABLE:
            return {"success": False, "error": "webauthn_unavailable"}

        found, credentials = self._held_credentials(user_id)
        if not found:
            return {"success": False, "error": "no_credentials",
                    "message": "No WebAuthn credentials registered"}

        if not credentials:
            return {"success": False, "error": "invalid_credentials"}

        rp = PublicKeyCredentialRpEntity(id=RP_ID, name=RP_NAME)
        server = Fido2Server(rp)
        auth_data, state = server.authenticate_begin(credentials)

        now = time.time()
        with self._challenges_lock:
            for stale in [k for k, v in self._webauthn_challenges.items() if now - v.get("timestamp", 0) > 300]:
                del self._webauthn_challenges[stale]
            self._webauthn_challenges[key or user_id] = {
                "state": state,
                "type": "auth",
                "timestamp": now,
                "user_id": user_id,
            }

        return {
            "success": True,
            "options": _serialize_webauthn_options(dict(auth_data)),
        }

    def webauthn_auth_complete(
        self, user_id: str, response_data: dict[str, Any], *, key: str | None = None
    ) -> dict[str, Any]:
        """Complete WebAuthn authentication, against the keys the account holds NOW: a key removed, or an account
        purged, while a login waits no longer answers it. The challenge is claimed while one assertion is checked --
        the same answer sent twice at once passes once -- and spent by the assertion that passes, or by its expiry;
        an assertion that fails gives it back to the next try, which the login's own attempt count bounds."""
        if not WEBAUTHN_AVAILABLE:
            return {"success": False, "error": "webauthn_unavailable"}

        slot = key or user_id
        with self._challenges_lock:
            challenge = self._webauthn_challenges.get(slot)
            if not challenge or challenge.get("type") != "auth" or challenge.get("user_id", user_id) != user_id:
                return {"success": False, "error": "no_challenge"}
            del self._webauthn_challenges[slot]

        if time.time() - challenge["timestamp"] > 300:
            return {"success": False, "error": "expired"}

        try:
            _found, credentials = self._held_credentials(user_id)
            rp = PublicKeyCredentialRpEntity(id=RP_ID, name=RP_NAME)
            server = Fido2Server(rp)
            server.authenticate_complete(
                challenge["state"],
                credentials,
                response_data,
            )

            # The assertion passed, and its challenge stays spent whatever follows: when the key was last used is
            # recorded where the store can write, and a store that cannot is no reason to refuse a verified key.
            try:
                conn = _get_2fa_conn()
                try:
                    conn.execute(
                        "UPDATE webauthn_credentials SET last_used = ? "
                        "WHERE user_id = ?",
                        (time.time(), user_id),
                    )
                    conn.commit()
                finally:
                    conn.close()
            except Exception as exc:  # noqa: BLE001 - the last use is best effort
                logger.warning("Security key's last use not recorded (%s)", type(exc).__name__)

            return {"success": True, "method": "webauthn"}
        except Exception as exc:
            logger.warning("WebAuthn authentication failed: %s", exc)
            with self._challenges_lock:
                self._webauthn_challenges.setdefault(slot, challenge)
            return {"success": False, "error": "auth_failed",
                    "message": str(exc)}

    def _held_credentials(self, user_id: str) -> tuple[int, list[Any]]:
        """The account's security keys as the store holds them now: how many rows, and those the library reads."""
        conn = _get_2fa_conn()
        try:
            rows = conn.execute(
                "SELECT credential_id, public_key FROM webauthn_credentials "
                "WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()
        credentials = []
        for row in rows:
            try:
                cred_data = base64.urlsafe_b64decode(row["public_key"] + "==")
                credentials.append(AttestedCredentialData(cred_data))
            except Exception:
                pass
        return len(rows), credentials

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

    def totp_setup(self, user_id: str, *, label: str | None = None) -> dict[str, Any]:
        """Generate a new TOTP secret for a user.

        Returns the secret and provisioning URI, whose account name is ``label`` (the account's username) when
        given, else the user id.  The secret must be verified with totp_verify() before it becomes active.  A secret
        that is active is never replaced, whoever asks and whenever: the setup then answers ``already_active``, in the
        same statement that would have replaced it.
        """
        if not TOTP_AVAILABLE:
            return {"success": False, "error": "totp_unavailable",
                    "message": "pyotp not installed"}

        secret = pyotp.random_base32()
        encrypted = _encrypt_secret(secret)

        conn = _get_2fa_conn()
        try:
            # Upsert: replace an unverified config only -- the condition and the write are one statement, so a
            # confirmation committed by another request since a status was read is never undone.
            replaced = conn.execute(
                """INSERT INTO totp_config
                   (user_id, secret_encrypted, verified, last_step, created_at)
                   VALUES (?, ?, 0, 0, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                     secret_encrypted = excluded.secret_encrypted, verified = 0, last_step = 0,
                     created_at = excluded.created_at
                   WHERE totp_config.verified = 0""",
                (user_id, encrypted, time.time()),
            ).rowcount
            conn.commit()
        finally:
            conn.close()
        if replaced == 0:
            return {"success": False, "error": "already_active",
                    "message": "TOTP is already active for this user"}

        totp = pyotp.TOTP(secret)
        uri = totp.provisioning_uri(name=label or user_id, issuer_name=RP_NAME)

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
        configured their authenticator app.  The step the code matched is
        recorded as consumed (AU-06): the code that confirmed the secret does
        not sign in.  The secret activated is the one the code was checked
        against, in the statement that activates it: one another setup put in
        its place meanwhile is never activated (``changed``).  Wrong codes are
        counted per account, five per window, each try reserved before its code
        is read, so that tries sent together are all counted; past them the
        answer is ``rate_limited`` until the window ends.
        """
        if not TOTP_AVAILABLE:
            return {"success": False, "error": "totp_unavailable"}

        now = time.time()
        with self._confirm_lock:
            failures = [t for t in self._totp_confirm_failures.get(user_id, []) if now - t < TOTP_WINDOW_SECONDS]
            if len(failures) >= MAX_TOTP_ATTEMPTS_PER_WINDOW:
                self._totp_confirm_failures[user_id] = failures
                logger.warning("TOTP confirmation rate limit exceeded for user %s", user_id)
                return {"success": False, "error": "rate_limited",
                        "message": "Too many wrong codes for this account: wait a few minutes"}
            failures.append(now)
            self._totp_confirm_failures[user_id] = failures

        conn = _get_2fa_conn()
        try:
            row = conn.execute(
                "SELECT secret_encrypted FROM totp_config WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if not row:
                self._release_confirm_try(user_id, now)
                return {"success": False, "error": "not_configured",
                        "message": "TOTP not set up for this user"}

            try:
                secret = _decrypt_secret(row["secret_encrypted"])
            except Exception:
                # No code was checked: the try goes back to the account's window, and the refusal is said.
                self._release_confirm_try(user_id, now)
                raise
            totp = pyotp.TOTP(secret)

            if not totp.verify(code, valid_window=1):
                return {"success": False, "error": "invalid_code",
                        "message": "Invalid TOTP code"}
            self._release_confirm_try(user_id, now)
            # The step is never written back below one a sign-in consumed: a confirmation of an active TOTP with an
            # earlier code would let a used code sign in again.
            activated = conn.execute(
                "UPDATE totp_config SET verified = 1, last_step = MAX(COALESCE(last_step, 0), ?) "
                "WHERE user_id = ? AND secret_encrypted = ?",
                (_totp_matched_step(totp, code, now) or 0, user_id, row["secret_encrypted"]),
            ).rowcount
            conn.commit()
            if activated == 0:
                return {"success": False, "error": "changed",
                        "message": "The TOTP secret was set up again while this code was checked"}
            return {"success": True, "message": "TOTP verified and activated"}
        finally:
            conn.close()

    def _give_back(self, lock: Any, windows: dict[str, list[float]], user_id: str, at: float) -> None:
        """Give back a try reserved at ``at`` in an account's window: nothing was checked under it (a secret or
        codes no key that loads can read), and a code that was not checked is never counted wrong."""
        with lock:
            held = windows.get(user_id, [])
            if at in held:
                held.remove(at)

    def _release_confirm_try(self, user_id: str, at: float) -> None:
        """Give back a try ``totp_verify`` reserved at ``at``: its code was right, or there was nothing to check."""
        with self._confirm_lock:
            held = self._totp_confirm_failures.get(user_id, [])
            if at in held:
                held.remove(at)

    def totp_validate(self, user_id: str, code: str) -> bool:
        """Validate a TOTP code during authentication.

        Rate limited: max 5 attempts per 5 minutes, each counted before its
        code is read, so that tries sent together are each counted. AU-06: a
        code is single-use within its validity window -- the consumed
        time-step is recorded per user and a code at or below the stored step
        is rejected, so a captured live code cannot be replayed during the
        +/-1 step window; the step is written only over an earlier one, here
        and at the setup's confirmation, so of two right codes sent together one
        signs in, and no confirmation brings a used code back.
        """
        if not TOTP_AVAILABLE:
            return False

        # Rate limiting
        now = time.time()
        with self._attempts_lock:
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
                # The write is the check: of two right codes read together, only the one that moves the step on
                # signs in, and a step is never written back below a later one.
                moved = conn.execute(
                    "UPDATE totp_config SET last_step = ? WHERE user_id = ? AND COALESCE(last_step, 0) < ?",
                    (matched_step, user_id, matched_step),
                ).rowcount
                conn.commit()
                if moved != 1:
                    logger.warning("TOTP replay rejected for user %s", user_id)
                    return False
            return True
        except Exception as exc:
            # The field encryption's refusal (encryption.EncryptionUnavailable), asked by its class's name, is said:
            # a secret that does not decrypt checked no code, and no code is called wrong for it -- its try goes back.
            if type(exc).__name__ == "EncryptionUnavailable" and type(exc).__module__ == "opti_oignon.encryption":
                self._give_back(self._attempts_lock, self._totp_attempts, user_id, now)
                raise
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
        # Hashed before the write opens: with no key each hash is a scrypt run, and the store's lock is not held
        # through them.
        codes = [secrets.token_hex(RECOVERY_CODE_LENGTH // 2) for _ in range(RECOVERY_CODE_COUNT)]
        hasher = _code_hasher()
        hashes = [hasher(code) for code in codes]
        now = time.time()
        conn = _get_2fa_conn()
        try:
            # Delete existing codes
            conn.execute(
                "DELETE FROM recovery_codes WHERE user_id = ?",
                (user_id,),
            )
            for code_hash in hashes:
                conn.execute(
                    """INSERT INTO recovery_codes
                       (user_id, code_hash, created_at) VALUES (?, ?, ?)""",
                    (user_id, code_hash, now),
                )

            conn.commit()
            return codes
        finally:
            conn.close()

    def validate_recovery_code(self, user_id: str, code: str) -> bool:
        """Validate and consume a recovery code.  One-time use.

        Rate limited: 3 attempts per hour, each counted before its code is
        read, as TOTP codes are. AU-03: matches the v2 (master-key-derived),
        the s1 (scrypt, no key) and the legacy hash schemes; a match against a
        legacy hash is still accepted, and the user's remaining recovery codes
        are reported as needing re-issue via recovery_reissue_required(). Keyed
        (v2) codes no key that loads can check raise ``EncryptionUnavailable``
        when no other code matches; when not one code could be checked, the try
        goes back. A code read by two tries together is used by one.
        """
        now = time.time()
        with self._attempts_lock:
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
            keyed = _derive_2fa_code_key() is not None
            unchecked = 0
            for r in rows:
                if r["code_hash"].startswith(_V2_PREFIX) and not keyed:
                    unchecked += 1
                    continue
                ok, _ = _verify_code(code, r["code_hash"])
                if ok:
                    matched_id = r["id"]
                    break
            if matched_id is None:
                if unchecked:
                    if unchecked == len(rows):
                        # Not one code was checked: the try goes back to the account's window.
                        self._give_back(self._attempts_lock, self._recovery_attempts, user_id, now)
                    from opti_oignon.encryption import EncryptionUnavailable
                    raise EncryptionUnavailable(
                        f"{unchecked} recovery code(s) are keyed and no key loads: none of them can be checked "
                        "until one does")
                return False

            # The write is the check: of two tries that read the same unused code, one uses it.
            used = conn.execute(
                "UPDATE recovery_codes SET used = 1, used_at = ? WHERE id = ? AND used = 0",
                (now, matched_id),
            ).rowcount
            conn.commit()
            if used != 1:
                return False

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
        returns True when any unused recovery code is of neither the v2 nor the
        s1 scheme, so the user can be prompted to re-issue (regenerate). It
        becomes False once regenerated, since new codes are of one of the two.
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
        return any(not r["code_hash"].startswith((_V2_PREFIX, _S1_PREFIX)) for r in rows)

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

        AU-03: matches the v2 (master-key-derived), the s1 (scrypt, no key) and
        the legacy hash schemes. A legacy hash is transparently rehashed to the
        current scheme (v2, or s1 with no key) on a successful match (the
        plaintext is available at validation time), so app passwords upgrade in
        place as they are used.
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
                # Rehash-on-use: upgrade the stored hash to the current scheme (v2, or s1 with no key).
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

        Raises ``EncryptionUnavailable`` when the code cannot be checked: a
        TOTP secret that does not decrypt, recovery codes keyed with no key.

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

    def purge_account(self, user_id: str) -> dict[str, int]:
        """Delete every factor of an account that is gone -- keys, TOTP, recovery codes and app passwords, deleted
        rather than revoked -- and forget its pending challenges and attempt counts. Returns the counts deleted."""
        conn = _get_2fa_conn()
        try:
            counts = {
                "webauthn": conn.execute(
                    "DELETE FROM webauthn_credentials WHERE user_id = ?", (user_id,)).rowcount,
                "totp": conn.execute("DELETE FROM totp_config WHERE user_id = ?", (user_id,)).rowcount,
                "recovery_codes": conn.execute(
                    "DELETE FROM recovery_codes WHERE user_id = ?", (user_id,)).rowcount,
                "app_passwords": conn.execute(
                    "DELETE FROM app_passwords WHERE user_id = ?", (user_id,)).rowcount,
            }
            conn.commit()
        finally:
            conn.close()
        with self._challenges_lock:
            for slot in [k for k, v in self._webauthn_challenges.items()
                         if k == user_id or v.get("user_id") == user_id]:
                del self._webauthn_challenges[slot]
        with self._attempts_lock:
            self._totp_attempts.pop(user_id, None)
            self._recovery_attempts.pop(user_id, None)
        with self._confirm_lock:
            self._totp_confirm_failures.pop(user_id, None)
        return counts

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
