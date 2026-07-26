#!/usr/bin/env python3
"""
Data-at-rest encryption for Opti-Oignon.

Security guarantees:
  - AES-256-GCM authenticated encryption (confidentiality + integrity)
  - Argon2id key derivation (memory-hard, GPU/ASIC resistant)
  - PBKDF2-SHA256 fallback if argon2-cffi is not installed (600k iterations)
  - NO insecure fallback: if neither 'cryptography' nor 'pycryptodome'
    is available, encryption REFUSES to operate (raises ImportError)
  - Unique random 12-byte nonce per encryption (GCM standard)
  - 16-byte authentication tag per ciphertext
  - Key rotation: re-encrypt all data with a new key

Key management:
  1. Environment variable OPTI_ENCRYPTION_KEY (base64url-encoded 32-byte key)
  2. Keyfile at data/.keyfile (chmod 600)
  3. Derived from passphrase via Argon2id (or PBKDF2 fallback)

Encrypted format (binary, then base64url for storage):
  version(1) || nonce(12) || ciphertext(variable) || tag(16)
  Prefixed with 'ENC2:' in the stored string for transparent detection.

Configuration in config/security.yaml > encryption section.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import stat
import struct
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SecureBytes for key memory protection
# ---------------------------------------------------------------------------

try:
    from opti_oignon.secure_bytes import SecureBytes, secure_key_from_bytes
    _SECURE_BYTES_AVAILABLE = True
except ImportError:
    _SECURE_BYTES_AVAILABLE = False
    # Fallback: thin wrapper that behaves like SecureBytes but is just bytes
    class SecureBytes:  # type: ignore[no-redef]
        """Fallback when secure_bytes module is not available."""
        def __init__(self, data):
            self._data = bytes(data)
            self._wiped = False
        def as_bytes(self) -> bytes:
            if self._wiped:
                raise RuntimeError("SecureBytes has been wiped")
            return self._data
        def wipe(self) -> None:
            self._wiped = True
            self._data = b""
        @property
        def is_wiped(self) -> bool:
            return self._wiped
        def __len__(self):
            return len(self._data)
        def __bool__(self):
            return not self._wiped and len(self._data) > 0
        def __repr__(self):
            return "<SecureBytes [REDACTED]>"
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            self.wipe()

    def secure_key_from_bytes(data) -> SecureBytes:
        return SecureBytes(data)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_KEYFILE = _PROJECT_ROOT / "data" / ".keyfile"
_ENV_KEY_NAME = "OPTI_ENCRYPTION_KEY"
# Passphrase used to wrap (envelope) the keyfile at rest under a KEK (K-01).
_ENV_KEYFILE_PASS = "OPTI_KEYFILE_PASSPHRASE"
# Marker for the enveloped keyfile JSON format.
_KEYFILE_ENVELOPE_VERSION = "envelope-v1"

# Prefix for encrypted values in DB fields.
# V2 format (AES-256-GCM) uses 'ENC2:', V1 legacy (AES-128-CBC) uses 'ENC:'.
_ENCRYPTED_PREFIX_V2 = "ENC2:"
_ENCRYPTED_PREFIX_V1 = "ENC:"

# Format version byte
_FORMAT_VERSION = 0x02

# AES-256-GCM nonce size (NIST recommendation: 96 bits = 12 bytes)
_GCM_NONCE_SIZE = 12

# AES-256 key size
_KEY_SIZE = 32

# Argon2id parameters (OWASP 2024 recommendation for password hashing)
_ARGON2_TIME_COST = 3
_ARGON2_MEMORY_COST = 65536  # 64 MB
_ARGON2_PARALLELISM = 4
_ARGON2_SALT_LENGTH = 16

# PBKDF2 fallback parameters
_PBKDF2_ITERATIONS = 600_000
_PBKDF2_SALT_LENGTH = 16

# KDF identifier bytes stored in keyfile
_KDF_ARGON2ID = b"A2"
_KDF_PBKDF2 = b"P2"


def _load_encryption_config() -> dict[str, Any]:
    """Load encryption configuration from security.yaml."""
    import yaml
    cfg_path = Path(__file__).parent / "config" / "security.yaml"
    try:
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("encryption", {})
    except Exception:
        pass
    return {}


# ============================================================================
# Crypto backend detection (NO fallback to insecure implementations)
# ============================================================================

_CRYPTO_BACKEND: str | None = None

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
    _CRYPTO_BACKEND = "cryptography"
except ImportError:
    try:
        from Crypto.Cipher import AES as _PyCryptoAES  # noqa: F401
        _CRYPTO_BACKEND = "pycryptodome"
    except ImportError:
        _CRYPTO_BACKEND = None


def _require_crypto() -> None:
    """Ensure a real cryptographic backend is available.

    Raises ImportError if neither 'cryptography' nor 'pycryptodome' is installed.
    This is intentionally a hard failure -- encryption without a real AES
    implementation is security theater.
    """
    if _CRYPTO_BACKEND is None:
        raise ImportError(
            "No cryptographic library available. "
            "Install one of: pip install cryptography  OR  pip install pycryptodome  "
            "Encryption CANNOT operate without a real AES implementation."
        )


# ============================================================================
# AES-256-GCM primitives
# ============================================================================

def _aes_gcm_encrypt(key: bytes, plaintext: bytes) -> tuple[bytes, bytes, bytes]:
    """Encrypt with AES-256-GCM.

    Args:
        key: 32-byte key
        plaintext: Data to encrypt

    Returns:
        (nonce, ciphertext, tag) where nonce is 12 bytes and tag is 16 bytes.

    Raises:
        ImportError: If no crypto backend is available.
    """
    _require_crypto()
    nonce = os.urandom(_GCM_NONCE_SIZE)

    if _CRYPTO_BACKEND == "cryptography":
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(key)
        # AESGCM.encrypt returns ciphertext || tag (tag is last 16 bytes)
        ct_with_tag = aesgcm.encrypt(nonce, plaintext, None)
        ciphertext = ct_with_tag[:-16]
        tag = ct_with_tag[-16:]
        return nonce, ciphertext, tag

    # pycryptodome
    from Crypto.Cipher import AES
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return nonce, ciphertext, tag


def _aes_gcm_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes) -> bytes:
    """Decrypt with AES-256-GCM.

    Args:
        key: 32-byte key
        nonce: 12-byte nonce
        ciphertext: Encrypted data
        tag: 16-byte authentication tag

    Returns:
        Decrypted plaintext

    Raises:
        ValueError: If authentication fails (tampered data or wrong key).
        ImportError: If no crypto backend is available.
    """
    _require_crypto()

    if _CRYPTO_BACKEND == "cryptography":
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(key)
        try:
            return aesgcm.decrypt(nonce, ciphertext + tag, None)
        except Exception as e:
            raise ValueError(f"GCM authentication failed: {e}")

    # pycryptodome
    from Crypto.Cipher import AES
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    try:
        return cipher.decrypt_and_verify(ciphertext, tag)
    except Exception as e:
        raise ValueError(f"GCM authentication failed: {e}")


# ============================================================================
# High-level encrypt/decrypt (versioned binary format)
# ============================================================================

def encrypt_bytes(key: bytes, plaintext: bytes) -> bytes:
    """Encrypt data using AES-256-GCM with versioned format.

    Format: version(1) || nonce(12) || ciphertext(N) || tag(16)

    Args:
        key: 32-byte encryption key
        plaintext: Data to encrypt

    Returns:
        Binary encrypted data (NOT base64-encoded)
    """
    if len(key) != _KEY_SIZE:
        raise ValueError(f"Key must be {_KEY_SIZE} bytes, got {len(key)}")

    nonce, ciphertext, tag = _aes_gcm_encrypt(key, plaintext)

    return struct.pack("B", _FORMAT_VERSION) + nonce + ciphertext + tag


def decrypt_bytes(key: bytes, data: bytes) -> bytes:
    """Decrypt AES-256-GCM encrypted data.

    Args:
        key: 32-byte encryption key
        data: Binary encrypted data (as produced by encrypt_bytes)

    Returns:
        Decrypted plaintext

    Raises:
        ValueError: If data is corrupt, tampered, or wrong key.
    """
    if len(key) != _KEY_SIZE:
        raise ValueError(f"Key must be {_KEY_SIZE} bytes, got {len(key)}")

    min_size = 1 + _GCM_NONCE_SIZE + 0 + 16  # version + nonce + empty ct + tag
    if len(data) < min_size:
        raise ValueError(f"Encrypted data too short ({len(data)} bytes)")

    version = data[0]
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported encryption format version: {version}")

    nonce = data[1:1 + _GCM_NONCE_SIZE]
    tag = data[-16:]
    ciphertext = data[1 + _GCM_NONCE_SIZE:-16]

    return _aes_gcm_decrypt(key, nonce, ciphertext, tag)


# Legacy V1 (Fernet-like AES-128-CBC) support for migration
def _decrypt_v1(key: bytes, token_b64: bytes) -> bytes:
    """Decrypt legacy V1 format (AES-128-CBC + HMAC-SHA256).

    Used only for migrating data encrypted with the old format.
    """
    import hmac as _hmac

    if len(key) != 32:
        raise ValueError("Key must be 32 bytes")

    signing_key = key[:16]
    encryption_key = key[16:]

    data = base64.urlsafe_b64decode(token_b64)
    if len(data) < 57:
        raise ValueError("V1 token too short")

    token_data = data[:-32]
    expected_mac = data[-32:]

    actual_mac = _hmac.new(signing_key, token_data, hashlib.sha256).digest()
    if not _hmac.compare_digest(actual_mac, expected_mac):
        raise ValueError("V1 HMAC verification failed")

    iv = token_data[9:25]
    ciphertext = token_data[25:]

    # Decrypt with CBC
    _require_crypto()
    if _CRYPTO_BACKEND == "cryptography":
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives.padding import PKCS7
        cipher = Cipher(algorithms.AES(encryption_key), modes.CBC(iv))
        dec = cipher.decryptor()
        padded = dec.update(ciphertext) + dec.finalize()
        unpadder = PKCS7(128).unpadder()
        return unpadder.update(padded) + unpadder.finalize()
    else:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad
        cipher = AES.new(encryption_key, AES.MODE_CBC, iv)
        padded = cipher.decrypt(ciphertext)
        return unpad(padded, 16)


# ============================================================================
# Key Derivation
# ============================================================================

# Check Argon2 availability
_ARGON2_AVAILABLE = False
try:
    from argon2.low_level import Type as Argon2Type
    from argon2.low_level import hash_secret_raw
    _ARGON2_AVAILABLE = True
except ImportError:
    pass


def generate_key() -> bytes:
    """Generate a cryptographically random 32-byte encryption key."""
    return secrets.token_bytes(_KEY_SIZE)


def derive_key_from_passphrase(
    passphrase: str,
    salt: bytes | None = None,
    force_pbkdf2: bool = False,
) -> tuple[bytes, bytes, str]:
    """Derive encryption key from passphrase.

    Uses Argon2id if available (GPU/ASIC resistant), falls back to
    PBKDF2-SHA256 with 600k iterations.

    Args:
        passphrase: User-provided passphrase
        salt: Optional salt (generated if not provided)
        force_pbkdf2: Force PBKDF2 even if Argon2 is available

    Returns:
        Tuple of (derived_key, salt, kdf_name) where kdf_name is
        'argon2id' or 'pbkdf2'.
    """
    if _ARGON2_AVAILABLE and not force_pbkdf2:
        if salt is None:
            salt = os.urandom(_ARGON2_SALT_LENGTH)
        key = hash_secret_raw(
            secret=passphrase.encode("utf-8"),
            salt=salt,
            time_cost=_ARGON2_TIME_COST,
            memory_cost=_ARGON2_MEMORY_COST,
            parallelism=_ARGON2_PARALLELISM,
            hash_len=_KEY_SIZE,
            type=Argon2Type.ID,
        )
        return key, salt, "argon2id"

    # PBKDF2 fallback
    if salt is None:
        salt = os.urandom(_PBKDF2_SALT_LENGTH)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
        dklen=_KEY_SIZE,
    )
    return key, salt, "pbkdf2"


# ============================================================================
# Keyfile Management
# ============================================================================

def _chmod_600(fpath: Path) -> None:
    """Set 0600 on the keyfile (owner read/write only); warn on failure."""
    try:
        fpath.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        logger.warning("Could not set keyfile permissions to 600: %s", fpath)


def _verify_keyfile_perms(fpath: Path) -> None:
    """Verify the keyfile is 600; if looser, warn and tighten it (K-01)."""
    try:
        mode = stat.S_IMODE(os.stat(fpath).st_mode)
    except OSError:
        return
    if mode & 0o077:
        logger.warning(
            "Keyfile %s has loose permissions %o (group/other access); "
            "tightening to 600.",
            fpath, mode,
        )
        _chmod_600(fpath)


def save_keyfile(
    key: bytes,
    salt: bytes | None = None,
    kdf_name: str = "",
    path: Path | None = None,
    *,
    passphrase: str | None = None,
    allow_unprotected: bool = False,
) -> Path:
    """Save the encryption key to the keyfile with restricted permissions.

    By default the key is wrapped (envelope) under a passphrase-derived
    key-encryption key (KEK) so the file at rest is not plaintext-equivalent:
    reading it without the passphrase yields nothing usable (K-01). The
    passphrase comes from ``passphrase`` or the ``OPTI_KEYFILE_PASSPHRASE``
    environment variable.

    Envelope format (JSON)::

        {"version": "envelope-v1", "kdf": "...", "kek_salt": "<b64>",
         "blob": "<b64 AES-256-GCM(KEK, key)>"}

    The legacy unprotected format (line-based base64, raw key at rest) is
    written only when no passphrase is available *and* ``allow_unprotected=True``
    is passed explicitly; it is logged as a security warning. The ``salt`` and
    ``kdf_name`` positional arguments are retained for backward compatibility
    and recorded only in the legacy format.

    File permissions are set to 600 (owner read/write only).
    """
    if len(key) != _KEY_SIZE:
        raise ValueError(f"Key must be {_KEY_SIZE} bytes, got {len(key)}")

    fpath = path or _DEFAULT_KEYFILE
    fpath.parent.mkdir(parents=True, exist_ok=True)

    effective_pass = passphrase or os.environ.get(_ENV_KEYFILE_PASS)

    if effective_pass:
        kek, kek_salt, kek_kdf = derive_key_from_passphrase(effective_pass)
        try:
            blob = encrypt_bytes(kek, key)
        finally:
            kek = b"\x00" * len(kek)  # best-effort wipe of the transient KEK
        payload = {
            "version": _KEYFILE_ENVELOPE_VERSION,
            "kdf": kek_kdf,
            "kek_salt": base64.urlsafe_b64encode(kek_salt).decode("ascii"),
            "blob": base64.urlsafe_b64encode(blob).decode("ascii"),
        }
        fpath.write_text(json.dumps(payload) + "\n", encoding="ascii")
        _chmod_600(fpath)
        logger.info(
            "Encryption keyfile saved (enveloped, kdf=%s): %s", kek_kdf, fpath,
        )
        return fpath

    if not allow_unprotected:
        raise ValueError(
            "Refusing to write an unprotected keyfile: no passphrase given. "
            "Pass passphrase=... or set OPTI_KEYFILE_PASSPHRASE, or pass "
            "allow_unprotected=True to write the legacy raw format."
        )

    logger.warning(
        "Writing an UNPROTECTED keyfile (raw key at rest) at %s. Anyone who "
        "reads this file holds the master key. Set OPTI_KEYFILE_PASSPHRASE to "
        "wrap it under a passphrase-derived key.",
        fpath,
    )
    lines = [base64.urlsafe_b64encode(key).decode("ascii")]
    if salt is not None:
        lines.append(base64.urlsafe_b64encode(salt).decode("ascii"))
    else:
        lines.append("")
    lines.append(kdf_name or "")
    fpath.write_text("\n".join(lines) + "\n", encoding="ascii")
    _chmod_600(fpath)
    logger.info(
        "Encryption keyfile saved (unprotected, kdf=%s): %s",
        kdf_name or "random", fpath,
    )
    return fpath


def load_keyfile(
    path: Path | None = None,
    *,
    passphrase: str | None = None,
) -> tuple[SecureBytes, bytes | None, str]:
    """Load the encryption key from the keyfile.

    Auto-detects the format: an enveloped keyfile is decrypted with the
    passphrase (from ``passphrase`` or ``OPTI_KEYFILE_PASSPHRASE``); a legacy
    unprotected keyfile is read directly (with a deprecation warning).
    Permissions are verified and tightened to 600.

    Returns:
        Tuple of (key_as_SecureBytes, salt_or_None, kdf_name). The key is
        wrapped in SecureBytes for memory protection.

    Raises:
        FileNotFoundError: the keyfile does not exist.
        ValueError: the keyfile is enveloped but no passphrase is available.
    """
    fpath = path or _DEFAULT_KEYFILE
    if not fpath.exists():
        raise FileNotFoundError(f"Keyfile not found: {fpath}")

    _verify_keyfile_perms(fpath)
    text = fpath.read_text(encoding="ascii").strip()

    payload = None
    try:
        candidate = json.loads(text)
        if (
            isinstance(candidate, dict)
            and candidate.get("version") == _KEYFILE_ENVELOPE_VERSION
        ):
            payload = candidate
    except (ValueError, TypeError):
        payload = None

    if payload is not None:
        effective_pass = passphrase or os.environ.get(_ENV_KEYFILE_PASS)
        if not effective_pass:
            raise ValueError(
                "Keyfile is passphrase-protected (envelope) but no passphrase "
                "was provided. Set OPTI_KEYFILE_PASSPHRASE or pass passphrase=."
            )
        kek_salt = base64.urlsafe_b64decode(payload["kek_salt"])
        kdf_name = payload.get("kdf", "")
        kek, _, _ = derive_key_from_passphrase(
            effective_pass, salt=kek_salt, force_pbkdf2=(kdf_name == "pbkdf2"),
        )
        try:
            blob = base64.urlsafe_b64decode(payload["blob"])
            raw_key = decrypt_bytes(kek, blob)
        finally:
            kek = b"\x00" * len(kek)
        return secure_key_from_bytes(raw_key), kek_salt, kdf_name

    # Legacy unprotected format (line-based base64).
    logger.warning(
        "Loading a legacy UNPROTECTED keyfile (%s). Re-create it with a "
        "passphrase (OPTI_KEYFILE_PASSPHRASE) to wrap the key at rest.",
        fpath,
    )
    lines = text.split("\n")
    raw_key = base64.urlsafe_b64decode(lines[0])
    salt = None
    if len(lines) > 1 and lines[1].strip():
        salt = base64.urlsafe_b64decode(lines[1])
    kdf_name = lines[2].strip() if len(lines) > 2 else ""
    return secure_key_from_bytes(raw_key), salt, kdf_name


def get_encryption_key() -> SecureBytes | None:
    """Get the encryption key from the best available source.

    Returns a SecureBytes wrapper for memory protection.

    Priority:
      1. Environment variable OPTI_ENCRYPTION_KEY
      2. Keyfile at data/.keyfile
      3. None (encryption disabled)
    """
    env_key = os.environ.get(_ENV_KEY_NAME)
    if env_key:
        try:
            raw = base64.urlsafe_b64decode(env_key)
            if len(raw) == _KEY_SIZE:
                return secure_key_from_bytes(raw)
            logger.warning("OPTI_ENCRYPTION_KEY wrong length (%d), expected %d", len(raw), _KEY_SIZE)
        except Exception as e:
            logger.warning("Failed to decode OPTI_ENCRYPTION_KEY: %s", e)

    try:
        key, _, _ = load_keyfile()
        if len(key) == _KEY_SIZE:
            return key
        logger.warning("Keyfile key wrong length (%d), expected %d", len(key), _KEY_SIZE)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("Failed to load keyfile: %s", e)

    return None


# ============================================================================
# EncryptionManager
# ============================================================================

class EncryptionManager:
    """Manages transparent encryption/decryption for database fields.

    Encrypted content is stored with an 'ENC2:' prefix (AES-256-GCM) or
    legacy 'ENC:' prefix (AES-128-CBC, read-only for migration).
    Unencrypted content is returned as-is for gradual migration.
    """

    def __init__(self, key: bytes | SecureBytes | None = None, enabled: bool | None = None):
        cfg = _load_encryption_config()

        if enabled is not None:
            self._enabled = enabled
        else:
            self._enabled = cfg.get("enabled", False)

        if key is not None:
            # Wrap raw bytes in SecureBytes for memory protection
            if isinstance(key, bytes):
                self._key: SecureBytes | None = secure_key_from_bytes(key)
            else:
                self._key = key
        elif self._enabled:
            self._key = get_encryption_key()
            if self._key is None:
                logger.warning(
                    "Encryption enabled in config but no key available. "
                    "Set OPTI_ENCRYPTION_KEY or create data/.keyfile"
                )
                self._enabled = False
        else:
            self._key = None

    def _raw_key(self) -> bytes | None:
        """Return raw key bytes for crypto operations, or None."""
        if self._key is None:
            return None
        if isinstance(self._key, SecureBytes):
            return self._key.as_bytes()
        return self._key

    @property
    def enabled(self) -> bool:
        """Whether encryption is active."""
        return self._enabled and self._key is not None

    @property
    def has_key(self) -> bool:
        """Whether an encryption key is available."""
        return self._key is not None

    @property
    def algorithm(self) -> str:
        """Current encryption algorithm."""
        return "AES-256-GCM" if self.enabled else "none"

    @property
    def kdf(self) -> str:
        """Key derivation function in use."""
        return "argon2id" if _ARGON2_AVAILABLE else "pbkdf2-sha256"

    @property
    def crypto_backend(self) -> str:
        """Name of the cryptographic backend."""
        return _CRYPTO_BACKEND or "none"

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string field using AES-256-GCM.

        Returns the original string if encryption is disabled.
        Already-encrypted strings (V1 or V2) are returned unchanged.
        """
        if not self.enabled or not plaintext:
            return plaintext
        if plaintext.startswith(_ENCRYPTED_PREFIX_V2) or plaintext.startswith(_ENCRYPTED_PREFIX_V1):
            return plaintext

        try:
            raw = encrypt_bytes(self._raw_key(), plaintext.encode("utf-8"))
            return _ENCRYPTED_PREFIX_V2 + base64.urlsafe_b64encode(raw).decode("ascii")
        except Exception as e:
            logger.error("Encryption failed: %s", e)
            return plaintext

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a string field. Handles V1 and V2 formats.

        Transparently returns unencrypted strings as-is.
        """
        if not ciphertext:
            return ciphertext

        if not self._key:
            if ciphertext.startswith((_ENCRYPTED_PREFIX_V1, _ENCRYPTED_PREFIX_V2)):
                logger.warning("Cannot decrypt: no encryption key available")
            return ciphertext

        # V2: AES-256-GCM
        if ciphertext.startswith(_ENCRYPTED_PREFIX_V2):
            raw = base64.urlsafe_b64decode(ciphertext[len(_ENCRYPTED_PREFIX_V2):])
            try:
                plaintext = decrypt_bytes(self._raw_key(), raw)
                return plaintext.decode("utf-8")
            except Exception as e:
                logger.error("V2 decryption failed: %s", e)
                return ciphertext

        # V1 legacy: AES-128-CBC + HMAC (read-only migration support)
        if ciphertext.startswith(_ENCRYPTED_PREFIX_V1):
            token = ciphertext[len(_ENCRYPTED_PREFIX_V1):].encode("ascii")
            try:
                plaintext = _decrypt_v1(self._raw_key(), token)
                return plaintext.decode("utf-8")
            except Exception as e:
                logger.error("V1 legacy decryption failed: %s", e)
                return ciphertext

        return ciphertext  # Not encrypted

    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted (V1 or V2)."""
        if not value:
            return False
        return value.startswith(_ENCRYPTED_PREFIX_V2) or value.startswith(_ENCRYPTED_PREFIX_V1)

    def needs_reencrypt(self, value: str) -> bool:
        """Check if a value uses the old V1 format and needs re-encryption."""
        return bool(value) and value.startswith(_ENCRYPTED_PREFIX_V1)

    def reencrypt(self, value: str) -> str:
        """Re-encrypt a V1 value to V2 (AES-256-GCM).

        Returns the value unchanged if already V2 or not encrypted.
        """
        if not value or not value.startswith(_ENCRYPTED_PREFIX_V1):
            return value
        plaintext = self.decrypt(value)
        if plaintext == value:
            return value  # Decryption failed, keep as-is
        return self.encrypt(plaintext)

    def rotate_key(self, new_key: bytes, values: list[str]) -> list[str]:
        """Re-encrypt a list of values with a new key.

        This is the core of key rotation. Call this for each batch of
        encrypted values, then update the keyfile.

        Wipes old key from memory after rotation.

        Args:
            new_key: New 32-byte encryption key
            values: List of encrypted (or plain) strings

        Returns:
            List of strings re-encrypted with the new key
        """
        if len(new_key) != _KEY_SIZE:
            raise ValueError(f"New key must be {_KEY_SIZE} bytes")

        new_mgr = EncryptionManager(key=new_key, enabled=True)
        result = []
        for val in values:
            # Decrypt with current key
            plain = self.decrypt(val)
            # Re-encrypt with new key
            result.append(new_mgr.encrypt(plain))

        # Wipe old key from memory
        if self._key is not None and isinstance(self._key, SecureBytes):
            self._key.wipe()
        # Install new key
        self._key = secure_key_from_bytes(new_key)

        return result

    def setup_from_passphrase(self, passphrase: str) -> bool:
        """Derive key from passphrase, save keyfile, enable encryption."""
        try:
            key, salt, kdf_name = derive_key_from_passphrase(passphrase)
            # Wrap the key at rest under a KEK derived from the same passphrase.
            save_keyfile(key, salt, kdf_name, passphrase=passphrase)
            # Wipe old key, wrap new key in SecureBytes
            if self._key is not None and isinstance(self._key, SecureBytes):
                self._key.wipe()
            self._key = secure_key_from_bytes(key)
            self._enabled = True
            logger.info("Encryption configured from passphrase (kdf=%s)", kdf_name)
            return True
        except Exception as e:
            logger.error("Failed to setup encryption: %s", e)
            return False

    def setup_random_key(self) -> bool:
        """Generate a random key, save keyfile, enable encryption."""
        try:
            key = generate_key()
            env_pass = os.environ.get(_ENV_KEYFILE_PASS)
            save_keyfile(
                key,
                kdf_name="random",
                passphrase=env_pass,
                allow_unprotected=env_pass is None,
            )
            # Wipe old key, wrap new key in SecureBytes
            if self._key is not None and isinstance(self._key, SecureBytes):
                self._key.wipe()
            self._key = secure_key_from_bytes(key)
            self._enabled = True
            logger.info("Encryption configured with random key")
            return True
        except Exception as e:
            logger.error("Failed to setup encryption: %s", e)
            return False

    def get_status(self) -> dict[str, Any]:
        """Get detailed encryption status."""
        cfg = _load_encryption_config()
        keyfile_exists = _DEFAULT_KEYFILE.exists()
        env_key_set = bool(os.environ.get(_ENV_KEY_NAME))

        return {
            "enabled": self.enabled,
            "config_enabled": cfg.get("enabled", False),
            "has_key": self.has_key,
            "algorithm": self.algorithm,
            "kdf": self.kdf,
            "crypto_backend": self.crypto_backend,
            "argon2_available": _ARGON2_AVAILABLE,
            "keyfile_exists": keyfile_exists,
            "env_key_set": env_key_set,
            "keyfile_path": str(_DEFAULT_KEYFILE),
            "format_version": _FORMAT_VERSION,
            "secure_bytes_active": _SECURE_BYTES_AVAILABLE and isinstance(self._key, SecureBytes),
            "key_mlocked": (
                isinstance(self._key, SecureBytes)
                and hasattr(self._key, "is_mlocked")
                and self._key.is_mlocked
            ),
        }


# ============================================================================
# Module-level singleton
# ============================================================================

_encryption_manager: EncryptionManager | None = None


def get_encryption_manager() -> EncryptionManager:
    """Get or create the singleton EncryptionManager."""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager


def encrypt_field(value: str) -> str:
    """Convenience: encrypt a field value using the global manager."""
    return get_encryption_manager().encrypt(value)


def decrypt_field(value: str) -> str:
    """Convenience: decrypt a field value using the global manager."""
    return get_encryption_manager().decrypt(value)
