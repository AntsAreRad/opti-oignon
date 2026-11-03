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
  - NO field in clear by mistake: encryption on and no key that loads, or a
    security.yaml that cannot say whether it is on, refuses every field
    write (EncryptionUnavailable), as a cipher error does; the manager
    follows security.yaml and the key file while the platform runs

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
import contextlib
import hashlib
import hmac
import json
import logging
import os
import secrets
import stat
import struct
import threading
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


def _security_path() -> Path:
    """The security.yaml the manager reads."""
    return Path(__file__).parent / "config" / "security.yaml"


def _encryption_section(path: str | Path | None = None) -> dict[str, Any] | None:
    """security.yaml's (or ``path``'s) ``encryption`` section, read strictly.

    {} for a missing file, or a file with no ``encryption`` section; the
    section for a mapping; None when the file cannot say -- no regular file (a
    FIFO is never waited on), empty (a save caught between its truncation and
    its write), unreadable, unparsable, not a mapping, or a section that is not
    one.
    """
    import yaml
    cfg_path = Path(path) if path is not None else _security_path()
    try:
        if not stat.S_ISREG(os.stat(cfg_path).st_mode):
            return None
        text = cfg_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except Exception:
        return None
    try:
        data = yaml.safe_load(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    section = data.get("encryption")
    if section is None:
        return {}
    if not isinstance(section, dict):
        return None
    return section


def _load_encryption_config() -> dict[str, Any] | None:
    """The ``encryption`` section of security.yaml as the manager reads it:
    strictly (``_encryption_section``), None when the file cannot say --
    never read as off."""
    return _encryption_section()


def encryption_toggle(path: str | Path | None = None) -> bool | None:
    """Whether security.yaml (or ``path``) turns encryption on, read strictly.

    False for a missing file, or a file with no ``encryption`` section or no
    switch in it; the switch's truth for a mapping; None when the file cannot
    say -- empty (a save caught between its truncation and its write),
    unparsable, not a mapping, or a section that is not one. A store that
    must never write in clear by mistake reads None as an unknown, never as
    off; the manager reads the same section (``_load_encryption_config``).
    """
    section = _encryption_section(path)
    return None if section is None else bool(section.get("enabled", False))


class EncryptionUnavailable(RuntimeError):
    """A field cannot be sealed: encryption is on and no key loads, security.yaml
    cannot say whether it is on, or the cipher failed. Raised where the value
    would otherwise be written in clear; its message is the reason, said as it
    is."""


_CANNOT_SAY = ("security.yaml cannot say whether encryption is on (it does not load, or its encryption section "
               "holds no mapping): nothing is written in clear until it is mended")
_NO_KEY = ("encryption is on and no key loads (OPTI_ENCRYPTION_KEY, or the key file; a wrapped key file needs "
           "OPTI_KEYFILE_PASSPHRASE): nothing is written in clear until a key loads -- a key file put back is read "
           "at the next write, a variable when the platform restarts")
_KEY_GONE = ("encryption is on and the key file the key in use came from is gone: nothing is written in clear "
             "until a key file is put back")
_KEY_CHANGED = ("encryption is on and the key file holds another key than the one in use: nothing is written in "
                "clear until the platform restarts")
_NO_CIPHER = ("encryption is on and no AES library is installed (cryptography or pycryptodome): nothing is written "
              "in clear")


def _seen_state() -> tuple:
    """What a manager built from the files follows: security.yaml's and the key
    file's identity (device, inode, size, modification and change times), None
    for one that cannot be looked at."""
    marks = []
    for path in (_security_path(), _DEFAULT_KEYFILE):
        try:
            found = os.stat(path)
        except OSError:
            marks.append(None)
        else:
            marks.append((found.st_dev, found.st_ino, found.st_size, found.st_mtime_ns, found.st_ctime_ns))
    return tuple(marks)


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

def _secret(name: str) -> str | None:
    """The platform's secret ``name`` as this process holds it: the environment's, else the value taken out of
    it at start (``opti_oignon.process_secrets``); raises ``SecretWiped`` once a held value is wiped."""
    from opti_oignon.process_secrets import secret

    return secret(name)


def _wiped(exc: BaseException) -> bool:
    """Whether ``exc`` says a held secret was wiped: a third state, never read as "no key". Asked by its class's
    name, so that a caller never imports the holder to ask."""
    return type(exc).__name__ == "SecretWiped" and type(exc).__module__ == "opti_oignon.process_secrets"


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


class KeyfileExists(RuntimeError):
    """A key file is already at this path: setup never replaces one."""


class PassphraseRefused(ValueError):
    """The key file's envelope did not open under the passphrase given: its blob, of the right length and version,
    failed its authentication under the key the passphrase derives by the derivation the envelope names. A
    ``ValueError``, as it always was, with the same message. A damage the format shows -- a length, a version, a
    derivation unknown or that cannot run here -- is another ``ValueError``; a byte changed for another in the
    salt, or in the blob past its version byte -- its nonce, the wrapped key, its tag --, fails the authentication
    as a wrong passphrase does, and cannot be told from one."""


# The derivations an envelope may name, with the length of the salt each writes, and the length of the blob that
# wraps a key (version, nonce, the 32-byte key, tag): what the format lets a damaged envelope be told by.
_ENVELOPE_KDFS = {"argon2id": _ARGON2_SALT_LENGTH, "pbkdf2": _PBKDF2_SALT_LENGTH}
_ENVELOPE_BLOB = 1 + _GCM_NONCE_SIZE + _KEY_SIZE + 16


def keyfile_present(path: Path | None = None) -> bool:
    """Whether anything is at the key file's path (default: the data key file).

    A symbolic link is not followed: a dangling one is present, since writing
    through it would put the key wherever it points. Setup writes no key file
    where this answers True.
    """
    return os.path.lexists(path or _DEFAULT_KEYFILE)


def _envelope(text: str) -> dict | None:
    """The key file's envelope when ``text`` is one, or None: the legacy unprotected format."""
    try:
        candidate = json.loads(text)
    except (ValueError, TypeError):
        return None
    if isinstance(candidate, dict) and candidate.get("version") == _KEYFILE_ENVELOPE_VERSION:
        return candidate
    return None


def _legacy_key(text: str) -> bool:
    """Whether ``text`` is a key file in the legacy unprotected format, read as ``load_keyfile`` reads one: its first
    line decodes to bytes. A text that opens an object is an envelope that does not parse, never this format."""
    if text.startswith("{"):
        return False
    try:
        return bool(base64.urlsafe_b64decode(text.split("\n")[0]))
    except ValueError:
        return False


def keyfile_under(root: Path) -> Path:
    """The key file of the platform whose project root is ``root``, where its loader reads it."""
    return Path(root) / "data" / _DEFAULT_KEYFILE.name


def keyfile_wrapped(path: Path | None = None) -> bool:
    """Whether the key file (default: the data key file) is wrapped under a passphrase, told as ``load_keyfile``
    tells it.

    False when nothing is at its path, a symbolic link included, or it is in the legacy unprotected format. A key
    file that is there and cannot be read raises (``OSError``, ``UnicodeError``), and so does one that is neither
    an envelope nor a key in the legacy format (``ValueError``): whoever asks cannot tell, and says so.
    """
    fpath = path or _DEFAULT_KEYFILE
    if not os.path.lexists(fpath):
        return False
    text = fpath.read_text(encoding="ascii").strip()
    if _envelope(text) is not None:
        return True
    if not _legacy_key(text):
        raise ValueError("the key file is neither wrapped under a passphrase nor a key in the unprotected format")
    return False


def _write_keyfile(fpath: Path, data: bytes) -> None:
    """Create the key file ``fpath`` holding ``data``; never replace one.

    The bytes go to a temp file beside it, created with mode 0600 by
    ``O_CREAT | O_EXCL | O_NOFOLLOW`` and synced, which is then hard-linked
    to ``fpath`` and unlinked, and the directory is synced. The link fails
    when anything is already at ``fpath``, so a key file another writer put
    in place after the existence check is refused, not replaced, and the key
    file never exists partly written or with looser permissions. A failure
    before the link removes the temp and writes nothing; there is no other
    way in, so a filesystem without hard links gets no key file.

    Once the link has succeeded the key file is whole and in place: a temp
    that cannot be removed (it still holds the key) or a directory that
    cannot be synced is logged as a warning naming it, never raised. A
    process killed between the temp's creation and its link leaves the temp,
    mode 0600, holding a key that never became the key file; nothing sweeps
    it, since a sweep could remove another writer's live temp.

    Raises ``KeyfileExists`` when anything is at ``fpath``.
    """
    parent = fpath.parent
    parent.mkdir(parents=True, exist_ok=True)
    if keyfile_present(fpath):
        raise KeyfileExists(str(fpath))
    temp = parent / ("." + fpath.name + ".tmp-" + os.urandom(4).hex())
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    handle = os.open(temp, flags, 0o600)
    try:
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(handle, view):]
            os.fsync(handle)
        finally:
            os.close(handle)
        try:
            os.link(temp, fpath)
        except FileExistsError:
            raise KeyfileExists(str(fpath)) from None
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise
    try:
        os.unlink(temp)
    except OSError as exc:
        logger.warning(
            "Key file %s is in place, but its temp %s could not be removed "
            "and still holds the key: %s", fpath, temp, exc,
        )
    try:
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        logger.warning(
            "Key file %s is in place, but its directory %s could not be "
            "synced: %s", fpath, parent, exc,
        )


@contextlib.contextmanager
def _mode_kept(key: bytes, fpath: Path):
    """The security mode's lockfile signed under ``key`` before the key file
    ``fpath`` is written, and put back as it was when it is not
    (``security_mode.signed_for_a_new_key``): making the key file never
    changes the mode the platform reads. A mode module that does not import,
    or keeps no lockfile (a stand-in), has no mode to keep.
    """
    try:
        from opti_oignon import security_mode
    except ImportError:
        security_mode = None
    keep = getattr(security_mode, "signed_for_a_new_key", None)
    if keep is None:
        yield
        return
    with keep(key, fpath):
        yield


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

    The file is created with mode 0600 (owner read/write only) and written
    whole, and an existing key file is never replaced: ``KeyfileExists``
    (see ``_write_keyfile``). The security mode reads the same mode before
    and after: its lockfile is signed under the key first (``_mode_kept``).
    """
    if len(key) != _KEY_SIZE:
        raise ValueError(f"Key must be {_KEY_SIZE} bytes, got {len(key)}")

    fpath = path or _DEFAULT_KEYFILE

    effective_pass = passphrase or _secret(_ENV_KEYFILE_PASS)

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
        with _mode_kept(key, fpath):
            _write_keyfile(fpath, (json.dumps(payload) + "\n").encode("ascii"))
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
    with _mode_kept(key, fpath):
        _write_keyfile(fpath, ("\n".join(lines) + "\n").encode("ascii"))
    logger.info(
        "Encryption keyfile saved (unprotected, kdf=%s): %s",
        kdf_name or "random", fpath,
    )
    return fpath


# The key file this process last unwrapped with the passphrase it holds (``OPTI_KEYFILE_PASSPHRASE``): the digest
# of its text, that passphrase -- held as a key is held --, and the key, or the refusal of a passphrase the envelope
# refused. The derivation (Argon2id over 64 MiB, or PBKDF2) then runs once for each text and passphrase, not at every
# load: a mode read, an approval, a status asked on the event loop would each stop every request for it. One text at
# a time: the slot a new one takes is wiped, so a key file replaced leaves neither its key nor its passphrase here. A
# passphrase given as an argument is never kept, nor what it opened; no digest of a passphrase is kept -- one would
# test guesses at a hash's speed --; a slot wiped (every key is, at exit and on a signal) is unwrapped again.
_unwrapped_slot: tuple[bytes, Any, Any] | None = None
_unwrapped_lock = threading.Lock()


def _unwrapped(text_digest: bytes, passphrase: str) -> Any:
    """The key or the refusal kept for this text and passphrase, or None. Compared and copied under the lock the slot
    is replaced under: a load never copies a key the replacement is zeroing."""
    with _unwrapped_lock:
        slot = _unwrapped_slot
        if slot is None or slot[0] != text_digest:
            return None
        _digest, held, outcome = slot
        try:
            if not hmac.compare_digest(held.as_bytes(), passphrase.encode("utf-8")):
                return None
            if isinstance(outcome, str):
                return outcome
            raw = outcome.as_bytes()
        except RuntimeError:  # wiped
            return None
    # A key every key's wipe (at exit, on a signal) zeroed as it was copied is no key: unwrapped again.
    return secure_key_from_bytes(raw) if any(raw) else None


def _remember_unwrapped(text_digest: bytes, passphrase: str, outcome: Any) -> None:
    global _unwrapped_slot
    slot = (text_digest, SecureBytes(bytearray(passphrase.encode("utf-8"))), outcome)
    with _unwrapped_lock:
        replaced, _unwrapped_slot = _unwrapped_slot, slot
    for held in replaced[1:] if replaced is not None else ():
        if not isinstance(held, str):
            held.wipe()


def load_keyfile(
    path: Path | None = None,
    *,
    passphrase: str | None = None,
) -> tuple[SecureBytes, bytes | None, str]:
    """Load the encryption key from the keyfile.

    Auto-detects the format: an enveloped keyfile is decrypted with the
    passphrase (from ``passphrase`` or ``OPTI_KEYFILE_PASSPHRASE``); a legacy
    unprotected keyfile is read directly (with a deprecation warning).
    Permissions are verified and tightened to 600, and the file read, at every
    call; an envelope opened with the passphrase the process holds is
    unwrapped once per process for each text and passphrase
    (``_unwrapped_slot``), each call given a key of its own.

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

    payload = _envelope(text)

    if payload is not None:
        effective_pass = passphrase or _secret(_ENV_KEYFILE_PASS)
        if not effective_pass:
            raise ValueError(
                "Keyfile is passphrase-protected (envelope) but no passphrase "
                "was provided. Set OPTI_KEYFILE_PASSPHRASE or pass passphrase=."
            )
        kek_salt = base64.urlsafe_b64decode(payload["kek_salt"])
        kdf_name = payload.get("kdf", "")
        # The derivation the envelope names, or none: a key derived another way opens nothing, and would be
        # said a wrong passphrase.
        if kdf_name not in _ENVELOPE_KDFS:
            raise ValueError(f"Keyfile envelope names a key derivation this version does not know: {kdf_name!r}")
        if kdf_name == "argon2id" and not _ARGON2_AVAILABLE:
            raise ValueError("Keyfile is wrapped with argon2id, and argon2-cffi does not import here")
        if len(kek_salt) != _ENVELOPE_KDFS[kdf_name]:
            raise ValueError("Keyfile envelope is damaged: its salt is not of the length its derivation writes")
        blob = base64.urlsafe_b64decode(payload["blob"])
        if len(blob) != _ENVELOPE_BLOB or blob[0] != _FORMAT_VERSION:
            raise ValueError("Keyfile envelope is damaged: its blob is not the wrapped key of this format")
        # Only the passphrase this process holds is kept (``_unwrapped_slot``); one given here never is.
        kept = not passphrase
        digest = hashlib.sha256(text.encode("ascii")).digest()
        known = _unwrapped(digest, effective_pass) if kept else None
        if isinstance(known, str):
            raise PassphraseRefused(known)
        if known is not None:
            return known, kek_salt, kdf_name
        kek, _, _ = derive_key_from_passphrase(
            effective_pass, salt=kek_salt, force_pbkdf2=(kdf_name == "pbkdf2"),
        )
        try:
            try:
                raw_key = decrypt_bytes(kek, blob)
            except ValueError as exc:
                if kept:
                    _remember_unwrapped(digest, effective_pass, str(exc))
                raise PassphraseRefused(str(exc)) from exc
        finally:
            kek = b"\x00" * len(kek)
        # The source of a SecureBytes is zeroed in place: the key kept is made from the key given, never from the
        # bytes the given one was made from.
        key = secure_key_from_bytes(raw_key)
        if kept:
            _remember_unwrapped(digest, effective_pass, secure_key_from_bytes(key.as_bytes()))
        return key, kek_salt, kdf_name

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
      3. None (no key: a manager that is on refuses every write)
    """
    env_key = _secret(_ENV_KEY_NAME)
    if env_key:
        try:
            raw = base64.urlsafe_b64decode(env_key)
            if len(raw) == _KEY_SIZE:
                return secure_key_from_bytes(raw)
            logger.warning("OPTI_ENCRYPTION_KEY wrong length (%d), expected %d", len(raw), _KEY_SIZE)
        except Exception as e:
            logger.warning("Failed to decode OPTI_ENCRYPTION_KEY: %s", e)

    try:
        if not stat.S_ISREG(os.stat(_DEFAULT_KEYFILE).st_mode):
            # A FIFO is never waited on, a device never read.
            raise ValueError("the key file is no regular file: not opened")
        key, _, _ = load_keyfile()
        if len(key) == _KEY_SIZE:
            return key
        logger.warning("Keyfile key wrong length (%d), expected %d", len(key), _KEY_SIZE)
    except FileNotFoundError:
        pass
    except Exception as e:
        if _wiped(e):
            raise
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

    Three states: on (the switch on, or ``enabled`` given True, and a key),
    off (the switch off, or ``enabled`` given False), and refusing
    (``refusing`` the reason): the switch on, or given True, and no key loads
    or no AES library is installed, security.yaml cannot say whether it is
    on, or the cipher failed. Refusing, ``encrypt`` raises
    ``EncryptionUnavailable`` and nothing is written in clear. A manager built
    from the files (no ``enabled`` given) follows them (``follow``): the
    switch turned on or off, the file mended, a key file put back; the setup
    methods leave the refusing state.
    """

    def __init__(self, key: bytes | SecureBytes | None = None, enabled: bool | None = None):
        self._refusing: str | None = None
        self._enabled = False
        self._follows = enabled is None
        self._following = threading.Lock()
        self._given = key is not None
        # Looked at before the switch is read: a change while it is read is followed next.
        self._seen = _seen_state() if self._follows else None
        if key is not None:
            # Wrap raw bytes in SecureBytes for memory protection
            if isinstance(key, bytes):
                self._key: SecureBytes | None = secure_key_from_bytes(key)
            else:
                self._key = key
        else:
            self._key = None
        self._decide(_load_encryption_config() if enabled is None else {"enabled": enabled})
        if self._refusing:
            logger.error("Field encryption refuses every write: %s", self._refusing)

    def _decide(self, cfg: dict[str, Any] | None) -> None:
        """The state from the switch ``cfg`` (None: the file cannot say) and the
        key held, a key loaded when none is held and the switch is on or cannot
        say. The reason is set before the switch goes off, and cleared after
        the key and the switch are in."""
        if cfg is None:
            self._refusing = _CANNOT_SAY
            self._enabled = False
            if self._key is None:
                # A key that loads still reads what is sealed; nothing is sealed with it.
                self._key = get_encryption_key()
            return
        if not cfg.get("enabled", False):
            self._enabled = False
            self._refusing = None
            return
        if self._key is None:
            self._key = get_encryption_key()
        if self._key is None or _CRYPTO_BACKEND is None:
            self._refusing = _NO_KEY if self._key is None else _NO_CIPHER
            self._enabled = False
            return
        failed = self._cipher_fails()
        if failed:
            self._refusing = failed
            self._enabled = False
            return
        self._enabled = True
        self._refusing = None

    def _cipher_fails(self) -> str | None:
        """Why the cipher does not seal a fixed value under the held key, or None."""
        try:
            encrypt_bytes(self._raw_key(), b"a fixed value the manager seals to try its cipher")
        except Exception as e:
            return f"the cipher failed ({type(e).__name__}): nothing is written in clear"
        return None

    def _check_held_key(self) -> None:
        """Refuse when the key held came from the key file (no key variable, none
        given) and the key file is gone, or holds another key now: rows sealed
        under it would not open at the next start. A key file that does not
        load here (wrapped, its passphrase not given) leaves the held key be."""
        if self._key is None or self._given or _secret(_ENV_KEY_NAME):
            return
        try:
            if not stat.S_ISREG(os.stat(_DEFAULT_KEYFILE).st_mode):
                raise ValueError("the key file is no regular file: not opened")
            on_disk, _, _ = load_keyfile()
        except FileNotFoundError:
            self._refusing = _KEY_GONE
            self._enabled = False
            self._key = None
            return
        except Exception as e:
            if _wiped(e):
                raise
            return
        if on_disk.as_bytes() != self._raw_key():
            self._refusing = _KEY_CHANGED
            self._enabled = False

    def follow(self) -> None:
        """Read security.yaml and the key file again when either changed since
        they were last read, for a manager built from them (one told its state
        never follows): the switch turned on or off, the file mended, a key file
        put back where none was held. A held key is kept."""
        if not self._follows or _seen_state() == self._seen:
            return
        with self._following:
            seen = _seen_state()
            if seen == self._seen:
                return
            before = self._refusing
            self._decide(_load_encryption_config())
            if self._enabled:
                self._check_held_key()
            self._seen = seen
            if self._refusing and self._refusing != before:
                logger.error("Field encryption refuses every write: %s", self._refusing)

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
    def refusing(self) -> str | None:
        """Why the manager refuses to seal, or None: encryption on and no key that
        loads, no AES library, the key file gone or holding another key, a
        cipher that fails, or security.yaml unable to say whether it is on."""
        return self._refusing

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

        Returns the original string if encryption is off. Refusing
        (``refusing``), or the cipher failing, raises ``EncryptionUnavailable``:
        never the string in clear. An empty string, and one already encrypted
        (V1 or V2), are returned unchanged. A string that is no text (a lone
        surrogate) raises its ``UnicodeEncodeError``, as it did; the manager
        refuses nothing more for it.
        """
        if not plaintext:
            return plaintext
        if plaintext.startswith(_ENCRYPTED_PREFIX_V2) or plaintext.startswith(_ENCRYPTED_PREFIX_V1):
            return plaintext
        # Read before the switch: a setup clears it last, once the key is in.
        refusing = self._refusing
        if refusing:
            raise EncryptionUnavailable(refusing)
        if not self.enabled:
            return plaintext

        data = plaintext.encode("utf-8")
        try:
            raw = encrypt_bytes(self._raw_key(), data)
        except Exception as e:
            logger.error("Encryption failed: %s", e)
            kept = self._cipher_fails()
            if kept:
                # Kept while a fixed value does not seal either: the next write, and whoever asks before it, is told.
                self._refusing = kept
            raise EncryptionUnavailable(
                f"the cipher failed ({type(e).__name__}): nothing is written in clear") from e
        return _ENCRYPTED_PREFIX_V2 + base64.urlsafe_b64encode(raw).decode("ascii")

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

    def enable_with_existing_key(self) -> bool:
        """Enable encryption with the key that already loads; write no key file.

        The key is the one ``get_encryption_key`` answers (OPTI_ENCRYPTION_KEY,
        then the key file). Answers False, and installs nothing, when none
        loads. Loading a key file tightens its permissions to 0600 when they
        are looser, as every load does.
        """
        key = get_encryption_key()
        if key is None:
            return False
        if self._key is not None and isinstance(self._key, SecureBytes):
            self._key.wipe()
        self._key = key
        self._enabled = True
        self._refusing = None  # last: the key and the switch are in
        # The key file's rights it tightened are not a change to follow.
        self._seen = _seen_state() if self._follows else None
        logger.info("Encryption enabled with the existing key")
        return True

    def setup_from_passphrase(self, passphrase: str) -> bool:
        """Derive key from passphrase, save keyfile, enable encryption.

        Raises ``KeyfileExists`` when a key file is already there.
        """
        try:
            key, salt, kdf_name = derive_key_from_passphrase(passphrase)
            # Wrap the key at rest under a KEK derived from the same passphrase.
            save_keyfile(key, salt, kdf_name, passphrase=passphrase)
            # Wipe old key, wrap new key in SecureBytes
            if self._key is not None and isinstance(self._key, SecureBytes):
                self._key.wipe()
            self._key = secure_key_from_bytes(key)
            self._enabled = True
            self._refusing = None  # last: the key and the switch are in
            # The key file it wrote is not a change to follow.
            self._seen = _seen_state() if self._follows else None
            logger.info("Encryption configured from passphrase (kdf=%s)", kdf_name)
            return True
        except KeyfileExists:
            raise
        except Exception as e:
            logger.error("Failed to setup encryption: %s", e)
            return False

    def setup_random_key(self) -> bool:
        """Generate a random key, save keyfile, enable encryption.

        Raises ``KeyfileExists`` when a key file is already there.
        """
        try:
            key = generate_key()
            env_pass = _secret(_ENV_KEYFILE_PASS)
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
            self._refusing = None  # last: the key and the switch are in
            # The key file it wrote is not a change to follow.
            self._seen = _seen_state() if self._follows else None
            logger.info("Encryption configured with random key")
            return True
        except KeyfileExists:
            raise
        except Exception as e:
            logger.error("Failed to setup encryption: %s", e)
            return False

    def get_status(self) -> dict[str, Any]:
        """Get detailed encryption status: ``refusing`` says why every write is
        refused, or None; ``config_enabled`` is the switch read strictly (False
        when security.yaml cannot say)."""
        cfg = _load_encryption_config()
        keyfile_exists = _DEFAULT_KEYFILE.exists()
        env_key_set = bool(_secret(_ENV_KEY_NAME))

        return {
            "enabled": self.enabled,
            "config_enabled": bool(cfg and cfg.get("enabled", False)),
            "refusing": self.refusing,
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
    else:
        _encryption_manager.follow()
    return _encryption_manager


def encrypt_field(value: str) -> str:
    """Convenience: encrypt a field value using the global manager; a manager
    that cannot be built (a held secret wiped) refuses as one that cannot seal."""
    try:
        manager = get_encryption_manager()
    except Exception as exc:
        raise EncryptionUnavailable(
            f"the field encryption cannot be built ({type(exc).__name__}): nothing is written in clear") from exc
    return manager.encrypt(value)


def field_refusal() -> str | None:
    """Why a field written now would be refused (the global manager's
    ``refusing``), or None: asked before work whose result a store must save,
    so that the refusal is said before anything runs."""
    return get_encryption_manager().refusing


def decrypt_field(value: str) -> str:
    """Convenience: decrypt a field value using the global manager."""
    return get_encryption_manager().decrypt(value)
