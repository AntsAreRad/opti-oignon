"""Notes data layer (N.1): the per-attachment AES-256-GCM blob store.

The second of the two independent at-rest layers for notes media (the first is
the SQLCipher database in :mod:`opti_oignon.notes.notes_store`). Each media blob
(audio / image / drawing) is encrypted on disk with AES-256-GCM under a
PER-ATTACHMENT subkey, with a fresh nonce per blob, and written already-encrypted
to ``<root>/notes_blobs/<attachment_id>.blob`` -- no plaintext temp file is ever
produced, and decryption happens in memory only. So even a full disk image yields
an encrypted database (needs the master key) and individually-encrypted blob
files (each needs its own derived subkey); compromising one blob's subkey does
not expose the others.

The per-attachment subkey (decision N1-D1) is derived by the codebase's
established domain-separated construction:

    subkey = HMAC-SHA256(master_key, b"oo-notes-attachment-" + attachment_id)

This is the same idiom signing.py's ``_wrap_subkey`` uses for the device signing
key, what db_encryption calls its "HKDF-like construction (HMAC-SHA256)", and what
auth_2fa derives "on its own HKDF domain". It honours the domain-separation string
NOTES_FEATURE_ROADMAP.md specifies (``info = "oo-notes-attachment-" +
attachment_id``) while staying consistent with the audited crypto surface rather
than introducing a formal HKDF primitive used nowhere else. The master key is the
only secret; the derivation is open (Kerckhoffs-clean). The AES-256-GCM sealing
itself reuses :func:`opti_oignon.encryption.encrypt_bytes` /
:func:`~opti_oignon.encryption.decrypt_bytes` (the version||nonce(12)||ct||tag
format, a fresh ``os.urandom`` nonce per call), the same primitive the signing
private-key-at-rest path uses.

The master key is injectable (for tests; production leaves it ``None`` so
``get_encryption_key`` is consulted at use time). Without a master key -- as in
this test container, where ``get_encryption_key`` returns ``None`` -- the seal
REFUSES with :class:`NotesBlobUnavailable` rather than persist plaintext, the
signing.py ``SigningUnavailable`` posture. Keys are held in SecureBytes (mlock)
where available. ``checkpoint_before_apply`` is hardcoded True; ``FEATURE_AVAILABLE``
gates graceful degradation; the singleton has a ``reset_notes_blob_store`` hook.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Hardcoded and never overridable: a checkpoint is taken before any mutation is
# applied. Project-wide non-negotiable for every new module.
checkpoint_before_apply = True

FEATURE_AVAILABLE = True

# The per-attachment subkey domain-separation prefix (N1-D1). Distinct by
# construction from the signing (oo-veilid-signing-v1), SQLCipher, and 2FA
# labels. The subkey is HMAC-SHA256(master_key, prefix + attachment_id).
ATTACHMENT_SUBKEY_PREFIX = b"oo-notes-attachment-"

BLOB_DIRNAME = "notes_blobs"
BLOB_SUFFIX = ".blob"
_NONCE_SIZE = 12  # AES-256-GCM nonce; bytes [1:1+12] of the encrypt_bytes format


# Guarded backend integration (the canonical_store / signing idiom): the real
# AES-256-GCM primitives and SecureBytes in the full backend; importable in
# isolation without fastapi / ollama. No silent crypto fallback -- if the
# primitives are unavailable the seal refuses rather than weaken the at-rest
# guarantee.
try:
    from ..encryption import decrypt_bytes, encrypt_bytes, get_encryption_key

    _HAS_ENCRYPTION = True
except Exception:
    _HAS_ENCRYPTION = False
    encrypt_bytes = None  # type: ignore[assignment]
    decrypt_bytes = None  # type: ignore[assignment]

    def get_encryption_key():  # type: ignore[misc]
        return None


try:
    from ..secure_bytes import SecureBytes, secure_key_from_bytes

    _HAS_SECURE_BYTES = True
except Exception:
    _HAS_SECURE_BYTES = False
    SecureBytes = None  # type: ignore[assignment,misc]

    def secure_key_from_bytes(data: bytes) -> Any:  # type: ignore[misc]
        return data


def _default_root() -> Path:
    try:
        from ..config import DATA_DIR

        return Path(DATA_DIR)
    except Exception:
        return Path("data")


class NotesBlobUnavailable(RuntimeError):
    """Sealing or opening cannot proceed: no master encryption key to derive the
    per-attachment subkey, or the AES-256-GCM primitives are unavailable. No
    plaintext blob is ever written to disk in this state (the signing.py
    ``SigningUnavailable`` posture)."""


def _attachment_subkey(master_raw: bytes, attachment_id: str) -> bytes:
    """HMAC-SHA256(master, prefix + attachment_id): the domain-separated
    per-attachment subkey (N1-D1). The caller does not retain ``master_raw``
    beyond the derivation."""
    info = ATTACHMENT_SUBKEY_PREFIX + attachment_id.encode("ascii")
    return hmac.new(master_raw, info, hashlib.sha256).digest()


class NotesBlobStore:
    """Per-attachment encrypted blob store (the second independent at-rest layer)."""

    def __init__(
        self,
        root: Optional[Path | str] = None,
        *,
        master_key: Optional[bytes] = None,
    ) -> None:
        base = Path(root) if root is not None else _default_root()
        self._blob_dir = base / BLOB_DIRNAME
        self._blob_dir.mkdir(parents=True, exist_ok=True)
        # The injected key is held only as raw bytes for derivation; tests pass a
        # deterministic key, production leaves it None so get_encryption_key is
        # consulted at use time.
        self._injected_master: Optional[bytes] = (
            bytes(master_key) if master_key is not None else None
        )

    @property
    def blob_dir(self) -> Path:
        return self._blob_dir

    def _blob_path(self, attachment_id: str) -> Path:
        return self._blob_dir / (attachment_id + BLOB_SUFFIX)

    def _master_raw(self) -> Optional[bytes]:
        if self._injected_master is not None:
            return self._injected_master
        key = get_encryption_key()
        if key is None:
            return None
        return key.as_bytes() if hasattr(key, "as_bytes") else bytes(key)

    def _subkey_for(self, attachment_id: str) -> bytes:
        if encrypt_bytes is None or decrypt_bytes is None:
            raise NotesBlobUnavailable(
                "AES-256-GCM primitives unavailable; refusing to handle blobs"
            )
        master_raw = self._master_raw()
        if master_raw is None:
            raise NotesBlobUnavailable(
                "no master encryption key: refusing to seal a notes attachment "
                "blob rather than persisting plaintext"
            )
        try:
            return _attachment_subkey(master_raw, attachment_id)
        finally:
            # Best-effort: drop our reference to the derived-from master bytes
            # immediately (an injected key belongs to the caller; a SecureBytes
            # master wipes its own buffer).
            master_raw = b""  # noqa: F841

    def seal(self, attachment_id: str, plaintext: bytes) -> Path:
        """Encrypt and write the blob; return its path.

        A fresh nonce per blob comes from ``encrypt_bytes`` (``os.urandom``). The
        ciphertext is written to a ``.tmp`` sibling and atomically renamed -- the
        temp itself is ciphertext, so no plaintext temp file is ever produced.
        """
        subkey = self._subkey_for(attachment_id)
        blob = encrypt_bytes(subkey, plaintext)
        path = self._blob_path(attachment_id)
        tmp = self._blob_dir / (attachment_id + BLOB_SUFFIX + ".tmp")
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)
        return path

    def open(self, attachment_id: str) -> bytes:
        """Decrypt and return the blob plaintext (in memory only).

        Raises on a subkey/ciphertext mismatch (the AES-256-GCM tag check), which
        is what binds a blob to its attachment id: a blob sealed for one id does
        not open under another's subkey.
        """
        subkey = self._subkey_for(attachment_id)
        blob = self._blob_path(attachment_id).read_bytes()
        return decrypt_bytes(subkey, blob)

    def open_secure(self, attachment_id: str) -> Any:
        """The blob plaintext wrapped in SecureBytes (mlock); the caller wipes."""
        return secure_key_from_bytes(self.open(attachment_id))

    def nonce_of(self, attachment_id: str) -> bytes:
        """The 12-byte nonce embedded in the on-disk blob (for the manifest row)."""
        blob = self._blob_path(attachment_id).read_bytes()
        return blob[1 : 1 + _NONCE_SIZE]

    def exists(self, attachment_id: str) -> bool:
        return self._blob_path(attachment_id).is_file()

    def delete(self, attachment_id: str) -> bool:
        path = self._blob_path(attachment_id)
        if path.is_file():
            path.unlink()
            return True
        return False


# Module-level singleton with a reset for test isolation (the S171 lesson).
_blob_store: Optional[NotesBlobStore] = None


def get_notes_blob_store() -> NotesBlobStore:
    global _blob_store
    if _blob_store is None:
        _blob_store = NotesBlobStore()
    return _blob_store


def reset_notes_blob_store() -> None:
    global _blob_store
    _blob_store = None
