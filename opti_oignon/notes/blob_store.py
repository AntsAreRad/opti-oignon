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
import struct
from pathlib import Path
from typing import Any

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

# Framed AEAD blob format (SYN-01, the vault on-demand transfer): a versioned,
# chunked container so multi-GB media is sealed and opened in CONSTANT memory.
# Header: MAGIC(4) || version(1) || u32(chunk_size). Then a sequence of frames,
# each: u32(frame_len) || encrypt_bytes(subkey, u32(index) || u8(is_final) ||
# chunk_plaintext). Every chunk is GCM-sealed under the SAME per-attachment
# subkey with a fresh random nonce (encrypt_bytes); the 5-byte index/final
# prefix is bound by that chunk's tag, so a reordered, dropped, duplicated, or
# truncated frame is detected on open WITHOUT introducing an AAD primitive --
# the seal stays on the audited encrypt_bytes surface. A distinct magic keeps
# legacy single-seal blobs (which have no magic) openable unchanged (D-A3).
FRAMED_MAGIC = b"OOBF"
FRAMED_VERSION = 1
_FRAMED_HEADER = struct.Struct(">4sBI")  # magic, version, chunk_size
_FRAME_PREFIX = struct.Struct(">IB")     # chunk index, is_final
_FRAME_LEN = struct.Struct(">I")         # length prefix of each frame
DEFAULT_CHUNK_SIZE = 1024 * 1024         # 1 MiB plaintext per chunk (D-A1)
# Sanity bound when reading a frame length from disk: reject an absurd u32 (a
# corrupt header could otherwise request a multi-GB allocation).
MAX_FRAME_BYTES = 64 * 1024 * 1024


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


def _rechunk(chunks: Any, size: int) -> Any:
    """Re-slice an iterable of byte pieces into fixed-``size`` pieces.

    Yields pieces of exactly ``size`` bytes while the running buffer overflows,
    then a final shorter piece with the remainder. This lets a caller stream
    plaintext in whatever piece sizes it has (a 64 KiB file read, a wire frame)
    while the on-disk frames stay uniform; the final (possibly short) piece is
    the one ``seal_stream`` marks is_final. An all-empty input yields nothing.
    """
    buf = bytearray()
    for c in chunks:
        if not c:
            continue
        buf += c
        while len(buf) >= size:
            yield bytes(buf[:size])
            del buf[:size]
    if buf:
        yield bytes(buf)


class NotesBlobStore:
    """Per-attachment encrypted blob store (the second independent at-rest layer)."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        master_key: bytes | None = None,
    ) -> None:
        base = Path(root) if root is not None else _default_root()
        self._blob_dir = base / BLOB_DIRNAME
        self._blob_dir.mkdir(parents=True, exist_ok=True)
        # The injected key is held only as raw bytes for derivation; tests pass a
        # deterministic key, production leaves it None so get_encryption_key is
        # consulted at use time.
        self._injected_master: bytes | None = (
            bytes(master_key) if master_key is not None else None
        )

    @property
    def blob_dir(self) -> Path:
        return self._blob_dir

    def _blob_path(self, attachment_id: str) -> Path:
        return self._blob_dir / (attachment_id + BLOB_SUFFIX)

    def _master_raw(self) -> bytes | None:
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

    def seal_stream(
        self,
        attachment_id: str,
        chunks: Any,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Path:
        """Seal a (possibly huge) plaintext stream into the framed blob format.

        ``chunks`` is any iterable of plaintext byte pieces -- a file read in a
        loop, the on-the-wire pieces of a synced blob, anything. The blob is
        written frame-by-frame to a ``.tmp`` sibling and atomically renamed,
        holding at most ONE chunk in memory at a time, so a multi-GB video never
        sits whole in RAM. Each frame is GCM-sealed under the per-attachment
        subkey with a fresh nonce; the index/final prefix binds ordering. An
        empty stream still yields a single final empty chunk (a valid one-frame
        blob). On any failure the partial ``.tmp`` is removed, never promoted.
        """
        if chunk_size <= 0:
            chunk_size = DEFAULT_CHUNK_SIZE
        subkey = self._subkey_for(attachment_id)
        path = self._blob_path(attachment_id)
        tmp = self._blob_dir / (attachment_id + BLOB_SUFFIX + ".tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(
                    _FRAMED_HEADER.pack(FRAMED_MAGIC, FRAMED_VERSION, chunk_size)
                )
                index = 0
                pending: bytes | None = None
                for piece in _rechunk(chunks, chunk_size):
                    if pending is not None:
                        self._write_frame(f, subkey, index, False, pending)
                        index += 1
                    pending = piece
                if pending is None:
                    pending = b""
                self._write_frame(f, subkey, index, True, pending)
            os.replace(tmp, path)
        except BaseException:
            try:
                if tmp.is_file():
                    tmp.unlink()
            except Exception:
                pass
            raise
        return path

    @staticmethod
    def _write_frame(
        f: Any, subkey: bytes, index: int, is_final: bool, data: bytes
    ) -> None:
        frame = encrypt_bytes(
            subkey, _FRAME_PREFIX.pack(index, 1 if is_final else 0) + data
        )
        f.write(_FRAME_LEN.pack(len(frame)))
        f.write(frame)

    def is_framed(self, attachment_id: str) -> bool:
        """Whether the on-disk blob uses the framed format (vs legacy single-seal)."""
        try:
            with open(self._blob_path(attachment_id), "rb") as f:
                return f.read(len(FRAMED_MAGIC)) == FRAMED_MAGIC
        except OSError:
            return False

    def open_stream(self, attachment_id: str) -> Any:
        """Yield the blob plaintext one chunk at a time, in constant memory.

        For a framed blob each frame is decrypted and verified independently:
        the per-frame GCM tag rejects any byte tamper, the index must advance
        with no gap, exactly one frame carries is_final and it must be last, and
        a stream that ends without a final frame is a truncation -- each raises
        :class:`NotesBlobUnavailable`. A legacy single-seal blob (no framed
        magic) is decrypted whole and yielded as one chunk, so a caller can read
        either format uniformly. The file handle is held for the generator's
        life and released when iteration ends or the consumer stops early.
        """
        subkey = self._subkey_for(attachment_id)
        with open(self._blob_path(attachment_id), "rb") as f:
            head = f.read(_FRAMED_HEADER.size)
            if len(head) < _FRAMED_HEADER.size or head[: len(FRAMED_MAGIC)] != FRAMED_MAGIC:
                # Legacy single-seal: decrypt the whole (small) blob.
                f.seek(0)
                yield decrypt_bytes(subkey, f.read())
                return
            _magic, version, _chunk_size = _FRAMED_HEADER.unpack(head)
            if version != FRAMED_VERSION:
                raise NotesBlobUnavailable(
                    "unsupported framed blob version: " + repr(version)
                )
            expected = 0
            seen_final = False
            while True:
                lp = f.read(_FRAME_LEN.size)
                if not lp:
                    break
                if len(lp) < _FRAME_LEN.size:
                    raise NotesBlobUnavailable("truncated frame length")
                (frame_len,) = _FRAME_LEN.unpack(lp)
                if frame_len <= 0 or frame_len > MAX_FRAME_BYTES:
                    raise NotesBlobUnavailable("implausible frame length")
                frame = f.read(frame_len)
                if len(frame) < frame_len:
                    raise NotesBlobUnavailable("truncated frame body")
                plain = decrypt_bytes(subkey, frame)  # GCM tag check
                if len(plain) < _FRAME_PREFIX.size:
                    raise NotesBlobUnavailable("malformed frame")
                index, final_flag = _FRAME_PREFIX.unpack(
                    plain[: _FRAME_PREFIX.size]
                )
                if seen_final:
                    raise NotesBlobUnavailable("a frame follows the final frame")
                if index != expected:
                    raise NotesBlobUnavailable("out-of-order frame")
                expected += 1
                if final_flag:
                    seen_final = True
                yield plain[_FRAME_PREFIX.size:]
            if not seen_final:
                raise NotesBlobUnavailable("blob ended without a final frame")

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


# Module-level singleton with a reset for test isolation.
_blob_store: NotesBlobStore | None = None


def get_notes_blob_store() -> NotesBlobStore:
    global _blob_store
    if _blob_store is None:
        _blob_store = NotesBlobStore()
    return _blob_store


def reset_notes_blob_store() -> None:
    global _blob_store
    _blob_store = None
