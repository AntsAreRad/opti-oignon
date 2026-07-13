#!/usr/bin/env python3
"""Per-device record signing for Veilid sync (S205, VL-01, sync cycle Bloc 2).

The authenticity layer over the wire records. The content hash (records.py) is
integrity only: any peer can compute the correct SHA-256 for a forged payload,
and a record's ``device`` provenance is self-asserted -- a paired-but-compromised
peer could forge records under another device's name and steer LWW merges (the
S184 register's VL-01). This module closes that: every device holds an
ML-DSA-65 signing keypair; a local publish attaches a signature over the
record's canonical bytes (``records.canonical_record_bytes`` -- v, kind, id,
clock, device, hash, payload, deleted, updated_at, so re-clocking or
re-attributing a signed record breaks it); the engine's apply seam verifies an
incoming record against the key registered for the record's ORIGIN device and
refuses what does not verify. This is the trust boundary that later lets a
lower-trust paired device (the phone, cas 2) be contained rather than blindly
trusted.

Key custody. The signing PRIVATE key never lands in the plaintext peer registry
(PEER-01 is unresolved until the RS-01 at-rest lot) and never lands on disk in
plaintext at all: it persists in ``data/.veilid_signing_key`` as a versioned
JSON envelope whose private half is AES-256-GCM-encrypted
(``encryption.encrypt_bytes``) under a domain-separated subkey
HMAC-SHA256(master_key, "oo-veilid-signing-v1") -- the db_encryption idiom --
with the file chmod 600. Without a master key (``get_encryption_key()`` is
None) minting REFUSES: the device journals unsigned and behaves as a pre-VL-01
peer, honestly, rather than writing a plaintext private key. In memory the
private key lives in SecureBytes (mlock where available) and is wiped after
each use. The PUBLIC key is Kerckhoffs-public material: it travels in the
pairing payload and sits in the peer registry like the routing key. The
keypair is minted lazily on first signing need (the SYN-02/CHF-05 lazy-mint
precedent) under an exclusive create, so the first mint wins under
concurrency. Rotation is out of scope this lot; the versioned envelope
("format", "algorithm", "created_at") does not preclude it -- a rotation is a
new keypair plus a re-publish and a re-pair.

The signer seam is injectable (the fake-engine idiom): :class:`RecordSigner`
is the protocol (sign(bytes) -> bytes, verify(bytes, sig, pub) -> bool,
public_key() -> bytes), :class:`PqcRecordSigner` is the liboqs-backed default
(``pqc_signatures.sign_bytes`` / ``verify_bytes``, the project's ML-DSA-65
suite), and tests inject a deterministic fake -- liboqs is absent in the test
container; the real path is host-verified by the shakedown's crypto item.

Migration (the bounded grace window, CLOSED at S208). Journals were young when
signing landed (the producers arrived this same cycle), so the chosen path was
a one-time full re-publish of the local set with signatures
(``SyncEngine.republish_signed``) plus a fleet re-pair to distribute the
public keys -- NOT a permanent accept-unsigned mode. During the mixed-fleet
window (S205..S207), an unsigned record whose ORIGIN device had no registered
signing key was accepted-and-counted (surfaced as ``unverified`` on the round)
under :data:`ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS`; an unsigned or invalid
record whose origin HAS a registered key was always refused, window or no
window. The window ended at this cycle's Bloc 4 release session (S208, 3.7.0):
the constant is False and an unkeyed-origin record refuses like the rest. The
fleet upgrade order and the honest recovery for a device that flips before its
peers republish are documented on the constant below.

Mode posture: minting and holding keys, signing a local publish, and the
republish are local-disk/local-CPU operations, permitted in any mode and never
gated -- the documented producers/journal posture. Only the wire is
Daily-gated, at the existing protocol/engine seams. Verification of incoming
data is reading, ungated in itself (it runs inside the already-gated round).

Kerckhoffs: the scheme is open. Security lives in the private signing keys,
never in the shape of the recipe or the envelope; the canonical byte recipe,
the file format, and this module are public by design.

Crypto imports are lazy and guarded (the change_feed safe_connect precedent),
so the veilid package keeps collecting standalone in any environment.
"""

from __future__ import annotations

import base64
import json
import logging
import stat
import threading
from pathlib import Path
from typing import Any, Protocol

from opti_oignon.veilid.records import SyncRecord, canonical_record_bytes

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# The signing key envelope format and suite. The algorithm string matches
# pqc_signatures (liboqs naming for ML-DSA-65).
SIGNING_KEY_FORMAT = "veilid-signing-v1"
# The DECLARED mechanism. What a minted envelope records is the mechanism that
# actually resolved in the installed liboqs (see _resolved_algorithm), never
# this constant: a name hardcoded here and a name resolved there are how an
# envelope comes to claim an algorithm it was not signed with.
SIGNING_ALGORITHM = "ML-DSA-65"
KEY_FILENAME = ".veilid_signing_key"

# The domain-separation label for the at-rest key-wrapping subkey
# (HMAC-SHA256(master_key, label) -- the db_encryption idiom). Distinct from
# the SQLCipher and field-level labels by construction.
_KEY_WRAP_LABEL = b"oo-veilid-signing-v1"

# S205 (VL-01) opened a bounded migration grace window under this constant;
# S208 (sync cycle Bloc 4, the 3.7.0 release) CLOSED it. False means an
# unsigned record from an origin with NO registered signing key REFUSES like
# the rest (counted, never applied, never holding the watermark). The window
# was a migration aid for mixed fleets, never a permanent mode, and it is a
# hard constant by design: a config knob would re-open the permanent
# accept-unsigned mode by the back door (one config write reversing a
# security boundary), and recovery does not need it -- see the fleet order
# below. Tests that exercise the historical open-window behaviour monkeypatch
# the module attribute (the read is at call time in the engine's verify seam).
#
# Fleet upgrade order (also in VEILID_SPEC.md section 8 and the CHANGELOG):
# upgrade every device to 3.7.0, re-pair each peer pair (ONE confirmation per
# peer by the S206 key-change demotion design), run the one-time
# ``SyncEngine.republish_signed`` on each device (the sync panel exposes it),
# then run rounds and watch the surfaced counters: ``refused`` falls to zero
# once every peer has republished. A device that flips BEFORE its peers
# republish refuses their unsigned records in the interim -- honestly stated
# recovery: those records re-arrive SIGNED after the peer's republish (the
# CHF-01 backstop or a CHF-05 epoch resync re-serves them), so nothing is
# lost; convergence is merely delayed. The pre-VL-01 verify-incapable posture
# (no PQC backend) is a DIFFERENT branch and is unchanged by the flip: such a
# device still accepts everything as ``unverified`` with a warning, because
# refusing what it cannot check would partition the fleet, not protect it.
ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS = False


class RecordSigner(Protocol):
    """The injectable signing seam: sign, verify, and expose the public key.

    The default implementation is liboqs-backed (:class:`PqcRecordSigner`);
    tests inject a deterministic fake because liboqs is absent in the test
    container. ``sign`` and ``public_key`` may raise when signing is
    unavailable (no backend, no master key); ``verify`` never raises and
    returns False on any problem, the pqc_signatures posture.
    """

    def sign(self, data: bytes) -> bytes:
        ...

    def verify(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        ...

    def public_key(self) -> bytes:
        ...


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text.encode("ascii"))


def _wrap_subkey() -> bytes | None:
    """The at-rest key-wrapping subkey, or ``None`` when no master key exists.

    HMAC-SHA256(master_key, label): domain-separated from the SQLCipher and
    field-level subkeys (the db_encryption idiom). The master key arrives as
    SecureBytes from ``get_encryption_key`` and its raw bytes are not retained
    beyond the derivation.
    """
    try:
        import hashlib
        import hmac

        from opti_oignon.encryption import get_encryption_key

        master = get_encryption_key()
        if master is None:
            return None
        raw = master.as_bytes() if hasattr(master, "as_bytes") else bytes(master)
        return hmac.new(raw, _KEY_WRAP_LABEL, hashlib.sha256).digest()
    except Exception:
        logger.debug("signing: master key unavailable", exc_info=True)
        return None


def _default_key_path() -> Path:
    from opti_oignon.config import DATA_DIR

    return Path(DATA_DIR) / KEY_FILENAME


def _resolved_algorithm() -> str:
    """The mechanism the signing backend ACTUALLY resolved, never a constant.

    Lazily imported so this module stays isolatable. Falls back to the declared
    name only when the backend cannot be reached at all -- in which case nothing
    can be minted anyway, and no envelope is written.
    """
    try:
        from opti_oignon.pqc_signatures import PQC_MECHANISM

        return PQC_MECHANISM or SIGNING_ALGORITHM
    except Exception:  # pragma: no cover - defensive
        return SIGNING_ALGORITHM


class SigningUnavailable(RuntimeError):
    """Signing cannot proceed: no PQC backend, or no master key to wrap with."""


class PqcRecordSigner:
    """The liboqs-backed default signer with encrypted-at-rest key custody.

    The keypair is minted lazily on first use (``sign`` / ``public_key``) and
    persisted as a versioned JSON envelope: the public key in plaintext base64
    (Kerckhoffs-public), the private key AES-256-GCM-encrypted under the
    domain-separated wrap subkey, chmod 600. Minting refuses without a master
    key or without liboqs (:class:`SigningUnavailable`): no plaintext private
    signing key ever lands on disk. In memory the private key is held in
    SecureBytes for the duration of one signing call and wiped after.
    ``verify`` needs no local key and never raises. The path is injectable so
    tests run against a temporary directory.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path: Path | None = Path(path) if path is not None else None
        self._lock = threading.Lock()

    def _key_path(self) -> Path:
        return self._path if self._path is not None else _default_key_path()

    # Custody

    def _mint(self, fpath: Path) -> dict[str, Any]:
        """Mint and persist the keypair envelope; first mint wins.

        Refuses (:class:`SigningUnavailable`) without liboqs or without a
        master key. The exclusive create ('x') keeps the first minted envelope
        under concurrency, the file-based analogue of the INSERT OR IGNORE
        identity-row idiom; a loser of the race re-reads the winner's file.
        """
        from opti_oignon.pqc_signatures import PQC_AVAILABLE, generate_pqc_keypair

        if not PQC_AVAILABLE:
            raise SigningUnavailable(
                "PQC backend (liboqs) unavailable: cannot mint the device "
                "signing keypair; records will be published unsigned "
                "(pre-VL-01 posture)"
            )
        subkey = _wrap_subkey()
        if subkey is None:
            raise SigningUnavailable(
                "no master encryption key: refusing to mint the device "
                "signing keypair rather than persisting a plaintext private "
                "key; records will be published unsigned (pre-VL-01 posture)"
            )
        from datetime import datetime, timezone

        from opti_oignon.encryption import encrypt_bytes

        public_key, private_key = generate_pqc_keypair()
        try:
            blob = encrypt_bytes(subkey, private_key)
        finally:
            private_key = b"\x00" * len(private_key)  # best-effort wipe
        envelope = {
            "format": SIGNING_KEY_FORMAT,
            "algorithm": _resolved_algorithm(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "public_key": _b64encode(public_key),
            "private_key_enc": _b64encode(blob),
        }
        fpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(fpath, "x", encoding="ascii") as f:
                f.write(json.dumps(envelope, indent=2) + "\n")
        except FileExistsError:
            # Lost the mint race: the winner's envelope is the keypair.
            return self._read_envelope(fpath)
        try:
            fpath.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            logger.warning(
                "signing: could not set key file permissions to 600: %s", fpath
            )
        logger.info(
            "signing: device signing keypair minted (%s): %s",
            _resolved_algorithm(),
            fpath,
        )
        return envelope

    def _read_envelope(self, fpath: Path) -> dict[str, Any]:
        raw = json.loads(fpath.read_text(encoding="ascii"))
        if not isinstance(raw, dict):
            raise ValueError("signing key file must contain a JSON object")
        if raw.get("format") != SIGNING_KEY_FORMAT:
            raise ValueError(
                "unsupported signing key format: {!r}".format(raw.get("format"))
            )
        if not raw.get("public_key") or not raw.get("private_key_enc"):
            raise ValueError("signing key file missing key material")
        return raw

    def _load_or_mint(self) -> dict[str, Any]:
        fpath = self._key_path()
        with self._lock:
            if fpath.is_file():
                return self._read_envelope(fpath)
            return self._mint(fpath)

    def _private_key_secure(self, envelope: dict[str, Any]) -> Any:
        """The decrypted private key wrapped in SecureBytes; caller wipes."""
        subkey = _wrap_subkey()
        if subkey is None:
            raise SigningUnavailable(
                "no master encryption key: cannot unwrap the device signing key"
            )
        from opti_oignon.encryption import decrypt_bytes
        from opti_oignon.secure_bytes import secure_key_from_bytes

        raw = decrypt_bytes(subkey, _b64decode(envelope["private_key_enc"]))
        return secure_key_from_bytes(raw)

    # RecordSigner protocol

    def public_key(self) -> bytes:
        """This device's signing public key, minting the keypair on first use."""
        envelope = self._load_or_mint()
        return _b64decode(envelope["public_key"])

    def sign(self, data: bytes) -> bytes:
        """Sign bytes with the device key, minting it on first use.

        The private key is unwrapped into SecureBytes for the duration of the
        call and wiped after, whatever the outcome.
        """
        from opti_oignon.pqc_signatures import sign_bytes

        envelope = self._load_or_mint()
        secret = self._private_key_secure(envelope)
        try:
            return sign_bytes(data, secret.as_bytes())
        finally:
            secret.wipe()

    def verify(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify a signature against a given public key; never raises."""
        try:
            from opti_oignon.pqc_signatures import verify_bytes

            return verify_bytes(data, signature, public_key)
        except Exception:
            logger.warning("signing: verification errored; treating as invalid")
            return False

    def verify_available(self) -> bool:
        """True when the verification backend exists (liboqs is importable).

        The engine duck-types this probe: a device that cannot verify AT ALL
        is wholly pre-VL-01 and accepts records as ``unverified`` (counted,
        warned) instead of refusing everything from keyed origins -- refusal
        there would partition the fleet, not protect it. A signer without
        this method is assumed capable (the injected test fakes).
        """
        try:
            from opti_oignon.pqc_signatures import PQC_AVAILABLE

            return bool(PQC_AVAILABLE)
        except Exception:
            return False


def attach_signature(record: SyncRecord, signer: RecordSigner) -> SyncRecord:
    """Return the record carrying a signature over its canonical bytes.

    The sign-at-publish helper: the signature covers
    :func:`records.canonical_record_bytes` (the one recipe), so it binds the
    clock and device alongside the content. Raises when the signer cannot sign
    (no backend, no master key); the caller decides whether to degrade to an
    unsigned publish (the engine does, with a warning).
    """
    import dataclasses

    raw = signer.sign(canonical_record_bytes(record))
    return dataclasses.replace(record, signature=_b64encode(raw))


def verify_record_signature(
    record: SyncRecord, public_key: bytes, signer: RecordSigner
) -> bool:
    """True when the record's signature verifies under ``public_key``.

    Never raises: an empty or undecodable signature, a missing key, or a
    verifier error all return False. The signature is checked over the same
    canonical bytes :func:`attach_signature` signed.
    """
    try:
        if not record.signature or not public_key:
            return False
        raw = _b64decode(record.signature)
        return bool(signer.verify(canonical_record_bytes(record), raw, public_key))
    except Exception:
        logger.debug("signing: unverifiable record signature", exc_info=True)
        return False


def encode_public_key(raw: bytes) -> str:
    """Encode a public key to base64url text (the wire and registry form).

    The producer-side counterpart of :func:`decode_public_key`: what the
    pairing payload carries (S205) and the peer registry stores. Validating,
    not defensive -- raises ``ValueError`` on a non-bytes or empty value (a
    programmer error; the key comes from this module's own custody).
    """
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise ValueError("public key must be non-empty bytes")
    return _b64encode(bytes(raw))


def decode_public_key(text: Any) -> bytes | None:
    """Decode a base64url public key defensively, or ``None``.

    The registry stores the public key as text (the pairing payload's
    encoding); a malformed stored value degrades to "no key" rather than
    raising into a round.
    """
    if not isinstance(text, str) or not text:
        return None
    try:
        return _b64decode(text)
    except Exception:
        return None


# Module-level singleton with a reset hook (one signer per process, testable);
# the SYN-04 guarded-singleton idiom.

_signer: RecordSigner | None = None
_signer_lock = threading.Lock()


def get_record_signer() -> RecordSigner:
    """Return the process record signer, creating the default once."""
    global _signer
    with _signer_lock:
        if _signer is None:
            _signer = PqcRecordSigner()
        return _signer


def set_record_signer(signer: RecordSigner | None) -> None:
    """Install a specific signer as the process singleton (used by tests)."""
    global _signer
    with _signer_lock:
        _signer = signer


def reset_record_signer() -> None:
    """Clear the process singleton so the next get creates a fresh one."""
    global _signer
    with _signer_lock:
        _signer = None
