#!/usr/bin/env python3
"""Model weight provenance -- hash-pinned GGUF integrity.

The path guard in ``inference_backend`` answers WHERE a model file lives:
``_resolve_model_path`` refuses absolute paths and traversal, so a load can
only ever reach a file inside a configured model directory. The SSRF guard in
``model_manager`` answers WHERE a model was fetched from. Neither answers WHAT
the bytes are. A file sitting at a legitimate path, fetched from a legitimate
host, is handed to the native GGUF parser with no integrity check at all, so a
compromised mirror, a corrupted transfer, or any local process able to write
into a model directory can substitute the weights that llama.cpp parses
in-process.

This module closes that gap. It keeps a manifest pinning each model file to
the sha256 of its bytes, seals that manifest with the strongest scheme the host
is ALLOWED to use, and gives the load seam one decision: verified, or refused.

Allowed, not merely available. Where a post-quantum signature is required -- the
operator asked, or the mode is a fortress -- HMAC-SHA512 is not a fallback for
it. A MAC is forgeable by whoever holds the shared secret and verifiable by
nobody else; a signature is neither. Substituting one for the other is not a
downgrade, it is a change of security property, and doing it on a warning line
is the whole of how a provenance seal comes to guarantee nothing. So a required
seal that cannot be produced is a refusal. Where nothing was required, the HMAC
seal stands.

Posture
-------
Fail-secure. Under enforcement, ONLY a model whose bytes hash to the pinned
digest, in a manifest whose seal verifies under the required scheme, is
allowed to load. Every other outcome -- no manifest, no key, a missing seal, a
downgraded seal, a broken seal, an unpinned model, a digest mismatch, an
unreadable file -- refuses.

The security mode decides enforcement and configuration cannot weaken it.
Bulbe always enforces. A mode that cannot be resolved, or that is not exactly
"daily", is treated as Bulbe. Only Daily consults configuration, and its
default is to observe without blocking, so an existing installation keeps
working until its models are enrolled.

No third-party import at module scope. That is a security property rather than
a style choice: the load seam treats an unresolvable provenance module as a
REFUSAL when the mode enforces, so this module must not be able to fail its
import for a reason as banal as a missing optional package.

Honest limits, stated rather than implied
-----------------------------------------
The seal is made with a key held by the same local user that runs the backend.
An attacker who already owns that account owns the key and can re-seal the
manifest. This design does not claim to defend against a full local compromise
of the account. It defends against a tampered or corrupted download, against a
substitution by any process that does NOT hold the key, and it turns "I think
I am running these weights" into an auditable, verifiable pin.

A residual window remains between hashing the file and llama.cpp opening it,
because llama-cpp-python takes a path rather than a file descriptor. Closing it
entirely would need an fd-based load the library does not expose.

Security derives from the key and from the verification, never from the format
being secret: the manifest layout, the canonical byte recipe and the seal
schemes are all public here.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as _hmac
import json
import logging
import os
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Stdlib-only at module scope, so importing this module cannot fail for a
# dependency reason. The load seam relies on that: an ImportError there is
# treated as a refusal under enforcement, and it must therefore be a genuine
# anomaly rather than a routine consequence of an optional package.
FEATURE_AVAILABLE = True

checkpoint_before_apply = True

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_MANIFEST_PATH = _DATA_DIR / "model_provenance.json"

MANIFEST_VERSION = 1

# Seal schemes, strongest first. The host requires the strongest one it can
# actually perform; a manifest presenting a weaker one is a downgrade and is
# refused before any cryptographic work happens.
SCHEME_PQC = "mldsa65"
SCHEME_HMAC = "hmac-sha512"

MODE_DAILY = "daily"
MODE_BULBE = "bulbe"

ENFORCE = "enforce"
WARN = "warn"

# Classification outcomes. Exactly one of them -- REASON_VERIFIED -- is an
# allowed load under enforcement.
REASON_VERIFIED = "verified"
REASON_MANIFEST_MISSING = "manifest_missing"
REASON_MANIFEST_UNREADABLE = "manifest_unreadable"
REASON_KEY_UNAVAILABLE = "key_unavailable"
REASON_SEAL_MISSING = "seal_missing"
REASON_SEAL_DOWNGRADE = "seal_downgrade"
REASON_SEAL_INVALID = "seal_invalid"
REASON_UNPINNED = "unpinned"
REASON_DIGEST_MISMATCH = "digest_mismatch"
REASON_FILE_UNREADABLE = "file_unreadable"

_CHUNK = 1024 * 1024


class ProvenanceError(Exception):
    """Base class for provenance failures."""


class ProvenanceRefusal(ProvenanceError):
    """A model load is refused because its provenance did not verify.

    Loud by design. The resource governor gate beside this one fails open
    because an unavailable resource governor is a degraded resource decision;
    an unavailable integrity proof is not a degraded security decision, it is
    the absence of one.
    """

    def __init__(self, decision: ProvenanceDecision):
        self.decision = decision
        super().__init__(
            f"Model provenance refused: {decision.model} ({decision.reason})"
        )


@dataclass
class SealKeys:
    """The key material for one seal scheme.

    For HMAC the two members are the same secret. For ML-DSA-65 they are the
    private and public halves of the keypair.
    """

    scheme: str
    sign_key: bytes | None = None
    verify_key: bytes | None = None


@dataclass
class ProvenanceDecision:
    """The verdict handed back to the load seam."""

    allowed: bool
    reason: str
    enforcement: str
    model: str
    digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "enforcement": self.enforcement,
            "model": self.model,
            "digest": self.digest,
        }


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------


def compute_digest(path: Path | str, chunk_size: int = _CHUNK) -> str:
    """Streaming sha256 over the file bytes.

    Streaming is mandatory rather than tidy: a GGUF is routinely tens of
    gigabytes, so the whole-file read a naive implementation would do is not
    an option. The digest must not depend on the chunk size -- that is what
    makes the pin portable between the writer and the reader.

    Raises:
        OSError: If the file cannot be read.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Canonical bytes -- the seal recipe
# ---------------------------------------------------------------------------


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    """The exact bytes a seal is computed over.

    pqc_signatures.sign_bytes is content-agnostic and says so: the caller owns
    the canonical recipe. That makes canonicalisation part of the security
    contract, not a serialisation detail. Two payloads that are logically the
    same object must produce byte-identical output regardless of key insertion
    order or incidental whitespace, or a seal made by the writer would not
    verify for the reader -- and, worse, an attacker could look for a
    re-serialisation that changes meaning while preserving the sealed bytes.

    Sorted keys, no separator whitespace, ASCII-escaped, UTF-8 encoded.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _payload_of(manifest: dict[str, Any]) -> dict[str, Any]:
    """The sealed half of a manifest: everything except the seal itself."""
    return {
        "version": manifest.get("version", MANIFEST_VERSION),
        "entries": manifest.get("entries", {}),
    }


# ---------------------------------------------------------------------------
# Key material
# ---------------------------------------------------------------------------


def _pqc_enabled() -> bool:
    """True when ML-DSA-65 signing is both available and configured on."""
    try:
        from opti_oignon.pqc_signatures import PQC_AVAILABLE, is_pqc_enabled

        return bool(PQC_AVAILABLE) and bool(is_pqc_enabled())
    except Exception:
        return False


def _hmac_key() -> Any:
    """The local HMAC secret, mirroring the plugin allowlist idiom."""
    try:
        from opti_oignon.encryption import load_keyfile

        key, _salt, _kdf = load_keyfile()
        return key
    except Exception:
        pass
    try:
        keyfile = _DATA_DIR / ".keyfile"
        if keyfile.is_file():
            raw = keyfile.read_bytes()
            if len(raw) >= 32:
                return raw[:32]
    except Exception:
        pass
    return None


def _extract_key_bytes(key: Any) -> bytes | None:
    """Raw bytes from a key that may be a SecureBytes wrapper."""
    if key is None:
        return None
    if hasattr(key, "as_bytes"):
        return key.as_bytes()
    return bytes(key)


def _pqc_required() -> bool:
    """May this host substitute a symmetric MAC for a signature?

    Fortress mode says no, and it does not say it through a policy file: there
    the signature is a property of the MODE, like the socket bind. The signing
    module computes this answer correctly and nothing here ever asked it.

    A module that cannot be IMPORTED is a machinery failure, not a mode verdict,
    and no function may manufacture a refusal out of a broken import.
    """
    try:
        from opti_oignon.pqc_signatures import pqc_required

        return bool(pqc_required())
    except Exception as exc:  # noqa: BLE001 - a broken import is not a verdict
        logger.error(
            "the post-quantum requirement could not be determined: %s", exc
        )
        return False


def _pqc_seal_keys() -> tuple[SealKeys | None, str | None]:
    """Post-quantum seal keys, or None and the reason there are none.

    This function does not decide what an unusable keypair MEANS -- it reports.
    The decision belongs to exactly one place, and putting it here as well was a
    mistake that a directed mutation caught: two refusal sites, and the second
    silently covered for the first, so neither could be pinned. A guard whose
    failure another guard conceals is not a guard, it is a comfort.
    """
    try:
        from opti_oignon.pqc_signatures import load_pqc_keypair

        public, private = load_pqc_keypair()
    except Exception as exc:  # noqa: BLE001 - the caller decides what it means
        return None, str(exc)

    if public and private:
        return SealKeys(scheme=SCHEME_PQC, sign_key=private, verify_key=public), None
    return None, "the keypair file yielded no key material"


def resolve_seal_keys() -> SealKeys | None:
    """The strongest seal scheme this host is ALLOWED to perform.

    Not the strongest it can manage. Asking ``is_pqc_enabled`` -- the policy
    file alone -- was the hole: with no signing block in that file the intent
    read False, and this function walked straight past a live signing primitive
    into the HMAC branch. A fortress host sealed its model manifests with a
    forgeable MAC while the boot checklist reported the primitive green, because
    the checklist was asking whether the primitive RESOLVED, and it had.

    Where a signature is required and none can be produced, this returns None
    rather than raising. None is not a shrug: the estate already has a meaning
    for it on each of the two paths that ask, and both are right.

      * Sealing refuses outright -- there is nothing to sign with, and
        ``record_model`` says so.
      * Classifying reports KEY_UNAVAILABLE, and the enforcement policy refuses
        the model through the seam built for exactly that. ``enforcement_mode``
        already ENFORCES for every mode that is not Daily, so a fortress refuses
        the load without any of this needing to know it is a fortress.

    Raising here would have been worse than the defect it replaced. It would
    have broken the "no policy, no raise" contract ``classify_model`` states
    about itself, and turned an unsealed model in a fortress from a refusal the
    caller can render into an exception it cannot -- a denial of service on the
    whole model-loading path, bought with no security at all.
    """
    required = _pqc_required()

    keys, why = (
        _pqc_seal_keys() if (_pqc_enabled() or required) else (None, None)
    )
    if keys is not None:
        return keys

    if required:
        logger.error(
            "a post-quantum seal was required and no keypair could be used "
            "(%s); refusing to substitute a symmetric MAC for a signature, "
            "which is forgeable by whoever holds the shared secret",
            why,
        )
        return None

    if why:
        logger.warning("PQC keypair unavailable for the model seal: %s", why)

    raw = _extract_key_bytes(_hmac_key())
    if raw:
        return SealKeys(scheme=SCHEME_HMAC, sign_key=raw, verify_key=raw)
    return None


# ---------------------------------------------------------------------------
# Seal
# ---------------------------------------------------------------------------


def compute_seal(
    payload: dict[str, Any],
    keys: SealKeys,
    *,
    signer: Callable[[bytes, bytes], bytes] | None = None,
) -> dict[str, str]:
    """Seal the canonical payload bytes under the given scheme."""
    data = canonical_bytes(payload)

    if keys.scheme == SCHEME_PQC:
        sign = signer
        if sign is None:
            from opti_oignon.pqc_signatures import sign_bytes as sign

        signature = sign(data, keys.sign_key or b"")
        return {
            "scheme": SCHEME_PQC,
            "value": base64.b64encode(signature).decode("ascii"),
        }

    if keys.scheme == SCHEME_HMAC:
        digest = _hmac.new(
            keys.sign_key or b"", data, hashlib.sha512
        ).hexdigest()
        return {"scheme": SCHEME_HMAC, "value": digest}

    raise ProvenanceError(f"Unknown seal scheme: {keys.scheme}")


def verify_seal(
    payload: dict[str, Any],
    seal: dict[str, Any] | None,
    keys: SealKeys,
    *,
    verifier: Callable[[bytes, bytes, bytes], bool] | None = None,
) -> str:
    """Verify a seal against the scheme this host REQUIRES.

    Returns REASON_VERIFIED, or the reason it failed.

    The scheme is checked before any cryptographic work. A host able to
    perform ML-DSA-65 requires ML-DSA-65: a manifest presenting an
    HMAC seal instead is a downgrade and is refused outright, even though the
    HMAC would verify. Accepting whichever scheme the artefact happens to name
    is the classic cryptographic-agility footgun -- it lets an attacker who
    holds only the weaker secret strip the stronger seal and substitute one it
    can forge.
    """
    if not isinstance(seal, dict) or not seal.get("value"):
        return REASON_SEAL_MISSING

    if seal.get("scheme") != keys.scheme:
        return REASON_SEAL_DOWNGRADE

    data = canonical_bytes(payload)

    if keys.scheme == SCHEME_PQC:
        verify = verifier
        if verify is None:
            from opti_oignon.pqc_signatures import verify_bytes as verify

        try:
            raw = base64.b64decode(str(seal.get("value")), validate=True)
        except Exception:
            return REASON_SEAL_INVALID
        ok = bool(verify(data, raw, keys.verify_key or b""))
        return REASON_VERIFIED if ok else REASON_SEAL_INVALID

    expected = _hmac.new(
        keys.verify_key or b"", data, hashlib.sha512
    ).hexdigest()
    ok = _hmac.compare_digest(expected, str(seal.get("value")))
    return REASON_VERIFIED if ok else REASON_SEAL_INVALID


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def current_mode() -> str:
    """The live security mode; an undeterminable mode is Bulbe.

    Same fail-secure resolution the Veilid sync gate uses: the import is lazy
    and per call, and any failure to name the mode is Bulbe.
    """
    try:
        from opti_oignon.security_mode import get_current_mode

        mode = str(get_current_mode() or "").strip().lower()
        return mode or MODE_BULBE
    except Exception:
        logger.warning(
            "Cannot determine security mode; treating as bulbe (fail-secure)."
        )
        return MODE_BULBE


def _load_config() -> dict[str, Any]:
    """The provenance section of the security config, or an empty dict."""
    try:
        import yaml

        path = Path(__file__).parent / "config" / "security.yaml"
        if not path.is_file():
            return {}
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        section = data.get("model_provenance")
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


# The backends whose load seam actually calls the gate above. Measured, not
# assumed: only LlamaCppBackend._get_or_load is wired to guard_model_load today.
# The other backends load weights with no provenance check, so a host that
# serves loads through them is not BRICKED by escalation -- it is a weaker
# posture, documented as a gap, not an escalation refusal that would buy no
# security. The escalation preflight keys off this set to tell a real brick from
# that gap; when the gate is wired into another backend, its name MUST join this
# set or the preflight stops protecting against the brick that wiring causes.
PROVENANCE_GATED_BACKENDS = frozenset({"llama_cpp"})


def backend_enforces_provenance(backend: str | None) -> bool:
    """True when this backend's load seam calls the provenance gate.

    The one place that answers "would a load through this backend be refused for
    an unpinned model". The escalation preflight asks it so that it blocks the
    brick a gated backend causes without false-bricking a host whose active
    backend never consults the gate at all.
    """
    return bool(backend) and backend in PROVENANCE_GATED_BACKENDS


def enforcement_mode(
    *,
    mode: str | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Resolve ENFORCE or WARN for the live mode.

    Anything that is not exactly "daily" enforces. That inequality is the
    fail-secure hinge: Bulbe enforces, and so does a mode that is empty,
    misspelled, or invented, without any of them needing to be enumerated.
    Configuration is consulted for Daily alone, so no config key can ever
    weaken the fortress.
    """
    resolved = str(
        mode if mode is not None else current_mode() or ""
    ).strip().lower()

    if resolved != MODE_DAILY:
        return ENFORCE

    cfg = config if config is not None else _load_config()
    requested = str(cfg.get("enforcement", WARN)).strip().lower()
    return ENFORCE if requested == ENFORCE else WARN


def decide(reason: str, enforcement: str) -> bool:
    """Whether a classified model may load.

    Under enforcement exactly one outcome passes: a verified pin. Under
    observation nothing blocks, so an installation whose models are not yet
    enrolled keeps working while the refusals it WOULD have taken are logged.
    """
    if enforcement != ENFORCE:
        return True
    return reason == REASON_VERIFIED


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def load_manifest(path: Path | str | None = None) -> dict[str, Any] | None:
    """Read the manifest, or None when it is absent or unparseable.

    The two cases are deliberately NOT merged at the call site: an absent
    manifest and a corrupt one are different reasons, and both refuse under
    enforcement.
    """
    target = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    if not target.is_file():
        return None
    try:
        with open(target, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Model provenance manifest unreadable: %s", exc)
        return {}


def save_manifest(
    manifest: dict[str, Any], path: Path | str | None = None
) -> Path:
    """Write the manifest atomically, owner-readable only."""
    target = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.chmod(temp, stat.S_IRUSR | stat.S_IWUSR)
    temp.replace(target)
    return target


def record_model(
    model_path: Path | str,
    *,
    manifest_path: Path | str | None = None,
    keys: SealKeys | None = None,
    signer: Callable[[bytes, bytes], bytes] | None = None,
    chunk_size: int = _CHUNK,
) -> dict[str, Any]:
    """Enrol (or re-pin) one model and re-seal the manifest.

    Enrolment is an explicit act. Nothing here pins a model behind the user's
    back: the whole value of the pin is that somebody decided these bytes were
    the right ones.
    """
    resolved = Path(model_path)
    digest = compute_digest(resolved, chunk_size=chunk_size)

    seal_keys = keys if keys is not None else resolve_seal_keys()
    if seal_keys is None:
        raise ProvenanceError(
            "No key material available to seal the model provenance manifest"
        )

    manifest = load_manifest(manifest_path) or {}
    entries = dict(manifest.get("entries") or {})
    entries[resolved.name] = {
        "sha256": digest,
        "size": resolved.stat().st_size,
        "recorded_at": time.time(),
    }

    payload = {"version": MANIFEST_VERSION, "entries": entries}
    sealed = dict(payload)
    sealed["seal"] = compute_seal(payload, seal_keys, signer=signer)
    save_manifest(sealed, manifest_path)

    logger.info("Model pinned: %s (sha256=%s)", resolved.name, digest[:16])
    return {"model": resolved.name, "sha256": digest, "scheme": seal_keys.scheme}


def enroll_models(
    model_paths: list[Path | str],
    *,
    manifest_path: Path | str | None = None,
    keys: SealKeys | None = None,
    signer: Callable[[bytes, bytes], bytes] | None = None,
    chunk_size: int = _CHUNK,
) -> dict[str, Any]:
    """Pin several on-disk models under ONE seal.

    ``record_model`` already enrols one model that is on disk -- it just happens
    to be reached only from the download path, so a host that already holds its
    weights had no way to pin them without fetching them again. This is that
    handle, and it exists because the manifest seal is GLOBAL: sealing once over
    the whole set is one operation, and sealing per model would be N-1 wasted
    signatures over payloads that never ship.

    Refuse rather than downgrade, exactly as sealing does everywhere else. If a
    signature is required here and no key can produce one, this raises and writes
    NOTHING. A partial manifest -- some pins, no seal, or a MAC where a signature
    was required -- is worse than an absent one, because the escalation preflight
    would then read a manifest that exists and believe the host was ready.

    The digests are computed from the BYTES on disk. That is the difference
    between this and re-sealing: enrolment is the act of deciding these bytes are
    the right ones, so it hashes them; re-sealing renews a decision already made
    and must never re-hash. Enrolling zero models writes nothing and says so.
    """
    resolved = [Path(p) for p in model_paths]
    if not resolved:
        return {"enrolled": [], "count": 0, "scheme": None}

    seal_keys = keys if keys is not None else resolve_seal_keys()
    if seal_keys is None:
        raise ProvenanceError(
            "No key material available to seal the model provenance manifest. "
            "Refusing to write a manifest this host cannot sign -- a partial "
            "one would be read as readiness the host does not have."
        )

    manifest = load_manifest(manifest_path) or {}
    entries = dict(manifest.get("entries") or {})
    enrolled: list[dict[str, Any]] = []
    for path in resolved:
        digest = compute_digest(path, chunk_size=chunk_size)
        entries[path.name] = {
            "sha256": digest,
            "size": path.stat().st_size,
            "recorded_at": time.time(),
        }
        enrolled.append({"model": path.name, "sha256": digest})

    payload = {"version": MANIFEST_VERSION, "entries": entries}
    sealed = dict(payload)
    sealed["seal"] = compute_seal(payload, seal_keys, signer=signer)
    save_manifest(sealed, manifest_path)

    logger.info(
        "Enrolled %d model(s) under %s", len(enrolled), seal_keys.scheme
    )
    return {
        "enrolled": enrolled,
        "count": len(enrolled),
        "scheme": seal_keys.scheme,
    }


def manifest_seal_scheme(path: Path | str | None = None) -> str | None:
    """The scheme the manifest on disk is sealed under, or None when there is none.

    Reported, never judged. Whether an HMAC seal is right depends on the host
    that reads it -- exactly right on a Daily machine, and a downgrade a fortress
    refuses -- so the comparison belongs to the caller and not here.
    """
    manifest = load_manifest(path)
    if not manifest:
        return None
    seal = manifest.get("seal")
    if not isinstance(seal, dict):
        return None
    scheme = seal.get("scheme")
    return str(scheme) if scheme else None


def reseal_manifest(
    *,
    manifest_path: Path | str | None = None,
    keys: SealKeys | None = None,
    signer: Callable[[bytes, bytes], bytes] | None = None,
) -> dict[str, Any]:
    """Re-seal the manifest under the scheme this host is ALLOWED to perform.

    The missing half of the requirement. A fortress requires a signature, and a
    host whose manifest is sealed with a MAC refuses every model it owns -- and
    until now nothing on this machine could turn that MAC into a signature. The
    seal was only ever written as a SIDE EFFECT of enrolling a model, so the one
    way to re-seal was to download every model again. A requirement with no way
    to comply is a trap, not a policy.

    NOTHING IS RE-PINNED. The entries are carried across verbatim: not re-hashed,
    not re-stat'ed, not touched. A re-seal that re-read the files on disk would
    bless whatever is sitting there now, which is precisely the substitution the
    manifest exists to refuse. An entry means somebody looked at those bytes and
    decided they were the right ones. Renewing that decision is not this
    operation's to make; it changes the SCHEME, never the CLAIM.

    The seal covers the whole payload, so this is one operation and not one per
    entry. Refuses rather than writing a seal the caller did not ask for: with no
    key material there is nothing to sign with, and a silent HMAC here would hand
    back a manifest the fortress will reject while the operator believes the
    migration happened.
    """
    manifest = load_manifest(manifest_path)
    if not manifest:
        raise ProvenanceError(
            "There is no model provenance manifest to re-seal. A manifest is "
            "written when a model is enrolled; enrol one first."
        )

    seal_keys = keys if keys is not None else resolve_seal_keys()
    if seal_keys is None:
        raise ProvenanceError(
            "No key material available to seal the model provenance manifest"
        )

    payload = _payload_of(manifest)
    previous = manifest.get("seal")
    previous_scheme = (
        previous.get("scheme") if isinstance(previous, dict) else None
    )

    sealed = dict(payload)
    sealed["seal"] = compute_seal(payload, seal_keys, signer=signer)
    save_manifest(sealed, manifest_path)

    entries = payload.get("entries") or {}
    logger.info(
        "Model provenance manifest re-sealed: %d entries, %s -> %s",
        len(entries),
        previous_scheme or "unsealed",
        seal_keys.scheme,
    )
    return {
        "scheme": seal_keys.scheme,
        "previous_scheme": previous_scheme,
        "entries": len(entries),
    }


# ---------------------------------------------------------------------------
# Classification and verdict
# ---------------------------------------------------------------------------


def classify_model(
    model_path: Path | str,
    manifest: dict[str, Any] | None,
    *,
    keys: SealKeys | None = None,
    verifier: Callable[[bytes, bytes, bytes], bool] | None = None,
    chunk_size: int = _CHUNK,
) -> tuple[str, str | None]:
    """Classify one model against a manifest. Pure: no policy, no raise.

    Returns (reason, digest). The digest is None whenever the file was never
    hashed, which is every outcome decided before the bytes are read.
    """
    resolved = Path(model_path)

    if manifest is None:
        return REASON_MANIFEST_MISSING, None
    if not manifest:
        return REASON_MANIFEST_UNREADABLE, None

    seal_keys = keys if keys is not None else resolve_seal_keys()
    if seal_keys is None:
        return REASON_KEY_UNAVAILABLE, None

    seal_reason = verify_seal(
        _payload_of(manifest), manifest.get("seal"), seal_keys, verifier=verifier
    )
    if seal_reason != REASON_VERIFIED:
        return seal_reason, None

    entries = manifest.get("entries") or {}
    entry = entries.get(resolved.name)
    if not isinstance(entry, dict) or not entry.get("sha256"):
        return REASON_UNPINNED, None

    try:
        digest = compute_digest(resolved, chunk_size=chunk_size)
    except OSError as exc:
        logger.warning("Model file unreadable for hashing: %s", exc)
        return REASON_FILE_UNREADABLE, None

    if not _hmac.compare_digest(digest, str(entry.get("sha256"))):
        return REASON_DIGEST_MISMATCH, digest

    return REASON_VERIFIED, digest


def verify_model(
    model_path: Path | str,
    *,
    manifest_path: Path | str | None = None,
    mode: str | None = None,
    config: dict[str, Any] | None = None,
    keys: SealKeys | None = None,
    verifier: Callable[[bytes, bytes, bytes], bool] | None = None,
    chunk_size: int = _CHUNK,
) -> ProvenanceDecision:
    """Classify a model and apply the enforcement policy to the outcome."""
    resolved = Path(model_path)
    enforcement = enforcement_mode(mode=mode, config=config)
    reason, digest = classify_model(
        resolved,
        load_manifest(manifest_path),
        keys=keys,
        verifier=verifier,
        chunk_size=chunk_size,
    )
    allowed = decide(reason, enforcement)

    if reason != REASON_VERIFIED:
        log = logger.error if not allowed else logger.warning
        log(
            "Model provenance %s for %s: %s",
            "REFUSED" if not allowed else "not verified (observing)",
            resolved.name,
            reason,
        )

    return ProvenanceDecision(
        allowed=allowed,
        reason=reason,
        enforcement=enforcement,
        model=resolved.name,
        digest=digest,
    )


def guard_model_load(
    model_path: Path | str,
    *,
    manifest_path: Path | str | None = None,
    mode: str | None = None,
    config: dict[str, Any] | None = None,
) -> ProvenanceDecision:
    """The gate the in-process load seam calls. Raises to refuse.

    Raises:
        ProvenanceRefusal: When the model must not be loaded.
    """
    decision = verify_model(
        model_path, manifest_path=manifest_path, mode=mode, config=config
    )
    if not decision.allowed:
        raise ProvenanceRefusal(decision)
    return decision
