#!/usr/bin/env python3
"""
Post-Quantum Cryptographic Signatures for Opti-Oignon (S129).

Provides ML-DSA-65 (Dilithium) digital signatures for backup integrity
verification, using liboqs-python when available.

Features:
  - ML-DSA-65 keypair generation, signing, and verification
  - Feature-flagged: graceful degradation if liboqs is not installed
  - Key persistence in data/.pqc_keypair (chmod 600)
  - Classical HMAC-SHA512 fallback is handled by the caller (backup_manager)

Configuration: config/security.yaml > pqc > backup_signatures

Usage::

    from opti_oignon.pqc_signatures import (
        PQC_AVAILABLE,
        generate_pqc_keypair,
        sign_backup,
        verify_backup,
        load_pqc_keypair,
    )

    if PQC_AVAILABLE:
        pub, priv = generate_pqc_keypair()
        sig = sign_backup(data, priv)
        ok = verify_backup(data, sig, pub)
"""

from __future__ import annotations

import base64
import json
import logging
import stat
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------

# Mechanism names, in preference order. ML-DSA-65 is the FIPS 204 name;
# Dilithium3 is the pre-standardisation name that older liboqs builds carry.
# They are NOT the same algorithm and their signatures do not interverify, so
# the mechanism that ACTUALLY resolved travels with every artefact rather than
# being assumed from a constant.
#
# A hardcoded name is a primitive waiting to die: the library renames a
# mechanism at standardisation, the constant stops resolving, and the signing
# primitive goes out under the operator's feet. Resolving against the installed
# build is what makes that survivable, and the contract suite refuses a build
# whose preferred names resolve to nothing.
_MECHANISM_PREFERENCE = ("ML-DSA-65", "Dilithium3")

PQC_AVAILABLE = False
PQC_MECHANISM: str | None = None

# WHY the primitive is unavailable, in one actionable line; None when it is
# available. A log line is not a reason: nothing above it can read a log. The
# posture and the startup surface carry this, so an operator can tell an absent
# optional package from a signing primitive that died in place -- two different
# situations with two different remedies, which reported identically until now.
PQC_UNAVAILABLE_REASON: str | None = None


def _resolve_mechanism(module: Any) -> str | None:
    """The first preferred mechanism the INSTALLED liboqs actually offers.

    Returns None when the build offers none of them -- which is a dead signing
    primitive, not a missing optional dependency.
    """
    try:
        offered = set(module.get_enabled_sig_mechanisms())
    except Exception:
        offered = set()
    for name in _MECHANISM_PREFERENCE:
        if offered and name not in offered:
            continue
        try:
            probe = module.Signature(name)
            del probe
            return name
        except Exception:
            continue
    return None


try:
    import oqs  # type: ignore[import-untyped]

    PQC_MECHANISM = _resolve_mechanism(oqs)
    if PQC_MECHANISM is None:
        try:
            _offered = ", ".join(oqs.get_enabled_sig_mechanisms()) or "(nothing)"
        except Exception:  # noqa: BLE001 - an unusable library is a reason too
            _offered = "(the build would not say)"
        PQC_UNAVAILABLE_REASON = (
            f"liboqs is installed but offers none of "
            f"{', '.join(_MECHANISM_PREFERENCE)}. It offers: {_offered}."
        )
        # liboqs is HERE and offers nothing this build can use. That is not an
        # absent optional dependency: it is the signing primitive dying in
        # place. Record signing will refuse and provenance falls back to a
        # symmetric MAC, which anyone holding the key can forge -- so this is
        # CRITICAL, and pqc_posture() carries it to the startup surface. It
        # must never pass for "PQC simply was not configured".
        logger.critical(
            "liboqs is installed but offers none of %s. Post-quantum signing "
            "is UNAVAILABLE: record signing will refuse, and provenance falls "
            "back to a symmetric MAC that is not publicly verifiable.",
            ", ".join(_MECHANISM_PREFERENCE),
        )
    else:
        PQC_AVAILABLE = True
        logger.info("PQC signatures available (liboqs mechanism: %s)", PQC_MECHANISM)
except ImportError:
    PQC_UNAVAILABLE_REASON = (
        "liboqs-python is not installed. Install with: "
        "pip install 'opti-oignon[pqc]'"
    )
    # Not info. Under Bulbe, and wherever the operator asked for signing, this
    # is the root of trust being absent, and the boot will refuse on it.
    logger.warning("PQC signatures unavailable -- %s", PQC_UNAVAILABLE_REASON)
except Exception as exc:  # pragma: no cover - defensive
    PQC_UNAVAILABLE_REASON = f"PQC signature init failed: {exc}"
    logger.critical("PQC signature init failed: %s", exc)

# The resolved mechanism is what every call site uses. Keeping the private name
# means the key envelope, the signer and the verifier all speak of the same
# mechanism, and it is the one that resolved rather than the one that was hoped
# for.
# It is the one that RESOLVED, or nothing. The old fallback to the first
# preferred name meant a host with no usable library still announced ML-DSA-65
# from its status surface -- the mechanism that was hoped for, which the comment
# above forbids in as many words. Every signing path is already guarded by
# `if not PQC_AVAILABLE: raise`, so None can reach no call to oqs; it can only
# reach the envelope and the status, which is exactly where the truth belongs.
_PQC_ALGORITHM: str | None = PQC_MECHANISM


class PQCUnavailable(RuntimeError):
    """Post-quantum signing was ASKED FOR and cannot be provided."""


def pqc_requested() -> bool:
    """The operator's INTENT, read from configuration alone.

    Deliberately blind to runtime availability. ``is_pqc_enabled`` answers "is
    it on", and it answers False both when post-quantum signing was never asked
    for AND when it was asked for and could not be provided. Those are not the
    same thing: the second is a broken promise. Collapsing them into one boolean
    is exactly how a dead primitive degrades in silence.
    """
    cfg = _load_pqc_config()
    if cfg is None:
        # Present and unreadable: the intent is UNKNOWN. The strict reading is
        # the only one that cannot be wrong, and the alternative -- reading it
        # as "never asked for" -- is precisely what disarmed the refusal below.
        return True
    return bool(cfg.get("backup_signatures", False))


def _bulbe() -> bool:
    """Is this host a fortress?

    security_mode owns the fail-secure determination of the mode itself (an
    unknown mode is already Bulbe there). A module that cannot even be IMPORTED
    is a machinery failure, not a mode verdict, and this function must not
    manufacture a boot refusal out of a broken import -- that would be a denial
    of service on ourselves with no security bought.
    """
    try:
        from opti_oignon.security_mode import get_current_mode

        return get_current_mode() == "bulbe"
    except Exception as exc:  # noqa: BLE001 - a broken import is not a verdict
        logger.error("the security mode could not be determined: %s", exc)
        return False


def pqc_required() -> bool:
    """Is the primitive REQUIRED here, whatever the operator asked for?

    Two ways to require it. The operator asks (``pqc_requested``). Or the mode
    is Bulbe -- and Bulbe is a physical constraint, not a policy. The socket
    bind is not configurable under Bulbe; it is physical. The root of trust
    cannot be less than that: a fortress does not ask politely for it in a
    config file, and one with no signing key has nothing left to be a fortress
    with. Whoever wants to run without the primitive runs Daily.
    """
    return pqc_requested() or _bulbe()


def pqc_posture() -> dict[str, Any]:
    """What was asked for, what is available, and whether they disagree."""
    requested = pqc_requested()
    required = pqc_required()
    return {
        "requested": requested,
        "required": required,
        "available": PQC_AVAILABLE,
        "mechanism": PQC_MECHANISM,
        "reason": PQC_UNAVAILABLE_REASON,
        "degraded": required and not PQC_AVAILABLE,
    }


def assert_pqc_posture() -> None:
    """Refuse a broken promise: asked for, and not there.

    A symmetric MAC substituted for a signature is not a weaker signature -- it
    is a different security property. A signature is publicly verifiable against
    a public key; a MAC is forgeable by anyone holding the shared secret. When
    the operator asked for post-quantum signing and the primitive is absent, the
    honest answer is a refusal, never a quiet substitution.
    """
    if pqc_required() and not PQC_AVAILABLE:
        raise PQCUnavailable(
            "Post-quantum backup signatures are enabled in configuration, but "
            "no usable liboqs mechanism resolved "
            f"(tried: {', '.join(_MECHANISM_PREFERENCE)}). Refusing to "
            "substitute a symmetric MAC for a signature. Install or repair "
            "liboqs, or turn the setting off deliberately."
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_KEYPAIR_PATH = _PROJECT_ROOT / "data" / ".pqc_keypair"


def _load_pqc_config() -> dict[str, Any] | None:
    """PQC configuration, or None when security.yaml cannot be READ.

    An ABSENT file is a default: the operator did not configure signing, and
    that is a documented choice. A file that is PRESENT and unreadable is not a
    default -- it is an unknown, and the two must never collapse into the same
    empty mapping.

    They did, and the consequence was that the whole refusal mechanism switched
    off. ``pqc_requested`` gates ``assert_pqc_posture``; an empty mapping made
    the intent read False; so a truncated write, a disk error or a YAML typo was
    enough to let a forgeable symmetric MAC be substituted for a signature,
    without one word anywhere. A configuration read that FAILS OPEN disarms the
    very refusal it gates.
    """
    cfg_path = Path(__file__).parent / "config" / "security.yaml"
    if not cfg_path.exists():
        return {}

    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001 - unreadable is a verdict, not a shrug
        logger.critical(
            "security.yaml is present and could not be read (%s). The "
            "post-quantum signing intent cannot be determined; treating it as "
            "REQUESTED, which is the only reading that cannot be wrong.", exc,
        )
        return None

    if data is None:
        return {}  # an empty file is an empty configuration, not a broken one
    if not isinstance(data, dict):
        logger.critical(
            "security.yaml does not contain a mapping. The post-quantum "
            "signing intent cannot be determined; treating it as REQUESTED."
        )
        return None

    section = data.get("pqc", {})
    if not isinstance(section, dict):
        logger.critical(
            "security.yaml has a 'pqc' key that is not a mapping. The "
            "post-quantum signing intent cannot be determined; treating it as "
            "REQUESTED."
        )
        return None
    return section


def is_pqc_enabled() -> bool:
    """Check if PQC backup signatures are enabled in config.

    PQC is enabled only if:
      1. liboqs is available (PQC_AVAILABLE == True)
      2. config/security.yaml > pqc > backup_signatures == true (default: false)

    Returns:
        True if PQC signing should be used for backups.
    """
    if not PQC_AVAILABLE:
        return False
    cfg = _load_pqc_config()
    if cfg is None:
        # The policy could not be read and the primitive IS there. Signing more
        # than was asked for is never the risk; signing less than was asked for,
        # in silence, is the whole of it.
        return True
    return bool(cfg.get("backup_signatures", False))


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def generate_pqc_keypair() -> tuple[bytes, bytes]:
    """Generate a new ML-DSA-65 (Dilithium) keypair.

    Returns:
        Tuple of (public_key, private_key) as raw bytes.

    Raises:
        RuntimeError: If liboqs is not available.
    """
    if not PQC_AVAILABLE:
        raise RuntimeError(
            "PQC not available. Install liboqs-python: pip install liboqs-python"
        )

    import oqs  # type: ignore[import-untyped]
    sig = oqs.Signature(_PQC_ALGORITHM)
    public_key = sig.generate_keypair()
    private_key = sig.export_secret_key()

    logger.info(
        "Generated PQC keypair (%s): pub=%d bytes, priv=%d bytes",
        _PQC_ALGORITHM, len(public_key), len(private_key),
    )
    return public_key, private_key


# ---------------------------------------------------------------------------
# Signing and verification
# ---------------------------------------------------------------------------

def sign_bytes(data: bytes, private_key: bytes) -> bytes:
    """Sign arbitrary bytes with ML-DSA-65 (S205, VL-01).

    The generic signing primitive. ``sign_backup`` is a thin delegation kept
    for the historical backup call sites; the Veilid per-record signing
    (veilid/signing.py) calls this directly. Content-agnostic: the caller owns
    the canonical byte recipe.

    Args:
        data: The bytes to sign.
        private_key: ML-DSA-65 private key bytes.

    Returns:
        Signature bytes.

    Raises:
        RuntimeError: If liboqs is not available.
        ValueError: If signing fails.
    """
    if not PQC_AVAILABLE:
        raise RuntimeError("PQC not available")

    import oqs  # type: ignore[import-untyped]
    try:
        sig = oqs.Signature(_PQC_ALGORITHM, private_key)
        signature = sig.sign(data)
        logger.debug(
            "PQC signature generated: %d bytes over %d bytes of data",
            len(signature), len(data),
        )
        return signature
    except Exception as exc:
        raise ValueError(f"PQC signing failed: {exc}") from exc


def verify_bytes(data: bytes, signature: bytes, public_key: bytes) -> bool:
    """Verify an ML-DSA-65 signature over arbitrary bytes (S205, VL-01).

    The generic verification primitive; ``verify_backup`` delegates here.
    Never raises: an unavailable backend or a verification error returns
    False, the same defensive posture as the historical backup path.

    Args:
        data: The signed bytes.
        signature: The ML-DSA-65 signature to verify.
        public_key: ML-DSA-65 public key bytes.

    Returns:
        True if the signature is valid, False otherwise.
    """
    if not PQC_AVAILABLE:
        logger.warning("Cannot verify PQC signature: liboqs not available")
        return False

    import oqs  # type: ignore[import-untyped]
    try:
        sig = oqs.Signature(_PQC_ALGORITHM)
        is_valid = sig.verify(data, signature, public_key)
        return bool(is_valid)
    except Exception as exc:
        logger.warning("PQC signature verification failed: %s", exc)
        return False


def sign_backup(backup_bytes: bytes, private_key: bytes) -> bytes:
    """Sign backup data with ML-DSA-65.

    Thin delegation to :func:`sign_bytes` (S205): the primitive was always
    content-agnostic; the historical name is kept for the backup call sites.

    Args:
        backup_bytes: Serialized backup data to sign.
        private_key: ML-DSA-65 private key bytes.

    Returns:
        Signature bytes.

    Raises:
        RuntimeError: If liboqs is not available.
        ValueError: If signing fails.
    """
    return sign_bytes(backup_bytes, private_key)


def verify_backup(
    backup_bytes: bytes,
    signature: bytes,
    public_key: bytes,
) -> bool:
    """Verify a PQC signature on backup data.

    Thin delegation to :func:`verify_bytes` (S205). Never raises.

    Args:
        backup_bytes: The signed backup data.
        signature: The ML-DSA-65 signature to verify.
        public_key: ML-DSA-65 public key bytes.

    Returns:
        True if the signature is valid, False otherwise.
    """
    return verify_bytes(backup_bytes, signature, public_key)


# ---------------------------------------------------------------------------
# Key persistence
# ---------------------------------------------------------------------------

def save_pqc_keypair(
    public_key: bytes,
    private_key: bytes,
    path: Path | None = None,
) -> Path:
    """Save PQC keypair to disk with restricted permissions.

    Format (JSON, ASCII-safe via base64):
    {
        "algorithm": "Dilithium3",
        "public_key": "<base64>",
        "private_key": "<base64>"
    }

    File permissions set to 600 (owner read/write only).

    Args:
        public_key: ML-DSA-65 public key bytes.
        private_key: ML-DSA-65 private key bytes.
        path: Optional custom path (default: data/.pqc_keypair).

    Returns:
        Path where the keypair was saved.
    """
    fpath = path or _DEFAULT_KEYPAIR_PATH
    fpath.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "algorithm": _PQC_ALGORITHM,
        "public_key": base64.urlsafe_b64encode(public_key).decode("ascii"),
        "private_key": base64.urlsafe_b64encode(private_key).decode("ascii"),
    }

    fpath.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="ascii",
    )

    try:
        fpath.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        logger.warning("Could not set PQC keypair file permissions to 600: %s", fpath)

    logger.info("PQC keypair saved: %s (%s)", fpath, _PQC_ALGORITHM)
    return fpath


def load_pqc_keypair(
    path: Path | None = None,
) -> tuple[bytes, bytes]:
    """Load PQC keypair from disk.

    Args:
        path: Optional custom path (default: data/.pqc_keypair).

    Returns:
        Tuple of (public_key, private_key) as raw bytes.

    Raises:
        FileNotFoundError: If keypair file does not exist.
        ValueError: If the file format is invalid.
    """
    fpath = path or _DEFAULT_KEYPAIR_PATH
    if not fpath.exists():
        raise FileNotFoundError(f"PQC keypair file not found: {fpath}")

    try:
        raw = json.loads(fpath.read_text(encoding="ascii"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid PQC keypair file format: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("PQC keypair file must contain a JSON object")

    pub_b64 = raw.get("public_key")
    priv_b64 = raw.get("private_key")
    if not pub_b64 or not priv_b64:
        raise ValueError("PQC keypair file missing public_key or private_key")

    public_key = base64.urlsafe_b64decode(pub_b64)
    private_key = base64.urlsafe_b64decode(priv_b64)

    logger.debug(
        "Loaded PQC keypair from %s (algorithm=%s)",
        fpath, raw.get("algorithm", "unknown"),
    )
    return public_key, private_key


def pqc_keypair_exists(path: Path | None = None) -> bool:
    """Check if a PQC keypair file exists on disk.

    Args:
        path: Optional custom path.

    Returns:
        True if the keypair file exists.
    """
    fpath = path or _DEFAULT_KEYPAIR_PATH
    return fpath.is_file()


def delete_pqc_keypair(path: Path | None = None) -> bool:
    """Delete the PQC keypair file from disk.

    Args:
        path: Optional custom path.

    Returns:
        True if the file was deleted, False if it did not exist.
    """
    fpath = path or _DEFAULT_KEYPAIR_PATH
    if fpath.is_file():
        # Zero file contents before deletion (defense in depth)
        try:
            size = fpath.stat().st_size
            fpath.write_bytes(b"\x00" * size)
        except Exception:
            pass
        fpath.unlink()
        logger.info("PQC keypair deleted: %s", fpath)
        return True
    return False


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def get_pqc_status() -> dict[str, Any]:
    """Get comprehensive PQC status information.

    Returns:
        Dict with availability, config, key status, and algorithm info.
    """
    cfg = _load_pqc_config()
    keypair_path = _DEFAULT_KEYPAIR_PATH

    status: dict[str, Any] = {
        "available": PQC_AVAILABLE,
        "algorithm": _PQC_ALGORITHM,
        "reason": PQC_UNAVAILABLE_REASON,
        "config_readable": cfg is not None,
        "config_enabled": bool((cfg or {}).get("backup_signatures", False)),
        "effective_enabled": is_pqc_enabled(),
        "keypair_exists": keypair_path.is_file(),
        "keypair_path": str(keypair_path),
    }

    # Add key details if keypair exists
    if keypair_path.is_file():
        try:
            raw = json.loads(keypair_path.read_text(encoding="ascii"))
            pub_b64 = raw.get("public_key", "")
            priv_b64 = raw.get("private_key", "")
            status["key_algorithm"] = raw.get("algorithm", "unknown")
            status["public_key_size"] = len(base64.urlsafe_b64decode(pub_b64)) if pub_b64 else 0
            status["private_key_size"] = len(base64.urlsafe_b64decode(priv_b64)) if priv_b64 else 0
        except Exception:
            status["key_algorithm"] = "error"

    return status
