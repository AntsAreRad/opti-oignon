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

PQC_AVAILABLE = False
_PQC_ALGORITHM = "Dilithium3"  # ML-DSA-65 in liboqs naming

try:
    import oqs  # type: ignore[import-untyped]
    # Verify the algorithm is actually available in this build
    _sig_test = oqs.Signature(_PQC_ALGORITHM)
    del _sig_test
    PQC_AVAILABLE = True
    logger.info("PQC signatures available (liboqs: %s)", _PQC_ALGORITHM)
except ImportError:
    logger.info(
        "liboqs-python not installed. PQC signatures disabled. "
        "Install with: pip install liboqs-python"
    )
except Exception as exc:
    logger.warning("PQC signature init failed: %s", exc)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_KEYPAIR_PATH = _PROJECT_ROOT / "data" / ".pqc_keypair"


def _load_pqc_config() -> dict[str, Any]:
    """Load PQC configuration from security.yaml."""
    try:
        import yaml
        cfg_path = Path(__file__).parent / "config" / "security.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("pqc", {})
    except Exception:
        pass
    return {}


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
        "config_enabled": bool(cfg.get("backup_signatures", False)),
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
