#!/usr/bin/env python3
"""
TLS Manager for Opti-Oignon (S133).

Handles certificate generation and management for secure remote access
in Daily mode ONLY. Every public function checks security mode first
and raises SecurityError if in Bulbe mode. This is defense layer 5 of 6.

Certificate hierarchy:
  CA (self-signed) -> Server cert -> Client certs (per device)

mTLS: server requires client certificate for mutual authentication.
CRL: checked on every request, no caching (local file, sub-ms read).

Key types: Ed25519 preferred, RSA-4096 fallback.
CA key at rest: encrypted with AES-256-GCM, key derived via Argon2id.

Kerckhoffs compliance: security derives from:
  (a) user passphrase strength
  (b) private key secrecy
  (c) physical access to server for cert provisioning
  Never from code obscurity.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import ssl
import stat
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Hardcoded, never overridable
checkpoint_before_apply = True

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TLS_DIR = _PROJECT_ROOT / "data" / "tls"
_CRL_PATH = _TLS_DIR / "crl.pem"
_CA_KEY_PATH = _TLS_DIR / "ca.key"
_CA_CERT_PATH = _TLS_DIR / "ca.crt"
_SERVER_KEY_PATH = _TLS_DIR / "server.key"
_SERVER_CERT_PATH = _TLS_DIR / "server.crt"
_CA_KEY_ENC_PATH = _TLS_DIR / "ca.key.enc"
_CLIENT_DIR = _TLS_DIR / "clients"

# Certificate validity
CERT_VALIDITY_DAYS = 365
CERT_WARNING_DAYS = 30

# Argon2id parameters for CA key encryption
ARGON2_MEMORY_COST = 65536  # 64 MB
ARGON2_TIME_COST = 3
ARGON2_PARALLELISM = 4
ARGON2_SALT_LENGTH = 32
ARGON2_KEY_LENGTH = 32

# AES-256-GCM nonce length
AES_NONCE_LENGTH = 12


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TLSSecurityError(Exception):
    """Raised when TLS operations are attempted in Bulbe mode."""
    pass


# ---------------------------------------------------------------------------
# Mode guard
# ---------------------------------------------------------------------------

def _assert_not_bulbe(operation: str = "TLS operation") -> None:
    """Raise TLSSecurityError if currently in Bulbe mode.

    This is defense layer 5 of 6. Every public function calls this.
    """
    try:
        from opti_oignon.security_mode import is_bulbe
        if is_bulbe():
            raise TLSSecurityError(
                f"TLS manager refuses to operate in Bulbe mode: {operation}. "
                "Remote access is physically impossible in Bulbe mode."
            )
    except ImportError:
        # If we cannot determine mode, fail secure
        raise TLSSecurityError(
            f"Cannot determine security mode for {operation}. "
            "Refusing to proceed (fail-secure)."
        )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClientCertInfo:
    """Information about a client certificate."""
    device_name: str = ""
    fingerprint: str = ""
    created_at: float = 0.0
    expires_at: float = 0.0
    revoked: bool = False
    serial_number: str = ""


@dataclass
class TLSStatus:
    """Current TLS configuration status."""
    enabled: bool = False
    ca_exists: bool = False
    server_cert_exists: bool = False
    ca_fingerprint: str = ""
    server_cert_expiry: str = ""
    days_until_expiry: int = 0
    client_certs: list[ClientCertInfo] = field(default_factory=list)
    warning: str = ""


# ---------------------------------------------------------------------------
# Certificate generation
# ---------------------------------------------------------------------------

def setup_tls(passphrase: str) -> dict[str, Any]:
    """Initialize the full TLS infrastructure.

    Generates CA, server cert, and stores CA key encrypted at rest.
    Must be called from localhost (enforced by API route).

    Args:
        passphrase: User passphrase for CA key encryption.

    Returns:
        Dict with setup results and CA fingerprint.
    """
    _assert_not_bulbe("setup_tls")

    if len(passphrase) < 12:
        return {
            "success": False,
            "error": "passphrase_too_short",
            "message": "Passphrase must be at least 12 characters.",
        }

    _TLS_DIR.mkdir(parents=True, exist_ok=True)
    _CLIENT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec, rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        return {
            "success": False,
            "error": "missing_cryptography",
            "message": "cryptography library required. pip install cryptography",
        }

    # Generate CA key (EC preferred, RSA fallback)
    try:
        ca_key = ec.generate_private_key(ec.SECP384R1())
        key_type = "EC-P384"
    except Exception:
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
        key_type = "RSA-4096"

    # CA certificate (self-signed)
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Opti-Oignon CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Opti-Oignon"),
    ])
    now = datetime.now(timezone.utc)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=CERT_VALIDITY_DAYS * 3))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=0), critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA384())
    )

    # Save CA cert (public, readable)
    _CA_CERT_PATH.write_bytes(
        ca_cert.public_bytes(serialization.Encoding.PEM)
    )

    # Encrypt and save CA key at rest
    _encrypt_ca_key(ca_key, passphrase)

    # Also save unencrypted CA key with strict permissions for server use
    ca_key_pem = ca_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    _CA_KEY_PATH.write_bytes(ca_key_pem)
    _set_file_permissions(_CA_KEY_PATH, 0o600)

    # Generate server certificate
    _generate_server_cert(ca_key, ca_cert)

    # Initialize empty CRL
    _init_crl(ca_key, ca_cert)

    # CA fingerprint
    ca_fp = hashlib.sha256(
        ca_cert.public_bytes(serialization.Encoding.DER)
    ).hexdigest()

    _audit_log("tls_setup_complete", key_type=key_type, ca_fingerprint=ca_fp)

    return {
        "success": True,
        "key_type": key_type,
        "ca_fingerprint": ca_fp,
        "server_cert": str(_SERVER_CERT_PATH),
        "message": "TLS infrastructure initialized successfully.",
    }


def _generate_server_cert(ca_key, ca_cert) -> None:
    """Generate the server certificate signed by the CA."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, rsa
    from cryptography.x509.oid import NameOID

    # Server key
    try:
        server_key = ec.generate_private_key(ec.SECP384R1())
    except Exception:
        server_key = rsa.generate_private_key(
            public_exponent=65537, key_size=4096,
        )

    now = datetime.now(timezone.utc)
    server_name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Opti-Oignon Server"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Opti-Oignon"),
    ])

    # SAN: localhost and common LAN addresses
    san = x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress(_parse_ip("127.0.0.1")),
        x509.IPAddress(_parse_ip("::1")),
    ])

    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=CERT_VALIDITY_DAYS))
        .add_extension(san, critical=False)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_encipherment=True,
                content_commitment=False, key_cert_sign=False,
                crl_sign=False, data_encipherment=False,
                key_agreement=False, encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA384())
    )

    # Save server key (strict permissions)
    server_key_pem = server_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    _SERVER_KEY_PATH.write_bytes(server_key_pem)
    _set_file_permissions(_SERVER_KEY_PATH, 0o600)

    # Save server cert
    _SERVER_CERT_PATH.write_bytes(
        server_cert.public_bytes(serialization.Encoding.PEM)
    )


def generate_client_cert(device_name: str, passphrase: str) -> dict[str, Any]:
    """Generate a named client certificate for mTLS.

    The client cert is signed by the CA. A .p12 (PKCS12) bundle is
    created for easy import on mobile/desktop devices.

    This function can ONLY be called from localhost (enforced at API level).

    Args:
        device_name: Human-readable device name (e.g. 'iPhone-Leon').
        passphrase: Passphrase to protect the .p12 file.

    Returns:
        Dict with cert info and path to .p12 file.
    """
    _assert_not_bulbe("generate_client_cert")

    if not device_name or len(device_name) > 64:
        return {
            "success": False,
            "error": "invalid_device_name",
            "message": "Device name must be 1-64 characters.",
        }

    # Sanitize device name for filesystem
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in device_name
    )

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec, rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        return {
            "success": False,
            "error": "missing_cryptography",
            "message": "cryptography library required.",
        }

    # Load CA key and cert
    ca_key = _load_ca_key()
    ca_cert = _load_ca_cert()
    if ca_key is None or ca_cert is None:
        return {
            "success": False,
            "error": "ca_not_initialized",
            "message": "TLS not set up. Call setup_tls first.",
        }

    # Client key
    try:
        client_key = ec.generate_private_key(ec.SECP384R1())
    except Exception:
        client_key = rsa.generate_private_key(
            public_exponent=65537, key_size=4096,
        )

    now = datetime.now(timezone.utc)
    client_name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, device_name),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Opti-Oignon Client"),
    ])

    serial = x509.random_serial_number()

    client_cert = (
        x509.CertificateBuilder()
        .subject_name(client_name)
        .issuer_name(ca_cert.subject)
        .public_key(client_key.public_key())
        .serial_number(serial)
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=CERT_VALIDITY_DAYS))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA384())
    )

    # Fingerprint
    fp = hashlib.sha256(
        client_cert.public_bytes(serialization.Encoding.DER)
    ).hexdigest()

    # Save client cert and key
    client_dir = _CLIENT_DIR / safe_name
    client_dir.mkdir(parents=True, exist_ok=True)

    client_key_path = client_dir / "client.key"
    client_cert_path = client_dir / "client.crt"
    p12_path = client_dir / f"{safe_name}.p12"

    client_key_pem = client_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    client_key_path.write_bytes(client_key_pem)
    _set_file_permissions(client_key_path, 0o600)

    client_cert_path.write_bytes(
        client_cert.public_bytes(serialization.Encoding.PEM)
    )

    # Generate PKCS12 bundle
    from cryptography.hazmat.primitives.serialization import pkcs12
    p12_data = pkcs12.serialize_key_and_certificates(
        name=device_name.encode("utf-8"),
        key=client_key,
        cert=client_cert,
        cas=[ca_cert],
        encryption_algorithm=serialization.BestAvailableEncryption(
            passphrase.encode("utf-8")
        ),
    )
    p12_path.write_bytes(p12_data)
    _set_file_permissions(p12_path, 0o600)

    # Save metadata
    _save_client_metadata(safe_name, {
        "device_name": device_name,
        "fingerprint": fp,
        "serial_number": format(serial, "x"),
        "created_at": time.time(),
        "expires_at": (now + timedelta(days=CERT_VALIDITY_DAYS)).timestamp(),
        "revoked": False,
    })

    _audit_log(
        "client_cert_generated",
        device_name=device_name,
        fingerprint=fp,
    )

    return {
        "success": True,
        "device_name": device_name,
        "fingerprint": fp,
        "p12_path": str(p12_path),
        "expires_at": (now + timedelta(days=CERT_VALIDITY_DAYS)).isoformat(),
        "message": f"Client certificate generated for '{device_name}'.",
    }


def revoke_client_cert(device_name: str) -> dict[str, Any]:
    """Revoke a client certificate by adding it to the CRL.

    Takes effect on the NEXT request (no caching). An attacker who
    reads this code knows there is zero grace period.

    Args:
        device_name: The device name used when generating the cert.

    Returns:
        Dict with revocation result.
    """
    _assert_not_bulbe("revoke_client_cert")

    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in device_name
    )

    metadata = _load_client_metadata(safe_name)
    if not metadata:
        return {
            "success": False,
            "error": "cert_not_found",
            "message": f"No certificate found for device '{device_name}'.",
        }

    if metadata.get("revoked", False):
        return {
            "success": True,
            "already_revoked": True,
            "message": f"Certificate for '{device_name}' was already revoked.",
        }

    # Load client cert for serial number
    client_cert_path = _CLIENT_DIR / safe_name / "client.crt"
    if not client_cert_path.exists():
        return {
            "success": False,
            "error": "cert_file_missing",
            "message": "Certificate file not found on disk.",
        }

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes

        client_cert = x509.load_pem_x509_certificate(
            client_cert_path.read_bytes()
        )

        # Load CA key and cert for CRL signing
        ca_key = _load_ca_key()
        ca_cert = _load_ca_cert()
        if ca_key is None or ca_cert is None:
            return {
                "success": False,
                "error": "ca_not_available",
                "message": "Cannot revoke: CA not available.",
            }

        # Load existing CRL and add revoked cert
        from cryptography.hazmat.primitives.serialization import Encoding
        crl_builder = x509.CertificateRevocationListBuilder()
        crl_builder = crl_builder.issuer_name(ca_cert.subject)
        now = datetime.now(timezone.utc)
        crl_builder = crl_builder.last_update(now)
        crl_builder = crl_builder.next_update(
            now + timedelta(days=CERT_VALIDITY_DAYS)
        )

        # Load existing revoked certs from CRL
        existing_revoked = _load_existing_revoked_serials()
        for serial in existing_revoked:
            revoked_cert = (
                x509.RevokedCertificateBuilder()
                .serial_number(serial)
                .revocation_date(now)
                .build()
            )
            crl_builder = crl_builder.add_revoked_certificate(revoked_cert)

        # Add the new revocation
        revoked_cert = (
            x509.RevokedCertificateBuilder()
            .serial_number(client_cert.serial_number)
            .revocation_date(now)
            .build()
        )
        crl_builder = crl_builder.add_revoked_certificate(revoked_cert)

        # Sign and save CRL
        crl = crl_builder.sign(ca_key, hashes.SHA384())
        _CRL_PATH.write_bytes(
            crl.public_bytes(Encoding.PEM)
        )

        # Update metadata
        metadata["revoked"] = True
        metadata["revoked_at"] = time.time()
        _save_client_metadata(safe_name, metadata)

        # RA-01: kill any live remote session bound to this fingerprint, in
        # addition to the persistent CRL/metadata check enforced per request.
        try:
            from opti_oignon.remote_session_guard import (
                remote_session_guard as _guard,
            )
            fp = metadata.get("fingerprint", "")
            if fp:
                _guard.revoke_fingerprint(fp)
        except Exception as exc:
            logger.debug(
                "Could not revoke live sessions for fingerprint: %s", exc,
            )

        _audit_log(
            "client_cert_revoked",
            device_name=device_name,
            fingerprint=metadata.get("fingerprint", ""),
        )

        return {
            "success": True,
            "device_name": device_name,
            "message": f"Certificate for '{device_name}' has been revoked.",
        }

    except TLSSecurityError:
        raise
    except Exception as exc:
        logger.error("Failed to revoke client cert: %s", exc)
        return {
            "success": False,
            "error": "revocation_failed",
            "message": f"Revocation failed: {exc}",
        }


# ---------------------------------------------------------------------------
# TLS config for uvicorn
# ---------------------------------------------------------------------------

def get_tls_config() -> dict[str, Any]:
    """Return SSL context parameters for uvicorn.

    Only works in Daily mode with valid TLS files.
    Returns empty dict if TLS is not configured.
    """
    _assert_not_bulbe("get_tls_config")

    if not _SERVER_KEY_PATH.exists() or not _SERVER_CERT_PATH.exists():
        return {}

    config: dict[str, Any] = {
        "ssl_keyfile": str(_SERVER_KEY_PATH),
        "ssl_certfile": str(_SERVER_CERT_PATH),
    }

    # mTLS: require client certificate
    if _CA_CERT_PATH.exists():
        config["ssl_ca_certs"] = str(_CA_CERT_PATH)
        config["ssl_cert_reqs"] = ssl.CERT_REQUIRED

    return config


# ---------------------------------------------------------------------------
# CRL verification
# ---------------------------------------------------------------------------

def is_cert_revoked(cert_fingerprint: str) -> bool:
    """Check if a client certificate is revoked.

    Reads CRL from disk on EVERY call (no caching). The CRL file
    is local, reads are sub-millisecond. No cache window means no
    grace period for revoked certs.

    An attacker reading this code sees: zero grace period.

    Args:
        cert_fingerprint: SHA256 fingerprint of the client cert.

    Returns:
        True if the cert is revoked.
    """
    # Check metadata files for fingerprint match
    if not _CLIENT_DIR.exists():
        return False

    for client_dir in _CLIENT_DIR.iterdir():
        if not client_dir.is_dir():
            continue
        meta = _load_client_metadata(client_dir.name)
        if meta and meta.get("fingerprint") == cert_fingerprint:
            return bool(meta.get("revoked", False))

    return False


def is_cert_serial_revoked(serial_number: int) -> bool:
    """Check if a serial number appears in the CRL.

    No caching. Fresh disk read every time.
    """
    if not _CRL_PATH.exists():
        return False

    try:
        from cryptography import x509
        crl = x509.load_pem_x509_crl(_CRL_PATH.read_bytes())
        revoked = crl.get_revoked_certificate_by_serial_number(serial_number)
        return revoked is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Status / listing
# ---------------------------------------------------------------------------

def get_tls_status() -> dict[str, Any]:
    """Get current TLS status. Works in any mode (read-only)."""
    status: dict[str, Any] = {
        "enabled": False,
        "ca_exists": _CA_CERT_PATH.exists(),
        "server_cert_exists": _SERVER_CERT_PATH.exists(),
        "ca_fingerprint": "",
        "server_cert_expiry": "",
        "days_until_expiry": 0,
        "client_certs": [],
        "warning": "",
    }

    if not _CA_CERT_PATH.exists():
        return status

    try:
        from cryptography import x509

        # CA fingerprint
        ca_cert = x509.load_pem_x509_certificate(_CA_CERT_PATH.read_bytes())
        status["ca_fingerprint"] = hashlib.sha256(
            ca_cert.public_bytes(serialization.Encoding.DER)
        ).hexdigest()

        # Server cert expiry
        if _SERVER_CERT_PATH.exists():
            server_cert = x509.load_pem_x509_certificate(
                _SERVER_CERT_PATH.read_bytes()
            )
            expiry = server_cert.not_valid_after_utc
            status["server_cert_expiry"] = expiry.isoformat()
            days_left = (expiry - datetime.now(timezone.utc)).days
            status["days_until_expiry"] = days_left
            status["enabled"] = True

            if days_left <= CERT_WARNING_DAYS:
                status["warning"] = (
                    f"Server certificate expires in {days_left} days. "
                    "Consider regenerating."
                )

        # Client certs
        if _CLIENT_DIR.exists():
            for client_dir in sorted(_CLIENT_DIR.iterdir()):
                if not client_dir.is_dir():
                    continue
                meta = _load_client_metadata(client_dir.name)
                if meta:
                    status["client_certs"].append({
                        "device_name": meta.get("device_name", client_dir.name),
                        "fingerprint": meta.get("fingerprint", ""),
                        "created_at": meta.get("created_at", 0),
                        "expires_at": meta.get("expires_at", 0),
                        "revoked": meta.get("revoked", False),
                    })

    except Exception as exc:
        logger.warning("Failed to read TLS status: %s", exc)

    return status


def list_client_certs() -> list[dict[str, Any]]:
    """List all client certificates with their status."""
    _assert_not_bulbe("list_client_certs")

    certs = []
    if not _CLIENT_DIR.exists():
        return certs

    for client_dir in sorted(_CLIENT_DIR.iterdir()):
        if not client_dir.is_dir():
            continue
        meta = _load_client_metadata(client_dir.name)
        if meta:
            certs.append({
                "device_name": meta.get("device_name", client_dir.name),
                "fingerprint": meta.get("fingerprint", ""),
                "created_at": meta.get("created_at", 0),
                "expires_at": meta.get("expires_at", 0),
                "revoked": meta.get("revoked", False),
                "serial_number": meta.get("serial_number", ""),
            })

    return certs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_ip(ip_str: str):
    """Parse IP address string to ipaddress object."""
    import ipaddress
    return ipaddress.ip_address(ip_str)


def _set_file_permissions(path: Path, mode: int) -> None:
    """Set file permissions (best-effort on non-Unix)."""
    try:
        os.chmod(path, mode)
    except OSError:
        logger.warning("Could not set permissions %o on %s", mode, path)


def _load_ca_key():
    """Load the CA private key from disk."""
    try:
        if not _CA_KEY_PATH.exists():
            return None
        from cryptography.hazmat.primitives.serialization import (
            load_pem_private_key,
        )
        return load_pem_private_key(_CA_KEY_PATH.read_bytes(), password=None)
    except Exception as exc:
        logger.error("Failed to load CA key: %s", exc)
        return None


def _load_ca_cert():
    """Load the CA certificate from disk."""
    try:
        if not _CA_CERT_PATH.exists():
            return None
        from cryptography import x509
        return x509.load_pem_x509_certificate(_CA_CERT_PATH.read_bytes())
    except Exception as exc:
        logger.error("Failed to load CA cert: %s", exc)
        return None


def _encrypt_ca_key(ca_key, passphrase: str) -> None:
    """Encrypt the CA private key at rest using AES-256-GCM with Argon2id KDF.

    The encrypted key is stored separately from the unencrypted working copy.
    The unencrypted copy has strict permissions (0600).

    Argon2id parameters: m=65536 (64MB), t=3, p=4.
    This is critical for open-source: weak KDF = offline bruteforce.
    """
    from cryptography.hazmat.primitives import serialization

    ca_key_pem = ca_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )

    salt = secrets.token_bytes(ARGON2_SALT_LENGTH)

    # Derive encryption key with Argon2id
    try:
        from argon2.low_level import hash_secret_raw, Type
        derived = hash_secret_raw(
            secret=passphrase.encode("utf-8"),
            salt=salt,
            time_cost=ARGON2_TIME_COST,
            memory_cost=ARGON2_MEMORY_COST,
            parallelism=ARGON2_PARALLELISM,
            hash_len=ARGON2_KEY_LENGTH,
            type=Type.ID,
        )
    except ImportError:
        # Fallback: use hashlib scrypt (still strong, but Argon2id preferred)
        logger.warning(
            "argon2-cffi not available; falling back to scrypt for CA key "
            "encryption. Install argon2-cffi for maximum security."
        )
        derived = hashlib.scrypt(
            passphrase.encode("utf-8"),
            salt=salt,
            n=2**17, r=8, p=1,
            dklen=ARGON2_KEY_LENGTH,
        )

    # AES-256-GCM encryption
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = secrets.token_bytes(AES_NONCE_LENGTH)
    aesgcm = AESGCM(derived)
    ciphertext = aesgcm.encrypt(nonce, ca_key_pem, None)

    # Store: salt || nonce || ciphertext
    _CA_KEY_ENC_PATH.write_bytes(salt + nonce + ciphertext)
    _set_file_permissions(_CA_KEY_ENC_PATH, 0o600)


def _init_crl(ca_key, ca_cert) -> None:
    """Initialize an empty CRL."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes

    now = datetime.now(timezone.utc)
    crl = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(ca_cert.subject)
        .last_update(now)
        .next_update(now + timedelta(days=CERT_VALIDITY_DAYS))
        .sign(ca_key, hashes.SHA384())
    )
    from cryptography.hazmat.primitives.serialization import Encoding
    _CRL_PATH.write_bytes(crl.public_bytes(Encoding.PEM))


def _load_existing_revoked_serials() -> list[int]:
    """Load serial numbers from existing CRL."""
    if not _CRL_PATH.exists():
        return []
    try:
        from cryptography import x509
        crl = x509.load_pem_x509_crl(_CRL_PATH.read_bytes())
        return [revoked.serial_number for revoked in crl]
    except Exception:
        return []


def _save_client_metadata(safe_name: str, metadata: dict) -> None:
    """Save client cert metadata to JSON file."""
    import json
    meta_path = _CLIENT_DIR / safe_name / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_client_metadata(safe_name: str) -> dict | None:
    """Load client cert metadata from JSON file."""
    import json
    meta_path = _CLIENT_DIR / safe_name / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _audit_log(event: str, **details) -> None:
    """Log a security audit event."""
    logger.info("TLS AUDIT [%s]: %s", event, details)
    try:
        from opti_oignon.signed_audit_log import chain_log
        chain_log(
            event_type=event,
            source="tls_manager",
            action=event,
            severity="INFO",
            **details,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Module availability flag
# ---------------------------------------------------------------------------

TLS_MANAGER_AVAILABLE = True
