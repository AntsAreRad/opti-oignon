#!/usr/bin/env python3
"""
Audit Chain External Anchor Export & Verification (S146).

Provides multiple ways to externalize the audit chain tip so that an
operator can later prove the chain has not been tampered with:

1. **QR Code** — PNG image encoding the chain tip as JSON.
2. **USB / JSON file** — Downloadable JSON with HMAC signature.
3. **Clipboard text** — Human-readable plain-text anchor.
4. **Verification** — Import a previously exported anchor and compare
   against the current chain state.

All functions are pure helpers that accept a ``SignedAuditLog`` instance
(or compatible object) and return data.  They never touch global state
directly — the API layer passes the singleton.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ANCHOR_VERSION = 1
_HMAC_KEY_MATERIAL = b"opti-oignon-audit-anchor-v1"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AnchorPayload:
    """Serialisable anchor payload."""

    chain_tip_hash: str
    entry_count: int
    timestamp: float
    version: str
    anchor_version: int = _ANCHOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


@dataclass
class SignedAnchor:
    """Anchor payload with HMAC signature for tamper detection."""

    payload: AnchorPayload
    hmac_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.payload.to_dict(),
            "hmac_sha256": self.hmac_sha256,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


@dataclass
class VerificationResult:
    """Result of anchor verification against current chain."""

    match: bool
    details: str
    current_entry_count: int
    anchor_entry_count: int
    current_tip_hash: str
    anchor_tip_hash: str
    hmac_valid: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# HMAC helpers
# ---------------------------------------------------------------------------


def _compute_anchor_hmac(payload_json: str) -> str:
    """Compute HMAC-SHA256 over the canonical JSON payload."""
    return hmac.new(
        _HMAC_KEY_MATERIAL,
        payload_json.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _verify_anchor_hmac(payload_json: str, expected_hmac: str) -> bool:
    """Verify HMAC-SHA256 of an anchor payload."""
    computed = _compute_anchor_hmac(payload_json)
    return hmac.compare_digest(computed, expected_hmac)


# ---------------------------------------------------------------------------
# Core: build anchor from chain
# ---------------------------------------------------------------------------


def _build_anchor_payload(chain: Any, app_version: str) -> AnchorPayload:
    """Extract current chain tip and build an AnchorPayload.

    ``chain`` must expose ``.entry_count()`` -> int and
    ``._get_tip_hash()`` -> str  (SignedAuditLog interface).
    """
    count = chain.entry_count()
    tip_hash = chain._get_tip_hash() if count > 0 else ""
    return AnchorPayload(
        chain_tip_hash=tip_hash,
        entry_count=count,
        timestamp=time.time(),
        version=app_version,
    )


def _build_signed_anchor(chain: Any, app_version: str) -> SignedAnchor:
    """Build a signed anchor from the current chain state."""
    payload = _build_anchor_payload(chain, app_version)
    payload_json = payload.to_json()
    mac = _compute_anchor_hmac(payload_json)
    return SignedAnchor(payload=payload, hmac_sha256=mac)


# ---------------------------------------------------------------------------
# Goal 1: QR Code Export
# ---------------------------------------------------------------------------


def generate_anchor_qr_png(chain: Any, app_version: str) -> bytes:
    """Generate a QR code PNG containing the anchor payload as JSON.

    Returns raw PNG bytes.

    Raises:
        ImportError: If ``qrcode`` or ``PIL`` is not installed.
        RuntimeError: If chain is empty.
    """
    payload = _build_anchor_payload(chain, app_version)
    return _qr_png_from_payload(payload)


def _qr_png_from_payload(payload: AnchorPayload) -> bytes:
    """Render a QR PNG from an already-built payload.

    EXP-04 (S194): single payload build, so the QR content and any
    returned payload metadata are byte-identical.
    """
    if payload.entry_count == 0:
        raise RuntimeError("Audit chain is empty — nothing to export.")

    payload_json = payload.to_json()

    import qrcode  # type: ignore[import-untyped]
    from qrcode.image.pil import PilImage  # type: ignore[import-untyped]

    qr = qrcode.QRCode(
        version=None,  # auto-size
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(payload_json)
    qr.make(fit=True)

    img: PilImage = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_anchor_qr_base64(chain: Any, app_version: str) -> dict[str, Any]:
    """Generate QR code and return as base64 string with metadata.

    Returns dict with ``qr_base64``, ``payload``, ``content_type``.
    The returned payload is the exact payload encoded in the QR.
    """
    payload = _build_anchor_payload(chain, app_version)
    png_bytes = _qr_png_from_payload(payload)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return {
        "qr_base64": b64,
        "payload": payload.to_dict(),
        "content_type": "image/png",
    }


# ---------------------------------------------------------------------------
# Goal 2: USB / JSON File Export
# ---------------------------------------------------------------------------


def generate_anchor_json(chain: Any, app_version: str) -> dict[str, Any]:
    """Generate a signed JSON anchor for USB / file export.

    Returns the full signed anchor dict (payload + HMAC).
    """
    signed = _build_signed_anchor(chain, app_version)
    if signed.payload.entry_count == 0:
        raise RuntimeError("Audit chain is empty — nothing to export.")
    return signed.to_dict()


def generate_anchor_json_bytes(chain: Any, app_version: str) -> bytes:
    """Generate signed anchor as pretty-printed JSON bytes (for download)."""
    data = generate_anchor_json(chain, app_version)
    return json.dumps(data, indent=2, sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------------------
# Goal 3: Clipboard / Plain-text Anchor
# ---------------------------------------------------------------------------


def generate_anchor_text(chain: Any, app_version: str) -> str:
    """Generate a human-readable anchor string for clipboard copy.

    Format::

        OPTI-OIGNON AUDIT ANCHOR
        ========================
        Chain Tip Hash : <hash>
        Entry Count    : <n>
        Timestamp      : <ISO 8601>
        Version        : <version>
        HMAC-SHA256    : <hmac>

    """
    signed = _build_signed_anchor(chain, app_version)
    payload = signed.payload

    from datetime import datetime, timezone

    ts_str = datetime.fromtimestamp(
        payload.timestamp, tz=timezone.utc,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "OPTI-OIGNON AUDIT ANCHOR",
        "========================",
        f"Chain Tip Hash : {payload.chain_tip_hash}",
        f"Entry Count    : {payload.entry_count}",
        f"Timestamp      : {ts_str}",
        f"Version        : {payload.version}",
        f"HMAC-SHA256    : {signed.hmac_sha256}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Goal 4: Anchor Verification
# ---------------------------------------------------------------------------


def verify_anchor(
    chain: Any,
    anchor_data: dict[str, Any],
    app_version: str,
) -> VerificationResult:
    """Verify an imported anchor against the current chain state.

    ``anchor_data`` should be a dict with at least ``chain_tip_hash`` and
    ``entry_count``.  If ``hmac_sha256`` is present, HMAC integrity is
    also checked.

    Verification logic:
    - HMAC check (if HMAC present in anchor)
    - Entry count comparison
    - Chain tip hash comparison (only if counts match)
    """
    # Extract anchor fields
    anchor_tip = anchor_data.get("chain_tip_hash", "")
    anchor_count = int(anchor_data.get("entry_count", 0))
    anchor_hmac = anchor_data.get("hmac_sha256")

    # Current chain state
    current_count = chain.entry_count()
    current_tip = chain._get_tip_hash() if current_count > 0 else ""

    # HMAC verification (if present)
    hmac_valid: bool | None = None
    if anchor_hmac:
        # Reconstruct the payload without the HMAC for verification
        payload_fields = {
            k: v for k, v in anchor_data.items() if k != "hmac_sha256"
        }
        # Rebuild canonical AnchorPayload
        try:
            test_payload = AnchorPayload(
                chain_tip_hash=str(payload_fields.get("chain_tip_hash", "")),
                entry_count=int(payload_fields.get("entry_count", 0)),
                timestamp=float(payload_fields.get("timestamp", 0)),
                version=str(payload_fields.get("version", "")),
                anchor_version=int(
                    payload_fields.get("anchor_version", _ANCHOR_VERSION)
                ),
            )
            hmac_valid = _verify_anchor_hmac(test_payload.to_json(), anchor_hmac)
        except (ValueError, TypeError):
            hmac_valid = False

    # If HMAC is present and invalid, report immediately
    if hmac_valid is False:
        return VerificationResult(
            match=False,
            details="HMAC signature is invalid — anchor may have been tampered with.",
            current_entry_count=current_count,
            anchor_entry_count=anchor_count,
            current_tip_hash=current_tip,
            anchor_tip_hash=anchor_tip,
            hmac_valid=False,
        )

    # Count comparison
    if current_count < anchor_count:
        return VerificationResult(
            match=False,
            details=(
                f"Chain has been TRUNCATED: anchor records {anchor_count} entries "
                f"but chain currently has only {current_count}."
            ),
            current_entry_count=current_count,
            anchor_entry_count=anchor_count,
            current_tip_hash=current_tip,
            anchor_tip_hash=anchor_tip,
            hmac_valid=hmac_valid,
        )

    if current_count > anchor_count:
        # Chain has grown — need to walk back to find the entry at anchor_count
        # and compare its hash
        historical_tip = _get_hash_at_entry(chain, anchor_count)
        if historical_tip == anchor_tip:
            return VerificationResult(
                match=True,
                details=(
                    f"Anchor verified. Chain has grown from {anchor_count} to "
                    f"{current_count} entries. Historical tip matches."
                ),
                current_entry_count=current_count,
                anchor_entry_count=anchor_count,
                current_tip_hash=current_tip,
                anchor_tip_hash=anchor_tip,
                hmac_valid=hmac_valid,
            )
        else:
            return VerificationResult(
                match=False,
                details=(
                    f"MISMATCH: chain has grown to {current_count} entries but "
                    f"hash at entry {anchor_count} does not match anchor."
                ),
                current_entry_count=current_count,
                anchor_entry_count=anchor_count,
                current_tip_hash=current_tip,
                anchor_tip_hash=anchor_tip,
                hmac_valid=hmac_valid,
            )

    # Same count — direct tip comparison
    if current_tip == anchor_tip:
        return VerificationResult(
            match=True,
            details=f"Anchor verified. Chain tip matches ({current_count} entries).",
            current_entry_count=current_count,
            anchor_entry_count=anchor_count,
            current_tip_hash=current_tip,
            anchor_tip_hash=anchor_tip,
            hmac_valid=hmac_valid,
        )
    else:
        return VerificationResult(
            match=False,
            details=(
                "MISMATCH: entry counts are equal but chain tip hashes differ. "
                "The chain may have been rewritten."
            ),
            current_entry_count=current_count,
            anchor_entry_count=anchor_count,
            current_tip_hash=current_tip,
            anchor_tip_hash=anchor_tip,
            hmac_valid=hmac_valid,
        )


def _get_hash_at_entry(chain: Any, entry_id: int) -> str:
    """Retrieve the entry_hash at a specific entry ID.

    Uses the chain's DB connection to fetch a single row.
    """
    from opti_oignon.db_utils import safe_connect

    conn = safe_connect(chain._db_path, check_same_thread=False)
    try:
        row = conn.execute(
            "SELECT entry_hash FROM audit_chain WHERE id = ?",
            (entry_id,),
        ).fetchone()
        return row[0] if row else ""
    finally:
        conn.close()
