#!/usr/bin/env python3
"""Audit chain external anchor: export and verification.

An anchor externalizes the current tip of the append-only audit chain so
that an operator can later prove the chain was not rewritten behind their
back. Three offline export formats carry the same anchor:

1. **JSON file** -- downloadable, pretty-printed, signed.
2. **QR code**   -- PNG encoding the signed anchor as compact JSON, so a
   scanned code can be pasted straight into verification.
3. **Clipboard** -- human-readable text embedding every signed field.

Signing follows the chain's own truncation-anchor discipline: the MAC key
is derived from the master encryption key (domain-separated) via
``SignedAuditLog._anchor_secret()``, together with a non-secret key id
that binds the signature to the secret that produced it. The security of
an anchor therefore rests on the key and the hash chain, never on the
format or any constant readable in this public source tree. When no
master key is configured the anchor degrades, explicitly, to an
accidental-corruption checksum (advisory only).

Verification of a re-presented anchor proves three distinct things: the
signature verifies under the same key id; the chain is internally intact,
with any broken link DETECTED AND LOCALIZED to its entry id via a full
walk; and the chain hash at the anchored height still equals the anchored
tip (growth beyond the anchor is fine, truncation and rewrites are not).

All functions are pure helpers that accept a ``SignedAuditLog`` instance
(or compatible object) and return data. They never touch global state and
never touch the network.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from opti_oignon.signed_audit_log import _anchor_mac

logger = logging.getLogger(__name__)

# House rule: modules that could feed an apply path checkpoint first.
# This module only reads the chain; the flag documents the posture.
checkpoint_before_apply = True

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Format 2: the signed payload carries the key id of the secret-derived
# anchor key. Format 1 anchors (no key id, MAC keyed on a public constant)
# are still readable for hash comparison but are never reported as
# authenticated: their signing material is part of the public source.
_ANCHOR_VERSION = 2

_PAYLOAD_FIELDS = (
    "anchor_version",
    "chain_tip_hash",
    "entry_count",
    "key_id",
    "timestamp",
    "version",
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AnchorPayload:
    """Serialisable anchor payload (the exact bytes that get signed)."""

    chain_tip_hash: str
    entry_count: int
    timestamp: float
    version: str
    key_id: str
    anchor_version: int = _ANCHOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        """Canonical compact JSON: the MAC input."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


@dataclass
class SignedAnchor:
    """Anchor payload plus its MAC, in the flat wire shape."""

    payload: AnchorPayload
    hmac_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Flat dict: payload fields and the MAC at top level.

        The same shape is used by the JSON file, the QR content and the
        verification request body, so any exported anchor can be pasted
        back into verification unchanged.
        """
        data = self.payload.to_dict()
        data["hmac_sha256"] = self.hmac_sha256
        return data

    def to_json(self) -> str:
        """Canonical compact JSON of the flat wire shape."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


@dataclass
class VerificationResult:
    """Outcome of verifying a re-presented anchor against the chain."""

    match: bool
    details: str
    current_entry_count: int
    anchor_entry_count: int
    current_tip_hash: str
    anchor_tip_hash: str
    hmac_valid: bool | None
    signature_scheme: str = "absent"
    chain_valid: bool = True
    first_divergent_entry: int | None = None
    divergence: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("extras", None)
        data.update(self.extras)
        return data


# ---------------------------------------------------------------------------
# Signing helpers
# ---------------------------------------------------------------------------


def _chain_anchor_secret(chain: Any) -> tuple[bytes | None, str]:
    """Return the chain's (anchor_key, key_id), degrading to unkeyed."""
    getter = getattr(chain, "_anchor_secret", None)
    if getter is None:
        return None, "nokey"
    try:
        key, key_id = getter()
        return key, str(key_id)
    except Exception:
        return None, "nokey"


def _build_anchor_payload(
    chain: Any, app_version: str, key_id: str
) -> AnchorPayload:
    """Extract the current chain tip and build an AnchorPayload.

    ``chain`` must expose ``entry_count() -> int`` and
    ``_get_tip_hash() -> str`` (the SignedAuditLog interface).
    """
    count = chain.entry_count()
    tip_hash = chain._get_tip_hash() if count > 0 else ""
    return AnchorPayload(
        chain_tip_hash=tip_hash,
        entry_count=count,
        timestamp=time.time(),
        version=app_version,
        key_id=key_id,
    )


def _build_signed_anchor(chain: Any, app_version: str) -> SignedAnchor:
    """Build a signed anchor from the current chain state.

    The MAC is computed over the canonical payload JSON with the chain's
    secret-derived anchor key; without a master key it degrades to a
    plain checksum whose key id says so ("nokey").
    """
    anchor_key, key_id = _chain_anchor_secret(chain)
    payload = _build_anchor_payload(chain, app_version, key_id)
    mac = _anchor_mac(payload.to_json(), anchor_key)
    return SignedAnchor(payload=payload, hmac_sha256=mac)


# ---------------------------------------------------------------------------
# Export: QR code
# ---------------------------------------------------------------------------


def _qr_png_from_signed(signed: SignedAnchor) -> bytes:
    """Render a QR PNG whose content is the signed anchor's wire JSON.

    Single build: the QR content and any returned anchor metadata are
    byte-identical, and a scanned QR verifies without transformation.
    """
    if signed.payload.entry_count == 0:
        raise RuntimeError("Audit chain is empty -- nothing to export.")

    content = signed.to_json()

    try:
        import qrcode  # type: ignore[import-untyped]
    except ImportError as exc:
        # An optional capability that is merely absent has to say so in those
        # words. Left bare, the error escaping here reads to an operator --
        # and to a test report -- exactly like a defect on the anchor path,
        # which is the one thing this module must never be ambiguous about.
        raise ImportError(
            "QR rendering requires qrcode, which is not installed. "
            "Install with: pip install 'opti-oignon[anchor]'"
        ) from exc

    qr = qrcode.QRCode(
        version=None,  # auto-size
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(content)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_anchor_qr_png(chain: Any, app_version: str) -> bytes:
    """Generate a QR code PNG containing the SIGNED anchor as JSON.

    Returns raw PNG bytes.

    Raises:
        ImportError: If ``qrcode`` or ``PIL`` is not installed.
        RuntimeError: If the chain is empty.
    """
    return _qr_png_from_signed(_build_signed_anchor(chain, app_version))


def generate_anchor_qr_base64(chain: Any, app_version: str) -> dict[str, Any]:
    """Generate a QR code and return it as base64 with its exact content.

    Returns a dict with ``qr_base64``, ``anchor`` (the flat signed dict
    encoded in the QR) and ``content_type``.
    """
    signed = _build_signed_anchor(chain, app_version)
    png_bytes = _qr_png_from_signed(signed)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return {
        "qr_base64": b64,
        "anchor": signed.to_dict(),
        "content_type": "image/png",
    }


# ---------------------------------------------------------------------------
# Export: JSON file
# ---------------------------------------------------------------------------


def generate_anchor_json(chain: Any, app_version: str) -> dict[str, Any]:
    """Generate a signed JSON anchor for USB / file export.

    Returns the flat signed anchor dict (payload fields plus MAC), the
    same shape the verification endpoint accepts.
    """
    signed = _build_signed_anchor(chain, app_version)
    if signed.payload.entry_count == 0:
        raise RuntimeError("Audit chain is empty -- nothing to export.")
    return signed.to_dict()


def generate_anchor_json_bytes(chain: Any, app_version: str) -> bytes:
    """Generate the signed anchor as pretty-printed JSON bytes."""
    data = generate_anchor_json(chain, app_version)
    return json.dumps(data, indent=2, sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------------------
# Export: clipboard text
# ---------------------------------------------------------------------------

_TEXT_HEADER = "OPTI-OIGNON AUDIT ANCHOR"
_TEXT_FIELD_MAP = {
    "Chain Tip Hash": ("chain_tip_hash", str),
    "Entry Count": ("entry_count", int),
    "Timestamp": ("timestamp", float),
    "Version": ("version", str),
    "Key Id": ("key_id", str),
    "Anchor Version": ("anchor_version", int),
    "HMAC-SHA256": ("hmac_sha256", str),
}


def generate_anchor_text(chain: Any, app_version: str) -> str:
    """Generate a human-readable anchor string for clipboard copy.

    Every signed field is present verbatim, so the text round-trips
    through :func:`parse_anchor_text` back to the exact wire anchor.
    """
    signed = _build_signed_anchor(chain, app_version)
    if signed.payload.entry_count == 0:
        raise RuntimeError("Audit chain is empty -- nothing to export.")
    wire = signed.to_dict()

    width = max(len(label) for label in _TEXT_FIELD_MAP)
    lines = [_TEXT_HEADER, "=" * len(_TEXT_HEADER)]
    for label, (key, _cast) in _TEXT_FIELD_MAP.items():
        value = wire[key]
        if key == "timestamp":
            value = repr(float(value))
        lines.append(f"{label:<{width}} : {value}")
    return "\n".join(lines)


def parse_anchor_text(text: str) -> dict[str, Any]:
    """Recover the flat anchor dict from clipboard text.

    Inverse of :func:`generate_anchor_text`. Unknown lines are ignored;
    known fields are cast back to their wire types.
    """
    out: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, _, raw = line.partition(":")
        label = label.strip()
        raw = raw.strip()
        entry = _TEXT_FIELD_MAP.get(label)
        if entry is None:
            continue
        key, cast = entry
        try:
            out[key] = cast(raw)
        except (TypeError, ValueError):
            out[key] = raw
    return out


# ---------------------------------------------------------------------------
# Export: single-build bundle
# ---------------------------------------------------------------------------


def export_anchor_bundle(chain: Any, app_version: str) -> dict[str, Any]:
    """Export ONE anchor rendered in all three formats.

    The signed anchor is built exactly once; the JSON bytes, the QR
    content/PNG and the clipboard text are all renderings of that same
    anchor, so their canonical digests are identical by construction.

    Returns a dict with ``anchor`` (flat signed dict), ``json_bytes``,
    ``qr_content``, ``qr_png``, ``text``.
    """
    signed = _build_signed_anchor(chain, app_version)
    if signed.payload.entry_count == 0:
        raise RuntimeError("Audit chain is empty -- nothing to export.")
    wire = signed.to_dict()

    json_bytes = json.dumps(wire, indent=2, sort_keys=True).encode("utf-8")
    qr_content = signed.to_json()
    qr_png = _qr_png_from_signed(signed)

    width = max(len(label) for label in _TEXT_FIELD_MAP)
    lines = [_TEXT_HEADER, "=" * len(_TEXT_HEADER)]
    for label, (key, _cast) in _TEXT_FIELD_MAP.items():
        value = wire[key]
        if key == "timestamp":
            value = repr(float(value))
        lines.append(f"{label:<{width}} : {value}")
    text = "\n".join(lines)

    return {
        "anchor": wire,
        "json_bytes": json_bytes,
        "qr_content": qr_content,
        "qr_png": qr_png,
        "text": text,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _rebuild_payload_json(anchor_data: dict[str, Any]) -> str | None:
    """Reconstruct the canonical payload JSON that was signed.

    Returns None when required fields are missing or malformed.
    """
    try:
        payload = AnchorPayload(
            chain_tip_hash=str(anchor_data.get("chain_tip_hash", "")),
            entry_count=int(anchor_data.get("entry_count", 0)),
            timestamp=float(anchor_data.get("timestamp", 0)),
            version=str(anchor_data.get("version", "")),
            key_id=str(anchor_data.get("key_id", "")),
            anchor_version=int(anchor_data.get("anchor_version", 0)),
        )
    except (TypeError, ValueError):
        return None
    return payload.to_json()


def verify_anchor(
    chain: Any,
    anchor_data: dict[str, Any],
    app_version: str,
) -> VerificationResult:
    """Verify a re-presented anchor against the current chain state.

    ``anchor_data`` is the flat wire shape (payload fields plus optional
    ``hmac_sha256``); a nested ``{"payload": ..., "hmac_sha256": ...}``
    shape from older exports is flattened on entry.

    Three independent guarantees are checked, in order:

    1. **Signature.** With a key id matching the chain's current anchor
       secret, the MAC must verify -- a failure is a net rejection. A
       different key id (rotated key, another install) makes the
       signature unverifiable here, which is reported but is not itself
       a tamper verdict. Anchors without a key id predate secret-derived
       signing; their MAC material is public, so they are never reported
       as authenticated.
    2. **Chain integrity, localized.** The chain is walked end to end
       (``verify_chain``); the first broken link is reported by entry id,
       including whether it falls inside or after the anchored history.
    3. **Anchored tip.** The chain hash at the anchored height must equal
       the anchored tip. Growth past the anchor is fine; truncation and
       internally-consistent rewrites are not (a rewrite is localized to
       "at or before" the anchored height -- a single tip cannot pinpoint
       a consistent rewrite more precisely).
    """
    if "payload" in anchor_data and isinstance(anchor_data["payload"], dict):
        flat = dict(anchor_data["payload"])
        if "hmac_sha256" in anchor_data:
            flat["hmac_sha256"] = anchor_data["hmac_sha256"]
        anchor_data = flat

    anchor_tip = str(anchor_data.get("chain_tip_hash", ""))
    try:
        anchor_count = int(anchor_data.get("entry_count", 0))
    except (TypeError, ValueError):
        anchor_count = 0
    anchor_mac_value = anchor_data.get("hmac_sha256")
    anchor_key_id = str(anchor_data.get("key_id", "") or "")

    current_count = chain.entry_count()
    current_tip = chain._get_tip_hash() if current_count > 0 else ""

    def _result(**kw: Any) -> VerificationResult:
        base: dict[str, Any] = dict(
            current_entry_count=current_count,
            anchor_entry_count=anchor_count,
            current_tip_hash=current_tip,
            anchor_tip_hash=anchor_tip,
        )
        base.update(kw)
        return VerificationResult(**base)

    # -- 1. Signature --------------------------------------------------------
    hmac_valid: bool | None = None
    signature_scheme = "absent"
    signature_note = "Anchor carries no signature."

    if anchor_mac_value:
        if not anchor_key_id:
            signature_scheme = "unauthenticated-legacy"
            signature_note = (
                "Anchor predates secret-derived signing (no key id); its "
                "signature material is public and is not authenticated."
            )
        else:
            anchor_key, current_key_id = _chain_anchor_secret(chain)
            if anchor_key_id != current_key_id:
                signature_scheme = "keyed-foreign"
                signature_note = (
                    f"Anchor was signed under key id {anchor_key_id!r} but "
                    f"this chain's key id is {current_key_id!r} (rotated key "
                    "or another install); the signature cannot be verified "
                    "here."
                )
            else:
                payload_json = _rebuild_payload_json(anchor_data)
                if payload_json is None:
                    hmac_valid = False
                else:
                    expected = _anchor_mac(payload_json, anchor_key)
                    hmac_valid = _consteq(expected, str(anchor_mac_value))
                if anchor_key is not None:
                    signature_scheme = "keyed"
                    signature_note = (
                        "Signature verified with the secret-derived anchor "
                        "key." if hmac_valid else ""
                    )
                else:
                    signature_scheme = "unkeyed-advisory"
                    signature_note = (
                        "No master key is configured: the anchor checksum "
                        "only detects accidental corruption, not an "
                        "adversary."
                    )
                if hmac_valid is False:
                    return _result(
                        match=False,
                        details=(
                            "Signature is INVALID: the anchor fails to "
                            "verify under its own key id. The anchor blob "
                            "was tampered with or corrupted."
                        ),
                        hmac_valid=False,
                        signature_scheme=signature_scheme,
                        divergence="invalid-signature",
                    )

    # -- 2. Chain integrity, localized ---------------------------------------
    chain_valid = True
    first_broken: int | None = None
    verify_walk = getattr(chain, "verify_chain", None)
    if callable(verify_walk):
        chain_valid, first_broken, _total = verify_walk()
    if not chain_valid:
        where = (
            "inside the anchored history"
            if first_broken is not None and first_broken <= anchor_count
            else "after the anchored history"
        )
        return _result(
            match=False,
            details=(
                f"Chain integrity is BROKEN at entry {first_broken} "
                f"({where}): the first divergent link is localized there. "
                f"{signature_note}".strip()
            ),
            hmac_valid=hmac_valid,
            signature_scheme=signature_scheme,
            chain_valid=False,
            first_divergent_entry=first_broken,
            divergence="broken-chain",
        )

    # -- 3. Anchored tip ------------------------------------------------------
    if current_count < anchor_count:
        return _result(
            match=False,
            details=(
                f"Chain has been TRUNCATED: the anchor records "
                f"{anchor_count} entries but the chain currently has only "
                f"{current_count}. {signature_note}".strip()
            ),
            hmac_valid=hmac_valid,
            signature_scheme=signature_scheme,
            divergence="truncated",
        )

    if current_count > anchor_count:
        historical_tip = _get_hash_at_entry(chain, anchor_count)
        if historical_tip == anchor_tip:
            return _result(
                match=True,
                details=(
                    f"Anchor verified. Chain has grown from {anchor_count} "
                    f"to {current_count} entries; the hash at the anchored "
                    f"height still matches. {signature_note}".strip()
                ),
                hmac_valid=hmac_valid,
                signature_scheme=signature_scheme,
            )
        return _result(
            match=False,
            details=(
                f"MISMATCH: the chain is internally consistent but its hash "
                f"at the anchored height ({anchor_count}) differs from the "
                f"anchored tip. History was REWRITTEN at or before entry "
                f"{anchor_count}; a single anchored tip cannot localize a "
                f"consistent rewrite more precisely. "
                f"{signature_note}".strip()
            ),
            hmac_valid=hmac_valid,
            signature_scheme=signature_scheme,
            divergence="rewritten-history",
        )

    if current_tip == anchor_tip:
        return _result(
            match=True,
            details=(
                f"Anchor verified. Chain tip matches at {current_count} "
                f"entries. {signature_note}".strip()
            ),
            hmac_valid=hmac_valid,
            signature_scheme=signature_scheme,
        )
    return _result(
        match=False,
        details=(
            "MISMATCH: entry counts are equal but the chain tip differs "
            "from the anchored tip while the chain is internally "
            "consistent. History was REWRITTEN at or before entry "
            f"{anchor_count}. {signature_note}".strip()
        ),
        hmac_valid=hmac_valid,
        signature_scheme=signature_scheme,
        divergence="rewritten-history",
    )


def _consteq(a: str, b: str) -> bool:
    """Constant-time string comparison."""
    import hmac as _hmac

    return _hmac.compare_digest(a, b)


def _get_hash_at_entry(chain: Any, entry_id: int) -> str:
    """Retrieve the entry_hash at a specific entry id.

    Uses the chain's DB path through the encrypted-connection helper.
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


__all__ = [
    "AnchorPayload",
    "SignedAnchor",
    "VerificationResult",
    "export_anchor_bundle",
    "generate_anchor_json",
    "generate_anchor_json_bytes",
    "generate_anchor_qr_base64",
    "generate_anchor_qr_png",
    "generate_anchor_text",
    "parse_anchor_text",
    "verify_anchor",
]
