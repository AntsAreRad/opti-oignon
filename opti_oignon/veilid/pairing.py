#!/usr/bin/env python3
"""Pairing key exchange for Veilid sync (S182 Goal 1, Theme 4 / Veilid Sync).

The ceremony that introduces two of a user's own devices to each other. The
per-peer store (``peers.py``) holds, for each paired peer, a stable identity and
the peer's public routing key; the live transport (``transport.py``) reaches a
peer over a private route to that key. S178-S181 built the store, the engine, the
route, and the live transport that consume a paired peer; this module builds what
populates the store: a device generates a pairing payload that carries its public
material, and a peer accepts that payload to register the device so the transport
can reach it.

A pairing payload carries the device's public material: its stable identity
within the user's set, its public Veilid routing key (how a private route to it
is opened), since S205 (VL-01) optionally its signing PUBLIC key (the trust
root record verification resolves against), since S258 (PAIR-03) optionally
its device CLASS (``phone`` / ``desktop``: the joining device declares itself
so the accepting desktop records it at the accept seam, under a MONOTONE rule
that can never escalate a stored class), and an integrity check over those
public fields. The integrity check is a plain
SHA-256 over the canonical JSON of the public fields -- the same construction as a
record's content hash (``records.content_hash_for``). It is tamper detection at
the edge, not a secret: it catches a garbled or altered payload (a mistyped QR, a
truncated transcription) before the routing key is trusted, while the actual
authenticity of a peer is anchored in the cryptographic routing key itself, which
the user holds. There is no secret in the payload and none in this code.

Since S206 (PAIR-02) the ceremony is completed by a mutual confirmation: a short
comparison code derived from BOTH devices' public material (this device's payload
and the peer's), displayed on both screens, compared by the humans, and confirmed
on both sides before the registry entry activates. The derivation
(:func:`confirmation_code`) is order-normalized -- the code comes out identical on
both devices regardless of who built which payload -- and covers every public
field of both payloads, including the routing key and the signing public key:
exactly the lot-1 trust material a substituted payload would replace. The digest
is hardened with scrypt (stdlib, memory-hard, public parameters; Kerckhoffs), so
the code is expensive to grind, not merely short.

What the confirmation defends, stated honestly. A payload substituted in the
out-of-band channel (a wrong QR, a hijacked clipboard, an opportunistic relay)
makes the two displayed codes disagree, so a STATIC substitution -- one prepared
without knowledge of both true payloads -- is detected with probability
1 - 10**-8 (the code is 8 decimal digits). The residual is an ACTIVE
man-in-the-middle who sees both true payloads during the exchange and grinds a
field of its substitute payload (its peer_id is free text) until the truncated
code collides: with the scrypt parameters here (n=2**14, r=8, p=1, 16 MiB per
evaluation, roughly 130 ms on commodity hardware) the expected grind is 10**8
evaluations, on the order of 150 core-days of memory-hard work that must
complete within the minutes a human ceremony stays open -- out of reach short
of state-scale parallelism, and stated rather than hand-waved. What the
confirmation does NOT defend: a compromised device that displays a matching
code and lies; a human who confirms without comparing; peers registered before
PAIR-02, which are grandfathered as confirmed without a ceremony; the
device-class field of a FRESH pairing (S258, PAIR-03), which an active
in-channel substitution can flip without moving the code -- the code material
deliberately EXCLUDES the class so both devices recompute it identically
against a legacy peer and against a registry whose post-policy class may
lawfully differ from the declared one, while the integrity digest still
rejects a stripped or garbled class, the monotone rule keeps a re-pair from
ever escalating one, and the recorded class rides the pending surface for the
human to review next to the code; and a route
rotated between the two halves of an exchange, which makes the codes visibly
disagree (a closed, re-runnable failure, not a silent one). The ceremony
authenticates the EXCHANGE, not the humans or the devices' integrity.

Two properties matter, the same two the record encoding guarantees. The encode
side is pure and validating: ``build_pairing_payload`` raises ``ValueError`` on a
programmer error (an empty identity or key) and reaches into no store and opens no
socket. The decode side is defensive: ``parse_pairing_payload`` never raises into a
caller -- a non-mapping, a wrong format version or type, a missing or mistyped
field, or an integrity check that does not match the public material all yield
``None``. An incoming payload is data, not trusted input.

Pairing management is local-disk and permitted in any mode, like the peer store it
populates: generating this device's payload, reading a peer's payload, and
registering, labelling, or removing a peer all run under Bulbe as well as Daily.
What is Daily-only is moving records over the wire (a round, a served answer), and
that gate lives at the binding layer in the engine, the transport, and the
responder, not here. ``accept_pairing_payload`` populates the store through the
engine's ``register_peer``, an upsert that preserves the watermark on a re-pair, so
re-pairing a device with a rotated route never resets how far it has synced; the
engine records the registration in the hash-chain audit log.

Kerckhoffs: the ceremony is open. A peer is addressed by a public routing key the
user holds; the payload carries public material and an integrity check, never a
secret derived from the shape of the exchange. This module imports only the
standard library, so it collects in any environment and is loadable in isolation;
it duck-types the engine it registers through, so it stays transport-agnostic and
store-agnostic.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# The pairing payload wire format. Bumped only on an incompatible change; a
# payload that does not carry exactly this version is rejected on parse.
PAIRING_FORMAT_VERSION = 1

# The payload type tag; a payload that does not carry exactly this type is rejected.
PAIRING_TYPE = "veilid_pairing"

# S258 (PAIR-03): the wire-format device-class vocabulary -- the values a v1
# payload may DECLARE. The registry-side allowlist is ``peers.DEVICE_CLASSES``;
# the two are the same two words, pinned EQUAL by test (anti-drift) rather
# than by a runtime import, so this module keeps its stdlib-only,
# loadable-in-isolation property. The producer (``build_pairing_payload``)
# REJECTS anything outside this vocabulary (a programmer error); the parser
# accepts any non-empty string (forward compatibility: a future word rides
# the wire and its digest) and the APPLY seam normalises an unknown word to
# the least-trusted class.
PAIRING_DEVICE_CLASS_PHONE = "phone"
PAIRING_DEVICE_CLASS_DESKTOP = "desktop"
PAIRING_DEVICE_CLASSES: frozenset[str] = frozenset(
    {PAIRING_DEVICE_CLASS_PHONE, PAIRING_DEVICE_CLASS_DESKTOP}
)

# PAIR-02 (S206): the mutual-confirmation code derivation, ONE documented
# construction. The two devices' canonical public materials are sorted
# (order normalization), length-prefix framed, and fed to scrypt under a fixed
# domain-separation salt with public, deterministic parameters (Kerckhoffs: the
# recipe is open; there is no secret). n=2**14, r=8, p=1 costs 16 MiB and
# roughly 130 ms per evaluation on commodity hardware, which is what makes the
# truncated code expensive to grind (see the module docstring for the stated
# bound). The code is 8 decimal digits, displayed as two groups of four -- the
# established short-authentication-string shape humans compare reliably.
CONFIRM_CODE_SALT = b"oo-pairing-confirm-v1"
CONFIRM_SCRYPT_N = 2**14
CONFIRM_SCRYPT_R = 8
CONFIRM_SCRYPT_P = 1
CONFIRM_SCRYPT_MAXMEM = 64 * 1024 * 1024
CONFIRM_CODE_DIGITS = 8


@dataclass(frozen=True)
class ParsedPairing:
    """A parsed, integrity-checked pairing payload.

    Attributes:
        peer_id: The stable identity of the device that generated the payload,
            within this user's set.
        routing_key: That device's public Veilid routing key; how a private route
            to it is opened. Public by design (Kerckhoffs).
        signing_pub: That device's ML-DSA-65 signing PUBLIC key, base64url
            (S205, VL-01), or ``None`` for a pre-VL-01 payload that carries
            none. Public material, like the routing key; the integrity check
            covers it when present.
        device_class: The device class the payload DECLARED (S258, PAIR-03),
            or ``None`` for a payload that carries none. Carried AS PARSED
            (any non-empty string: the wire vocabulary is the apply seam's
            concern, never the parser's); the integrity check covers it when
            present.
    """

    peer_id: str
    routing_key: str
    signing_pub: Optional[str] = None
    device_class: Optional[str] = None


def pairing_canonical_material(
    peer_id: str,
    routing_key: str,
    signing_pub: Optional[str] = None,
    device_class: Optional[str] = None,
) -> str:
    """The canonical JSON of a payload's public fields (no integrity field).

    The single serialisation the integrity check is computed over: the format
    version, the type tag, the identity, the routing key, when present (S205,
    VL-01) the signing public key, and when present (S258, PAIR-03) the
    device class, as compact JSON with sorted keys, so it is independent of
    key order and stable across a JSON round-trip. The PAIR-02 confirmation
    code is computed over the SAME serialisation WITHOUT the class (callers
    pass class-less material; see :func:`confirmation_code` for why).
    Producer-side and validating: raises ``ValueError`` on a programmer error
    (an empty identity or key), like :func:`build_pairing_payload`; the
    device class is validated for SHAPE only (a non-empty string or ``None``)
    because the parser recomputes a foreign payload's digest with whatever
    word rode the wire -- the VOCABULARY is the builder's and the apply
    seam's concern. The defensive path for untrusted input is
    :func:`parse_pairing_payload`.
    """
    if not isinstance(peer_id, str) or not peer_id:
        raise ValueError("peer_id must be a non-empty string")
    if not isinstance(routing_key, str) or not routing_key:
        raise ValueError("routing_key must be a non-empty string")
    if signing_pub is not None and (
        not isinstance(signing_pub, str) or not signing_pub
    ):
        raise ValueError("signing_pub must be a non-empty string or None")
    if device_class is not None and (
        not isinstance(device_class, str) or not device_class
    ):
        raise ValueError("device_class must be a non-empty string or None")
    canonical: dict[str, Any] = {
        "v": PAIRING_FORMAT_VERSION,
        "type": PAIRING_TYPE,
        "peer_id": peer_id,
        "routing_key": routing_key,
    }
    if signing_pub is not None:
        canonical["signing_pub"] = signing_pub
    if device_class is not None:
        canonical["device_class"] = device_class
    return json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def pairing_integrity(
    peer_id: str,
    routing_key: str,
    signing_pub: Optional[str] = None,
    device_class: Optional[str] = None,
) -> str:
    """The integrity check for a pairing payload: SHA-256 over the public fields.

    Covers the format version, the type tag, the identity, the routing key,
    when the payload carries one (S205, VL-01) the signing public key, and
    when the payload carries one (S258, PAIR-03) the device class:
    the canonical PRESENT public fields, nothing secret. Computed over the one
    canonical serialisation (:func:`pairing_canonical_material`), so it is
    independent of key order and stable across a JSON round-trip. This is the
    same construction as a record's content hash; it detects a tampered or
    garbled payload, it is not an authenticator derived from a secret.

    The present-fields recipe is what the compat matrix rests on: a pre-VL-01
    payload (no signing key) keeps the historical four-field digest, so a new
    reader accepts old payloads unchanged; a payload that DOES carry a key --
    or, since S258, a device class -- folds it into the digest, so a tampered
    or stripped-in-transit field no longer matches -- at the deliberate cost
    that an OLD reader (recomputing its fixed fields) rejects a new payload.
    That failure is closed and visible during a rare human ceremony (upgrade,
    re-pair), which is the safe direction; the alternative -- leaving the new
    field OUTSIDE the integrity so old readers accept -- would ship the new
    trust material unprotected.
    """
    blob = pairing_canonical_material(
        peer_id, routing_key, signing_pub, device_class
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def confirmation_code(material_a: str, material_b: str) -> str:
    """The PAIR-02 mutual-confirmation code over two devices' public material.

    Each argument is one device's canonical material
    (:func:`pairing_canonical_material`) -- this device's own and the parsed
    peer's. The derivation is order-normalized: the two materials are sorted
    lexicographically before framing, so the code comes out IDENTICAL on both
    devices regardless of who built which payload. Framing is unambiguous
    (each part is prefixed with its UTF-8 byte length), the digest is
    ``hashlib.scrypt`` under the fixed public parameters above (stdlib,
    memory-hard, deterministic; no secret -- Kerckhoffs), and the code is the
    first 8 bytes reduced modulo 10**8, rendered as two groups of four decimal
    digits ("1234 5678"; the modulo bias over a 64-bit value is below 10**-11,
    negligible against the stated bounds).

    S258 (PAIR-03): the code material deliberately EXCLUDES the device class
    -- callers pass class-less canonical material on both sides. The code
    must recompute identically on both devices from material both can derive:
    a legacy joiner pins class-less material, and the accepting registry's
    post-policy class may lawfully differ from a payload's declared one (the
    fail-secure default, a normalisation, a monotone keep), so folding the
    class in would break the identical-recompute property the ceremony rests
    on. The residual, and its compensating controls, are stated in the module
    docstring.

    The collision bound, stated: the code space is 10**8, so a substitution
    prepared WITHOUT knowledge of both true payloads is detected with
    probability 1 - 10**-8. An active man-in-the-middle who sees both true
    payloads can grind a free field of its substitute (its peer_id) toward a
    colliding code; each attempt costs one scrypt evaluation (16 MiB, roughly
    130 ms on commodity hardware), so the expected grind is 10**8 evaluations
    -- on the order of 150 core-days of memory-hard work that must land inside
    the minutes a human ceremony stays open. That residual, and what the
    ceremony does and does not authenticate, are stated in the module
    docstring. Producer-side and validating: raises ``ValueError`` on empty or
    non-string material (a programmer error; both materials come from this
    module's own builders).
    """
    if not isinstance(material_a, str) or not material_a:
        raise ValueError("material_a must be a non-empty string")
    if not isinstance(material_b, str) or not material_b:
        raise ValueError("material_b must be a non-empty string")
    parts = sorted((material_a, material_b))
    framed = "".join(
        "{}:{}".format(len(p.encode("utf-8")), p) for p in parts
    ).encode("utf-8")
    digest = hashlib.scrypt(
        framed,
        salt=CONFIRM_CODE_SALT,
        n=CONFIRM_SCRYPT_N,
        r=CONFIRM_SCRYPT_R,
        p=CONFIRM_SCRYPT_P,
        maxmem=CONFIRM_SCRYPT_MAXMEM,
        dklen=32,
    )
    value = int.from_bytes(digest[:8], "big") % (10**CONFIRM_CODE_DIGITS)
    half = 10 ** (CONFIRM_CODE_DIGITS // 2)
    return "{:04d} {:04d}".format(value // half, value % half)


def build_pairing_payload(
    peer_id: str,
    routing_key: str,
    signing_pub: Optional[str] = None,
    device_class: Optional[str] = None,
) -> dict[str, Any]:
    """Build this device's pairing payload, computing its integrity check.

    The producer-side constructor. Validates its inputs and raises ``ValueError``
    on a programmer error (an empty identity or routing key, an empty
    ``signing_pub`` when one is given, a ``device_class`` outside
    :data:`PAIRING_DEVICE_CLASSES` -- our own producer never emits free
    text); the defensive path for untrusted input
    is :func:`parse_pairing_payload`, which never raises. Pure: it reaches into
    no store and opens no socket. The label a peer assigns is local to the
    accepting device, so it is not part of the payload or the integrity check.
    The signing public key (S205, VL-01) is included -- and folded into the
    integrity -- only when given, so a device without a signing keypair (no
    master key, no liboqs) still pairs as a pre-VL-01 peer. The device class
    (S258, PAIR-03) follows the same recipe: included and folded only when
    given, so a class-less payload keeps the historical digest byte for byte.
    """
    if not isinstance(peer_id, str) or not peer_id:
        raise ValueError("peer_id must be a non-empty string")
    if not isinstance(routing_key, str) or not routing_key:
        raise ValueError("routing_key must be a non-empty string")
    if signing_pub is not None and (
        not isinstance(signing_pub, str) or not signing_pub
    ):
        raise ValueError("signing_pub must be a non-empty string or None")
    if device_class is not None and device_class not in PAIRING_DEVICE_CLASSES:
        raise ValueError(
            "device_class must be one of {} or None".format(
                sorted(PAIRING_DEVICE_CLASSES)
            )
        )
    payload: dict[str, Any] = {
        "v": PAIRING_FORMAT_VERSION,
        "type": PAIRING_TYPE,
        "peer_id": peer_id,
        "routing_key": routing_key,
    }
    if signing_pub is not None:
        payload["signing_pub"] = signing_pub
    if device_class is not None:
        payload["device_class"] = device_class
    payload["integrity"] = pairing_integrity(
        peer_id, routing_key, signing_pub, device_class
    )
    return payload


def parse_pairing_payload(obj: Any) -> Optional[ParsedPairing]:
    """Parse a pairing payload defensively, or return ``None`` on any problem.

    Never raises into the caller. A non-mapping, a wrong format version or type, a
    missing or mistyped field, or an integrity check that does not match the public
    material all yield ``None``. The integrity check is recomputed from the parsed
    public fields and compared, so a tampered routing key (or identity) no longer
    matches its check and is rejected before it is ever stored.
    """
    try:
        if not isinstance(obj, Mapping):
            return None
        if obj.get("v") != PAIRING_FORMAT_VERSION:
            return None
        if obj.get("type") != PAIRING_TYPE:
            return None
        peer_id = obj.get("peer_id")
        if not isinstance(peer_id, str) or not peer_id:
            return None
        routing_key = obj.get("routing_key")
        if not isinstance(routing_key, str) or not routing_key:
            return None
        # S205 (VL-01): the signing public key is read defensively. Present
        # and a non-empty string, it joins the integrity recomputation;
        # absent, the historical four-field digest applies (a pre-VL-01
        # payload). Present but mistyped, it is treated as absent for the
        # recomputation, so a payload whose key was garbled into a non-string
        # fails its own five-field integrity and is rejected -- tampering
        # never degrades silently into "no key".
        signing_raw = obj.get("signing_pub")
        signing_pub = (
            signing_raw if isinstance(signing_raw, str) and signing_raw else None
        )
        # S258 (PAIR-03): the device class is read defensively, the S205
        # signing-key idiom exactly. Present and a non-empty string, it joins
        # the integrity recomputation -- ANY word: the vocabulary is the
        # apply seam's concern, so a future class never fails parse. Absent,
        # the historical digest applies. Present but mistyped or empty, it is
        # treated as absent for the recomputation, so a payload whose class
        # was garbled fails its own digest and is rejected -- tampering never
        # degrades silently into "no class".
        class_raw = obj.get("device_class")
        device_class = (
            class_raw if isinstance(class_raw, str) and class_raw else None
        )
        integrity = obj.get("integrity")
        if not isinstance(integrity, str) or not integrity:
            return None
        expected = pairing_integrity(
            peer_id, routing_key, signing_pub, device_class
        )
        if not _constant_time_equals(expected, integrity):
            return None
        return ParsedPairing(
            peer_id=peer_id,
            routing_key=routing_key,
            signing_pub=signing_pub,
            device_class=device_class,
        )
    except Exception:
        logger.debug("Rejected an unparseable pairing payload", exc_info=True)
        return None


def _constant_time_equals(a: str, b: str) -> bool:
    """Compare two hex digests in constant time; never raises."""
    try:
        import hmac

        return hmac.compare_digest(a, b)
    except Exception:  # pragma: no cover - defensive
        return a == b


def verify_pairing_payload(obj: Any) -> bool:
    """True when a payload parses and its integrity check matches; never raises."""
    return parse_pairing_payload(obj) is not None


def encode_pairing_json(payload: Mapping[str, Any]) -> str:
    """Serialise a built pairing payload to compact JSON (the QR / transcription text).

    Producer side; may raise if the payload is not JSON-safe (a programmer error,
    never untrusted input).
    """
    return json.dumps(dict(payload), separators=(",", ":"), ensure_ascii=False)


def decode_pairing_json(text: Any) -> Optional[ParsedPairing]:
    """Parse a pairing payload from JSON text defensively, or return ``None``.

    Never raises: invalid JSON, or a top-level value that is not an object, yields
    ``None``; an object is validated by :func:`parse_pairing_payload`, so the
    integrity check still applies to a scanned or transcribed payload.
    """
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, Mapping):
        return None
    return parse_pairing_payload(data)


def resolve_pairing_device_class(
    declared: Optional[str],
    prior_exists: Optional[bool],
    prior_class: Optional[str],
) -> tuple[bool, Optional[str]]:
    """Decide the class the accept seam records: the MONOTONE rule (S258).

    Pure and stdlib-only, so the whole policy is testable in isolation.
    ``declared`` is the payload's class as parsed (any string, or ``None``
    for a payload that carried none); ``prior_exists`` is ``True`` for an
    existing registry row, ``False`` for an affirmatively fresh one, and
    ``None`` when the seam could not determine it; ``prior_class`` is the
    existing row's stored class. Returns ``(apply, value)``: ``apply`` False
    means leave the stored class untouched.

    The table, fail-secure throughout (N9-D2 carried):

    - Free text never trusted: a declaration outside
      :data:`PAIRING_DEVICE_CLASSES` normalises to the least-trusted class
      (``phone``) -- a future vocabulary restricts, never escalates, and
      never bricks the ceremony.
    - FRESH row: record affirmatively; an absent declaration records
      ``phone`` (an undeclared new device is least-trusted; the human
      escalates at the control surface).
    - EXISTING row: monotone. Rank ``NULL`` with ``desktop`` (the
      grandfathered meaning); a declaration may keep or RESTRICT
      (``desktop -> phone``, ``NULL -> phone``) or make the grandfathered
      ``NULL`` explicit (``NULL -> desktop``, a same-rank write), and NEVER
      escalates: ``phone -> desktop`` is refused here -- the control
      surface's :func:`set_device_class` is the human-confirmed path. An
      absent declaration is no statement and changes nothing. A stored value
      outside the vocabulary (defensive; the store's setter cannot write
      one) ranks lowest, so nothing escalates from it.
    - INDETERMINABLE prior: only a ``phone`` resolution may write (a
      restriction is always safe); never a blind escalation, never a blind
      default.
    """
    if declared is None:
        target: Optional[str] = None
    elif isinstance(declared, str) and declared in PAIRING_DEVICE_CLASSES:
        target = declared
    else:
        target = PAIRING_DEVICE_CLASS_PHONE
    if prior_exists is False:
        if target is None:
            return (True, PAIRING_DEVICE_CLASS_PHONE)
        return (True, target)
    if prior_exists is not True:
        if target == PAIRING_DEVICE_CLASS_PHONE:
            return (True, target)
        return (False, None)
    if target is None:
        return (False, None)
    if target == PAIRING_DEVICE_CLASS_PHONE:
        if prior_class == PAIRING_DEVICE_CLASS_PHONE:
            return (False, None)
        return (True, target)
    if prior_class is None:
        return (True, target)
    return (False, None)


def _apply_pairing_device_class(
    engine: Any,
    parsed: ParsedPairing,
    prior_exists: Optional[bool],
    prior_class: Optional[str],
) -> None:
    """Record the declared class through the engine's setter, best-effort.

    Runs AFTER a successful registration and never voids it: a missing
    setter, a refused write, or a raising engine warns and returns -- the
    trust material landed, and the class stays the operator's to set, exactly
    the pre-S258 world. The value passed to the setter is always the
    decision function's output, which only ever emits vocabulary words, so
    the store's allowlist ``ValueError`` (free text, a programming error)
    cannot fire from here.
    """
    declared = parsed.device_class
    if declared is not None and declared not in PAIRING_DEVICE_CLASSES:
        logger.warning(
            "pairing: unknown device class %r normalised to %r for peer %s",
            declared,
            PAIRING_DEVICE_CLASS_PHONE,
            parsed.peer_id,
        )
    apply_it, value = resolve_pairing_device_class(
        declared, prior_exists, prior_class
    )
    if not apply_it:
        if declared == PAIRING_DEVICE_CLASS_DESKTOP:
            if (
                prior_exists is True
                and prior_class == PAIRING_DEVICE_CLASS_PHONE
            ):
                logger.warning(
                    "pairing: refusing the phone -> desktop escalation for "
                    "peer %s; set_device_class at the control surface is "
                    "the human-confirmed path",
                    parsed.peer_id,
                )
            elif prior_exists is not True:
                logger.warning(
                    "pairing: prior class indeterminable; refusing a blind "
                    "desktop class for peer %s",
                    parsed.peer_id,
                )
        return
    setter = getattr(engine, "set_device_class", None)
    if not callable(setter):
        logger.warning(
            "pairing: engine exposes no set_device_class; class %r for "
            "peer %s is left to the operator",
            value,
            parsed.peer_id,
        )
        return
    try:
        setter(parsed.peer_id, value)
    except Exception:
        logger.warning(
            "pairing: device class application failed for peer %s; the "
            "registration stands",
            parsed.peer_id,
            exc_info=True,
        )


def accept_pairing_payload(
    engine: Any, obj: Any, *, label: str = "", store: Any = None
) -> Optional[Any]:
    """Accept a peer's pairing payload and register it through the engine.

    Parses the payload defensively; on anything malformed or tampered returns
    ``None`` and registers nothing. Refuses self-pairing (PAIR-01): a payload
    whose ``peer_id`` is this engine's own device identity would register the
    device as its own peer -- a wasteful self-round loop and a parasitic
    registry entry -- so it is rejected like a malformed payload; the check is
    duck-typed and guarded (an engine exposing no ``device`` skips it). On a
    valid payload it populates the peer store
    via the engine's ``register_peer`` (an upsert that preserves the watermark on a
    re-pair and records the registration in the hash-chain audit log), returning the
    stored peer record. The ``label`` is the local, human-readable name the
    accepting device assigns; it is not part of the payload or its integrity check.

    The engine is duck-typed (anything exposing ``register_peer(peer_id,
    routing_key, *, label)``), so this stays transport-agnostic and is exercised
    with a fake engine in isolation. Registration is local-disk and permitted in any
    mode; only moving records over the wire is Daily-only.

    S258 (PAIR-03): when the payload declares a device class, it is recorded
    AFTER a successful registration through the engine's audited
    ``set_device_class``, under the monotone rule
    (:func:`resolve_pairing_device_class`) -- a declaration may keep or
    restrict a stored class, never escalate it. The optional ``store``
    (anything exposing ``get_peer``) lets the seam read the PRIOR row state
    BEFORE the upsert creates it, which is what distinguishes a fresh pairing
    (where an absent declaration records ``phone``, fail-secure) from a
    legacy re-pair (where an absent declaration changes nothing). With no
    ``store`` the prior is indeterminable and only a ``phone`` resolution is
    ever written. Best-effort throughout: a class-application failure warns
    and never voids the registration.
    """
    parsed = parse_pairing_payload(obj)
    if parsed is None:
        return None
    own = getattr(engine, "device", None)
    if isinstance(own, str) and own and parsed.peer_id == own:
        logger.warning(
            "pairing: refusing to register this device as its own peer (%s)",
            parsed.peer_id,
        )
        return None
    if not isinstance(label, str):
        label = ""
    # S258 (PAIR-03): the prior row state is read BEFORE the upsert -- after
    # it, a fresh row and a grandfathered re-pair are indistinguishable.
    # Guarded and fail-secure: a missing or raising lookup leaves the prior
    # INDETERMINABLE, under which the decision function only ever writes a
    # phone resolution.
    prior_exists: Optional[bool] = None
    prior_class: Optional[str] = None
    if store is not None:
        getter = getattr(store, "get_peer", None)
        if callable(getter):
            try:
                prior = getter(parsed.peer_id)
            except Exception:
                logger.warning(
                    "pairing: prior-state lookup failed for peer %s; the "
                    "device class applies fail-secure (a phone resolution "
                    "only)",
                    parsed.peer_id,
                    exc_info=True,
                )
            else:
                prior_exists = prior is not None
                prior_class = (
                    getattr(prior, "device_class", None)
                    if prior is not None
                    else None
                )
    # S206 (PAIR-02): the ceremony registers the peer PENDING -- the entry
    # gates nothing (no round, no serving, no trusted key lookup) until both
    # humans have compared the confirmation code and confirmed on both
    # devices. S205 (VL-01): a payload carrying a signing key registers it
    # with the peer. The kwarg cascade mirrors the S205 threading: each
    # TypeError falls back to the next-older engine signature, so a
    # pre-PAIR-02 engine (or an old fake in a test) still registers the peer
    # -- as an immediately-active, pre-ceremony entry -- and a pre-VL-01
    # engine registers it unkeyed, rather than failing the ceremony. Every
    # SUCCESSFUL path then applies the declared device class (S258,
    # best-effort, monotone) before returning.
    attempts: list[dict[str, Any]] = []
    newest: dict[str, Any] = {"label": label, "pending": True}
    if parsed.signing_pub is not None:
        newest["signing_pub"] = parsed.signing_pub
    attempts.append(newest)
    if parsed.signing_pub is not None:
        attempts.append({"label": label, "signing_pub": parsed.signing_pub})
    for kw in attempts:
        try:
            rec = engine.register_peer(parsed.peer_id, parsed.routing_key, **kw)
        except TypeError:
            logger.warning(
                "pairing: engine does not accept %s; falling back to an "
                "older registration signature for peer %s",
                sorted(k for k in kw if k != "label"),
                parsed.peer_id,
            )
            continue
        _apply_pairing_device_class(engine, parsed, prior_exists, prior_class)
        return rec
    rec = engine.register_peer(parsed.peer_id, parsed.routing_key, label=label)
    _apply_pairing_device_class(engine, parsed, prior_exists, prior_class)
    return rec
