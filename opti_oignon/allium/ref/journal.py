"""The identity of a fact: its envelope checked field by field, its body digest, its event id.

A fact is ``{"being", "body", "kind", "laws", "origin", "oseq", "t"}`` with
its body an object. Its envelope is the same object with the body replaced
by the SHA-256 hex of the body's canonical bytes, and its event id (eid) is
the SHA-256 hex of the envelope's canonical bytes. Nothing local -- no
sequence number, no link -- enters the envelope, so a fact has the same eid
on every device that holds it.

``fact_id`` takes a fact with its body object and answers the body digest
and the eid. ``fact_envelope`` takes envelopes whose body is already a
digest and answers their eids: a fact whose body has been redacted keeps a
verifiable eid that way, since only the digest of the body enters the eid.

Every check raises ``wire.Refused``, in a fixed order, so the first defect
met decides the refusal. The Rust twin (``rust/allium/src/journal.rs``)
mirrors this module function by function and refuses with the same codes
and details.
"""

import hashlib

from .. import wire
from ..wire import Refused

checkpoint_before_apply = True

FACT_FIELDS = ("being", "body", "kind", "laws", "origin", "oseq", "t")
BODY_LIMIT = 4096
_HEX = "0123456789abcdef"


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex(value, length):
    if not isinstance(value, str) or len(value) != length:
        return False
    for char in value:
        if char not in _HEX:
            return False
    return True


def check_envelope(fact, digest_body):
    """Refuse, by name, a fact that is not well formed; nothing else happens.

    The key set comes first, then ``being``, ``kind``, ``laws``, ``origin``,
    ``oseq`` and ``t``, then the body: an object when ``digest_body`` is
    false, a lowercase hex64 digest when it is true.
    """
    if not isinstance(fact, dict):
        raise Refused("bad_fact", "fact")
    for name in fact:
        if name not in FACT_FIELDS:
            raise Refused("bad_fact", "fields")
    for name in FACT_FIELDS:
        if name not in fact:
            raise Refused("bad_fact", "fields")
    if not _is_hex(fact["being"], 32):
        raise Refused("bad_fact", "being")
    kind = fact["kind"]
    if not isinstance(kind, str) or not 1 <= len(kind) <= 32:
        raise Refused("bad_fact", "kind")
    for char in kind:
        if not ("a" <= char <= "z" or "0" <= char <= "9" or char == "_"):
            raise Refused("bad_fact", "kind")
    if not _is_int(fact["laws"]) or fact["laws"] < 0:
        raise Refused("bad_fact", "laws")
    if not _is_hex(fact["origin"], 16):
        raise Refused("bad_fact", "origin")
    if not _is_int(fact["oseq"]) or fact["oseq"] < 0:
        raise Refused("bad_fact", "oseq")
    if not _is_int(fact["t"]):
        raise Refused("bad_fact", "t")
    body = fact["body"]
    if digest_body:
        if not _is_hex(body, 64):
            raise Refused("bad_fact", "body")
    elif not isinstance(body, dict):
        raise Refused("bad_fact", "body")


def body_digest(body):
    """The SHA-256 hex of a body's canonical bytes; a body over ``BODY_LIMIT`` bytes is refused."""
    body_bytes = wire.emit(body)
    if len(body_bytes) > BODY_LIMIT:
        raise Refused("limit", "body size")
    return hashlib.sha256(body_bytes).hexdigest()


def eid(fact, digest):
    """The eid of a checked fact: the SHA-256 hex of its envelope, its body replaced by ``digest``."""
    envelope = {name: fact[name] for name in FACT_FIELDS}
    envelope["body"] = digest
    return hashlib.sha256(wire.emit(envelope)).hexdigest()


def fact_id(fact):
    """A fact's body digest and eid, ``{"body", "eid"}``; a malformed fact is refused by name."""
    check_envelope(fact, False)
    digest = body_digest(fact["body"])
    return {"body": digest, "eid": eid(fact, digest)}


def fact_envelope(envelopes):
    """The eids of a list of envelopes, ``{"eids": [...]}`` in their order.

    Every envelope is checked, left to right, before any eid is computed, so
    one malformed envelope refuses the whole list.
    """
    for envelope in envelopes:
        check_envelope(envelope, True)
    return {"eids": [eid(envelope, envelope["body"]) for envelope in envelopes]}
