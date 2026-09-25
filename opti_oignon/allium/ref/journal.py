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

The kinds table is law data, and this module holds its grammar:
``validate_table`` refuses a table that is not whole, ``check_body`` a body
that does not match its kind's schema, and ``check_fact`` a fact of a life
that its kind's entry does not admit. The membrane wraps the first two for
the platform, with the same details. The twin holds ``check_body`` only: it
reads the embedded table, whose digest the handshake proves, so it never
validates a table.
"""

import hashlib

from .. import wire
from ..wire import Refused

checkpoint_before_apply = True

FACT_FIELDS = ("being", "body", "kind", "laws", "origin", "oseq", "t")
BODY_LIMIT = 4096
MAX_INT = (1 << 53) - 1
TABLE_KEYS = ("kinds", "name", "schema")
ENTRY_KEYS = ("body", "payload", "producer", "redact_by", "scope")
SCOPES = ("rhythm", "trunk")
# Who alone may write each kind: the membrane (a gesture), sowing, a claim, a
# resume, the recorder, the rhythm layer, or the laws writer.
PRODUCERS = ("claim", "laws", "membrane", "recorder", "resume", "rhythm", "sow")
HOUR_FIELDS = ("hour", "observed_hour")
HEX32 = {"len": 32, "type": "hex"}
HEX64 = {"len": 64, "type": "hex"}
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


# ---------------------------------------------------------------------------
# The kinds table: its own soundness, the bodies it admits, the facts of a life
# ---------------------------------------------------------------------------
def _is_text(value, most):
    if not isinstance(value, str) or not 1 <= len(value) <= most:
        return False
    for char in value:
        if not 0x20 <= ord(char) <= 0x7E:
            return False
    return value[0] != " " and value[-1] != " "


def _exactly(value, names):
    if not isinstance(value, dict):
        return False
    for name in value:
        if name not in names:
            return False
    for name in names:
        if name not in value:
            return False
    return True


def _spec_defect(spec):
    """What is wrong with one field spec, or ``None``."""
    if not isinstance(spec, dict):
        return "a spec is not an object"
    kind = spec.get("type")
    if kind == "int":
        if not _exactly(spec, ("hi", "lo", "type")) or not _is_int(spec["lo"]) or not _is_int(spec["hi"]):
            return "int"
        if not -MAX_INT <= spec["lo"] <= spec["hi"] <= MAX_INT:
            return "int bounds"
        return None
    if kind == "bool":
        return None if _exactly(spec, ("type",)) else "bool"
    if kind == "symbol":
        members = spec.get("of")
        if not _exactly(spec, ("of", "type")) or not isinstance(members, list) or not members:
            return "symbol"
        for member in members:
            if not _is_text(member, 32):
                return "symbol member"
        for index in range(1, len(members)):
            if not members[index - 1] < members[index]:
                return "symbol order"
        return None
    if kind == "hex":
        return None if _exactly(spec, ("len", "type")) and _is_int(spec["len"]) and spec["len"] >= 1 else "hex"
    if kind == "text":
        return None if _exactly(spec, ("max", "type")) and _is_int(spec["max"]) and spec["max"] >= 1 else "text"
    if kind == "object":
        if not _exactly(spec, ("fields", "type")):
            return "object"
        return _schema_defect(spec["fields"])
    return "unknown type"


def _schema_defect(schema):
    if not isinstance(schema, dict):
        return "a schema is not an object"
    for name, spec in schema.items():
        if not isinstance(name, str) or not name:
            return "a field name"
        defect = _spec_defect(spec)
        if defect is not None:
            return f"{name}: {defect}"
    return None


def _field_names(schema):
    names = []
    for name, spec in schema.items():
        names.append(name)
        if isinstance(spec, dict) and spec.get("type") == "object" and isinstance(spec.get("fields"), dict):
            names.extend(_field_names(spec["fields"]))
    return names


def _kind_name(kind):
    if not isinstance(kind, str) or not 1 <= len(kind) <= 32:
        return False
    for char in kind:
        if not ("a" <= char <= "z" or "0" <= char <= "9" or char == "_"):
            return False
    return True


def _table_refused(detail):
    return Refused("unknown_law", detail)


def validate_table(table):
    """Refuse, by name, a kinds table that is not whole: ``Refused("unknown_law", <defect>)``.

    This is the table's own soundness: keys, names, scopes and producers,
    body and payload schemas, redaction targets, and no hour in the trunk.
    Which surfaces may write each kind is the platform's, not the table's.
    """
    if not _exactly(table, TABLE_KEYS) or not isinstance(table["name"], str) or table["schema"] != 1:
        raise _table_refused("the kinds table: keys, name or schema")
    kinds = table["kinds"]
    if not isinstance(kinds, dict):
        raise _table_refused("the kinds table: kinds")
    for kind, entry in kinds.items():
        if not _kind_name(kind):
            raise _table_refused(f"{kind}: name")
        if not _exactly(entry, ENTRY_KEYS):
            raise _table_refused(f"{kind}: keys")
        if entry["scope"] not in SCOPES or entry["producer"] not in PRODUCERS:
            raise _table_refused(f"{kind}: scope or producer")
        if entry["scope"] == "rhythm" and (entry["producer"] != "rhythm" or entry["redact_by"] is not None):
            raise _table_refused(f"{kind}: a rhythm kind")
        body = entry["body"]
        if body is not None:
            defect = _schema_defect(body)
            if defect is not None:
                raise _table_refused(f"{kind}: body {defect}")
        payload = entry["payload"]
        has_payload = body is not None and "payload" in body
        if (payload is not None) != has_payload:
            raise _table_refused(f"{kind}: payload")
        if payload is not None and (not _exactly(payload, ("max", "type")) or payload["type"] != "word"
                                    or not _is_int(payload["max"]) or payload["max"] < 1):
            raise _table_refused(f"{kind}: payload spec")
        redactor = entry["redact_by"]
        if redactor is not None:
            target = kinds.get(redactor) if isinstance(redactor, str) else None
            if (not isinstance(target, dict) or target.get("scope") != "trunk" or target.get("redact_by") is not None
                    or target.get("body") != {"target": HEX64}):
                raise _table_refused(f"{kind}: redact_by")
            if body is not None and (body.get("payload") != HEX32 or payload is None):
                raise _table_refused(f"{kind}: redactable without a payload reference")
        if entry["scope"] == "trunk" and body is not None:
            for name in _field_names(body):
                if name in HOUR_FIELDS:
                    raise _table_refused(f"{kind}: an hour in the trunk")


def _refuse_body(path, problem):
    return Refused("bad_fact", " ".join(part for part in ("body", path, problem) if part))


def _check_value(spec, value, path):
    kind = spec["type"]
    if kind == "int":
        if not _is_int(value):
            raise _refuse_body(path, "int")
        if not spec["lo"] <= value <= spec["hi"]:
            raise _refuse_body(path, "range")
    elif kind == "bool":
        if not isinstance(value, bool):
            raise _refuse_body(path, "bool")
    elif kind == "symbol":
        if not isinstance(value, str) or value not in spec["of"]:
            raise _refuse_body(path, "symbol")
    elif kind == "hex":
        if not _is_hex(value, spec["len"]):
            raise _refuse_body(path, "hex")
    elif kind == "text":
        if not _is_text(value, spec["max"]):
            raise _refuse_body(path, "text")
    else:
        check_body(spec["fields"], value, path)


def check_body(schema, body, path=""):
    """Refuse a body that is not an object with exactly the schema's keys and valid values.

    The schema is a validated table's; the refusal is ``bad_fact`` with the
    detail ``body <path> <problem>``, for example ``body laws.sha256 hex``.
    Unknown keys, then missing ones, are ``fields``; then each field, in
    the order of its name, is checked against its spec.
    """
    if not isinstance(body, dict):
        raise _refuse_body(path, "object")
    for name in body:
        if not isinstance(name, str) or name not in schema:
            raise _refuse_body(path, "fields")
    for name in schema:
        if name not in body:
            raise _refuse_body(path, "fields")
    for name in sorted(schema):
        _check_value(schema[name], body[name], f"{path}.{name}" if path else name)


def check_fact(fact, table):
    """What a validated kinds table says of one fact of a life; returns the kind's entry.

    The fact's envelope is already checked, its body an object or, for a
    redacted fact, the hex64 digest of one. In this order: the kind is in
    the table (``kind``), in the trunk (``scope``), not the genesis
    (``genesis``, which only opens a life), and has a body schema
    (``reserved``); an object body matches it (``body ...``), and a digest
    stands in for a body only where the kind can be redacted
    (``redaction``). Every refusal is ``bad_fact``.
    """
    entry = table["kinds"].get(fact["kind"])
    if entry is None:
        raise Refused("bad_fact", "kind")
    if entry["scope"] != "trunk":
        raise Refused("bad_fact", "scope")
    if fact["kind"] == "genesis":
        raise Refused("bad_fact", "genesis")
    if entry["body"] is None:
        raise Refused("bad_fact", "reserved")
    body = fact["body"]
    if isinstance(body, dict):
        check_body(entry["body"], body)
    elif entry["redact_by"] is None:
        raise Refused("bad_fact", "redaction")
    return entry
