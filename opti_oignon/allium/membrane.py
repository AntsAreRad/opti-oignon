"""The membrane: the one way a fact gets into a being's life, and the rules it answers to.

Nothing is journaled that did not pass here, in this order: the kind is in
the kinds table; the surface the transport comes through may write it; its
producer is the membrane (genesis is written by sowing, ``resumed`` by a
resume, ``owner`` by a claim, ``tz`` and ``clock`` by the recorder, rhythm
rows by the rhythm layer); its body schema is not reserved; it carries no
taint; its body matches the schema exactly, and its payload the payload
spec; and the account behind the transport owns the being. The store then
reads the clock (never backwards, never before birth) and writes under its
own lock and transaction.

The surface and the actor are derived from the ``Transport`` record alone:

* ``cli`` is ``cli_tty``; ``attended`` (the strong terminal test, built by
  the service) matters only to the consent verbs ``grant`` and ``revoke``;
* ``web`` is ``web_session`` only with a validated access token that has not
  expired (``exp > now``), and ``web_local`` otherwise -- the synthetic
  single-user principal included;
* ``light_hook``, ``idle_timer`` and ``passport`` are themselves, and ``sync``
  is ``own_device_sync``, which appears in no row: it cannot append.

What the request's own bytes claim (``Transport.claimed``) is never read.

The kinds table is law data: it is validated whole, and the daily budgets
the law pins beside it must name exactly the budgeted trunk kinds. Both are
remembered per process by the digest of the bytes they came from.
"""

import hashlib
from typing import NamedTuple

checkpoint_before_apply = True

LOCAL_USER = "local"

_C = "cli_tty"
_S = "web_session"
_L = "web_local"
_H = "light_hook"
_I = "idle_timer"
_P = "passport"
_CS = (_C, _S)
_CSL = (_C, _S, _L)

# The surfaces each kind may be written from. A kind no surface may write
# (the recorder's) has an empty row; own_device_sync is in no row at all.
MATRIX = {
    "act": _CSL,
    "braid": _CSL,
    "bury": _CS,
    "celebrate": _CSL,
    "clock": (),
    "dream_depth": _CS,
    "evolve": (_C, _S, _H, _I),
    "excavate": _CSL,
    "forget_rhythm": _CS,
    "genesis": _CS,
    "keep_dream": _CSL,
    "lang_borrow": _CSL,
    "lang_forget": _CS,
    "lang_insist": _CSL,
    "lang_reaction": _CSL,
    "lang_taboo_add": _CS,
    "lang_talk": _CSL,
    "lang_teach": _CSL,
    "laws_pin": _CS,
    "laws_unpin": _CS,
    "letter_sealed": _CSL,
    "light_day": (_H,),
    "meeting_in": (_P,),
    "move_pot": _CSL,
    "name": _CS,
    "offering_verdict": _CSL,
    "owner": _CS,
    "pollen_in": (_P,),
    "presence_hour": (_H,),
    "rest_begin": _CS,
    "rest_end": _CS,
    "resumed": _CS,
    "rhythm_seal": (_I,),
    "seal_broken": _CSL,
    "seed_in": (_P,),
    "tz": (),
}

# The verbs, which are not journaled: all of them from a terminal or a session only.
VERBS = {
    "compost": _CS,
    "export": _CS,
    "finish_sowing": _CS,
    "grant": _CS,
    "import": _CS,
    "resume": _CS,
    "revoke": _CS,
    "sow": _CS,
    "unclaimed": _CS,
}
ATTENDED = ("grant", "revoke")

TABLE_KEYS = ("kinds", "name", "schema")
ENTRY_KEYS = ("body", "payload", "producer", "redact_by", "scope")
SCOPES = ("rhythm", "trunk")
PRODUCERS = ("claim", "membrane", "recorder", "resume", "rhythm", "sow")
EXEMPT = ("clock", "genesis", "owner", "resumed", "tz")
BUDGET_MAX = 4096
MAX_INT = (1 << 53) - 1
MINUTES_A_DAY = 1440
HOUR_FIELDS = ("hour", "observed_hour")
HEX32 = {"len": 32, "type": "hex"}
HEX64 = {"len": 64, "type": "hex"}
_HEX = "0123456789abcdef"
_PRODUCED_BY = {
    "claim": "written only by claim",
    "recorder": "written only by the recorder",
    "resume": "written only by resume",
    "rhythm": "written only by the rhythm layer",
    "sow": "written only by sow",
}


class MembraneRefused(ValueError):
    """A write the membrane refuses, named by a code from ``CODES``; nothing was written."""

    CODES = ("kind", "surface", "attended", "producer", "reserved", "taint", "body", "payload", "owner",
             "clock", "target", "consent")

    def __init__(self, code, detail=""):
        if code not in self.CODES:
            raise ValueError(f"unknown membrane refusal: {code}")
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class Transport(NamedTuple):
    """How a request reached the store: the only thing surfaces and actors are derived from."""

    channel: str
    principal: object = None
    attended: bool = False
    claimed: object = None


class Appended(NamedTuple):
    """A fact written: its eid, its local seq, its oseq and its minute of life."""

    eid: str
    seq: int
    oseq: int
    t: int


class Dropped(NamedTuple):
    """A fact not written because its kind's daily budget was spent; the drop is counted."""

    reason: str


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex(value, length):
    if not isinstance(value, str) or len(value) != length:
        return False
    for char in value:
        if char not in _HEX:
            return False
    return True


def _is_text(value, most):
    if not isinstance(value, str) or not 1 <= len(value) <= most:
        return False
    for char in value:
        if not 0x20 <= ord(char) <= 0x7E:
            return False
    return value[0] != " " and value[-1] != " "


def _is_word(value, most):
    if not isinstance(value, str) or not 1 <= len(value) <= most:
        return False
    for char in value:
        if not "a" <= char <= "z":
            return False
    return True


# ---------------------------------------------------------------------------
# Surfaces and actors
# ---------------------------------------------------------------------------
def _validated(principal, now):
    """A validated access token that has not expired."""
    if not isinstance(principal, dict):
        return False
    sub = principal.get("sub")
    iat = principal.get("iat")
    exp = principal.get("exp")
    return (isinstance(sub, str) and sub != "" and _is_int(iat) and _is_int(exp) and _is_int(now)
            and exp > now and principal.get("type") == "access")


def surface_of(transport, now):
    """The surface a transport comes through; an unknown channel is refused ``surface``."""
    if not isinstance(transport, Transport):
        raise MembraneRefused("surface", "not a transport")
    channel = transport.channel
    if channel == "cli":
        return _C
    if channel == "web":
        return _S if _validated(transport.principal, now) else _L
    if channel in (_H, _I, _P):
        return channel
    if channel == "sync":
        return "own_device_sync"
    raise MembraneRefused("surface", "unknown channel")


def actor_of(transport, single_user, now):
    """The account behind a transport: the local user in single-user mode, else a session's subject."""
    if single_user:
        return LOCAL_USER
    if surface_of(transport, now) == _S:
        return transport.principal["sub"]
    raise MembraneRefused("owner", "no authenticated account")


def permit(action, transport, now):
    """The surface of a verb's transport, or its refusal (``surface``, ``attended``)."""
    surface = surface_of(transport, now)
    allowed = VERBS.get(action) if isinstance(action, str) else None
    if allowed is None:
        raise MembraneRefused("kind", "unknown action")
    if surface not in allowed:
        raise MembraneRefused("surface", f"{action} is not allowed from {surface}")
    if action in ATTENDED and surface == _C and transport.attended is not True:
        raise MembraneRefused("attended", f"{action} needs a person at the terminal")
    return surface


# ---------------------------------------------------------------------------
# The kinds table and the law's pin
# ---------------------------------------------------------------------------
def _law_refused(detail):
    from .store import StoreRefused

    return StoreRefused("law", detail)


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


def validate_table(table):
    """Refuse, by name, a kinds table that is not whole; ``StoreRefused("law", <defect>)``."""
    if not _exactly(table, TABLE_KEYS) or not isinstance(table["name"], str) or table["schema"] != 1:
        raise _law_refused("the kinds table: keys, name or schema")
    kinds = table["kinds"]
    if not isinstance(kinds, dict):
        raise _law_refused("the kinds table: kinds")
    for kind, entry in kinds.items():
        if not _kind_name(kind):
            raise _law_refused(f"{kind}: name")
        if not _exactly(entry, ENTRY_KEYS):
            raise _law_refused(f"{kind}: keys")
        if entry["scope"] not in SCOPES or entry["producer"] not in PRODUCERS:
            raise _law_refused(f"{kind}: scope or producer")
        if entry["scope"] == "rhythm" and (entry["producer"] != "rhythm" or entry["redact_by"] is not None):
            raise _law_refused(f"{kind}: a rhythm kind")
        body = entry["body"]
        if body is not None:
            defect = _schema_defect(body)
            if defect is not None:
                raise _law_refused(f"{kind}: body {defect}")
        payload = entry["payload"]
        has_payload = body is not None and "payload" in body
        if (payload is not None) != has_payload:
            raise _law_refused(f"{kind}: payload")
        if payload is not None and (not _exactly(payload, ("max", "type")) or payload["type"] != "word"
                                    or not _is_int(payload["max"]) or payload["max"] < 1):
            raise _law_refused(f"{kind}: payload spec")
        redactor = entry["redact_by"]
        if redactor is not None:
            target = kinds.get(redactor) if isinstance(redactor, str) else None
            if (not isinstance(target, dict) or target.get("scope") != "trunk" or target.get("redact_by") is not None
                    or target.get("body") != {"target": HEX64}):
                raise _law_refused(f"{kind}: redact_by")
            if body is not None and (body.get("payload") != HEX32 or payload is None):
                raise _law_refused(f"{kind}: redactable without a payload reference")
        if entry["scope"] == "trunk" and body is not None:
            for name in _field_names(body):
                if name in HOUR_FIELDS:
                    raise _law_refused(f"{kind}: an hour in the trunk")
    if sorted(kinds) != sorted(MATRIX):
        raise _law_refused("the kinds table and the surface matrix name different kinds")


_PINS = {}


def validate_pin(law, table_bytes):
    """The law's journal pin, checked whole against the table it names.

    ``law`` is the law's bytes or its parsed value; ``table_bytes`` the kinds
    table's file bytes. Returns ``{"budgets", "table", "table_name"}``; any defect
    is ``StoreRefused("law", <defect>)``. Remembered per process by the
    digests of both inputs.
    """
    from . import lawfiles, wire

    law_bytes = bytes(law) if isinstance(law, (bytes, bytearray)) else wire.emit(law)
    key = hashlib.sha256(law_bytes).hexdigest() + hashlib.sha256(bytes(table_bytes)).hexdigest()
    if key in _PINS:
        return _PINS[key]
    try:
        law_value = wire.parse(law_bytes, lenient=True)
        table = wire.parse(bytes(table_bytes), lenient=True)
    except wire.Refused as refusal:
        raise _law_refused(f"the law or the table does not parse ({refusal.code})") from None
    pin = law_value.get("journal") if isinstance(law_value, dict) else None
    if not _exactly(pin, ("budgets", "table")):
        raise _law_refused("the journal pin: keys")
    named = pin["table"]
    if not _exactly(named, ("name", "sha256")):
        raise _law_refused("the journal pin: table keys")
    if not isinstance(named["name"], str) or named["name"] not in lawfiles.JOURNAL_TABLES:
        raise _law_refused("the journal pin: table name")
    if named["sha256"] != lawfiles.digest(table):
        raise _law_refused("the journal pin: table digest")
    validate_table(table)
    budgeted = sorted(kind for kind, entry in table["kinds"].items()
                      if entry["scope"] == "trunk" and kind not in EXEMPT)
    budgets = pin["budgets"]
    if not isinstance(budgets, dict):
        raise _law_refused("the journal pin: budgets")
    for kind in budgeted:
        if kind not in budgets:
            raise _law_refused(f"the journal pin: no budget for {kind}")
    for kind in budgets:
        if kind not in budgeted:
            raise _law_refused(f"the journal pin: a budget for {kind}, which has none")
    for kind in budgeted:
        value = budgets[kind]
        if not _is_int(value) or not 1 <= value <= BUDGET_MAX:
            raise _law_refused(f"the journal pin: budget {kind}")
    result = {"budgets": {kind: budgets[kind] for kind in budgeted}, "table": table, "table_name": named["name"]}
    _PINS[key] = result
    return result


_LAWS = {}


def law_pin(name):
    """A carried law's identity and its validated journal pin.

    Returns ``{"budgets", "name", "provisional", "sha256", "table", "table_name", "version"}``;
    a law this engine does not carry, or one whose pin is not whole, is
    ``StoreRefused("law")``.
    """
    from . import lawfiles, wire

    if not isinstance(name, str) or name not in lawfiles.LAWS:
        raise _law_refused("a law this engine does not carry")
    law_bytes = lawfiles.law_bytes(name)
    key = hashlib.sha256(law_bytes).hexdigest()
    if key in _LAWS:
        return _LAWS[key]
    law = wire.parse(law_bytes, lenient=True)
    named = law.get("journal", {}).get("table", {}) if isinstance(law.get("journal"), dict) else {}
    table_name = named.get("name") if isinstance(named, dict) else None
    if not isinstance(table_name, str) or table_name not in lawfiles.JOURNAL_TABLES:
        raise _law_refused("the journal pin: table name")
    pin = validate_pin(law_bytes, lawfiles.table_bytes(table_name))
    version = law.get("version")
    provisional = law.get("provisional")
    if not _is_int(version) or not 0 <= version <= 65535 or not isinstance(provisional, bool):
        raise _law_refused("the law's version or provisional flag")
    result = dict(pin, name=name, provisional=provisional, sha256=lawfiles.digest(law), version=version)
    _LAWS[key] = result
    return result


# ---------------------------------------------------------------------------
# Bodies, payloads, taint, producers
# ---------------------------------------------------------------------------
def _refuse_body(path, problem):
    return MembraneRefused("body", " ".join(part for part in ("body", path, problem) if part))


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

    The detail reads ``body <path> <problem>``, for example
    ``body laws.sha256 hex``.
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


def check_payload(spec, text):
    """Refuse a payload that is not a word of the spec: lowercase letters, 1..=max of them."""
    if not _is_word(text, spec["max"]):
        raise MembraneRefused("payload", f"the payload is not a word of 1..={spec['max']} lowercase letters")


def check_taint(grant_ref):
    """Refuse any taint: no compartment exists yet, so nothing tainted is written anywhere."""
    if grant_ref is None:
        return
    if isinstance(grant_ref, (list, tuple, set, frozenset)):
        raise MembraneRefused("taint", "a taint of more than one grant")
    raise MembraneRefused("taint", "no compartment exists for this grant; nothing is written")


def admit(kind, body, *, transport, grant_ref, payload, now, single_user, owner, table):
    """Steps 1 to 7 of an append, in their order; returns the kind's table entry.

    ``owner`` is the being's owner tag, ``table`` the validated kinds table.
    """
    from . import anchors

    entry = table["kinds"].get(kind) if isinstance(kind, str) else None
    if entry is None:
        raise MembraneRefused("kind", "not a kind of the table")
    surface = surface_of(transport, now)
    if surface not in MATRIX.get(kind, ()):
        raise MembraneRefused("surface", f"{kind} is not written from {surface}")
    if entry["producer"] != "membrane":
        raise MembraneRefused("producer", _PRODUCED_BY.get(entry["producer"], "not the membrane's"))
    schema = entry["body"]
    if schema is None:
        raise MembraneRefused("reserved", f"{kind} has no body yet")
    check_taint(grant_ref)
    check_body({name: spec for name, spec in schema.items() if name != "payload"}, body)
    if (payload is not None) != (entry["payload"] is not None):
        raise MembraneRefused("payload", "a payload is given exactly when the kind carries one")
    if payload is not None:
        check_payload(entry["payload"], payload)
    if anchors.owner_tag(actor_of(transport, single_user, now)) != owner:
        raise MembraneRefused("owner", "this account does not own this being")
    return entry


# ---------------------------------------------------------------------------
# The recorder: never backwards, never before birth
# ---------------------------------------------------------------------------
def recorder_wall(wall, birth_wall):
    """Never before birth: a wall reading that is not an integer, or is before birth, is refused ``clock``."""
    if not _is_int(wall):
        raise MembraneRefused("clock", "unreadable")
    if wall < birth_wall:
        raise MembraneRefused("clock", "before birth")
    return wall


def recorder_t(wall, birth_wall, t_max):
    """Never backwards: the minute of life, never before the latest one already linked."""
    t = (wall - birth_wall) // 60
    return t if t_max is None or t >= t_max else t_max


def day_of(t):
    """The budget day: the day of life."""
    return t // MINUTES_A_DAY
