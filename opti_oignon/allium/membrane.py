"""The membrane: the one way a fact gets into a being's life, and the rules it answers to.

Nothing is journaled that did not pass here, in this order: the kind is in
the kinds table; the surface the transport comes through may write it; its
producer is the membrane (genesis is written by sowing, ``resumed`` by a
resume, ``owner`` by a claim, ``tz`` and ``clock`` by the recorder, rhythm
rows by the rhythm layer, ``evolve`` and the law pins by the laws writer);
its body schema is not reserved; it carries no
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
remembered per process by the digest of the bytes they came from. The
table's grammar -- its own soundness and the bodies it admits -- is the
reference engine's (``ref/journal.py``); the membrane wraps it with the
same details and adds the surfaces.
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
    "laws_apply": _CS,
    "resume": _CS,
    "revoke": _CS,
    "sow": _CS,
    "unclaimed": _CS,
}
ATTENDED = ("grant", "revoke")

# The producers the kinds table names, as ``ref/journal.PRODUCERS`` lists them.
PRODUCERS = ("claim", "laws", "membrane", "recorder", "resume", "rhythm", "sow")
EXEMPT = ("clock", "genesis", "owner", "resumed", "tz")
BUDGET_MAX = 4096
MAX_INT = (1 << 53) - 1
MINUTES_A_DAY = 1440
# The last minute of life a fact may carry: two days short of the largest integer, so that the next local
# midnight after it, and a law update's minute, stay in range.
LAST_MINUTE = MAX_INT - 2 * MINUTES_A_DAY
_PRODUCED_BY = {
    "claim": "written only by claim",
    "laws": "written only by the laws writer",
    "membrane": "written only by the membrane",
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


def validate_table(table):
    """Refuse, by name, a kinds table that is not whole; ``StoreRefused("law", <defect>)``.

    The table's own soundness is the reference's (``ref/journal.validate_table``),
    with the same details; the platform adds that the table and ``MATRIX``
    name the same kinds.
    """
    from . import wire
    from .ref import journal

    try:
        journal.validate_table(table)
    except wire.Refused as refusal:
        raise _law_refused(refusal.detail) from None
    if sorted(table["kinds"]) != sorted(MATRIX):
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


def _law_params(law, table):
    """The ranges of the params a genesis under ``law`` freezes, checked against the table's params schema."""
    schema = table["kinds"]["genesis"]["body"]["laws"]["fields"]["params"]["fields"]
    params = law.get("params")
    if not _exactly(params, tuple(schema)):
        raise _law_refused("the law's params")
    out = {}
    for name in sorted(schema):
        spec = params[name]
        if (not _exactly(spec, ("default", "hi", "lo")) or not all(_is_int(spec[key]) for key in spec)
                or not schema[name]["lo"] <= spec["lo"] <= spec["default"] <= spec["hi"] <= schema[name]["hi"]):
            raise _law_refused(f"the law's params: {name}")
        out[name] = {"default": spec["default"], "hi": spec["hi"], "lo": spec["lo"]}
    return out


def law_pin(name):
    """A carried law's identity, its validated journal pin, and the ranges of its params.

    Returns ``{"budgets", "name", "params", "provisional", "sha256", "table", "table_name", "version"}``;
    a law this engine does not carry, or one whose pin or params are not
    whole, is ``StoreRefused("law")``.
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
    params = _law_params(law, pin["table"])
    result = dict(pin, name=name, params=params, provisional=provisional, sha256=lawfiles.digest(law),
                  version=version)
    _LAWS[key] = result
    return result


# ---------------------------------------------------------------------------
# Bodies, payloads, taint, producers
# ---------------------------------------------------------------------------
def check_body(schema, body, path=""):
    """Refuse a body that is not an object with exactly the schema's keys and valid values.

    The check is the reference's (``ref/journal.check_body``); its refusal
    reaches the platform as ``MembraneRefused("body")`` with the same
    detail, ``body <path> <problem>``, for example ``body laws.sha256 hex``.
    """
    from . import wire
    from .ref import journal

    try:
        journal.check_body(schema, body, path)
    except wire.Refused as refusal:
        raise MembraneRefused("body", refusal.detail) from None


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


def admit(kind, body, *, transport, grant_ref, payload, now, single_user, owner, table, producer="membrane"):
    """Steps 1 to 7 of an append, in their order; returns the kind's table entry.

    ``owner`` is the being's owner tag, ``table`` the validated kinds table.
    ``producer`` is who writes: a gesture's append is the membrane's, and
    the laws writer names itself; a kind another producer owns is refused
    ``producer``, after the surface.
    """
    from . import anchors

    entry = table["kinds"].get(kind) if isinstance(kind, str) else None
    if entry is None:
        raise MembraneRefused("kind", "not a kind of the table")
    surface = surface_of(transport, now)
    if surface not in MATRIX.get(kind, ()):
        raise MembraneRefused("surface", f"{kind} is not written from {surface}")
    if producer not in PRODUCERS or entry["producer"] != producer:
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
    """Never before birth: a wall reading that is not an integer, is before birth, or is past the last minute
    of life a fact may carry (``LAST_MINUTE``), is refused ``clock``."""
    if not _is_int(wall):
        raise MembraneRefused("clock", "unreadable")
    if wall < birth_wall:
        raise MembraneRefused("clock", "before birth")
    if (wall - birth_wall) // 60 > LAST_MINUTE:
        raise MembraneRefused("clock", "past the last minute of life a fact may carry")
    return wall


def recorder_t(wall, birth_wall, t_max):
    """Never backwards: the minute of life, never before the latest one already linked."""
    t = (wall - birth_wall) // 60
    return t if t_max is None or t >= t_max else t_max


def day_of(t):
    """The budget day: the day of life."""
    return t // MINUTES_A_DAY
