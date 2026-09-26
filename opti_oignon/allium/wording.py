"""Every English line the garden prints, as a closed catalogue of templates with typed slots.

``TEMPLATES`` holds each line once, by key; nothing the garden prints is
written anywhere else. ``say(key, **slots)`` renders one line and checks it
before it can be printed: every slot is typed (``SLOTS``) and a value
outside its type or its closed map is refused, then the lexical nets of
``ethics`` read the template with its rendered values -- the slots that
carry a person's own text (a name, a path, a template key) are masked with
a neutral ``x`` -- and every character of the real rendering must be
printable ASCII. A failure raises ``CopyRefused(key)``, so a line that fails
its own check is never printed.

The slot types:

* ``int`` -- a whole number in 0..=2^53-1, in digits;
* ``("hex", n)`` -- exactly ``n`` lowercase hexadecimal digits;
* ``when`` -- ``YYYY-MM-DD HH:MM (+HH:MM)``; ``offset`` -- ``+HH:MM``;
* ``code`` -- one value of a closed map built when the line is rendered
  (``codes(slot)``): the gestures, the laws this engine carries, the params
  of a proposal, the genesis's symbols and where each came from, the two
  engines, the reasons a store gives, and the refusal codes of the classes'
  own ``CODES``;
* ``params`` -- a mapping of param names to whole numbers; ``names`` -- a
  sequence of param names;
* ``name`` -- a person's text; ``path`` -- a file path, escaped with
  ``backslashreplace``; ``key`` -- a template key. These three are masked.

``WEB`` holds the lines only the API says: its refusals, and the line a
capped view adds after its label. They pass the same nets; ``web(key)``
renders one as a checked ``Line`` (they have no slots), and
``WEB_FALLBACK`` is the one fixed string said when a line fails its check.

``REFUSALS`` is the closed refusal table: for each refusal class the
garden serves, each of its codes names the lines it is said with and its
exit. ``describe.refusal`` renders it; no exception text and no detail ever
reaches a stream. A refusal raised after its verb's write returned is said
as such (``refuse.written``, or ``refuse.copy.written`` for a line that
failed its check), never as a write that did not happen.

Nothing is imported at module level but the standard library.
"""

import re
from typing import NamedTuple

checkpoint_before_apply = True

MAX_INT = (1 << 53) - 1

TEMPLATES = {
    "status.disabled": "The garden is off. It is on only when its settings file says enabled: true:",
    "status.disabled.unreadable": (
        "The garden is off: its settings file cannot be read, and a switch that cannot be read is off:"),
    "status.disabled.kept": "An onion sown here keeps its record, and its time runs on while the garden is off.",
    "status.stopped": (
        "The emergency stop is on in this server: the garden computes nothing until it is lifted. The time of an "
        "onion sown here runs on."),
    "status.stopped.unknown": (
        "The emergency stop cannot be read in this server, so the garden computes nothing. The time of an onion "
        "sown here runs on."),
    "status.unavailable": "The garden cannot open its store: {reason}.",
    "status.unavailable.view": "The garden cannot compute this onion now: {reason}.",
    "status.awaiting_soil": (
        "A seed has no soil yet: a seed is planted only in an encrypted store (a readable master key and "
        "SQLCipher)."),
    "status.awaiting_soil.bulbe": "A glass jar is never offered while Bulbe's rules apply.",
    "status.ready": "No onion has been sown here (oo garden sow).",
    "status.resting": "Resting in the shed.",
    "status.sealed_bulbe": "Its glass jar stays closed while Bulbe's rules apply; nothing of it is shown.",
    "status.missing": "Its store is missing. It was not composted.",
    "status.missing.finish": "An interrupted sowing of this onion was found: oo garden keep finish {tag}",
    "status.missing.nothing": "There is no way to restore it in this version.",
    "status.retired_prototype": (
        "Its life can no longer be computed. Its record is kept as it is; there is no way yet to compost it."),
    "status.unreadable": "Its record could not be verified. It is not shown rather than shown wrong.",
    "status.unreadable.resume": (
        "It can resume from its last verified event (#{kept}); {discarded} later events would be deleted for "
        "good: oo garden keep resume {kept} {discarded}"),
    "status.unreadable.nothing": (
        "No verified event is left to resume from; there is no way to restore it in this version."),
    "status.unreadable.store": "The garden leaves its store as it is: {reason}.",
    "label.prototype": (
        "Prototype: this onion lives under a provisional world. It may have to be composted when the stable "
        "world arrives."),
    "label.prototype.short": "Prototype (provisional world): it may have to be composted later.",
    "label.retired": "Prototype: its provisional world was retired from this engine.",
    "label.glass_jar": "This onion lives in a glass jar: its store is not encrypted.",
    "label.catching_up": "Shown as of {when}: its life after that is not computed yet.",
    "label.frozen": "Shown as of {when}: the engine stopped on a fault after that. Nothing was replaced.",
    "label.bulbe": "Bulbe: this machine is in Bulbe mode. Its life goes on as in Daily.",
    "label.mode_unknown": "The security mode cannot be read, so Bulbe's rules apply. Its life goes on.",
    "doctrine": (
        "This is a simulation of an onion. It does not feel anything; what it does follows the laws its "
        "laboratory names (oo garden lab laws). It will never ask you to come back."),
    "sow.sees": "It is a seed; in this version it does not grow further. Nothing about it leaves this machine.",
    "sow.glass": (
        "No master key is configured, so it would live in a glass jar: its store would not be encrypted "
        "(require_encryption: false)."),
    "sow.place": (
        "Hemisphere {hemisphere} ({hemisphere_from}), daylight band {band} ({band_from}), weather {weather} "
        "({weather_from}). None of them changes after sowing."),
    "sow.windowsill": "No rain reaches a windowsill: only the water you give.",
    "sow.rhythm": "Learning your rhythm is not offered in this version: a seed sown now never learns it.",
    "sow.exit": (
        "There is no way yet to compost it or put it to rest. Turning the garden off hides it; its record stays "
        "on this machine."),
    "sow.ask.name": (
        "A name, or an empty line for none: 1 to 32 letters, digits, spaces, hyphens, apostrophes or periods. "
        "Every name given stays in its record for good."),
    "sow.ask.confirm": "One seed; no preview and no second draw. Sow it? Answer yes or no:",
    "sow.done": "Beetle: Sown. It is a seed, day 0.",
    "sow.identity": "Genome {genome}. Being {tag}.",
    "sow.cancelled": "Nothing was sown.",
    "sow.unnamed": "It was sown without its name. oo garden keep name names it.",
    "keep.ask.name": (
        "A name: 1 to 32 letters, digits, spaces, hyphens, apostrophes or periods. Every name given stays in its "
        "record for good."),
    "keep.named": "Beetle: Named {name}.",
    "care.noted": "Beetle: {act} noted.",
    "laws.carried": "Beetle: With it, a law update to {law} version {v}, in force from day {day}.",
    "verify.chain": "Deep verification: {facts} events over {days} days of life; the chain is intact.",
    "verify.kept": "{agreed} of {kept} kept states agree with a replay on the reference engine.",
    "verify.none": "No kept state to compare yet.",
    "verify.stale": "{stale} kept states were stale and skipped.",
    "verify.ahead": "{ahead} kept states are past the minute shown now and were skipped.",
    "verify.served": "The state shown now agrees with the reference replay ({engine} engine).",
    "verify.diverged.kept": (
        "A kept state on day {day}, law version {v}, disagrees with the reference replay. Nothing was replaced."),
    "verify.diverged.blob": (
        "A kept state on day {day}, law version {v}, does not hold what it names. Nothing was replaced."),
    "verify.diverged.served": (
        "The state shown now, day {day}, law version {v}, disagrees with the reference replay ({engine} engine). "
        "Nothing was replaced."),
    "verify.diverged.after": "Until this is looked into, what the garden shows may start from a kept state.",
    "verify.refused": "The engine refused ({code}). Nothing was replaced.",
    "lab.record": "Its record was checked when it was opened: {events} events, chain intact.",
    "lab.readings": "Readings: oo garden lab laws.",
    "lab.title": "Laws of its world",
    "lab.born.provisional": "Born under {law} version {v}, provisional, digest {digest}.",
    "lab.born.stable": "Born under {law} version {v}, digest {digest}.",
    "lab.in_force": "In force: {law} version {v}; params {params}.",
    "lab.pinned": "Pinned: no law update applies until oo garden keep laws unpin.",
    "lab.unpinned": "Not pinned.",
    "lab.pending": "A law update to {law} version {v} takes effect on day {day}.",
    "lab.available": (
        "A law update to {law} version {v} is available: the next gesture from this terminal writes it, unless "
        "oo garden keep laws pin keeps the laws in force."),
    "lab.history.update": "Day {day} (#{seq}): a law update to {law} version {v}, in force from day {from_day}.",
    "lab.history.pin": "Day {day} (#{seq}): pinned.",
    "lab.history.unpin": "Day {day} (#{seq}): unpinned.",
    "lab.history.none": "No law update has been written.",
    "lab.proposal.same": "Your allium.yaml laws proposal is the same as the params in force.",
    "lab.proposal.differs": (
        "Your allium.yaml laws proposal differs in {names}. It applies only through oo garden keep laws diff, "
        "then apply."),
    "lab.proposal.refused": "Your allium.yaml laws proposal cannot be used, so nothing is compared. The file is:",
    "lab.offset": (
        "Its local time follows the offset last recorded ({recorded}); this machine reads {now}. The next "
        "gesture records it."),
    "laws.diff.head": "A law update from your allium.yaml proposal, to {law} version {v}:",
    "laws.diff.row": "  {param}: {before} -> {after}",
    "laws.diff.confirm": (
        "Written with: oo garden keep laws apply {confirm}. It takes effect at the next local midnight after it "
        "is written."),
    "laws.diff.nothing": "Your allium.yaml laws proposal is what is in force or pending; there is nothing to apply.",
    "laws.applied": "Beetle: Law update noted; in force from day {day}.",
    "laws.pinned": "Beetle: Pinned to the laws in force.",
    "laws.unpinned": "Beetle: Unpinned.",
    "beetle.unreadable": "Beetle: The trail of stars breaks at event #{seq}. Nothing was replaced.",
    "path.line": "  {path}",
    "later": "Not in this version; nothing was read or written.",
    "refuse.argv": "Text is never read from the command line; what was given is not repeated.",
    "refuse.subtopic": "No such subtopic; the words given are not repeated.",
    "refuse.later": "Not in this version; nothing was read or written.",
    "refuse.share.none": "No consent request is open, and none can be opened in this version. Nothing was written.",
    "refuse.attended": "This is done only from an interactive terminal in the foreground. Nothing was written.",
    "refuse.answer": "Answer yes or no. Nothing was sown.",
    "refuse.eof": "The answers ended early: sowing reads two lines, a name and then yes or no. Nothing was sown.",
    "refuse.eof.name": "No name was given. Nothing was written.",
    "refuse.name": (
        "A name is 1 to 32 letters, digits, spaces, hyphens, apostrophes or periods, starts with a letter or a "
        "digit and not with Beetle. Nothing was written."),
    "refuse.line": "An answer is one line of at most 255 characters. Nothing was written.",
    "refuse.exists": "An onion is already sown here, or was: there is one seed.",
    "refuse.budget": "Not written: the budget for this kind of fact on this day of its life is spent.",
    "refuse.laws.pinned": "Pinned: no law update applies until oo garden keep laws unpin. Nothing was written.",
    "refuse.laws.unpinned": "It is not pinned. Nothing was written.",
    "refuse.laws.nothing": "Your allium.yaml laws proposal is what is in force or pending. Nothing was written.",
    "refuse.laws.confirm": "That code is not the one the diff shows now. Nothing was written.",
    "refuse.laws.params": "Your allium.yaml laws proposal cannot be used. Nothing was written. The file is:",
    "refuse.laws.law": "Its law is not carried by this engine. Nothing was written.",
    "refuse.sowing": "The sowing answers in allium.yaml cannot be used. Nothing was sown. The file is:",
    "refuse.resume.nothing": "There is nothing to resume.",
    "refuse.resume.confirm": "Those numbers are not the ones the garden shows now. Nothing was written.",
    "refuse.finish.nothing": "There is no interrupted sowing to finish.",
    "refuse.finish.confirm": "That tag is not the one the garden shows now. Nothing was written.",
    "refuse.channel": (
        "No law of this garden's channel is both carried and not retired by this engine. Nothing was sown."),
    "refuse.copy": "A line of the garden's text failed its own check ({key}). Nothing was shown.",
    "refuse.copy.written": "A line of the garden's text failed its own check ({key}); the write itself was done.",
    "refuse.written": "Refused ({code}) after the write; the write itself was done.",
    "refuse.clock": "The clock cannot be read, or reads before its birth. Nothing was written.",
    "refuse.clock.sow": "The clock cannot be read, or reads before the second day of 1970. Nothing was sown.",
    "refuse.offset": (
        "The local offset cannot be read as whole quarter hours within 14 hours of UTC. Nothing was sown."),
    "refuse.plaintext": "The new store was not written encrypted; it was removed. Nothing was sown.",
    "refuse.store": "The garden cannot use its store: {reason}. Nothing was written.",
    "refuse.owner": "This account does not own this onion. Nothing was written.",
    "refuse.membrane": "The membrane refused this write ({code}). Nothing was written.",
    "refuse.engine": "The engine refused ({code}). Nothing was written.",
    "refuse.other": "Refused ({code}). Nothing was written.",
    "reason.account": "this machine requires an account, or its accounts cannot be read, and none was given",
    "reason.path": "persistence.path in allium.yaml is refused, or no data directory is reachable",
    "reason.key": (
        "a master key is configured and this process cannot read it (a key file under a passphrase opens only "
        "with OPTI_KEYFILE_PASSPHRASE set)"),
    "reason.cipher": "a master key is readable and SQLCipher is not available",
    "reason.soil": "its store is not the kind of file its name says",
    "reason.pages": "the pages of its store are damaged",
    "reason.anchor_unwritten": "its birth could not be anchored in the audit log",
    "reason.audit": "the audit log cannot be read or does not verify",
    "reason.unanchored": "its birth is not anchored in the audit log",
    "reason.foreign": "this onion was sown under another key",
    "reason.unsown": "the audit log names another onion",
    "reason.busy": "another writer holds the store",
    "reason.divergence": "a kept state does not hold what it names",
    "reason.owner": "its store names another owner",
    "reason.law": "its law is not carried by this engine",
    "reason.local": "the records of its own store cannot be read",
    "reason.unclaimed": "accounts exist on this machine, and the onion sown before them belongs to no account yet",
    "reason.clock": "the clock cannot be read, or reads before its birth",
    "reason.engine": "the engine refused to serve even its last kept state",
}

# The lines only the API says, beside the catalogue: its refusals, by the code of each, and the line a view the
# API capped says after its label. None has a slot.
WEB = {
    "web.host": (
        "This garden answers only a request addressed to this machine as 127.0.0.1, localhost or [::1], or by a "
        "name listed in api.hosts in allium.yaml. No garden was opened, and nothing in it was written."),
    "web.origin": (
        "This garden answers only a request whose Origin is this machine, or a name listed in api.hosts in "
        "allium.yaml over https, with no path. No garden was opened, and nothing in it was written."),
    "web.site": (
        "This garden answers no request made by a page of another site. No garden was opened, and nothing in it "
        "was written."),
    "web.sign_in": "The garden answers only a signed-in session. No garden was opened, and nothing in it was written.",
    "web.auth_unavailable": (
        "The garden cannot check the request's account: the authentication module is not available. No garden "
        "was opened, and nothing in it was written."),
    "web.request": (
        "The garden does not read this request: its fields are not the ones this route takes. No garden was "
        "opened, and nothing in it was written."),
    "web.fault": "The garden could not answer: an unexpected fault. Nothing in it was written.",
    "web.catching_up": "The next write made in oo garden computes it.",
}
# The one fixed string the API says when one of its own lines fails its check.
WEB_FALLBACK = "The garden could not answer."

# The type of every slot of the catalogue.
SLOTS = {
    "act": "code", "after": "int", "agreed": "int", "ahead": "int", "band": "code", "band_from": "code",
    "before": "int", "code": "code", "confirm": ("hex", 16), "day": "int", "days": "int", "digest": ("hex", 12),
    "discarded": "int", "engine": "code", "events": "int", "facts": "int", "from_day": "int",
    "genome": ("hex", 12), "hemisphere": "code", "hemisphere_from": "code", "kept": "int", "key": "key",
    "law": "code", "name": "name", "names": "names", "now": "offset", "param": "code", "params": "params",
    "path": "path", "reason": "code", "recorded": "offset", "seq": "int", "stale": "int", "tag": ("hex", 8),
    "v": "int", "weather": "code", "weather_from": "code", "when": "when",
}
# The slots whose values are a person's text or a key: the lexical nets read an ``x`` in their place.
MASKED = ("name", "path", "key")
# The description's own lines, which a ``key`` slot may name besides the catalogue's (``json``: the one line
# ``show --json`` prints).
LINE_KEYS = ("art", "identity", "conditions", "json")

# The gestures, as the acknowledgement says them.
ACTS = {"greet": "Greet", "play": "Play", "warm": "Warm", "water": "Water"}
# The genesis's own symbols (the journal table's genesis body), in the order the table gives them.
HEMISPHERES = ("north", "south")
BANDS = ("long", "medium", "short")
WEATHERS = ("garden", "windowsill")
# Where a sowing field came from: an option on the command line, the settings file, or the default.
SOURCES = ("option", "file", "default")
ENGINES = ("native", "reference")
# The reasons the garden gives for a store it cannot open or an onion it cannot compute.
REASONS = ("account", "path", "key", "cipher", "soil", "pages", "anchor_unwritten", "audit", "unanchored", "foreign",
           "unsown", "busy", "divergence", "owner", "law", "local", "unclaimed", "clock", "engine")

# The closed refusal table. For each class, each code gives the lines it is said with and its exit; ``*``
# stands for every other code of that class. A line key starting with ``@`` is a form the renderer builds:
# ``@disabled`` and ``@stopped`` (the switch's and the stop's form), ``@form`` (the status form of the look
# the refusal concerns), ``@unreadable`` (the broken record's form), ``@labels`` (the being's label lines),
# ``@path`` (the settings file's path line). ``BY_VERB`` overrides an entry for one verb.
REFUSALS = {
    "ServiceRefused": {
        "off": (("@disabled",), 1),
        "stopped": (("@stopped",), 1),
        "status": (("@form",), 1),
        "exists": (("refuse.exists", "@form"), 1),
        "channel": (("refuse.channel",), 1),
        "name": (("refuse.name",), 2),
        "offset": (("refuse.offset",), 1),
        "attended": (("refuse.attended",), 1),
        "consent": (("refuse.share.none",), 1),
        "budget": (("refuse.budget",), 1),
    },
    "StoreRefused": {
        "plaintext": (("refuse.plaintext",), 1),
        "no_soil": (("status.awaiting_soil",), 2),
        "exists": (("refuse.exists",), 1),
        "sealed": (("status.sealed_bulbe", "@labels"), 1),
        "retired": (("status.retired_prototype", "@labels"), 1),
        # Every other code is said as the store it names, when the catalogue has its reason.
        "*": (("refuse.store",), 1),
    },
    "ChainRefused": {"*": (("@unreadable",), 1)},
    "MembraneRefused": {
        "clock": (("refuse.clock",), 1),
        "owner": (("refuse.owner",), 1),
        "attended": (("refuse.attended",), 1),
        "*": (("refuse.membrane",), 1),
    },
    "LawsRefused": {
        "pinned": (("refuse.laws.pinned",), 1),
        "unpinned": (("refuse.laws.unpinned",), 1),
        "nothing": (("refuse.laws.nothing",), 1),
        "confirm": (("refuse.laws.confirm",), 1),
        "law": (("refuse.laws.law",), 1),
        "params": (("refuse.laws.params", "@path"), 1),
        "sowing": (("refuse.sowing", "@path"), 1),
    },
    "LifeRefused": {"*": (("refuse.engine",), 1)},
    "ResumeRefused": {
        "nothing": (("refuse.resume.nothing",), 1),
        "confirm": (("refuse.resume.confirm",), 1),
    },
    "Diverged": {
        "kept": (("verify.diverged.kept", "verify.diverged.after"), 1),
        "blob": (("verify.diverged.blob", "verify.diverged.after"), 1),
        "served": (("verify.diverged.served", "verify.diverged.after"), 1),
    },
    "Dropped": {"budget": (("refuse.budget",), 1)},
    "CopyRefused": {"*": (("refuse.copy",), 1)},
}
BY_VERB = {
    ("MembraneRefused", "clock", "sow"): (("refuse.clock.sow",), 1),
    ("MembraneRefused", "body", "name"): (("refuse.name",), 2),
    ("LifeRefused", "*", "verify"): (("verify.refused",), 1),
    ("ResumeRefused", "nothing", "finish"): (("refuse.finish.nothing",), 1),
    ("ResumeRefused", "confirm", "finish"): (("refuse.finish.confirm",), 1),
}

_HEX = frozenset("0123456789abcdef")
_WHEN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} \([+-][0-9]{2}:[0-9]{2}\)")
_OFFSET = re.compile(r"[+-][0-9]{2}:[0-9]{2}")
_SLOT = re.compile(r"\{(\w+)\}")
_PRINTABLE = re.compile(r"[ -~]*")


class Line(NamedTuple):
    """One line of a form: the catalogue key (or the description's own), and its text, unwrapped."""

    key: str
    text: str


class CopyRefused(ValueError):
    """A line of the catalogue that failed its own check; ``key`` names it. ``written``: after a write."""

    def __init__(self, key, written=False):
        super().__init__(key)
        self.key = key
        self.written = written is True


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _refusal_codes():
    """Every refusal code of the classes the garden serves, read from their own ``CODES`` now."""
    from . import evolution, membrane, service, store, wire

    groups = (service.ServiceRefused.CODES, store.StoreRefused.CODES, store.ChainRefused.REASONS,
              membrane.MembraneRefused.CODES, evolution.LawsRefused.CODES, wire.REFUSALS,
              store.ResumeRefused.CODES, ("budget",))
    out = []
    for group in groups:
        for code in group:
            if code not in out:
                out.append(code)
    return tuple(out)


def codes(slot):
    """The closed map of a ``code`` slot, built now: the values ``say`` accepts for it, in order."""
    if slot == "act":
        return tuple(sorted(ACTS))
    if slot == "law":
        from . import lawfiles

        return tuple(lawfiles.LAWS)
    if slot == "param":
        from . import settings

        return tuple(settings.PARAM_KEYS)
    if slot == "hemisphere":
        return HEMISPHERES
    if slot == "band":
        return BANDS
    if slot == "weather":
        return WEATHERS
    if slot in ("hemisphere_from", "band_from", "weather_from"):
        return SOURCES
    if slot == "engine":
        return ENGINES
    if slot == "reason":
        return REASONS
    if slot == "code":
        return _refusal_codes()
    raise KeyError(slot)


def _source(slot, value):
    if value == "option":
        return "from --" + slot[:-len("_from")]
    if value == "file":
        return "from allium.yaml"
    return "the default"


def _render(key, slot, value):
    """``(text, masked)`` for one slot's value; ``CopyRefused(key)`` when it is not of the slot's type."""
    kind = SLOTS.get(slot)
    if kind is None:
        raise CopyRefused(key)
    if kind == "int":
        if not _is_int(value) or not 0 <= value <= MAX_INT:
            raise CopyRefused(key)
        return str(value), False
    if isinstance(kind, tuple) and kind[0] == "hex":
        if not isinstance(value, str) or len(value) != kind[1] or not set(value) <= _HEX:
            raise CopyRefused(key)
        return value, False
    if kind == "when":
        if not isinstance(value, str) or not _WHEN.fullmatch(value):
            raise CopyRefused(key)
        return value, False
    if kind == "offset":
        if not isinstance(value, str) or not _OFFSET.fullmatch(value):
            raise CopyRefused(key)
        return value, False
    if kind == "code":
        if not isinstance(value, str) or value not in codes(slot):
            raise CopyRefused(key)
        if slot == "act":
            return ACTS[value], False
        if slot.endswith("_from"):
            return _source(slot, value), False
        if slot == "reason":
            return TEMPLATES["reason." + value], False
        return value, False
    if kind in ("params", "names"):
        return _render_params(key, kind, value), False
    if kind == "name":
        if not isinstance(value, str) or not value:
            raise CopyRefused(key)
        return value, True
    if kind == "path":
        try:
            text = str(value).encode("ascii", "backslashreplace").decode("ascii")
        except (TypeError, ValueError):
            raise CopyRefused(key) from None
        return text, True
    if kind == "key":
        if not isinstance(value, str) or (value not in TEMPLATES and value not in LINE_KEYS):
            raise CopyRefused(key)
        return value, True
    raise CopyRefused(key)


def _render_params(key, kind, value):
    names = codes("param")
    if kind == "params":
        if not isinstance(value, dict) or not value:
            raise CopyRefused(key)
        for name, number in value.items():
            if name not in names or not _is_int(number) or not 0 <= number <= MAX_INT:
                raise CopyRefused(key)
        return ", ".join(f"{name} {value[name]}" for name in sorted(value))
    if not isinstance(value, (list, tuple)) or not value or len(set(value)) != len(value):
        raise CopyRefused(key)
    if any(not isinstance(name, str) or name not in names for name in value):
        raise CopyRefused(key)
    return ", ".join(value)


def say(key, /, **slots):
    """The catalogue's line ``key`` with ``slots`` rendered, as a checked ``Line``; ``CopyRefused(key)`` if not.

    ``KeyError`` for a key the catalogue does not hold: that is a defect of
    the caller, never a line to print.
    """
    from . import ethics

    template = TEMPLATES[key]
    wanted = set(_SLOT.findall(template))
    if set(slots) != wanted:
        raise CopyRefused(key)
    rendered = {}
    lexical = {}
    for slot in wanted:
        text, masked = _render(key, slot, slots[slot])
        rendered[slot] = text
        lexical[slot] = "x" if masked else text
    out = _SLOT.sub(lambda match: rendered[match.group(1)], template)
    if ethics.check(_SLOT.sub(lambda match: lexical[match.group(1)], template)):
        raise CopyRefused(key)
    if not _PRINTABLE.fullmatch(out):
        raise CopyRefused(key)
    return Line(key, out)


def web(key):
    """The API's own line ``key`` (``WEB``) as a checked ``Line``; ``CopyRefused(key)`` when it fails its check.

    ``KeyError`` for a key ``WEB`` does not hold: that is a defect of the
    caller, never a line to serve.
    """
    from . import ethics

    text = WEB[key]
    if not isinstance(text, str) or ethics.check(text) or not _PRINTABLE.fullmatch(text):
        raise CopyRefused(key)
    return Line(key, text)
