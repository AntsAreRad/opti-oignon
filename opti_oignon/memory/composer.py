#!/usr/bin/env python3
"""The window composer: registry state plus budget plus retrieval set gives a prompt.

A pure function. Given the system prompt, the Core, the receipt ledger and
its Cellar, the retrieval set, the Flesh, the tail and the current turn, it
assembles the window in a fixed order -- system, Core, receipts digest,
selected Peels, Flesh, tail, current turn -- under the caps of one rung of
``onion.yaml`` and returns the same prompt for the same inputs, having
mutated none of them. Two kinds of layer get two kinds of
treatment. Peels are a selection, so a retrieval set that does not fit is
cut to whole items and the cut is counted. Core, Flesh and the current turn
are not the composer's to cut: an oversize Core is refused because its bytes
must be the registry's, an oversize Flesh is refused because a turn leaves
the window only with a receipt, an oversize turn is refused because
truncating what the user just said is a lie about the question; the system
prompt and the tail are refused by name the same way, never cut.

Every recalled segment -- receipts, Peels, Flesh, the tail -- carries its
provenance and is marked as data, never as instruction; only the system
prompt, the Core and the current turn bear instruction. Rendering keeps the
tag. The token estimator is a seam: the default is the retriever's own
fallback and a caller injects its own count, so the window is counted once,
with the counter of the turn it is sent in. The budgets are one table per
rung of the context ladder (``ctx_ladder``, read from the governor's file)
and one reserve for all; a configuration where a rung has no table, a table
holds for no rung, a layer is missing or unknown, a table does not hold at
its rung, or a layer shrinks as the rung grows, is refused by name. A caller
that names no rung composes under ``home_rung``'s table without the system
prompt and the tail, which it does not lay, nor the room they hold: the
window before rungs, cap for cap.

The files the onion reads, ``onion.yaml`` and the governor's, are parsed
once for each text of the file (``read_onion``): every read reads the file
as the loaders read it before, and a text the kept parse was made from is
not parsed again. Every loader of the onion -- the budgets, the journal and
the ladder here, the gate, the queue's ladder and the librarian's own --
reads that parse, each with a copy of its own, and a file that does not
read is refused as each loader refused it.
"""

import re
import threading
from dataclasses import dataclass, replace
from pathlib import Path

checkpoint_before_apply = True

LAYERS = ("system", "core", "receipts", "peels", "flesh", "tail", "turn")
_INSTRUCTION_BEARING = frozenset({"system", "core", "turn"})
_CONFIG = Path(__file__).resolve().parent.parent / "config" / "onion.yaml"
# The governor's file, read as a file for its ladder: the governor's module is
# not the onion's to import.
_GOVERNOR_CONFIG = Path(__file__).resolve().parent.parent / "config" / "resource_governor.yaml"

# A frame marker found inside a segment's own text: the closing tag, bare or
# with attributes, or the opening bracket of a frame with its first attribute.
# Only ``render`` writes frames, so one found in a text was forged there and is
# defanged; a bracket holding the bare word is ordinary code and stays.
_FRAME_RE = re.compile(
    r"\[\s*/\s*data(?:\s*\]|\s+[^\]\n]*\]?)|\[\s*data(?:\s*:\s*|\s+)[\"']?\w+[\"']?\s*=[^\]\n]*\]?",
    re.IGNORECASE,
)
_FRAME_REDACTED = "[redacted-frame-marker]"
# The start of a marker left open at the very end of a text: segments are
# joined by blank lines, which a marker may hold, so ``[data`` closing one
# segment and ``]`` opening the next would make a frame between them. An
# opening is read on up to its first attribute's name, whose sign the next
# segment may hold. Its spaces are read one way only, so a long run of them
# costs their length, not its square.
_FRAME_TAIL_RE = re.compile(r"\[\s*(?:/\s*)?(?:data(?:(?:\s*:\s*|\s+)(?:[\"']?\w+[\"']?\s*)?)?)?$", re.IGNORECASE)
# What a frame header may carry from a tag: no bracket, no line break.
_TAG_UNSAFE = re.compile(r"[\[\]\r\n]")
# The head of a frame marker -- the closing tag, closed at once or not, or the
# opening bracket with its first attribute and its sign -- and no word after
# it: every match of ``_FRAME_RE`` opens on one, so a text with none left
# holds no marker, and the words a marker left open would have run on to stay.
# The same language as that pattern's head, its spaces read one way only.
_FRAME_HEAD_RE = re.compile(r"\[\s*/\s*data(?:\s*\])?|\[\s*data(?:\s*:\s*|\s+)[\"']?\w+[\"']?\s*=", re.IGNORECASE)
# The head of a marker of the untrusted-data envelope the executor wraps the
# window in -- its tag's name, open or closing, with the bracket that closes it
# at once -- and no word after it. The wrapper's pattern reads a marker from
# such a name to the next ``>``, across lines: a name left in the window would
# take every later byte with it, references included. Its spaces are read one
# way only.
_ENVELOPE_HEAD_RE = re.compile(r"</?\s*untrusted_data\b(?:\s*(?:/\s*)?>)?", re.IGNORECASE)
# The start of such a marker left open at the very end of a text: its
# bracket, with the slash of a closing tag or not -- the name may open the
# next part, past the line break that joins them.
_ENVELOPE_TAIL_RE = re.compile(r"</?\s*$")
_ENVELOPE_REDACTED = "[redacted-untrusted-marker]"


def defanged(text):
    """``text`` as the window and its envelope carry it, and how many markers that took.

    Each marker is defanged by its head alone: a frame's, then one left open
    at the very end, then the envelope's tag's name, then the start of one
    left open at the very end -- the words after it are kept, so a marker the
    user typed costs the reader its name and nothing more. A text defanged
    here holds no head the composer or the wrapper read a marker from, nor the
    start of one a part after it could finish, so both pass it unchanged: a
    caller that shows it shows the bytes the model reads, and can say how many
    it changed.
    """
    count = 0

    def counted(redacted):
        def sub(_match):
            nonlocal count
            count += 1
            return redacted
        return sub

    text = _FRAME_HEAD_RE.sub(counted(_FRAME_REDACTED), str(text))
    text = _FRAME_TAIL_RE.sub(counted(_FRAME_REDACTED), text)
    text = _ENVELOPE_HEAD_RE.sub(counted(_ENVELOPE_REDACTED), text)
    text = _ENVELOPE_TAIL_RE.sub(counted(_ENVELOPE_REDACTED), text)
    return text, count


class BudgetError(ValueError):
    """The window cannot be assembled under this budget without cutting what may not be cut.

    ``motive`` says what refused it, from a closed set -- a layer of
    ``LAYERS``, or ``window``, ``rung``, ``count``, ``config``, ``document``
    or ``wrap`` -- and ``figures`` its whole numbers, the count then the cap
    where there are both: what a caller shows of a refusal, never its words.
    """

    def __init__(self, message="", *, motive="", figures=()):
        super().__init__(message)
        self.motive = motive
        self.figures = tuple(figures)


def estimate_tokens(text):
    """The fallback estimate, the same as the retriever's; a tokenizer replaces it."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


def _whole(value):
    """Whether ``value`` is a whole number of tokens: an integer, never a bool, never negative."""
    return type(value) is int and value >= 0


@dataclass(frozen=True)
class Budget:
    window: int
    reserve: int
    core: int
    receipts: int
    peels: int
    flesh: int
    turn: int
    system: int = 0
    tail: int = 0
    # The share of the Core's cap a pin, a supersession or an accepted
    # proposal leaves free: the Core is counted with a turn's counter, and the
    # next turn may count it with another model's.
    core_margin: float = 0.0

    def core_room(self):
        """What the Core may grow to: its cap, less the share held back for the counts of other models and, with
        any share held back, one token for the counts' rounding down."""
        return self.core - int(self.core * self.core_margin) - (1 if self.core_margin else 0)

    def validate(self):
        """Every reason this budget cannot be assembled under, or an empty list."""
        errors = []
        for name in ("window", "reserve") + LAYERS:
            value = getattr(self, name)
            if not _whole(value):
                errors.append(f"{name}: {value!r} is not a non-negative integer")
        margin = self.core_margin
        if type(margin) not in (int, float) or not 0.0 <= margin < 1.0:
            errors.append(f"core_margin: {margin!r} is not a share in [0, 1)")
        if errors:
            return errors
        if self.core <= 0 or self.turn <= 0:
            errors.append("core and turn caps must be positive: both are refused when over, never cut")
        caps = sum(getattr(self, layer) for layer in LAYERS)
        if caps + self.reserve > self.window:
            errors.append(
                f"oversubscribed: layer caps {caps} + reserve {self.reserve} exceed the window {self.window}"
            )
        return errors


@dataclass(frozen=True)
class Budgets:
    """One budget per rung of the context ladder, one reserve for all, and the rung whose table a caller that
    names none composes under."""

    reserve: int
    home: int
    tables: tuple

    @property
    def rungs(self):
        return tuple(rung for rung, _budget in self.tables)

    def at(self, num_ctx):
        """The budget of the largest rung at or under ``num_ctx``; refused by name under the smallest."""
        if not _whole(num_ctx):
            raise BudgetError(f"num_ctx {num_ctx!r} is not a whole number of tokens", motive="rung")
        held = [budget for rung, budget in self.tables if rung <= num_ctx]
        if not held:
            raise BudgetError(f"num_ctx {num_ctx} is under the smallest rung {self.rungs[0]}: no table holds there",
                              motive="rung", figures=(num_ctx, self.rungs[0]))
        return held[-1]

    def unnamed(self):
        """What a caller that names no rung composes under: ``home_rung``'s table without the system prompt and
        the tail, which such a caller does not lay, nor the room they hold."""
        home = self.at(self.home)
        return replace(home, window=home.window - home.system - home.tail, system=0, tail=0)


# The one parse kept of each file the onion reads, by the file's absolute path, the last ``_PARSES`` of them, keyed
# on the text it was made from. Every read reads the file by the call each loader made before any parse was kept, so
# a read that fails fails there, as it did; a text equal to the kept one is served the kept parse, and any other is
# parsed, and kept unless the parse refused anything: a refusal is never kept, and the next read parses again. Equal
# text is a test no clock, attribute cache or file system can get wrong, and nothing of the file but its text is
# read. One reload at a time, the kept parse looked for again under the lock; a read of a kept text takes no lock.
_PARSES = 8
_parses = {}
_parse_lock = threading.Lock()


@dataclass(frozen=True)
class _Parse:
    """One parse of a file: the text it was made from (``key``), and what each reader takes of it -- the document
    ``yaml.safe_load`` builds, and the keys named twice that the budgets' and the journal's readers find on the
    composed node -- each as ``(value, refusal)``."""

    key: str
    document: tuple
    budgets: tuple
    journal: tuple

    @property
    def whole(self):
        """Whether every reader reads it: no refusal on the way."""
        return all(refusal is None for _value, refusal in (self.document, self.budgets, self.journal))

    def value(self):
        """The document, a copy of its own for each caller; raises what refused it."""
        import copy

        value, refusal = self.document
        if refusal is not None:
            raise refusal
        return copy.deepcopy(value)

    def twice(self, reader):
        """The keys ``reader`` -- ``budgets`` or ``journal`` -- finds named twice, a new list; raises what refused it."""
        found, refusal = getattr(self, reader)
        if refusal is not None:
            raise refusal
        return list(found)


def _attempt(call, *args):
    """``(call(*args), None)``, or ``(None, what it raised)``."""
    try:
        return call(*args), None
    except Exception as exc:  # noqa: BLE001 - kept, and raised by the reader that reads it
        return None, exc


def _read_twice(check, root, name):
    """``check`` read on ``root`` with a constructor of its own: a key one reader cannot build stops no other."""
    import yaml

    loader = yaml.SafeLoader("")
    try:
        return check(loader, root, name)
    finally:
        loader.dispose()


def _parse(text, name, key=None):
    """``text`` composed once and read by every reader of the onion: the keys named twice read off the composed
    node first, the budgets' and the journal's -- a merge key rewrites the node as the document is built -- then the
    document; a text the reader or the composer refuses is refused to every reader."""
    import yaml

    try:
        loader = yaml.SafeLoader(text)
    except Exception as exc:  # noqa: BLE001 - kept, and raised by the reader that reads it
        refused = (None, exc)
        return _Parse(key, refused, refused, refused)
    try:
        root, refusal = _attempt(loader.get_single_node)
        if refusal is not None:
            refused = (None, refusal)
            return _Parse(key, refused, refused, refused)
        budgets = _attempt(_read_twice, _named_twice, root, name)
        journal = _attempt(_read_twice, _journal_twice, root, name)
        document = _attempt(lambda: loader.construct_document(root) if root is not None else None)
    finally:
        loader.dispose()
    return _Parse(key, document, budgets, journal)


def _parsed(target):
    """The parse of the file at ``target`` as it reads now: the one kept for its text, or a new one.

    The file is read as each loader read it before any parse was kept, and
    its text compared with the one the kept parse was made from: a parse is
    made only for a text none was kept for, and kept only when it refused
    nothing.
    """
    import os

    target = Path(target)
    text = target.read_text(encoding="utf-8")
    spelled = os.path.abspath(target)
    key = text
    held = _parses.get(spelled)
    if held is not None and held.key == key:
        return held
    with _parse_lock:
        held = _parses.get(spelled)
        if held is not None and held.key == key:
            return held
        parse = _parse(text, target.name, key)
        _parses.pop(spelled, None)
        if parse.whole:
            _parses[spelled] = parse
            while len(_parses) > _PARSES:
                _parses.pop(next(iter(_parses)))
        return parse


def read_onion(path=None):
    """The onion's file -- ``onion.yaml``, or the file at ``path`` -- as ``yaml.safe_load`` reads it, from the one
    parse kept for the file's text, and refused as that read refuses it. Every caller has a copy of its own: what
    one does to it reaches no other caller, nor the next read."""
    return _parsed(path or _CONFIG).value()


def read_ladder(path=None):
    """The context ladder of the governor's file, ascending, and "" -- or None and why it cannot be read.

    The onion holds the ladder to what was written: a list of distinct
    positive whole numbers. The governor reads it more loosely -- it keeps
    its own default where its file holds no list, an empty one, or a value
    it cannot read as a positive integer; it truncates a value it can read
    as a number (4096.0, "4096", 4096.7); it keeps a rung named twice -- and
    the onion refuses each of these by name: a budget for a ladder nobody
    wrote is a guess. The file is read from its one parse (``_parsed``).
    """
    target = Path(path or _GOVERNOR_CONFIG)
    try:
        raw = _parsed(target).value() or {}
    except Exception as exc:  # noqa: BLE001 - said, and refused by name
        return None, f"ctx_ladder unreadable in {target.name}: {type(exc).__name__}"
    ladder = raw.get("ctx_ladder") if isinstance(raw, dict) else None
    if not isinstance(ladder, list) or not ladder:
        return None, f"ctx_ladder unreadable in {target.name}: missing or empty"
    if not all(_whole(rung) and rung > 0 for rung in ladder):
        return None, f"ctx_ladder unreadable in {target.name}: {ladder!r} holds a value that is no positive whole number"
    if len(set(ladder)) != len(ladder):
        return None, f"ctx_ladder unreadable in {target.name}: {ladder!r} names a rung twice"
    return tuple(sorted(ladder)), ""


def _named_twice(loader, root, name):
    """The keys named twice where the budgets are read -- at the top of the file, in ``budgets`` and in each of its
    tables -- which a load takes at their last value without a word: keys compared as the load builds them, read
    off the composed node before a merge key rewrites it; a merge key is a merge, never a key named twice."""
    import yaml

    found = []

    def keys(node, where):
        if not isinstance(node, yaml.MappingNode):
            return {}
        held = {}
        for key, value in node.value:
            if key.tag == "tag:yaml.org,2002:merge" or not isinstance(key, yaml.ScalarNode):
                continue
            built = loader.construct_object(key)
            if built in held:
                found.append(f"{where} names {built!r} twice: the last would win without a word")
            held[built] = value
        return held

    rows = keys(keys(root, name).get("budgets"), "budgets")
    for rung, row in rows.items():
        keys(row, f"the table of {rung}")
    return found


def load_budgets(path=None, *, governor_path=None):
    """The budgets of ``onion.yaml``, one per rung of the governor's ladder; refused by name, every reason at once."""
    target = Path(path or _CONFIG)
    parse = _parsed(target)
    errors = parse.twice("budgets")
    raw = parse.value() or {}
    if not isinstance(raw, dict):
        raise BudgetError(f"{target.name} holds no mapping: no budget to read", motive="config")
    ladder, why = read_ladder(governor_path)
    if why:
        errors.append(why)
    if "window" in raw or "layers" in raw:
        errors.append("window and layers are gone: the budget is a table per rung (budgets)")
    reserve = raw.get("reserve")
    if not _whole(reserve):
        errors.append(f"reserve: {reserve!r} is not a non-negative integer")
    margin = raw.get("core_margin")
    margin_reads = type(margin) in (int, float) and 0.0 <= margin < 1.0
    if not margin_reads:
        errors.append(f"core_margin: {margin!r} is not a share in [0, 1)")
    rows = raw.get("budgets")
    if not isinstance(rows, dict) or not rows:
        errors.append("budgets: no table per rung")
        rows = {}
    tables = {}
    for key, row in rows.items():
        if not (_whole(key) and key > 0):
            errors.append(f"table key {key!r} is no positive whole number of tokens")
            continue
        if ladder is not None and key not in ladder:
            errors.append(f"table {key} is for no rung of ctx_ladder {list(ladder)}")
        if not isinstance(row, dict):
            errors.append(f"rung {key}: its table is not a mapping of layers")
            continue
        unknown = sorted(str(name) for name in row if name not in LAYERS)
        missing = [name for name in LAYERS if name not in row]
        errors.extend(f"rung {key}: unknown layer {name!r}" for name in unknown)
        errors.extend(f"rung {key}: {name} is missing" for name in missing)
        if unknown or missing:
            # The caps it does hold are still read: every reason at once.
            errors.extend(f"rung {key}: {name}: {row[name]!r} is not a non-negative integer"
                          for name in LAYERS if name in row and not _whole(row[name]))
            continue
        # A reserve or a margin that does not read is refused on its own; the
        # tables are still read, held at their rungs and compared.
        budget = Budget(window=key, reserve=reserve if _whole(reserve) else 0, **{name: row[name] for name in LAYERS},
                        core_margin=margin if margin_reads else 0.0)
        found = budget.validate()
        errors.extend(f"rung {key}: {error}" for error in found)
        if not found:
            tables[key] = budget
    if ladder is not None:
        errors.extend(f"rung {rung} of ctx_ladder {list(ladder)} has no table" for rung in ladder if rung not in rows)
    ordered = sorted(tables.items())
    for (low, below), (high, above) in zip(ordered, ordered[1:]):
        for name in LAYERS:
            if getattr(above, name) < getattr(below, name):
                errors.append(f"rung {high}: {name} {getattr(above, name)} is under rung {low}'s "
                              f"{getattr(below, name)}: a layer never shrinks as the rung grows")
    home = raw.get("home_rung")
    if not (_whole(home) and home in rows and (ladder is None or home in ladder)):
        errors.append(f"home_rung {home!r} is no rung with a table")
    if errors:
        raise BudgetError("; ".join(errors), motive="config")
    return Budgets(reserve=reserve, home=home, tables=tuple(ordered))


def load_budget(path=None, *, rung=None, governor_path=None):
    """The budget of ``onion.yaml`` at ``rung`` -- the largest rung at or under it -- or, when none is named,
    what a caller that names none composes under (``Budgets.unnamed``)."""
    budgets = load_budgets(path, governor_path=governor_path)
    return budgets.unnamed() if rung is None else budgets.at(rung)


@dataclass(frozen=True)
class Peel:
    """One retrieved summary, with where it came from."""

    text: str
    provenance: str
    score: float = 0.0


@dataclass(frozen=True)
class Segment:
    layer: str
    text: str
    provenance: str
    tokens: int
    instruction_bearing: bool


@dataclass(frozen=True)
class Prompt:
    segments: tuple
    tokens: int
    core_root: str
    dropped_peels: int
    # How many open receipts the digest folded into one line to keep the
    # receipts layer within its cap.
    folded_receipts: int = 0

    def render(self):
        """The window as text. Recalled segments are framed as quoted data with their tag.

        No segment text opens or closes a frame: a marker found in any of
        them, the Core and the turn included, is defanged first, so the only
        frames in the window are the ones written here. Nor does one hold the
        name of the envelope's tag, which the wrapper reads a marker from up to
        its next ``>``: the names are defanged here, each alone, so no segment
        takes the bytes of the next ones with it; nor the start of either
        marker left open at its end, which the next segment -- the turn after
        the Core, with no frame between them -- could finish.
        """
        parts = []
        for seg in self.segments:
            text = _defang(seg.text)
            parts.append(text if seg.instruction_bearing else _framed(seg.layer, seg.provenance, text))
        return "\n\n".join(parts)


def _defang(text):
    """``text`` with every frame and envelope marker defanged, and the start of either left open at its end."""
    text = _FRAME_TAIL_RE.sub(_FRAME_REDACTED, _FRAME_RE.sub(_FRAME_REDACTED, str(text)))
    return _ENVELOPE_TAIL_RE.sub(_ENVELOPE_REDACTED, _ENVELOPE_HEAD_RE.sub(_ENVELOPE_REDACTED, text))


def _framed(layer, provenance, text):
    """A recalled part as the window frames it, quoted data with its tag: ``text`` defanged already."""
    layer = _TAG_UNSAFE.sub("", str(layer))
    provenance = _TAG_UNSAFE.sub("", str(provenance))
    return f"[data layer={layer} provenance={provenance}]\n{text}\n[/data]"


def _segment(layer, text, provenance, estimate):
    return Segment(
        layer=layer,
        text=text,
        provenance=provenance,
        tokens=estimate(text),
        instruction_bearing=layer in _INSTRUCTION_BEARING,
    )


def _refuse_over(layer, tokens, cap, why):
    if tokens > cap:
        raise BudgetError(f"{layer} is {tokens} tokens against a cap of {cap}: {why}", motive=layer,
                          figures=(tokens, cap))


# The words of a refusal over a cap, as both assemblies write them: its layer, then its count and its cap.
_OVER_RE = re.compile(r"(\w+) is (\d+) tokens against a cap of (\d+):")


def _refused_in(words):
    """The refusal ``words`` say, its motive and figures read back from them; words that name no cap are the
    count's."""
    found = _OVER_RE.match(words)
    if found is None:
        return BudgetError(words, motive="count")
    return BudgetError(words, motive=found.group(1), figures=(int(found.group(2)), int(found.group(3))))


# The shape of the native assembly's arguments: a core whose COMPOSE_FORMAT
# says another is not asked, and the reference assembles.
_COMPOSE_FORMAT = 2
# A count stays under this: far past any window, and low enough that no sum
# of parts leaves the native core's integers.
_MOST_TOKENS = 2 ** 32

_WHY = {
    "system": "the system prompt is never cut",
    "core": "the Core is never cut, it is the registry's bytes",
    "receipts": "resolve or archive receipts first",
    "flesh": "a turn leaves the window only with a receipt; evict first, the composer drops nothing",
    "tail": "the tail is never cut here; its caller fits its sources to the cap",
    "turn": "the current turn is never truncated",
    "window": "the generation reserve is not assembled into",
}


def _native_compose():
    """The native assembly of counted parts, or None: asked at the call, never at import, and only of a core that
    takes the parts' shape."""
    try:
        from opti_oignon.native import load
    except Exception:  # noqa: BLE001 - absence is the reference path
        return None
    core = load()
    if core is None or getattr(core, "COMPOSE_FORMAT", None) != _COMPOSE_FORMAT:
        return None
    return getattr(core, "compose_counted", None)


@dataclass(frozen=True)
class _Parts:
    """Every part of one window, in order, each counted once: ``(text, provenance, tokens)``, or None for a part
    that is not laid."""

    system: tuple | None
    core: tuple
    receipts: tuple | None
    peels: tuple
    flesh: tuple
    tail: tuple | None
    turn: tuple
    core_root: str
    folded: int


def _part(layer, text, provenance, estimate):
    tokens = estimate(text)
    if not (_whole(tokens) and tokens < _MOST_TOKENS):
        raise BudgetError(f"the counter gave {tokens!r} for a part of the {layer}: no whole number of tokens "
                          f"under {_MOST_TOKENS}", motive="count")
    return text, provenance, tokens


def _count(core, ledger, cellar, retrieval, flesh, turn, budget, estimate, system, tail):
    """The parts of the window, each counted once, before anything is laid: both assemblies take the same counts."""
    core_text = core.text()
    core_root = core.root()
    # The ledger folds its digest with the same counter, each of its counts
    # held to what a part's count is held to.
    digest, folded = ledger.render(
        cellar, cap=budget.receipts, estimate=lambda text: _part("receipts", text, "receipts", estimate)[2],
    )
    turns = flesh.turns() if hasattr(flesh, "turns") else [dict(t) for t in flesh]
    return _Parts(
        system=_part("system", system, "system", estimate) if system else None,
        core=_part("core", core_text, f"core:{core_root}", estimate),
        receipts=_part("receipts", digest, "receipts", estimate) if digest else None,
        peels=tuple(_part("peels", peel.text, peel.provenance, estimate) for peel in retrieval),
        flesh=tuple(
            _part("flesh", str(t.get("text", "")), f"flesh:{t.get('turn_id', '')}:{t.get('role', '')}", estimate)
            for t in turns
        ),
        tail=_part("tail", tail, "tail", estimate) if tail else None,
        turn=_part("turn", turn, "turn", estimate),
        core_root=core_root,
        folded=folded,
    )


def _assemble(parts, budget):
    """The reference assembly of counted parts: ``(segments, total, dropped)``.

    What may not be cut is refused by name over its cap; the peels are cut
    to whole items and the cut is counted; the total is held under the
    window less the reserve.
    """
    segments = []

    def lay(layer, part, refused=True):
        text, provenance, tokens = part
        if refused:
            _refuse_over(layer, tokens, getattr(budget, layer), _WHY[layer])
        segments.append(Segment(layer, text, provenance, tokens, layer in _INSTRUCTION_BEARING))

    if parts.system is not None:
        lay("system", parts.system)
    lay("core", parts.core)
    # The digest folds its oldest lines to keep within its cap, so the
    # receipts layer can no longer refuse the block; the check stays as the
    # statement of the bound.
    if parts.receipts is not None:
        lay("receipts", parts.receipts)
    used, dropped = 0, 0
    for part in parts.peels:
        if used + part[2] > budget.peels:
            dropped += 1
            continue
        used += part[2]
        lay("peels", part, refused=False)
    _refuse_over("flesh", sum(part[2] for part in parts.flesh), budget.flesh, _WHY["flesh"])
    for part in parts.flesh:
        lay("flesh", part, refused=False)
    if parts.tail is not None:
        lay("tail", parts.tail)
    lay("turn", parts.turn)
    total = sum(s.tokens for s in segments)
    _refuse_over("window", total, budget.window - budget.reserve, _WHY["window"])
    return segments, total, dropped


def _assemble_natively(assemble, parts, budget):
    """The same assembly through the native core; its refusals are the reference's words. A text or a number the
    core cannot carry -- a lone surrogate, a cap past its integers -- goes to the reference."""

    def carried(part):
        return None if part is None else (str(part[0]), str(part[1]), part[2])

    try:
        segments, total, dropped = assemble(
            carried(parts.system),
            carried(parts.core),
            carried(parts.receipts),
            [carried(part) for part in parts.peels],
            [carried(part) for part in parts.flesh],
            carried(parts.tail),
            carried(parts.turn),
            (budget.window, budget.reserve) + tuple(getattr(budget, layer) for layer in LAYERS),
        )
    except (UnicodeError, OverflowError):
        return _assemble(parts, budget)
    except ValueError as exc:
        raise _refused_in(str(exc)) from None
    return [Segment(*segment) for segment in segments], total, dropped


def compose(*, core, ledger, cellar, retrieval, flesh, turn, budget, estimate=None, system="", tail=""):
    """Assemble the window. Pure: same inputs, same prompt, inputs untouched.

    ``system`` opens the window and bears instruction; ``tail`` -- the
    sources a turn brings, as one text -- comes after the Flesh, before the
    turn, framed as data like every recalled part; each is laid only when
    given, and refused by name over its cap, never cut. Every part is
    counted once, with ``estimate`` or the default, before anything is
    laid; the counts are then assembled by the native core when it is
    present and takes their shape, by the reference otherwise.
    """
    errors = budget.validate()
    if errors:
        raise BudgetError("; ".join(errors), motive="config")
    parts = _count(core, ledger, cellar, retrieval, flesh, turn, budget, estimate or estimate_tokens, system, tail)
    assemble = _native_compose()
    if assemble is not None:
        segments, total, dropped = _assemble_natively(assemble, parts, budget)
    else:
        segments, total, dropped = _assemble(parts, budget)
    return Prompt(
        segments=tuple(segments), tokens=total, core_root=parts.core_root, dropped_peels=dropped,
        folded_receipts=parts.folded,
    )


# ---------------------------------------------------------------------------
# The journal: a turn's window as messages
# ---------------------------------------------------------------------------

# The layers of a journal, in the order its bytes lay them: what changes least
# comes first, so the engine reuses the longest prefix from one turn to the next.
JOURNAL = ("system", "core", "peels", "receipts", "flesh", "tail", "turn")
# The header the Core is laid under. Written here alone: one found in any other
# part was forged there, and is neutralized. Its words are found whatever root
# precedes them, or none: the model cannot check a root.
_CORE_HEADER = "Core {root}, pinned by the user:"
_CORE_HEADER_RE = re.compile(r"\bpinned\s+by\s+the\s+user\s*:", re.IGNORECASE)
# A header's words can run across a joint, their start ending one bare part and their rest beginning the next:
# the end of the part before is read for a start and the head of the part after for the rest, each within this
# reach of the joint, so a joint costs the same whatever the parts' lengths.
_JOINT_REACH = 256
_HEADER_START_RE = re.compile(r"\bpinned(?:\s+by(?:\s+the(?:\s+user)?)?)?\s*\Z", re.IGNORECASE)
# What completes a header after its first one, two, three or four words, at the very start of a part.
_HEADER_REST_RE = {
    1: re.compile(r"\A\s*by\s+the\s+user\s*:", re.IGNORECASE),
    2: re.compile(r"\A\s*the\s+user\s*:", re.IGNORECASE),
    3: re.compile(r"\A\s*user\s*:", re.IGNORECASE),
    4: re.compile(r"\A\s*:"),
}
_CORE_HEADER_REDACTED = "[redacted-core-header]"
# The kinds of source a turn's tail brings, in the order the current turn's
# message lays them, each held to its share of the tail's cap.
TAIL_KINDS = ("memory", "rag", "web")
_SOURCE = {"memory": "memory", "rag": "file", "web": "web"}


@dataclass(frozen=True)
class Message:
    """One message of a journal: its role, its content, the labels of the parts it holds and their layers."""

    role: str
    content: str
    labels: tuple = ()
    layers: tuple = ()


@dataclass(frozen=True)
class Journal:
    """A turn's window as messages, the same bytes from one turn to the next up to the current turn's message.

    ``tokens`` is what the request costs: each message's count and the
    overhead of a message, and the charge of the turn's images. ``layers``
    holds what each layer counts, in ``JOURNAL`` order, then that charge
    (``images``), and ``cut`` what each kind of part left out.
    ``pieces`` place every part in ``canonical()``, ``(layer, key, start,
    end)`` in bytes, tiling it, the joint before a part counted with it;
    ``stable_end`` is where the current turn's part begins, the bytes before
    it the prefix the next turn keeps.
    """

    messages: tuple
    tokens: int
    layers: tuple
    cut: tuple
    core_root: str
    stable_end: int
    pieces: tuple

    def canonical(self):
        """The journal as bytes: each message its role, a NUL, its content in UTF-8 and a record separator."""
        return b"".join(_canonical(message.role, message.content) for message in self.messages)


def _canonical(role, content):
    return role.encode("utf-8") + b"\x00" + str(content).encode("utf-8", "surrogatepass") + b"\x1e"


class _Laid:
    """A journal's messages as they are laid: each a role, its parts ``(layer, key, text, joint)`` and their labels.

    A user message laid after another is one message with it, its first part
    joined to the last: two user messages are never adjacent.
    """

    def __init__(self):
        self.messages = []
        self.mark = None

    def joined(self, role, text):
        """``text`` as a part of ``role`` laid next: a user part that joins the user message before it has the rest
        of a header that message's last part begins neutralized at its start (``_joined``); what is laid already
        keeps its bytes."""
        if role == "user" and self.messages and self.messages[-1][0] == "user" and self.messages[-1][1]:
            return _joined(self.messages[-1][1][-1][2], text)
        return text

    def enter(self, role, parts, labels=(), *, stable=False):
        parts = [part for part in parts if part[2]]
        if not parts:
            return
        if self.messages and role == "user" and self.messages[-1][0] == "user":
            layer, key, text, _joint = parts[0]
            parts[0] = (layer, key, text, "\n\n")
            held = self.messages[-1]
        else:
            held = [role, [], []]
            self.messages.append(held)
        if stable:
            self.mark = (len(self.messages) - 1, len(held[1]))
        held[1].extend(parts)
        held[2].extend(labels)

    def serialized(self):
        """``(messages, pieces, stable_end)``: the pieces tile the canonical bytes, each a part with the joint before
        it, the first of a message with its role, the last with its separator."""
        messages, pieces, offset, stable_end = [], [], 0, None
        for index, (role, parts, labels) in enumerate(self.messages):
            texts = []
            for place, (layer, key, text, joint) in enumerate(parts):
                piece = (joint if place else "") + text
                texts.append(piece)
                size = len(piece.encode("utf-8", "surrogatepass"))
                if place == 0:
                    size += len(role.encode("utf-8")) + 1
                if self.mark == (index, place):
                    stable_end = offset
                pieces.append([layer, key, offset, offset + size])
                offset += size
            offset += 1
            pieces[-1][3] = offset
            messages.append(Message(role, "".join(texts), tuple(labels),
                                    tuple(dict.fromkeys(layer for layer, _key, _text, _joint in parts))))
        return tuple(messages), tuple(tuple(piece) for piece in pieces), offset if stable_end is None else stable_end


def _bare(text):
    """A part the journal lays as it is: defanged, and a Core header found in it neutralized."""
    return _CORE_HEADER_RE.sub(_CORE_HEADER_REDACTED, _defang(text))


def _joined(before, text):
    """``text`` as it is laid after ``before`` across a joint: when the end of ``before`` starts a Core header that
    the start of ``text`` completes, the completion is neutralized at the start of ``text``; ``before`` keeps its
    bytes. Only the ends near the joint are read."""
    start = _HEADER_START_RE.search(before[-_JOINT_REACH:])
    if start is None:
        return text
    rest = _HEADER_REST_RE[len(start.group(0).split())].match(text, 0, _JOINT_REACH)
    if rest is None:
        return text
    return _CORE_HEADER_REDACTED + text[rest.end():]


def region_cut(region, budget, count):
    """How many of ``region``'s oldest items fall out of it: the newest whole items it holds within the peels' cap,
    each counted by its text, as ``compose`` counts a peel. ``region`` is ``(key, text, provenance, label)`` in the
    tree's order."""
    used, kept = 0, 0
    for key, text, _provenance, _label in reversed(tuple(region)):
        tokens = _part("peels", _bare(text), key, count)[2]
        if used + tokens > budget.peels:
            break
        used += tokens
        kept += 1
    return len(tuple(region)) - kept


def _wrapped(wrap, text, source, frames=False):
    if wrap is None:
        raise BudgetError(f"a {source} part has no envelope to ride in: the journal lays no data bare", motive="wrap")
    return wrap(text, source=source, frames=frames)


def _rendered(pieces, wrap):
    """A turn's pieces as its message lays them: words bare, a document in a file envelope at its place."""
    merged = []
    for kind, text in pieces:
        if not str(text):
            continue
        if merged and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1] + str(text))
        else:
            merged.append((kind, str(text)))
    return [(kind, _wrapped(wrap, _bare(text), "file") if kind == "file" else _bare(text)) for kind, text in merged]


def turn_text(pieces, wrap):
    """A turn as its message lays it: its pieces rendered (``_rendered``) and joined by a blank line."""
    return "\n\n".join(text for _kind, text in _rendered(pieces, wrap))


def _fold_part(fold, wrap):
    """A shadow's fold as the memory lays it: ``(its item, the item alone in the memory's envelope)``, bare as every
    item of the tail is."""
    _kind, _key, text, frame, _label = fold
    item = _framed(frame[0], frame[1], _bare(text))
    return item, _wrapped(wrap, item, "memory", frames=True)


def fold_cost(fold, wrap, count):
    """What a shadow's fold costs laid alone among the memory, its envelope included: what the Flesh room its
    eviction freed must hold of it."""
    return _part("tail", _fold_part(fold, wrap)[1], "tail", count)[2]


def compose_journal(*, system, pinned, region, digest, flesh, tail, turn, budget, count, overhead, wrap, shares,
                    prompt_cap=None, tail_room=None, images=0, image_tokens=0, fold=None):
    """A turn's window as a journal under ``budget``. Pure: same inputs, same journal, inputs untouched.

    The system prompt, and a manifest when given (``system``), open it, one
    message each. The next message holds the Core under its header (``pinned``,
    its root and its text), then one memory envelope holding the region --
    the peels the tree accepted, in its order, the newest that fit the
    peels' cap -- and the receipts digest, then the first Flesh turn when it
    is the user's. The Flesh follows in its own roles (``flesh``: key, role,
    pieces, label), each document in a file envelope at its place, its words
    bare. The last message holds the tail (``tail``: kind, key, text, frame,
    label, the memory framed as data) -- memory, retrieval, web, each cut
    to whole items within its share of the tail's cap (``shares``) and
    counted in its envelope -- then the current turn (``turn``: pieces,
    label) laid as the Flesh lays a turn, its documents held to the
    document share. Every bare part is defanged, and a Core header forged in
    any part but the Core is neutralized, and one forged across the joint
    where two bare parts meet at the start of the second. The turn's ``images`` ride beside
    its message, never in its text: each is charged ``image_tokens``,
    counted on the turn's message and so on the whole, the counter reading
    no image. The current turn comes first: their charge takes its room
    from the Flesh, held to its cap less the charge, and a charge over the
    whole Flesh cap is refused ``document``.

    A shadow's ``fold`` -- one memory item naming the turns its eviction set
    aside -- is laid first among the memory and charged to the Flesh room
    that eviction freed: the Flesh is held to its cap less the fold, the
    tail's cap and ``tail_room`` raised by it, the memory share left whole
    for what follows it; a fold the Flesh cannot hold beside the turn's
    images is refused ``flesh``.

    Every part is counted once with ``count``: a peel, the digest and a
    turn's words by their text, as ``compose`` counts them, a Flesh turn and
    the tail as laid, envelopes included; the caps are held by the
    composer's own assembly, native or reference, the region cut first so it
    drops nothing; the whole, each message's count and ``overhead``, frames
    and envelopes included, by the window less the reserve, and by
    ``prompt_cap`` when it is lower (the refusal's figures are the whole's
    count and the cap). ``tail_room``, when given, is what the tail may take
    in all: each kind, in order, within its share and what the kinds before
    it left of it. What may not be cut is refused by name
    (``BudgetError``, with its motive and figures).
    """
    errors = budget.validate()
    if errors:
        raise BudgetError("; ".join(errors), motive="config")
    if not (_whole(overhead) and overhead < _MOST_TOKENS):
        raise BudgetError(f"a message's overhead {overhead!r} is no whole number of tokens", motive="config")
    if not (_whole(image_tokens) and image_tokens < _MOST_TOKENS):
        raise BudgetError(f"an image's charge {image_tokens!r} is no whole number of tokens", motive="config")
    if not _whole(images):
        raise BudgetError(f"a turn's images {images!r} are no whole number", motive="turn")
    image_cost = images * image_tokens
    if image_cost > budget.flesh:
        raise BudgetError(f"images are {image_cost} tokens against the Flesh's cap of {budget.flesh}: a turn's "
                          "images take their room from the Flesh, never cut", motive="document",
                          figures=(image_cost, budget.flesh))
    # The current turn comes first: the Flesh is held to what its images leave of its cap.
    laid_under = replace(budget, flesh=budget.flesh - image_cost)
    # A shadow's fold is charged to the Flesh room the eviction it names freed, outside the memory share.
    fold_item, fold_alone, charge = None, "", 0
    if fold is not None:
        fold_item, fold_alone = _fold_part(fold, wrap)
        charge = _part("tail", fold_alone, "tail", count)[2]
        if image_cost + charge > budget.flesh:
            raise BudgetError(f"a shadow's fold of {charge} tokens has no room in the Flesh's cap of {budget.flesh} "
                              f"beside images of {image_cost}: no turn is set aside without its trace",
                              motive="flesh", figures=(image_cost + charge, budget.flesh))
        laid_under = replace(laid_under, flesh=laid_under.flesh - charge, tail=laid_under.tail + charge)
        if tail_room is not None:
            # The fold's room comes from the Flesh: the tail's kinds keep the room they were given.
            tail_room += charge

    def counted(layer, text):
        return _part(layer, text, layer, count)[2]

    laid, cut = _Laid(), {}
    system = (system,) if isinstance(system, str) else (system or ())
    system = tuple(str(text) for text in system if str(text))
    for place, text in enumerate(system):
        laid.enter("system", [("system", f"system:{place}", text, "")])
    system_tokens = sum(counted("system", text) for text in system)
    root, core_text = pinned
    core_part = f"{_CORE_HEADER.format(root=str(root)[:16])}\n{_defang(core_text)}" if str(core_text) else ""
    core_tokens = counted("core", core_part) if core_part else 0
    region = tuple(region)
    cut["peels_full"] = region_cut(region, budget, count)
    held = region[cut["peels_full"]:]
    frames = [("peels", key, _framed("peels", provenance, _bare(text)), label) for key, text, provenance, label in held]
    peel_tokens = tuple(counted("peels", _bare(text)) for _key, text, _provenance, _label in held)
    digest_frame = _framed("receipts", "receipts", _bare(digest)) if str(digest) else ""
    digest_tokens = counted("receipts", _bare(digest)) if digest_frame else 0
    if digest_frame:
        frames.append(("receipts", "digest", digest_frame, None))
    memory = []
    if frames:
        content = "\n\n".join(frame for _layer, _key, frame, _label in frames)
        envelope = _wrapped(wrap, content, "memory", frames=True)
        at = envelope.find(content)
        # A wrapper that rewrites what it wraps lays its envelope whole, one piece.
        if at < 0:
            memory = [("peels", "envelope", envelope, "\n\n")]
        else:
            memory = ([("peels", "envelope:open", envelope[:at], "\n\n")]
                      + [(layer, key, frame, "\n\n" if place else "")
                         for place, (layer, key, frame, _label) in enumerate(frames)]
                      + [("receipts", "envelope:close", envelope[at + len(content):], "")])
    laid.enter("user", ([("core", str(root), core_part, "")] if core_part else []) + memory,
             [label for _layer, _key, _frame, label in frames if label is not None])
    flesh_parts = []
    for key, role, pieces, label in flesh:
        text = laid.joined(str(role), turn_text(pieces, wrap))
        flesh_parts.append((text, f"flesh:{key}", counted("flesh", text) + overhead))
        laid.enter(str(role), [("flesh", str(key), text, "")], [label])
    tail_text, tail_tokens, tail_labels = [], 0, []
    by_kind = {kind: [item for item in tail if item[0] == kind] for kind in TAIL_KINDS + ("citation",)}
    for kind in TAIL_KINDS:
        cap = int(shares.get(kind, 0) * budget.tail)
        if tail_room is not None:
            cap = min(cap, max(tail_room - tail_tokens, 0))
        kept, dropped, placed = [], 0, ""
        if kind == "memory" and fold_item is not None:
            # The fold first, never cut, on the room its eviction freed: the share holds what follows it.
            kept, placed = [(fold_item, fold[4])], fold_alone
            cap = cap + charge if tail_room is None else min(cap + charge, tail_room)
        for _kind, key, text, frame, label in by_kind[kind]:
            item = _framed(frame[0], frame[1], _bare(text)) if frame else _bare(text)
            if kept:
                # Each item is bare alone; the rest of a header the item before it begins is neutralized here.
                item = _joined(kept[-1][0], item)
            trial = _wrapped(wrap, "\n\n".join([item for item, _label in kept] + [item]), _SOURCE[kind],
                             frames=kind == "memory")
            if kind == "web" and by_kind["citation"]:
                trial += "\n\n" + _bare(by_kind["citation"][0][2])
            if counted("tail", trial) > cap:
                dropped += 1
                continue
            kept.append((item, label))
            placed = trial
        cut[kind] = dropped
        if placed:
            tail_text.append((kind, placed))
            tail_tokens += counted("tail", placed)
            tail_labels.extend(label for _item, label in kept if label is not None)
    pieces, turn_label = turn
    rendered = _rendered(pieces, wrap)
    if rendered and not tail_text:
        # With no tail before it, the turn's first part meets the message before it.
        rendered[0] = (rendered[0][0], laid.joined("user", rendered[0][1]))
    words = "\n\n".join(text for kind, text in rendered if kind != "file")
    documents = sum(counted("tail", text) for kind, text in rendered if kind == "file")
    share = int(shares.get("document", 0) * budget.tail)
    if documents > share:
        raise BudgetError(f"document is {documents} tokens against a cap of {share}: the documents of a turn are "
                          "never cut", motive="document", figures=(documents, share))
    tail_tokens += documents
    laid.enter("user", [("tail", kind, text, "\n\n") for kind, text in tail_text]
             + [("turn" if kind != "file" else "tail", f"turn:{place}", text, "\n\n")
                for place, (kind, text) in enumerate(rendered)], tail_labels + [turn_label], stable=True)
    parts = _Parts(
        system=("\n\n".join(system), "system", system_tokens) if system else None,
        core=(core_part, f"core:{root}", core_tokens),
        receipts=(digest_frame, "receipts", digest_tokens) if digest_frame else None,
        peels=tuple((frame, key, tokens) for (_layer, key, frame, _label), tokens in zip(frames, peel_tokens)),
        flesh=tuple(flesh_parts),
        tail=("\n\n".join(text for _kind, text in tail_text), "tail", tail_tokens) if tail_tokens else None,
        turn=(words, "turn", counted("turn", words) if words else 0),
        core_root=str(root),
        folded=0,
    )
    assemble = _native_compose()
    _segments, _total, dropped = (_assemble_natively(assemble, parts, laid_under) if assemble is not None
                                  else _assemble(parts, laid_under))
    if dropped:
        raise BudgetError(f"{dropped} peels left out after the region was cut to its cap", motive="count")
    messages, laid_pieces, stable_end = laid.serialized()
    # The turn's images ride on its message: their charge is counted on the whole.
    tokens = sum(counted("window", message.content) + overhead for message in messages) + image_cost
    cap = budget.window - budget.reserve if prompt_cap is None else min(budget.window - budget.reserve, prompt_cap)
    _refuse_over("window", tokens, cap, _WHY["window"])
    layers = (("system", system_tokens), ("core", core_tokens), ("peels", sum(peel_tokens)),
              ("receipts", digest_tokens), ("flesh", sum(part[2] for part in flesh_parts)), ("tail", tail_tokens),
              ("turn", parts.turn[2]), ("images", image_cost))
    return Journal(messages=messages, tokens=tokens, layers=layers, cut=tuple(sorted(cut.items())),
                   core_root=str(root), stable_end=stable_end, pieces=laid_pieces)


def over_cap(budget, counts, document_share=0.0, image_charge=0):
    """The first of ``counts`` -- ``(layer, tokens)`` -- that ``budget`` does not hold, as ``(layer, tokens, cap)``;
    None when it holds them all. The documents are held to ``document_share`` of the tail's cap; a turn's images
    (``images``, their charge) to the whole Flesh cap, whose room they take, and the Flesh to what they leave of
    it, its cap less ``image_charge``."""
    for layer, tokens in counts:
        if layer == "document":
            cap = int(document_share * budget.tail)
        elif layer == "images":
            cap = budget.flesh
        else:
            cap = getattr(budget, layer) - (image_charge if layer == "flesh" else 0)
        if tokens > cap:
            return layer, tokens, cap
    return None


def layer_cap(budget, layer, image_charge=0):
    """What ``budget`` holds of ``layer``, the Flesh less what a turn's images take of it (``image_charge``)."""
    return getattr(budget, layer) - (image_charge if layer == "flesh" else 0)


def at_mark(budget, mark):
    """``budget`` with the Flesh held to ``mark`` of its cap, rounded down: where a burst aims once a journal laid
    the conversation, so the Flesh is evicted before it reaches its cap."""
    import math

    return replace(budget, flesh=math.floor(mark * budget.flesh))


def accounted(tokens, margin):
    """What a window of ``tokens`` is held to in its context: its count raised by ``margin``, rounded up."""
    import math

    return math.ceil(tokens * (1 + margin))


def prompt_cap(budget, num_ctx, margin, answer_floor):
    """The most a window may count sent with ``num_ctx`` so that, accounted with ``margin`` (``accounted``), the
    answer keeps at least ``answer_floor`` tokens: never past the window less the reserve."""
    import math

    room = num_ctx - answer_floor
    cap = math.floor(room / (1 + margin))
    # The division rounds: a cap raised again by the margin can pass the room by a token, and is lowered until
    # its own account leaves the floor whole.
    while cap > 0 and accounted(cap, margin) > room:
        cap -= 1
    return min(budget.window - budget.reserve, cap)


@dataclass(frozen=True)
class JournalConfig:
    """How a journal is laid and sent: the share of its count held back for the counter's error, a message's
    overhead in tokens, the share of the tail's cap each kind of source may take, the room the answer always
    keeps (``answer_floor``) and what one image of a turn is charged, its room taken from the Flesh
    (``image_tokens``). How a plan catches the Flesh up when it finds it over its cap: the turns past the
    newest ``catch_up_reach`` leave bare, then the queue runs at most ``catch_up_steps`` steps with no model, then
    bare spans until it fits. Where a burst aims once a journal laid the conversation: ``burst_mark`` of the
    Flesh's cap."""

    margin: float
    message_overhead: int
    tail_shares: tuple
    answer_floor: int = 1024
    image_tokens: int = 1024
    catch_up_reach: int = 64
    catch_up_steps: int = 32
    burst_mark: float = 0.8

    def shares(self):
        return dict(self.tail_shares)


_JOURNAL_KEYS = ("margin", "message_overhead", "tail_shares", "answer_floor", "image_tokens")
# Keys a section may leave out, read at their defaults when it does: a section written before them reads alike.
_JOURNAL_OPTIONAL = ("catch_up", "burst_mark")
_CATCH_UP_KEYS = ("reach", "max_steps")
_SHARE_KINDS = TAIL_KINDS + ("document",)


def _share(value):
    return type(value) in (int, float) and 0.0 <= value <= 1.0


def _scalar_keys(loader, node):
    """The keys of a mapping node as a load builds them, each with its value node, merges aside; none for no mapping."""
    import yaml

    if not isinstance(node, yaml.MappingNode):
        return []
    return [(loader.construct_object(key), value) for key, value in node.value
            if key.tag != "tag:yaml.org,2002:merge" and isinstance(key, yaml.ScalarNode)]


def _twice(loader, node, where, only=None):
    """Each key of ``node`` named twice, which a load takes at its last value without a word; ``only`` limits the
    check to one key."""
    seen, found = {}, []
    for built, _value in _scalar_keys(loader, node):
        if built in seen and (only is None or built == only):
            found.append(f"{where} names {built!r} twice: the last would win without a word")
        seen[built] = True
    return found


def _child(loader, node, name):
    """The value node ``node`` maps ``name`` to, the last when it is named twice; None for none."""
    held = None
    for built, value in _scalar_keys(loader, node):
        if built == name:
            held = value
    return held


def _journal_twice(loader, root, name):
    """The keys named twice where the journal is read: ``journal`` at the top of the file, then in the section, in
    its ``tail_shares`` and in its ``catch_up``, read off the composed node in that order."""
    found = _twice(loader, root, name, only="journal")
    node = _child(loader, root, "journal")
    found.extend(_twice(loader, node, "journal"))
    found.extend(_twice(loader, _child(loader, node, "tail_shares"), "journal.tail_shares"))
    found.extend(_twice(loader, _child(loader, node, "catch_up"), "journal.catch_up"))
    return found


def load_journal(path=None):
    """The ``journal`` section of ``onion.yaml``; refused by name, every reason at once."""
    target = Path(path or _CONFIG)
    parse = _parsed(target)
    errors = parse.twice("journal")
    raw = parse.value() or {}
    section = raw.get("journal") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        raise BudgetError(f"{target.name} has no journal section: no journal to lay", motive="config")
    if "home_rung" in section:
        errors.append("journal: home_rung is read once, at the top of the file, never named again here")
    errors.extend(f"journal: unknown key {name!r}" for name in sorted(str(n) for n in section)
                  if name not in _JOURNAL_KEYS + _JOURNAL_OPTIONAL and name != "home_rung")
    errors.extend(f"journal: {name} is missing" for name in _JOURNAL_KEYS if name not in section)
    margin = section.get("margin")
    if "margin" in section and not (type(margin) in (int, float) and 0.0 <= margin < 1.0):
        errors.append(f"journal.margin: {margin!r} is not a share in [0, 1)")
    overhead = section.get("message_overhead")
    if "message_overhead" in section and not (_whole(overhead) and overhead < _MOST_TOKENS):
        errors.append(f"journal.message_overhead: {overhead!r} is not a whole number of tokens")
    shares = section.get("tail_shares")
    if "tail_shares" in section:
        if not isinstance(shares, dict):
            errors.append("journal.tail_shares: not a mapping of kinds to shares")
            shares = {}
        errors.extend(f"journal.tail_shares: unknown kind {name!r}" for name in sorted(str(n) for n in shares)
                      if name not in _SHARE_KINDS)
        errors.extend(f"journal.tail_shares: {kind} is missing" for kind in _SHARE_KINDS if kind not in shares)
        errors.extend(f"journal.tail_shares.{kind}: {shares[kind]!r} is not a share in [0, 1]"
                      for kind in _SHARE_KINDS if kind in shares and not _share(shares[kind]))
        if all(kind in shares and _share(shares[kind]) for kind in _SHARE_KINDS):
            total = sum(shares[kind] for kind in _SHARE_KINDS)
            if total > 1.0 + 1e-9:
                errors.append(f"journal.tail_shares: they sum to {round(total, 6)}, over the whole tail")
    floor = section.get("answer_floor")
    if "answer_floor" in section:
        tables = raw.get("budgets") if isinstance(raw.get("budgets"), dict) else {}
        rungs = sorted(rung for rung in tables if _whole(rung) and rung > 0)
        if not (_whole(floor) and floor >= 1):
            errors.append(f"journal.answer_floor: {floor!r} is not a whole number of tokens of at least 1")
        elif not rungs:
            errors.append("journal.answer_floor: no rung of budgets to hold it under")
        elif floor >= rungs[0]:
            errors.append(f"journal.answer_floor: {floor} is not under the smallest rung {rungs[0]}")
    image = section.get("image_tokens")
    if "image_tokens" in section and not (_whole(image) and 1 <= image < _MOST_TOKENS):
        errors.append(f"journal.image_tokens: {image!r} is not a whole number of tokens of at least 1")
    defaults = JournalConfig(margin=0.0, message_overhead=0, tail_shares=())
    catch_up = section.get("catch_up", {"reach": defaults.catch_up_reach, "max_steps": defaults.catch_up_steps})
    if not isinstance(catch_up, dict):
        errors.append("journal.catch_up: not a mapping of reach and max_steps")
        catch_up = {}
    errors.extend(f"journal.catch_up: unknown key {name!r}" for name in sorted(str(n) for n in catch_up)
                  if name not in _CATCH_UP_KEYS)
    errors.extend(f"journal.catch_up: {name} is missing" for name in _CATCH_UP_KEYS
                  if "catch_up" in section and name not in catch_up)
    reach, steps = catch_up.get("reach", defaults.catch_up_reach), catch_up.get("max_steps", defaults.catch_up_steps)
    if not (_whole(reach) and reach >= 1):
        errors.append(f"journal.catch_up.reach: {reach!r} is not a whole number of turns of at least 1")
    if not _whole(steps):
        errors.append(f"journal.catch_up.max_steps: {steps!r} is not a whole number of steps")
    mark = section.get("burst_mark", defaults.burst_mark)
    if not (type(mark) in (int, float) and 0.0 < mark <= 1.0):
        errors.append(f"journal.burst_mark: {mark!r} is not a share in (0, 1]")
    if errors:
        raise BudgetError("; ".join(errors), motive="config")
    return JournalConfig(margin=float(margin), message_overhead=overhead,
                         tail_shares=tuple((kind, float(shares[kind])) for kind in _SHARE_KINDS), answer_floor=floor,
                         image_tokens=image, catch_up_reach=reach, catch_up_steps=steps, burst_mark=float(mark))
