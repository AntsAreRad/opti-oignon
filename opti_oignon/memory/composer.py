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
"""

import re
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
    """The window cannot be assembled under this budget without cutting what may not be cut."""


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
            raise BudgetError(f"num_ctx {num_ctx!r} is not a whole number of tokens")
        held = [budget for rung, budget in self.tables if rung <= num_ctx]
        if not held:
            raise BudgetError(f"num_ctx {num_ctx} is under the smallest rung {self.rungs[0]}: no table holds there")
        return held[-1]

    def unnamed(self):
        """What a caller that names no rung composes under: ``home_rung``'s table without the system prompt and
        the tail, which such a caller does not lay, nor the room they hold."""
        home = self.at(self.home)
        return replace(home, window=home.window - home.system - home.tail, system=0, tail=0)


def read_ladder(path=None):
    """The context ladder of the governor's file, ascending, and "" -- or None and why it cannot be read.

    The onion holds the ladder to what was written: a list of distinct
    positive whole numbers. The governor reads it more loosely -- it keeps
    its own default where its file holds no list, an empty one, or a value
    it cannot read as a positive integer; it truncates a value it can read
    as a number (4096.0, "4096", 4096.7); it keeps a rung named twice -- and
    the onion refuses each of these by name: a budget for a ladder nobody
    wrote is a guess.
    """
    import yaml

    target = Path(path or _GOVERNOR_CONFIG)
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
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
    import yaml

    target = Path(path or _CONFIG)
    loader = yaml.SafeLoader(target.read_text(encoding="utf-8"))
    try:
        root = loader.get_single_node()
        errors = _named_twice(loader, root, target.name)
        raw = (loader.construct_document(root) if root is not None else None) or {}
    finally:
        loader.dispose()
    if not isinstance(raw, dict):
        raise BudgetError(f"{target.name} holds no mapping: no budget to read")
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
        raise BudgetError("; ".join(errors))
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
            text = _FRAME_TAIL_RE.sub(_FRAME_REDACTED, _FRAME_RE.sub(_FRAME_REDACTED, str(seg.text)))
            text = _ENVELOPE_TAIL_RE.sub(_ENVELOPE_REDACTED, _ENVELOPE_HEAD_RE.sub(_ENVELOPE_REDACTED, text))
            if seg.instruction_bearing:
                parts.append(text)
            else:
                layer = _TAG_UNSAFE.sub("", str(seg.layer))
                provenance = _TAG_UNSAFE.sub("", str(seg.provenance))
                parts.append(f"[data layer={layer} provenance={provenance}]\n{text}\n[/data]")
        return "\n\n".join(parts)


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
        raise BudgetError(f"{layer} is {tokens} tokens against a cap of {cap}: {why}")


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
                          f"under {_MOST_TOKENS}")
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
        raise BudgetError(str(exc)) from None
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
        raise BudgetError("; ".join(errors))
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
