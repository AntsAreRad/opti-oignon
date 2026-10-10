#!/usr/bin/env python3
"""The window composer: registry state plus budget plus retrieval set gives a prompt.

A pure function. Given the Core, the receipt ledger and its Cellar, the
retrieval set, the Flesh and the current turn, it assembles the window in a
fixed order -- Core, receipts digest, selected Peels, Flesh, current turn --
under the caps of ``onion.yaml`` and returns the same prompt for the same
inputs, having mutated none of them. Two kinds of layer get two kinds of
treatment. Peels are a selection, so a retrieval set that does not fit is
cut to whole items and the cut is counted. Core, Flesh and the current turn
are not the composer's to cut: an oversize Core is refused because its bytes
must be the registry's, an oversize Flesh is refused because a turn leaves
the window only with a receipt, an oversize turn is refused because
truncating what the user just said is a lie about the question.

Every recalled segment -- receipts, Peels, Flesh -- carries its provenance
and is marked as data, never as instruction; only the Core and the current
turn bear instruction. Rendering keeps the tag. The token estimator is a
seam: the default is the retriever's own fallback and a caller can inject
the real tokenizer. The budget is read from YAML and refused when the caps
plus the reserve exceed the window. Nothing on the chat path imports this
module yet, and a contract on the tree says so.
"""

import re
from dataclasses import dataclass
from pathlib import Path

checkpoint_before_apply = True

LAYERS = ("core", "receipts", "peels", "flesh", "turn")
_INSTRUCTION_BEARING = frozenset({"core", "turn"})
_CONFIG = Path(__file__).resolve().parent.parent / "config" / "onion.yaml"

# A frame marker found inside a segment's own text: the closing tag, bare or
# with attributes, or the opening bracket of a frame with its first attribute.
# Only ``render`` writes frames, so one found in a text was forged there and is
# defanged; a bracket holding the bare word is ordinary code and stays.
_FRAME_RE = re.compile(
    r"\[\s*/\s*data(?:\s*\]|\s+[^\]\n]*\]?)|\[\s*data\s*[:\s]\s*[\"']?\w+[\"']?\s*=[^\]\n]*\]?",
    re.IGNORECASE,
)
_FRAME_REDACTED = "[redacted-frame-marker]"
# The start of a marker left open at the very end of a text: segments are
# joined by blank lines, which a marker may hold, so ``[data`` closing one
# segment and ``]`` opening the next would make a frame between them.
_FRAME_TAIL_RE = re.compile(r"\[\s*(?:/\s*)?(?:data\s*)?$", re.IGNORECASE)
# The same start, an opening read on up to its first attribute's name, whose
# sign the part after it may hold: what ``defanged`` reads at a text's end.
# Its spaces are read one way only, so a long run of them costs their length,
# not its square.
_FRAME_START_RE = re.compile(r"\[\s*(?:/\s*)?(?:data(?:(?:\s*:\s*|\s+)(?:[\"']?\w+[\"']?\s*)?)?)?$", re.IGNORECASE)
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
    text = _FRAME_START_RE.sub(counted(_FRAME_REDACTED), text)
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


@dataclass(frozen=True)
class Budget:
    window: int
    reserve: int
    core: int
    receipts: int
    peels: int
    flesh: int
    turn: int

    def validate(self):
        """Every reason this budget cannot be assembled under, or an empty list."""
        errors = []
        for name in ("window", "reserve") + LAYERS:
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                errors.append(f"{name}: {value!r} is not a non-negative integer")
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


def load_budget(path=None):
    """The budget of ``onion.yaml``, refused if it does not add up."""
    import yaml

    raw = yaml.safe_load(Path(path or _CONFIG).read_text(encoding="utf-8")) or {}
    layers = raw.get("layers") or {}
    try:
        budget = Budget(
            window=int(raw["window"]),
            reserve=int(raw["reserve"]),
            **{layer: int(layers[layer]) for layer in LAYERS},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BudgetError(f"onion budget is incomplete or malformed: {exc!r}") from exc
    errors = budget.validate()
    if errors:
        raise BudgetError("; ".join(errors))
    return budget


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
        takes the bytes of the next ones with it.
        """
        parts = []
        for seg in self.segments:
            text = _FRAME_TAIL_RE.sub(_FRAME_REDACTED, _FRAME_RE.sub(_FRAME_REDACTED, str(seg.text)))
            text = _ENVELOPE_HEAD_RE.sub(_ENVELOPE_REDACTED, text)
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


def _native_compose():
    """The native assembly, or None: asked at the call, never at import."""
    try:
        from opti_oignon.native import load
    except Exception:  # noqa: BLE001 - absence is the reference path
        return None
    core = load()
    return getattr(core, "compose_segments", None) if core is not None else None


def _compose_natively(assemble, core, ledger, cellar, retrieval, flesh, turn, budget):
    """The same assembly through the native core; its refusals are the reference's words."""
    core_text = core.text()
    core_root = core.root()
    turns = flesh.turns() if hasattr(flesh, "turns") else [dict(t) for t in flesh]
    digest, folded = ledger.render(cellar, cap=budget.receipts, estimate=estimate_tokens)
    try:
        segments, total, dropped = assemble(
            core_text,
            core_root,
            digest,
            [(str(p.text), str(p.provenance)) for p in retrieval],
            [(str(t.get("text", "")), str(t.get("turn_id", "")), str(t.get("role", ""))) for t in turns],
            turn,
            (budget.window, budget.reserve, budget.core, budget.receipts, budget.peels, budget.flesh, budget.turn),
        )
    except ValueError as exc:
        raise BudgetError(str(exc)) from None
    return Prompt(
        segments=tuple(Segment(layer, text, provenance, tokens, bearing) for layer, text, provenance, tokens, bearing in segments),
        tokens=total,
        core_root=core_root,
        dropped_peels=dropped,
        folded_receipts=folded,
    )


def compose(*, core, ledger, cellar, retrieval, flesh, turn, budget, estimate=None):
    """Assemble the window. Pure: same inputs, same prompt, inputs untouched.

    With the native core present and the default estimator, the assembly is
    the core's; an injected estimator keeps the reference path, which is
    the only one that can call it.
    """
    errors = budget.validate()
    if errors:
        raise BudgetError("; ".join(errors))
    if estimate is None:
        assemble = _native_compose()
        if assemble is not None:
            return _compose_natively(assemble, core, ledger, cellar, retrieval, flesh, turn, budget)
    estimate = estimate or estimate_tokens
    segments = []

    core_text = core.text()
    core_segment = _segment("core", core_text, f"core:{core.root()}", estimate)
    _refuse_over("core", core_segment.tokens, budget.core, "the Core is never cut, it is the registry's bytes")
    segments.append(core_segment)

    # The digest folds its oldest lines to keep within its cap, so the
    # receipts layer can no longer refuse the block; the check stays as the
    # statement of the bound.
    digest, folded = ledger.render(cellar, cap=budget.receipts, estimate=estimate)
    if digest:
        receipts_segment = _segment("receipts", digest, "receipts", estimate)
        _refuse_over("receipts", receipts_segment.tokens, budget.receipts, "resolve or archive receipts first")
        segments.append(receipts_segment)

    used, dropped = 0, 0
    for peel in retrieval:
        seg = _segment("peels", peel.text, peel.provenance, estimate)
        if used + seg.tokens > budget.peels:
            dropped += 1
            continue
        used += seg.tokens
        segments.append(seg)

    turns = flesh.turns() if hasattr(flesh, "turns") else [dict(t) for t in flesh]
    flesh_segments = [
        _segment("flesh", str(t.get("text", "")), f"flesh:{t.get('turn_id', '')}:{t.get('role', '')}", estimate)
        for t in turns
    ]
    _refuse_over(
        "flesh", sum(s.tokens for s in flesh_segments), budget.flesh,
        "a turn leaves the window only with a receipt; evict first, the composer drops nothing",
    )
    segments.extend(flesh_segments)

    turn_segment = _segment("turn", turn, "turn", estimate)
    _refuse_over("turn", turn_segment.tokens, budget.turn, "the current turn is never truncated")
    segments.append(turn_segment)

    total = sum(s.tokens for s in segments)
    _refuse_over("window", total, budget.window - budget.reserve, "the generation reserve is not assembled into")
    return Prompt(
        segments=tuple(segments), tokens=total, core_root=core.root(), dropped_peels=dropped, folded_receipts=folded,
    )
