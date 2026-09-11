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

from dataclasses import dataclass
from pathlib import Path

checkpoint_before_apply = True

LAYERS = ("core", "receipts", "peels", "flesh", "turn")
_INSTRUCTION_BEARING = frozenset({"core", "turn"})
_CONFIG = Path(__file__).resolve().parent.parent / "config" / "onion.yaml"


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

    def render(self):
        """The window as text. Recalled segments are framed as quoted data with their tag."""
        parts = []
        for seg in self.segments:
            if seg.instruction_bearing:
                parts.append(seg.text)
            else:
                parts.append(f"[data layer={seg.layer} provenance={seg.provenance}]\n{seg.text}\n[/data]")
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


def compose(*, core, ledger, cellar, retrieval, flesh, turn, budget, estimate=None):
    """Assemble the window. Pure: same inputs, same prompt, inputs untouched."""
    estimate = estimate or estimate_tokens
    errors = budget.validate()
    if errors:
        raise BudgetError("; ".join(errors))
    segments = []

    core_text = core.text()
    core_segment = _segment("core", core_text, f"core:{core.root()}", estimate)
    _refuse_over("core", core_segment.tokens, budget.core, "the Core is never cut, it is the registry's bytes")
    segments.append(core_segment)

    digest = ledger.digest(cellar)
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
    return Prompt(segments=tuple(segments), tokens=total, core_root=core.root(), dropped_peels=dropped)
