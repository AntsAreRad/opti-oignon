#!/usr/bin/env python3
"""Eviction receipts, the Flesh they are cut from, and the Cellar they point into.

A turn that leaves the window must leave a trace the model can see: a
one-line stub with a recall key. With it the model knows what it no longer
sees and can ask for the span back instead of filling the gap from
imagination -- fail-secure applied to memory. Two invariants make the
receipts worth reading. No eviction without a receipt: the only way a turn
leaves Flesh here is through a call that stores the span in the Cellar and
appends the receipt in the same step. No dangling key: the digest re-checks
every key against the Cellar before it renders a single line, and refuses by
name rather than show the model a key that leads nowhere.

The Cellar is the sole legal source of every later compression: an
in-memory, content-addressed archive whose rows the onion store writes to
its encrypted table and re-hashes on the way back. The ledger appends,
resolves and supersedes; it never forgets. The executor reaches this module
through the librarian and nothing else does; a contract on the tree says so.
"""

import hashlib
import json
from dataclasses import dataclass, field, replace

checkpoint_before_apply = True

# What stands for an evicted span in the window, by name: a peel the gate
# accepted, anchors kept verbatim where no peel could pass, or nothing. A
# superseded span is one the conversation no longer holds as it was: nothing
# stands for it in the window, and the Cellar keeps it as it was.
RECEIPT_KINDS = ("accepted", "held", "bare", "superseded")


def _canonical(span):
    return json.dumps(list(span), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _native():
    """The native core, or None: asked at the call, never at import."""
    try:
        from opti_oignon.native import load
    except Exception:  # noqa: BLE001 - absence is the reference path
        return None
    return load()


def span_key(span):
    """The recall key of a span: SHA-256 of its canonical JSON."""
    core = _native()
    if core is not None:
        try:
            return core.span_key(list(span))
        except TypeError:
            pass  # a value shape the core refuses: the reference formats it
    return hashlib.sha256(_canonical(span).encode("utf-8")).hexdigest()


class DanglingReceiptError(KeyError):
    """A receipt names a key the Cellar does not hold."""

    def __str__(self):
        return self.args[0] if self.args else ""


class Cellar:
    """Content-addressed archive of spans. Append-only; a read is a copy."""

    def __init__(self):
        self._spans = {}

    def store(self, span):
        key = span_key(span)
        if key not in self._spans:
            self._spans[key] = json.loads(_canonical(span))
        return key

    def has(self, key):
        return key in self._spans

    def keys(self):
        """Every key held, in insertion order."""
        return list(self._spans)

    def get(self, key):
        return json.loads(_canonical(self._spans[key]))

    def __len__(self):
        return len(self._spans)


@dataclass(frozen=True)
class Receipt:
    """One line the model sees in place of a span it no longer sees."""

    key: str
    stub: str
    turn_ids: tuple
    resolved: bool = False
    # What stands for the span in the window, one of ``RECEIPT_KINDS``. A
    # receipt read back from a store that predates kinds was made by the
    # gated eviction, then the only writer: its peel stands for it.
    kind: str = "accepted"
    # Where in its span the words kept verbatim for a held span lie, each
    # ``(turn_id, start, stop)`` into that turn's text in the Cellar: the
    # words are read from the Cellar, never copied here.
    anchors: tuple = ()
    # The origins its span's words declare, sorted: a cache of what the
    # Cellar holds, so no part of what the receipt is. Empty for a receipt
    # read back from a store, and read from the Cellar when its line is drawn.
    origins: tuple = field(default=(), compare=False)


def span_origins(span):
    """Every origin a span's turns declare, sorted: the labels its receipt and peels answer for.

    Gathered as declared, never re-judged here: the store refuses a
    declaration outside the grammar, the mirror makes one legacy, and the
    probe reader is the place a label is relied on. A turn that declares
    nothing is legacy.
    """
    found = set()
    for turn in span:
        origin = turn.get("origin")
        found.add(origin if isinstance(origin, str) and origin else "legacy")
        segments = turn.get("segments")
        if isinstance(segments, (list, tuple)):
            for segment in segments:
                if isinstance(segment, (list, tuple)) and len(segment) == 3 and isinstance(segment[2], str):
                    found.add(segment[2])
    return tuple(sorted(found))


def _require_kind(kind):
    if kind not in RECEIPT_KINDS:
        raise ValueError(f"{kind!r} is not a receipt kind: one of {', '.join(RECEIPT_KINDS)}")


def _turns(ids):
    if not ids:
        return "(none)"
    return ids[0] if len(ids) == 1 else f"{ids[0]}..{ids[-1]}"


def _line(key, ids, kind, origins):
    """A receipt's line: its key, its turns, its kind and its origins; never a word its span said."""
    return f"{key[:12]} turns {_turns(ids)} ({kind}; {', '.join(origins) or 'legacy'})"


def _folded(receipts):
    """One line for the oldest open receipts: how many spans, and the turns they ran over."""
    first, last = receipts[0].turn_ids, receipts[-1].turn_ids
    start = first[0] if first else "(none)"
    end = last[-1] if last else "(none)"
    return f"{len(receipts)} earlier spans folded: turns {start}..{end}"


def make_receipt(span, key, kind="bare", anchors=()):
    """The receipt of ``span`` stored under ``key``: no word of the span, only where and what it was."""
    _require_kind(kind)
    ids = tuple(str(t.get("turn_id", "")) for t in span)
    origins = span_origins(span)
    return Receipt(key=key, stub=_line(key, ids, kind, origins), turn_ids=ids, kind=kind,
                   anchors=tuple(tuple(anchor) for anchor in anchors), origins=origins)


class ReceiptLedger:
    """Append, resolve and supersede. A receipt is never removed."""

    def __init__(self):
        self._receipts = []

    def append(self, receipt):
        self._receipts.append(receipt)
        return receipt

    def all(self):
        return list(self._receipts)

    def open(self):
        """The receipts the window shows: neither resolved by the user nor superseded by the conversation."""
        return [r for r in self._receipts if not r.resolved and r.kind != "superseded"]

    def supersede(self, key):
        """Mark the receipt of ``key`` superseded: the conversation no longer holds its span as it was.

        The receipt stays in the ledger and its span in the Cellar, as they
        were; only what the window shows changes. An unknown key is refused
        by name.
        """
        for i, receipt in enumerate(self._receipts):
            if receipt.key == key:
                self._receipts[i] = replace(receipt, kind="superseded")
                return self._receipts[i]
        raise DanglingReceiptError(f"receipt {key} is not in the ledger")

    def _check(self, cellar):
        for receipt in self._receipts:
            if not cellar.has(receipt.key):
                raise DanglingReceiptError(
                    f"receipt {receipt.key} resolves to no Cellar span: refused before it reaches the model"
                )

    def read(self, key, cellar):
        """Hand the span back; no receipt changes."""
        self._check(cellar)
        if not any(receipt.key == key for receipt in self._receipts):
            raise DanglingReceiptError(f"receipt {key} is not in the ledger")
        return cellar.get(key)

    def resolve(self, key, cellar):
        """Hand the span back and mark its receipt resolved."""
        self._check(cellar)
        for i, receipt in enumerate(self._receipts):
            if receipt.key == key:
                self._receipts[i] = replace(receipt, resolved=True)
                return cellar.get(key)
        raise DanglingReceiptError(f"receipt {key} is not in the ledger")

    def digest(self, cellar, cap=None, estimate=None):
        """One line per open receipt, only once every key is known to resolve; within ``cap`` when given."""
        return self.render(cellar, cap, estimate)[0]

    def render(self, cellar, cap=None, estimate=None):
        """The digest and how many open receipts it folded to keep within ``cap`` tokens.

        Each line is drawn from the receipt's fields, never from a stored
        stub, so no line carries a word of its span. Without a cap every
        open receipt has its line. With one, counted by ``estimate`` -- the
        composer's, so the cap is counted as the window counts it -- the
        oldest open receipts fold into a single line naming how many spans
        they were and the turns they ran over, and the newest keep a line
        each, as many as fit beside it. A cap too small for the folded line
        shows nothing and counts every open receipt folded. The ledger
        itself never changes.
        """
        self._check(cellar)
        opened = self.open()
        lines = [self._line_of(r, cellar) for r in opened]
        whole = "\n".join(lines)
        if cap is None:
            return whole, 0
        if estimate is None:
            raise ValueError("a digest held to a cap needs the estimator its cap is counted in")
        if estimate(whole) <= cap:
            return whole, 0

        def held(kept):
            text = "\n".join([_folded(opened[: len(opened) - kept])] + lines[len(lines) - kept:])
            return text if estimate(text) <= cap else None

        # The most newest receipts kept whole beside one folded line, by
        # bisection: one line more never costs fewer tokens.
        best, low, high = None, 0, len(lines) - 1
        while low <= high:
            middle = (low + high) // 2
            text = held(middle)
            if text is None:
                high = middle - 1
            else:
                best, low = (middle, text), middle + 1
        if best is None:
            return "", len(opened)
        kept, text = best
        return text, len(opened) - kept

    def _line_of(self, receipt, cellar):
        origins = receipt.origins or span_origins(cellar.get(receipt.key))
        return _line(receipt.key, receipt.turn_ids, receipt.kind, origins)

    def origins(self, key, cellar):
        """The origins the receipt's span declares, read from the Cellar now; no receipt changes."""
        return span_origins(self.read(key, cellar))


class Flesh:
    """The last turns, verbatim. A turn leaves only through an eviction, or taken back by the mirror."""

    def __init__(self, turns=()):
        self._turns = [dict(t) for t in turns]

    def append(self, turn):
        self._turns.append(dict(turn))

    def take_back(self, count):
        """Take back the newest ``count`` turns, which the conversation no longer holds as they were; them, in order.

        Only the mirror calls this. No turn here is under a receipt, so
        nothing the window was told of leaves without one.
        """
        count = int(count)
        if count < 0 or count > len(self._turns):
            raise ValueError(f"{count} turns cannot be taken back from a Flesh of {len(self._turns)}")
        taken = self._turns[len(self._turns) - count:]
        del self._turns[len(self._turns) - count:]
        return taken

    def turns(self):
        return [dict(t) for t in self._turns]

    def tokens(self, estimate):
        return sum(estimate(str(t.get("text", ""))) for t in self._turns)

    def evict_oldest(self, cellar, ledger, kind="bare"):
        """Move the oldest turn to the Cellar and leave its receipt. One step."""
        _require_kind(kind)
        if not self._turns:
            raise ValueError("nothing to evict: Flesh is empty")
        span = [self._turns.pop(0)]
        key = cellar.store(span)
        return ledger.append(make_receipt(span, key, kind))

    def evict_span(self, count, cellar, ledger, kind="bare", anchors=()):
        """Move the oldest ``count`` turns to the Cellar as one span, under one receipt of ``kind``.

        The kind names what stands for the span in the window; this method
        places nothing itself, so a caller that placed no peel leaves the
        default, bare. A held span names its ``anchors``, places in its own
        turns. A kind outside ``RECEIPT_KINDS`` is refused before anything
        leaves.
        """
        _require_kind(kind)
        count = int(count)
        if count < 1 or not self._turns:
            raise ValueError("a span of at least one turn is evicted, from a Flesh that has one")
        span, self._turns = self._turns[:count], self._turns[count:]
        key = cellar.store(span)
        return ledger.append(make_receipt(span, key, kind, anchors))

    def evict_until_fits(self, cap, estimate, cellar, ledger, kind="bare"):
        """Evict from the oldest until the remainder fits ``cap``. Receipts, in order."""
        receipts = []
        while self._turns and self.tokens(estimate) > cap:
            receipts.append(self.evict_oldest(cellar, ledger, kind))
        return receipts
