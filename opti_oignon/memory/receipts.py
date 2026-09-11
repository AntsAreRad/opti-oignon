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

The Cellar is the sole legal source of every later compression; here it is
an in-memory, content-addressed archive with the same surface the encrypted
one will have. The ledger appends and resolves; it never forgets. Nothing on
the chat path imports this module yet, and a contract on the tree says so.
"""

import hashlib
import json
from dataclasses import dataclass, replace

checkpoint_before_apply = True

_STUB_HEAD = 48


def _canonical(span):
    return json.dumps(list(span), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def span_key(span):
    """The recall key of a span: SHA-256 of its canonical JSON."""
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


def make_receipt(span, key):
    ids = tuple(str(t.get("turn_id", "")) for t in span)
    head = " ".join(str(span[0].get("text", "")).split())[:_STUB_HEAD] if span else ""
    span_ids = ids[0] if len(ids) == 1 else f"{ids[0]}..{ids[-1]}"
    return Receipt(key=key, stub=f"{key[:12]} turns {span_ids}: {head}", turn_ids=ids)


class ReceiptLedger:
    """Append and resolve. A receipt is never removed."""

    def __init__(self):
        self._receipts = []

    def append(self, receipt):
        self._receipts.append(receipt)
        return receipt

    def all(self):
        return list(self._receipts)

    def open(self):
        return [r for r in self._receipts if not r.resolved]

    def _check(self, cellar):
        for receipt in self._receipts:
            if not cellar.has(receipt.key):
                raise DanglingReceiptError(
                    f"receipt {receipt.key} resolves to no Cellar span: refused before it reaches the model"
                )

    def resolve(self, key, cellar):
        """Hand the span back and mark its receipt resolved."""
        self._check(cellar)
        for i, receipt in enumerate(self._receipts):
            if receipt.key == key:
                self._receipts[i] = replace(receipt, resolved=True)
                return cellar.get(key)
        raise DanglingReceiptError(f"receipt {key} is not in the ledger")

    def digest(self, cellar):
        """One line per open receipt, only once every key is known to resolve."""
        self._check(cellar)
        return "\n".join(r.stub for r in self.open())


class Flesh:
    """The last turns, verbatim. A turn leaves only through an eviction."""

    def __init__(self, turns=()):
        self._turns = [dict(t) for t in turns]

    def append(self, turn):
        self._turns.append(dict(turn))

    def turns(self):
        return [dict(t) for t in self._turns]

    def tokens(self, estimate):
        return sum(estimate(str(t.get("text", ""))) for t in self._turns)

    def evict_oldest(self, cellar, ledger):
        """Move the oldest turn to the Cellar and leave its receipt. One step."""
        if not self._turns:
            raise ValueError("nothing to evict: Flesh is empty")
        span = [self._turns.pop(0)]
        key = cellar.store(span)
        return ledger.append(make_receipt(span, key))

    def evict_until_fits(self, cap, estimate, cellar, ledger):
        """Evict from the oldest until the remainder fits ``cap``. Receipts, in order."""
        receipts = []
        while self._turns and self.tokens(estimate) > cap:
            receipts.append(self.evict_oldest(cellar, ledger))
        return receipts
