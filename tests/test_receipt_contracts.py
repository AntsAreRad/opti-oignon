#!/usr/bin/env python3
"""Contracts for eviction receipts and the Cellar of the onion memory.

When a span leaves Flesh, a one-line stub with a recall key stays in the
receipts digest, so the model knows what it no longer sees and can ask for
it back instead of confabulating. Two invariants: no eviction without a
receipt, and no dangling receipt key -- every key resolves to a Cellar span.

  * RE1 -- the Cellar is content-addressed and round-trips a span; an
    unknown key is refused.
  * RE2 -- no eviction without a receipt: every turn that leaves Flesh is
    named by a receipt whose key resolves.
  * RE3 -- no dangling key: a receipt whose key does not resolve is refused
    by name, by the digest and by resolution alike.
  * RE4 -- the instrument reads non-zero: evicting from a Flesh that fits
    yields nothing, from one that does not yields at least one receipt, and
    the remainder fits.
  * RE5 -- append and resolve: a resolved receipt leaves the digest and stays
    in the ledger; the ledger never forgets.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _open():
    loaded, restore = isolate(
        targets={"opti_oignon.memory.receipts": source("memory", "receipts.py")},
        packages=("opti_oignon.memory",),
    )
    return loaded["opti_oignon.memory.receipts"], restore


def _turn(i):
    return {"turn_id": f"t{i:02d}", "role": "user" if i % 2 else "assistant",
            "text": f"Turn {i} records that service {i} was migrated on day {i}."}


def _words(text):
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# RE1 -- the Cellar
# ---------------------------------------------------------------------------
def test_re1_the_cellar_is_content_addressed_and_round_trips():
    mod, restore = _open()
    try:
        cellar = mod.Cellar()
        span = [_turn(1), _turn(2)]
        key = cellar.store(span)
        canonical = json.dumps(span, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert key == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert key == mod.span_key(span)
        assert cellar.get(key) == span
        assert cellar.get(key) is not span, "a copy comes back, the archive is not a handle"
        assert cellar.store(span) == key and len(cellar) == 1
        with pytest.raises(KeyError):
            cellar.get("0" * 64)
        assert cellar.has(key) is True and cellar.has("0" * 64) is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# RE2 -- no eviction without a receipt
# ---------------------------------------------------------------------------
def test_re2_every_turn_that_leaves_flesh_is_named_by_a_resolving_receipt():
    mod, restore = _open()
    try:
        cellar, ledger = mod.Cellar(), mod.ReceiptLedger()
        flesh = mod.Flesh([_turn(i) for i in range(1, 6)])
        receipt = flesh.evict_oldest(cellar, ledger)
        assert isinstance(receipt, mod.Receipt)
        assert receipt.turn_ids == ("t01",)
        assert [t["turn_id"] for t in flesh.turns()] == ["t02", "t03", "t04", "t05"]
        assert ledger.open() == [receipt]
        assert cellar.get(receipt.key) == [_turn(1)]
        assert receipt.stub and receipt.key[:12] in receipt.stub
        evicted = flesh.evict_until_fits(_words(_turn(5)["text"]), _words, cellar, ledger)
        named = {tid for r in ledger.open() for tid in r.turn_ids}
        assert named == {"t01", "t02", "t03", "t04"}, "every evicted turn is named by a receipt"
        assert len(evicted) >= 1
        for r in ledger.open():
            assert cellar.has(r.key), "and every receipt resolves"
        assert [t["turn_id"] for t in flesh.turns()] == ["t05"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# RE3 -- no dangling key
# ---------------------------------------------------------------------------
def test_re3_a_receipt_whose_key_does_not_resolve_is_refused_by_name():
    mod, restore = _open()
    try:
        cellar, ledger = mod.Cellar(), mod.ReceiptLedger()
        flesh = mod.Flesh([_turn(i) for i in range(1, 4)])
        flesh.evict_oldest(cellar, ledger)
        assert ledger.digest(cellar).count("\n") == 0 and ledger.digest(cellar), "control: one line"
        dangling = mod.Receipt(key="f" * 64, stub="ffffffffffff turns t99: nothing behind it", turn_ids=("t99",))
        ledger.append(dangling)
        with pytest.raises(mod.DanglingReceiptError) as caught:
            ledger.digest(cellar)
        assert "f" * 64 in str(caught.value)
        with pytest.raises(mod.DanglingReceiptError):
            ledger.resolve("f" * 64, cellar)
        assert len(ledger.open()) == 2, "the refusal is not a silent drop of the receipt"
    finally:
        restore()


# ---------------------------------------------------------------------------
# RE4 -- the instrument reads non-zero
# ---------------------------------------------------------------------------
def test_re4_eviction_yields_nothing_when_flesh_fits_and_something_when_it_does_not():
    mod, restore = _open()
    try:
        cellar, ledger = mod.Cellar(), mod.ReceiptLedger()
        turns = [_turn(i) for i in range(1, 5)]
        total = sum(_words(t["text"]) for t in turns)
        flesh = mod.Flesh(turns)
        assert flesh.tokens(_words) == total
        assert flesh.evict_until_fits(total, _words, cellar, ledger) == []
        assert ledger.open() == [] and len(flesh.turns()) == 4
        receipts = flesh.evict_until_fits(total - 1, _words, cellar, ledger)
        assert len(receipts) >= 1
        assert flesh.tokens(_words) <= total - 1
        assert len(ledger.open()) == len(receipts)
        with pytest.raises(ValueError):
            mod.Flesh([]).evict_oldest(cellar, ledger)
    finally:
        restore()


# ---------------------------------------------------------------------------
# RE5 -- append and resolve
# ---------------------------------------------------------------------------
def test_re5_a_resolved_receipt_leaves_the_digest_and_stays_in_the_ledger():
    mod, restore = _open()
    try:
        cellar, ledger = mod.Cellar(), mod.ReceiptLedger()
        flesh = mod.Flesh([_turn(i) for i in range(1, 4)])
        first = flesh.evict_oldest(cellar, ledger)
        second = flesh.evict_oldest(cellar, ledger)
        assert ledger.digest(cellar).count("\n") == 1, "control: two lines"
        span = ledger.resolve(first.key, cellar)
        assert span == [_turn(1)]
        assert ledger.open() == [second]
        assert [r.key for r in ledger.all()] == [first.key, second.key]
        assert [r.resolved for r in ledger.all()] == [True, False]
        assert first.key[:12] not in ledger.digest(cellar)
        for name in ("remove", "delete", "clear", "pop"):
            assert not hasattr(ledger, name), f"no {name}: append and resolve only"
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
