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
  * RE6 -- no receipt line carries a word of its span: its key, its turns,
    its kind and its origins, nothing the span said.
  * RE7 -- a receipt names its kind; an eviction that places no peel is
    bare, and a kind outside the list is refused before anything leaves.
  * RE8 -- over its cap the digest folds the oldest open receipts into one
    line and keeps the newest whole, in order; the ledger never changes.
  * RE9 -- the instrument reads non-zero: the digest counts the receipts it
    folded, zero when it fits.
  * RE10 -- a cap of zero shows no receipt and counts them all folded.

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


# ---------------------------------------------------------------------------
# RE6-RE10 -- what a receipt line says, and how the digest keeps to its cap
# ---------------------------------------------------------------------------
def _bare_words(text):
    return set(text.replace("(", " ").replace(")", " ").replace(";", " ").replace(",", " ").split())


def test_re6_no_receipt_line_carries_a_word_of_its_span():
    mod, restore = _open()
    try:
        cellar, ledger = mod.Cellar(), mod.ReceiptLedger()
        flesh = mod.Flesh([_turn(i) for i in range(1, 5)])
        receipt = flesh.evict_span(2, cellar, ledger)
        line = ledger.digest(cellar)
        assert receipt.key[:12] in line and "t01..t02" in line, "control: the line names its key and its turns"
        span_words = {word for turn in cellar.get(receipt.key) for word in turn["text"].split()}
        assert len(span_words) >= 8, "control: the span has words to leak"
        for text in (line, receipt.stub):
            assert not span_words & _bare_words(text), f"no word of the span in {text!r}"
    finally:
        restore()


def test_re7_a_receipt_names_its_kind_and_an_eviction_that_places_no_peel_is_bare():
    mod, restore = _open()
    try:
        cellar, ledger = mod.Cellar(), mod.ReceiptLedger()
        flesh = mod.Flesh([_turn(i) for i in range(1, 7)])
        plain = flesh.evict_span(2, cellar, ledger)
        held = flesh.evict_span(2, cellar, ledger, kind="held")
        assert plain.kind == "bare" and held.kind == "held"
        lines = ledger.digest(cellar).splitlines()
        assert "(bare;" in lines[0] and "(held;" in lines[1]
        with pytest.raises(ValueError, match="receipt kind"):
            flesh.evict_span(2, cellar, ledger, kind="kept")
        assert len(flesh.turns()) == 2 and len(ledger.all()) == 2, "a refused kind evicts nothing"
    finally:
        restore()


def test_re8_over_its_cap_the_digest_folds_the_oldest_and_keeps_the_newest_whole():
    mod, restore = _open()
    try:
        cellar, ledger = mod.Cellar(), mod.ReceiptLedger()
        flesh = mod.Flesh([_turn(i) for i in range(1, 41)])
        made = [flesh.evict_span(2, cellar, ledger) for _ in range(20)]
        whole = ledger.digest(cellar)
        cap = _words(whole) // 3
        folded = ledger.digest(cellar, cap=cap, estimate=_words)
        assert _words(folded) <= cap
        lines = folded.splitlines()
        assert 1 < len(lines) < 20, "some receipts whole, the rest folded"
        assert lines[1:] == whole.splitlines()[21 - len(lines):], "the newest stay whole, in order"
        assert "t01.." in lines[0] and "folded" in lines[0], "one line for the oldest, from their first turn"
        assert ledger.all() == made, "the ledger never changes"
    finally:
        restore()


def test_re9_the_digest_counts_what_it_folded_and_zero_when_it_fits():
    mod, restore = _open()
    try:
        cellar, ledger = mod.Cellar(), mod.ReceiptLedger()
        flesh = mod.Flesh([_turn(i) for i in range(1, 41)])
        for _ in range(20):
            flesh.evict_span(2, cellar, ledger)
        whole = ledger.digest(cellar)
        assert ledger.render(cellar) == (whole, 0)
        assert ledger.render(cellar, cap=_words(whole), estimate=_words) == (whole, 0)
        text, folded = ledger.render(cellar, cap=_words(whole) // 3, estimate=_words)
        assert folded >= 1
        assert folded + len(text.splitlines()) - 1 == 20, "folded and whole account for every open receipt"
    finally:
        restore()


def test_re10_a_cap_of_zero_shows_no_receipt_and_folds_them_all():
    mod, restore = _open()
    try:
        cellar, ledger = mod.Cellar(), mod.ReceiptLedger()
        flesh = mod.Flesh([_turn(i) for i in range(1, 7)])
        for _ in range(3):
            flesh.evict_span(2, cellar, ledger)
        assert ledger.render(cellar, cap=0, estimate=_words) == ("", 3)
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
