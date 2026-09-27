#!/usr/bin/env python3
"""A passage counts only where the host finds it, in a chunk whose hash it recomputed.

The citation is the external verifier: the host locates the passage itself,
character for character after a fixed fold, inside a chunk whose SHA-256 it
recomputes. These contracts pin the passage check and the fold:

  * LQ1 -- a passage is located verbatim in a chunk whose hash is recomputed,
    at the host's positions in code points: a quote one character off is not
    found; a chunk edited under its old hash is refused and nothing is
    located in it; every occurrence is found, sorted and counted, and past
    the cap the first ones come back with a flag; a quote that crosses two
    chunks is located in neither; an empty quote and a quote under the floor
    are refused by name; and the check searches each chunk on its own, so a
    sentence split across two chunks supports nothing.
  * LQ2 -- fold v1 is exactly its table: every class maps as written; NFC
    only, never NFKC (ten with a superscript nine is not 109); a chunk that
    is not NFC is refused, never repaired; case is kept; an expansion is
    atomic, so a quote starting inside a ligature is not an occurrence and a
    match ends before a removed invisible code point; the fold's version and
    table are in the rules digest.
  * LQ3 -- no path yields "supported" without a located span from an
    admitted source valid at the time: over the canary and seeded
    combinations of it (the restated sentence removed, every source refused,
    every source made historical) nothing is supported, and every supported
    record's support spans come from sources it records admitted and valid;
    a verbatim sentence in a model-authored, unknown-author, snippet or
    model-quoted source is not enough evidence with the refusal recorded;
    one in a source valid only later, or expired, is not supported; and when
    a sentence restates the claim, the reason is the check it failed, never
    that the claim was not restated.

The prefix LQ ("a located quote") names these three contracts; PC is taken in
this directory by another suite.

Loaded through the shared isolation window (``tests/_factcheck.py``), with the
native core unreachable. Nothing reaches a model, the network or the
maintainer's data.
"""

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _factcheck as F  # noqa: E402

BUDGET_S = {
    "test_lq1_a_passage_is_located_verbatim_at_the_hosts_positions": 2.0,
    "test_lq2_fold_v1_is_exactly_its_table": 2.0,
    "test_lq3_no_path_yields_supported_without_a_located_admitted_valid_span": 2.0,
}

LIMITS = {"min_quote_chars": 12, "max_occurrences": 16}


@pytest.fixture
def fc():
    ns, restore = F.load()
    try:
        yield ns
    finally:
        restore()


def _locate(fc, text, quote, *, sha256=None, **limits):
    bounds = dict(LIMITS, **limits)
    return fc.passage.locate(text, sha256 or F.sha(text), quote, **bounds)


def test_lq1_a_passage_is_located_verbatim_at_the_hosts_positions(fc):
    e_acute, a_grave = chr(0xE9), chr(0xE0)
    text = (f"{e_acute}t{e_acute} {a_grave} Montr{e_acute}al : le mus{e_acute}e ouvre {a_grave} 9 h. "
            "The harbour opens at nine every morning.")
    quote = "The harbour opens at nine every morning"

    # c1 -- located once, at code-point offsets the host computed.
    found = _locate(fc, text, quote)
    assert found.refusal is None and found.count == 1 and not found.truncated, found
    (start, end), = found.occurrences
    assert start == text.index("The harbour") and end == start + len(quote)
    assert fc.passage.fold_quote(text[start:end]) == fc.passage.fold_quote(quote)
    assert len(text[:start].encode("utf-8")) != start, "the fixture must tell bytes from code points"

    # c2 -- one character off.
    off = _locate(fc, text, "The harbour opens at nine every mornimg")
    assert off.refusal == "quote_not_found" and off.occurrences == (), off

    # c3 -- a chunk edited by one character, carrying its old hash.
    edited = text.replace("nine", "nina")
    changed = fc.passage.locate(edited, F.sha(text), "The harbour opens at nina every morning", **LIMITS)
    assert changed.refusal == "chunk_changed" and changed.occurrences == (), changed

    # c4 -- every occurrence, sorted, counted; past the cap, the first ones and a flag.
    triple = " ".join(["The tide turns at noon today."] * 3)
    every = _locate(fc, triple, "The tide turns at noon today")
    assert every.count == 3 and not every.truncated
    assert list(every.occurrences) == sorted(every.occurrences) and len(every.occurrences) == 3
    assert [triple[s:e] for s, e in every.occurrences] == ["The tide turns at noon today"] * 3
    capped = _locate(fc, triple, "The tide turns at noon today", max_occurrences=2)
    assert capped.truncated and capped.occurrences == every.occurrences[:2], capped

    # c5 -- a quote across two adjacent chunks, each with its correct hash, is
    # located in neither; a chunk holding both halves locates it; and the check
    # searches each chunk on its own, never their join.
    first, second = "Notes. The Brest harbour opens at", " nine every morning. More notes."
    harbour = "The Brest harbour opens at nine every morning"
    split = [fc.passage.locate(part, F.sha(part), harbour, **LIMITS) for part in (first, second)]
    assert [located.occurrences for located in split] == [(), ()], split
    whole = fc.passage.locate(first + second, F.sha(first + second), harbour, **LIMITS)
    assert len(whole.occurrences) == 1, whole
    claim = harbour + "."
    two = F.run(fc, claim, [F.item(fc, "library:harbour", [first, second])])
    assert two.value == "not_enough_evidence" and two.record["checks"] == [], two.reasons
    one = F.run(fc, claim, [F.item(fc, "library:harbour", first + second)])
    assert one.value == "supported", one.reasons

    # c6 -- an empty quote and a quote under the floor are refused by name.
    assert _locate(fc, text, "   ").refusal == "quote_empty"
    assert _locate(fc, text, "The harbour").refusal == "quote_too_short"


def test_lq2_fold_v1_is_exactly_its_table(fc):
    fold = fc.passage.fold_text

    # c1 -- every class of the table maps as written.
    classes = {
        "whitespace": ([0x09, 0x0A, 0x0D, 0xA0, 0x2003, 0x202F, 0x3000], " "),
        "invisible": ([0xAD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF], ""),
        "apostrophe": ([0x2018, 0x2019, 0x201A, 0x201B, 0x02BC, 0x2032, 0x00B4, 0x0060, 0xFF07], "'"),
        "double_quote": ([0x201C, 0x201D, 0x201E, 0x201F, 0x00AB, 0x00BB, 0x2033, 0xFF02], '"'),
        "dash": ([0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0xFE63, 0xFF0D], "-"),
        "ellipsis": ([0x2026], "..."),
    }
    for name, (points, becomes) in classes.items():
        for point in points:
            assert fold(f"a{chr(point)}b") == f"a{becomes}b", (name, hex(point))
    assert fold("a \t\n  b") == "a b"
    ligatures = {0xFB00: "ff", 0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl",
                 0xFB05: "st", 0xFB06: "st"}
    for point, becomes in ligatures.items():
        assert fold(f"a{chr(point)}b") == f"a{becomes}b", hex(point)
    # Witness: each class met by a located quote of twelve characters or more.
    witnesses = {
        "whitespace": (f"The ferry{chr(0xA0)}leaves at noon sharp.", "The ferry leaves at noon"),
        "invisible": (f"The ferry{chr(0x200B)} leaves at noon sharp.", "The ferry leaves at noon"),
        "apostrophe": (f"The captain{chr(0x2019)}s log was kept.", "The captain's log was kept"),
        "double_quote": (f"The ship {chr(0xAB)}Belem{chr(0xBB)} is moored.", 'The ship "Belem" is moored'),
        "dash": (f"The Paris{chr(0x2013)}Rouen line opened.", "The Paris-Rouen line opened"),
        "ellipsis": (f"The crew waited{chr(0x2026)} and waited.", "The crew waited... and waited"),
        "ligature": (f"The sta{chr(0xFB00)} o{chr(0xFB03)}ce is closed.", "The staff office is closed"),
    }
    for name, (text, quote) in witnesses.items():
        located = _locate(fc, text, quote)
        assert located.refusal is None and located.count == 1, (name, located)

    # c2 -- NFC only, never NFKC.
    sup9 = chr(0x2079)
    text = f"The sample had a mass of 10{sup9} kg."
    assert _locate(fc, text, "a mass of 109 kg.").refusal == "quote_not_found"
    assert _locate(fc, text, f"a mass of 10{sup9} kg.").count == 1

    # c3 -- a chunk carrying a decomposed letter is refused; its NFC form passes.
    decomposed = f"The cafe{chr(0x301)} opens at noon on Sundays."
    assert _locate(fc, decomposed, "opens at noon on Sundays").refusal == "chunk_not_nfc"
    composed = f"The caf{chr(0xE9)} opens at noon on Sundays."
    assert _locate(fc, composed, "opens at noon on Sundays").count == 1

    # c4 -- case is kept.
    assert _locate(fc, "In 2019 the Apple harvest failed.", "the apple harvest failed").refusal == "quote_not_found"
    assert _locate(fc, "In 2019 the apple harvest failed.", "the apple harvest failed").count == 1

    # c5 -- expansions are atomic; a match ends before a removed invisible code point.
    ffi = chr(0xFB03)
    office = f"Our o{ffi}ce is closed on Sundays."
    assert _locate(fc, office, "fice is closed on Sundays").refusal == "quote_not_found"
    assert _locate(fc, office, "Our office is closed on Sunday").refusal is None
    assert _locate(fc, f"Look at our o{ffi}ce today", "Look at our of").refusal == "quote_not_found"
    assert _locate(fc, "Look at our office today", "Look at our of").count == 1
    covered = _locate(fc, office, "office is closed on Sundays")
    (start, end), = covered.occurrences
    assert start == office.index("o" + ffi) and office[start:end].startswith("o" + ffi), (start, end)
    shy = chr(0xAD)
    noon = f"The ferry leaves at noon{shy} sharp."
    (start, end), = _locate(fc, noon, "The ferry leaves at noon").occurrences
    assert end == noon.index(shy) and noon[start:end] == "The ferry leaves at noon", (start, end)

    # c6 -- the fold's version and table are in the rules digest.
    before = fc.decide.rules_digest()
    assert fc.decide.rules_digest() == before
    document = fc.decide.rules_document()
    assert document["fold"]["version"] == fc.passage.FOLD_VERSION == 1
    table = list(fc.passage.FOLD_TABLE)
    changed_row = list(table[2])
    changed_row[-1] = '"'
    table[2] = tuple(changed_row)
    original = fc.passage.FOLD_TABLE
    fc.passage.FOLD_TABLE = tuple(table)
    try:
        assert fc.decide.rules_digest() != before
    finally:
        fc.passage.FOLD_TABLE = original
    assert fc.decide.rules_digest() == before


def _without_restatement(fc, row, cfg):
    """The row's items with every sentence equal to the claim removed."""
    claim = row.claim
    out = []
    for one in row.items:
        chunks = []
        for c in one.chunks:
            text = c.text
            for sentence in fc.passage.split_text(text, markdown=one.kind == "note"):
                if fc.decide.restates(claim, sentence, owner=True) or fc.decide.restates(claim, sentence, owner=False):
                    text = text.replace(sentence, "")
            chunks.append(dataclasses.replace(c, text=text, sha256=F.sha(text)))
        out.append(dataclasses.replace(one, chunks=tuple(chunks)))
    return out


def test_lq3_no_path_yields_supported_without_a_located_admitted_valid_span(fc):
    cfg = F.config(fc)

    # c2 -- a verbatim sentence in a refused source, or in a model-quoted range.
    sentence = "The Rhine flows into the North Sea near Rotterdam."
    refused = {
        "author_model": F.item(fc, "library:m", sentence, author="model"),
        "author_unknown": F.item(fc, "library:u", sentence, author="unknown"),
        "snippet": F.item(fc, "web:s", sentence, kind="snippet"),
    }
    for code, one in refused.items():
        verdict = F.run(fc, sentence, [one], cfg=cfg)
        assert verdict.value == "not_enough_evidence", (code, verdict.value)
        assert verdict.record["sources_searched"][0]["admission"] == code
    text = f"From the chat. {sentence}"
    quoted = F.item(fc, "note:q", text, kind="note", author="user",
                    chunk_fields={"model_quoted": ((0, len(text)),)})
    verdict = F.run(fc, sentence, [quoted], cfg=cfg)
    assert verdict.value == "not_enough_evidence", verdict.reasons
    entry = verdict.record["sources_searched"][0]
    assert [r["refusal"] for r in entry["span_refusals"]] == ["author_model_quoted"], entry
    # The passage restated the claim: the reason is its refusal, never "not restated".
    assert verdict.leading == "no_admissible_source" and "no_judge" not in verdict.reasons, verdict.reasons
    assert "Not restated" not in verdict.text and "note:q, characters" in verdict.text, verdict.text

    # c3 -- valid only after t, and expired before t with no successor.
    later = F.item(fc, "library:later", sentence, valid_from="2026-10-01")
    verdict = F.run(fc, sentence, [later], cfg=cfg)
    assert (verdict.value, verdict.leading) == ("not_enough_evidence", "no_valid_source"), verdict.reasons
    gone = F.item(fc, "library:gone", sentence, valid_from="2025-01-01", valid_until="2026-02-01")
    verdict = F.run(fc, sentence, [gone], cfg=cfg)
    assert verdict.value == "not_enough_evidence" and "expired" in verdict.reasons, verdict.reasons
    assert verdict.details["expired"]["valid_until"] == "2026-02-01"
    assert "2026-02-01" in verdict.text, verdict.text
    # Beside a source valid at t that says something else (another statement
    # altogether, so no later reader of values shares a frame with it), the
    # restating source still gives its own reason, never "not restated".
    other = F.item(fc, "library:other", "The Danube is the second longest river of Europe.")
    verdict = F.run(fc, sentence, [later, other], cfg=cfg)
    assert verdict.value == "not_enough_evidence" and "no_valid_source" in verdict.reasons, verdict.reasons
    assert "no_judge" not in verdict.reasons and "Not restated" not in verdict.text, verdict.text
    assert "library:later (from 2026-10-01)" in verdict.text, verdict.text
    verdict = F.run(fc, sentence, [gone, other], cfg=cfg)
    assert "expired" in verdict.reasons, verdict.reasons
    assert "no_judge" not in verdict.reasons, verdict.reasons

    # c1 -- over the canary and seeded combinations of it.
    supported_seen = 0
    variants_seen = 0
    for row in fc.canary.rows():
        base = fc.decide.check(row.claim, row.items, as_of=row.as_of, read_on=row.read_on, config=cfg)
        if base.value == "supported":
            supported_seen += 1
            record = base.record
            spans = [s for s in record["spans"] if s["role"] == "support"]
            assert spans, (row.name, "a supported record lists no span")
            searched = {e["source_id"]: e for e in record["sources_searched"]}
            for span in spans:
                entry = searched[span["source_id"]]
                assert entry["admission"] == "admitted" and entry["validity"]["valid_at"], (row.name, entry)
        seeds = {
            "restatement removed": _without_restatement(fc, row, cfg),
            "every source refused": [dataclasses.replace(i, author="model") for i in row.items],
            "every source historical": [
                dataclasses.replace(i, valid_from="2020-01-01", valid_until="2020-06-01", superseded_by=None)
                for i in row.items],
        }
        for label, items in seeds.items():
            verdict = fc.decide.check(row.claim, items, as_of=row.as_of, read_on=row.read_on, config=cfg)
            assert verdict.value != "supported", (row.name, label, verdict.reasons)
            variants_seen += 1
    assert supported_seen >= 8 and variants_seen >= 120, (supported_seen, variants_seen)
