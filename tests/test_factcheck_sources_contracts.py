#!/usr/bin/env python3
"""What counts as evidence, and when it counted.

Evidence is what the owner or a named third party wrote, handed in with its
author, its consent and its dates; a model's words are never evidence, and a
source is read at the time the claim is about:

  * AD1 -- a model-authored, unknown-author, model-quoted, snippet or
    unconsented source is refused by name, and a retracted one is shown as a
    flag and never used; a corrected source is admitted and shown, and a
    chunk ingested before its correction is not used for support; a preprint
    is shown; a claim about the world supported only by a source that cites
    a retracted work is not supported; a chunk from a table, OCR or
    multi-column extraction supports a sentence without a digit, "in the
    extracted text of" its source, and never one with a digit; the tiers of
    the support travel into the verdict, a claim about the world supported
    by the owner's notes alone says so, and the owner's own claim ignores
    library and web sources and says, when nothing else was handed in, that
    no decision or note of his was.
  * TV1 -- validity at a date, on the drift ledger as it really stores
    supersession (written by the real store: both validity fields empty, the
    old fact linked to its successor): a past decision the owner replaced is
    conflicting, "superseded", with both dates labelled "recorded on" and
    both ids, and a decision replaced twice is dated by the one that holds,
    the one in between named; a successor with no date, or one whose explicit
    end comes after its replacement, never lets the replaced decision stand,
    and a successor dated before what it replaced makes both dates unknown,
    never "later"; a decision with its own date, or in the English present,
    is "no longer held", not conflicting; before the replacement it is
    supported, and the new decision is supported after it; a present state
    that was replaced is "no longer held", with the date and the replacement
    quoted, never conflicting and never supported; the second person in
    assistant text and the French pair give the same verdict, while the
    owner's own text in the second person, the assistant's own first person,
    a swapped possessive and a swapped object pronoun never do; a note
    carrying a successor keeps its sentences valid and is flagged coarse; a
    successor not handed in refuses the old fact; a missing start falls back
    to the recording date, and with none at all the source is valid and its
    validity shown unknown.
  * TV2 -- each cited source's own date is shown beside the date read, or
    "undated", with the date the claim is checked for, and a supporting
    decision replaced since says so; a claim about the present found only in
    undated sources is not enough evidence, and supported in a dated one, but
    never by a source dated after the date it is checked for; a check for a
    date after the day it runs is refused; a timeless claim is supported by
    an undated source, which says it is undated.

Loaded through the shared isolation window (``tests/_factcheck.py``), with the
native core unreachable; the drift ledger is the real store, on a file under
the contract's temporary directory. Nothing reaches a model, the network or
the maintainer's data.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _factcheck as F  # noqa: E402
from _isolation import source  # noqa: E402

BUDGET_S = {
    "test_ad1_only_the_owner_or_a_named_third_party_is_evidence": 2.0,
    "test_tv1_validity_on_the_ledger_as_it_stores_supersession": 2.0,
    "test_tv2_a_sources_own_date_is_shown_and_undated_does_not_vouch_for_now": 2.0,
}

RHINE = "The Rhine flows into the North Sea near Rotterdam."


@pytest.fixture
def fc():
    ns, restore = F.load()
    try:
        yield ns
    finally:
        restore()


@pytest.fixture
def fc_ledger():
    ns, restore = F.load(
        extra_targets={
            "opti_oignon.memory.drift": source("memory", "drift.py"),
            "opti_oignon.memory.ledger_store": source("memory", "ledger_store.py"),
        },
        packages=("opti_oignon.memory",),
    )
    try:
        yield ns
    finally:
        restore()


def _searched(verdict, source_id):
    return next(e for e in verdict.record["sources_searched"] if e["source_id"] == source_id)


def test_ad1_only_the_owner_or_a_named_third_party_is_evidence(fc):
    cfg = F.config(fc)

    # c1 -- refused by name; retracted shown as a flag, never used.
    text = f"From the chat. {RHINE}"
    refused = {
        "author_model": F.item(fc, "library:m", RHINE, author="model"),
        "author_unknown": F.item(fc, "library:u", RHINE, author="unknown"),
        "snippet": F.item(fc, "web:s", RHINE, kind="snippet"),
        "no_consent": F.item(fc, "library:c", RHINE, consent=None),
        "retracted": F.item(fc, "library:r", RHINE, flags={"retracted": "2025-05-01"}),
    }
    for code, one in refused.items():
        verdict = F.run(fc, RHINE, [one], cfg=cfg)
        assert verdict.value == "not_enough_evidence", (code, verdict.value)
        assert _searched(verdict, one.source_id)["admission"] == code
    verdict = F.run(fc, RHINE, [refused["retracted"]], cfg=cfg)
    assert _searched(verdict, "library:r")["flags"] == {"retracted": "2025-05-01"}
    assert "retracted on 2025-05-01" in verdict.text, verdict.text
    quoted = F.item(fc, "note:q", text, kind="note", author="user",
                    chunk_fields={"model_quoted": ((5, len(text) - 3),)})
    verdict = F.run(fc, RHINE, [quoted], cfg=cfg)
    assert verdict.value == "not_enough_evidence"
    assert [r["refusal"] for r in _searched(verdict, "note:q")["span_refusals"]] == ["author_model_quoted"]

    # c2 -- corrected admitted and shown; ingested before its correction;
    # preprint shown; cites_retracted.
    fixed = F.item(fc, "library:fixed", RHINE, flags={"corrected": "2026-03-01"},
                   chunk_fields={"ingested_at": "2026-04-02"})
    verdict = F.run(fc, RHINE, [fixed], cfg=cfg)
    assert verdict.value == "supported" and "corrected on 2026-03-01" in verdict.text, verdict.text
    early = F.item(fc, "library:early", RHINE, flags={"corrected": "2026-03-01"},
                   chunk_fields={"ingested_at": "2026-01-15"})
    verdict = F.run(fc, RHINE, [early], cfg=cfg)
    assert verdict.value == "not_enough_evidence" and "ingested_before_correction" in verdict.reasons
    preprint = F.item(fc, "library:pre", RHINE, flags={"preprint": ""})
    verdict = F.run(fc, RHINE, [preprint], cfg=cfg)
    assert verdict.value == "supported" and "(preprint)" in verdict.text, verdict.text
    citing = F.item(fc, "library:cites", RHINE, flags={"cites_retracted": ""})
    verdict = F.run(fc, RHINE, [citing], cfg=cfg)
    assert verdict.value == "not_enough_evidence" and "cites_retracted" in verdict.reasons, verdict.reasons

    # c3 -- extraction flags: no digit supported "in the extracted text of";
    # a digit is extraction_uncertain.
    digits = "The Rhine is 1,230 km long from source to mouth."
    for flag in ("table", "ocr", "multi_column"):
        extraction = {"extractor": "pdf-text", "extractor_version": "1.0", "flags": (flag,)}
        pdf = F.item(fc, f"library:{flag}", f"{RHINE} {digits}",
                     chunk_fields={"extraction": extraction, "locator": {"page": 12}})
        verdict = F.run(fc, RHINE, [pdf], cfg=cfg)
        assert verdict.value == "supported", (flag, verdict.reasons)
        assert f"in the extracted text of library:{flag}, p. 12" in verdict.text, verdict.text
        verdict = F.run(fc, digits, [pdf], cfg=cfg)
        assert verdict.value == "not_enough_evidence" and "extraction_uncertain" in verdict.reasons, (
            flag, verdict.reasons)

    # c4 -- tiers in the verdict; own notes alone say so; an own claim ignores
    # library and web sources.
    own_note = F.item(fc, "note:rhine", RHINE, kind="note", author="user")
    verdict = F.run(fc, RHINE, [own_note], cfg=cfg)
    assert verdict.value == "supported" and verdict.record["verdict"]["support_tiers"] == ["own_note"]
    assert verdict.text.startswith("Your notes say"), verdict.text
    both = [own_note, F.item(fc, "library:rhine", RHINE)]
    verdict = F.run(fc, RHINE, both, cfg=cfg)
    assert verdict.record["verdict"]["support_tiers"] == ["library", "own_note"]
    assert not verdict.text.startswith("Your notes say"), verdict.text
    decision = "We decided to hold the offsite in Berlin."
    others = [F.item(fc, "library:plan", decision), F.item(fc, "web:plan", decision, kind="web")]
    verdict = F.run(fc, decision, others, cfg=cfg)
    assert verdict.record["claim"]["kind"] == "own"
    assert verdict.value == "not_enough_evidence", verdict.reasons
    assert [_searched(verdict, i.source_id)["counted"] for i in others] == [False, False]
    mine = F.item(fc, "ledger:plan", decision, kind="ledger", author="user")
    verdict = F.run(fc, decision, [*others, mine], cfg=cfg)
    assert verdict.value == "supported" and verdict.record["verdict"]["support_tiers"] == ["own_decision"]
    # With only other tiers handed in, the text says no decision or note of his was.
    verdict = F.run(fc, decision, others, cfg=cfg)
    assert verdict.text.startswith("No decision or note of yours was handed in"), verdict.text


def _ledger(fc, tmp_path, name, old, new):
    """Hold ``old`` then supersede it with ``new`` through the real store."""
    drift = fc.drift
    store = fc.ledger_store.LedgerStore(tmp_path / f"{name}.sqlite", connect=sqlite3.connect)
    store.add(drift.LedgerFact(id=f"{name}-x", statement=old, kind="project", provenance=["turn:1"]))
    store.supersede(f"{name}-x", drift.LedgerFact(id=f"{name}-y", statement=new, kind="project",
                                                  provenance=["turn:2"]))
    rows = {fact.id: fact for fact in store.all()}
    x, y = rows[f"{name}-x"], rows[f"{name}-y"]
    # The premise, read from the store: nothing but the link is written.
    assert (x.valid_from, x.valid_until, x.superseded_by) == ("", "", f"{name}-y")
    assert (y.valid_from, y.valid_until, y.superseded_by) == ("", "", None)
    recorded = {f"{name}-x": "2026-06-02", f"{name}-y": "2026-07-10"}
    items = []
    for fact in (x, y):
        items.append(F.item(
            fc, f"ledger:{fact.id}", fact.statement, kind="ledger", author="user",
            valid_from=fact.valid_from, valid_until=fact.valid_until,
            superseded_by=f"ledger:{fact.superseded_by}" if fact.superseded_by else None,
            recorded_at=recorded[fact.id], source_date=None))
    return items


def test_tv1_validity_on_the_ledger_as_it_stores_supersession(fc_ledger, tmp_path):
    fc = fc_ledger
    cfg = F.config(fc)
    berlin = "We decided to hold the offsite in Berlin."
    oslo = "We decided to hold the offsite in Oslo."
    pair = _ledger(fc, tmp_path, "d", berlin, oslo)

    # c6 -- an empty start falls back to the recording date; none at all:
    # valid, validity unknown, shown.
    dated = F.item(fc, "library:dated", RHINE, recorded_at="2026-03-01")
    entry = _searched(F.run(fc, RHINE, [dated], cfg=cfg), "library:dated")
    assert entry["validity"]["start"] == {"date": "2026-03-01", "from": "recorded_at"}
    unknown = F.item(fc, "library:unknown", RHINE, recorded_at="")
    verdict = F.run(fc, RHINE, [unknown], cfg=cfg)
    assert verdict.value == "supported"
    assert "validity_unknown" in _searched(verdict, "library:unknown")["validity"]["flags"]
    assert "validity unknown" in verdict.text, verdict.text

    # c5 -- a note carrying a successor is coarse and stays valid; a successor
    # not handed in refuses the old fact.
    note = F.item(fc, "note:plan", "The ferry to Belle-Ile leaves Quiberon at noon every day.",
                  kind="note", author="user", superseded_by="note:plan-2")
    verdict = F.run(fc, "The ferry to Belle-Ile leaves Quiberon at noon every day.", [note], cfg=cfg)
    assert verdict.value == "supported"
    assert "supersession_coarse" in _searched(verdict, "note:plan")["validity"]["flags"]
    orphan = F.item(fc, "ledger:o", berlin, kind="ledger", author="user", superseded_by="ledger:gone",
                    recorded_at="2026-06-02")
    verdict = F.run(fc, berlin, [orphan], cfg=cfg)
    assert verdict.value == "not_enough_evidence"
    assert _searched(verdict, "ledger:o")["admission"] == "successor_missing"

    # c2 -- the validity filter: before the replacement the old decision is
    # supported, after it the new one, and nothing supports before it holds.
    verdict = F.run(fc, berlin, pair, as_of="2026-06-15", cfg=cfg)
    assert verdict.value == "supported", verdict.reasons
    assert [s["source_id"] for s in verdict.record["spans"] if s["role"] == "support"] == ["ledger:d-x"]
    verdict = F.run(fc, oslo, pair, as_of="2026-09-26", cfg=cfg)
    assert verdict.value == "supported"
    assert [s["source_id"] for s in verdict.record["spans"] if s["role"] == "support"] == ["ledger:d-y"]
    assert F.run(fc, oslo, pair, as_of="2026-06-15", cfg=cfg).value != "supported"
    dated = F.item(fc, "library:dated", RHINE, recorded_at="2026-03-01")
    verdict = F.run(fc, RHINE, [dated], as_of="2026-02-01", cfg=cfg)
    assert verdict.value == "not_enough_evidence" and verdict.leading == "no_valid_source"

    # c1 -- the past decision replaced: conflicting, superseded, both dates.
    verdict = F.run(fc, berlin, pair, as_of="2026-09-26", cfg=cfg)
    assert (verdict.value, verdict.basis, verdict.reasons) == ("conflicting", "deterministic", ("superseded",))
    detail = verdict.details["superseded"]
    assert (detail["d1"]["date"], detail["d1"]["label"], detail["d1"]["source_id"]) == (
        "2026-06-02", "recorded on", "ledger:d-x")
    assert (detail["d2"]["date"], detail["d2"]["label"], detail["d2"]["source_id"]) == (
        "2026-07-10", "recorded on", "ledger:d-y")
    assert "recorded on 2026-06-02" in verdict.text and "recorded on 2026-07-10" in verdict.text
    assert oslo in verdict.text and verdict.text.index("2026-06-02") < verdict.text.index("2026-07-10")
    # Replaced twice, through the real store: the later decision named is the one
    # quoted, with its own date, and the one in between is named with its date.
    lisbon = "We decided to hold the offsite in Lisbon."
    drift = fc.drift
    store = fc.ledger_store.LedgerStore(tmp_path / "chain.sqlite", connect=sqlite3.connect)
    store.add(drift.LedgerFact(id="c-x", statement=berlin, kind="project", provenance=["turn:1"]))
    store.supersede("c-x", drift.LedgerFact(id="c-y", statement=oslo, kind="project", provenance=["turn:2"]))
    store.supersede("c-y", drift.LedgerFact(id="c-z", statement=lisbon, kind="project", provenance=["turn:3"]))
    recorded = {"c-x": "2026-06-02", "c-y": "2026-07-10", "c-z": "2026-08-20"}
    chain = [F.item(fc, f"ledger:{fact.id}", fact.statement, kind="ledger", author="user",
                    valid_from=fact.valid_from, valid_until=fact.valid_until,
                    superseded_by=f"ledger:{fact.superseded_by}" if fact.superseded_by else None,
                    recorded_at=recorded[fact.id], source_date=None) for fact in store.all()]
    verdict = F.run(fc, berlin, chain, as_of="2026-09-26", cfg=cfg)
    assert verdict.reasons == ("superseded",), verdict.reasons
    detail = verdict.details["superseded"]
    assert (detail["d2"]["date"], detail["d2"]["source_id"]) == ("2026-08-20", "ledger:c-z"), detail
    assert [(m["source_id"], m["date"]) for m in detail["via"]] == [("ledger:c-y", "2026-07-10")], detail
    assert "recorded on 2026-08-20 (ledger:c-z" in verdict.text and lisbon in verdict.text, verdict.text
    assert "through ledger:c-y, recorded on 2026-07-10" in verdict.text, verdict.text
    # A successor with no date, or one before an explicit end: the replaced decision never stands.
    def replaced(y_recorded, x_until=""):
        return [F.item(fc, "ledger:x", berlin, kind="ledger", author="user", superseded_by="ledger:y",
                       recorded_at="2026-06-02", valid_until=x_until, source_date=None),
                F.item(fc, "ledger:y", oslo, kind="ledger", author="user", recorded_at=y_recorded,
                       source_date=None)]
    for as_of in ("2026-06-15", "2026-09-26"):
        verdict = F.run(fc, berlin, replaced(""), as_of=as_of, cfg=cfg)
        assert (verdict.value, verdict.reasons) == ("conflicting", ("superseded",)), (as_of, verdict.reasons)
        assert "validity_unknown" in _searched(verdict, "ledger:x")["validity"]["flags"]
    verdict = F.run(fc, berlin, replaced("2026-07-10", x_until="2026-12-31"), cfg=cfg)
    assert (verdict.value, verdict.reasons) == ("conflicting", ("superseded",)), verdict.reasons
    assert _searched(verdict, "ledger:x")["validity"]["end"] == {"date": "2026-07-10", "from": "successor_start"}
    # A successor dated before what it replaced: both dates unknown, never "later".
    verdict = F.run(fc, berlin, replaced("2026-05-01"), cfg=cfg)
    assert verdict.value == "not_enough_evidence" and "no_valid_source" in verdict.reasons, verdict.reasons
    assert all("validity_unknown" in e["validity"]["flags"] for e in verdict.record["sources_searched"])
    assert "later" not in verdict.text and "when it held is unknown" in verdict.text, verdict.text

    # c3 -- a present state replaced: no_longer_held, never conflicting or supported.
    state = _ledger(fc, tmp_path, "s", "The offsite is in Berlin.", "The offsite is in Oslo.")
    verdict = F.run(fc, "The offsite is in Berlin.", state, as_of="2026-09-26", cfg=cfg)
    assert verdict.value == "not_enough_evidence" and "no_longer_held" in verdict.reasons, verdict.reasons
    assert "held until 2026-07-10" in verdict.text.lower() and "The offsite is in Oslo." in verdict.text
    # A decision with its own date, or in the English present, is no longer held, not conflicting.
    dated = "We decided in June 2026 to hold the offsite in Berlin."
    present = "We decide to hold the offsite in Berlin."
    for claim in (dated, present):
        items = [F.item(fc, "ledger:x", claim, kind="ledger", author="user", superseded_by="ledger:y",
                        recorded_at="2026-06-02", source_date=None),
                 F.item(fc, "ledger:y", oslo, kind="ledger", author="user", recorded_at="2026-07-10",
                        source_date=None)]
        verdict = F.run(fc, claim, items, cfg=cfg)
        assert verdict.value == "not_enough_evidence" and "no_longer_held" in verdict.reasons, (
            claim, verdict.reasons)

    # c4 -- the second person in assistant text, and the French pair.
    you = fc.scope.Claim(text="You decided to hold the offsite in Berlin.", lang="en",
                         origin={"author": "assistant"})
    assert F.run(fc, you, pair, as_of="2026-09-26", cfg=cfg).reasons == ("superseded",)
    assert F.run(fc, you, pair, as_of="2026-06-15", cfg=cfg).value == "supported"
    e, a = chr(0xE9), chr(0xE0)
    nous_berlin = f"Nous avons d{e}cid{e} de faire le s{e}minaire {a} Berlin."
    nous_oslo = f"Nous avons d{e}cid{e} de faire le s{e}minaire {a} Oslo."
    french = _ledger(fc, tmp_path, "f", nous_berlin, nous_oslo)
    tu = fc.scope.Claim(text=f"Tu as d{e}cid{e} de faire le s{e}minaire {a} Berlin.", lang="fr",
                        origin={"author": "assistant"})
    assert F.run(fc, tu, french, as_of="2026-09-26", cfg=cfg).reasons == ("superseded",)
    assert F.run(fc, tu, french, as_of="2026-06-15", cfg=cfg).value == "supported"
    # Who the owner is depends on who wrote each side, and only the opening
    # subject, its auxiliary and that person's possessives are rewritten.
    swaps = [
        ("the owner's note in the second person", fc.scope.Claim(text=berlin, lang="en"),
         F.item(fc, "note:plan", "You decided to hold the offsite in Berlin.", kind="note", author="user")),
        ("the assistant's own first person",
         fc.scope.Claim(text="I decided to write the script in Python.", lang="en", origin={"author": "assistant"}),
         F.item(fc, "ledger:py", "I decided to write the script in Python.", kind="ledger", author="user")),
        ("a possessive swapped", fc.scope.Claim(text="I decided to sell my car.", lang="en"),
         F.item(fc, "note:car", "I decided to sell your car.", kind="note", author="user")),
        ("a subject past the opening", fc.scope.Claim(text="We decided that we pay the deposit.", lang="en"),
         F.item(fc, "note:deposit", "We decided that I pay the deposit.", kind="note", author="user")),
        ("an object pronoun swapped",
         fc.scope.Claim(text=f"Vous avez d{e}cid{e} de nous payer 500 euros.", lang="fr",
                        origin={"author": "assistant"}),
         F.item(fc, "ledger:pay", f"Nous avons d{e}cid{e} de vous payer 500 euros.", kind="ledger",
                author="user", lang="fr")),
    ]
    for label, claim, evidence in swaps:
        verdict = F.run(fc, claim, [evidence], cfg=cfg)
        assert verdict.record["claim"]["kind"] == "own" and verdict.value == "not_enough_evidence", (
            label, verdict.value, verdict.reasons)
    mine = F.item(fc, "ledger:car", "We decided to sell our car.", kind="ledger", author="user")
    yours = fc.scope.Claim(text="You decided to sell your car.", lang="en", origin={"author": "assistant"})
    assert F.run(fc, yours, [mine], cfg=cfg).value == "supported"


def test_tv2_a_sources_own_date_is_shown_and_undated_does_not_vouch_for_now(fc):
    cfg = F.config(fc)

    # c1 -- the source's own date, or "undated", beside the date read.
    verdict = F.run(fc, RHINE, [F.item(fc, "library:dated", RHINE, source_date="2025-11-02")], cfg=cfg)
    assert verdict.value == "supported" and "(2025-11-02)" in verdict.text and F.READ_ON in verdict.text
    # Each cited source, not the first alone; the date the claim is checked for.
    two = [F.item(fc, "library:a", RHINE, source_date="2024-01-05"),
           F.item(fc, "library:b", RHINE, source_date="2023-02-01", flags={"preprint": ""})]
    verdict = F.run(fc, RHINE, two, cfg=cfg)
    assert "library:a (2024-01-05)" in verdict.text and "library:b (2023-02-01 (preprint))" in verdict.text, (
        verdict.text)
    assert f"Checked for {F.AS_OF}" in verdict.text, verdict.text
    # A decision supported at a past date, replaced since: the text says so.
    berlin, oslo = "We decided to hold the offsite in Berlin.", "We decided to hold the offsite in Oslo."
    pair = [F.item(fc, "ledger:x", berlin, kind="ledger", author="user", superseded_by="ledger:y",
                   recorded_at="2026-06-02", source_date="2026-06-02"),
            F.item(fc, "ledger:y", oslo, kind="ledger", author="user", recorded_at="2026-07-10",
                   source_date="2026-07-10")]
    verdict = F.run(fc, berlin, pair, as_of="2026-06-15", cfg=cfg)
    assert verdict.value == "supported" and "Checked for 2026-06-15" in verdict.text, verdict.text
    assert "Later replaced, on 2026-07-10, by ledger:y" in verdict.text and oslo in verdict.text, verdict.text

    # c3 -- a timeless claim in an undated source: supported, "undated" shown.
    moon = "The Moon is 384,400 km away."
    verdict = F.run(fc, moon, [F.item(fc, "library:moon", moon, source_date=None)], cfg=cfg)
    assert verdict.value == "supported" and "(undated)" in verdict.text, verdict.text
    verdict = F.run(fc, RHINE, [F.item(fc, "library:undated", RHINE, source_date=None)], cfg=cfg)
    assert "(undated)" in verdict.text and F.READ_ON in verdict.text, verdict.text

    # c2 -- a claim about the present, in an undated source and in a dated one.
    e_grave = chr(0xE8)
    for claim, lang in (("The current CEO of Acme is Jane Roe.", "en"),
                        (f"La derni{e_grave}re version de Firefox est la 131.", "fr")):
        undated = F.item(fc, "library:u", claim, source_date=None, lang=lang)
        verdict = F.run(fc, fc.scope.Claim(text=claim, lang=lang), [undated], cfg=cfg)
        assert (verdict.value, verdict.leading) == ("not_enough_evidence", "source_undated"), (
            claim, verdict.reasons)
        dated = F.item(fc, "library:d", claim, source_date="2026-09-01", lang=lang)
        verdict = F.run(fc, fc.scope.Claim(text=claim, lang=lang), [dated], cfg=cfg)
        assert verdict.value == "supported" and "(2026-09-01)" in verdict.text, (claim, verdict.text)
        # A source dated after the date checked for says what held on its own date, not then.
        verdict = F.run(fc, fc.scope.Claim(text=claim, lang=lang), [dated], as_of="2026-01-01", cfg=cfg)
        assert verdict.value == "not_enough_evidence" and "no_valid_source" in verdict.reasons, (
            claim, verdict.reasons)
        assert _searched(verdict, "library:d")["validity"]["start"] == {"date": "2026-09-01", "from": "source_date"}
    # A check for a date after the day it runs is refused by name.
    with pytest.raises(ValueError, match="after read_on"):
        F.run(fc, RHINE, [F.item(fc, "library:dated", RHINE)], as_of="2030-01-01", cfg=cfg)
