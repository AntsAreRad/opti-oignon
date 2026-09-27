#!/usr/bin/env python3
"""Every verdict is a record that replays, and its text never says more than the evidence.

A verdict carries a record in canonical JSON: the claim, every source handed
in, every span with its window, every check, the verdict with its reasons, and
the rules that decided it, under a digest. The text shown is built from closed
templates:

  * RR1 -- same inputs, same record: two calls give byte-identical canonical
    JSON and the same id, the creation time outside it and the date read
    inside it; items and chunks in any order give the same id; no float
    anywhere; a replay reproduces the id, and says the source changed when
    one byte of a chunk did and the rules changed when one lexicon word did;
    the record names the path that computed it, outside the digest, and
    says it is a digest, not a signature.
  * RR2 -- the record holds what a reviewer needs: for every verdict the core
    can give, every field it writes is present, the judge null with its
    reason, the configuration's digest the digest of the configuration
    recorded, and every candidate checked with its passage outcome and what
    blocked it; every span carries its source, chunk hash, local and
    absolute offsets, the located text's hash and its window; a passage
    located past the cap says so; every refused source carries its reason;
    and every verdict other than "supported" sets side by side the claim as
    written, with its offsets, and each source passage examined, with its
    source, chunk hash and offsets, each differing run of words named: the
    nearest sentence of a chunk that restates nothing (one digit changed,
    named beside the claim's), both sides of a replaced decision with their
    sources and tiers, and a restating passage from a source not valid at
    the time with when it held.
  * VT1 -- the display never says more than the evidence: no template, and no
    string literal of the module that builds the texts, holds a word of truth
    or verification, in English or folded French, inflected or not, and no
    text the canary produces does outside the source's own quoted words; a
    supported or conflicting text names its source, passage, the source's
    own date or "undated" and the date read, an extracted source says "in
    the extracted text of", "not enough evidence" names the sources searched
    or says none were, a replaced decision gives both dates; the per-answer
    summary has no percentage, leads with the most severe verdict, keeps the
    claims in answer order and counts what was not checked by reason.

Loaded through the shared isolation window (``tests/_factcheck.py``), with the
native core unreachable. Nothing reaches a model, the network or the
maintainer's data.
"""

import ast
import copy
import dataclasses
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _factcheck as F  # noqa: E402

BUDGET_S = {
    "test_rr1_same_inputs_same_record_and_the_record_replays": 2.0,
    "test_rr2_the_record_holds_what_a_reviewer_needs": 2.0,
    "test_vt1_the_display_never_says_more_than_the_evidence": 2.0,
}

MOON = "The Moon is 384,400 km away."
RHINE = "The Rhine flows into the North Sea near Rotterdam."
BERLIN = "We decided to hold the offsite in Berlin."
OSLO = "We decided to hold the offsite in Oslo."


@pytest.fixture
def fc():
    ns, restore = F.load()
    try:
        yield ns
    finally:
        restore()


def _items(fc):
    return [
        F.item(fc, "library:moon", ["Astronomy notes. " + MOON, "The Sun is a star of type G."],
               chunk_fields={"offset_in_source": 1000}),
        F.item(fc, "note:moon", f"# Sky\n\n{MOON} Tides follow it.\n", kind="note", author="user"),
        F.item(fc, "library:model", MOON, author="model"),
    ]


def _pair(fc):
    return [
        F.item(fc, "ledger:x", BERLIN, kind="ledger", author="user", superseded_by="ledger:y",
               recorded_at="2026-06-02", source_date=None),
        F.item(fc, "ledger:y", OSLO, kind="ledger", author="user", recorded_at="2026-07-10",
               source_date=None),
    ]


def _floats(value):
    if isinstance(value, float):
        return [value]
    if isinstance(value, dict):
        return [f for v in value.values() for f in _floats(v)]
    if isinstance(value, (list, tuple)):
        return [f for v in value for f in _floats(v)]
    return []


def test_rr1_same_inputs_same_record_and_the_record_replays(fc):
    stamps = iter(["2026-09-27T08:00:00+00:00", "2026-09-27T09:30:00+00:00", "2026-09-27T11:00:00+00:00"])
    checker = fc.checker.FactChecker(clock=lambda: next(stamps))
    items = _items(fc)

    # c1 -- two calls, byte-identical canonical JSON, same id; created_at
    # outside the digest, read_on inside it.
    one = checker.check(MOON, items, as_of=F.AS_OF, read_on=F.READ_ON)
    two = checker.check(MOON, items, as_of=F.AS_OF, read_on=F.READ_ON)
    assert one.record["created_at"] != two.record["created_at"]
    assert fc.record.canonical(fc.record.inside(one.record)) == fc.record.canonical(fc.record.inside(two.record))
    assert one.record["record_id"] == two.record["record_id"]
    assert one.record["record_id"] == fc.record.digest(fc.record.inside(one.record))
    later = checker.check(MOON, items, as_of=F.AS_OF, read_on="2026-09-28")
    assert later.record["record_id"] != one.record["record_id"]

    # c2 -- permuted items and chunks: the same id.
    permuted = [dataclasses.replace(i, chunks=tuple(reversed(i.chunks))) for i in reversed(items)]
    again = fc.decide.check(MOON, permuted, as_of=F.AS_OF, read_on=F.READ_ON, config=checker.config,
                            canary=checker.canary.summary())
    assert again.record["record_id"] == one.record["record_id"]

    # c3 -- no float anywhere; the canonical writer refuses one.
    for verdict in (one, fc.decide.check(BERLIN, _pair(fc), as_of=F.AS_OF, read_on=F.READ_ON,
                                         config=checker.config)):
        assert _floats(json.loads(fc.record.canonical(verdict.record))) == []
        # The record itself, before any writer could refuse a float.
        assert _floats(verdict.record) == [], _floats(verdict.record)
    assert _floats({"a": [1, 0.5], "b": ("x", 2.0)}) == [0.5, 2.0], "the float walk finds a float"
    with pytest.raises((TypeError, ValueError)):
        fc.record.canonical({"score": 0.5})

    # c4 -- replay reproduces the id; a byte changed is source_changed; a
    # lexicon word changed is rules_changed.
    texts = [c.text for i in items for c in i.chunks]
    replay = fc.record.replay(one.record, texts)
    assert (replay.outcome, replay.path, replay.record_id) == ("reproduced", "reference", one.record["record_id"])
    tampered = [t.replace("384,400", "384,401") for t in texts]
    assert fc.record.replay(one.record, tampered).outcome == "source_changed"
    original = fc.scope.QUALIFYING_MARKERS
    fc.scope.QUALIFYING_MARKERS = (*original, "apocryphal")
    try:
        assert fc.record.replay(one.record, texts).outcome == "rules_changed"
    finally:
        fc.scope.QUALIFYING_MARKERS = original
    assert fc.record.replay(one.record, texts).outcome == "reproduced"

    # c5 -- the path outside the digest; a digest, not a signature.
    assert one.record["computed_by"] == {"path": "reference"}
    assert one.record["integrity"] == "digest_only"
    inside = fc.record.inside(one.record)
    assert "computed_by" not in inside and "created_at" not in inside and "record_id" not in inside
    moved = copy.deepcopy(one.record)
    moved["computed_by"] = {"path": "native", "artefact": "0" * 64}
    assert fc.record.digest(fc.record.inside(moved)) == one.record["record_id"]


RECORD_FIELDS = ("record_version", "rules_digest", "rules_version", "unicode_version", "config_digest",
                 "config", "canary", "claim", "parts", "sources_searched", "spans", "checks", "judge",
                 "verdict", "text", "integrity", "record_id", "created_at", "computed_by")
CLAIM_FIELDS = ("text", "sha256", "lang", "kind", "as_of", "read_on", "origin", "wrapper", "marks",
                "scope", "checked", "markup", "split")
VERDICT_FIELDS = ("value", "basis", "reasons", "leading", "details", "support_tiers", "contradict_tiers")
SPAN_FIELDS = ("source_id", "chunk_sha256", "start", "end", "abs_start", "abs_end", "text_sha256",
               "window", "role")
WINDOW_FIELDS = ("sentence", "before", "after", "heading", "lead_in")


def test_rr2_the_record_holds_what_a_reviewer_needs(fc):
    cfg = F.config(fc)
    qualified = F.item(fc, "note:myth", f"# Myths\n\n{RHINE}\n", kind="note", author="user")
    cases = {
        "supported": F.run(fc, MOON, _items(fc), cfg=cfg),
        "conflicting": F.run(fc, BERLIN, _pair(fc), cfg=cfg),
        "not_enough_evidence": F.run(fc, RHINE, [qualified, F.item(fc, "library:m", RHINE, author="model")],
                                     cfg=cfg),
        "out_of_scope": F.run(fc, "Is the Moon far?", _items(fc), cfg=cfg),
    }

    # c1 -- every field this core writes, for every value it can give.
    for value, verdict in cases.items():
        record = verdict.record
        assert verdict.value == value, (value, verdict.reasons)
        missing = [f for f in RECORD_FIELDS if f not in record]
        missing += [f"claim.{f}" for f in CLAIM_FIELDS if f not in record["claim"]]
        missing += [f"verdict.{f}" for f in VERDICT_FIELDS if f not in record["verdict"]]
        missing += [f"canary.{f}" for f in ("digest", "n", "outcome") if f not in record["canary"]]
        assert missing == [], (value, missing)
        assert (record["judge"], record.get("judge_reason")) == (None, "no_judge"), (value, "a judge is null with its reason")
        assert record["config_digest"] == fc.record.digest(record["config"]), value
        if value in ("supported", "conflicting", "not_enough_evidence"):
            assert record["spans"], (value, "no span recorded")
            for span in record["spans"]:
                assert [f for f in SPAN_FIELDS if f not in span] == [], (value, span)
                assert [f for f in WINDOW_FIELDS if f not in span["window"]] == [], (value, span["window"])

    # c2 -- spans carry local and absolute offsets, the located text's hash and
    # the window; refused sources carry their reason.
    record = cases["supported"].record
    library = [s for s in record["spans"] if s["source_id"] == "library:moon"]
    assert library and library[0]["abs_start"] == 1000 + library[0]["start"]
    assert library[0]["abs_end"] == 1000 + library[0]["end"]
    assert library[0]["text_sha256"] == F.sha(MOON)
    note = [s for s in record["spans"] if s["source_id"] == "note:moon"]
    assert note and note[0]["abs_start"] is None and note[0]["window"]["heading"] is not None
    for verdict in cases.values():
        for entry in verdict.record["sources_searched"]:
            assert entry["admission"] == "admitted" or entry["admission"] in fc.vocabulary.REFUSALS, entry
    # Every candidate is checked with its passage outcome and what blocked it.
    checks = cases["not_enough_evidence"].record["checks"]
    assert [(c["source_id"], c["passage"], [b["reason"] for b in c["blockers"]]) for c in checks] == [
        ("note:myth", "located", ["context_qualified"])], checks
    # A passage the host located past its cap says so.
    repeated = " ".join([MOON] * 20)
    capped = F.run(fc, MOON, [F.item(fc, "library:many", repeated)], cfg=cfg)
    outcomes = [(c["passage"], c["truncated"]) for c in capped.record["checks"]]
    assert outcomes.count(("located", True)) == cfg["limits"]["max_occurrences"], outcomes
    assert outcomes.count(("truncated", True)) == 20 - cfg["limits"]["max_occurrences"], outcomes

    # c3 -- every verdict other than supported sets the claim as written beside
    # each passage examined, each differing run named; both sides of a
    # replaced decision with their sources and tiers.
    assert cases["supported"].record["side_by_side"] is None
    for value in ("conflicting", "not_enough_evidence", "out_of_scope"):
        side = cases[value].record["side_by_side"]
        claim_text = cases[value].record["claim"]["text"]
        assert side["claim"]["text"] == claim_text, (value, side["claim"])
        assert (side["claim"]["start"], side["claim"]["end"]) == (0, len(claim_text))
        for passage in side["passages"]:
            assert {"source_id", "tier", "chunk_sha256", "start", "end", "role", "differences"} <= set(passage)
    conflict = cases["conflicting"].record["side_by_side"]["passages"]
    roles = {p["role"]: p for p in conflict}
    assert set(roles) == {"restates", "replaced_by"}, roles
    assert (roles["restates"]["source_id"], roles["restates"]["tier"]) == ("ledger:x", "own_decision")
    assert (roles["replaced_by"]["source_id"], roles["replaced_by"]["tier"]) == ("ledger:y", "own_decision")
    assert {"claim": "Berlin", "source": "Oslo"} in roles["replaced_by"]["differences"], roles["replaced_by"]
    assert roles["restates"]["differences"] == []
    qualified_side = cases["not_enough_evidence"].record["side_by_side"]["passages"]
    blocked = [p for p in qualified_side if p["role"] == "restates"]
    assert blocked and blocked[0]["blocked_by"][0]["reason"] == "context_qualified", qualified_side
    paraphrase = F.run(fc, "Rotterdam lies where the Rhine meets the North Sea.",
                       [F.item(fc, "library:rhine", RHINE)], cfg=cfg)
    searched = paraphrase.record["side_by_side"]["passages"]
    assert [(p["source_id"], p["role"], p["text"]) for p in searched] == [("library:rhine", "nearest", RHINE)]
    assert searched[0]["differences"] and searched[0]["sentences_compared"] == 1, searched
    # A sentence one digit away is shown beside the claim, the digit named on both sides.
    bridge = "Paris bridges. The Pont Neuf was completed in 1607 under Henri IV. It spans the Seine."
    near = F.run(fc, "The Pont Neuf was completed in 1608 under Henri IV.",
                 [F.item(fc, "library:bridge", bridge)], cfg=cfg)
    passages = near.record["side_by_side"]["passages"]
    assert [(p["role"], bridge[p["start"]:p["end"]]) for p in passages] == [
        ("nearest", "The Pont Neuf was completed in 1607 under Henri IV.")], passages
    assert {"claim": "1608", "source": "1607"} in passages[0]["differences"], passages
    assert '"1608"' in near.text and '"1607"' in near.text, near.text
    # A chunk with no sentence near the claim: searched, its sentences counted.
    far = F.run(fc, "Rotterdam lies where the Rhine meets the North Sea.",
                [F.item(fc, "library:sun", "The Sun is a star. It shines.")], cfg=cfg)
    assert [(p["role"], p["sentences_compared"]) for p in far.record["side_by_side"]["passages"]] == [
        ("searched", 2)], far.record["side_by_side"]
    # Sources not read as evidence are named with why, their text not examined.
    assert cases["not_enough_evidence"].record["side_by_side"]["not_examined"] == [
        {"source_id": "library:m", "why": "author_model"}]
    assert cases["out_of_scope"].record["side_by_side"]["passages"] == []
    # A passage that restates the claim from a source not valid at t says when it held.
    later = F.run(fc, RHINE, [F.item(fc, "library:later", RHINE, valid_from="2026-10-01")], cfg=cfg)
    restating = [p for p in later.record["side_by_side"]["passages"] if p["role"] == "restates"]
    assert [b["reason"] for b in restating[0]["blocked_by"]] == ["no_valid_source"], restating
    assert (restating[0]["blocked_by"][0]["start"], restating[0]["blocked_by"][0]["start_from"]) == (
        "2026-10-01", "valid_from"), restating


def _module_literals(source_text):
    """Every string literal of a module, and the constant parts of its f-strings, docstrings left out."""
    tree = ast.parse(source_text)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            found.append(node.value)
    return found


def _unquoted(text):
    """A text with every double-quoted run cut out: the source's own words, which may say anything."""
    return re.sub(r'"[^"]*"', '""', text)


def _strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (*_strings(k), *_strings(v))]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [s for v in value for s in _strings(v)]
    return []


def test_vt1_the_display_never_says_more_than_the_evidence(fc):
    cfg = F.config(fc)
    render = fc.render

    # c1 -- no template holds a forbidden word; witness: a planted one is flagged.
    texts = []
    for name, value in vars(render).items():
        if name.startswith("__"):
            continue
        texts.extend(_strings(value))
    assert len(texts) >= 30, len(texts)
    flagged = {t: render.forbidden_in(t) for t in texts if render.forbidden_in(t)}
    assert flagged == {}, flagged
    assert render.forbidden_in("Verified by {source}.") == ["verified"]
    e = chr(0xE9)
    assert render.forbidden_in(f"C'est v{e}rifi{e}.") == ["verifie"]
    # Inflected forms too.
    inflected = {f"V{e}rifi{e}e par la source.": ["verifiee"], "These facts hold.": ["facts"],
                 f"La r{e}ponse est vraie.": ["vraie"], "The truth is out.": ["truth"], "It proves it.": ["proves"]}
    for text, words in inflected.items():
        assert render.forbidden_in(text) == words, (text, render.forbidden_in(text))
    # Every string literal of the module that builds the texts, inside functions
    # too; witness: a literal planted in a function body is found.
    literals = _module_literals(Path(fc.render.__file__).read_text(encoding="utf-8"))
    assert len(literals) >= 60, len(literals)
    flagged = {t: render.forbidden_in(t) for t in literals if render.forbidden_in(t)}
    assert flagged == {}, flagged
    planted = _module_literals("def f(x):\n    return x + ' This is verified and true.'\n")
    assert [render.forbidden_in(t) for t in planted] == [["verified", "true"]], planted
    # Every text the canary produces, the source's own quoted words cut out.
    produced = [v.text for _, v in fc.canary.run(fc.decide.check, config=cfg).results]
    assert len(produced) >= 90, len(produced)
    flagged = {t: render.forbidden_in(_unquoted(t)) for t in produced if render.forbidden_in(_unquoted(t))}
    assert flagged == {}, flagged
    assert render.forbidden_in(_unquoted('Stated in x: "It is true." Verified.')) == ["verified"]

    # c2 -- what supported, conflicting, extracted and not-enough texts name.
    supported = F.run(fc, MOON, [F.item(fc, "library:moon", MOON, source_date="2025-11-02")], cfg=cfg)
    for part in ("library:moon", MOON, "(2025-11-02)", F.READ_ON):
        assert part in supported.text, (part, supported.text)
    undated = F.run(fc, MOON, [F.item(fc, "library:moon", MOON, source_date=None)], cfg=cfg)
    assert "(undated)" in undated.text and F.READ_ON in undated.text
    extracted = F.run(fc, MOON, [F.item(fc, "library:pdf", MOON, chunk_fields={
        "extraction": {"extractor": "pdf-text", "extractor_version": "1.0", "flags": ()}})], cfg=cfg)
    assert "in the extracted text of library:pdf" in extracted.text, extracted.text
    conflicting = F.run(fc, BERLIN, _pair(fc), cfg=cfg)
    for part in ("ledger:x", "ledger:y", BERLIN, OSLO, "2026-06-02", "2026-07-10", "undated", F.READ_ON):
        assert part in conflicting.text, (part, conflicting.text)
    searched = F.run(fc, "Rotterdam lies where the Rhine meets the North Sea.",
                     [F.item(fc, "library:rhine", RHINE), F.item(fc, "library:m", RHINE, author="model")],
                     cfg=cfg)
    assert "library:rhine" in searched.text and "library:m" in searched.text, searched.text
    assert "Not restated as a whole sentence in library:rhine" in searched.text, searched.text
    far = F.run(fc, "Rotterdam lies where the Rhine meets the North Sea.",
                [F.item(fc, "library:sun", "The Sun is a star. It shines.")], cfg=cfg)
    assert "library:sun" in far.text and "nearest" not in far.text, far.text
    nothing = F.run(fc, RHINE, [], cfg=cfg)
    assert "nothing was searched" in nothing.text.lower(), nothing.text

    # c3 -- the per-answer summary.
    heading = fc.scope.claims_from_text("# Rivers", lang="en", origin={}, max_claim_chars=600)[0]
    answer = [
        F.run(fc, MOON, [F.item(fc, "library:moon", MOON)], cfg=cfg),
        F.run(fc, "Rotterdam lies where the Rhine meets the North Sea.", [F.item(fc, "library:r", RHINE)],
              cfg=cfg),
        F.run(fc, heading, [], cfg=cfg),
        F.run(fc, RHINE, [F.item(fc, "library:r", RHINE)], cfg=cfg),
    ]
    summary = render.summarise(answer)
    lines = summary.splitlines()
    assert render.VERDICT_WORDS["not_enough_evidence"] in lines[0], lines[0]
    assert "%" not in summary and "percent" not in summary.lower(), summary
    positions = [summary.index(MOON), summary.index("Rotterdam lies"), summary.index(RHINE, summary.index("Rotterdam lies"))]
    assert positions == sorted(positions), positions
    assert f"{render.PLAIN_REASONS['heading']} 1" in summary, summary
