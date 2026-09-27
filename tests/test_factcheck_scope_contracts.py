#!/usr/bin/env python3
"""Only a proposition is checked, and a sentence is read with what stands around it.

A claim is a sentence of an answer or a note, read through a closed markdown
subset; what is not a proposition is not checked and says why, and a source
sentence is read in its window (its heading, its list's lead-in, the
sentences beside it), which can qualify it:

  * SQ1 -- a question, a code line, a heading, a table row, an image, an
    empty and an over-long sentence are out of scope, each with its reason;
    an opinion opener is stripped and the rest is checked ("I think the Moon
    is 384,400 km away." is supported by the sentence it restates, and so is
    "I think that ..."), an advice opener is stripped and the rest is checked
    only when it carries a digit or a name, else it is an instruction, and a
    wrapper with no word after it leaves an empty claim; a subject that
    refers elsewhere (a pronoun, a possessive, "another", "both", "one of",
    "such") and a definite subject with no name (a month is not one) are out
    of scope although a source restates them, while a demonstrative before a
    name is a determiner; in assistant text a second-person decision is the
    owner's own, in the owner's text it is a claim about the world, whose
    subject names whoever wrote it and is out of scope; an English "On"
    opening a date is never a subject.
  * SQ2 -- claims_from_text reads markdown into sentences: one claim per
    prose sentence, its text free of markup, its offsets pointing at the
    sentence's original text in the answer, markup included; headings,
    table rows, code and images come back out of scope with their reasons,
    never dropped, and the summary counts them by reason; a blockquoted
    sentence is marked quoted; a list item under a heading keeps the heading
    in its window; a raw HTML block and struck-through text are never a
    claim.
  * WQ1 -- the window qualifies what it holds: a restated sentence under the
    heading "Myths" (anywhere on its heading path, or in the heading its
    store gives, and in a source that is not markdown too), under a label
    set above it (bold, a colon, a raw HTML heading), under the French
    lead-in meaning "to verify", under a parent list item, after "The
    following statement is false.", before "This is a myth." or a sibling
    item saying so, beside a denial, or struck through, is not enough
    evidence, the marker and where it stood recorded; a heading or lead-in
    that sets a condition, a forecast or a narrower population blocks it
    with that reason; under a neutral heading it is supported; a list item
    with no lead-in and no heading, a first sentence that opens in
    lowercase, a sentence that may begin or end inside a sentence cut by the
    chunk, and a sentence after a period that may not end the one before
    are incomplete context; after the lead-in "According to the vendor:" or
    its French form, in a heading, a sentence before or after that
    attributes it, or a quotation, it is attributed.

Loaded through the shared isolation window (``tests/_factcheck.py``), with the
native core unreachable. Nothing reaches a model, the network or the
maintainer's data.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _factcheck as F  # noqa: E402

BUDGET_S = {
    "test_sq1_only_a_proposition_is_checked_and_a_wrapper_is_stripped": 2.0,
    "test_sq2_claims_from_text_reads_markdown_with_offsets_and_counts_the_rest": 2.0,
    "test_wq1_the_context_window_qualifies_what_it_holds": 2.0,
}

MOON = "The Moon is 384,400 km away."


@pytest.fixture
def fc():
    ns, restore = F.load()
    try:
        yield ns
    finally:
        restore()


def test_sq1_only_a_proposition_is_checked_and_a_wrapper_is_stripped(fc):
    cfg = F.config(fc)
    library = [F.item(fc, "library:moon", MOON)]

    # c1 -- out of scope, each with its reason.
    long_claim = "The Moon " + "is far away and " * 40 + "that is all."
    assert len(long_claim) > cfg["limits"]["max_claim_chars"]
    out = {
        "question": "Is the Moon 384,400 km away?",
        "code": f"```\n{MOON}\n```",
        "heading": f"# {MOON}",
        "table_row": "| Moon | 384,400 km |\n| --- | --- |",
        "image": f"![{MOON}](moon.png)",
        "empty": "  ... ",
        "too_long": long_claim,
    }
    for reason, claim in out.items():
        verdict = F.run(fc, claim, library, cfg=cfg)
        assert (verdict.value, verdict.basis, verdict.leading) == ("out_of_scope", "none", reason), (
            reason, verdict.value, verdict.reasons)

    # c2 -- an opinion opener is stripped; an advice opener is stripped and the
    # rest checked only when it carries a digit or a name.
    hedged = F.run(fc, "I think the Moon is 384,400 km away.", library, cfg=cfg)
    claim_record = hedged.record["claim"]
    assert claim_record["wrapper"] == "hedged"
    assert claim_record["checked"]["text"].rstrip(".") == "the Moon is 384,400 km away"
    assert (hedged.value, hedged.leading) == ("supported", "verbatim_sentence"), hedged.reasons
    that = F.run(fc, "I think that the Moon is 384,400 km away.", library, cfg=cfg)
    assert that.record["claim"]["checked"]["text"].rstrip(".") == "the Moon is 384,400 km away", that.record["claim"]
    assert (that.value, that.record["claim"]["wrapper"]) == ("supported", "hedged"), that.reasons
    advice = F.run(fc, "You should take 1 g every 4 hours.", library, cfg=cfg)
    assert advice.value != "out_of_scope" and advice.record["claim"]["wrapper"] == "advice"
    assert advice.record["claim"]["checked"]["text"].rstrip(".") == "take 1 g every 4 hours"
    instruction = F.run(fc, "Make sure to back up your notes.", library, cfg=cfg)
    assert (instruction.value, instruction.leading) == ("out_of_scope", "instruction")
    # A wrapper with no word after it leaves nothing to check, even beside a
    # source holding a sentence of punctuation alone.
    dots = [F.item(fc, "library:dots", "Intro. . More text follows here.")]
    for claim in ("I think.", "Je pense que."):
        verdict = F.run(fc, claim, dots, cfg=cfg)
        assert (verdict.value, verdict.leading, verdict.record["claim"]["wrapper"]) == (
            "out_of_scope", "empty", "hedged"), (claim, verdict.value, verdict.reasons)

    # c3 -- a pronoun subject and a definite subject with no name are out of
    # scope although a source restates each; a named subject is in scope.
    released = "It was first released in 1991."
    trial = "The trial enrolled 300 patients."
    sources = [F.item(fc, "library:both", f"{released} {trial} {MOON}")]
    verdict = F.run(fc, released, sources, cfg=cfg)
    assert (verdict.value, verdict.leading) == ("out_of_scope", "not_standalone"), verdict.reasons
    verdict = F.run(fc, trial, sources, cfg=cfg)
    assert (verdict.value, verdict.leading) == ("out_of_scope", "subject_unresolved"), verdict.reasons
    verdict = F.run(fc, MOON, sources, cfg=cfg)
    assert verdict.value == "supported", verdict.reasons
    # A possessive or anaphoric opener refers elsewhere; a month is not a name.
    a_grave, e_grave = chr(0xE0), chr(0xE8)
    elsewhere = [
        ("Its capital is Oslo.", "Norway is a Nordic country. Its capital is Oslo.", "en", "not_standalone"),
        ("Sa capitale est Oslo.", f"La Norv{e_grave}ge est un pays nordique. Sa capitale est Oslo.", "fr",
         "not_standalone"),
        ("Another study found no effect in 2019.", "Another study found no effect in 2019.", "en",
         "not_standalone"),
        ("Both drugs were approved in 2019.", "Both drugs were approved in 2019.", "en", "not_standalone"),
        ("One of them was approved in 2019.", "One of them was approved in 2019.", "en", "not_standalone"),
        ("Such a device costs 500 euros in France.", "Such a device costs 500 euros in France.", "en",
         "not_standalone"),
        ("This is 95% effective.", "This is 95% effective.", "en", "not_standalone"),
        (f"C'est la capitale {a_grave} 400 km d'ici.", f"C'est la capitale {a_grave} 400 km d'ici.", "fr",
         "not_standalone"),
        ("The trial enrolled 300 patients in March.", "The trial enrolled 300 patients in March.", "en",
         "subject_unresolved"),
    ]
    for claim, text, lang, reason in elsewhere:
        verdict = F.run(fc, fc.scope.Claim(text=claim, lang=lang), [F.item(fc, "library:x", text, lang=lang)],
                        cfg=cfg)
        assert (verdict.value, verdict.leading) == ("out_of_scope", reason), (claim, verdict.reasons)
    named = "This COVID-19 vaccine is 95% effective."
    verdict = F.run(fc, named, [F.item(fc, "library:x", named)], cfg=cfg)
    assert verdict.value == "supported", (named, verdict.reasons)

    # c4 -- in assistant text a second-person decision is own; in the owner's
    # text the same sentence is a claim about the world.
    text = "You decided to hold the offsite in Berlin."
    by_assistant = fc.scope.Claim(text=text, lang="en", origin={"author": "assistant"})
    by_owner = fc.scope.Claim(text=text, lang="en", origin={"author": "user"})
    one = F.run(fc, by_assistant, [], cfg=cfg)
    other = F.run(fc, by_owner, [], cfg=cfg)
    assert one.value != "out_of_scope" and one.record["claim"]["kind"] == "own"
    assert other.record["claim"]["kind"] == "world"
    # About the world, a second-person subject names whoever wrote it.
    assert (other.value, other.leading) == ("out_of_scope", "not_standalone"), other.reasons
    board = "On 12 May 2026 the board agreed to the merger."
    verdict = F.run(fc, fc.scope.Claim(text=board, lang="en"), [F.item(fc, "library:board", board)], cfg=cfg)
    assert verdict.record["claim"]["kind"] == "world" and verdict.value == "supported", verdict.reasons


ANSWER = """# Travel notes

The **Moon** is 384,400 km away. See [the atlas](https://example.org/atlas) for the Loire.

- The `ferry` leaves Quiberon at noon every day.
- Belle-Ile has four towns and one harbour.

> Victor Hugo spent nineteen years in exile.

| Island | Towns |
| --- | --- |
| Belle-Ile | 4 |

```python
print("The Moon is far")
```

![Map of the island](map.png)
"""


def test_sq2_claims_from_text_reads_markdown_with_offsets_and_counts_the_rest(fc):
    cfg = F.config(fc)
    claims = fc.scope.claims_from_text(ANSWER, lang="en", origin={"author": "assistant"},
                                       max_claim_chars=cfg["limits"]["max_claim_chars"])
    verdicts = [F.run(fc, c, [], cfg=cfg) for c in claims]

    # c1 -- one claim per prose sentence, free of markup, offsets into the answer.
    prose = [(c, v) for c, v in zip(claims, verdicts) if v.value != "out_of_scope"]
    expected = [
        ("The Moon is 384,400 km away.", "The **Moon** is 384,400 km away."),
        ("See the atlas for the Loire.", "See [the atlas](https://example.org/atlas) for the Loire."),
        ("The ferry leaves Quiberon at noon every day.", "The `ferry` leaves Quiberon at noon every day."),
        ("Belle-Ile has four towns and one harbour.", "Belle-Ile has four towns and one harbour."),
        ("Victor Hugo spent nineteen years in exile.", "Victor Hugo spent nineteen years in exile."),
    ]
    assert [c.text for c, _ in prose] == [plain for plain, _ in expected]
    assert [ANSWER[c.start:c.end] for c, _ in prose] == [original for _, original in expected]
    assert [c.start for c in claims] == sorted(c.start for c in claims), "claims in answer order"

    # c2 -- headings, table rows, code and images come back out of scope, never
    # dropped, and the summary counts them by reason.
    counted = {}
    for verdict in verdicts:
        if verdict.value == "out_of_scope":
            counted[verdict.leading] = counted.get(verdict.leading, 0) + 1
    assert counted == {"heading": 1, "table_row": 2, "code": 1, "image": 1}, counted
    summary = fc.render.summarise(verdicts)
    for reason, n in counted.items():
        assert f"{fc.render.PLAIN_REASONS[reason]} {n}" in summary, (reason, summary)

    # c3 -- a blockquoted sentence carries "quoted"; a list item under a
    # heading keeps the heading in its window.
    quoted = [c for c, _ in prose if "quoted" in c.marks]
    assert [c.text for c in quoted] == ["Victor Hugo spent nineteen years in exile."]
    assert all("quoted" not in c.marks for c, _ in prose if c not in quoted)
    note = "## Travel\n\n- The ferry to Belle-Ile leaves Quiberon at noon every day.\n"
    verdict = F.run(fc, "The ferry to Belle-Ile leaves Quiberon at noon every day.",
                    [F.item(fc, "note:travel", note, kind="note", author="user")], cfg=cfg)
    assert verdict.value == "supported", verdict.reasons
    heading = verdict.record["spans"][0]["window"]["heading"]
    assert note[heading["start"]:heading["end"]] == "Travel", heading

    # c4 -- a raw HTML block gives markup_unparsed, never a claim.
    html = f"<div>\n{MOON}\n</div>\n\nThe Loire is the longest river in France.\n"
    claims = fc.scope.claims_from_text(html, lang="en", origin={}, max_claim_chars=600)
    verdicts = [F.run(fc, c, [], cfg=cfg) for c in claims]
    reasons = [v.leading for v in verdicts if v.value == "out_of_scope"]
    assert reasons.count("markup_unparsed") == 3, reasons
    checked = [c.text for c, v in zip(claims, verdicts) if v.value != "out_of_scope"]
    assert checked == ["The Loire is the longest river in France."], checked
    # Struck-through text is taken back: never a claim.
    struck = "~~The Moon is 500,000 km away.~~ The Moon is 384,400 km away. It is ~~not~~ far.\n"
    claims = fc.scope.claims_from_text(struck, lang="en", origin={}, max_claim_chars=600)
    verdicts = [F.run(fc, c, [], cfg=cfg) for c in claims]
    assert [(c.text, v.leading) for c, v in zip(claims, verdicts) if v.value == "out_of_scope"] == [
        ("The Moon is 500,000 km away.", "markup_unparsed"), ("It is not far.", "markup_unparsed")], claims
    assert [c.text for c, v in zip(claims, verdicts) if v.value != "out_of_scope"] == [MOON], claims


WALL = "The Great Wall of China stretches across northern China."


def test_wq1_the_context_window_qualifies_what_it_holds(fc):
    cfg = F.config(fc)
    a_grave, e_acute = chr(0xE0), chr(0xE9)

    def note(text, **fields):
        return [F.item(fc, "note:wall", text, kind="note", author="user", **fields)]

    # c1 -- a qualifying marker in the heading, the lead-in, the sentence
    # before or the sentence after.
    qualified = {
        "heading": (f"# Myths\n\n{WALL}\n", "myths"),
        "lead_in": (f"{a_grave} v{e_acute}rifier :\n- {WALL}\n", "a verifier"),
        "before": (f"The following statement is false. {WALL}\n", "false"),
        "after": (f"{WALL} This is a myth.\n", "myth"),
    }
    for where, (text, marker) in qualified.items():
        verdict = F.run(fc, WALL, note(text), cfg=cfg)
        assert verdict.value == "not_enough_evidence" and "context_qualified" in verdict.reasons, (
            where, verdict.reasons)
        detail = verdict.details["context_qualified"]
        assert (detail["where"], detail["marker"]) == (where, marker), detail
        blocked = [s for s in verdict.record["spans"] if s["source_id"] == "note:wall"]
        assert blocked and blocked[0]["window"][where] is not None, blocked

    myths = [
        ("heading path", f"# Myths\n\n## Space\n\n{WALL}\n", "context_qualified", "heading", "myths"),
        ("heading from the store", f"## Space\n\n{WALL}\n", "context_qualified", "heading", "myths"),
        ("bold label", f"**Myths**\n\n{WALL}\n", "context_qualified", "lead_in", "myths"),
        ("colon label", f"# Travel\n\nMyth:\n\n{WALL}\n", "context_qualified", "lead_in", "myth"),
        ("raw HTML heading", f"<h2>Myths</h2>\n\n{WALL}\n", "context_qualified", "lead_in", "myths"),
        ("parent item", f"# Travel notes\n\n- Myths:\n  - {WALL}\n", "context_qualified", "lead_in", "myths"),
        ("sibling item", f"Travel tips:\n- {WALL}\n- This is a myth.\n", "context_qualified", "after", "myth"),
        ("a denial after", f"{WALL} No, it is not.\n", "context_qualified", "after", "no"),
        ("not sure before", f"I am not sure about this. {WALL}\n", "context_qualified", "before", "not sure"),
        ("refuted after", f"{WALL} This has been refuted.\n", "context_qualified", "after", "refuted"),
        ("wrongly", f"# Things people wrongly believe\n\n{WALL}\n", "context_qualified", "heading", "wrongly"),
        ("falsehoods", f"# Common falsehoods\n\n{WALL}\n", "context_qualified", "heading", "falsehoods"),
        ("struck", f"# Travel\n\n~~{WALL}~~\n", "context_qualified", "sentence", "~~"),
        ("a negated lead-in", f"No longer valid:\n- {WALL}\n", "context_qualified", "lead_in", "no longer"),
        ("a condition", f"If the merger closes next year:\n- {WALL}\n", "conditional", "lead_in", "if"),
        ("a forecast", f"# Forecasts\n\n{WALL}\n", "evidence_hedged", "heading", "forecasts"),
        ("a scenario", f"# Scenario B\n\n{WALL}\n", "evidence_hedged", "heading", "scenario"),
        ("a population", f"Results of the 2019 study, in mice:\n- {WALL}\n", "population_narrower", "lead_in",
         "in mice"),
    ]
    for label, text, reason, where, marker in myths:
        fields = {"chunk_fields": {"locator": {"heading": "Myths"}}} if label == "heading from the store" else {}
        verdict = F.run(fc, WALL, note(text, **fields), cfg=cfg)
        assert verdict.value == "not_enough_evidence" and reason in verdict.reasons, (label, verdict.reasons)
        detail = verdict.details[reason]
        assert (detail["where"], detail["marker"]) == (where, marker), (label, detail)
        blocked = [s for s in verdict.record["spans"] if s["source_id"] == "note:wall"]
        if where != "sentence":
            assert blocked and blocked[0]["window"][where] is not None, (label, blocked)
    library = F.item(fc, "library:wall", f"# Myths\n\n{WALL}\n")
    verdict = F.run(fc, WALL, [library], cfg=cfg)
    assert "context_qualified" in verdict.reasons and verdict.details["context_qualified"]["where"] == "heading"
    french = fc.scope.Claim(text="Le Rhin se jette dans la mer du Nord.", lang="fr")
    for text in (f"Ce qui n'est plus vrai :\n- {french.text}\n", f"{french.text} Non, pas du tout.\n"):
        verdict = F.run(fc, french, note(text, lang="fr"), cfg=cfg)
        assert "context_qualified" in verdict.reasons, (text, verdict.reasons)

    # c2 -- witness: under a neutral heading the same sentence is supported.
    verdict = F.run(fc, WALL, note(f"# Travel in Asia\n\n{WALL}\n"), cfg=cfg)
    assert (verdict.value, verdict.leading) == ("supported", "verbatim_sentence"), verdict.reasons
    # A frame the claim itself carries does not block it; a month is not a hedge.
    hedged_claim = "The Great Wall of China may be visible from space."
    verdict = F.run(fc, hedged_claim, note(f"Possible answers:\n- {hedged_claim}\n"), cfg=cfg)
    assert verdict.value == "supported", verdict.reasons
    verdict = F.run(fc, WALL, note(f"# May 2026 trip\n\n{WALL}\n"), cfg=cfg)
    assert verdict.value == "supported", verdict.reasons

    # c3 -- a list item with no lead-in and no heading; a first sentence that
    # opens in lowercase.
    for text in (f"- {WALL}\n", WALL[0].lower() + WALL[1:] + " It took centuries.\n"):
        verdict = F.run(fc, WALL, note(text), cfg=cfg)
        assert verdict.value == "not_enough_evidence" and "context_incomplete" in verdict.reasons, (
            text, verdict.reasons)
    # A chunk of a longer source may start or end inside a sentence; a period
    # after a short capitalised word may not end the sentence before.
    nasa = "NASA faked the Moon landing in 1969."
    cuts = {
        "chunk_start": [F.item(fc, "library:nasa", ["Nobody seriously believes that", f"{nasa} Next topic."])],
        "chunk_end": [F.item(fc, "library:nasa", [f"Old notes. {nasa[:-1]}", ", said no serious historian."])],
        "uncertain_period": [F.item(fc, "library:nasa", f"Critics say Pvt. {nasa}")],
    }
    for cause, items in cuts.items():
        verdict = F.run(fc, nasa, items, cfg=cfg)
        assert verdict.value == "not_enough_evidence" and "context_incomplete" in verdict.reasons, (
            cause, verdict.reasons)
        assert verdict.details["context_incomplete"]["cause"] == cause, (cause, verdict.details)
    # Witness: a chunk the store says starts its source, or a source of one chunk, is whole.
    known = [F.item(fc, "library:nasa", [f"{nasa} It was filmed.", "Other notes follow."],
                    chunk_fields={"offset_in_source": 0})]
    assert F.run(fc, nasa, known, cfg=cfg).value == "supported"
    assert F.run(fc, nasa, [F.item(fc, "library:nasa", f"{nasa} Next topic.")], cfg=cfg).value == "supported"

    # c4 -- after an attribution lead-in.
    claim = "The Pixel 9 battery lasts ten hours on a single charge."
    for lead in ("According to the vendor:", "Selon le fournisseur :"):
        verdict = F.run(fc, claim, note(f"{lead}\n- {claim}\n"), cfg=cfg)
        assert verdict.value == "not_enough_evidence" and "attributed" in verdict.reasons, (
            lead, verdict.reasons)
    attributed = [
        (f"# According to the vendor\n\n{claim}\n", "heading"),
        (f"The vendor claims the following. {claim}\n", "before"),
        (f"{claim} That is what the vendor says.\n", "after"),
        (f"# Specs\n\n> {claim}\n", "quoted"),
    ]
    for text, where in attributed:
        verdict = F.run(fc, claim, note(text), cfg=cfg)
        assert "attributed" in verdict.reasons and verdict.details["attributed"]["where"] == where, (
            text, verdict.reasons, verdict.details)
