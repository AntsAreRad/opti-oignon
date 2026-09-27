"""The canary: planted errors and positive controls, run at every construction of a checker.

A fixed set of claims, each with its evidence, its date, and the verdict it
must receive, in English and in French: planted errors that must never come out
supported, and positive controls that must. It runs through the very check
function that answers the owner. If any item comes out otherwise, the checker
is built refused, and every call names the failing items instead of answering.
Its digest and outcome are in every record.

What it is not: a measure of power. It is written together with the rules it
tests, so it proves the verifier able to refuse on the classes it holds, and no
more; what lies beyond it is measured elsewhere, on data the rules were not
written on.

Which rule each row reaches: every row goes through the whole check, so a row
proves a refusal on the check's own path. The check only ever locates a whole
sentence of the chunk's own view, so the passage rows prove refusal at
admission (a changed hash, a text not NFC) and by sentence equality (one
character off, a quote cut inside a ligature, a sentence split across two
chunks); the alignment rule of the passage search, which the check's path
never reaches, is proven on the passage search itself by its own contract.

Every character outside ASCII is built with ``chr()``.
"""

import dataclasses
import hashlib
from dataclasses import dataclass

from . import record, scope, sources
from . import vocabulary as V

checkpoint_before_apply = True

CANARY_VERSION = 1
CLASSES = ("passage", "restatement", "provenance", "time", "scope", "context", "person", "claim_only")
AS_OF = "2026-09-26"
READ_ON = "2026-09-27"
GRANT = "canary-grant"

_E_ACUTE, _E_GRAVE, _A_GRAVE, _O_CIRC = chr(0xE9), chr(0xE8), chr(0xE0), chr(0xF4)
_RSQUO, _NBSP, _FFI, _ACUTE = chr(0x2019), chr(0xA0), chr(0xFB03), chr(0x301)

MOON = "The Moon orbits the Earth at an average distance of 384,400 km."
BRIDGE = "The Pont Neuf was completed in 1607 under Henri IV."
RHINE = "The Rhine flows into the North Sea near Rotterdam."
WALL = "The Great Wall of China stretches across the north of China."
PIXEL = "The Pixel 9 battery lasts ten hours on a single charge."
BERLIN = "We decided to hold the offsite in Berlin."
OSLO = "We decided to hold the offsite in Oslo."
CEO = "The current CEO of Acme is Jane Roe."
EIFFEL = f"La Tour Eiffel a {_E_ACUTE}t{_E_ACUTE} inaugur{_E_ACUTE}e en 1889 {_A_GRAVE} Paris."
RHIN = f"Le Rhin se jette dans la mer du Nord pr{_E_GRAVE}s de Rotterdam."
MURAILLE = "La Grande Muraille de Chine traverse le nord de la Chine."
NOUS_BERLIN = f"Nous avons d{_E_ACUTE}cid{_E_ACUTE} de faire le s{_E_ACUTE}minaire {_A_GRAVE} Berlin."
NOUS_OSLO = f"Nous avons d{_E_ACUTE}cid{_E_ACUTE} de faire le s{_E_ACUTE}minaire {_A_GRAVE} Oslo."
FIREFOX = f"La derni{_E_GRAVE}re version de Firefox est la 131."
MANNING = "Manning leaked the files in 2010."
NASA = "NASA faked the Moon landing in 1969."
GLOBEX = "Acme owns Globex."


@dataclass(frozen=True)
class Row:
    """One canary item and the verdict it must receive."""

    name: str
    cls: str
    lang: str
    role: str
    claim: object
    items: tuple
    expected_value: str
    expected_reason: object = None
    as_of: str = AS_OF
    read_on: str = READ_ON


@dataclass(frozen=True)
class CanaryResult:
    """What a run found."""

    outcome: str
    n: int
    digest: str
    failing: tuple
    results: tuple

    def summary(self):
        return {"digest": self.digest, "n": self.n, "outcome": self.outcome}


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk(text, **fields):
    digest = fields.pop("sha256", None) or _sha(text)
    return sources.Chunk(text=text, sha256=digest, **fields)


def _item(source_id, texts, *, kind="library", author="third_party", chunk_fields=None, **fields):
    if isinstance(texts, str):
        texts = [texts]
    fields.setdefault("consent", GRANT)
    fields.setdefault("recorded_at", "2026-01-10")
    fields.setdefault("source_date", "2025-11-02")
    chunks = tuple(_chunk(text, **dict(chunk_fields or {})) for text in texts)
    return sources.SourceItem(source_id=source_id, kind=kind, author=author, chunks=chunks, **fields)


def _note(source_id, text, **fields):
    return _item(source_id, text, kind="note", author="user", **fields)


def _ledger(prefix, old, new, *, lang="en", y_recorded="2026-07-10", x_until=""):
    """A decision replaced as the drift ledger records it: no end date, a link to the successor."""
    return (
        _item(f"ledger:{prefix}-x", old, kind="ledger", author="user", superseded_by=f"ledger:{prefix}-y",
              recorded_at="2026-06-02", source_date=None, lang=lang, valid_until=x_until),
        _item(f"ledger:{prefix}-y", new, kind="ledger", author="user", recorded_at=y_recorded,
              source_date=None, lang=lang),
    )


def _planted(name, cls, lang, claim, items, value=V.NOT_ENOUGH_EVIDENCE, reason=None, **fields):
    return Row(name, cls, lang, "planted", claim, tuple(items), value, reason, **fields)


def _control(name, cls, lang, claim, items, **fields):
    return Row(name, cls, lang, "control", claim, tuple(items), V.SUPPORTED, "verbatim_sentence", **fields)


def rows():
    """Every canary item, in a fixed order."""
    fr = scope.Claim
    out = []
    # passage
    out += [
        _control("passage_exact", "passage", "en", MOON, [_item("library:moon", f"Astronomy basics. {MOON}")]),
        _planted("passage_one_character_off", "passage", "en", MOON,
                 [_item("library:moon", MOON.replace("Earth", "Earht"))], reason="no_judge"),
        _planted("passage_changed_under_old_hash", "passage", "en", MOON,
                 [_item("library:moon", MOON, chunk_fields={"sha256": _sha(MOON.replace("384,400", "384,401"))})],
                 reason="no_admissible_source"),
        _planted("passage_split_across_chunks", "passage", "en", MOON,
                 [_item("library:moon", ["Astronomy basics. The Moon orbits the Earth at an average",
                                         "distance of 384,400 km. Tides follow it."])], reason="no_judge"),
        _planted("passage_not_nfc", "passage", "en",
                 f"The Mus{_E_ACUTE}e d'Orsay opened to the public in December 1986.",
                 [_item("library:orsay", f"The Muse{_ACUTE} d'Orsay opened to the public in December 1986.")],
                 reason="no_admissible_source"),
        _planted("passage_inside_ligature", "passage", "en", "fice opens at nine on weekdays.",
                 [_item("library:office", f"Our o{_FFI}ce opens at nine on weekdays.")], reason="no_judge"),
        _control("passage_exact_fr", "passage", "fr", fr(EIFFEL, lang="fr"),
                 [_item("library:eiffel", f"Monuments. {EIFFEL}", lang="fr")]),
        _planted("passage_one_character_off_fr", "passage", "fr", fr(EIFFEL, lang="fr"),
                 [_item("library:eiffel", EIFFEL.replace("inaugur" + _E_ACUTE + "e", "inaugur" + _E_ACUTE + "s"),
                        lang="fr")], reason="no_judge"),
    ]
    # restatement
    typographic = f"The Loire{_RSQUO}s estuary lies west of{_NBSP}Nantes."
    hotel = f"L{_RSQUO}H{_O_CIRC}tel de Ville de Lyon date du XVIIe si{_E_GRAVE}cle."
    out += [
        _planted("restatement_inside_a_longer_sentence", "restatement", "en", MOON,
                 [_item("library:moon", "It is false that t" + MOON[1:])], reason="no_judge"),
        _planted("restatement_negation_moved", "restatement", "en", "Version 2 is supported, version 3 is not.",
                 [_item("library:versions", "Version 2 is not supported, version 3 is.")], reason="no_judge"),
        _planted("restatement_one_digit", "restatement", "en", BRIDGE.replace("1607", "1608"),
                 [_item("library:bridge", BRIDGE)], reason="no_judge"),
        _planted("restatement_one_word_added", "restatement", "en", BRIDGE.replace("was", "was finally"),
                 [_item("library:bridge", BRIDGE)], reason="no_judge"),
        _control("restatement_typographic_after_a_wrapper", "restatement", "en",
                 "I think the Loire's estuary lies west of Nantes.", [_item("library:loire", typographic)]),
        _planted("restatement_inside_a_longer_sentence_fr", "restatement", "fr", fr(EIFFEL, lang="fr"),
                 [_item("library:eiffel", "Il est faux que l" + EIFFEL[1:], lang="fr")], reason="no_judge"),
        _planted("restatement_one_digit_fr", "restatement", "fr", fr(EIFFEL.replace("1889", "1890"), lang="fr"),
                 [_item("library:eiffel", EIFFEL, lang="fr")], reason="no_judge"),
        _planted("restatement_after_an_initial", "restatement", "en", "Bush won the popular vote in 2000.",
                 [_item("library:vote", "Many wrongly believe that George W. Bush won the popular vote in 2000.")],
                 reason="no_judge"),
        _planted("restatement_after_an_abbreviation_fr", "restatement", "fr",
                 fr(f"Raoult a gu{_E_ACUTE}ri 100 patients.", lang="fr"),
                 [_item("library:raoult", f"Rien ne prouve que le Pr. Raoult a gu{_E_ACUTE}ri 100 patients.",
                        lang="fr")], reason="no_judge"),
        _planted("restatement_under_the_floor", "restatement", "en", "Tea is hot.",
                 [_item("library:tea", "Tea is hot. It always is.")], reason="quote_not_found"),
        _control("restatement_without_final_period", "restatement", "en", MOON[:-1],
                 [_item("library:moon", f"Astronomy basics. {MOON}")]),
        _control("restatement_typographic_after_a_wrapper_fr", "restatement", "fr",
                 fr("Je pense que l'H" + _O_CIRC + "tel de Ville de Lyon date du XVIIe si" + _E_GRAVE + "cle.",
                    lang="fr"), [_item("library:lyon", hotel, lang="fr")]),
    ]
    # provenance
    ocr = {"extractor": "pdf-text", "extractor_version": "1.0", "flags": ("ocr",)}
    length = "The Rhine is 1,230 km long from its source to its mouth."
    out += [
        _planted("provenance_model_author", "provenance", "en", RHINE,
                 [_item("library:rhine", RHINE, author="model")], reason="no_admissible_source"),
        _planted("provenance_unknown_author", "provenance", "en", RHINE,
                 [_item("library:rhine", RHINE, author="unknown")], reason="no_admissible_source"),
        _planted("provenance_model_quoted", "provenance", "en", RHINE,
                 [_note("note:rhine", f"From the chat. {RHINE}",
                        chunk_fields={"model_quoted": ((15, 15 + len(RHINE)),)})], reason="no_admissible_source"),
        _planted("provenance_snippet", "provenance", "en", RHINE,
                 [_item("web:rhine", RHINE, kind="snippet")], reason="no_admissible_source"),
        _planted("provenance_no_consent", "provenance", "en", RHINE,
                 [_item("library:rhine", RHINE, consent=None)], reason="no_admissible_source"),
        _planted("provenance_retracted", "provenance", "en", RHINE,
                 [_item("library:rhine", RHINE, flags={"retracted": "2025-05-01"})],
                 reason="no_admissible_source"),
        _planted("provenance_ingested_before_correction", "provenance", "en", RHINE,
                 [_item("library:rhine", RHINE, flags={"corrected": "2026-03-01"},
                        chunk_fields={"ingested_at": "2026-01-15"})], reason="ingested_before_correction"),
        _planted("provenance_numeral_from_ocr", "provenance", "en", length,
                 [_item("library:rhine", length, chunk_fields={"extraction": ocr})],
                 reason="extraction_uncertain"),
        _planted("provenance_cites_retracted", "provenance", "en", RHINE,
                 [_item("library:rhine", RHINE, flags={"cites_retracted": ""})], reason="cites_retracted"),
        _planted("provenance_own_claim_in_a_library_only", "provenance", "en", BERLIN,
                 [_item("library:plan", BERLIN)], reason="no_admissible_source"),
        _control("provenance_owner_note", "provenance", "en", RHINE, [_note("note:rhine", RHINE)]),
        _control("provenance_dated_library", "provenance", "en", RHINE,
                 [_item("library:rhine", RHINE, source_date="2024-04-10")]),
        _planted("provenance_model_author_fr", "provenance", "fr", fr(RHIN, lang="fr"),
                 [_item("library:rhin", RHIN, author="model", lang="fr")], reason="no_admissible_source"),
        _control("provenance_owner_note_fr", "provenance", "fr", fr(RHIN, lang="fr"),
                 [_note("note:rhin", RHIN, lang="fr")]),
    ]
    # time
    out += [
        _planted("time_past_decision_replaced", "time", "en", BERLIN, _ledger("d", BERLIN, OSLO),
                 value=V.CONFLICTING, reason="superseded"),
        _planted("time_present_state_replaced", "time", "en", "The offsite is in Berlin.",
                 _ledger("s", "The offsite is in Berlin.", "The offsite is in Oslo."), reason="no_longer_held"),
        _planted("time_expired", "time", "en", RHINE,
                 [_item("library:rhine", RHINE, valid_from="2025-01-01", valid_until="2026-02-01")],
                 reason="expired"),
        _planted("time_valid_only_later", "time", "en", RHINE,
                 [_item("library:rhine", RHINE, valid_from="2026-10-01")], reason="no_valid_source"),
        _planted("time_present_claim_undated_source", "time", "en", CEO,
                 [_item("library:acme", CEO, source_date=None)], reason="source_undated"),
        _planted("time_successor_missing", "time", "en", BERLIN,
                 [_item("ledger:d-x", BERLIN, kind="ledger", author="user", superseded_by="ledger:d-gone",
                        recorded_at="2026-06-02", source_date=None)], reason="no_admissible_source"),
        _planted("time_successor_undated", "time", "en", BERLIN, _ledger("u", BERLIN, OSLO, y_recorded=""),
                 value=V.CONFLICTING, reason="superseded", as_of="2026-06-15"),
        _planted("time_explicit_end_past_the_replacement", "time", "en", BERLIN,
                 _ledger("e", BERLIN, OSLO, x_until="2026-12-31"), value=V.CONFLICTING, reason="superseded"),
        _planted("time_successor_dated_before_it", "time", "en", BERLIN,
                 _ledger("b", BERLIN, OSLO, y_recorded="2026-05-01"), reason="no_valid_source"),
        _planted("time_present_claim_source_dated_later", "time", "en", CEO,
                 [_item("library:acme", CEO, recorded_at="2025-03-01", source_date="2026-09-01")],
                 reason="no_valid_source", as_of="2026-01-01"),
        _planted("time_present_decision_replaced", "time", "en", "We decide to hold the offsite in Berlin.",
                 _ledger("p", "We decide to hold the offsite in Berlin.", "We decide to hold the offsite in Oslo."),
                 reason="no_longer_held"),
        _planted("time_dated_past_decision_replaced", "time", "en",
                 "We decided in June 2026 to hold the offsite in Berlin.",
                 _ledger("j", "We decided in June 2026 to hold the offsite in Berlin.", OSLO),
                 reason="no_longer_held"),
        _control("time_decision_at_its_date", "time", "en", BERLIN, _ledger("d", BERLIN, OSLO),
                 as_of="2026-06-15"),
        _control("time_note_link_ignored", "time", "en", RHINE,
                 [_note("note:rhine", RHINE, superseded_by="note:rhine-2", recorded_at="2026-01-10"),
                  _note("note:rhine-2", "The Rhine is a river of western Europe.", recorded_at="2026-02-01")]),
        _control("time_timeless_claim_undated_source", "time", "en", MOON,
                 [_item("library:moon", MOON, source_date=None)]),
        _planted("time_past_decision_replaced_fr", "time", "fr", fr(NOUS_BERLIN, lang="fr"),
                 _ledger("f", NOUS_BERLIN, NOUS_OSLO, lang="fr"), value=V.CONFLICTING, reason="superseded"),
        _planted("time_present_claim_undated_source_fr", "time", "fr", fr(FIREFOX, lang="fr"),
                 [_item("library:firefox", FIREFOX, source_date=None, lang="fr")], reason="source_undated"),
    ]
    # scope
    released = "It was first released in 1991."
    trial = "The trial enrolled 300 patients."
    publie = f"Il a {_E_ACUTE}t{_E_ACUTE} publi{_E_ACUTE} en 1991."
    out += [
        _planted("scope_pronoun_subject", "scope", "en", released, [_item("library:x", released)],
                 value=V.OUT_OF_SCOPE, reason="not_standalone"),
        _planted("scope_definite_subject_without_a_name", "scope", "en", trial, [_item("library:x", trial)],
                 value=V.OUT_OF_SCOPE, reason="subject_unresolved"),
        _planted("scope_question", "scope", "en", "Is the Moon 384,400 km away from the Earth?",
                 [_item("library:moon", MOON)], value=V.OUT_OF_SCOPE, reason="question"),
        _planted("scope_heading", "scope", "en", f"# {MOON}", [_item("library:moon", MOON)],
                 value=V.OUT_OF_SCOPE, reason="heading"),
        _planted("scope_table_row", "scope", "en", "| Moon | 384,400 km |\n| --- | --- |",
                 [_item("library:moon", MOON)], value=V.OUT_OF_SCOPE, reason="table_row"),
        _planted("scope_possessive_subject", "scope", "en", "Its capital is Oslo.",
                 [_item("library:norway", "Norway is a Nordic country. Its capital is Oslo.")],
                 value=V.OUT_OF_SCOPE, reason="not_standalone"),
        _planted("scope_month_is_not_a_name", "scope", "en", "The trial enrolled 300 patients in March.",
                 [_item("library:x", "The trial enrolled 300 patients in March.")], value=V.OUT_OF_SCOPE,
                 reason="subject_unresolved"),
        _planted("scope_first_person_about_the_world", "scope", "en", "We live in Paris.",
                 [_item("library:x", "We live in Paris.")], value=V.OUT_OF_SCOPE, reason="not_standalone"),
        _planted("scope_possessive_subject_fr", "scope", "fr", fr("Sa capitale est Oslo.", lang="fr"),
                 [_item("library:norvege", f"La Norv{_E_GRAVE}ge est un pays nordique. Sa capitale est Oslo.",
                        lang="fr")], value=V.OUT_OF_SCOPE, reason="not_standalone"),
        _control("scope_subject_named", "scope", "en", "The HALT trial enrolled 300 patients.",
                 [_item("library:halt", "The HALT trial enrolled 300 patients.")]),
        _control("scope_opinion_stripped", "scope", "en", f"I think the Rhine{RHINE[9:]}",
                 [_item("library:rhine", RHINE)]),
        _planted("scope_pronoun_subject_fr", "scope", "fr", fr(publie, lang="fr"),
                 [_item("library:x", publie, lang="fr")], value=V.OUT_OF_SCOPE, reason="not_standalone"),
        _control("scope_opinion_stripped_fr", "scope", "fr", fr("Je pense que le" + RHIN[2:], lang="fr"),
                 [_item("library:rhin", RHIN, lang="fr")]),
    ]
    # context
    a_verifier = f"{chr(0xC0)} v{_E_ACUTE}rifier :"
    out += [
        _planted("context_heading_myths", "context", "en", WALL, [_note("note:wall", f"# Myths\n\n{WALL}\n")],
                 reason="context_qualified"),
        _planted("context_after_a_statement_called_false", "context", "en", WALL,
                 [_note("note:wall", f"The following statement is false. {WALL}\n")], reason="context_qualified"),
        _planted("context_before_this_is_a_myth", "context", "en", WALL,
                 [_note("note:wall", f"{WALL} This is a myth.\n")], reason="context_qualified"),
        _planted("context_list_item_without_lead_in", "context", "en", WALL,
                 [_note("note:wall", f"- {WALL}\n")], reason="context_incomplete"),
        _planted("context_according_to_the_vendor", "context", "en", PIXEL,
                 [_note("note:pixel", f"According to the vendor:\n- {PIXEL}\n")], reason="attributed"),
        _planted("context_heading_path", "context", "en", WALL,
                 [_note("note:wall", f"# Myths\n\n## Space\n\n{WALL}\n")], reason="context_qualified"),
        _planted("context_label_paragraph", "context", "en", WALL,
                 [_note("note:wall", f"# Travel\n\nMyth:\n\n{WALL}\n")], reason="context_qualified"),
        _planted("context_parent_item", "context", "en", WALL,
                 [_note("note:wall", f"# Travel notes\n\n- Myths:\n  - {WALL}\n")], reason="context_qualified"),
        _planted("context_sibling_item", "context", "en", WALL,
                 [_note("note:wall", f"Travel tips:\n- {WALL}\n- This is a myth.\n")], reason="context_qualified"),
        _planted("context_heading_in_a_library_item", "context", "en", WALL,
                 [_item("library:wall", f"# Myths\n\n{WALL}\n")], reason="context_qualified"),
        _planted("context_struck_sentence", "context", "en", WALL,
                 [_note("note:wall", f"# Travel\n\n~~{WALL}~~\n")], reason="context_qualified"),
        _planted("context_struck_negation", "context", "en", "The Great Wall of China is not visible from space.",
                 [_note("note:wall", "# Travel\n\nThe Great Wall of China is ~~not~~ visible from space.\n")],
                 reason="context_qualified"),
        _planted("context_attribution_in_the_sentence_before", "context", "en", PIXEL,
                 [_note("note:pixel", f"The vendor claims the following. {PIXEL}\n")], reason="attributed"),
        _planted("context_blockquote", "context", "en", PIXEL,
                 [_note("note:pixel", f"# Specs\n\n> {PIXEL}\n")], reason="attributed"),
        _planted("context_cut_continuation", "context", "en", WALL,
                 [_item("library:wall", f"t{WALL[1:]} It took centuries.")], reason="context_incomplete"),
        _planted("context_chunk_may_start_inside_a_sentence", "context", "en", NASA,
                 [_item("library:nasa", ["Nobody seriously believes that", f"{NASA} Next topic."])],
                 reason="context_incomplete"),
        _planted("context_uncertain_period", "context", "en", MANNING,
                 [_item("library:leak", f"Critics say Pvt. {MANNING}")], reason="context_incomplete"),
        _planted("context_conditional_lead_in", "context", "en", GLOBEX,
                 [_note("note:acme", f"If the merger closes next year:\n- {GLOBEX}\n")], reason="conditional"),
        _planted("context_forecast_heading", "context", "en", "Acme sells 3 million cars in 2030.",
                 [_note("note:acme", "# Forecasts\n\nAcme sells 3 million cars in 2030.\n")],
                 reason="evidence_hedged"),
        _planted("context_population_lead_in", "context", "en", "Metformin cures pancreatic cancer.",
                 [_note("note:metformin", "Results of the 2019 study, in mice:\n- Metformin cures pancreatic cancer.\n")],
                 reason="population_narrower"),
        _control("context_neutral_heading", "context", "en", WALL,
                 [_note("note:wall", f"# Travel in Asia\n\n{WALL}\n")]),
        _control("context_chunk_at_its_source_start", "context", "en", NASA,
                 [_item("library:nasa", [f"{NASA} It is filmed.", "Other notes follow."],
                        chunk_fields={"offset_in_source": 0})]),
        _planted("context_lead_in_a_verifier_fr", "context", "fr", fr(MURAILLE, lang="fr"),
                 [_note("note:muraille", f"{a_verifier}\n- {MURAILLE}\n", lang="fr")], reason="context_qualified"),
        _planted("context_selon_le_fournisseur_fr", "context", "fr",
                 fr("La batterie du Pixel 9 tient dix heures.", lang="fr"),
                 [_note("note:pixel", "Selon le fournisseur :\n- La batterie du Pixel 9 tient dix heures.\n",
                        lang="fr")], reason="attributed"),
        _planted("context_negated_lead_in_fr", "context", "fr", fr(RHIN, lang="fr"),
                 [_note("note:rhin", f"Ce qui n'est plus vrai :\n- {RHIN}\n", lang="fr")], reason="context_qualified"),
        _planted("context_denial_after_fr", "context", "fr", fr(MURAILLE, lang="fr"),
                 [_note("note:muraille", f"# Voyage\n\n{MURAILLE} Non, pas du tout.\n", lang="fr")],
                 reason="context_qualified"),
        _control("context_neutral_heading_fr", "context", "fr", fr(MURAILLE, lang="fr"),
                 [_note("note:muraille", f"# Voyage en Chine\n\n{MURAILLE}\n", lang="fr")]),
    ]
    # person
    lisbon = "We decided to hold the offsite in Lisbon."
    lisbonne = f"Nous avons d{_E_ACUTE}cid{_E_ACUTE} de faire le s{_E_ACUTE}minaire {_A_GRAVE} Lisbonne."
    out += [
        _control("person_you_against_we", "person", "en",
                 scope.Claim("You decided to hold the offsite in Lisbon.", lang="en", origin={"author": "assistant"}),
                 [_item("ledger:l", lisbon, kind="ledger", author="user", source_date=None)]),
        _control("person_tu_against_nous_fr", "person", "fr",
                 scope.Claim(f"Tu as d{_E_ACUTE}cid{_E_ACUTE} de faire le s{_E_ACUTE}minaire {_A_GRAVE} Lisbonne.",
                             lang="fr", origin={"author": "assistant"}),
                 [_item("ledger:l", lisbonne, kind="ledger", author="user", source_date=None, lang="fr")]),
    ]
    out += [
        _planted("person_owner_note_in_the_second_person", "person", "en", lisbon,
                 [_note("note:plan", "You decided to hold the offsite in Lisbon.")], reason="no_judge"),
        _planted("person_assistant_first_person", "person", "en",
                 scope.Claim("I decided to write the script in Python.", lang="en", origin={"author": "assistant"}),
                 [_item("ledger:py", "I decided to write the script in Python.", kind="ledger", author="user",
                        source_date=None)], reason="no_judge"),
        _planted("person_possessive_swapped", "person", "en", "I decided to sell my car.",
                 [_note("note:car", "I decided to sell your car.")], reason="no_judge"),
        _planted("person_object_pronoun_fr", "person", "fr",
                 scope.Claim(f"Vous avez d{_E_ACUTE}cid{_E_ACUTE} de nous payer 500 euros.", lang="fr",
                             origin={"author": "assistant"}),
                 [_item("ledger:pay", f"Nous avons d{_E_ACUTE}cid{_E_ACUTE} de vous payer 500 euros.",
                        kind="ledger", author="user", source_date=None, lang="fr")], reason="no_judge"),
        _control("person_english_on_is_a_preposition", "person", "en",
                 scope.Claim("On 12 May 2026 the board agreed to the merger.", lang="en"),
                 [_item("library:board", "On 12 May 2026 the board agreed to the merger.")]),
    ]
    # claim only
    out += [
        _planted("claim_only_irrelevant_passage", "claim_only", "en", MOON, [_item("library:rhine", RHINE)],
                 reason="no_judge"),
        _planted("claim_only_passage_blanked", "claim_only", "en", MOON, [_item("library:moon", "")],
                 reason="no_judge"),
        _planted("claim_only_irrelevant_passage_fr", "claim_only", "fr", fr(EIFFEL, lang="fr"),
                 [_item("library:rhin", RHIN, lang="fr")], reason="no_judge"),
    ]
    return out


def _plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def digest(items=None):
    """The digest of the canary's items: any change to one changes it."""
    items = rows() if items is None else items
    return record.digest({"version": CANARY_VERSION, "rows": [_plain(row) for row in items]})


def run(check, judges=(), *, config):
    """Every item through ``check``; the outcome is "pass" only if every item came out as planted."""
    if judges:
        raise ValueError("this core has no judge seam; the canary runs without judges")
    items = rows()
    summary = {"digest": digest(items), "n": len(items), "outcome": "running"}
    failing = []
    results = []
    for row in items:
        try:
            verdict = check(row.claim, list(row.items), as_of=row.as_of, read_on=row.read_on, config=config,
                            canary=summary)
        except Exception:  # noqa: BLE001 - an item that raises is a failing item, named
            failing.append(row.name)
            results.append((row, None))
            continue
        ok = isinstance(verdict, V.Verdict) and verdict.value == row.expected_value
        if ok and row.expected_reason:
            ok = row.expected_reason in verdict.reasons
        if row.role == "planted" and getattr(verdict, "value", None) == V.SUPPORTED:
            ok = False
        if row.role == "control" and getattr(verdict, "value", None) != V.SUPPORTED:
            ok = False
        if not ok:
            failing.append(row.name)
        results.append((row, verdict))
    outcome = "pass" if not failing else "fail"
    return CanaryResult(outcome, len(items), summary["digest"], tuple(failing), tuple(results))
