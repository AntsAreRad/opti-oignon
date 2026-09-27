#!/usr/bin/env python3
"""A verdict is "supported" only when the reply opens with SUPPORTED and negates nothing.

The claim verifier behind ``POST /api/claims/verify`` and the answer and
citation routes maps a model's free-text reply to supported, unsupported or
uncertain. It used to promote a lead that merely contained "confirm",
"consistent with" or "corroborat" anywhere, so "The source does not confirm
the claim." and "This is not consistent with the source." were read as
support. A word list of negations over those substrings would still leak
("disconfirms", "doesnt", a typographic apostrophe, "Hardly consistent"), so
the root cause goes: the lead must open with the verdict word the instruction
asks for -- markdown emphasis, a quote marker or a "verdict:" label may come
before it, a question mark may not follow it -- and it must carry no negation
cue, English or French, whatever the apostrophe, contracted with or without
one, or carried by a negating prefix on a word of support.

Everything else falls to uncertain, or to unsupported when the body says so.
The rule is strict on purpose: the module's doctrine is that an indeterminate
verification never asserts support, and its cost is pinned below so that a
later relaxation has to supersede that clause by name.

  * VD1 -- the negated and hedged phrasings are never supported, the plain
    SUPPORTED phrasings still are, the verifier and the per-answer aggregate
    the routes use agree, and the stated cost holds. The instruction asks
    for the verdict word first and a reason after it, often on the next
    line: a negation or a hedge on any line of the reply keeps it from
    support (c5), and so do the cues a review found the first word list
    missing -- a negating prefix on more words of support, refusals and
    contradictions in plain words, a hyphen typed another way or hidden, a
    word mixing scripts (c6).

Loaded through the shared isolation window; the model is a scripted callable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_vd1_a_verdict_is_supported_only_when_it_opens_with_supported_and_negates_nothing": 2.0,
}

_WRAPPER = "opti_oignon.agent.untrusted_context"
_VERIFY = "opti_oignon.agent.claim_verification"
_AGGREGATE = "opti_oignon.agent.claim_aggregation"

_CLAIM = "Water boils at 100C at sea level."
_SOURCE = "The reference notes that water boils at 100 degrees Celsius at sea level."

# Non-ASCII characters are built, never typed.
_RIGHT_QUOTE = chr(0x2019)
_MODIFIER_APOSTROPHE = chr(0x02BC)
_E_ACUTE = chr(0xE9)

_NEVER_SUPPORTED = (
    # measured at supported before this contract
    "The source does not confirm the claim.",
    "This is not consistent with the source.",
    "Nothing in the source corroborates this.",
    "The claim is inconsistent with the source.",
    "The claim is unconfirmed by the source.",
    "La source ne confirme pas cette affirmation.",
    "No, the source never confirms it.",
    # contractions, apostrophes, failures and refusals
    "The source doesn't confirm it.",
    "The source doesn" + _RIGHT_QUOTE + "t confirm it.",
    "The source fails to confirm the claim.",
    "No passage supports the claim.",
    "SUPPORTED? No.",
    "Rien dans la source ne confirme cela.",
    "Non confirm" + _E_ACUTE + " par la source.",
    # what a negation word list alone would still have promoted
    "The source disconfirms the claim.",
    "Unable to confirm the claim from the source.",
    "The source doesnt confirm it.",
    "The claim is un-corroborated by the source.",
    "The claim is in-consistent with the source.",
    "Hardly consistent with the source.",
    "The claim is false, though the date is consistent with the source.",
    "Supported? Hard to say; the source is silent.",
    # hedges that open with the verdict word
    "SUPPORTED -- not by this source.",
    "SUPPORTED, but the source doesnt say so.",
    "SUPPORTED, but the source doesn" + _MODIFIER_APOSTROPHE + "t say so.",
    "SUPPORTED, but the source doesn't say so.",
    "SUPPORTED, though the source disconfirms the date.",
    "SUPPORTED, mais la source ne le dit pas.",
)

_STILL_SUPPORTED = (
    # the phrasings the verifier's own suite holds
    "SUPPORTED. The source states this directly.",
    "SUPPORTED, the source confirms it.",
    "SUPP" + "ORTED, confirmed.",
    # the verdict word, alone or behind markdown or a label
    "SUPPORTED",
    "Supported.",
    "**SUPPORTED** -- the figure matches.",
    "Verdict: SUPPORTED. The date matches.",
    "> SUPPORTED, the source states it.",
)

_UNSUPPORTED_BODY = "\nThe claim is not supported."

_CYRILLIC_O = chr(0x043E)
_SOFT_HYPHEN = chr(0x00AD)

# The verdict word alone on its line, the reason on the next: the layout the
# instruction asks for.
_NEGATED_ON_THE_REASON_LINE = (
    "SUPPORTED\nThe source does not confirm the claim.",
    "SUPPORTED.\nHowever, nothing in the source corroborates the date.",
    "**SUPPORTED**\n\nActually no: the source never says this.",
    "SUPPORTED.\nThe claim is not supported by the source.",
    "SUPPORTED\nNo, the source never confirms it.",
    "SUPPORTED\nThe source is ambiguous about the year.",
    "SUPPORTED\nLa source ne le dit pas.",
    "SUPPORTED\nNothing contradicts it.",
)
_SUPPORTED_ON_TWO_LINES = (
    "SUPPORTED\nThe source states the date exactly.",
    "SUPPORTED\n\nThe source confirms the boiling point at sea level.",
    "SUPPORTED.\nLa source le confirme.",
)
_NEVER_BY_THE_LEXICON = (
    # a negating prefix on a word of support the first list did not name
    "SUPPORTED: unsubstantiated.",
    "SUPPORTED: the source disagrees with the claim.",
    "SUPPORTED: the figure is incorrect.",
    "SUPPORTED: mismatch with the source.",
    "SUPPORTED: the claim is untrue.",
    "SUPPORTED, the claim is inaccurate.",
    "SUPPORTED, the claim is invalid.",
    "SUPPORTED, the figure is inexact.",
    # refusals and contradictions in plain words, English then French
    "SUPPORTED: nope.",
    "SUPPORTED: the date is wrong.",
    "SUPPORTED: the source is silent on this.",
    "SUPPORTED, but the source says otherwise.",
    "SUPPORTED. The source says the opposite.",
    "SUPPORTED, la source dit le contraire.",
    "SUPPORTED, mais la source l" + _RIGHT_QUOTE + "infirme.",
    "SUPPORTED, but the figure is faux.",
    # a hyphen typed another way, hidden, or left out
    "SUPPORTED: un" + chr(0x2010) + "confirmed.",
    "SUPPORTED: un" + chr(0x2011) + "confirmed.",
    "SUPPORTED: in" + chr(0x2212) + "consistent.",
    "SUPPORTED: in" + _SOFT_HYPHEN + "consistent.",
    "SUPPORTED: un confirmed.",
    # a word the rule cannot read, and a longer word the verdict word begins
    "SUPPORTED: n" + _CYRILLIC_O + "t by the source.",
    "SUPPORTED-ish.",
)


def _load():
    loaded, restore = isolate(
        targets={
            _WRAPPER: source("agent", "untrusted_context.py"),
            _VERIFY: source("agent", "claim_verification.py"),
            _AGGREGATE: source("agent", "claim_aggregation.py"),
        },
        packages=("opti_oignon.agent",),
    )
    return loaded[_VERIFY], loaded[_AGGREGATE], restore


class _Replies:
    """A scripted one-shot client that answers each call with the next reply."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        return self.replies.pop(0)


def test_vd1_a_verdict_is_supported_only_when_it_opens_with_supported_and_negates_nothing():
    verify_mod, aggregate_mod, restore = _load()
    try:
        normalize = verify_mod.normalize_verdict
        supported = verify_mod.VERDICT_SUPPORTED
        uncertain = verify_mod.VERDICT_UNCERTAIN
        unsupported = verify_mod.VERDICT_UNSUPPORTED

        # Presence: both families are met, one verdict read per phrasing.
        assert len(_NEVER_SUPPORTED) == 28 and len(_STILL_SUPPORTED) == 8

        # c1 -- never supported: uncertain alone, unsupported with a body
        # that says so.
        leaked = [text for text in _NEVER_SUPPORTED if normalize(text) == supported]
        assert leaked == [], leaked
        wrong = {text: normalize(text) for text in _NEVER_SUPPORTED if normalize(text) != uncertain}
        assert wrong == {}, wrong
        with_body = {
            text: normalize(text + _UNSUPPORTED_BODY)
            for text in _NEVER_SUPPORTED
            if normalize(text + _UNSUPPORTED_BODY) != unsupported
        }
        assert with_body == {}, with_body

        # c2 -- the plain verdict word is still support.
        lost = {text: normalize(text) for text in _STILL_SUPPORTED if normalize(text) != supported}
        assert lost == {}, lost

        # c3 -- through the path the routes take.
        client = _Replies("The source does not confirm the claim.")
        result = verify_mod.make_claim_verifier(client)(_CLAIM, _SOURCE)
        assert client.calls == 1
        assert result.ok is True and result.verdict != supported, result
        client = _Replies("SUPPORTED.", "The source does not confirm the claim.")
        answer = aggregate_mod.make_answer_verifier(client)([(_CLAIM, _SOURCE), (_CLAIM, _SOURCE)])
        assert client.calls == 2
        assert [r.verdict for r in answer.results][0] == supported, "control: the plain reply is support"
        assert answer.verdict != supported, answer

        # c4 -- the stated cost. A reply that does not open with the verdict
        # word, or that carries a cue, is not support; a later relaxation
        # supersedes this clause by name.
        assert normalize("The source confirms the claim.") == uncertain
        assert normalize("SUPPORTED. The source confirms the figure (no rounding).") == uncertain

        # c5 -- the reason line counts: a negation or a hedge on any line of
        # the reply keeps it from support, and an unsupported body says so.
        assert len(_NEGATED_ON_THE_REASON_LINE) == 8 and len(_SUPPORTED_ON_TWO_LINES) == 3
        leaked = [text for text in _NEGATED_ON_THE_REASON_LINE if normalize(text) == supported]
        assert leaked == [], leaked
        with_body = {
            text: normalize(text + _UNSUPPORTED_BODY)
            for text in _NEGATED_ON_THE_REASON_LINE
            if normalize(text + _UNSUPPORTED_BODY) != unsupported
        }
        assert with_body == {}, with_body
        lost = {text: normalize(text) for text in _SUPPORTED_ON_TWO_LINES if normalize(text) != supported}
        assert lost == {}, lost

        # c6 -- the cues the first word list missed.
        assert len(_NEVER_BY_THE_LEXICON) == 23
        leaked = [text for text in _NEVER_BY_THE_LEXICON if normalize(text) == supported]
        assert leaked == [], leaked
        wrong = {text: normalize(text) for text in _NEVER_BY_THE_LEXICON if normalize(text) != uncertain}
        assert wrong == {}, wrong
    finally:
        restore()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
