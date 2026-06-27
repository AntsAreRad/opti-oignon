#!/usr/bin/env python3
"""Tests for claim-vs-source verification (agent.claim_verification).

This is the agentic-output check: given a claim and a cited source, a model is
asked whether the source supports the claim, and its free-text answer is mapped
to one of {supported, unsupported, uncertain}. The whole design is fail-secure
-- the verifier never raises and never *promotes* an ambiguous answer to
``supported``; only an explicit signal does that.

``normalize_verdict`` is the security heart: the verdict ordering matters
because "unsupported" contains "supported" as a substring, so unsupported and
uncertain are matched before supported, and a body that merely says "supports"
is never read as support (only a whole-text *unsupported* signal moves off
uncertain). The orchestration is tested with an injected fake model client, so
no network or Ollama is involved.
"""

import pytest

from opti_oignon.agent.claim_verification import (
    VERDICT_SUPPORTED,
    VERDICT_UNCERTAIN,
    VERDICT_UNSUPPORTED,
    build_messages,
    make_claim_verifier,
    normalize_verdict,
)

CLAIM = "Water boils at 100C at sea level."
SOURCE = "The reference notes that water boils at 100 degrees Celsius at sea level."


# ===========================================================================
# normalize_verdict — empty / lead-word mapping
# ===========================================================================

def test_none_or_blank_is_uncertain():
    assert normalize_verdict(None) == VERDICT_UNCERTAIN
    assert normalize_verdict("") == VERDICT_UNCERTAIN
    assert normalize_verdict("   \n  ") == VERDICT_UNCERTAIN


def test_supported_lead():
    assert normalize_verdict("SUPPORTED. The source states this directly.") == VERDICT_SUPPORTED


def test_unsupported_lead():
    assert normalize_verdict("UNSUPPORTED. The source contradicts it.") == VERDICT_UNSUPPORTED


def test_uncertain_lead():
    assert normalize_verdict("UNCERTAIN. The source does not settle it.") == VERDICT_UNCERTAIN


def test_unsupported_not_misread_as_supported():
    # "unsupported" contains "supported"; the ordering must resolve to unsupported.
    assert normalize_verdict("UNSUPPORTED") == VERDICT_UNSUPPORTED


def test_other_unsupported_markers_in_lead():
    assert normalize_verdict("Not supported by the cited text.") == VERDICT_UNSUPPORTED
    assert normalize_verdict("This contradicts the source.") == VERDICT_UNSUPPORTED


# ===========================================================================
# normalize_verdict — fail-secure asymmetry
# ===========================================================================

def test_ambiguous_answer_is_uncertain_not_supported():
    # No verdict word in the lead and no unsupported signal anywhere -> uncertain,
    # never promoted to supported.
    assert normalize_verdict("The document discusses several topics in detail.") == VERDICT_UNCERTAIN


def test_whole_text_unsupported_override():
    # Lead has no verdict word, but the body explicitly says "not supported".
    text = "Here is my analysis.\nThe claim is not supported by the source."
    assert normalize_verdict(text) == VERDICT_UNSUPPORTED


def test_body_support_is_never_promoted():
    # Lead has no verdict word; body says it "supports the claim" -- this must
    # stay uncertain (no whole-text supported promotion), the anti-hallucination
    # guarantee.
    text = "Let me analyze this.\nThe source supports the claim clearly."
    assert normalize_verdict(text) == VERDICT_UNCERTAIN


# ===========================================================================
# build_messages
# ===========================================================================

def test_build_messages_shape():
    msgs = build_messages(CLAIM, SOURCE)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "SUPPORTED" in msgs[0]["content"]      # instruction names the verdict words
    assert msgs[1]["role"] == "user"


def test_build_messages_rejects_empty():
    with pytest.raises(ValueError):
        build_messages("", SOURCE)
    with pytest.raises(ValueError):
        build_messages(CLAIM, "   ")


# ===========================================================================
# make_claim_verifier — orchestration with an injected client
# ===========================================================================

def _client_returning(text):
    return lambda messages: text


class _StreamClient:
    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, messages):
        return list(self._chunks)


def _raising_client(messages):
    raise RuntimeError("model down")


def test_verify_maps_supported():
    verify = make_claim_verifier(_client_returning("SUPPORTED, the source confirms it."))
    result = verify(CLAIM, SOURCE)
    assert result.ok is True
    assert result.verdict == VERDICT_SUPPORTED
    assert "SUPPORTED" in result.raw_text


def test_verify_coalesces_streaming_chunks():
    verify = make_claim_verifier(_StreamClient(["SUPP", "ORTED, confirmed."]))
    assert verify(CLAIM, SOURCE).verdict == VERDICT_SUPPORTED


def test_verify_without_client_fails_secure():
    verify = make_claim_verifier(None)
    result = verify(CLAIM, SOURCE)
    assert result.ok is False
    assert result.verdict == VERDICT_UNCERTAIN
    assert "unavailable" in result.reason.lower()


def test_verify_empty_claim_fails_secure():
    verify = make_claim_verifier(_client_returning("SUPPORTED"))
    result = verify("", SOURCE)
    assert result.ok is False
    assert result.verdict == VERDICT_UNCERTAIN
    assert "claim" in result.reason.lower()


def test_verify_never_raises_on_client_error():
    verify = make_claim_verifier(_raising_client)
    result = verify(CLAIM, SOURCE)
    assert result.ok is False
    assert result.verdict == VERDICT_UNCERTAIN
    assert "failed" in result.reason.lower()
