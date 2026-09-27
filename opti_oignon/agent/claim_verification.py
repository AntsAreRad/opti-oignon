#!/usr/bin/env python3
"""Claim-vs-source verification role (the gated verification surface).

A first implementation lot for the logged DEBT_LOT_S261 roadmap item: a role
that checks a model-generated claim against its cited source and returns a
fail-secure verdict. It is local, deterministic in its plumbing, and routes its
single inference call through an injected one-shot seam, so it is 100% local /
Python / Ollama with no backend coupling at module load.

Design notes:

- This is NOT a model-reachable tool. Unlike the N.4 ``manage_notes`` tool,
  this surface is driven by a caller handing in a (claim, source) pair, not by
  the model's tool calling. It defines no ``ToolSchema`` and registers nothing
  in the agent tool registry, so it grows no schema-count or allowlist pin.
- Anti-injection. Both the claim (model-generated, untrusted) and the cited
  source (external, untrusted) are wrapped as untrusted data under one policy
  header via :func:`opti_oignon.agent.untrusted_context.untrusted_message_many`
  (the core anti-injection idiom). The verification instruction is the only trusted
  message; both pieces ride the user role inside untrusted-data markers, so
  injection-looking text in either piece cannot steer the model.
- Fail-secure verdict. The taxonomy is supported / unsupported / uncertain. The
  mapping of free-text model output is asymmetric on purpose: an unparseable or
  ambiguous reply defaults to UNCERTAIN, never to SUPPORTED, and only an
  explicit unsupported signal moves a lead-ambiguous reply off uncertain. A
  verification role that rubber-stamped "supported" on ambiguity would be
  dangerous; an indeterminate verification never asserts support.
- No egress, no mode gate. Verification reads only the supplied source plus the
  local model and reaches no network, so the role runs identically in Daily and
  Bulbe with no mode resolution and no web action. There is deliberately no
  mode provider.
- Dependency injection. The model client is a one-shot inference seam the caller
  injects (a callable over the built messages, or an object exposing
  ``stream``). An un-injected verifier reports a clean failure rather than
  guessing a model, exactly the N.3 posture. A later route / UI lot wires the
  client from the user's selected model.
- The inference seam is injectable for tests; nothing here imports the backend
  at module load, so the surface is exercised directly by pytest with no
  fastapi / ollama chain.

``checkpoint_before_apply`` is hardcoded True and never overridable;
``FEATURE_AVAILABLE`` gates graceful degradation.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

from .untrusted_context import untrusted_message_many

logger = logging.getLogger(__name__)

# Hardcoded and never overridable: a checkpoint is taken before any mutation is
# applied. Project-wide non-negotiable for every new module.
checkpoint_before_apply = True

FEATURE_AVAILABLE = True

# The source labels the two untrusted pieces are wrapped under (sanitised by
# untrusted_context to safe tag attributes).
SOURCE_CLAIM = "claim"
SOURCE_SOURCE = "source"

# The verdict taxonomy.
VERDICT_SUPPORTED = "supported"
VERDICT_UNSUPPORTED = "unsupported"
VERDICT_UNCERTAIN = "uncertain"

ALL_VERDICTS: frozenset[str] = frozenset(
    {VERDICT_SUPPORTED, VERDICT_UNSUPPORTED, VERDICT_UNCERTAIN}
)

# The trusted verification instruction. It names the job, the source-only rule,
# and asks the model to lead with the verdict word so the mapping is robust.
_VERIFY_INSTRUCTION = (
    "You are a verification role. The untrusted-data block below contains a "
    "claim labelled 'claim' and a cited source labelled 'source'. Decide "
    "whether the claim is supported by the source, using only the source and "
    "no outside knowledge. Begin your answer with exactly one word: SUPPORTED, "
    "UNSUPPORTED, or UNCERTAIN, then give a brief reason grounded in the "
    "source. Answer UNCERTAIN when the source does not settle the claim either "
    "way; do not guess."
)

# Verdict markers, scanned fail-secure. Unsupported and uncertain are checked
# before supported because "unsupported" contains "supported" as a substring.
# The lead decides unsupported and uncertain; support needs the whole reply,
# since the instruction asks for the verdict word first and the reason after
# it, often on the next line. When the reply is not support, an unsupported
# marker anywhere in it makes it unsupported; there is never a whole-text
# supported promotion.
_UNSUPPORTED_MARKERS = (
    "unsupported",
    "not supported",
    "not support",
    "no support",
    "contradict",
    "refute",
)
_UNCERTAIN_MARKERS = (
    "uncertain",
    "unclear",
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "insufficient",
    "not enough",
    "no information",
    "not addressed",
    "does not address",
    "doesn't address",
    "not sure",
    "unsure",
    "ambiguous",
    "maybe",
)

# Support is read from the opening word alone, never from a substring: a
# reply that merely contains "confirm" or "consistent with" is as likely to
# say "does not confirm" or "inconsistent with". A reply is supported only
# when its first line opens with the verdict word the instruction asks for,
# and no line of it carries a negation cue, a hedge or an unsupported marker.
#
# Markdown emphasis, a quote marker, a list dash or a heading may come before
# the word, and so may one "verdict:" or "answer:" label. A word it only
# begins -- "SUPPORTED-ish" -- is not the verdict word.
_LEAD_NOISE = " \t*_#>-`"
_LEAD_LABEL = re.compile(r"(?:verdict|answer)\s*[:\-]\s*")
_OPENS_SUPPORTED = re.compile(r"supported\b")
_COMPOUND = re.compile(r"[-'][^\W\d_]")

# Every apostrophe a model or a keyboard produces is read as the ASCII one,
# every hyphen as the ASCII hyphen, and a soft hyphen, which a renderer
# hides, is dropped.
_APOSTROPHES = tuple(chr(c) for c in (0x2018, 0x2019, 0x02BC, 0x0060, 0x00B4, 0x2032, 0xFF07))
_HYPHENS = {c: "-" for c in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0xFE63, 0xFF0D)}
_HYPHENS[0x00AD] = None
# A negating prefix on a word of support negates it: "unconfirmed",
# "inconsistent", "disagrees", "mismatch", "untrue", "inaccurate". Written
# with a hyphen it is joined to its word; written apart, it is joined when a
# word of support follows.
_NEGATING_PREFIXES = ("un", "in", "non", "dis", "mis")
_POSITIVE_STEMS = (
    "confirm", "consistent", "corroborat", "support", "verif", "prov",
    "substantiat", "agree", "correct", "accura", "valid", "true", "exact",
    "match", "back",
)
_JOINED_PREFIX = re.compile(r"\b(un|in|non|dis|mis)-+(?=[^\W\d_])")
_SPACED_PREFIX = re.compile(r"\b(un|non|dis|mis)\s+(?=(?:" + "|".join(_POSITIVE_STEMS) + r"))")
_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
# Negation and refusal cue words, English then French, read without accents.
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "nothing", "none", "neither", "nor", "without",
    "cannot", "fail", "fails", "failed", "lack", "lacks", "lacking",
    "unable", "impossible", "hardly", "barely", "false", "deny", "denies",
    "denied", "nope", "nah", "wrong", "wrongly", "silent", "otherwise",
    "contrary", "opposite", "absent",
    "ne", "pas", "jamais", "rien", "aucun", "aucune", "non", "sans", "ni",
    "faux", "fausse", "contraire", "infirme", "infirment", "infirmer",
    "contredit", "contredisent", "errone", "erronee", "errones", "erronees",
})
# Contractions written without their apostrophe.
_BARE_CONTRACTIONS = frozenset({
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "cant",
    "couldnt", "wont", "wouldnt", "shouldnt", "hasnt", "havent", "hadnt",
    "aint", "mustnt", "neednt",
})


@dataclass
class ClaimVerificationResult:
    """The outcome of a claim-vs-source verification.

    ``ok`` True carries the mapped ``verdict`` and the model's ``raw_text``; any
    failure is ``ok`` False with a ``reason``, the ``verdict`` held at the
    fail-secure ``uncertain``. The verifier never raises.
    """

    verdict: str
    ok: bool
    reason: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "ok": self.ok,
            "reason": self.reason,
            "raw_text": self.raw_text,
        }


def _has(markers: tuple[str, ...], hay: str) -> bool:
    return any(m in hay for m in markers)


def _fold(text: str) -> str:
    """Apostrophes, hyphens and compatibility forms folded, lowercased.

    Apostrophes are folded before and after NFKC, which splits one of them
    into a space and a combining accent.
    """
    for mark in _APOSTROPHES:
        text = text.replace(mark, "'")
    text = unicodedata.normalize("NFKC", text.translate(_HYPHENS)).translate(_HYPHENS)
    for mark in _APOSTROPHES:
        text = text.replace(mark, "'")
    return text.lower()


def _opens_with_supported(lead: str) -> bool:
    """True when the lead's first word is SUPPORTED and no question mark follows it."""
    text = _fold(lead).lstrip(_LEAD_NOISE)
    label = _LEAD_LABEL.match(text)
    if label:
        text = text[label.end():].lstrip(_LEAD_NOISE)
    opening = _OPENS_SUPPORTED.match(text)
    if not opening or _COMPOUND.match(text, opening.end()):
        return False
    rest = text[opening.end():].lstrip(_LEAD_NOISE)
    return not rest.startswith("?")


def _unaccented(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", word) if not unicodedata.combining(c))


def _mixed_script(word: str) -> bool:
    """A word mixing ASCII letters with letters of another script cannot be read."""
    if not any("a" <= c <= "z" for c in word):
        return False
    return any(
        c.isalpha() and not ("a" <= c <= "z") and not unicodedata.name(c, "").startswith("LATIN")
        for c in word
    )


def _negated(text: str) -> bool:
    """True when the text carries a negation or refusal cue, or a word it cannot read."""
    text = _fold(text)
    text = _JOINED_PREFIX.sub(r"\1", text)
    text = _SPACED_PREFIX.sub(r"\1", text)
    for token in _TOKEN.findall(text):
        if _mixed_script(token):
            return True
        word = _unaccented(token)
        if word.endswith("n't") or word.startswith("n'"):
            return True
        for part in [word, *word.split("'")]:
            if part in _NEGATION_WORDS or part in _BARE_CONTRACTIONS:
                return True
            for prefix in _NEGATING_PREFIXES:
                if part.startswith(prefix) and part[len(prefix):].startswith(_POSITIVE_STEMS):
                    return True
    return False


def _qualified(reply: str) -> bool:
    """True when any line of the reply negates, hedges or says unsupported."""
    return _negated(reply) or _has(_UNCERTAIN_MARKERS, reply) or _has(_UNSUPPORTED_MARKERS, reply)


def normalize_verdict(text: Any) -> str:
    """Map free-text model output to a verdict, fail-secure to uncertain.

    The lead (first line) decides unsupported and uncertain, tested before
    supported so the "unsupported" substring is never read as support. A
    reply is supported only when its lead opens with the word SUPPORTED --
    markdown or a "verdict:" label allowed before it, no question mark after
    it, not a longer word it begins -- and no line of the reply carries a
    negation cue (English or French, whatever the apostrophe or the hyphen,
    a negating prefix on a word of support, a word mixing scripts), a hedge
    or an unsupported marker: the reason often comes on the next line. Any
    other reply is uncertain, or unsupported when an explicit unsupported
    signal appears anywhere in it; an ambiguous reply is never promoted to
    supported.
    """
    if text is None or not str(text).strip():
        return VERDICT_UNCERTAIN
    low = str(text).strip().lower()
    head = low.split("\n", 1)[0]
    if _has(_UNSUPPORTED_MARKERS, head):
        return VERDICT_UNSUPPORTED
    if _has(_UNCERTAIN_MARKERS, head):
        return VERDICT_UNCERTAIN
    if _opens_with_supported(head) and not _qualified(low):
        return VERDICT_SUPPORTED
    if _has(_UNSUPPORTED_MARKERS, low):
        return VERDICT_UNSUPPORTED
    return VERDICT_UNCERTAIN


def build_messages(claim: str, source: str) -> list[dict[str, str]]:
    """Build the one-shot [system, user] messages for a verification.

    The system message is the trusted verification instruction; the user message
    wraps the claim and the cited source as untrusted data under one policy
    header (the anti-injection core). Raises ``ValueError`` on an empty claim or
    empty source -- the runner guards both before building.
    """
    c = "" if claim is None else str(claim)
    s = "" if source is None else str(source)
    if not c.strip():
        raise ValueError("Empty claim: nothing to verify.")
    if not s.strip():
        raise ValueError("Empty source: nothing to verify the claim against.")
    user_msg = untrusted_message_many(
        [(SOURCE_CLAIM, c), (SOURCE_SOURCE, s)]
    )
    if user_msg is None:  # pragma: no cover - guarded by the strip checks above
        raise ValueError("No untrusted content to wrap.")
    return [{"role": "system", "content": _VERIFY_INSTRUCTION}, user_msg]


def _default_model_client() -> Any:
    """No process-default model client: the caller injects a one-shot client.

    Returns None so an un-injected verifier reports a clean failure rather than
    guessing a model. A later route / UI lot wires the client from the user's
    selected model, the same dependency-injection posture as the N.3 surface.
    """
    return None


def _invoke_once(model_client: Any, messages: list[dict[str, str]]) -> str:
    """Invoke the one-shot inference seam and coerce its output to text.

    Mirrors the N.3 tolerance: ``model_client`` may expose ``stream`` or be a
    plain callable taking the messages. The return may be a string or an
    iterable of chunks (strings, ``{"content": ...}`` dicts, or objects with
    ``content``).
    """
    fn = getattr(model_client, "stream", None)
    if fn is None and callable(model_client):
        fn = model_client
    if fn is None:
        raise TypeError("model client is not callable and has no stream method")
    out = fn(messages)
    if isinstance(out, str):
        return out
    parts: list[str] = []
    for chunk in out:
        if isinstance(chunk, str):
            parts.append(chunk)
        elif isinstance(chunk, dict):
            parts.append(str(chunk.get("content", "")))
        else:
            parts.append(str(getattr(chunk, "content", "")))
    return "".join(parts)


def make_claim_verifier(
    model_client: Any = None,
) -> Callable[[str, str], ClaimVerificationResult]:
    """Build a claim-vs-source verifier, injecting the one-shot inference seam.

    ``model_client`` is the inference seam (a callable over the built messages,
    or an object with ``stream``); when None the default resolver is used (which
    returns None unless wired by a caller, yielding a clean failure). There is
    deliberately no mode provider: the role reaches no network and runs the same
    in Daily and Bulbe.

    The returned ``verify(claim, source)`` refuses an empty claim or source with
    a structured fail-secure result, wraps both as untrusted data, invokes the
    model once, maps the output to a verdict (fail-secure to uncertain), and
    returns a :class:`ClaimVerificationResult`. It never raises.
    """

    def verify(claim: str, source: str) -> ClaimVerificationResult:
        c = "" if claim is None else str(claim)
        s = "" if source is None else str(source)
        if not c.strip():
            return ClaimVerificationResult(
                verdict=VERDICT_UNCERTAIN,
                ok=False,
                reason="Empty claim: nothing to verify.",
            )
        if not s.strip():
            return ClaimVerificationResult(
                verdict=VERDICT_UNCERTAIN,
                ok=False,
                reason="Empty source: nothing to verify the claim against.",
            )
        client = model_client if model_client is not None else _default_model_client()
        if client is None:
            return ClaimVerificationResult(
                verdict=VERDICT_UNCERTAIN,
                ok=False,
                reason="Model client unavailable.",
            )
        try:
            messages = build_messages(c, s)
            text = _invoke_once(client, messages)
        except Exception as exc:
            return ClaimVerificationResult(
                verdict=VERDICT_UNCERTAIN,
                ok=False,
                reason="Verification failed: " + str(exc),
            )
        verdict = normalize_verdict(text)
        return ClaimVerificationResult(verdict=verdict, ok=True, raw_text=text)

    return verify
