#!/usr/bin/env python3
"""Citation extraction: turn a produced answer plus its sources into the
(claim, source) pairs the per-answer verification consumes.

The producer half of the claim-vs-source verification arc (the role S267, the
aggregation S271, the route S272). Every layer downstream consumes a list of
(claim, source) pairs that it is simply handed; the S271 aggregation names them
"the kind a citation-extraction step would hand it" and defers extracting them
from a produced answer to a later lot. This module is that parser: given an
answer carrying inline numeric citation markers and the ordered sources those
markers index, it produces the (claim, source) pairs that
:func:`opti_oignon.agent.claim_aggregation.make_answer_verifier` runs through
the verification role.

Design notes:

- Pure and local. This module imports only the standard library; it reaches no
  network, opens no database, and pulls in no backend at load, so it is 100%
  local / Python and is exercised directly by pytest with no fastapi / ollama
  chain (the S243 isolation lesson). It is deterministic: the same answer and
  sources always yield the same pairs.
- Not a model-reachable tool. Like the role and the aggregation, this is a
  caller-driven surface: a producing path (or a UI, or a later agent step) hands
  in the answer and its sources, never the model's tool calling. It defines no
  tool schema and registers nothing in the agent tool registry, so it grows no
  schema-count or allowlist pin.
- No egress, no mode gate. The extractor reads only the supplied answer and
  sources, so it runs identically in Daily and Bulbe with no mode resolution and
  no mode provider (CV-D4, the verification arc's posture carried up to the
  producer).
- Never raises. Malformed input -- a non-string answer, non-iterable sources,
  mixed-type or empty sources, malformed markers -- degrades to fewer or zero
  pairs, never an exception. A caller can feed it raw model output without a
  try/except.

The citation contract (explicit and dependency-free):

- The answer carries inline numeric citation markers of the form ``[n]``, where
  ``n`` is a 1-based integer index into the ordered ``sources`` sequence
  (``[1]`` -> ``sources[0]``). This is the common RAG citation convention; a
  producing path that wires this module should emit markers in this form, placed
  inside the sentence they support (before the sentence terminator).
- A claim is a sentence that bears at least one citation marker. Sentences are
  segmented deterministically on terminal ``.`` / ``!`` / ``?`` followed by
  whitespace; segmentation is intentionally simple (a heavier NLP segmenter
  would add a dependency, against the local-first posture).
- For each cited sentence, one ``(claim, source)`` pair is emitted per distinct
  in-range marker in that sentence: the claim is the sentence text with the
  numeric citation markers stripped and surrounding whitespace collapsed, and
  the source is the indexed source string. A sentence citing several sources is
  verified against each of them.
- Fail-closed by omission. A marker out of range, a ``[0]`` marker, a malformed
  marker (``[]`` / ``[abc]`` / ``[1.2]``), a resolved source that is empty or
  whitespace, or a claim that is empty after stripping all yield no pair -- the
  extractor never emits a pair against the wrong source or against nothing.
  Pairs are de-duplicated per ``(claim, source-index)`` preserving first-seen
  order.
- The output is a ``list[tuple[str, str]]`` of ``(claim, source)`` pairs in
  answer order, exactly the shape ``verify_answer`` consumes (its
  ``_coerce_pair`` indexes ``pair[0]`` / ``pair[1]``).

``checkpoint_before_apply`` is hardcoded True and never overridable;
``FEATURE_AVAILABLE`` gates graceful degradation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Hardcoded and never overridable: a checkpoint is taken before any mutation is
# applied. Project-wide non-negotiable for every new module.
checkpoint_before_apply = True

FEATURE_AVAILABLE = True

# A citation marker is an open bracket, one or more digits, a close bracket.
# Non-integer or empty bracket forms are not markers; range validity (1-based,
# within the sources) is enforced by extract_pairs, not by this pattern.
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# A sentence terminator followed by whitespace splits two sentences. The
# look-behind keeps the terminator attached to the sentence it ends.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A citation marker together with any whitespace immediately preceding it, so
# stripping "mammals [1]." leaves "mammals." rather than "mammals .".
_MARKER_WITH_LEAD_WS = re.compile(r"\s*\[\d+\]")

# Collapse any run of whitespace to a single space.
_WS_RUN = re.compile(r"\s+")


def split_sentences(text: Any) -> list[str]:
    """Segment ``text`` into sentences, deterministically and fail-soft.

    Splits on a terminal ``.`` / ``!`` / ``?`` followed by whitespace; the
    terminator stays with the sentence it ends. A ``None`` or whitespace-only
    input yields an empty list. Text with no terminator is a single sentence.
    """
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    parts = _SENTENCE_SPLIT.split(s)
    return [p for p in (part.strip() for part in parts) if p]


def find_citation_indices(text: Any) -> list[int]:
    """Return the citation indices in ``text``, in order and de-duplicated.

    A pure scanner over the ``[n]`` markers: it returns every well-formed
    integer marker (including a ``0`` or an out-of-range value) in first-seen
    order, with duplicates removed. Range validity is the caller's concern
    (:func:`extract_pairs` filters to 1-based, in-range markers).
    """
    if text is None:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for match in CITATION_PATTERN.finditer(str(text)):
        n = int(match.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def strip_citation_markers(text: Any) -> str:
    """Remove numeric citation markers from ``text``, collapsing whitespace.

    Only well-formed ``[n]`` markers are removed (with any whitespace that
    immediately precedes them); other bracketed text is left untouched. The
    result has runs of whitespace collapsed to one space and is stripped.
    """
    if text is None:
        return ""
    stripped = _MARKER_WITH_LEAD_WS.sub("", str(text))
    return _WS_RUN.sub(" ", stripped).strip()


def extract_pairs(
    answer: Any, sources: Sequence[Any]
) -> list[tuple[str, str]]:
    """Parse ``answer`` and its ``sources`` into (claim, source) pairs.

    The answer carries inline numeric citation markers ``[n]`` (1-based) that
    index ``sources`` by position. Each sentence bearing at least one marker
    becomes a claim (markers stripped, whitespace collapsed) paired with each
    in-range source it cites. Out-of-range, ``[0]``, malformed, or empty-source
    citations yield no pair; pairs are de-duplicated per (claim, source-index)
    in first-seen order. Returns ``list[tuple[str, str]]`` -- exactly what
    ``verify_answer`` consumes. Never raises.
    """
    try:
        text = "" if answer is None else str(answer)
    except Exception:  # pragma: no cover - defensive, str() is total in practice
        return []
    if not text.strip():
        return []

    try:
        srcs: list[Any] = [] if sources is None else list(sources)
    except TypeError:
        # A non-iterable sources argument cannot index any marker.
        return []
    if not srcs:
        return []

    out: list[tuple[str, str]] = []
    seen: set[tuple[str, int]] = set()
    for sentence in split_sentences(text):
        indices = find_citation_indices(sentence)
        if not indices:
            continue
        claim = strip_citation_markers(sentence)
        if not claim:
            continue
        for n in indices:
            if n < 1 or n > len(srcs):
                # 1-based and within the sources; never resolve to a wrong or
                # absent source.
                continue
            raw = srcs[n - 1]
            source = "" if raw is None else str(raw)
            if not source.strip():
                # An empty or whitespace source cannot verify anything.
                continue
            key = (claim, n)
            if key in seen:
                continue
            seen.add(key)
            out.append((claim, source))
    return out
