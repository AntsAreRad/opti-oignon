#!/usr/bin/env python3
"""Cross-source deduplication of retrieved snippets, before they are injected.

Several retrieval sources feed one prompt: personal memory, project
documents, web results, and the conversation archive. Each runs its own
retriever, none of them can see what the others already contributed, and
they draw on overlapping material, so the same passage reaches the model
twice or three times over. That costs context budget twice over as well,
and it teaches the model that a repeated claim is a corroborated one.

This module compares a source's candidates against the text already
composed for this turn and drops the ones that are already there. Three
properties are what make it safe to run on the composition path:

  * It only ever REMOVES candidates. It never rewrites a survivor, never
    merges two snippets into one, never reorders what it keeps, and never
    moves content from one source into another. Whatever survives is
    byte-for-byte what its own source produced.
  * It never emits content of its own. A dropped candidate is dropped; a
    kept candidate is the source's own object, handed back unchanged. So
    it cannot carry a passage across a trust boundary, and it cannot turn
    untrusted text into text that speaks with the platform's authority --
    it has no output channel for text at all.
  * It is pure and deterministic. No clock, no network, no persistence, no
    model, no optional dependency. The same inputs give the same survivors
    on every host and in every process, which is the precondition for any
    threshold measured on a retrieval path to mean anything twice.

Matching is deliberately conservative. A candidate is dropped only when it
is contained in what is already composed, or when it overlaps it above a
stated threshold. Under-dropping wastes a little budget; over-dropping
silently removes evidence the model needed and leaves no trace in the
answer, so the asymmetry is resolved in favour of keeping.
"""

import re
from typing import Any, Callable, Iterable, Sequence

# Module conventions (project-wide).
checkpoint_before_apply = True

# Fraction of a candidate's word windows that must already be present before
# it counts as a duplicate. High on purpose: see the note on asymmetry above.
DEFAULT_OVERLAP = 0.8

# Width of the word window used for near-duplicate comparison. Long enough
# that ordinary shared phrasing does not match, short enough that a snippet
# reformatted at the margins still does.
SHINGLE_WIDTH = 5

_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Fold a passage to the form the comparisons run on.

    Case, punctuation and whitespace differ between sources that quote the
    same passage -- one wraps lines, another strips markup, a third adds a
    role prefix. Comparing raw text would therefore miss duplicates that a
    reader sees immediately.
    """
    if not text:
        return ""
    return " ".join(_WORD.findall(text.casefold()))


def tokens(text: str) -> list[str]:
    """The normalized words of a passage, in order."""
    normalized = normalize(text)
    return normalized.split() if normalized else []


def shingles(words: Sequence[str], width: int = SHINGLE_WIDTH) -> set[tuple]:
    """Every window of ``width`` consecutive words, as a set.

    A passage shorter than one window yields no windows at all; callers
    fall back to containment for those, which is why this returns an empty
    set instead of pretending a short passage is a window of its own.
    """
    if width <= 0 or len(words) < width:
        return set()
    return {tuple(words[i:i + width]) for i in range(len(words) - width + 1)}


def compose_already_injected(
    system_prompt: str,
    volatile_parts: Iterable[str] = (),
) -> str:
    """The text a later source has to be compared against.

    A turn composes its prompt in two places: the system prompt itself, and
    the volatile parts that are held back and appended after the stable
    head. A comparison that looked at only one of them would be blind on
    exactly the layout the other is active for, and would let duplicates
    through whenever that layout is in use.
    """
    return (system_prompt or "") + "".join(volatile_parts or ())


def is_already_present(
    candidate: str,
    corpus: str,
    *,
    threshold: float = DEFAULT_OVERLAP,
) -> bool:
    """Whether ``candidate`` says something ``corpus`` does not already say."""
    return _present(tokens(candidate), normalize(corpus), None, threshold)


def _present(
    candidate_words: list[str],
    corpus_normalized: str,
    corpus_shingles: set[tuple] | None,
    threshold: float,
) -> bool:
    """Shared decision, with the corpus prepared once by the caller."""
    if not candidate_words:
        # Nothing to inject. Dropping it removes no evidence.
        return True

    candidate_normalized = " ".join(candidate_words)
    if candidate_normalized in corpus_normalized:
        return True

    windows = shingles(candidate_words)
    if not windows:
        # Too short to compare by windows, and containment already said no.
        return False

    if corpus_shingles is None:
        corpus_shingles = shingles(corpus_normalized.split())

    covered = len(windows & corpus_shingles)
    return covered / len(windows) >= threshold


def drop_duplicates(
    candidates: Sequence[Any],
    corpus: str,
    *,
    key: Callable[[Any], str],
    threshold: float = DEFAULT_OVERLAP,
) -> tuple[list[Any], list[Any]]:
    """Split ``candidates`` into what to inject and what is already said.

    ``key`` reads the text out of a candidate; the candidate objects
    themselves are handed back untouched, so a caller keeps every field its
    own source attached (score, role, provenance).

    Survivors accumulate: a candidate that duplicates an earlier survivor of
    the same batch is dropped too, because by the time it would be injected
    the earlier one already says it. Order is preserved on both lists.
    """
    corpus_normalized = normalize(corpus)
    corpus_shingles = shingles(corpus_normalized.split())

    kept: list[Any] = []
    dropped: list[Any] = []

    for candidate in candidates:
        words = tokens(key(candidate))
        if _present(words, corpus_normalized, corpus_shingles, threshold):
            dropped.append(candidate)
            continue

        kept.append(candidate)
        # The survivor is now part of what the next candidate is measured
        # against, since it will have been injected before that one is.
        corpus_normalized = (corpus_normalized + " " + " ".join(words)).strip()
        corpus_shingles = corpus_shingles | shingles(words)

    return kept, dropped
