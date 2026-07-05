#!/usr/bin/env python3
"""User-facing response hygiene for tool-driven generations.

Three small, dependency-free helpers shared by the tool loop and the request
classifier:

``normalize_for_match``
    Accent-stripped, apostrophe-folded, lowercased view of a string, so
    keyword detection matches accented French phrasings ("actualit\u00e9",
    "ex\u00e9cute", typographic apostrophes) against unaccented keyword lists.

``strip_internal_markers``
    Removes lines of internal scaffolding (environment framing, tool-result
    headers, verification notes, round reminders, untrusted-data markers)
    that a model sometimes echoes into its final answer. Line-anchored and
    fence-aware: content inside fenced code blocks is preserved verbatim.

``detect_misattribution``
    Flags second-person action claims ("you created the file", "tu as cr\u00e9\u00e9
    le fichier") in a final answer. Tool actions in this project are executed
    by the assistant through the runtime, never by the user, so such claims
    signal an attribution error. This is a detector for logging and
    evaluation, NOT an automatic rewriter: a second-person claim can be
    legitimate when it refers to something the user actually did earlier, so
    the caller decides what to do with the signal.
"""

import re
import unicodedata

__all__ = [
    "normalize_for_match",
    "strip_internal_markers",
    "detect_misattribution",
    "StreamMarkerFilter",
]

# Typographic apostrophes folded to the ASCII one before accent stripping.
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u02bc": "'", "\u2032": "'"})


def normalize_for_match(text: str) -> str:
    """Accent-stripped, apostrophe-folded, lowercased view of ``text``.

    Combining marks are removed after NFD decomposition, so "\u00e9tape" and
    "etape" compare equal. Safe on empty or None-ish input.
    """
    if not text:
        return ""
    folded = str(text).translate(_APOSTROPHES)
    decomposed = unicodedata.normalize("NFD", folded)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower()


# One line of internal scaffolding, anchored at line start. Covers both the
# current environment framing and the legacy prefixes so an echo of either
# family is caught.
_SCAFFOLD_LINE = re.compile(
    r"^\s*("
    r"\[environment\]"
    r"|\[tool:"
    r"|\[prior tool call"
    r"|\[verification\]"
    r"|\[\d+\s+rounds?\s+remain"
    r"|previous tool results:"
    r"|here are the tool results:"
    r"|</?untrusted_data\b"
    r"|the block below is untrusted data"
    r")",
    re.IGNORECASE,
)

_FENCE = re.compile(r"^\s*(```|~~~)")


def strip_internal_markers(text: str) -> tuple[str, int]:
    """Drop scaffold lines outside fenced code blocks; return (text, dropped).

    Lines matching the scaffold patterns are removed only when they sit
    outside a fenced code block, so quoted code that legitimately contains a
    bracketed prefix is left untouched. The count of dropped lines is
    returned for logging and evaluation.
    """
    if not text:
        return text, 0
    out: list[str] = []
    dropped = 0
    in_fence = False
    for line in str(text).splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and _SCAFFOLD_LINE.match(line):
            dropped += 1
            continue
        out.append(line)
    cleaned = "\n".join(out)
    if text.endswith("\n") and not cleaned.endswith("\n"):
        cleaned += "\n"
    return cleaned, dropped


# Action verbs (normalized forms) that mark a tool-style action. French verbs
# are matched on their accent-stripped past participles; agreement suffixes
# (e/s) are tolerated.
_FR_ACTION = (
    r"(?:cree|ecrit|execute|lance|modifie|genere|corrige|teste|installe|"
    r"supprime|sauvegarde|repare|compile|construit)e?s?"
)
_EN_ACTION = (
    r"(?:created|wrote|written|ran|executed|made|modified|generated|fixed|"
    r"deleted|installed|saved|built|compiled)"
)

_MISATTRIBUTION = re.compile(
    r"\b(?:"
    r"(?:tu as|vous avez)\s+" + _FR_ACTION +
    r"|you(?:'ve| have)?\s+" + _EN_ACTION +
    r")\b"
)


def detect_misattribution(text: str) -> list[str]:
    """Second-person action claims found in ``text`` (normalized snippets).

    Matching runs on the normalized view, so accented French forms are
    caught. Returns the matched snippets (empty list when none), for logging
    and evaluation; see the module docstring for why this never rewrites.
    """
    if not text:
        return []
    return _MISATTRIBUTION.findall(normalize_for_match(text))


# Literal scaffold prefixes (lowercased) used by the incremental filter to
# decide whether a partially received line could still become a scaffold
# line. The digit-bearing reminder pattern is covered by the bracket rule.
_LITERAL_PREFIXES = (
    "[environment]",
    "[tool:",
    "[prior tool call",
    "[verification]",
    "previous tool results:",
    "here are the tool results:",
    "<untrusted_data",
    "</untrusted_data",
    "the block below is untrusted data",
)

_FENCE_MARKS = ("```", "~~~")


class StreamMarkerFilter:
    """Incremental ``strip_internal_markers`` for streamed text.

    Feed chunks as they arrive; the filter emits text as soon as the current
    line can no longer be a scaffold line, holding at most ``_HOLD_LIMIT``
    characters of a line while the decision is ambiguous. Scaffold lines are
    swallowed whole (including their newline); fenced code blocks pass
    verbatim. ``flush()`` releases whatever a truncated last line held, and
    ``dropped`` counts the removed lines -- so a fully streamed text yields
    exactly what ``strip_internal_markers`` would have produced.
    """

    _HOLD_LIMIT = 48

    def __init__(self) -> None:
        self._pending = ""
        self._mode = "hold"  # hold | pass | drop | fence
        self._in_fence = False
        self.dropped = 0

    def feed(self, text: str) -> str:
        """Consume a chunk; return the text safe to emit now."""
        if not text:
            return ""
        out: list[str] = []
        for piece in re.split(r"(\n)", str(text)):
            if piece == "":
                continue
            if piece == "\n":
                out.append(self._end_line())
                continue
            out.append(self._feed_segment(piece))
        return "".join(out)

    def flush(self) -> str:
        """Release the held tail of a text that ended mid-line."""
        released = ""
        if self._mode == "hold":
            released = self._resolve(final=True)
        tail = released if self._mode in ("pass", "fence") else ""
        self._pending = ""
        self._mode = "hold"
        return tail

    # -- per-line machinery -------------------------------------------------

    def _feed_segment(self, segment: str) -> str:
        if self._mode == "drop":
            return ""
        if self._mode in ("pass", "fence"):
            return segment
        self._pending += segment
        emitted = self._resolve(final=False)
        return emitted

    def _end_line(self) -> str:
        released = ""
        if self._mode == "hold":
            released = self._resolve(final=True)
        mode = self._mode
        if mode == "fence":
            self._in_fence = not self._in_fence
        self._pending = ""
        self._mode = "hold"
        if mode == "drop":
            return ""
        return released + "\n"

    def _resolve(self, final: bool) -> str:
        """Decide the current line's fate; return text released by PASS."""
        stripped = self._pending.lstrip()
        lowered = stripped.lower()

        if any(mark.startswith(lowered) and lowered != ""
               and len(lowered) < 3 for mark in _FENCE_MARKS) and not final:
            return ""  # could still become a fence marker
        if any(lowered.startswith(mark) for mark in _FENCE_MARKS):
            self._mode = "fence"
            released, self._pending = self._pending, ""
            return released
        if self._in_fence:
            self._mode = "pass"
            released, self._pending = self._pending, ""
            return released
        if lowered == "":
            if final:
                self._mode = "pass"
                released, self._pending = self._pending, ""
                return released
            return ""  # leading whitespace only; keep holding
        if _SCAFFOLD_LINE.match(self._pending):
            self._mode = "drop"
            self._pending = ""
            self.dropped += 1
            return ""
        ambiguous = (
            any(lit.startswith(lowered) for lit in _LITERAL_PREFIXES)
            or lowered.startswith("[")
        )
        if ambiguous and not final and len(self._pending) < self._HOLD_LIMIT:
            return ""
        self._mode = "pass"
        released, self._pending = self._pending, ""
        return released
