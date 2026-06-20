#!/usr/bin/env python3
"""
Tests for S186 -- HY-01: English-only / no-emoji inference core.

S139 introduced an English-only guard, but its accent detector is neutered: its
exclude list contains ``re.compile(r"[À-ÿ]")``, which short-circuits the
``_FRENCH_ACCENTS`` branch (every accented line is excluded before it is
reached), and emoji are never checked at all. That is why accented French prose
and emoji glyphs survived in the inference core.

HY-01 was a dedicated mechanical pass that translated French
comments/docstrings/log strings to English and normalized non-ASCII symbols
(em/en dashes, arrows) to ASCII across the inference/context/routing core. This
module is the regression guard for that pass. It is intentionally strict:

  1. Zero emoji anywhere in the swept files (zero tolerance, even on regex
     lines: an injection/intent matcher never needs a pictograph).
  2. Zero non-ASCII anywhere in the swept files, with exactly two documented
     exceptions:
       (a) lines carrying a raw-string regex prefix (``r"..."`` / ``r'...'``),
           which may legitimately contain accented characters used to MATCH
           French user text (e.g. the past-reference detectors and the
           ``[a-zA-ZÀ-ÿ]`` tokenizer class in conversation_compressor.py);
       (b) the author-name token ``Léon`` / ``León`` in a file header.
  3. AST validity of every swept file.

Intentional ASCII French (data, not prose) is out of scope here because a
non-ASCII guard cannot and must not flag it -- e.g. the natural-language tool
triggers in tool_executor.py ("lis le fichier", "cherche", ...) and the French
stop-word list in conversation_compressor.py. Those are matching data and are
deliberately preserved.

EXCLUDED from the swept set (NOT covered by this guard): context_manager.py and
pipeline_manager.py. Their non-ASCII is emoji used as semantic UI data (status
circles, the progress bar, suggestion-prefix glyphs, the pipeline ``emoji``
field defaults and icons). Stripping those would change UI output, so they are
not part of HY-01's "comments/docstrings/log strings" prose sweep. They are
tracked separately as a UI-string product decision.
"""

import ast
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root + swept set
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OPTI_ROOT = _PROJECT_ROOT / "opti_oignon"

# The inference/context/routing core swept by HY-01. Three of these
# (tool_executor, consensus, model_warmup) were already accent/emoji-free but
# are included so the guard locks them down too.
_SWEPT_FILES = [
    "executor.py",
    "context_window.py",
    "model_manager.py",
    "context_summary.py",
    "conversation_compressor.py",
    "router.py",
    "learned_router.py",
    "dynamic_planning.py",
    "tool_executor.py",
    "consensus.py",
    "model_warmup.py",
]

# Files intentionally NOT swept (emoji-as-UI-data); kept here so the rationale
# is discoverable and a future contributor does not "fix" them by accident.
_EXCLUDED_EMOJI_AS_DATA = ["context_manager.py", "pipeline_manager.py"]


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

# Pictographic emoji ranges (emoticons, symbols & pictographs, transport,
# supplemental symbols, dingbats, misc symbols, flags, variation selectors).
_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF"
    "\U0000FE00-\U0000FE0F"
    "\U00002300-\U000023FF"
    "]"
)

# A raw-string prefix where ``r`` actually starts a string token (preceded by a
# non-identifier char), so ``logger.info("error")`` does NOT count as a raw
# string but ``re.findall(r"...")`` and a bare ``r"..."`` literal do.
_RAW_STRING_LINE = re.compile(r"(?<![\w])r['\"]")

# Author-name token allowed to carry an accent in a file header.
_ALLOWED_NAME_TOKENS = {"Léon", "León"}
_NONASCII_TOKEN = re.compile(r"\w*[^\x00-\x7f]\w*")


def _emoji_hits(line: str) -> list[str]:
    return _EMOJI.findall(line)


def _non_ascii_offenders(line: str) -> list[str]:
    """Non-ASCII chars on a line that are NOT covered by an allowed exception."""
    if _RAW_STRING_LINE.search(line):
        # Regex pattern lines may carry accented matchers on purpose.
        return []
    bad = [c for c in line if ord(c) > 127]
    if not bad:
        return []
    tokens = _NONASCII_TOKEN.findall(line)
    if tokens and all(tok in _ALLOWED_NAME_TOKENS for tok in tokens):
        return []
    return bad


def _swept_paths() -> list[Path]:
    paths = []
    for name in _SWEPT_FILES:
        p = _OPTI_ROOT / name
        assert p.exists(), f"swept file missing (renamed/moved?): {name}"
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoEmojiInSweptCore:
    """Zero pictographic emoji in the swept inference core."""

    def test_no_emoji(self):
        violations = {}
        for p in _swept_paths():
            hits = []
            for i, line in enumerate(p.read_text("utf-8").splitlines(), 1):
                found = _emoji_hits(line)
                if found:
                    hits.append((i, "".join(found), line.strip()[:90]))
            if hits:
                violations[p.name] = hits
        if violations:
            parts = []
            for fname, hits in violations.items():
                parts.append(f"\n  {fname}:")
                for lineno, glyphs, text in hits[:6]:
                    parts.append(f"    L{lineno}: [{glyphs}] {text}")
            pytest.fail("Emoji found in swept inference core:" + "".join(parts))


class TestNoNonAsciiInSweptCore:
    """Swept files must be pure ASCII apart from regex matchers and the author name."""

    def test_pure_ascii_except_allowed(self):
        violations = {}
        for p in _swept_paths():
            hits = []
            for i, line in enumerate(p.read_text("utf-8").splitlines(), 1):
                offenders = _non_ascii_offenders(line)
                if offenders:
                    hits.append((i, "".join(sorted(set(offenders))), line.strip()[:90]))
            if hits:
                violations[p.name] = hits
        if violations:
            parts = []
            for fname, hits in violations.items():
                parts.append(f"\n  {fname} ({len(hits)} lines):")
                for lineno, chars, text in hits[:8]:
                    parts.append(f"    L{lineno}: ({chars}) {text}")
                if len(hits) > 8:
                    parts.append(f"    ... and {len(hits) - 8} more")
            pytest.fail(
                "Non-ASCII found in swept inference core "
                "(translate prose / normalize symbols to ASCII):" + "".join(parts)
            )


class TestSweptCoreAstValid:
    """Every swept file must still parse."""

    def test_ast_valid(self):
        failures = []
        for p in _swept_paths():
            try:
                ast.parse(p.read_text("utf-8"))
            except SyntaxError as exc:  # pragma: no cover - guard
                failures.append(f"{p.name}: {exc}")
        if failures:
            pytest.fail("AST failures:\n" + "\n".join(f"  {f}" for f in failures))


class TestAllowedExceptionsPresent:
    """
    Pin the intentional exceptions so a future refactor cannot silently drop the
    French matchers (which would degrade French-conversation handling) without
    this guard turning red.
    """

    def test_conversation_compressor_french_matchers_retained(self):
        text = (_OPTI_ROOT / "conversation_compressor.py").read_text("utf-8")
        # Past-reference detectors (accented) and the accented tokenizer class.
        assert "discut" in text and "parl" in text and "tôt" in text, (
            "French past-reference regex matchers appear to have been removed"
        )
        assert "a-zA-ZÀ-ÿ" in text, (
            "Accented tokenizer character class was removed from conversation_compressor"
        )

    def test_tool_executor_french_triggers_retained(self):
        text = (_OPTI_ROOT / "tool_executor.py").read_text("utf-8")
        for phrase in ("lis le fichier", "cherche", "ecris dans", "liste les fichiers"):
            assert phrase in text, f"French natural-language tool trigger lost: {phrase!r}"

    def test_excluded_files_documented(self):
        # The excluded emoji-as-data files still exist; this guard does not touch
        # them by design.
        for name in _EXCLUDED_EMOJI_AS_DATA:
            assert (_OPTI_ROOT / name).exists(), f"excluded file missing: {name}"
