#!/usr/bin/env python3
"""S273 -- the citation-extraction surface (the missing producer half of the
per-answer verification arc).

The verification arc so far: the role S267
(``opti_oignon.agent.claim_verification``) checks one (claim, source) pair and
returns a fail-secure verdict; the aggregation S271
(``opti_oignon.agent.claim_aggregation``) runs a list of (claim, source) pairs
through the role and folds the per-pair verdicts into one per-answer verdict;
the route S272 (``opti_oignon.api.routes_answer_verification``) exposes that
aggregation over HTTP. Every one of those layers consumes a list of
(claim, source) pairs that it is simply handed: the S271 docstring names them
"the kind a citation-extraction step would hand it" and records that extracting
the pairs from a produced answer is "a later lot". This is that producer half:
a standalone module that parses a produced answer plus its ordered retrieved
sources into exactly the (claim, source) pairs the aggregation consumes. The
final wiring -- calling ``verify_answer`` on the extracted pairs inside a live
producing path -- is deferred to a later lot; this module is the parser that lot
will call.

The contract under test (an explicitly-defined, dependency-free citation
contract):

- ``opti_oignon/agent/citation_extraction.py`` is a surface that is NOT a
  model-reachable tool: it is driven by a caller handing in an answer and its
  sources, never by the model's tool calling. Like the role and the
  aggregation it defines no ``ToolSchema`` and registers nothing in the agent
  tool registry, so it grows no schema-count or allowlist pin -- the
  supersession forecast is zero.
- It is pure and deterministic: it imports only the standard library (no
  ``fastapi`` / ``ollama`` / ``sqlite3`` and no backend at module load), reaches
  no network and has no mode gate (it runs identically in Daily and Bulbe), and
  never raises -- malformed input degrades to fewer or zero pairs, never an
  exception.
- The citation contract. The answer carries inline numeric citation markers of
  the form ``[n]`` (1-based) indexing the ordered ``sources`` by position
  (``[1]`` -> ``sources[0]``). A claim is a sentence bearing at least one
  marker; for each cited sentence one (claim, source) pair is emitted per
  distinct in-range marker, where the claim is the sentence text with the
  numeric markers stripped and whitespace collapsed, and the source is the
  indexed source string.
- Fail-closed by omission. An out-of-range marker, a ``[0]`` marker, a
  malformed marker (``[]`` / ``[abc]`` / ``[1.2]``), a resolved source that is
  empty or whitespace, or a claim that is empty after stripping all yield no
  pair -- the extractor never emits a pair against the wrong source or against
  nothing. Pairs are de-duplicated per (claim, source-index) preserving
  first-seen order.
- The output shape composes directly with S271: ``extract_pairs`` returns a
  list of ``(claim, source)`` 2-tuples, exactly what ``make_answer_verifier``'s
  ``verify_answer`` consumes (its ``_coerce_pair`` indexes ``pair[0]`` /
  ``pair[1]``).

``checkpoint_before_apply`` is hardcoded True and never overridable;
``FEATURE_AVAILABLE`` gates graceful degradation.

Red-before discipline: on the pristine S272 tree (no citation_extraction.py)
every source-reading PRESENCE pin and every behavioural pin fails on a bare
assert -- the source helper returns an empty string so a presence assertion is
a bare-assert failure, and the behavioural families guard the load via
``_load_extraction_or_none`` and assert the module is present so absence is a
bare-assert failure during the call phase, never a collection error. The
negative pins (not-a-tool, no backend import, no SQL, no mode gate, ascii of an
empty source) and the suite-structure pins (this suite parses, is ASCII, avoids
the selection literal) pass by design before and after.

Isolation (the S243 lesson, the S267 / S272 idiom): the extraction module
imports only the standard library, so it loads under its dotted name with light
package stubs and no backend chain. The single composition pin additionally
loads the real untrusted_context, claim_verification and claim_aggregation
dotted (so the aggregation resolves) and drives it with an injected recording
client, so ollama is never invoked in-container.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "opti_oignon"

EXTRACTION_PATH = PKG / "agent" / "citation_extraction.py"
CLAIM_AGGREGATION_PATH = PKG / "agent" / "claim_aggregation.py"
CLAIM_VERIFICATION_PATH = PKG / "agent" / "claim_verification.py"
UNTRUSTED_PATH = PKG / "agent" / "untrusted_context.py"
THIS_PATH = Path(__file__).resolve()


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Isolation harness (the S243 lesson, the S267 / S272 idiom)
# ---------------------------------------------------------------------------


def _ensure_pkg(name: str, path: Path) -> None:
    """Ensure ``name`` exists in sys.modules and is package-like.

    Non-destructive: keeps any pre-existing stub object (an earlier suite's),
    only granting it a ``__path__`` so a dotted ``spec_from_file_location`` load
    of a submodule resolves.
    """
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if not hasattr(mod, "__path__"):
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]


def _load_dotted(name: str, path: Path):
    """Load a module under its real dotted name, reusing an existing load."""
    existing = sys.modules.get(name)
    if existing is not None and hasattr(existing, "__file__"):
        return existing
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_extraction_or_none():
    """Load citation_extraction under its dotted name, or None on absence.

    The module imports only the standard library, so it needs no backend chain;
    light package stubs let the dotted load resolve. On the pristine tree the
    module is absent and this returns None, so the caller fails on a bare assert
    -- never a collection or import error.
    """
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.agent", PKG / "agent")
    name = "opti_oignon.agent.citation_extraction"
    try:
        return _load_dotted(name, EXTRACTION_PATH)
    except Exception:
        # On the pristine tree exec_module raises after a partial module may be
        # in sys.modules; pop it so a subsequent call re-attempts cleanly rather
        # than returning a poisoned empty module. Every caller then fails on its
        # bare ``assert ext is not None``.
        sys.modules.pop(name, None)
        return None


def _untrusted():
    """The real (light) untrusted_context module, dotted."""
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.agent", PKG / "agent")
    return _load_dotted("opti_oignon.agent.untrusted_context", UNTRUSTED_PATH)


def _claim_verification():
    """The real S267 claim_verification role module, dotted."""
    _untrusted()
    return _load_dotted(
        "opti_oignon.agent.claim_verification", CLAIM_VERIFICATION_PATH
    )


def _claim_aggregation():
    """The real S271 claim_aggregation module, dotted (chains the role)."""
    _claim_verification()
    return _load_dotted(
        "opti_oignon.agent.claim_aggregation", CLAIM_AGGREGATION_PATH
    )


class _RecordingClient:
    """A one-shot client (callable over messages) returning canned text.

    ``text`` may be a single string (repeated for every pair) or a list of
    strings returned in order (the last repeats once exhausted). Every call's
    messages are recorded so the composition pin can confirm the role wrapped
    the extracted claim and source as untrusted data; ollama is never invoked.
    """

    def __init__(self, text="SUPPORTED. The source states this.") -> None:
        self._texts = [text] if isinstance(text, str) else list(text)
        self.calls: list = []
        self.called = False

    def __call__(self, messages):
        self.called = True
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self._texts) - 1)
        return self._texts[idx]


# ---------------------------------------------------------------------------
# Family 1 -- source / structure
# ---------------------------------------------------------------------------


class TestModuleSource:
    def test_file_exists_and_titled(self):
        assert EXTRACTION_PATH.exists(), (
            "opti_oignon/agent/citation_extraction.py missing"
        )
        src = _read(EXTRACTION_PATH)
        assert "citation" in src.lower()

    def test_discipline_constants(self):
        src = _read(EXTRACTION_PATH)
        assert "checkpoint_before_apply = True" in src

    def test_feature_available_flag(self):
        assert "FEATURE_AVAILABLE" in _read(EXTRACTION_PATH)

    def test_not_a_model_reachable_tool(self):
        src = _read(EXTRACTION_PATH)
        assert "ToolSchema" not in src
        assert "ALL_SCHEMAS" not in src
        assert "register_tool" not in src

    def test_no_backend_import_at_load(self):
        # Pure / local: no fastapi, ollama, or sqlite import in the module.
        src = _read(EXTRACTION_PATH)
        assert "import fastapi" not in src
        assert "from fastapi" not in src
        assert "import ollama" not in src
        assert "from ollama" not in src
        assert "import sqlite3" not in src

    def test_no_direct_sql(self):
        src = _read(EXTRACTION_PATH)
        assert ".execute(" not in src
        assert "SELECT " not in src

    def test_no_mode_gate(self):
        # CV-D4: the extraction surface has no egress and no mode gate.
        src = _read(EXTRACTION_PATH)
        assert "get_current_mode" not in src
        assert "security_mode" not in src
        assert "mode_provider" not in src
        assert "MODE_DAILY" not in src
        assert "MODE_BULBE" not in src

    def test_pure_ascii_no_decoration(self):
        raw = _read(EXTRACTION_PATH)
        assert raw.isascii(), "extraction module must be pure ASCII"
        assert "====" not in raw


# ---------------------------------------------------------------------------
# Family 2 -- sentence segmentation
# ---------------------------------------------------------------------------


class TestSentenceSegmentation:
    def test_single_sentence(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.split_sentences("Cats are mammals.") == ["Cats are mammals."]

    def test_multiple_sentences(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        out = ext.split_sentences("Cats purr. Whales sing! Do birds fly?")
        assert out == ["Cats purr.", "Whales sing!", "Do birds fly?"]

    def test_no_terminator_is_one_sentence(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.split_sentences("a claim without punctuation") == [
            "a claim without punctuation"
        ]

    def test_empty_and_none(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.split_sentences("") == []
        assert ext.split_sentences(None) == []

    def test_whitespace_only(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.split_sentences("   \n\t  ") == []


# ---------------------------------------------------------------------------
# Family 3 -- citation marker parsing
# ---------------------------------------------------------------------------


class TestCitationParsing:
    def test_find_single_index(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.find_citation_indices("a claim [1].") == [1]

    def test_find_multiple_in_order(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.find_citation_indices("a [2] b [1] c [3]") == [2, 1, 3]

    def test_find_deduped(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.find_citation_indices("a [1] b [1] c [1]") == [1]

    def test_ignores_malformed_forms(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        # Non-integer or empty bracket forms are not citation markers; a
        # well-formed [0] and [2] are matched (range filtering is extract_pairs').
        assert ext.find_citation_indices("[abc] [] [1.2]") == []
        assert ext.find_citation_indices("[0] and [2]") == [0, 2]

    def test_strip_markers(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.strip_citation_markers("Cats are mammals [1].") == (
            "Cats are mammals."
        )
        assert ext.strip_citation_markers("claim [1] and more [2].") == (
            "claim and more."
        )

    def test_strip_leaves_other_brackets(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        # Only numeric citation markers are stripped; other bracketed text stays.
        assert ext.strip_citation_markers("see [note] and [1]") == "see [note] and"


# ---------------------------------------------------------------------------
# Family 4 -- extract_pairs core contract
# ---------------------------------------------------------------------------


class TestExtractPairsCore:
    def test_two_cited_sentences_two_pairs(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        answer = "Cats are mammals [1]. Whales are mammals [2]."
        sources = ["Cats nurse their young.", "Whales breathe air."]
        pairs = ext.extract_pairs(answer, sources)
        assert pairs == [
            ("Cats are mammals.", "Cats nurse their young."),
            ("Whales are mammals.", "Whales breathe air."),
        ]

    def test_markers_stripped_from_claim(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        pairs = ext.extract_pairs("The sky is blue [1].", ["the sky appears blue"])
        assert len(pairs) == 1
        claim, source = pairs[0]
        assert "[1]" not in claim
        assert claim == "The sky is blue."
        assert source == "the sky appears blue"

    def test_one_based_resolution(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        # [2] resolves to sources[1], not sources[2].
        pairs = ext.extract_pairs("a claim [2].", ["first", "second", "third"])
        assert pairs == [("a claim.", "second")]

    def test_multi_marker_sentence_one_pair_per_source(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        pairs = ext.extract_pairs("a combined claim [1][2].", ["src one", "src two"])
        assert pairs == [
            ("a combined claim.", "src one"),
            ("a combined claim.", "src two"),
        ]


# ---------------------------------------------------------------------------
# Family 5 -- extract_pairs fail-closed edges (never raises, never wrong source)
# ---------------------------------------------------------------------------


class TestExtractPairsFailClosed:
    def test_empty_answer(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.extract_pairs("", ["a source"]) == []

    def test_empty_sources(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.extract_pairs("a claim [1].", []) == []

    def test_no_markers_yields_no_pairs(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.extract_pairs("an uncited claim.", ["a source"]) == []

    def test_out_of_range_marker_skipped(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        # [3] has no third source -> no pair (fail-closed, never a wrong source).
        assert ext.extract_pairs("a claim [3].", ["one", "two"]) == []

    def test_zero_index_skipped(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        # Markers are 1-based; [0] resolves to nothing.
        assert ext.extract_pairs("a claim [0].", ["one"]) == []

    def test_malformed_markers_ignored(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.extract_pairs("a claim [abc] [] [1.2].", ["one"]) == []

    def test_empty_resolved_source_skipped(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        # A resolved source that is empty/whitespace cannot verify anything.
        assert ext.extract_pairs("a claim [1].", ["   "]) == []

    def test_none_inputs(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        assert ext.extract_pairs(None, ["a source"]) == []
        assert ext.extract_pairs("a claim [1].", None) == []

    def test_dangling_marker_no_claim_text(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        # A marker with no surrounding claim text yields no pair after stripping.
        assert ext.extract_pairs("[1]", ["a source"]) == []

    def test_dedup_repeated_pair(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        pairs = ext.extract_pairs("a claim [1][1].", ["only source"])
        assert pairs == [("a claim.", "only source")]

    def test_never_raises_on_garbage(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        # Non-string answer, non-iterable sources, mixed-type sources: no raise.
        assert ext.extract_pairs(12345, ["s"]) == []
        assert ext.extract_pairs("a claim [1].", 999) == []
        out = ext.extract_pairs("a claim [1] and [2].", [None, 5])
        # source[0] is None (skipped), source[1] is 5 -> "5".
        assert out == [("a claim and.", "5")]


# ---------------------------------------------------------------------------
# Family 6 -- composition with S271 + structure
# ---------------------------------------------------------------------------


class TestCompositionAndStructure:
    def test_extracted_pairs_feed_verify_answer(self):
        ext = _load_extraction_or_none()
        assert ext is not None, "citation_extraction did not load"
        agg = _claim_aggregation()
        # The extractor output is exactly what verify_answer consumes.
        answer = "Cats are mammals [1]. Whales are mammals [2]."
        sources = ["Cats nurse their young.", "Whales breathe air."]
        pairs = ext.extract_pairs(answer, sources)
        assert len(pairs) == 2
        rec = _RecordingClient("SUPPORTED. The source states this.")
        verify_answer = agg.make_answer_verifier(model_client=rec)
        result = verify_answer(pairs)
        assert result.verdict == "supported"
        assert result.ok is True
        assert len(result.results) == 2
        # The role wrapped each extracted claim and source as untrusted data:
        # the trusted instruction is the only system message, the claim rides
        # the user role inside untrusted-data markers.
        assert rec.called is True
        assert len(rec.calls) == 2
        first = rec.calls[0]
        assert isinstance(first, list) and len(first) == 2
        assert first[0]["role"] == "system"
        assert "verification role" in first[0]["content"].lower()
        assert first[1]["role"] == "user"
        assert "untrusted data" in first[1]["content"].lower()
        assert "Cats are mammals." in first[1]["content"]

    def test_suite_parses(self):
        ast.parse(_read(Path(__file__)))

    def test_suite_pure_ascii(self):
        assert _read(Path(__file__)).isascii()

    def test_suite_avoids_selection_literal(self):
        # The canonical selection greps tests for the sandbox-manager literal;
        # this suite must not be swept into that set, so the literal is built in
        # split form here and asserted absent from the raw file.
        literal = "sandbox" + "_" + "manager"
        assert literal not in _read(Path(__file__))

    def test_module_parses_and_ascii(self):
        src = _read(EXTRACTION_PATH)
        assert src, "extraction module missing"
        ast.parse(src)
        assert src.isascii()
