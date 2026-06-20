#!/usr/bin/env python3
"""S281 -- the citation-extraction live-path wiring (the additive-module staging).

The verification arc so far: the role S267
(``opti_oignon.agent.claim_verification``) checks one (claim, source) pair and
returns a fail-secure verdict; the aggregation S271
(``opti_oignon.agent.claim_aggregation``) folds a list of (claim, source) pairs
into one per-answer verdict; the extractor S273
(``opti_oignon.agent.citation_extraction``) parses a produced answer plus its
ordered sources into exactly those pairs; the route S274
(``opti_oignon.api.routes_citation_verification``) exposes the whole producer
-> aggregation pipeline over HTTP. The S271 and S273 docstrings both record that
the final wiring -- calling ``verify_answer`` on the extracted pairs inside a
live producing path -- is "a later lot". This is the staging for that lot: a
standalone wiring module that composes ``extract_pairs`` and
``make_answer_verifier`` -> ``verify_answer`` behind a flag, fully unit-tested
with an injected seam so ollama is never invoked, leaving the eventual
``loop.py`` call a one-liner that imports and calls it. No producing-path edit
lands this bloc.

The contract under test:

- ``opti_oignon/agent/citation_gate.py`` is the wiring surface. Like the role,
  the aggregation, and the extractor it is NOT a model-reachable tool: a caller
  (a producing path, or a UI, or a later agent step) hands in an answer and its
  sources, never the model's tool calling. It defines no ``ToolSchema`` and
  registers nothing in the agent tool registry, so it grows no schema-count or
  allowlist pin -- the supersession forecast is zero, a clean twin.
- It imports only the standard library plus the three pure upstream agent
  modules (no ``fastapi`` / ``ollama`` / ``sqlite3`` and no backend at module
  load), reaches no network and has no mode gate (it runs identically in Daily
  and Bulbe), and never raises.
- ``run_gate(answer, sources, verify_answer)`` is the pure composition: it calls
  ``extract_pairs`` over the answer and its sources and applies the supplied
  ``verify_answer`` to the extracted pairs, returning a ``CitationGateResult``
  carrying the aggregate verdict, ok, reason, the per-pair results, and the
  extracted pairs (positionally aligned with the results, the S274 result
  shape).
- ``make_citation_gate(model_client=None)`` wires the real ``verify_answer`` via
  ``make_answer_verifier`` and returns ``gate(answer, sources)``;
  ``verify_answer_citations(answer, sources, model_client=None)`` is the
  one-liner the producing path will call.
- ``should_block_answer(result, *, enabled=None)`` is the gating decision behind
  the ``GATE_ANSWERS_ON_PRODUCE`` flag (default off): inert until enabled, and
  even when enabled it blocks only on a positively-refuted ``unsupported``
  verdict, never on the fail-secure ``uncertain``.
- ``checkpoint_before_apply`` is hardcoded True and never overridable;
  ``FEATURE_AVAILABLE`` gates graceful degradation.

Red-before discipline: on the pristine S280 tree (no citation_gate.py) every
source-reading PRESENCE pin and every behavioural pin fails on a bare assert --
the source helper returns an empty string so a presence assertion is a
bare-assert failure, and the behavioural families guard the load via
``_load_gate_or_none`` and assert the module is present so absence is a
bare-assert failure during the call phase, never a collection error. The
negative pins (not-a-tool, no backend import, no SQL, no mode gate, ascii of an
empty source) and the suite-structure pins (this suite parses, is ASCII, avoids
the selection literal) pass by design before and after.

Isolation (the S243 lesson, the S267 / S272 / S273 idiom): the wiring module
loads under its dotted name with light package stubs after the pure upstream
modules (untrusted_context, claim_verification, claim_aggregation,
citation_extraction) are loaded dotted, so the heavy
``opti_oignon.agent.__init__`` (which pulls ollama via ``.loop``) is never
triggered and ollama is never invoked in-container.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "opti_oignon"

GATE_PATH = PKG / "agent" / "citation_gate.py"
EXTRACTION_PATH = PKG / "agent" / "citation_extraction.py"
CLAIM_AGGREGATION_PATH = PKG / "agent" / "claim_aggregation.py"
CLAIM_VERIFICATION_PATH = PKG / "agent" / "claim_verification.py"
UNTRUSTED_PATH = PKG / "agent" / "untrusted_context.py"


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Isolation harness (the S243 lesson, the S267 / S272 / S273 idiom)
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


def _citation_extraction():
    """The real S273 citation_extraction module, dotted."""
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.agent", PKG / "agent")
    return _load_dotted(
        "opti_oignon.agent.citation_extraction", EXTRACTION_PATH
    )


def _load_gate_or_none():
    """Load citation_gate under its dotted name, or None on absence.

    The pure upstream modules are loaded dotted first so the wiring module's
    qualified imports resolve without triggering opti_oignon.agent.__init__. On
    the pristine tree citation_gate is absent and this returns None, so every
    caller fails on its bare ``assert gate is not None`` -- never a collection
    or import error.
    """
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.agent", PKG / "agent")
    try:
        _claim_aggregation()
        _citation_extraction()
    except Exception:
        # Upstream should always load on a real tree; if it does not, the gate
        # cannot load either and callers fail cleanly on their bare assert.
        pass
    name = "opti_oignon.agent.citation_gate"
    try:
        return _load_dotted(name, GATE_PATH)
    except Exception:
        # On the pristine tree exec_module raises after a partial module may be
        # in sys.modules; pop it so a subsequent call re-attempts cleanly rather
        # than returning a poisoned empty module.
        sys.modules.pop(name, None)
        return None


class _RecordingClient:
    """A one-shot client (callable over messages) returning canned text.

    ``text`` may be a single string (repeated for every pair) or a list of
    strings returned in order (the last repeats once exhausted). Every call's
    messages are recorded so a pin can confirm the role wrapped the extracted
    claim and source as untrusted data; ollama is never invoked.
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


class _FakeAggResult:
    """A stand-in for AnswerVerificationResult, for isolating run_gate."""

    def __init__(self, verdict, ok, results, reason) -> None:
        self.verdict = verdict
        self.ok = ok
        self.results = results
        self.reason = reason


class _FakeVerifyAnswer:
    """A recording verify_answer seam, returning a controlled aggregate.

    Records the exact pairs it was handed so a pin can confirm run_gate passed
    the extractor's output straight through, and returns a controlled result so
    a pin can confirm run_gate surfaces the aggregate verbatim.
    """

    def __init__(self, verdict="supported", ok=True, results=None, reason="") -> None:
        self.verdict = verdict
        self.ok = ok
        self.results = results if results is not None else []
        self.reason = reason
        self.received = None

    def __call__(self, pairs):
        self.received = list(pairs)
        return _FakeAggResult(self.verdict, self.ok, list(self.results), self.reason)


# ---------------------------------------------------------------------------
# Family 1 -- module source, negatives, structure
# ---------------------------------------------------------------------------


class TestModuleSource:
    def test_file_exists_and_titled(self):
        assert GATE_PATH.exists(), "opti_oignon/agent/citation_gate.py missing"
        src = _read(GATE_PATH)
        assert "citation" in src.lower()

    def test_discipline_constant(self):
        assert "checkpoint_before_apply = True" in _read(GATE_PATH)

    def test_feature_available_flag(self):
        assert "FEATURE_AVAILABLE" in _read(GATE_PATH)

    def test_gate_activation_flag_default_off_in_source(self):
        # The activation flag is present and defaults off in the source, so a
        # producing path that imports the wiring stays inert until enabled.
        src = _read(GATE_PATH)
        assert "GATE_ANSWERS_ON_PRODUCE = False" in src

    def test_not_a_model_reachable_tool(self):
        src = _read(GATE_PATH)
        assert "ToolSchema" not in src
        assert "ALL_SCHEMAS" not in src
        assert "register_tool" not in src

    def test_no_backend_import_at_load(self):
        # Pure / local: no fastapi, ollama, or sqlite import in the module; the
        # injected client is what reaches a model, supplied by the caller.
        src = _read(GATE_PATH)
        assert "import fastapi" not in src
        assert "from fastapi" not in src
        assert "import ollama" not in src
        assert "from ollama" not in src
        assert "import sqlite3" not in src

    def test_no_direct_sql(self):
        src = _read(GATE_PATH)
        assert ".execute(" not in src
        assert "SELECT " not in src

    def test_no_mode_gate(self):
        # CV-D4: the wiring surface has no egress and no mode gate.
        src = _read(GATE_PATH)
        assert "get_current_mode" not in src
        assert "security_mode" not in src
        assert "mode_provider" not in src
        assert "MODE_DAILY" not in src
        assert "MODE_BULBE" not in src

    def test_pure_ascii_no_decoration(self):
        raw = _read(GATE_PATH)
        assert raw.isascii(), "citation_gate module must be pure ASCII"
        assert "====" not in raw


# ---------------------------------------------------------------------------
# Family 2 -- run_gate composition (the wiring, isolated from the role)
# ---------------------------------------------------------------------------


class TestRunGateComposition:
    def test_run_gate_calls_extract_pairs(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        ext = _citation_extraction()
        answer = "Cats are mammals [1]. Whales are mammals [2]."
        sources = ["Cats nurse their young.", "Whales breathe air."]
        expected = ext.extract_pairs(answer, sources)
        fake = _FakeVerifyAnswer()
        gate.run_gate(answer, sources, fake)
        # run_gate handed verify_answer exactly the extractor's output.
        assert fake.received is not None
        assert [tuple(p) for p in fake.received] == [tuple(p) for p in expected]

    def test_run_gate_surfaces_aggregate_and_pairs(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        ext = _citation_extraction()
        answer = "Cats are mammals [1]. Whales are mammals [2]."
        sources = ["Cats nurse their young.", "Whales breathe air."]
        expected = ext.extract_pairs(answer, sources)
        fake = _FakeVerifyAnswer(verdict="supported", ok=True, reason="")
        result = gate.run_gate(answer, sources, fake)
        # The aggregate is surfaced verbatim, plus the extracted pairs.
        assert result.verdict == "supported"
        assert result.ok is True
        assert result.reason == ""
        assert [tuple(p) for p in result.pairs] == [tuple(p) for p in expected]

    def test_run_gate_surfaces_not_ok_aggregate(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        answer = "Cats are reptiles [1]."
        sources = ["Cats are mammals that nurse their young."]
        fake = _FakeVerifyAnswer(
            verdict="unsupported", ok=False, reason="1 of 1 pair(s) refuted."
        )
        result = gate.run_gate(answer, sources, fake)
        assert result.verdict == "unsupported"
        assert result.ok is False
        assert "refuted" in result.reason

    def test_run_gate_result_to_dict(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        answer = "Cats are mammals [1]."
        sources = ["Cats nurse their young."]
        fake = _FakeVerifyAnswer(verdict="supported", ok=True)
        result = gate.run_gate(answer, sources, fake)
        d = result.to_dict()
        assert d["verdict"] == "supported"
        assert d["ok"] is True
        assert "pairs" in d
        assert isinstance(d["pairs"], list)


# ---------------------------------------------------------------------------
# Family 3 -- factory and one-liner (the full chain under a fake client)
# ---------------------------------------------------------------------------


class TestFactoryAndOneLiner:
    def test_make_citation_gate_full_chain_supported(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        rec = _RecordingClient("SUPPORTED. The source states this.")
        run = gate.make_citation_gate(model_client=rec)
        answer = "Cats are mammals [1]. Whales are mammals [2]."
        sources = ["Cats nurse their young.", "Whales breathe air."]
        result = run(answer, sources)
        assert result.verdict == "supported"
        assert result.ok is True
        assert len(result.results) == 2
        assert len(result.pairs) == 2
        # The role was actually invoked through the injected client per pair.
        assert rec.called is True
        assert len(rec.calls) == 2

    def test_one_liner_matches_factory(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        answer = "Cats are mammals [1]."
        sources = ["Cats nurse their young."]
        rec_a = _RecordingClient("SUPPORTED. Stated.")
        rec_b = _RecordingClient("SUPPORTED. Stated.")
        via_factory = gate.make_citation_gate(model_client=rec_a)(answer, sources)
        via_one_liner = gate.verify_answer_citations(
            answer, sources, model_client=rec_b
        )
        assert via_one_liner.verdict == via_factory.verdict
        assert via_one_liner.ok == via_factory.ok
        assert [tuple(p) for p in via_one_liner.pairs] == [
            tuple(p) for p in via_factory.pairs
        ]

    def test_full_chain_unsupported_aggregates(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        # One pair refuted dominates the aggregate, fail-secure.
        rec = _RecordingClient(
            [
                "SUPPORTED. The source states this.",
                "UNSUPPORTED. The source contradicts this.",
            ]
        )
        answer = "Cats are mammals [1]. Cats are reptiles [2]."
        sources = ["Cats nurse their young.", "Cats are mammals, not reptiles."]
        result = gate.verify_answer_citations(answer, sources, model_client=rec)
        assert result.verdict == "unsupported"
        # ok reflects clean verification (both pairs verified), not the verdict;
        # the unsupported verdict is the fail-secure aggregate of the two pairs.
        assert result.ok is True
        assert len(result.results) == 2


# ---------------------------------------------------------------------------
# Family 4 -- fail-secure
# ---------------------------------------------------------------------------


class TestFailSecure:
    def test_no_citations_is_uncertain_not_supported(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        # No markers means no pairs; the aggregate defaults to uncertain and
        # never asserts support even if the client would have said supported.
        rec = _RecordingClient("SUPPORTED. Stated.")
        result = gate.verify_answer_citations(
            "Cats are mammals.", ["Cats nurse their young."], model_client=rec
        )
        assert result.verdict == "uncertain"
        assert result.ok is False
        assert result.pairs == []
        assert rec.called is False

    def test_none_inputs_do_not_raise(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        result = gate.verify_answer_citations(None, None, model_client=None)
        assert result.verdict == "uncertain"
        assert result.ok is False

    def test_out_of_range_marker_omitted(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        rec = _RecordingClient("SUPPORTED. Stated.")
        # Marker [3] is out of range for a single source; the extractor omits
        # the pair, so no pair is verified and the aggregate is uncertain.
        result = gate.verify_answer_citations(
            "Cats are mammals [3].", ["Cats nurse their young."], model_client=rec
        )
        assert result.pairs == []
        assert result.verdict == "uncertain"
        assert result.ok is False

    def test_uninjected_client_degrades_cleanly(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        # No client: the role reports a clean per-pair failure, so the
        # aggregate is not ok and never asserts support, without raising.
        result = gate.verify_answer_citations(
            "Cats are mammals [1].", ["Cats nurse their young."], model_client=None
        )
        assert result.ok is False
        assert result.verdict != "supported"


# ---------------------------------------------------------------------------
# Family 5 -- the gating decision behind the flag
# ---------------------------------------------------------------------------


class TestGatingPolicy:
    def test_flag_default_off_attribute(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        assert gate.GATE_ANSWERS_ON_PRODUCE is False

    def test_should_block_inert_when_flag_off(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        # Default off: even an unsupported verdict is not blocked.
        result = _FakeAggResult("unsupported", False, [], "refuted")
        assert gate.should_block_answer(result) is False

    def test_should_block_unsupported_when_enabled(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        result = _FakeAggResult("unsupported", False, [], "refuted")
        assert gate.should_block_answer(result, enabled=True) is True

    def test_should_not_block_supported_or_uncertain_when_enabled(self):
        gate = _load_gate_or_none()
        assert gate is not None, "citation_gate did not load"
        # Conservative: only a positively-refuted answer is blocked; an
        # uncertain (including unverifiable) answer is never blocked.
        supported = _FakeAggResult("supported", True, [], "")
        uncertain = _FakeAggResult("uncertain", False, [], "")
        assert gate.should_block_answer(supported, enabled=True) is False
        assert gate.should_block_answer(uncertain, enabled=True) is False


# ---------------------------------------------------------------------------
# Family 6 -- suite structure
# ---------------------------------------------------------------------------


class TestSuiteStructure:
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
        src = _read(GATE_PATH)
        assert src, "citation_gate module missing"
        ast.parse(src)
        assert src.isascii()
