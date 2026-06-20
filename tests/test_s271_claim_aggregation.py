#!/usr/bin/env python3
"""S271 -- per-answer claim aggregation over the S267 verification role.

The higher-value continuation of the claim-vs-source verification arc (the role
S267, the route S268, the UI S269, the nav S270): a standalone aggregation
module that takes a list of (claim, source) pairs, runs each one through the
S267 role's injected one-shot seam, and aggregates the per-pair verdicts
fail-secure into a single per-answer verdict. It is a new module added alongside
the verification surface; it edits no pinned module, defines no model-reachable
tool, adds no route, and has no mode gate -- 100% local / Python / Ollama. This
suite is the red-before contract for that lot.

The contract under test (the established S267 idiom):

- ``opti_oignon/agent/claim_aggregation.py`` is NOT a model-reachable tool and
  carries no backend coupling at module load: it imports only the S267 role's
  factory and result type, so it grows no schema-count, frozenset, or allowlist
  pin, and forces no fastapi / ollama chain when exercised by pytest.
- ``aggregate_verdicts(verdicts)`` is a pure function that mirrors the role's
  asymmetry: UNSUPPORTED if any pair is unsupported; otherwise UNCERTAIN if any
  pair is uncertain; SUPPORTED only when every pair is supported; an empty list
  or any unknown verdict defaults to UNCERTAIN, never to SUPPORTED. An answer
  whose every cited claim is unverifiable never asserts support.
- ``make_answer_verifier(model_client=None)`` builds an answer verifier on the
  S267 factory idiom (the same injected one-shot seam). The returned
  ``verify_answer(pairs)`` runs each (claim, source) pair through the role's
  verify (so each pair is wrapped as untrusted data by the role, the
  anti-injection core), aggregates the verdicts fail-secure, reports ``ok`` only
  when every pair verified cleanly and at least one pair was supplied, and
  returns an :class:`AnswerVerificationResult` carrying the aggregate verdict and
  the per-pair results. It never raises.
- ``checkpoint_before_apply`` is hardcoded True and never overridable. The
  module reaches no network and has no mode gate: it runs identically in Daily
  and Bulbe, with no mode provider on the factory.

Red-before discipline: on the pristine S270 tree (no claim_aggregation.py) every
behavioural and source-reading PRESENCE pin fails on a bare assert -- the source
helper returns an empty string so absence is a bare-assert failure, and the
behavioural families guard the load and assert the module is present so absence
is a bare-assert failure during the call phase, never a collection-time import
error. The negative families (no ToolSchema, no backend import, no egress / no
mode gate) and the structure family (this suite parses, is ASCII, avoids the
selection literal) pass by design before and after. The behavioural families
load the target under its dotted name with the real light untrusted_context and
the real claim_verification role registered at their dotted keys, so the
relative import resolves and no fastapi / ollama chain is touched.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "opti_oignon"

AGG_PATH = PKG / "agent" / "claim_aggregation.py"
CLAIM_VERIF_PATH = PKG / "agent" / "claim_verification.py"
UNTRUSTED_PATH = PKG / "agent" / "untrusted_context.py"
THIS_PATH = Path(__file__).resolve()

# The verdict taxonomy the surface aggregates over.
EXPECTED_VERDICTS = ("supported", "unsupported", "uncertain")


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _ensure_pkg(name: str) -> None:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = []  # mark as a package for submodule resolution
        sys.modules[name] = mod


def _load_claim_aggregation():
    """Load claim_aggregation.py in isolation (the S243 / S267 lesson).

    Package-like stubs for opti_oignon / opti_oignon.agent (only when the real
    package is not already imported, so the full sweep is never clobbered), the
    real light untrusted_context and the real claim_verification role registered
    at their dotted keys so the target's relative imports resolve, then the
    target loaded under its dotted name. No fastapi / ollama chain is forced.
    """
    _ensure_pkg("opti_oignon")
    _ensure_pkg("opti_oignon.agent")

    uc_key = "opti_oignon.agent.untrusted_context"
    if uc_key not in sys.modules:
        uc_spec = importlib.util.spec_from_file_location(uc_key, UNTRUSTED_PATH)
        uc_mod = importlib.util.module_from_spec(uc_spec)
        sys.modules[uc_key] = uc_mod
        uc_spec.loader.exec_module(uc_mod)

    cv_key = "opti_oignon.agent.claim_verification"
    if cv_key not in sys.modules:
        cv_spec = importlib.util.spec_from_file_location(cv_key, CLAIM_VERIF_PATH)
        cv_mod = importlib.util.module_from_spec(cv_spec)
        sys.modules[cv_key] = cv_mod
        cv_spec.loader.exec_module(cv_mod)

    agg_key = "opti_oignon.agent.claim_aggregation"
    agg_spec = importlib.util.spec_from_file_location(agg_key, AGG_PATH)
    agg_mod = importlib.util.module_from_spec(agg_spec)
    sys.modules[agg_key] = agg_mod
    agg_spec.loader.exec_module(agg_mod)
    return agg_mod


def _load_or_none():
    """Return the loaded module, or None when the file is absent.

    On the pristine tree the file is absent so this returns None and the
    behavioural families fail on a bare assert (the red-before), never on a
    collection-time import error. Any other load error (a real bug in the
    delivered module) is allowed to propagate so it surfaces clearly.
    """
    if not AGG_PATH.exists():
        return None
    return _load_claim_aggregation()


class _MappingClient:
    """A one-shot model client seam whose reply depends on the claim it sees.

    Records the number of calls. ``rules`` is a list of (needle, reply): the
    first needle found in the wrapped user content selects its reply, else the
    default reply is returned.
    """

    def __init__(self, default: str = "SUPPORTED: ok", rules=None) -> None:
        self.default = default
        self.rules = rules or []
        self.calls = 0
        self.seen = []

    def __call__(self, messages):
        self.calls += 1
        user = ""
        for msg in messages:
            if msg.get("role") == "user":
                user = msg.get("content", "")
        self.seen.append(user)
        for needle, reply in self.rules:
            if needle in user:
                return reply
        return self.default


class _RaisingClient:
    def __call__(self, messages):
        raise RuntimeError("model exploded")


# ---------------------------------------------------------------------------
# Family A -- module source / discipline / the not-a-tool negative
# ---------------------------------------------------------------------------


class TestModuleSource:
    def test_file_exists_and_titled(self):
        src = _read(AGG_PATH)
        assert src != "", "claim_aggregation.py missing"
        assert "aggregation" in src.lower()
        assert len(src) > 2000

    def test_discipline_constants(self):
        src = _read(AGG_PATH)
        assert "checkpoint_before_apply = True" in src
        assert "FEATURE_AVAILABLE = True" in src

    def test_checkpoint_value_true_at_runtime(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        assert mod.checkpoint_before_apply is True
        assert mod.FEATURE_AVAILABLE is True

    def test_not_a_model_reachable_tool(self):
        # Like the S267 role, this surface is caller-driven, not a model tool:
        # it must define no ToolSchema and join no schema set, so it grows no
        # schema-count or frozenset pin. Design-green before and after.
        src = _read(AGG_PATH)
        assert "ToolSchema(" not in src
        assert "ALL_SCHEMAS" not in src

    def test_no_backend_import_at_load(self):
        # Nothing imports the backend at module load (the S243 lesson). Design
        # green before and after: an empty src trivially satisfies this.
        src = _read(AGG_PATH)
        assert "import fastapi" not in src
        assert "from fastapi" not in src
        assert "import ollama" not in src

    def test_builds_on_the_role_factory(self):
        # The module composes the S267 role rather than reimplementing it.
        src = _read(AGG_PATH)
        assert "make_claim_verifier" in src
        assert "claim_verification" in src


# ---------------------------------------------------------------------------
# Family B -- aggregate_verdicts: the pure fail-secure aggregation
# ---------------------------------------------------------------------------


class TestAggregateVerdicts:
    def test_empty_is_uncertain_failsecure(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        # No pairs verified: nothing is asserted supported.
        assert mod.aggregate_verdicts([]) == "uncertain"

    def test_all_supported_is_supported(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        assert mod.aggregate_verdicts(["supported", "supported"]) == "supported"
        assert mod.aggregate_verdicts(["supported"]) == "supported"

    def test_any_unsupported_dominates(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        assert mod.aggregate_verdicts(["supported", "unsupported"]) == "unsupported"
        assert mod.aggregate_verdicts(["unsupported", "supported", "supported"]) == "unsupported"

    def test_unsupported_beats_uncertain(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        assert mod.aggregate_verdicts(["uncertain", "unsupported"]) == "unsupported"
        assert mod.aggregate_verdicts(["unsupported", "uncertain", "supported"]) == "unsupported"

    def test_uncertain_over_supported_when_no_unsupported(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        assert mod.aggregate_verdicts(["supported", "uncertain"]) == "uncertain"
        assert mod.aggregate_verdicts(["uncertain", "supported", "supported"]) == "uncertain"

    def test_unknown_verdict_is_uncertain_never_supported(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        # A junk verdict with no unsupported present is fail-secure to uncertain.
        assert mod.aggregate_verdicts(["supported", "garbage"]) == "uncertain"
        assert mod.aggregate_verdicts(["garbage"]) != "supported"


# ---------------------------------------------------------------------------
# Family C -- the AnswerVerificationResult shape
# ---------------------------------------------------------------------------


class TestResultShape:
    def test_result_fields(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        res = mod.AnswerVerificationResult(verdict="supported", ok=True)
        assert res.verdict == "supported"
        assert res.ok is True
        assert hasattr(res, "results")
        assert hasattr(res, "reason")

    def test_result_to_dict(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        res = mod.AnswerVerificationResult(verdict="uncertain", ok=False, reason="x")
        d = res.to_dict()
        assert d["verdict"] == "uncertain"
        assert d["ok"] is False
        assert d["reason"] == "x"
        assert isinstance(d["results"], list)


# ---------------------------------------------------------------------------
# Family D -- the answer verifier runner (injected seam, fail-secure, no raise)
# ---------------------------------------------------------------------------


class TestAnswerVerifier:
    def test_empty_pairs_clean_failure(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        verify_answer = mod.make_answer_verifier(_MappingClient("SUPPORTED"))
        res = verify_answer([])
        assert res.ok is False
        assert res.verdict == "uncertain"
        assert res.results == []
        assert "pair" in res.reason.lower()

    def test_no_client_clean_failure(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        verify_answer = mod.make_answer_verifier()
        res = verify_answer([("a claim", "a source")])
        assert res.ok is False
        assert res.verdict == "uncertain"
        assert len(res.results) == 1

    def test_all_supported_answer(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        client = _MappingClient("SUPPORTED: the source says so.")
        verify_answer = mod.make_answer_verifier(client)
        res = verify_answer([("c1", "s1"), ("c2", "s2")])
        assert res.ok is True
        assert res.verdict == "supported"
        assert len(res.results) == 2
        assert client.calls == 2

    def test_one_unsupported_makes_answer_unsupported(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        client = _MappingClient(
            default="SUPPORTED: yes",
            rules=[("BADCLAIM", "UNSUPPORTED: not in the source")],
        )
        verify_answer = mod.make_answer_verifier(client)
        res = verify_answer([("GOODCLAIM", "s"), ("BADCLAIM", "s")])
        assert res.ok is True
        assert res.verdict == "unsupported"

    def test_uncertain_pair_pulls_answer_to_uncertain(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        client = _MappingClient(
            default="SUPPORTED: yes",
            rules=[("MEH", "UNCERTAIN: the source is unclear")],
        )
        verify_answer = mod.make_answer_verifier(client)
        res = verify_answer([("GOODCLAIM", "s"), ("MEH", "s")])
        assert res.ok is True
        assert res.verdict == "uncertain"

    def test_unverifiable_pair_blocks_support_and_marks_not_ok(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        client = _MappingClient("SUPPORTED: yes")
        verify_answer = mod.make_answer_verifier(client)
        # An empty source makes the role refuse that pair (ok False, uncertain);
        # the aggregate cannot be supported and is not clean-ok.
        res = verify_answer([("good claim", "good source"), ("orphan claim", "")])
        assert res.ok is False
        assert res.verdict != "supported"
        assert res.verdict == "uncertain"

    def test_raising_client_clean_failure_never_raises(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        verify_answer = mod.make_answer_verifier(_RaisingClient())
        res = verify_answer([("c1", "s1"), ("c2", "s2")])
        assert res.ok is False
        assert res.verdict == "uncertain"
        assert len(res.results) == 2

    def test_each_pair_wrapped_untrusted_by_the_role(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        client = _MappingClient("SUPPORTED")
        verify_answer = mod.make_answer_verifier(client)
        verify_answer([("CLAIMTOKEN", "SOURCETOKEN")])
        assert len(client.seen) == 1
        user = client.seen[0]
        # The role wraps both pieces as untrusted data under one policy header.
        assert "<untrusted_data" in user
        assert "CLAIMTOKEN" in user
        assert "SOURCETOKEN" in user


# ---------------------------------------------------------------------------
# Family E -- no egress, no mode gate (runs identically Daily and Bulbe)
# ---------------------------------------------------------------------------


class TestNoEgressModeIndependence:
    def test_no_network_or_mode_gate_in_source(self):
        # The aggregation reaches no network and has no mode gate. Design-green
        # before and after: an empty src trivially satisfies these absences.
        src = _read(AGG_PATH)
        assert "NETWORK_TOOLS" not in src
        assert "web_search" not in src
        assert "security_mode" not in src
        assert "MODE_DAILY" not in src
        assert "MODE_BULBE" not in src
        assert "get_current_mode" not in src

    def test_factory_has_no_mode_provider_param(self):
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        sig = inspect.signature(mod.make_answer_verifier)
        assert "mode_provider" not in sig.parameters
        assert "mode" not in sig.parameters

    def test_runs_without_any_mode(self):
        # With no mode plumbing, an answer verification succeeds purely on its
        # injected seam, regardless of any ambient security mode.
        mod = _load_or_none()
        assert mod is not None, "claim_aggregation.py missing"
        verify_answer = mod.make_answer_verifier(_MappingClient("SUPPORTED"))
        res = verify_answer([("c", "s")])
        assert res.ok is True
        assert res.verdict == "supported"


# ---------------------------------------------------------------------------
# Family F -- structure (this suite parses, is ASCII, avoids the selection literal)
# ---------------------------------------------------------------------------


class TestStructure:
    def test_suite_parses(self):
        src = _read(THIS_PATH)
        assert src != ""
        ast.parse(src)

    def test_suite_pure_ascii(self):
        raw = THIS_PATH.read_bytes()
        assert all(b < 128 for b in raw), "non-ASCII byte in the suite"

    def test_suite_avoids_selection_literal(self):
        # The canonical selection's component 1 greps tests/*.py for the
        # sandbox-manager literal; this suite must not contain it (built here
        # only in split form) so the raw grep count stays unchanged.
        needle = "sandbox" + "_manager"
        assert needle not in _read(THIS_PATH)

    def test_module_parses_and_ascii(self):
        src = _read(AGG_PATH)
        assert src != "", "claim_aggregation.py missing"
        ast.parse(src)
        raw = AGG_PATH.read_bytes()
        assert all(b < 128 for b in raw), "non-ASCII byte in the module"
