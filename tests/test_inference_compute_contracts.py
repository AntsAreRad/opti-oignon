#!/usr/bin/env python3
"""Contracts for inference-time compute: candidates sampled through the
registry, verified before one is chosen.

Tokens are cheap and time is cheap on a local machine; what is not cheap is
trusting an answer. The module samples N candidates for one prompt through
the inference registry, hands each to a verifier -- tests run in a sandbox
for code, agreement across the pool for the rest -- and selects the first
that is verified, stopping there. When nothing is verified it says so: the
best-scored candidate is returned as unverified, never dressed up, and with
no score at all the selection is empty and carries the verdicts.

  * IC1 -- candidates come through the registry with their provenance, the
    budget caps the count and the tokens, and no backend is a refusal.
  * IC2 -- the tests oracle reads pass and fail from the runner and never
    invents a verdict: a runner error or a run with no tests is unknown.
  * IC3 -- selection stops at the first verified candidate.
  * IC4 -- with no verified candidate the best score is returned as
    unverified, and with no score at all the selection is empty.
  * IC5 -- the agreement verifier is a strict majority over the pool, and
    a pool of one is no evidence.
  * IC6 -- the budget is the YAML's and an incoherent budget is refused.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window over the registry bridge.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

_YAML = REPO / "opti_oignon" / "config" / "inference_compute.yaml"
_MSGS = [{"role": "user", "content": "write the sieve"}]


class _Scripted:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        i = len(self.calls) - 1
        text = self.replies[i] if i < len(self.replies) else f"reply {i}"
        return {"message": {"content": text}}


def _open(replies=("a", "b", "c", "d", "e")):
    seeded = {}
    scripted = _Scripted(replies)
    seed_registry(seeded, scripted)
    loaded, restore = isolate(
        targets={"opti_oignon.inference_compute": source("inference_compute.py")},
        seeded=seeded,
        packages=("opti_oignon",),
    )
    return loaded["opti_oignon.inference_compute"], scripted, restore


def _budget(mod, **over):
    fields = dict(max_candidates=4, max_total_tokens=400, temperatures=(0.2, 0.7), stop_at_first_pass=True)
    fields.update(over)
    return mod.Budget(**fields)


# ---------------------------------------------------------------------------
# IC1 -- sampling through the registry
# ---------------------------------------------------------------------------
def test_ic1_candidates_come_through_the_registry_with_provenance_and_under_budget():
    mod, scripted, restore = _open(("first", "second", "third"))
    try:
        budget = _budget(mod)
        out = mod.sample_candidates("m", _MSGS, 3, budget=budget)
        assert [c.text for c in out] == ["first", "second", "third"]
        assert [c.index for c in out] == [0, 1, 2]
        assert [c.temperature for c in out] == [0.2, 0.7, 0.2], "temperatures cycle through the budget's list"
        assert [c["options"]["temperature"] for c in scripted.calls] == [0.2, 0.7, 0.2]
        assert all(c["messages"] == _MSGS and c["model"] == "m" for c in scripted.calls)
        assert all(c.tokens >= 1 and c.tokens_source == "estimated" for c in out), "the bridge reports no count"
        reported = SimpleNamespace(content="four words in here", extra={"eval_count": 9})
        backend = SimpleNamespace(generate=lambda **kw: reported)
        one = mod.sample_candidates("m", _MSGS, 1, budget=budget, resolve=lambda model: backend)
        assert one[0].tokens == 9 and one[0].tokens_source == "reported"
        with pytest.raises(mod.ComputeBudgetError):
            mod.sample_candidates("m", _MSGS, 5, budget=budget)
        tight = _budget(mod, max_total_tokens=1)
        mod2, scripted2, restore2 = _open(("one two three", "four five six", "seven"))
        try:
            few = mod2.sample_candidates("m", _MSGS, 3, budget=tight)
            assert len(few) == 1, "the token budget stops sampling after the first candidate crosses it"
            assert len(scripted2.calls) == 1
        finally:
            restore2()
        with pytest.raises(RuntimeError, match="registry"):
            mod.sample_candidates("m", _MSGS, 1, budget=budget, resolve=lambda model: None)
    finally:
        restore()


# ---------------------------------------------------------------------------
# IC2 -- the tests oracle
# ---------------------------------------------------------------------------
def test_ic2_the_tests_oracle_reads_the_runner_and_never_invents_a_verdict():
    mod, scripted, restore = _open()
    try:
        assert mod.parse_pytest_summary("") is None
        assert mod.parse_pytest_summary("3 passed in 0.12s") == (3, 0, 0)
        assert mod.parse_pytest_summary("2 passed, 1 failed, 1 error in 0.3s") == (2, 1, 1)
        assert mod.parse_pytest_summary("1 failed, 4 passed in 1s") == (4, 1, 0)
        runs = []

        def runner(text, command):
            runs.append((text, command))
            return {"good": (0, "3 passed in 0.1s"), "bad": (1, "2 passed, 1 failed in 0.1s"),
                    "empty": (5, "no tests ran in 0.01s"), "boom": None}[text]

        verify = mod.tests_oracle(runner, command="python3 -m pytest -q")
        cand = lambda t: mod.Candidate(index=0, text=t, temperature=0.2, tokens=1, tokens_source="estimated", model="m")  # noqa: E731
        good = verify(cand("good"), [])
        assert good.passed is True and good.score == 1.0 and "3 passed" in good.evidence
        bad = verify(cand("bad"), [])
        assert bad.passed is False and bad.score == round(2 / 3, 4)
        empty = verify(cand("empty"), [])
        assert empty.passed is None and empty.score is None, "no tests ran: unknown, not a pass and not a fail"
        boom = verify(cand("boom"), [])
        assert boom.passed is None and "runner" in boom.evidence
        assert runs[0] == ("good", "python3 -m pytest -q")
    finally:
        restore()


# ---------------------------------------------------------------------------
# IC3 -- stop at the first verified candidate
# ---------------------------------------------------------------------------
def test_ic3_selection_stops_at_the_first_verified_candidate():
    mod, scripted, restore = _open(("bad", "good", "also good"))
    try:
        seen = []

        def verify(candidate, pool):
            seen.append(candidate.text)
            return mod.Verdict(passed=candidate.text.startswith("good"), score=None, evidence=candidate.text)

        selection = mod.best_of("m", _MSGS, verify, n=3, budget=_budget(mod))
        assert selection.verified is True and selection.source == "verified"
        assert selection.candidate.text == "good" and selection.candidate.index == 1
        assert selection.sampled == 2 and len(selection.verdicts) == 2, "the third candidate was never sampled"
        assert seen == ["bad", "good"] and len(scripted.calls) == 2
    finally:
        restore()
    mod, scripted, restore = _open(("bad", "good", "also good"))
    try:
        verify = lambda c, pool: mod.Verdict(passed=c.text == "good", score=None, evidence=c.text)  # noqa: E731
        everything = mod.best_of("m", _MSGS, verify, n=3, budget=_budget(mod, stop_at_first_pass=False))
        assert everything.sampled == 3 and len(scripted.calls) == 3, "without the stop, every candidate is sampled"
        assert everything.verified is True and everything.candidate.text == "good"
    finally:
        restore()


# ---------------------------------------------------------------------------
# IC4 -- nothing verified
# ---------------------------------------------------------------------------
def test_ic4_without_a_verified_candidate_the_best_score_is_unverified_and_no_score_is_empty():
    mod, scripted, restore = _open(("x", "y", "z"))
    try:
        scores = {"x": 0.2, "y": 0.6, "z": 0.4}
        verify = lambda c, pool: mod.Verdict(passed=False, score=scores[c.text], evidence="")  # noqa: E731
        selection = mod.best_of("m", _MSGS, verify, n=3, budget=_budget(mod))
        assert selection.verified is False and selection.source == "unverified-best"
        assert selection.candidate.text == "y" and selection.sampled == 3
        assert "unverified" in selection.reason
        unknown = lambda c, pool: mod.Verdict(passed=None, score=None, evidence="runner down")  # noqa: E731
        mod2, scripted2, restore2 = _open(("x", "y", "z"))
        try:
            empty = mod2.best_of("m", _MSGS, unknown, n=3, budget=_budget(mod2))
            assert empty.candidate is None and empty.verified is False and empty.source == "none"
            assert len(empty.verdicts) == 3 and all(v.passed is None for v in empty.verdicts)
        finally:
            restore2()
    finally:
        restore()


# ---------------------------------------------------------------------------
# IC5 -- agreement
# ---------------------------------------------------------------------------
def test_ic5_agreement_is_a_strict_majority_and_a_pool_of_one_is_no_evidence():
    mod, scripted, restore = _open()
    try:
        cand = lambda i, t: mod.Candidate(index=i, text=t, temperature=0.2, tokens=1, tokens_source="estimated", model="m")  # noqa: E731
        pool = [cand(0, "42"), cand(1, " 42 "), cand(2, "41")]
        verify = mod.agreement()
        top = verify(pool[0], pool)
        assert top.passed is True and top.score == round(2 / 3, 4)
        assert verify(pool[1], pool).passed is True, "whitespace and case do not split a vote"
        low = verify(pool[2], pool)
        assert low.passed is False and low.score == round(1 / 3, 4)
        alone = verify(pool[0], [pool[0]])
        assert alone.passed is None and alone.score is None
        split = [cand(0, "a"), cand(1, "b")]
        assert verify(split[0], split).passed is False, "one of two is not a strict majority"
    finally:
        restore()


# ---------------------------------------------------------------------------
# IC6 -- the budget is the YAML's
# ---------------------------------------------------------------------------
def test_ic6_the_budget_is_read_from_the_yaml_and_an_incoherent_one_is_refused():
    import yaml

    mod, scripted, restore = _open()
    try:
        raw = yaml.safe_load(_YAML.read_text(encoding="utf-8"))["inference_compute"]
        budget = mod.load_budget()
        assert budget.max_candidates == int(raw["max_candidates"]) >= 2
        assert budget.max_total_tokens == int(raw["max_total_tokens"]) >= 1
        assert budget.temperatures == tuple(float(t) for t in raw["temperatures"])
        assert budget.stop_at_first_pass is bool(raw["stop_at_first_pass"])
        assert budget.validate() == []
        assert _budget(mod, max_candidates=0).validate() != []
        assert _budget(mod, temperatures=()).validate() != []
        assert _budget(mod, temperatures=(0.2, 3.0)).validate() != []
        assert _budget(mod, max_total_tokens=0).validate() != []
        with pytest.raises(mod.ComputeBudgetError):
            mod.best_of("m", _MSGS, lambda c, p: mod.Verdict(True, 1.0, ""), n=1, budget=_budget(mod, max_candidates=0))
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
