#!/usr/bin/env python3
"""Inference-time compute: sample candidates through the registry, verify
each, choose the first that is verified.

Tokens are cheap on a local machine and so is time; what is not cheap is
trusting an answer. This module spends the cheap thing to earn the dear
one. It samples N candidates for one prompt through the inference registry
-- never the client behind it -- under a budget of candidates and tokens
from ``inference_compute.yaml``, each candidate carrying its provenance:
index, temperature, token count and whether that count was reported by the
backend or estimated. It hands each candidate to a verifier and selects the
first that is verified, stopping there when the budget says to.

Two verifiers are provided. The tests oracle runs a candidate through an
injected sandbox runner -- the real one is the sandbox manager's command
execution, host-only -- and reads the verdict from the return code and the
test summary; a runner error or a run with no tests is unknown, never a
pass and never a fail. Agreement is a strict majority over the pool of
candidates by normalised text, for answers no test can check; a pool of
one is no evidence.

What the selection says is what happened. A verified candidate is marked
verified. When nothing is verified, the best-scored candidate comes back
marked unverified, with the reason; with no score at all the selection is
empty and carries every verdict. Nothing here dresses up a guess.
"""

import re
from dataclasses import dataclass
from pathlib import Path

checkpoint_before_apply = True

_CONFIG = Path(__file__).resolve().parent / "config" / "inference_compute.yaml"
_SUMMARY = re.compile(r"(\d+)\s+(passed|failed|error|errors)\b")


class ComputeBudgetError(ValueError):
    """The budget cannot be spent as asked."""


def estimate_tokens(text):
    """The fallback estimate, the same as the memory's; a reported count replaces it."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


@dataclass(frozen=True)
class Budget:
    max_candidates: int
    max_total_tokens: int
    temperatures: tuple
    stop_at_first_pass: bool

    def validate(self):
        errors = []
        if not isinstance(self.max_candidates, int) or self.max_candidates < 1:
            errors.append(f"max_candidates: {self.max_candidates!r} is not a positive integer")
        if not isinstance(self.max_total_tokens, int) or self.max_total_tokens < 1:
            errors.append(f"max_total_tokens: {self.max_total_tokens!r} is not a positive integer")
        if not self.temperatures:
            errors.append("temperatures: empty; at least one is needed")
        for t in self.temperatures:
            if not isinstance(t, (int, float)) or not 0.0 <= float(t) <= 2.0:
                errors.append(f"temperatures: {t!r} is not within [0, 2]")
        return errors


def load_budget(path=None):
    """The budget of ``inference_compute.yaml``, refused if incoherent."""
    import yaml

    raw = yaml.safe_load(Path(path or _CONFIG).read_text(encoding="utf-8")) or {}
    section = raw.get("inference_compute") or {}
    try:
        budget = Budget(
            max_candidates=int(section["max_candidates"]),
            max_total_tokens=int(section["max_total_tokens"]),
            temperatures=tuple(float(t) for t in section["temperatures"]),
            stop_at_first_pass=bool(section.get("stop_at_first_pass", True)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ComputeBudgetError(f"inference compute budget is incomplete or malformed: {exc!r}") from exc
    errors = budget.validate()
    if errors:
        raise ComputeBudgetError("; ".join(errors))
    return budget


@dataclass(frozen=True)
class Candidate:
    index: int
    text: str
    temperature: float
    tokens: int
    tokens_source: str
    model: str


@dataclass(frozen=True)
class Verdict:
    """What a verifier could say about one candidate. None is unknown, not false."""

    passed: object
    score: object
    evidence: str = ""


@dataclass(frozen=True)
class Selection:
    candidate: object
    verified: bool
    verdicts: tuple
    sampled: int
    reason: str
    source: str


def _resolve_through_registry(model):
    try:
        from opti_oignon.inference_backend import get_backend_registry
    except Exception:  # noqa: BLE001 - absence is an answer
        return None
    try:
        return get_backend_registry().resolve_backend(model)
    except Exception:  # noqa: BLE001 - a broken registry is absence
        return None


def _token_count(response, text):
    extra = getattr(response, "extra", None)
    if isinstance(extra, dict):
        for key in ("eval_count", "completion_tokens"):
            value = extra.get(key)
            if isinstance(value, int) and value >= 0:
                return value, "reported"
    return estimate_tokens(text), "estimated"


def _one(backend, model, messages, index, temperature):
    response = backend.generate(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    text = str(getattr(response, "content", "") or "")
    tokens, source = _token_count(response, text)
    return Candidate(index=index, text=text, temperature=temperature, tokens=tokens,
                     tokens_source=source, model=model)


def sample_candidates(model, messages, n, *, budget=None, resolve=None):
    """``n`` candidates through the registry, fewer if the token budget runs out first."""
    budget = budget or load_budget()
    errors = budget.validate()
    if errors:
        raise ComputeBudgetError("; ".join(errors))
    n = int(n)
    if n < 1 or n > budget.max_candidates:
        raise ComputeBudgetError(f"{n} candidates asked, the budget allows 1 to {budget.max_candidates}")
    backend = (resolve or _resolve_through_registry)(model)
    if backend is None:
        raise RuntimeError(f"no inference backend is registered in the registry for {model!r}; refusing to sample")
    out, spent = [], 0
    for index in range(n):
        temperature = budget.temperatures[index % len(budget.temperatures)]
        candidate = _one(backend, model, messages, index, temperature)
        out.append(candidate)
        spent += candidate.tokens
        if spent >= budget.max_total_tokens:
            break
    return out


def parse_pytest_summary(output):
    """(passed, failed, errors) from a pytest summary line, or None when there is none."""
    counts = {"passed": 0, "failed": 0, "error": 0}
    found = False
    for number, word in _SUMMARY.findall(output or ""):
        found = True
        counts["error" if word.startswith("error") else word] += int(number)
    if not found:
        return None
    return counts["passed"], counts["failed"], counts["error"]


def tests_oracle(run, *, command="python3 -m pytest -q"):
    """A verifier that runs the candidate's tests through ``run(text, command)``.

    ``run`` returns ``(returncode, output)``; the real runner is the sandbox
    manager's command execution on the host. A runner that raises, or a run
    whose output carries no test summary, yields an unknown verdict.
    """

    def verify(candidate, pool):
        try:
            outcome = run(candidate.text, command)
            returncode, output = outcome
        except Exception as exc:  # noqa: BLE001 - the verdict is unknown, not false
            return Verdict(passed=None, score=None, evidence=f"runner error: {exc!r}")
        summary = parse_pytest_summary(str(output or ""))
        if summary is None:
            return Verdict(passed=None, score=None, evidence="no test summary in the output")
        passed, failed, errors = summary
        total = passed + failed + errors
        if total == 0:
            return Verdict(passed=None, score=None, evidence="no tests ran")
        return Verdict(
            passed=(int(returncode) == 0 and failed == 0 and errors == 0 and passed >= 1),
            score=round(passed / total, 4),
            evidence=f"rc={returncode}: {passed} passed, {failed} failed, {errors} errors",
        )

    return verify


def _normalise(text):
    return " ".join(str(text or "").split()).casefold()


def agreement(normalize=None):
    """A verifier by strict majority over the pool; a pool of one is no evidence."""
    normalize = normalize or _normalise

    def verify(candidate, pool):
        pool = list(pool)
        if len(pool) < 2:
            return Verdict(passed=None, score=None, evidence="a pool of one cannot agree with itself")
        mine = normalize(candidate.text)
        same = sum(1 for c in pool if normalize(c.text) == mine)
        share = round(same / len(pool), 4)
        return Verdict(passed=share > 0.5, score=share, evidence=f"{same} of {len(pool)} agree")

    return verify


def best_of(model, messages, verify, *, n=None, budget=None, resolve=None):
    """Sample, verify, choose: the first verified candidate, else the best score, else nothing."""
    budget = budget or load_budget()
    errors = budget.validate()
    if errors:
        raise ComputeBudgetError("; ".join(errors))
    n = budget.max_candidates if n is None else int(n)
    if n < 1 or n > budget.max_candidates:
        raise ComputeBudgetError(f"{n} candidates asked, the budget allows 1 to {budget.max_candidates}")
    backend = (resolve or _resolve_through_registry)(model)
    if backend is None:
        raise RuntimeError(f"no inference backend is registered in the registry for {model!r}; refusing to sample")

    pool, verdicts, spent = [], [], 0
    for index in range(n):
        temperature = budget.temperatures[index % len(budget.temperatures)]
        candidate = _one(backend, model, messages, index, temperature)
        pool.append(candidate)
        spent += candidate.tokens
        verdict = verify(candidate, list(pool))
        verdicts.append(verdict)
        if budget.stop_at_first_pass and verdict.passed is True:
            return Selection(candidate, True, tuple(verdicts), len(pool), "verified by the verifier", "verified")
        if spent >= budget.max_total_tokens:
            break

    for candidate, verdict in zip(pool, verdicts):
        if verdict.passed is True:
            return Selection(candidate, True, tuple(verdicts), len(pool), "verified by the verifier", "verified")
    scored = [(v.score, -c.index, c) for c, v in zip(pool, verdicts) if v.score is not None]
    if scored:
        score, _neg, candidate = max(scored)
        return Selection(
            candidate, False, tuple(verdicts), len(pool),
            f"unverified: no candidate passed; best score {score}", "unverified-best",
        )
    return Selection(None, False, tuple(verdicts), len(pool), "no candidate could be scored", "none")
