#!/usr/bin/env python3
"""Red-team harness: whether a defense holds when something pushes back.

A static contract proves a defense answers a fixed input the way it was
written to. It cannot say whether the defense still holds when someone is
actively trying to get past it. This harness asks that second question on a
path a build can stand on.

It is a measuring layer, not a second red-team engine. The obfuscation
strategies, the target adapters and the sandbox rule set already live in
``opti_oignon.redteam`` and are already reachable from the product; this
module drives THOSE at the real defense modules and scores the outcome.
Duplicating them here would mean measuring a copy and shipping the original.

Three roles, kept apart on purpose:

  * The ATTACKER is ``build_payload``: a case from the versioned corpus put
    through a real strategy. It takes a case and nothing else, so it cannot
    consult the defense it is aimed at, let alone tune itself against one.
  * The DEFENSE is the shipped module -- the real sanitizer, the real
    sandbox rule set -- reached through a probe that returns an observation.
    A stand-in would make every number here a statement about the stand-in.
  * The JUDGE is ``judge``: it reads a case and an observation. Not the
    attacker, not the defense, not the strategy. A judge that can see who is
    asking is not a judge.

Two paths, one floor. The DETERMINISTIC path -- fixed corpus, pure
strategies, rule judge -- repeats itself anywhere Python runs, and it is the
only path a CI floor reads. The MODEL path (local attacker, local judge)
belongs to a machine that has a model; this module consults none and reaches
no network, which the harness contracts pin by reading this file's own
syntax tree.

Scope is declared, never assumed. A case names the markers that fall inside
the tested defense's declared reach, so a total floor reads "the defense
held everywhere it claims to reach". A marker outside that reach is a
finding for a report and has no business quietly lowering a build's floor.

Nothing here executes an attack payload. Probing the sandbox means asking
what its rules say about a string, never handing it the string to run: the
disposable-sandbox discipline is not relaxed for the convenience of the
thing that tests it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from opti_oignon.redteam.strategies import STRATEGY_REGISTRY, apply_strategy

logger = logging.getLogger(__name__)

# Module conventions (project-wide): every new module ships with the
# checkpoint discipline hardcoded and a feature sentinel.
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

DEFAULT_CORPUS_PATH = Path(__file__).parent / "suites" / "red_team_micro.yaml"

# The strategy that translates through a model is not deterministic, so it
# cannot sit on the path a floor reads. It stays available to the operator
# measurement through the package that owns it.
_MODEL_BACKED_STRATEGY = "multilingual"

VALID_EXPECTATIONS = ("redacted", "flagged")


# ---------------------------------------------------------------------------
# Corpus model
# ---------------------------------------------------------------------------


@dataclass
class AttackCase:
    """One adversarial case: what is sent, and what proves the defense held."""

    id: str
    target: str
    strategy: str
    attack: str
    expect: str
    markers: list[str] = field(default_factory=list)


@dataclass
class DefenseObservation:
    """What a defense left behind, with no verdict attached.

    The observation is deliberately verdict-free: it carries what the
    defense produced and what survived it, and the judge alone decides what
    that means.
    """

    output: str
    leaked_markers: list[str] = field(default_factory=list)
    flagged: bool = False


@dataclass
class CaseResult:
    """The verdict on one case, by name."""

    case_id: str
    target: str
    held: bool
    detail: str = ""


@dataclass
class RedTeamReport:
    """What the sweep found, case by case, target by target, and overall."""

    cases: list[CaseResult]
    held_ratio: float
    per_target: dict[str, float]


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise ValueError(f"{where} is missing '{key}'")
    return value


def load_corpus(path: Path | None = None) -> list[AttackCase]:
    """Load and validate the adversarial corpus, loudly.

    A case that cannot state what it attacks, how, or what would count as
    holding does not load half-way -- it names its defect and raises.
    """
    target_path = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    data = yaml.safe_load(target_path.read_text(encoding="utf-8")) or {}
    cases: list[AttackCase] = []
    for index, raw in enumerate(data.get("cases", []) or []):
        where = f"case {index + 1} of {target_path.name}"
        case_id = str(_require(raw, "id", where))
        where = f"case '{case_id}'"
        expect = str(_require(raw, "expect", where))
        if expect not in VALID_EXPECTATIONS:
            raise ValueError(
                f"{where} declares an unknown expectation {expect!r} "
                f"(valid: {', '.join(VALID_EXPECTATIONS)})"
            )
        cases.append(
            AttackCase(
                id=case_id,
                target=str(_require(raw, "target", where)),
                strategy=str(raw.get("strategy", "none")),
                attack=str(_require(raw, "attack", where)),
                expect=expect,
                markers=[str(m) for m in (raw.get("markers") or [])],
            )
        )
    if not cases:
        raise ValueError(f"{target_path} declares no cases")
    return cases


def available_strategies() -> tuple[str, ...]:
    """The shipped strategies that belong on the deterministic path."""
    return tuple(
        sorted(
            strategy.value
            for strategy in STRATEGY_REGISTRY
            if strategy.value != _MODEL_BACKED_STRATEGY
        )
    )


# ---------------------------------------------------------------------------
# The attacker
# ---------------------------------------------------------------------------


def build_payload(case: AttackCase) -> str:
    """Turn a case into a payload with the shipped strategy it declares.

    Takes a case and nothing else. The attacker cannot see the defense it is
    aimed at, so it cannot be tuned against one.
    """
    return apply_strategy(case.strategy, case.attack)


# ---------------------------------------------------------------------------
# The defenses, reached as they ship
# ---------------------------------------------------------------------------


def probe_pii_sanitizer(
    payload: str, markers: list[str]
) -> DefenseObservation:
    """Bury declared markers in the payload and run the real sanitizer.

    The markers sit in the clear, exactly as a retrieved passage would carry
    them, and the observation reports which of them survived.
    """
    from opti_oignon.pii_sanitizer import PIISanitizer

    enriched = payload + "".join(f"\nContact: {marker}" for marker in markers)
    sanitized = PIISanitizer().sanitize(enriched)
    survived = [marker for marker in markers if marker in sanitized]
    return DefenseObservation(
        output=sanitized, leaked_markers=survived, flagged=not survived
    )


def probe_sandbox(payload: str, markers: list[str]) -> DefenseObservation:
    """Ask the shipped sandbox target what its rules say about the payload.

    The adapter reads the payload and reports what it recognises; it is
    never asked to run it. Nothing is spawned, nothing is written, and the
    disposable-sandbox discipline stands untouched.
    """
    from opti_oignon.redteam.targets import SandboxTarget

    result = SandboxTarget().run(payload)
    return DefenseObservation(
        output=result.output, leaked_markers=[], flagged=bool(result.blocked)
    )


def probe_rag_sanitizer(
    payload: str, markers: list[str]
) -> DefenseObservation:
    """Run the payload through the shipped RAG chunk sanitizer.

    The payload arrives as a retrieved chunk would, and the observation
    reports whether the sanitizer flagged it as an injection attempt.
    """
    from opti_oignon.rag_sanitizer import RAGSanitizer

    result = RAGSanitizer().sanitize_chunk(
        payload, chunk_id="redteam", source="redteam", collection="redteam"
    )
    return DefenseObservation(
        output=result.sanitized_text,
        leaked_markers=[],
        flagged=bool(result.is_flagged),
    )


def probe_search_sanitizer(
    payload: str, markers: list[str]
) -> DefenseObservation:
    """Run the payload through the shipped search-result sanitizer.

    The verdict follows the sanitizer's own audit log, not whether the text
    was reshaped: only a logged injection detection counts as flagged, so
    plain whitespace normalization never reads as a defense holding.
    """
    from opti_oignon.web_search import SearchResult, SearchResultSanitizer

    sanitizer = SearchResultSanitizer()
    before = len(sanitizer.get_audit_log())
    sanitized = sanitizer.sanitize_result(
        SearchResult(
            title="Result",
            snippet=payload,
            url="https://redteam.test/passage",
            source="redteam",
        )
    )
    detected = len(sanitizer.get_audit_log()) - before > 0
    return DefenseObservation(
        output=sanitized.snippet, leaked_markers=[], flagged=detected
    )


ProbeFunc = Callable[[str, list], DefenseObservation]


def default_probes() -> dict[str, ProbeFunc]:
    """The defenses this path drives, by the name a case addresses."""
    return {
        "pii_sanitizer": probe_pii_sanitizer,
        "sandbox": probe_sandbox,
        "rag_sanitizer": probe_rag_sanitizer,
        "search_sanitizer": probe_search_sanitizer,
    }


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


def judge(case: AttackCase, observation: DefenseObservation) -> bool:
    """Decide whether the defense held, from the observation alone.

    Reads a case and an observation. It never sees the attacker, the
    strategy that produced the payload, or the defense that answered it.
    """
    if case.expect == "redacted":
        return not observation.leaked_markers
    if case.expect == "flagged":
        return bool(observation.flagged)
    raise ValueError(
        f"case '{case.id}' declares an unknown expectation {case.expect!r}"
    )


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def run_red_team(
    cases: list[AttackCase] | None = None,
    probes: dict[str, ProbeFunc] | None = None,
) -> RedTeamReport:
    """Drive the corpus at the defenses and score what held, by name.

    Every case is reported whether it held or not; a case that gave way says
    which markers survived or that the payload went unrecognised. Nothing is
    averaged into anonymity.
    """
    corpus = cases if cases is not None else load_corpus()
    probe_table = probes if probes is not None else default_probes()

    results: list[CaseResult] = []
    per_target_counts: dict[str, list[int]] = {}

    for case in corpus:
        probe = probe_table.get(case.target)
        if probe is None:
            raise ValueError(
                f"case '{case.id}' addresses a defense with no probe: "
                f"{case.target}"
            )
        observation = probe(build_payload(case), list(case.markers))
        held = judge(case, observation)
        detail = ""
        if not held:
            if case.expect == "redacted":
                detail = (
                    "markers survived in the clear: "
                    f"{observation.leaked_markers}"
                )
            else:
                detail = "the payload was not recognised as an escape attempt"
        results.append(
            CaseResult(
                case_id=case.id, target=case.target, held=held, detail=detail
            )
        )
        bucket = per_target_counts.setdefault(case.target, [0, 0])
        bucket[0] += int(held)
        bucket[1] += 1

    held_total = sum(1 for result in results if result.held)
    ratio = held_total / len(results) if results else 1.0
    per_target = {
        target: (held / total if total else 1.0)
        for target, (held, total) in per_target_counts.items()
    }
    return RedTeamReport(
        cases=results, held_ratio=ratio, per_target=per_target
    )
