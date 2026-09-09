#!/usr/bin/env python3
"""Compression-fidelity harness: what a summary keeps, measured.

A summary that "preserves the important facts" is an assertion until
something checks it. This harness declares the facts inside fixture
conversations, drives those conversations through the REAL tier machinery,
and scores retention by keyword presence in the composed block --
deterministic, judge-free, and runnable anywhere Python runs.

Two instruments, two honest scopes:

  * The DETERMINISTIC path uses the extractive instrument below as the
    summarizer. It measures the tier machinery itself -- segmentation,
    rollup provenance, verification, composition -- because the instrument
    is fixed and lossless for declared, digit-bearing facts. This is the
    path a CI floor can stand on: no model, no variance, no excuse.
  * The MODEL path injects the live summarizer instead. Its numbers depend
    on a local model and belong to the operator's machine; nothing here
    gates on them and nothing here downloads anything.

The needle probe is a separate question: after the prompt has moved on, is
a fact buried in the archive still recoverable AT ALL? A nonce is planted
at a chosen depth in a real throwaway conversation store, and the same
archive retriever the product uses is asked to find it again. Recovery is
earned by the retriever or it is not reported.

Local by construction: no network-capable module is imported anywhere in
this file, and no model client is imported at module level.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from opti_oignon.context_summary_tiers import (
    TierManager,
    TierState,
    covered_up_to,
)

logger = logging.getLogger(__name__)

# Module conventions (project-wide): every new module ships with the
# checkpoint discipline hardcoded and a feature sentinel.
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

DEFAULT_FIXTURE_PATH = Path(__file__).parent / "suites" / "fidelity_micro.yaml"
DEFAULT_NONCE = "needle token 734992817364"
DEFAULT_HAYSTACK_SIZES = (12, 40)
DEFAULT_DEPTHS = ("first", "middle", "last")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Fixture model
# ---------------------------------------------------------------------------


@dataclass
class FactSpec:
    """One declared fact: how to ask for it, and what proves it survived."""

    id: str
    probe: str
    keywords: list[str]


@dataclass
class FixtureConversation:
    """One fixture conversation with its measurement parameters."""

    id: str
    segment_budget_tokens: int
    tail_keep_messages: int
    facts: list[FactSpec]
    messages: list[dict[str, Any]]


@dataclass
class FidelityFactResult:
    """The verdict on one declared fact, by name."""

    conversation_id: str
    fact_id: str
    retained: bool
    missing_keywords: list[str] = field(default_factory=list)


@dataclass
class FidelityReport:
    """What the measurement found, fact by fact and overall."""

    conversations: dict[str, float]
    facts: list[FidelityFactResult]
    overall_ratio: float
    composed: dict[str, str]


@dataclass
class NeedleCase:
    """One planted needle: where it was buried, and whether it came back."""

    haystack_size: int
    depth: str
    recovered: bool


@dataclass
class NeedleReport:
    """The sweep as a whole."""

    cases: list[NeedleCase]
    recovery_ratio: float


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    value = mapping.get(key)
    if value is None or value == [] or value == "":
        raise ValueError(f"{where} is missing '{key}'")
    return value


def load_fixture(path: Path | None = None) -> list[FixtureConversation]:
    """Load and validate a fixture file, loudly.

    Every conversation must carry facts and messages; every fact must carry
    a probe and keywords. A fixture that cannot state what it measures does
    not load half-way -- it names its defect and raises.
    """
    target = Path(path) if path is not None else DEFAULT_FIXTURE_PATH
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    conversations: list[FixtureConversation] = []
    for raw in data.get("conversations", []) or []:
        conv_id = str(_require(raw, "id", "a fixture conversation"))
        where = f"conversation '{conv_id}'"
        facts = []
        for raw_fact in _require(raw, "facts", where):
            fact_id = str(_require(raw_fact, "id", f"a fact in {where}"))
            fact_where = f"fact '{fact_id}' in {where}"
            facts.append(
                FactSpec(
                    id=fact_id,
                    probe=str(_require(raw_fact, "probe", fact_where)),
                    keywords=[
                        str(k)
                        for k in _require(raw_fact, "keywords", fact_where)
                    ],
                )
            )
        messages = []
        for index, raw_message in enumerate(_require(raw, "messages", where)):
            message_where = f"message {index + 1} of {where}"
            messages.append(
                {
                    "id": index + 1,
                    "role": str(_require(raw_message, "role", message_where)),
                    "content": str(
                        _require(raw_message, "content", message_where)
                    ),
                }
            )
        conversations.append(
            FixtureConversation(
                id=conv_id,
                segment_budget_tokens=int(
                    raw.get("segment_budget_tokens", 160)
                ),
                tail_keep_messages=int(raw.get("tail_keep_messages", 2)),
                facts=facts,
                messages=messages,
            )
        )
    if not conversations:
        raise ValueError(f"{target} declares no conversations")
    return conversations


# ---------------------------------------------------------------------------
# The deterministic instrument
# ---------------------------------------------------------------------------


def extractive_instrument(messages: list[dict[str, Any]]) -> str | None:
    """A fixed, deterministic summarizer for measurement purposes.

    Keeps every sentence that carries a digit -- declared facts in the
    shipped fixture are digit-bearing on purpose -- and keeps short
    messages whole. Reapplying it to its own output loses nothing, so a
    fact that vanishes between the archive and the composed block vanished
    in the machinery, not in the instrument.
    """
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()
        pieces: list[str] = []
        for line in content.splitlines():
            for sentence in _SENTENCE_SPLIT.split(line):
                sentence = sentence.strip()
                if sentence and any(ch.isdigit() for ch in sentence):
                    pieces.append(sentence)
        if not pieces and content and len(content) < 80:
            pieces = [content]
        if pieces:
            parts.append(f"[{role}] " + " ".join(pieces))
    return "\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Fidelity measurement
# ---------------------------------------------------------------------------


def run_fidelity(
    conversations: list[FixtureConversation] | None = None,
    summarize_fn: Callable[[list[dict[str, Any]]], str | None] | None = None,
) -> FidelityReport:
    """Drive the fixture through the real tier machinery and score it.

    Each conversation is frozen into segments and a rollup by a
    ``TierManager`` reading the fixture as its archive; the live partial is
    the instrument applied to whatever no segment covers. Retention is
    keyword presence in the composed block, case-insensitive, named fact by
    fact -- never averaged into anonymity.
    """
    convs = conversations if conversations is not None else load_fixture()
    instrument = summarize_fn or extractive_instrument

    fact_results: list[FidelityFactResult] = []
    per_conversation: dict[str, float] = {}
    composed_blocks: dict[str, str] = {}

    for conv in convs:
        archived = [dict(m) for m in conv.messages]

        def _reader(_cid: str, _messages=archived):
            return [dict(m) for m in _messages]

        manager = TierManager(
            archive_reader=_reader,
            segment_budget_tokens=conv.segment_budget_tokens,
            tail_keep_messages=conv.tail_keep_messages,
        )
        record = manager.advance(conv.id, {}, summarize_fn=instrument)
        state = TierState.from_metadata(record)
        uncovered = [
            m for m in conv.messages if m["id"] > covered_up_to(state)
        ]
        partial = instrument(uncovered) if uncovered else None
        composed = manager.compose(conv.id, record, live_partial=partial or "")
        composed_blocks[conv.id] = composed
        haystack = composed.casefold()

        retained_here = 0
        for fact in conv.facts:
            missing = [
                keyword
                for keyword in fact.keywords
                if keyword.casefold() not in haystack
            ]
            retained = not missing
            retained_here += int(retained)
            fact_results.append(
                FidelityFactResult(
                    conversation_id=conv.id,
                    fact_id=fact.id,
                    retained=retained,
                    missing_keywords=missing,
                )
            )
        per_conversation[conv.id] = (
            retained_here / len(conv.facts) if conv.facts else 1.0
        )

    total = len(fact_results)
    retained_total = sum(1 for r in fact_results if r.retained)
    overall = retained_total / total if total else 1.0
    return FidelityReport(
        conversations=per_conversation,
        facts=fact_results,
        overall_ratio=overall,
        composed=composed_blocks,
    )


# ---------------------------------------------------------------------------
# Needle probe
# ---------------------------------------------------------------------------

_FILLER_LINES = (
    "The team reviewed the onboarding notes and agreed to tidy the wording.",
    "A quiet afternoon went into rearranging the shared reading list.",
    "Someone asked about the garden layout and the answer was mostly shrugs.",
    "The draft agenda moved around twice before anyone commented on it.",
)


def _plant(manager: Any, size: int, depth: str, nonce: str) -> str:
    """Create one throwaway conversation with the nonce buried at depth."""
    conversation = manager.create_conversation(title=f"haystack {size} {depth}")
    positions = {"first": 0, "middle": size // 2, "last": size - 1}
    target = positions.get(depth, size // 2)
    for index in range(size):
        role = "user" if index % 2 == 0 else "assistant"
        if index == target:
            content = f"For the record, the reference marker is {nonce}."
        else:
            content = _FILLER_LINES[index % len(_FILLER_LINES)]
        manager.add_message(conversation.id, role, content)
    return conversation.id


def probe_needle(manager: Any, conversation_id: str, nonce: str) -> bool:
    """Ask the real archive retriever whether the nonce is recoverable.

    Recovery means a returned snippet actually contains the nonce; an empty
    result or a snippet without it is a miss. The verdict is the
    retriever's, never the probe's.
    """
    from opti_oignon.conversation_compressor import ArchiveRetriever

    results = ArchiveRetriever().retrieve(
        conversation_id, nonce, top_k=3, manager=manager
    )
    return any(nonce in result.snippet for result in results)


def run_needle_sweep(
    store_path: Path,
    *,
    sizes: tuple[int, ...] = DEFAULT_HAYSTACK_SIZES,
    depths: tuple[str, ...] = DEFAULT_DEPTHS,
    nonce: str = DEFAULT_NONCE,
) -> NeedleReport:
    """Plant the nonce at every declared depth and size; count recoveries.

    The store is a real conversation database created under ``store_path``
    and used for nothing else; the shipped data tree is never touched.
    """
    from opti_oignon.conversation import ConversationManager

    store_path = Path(store_path)
    store_path.mkdir(parents=True, exist_ok=True)
    manager = ConversationManager(db_path=store_path / "needle_store.db")

    cases: list[NeedleCase] = []
    for size in sizes:
        for depth in depths:
            conversation_id = _plant(manager, size, depth, nonce)
            recovered = probe_needle(manager, conversation_id, nonce)
            cases.append(
                NeedleCase(haystack_size=size, depth=depth, recovered=recovered)
            )
    ratio = (
        sum(1 for c in cases if c.recovered) / len(cases) if cases else 1.0
    )
    return NeedleReport(cases=cases, recovery_ratio=ratio)
