#!/usr/bin/env python3
"""Tiered conversation summary, backed by the archive it summarizes.

The live summarizer keeps one cumulative paragraph and re-merges it on every
overflow. Nothing bounds that paragraph as a conversation grows, and nothing
notices when the ground under it moves: a branch switch or a wiped span
leaves the stored summary narrating messages that no longer exist. This
module gives the summary levels, and gives every level a proof.

Two tiers, from frozen to live:

  * SEGMENT -- a frozen summary of one exact span of archived messages,
    stamped with the ids it covers and a digest of the very bytes it was
    built from.
  * The live partial -- whatever the caller summarizes on the fly, from the
    archived turns, for the span no segment covers yet. It belongs to the
    caller; this module only leaves room for it.

There is no tier above the segments. A summary of summaries restates the
last pass in fresh words and keeps whatever an earlier pass let in, so the
composition does not summarize again: it selects. It keeps the first
segment, where a conversation usually states its task, then the newest
segments that fit its budget, and says how many it left out; their turns
stay in the archive. A record written before this rule may still carry a
rollup: it is read without failing and never composed, and the next advance
drops it.

The archive is the ground truth and it is read-only here: this module never
opens the store itself and never writes anywhere. Reads go through an
injected reader; the updated record is handed BACK to the caller as a
metadata mapping, and whether it is persisted is the caller's decision.

Staleness is refused, not hoped away. On every load the span digests are
recomputed against the archive: a segment whose span changed is dropped,
and what still matches survives byte for byte. An archive that cannot be
read proves nothing, so nothing is composed from tiers and -- just as
important -- nothing is destroyed: refusing to verify must never persist
the refusal.

The budgets live in the ``summary_tiers`` section of ``compression.yaml``,
checked when read; a value that cannot be right is refused by its full name.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# A checkpoint is taken before any apply-type action. Hardcoded on purpose:
# this is a posture of the codebase, not a configuration surface.
checkpoint_before_apply = True

TIERS_METADATA_KEY = "context_summary_tiers"
TIERS_VERSION = 1

_CONFIG = Path(__file__).resolve().parent / "config" / "compression.yaml"


class TierSettingsError(ValueError):
    """A summary-tier setting that cannot be right, named in full."""


@dataclass(frozen=True)
class TierSettings:
    """The ``summary_tiers`` section of ``compression.yaml``, checked."""

    segment_budget_tokens: int
    tail_keep_messages: int
    compose_budget_tokens: int
    compose_share: float

    def compose_bound(self, soft_limit_tokens: int) -> int:
        """The most a composition may take in a window of this soft limit."""
        return max(0, min(self.compose_budget_tokens, int(self.compose_share * soft_limit_tokens)))


def load_tier_settings(path=None) -> TierSettings:
    """Read and check the tier budgets; refuse a bad one by its full name."""
    import yaml

    source = Path(path or _CONFIG)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - any failure to read or build the file is a refusal by name
        raise TierSettingsError(f"summary_tiers: {source.name} cannot be read: {exc}") from exc
    section = raw.get("summary_tiers") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        raise TierSettingsError("summary_tiers: the section is missing or is not a mapping")

    def integer(key, low):
        value = section.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TierSettingsError(f"summary_tiers.{key}: {value!r} is not an integer")
        if value < low:
            raise TierSettingsError(f"summary_tiers.{key}: {value!r} is below {low}")
        return value

    share = section.get("compose_share")
    if isinstance(share, bool) or not isinstance(share, (int, float)):
        raise TierSettingsError(f"summary_tiers.compose_share: {share!r} is not a number")
    if not 0 < share <= 1:
        raise TierSettingsError(f"summary_tiers.compose_share: {share!r} is outside (0, 1]")
    return TierSettings(
        segment_budget_tokens=integer("segment_budget_tokens", 1),
        tail_keep_messages=integer("tail_keep_messages", 0),
        compose_budget_tokens=integer("compose_budget_tokens", 1),
        compose_share=float(share),
    )


# reader(conversation_id) -> ordered user/assistant messages carrying
# ``id``, ``role`` and ``content`` -- or None when the archive cannot be
# read, which is a different statement than an archive that is empty.
ArchiveReader = Callable[[str], "list[dict[str, Any]] | None"]
SummarizeFn = Callable[["list[dict[str, Any]]"], "str | None"]


def _default_estimate(text: str) -> int:
    """The same rough measure the context zones use elsewhere."""
    return max(1, round(len(text) / 3.7))


def span_digest(messages: list[dict[str, Any]]) -> str:
    """A statement about exact bytes: id, role and content, in order.

    Any changed byte, any changed role, any renumbered id and any reordering
    yields a different digest, so a segment can prove that the span it
    summarizes is still the span it was built from.
    """
    lines = [
        f"{m.get('id')}:{m.get('role')}:{m.get('content')}" for m in messages
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class SegmentRecord:
    """One frozen span: the ids it covers, the proof, and the text."""

    first_id: int
    last_id: int
    digest: str
    text: str
    created_at: str = field(default_factory=_now)

    def to_value(self) -> dict[str, Any]:
        return {
            "first_id": self.first_id,
            "last_id": self.last_id,
            "digest": self.digest,
            "text": self.text,
            "created_at": self.created_at,
        }


@dataclass
class ConversationRollup:
    """The conversation tier, naming the segments it was built from."""

    text: str
    built_from: list[str]
    created_at: str = field(default_factory=_now)

    def to_value(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "built_from": list(self.built_from),
            "created_at": self.created_at,
        }


@dataclass
class TierState:
    """The whole record as it rides the conversation metadata."""

    version: int = TIERS_VERSION
    segments: list[SegmentRecord] = field(default_factory=list)
    rollup: ConversationRollup | None = None

    def to_metadata_value(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "segments": [s.to_value() for s in self.segments],
            "rollup": self.rollup.to_value() if self.rollup else None,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> TierState:
        """Parse the record out of conversation metadata, tolerantly.

        An absent key, a foreign version or any malformed shape is an empty
        state: a record this module cannot vouch for contributes nothing,
        and it never raises past this seam.
        """
        try:
            value = (metadata or {}).get(TIERS_METADATA_KEY)
            if not isinstance(value, dict):
                return cls()
            if value.get("version") != TIERS_VERSION:
                return cls()
            segments = []
            for raw in value.get("segments", []):
                segments.append(
                    SegmentRecord(
                        first_id=int(raw["first_id"]),
                        last_id=int(raw["last_id"]),
                        digest=str(raw["digest"]),
                        text=str(raw["text"]),
                        created_at=str(raw.get("created_at", "")),
                    )
                )
            rollup = None
            raw_rollup = value.get("rollup")
            if isinstance(raw_rollup, dict):
                rollup = ConversationRollup(
                    text=str(raw_rollup["text"]),
                    built_from=[str(d) for d in raw_rollup["built_from"]],
                    created_at=str(raw_rollup.get("created_at", "")),
                )
            return cls(segments=segments, rollup=rollup)
        except Exception:
            return cls()


def omission_line(count: int) -> str:
    """The line a composition writes for the segments it leaves out."""
    plural = "s" if count != 1 else ""
    return (
        f"[{count} earlier summary segment{plural} omitted here; "
        "the archive keeps their turns]"
    )


def covered_up_to(state: TierState) -> int:
    """The highest archived message id any segment covers, or zero."""
    if not state.segments:
        return 0
    return max(s.last_id for s in state.segments)


def _default_archive_reader(conversation_id: str) -> list[dict[str, Any]] | None:
    """Read the conversation archive through the project's own API.

    Returns None when the archive cannot be consulted: unreadable is a
    different statement than empty, and the caller must be able to tell
    them apart.
    """
    try:
        from .conversation import conversation_manager

        if conversation_manager is None:
            return None
        messages = conversation_manager.get_messages(conversation_id)
    except Exception as exc:
        logger.debug("Archive unreadable for tier verification: %s", exc)
        return None
    return [
        {"id": m.id, "role": m.role, "content": m.content}
        for m in messages
        if m.role in ("user", "assistant")
    ]


def _default_summarize(messages: list[dict[str, Any]]) -> str | None:
    """Summarize through the live summarizer when one is installed."""
    try:
        from .context_summary import context_summarizer
    except Exception:
        return None
    stripped = [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in messages
    ]
    try:
        return context_summarizer.summarize_messages(messages=stripped)
    except Exception as exc:
        logger.debug("Tier summarization declined: %s", exc)
        return None


class TierManager:
    """Verify, advance and compose the tier record for one conversation."""

    def __init__(
        self,
        archive_reader: ArchiveReader | None = None,
        *,
        segment_budget_tokens: int | None = None,
        tail_keep_messages: int | None = None,
        compose_budget_tokens: int | None = None,
        settings_path=None,
        estimate: Callable[[str], int] | None = None,
    ) -> None:
        """A budget given here wins; any other comes from ``compression.yaml``.

        Settings that cannot be read or checked raise ``TierSettingsError``
        by name: a tier manager never runs on a budget it guessed.
        """
        settings = None
        if None in (segment_budget_tokens, tail_keep_messages, compose_budget_tokens):
            settings = load_tier_settings(settings_path)
        self._reader = archive_reader or _default_archive_reader
        self._segment_budget = max(1, int(
            settings.segment_budget_tokens if segment_budget_tokens is None else segment_budget_tokens
        ))
        self._tail_keep = max(0, int(
            settings.tail_keep_messages if tail_keep_messages is None else tail_keep_messages
        ))
        self._compose_budget = max(1, int(
            settings.compose_budget_tokens if compose_budget_tokens is None else compose_budget_tokens
        ))
        self._estimate = estimate or _default_estimate

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, state: TierState, conversation_id: str) -> TierState:
        """Recompute every span digest against the archive; refuse mismatch.

        An unreadable archive proves nothing: the returned state is empty so
        nothing unverifiable is ever composed. Persisting that emptiness is
        the caller's decision and ``advance`` explicitly refuses to.
        """
        messages = self._reader(conversation_id)
        if messages is None:
            return TierState()
        return self._verify_against(state, messages)

    def _verify_against(
        self, state: TierState, messages: list[dict[str, Any]]
    ) -> TierState:
        by_id = {int(m["id"]): m for m in messages}
        kept: list[SegmentRecord] = []
        for segment in state.segments:
            span = [
                by_id[i]
                for i in range(segment.first_id, segment.last_id + 1)
                if i in by_id
            ]
            if span and span_digest(span) == segment.digest:
                kept.append(segment)
            else:
                logger.info(
                    "Dropping stale summary segment %s-%s: archived span "
                    "no longer matches its digest",
                    segment.first_id,
                    segment.last_id,
                )
        rollup = state.rollup
        if rollup is not None:
            kept_digests = {s.digest for s in kept}
            if not set(rollup.built_from) <= kept_digests:
                logger.info(
                    "Dropping conversation rollup: it stands on a dropped "
                    "segment"
                )
                rollup = None
        return TierState(segments=kept, rollup=rollup)

    # ------------------------------------------------------------------
    # Advancing
    # ------------------------------------------------------------------

    def advance(
        self,
        conversation_id: str,
        metadata: dict[str, Any] | None,
        summarize_fn: SummarizeFn | None = None,
        *,
        up_to_id: int | None = None,
    ) -> dict[str, Any]:
        """Verify the record and freeze what has grown past the budget.

        Freezing is oldest-first over the span no segment covers yet, never
        touches the verbatim tail nor, when ``up_to_id`` is given, any message
        after it, and is fail-safe: a summarizer that declines freezes nothing
        and destroys nothing. Every summarizer input is a run of archived
        turns. The updated record is returned as a metadata mapping for the
        caller to persist; when the archive cannot be read the mapping is
        empty, because refusing to verify must never persist anything.
        """
        messages = self._reader(conversation_id)
        if messages is None:
            return {}
        summarize = summarize_fn or _default_summarize

        state = self._verify_against(
            TierState.from_metadata(metadata), messages
        )
        self._freeze(state, messages, summarize, up_to_id)
        # A rollup read from an older record is never composed; drop it.
        state.rollup = None
        return {TIERS_METADATA_KEY: state.to_metadata_value()}

    def _freeze(
        self,
        state: TierState,
        messages: list[dict[str, Any]],
        summarize: SummarizeFn,
        up_to_id: int | None = None,
    ) -> bool:
        """Freeze budget-sized spans off the uncovered prefix, oldest-first."""
        boundary = len(messages) - self._tail_keep
        if up_to_id is not None:
            boundary = min(
                boundary, sum(1 for m in messages if int(m["id"]) <= up_to_id)
            )
        covered = covered_up_to(state)
        freezable = [
            m for i, m in enumerate(messages)
            if i < boundary and int(m["id"]) > covered
        ]

        frozen_any = False
        span: list[dict[str, Any]] = []
        span_tokens = 0
        for message in freezable:
            span.append(message)
            span_tokens += self._estimate(str(message.get("content", "")))
            if span_tokens < self._segment_budget:
                continue
            text = summarize(list(span))
            if text is None:
                # Fail-safe: nothing is written for a span the summarizer
                # declined, and freezing stops rather than skipping ahead --
                # a gap between segments would break span contiguity.
                return frozen_any
            state.segments.append(
                SegmentRecord(
                    first_id=int(span[0]["id"]),
                    last_id=int(span[-1]["id"]),
                    digest=span_digest(span),
                    text=text,
                )
            )
            frozen_any = True
            span = []
            span_tokens = 0
        return frozen_any

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(
        self,
        conversation_id: str,
        metadata: dict[str, Any] | None,
        live_partial: str = "",
        *,
        up_to_id: int | None = None,
        budget_tokens: int | None = None,
    ) -> str:
        """The injectable block: selected verified segments, then the partial.

        Only segments wholly at or before ``up_to_id`` take part, when it is
        given. The budget is the configured one, or ``budget_tokens`` when the
        caller has less room, and the whole block keeps to it, the line that
        counts the omitted segments included. The first segment stays when it
        fits; then the newest that fit, kept in order; a line says how many
        were left out, their turns kept in the archive. Nothing here is
        summarized again. Tiers that cannot be verified contribute nothing --
        the partial stands alone, exactly as it did before this module existed.
        """
        state = self.verify(TierState.from_metadata(metadata), conversation_id)
        segments = sorted(
            (s for s in state.segments if up_to_id is None or s.last_id <= up_to_id),
            key=lambda s: s.first_id,
        )
        budget = self._compose_budget
        if budget_tokens is not None:
            budget = min(budget, int(budget_tokens))
        blocks: list[str] = []
        if segments:
            # The whole block keeps to the budget: the line that counts what
            # is left out is paid for first, at the most it could cost.
            room = budget - self._estimate(omission_line(len(segments)))
            first, rest = segments[0], segments[1:]
            keep_first = self._estimate(first.text) <= room
            if keep_first:
                room -= self._estimate(first.text)
            newest: list[SegmentRecord] = []
            for segment in reversed(rest):
                cost = self._estimate(segment.text)
                if cost > room:
                    break
                newest.append(segment)
                room -= cost
            newest.reverse()
            omitted = len(segments) - len(newest) - (1 if keep_first else 0)
            if keep_first:
                blocks.append(first.text)
            if omitted:
                blocks.append(omission_line(omitted))
            blocks.extend(s.text for s in newest)
        if live_partial:
            blocks.append(live_partial)
        return "\n\n".join(b for b in blocks if b)
