#!/usr/bin/env python3
"""Tiered conversation summary, backed by the archive it summarizes.

The live summarizer keeps one cumulative paragraph and re-merges it on every
overflow. Nothing bounds that paragraph as a conversation grows, and nothing
notices when the ground under it moves: a branch switch or a wiped span
leaves the stored summary narrating messages that no longer exist. This
module gives the summary levels, and gives every level a proof.

Three tiers, from coarse to live:

  * SEGMENT -- a frozen summary of one exact span of archived messages,
    stamped with the ids it covers and a digest of the very bytes it was
    built from.
  * ROLLUP -- one conversation-level summary built from the segment texts
    ONLY, never from the raw messages a second time, naming the exact
    segments it stands on.
  * The live partial -- whatever the caller still summarizes on the fly for
    the span no segment covers yet. It belongs to the caller; this module
    only leaves room for it.

The archive is the ground truth and it is read-only here: this module never
opens the store itself and never writes anywhere. Reads go through an
injected reader; the updated record is handed BACK to the caller as a
metadata mapping, and whether it is persisted is the caller's decision.

Staleness is refused, not hoped away. On every load the span digests are
recomputed against the archive: a segment whose span changed is dropped, a
rollup standing on a dropped segment goes with it, and what still matches
survives byte for byte. An archive that cannot be read proves nothing, so
nothing is composed from tiers and -- just as important -- nothing is
destroyed: refusing to verify must never persist the refusal.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# A checkpoint is taken before any apply-type action. Hardcoded on purpose:
# this is a posture of the codebase, not a configuration surface.
checkpoint_before_apply = True

TIERS_METADATA_KEY = "context_summary_tiers"
TIERS_VERSION = 1

DEFAULT_SEGMENT_BUDGET_TOKENS = 1200
DEFAULT_TAIL_KEEP_MESSAGES = 4
ROLLUP_MIN_SEGMENTS = 2

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
        segment_budget_tokens: int = DEFAULT_SEGMENT_BUDGET_TOKENS,
        tail_keep_messages: int = DEFAULT_TAIL_KEEP_MESSAGES,
        estimate: Callable[[str], int] | None = None,
    ) -> None:
        self._reader = archive_reader or _default_archive_reader
        self._segment_budget = max(1, int(segment_budget_tokens))
        self._tail_keep = max(0, int(tail_keep_messages))
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
    ) -> dict[str, Any]:
        """Verify the record, freeze what has grown past the budget, roll up.

        Freezing is oldest-first over the span no segment covers yet, never
        touches the verbatim tail, and is fail-safe: a summarizer that
        declines freezes nothing and destroys nothing. The updated record is
        returned as a metadata mapping for the caller to persist; when the
        archive cannot be read the mapping is empty, because refusing to
        verify must never persist anything.
        """
        messages = self._reader(conversation_id)
        if messages is None:
            return {}
        summarize = summarize_fn or _default_summarize

        state = self._verify_against(
            TierState.from_metadata(metadata), messages
        )
        frozen_any = self._freeze(state, messages, summarize)
        self._refresh_rollup(state, summarize, frozen_any)
        return {TIERS_METADATA_KEY: state.to_metadata_value()}

    def _freeze(
        self,
        state: TierState,
        messages: list[dict[str, Any]],
        summarize: SummarizeFn,
    ) -> bool:
        """Freeze budget-sized spans off the uncovered prefix, oldest-first."""
        boundary = len(messages) - self._tail_keep
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

    def _refresh_rollup(
        self, state: TierState, summarize: SummarizeFn, frozen_any: bool
    ) -> None:
        """Rebuild the rollup from segment texts only, when it is due."""
        if len(state.segments) < ROLLUP_MIN_SEGMENTS:
            return
        current = [s.digest for s in state.segments]
        if (
            not frozen_any
            and state.rollup is not None
            and state.rollup.built_from == current
        ):
            return
        text = summarize(
            [{"role": "summary", "content": s.text} for s in state.segments]
        )
        if text is None:
            # A stale rollup is worse than none: keep the previous one only
            # if it still names exactly the segments that exist.
            if state.rollup is not None and state.rollup.built_from != current:
                state.rollup = None
            return
        state.rollup = ConversationRollup(text=text, built_from=current)

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(
        self,
        conversation_id: str,
        metadata: dict[str, Any] | None,
        live_partial: str = "",
    ) -> str:
        """The injectable block: verified rollup first, then the partial.

        Without a rollup the segment texts stand in, oldest first. Tiers
        that cannot be verified contribute nothing -- the partial stands
        alone, exactly as it did before this module existed.
        """
        state = self.verify(TierState.from_metadata(metadata), conversation_id)
        blocks: list[str] = []
        if state.rollup is not None:
            blocks.append(state.rollup.text)
        else:
            blocks.extend(s.text for s in state.segments)
        if live_partial:
            blocks.append(live_partial)
        return "\n\n".join(b for b in blocks if b)
