#!/usr/bin/env python3
"""
FINE-TUNE DATA EXPORT -- Training Dataset Generation (S96)
===========================================================

Exports conversation data as training datasets for fine-tuning local
models. Supports ShareGPT JSON, Alpaca JSON, and JSONL formats with
filtering by conversation, date range, model, and quality score.

Includes a quality scoring engine that combines user feedback (thumbs
up/down) with benchmark scores to rank conversations.

Author: Leon
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# FT-04 (S194): guard the yaml import so a missing PyYAML degrades the
# module instead of breaking its import (VL-02 sibling-consistency class).
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - PyYAML is a core dependency
    yaml = None  # type: ignore[assignment]
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "fine_tune.yaml"

FORMAT_SHAREGPT = "sharegpt"
FORMAT_ALPACA = "alpaca"
FORMAT_JSONL = "jsonl"
VALID_FORMATS = {FORMAT_SHAREGPT, FORMAT_ALPACA, FORMAT_JSONL}

# Role mapping for export formats
ROLE_MAP_SHAREGPT = {"user": "human", "assistant": "gpt", "system": "system"}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExportFilter:
    """Filters for conversation export."""

    conversation_ids: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    model: str | None = None
    min_quality: float = 0.0
    min_turns: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "conversation_ids": self.conversation_ids,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "model": self.model,
            "min_quality": self.min_quality,
            "min_turns": self.min_turns,
        }


@dataclass
class QualityScore:
    """Quality score for a conversation."""

    conversation_id: str = ""
    feedback_score: float = 0.0
    benchmark_score: float = 0.0
    combined_score: float = 0.0
    feedback_count: int = 0
    has_feedback: bool = False
    has_benchmarks: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "feedback_score": round(self.feedback_score, 4),
            "benchmark_score": round(self.benchmark_score, 4),
            "combined_score": round(self.combined_score, 4),
            "feedback_count": self.feedback_count,
            "has_feedback": self.has_feedback,
            "has_benchmarks": self.has_benchmarks,
        }


@dataclass
class ExportResult:
    """Result of an export operation."""

    format: str = ""
    conversation_count: int = 0
    message_count: int = 0
    data: str = ""
    timestamp: str = ""
    filters_applied: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (without raw data for API responses)."""
        return {
            "format": self.format,
            "conversation_count": self.conversation_count,
            "message_count": self.message_count,
            "data_size_bytes": len(self.data.encode("utf-8")) if self.data else 0,
            "timestamp": self.timestamp,
            "filters_applied": self.filters_applied,
        }


# =============================================================================
# QUALITY SCORER
# =============================================================================

class QualityScorer:
    """Scores conversations based on feedback and benchmark data.

    Combines user feedback (thumbs up/down ratio) with benchmark
    performance scores using configurable weights.
    """

    def __init__(
        self,
        feedback_weight: float = 0.6,
        benchmark_weight: float = 0.4,
        default_score: float = 0.5,
        min_feedback_count: int = 1,
    ):
        self._feedback_weight = feedback_weight
        self._benchmark_weight = benchmark_weight
        self._default_score = default_score
        self._min_feedback_count = min_feedback_count

    @property
    def feedback_weight(self) -> float:
        return self._feedback_weight

    @property
    def benchmark_weight(self) -> float:
        return self._benchmark_weight

    @property
    def default_score(self) -> float:
        return self._default_score

    def score_conversation(
        self,
        conversation_id: str,
        feedback_entries: list[dict[str, Any]] | None = None,
        benchmark_scores: list[float] | None = None,
    ) -> QualityScore:
        """Compute quality score for a single conversation.

        Args:
            conversation_id: Conversation UUID.
            feedback_entries: List of feedback dicts with rating_type and rating_value.
            benchmark_scores: List of normalized benchmark scores (0.0-1.0).

        Returns:
            QualityScore with combined score.
        """
        qs = QualityScore(conversation_id=conversation_id)

        # Feedback scoring
        if feedback_entries and len(feedback_entries) >= self._min_feedback_count:
            qs.has_feedback = True
            qs.feedback_count = len(feedback_entries)
            qs.feedback_score = self._compute_feedback_score(feedback_entries)
        else:
            qs.feedback_score = self._default_score

        # Benchmark scoring
        if benchmark_scores:
            qs.has_benchmarks = True
            qs.benchmark_score = sum(benchmark_scores) / len(benchmark_scores)
        else:
            qs.benchmark_score = self._default_score

        # Combined score with weights
        if qs.has_feedback and qs.has_benchmarks:
            qs.combined_score = (
                self._feedback_weight * qs.feedback_score
                + self._benchmark_weight * qs.benchmark_score
            )
        elif qs.has_feedback:
            qs.combined_score = qs.feedback_score
        elif qs.has_benchmarks:
            qs.combined_score = qs.benchmark_score
        else:
            qs.combined_score = self._default_score

        return qs

    def _compute_feedback_score(
        self, entries: list[dict[str, Any]]
    ) -> float:
        """Compute normalized score from feedback entries.

        Thumbs: up=1.0, down=0.0.
        Stars: normalized to 0.0-1.0 range.
        """
        if not entries:
            return self._default_score

        scores: list[float] = []
        for entry in entries:
            rating_type = entry.get("rating_type", "thumbs")
            rating_value = entry.get("rating_value", 1)

            if rating_type == "thumbs":
                scores.append(float(rating_value))
            elif rating_type == "stars":
                scores.append((rating_value - 1) / 4.0)

        return sum(scores) / len(scores) if scores else self._default_score


# =============================================================================
# FINE-TUNE EXPORTER
# =============================================================================

class FineTuneExporter:
    """Exports conversation data as training datasets for fine-tuning.

    Supports ShareGPT JSON, Alpaca JSON, and JSONL formats. Integrates
    with the conversation manager and feedback store for data retrieval
    and quality scoring.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        conversation_manager: Any | None = None,
        feedback_store: Any | None = None,
    ):
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._config: dict[str, Any] = {}
        self._conversation_manager = conversation_manager
        self._feedback_store = feedback_store
        self._scorer: QualityScorer | None = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if YAML_AVAILABLE and self._config_path.exists():
                with open(self._config_path, encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to load fine-tune config: %s", exc)
            self._config = {}

        # Initialize quality scorer from config
        quality_cfg = self._config.get("quality", {})
        self._scorer = QualityScorer(
            feedback_weight=quality_cfg.get("feedback_weight", 0.6),
            benchmark_weight=quality_cfg.get("benchmark_weight", 0.4),
            default_score=quality_cfg.get("default_score", 0.5),
            min_feedback_count=quality_cfg.get("min_feedback_count", 1),
        )

    @property
    def config(self) -> dict[str, Any]:
        """Return current configuration."""
        return dict(self._config)

    @property
    def scorer(self) -> QualityScorer:
        """Return the quality scorer instance."""
        return self._scorer

    @property
    def default_format(self) -> str:
        """Default export format from config."""
        return self._config.get("export", {}).get("default_format", FORMAT_SHAREGPT)

    @property
    def include_system_messages(self) -> bool:
        """Whether to include system messages in exports."""
        return self._config.get("export", {}).get("include_system_messages", True)

    @property
    def strip_whitespace(self) -> bool:
        """Whether to strip whitespace from messages."""
        return self._config.get("export", {}).get("strip_whitespace", True)

    def export(
        self,
        fmt: str | None = None,
        filters: ExportFilter | None = None,
    ) -> ExportResult:
        """Export conversations as training data.

        Args:
            fmt: Export format (sharegpt, alpaca, jsonl). Uses default if None.
            filters: Export filters. Uses defaults if None.

        Returns:
            ExportResult with serialized data.

        Raises:
            ValueError: If format is invalid.
        """
        fmt = fmt or self.default_format
        if fmt not in VALID_FORMATS:
            raise ValueError(
                f"Invalid export format '{fmt}'. Must be one of: {', '.join(sorted(VALID_FORMATS))}"
            )

        filters = filters or ExportFilter()
        conversations = self._fetch_conversations(filters)

        result = ExportResult(
            format=fmt,
            timestamp=datetime.utcnow().isoformat() + "Z",
            filters_applied=filters.to_dict(),
        )

        if not conversations:
            result.data = "[]" if fmt != FORMAT_JSONL else ""
            return result

        if fmt == FORMAT_SHAREGPT:
            result.data, result.message_count = self._format_sharegpt(conversations)
        elif fmt == FORMAT_ALPACA:
            result.data, result.message_count = self._format_alpaca(conversations)
        elif fmt == FORMAT_JSONL:
            result.data, result.message_count = self._format_jsonl(conversations)

        result.conversation_count = len(conversations)
        return result

    def preview(
        self,
        fmt: str | None = None,
        filters: ExportFilter | None = None,
        max_preview: int = 3,
    ) -> dict[str, Any]:
        """Preview export without generating full data.

        Returns conversation count, sample entries, and quality scores.
        """
        fmt = fmt or self.default_format
        filters = filters or ExportFilter()
        conversations = self._fetch_conversations(filters)

        # Score each conversation
        scores = []
        for conv in conversations:
            score = self._score_conversation(conv)
            scores.append(score.to_dict())

        # Generate sample from first max_preview conversations
        sample_convs = conversations[:max_preview]
        sample_result = ExportResult(format=fmt)

        if sample_convs:
            if fmt == FORMAT_SHAREGPT:
                sample_result.data, sample_result.message_count = self._format_sharegpt(sample_convs)
            elif fmt == FORMAT_ALPACA:
                sample_result.data, sample_result.message_count = self._format_alpaca(sample_convs)
            elif fmt == FORMAT_JSONL:
                sample_result.data, sample_result.message_count = self._format_jsonl(sample_convs)

        return {
            "total_conversations": len(conversations),
            "total_messages": sum(len(c.get("messages", [])) for c in conversations),
            "format": fmt,
            "sample_data": sample_result.data,
            "sample_count": len(sample_convs),
            "quality_scores": scores,
            "filters": filters.to_dict(),
        }

    def get_quality_scores(
        self,
        conversation_ids: list[str] | None = None,
        limit: int = 50,
    ) -> list[QualityScore]:
        """Get quality scores for conversations.

        Args:
            conversation_ids: Specific IDs to score. If None, scores recent conversations.
            limit: Maximum number to return.

        Returns:
            List of QualityScore objects sorted by combined_score descending.
        """
        conversations = self._fetch_all_conversations(conversation_ids, limit)
        scores = [self._score_conversation(conv) for conv in conversations]
        scores.sort(key=lambda s: s.combined_score, reverse=True)
        return scores

    # =========================================================================
    # FORMAT METHODS
    # =========================================================================

    def _format_sharegpt(
        self, conversations: list[dict[str, Any]]
    ) -> tuple[str, int]:
        """Format conversations as ShareGPT JSON.

        ShareGPT format:
        [
          {
            "conversations": [
              {"from": "human", "value": "..."},
              {"from": "gpt", "value": "..."}
            ]
          }
        ]
        """
        output = []
        total_msgs = 0

        for conv in conversations:
            messages = conv.get("messages", [])
            turns = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if not self.include_system_messages and role == "system":
                    continue

                mapped_role = ROLE_MAP_SHAREGPT.get(role, role)
                if self.strip_whitespace:
                    content = content.strip()

                if content:
                    turns.append({"from": mapped_role, "value": content})
                    total_msgs += 1

            if turns:
                entry: dict[str, Any] = {"conversations": turns}
                if conv.get("id"):
                    entry["id"] = conv["id"]
                output.append(entry)

        return json.dumps(output, ensure_ascii=False, indent=2), total_msgs

    def _format_alpaca(
        self, conversations: list[dict[str, Any]]
    ) -> tuple[str, int]:
        """Format conversations as Alpaca JSON.

        Alpaca format (one entry per user-assistant pair):
        [
          {
            "instruction": "user message",
            "input": "",
            "output": "assistant response"
          }
        ]

        System messages are prepended to the instruction if present.
        """
        output = []
        total_msgs = 0

        for conv in conversations:
            messages = conv.get("messages", [])
            system_context = ""
            i = 0

            # Extract system message if present
            if messages and messages[0].get("role") == "system":
                if self.include_system_messages:
                    system_content = messages[0].get("content", "")
                    if self.strip_whitespace:
                        system_content = system_content.strip()
                    if system_content:
                        system_context = system_content
                i = 1

            # Pair user/assistant messages
            while i < len(messages) - 1:
                user_msg = messages[i]
                asst_msg = messages[i + 1]

                if (
                    user_msg.get("role") == "user"
                    and asst_msg.get("role") == "assistant"
                ):
                    instruction = user_msg.get("content", "")
                    response = asst_msg.get("content", "")

                    if self.strip_whitespace:
                        instruction = instruction.strip()
                        response = response.strip()

                    if instruction and response:
                        entry: dict[str, str] = {
                            "instruction": instruction,
                            "input": system_context,
                            "output": response,
                        }
                        output.append(entry)
                        total_msgs += 2

                    i += 2
                else:
                    i += 1

        return json.dumps(output, ensure_ascii=False, indent=2), total_msgs

    def _format_jsonl(
        self, conversations: list[dict[str, Any]]
    ) -> tuple[str, int]:
        """Format conversations as JSONL (one conversation per line).

        JSONL format:
        {"messages": [{"role": "user", "content": "..."}, ...]}
        """
        lines = []
        total_msgs = 0

        for conv in conversations:
            messages = conv.get("messages", [])
            filtered = []

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if not self.include_system_messages and role == "system":
                    continue

                if self.strip_whitespace:
                    content = content.strip()

                if content:
                    filtered.append({"role": role, "content": content})
                    total_msgs += 1

            if filtered:
                entry: dict[str, Any] = {"messages": filtered}
                if conv.get("id"):
                    entry["id"] = conv["id"]
                lines.append(json.dumps(entry, ensure_ascii=False))

        return "\n".join(lines), total_msgs

    # =========================================================================
    # DATA RETRIEVAL
    # =========================================================================

    def _fetch_conversations(
        self, filters: ExportFilter
    ) -> list[dict[str, Any]]:
        """Fetch and filter conversations from the conversation manager."""
        if self._conversation_manager is None:
            logger.warning("No conversation manager available for export")
            return []

        # FT-02 (S194): the manager passes `limit` straight into SQL
        # (LIMIT ? OFFSET ?), and SQLite returns zero rows for LIMIT 0,
        # so "limit=0 means all" never held. Page through the store in
        # fixed-size chunks until exhaustion instead.
        all_convs: list[Any] = []
        chunk_size = 500
        offset = 0
        try:
            while True:
                page = self._conversation_manager.list_conversations(
                    limit=chunk_size, offset=offset
                )
                if not page:
                    break
                all_convs.extend(page)
                if len(page) < chunk_size:
                    break
                offset += chunk_size
        except Exception as exc:
            logger.error("Failed to list conversations: %s", exc)
            return []

        results = []
        for conv in all_convs:
            conv_dict = conv.to_dict() if hasattr(conv, "to_dict") else conv
            conv_id = conv_dict.get("id", "")

            # Filter by specific conversation IDs
            if filters.conversation_ids and conv_id not in filters.conversation_ids:
                continue

            # Filter by date range
            updated = conv_dict.get("updated_at", "")
            if filters.date_from and updated < filters.date_from:
                continue
            if filters.date_to and updated > filters.date_to:
                continue

            # Filter by model
            if filters.model and conv_dict.get("model") != filters.model:
                continue

            # Fetch messages
            try:
                messages = self._conversation_manager.get_messages(conv_id)
                msg_list = [
                    m.to_dict() if hasattr(m, "to_dict") else m
                    for m in messages
                ]
            except Exception as exc:
                logger.debug("Failed to get messages for %s: %s", conv_id, exc)
                continue

            # Filter by minimum turns
            user_count = sum(1 for m in msg_list if m.get("role") == "user")
            if user_count < filters.min_turns:
                continue

            # Filter by quality score
            if filters.min_quality > 0:
                score = self._score_conversation(
                    {"id": conv_id, "messages": msg_list}
                )
                if score.combined_score < filters.min_quality:
                    continue

            conv_dict["messages"] = msg_list
            results.append(conv_dict)

        # Apply max_conversations limit from config
        max_convs = self._config.get("export", {}).get("max_conversations", 0)
        if max_convs > 0:
            results = results[:max_convs]

        return results

    def _fetch_all_conversations(
        self,
        conversation_ids: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch conversations for quality scoring."""
        if self._conversation_manager is None:
            return []

        try:
            convs = self._conversation_manager.list_conversations(
                limit=limit, offset=0
            )
        except Exception:
            return []

        results = []
        for conv in convs:
            conv_dict = conv.to_dict() if hasattr(conv, "to_dict") else conv
            conv_id = conv_dict.get("id", "")

            if conversation_ids and conv_id not in conversation_ids:
                continue

            try:
                messages = self._conversation_manager.get_messages(conv_id)
                conv_dict["messages"] = [
                    m.to_dict() if hasattr(m, "to_dict") else m
                    for m in messages
                ]
            except Exception:
                conv_dict["messages"] = []

            results.append(conv_dict)

        return results

    def _score_conversation(
        self, conv: dict[str, Any]
    ) -> QualityScore:
        """Score a single conversation using feedback and benchmarks."""
        conv_id = conv.get("id", "")

        # Gather feedback entries
        feedback_entries = None
        if self._feedback_store is not None:
            try:
                entries = self._feedback_store.list_feedback(
                    conversation_id=conv_id, limit=100
                )
                if entries:
                    feedback_entries = [
                        e.to_dict() if hasattr(e, "to_dict") else e
                        for e in entries
                    ]
            except Exception as exc:
                logger.debug("Failed to get feedback for %s: %s", conv_id, exc)

        # No benchmark integration for now (placeholder for future)
        benchmark_scores = None

        return self._scorer.score_conversation(
            conversation_id=conv_id,
            feedback_entries=feedback_entries,
            benchmark_scores=benchmark_scores,
        )


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

try:
    from .conversation import conversation_manager as _conv_mgr
except ImportError:
    _conv_mgr = None

try:
    from .feedback import feedback_store as _fb_store
except ImportError:
    _fb_store = None

FINE_TUNE_EXPORT_AVAILABLE = True

try:
    fine_tune_exporter = FineTuneExporter(
        conversation_manager=_conv_mgr,
        feedback_store=_fb_store,
    )
except Exception as exc:
    logger.warning("Failed to initialize FineTuneExporter: %s", exc)
    fine_tune_exporter = None
    FINE_TUNE_EXPORT_AVAILABLE = False
