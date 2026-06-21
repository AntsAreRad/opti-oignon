#!/usr/bin/env python3
"""
Red Team Feedback Loop -- Opti-Oignon S157.

When a red team campaign produces bypasses with high confidence, this module
auto-generates candidate injection-detection patterns (regex) that can be
fed back into the RAG sanitizer's custom_patterns list.

Flow:
1. extract_suggestions(campaign_score, attack_scores) -> list of Suggestion
2. Suggestions are stored in the report record
3. Accept/reject via API
4. Accepted patterns are appended to config/rag.yaml > custom_patterns
"""

__all__ = [
    "Suggestion",
    "SuggestionStore",
    "extract_suggestions",
    "apply_suggestion_to_config",
]

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Sentinel -- all new S157 modules carry this
checkpoint_before_apply = True

# Path to RAG config
_RAG_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "rag.yaml"


@dataclass
class Suggestion:
    """A candidate injection pattern derived from a red team bypass.

    Attributes
    ----------
    id : str
        Unique suggestion identifier.
    pattern_name : str
        Human-readable name for the pattern.
    regex : str
        Regex pattern to detect the attack variant.
    source_category : str
        Attack category that triggered the bypass.
    source_strategy : str
        Strategy that was used in the bypass.
    source_payload_hash : str
        Hash of the attack payload.
    confidence : float
        Confidence that this pattern is useful (0.0-1.0).
    status : str
        One of "pending", "accepted", "rejected".
    """

    id: str
    pattern_name: str
    regex: str
    source_category: str
    source_strategy: str
    source_payload_hash: str = ""
    confidence: float = 0.0
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pattern_name": self.pattern_name,
            "regex": self.regex,
            "source_category": self.source_category,
            "source_strategy": self.source_strategy,
            "source_payload_hash": self.source_payload_hash,
            "confidence": round(self.confidence, 4),
            "status": self.status,
        }


class SuggestionStore:
    """In-memory store for feedback suggestions.

    Keyed by suggestion ID for O(1) lookup.
    """

    def __init__(self) -> None:
        self._suggestions: dict[str, Suggestion] = {}
        self._counter: int = 0

    def add(self, suggestion: Suggestion) -> str:
        """Add a suggestion and return its ID."""
        self._suggestions[suggestion.id] = suggestion
        return suggestion.id

    def get(self, suggestion_id: str) -> Suggestion | None:
        """Retrieve a suggestion by ID."""
        return self._suggestions.get(suggestion_id)

    def list_all(self) -> list[Suggestion]:
        """Return all suggestions."""
        return list(self._suggestions.values())

    def list_pending(self) -> list[Suggestion]:
        """Return only pending suggestions."""
        return [s for s in self._suggestions.values() if s.status == "pending"]

    def accept(self, suggestion_id: str) -> Suggestion | None:
        """Mark a suggestion as accepted."""
        s = self._suggestions.get(suggestion_id)
        if s is not None:
            s.status = "accepted"
        return s

    def reject(self, suggestion_id: str) -> Suggestion | None:
        """Mark a suggestion as rejected."""
        s = self._suggestions.get(suggestion_id)
        if s is not None:
            s.status = "rejected"
        return s

    def next_id(self) -> str:
        """Generate the next suggestion ID."""
        self._counter += 1
        return f"sg-{self._counter:04d}"

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Serialize all suggestions to a list of dicts."""
        return [s.to_dict() for s in self._suggestions.values()]


# Module-level singleton
suggestion_store = SuggestionStore()


# ---------------------------------------------------------------------------
# Pattern extraction from bypass results
# ---------------------------------------------------------------------------

# Common injection markers to build patterns from
_INJECTION_MARKERS = [
    (r"ignore\s+(all\s+)?previous\s+instructions", "ignore_previous"),
    (r"disregard\s+(all\s+)?(above|prior|previous)", "disregard_prior"),
    (r"you\s+are\s+now\s+(?:a|an)\s+", "role_override"),
    (r"system\s*:\s*", "system_prefix"),
    (r"\[INST\]", "inst_tag"),
    (r"<\|im_start\|>", "im_start_tag"),
    (r"forget\s+(?:everything|all|your)\s+(?:\w+\s+)*(instructions|rules|guidelines)", "forget_rules"),
    (r"new\s+instructions?\s*:", "new_instructions"),
    (r"override\s+(safety|content|system)", "override_safety"),
    (r"act\s+as\s+if\s+you\s+(have\s+)?no\s+(restrictions|rules|limits)", "no_restrictions"),
]


def _generate_pattern_from_payload(payload: str) -> tuple[str, str] | None:
    """Try to derive a regex pattern from an attack payload.

    Returns (regex, name) or None if no recognizable pattern is found.
    """
    # Check against known markers
    for pattern, name in _INJECTION_MARKERS:
        if re.search(pattern, payload, re.IGNORECASE):
            return pattern, name

    # Heuristic: look for delimiter-like structures
    # Detect payloads that use triple backticks, XML-like tags, or similar
    if re.search(r"```\s*(system|admin|root)", payload, re.IGNORECASE):
        fragment = re.escape("```") + r"\s*(system|admin|root)"
        return fragment, "backtick_system"

    if re.search(r"</?(?:system|admin|root|override)[^>]*>", payload, re.IGNORECASE):
        return r"</?(?:system|admin|root|override)[^>]*>", "xml_system_tag"

    return None


def extract_suggestions(
    attack_scores: list[Any],
    min_confidence: float = 0.7,
) -> list[Suggestion]:
    """Extract pattern suggestions from bypass results.

    Only considers bypasses (classification == "bypass") with low defense
    scores (high confidence that the attack succeeded).

    Parameters
    ----------
    attack_scores : list[AttackScore]
        Scored attack results from a campaign.
    min_confidence : float
        Minimum confidence threshold for suggestions. A bypass with
        defense_score=0.0 gets confidence=1.0; defense_score=0.3 gets
        confidence=0.7.

    Returns
    -------
    list[Suggestion]
        Generated suggestions ready for review.
    """
    suggestions: list[Suggestion] = []
    seen_patterns: set[str] = set()

    for score in attack_scores:
        # Only process bypasses
        classification = getattr(score, "classification", "")
        if classification != "bypass":
            continue

        # Compute confidence: inverse of defense score
        defense_score = getattr(score, "defense_score", 0.0)
        confidence = 1.0 - defense_score

        if confidence < min_confidence:
            continue

        # Try to extract a pattern from metadata or payload hash
        payload = getattr(score, "metadata", {}).get("payload", "")
        if not payload:
            # Fallback: generate a generic category-based pattern
            category = getattr(score, "category", "unknown")
            strategy = getattr(score, "strategy", "unknown")
            payload_hash = getattr(score, "payload_hash", "")

            # Skip if we have no payload to analyze
            if not payload_hash:
                continue

            # Create a placeholder suggestion based on category
            pattern_name = f"rt_{category}_{strategy}"
            if pattern_name in seen_patterns:
                continue
            seen_patterns.add(pattern_name)

            suggestion = Suggestion(
                id=suggestion_store.next_id(),
                pattern_name=pattern_name,
                regex=f"(?i).*{re.escape(category)}.*",
                source_category=category,
                source_strategy=strategy,
                source_payload_hash=payload_hash,
                confidence=confidence,
            )
            suggestions.append(suggestion)
            suggestion_store.add(suggestion)
            continue

        # Try to derive pattern from payload content
        result = _generate_pattern_from_payload(payload)
        if result is None:
            continue

        regex, base_name = result
        category = getattr(score, "category", "unknown")
        strategy = getattr(score, "strategy", "unknown")
        payload_hash = getattr(score, "payload_hash", "")

        pattern_name = f"rt_{category}_{base_name}"
        if pattern_name in seen_patterns:
            continue
        seen_patterns.add(pattern_name)

        suggestion = Suggestion(
            id=suggestion_store.next_id(),
            pattern_name=pattern_name,
            regex=regex,
            source_category=category,
            source_strategy=strategy,
            source_payload_hash=payload_hash,
            confidence=confidence,
        )
        suggestions.append(suggestion)
        suggestion_store.add(suggestion)

    return suggestions


# ---------------------------------------------------------------------------
# Apply accepted pattern to RAG config
# ---------------------------------------------------------------------------

def apply_suggestion_to_config(
    suggestion: Suggestion,
    config_path: str | Path | None = None,
) -> bool:
    """Append an accepted suggestion's pattern to rag.yaml custom_patterns.

    Parameters
    ----------
    suggestion : Suggestion
        The accepted suggestion to apply.
    config_path : str or Path or None
        Path to rag.yaml. Uses default if None.

    Returns
    -------
    bool
        True if successfully applied, False on error.
    """
    if suggestion.status != "accepted":
        logger.warning(
            "Cannot apply suggestion %s: status is %s, not accepted",
            suggestion.id, suggestion.status,
        )
        return False

    path = Path(config_path) if config_path else _RAG_CONFIG_PATH

    try:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Failed to read rag.yaml: %s", exc)
        return False

    # Navigate to sanitization.custom_patterns
    rag_section = config.get("rag", config)
    sanitization = rag_section.get("sanitization", {})
    custom_patterns = sanitization.get("custom_patterns", [])

    if not isinstance(custom_patterns, list):
        custom_patterns = []

    # Check for duplicate
    existing_names = {p.get("name") for p in custom_patterns if isinstance(p, dict)}
    if suggestion.pattern_name in existing_names:
        logger.info(
            "Pattern '%s' already exists in custom_patterns, skipping",
            suggestion.pattern_name,
        )
        return True

    # Validate regex compiles
    try:
        re.compile(suggestion.regex)
    except re.error as exc:
        logger.error(
            "Invalid regex in suggestion %s: %s", suggestion.id, exc
        )
        return False

    # Append new pattern
    new_entry = {
        "name": suggestion.pattern_name,
        "regex": suggestion.regex,
    }
    custom_patterns.append(new_entry)
    sanitization["custom_patterns"] = custom_patterns

    # Write back
    if "rag" in config:
        config["rag"]["sanitization"] = sanitization
    else:
        config["sanitization"] = sanitization

    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(
            "Applied suggestion %s: pattern '%s' added to rag.yaml",
            suggestion.id, suggestion.pattern_name,
        )
        return True
    except Exception as exc:
        logger.error("Failed to write rag.yaml: %s", exc)
        return False
