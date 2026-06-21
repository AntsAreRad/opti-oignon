#!/usr/bin/env python3
"""
Humanizer Engine -- S86

Post-processing to reduce "statistically perfect" LLM output.
Makes local model responses feel more natural and less machine-generated.

Strategies:
  - Vocabulary diversity: replace overused LLM words
  - Filler reduction: strip formulaic phrases
  - Contraction injection: configurable formality
  - Hedging calibration: reduce excessive hedging
  - LLM rewrite pass: prompt-based naturalness rewrite

Modes:
  - rewrite: LLM-based rewrite (always available)
  - logprobs: token-level intervention (requires Ollama logprobs)
  - hybrid: apply rule-based strategies first, then LLM rewrite
"""

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# HUM-05 (S194): guard the yaml import so a missing PyYAML degrades the
# module instead of breaking its import (VL-02 sibling-consistency class).
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - PyYAML is a core dependency
    yaml = None  # type: ignore[assignment]
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)
# S136 audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# ---------------------------------------------------------------------------
# Ollama import
# ---------------------------------------------------------------------------

try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    _ollama = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "humanizer.yaml"
_DATA_DIR = Path(__file__).parent.parent / "data"

VALID_MODES = ("rewrite", "logprobs", "hybrid")
VALID_INTENSITIES = ("light", "moderate", "heavy")
VALID_FORMALITIES = ("casual", "neutral", "formal")


def _load_config(path: Path | None = None) -> dict[str, Any]:
    """Load humanizer config from YAML."""
    config_path = path or _DEFAULT_CONFIG_PATH
    if not YAML_AVAILABLE or not config_path.exists():
        logger.warning("Humanizer config not found at %s, using defaults", config_path)
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to load humanizer config: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HumanizerConfig:
    """Runtime configuration for the humanizer engine."""
    enabled: bool = False
    mode: str = "rewrite"
    intensity: str = "moderate"
    formality: str = "neutral"
    rewrite_model: str | None = None
    max_input_length: int = 8000
    banned_phrases: list[str] = field(default_factory=list)
    vocabulary_replacements: dict[str, str] = field(default_factory=dict)
    contractions: dict[str, str] = field(default_factory=dict)
    hedging_excess: list[str] = field(default_factory=list)
    feedback_db: str = "humanizer_feedback.db"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "intensity": self.intensity,
            "formality": self.formality,
            "rewrite_model": self.rewrite_model,
            "max_input_length": self.max_input_length,
            "banned_phrases": self.banned_phrases,
            "vocabulary_replacements": self.vocabulary_replacements,
            "contractions": self.contractions,
            "hedging_excess": self.hedging_excess,
            "feedback_db": self.feedback_db,
        }


@dataclass
class HumanizerResult:
    """Result of a humanization pass."""
    original: str
    humanized: str
    strategies_applied: list[str] = field(default_factory=list)
    replacements_count: int = 0
    rewrite_model: str | None = None
    latency_ms: float = 0.0
    mode: str = "rewrite"
    intensity: str = "moderate"
    comparison_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dict."""
        return {
            "original": self.original,
            "humanized": self.humanized,
            "strategies_applied": self.strategies_applied,
            "replacements_count": self.replacements_count,
            "rewrite_model": self.rewrite_model,
            "latency_ms": self.latency_ms,
            "mode": self.mode,
            "intensity": self.intensity,
            "comparison_id": self.comparison_id,
        }


@dataclass
class FeedbackStats:
    """Aggregated feedback statistics."""
    total_ratings: int = 0
    humanized_wins: int = 0
    original_wins: int = 0
    ties: int = 0
    win_rate: float = 0.0
    by_strategy: dict[str, dict[str, int]] = field(default_factory=dict)
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    by_intensity: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats to dict."""
        return {
            "total_ratings": self.total_ratings,
            "humanized_wins": self.humanized_wins,
            "original_wins": self.original_wins,
            "ties": self.ties,
            "win_rate": self.win_rate,
            "by_strategy": self.by_strategy,
            "by_model": self.by_model,
            "by_intensity": self.by_intensity,
        }


# ---------------------------------------------------------------------------
# Feedback database
# ---------------------------------------------------------------------------

class HumanizerFeedbackDB:
    """SQLite storage for A/B comparison feedback."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or (_DATA_DIR / "humanizer_feedback.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        try:
            with _safe_connect(self._db_path, check_same_thread=False) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS comparisons (
                        id TEXT PRIMARY KEY,
                        original TEXT NOT NULL,
                        humanized TEXT NOT NULL,
                        strategies TEXT NOT NULL,
                        model TEXT,
                        intensity TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ratings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        comparison_id TEXT NOT NULL,
                        winner TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        FOREIGN KEY (comparison_id) REFERENCES comparisons(id)
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ratings_comparison
                    ON ratings(comparison_id)
                """)
                conn.commit()
        except Exception as e:
            logger.error("Failed to initialize humanizer feedback DB: %s", e)

    def store_comparison(
        self,
        comparison_id: str,
        original: str,
        humanized: str,
        strategies: list[str],
        model: str | None,
        intensity: str,
        mode: str,
    ) -> bool:
        """Store an A/B comparison for later rating."""
        try:
            with _safe_connect(self._db_path, check_same_thread=False) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO comparisons
                       (id, original, humanized, strategies, model, intensity, mode, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        comparison_id,
                        original,
                        humanized,
                        ",".join(strategies),
                        model,
                        intensity,
                        mode,
                        time.time(),
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to store comparison: %s", e)
            return False

    def store_rating(self, comparison_id: str, winner: str) -> bool:
        """Store a user rating for a comparison.

        Args:
            comparison_id: ID of the comparison.
            winner: "humanized", "original", or "tie".
        """
        if winner not in ("humanized", "original", "tie"):
            return False
        try:
            with _safe_connect(self._db_path, check_same_thread=False) as conn:
                # Verify comparison exists
                row = conn.execute(
                    "SELECT id FROM comparisons WHERE id = ?",
                    (comparison_id,),
                ).fetchone()
                if not row:
                    return False
                conn.execute(
                    """INSERT INTO ratings (comparison_id, winner, created_at)
                       VALUES (?, ?, ?)""",
                    (comparison_id, winner, time.time()),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to store rating: %s", e)
            return False

    def get_stats(self) -> FeedbackStats:
        """Compute aggregated feedback statistics."""
        stats = FeedbackStats()
        try:
            with _safe_connect(self._db_path, check_same_thread=False) as conn:
                # Overall counts
                rows = conn.execute(
                    "SELECT winner, COUNT(*) FROM ratings GROUP BY winner"
                ).fetchall()
                for winner, count in rows:
                    stats.total_ratings += count
                    if winner == "humanized":
                        stats.humanized_wins = count
                    elif winner == "original":
                        stats.original_wins = count
                    elif winner == "tie":
                        stats.ties = count

                if stats.total_ratings > 0:
                    stats.win_rate = stats.humanized_wins / stats.total_ratings

                # By strategy
                strategy_rows = conn.execute("""
                    SELECT c.strategies, r.winner, COUNT(*)
                    FROM ratings r JOIN comparisons c ON r.comparison_id = c.id
                    GROUP BY c.strategies, r.winner
                """).fetchall()
                for strategies_str, winner, count in strategy_rows:
                    for strategy in strategies_str.split(","):
                        strategy = strategy.strip()
                        if not strategy:
                            continue
                        if strategy not in stats.by_strategy:
                            stats.by_strategy[strategy] = {
                                "humanized": 0, "original": 0, "tie": 0,
                            }
                        stats.by_strategy[strategy][winner] = (
                            stats.by_strategy[strategy].get(winner, 0) + count
                        )

                # By model
                model_rows = conn.execute("""
                    SELECT c.model, r.winner, COUNT(*)
                    FROM ratings r JOIN comparisons c ON r.comparison_id = c.id
                    WHERE c.model IS NOT NULL
                    GROUP BY c.model, r.winner
                """).fetchall()
                for model, winner, count in model_rows:
                    if model not in stats.by_model:
                        stats.by_model[model] = {
                            "humanized": 0, "original": 0, "tie": 0,
                        }
                    stats.by_model[model][winner] = (
                        stats.by_model[model].get(winner, 0) + count
                    )

                # By intensity
                intensity_rows = conn.execute("""
                    SELECT c.intensity, r.winner, COUNT(*)
                    FROM ratings r JOIN comparisons c ON r.comparison_id = c.id
                    GROUP BY c.intensity, r.winner
                """).fetchall()
                for intensity, winner, count in intensity_rows:
                    if intensity not in stats.by_intensity:
                        stats.by_intensity[intensity] = {
                            "humanized": 0, "original": 0, "tie": 0,
                        }
                    stats.by_intensity[intensity][winner] = (
                        stats.by_intensity[intensity].get(winner, 0) + count
                    )

        except Exception as e:
            logger.error("Failed to compute feedback stats: %s", e)

        return stats


# ---------------------------------------------------------------------------
# Humanizer strategies (rule-based)
# ---------------------------------------------------------------------------

# HUM-01 (S194): fenced blocks and inline code are masked before any
# rule-based transform and restored afterwards, so vocabulary swaps,
# contraction injection, phrase stripping, and whitespace cleanup can
# never touch code. Placeholders use a private-use unicode sentinel
# that no strategy pattern can match or alter.
_CODE_SEGMENT_RE = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)
_SEG_SENTINEL = "\ue000"


def _protect_code_segments(text: str) -> tuple[str, list[str]]:
    """Replace code segments with placeholders.

    Returns (masked_text, segments). An unterminated fence is left
    unmasked (same convention as the artifact detector).
    """
    segments: list[str] = []

    def _stash(m: re.Match) -> str:
        token = f"{_SEG_SENTINEL}OOSEG{len(segments)}{_SEG_SENTINEL}"
        segments.append(m.group(0))
        return token

    return _CODE_SEGMENT_RE.sub(_stash, text), segments


def _restore_code_segments(text: str, segments: list[str]) -> str:
    """Restore previously masked code segments."""
    for i, seg in enumerate(segments):
        text = text.replace(f"{_SEG_SENTINEL}OOSEG{i}{_SEG_SENTINEL}", seg)
    return text


def _apply_vocabulary_replacements(
    text: str, replacements: dict[str, str]
) -> tuple[str, int]:
    """Replace overused LLM words with natural alternatives.

    Uses word-boundary matching to avoid partial replacements.
    Returns (modified_text, replacement_count).
    """
    count = 0
    for word, replacement in replacements.items():
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            count += len(matches)
            # Preserve capitalization of first char
            def _replace_match(m: re.Match) -> str:
                original = m.group(0)
                if original[0].isupper():
                    return replacement[0].upper() + replacement[1:]
                return replacement
            text = pattern.sub(_replace_match, text)
    return text, count


def _strip_banned_phrases(text: str, phrases: list[str]) -> tuple[str, int]:
    """Remove banned filler phrases from text.

    HUM-04 (S194): patterns are anchored on a leading word boundary
    (no substring matches like "note that" inside "denote that"), and
    the whitespace cleanup only runs when something was removed, only
    collapses runs that do not start a line (markdown indentation is
    preserved).

    Returns (modified_text, removal_count).
    """
    count = 0
    for phrase in phrases:
        prefix = r"\b" if phrase and re.match(r"\w", phrase[0]) else ""
        suffix = r"\b" if phrase and re.match(r"\w", phrase[-1]) else ""
        pattern = re.compile(
            prefix + re.escape(phrase) + suffix + r"[,.:;]?\s*",
            re.IGNORECASE,
        )
        matches = pattern.findall(text)
        if matches:
            count += len(matches)
            text = pattern.sub("", text)
    if count > 0:
        text = re.sub(r"(?<=\S)  +", " ", text).strip()
    return text, count


def _apply_contractions(
    text: str, contractions: dict[str, str], formality: str
) -> tuple[str, int]:
    """Inject contractions based on formality level.

    Formal: no contractions applied.
    Neutral: contractions applied except in first sentence.
    Casual: all contractions applied.

    Returns (modified_text, contraction_count).
    """
    if formality == "formal":
        return text, 0

    count = 0
    # HUM-02 (S194): split CAPTURING the separators and rejoin them
    # exactly, so newlines and paragraph breaks after sentence enders
    # survive (the previous " ".join flattened all structure).
    parts = re.split(r"((?<=[.!?])\s+)", text)
    sentence_positions = list(range(0, len(parts), 2))
    start_idx = 1 if formality == "neutral" else 0

    for n, i in enumerate(sentence_positions):
        if n < start_idx:
            continue
        for full_form, contracted in contractions.items():
            pattern = re.compile(r"\b" + re.escape(full_form) + r"\b", re.IGNORECASE)
            matches = pattern.findall(parts[i])
            if matches:
                count += len(matches)

                def _contract_match(m: re.Match) -> str:
                    original = m.group(0)
                    if original[0].isupper():
                        return contracted[0].upper() + contracted[1:]
                    return contracted

                parts[i] = pattern.sub(_contract_match, parts[i])

    return "".join(parts), count


def _reduce_hedging(text: str, hedging_phrases: list[str]) -> tuple[str, int]:
    """Reduce excessive hedging phrases.

    HUM-04 (S194): leading word-boundary anchor, cleanup conditional on
    a removal and scoped off line starts, plus capitalization of a
    sentence left lowercase at the very start of the text.

    Returns (modified_text, reduction_count).
    """
    count = 0
    for phrase in hedging_phrases:
        prefix = r"\b" if phrase and re.match(r"\w", phrase[0]) else ""
        suffix = r"\b" if phrase and re.match(r"\w", phrase[-1]) else ""
        pattern = re.compile(
            prefix + re.escape(phrase) + suffix + r"\s*", re.IGNORECASE
        )
        matches = pattern.findall(text)
        if matches:
            count += len(matches)
            text = pattern.sub("", text)
    if count > 0:
        text = re.sub(r"(?<=\S)  +", " ", text).strip()
        # Fix sentences starting with lowercase after removal
        text = re.sub(r"(?<=[.!?]\s)([a-z])", lambda m: m.group(1).upper(), text)
        text = re.sub(r"^([a-z])", lambda m: m.group(1).upper(), text)
    return text, count


# ---------------------------------------------------------------------------
# Rewrite prompt templates
# ---------------------------------------------------------------------------

_REWRITE_PROMPTS: dict[str, str] = {
    "light": (
        "Lightly edit the following text to sound slightly more natural. "
        "Keep the same meaning, structure, and length. Only fix the most "
        "obvious robotic phrasing. Do not add new information. "
        "Keep all code blocks and inline code exactly unchanged.\n\n"
        "Text:\n{text}\n\nEdited text:"
    ),
    "moderate": (
        "Rewrite the following text to sound more natural and human-written. "
        "Vary sentence lengths, use more conversational phrasing where appropriate, "
        "and remove formulaic expressions. Keep the same meaning and approximate length. "
        "Do not add new information or change factual content. "
        "Keep all code blocks and inline code exactly unchanged.\n\n"
        "Text:\n{text}\n\nRewritten text:"
    ),
    "heavy": (
        "Substantially rewrite the following text to sound like it was written by "
        "a knowledgeable human, not a language model. Vary rhythm and structure, "
        "use natural transitions, drop unnecessary qualifiers, and make it engaging. "
        "Preserve all factual content and the overall message. "
        "Do not add new information. "
        "Keep all code blocks and inline code exactly unchanged.\n\n"
        "Text:\n{text}\n\nRewritten text:"
    ),
}


# ---------------------------------------------------------------------------
# Humanizer Engine
# ---------------------------------------------------------------------------

class HumanizerEngine:
    """Main humanizer engine orchestrating strategies and feedback."""

    def __init__(self, config_path: Path | None = None):
        raw = _load_config(config_path)
        self._config = HumanizerConfig(
            enabled=raw.get("enabled", False),
            mode=raw.get("mode", "rewrite"),
            intensity=raw.get("intensity", "moderate"),
            formality=raw.get("formality", "neutral"),
            rewrite_model=raw.get("rewrite_model"),
            max_input_length=raw.get("max_input_length", 8000),
            banned_phrases=raw.get("banned_phrases", []),
            vocabulary_replacements=raw.get("vocabulary_replacements", {}),
            contractions=raw.get("contractions", {}),
            hedging_excess=raw.get("hedging_excess", []),
            feedback_db=raw.get("feedback_db", "humanizer_feedback.db"),
        )

        db_path = _DATA_DIR / self._config.feedback_db
        self._feedback_db = HumanizerFeedbackDB(db_path)
        logger.info(
            "HumanizerEngine initialized (enabled=%s, mode=%s, intensity=%s)",
            self._config.enabled,
            self._config.mode,
            self._config.intensity,
        )

    # -- Config accessors --

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def get_config(self) -> dict[str, Any]:
        """Return current config as dict."""
        return self._config.to_dict()

    def update_config(self, **kwargs: Any) -> dict[str, Any]:
        """Update config fields. Only updates known fields with valid values."""
        if "enabled" in kwargs and isinstance(kwargs["enabled"], bool):
            self._config.enabled = kwargs["enabled"]
        if "mode" in kwargs and kwargs["mode"] in VALID_MODES:
            self._config.mode = kwargs["mode"]
        if "intensity" in kwargs and kwargs["intensity"] in VALID_INTENSITIES:
            self._config.intensity = kwargs["intensity"]
        if "formality" in kwargs and kwargs["formality"] in VALID_FORMALITIES:
            self._config.formality = kwargs["formality"]
        if "rewrite_model" in kwargs:
            self._config.rewrite_model = kwargs["rewrite_model"]
        if "max_input_length" in kwargs and isinstance(kwargs["max_input_length"], int):
            self._config.max_input_length = max(100, kwargs["max_input_length"])
        if "banned_phrases" in kwargs and isinstance(kwargs["banned_phrases"], list):
            self._config.banned_phrases = kwargs["banned_phrases"]
        if "vocabulary_replacements" in kwargs and isinstance(kwargs["vocabulary_replacements"], dict):
            self._config.vocabulary_replacements = kwargs["vocabulary_replacements"]

        logger.info("Humanizer config updated: %s", {k: v for k, v in kwargs.items() if k != "banned_phrases"})
        return self._config.to_dict()

    # -- Rule-based strategies --

    def _apply_rules(self, text: str) -> tuple[str, list[str], int]:
        """Apply all rule-based strategies.

        HUM-01 (S194): code segments are masked for the whole pass.
        HUM-03 (S194): hedging runs BEFORE contractions, otherwise
        contracting "it is" -> "it's" makes multiword hedges like
        "It is possible that" unmatchable.

        Returns (modified_text, strategies_applied, total_replacements).
        """
        strategies: list[str] = []
        total_replacements = 0

        text, code_segments = _protect_code_segments(text)

        # Vocabulary diversity
        if self._config.vocabulary_replacements:
            text, count = _apply_vocabulary_replacements(
                text, self._config.vocabulary_replacements
            )
            if count > 0:
                strategies.append("vocabulary_diversity")
                total_replacements += count

        # Filler reduction
        if self._config.banned_phrases:
            text, count = _strip_banned_phrases(text, self._config.banned_phrases)
            if count > 0:
                strategies.append("filler_reduction")
                total_replacements += count

        # Hedging calibration (before contractions, HUM-03)
        if self._config.hedging_excess:
            text, count = _reduce_hedging(text, self._config.hedging_excess)
            if count > 0:
                strategies.append("hedging_calibration")
                total_replacements += count

        # Contraction injection
        if self._config.contractions and self._config.formality != "formal":
            text, count = _apply_contractions(
                text, self._config.contractions, self._config.formality
            )
            if count > 0:
                strategies.append("contraction_injection")
                total_replacements += count

        text = _restore_code_segments(text, code_segments)

        return text, strategies, total_replacements

    # -- LLM rewrite --

    def _rewrite_with_llm(
        self, text: str, model: str | None = None, intensity: str | None = None
    ) -> str | None:
        """Perform LLM rewrite pass.

        Returns rewritten text or None if LLM is unavailable.
        """
        if not OLLAMA_AVAILABLE or _ollama is None:
            logger.warning("Ollama unavailable, skipping LLM rewrite")
            return None

        effective_model = model or self._config.rewrite_model
        if not effective_model:
            logger.warning("No rewrite model specified, skipping LLM rewrite")
            return None

        effective_intensity = intensity or self._config.intensity
        prompt_template = _REWRITE_PROMPTS.get(effective_intensity, _REWRITE_PROMPTS["moderate"])
        prompt = prompt_template.format(text=text)

        try:
            response = _ollama.generate(
                model=effective_model,
                prompt=prompt,
                options={"temperature": 0.7, "num_predict": len(text) * 2},
            )
            result = ""
            if hasattr(response, "response"):
                result = response.response
            elif isinstance(response, dict):
                result = response.get("response", "")
            return result.strip() if result else None
        except Exception as e:
            logger.error("LLM rewrite failed: %s", e)
            return None

    # -- Main humanize method --

    def humanize(
        self,
        text: str,
        model: str | None = None,
        mode: str | None = None,
        intensity: str | None = None,
        formality: str | None = None,
    ) -> HumanizerResult:
        """Humanize a text passage.

        Args:
            text: Input text to humanize.
            model: Override model for rewrite.
            mode: Override mode (rewrite/logprobs/hybrid).
            intensity: Override intensity.
            formality: Override formality.

        Returns:
            HumanizerResult with original and humanized text.
        """
        start_time = time.time()
        effective_mode = mode or self._config.mode
        effective_intensity = intensity or self._config.intensity
        comparison_id = str(uuid.uuid4())

        # Apply formality override temporarily
        original_formality = self._config.formality
        if formality and formality in VALID_FORMALITIES:
            self._config.formality = formality

        try:
            return self._do_humanize(
                text=text,
                model=model,
                mode=effective_mode,
                intensity=effective_intensity,
                comparison_id=comparison_id,
                start_time=start_time,
            )
        finally:
            self._config.formality = original_formality

    def _do_humanize(
        self,
        text: str,
        model: str | None,
        mode: str,
        intensity: str,
        comparison_id: str,
        start_time: float,
    ) -> HumanizerResult:
        """Internal humanization logic."""
        strategies: list[str] = []
        total_replacements = 0
        humanized = text

        # Skip very short or empty text
        if not text or len(text.strip()) < 20:
            return HumanizerResult(
                original=text,
                humanized=text,
                mode=mode,
                intensity=intensity,
                comparison_id=comparison_id,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Skip if text exceeds max length
        if len(text) > self._config.max_input_length:
            logger.info(
                "Text exceeds max_input_length (%d > %d), skipping",
                len(text),
                self._config.max_input_length,
            )
            return HumanizerResult(
                original=text,
                humanized=text,
                mode=mode,
                intensity=intensity,
                comparison_id=comparison_id,
                latency_ms=(time.time() - start_time) * 1000,
            )

        effective_model = model or self._config.rewrite_model

        if mode == "hybrid" or mode == "logprobs":
            # Apply rule-based strategies first
            humanized, strategies, total_replacements = self._apply_rules(humanized)

            if mode == "hybrid":
                # Then LLM rewrite on the already-cleaned text
                rewritten = self._rewrite_with_llm(humanized, effective_model, intensity)
                if rewritten:
                    humanized = rewritten
                    strategies.append("llm_rewrite")

        elif mode == "rewrite":
            # Pure LLM rewrite
            rewritten = self._rewrite_with_llm(text, effective_model, intensity)
            if rewritten:
                humanized = rewritten
                strategies.append("llm_rewrite")
            else:
                # Fallback to rule-based if LLM unavailable
                humanized, strategies, total_replacements = self._apply_rules(humanized)

        latency_ms = (time.time() - start_time) * 1000

        result = HumanizerResult(
            original=text,
            humanized=humanized,
            strategies_applied=strategies,
            replacements_count=total_replacements,
            rewrite_model=effective_model,
            latency_ms=latency_ms,
            mode=mode,
            intensity=intensity,
            comparison_id=comparison_id,
        )

        # Store comparison for A/B feedback
        self._feedback_db.store_comparison(
            comparison_id=comparison_id,
            original=text,
            humanized=humanized,
            strategies=strategies,
            model=effective_model,
            intensity=intensity,
            mode=mode,
        )

        logger.info(
            "Humanized text (mode=%s, intensity=%s, strategies=%s, "
            "replacements=%d, latency=%.1fms)",
            mode,
            intensity,
            strategies,
            total_replacements,
            latency_ms,
        )

        return result

    # -- Feedback --

    def submit_feedback(self, comparison_id: str, winner: str) -> bool:
        """Submit an A/B rating for a comparison."""
        return self._feedback_db.store_rating(comparison_id, winner)

    def get_stats(self) -> FeedbackStats:
        """Get aggregated feedback statistics."""
        return self._feedback_db.get_stats()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

try:
    humanizer_engine = HumanizerEngine()
    HUMANIZER_AVAILABLE = True
except Exception as _init_err:
    logger.error("Failed to initialize HumanizerEngine: %s", _init_err)
    humanizer_engine = None  # type: ignore[assignment]
    HUMANIZER_AVAILABLE = False
