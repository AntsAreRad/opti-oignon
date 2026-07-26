#!/usr/bin/env python3
"""
Context Window Management -- Opti-Oignon v1.4.0 (Session 16)
=============================================================

C1: SlidingWindowManager -- intelligent message selection for context
C2: TokenBudgetManager -- token budget allocation per model

The module decides which messages to keep, summarize, or drop to
optimize the use of the context window for each model.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# C2: Token Budget Manager
# ============================================================================

@dataclass
class TokenBudget:
    """Token budget allocation for a model.

    Attributes:
        model: Model name
        context_window: Total context window size (tokens)
        system_ratio: Fraction reserved for the system prompt (0.0-1.0)
        history_ratio: Fraction reserved for the conversation history
        generation_ratio: Fraction reserved for response generation
    """

    model: str
    context_window: int
    system_ratio: float = 0.10
    history_ratio: float = 0.60
    generation_ratio: float = 0.30

    @property
    def system_budget(self) -> int:
        """Tokens available for the system prompt."""
        return int(self.context_window * self.system_ratio)

    @property
    def history_budget(self) -> int:
        """Tokens available for the conversation history."""
        return int(self.context_window * self.history_ratio)

    @property
    def generation_budget(self) -> int:
        """Tokens reserved for response generation."""
        return int(self.context_window * self.generation_ratio)

    @property
    def total_allocated(self) -> int:
        """Total allocated tokens (should be <= context_window)."""
        return self.system_budget + self.history_budget + self.generation_budget

    def available_for_history(self, system_tokens: int = 0) -> int:
        """Tokens effectively available for the history.

        If the system prompt exceeds its budget, we borrow from the history.
        If the system prompt is shorter than expected, the history gains.

        Args:
            system_tokens: Actual number of system prompt tokens

        Returns:
            Number of tokens available for the history
        """
        # Total space minus generation = space for system + history
        usable = self.context_window - self.generation_budget
        # Subtract the actual system tokens
        available = usable - system_tokens
        return max(0, available)


# Known model profiles (verified context windows from Ollama/HuggingFace)
_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    # Large models -- generous context
    "qwen3:32b": {"context_window": 32768, "generation_ratio": 0.25},
    "qwen3.5:32b": {"context_window": 32768, "generation_ratio": 0.25},
    "qwen3-coder:30b": {"context_window": 262144, "generation_ratio": 0.15},
    "qwen3-coder-next": {"context_window": 262144, "generation_ratio": 0.15},
    "qwen3.5:35b": {"context_window": 131072, "generation_ratio": 0.20},
    "deepseek-r1:32b": {"context_window": 131072, "generation_ratio": 0.25},
    "nemotron-3-nano:30b": {"context_window": 131072, "generation_ratio": 0.20},
    "gemma3:27b": {"context_window": 131072, "generation_ratio": 0.20},
    "llama3.3": {"context_window": 131072, "generation_ratio": 0.20},
    "mistral-small3.2": {"context_window": 131072, "generation_ratio": 0.20},
    "qwen3-vl:32b": {"context_window": 32768, "generation_ratio": 0.25},
    "dolphin-mixtral": {"context_window": 32768, "generation_ratio": 0.25},
    # Medium models
    "qwen3.5:9b": {"context_window": 131072, "generation_ratio": 0.20},
    "qwen3.5:4b": {"context_window": 131072, "generation_ratio": 0.25},
    "qwen3.5:2b": {"context_window": 131072, "generation_ratio": 0.30},
    "qwen3.5:0.8b": {"context_window": 131072, "generation_ratio": 0.30},
    "qwen3:8b": {"context_window": 32768, "generation_ratio": 0.30},
    "qwen3:4b": {"context_window": 32768, "generation_ratio": 0.35},
    "gemma3:9b": {"context_window": 131072, "generation_ratio": 0.25},
    "gemma2:9b": {"context_window": 8192, "generation_ratio": 0.30},
    "llama3:8b": {"context_window": 8192, "generation_ratio": 0.30},
    "llama3.2": {"context_window": 131072, "generation_ratio": 0.20},
    "phi3:mini": {"context_window": 4096, "generation_ratio": 0.40},
    "wizard-math:13b": {"context_window": 32768, "generation_ratio": 0.30},
    "translategem": {"context_window": 8192, "generation_ratio": 0.30},
    # Code models
    "qwen2.5-coder": {"context_window": 131072, "generation_ratio": 0.25},
    "codellama:13b": {"context_window": 16384, "generation_ratio": 0.35},
    "starcoder2:15b": {"context_window": 16384, "generation_ratio": 0.35},
    "devsral-small-2": {"context_window": 131072, "generation_ratio": 0.25},
    # Thinking models
    "lfm2.5-thinking": {"context_window": 8192, "generation_ratio": 0.35},
    "lfm2": {"context_window": 8192, "generation_ratio": 0.35},
}

# Default values for unknown models
_DEFAULT_CONTEXT_WINDOW = 8192
_DEFAULT_SYSTEM_RATIO = 0.10
_DEFAULT_HISTORY_RATIO = 0.60
_DEFAULT_GENERATION_RATIO = 0.30


class TokenBudgetManager:
    """3-zone token budget manager.

    .. deprecated:: 2.4.0
        Superseded by ``PromptTokenBudgetManager`` and
        ``ContextOptimizer``. Retained for
        backward compatibility -- modules like ``routes_context.py`` and
        ``SlidingWindowManager`` still reference it. Do not use in new code.
    """

    def __init__(self, custom_profiles: dict[str, dict[str, Any]] | None = None):
        """Initialize with default profiles + optional custom profiles.

        Args:
            custom_profiles: Additional profiles {model_name: {context_window, ...}}
        """
        self._profiles = dict(_MODEL_PROFILES)
        if custom_profiles:
            self._profiles.update(custom_profiles)

    @property
    def known_models(self) -> list[str]:
        """List of models with a known profile."""
        return sorted(self._profiles.keys())

    def _match_profile(self, model: str) -> dict[str, Any] | None:
        """Find the profile matching a model.

        Look first for an exact match, then by prefix.

        Args:
            model: Model name (e.g. "qwen3-coder:30b")

        Returns:
            Profile dict or None
        """
        # Correspondance exacte
        if model in self._profiles:
            return self._profiles[model]

        # Prefix match (for variants like "qwen3:32b-q4_0")
        for prefix, profile in self._profiles.items():
            if model.startswith(prefix):
                return profile

        return None

    def get_budget(self, model: str, context_window_override: int = 0) -> TokenBudget:
        """Get token budget for a model.

        Priority: override > profile match > Ollama API > default fallback.

        Args:
            model: Model name
            context_window_override: Explicit context size (0 = auto)

        Returns:
            TokenBudget with optimized allocation
        """
        profile = self._match_profile(model) or {}

        ctx = context_window_override or profile.get("context_window", 0)

        # Fallback: query Ollama API for unknown models
        if ctx == 0:
            ctx = self._fetch_ollama_context_window(model) or _DEFAULT_CONTEXT_WINDOW

        gen_ratio = profile.get("generation_ratio", _DEFAULT_GENERATION_RATIO)
        sys_ratio = profile.get("system_ratio", _DEFAULT_SYSTEM_RATIO)

        # History gets the rest
        hist_ratio = 1.0 - gen_ratio - sys_ratio
        # Safety: minimum 20% for history
        if hist_ratio < 0.20:
            hist_ratio = 0.20
            gen_ratio = 1.0 - hist_ratio - sys_ratio

        return TokenBudget(
            model=model,
            context_window=ctx,
            system_ratio=sys_ratio,
            history_ratio=hist_ratio,
            generation_ratio=gen_ratio,
        )

    def _fetch_ollama_context_window(self, model: str) -> int:
        """Query Ollama for a model's context window size.

        Returns 0 if unavailable.
        """
        try:
            import ollama as _ollama
            info = _ollama.show(model)
            # Ollama returns modelinfo dict with various fields
            if isinstance(info, dict):
                # Try model_info.context_length first
                model_info = info.get("model_info", {})
                if isinstance(model_info, dict):
                    for key, val in model_info.items():
                        if "context_length" in key and isinstance(val, (int, float)):
                            ctx = int(val)
                            if ctx > 0:
                                logger.info("Got context_window=%d for %s from Ollama API", ctx, model)
                                # Cache it for next time
                                self._profiles[model] = {
                                    "context_window": ctx,
                                    "generation_ratio": _DEFAULT_GENERATION_RATIO,
                                }
                                return ctx
        except Exception as exc:
            logger.debug("Ollama context query failed for %s: %s", model, exc)
        return 0

    def add_profile(self, model: str, context_window: int, **kwargs):
        """Add or update a model profile.

        Args:
            model: Model name
            context_window: Context window size
            **kwargs: Overrides (generation_ratio, system_ratio, etc.)
        """
        self._profiles[model] = {"context_window": context_window, **kwargs}

    def allocate(
        self, model: str, system_tokens: int, history_tokens: int
    ) -> dict[str, Any]:
        """Compute allocation and recommendations for a concrete case.

        Args:
            model: Model name
            system_tokens: System prompt tokens
            history_tokens: Tokens of the current history

        Returns:
            Dict with budget, tokens_available, tokens_to_trim, needs_trimming
        """
        budget = self.get_budget(model)
        available = budget.available_for_history(system_tokens)
        to_trim = max(0, history_tokens - available)

        return {
            "budget": budget,
            "history_available": available,
            "history_current": history_tokens,
            "tokens_to_trim": to_trim,
            "needs_trimming": to_trim > 0,
            "utilization": (
                (system_tokens + history_tokens) / budget.context_window
                if budget.context_window > 0
                else 0.0
            ),
        }


# Singleton global
token_budget_manager = TokenBudgetManager()


# ============================================================================
# C1: Sliding Window Manager
# ============================================================================

# Pattern to detect code blocks
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)
# Pattern for detected artifacts (frequent identifiers)
_ARTIFACT_MARKERS = [
    "```html", "```svg", "```csv", "```json",
    "<!DOCTYPE", "<html", "<svg", "<?xml",
]
# Summary-message pattern (compatible with context_summary)
_SUMMARY_PREFIX = "[Summary of earlier conversation]"


@dataclass
class MessageScore:
    """Importance score of a message in the history.

    Attributes:
        index: Position in the message list
        role: Role (user/assistant/system)
        token_estimate: Estimated number of tokens
        importance: Normalized importance score [0.0, 1.0]
        has_code: Contains code blocks
        has_artifact: Likely contains an artifact
        is_summary: Is a summary message (context_summary)
    """

    index: int
    role: str
    token_estimate: int
    importance: float = 0.0
    has_code: bool = False
    has_artifact: bool = False
    is_summary: bool = False


class SlidingWindowManager:
    """Sliding-window manager for the context.

    Decides which messages to keep in the context based on
    of token budget, recency and content importance.

    Strategie:
    1. Always keep the N most recent messages (user/assistant pairs)
    2. Keep existing summary messages (they represent the compressed history)
    3. Score the intermediate messages by importance
    4. Drop the least important messages first
    """

    # Minimum number of recent pairs to always keep
    MIN_RECENT_PAIRS: int = 3

    # Importance weights for scoring
    WEIGHT_RECENCY: float = 0.40
    WEIGHT_CODE: float = 0.25
    WEIGHT_ARTIFACT: float = 0.15
    WEIGHT_LENGTH: float = 0.10
    WEIGHT_USER: float = 0.10

    def __init__(
        self,
        min_recent_pairs: int = 3,
        budget_manager: TokenBudgetManager | None = None,
    ):
        """Initialize the sliding window manager.

        Args:
            min_recent_pairs: Minimum number of recent user/assistant pairs
            budget_manager: Budget manager (uses singleton by default)
        """
        self.MIN_RECENT_PAIRS = max(1, min_recent_pairs)
        self._budget_manager = budget_manager or token_budget_manager

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Quick token-count estimate (heuristic ~1.3 tokens/word).

        Args:
            text: Text to estimate

        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
        # Heuristic: ~1.3 tokens per word (good for French and English)
        words = len(text.split())
        return max(1, int(words * 1.3))

    @staticmethod
    def _has_code_blocks(text: str) -> bool:
        """Detect the presence of code blocks.

        Args:
            text: Contenu du message

        Returns:
            True if the message contains code blocks
        """
        return bool(_CODE_BLOCK_RE.search(text))

    @staticmethod
    def _has_artifact_markers(text: str) -> bool:
        """Detecte la presence probable d'artefacts.

        Args:
            text: Contenu du message

        Returns:
            True if the message contains artifact markers
        """
        text_lower = text.lower()
        return any(marker.lower() in text_lower for marker in _ARTIFACT_MARKERS)

    @staticmethod
    def _is_summary_message(msg: dict[str, str]) -> bool:
        """Check if a message is a context summary.

        Args:
            msg: Message au format {role, content}

        Returns:
            True if it is a summary message
        """
        if msg.get("role") != "system":
            return False
        content = msg.get("content", "")
        return content.startswith(_SUMMARY_PREFIX)

    def _score_message(
        self, msg: dict[str, str], index: int, total: int
    ) -> MessageScore:
        """Compute the importance score of a message.

        Args:
            msg: Message au format {role, content}
            index: Position in the history (0 = oldest)
            total: Total number of messages

        Returns:
            MessageScore with the computed importance
        """
        content = msg.get("content", "")
        role = msg.get("role", "user")
        tokens = self._estimate_tokens(content)
        has_code = self._has_code_blocks(content)
        has_artifact = self._has_artifact_markers(content)
        is_summary = self._is_summary_message(msg)

        # Recency score: 0.0 (old) -> 1.0 (recent)
        recency = index / max(1, total - 1) if total > 1 else 1.0

        # Code score: bonus if it contains code
        code_score = 1.0 if has_code else 0.0

        # Artifact score: bonus if it contains an artifact
        artifact_score = 1.0 if has_artifact else 0.0

        # Length score: longer messages tend to be more important
        # Normalized on a log scale
        length_score = min(1.0, tokens / 500.0)

        # Role score: user messages are slightly more important
        user_score = 1.0 if role == "user" else 0.5

        # Weighted global score
        importance = (
            self.WEIGHT_RECENCY * recency
            + self.WEIGHT_CODE * code_score
            + self.WEIGHT_ARTIFACT * artifact_score
            + self.WEIGHT_LENGTH * length_score
            + self.WEIGHT_USER * user_score
        )

        # Summary messages always have high importance
        if is_summary:
            importance = 0.95

        return MessageScore(
            index=index,
            role=role,
            token_estimate=tokens,
            importance=importance,
            has_code=has_code,
            has_artifact=has_artifact,
            is_summary=is_summary,
        )

    def _identify_recent_boundary(self, messages: list[dict[str, str]]) -> int:
        """Find the start index of the recent messages to keep.

        Keeps at least MIN_RECENT_PAIRS user/assistant pairs from the end.

        Args:
            messages: List of messages

        Returns:
            Start index of the recent messages (inclusive)
        """
        if not messages:
            return 0

        # Count the pairs from the end
        pairs_found = 0
        boundary = len(messages)

        i = len(messages) - 1
        while i >= 0 and pairs_found < self.MIN_RECENT_PAIRS:
            if messages[i].get("role") == "assistant":
                # Find the matching user message just before
                if i > 0 and messages[i - 1].get("role") == "user":
                    pairs_found += 1
                    boundary = i - 1
                    i -= 2
                    continue
            # If no clean pair is found, advance anyway
            boundary = i
            i -= 1

        return max(0, boundary)

    def prepare_messages(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_tokens: int = 0,
        context_window_override: int = 0,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Prepare the messages for the context within the token budget.

        Return selected messages and statistics.

        Args:
            messages: Complete list of conversation messages
            model: Model name cible
            system_tokens: Tokens used by the system prompt
            context_window_override: Context size override (0 = auto)

        Returns:
            Tuple (messages_filtres, stats_dict)
        """
        if not messages:
            return [], {"strategy": "empty", "kept": 0, "dropped": 0, "total_tokens": 0}

        # Get the budget
        budget = self._budget_manager.get_budget(model, context_window_override)
        available = budget.available_for_history(system_tokens)

        # Score all messages
        scores = [
            self._score_message(msg, i, len(messages))
            for i, msg in enumerate(messages)
        ]

        # Calculer le total actuel
        total_tokens = sum(s.token_estimate for s in scores)

        # If everything fits in the budget, keep all
        if total_tokens <= available:
            return list(messages), {
                "strategy": "keep_all",
                "kept": len(messages),
                "dropped": 0,
                "total_tokens": total_tokens,
                "budget_tokens": available,
                "utilization": total_tokens / available if available > 0 else 0.0,
            }

        # --- Windowing strategy ---
        # Step 1: Identify the recent messages to keep at all costs
        recent_boundary = self._identify_recent_boundary(messages)

        # Step 2: Separate old messages (deletion candidates) from recent ones
        old_indices = list(range(0, recent_boundary))
        recent_indices = list(range(recent_boundary, len(messages)))

        # Tokens of the recent messages (always kept)
        recent_tokens = sum(scores[i].token_estimate for i in recent_indices)

        # If even the recent messages exceed the budget, keep just the recent ones
        if recent_tokens >= available:
            kept_messages = [messages[i] for i in recent_indices]
            return kept_messages, {
                "strategy": "recent_only",
                "kept": len(recent_indices),
                "dropped": len(old_indices),
                "total_tokens": recent_tokens,
                "budget_tokens": available,
                "utilization": recent_tokens / available if available > 0 else 0.0,
            }

        # Step 3: Remaining budget for the old messages
        remaining_budget = available - recent_tokens

        # Step 4: Sort the old messages by importance (most important first)
        old_scored = [(i, scores[i]) for i in old_indices]
        old_scored.sort(key=lambda x: x[1].importance, reverse=True)

        # Step 5: Keep the most important old messages that fit
        kept_old_indices = []
        used_tokens = 0
        for idx, score in old_scored:
            if used_tokens + score.token_estimate <= remaining_budget:
                kept_old_indices.append(idx)
                used_tokens += score.token_estimate

        # Sort by index to keep chronological order
        kept_old_indices.sort()

        # Assemble the final result
        kept_indices = kept_old_indices + recent_indices
        kept_messages = [messages[i] for i in kept_indices]
        kept_tokens = used_tokens + recent_tokens
        dropped_count = len(messages) - len(kept_messages)

        return kept_messages, {
            "strategy": "sliding_window",
            "kept": len(kept_messages),
            "dropped": dropped_count,
            "kept_old": len(kept_old_indices),
            "kept_recent": len(recent_indices),
            "total_tokens": kept_tokens,
            "budget_tokens": available,
            "utilization": kept_tokens / available if available > 0 else 0.0,
        }

    def get_window_stats(
        self,
        messages: list[dict[str, str]],
        model: str,
        system_tokens: int = 0,
    ) -> dict[str, Any]:
        """Compute window statistics without modifying messages.

        Useful for the context bar and diagnostics.

        Args:
            messages: List of messages
            model: Model name
            system_tokens: System prompt tokens

        Returns:
            Dict with detailed statistics
        """
        budget = self._budget_manager.get_budget(model)
        available = budget.available_for_history(system_tokens)

        scores = [
            self._score_message(msg, i, len(messages))
            for i, msg in enumerate(messages)
        ]

        total_tokens = sum(s.token_estimate for s in scores)
        code_count = sum(1 for s in scores if s.has_code)
        artifact_count = sum(1 for s in scores if s.has_artifact)
        summary_count = sum(1 for s in scores if s.is_summary)

        return {
            "message_count": len(messages),
            "total_tokens": total_tokens,
            "budget_tokens": available,
            "context_window": budget.context_window,
            "needs_trimming": total_tokens > available,
            "overflow_tokens": max(0, total_tokens - available),
            "utilization": total_tokens / available if available > 0 else 0.0,
            "code_messages": code_count,
            "artifact_messages": artifact_count,
            "summary_messages": summary_count,
            "model": model,
        }


# Singleton global
sliding_window_manager = SlidingWindowManager()
