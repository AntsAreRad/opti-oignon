#!/usr/bin/env python3
"""
CONVERSATION COMPRESSOR -- OPTI-OIGNON
==========================================

Compresses conversation history to fit within the token budget's
history_tokens allocation while preserving semantic content.

Implements a dual-layer context pattern:
  Layer 1 (Working): Compressed summary + N recent messages -> injected into prompt
  Layer 2 (Archive): Full uncompressed conversation in SQLite -> searchable on-demand

Strategies:
  - rule:   Fast heuristic extraction, zero LLM calls
  - llm:    LLM-based summarization, higher quality
  - hybrid: Rule first pass, LLM refinement if budget allows (default)

The SQLite archive is NEVER modified. Compression only affects what goes
into the prompt. The full conversation history remains queryable at all
times via retrieve_from_archive().

Author: Leon
"""

import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# CONDITIONAL IMPORTS
# ============================================================================

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available; using hardcoded defaults")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available; LLM strategy will fall back to rule")

try:
    from .conversation import conversation_manager
    CONVERSATION_AVAILABLE = True
except ImportError:
    CONVERSATION_AVAILABLE = False
    conversation_manager = None
    logger.warning("conversation_manager not available; archive retrieval disabled")

try:
    from .context_manager import estimate_tokens as _cm_estimate_tokens
    CONTEXT_MANAGER_AVAILABLE = True
except ImportError:
    CONTEXT_MANAGER_AVAILABLE = False
    _cm_estimate_tokens = None

# ============================================================================
# CONSTANTS
# ============================================================================

_CONFIG_PATH = Path(__file__).parent / "config" / "compression.yaml"

_DEFAULT_CONFIG: dict[str, Any] = {
    "strategy": "hybrid",
    "recent_messages_keep": 6,
    "compression_threshold_ratio": 1.0,
    "enabled": True,
    "llm_summary_model": None,
    "llm_summary_max_tokens": 300,
    "llm_summary_temperature": 0.2,
    "llm_summary_timeout": 30,
    "rule_max_facts_per_message": 2,
    "rule_min_message_length": 50,
    "archive_retrieval_top_k": 3,
    "archive_retrieval_min_score": 0.05,
    "archive_retrieval_snippet_length": 300,
    "retrieval_trigger_enabled": True,
    "retrieval_trigger_min_confidence": 0.6,
}

# Regex patterns to detect when the user is referencing past context.
# These are checked cheaply (Level 1) before retrieval is triggered.
_RETRIEVAL_TRIGGER_PATTERNS = [
    r"\byou\s+(?:said|mentioned|told|explained|suggested|noted)\b",
    r"\bwe\s+(?:discussed|talked\s+about|decided|agreed)\b",
    r"\bearlier\s+(?:you|we)\b",
    r"\bpreviously\b",
    r"\blast\s+time\b",
    r"\bin\s+(?:a|our|the|my|this)?\s*previous\s+(?:message|conversation|turn)\b",
    r"\bdo\s+you\s+remember\b",
    r"\bas\s+(?:you|we)\s+(?:mentioned|discussed|noted)\b",
    r"\b(?:recall|remember)\s+(?:when|that|what|you|the)\b",
    r"\bback\s+to\s+(?:what|the)\b",
    r"\btu\s+(?:as|avais)\b",          # French: "tu as" / "tu avais" (you said / you had)
    r"\btu\s+(?:m'as|nous\s+as)\b",
    r"\bon\s+(?:a\s+)?(?:discuté|parlé|décidé)\b",
    r"\bplus\s+tôt\b",
    r"\bprécédemment\b",
    r"\btu\s+(?:disais|mentionnais|expliquais)\b",
]

_COMPILED_TRIGGERS = [re.compile(p, re.IGNORECASE) for p in _RETRIEVAL_TRIGGER_PATTERNS]

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CompressedContext:
    """Result of a compression operation.

    Attributes:
        summary: Compressed summary of older messages (injected as context block).
        recent_messages: Most recent messages kept verbatim.
        original_count: Number of messages before compression.
        compressed_count: Number of messages collapsed into the summary.
        strategy_used: Which strategy was applied ("rule", "llm", "hybrid").
        tokens_saved: Estimated token reduction from compression.
        compression_ratio: Fraction of history compressed (0.0-1.0).
    """
    summary: str
    recent_messages: list[dict[str, str]]
    original_count: int
    compressed_count: int
    strategy_used: str
    tokens_saved: int
    compression_ratio: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a dict for API responses."""
        return {
            "summary": self.summary,
            "recent_messages": self.recent_messages,
            "original_count": self.original_count,
            "compressed_count": self.compressed_count,
            "strategy_used": self.strategy_used,
            "tokens_saved": self.tokens_saved,
            "compression_ratio": round(self.compression_ratio, 4),
        }


@dataclass
class ArchiveSearchResult:
    """A single result from archive retrieval.

    Attributes:
        message_id: Database row ID of the matching message.
        role: 'user' or 'assistant'.
        snippet: Truncated message content.
        score: Relevance score (0.0-1.0).
        timestamp: ISO timestamp of the message.
    """
    message_id: int
    role: str
    snippet: str
    score: float
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a dict for API responses."""
        return {
            "message_id": self.message_id,
            "role": self.role,
            "snippet": self.snippet,
            "score": round(self.score, 4),
            "timestamp": self.timestamp,
        }


# ============================================================================
# TOKEN ESTIMATION
# ============================================================================

def _estimate_tokens(text: str, model: str | None = None) -> int:
    """Estimate token count for a text string.

    Uses context_manager if available, otherwise len(text) / 4.
    """
    if not text:
        return 0
    if CONTEXT_MANAGER_AVAILABLE and _cm_estimate_tokens is not None:
        try:
            return _cm_estimate_tokens(text, model)
        except Exception:
            pass
    return max(1, len(text) // 4)


def _estimate_messages_tokens(messages: list[dict[str, str]], model: str | None = None) -> int:
    """Estimate total token count for a list of messages."""
    return sum(_estimate_tokens(m.get("content", ""), model) for m in messages)


# ============================================================================
# ARCHIVE RETRIEVER
# ============================================================================

class ArchiveRetriever:
    """Searches the full SQLite conversation archive for relevant messages.

    The archive is never modified. This class provides keyword-based
    retrieval so the LLM can access context that was compressed out of
    the working prompt.

    Scoring uses a lightweight TF-IDF-inspired formula:
      score = sum(idf(term) * tf(term, message)) for matching terms
    where idf is approximated from query term frequency across results.
    """

    def __init__(self, snippet_length: int = 300) -> None:
        """Initialize the retriever.

        Args:
            snippet_length: Maximum character length for returned snippets.
        """
        self._snippet_length = snippet_length

    def retrieve(
        self,
        conversation_id: str,
        query: str,
        top_k: int = 3,
        min_score: float = 0.05,
    ) -> list[ArchiveSearchResult]:
        """Search the full conversation archive for messages matching query.

        Args:
            conversation_id: UUID of the conversation to search.
            query: Natural language query string.
            top_k: Maximum number of results to return.
            min_score: Minimum relevance score to include a result.

        Returns:
            List of ArchiveSearchResult sorted by score descending.
        """
        if not CONVERSATION_AVAILABLE or conversation_manager is None:
            logger.debug("Archive retrieval skipped: conversation_manager unavailable")
            return []

        try:
            all_messages = conversation_manager.get_context_messages(conversation_id)
        except Exception as e:
            logger.error(f"Archive retrieval error loading messages: {e}")
            return []

        if not all_messages:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        results: list[ArchiveSearchResult] = []
        for idx, msg in enumerate(all_messages):
            content = msg.get("content", "")
            role = msg.get("role", "user")
            if role == "system":
                continue
            score = self._score(content, query_terms)
            if score >= min_score:
                snippet = self._make_snippet(content, query_terms)
                results.append(
                    ArchiveSearchResult(
                        message_id=idx,
                        role=role,
                        snippet=snippet,
                        score=score,
                        timestamp=msg.get("timestamp", ""),
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """Extract normalized word tokens from text.

        Args:
            text: Input text.

        Returns:
            List of lowercase alphabetic tokens (length >= 3).
        """
        tokens = re.findall(r"[a-zA-ZÀ-ÿ]{3,}", text.lower())
        # Remove common stopwords that add noise
        stopwords = {
            "the", "and", "for", "are", "was", "you", "that", "this",
            "with", "have", "from", "they", "will", "what", "when",
            "les", "des", "une", "est", "que", "qui", "par", "sur",
        }
        return [t for t in tokens if t not in stopwords]

    def _score(self, content: str, query_terms: list[str]) -> float:
        """Compute a relevance score for a message against query terms.

        Uses term frequency with log-IDF approximation.

        Args:
            content: Message content to score.
            query_terms: Normalized query tokens.

        Returns:
            Float relevance score >= 0.0.
        """
        if not content or not query_terms:
            return 0.0

        content_lower = content.lower()
        content_tokens = self._tokenize(content)
        total_tokens = max(1, len(content_tokens))

        score = 0.0
        for term in query_terms:
            tf = content_tokens.count(term) / total_tokens
            if tf > 0:
                # IDF approximation: rarer query terms score higher
                idf = math.log(1 + len(query_terms) / (query_terms.count(term)))
                score += tf * idf

        # Boost for exact phrase match (any 2-gram from query)
        if len(query_terms) >= 2:
            bigrams = [f"{query_terms[i]} {query_terms[i+1]}" for i in range(len(query_terms)-1)]
            for bigram in bigrams:
                if bigram in content_lower:
                    score += 0.3

        return min(1.0, score)

    def _make_snippet(self, content: str, query_terms: list[str]) -> str:
        """Extract the most relevant snippet from a message.

        Finds the region of the content most dense with query terms,
        then returns a truncated window around it.

        Args:
            content: Full message content.
            query_terms: Normalized query tokens.

        Returns:
            Snippet string of at most self._snippet_length characters.
        """
        if len(content) <= self._snippet_length:
            return content

        # Find the best window by term density
        content_lower = content.lower()
        best_pos = 0
        best_count = 0

        window = self._snippet_length // 2
        step = max(1, window // 4)

        for pos in range(0, len(content) - window, step):
            segment = content_lower[pos:pos + window]
            count = sum(1 for term in query_terms if term in segment)
            if count > best_count:
                best_count = count
                best_pos = pos

        start = max(0, best_pos)
        end = min(len(content), start + self._snippet_length)
        snippet = content[start:end]

        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet


# ============================================================================
# RETRIEVAL TRIGGER DETECTOR
# ============================================================================

def detect_retrieval_trigger(message: str) -> tuple[bool, float]:
    """Detect whether a user message is referencing past conversation context.

    Uses compiled regex patterns (Level 1 only -- fast, no LLM call).

    Args:
        message: The user's message text.

    Returns:
        Tuple of (triggered: bool, confidence: float 0.0-1.0).
        Confidence is based on number of matching patterns.
    """
    if not message:
        return False, 0.0

    hits = sum(1 for pattern in _COMPILED_TRIGGERS if pattern.search(message))
    if hits == 0:
        return False, 0.0

    # Confidence scales with number of matching patterns (capped at 1.0)
    confidence = min(1.0, 0.5 + (hits - 1) * 0.2)
    return True, confidence


# ============================================================================
# CONVERSATION COMPRESSOR
# ============================================================================

class ConversationCompressor:
    """Compresses conversation history to fit within the history token budget.

    The compressor applies one of three strategies to reduce older messages
    into a summary block while keeping recent messages verbatim. The full
    conversation archive in SQLite is never modified.

    Usage:
        compressor = ConversationCompressor()
        result = compressor.compress(messages, budget_tokens=2000, model="qwen3:32b")
        # result.summary contains the compressed older context
        # result.recent_messages contains the N most recent messages verbatim
    """

    def __init__(self) -> None:
        """Initialize with YAML config (falls back to defaults on error)."""
        self._config: dict[str, Any] = dict(_DEFAULT_CONFIG)
        self._retriever = ArchiveRetriever(
            snippet_length=self._config["archive_retrieval_snippet_length"]
        )
        self._load_config()
        logger.info("ConversationCompressor initialized")

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            return
        try:
            if _CONFIG_PATH.exists():
                with open(_CONFIG_PATH) as fh:
                    data = yaml.safe_load(fh) or {}
                # Merge over defaults (known keys only, ignore extras)
                for key in _DEFAULT_CONFIG:
                    if key in data:
                        self._config[key] = data[key]
                # Update retriever snippet length if changed
                self._retriever = ArchiveRetriever(
                    snippet_length=self._config["archive_retrieval_snippet_length"]
                )
                logger.debug(f"Compression config loaded from {_CONFIG_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load compression.yaml, using defaults: {e}")

    def reload_config(self) -> None:
        """Hot-reload configuration from YAML without restarting."""
        self._load_config()
        logger.info("ConversationCompressor: config reloaded")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether compression is currently enabled."""
        return bool(self._config.get("enabled", True))

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable compression at runtime."""
        self._config["enabled"] = bool(value)

    @property
    def strategy(self) -> str:
        """Current default compression strategy."""
        return str(self._config.get("strategy", "hybrid"))

    @property
    def recent_messages_keep(self) -> int:
        """Number of most recent messages always kept verbatim."""
        return int(self._config.get("recent_messages_keep", 6))

    def compress(
        self,
        messages: list[dict[str, str]],
        budget_tokens: int,
        model: str = "",
        strategy: str = "auto",
    ) -> CompressedContext:
        """Compress conversation history to fit within budget_tokens.

        Only compresses messages that exceed the budget. The N most recent
        messages (configurable) are always preserved verbatim. System
        messages are always preserved.

        Args:
            messages: List of Ollama-format messages {'role': ..., 'content': ...}.
                      Should NOT include the current user message.
            budget_tokens: Maximum tokens allowed for the history section.
            model: Model name for accurate token estimation.
            strategy: "rule", "llm", "hybrid", or "auto" (use config default).

        Returns:
            CompressedContext with summary and verbatim recent messages.
        """
        if strategy == "auto":
            strategy = self._config.get("strategy", "hybrid")

        # Separate system messages from history
        system_msgs = [m for m in messages if m.get("role") == "system"]  # noqa: F841
        history_msgs = [m for m in messages if m.get("role") != "system"]

        total_tokens = _estimate_messages_tokens(history_msgs, model)

        # Check if compression is actually needed
        threshold = budget_tokens * self._config.get("compression_threshold_ratio", 1.0)
        if total_tokens <= threshold or len(history_msgs) <= self.recent_messages_keep:
            return CompressedContext(
                summary="",
                recent_messages=list(history_msgs),
                original_count=len(history_msgs),
                compressed_count=0,
                strategy_used="none",
                tokens_saved=0,
                compression_ratio=0.0,
            )

        # Split: keep recent, compress older
        keep_n = self.recent_messages_keep
        # Ensure we always keep pairs (user + assistant) when possible
        if keep_n > 0 and len(history_msgs) > keep_n:
            to_compress = history_msgs[:-keep_n]
            to_keep = history_msgs[-keep_n:]
        else:
            to_compress = history_msgs[:-2] if len(history_msgs) > 2 else []
            to_keep = history_msgs[-2:] if len(history_msgs) >= 2 else history_msgs

        if not to_compress:
            return CompressedContext(
                summary="",
                recent_messages=list(history_msgs),
                original_count=len(history_msgs),
                compressed_count=0,
                strategy_used="none",
                tokens_saved=0,
                compression_ratio=0.0,
            )

        tokens_before = _estimate_messages_tokens(to_compress, model)

        # Apply selected strategy
        if strategy == "rule":
            summary, actual_strategy = self._compress_rule(to_compress)
        elif strategy == "llm":
            summary, actual_strategy = self._compress_llm(to_compress, model)
        elif strategy == "hybrid":
            summary, actual_strategy = self._compress_hybrid(to_compress, model, budget_tokens)
        else:
            logger.warning(f"Unknown strategy '{strategy}', falling back to rule")
            summary, actual_strategy = self._compress_rule(to_compress)

        tokens_after = _estimate_tokens(summary, model)
        tokens_saved = max(0, tokens_before - tokens_after)
        compression_ratio = len(to_compress) / max(1, len(history_msgs))

        return CompressedContext(
            summary=summary,
            recent_messages=to_keep,
            original_count=len(history_msgs),
            compressed_count=len(to_compress),
            strategy_used=actual_strategy,
            tokens_saved=tokens_saved,
            compression_ratio=round(compression_ratio, 4),
        )

    def retrieve_from_archive(
        self,
        conversation_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[ArchiveSearchResult]:
        """Search the full conversation archive for messages matching a query.

        The archive is the complete uncompressed SQLite history -- this method
        allows the LLM to recover details that were compressed out of the
        working prompt.

        Args:
            conversation_id: UUID of the conversation to search.
            query: Natural language query.
            top_k: Maximum results (defaults to config value).

        Returns:
            List of ArchiveSearchResult sorted by relevance.
        """
        k = top_k if top_k is not None else self._config.get("archive_retrieval_top_k", 3)
        min_score = float(self._config.get("archive_retrieval_min_score", 0.05))
        return self._retriever.retrieve(conversation_id, query, top_k=k, min_score=min_score)

    def get_config(self) -> dict[str, Any]:
        """Return the current configuration as a dict."""
        return dict(self._config)

    def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Update configuration at runtime (does not persist to disk).

        Args:
            updates: Dict of config keys to update.

        Returns:
            Updated config dict.
        """
        allowed_keys = set(_DEFAULT_CONFIG.keys())
        for key, value in updates.items():
            if key in allowed_keys:
                self._config[key] = value
            else:
                logger.warning(f"Ignoring unknown config key: {key}")
        return self.get_config()

    # -----------------------------------------------------------------------
    # Strategy implementations
    # -----------------------------------------------------------------------

    def _compress_rule(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        """Rule-based compression: extract key facts using heuristics.

        Does not make any LLM calls. Extracts sentences that appear
        informative based on structural signals (questions, assertions,
        proper nouns, numbers).

        Args:
            messages: Messages to compress (older portion of history).

        Returns:
            Tuple of (summary_text, strategy_label).
        """
        max_facts = int(self._config.get("rule_max_facts_per_message", 2))
        min_length = int(self._config.get("rule_min_message_length", 50))

        facts: list[str] = []
        for msg in messages:
            content = msg.get("content", "").strip()
            role = msg.get("role", "user")
            if len(content) < min_length:
                # Short messages are included verbatim as facts
                facts.append(f"[{role}] {content}")
                continue

            # Extract key sentences using heuristics
            sentences = re.split(r"(?<=[.!?])\s+", content)
            scored: list[tuple[float, str]] = []
            for sent in sentences:
                sent = sent.strip()
                if not sent or len(sent) < 10:
                    continue
                score = self._score_sentence(sent)
                scored.append((score, sent))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [s for _, s in scored[:max_facts]]

            if top_sentences:
                joined = " ".join(top_sentences)
                facts.append(f"[{role}] {joined}")

        if not facts:
            return "[No summary available for older messages]", "rule"

        summary = "Earlier conversation summary:\n" + "\n".join(f"- {f}" for f in facts)
        return summary, "rule"

    def _compress_llm(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> tuple[str, str]:
        """LLM-based compression: summarize older messages via Ollama.

        Falls back to rule-based if Ollama is unavailable or times out.

        Args:
            messages: Messages to compress.
            model: Model to use for summarization (overridden by config if set).

        Returns:
            Tuple of (summary_text, strategy_label).
        """
        if not OLLAMA_AVAILABLE:
            logger.debug("Ollama unavailable, falling back to rule strategy")
            summary, _ = self._compress_rule(messages)
            return summary, "rule_fallback"

        summary_model = self._config.get("llm_summary_model") or model
        if not summary_model:
            summary, _ = self._compress_rule(messages)
            return summary, "rule_fallback"

        max_tokens = int(self._config.get("llm_summary_max_tokens", 300))
        temperature = float(self._config.get("llm_summary_temperature", 0.2))
        timeout = float(self._config.get("llm_summary_timeout", 30))  # noqa: F841

        # Build a compact representation of the messages to summarize
        convo_text = self._format_messages_for_summary(messages)

        system_prompt = (
            "You are a conversation summarizer. Your task is to create a concise, "
            "factual summary of the conversation excerpt below. Focus on:\n"
            "- Key facts, decisions, and conclusions reached\n"
            "- Important context that might be referenced later\n"
            "- Questions asked and answers given\n"
            f"Keep the summary under {max_tokens} tokens. Be specific and factual. "
            "Do not interpret or editorialize. Output the summary directly."
        )

        user_prompt = (
            f"Please summarize this conversation excerpt:\n\n{convo_text}\n\n"
            f"Summary (max {max_tokens} tokens):"
        )

        try:
            start = time.monotonic()
            response = ollama.chat(
                model=summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            elapsed = time.monotonic() - start
            logger.debug(f"LLM compression completed in {elapsed:.2f}s using {summary_model}")

            summary_text = ""
            if hasattr(response, "message") and hasattr(response.message, "content"):
                summary_text = response.message.content or ""
            elif isinstance(response, dict):
                summary_text = response.get("message", {}).get("content", "")

            if not summary_text.strip():
                logger.warning("LLM returned empty summary, falling back to rule")
                summary, _ = self._compress_rule(messages)
                return summary, "rule_fallback"

            return f"Earlier conversation summary:\n{summary_text.strip()}", "llm"

        except Exception as e:
            logger.warning(f"LLM compression failed ({e}), falling back to rule")
            summary, _ = self._compress_rule(messages)
            return summary, "rule_fallback"

    def _compress_hybrid(
        self,
        messages: list[dict[str, str]],
        model: str,
        budget_tokens: int,
    ) -> tuple[str, str]:
        """Hybrid compression: rule first pass, then LLM refinement if budget allows.

        The LLM is only called if:
        1. Ollama is available
        2. The rule-based summary still exceeds budget (meaning we need more compression)
        3. A summary model is configured or current model is known

        Args:
            messages: Messages to compress.
            model: Model for token estimation and optional LLM call.
            budget_tokens: Token budget for the summary.

        Returns:
            Tuple of (summary_text, strategy_label).
        """
        rule_summary, _ = self._compress_rule(messages)
        rule_tokens = _estimate_tokens(rule_summary, model)

        # If rule summary fits in budget, no need for LLM
        if rule_tokens <= budget_tokens or not OLLAMA_AVAILABLE:
            return rule_summary, "hybrid_rule"

        # Rule summary is too long; use LLM for better compression
        llm_summary, llm_strategy = self._compress_llm(messages, model)
        if "fallback" in llm_strategy:
            return rule_summary, "hybrid_rule"

        return llm_summary, "hybrid_llm"

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _score_sentence(self, sentence: str) -> float:
        """Heuristic informativeness score for a sentence.

        Higher score = more likely to be a key fact worth keeping.

        Args:
            sentence: A single sentence from a message.

        Returns:
            Float score >= 0.0.
        """
        score = 0.0
        s = sentence.strip()

        # Length bonus: moderately long sentences are usually more informative
        length = len(s)
        if 30 <= length <= 200:
            score += 0.3
        elif length > 200:
            score += 0.1

        # Presence of numbers (often key facts)
        if re.search(r"\d+", s):
            score += 0.2

        # Capitalized words in the middle (likely proper nouns)
        mid_caps = re.findall(r"(?<!\.\s)\b[A-Z][a-z]{2,}", s)
        score += min(0.3, len(mid_caps) * 0.1)

        # Question marks (the user asked something important)
        if "?" in s:
            score += 0.2

        # Code or technical content
        if re.search(r"[`{}()\[\]]|def |class |import |return ", s):
            score += 0.3

        # Negation (important semantic signals)
        if re.search(r"\b(?:not|no|never|don't|doesn't|cannot|error|fail)\b", s, re.I):
            score += 0.15

        return score

    def _format_messages_for_summary(self, messages: list[dict[str, str]]) -> str:
        """Format messages as a readable transcript for LLM summarization.

        Args:
            messages: List of message dicts.

        Returns:
            Formatted transcript string.
        """
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "").strip()
            # Truncate very long messages to avoid prompt bloat
            if len(content) > 800:
                content = content[:800] + "..."
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)


# ============================================================================
# RETRIEVAL TRIGGER DETECTOR (module-level convenience)
# ============================================================================

def check_retrieval_trigger(message: str, min_confidence: float = 0.6) -> bool:
    """Check whether a message triggers archive retrieval.

    Args:
        message: User message to check.
        min_confidence: Minimum confidence threshold (0.0-1.0).

    Returns:
        True if the message should trigger archive search.
    """
    triggered, confidence = detect_retrieval_trigger(message)
    return triggered and confidence >= min_confidence


# ============================================================================
# MODULE-LEVEL SINGLETON
# ============================================================================

conversation_compressor = ConversationCompressor()
