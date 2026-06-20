#!/usr/bin/env python3
"""
CONTEXT SUMMARY -- OPTI-OIGNON 1.4.0
====================================

Intelligent context summarization for the sliding window.

Instead of dropping old messages when the context window fills up,
this module compresses them into a compact summary that preserves
key information: facts, decisions, code references, and user intent.

Architecture:
    - ContextSummarizer: main class, stateless, thread-safe
    - summarize_messages(): compress N messages into ~300 tokens
    - Cumulative summaries: merges existing summary with new messages
    - Fallback: returns None on failure -> executor falls back to drop

Author: Léon
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Import Ollama -- needed to call the summary model
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama unavailable -- context_summary disabled")

# Token estimation -- reuses context_manager if available
try:
    from .context_manager import estimate_tokens as cm_estimate_tokens
    CM_AVAILABLE = True
except ImportError:
    CM_AVAILABLE = False


# =============================================================================
# SUMMARY PROMPTS
# =============================================================================

SUMMARY_SYSTEM_PROMPT = """You are a conversation summarizer. Your task is to compress a conversation into a minimal but information-dense summary.

## RULES
1. Extract key FACTS, DECISIONS, and CONTEXT from the messages
2. Preserve TECHNICAL DETAILS: code snippets referenced, file names, error messages, tools used
3. Note the user's INTENT and any UNRESOLVED questions
4. Be extremely concise: target 100-250 words
5. Use bullet points or short sentences
6. Write in the SAME LANGUAGE as the conversation (French if French, English if English)
7. Never invent information not present in the messages
8. Prioritize recent and actionable information over small talk

## OUTPUT FORMAT
Write a single compact paragraph or short bullet list. No preamble, no "Here is the summary:", just the summary itself."""

CUMULATIVE_SUMMARY_PROMPT = """You are a conversation summarizer. You have an existing summary of earlier conversation, and new messages to incorporate.

## EXISTING SUMMARY
{existing_summary}

## TASK
Merge the existing summary with the new messages below into ONE updated summary.
- Keep all important facts from the existing summary
- Add new information from the new messages
- Remove redundant or superseded information
- Target 150-300 words total
- Write in the SAME LANGUAGE as the conversation
- No preamble, just output the merged summary."""


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class ContextSummarizer:
    """Summarizes conversation history to compress context.

    Uses a lightweight model (qwen3:8b by default) to summarize
    the old messages instead of dropping them.

    Thread-safe: can be called from the execution thread
    of the executor safely.
    """

    # --- Configuration ---
    SUMMARY_MODEL = "qwen3:8b"              # Fast model for summaries
    FALLBACK_MODELS = [                     # Fallback chain
        "qwen3:8b",
        "nemotron-3-nano:8b",
        "qwen3:4b",
        "qwen3:1.7b",
    ]
    SUMMARY_TEMPERATURE = 0.3               # Low for factual output
    MAX_SUMMARY_TOKENS = 400                # Cible : ~300 tokens output
    SUMMARY_TIMEOUT = 15                    # Timeout en secondes
    MAX_INPUT_TOKENS = 4000                 # Max tokens for messages to summarize
    SUMMARY_THRESHOLD = 4                   # Min messages before summarizing

    def __init__(self):
        """Initialize the summarizer."""
        self._lock = threading.Lock()
        self._available_model: str | None = None  # Cache of verified model
        self._model_checked_at: float = 0.0
        self._model_cache_ttl: float = 300.0  # Re-check every 5 min

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if CM_AVAILABLE:
            return cm_estimate_tokens(text, None)
        return len(text) // 4

    def _find_available_model(self) -> str | None:
        """Find a suitable model for summarization.

        Checks cache first, then queries Ollama for available models.

        Returns:
            Model name string, or None if no model available
        """
        # Cache valid?
        now = time.time()
        if self._available_model and (now - self._model_checked_at) < self._model_cache_ttl:
            return self._available_model

        if not OLLAMA_AVAILABLE:
            return None

        try:
            # List loaded/available models
            models_response = ollama.list()
            available_names = set()
            if hasattr(models_response, "models"):
                # Format ollama-python >= 0.3
                for m in models_response.models:
                    available_names.add(m.model if hasattr(m, "model") else str(m))
            elif isinstance(models_response, dict) and "models" in models_response:
                for m in models_response["models"]:
                    name = m.get("model", m.get("name", ""))
                    available_names.add(name)

            # Search in order of preference
            for candidate in self.FALLBACK_MODELS:
                # Correspondance exacte ou partielle (qwen3:8b match qwen3:8b-q4_K_M)
                if candidate in available_names:
                    self._available_model = candidate
                    self._model_checked_at = now
                    logger.info(f"Summary model selected: {candidate}")
                    return candidate
                # Prefix match (e.g. "qwen3:8b" matches "qwen3:8b-q5_K_M")
                for avail in available_names:
                    if avail.startswith(candidate.split(":")[0] + ":"):
                        self._available_model = avail
                        self._model_checked_at = now
                        logger.info(f"Summary model (partial match): {avail}")
                        return avail

            # No preferred model found -- using first available
            if available_names:
                first = sorted(available_names)[0]
                self._available_model = first
                self._model_checked_at = now
                logger.warning(
                    f"No preferred summary model, using: {first}"
                )
                return first

        except Exception as e:
            logger.error(f"Error during model search: {e}")

        return None

    def _format_messages_for_summary(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Format messages as a readable text block for the summarizer.

        Args:
            messages: List of {role, content} dicts

        Returns:
            Formatted string like:
            User: ...
            Assistant: ...
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "").strip()
            if content:
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _truncate_input(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> list[dict[str, str]]:
        """Truncate messages to fit within token budget.

        If messages exceed max_tokens, remove the oldest
        first (keep the most recent which are more relevant).

        Args:
            messages: Messages to potentially truncate
            max_tokens: Maximum token budget

        Returns:
            Truncated list of messages
        """
        total = sum(self._estimate_tokens(m.get("content", "")) for m in messages)

        if total <= max_tokens:
            return messages

        # Remove from the beginning (oldest)
        truncated = list(messages)
        while total > max_tokens and len(truncated) > 1:
            removed = truncated.pop(0)
            total -= self._estimate_tokens(removed.get("content", ""))

        logger.info(
            f"Truncated messages for summary: "
            f"{len(messages)} -> {len(truncated)} messages"
        )
        return truncated

    def summarize_messages(
        self,
        messages: list[dict[str, str]],
        existing_summary: str | None = None,
        model: str | None = None,
    ) -> str | None:
        """Summarize a list of messages into a compact paragraph.

        If existing_summary is provided, merge the existing summary
        with the new messages (cumulative summary).

        Args:
            messages: List of {"role": ..., "content": ...} to summarize
            existing_summary: Previous summary to incorporate
            model: Override summary model (otherwise auto-detection)

        Returns:
            Compact summary string (~300 tokens), or None on failure
        """
        if not messages:
            logger.warning("No messages to summarize")
            return None

        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama unavailable -- summarization impossible")
            return None

        # Model selection
        summary_model = model or self._find_available_model()
        if not summary_model:
            logger.warning("No model available for summarization")
            return None

        # Truncate the messages if too long
        truncated_messages = self._truncate_input(messages, self.MAX_INPUT_TOKENS)

        # Format the messages
        formatted = self._format_messages_for_summary(truncated_messages)
        input_tokens = self._estimate_tokens(formatted)

        # Choose prompt (simple vs cumulative)
        if existing_summary:
            system_prompt = CUMULATIVE_SUMMARY_PROMPT.format(
                existing_summary=existing_summary
            )
            log_prefix = "Cumulative summary"
        else:
            system_prompt = SUMMARY_SYSTEM_PROMPT
            log_prefix = "Context summary"

        logger.info(
            f"{log_prefix}: {len(messages)} messages "
            f"(~{input_tokens} tokens) -> model {summary_model}"
        )

        # Call model with timeout
        start_time = time.time()
        try:
            response = ollama.chat(
                model=summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted},
                ],
                options={
                    "temperature": self.SUMMARY_TEMPERATURE,
                    "num_predict": self.MAX_SUMMARY_TOKENS,
                },
            )

            elapsed = time.time() - start_time

            # Timeout check (call is blocking, but log if slow)
            if elapsed > self.SUMMARY_TIMEOUT:
                logger.warning(
                    f"Slow summary: {elapsed:.1f}s "
                    f"(timeout = {self.SUMMARY_TIMEOUT}s)"
                )

            summary = response["message"]["content"].strip()

            # Cleanup: strip the think tags if qwen3 is in think mode
            summary = self._clean_think_tags(summary)

            # Validation basique
            if not summary or len(summary) < 10:
                logger.warning(f"Summary too short or empty: '{summary[:50]}'")
                return None

            summary_tokens = self._estimate_tokens(summary)

            # Log result
            if existing_summary:
                existing_tokens = self._estimate_tokens(existing_summary)
                logger.info(
                    f"{log_prefix}: merged with existing summary "
                    f"({existing_tokens}t) -> {summary_tokens}t "
                    f"({elapsed:.1f}s)"
                )
            else:
                logger.info(
                    f"{log_prefix}: compressed {len(messages)} messages "
                    f"(~{input_tokens}t) -> {summary_tokens}t "
                    f"({elapsed:.1f}s)"
                )

            return summary

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Error during summarization ({elapsed:.1f}s): {e}"
            )
            return None

    def _clean_think_tags(self, text: str) -> str:
        """Remove <think>...</think> blocks from qwen3 responses.

        qwen3 in non-/nothink mode may still insert
        thinking blocks. We remove them from the summary.

        Args:
            text: Raw response text

        Returns:
            Cleaned text without think blocks
        """
        import re
        # Strip the <think>...</think> blocks (multiline)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Also strip the orphan tags
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")
        return cleaned.strip()

    def create_summary_message(self, summary: str) -> dict[str, str]:
        """Create a system-role message containing the summary.

        The format is recognized by the executor to detect summaries
        existing during cumulative summaries.

        Args:
            summary: Summary text

        Returns:
            Dict with role="system" and formatted content
        """
        return {
            "role": "system",
            "content": f"[Summary of earlier conversation]\n{summary}",
        }

    @staticmethod
    def is_summary_message(message: dict[str, str]) -> bool:
        """Check if a message is a context summary.

        Args:
            message: Message dict with role and content

        Returns:
            True if the message is a summary
        """
        return (
            message.get("role") == "system"
            and "[Summary" in message.get("content", "")
        )

    @staticmethod
    def extract_summary_text(message: dict[str, str]) -> str | None:
        """Extract the summary text from a summary message.

        Args:
            message: A summary message (as returned by create_summary_message)

        Returns:
            The summary text without the header, or None if not a summary
        """
        if not ContextSummarizer.is_summary_message(message):
            return None
        content = message.get("content", "")
        # Strip the "[Summary of earlier conversation]\n" header
        lines = content.split("\n", 1)
        if len(lines) > 1:
            return lines[1].strip()
        return content.strip()


# =============================================================================
# INSTANCE GLOBALE
# =============================================================================

context_summarizer = ContextSummarizer()

# Convenience functions
summarize_messages = context_summarizer.summarize_messages
is_summary_message = ContextSummarizer.is_summary_message
extract_summary_text = ContextSummarizer.extract_summary_text
