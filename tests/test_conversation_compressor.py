"""
Tests for ConversationCompressor and ArchiveRetriever — Opti-Oignon S66.

Covers:
  - CompressedContext dataclass
  - ArchiveRetriever: tokenization, scoring, snippet extraction
  - ConversationCompressor: all three strategies, budget thresholds, edge cases
  - Retrieval trigger detection
  - Config loading and update
  - Module-level singleton
"""

import math
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# Helpers
# ============================================================================

def make_messages(n: int, role_pair: bool = True) -> list[dict[str, str]]:
    """Generate a list of mock conversation messages."""
    messages = []
    for i in range(n):
        if role_pair:
            messages.append({"role": "user", "content": f"User message {i}: What about topic {i}?"})
            messages.append({"role": "assistant", "content": f"Assistant reply {i}: Here is information about topic {i}. It involves numbers like {i*10} and concepts XYZ."})
        else:
            messages.append({"role": "user", "content": f"Message {i}"})
    return messages


# ============================================================================
# Imports
# ============================================================================

from opti_oignon.conversation_compressor import (
    ArchiveRetriever,
    ArchiveSearchResult,
    CompressedContext,
    ConversationCompressor,
    check_retrieval_trigger,
    conversation_compressor,
    detect_retrieval_trigger,
)


# ============================================================================
# CompressedContext tests
# ============================================================================

class TestCompressedContext:
    def test_as_dict_contains_all_fields(self):
        ctx = CompressedContext(
            summary="Summary text",
            recent_messages=[{"role": "user", "content": "hi"}],
            original_count=10,
            compressed_count=7,
            strategy_used="hybrid",
            tokens_saved=150,
            compression_ratio=0.7,
        )
        d = ctx.as_dict()
        assert d["summary"] == "Summary text"
        assert d["original_count"] == 10
        assert d["compressed_count"] == 7
        assert d["strategy_used"] == "hybrid"
        assert d["tokens_saved"] == 150
        assert abs(d["compression_ratio"] - 0.7) < 1e-4

    def test_as_dict_rounds_compression_ratio(self):
        ctx = CompressedContext(
            summary="",
            recent_messages=[],
            original_count=3,
            compressed_count=2,
            strategy_used="rule",
            tokens_saved=50,
            compression_ratio=0.666667,
        )
        d = ctx.as_dict()
        assert len(str(d["compression_ratio"]).split(".")[-1]) <= 4

    def test_zero_compression(self):
        ctx = CompressedContext(
            summary="",
            recent_messages=[{"role": "user", "content": "hi"}],
            original_count=1,
            compressed_count=0,
            strategy_used="none",
            tokens_saved=0,
        )
        assert ctx.tokens_saved == 0
        assert ctx.strategy_used == "none"


# ============================================================================
# ArchiveSearchResult tests
# ============================================================================

class TestArchiveSearchResult:
    def test_as_dict(self):
        result = ArchiveSearchResult(
            message_id=3,
            role="user",
            snippet="Some snippet",
            score=0.75432,
            timestamp="2024-01-01T00:00:00",
        )
        d = result.as_dict()
        assert d["message_id"] == 3
        assert d["role"] == "user"
        assert d["snippet"] == "Some snippet"
        assert abs(d["score"] - 0.7543) < 1e-3
        assert d["timestamp"] == "2024-01-01T00:00:00"


# ============================================================================
# ArchiveRetriever tests
# ============================================================================

class TestArchiveRetriever:
    def setup_method(self):
        self.retriever = ArchiveRetriever(snippet_length=200)

    def test_tokenize_basic(self):
        tokens = self.retriever._tokenize("The quick brown fox jumps")
        assert "quick" in tokens
        assert "brown" in tokens
        # Stopwords should be removed
        assert "the" not in tokens

    def test_tokenize_empty(self):
        assert self.retriever._tokenize("") == []

    def test_tokenize_removes_short_words(self):
        tokens = self.retriever._tokenize("hi go to the")
        # Words with < 3 chars removed
        assert "hi" not in tokens
        assert "go" not in tokens

    def test_score_zero_for_empty(self):
        score = self.retriever._score("", ["word"])
        assert score == 0.0

    def test_score_zero_for_no_match(self):
        score = self.retriever._score("completely unrelated text", ["python", "machine", "learning"])
        assert score == 0.0

    def test_score_positive_for_match(self):
        score = self.retriever._score("I was working on python code today", ["python", "code"])
        assert score > 0.0

    def test_score_bigram_boost(self):
        score_bigram = self.retriever._score("python code example", ["python", "code", "example"])
        score_no_bigram = self.retriever._score("python but also code here", ["python", "code", "example"])
        # Exact bigram "python code" should boost the first
        assert score_bigram >= score_no_bigram

    def test_score_capped_at_one(self):
        # Repeated matches should not exceed 1.0
        content = " ".join(["python"] * 100)
        score = self.retriever._score(content, ["python"] * 10)
        assert score <= 1.0

    def test_make_snippet_short_content(self):
        content = "Short text"
        snippet = self.retriever._make_snippet(content, ["short"])
        assert snippet == content

    def test_make_snippet_long_content(self):
        content = "A" * 500 + " relevant_keyword " + "B" * 500
        snippet = self.retriever._make_snippet(content, ["relevant", "keyword"])
        assert len(snippet) <= 250  # snippet_length=200 + ellipsis chars
        assert "relevant_keyword" in snippet

    def test_make_snippet_adds_ellipsis(self):
        content = "prefix " * 100 + "target content" + " suffix" * 100
        snippet = self.retriever._make_snippet(content, ["target", "content"])
        # Should have ellipsis at start or end (long content)
        assert "..." in snippet

    def test_retrieve_no_conversation_manager(self):
        """Should return empty list when conversation_manager unavailable."""
        with patch("opti_oignon.conversation_compressor.CONVERSATION_AVAILABLE", False):
            with patch("opti_oignon.conversation_compressor.conversation_manager", None):
                retriever = ArchiveRetriever()
                results = retriever.retrieve("conv-123", "query")
                assert results == []

    def test_retrieve_returns_sorted_by_score(self):
        """Results should be sorted descending by score."""
        mock_messages = [
            {"role": "user", "content": "I love machine learning and neural networks"},
            {"role": "assistant", "content": "Great choice, machine learning is powerful"},
            {"role": "user", "content": "What about cooking recipes?"},
            {"role": "assistant", "content": "Pasta is delicious"},
        ]
        mock_manager = MagicMock()
        mock_manager.get_context_messages.return_value = mock_messages

        with patch("opti_oignon.conversation_compressor.CONVERSATION_AVAILABLE", True):
            with patch("opti_oignon.conversation_compressor.conversation_manager", mock_manager):
                retriever = ArchiveRetriever()
                results = retriever.retrieve("conv-123", "machine learning neural networks")

        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

    def test_retrieve_filters_by_min_score(self):
        """Results below min_score should be excluded."""
        mock_messages = [
            {"role": "user", "content": "completely unrelated topic"},
            {"role": "assistant", "content": "yes indeed different"},
        ]
        mock_manager = MagicMock()
        mock_manager.get_context_messages.return_value = mock_messages

        with patch("opti_oignon.conversation_compressor.CONVERSATION_AVAILABLE", True):
            with patch("opti_oignon.conversation_compressor.conversation_manager", mock_manager):
                retriever = ArchiveRetriever()
                results = retriever.retrieve(
                    "conv-123",
                    "python bioinformatics genome",
                    min_score=0.5,
                )
        # Should return 0 results because no match
        assert all(r.score >= 0.5 for r in results)

    def test_retrieve_skips_system_messages(self):
        """System messages should not appear in results."""
        mock_messages = [
            {"role": "system", "content": "You are a python expert assistant"},
            {"role": "user", "content": "Tell me about python"},
        ]
        mock_manager = MagicMock()
        mock_manager.get_context_messages.return_value = mock_messages

        with patch("opti_oignon.conversation_compressor.CONVERSATION_AVAILABLE", True):
            with patch("opti_oignon.conversation_compressor.conversation_manager", mock_manager):
                retriever = ArchiveRetriever()
                results = retriever.retrieve("conv-123", "python expert")

        assert all(r.role != "system" for r in results)

    def test_retrieve_respects_top_k(self):
        """Should return at most top_k results."""
        mock_messages = [
            {"role": "user", "content": f"message about python topic {i}"}
            for i in range(20)
        ]
        mock_manager = MagicMock()
        mock_manager.get_context_messages.return_value = mock_messages

        with patch("opti_oignon.conversation_compressor.CONVERSATION_AVAILABLE", True):
            with patch("opti_oignon.conversation_compressor.conversation_manager", mock_manager):
                retriever = ArchiveRetriever()
                results = retriever.retrieve("conv-123", "python topic", top_k=3)

        assert len(results) <= 3

    def test_retrieve_handles_exception_gracefully(self):
        """Should return empty list if conversation_manager raises."""
        mock_manager = MagicMock()
        mock_manager.get_context_messages.side_effect = RuntimeError("DB error")

        with patch("opti_oignon.conversation_compressor.CONVERSATION_AVAILABLE", True):
            with patch("opti_oignon.conversation_compressor.conversation_manager", mock_manager):
                retriever = ArchiveRetriever()
                results = retriever.retrieve("conv-123", "query")

        assert results == []


# ============================================================================
# Retrieval trigger detection tests
# ============================================================================

class TestRetrievalTriggerDetection:
    def test_no_trigger_on_simple_message(self):
        triggered, confidence = detect_retrieval_trigger("What is the capital of France?")
        assert not triggered
        assert confidence == 0.0

    def test_trigger_on_you_said(self):
        triggered, confidence = detect_retrieval_trigger("You said the function was ready")
        assert triggered
        assert confidence >= 0.5

    def test_trigger_on_we_discussed(self):
        triggered, confidence = detect_retrieval_trigger("We discussed this approach earlier")
        assert triggered

    def test_trigger_on_earlier(self):
        triggered, confidence = detect_retrieval_trigger("Earlier you mentioned a different method")
        assert triggered

    def test_trigger_on_french_pattern(self):
        triggered, confidence = detect_retrieval_trigger("Tu m'as dit que le modele fonctionnait")
        assert triggered

    def test_trigger_on_do_you_remember(self):
        triggered, confidence = detect_retrieval_trigger("Do you remember what we decided?")
        assert triggered

    def test_confidence_scales_with_matches(self):
        single = detect_retrieval_trigger("You said this works")[1]
        multi = detect_retrieval_trigger("Earlier you said we discussed this, do you remember?")[1]
        assert multi >= single

    def test_confidence_capped_at_one(self):
        _, confidence = detect_retrieval_trigger(
            "You mentioned, we discussed, earlier you said, do you remember, as you noted, previously"
        )
        assert confidence <= 1.0

    def test_check_retrieval_trigger_threshold(self):
        # High confidence trigger
        assert check_retrieval_trigger("You said this earlier", min_confidence=0.5)
        # Low confidence threshold
        assert check_retrieval_trigger("You said this", min_confidence=0.3)
        # No trigger at all
        assert not check_retrieval_trigger("Hello, how are you?", min_confidence=0.5)

    def test_empty_message_no_trigger(self):
        triggered, confidence = detect_retrieval_trigger("")
        assert not triggered
        assert confidence == 0.0


# ============================================================================
# ConversationCompressor tests
# ============================================================================

class TestConversationCompressor:
    def setup_method(self):
        self.compressor = ConversationCompressor()
        # Enable by default
        self.compressor._config["enabled"] = True

    # --- Config tests ---

    def test_default_strategy(self):
        # Should default to "hybrid" from config
        assert self.compressor.strategy in ("rule", "llm", "hybrid")

    def test_enabled_property(self):
        self.compressor.enabled = False
        assert not self.compressor.enabled
        self.compressor.enabled = True
        assert self.compressor.enabled

    def test_get_config_returns_dict(self):
        cfg = self.compressor.get_config()
        assert isinstance(cfg, dict)
        assert "strategy" in cfg
        assert "recent_messages_keep" in cfg
        assert "enabled" in cfg

    def test_update_config_known_keys(self):
        cfg = self.compressor.update_config({"recent_messages_keep": 4})
        assert cfg["recent_messages_keep"] == 4

    def test_update_config_unknown_keys_ignored(self):
        original = self.compressor.get_config()
        self.compressor.update_config({"nonexistent_key": "value"})
        assert "nonexistent_key" not in self.compressor.get_config()

    def test_recent_messages_keep_property(self):
        self.compressor._config["recent_messages_keep"] = 8
        assert self.compressor.recent_messages_keep == 8

    # --- No compression needed ---

    def test_no_compression_when_under_budget(self):
        messages = make_messages(2)  # 4 messages, small
        result = self.compressor.compress(messages, budget_tokens=10000, model="")
        assert result.strategy_used == "none"
        assert result.compressed_count == 0
        assert result.tokens_saved == 0
        assert len(result.recent_messages) == len(messages)

    def test_no_compression_when_few_messages(self):
        messages = make_messages(1)  # 2 messages
        self.compressor._config["recent_messages_keep"] = 6
        result = self.compressor.compress(messages, budget_tokens=1, model="")
        # Only 2 messages <= keep_n=6, should not compress
        assert result.compressed_count == 0

    def test_no_compression_empty_history(self):
        result = self.compressor.compress([], budget_tokens=100, model="")
        assert result.compressed_count == 0
        assert result.recent_messages == []

    # --- Rule-based strategy ---

    def test_rule_compress_returns_summary(self):
        messages = make_messages(10)  # 20 messages
        summary, strategy = self.compressor._compress_rule(messages)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert strategy == "rule"

    def test_rule_compress_includes_role_labels(self):
        messages = [
            {"role": "user", "content": "I need help with bioinformatics analysis in R"},
            {"role": "assistant", "content": "Sure, I can help you with vegan and DESeq2 packages"},
        ]
        summary, _ = self.compressor._compress_rule(messages)
        assert "[user]" in summary.lower() or "[User]" in summary

    def test_rule_compress_short_messages_verbatim(self):
        messages = [{"role": "user", "content": "Yes"}]
        summary, strategy = self.compressor._compress_rule(messages)
        assert "Yes" in summary
        assert strategy == "rule"

    def test_rule_compress_empty_returns_fallback(self):
        summary, strategy = self.compressor._compress_rule([])
        assert strategy == "rule"
        assert isinstance(summary, str)

    # --- Score sentence heuristics ---

    def test_score_sentence_code_boost(self):
        code_sentence = "def process_data(x): return x * 2"
        plain_sentence = "The result was computed."
        score_code = self.compressor._score_sentence(code_sentence)
        score_plain = self.compressor._score_sentence(plain_sentence)
        assert score_code > score_plain

    def test_score_sentence_number_boost(self):
        with_number = "The model achieved 95% accuracy on 1000 samples."
        without_number = "The model achieved good results on the dataset."
        assert self.compressor._score_sentence(with_number) >= self.compressor._score_sentence(without_number)

    def test_score_sentence_returns_nonnegative(self):
        assert self.compressor._score_sentence("anything") >= 0.0

    # --- Hybrid strategy ---

    def test_hybrid_uses_rule_when_ollama_unavailable(self):
        messages = make_messages(5)
        with patch("opti_oignon.conversation_compressor.OLLAMA_AVAILABLE", False):
            summary, strategy = self.compressor._compress_hybrid(messages, "", 100)
        assert "rule" in strategy

    def test_hybrid_uses_rule_when_fits_budget(self):
        messages = make_messages(2)
        # Rule summary of 2 messages will easily fit in a large budget
        summary, strategy = self.compressor._compress_hybrid(messages, "", budget_tokens=100000)
        assert "rule" in strategy

    # --- LLM strategy ---

    def test_llm_falls_back_to_rule_when_ollama_unavailable(self):
        messages = make_messages(3)
        with patch("opti_oignon.conversation_compressor.OLLAMA_AVAILABLE", False):
            summary, strategy = self.compressor._compress_llm(messages, "some-model")
        assert "rule" in strategy

    def test_llm_falls_back_when_no_model(self):
        messages = make_messages(3)
        self.compressor._config["llm_summary_model"] = None
        summary, strategy = self.compressor._compress_llm(messages, "")
        assert isinstance(summary, str)

    def test_llm_uses_ollama_when_available(self):
        messages = make_messages(3)
        mock_response = MagicMock()
        mock_response.message.content = "This is the LLM summary of the conversation."

        with patch("opti_oignon.conversation_compressor.OLLAMA_AVAILABLE", True):
            with patch("opti_oignon.conversation_compressor.ollama") as mock_ollama:
                mock_ollama.chat.return_value = mock_response
                self.compressor._config["llm_summary_model"] = "qwen3:32b"
                summary, strategy = self.compressor._compress_llm(messages, "qwen3:32b")

        assert strategy == "llm"
        assert "LLM summary" in summary or "summary" in summary.lower()

    def test_llm_handles_exception_gracefully(self):
        messages = make_messages(3)
        with patch("opti_oignon.conversation_compressor.OLLAMA_AVAILABLE", True):
            with patch("opti_oignon.conversation_compressor.ollama") as mock_ollama:
                mock_ollama.chat.side_effect = RuntimeError("Connection refused")
                self.compressor._config["llm_summary_model"] = "qwen3:32b"
                summary, strategy = self.compressor._compress_llm(messages, "qwen3:32b")

        assert "rule" in strategy  # fallback
        assert isinstance(summary, str)

    # --- Full compress() method ---

    def test_compress_splits_correctly(self):
        messages = make_messages(8)  # 16 messages
        self.compressor._config["recent_messages_keep"] = 4
        self.compressor._config["strategy"] = "rule"
        # Force compression by setting a tiny budget
        result = self.compressor.compress(messages, budget_tokens=1, model="")
        assert result.original_count == 16
        assert result.compressed_count > 0
        assert len(result.recent_messages) <= 4

    def test_compress_preserves_recent_messages(self):
        messages = make_messages(6)  # 12 messages
        self.compressor._config["recent_messages_keep"] = 4
        self.compressor._config["strategy"] = "rule"
        result = self.compressor.compress(messages, budget_tokens=1, model="")
        # The recent messages should be the last N from original
        if result.compressed_count > 0:
            assert result.recent_messages == messages[-4:]

    def test_compress_system_messages_not_in_history(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello " * 50},
            {"role": "assistant", "content": "Hi " * 50},
        ] * 5
        self.compressor._config["strategy"] = "rule"
        result = self.compressor.compress(messages, budget_tokens=1, model="")
        # System messages should not appear in recent_messages
        assert all(m.get("role") != "system" for m in result.recent_messages)

    def test_compress_auto_uses_config_strategy(self):
        messages = make_messages(10)
        self.compressor._config["strategy"] = "rule"
        result = self.compressor.compress(messages, budget_tokens=1, model="", strategy="auto")
        assert result.strategy_used in ("rule", "none")

    def test_compress_tokens_saved_nonnegative(self):
        messages = make_messages(8)
        result = self.compressor.compress(messages, budget_tokens=1, model="")
        assert result.tokens_saved >= 0

    def test_compress_compression_ratio_range(self):
        messages = make_messages(8)
        result = self.compressor.compress(messages, budget_tokens=1, model="")
        assert 0.0 <= result.compression_ratio <= 1.0

    # --- retrieve_from_archive ---

    def test_retrieve_from_archive_delegates_to_retriever(self):
        mock_results = [
            ArchiveSearchResult(0, "user", "snippet", 0.8)
        ]
        with patch.object(self.compressor._retriever, "retrieve", return_value=mock_results) as mock_r:
            results = self.compressor.retrieve_from_archive("conv-123", "query")
        mock_r.assert_called_once()
        assert results == mock_results

    def test_retrieve_from_archive_uses_config_top_k(self):
        self.compressor._config["archive_retrieval_top_k"] = 5
        with patch.object(self.compressor._retriever, "retrieve", return_value=[]) as mock_r:
            self.compressor.retrieve_from_archive("conv-123", "query")
        call_kwargs = mock_r.call_args
        assert call_kwargs[1].get("top_k") == 5 or (call_kwargs[0] and call_kwargs[0][2] == 5)

    # --- Format messages helper ---

    def test_format_messages_truncates_long_content(self):
        messages = [{"role": "user", "content": "X" * 2000}]
        result = self.compressor._format_messages_for_summary(messages)
        assert len(result) < 2000 + 50  # Should be truncated

    def test_format_messages_empty(self):
        result = self.compressor._format_messages_for_summary([])
        assert result == ""


# ============================================================================
# Module-level singleton test
# ============================================================================

class TestModuleSingleton:
    def test_singleton_is_compressor_instance(self):
        assert isinstance(conversation_compressor, ConversationCompressor)

    def test_singleton_has_default_config(self):
        cfg = conversation_compressor.get_config()
        assert "strategy" in cfg
        assert "enabled" in cfg
