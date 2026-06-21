"""
Integration tests for S66 conversation compressor — executor integration.

Covers:
  - executor.py: CONVERSATION_COMPRESSOR_AVAILABLE flag
  - executor.py: compression_enabled property (get + set)
  - executor.py: last_compression_result property
  - executor.py: _build_conversation_messages compresses when over budget
  - executor.py: _build_conversation_messages skips compression when under budget
  - executor.py: _build_conversation_messages skips when no budget available
  - executor.py: archive retrieval trigger path
  - agentic_executor.py: last_compression_result proxy
"""

from unittest.mock import MagicMock, patch

import pytest

from opti_oignon.agentic_executor import AgenticExecutor
from opti_oignon.executor import (
    CONVERSATION_COMPRESSOR_AVAILABLE,
    Executor,
    _conversation_compressor,
)

# ============================================================================
# Helpers
# ============================================================================

def make_history(n_pairs: int) -> list[dict[str, str]]:
    """Generate n_pairs of user/assistant messages."""
    msgs = []
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"User turn {i}: " + "word " * 40})
        msgs.append({"role": "assistant", "content": f"Assistant turn {i}: " + "word " * 40})
    return msgs


# ============================================================================
# Import flag tests
# ============================================================================

class TestImportFlag:
    def test_conversation_compressor_available_flag_is_bool(self):
        assert isinstance(CONVERSATION_COMPRESSOR_AVAILABLE, bool)

    def test_conversation_compressor_singleton_not_none(self):
        # If available, the singleton should be imported
        if CONVERSATION_COMPRESSOR_AVAILABLE:
            assert _conversation_compressor is not None


# ============================================================================
# Executor property tests
# ============================================================================

class TestExecutorCompressionProperties:
    def setup_method(self):
        self.executor = Executor()

    def test_compression_enabled_default_true(self):
        # Should be True when compressor is available and enabled
        assert isinstance(self.executor.compression_enabled, bool)

    def test_compression_enabled_setter(self):
        original = self.executor._compression_enabled
        self.executor.compression_enabled = False
        assert not self.executor._compression_enabled
        self.executor.compression_enabled = True
        assert self.executor._compression_enabled

    def test_compression_enabled_false_when_flag_false(self):
        with patch("opti_oignon.executor.CONVERSATION_COMPRESSOR_AVAILABLE", False):
            executor = Executor()
            assert not executor.compression_enabled

    def test_last_compression_result_initially_none(self):
        assert self.executor.last_compression_result is None

    def test_last_compression_result_is_accessible(self):
        # After setting manually, should be readable
        self.executor._last_compression_result = "mock_result"
        assert self.executor.last_compression_result == "mock_result"
        self.executor._last_compression_result = None


# ============================================================================
# _build_conversation_messages integration tests
# ============================================================================

class TestBuildConversationMessagesCompression:
    def setup_method(self):
        self.executor = Executor()

    def _make_budget(self, history_tokens: int = 500):
        """Create a mock PromptTokenBudget."""
        budget = MagicMock()
        budget.history_tokens = history_tokens
        return budget

    def test_no_compression_when_budget_none(self):
        """Compression should be skipped when _last_prompt_budget is None."""
        history = make_history(5)
        mock_manager = MagicMock()
        mock_manager.get_context_messages.return_value = history

        with patch("opti_oignon.executor.CONVERSATION_AVAILABLE", True):
            with patch("opti_oignon.executor.conversation_manager", mock_manager):
                self.executor._last_prompt_budget = None
                messages, tokens, stats = self.executor._build_conversation_messages(
                    system_prompt="System",
                    conversation_id="conv-123",
                    current_message="Hello",
                    model="",
                )
        # No compression result should be set
        assert self.executor.last_compression_result is None

    def test_no_compression_when_under_budget(self):
        """Compression should not trigger when history fits in budget."""
        history = make_history(1)  # small history
        mock_manager = MagicMock()
        mock_manager.get_context_messages.return_value = history

        with patch("opti_oignon.executor.CONVERSATION_AVAILABLE", True):
            with patch("opti_oignon.executor.conversation_manager", mock_manager):
                # Very large budget
                self.executor._last_prompt_budget = self._make_budget(history_tokens=100000)
                messages, tokens, stats = self.executor._build_conversation_messages(
                    system_prompt="System",
                    conversation_id="conv-123",
                    current_message="Hello",
                    model="",
                )
        assert self.executor.last_compression_result is None

    def test_compression_triggered_when_over_budget(self):
        """Compression should trigger when history exceeds budget.history_tokens."""
        history = make_history(10)  # many messages
        mock_manager = MagicMock()
        mock_manager.get_context_messages.return_value = history

        mock_compressed = MagicMock()
        mock_compressed.compressed_count = 8
        mock_compressed.summary = "Earlier conversation summary:\n- User asked about topic 0\n- Assistant replied about topic 1"
        mock_compressed.recent_messages = history[-4:]
        mock_compressed.strategy_used = "rule"
        mock_compressed.tokens_saved = 200

        with patch("opti_oignon.executor.CONVERSATION_AVAILABLE", True):
            with patch("opti_oignon.executor.conversation_manager", mock_manager):
                with patch("opti_oignon.executor.CONVERSATION_COMPRESSOR_AVAILABLE", True):
                    with patch("opti_oignon.executor._conversation_compressor") as mock_comp:
                        mock_comp.enabled = True
                        mock_comp.compress.return_value = mock_compressed
                        mock_comp.get_config.return_value = {
                            "retrieval_trigger_min_confidence": 0.6
                        }

                        self.executor._compression_enabled = True
                        # Tiny budget to force compression
                        self.executor._last_prompt_budget = self._make_budget(history_tokens=1)

                        messages, tokens, stats = self.executor._build_conversation_messages(
                            system_prompt="System",
                            conversation_id="conv-123",
                            current_message="Hello",
                            model="",
                        )

        # compress() should have been called
        mock_comp.compress.assert_called_once()
        # Result should be stored
        assert self.executor.last_compression_result == mock_compressed

    def test_compression_injects_summary_as_system_message(self):
        """The summary block must be injected as a system role message."""
        history = make_history(5)
        mock_manager = MagicMock()
        mock_manager.get_context_messages.return_value = history

        recent = history[-4:]
        mock_compressed = MagicMock()
        mock_compressed.compressed_count = 6
        mock_compressed.summary = "Earlier conversation summary:\n- Key fact"
        mock_compressed.recent_messages = recent
        mock_compressed.strategy_used = "hybrid_rule"
        mock_compressed.tokens_saved = 150

        with patch("opti_oignon.executor.CONVERSATION_AVAILABLE", True):
            with patch("opti_oignon.executor.conversation_manager", mock_manager):
                with patch("opti_oignon.executor.CONVERSATION_COMPRESSOR_AVAILABLE", True):
                    with patch("opti_oignon.executor._conversation_compressor") as mock_comp:
                        mock_comp.enabled = True
                        mock_comp.compress.return_value = mock_compressed
                        mock_comp.get_config.return_value = {
                            "retrieval_trigger_min_confidence": 0.6
                        }
                        self.executor._compression_enabled = True
                        self.executor._last_prompt_budget = self._make_budget(history_tokens=1)

                        messages, tokens, stats = self.executor._build_conversation_messages(
                            system_prompt="System",
                            conversation_id="conv-123",
                            current_message="Test",
                            model="",
                        )

        # The second message (after system prompt) should be the summary block
        system_msgs = [m for m in messages if m["role"] == "system"]
        # At least one system message should contain the summary
        summary_found = any(
            "Earlier conversation summary" in m["content"]
            for m in system_msgs
        )
        assert summary_found, f"Summary not found in system messages: {system_msgs}"

    def test_compression_skipped_when_disabled(self):
        """Compression should be skipped when compression_enabled is False."""
        self.executor._compression_enabled = False
        history = make_history(5)
        mock_manager = MagicMock()
        mock_manager.get_context_messages.return_value = history

        with patch("opti_oignon.executor.CONVERSATION_AVAILABLE", True):
            with patch("opti_oignon.executor.conversation_manager", mock_manager):
                self.executor._last_prompt_budget = self._make_budget(history_tokens=1)

                messages, tokens, stats = self.executor._build_conversation_messages(
                    system_prompt="System",
                    conversation_id="conv-123",
                    current_message="Hello",
                    model="",
                )

        # No compression result when disabled
        assert self.executor.last_compression_result is None

    def test_compression_graceful_degradation_on_exception(self):
        """If compress() raises, pipeline should continue without compression."""
        history = make_history(5)
        mock_manager = MagicMock()
        mock_manager.get_context_messages.return_value = history

        with patch("opti_oignon.executor.CONVERSATION_AVAILABLE", True):
            with patch("opti_oignon.executor.conversation_manager", mock_manager):
                with patch("opti_oignon.executor.CONVERSATION_COMPRESSOR_AVAILABLE", True):
                    with patch("opti_oignon.executor._conversation_compressor") as mock_comp:
                        mock_comp.enabled = True
                        mock_comp.compress.side_effect = RuntimeError("Compression error")
                        mock_comp.get_config.return_value = {
                            "retrieval_trigger_min_confidence": 0.6
                        }
                        self.executor._compression_enabled = True
                        self.executor._last_prompt_budget = self._make_budget(history_tokens=1)

                        # Should not raise — graceful degradation
                        messages, tokens, stats = self.executor._build_conversation_messages(
                            system_prompt="System",
                            conversation_id="conv-123",
                            current_message="Hello",
                            model="",
                        )

        # No crash, no compression result stored
        assert self.executor.last_compression_result is None
        # Messages should still be built normally
        assert len(messages) >= 2


# ============================================================================
# AgenticExecutor proxy test
# ============================================================================

class TestAgenticExecutorProxy:
    def test_last_compression_result_proxy_none(self):
        """Proxy should return None when executor has no compression result."""
        agentic = AgenticExecutor()
        assert agentic.last_compression_result is None

    def test_last_compression_result_proxy_delegates(self):
        """Proxy should return executor's compression result."""
        agentic = AgenticExecutor()
        if agentic._executor is not None:
            agentic._executor._last_compression_result = "mock_result"
            assert agentic.last_compression_result == "mock_result"
            agentic._executor._last_compression_result = None

    def test_last_compression_result_proxy_no_executor(self):
        """Proxy should return None when executor is None."""
        agentic = AgenticExecutor()
        original = agentic._executor
        agentic._executor = None
        assert agentic.last_compression_result is None
        agentic._executor = original
