#!/usr/bin/env python3
"""
CHAT CODING AGENT - OPTI-OIGNON v2.2.0 (S118)
===============================================

Conversational coding agent that lives in the chat, combining the
multi-step intelligence of the Coding Agent (S74-S81) with the natural
conversation flow of regular chat.

Unlike the standalone CodingAgent panel (one-shot task, fixed pipeline),
the ChatCodingAgent supports:
  - Multi-turn conversation with persistent sandbox across turns
  - Adaptive pipeline (plan -> implement -> test -> fix, any phase skippable)
  - Natural language directives ("skip tests", "just write it", "try harder")
  - Smart follow-up detection (minimal diff plans for modifications)
  - Full conversation memory: S66 compression, archive retrieval,
    context window management, working memory state injection

Architecture:
  - ChatCodingSession: one per conversation_id, wraps a persistent sandbox
    and maintains context state (files, test results, previous instructions)
  - ChatCodingManager: pool of active sessions, cleanup, config

Security:
  - Same bwrap isolation as the Coding Agent sandbox
  - No auto-approve: user must explicitly approve & download files
  - Sessions auto-expire after configurable timeout

Author: Leon
"""

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from opti_oignon.sandbox_manager import (
        SANDBOX_AVAILABLE,
        SandboxManager,
        SandboxSession,
    )
    from opti_oignon.sandbox_manager import (
        sandbox_manager as _default_sandbox_manager,
    )
except ImportError:
    SANDBOX_AVAILABLE = False
    _default_sandbox_manager = None
    SandboxManager = None
    SandboxSession = None

try:
    from opti_oignon.file_tools import (
        FILE_TOOLS_AVAILABLE,
        _handle_sandbox_bash,
        _handle_sandbox_create_file,
        _handle_sandbox_view,
    )
except ImportError:
    FILE_TOOLS_AVAILABLE = False
    _handle_sandbox_bash = None
    _handle_sandbox_view = None
    _handle_sandbox_create_file = None

try:
    from opti_oignon.conversation import (
        conversation_manager as _conversation_manager,
    )
    CONVERSATION_AVAILABLE = True
except ImportError:
    CONVERSATION_AVAILABLE = False
    _conversation_manager = None

try:
    from opti_oignon.conversation_compressor import (
        CompressedContext,
        ConversationCompressor,  # noqa: F401
        check_retrieval_trigger,
    )
    from opti_oignon.conversation_compressor import (
        conversation_compressor as _conversation_compressor,
    )
    COMPRESSOR_AVAILABLE = True
except ImportError:
    COMPRESSOR_AVAILABLE = False
    _conversation_compressor = None
    CompressedContext = None
    check_retrieval_trigger = None

try:
    from opti_oignon.context_manager import (
        estimate_tokens as _estimate_tokens_cm,
    )
    from opti_oignon.context_manager import (
        get_model_limits as _get_model_limits,
    )
    CONTEXT_MANAGER_AVAILABLE = True
except ImportError:
    CONTEXT_MANAGER_AVAILABLE = False
    _get_model_limits = None
    _estimate_tokens_cm = None

try:
    from opti_oignon.prompt_optimization import (
        prompt_budget_manager as _prompt_budget_manager,
    )
    from opti_oignon.prompt_optimization import (
        prompt_template_engine as _prompt_template_engine,
    )
    PROMPT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PROMPT_OPTIMIZATION_AVAILABLE = False
    _prompt_budget_manager = None
    _prompt_template_engine = None

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

CHAT_CODING_AVAILABLE = SANDBOX_AVAILABLE and FILE_TOOLS_AVAILABLE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChatCodingConfig:
    """Configuration for the chat coding agent."""
    enabled: bool = False
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 3
    max_fix_retries: int = 3
    max_plan_retries: int = 2
    auto_test: bool = True
    auto_test_command: str = "python3 -m pytest -x --tb=short"
    default_model: str | None = None
    planning_model: str | None = None
    command_timeout: int = 30


def _load_config() -> ChatCodingConfig:
    """Load chat coding config from coding_agent.yaml."""
    cfg = ChatCodingConfig()
    if not YAML_AVAILABLE:
        return cfg
    try:
        import os
        config_path = os.path.join(
            os.path.dirname(__file__), "config", "coding_agent.yaml"
        )
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            cc = data.get("chat_coding", {})
            if isinstance(cc, dict):
                cfg.enabled = bool(cc.get("enabled", False))
                cfg.session_timeout_minutes = int(
                    cc.get("session_timeout_minutes", 60)
                )
                cfg.max_concurrent_sessions = int(
                    cc.get("max_concurrent_sessions", 3)
                )
                cfg.max_fix_retries = int(
                    cc.get("max_fix_retries",
                           data.get("max_fix_retries", 3))
                )
                cfg.max_plan_retries = int(
                    cc.get("max_plan_retries",
                           data.get("max_plan_retries", 2))
                )
                cfg.auto_test = bool(
                    cc.get("auto_test", data.get("auto_test", True))
                )
                cfg.auto_test_command = str(
                    cc.get("auto_test_command",
                           data.get("auto_test_command",
                                    "python3 -m pytest -x --tb=short"))
                )
                cfg.default_model = cc.get(
                    "default_model", data.get("default_model")
                )
                cfg.planning_model = cc.get(
                    "planning_model", data.get("planning_model")
                )
                cfg.command_timeout = int(
                    cc.get("command_timeout", 30)
                )
    except Exception as exc:
        logger.warning("Failed to load chat_coding config: %s", exc)
    return cfg


# ---------------------------------------------------------------------------
# Streaming event types
# ---------------------------------------------------------------------------

@dataclass
class CodingEvent:
    """Streaming event emitted by the chat coding agent pipeline."""
    event_type: str  # coding_plan, coding_step, coding_test, coding_fix,
                     # coding_done, coding_error, coding_status, coding_token
    data: dict[str, Any] = field(default_factory=dict)
    content: str = ""


# ---------------------------------------------------------------------------
# Rich LLM call context — same capabilities as regular chat pipeline
# ---------------------------------------------------------------------------

@dataclass
class LLMCallContext:
    """Context passed to each LLM call within the coding agent.

    Mirrors the capabilities of the regular chat pipeline so the coding
    agent has access to all the same tools and features:
    - Vision delegation (S95): images analyzed by vision-capable model
    - Web search (S42): search and inject results
    - Think mode (S42): chain-of-thought reasoning
    - Tool calls (S44/S45): all registered tools (file, CSV, XLSX, etc.)
    - Plugin hooks (S114): pre/post inference hooks
    """
    images: list[str] | None = None
    web_search: bool = False
    think: bool = False
    tools_enabled: bool = True
    conversation_id: str | None = None


@dataclass
class LLMCallResult:
    """Result from a rich LLM call.

    Contains the text response plus metadata from the full pipeline
    (tool calls made, vision delegation info, plugin annotations).
    """
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    vision_meta: dict[str, Any] = field(default_factory=dict)
    plugin_annotations: list[dict[str, Any]] = field(default_factory=list)
    thinking: str = ""
    error: str = ""


# Type alias for the rich LLM callback.
# Signature: (messages, model, context) -> LLMCallResult
# The callback wraps the existing agentic/executor pipeline so the
# coding agent gets vision, tools, plugins, web search automatically.
RichLLMCall = Callable[
    [list[dict[str, str]], str, LLMCallContext],
    LLMCallResult,
]

# Legacy simple callback for backward compatibility.
# Signature: (prompt, system, model) -> str
SimpleLLMCall = Callable[[str, str, str], str]


# ---------------------------------------------------------------------------
# Directive parser
# ---------------------------------------------------------------------------

@dataclass
class TurnDirectives:
    """Runtime options parsed from the user's message."""
    skip_test: bool = False
    skip_plan: bool = False
    skip_fix: bool = False
    plan_only: bool = False
    max_fix_retries: int | None = None
    raw_message: str = ""

    @property
    def cleaned_message(self) -> str:
        """Message with directive flags removed."""
        return self.raw_message


_DIRECTIVE_PATTERNS = [
    # --flag style
    (re.compile(r"--no-?test\b", re.IGNORECASE), "skip_test"),
    (re.compile(r"--no-?fix\b", re.IGNORECASE), "skip_fix"),
    (re.compile(r"--no-?plan\b", re.IGNORECASE), "skip_plan"),
    (re.compile(r"--skip-?plan\b", re.IGNORECASE), "skip_plan"),
    (re.compile(r"--plan-?only\b", re.IGNORECASE), "plan_only"),
    (re.compile(r"--max-?retries?\s+(\d+)", re.IGNORECASE), "max_fix_retries"),
    # Natural language variants
    (re.compile(
        r"\b(?:don'?t|do not|no need to|skip)\s+test",
        re.IGNORECASE,
    ), "skip_test"),
    (re.compile(
        r"\bjust\s+(?:write|code|implement|do)\s+it\b",
        re.IGNORECASE,
    ), "skip_test"),
    (re.compile(
        r"\bskip\s+(?:the\s+)?(?:test|testing)\b",
        re.IGNORECASE,
    ), "skip_test"),
    (re.compile(
        r"\b(?:don'?t|do not|no)\s+(?:auto[- ]?)?fix\b",
        re.IGNORECASE,
    ), "skip_fix"),
    (re.compile(
        r"\b(?:only|just)\s+plan\b",
        re.IGNORECASE,
    ), "plan_only"),
    (re.compile(
        r"\bplan\s+only\b",
        re.IGNORECASE,
    ), "plan_only"),
    (re.compile(
        r"\bskip\s+plan(?:ning)?\b",
        re.IGNORECASE,
    ), "skip_plan"),
    (re.compile(
        r"\btry\s+harder\b",
        re.IGNORECASE,
    ), "max_fix_retries_5"),
]


def parse_directives(message: str) -> TurnDirectives:
    """Parse runtime directives from a user message.

    Detects both --flags and natural language patterns.
    Returns a TurnDirectives with the cleaned message.
    """
    directives = TurnDirectives(raw_message=message)
    cleaned = message

    for pattern, attr in _DIRECTIVE_PATTERNS:
        m = pattern.search(cleaned)
        if m:
            if attr == "max_fix_retries":
                directives.max_fix_retries = int(m.group(1))
            elif attr == "max_fix_retries_5":
                directives.max_fix_retries = 5
            else:
                setattr(directives, attr, True)
            # Remove the matched directive from the cleaned message
            cleaned = pattern.sub("", cleaned).strip()

    directives.raw_message = cleaned if cleaned else message
    return directives


# ---------------------------------------------------------------------------
# Working memory: sandbox state injected into LLM context each turn
# ---------------------------------------------------------------------------

@dataclass
class SandboxState:
    """Snapshot of the current sandbox state for context injection."""
    files: list[str] = field(default_factory=list)
    last_test_passed: bool | None = None
    last_test_output: str = ""
    last_error: str = ""
    turn_count: int = 0
    cumulative_summary: str = ""

    def as_context_block(self) -> str:
        """Format the sandbox state for injection into the LLM prompt."""
        parts = []
        if self.files:
            parts.append(
                f"Files in sandbox ({len(self.files)}): "
                + ", ".join(self.files[:30])
            )
            if len(self.files) > 30:
                parts.append(f"  ... and {len(self.files) - 30} more")

        if self.last_test_passed is not None:
            status = "PASSED" if self.last_test_passed else "FAILED"
            parts.append(f"Last test result: {status}")
            if self.last_test_output:
                # Truncate test output to avoid flooding context
                trunc = self.last_test_output[:1500]
                if len(self.last_test_output) > 1500:
                    trunc += "\n... (truncated)"
                parts.append(f"Test output:\n{trunc}")

        if self.last_error:
            parts.append(f"Last error: {self.last_error[:500]}")

        if self.cumulative_summary:
            parts.append(
                f"Previous turns summary:\n{self.cumulative_summary}"
            )

        if not parts:
            return ""

        return (
            "[SANDBOX STATE]\n"
            + "\n".join(parts)
            + "\n[/SANDBOX STATE]"
        )


# ---------------------------------------------------------------------------
# ChatCodingSession
# ---------------------------------------------------------------------------

class ChatCodingSession:
    """Persistent coding session attached to a chat conversation.

    Wraps a sandbox that survives across chat turns and maintains
    full conversation context including compression and retrieval.

    Lifecycle:
      1. Created on first /code message or Code Agent toggle
      2. Sandbox persists across turns within the conversation
      3. Destroyed on explicit action, conversation change, or timeout
    """

    def __init__(
        self,
        conversation_id: str,
        sandbox_mgr: "SandboxManager | None" = None,
        config: ChatCodingConfig | None = None,
        llm_call: "RichLLMCall | SimpleLLMCall | None" = None,
    ):
        """Initialize a chat coding session.

        Args:
            conversation_id: The conversation this session is attached to.
            sandbox_mgr: Sandbox manager for isolation.
            config: Chat coding configuration.
            llm_call: Callable for LLM interaction. Supports two signatures:
                Rich (preferred): (messages, model, LLMCallContext) -> LLMCallResult
                    Provides full pipeline: vision, tools, plugins, web search.
                Simple (legacy): (prompt, system, model) -> str
                    Basic text-in/text-out without extra capabilities.
        """
        self._conversation_id = conversation_id
        self._session_id = f"cc-{conversation_id[:12]}-{uuid.uuid4().hex[:8]}"
        self._mgr = sandbox_mgr or _default_sandbox_manager
        self._config = config or _load_config()
        self._llm_call = llm_call
        self._is_rich_llm = False  # detected on first call

        # Sandbox state
        self._sandbox_session: SandboxSession | None = None
        self._sandbox_state = SandboxState()

        # Timing
        self._created_at = time.time()
        self._last_activity = time.time()
        self._timeout_seconds = self._config.session_timeout_minutes * 60

        # Thread safety
        self._lock = threading.Lock()

        # Conversation context (full multi-turn memory)
        self._turn_history: list[dict[str, str]] = []
        self._last_compression_result = None

        # Per-turn feature state (set by execute_task)
        self._turn_images: list[str] | None = None
        self._turn_web_search: bool = False
        self._turn_think: bool = False

        # Last call metadata (tool calls, vision, plugins from last LLM call)
        self._last_tool_calls: list[dict[str, Any]] = []
        self._last_vision_meta: dict[str, Any] = {}
        self._last_plugin_annotations: list[dict[str, Any]] = []

    # -- Properties -----------------------------------------------------------

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def active(self) -> bool:
        return (
            self._sandbox_session is not None
            and self._sandbox_session.active
        )

    @property
    def expired(self) -> bool:
        return (
            time.time() - self._last_activity > self._timeout_seconds
        )

    @property
    def created_at(self) -> float:
        return self._created_at

    @property
    def sandbox_state(self) -> SandboxState:
        return self._sandbox_state

    @property
    def turn_count(self) -> int:
        return self._sandbox_state.turn_count

    # -- Sandbox management ---------------------------------------------------

    def _ensure_sandbox(self) -> None:
        """Lazily create the sandbox on first use."""
        if self._sandbox_session is not None and self._sandbox_session.active:
            self._last_activity = time.time()
            return
        if self._mgr is None:
            raise RuntimeError("Sandbox manager not available")
        self._sandbox_session = self._mgr.create_sandbox(
            self._session_id, allow_degraded=True
        )
        logger.info(
            "Chat coding sandbox created: %s (conv=%s)",
            self._session_id, self._conversation_id[:8],
        )

    def _exec_in_sandbox(
        self,
        cmd: str,
        timeout: int | None = None,
    ) -> str:
        """Execute a shell command inside the sandbox."""
        self._ensure_sandbox()
        self._last_activity = time.time()
        try:
            return _handle_sandbox_bash(
                self._session_id, cmd,
                timeout or self._config.command_timeout,
                _sandbox_manager=self._mgr,
            )
        except Exception as exc:
            return f"Sandbox execution error: {exc}"

    def _write_file(self, path: str, content: str) -> str:
        """Write a file inside the sandbox."""
        self._ensure_sandbox()
        self._last_activity = time.time()
        try:
            result = _handle_sandbox_create_file(
                self._session_id, path, content,
                _sandbox_manager=self._mgr,
            )
            with self._lock:
                if path not in self._sandbox_state.files:
                    self._sandbox_state.files.append(path)
            return result
        except Exception as exc:
            return f"Sandbox write error: {exc}"

    def _read_file(self, path: str) -> str:
        """Read a file from inside the sandbox."""
        self._ensure_sandbox()
        self._last_activity = time.time()
        try:
            return _handle_sandbox_view(
                self._session_id, path, 0, 0,
                _sandbox_manager=self._mgr,
            )
        except Exception as exc:
            return f"Sandbox read error: {exc}"

    def _list_sandbox_files(self) -> list[str]:
        """List files currently in the sandbox."""
        if not self.active:
            return list(self._sandbox_state.files)
        try:
            raw = _handle_sandbox_view(
                self._session_id, ".", 0, 0,
                _sandbox_manager=self._mgr,
            )
            # Parse simple listing
            files = [
                line.strip() for line in raw.splitlines()
                if line.strip() and not line.startswith("total ")
            ]
            self._sandbox_state.files = files
            return files
        except Exception:
            return list(self._sandbox_state.files)

    def get_sandbox_files_for_ui(self) -> list[dict[str, Any]]:
        """Get sandbox files for the frontend SandboxFileManager."""
        if not self.active or self._mgr is None:
            return []
        try:
            return self._mgr.extract_files(self._session_id)
        except Exception:
            return []

    # -- Context management (conversation memory) -----------------------------

    def _estimate_tokens(self, text: str, model: str = "") -> int:
        """Estimate token count for a text string."""
        if CONTEXT_MANAGER_AVAILABLE and _estimate_tokens_cm is not None:
            return _estimate_tokens_cm(text, model)
        # Rough fallback: ~0.75 tokens per character
        return max(1, int(len(text) * 0.75))

    def _get_model_context_budget(self, model: str) -> int:
        """Get the available history token budget for the given model."""
        if (
            PROMPT_OPTIMIZATION_AVAILABLE
            and _prompt_budget_manager is not None
        ):
            try:
                budget = _prompt_budget_manager.calculate_budget(model=model)
                return budget.history_tokens
            except Exception:
                pass

        # Fallback: use 60% of model context window for history
        if CONTEXT_MANAGER_AVAILABLE and _get_model_limits is not None:
            try:
                limits = _get_model_limits(model)
                return int(limits.context_window * 0.6)
            except Exception:
                pass

        # Conservative default
        return 8192

    def _build_conversation_messages(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
    ) -> list[dict[str, str]]:
        """Build the full messages array with conversation history.

        Loads conversation history from the conversation backend,
        applies S66 compression if history exceeds budget, and injects
        the sandbox state as context. Also supports archive retrieval
        for follow-up questions that reference older context.

        This mirrors the same logic as executor._build_conversation_messages
        to provide equally robust conversation memory.

        Args:
            system_prompt: The system prompt for this coding turn.
            user_message: The current user message.
            model: Model name for token estimation.

        Returns:
            Ollama-format messages list.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # -- Load conversation history from backend --------------------------
        history: list[dict[str, str]] = []
        if (
            CONVERSATION_AVAILABLE
            and _conversation_manager is not None
            and self._conversation_id
        ):
            try:
                history = _conversation_manager.get_context_messages(
                    self._conversation_id
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load conversation history: %s", exc
                )

        # -- S66: Compress history if it exceeds token budget ----------------
        history_budget = self._get_model_context_budget(model)
        self._last_compression_result = None

        if (
            COMPRESSOR_AVAILABLE
            and _conversation_compressor is not None
            and _conversation_compressor.enabled
            and history
        ):
            history_tokens = sum(
                self._estimate_tokens(m.get("content", ""), model)
                for m in history
                if m.get("role") != "system"
            )
            if history_tokens > history_budget:
                try:
                    compressed = _conversation_compressor.compress(
                        messages=history,
                        budget_tokens=history_budget,
                        model=model,
                    )
                    if compressed.compressed_count > 0 and compressed.summary:
                        # Rebuild history: summary + recent messages
                        summary_block = {
                            "role": "system",
                            "content": (
                                "[CONVERSATION SUMMARY]\n"
                                + compressed.summary
                                + "\n[/CONVERSATION SUMMARY]"
                            ),
                        }
                        history = (
                            [summary_block]
                            + list(compressed.recent_messages)
                        )
                        self._last_compression_result = compressed
                        logger.info(
                            "S66 chat coding compression: %d msgs -> "
                            "%d compressed + %d kept, %dt saved",
                            compressed.original_count,
                            compressed.compressed_count,
                            len(compressed.recent_messages),
                            compressed.tokens_saved,
                        )
                except Exception as exc:
                    logger.warning(
                        "S66 compression failed in chat coding: %s", exc
                    )

        # -- Archive retrieval: search old context on follow-ups -------------
        if (
            COMPRESSOR_AVAILABLE
            and _conversation_compressor is not None
            and check_retrieval_trigger is not None
        ):
            if check_retrieval_trigger(user_message, min_confidence=0.5):
                try:
                    # All messages in conversation backend are searchable
                    all_history = []
                    if CONVERSATION_AVAILABLE and _conversation_manager:
                        all_history = (
                            _conversation_manager.get_context_messages(
                                self._conversation_id
                            )
                        )
                    if all_history:
                        results = _conversation_compressor.retrieve_from_archive(
                            all_history, user_message, max_results=3
                        )
                        if results:
                            retrieval_block = "[RETRIEVED FROM ARCHIVE]\n"
                            for r in results:
                                retrieval_block += (
                                    f"- [{r.role}] {r.snippet}\n"
                                )
                            retrieval_block += "[/RETRIEVED FROM ARCHIVE]"
                            history.append({
                                "role": "system",
                                "content": retrieval_block,
                            })
                            logger.info(
                                "Archive retrieval: %d results for coding turn",
                                len(results),
                            )
                except Exception as exc:
                    logger.debug("Archive retrieval failed: %s", exc)

        # -- Inject sandbox state as context ---------------------------------
        sandbox_context = self._sandbox_state.as_context_block()
        if sandbox_context:
            history.append({
                "role": "system",
                "content": sandbox_context,
            })

        # -- Assemble final messages list ------------------------------------
        # Filter out system messages from history (they are context blocks)
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def _save_turn_to_conversation(
        self,
        user_message: str,
        assistant_response: str,
        model: str,
    ) -> None:
        """Save the coding turn to the conversation backend.

        Both user and assistant messages are persisted so they appear
        in the full conversation history and are available for
        compression and retrieval in future turns.
        """
        if not CONVERSATION_AVAILABLE or _conversation_manager is None:
            return
        try:
            _conversation_manager.add_message(
                self._conversation_id, "user", user_message
            )
            _conversation_manager.add_message(
                self._conversation_id, "assistant", assistant_response,
                model=model,
            )
        except Exception as exc:
            logger.warning(
                "Failed to save coding turn to conversation: %s", exc
            )

    def _update_cumulative_summary(
        self,
        user_message: str,
        result_summary: str,
    ) -> None:
        """Append a brief summary of this turn to the cumulative state.

        This is a lightweight working memory that helps the LLM understand
        what has been done across turns without relying solely on the
        full conversation history (which may be compressed).
        """
        turn_num = self._sandbox_state.turn_count
        entry = (
            f"Turn {turn_num}: User asked: "
            f"{user_message[:200]}{'...' if len(user_message) > 200 else ''}"
            f" | Result: {result_summary[:200]}"
        )

        existing = self._sandbox_state.cumulative_summary
        if existing:
            # Keep last 5 turn summaries to avoid unbounded growth
            lines = existing.strip().split("\n")
            if len(lines) >= 5:
                lines = lines[-4:]
            lines.append(entry)
            self._sandbox_state.cumulative_summary = "\n".join(lines)
        else:
            self._sandbox_state.cumulative_summary = entry

    # -- LLM interaction ------------------------------------------------------

    def _detect_rich_callback(self) -> bool:
        """Detect whether the llm_call is a rich callback or simple.

        Rich: (messages, model, LLMCallContext) -> LLMCallResult
        Simple: (prompt, system, model) -> str
        """
        if self._llm_call is None:
            return False
        # Check via type hints or duck-typing on first call
        import inspect
        try:
            sig = inspect.signature(self._llm_call)
            params = list(sig.parameters.keys())
            # Rich callback has 3 params: messages, model, context
            # Simple callback has 3 params: prompt, system, model
            # Disambiguate by param names
            if "context" in params or "llm_context" in params:
                return True
            if "messages" in params:
                return True
        except (ValueError, TypeError):
            pass
        return False

    def _call_llm(
        self,
        prompt: str,
        system: str,
        model: str,
        phase: str = "general",
    ) -> LLMCallResult:
        """Call the LLM with full conversation context and pipeline features.

        Routes through the rich callback if available, which provides:
        - Vision delegation (images analyzed by vision-capable model)
        - Tool calls (file tools routed to sandbox, CSV/XLSX/etc.)
        - Plugin hooks (pre/post inference)
        - Web search (documentation lookup)
        - Think mode (chain-of-thought reasoning)

        Falls back to the simple callback for basic text-in/text-out.

        Args:
            prompt: The prompt for this phase.
            system: The system prompt.
            model: Model to use.
            phase: Pipeline phase name (for logging/events).

        Returns:
            LLMCallResult with text response and metadata.
        """
        if self._llm_call is None:
            return LLMCallResult(
                error="No LLM callable provided to ChatCodingSession"
            )

        # Build messages with full conversation context
        messages = self._build_conversation_messages(
            system_prompt=system,
            user_message=prompt,
            model=model,
        )

        # Detect callback type on first call
        if not hasattr(self, "_callback_detected"):
            self._is_rich_llm = self._detect_rich_callback()
            self._callback_detected = True

        if self._is_rich_llm:
            # Rich callback: pass full messages + context with all features
            ctx = LLMCallContext(
                images=self._turn_images if phase == "implement" else None,
                web_search=self._turn_web_search,
                think=self._turn_think,
                tools_enabled=True,
                conversation_id=self._conversation_id,
            )
            try:
                result = self._llm_call(messages, model, ctx)
                # Capture metadata from the pipeline
                if hasattr(result, "tool_calls"):
                    self._last_tool_calls = result.tool_calls or []
                if hasattr(result, "vision_meta"):
                    self._last_vision_meta = result.vision_meta or {}
                if hasattr(result, "plugin_annotations"):
                    self._last_plugin_annotations = (
                        result.plugin_annotations or []
                    )
                return result
            except Exception as exc:
                logger.warning(
                    "Rich LLM call failed in %s phase: %s", phase, exc
                )
                return LLMCallResult(error=str(exc))

        else:
            # Simple callback: extract context into augmented system prompt
            augmented_system = system
            system_extras = [
                msg["content"] for msg in messages
                if msg["role"] == "system" and msg["content"] != system
            ]
            if system_extras:
                augmented_system += "\n\n" + "\n\n".join(system_extras)

            try:
                text = self._llm_call(prompt, augmented_system, model)
                return LLMCallResult(text=text)
            except Exception as exc:
                logger.warning(
                    "Simple LLM call failed in %s phase: %s", phase, exc
                )
                return LLMCallResult(error=str(exc))

    # -- Pipeline phases ------------------------------------------------------

    def _phase_plan(
        self,
        user_message: str,
        model: str,
        is_followup: bool,
    ) -> Generator[CodingEvent, None, str]:
        """Planning phase: generate a coding plan from the user request.

        For follow-up turns (is_followup=True), generates a minimal
        diff plan rather than a full from-scratch plan.

        Yields CodingEvents and returns the plan text.
        """
        yield CodingEvent("coding_status", content="Planning...")

        if is_followup and self._sandbox_state.files:
            plan_prompt = (
                f"The user wants to modify the existing code in the sandbox.\n"
                f"Current files: {', '.join(self._sandbox_state.files[:20])}\n"
                f"User request: {user_message}\n\n"
                f"Generate a MINIMAL modification plan. Only list the files "
                f"that need to change and what changes are needed. "
                f"Do NOT recreate files that don't need changes.\n"
                f"Format: numbered list of steps."
            )
        else:
            plan_prompt = (
                f"User request: {user_message}\n\n"
                f"Generate a step-by-step coding plan to implement this.\n"
                f"List each file to create/modify with a brief description.\n"
                f"Format: numbered list of steps."
            )

        system = (
            "You are a coding agent inside an isolated sandbox. "
            "Generate a clear, concise plan. "
            "Each step should be actionable: create a file, modify a file, "
            "or run a command. Keep the plan focused and minimal."
        )

        plan_model = self._config.planning_model or model
        result = self._call_llm(plan_prompt, system, plan_model, phase="plan")
        plan_text = result.text or result.error or ""

        # Emit tool calls from planning phase (e.g. web search for docs)
        if result.tool_calls:
            yield CodingEvent(
                "coding_status",
                data={"tool_calls": result.tool_calls},
                content=f"Planning used {len(result.tool_calls)} tool(s)",
            )

        # Parse plan steps
        steps = []
        for line in plan_text.strip().split("\n"):
            line = line.strip()
            if line and re.match(r"^\d+[\.\)]\s+", line):
                step = re.sub(r"^\d+[\.\)]\s+", "", line)
                steps.append(step)

        if not steps:
            # Fallback: treat the whole plan as a single step
            steps = [plan_text.strip()[:500]]

        yield CodingEvent(
            "coding_plan",
            data={
                "steps": steps,
                "step_count": len(steps),
                "is_followup": is_followup,
            },
            content=plan_text,
        )

        return plan_text

    def _phase_implement(
        self,
        user_message: str,
        plan_text: str,
        model: str,
    ) -> Generator[CodingEvent, None, str]:
        """Implementation phase: write code based on the plan.

        Asks the LLM to generate code, then writes it to the sandbox.
        Yields CodingEvents and returns a summary of what was implemented.

        Returns the full implementation response text.
        """
        yield CodingEvent("coding_status", content="Implementing...")

        impl_prompt = (
            f"Based on this plan:\n{plan_text}\n\n"
            f"User request: {user_message}\n\n"
            f"Now implement the code. For each file, output the content "
            f"in this format:\n"
            f"--- FILE: <filepath> ---\n"
            f"<file content>\n"
            f"--- END FILE ---\n\n"
            f"Write complete, working code. Include all imports and "
            f"necessary setup."
        )

        system = (
            "You are a coding agent inside an isolated sandbox. "
            "Write complete, working code files. "
            "Use the exact format specified for file output. "
            "Be thorough but concise. "
            "You have access to all tools: file manipulation, web search "
            "for documentation, image analysis (if images provided), "
            "and any active plugins."
        )

        impl_result = self._call_llm(
            impl_prompt, system, model, phase="implement"
        )
        response = impl_result.text or ""

        # Emit vision and tool call info from implementation
        if impl_result.vision_meta:
            yield CodingEvent(
                "coding_status",
                data={"vision_meta": impl_result.vision_meta},
                content="Image analyzed for implementation context",
            )
        if impl_result.tool_calls:
            yield CodingEvent(
                "coding_status",
                data={"tool_calls": impl_result.tool_calls},
                content=f"Used {len(impl_result.tool_calls)} tool(s) during implementation",
            )

        # Parse file blocks from the response
        files_written = []
        file_pattern = re.compile(
            r"---\s*FILE:\s*(.+?)\s*---\s*\n(.*?)---\s*END\s*FILE\s*---",
            re.DOTALL,
        )
        matches = file_pattern.findall(response)

        for filepath, content in matches:
            filepath = filepath.strip()
            content = content.strip()
            if filepath and content:
                write_result = self._write_file(filepath, content)
                files_written.append(filepath)
                yield CodingEvent(
                    "coding_step",
                    data={
                        "action": "write_file",
                        "file": filepath,
                        "size": len(content),
                        "result": write_result[:200],
                    },
                    content=f"Created {filepath}",
                )

        if not matches:
            # If no structured file output, try to extract code blocks
            code_blocks = re.findall(
                r"```(?:\w+)?\n(.*?)```", response, re.DOTALL
            )
            if code_blocks and len(code_blocks) == 1:
                # Single code block: write as main file
                filepath = "main.py"
                write_result = self._write_file(filepath, code_blocks[0].strip())
                files_written.append(filepath)
                yield CodingEvent(
                    "coding_step",
                    data={
                        "action": "write_file",
                        "file": filepath,
                        "size": len(code_blocks[0]),
                    },
                    content=f"Created {filepath}",
                )

        # Update sandbox state
        self._list_sandbox_files()

        summary = (
            f"Implemented {len(files_written)} file(s): "
            + ", ".join(files_written)
        ) if files_written else "No files written (check LLM response format)"

        yield CodingEvent(
            "coding_step",
            data={
                "action": "implement_done",
                "files_written": files_written,
                "total_files": len(files_written),
            },
            content=summary,
        )

        return response

    def _phase_test(
        self,
        model: str,
    ) -> Generator[CodingEvent, None, tuple[bool, str]]:
        """Test phase: run tests in the sandbox.

        Yields CodingEvents and returns (passed, output).
        """
        yield CodingEvent("coding_status", content="Running tests...")

        test_cmd = self._config.auto_test_command
        output = self._exec_in_sandbox(test_cmd, timeout=60)

        # Determine pass/fail from output
        passed = (
            "passed" in output.lower()
            and "failed" not in output.lower()
            and "error" not in output.lower()
        ) or (
            "0 errors" in output.lower()
            or "OK" in output
        )

        # Also consider exit code patterns
        if "no tests ran" in output.lower():
            # No tests found: consider as neutral pass
            passed = True

        self._sandbox_state.last_test_passed = passed
        self._sandbox_state.last_test_output = output

        yield CodingEvent(
            "coding_test",
            data={
                "passed": passed,
                "output_preview": output[:1000],
                "command": test_cmd,
            },
            content=f"Tests {'passed' if passed else 'FAILED'}",
        )

        return passed, output

    def _phase_fix(
        self,
        test_output: str,
        model: str,
        max_retries: int | None = None,
    ) -> Generator[CodingEvent, None, bool]:
        """Fix phase: attempt to fix failing tests.

        Iteratively asks the LLM to fix the code based on test output,
        re-runs tests after each fix attempt.

        Yields CodingEvents and returns True if tests eventually pass.
        """
        retries = max_retries or self._config.max_fix_retries
        attempt = 0

        while attempt < retries:
            attempt += 1
            yield CodingEvent(
                "coding_fix",
                data={
                    "attempt": attempt,
                    "max_retries": retries,
                },
                content=f"Fix attempt {attempt}/{retries}...",
            )

            fix_prompt = (
                f"The tests failed with this output:\n{test_output[:3000]}\n\n"
                f"Fix the code. Output the corrected files in this format:\n"
                f"--- FILE: <filepath> ---\n<content>\n--- END FILE ---\n\n"
                f"Only output files that need to change."
            )

            system = (
                "You are a coding agent fixing test failures. "
                "Analyze the error carefully and fix the root cause. "
                "Output only the corrected files. "
                "You can use web search to look up documentation if needed."
            )

            fix_result = self._call_llm(
                fix_prompt, system, model, phase="fix"
            )
            response = fix_result.text or ""

            # Apply fixes
            file_pattern = re.compile(
                r"---\s*FILE:\s*(.+?)\s*---\s*\n(.*?)---\s*END\s*FILE\s*---",
                re.DOTALL,
            )
            matches = file_pattern.findall(response)
            for filepath, content in matches:
                filepath = filepath.strip()
                content = content.strip()
                if filepath and content:
                    self._write_file(filepath, content)

            # Re-run tests
            test_output = self._exec_in_sandbox(
                self._config.auto_test_command, timeout=60
            )
            passed = (
                "passed" in test_output.lower()
                and "failed" not in test_output.lower()
                and "error" not in test_output.lower()
            ) or "OK" in test_output

            self._sandbox_state.last_test_passed = passed
            self._sandbox_state.last_test_output = test_output

            yield CodingEvent(
                "coding_test",
                data={
                    "passed": passed,
                    "output_preview": test_output[:1000],
                    "fix_attempt": attempt,
                },
                content=(
                    f"Fix {attempt}: tests {'passed' if passed else 'still failing'}"
                ),
            )

            if passed:
                return True

        return False

    # -- Main execution -------------------------------------------------------

    def execute_task(
        self,
        message: str,
        model: str,
        directives: TurnDirectives | None = None,
        images: list[str] | None = None,
        web_search: bool = False,
        think: bool = False,
    ) -> Generator[CodingEvent, None, dict[str, Any]]:
        """Execute a coding task as part of the ongoing conversation.

        This is the main entry point. Runs the adaptive pipeline:
        PLAN -> IMPLEMENT -> TEST (optional) -> FIX (optional) -> DONE

        Each phase can be skipped via directives.
        The LLM calls within each phase have access to the full chat
        pipeline capabilities: vision delegation (images), web search,
        think mode, all registered tools, and active plugins.

        Args:
            message: User message (after /code prefix removal).
            model: Model to use for inference.
            directives: Parsed runtime directives (or auto-parsed if None).
            images: Base64-encoded images for vision delegation (S95).
                If the model lacks vision, a vision-capable model
                analyzes the images and injects the description.
            web_search: Enable web search for documentation lookup.
            think: Enable chain-of-thought reasoning mode.

        Yields:
            CodingEvent instances for streaming.

        Returns:
            Result dict with summary, files, test status, etc.
        """
        if directives is None:
            directives = parse_directives(message)

        # Set per-turn feature state (read by _call_llm)
        self._turn_images = images
        self._turn_web_search = web_search
        self._turn_think = think
        self._last_tool_calls = []
        self._last_vision_meta = {}
        self._last_plugin_annotations = []

        self._last_activity = time.time()
        self._sandbox_state.turn_count += 1
        turn_num = self._sandbox_state.turn_count

        # Detect follow-up vs. fresh task
        is_followup = (
            turn_num > 1
            and len(self._sandbox_state.files) > 0
        )

        # Auto-skip plan for simple follow-ups unless user asks complex changes
        if is_followup and not directives.skip_plan:
            # Heuristic: short modification requests can skip planning
            word_count = len(message.split())
            if word_count <= 15:
                directives.skip_plan = True

        yield CodingEvent(
            "coding_status",
            data={
                "turn": turn_num,
                "is_followup": is_followup,
                "has_images": bool(images),
                "web_search": web_search,
                "think": think,
                "directives": {
                    "skip_test": directives.skip_test,
                    "skip_plan": directives.skip_plan,
                    "skip_fix": directives.skip_fix,
                    "plan_only": directives.plan_only,
                },
            },
            content=f"Turn {turn_num}: starting...",
        )

        result: dict[str, Any] = {
            "turn": turn_num,
            "plan": "",
            "files_written": [],
            "test_passed": None,
            "fix_attempts": 0,
            "summary": "",
        }

        full_response_parts: list[str] = []

        # -- PHASE 1: PLAN (unless skipped) ----------------------------------
        plan_text = ""
        if not directives.skip_plan:
            plan_gen = self._phase_plan(message, model, is_followup)
            try:
                while True:
                    event = next(plan_gen)
                    yield event
            except StopIteration as e:
                plan_text = e.value or ""
            result["plan"] = plan_text
            full_response_parts.append(f"Plan:\n{plan_text}")

            if directives.plan_only:
                result["summary"] = "Plan generated (plan-only mode)"
                self._update_cumulative_summary(message, result["summary"])
                self._save_turn_to_conversation(
                    message,
                    f"[Code Agent - Plan Only]\n{plan_text}",
                    model,
                )
                yield CodingEvent(
                    "coding_done",
                    data=result,
                    content="Plan generated (plan-only mode)",
                )
                return result

        # -- PHASE 2: IMPLEMENT -----------------------------------------------
        impl_gen = self._phase_implement(
            directives.cleaned_message, plan_text, model
        )
        impl_response = ""
        try:
            while True:
                event = next(impl_gen)
                yield event
        except StopIteration as e:
            impl_response = e.value or ""  # noqa: F841

        # Update files list from sandbox
        current_files = self._list_sandbox_files()  # noqa: F841
        result["files_written"] = list(self._sandbox_state.files)
        full_response_parts.append(
            f"Files: {', '.join(self._sandbox_state.files)}"
        )

        # -- PHASE 3: TEST (unless skipped) -----------------------------------
        test_passed = None
        test_output = ""

        if (
            self._config.auto_test
            and not directives.skip_test
            and self._sandbox_state.files
        ):
            test_gen = self._phase_test(model)
            try:
                while True:
                    event = next(test_gen)
                    yield event
            except StopIteration as e:
                test_passed, test_output = e.value

            result["test_passed"] = test_passed

            # -- PHASE 4: FIX (if tests failed and not skipped) ----------------
            if not test_passed and not directives.skip_fix:
                fix_retries = (
                    directives.max_fix_retries
                    or self._config.max_fix_retries
                )
                fix_gen = self._phase_fix(test_output, model, fix_retries)
                try:
                    while True:
                        event = next(fix_gen)
                        yield event
                except StopIteration as e:
                    test_passed = e.value
                    result["test_passed"] = test_passed

                result["fix_attempts"] = fix_retries

        # -- DONE -------------------------------------------------------------
        if test_passed is True:
            summary = "Implementation complete, all tests passing"
        elif test_passed is False:
            summary = "Implementation complete, tests still failing"
        elif directives.skip_test:
            summary = "Implementation complete (tests skipped)"
        else:
            summary = "Implementation complete"

        result["summary"] = summary
        full_response_parts.append(summary)

        # Include pipeline metadata in result (vision, tools, plugins)
        if self._last_vision_meta:
            result["vision_meta"] = self._last_vision_meta
        if self._last_tool_calls:
            result["tool_calls"] = self._last_tool_calls
        if self._last_plugin_annotations:
            result["plugin_annotations"] = self._last_plugin_annotations

        # Update cumulative working memory
        self._update_cumulative_summary(message, summary)

        # Save to conversation history for full memory across turns
        assistant_content = (
            f"[Code Agent - Turn {turn_num}]\n"
            + "\n".join(full_response_parts)
        )
        self._save_turn_to_conversation(message, assistant_content, model)

        # Clear per-turn feature state
        self._turn_images = None
        self._turn_web_search = False
        self._turn_think = False

        yield CodingEvent(
            "coding_done",
            data=result,
            content=summary,
        )

        return result

    # -- Cleanup --------------------------------------------------------------

    def destroy(self) -> bool:
        """Destroy the sandbox session and clean up."""
        if self._mgr is None or self._sandbox_session is None:
            return False
        try:
            result = self._mgr.destroy_sandbox(self._session_id)
            self._sandbox_session = None
            logger.info(
                "Chat coding session destroyed: %s (conv=%s)",
                self._session_id, self._conversation_id[:8],
            )
            return result
        except Exception as exc:
            logger.warning(
                "Chat coding destroy failed: %s: %s",
                self._session_id, exc,
            )
            return False

    def get_status(self) -> dict[str, Any]:
        """Get session status for API responses."""
        return {
            "session_id": self._session_id,
            "conversation_id": self._conversation_id,
            "active": self.active,
            "expired": self.expired,
            "created_at": self._created_at,
            "last_activity": self._last_activity,
            "turn_count": self._sandbox_state.turn_count,
            "files": list(self._sandbox_state.files),
            "last_test_passed": self._sandbox_state.last_test_passed,
            "compression_active": self._last_compression_result is not None,
        }


# ---------------------------------------------------------------------------
# ChatCodingManager
# ---------------------------------------------------------------------------

class ChatCodingManager:
    """Pool of active chat coding sessions.

    Manages session lifecycle, expiry cleanup, and provides the
    entry point for conversation-keyed coding sessions.
    """

    def __init__(
        self,
        sandbox_mgr: "SandboxManager | None" = None,
        config: ChatCodingConfig | None = None,
    ):
        self._mgr = sandbox_mgr or _default_sandbox_manager
        self._config = config or _load_config()
        self._sessions: dict[str, ChatCodingSession] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._config.enabled = value

    @property
    def config(self) -> ChatCodingConfig:
        return self._config

    @property
    def available(self) -> bool:
        """Whether chat coding can operate (dependencies present)."""
        return (
            CHAT_CODING_AVAILABLE
            and self._mgr is not None
        )

    def get_or_create_session(
        self,
        conversation_id: str,
        llm_call: "RichLLMCall | SimpleLLMCall | None" = None,
    ) -> ChatCodingSession:
        """Get an existing session or create a new one for a conversation.

        Args:
            conversation_id: The conversation to attach to.
            llm_call: Callable for LLM interaction. Supports two signatures:
                Rich: (messages, model, LLMCallContext) -> LLMCallResult
                    Full pipeline: vision, tools, plugins, web search.
                Simple: (prompt, system, model) -> str
                    Basic text-in/text-out.

        Returns:
            A ChatCodingSession ready for task execution.

        Raises:
            RuntimeError: If max concurrent sessions exceeded or
                dependencies unavailable.
        """
        if not self.available:
            raise RuntimeError(
                "Chat coding not available (missing dependencies)"
            )

        with self._lock:
            # Return existing active session
            existing = self._sessions.get(conversation_id)
            if existing is not None and not existing.expired:
                return existing

            # Clean up expired session
            if existing is not None and existing.expired:
                existing.destroy()
                del self._sessions[conversation_id]

            # Check concurrent limit
            active_count = sum(
                1 for s in self._sessions.values() if not s.expired
            )
            if active_count >= self._config.max_concurrent_sessions:
                raise RuntimeError(
                    f"Maximum concurrent chat coding sessions reached "
                    f"({self._config.max_concurrent_sessions})"
                )

            # Create new session
            session = ChatCodingSession(
                conversation_id=conversation_id,
                sandbox_mgr=self._mgr,
                config=self._config,
                llm_call=llm_call,
            )
            self._sessions[conversation_id] = session
            return session

    def get_session(
        self, conversation_id: str
    ) -> ChatCodingSession | None:
        """Get an existing session by conversation ID."""
        with self._lock:
            session = self._sessions.get(conversation_id)
            if session is not None and session.expired:
                session.destroy()
                del self._sessions[conversation_id]
                return None
            return session

    def destroy_session(self, conversation_id: str) -> bool:
        """Destroy a specific chat coding session."""
        with self._lock:
            session = self._sessions.pop(conversation_id, None)
        if session is None:
            return False
        return session.destroy()

    def cleanup_expired(self) -> int:
        """Destroy all expired sessions. Returns count destroyed."""
        to_destroy: list[str] = []
        with self._lock:
            for cid, session in self._sessions.items():
                if session.expired:
                    to_destroy.append(cid)

        count = 0
        for cid in to_destroy:
            if self.destroy_session(cid):
                count += 1
        return count

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all active chat coding sessions."""
        with self._lock:
            result = []
            for cid, session in self._sessions.items():
                result.append(session.get_status())
            return result

    def active_session_count(self) -> int:
        """Count of active (non-expired) sessions."""
        with self._lock:
            return sum(
                1 for s in self._sessions.values()
                if not s.expired
            )

    def get_status(self) -> dict[str, Any]:
        """Get chat coding status for API responses."""
        return {
            "enabled": self._config.enabled,
            "available": self.available,
            "session_timeout_minutes": self._config.session_timeout_minutes,
            "max_concurrent_sessions": self._config.max_concurrent_sessions,
            "active_sessions": self.active_session_count(),
            "auto_test": self._config.auto_test,
            "max_fix_retries": self._config.max_fix_retries,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

chat_coding_manager = ChatCodingManager()
