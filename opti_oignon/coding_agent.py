#!/usr/bin/env python3
"""
CODING AGENT - OPTI-OIGNON v1.8.3 (S74/S76/S77/S78/S79/S80/S81)
=========================================

Multi-step autonomous coding agent: plan -> implement -> test -> fix -> review -> apply.
The "local Claude Code" milestone.

Uses S73 SandboxToolSession for all filesystem operations (fully isolated).
Uses S65 token budget for context window management.
Uses S66 conversation compressor for long coding sessions.

S77: Batch file reads via tar+base64 (SQ-06), background execution support (SQ-07).
S79: Auto-retry on transient sandbox errors with exponential backoff.
S80: Robust JSON parsing (json_repair), plan retries with fallback,
     WorkingMemory structured scratchpad for cross-step context.
S81: Cascading model escalation on fix failures, optional per-step routing,
     security audit hardening.

SECURITY: The apply phase is the ONLY exit from the sandbox.
checkpoint_before_apply MUST always be True and is NOT overridable.

Author: Leon
"""

import base64
import difflib
import enum
import io
import logging
import os
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from opti_oignon.sandbox_tools import SandboxToolSession
    SANDBOX_TOOLS_AVAILABLE = True
except ImportError:
    SANDBOX_TOOLS_AVAILABLE = False
    SandboxToolSession = None

try:
    from opti_oignon.sandbox_manager import (
        SANDBOX_AVAILABLE,
        SandboxManager,
    )
    from opti_oignon.sandbox_manager import (
        sandbox_manager as _default_sandbox_manager,
    )
except ImportError:
    SANDBOX_AVAILABLE = False
    SandboxManager = None
    _default_sandbox_manager = None

try:
    from opti_oignon.prompt_optimization import (
        prompt_budget_manager,
        prompt_template_engine,
    )
    PROMPT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PROMPT_OPTIMIZATION_AVAILABLE = False
    prompt_budget_manager = None
    prompt_template_engine = None

try:
    from opti_oignon.conversation_compressor import conversation_compressor
    COMPRESSOR_AVAILABLE = True
except ImportError:
    COMPRESSOR_AVAILABLE = False
    conversation_compressor = None

try:
    from opti_oignon.session_fingerprint import (
        FINGERPRINT_AVAILABLE,
        FingerprintManager,
        UserPreferencesStore,
    )
except ImportError:
    FINGERPRINT_AVAILABLE = False
    FingerprintManager = None
    UserPreferencesStore = None

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

try:
    from opti_oignon.json_repair import (
        JSON_RETRY_SUFFIX,
        SIMPLIFIED_PLAN_SUFFIX,
        parse_numbered_list,
        repair_json,
        repair_json_or_list,
    )
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    repair_json = None
    repair_json_or_list = None
    parse_numbered_list = None
    JSON_RETRY_SUFFIX = ""
    SIMPLIFIED_PLAN_SUFFIX = ""

try:
    from opti_oignon.coding_history import (
        CODING_HISTORY_AVAILABLE as _HISTORY_AVAILABLE,
    )
    from opti_oignon.coding_history import (
        coding_history_store as _history_store,
    )
except ImportError:
    _HISTORY_AVAILABLE = False
    _history_store = None

try:
    from opti_oignon.cascading import (
        CascadeResult,
        CascadingInference,
    )
    CASCADING_AVAILABLE = True
except ImportError:
    CASCADING_AVAILABLE = False
    CascadingInference = None
    CascadeResult = None

try:
    from opti_oignon.smart_router import SmartRouter
    SMART_ROUTER_AVAILABLE = True
except ImportError:
    SMART_ROUTER_AVAILABLE = False
    SmartRouter = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "config", "coding_agent.yaml"
)


@dataclass
class CodingAgentConfig:
    """Configuration for the coding agent."""

    enabled: bool = True
    max_iterations: int = 10
    max_fix_retries: int = 3
    max_auto_retries: int = 2
    max_plan_retries: int = 3
    retry_backoff_seconds: list[float] = field(
        default_factory=lambda: [1.0, 2.0]
    )
    auto_test: bool = True
    auto_test_command: str = "python3 -m pytest -x --tb=short"
    default_model: str | None = None
    planning_model: str | None = None
    checkpoint_after_plan: bool = True
    checkpoint_before_apply: bool = True  # NEVER set to False
    context_window_reserve: int = 2048
    # Cascading escalation in fix loop (S81)
    enable_cascading: bool = True
    escalate_after_failures: int = 2
    # Per-step routing: route simple steps to fast tier (S81, experimental)
    per_step_routing: bool = False


def _load_config() -> CodingAgentConfig:
    """Load coding agent configuration from YAML with safe defaults."""
    try:
        if _yaml is not None and os.path.isfile(_CONFIG_PATH):
            with open(_CONFIG_PATH, encoding="utf-8") as fh:
                raw = _yaml.safe_load(fh) or {}
            cfg = CodingAgentConfig(
                enabled=raw.get("enabled", True),
                max_iterations=raw.get("max_iterations", 10),
                max_fix_retries=raw.get("max_fix_retries", 3),
                max_auto_retries=raw.get("max_auto_retries", 2),
                max_plan_retries=raw.get("max_plan_retries", 3),
                retry_backoff_seconds=raw.get(
                    "retry_backoff_seconds", [1.0, 2.0]
                ),
                auto_test=raw.get("auto_test", True),
                auto_test_command=raw.get(
                    "auto_test_command",
                    "python3 -m pytest -x --tb=short",
                ),
                default_model=raw.get("default_model"),
                planning_model=raw.get("planning_model"),
                checkpoint_after_plan=raw.get("checkpoint_after_plan", True),
                context_window_reserve=raw.get("context_window_reserve", 2048),
                enable_cascading=raw.get("enable_cascading", True),
                escalate_after_failures=raw.get("escalate_after_failures", 2),
                per_step_routing=raw.get("per_step_routing", False),
            )
            # SECURITY: checkpoint_before_apply is ALWAYS True
            cfg.checkpoint_before_apply = True
            return cfg
    except Exception as exc:
        logger.warning("Failed to load coding agent config: %s", exc)
    return CodingAgentConfig()


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class CodingPhase(str, enum.Enum):
    """Phases of the coding agent loop."""

    IDLE = "idle"
    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    FIXING = "fixing"
    REVIEWING = "reviewing"
    APPLYING = "applying"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class CheckpointResult(str, enum.Enum):
    """Human checkpoint decision."""

    APPROVE = "approve"
    MODIFY = "modify"
    ABORT = "abort"


class PlanStepType(str, enum.Enum):
    """Types of plan steps."""

    CREATE = "create"
    EDIT = "edit"
    TEST = "test"
    BASH = "bash"


@dataclass
class PlanStep:
    """A single step in the coding plan."""

    step_number: int
    step_type: PlanStepType
    description: str
    file_path: str = ""
    command: str = ""
    content: str = ""
    old_str: str = ""
    new_str: str = ""
    completed: bool = False
    result: str = ""
    error: str = ""


@dataclass
class CodingPlan:
    """Structured plan generated by the LLM."""

    task: str
    steps: list[PlanStep] = field(default_factory=list)
    summary: str = ""
    estimated_files: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.completed)

    def to_dict(self) -> dict[str, Any]:
        """Serialize plan to dict."""
        return {
            "task": self.task,
            "summary": self.summary,
            "estimated_files": self.estimated_files,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "steps": [
                {
                    "step_number": s.step_number,
                    "step_type": s.step_type.value,
                    "description": s.description,
                    "file_path": s.file_path,
                    "command": s.command,
                    "completed": s.completed,
                    "result": s.result[:500] if s.result else "",
                    "error": s.error[:500] if s.error else "",
                }
                for s in self.steps
            ],
        }


@dataclass
class CodingHistoryEntry:
    """A single action in the coding history."""

    timestamp: float = field(default_factory=time.time)
    phase: str = ""
    action: str = ""
    detail: str = ""
    success: bool = True


@dataclass
class FileDiff:
    """Represents changes to a single file."""

    path: str
    is_new: bool = False
    is_deleted: bool = False
    diff_lines: list[str] = field(default_factory=list)
    original_content: str = ""
    modified_content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "is_new": self.is_new,
            "is_deleted": self.is_deleted,
            "diff": "\n".join(self.diff_lines),
        }


@dataclass
class TestResult:
    """Result of running tests in the sandbox."""

    passed: bool = False
    output: str = ""
    error: str = ""
    return_code: int = -1
    test_count: int = 0
    failures: int = 0


@dataclass
class WorkingMemory:
    """Structured scratchpad maintained across coding agent steps.

    Prevents context loss over many steps by giving the LLM a compact
    summary of what happened so far. Each field is updated by the LLM
    after each step via a structured memory update prompt.

    Token budget: ~200-400 tokens when serialized to compact form.
    Persisted to SQLite via coding_history.db (working_memory table).
    """

    task_id: str = ""
    decisions: list[str] = field(default_factory=list)
    modified_files: dict[str, str] = field(default_factory=dict)
    errors_encountered: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    progress_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize working memory to dict."""
        return {
            "task_id": self.task_id,
            "decisions": self.decisions,
            "modified_files": self.modified_files,
            "errors_encountered": self.errors_encountered,
            "open_questions": self.open_questions,
            "progress_notes": self.progress_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkingMemory":
        """Deserialize working memory from dict."""
        return cls(
            task_id=data.get("task_id", ""),
            decisions=data.get("decisions", []),
            modified_files=data.get("modified_files", {}),
            errors_encountered=data.get("errors_encountered", []),
            open_questions=data.get("open_questions", []),
            progress_notes=data.get("progress_notes", []),
        )

    def to_compact(self, max_tokens: int = 400) -> str:
        """Serialize to compact text for LLM context injection.

        Produces a structured text block that fits within max_tokens
        (approximate, using 4 chars per token heuristic).

        Args:
            max_tokens: Approximate token budget for the compact form.

        Returns:
            Compact text representation of working memory.
        """
        max_chars = max_tokens * 4
        parts = []

        if self.decisions:
            items = "; ".join(self.decisions[-5:])
            parts.append(f"DECISIONS: {items}")

        if self.modified_files:
            items = "; ".join(
                f"{k}: {v}" for k, v in list(self.modified_files.items())[-5:]
            )
            parts.append(f"MODIFIED: {items}")

        if self.errors_encountered:
            items = "; ".join(self.errors_encountered[-3:])
            parts.append(f"ERRORS: {items}")

        if self.open_questions:
            items = "; ".join(self.open_questions[-3:])
            parts.append(f"OPEN: {items}")

        if self.progress_notes:
            items = "; ".join(self.progress_notes[-3:])
            parts.append(f"PROGRESS: {items}")

        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."
        return text

    def update_from_step(
        self,
        step_number: int,
        step_type: str,
        file_path: str,
        result: str,
        error: str,
    ) -> None:
        """Update working memory after a step execution.

        This is the fallback update when the LLM does not provide
        a structured memory update. Extracts basic info from step
        results.

        Args:
            step_number: The step number that just executed.
            step_type: Type of step (create, edit, bash, test).
            file_path: File path involved (if any).
            result: Step result string.
            error: Step error string (empty if success).
        """
        if error:
            self.errors_encountered.append(
                f"Step {step_number} ({step_type}): {error[:200]}"
            )
        elif file_path and step_type in ("create", "edit"):
            action = "created" if step_type == "create" else "edited"
            self.modified_files[file_path] = (
                f"{action} (step {step_number})"
            )
        if result and step_type == "test":
            note = "passed" if "passed" in result.lower() else "ran"
            self.progress_notes.append(
                f"Step {step_number}: tests {note}"
            )

    def update_from_llm(self, update_data: dict[str, Any]) -> None:
        """Apply a structured memory update from the LLM.

        The LLM may return a dict with any subset of the memory
        fields to update/append.

        Args:
            update_data: Dict with optional keys matching memory fields.
        """
        if "decisions" in update_data:
            new_decisions = update_data["decisions"]
            if isinstance(new_decisions, list):
                self.decisions.extend(str(d) for d in new_decisions)
            elif isinstance(new_decisions, str):
                self.decisions.append(new_decisions)

        if "modified_files" in update_data:
            new_files = update_data["modified_files"]
            if isinstance(new_files, dict):
                self.modified_files.update(
                    {str(k): str(v) for k, v in new_files.items()}
                )

        if "errors_encountered" in update_data:
            new_errors = update_data["errors_encountered"]
            if isinstance(new_errors, list):
                self.errors_encountered.extend(str(e) for e in new_errors)
            elif isinstance(new_errors, str):
                self.errors_encountered.append(new_errors)

        if "open_questions" in update_data:
            new_qs = update_data["open_questions"]
            if isinstance(new_qs, list):
                self.open_questions = [str(q) for q in new_qs]
            elif isinstance(new_qs, str):
                self.open_questions = [new_qs]

        if "progress_notes" in update_data:
            new_notes = update_data["progress_notes"]
            if isinstance(new_notes, list):
                self.progress_notes.extend(str(n) for n in new_notes)
            elif isinstance(new_notes, str):
                self.progress_notes.append(new_notes)

    def trim(self, max_items: int = 10) -> None:
        """Trim memory lists to prevent unbounded growth.

        Keeps only the most recent items per field.

        Args:
            max_items: Maximum items to keep per list field.
        """
        self.decisions = self.decisions[-max_items:]
        self.errors_encountered = self.errors_encountered[-max_items:]
        self.open_questions = self.open_questions[-max_items:]
        self.progress_notes = self.progress_notes[-max_items:]
        # Keep only recent modified files
        if len(self.modified_files) > max_items:
            keys = list(self.modified_files.keys())
            for k in keys[:-max_items]:
                del self.modified_files[k]


# ---------------------------------------------------------------------------
# LLM interaction helpers
# ---------------------------------------------------------------------------

_PLANNING_SYSTEM_PROMPT = """You are a coding agent with access to a sandboxed workspace.
Your task is to decompose a coding task into concrete steps.

Available tools in the sandbox:
- bash(command): Execute shell commands
- view(path): Read file contents or list directories
- create_file(path, content): Create or overwrite files
- str_replace(path, old_str, new_str): Edit files by replacing unique strings

Respond with a JSON object containing:
{
  "summary": "Brief description of what will be done",
  "estimated_files": <number of files to create/modify>,
  "steps": [
    {
      "step_type": "create|edit|test|bash",
      "description": "What this step does",
      "file_path": "path/to/file (for create/edit)",
      "command": "command to run (for bash/test)",
      "content": "file content (for create)",
      "old_str": "string to find (for edit)",
      "new_str": "replacement string (for edit)"
    }
  ]
}

Rules:
- All paths are relative to /workspace/
- Keep steps atomic and testable
- Include test steps after implementation
- Do NOT use sudo, curl, wget, or any network commands
- Respond ONLY with the JSON object, no markdown fences"""

_FIX_SYSTEM_PROMPT = """You are a coding agent fixing test failures in a sandboxed workspace.

Analyze the test failure output and the relevant file contents.
Propose a fix using one of these approaches:
- str_replace: Find and replace a specific string in a file
- create_file: Rewrite the entire file with corrections

Respond with a JSON object:
{
  "analysis": "Brief explanation of the bug",
  "fix_type": "str_replace|create_file",
  "file_path": "path/to/file",
  "old_str": "string to replace (for str_replace)",
  "new_str": "replacement (for str_replace)",
  "content": "full file content (for create_file)"
}

Respond ONLY with the JSON object, no markdown fences"""


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON response from the LLM, with tolerant repair.

    Uses the json_repair module (S80) for progressive repair when
    available, falling back to basic fence-stripping + json.loads.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If parsing fails after all repair strategies.
    """
    import json

    if JSON_REPAIR_AVAILABLE and repair_json is not None:
        try:
            result = repair_json(text)
            if isinstance(result, dict):
                return result
            # If we got a list, wrap it
            if isinstance(result, list):
                return {"steps": result}
        except ValueError as exc:
            raise ValueError(f"JSON repair failed: {exc}") from exc
    else:
        # Fallback: basic fence stripping (pre-S80 behavior)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM JSON response: %s", exc)
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc


def _build_plan_from_response(
    task: str, response_data: dict[str, Any]
) -> CodingPlan:
    """Build a CodingPlan from parsed LLM response."""
    steps = []
    for i, step_raw in enumerate(response_data.get("steps", []), start=1):
        step_type_str = step_raw.get("step_type", "bash")
        try:
            step_type = PlanStepType(step_type_str)
        except ValueError:
            step_type = PlanStepType.BASH

        steps.append(PlanStep(
            step_number=i,
            step_type=step_type,
            description=step_raw.get("description", ""),
            file_path=step_raw.get("file_path", ""),
            command=step_raw.get("command", ""),
            content=step_raw.get("content", ""),
            old_str=step_raw.get("old_str", ""),
            new_str=step_raw.get("new_str", ""),
        ))

    return CodingPlan(
        task=task,
        steps=steps,
        summary=response_data.get("summary", ""),
        estimated_files=response_data.get("estimated_files", 0),
    )


def _build_fix_from_response(
    response_data: dict[str, Any],
) -> dict[str, str]:
    """Extract fix instructions from parsed LLM response."""
    return {
        "analysis": response_data.get("analysis", ""),
        "fix_type": response_data.get("fix_type", "str_replace"),
        "file_path": response_data.get("file_path", ""),
        "old_str": response_data.get("old_str", ""),
        "new_str": response_data.get("new_str", ""),
        "content": response_data.get("content", ""),
    }


# Directories that must NEVER be used as apply targets
_FORBIDDEN_TARGETS = frozenset({
    "/", "/bin", "/sbin", "/usr", "/usr/bin", "/usr/sbin",
    "/lib", "/lib64", "/boot", "/dev", "/proc", "/sys",
    "/etc", "/root", "/home", "/mnt", "/var", "/tmp",
    "/run", "/snap", "/opt",
})


def _validate_apply_target(target: str) -> None:
    """Validate that target_path is not a sensitive system directory.

    Raises:
        ValueError: If target resolves to a forbidden path.
    """
    resolved = os.path.realpath(os.path.expanduser(target))
    # Check exact match with forbidden roots
    if resolved in _FORBIDDEN_TARGETS:
        raise ValueError(
            f"Refusing to apply to system directory: {resolved}"
        )
    # Check if target is a parent of common sensitive paths
    for forbidden in _FORBIDDEN_TARGETS:
        if forbidden != "/" and resolved == forbidden:
            raise ValueError(
                f"Refusing to apply to protected path: {resolved}"
            )
    # Block anything directly under / that is not a user project
    parts = resolved.strip("/").split("/")
    if len(parts) <= 1 and resolved.startswith("/"):
        raise ValueError(
            f"Refusing to apply to top-level system path: {resolved}"
        )


def _is_within_target(target: str, dest: str) -> bool:
    """Return True iff ``dest`` resolves to a path inside ``target``.

    Defends the apply boundary (the only exit from the sandbox) against a
    per-file diff path that contains ``..`` or that resolves -- via an existing
    symlink in the project tree -- outside the project. ``os.path.realpath``
    resolves both, then ``commonpath`` confirms containment (which also avoids
    the ``startswith`` sibling-prefix pitfall, e.g. ``/a/proj`` vs ``/a/projX``).
    """
    try:
        target_real = os.path.realpath(target)
        dest_real = os.path.realpath(dest)
    except OSError:
        return False
    if dest_real == target_real:
        return True
    try:
        return os.path.commonpath([dest_real, target_real]) == target_real
    except ValueError:
        # Different roots (or a relative path slipped through) -> not contained.
        return False


# ---------------------------------------------------------------------------
# CodingAgent
# ---------------------------------------------------------------------------

class CodingAgent:
    """Multi-step autonomous coding agent with sandbox isolation.

    Phases: plan -> implement -> test -> fix -> review -> apply.

    The agent uses a SandboxToolSession for ALL filesystem operations.
    Human checkpoints gate critical transitions (plan approval, apply).

    SECURITY: checkpoint_before_apply is ALWAYS enforced. The apply
    phase is the ONLY way files leave the sandbox, and it requires
    explicit human approval.
    """

    def __init__(
        self,
        sandbox_session: "SandboxToolSession | None" = None,
        sandbox_manager: "SandboxManager | None" = None,
        model: str | None = None,
        llm_call: Callable | None = None,
        config: CodingAgentConfig | None = None,
        fingerprint_manager: "FingerprintManager | None" = None,
        cascading_engine: "CascadingInference | None" = None,
        smart_router: "SmartRouter | None" = None,
    ):
        """Initialize the coding agent.

        Args:
            sandbox_session: Pre-configured SandboxToolSession, or None
                to create one internally.
            sandbox_manager: SandboxManager instance (used if
                sandbox_session is None).
            model: Model name for LLM calls. Falls back to config default.
            llm_call: Callable for LLM interaction. Signature:
                llm_call(prompt: str, system: str, model: str) -> str
            config: Agent configuration. Loaded from YAML if None.
            fingerprint_manager: Session fingerprint tracker (S75).
                Created automatically if None and feature is available.
            cascading_engine: CascadingInference instance for model
                escalation on fix failures (S81). Auto-created if None
                and feature is available + enabled in config.
            smart_router: SmartRouter instance for per-step routing
                (S81, experimental). Auto-created if None and
                per_step_routing is enabled in config.
        """
        self._config = config or _load_config()
        # SECURITY: Always enforce checkpoint before apply
        self._config.checkpoint_before_apply = True

        self._model = model or self._config.default_model
        self._planning_model = self._config.planning_model or self._model
        self._llm_call = llm_call

        # Sandbox session management
        if sandbox_session is not None:
            self._session = sandbox_session
            self._owns_session = False
        else:
            mgr = sandbox_manager or _default_sandbox_manager
            if mgr is not None and SANDBOX_TOOLS_AVAILABLE:
                self._session = SandboxToolSession(sandbox_mgr=mgr)
                self._owns_session = True
            else:
                self._session = None
                self._owns_session = False

        # State
        self._phase = CodingPhase.IDLE
        self._plan: CodingPlan | None = None
        self._history: list[CodingHistoryEntry] = []
        self._test_results: list[TestResult] = []
        self._diffs: list[FileDiff] = []
        self._task: str = ""
        self._project_path: str | None = None
        self._original_files: dict[str, str] = {}
        self._current_step: int = 0
        self._iteration: int = 0
        self._fix_count: int = 0
        self._task_id: str = ""
        self._lock = threading.Lock()
        self._progress_callbacks: list[Callable] = []
        self._diffs_hash: str = ""  # integrity hash set by generate_diffs

        # Session fingerprint (S75)
        if fingerprint_manager is not None:
            self._fingerprint = fingerprint_manager
        elif FINGERPRINT_AVAILABLE and FingerprintManager is not None:
            self._fingerprint = FingerprintManager()
        else:
            self._fingerprint = None

        # Persistent history store (S76)
        self._history_store = (
            _history_store if _HISTORY_AVAILABLE else None
        )
        self._test_run_counter = 0

        # Working memory (S80)
        self._working_memory: WorkingMemory | None = None

        # Cascading model escalation (S81)
        if cascading_engine is not None:
            self._cascading = cascading_engine
        elif (
            CASCADING_AVAILABLE
            and CascadingInference is not None
            and self._config.enable_cascading
        ):
            try:
                self._cascading = CascadingInference()
            except Exception as exc:
                logger.debug("Cascading engine init failed: %s", exc)
                self._cascading = None
        else:
            self._cascading = None

        # Smart router for per-step routing (S81, experimental)
        if smart_router is not None:
            self._smart_router = smart_router
        elif (
            SMART_ROUTER_AVAILABLE
            and SmartRouter is not None
            and self._config.per_step_routing
        ):
            try:
                self._smart_router = SmartRouter()
            except Exception as exc:
                logger.debug("SmartRouter init failed: %s", exc)
                self._smart_router = None
        else:
            self._smart_router = None

        # Track consecutive fix failures for escalation (S81)
        self._consecutive_fix_failures: int = 0
        self._escalated_model: str | None = None

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def phase(self) -> CodingPhase:
        return self._phase

    @property
    def plan(self) -> CodingPlan | None:
        return self._plan

    @property
    def history(self) -> list[CodingHistoryEntry]:
        return list(self._history)

    @property
    def test_results(self) -> list[TestResult]:
        return list(self._test_results)

    @property
    def diffs(self) -> list[FileDiff]:
        return list(self._diffs)

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def config(self) -> CodingAgentConfig:
        return self._config

    @property
    def session_active(self) -> bool:
        return self._session is not None and self._session.active

    # -----------------------------------------------------------------
    # Progress reporting
    # -----------------------------------------------------------------

    def add_progress_callback(self, callback: Callable) -> None:
        """Register a callback for progress updates.

        Callback signature: callback(event: dict)
        """
        self._progress_callbacks.append(callback)

    def _emit(self, event_type: str, data: dict | None = None) -> None:
        """Emit a progress event to all registered callbacks."""
        event = {
            "type": event_type,
            "phase": self._phase.value,
            "task_id": self._task_id,
            "timestamp": time.time(),
            **(data or {}),
        }
        for cb in self._progress_callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.debug("Progress callback error: %s", exc)

    def _log(
        self, phase: str, action: str, detail: str = "", success: bool = True
    ) -> None:
        """Add a history entry and emit progress."""
        entry = CodingHistoryEntry(
            phase=phase,
            action=action,
            detail=detail[:2000],
            success=success,
        )
        self._history.append(entry)
        self._emit("log", {
            "action": action,
            "detail": detail[:500],
            "success": success,
        })

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def start_task(
        self,
        task: str,
        project_path: str | None = None,
        allow_degraded: bool = False,
    ) -> str:
        """Start a new coding task.

        Creates a sandbox session, optionally injects project files,
        and prepares for planning.

        Args:
            task: Natural language description of the coding task.
            project_path: Optional path to inject into sandbox.
            allow_degraded: Allow tempdir sandbox without confirmation.

        Returns:
            Task ID string.

        Raises:
            RuntimeError: If agent is busy or sandbox unavailable.
        """
        with self._lock:
            if self._phase not in (
                CodingPhase.IDLE,
                CodingPhase.COMPLETED,
                CodingPhase.ABORTED,
                CodingPhase.FAILED,
            ):
                raise RuntimeError(
                    f"Agent busy in phase: {self._phase.value}"
                )

            if self._session is None:
                raise RuntimeError("Sandbox session not available")

            # Reset state
            self._task = task
            self._project_path = project_path
            self._task_id = f"coding-{uuid.uuid4().hex[:12]}"
            self._phase = CodingPhase.IDLE
            self._plan = None
            self._history = []
            self._test_results = []
            self._diffs = []
            self._original_files = {}
            self._current_step = 0
            self._iteration = 0
            self._fix_count = 0
            self._diffs_hash = ""

            # Reset cascading escalation state (S81)
            self._consecutive_fix_failures = 0
            self._escalated_model = None

            # Initialize working memory (S80)
            self._working_memory = WorkingMemory(task_id=self._task_id)

            # Start sandbox session
            if not self._session.active:
                self._session.start(
                    session_id=self._task_id,
                    allow_degraded=allow_degraded,
                )

            self._log("init", "task_started", f"Task: {task[:200]}")

            # Persist task start (S76)
            if self._history_store is not None:
                try:
                    self._history_store.record_task_start(
                        task_id=self._task_id,
                        task_text=task,
                        project_path=project_path or "",
                        model=self._model or "",
                    )
                except Exception as exc:
                    logger.debug("History record_task_start failed: %s", exc)
            self._test_run_counter = 0

            # Initialize session fingerprint (S75)
            if self._fingerprint is not None:
                self._fingerprint.set_task(task)

            # Inject project if provided
            if project_path and os.path.isdir(project_path):
                try:
                    count = self._session.inject_directory(project_path)
                    self._log(
                        "init", "project_injected",
                        f"Injected {count} files from {project_path}",
                    )
                    # Snapshot original files for diff
                    self._snapshot_originals()
                except Exception as exc:
                    self._log(
                        "init", "inject_failed", str(exc), success=False
                    )

            self._emit("task_started", {"task": task[:500]})
            return self._task_id

    def _snapshot_originals(self) -> None:
        """Capture original file contents for diff generation.

        Uses batch tar+base64 read for efficiency (single subprocess
        call instead of one per file). Falls back to per-file reads
        if batch mode fails.
        """
        try:
            files = self._session.extract_files()
            paths = [f.get("path", "") for f in files if f.get("path")]
            if not paths:
                return
            self._original_files = self._batch_read_files(paths)
            self._log(
                "init", "snapshot_originals",
                f"Snapshotted {len(self._original_files)} files (batch)",
            )
        except Exception as exc:
            logger.debug("Snapshot failed: %s", exc)

    def _read_raw(self, path: str) -> str:
        """Read raw file content from sandbox (no line numbers).

        Uses bash cat instead of view to avoid formatted output.
        Path is relative to the workspace root.
        """
        import shlex
        # Strip leading /workspace/ or / if present
        clean = path
        if clean.startswith("/workspace/"):
            clean = clean[len("/workspace/"):]
        elif clean.startswith("/"):
            clean = clean[1:]
        return self._session.bash(f"cat {shlex.quote(clean)}")

    def _batch_read_files(self, paths: list[str]) -> dict[str, str]:
        """Read multiple files from sandbox in a single tar+base64 call.

        Creates a tar archive of the requested files inside the sandbox,
        pipes through base64, decodes in Python, and extracts contents.
        Falls back to per-file _read_raw if tar/base64 is unavailable
        or the batch command fails.

        Args:
            paths: List of file paths (relative to workspace root).

        Returns:
            Dict mapping path -> file content string.
        """
        if not paths:
            return {}

        # Normalize paths: strip /workspace/ prefix
        clean_paths = []
        path_map = {}  # clean -> original
        for p in paths:
            clean = p
            if clean.startswith("/workspace/"):
                clean = clean[len("/workspace/"):]
            elif clean.startswith("/"):
                clean = clean[1:]
            clean_paths.append(clean)
            path_map[clean] = p

        # Attempt batch read via tar + base64
        try:
            return self._batch_read_tar(clean_paths, path_map)
        except Exception as exc:
            logger.debug("Batch tar read failed, falling back to per-file: %s", exc)

        # Fallback: per-file reads
        return self._batch_read_fallback(paths)

    def _batch_read_tar(
        self, clean_paths: list[str], path_map: dict[str, str]
    ) -> dict[str, str]:
        """Internal: batch read via tar archive piped through base64.

        Args:
            clean_paths: Normalized paths (relative to workspace).
            path_map: Mapping from clean path to original path.

        Returns:
            Dict mapping original path -> content string.

        Raises:
            RuntimeError: If tar/base64 command fails.
        """
        import shlex

        # Build tar command with all files
        # Use tar -cf - to stdout, pipe to base64 for safe transport
        quoted = " ".join(shlex.quote(p) for p in clean_paths)
        cmd = f"tar -cf - {quoted} 2>/dev/null | base64 -w 0"

        raw_b64 = self._session.bash(cmd)
        if not raw_b64 or not raw_b64.strip():
            raise RuntimeError("Empty tar+base64 output from sandbox")

        # Decode base64 to bytes
        tar_bytes = base64.b64decode(raw_b64.strip())

        # Extract files from tar archive
        result: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                content_bytes = extracted.read()
                # Map tar member name back to original path
                member_name = member.name
                # tar may include ./ prefix
                if member_name.startswith("./"):
                    member_name = member_name[2:]
                original_path = path_map.get(member_name, member_name)
                try:
                    result[original_path] = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    # Skip binary files
                    logger.debug(
                        "Skipping binary file in batch read: %s", member_name
                    )

        return result

    def _batch_read_fallback(self, paths: list[str]) -> dict[str, str]:
        """Internal: per-file fallback when batch tar read fails.

        Args:
            paths: File paths (original format).

        Returns:
            Dict mapping path -> content string (skipping failures).
        """
        result: dict[str, str] = {}
        for path in paths:
            try:
                content = self._read_raw(path)
                # Detect sandbox command failure markers
                # Format: "STDERR:\n...\n\n[exit code: N]"
                if content and "[exit code:" in content:
                    # Non-zero exit code indicates failure
                    idx = content.rfind("[exit code: ")
                    if idx >= 0:
                        code_str = content[idx + 12:].split("]")[0].strip()
                        if code_str != "0":
                            continue
                result[path] = content
            except Exception:
                pass
        return result

    # -----------------------------------------------------------------
    # Phase 1: Planning
    # -----------------------------------------------------------------

    def generate_plan(self) -> CodingPlan:
        """Generate a coding plan from the task description.

        Sends the task to the LLM with the planning system prompt.
        Parses the structured response into a CodingPlan.

        S80: Retries with reinforced prompts on JSON parse failure.
        Falls back to numbered list parsing after max_plan_retries.

        Returns:
            The generated CodingPlan.

        Raises:
            RuntimeError: If no LLM callable or task not started.
        """
        self._phase = CodingPhase.PLANNING
        self._emit("phase_change", {"phase": "planning"})

        if not self._task:
            raise RuntimeError("No task set. Call start_task() first.")

        # Build context: list workspace files if project was injected
        workspace_context = ""
        if self._original_files:
            file_list = "\n".join(
                f"  - {p}" for p in sorted(self._original_files.keys())
            )
            workspace_context = (
                f"\n\nExisting files in workspace:\n{file_list}"
            )

        base_prompt = (
            f"Task: {self._task}"
            f"{workspace_context}"
            f"\n\nGenerate a step-by-step plan to complete this task."
        )

        if self._llm_call is not None:
            self._plan = self._generate_plan_with_retries(base_prompt)
        else:
            # No LLM available - create empty plan for manual steps
            self._plan = CodingPlan(
                task=self._task,
                summary="No LLM available. Manual plan required.",
            )

        self._log(
            "planning", "plan_generated",
            f"{self._plan.total_steps} steps planned",
        )
        self._emit("plan_ready", {"plan": self._plan.to_dict()})

        # Persist plan (S76)
        if self._history_store is not None and self._plan:
            try:
                self._history_store.update_task_status(
                    self._task_id,
                    "planning",
                    plan_json=self._plan.to_dict(),
                )
            except Exception as exc:
                logger.debug("History update_task_status failed: %s", exc)

        # Update fingerprint with plan size (S75)
        if self._fingerprint is not None and self._plan:
            self._fingerprint.set_task(self._task, self._plan.total_steps)

        return self._plan

    def _generate_plan_with_retries(self, base_prompt: str) -> CodingPlan:
        """Generate plan with retry on JSON parse failure.

        Retry strategy (S80):
        1. First attempt: standard prompt
        2. Retries 1..N-1: append JSON_RETRY_SUFFIX to prompt
        3. Final attempt: request numbered list via SIMPLIFIED_PLAN_SUFFIX

        Args:
            base_prompt: The base planning prompt.

        Returns:
            CodingPlan (may be minimal fallback on total failure).
        """
        max_retries = self._config.max_plan_retries
        last_error = None

        for attempt in range(max_retries + 1):
            # Build prompt with progressive reinforcement
            if attempt == 0:
                prompt = base_prompt
            elif attempt < max_retries:
                # Retry with JSON reinforcement
                suffix = JSON_RETRY_SUFFIX if JSON_REPAIR_AVAILABLE else ""
                prompt = base_prompt + suffix
            else:
                # Final attempt: request numbered list
                suffix = SIMPLIFIED_PLAN_SUFFIX if JSON_REPAIR_AVAILABLE else ""
                prompt = base_prompt + suffix

            try:
                response_text = self._llm_call(
                    prompt,
                    system=_PLANNING_SYSTEM_PROMPT,
                    model=self._planning_model or self._model,
                )

                # Try JSON parsing first
                try:
                    response_data = _parse_json_response(response_text)
                    return _build_plan_from_response(
                        self._task, response_data
                    )
                except ValueError as parse_exc:
                    last_error = parse_exc

                    # On final attempt, try numbered list fallback
                    if (
                        attempt >= max_retries
                        and JSON_REPAIR_AVAILABLE
                        and repair_json_or_list is not None
                    ):
                        _, list_steps = repair_json_or_list(response_text)
                        if list_steps:
                            self._log(
                                "planning", "plan_from_numbered_list",
                                f"Parsed {len(list_steps)} steps from "
                                f"numbered list fallback",
                            )
                            return _build_plan_from_response(
                                self._task, {"steps": list_steps}
                            )

                    if attempt < max_retries:
                        self._log(
                            "planning", "plan_parse_retry",
                            f"Attempt {attempt + 1}/{max_retries + 1}: "
                            f"{parse_exc}",
                            success=False,
                        )
                        self._emit("plan_retry", {
                            "attempt": attempt + 1,
                            "max_retries": max_retries + 1,
                            "error": str(parse_exc)[:500],
                        })
                        continue

                    raise

            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    self._log(
                        "planning", "plan_generation_retry",
                        f"Attempt {attempt + 1}: {exc}",
                        success=False,
                    )
                    continue

                self._log(
                    "planning", "plan_generation_failed",
                    str(exc), success=False,
                )

        # All retries exhausted - return minimal fallback plan
        return CodingPlan(
            task=self._task,
            summary=f"Planning failed after {max_retries + 1} attempts: "
                    f"{last_error}",
        )

    def set_plan(self, plan: CodingPlan) -> None:
        """Set or replace the current plan (for human modification).

        Args:
            plan: The modified plan to use.
        """
        self._plan = plan
        self._current_step = 0
        self._log("planning", "plan_modified", f"{plan.total_steps} steps")

    # -----------------------------------------------------------------
    # Phase 2: Implementation
    # -----------------------------------------------------------------

    # -- Error classification (S79) --

    # Patterns indicating transient sandbox errors that may succeed on retry
    _TRANSIENT_PATTERNS = (
        "timeout",
        "timed out",
        "bwrap",
        "bubblewrap",
        "connection refused",
        "resource temporarily unavailable",
        "no space left on device",
        "broken pipe",
        "errno 11",
        "errno 110",
        "sandbox startup",
        "failed to start sandbox",
    )

    @staticmethod
    def is_transient_error(error_msg: str) -> bool:
        """Classify an error as transient (retryable) or permanent.

        Transient errors are sandbox-level failures that may succeed
        on retry: timeouts, bwrap startup failures, I/O interrupts.

        Permanent errors are logical failures: syntax errors, missing
        files, permission denied, import errors, etc.

        Args:
            error_msg: The error message string.

        Returns:
            True if the error is likely transient.
        """
        if not error_msg:
            return False
        lower = error_msg.lower()
        return any(p in lower for p in CodingAgent._TRANSIENT_PATTERNS)

    def _execute_step_with_retry(self, step: PlanStep) -> str:
        """Execute a step with auto-retry on transient errors.

        Uses exponential backoff from config. Emits retry progress
        events for the frontend.

        Args:
            step: The plan step to execute.

        Returns:
            Result string from successful execution.

        Raises:
            Exception: Re-raises the last error if all retries fail,
                or immediately on permanent errors.
        """
        max_retries = self._config.max_auto_retries
        backoff = self._config.retry_backoff_seconds

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self._execute_step(step)
            except Exception as exc:
                last_error = exc
                error_msg = str(exc)

                # Permanent error: fail immediately
                if not self.is_transient_error(error_msg):
                    raise

                # Last attempt: no more retries
                if attempt >= max_retries:
                    raise

                # Transient error: retry with backoff
                delay = (
                    backoff[attempt]
                    if attempt < len(backoff)
                    else backoff[-1] if backoff else 1.0
                )
                logger.warning(
                    "Transient error on step %d (attempt %d/%d): %s. "
                    "Retrying in %.1fs...",
                    step.step_number, attempt + 1, max_retries + 1,
                    error_msg[:200], delay,
                )
                self._emit("retry", {
                    "step": step.step_number,
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "error": error_msg[:500],
                    "delay": delay,
                })
                time.sleep(delay)

        # Should not reach here, but safety net
        raise last_error  # type: ignore[misc]

    def execute_next_step(self) -> PlanStep | None:
        """Execute the next step in the plan.

        Returns:
            The executed PlanStep, or None if no more steps.

        Raises:
            RuntimeError: If no plan or sandbox not active.
        """
        if self._plan is None:
            raise RuntimeError("No plan available")
        if not self.session_active:
            raise RuntimeError("Sandbox session not active")

        if self._current_step >= self._plan.total_steps:
            return None

        self._phase = CodingPhase.IMPLEMENTING
        self._iteration += 1

        if self._iteration > self._config.max_iterations:
            self._phase = CodingPhase.FAILED
            self._log(
                "implementing", "max_iterations_reached",
                f"Exceeded {self._config.max_iterations} iterations",
                success=False,
            )
            # Persist failure (S76)
            if self._history_store is not None:
                try:
                    self._history_store.update_task_status(
                        self._task_id, "failed"
                    )
                except Exception as exc:
                    logger.debug("History max_iter persist failed: %s", exc)
            return None

        step = self._plan.steps[self._current_step]
        self._emit("step_start", {
            "step_number": step.step_number,
            "step_type": step.step_type.value,
            "description": step.description,
        })

        try:
            result = self._execute_step_with_retry(step)
            step.completed = True
            step.result = result
            self._log(
                "implementing",
                f"step_{step.step_number}_{step.step_type.value}",
                result[:500],
            )
        except Exception as exc:
            step.error = str(exc)
            self._log(
                "implementing",
                f"step_{step.step_number}_failed",
                str(exc),
                success=False,
            )

        self._current_step += 1
        self._emit("step_complete", {
            "step_number": step.step_number,
            "completed": step.completed,
            "error": step.error,
        })

        # Update session fingerprint (S75)
        if self._fingerprint is not None:
            self._fingerprint.on_step({
                "file_path": step.file_path or "",
                "step_type": step.step_type.value if step.step_type else "",
                "content": step.content or "",
                "size": len(step.content.encode()) if step.content else 0,
                "completed": step.completed,
            })

        # Update working memory (S80)
        if self._working_memory is not None:
            self._working_memory.update_from_step(
                step_number=step.step_number,
                step_type=step.step_type.value if step.step_type else "",
                file_path=step.file_path or "",
                result=step.result or "",
                error=step.error or "",
            )
            self._working_memory.trim()
            # Persist working memory
            if self._history_store is not None:
                try:
                    self._history_store.save_working_memory(
                        task_id=self._task_id,
                        memory_data=self._working_memory.to_dict(),
                    )
                except Exception as exc:
                    logger.debug("Working memory persist failed: %s", exc)

        # Persist step (S76)
        if self._history_store is not None:
            try:
                self._history_store.record_step(
                    task_id=self._task_id,
                    step_number=step.step_number,
                    step_type=step.step_type.value if step.step_type else "",
                    file_path=step.file_path or "",
                    status="completed" if step.completed else "failed",
                    result=step.result[:2000] if step.result else step.error[:2000] if step.error else "",
                )
                self._history_store.update_task_status(
                    self._task_id, "implementing"
                )
            except Exception as exc:
                logger.debug("History record_step failed: %s", exc)

        return step

    def execute_all_steps(
        self, should_stop: Callable[[], bool] | None = None,
    ) -> list[PlanStep]:
        """Execute all remaining steps in the plan.

        Args:
            should_stop: Optional callback that returns True to signal
                graceful stop. Checked before each step execution.

        Returns:
            List of executed steps.
        """
        executed = []
        while self._current_step < (
            self._plan.total_steps if self._plan else 0
        ):
            # Check for graceful stop signal
            if should_stop is not None and should_stop():
                self._log(
                    "implementing", "stopped",
                    f"Graceful stop after {len(executed)} steps",
                )
                self._emit("stopped", {"executed_steps": len(executed)})
                break

            step = self.execute_next_step()
            if step is None:
                break
            executed.append(step)

            # Auto-test after implementation steps
            if (
                self._config.auto_test
                and step.step_type in (PlanStepType.CREATE, PlanStepType.EDIT)
                and step.completed
            ):
                # Check if there are test files
                if self._has_test_files():
                    test_result = self.run_tests()
                    if not test_result.passed:
                        # Enter fix loop
                        fixed = self._fix_loop(test_result)
                        if not fixed:
                            self._log(
                                "implementing",
                                "fix_loop_exhausted",
                                "Max fix retries reached",
                                success=False,
                            )

        return executed

    def _execute_step(self, step: PlanStep) -> str:
        """Execute a single plan step in the sandbox.

        Returns:
            Result string from the sandbox operation.
        """
        if step.step_type == PlanStepType.CREATE:
            return self._session.create_file(step.file_path, step.content)
        elif step.step_type == PlanStepType.EDIT:
            return self._session.str_replace(
                step.file_path, step.old_str, step.new_str
            )
        elif step.step_type == PlanStepType.BASH:
            return self._session.bash(step.command)
        elif step.step_type == PlanStepType.TEST:
            cmd = step.command or self._config.auto_test_command
            return self._session.bash(cmd)
        else:
            return f"Unknown step type: {step.step_type}"

    def _has_test_files(self) -> bool:
        """Check if the sandbox workspace contains test files."""
        try:
            listing = self._session.bash(
                "find /workspace -name 'test_*.py' -o -name '*_test.py' "
                "2>/dev/null | head -5"
            )
            return bool(listing.strip())
        except Exception:
            return False

    # -----------------------------------------------------------------
    # Phase 3: Testing
    # -----------------------------------------------------------------

    def run_tests(self, command: str | None = None) -> TestResult:
        """Run tests in the sandbox.

        Args:
            command: Custom test command. Uses config default if None.

        Returns:
            TestResult with pass/fail status and output.
        """
        self._phase = CodingPhase.TESTING
        cmd = command or self._config.auto_test_command
        self._emit("testing", {"command": cmd})

        result = TestResult()
        try:
            output = self._session.bash(cmd, timeout=60)
            result.output = output
            result.return_code = 0
            # Parse pytest output for pass/fail
            if "passed" in output.lower():
                result.passed = True
            if "failed" in output.lower() or "error" in output.lower():
                result.passed = False
                result.return_code = 1
            if "no tests ran" in output.lower():
                result.passed = True
                result.test_count = 0
        except Exception as exc:
            error_str = str(exc)
            result.error = error_str
            result.passed = False
            result.return_code = 1
            # Check if the error contains test output
            if "FAILED" in error_str or "Error" in error_str:
                result.output = error_str

        self._test_results.append(result)
        self._log(
            "testing",
            "tests_passed" if result.passed else "tests_failed",
            result.output[:500] or result.error[:500],
            success=result.passed,
        )
        self._emit("test_result", {
            "passed": result.passed,
            "output": result.output[:500],
        })

        # Update session fingerprint (S75)
        if self._fingerprint is not None:
            self._fingerprint.on_test({
                "passed": result.passed,
                "output": result.output or "",
                "error": result.error or "",
            })

        # Persist test result (S76)
        if self._history_store is not None:
            try:
                self._test_run_counter += 1
                self._history_store.record_test(
                    task_id=self._task_id,
                    run_number=self._test_run_counter,
                    passed=result.passed,
                    output=(result.output or result.error)[:5000],
                )
            except Exception as exc:
                logger.debug("History record_test failed: %s", exc)

        return result

    # -----------------------------------------------------------------
    # Phase 4: Fix loop
    # -----------------------------------------------------------------

    def _fix_loop(self, test_result: TestResult) -> bool:
        """Attempt to fix test failures using the LLM.

        S81: After escalate_after_failures consecutive failures on the
        current model, auto-escalate to the next cascade tier if
        cascading is enabled.

        Returns:
            True if tests eventually pass, False if retries exhausted.
        """
        self._phase = CodingPhase.FIXING
        self._consecutive_fix_failures = 0

        for attempt in range(self._config.max_fix_retries):
            self._fix_count += 1

            # Determine which model to use (may be escalated)
            current_model = self._escalated_model or self._model

            self._emit("fix_attempt", {
                "attempt": attempt + 1,
                "max_retries": self._config.max_fix_retries,
                "model": current_model or "",
            })

            if self._llm_call is None:
                self._log(
                    "fixing", "no_llm",
                    "Cannot auto-fix without LLM", success=False,
                )
                return False

            # Build fix prompt with failure context
            fix_prompt = self._build_fix_prompt(test_result)

            try:
                response_text = self._llm_call(
                    fix_prompt,
                    system=_FIX_SYSTEM_PROMPT,
                    model=current_model,
                )
                fix_data = _parse_json_response(response_text)
                fix_instructions = _build_fix_from_response(fix_data)

                # Apply the fix
                self._apply_fix(fix_instructions)

                # Re-run tests
                new_result = self.run_tests()
                if new_result.passed:
                    self._log(
                        "fixing", "fix_succeeded",
                        f"Fixed on attempt {attempt + 1}"
                        + (f" (model: {current_model})" if current_model else ""),
                    )
                    self._consecutive_fix_failures = 0
                    return True

                test_result = new_result
                self._consecutive_fix_failures += 1

                # S81: Check if we should escalate to a stronger model
                self._maybe_escalate(attempt)

            except Exception as exc:
                self._log(
                    "fixing", "fix_attempt_failed",
                    str(exc), success=False,
                )
                self._consecutive_fix_failures += 1
                self._maybe_escalate(attempt)

        return False

    def _maybe_escalate(self, current_attempt: int) -> None:
        """Check if fix failures warrant model escalation (S81).

        If cascading is enabled and we have reached
        escalate_after_failures consecutive failures, find the next
        cascade tier and switch to it.

        Args:
            current_attempt: Current fix attempt index (0-based).
        """
        if not self._config.enable_cascading:
            return
        if self._cascading is None:
            return
        if self._consecutive_fix_failures < self._config.escalate_after_failures:
            return

        # Find current tier index
        tiers = self._cascading.tiers
        if not tiers:
            return

        current_model = self._escalated_model or self._model or ""
        current_tier_idx = -1
        for i, tier in enumerate(tiers):
            if tier.model == current_model:
                current_tier_idx = i
                break

        # If current model is not in tiers, start from tier 0
        # and try to escalate to tier 1 (or the first tier that differs)
        next_tier_idx = current_tier_idx + 1
        if next_tier_idx >= len(tiers):
            # Already at highest tier, cannot escalate further
            return

        next_tier = tiers[next_tier_idx]
        previous_model = current_model
        self._escalated_model = next_tier.model
        self._consecutive_fix_failures = 0

        self._log(
            "fixing", "model_escalated",
            f"Escalated from '{previous_model}' to "
            f"'{next_tier.model}' (tier {next_tier.name}) "
            f"after {self._config.escalate_after_failures} failures",
        )
        self._emit("escalated", {
            "from_model": previous_model,
            "to_model": next_tier.model,
            "tier_name": next_tier.name,
            "tier_index": next_tier_idx,
        })

    # Step types considered "simple" for per-step routing (S81)
    _SIMPLE_STEP_TYPES = frozenset({PlanStepType.BASH, PlanStepType.TEST})
    _COMPLEX_STEP_TYPES = frozenset({PlanStepType.CREATE, PlanStepType.EDIT})

    def _get_model_for_step(self, step: PlanStep) -> str | None:
        """Resolve the model to use for a given step (S81 per-step routing).

        When per_step_routing is enabled and a cascading engine is
        available, simple steps (bash, test) are routed to the fastest
        tier, while complex steps (create, edit) use the standard or
        power tier.

        If per_step_routing is disabled or unavailable, returns the
        current model (possibly already escalated).

        Args:
            step: The plan step to route.

        Returns:
            Model name string, or None to use default.
        """
        # If model was escalated in fix loop, honor escalation
        if self._escalated_model:
            return self._escalated_model

        # Per-step routing (experimental, opt-in)
        if (
            self._config.per_step_routing
            and self._cascading is not None
            and self._cascading.tiers
        ):
            tiers = self._cascading.tiers
            if step.step_type in self._SIMPLE_STEP_TYPES and len(tiers) >= 1:
                return tiers[0].model  # fastest tier
            if step.step_type in self._COMPLEX_STEP_TYPES and len(tiers) >= 2:
                return tiers[min(1, len(tiers) - 1)].model  # standard tier

        return self._model

    def _build_fix_prompt(self, test_result: TestResult) -> str:
        """Build a prompt for the LLM to fix test failures."""
        # Get relevant file content
        file_context = ""
        if self._plan:
            for step in self._plan.steps:
                if step.file_path and step.completed:
                    try:
                        content = self._session.view(step.file_path)
                        file_context += (
                            f"\n--- {step.file_path} ---\n"
                            f"{content[:3000]}\n"
                        )
                    except Exception:
                        pass

        # Inject working memory context (S80)
        memory_context = ""
        if self._working_memory is not None:
            compact = self._working_memory.to_compact()
            if compact:
                memory_context = (
                    f"\n\nWorking memory (context from previous steps):\n"
                    f"{compact}\n"
                )

        failure_output = test_result.output or test_result.error
        return (
            f"Test failure output:\n{failure_output[:4000]}"
            f"\n\nRelevant files:{file_context}"
            f"{memory_context}"
            f"\n\nPropose a fix."
        )

    def _apply_fix(self, fix: dict[str, str]) -> None:
        """Apply a fix from the LLM to the sandbox."""
        fix_type = fix.get("fix_type", "str_replace")
        file_path = fix.get("file_path", "")

        if not file_path:
            raise ValueError("Fix missing file_path")

        if fix_type == "create_file":
            content = fix.get("content", "")
            result = self._session.create_file(file_path, content)
            self._log("fixing", "create_file", f"{file_path}: {result[:200]}")
        else:
            old_str = fix.get("old_str", "")
            new_str = fix.get("new_str", "")
            if not old_str:
                raise ValueError("Fix missing old_str for str_replace")
            result = self._session.str_replace(file_path, old_str, new_str)
            self._log(
                "fixing", "str_replace", f"{file_path}: {result[:200]}"
            )

    # -----------------------------------------------------------------
    # Phase 5: Review (diff generation)
    # -----------------------------------------------------------------

    def generate_diffs(self) -> list[FileDiff]:
        """Generate diffs between original and modified files.

        Uses batch tar+base64 read for efficiency when reading
        current sandbox file contents.

        Returns:
            List of FileDiff objects.
        """
        self._phase = CodingPhase.REVIEWING
        self._diffs = []

        if not self.session_active:
            return self._diffs

        try:
            current_files = self._session.extract_files()
        except Exception as exc:
            self._log("reviewing", "extract_failed", str(exc), success=False)
            return self._diffs

        current_paths = {f["path"] for f in current_files}

        # Batch-read all current file contents
        all_paths = [f["path"] for f in current_files if f.get("path")]
        current_contents = self._batch_read_files(all_paths)

        # Modified and new files
        for file_info in current_files:
            path = file_info["path"]
            new_content = current_contents.get(path)
            if new_content is None:
                continue

            if path in self._original_files:
                old_content = self._original_files[path]
                if old_content != new_content:
                    diff_lines = list(difflib.unified_diff(
                        old_content.splitlines(keepends=True),
                        new_content.splitlines(keepends=True),
                        fromfile=f"a/{path}",
                        tofile=f"b/{path}",
                    ))
                    if diff_lines:
                        self._diffs.append(FileDiff(
                            path=path,
                            diff_lines=diff_lines,
                            original_content=old_content,
                            modified_content=new_content,
                        ))
            else:
                # New file
                self._diffs.append(FileDiff(
                    path=path,
                    is_new=True,
                    modified_content=new_content,
                    diff_lines=[f"+++ b/{path}\n"]
                    + [f"+{line}\n" for line in new_content.splitlines()],
                ))

        # Deleted files
        for path in self._original_files:
            if path not in current_paths:
                self._diffs.append(FileDiff(
                    path=path,
                    is_deleted=True,
                    original_content=self._original_files[path],
                    diff_lines=[f"--- a/{path}\n", "+++ /dev/null\n"],
                ))

        self._log(
            "reviewing", "diffs_generated",
            f"{len(self._diffs)} files changed",
        )
        # SECURITY: Store integrity hash of diffs for apply-time verification
        self._diffs_hash = self._compute_diffs_hash()
        self._emit("diffs_ready", {
            "count": len(self._diffs),
            "files": [d.path for d in self._diffs],
        })
        return self._diffs

    def _compute_diffs_hash(self) -> str:
        """Compute a SHA-256 hash of all current diffs for integrity check."""
        import hashlib
        h = hashlib.sha256()
        for d in self._diffs:
            h.update(d.path.encode())
            h.update(d.modified_content.encode() if d.modified_content else b"")
            h.update(b"1" if d.is_new else b"0")
            h.update(b"1" if d.is_deleted else b"0")
        return h.hexdigest()

    # -----------------------------------------------------------------
    # Phase 6: Apply (HUMAN-GATED)
    # -----------------------------------------------------------------

    def apply_changes(self, target_path: str | None = None) -> dict[str, Any]:
        """Apply sandbox changes to the real filesystem.

        SECURITY: This method ALWAYS requires prior human approval.
        The checkpoint_before_apply config is hardcoded to True and
        cannot be overridden.

        SECURITY: target_path is validated against the original
        project_path. If target_path differs from project_path,
        it must not point to sensitive system directories.

        Args:
            target_path: Destination directory. Falls back to
                original project_path.

        Returns:
            Dict with applied file count and details.

        Raises:
            RuntimeError: If no diffs or no target path.
            ValueError: If target_path is unsafe.
        """
        self._phase = CodingPhase.APPLYING

        target = target_path or self._project_path
        if not target:
            raise RuntimeError(
                "No target path for apply. Provide target_path or "
                "set project_path in start_task()."
            )

        # SECURITY: Validate target_path is not a sensitive directory
        _validate_apply_target(target)

        # SECURITY: Verify diff integrity (hash match)
        if self._diffs_hash and self._diffs:
            current_hash = self._compute_diffs_hash()
            if current_hash != self._diffs_hash:
                raise RuntimeError(
                    "Diff integrity check failed: sandbox contents "
                    "changed since review. Re-run /diff to review."
                )

        if not self._diffs:
            self.generate_diffs()

        if not self._diffs:
            self._log("applying", "no_changes", "No changes to apply")
            self._phase = CodingPhase.COMPLETED
            return {"applied": 0, "files": []}

        applied_files = []
        errors = []

        for diff in self._diffs:
            dest = os.path.join(target, diff.path)
            # SECURITY (S184): the apply phase is the ONLY exit from the sandbox.
            # Constrain every write/delete to within ``target``. A diff path that
            # contains ".." (or resolves outside the project via a symlink) must
            # not escape onto the host filesystem. _validate_apply_target only
            # checks ``target`` itself; this guards ``target`` + diff.path.
            if not _is_within_target(target, dest):
                errors.append({
                    "path": diff.path,
                    "error": "refused: path escapes the apply target",
                })
                self._log(
                    "applying", "apply_path_escape",
                    f"{diff.path}: refused (escapes {target})", success=False,
                )
                continue
            try:
                if diff.is_deleted:
                    if os.path.exists(dest):
                        os.remove(dest)
                        applied_files.append(
                            {"path": diff.path, "action": "deleted"}
                        )
                elif diff.is_new or diff.diff_lines:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    content = diff.modified_content
                    with open(dest, "w", encoding="utf-8") as fh:
                        fh.write(content)
                    action = "created" if diff.is_new else "modified"
                    applied_files.append(
                        {"path": diff.path, "action": action}
                    )
            except Exception as exc:
                errors.append({"path": diff.path, "error": str(exc)})
                self._log(
                    "applying", "apply_file_error",
                    f"{diff.path}: {exc}", success=False,
                )

        self._phase = CodingPhase.COMPLETED
        self._log(
            "applying", "changes_applied",
            f"Applied {len(applied_files)} files, {len(errors)} errors",
        )
        self._emit("apply_complete", {
            "applied": len(applied_files),
            "errors": len(errors),
        })

        # Persist completion (S76)
        if self._history_store is not None:
            try:
                self._history_store.record_checkpoint(
                    task_id=self._task_id,
                    phase="applying",
                    decision="apply",
                    current_step=self._current_step,
                    originals_hash=self._diffs_hash,
                )
                self._history_store.update_task_status(
                    self._task_id, "completed"
                )
            except Exception as exc:
                logger.debug("History apply persist failed: %s", exc)

        return {
            "applied": len(applied_files),
            "files": applied_files,
            "errors": errors,
        }

    # -----------------------------------------------------------------
    # Abort and cleanup
    # -----------------------------------------------------------------

    def abort(self) -> bool:
        """Abort the current task and destroy the sandbox.

        Returns:
            True if cleanup succeeded.
        """
        self._phase = CodingPhase.ABORTED
        self._log("abort", "task_aborted", f"Task {self._task_id} aborted")
        self._emit("aborted", {})

        # Persist abort (S76)
        if self._history_store is not None:
            try:
                self._history_store.record_checkpoint(
                    task_id=self._task_id,
                    phase="abort",
                    decision="abort",
                    current_step=self._current_step,
                )
                self._history_store.update_task_status(
                    self._task_id, "aborted"
                )
            except Exception as exc:
                logger.debug("History abort persist failed: %s", exc)

        # Record abort in fingerprint (S75)
        if self._fingerprint is not None:
            self._fingerprint.on_checkpoint({
                "action": "abort",
                "phase": "abort",
            })

        return self._cleanup()

    # -----------------------------------------------------------------
    # Session fingerprint (S75)
    # -----------------------------------------------------------------

    def record_checkpoint(
        self, action: str, phase: str = "", context: str = "",
        anchor: str = "",
    ) -> None:
        """Record a human checkpoint decision in session fingerprint and history.

        Args:
            action: One of 'approve', 'modify', 'abort'.
            phase: The phase when the decision was made.
            context: Additional context string.
            anchor: Optional context anchor to add (D10).
        """
        if self._fingerprint is not None:
            self._fingerprint.on_checkpoint({
                "action": action,
                "phase": phase or self._phase.value,
                "context": context,
                "anchor": anchor,
            })

        # Persist checkpoint (S76)
        if self._history_store is not None:
            try:
                plan_snap = self._plan.to_dict() if self._plan else None
                self._history_store.record_checkpoint(
                    task_id=self._task_id,
                    phase=phase or self._phase.value,
                    decision=action,
                    current_step=self._current_step,
                    originals_hash=self._diffs_hash,
                    plan_snapshot=plan_snap,
                )
            except Exception as exc:
                logger.debug("History record_checkpoint failed: %s", exc)

    @property
    def fingerprint(self) -> "FingerprintManager | None":
        """Access the session fingerprint manager."""
        return self._fingerprint

    def get_fingerprint_compact(self) -> str:
        """Get compact fingerprint string for context injection.

        Returns:
            Compact YAML/JSON string, or empty string if unavailable.
        """
        if self._fingerprint is not None:
            return self._fingerprint.serialize_compact()
        return ""

    @property
    def working_memory(self) -> WorkingMemory | None:
        """Access the current working memory (S80)."""
        return self._working_memory

    @property
    def cascading_engine(self) -> "CascadingInference | None":
        """Access the cascading inference engine (S81)."""
        return self._cascading

    @property
    def escalated_model(self) -> str | None:
        """Current escalated model, or None if not escalated (S81)."""
        return self._escalated_model

    def get_working_memory_compact(self) -> str:
        """Get compact working memory string for context injection.

        Returns:
            Compact text representation, or empty string if unavailable.
        """
        if self._working_memory is not None:
            return self._working_memory.to_compact()
        return ""

    def _cleanup(self) -> bool:
        """Clean up sandbox resources."""
        if self._session is not None and self._owns_session:
            try:
                return self._session.stop()
            except Exception as exc:
                logger.warning("Cleanup failed: %s", exc)
                return False
        return True

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get the current agent status.

        Returns:
            Dict with phase, plan, history, and metrics.
        """
        return {
            "task_id": self._task_id,
            "task": self._task,
            "phase": self._phase.value,
            "session_active": self.session_active,
            "plan": self._plan.to_dict() if self._plan else None,
            "current_step": self._current_step,
            "total_steps": self._plan.total_steps if self._plan else 0,
            "iteration": self._iteration,
            "max_iterations": self._config.max_iterations,
            "fix_count": self._fix_count,
            "max_fix_retries": self._config.max_fix_retries,
            "test_results": [
                {
                    "passed": t.passed,
                    "output": t.output[:500],
                    "error": t.error[:500],
                }
                for t in self._test_results
            ],
            "diffs": [d.to_dict() for d in self._diffs],
            "history_count": len(self._history),
            "history": [
                {
                    "timestamp": h.timestamp,
                    "phase": h.phase,
                    "action": h.action,
                    "detail": h.detail[:300],
                    "success": h.success,
                }
                for h in self._history[-50:]
            ],
            "working_memory": (
                self._working_memory.to_dict()
                if self._working_memory else None
            ),
            "cascading": {
                "enabled": self._config.enable_cascading,
                "available": self._cascading is not None,
                "escalated_model": self._escalated_model,
                "consecutive_fix_failures": self._consecutive_fix_failures,
                "escalate_after_failures": self._config.escalate_after_failures,
                "per_step_routing": self._config.per_step_routing,
            },
        }


# ---------------------------------------------------------------------------
# Module-level availability
# ---------------------------------------------------------------------------

CODING_AGENT_AVAILABLE = SANDBOX_TOOLS_AVAILABLE and SANDBOX_AVAILABLE

# Module-level config (loaded once)
coding_agent_config = _load_config()
