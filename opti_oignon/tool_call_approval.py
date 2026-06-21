"""
Tool Call Approval — Opti-Oignon S128.

In Bulbe mode, every LLM tool call requires explicit human approval
before execution. This module implements a thread-safe approval queue
with a 30-second auto-deny timeout (fail-secure).

Architecture:
- Tool executor thread submits a pending approval and blocks on Event.
- Frontend polls /api/security/tool-approval/pending for new items.
- User clicks Allow/Deny, which sets the Event and unblocks the thread.
- If no response within 30s, the call is automatically denied.

All decisions are audit-logged.
"""

import logging
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS = 30
MAX_PENDING_ITEMS = 50
MAX_AUDIT_LOG_SIZE = 500


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ApprovalStatus(str, Enum):
    """Status of a tool call approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"


@dataclass
class ApprovalRequest:
    """A single tool call awaiting human approval."""
    approval_id: str
    conversation_id: str
    tool_name: str
    arguments: dict[str, Any]
    arguments_summary: str
    risk_level: str  # "low", "medium", "high"
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = 0.0
    resolved_at: float = 0.0
    resolved_by: str = ""  # "user" or "timeout"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "approval_id": self.approval_id,
            "conversation_id": self.conversation_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "arguments_summary": self.arguments_summary,
            "risk_level": self.risk_level,
            "status": self.status.value,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at if self.resolved_at else None,
            "resolved_by": self.resolved_by or None,
            "timeout_remaining": self._timeout_remaining(),
        }

    def _timeout_remaining(self) -> float:
        """Seconds remaining before auto-deny."""
        if self.status != ApprovalStatus.PENDING:
            return 0.0
        elapsed = time.time() - self.created_at
        return max(0.0, DEFAULT_TIMEOUT_SECONDS - elapsed)


@dataclass
class AuditEntry:
    """Audit log entry for a tool call approval decision."""
    approval_id: str
    tool_name: str
    status: str
    resolved_by: str
    timestamp: float
    conversation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------

# Tools that modify state or access external resources are higher risk. The
# risk level is advisory metadata shown in the approval UI; it does not gate the
# decision (approval always requires an explicit human Allow). The sets list
# both the generic names and the actual tool names in use (the Odysseus
# agent/tools set and the legacy tool_registry), so a real tool is not
# mislabelled "low" in the UI.
_HIGH_RISK_TOOLS = frozenset({
    # Generic names.
    "web_search", "web_fetch", "http_request",
    "file_write", "file_delete", "shell_exec",
    "code_execute", "sandbox_exec",
    # Actual names: arbitrary execution, host/sandbox writes, state mutation.
    "bash", "sandbox_bash", "execute_code",
    "create_file", "write_file", "str_replace",
    "manage_memory", "manage_skills",
})

_MEDIUM_RISK_TOOLS = frozenset({
    "file_read", "file_list", "database_query",
    "rag_search", "memory_write",
    # Actual read / inspect names.
    "view", "read_file", "list_files",
})


def assess_risk(tool_name: str) -> str:
    """Assess the risk level of a tool call.

    Returns 'low', 'medium', or 'high'.
    """
    name_lower = tool_name.lower()
    if name_lower in _HIGH_RISK_TOOLS:
        return "high"
    if name_lower in _MEDIUM_RISK_TOOLS:
        return "medium"
    return "low"


def sanitize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Sanitize tool call arguments for display.

    Truncates long strings to prevent UI overflow and redacts
    potentially sensitive values.
    """
    sanitized: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            if len(value) > 200:
                sanitized[key] = value[:200] + "..."
            else:
                sanitized[key] = value
        elif isinstance(value, (int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = f"[list of {len(value)} items]"
        elif isinstance(value, dict):
            sanitized[key] = f"{{dict with {len(value)} keys}}"
        else:
            sanitized[key] = str(value)[:100]
    return sanitized


def summarize_arguments(tool_name: str, arguments: dict[str, Any]) -> str:
    """Generate a human-readable summary of tool call arguments."""
    parts = []
    for key, value in arguments.items():
        val_str = str(value)
        if len(val_str) > 60:
            val_str = val_str[:60] + "..."
        parts.append(f"{key}={val_str}")
    summary = ", ".join(parts)
    if len(summary) > 200:
        summary = summary[:200] + "..."
    return summary


# ---------------------------------------------------------------------------
# ToolCallApprovalManager
# ---------------------------------------------------------------------------

class ToolCallApprovalManager:
    """Thread-safe manager for tool call approval in Bulbe mode.

    Lifecycle of a tool call:
    1. submit() is called from the tool executor thread, returns an Event.
    2. The tool executor thread waits on the Event (with timeout).
    3. Frontend polls pending() and calls approve() or deny().
    4. On resolution (or timeout), the Event is set and the thread unblocks.
    5. The tool executor checks the status to proceed or skip.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, ApprovalRequest] = {}
        self._events: dict[str, threading.Event] = {}
        self._audit_log: list[AuditEntry] = []
        # Background reaper for expired requests
        self._reaper_active = False

    # -- Submission (called from tool executor thread) ----------------------

    def submit(
        self,
        conversation_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, threading.Event]:
        """Submit a tool call for approval.

        Returns (approval_id, event). The caller should wait on the
        event with a timeout. After the event fires, check get_status()
        to determine if the call was approved.
        """
        approval_id = secrets.token_urlsafe(12)
        sanitized = sanitize_arguments(arguments)
        summary = summarize_arguments(tool_name, arguments)
        risk = assess_risk(tool_name)

        request = ApprovalRequest(
            approval_id=approval_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            arguments=sanitized,
            arguments_summary=summary,
            risk_level=risk,
            created_at=time.time(),
        )

        event = threading.Event()

        with self._lock:
            # Enforce max pending limit
            if len(self._pending) >= MAX_PENDING_ITEMS:
                # Remove oldest
                oldest_id = min(
                    self._pending,
                    key=lambda k: self._pending[k].created_at,
                )
                self._resolve(oldest_id, ApprovalStatus.DENIED, "overflow")

            self._pending[approval_id] = request
            self._events[approval_id] = event

        self._ensure_reaper()

        logger.info(
            "Tool call approval submitted: %s (%s) [%s risk] for conv=%s",
            tool_name, approval_id, risk, conversation_id[:8] if conversation_id else "?",
        )
        return approval_id, event

    # -- Resolution (called from API route) ---------------------------------

    def approve(self, approval_id: str, user_id: str = "admin") -> bool:
        """Approve a pending tool call. Returns True if found and approved."""
        with self._lock:
            return self._resolve(approval_id, ApprovalStatus.APPROVED, user_id)

    def deny(self, approval_id: str, user_id: str = "admin") -> bool:
        """Deny a pending tool call. Returns True if found and denied."""
        with self._lock:
            return self._resolve(approval_id, ApprovalStatus.DENIED, user_id)

    def _resolve(
        self,
        approval_id: str,
        status: ApprovalStatus,
        resolved_by: str,
    ) -> bool:
        """Resolve a pending approval (must hold self._lock)."""
        request = self._pending.pop(approval_id, None)
        if not request:
            return False

        request.status = status
        request.resolved_at = time.time()
        request.resolved_by = resolved_by

        # Audit log
        entry = AuditEntry(
            approval_id=approval_id,
            tool_name=request.tool_name,
            status=status.value,
            resolved_by=resolved_by,
            timestamp=request.resolved_at,
            conversation_id=request.conversation_id,
        )
        self._audit_log.append(entry)
        if len(self._audit_log) > MAX_AUDIT_LOG_SIZE:
            self._audit_log = self._audit_log[-MAX_AUDIT_LOG_SIZE:]

        # S130: Forward to hash-chain signed audit log
        try:
            from opti_oignon.signed_audit_log import chain_log
            chain_log(
                event_type=f"tool_call_{status.value}",
                source="tool_call_approval",
                action=f"{request.tool_name} {status.value}",
                severity="WARNING" if status.value == "denied" else "INFO",
                approval_id=approval_id,
                tool_name=request.tool_name,
                resolved_by=resolved_by,
                conversation_id=request.conversation_id,
            )
        except Exception:
            pass

        # Unblock the waiting thread
        event = self._events.pop(approval_id, None)
        if event:
            event.set()

        logger.info(
            "Tool call %s: %s (%s) by %s",
            status.value, request.tool_name, approval_id, resolved_by,
        )
        return True

    # -- Query --------------------------------------------------------------

    def get_status(self, approval_id: str) -> ApprovalStatus | None:
        """Get the status of an approval request.

        Checks pending dict first, then audit log.
        """
        with self._lock:
            if approval_id in self._pending:
                return self._pending[approval_id].status
            for entry in reversed(self._audit_log):
                if entry.approval_id == approval_id:
                    return ApprovalStatus(entry.status)
        return None

    def pending(self) -> list[dict[str, Any]]:
        """Return all pending approval requests."""
        with self._lock:
            return [r.to_dict() for r in self._pending.values()]

    def audit_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent audit log entries."""
        with self._lock:
            entries = self._audit_log[-limit:]
            return [e.to_dict() for e in reversed(entries)]

    def pending_count(self) -> int:
        """Return number of pending approvals."""
        with self._lock:
            return len(self._pending)

    # -- Timeout reaper -----------------------------------------------------

    def _ensure_reaper(self) -> None:
        """Start the background reaper thread if not already running."""
        if self._reaper_active:
            return
        self._reaper_active = True
        t = threading.Thread(target=self._reaper_loop, daemon=True)
        t.start()

    def _reaper_loop(self) -> None:
        """Background loop that auto-denies expired requests."""
        try:
            while True:
                time.sleep(1.0)
                now = time.time()
                expired_ids = []

                with self._lock:
                    if not self._pending:
                        self._reaper_active = False
                        return
                    for aid, req in self._pending.items():
                        if (now - req.created_at) >= DEFAULT_TIMEOUT_SECONDS:
                            expired_ids.append(aid)

                # Resolve expired outside the iteration
                for aid in expired_ids:
                    with self._lock:
                        self._resolve(aid, ApprovalStatus.TIMEOUT, "timeout")
                    logger.warning("Tool call auto-denied (timeout): %s", aid)
        except Exception as exc:
            logger.error("Reaper thread error: %s", exc)
            self._reaper_active = False

    # -- Cleanup ------------------------------------------------------------

    def clear_all(self) -> int:
        """Deny and remove all pending requests. Returns count cleared."""
        with self._lock:
            ids = list(self._pending.keys())
            for aid in ids:
                self._resolve(aid, ApprovalStatus.DENIED, "clear_all")
            return len(ids)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

tool_call_approval = ToolCallApprovalManager()
