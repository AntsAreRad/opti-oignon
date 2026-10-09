"""
Tool Call Approval -- Opti-Oignon.

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

import json
import logging
import math
import secrets
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field
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
    # Where each argument came from (typed by the user in the turn, the
    # tool's default, or unendorsed), and the call's effect class; empty when
    # the caller did not say.
    labels: dict[str, str] = field(default_factory=dict)
    effect: str = ""
    # Each value's own length in characters and its count of lines, keyed as
    # ``arguments``: a shown value is longer than the value by its escapes.
    sizes: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "approval_id": self.approval_id,
            "conversation_id": self.conversation_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "arguments_summary": self.arguments_summary,
            "risk_level": self.risk_level,
            "labels": dict(self.labels),
            "effect": self.effect,
            "sizes": {name: dict(size) for name, size in self.sizes.items()},
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
    "manage_memory", "manage_skills", "publish_skill",
})

_MEDIUM_RISK_TOOLS = frozenset({
    "file_read", "file_list", "database_query",
    "rag_search", "memory_write",
    # Actual read / inspect names.
    "view", "read_file", "list_files",
})


def assess_risk(tool_name: str) -> str:
    """Assess the risk level of a tool call.

    Returns 'low', 'medium', or 'high'. A label of the form ``tool:action``
    is judged by its tool: the action narrows what the tool does, never what
    it can reach.
    """
    name_lower = tool_name.lower().partition(":")[0]
    if name_lower in _HIGH_RISK_TOOLS:
        return "high"
    if name_lower in _MEDIUM_RISK_TOOLS:
        return "medium"
    return "low"


# The longest value, in characters, the approval surfaces show whole: the
# drawer, the socket event and the terminal. A longer value is cut there and
# says how long it is. The provenance gate refuses to ask about a network
# value longer than this (``provenance.SHOWN_LIMIT``, held equal by a
# contract): the person could not see all that would leave.
SHOWN_CHARS = 2000

# Characters a screen does not show as themselves: controls (a line break
# excepted), formats such as a direction override or a zero-width space, line
# and paragraph separators, private and unassigned code points; every blank but
# the plain space (a tab, a no-break space, an em space draw like spaces of
# other widths); and every code point Unicode says may be ignored when drawn
# (Default_Ignorable_Code_Point: variation selectors, tag characters, the Hangul
# fillers), with the symbols drawn blank (the Braille pattern, the Khitan
# filler, the null notehead). Each one could carry bits the person never sees.
_HIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp", "Co", "Cs", "Cn"})
_IGNORABLE = ((0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5),
              (0x180B, 0x180F), (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x2800, 0x2800),
              (0x3164, 0x3164), (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF), (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8),
              (0x16FE4, 0x16FE4), (0x1BCA0, 0x1BCA3), (0x1D159, 0x1D159), (0x1D173, 0x1D17A), (0xE0000, 0xE0FFF))


def _hidden(ch: str) -> bool:
    if ch == "\n":
        return False
    category = unicodedata.category(ch)
    if category in _HIDDEN_CATEGORIES or (category == "Zs" and ch != " "):
        return True
    point = ord(ch)
    return any(low <= point <= high for low, high in _IGNORABLE)


def _visible(text: str) -> str:
    """``text`` with every character a screen would hide written as its escape.

    A backslash is doubled, so an escape the text itself contains never
    reads as a hidden character, and what is shown determines the bytes.
    """
    return "".join("\\\\" if ch == "\\" else ascii(ch)[1:-1] if _hidden(ch) else ch for ch in text)


def _name(key: Any) -> str:
    """An argument's name as every surface shows it, and keys its value, label and size by: on one line, nothing
    hidden."""
    return _visible(str(key)).replace("\n", "\\n")


def _prefix(text: str, limit: int) -> str:
    """The longest start of a shown ``text`` within ``limit`` characters that never ends inside an escape.

    Every escape the surfaces write starts with a backslash, and the letter
    after it gives its length: ``x`` four, ``u`` six, ``U`` ten, any other
    two (a doubled backslash, a quote, a JSON ``n``).
    """
    end = 0
    while end < len(text):
        step = 1 if text[end] != "\\" else {"x": 4, "u": 6, "U": 10}.get(text[end + 1:end + 2], 2)
        if end + step > limit:
            break
        end += step
    return text[:end]


def _shown(value: Any) -> Any:
    """A value as a person is shown it before deciding.

    Text whole up to ``SHOWN_CHARS``, every hidden character written as its
    escape and a backslash doubled; a longer text cut there, saying how long
    it is; numbers and booleans as they are, but a number JSON cannot carry
    (NaN, an infinity), a browser would print otherwise (a negative zero) or
    cannot read exactly (a whole number past 2**53) as its text, under the
    same bound; anything else as its JSON with every character past ASCII
    escaped, read by JSON's own rule (never escaped a second time, so it
    reads one way only), under the same bound on what is shown. A cut value
    says the length ``argument_sizes`` says, and, for JSON, how long it is
    as shown.
    """
    if isinstance(value, float) and (not math.isfinite(value) or (value == 0 and math.copysign(1.0, value) < 0)):
        return json.dumps(value)
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2 ** 53:
        digits = str(value)
        if len(digits) > SHOWN_CHARS:
            return digits[:SHOWN_CHARS] + f"... [{len(digits)} characters in all; the rest is not shown]"
        return digits
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > SHOWN_CHARS:
            return _visible(value[:SHOWN_CHARS]) + f"... [{len(value)} characters in all; the rest is not shown]"
        return _visible(value)
    text = json.dumps(value, ensure_ascii=True, default=str)
    if len(text) > SHOWN_CHARS:
        whole = len(json.dumps(value, ensure_ascii=False, default=str))
        return (_prefix(text, SHOWN_CHARS)
                + f"... [{whole} characters in all, {len(text)} as shown; the rest is not shown]")
    return text


def sanitize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """The arguments as the approval surfaces show them: each name and value whole up to its bound, nothing hidden."""
    return {_name(key): _shown(value) for key, value in arguments.items()}


def argument_sizes(arguments: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Each value's own length in characters and its count of lines, keyed as the shown arguments are.

    A text is counted as it is; anything else as its JSON. The surfaces say
    these, so neither an escape nor a cut changes what the person is told.
    """
    sizes = {}
    for key, value in arguments.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        sizes[_name(key)] = {"chars": len(text), "lines": text.count("\n") + 1}
    return sizes


def summarize_arguments(tool_name: str, arguments: dict[str, Any]) -> str:
    """A one-line summary of the arguments, read as the values themselves are shown, a line break as its escape."""
    parts = []
    for key, value in arguments.items():
        val_str = str(_shown(value)).replace("\n", "\\n")
        if len(val_str) > 60:
            val_str = _prefix(val_str, 60) + "..."
        parts.append(f"{_name(key)}={val_str}")
    summary = ", ".join(parts)
    if len(summary) > 200:
        summary = _prefix(summary, 200) + "..."
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
        *,
        labels: dict[str, str] | None = None,
        effect: str | None = None,
    ) -> tuple[str, threading.Event]:
        """Submit a tool call for approval.

        Returns (approval_id, event). The caller should wait on the
        event with a timeout. After the event fires, check get_status()
        to determine if the call was approved. ``labels`` (where each
        argument came from) and ``effect`` (the call's class) are shown
        with the request when the caller gives them.
        """
        approval_id = secrets.token_urlsafe(12)
        sanitized = sanitize_arguments(arguments)
        sizes = argument_sizes(arguments)
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
            labels={_name(k): str(v) for k, v in (labels or {}).items()},
            effect=str(effect or ""),
            sizes=sizes,
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

    def withdraw(self, approval_id: str, reason: str = "turn_stopped") -> bool:
        """Withdraw a pending tool call on behalf of its own stopped turn.

        The request is denied and leaves the queue at once, and its waiter
        is released. This resolves on behalf of the request's own turn,
        never of a person: ``reason`` is recorded as the resolver in both
        audits, as ``clear_all`` already does. Returns False when the
        request is no longer pending.
        """
        with self._lock:
            return self._resolve(approval_id, ApprovalStatus.DENIED, reason)

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

        # Forward to hash-chain signed audit log
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
