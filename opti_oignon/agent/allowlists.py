#!/usr/bin/env python3
"""Per-mode tool gating for the agent loop (S175, Theme 3 / Odysseus Core).

Tool availability is gated per security context by Daily and Bulbe through
``frozenset`` allowlists (ODYSSEUS_SPEC.md Section 5.4). Two rules:

- The dispatch consults the allowlist for the active mode (from
  ``security_mode``) before any tool runs. A tool that is not in the active
  mode's allowlist is refused.
- In Bulbe every allowed tool call additionally passes through the existing
  ``tool_call_approval`` human gate, fail-secure: anything other than an
  explicit human approval (timeout, denial, no gate available, or any error)
  denies the call. Daily reviews at the copy-out gate instead, so it does not
  require per-call approval.

Bulbe is structurally tighter than Daily: it is derived from the Daily set by
removing the network tool and the persistent-state mutation tools, so the
subset relation cannot drift. What remains is the sandboxed filesystem /
shell / code tool set, plus (S228) the pure session-state tool ``todo`` and
the bounded subagent tool ``task``. In Bulbe the agent may inspect and edit
inside the disposable sandbox under per-call human approval, but may not reach
the network or mutate persistent memory or skills autonomously.

Importlib-isolatable: ``security_mode`` and ``tool_call_approval`` are imported
lazily and guarded, so this module loads with no backend dependency and the
gating primitives are exercised with an injected mode and an injected approval
function in the runtime tests. There is no module-level singleton to reset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Module conventions (Theme 3).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Canonical security-mode values, mirrored locally so this module is
# self-sufficient when ``security_mode`` cannot be imported (the values are a
# stable contract). ``security_mode`` itself fails secure to Bulbe, and so do
# we: an unknown or unresolved mode is treated as Bulbe.
MODE_DAILY = "daily"
MODE_BULBE = "bulbe"
VALID_MODES = (MODE_DAILY, MODE_BULBE)

# The sandboxed filesystem / shell / code tools, the ones that MUST dispatch
# through the S73/S74 disposable bwrap sandbox (consumed by dispatch.py).
# S228 (AGT_SPEC 5.1/5.5): grep, glob and ls join as read-only workspace
# tools on the view precedent (trusted host-side reads, path-confined,
# active session required); they ride the same dispatch seam and the same
# per-call Bulbe approvals as the original four.
SANDBOX_TOOL_NAMES = frozenset(
    {"bash", "view", "create_file", "str_replace", "grep", "glob", "ls"}
)

# Tools that reach the network (data-exfiltration surface).
NETWORK_TOOLS = frozenset({"web_search"})

# Tools that mutate persistent state the agent must not change autonomously in
# maximum-security mode (memory, skill, and note poisoning surfaces). manage_notes
# (N.4) joins here, so it is auto in Daily and forbidden in Bulbe by the structural
# BULBE_ALLOWLIST derivation below -- no new gating path is introduced.
STATE_MUTATION_TOOLS = frozenset(
    {"manage_memory", "manage_skills", "manage_notes"}
)

# Pure per-run session-state tools (S228, AGT_SPEC 5.3). They mutate nothing
# outside the run, touch no filesystem and no network, so they are present in
# BOTH modes and exempt from the per-call Bulbe approval ceremony (the 5.6
# mode-posture table: the ceremony gates actions with consequences).
SESSION_STATE_TOOLS = frozenset({"todo"})

# The bounded subagent tool (S228, AGT_SPEC 5.4): loop-managed, present in
# both modes. The call itself launches nothing ungated -- the child registry
# is the real gate and every child sandbox call inherits the Bulbe per-call
# approval -- so the launch is exempt from the per-call approval like the
# session-state tools.
SUBAGENT_TOOLS = frozenset({"task"})

# Daily: frictionless. The sandboxed tools plus the network and state tools;
# the sandbox plus the copy-out review carry the safety. The session-state
# and subagent tools (S228) join the base set and survive the Bulbe
# derivation by construction.
DAILY_ALLOWLIST = frozenset(
    SANDBOX_TOOL_NAMES
    | NETWORK_TOOLS
    | STATE_MUTATION_TOOLS
    | SESSION_STATE_TOOLS
    | SUBAGENT_TOOLS
)

# Bulbe: derived from Daily by removing the network and state-mutation tools,
# so the subset relation is structural. What remains is the sandboxed tool
# set, each call additionally human-approved, plus the session-state and
# subagent tools (S228), whose launches carry no approval of their own (todo
# mutates nothing outside the run; a task launch is bounded by the child
# registry, with every child sandbox call human-approved).
BULBE_ALLOWLIST = frozenset(DAILY_ALLOWLIST - NETWORK_TOOLS - STATE_MUTATION_TOOLS)

# Default wait for the Bulbe human gate, mirroring the approval manager's own
# fail-secure auto-deny timeout.
APPROVAL_TIMEOUT = 30.0

# Reasons surfaced by ``evaluate`` (kept as plain strings for observations).
REASON_ALLOWED = "allowed"
REASON_NOT_ALLOWED = "not_in_allowlist"
REASON_DENIED = "denied_by_human"


@dataclass
class GateDecision:
    """The outcome of gating one tool call for the active mode."""

    allowed: bool
    reason: str
    mode: str


# Mode resolution


def _import_security_mode():
    """Lazily import ``security_mode``; return the module or None.

    Guarded so this module stays importlib-isolatable. Python caches the
    submodule import, so repeated calls are cheap.
    """
    try:
        from opti_oignon import security_mode as sm

        return sm
    except Exception:  # pragma: no cover - defensive guard
        return None


def _resolve_mode(sm: Any) -> str:
    """Pure mode resolution from a (possibly None) security_mode module.

    Fail-secure: a missing module, a failing call, or an unrecognised value
    all resolve to Bulbe.
    """
    if sm is None:
        return MODE_BULBE
    try:
        mode = sm.get_current_mode()
    except Exception:
        return MODE_BULBE
    return mode if mode in VALID_MODES else MODE_BULBE


def current_mode() -> str:
    """The active security mode, fail-secure to Bulbe when unresolved."""
    return _resolve_mode(_import_security_mode())


def _resolve_arg_mode(mode: str | None) -> str:
    """Resolve a mode argument to a valid mode.

    ``None`` means no explicit mode was given, so resolve from the live system
    (``current_mode``, itself fail-secure to Bulbe). An explicit but invalid
    value is a caller error or corruption and is treated as Bulbe, fail-secure.
    """
    if mode is None:
        return current_mode()
    return mode if mode in VALID_MODES else MODE_BULBE


# Allowlist gate


def allowlist_for(mode: str | None) -> frozenset[str]:
    """The tool allowlist for a mode. Unknown modes get the tighter Bulbe set."""
    if mode == MODE_DAILY:
        return DAILY_ALLOWLIST
    return BULBE_ALLOWLIST


def is_tool_allowed(tool_name: str, mode: str | None = None) -> bool:
    """Whether ``tool_name`` is in the active mode's allowlist.

    ``mode`` is resolved from ``security_mode`` when not given; an invalid
    explicit mode is treated as Bulbe.
    """
    return tool_name in allowlist_for(_resolve_arg_mode(mode))


def requires_approval(mode: str | None = None) -> bool:
    """Whether per-call human approval is required (Bulbe only)."""
    return _resolve_arg_mode(mode) == MODE_BULBE


def is_sandbox_tool(tool_name: str) -> bool:
    """Whether the tool must dispatch through the sandbox (FS / shell / code)."""
    return tool_name in SANDBOX_TOOL_NAMES


# Human approval seam (Bulbe), fail-secure


def _approval_manager():
    """Lazily fetch the ``tool_call_approval`` singleton, or None."""
    try:
        from opti_oignon.tool_call_approval import tool_call_approval as mgr

        return mgr
    except Exception:  # pragma: no cover - defensive guard
        return None


def _is_approved(status: Any) -> bool:
    """True only for an explicit approved status (enum or string); else False."""
    if status is None:
        return False
    value = getattr(status, "value", status)
    return value == "approved"


def request_approval(
    conversation_id: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    manager: Any = None,
    timeout: float | None = None,
) -> bool:
    """Submit a tool call to the human gate and wait, fail-secure.

    Returns True only when the human explicitly approves before the timeout.
    A missing gate, a submit failure, a timeout, a denial, or any error all
    return False, so the absence of a positive signal denies the call.
    """
    mgr = manager if manager is not None else _approval_manager()
    if mgr is None:
        logger.warning("Bulbe approval gate unavailable; denying %s", tool_name)
        return False
    try:
        approval_id, event = mgr.submit(conversation_id, tool_name, dict(arguments or {}))
    except Exception:
        logger.warning("Approval submit failed for %s; denying", tool_name)
        return False
    wait_s = APPROVAL_TIMEOUT if timeout is None else float(timeout)
    try:
        event.wait(wait_s)
        status = mgr.get_status(approval_id)
    except Exception:
        return False
    return _is_approved(status)


def _default_approval_fn(
    conversation_id: str, tool_name: str, arguments: dict[str, Any]
) -> bool:
    """The default Bulbe approval callable, wrapping the real manager."""
    return request_approval(conversation_id, tool_name, arguments)


def evaluate(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    mode: str | None = None,
    conversation_id: str = "",
    approval_fn: Callable[[str, str, dict[str, Any]], bool] | None = None,
) -> GateDecision:
    """Full per-mode gate for one tool call: allowlist, then Bulbe approval.

    The allowlist is consulted first. In Bulbe an allowed call must then pass
    the human gate (the injected ``approval_fn``, or the default manager-backed
    one). Any exception in the approval path is treated as a denial. The
    session-state and subagent tools (S228) are exempt from the per-call gate
    per the 5.6 mode-posture table: todo mutates nothing outside the run, and
    a task launch is bounded by the child registry, every child sandbox call
    riding its own approval.
    """
    resolved = _resolve_arg_mode(mode)
    args = dict(arguments or {})
    if not is_tool_allowed(tool_name, resolved):
        return GateDecision(False, REASON_NOT_ALLOWED, resolved)
    if requires_approval(resolved) and tool_name not in (
        SESSION_STATE_TOOLS | SUBAGENT_TOOLS
    ):
        fn = approval_fn if approval_fn is not None else _default_approval_fn
        try:
            ok = bool(fn(conversation_id, tool_name, args))
        except Exception:
            ok = False  # fail-secure
        if not ok:
            return GateDecision(False, REASON_DENIED, resolved)
    return GateDecision(True, REASON_ALLOWED, resolved)
