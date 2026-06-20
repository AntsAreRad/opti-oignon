#!/usr/bin/env python3
"""Tests for S175 -- per-mode tool gating (Theme 3 / Odysseus Core).

Exercises ``opti_oignon/agent/allowlists.py``: the Daily and Bulbe
``frozenset`` allowlists and the structural guarantee that Bulbe is tighter,
the mode resolution (fail-secure to Bulbe when ``security_mode`` is
unresolved), and the Bulbe human-approval seam over ``tool_call_approval``
(fail-secure auto-deny). The approval manager is faked or forced unavailable so
no test blocks on the real 30-second gate. Loaded in isolation via
``spec_from_file_location`` with ``opti_oignon`` stubbed.
"""

import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
AGENT = OO / "agent"


def _ensure_pkg():
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(OO)]
        sys.modules["opti_oignon"] = pkg
    if "opti_oignon.agent" not in sys.modules:
        apkg = types.ModuleType("opti_oignon.agent")
        apkg.__path__ = [str(AGENT)]
        sys.modules["opti_oignon.agent"] = apkg


def _ensure_agent(name: str):
    full = f"opti_oignon.agent.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(AGENT / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_pkg()
al = _ensure_agent("allowlists")


class FakeManager:
    """Stand-in for the tool_call_approval manager with a fixed verdict.

    ``decision`` controls the status returned by ``get_status``; ``preset``
    controls whether the event is already set (so ``event.wait`` returns at
    once) or left unset (so a small timeout elapses, exercising the auto-deny).
    """

    def __init__(self, decision: str = "approved", *, preset: bool = True):
        self.decision = decision
        self.preset = preset
        self.submitted: list[tuple] = []

    def submit(self, conversation_id, tool_name, arguments):
        self.submitted.append((conversation_id, tool_name, dict(arguments)))
        ev = threading.Event()
        if self.preset:
            ev.set()
        return "approval-1", ev

    def get_status(self, approval_id):
        return self.decision


class RaisingSubmitManager:
    def submit(self, *a, **k):
        raise RuntimeError("submit failed")


class RaisingStatusManager:
    def submit(self, *a, **k):
        ev = threading.Event()
        ev.set()
        return "x", ev

    def get_status(self, approval_id):
        raise RuntimeError("status failed")


# Module shape and allowlist sets


class TestModuleShape:
    def test_sentinels(self):
        assert al.checkpoint_before_apply is True
        assert al.FEATURE_AVAILABLE is True

    def test_valid_modes(self):
        assert al.MODE_DAILY == "daily"
        assert al.MODE_BULBE == "bulbe"
        assert al.VALID_MODES == ("daily", "bulbe")

    def test_allowlists_are_frozensets(self):
        assert isinstance(al.DAILY_ALLOWLIST, frozenset)
        assert isinstance(al.BULBE_ALLOWLIST, frozenset)
        assert isinstance(al.SANDBOX_TOOL_NAMES, frozenset)


class TestAllowlistContents:
    def test_sandbox_tools(self):
        assert al.SANDBOX_TOOL_NAMES == frozenset(
            {"bash", "view", "create_file", "str_replace"}
        )

    def test_daily_contains_sandbox_network_and_state_tools(self):
        assert al.SANDBOX_TOOL_NAMES <= al.DAILY_ALLOWLIST
        assert al.NETWORK_TOOLS <= al.DAILY_ALLOWLIST
        assert al.STATE_MUTATION_TOOLS <= al.DAILY_ALLOWLIST

    def test_bulbe_is_strict_subset_of_daily(self):
        assert al.BULBE_ALLOWLIST < al.DAILY_ALLOWLIST

    def test_bulbe_equals_sandbox_tools(self):
        # In Bulbe only the sandboxed tools are allowed.
        assert al.BULBE_ALLOWLIST == al.SANDBOX_TOOL_NAMES

    def test_bulbe_excludes_network_and_state_tools(self):
        assert not (al.NETWORK_TOOLS & al.BULBE_ALLOWLIST)
        assert not (al.STATE_MUTATION_TOOLS & al.BULBE_ALLOWLIST)

    def test_bulbe_derivation_is_structural(self):
        assert al.BULBE_ALLOWLIST == frozenset(
            al.DAILY_ALLOWLIST - al.NETWORK_TOOLS - al.STATE_MUTATION_TOOLS
        )


# Allowlist gate


class TestAllowlistGate:
    def test_allowlist_for_daily_and_bulbe(self):
        assert al.allowlist_for("daily") is al.DAILY_ALLOWLIST
        assert al.allowlist_for("bulbe") is al.BULBE_ALLOWLIST

    def test_allowlist_for_unknown_is_bulbe(self):
        assert al.allowlist_for("weird") is al.BULBE_ALLOWLIST
        assert al.allowlist_for(None) is al.BULBE_ALLOWLIST

    def test_is_tool_allowed_daily(self):
        assert al.is_tool_allowed("bash", "daily")
        assert al.is_tool_allowed("web_search", "daily")
        assert al.is_tool_allowed("manage_skills", "daily")

    def test_is_tool_allowed_bulbe(self):
        assert al.is_tool_allowed("view", "bulbe")
        assert not al.is_tool_allowed("web_search", "bulbe")
        assert not al.is_tool_allowed("manage_memory", "bulbe")

    def test_is_tool_allowed_unknown_tool(self):
        assert not al.is_tool_allowed("rm_rf_host", "daily")
        assert not al.is_tool_allowed("rm_rf_host", "bulbe")

    def test_is_sandbox_tool(self):
        assert al.is_sandbox_tool("bash")
        assert al.is_sandbox_tool("str_replace")
        assert not al.is_sandbox_tool("web_search")

    def test_requires_approval(self):
        assert al.requires_approval("bulbe") is True
        assert al.requires_approval("daily") is False
        # Unknown mode resolves to Bulbe, which requires approval.
        assert al.requires_approval("weird") is True


# Mode resolution (fail-secure)


class TestModeResolution:
    def test_resolve_none_is_bulbe(self):
        assert al._resolve_mode(None) == "bulbe"

    def test_resolve_daily(self):
        class SM:
            def get_current_mode(self):
                return "daily"

        assert al._resolve_mode(SM()) == "daily"

    def test_resolve_raising_is_bulbe(self):
        class SM:
            def get_current_mode(self):
                raise RuntimeError("boom")

        assert al._resolve_mode(SM()) == "bulbe"

    def test_resolve_invalid_value_is_bulbe(self):
        class SM:
            def get_current_mode(self):
                return "lockdown"

        assert al._resolve_mode(SM()) == "bulbe"

    def test_current_mode_returns_valid_mode(self):
        assert al.current_mode() in al.VALID_MODES


# Approval primitive (_is_approved, request_approval)


class TestApprovalPrimitive:
    def test_is_approved_string(self):
        assert al._is_approved("approved") is True
        assert al._is_approved("denied") is False
        assert al._is_approved("timeout") is False

    def test_is_approved_enum_like(self):
        class Status:
            value = "approved"

        assert al._is_approved(Status()) is True

    def test_is_approved_none(self):
        assert al._is_approved(None) is False

    def test_request_approval_approved(self):
        mgr = FakeManager("approved")
        assert al.request_approval("c", "bash", {"x": 1}, manager=mgr, timeout=0.01) is True
        assert mgr.submitted and mgr.submitted[0][1] == "bash"

    def test_request_approval_denied(self):
        mgr = FakeManager("denied")
        assert al.request_approval("c", "bash", {}, manager=mgr, timeout=0.01) is False

    def test_request_approval_timeout_is_deny(self):
        # Event never set; the small wait elapses and the status is a timeout.
        mgr = FakeManager("timeout", preset=False)
        assert al.request_approval("c", "bash", {}, manager=mgr, timeout=0.02) is False

    def test_request_approval_no_manager_is_deny(self):
        assert al.request_approval("c", "bash", {}, manager=None, timeout=0.01) is False

    def test_request_approval_submit_raises_is_deny(self):
        assert (
            al.request_approval("c", "bash", {}, manager=RaisingSubmitManager(), timeout=0.01)
            is False
        )

    def test_request_approval_status_raises_is_deny(self):
        assert (
            al.request_approval("c", "bash", {}, manager=RaisingStatusManager(), timeout=0.01)
            is False
        )


# Full gate (evaluate)


class TestEvaluate:
    def test_daily_allowed_no_approval(self):
        d = al.evaluate("bash", {"command": "ls"}, mode="daily")
        assert d.allowed is True
        assert d.reason == al.REASON_ALLOWED
        assert d.mode == "daily"

    def test_daily_not_in_allowlist(self):
        d = al.evaluate("rm_rf_host", mode="daily")
        assert d.allowed is False
        assert d.reason == al.REASON_NOT_ALLOWED

    def test_bulbe_allowed_with_approval(self):
        d = al.evaluate("bash", mode="bulbe", approval_fn=lambda c, t, a: True)
        assert d.allowed is True
        assert d.reason == al.REASON_ALLOWED

    def test_bulbe_denied_by_human(self):
        d = al.evaluate("bash", mode="bulbe", approval_fn=lambda c, t, a: False)
        assert d.allowed is False
        assert d.reason == al.REASON_DENIED

    def test_bulbe_approval_exception_is_deny(self):
        def boom(c, t, a):
            raise RuntimeError("gate error")

        d = al.evaluate("bash", mode="bulbe", approval_fn=boom)
        assert d.allowed is False
        assert d.reason == al.REASON_DENIED

    def test_bulbe_not_in_allowlist_skips_approval(self):
        called = {"n": 0}

        def gate(c, t, a):
            called["n"] += 1
            return True

        d = al.evaluate("web_search", mode="bulbe", approval_fn=gate)
        assert d.allowed is False
        assert d.reason == al.REASON_NOT_ALLOWED
        assert called["n"] == 0  # approval never consulted for a disallowed tool

    def test_bulbe_default_gate_is_fail_secure(self, monkeypatch):
        # With no approval_fn and the manager forced unavailable, the default
        # path denies instantly (no 30s wait, no real submit).
        monkeypatch.setattr(al, "_approval_manager", lambda: None)
        d = al.evaluate("bash", mode="bulbe")
        assert d.allowed is False
        assert d.reason == al.REASON_DENIED

    def test_evaluate_arguments_passed_to_gate(self):
        seen = {}

        def gate(c, t, a):
            seen.update({"conv": c, "tool": t, "args": a})
            return True

        al.evaluate("bash", {"command": "id"}, mode="bulbe", conversation_id="conv-7", approval_fn=gate)
        assert seen == {"conv": "conv-7", "tool": "bash", "args": {"command": "id"}}
