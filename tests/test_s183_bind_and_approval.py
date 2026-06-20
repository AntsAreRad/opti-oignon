#!/usr/bin/env python3
"""
S183 / N-01 and AP-01 verifications.

N-01: the launcher must run a deterministic bind backstop -- in Bulbe mode a
non-loopback bind address terminates startup (sys.exit) -- and main.py must call
it before uvicorn.run.

AP-01: the approval-gate caller (request_approval) must treat a wait-timeout
(no explicit approval) as a denial; only an explicit "approved" status allows.
"""

import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(rel_path, name, parents=("opti_oignon",)):
    fpath = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(fpath))
    mod = importlib.util.module_from_spec(spec)
    for pkg in parents:
        sys.modules.setdefault(pkg, types.ModuleType(pkg))
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# N-01: deterministic bind backstop
# ---------------------------------------------------------------------------

guard = _load("opti_oignon/network_bind_guard.py", "opti_oignon.network_bind_guard")
guard._audit_critical_event = lambda *a, **k: None  # isolate from audit chain

MAIN_SRC = (ROOT / "opti_oignon" / "main.py").read_text(encoding="utf-8")


class TestBindBackstop:
    def test_bulbe_non_loopback_exits(self, monkeypatch):
        monkeypatch.setattr(guard, "_get_current_mode", lambda: "bulbe")
        with pytest.raises(SystemExit):
            guard.assert_safe_bind_address("0.0.0.0")

    def test_bulbe_loopback_ok(self, monkeypatch):
        monkeypatch.setattr(guard, "_get_current_mode", lambda: "bulbe")
        guard.assert_safe_bind_address("127.0.0.1")  # no exit
        guard.assert_safe_bind_address("::1")  # loopback v6, no exit

    def test_daily_non_loopback_allowed(self, monkeypatch):
        # Daily may bind non-loopback (remote access); the backstop is Bulbe-only.
        monkeypatch.setattr(guard, "_get_current_mode", lambda: "daily")
        guard.assert_safe_bind_address("0.0.0.0")  # no exit

    def test_main_calls_backstop_before_run(self):
        assert "assert_safe_bind_address(actual_host)" in MAIN_SRC
        # It must appear before the uvicorn.run call site.
        assert MAIN_SRC.index("assert_safe_bind_address(actual_host)") < MAIN_SRC.index(
            "uvicorn.run(app"
        )


# ---------------------------------------------------------------------------
# AP-01: approval caller treats wait-timeout / non-approval as deny
# ---------------------------------------------------------------------------

allow = _load(
    "opti_oignon/agent/allowlists.py",
    "opti_oignon.agent.allowlists",
    parents=("opti_oignon", "opti_oignon.agent"),
)
TEXEC_SRC = (ROOT / "opti_oignon" / "tool_executor.py").read_text(encoding="utf-8")


class _FakeMgr:
    def __init__(self, status, set_event=True, raise_submit=False):
        self.status = status
        self.set_event = set_event
        self.raise_submit = raise_submit

    def submit(self, conv, tool, args):
        if self.raise_submit:
            raise RuntimeError("submit boom")
        ev = threading.Event()
        if self.set_event:
            ev.set()
        return "aid-1", ev

    def get_status(self, aid):
        return self.status


class TestApprovalFailSecure:
    def test_timeout_is_denied(self):
        # Event never set and status still pending -> the wait elapses -> deny.
        mgr = _FakeMgr(status="pending", set_event=False)
        assert allow.request_approval(
            "conv", "bash", {}, manager=mgr, timeout=0.05,
        ) is False

    def test_explicit_approval_allows(self):
        mgr = _FakeMgr(status="approved", set_event=True)
        assert allow.request_approval("conv", "bash", {}, manager=mgr) is True

    def test_denied_status_denies(self):
        mgr = _FakeMgr(status="denied", set_event=True)
        assert allow.request_approval("conv", "bash", {}, manager=mgr) is False

    def test_timeout_status_denies(self):
        mgr = _FakeMgr(status="timeout", set_event=True)
        assert allow.request_approval("conv", "bash", {}, manager=mgr) is False

    def test_submit_failure_denies(self):
        mgr = _FakeMgr(status="approved", raise_submit=True)
        assert allow.request_approval("conv", "bash", {}, manager=mgr) is False

    def test_missing_gate_denies(self, monkeypatch):
        monkeypatch.setattr(allow, "_approval_manager", lambda: None)
        assert allow.request_approval("conv", "bash", {}, manager=None) is False


class TestExecutorHonorsDeny:
    def test_executor_treats_hook_false_and_error_as_denied(self):
        assert "Tool call denied by approval gate" in TEXEC_SRC
        assert "approval error" in TEXEC_SRC
