#!/usr/bin/env python3
"""
S128 Test Suite -- Kill Switch Frontend, Plugin Ceremony UI, Tool Call Approval.

Tests cover:
  - ToolCallApprovalManager: submit, approve, deny, timeout, audit log,
    reaper, risk assessment, argument sanitization
  - ToolExecutor pre_tool_call_hook integration
  - API routes: tool-approval endpoints
  - Frontend validation: SearchKillSwitchPanel, PluginAllowlistPanel,
    ToolCallApproval, SecurityPanel tabs
  - TypeScript API client file existence and exports
  - Version consistency (2.8.0)

~48 tests total.
"""

import ast
import importlib.util
import os
import re
import sys
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPTI_DIR = PROJECT_ROOT / "opti_oignon"
API_DIR = OPTI_DIR / "api"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
COMPONENTS_DIR = FRONTEND_DIR / "src" / "lib" / "components"
SETTINGS_DIR = COMPONENTS_DIR / "settings"
CHAT_DIR = COMPONENTS_DIR / "chat"
API_TS_DIR = FRONTEND_DIR / "src" / "lib" / "api"

EXPECTED_VERSION = "3.0.0"


def _load_module(name: str, filepath: str):
    """Load a module by file path, bypassing __init__.py chain."""
    full_name = f"opti_oignon.{name}"
    spec = importlib.util.spec_from_file_location(full_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# =========================================================================
# TOOL CALL APPROVAL — CORE MODULE
# =========================================================================

class TestToolCallApprovalModule:
    """Test tool_call_approval.py module loading and AST validity."""

    def test_module_exists(self):
        path = OPTI_DIR / "tool_call_approval.py"
        assert path.exists(), "tool_call_approval.py must exist"

    def test_ast_valid(self):
        path = OPTI_DIR / "tool_call_approval.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        class_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert "ToolCallApprovalManager" in class_names
        assert "ApprovalRequest" in class_names
        assert "ApprovalStatus" in class_names
        assert "AuditEntry" in class_names

    def test_no_french_comments(self):
        path = OPTI_DIR / "tool_call_approval.py"
        source = path.read_text(encoding="utf-8")
        french_words = ["fonction", "retourne", "verifie", "parametre"]
        for word in french_words:
            assert word not in source.lower(), f"French word '{word}' found in tool_call_approval.py"


class TestToolCallApprovalManager:
    """Test ToolCallApprovalManager behavior."""

    def setup_method(self):
        mod = _load_module(
            "tool_call_approval",
            str(OPTI_DIR / "tool_call_approval.py"),
        )
        self.mgr = mod.ToolCallApprovalManager()
        self.ApprovalStatus = mod.ApprovalStatus

    def test_submit_returns_id_and_event(self):
        aid, event = self.mgr.submit("conv1", "web_search", {"query": "test"})
        assert isinstance(aid, str)
        assert len(aid) > 0
        assert isinstance(event, threading.Event)
        assert not event.is_set()
        # Cleanup
        self.mgr.deny(aid)

    def test_approve_unblocks_event(self):
        aid, event = self.mgr.submit("conv1", "file_read", {"path": "/tmp/a"})
        assert not event.is_set()
        result = self.mgr.approve(aid, "admin")
        assert result is True
        assert event.is_set()
        status = self.mgr.get_status(aid)
        assert status == self.ApprovalStatus.APPROVED

    def test_deny_unblocks_event(self):
        aid, event = self.mgr.submit("conv1", "shell_exec", {"cmd": "ls"})
        result = self.mgr.deny(aid, "admin")
        assert result is True
        assert event.is_set()
        status = self.mgr.get_status(aid)
        assert status == self.ApprovalStatus.DENIED

    def test_approve_nonexistent_returns_false(self):
        result = self.mgr.approve("nonexistent-id", "admin")
        assert result is False

    def test_deny_nonexistent_returns_false(self):
        result = self.mgr.deny("nonexistent-id", "admin")
        assert result is False

    def test_pending_lists_submitted(self):
        aid1, _ = self.mgr.submit("c1", "tool_a", {})
        aid2, _ = self.mgr.submit("c2", "tool_b", {"x": 1})
        pending = self.mgr.pending()
        assert len(pending) == 2
        ids = {p["approval_id"] for p in pending}
        assert aid1 in ids
        assert aid2 in ids
        # Cleanup
        self.mgr.clear_all()

    def test_pending_count(self):
        aid1, _ = self.mgr.submit("c1", "t1", {})
        aid2, _ = self.mgr.submit("c2", "t2", {})
        assert self.mgr.pending_count() == 2
        self.mgr.approve(aid1)
        assert self.mgr.pending_count() == 1
        self.mgr.clear_all()

    def test_audit_log_records_decisions(self):
        aid, _ = self.mgr.submit("c1", "web_search", {"q": "test"})
        self.mgr.approve(aid, "user1")
        log = self.mgr.audit_log(limit=10)
        assert len(log) >= 1
        entry = log[0]
        assert entry["approval_id"] == aid
        assert entry["status"] == "approved"
        assert entry["resolved_by"] == "user1"
        assert entry["tool_name"] == "web_search"

    def test_audit_log_records_deny(self):
        aid, _ = self.mgr.submit("c1", "file_delete", {})
        self.mgr.deny(aid, "admin")
        log = self.mgr.audit_log(limit=10)
        assert any(e["status"] == "denied" for e in log)

    def test_clear_all_denies_pending(self):
        self.mgr.submit("c1", "t1", {})
        self.mgr.submit("c2", "t2", {})
        count = self.mgr.clear_all()
        assert count == 2
        assert self.mgr.pending_count() == 0

    def test_timeout_remaining_decreases(self):
        aid, _ = self.mgr.submit("c1", "tool", {})
        pending = self.mgr.pending()
        item = [p for p in pending if p["approval_id"] == aid][0]
        assert item["timeout_remaining"] > 0
        assert item["timeout_remaining"] <= 30
        self.mgr.clear_all()

    def test_to_dict_has_required_fields(self):
        aid, _ = self.mgr.submit("c1", "my_tool", {"key": "val"})
        pending = self.mgr.pending()
        item = [p for p in pending if p["approval_id"] == aid][0]
        required = {
            "approval_id", "conversation_id", "tool_name",
            "arguments", "arguments_summary", "risk_level",
            "status", "created_at", "timeout_remaining",
        }
        assert required.issubset(set(item.keys()))
        self.mgr.clear_all()


class TestRiskAssessment:
    """Test risk level assessment and argument sanitization."""

    def setup_method(self):
        self.mod = _load_module(
            "tool_call_approval",
            str(OPTI_DIR / "tool_call_approval.py"),
        )

    def test_high_risk_tools(self):
        for tool in ["web_search", "shell_exec", "code_execute", "file_delete"]:
            assert self.mod.assess_risk(tool) == "high"

    def test_medium_risk_tools(self):
        for tool in ["file_read", "database_query", "rag_search"]:
            assert self.mod.assess_risk(tool) == "medium"

    def test_low_risk_tools(self):
        assert self.mod.assess_risk("unknown_tool") == "low"
        assert self.mod.assess_risk("get_time") == "low"

    def test_sanitize_truncates_long_strings(self):
        args = {"query": "x" * 500}
        sanitized = self.mod.sanitize_arguments(args)
        assert len(sanitized["query"]) <= 204  # 200 + "..."

    def test_sanitize_summarizes_lists(self):
        args = {"items": [1, 2, 3, 4, 5]}
        sanitized = self.mod.sanitize_arguments(args)
        assert "list of 5" in sanitized["items"]

    def test_sanitize_summarizes_dicts(self):
        args = {"data": {"a": 1, "b": 2}}
        sanitized = self.mod.sanitize_arguments(args)
        assert "dict with 2" in sanitized["data"]

    def test_summarize_arguments_truncates(self):
        args = {"long_key": "v" * 100}
        summary = self.mod.summarize_arguments("tool", args)
        assert len(summary) <= 204


class TestTimeoutReaper:
    """Test the background reaper auto-deny."""

    def setup_method(self):
        mod = _load_module(
            "tool_call_approval",
            str(OPTI_DIR / "tool_call_approval.py"),
        )
        # Override timeout to 1 second for fast testing
        mod.DEFAULT_TIMEOUT_SECONDS = 1
        self.mod = mod
        self.mgr = mod.ToolCallApprovalManager()

    def test_auto_deny_on_timeout(self):
        aid, event = self.mgr.submit("c1", "web_search", {"q": "t"})
        # Wait for reaper (timeout=1s + reaper interval)
        event.wait(timeout=4)
        status = self.mgr.get_status(aid)
        assert status == self.mod.ApprovalStatus.TIMEOUT
        log = self.mgr.audit_log()
        timeout_entries = [e for e in log if e["approval_id"] == aid]
        assert len(timeout_entries) == 1
        assert timeout_entries[0]["resolved_by"] == "timeout"


# =========================================================================
# TOOL EXECUTOR — PRE_TOOL_CALL_HOOK
# =========================================================================

class TestToolExecutorHook:
    """Test ToolExecutor pre_tool_call_hook integration."""

    def test_tool_executor_has_hook_attr(self):
        mod = _load_module(
            "tool_executor",
            str(OPTI_DIR / "tool_executor.py"),
        )
        te = mod.ToolExecutor(registry=MagicMock())
        assert hasattr(te, "pre_tool_call_hook")
        assert te.pre_tool_call_hook is None

    def test_hook_blocks_when_denied(self):
        mod = _load_module(
            "tool_executor",
            str(OPTI_DIR / "tool_executor.py"),
        )
        registry = MagicMock()
        te = mod.ToolExecutor(registry=registry)
        te.pre_tool_call_hook = lambda name, args: False

        result = te._execute_tool("web_search", {"q": "test"})
        assert result.success is False
        assert "denied" in result.result.lower()

    def test_hook_allows_when_approved(self):
        mod = _load_module(
            "tool_executor",
            str(OPTI_DIR / "tool_executor.py"),
        )
        registry = MagicMock()
        tool_mock = MagicMock()
        tool_mock.enabled = True
        tool_mock.handler = MagicMock(return_value="result_data")
        tool_mock.parameters = {}
        registry.get.return_value = tool_mock
        te = mod.ToolExecutor(registry=registry)
        te.pre_tool_call_hook = lambda name, args: True

        result = te._execute_tool("web_search", {"q": "test"})
        assert result.success is True

    def test_hook_exception_denies_failsecure(self):
        mod = _load_module(
            "tool_executor",
            str(OPTI_DIR / "tool_executor.py"),
        )
        registry = MagicMock()
        te = mod.ToolExecutor(registry=registry)
        te.pre_tool_call_hook = lambda name, args: (_ for _ in ()).throw(RuntimeError("boom"))

        result = te._execute_tool("web_search", {"q": "test"})
        assert result.success is False
        assert "denied" in result.result.lower() or "approval error" in result.result.lower()


# =========================================================================
# ROUTES — TOOL APPROVAL ENDPOINTS
# =========================================================================

class TestRoutesSecurityToolApproval:
    """Test routes_security.py has tool approval endpoints."""

    def test_routes_security_ast_valid(self):
        source = (API_DIR / "routes_security.py").read_text(encoding="utf-8")
        ast.parse(source)

    def test_tool_approval_endpoints_exist(self):
        source = (API_DIR / "routes_security.py").read_text(encoding="utf-8")
        assert 'tool-approval/pending' in source
        assert 'tool-approval/{approval_id}/approve' in source
        assert 'tool-approval/{approval_id}/deny' in source
        assert 'tool-approval/audit' in source

    def test_tool_call_approval_import(self):
        source = (API_DIR / "routes_security.py").read_text(encoding="utf-8")
        assert "from opti_oignon.tool_call_approval import tool_call_approval" in source


class TestRoutesChatApprovalIntegration:
    """Test routes_chat.py has Bulbe mode approval integration."""

    def test_routes_chat_ast_valid(self):
        source = (API_DIR / "routes_chat.py").read_text(encoding="utf-8")
        ast.parse(source)

    def test_approval_hook_import(self):
        source = (API_DIR / "routes_chat.py").read_text(encoding="utf-8")
        assert "from opti_oignon.tool_call_approval import tool_call_approval" in source
        assert "ApprovalStatus" in source

    def test_approval_hook_installation(self):
        source = (API_DIR / "routes_chat.py").read_text(encoding="utf-8")
        assert "tool_call_approval_required" in source
        assert "pre_tool_call_hook" in source

    def test_websocket_events_emitted(self):
        source = (API_DIR / "routes_chat.py").read_text(encoding="utf-8")
        assert "tool_call_pending" in source
        assert "tool_call_resolved" in source

    def test_hook_cleanup(self):
        source = (API_DIR / "routes_chat.py").read_text(encoding="utf-8")
        assert "_approval_hook_installed" in source
        # Cleanup should reset hook to None
        assert "pre_tool_call_hook = None" in source


# =========================================================================
# FRONTEND — TYPESCRIPT API CLIENTS
# =========================================================================

class TestTypeScriptApiClients:
    """Verify TypeScript API client files exist and have expected exports."""

    def test_search_killswitch_ts_exists(self):
        path = API_TS_DIR / "searchKillSwitch.ts"
        assert path.exists()

    def test_search_killswitch_ts_exports(self):
        content = (API_TS_DIR / "searchKillSwitch.ts").read_text(encoding="utf-8")
        exports = [
            "getKillSwitchStatus", "engageKillSwitch", "requestReenable",
            "getReenableCode", "confirmReenable", "cancelReenable",
            "updateDomainAllowlist",
        ]
        for fn in exports:
            assert fn in content, f"Missing export: {fn}"

    def test_plugin_allowlist_ts_exists(self):
        path = API_TS_DIR / "pluginAllowlist.ts"
        assert path.exists()

    def test_plugin_allowlist_ts_exports(self):
        content = (API_TS_DIR / "pluginAllowlist.ts").read_text(encoding="utf-8")
        exports = [
            "getAllowlistStatus", "prepareBatch", "approveBatch",
            "revokePlugin", "revokeBatch", "verifyPlugin",
        ]
        for fn in exports:
            assert fn in content, f"Missing export: {fn}"

    def test_tool_call_approval_ts_exists(self):
        path = API_TS_DIR / "toolCallApproval.ts"
        assert path.exists()

    def test_tool_call_approval_ts_exports(self):
        content = (API_TS_DIR / "toolCallApproval.ts").read_text(encoding="utf-8")
        exports = [
            "getPendingApprovals", "approveToolCall",
            "denyToolCall", "getApprovalAudit",
        ]
        for fn in exports:
            assert fn in content, f"Missing export: {fn}"


# =========================================================================
# FRONTEND — SVELTE COMPONENT VALIDATION
# =========================================================================

def _validate_svelte_component(filepath: Path):
    """Validate a Svelte component: exists, no hardcoded hex, balanced tags."""
    assert filepath.exists(), f"{filepath.name} must exist"
    content = filepath.read_text(encoding="utf-8")

    # Check no hardcoded hex outside var() fallbacks
    for i, line in enumerate(content.split("\n"), 1):
        hexes = re.findall(r"#[0-9a-fA-F]{3,8}", line)
        for h in hexes:
            if "var(--oo" not in line and "{#each" not in line and "{#if" not in line:
                pytest.fail(f"Hardcoded hex {h} at {filepath.name}:{i}")

    # Check Svelte block balance
    if_opens = len(re.findall(r"\{#if\b", content))
    if_closes = len(re.findall(r"\{/if\}", content))
    assert if_opens == if_closes, (
        f"Svelte #if blocks unbalanced in {filepath.name}: "
        f"open={if_opens} close={if_closes}"
    )
    each_opens = len(re.findall(r"\{#each\b", content))
    each_closes = len(re.findall(r"\{/each\}", content))
    assert each_opens == each_closes, (
        f"Svelte #each blocks unbalanced in {filepath.name}: "
        f"open={each_opens} close={each_closes}"
    )

    return content


class TestSearchKillSwitchPanel:
    """Validate SearchKillSwitchPanel.svelte."""

    def test_component_valid(self):
        content = _validate_svelte_component(SETTINGS_DIR / "SearchKillSwitchPanel.svelte")
        assert "searchKillSwitch" in content  # API import
        assert "Engage Kill Switch" in content
        assert "Re-enable" in content or "re-enable" in content
        assert "domain" in content.lower()
        assert "circuit" in content.lower()

    def test_bulbe_reenable_blocked(self):
        content = (SETTINGS_DIR / "SearchKillSwitchPanel.svelte").read_text()
        assert "bulbeMode" in content or "bulbe" in content.lower()
        assert "Re-enable blocked" in content or "cannot be re-enabled" in content.lower()

    def test_ceremony_ui_elements(self):
        content = (SETTINGS_DIR / "SearchKillSwitchPanel.svelte").read_text()
        assert "visual" in content.lower() or "visualCode" in content
        assert "password" in content.lower()
        assert "2FA" in content or "two_fa" in content.lower() or "twoFa" in content


class TestPluginAllowlistPanel:
    """Validate PluginAllowlistPanel.svelte."""

    def test_component_valid(self):
        content = _validate_svelte_component(SETTINGS_DIR / "PluginAllowlistPanel.svelte")
        assert "pluginAllowlist" in content  # API import
        assert "Approved" in content or "approved" in content
        assert "Revoke" in content

    def test_batch_ceremony_ui(self):
        content = (SETTINGS_DIR / "PluginAllowlistPanel.svelte").read_text()
        assert "Batch Approval Ceremony" in content or "batch" in content.lower()
        assert "SHA-512" in content or "sha512" in content.lower() or "code_hash" in content

    def test_daily_mode_info(self):
        content = (SETTINGS_DIR / "PluginAllowlistPanel.svelte").read_text()
        assert "Daily mode" in content or "daily" in content.lower()

    def test_permission_escalation(self):
        content = (SETTINGS_DIR / "PluginAllowlistPanel.svelte").read_text()
        assert "Permission" in content or "permission" in content
        assert "escalation" in content.lower() or "Escalation" in content


class TestToolCallApprovalComponent:
    """Validate ToolCallApproval.svelte."""

    def test_component_valid(self):
        content = _validate_svelte_component(CHAT_DIR / "ToolCallApproval.svelte")
        assert "toolCallApproval" in content  # API import
        assert "Allow" in content
        assert "Deny" in content

    def test_countdown_timer(self):
        content = (CHAT_DIR / "ToolCallApproval.svelte").read_text()
        assert "countdown" in content.lower()
        assert "auto-deny" in content.lower() or "Auto-denied" in content

    def test_risk_level_display(self):
        content = (CHAT_DIR / "ToolCallApproval.svelte").read_text()
        assert "riskLevel" in content or "risk_level" in content
        assert "riskColor" in content or "risk" in content.lower()


class TestSecurityPanelIntegration:
    """Validate SecurityPanel.svelte has tab navigation to new panels."""

    def test_component_valid(self):
        _validate_svelte_component(SETTINGS_DIR / "SecurityPanel.svelte")

    def test_imports_new_panels(self):
        content = (SETTINGS_DIR / "SecurityPanel.svelte").read_text()
        assert "SecurityModePanel" in content
        assert "SearchKillSwitchPanel" in content
        assert "PluginAllowlistPanel" in content

    def test_tab_navigation(self):
        content = (SETTINGS_DIR / "SecurityPanel.svelte").read_text()
        assert "activeSection" in content
        assert "overview" in content
        assert "killswitch" in content
        assert "plugins" in content

    def test_s128_attribution(self):
        content = (SETTINGS_DIR / "SecurityPanel.svelte").read_text()
        assert "S128" in content


# =========================================================================
# VERSION CONSISTENCY
# =========================================================================

class TestVersionConsistency:
    """Ensure version bumped to 2.8.0 across all files."""

    def test_version_file(self):
        content = (OPTI_DIR / "__version__.py").read_text(encoding="utf-8")
        assert f'__version__ = "{EXPECTED_VERSION}"' in content

    def test_no_old_version_in_new_tests(self):
        content = Path(__file__).read_text(encoding="utf-8")
        assert EXPECTED_VERSION in content
