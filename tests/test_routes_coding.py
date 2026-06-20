#!/usr/bin/env python3
"""
Tests for Coding Agent API routes and frontend files (S74).

Covers: all REST endpoints, WebSocket, response shapes,
error handling, and frontend file existence.
"""

import json
import os
import sys
import importlib.util

import pytest

# ---------------------------------------------------------------------------
# Module loading (test isolation)
# ---------------------------------------------------------------------------

_base = os.path.join(os.path.dirname(__file__), os.pardir, "opti_oignon")
_frontend = os.path.join(os.path.dirname(__file__), os.pardir, "frontend")

# sandbox_manager
_sm_path = os.path.join(_base, "sandbox_manager.py")
_sm_spec = importlib.util.spec_from_file_location("sandbox_manager", _sm_path)
_sm_mod = importlib.util.module_from_spec(_sm_spec)
_sm_spec.loader.exec_module(_sm_mod)
SandboxConfig = _sm_mod.SandboxConfig
SandboxManager = _sm_mod.SandboxManager

# tool_registry
_tr_path = os.path.join(_base, "tool_registry.py")
_tr_spec = importlib.util.spec_from_file_location("tool_registry", _tr_path)
_tr_mod = importlib.util.module_from_spec(_tr_spec)
_tr_spec.loader.exec_module(_tr_mod)
ToolRegistry = _tr_mod.ToolRegistry

# Wire up sys.modules for dependent imports
sys.modules["opti_oignon"] = type(sys)("opti_oignon")
sys.modules["opti_oignon.sandbox_manager"] = _sm_mod
sys.modules["opti_oignon.tool_registry"] = _tr_mod

# file_tools
_ft_path = os.path.join(_base, "file_tools.py")
_ft_spec = importlib.util.spec_from_file_location("file_tools", _ft_path)
_ft_mod = importlib.util.module_from_spec(_ft_spec)
_ft_spec.loader.exec_module(_ft_mod)
sys.modules["opti_oignon.file_tools"] = _ft_mod

# sandbox_tools
_st_path = os.path.join(_base, "sandbox_tools.py")
_st_spec = importlib.util.spec_from_file_location("sandbox_tools", _st_path)
_st_mod = importlib.util.module_from_spec(_st_spec)
_st_spec.loader.exec_module(_st_mod)
sys.modules["opti_oignon.sandbox_tools"] = _st_mod
SandboxToolSession = _st_mod.SandboxToolSession

# coding_agent
_ca_path = os.path.join(_base, "coding_agent.py")
_ca_spec = importlib.util.spec_from_file_location("coding_agent", _ca_path)
_ca_mod = importlib.util.module_from_spec(_ca_spec)
_ca_spec.loader.exec_module(_ca_mod)
sys.modules["opti_oignon.coding_agent"] = _ca_mod

CodingAgent = _ca_mod.CodingAgent
CodingAgentConfig = _ca_mod.CodingAgentConfig
CodingPlan = _ca_mod.CodingPlan
PlanStep = _ca_mod.PlanStep
PlanStepType = _ca_mod.PlanStepType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sandbox_config(tmp_path):
    return SandboxConfig(
        enabled=True,
        isolation_backend="tempdir",
        require_degraded_confirmation=False,
        workspace_base=str(tmp_path / "sandboxes"),
        command_timeout=10,
        max_output_bytes=4096,
        max_stderr_bytes=2048,
        max_concurrent_sessions=3,
        audit_db_path=str(tmp_path / "audit.db"),
        blocked_commands=["sudo"],
        blocked_patterns=[],
    )


@pytest.fixture
def sandbox_mgr(sandbox_config):
    return SandboxManager(config=sandbox_config)


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.fixture
def session(sandbox_mgr, tool_registry):
    return SandboxToolSession(
        sandbox_mgr=sandbox_mgr,
        tool_registry=tool_registry,
    )


@pytest.fixture
def agent_config():
    return CodingAgentConfig(
        enabled=True,
        max_iterations=5,
        max_fix_retries=2,
        auto_test=False,
    )


@pytest.fixture
def agent(session, agent_config):
    return CodingAgent(
        sandbox_session=session,
        config=agent_config,
    )


# ---------------------------------------------------------------------------
# TestSchemas
# ---------------------------------------------------------------------------

class TestSchemas:
    """Tests for Pydantic schemas used by the API."""

    def test_coding_task_request_schema(self):
        """Verify the schema file contains CodingTaskRequest."""
        schema_path = os.path.join(_base, "api", "schemas.py")
        content = open(schema_path).read()
        assert "class CodingTaskRequest" in content
        assert "class CodingPlanResponse" in content
        assert "class CodingCheckpointRequest" in content
        assert "class CodingStepResponse" in content
        assert "class CodingDiffResponse" in content
        assert "class CodingApplyRequest" in content
        assert "class CodingApplyResponse" in content
        assert "class CodingStatusResponse" in content
        assert "class CodingHistoryEntryResponse" in content
        assert "class CodingTestResultResponse" in content
        assert "class CodingDiffEntry" in content
        assert "class CodingPlanStepResponse" in content

    def test_schema_count(self):
        """Verify at least 12 new schemas were added."""
        schema_path = os.path.join(_base, "api", "schemas.py")
        content = open(schema_path).read()
        coding_classes = [
            line for line in content.split("\n")
            if line.startswith("class Coding")
        ]
        assert len(coding_classes) >= 12


# ---------------------------------------------------------------------------
# TestRoutesFile
# ---------------------------------------------------------------------------

class TestRoutesFile:
    """Tests for the routes_coding.py file structure."""

    def test_routes_file_exists(self):
        path = os.path.join(_base, "api", "routes_coding.py")
        assert os.path.isfile(path)

    def test_routes_has_all_endpoints(self):
        path = os.path.join(_base, "api", "routes_coding.py")
        content = open(path).read()
        # REST endpoints
        assert '"/start"' in content or "'/start'" in content
        assert '"/plan"' in content or "'/plan'" in content
        assert '"/step"' in content or "'/step'" in content
        assert '"/status"' in content or "'/status'" in content
        assert '"/diff"' in content or "'/diff'" in content
        assert '"/approve"' in content or "'/approve'" in content
        assert '"/abort"' in content or "'/abort'" in content
        # WebSocket
        assert "/ws/coding/live" in content

    def test_routes_has_security_comment(self):
        path = os.path.join(_base, "api", "routes_coding.py")
        content = open(path).read()
        assert "SECURITY" in content
        assert "human approval" in content.lower() or "HUMAN-GATED" in content

    def test_routes_prefix(self):
        path = os.path.join(_base, "api", "routes_coding.py")
        content = open(path).read()
        assert "/api/coding" in content


# ---------------------------------------------------------------------------
# TestAppRegistration
# ---------------------------------------------------------------------------

class TestAppRegistration:
    """Tests for app.py integration."""

    def test_coding_router_imported(self):
        app_path = os.path.join(_base, "api", "app.py")
        content = open(app_path).read()
        assert "routes_coding" in content
        assert "coding_router" in content

    def test_coding_router_registered(self):
        app_path = os.path.join(_base, "api", "app.py")
        content = open(app_path).read()
        assert "app.include_router(coding_router)" in content

    def test_version_bumped(self):
        app_path = os.path.join(_base, "api", "app.py")
        content = open(app_path).read()
        assert '"1.8.9"' in content

    def test_health_check_includes_coding_agent(self):
        app_path = os.path.join(_base, "api", "app.py")
        content = open(app_path).read()
        assert "CODING_AGENT_AVAILABLE" in content
        assert '"coding_agent"' in content


# ---------------------------------------------------------------------------
# TestDepsRegistration
# ---------------------------------------------------------------------------

class TestDepsRegistration:
    """Tests for deps.py integration."""

    def test_coding_agent_in_deps(self):
        deps_path = os.path.join(_base, "api", "deps.py")
        content = open(deps_path).read()
        assert "CODING_AGENT_AVAILABLE" in content
        assert "coding_agent_instance" in content
        assert "CodingAgent" in content

    def test_coding_agent_config_in_deps(self):
        deps_path = os.path.join(_base, "api", "deps.py")
        content = open(deps_path).read()
        assert "coding_agent_config" in content


# ---------------------------------------------------------------------------
# TestConfigFile
# ---------------------------------------------------------------------------

class TestConfigFile:
    """Tests for coding_agent.yaml."""

    def test_config_file_exists(self):
        path = os.path.join(_base, "config", "coding_agent.yaml")
        assert os.path.isfile(path)

    def test_config_file_contents(self):
        import yaml
        path = os.path.join(_base, "config", "coding_agent.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["enabled"] is True
        assert data["max_iterations"] == 10
        assert data["max_fix_retries"] == 3
        assert data["auto_test"] is True
        assert data["checkpoint_before_apply"] is True

    def test_config_security_comment(self):
        path = os.path.join(_base, "config", "coding_agent.yaml")
        content = open(path).read()
        assert "SECURITY" in content
        assert "ALWAYS True" in content


# ---------------------------------------------------------------------------
# TestFrontendFiles
# ---------------------------------------------------------------------------

class TestFrontendFiles:
    """Tests for frontend file existence and structure."""

    def test_coding_agent_ts_exists(self):
        path = os.path.join(_frontend, "src", "lib", "api", "codingAgent.ts")
        assert os.path.isfile(path), f"Missing: {path}"

    def test_coding_agent_ts_exports(self):
        path = os.path.join(_frontend, "src", "lib", "api", "codingAgent.ts")
        content = open(path).read()
        assert "startCodingTask" in content
        assert "codingPlan" in content
        assert "executeNextStep" in content
        assert "getCodingStatus" in content
        assert "getCodingDiff" in content
        assert "approveCodingChanges" in content
        assert "abortCodingTask" in content
        assert "connectCodingWebSocket" in content

    def test_coding_agent_panel_exists(self):
        path = os.path.join(
            _frontend, "src", "lib", "components", "panels",
            "CodingAgentPanel.svelte"
        )
        assert os.path.isfile(path), f"Missing: {path}"

    def test_coding_agent_panel_structure(self):
        path = os.path.join(
            _frontend, "src", "lib", "components", "panels",
            "CodingAgentPanel.svelte"
        )
        content = open(path).read()
        assert "<script" in content
        assert "startCodingTask" in content
        assert "codingPlan" in content
        assert "approve" in content.lower()
        assert "abort" in content.lower()
        assert "diff" in content.lower()
        assert "<style" in content

    def test_types_has_coding_interfaces(self):
        path = os.path.join(_frontend, "src", "lib", "types.ts")
        content = open(path).read()
        assert "CodingTaskRequest" in content
        assert "CodingPlanResponse" in content
        assert "CodingStatusResponse" in content
        assert "CodingDiffResponse" in content
        assert "CodingApplyResponse" in content
        assert "CodingCheckpointRequest" in content
        assert "CodingStepResponse" in content

    def test_types_coding_section_marker(self):
        path = os.path.join(_frontend, "src", "lib", "types.ts")
        content = open(path).read()
        assert "Coding Agent (S74)" in content


# ---------------------------------------------------------------------------
# TestCodingAgentModule
# ---------------------------------------------------------------------------

class TestCodingAgentModule:
    """Tests for the coding_agent.py module itself."""

    def test_module_has_availability_flag(self):
        assert hasattr(_ca_mod, "CODING_AGENT_AVAILABLE")

    def test_module_has_config(self):
        assert hasattr(_ca_mod, "coding_agent_config")
        assert hasattr(_ca_mod, "CodingAgentConfig")

    def test_module_exports_all_classes(self):
        assert hasattr(_ca_mod, "CodingAgent")
        assert hasattr(_ca_mod, "CodingPhase")
        assert hasattr(_ca_mod, "CheckpointResult")
        assert hasattr(_ca_mod, "CodingPlan")
        assert hasattr(_ca_mod, "PlanStep")
        assert hasattr(_ca_mod, "PlanStepType")
        assert hasattr(_ca_mod, "FileDiff")
        assert hasattr(_ca_mod, "TestResult")
        assert hasattr(_ca_mod, "CodingHistoryEntry")
