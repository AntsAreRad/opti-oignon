#!/usr/bin/env python3
"""
Tests for Batch File Reads (SQ-06) — S77.

Covers: _batch_read_files, _batch_read_tar, _batch_read_fallback,
refactored _snapshot_originals, refactored generate_diffs with batch reads.
"""

import base64
import io
import json
import os
import sys
import importlib.util
import tarfile
import tempfile
import time

import pytest

# ---------------------------------------------------------------------------
# Module loading (test isolation — no ollama needed)
# ---------------------------------------------------------------------------

_base = os.path.join(os.path.dirname(__file__), os.pardir, "opti_oignon")

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

# Ensure opti_oignon sub-modules are findable
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

CodingAgent = _ca_mod.CodingAgent
CodingAgentConfig = _ca_mod.CodingAgentConfig
CodingPhase = _ca_mod.CodingPhase
CodingPlan = _ca_mod.CodingPlan
PlanStep = _ca_mod.PlanStep
PlanStepType = _ca_mod.PlanStepType
FileDiff = _ca_mod.FileDiff


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
        max_output_bytes=65536,
        max_stderr_bytes=4096,
        max_concurrent_sessions=3,
        audit_db_path=str(tmp_path / "audit.db"),
        blocked_commands=["sudo", "curl", "wget"],
        blocked_patterns=[],
    )


@pytest.fixture
def sandbox_mgr(sandbox_config):
    return SandboxManager(config=sandbox_config)


@pytest.fixture
def session(sandbox_mgr):
    return SandboxToolSession(
        sandbox_mgr=sandbox_mgr,
        tool_registry=ToolRegistry(),
    )


@pytest.fixture
def agent_config():
    return CodingAgentConfig(
        enabled=True,
        max_iterations=5,
        max_fix_retries=2,
        auto_test=False,
        checkpoint_before_apply=True,
    )


@pytest.fixture
def agent(session, agent_config):
    return CodingAgent(
        sandbox_session=session,
        config=agent_config,
    )


def _create_tar_b64(file_dict: dict[str, str]) -> str:
    """Create a base64-encoded tar archive from a dict of path->content."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:") as tf:
        for name, content in file_dict.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Tests: _batch_read_files
# ---------------------------------------------------------------------------


class TestBatchReadFiles:
    """Tests for CodingAgent._batch_read_files."""

    def test_empty_paths_returns_empty(self, agent):
        """Empty path list returns empty dict."""
        result = agent._batch_read_files([])
        assert result == {}

    def test_batch_read_with_real_sandbox(self, agent):
        """Batch read from a real sandbox with actual files."""
        task_id = agent.start_task("test batch", allow_degraded=True)
        assert task_id

        # Create files in sandbox
        agent._session.create_file("file_a.txt", "content alpha")
        agent._session.create_file("file_b.txt", "content beta")

        result = agent._batch_read_files(["file_a.txt", "file_b.txt"])
        assert "file_a.txt" in result
        assert "file_b.txt" in result
        assert result["file_a.txt"] == "content alpha"
        assert result["file_b.txt"] == "content beta"

        agent.abort()

    def test_batch_read_strips_workspace_prefix(self, agent):
        """Paths with /workspace/ prefix are normalized."""
        task_id = agent.start_task("test prefix", allow_degraded=True)

        agent._session.create_file("hello.py", "print('hi')")

        result = agent._batch_read_files(["/workspace/hello.py"])
        assert "/workspace/hello.py" in result
        assert result["/workspace/hello.py"] == "print('hi')"

        agent.abort()

    def test_batch_read_single_file(self, agent):
        """Batch read with a single file works correctly."""
        task_id = agent.start_task("test single", allow_degraded=True)

        agent._session.create_file("only.txt", "solo content")

        result = agent._batch_read_files(["only.txt"])
        assert len(result) == 1
        assert result["only.txt"] == "solo content"

        agent.abort()

    def test_batch_read_many_files(self, agent):
        """Batch read handles many files efficiently."""
        task_id = agent.start_task("test many", allow_degraded=True)

        file_count = 20
        for i in range(file_count):
            agent._session.create_file(f"file_{i:03d}.txt", f"content {i}")

        paths = [f"file_{i:03d}.txt" for i in range(file_count)]
        result = agent._batch_read_files(paths)

        assert len(result) == file_count
        for i in range(file_count):
            assert result[f"file_{i:03d}.txt"] == f"content {i}"

        agent.abort()

    def test_batch_read_nested_paths(self, agent):
        """Batch read handles files in subdirectories."""
        task_id = agent.start_task("test nested", allow_degraded=True)

        agent._session.create_file("src/main.py", "def main(): pass")
        agent._session.create_file("src/utils/helper.py", "def help(): pass")

        result = agent._batch_read_files([
            "src/main.py", "src/utils/helper.py"
        ])
        assert result["src/main.py"] == "def main(): pass"
        assert result["src/utils/helper.py"] == "def help(): pass"

        agent.abort()

    def test_batch_read_multiline_content(self, agent):
        """Batch read preserves multiline content."""
        task_id = agent.start_task("test multiline", allow_degraded=True)

        content = "line 1\nline 2\nline 3\n"
        agent._session.create_file("multi.txt", content)

        result = agent._batch_read_files(["multi.txt"])
        assert result["multi.txt"] == content

        agent.abort()

    def test_batch_read_special_characters(self, agent):
        """Batch read handles files with special characters in content."""
        task_id = agent.start_task("test special", allow_degraded=True)

        content = 'key = "value"\ntab\there\n'
        agent._session.create_file("special.txt", content)

        result = agent._batch_read_files(["special.txt"])
        assert result["special.txt"] == content

        agent.abort()

    def test_batch_read_empty_file(self, agent):
        """Batch read handles empty files."""
        task_id = agent.start_task("test empty", allow_degraded=True)

        agent._session.create_file("empty.txt", "")

        result = agent._batch_read_files(["empty.txt"])
        assert "empty.txt" in result
        assert result["empty.txt"] == ""

        agent.abort()


class TestBatchReadTar:
    """Tests for the tar+base64 internal method."""

    def test_tar_decode_simple(self, agent):
        """_batch_read_tar decodes a simple tar archive."""
        task_id = agent.start_task("test tar", allow_degraded=True)

        agent._session.create_file("a.txt", "aaa")
        agent._session.create_file("b.txt", "bbb")

        # _batch_read_tar is called internally by _batch_read_files
        clean_paths = ["a.txt", "b.txt"]
        path_map = {"a.txt": "a.txt", "b.txt": "b.txt"}
        result = agent._batch_read_tar(clean_paths, path_map)

        assert result["a.txt"] == "aaa"
        assert result["b.txt"] == "bbb"

        agent.abort()

    def test_tar_with_dotslash_prefix(self, agent):
        """Tar members with ./ prefix are handled."""
        task_id = agent.start_task("test dotslash", allow_degraded=True)

        agent._session.create_file("test.py", "x = 1")

        clean_paths = ["test.py"]
        path_map = {"test.py": "test.py"}
        result = agent._batch_read_tar(clean_paths, path_map)

        assert "test.py" in result

        agent.abort()

    def test_tar_maps_back_to_original_path(self, agent):
        """Tar read maps clean paths back to original paths."""
        task_id = agent.start_task("test mapping", allow_degraded=True)

        agent._session.create_file("code.py", "pass")

        clean_paths = ["code.py"]
        path_map = {"code.py": "/workspace/code.py"}
        result = agent._batch_read_tar(clean_paths, path_map)

        assert "/workspace/code.py" in result

        agent.abort()


class TestBatchReadFallback:
    """Tests for the per-file fallback method."""

    def test_fallback_reads_files(self, agent):
        """Fallback reads files one by one."""
        task_id = agent.start_task("test fallback", allow_degraded=True)

        agent._session.create_file("f1.txt", "one")
        agent._session.create_file("f2.txt", "two")

        result = agent._batch_read_fallback(["f1.txt", "f2.txt"])
        assert result["f1.txt"] == "one"
        assert result["f2.txt"] == "two"

        agent.abort()

    def test_fallback_skips_missing_files(self, agent):
        """Fallback skips files that cannot be read."""
        task_id = agent.start_task("test skip", allow_degraded=True)

        agent._session.create_file("exists.txt", "here")

        result = agent._batch_read_fallback([
            "exists.txt", "missing.txt"
        ])
        assert "exists.txt" in result
        assert "missing.txt" not in result

        agent.abort()

    def test_fallback_empty_list(self, agent):
        """Fallback with empty list returns empty dict."""
        result = agent._batch_read_fallback([])
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: refactored _snapshot_originals
# ---------------------------------------------------------------------------


class TestSnapshotOriginals:
    """Tests for the refactored _snapshot_originals using batch reads."""

    def test_snapshot_captures_injected_files(self, agent, tmp_path):
        """Snapshot captures all files injected into sandbox."""
        # Create project directory
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "main.py").write_text("print('hello')")
        (proj / "utils.py").write_text("def util(): pass")

        task_id = agent.start_task(
            "test snapshot",
            project_path=str(proj),
            allow_degraded=True,
        )

        assert len(agent._original_files) == 2
        assert "main.py" in agent._original_files or any(
            "main.py" in k for k in agent._original_files
        )

        agent.abort()

    def test_snapshot_empty_project(self, agent, tmp_path):
        """Snapshot with empty project directory captures nothing."""
        proj = tmp_path / "empty_proj"
        proj.mkdir()

        task_id = agent.start_task(
            "test empty snap",
            project_path=str(proj),
            allow_degraded=True,
        )

        assert len(agent._original_files) == 0

        agent.abort()

    def test_snapshot_preserves_content(self, agent, tmp_path):
        """Snapshot preserves exact file content."""
        proj = tmp_path / "proj"
        proj.mkdir()
        content = "line1\nline2\nline3\n"
        (proj / "data.txt").write_text(content)

        task_id = agent.start_task(
            "test content",
            project_path=str(proj),
            allow_degraded=True,
        )

        matched = [v for k, v in agent._original_files.items() if "data.txt" in k]
        assert len(matched) == 1
        assert matched[0] == content

        agent.abort()


# ---------------------------------------------------------------------------
# Tests: refactored generate_diffs
# ---------------------------------------------------------------------------


class TestGenerateDiffsBatch:
    """Tests for generate_diffs using batch reads."""

    def test_diffs_detect_modification(self, agent, tmp_path):
        """Diffs detect file modifications after batch read."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "code.py").write_text("old content")

        task_id = agent.start_task(
            "test diff mod",
            project_path=str(proj),
            allow_degraded=True,
        )

        # Modify file in sandbox (injected under proj/ subdir)
        agent._session.create_file("proj/code.py", "new content")

        diffs = agent.generate_diffs()
        mod_diffs = [d for d in diffs if "code.py" in d.path and not d.is_new]
        assert len(mod_diffs) == 1
        assert not mod_diffs[0].is_deleted
        assert mod_diffs[0].modified_content == "new content"

        agent.abort()

    def test_diffs_detect_new_file(self, agent, tmp_path):
        """Diffs detect newly created files."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "existing.py").write_text("keep")

        task_id = agent.start_task(
            "test diff new",
            project_path=str(proj),
            allow_degraded=True,
        )

        agent._session.create_file("brand_new.py", "fresh code")

        diffs = agent.generate_diffs()
        new_diffs = [d for d in diffs if "brand_new.py" in d.path]
        assert len(new_diffs) == 1
        assert new_diffs[0].is_new

        agent.abort()

    def test_diffs_detect_deletion(self, agent, tmp_path):
        """Diffs detect deleted files."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "to_delete.py").write_text("bye")

        task_id = agent.start_task(
            "test diff del",
            project_path=str(proj),
            allow_degraded=True,
        )

        # Delete file via bash (injected under proj/ subdir)
        agent._session.bash("rm -f proj/to_delete.py")

        diffs = agent.generate_diffs()
        del_diffs = [d for d in diffs if "to_delete.py" in d.path]
        assert len(del_diffs) == 1
        assert del_diffs[0].is_deleted

        agent.abort()

    def test_diffs_no_changes(self, agent, tmp_path):
        """No diffs when no files changed."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "stable.py").write_text("same")

        task_id = agent.start_task(
            "test no diff",
            project_path=str(proj),
            allow_degraded=True,
        )

        diffs = agent.generate_diffs()
        assert len(diffs) == 0

        agent.abort()

    def test_diffs_integrity_hash_set(self, agent, tmp_path):
        """Diffs hash is computed after generate_diffs."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "f.py").write_text("old")

        task_id = agent.start_task(
            "test hash",
            project_path=str(proj),
            allow_degraded=True,
        )

        agent._session.create_file("proj/f.py", "new")
        agent.generate_diffs()

        assert agent._diffs_hash
        assert len(agent._diffs_hash) == 64  # SHA-256 hex

        agent.abort()

    def test_diffs_phase_set_to_reviewing(self, agent, tmp_path):
        """Phase transitions to REVIEWING during diff generation."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "x.py").write_text("x")

        task_id = agent.start_task(
            "test phase",
            project_path=str(proj),
            allow_degraded=True,
        )

        agent.generate_diffs()
        assert agent.phase == CodingPhase.REVIEWING

        agent.abort()

    def test_diffs_inactive_session(self, agent_config):
        """generate_diffs returns empty when session is not active."""
        agent = CodingAgent(config=agent_config)
        diffs = agent.generate_diffs()
        assert diffs == []
