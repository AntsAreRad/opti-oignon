#!/usr/bin/env python3
"""
Tests for the Projects module (S57).

Covers ProjectStore CRUD, file management, output management,
conversation linking, cascade deletion, validation, config loading,
file type detection, and API endpoint integration.

Target: 70+ tests, 0 regressions.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test data."""
    d = tempfile.mkdtemp(prefix="opti_projects_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def config_path(tmp_dir):
    """Create a test config file."""
    config = {
        "projects": {
            "enabled": True,
            "storage_path": "projects",
            "max_projects": 5,
            "max_files_per_project": 10,
            "max_file_size_mb": 1,
            "allowed_extensions": [
                ".txt", ".py", ".md", ".csv", ".json",
                ".pdf", ".png", ".zip",
            ],
            "default_settings": {
                "default_model": "",
                "default_pipeline": "direct",
                "context_budget_tokens": 4096,
                "auto_index": True,
            },
            "file_type_categories": {
                "text": [".txt", ".md"],
                "code": [".py", ".json"],
                "data": [".csv"],
                "document": [".pdf"],
                "image": [".png"],
                "archive": [".zip"],
            },
        }
    }
    p = tmp_dir / "projects.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)
    return p


@pytest.fixture
def store(tmp_dir, config_path):
    """Create a fresh ProjectStore with temp paths."""
    from opti_oignon.projects import ProjectStore
    db_path = tmp_dir / "test_projects.db"
    storage_base = tmp_dir / "storage"
    return ProjectStore(
        db_path=db_path,
        config_path=config_path,
        storage_base=storage_base,
    )


@pytest.fixture
def project(store):
    """Create a sample project."""
    return store.create_project(
        name="Test Project",
        description="A test project for unit tests",
    )


# =============================================================================
# DATA CLASSES
# =============================================================================

class TestProjectDataclass:
    """Tests for the Project dataclass."""

    def test_auto_id_generation(self):
        from opti_oignon.projects import Project
        p = Project(name="Test")
        assert p.id != ""
        assert len(p.id) > 0

    def test_auto_timestamps(self):
        from opti_oignon.projects import Project
        p = Project(name="Test")
        assert p.created_at != ""
        assert p.updated_at != ""

    def test_to_dict(self):
        from opti_oignon.projects import Project
        p = Project(name="Test", description="Desc", settings={"key": "val"})
        d = p.to_dict()
        assert d["name"] == "Test"
        assert d["description"] == "Desc"
        assert d["settings"] == {"key": "val"}

    def test_from_dict_basic(self):
        from opti_oignon.projects import Project
        data = {"id": "abc123", "name": "FromDict", "description": "D"}
        p = Project.from_dict(data)
        assert p.id == "abc123"
        assert p.name == "FromDict"

    def test_from_dict_settings_json_string(self):
        from opti_oignon.projects import Project
        data = {"name": "Test", "settings": '{"model": "qwen3"}'}
        p = Project.from_dict(data)
        assert p.settings == {"model": "qwen3"}

    def test_from_dict_settings_invalid_json(self):
        from opti_oignon.projects import Project
        data = {"name": "Test", "settings": "not-json"}
        p = Project.from_dict(data)
        assert p.settings == {}

    def test_from_dict_ignores_unknown_keys(self):
        from opti_oignon.projects import Project
        data = {"name": "Test", "unknown_field": 42}
        p = Project.from_dict(data)
        assert p.name == "Test"
        assert not hasattr(p, "unknown_field")


class TestProjectFileDataclass:
    """Tests for the ProjectFile dataclass."""

    def test_auto_id_generation(self):
        from opti_oignon.projects import ProjectFile
        pf = ProjectFile(project_id="p1", filename="test.py")
        assert pf.id != ""

    def test_auto_timestamps(self):
        from opti_oignon.projects import ProjectFile
        pf = ProjectFile(project_id="p1", filename="test.py")
        assert pf.uploaded_at != ""
        assert pf.updated_at != ""

    def test_from_dict_key_terms_json(self):
        from opti_oignon.projects import ProjectFile
        data = {"project_id": "p1", "filename": "f.py", "key_terms": '["a","b"]'}
        pf = ProjectFile.from_dict(data)
        assert pf.key_terms == ["a", "b"]

    def test_from_dict_indexed_int_to_bool(self):
        from opti_oignon.projects import ProjectFile
        data = {"project_id": "p1", "filename": "f.py", "indexed": 1}
        pf = ProjectFile.from_dict(data)
        assert pf.indexed is True

    def test_to_dict(self):
        from opti_oignon.projects import ProjectFile
        pf = ProjectFile(project_id="p1", filename="test.py", file_type="code")
        d = pf.to_dict()
        assert d["filename"] == "test.py"
        assert d["file_type"] == "code"


class TestProjectOutputDataclass:
    """Tests for the ProjectOutput dataclass."""

    def test_auto_id_generation(self):
        from opti_oignon.projects import ProjectOutput
        po = ProjectOutput(project_id="p1", filename="out.csv")
        assert po.id != ""

    def test_from_dict(self):
        from opti_oignon.projects import ProjectOutput
        data = {"project_id": "p1", "filename": "out.csv", "output_type": "data"}
        po = ProjectOutput.from_dict(data)
        assert po.output_type == "data"

    def test_from_dict_ignores_unknown(self):
        from opti_oignon.projects import ProjectOutput
        data = {"project_id": "p1", "filename": "out.csv", "extra": True}
        po = ProjectOutput.from_dict(data)
        assert not hasattr(po, "extra")


# =============================================================================
# FILE TYPE DETECTION
# =============================================================================

class TestFileTypeDetection:
    """Tests for detect_file_type helper."""

    def test_detect_python(self, store):
        from opti_oignon.projects import detect_file_type
        assert detect_file_type("script.py") == "code"

    def test_detect_csv(self, store):
        from opti_oignon.projects import detect_file_type
        assert detect_file_type("data.csv") == "data"

    def test_detect_pdf(self, store):
        from opti_oignon.projects import detect_file_type
        assert detect_file_type("report.pdf") == "document"

    def test_detect_image(self, store):
        from opti_oignon.projects import detect_file_type
        assert detect_file_type("photo.png") == "image"

    def test_detect_text(self, store):
        from opti_oignon.projects import detect_file_type
        assert detect_file_type("readme.md") == "text"

    def test_detect_archive(self, store):
        from opti_oignon.projects import detect_file_type
        assert detect_file_type("bundle.zip") == "archive"

    def test_detect_unknown(self, store):
        from opti_oignon.projects import detect_file_type
        assert detect_file_type("weird.xyz") == "unknown"

    def test_detect_case_insensitive(self, store):
        from opti_oignon.projects import detect_file_type
        assert detect_file_type("SCRIPT.PY") == "code"

    def test_detect_no_extension(self, store):
        from opti_oignon.projects import detect_file_type
        assert detect_file_type("Makefile") == "unknown"


# =============================================================================
# FILE METADATA EXTRACTION
# =============================================================================

class TestFileMetadataExtraction:
    """Tests for extract_file_metadata helper."""

    def test_text_file_line_count(self, tmp_dir, store):
        from opti_oignon.projects import extract_file_metadata
        fp = tmp_dir / "test.py"
        fp.write_text("line1\nline2\nline3\n")
        meta = extract_file_metadata(fp)
        assert meta["size_bytes"] > 0
        assert meta["line_count"] == 3

    def test_binary_file_no_lines(self, tmp_dir, store):
        from opti_oignon.projects import extract_file_metadata
        fp = tmp_dir / "image.png"
        fp.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        meta = extract_file_metadata(fp)
        assert meta["size_bytes"] > 0
        assert meta["line_count"] is None

    def test_nonexistent_file(self, store):
        from opti_oignon.projects import extract_file_metadata
        meta = extract_file_metadata(Path("/nonexistent/file.txt"))
        assert meta["size_bytes"] == 0


# =============================================================================
# PROJECT STORE: CONFIG
# =============================================================================

class TestProjectStoreConfig:
    """Tests for ProjectStore configuration loading."""

    def test_config_loaded(self, store):
        assert store.enabled is True
        assert store.max_projects == 5
        assert store.max_files_per_project == 10
        assert store.max_file_size_mb == 1

    def test_allowed_extensions(self, store):
        exts = store.allowed_extensions
        assert ".py" in exts
        assert ".txt" in exts

    def test_default_settings(self, store):
        ds = store.default_settings
        assert "default_pipeline" in ds
        assert ds["context_budget_tokens"] == 4096

    def test_max_file_size_bytes(self, store):
        assert store.max_file_size_bytes == 1 * 1024 * 1024

    def test_missing_config_uses_defaults(self, tmp_dir):
        from opti_oignon.projects import ProjectStore
        s = ProjectStore(
            db_path=tmp_dir / "db.sqlite",
            config_path=tmp_dir / "nonexistent.yaml",
            storage_base=tmp_dir / "storage",
        )
        assert s.enabled is True
        assert s.max_projects == 50


# =============================================================================
# PROJECT STORE: PROJECT CRUD
# =============================================================================

class TestProjectCRUD:
    """Tests for project create/read/update/delete."""

    def test_create_project(self, store):
        p = store.create_project("My Project", description="Desc")
        assert p.name == "My Project"
        assert p.description == "Desc"
        assert p.id != ""
        assert p.created_at != ""

    def test_create_project_with_settings(self, store):
        p = store.create_project("S", settings={"default_model": "qwen3:32b"})
        assert p.settings["default_model"] == "qwen3:32b"
        # Defaults should still be present
        assert "default_pipeline" in p.settings

    def test_create_project_with_system_instructions(self, store):
        p = store.create_project("S", system_instructions="You are a bioinformatician.")
        assert p.system_instructions == "You are a bioinformatician."

    def test_create_project_empty_name_fails(self, store):
        with pytest.raises(ValueError, match="empty"):
            store.create_project("")

    def test_create_project_whitespace_name_fails(self, store):
        with pytest.raises(ValueError, match="empty"):
            store.create_project("   ")

    def test_create_project_strips_whitespace(self, store):
        p = store.create_project("  Trimmed  ", description="  Also trimmed  ")
        assert p.name == "Trimmed"
        assert p.description == "Also trimmed"

    def test_create_project_limit(self, store):
        for i in range(5):
            store.create_project(f"Project {i}")
        with pytest.raises(ValueError, match="Maximum"):
            store.create_project("One too many")

    def test_create_project_creates_directories(self, store):
        p = store.create_project("DirTest")
        project_dir = store._storage_base / p.id
        assert project_dir.exists()
        assert (project_dir / "files").exists()
        assert (project_dir / "outputs").exists()

    def test_get_project(self, store, project):
        got = store.get_project(project.id)
        assert got is not None
        assert got.name == "Test Project"
        assert got.description == "A test project for unit tests"

    def test_get_project_not_found(self, store):
        assert store.get_project("nonexistent") is None

    def test_list_projects(self, store):
        store.create_project("A")
        store.create_project("B")
        projects = store.list_projects()
        assert len(projects) == 2

    def test_list_projects_empty(self, store):
        assert store.list_projects() == []

    def test_list_projects_order_by_updated(self, store):
        p1 = store.create_project("First")
        p2 = store.create_project("Second")
        # Update p1 to make it most recent
        store.update_project(p1.id, name="First Updated")
        projects = store.list_projects()
        assert projects[0].id == p1.id

    def test_update_project_name(self, store, project):
        updated = store.update_project(project.id, name="New Name")
        assert updated.name == "New Name"

    def test_update_project_description(self, store, project):
        updated = store.update_project(project.id, description="New desc")
        assert updated.description == "New desc"

    def test_update_project_system_instructions(self, store, project):
        updated = store.update_project(
            project.id, system_instructions="New instructions"
        )
        assert updated.system_instructions == "New instructions"

    def test_update_project_settings_merge(self, store, project):
        updated = store.update_project(
            project.id, settings={"custom_key": "custom_val"}
        )
        # Original defaults should still be present
        assert "default_pipeline" in updated.settings
        # New key added
        assert updated.settings["custom_key"] == "custom_val"

    def test_update_project_empty_name_fails(self, store, project):
        with pytest.raises(ValueError, match="empty"):
            store.update_project(project.id, name="")

    def test_update_project_not_found(self, store):
        assert store.update_project("nonexistent", name="X") is None

    def test_update_project_updates_timestamp(self, store, project):
        old_updated = project.updated_at
        import time; time.sleep(0.01)
        updated = store.update_project(project.id, name="Newer")
        assert updated.updated_at >= old_updated

    def test_delete_project(self, store, project):
        assert store.delete_project(project.id) is True
        assert store.get_project(project.id) is None

    def test_delete_project_not_found(self, store):
        assert store.delete_project("nonexistent") is False

    def test_delete_project_removes_directory(self, store, project):
        project_dir = store._storage_base / project.id
        assert project_dir.exists()
        store.delete_project(project.id)
        assert not project_dir.exists()

    def test_delete_project_cascades_files(self, store, project):
        store.add_file(project.id, "test.txt", b"content")
        files_before = store.list_files(project.id)
        assert len(files_before) == 1
        store.delete_project(project.id)
        # Files table should be clean
        conn = store._get_conn()
        try:
            count = conn.execute(
                "SELECT COUNT(*) as c FROM project_files WHERE project_id = ?",
                (project.id,),
            ).fetchone()["c"]
            assert count == 0
        finally:
            conn.close()

    def test_delete_project_cascades_conversations(self, store, project):
        store.link_conversation(project.id, "conv-1")
        store.delete_project(project.id)
        conn = store._get_conn()
        try:
            count = conn.execute(
                "SELECT COUNT(*) as c FROM project_conversations WHERE project_id = ?",
                (project.id,),
            ).fetchone()["c"]
            assert count == 0
        finally:
            conn.close()


# =============================================================================
# PROJECT STORE: FILE MANAGEMENT
# =============================================================================

class TestFileManagement:
    """Tests for file add/get/list/remove/read."""

    def test_add_file(self, store, project):
        pf = store.add_file(project.id, "script.py", b"print('hello')")
        assert pf.filename == "script.py"
        assert pf.file_type == "code"
        assert pf.file_size_bytes == len(b"print('hello')")
        assert pf.project_id == project.id

    def test_add_file_writes_to_disk(self, store, project):
        pf = store.add_file(project.id, "data.csv", b"a,b\n1,2")
        fp = Path(pf.file_path)
        assert fp.exists()
        assert fp.read_bytes() == b"a,b\n1,2"

    def test_add_file_project_not_found(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.add_file("nonexistent", "f.txt", b"data")

    def test_add_file_extension_not_allowed(self, store, project):
        with pytest.raises(ValueError, match="extension not allowed"):
            store.add_file(project.id, "malware.exe", b"bad")

    def test_add_file_too_large(self, store, project):
        # max_file_size_mb = 1 in test config
        big_content = b"x" * (2 * 1024 * 1024)
        with pytest.raises(ValueError, match="exceeds maximum"):
            store.add_file(project.id, "big.txt", big_content)

    def test_add_file_limit(self, store, project):
        # max_files_per_project = 10 in test config
        for i in range(10):
            store.add_file(project.id, f"file{i}.txt", b"data")
        with pytest.raises(ValueError, match="Maximum files"):
            store.add_file(project.id, "one_more.txt", b"data")

    def test_add_file_sanitizes_filename(self, store, project):
        pf = store.add_file(project.id, "../../../etc/passwd.txt", b"data")
        assert "/" not in pf.filename
        assert ".." not in pf.filename
        assert pf.filename == "passwd.txt"

    def test_add_file_null_byte_in_filename(self, store, project):
        pf = store.add_file(project.id, "test\x00.txt", b"data")
        assert "\x00" not in pf.filename

    def test_get_file(self, store, project):
        pf = store.add_file(project.id, "f.txt", b"data")
        got = store.get_file(pf.id)
        assert got is not None
        assert got.filename == "f.txt"

    def test_get_file_not_found(self, store):
        assert store.get_file("nonexistent") is None

    def test_list_files(self, store, project):
        store.add_file(project.id, "a.txt", b"1")
        store.add_file(project.id, "b.txt", b"2")
        files = store.list_files(project.id)
        assert len(files) == 2

    def test_list_files_empty(self, store, project):
        assert store.list_files(project.id) == []

    def test_remove_file(self, store, project):
        pf = store.add_file(project.id, "temp.txt", b"data")
        fp = Path(pf.file_path)
        assert fp.exists()
        assert store.remove_file(pf.id) is True
        assert not fp.exists()
        assert store.get_file(pf.id) is None

    def test_remove_file_not_found(self, store):
        assert store.remove_file("nonexistent") is False

    def test_read_file_content(self, store, project):
        content = b"Hello, world!"
        pf = store.add_file(project.id, "hello.txt", content)
        read_back = store.read_file_content(pf.id)
        assert read_back == content

    def test_read_file_content_not_found(self, store):
        assert store.read_file_content("nonexistent") is None

    def test_add_file_touches_project(self, store, project):
        old_updated = project.updated_at
        import time; time.sleep(0.01)
        store.add_file(project.id, "new.txt", b"data")
        updated_project = store.get_project(project.id)
        assert updated_project.updated_at >= old_updated


# =============================================================================
# PROJECT STORE: OUTPUT MANAGEMENT
# =============================================================================

class TestOutputManagement:
    """Tests for output add/get/list/remove."""

    def test_add_output(self, store, project):
        po = store.add_output(
            project.id, "result.csv", b"a,b\n1,2",
            output_type="data", description="Results",
        )
        assert po.filename == "result.csv"
        assert po.output_type == "data"
        assert po.description == "Results"

    def test_add_output_writes_to_disk(self, store, project):
        po = store.add_output(project.id, "out.txt", b"output data")
        fp = Path(po.file_path)
        assert fp.exists()
        assert fp.read_bytes() == b"output data"

    def test_add_output_with_conversation_id(self, store, project):
        po = store.add_output(
            project.id, "out.py", b"code",
            source_conversation_id="conv-123",
        )
        assert po.source_conversation_id == "conv-123"

    def test_add_output_project_not_found(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.add_output("nonexistent", "f.txt", b"data")

    def test_get_output(self, store, project):
        po = store.add_output(project.id, "out.txt", b"data")
        got = store.get_output(po.id)
        assert got is not None
        assert got.filename == "out.txt"

    def test_get_output_not_found(self, store):
        assert store.get_output("nonexistent") is None

    def test_list_outputs(self, store, project):
        store.add_output(project.id, "a.txt", b"1")
        store.add_output(project.id, "b.txt", b"2")
        outputs = store.list_outputs(project.id)
        assert len(outputs) == 2

    def test_list_outputs_empty(self, store, project):
        assert store.list_outputs(project.id) == []

    def test_remove_output(self, store, project):
        po = store.add_output(project.id, "temp.txt", b"data")
        fp = Path(po.file_path)
        assert fp.exists()
        assert store.remove_output(po.id) is True
        assert not fp.exists()
        assert store.get_output(po.id) is None

    def test_remove_output_not_found(self, store):
        assert store.remove_output("nonexistent") is False

    def test_delete_project_cascades_outputs(self, store, project):
        store.add_output(project.id, "out.txt", b"data")
        store.delete_project(project.id)
        conn = store._get_conn()
        try:
            count = conn.execute(
                "SELECT COUNT(*) as c FROM project_outputs WHERE project_id = ?",
                (project.id,),
            ).fetchone()["c"]
            assert count == 0
        finally:
            conn.close()


# =============================================================================
# PROJECT STORE: CONVERSATION LINKING
# =============================================================================

class TestConversationLinking:
    """Tests for conversation link/unlink/list."""

    def test_link_conversation(self, store, project):
        result = store.link_conversation(project.id, "conv-001")
        assert result is True

    def test_link_conversation_project_not_found(self, store):
        result = store.link_conversation("nonexistent", "conv-001")
        assert result is False

    def test_link_conversation_idempotent(self, store, project):
        store.link_conversation(project.id, "conv-001")
        store.link_conversation(project.id, "conv-001")
        convs = store.list_conversations(project.id)
        assert len(convs) == 1

    def test_link_multiple_conversations(self, store, project):
        store.link_conversation(project.id, "conv-001")
        store.link_conversation(project.id, "conv-002")
        store.link_conversation(project.id, "conv-003")
        convs = store.list_conversations(project.id)
        assert len(convs) == 3

    def test_unlink_conversation(self, store, project):
        store.link_conversation(project.id, "conv-001")
        result = store.unlink_conversation(project.id, "conv-001")
        assert result is True
        convs = store.list_conversations(project.id)
        assert len(convs) == 0

    def test_unlink_conversation_not_found(self, store, project):
        result = store.unlink_conversation(project.id, "nonexistent")
        assert result is False

    def test_list_conversations(self, store, project):
        store.link_conversation(project.id, "conv-a")
        store.link_conversation(project.id, "conv-b")
        convs = store.list_conversations(project.id)
        assert len(convs) == 2
        conv_ids = {c["conversation_id"] for c in convs}
        assert "conv-a" in conv_ids
        assert "conv-b" in conv_ids

    def test_list_conversations_empty(self, store, project):
        assert store.list_conversations(project.id) == []

    def test_get_project_for_conversation(self, store, project):
        store.link_conversation(project.id, "conv-001")
        found = store.get_project_for_conversation("conv-001")
        assert found == project.id

    def test_get_project_for_conversation_not_linked(self, store):
        found = store.get_project_for_conversation("unknown")
        assert found is None


# =============================================================================
# PROJECT STORE: STATS
# =============================================================================

class TestProjectStats:
    """Tests for get_project_stats."""

    def test_empty_stats(self, store, project):
        stats = store.get_project_stats(project.id)
        assert stats["file_count"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["output_count"] == 0
        assert stats["conversation_count"] == 0

    def test_stats_with_data(self, store, project):
        store.add_file(project.id, "a.txt", b"hello")
        store.add_file(project.id, "b.txt", b"world!!")
        store.add_output(project.id, "out.txt", b"result")
        store.link_conversation(project.id, "conv-1")
        store.link_conversation(project.id, "conv-2")
        stats = store.get_project_stats(project.id)
        assert stats["file_count"] == 2
        assert stats["total_size_bytes"] == len(b"hello") + len(b"world!!")
        assert stats["output_count"] == 1
        assert stats["conversation_count"] == 2


# =============================================================================
# API ENDPOINTS
# =============================================================================

class TestProjectAPI:
    """Tests for project API endpoints via TestClient."""

    @pytest.fixture(autouse=True)
    def setup_client(self, store):
        """Patch the global project_store in routes with our test store."""
        from opti_oignon.api import routes_projects
        self._original_store = routes_projects.project_store
        self._original_available = routes_projects.PROJECTS_AVAILABLE
        routes_projects.project_store = store
        routes_projects.PROJECTS_AVAILABLE = True

        from fastapi.testclient import TestClient

        from opti_oignon.api.app import app
        self.client = TestClient(app)
        yield
        routes_projects.project_store = self._original_store
        routes_projects.PROJECTS_AVAILABLE = self._original_available

    def test_create_project(self):
        r = self.client.post("/api/projects", json={"name": "API Project"})
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "API Project"
        assert "id" in data

    def test_create_project_empty_name(self):
        r = self.client.post("/api/projects", json={"name": ""})
        assert r.status_code == 422

    def test_list_projects(self):
        self.client.post("/api/projects", json={"name": "P1"})
        self.client.post("/api/projects", json={"name": "P2"})
        r = self.client.get("/api/projects")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_get_project_detail(self):
        cr = self.client.post("/api/projects", json={"name": "Detail"})
        pid = cr.json()["id"]
        r = self.client.get(f"/api/projects/{pid}")
        assert r.status_code == 200
        data = r.json()
        assert "files" in data
        assert "outputs" in data
        assert "conversations" in data
        assert "stats" in data

    def test_get_project_not_found(self):
        r = self.client.get("/api/projects/nonexistent")
        assert r.status_code == 404

    def test_update_project(self):
        cr = self.client.post("/api/projects", json={"name": "Original"})
        pid = cr.json()["id"]
        r = self.client.put(f"/api/projects/{pid}", json={"name": "Updated"})
        assert r.status_code == 200
        assert r.json()["name"] == "Updated"

    def test_update_project_not_found(self):
        r = self.client.put("/api/projects/nonexistent", json={"name": "X"})
        assert r.status_code == 404

    def test_delete_project(self):
        cr = self.client.post("/api/projects", json={"name": "ToDelete"})
        pid = cr.json()["id"]
        r = self.client.delete(f"/api/projects/{pid}")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        # Verify gone
        r2 = self.client.get(f"/api/projects/{pid}")
        assert r2.status_code == 404

    def test_delete_project_not_found(self):
        r = self.client.delete("/api/projects/nonexistent")
        assert r.status_code == 404

    def test_upload_file(self):
        cr = self.client.post("/api/projects", json={"name": "FileTest"})
        pid = cr.json()["id"]
        r = self.client.post(
            f"/api/projects/{pid}/files",
            files={"file": ("test.py", b"print('hi')", "text/plain")},
        )
        assert r.status_code == 201
        data = r.json()
        assert data["filename"] == "test.py"
        assert data["file_type"] == "code"

    def test_upload_file_bad_extension(self):
        cr = self.client.post("/api/projects", json={"name": "FileTest2"})
        pid = cr.json()["id"]
        r = self.client.post(
            f"/api/projects/{pid}/files",
            files={"file": ("bad.exe", b"data", "application/octet-stream")},
        )
        assert r.status_code == 422

    def test_list_files(self):
        cr = self.client.post("/api/projects", json={"name": "ListFiles"})
        pid = cr.json()["id"]
        self.client.post(
            f"/api/projects/{pid}/files",
            files={"file": ("a.txt", b"1", "text/plain")},
        )
        r = self.client.get(f"/api/projects/{pid}/files")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_delete_file(self):
        cr = self.client.post("/api/projects", json={"name": "DelFile"})
        pid = cr.json()["id"]
        fr = self.client.post(
            f"/api/projects/{pid}/files",
            files={"file": ("del.txt", b"data", "text/plain")},
        )
        fid = fr.json()["id"]
        r = self.client.delete(f"/api/projects/{pid}/files/{fid}")
        assert r.status_code == 200

    def test_link_conversation(self):
        cr = self.client.post("/api/projects", json={"name": "ConvLink"})
        pid = cr.json()["id"]
        r = self.client.post(f"/api/projects/{pid}/conversations/conv-test")
        assert r.status_code == 201

    def test_unlink_conversation(self):
        cr = self.client.post("/api/projects", json={"name": "ConvUnlink"})
        pid = cr.json()["id"]
        self.client.post(f"/api/projects/{pid}/conversations/conv-test")
        r = self.client.delete(f"/api/projects/{pid}/conversations/conv-test")
        assert r.status_code == 200

    def test_list_conversations(self):
        cr = self.client.post("/api/projects", json={"name": "ConvList"})
        pid = cr.json()["id"]
        self.client.post(f"/api/projects/{pid}/conversations/c1")
        self.client.post(f"/api/projects/{pid}/conversations/c2")
        r = self.client.get(f"/api/projects/{pid}/conversations")
        assert r.status_code == 200
        assert len(r.json()["conversations"]) == 2

    def test_health_includes_projects(self):
        r = self.client.get("/api/health")
        data = r.json()
        assert "projects" in data["modules"]
