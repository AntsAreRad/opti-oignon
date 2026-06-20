#!/usr/bin/env python3
"""
Tests for S102 -- Plugin Marketplace & Community.

Covers:
- PluginIndex: CRUD, search, list, sort, remote refresh, bulk load
- RemotePluginInstaller: URL normalization, extraction, manifest validation,
  hash verification, rollback
- PluginReviewStore: add, delete, query, summaries, sorting
- PluginTemplateGenerator: scaffold generation, hooks, files
- routes_plugin_marketplace: endpoint schemas
"""

import importlib.util
import json
import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
import textwrap
import time
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =========================================================================
# MODULE LOADING (importlib isolation)
# =========================================================================

ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, filepath: Path) -> ModuleType:
    """Load a module by file path without requiring the full package."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    # Stub opti_oignon.config if needed
    if "opti_oignon.config" not in sys.modules:
        cfg_stub = ModuleType("opti_oignon.config")
        cfg_stub.DATA_DIR = tempfile.mkdtemp()
        sys.modules["opti_oignon.config"] = cfg_stub
    if "opti_oignon" not in sys.modules:
        parent = ModuleType("opti_oignon")
        parent.__path__ = [str(ROOT / "opti_oignon")]
        sys.modules["opti_oignon"] = parent
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load modules under test
# Load plugin_manifest first (dependency for installer)
manifest_mod = _load_module(
    "opti_oignon.plugin_manifest",
    ROOT / "opti_oignon" / "plugin_manifest.py",
)

index_mod = _load_module(
    "opti_oignon.plugin_index",
    ROOT / "opti_oignon" / "plugin_index.py",
)
installer_mod = _load_module(
    "opti_oignon.plugin_installer",
    ROOT / "opti_oignon" / "plugin_installer.py",
)
reviews_mod = _load_module(
    "opti_oignon.plugin_reviews",
    ROOT / "opti_oignon" / "plugin_reviews.py",
)
template_mod = _load_module(
    "opti_oignon.plugin_template",
    ROOT / "opti_oignon" / "plugin_template.py",
)

PluginIndex = index_mod.PluginIndex
IndexEntry = index_mod.IndexEntry

RemotePluginInstaller = installer_mod.RemotePluginInstaller
PluginInstallError = installer_mod.PluginInstallError

PluginReviewStore = reviews_mod.PluginReviewStore
PluginReview = reviews_mod.PluginReview
PluginRatingSummary = reviews_mod.PluginRatingSummary
PluginReviewError = reviews_mod.PluginReviewError

PluginTemplateGenerator = template_mod.PluginTemplateGenerator


# =========================================================================
# FIXTURES
# =========================================================================

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def plugin_index(tmp_dir):
    db = tmp_dir / "test_index.db"
    return PluginIndex(db_path=db, index_url="", cache_ttl=60)


@pytest.fixture
def review_store(tmp_dir):
    db = tmp_dir / "test_reviews.db"
    return PluginReviewStore(db_path=db)


@pytest.fixture
def installer(tmp_dir):
    plugins_dir = tmp_dir / "plugins"
    plugins_dir.mkdir()
    return RemotePluginInstaller(plugins_dir=plugins_dir)


@pytest.fixture
def template_gen(tmp_dir):
    return PluginTemplateGenerator(output_base_dir=tmp_dir)


def _sample_entry(**overrides) -> dict:
    """Create a sample index entry dict."""
    base = {
        "name": "test-plugin",
        "version": "1.0.0",
        "description": "A test plugin",
        "author": "Test Author",
        "url": "https://github.com/test/test-plugin",
        "tags": ["test", "utility"],
        "hooks": ["post_inference"],
        "permissions": ["conversation_read"],
        "stars": 5,
        "downloads": 100,
        "sha256": "",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    base.update(overrides)
    return base


def _create_plugin_dir(base_dir: Path, name: str = "test-plugin") -> Path:
    """Create a minimal valid plugin directory."""
    plugin_dir = base_dir / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": name,
        "version": "1.0.0",
        "author": "Test",
        "description": "Test plugin",
        "entry_point": "entry_point.py",
        "hooks": ["post_inference"],
        "permissions": [],
    }
    try:
        import yaml
        (plugin_dir / "manifest.yaml").write_text(
            yaml.dump(manifest), encoding="utf-8",
        )
    except ImportError:
        # Fallback: write YAML manually
        lines = [
            f"name: \"{name}\"",
            "version: \"1.0.0\"",
            "author: \"Test\"",
            "description: \"Test plugin\"",
            "entry_point: \"entry_point.py\"",
            "hooks:",
            "  - post_inference",
            "permissions: []",
        ]
        (plugin_dir / "manifest.yaml").write_text(
            "\n".join(lines), encoding="utf-8",
        )

    (plugin_dir / "entry_point.py").write_text(
        textwrap.dedent("""\
            def init():
                pass
            def shutdown():
                pass
            def hook_post_inference(data):
                return data
            HOOKS = {"post_inference": hook_post_inference}
        """),
        encoding="utf-8",
    )
    return plugin_dir


def _create_plugin_zip(base_dir: Path, name: str = "test-plugin") -> Path:
    """Create a zip archive of a valid plugin directory."""
    plugin_dir = _create_plugin_dir(base_dir, name)
    zip_path = base_dir / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fpath in plugin_dir.rglob("*"):
            if fpath.is_file():
                arcname = f"{name}/{fpath.relative_to(plugin_dir)}"
                zf.write(fpath, arcname)
    return zip_path


# =========================================================================
# PLUGIN INDEX TESTS
# =========================================================================

class TestIndexEntry:
    """Tests for IndexEntry dataclass."""

    def test_from_dict_basic(self):
        entry = IndexEntry.from_dict(_sample_entry())
        assert entry.name == "test-plugin"
        assert entry.version == "1.0.0"
        assert entry.author == "Test Author"
        assert entry.tags == ["test", "utility"]

    def test_from_dict_defaults(self):
        entry = IndexEntry.from_dict({"name": "minimal"})
        assert entry.version == "0.0.0"
        assert entry.description == ""
        assert entry.tags == []
        assert entry.stars == 0

    def test_to_dict_roundtrip(self):
        data = _sample_entry()
        entry = IndexEntry.from_dict(data)
        d = entry.to_dict()
        assert d["name"] == data["name"]
        assert d["tags"] == data["tags"]
        assert d["hooks"] == data["hooks"]


class TestPluginIndex:
    """Tests for PluginIndex class."""

    def test_upsert_and_get(self, plugin_index):
        entry = IndexEntry.from_dict(_sample_entry())
        plugin_index.upsert(entry)
        result = plugin_index.get("test-plugin")
        assert result is not None
        assert result.name == "test-plugin"
        assert result.version == "1.0.0"

    def test_get_nonexistent(self, plugin_index):
        assert plugin_index.get("nonexistent") is None

    def test_remove(self, plugin_index):
        entry = IndexEntry.from_dict(_sample_entry())
        plugin_index.upsert(entry)
        assert plugin_index.remove("test-plugin") is True
        assert plugin_index.get("test-plugin") is None

    def test_remove_nonexistent(self, plugin_index):
        assert plugin_index.remove("nonexistent") is False

    def test_count(self, plugin_index):
        assert plugin_index.count == 0
        plugin_index.upsert(IndexEntry.from_dict(_sample_entry(name="p-one")))
        plugin_index.upsert(IndexEntry.from_dict(_sample_entry(name="p-two")))
        assert plugin_index.count == 2

    def test_list_all_sorted_by_name(self, plugin_index):
        plugin_index.upsert(IndexEntry.from_dict(_sample_entry(name="z-plugin")))
        plugin_index.upsert(IndexEntry.from_dict(_sample_entry(name="a-plugin")))
        results = plugin_index.list_all(sort_by="name")
        assert results[0].name == "a-plugin"
        assert results[1].name == "z-plugin"

    def test_list_all_sorted_by_stars(self, plugin_index):
        plugin_index.upsert(IndexEntry.from_dict(_sample_entry(name="low-star", stars=1)))
        plugin_index.upsert(IndexEntry.from_dict(_sample_entry(name="high-star", stars=99)))
        results = plugin_index.list_all(sort_by="stars")
        assert results[0].name == "high-star"

    def test_list_all_pagination(self, plugin_index):
        for i in range(5):
            plugin_index.upsert(IndexEntry.from_dict(
                _sample_entry(name=f"plugin-{i:02d}")
            ))
        page1 = plugin_index.list_all(sort_by="name", limit=2, offset=0)
        page2 = plugin_index.list_all(sort_by="name", limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].name != page2[0].name

    def test_search_by_keyword(self, plugin_index):
        plugin_index.upsert(IndexEntry.from_dict(
            _sample_entry(name="markdown-helper", description="Helps with markdown")
        ))
        plugin_index.upsert(IndexEntry.from_dict(
            _sample_entry(name="code-runner", description="Runs code")
        ))
        results = plugin_index.search(keyword="markdown")
        assert len(results) == 1
        assert results[0].name == "markdown-helper"

    def test_search_by_tag(self, plugin_index):
        plugin_index.upsert(IndexEntry.from_dict(
            _sample_entry(name="tagged-one", tags=["nlp", "text"])
        ))
        plugin_index.upsert(IndexEntry.from_dict(
            _sample_entry(name="tagged-two", tags=["image", "vision"])
        ))
        results = plugin_index.search(tag="nlp")
        assert len(results) == 1
        assert results[0].name == "tagged-one"

    def test_search_by_author(self, plugin_index):
        plugin_index.upsert(IndexEntry.from_dict(
            _sample_entry(name="p-alice", author="Alice")
        ))
        plugin_index.upsert(IndexEntry.from_dict(
            _sample_entry(name="p-bob", author="Bob")
        ))
        results = plugin_index.search(author="alice")
        assert len(results) == 1
        assert results[0].name == "p-alice"

    def test_search_by_hook(self, plugin_index):
        plugin_index.upsert(IndexEntry.from_dict(
            _sample_entry(name="hook-pre", hooks=["pre_prompt"])
        ))
        plugin_index.upsert(IndexEntry.from_dict(
            _sample_entry(name="hook-post", hooks=["post_inference"])
        ))
        results = plugin_index.search(hook="pre_prompt")
        assert len(results) == 1
        assert results[0].name == "hook-pre"

    def test_search_combined(self, plugin_index):
        plugin_index.upsert(IndexEntry.from_dict(
            _sample_entry(name="combo-match", author="Alice", tags=["nlp"])
        ))
        plugin_index.upsert(IndexEntry.from_dict(
            _sample_entry(name="no-match", author="Bob", tags=["image"])
        ))
        results = plugin_index.search(author="alice", tag="nlp")
        assert len(results) == 1
        assert results[0].name == "combo-match"

    def test_load_from_json(self, plugin_index):
        entries = [
            _sample_entry(name="bulk-one"),
            _sample_entry(name="bulk-two"),
            {"invalid": True},  # Should be skipped
        ]
        count = plugin_index.load_from_json(entries)
        assert count == 2
        assert plugin_index.count == 2

    def test_increment_downloads(self, plugin_index):
        plugin_index.upsert(IndexEntry.from_dict(
            _sample_entry(name="dl-counter", downloads=10)
        ))
        assert plugin_index.increment_downloads("dl-counter") is True
        entry = plugin_index.get("dl-counter")
        assert entry.downloads == 11

    def test_increment_downloads_nonexistent(self, plugin_index):
        assert plugin_index.increment_downloads("ghost") is False

    def test_is_stale_initial(self, plugin_index):
        assert plugin_index.is_stale is True

    def test_is_stale_after_refresh_no_url(self, plugin_index):
        # No URL configured, refresh does nothing but does not error
        count = plugin_index.refresh_from_remote()
        assert count == 0


# =========================================================================
# REMOTE INSTALLER TESTS
# =========================================================================

class TestRemotePluginInstaller:
    """Tests for RemotePluginInstaller class."""

    def test_normalize_github_url(self, installer):
        url = "https://github.com/user/my-plugin"
        result = installer._normalize_url(url)
        assert result == "https://github.com/user/my-plugin/archive/refs/heads/main.zip"

    def test_normalize_github_url_with_git(self, installer):
        url = "https://github.com/user/my-plugin.git"
        result = installer._normalize_url(url)
        assert result == "https://github.com/user/my-plugin/archive/refs/heads/main.zip"

    def test_normalize_github_url_trailing_slash(self, installer):
        url = "https://github.com/user/my-plugin/"
        result = installer._normalize_url(url)
        assert result == "https://github.com/user/my-plugin/archive/refs/heads/main.zip"

    def test_normalize_direct_zip_url(self, installer):
        url = "https://example.com/plugin.zip"
        result = installer._normalize_url(url)
        assert result == url

    def test_normalize_github_archive_url_untouched(self, installer):
        url = "https://github.com/user/repo/archive/refs/heads/main.zip"
        result = installer._normalize_url(url)
        assert result == url

    def test_compute_sha256(self, tmp_dir):
        test_file = tmp_dir / "test.bin"
        content = b"hello world"
        test_file.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        result = RemotePluginInstaller._compute_sha256(test_file)
        assert result == expected

    def test_find_plugin_root_direct(self, installer, tmp_dir):
        plugin_dir = _create_plugin_dir(tmp_dir, "direct-plugin")
        result = installer._find_plugin_root(plugin_dir)
        # The manifest is inside plugin_dir itself
        assert result is not None

    def test_find_plugin_root_nested(self, installer, tmp_dir):
        # Simulate GitHub archive: extract_dir/repo-main/manifest.yaml
        extract_dir = tmp_dir / "extract"
        extract_dir.mkdir()
        inner = extract_dir / "repo-main"
        _create_plugin_dir(inner.parent, inner.name)
        result = installer._find_plugin_root(extract_dir)
        assert result is not None
        assert (result / "manifest.yaml").exists()

    def test_find_plugin_root_none(self, installer, tmp_dir):
        empty = tmp_dir / "empty"
        empty.mkdir()
        result = installer._find_plugin_root(empty)
        assert result is None

    def test_extract_zip(self, installer, tmp_dir):
        zip_path = _create_plugin_zip(tmp_dir, "zip-test")
        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir()
        installer._extract(zip_path, extract_dir)
        # Should find files inside
        assert any(extract_dir.rglob("manifest.yaml"))

    def test_validate_manifest_valid(self, installer, tmp_dir):
        plugin_dir = _create_plugin_dir(tmp_dir, "valid-plugin")
        result = installer._validate_manifest(plugin_dir)
        assert result["name"] == "valid-plugin"
        assert result["version"] == "1.0.0"

    def test_validate_manifest_missing_entry_point(self, installer, tmp_dir):
        plugin_dir = tmp_dir / "bad-plugin"
        plugin_dir.mkdir()
        try:
            import yaml
            manifest = {
                "name": "bad-plugin",
                "version": "1.0.0",
                "author": "Test",
                "description": "Bad",
                "entry_point": "nonexistent.py",
                "hooks": [],
                "permissions": [],
            }
            (plugin_dir / "manifest.yaml").write_text(
                yaml.dump(manifest), encoding="utf-8",
            )
        except ImportError:
            pytest.skip("PyYAML required")

        with pytest.raises(PluginInstallError, match="Entry point file not found"):
            installer._validate_manifest(plugin_dir)

    def test_install_from_url_no_url(self, installer):
        result = installer.install_from_url("")
        assert result["success"] is False

    def test_hash_mismatch(self, installer, tmp_dir):
        # Create a valid zip
        zip_path = _create_plugin_zip(tmp_dir, "hash-test")

        # Mock download to return our zip
        def mock_download(url, dest_dir):
            dest = dest_dir / "hash-test.zip"
            shutil.copy2(zip_path, dest)
            return dest

        installer._download = mock_download
        result = installer.install_from_url(
            "https://example.com/hash-test.zip",
            expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        )
        assert result["success"] is False
        assert "Hash mismatch" in result["error"]


# =========================================================================
# REVIEW STORE TESTS
# =========================================================================

class TestPluginReviewStore:
    """Tests for PluginReviewStore class."""

    def test_add_review(self, review_store):
        review = review_store.add_review("my-plugin", 4, title="Good", text="Works well")
        assert review.id > 0
        assert review.plugin_name == "my-plugin"
        assert review.rating == 4
        assert review.title == "Good"
        assert review.author == "anonymous"

    def test_add_review_with_author(self, review_store):
        review = review_store.add_review("my-plugin", 5, author="Alice")
        assert review.author == "Alice"

    def test_add_review_invalid_rating_low(self, review_store):
        with pytest.raises(PluginReviewError, match="Rating must be"):
            review_store.add_review("my-plugin", 0)

    def test_add_review_invalid_rating_high(self, review_store):
        with pytest.raises(PluginReviewError, match="Rating must be"):
            review_store.add_review("my-plugin", 6)

    def test_add_review_empty_name(self, review_store):
        with pytest.raises(PluginReviewError, match="Plugin name is required"):
            review_store.add_review("", 3)

    def test_get_reviews(self, review_store):
        review_store.add_review("p1", 5, title="Great")
        review_store.add_review("p1", 3, title="OK")
        review_store.add_review("p2", 4, title="Other")
        reviews = review_store.get_reviews("p1")
        assert len(reviews) == 2
        # Default sort: created_at DESC, so most recent first
        assert reviews[0].title == "OK"

    def test_get_reviews_sorted_by_rating(self, review_store):
        review_store.add_review("p1", 2, title="Low")
        review_store.add_review("p1", 5, title="High")
        reviews = review_store.get_reviews("p1", sort_by="rating")
        assert reviews[0].title == "High"

    def test_get_rating_summary(self, review_store):
        review_store.add_review("p1", 5)
        review_store.add_review("p1", 4)
        review_store.add_review("p1", 5)
        summary = review_store.get_rating_summary("p1")
        assert summary.review_count == 3
        assert abs(summary.average_rating - 4.666) < 0.01
        assert summary.rating_distribution[5] == 2
        assert summary.rating_distribution[4] == 1
        assert summary.rating_distribution[1] == 0

    def test_get_rating_summary_empty(self, review_store):
        summary = review_store.get_rating_summary("no-reviews")
        assert summary.review_count == 0
        assert summary.average_rating == 0.0

    def test_delete_review(self, review_store):
        review = review_store.add_review("p1", 4)
        assert review_store.delete_review(review.id) is True
        assert review_store.total_reviews == 0

    def test_delete_review_nonexistent(self, review_store):
        assert review_store.delete_review(9999) is False

    def test_delete_reviews_for_plugin(self, review_store):
        review_store.add_review("p1", 5)
        review_store.add_review("p1", 3)
        review_store.add_review("p2", 4)
        deleted = review_store.delete_reviews_for_plugin("p1")
        assert deleted == 2
        assert review_store.total_reviews == 1

    def test_get_top_rated(self, review_store):
        review_store.add_review("low-rated", 1)
        review_store.add_review("high-rated", 5)
        review_store.add_review("high-rated", 5)
        top = review_store.get_top_rated(limit=5)
        assert len(top) == 2
        assert top[0]["plugin_name"] == "high-rated"
        assert top[0]["average_rating"] == 5.0

    def test_get_most_reviewed(self, review_store):
        review_store.add_review("popular", 4)
        review_store.add_review("popular", 3)
        review_store.add_review("popular", 5)
        review_store.add_review("niche", 5)
        most = review_store.get_most_reviewed(limit=5)
        assert most[0]["plugin_name"] == "popular"
        assert most[0]["review_count"] == 3

    def test_get_recent_reviews(self, review_store):
        review_store.add_review("p1", 3, title="First")
        review_store.add_review("p2", 4, title="Second")
        recent = review_store.get_recent_reviews(limit=5)
        assert len(recent) == 2
        assert recent[0].title == "Second"  # Most recent first

    def test_total_reviews(self, review_store):
        assert review_store.total_reviews == 0
        review_store.add_review("p1", 4)
        review_store.add_review("p2", 5)
        assert review_store.total_reviews == 2

    def test_review_to_dict(self, review_store):
        review = review_store.add_review("p1", 4, title="Nice", text="Body", author="Bob")
        d = review.to_dict()
        assert d["plugin_name"] == "p1"
        assert d["rating"] == 4
        assert d["title"] == "Nice"
        assert d["author"] == "Bob"

    def test_rating_summary_to_dict(self, review_store):
        review_store.add_review("p1", 5)
        summary = review_store.get_rating_summary("p1")
        d = summary.to_dict()
        assert d["average_rating"] == 5.0
        assert d["review_count"] == 1
        assert isinstance(d["rating_distribution"], dict)


# =========================================================================
# TEMPLATE GENERATOR TESTS
# =========================================================================

class TestPluginTemplateGenerator:
    """Tests for PluginTemplateGenerator class."""

    def test_generate_creates_files(self, template_gen, tmp_dir):
        result = template_gen.generate(
            "my-test-plugin",
            output_dir=tmp_dir / "my-test-plugin",
        )
        assert result["success"] is True
        assert len(result["files"]) == 3
        assert "manifest.yaml" in result["files"]
        assert "entry_point.py" in result["files"]
        assert "README.md" in result["files"]

    def test_generate_manifest_content(self, template_gen, tmp_dir):
        out = tmp_dir / "gen-test"
        template_gen.generate("gen-test", author="Alice", output_dir=out)
        manifest = (out / "manifest.yaml").read_text(encoding="utf-8")
        assert 'name: "gen-test"' in manifest
        assert 'author: "Alice"' in manifest
        assert "post_inference" in manifest

    def test_generate_entry_point_hooks(self, template_gen, tmp_dir):
        out = tmp_dir / "hook-test"
        template_gen.generate(
            "hook-test",
            hooks=["pre_prompt", "post_inference"],
            output_dir=out,
        )
        code = (out / "entry_point.py").read_text(encoding="utf-8")
        assert "def hook_pre_prompt" in code
        assert "def hook_post_inference" in code
        assert '"pre_prompt": hook_pre_prompt' in code
        assert '"post_inference": hook_post_inference' in code

    def test_generate_readme_content(self, template_gen, tmp_dir):
        out = tmp_dir / "readme-test"
        template_gen.generate(
            "readme-test",
            author="Bob",
            description="A readme test plugin.",
            output_dir=out,
        )
        readme = (out / "README.md").read_text(encoding="utf-8")
        assert "# readme-test" in readme
        assert "Bob" in readme
        assert "A readme test plugin." in readme

    def test_generate_with_defaults(self, template_gen, tmp_dir):
        out = tmp_dir / "default-test"
        result = template_gen.generate("default-test", output_dir=out)
        assert result["success"] is True
        code = (out / "entry_point.py").read_text(encoding="utf-8")
        # Default hook is post_inference
        assert "def hook_post_inference" in code

    def test_generate_uses_output_base(self, template_gen):
        result = template_gen.generate("base-dir-test")
        assert result["success"] is True
        assert "base-dir-test" in result["path"]
        # Cleanup
        shutil.rmtree(result["path"], ignore_errors=True)

    def test_available_hooks(self):
        hooks = PluginTemplateGenerator.available_hooks()
        assert len(hooks) == 7
        names = [h["name"] for h in hooks]
        assert "pre_prompt" in names
        assert "post_inference" in names
        assert "tool_call" in names

    def test_available_permissions(self):
        perms = PluginTemplateGenerator.available_permissions()
        assert "conversation_read" in perms
        assert "network_outbound" in perms
        assert len(perms) == 9

    def test_generate_entry_point_has_init_shutdown(self, template_gen, tmp_dir):
        out = tmp_dir / "lifecycle-test"
        template_gen.generate("lifecycle-test", output_dir=out)
        code = (out / "entry_point.py").read_text(encoding="utf-8")
        assert "def init():" in code
        assert "def shutdown():" in code

    def test_generate_invalid_dir_returns_error(self, template_gen):
        # Write to a read-only path (unlikely to succeed)
        result = template_gen.generate(
            "fail-test",
            output_dir=Path("/proc/nonexistent/fail-test"),
        )
        assert result["success"] is False
        assert result["error"] is not None


# =========================================================================
# ROUTES SCHEMA TESTS
# =========================================================================

class TestRoutesSchemas:
    """Test that route Pydantic models can be imported and instantiated."""

    def test_import_routes_module(self):
        routes_mod = _load_module(
            "opti_oignon.api.routes_plugin_marketplace",
            ROOT / "opti_oignon" / "api" / "routes_plugin_marketplace.py",
        )
        assert hasattr(routes_mod, "router")

    def test_index_entry_response_schema(self):
        routes_mod = sys.modules.get(
            "opti_oignon.api.routes_plugin_marketplace"
        )
        if routes_mod is None:
            pytest.skip("Routes module not loaded")
        resp = routes_mod.IndexEntryResponse(
            name="test", version="1.0.0", description="d", author="a",
        )
        assert resp.name == "test"
        assert resp.average_rating == 0.0

    def test_remote_install_request_schema(self):
        routes_mod = sys.modules.get(
            "opti_oignon.api.routes_plugin_marketplace"
        )
        if routes_mod is None:
            pytest.skip("Routes module not loaded")
        req = routes_mod.RemoteInstallRequest(url="https://example.com/p.zip")
        assert req.url == "https://example.com/p.zip"
        assert req.auto_enable is False

    def test_template_request_schema(self):
        routes_mod = sys.modules.get(
            "opti_oignon.api.routes_plugin_marketplace"
        )
        if routes_mod is None:
            pytest.skip("Routes module not loaded")
        req = routes_mod.TemplateRequest(name="my-plugin")
        assert req.name == "my-plugin"
        assert req.hooks == ["post_inference"]
        assert req.permissions == []

    def test_add_review_request_schema(self):
        routes_mod = sys.modules.get(
            "opti_oignon.api.routes_plugin_marketplace"
        )
        if routes_mod is None:
            pytest.skip("Routes module not loaded")
        req = routes_mod.AddReviewRequest(rating=4, title="Nice")
        assert req.rating == 4
        assert req.author == "anonymous"
