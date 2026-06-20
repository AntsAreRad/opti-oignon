#!/usr/bin/env python3
"""
Tests for S101 -- Plugin Architecture.

Covers:
- PluginManifest: validation, from_dict, to_dict, edge cases
- PluginRegistry: register, unregister, state, config, dependencies, discover
- PluginLoader: load, sandbox, lifecycle, install, uninstall
- HookManager: register, execute, priority, error isolation, stats
- Built-in plugins: calculator, code_formatter, citation_gen
- routes_plugins: endpoint schemas
"""

import importlib.util
import json
import math
import os
import sqlite3
import sys
import tempfile
import textwrap
import time
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
manifest_mod = _load_module(
    "opti_oignon.plugin_manifest",
    ROOT / "opti_oignon" / "plugin_manifest.py",
)
hooks_mod = _load_module(
    "opti_oignon.plugin_hooks",
    ROOT / "opti_oignon" / "plugin_hooks.py",
)
loader_mod = _load_module(
    "opti_oignon.plugin_loader",
    ROOT / "opti_oignon" / "plugin_loader.py",
)

PluginManifest = manifest_mod.PluginManifest
PluginManifestError = manifest_mod.PluginManifestError
PluginRegistry = manifest_mod.PluginRegistry
PluginRecord = manifest_mod.PluginRecord
VALID_HOOKS = manifest_mod.VALID_HOOKS
VALID_PERMISSIONS = manifest_mod.VALID_PERMISSIONS
PLUGIN_STATE_INSTALLED = manifest_mod.PLUGIN_STATE_INSTALLED
PLUGIN_STATE_ENABLED = manifest_mod.PLUGIN_STATE_ENABLED
PLUGIN_STATE_DISABLED = manifest_mod.PLUGIN_STATE_DISABLED

HookManager = hooks_mod.HookManager
HookContext = hooks_mod.HookContext
HookResult = hooks_mod.HookResult
HookExecutionReport = hooks_mod.HookExecutionReport

PluginLoader = loader_mod.PluginLoader
PluginLoadError = loader_mod.PluginLoadError
PluginSandboxViolation = loader_mod.PluginSandboxViolation
LoadedPlugin = loader_mod.LoadedPlugin


# =========================================================================
# FIXTURES
# =========================================================================

def _valid_manifest_dict(**overrides: Any) -> dict:
    """Return a minimal valid manifest dict."""
    base = {
        "name": "test-plugin",
        "version": "1.0.0",
        "author": "Tester",
        "description": "A test plugin",
        "entry_point": "entry_point.py",
        "hooks": ["post_inference"],
        "permissions": ["tool_register"],
        "dependencies": [],
    }
    base.update(overrides)
    return base


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temp SQLite DB path."""
    return tmp_path / "plugins.db"


@pytest.fixture
def registry(tmp_db, tmp_path):
    """Provide a fresh PluginRegistry."""
    return PluginRegistry(db_path=tmp_db, plugins_dir=tmp_path / "plugins")


@pytest.fixture
def hook_mgr():
    """Provide a fresh HookManager."""
    return HookManager()


@pytest.fixture
def plugin_dir(tmp_path):
    """Create a minimal plugin directory with manifest and entry_point."""
    pdir = tmp_path / "my-plugin"
    pdir.mkdir()
    manifest = {
        "name": "my-plugin",
        "version": "1.0.0",
        "author": "Test",
        "description": "Test plugin",
        "entry_point": "entry_point.py",
        "hooks": ["post_inference"],
        "permissions": [],
        "dependencies": [],
    }
    import yaml
    (pdir / "manifest.yaml").write_text(yaml.dump(manifest))
    (pdir / "entry_point.py").write_text(textwrap.dedent("""
        __plugin_name__ = "my-plugin"
        __plugin_version__ = "1.0.0"

        def hook_post_inference(ctx):
            return {"modified": True}

        HOOKS = {"post_inference": hook_post_inference}

        def init():
            pass

        def shutdown():
            pass
    """))
    return pdir


# =========================================================================
# MANIFEST TESTS
# =========================================================================

class TestPluginManifest:
    """Tests for PluginManifest dataclass and validation."""

    def test_from_dict_valid(self):
        m = PluginManifest.from_dict(_valid_manifest_dict())
        assert m.name == "test-plugin"
        assert m.version == "1.0.0"
        assert m.author == "Tester"
        assert m.hooks == ["post_inference"]

    def test_from_dict_minimal(self):
        data = {
            "name": "minimal",
            "version": "0.1.0",
            "author": "A",
            "description": "Minimal",
            "entry_point": "main.py",
        }
        m = PluginManifest.from_dict(data)
        assert m.hooks == []
        assert m.permissions == []
        assert m.dependencies == []

    def test_to_dict_roundtrip(self):
        original = _valid_manifest_dict()
        m = PluginManifest.from_dict(original)
        d = m.to_dict()
        assert d["name"] == "test-plugin"
        assert d["version"] == "1.0.0"
        assert d["hooks"] == ["post_inference"]

    def test_missing_name(self):
        data = _valid_manifest_dict()
        del data["name"]
        with pytest.raises(PluginManifestError, match="Missing required"):
            PluginManifest.from_dict(data)

    def test_missing_version(self):
        data = _valid_manifest_dict()
        del data["version"]
        with pytest.raises(PluginManifestError, match="Missing required"):
            PluginManifest.from_dict(data)

    def test_missing_entry_point(self):
        data = _valid_manifest_dict()
        del data["entry_point"]
        with pytest.raises(PluginManifestError, match="Missing required"):
            PluginManifest.from_dict(data)

    def test_invalid_name_uppercase(self):
        with pytest.raises(PluginManifestError, match="Invalid plugin name"):
            PluginManifest.from_dict(_valid_manifest_dict(name="MyPlugin"))

    def test_invalid_name_too_short(self):
        with pytest.raises(PluginManifestError, match="Invalid plugin name"):
            PluginManifest.from_dict(_valid_manifest_dict(name="x"))

    def test_invalid_version_format(self):
        with pytest.raises(PluginManifestError, match="Invalid version"):
            PluginManifest.from_dict(_valid_manifest_dict(version="1.0"))

    def test_version_with_prerelease(self):
        m = PluginManifest.from_dict(_valid_manifest_dict(version="1.0.0-beta.1"))
        assert m.version == "1.0.0-beta.1"

    def test_invalid_entry_point_traversal(self):
        with pytest.raises(PluginManifestError, match="Invalid entry_point"):
            PluginManifest.from_dict(_valid_manifest_dict(entry_point="../evil.py"))

    def test_invalid_entry_point_absolute(self):
        with pytest.raises(PluginManifestError, match="Invalid entry_point"):
            PluginManifest.from_dict(_valid_manifest_dict(entry_point="/etc/evil.py"))

    def test_invalid_entry_point_not_py(self):
        with pytest.raises(PluginManifestError, match="must be a .py file"):
            PluginManifest.from_dict(_valid_manifest_dict(entry_point="main.sh"))

    def test_invalid_hook(self):
        with pytest.raises(PluginManifestError, match="Invalid hooks"):
            PluginManifest.from_dict(_valid_manifest_dict(hooks=["nonexistent_hook"]))

    def test_invalid_permission(self):
        with pytest.raises(PluginManifestError, match="Invalid permissions"):
            PluginManifest.from_dict(_valid_manifest_dict(permissions=["root_access"]))

    def test_all_valid_hooks_accepted(self):
        m = PluginManifest.from_dict(_valid_manifest_dict(hooks=list(VALID_HOOKS)))
        assert set(m.hooks) == VALID_HOOKS

    def test_all_valid_permissions_accepted(self):
        m = PluginManifest.from_dict(_valid_manifest_dict(permissions=list(VALID_PERMISSIONS)))
        assert set(m.permissions) == VALID_PERMISSIONS

    def test_config_schema_preserved(self):
        schema = {"max_length": {"type": "integer", "default": 100}}
        m = PluginManifest.from_dict(_valid_manifest_dict(config_schema=schema))
        assert m.config_schema == schema


# =========================================================================
# REGISTRY TESTS
# =========================================================================

class TestPluginRegistry:
    """Tests for PluginRegistry: register, state, dependencies, discover."""

    def test_register_and_get(self, registry):
        m = PluginManifest.from_dict(_valid_manifest_dict())
        record = registry.register(m, "/tmp/test-plugin")
        assert record.manifest.name == "test-plugin"
        assert record.state == PLUGIN_STATE_INSTALLED
        assert registry.get("test-plugin") is not None

    def test_register_auto_enable(self, registry):
        m = PluginManifest.from_dict(_valid_manifest_dict())
        record = registry.register(m, "/tmp/test", auto_enable=True)
        assert record.state == PLUGIN_STATE_ENABLED

    def test_register_update_existing(self, registry):
        m1 = PluginManifest.from_dict(_valid_manifest_dict(version="1.0.0"))
        registry.register(m1, "/tmp/test")
        m2 = PluginManifest.from_dict(_valid_manifest_dict(version="1.1.0"))
        record = registry.register(m2, "/tmp/test")
        assert record.manifest.version == "1.1.0"
        assert registry.plugin_count == 1

    def test_unregister(self, registry):
        m = PluginManifest.from_dict(_valid_manifest_dict())
        registry.register(m, "/tmp/test")
        assert registry.unregister("test-plugin") is True
        assert registry.get("test-plugin") is None
        assert registry.plugin_count == 0

    def test_unregister_nonexistent(self, registry):
        assert registry.unregister("nonexistent") is False

    def test_list_plugins(self, registry):
        for i in range(3):
            m = PluginManifest.from_dict(_valid_manifest_dict(name=f"plugin-{i:02d}"))
            registry.register(m, f"/tmp/plugin-{i:02d}")
        assert len(registry.list_plugins()) == 3

    def test_list_plugins_filter_state(self, registry):
        m1 = PluginManifest.from_dict(_valid_manifest_dict(name="enabled-one"))
        registry.register(m1, "/tmp/e", auto_enable=True)
        m2 = PluginManifest.from_dict(_valid_manifest_dict(name="installed-one"))
        registry.register(m2, "/tmp/i", auto_enable=False)
        enabled = registry.list_plugins(state=PLUGIN_STATE_ENABLED)
        assert len(enabled) == 1
        assert enabled[0].manifest.name == "enabled-one"

    def test_set_state(self, registry):
        m = PluginManifest.from_dict(_valid_manifest_dict())
        registry.register(m, "/tmp/test")
        assert registry.set_state("test-plugin", PLUGIN_STATE_ENABLED)
        record = registry.get("test-plugin")
        assert record.state == PLUGIN_STATE_ENABLED

    def test_set_state_invalid(self, registry):
        m = PluginManifest.from_dict(_valid_manifest_dict())
        registry.register(m, "/tmp/test")
        with pytest.raises(ValueError, match="Invalid state"):
            registry.set_state("test-plugin", "broken")

    def test_set_config(self, registry):
        m = PluginManifest.from_dict(_valid_manifest_dict())
        registry.register(m, "/tmp/test")
        assert registry.set_config("test-plugin", {"key": "value"})
        record = registry.get("test-plugin")
        assert record.config == {"key": "value"}

    def test_version_history(self, registry):
        m = PluginManifest.from_dict(_valid_manifest_dict())
        registry.register(m, "/tmp/test")
        history = registry.get_version_history("test-plugin")
        assert len(history) >= 1
        assert history[0]["action"] == "installed"

    def test_resolve_dependencies_simple(self, registry):
        dep = PluginManifest.from_dict(_valid_manifest_dict(name="dep-plugin"))
        registry.register(dep, "/tmp/dep")
        main = PluginManifest.from_dict(
            _valid_manifest_dict(name="main-plugin", dependencies=["dep-plugin"])
        )
        registry.register(main, "/tmp/main")
        order = registry.resolve_dependencies("main-plugin")
        assert order.index("dep-plugin") < order.index("main-plugin")

    def test_resolve_dependencies_missing(self, registry):
        main = PluginManifest.from_dict(
            _valid_manifest_dict(name="main-plugin", dependencies=["missing-dep"])
        )
        registry.register(main, "/tmp/main")
        with pytest.raises(PluginManifestError, match="Missing dependency"):
            registry.resolve_dependencies("main-plugin")

    def test_resolve_dependencies_circular(self, registry):
        a = PluginManifest.from_dict(
            _valid_manifest_dict(name="plugin-a", dependencies=["plugin-b"])
        )
        b = PluginManifest.from_dict(
            _valid_manifest_dict(name="plugin-b", dependencies=["plugin-a"])
        )
        registry.register(a, "/tmp/a")
        registry.register(b, "/tmp/b")
        with pytest.raises(PluginManifestError, match="Circular dependency"):
            registry.resolve_dependencies("plugin-a")

    def test_enabled_count(self, registry):
        m1 = PluginManifest.from_dict(_valid_manifest_dict(name="en-one"))
        registry.register(m1, "/tmp/e1", auto_enable=True)
        m2 = PluginManifest.from_dict(_valid_manifest_dict(name="en-two"))
        registry.register(m2, "/tmp/e2", auto_enable=True)
        m3 = PluginManifest.from_dict(_valid_manifest_dict(name="dis-one"))
        registry.register(m3, "/tmp/d1", auto_enable=False)
        assert registry.enabled_count == 2
        assert registry.plugin_count == 3

    def test_persistence_across_instances(self, tmp_db, tmp_path):
        """Registry data survives creating a new instance."""
        r1 = PluginRegistry(db_path=tmp_db, plugins_dir=tmp_path)
        m = PluginManifest.from_dict(_valid_manifest_dict())
        r1.register(m, "/tmp/test", auto_enable=True)
        # New instance should load from DB
        r2 = PluginRegistry(db_path=tmp_db, plugins_dir=tmp_path)
        assert r2.plugin_count == 1
        record = r2.get("test-plugin")
        assert record is not None
        assert record.state == PLUGIN_STATE_ENABLED

    def test_discover_plugins(self, tmp_path):
        """Discover plugins from a directory tree."""
        import yaml
        pdir = tmp_path / "discover_root"
        for name in ["discovered-one", "discovered-two"]:
            d = pdir / name
            d.mkdir(parents=True)
            manifest = _valid_manifest_dict(name=name)
            (d / "manifest.yaml").write_text(yaml.dump(manifest))
            (d / "entry_point.py").write_text("pass")

        db = tmp_path / "disc.db"
        reg = PluginRegistry(db_path=db, plugins_dir=pdir)
        manifests = reg.discover()
        assert len(manifests) == 2
        names = {m.name for m in manifests}
        assert "discovered-one" in names
        assert "discovered-two" in names


# =========================================================================
# HOOK MANAGER TESTS
# =========================================================================

class TestHookManager:
    """Tests for HookManager: register, execute, priority, isolation."""

    def test_register_valid_hook(self, hook_mgr):
        assert hook_mgr.register("post_inference", "test", lambda ctx: None)
        assert hook_mgr.get_hook_count("post_inference") == 1

    def test_register_invalid_hook_name(self, hook_mgr):
        assert not hook_mgr.register("invalid_hook", "test", lambda ctx: None)

    def test_register_non_callable(self, hook_mgr):
        assert not hook_mgr.register("post_inference", "test", "not_callable")

    def test_unregister(self, hook_mgr):
        hook_mgr.register("post_inference", "test", lambda ctx: None)
        assert hook_mgr.unregister("post_inference", "test") == 1
        assert hook_mgr.get_hook_count("post_inference") == 0

    def test_unregister_plugin(self, hook_mgr):
        hook_mgr.register("pre_prompt", "test", lambda ctx: None)
        hook_mgr.register("post_inference", "test", lambda ctx: None)
        removed = hook_mgr.unregister_plugin("test")
        assert removed == 2
        assert hook_mgr.get_hook_count() == 0

    def test_execute_basic(self, hook_mgr):
        hook_mgr.register("post_inference", "test", lambda ctx: {"result": "ok"})
        report = hook_mgr.execute("post_inference", data={"input": "hi"})
        assert report.successful == 1
        assert report.failed == 0
        assert report.final_data["result"] == "ok"

    def test_execute_data_chaining(self, hook_mgr):
        hook_mgr.register("post_inference", "p1", lambda ctx: {"step": 1}, priority=10)
        hook_mgr.register("post_inference", "p2", lambda ctx: {"step": ctx.data.get("step", 0) + 1}, priority=20)
        report = hook_mgr.execute("post_inference")
        assert report.final_data["step"] == 2

    def test_execute_priority_order(self, hook_mgr):
        order = []
        hook_mgr.register("pre_prompt", "low", lambda ctx: order.append("low"), priority=200)
        hook_mgr.register("pre_prompt", "high", lambda ctx: order.append("high"), priority=10)
        hook_mgr.register("pre_prompt", "med", lambda ctx: order.append("med"), priority=100)
        hook_mgr.execute("pre_prompt")
        assert order == ["high", "med", "low"]

    def test_execute_error_isolation(self, hook_mgr):
        def failing_hook(ctx):
            raise RuntimeError("boom")

        hook_mgr.register("post_inference", "bad", failing_hook, priority=10)
        hook_mgr.register("post_inference", "good", lambda ctx: {"ok": True}, priority=20)
        report = hook_mgr.execute("post_inference")
        assert report.failed == 1
        assert report.successful == 1
        assert report.final_data.get("ok") is True

    def test_execute_no_hooks(self, hook_mgr):
        report = hook_mgr.execute("pre_prompt")
        assert report.total_hooks == 0
        assert report.successful == 0

    def test_has_hooks(self, hook_mgr):
        assert not hook_mgr.has_hooks("post_inference")
        hook_mgr.register("post_inference", "test", lambda ctx: None)
        assert hook_mgr.has_hooks("post_inference")

    def test_list_hooks(self, hook_mgr):
        hook_mgr.register("pre_prompt", "p1", lambda ctx: None, priority=50)
        hook_mgr.register("post_inference", "p2", lambda ctx: None, priority=100)
        all_hooks = hook_mgr.list_hooks()
        assert len(all_hooks) == 2
        filtered = hook_mgr.list_hooks(plugin_name="p1")
        assert len(filtered) == 1

    def test_set_hook_enabled(self, hook_mgr):
        hook_mgr.register("post_inference", "test", lambda ctx: {"ran": True})
        hook_mgr.set_hook_enabled("post_inference", "test", False)
        report = hook_mgr.execute("post_inference")
        # Disabled hook should not run
        assert report.successful == 0
        assert "ran" not in report.final_data

    def test_stats_tracking(self, hook_mgr):
        hook_mgr.register("post_inference", "test", lambda ctx: None)
        hook_mgr.execute("post_inference")
        hook_mgr.execute("post_inference")
        stats = hook_mgr.get_stats()
        assert "test:post_inference" in stats
        assert stats["test:post_inference"]["calls"] == 2

    def test_reset_stats(self, hook_mgr):
        hook_mgr.register("post_inference", "test", lambda ctx: None)
        hook_mgr.execute("post_inference")
        hook_mgr.reset_stats()
        assert hook_mgr.get_stats() == {}

    def test_clear(self, hook_mgr):
        hook_mgr.register("pre_prompt", "test", lambda ctx: None)
        hook_mgr.register("post_inference", "test", lambda ctx: None)
        hook_mgr.clear()
        assert hook_mgr.get_hook_count() == 0

    def test_hook_context_get_set(self):
        ctx = HookContext(hook_name="test", plugin_name="p1", data={"a": 1})
        assert ctx.get("a") == 1
        assert ctx.get("b", "default") == "default"
        ctx.set("b", 2)
        assert ctx.get("b") == 2

    def test_hook_execution_report_fields(self, hook_mgr):
        hook_mgr.register("post_inference", "test", lambda ctx: {"x": 1})
        report = hook_mgr.execute("post_inference", data={"y": 2})
        assert report.hook_name == "post_inference"
        assert report.total_hooks == 1
        assert report.total_duration_ms >= 0
        assert len(report.results) == 1
        assert report.results[0].success is True


# =========================================================================
# PLUGIN LOADER TESTS
# =========================================================================

class TestPluginLoader:
    """Tests for PluginLoader: load, sandbox, lifecycle."""

    def test_load_plugin(self, plugin_dir):
        loader = PluginLoader()
        loaded = loader.load_plugin(plugin_dir, sandbox=False)
        assert loaded.name == "my-plugin"
        assert loaded.version == "1.0.0"
        assert "post_inference" in loaded.hooks

    def test_load_plugin_hooks_callable(self, plugin_dir):
        loader = PluginLoader()
        loaded = loader.load_plugin(plugin_dir, sandbox=False)
        hook = loaded.get_hook("post_inference")
        assert callable(hook)

    def test_load_plugin_initialize(self, plugin_dir):
        loader = PluginLoader()
        loaded = loader.load_plugin(plugin_dir, sandbox=False)
        loaded.initialize()  # Should not raise

    def test_load_plugin_shutdown(self, plugin_dir):
        loader = PluginLoader()
        loaded = loader.load_plugin(plugin_dir, sandbox=False)
        loaded.initialize()
        loaded.shutdown()  # Should not raise

    def test_unload_plugin(self, plugin_dir):
        loader = PluginLoader()
        loader.load_plugin(plugin_dir, sandbox=False)
        assert "my-plugin" in loader.loaded_plugins
        assert loader.unload_plugin("my-plugin") is True
        assert "my-plugin" not in loader.loaded_plugins

    def test_unload_nonexistent(self):
        loader = PluginLoader()
        assert loader.unload_plugin("nonexistent") is False

    def test_load_missing_dir(self):
        loader = PluginLoader()
        with pytest.raises(PluginLoadError, match="not found"):
            loader.load_plugin("/nonexistent/path")

    def test_load_missing_manifest(self, tmp_path):
        pdir = tmp_path / "no-manifest"
        pdir.mkdir()
        loader = PluginLoader()
        with pytest.raises(PluginLoadError, match="No manifest.yaml"):
            loader.load_plugin(pdir)

    def test_load_missing_entry_point(self, tmp_path):
        import yaml
        pdir = tmp_path / "no-entry"
        pdir.mkdir()
        manifest = _valid_manifest_dict(name="no-entry")
        (pdir / "manifest.yaml").write_text(yaml.dump(manifest))
        loader = PluginLoader()
        with pytest.raises(PluginLoadError, match="Entry point not found"):
            loader.load_plugin(pdir)

    def test_reload_plugin(self, plugin_dir):
        """Loading an already loaded plugin should reload it."""
        loader = PluginLoader()
        loaded1 = loader.load_plugin(plugin_dir, sandbox=False)
        loaded2 = loader.load_plugin(plugin_dir, sandbox=False)
        assert loaded2.name == "my-plugin"
        assert len(loader.loaded_plugins) == 1

    def test_sandbox_blocks_subprocess(self, tmp_path):
        import yaml
        pdir = tmp_path / "evil-plugin"
        pdir.mkdir()
        manifest = _valid_manifest_dict(name="evil-plugin")
        (pdir / "manifest.yaml").write_text(yaml.dump(manifest))
        (pdir / "entry_point.py").write_text("import subprocess\n")
        loader = PluginLoader()
        with pytest.raises(PluginSandboxViolation, match="subprocess"):
            loader.load_plugin(pdir, sandbox=True)

    def test_sandbox_blocks_ctypes(self, tmp_path):
        import yaml
        pdir = tmp_path / "ctypes-plugin"
        pdir.mkdir()
        manifest = _valid_manifest_dict(name="ctypes-plugin")
        (pdir / "manifest.yaml").write_text(yaml.dump(manifest))
        (pdir / "entry_point.py").write_text("import ctypes\n")
        loader = PluginLoader()
        with pytest.raises(PluginSandboxViolation, match="ctypes"):
            loader.load_plugin(pdir, sandbox=True)

    def test_shutdown_all(self, plugin_dir):
        loader = PluginLoader()
        loader.load_plugin(plugin_dir, sandbox=False)
        assert len(loader.loaded_plugins) == 1
        loader.shutdown_all()
        assert len(loader.loaded_plugins) == 0


# =========================================================================
# BUILT-IN PLUGIN TESTS
# =========================================================================

class TestCalculatorPlugin:
    """Tests for the built-in calculator plugin."""

    @pytest.fixture(autouse=True)
    def load_calc(self):
        self.calc = _load_module(
            "opti_oignon.plugins.calculator.entry_point",
            ROOT / "opti_oignon" / "plugins" / "calculator" / "entry_point.py",
        )

    def test_evaluate_basic_arithmetic(self):
        assert self.calc.evaluate("2 + 3") == 5
        assert self.calc.evaluate("10 - 4") == 6
        assert self.calc.evaluate("3 * 7") == 21
        assert self.calc.evaluate("15 / 3") == 5.0

    def test_evaluate_exponent(self):
        assert self.calc.evaluate("2 ** 10") == 1024

    def test_evaluate_functions(self):
        assert abs(self.calc.evaluate("sqrt(16)") - 4.0) < 1e-10
        assert abs(self.calc.evaluate("sin(0)")) < 1e-10

    def test_evaluate_constants(self):
        assert abs(self.calc.evaluate("pi") - math.pi) < 1e-10
        assert abs(self.calc.evaluate("e") - math.e) < 1e-10

    def test_evaluate_empty(self):
        with pytest.raises(self.calc.CalculatorError, match="Empty"):
            self.calc.evaluate("")

    def test_evaluate_syntax_error(self):
        with pytest.raises(self.calc.CalculatorError, match="Syntax"):
            self.calc.evaluate("2 +")

    def test_evaluate_huge_exponent_blocked(self):
        with pytest.raises(self.calc.CalculatorError, match="too large"):
            self.calc.evaluate("2 ** 100000")

    def test_evaluate_unknown_function(self):
        with pytest.raises(self.calc.CalculatorError, match="Unknown function"):
            self.calc.evaluate("evil(42)")

    def test_hook_tool_call(self):
        ctx = MagicMock()
        ctx.data = {"tool_name": "calculator", "expression": "2 + 3"}
        result = self.calc.hook_tool_call(ctx)
        assert result["result"] == "5"
        assert result["error"] is None

    def test_hook_tool_call_wrong_tool(self):
        ctx = MagicMock()
        ctx.data = {"tool_name": "other", "expression": "2 + 3"}
        assert self.calc.hook_tool_call(ctx) is None

    def test_format_result_int(self):
        assert self.calc.format_result(42) == "42"

    def test_format_result_float_whole(self):
        assert self.calc.format_result(5.0) == "5"


class TestCodeFormatterPlugin:
    """Tests for the built-in code_formatter plugin."""

    @pytest.fixture(autouse=True)
    def load_fmt(self):
        self.fmt = _load_module(
            "opti_oignon.plugins.code_formatter.entry_point",
            ROOT / "opti_oignon" / "plugins" / "code_formatter" / "entry_point.py",
        )

    def test_format_python_valid(self):
        code = "x=1\ny  = 2\n"
        result = self.fmt.format_python(code)
        assert "x" in result

    def test_format_python_invalid_returns_original(self):
        code = "def broken(:"
        result = self.fmt.format_python(code)
        assert result == code

    def test_format_json(self):
        code = '{"a":1,"b":2}'
        result = self.fmt.format_json(code)
        assert '"a": 1' in result

    def test_format_json_invalid_returns_original(self):
        code = "{broken json"
        result = self.fmt.format_json(code)
        assert result == code

    def test_hook_post_inference_formats_blocks(self):
        ctx = MagicMock()
        ctx.data = {"response": '```json\n{"a":1}\n```'}
        result = self.fmt.hook_post_inference(ctx)
        assert result is not None
        assert result["code_blocks_formatted"] >= 1

    def test_hook_post_inference_no_blocks(self):
        ctx = MagicMock()
        ctx.data = {"response": "No code here"}
        assert self.fmt.hook_post_inference(ctx) is None


class TestCitationGenPlugin:
    """Tests for the built-in citation_gen plugin."""

    @pytest.fixture(autouse=True)
    def load_cite(self):
        self.cite = _load_module(
            "opti_oignon.plugins.citation_gen.entry_point",
            ROOT / "opti_oignon" / "plugins" / "citation_gen" / "entry_point.py",
        )

    def test_format_citation_apa(self):
        source = {"author": "Smith, J.", "title": "A Study", "year": "2024"}
        result = self.cite.format_citation(source, "apa")
        assert "Smith, J." in result
        assert "2024" in result

    def test_format_citation_mla(self):
        source = {"author": "Smith, J.", "title": "A Study", "year": "2024"}
        result = self.cite.format_citation(source, "mla")
        assert "A Study" in result
        assert "Smith, J." in result

    def test_format_citation_chicago(self):
        source = {"author": "Smith, J.", "title": "A Study", "year": "2024"}
        result = self.cite.format_citation(source, "chicago")
        assert "Smith, J." in result

    def test_format_citations_dedup(self):
        sources = [
            {"author": "A", "title": "Same Title", "year": "2024"},
            {"author": "B", "title": "Same Title", "year": "2024"},
        ]
        result = self.cite.format_citations(sources)
        assert len(result) == 1  # deduplicated

    def test_build_references_section(self):
        citations = ["Smith (2024). Test."]
        result = self.cite.build_references_section(citations, "apa")
        assert "References" in result
        assert "[1]" in result

    def test_build_references_section_mla(self):
        result = self.cite.build_references_section(["Cite 1"], "mla")
        assert "Works Cited" in result

    def test_hook_tool_call(self):
        ctx = MagicMock()
        ctx.data = {
            "tool_name": "cite",
            "sources": [{"author": "A", "title": "B", "year": "2024"}],
            "style": "apa",
        }
        result = self.cite.hook_tool_call(ctx)
        assert result is not None
        assert result["count"] == 1
        assert result["error"] is None

    def test_hook_tool_call_wrong_tool(self):
        ctx = MagicMock()
        ctx.data = {"tool_name": "other"}
        assert self.cite.hook_tool_call(ctx) is None

    def test_extract_sources_from_rag(self):
        data = {"rag_results": [{"title": "T1"}, {"title": "T2"}]}
        sources = self.cite.extract_sources_from_rag(data)
        assert len(sources) == 2


# =========================================================================
# ROUTES SCHEMA TESTS
# =========================================================================

class TestRoutesPluginsSchemas:
    """Test that route Pydantic schemas instantiate correctly."""

    def _load_routes(self):
        pytest.importorskip("fastapi")
        return _load_module(
            "opti_oignon.api.routes_plugins",
            ROOT / "opti_oignon" / "api" / "routes_plugins.py",
        )

    def test_plugin_info_schema(self):
        mod = self._load_routes()
        info = mod.PluginInfo(
            name="test", version="1.0.0", author="A",
            description="D", entry_point="e.py",
            hooks=["post_inference"], state="enabled",
        )
        assert info.name == "test"
        assert info.state == "enabled"

    def test_plugin_list_response_schema(self):
        mod = self._load_routes()
        resp = mod.PluginListResponse(plugins=[], total=0, enabled=0)
        assert resp.total == 0

    def test_install_request_schema(self):
        mod = self._load_routes()
        req = mod.InstallRequest(source_dir="/tmp/test")
        assert req.auto_enable is False

    def test_install_response_schema(self):
        mod = self._load_routes()
        resp = mod.InstallResponse(success=True, name="t", version="1.0.0", message="ok")
        assert resp.success is True

    def test_state_change_response_schema(self):
        mod = self._load_routes()
        resp = mod.StateChangeResponse(success=True, name="t", state="enabled")
        assert resp.state == "enabled"

    def test_plugin_config_response_schema(self):
        mod = self._load_routes()
        resp = mod.PluginConfigResponse(
            name="t", config={"k": "v"},
            config_schema={"k": {"type": "string"}},
        )
        assert resp.config["k"] == "v"

    def test_update_config_response_schema(self):
        mod = self._load_routes()
        resp = mod.UpdateConfigResponse(
            success=True, name="t", config={"k": "v"}, message="ok",
        )
        assert resp.success is True
