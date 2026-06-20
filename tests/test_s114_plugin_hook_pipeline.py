#!/usr/bin/env python3
"""
Tests for S114 — Plugin Hook Pipeline Integration (Bug Fix).

Validates that:
1. PluginLoader._register_hooks() correctly bridges to HookManager
2. PluginLoader._unregister_hooks() cleans up properly
3. enable_plugin / disable_plugin register/unregister hooks
4. load_all_enabled registers hooks for each loaded plugin
5. routes_chat.py imports and dispatches hooks (pre_inference, post_inference, tool_call)
6. routes_plugins.py /debug endpoint exposes hook state
7. Hooks in the chat pipeline can modify messages and annotate responses
"""

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module isolation helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def _stub_opti():
    """Ensure opti_oignon parent module is stubbed for isolated loading."""
    if "opti_oignon" not in sys.modules or not hasattr(sys.modules["opti_oignon"], "__path__"):
        stub = types.ModuleType("opti_oignon")
        stub.__path__ = [str(ROOT / "opti_oignon")]
        sys.modules["opti_oignon"] = stub

    if "opti_oignon.config" not in sys.modules:
        cfg = types.ModuleType("opti_oignon.config")
        cfg.DATA_DIR = "/tmp/oo-test-s114"
        sys.modules["opti_oignon.config"] = cfg


def _load_module(name: str, filename: str):
    """Load a module via importlib for isolation."""
    _stub_opti()
    filepath = ROOT / "opti_oignon" / filename
    spec = importlib.util.spec_from_file_location(f"opti_oignon.{name}", str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"opti_oignon.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_hook_manager():
    """Ensure hook_manager is clean before/after each test."""
    ph = _load_module("plugin_hooks", "plugin_hooks.py")
    ph.hook_manager.clear()
    yield
    ph.hook_manager.clear()


@pytest.fixture
def hook_manager():
    ph = _load_module("plugin_hooks", "plugin_hooks.py")
    return ph.hook_manager


@pytest.fixture
def plugin_manifest_mod():
    return _load_module("plugin_manifest", "plugin_manifest.py")


@pytest.fixture
def plugin_loader_mod(plugin_manifest_mod):
    return _load_module("plugin_loader", "plugin_loader.py")


@pytest.fixture
def make_loaded_plugin(plugin_loader_mod):
    """Factory to create LoadedPlugin instances for testing."""
    def _make(name="test-plugin", version="1.0.0", hooks=None):
        mod = types.ModuleType(f"_opti_plugin_{name}")
        return plugin_loader_mod.LoadedPlugin(
            name=name,
            version=version,
            module=mod,
            plugin_dir=Path("/tmp") / name,
            hooks=hooks or {},
        )
    return _make


# =========================================================================
# Test Group 1 — _register_hooks / _unregister_hooks
# =========================================================================

class TestHookRegistrationBridge:
    """Test PluginLoader._register_hooks and _unregister_hooks."""

    def test_register_hooks_single(self, plugin_loader_mod, hook_manager, make_loaded_plugin):
        """A plugin with one hook should register it in HookManager."""
        def my_hook(ctx):
            return {"modified": True}

        loader = plugin_loader_mod.PluginLoader()
        lp = make_loaded_plugin(hooks={"post_inference": my_hook})

        count = loader._register_hooks(lp)
        assert count == 1
        assert hook_manager.get_hook_count() == 1
        assert hook_manager.has_hooks("post_inference")

    def test_register_hooks_multiple(self, plugin_loader_mod, hook_manager, make_loaded_plugin):
        """A plugin with multiple hooks should register all of them."""
        loader = plugin_loader_mod.PluginLoader()
        lp = make_loaded_plugin(hooks={
            "pre_inference": lambda ctx: None,
            "post_inference": lambda ctx: {"done": True},
            "tool_call": lambda ctx: None,
        })

        count = loader._register_hooks(lp)
        assert count == 3
        assert hook_manager.get_hook_count() == 3

    def test_register_hooks_no_hooks(self, plugin_loader_mod, hook_manager, make_loaded_plugin):
        """A plugin with no hooks should register zero."""
        loader = plugin_loader_mod.PluginLoader()
        lp = make_loaded_plugin(hooks={})

        count = loader._register_hooks(lp)
        assert count == 0
        assert hook_manager.get_hook_count() == 0

    def test_unregister_hooks(self, plugin_loader_mod, hook_manager, make_loaded_plugin):
        """_unregister_hooks should remove all hooks for a given plugin."""
        loader = plugin_loader_mod.PluginLoader()
        lp = make_loaded_plugin(hooks={
            "pre_inference": lambda ctx: None,
            "post_inference": lambda ctx: None,
        })
        loader._register_hooks(lp)
        assert hook_manager.get_hook_count() == 2

        removed = loader._unregister_hooks("test-plugin")
        assert removed == 2
        assert hook_manager.get_hook_count() == 0

    def test_unregister_hooks_nonexistent(self, plugin_loader_mod, hook_manager):
        """_unregister_hooks for a non-loaded plugin should return 0."""
        loader = plugin_loader_mod.PluginLoader()
        removed = loader._unregister_hooks("ghost-plugin")
        assert removed == 0

    def test_register_hooks_multiple_plugins(self, plugin_loader_mod, hook_manager, make_loaded_plugin):
        """Multiple plugins can register hooks at the same hook point."""
        loader = plugin_loader_mod.PluginLoader()

        lp1 = make_loaded_plugin(name="plugin-a", hooks={"post_inference": lambda ctx: None})
        lp2 = make_loaded_plugin(name="plugin-b", hooks={"post_inference": lambda ctx: None})

        loader._register_hooks(lp1)
        loader._register_hooks(lp2)
        assert hook_manager.get_hook_count() == 2

        # Unregister only plugin-a
        loader._unregister_hooks("plugin-a")
        assert hook_manager.get_hook_count() == 1
        hooks_list = hook_manager.list_hooks()
        assert hooks_list[0]["plugin_name"] == "plugin-b"

    def test_register_hooks_invalid_hook_name_skipped(self, plugin_loader_mod, hook_manager, make_loaded_plugin):
        """A hook with an invalid name should not register (HookManager rejects it)."""
        loader = plugin_loader_mod.PluginLoader()
        lp = make_loaded_plugin(hooks={
            "post_inference": lambda ctx: None,
            "totally_invalid_hook": lambda ctx: None,
        })

        count = loader._register_hooks(lp)
        # Only post_inference should register; totally_invalid_hook is rejected
        assert count == 1
        assert hook_manager.get_hook_count() == 1


# =========================================================================
# Test Group 2 — Hook execution chain
# =========================================================================

class TestHookExecution:
    """Test that hooks execute correctly and can modify data."""

    def test_pre_inference_modifies_message(self, hook_manager):
        """A pre_inference hook can modify the message."""
        def add_prefix(ctx):
            return {"message": "[ENHANCED] " + ctx.data.get("message", "")}

        hook_manager.register("pre_inference", "cot-enforcer", add_prefix)
        report = hook_manager.execute("pre_inference", data={"message": "Hello"})

        assert report.successful == 1
        assert report.final_data["message"] == "[ENHANCED] Hello"

    def test_post_inference_annotation(self, hook_manager):
        """A post_inference hook can return an annotation."""
        def fact_check(ctx):
            return {"annotation": {"type": "fact-check", "verdict": "verified"}}

        hook_manager.register("post_inference", "fact-checker", fact_check)
        report = hook_manager.execute("post_inference", data={"response": "The sky is blue"})

        assert report.successful == 1
        ann = report.results[0].modified_data.get("annotation")
        assert ann is not None
        assert ann["type"] == "fact-check"

    def test_post_inference_response_suffix(self, hook_manager):
        """A post_inference hook can return a response_suffix."""
        def add_tldr(ctx):
            return {"response_suffix": "\n\nTL;DR: short version"}

        hook_manager.register("post_inference", "auto-tldr", add_tldr)
        report = hook_manager.execute("post_inference", data={"response": "Long text here"})

        assert report.successful == 1
        assert report.results[0].modified_data["response_suffix"] == "\n\nTL;DR: short version"

    def test_multiple_hooks_chain(self, hook_manager):
        """Multiple hooks at the same point execute in order and chain data."""
        def hook_a(ctx):
            return {"annotation": "from A", "counter": 1}

        def hook_b(ctx):
            prev = ctx.data.get("counter", 0)
            return {"counter": prev + 1}

        hook_manager.register("post_inference", "plugin-a", hook_a, priority=50)
        hook_manager.register("post_inference", "plugin-b", hook_b, priority=100)

        report = hook_manager.execute("post_inference", data={"response": "test"})
        assert report.successful == 2
        assert report.final_data["counter"] == 2
        assert report.final_data["annotation"] == "from A"

    def test_hook_error_isolation(self, hook_manager):
        """A failing hook should not prevent other hooks from executing."""
        def bad_hook(ctx):
            raise ValueError("Intentional error")

        def good_hook(ctx):
            return {"success": True}

        hook_manager.register("post_inference", "bad-plugin", bad_hook, priority=50)
        hook_manager.register("post_inference", "good-plugin", good_hook, priority=100)

        report = hook_manager.execute("post_inference", data={})
        assert report.failed == 1
        assert report.successful == 1
        assert report.final_data.get("success") is True

    def test_tool_call_hook(self, hook_manager):
        """tool_call hooks receive tool data."""
        received_data = {}

        def capture_tool(ctx):
            received_data.update(ctx.data)
            return None

        hook_manager.register("tool_call", "spy-plugin", capture_tool)
        report = hook_manager.execute("tool_call", data={
            "tool_name": "web_search",
            "arguments": {"query": "test"},
            "result": "some result",
            "success": True,
        })

        assert report.successful == 1
        assert received_data["tool_name"] == "web_search"
        assert received_data["success"] is True


# =========================================================================
# Test Group 3 — AST verification of modified files
# =========================================================================

class TestASTIntegrity:
    """Verify AST validity of all modified files."""

    @pytest.mark.parametrize("filepath", [
        "opti_oignon/plugin_loader.py",
        "opti_oignon/plugin_hooks.py",
        "opti_oignon/api/routes_plugins.py",
        "opti_oignon/api/routes_chat.py",
    ])
    def test_ast_valid(self, filepath):
        full_path = ROOT / filepath
        source = full_path.read_text()
        tree = ast.parse(source)
        assert tree is not None

    def test_routes_chat_imports_plugin_hooks(self):
        """routes_chat.py should import plugin_hooks conditionally."""
        source = (ROOT / "opti_oignon" / "api" / "routes_chat.py").read_text()
        assert "plugin_hooks" in source
        assert "PLUGIN_HOOKS_AVAILABLE" in source
        assert "_hook_manager" in source

    def test_routes_chat_dispatches_pre_inference(self):
        """routes_chat.py should dispatch pre_inference hooks."""
        source = (ROOT / "opti_oignon" / "api" / "routes_chat.py").read_text()
        assert '"pre_inference"' in source

    def test_routes_chat_dispatches_post_inference(self):
        """routes_chat.py should dispatch post_inference hooks."""
        source = (ROOT / "opti_oignon" / "api" / "routes_chat.py").read_text()
        assert '"post_inference"' in source

    def test_routes_chat_dispatches_tool_call(self):
        """routes_chat.py should dispatch tool_call hooks."""
        source = (ROOT / "opti_oignon" / "api" / "routes_chat.py").read_text()
        assert "tool_call" in source
        # Ensure it's the hook dispatch, not just the event type
        assert '_hook_manager.execute(\n                    "tool_call"' in source or \
               '_hook_manager.execute(\n                    "tool_call"' in source.replace("    ", "")

    def test_routes_chat_plugin_annotations_in_done(self):
        """routes_chat.py should include plugin_annotations in done metadata."""
        source = (ROOT / "opti_oignon" / "api" / "routes_chat.py").read_text()
        assert "plugin_annotations" in source

    def test_routes_plugins_debug_endpoint(self):
        """routes_plugins.py should have a /debug endpoint."""
        source = (ROOT / "opti_oignon" / "api" / "routes_plugins.py").read_text()
        assert '"/debug"' in source or "'/debug'" in source
        assert "plugin_debug" in source

    def test_plugin_loader_has_register_hooks(self):
        """plugin_loader.py should have _register_hooks method."""
        source = (ROOT / "opti_oignon" / "plugin_loader.py").read_text()
        assert "_register_hooks" in source
        assert "_unregister_hooks" in source

    def test_plugin_loader_enable_calls_register(self):
        """enable_plugin should call _register_hooks."""
        source = (ROOT / "opti_oignon" / "plugin_loader.py").read_text()
        # Find enable_plugin method and check it references _register_hooks
        idx = source.index("def enable_plugin")
        chunk = source[idx:idx + 800]
        assert "_register_hooks" in chunk

    def test_plugin_loader_disable_calls_unregister(self):
        """disable_plugin should call _unregister_hooks."""
        source = (ROOT / "opti_oignon" / "plugin_loader.py").read_text()
        idx = source.index("def disable_plugin")
        chunk = source[idx:idx + 500]
        assert "_unregister_hooks" in chunk

    def test_plugin_loader_load_all_enabled_calls_register(self):
        """load_all_enabled should call _register_hooks."""
        source = (ROOT / "opti_oignon" / "plugin_loader.py").read_text()
        idx = source.index("def load_all_enabled")
        chunk = source[idx:idx + 600]
        assert "_register_hooks" in chunk

    def test_routes_plugins_debug_before_parametric(self):
        """The /debug endpoint must appear before /{name}/ routes to avoid shadowing."""
        source = (ROOT / "opti_oignon" / "api" / "routes_plugins.py").read_text()
        debug_pos = source.index('"/debug"')
        enable_pos = source.index('"/{name}/enable"')
        assert debug_pos < enable_pos, "/debug must be registered before /{name}/enable"


# =========================================================================
# Test Group 4 — Debug endpoint logic
# =========================================================================

class TestDebugEndpoint:
    """Test the /api/plugins/debug endpoint logic directly."""

    def test_debug_endpoint_function(self, plugin_loader_mod, hook_manager, make_loaded_plugin):
        """Simulate calling the debug endpoint logic."""
        # Register some hooks
        loader = plugin_loader_mod.PluginLoader()
        lp = make_loaded_plugin(
            name="fact-checker",
            hooks={"post_inference": lambda ctx: None},
        )
        loader._register_hooks(lp)

        # Simulate debug response
        registered = hook_manager.list_hooks()
        assert len(registered) == 1
        assert registered[0]["plugin_name"] == "fact-checker"
        assert registered[0]["hook_name"] == "post_inference"

        stats = hook_manager.get_stats()
        assert isinstance(stats, dict)

    def test_debug_after_execution(self, hook_manager):
        """After executing hooks, stats should reflect calls."""
        def noop(ctx):
            return None

        hook_manager.register("post_inference", "stats-test", noop)
        hook_manager.execute("post_inference", data={})
        hook_manager.execute("post_inference", data={})

        stats = hook_manager.get_stats()
        key = "stats-test:post_inference"
        assert key in stats
        assert stats[key]["calls"] == 2
        assert stats[key]["errors"] == 0
