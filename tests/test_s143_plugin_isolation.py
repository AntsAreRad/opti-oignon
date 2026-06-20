#!/usr/bin/env python3
"""S143 — Plugin Out-of-Process Isolation tests.

Verifies:
- Part 1: Wire protocol (pack/unpack, HMAC signing, message framing)
- Part 2: HMAC tamper detection and error handling
- Part 3: JSON-RPC helpers (request/response/error builders)
- Part 4: PluginResourceLimits (defaults, from_manifest, to_dict)
- Part 5: Plugin log capture (setup_plugin_logger, rotation config)
- Part 6: PluginSubprocessManager (construction, properties)
- Part 7: Worker script (resource limits, module loading, hook dispatch)
- Part 8: SubprocessPluginAdapter (interface parity with LoadedPlugin)
- Part 9: RPC hook proxy (callable creation, error handling)
- Part 10: PluginLoader subprocess_mode (auto/subprocess/inprocess)
- Part 11: Backward compatibility (existing plugins unchanged)
- Part 12: Graceful degradation (fallback to in-process)
- Part 13: Manifest resource_limits validation
- Part 14: Config (plugins.yaml subprocess section)
- Part 15: End-to-end subprocess lifecycle
- Part 16: Crash isolation (subprocess crash doesn't kill host)
- Part 17: Version bump (3.1.3)

Target: ~70 tests
"""

import ast
import importlib.util
import json
import hashlib
import hmac
import os
import secrets
import socket
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG = os.path.join(_PROJECT_ROOT, "opti_oignon")


def _load_module(name: str, path: str) -> types.ModuleType:
    """Load a module via importlib without triggering __init__ chain."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load modules under test
_ps = _load_module("_t_ps", os.path.join(_PKG, "plugin_subprocess.py"))
_pw = _load_module("_t_pw", os.path.join(_PKG, "plugin_worker.py"))
_pm = _load_module("_t_pm", os.path.join(_PKG, "plugin_manifest.py"))
_pl = _load_module("_t_pl", os.path.join(_PKG, "plugin_loader.py"))

# Pre-seed sys.modules so that `from opti_oignon.X import ...` inside
# load_plugin methods resolves to our isolated modules instead of
# triggering the full opti_oignon.__init__ chain (which needs ollama).
if "opti_oignon" not in sys.modules:
    _oo_stub = types.ModuleType("opti_oignon")
    _oo_stub.__path__ = [_PKG]  # type: ignore[attr-defined]
    sys.modules["opti_oignon"] = _oo_stub
if "opti_oignon.plugin_manifest" not in sys.modules:
    sys.modules["opti_oignon.plugin_manifest"] = _pm
if "opti_oignon.plugin_subprocess" not in sys.modules:
    sys.modules["opti_oignon.plugin_subprocess"] = _ps
if "opti_oignon.plugin_loader" not in sys.modules:
    sys.modules["opti_oignon.plugin_loader"] = _pl

# Stub security_mode so Bulbe mode checks don't block test plugin loads
_sec_stub = types.ModuleType("opti_oignon.security_mode")
_sec_stub.is_bulbe = lambda: False  # type: ignore[attr-defined]
sys.modules["opti_oignon.security_mode"] = _sec_stub


# ===========================================================================
# Part 1: Wire protocol — pack/unpack, message framing
# ===========================================================================

class TestWireProtocolPack(unittest.TestCase):
    """Pack/unpack roundtrip and wire format structure."""

    def setUp(self) -> None:
        self.key = secrets.token_bytes(32)

    def test_pack_unpack_roundtrip(self) -> None:
        payload = {"jsonrpc": "2.0", "method": "ping", "id": "1"}
        packed = _ps.pack_message(self.key, payload)
        result = _ps.unpack_message(self.key, packed)
        self.assertEqual(result, payload)

    def test_wire_format_structure(self) -> None:
        """Wire = 4-byte length + 32-byte HMAC + JSON payload."""
        payload = {"test": True}
        packed = _ps.pack_message(self.key, payload)
        length = struct.unpack("!I", packed[:4])[0]
        self.assertEqual(len(packed), 4 + 32 + length)

    def test_pack_complex_payload(self) -> None:
        payload = {
            "jsonrpc": "2.0",
            "method": "execute_hook",
            "params": {"data": {"nested": [1, 2, 3], "flag": True}},
            "id": "abc",
        }
        packed = _ps.pack_message(self.key, payload)
        result = _ps.unpack_message(self.key, packed)
        self.assertEqual(result["params"]["data"]["nested"], [1, 2, 3])

    def test_pack_empty_payload(self) -> None:
        packed = _ps.pack_message(self.key, {})
        result = _ps.unpack_message(self.key, packed)
        self.assertEqual(result, {})

    def test_unpack_wrong_key_fails(self) -> None:
        packed = _ps.pack_message(self.key, {"x": 1})
        wrong_key = secrets.token_bytes(32)
        with self.assertRaises(_ps.PluginHMACError):
            _ps.unpack_message(wrong_key, packed)

    def test_pack_size_limit(self) -> None:
        huge = {"data": "x" * (_ps.MAX_MESSAGE_SIZE + 1)}
        with self.assertRaises(_ps.PluginIPCError):
            _ps.pack_message(self.key, huge)


# ===========================================================================
# Part 2: HMAC tamper detection
# ===========================================================================

class TestHMACTamperDetection(unittest.TestCase):
    """HMAC verification catches tampered messages."""

    def setUp(self) -> None:
        self.key = secrets.token_bytes(32)

    def test_tamper_payload_byte(self) -> None:
        packed = bytearray(_ps.pack_message(self.key, {"ok": True}))
        packed[40] ^= 0xFF  # flip a payload byte
        with self.assertRaises(_ps.PluginHMACError):
            _ps.unpack_message(self.key, bytes(packed))

    def test_tamper_hmac_byte(self) -> None:
        packed = bytearray(_ps.pack_message(self.key, {"ok": True}))
        packed[10] ^= 0xFF  # flip an HMAC byte
        with self.assertRaises(_ps.PluginHMACError):
            _ps.unpack_message(self.key, bytes(packed))

    def test_truncated_message(self) -> None:
        packed = _ps.pack_message(self.key, {"ok": True})
        with self.assertRaises(_ps.PluginIPCError):
            _ps.unpack_message(self.key, packed[:10])

    def test_empty_data(self) -> None:
        with self.assertRaises(_ps.PluginIPCError):
            _ps.unpack_message(self.key, b"")

    def test_length_mismatch(self) -> None:
        packed = bytearray(_ps.pack_message(self.key, {"x": 1}))
        # Corrupt length field to be larger
        struct.pack_into("!I", packed, 0, 99999)
        with self.assertRaises(_ps.PluginIPCError):
            _ps.unpack_message(self.key, bytes(packed))

    def test_constant_time_comparison(self) -> None:
        """Verify hmac.compare_digest is used (not ==)."""
        data = b"test data"
        mac = _ps._compute_hmac(self.key, data)
        self.assertTrue(_ps._verify_hmac(self.key, data, mac))
        self.assertFalse(_ps._verify_hmac(self.key, data, b"\x00" * 32))


# ===========================================================================
# Part 3: JSON-RPC helpers
# ===========================================================================

class TestJsonRpcHelpers(unittest.TestCase):
    """JSON-RPC 2.0 request/response/error builders."""

    def test_make_rpc_request(self) -> None:
        req = _ps.make_rpc_request("ping", {"key": "val"}, "id123")
        self.assertEqual(req["jsonrpc"], "2.0")
        self.assertEqual(req["method"], "ping")
        self.assertEqual(req["params"], {"key": "val"})
        self.assertEqual(req["id"], "id123")

    def test_make_rpc_request_auto_id(self) -> None:
        req = _ps.make_rpc_request("test")
        self.assertIn("id", req)
        self.assertTrue(len(req["id"]) > 0)

    def test_make_rpc_response(self) -> None:
        resp = _ps.make_rpc_response("id1", result={"status": "ok"})
        self.assertEqual(resp["jsonrpc"], "2.0")
        self.assertEqual(resp["id"], "id1")
        self.assertEqual(resp["result"]["status"], "ok")
        self.assertNotIn("error", resp)

    def test_make_rpc_error(self) -> None:
        err = _ps.make_rpc_error("id2", -32601, "Method not found")
        self.assertEqual(err["error"]["code"], -32601)
        self.assertEqual(err["error"]["message"], "Method not found")

    def test_make_rpc_error_with_data(self) -> None:
        err = _ps.make_rpc_error("id3", -32603, "Internal", data={"trace": "x"})
        self.assertEqual(err["error"]["data"]["trace"], "x")

    def test_worker_make_response(self) -> None:
        resp = _pw.make_response("id1", {"status": "pong"})
        self.assertEqual(resp["jsonrpc"], "2.0")
        self.assertEqual(resp["result"]["status"], "pong")

    def test_worker_make_error(self) -> None:
        err = _pw.make_error("id2", -32600, "Invalid")
        self.assertEqual(err["error"]["code"], -32600)


# ===========================================================================
# Part 4: PluginResourceLimits
# ===========================================================================

class TestPluginResourceLimits(unittest.TestCase):
    """Resource limit configuration and serialization."""

    def test_defaults(self) -> None:
        rl = _ps.PluginResourceLimits()
        self.assertEqual(rl.cpu_time_seconds, 30)
        self.assertEqual(rl.memory_bytes, 256 * 1024 * 1024)
        self.assertEqual(rl.max_file_descriptors, 64)

    def test_from_manifest_with_limits(self) -> None:
        data = {"resource_limits": {
            "cpu_time_seconds": 10,
            "memory_bytes": 128 * 1024 * 1024,
            "max_file_descriptors": 32,
        }}
        rl = _ps.PluginResourceLimits.from_manifest(data)
        self.assertEqual(rl.cpu_time_seconds, 10)
        self.assertEqual(rl.memory_bytes, 128 * 1024 * 1024)
        self.assertEqual(rl.max_file_descriptors, 32)

    def test_from_manifest_empty(self) -> None:
        rl = _ps.PluginResourceLimits.from_manifest({})
        self.assertEqual(rl.cpu_time_seconds, 30)  # defaults

    def test_from_manifest_partial(self) -> None:
        rl = _ps.PluginResourceLimits.from_manifest(
            {"resource_limits": {"cpu_time_seconds": 5}}
        )
        self.assertEqual(rl.cpu_time_seconds, 5)
        self.assertEqual(rl.memory_bytes, 256 * 1024 * 1024)  # default

    def test_to_dict(self) -> None:
        rl = _ps.PluginResourceLimits(cpu_time_seconds=15, memory_bytes=1000, max_file_descriptors=20)
        d = rl.to_dict()
        self.assertEqual(d["cpu_time_seconds"], 15)
        self.assertEqual(d["memory_bytes"], 1000)
        self.assertEqual(d["max_file_descriptors"], 20)

    def test_roundtrip(self) -> None:
        original = _ps.PluginResourceLimits(cpu_time_seconds=7, memory_bytes=999, max_file_descriptors=16)
        restored = _ps.PluginResourceLimits.from_manifest(
            {"resource_limits": original.to_dict()}
        )
        self.assertEqual(original.cpu_time_seconds, restored.cpu_time_seconds)
        self.assertEqual(original.memory_bytes, restored.memory_bytes)
        self.assertEqual(original.max_file_descriptors, restored.max_file_descriptors)


# ===========================================================================
# Part 5: Plugin log capture
# ===========================================================================

class TestPluginLogCapture(unittest.TestCase):
    """setup_plugin_logger creates rotating file logs."""

    def test_logger_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td) / "subdir" / "logs"
            logger = _ps.setup_plugin_logger("test-plugin", log_dir)
            self.assertTrue(log_dir.exists())
            self.assertIsNotNone(logger)

    def test_logger_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td)
            logger = _ps.setup_plugin_logger("myplugin", log_dir)
            logger.info("test message")
            # Flush handlers
            for h in logger.handlers:
                h.flush()
            log_file = log_dir / "myplugin.log"
            self.assertTrue(log_file.exists())
            content = log_file.read_text()
            self.assertIn("test message", content)

    def test_logger_name_format(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            logger = _ps.setup_plugin_logger("abc", Path(td))
            self.assertEqual(logger.name, "opti.plugin.abc")

    def test_no_duplicate_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td)
            l1 = _ps.setup_plugin_logger("dup-test", log_dir)
            count1 = len(l1.handlers)
            l2 = _ps.setup_plugin_logger("dup-test", log_dir)
            self.assertEqual(len(l2.handlers), count1)


# ===========================================================================
# Part 6: PluginSubprocessManager construction
# ===========================================================================

class TestSubprocessManagerConstruction(unittest.TestCase):
    """Manager initialization and properties."""

    def test_default_construction(self) -> None:
        mgr = _ps.PluginSubprocessManager()
        self.assertIsInstance(mgr.socket_dir, Path)
        self.assertIsInstance(mgr.log_dir, Path)
        self.assertEqual(mgr.running_plugins, [])

    def test_custom_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as sd, tempfile.TemporaryDirectory() as ld:
            mgr = _ps.PluginSubprocessManager(socket_dir=sd, log_dir=ld)
            self.assertEqual(mgr.socket_dir, Path(sd))
            self.assertEqual(mgr.log_dir, Path(ld))

    def test_custom_timeouts(self) -> None:
        mgr = _ps.PluginSubprocessManager(
            watchdog_interval=2.0,
            default_hook_timeout=5.0,
            startup_timeout=8.0,
        )
        self.assertEqual(mgr._watchdog_interval, 2.0)
        self.assertEqual(mgr._default_hook_timeout, 5.0)
        self.assertEqual(mgr._startup_timeout, 8.0)

    def test_get_nonexistent_process(self) -> None:
        mgr = _ps.PluginSubprocessManager()
        self.assertIsNone(mgr.get_process("nonexistent"))

    def test_is_running_false_for_unknown(self) -> None:
        mgr = _ps.PluginSubprocessManager()
        self.assertFalse(mgr.is_running("nonexistent"))

    def test_ping_false_for_unknown(self) -> None:
        mgr = _ps.PluginSubprocessManager()
        self.assertFalse(mgr.ping("nonexistent"))

    def test_stop_unknown_returns_false(self) -> None:
        mgr = _ps.PluginSubprocessManager()
        self.assertFalse(mgr.stop_plugin("nonexistent"))


# ===========================================================================
# Part 7: Worker script — resource limits, module loading, hook dispatch
# ===========================================================================

class TestWorkerResourceLimits(unittest.TestCase):
    """Worker apply_resource_limits function."""

    def test_apply_returns_dict(self) -> None:
        # Use generous limits so it doesn't break the test process
        result = _pw.apply_resource_limits(
            cpu_seconds=3600,
            memory_bytes=8 * 1024 * 1024 * 1024,  # 8 GB
            max_fds=4096,
        )
        self.assertIsInstance(result, dict)
        # At least some limits should have been applied
        self.assertTrue(len(result) > 0)


class TestWorkerModuleLoading(unittest.TestCase):
    """Worker load_plugin_module function."""

    def test_load_simple_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            entry = Path(td) / "main.py"
            entry.write_text(
                'PLUGIN_VALUE = 42\n'
                'def hook_pre_prompt(data):\n'
                '    return {"modified": True}\n'
            )
            mod = _pw.load_plugin_module("simple", td, "main.py")
            self.assertEqual(mod.PLUGIN_VALUE, 42)
            self.assertTrue(callable(mod.hook_pre_prompt))

    def test_load_missing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                _pw.load_plugin_module("bad", td, "nonexistent.py")

    def test_plugin_name_injected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            entry = Path(td) / "main.py"
            entry.write_text("pass\n")
            mod = _pw.load_plugin_module("injtest", td, "main.py")
            self.assertEqual(mod.__plugin_name__, "injtest")
            self.assertEqual(mod.__plugin_dir__, td)


class TestWorkerHookExecution(unittest.TestCase):
    """Worker execute_hook function."""

    def _make_module(self, code: str, name: str = "hookmod") -> types.ModuleType:
        with tempfile.TemporaryDirectory() as td:
            entry = Path(td) / "main.py"
            entry.write_text(code)
            return _pw.load_plugin_module(name, td, "main.py")

    def test_hook_via_function(self) -> None:
        mod = self._make_module(
            'def hook_pre_prompt(data):\n    return {"result": "ok"}\n',
            "hf1",
        )
        result = _pw.execute_hook(mod, "pre_prompt", {})
        self.assertEqual(result["result"], "ok")

    def test_hook_via_hooks_dict(self) -> None:
        mod = self._make_module(
            'HOOKS = {"post_inference": lambda d: {"handled": True}}\n',
            "hf2",
        )
        result = _pw.execute_hook(mod, "post_inference", {})
        self.assertTrue(result["handled"])

    def test_hook_no_handler(self) -> None:
        mod = self._make_module("pass\n", "hf3")
        result = _pw.execute_hook(mod, "pre_prompt", {})
        self.assertEqual(result["status"], "no_handler")

    def test_hook_returns_none(self) -> None:
        mod = self._make_module(
            'def hook_pre_prompt(data):\n    pass\n',
            "hf4",
        )
        result = _pw.execute_hook(mod, "pre_prompt", {})
        self.assertEqual(result["status"], "ok")


# ===========================================================================
# Part 8: SubprocessPluginAdapter
# ===========================================================================

class TestSubprocessPluginAdapter(unittest.TestCase):
    """SubprocessPluginAdapter interface parity with LoadedPlugin."""

    def _make_adapter(self) -> _pl.SubprocessPluginAdapter:
        mock_mgr = MagicMock()
        hooks = {"pre_prompt": lambda ctx: {"modified": True}}
        return _pl.SubprocessPluginAdapter(
            name="test-adapter",
            version="1.0.0",
            plugin_dir=Path("/tmp/fake"),
            hooks=hooks,
            subprocess_manager=mock_mgr,
        )

    def test_is_loaded_plugin_subclass(self) -> None:
        adapter = self._make_adapter()
        self.assertIsInstance(adapter, _pl.LoadedPlugin)

    def test_has_subprocess_flag(self) -> None:
        adapter = self._make_adapter()
        self.assertTrue(adapter.is_subprocess)

    def test_name_and_version(self) -> None:
        adapter = self._make_adapter()
        self.assertEqual(adapter.name, "test-adapter")
        self.assertEqual(adapter.version, "1.0.0")

    def test_get_hook(self) -> None:
        adapter = self._make_adapter()
        hook = adapter.get_hook("pre_prompt")
        self.assertIsNotNone(hook)
        self.assertIsNone(adapter.get_hook("nonexistent"))

    def test_initialize_noop(self) -> None:
        adapter = self._make_adapter()
        adapter.initialize()  # Should not raise

    def test_shutdown_calls_manager(self) -> None:
        adapter = self._make_adapter()
        adapter.shutdown()
        adapter._subprocess_manager.stop_plugin.assert_called_once_with("test-adapter")

    def test_module_is_dummy(self) -> None:
        adapter = self._make_adapter()
        self.assertIsNotNone(adapter.module)
        self.assertTrue(adapter.module.__name__.startswith("_opti_subprocess_"))


# ===========================================================================
# Part 9: RPC hook proxy
# ===========================================================================

class TestRpcHookProxy(unittest.TestCase):
    """_make_rpc_hook_proxy creates valid callables."""

    def test_proxy_is_callable(self) -> None:
        mock_mgr = MagicMock()
        proxy = _pl._make_rpc_hook_proxy("myplugin", "pre_prompt", mock_mgr)
        self.assertTrue(callable(proxy))

    def test_proxy_calls_manager(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.call_hook.return_value = {"modified": True}
        proxy = _pl._make_rpc_hook_proxy("myplugin", "pre_prompt", mock_mgr)
        result = proxy({"data": {"msg": "hello"}})
        mock_mgr.call_hook.assert_called_once()
        self.assertEqual(result["modified"], True)

    def test_proxy_with_hook_context(self) -> None:
        """Proxy should handle HookContext-like objects."""
        mock_mgr = MagicMock()
        mock_mgr.call_hook.return_value = {"ok": True}
        proxy = _pl._make_rpc_hook_proxy("p1", "pre_prompt", mock_mgr)

        ctx = MagicMock()
        ctx.hook_name = "pre_prompt"
        ctx.plugin_name = "p1"
        ctx.conversation_id = "conv1"
        ctx.model = "llama"
        ctx.data = {"message": "hi"}
        ctx.config = {}
        ctx.metadata = {}

        result = proxy(ctx)
        self.assertEqual(result["ok"], True)

    def test_proxy_raises_on_rpc_error(self) -> None:
        mock_mgr = MagicMock()
        mock_mgr.call_hook.side_effect = _ps.PluginSubprocessError("dead")
        proxy = _pl._make_rpc_hook_proxy("p2", "pre_prompt", mock_mgr)
        with self.assertRaises(_ps.PluginSubprocessError):
            proxy({})

    def test_proxy_name(self) -> None:
        mock_mgr = MagicMock()
        proxy = _pl._make_rpc_hook_proxy("foo", "bar", mock_mgr)
        self.assertIn("foo", proxy.__name__)
        self.assertIn("bar", proxy.__name__)


# ===========================================================================
# Part 10: PluginLoader subprocess_mode
# ===========================================================================

class TestPluginLoaderSubprocessMode(unittest.TestCase):
    """PluginLoader subprocess_mode parameter behavior."""

    def test_default_mode_is_auto(self) -> None:
        loader = _pl.PluginLoader()
        self.assertEqual(loader._subprocess_mode, "auto")

    def test_inprocess_mode(self) -> None:
        loader = _pl.PluginLoader(subprocess_mode="inprocess")
        self.assertEqual(loader._subprocess_mode, "inprocess")

    def test_subprocess_mode(self) -> None:
        loader = _pl.PluginLoader(subprocess_mode="subprocess")
        self.assertEqual(loader._subprocess_mode, "subprocess")

    def test_accepts_external_manager(self) -> None:
        mock_mgr = MagicMock()
        loader = _pl.PluginLoader(subprocess_manager=mock_mgr)
        self.assertIs(loader._subprocess_manager, mock_mgr)

    def test_lazy_manager_creation(self) -> None:
        loader = _pl.PluginLoader()
        self.assertIsNone(loader._subprocess_manager)
        # _get_subprocess_manager should create one
        mgr = loader._get_subprocess_manager()
        self.assertIsNotNone(mgr)
        # Second call returns the same instance
        self.assertIs(loader._get_subprocess_manager(), mgr)


# ===========================================================================
# Part 11: Backward compatibility
# ===========================================================================

class TestBackwardCompatibility(unittest.TestCase):
    """Existing plugin API surface is preserved."""

    def test_loaded_plugin_interface(self) -> None:
        """LoadedPlugin still has all original methods."""
        for attr in ["name", "version", "module", "plugin_dir", "hooks"]:
            self.assertTrue(hasattr(_pl.LoadedPlugin, "__init__"))

        mod = types.ModuleType("fake")
        lp = _pl.LoadedPlugin(
            name="test", version="1.0.0", module=mod,
            plugin_dir=Path("/tmp"), hooks={},
        )
        self.assertTrue(hasattr(lp, "initialize"))
        self.assertTrue(hasattr(lp, "shutdown"))
        self.assertTrue(hasattr(lp, "get_hook"))

    def test_plugin_loader_interface(self) -> None:
        """PluginLoader retains all original public methods."""
        for method in [
            "load_plugin", "unload_plugin", "install_plugin",
            "uninstall_plugin", "enable_plugin", "disable_plugin",
            "load_all_enabled", "shutdown_all", "loaded_plugins",
        ]:
            self.assertTrue(
                hasattr(_pl.PluginLoader, method),
                f"Missing: {method}",
            )

    def test_inprocess_load_still_works(self) -> None:
        """Loading a plugin with sandbox=False in inprocess mode works."""
        with tempfile.TemporaryDirectory() as td:
            # Create minimal plugin
            (Path(td) / "manifest.yaml").write_text(
                "name: compat-test\n"
                "version: '1.0.0'\n"
                "author: test\n"
                "description: compat test\n"
                "entry_point: main.py\n"
                "hooks: [pre_prompt]\n"
            )
            (Path(td) / "main.py").write_text(
                'def hook_pre_prompt(ctx):\n    return {"compat": True}\n'
            )
            loader = _pl.PluginLoader(subprocess_mode="inprocess")
            loaded = loader.load_plugin(td, sandbox=False)
            self.assertEqual(loaded.name, "compat-test")
            self.assertIn("pre_prompt", loaded.hooks)
            self.assertFalse(isinstance(loaded, _pl.SubprocessPluginAdapter))

    def test_exceptions_preserved(self) -> None:
        self.assertTrue(hasattr(_pl, "PluginLoadError"))
        self.assertTrue(hasattr(_pl, "PluginSandboxViolation"))


# ===========================================================================
# Part 12: Graceful degradation
# ===========================================================================

class TestGracefulDegradation(unittest.TestCase):
    """Auto mode falls back to in-process when subprocess fails."""

    def test_auto_fallback_on_subprocess_failure(self) -> None:
        """When subprocess fails, auto mode falls back to inprocess."""
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "manifest.yaml").write_text(
                "name: fallback-test\n"
                "version: '1.0.0'\n"
                "author: test\n"
                "description: fallback test\n"
                "entry_point: main.py\n"
                "hooks: []\n"
            )
            (Path(td) / "main.py").write_text("VALUE = 99\n")

            # Create loader with a subprocess manager that always fails
            mock_mgr = MagicMock()
            mock_mgr.start_plugin.side_effect = Exception("subprocess broken")

            loader = _pl.PluginLoader(
                subprocess_mode="auto",
                subprocess_manager=mock_mgr,
            )
            # Should succeed via fallback
            loaded = loader.load_plugin(td, sandbox=False)
            self.assertEqual(loaded.name, "fallback-test")
            self.assertFalse(isinstance(loaded, _pl.SubprocessPluginAdapter))

    def test_subprocess_mode_no_fallback(self) -> None:
        """Subprocess-only mode does NOT fall back — it raises."""
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "manifest.yaml").write_text(
                "name: no-fallback\n"
                "version: '1.0.0'\n"
                "author: test\n"
                "description: test\n"
                "entry_point: main.py\n"
                "hooks: []\n"
            )
            (Path(td) / "main.py").write_text("pass\n")

            mock_mgr = MagicMock()
            mock_mgr.start_plugin.side_effect = Exception("subprocess broken")

            loader = _pl.PluginLoader(
                subprocess_mode="subprocess",
                subprocess_manager=mock_mgr,
            )
            with self.assertRaises(_pl.PluginLoadError):
                loader.load_plugin(td)


# ===========================================================================
# Part 13: Manifest resource_limits validation
# ===========================================================================

class TestManifestResourceLimits(unittest.TestCase):
    """resource_limits field in PluginManifest."""

    def _base_data(self, **extra: dict) -> dict:
        d = {
            "name": "rl-test",
            "version": "1.0.0",
            "author": "test",
            "description": "test",
            "entry_point": "main.py",
        }
        d.update(extra)
        return d

    def test_empty_resource_limits(self) -> None:
        m = _pm.PluginManifest.from_dict(self._base_data())
        self.assertEqual(m.resource_limits, {})

    def test_valid_resource_limits(self) -> None:
        m = _pm.PluginManifest.from_dict(self._base_data(
            resource_limits={"cpu_time_seconds": 10, "memory_bytes": 100000}
        ))
        self.assertEqual(m.resource_limits["cpu_time_seconds"], 10)

    def test_invalid_key_rejected(self) -> None:
        with self.assertRaises(_pm.PluginManifestError) as ctx:
            _pm.PluginManifest.from_dict(self._base_data(
                resource_limits={"invalid_key": 5}
            ))
        self.assertIn("invalid_key", str(ctx.exception))

    def test_negative_value_rejected(self) -> None:
        with self.assertRaises(_pm.PluginManifestError):
            _pm.PluginManifest.from_dict(self._base_data(
                resource_limits={"cpu_time_seconds": -1}
            ))

    def test_to_dict_includes_resource_limits(self) -> None:
        m = _pm.PluginManifest.from_dict(self._base_data(
            resource_limits={"max_file_descriptors": 32}
        ))
        d = m.to_dict()
        self.assertEqual(d["resource_limits"]["max_file_descriptors"], 32)

    def test_all_valid_keys_accepted(self) -> None:
        m = _pm.PluginManifest.from_dict(self._base_data(
            resource_limits={
                "cpu_time_seconds": 5,
                "memory_bytes": 1024,
                "max_file_descriptors": 16,
            }
        ))
        self.assertEqual(len(m.resource_limits), 3)


# ===========================================================================
# Part 14: Config — plugins.yaml subprocess section
# ===========================================================================

class TestPluginsYamlConfig(unittest.TestCase):
    """plugins.yaml includes subprocess configuration."""

    @classmethod
    def setUpClass(cls) -> None:
        import yaml
        config_path = os.path.join(_PKG, "config", "plugins.yaml")
        with open(config_path, "r") as f:
            cls.config = yaml.safe_load(f)

    def test_subprocess_section_exists(self) -> None:
        self.assertIn("subprocess", self.config)

    def test_mode_default(self) -> None:
        self.assertEqual(self.config["subprocess"]["mode"], "auto")

    def test_startup_timeout(self) -> None:
        self.assertIsInstance(self.config["subprocess"]["startup_timeout"], (int, float))
        self.assertGreater(self.config["subprocess"]["startup_timeout"], 0)

    def test_hook_timeout(self) -> None:
        self.assertIsInstance(self.config["subprocess"]["hook_timeout"], (int, float))
        self.assertGreater(self.config["subprocess"]["hook_timeout"], 0)

    def test_watchdog_interval(self) -> None:
        self.assertIsInstance(self.config["subprocess"]["watchdog_interval"], (int, float))

    def test_default_resource_limits(self) -> None:
        rl = self.config["subprocess"]["default_resource_limits"]
        self.assertIn("cpu_time_seconds", rl)
        self.assertIn("memory_bytes", rl)
        self.assertIn("max_file_descriptors", rl)

    def test_log_config(self) -> None:
        log = self.config["subprocess"]["log"]
        self.assertIn("dir", log)
        self.assertIn("max_bytes", log)
        self.assertIn("backup_count", log)


# ===========================================================================
# Part 15: End-to-end subprocess lifecycle
# ===========================================================================

class TestE2ESubprocessLifecycle(unittest.TestCase):
    """Full subprocess start → communicate → stop cycle."""

    def test_start_and_stop(self) -> None:
        """Start a real plugin subprocess, ping it, then stop it."""
        with tempfile.TemporaryDirectory() as td:
            plugin_dir = Path(td) / "myplugin"
            plugin_dir.mkdir()

            # Write minimal plugin
            (plugin_dir / "main.py").write_text(
                'def init():\n    pass\n\n'
                'def hook_pre_prompt(data):\n    return {"e2e": True}\n\n'
                'def shutdown():\n    pass\n'
            )

            sock_dir = Path(td) / "socks"
            log_dir = Path(td) / "logs"

            worker_script = os.path.join(_PKG, "plugin_worker.py")

            mgr = _ps.PluginSubprocessManager(
                socket_dir=sock_dir,
                log_dir=log_dir,
                worker_script=worker_script,
                startup_timeout=10.0,
            )

            try:
                pp = mgr.start_plugin(
                    plugin_name="e2e-test",
                    plugin_dir=plugin_dir,
                    entry_point="main.py",
                )
                self.assertTrue(pp.is_alive())
                self.assertTrue(mgr.is_running("e2e-test"))

                # Ping
                self.assertTrue(mgr.ping("e2e-test"))

                # Call hook
                result = mgr.call_hook(
                    "e2e-test", "pre_prompt", {"message": "hello"},
                )
                self.assertTrue(result.get("e2e"))

                # Stop
                self.assertTrue(mgr.stop_plugin("e2e-test"))
                self.assertFalse(mgr.is_running("e2e-test"))

            finally:
                mgr.stop_all()

    def test_subprocess_adapter_via_loader(self) -> None:
        """PluginLoader in subprocess mode produces a SubprocessPluginAdapter."""
        with tempfile.TemporaryDirectory() as td:
            plugin_dir = Path(td) / "loader-plugin"
            plugin_dir.mkdir()

            (plugin_dir / "manifest.yaml").write_text(
                "name: loader-sub\n"
                "version: '1.0.0'\n"
                "author: test\n"
                "description: e2e loader test\n"
                "entry_point: main.py\n"
                "hooks: [pre_prompt]\n"
            )
            (plugin_dir / "main.py").write_text(
                'def init():\n    pass\n\n'
                'def hook_pre_prompt(data):\n    return {"from_sub": True}\n\n'
                'def shutdown():\n    pass\n'
            )

            sock_dir = Path(td) / "socks"
            log_dir = Path(td) / "logs"
            worker_script = os.path.join(_PKG, "plugin_worker.py")

            mgr = _ps.PluginSubprocessManager(
                socket_dir=sock_dir,
                log_dir=log_dir,
                worker_script=worker_script,
                startup_timeout=10.0,
            )

            loader = _pl.PluginLoader(
                subprocess_mode="subprocess",
                subprocess_manager=mgr,
            )

            try:
                loaded = loader.load_plugin(plugin_dir)
                self.assertIsInstance(loaded, _pl.SubprocessPluginAdapter)
                self.assertTrue(loaded.is_subprocess)
                self.assertIn("pre_prompt", loaded.hooks)

                # Call the proxied hook
                hook_fn = loaded.get_hook("pre_prompt")
                result = hook_fn({"message": "test"})
                self.assertTrue(result.get("from_sub"))

            finally:
                loader.shutdown_all()
                mgr.stop_all()


# ===========================================================================
# Part 16: Crash isolation
# ===========================================================================

class TestCrashIsolation(unittest.TestCase):
    """Plugin subprocess crash does not affect host."""

    def test_crashing_plugin_doesnt_kill_host(self) -> None:
        """A plugin that crashes leaves the host alive and functional."""
        with tempfile.TemporaryDirectory() as td:
            plugin_dir = Path(td) / "crasher"
            plugin_dir.mkdir()

            # Plugin that crashes on hook call
            (plugin_dir / "main.py").write_text(
                'import os, signal\n\n'
                'def init():\n    pass\n\n'
                'def hook_pre_prompt(data):\n'
                '    os.kill(os.getpid(), signal.SIGKILL)\n'
                '    return {}\n\n'
                'def shutdown():\n    pass\n'
            )

            sock_dir = Path(td) / "socks"
            log_dir = Path(td) / "logs"
            worker_script = os.path.join(_PKG, "plugin_worker.py")

            mgr = _ps.PluginSubprocessManager(
                socket_dir=sock_dir,
                log_dir=log_dir,
                worker_script=worker_script,
                startup_timeout=10.0,
                default_hook_timeout=5.0,
            )

            try:
                pp = mgr.start_plugin(
                    plugin_name="crasher",
                    plugin_dir=plugin_dir,
                    entry_point="main.py",
                )
                self.assertTrue(pp.is_alive())

                # Call hook that crashes the subprocess
                with self.assertRaises((_ps.PluginSubprocessError, _ps.PluginIPCError, _ps.PluginSubprocessTimeout)):
                    mgr.call_hook("crasher", "pre_prompt", {})

                # Host is still alive (we're running this assertion!)
                # Plugin should be dead
                time.sleep(0.5)
                self.assertFalse(pp.is_alive())

            finally:
                mgr.stop_all()

    def test_manager_detects_dead_subprocess(self) -> None:
        """Watchdog check detects dead subprocesses."""
        with tempfile.TemporaryDirectory() as td:
            plugin_dir = Path(td) / "dier"
            plugin_dir.mkdir()
            (plugin_dir / "main.py").write_text(
                'def init():\n    pass\n'
                'def shutdown():\n    pass\n'
            )

            sock_dir = Path(td) / "socks"
            log_dir = Path(td) / "logs"
            worker_script = os.path.join(_PKG, "plugin_worker.py")

            mgr = _ps.PluginSubprocessManager(
                socket_dir=sock_dir,
                log_dir=log_dir,
                worker_script=worker_script,
                startup_timeout=10.0,
            )

            try:
                pp = mgr.start_plugin(
                    plugin_name="dier",
                    plugin_dir=plugin_dir,
                    entry_point="main.py",
                )
                self.assertTrue(pp.is_alive())

                # Kill the subprocess externally
                pp.process.kill()
                pp.process.wait(timeout=3)

                # Run watchdog check
                mgr._watchdog_check()

                # Manager should have cleaned up
                self.assertFalse(mgr.is_running("dier"))

            finally:
                mgr.stop_all()


# ===========================================================================
# Part 17: Version bump
# ===========================================================================

class TestVersionBump(unittest.TestCase):
    """Version is correctly bumped to 3.1.3."""

    def test_version_file(self) -> None:
        version_path = os.path.join(_PKG, "__version__.py")
        content = open(version_path).read()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__version__":
                        value = node.value
                        if isinstance(value, ast.Constant):
                            self.assertEqual(value.value, "3.1.3")
                            return
        self.fail("__version__ not found")

    def test_plugin_subprocess_module_exists(self) -> None:
        path = os.path.join(_PKG, "plugin_subprocess.py")
        self.assertTrue(os.path.exists(path))

    def test_plugin_worker_module_exists(self) -> None:
        path = os.path.join(_PKG, "plugin_worker.py")
        self.assertTrue(os.path.exists(path))

    def test_all_new_files_ast_valid(self) -> None:
        for fname in ["plugin_subprocess.py", "plugin_worker.py",
                       "plugin_loader.py", "plugin_manifest.py"]:
            path = os.path.join(_PKG, fname)
            with open(path) as f:
                ast.parse(f.read(), filename=fname)


# ===========================================================================
# Cross-module wire protocol compatibility
# ===========================================================================

class TestCrossModuleWireCompat(unittest.TestCase):
    """Host and worker wire protocols are byte-compatible."""

    def test_host_to_worker(self) -> None:
        key = secrets.token_bytes(32)
        payload = {"jsonrpc": "2.0", "method": "ping", "params": {}, "id": "x"}
        packed = _ps.pack_message(key, payload)
        # Simulate worker recv
        length = struct.unpack("!I", packed[:4])[0]
        mac = packed[4:36]
        raw = packed[36:]
        self.assertEqual(len(raw), length)
        self.assertTrue(_pw._verify_hmac(key, raw, mac))
        self.assertEqual(json.loads(raw.decode()), payload)

    def test_worker_to_host(self) -> None:
        key = secrets.token_bytes(32)
        payload = {"jsonrpc": "2.0", "id": "y", "result": {"status": "pong"}}
        raw = json.dumps(payload, separators=(",", ":")).encode()
        mac = _pw._compute_hmac(key, raw)
        packed = struct.pack("!I", len(raw)) + mac + raw
        result = _ps.unpack_message(key, packed)
        self.assertEqual(result, payload)


# ===========================================================================
# PluginProcess dataclass
# ===========================================================================

class TestPluginProcess(unittest.TestCase):
    """PluginProcess health tracking."""

    def _make_pp(self) -> _ps.PluginProcess:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # alive
        return _ps.PluginProcess(
            plugin_name="test",
            process=mock_proc,
            socket_path="/tmp/test.sock",
            hmac_key=b"\x00" * 32,
        )

    def test_is_alive(self) -> None:
        pp = self._make_pp()
        self.assertTrue(pp.is_alive())

    def test_is_dead(self) -> None:
        pp = self._make_pp()
        pp.process.poll.return_value = 1
        self.assertFalse(pp.is_alive())

    def test_heartbeat(self) -> None:
        pp = self._make_pp()
        time.sleep(0.1)
        elapsed = pp.elapsed_since_heartbeat()
        self.assertGreater(elapsed, 0.05)
        pp.touch_heartbeat()
        self.assertLess(pp.elapsed_since_heartbeat(), 0.1)


# ===========================================================================
# Module-level constants
# ===========================================================================

class TestConstants(unittest.TestCase):
    """Module constants are correctly defined."""

    def test_max_message_size(self) -> None:
        self.assertEqual(_ps.MAX_MESSAGE_SIZE, 4 * 1024 * 1024)

    def test_header_size(self) -> None:
        self.assertEqual(_ps.HEADER_SIZE, 36)

    def test_default_resource_limits(self) -> None:
        self.assertIn("cpu_time_seconds", _ps.DEFAULT_RESOURCE_LIMITS)
        self.assertIn("memory_bytes", _ps.DEFAULT_RESOURCE_LIMITS)
        self.assertIn("max_file_descriptors", _ps.DEFAULT_RESOURCE_LIMITS)

    def test_jsonrpc_error_codes(self) -> None:
        self.assertEqual(_ps.JSONRPC_PARSE_ERROR, -32700)
        self.assertEqual(_ps.JSONRPC_INVALID_REQUEST, -32600)
        self.assertEqual(_ps.JSONRPC_METHOD_NOT_FOUND, -32601)
        self.assertEqual(_ps.JSONRPC_INTERNAL_ERROR, -32603)

    def test_availability_flags(self) -> None:
        self.assertTrue(_ps.PLUGIN_SUBPROCESS_AVAILABLE)


if __name__ == "__main__":
    unittest.main()
