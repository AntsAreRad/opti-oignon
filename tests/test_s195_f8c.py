#!/usr/bin/env python3
"""
Tests for S195 F8c -- plugin subprocess manager / worker.

Per-fix coverage:
- PSB-02: a timed-out or IPC-failed hook RPC tears the worker down
          (channel desync prevention); a clean JSON-RPC error response
          does NOT kill the worker
- PSB-04: the worker is launched with sys.executable (not a bare
          "python3" resolved via PATH)
- PSB-05: the watchdog lazily starts with the first worker and
          start_watchdog() is idempotent
- end-to-end: real worker round trip (launch, hook call over the
          HMAC-framed socket protocol, ping, watchdog alive, stop_all)

Loader idiom: spec_from_file_location with sys.modules registration BEFORE
exec_module; package stub with real __path__; opti_oignon.config pre-seeded;
added modules/stubs cleaned at module teardown (S194 hardening).
"""

import importlib.util
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent

_ADDED_MODULES: list[str] = []
_TMP_DATA_DIR = tempfile.mkdtemp(prefix="oo_s195_f8c_")


def _seed_stub(name: str, mod: ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = mod
        _ADDED_MODULES.append(name)


def _seed_common_stubs() -> None:
    if "opti_oignon" not in sys.modules:
        pkg = ModuleType("opti_oignon")
        pkg.__path__ = [str(ROOT / "opti_oignon")]
        _seed_stub("opti_oignon", pkg)
    if "opti_oignon.config" not in sys.modules:
        cfg = ModuleType("opti_oignon.config")
        cfg.DATA_DIR = _TMP_DATA_DIR
        _seed_stub("opti_oignon.config", cfg)
    if "opti_oignon.db_utils" not in sys.modules:
        dbu = ModuleType("opti_oignon.db_utils")
        dbu.safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)
        _seed_stub("opti_oignon.db_utils", dbu)


def _load(name: str, relpath: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    _seed_common_stubs()
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register BEFORE exec_module (3.12+ dataclasses)
    _ADDED_MODULES.append(name)
    spec.loader.exec_module(mod)
    return mod


sub_mod = _load(
    "opti_oignon.plugin_subprocess", "opti_oignon/plugin_subprocess.py",
)

PluginSubprocessManager = sub_mod.PluginSubprocessManager
PluginSubprocessError = sub_mod.PluginSubprocessError
PluginSubprocessTimeout = sub_mod.PluginSubprocessTimeout
PluginIPCError = sub_mod.PluginIPCError


@pytest.fixture(scope="module", autouse=True)
def _cleanup_added_modules():
    yield
    for name in _ADDED_MODULES:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Fakes for PSB-02 unit tests
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self):
        self.returncode = None
        self.killed = False

    def poll(self):
        return -9 if self.killed else None

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class _FakePP:
    def __init__(self, name="victim"):
        self.plugin_name = name
        self.process = _FakeProc()
        self.socket_path = "/tmp/oo_fake_never_exists.sock"
        self.hmac_key = b"0" * 32
        self.conn = None
        self.plugin_logger = None
        self._lock = threading.Lock()

    def is_alive(self):
        return self.process.poll() is None

    def touch_heartbeat(self):
        pass


@pytest.fixture
def mgr_with_fake(monkeypatch, tmp_path):
    mgr = PluginSubprocessManager(
        socket_dir=tmp_path / "sock", log_dir=tmp_path / "logs",
    )
    fake = _FakePP("victim")
    mgr._processes["victim"] = fake
    yield mgr, fake
    mgr._processes.clear()
    mgr.stop_watchdog()


# ---------------------------------------------------------------------------
# PSB-02 -- channel desync prevention
# ---------------------------------------------------------------------------

class TestPSB02DesyncTeardown:
    def test_timeout_kills_worker_and_reraises(
        self, monkeypatch, mgr_with_fake,
    ):
        mgr, fake = mgr_with_fake

        def boom(pp, method, params, timeout=10.0):
            raise PluginSubprocessTimeout("simulated hook timeout")

        monkeypatch.setattr(mgr, "_rpc_call", boom)
        with pytest.raises(PluginSubprocessTimeout):
            mgr.call_hook("victim", "pre_prompt", {"x": 1})

        assert "victim" not in mgr._processes
        assert fake.process.killed is True

    def test_ipc_error_kills_worker_and_reraises(
        self, monkeypatch, mgr_with_fake,
    ):
        mgr, fake = mgr_with_fake

        def boom(pp, method, params, timeout=10.0):
            raise PluginIPCError("simulated ID mismatch")

        monkeypatch.setattr(mgr, "_rpc_call", boom)
        with pytest.raises(PluginIPCError):
            mgr.call_hook("victim", "pre_prompt", {})

        assert "victim" not in mgr._processes
        assert fake.process.killed is True

    def test_clean_rpc_error_response_does_not_kill(
        self, monkeypatch, mgr_with_fake,
    ):
        """A worker-side hook failure arrives as a JSON-RPC error and is
        raised as a bare PluginSubprocessError: the channel stays in sync
        and the worker must NOT be torn down."""
        mgr, fake = mgr_with_fake

        def hook_failed(pp, method, params, timeout=10.0):
            raise PluginSubprocessError(
                "Plugin RPC error [-32603]: Hook execution failed: kaput"
            )

        monkeypatch.setattr(mgr, "_rpc_call", hook_failed)
        with pytest.raises(PluginSubprocessError):
            mgr.call_hook("victim", "pre_prompt", {})

        assert "victim" in mgr._processes
        assert fake.process.killed is False

    def test_success_path_unchanged(self, monkeypatch, mgr_with_fake):
        mgr, fake = mgr_with_fake
        monkeypatch.setattr(
            mgr, "_rpc_call",
            lambda pp, method, params, timeout=10.0: {"ok": 1},
        )
        assert mgr.call_hook("victim", "pre_prompt", {}) == {"ok": 1}
        assert "victim" in mgr._processes


# ---------------------------------------------------------------------------
# PSB-04 / PSB-05 -- structural locks
# ---------------------------------------------------------------------------

class TestPSB04PSB05Structure:
    def test_start_plugin_uses_sys_executable(self):
        src = (ROOT / "opti_oignon" / "plugin_subprocess.py").read_text()
        idx = src.index("def start_plugin")
        chunk = src[idx:src.index("def stop_plugin")]
        assert "sys.executable," in chunk
        # The argv element form (with trailing comma) must be gone; the
        # explanatory comment may still mention the bare word.
        assert '"python3",' not in chunk

    def test_start_plugin_starts_watchdog(self):
        src = (ROOT / "opti_oignon" / "plugin_subprocess.py").read_text()
        idx = src.index("def start_plugin")
        chunk = src[idx:src.index("def stop_plugin")]
        assert "self.start_watchdog()" in chunk

    def test_start_watchdog_idempotent(self, tmp_path):
        mgr = PluginSubprocessManager(
            socket_dir=tmp_path / "sock", log_dir=tmp_path / "logs",
        )
        try:
            mgr.start_watchdog()
            t1 = mgr._watchdog_thread
            assert t1 is not None and t1.is_alive()
            mgr.start_watchdog()
            assert mgr._watchdog_thread is t1
        finally:
            mgr.stop_watchdog()
        assert mgr._watchdog_thread is None


# ---------------------------------------------------------------------------
# End-to-end -- real worker round trip
# ---------------------------------------------------------------------------

class TestRealWorkerRoundTrip:
    def test_launch_hook_ping_watchdog_stop(self, tmp_path):
        pdir = tmp_path / "rt-plugin"
        pdir.mkdir()
        (pdir / "entry_point.py").write_text(
            "def hook_pre_prompt(data):\n"
            "    return {\"echo\": data.get(\"x\"), \"status\": \"ok\"}\n"
        )

        mgr = PluginSubprocessManager(log_dir=tmp_path / "logs")
        try:
            mgr.start_plugin("rt-plugin", pdir, "entry_point.py")
            assert mgr.is_running("rt-plugin")

            # PSB-05: the watchdog came up with the first worker
            assert mgr._watchdog_thread is not None
            assert mgr._watchdog_thread.is_alive()

            # HMAC-framed socket protocol round trip (also proves the
            # PSB-04 sys.executable launch works)
            res = mgr.call_hook("rt-plugin", "pre_prompt", {"x": 41})
            assert res.get("echo") == 41
            assert mgr.ping("rt-plugin") is True
        finally:
            mgr.stop_all()

        assert not mgr.is_running("rt-plugin")
        assert mgr._watchdog_thread is None

    def test_worker_hook_exception_is_clean_rpc_error(self, tmp_path):
        """A raising hook inside the worker must surface as a clean
        PluginSubprocessError (JSON-RPC error), and the worker survives
        (PSB-02 boundary: no teardown on clean errors)."""
        pdir = tmp_path / "boom-plugin"
        pdir.mkdir()
        (pdir / "entry_point.py").write_text(
            "def hook_pre_prompt(data):\n"
            "    raise RuntimeError(\"kaput\")\n"
        )

        mgr = PluginSubprocessManager(log_dir=tmp_path / "logs")
        try:
            mgr.start_plugin("boom-plugin", pdir, "entry_point.py")
            with pytest.raises(PluginSubprocessError, match="kaput"):
                mgr.call_hook("boom-plugin", "pre_prompt", {})
            # Clean error: worker still up and usable
            assert mgr.is_running("boom-plugin")
            assert mgr.ping("boom-plugin") is True
        finally:
            mgr.stop_all()
