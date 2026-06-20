#!/usr/bin/env python3
"""
Tests for S195 F8a -- plugin manifest / loader / installer / allowlist / index.

Per-fix coverage:
- PI-10: enable_plugin flips registry state only after a successful load
- PI-11: install with auto_enable routes through the full enable flow
         (hooks registered with the HookManager)
- PI-12: resource_limits round-trips through the registry DB (+ migration)
- PI-13: PluginIndex restores the persisted last_refresh on construction
- PI-14: re-assertion of the S101 sandbox-blocks contract with explicit
         in-process mode (supersedes the two deselected S101 tests)
- PI-15: uninstall_plugin reports the actual effect (False for unknown)
- PI-21: _discover_builtins refreshes registry records on version change
- PI-08: archive extraction rejects sibling-prefix traversal
         (is_relative_to instead of str.startswith)

Loader idiom: spec_from_file_location with sys.modules registration BEFORE
exec_module; package stub with real __path__ so lazy absolute imports
resolve; opti_oignon.config and opti_oignon.db_utils pre-seeded; stubs and
modules ADDED by this file are cleaned from sys.modules at module teardown
(S194 hardening -- prevents cross-suite pollution).
"""

import importlib.util
import io
import json
import sqlite3
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Module loading (S194-hardened idiom)
# ---------------------------------------------------------------------------

_ADDED_MODULES: list[str] = []
_TMP_DATA_DIR = tempfile.mkdtemp(prefix="oo_s195_f8a_")


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
    if "opti_oignon.security_mode" not in sys.modules:
        sec = ModuleType("opti_oignon.security_mode")
        sec.is_bulbe = lambda: False
        sec._audit_log = lambda *a, **kw: None
        _seed_stub("opti_oignon.security_mode", sec)


def _load(name: str, relpath: str) -> ModuleType:
    """Load (or reuse) a module under its package name."""
    if name in sys.modules:
        return sys.modules[name]
    _seed_common_stubs()
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register BEFORE exec_module (3.12+ dataclasses)
    _ADDED_MODULES.append(name)
    spec.loader.exec_module(mod)
    return mod


manifest_mod = _load(
    "opti_oignon.plugin_manifest", "opti_oignon/plugin_manifest.py",
)
hooks_mod = _load(
    "opti_oignon.plugin_hooks", "opti_oignon/plugin_hooks.py",
)
loader_mod = _load(
    "opti_oignon.plugin_loader", "opti_oignon/plugin_loader.py",
)
index_mod = _load(
    "opti_oignon.plugin_index", "opti_oignon/plugin_index.py",
)
installer_mod = _load(
    "opti_oignon.plugin_installer", "opti_oignon/plugin_installer.py",
)

PluginRegistry = manifest_mod.PluginRegistry
PluginManifest = manifest_mod.PluginManifest
PluginLoader = loader_mod.PluginLoader
PluginLoadError = loader_mod.PluginLoadError
PluginSandboxViolation = loader_mod.PluginSandboxViolation
PluginIndex = index_mod.PluginIndex
RemotePluginInstaller = installer_mod.RemotePluginInstaller
PluginInstallError = installer_mod.PluginInstallError
hook_manager = hooks_mod.hook_manager


@pytest.fixture(scope="module", autouse=True)
def _cleanup_added_modules():
    """S194 hardening: remove the modules/stubs this file added."""
    yield
    for name in _ADDED_MODULES:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _clean_hooks():
    hook_manager.clear()
    yield
    hook_manager.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manifest_dict(name: str = "demo-plugin", version: str = "1.0.0",
                   hooks: list | None = None) -> dict:
    return {
        "name": name,
        "version": version,
        "author": "tester",
        "description": "test plugin",
        "entry_point": "entry_point.py",
        "hooks": hooks or [],
    }


def _write_plugin(pdir: Path, name: str, *, hooks: list | None = None,
                  entry_body: str = "VALUE = 1\n",
                  version: str = "1.0.0") -> Path:
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "manifest.yaml").write_text(
        yaml.dump(_manifest_dict(name, version=version, hooks=hooks))
    )
    (pdir / "entry_point.py").write_text(entry_body)
    return pdir


@pytest.fixture
def registry(tmp_path):
    return PluginRegistry(db_path=tmp_path / "plugins.db")


# ---------------------------------------------------------------------------
# PI-10 -- enable flips state only after a successful load
# ---------------------------------------------------------------------------

class TestPI10EnableStateAfterLoad:
    def test_enable_failure_leaves_state_installed(self, tmp_path, registry):
        m = PluginManifest.from_dict(_manifest_dict("ghost-dir"))
        registry.register(m, str(tmp_path / "does-not-exist"))
        loader = PluginLoader(registry=registry, subprocess_mode="inprocess")

        with pytest.raises(PluginLoadError):
            loader.enable_plugin("ghost-dir")

        assert registry.get("ghost-dir").state == "installed"

    def test_enable_success_sets_enabled(self, tmp_path, registry):
        pdir = _write_plugin(tmp_path / "okp", "okp")
        m = PluginManifest.from_dict(_manifest_dict("okp"))
        registry.register(m, str(pdir))
        loader = PluginLoader(registry=registry, subprocess_mode="inprocess")

        loaded = loader.enable_plugin("okp")
        try:
            assert loaded is not None
            assert registry.get("okp").state == "enabled"
        finally:
            loader.unload_plugin("okp")

    def test_source_order_load_before_state_flip(self):
        src = (ROOT / "opti_oignon" / "plugin_loader.py").read_text()
        idx = src.index("def enable_plugin")
        chunk = src[idx:idx + 900]
        assert chunk.index("load_plugin") < chunk.index('set_state(name, "enabled")')


# ---------------------------------------------------------------------------
# PI-11 -- install with auto_enable registers hooks
# ---------------------------------------------------------------------------

class TestPI11InstallRegistersHooks:
    def test_install_auto_enable_registers_hooks(self, tmp_path, registry):
        src_dir = _write_plugin(
            tmp_path / "src" / "hooky", "hooky",
            hooks=["pre_prompt"],
            entry_body=(
                "def hook_pre_prompt(context):\n"
                "    return {\"tag\": \"hooked\"}\n"
            ),
        )
        base = tmp_path / "base"
        base.mkdir()
        loader = PluginLoader(
            registry=registry,
            plugins_base_dir=base,
            subprocess_mode="inprocess",
        )

        loaded = loader.install_plugin(src_dir, auto_enable=True)
        try:
            assert loaded is not None
            assert registry.get("hooky").state == "enabled"
            registered = hook_manager.list_hooks(plugin_name="hooky")
            assert any(h["hook_name"] == "pre_prompt" for h in registered)
            assert hook_manager.get_hook_count() >= 1
        finally:
            loader.uninstall_plugin("hooky")

    def test_install_without_auto_enable_stays_installed(
        self, tmp_path, registry,
    ):
        src_dir = _write_plugin(tmp_path / "src" / "calm", "calm")
        base = tmp_path / "base"
        base.mkdir()
        loader = PluginLoader(
            registry=registry,
            plugins_base_dir=base,
            subprocess_mode="inprocess",
        )

        result = loader.install_plugin(src_dir, auto_enable=False)
        assert result is None
        assert registry.get("calm").state == "installed"
        assert hook_manager.get_hook_count() == 0


# ---------------------------------------------------------------------------
# PI-12 -- resource_limits round-trips through the registry DB
# ---------------------------------------------------------------------------

class TestPI12ResourceLimitsRoundTrip:
    def test_round_trip(self, tmp_path):
        db = tmp_path / "plugins.db"
        r1 = PluginRegistry(db_path=db)
        data = _manifest_dict("limited")
        data["resource_limits"] = {"cpu_time_seconds": 12, "memory_bytes": 1024}
        m = PluginManifest.from_dict(data)
        r1.register(m, "/tmp/limited")

        r2 = PluginRegistry(db_path=db)
        rec = r2.get("limited")
        assert rec is not None
        assert rec.manifest.resource_limits == {
            "cpu_time_seconds": 12, "memory_bytes": 1024,
        }

    def test_migration_from_pre_s195_schema(self, tmp_path):
        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE plugins ("
            " name TEXT PRIMARY KEY,"
            " version TEXT NOT NULL,"
            " author TEXT NOT NULL DEFAULT '',"
            " description TEXT NOT NULL DEFAULT '',"
            " entry_point TEXT NOT NULL DEFAULT 'entry_point.py',"
            " hooks TEXT NOT NULL DEFAULT '[]',"
            " dependencies TEXT NOT NULL DEFAULT '[]',"
            " permissions TEXT NOT NULL DEFAULT '[]',"
            " min_opti_version TEXT NOT NULL DEFAULT '1.0.0',"
            " config_schema TEXT NOT NULL DEFAULT '{}',"
            " state TEXT NOT NULL DEFAULT 'installed',"
            " plugin_dir TEXT NOT NULL DEFAULT '',"
            " config TEXT NOT NULL DEFAULT '{}',"
            " installed_at REAL NOT NULL DEFAULT 0,"
            " updated_at REAL NOT NULL DEFAULT 0)"
        )
        conn.execute(
            "INSERT INTO plugins (name, version, author, description) "
            "VALUES ('legacy', '0.9.0', 'old', 'pre-S195 row')"
        )
        conn.commit()
        conn.close()

        reg = PluginRegistry(db_path=db)  # guarded ALTER must run
        legacy = reg.get("legacy")
        assert legacy is not None
        assert legacy.manifest.resource_limits == {}

        data = _manifest_dict("fresh")
        data["resource_limits"] = {"max_file_descriptors": 32}
        reg.register(PluginManifest.from_dict(data), "/tmp/fresh")

        reg2 = PluginRegistry(db_path=db)
        assert reg2.get("fresh").manifest.resource_limits == {
            "max_file_descriptors": 32,
        }


# ---------------------------------------------------------------------------
# PI-13 -- index last_refresh restored on construction
# ---------------------------------------------------------------------------

class TestPI13IndexLastRefreshPersistence:
    def test_fresh_index_is_stale(self, tmp_path):
        idx = PluginIndex(db_path=tmp_path / "idx.db", cache_ttl=3600)
        assert idx.is_stale is True

    def test_restart_restores_last_refresh(self, tmp_path):
        db = tmp_path / "idx.db"
        PluginIndex(db_path=db, cache_ttl=3600)  # create schema

        now = time.time()
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
            ("last_refresh", str(now)),
        )
        conn.commit()
        conn.close()

        idx2 = PluginIndex(db_path=db, cache_ttl=3600)
        assert idx2._last_refresh == pytest.approx(now)
        assert idx2.is_stale is False

    def test_garbage_meta_value_falls_back_to_stale(self, tmp_path):
        db = tmp_path / "idx.db"
        PluginIndex(db_path=db, cache_ttl=3600)
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
            ("last_refresh", "not-a-float"),
        )
        conn.commit()
        conn.close()

        idx2 = PluginIndex(db_path=db, cache_ttl=3600)
        assert idx2.is_stale is True


# ---------------------------------------------------------------------------
# PI-14 -- sandbox-blocks re-assertion (explicit in-process mode)
# Supersedes the two deselected tests in test_plugins_s101.py, which relied
# on the in-process path being taken implicitly (mode "auto" now prefers the
# subprocess path, where import policy is a separate concern -- see F8c).
# ---------------------------------------------------------------------------

class TestPI14SandboxBlocksInprocess:
    def test_blocks_subprocess_import(self, tmp_path):
        pdir = _write_plugin(
            tmp_path / "evil", "evil-plugin",
            entry_body="import subprocess\n",
        )
        loader = PluginLoader(subprocess_mode="inprocess")
        with pytest.raises(PluginSandboxViolation, match="subprocess"):
            loader.load_plugin(pdir, sandbox=True)

    def test_blocks_ctypes_import(self, tmp_path):
        pdir = _write_plugin(
            tmp_path / "evilc", "ctypes-plugin",
            entry_body="import ctypes\n",
        )
        loader = PluginLoader(subprocess_mode="inprocess")
        with pytest.raises(PluginSandboxViolation, match="ctypes"):
            loader.load_plugin(pdir, sandbox=True)


# ---------------------------------------------------------------------------
# PI-15 -- uninstall reports the actual effect
# ---------------------------------------------------------------------------

class TestPI15UninstallReturnSemantics:
    def test_unknown_plugin_returns_false(self, registry):
        loader = PluginLoader(registry=registry, subprocess_mode="inprocess")
        assert loader.uninstall_plugin("ghost") is False

    def test_registered_plugin_returns_true(self, tmp_path, registry):
        pdir = _write_plugin(tmp_path / "gone", "gone")
        m = PluginManifest.from_dict(_manifest_dict("gone"))
        registry.register(m, str(pdir))
        loader = PluginLoader(registry=registry, subprocess_mode="inprocess")

        assert loader.uninstall_plugin("gone") is True
        assert registry.get("gone") is None


# ---------------------------------------------------------------------------
# PI-21 -- builtin manifest updates refresh the registry record
# ---------------------------------------------------------------------------

class TestPI21BuiltinRefresh:
    def test_version_bump_propagates_and_preserves_state(self, tmp_path):
        builtin_dir = tmp_path / "plugins"
        _write_plugin(builtin_dir / "bdemo", "bdemo", version="1.0.0")
        reg = PluginRegistry(db_path=tmp_path / "plugins.db")

        added = manifest_mod._discover_builtins(reg, builtin_dir=builtin_dir)
        assert added == 1
        rec = reg.get("bdemo")
        assert rec.manifest.version == "1.0.0"
        assert rec.state == "enabled"
        installed_at = rec.installed_at

        (builtin_dir / "bdemo" / "manifest.yaml").write_text(
            yaml.dump(_manifest_dict("bdemo", version="1.1.0"))
        )
        added2 = manifest_mod._discover_builtins(reg, builtin_dir=builtin_dir)
        assert added2 == 0  # refresh, not a new registration
        rec2 = reg.get("bdemo")
        assert rec2.manifest.version == "1.1.0"
        assert rec2.state == "enabled"
        assert rec2.installed_at == installed_at

    def test_same_version_is_left_untouched(self, tmp_path):
        builtin_dir = tmp_path / "plugins"
        _write_plugin(builtin_dir / "still", "still", version="2.0.0")
        reg = PluginRegistry(db_path=tmp_path / "plugins.db")
        manifest_mod._discover_builtins(reg, builtin_dir=builtin_dir)
        before = reg.get("still").updated_at

        manifest_mod._discover_builtins(reg, builtin_dir=builtin_dir)
        assert reg.get("still").updated_at == before


# ---------------------------------------------------------------------------
# PI-08 -- extraction rejects sibling-prefix traversal
# ---------------------------------------------------------------------------

class TestPI08ExtractionTraversal:
    @pytest.fixture
    def installer(self, tmp_path):
        return RemotePluginInstaller(plugins_dir=tmp_path / "plugins")

    def test_zip_sibling_prefix_rejected(self, tmp_path, installer):
        dest = tmp_path / "extracted"
        dest.mkdir()
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../extracted_evil/pwn.txt", "x")

        with pytest.raises(PluginInstallError, match="traversal"):
            installer._extract(archive, dest)
        assert not (tmp_path / "extracted_evil").exists()

    def test_tar_sibling_prefix_rejected(self, tmp_path, installer):
        dest = tmp_path / "extracted"
        dest.mkdir()
        archive = tmp_path / "evil.tar.gz"
        payload = b"x"
        with tarfile.open(archive, "w:gz") as tf:
            info = tarfile.TarInfo(name="../extracted_evil/pwn.txt")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))

        with pytest.raises(PluginInstallError, match="traversal"):
            installer._extract(archive, dest)
        assert not (tmp_path / "extracted_evil").exists()

    def test_clean_zip_extracts(self, tmp_path, installer):
        dest = tmp_path / "extracted"
        dest.mkdir()
        archive = tmp_path / "ok.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("plug/manifest.yaml", "name: x\n")
            zf.writestr("plug/entry_point.py", "VALUE = 1\n")

        installer._extract(archive, dest)
        assert (dest / "plug" / "manifest.yaml").exists()


# ---------------------------------------------------------------------------
# PI-11 (remote installer leg) -- enable flow used when loader available
# ---------------------------------------------------------------------------

class TestPI11RemoteInstallerEnableFlow:
    def test_source_routes_through_enable_plugin(self):
        src = (ROOT / "opti_oignon" / "plugin_installer.py").read_text()
        idx = src.index("def install_from_url")
        chunk = src[idx:idx + 4200]
        assert "enable_plugin" in chunk
        assert 'auto_enable=False' in chunk
