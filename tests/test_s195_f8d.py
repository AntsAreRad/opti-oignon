#!/usr/bin/env python3
"""
Tests for S195 F8d -- plugin user config / reviews / template.

Per-fix coverage:
- TPL-1: scaffold generator validates the plugin name (rejects path
         traversal and injection; valid names still generate)
- TPL-2: available_permissions() is derived from VALID_PERMISSIONS (the
         three S124 permissions are present)
- PUC-2: concurrent set_config inserts do not crash with IntegrityError

Loader idiom: spec_from_file_location with sys.modules registration BEFORE
exec_module; package stub with real __path__; opti_oignon.config /
opti_oignon.db_utils pre-seeded; added modules/stubs cleaned at module
teardown (S194 hardening).
"""

import importlib.util
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent

_ADDED_MODULES: list[str] = []
_TMP_DATA_DIR = tempfile.mkdtemp(prefix="oo_s195_f8d_")


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


manifest_mod = _load(
    "opti_oignon.plugin_manifest", "opti_oignon/plugin_manifest.py",
)
template_mod = _load(
    "opti_oignon.plugin_template", "opti_oignon/plugin_template.py",
)
puc_mod = _load(
    "opti_oignon.plugin_user_config", "opti_oignon/plugin_user_config.py",
)

PluginTemplateGenerator = template_mod.PluginTemplateGenerator
PluginUserConfigStore = puc_mod.PluginUserConfigStore


@pytest.fixture(scope="module", autouse=True)
def _cleanup_added_modules():
    yield
    for name in _ADDED_MODULES:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# TPL-1 -- scaffold name validation
# ---------------------------------------------------------------------------

class TestTPL1NameValidation:
    @pytest.mark.parametrize("bad", [
        "../escape",
        "a/b",
        "..",
        "/etc/passwd",
        'evil"\nname: hacked',
        "UPPER",
        "x",          # too short (min 2)
        "",
    ])
    def test_invalid_name_rejected_no_files(self, tmp_path, bad):
        gen = PluginTemplateGenerator(output_base_dir=tmp_path / "plugins")
        result = gen.generate(name=bad)
        assert result["success"] is False
        assert "Invalid plugin name" in result["error"]
        # Nothing must be written, in particular nothing OUTSIDE the base
        assert not (tmp_path / "escape").exists()
        assert list((tmp_path / "plugins").glob("**/*")) == [] or \
            not any(p.is_file() for p in (tmp_path / "plugins").glob("**/*"))

    def test_valid_name_generates(self, tmp_path):
        gen = PluginTemplateGenerator(output_base_dir=tmp_path / "plugins")
        result = gen.generate(name="my-plugin", hooks=["post_inference"])
        assert result["success"] is True
        target = tmp_path / "plugins" / "my-plugin"
        assert (target / "manifest.yaml").exists()
        assert (target / "entry_point.py").exists()
        assert (target / "README.md").exists()


# ---------------------------------------------------------------------------
# TPL-2 -- available_permissions derived from VALID_PERMISSIONS
# ---------------------------------------------------------------------------

class TestTPL2PermissionsParity:
    def test_matches_valid_permissions(self):
        # Supersedes test_plugin_marketplace_s102::test_available_permissions,
        # which asserted the stale count of 9 (deselect-plus-reassert).
        perms = PluginTemplateGenerator.available_permissions()
        assert set(perms) == set(manifest_mod.VALID_PERMISSIONS)
        assert len(perms) == len(manifest_mod.VALID_PERMISSIONS)
        assert "conversation_read" in perms
        assert "network_outbound" in perms

    def test_s124_permissions_present(self):
        perms = set(PluginTemplateGenerator.available_permissions())
        assert {"filesystem_read", "filesystem_write", "inference_content"} <= perms


# ---------------------------------------------------------------------------
# PUC-2 -- concurrent insert race does not crash
# ---------------------------------------------------------------------------

class TestPUC2InsertRace:
    def test_basic_insert_then_update(self, tmp_path):
        store = PluginUserConfigStore(db_path=tmp_path / "puc.db")
        store.set_config("u1", "p1", enabled=True, preferences={"a": 1})
        store.set_config("u1", "p1", preferences={"b": 2})
        cfg = store.get_config("u1", "p1")
        assert cfg["preferences"] == {"a": 1, "b": 2}
        assert cfg["enabled"] is True

    def test_concurrent_first_writes_do_not_raise(self, tmp_path):
        store = PluginUserConfigStore(db_path=tmp_path / "puc.db")
        n = 8
        barrier = threading.Barrier(n)
        errors: list[Exception] = []

        def worker(idx: int):
            try:
                barrier.wait(timeout=5)
                store.set_config("shared-user", "shared-plugin",
                                 preferences={f"k{idx}": idx})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"set_config raised under contention: {errors}"
        cfg = store.get_config("shared-user", "shared-plugin")
        assert cfg is not None

    def test_per_user_isolation_and_wipe(self, tmp_path):
        store = PluginUserConfigStore(db_path=tmp_path / "puc.db")
        store.set_config("alice", "p1", preferences={"x": 1})
        store.set_config("bob", "p1", preferences={"y": 2})
        assert store.get_config("alice", "p1")["preferences"] == {"x": 1}
        assert store.get_config("bob", "p1")["preferences"] == {"y": 2}

        removed = store.delete_all_configs("alice")
        assert removed == 1
        assert store.get_config("alice", "p1") is None
        # bob untouched
        assert store.get_config("bob", "p1")["preferences"] == {"y": 2}
