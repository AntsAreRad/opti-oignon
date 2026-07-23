#!/usr/bin/env python3
"""Tests for plugin_manifest: PluginManifest validation + PluginRegistry.

Two load-bearing, security-relevant surfaces:

  * PluginManifest.validate -- the gate that rejects malformed or unsafe
    manifests before a plugin is ever registered. Security-critical rules:
    entry_point path traversal ("../x.py" / "/abs.py"), the hook allowlist,
    and the permission allowlist.
  * PluginRegistry -- the SQLite-backed install/enable/disable/uninstall
    state machine, including persistence across process restarts (a new
    registry on the same DB must reload the records).

Registry tests use pytest's tmp_path so every case gets a fresh database.
"""

import pytest

from opti_oignon.plugin_manifest import (
    PLUGIN_STATE_DISABLED,
    PLUGIN_STATE_ENABLED,
    PLUGIN_STATE_INSTALLED,
    PluginManifest,
    PluginManifestError,
    PluginRegistry,
)

# A minimal valid manifest dict; tests override individual fields.
_BASE = {
    "name": "my-plugin",
    "version": "1.0.0",
    "author": "tester",
    "description": "a test plugin",
    "entry_point": "entry_point.py",
    "hooks": ["pre_inference"],
    "permissions": ["conversation_read"],
}


def _manifest(**over) -> PluginManifest:
    data = dict(_BASE)
    data.update(over)
    return PluginManifest.from_dict(data)


# ===========================================================================
# PluginManifest.from_dict / validate
# ===========================================================================

def test_from_dict_valid():
    m = _manifest()
    assert m.name == "my-plugin"
    assert m.version == "1.0.0"
    assert m.entry_point == "entry_point.py"
    assert m.hooks == ["pre_inference"]
    assert m.permissions == ["conversation_read"]
    assert m.min_opti_version == "1.0.0"  # default


def test_from_dict_missing_required_raises():
    data = dict(_BASE)
    del data["name"]
    with pytest.raises(PluginManifestError):
        PluginManifest.from_dict(data)


def test_from_dict_empty_required_raises():
    with pytest.raises(PluginManifestError):
        _manifest(author="")


@pytest.mark.parametrize("bad_name", ["BadName", "1leading", "x", "no spaces", "-startshyphen"])
def test_validate_rejects_bad_name(bad_name):
    with pytest.raises(PluginManifestError):
        _manifest(name=bad_name)


@pytest.mark.parametrize("bad_version", ["1.0", "v1.0.0", "1.0.0.0", "latest", ""])
def test_validate_rejects_bad_version(bad_version):
    with pytest.raises(PluginManifestError):
        _manifest(version=bad_version)


# --- entry_point safety (security) ---

def test_validate_rejects_entry_point_traversal():
    with pytest.raises(PluginManifestError):
        _manifest(entry_point="../evil.py")


def test_validate_rejects_entry_point_absolute():
    with pytest.raises(PluginManifestError):
        _manifest(entry_point="/etc/passwd.py")


def test_validate_rejects_non_py_entry_point():
    with pytest.raises(PluginManifestError):
        _manifest(entry_point="entry_point.txt")


# --- allowlists (security) ---

def test_validate_rejects_invalid_hook():
    with pytest.raises(PluginManifestError):
        _manifest(hooks=["pre_inference", "not_a_real_hook"])


def test_validate_rejects_invalid_permission():
    with pytest.raises(PluginManifestError):
        _manifest(permissions=["root_access"])


# --- resource_limits ---

def test_validate_rejects_unknown_resource_limit_key():
    with pytest.raises(PluginManifestError):
        _manifest(resource_limits={"unlimited_power": 1})


def test_validate_rejects_negative_resource_limit():
    with pytest.raises(PluginManifestError):
        _manifest(resource_limits={"cpu_time_seconds": -1})


def test_validate_accepts_valid_resource_limits():
    m = _manifest(resource_limits={"cpu_time_seconds": 5, "memory_bytes": 1024})
    assert m.resource_limits["cpu_time_seconds"] == 5


def test_to_dict_roundtrip():
    m = _manifest(hooks=["pre_inference", "post_inference"], permissions=["conversation_read"])
    m2 = PluginManifest.from_dict(m.to_dict())
    assert m2.to_dict() == m.to_dict()


# ===========================================================================
# PluginRegistry state machine + persistence
# ===========================================================================

def _registry(tmp_path) -> PluginRegistry:
    return PluginRegistry(tmp_path / "plugins.db")


def test_register_installed_by_default(tmp_path):
    reg = _registry(tmp_path)
    rec = reg.register(_manifest(), plugin_dir="/plugins/my-plugin")
    assert rec.state == PLUGIN_STATE_INSTALLED
    assert reg.get("my-plugin").state == PLUGIN_STATE_INSTALLED
    assert reg.get("nope") is None


def test_register_auto_enable(tmp_path):
    reg = _registry(tmp_path)
    rec = reg.register(_manifest(), plugin_dir="/p", auto_enable=True)
    assert rec.state == PLUGIN_STATE_ENABLED


def test_register_update_preserves_installed_at(tmp_path):
    reg = _registry(tmp_path)
    r1 = reg.register(_manifest(version="1.0.0"), plugin_dir="/p")
    installed_at = r1.installed_at
    r2 = reg.register(_manifest(version="2.0.0"), plugin_dir="/p")
    assert r2.manifest.version == "2.0.0"
    assert r2.installed_at == installed_at        # preserved across update
    assert reg.plugin_count == 1                # same plugin, not a new row


def test_list_plugins_by_state(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_manifest(name="enabled-one"), plugin_dir="/p", auto_enable=True)
    reg.register(_manifest(name="installed-one"), plugin_dir="/p")
    enabled = reg.list_plugins(state=PLUGIN_STATE_ENABLED)
    assert [r.manifest.name for r in enabled] == ["enabled-one"]
    assert len(reg.list_plugins()) == 2           # no filter -> all


def test_set_state_transition(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_manifest(), plugin_dir="/p", auto_enable=True)
    assert reg.set_state("my-plugin", PLUGIN_STATE_DISABLED) is True
    assert reg.get("my-plugin").state == PLUGIN_STATE_DISABLED


def test_set_state_invalid_raises(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_manifest(), plugin_dir="/p")
    with pytest.raises(ValueError):
        reg.set_state("my-plugin", "bogus_state")


def test_set_state_unknown_plugin_returns_false(tmp_path):
    reg = _registry(tmp_path)
    assert reg.set_state("ghost", PLUGIN_STATE_ENABLED) is False


def test_unregister(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_manifest(), plugin_dir="/p")
    assert reg.unregister("my-plugin") is True
    assert reg.get("my-plugin") is None
    assert reg.unregister("my-plugin") is False    # already gone


def test_set_config(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_manifest(), plugin_dir="/p")
    assert reg.set_config("my-plugin", {"key": "value"}) is True
    assert reg.get("my-plugin").config == {"key": "value"}
    assert reg.set_config("ghost", {}) is False


def test_plugin_count_and_enabled_count(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_manifest(name="plugin-a"), plugin_dir="/p", auto_enable=True)
    reg.register(_manifest(name="plugin-b"), plugin_dir="/p")
    assert reg.plugin_count == 2
    assert reg.enabled_count == 1


def test_persistence_reload_from_db(tmp_path):
    db = tmp_path / "plugins.db"
    reg = PluginRegistry(db)
    reg.register(_manifest(version="1.2.3"), plugin_dir="/p", auto_enable=True)

    # A brand-new registry on the same DB must reload the record (this is the
    # exact mechanism the boot path relies on to know which plugins to load).
    reg2 = PluginRegistry(db)
    rec = reg2.get("my-plugin")
    assert rec is not None
    assert rec.state == PLUGIN_STATE_ENABLED
    assert rec.manifest.version == "1.2.3"
    assert rec.plugin_dir == "/p"


def test_version_history_records_events(tmp_path):
    reg = _registry(tmp_path)
    reg.register(_manifest(), plugin_dir="/p")
    reg.set_state("my-plugin", PLUGIN_STATE_ENABLED)
    actions = [e["action"] for e in reg.get_version_history("my-plugin")]
    assert "installed" in actions
    assert "state:enabled" in actions
