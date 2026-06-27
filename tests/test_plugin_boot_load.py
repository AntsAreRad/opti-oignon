#!/usr/bin/env python3
"""End-to-end test of the plugin boot path (PluginLoader.load_all_enabled).

This is the proof for the boot-load fix: a plugin marked "enabled" in the
registry must be (re)loaded and have its hooks registered at startup, so its
effects apply during inference -- not only right after enable_plugin().

Each test builds a REAL plugin on disk (manifest.yaml + entry_point.py),
registers it, and drives a fresh PluginLoader in in-process mode (subprocess
isolation needs bubblewrap, which is host-only). The plugin's hook is then
fired through the shared HookManager to confirm the effect actually applies.

An autouse fixture scrubs the two process-global side effects (the HookManager
singleton and the ``_opti_plugin_*`` entries in sys.modules) so the cases stay
isolated.
"""

import sys

import pytest

from opti_oignon.plugin_hooks import hook_manager
from opti_oignon.plugin_loader import PluginLoader
from opti_oignon.plugin_manifest import PluginManifest, PluginRegistry

# A hook that records that it ran and contributes to the chain data.
GOOD_ENTRY = (
    'def hook_pre_inference(context):\n'
    '    return {"fixture_marker": "ran"}\n'
)
# A plugin that fails at load time (top-level raise) -- used to prove the boot
# loop isolates one bad plugin from the rest.
BROKEN_ENTRY = 'raise RuntimeError("intentional load failure")\n'


def _write_plugin(base, name, entry_src):
    """Create base/<name>/{manifest.yaml, entry_point.py} and return the dir."""
    pdir = base / name
    pdir.mkdir(parents=True)
    (pdir / "manifest.yaml").write_text(
        f"name: {name}\n"
        "version: 1.0.0\n"
        "author: tester\n"
        "description: boot-load fixture\n"
        "entry_point: entry_point.py\n"
        "hooks:\n"
        "  - pre_inference\n"
    )
    (pdir / "entry_point.py").write_text(entry_src)
    return pdir


def _manifest(name):
    return PluginManifest.from_dict({
        "name": name,
        "version": "1.0.0",
        "author": "tester",
        "description": "boot-load fixture",
        "entry_point": "entry_point.py",
        "hooks": ["pre_inference"],
    })


def _loader(reg):
    # in-process mode: subprocess isolation requires bubblewrap (host-only).
    return PluginLoader(registry=reg, subprocess_mode="inprocess")


@pytest.fixture(autouse=True)
def _isolate_plugin_globals():
    """Undo the process-global side effects each test produces."""
    before = {k for k in sys.modules if k.startswith("_opti_plugin_")}
    yield
    after = {k for k in sys.modules if k.startswith("_opti_plugin_")}
    for key in after - before:
        name = key[len("_opti_plugin_"):]
        hook_manager.unregister_plugin(name)
        sys.modules.pop(key, None)
    hook_manager.reset_stats()


# ===========================================================================
# The core proof
# ===========================================================================

def test_load_all_enabled_registers_and_fires_hook(tmp_path):
    base = tmp_path / "plugins"
    pdir = _write_plugin(base, "boot-good", GOOD_ENTRY)
    reg = PluginRegistry(tmp_path / "plugins.db")
    reg.register(_manifest("boot-good"), plugin_dir=str(pdir), auto_enable=True)

    loader = _loader(reg)
    loaded = loader.load_all_enabled()

    assert [p.name for p in loaded] == ["boot-good"]
    assert "boot-good" in loader.loaded_plugins
    # the hook reached the shared HookManager ...
    assert hook_manager.has_hooks("pre_inference")
    # ... and actually fires, contributing to the chain data.
    report = hook_manager.execute("pre_inference", data={})
    assert report.final_data.get("fixture_marker") == "ran"


def test_enable_then_fresh_loader_revives_plugin(tmp_path):
    # Reproduces the original bug end to end: a plugin works right after
    # enable, but a restart (modeled as a brand-new loader with nothing
    # loaded) must bring it back via load_all_enabled.
    base = tmp_path / "plugins"
    pdir = _write_plugin(base, "boot-revive", GOOD_ENTRY)
    reg = PluginRegistry(tmp_path / "plugins.db")
    reg.register(_manifest("boot-revive"), plugin_dir=str(pdir))  # installed

    loader1 = _loader(reg)
    assert loader1.enable_plugin("boot-revive") is not None       # loads now
    assert "boot-revive" in loader1.loaded_plugins

    loader2 = _loader(reg)                                         # "restart"
    assert "boot-revive" not in loader2.loaded_plugins            # nothing loaded yet
    revived = loader2.load_all_enabled()                          # the boot path
    assert [p.name for p in revived] == ["boot-revive"]
    assert hook_manager.has_hooks("pre_inference")


# ===========================================================================
# State filtering + isolation + shutdown
# ===========================================================================

def test_load_all_enabled_skips_non_enabled(tmp_path):
    base = tmp_path / "plugins"
    p_on = _write_plugin(base, "boot-on", GOOD_ENTRY)
    p_off = _write_plugin(base, "boot-off", GOOD_ENTRY)
    reg = PluginRegistry(tmp_path / "plugins.db")
    reg.register(_manifest("boot-on"), plugin_dir=str(p_on), auto_enable=True)
    reg.register(_manifest("boot-off"), plugin_dir=str(p_off))   # installed only

    loaded = _loader(reg).load_all_enabled()
    assert [p.name for p in loaded] == ["boot-on"]


def test_load_all_enabled_empty_when_none_enabled(tmp_path):
    base = tmp_path / "plugins"
    pdir = _write_plugin(base, "boot-installed", GOOD_ENTRY)
    reg = PluginRegistry(tmp_path / "plugins.db")
    reg.register(_manifest("boot-installed"), plugin_dir=str(pdir))  # not enabled

    assert _loader(reg).load_all_enabled() == []


def test_load_all_enabled_isolates_failing_plugin(tmp_path):
    # One enabled plugin fails to load; the boot loop must isolate it and
    # still load the healthy one.
    base = tmp_path / "plugins"
    p_bad = _write_plugin(base, "boot-bad", BROKEN_ENTRY)
    p_good = _write_plugin(base, "boot-ok", GOOD_ENTRY)
    reg = PluginRegistry(tmp_path / "plugins.db")
    reg.register(_manifest("boot-bad"), plugin_dir=str(p_bad), auto_enable=True)
    reg.register(_manifest("boot-ok"), plugin_dir=str(p_good), auto_enable=True)

    loaded = {p.name for p in _loader(reg).load_all_enabled()}
    assert "boot-ok" in loaded          # healthy plugin still loaded
    assert "boot-bad" not in loaded     # broken plugin isolated, no crash


def test_shutdown_all_unloads_plugins(tmp_path):
    base = tmp_path / "plugins"
    pdir = _write_plugin(base, "boot-shutdown", GOOD_ENTRY)
    reg = PluginRegistry(tmp_path / "plugins.db")
    reg.register(_manifest("boot-shutdown"), plugin_dir=str(pdir), auto_enable=True)

    loader = _loader(reg)
    loader.load_all_enabled()
    assert "boot-shutdown" in loader.loaded_plugins

    loader.shutdown_all()
    assert loader.loaded_plugins == {}
