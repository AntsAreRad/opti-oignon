#!/usr/bin/env python3
"""The boot path loads every enabled plugin into a worker of its own.

PluginLoader.load_all_enabled is what brings the plugins back after a
restart. These contracts re-assert, unchanged in what they pin, the six of
tests/test_plugin_boot_load.py, which drove an in-process loader that no
longer exists; here each plugin starts in a real worker process:

  * BT1 -- an enabled plugin is loaded, its hook registered, and the hook
    fires with its effect on the chain;
  * BT2 -- a plugin enabled through one loader comes back through a fresh
    one, the restart's boot path;
  * BT3 -- a plugin installed but not enabled is not loaded;
  * BT4 -- with no plugin enabled, nothing is loaded;
  * BT5 -- a plugin that fails at load is left out, and the healthy one
    still loads;
  * BT6 -- shutting the loader down unloads its plugins.

The workers run as plain processes under a posture each contract sets: the
walls are the business of the plugin isolation suite, the boot path the
business of this one. The modules are resolved when a contract runs.
Local-only.
"""

import importlib

import pytest

GOOD_ENTRY = (
    'def hook_pre_inference(context):\n'
    '    return {"fixture_marker": "ran"}\n'
)
# Raises at import: the worker cannot initialize it.
BROKEN_ENTRY = 'raise RuntimeError("intentional load failure")\n'


def _live(name):
    return importlib.import_module(name)


def _write_plugin(base, name, entry_src):
    pdir = base / name
    pdir.mkdir(parents=True)
    (pdir / "manifest.yaml").write_text(
        f"name: {name}\n"
        "version: 1.0.0\n"
        "author: tester\n"
        "description: boot-load fixture\n"
        "entry_point: entry_point.py\n"
        "hooks:\n"
        "  - pre_inference\n",
        encoding="utf-8",
    )
    (pdir / "entry_point.py").write_text(entry_src, encoding="utf-8")
    return pdir


def _manifest(name):
    return _live("opti_oignon.plugin_manifest").PluginManifest.from_dict({
        "name": name,
        "version": "1.0.0",
        "author": "tester",
        "description": "boot-load fixture",
        "entry_point": "entry_point.py",
        "hooks": ["pre_inference"],
    })


def _registry(tmp_path):
    return _live("opti_oignon.plugin_manifest").PluginRegistry(tmp_path / "plugins.db")


@pytest.fixture
def boot(tmp_path):
    """Build loaders over real workers; stop every worker and hook afterwards."""
    isolation = _live("opti_oignon.plugin_isolation")
    workers = _live("opti_oignon.plugin_subprocess")
    loaders = _live("opti_oignon.plugin_loader")
    hooks = _live("opti_oignon.plugin_hooks").hook_manager
    built = []

    def make(registry):
        manager = workers.PluginSubprocessManager(
            log_dir=tmp_path / "logs",
            data_root=tmp_path / "data",
            posture_provider=lambda: isolation.PluginPosture(
                mode=isolation.MODE_DIRECT, strict=False),
        )
        loader = loaders.PluginLoader(registry=registry, subprocess_manager=manager)
        built.append((loader, manager))
        return loader

    yield make
    for loader, manager in built:
        for name in list(loader.loaded_plugins):
            hooks.unregister_plugin(name)
        loader.shutdown_all()
        manager.stop_all()
    hooks.reset_stats()


def test_bt1_an_enabled_plugin_is_loaded_and_its_hook_fires(tmp_path, boot):
    hooks = _live("opti_oignon.plugin_hooks").hook_manager
    pdir = _write_plugin(tmp_path / "plugins", "boot-good", GOOD_ENTRY)
    reg = _registry(tmp_path)
    reg.register(_manifest("boot-good"), plugin_dir=str(pdir), auto_enable=True)

    loader = boot(reg)
    loaded = loader.load_all_enabled()

    assert [p.name for p in loaded] == ["boot-good"]
    assert "boot-good" in loader.loaded_plugins
    assert hooks.has_hooks("pre_inference")
    report = hooks.execute("pre_inference", data={})
    assert report.final_data.get("fixture_marker") == "ran"


def test_bt2_a_plugin_enabled_through_one_loader_comes_back_through_a_fresh_one(tmp_path, boot):
    hooks = _live("opti_oignon.plugin_hooks").hook_manager
    pdir = _write_plugin(tmp_path / "plugins", "boot-revive", GOOD_ENTRY)
    reg = _registry(tmp_path)
    reg.register(_manifest("boot-revive"), plugin_dir=str(pdir))  # installed

    loader1 = boot(reg)
    assert loader1.enable_plugin("boot-revive") is not None       # loads now
    assert "boot-revive" in loader1.loaded_plugins

    loader2 = boot(reg)                                            # "restart"
    assert "boot-revive" not in loader2.loaded_plugins             # nothing loaded yet
    revived = loader2.load_all_enabled()                           # the boot path
    assert [p.name for p in revived] == ["boot-revive"]
    assert hooks.has_hooks("pre_inference")


def test_bt3_a_plugin_installed_but_not_enabled_is_not_loaded(tmp_path, boot):
    base = tmp_path / "plugins"
    p_on = _write_plugin(base, "boot-on", GOOD_ENTRY)
    p_off = _write_plugin(base, "boot-off", GOOD_ENTRY)
    reg = _registry(tmp_path)
    reg.register(_manifest("boot-on"), plugin_dir=str(p_on), auto_enable=True)
    reg.register(_manifest("boot-off"), plugin_dir=str(p_off))    # installed only

    loaded = boot(reg).load_all_enabled()
    assert [p.name for p in loaded] == ["boot-on"]


def test_bt4_with_no_plugin_enabled_nothing_is_loaded(tmp_path, boot):
    pdir = _write_plugin(tmp_path / "plugins", "boot-installed", GOOD_ENTRY)
    reg = _registry(tmp_path)
    reg.register(_manifest("boot-installed"), plugin_dir=str(pdir))  # not enabled

    assert boot(reg).load_all_enabled() == []


def test_bt5_a_plugin_that_fails_at_load_is_left_out_and_the_healthy_one_loads(tmp_path, boot):
    base = tmp_path / "plugins"
    p_bad = _write_plugin(base, "boot-bad", BROKEN_ENTRY)
    p_good = _write_plugin(base, "boot-ok", GOOD_ENTRY)
    reg = _registry(tmp_path)
    reg.register(_manifest("boot-bad"), plugin_dir=str(p_bad), auto_enable=True)
    reg.register(_manifest("boot-ok"), plugin_dir=str(p_good), auto_enable=True)

    loaded = {p.name for p in boot(reg).load_all_enabled()}
    assert "boot-ok" in loaded          # healthy plugin still loaded
    assert "boot-bad" not in loaded     # broken plugin isolated, no crash


def test_bt6_shutting_the_loader_down_unloads_its_plugins(tmp_path, boot):
    pdir = _write_plugin(tmp_path / "plugins", "boot-shutdown", GOOD_ENTRY)
    reg = _registry(tmp_path)
    reg.register(_manifest("boot-shutdown"), plugin_dir=str(pdir), auto_enable=True)

    loader = boot(reg)
    loader.load_all_enabled()
    assert "boot-shutdown" in loader.loaded_plugins

    loader.shutdown_all()
    assert loader.loaded_plugins == {}
