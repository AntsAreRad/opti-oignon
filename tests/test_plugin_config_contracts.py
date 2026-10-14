#!/usr/bin/env python3
"""A plugin's hooks are given its configuration as it is stored for it.

The configuration a user saves for a plugin (``PUT /api/plugins/{name}/config``,
the editor of the Plugins panel) was kept in the registry and handed to
nothing: the chat routes call the hooks without one, and every hook was shown
an empty configuration, so that each plugin ran on its own defaults whatever
was saved -- the tone shifter in its mode "none", which changes nothing. The
hook manager now shows each hook its plugin's configuration, read from the
registry at the call:

  * CG1 -- a delivered plugin, in its real worker, is given the configuration
    saved for it, and it acts on it;
  * CG2 -- the configuration is read at each call: what is saved after the
    plugin was loaded holds from the next call on, without a restart;
  * CG3 -- a plugin is never shown another plugin's configuration;
  * CG4 -- what a hook does to the configuration it is shown reaches neither
    the store nor the next call;
  * CG5 -- a hook whose configuration cannot be read is not run, and is
    reported as failed: a setting the user saved is never set aside for the
    plugin's defaults (a plugin told not to reach the network, for one), and
    the other hooks still run.

CG1 to CG3 start real workers, as plain processes under a posture each
contract sets; CG4 and CG5 drive the hook manager directly. The modules are
resolved when a contract runs. Local-only.
"""

import importlib
import shutil
from pathlib import Path

import pytest

BUDGET_S = {
    "test_cg1_a_plugin_in_its_worker_is_given_the_configuration_saved_for_it": 5.0,
    "test_cg2_the_configuration_is_read_at_each_call_and_needs_no_restart": 5.0,
    "test_cg3_a_plugin_is_never_shown_another_plugin_s_configuration": 5.0,
    "test_cg4_what_a_hook_does_to_its_configuration_stays_with_that_call": 2.0,
    "test_cg5_a_hook_whose_configuration_cannot_be_read_is_not_run_and_the_others_are": 2.0,
}

_PLUGINS = Path(__file__).resolve().parent.parent / "opti_oignon" / "plugins"

SHOWS_ITS_CONFIGURATION = (
    'def hook_pre_inference(context):\n'
    '    return {context.plugin_name + " saw": dict(context.config)}\n'
)


def _live(name):
    return importlib.import_module(name)


def _manifest_of(pdir):
    import yaml

    manifests = _live("opti_oignon.plugin_manifest")
    data = yaml.safe_load((pdir / "manifest.yaml").read_text(encoding="utf-8"))
    return manifests.PluginManifest.from_dict(data)


def _write_fixture(base, name):
    pdir = base / name
    pdir.mkdir(parents=True)
    (pdir / "manifest.yaml").write_text(
        f"name: {name}\n"
        "version: 1.0.0\n"
        "author: tester\n"
        "description: configuration fixture\n"
        "entry_point: entry_point.py\n"
        "hooks:\n"
        "  - pre_inference\n",
        encoding="utf-8",
    )
    (pdir / "entry_point.py").write_text(SHOWS_ITS_CONFIGURATION, encoding="utf-8")
    return pdir


@pytest.fixture
def plugins(tmp_path):
    """(registry, loader): a registry of this contract's own, and a loader over real workers;
    every worker stopped and every hook unregistered afterwards."""
    isolation = _live("opti_oignon.plugin_isolation")
    workers = _live("opti_oignon.plugin_subprocess")
    loaders = _live("opti_oignon.plugin_loader")
    hooks = _live("opti_oignon.plugin_hooks").hook_manager
    registry = _live("opti_oignon.plugin_manifest").PluginRegistry(tmp_path / "plugins.db")
    manager = workers.PluginSubprocessManager(
        log_dir=tmp_path / "logs",
        data_root=tmp_path / "data",
        posture_provider=lambda: isolation.PluginPosture(mode=isolation.MODE_DIRECT, strict=False),
    )
    loader = loaders.PluginLoader(registry=registry, subprocess_manager=manager)
    yield registry, loader
    for name in list(loader.loaded_plugins):
        hooks.unregister_plugin(name)
    loader.shutdown_all()
    manager.stop_all()
    hooks.reset_stats()


def _tone_shifter(tmp_path, registry):
    pdir = tmp_path / "plugins" / "tone-shifter"
    shutil.copytree(_PLUGINS / "tone-shifter", pdir, ignore=shutil.ignore_patterns("__pycache__"))
    registry.register(_manifest_of(pdir), plugin_dir=str(pdir), auto_enable=True)
    return "tone-shifter"


def _reply(text):
    hooks = _live("opti_oignon.plugin_hooks").hook_manager
    report = hooks.execute("post_inference", data={"response": text})
    assert report.failed == 0, [result.error for result in report.results]
    return report.final_data["response"]


# ---------------------------------------------------------------------------
# CG1 -- a plugin in its worker is given the configuration saved for it
# ---------------------------------------------------------------------------
def test_cg1_a_plugin_in_its_worker_is_given_the_configuration_saved_for_it(tmp_path, plugins):
    registry, loader = plugins
    name = _tone_shifter(tmp_path, registry)
    assert registry.set_config(name, {"active_mode": "academic"})
    assert [plugin.name for plugin in loader.load_all_enabled()] == [name]
    assert loader.loaded_plugins[name].is_subprocess

    assert _reply("It is plain.") == "It appears to be plain.", "the mode saved is the one it runs"


# ---------------------------------------------------------------------------
# CG2 -- the configuration is read at each call
# ---------------------------------------------------------------------------
def test_cg2_the_configuration_is_read_at_each_call_and_needs_no_restart(tmp_path, plugins):
    registry, loader = plugins
    name = _tone_shifter(tmp_path, registry)
    assert [plugin.name for plugin in loader.load_all_enabled()] == [name]
    assert _reply("It is plain.") == "It is plain.", "control: nothing saved, its own mode, none"

    assert registry.set_config(name, {"active_mode": "academic"})
    assert _reply("It is plain.") == "It appears to be plain."
    assert registry.set_config(name, {"active_mode": "none"})
    assert _reply("It is plain.") == "It is plain."


# ---------------------------------------------------------------------------
# CG3 -- a plugin is never shown another plugin's configuration
# ---------------------------------------------------------------------------
def test_cg3_a_plugin_is_never_shown_another_plugin_s_configuration(tmp_path, plugins):
    registry, loader = plugins
    saved = {
        "cg-alpha": {"token": "alpha's own", "nested": {"depth": 1}},
        "cg-beta": {"level": 2},
        "cg-gamma": {},
    }
    for name, config in saved.items():
        pdir = _write_fixture(tmp_path / "plugins", name)
        registry.register(_manifest_of(pdir), plugin_dir=str(pdir), auto_enable=True)
        assert registry.set_config(name, config)
    assert sorted(plugin.name for plugin in loader.load_all_enabled()) == sorted(saved)

    report = _live("opti_oignon.plugin_hooks").hook_manager.execute("pre_inference", data={})
    seen = {
        result.plugin_name: (result.modified_data or {}).get(result.plugin_name + " saw")
        for result in report.results
    }
    assert seen == saved, seen


# ---------------------------------------------------------------------------
# CG4 -- what a hook does to its configuration stays with that call
# ---------------------------------------------------------------------------
def test_cg4_what_a_hook_does_to_its_configuration_stays_with_that_call():
    manager = _live("opti_oignon.plugin_hooks").HookManager()
    stored = {"rules": ["kept"], "mode": "kept"}
    shown = []

    def meddles(context):
        shown.append({"rules": list(context.config["rules"]), "mode": context.config["mode"]})
        context.config["rules"].append("added by the hook")
        context.config["mode"] = "changed by the hook"

    assert manager.register("post_inference", "cg-meddler", meddles, config_source=lambda: stored)
    for _ in range(2):
        assert manager.execute("post_inference", data={}).failed == 0
    assert stored == {"rules": ["kept"], "mode": "kept"}, stored
    assert shown == [{"rules": ["kept"], "mode": "kept"}] * 2, shown


# ---------------------------------------------------------------------------
# CG5 -- a hook whose configuration cannot be read is not run
# ---------------------------------------------------------------------------
def test_cg5_a_hook_whose_configuration_cannot_be_read_is_not_run_and_the_others_are():
    manager = _live("opti_oignon.plugin_hooks").HookManager()
    ran = []

    def unreadable():
        raise OSError("the store cannot be read")

    assert manager.register("post_inference", "cg-unread", lambda context: ran.append("cg-unread"),
                            config_source=unreadable)
    assert manager.register("post_inference", "cg-read", lambda context: ran.append("cg-read"),
                            config_source=lambda: {"auto_link": False})
    report = manager.execute("post_inference", data={})

    assert ran == ["cg-read"], ran
    assert (report.successful, report.failed) == (1, 1)
    failed = [result for result in report.results if not result.success]
    assert [result.plugin_name for result in failed] == ["cg-unread"]
    assert "the store cannot be read" in failed[0].error, failed[0].error
