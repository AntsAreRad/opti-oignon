#!/usr/bin/env python3
"""A plugin never runs inside the server because its subprocess failed.

A plugin is third-party code, and its own subprocess is where it runs. A
load that fails there leaves the plugin unloaded with the cause named;
nothing re-executes it inside the server process:

  * PL1 -- with the default loader, a subprocess start that fails raises a
    load error naming the cause, loads nothing, and runs none of the
    plugin's code in this process;
  * PL2 -- an execution mode the loader does not know is refused when the
    loader is built, rather than falling through to a fallback;
  * PL3 -- under Bulbe, an in-process loader refuses to load even a plugin
    the allowlist accepts.

Each contract builds a real plugin on disk and stands in for the subprocess
manager; no process is spawned. The modules are resolved when a contract
runs, not when the file is collected: an isolation window in an earlier
suite may have replaced the objects a collection-time import would hold,
and a stand-in set on a stale object never reaches the loader. Local-only.
"""

import importlib
import sys

import pytest

ENTRY = (
    "RAN_IN_THIS_PROCESS = True\n"
    "def hook_pre_inference(context):\n"
    "    return {'marker': 'ran'}\n"
)


def _write_plugin(base, name):
    plugin = base / name
    plugin.mkdir(parents=True)
    (plugin / "manifest.yaml").write_text(
        f"name: {name}\n"
        "version: 1.0.0\n"
        "author: tester\n"
        "description: out-of-process fixture\n"
        "entry_point: entry_point.py\n"
        "hooks:\n"
        "  - pre_inference\n",
        encoding="utf-8",
    )
    (plugin / "entry_point.py").write_text(ENTRY, encoding="utf-8")
    return plugin


def _live(name):
    return importlib.import_module(name)


def _ran_here(name):
    return sorted(key for key in sys.modules if key == f"_opti_plugin_{name}")


@pytest.fixture(autouse=True)
def _scrub_plugin_modules():
    yield
    for key in [k for k in sys.modules if k.startswith("_opti_plugin_pl")]:
        del sys.modules[key]


class _FailingManager:
    """A subprocess manager whose plugin never completes its handshake."""

    def start_plugin(self, **_kwargs):
        raise RuntimeError("stand-in: handshake refused")


def test_pl1_a_failed_subprocess_start_loads_nothing_and_runs_nothing_here(tmp_path, monkeypatch):
    # Daily, pinned: the verdict must not depend on the machine's security mode.
    monkeypatch.setattr(_live("opti_oignon.security_mode"), "is_bulbe", lambda: False)
    loaders = _live("opti_oignon.plugin_loader")
    plugin = _write_plugin(tmp_path, "pl1_plugin")
    loader = loaders.PluginLoader(subprocess_manager=_FailingManager())
    with pytest.raises(loaders.PluginLoadError) as failure:
        loader.load_plugin(plugin)
    assert "handshake refused" in str(failure.value), str(failure.value)
    assert loader.loaded_plugins == {}, loader.loaded_plugins
    assert _ran_here("pl1_plugin") == [], "the plugin's code ran inside the server"


def test_pl2_an_unknown_execution_mode_is_refused_when_the_loader_is_built():
    with pytest.raises(ValueError):
        _live("opti_oignon.plugin_loader").PluginLoader(subprocess_mode="subproces")


def test_pl3_bulbe_refuses_an_in_process_load_the_allowlist_accepts(tmp_path, monkeypatch):
    loaders = _live("opti_oignon.plugin_loader")
    monkeypatch.setattr(_live("opti_oignon.security_mode"), "is_bulbe", lambda: True)
    monkeypatch.setattr(
        _live("opti_oignon.plugin_allowlist").plugin_allowlist_manager,
        "verify_plugin",
        lambda **_kwargs: {"allowed": True},
    )
    plugin = _write_plugin(tmp_path, "pl3_plugin")
    loader = loaders.PluginLoader(subprocess_mode="inprocess")
    with pytest.raises(loaders.PluginLoadError):
        loader.load_plugin(plugin)
    assert loader.loaded_plugins == {}, loader.loaded_plugins
    assert _ran_here("pl3_plugin") == [], "the plugin's code ran inside the server"
