#!/usr/bin/env python3
"""A delivered plugin runs only once the user enables it.

Discovery registered every plugin delivered with the application enabled,
and every enabled plugin starts a worker of its own at boot: fifteen workers,
about 300 MB of memory between them, for plugins that change nothing in a
reply (none of them is shown a reply's content, and the chat routes keep none
of what they return). A delivered plugin is now registered installed, and a
pass after discovery sets to installed each one that was enabled that way and
that the user never touched:

  * DE1 -- a delivered plugin found for the first time is installed, not
    enabled;
  * DE2 -- a delivered plugin enabled by default and never touched is set to
    installed, and its history says so;
  * DE3 -- a state the user ever set holds, through every later pass: an
    enable after the pass is the user's, and it stays;
  * DE4 -- a delivered plugin the user saved a configuration for keeps its
    state;
  * DE5 -- a plugin the user installed is never touched, even under the
    name of a delivered one: only a record of the delivered plugin's own
    folder is;
  * DE6 -- at boot, after the pass, no delivered plugin the user never
    touched starts a worker, and one the user enabled still does.

The registry is loaded through the shared isolation window, each contract
with a database of its own; DE1 and DE6 read the plugins delivered in the
tree. DE1 reads discovery alone: the pass that follows it on the server's
registry would set a builtin registered enabled back to installed, and hide
it. Local-only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_de1_a_delivered_plugin_found_for_the_first_time_is_installed_not_enabled": 2.0,
    "test_de2_a_delivered_plugin_enabled_by_default_and_never_touched_is_set_to_installed": 2.0,
    "test_de3_a_state_the_user_ever_set_holds_through_every_later_pass": 2.0,
    "test_de4_a_delivered_plugin_the_user_configured_keeps_its_state": 2.0,
    "test_de5_a_plugin_the_user_installed_is_never_touched": 2.0,
    "test_de6_at_boot_no_untouched_delivered_plugin_starts_a_worker": 3.0,
}

_DELIVERED = Path(__file__).resolve().parent.parent / "opti_oignon" / "plugins"


def _modules(*, loader=False):
    targets = {"opti_oignon.plugin_manifest": source("plugin_manifest.py")}
    if loader:
        targets["opti_oignon.plugin_loader"] = source("plugin_loader.py")
    loaded, restore = isolate(targets=targets)
    return loaded, restore


def _write(base, name):
    pdir = base / name
    pdir.mkdir(parents=True)
    (pdir / "manifest.yaml").write_text(
        f"name: {name}\nversion: 1.0.0\nauthor: tester\ndescription: enablement fixture\n"
        "entry_point: entry_point.py\nhooks:\n  - post_inference\n",
        encoding="utf-8",
    )
    (pdir / "entry_point.py").write_text("def hook_post_inference(context):\n    return None\n", encoding="utf-8")
    return pdir


def _registered_as_before(registry, base, names):
    """The records discovery used to leave: each plugin registered enabled, with no state of the user's."""
    for name in names:
        pdir = _write(base, name)
        manifest = next(m for m, d in registry.discover_with_paths(base) if d == pdir)
        registry.register(manifest, str(pdir), auto_enable=True)


def _states(registry):
    return {record.manifest.name: record.state for record in registry.list_plugins()}


def _actions(registry, name):
    return [event["action"] for event in registry.get_version_history(name)]


# ---------------------------------------------------------------------------
# DE1 -- found for the first time, installed
# ---------------------------------------------------------------------------
def test_de1_a_delivered_plugin_found_for_the_first_time_is_installed_not_enabled(tmp_path):
    loaded, restore = _modules()
    try:
        mod = loaded["opti_oignon.plugin_manifest"]
        delivered = sorted(m.name for m, _ in mod.PluginRegistry(tmp_path / "scan.db").discover_with_paths(_DELIVERED))
        registry = mod.PluginRegistry(tmp_path / "plugins.db")
        mod._discover_builtins(registry)
        states = _states(registry)
    finally:
        restore()
    assert delivered and sorted(states) == delivered, ("control: discovery reads the delivered plugins", states)
    assert set(states.values()) == {"installed"}, states


# ---------------------------------------------------------------------------
# DE2 -- enabled by default and never touched: set to installed
# ---------------------------------------------------------------------------
def test_de2_a_delivered_plugin_enabled_by_default_and_never_touched_is_set_to_installed(tmp_path):
    loaded, restore = _modules()
    try:
        mod = loaded["opti_oignon.plugin_manifest"]
        registry = mod.PluginRegistry(tmp_path / "plugins.db")
        _registered_as_before(registry, tmp_path / "delivered", ["de-alpha", "de-beta"])
        assert _states(registry) == {"de-alpha": "enabled", "de-beta": "enabled"}, "control: the old records"
        withdrawn = mod._withdraw_default_enablement(registry, builtin_dir=tmp_path / "delivered")
        states, alpha = _states(registry), _actions(registry, "de-alpha")
    finally:
        restore()
    assert sorted(withdrawn) == ["de-alpha", "de-beta"], withdrawn
    assert states == {"de-alpha": "installed", "de-beta": "installed"}, states
    assert "state:installed" in alpha, alpha


# ---------------------------------------------------------------------------
# DE3 -- a state the user set holds, through every later pass
# ---------------------------------------------------------------------------
def test_de3_a_state_the_user_ever_set_holds_through_every_later_pass(tmp_path):
    loaded, restore = _modules()
    try:
        mod = loaded["opti_oignon.plugin_manifest"]
        registry = mod.PluginRegistry(tmp_path / "plugins.db")
        base = tmp_path / "delivered"
        _registered_as_before(registry, base, ["de-kept", "de-off", "de-later"])
        registry.set_state("de-kept", "disabled")
        registry.set_state("de-kept", "enabled")
        registry.set_state("de-off", "disabled")
        first = mod._withdraw_default_enablement(registry, builtin_dir=base)
        registry.set_state("de-later", "enabled")
        second = mod._withdraw_default_enablement(registry, builtin_dir=base)
        states = _states(registry)
    finally:
        restore()
    assert first == ["de-later"], ("control: the untouched one is withdrawn", first)
    assert second == [], second
    assert states == {"de-kept": "enabled", "de-off": "disabled", "de-later": "enabled"}, states


# ---------------------------------------------------------------------------
# DE4 -- configured by the user: kept
# ---------------------------------------------------------------------------
def test_de4_a_delivered_plugin_the_user_configured_keeps_its_state(tmp_path):
    loaded, restore = _modules()
    try:
        mod = loaded["opti_oignon.plugin_manifest"]
        registry = mod.PluginRegistry(tmp_path / "plugins.db")
        base = tmp_path / "delivered"
        _registered_as_before(registry, base, ["de-set", "de-bare"])
        assert registry.set_config("de-set", {"active_mode": "academic"})
        withdrawn = mod._withdraw_default_enablement(registry, builtin_dir=base)
        states = _states(registry)
    finally:
        restore()
    assert withdrawn == ["de-bare"], withdrawn
    assert states == {"de-set": "enabled", "de-bare": "installed"}, states


# ---------------------------------------------------------------------------
# DE5 -- installed by the user: never touched
# ---------------------------------------------------------------------------
def test_de5_a_plugin_the_user_installed_is_never_touched(tmp_path):
    loaded, restore = _modules()
    try:
        mod = loaded["opti_oignon.plugin_manifest"]
        registry = mod.PluginRegistry(tmp_path / "plugins.db")
        _registered_as_before(registry, tmp_path / "delivered", ["de-own"])
        _registered_as_before(registry, tmp_path / "installed", ["de-mine", "de-named"])
        _write(tmp_path / "delivered", "de-named")
        withdrawn = mod._withdraw_default_enablement(registry, builtin_dir=tmp_path / "delivered")
        states = _states(registry)
    finally:
        restore()
    assert withdrawn == ["de-own"], withdrawn
    assert states == {"de-own": "installed", "de-mine": "enabled", "de-named": "enabled"}, (
        "a plugin the user installed, even under a delivered plugin's name, keeps its state", states)


# ---------------------------------------------------------------------------
# DE6 -- at boot
# ---------------------------------------------------------------------------
class _Manager:
    """A worker manager stand-in: it records the plugins it is asked to start, and starts nothing."""

    def __init__(self):
        self.started = []

    def start_plugin(self, *, plugin_name, **_kwargs):
        self.started.append(plugin_name)

    def stop_plugin(self, plugin_name, timeout=5.0):
        return True

    def stop_watchdog(self):
        return None


def test_de6_at_boot_no_untouched_delivered_plugin_starts_a_worker(tmp_path):
    loaded, restore = _modules(loader=True)
    try:
        mod, loaders = loaded["opti_oignon.plugin_manifest"], loaded["opti_oignon.plugin_loader"]
        before = mod.PluginRegistry(tmp_path / "plugins.db")
        pairs = before.discover_with_paths(_DELIVERED)
        for manifest, pdir in pairs:
            before.register(manifest, str(pdir), auto_enable=True)
        chosen = pairs[0][0].name
        before.set_state(chosen, "enabled")

        registry = mod.PluginRegistry(tmp_path / "plugins.db", discover_builtins=True)
        manager = _Manager()
        loader = loaders.PluginLoader(registry=registry, subprocess_manager=manager)
        booted = [plugin.name for plugin in loader.load_all_enabled()]
        loader.shutdown_all()
    finally:
        restore()
    assert len(pairs) > 1, "control: the tree delivers plugins"
    assert booted == [chosen] and manager.started == [chosen], (booted, manager.started)
