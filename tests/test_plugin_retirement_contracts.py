#!/usr/bin/env python3
"""A plugin that is no longer delivered leaves nothing behind.

The fact checker sent the claims of a reply to a web search engine, while the
package checks claims against the user's own sources and takes no search
engine's excerpt as one (``factcheck``). It is no longer delivered, and an
installation that had it keeps no trace that the boot would try to load:

  * RW1 -- the fact checker is no longer delivered: no folder, no manifest,
    no line of the delivered list names it, and no module of the package
    names it but the list of retired plugins;
  * RW2 -- at the first start, the registry the server reads forgets its
    record, whatever the record's state, and the boot asks no worker for it;
  * RW3 -- a plugin the user installed under that name, in a folder of their
    own that exists, keeps its record;
  * RW4 -- the field plugins still delivered load in their worker with their
    fallbacks engaged (it gives way for c3 of
    tests/test_plugin_worker_isolation_contracts.py, which loaded the fact
    checker with them, in a window the isolation ledger still owes for).

The registry and the worker are loaded through the shared isolation window,
each contract with a database or a folder of its own. Local-only.
"""

import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_rw1_the_fact_checker_is_no_longer_delivered_and_only_the_retired_list_names_it": 2.0,
    "test_rw2_its_record_is_forgotten_at_the_first_start_whatever_its_state": 3.0,
    "test_rw3_a_plugin_the_user_installed_under_that_name_keeps_its_record": 2.0,
    "test_rw4_the_field_plugins_still_delivered_load_in_their_worker_with_their_fallbacks": 5.0,
}

_PACKAGE = Path(__file__).resolve().parent.parent / "opti_oignon"
_NAME = "fact-checker"
_MANIFEST = (f"name: {_NAME}\nversion: 1.0.0\nauthor: tester\ndescription: retired fixture\n"
             "entry_point: entry_point.py\nhooks:\n  - post_inference\n")


def _naming(root, name):
    """The package files under ``root`` whose text names ``name``; ``__pycache__`` and ``data`` are not walked, the
    package's data directory holding the maintainer's stores, never a module."""
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", "data"))
        for file in sorted(files):
            if file.endswith((".py", ".yaml", ".yml", ".json")):
                path = Path(folder, file)
                if name in path.read_text(encoding="utf-8", errors="replace"):
                    found.append(path.relative_to(root).as_posix())
    return found


def _manifests():
    loaded, restore = isolate(targets={"opti_oignon.plugin_manifest": source("plugin_manifest.py")})
    return loaded["opti_oignon.plugin_manifest"], restore


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


# ---------------------------------------------------------------------------
# RW1 -- no longer delivered
# ---------------------------------------------------------------------------
def test_rw1_the_fact_checker_is_no_longer_delivered_and_only_the_retired_list_names_it(tmp_path):
    planted = tmp_path / "pkg"
    (planted / "data").mkdir(parents=True)
    (planted / "router.py").write_text(f"PLUGIN = '{_NAME}'\n", encoding="utf-8")
    (planted / "data" / "store.json").write_text(f'"{_NAME}"', encoding="utf-8")
    assert _naming(planted, _NAME) == ["router.py"], "witness: a module naming it is found, a data folder is not walked"

    mod, restore = _manifests()
    try:
        scan = mod.PluginRegistry(tmp_path / "scan.db")
        delivered = [m.name for m, _ in scan.discover_with_paths(_PACKAGE / "plugins")]
        retired = tuple(mod._RETIRED_BUILTINS)
    finally:
        restore()
    assert delivered and _NAME not in delivered, delivered
    assert not (_PACKAGE / "plugins" / _NAME).exists()
    assert _NAME in retired, retired
    assert _naming(_PACKAGE, _NAME) == ["plugin_manifest.py"], _naming(_PACKAGE, _NAME)


# ---------------------------------------------------------------------------
# RW2 -- forgotten at the first start
# ---------------------------------------------------------------------------
def test_rw2_its_record_is_forgotten_at_the_first_start_whatever_its_state(tmp_path):
    import yaml

    seen = {}
    for kind in ("enabled by default", "enabled by the user", "configured", "disabled"):
        db = tmp_path / kind.replace(" ", "-") / "plugins.db"
        db.parent.mkdir(parents=True)
        gone = tmp_path / "release" / _NAME
        mod, restore = _manifests()
        loaders = None
        try:
            before = mod.PluginRegistry(db)
            manifest = mod.PluginManifest.from_dict(yaml.safe_load(_MANIFEST))
            before.register(manifest, str(gone), auto_enable=True)
            if kind == "enabled by the user":
                before.set_state(_NAME, "disabled")
                before.set_state(_NAME, "enabled")
            elif kind == "configured":
                before.set_config(_NAME, {"aggressiveness": "low"})
            elif kind == "disabled":
                before.set_state(_NAME, "disabled")
            assert before.get(_NAME) is not None, "control: the old record"

            registry = mod.PluginRegistry(db, discover_builtins=True)
            record = registry.get(_NAME)
            actions = [event["action"] for event in registry.get_version_history(_NAME)]
        finally:
            restore()
        loaded, restore = isolate(targets={"opti_oignon.plugin_manifest": source("plugin_manifest.py"),
                                           "opti_oignon.plugin_loader": source("plugin_loader.py")})
        try:
            mod, loaders = loaded["opti_oignon.plugin_manifest"], loaded["opti_oignon.plugin_loader"]
            manager = _Manager()
            loader = loaders.PluginLoader(registry=mod.PluginRegistry(db, discover_builtins=True),
                                          subprocess_manager=manager)
            loader.load_all_enabled()
            loader.shutdown_all()
        finally:
            restore()
        seen[kind] = (record, "uninstalled" in actions, _NAME in manager.started)
    assert seen == {kind: (None, True, False) for kind in seen}, seen


# ---------------------------------------------------------------------------
# RW3 -- installed by the user under that name: kept
# ---------------------------------------------------------------------------
def test_rw3_a_plugin_the_user_installed_under_that_name_keeps_its_record(tmp_path):
    import yaml

    own = tmp_path / "installed" / _NAME
    own.mkdir(parents=True)
    (own / "manifest.yaml").write_text(_MANIFEST, encoding="utf-8")
    (own / "entry_point.py").write_text("def hook_post_inference(context):\n    return None\n", encoding="utf-8")
    mod, restore = _manifests()
    try:
        before = mod.PluginRegistry(tmp_path / "plugins.db")
        before.register(mod.PluginManifest.from_dict(yaml.safe_load(_MANIFEST)), str(own), auto_enable=True)
        forgotten = mod._forget_retired_builtins(before)
        record = before.get(_NAME)
    finally:
        restore()
    assert forgotten == [], forgotten
    assert record is not None and Path(record.plugin_dir) == own, record


# ---------------------------------------------------------------------------
# RW4 -- the field plugins still delivered, in their worker
# ---------------------------------------------------------------------------
def test_rw4_the_field_plugins_still_delivered_load_in_their_worker_with_their_fallbacks(tmp_path):
    names = ("github-connector", "scratchpad", "task-extractor")
    loaded, restore = isolate(targets={"opti_oignon.plugin_worker": source("plugin_worker.py")})
    failures = []
    try:
        worker = loaded["opti_oignon.plugin_worker"]
        for name in names:
            work_dir = tmp_path / name
            shutil.copytree(_PACKAGE / "plugins" / name, work_dir, ignore=shutil.ignore_patterns("__pycache__"))
            started = time.perf_counter()
            try:
                module = worker.load_plugin_module(name, str(work_dir), "entry_point.py")
            except Exception as exc:
                failures.append(f"{name}: load failed: {type(exc).__name__}: {exc}")
                continue
            if time.perf_counter() - started > 2.0:
                failures.append(f"{name}: load took more than 2 s")
            fallback = getattr(module, "_safe_connect", None)
            if getattr(fallback, "__name__", "") != "<lambda>":
                failures.append(f"{name}: sqlite fallback not engaged ({fallback!r})")
    finally:
        restore()
        for name in names:
            sys.modules.pop(f"_oo_worker_plugin_{name}", None)
    assert not failures, failures
