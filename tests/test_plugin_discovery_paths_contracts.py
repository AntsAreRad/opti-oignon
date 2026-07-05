#!/usr/bin/env python3
"""Contracts for path-carrying plugin discovery and record healing.

Builtin discovery scanned ``plugins/*/manifest.yaml`` and then threw the
source directory away: registration re-derived it from the MANIFEST
NAME (``builtin_dir / manifest.name``). Any plugin whose folder name
differs from its manifest name -- two delivered builtins do -- was
registered under a phantom path, the registry persisted it, and enable
failed with "Plugin directory not found". These contracts pin the fix:

  * Contract 1 -- discovery carries paths: ``discover_with_paths``
    returns (manifest, real directory) pairs, and the historical
    ``discover`` keeps returning exactly the same manifests.
  * Contract 2 -- builtin registration records the REAL directory: a
    folder whose name differs from its manifest name registers under
    the folder's actual path (which exists), and a folder whose name
    matches keeps the same, unchanged convention.
  * Contract 3 -- healing: an already-persisted record whose stored
    directory no longer exists is refreshed to the real discovered
    directory on the next discovery pass, and its enabled state is
    preserved (register without auto_enable never flips state).
  * Contract 4 -- the version-refresh branch also records the real
    directory, never the name-derived phantom.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. The manifest module is loaded in isolation under a
stub package (the config-backed singleton stays inert); registries use
a temporary database, plugin trees live under a temporary directory.
"""

import importlib.util
import sys
import tempfile
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load_manifest_module():
    keys = ("opti_oignon", "opti_oignon.plugin_manifest")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.plugin_manifest", _OO / "plugin_manifest.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.plugin_manifest"] = mod
    spec.loader.exec_module(mod)
    pkg.plugin_manifest = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


_MANIFEST_TEMPLATE = """\
name: {name}
version: "{version}"
author: Contract Harness
description: Discovery contract fixture.
entry_point: entry_point.py
hooks:
  - post_inference
dependencies: []
permissions: []
"""


def _write_plugin(base: Path, folder: str, name: str, version: str = "1.0.0"):
    plugin_dir = base / folder
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "manifest.yaml").write_text(
        _MANIFEST_TEMPLATE.format(name=name, version=version),
    )
    (plugin_dir / "entry_point.py").write_text(
        'HOOKS = {"post_inference": lambda ctx: None}\n',
    )
    return plugin_dir


# ---------------------------------------------------------------------------
# Contract 1 -- discovery carries the real source directories
# ---------------------------------------------------------------------------
def test_c1_discover_with_paths_returns_real_directories():
    mod, restore = _load_manifest_module()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            weird_dir = _write_plugin(base, "weird_folder", "weird-name")
            same_dir = _write_plugin(base, "samesame", "samesame")

            registry = mod.PluginRegistry(base / "plugins.db")
            pairs = registry.discover_with_paths(base)
            by_name = {m.name: p for m, p in pairs}
            assert set(by_name) == {"weird-name", "samesame"}, by_name
            assert by_name["weird-name"].resolve() == weird_dir.resolve(), (
                f"discovery must carry the REAL folder: {by_name['weird-name']}"
            )
            assert by_name["samesame"].resolve() == same_dir.resolve()

            # The historical surface is unchanged.
            names = sorted(m.name for m in registry.discover(base))
            assert names == ["samesame", "weird-name"], names
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- builtin registration records the real directory
# ---------------------------------------------------------------------------
def test_c2_builtins_register_under_the_real_directory():
    mod, restore = _load_manifest_module()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            weird_dir = _write_plugin(base, "weird_folder", "weird-name")
            same_dir = _write_plugin(base, "samesame", "samesame")

            registry = mod.PluginRegistry(base / "plugins.db")
            registered = mod._discover_builtins(registry, builtin_dir=base)
            assert registered == 2, registered

            weird = registry.get("weird-name")
            assert weird is not None
            assert Path(weird.plugin_dir).resolve() == weird_dir.resolve(), (
                f"phantom name-derived path registered: {weird.plugin_dir}"
            )
            assert Path(weird.plugin_dir).is_dir(), (
                "the registered plugin directory must exist"
            )
            same = registry.get("samesame")
            assert Path(same.plugin_dir).resolve() == same_dir.resolve()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- a persisted phantom record heals, state preserved
# ---------------------------------------------------------------------------
def test_c3_persisted_phantom_record_heals_and_keeps_state():
    mod, restore = _load_manifest_module()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            weird_dir = _write_plugin(base, "weird_folder", "weird-name")

            registry = mod.PluginRegistry(base / "plugins.db")
            # The pre-fix world: same version, phantom name-derived path,
            # plugin enabled by an earlier boot.
            with open(weird_dir / "manifest.yaml", encoding="utf-8") as fh:
                import yaml
                manifest = mod.PluginManifest.from_dict(yaml.safe_load(fh))
            registry.register(
                manifest, str(base / "weird-name"), auto_enable=True,
            )
            stale = registry.get("weird-name")
            assert not Path(stale.plugin_dir).exists(), "fixture: phantom"
            assert stale.state == mod.PLUGIN_STATE_ENABLED

            mod._discover_builtins(registry, builtin_dir=base)

            healed = registry.get("weird-name")
            assert Path(healed.plugin_dir).resolve() == weird_dir.resolve(), (
                f"a dead persisted path must heal on discovery: "
                f"{healed.plugin_dir}"
            )
            assert healed.state == mod.PLUGIN_STATE_ENABLED, (
                "healing must not flip the plugin's enabled state"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- the version-refresh branch records the real directory
# ---------------------------------------------------------------------------
def test_c4_version_refresh_records_the_real_directory():
    mod, restore = _load_manifest_module()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            weird_dir = _write_plugin(
                base, "weird_folder", "weird-name", version="1.0.0",
            )

            registry = mod.PluginRegistry(base / "plugins.db")
            import yaml
            with open(weird_dir / "manifest.yaml", encoding="utf-8") as fh:
                old_manifest = mod.PluginManifest.from_dict(yaml.safe_load(fh))
            registry.register(
                old_manifest, str(base / "weird-name"), auto_enable=False,
            )

            # A new version lands on disk; discovery must refresh the
            # record with the REAL directory, not the phantom.
            _write_plugin(base, "weird_folder", "weird-name", version="1.1.0")
            mod._discover_builtins(registry, builtin_dir=base)

            record = registry.get("weird-name")
            assert record.manifest.version == "1.1.0", record.manifest.version
            assert Path(record.plugin_dir).resolve() == weird_dir.resolve(), (
                f"the refresh branch registered a phantom: {record.plugin_dir}"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner (pytest picks up the test_ functions; direct execution works too)
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    names = sorted(n for n in globals() if n.startswith("test_"))
    selected = [
        n for n in names if not argv or any(fragment in n for fragment in argv)
    ]
    failures = 0
    for name in selected:
        try:
            globals()[name]()
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
