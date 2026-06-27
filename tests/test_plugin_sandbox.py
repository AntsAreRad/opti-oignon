#!/usr/bin/env python3
"""Security tests for the in-process plugin sandbox (plugin_loader).

These three context managers are the in-process defense layer that runs while
a plugin's entry point is exec'd:

  * _RestrictedImporter   -- a meta-path finder that raises on blocked imports
    (os, sys, subprocess, importlib, pickle, ...) and on network modules unless
    the plugin holds network_outbound permission.
  * _RestrictedPathAccessor -- patches open / io.open / Path.* so a plugin can
    only touch files inside its own directory (plus stdlib / site-packages).
  * _RestrictedBuiltins   -- blocks globals/vars outright, blocks exec/eval/
    compile when the *caller* is plugin code, and wraps __import__ to block the
    same module set as the importer (the importlib / __import__ bypass).

The context-manager cases patch GLOBAL builtins for the duration of the block,
so each test performs the risky operation inside the ``with`` and records the
outcome, then asserts AFTER the block exits (when the originals are restored)
to avoid disturbing pytest's own machinery mid-patch.
"""

import sys
from pathlib import Path

from opti_oignon.plugin_loader import (
    _BLOCKED_IMPORTS,
    _NETWORK_MODULES,
    PluginSandboxViolation,
    _RestrictedBuiltins,
    _RestrictedImporter,
    _RestrictedPathAccessor,
)

# Sanity: the modules we rely on for the security assertions are actually in
# the blocked sets (guards against the sets being gutted).
assert "os" in _BLOCKED_IMPORTS
assert "importlib" in _BLOCKED_IMPORTS      # the documented bypass vector
assert "socket" in _NETWORK_MODULES


# ===========================================================================
# _RestrictedImporter  (pure -- find_spec raises or returns None)
# ===========================================================================

def _importer(has_network: bool = False) -> _RestrictedImporter:
    return _RestrictedImporter(_BLOCKED_IMPORTS, _NETWORK_MODULES, has_network)


def test_importer_blocks_os():
    imp = _importer()
    try:
        imp.find_spec("os")
        raised = False
    except PluginSandboxViolation:
        raised = True
    assert raised


def test_importer_blocks_importlib_bypass():
    imp = _importer()
    try:
        imp.find_spec("importlib")
        raised = False
    except PluginSandboxViolation:
        raised = True
    assert raised


def test_importer_blocks_subprocess():
    imp = _importer()
    try:
        imp.find_spec("subprocess")
        raised = False
    except PluginSandboxViolation:
        raised = True
    assert raised


def test_importer_blocks_submodule_by_top_level():
    # os.path must be blocked because its top-level package (os) is blocked.
    imp = _importer()
    try:
        imp.find_spec("os.path")
        raised = False
    except PluginSandboxViolation:
        raised = True
    assert raised


def test_importer_allows_safe_module():
    imp = _importer()
    assert imp.find_spec("json") is None        # None == allowed


def test_importer_blocks_network_without_permission():
    imp = _importer(has_network=False)
    try:
        imp.find_spec("socket")
        raised = False
    except PluginSandboxViolation:
        raised = True
    assert raised


def test_importer_allows_network_with_permission():
    imp = _importer(has_network=True)
    assert imp.find_spec("socket") is None      # permitted -> allowed


def test_importer_deactivate_stops_blocking():
    imp = _importer()
    imp.deactivate()
    assert imp.find_spec("os") is None          # no longer blocking


# ===========================================================================
# _RestrictedPathAccessor._is_allowed  (pure predicate)
# ===========================================================================

def test_is_allowed_inside_plugin_dir(tmp_path):
    pa = _RestrictedPathAccessor([tmp_path])
    assert pa._is_allowed(tmp_path / "data.txt") is True


def test_is_allowed_rejects_outside_dir(tmp_path):
    pa = _RestrictedPathAccessor([tmp_path / "plugin"])
    assert pa._is_allowed(Path("/etc/hostname")) is False


def test_is_allowed_permits_stdlib_paths(tmp_path):
    pa = _RestrictedPathAccessor([tmp_path / "plugin"])
    # A path under any sys.path entry (stdlib / site-packages) is permitted.
    sp_dir = next(
        p for p in sys.path if p and p not in ("", ".") and Path(p).is_dir()
    )
    assert pa._is_allowed(Path(sp_dir) / "some_module.py") is True


# ===========================================================================
# _RestrictedPathAccessor enforcement  (global open patch -- careful)
# ===========================================================================

def test_path_accessor_blocks_open_outside_dir(tmp_path):
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    inside = plugin_dir / "data.txt"
    inside.write_text("ok")
    outside = tmp_path / "outside.txt"      # sibling of plugin_dir, not allowed
    outside.write_text("secret")

    outcome = {}
    with _RestrictedPathAccessor([plugin_dir]):
        try:
            with open(inside) as f:
                f.read()
            outcome["inside"] = "ok"
        except PluginSandboxViolation:
            outcome["inside"] = "blocked"
        try:
            open(outside)
            outcome["outside"] = "ok"
        except PluginSandboxViolation:
            outcome["outside"] = "blocked"

    assert outcome["inside"] == "ok"        # files in the plugin dir are fine
    assert outcome["outside"] == "blocked"  # everything else is denied


# ===========================================================================
# _RestrictedBuiltins  (global builtins patch -- careful)
# ===========================================================================

def test_builtins_block_globals():
    with _RestrictedBuiltins():
        try:
            globals()
            raised = False
        except PluginSandboxViolation:
            raised = True
    assert raised


def test_builtins_block_vars():
    with _RestrictedBuiltins():
        try:
            vars()
            raised = False
        except PluginSandboxViolation:
            raised = True
    assert raised


def test_builtins_block_import_of_blocked_module():
    with _RestrictedBuiltins():
        try:
            __import__("subprocess")
            blocked = False
        except PluginSandboxViolation:
            blocked = True
        # a safe module still imports
        safe = __import__("json")
    assert blocked is True
    assert safe is not None


def test_builtins_eval_blocked_only_for_plugin_callers():
    # A function whose module name marks it as plugin code.
    plugin_ns = {"__name__": "_opti_plugin_fake"}
    exec("def calls_eval():\n    return eval('1 + 1')", plugin_ns)
    calls_eval = plugin_ns["calls_eval"]

    with _RestrictedBuiltins():
        # Caller is this (non-plugin) test module -> eval permitted.
        try:
            direct = eval("2 + 2")
            direct_ok = True
        except PluginSandboxViolation:
            direct_ok = False
            direct = None
        # Caller is plugin code -> eval blocked.
        try:
            calls_eval()
            plugin_blocked = False
        except PluginSandboxViolation:
            plugin_blocked = True

    assert direct_ok is True
    assert direct == 4
    assert plugin_blocked is True


def test_builtins_restored_after_context():
    # After the context exits, the real builtins are back.
    before = globals  # the real one
    with _RestrictedBuiltins():
        pass
    assert globals is before
    assert globals().get("__name__") is not None   # callable again, no raise
