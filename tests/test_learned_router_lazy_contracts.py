#!/usr/bin/env python3
"""Contracts for deferring the classifier's dependency until it is used.

The learned router names a classifier it may never build. Its dependency was
imported at module level inside a ``try``, which the flag pattern makes look
optional and is not: a ``try`` at module scope runs at import like anything
else, and the import succeeds, so the cost is paid in full. Measured, that
one import is the single most expensive thing the package does at load, and
it drags two more libraries behind it that nothing here imports directly.

Availability is a different question from readiness. Whether the dependency
CAN be imported is answerable without importing it, and that is the whole of
what the flag needs to say. The concrete names are imported where they are
used, which is inside three methods that only run once someone asks the
router to train, score or load a model.

  * LR1 -- loading the module does not load the dependency.
  * LR2 -- the availability flag is still true where the dependency exists,
    so nothing downstream changes behaviour.
  * LR3 -- availability is decided by locating the dependency, not by
    importing it. Stated structurally, because the difference is invisible
    in the flag's value and is the entire point.
  * LR4 -- the module carries no module-level import of the dependency.
  * LR5 -- the package import no longer loads it, nor the two libraries it
    pulled behind it. Measured by importing, not asserted about.
"""

import ast
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODULE = REPO / "opti_oignon" / "learned_router.py"
GUARD = REPO / ".github" / "scripts" / "import_footprint_guard.py"

# Loaded on its own, by path, in a subprocess: the question is what THIS
# module costs, and importing it through the package would import everything.
_PROBE = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("_lr_under_contract", {path!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.stdout.write("<<LR>>" + json.dumps({{
    "sklearn_loaded": "sklearn" in sys.modules,
    "available": bool(getattr(module, "SKLEARN_AVAILABLE", None)),
    "router_available": bool(getattr(module, "LEARNED_ROUTER_AVAILABLE", None)),
}}))
"""


def _probe():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(path=str(MODULE))],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert "<<LR>>" in result.stdout, (
        f"the probe did not run: {result.stderr[-800:]}"
    )
    import json
    return json.loads(result.stdout.split("<<LR>>", 1)[1])


def _guard():
    loaded, restore = isolate(targets={"_footprint": GUARD})
    return loaded["_footprint"], restore


def test_lr1_loading_the_module_does_not_load_the_dependency():
    seen = _probe()
    assert seen["sklearn_loaded"] is False, (
        "the dependency is loaded merely by loading the module that might "
        "one day use it; a capability costs nothing until it is called"
    )


def test_lr2_the_availability_flag_is_still_true_where_it_exists():
    seen = _probe()
    assert seen["available"] is True, (
        "deferring the import must not change what the flag reports, or "
        "every caller that branches on it changes behaviour"
    )
    assert seen["router_available"] is True


def test_lr3_availability_is_decided_by_locating_not_importing():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "find_spec" in names, (
        "the flag must be decided by locating the dependency; importing it "
        "to find out whether it can be imported pays the whole cost to "
        "answer a question that does not need it"
    )


def test_lr4_no_module_level_import_of_the_dependency():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    offenders = []

    def walk(node, depth):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # inside a function is exactly where it belongs
            if isinstance(child, ast.ImportFrom) and child.module:
                if child.module.split(".")[0] in {"sklearn", "joblib"}:
                    offenders.append(f"line {child.lineno}: {child.module}")
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name.split(".")[0] in {"sklearn", "joblib"}:
                        offenders.append(f"line {child.lineno}: {alias.name}")
            walk(child, depth + 1)

    walk(tree, 0)
    assert not offenders, (
        "a try at module scope runs at import like anything else, so an "
        f"import nested in one is not deferred: {offenders}"
    )


def test_lr5_the_package_import_no_longer_loads_it():
    """Measured by importing the package, not asserted about it."""
    guard, restore = _guard()
    try:
        seen = guard.observe()
        assert seen is not None, "the probe did not run; nothing was measured"
        for name in ("sklearn", "pandas", "scipy"):
            assert name not in seen["heavy"], (
                f"{name} is still loaded by importing the package: "
                f"{seen['heavy']}"
            )
            # Inside the loop on purpose. Asserted after it, this read the
            # leaked loop variable and only ever checked the last name --
            # which its own blade caught by putting a different one back.
            assert name not in guard.LEDGER["heavy"], (
                f"{name} no longer loads, so it must come off the ledger too"
            )
        # Proven capable: the probe still reports the ones that remain.
        assert seen["heavy"], "the probe reported no heavy module at all"
    finally:
        restore()


def test_lr6_the_package_import_loads_no_heavy_module_at_all():
    """Supersedes lr5: its control asked the probe for a heavy module at
    import, and there is none since the facade resolves its exports lazily."""
    guard, restore = _guard()
    try:
        seen = guard.observe()
        assert seen is not None, "the probe did not run; nothing was measured"
        for name in ("sklearn", "pandas", "scipy"):
            assert name not in seen["heavy"], f"{name} is still loaded by importing the package: {seen['heavy']}"
            assert name not in guard.LEDGER["heavy"], f"{name} no longer loads, so it must come off the ledger too"
        assert seen["heavy"] == [] and guard.LEDGER["heavy"] == frozenset()
        assert seen["modules"] >= 20, "proven capable: the probe counts the modules the interpreter loads"
    finally:
        restore()
