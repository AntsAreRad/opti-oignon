#!/usr/bin/env python3
"""The componion engine's modules, opened in the shared isolation window.

The reference modules are loaded from their files; the native loader too,
when a contract compares the two engines, so it finds the artefact that
``scripts/build_oo_core.sh`` installed. Nothing else in the package is
reachable from inside the window.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

REFERENCE = ("wire", "fx", "rng", "lawfiles")
ORGANS = ("genome", "compile", "bounds")
PACKAGES = ("opti_oignon.allium", "opti_oignon.allium.ref", "opti_oignon.allium.ref.organs")


def open_allium(*, native=True, seeded=None):
    """Load the reference, the seam and (when asked) the native loader; ``(loaded, restore)``."""
    targets = {f"opti_oignon.allium.{name}": source("allium", f"{name}.py") for name in REFERENCE}
    # The organs before the protocol: it imports them when it is executed.
    for name in ORGANS:
        targets[f"opti_oignon.allium.ref.organs.{name}"] = source("allium", "ref", "organs", f"{name}.py")
    targets["opti_oignon.allium.ref.protocol"] = source("allium", "ref", "protocol.py")
    targets["opti_oignon.allium.engine"] = source("allium", "engine.py")
    if native:
        targets["opti_oignon.native"] = source("native", "__init__.py")
    return isolate(targets=targets, packages=PACKAGES, seeded=seeded)


def native_module(loaded):
    """The built native core, or a failure that says how to build it."""
    module = loaded["opti_oignon.native"].load()
    assert module is not None and hasattr(module, "allium_call"), (
        "the native core with the componion's engine is not built here: run scripts/build_oo_core.sh"
    )
    return module
