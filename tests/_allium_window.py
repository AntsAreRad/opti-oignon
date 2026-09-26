#!/usr/bin/env python3
"""The componion engine's modules, opened in the shared isolation window.

The reference modules are loaded from their files; the native loader too,
when a contract compares the two engines, so it finds the artefact that
``scripts/build_oo_core.sh`` installed. The platform -- the store, the
membrane, the chain and the anchors, which have no twin -- loads only when a
contract asks for it, after the seam and in a fixed order, since the window
executes its targets in order. Nothing else in the package is reachable from
inside the window, and every name a contract declares blocked is proven
unreachable before anything runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

REFERENCE = ("wire", "fx", "rng", "lawfiles")
ORGANS = ("genome", "compile", "bounds", "phon")
# The life's reference modules under ``ref/``, after the fact identity and before the protocol, in this order.
LIFE_REF = ("civil", "lawdata")
# The life's organs under ``ref/organs/``, after them: the weather first, which the soil reads.
LIFE_ORGANS = ("weather", "clock", "chem", "soil", "stage")
PACKAGES = ("opti_oignon.allium", "opti_oignon.allium.ref", "opti_oignon.allium.ref.organs")
# The platform modules, in the order they load: each may name the ones before it. The life's two
# (the law timeline and the views) come last.
PLATFORM = ("settings", "mode", "chain", "membrane", "anchors", "store", "evolution", "life")


def open_allium(*, native=True, seeded=None, blocked=(), platform=False, extra=None):
    """Load the reference, the seam and (when asked) the platform and the native loader; ``(loaded, restore)``.

    ``extra`` maps further dotted names to source files, loaded last: a
    contract that names a real platform module (the mode manager, say) loads
    it here, and must then leave it out of ``blocked``.
    """
    targets = {f"opti_oignon.allium.{name}": source("allium", f"{name}.py") for name in REFERENCE}
    # The organs before the protocol: it imports them when it is executed.
    for name in ORGANS:
        targets[f"opti_oignon.allium.ref.organs.{name}"] = source("allium", "ref", "organs", f"{name}.py")
    # The fact identity before the protocol too, for the same reason.
    targets["opti_oignon.allium.ref.journal"] = source("allium", "ref", "journal.py")
    for name in LIFE_REF:
        targets[f"opti_oignon.allium.ref.{name}"] = source("allium", "ref", f"{name}.py")
    for name in LIFE_ORGANS:
        targets[f"opti_oignon.allium.ref.organs.{name}"] = source("allium", "ref", "organs", f"{name}.py")
    # The life itself, which folds a being over the organs, before the protocol that answers for it.
    targets["opti_oignon.allium.ref.world"] = source("allium", "ref", "world.py")
    targets["opti_oignon.allium.ref.protocol"] = source("allium", "ref", "protocol.py")
    targets["opti_oignon.allium.engine"] = source("allium", "engine.py")
    if platform:
        for name in PLATFORM:
            targets[f"opti_oignon.allium.{name}"] = source("allium", f"{name}.py")
    if native:
        targets["opti_oignon.native"] = source("native", "__init__.py")
    targets.update(extra or {})
    return isolate(targets=targets, packages=PACKAGES, seeded=seeded, blocked=blocked)


def native_module(loaded):
    """The built native core, or a failure that says how to build it."""
    module = loaded["opti_oignon.native"].load()
    assert module is not None and hasattr(module, "allium_call"), (
        "the native core with the componion's engine is not built here: run scripts/build_oo_core.sh"
    )
    return module
