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
# (the law timeline and the views) come next, then the garden's five: the habitat, the ethics nets, the
# catalogue of lines, the description and the service the terminal uses.
PLATFORM = ("settings", "mode", "chain", "membrane", "anchors", "store", "evolution", "life",
            "habitat", "ethics", "wording", "describe", "service")
# The terminal's modules, loaded after the platform when a contract drives ``oo garden``.
CLI = ("config", "client", "output", "main", "garden")
# The API's modules for the garden's routes, loaded last (after the terminal when both are asked), in this
# order: the schemas, the platform's auth routes whose user dependency the garden's router carries, the router.
API = ("schemas_allium", "routes_auth", "routes_allium")


def open_allium(*, native=True, seeded=None, blocked=(), platform=False, extra=None, cli=False, api=False):
    """Load the reference, the seam and (when asked) the platform and the native loader; ``(loaded, restore)``.

    ``extra`` maps further dotted names to source files, loaded last: a
    contract that names a real platform module (the mode manager, say) loads
    it here, and must then leave it out of ``blocked``. ``cli`` adds the
    terminal's modules after the platform, under a stand-in ``opti_oignon.cli``
    package, so ``oo garden`` runs inside the window. ``api`` adds the garden's
    routes (``API``) after them, under a stand-in ``opti_oignon.api`` package;
    the platform's dependency module (``opti_oignon.api.deps``) and the
    emergency stop are the caller's to seed.
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
    packages = PACKAGES
    if cli:
        packages = PACKAGES + ("opti_oignon.cli",)
        for name in CLI:
            targets[f"opti_oignon.cli.{name}"] = source("cli", f"{name}.py")
    if api:
        packages = packages + ("opti_oignon.api",)
        for name in API:
            targets[f"opti_oignon.api.{name}"] = source("api", f"{name}.py")
    if native:
        targets["opti_oignon.native"] = source("native", "__init__.py")
    targets.update(extra or {})
    return isolate(targets=targets, packages=packages, seeded=seeded, blocked=blocked)


def native_module(loaded):
    """The built native core, or a failure that says how to build it."""
    module = loaded["opti_oignon.native"].load()
    assert module is not None and hasattr(module, "allium_call"), (
        "the native core with the componion's engine is not built here: run scripts/build_oo_core.sh"
    )
    return module
