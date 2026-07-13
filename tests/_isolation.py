#!/usr/bin/env python3
"""Shared isolation window for contract suites that load a module from file.

A contract suite that exercises one module in isolation has to manufacture a
package window: the module under test is loaded from its file, and every other
project module it reaches for is either seeded by the test or made genuinely
unreachable. Absence has to be MANUFACTURED, not hoped for. A suite that draws
a conclusion from an absence it never created proves nothing, and a contract
that passes for a reason it never established is worse than one that fails --
it reassures.

Two INDEPENDENT routes let a real project module resolve behind the test's
back. A window has to close both, and closing one is not closing the other:

  * The module cache. Python consults ``sys.modules`` before it consults any
    finder. A module some earlier suite imported for real resolves straight
    out of the cache and no meta-path guard is ever asked. Evicting the key
    (``pop``) is not enough on its own -- see the second route.

  * A name-answering finder. An editable install registers a finder that
    resolves a submodule from a NAME table and ignores the parent package's
    ``__path__``. A stand-in parent whose path is empty therefore stops
    nothing, and the real module loads. Whether a window holds must not depend
    on how the project happens to be installed.

Neutralising a cache key to ``None`` closes the first route: the import
machinery raises ImportError on a ``None`` entry ahead of every finder. A
guard at the head of ``sys.meta_path`` closes the second: it refuses every
project name the test did not seed, whatever a finder further down the chain
would have answered. This module always installs both, and it closes the
window by construction -- every project key that is neither seeded nor a
target is neutralised, so nothing is left to whatever an earlier suite
happened to leave in the cache.

The window PROVES ITSELF. Before the module under test is executed, every name
the caller declared absent is import-attempted for real and must fail. A seal
that does not block what it claims to block raises ``SealFailure`` here,
loudly, at the point of the defect -- instead of handing back a window that
blocks nothing and letting the suite reach a green verdict on empty air. The
proof runs on every call, in every install layout, in any execution order.

Import-safe and stdlib-only. Used by contract suites; never by the package.
"""

import contextlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = "opti_oignon"

_PREFIX = PACKAGE + "."
_ABSENT = object()


class SealFailure(AssertionError):
    """The window does not block what the caller declared absent.

    Raised by the witness, not by a test body. It means the isolation is void:
    the suite was about to test something other than what it says it tests.
    """


class _NameGuard:
    """Refuse every project name the caller did not seed.

    Sits at the head of ``sys.meta_path``, ahead of every other finder, so a
    finder that answers on the module NAME and ignores the parent path cannot
    resolve a project module behind the window's back. Seeded names and
    targets are in ``sys.modules`` before this guard can ever be consulted, so
    refusing the whole namespace here is safe and is the strictest form.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == PACKAGE or fullname.startswith(_PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


def source(*parts):
    """Path to a project source file, e.g. ``source("agent", "tools.py")``."""
    return REPO.joinpath(PACKAGE, *parts)


def _attach(stubs, name, module):
    """Bind ``module`` as an attribute of its stand-in parent package."""
    parent, _, leaf = name.rpartition(".")
    if parent in stubs:
        setattr(stubs[parent], leaf, module)


def prove_unreachable(name):
    """Import ``name`` for real inside the window; it MUST fail.

    This is the witness, and it is the whole point of the module. It exercises
    both ways production code reaches a submodule -- the direct import and the
    ``from <parent> import <leaf>`` form, which resolves through the parent's
    attributes and the cache rather than only through ``import_module``.

    Raises SealFailure when either one still resolves.
    """
    try:
        importlib.import_module(name)
    except ImportError:
        pass
    else:
        raise SealFailure(
            f"isolation seal is void: {name} still imports inside the window"
        )

    parent, _, leaf = name.rpartition(".")
    if not parent:
        return
    try:
        module = __import__(parent, fromlist=[leaf])
        getattr(module, leaf)
    except (ImportError, AttributeError):
        return
    raise SealFailure(
        f"isolation seal is void: 'from {parent} import {leaf}' still resolves"
    )


def isolate(*, targets, blocked=(), seeded=None, packages=()):
    """Open an isolation window and load ``targets`` from their source files.

    targets  -- ordered mapping of dotted name -> source path. Each is
                registered in ``sys.modules`` before it is executed, so one
                target may import another.
    blocked  -- dotted names the caller declares UNREACHABLE. Each is
                neutralised and then PROVEN unreachable before any target
                runs. Naming a module here is a contract statement: the suite
                asserts something about what the code does in its absence.
    seeded   -- mapping of dotted name -> stand-in module supplied by the
                caller. Seeded names resolve; nothing else in the package does.
    packages -- extra stand-in parent packages, e.g. ``"opti_oignon.agent"``.
                The root package is always stood in for.

    Returns ``(loaded, restore)``. ``loaded`` maps each target name to its
    module. ``restore`` closes the window and puts ``sys.modules`` and
    ``sys.meta_path`` back exactly as they were, including keys that were
    absent before.
    """
    targets = dict(targets)
    seeded = dict(seeded or {})
    package_names = [PACKAGE, *packages]

    # Every project key already in the cache is snapshotted AND neutralised
    # below: those keys are precisely the ones that would otherwise resolve
    # out of the cache, ahead of any guard, and silently hand the test live
    # code. Nothing is left to what an earlier suite happened to leave behind.
    cached = [k for k in sys.modules if k == PACKAGE or k.startswith(_PREFIX)]
    touched = (
        set(cached)
        | set(package_names)
        | set(targets)
        | set(blocked)
        | set(seeded)
    )
    saved = {key: sys.modules.get(key, _ABSENT) for key in touched}

    guard = _NameGuard()
    sys.meta_path.insert(0, guard)

    def restore():
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)
        for key, value in saved.items():
            if value is _ABSENT:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    try:
        stubs = {}
        for name in package_names:
            stub = types.ModuleType(name)
            stub.__path__ = []
            sys.modules[name] = stub
            stubs[name] = stub
        for name, stub in stubs.items():
            _attach(stubs, name, stub)

        for key in set(cached) | set(blocked):
            if key in stubs or key in seeded or key in targets:
                continue
            sys.modules[key] = None

        for name, module in seeded.items():
            sys.modules[name] = module
            _attach(stubs, name, module)

        for name in blocked:
            prove_unreachable(name)

        loaded = {}
        for name, path in targets.items():
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            _attach(stubs, name, module)
            spec.loader.exec_module(module)
            loaded[name] = module
    except BaseException:
        restore()
        raise

    return loaded, restore


@contextlib.contextmanager
def isolation(**kwargs):
    """Context-manager form of ``isolate``; restores on any exit path."""
    loaded, restore = isolate(**kwargs)
    try:
        yield loaded
    finally:
        restore()
