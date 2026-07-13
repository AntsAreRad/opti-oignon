#!/usr/bin/env python3
"""Contracts for the shared isolation window.

Every contract suite that loads a project module from its file depends on this
window to manufacture the absences it reasons about. The window is therefore a
single point of failure for the whole contract estate, and a copied idiom that
nothing proves is exactly how an estate ends up green for the wrong reason.
So the window is pinned here, against both routes that let a real project
module resolve behind a test's back:

  * I1 -- a name declared absent is unreachable inside the window, by BOTH the
    forms production code uses: the direct import and ``from <parent> import
    <leaf>``, which resolves through the parent's attributes rather than
    through the import machinery alone;
  * I2 -- the seal holds against a POLLUTED CACHE. The real module is imported
    for real first; inside the window it is still unreachable. Python consults
    ``sys.modules`` before any finder, so a cache key left in place resolves
    live code and no guard is ever asked;
  * I3 -- the seal holds against a finder that ANSWERS ON THE NAME and ignores
    the parent package's path. Such a finder is what an editable install
    registers, so whether a window holds must not depend on how the project
    happens to be installed;
  * I4 -- a project name the caller NEVER NAMED and never seeded is refused
    too, even when a name-answering finder would have resolved it. The window
    is closed by construction, not by an enumeration a human has to get right;
  * I5 -- the window PROVES ITSELF: a name declared absent that nonetheless
    resolves raises rather than yielding a window that blocks nothing;
  * I6 -- a seeded name resolves to the caller's stand-in, never to the real
    module;
  * I7 -- restore puts ``sys.modules`` and ``sys.meta_path`` back exactly,
    including keys that were absent before the window opened;
  * I8 -- a failure while loading a target still restores: no guard is left on
    the meta path and no stand-in package is left in the cache.

I1 and I3 are conjunction clauses: each of the two seals is on its own
sufficient for a name the caller declared, so neither seal alone accounts for
them. That redundancy is the property they assert. I2 and I4 map one to one
onto the two seals and are what fails when either is removed.

Local-only, stdlib-only. Runs under pytest or the __main__ runner.
"""

import importlib
import importlib.util
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _isolation  # noqa: E402
from _isolation import PACKAGE, SealFailure, isolate, prove_unreachable, source  # noqa: E402

# A project module that is stdlib-only at module top and free of import-time
# side effects, so importing it for real (I2) costs nothing and proves the
# cache route without dragging the runtime in.
_REAL = "opti_oignon.model_provenance"
_REAL_PATH = source("model_provenance.py")

# A name no real module answers to. Only the stand-in finder below claims it,
# which is what makes it the probe for the guard: nothing but a name-answering
# finder could ever resolve it, so if it resolves, the guard is not in force.
_UNNAMED = "opti_oignon.probe_not_seeded"


class _NameAnsweringFinder:
    """Stand-in for the finder an editable install registers.

    It resolves a submodule from a NAME table and ignores the parent package's
    ``__path__`` entirely -- which is precisely why a stand-in parent whose
    path is empty stops nothing, and why the window cannot rely on one.
    """

    def __init__(self, table):
        self._table = dict(table)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self._table:
            return importlib.util.spec_from_file_location(
                fullname, self._table[fullname]
            )
        return None


def _live_source(tmp_path, name):
    """A source file that marks itself as live code, for the finder to serve."""
    path = Path(tmp_path) / f"{name}.py"
    path.write_text("MARK = 'live'\n", encoding="ascii")
    return path


def _with_finder(table):
    """Install a name-answering finder at the TAIL of the meta path.

    The tail is where an editable install puts its finder, and it is the
    position that matters: a window whose guard does not sit ahead of every
    finder is a window such a finder walks straight through.
    """
    finder = _NameAnsweringFinder(table)
    sys.meta_path.append(finder)
    return finder


def _drop_finder(finder):
    if finder in sys.meta_path:
        sys.meta_path.remove(finder)


def _forget(name):
    """Drop a project module from the cache, so a clause starts from a known
    state instead of inheriting whatever a neighbouring clause left behind."""
    sys.modules.pop(name, None)


def _unreachable(name):
    """True when ``name`` cannot be imported by either form production uses."""
    try:
        prove_unreachable(name)
    except SealFailure:
        return False
    return True


# I1 -- a declared absence is a real absence, by both import forms


def test_i1_a_declared_absence_is_unreachable_by_both_import_forms(tmp_path):
    _forget("opti_oignon.security_mode")
    loaded, restore = isolate(
        targets={_REAL: _REAL_PATH},
        blocked=("opti_oignon.security_mode",),
    )
    try:
        with_error = None
        try:
            importlib.import_module("opti_oignon.security_mode")
        except ImportError as exc:
            with_error = exc
        assert with_error is not None, (
            "a name declared absent must not import inside the window"
        )

        from_error = None
        try:
            module = __import__(PACKAGE, fromlist=["security_mode"])
            getattr(module, "security_mode")
        except (ImportError, AttributeError) as exc:
            from_error = exc
        assert from_error is not None, (
            "'from <parent> import <leaf>' is how production reaches a "
            "submodule; the window must close that form too"
        )
        assert loaded[_REAL] is sys.modules[_REAL]
    finally:
        restore()


# I2 -- the cache route: a key an earlier suite imported for real


def test_i2_the_seal_holds_against_a_polluted_module_cache(tmp_path):
    importlib.import_module("opti_oignon.security_mode")
    assert sys.modules.get("opti_oignon.security_mode") is not None, (
        "the pollution this clause is about was not established"
    )

    loaded, restore = isolate(
        targets={_REAL: _REAL_PATH},
        blocked=("opti_oignon.security_mode",),
    )
    try:
        assert _unreachable("opti_oignon.security_mode"), (
            "a module an earlier suite imported for real resolves out of "
            "sys.modules ahead of every finder; evicting the key is what "
            "makes the absence real"
        )
    finally:
        restore()


# I3 -- the finder route: a finder that answers on the name


def test_i3_the_seal_holds_against_a_name_answering_finder(tmp_path):
    _forget("opti_oignon.security_mode")
    finder = _with_finder(
        {"opti_oignon.security_mode": _live_source(tmp_path, "security_mode")}
    )
    try:
        loaded, restore = isolate(
            targets={_REAL: _REAL_PATH},
            blocked=("opti_oignon.security_mode",),
        )
        try:
            assert _unreachable("opti_oignon.security_mode"), (
                "a finder that answers on the name ignores the parent path, "
                "so an empty stand-in path stops nothing; the window must "
                "hold whatever the install layout is"
            )
        finally:
            restore()
    finally:
        _drop_finder(finder)


# I4 -- closed by construction: a name the caller never enumerated


def test_i4_a_name_the_caller_never_declared_is_refused_too(tmp_path):
    _forget(_UNNAMED)
    leaf = _UNNAMED.rpartition(".")[2]
    finder = _with_finder({_UNNAMED: _live_source(tmp_path, leaf)})
    try:
        loaded, restore = isolate(targets={_REAL: _REAL_PATH})
        try:
            assert _unreachable(_UNNAMED), (
                "the window must not depend on a human enumerating every "
                "lazy import correctly: a project name that is neither "
                "seeded nor a target is refused, whatever a finder answers"
            )
        finally:
            restore()
    finally:
        _drop_finder(finder)
    assert _UNNAMED not in sys.modules


# I5 -- the window proves itself


def test_i5_a_void_seal_raises_instead_of_yielding_a_dead_window(tmp_path):
    reachable = importlib.import_module("json")
    assert reachable is sys.modules["json"]

    raised = None
    try:
        prove_unreachable("json")
    except SealFailure as exc:
        raised = exc
    assert raised is not None, (
        "the witness is the whole point: a name that still resolves must "
        "fail loudly at the seal, not silently at the assertion"
    )

    # A window stands its root package in, so declaring that same root absent
    # is a contradiction in terms -- and one that borrows nothing from any
    # other part of the window, so this clause answers for the witness alone.
    contradicted = None
    try:
        isolate(targets={_REAL: _REAL_PATH}, blocked=(PACKAGE,))
    except SealFailure as exc:
        contradicted = exc
    assert contradicted is not None, (
        "isolate must run the witness itself: a name declared absent that "
        "nonetheless resolves is a window that proves nothing"
    )


# I6 -- a seeded name resolves to the caller's stand-in


def test_i6_a_seeded_name_resolves_to_the_stand_in(tmp_path):
    _forget("opti_oignon.security_mode")
    stub = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("opti_oignon.security_mode", loader=None)
    )
    stub.MARK = "stand-in"
    loaded, restore = isolate(
        targets={_REAL: _REAL_PATH},
        seeded={"opti_oignon.security_mode": stub},
    )
    try:
        resolved = importlib.import_module("opti_oignon.security_mode")
        assert resolved is stub, "a seeded name must resolve to the stand-in"
        assert getattr(resolved, "MARK", None) == "stand-in", (
            "the real module must never be what a seeded name resolves to"
        )
    finally:
        restore()


# I7 -- restore is exact


def test_i7_restore_puts_the_cache_and_the_meta_path_back_exactly(tmp_path):
    importlib.import_module("opti_oignon.security_mode")
    before_modules = {
        key: value
        for key, value in sys.modules.items()
        if key == PACKAGE or key.startswith(PACKAGE + ".")
    }
    before_meta = list(sys.meta_path)

    loaded, restore = isolate(
        targets={_REAL: _REAL_PATH},
        blocked=("opti_oignon.telemetry",),
    )
    restore()

    after_modules = {
        key: value
        for key, value in sys.modules.items()
        if key == PACKAGE or key.startswith(PACKAGE + ".")
    }
    assert after_modules.keys() == before_modules.keys(), (
        "restore must leave no key behind and take none away"
    )
    for key, value in before_modules.items():
        assert after_modules[key] is value, (
            f"restore must put the same object back for {key}"
        )
    assert list(sys.meta_path) == before_meta, (
        "a guard left on the meta path poisons every later suite in the "
        "process"
    )


# I8 -- a failed load still restores


def test_i8_a_failure_while_loading_still_closes_the_window(tmp_path):
    before_meta = list(sys.meta_path)

    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('boom')\n", encoding="ascii")

    raised = None
    try:
        isolate(targets={"opti_oignon.broken": broken})
    except RuntimeError as exc:
        raised = exc
    assert raised is not None, "the loading failure must reach the caller"

    assert list(sys.meta_path) == before_meta, (
        "a window that fails to open must still take its guard back off; a "
        "guard left behind refuses project imports for every later suite in "
        "the process. Whether restore is EXACT is I7's subject -- one restore "
        "serves both paths -- so this clause answers only for it being CALLED."
    )


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fn(Path(tmp))
                print(f"PASS {fn.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL {fn.__name__}")
                traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
