"""Contracts for the cipher driver's errors: sqlite3's, and still the driver's own.

Every connection to SQLCipher is made in ``db_encryption``, and at its import each of the driver's eight error
subclasses is given its sqlite3 namesake as a base, so the package's ``except sqlite3.<Name>`` clauses catch the
driver's errors. Until 198 the namesake replaced the driver's own bases: the driver's ``OperationalError`` was no
longer its ``DatabaseError`` nor its ``Error``, and a class that refused was passed over in silence. From 186 to 197
the tree said that the cipher driver's errors are not sqlite3's, from a probe of the driver run without
``db_encryption``.

  * DX1 -- ON THE REAL DRIVER, ITS ERRORS ARE SQLITE3'S AND ITS OWN: once ``db_encryption`` has loaded, each of the
    eight classes descends from its sqlite3 namesake and keeps the driver's lineage; real errors of a real keyed file
    -- a duplicate key, a missing table, a column added twice, a lock, a wrong key, a closed connection, a parameter
    that cannot be bound -- are each caught by ``except sqlite3.<its name>`` and by the driver's own classes above it.
  * DX2 -- A CLASS THAT REFUSES IS SAID: a driver class that cannot take its namesake is logged as an error and named
    in ``_UNALIGNED``, the driver's other classes aligned all the same; on the real driver, what ``db_encryption``
    names as staying apart from sqlite3 does: its root ``Error`` and its ``Warning``, which CPython refuses to rebase,
    are not sqlite3's, and its ``Warning`` for two statements in one ``execute`` and its ValueError for a NUL in a
    statement's text escape an ``except sqlite3.Error`` (round 1).
  * DX3 -- THE PACKAGE SAYS WHAT THE DRIVER DOES: no module says again that the cipher driver's errors are not (or
    are never) sqlite3's, or that it does not share its classes with sqlite3 -- the sayings of 186 to 197, read in
    any comment or docstring, over line breaks and in their contracted forms (round 1).

Local-only, and skipped where the SQLCipher binding is not installed.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import sys
import types
from pathlib import Path

import pytest

sqlcipher3 = pytest.importorskip("sqlcipher3")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_census import modules as package_modules  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_dx1_on_the_real_driver_its_errors_are_sqlite3s_and_its_own": 3.0,
    "test_dx2_a_class_that_refuses_is_said": 3.0,
    "test_dx3_the_package_says_what_the_driver_does": 3.0,
}

_DBE = "opti_oignon.db_encryption"
_NAMES = ("InterfaceError", "DatabaseError", "DataError", "OperationalError", "IntegrityError", "InternalError",
          "ProgrammingError", "NotSupportedError")
# The driver's own lineage, as DB-API 2.0 and the driver define it.
_PARENT = {"InterfaceError": "Error", "DatabaseError": "Error", "DataError": "DatabaseError",
           "OperationalError": "DatabaseError", "IntegrityError": "DatabaseError", "InternalError": "DatabaseError",
           "ProgrammingError": "DatabaseError", "NotSupportedError": "DatabaseError"}


class _Said(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _loaded(seeded_driver=None):
    """``db_encryption`` loaded from its source, with what it logged at import; ``seeded_driver``, a stand-in
    ``pysqlcipher3.dbapi2``, is the binding it finds first when given. ``sqlite3.Row``, which the module aliases,
    and the stand-in are taken back by the closer."""
    said = _Said()
    logger = logging.getLogger(_DBE)
    logger.addHandler(said)
    level, row = logger.level, sqlite3.Row
    logger.setLevel(logging.DEBUG)
    planted = {}
    if seeded_driver is not None:
        package = types.ModuleType("pysqlcipher3")
        package.__path__ = []
        package.dbapi2 = seeded_driver
        planted = {"pysqlcipher3": package, "pysqlcipher3.dbapi2": seeded_driver}
    before = {name: sys.modules.get(name) for name in planted}
    sys.modules.update(planted)
    try:
        loaded, restore = isolate(targets={_DBE: source("db_encryption.py")})
    except BaseException:
        _close(planted, before, logger, said, level, row)
        raise

    def close():
        try:
            restore()
        finally:
            _close(planted, before, logger, said, level, row)

    return loaded[_DBE], said.records, close


def _close(planted, before, logger, said, level, row):
    for name in planted:
        if before[name] is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = before[name]
    logger.removeHandler(said)
    logger.setLevel(level)
    sqlite3.Row = row


def _caught(exc, name, driver):
    """Whether ``exc`` is caught by ``except sqlite3.<name>``, ``except sqlite3.Error`` and by each of the driver's own
    classes above ``name``."""
    above, klass = [], name
    while klass in _PARENT:
        klass = _PARENT[klass]
        above.append(klass)
    return {"sqlite3." + name: isinstance(exc, getattr(sqlite3, name)), "sqlite3.Error": isinstance(exc, sqlite3.Error),
            **{"driver." + k: isinstance(exc, getattr(driver, k)) for k in (name, *above)}}


def _raised(action):
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - what the driver raised is what is judged
        return exc
    raise AssertionError("the driver raised nothing")


def test_dx1_on_the_real_driver_its_errors_are_sqlite3s_and_its_own(tmp_path):
    dbe, said, close = _loaded()
    try:
        driver = dbe._sqlcipher_module
        assert dbe.SQLCIPHER_AVAILABLE and driver is not None
        for name in _NAMES:
            theirs = getattr(driver, name)
            assert issubclass(theirs, getattr(sqlite3, name)), (name, "not sqlite3's")
            assert issubclass(theirs, getattr(driver, _PARENT[name])), (name, "the driver's lineage dropped",
                                                                        [k.__name__ for k in theirs.__mro__])
            assert issubclass(theirs, driver.Error), (name, "no longer the driver's Error")
        assert dbe._UNALIGNED == [], dbe._UNALIGNED
        assert not [r for r in said if r.levelno >= logging.ERROR], [r.getMessage() for r in said]

        path = tmp_path / "keyed.db"
        conn = driver.connect(str(path), timeout=0.2)
        conn.execute("PRAGMA key = 'the right key'")
        conn.execute("CREATE TABLE t (a INTEGER PRIMARY KEY, b TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'x')")
        conn.commit()
        errors = {
            "a duplicate key": ("IntegrityError", _raised(lambda: conn.execute("INSERT INTO t VALUES (1, 'y')"))),
            "a missing table": ("OperationalError", _raised(lambda: conn.execute("SELECT * FROM nowhere"))),
            "a column added twice": ("OperationalError", _raised(lambda: conn.execute("ALTER TABLE t ADD COLUMN b"))),
            "a parameter that cannot be bound": ("InterfaceError",
                                                 _raised(lambda: conn.execute("SELECT ?", (object(),)))),
        }
        conn.rollback()
        other = driver.connect(str(path), timeout=0.2)
        other.execute("PRAGMA key = 'the right key'")
        other.execute("BEGIN IMMEDIATE")
        try:
            errors["a lock"] = ("OperationalError", _raised(lambda: conn.execute("BEGIN IMMEDIATE")))
        finally:
            other.rollback()
            other.close()
        wrong = driver.connect(str(path))
        wrong.execute("PRAGMA key = 'a wrong key'")
        try:
            errors["a wrong key"] = ("DatabaseError", _raised(lambda: wrong.execute("SELECT * FROM t").fetchall()))
        finally:
            wrong.close()
        conn.close()
        errors["a closed connection"] = ("ProgrammingError", _raised(lambda: conn.execute("SELECT 1")))

        for what, (name, exc) in errors.items():
            assert type(exc) is getattr(driver, name), (what, "the driver raised", type(exc))
            caught = _caught(exc, name, driver)
            assert all(caught.values()), (what, caught)
    finally:
        close()


def _stand_in():
    """A driver like the real one, but for its InternalError, which derives from Exception alone: CPython refuses it
    sqlite3.InternalError as a base, as it refuses the real driver's root Error."""
    driver = types.ModuleType("pysqlcipher3.dbapi2")
    driver.Error = type("Error", (Exception,), {})
    driver.Warning = type("Warning", (Exception,), {})
    driver.InterfaceError = type("InterfaceError", (driver.Error,), {})
    driver.DatabaseError = type("DatabaseError", (driver.Error,), {})
    for name in ("DataError", "OperationalError", "IntegrityError", "ProgrammingError", "NotSupportedError"):
        setattr(driver, name, type(name, (driver.DatabaseError,), {}))
    driver.InternalError = type("InternalError", (Exception,), {})
    driver.connect = sqlite3.connect
    return driver


def test_dx2_a_class_that_refuses_is_said():
    driver = _stand_in()
    dbe, said, close = _loaded(driver)
    try:
        assert dbe._sqlcipher_module is driver
        assert not issubclass(driver.InternalError, sqlite3.Error)
        errors = [r.getMessage() for r in said if r.levelno >= logging.ERROR]
        assert len(errors) == 1 and "InternalError" in errors[0] and "sqlite3.InternalError" in errors[0], errors
        assert getattr(dbe, "_UNALIGNED", None) == ["pysqlcipher3.dbapi2.InternalError"]
        for name in _NAMES:
            if name == "InternalError":
                continue
            theirs = getattr(driver, name)
            assert issubclass(theirs, getattr(sqlite3, name)) and issubclass(theirs, getattr(driver, _PARENT[name])), (
                name, [k.__name__ for k in theirs.__mro__])
    finally:
        close()

    real, said, close = _loaded()
    try:
        driver = real._sqlcipher_module
        assert not issubclass(driver.Error, sqlite3.Error) and not issubclass(driver.Warning, sqlite3.Error)
        conn = driver.connect(":memory:")
        try:
            exc = _raised(lambda: conn.execute("SELECT 1; SELECT 2"))
        finally:
            conn.close()
        assert type(exc) is driver.Warning and not isinstance(exc, sqlite3.Error), type(exc)
        conn = driver.connect(":memory:")
        try:
            exc = _raised(lambda: conn.execute("SELECT 1\x00"))
        finally:
            conn.close()
        assert type(exc) is ValueError and not isinstance(exc, sqlite3.Error), type(exc)
        assert real._UNALIGNED == [], real._UNALIGNED
    finally:
        close()


# The sayings of 186 to 197: the driver's errors not sqlite3's, its classes not shared with sqlite3.
_SAID_WRONG = (
    re.compile(r"errors (are not|aren't|are never|were never|are no) sqlite3'?s"),
    re.compile(r"(does not|doesn't|do not|don't|never) share[sd]? (its|their)? ?(classes |errors )?with sqlite3"),
    re.compile(r"not shared with sqlite3"),
)


def _said_wrong(text):
    """The wrong sayings in ``text``, its comment and docstring lines read as one run of words, lowercased."""
    flat = " ".join(line.strip().lstrip("#").strip() for line in text.lower().splitlines())
    return [found.group(0) for said in _SAID_WRONG for found in [said.search(flat)] if found]


def test_dx3_the_package_says_what_the_driver_does():
    found = {}
    walked = package_modules()
    for name, path in walked.items():
        wrong = _said_wrong(path.read_text(encoding="utf-8"))
        if wrong:
            found[name] = wrong
    assert not found, found
    assert "opti_oignon.db_encryption" in walked and len(walked) > 300, len(walked)
    # The reading finds a saying split over two comment lines, and its other forms.
    assert _said_wrong("            # noqa: BLE001 - the cipher driver's errors\n            # are not sqlite3's\n") == [
        "errors are not sqlite3's"]
    assert _said_wrong('    """Its classes, which the cipher\'s\n    driver does not share with sqlite3."""') == [
        "does not share with sqlite3"]
    assert _said_wrong("# the driver's errors aren't sqlite3's") and _said_wrong("# classes not shared with sqlite3")
    assert not _said_wrong("# the cipher's driver raises its own classes, each also sqlite3's")
