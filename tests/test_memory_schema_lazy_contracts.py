#!/usr/bin/env python3
"""Contracts for building the memory schema on use, not on construction.

The memory manager is constructed at module scope and the executor imports
the package that imports it, so importing the package opened another database
with encryption not enforced before any caller had asked for anything.

Ten methods in this class hold the instance lock and then ask for a
connection, which is the shape that deadlocked the conversation manager when
the schema was built from inside the connection helper. The remedy is the
same and it is the reason this module is done on its own rather than batched
with the caches: the builder is handed a connection instead of opening one,
and takes no lock.

  * MS1 -- constructing the manager opens no database.
  * MS2 -- the schema is BUILT on the first connection, checked against a
    database that does not exist yet. Against the real file the table is
    already there and this would pass with the build removed.
  * MS3 -- the constructor carries no call that builds it.
  * MS4 -- the builder takes no lock, so a caller already holding it cannot
    deadlock. Stated structurally, because the failure it prevents is a hang
    rather than a wrong answer, and a hang is what a suite reports last.
  * MS5 -- the guard's ledger no longer carries that database.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODULE = REPO / "opti_oignon" / "memory" / "legacy.py"
GUARD = REPO / ".github" / "scripts" / "import_footprint_guard.py"
_DATABASE = "memories.db"

_PROBE = """
import json, os, pathlib, sys, tempfile

opened = []
sys.addaudithook(
    lambda event, args: opened.append(os.path.basename(str(args[0])))
    if event == "sqlite3.connect" and args else None
)

sys.path.insert(0, {repo!r})
from opti_oignon.memory.legacy import MemoryManager

fresh = pathlib.Path(tempfile.mkdtemp()) / "memories.db"
mark = len(opened)
manager = MemoryManager(db_path=fresh)
at_construction = opened[mark:]

tables = []
try:
    conn = manager._get_connection()
    tables = sorted(
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    )
    conn.close()
except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
    tables = ["ERROR: " + type(exc).__name__ + ": " + str(exc)]

sys.stdout.write("<<MS>>" + json.dumps({{
    "at_construction": at_construction,
    "after_connect": opened[mark:],
    "tables": tables,
}}))
"""


def _probe():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(repo=str(REPO))],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert "<<MS>>" in result.stdout, (
        f"the probe did not run: {result.stderr[-900:]}"
    )
    return json.loads(result.stdout.split("<<MS>>", 1)[1])


def _guard():
    loaded, restore = isolate(targets={"_footprint": GUARD})
    return loaded["_footprint"], restore


def _function(name):
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_ms1_constructing_the_manager_opens_no_database():
    seen = _probe()
    assert _DATABASE not in seen["at_construction"], (
        f"construction opened {seen['at_construction']}; the package reaches "
        "this module on every import"
    )


def test_ms2_the_schema_is_built_on_the_first_connection():
    seen = _probe()
    assert _DATABASE in seen["after_connect"], (
        "the first connection must open the database, or the probe has not "
        "exercised the path this contract is about"
    )
    assert "memories" in seen["tables"], (
        f"the schema must be built on first use, got {seen['tables']}"
    )


def test_ms3_the_constructor_does_not_build_the_schema():
    node = _function("__init__")
    assert node is not None, "the constructor was not found"
    offenders = [
        inner.lineno
        for inner in ast.walk(node)
        if isinstance(inner, ast.Call)
        and getattr(inner.func, "attr", None) in {"_init_db", "_create_schema"}
    ]
    assert not offenders, (
        f"the constructor builds the schema at line {offenders}, and this "
        "manager is constructed at module scope"
    )


def test_ms4_the_builder_takes_no_lock():
    node = _function("_create_schema")
    assert node is not None, (
        "the schema builder must exist separately from the method that opens "
        "its own connection, or there is nothing to hand a connection to"
    )
    locks = [
        item.lineno
        for item in ast.walk(node)
        if isinstance(item, ast.With)
        for expr in item.items
        if "lock" in ast.dump(expr.context_expr).lower()
    ]
    assert not locks, (
        f"the builder takes the instance lock at line {locks}; ten methods "
        "in this class hold that lock and then ask for a connection, so the "
        "first of them to run before the schema existed would hang"
    )


def test_ms5_the_guard_ledger_no_longer_carries_that_database():
    guard, restore = _guard()
    try:
        assert _DATABASE not in guard.LEDGER["databases"], (
            "the debt this pays must come off the ledger"
        )
        assert guard.LEDGER["databases"], "the ledger is empty"
    finally:
        restore()
