#!/usr/bin/env python3
"""Contracts for building the conversation schema on use, not on construction.

The conversation manager is constructed at module scope, and its constructor
opened a database and created its schema. Since the executor imports this
module and the package imports the executor, importing the package opened a
database with encryption not enforced, before any caller had asked for
anything. That is the largest of the remaining nineteen and the one in the
package's own import chain.

The same shape as the second factor's, and deliberately so: the flag is set
before the initialiser runs rather than after, because the initialiser asks
for its own connection through the same helper and would otherwise recur --
and here it would deadlock rather than recur, since the lock it holds is not
reentrant.

  * CS1 -- constructing the manager opens no database.
  * CS2 -- the schema is BUILT on the first connection, checked against a
    database that does not exist yet. Against the real file the tables are
    already there, and this would pass with the build removed.
  * CS3 -- the constructor carries no call that builds the schema.
  * CS4 -- the guard's ledger no longer carries that database.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODULE = REPO / "opti_oignon" / "conversation.py"
GUARD = REPO / ".github" / "scripts" / "import_footprint_guard.py"
_DATABASE = "conversations.db"

_PROBE = """
import json, os, pathlib, sys, tempfile

opened = []
sys.addaudithook(
    lambda event, args: opened.append(os.path.basename(str(args[0])))
    if event == "sqlite3.connect" and args else None
)

sys.path.insert(0, {repo!r})
from opti_oignon.conversation import ConversationManager

fresh = pathlib.Path(tempfile.mkdtemp()) / "conversations.db"
manager = ConversationManager(db_path=fresh)
at_construction = list(opened)

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

sys.stdout.write("<<CS>>" + json.dumps({{
    "at_construction": at_construction,
    "after_connect": list(opened),
    "tables": tables,
}}))
"""


def _probe():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(repo=str(REPO))],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert "<<CS>>" in result.stdout, (
        f"the probe did not run: {result.stderr[-900:]}"
    )
    return json.loads(result.stdout.split("<<CS>>", 1)[1])


def _guard():
    loaded, restore = isolate(targets={"_footprint": GUARD})
    return loaded["_footprint"], restore


def test_cs1_constructing_the_manager_opens_no_database():
    seen = _probe()
    assert _DATABASE not in seen["at_construction"], (
        f"constructing the manager opened {seen['at_construction']}; the "
        "package imports this module, so that cost is paid on every import"
    )


def test_cs2_the_schema_is_built_on_the_first_connection():
    seen = _probe()
    assert _DATABASE in seen["after_connect"], (
        "the first connection must open the database, or the probe has not "
        "exercised the path this contract is about"
    )
    assert "conversations" in seen["tables"], (
        f"the schema must be built on first use, got {seen['tables']}"
    )


def test_cs3_the_constructor_does_not_build_the_schema():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call)
                    and getattr(inner.func, "attr", None) == "_init_db"):
                offenders.append(inner.lineno)
    assert not offenders, (
        f"the constructor builds the schema at line {offenders}, and this "
        "manager is constructed at module scope"
    )


def test_cs4_the_guard_ledger_no_longer_carries_that_database():
    guard, restore = _guard()
    try:
        assert _DATABASE not in guard.LEDGER["databases"], (
            "the debt this pays must come off the ledger"
        )
        assert guard.LEDGER["databases"], "the ledger is empty"
    finally:
        restore()


def test_cs5_the_ledger_is_empty_and_still_read():
    """Supersedes cs4: its control asked the ledger to be
    non-empty, and the ledger is empty since the package import opens nothing."""
    guard, restore = _guard()
    try:
        assert _DATABASE not in guard.LEDGER["databases"], "the debt this pays must come off the ledger"
        assert guard.LEDGER["databases"] == frozenset(), "the import opens no database: the ledger says so"
        seen = {"databases": [], "files": [], "heavy": [], "threads": 1,
                "modules": 85, "import_ms": 1.0, "rss_mib": 13.0}
        # Proven capable: the ledger is empty because the debt is paid, not
        # because the guard stopped reading it. An entry put back on it
        # that the observation no longer reaches is reported stale.
        guard.LEDGER = {**guard.LEDGER, "databases": frozenset({_DATABASE})}
        assert _DATABASE in " ".join(guard.stale_entries(seen)), "the ledger is read against the observation"
    finally:
        restore()
