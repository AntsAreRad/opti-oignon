#!/usr/bin/env python3
"""Contracts for creating the second-factor schema on use, not on import.

The module ended with a bare call that built its database schema, under a
comment saying it initialised on import. It did exactly that: importing the
module opened a database, created tables, ran a migration and committed,
before any caller had asked for anything.

Two costs, and the second is the one that matters. The first is that a
capability nobody called was paid for. The second is that the connection is
opened with encryption not enforced, so on a machine configured to refuse
unencrypted storage the refusal would arrive during an import rather than at
the call it belongs to -- and an import that cannot fail cleanly takes the
whole package down with it.

Ensuring the schema on the first connection costs the same once and nothing
until then, and every one of the module's twenty-one connection sites goes
through the same helper, so there is one place to be right.

  * TF1 -- importing the module opens no database. Measured by importing it
    alone in a subprocess with an audit hook, not asserted about.
  * TF2 -- the schema is BUILT on the first connection, checked against a
    database that does not exist yet. Against the real file the tables are
    already there from an earlier run, and the contract would pass with the
    build removed.
  * TF3 -- the module carries no call at its own scope that builds it.
  * TF4 -- the guard's ledger no longer carries that database.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODULE = REPO / "opti_oignon" / "auth_2fa.py"
GUARD = REPO / ".github" / "scripts" / "import_footprint_guard.py"
_DATABASE = "auth_2fa.db"

_PROBE = """
import importlib.util, json, os, sys

opened = []
sys.addaudithook(
    lambda event, args: opened.append(os.path.basename(str(args[0])))
    if event == "sqlite3.connect" and args else None
)

spec = importlib.util.spec_from_file_location("_tf_under_contract", {path!r})
module = importlib.util.module_from_spec(spec)
sys.modules["_tf_under_contract"] = module
spec.loader.exec_module(module)
at_import = list(opened)

# Point the module at a database that does not exist yet, then ask for a
# connection. Against the real file the tables are already there from an
# earlier run, so this contract would pass with the schema build removed --
# which is exactly what its own blade caught.
import pathlib, tempfile
fresh = pathlib.Path(tempfile.mkdtemp()) / "auth_2fa.db"
module._2FA_DB_PATH = fresh
module._DATA_DIR = fresh.parent

tables = []
try:
    conn = module._get_2fa_conn()
    tables = sorted(
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    )
    conn.close()
except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
    tables = ["ERROR: " + type(exc).__name__ + ": " + str(exc)]

sys.stdout.write("<<TF>>" + json.dumps({{
    "at_import": at_import,
    "after_connect": list(opened),
    "tables": tables,
}}))
"""


def _probe():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(path=str(MODULE))],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert "<<TF>>" in result.stdout, (
        f"the probe did not run: {result.stderr[-900:]}"
    )
    return json.loads(result.stdout.split("<<TF>>", 1)[1])


def _guard():
    loaded, restore = isolate(targets={"_footprint": GUARD})
    return loaded["_footprint"], restore


def test_tf1_importing_the_module_opens_no_database():
    seen = _probe()
    assert _DATABASE not in seen["at_import"], (
        f"importing the module opened {seen['at_import']}; a capability "
        "costs nothing until it is called"
    )


def test_tf2_the_schema_is_there_once_a_connection_is_asked_for():
    seen = _probe()
    assert _DATABASE in seen["after_connect"], (
        "the first connection must open the database, or the probe has not "
        "exercised the path this contract is about"
    )
    assert "totp_config" in seen["tables"], (
        f"the schema must be built on first use, got {seen['tables']}"
    )
    assert "webauthn_credentials" in seen["tables"], seen["tables"]


def test_tf3_no_call_at_module_scope_builds_the_schema():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "_init_2fa_db"
    ]
    assert not offenders, (
        f"a call at module scope runs at import: line {offenders}"
    )


def test_tf4_the_guard_ledger_no_longer_carries_that_database():
    guard, restore = _guard()
    try:
        assert _DATABASE not in guard.LEDGER["databases"], (
            "the debt this pays must come off the ledger, or the count "
            "stops meaning anything"
        )
        # Proven capable: the ledger still carries the ones not yet paid.
        assert guard.LEDGER["databases"], "the ledger is empty"
    finally:
        restore()
