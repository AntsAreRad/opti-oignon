#!/usr/bin/env python3
"""Contracts for three history stores that built their schema on construction.

Each is constructed at module scope, so importing the package opened three
more databases with encryption not enforced before any caller had asked for
anything.

Both carry an instance lock, but not in the same place, and that difference
decides the remedy. One takes the lock inside the method that builds the
schema, so building from the connection helper would reacquire it and hang;
it gets a builder that is handed a connection and takes no lock. The other
does not take it there, so the straightforward form is safe for it.

A THIRD store was attempted here and then reverted, which is why this file
names two. Its module-level availability flag is set to false only when
constructing the manager raises, and constructing it raised because it opened
the database. Deferring that meant a broken connection factory was no longer
detected and the flag stayed true -- a fail-secure property lost, not merely
a contract encoded around the old timing. Not every database opened at import
is waste; that one is a readiness check, and changing what its flag means is
a decision about the feature rather than a mechanical deferral.

  * HS1 -- constructing either of them opens no database.
  * HS2 -- each builds its schema on the first connection, checked against a
    database that does not exist yet.
  * HS3 -- no constructor carries a call that builds it.
  * HS4 -- for the two whose schema build sat behind the lock, the builder
    now takes none. Stated structurally: the failure it prevents is a hang,
    and a hang is what a suite reports last.
  * HS5 -- the guard's ledger no longer carries either.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GUARD = REPO / ".github" / "scripts" / "import_footprint_guard.py"

STORES = (
    ("benchmark_history", "BenchmarkHistory", "_get_conn",
     "benchmark_history.db"),
    ("coding_history", "CodingHistoryStore", "_get_conn", "coding_history.db"),
)

# The one that built its schema behind the lock.
HANDED_A_CONNECTION = ("benchmark_history",)

_PROBE = """
import importlib, json, os, pathlib, sys, tempfile

opened = []
sys.addaudithook(
    lambda event, args: opened.append(os.path.basename(str(args[0])))
    if event == "sqlite3.connect" and args else None
)

sys.path.insert(0, {repo!r})
report = {{}}
root = pathlib.Path(tempfile.mkdtemp())

for module_name, class_name, helper, database in {stores!r}:
    module = importlib.import_module("opti_oignon." + module_name)
    cls = getattr(module, class_name)
    fresh = root / (module_name + "_" + database)
    mark = len(opened)
    try:
        store = cls(db_path=fresh)
    except TypeError:
        store = cls(str(fresh))
    at_construction = opened[mark:]
    tables = []
    try:
        conn = getattr(store, helper)()
        tables = sorted(
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        )
        conn.close()
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        tables = ["ERROR: " + type(exc).__name__ + ": " + str(exc)]
    report[module_name] = {{
        "at_construction": at_construction,
        "after": opened[mark:],
        "tables": tables,
    }}

sys.stdout.write("<<HS>>" + json.dumps(report))
"""


def _probe():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(repo=str(REPO), stores=STORES)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert "<<HS>>" in result.stdout, (
        f"the probe did not run: {result.stderr[-1200:]}"
    )
    return json.loads(result.stdout.split("<<HS>>", 1)[1])


def _guard():
    loaded, restore = isolate(targets={"_footprint": GUARD})
    return loaded["_footprint"], restore


def _function(module_name, name):
    tree = ast.parse(
        (REPO / "opti_oignon" / f"{module_name}.py").read_text(encoding="utf-8")
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_hs1_constructing_any_of_them_opens_no_database():
    seen = _probe()
    offenders = {
        name: report["at_construction"]
        for name, report in seen.items() if report["at_construction"]
    }
    assert not offenders, (
        f"construction opened databases: {offenders}; each is constructed at "
        "module scope, so that cost lands on every import"
    )


def test_hs2_each_builds_its_schema_on_the_first_connection():
    seen = _probe()
    for module_name, _cls, _helper, _database in STORES:
        report = seen[module_name]
        assert report["after"], (
            f"{module_name}: the first connection opened nothing, so the "
            "probe has not exercised the path this contract is about"
        )
        assert report["tables"] and not any(
            one.startswith("ERROR") for one in report["tables"]
        ), f"{module_name}: {report['tables']}"


def test_hs3_no_constructor_builds_the_schema():
    offenders = {}
    for module_name, _cls, _helper, _database in STORES:
        node = _function(module_name, "__init__")
        if node is None:
            continue
        lines = [
            inner.lineno
            for inner in ast.walk(node)
            if isinstance(inner, ast.Call)
            and getattr(inner.func, "attr", None) in {"_init_db", "_create_schema"}
        ]
        if lines:
            offenders[module_name] = lines
    assert not offenders, f"constructors still build their schema: {offenders}"


def test_hs4_the_handed_builders_take_no_lock():
    for module_name in HANDED_A_CONNECTION:
        node = _function(module_name, "_create_schema")
        assert node is not None, (
            f"{module_name}: the builder must exist separately from the "
            "method that opens its own connection"
        )
        locks = [
            item.lineno
            for item in ast.walk(node)
            if isinstance(item, ast.With)
            for expr in item.items
            if "lock" in ast.dump(expr.context_expr).lower()
        ]
        assert not locks, (
            f"{module_name}: the builder takes the instance lock at line "
            f"{locks}; callers hold it and then ask for a connection, so the "
            "first to run before the schema existed would hang"
        )


def test_hs5_the_guard_ledger_no_longer_carries_any_of_them():
    guard, restore = _guard()
    try:
        for _module, _cls, _helper, database in STORES:
            assert database not in guard.LEDGER["databases"], (
                f"{database} no longer opens at import, so it must come off"
            )
        assert guard.LEDGER["databases"], "the ledger is empty"
    finally:
        restore()


def test_hs6_the_ledger_is_empty_and_still_read():
    """Supersedes hs5: its control asked the ledger to be
    non-empty, and the ledger is empty since the package import opens nothing."""
    guard, restore = _guard()
    try:
        for _module, _cls, _helper, database in STORES:
            assert database not in guard.LEDGER["databases"], f"{database} no longer opens at import, so it must come off"
        assert guard.LEDGER["databases"] == frozenset(), "the import opens no database: the ledger says so"
        seen = {"databases": [], "files": [], "heavy": [], "threads": 1,
                "modules": 85, "import_ms": 1.0, "rss_mib": 13.0}
        # Proven capable: the ledger is empty because the debt is paid, not
        # because the guard stopped reading it. An entry put back on it
        # that the observation no longer reaches is reported stale.
        guard.LEDGER = {**guard.LEDGER, "databases": frozenset({STORES[0][3]})}
        assert STORES[0][3] in " ".join(guard.stale_entries(seen)), "the ledger is read against the observation"
    finally:
        restore()
