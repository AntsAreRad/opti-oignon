#!/usr/bin/env python3
"""Contracts for five stores that built their schema on construction.

Each is constructed at module scope, so importing the package opened five
more databases with encryption not enforced before any caller had asked for
anything. None of the five takes an instance lock, which is what makes the
straightforward remedy safe here and is why they are done together: the
schema is ensured on the first connection, through the single helper each
already uses.

They are checked as a group and by measurement rather than by reading. A
per-module contract would say the same thing five times; what matters is that
none of the five opens anything on construction and that each still builds
what its callers depend on when someone finally asks.

  * ST1 -- constructing any of the five opens no database.
  * ST2 -- each builds its schema on the first connection, checked against a
    database that does not exist yet. Against the real files the tables are
    already there and this would pass with every build removed.
  * ST3 -- none of the five constructors carries a call that builds it.
  * ST4 -- the guard's ledger no longer carries any of the five.
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

# module, class, connection helper, database, a table its callers depend on
STORES = (
    ("feedback", "FeedbackStore", "_get_conn", "feedback.db"),
    ("analytics", "PerformanceTracker", "_get_conn", "analytics.db"),
    ("performance_monitor", "PerformanceMonitor", "_get_conn",
     "performance_metrics.db"),
    ("sync_queue", "SyncQueue", "_get_connection", "sync_queue.db"),
    ("projects", "ProjectStore", "_get_conn", "projects.db"),
)

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

sys.stdout.write("<<ST>>" + json.dumps(report))
"""


def _probe():
    result = subprocess.run(
        [sys.executable, "-c",
         _PROBE.format(repo=str(REPO), stores=STORES)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert "<<ST>>" in result.stdout, (
        f"the probe did not run: {result.stderr[-1200:]}"
    )
    return json.loads(result.stdout.split("<<ST>>", 1)[1])


def _guard():
    loaded, restore = isolate(targets={"_footprint": GUARD})
    return loaded["_footprint"], restore


def test_st1_constructing_any_of_them_opens_no_database():
    seen = _probe()
    offenders = {
        name: report["at_construction"]
        for name, report in seen.items() if report["at_construction"]
    }
    assert not offenders, (
        f"construction opened databases: {offenders}; each of these is "
        "constructed at module scope, so that cost lands on every import"
    )


def test_st2_each_builds_its_schema_on_the_first_connection():
    seen = _probe()
    for module_name, _cls, _helper, database in STORES:
        report = seen[module_name]
        assert report["after"], (
            f"{module_name}: the first connection opened nothing, so the "
            "probe has not exercised the path this contract is about"
        )
        assert report["tables"] and not any(
            one.startswith("ERROR") for one in report["tables"]
        ), f"{module_name}: {report['tables']}"


def test_st3_no_constructor_builds_the_schema():
    offenders = {}
    for module_name, class_name, _helper, _database in STORES:
        tree = ast.parse(
            (REPO / "opti_oignon" / f"{module_name}.py").read_text(
                encoding="utf-8")
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
                continue
            lines = [
                inner.lineno
                for inner in ast.walk(node)
                if isinstance(inner, ast.Call)
                and getattr(inner.func, "attr", None) == "_init_db"
            ]
            if lines:
                offenders[module_name] = lines
    assert not offenders, (
        f"constructors still build their schema: {offenders}"
    )


def test_st4_the_guard_ledger_no_longer_carries_any_of_them():
    guard, restore = _guard()
    try:
        for _module, _cls, _helper, database in STORES:
            assert database not in guard.LEDGER["databases"], (
                f"{database} no longer opens at import, so it must come off"
            )
        assert guard.LEDGER["databases"], "the ledger is empty"
    finally:
        restore()


def test_st5_the_ledger_is_empty_and_still_read():
    """Supersedes st4: its control asked the ledger to be
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
