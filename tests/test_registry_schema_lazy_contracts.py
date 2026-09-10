#!/usr/bin/env python3
"""Contracts for three more stores that built their schema on construction.

Each is constructed at module scope, so importing the package opened three
more databases with encryption not enforced before any caller had asked for
anything. None of the three takes an instance lock, so the straightforward
remedy is safe: the schema is ensured at the first connection, through the
helper each already uses.

The routing store is the one worth naming. Its module also defers the
classifier's dependency, which took eleven hundred modules off the import;
this takes the database its history lives in off as well, so nothing about
that feature is paid for until someone routes something.

  * RG1 -- constructing any of the three opens no database.
  * RG2 -- each builds its schema on the first connection, checked against a
    database that does not exist yet.
  * RG3 -- no constructor carries a call that builds it.
  * RG4 -- the ledger loses the one whose file has a single owner, and
    keeps the two that a second eager module still opens.
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
    ("learned_router", "LearnedRouter", "_get_conn", "learned_router.db"),
    ("plugin_manifest", "PluginRegistry", "_get_conn", "plugins.db"),
    ("user_isolation", "UserSettingsStore", "_get_conn", "auth.db"),
)

_PROBE = """
import importlib, inspect, json, os, pathlib, sys, tempfile

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
    # Some of these name the argument db_path, some just take it first.
    names = list(inspect.signature(cls.__init__).parameters)[1:]
    mark = len(opened)
    if "db_path" in names:
        store = cls(db_path=fresh)
    else:
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

sys.stdout.write("<<RG>>" + json.dumps(report))
"""


def _probe():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(repo=str(REPO), stores=STORES)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert "<<RG>>" in result.stdout, (
        f"the probe did not run: {result.stderr[-1200:]}"
    )
    return json.loads(result.stdout.split("<<RG>>", 1)[1])


def _guard():
    loaded, restore = isolate(targets={"_footprint": GUARD})
    return loaded["_footprint"], restore


def test_rg1_constructing_any_of_them_opens_no_database():
    seen = _probe()
    offenders = {
        name: report["at_construction"]
        for name, report in seen.items() if report["at_construction"]
    }
    assert not offenders, (
        f"construction opened databases: {offenders}; each is constructed at "
        "module scope, so that cost lands on every import"
    )


def test_rg2_each_builds_its_schema_on_the_first_connection():
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


def test_rg3_no_constructor_builds_the_schema():
    offenders = {}
    for module_name, class_name, _helper, _database in STORES:
        tree = ast.parse(
            (REPO / "opti_oignon" / f"{module_name}.py").read_text(
                encoding="utf-8")
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            for inner in node.body:
                if not isinstance(inner, ast.FunctionDef) or inner.name != "__init__":
                    continue
                lines = [
                    call.lineno
                    for call in ast.walk(inner)
                    if isinstance(call, ast.Call)
                    and getattr(call.func, "attr", None) == "_init_db"
                ]
                if lines:
                    offenders[module_name] = lines
    assert not offenders, f"constructors still build their schema: {offenders}"


# Of the three files, only this one has a single owner. The other two are
# each opened by a second module that still builds eagerly: the plugin file
# by a discovery pass that auto-registers builtins at import, and the
# authentication file by the manager that also mints a signing secret in its
# constructor. Deferring either is a decision about what those features do at
# startup, not a mechanical move, so the ledger keeps them.
SINGLE_OWNER = "learned_router.db"


def test_rg4_the_ledger_loses_the_one_with_a_single_owner():
    guard, restore = _guard()
    try:
        assert SINGLE_OWNER not in guard.LEDGER["databases"], (
            f"{SINGLE_OWNER} no longer opens at import, so it must come off"
        )
        for _module, _cls, _helper, database in STORES:
            if database == SINGLE_OWNER:
                continue
            assert database in guard.LEDGER["databases"], (
                f"{database} is still opened at import by a second owner, so "
                "it must stay on the ledger until that one is paid too"
            )
    finally:
        restore()
