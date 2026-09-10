#!/usr/bin/env python3
"""Contracts for building the caches' schemas on use, not on construction.

Both caches are constructed at module scope and both built their schema in
the constructor, so importing the package opened two more databases with
encryption not enforced before any caller had asked for anything.

Neither takes an instance lock, which is what makes them safe to change the
straightforward way -- and it is why they are done together and separately
from the conversation manager, whose fifteen lock sites made the same shape
deadlock.

One asymmetry is preserved deliberately. The semantic cache's connection
helper may return ``None``: the storage layer can be missing, or the enforced
encryption posture can have no working cipher, and the cache then degrades to
a miss rather than failing. Building a schema on first use must respect that
and do nothing when there is no connection to build on.

  * CL1 -- constructing the response cache opens no database.
  * CL2 -- its schema is built on the first connection, checked against a
    database that does not exist yet.
  * CL3 -- constructing the semantic cache opens no database.
  * CL4 -- its schema is built on the first connection.
  * CL5 -- a semantic cache whose storage layer is unavailable still
    constructs, still degrades, and builds nothing.
  * CL6 -- the guard's ledger no longer carries either database.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GUARD = REPO / ".github" / "scripts" / "import_footprint_guard.py"

_PROBE = """
import json, os, pathlib, sys, tempfile

opened = []
sys.addaudithook(
    lambda event, args: opened.append(os.path.basename(str(args[0])))
    if event == "sqlite3.connect" and args else None
)

sys.path.insert(0, {repo!r})
from opti_oignon.response_cache import ResponseCache
from opti_oignon.semantic_cache import SemanticCache

root = pathlib.Path(tempfile.mkdtemp())
report = {{}}

# Sliced, not filtered by membership: a database opened earlier in this
# process would otherwise vanish from the reading and make a constructor
# that opens one look innocent.
mark = len(opened)
rc = ResponseCache(db_path=root / "response_cache.db")
report["rc_at_construction"] = opened[mark:]
conn = rc._get_connection()
report["rc_tables"] = sorted(
    row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
)
conn.close()
report["rc_after"] = list(opened)

mark = len(opened)
sc = SemanticCache(db_path=root / "semantic_cache.db")
report["sc_at_construction"] = opened[mark:]
conn = sc._get_connection()
report["sc_none"] = conn is None
report["sc_tables"] = [] if conn is None else sorted(
    row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
)
if conn is not None:
    conn.close()

# The degraded posture: no storage layer at all.
module = sys.modules["opti_oignon.semantic_cache"]
saved, module._safe_connect = module._safe_connect, None
try:
    degraded = SemanticCache(db_path=root / "degraded.db")
    report["degraded_constructs"] = True
    report["degraded_connection"] = degraded._get_connection() is None
except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
    report["degraded_constructs"] = "ERROR: " + type(exc).__name__ + ": " + str(exc)
    report["degraded_connection"] = None
finally:
    module._safe_connect = saved

sys.stdout.write("<<CL>>" + json.dumps(report))
"""


def _probe():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(repo=str(REPO))],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert "<<CL>>" in result.stdout, (
        f"the probe did not run: {result.stderr[-900:]}"
    )
    return json.loads(result.stdout.split("<<CL>>", 1)[1])


def _guard():
    loaded, restore = isolate(targets={"_footprint": GUARD})
    return loaded["_footprint"], restore


def test_cl1_constructing_the_response_cache_opens_no_database():
    seen = _probe()
    assert "response_cache.db" not in seen["rc_at_construction"], (
        f"construction opened {seen['rc_at_construction']}"
    )


def test_cl2_the_response_cache_builds_its_schema_on_first_connection():
    seen = _probe()
    assert "response_cache.db" in seen["rc_after"], (
        "the first connection must open the database, or the probe has not "
        "exercised the path this contract is about"
    )
    assert "response_cache" in seen["rc_tables"], seen["rc_tables"]


def test_cl3_constructing_the_semantic_cache_opens_no_database():
    seen = _probe()
    assert seen["sc_at_construction"] == [], (
        f"construction opened {seen['sc_at_construction']}"
    )


def test_cl4_the_semantic_cache_builds_its_schema_on_first_connection():
    seen = _probe()
    assert seen["sc_none"] is False, (
        "the probe got no connection at all, so it has not exercised the "
        "path this contract is about"
    )
    assert seen["sc_tables"], seen["sc_tables"]


def test_cl5_a_degraded_semantic_cache_builds_nothing_and_still_works():
    seen = _probe()
    assert seen["degraded_constructs"] is True, (
        f"the cache must construct without a storage layer: "
        f"{seen['degraded_constructs']}"
    )
    assert seen["degraded_connection"] is True, (
        "with no storage layer the connection is None and the cache degrades "
        "to a miss; building a schema on first use must not change that"
    )


def test_cl6_the_guard_ledger_no_longer_carries_either():
    guard, restore = _guard()
    try:
        for name in ("response_cache.db", "semantic_cache.db"):
            assert name not in guard.LEDGER["databases"], (
                f"{name} no longer opens at import, so it must come off"
            )
        assert guard.LEDGER["databases"], "the ledger is empty"
    finally:
        restore()
