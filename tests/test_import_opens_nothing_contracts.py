#!/usr/bin/env python3
"""Contracts for the last databases opened by importing the package.

Six remained after the mechanical pass, each needing a decision about what a
feature should do at startup rather than a rule applied. The decisions, and
what each rests on:

  * The plugin discovery pass auto-registered builtin plugins at import.
    Registering is work, not setup, and it belongs to the first caller who
    consults the registry. Deferred, and the builtins are still registered
    when that caller arrives.

  * The authentication manager mints a signing secret in its constructor and
    also built its schema there. The secret never touches the database -- it
    is generated into the configuration in memory -- so deferring the schema
    leaves the secret exactly where it was. Only the database open moves.

  * The signed audit log verified its chain at import. That check runs today
    in every process that imports the package, including tooling that will
    never read the log and where the warning reaches nobody. In the process
    where it matters the log IS used, so the check still runs there. Moved to
    the first connection: it now fires once, where someone is about to rely
    on what it checks.

  * The fingerprint preference store was reached through a coding agent the
    dependency module builds eagerly. Deferring the agent there was the
    shorter change and the wrong one: a caller reads that name and refuses
    when it is None, so a proxy standing in for it would answer a readiness
    question it cannot answer. The open is paid where it happens instead, in
    the store itself, which also pays it for every other caller that builds
    one. That store keeps no connection helper either -- each method opens
    its own under the instance lock -- so it gets one, and the builder it
    calls takes no lock.

  * The humanizer feedback store has no connection helper to ensure a schema
    in, so its three public methods ensure it themselves. Three call sites
    named rather than one seam invented.

  * The sandbox audit log holds its lock while asking for a connection, so it
    gets a builder handed one, as its neighbours did.

One database stays, deliberately, and FN6 pins that it does.

  * FN1 -- importing the package opens nothing but what the ledger carries.
  * FN2 -- the audit chain is still verified, at the first connection.
  * FN3 -- the signing secret is still minted at construction.
  * FN4 -- the builtin plugins are still registered, at first use.
  * FN5 -- the humanizer store builds its schema before it writes.
  * FN6 -- the one remaining entry is the one whose flag depends on the open.
  * FN7 -- the preference store builds its schema before it records.
  * FN8 -- supersedes FN1: importing the package opens nothing at all, and
    the probe shows it can see an open by importing, explicitly, the one
    module that opens its database at its own import.
  * FN9 -- supersedes FN6: the ledger carries no database, and the module
    that kept its open-at-import for its flag still opens at ITS import.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GUARD = REPO / ".github" / "scripts" / "import_footprint_guard.py"

# The one database whose module-level availability flag is false only when
# construction raises, and construction raises because it opens this file.
KEPT = "branches.db"

_PROBE = """
import json, os, pathlib, sys, tempfile

opened = []
sys.addaudithook(
    lambda event, args: opened.append(os.path.basename(str(args[0])))
    if event == "sqlite3.connect" and args else None
)

sys.path.insert(0, {repo!r})
import opti_oignon  # noqa: F401
at_import = sorted(set(opened))

report = {{"at_import": at_import}}
# Capability control for the empty case: the hook must see a database
# when a module that opens one at its own import is imported explicitly.
mark = len(opened)
import opti_oignon.conversation_branches  # noqa: F401,E402
report["on_module_import"] = sorted(set(opened[mark:]))

# The audit chain is still verified -- at the first connection now.
from opti_oignon.signed_audit_log import SignedAuditLog
root = pathlib.Path(tempfile.mkdtemp())
log = SignedAuditLog(db_path=str(root / "audit_chain.db"))
checked = {{"n": 0}}
original = log._check_integrity_on_init
log._check_integrity_on_init = lambda: checked.__setitem__("n", checked["n"] + 1)
report["audit_at_construction"] = checked["n"]
log._get_conn().close()
report["audit_after_connect"] = checked["n"]
log._check_integrity_on_init = original

# The signing secret is still minted where it was.
from opti_oignon.auth import AuthManager
mark = len(opened)
manager = AuthManager(db_path=str(root / "auth.db"))
report["auth_opened_at_construction"] = opened[mark:]
report["auth_has_secret"] = bool(
    manager.config.get("jwt", {{}}).get("secret_key")
)

# The builtins are still registered, once someone asks. Read against the
# real registry this says little -- its file already holds records from
# earlier runs, which is how the first version of this passed with
# discovery removed. So it is also measured on a database that does not
# exist yet, with the flag on and off, which makes the flag the thing
# observed rather than the contents of a file nobody cleaned.
from opti_oignon import plugin_manifest as pm
report["plugins_registered"] = len(pm.plugin_registry._plugins) if pm.plugin_registry else -1
report["singleton_discovers"] = bool(
    getattr(pm.plugin_registry, "_discover_builtins", False)
)
builtins_dir = pathlib.Path(pm.__file__).parent / "plugins"
report["builtins_on_disk"] = len(
    [one for one in builtins_dir.iterdir() if one.is_dir()]
) if builtins_dir.is_dir() else 0

mark = len(opened)
fresh_on = pm.PluginRegistry(
    db_path=root / "fresh_on.db", plugins_dir=root / "pdir",
    discover_builtins=True,
)
report["fresh_opened_at_construction"] = opened[mark:]
report["fresh_with_discovery"] = len(fresh_on._plugins)
fresh_off = pm.PluginRegistry(
    db_path=root / "fresh_off.db", plugins_dir=root / "pdir",
    discover_builtins=False,
)
report["fresh_without_discovery"] = len(fresh_off._plugins)

sys.stdout.write("<<FN>>" + json.dumps(report))
"""


_HUMANIZER_PROBE = """
import json, pathlib, sqlite3, sys, tempfile
sys.path.insert(0, {repo!r})
from opti_oignon.humanizer import HumanizerFeedbackDB
fresh = pathlib.Path(tempfile.mkdtemp()) / "humanizer_feedback.db"
store = HumanizerFeedbackDB(db_path=fresh)
existed = fresh.exists()
stats = store.get_stats()
tables = []
if fresh.exists():
    conn = sqlite3.connect(fresh)
    tables = sorted(
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    )
    conn.close()
sys.stdout.write("<<HU>>" + json.dumps({{
    "at_construction": existed,
    "after_use": fresh.exists(),
    "tables": tables,
    "stats_total": stats.total_ratings,
}}))
"""

_PREFERENCE_PROBE = """
import json, pathlib, sys, tempfile
sys.path.insert(0, {repo!r})
from opti_oignon.session_fingerprint import UserPreferencesStore
fresh = pathlib.Path(tempfile.mkdtemp()) / "fingerprint.db"
store = UserPreferencesStore(db_path=str(fresh))
existed = fresh.exists()
store.record("approve", phase="plan")
sys.stdout.write("<<PR>>" + json.dumps({{
    "at_construction": existed,
    "after_use": fresh.exists(),
    "total": store.total_decisions,
}}))
"""

def _probe():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(repo=str(REPO))],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert "<<FN>>" in result.stdout, (
        f"the probe did not run: {result.stderr[-1200:]}"
    )
    return json.loads(result.stdout.split("<<FN>>", 1)[1])


def _guard():
    loaded, restore = isolate(targets={"_footprint": GUARD})
    return loaded["_footprint"], restore


def test_fn1_the_import_opens_nothing_but_what_the_ledger_carries():
    seen = _probe()
    guard, restore = _guard()
    try:
        extra = sorted(set(seen["at_import"]) - guard.LEDGER["databases"])
        assert not extra, f"opened at import and not on the ledger: {extra}"
        assert seen["at_import"], (
            "the probe reported no database at all, so it has measured "
            "nothing rather than measured a clean import"
        )
    finally:
        restore()


def test_fn2_the_audit_chain_is_verified_at_the_first_connection():
    seen = _probe()
    assert seen["audit_at_construction"] == 0, (
        "construction must verify nothing; the check belongs where someone "
        "is about to rely on what it checks"
    )
    assert seen["audit_after_connect"] >= 1, (
        "the chain must still be verified -- moving the check must not lose "
        "it, and a tampered log that nothing examines is worse than the cost"
    )


def test_fn3_the_signing_secret_is_still_minted_at_construction():
    seen = _probe()
    assert seen["auth_opened_at_construction"] == [], (
        f"construction opened {seen['auth_opened_at_construction']}"
    )
    assert seen["auth_has_secret"] is True, (
        "the secret never touched the database, so deferring the schema must "
        "leave it exactly where it was"
    )


def test_fn4_the_builtin_plugins_are_still_registered():
    seen = _probe()
    assert seen["plugins_registered"] > 0, (
        f"the registry reports {seen['plugins_registered']} plugins; "
        "deferring discovery must move when it runs, not whether it does"
    )
    assert seen["singleton_discovers"] is True, (
        "the module-level registry must be the one that discovers; the "
        "assertion above reads a file that already holds records from "
        "earlier runs, so on its own it would pass with discovery removed"
    )
    assert seen["fresh_opened_at_construction"] == [], (
        f"construction opened {seen['fresh_opened_at_construction']}"
    )
    assert seen["fresh_with_discovery"] == seen["builtins_on_disk"], (
        f"a registry with discovery on and an empty database registered "
        f"{seen['fresh_with_discovery']} of {seen['builtins_on_disk']} "
        "builtin(s) on the first read"
    )
    assert seen["fresh_without_discovery"] == 0, (
        "with discovery off the same empty database must stay empty, or the "
        "flag is not what decides and the assertion above measures nothing"
    )


def test_fn5_the_humanizer_store_builds_before_it_writes():
    probe = _HUMANIZER_PROBE.format(repo=str(REPO))
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        cwd=str(REPO),
    )
    assert "<<HU>>" in result.stdout, f"probe failed: {result.stderr[-800:]}"
    seen = json.loads(result.stdout.split("<<HU>>", 1)[1])
    assert seen["at_construction"] is False, (
        "construction created the database file"
    )
    assert seen["after_use"] is True, (
        "a read must still build the schema it reads from"
    )
    # The file existing proves nothing: opening a path creates it. Its own
    # blade found that out -- removing the ensure call left this contract
    # green. What the read must leave behind are the tables.
    assert set(seen["tables"]) >= {"comparisons", "ratings"}, (
        f"a read must build the schema it reads from, got {seen['tables']}"
    )
    assert seen["stats_total"] == 0, (
        "the read must return its empty aggregate rather than swallow a "
        "missing table into the same answer"
    )


def test_fn6_the_one_kept_entry_is_the_one_that_cannot_move():
    guard, restore = _guard()
    try:
        assert guard.LEDGER["databases"] == frozenset({KEPT}), (
            f"the ledger should carry {KEPT} alone, got "
            f"{sorted(guard.LEDGER['databases'])}"
        )
    finally:
        restore()


def test_fn8_the_import_opens_nothing_and_the_probe_can_still_see_an_open():
    seen = _probe()
    guard, restore = _guard()
    try:
        assert seen["at_import"] == [], f"opened by importing the package: {seen['at_import']}"
        assert guard.LEDGER["databases"] == frozenset(), "the ledger carries no database any more"
        assert KEPT in seen["on_module_import"], (
            "control: importing the module that opens its database at its own "
            "import is seen by the hook, so an empty list at package import is "
            "a measurement and not a blind probe"
        )
    finally:
        restore()


def test_fn9_the_kept_open_moved_with_its_module_not_with_the_package():
    seen = _probe()
    assert seen["on_module_import"] == [KEPT], (
        "the flag-bearing module still opens exactly its own database when it "
        "is imported; the package import simply no longer reaches it"
    )
    assert KEPT not in seen["at_import"]


def test_fn7_the_preference_store_builds_before_it_records():
    probe = _PREFERENCE_PROBE.format(repo=str(REPO))
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        cwd=str(REPO),
    )
    assert "<<PR>>" in result.stdout, f"probe failed: {result.stderr[-800:]}"
    seen = json.loads(result.stdout.split("<<PR>>", 1)[1])
    assert seen["at_construction"] is False, (
        "construction created the database file"
    )
    assert seen["after_use"] is True, "recording built nothing"
    assert seen["total"] == 1, (
        f"the decision was not recorded ({seen['total']}); this store swallows "
        "every database error into a warning, so a schema that never got "
        "built would otherwise look exactly like a store with nothing in it"
    )
