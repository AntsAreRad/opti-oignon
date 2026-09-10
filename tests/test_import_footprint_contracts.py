#!/usr/bin/env python3
"""Contracts for the guard that pins what importing the package costs.

A written rule already says it: importing the package creates no database,
starts no thread, touches no network. Measured, the first clause is false --
the import opens twenty databases through singletons built at module scope,
with encryption not enforced on any of them. In the strict mode the first
would raise and take the import down with it.

So the guard cannot simply refuse the tree it ships with, or it would be red
from birth and get silenced. It carries a ledger of the debt that predates
it, exactly as the isolation guard does: the ledger MAY ONLY SHRINK, so
nothing new can appear while the existing debt is paid down.

What is pinned and what is not, decided by measurement rather than symmetry:

  * The module count is byte-stable across runs. It is a ceiling.
  * Wall time is not: 14.7 per cent spread on an idle machine, and two
    independent sets of readings that do not overlap. It is RECORDED, never
    enforced -- a ceiling that moves with the machine is the defect this
    repository named h-b10 and removed a fortnight ago.
  * Resident memory is machine-only for the same reason, at a smaller
    amplitude. Recorded.

  * FP1 -- an observation matching the ledger exactly is accepted.
  * FP2 -- a database the ledger does not carry is refused, by name.
  * FP3 -- a file created in the working directory is refused, by name.
  * FP4 -- a heavy module the ledger does not carry is refused, by name.
  * FP5 -- a second thread is refused. This one is TRUE on the tree today,
    so it is born green and only its blade proves it.
  * FP6 -- a module count above the ceiling is refused.
  * FP7 -- a ledger entry the import no longer reaches is reported stale, so
    the debt count cannot drift from the debt.
  * FP8 -- an empty observation is REFUSED, not accepted. A probe that
    measured nothing must never read as a tree that owes nothing.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GUARD = REPO / ".github" / "scripts" / "import_footprint_guard.py"
_UNDER_CONTRACT = "_import_footprint_guard_under_contract"


def _guard():
    """Load the guard through the shared window; returns (module, restore)."""
    loaded, restore = isolate(targets={_UNDER_CONTRACT: GUARD})
    return loaded[_UNDER_CONTRACT], restore


def _observation(guard, **overrides):
    """An observation identical to the ledger, before any override."""
    seen = {
        "databases": sorted(guard.LEDGER["databases"]),
        "files": sorted(guard.LEDGER["files"]),
        "heavy": sorted(guard.LEDGER["heavy"]),
        "threads": 1,
        "modules": guard.MODULE_CEILING - 1,
        "import_ms": 2800.0,
        "rss_mib": 291.0,
    }
    seen.update(overrides)
    return seen


def test_fp1_an_observation_matching_the_ledger_is_accepted():
    guard, restore = _guard()
    try:
        assert guard.verdict(_observation(guard)) == [], (
            "the tree as it stands is the debt the ledger records; refusing "
            "it would leave no way to ship the guard at all"
        )
    finally:
        restore()


def test_fp2_a_database_outside_the_ledger_is_refused():
    guard, restore = _guard()
    try:
        seen = _observation(guard)
        seen["databases"] = sorted(set(seen["databases"]) | {"newcomer.db"})
        reasons = guard.verdict(seen)
        assert reasons, "a twenty-first database at import must be refused"
        assert any("newcomer.db" in one for one in reasons), reasons
    finally:
        restore()


def test_fp3_a_file_created_in_the_working_directory_is_refused():
    guard, restore = _guard()
    try:
        seen = _observation(guard)
        seen["files"] = sorted(set(seen["files"]) | {"stray.log"})
        reasons = guard.verdict(seen)
        assert reasons, "an import that writes into the caller's directory "
        assert any("stray.log" in one for one in reasons), reasons
    finally:
        restore()


def test_fp4_a_heavy_module_outside_the_ledger_is_refused():
    guard, restore = _guard()
    try:
        seen = _observation(guard)
        seen["heavy"] = sorted(set(seen["heavy"]) | {"torch"})
        reasons = guard.verdict(seen)
        assert reasons, "a new heavy dependency loaded at import"
        assert any("torch" in one for one in reasons), reasons
    finally:
        restore()


def test_fp5_a_second_thread_is_refused():
    """True on the tree today, so this is born green; its blade proves it."""
    guard, restore = _guard()
    try:
        reasons = guard.verdict(_observation(guard, threads=2))
        assert reasons, "a thread started at import must be refused"
        assert any("thread" in one for one in reasons), reasons
    finally:
        restore()


def test_fp6_a_module_count_above_the_ceiling_is_refused():
    guard, restore = _guard()
    try:
        seen = _observation(guard, modules=guard.MODULE_CEILING + 1)
        reasons = guard.verdict(seen)
        assert reasons, "the module count is the one figure stable enough to "
        assert any("module" in one for one in reasons), reasons
        # And the ceiling sits above what the tree actually loads, or it
        # would be a ceiling nothing could satisfy.
        assert guard.MODULE_CEILING > 2814, guard.MODULE_CEILING
    finally:
        restore()


def test_fp7_a_ledger_entry_no_longer_reached_is_reported_stale():
    guard, restore = _guard()
    try:
        seen = _observation(guard)
        seen["databases"] = sorted(set(seen["databases"]) - {"conversations.db"})
        stale = guard.stale_entries(seen)
        assert "conversations.db" in " ".join(stale), (
            "a debt that has been paid must come off the ledger, or the "
            "count stops meaning anything"
        )
    finally:
        restore()


def test_fp8_an_empty_observation_is_refused_not_accepted():
    guard, restore = _guard()
    try:
        empty = {"databases": [], "files": [], "heavy": [], "threads": 0,
                 "modules": 0, "import_ms": 0.0, "rss_mib": 0.0}
        reasons = guard.verdict(empty)
        assert reasons, (
            "a probe that measured nothing reports nothing, and nothing "
            "reads exactly like a tree that owes nothing"
        )
        assert any("measured nothing" in one or "no module" in one
                   for one in reasons), reasons
    finally:
        restore()


def test_fp10_an_observation_that_did_not_happen_is_refused():
    """Met for real while writing this guard, not invented as a case.

    The probe first ran with a replaced environment, which dropped the user
    site directory; the package then failed to import for a reason having
    nothing to do with its footprint. Had that read as an empty observation,
    the guard would have reported a tree that owes nothing.
    """
    guard, restore = _guard()
    try:
        reasons = guard.verdict(None)
        assert reasons, "a probe that did not run must never read as clean"
        assert any("could not be run" in one for one in reasons), reasons
    finally:
        restore()


def test_fp9_the_guard_is_shaped_like_its_neighbours():
    text = GUARD.read_text(encoding="utf-8")
    assert "checkpoint_before_apply = True" in text
    assert "Path(__file__).resolve()" in text, (
        "anchored on the file, not on the working directory: the ladder runs "
        "guards from the repository root and CI does not"
    )
    guarded = text.index('if __name__ == "__main__"')
    for call in re.finditer(r"^main\(", text, re.M):
        assert call.start() > guarded, "importing the guard must not run it"
