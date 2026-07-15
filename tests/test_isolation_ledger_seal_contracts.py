#!/usr/bin/env python3
"""The ledger is a SEAL, not a list of names.

A ratchet that only counts is a ratchet on the count. The guard used to skip a
debted suite whole -- ``if name in LEDGER: continue`` -- so a suite carrying an
unsound window could absorb new contracts for as long as anyone cared to write
them, and the debt would still read the same figure. "May only shrink" was true
of the number and said nothing whatever about the estate.

What the debt actually IS, measured rather than assumed. Most owed suites
manufacture a window that closes ONE of the two routes into the package: the
cache key is evicted, and the finder is not guarded. Under an editable install
the finder answers on the NAME and ignores the stand-in parent's empty path, so
the real module loads behind the test's back. This is not a theory. Real project
modules resolve inside those windows on every run today -- the module that
decides the security mode among them. No contract's verdict happens to depend on
it, which is luck. Luck is not a seal.

So the ledger carries a digest of every suite it owes for. An owed suite that
changes by one line no longer matches its seal and becomes a violation: touch
it, and you migrate it. The debt is frozen exactly as it was found. It can be
paid. It cannot grow -- not in names, and no longer in lines.

Three questions, three answers, and their domains do not overlap, so none can
cover for another:

  * ``find_violations``          -- a window hand-rolled by a suite nobody owes for.
  * ``find_broken_seals``        -- an owed suite whose bytes moved.
  * ``find_stale_ledger_entries``-- an owed suite that migrated, or vanished.

This suite goes through the shared window itself. A guard whose own contracts
break its rule would be an argument against the rule.
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate  # noqa: E402

_GUARD = REPO / ".github" / "scripts" / "isolation_seal_guard.py"
_TESTS = REPO / "tests"

# A window built the old way: loads a module from its file, stands in a parent
# whose path is empty, and never guards the finder. This is the shape of the
# debt, written out once so the contracts below do not have to describe it.
_CACHE_ONLY_WINDOW = (
    "import importlib.util\n"
    "spec = importlib.util.spec_from_file_location('x', '/tmp/x.py')\n"
)

_SHARED_WINDOW = "from _isolation import isolate, source\n"


def _load():
    loaded, restore = isolate(
        targets={"isolation_seal_guard_under_test": _GUARD},
    )
    return loaded["isolation_seal_guard_under_test"], restore


def _real_files():
    """Every test suite, exactly as the guard's own ``main`` reads them."""
    return [
        (p.name, p.read_text(encoding="utf-8", errors="ignore"))
        for p in sorted(_TESTS.glob("test_*.py"))
    ]


def _an_owed_name(guard):
    """One name the ledger owes for, taken from the ledger itself."""
    return sorted(guard.LEDGER)[0]


# ---------------------------------------------------------------------------
# The ledger is a mapping, and every name it owes for carries a digest
# ---------------------------------------------------------------------------


def test_l1_every_owed_name_carries_a_seal():
    guard, restore = _load()
    try:
        assert guard.LEDGER, "the ledger is empty and the debt is not zero"
        for name, sealed in guard.LEDGER.items():
            assert isinstance(sealed, str) and len(sealed) == 64, (
                f"{name} is owed for and carries no digest. A name without a "
                f"seal is the old ledger wearing a new shape: it tolerates the "
                f"file whatever the file becomes."
            )
            int(sealed, 16)
    finally:
        restore()


def test_l2_the_digest_is_taken_on_the_text_the_guard_reads():
    guard, restore = _load()
    try:
        text = "any suite\n"
        assert guard.digest(text) == hashlib.sha256(text.encode()).hexdigest()
    finally:
        restore()


# ---------------------------------------------------------------------------
# The three questions
# ---------------------------------------------------------------------------


def test_l3_a_window_nobody_owes_for_is_a_violation():
    guard, restore = _load()
    try:
        files = [("test_brand_new_contracts.py", _CACHE_ONLY_WINDOW)]
        assert guard.find_violations(files) == ["test_brand_new_contracts.py"]
    finally:
        restore()


def test_l4_an_owed_suite_that_has_not_moved_is_tolerated():
    guard, restore = _load()
    try:
        name = _an_owed_name(guard)
        text = (_TESTS / name).read_text(encoding="utf-8", errors="ignore")
        files = [(name, text)]
        assert guard.find_violations(files) == []
        assert guard.find_broken_seals(files) == [], (
            "an owed suite whose bytes have not moved is the debt as it was "
            "found, and the ratchet must not fire on the status quo"
        )
    finally:
        restore()


def test_l5_an_owed_suite_that_gained_a_line_is_a_broken_seal():
    guard, restore = _load()
    try:
        name = _an_owed_name(guard)
        text = (_TESTS / name).read_text(encoding="utf-8", errors="ignore")
        grown = text + "\n\ndef test_one_more_contract():\n    assert True\n"
        assert guard.find_broken_seals([(name, grown)]) == [name], (
            "THE defect. An owed suite absorbed a contract on a window that "
            "closes only one of the two routes into the package, and the guard "
            "skipped the file whole because the NAME was on the list. The debt "
            "would still read the same number, and the new contract would draw "
            "its verdict from an absence nobody manufactured."
        )
    finally:
        restore()


def test_l6_paying_the_debt_is_the_way_out_and_it_is_not_a_violation():
    guard, restore = _load()
    try:
        name = _an_owed_name(guard)
        migrated = _SHARED_WINDOW + "\n\ndef test_one_more_contract():\n    assert True\n"
        # The whole estate, with exactly one suite migrated. Handing the guard a
        # single file would report the other 120 as vanished, which they are not.
        files = [(n, t) for n, t in _real_files() if n != name] + [(name, migrated)]
        assert guard.find_broken_seals(files) == [], (
            "the suite migrated to the shared window, which is exactly what the "
            "broken seal asks for. Charging it anyway would leave no way to pay."
        )
        assert guard.find_violations(files) == []
        assert guard.find_stale_ledger_entries(files) == [name], (
            "it migrated, so it must come OFF the ledger, or the debt count "
            "stops meaning anything"
        )
    finally:
        restore()


def test_l7_an_owed_suite_that_vanished_is_stale():
    guard, restore = _load()
    try:
        name = _an_owed_name(guard)
        others = [(n, "") for n in sorted(guard.LEDGER) if n != name]
        assert name in guard.find_stale_ledger_entries(others)
    finally:
        restore()


# ---------------------------------------------------------------------------
# The census: the guard must report what it tolerates, not merely count it
# ---------------------------------------------------------------------------


def test_l8_a_window_that_guards_the_finder_is_told_apart_from_one_that_does_not():
    guard, restore = _load()
    try:
        assert guard.guards_the_name_route("sys.meta_path.insert(0, guard)\n") is True
        assert guard.guards_the_name_route(_CACHE_ONLY_WINDOW) is False, (
            "a window that evicts the cache key and never guards the finder "
            "closes nothing under an editable install: the finder answers on "
            "the NAME and the stand-in parent's empty path stops no one"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# The tree as it stands: the seal shipped must be the seal of the estate
# ---------------------------------------------------------------------------


def test_l9_the_estate_satisfies_its_own_guard():
    guard, restore = _load()
    try:
        files = _real_files()
        assert guard.find_violations(files) == []
        assert guard.find_broken_seals(files) == [], (
            "a seal shipped against a tree it does not match is worse than no "
            "seal: it would fire on the next honest edit and be silenced by "
            "regenerating it, which is the habit the ratchet exists to break"
        )
        assert guard.find_stale_ledger_entries(files) == []
    finally:
        restore()


def test_l11_the_entry_point_consults_the_seal_and_refuses(tmp_path, capsys):
    guard, restore = _load()
    try:
        for name, text in _real_files():
            (tmp_path / name).write_text(text, encoding="utf-8")

        owed = _an_owed_name(guard)
        target = tmp_path / owed
        target.write_text(
            target.read_text(encoding="utf-8")
            + "\n\ndef test_one_more_contract():\n    assert True\n",
            encoding="utf-8",
        )

        code = guard.main(["isolation_seal_guard.py", str(tmp_path)])
        out = capsys.readouterr().out

        assert code != 0, (
            "an owed suite grew a contract on a window that closes only the "
            "cache route, and the entry point exited clean. A seal nobody "
            "consults is a comment."
        )
        assert owed in out
    finally:
        restore()


def test_l12_the_entry_point_accepts_the_estate_as_it_stands(tmp_path, capsys):
    guard, restore = _load()
    try:
        for name, text in _real_files():
            (tmp_path / name).write_text(text, encoding="utf-8")

        code = guard.main(["isolation_seal_guard.py", str(tmp_path)])
        out = capsys.readouterr().out

        assert code == 0, (
            "the seal shipped does not match the tree it was taken on. A guard "
            "that is red on arrival gets silenced by regenerating it, which is "
            "the habit the ratchet exists to break."
        )
        assert "owed" in out and "finder" in out, (
            "the census must say what it TOLERATES, not merely how much: a "
            "count that hides the shape of the debt is the comfort this guard "
            "was written against"
        )
    finally:
        restore()


def test_l10_this_suite_obeys_the_rule_it_enforces():
    guard, restore = _load()
    try:
        text = Path(__file__).read_text(encoding="utf-8")
        assert guard.uses_shared_window(text), (
            "the contracts for the ratchet hand-rolled a window of their own. "
            "A rule its own guard breaks is an argument against the rule."
        )
        assert Path(__file__).name not in guard.LEDGER, (
            "a suite born onto the ledger is a debt taken out to buy nothing"
        )
    finally:
        restore()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
