#!/usr/bin/env python3
"""Contracts for the guard that keeps inference inside the registry.

Four sessions made BackendRegistry worth routing through: admission on every
head, a provenance label on every figure, a schema and a tool list as engine
options, a probed VRAM capacity, a sealed recipe. None of it applies to a
request that reaches the client behind the registry's back -- and at the time
this guard was written, twenty-nine modules did, at fifty-one sites.

This guard is a RATCHET in the shape of the isolation-seal guard, and for the
same reason: a ratchet that only counts is a ratchet on the count. Every owed
module carries the digest of its text as the debt was enumerated. An owed
module that changes while still calling the client directly no longer matches
its seal and becomes a violation: touch it, and you migrate it. The debt is
frozen as found, it can be paid, and it cannot grow -- not in modules, and
not in lines.

The census is taken on the syntax tree, never on the text. A docstring that
says "passed straight to ollama.chat" is not a request, and a guard that
charged for prose would be green and red for reasons unrelated to what leaves
the process.

  * RF1 -- every owed name carries a full seal.
  * RF2 -- the census counts calls, not prose, and knows every spelling: the
    bare module, an alias, a name imported from the module, and a client
    constructed from it.
  * RF3 -- a direct caller nobody owes for is a violation.
  * RF4 -- an owed module that has not moved is tolerated.
  * RF5 -- an owed module that gained a line while still calling directly is
    a broken seal.
  * RF6 -- paying the debt is the way out: a module that no longer calls the
    client directly is not a violation, and its entry becomes stale.
  * RF7 -- an owed module that vanished is stale.
  * RF8 -- the estate satisfies its own guard as it stands.
  * RF9 -- the entry point refuses a violation and accepts the estate.
  * RF10 -- the probe is proven able to count: on the real tree it finds the
    debt the ledger records, and that debt is not zero.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window.
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate  # noqa: E402

_GUARD = REPO / ".github" / "scripts" / "registry_funnel_guard.py"
_PACKAGE = REPO / "opti_oignon"

_DIRECT = "import ollama\n\ndef ask():\n    return ollama.chat(model='m', messages=[])\n"
_ALIASED = "import ollama as _o\n\ndef ask():\n    return _o.generate(model='m', prompt='p')\n"
_BARE = "from ollama import chat\n\ndef ask():\n    return chat(model='m', messages=[])\n"
_CLIENT = "import ollama\n\ndef ask():\n    return ollama.Client(host='h').chat(model='m', messages=[])\n"
_PROSE = '"""This module used to call ollama.chat directly."""\n\ndef ask():\n    return None\n'
_ROUTED = (
    "from opti_oignon.inference_backend import get_backend_registry\n\n"
    "def ask():\n    return get_backend_registry().resolve_backend('m').generate('m', [])\n"
)


def _load():
    loaded, restore = isolate(targets={"registry_funnel_guard": _GUARD})
    return loaded["registry_funnel_guard"], restore


def _real_files():
    """The estate as the guard reads it: repo-relative posix paths and text."""
    out = []
    for p in sorted(_PACKAGE.rglob("*.py")):
        rel = p.relative_to(REPO).as_posix()
        out.append((rel, p.read_text(encoding="utf-8", errors="ignore")))
    return out


def _an_owed_name(guard):
    return sorted(guard.LEDGER)[0]


# ---------------------------------------------------------------------------
# RF1 -- every owed name carries a full seal
# ---------------------------------------------------------------------------
def test_rf1_every_owed_name_carries_a_seal():
    guard, restore = _load()
    try:
        assert guard.LEDGER, "the ledger is not empty: the debt is real"
        for name, seal in guard.LEDGER.items():
            assert name.startswith("opti_oignon/") and name.endswith(".py"), (
                f"an owed name is a repo-relative module path: {name!r}"
            )
            assert len(seal) == 64 and int(seal, 16) >= 0, (
                f"the seal of {name} is a full sha256, not a prefix or a count"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF2 -- the census counts calls, not prose, in every spelling
# ---------------------------------------------------------------------------
def test_rf2_the_census_counts_calls_in_every_spelling_and_never_prose():
    guard, restore = _load()
    try:
        assert guard.count_sites(_DIRECT) == 1, "the bare module form is one site"
        assert guard.count_sites(_ALIASED) == 1, "an alias is not a disguise"
        assert guard.count_sites(_BARE) == 1, (
            "a name imported from the module is still the module"
        )
        assert guard.count_sites(_CLIENT) >= 1, (
            "a client constructed from the module is a direct route too"
        )
        assert guard.count_sites(_PROSE) == 0, (
            "a docstring that names the client is not a request"
        )
        assert guard.count_sites(_ROUTED) == 0, (
            "a request through the registry is what the guard exists to allow"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF3 -- a direct caller nobody owes for is a violation
# ---------------------------------------------------------------------------
def test_rf3_a_direct_caller_nobody_owes_for_is_a_violation():
    guard, restore = _load()
    try:
        files = [("opti_oignon/newcomer.py", _DIRECT)]
        assert guard.find_violations(files) == ["opti_oignon/newcomer.py"]
        assert guard.find_violations([("opti_oignon/newcomer.py", _ROUTED)]) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF4 -- an owed module that has not moved is tolerated
# ---------------------------------------------------------------------------
def test_rf4_an_owed_module_that_has_not_moved_is_tolerated():
    guard, restore = _load()
    try:
        files = _real_files()
        assert guard.find_violations(files) == [], (
            "the debt as enumerated is carried, not charged"
        )
        assert guard.find_broken_seals(files) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF5 -- an owed module that gained a line is a broken seal
# ---------------------------------------------------------------------------
def test_rf5_an_owed_module_that_gained_a_line_is_a_broken_seal():
    guard, restore = _load()
    try:
        name = _an_owed_name(guard)
        files = [
            (n, t + "\n# one more line\n" if n == name else t)
            for n, t in _real_files()
        ]
        assert guard.find_broken_seals(files) == [name], (
            "an owed module that changed while still calling directly no "
            "longer matches its seal: touch it, and you migrate it"
        )
        assert name not in guard.find_violations(files), (
            "and it is charged in exactly one place, not two"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF6 -- paying the debt is the way out
# ---------------------------------------------------------------------------
def test_rf6_paying_the_debt_is_not_a_violation_and_makes_the_entry_stale():
    guard, restore = _load()
    try:
        name = _an_owed_name(guard)
        files = [(n, _ROUTED if n == name else t) for n, t in _real_files()]
        assert guard.find_broken_seals(files) == [], (
            "a module that migrated is not broken; charging it would leave "
            "no way to pay"
        )
        assert guard.find_violations(files) == []
        assert guard.find_stale_ledger_entries(files) == [name], (
            "it migrated, so it must come off the ledger, or the debt count "
            "stops meaning anything"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF7 -- an owed module that vanished is stale
# ---------------------------------------------------------------------------
def test_rf7_an_owed_module_that_vanished_is_stale():
    guard, restore = _load()
    try:
        name = _an_owed_name(guard)
        files = [(n, t) for n, t in _real_files() if n != name]
        assert guard.find_stale_ledger_entries(files) == [name]
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF8 -- the estate satisfies its own guard
# ---------------------------------------------------------------------------
def test_rf8_the_estate_satisfies_its_own_guard():
    guard, restore = _load()
    try:
        files = _real_files()
        assert guard.find_violations(files) == []
        assert guard.find_broken_seals(files) == []
        assert guard.find_stale_ledger_entries(files) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF9 -- the entry point refuses a violation and accepts the estate
# ---------------------------------------------------------------------------
def test_rf9_the_entry_point_refuses_a_violation_and_accepts_the_estate(
    tmp_path, capsys,
):
    guard, restore = _load()
    try:
        bad = tmp_path / "opti_oignon"
        bad.mkdir()
        (bad / "newcomer.py").write_text(_DIRECT, encoding="utf-8")
        assert guard.main(["guard", str(tmp_path)]) == 1, (
            "a tree with an unowed direct caller is refused"
        )
        assert "newcomer.py" in capsys.readouterr().out

        assert guard.main(["guard", str(REPO)]) == 0, (
            "the estate as it stands is accepted"
        )
        out = capsys.readouterr().out
        assert "owed" in out and "may only shrink" in out, (
            "and the acceptance names the debt it is carrying"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RF10 -- the probe is proven able to count
# ---------------------------------------------------------------------------
def test_rf10_the_probe_finds_the_debt_the_ledger_records():
    guard, restore = _load()
    try:
        files = dict(_real_files())
        total = 0
        for name in guard.LEDGER:
            n = guard.count_sites(files[name])
            assert n > 0, (
                f"{name} is owed for, so the census must find at least one "
                "site in it -- a zero here would mean the ledger and the "
                "probe disagree about what a direct call is"
            )
            total += n
        assert total >= len(guard.LEDGER), "the debt is not zero"
        assert hashlib.sha256(files[_an_owed_name(guard)].encode("utf-8")).hexdigest() \
            == guard.LEDGER[_an_owed_name(guard)], (
            "and the seal is taken on the same text the census reads"
        )
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
