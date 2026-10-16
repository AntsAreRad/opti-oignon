#!/usr/bin/env python3
"""Contracts for the onion's configuration as a turn's plan reads it: one parse for each text of its file.

A plan reads ``onion.yaml`` through five loaders -- the budgets, the
journal, the librarian's own, the queue's ladder and the gate -- and a turn
through more. Every read reads the file by the call the loaders made before
any parse was kept, and the file is parsed again only when its text is not
the text the kept parse was made from; every loader reads that parse, each
with a copy of its own.

  * JW32a -- eight request threads planning at once over a text none of
    them read, then twelve plans more, parse onion.yaml once, and the
    governor's file, read for its ladder, once.
  * JW32b -- the file replaced by a rename with a text of the same size,
    its modification time kept: the next read reads the new value, parsed
    once, and the old text renamed back reads as before.
  * JW32c -- an edit in place that does not parse is refused by name as a
    parse of its own refuses it, at every loader and at every plan, each
    read raising a refusal of its own and nothing kept for the file; an
    edit one loader refuses is refused by that loader in its words, and the
    plan with it, while the switch reads; the file restored reads as before.
  * JW32d -- a caller that changes the configuration it was handed changes
    no other read: the next read, and every loader, read the file.
  * JW32e -- a rewrite in place of the same size, its stat the old
    version's to the last field, is read anew.
  * JW32f -- a file its reader may no longer read, its mode changed and
    nothing else, is refused at the next read by the same call, in the same
    words, at every loader and at the plan; its mode restored, it reads.
  * JW32g -- the governor's ladder broken, the plan is refused; restored,
    the next plan reads: what a refused read adds to the keys named twice
    is not kept for the next read.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source; the registry and the native core are
blocked. Each version of the file is written with a modification time long
past, as a file is that a running process reads; the parses are counted by
wrapping the YAML parser in this process. JW32e hands the cache the stat a
clock, an attribute cache or a file system that saw no change would give:
the old version's own. JW32f is skipped for root, whom a file's mode
refuses nothing.
"""

import os
import sys
import threading
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_jw32a_twenty_plans_eight_of_them_at_once_over_a_text_that_does_not_change_parse_it_once": 5.0,
    "test_jw32b_a_rename_to_a_text_of_the_same_size_its_modification_time_kept_is_read_at_the_next_read": 5.0,
    "test_jw32c_an_edit_that_does_not_read_is_refused_by_name_as_before_and_the_file_restored_reads_again": 5.0,
    "test_jw32d_a_caller_that_changes_its_configuration_changes_no_other_read": 5.0,
    "test_jw32e_a_rewrite_in_place_whose_stat_is_the_old_versions_is_read_anew": 5.0,
    "test_jw32f_a_file_its_reader_may_no_longer_read_is_refused_by_the_same_call_as_before": 5.0,
    "test_jw32g_a_governor_ladder_broken_then_restored_plans_again": 5.0,
}

_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian")
_COMPOSER, _PEELS, _LIBRARIAN = (f"opti_oignon.memory.{name}" for name in ("composer", "peels", "librarian"))
_WRAPPER = "opti_oignon.agent.untrusted_context"
_ONION_YAML = REPO / "opti_oignon" / "config" / "onion.yaml"
_GOVERNOR_YAML = REPO / "opti_oignon" / "config" / "resource_governor.yaml"
# An instant long past: a version written then is older than any tick of a file system's clock.
_LONG_AGO_NS = 1_700_000_000 * 1_000_000_000
_SECOND_NS = 1_000_000_000
_SYSTEM = ("You are a careful assistant. Answer from what you are shown.",)
_RHO = "\nqueue:\n  rho: 0.5 "
_SWITCH = "\nenabled: false\n"
_LADDER = "\nctx_ladder: [32768, 16384, 8192, 4096]\n"


def _shipped():
    return _ONION_YAML.read_text(encoding="utf-8")


def _write(path, text, at):
    """``text`` written in place at ``path`` -- the same inode -- its modification time set to ``at``."""
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(at, at))


def _renamed(path, text, at):
    """``text`` written beside ``path`` and renamed over it: another inode, its modification time ``at``."""
    fresh = path.with_name(path.name + ".new")
    _write(fresh, text, at)
    os.replace(fresh, path)


def _window(tmp_path):
    """The onion's modules and the wrapper from source, their ``onion.yaml`` a copy of the shipped one, long past."""
    targets = {f"opti_oignon.memory.{name}": source("memory", f"{name}.py") for name in _MODULES}
    targets[_WRAPPER] = source("agent", "untrusted_context.py")
    loaded, restore = isolate(
        targets=targets,
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils", "opti_oignon.native"),
        packages=("opti_oignon.memory", "opti_oignon.agent"),
    )
    onion = tmp_path / "onion.yaml"
    _write(onion, _shipped(), _LONG_AGO_NS)
    for name in (_COMPOSER, _PEELS, _LIBRARIAN):
        loaded[name]._CONFIG = onion
    loaded[_LIBRARIAN].reset_librarian()
    return loaded, onion, restore


def _count(text):
    """A turn's counter: two tokens a word, and one more for every forty characters."""
    return 2 * len(text.split()) + len(text) // 40


def _plan(loaded, cid, words="What did Alice review?"):
    """One turn's plan in conversation ``cid``, with no configuration handed over, as the executor plans one."""
    turn = {"role": "user", "content": words, "origin": "typed", "segments": [[0, len(words), "typed"]]}
    return loaded[_LIBRARIAN].plan_window(cid, model="model-a", count=_count, system=_SYSTEM, turn=turn,
                                          wrap=loaded[_WRAPPER].wrap)


def _parses(monkeypatch):
    """The text of every YAML parse this process starts from now on: the parser wrapped where each is made."""
    made = []
    making = yaml.SafeLoader.__init__

    def counted(self, stream):
        made.append(stream)
        making(self, stream)

    monkeypatch.setattr(yaml.SafeLoader, "__init__", counted)
    return made


def _raised(read):
    """What ``read`` raises, the exception itself."""
    with pytest.raises(Exception) as caught:
        read()
    return caught.value


def _refusal(read):
    """What ``read`` raises: its type and its words."""
    raised = _raised(read)
    return type(raised), str(raised)


def _readers(loaded):
    """Every loader a plan reads ``onion.yaml`` through, by name."""
    composer, peels, lib = loaded[_COMPOSER], loaded[_PEELS], loaded[_LIBRARIAN]
    return {"budgets": composer.load_budgets, "journal": composer.load_journal, "librarian": lib.load_config,
            "ladder": peels.load_ladder, "gate": peels.load_gate}


# ---------------------------------------------------------------------------
# JW32a -- one parse for many plans
# ---------------------------------------------------------------------------
def test_jw32a_twenty_plans_eight_of_them_at_once_over_a_text_that_does_not_change_parse_it_once(tmp_path,
                                                                                                  monkeypatch):
    loaded, _onion, restore = _window(tmp_path)
    try:
        shipped, governor = _shipped(), _GOVERNOR_YAML.read_text(encoding="utf-8")
        made = _parses(monkeypatch)
        start, rungs = threading.Barrier(8), []

        def turn(index):
            start.wait()
            rungs.append(_plan(loaded, f"t{index}").rung)

        threads = [threading.Thread(target=turn, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert sorted(rungs) == [16384] * 8, ("each thread reads the home rung the file names", rungs)
        plans = [_plan(loaded, "c1", f"Turn {i}: what did Alice review?") for i in range(12)]
        assert [(plan.refusal, plan.rung) for plan in plans] == [(None, 16384)] * 12, plans
        assert made.count(shipped) == 1, f"twenty plans, eight at once, parsed onion.yaml {made.count(shipped)} times"
        assert made.count(governor) == 1, f"the governor's file, read for its ladder, parsed {made.count(governor)} times"
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW32b -- a rename to a text of the same size, its modification time kept
# ---------------------------------------------------------------------------
def test_jw32b_a_rename_to_a_text_of_the_same_size_its_modification_time_kept_is_read_at_the_next_read(
        tmp_path, monkeypatch):
    loaded, onion, restore = _window(tmp_path)
    try:
        peels = loaded[_PEELS]
        shipped = _shipped()
        first = _plan(loaded, "c1")
        assert first.refusal is None and peels.load_ladder().rho == 0.5, ("control: the shipped file reads", first)
        seen = os.stat(onion)
        assert shipped.count(_RHO) == 1, "control: rho, named once"
        changed = shipped.replace(_RHO, "\nqueue:\n  rho: 0.4 ")
        made = _parses(monkeypatch)
        _renamed(onion, changed, seen.st_mtime_ns)
        now = os.stat(onion)
        assert (now.st_size, now.st_mtime_ns) == (seen.st_size, seen.st_mtime_ns) and now.st_ino != seen.st_ino, (
            "control: another inode, its size and modification time kept")
        assert peels.load_ladder().rho == 0.4, "the next read reads the new value"
        assert _plan(loaded, "c1").refusal is None and peels.load_ladder().rho == 0.4, "and every read after it"
        assert made.count(changed) == 1, f"the new text parsed {made.count(changed)} times"
        _renamed(onion, shipped, seen.st_mtime_ns)
        assert peels.load_ladder().rho == 0.5, "the old text renamed back reads as before"
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW32c -- an edit that does not read, then the file restored
# ---------------------------------------------------------------------------
def test_jw32c_an_edit_that_does_not_read_is_refused_by_name_as_before_and_the_file_restored_reads_again(tmp_path):
    loaded, onion, restore = _window(tmp_path)
    try:
        composer, peels, lib, readers = loaded[_COMPOSER], loaded[_PEELS], loaded[_LIBRARIAN], _readers(loaded)
        shipped = _shipped()
        assert shipped.count(_SWITCH) == 1 and shipped.count(_RHO) == 1, "control: the switch and rho, named once"
        on = shipped.replace(_SWITCH, "\nenabled: true\n")
        _write(onion, on, _LONG_AGO_NS + _SECOND_NS)
        assert lib.onion_enabled() is True and _plan(loaded, "c1").rung == 16384, "control: the file switched on reads"
        inode = os.stat(onion).st_ino
        # A text the YAML parser refuses, written in place: refused at every loader as a parse of its own refuses
        # it, at every plan, each read raising a refusal of its own, nothing kept; the switch that cannot be read is
        # off.
        broken = on.replace(_RHO, "\nqueue:\n  rho: [0.5 ")
        own = _refusal(lambda: yaml.safe_load(broken))
        assert issubclass(own[0], yaml.YAMLError), ("control: the YAML parser refuses the edit", own)
        _write(onion, broken, _LONG_AGO_NS + 2 * _SECOND_NS)
        assert os.stat(onion).st_ino == inode, "control: written in place, the same inode"
        for attempt in range(2):
            plan = _plan(loaded, "c1")
            assert plan.refusal is not None and plan.refusal.motive == "config", (attempt, plan)
            for name, read in readers.items():
                assert _refusal(read) == own, (attempt, name, _refusal(read), own)
            assert lib.onion_enabled() is False, (attempt, "a switch that cannot be read is off")
        first, second = _raised(readers["ladder"]), _raised(readers["ladder"])
        assert first is not second, "two reads of a file that does not parse raise two refusals, none kept"
        assert os.path.abspath(onion) not in composer._parses, "nothing is kept for a file that does not parse"
        # A text one loader refuses, in place and its size kept: that loader refuses it in its words, the plan
        # with it, and the switch reads.
        refused = on.replace(_RHO, "\nqueue:\n  rho: 1.5 ")
        _write(onion, refused, _LONG_AGO_NS + 3 * _SECOND_NS)
        assert os.stat(onion).st_size == len(on.encode("utf-8")), "control: the size kept"
        plan = _plan(loaded, "c1")
        assert plan.refusal is not None and plan.refusal.motive == "config", plan
        assert _refusal(readers["ladder"]) == (peels.GateError, "queue: rho: 1.5 is not a number in (0, 1]"), (
            _refusal(readers["ladder"]))
        assert lib.onion_enabled() is True, "the switch reads a file its own loader reads"
        # The file restored, in place: it reads as before.
        _write(onion, on, _LONG_AGO_NS + 4 * _SECOND_NS)
        plan = _plan(loaded, "c1")
        assert plan.refusal is None and plan.rung == 16384, ("the file restored plans as before", plan)
        assert readers["ladder"]().rho == 0.5 and lib.onion_enabled() is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW32d -- a caller's changes reach no other read
# ---------------------------------------------------------------------------
def test_jw32d_a_caller_that_changes_its_configuration_changes_no_other_read(tmp_path):
    loaded, _onion, restore = _window(tmp_path)
    try:
        composer, peels, lib = loaded[_COMPOSER], loaded[_PEELS], loaded[_LIBRARIAN]
        document = yaml.safe_load(_shipped())
        handed = composer.read_onion()
        assert handed == document, "control: the document a parse of its own builds"
        assert composer.read_onion() is not handed, "each read hands a copy of its own"
        handed["queue"]["rho"] = 0.75
        handed["budgets"][16384]["flesh"] = 1
        handed["directives"]["en"]["lead_ins"].append("meanwhile")
        handed["decisions"].clear()
        del handed["gate"]
        assert composer.read_onion() == document, "the next read is the file's, not what a caller made of its copy"
        assert peels.load_ladder().rho == 0.5, "the queue's ladder reads the file"
        assert composer.load_budgets().at(16384).flesh == document["budgets"][16384]["flesh"], "the budgets too"
        assert composer.load_journal().margin == document["journal"]["margin"], "the journal too"
        assert lib.load_config().model == document["librarian"]["model"], "the librarian's own too"
        assert peels.load_gate().directives is not None, "and the gate"
        plan = _plan(loaded, "c1")
        assert plan.refusal is None and plan.rung == 16384, plan
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW32e -- a rewrite whose stat is the old version's
# ---------------------------------------------------------------------------
def test_jw32e_a_rewrite_in_place_whose_stat_is_the_old_versions_is_read_anew(tmp_path, monkeypatch):
    loaded, onion, restore = _window(tmp_path)
    try:
        peels = loaded[_PEELS]
        shipped = _shipped()
        assert peels.load_ladder().rho == 0.5, "control: the version read"
        seen = os.stat(onion)
        assert shipped.count(_RHO) == 1, "control: rho, named once"
        onion.write_text(shipped.replace(_RHO, "\nqueue:\n  rho: 0.4 "), encoding="utf-8")
        written = os.stat(onion)
        assert (written.st_size, written.st_ino) == (seen.st_size, seen.st_ino), "control: in place, its size kept"
        real_stat = os.stat

        def unchanged(path, *args, **kwargs):
            """The stat of the file as one that saw no change gives it: the old version's, to the last field."""
            if not isinstance(path, int) and os.fspath(path) == str(onion):
                return seen
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", unchanged)
        assert os.stat(onion) == seen, "control: the stat is the old version's"
        assert peels.load_ladder().rho == 0.4, "a rewrite whose stat is the old version's is read anew"
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW32f -- a file its reader may no longer read
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode: no read of it is refused")
def test_jw32f_a_file_its_reader_may_no_longer_read_is_refused_by_the_same_call_as_before(tmp_path):
    loaded, onion, restore = _window(tmp_path)
    mode = os.stat(onion).st_mode
    try:
        lib, readers = loaded[_LIBRARIAN], _readers(loaded)
        _write(onion, _shipped().replace(_SWITCH, "\nenabled: true\n"), _LONG_AGO_NS + _SECOND_NS)
        assert lib.onion_enabled() is True and _plan(loaded, "c1").rung == 16384, "control: the file read, and kept"
        os.chmod(onion, 0)
        own = _refusal(lambda: onion.read_text(encoding="utf-8"))
        assert own[0] is PermissionError, ("control: the call every loader made refuses it", own)
        for name, read in readers.items():
            assert _refusal(read) == own, (name, _refusal(read), own)
        plan = _plan(loaded, "c1")
        assert plan.refusal is not None and plan.refusal.motive == "config", plan
        assert lib.onion_enabled() is False, "a switch that cannot be read is off"
        os.chmod(onion, mode)
        assert lib.onion_enabled() is True and _plan(loaded, "c1").rung == 16384, "its mode restored, the file reads"
    finally:
        os.chmod(onion, mode)
        restore()


# ---------------------------------------------------------------------------
# JW32g -- the governor's ladder broken, then restored
# ---------------------------------------------------------------------------
def test_jw32g_a_governor_ladder_broken_then_restored_plans_again(tmp_path):
    loaded, _onion, restore = _window(tmp_path)
    try:
        composer = loaded[_COMPOSER]
        written = _GOVERNOR_YAML.read_text(encoding="utf-8")
        assert written.count(_LADDER) == 1, "control: the shipped ladder, named once"
        governor = tmp_path / _GOVERNOR_YAML.name
        _write(governor, written, _LONG_AGO_NS)
        composer._GOVERNOR_CONFIG = governor
        assert _plan(loaded, "c1").rung == 16384, "control: the plan reads"
        _write(governor, written.replace(_LADDER, "\nctx_ladder: []\n"), _LONG_AGO_NS + _SECOND_NS)
        for attempt in range(2):
            plan = _plan(loaded, "c1")
            assert plan.refusal is not None and plan.refusal.motive == "config", (attempt, plan)
            assert _refusal(composer.load_budgets) == (
                composer.BudgetError, f"ctx_ladder unreadable in {governor.name}: missing or empty"), (
                attempt, _refusal(composer.load_budgets))
        _write(governor, written, _LONG_AGO_NS + 2 * _SECOND_NS)
        plan = _plan(loaded, "c1")
        assert plan.refusal is None and plan.rung == 16384, ("the ladder restored, the plan reads", plan)
        assert composer.load_budgets().rungs == (4096, 8192, 16384, 32768), "and the budgets read every rung"
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
