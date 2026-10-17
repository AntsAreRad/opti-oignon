#!/usr/bin/env python3
"""Contracts for a machine's own switches: ``opti_oignon/config/local.yaml``, never committed.

The onion memory and the run journal are off as shipped, and their
switches live in tracked files (``config/onion.yaml``,
``config/run_journal.yaml``). A machine turns them on in ``local.yaml``
beside them, which git ignores, so a switch set for one machine changes no
tracked file:

  * LS1 -- THE ONION'S SWITCH: the shipped switch gives way to
    ``onion: {enabled: <bool>}`` in ``local.yaml``, both ways; anything but
    a YAML boolean, a file that does not parse and a top level that is no
    mapping leave the shipped switch, the reason said once; a path given to
    the loader is read as it stands.
  * LS2 -- THE RUN JOURNAL'S SWITCH: the same for ``run_journal:
    {enabled: <bool>}``, read at each call so a change shows at the next,
    and never over a shipped file that cannot be read, which stays off.
  * LS3 -- NEVER COMMITTED: git ignores ``opti_oignon/config/local.yaml``.

The onion's modules and the journal are loaded through the shared isolation
window, their shipped files copied into a temporary directory, where each
case writes its own ``local.yaml``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_ls1_the_onions_switch_gives_way_to_the_machines_own": 5.0,
    "test_ls2_the_run_journals_switch_gives_way_to_the_machines_own_and_never_over_an_unreadable_file": 5.0,
    "test_ls3_git_ignores_the_machines_own_switches": 2.0,
}

_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian")
_LIBRARIAN = "opti_oignon.memory.librarian"
_COMPOSER = "opti_oignon.memory.composer"
_PEELS = "opti_oignon.memory.peels"
_WRAPPER = "opti_oignon.agent.untrusted_context"
_RJ = "opti_oignon.run_journal"
_SHIPPED_ONION = REPO / "opti_oignon" / "config" / "onion.yaml"
_SHIPPED_JOURNAL = REPO / "opti_oignon" / "config" / "run_journal.yaml"


def _onion_window(tmp_path):
    """The onion's modules from source, their ``onion.yaml`` a copy of the shipped one in ``tmp_path``."""
    targets = {f"opti_oignon.memory.{name}": source("memory", f"{name}.py") for name in _MODULES}
    targets[_WRAPPER] = source("agent", "untrusted_context.py")
    loaded, restore = isolate(
        targets=targets,
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils", "opti_oignon.native"),
        packages=("opti_oignon.memory", "opti_oignon.agent"),
    )
    onion = tmp_path / "onion.yaml"
    onion.write_text(_SHIPPED_ONION.read_text(encoding="utf-8"), encoding="utf-8")
    for name in (_COMPOSER, _PEELS, _LIBRARIAN):
        loaded[name]._CONFIG = onion
    loaded[_LIBRARIAN].reset_librarian()
    return loaded[_LIBRARIAN], onion, restore


def _local(tmp_path, text):
    path = tmp_path / "local.yaml"
    if text is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def test_ls1_the_onions_switch_gives_way_to_the_machines_own(tmp_path, caplog):
    """LS1: ``onion: {enabled: <bool>}`` in local.yaml wins over the shipped switch, both ways; nothing else does."""
    librarian, onion, restore = _onion_window(tmp_path)
    try:
        shipped = onion.read_text(encoding="utf-8")
        assert "\nenabled: false\n" in shipped, "the onion is shipped off"
        seen = {}
        for case, text in (
            ("no local file", None),
            ("on locally", "onion:\n  enabled: true\n"),
            ("a string, not a boolean", "onion:\n  enabled: 'true'\n"),
            ("a one, not a boolean", "onion:\n  enabled: 1\n"),
            ("no mapping at the top", "- onion\n"),
            ("does not parse", "onion: [\n"),
            ("another section only", "run_journal:\n  enabled: true\n"),
        ):
            _local(tmp_path, text)
            seen[case] = (librarian.load_config().enabled, librarian.onion_enabled())
        assert seen == {
            "no local file": (False, False),
            "on locally": (True, True),
            "a string, not a boolean": (False, False),
            "a one, not a boolean": (False, False),
            "no mapping at the top": (False, False),
            "does not parse": (False, False),
            "another section only": (False, False),
        }, ("only a boolean onion.enabled in local.yaml turns the onion on", seen)
        caplog.clear()
        librarian._LOCAL_SAID.clear()
        with caplog.at_level(logging.WARNING):
            _local(tmp_path, "onion:\n  enabled: 'yes'\n")
            librarian.load_config()
            librarian.load_config()
        said = [r.getMessage() for r in caplog.records if "local.yaml" in r.getMessage()]
        assert len(said) == 1 and "onion" in said[0], ("a switch that is no boolean is said once", said)
        onion.write_text(shipped.replace("\nenabled: false\n", "\nenabled: true\n", 1), encoding="utf-8")
        _local(tmp_path, "onion:\n  enabled: false\n")
        assert (librarian.load_config().enabled, librarian.onion_enabled()) == (False, False), (
            "the machine's own switch turns a shipped switch off too")
        _local(tmp_path, None)
        assert librarian.onion_enabled() is True, "with no local file the shipped switch stands"
        onion.write_text(shipped, encoding="utf-8")
        _local(tmp_path, "onion:\n  enabled: true\n")
        assert librarian.load_config(onion).enabled is False, "a path given to the loader is read as it stands"
    finally:
        restore()


def test_ls2_the_run_journals_switch_gives_way_to_the_machines_own_and_never_over_an_unreadable_file(tmp_path):
    """LS2: ``run_journal: {enabled: <bool>}`` wins, read at each call; an unreadable shipped file stays off."""
    loaded, restore = isolate(targets={_RJ: source("run_journal.py")})
    try:
        rj = loaded[_RJ]
        shipped = tmp_path / "run_journal.yaml"
        shipped.write_text(_SHIPPED_JOURNAL.read_text(encoding="utf-8"), encoding="utf-8")
        rj.CONFIG_FILE = shipped
        assert rj.settings()["enabled"] is False, "the run journal is shipped off"
        seen = {}
        for case, text in (
            ("no local file", None),
            ("on locally", "run_journal:\n  enabled: true\n"),
            ("a string, not a boolean", "run_journal:\n  enabled: 'true'\n"),
            ("does not parse", "run_journal: [\n"),
            ("another section only", "onion:\n  enabled: true\n"),
        ):
            _local(tmp_path, text)
            got = rj.settings()
            seen[case] = (got["enabled"], got["state"], rj.enabled())
        assert seen == {
            "no local file": (False, "off", False),
            "on locally": (True, "on", True),
            "a string, not a boolean": (False, "off", False),
            "does not parse": (False, "off", False),
            "another section only": (False, "off", False),
        }, ("only a boolean run_journal.enabled in local.yaml turns the journal on, at the next call", seen)
        _local(tmp_path, "run_journal:\n  enabled: true\n")
        shipped.write_text("enabled: [\n", encoding="utf-8")
        got = rj.settings()
        assert (got["enabled"], got["state"]) == (False, "unreadable"), (
            "a shipped file that cannot be read stays off whatever the local switch says", got)
        shipped.write_text(_SHIPPED_JOURNAL.read_text(encoding="utf-8"), encoding="utf-8")
        assert rj.settings(shipped)["enabled"] is False, "a path given to the reader is read as it stands"
    finally:
        restore()


def test_ls3_git_ignores_the_machines_own_switches():
    """LS3: ``opti_oignon/config/local.yaml`` is a line of the repository's .gitignore."""
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "opti_oignon/config/local.yaml" in lines, "a machine's own switches are never committed"
