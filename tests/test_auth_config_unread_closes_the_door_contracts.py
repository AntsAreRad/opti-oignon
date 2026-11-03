"""Contracts for the auth configuration that does not load whole: it closes the door, never opens it.

``AuthManager._load_config`` read ``config/auth.yaml`` and, on any exception while opening or parsing it, used the
built-in defaults: single-user mode on and registration open. An operator who turned single-user off and closed
registration, then saved a file that does not parse or cannot be read, got open registration and, while at most one
account existed, no authentication (the single-user property serves the synthetic admin). A file that loaded but was
cut before those two settings, or gave them misspelt, read them open the same way: every missing setting fell back
to the open default. Now a file that is there and does not load whole closes the door (single-user off, registration
closed, the reason kept and said), and a file that loads has each of the two settings it does not give as a boolean
closed.

  * UC1 -- A FILE THAT DOES NOT LOAD WHOLE CLOSES THE DOOR: a file that does not parse, one that parses to a list,
    one the parser reads by raising (an invalid date), one whose sections are no mappings, a folder at its name, a
    FIFO (answered at once), a file in a folder that cannot be searched: not single-user, registration closed, the
    reason kept (``config_unread``) and an error that says it once, never the file's text (it holds a secret).
  * UC2 -- A FILE THAT LOADS, OR NONE: a file that gives both settings as booleans and no file answer as before; an
    empty file, a file cut before its settings, one that lacks the registration setting, one that gives a misspelt
    or a quoted false, read each missing or misspelt setting closed, with a warning, the file's other settings kept
    (round 2: per setting, where the first version closed the whole file).
  * UC3 -- THE ROUTES SAY IT AND NEVER OPEN IT, the real route functions through the 2FA route suite's window: the
    status says when the configuration does not load; the mode route refuses (409) while it does not, the file and
    the mode untouched; it writes the file whole, keeping its rights (owner-only for a new one), through a link at
    its name, the mode in memory only after; a write that fails changes nothing; the security score reads the
    flag the configuration gives (it read an attribute the manager does not have), opening no account store.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import logging
import os
import stat
import sys
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_auth_2fa_routes_seam_contracts as rt_t  # noqa: E402
import test_auth_core_contracts as ac_t  # noqa: E402
from _isolation import source  # noqa: E402

BUDGET_S = {
    "test_uc1_a_file_that_does_not_load_whole_closes_the_door": 5.0,
    "test_uc2_a_file_that_loads_or_none": 5.0,
    "test_uc3_the_routes_say_it_and_never_open_it": 5.0,
}

_CLOSED_KEYS = "single_user_mode: false\nusers:\n  allow_registration: false\n"


def _manager(auth, tmpdir, config_path):
    """An ``AuthManager`` on its own database, reading ``config_path``."""
    return auth.AuthManager(config_path=config_path, db_path=os.path.join(tmpdir, uuid.uuid4().hex + ".db"))


def _state(manager):
    """(single-user, registration open, the reason kept)."""
    return (manager.single_user_mode, manager.config.get("users", {}).get("allow_registration", True),
            getattr(manager, "config_unread", None))


def _built_within(auth, tmpdir, path, fifo=None, seconds=3.0):
    """The state of a manager built on ``path`` within ``seconds``; a build still waiting then (an ``open`` that
    blocks on a FIFO) reads ``("hung", ...)``, the FIFO opened for writing to let it go."""
    done = []

    def build():
        try:
            done.append(_state(_manager(auth, tmpdir, path)))
        except Exception as exc:  # noqa: BLE001 -- a manager that cannot be built is the answer read
            done.append((f"raised {type(exc).__name__}", None, None))

    worker = threading.Thread(target=build, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        if fifo is not None:
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        worker.join(seconds)
        return ("hung", None, None)
    return done[0]


def _said(caplog, words):
    return [r.getMessage() for r in caplog.records if words in r.getMessage()]


def test_uc1_a_file_that_does_not_load_whole_closes_the_door(tmp_path, caplog):
    auth, restore, _counter, tmpdir = ac_t._load()
    seen = {}
    try:
        files = {
            "does not parse": "jwt:\n  secret_key: SECRET-uc1-0042\n bad: [\n",
            "a list": "- single_user_mode\n- false\n",
            "an invalid date": "since: 2026-13-45\n" + _CLOSED_KEYS,
            "users not a mapping": "single_user_mode: true\nusers: []\n",
            "jwt not a mapping": "jwt: []\nsingle_user_mode: true\nusers:\n  allow_registration: true\n",
        }
        paths = {}
        for name, text in files.items():
            paths[name] = tmp_path / f"{name.replace(' ', '_')}.yaml"
            paths[name].write_text(text, encoding="utf-8")
        paths["cannot be read"] = tmp_path / "a_folder.yaml"
        paths["cannot be read"].mkdir()
        fifo = tmp_path / "a_fifo.yaml"
        os.mkfifo(fifo)
        paths["a FIFO"] = fifo
        hidden = tmp_path / "unsearchable"
        hidden.mkdir()
        (hidden / "auth.yaml").write_text(_CLOSED_KEYS, encoding="utf-8")
        if os.geteuid() != 0:  # whoever runs as root searches it: no case to make
            paths["an unsearchable folder"] = hidden / "auth.yaml"
        hidden.chmod(0)
        try:
            for name, path in paths.items():
                caplog.clear()
                with caplog.at_level(logging.WARNING):
                    state = _built_within(auth, tmpdir, path, fifo if name == "a FIFO" else None)
                seen[name] = (state, _said(caplog, "does not load"), [r.getMessage() for r in caplog.records])
        finally:
            hidden.chmod(0o700)
    finally:
        restore()
    for name, ((single_user, registration, reason), said, lines) in seen.items():
        assert single_user is False, f"{name}: a file that does not load left single-user mode on (no authentication)"
        assert registration is False, f"{name}: a file that does not load left registration open"
        assert reason, f"{name}: the reason the file did not load was not kept"
        assert len(said) == 1 and "registration is" in said[0] and "restart" in said[0], f"{name}: not said once: {said}"
        assert "SECRET-uc1-0042" not in str(reason) and not any("SECRET-uc1-0042" in line for line in lines), (
            f"{name}: the reason or the log repeated the file's text, its secret with it")


def test_uc2_a_file_that_loads_or_none(tmp_path, caplog):
    auth, restore, _counter, tmpdir = ac_t._load()
    seen = {}
    try:
        files = {
            "single-user on": "single_user_mode: true\nusers:\n  allow_registration: true\n",
            "single-user off": _CLOSED_KEYS,
            "empty": "",
            "cut before its settings": "db_path: data/auth.db\npassword:\n  min_length: 12\n",
            "no registration setting": "single_user_mode: true\n",
            "a misspelt false": "single_user_mode: flase\nusers:\n  allow_registration: false\n",
            "a quoted false": "single_user_mode: 'false'\nusers:\n  allow_registration: 'no'\n",
        }
        for name, text in files.items():
            path = tmp_path / f"{name.replace(' ', '_')}.yaml"
            path.write_text(text, encoding="utf-8")
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                manager = _manager(auth, tmpdir, path)
            seen[name] = (_state(manager), len(_said(caplog, "gives no")), manager.config)
        seen["no file"] = (_state(_manager(auth, tmpdir, tmp_path / "absent.yaml")), 0, None)
    finally:
        restore()
    expected = {
        "single-user on": ((True, True, None), 0),
        "single-user off": ((False, False, None), 0),
        "no file": ((True, True, None), 0),
        "empty": ((False, False, None), 2),
        "cut before its settings": ((False, False, None), 2),
        "no registration setting": ((True, False, None), 1),
        "a misspelt false": ((False, False, None), 1),
        "a quoted false": ((False, False, None), 2),
    }
    for name, (state, warned) in expected.items():
        assert seen[name][:2] == (state, warned), (
            f"{name}: a setting a loaded file does not give as true or false was not read closed and said: {seen[name][:2]}")
    assert seen["cut before its settings"][2]["password"]["min_length"] == 12, (
        "a loaded file's other settings were not kept when a missing one was read closed")


class _Manager:
    """The auth manager the routes see: its configuration, the reason it did not load, one account."""

    def __init__(self, unread=None):
        self.config = {"single_user_mode": False, "users": {"allow_registration": False}}
        self.config_unread = unread

    @property
    def single_user_mode(self):
        return bool(self.config.get("single_user_mode", True))

    def count_users(self):
        return 1


def test_uc3_the_routes_say_it_and_never_open_it(tmp_path):
    import yaml
    from fastapi import HTTPException

    owner = {"sub": "local", "role": "admin"}
    seen = {}
    ctx = rt_t._load()
    try:
        deps = sys.modules["opti_oignon.api.deps"]
        routes = ctx.routes
        routes.__file__ = str(tmp_path / "api" / "routes_auth.py")
        config = tmp_path / "config"
        config.mkdir()
        target = config / "auth.yaml"

        def mode(flag, manager):
            deps.auth_manager = manager
            try:
                return "answered", routes.set_auth_mode(routes.AuthModeRequest(single_user_mode=flag), owner)
            except HTTPException as exc:
                return exc.status_code, exc.detail

        # The status
        deps.auth_manager = _Manager(unread="ScannerError")
        seen["status closed"] = routes.auth_status().get("auth_config_unread")
        deps.auth_manager = _Manager()
        seen["status loaded"] = routes.auth_status().get("auth_config_unread")
        # Closed: refused, nothing changed
        target.write_text(_CLOSED_KEYS, encoding="utf-8")
        closed = _Manager(unread="ScannerError")
        seen["closed"] = (mode(True, closed)[0], closed.config["single_user_mode"], target.read_text(encoding="utf-8"))
        # No file: the flag alone, owner-only
        target.unlink()
        fresh = _Manager()
        answer = mode(True, fresh)
        seen["no file"] = (answer[0], yaml.safe_load(target.read_text(encoding="utf-8")),
                           stat.S_IMODE(target.stat().st_mode), fresh.config["single_user_mode"])
        # A file's own rights kept
        target.chmod(0o640)
        mode(False, _Manager())
        seen["rights kept"] = stat.S_IMODE(target.stat().st_mode)
        # Through a link at its name
        real = tmp_path / "elsewhere.yaml"
        real.write_text(_CLOSED_KEYS, encoding="utf-8")
        target.unlink()
        target.symlink_to(real)
        mode(True, _Manager())
        seen["link"] = (target.is_symlink(), yaml.safe_load(real.read_text(encoding="utf-8"))["single_user_mode"])
        # A write that fails changes nothing
        target.unlink()
        target.write_text(_CLOSED_KEYS, encoding="utf-8")
        failing = _Manager()
        if os.geteuid() != 0:
            config.chmod(0o500)
            try:
                seen["failed write"] = (mode(True, failing)[0], failing.config["single_user_mode"],
                                        sorted(p.name for p in config.iterdir()))
            finally:
                config.chmod(0o700)
    finally:
        ctx.restore()
    score = source("api", "routes_security.py").read_text(encoding="utf-8")
    assert seen["status closed"] is True and seen["status loaded"] is False, (
        f"the status does not say when the auth configuration does not load: {seen['status closed']} {seen['status loaded']}")
    assert seen["closed"] == (409, False, _CLOSED_KEYS), (
        f"the mode route changed the mode over a configuration that does not load: {seen['closed']}")
    assert seen["no file"] == ("answered", {"single_user_mode": True}, 0o600, True), (
        f"the mode route did not write the flag alone, owner-only, before the mode in memory: {seen['no file']}")
    assert seen["rights kept"] == 0o640, f"the mode route did not keep the file's rights: {oct(seen['rights kept'])}"
    assert seen["link"] == (True, True), f"the mode route replaced a link at the file's name: {seen['link']}"
    if "failed write" in seen:
        assert seen["failed write"] == (500, False, ["auth.yaml"]), (
            f"a write that failed changed the mode in memory or left a file behind: {seen['failed write']}")
    assert 'passed = not getattr(auth_manager, "config", {}).get("single_user_mode", True)' in score, (
        "the security score reads an attribute the auth manager does not have, or opens the account store")
