#!/usr/bin/env python3
"""Contracts: the security configuration is changed by the owner's password, never over a file that does not load.

``PUT /api/security/config`` merges the sections it is given into security.yaml: the login limiter the ceremonies
lean on, encryption, the cookie settings, CORS, the headers, the sandbox, the plugins, the search defense, the backup.
It asked nothing but a session, and in single-user Daily every local request is the synthetic admin. The four routes
that write security.yaml by loading it, changing it and writing it back -- that one, encryption's setup, and remote
access on and off -- took a file that did not load for an empty one and wrote back the sections they held alone, the
security mode among those dropped.

  * SY1 -- every change of the configuration route asks the owner's password through the owner's own session, as the
    other ceremonies ask it (``_ceremony_password``), whatever the section and before security.yaml is read: a wrong
    password, one that cannot be checked, no account or another account's session refuses, audited by name with no
    secret, and leaves the file as it was; the right one merges the section, keeps the rest and is audited by its
    sections' names.
  * SY2 -- a security.yaml that is there and does not load whole (it cannot be read or looked at, is no regular file,
    a FIFO or a folder swapped in after the look included, no YAML, or holds no mapping), or whose section a writer
    changes holds no mapping, is left as it is by each of the four writers, with 409 and its reason, before any key is
    enabled or made and before any TLS file is made; a device the look sees is not opened, and no descriptor of the
    file stays open. No file, an empty one, one of comments only and a null section are written as before. The
    configuration route says the reason too.
  * SY3 -- turning encryption on through the route seals the run journal at once, whole (rj7 of
    tests/test_run_journal_contracts.py, which called the route bare, run here with the owner's password).
  * SY4 -- the security mode and the security routes replace security.yaml whole, its mode kept (rj15, the route
    given the owner's password, a wrong one refused first).

The window is the one of tests/test_checks_fail_closed_contracts.py: ``ceremony_factors`` and ``routes_security``
from their sources, the account store, the limiter and bcrypt stood in for.
"""

import asyncio
import contextlib
import inspect
import os
import stat
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_checks_fail_closed_contracts as nf_t  # noqa: E402
import test_run_journal_contracts as rj_t  # noqa: E402
from _isolation import isolate, source  # noqa: E402

FACTORS = "opti_oignon.ceremony_factors"
ROUTES = "opti_oignon.api.routes_security"
TEXT = "security_mode: daily\nheaders:\n  x_frame_options: SAMEORIGIN\nrate_limiting:\n  enabled: true\n"
LOCAL = "127.0.0.1"
# The sections the configuration route merges, closed: one more is a change of this census.
SECTIONS = ("cors", "headers", "rate_limiting", "sandbox", "plugins", "encryption", "jwt", "search_safety", "backup")


def _case_id(case):
    return case[0]


# Seconds per junit case, the parametrized ids included: the ladder reads this literal alone.
BUDGET_S = {
    "test_sy1_every_change_of_the_configuration_asks_the_owners_password": 3.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[put_over_not_UTF-8]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[put_over_not_YAML]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[put_over_no_mapping]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[put_over_a_folder]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[put_over_a_link_to_a_device]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[put_over_a_FIFO_swapped_in_after_the_look]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[put_over_a_folder_swapped_in_after_the_look]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[put_over_its_rights]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[put_over_its_folder_not_searchable]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[setup_over_not_UTF-8]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[setup_over_not_YAML]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[setup_over_no_mapping]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[setup_over_a_folder]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[setup_over_a_link_to_a_device]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[setup_over_a_FIFO_swapped_in_after_the_look]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[setup_over_a_folder_swapped_in_after_the_look]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[setup_over_its_rights]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[setup_over_its_folder_not_searchable]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_on_over_not_UTF-8]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_on_over_not_YAML]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_on_over_no_mapping]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_on_over_a_folder]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_on_over_a_link_to_a_device]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is"
    "[remote_on_over_a_FIFO_swapped_in_after_the_look]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is"
    "[remote_on_over_a_folder_swapped_in_after_the_look]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_on_over_its_rights]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_on_over_its_folder_not_searchable]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_off_over_not_UTF-8]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_off_over_not_YAML]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_off_over_no_mapping]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_off_over_a_folder]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_off_over_a_link_to_a_device]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is"
    "[remote_off_over_a_FIFO_swapped_in_after_the_look]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is"
    "[remote_off_over_a_folder_swapped_in_after_the_look]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_off_over_its_rights]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_off_over_its_folder_not_searchable]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[put_over_a_section_that_holds_no_mapping]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[setup_over_a_section_that_holds_no_mapping]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is[remote_on_over_a_section_that_holds_no_mapping]": 2.0,
    "test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is"
    "[remote_off_over_a_section_that_holds_no_mapping]": 2.0,
    "test_sy3_turning_encryption_on_through_the_route_seals_the_run_journal_at_once": 5.0,
    "test_sy4_the_security_settings_are_written_whole_by_the_owners_password": 2.0,
}


def _window(tmp_path, *, accounts=None, limiter=None):
    """``routes_security`` in a window with the real check, security.yaml at ``tmp_path``; the account store, the
    limiter, bcrypt, the security mode, the run journal, the encryption manager, the TLS manager and the signed audit
    chain stood in for, each recording what it is asked."""
    accounts = accounts if accounts is not None else nf_t._Accounts(
        nf_t.RESIDENT, nf_t._Account("uid-user", "registered", nf_t.RIGHT, role="user"))
    limiter = limiter if limiter is not None else nf_t._Limiter()
    verifier = nf_t._Verifier()
    seen = SimpleNamespace(audit=[], sealed=[], keys=[], tls=[], chained=[])

    auth_routes = types.ModuleType("opti_oignon.api.routes_auth")
    auth_routes._get_current_user = lambda: dict(nf_t.SESSION)
    auth_routes._extract_client_ip = lambda request: nf_t.CLIENT

    mode = types.ModuleType("opti_oignon.security_mode")
    mode.MODE_BULBE, mode.MODE_DAILY, mode.VALID_MODES = "bulbe", "daily", ("bulbe", "daily")
    mode.security_mode_manager = SimpleNamespace()
    mode.is_bulbe = lambda: False
    mode._audit_log = lambda event, severity="INFO", **details: seen.audit.append((event, severity, details))

    journal = types.ModuleType("opti_oignon.run_journal")
    journal.on_encryption_enabled = lambda: seen.sealed.append(1)

    enc = types.ModuleType("opti_oignon.encryption")

    class KeyfileExists(Exception):
        pass

    manager = SimpleNamespace(enabled=False, get_status=lambda: {"enabled": True})
    manager.enable_with_existing_key = lambda: seen.keys.append("existing") and False
    manager.setup_random_key = lambda: seen.keys.append("made") or True
    manager.setup_from_passphrase = lambda phrase: seen.keys.append("derived") or True
    enc.KeyfileExists, enc.get_encryption_manager, enc.keyfile_present = KeyfileExists, lambda: manager, lambda: False

    secrets = types.ModuleType("opti_oignon.process_secrets")
    secrets.secret = lambda name: None

    tls = types.ModuleType("opti_oignon.tls_manager")
    tls.get_tls_status = lambda: {"enabled": False}
    tls.setup_tls = lambda passphrase: seen.tls.append(1) or {"success": True}

    chain = types.ModuleType("opti_oignon.signed_audit_log")
    chain.chain_log = lambda **details: seen.chained.append(details["event_type"])

    loaded, restore = isolate(
        targets={FACTORS: source("ceremony_factors.py"), ROUTES: source("api", "routes_security.py")},
        seeded={"opti_oignon.auth": nf_t._auth(accounts, limiter, verifier), "opti_oignon.api.routes_auth": auth_routes,
                "opti_oignon.security_mode": mode, "opti_oignon.run_journal": journal, "opti_oignon.encryption": enc,
                "opti_oignon.process_secrets": secrets, "opti_oignon.tls_manager": tls,
                "opti_oignon.signed_audit_log": chain},
        packages=("opti_oignon.api",),
    )
    try:
        routes = loaded[ROUTES]
        config = tmp_path / "security.yaml"
        routes._SECURITY_YAML_PATH = str(config)
        routes._plaintext_stores = lambda: []
    except BaseException:
        restore()
        raise
    return SimpleNamespace(routes=routes, config=config, seen=seen, verifier=verifier, limiter=limiter,
                           restore=restore)


def _request():
    return SimpleNamespace(client=SimpleNamespace(host=LOCAL), headers={})


def _route_put(route, update, session):
    if list(inspect.signature(route).parameters) == ["update"]:
        # The route before block 221 took the body alone: the red run on 220 reaches the claims through it.
        return route(update)
    return route(update, _request(), dict(session))


def _put(w, password, session=nf_t.SESSION, **sections):
    update = w.routes.SecurityConfigUpdate(password=password, **sections)
    return _route_put(w.routes.update_security_config, update, session)


def _state(path):
    """What is at ``path``: its kind, its rights and its bytes when it is a file, or the error that met the look."""
    try:
        found = os.lstat(path)
    except OSError as exc:
        return ("cannot look", type(exc).__name__)
    if stat.S_ISREG(found.st_mode):
        try:
            held = Path(path).read_bytes()
        except OSError as exc:
            held = type(exc).__name__
        return ("file", stat.S_IMODE(found.st_mode), found.st_ino, found.st_mtime_ns, held)
    return ("other", stat.S_IFMT(found.st_mode), stat.S_IMODE(found.st_mode))


# --------------------------------------------------------------------------- SY1


def test_sy1_every_change_of_the_configuration_asks_the_owners_password(tmp_path):
    import pydantic
    import yaml
    from fastapi import HTTPException

    w = _window(tmp_path)
    try:
        w.config.write_text(TEXT, encoding="utf-8")
        before = _state(w.config)
        for section in SECTIONS:
            for password, session, status, error in ((nf_t.WRONG, nf_t.SESSION, 403, "wrong_password"),
                                                     (nf_t.LONG, nf_t.SESSION, 503, "password_unverifiable"),
                                                     (nf_t.RIGHT, {"sub": "uid-nobody"}, 403, "no_account"),
                                                     (nf_t.RIGHT, {"sub": "uid-user"}, 403, "not_owner")):
                w.seen.audit.clear()
                with pytest.raises(HTTPException) as caught:
                    _put(w, password, session, **{section: {"enabled": False}})
                assert caught.value.status_code == status, (section, error)
                assert _state(w.config) == before, (section, error, "the file is left as it was")
                assert [(event, details["ceremony"], details["error"], details["user_id"])
                        for event, _sev, details in w.seen.audit] == [
                    ("ceremony_password_refused", "security_config", error, session["sub"])], (section, error)
                said = repr(w.seen.audit) + str(caught.value.detail)
                assert not any(secret in said for secret in (nf_t.RIGHT, nf_t.WRONG, nf_t.LONG)), (section, error)
        assert w.seen.sealed == [] and w.seen.keys == []
        # The limiter saw the client's address, as the login's helper reads it.
        assert set(w.limiter.asked) == {(nf_t.CLIENT, "resident")}
        routes = w.routes
        # A plain function: FastAPI runs it in a worker thread, so bcrypt never holds the event loop.
        assert not inspect.iscoroutinefunction(routes.update_security_config)
        # The sections the route merges are the body's fields but the password, all of them: none is told safe.
        fields = set(routes.SecurityConfigUpdate.model_fields) - {"password"}
        assert fields == set(routes._CONFIG_SECTIONS) == set(SECTIONS), (sorted(fields), routes._CONFIG_SECTIONS)
        with pytest.raises(pydantic.ValidationError):
            routes.SecurityConfigUpdate(headers={"x_frame_options": "DENY"})
        # The right password: each section merged into what was there, the rest kept, audited by the sections' names.
        for section in SECTIONS:
            w.seen.audit.clear()
            answer = _put(w, nf_t.RIGHT, **{section: {"enabled": False}})
            assert answer["updated"] is True and answer["sections"] == [section], answer
            written = yaml.safe_load(w.config.read_text(encoding="utf-8"))
            assert written["security_mode"] == "daily" and written[section]["enabled"] is False, (section, written)
            assert [(event, severity, details) for event, severity, details in w.seen.audit] == [
                ("security_config_changed", "WARNING", {"sections": [section], "user_id": "uid-resident"})], section
        assert written["headers"] == {"x_frame_options": "SAMEORIGIN", "enabled": False}, written
        assert sorted(written) == sorted({"security_mode", *SECTIONS}), sorted(written)
        # A body that names no section changes nothing and checks no password.
        calls = w.verifier.calls
        now = _state(w.config)
        assert _put(w, nf_t.WRONG) == {"updated": False, "detail": "No sections provided"}
        assert w.verifier.calls == calls and _state(w.config) == now
        # The password comes before security.yaml is read: over a file that does not load, or a section that holds no
        # mapping, a wrong password is refused as anywhere else, and the right one meets the file's own refusal.
        for text in ("security_mode: daily\nheaders: [x_frame_options\n", "security_mode: daily\nheaders: deny\n"):
            w.config.write_text(text, encoding="utf-8")
            spoiled = _state(w.config)
            w.seen.audit.clear()
            with pytest.raises(HTTPException) as caught:
                _put(w, nf_t.WRONG, headers={"x_frame_options": "DENY"})
            assert caught.value.status_code == 403 and [event for event, _sev, _d in w.seen.audit] == [
                "ceremony_password_refused"], (text, caught.value.detail, w.seen.audit)
            with pytest.raises(HTTPException) as caught:
                _put(w, nf_t.RIGHT, headers={"x_frame_options": "DENY"})
            assert caught.value.status_code == 409 and _state(w.config) == spoiled, (text, caught.value.detail)
    finally:
        w.restore()
    # The synthetic single-user session stands for the owner: the change is made in the owner's name.
    (tmp_path / "local").mkdir()
    later = nf_t._Account("uid-later", "later", nf_t.RIGHT, role="user")
    w = _window(tmp_path / "local", accounts=nf_t._Accounts(later, nf_t.RESIDENT))
    try:
        assert _put(w, nf_t.RIGHT, {"sub": "local", "username": "local"}, jwt={"secure": True})["updated"] is True
        assert w.seen.audit[-1][2] == {"sections": ["jwt"], "user_id": "uid-resident"}, w.seen.audit
    finally:
        w.restore()
    # The limiter holds the account back: 429, nothing written.
    (tmp_path / "held").mkdir()
    w = _window(tmp_path / "held", limiter=nf_t._Limiter(allowed=False))
    try:
        w.config.write_text(TEXT, encoding="utf-8")
        before = _state(w.config)
        with pytest.raises(HTTPException) as caught:
            _put(w, nf_t.RIGHT, rate_limiting={"enabled": False})
        assert caught.value.status_code == 429 and _state(w.config) == before
    finally:
        w.restore()


# --------------------------------------------------------------------------- SY2


class _LookingOs:
    """The ``os`` module as ``routes_security`` sees it: every name the real one's, the paths ``open`` is asked
    recorded, and ``stat`` made to see a regular file at ``lie_at`` -- a FIFO swapped in after the look."""

    def __init__(self, real, lie_at=None, regular=None):
        self._real, self._lie_at, self._regular, self.opened = real, lie_at, regular, []

    def stat(self, path, *args, **kwargs):
        if self._lie_at is not None and os.fspath(path) == self._lie_at:
            return self._real.stat(self._regular)
        return self._real.stat(path, *args, **kwargs)

    def open(self, path, *args, **kwargs):
        self.opened.append(os.fspath(path))
        return self._real.open(path, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _spoil(kind, w, home):
    """security.yaml made one that does not load whole; the reason the routes give for it."""
    config = w.config
    if kind == "not_UTF-8":
        config.write_bytes(b"security_mode: daily\nheaders: \xff\xfe\n")
        return "security.yaml is there and does not load here (UnicodeDecodeError)"
    if kind == "not_YAML":
        config.write_text("security_mode: daily\nheaders: [x_frame_options\n", encoding="utf-8")
        return "security.yaml is there and does not load here ("
    if kind == "no_mapping":
        config.write_text("- security_mode\n- daily\n", encoding="utf-8")
        return "security.yaml is there and holds no mapping"
    if kind == "a_folder":
        config.mkdir()
        return "security.yaml is there and is not a regular file"
    if kind == "a_link_to_a_device":
        config.symlink_to(os.devnull)
        w.routes.os = _LookingOs(os)
        return "security.yaml is there and is not a regular file"
    if kind in ("a_FIFO_swapped_in_after_the_look", "a_folder_swapped_in_after_the_look"):
        if kind.startswith("a_FIFO"):
            os.mkfifo(config)
        else:
            config.mkdir()
        regular = home.parent / "regular.yaml"
        regular.write_text(TEXT, encoding="utf-8")
        w.routes.os = _LookingOs(os, lie_at=str(config), regular=str(regular))
        return "security.yaml is there and is not a regular file"
    if kind == "its_rights":
        config.write_text(TEXT, encoding="utf-8")
        os.chmod(config, 0)
        return "security.yaml is there and does not load here (PermissionError)"
    if kind == "its_folder_not_searchable":
        config.write_text(TEXT, encoding="utf-8")
        os.chmod(home, 0o600)
        return "security.yaml is there and does not load here (PermissionError)"
    raise AssertionError(kind)


_KINDS = ("not_UTF-8", "not_YAML", "no_mapping", "a_folder", "a_link_to_a_device", "a_FIFO_swapped_in_after_the_look",
          "a_folder_swapped_in_after_the_look", "its_rights", "its_folder_not_searchable")
# The section each writer changes, and what it writes there.
_SECTION_OF = {"put": ("rate_limiting", False), "setup": ("encryption", True), "remote_on": ("remote_access", True),
               "remote_off": ("remote_access", False)}


def _writers(w):
    """The four routes that write security.yaml back, each as its client asks it."""
    routes = w.routes
    return {
        "put": lambda: _put(w, nf_t.RIGHT, rate_limiting={"enabled": False}),
        "setup": lambda: routes.setup_encryption(routes.EncryptionSetupRequest(mode="random")),
        "remote_on": lambda: asyncio.run(routes.remote_access_enable(
            _request(), routes.RemoteAccessEnableRequest(passphrase="a passphrase of twelve", confirm=True))),
        "remote_off": lambda: asyncio.run(routes.remote_access_disable(_request())),
    }


def _held_open(path):
    """How many of this process's descriptors stand open on ``path``."""
    held = 0
    for name in os.listdir("/proc/self/fd"):
        with contextlib.suppress(OSError):
            held += os.readlink(f"/proc/self/fd/{name}") == str(path)
    return held


def _bounded(call, fifo):
    """``call`` on a thread of its own, joined for ten seconds: its answer or its exception, and whether it was still
    waiting -- then released, a FIFO at ``fifo`` given a writer that leaves at once."""
    out = {}

    def run():
        try:
            out["answer"] = call()
        except BaseException as exc:  # noqa: BLE001 - the caller reads it
            out["raised"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(10)
    waited = thread.is_alive()
    if waited:
        with contextlib.suppress(OSError):
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        thread.join(10)
    return out, waited


_CASES = [(f"{writer}_over_{kind}", writer, kind) for writer in ("put", "setup", "remote_on", "remote_off")
          for kind in _KINDS] + [(f"{writer}_over_a_section_that_holds_no_mapping", writer, "section")
                                 for writer in ("put", "setup", "remote_on", "remote_off")]


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_sy2_a_security_yaml_that_does_not_load_is_left_as_it_is(tmp_path, case):
    import yaml
    from fastapi import HTTPException

    _name, writer, kind = case
    if kind.startswith("its_") and os.geteuid() == 0:
        pytest.skip("root reads past the rights this case takes away")
    home = tmp_path / "config"
    home.mkdir()
    w = _window(home)
    try:
        if kind == "section":
            name, _value = _SECTION_OF[writer]
            w.config.write_text(f"security_mode: bulbe\n{name}: off\n", encoding="utf-8")
            reason = f"The {name} section of security.yaml holds no mapping: it is left as it is."
        else:
            reason = _spoil(kind, w, home)
        try:
            os.chmod(home, 0o700)
            before = _state(w.config)
            if kind == "its_folder_not_searchable":
                os.chmod(home, 0o600)
            out, waited = _bounded(_writers(w)[writer], str(w.config))
            held = _held_open(w.config)
        finally:
            os.chmod(home, 0o700)
        after = _state(w.config)
        assert not waited, "the writer waited on what is at security.yaml's name"
        assert held == 0, ("no descriptor of what is at security.yaml's name stays open", held)
        refused = out.get("raised")
        assert isinstance(refused, HTTPException), out
        assert refused.status_code == 409, (refused.status_code, refused.detail)
        assert refused.detail.startswith(reason), (reason, refused.detail)
        assert after == before, "the file is left as it was"
        assert sorted(os.listdir(home)) == ["security.yaml"], sorted(os.listdir(home))
        assert (w.seen.keys, w.seen.tls, w.seen.sealed, w.seen.chained) == ([], [], [], []), w.seen
        if kind == "a_link_to_a_device":
            assert w.routes.os.opened == [], ("no device is opened", w.routes.os.opened)
        if kind != "section":
            # The configuration route says the reason too, and shows nothing for a file it could not load.
            try:
                if kind == "its_folder_not_searchable":
                    os.chmod(home, 0o600)
                out, waited = _bounded(w.routes.get_security_config, str(w.config))
            finally:
                os.chmod(home, 0o700)
            shown = out.get("answer")
            assert not waited and shown is not None, out
            assert shown["loaded"] is False and shown["config"] == {}, shown
            assert shown["detail"].startswith(reason) and "left as it is" in shown["detail"], shown
    finally:
        os.chmod(home, 0o700)
        w.restore()
    # No file, an empty one, one of comments only and a null section: each writer writes as before, the section a
    # mapping holding what it changed.
    name, value = _SECTION_OF[writer]
    for label, start in (("none", None), ("empty", b""), ("comments", b"# nothing set yet\n"),
                         ("null", f"security_mode: bulbe\n{name}:\n".encode())):
        place = tmp_path / label
        place.mkdir()
        w = _window(place)
        try:
            if start is not None:
                w.config.write_bytes(start)
            _writers(w)[writer]()
            written = yaml.safe_load(w.config.read_text(encoding="utf-8"))
            assert written[name] == {"enabled": value}, (label, written)
            assert written.get("security_mode") == ("bulbe" if label == "null" else None), (label, written)
        finally:
            w.restore()


# --------------------------------------------------------------------------- SY3


def test_sy3_turning_encryption_on_through_the_route_seals_the_run_journal_at_once(tmp_path, monkeypatch, caplog):
    """rj7, whole: its route step now gives the owner's password, the check stood in for by one that passes it
    alone (SY1 holds the real one), and records that it was asked."""
    from fastapi import HTTPException

    asked = []
    built = rj_t._world

    class _Asking:
        """The security routes as rj7 calls them, the owner's password given, the session the owner's."""

        def __init__(self, sec):
            self._sec = sec

        def SecurityConfigUpdate(self, **sections):
            return self._sec.SecurityConfigUpdate(password=nf_t.RIGHT, **sections)

        def update_security_config(self, update):
            return _route_put(self._sec.update_security_config, update, nf_t.SESSION)

        def __getattr__(self, name):
            return getattr(self._sec, name)

    def _password(ceremony, current, password, request):
        asked.append((ceremony, password, current.get("sub")))
        if password != nf_t.RIGHT:
            raise HTTPException(status_code=403, detail="not the owner's password")
        return current["sub"]

    def asking_world(*args, **kwargs):
        world = built(*args, **kwargs)
        if world.sec is not None:
            world.sec._ceremony_password = _password
            world.sec = _Asking(world.sec)
        return world

    monkeypatch.setattr(rj_t, "_world", asking_world)
    rj_t.test_rj7_sealed_and_never_in_plaintext_where_it_must_not_be(tmp_path, monkeypatch, caplog)
    assert asked == [("security_config", nf_t.RIGHT, nf_t.SESSION["sub"])], asked


# --------------------------------------------------------------------------- SY4


def test_sy4_the_security_settings_are_written_whole_by_the_owners_password(tmp_path):
    from fastapi import HTTPException

    by_mode_dir, by_route_dir = tmp_path / "mode", tmp_path / "route"
    by_mode_dir.mkdir()
    by_route_dir.mkdir()
    loaded, restore = isolate(targets={rj_t._MODE: source("security_mode.py")})
    try:
        mode = loaded[rj_t._MODE]
        mode._SECURITY_YAML = by_mode_dir / "security.yaml"
        by_mode = rj_t._replaced(mode._SECURITY_YAML, lambda: mode._write_yaml_mode("daily"))
    finally:
        restore()
    w = _window(by_route_dir)
    try:
        refused = []

        def wrong_then_right():
            with pytest.raises(HTTPException) as caught:
                _put(w, nf_t.WRONG, headers={"x_frame_options": "DENY"})
            refused.append((caught.value.status_code, w.config.read_text(encoding="utf-8")))
            _put(w, nf_t.RIGHT, headers={"x_frame_options": "DENY"})

        by_route = rj_t._replaced(w.config, wrong_then_right)
    finally:
        w.restore()
    assert refused == [(403, rj_t._SETTINGS_TEXT)], ("a wrong password leaves the file as it was", refused)
    assert by_mode == (True, 0o640, {"encryption": {"enabled": True}, "security_mode": "daily"},
                       ["security.yaml"]), ("the security mode replaces the file whole, its mode kept", by_mode)
    assert by_route == (True, 0o640, {"encryption": {"enabled": True}, "security_mode": "bulbe",
                                      "headers": {"x_frame_options": "DENY"}}, ["security.yaml"]), (
        "the security routes replace the file whole, its mode kept", by_route)
