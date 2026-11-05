#!/usr/bin/env python3
"""Contracts for the plugin installed from a URL, which asks its digest, and for registration, closed past the first
account (block 229; D7 and D5 of the security census, decided by Leon on 3 November).

A plugin installed from a URL took no digest as shipped (``install.require_hash: false``): whoever served the URL,
or any URL it redirected to, chose the code; a digest of the wrong shape was found wrong only after the download.
Registration was open as shipped and allowed in Bulbe: anyone who reached the server made an account, and a second
account turned single-user mode off for good. The accounts were counted on one connection and written on another,
so accounts sent together all passed ``users.max_users``.

  * DK1 -- THE DIGEST IS ASKED AS SHIPPED: the shipped ``plugin_marketplace.yaml`` and the installer's defaults ask
    one; an install from a URL given none is refused before any request, in words that name the setting, nothing
    written; with ``require_hash: false`` it installs as before.
  * DK2 -- A DIGEST OF THE WRONG SHAPE IS REFUSED BEFORE ANY REQUEST, whether one is asked or not: too short, too
    long, prefixed, not hexadecimal; a right one in capitals or with spaces around installs; a wrong one of the
    right shape fails as a mismatch after the download, nothing installed.
  * DK3 -- in place of og2 of the outbound gate suite, which read a configuration without the key as "no digest
    asked": og2 whole, its two clauses on a partial configuration given the archive's digest.
  * DK4 -- WHO REGISTERS: the first account of a store that holds none, in Daily and in Bulbe, the configuration
    closed; past it, refused (403) before the password is hashed, in any mode, unless the configuration opens
    registration, the mode is Daily and the store is below ``users.max_users``; never in Bulbe or a mode that cannot
    be read; a configuration that does not load whole refuses the first account too; a store that cannot be read
    refuses (503); a registration that finds the first account made meanwhile, or the mode raised to Bulbe while
    its password was hashed, is refused as closed (the mode read again when the account is written); a first
    account the store refuses for its own rules is a 400, not a closed registration. In place of en5 of the Bulbe
    enrolment suite: the first
    account registered in Bulbe answers the enrolment and enrols through it, in Daily with registration open the
    tokens come at once, and a two-factor module that cannot be reached, or a factor that cannot be set up in Bulbe,
    refuses it (503) with no account left behind.
  * DK5 -- ACCOUNTS SENT TOGETHER ARE COUNTED ONE AFTER THE OTHER, the real account store: two registrations of a
    first account sent together on an empty store make one account, and two accounts sent together under
    ``max_users: 1`` make one (a rendezvous at the count); ``first_only`` given as a callable is asked after the
    count; a password of more than 72 bytes is refused (None, nothing written), one of 72 is not, and so is a new
    one by ``change_password``; a ``users.max_users`` that is no whole number reads 1 at load, the first account
    made, the second refused, never a raise.
  * DK6 -- THE OWNER ADDS AN ACCOUNT (``POST /api/auth/users``): with the owner's password, through the owner's own
    session or the single-user session, the role the configuration gives or one asked; refused for another account
    (403), a wrong password (403, counted), a password being checked (429), a role outside admin, user and viewer
    (400, before the password is asked), and in Bulbe where no factor can be set up (503); no session given; audited.
    The owner lists and removes accounts and reads the audit, which took an admin's token: another account cannot,
    nor an admin's token whose account the store does not name the owner; under the single-user session the add and
    the removal are audited under the owner's account, and the owner cannot remove its own.
  * DK7 -- THE STATUS SAYS WHAT THE REGISTRATION DOES, in each case of DK4.
  * DK8 -- in place of uc2 of the unreadable configuration suite, which read an install with no file as registration
    open: uc2 whole, no file reading registration closed.
  * DK9 -- THE TEXTS: the shipped files, the route's field, the UI's field and form, the register page, the docs and
    the changelog say what the code does, and nothing says the old defaults.
  * DK10 -- in place of a36 of the auth core suite: its walk word for word with one site more, the write
    transaction an account is made in (``BEGIN IMMEDIATE``), whose count of the accounts comes after it and before
    the write, the count that stood before it gone.

The installer runs in the outbound gate suite's window (its page fetch, gate and network stand-ins), the routes in
the second-factor suite's window (the real second-factor module, the real login, security and auth routes, the seam
suite's in-memory store), the account store in the auth core suite's window on files under a temporary folder.
Nothing reads or writes the platform's data.

Local-only (the public distribution ships no tests).
"""

import ast
import hashlib
import os
import sqlite3
import sys
import tempfile
import threading
import types
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_auth_config_unread_closes_the_door_contracts as uc  # noqa: E402
import test_auth_core_contracts as ac_t  # noqa: E402
import test_outbound_gate_contracts as og  # noqa: E402
import test_two_factor_account_contracts as tf  # noqa: E402
from _isolation import isolate, source  # noqa: E402

A, B, PW_A, PW_B = tf.A, tf.B, tf.PW_A, tf.PW_B
REGISTER, STATUS, USERS = "/api/auth/register", "/api/auth/status", "/api/auth/users"
BEGIN, COMPLETE = "/api/auth/login/2fa/enrol/begin", "/api/auth/login/2fa/enrol/complete"
_ROOT = source().parent

BUDGET_S = {
    "test_dk1_the_digest_is_asked_as_shipped": 5.0,
    "test_dk2_a_digest_of_the_wrong_shape_is_refused_before_any_request": 5.0,
    "test_dk3_the_marketplace_install_is_gated_destination_checked_and_configured": 10.0,
    "test_dk4_who_registers": 6.0,
    "test_dk5_accounts_sent_together_are_counted_one_after_the_other": 12.0,
    "test_dk6_the_owner_adds_an_account": 6.0,
    "test_dk7_the_status_says_what_the_registration_does": 6.0,
    "test_dk8_a_file_that_loads_or_none": 5.0,
    "test_dk9_the_texts": 2.0,
    "test_dk10_sql_is_parameterised_never_f_string_the_accounts_written_in_one_transaction": 2.0,
}


def _read(*parts):
    return _ROOT.joinpath(*parts).read_text(encoding="utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- DK1


def test_dk1_the_digest_is_asked_as_shipped(tmp_path):
    import yaml

    shipped = _read("opti_oignon", "config", "plugin_marketplace.yaml")
    assert yaml.safe_load(shipped)["install"]["require_hash"] is True, "the shipped file asks no digest"
    archive = og._zip_archive()
    url = "https://plugins.example.org/p.zip"
    with og._plugin_world(tmp_path) as w:
        pi = w.mods[og._PI]
        assert pi._INSTALL_DEFAULTS["require_hash"] is True, "the installer's default asks no digest"
        _serve = og._serve
        _serve(w, url, archive)
        for name, config in (("the shipped file", shipped), ("no install section", "index:\n  url: ''\n"),
                             ("an empty file", "")):
            og._reset(w)
            before = og._tree(w.plugins)
            result = og._installer(w, config).install_from_url(url)
            assert result["success"] is False, (name, result)
            assert w.opener.calls == [] and w.net.quiet() and w.gate == [], (name, "a request was made")
            assert "install.require_hash is true" in (result["error"] or ""), (name, result["error"])
            assert og._tree(w.plugins) == before and og._tree(w.scratch) == [], name
            for blank in ("   ", "\n"):
                assert og._installer(w, config).install_from_url(url, expected_sha256=blank)["success"] is False
            assert w.opener.calls == [] and w.net.quiet(), (name, "a blank digest made a request")
        og._reset(w)
        result = og._installer(w, "install:\n  require_hash: false\n").install_from_url(url)
        assert result["success"] is True and result["name"] == og._PLUGIN, result
        og._reset(w)
        result = og._installer(w, shipped).install_from_url(url, expected_sha256=_sha(archive))
        assert result["success"] is True, ("the shipped file with the archive's digest", result)
        assert (w.plugins / og._PLUGIN / "payload.bin").read_bytes() == og._PAYLOAD


# --------------------------------------------------------------------------- DK2


def test_dk2_a_digest_of_the_wrong_shape_is_refused_before_any_request(tmp_path):
    archive = og._zip_archive()
    digest = _sha(archive)
    url = "https://plugins.example.org/p.zip"
    wrong_shapes = {
        "too short": digest[:63],
        "too long": digest + "0",
        "prefixed": "sha256:" + digest,
        "not hexadecimal": "g" * 64,
        "a space inside": digest[:32] + " " + digest[33:],
        "fullwidth digits": "\uff10" * 64,
    }
    with og._plugin_world(tmp_path) as w:
        og._serve(w, url, archive)
        for key in ("true", "false"):
            config = f"install:\n  require_hash: {key}\n"
            for name, given in wrong_shapes.items():
                og._reset(w)
                before = og._tree(w.plugins)
                result = og._installer(w, config).install_from_url(url, expected_sha256=given)
                assert result["success"] is False, (key, name, result)
                assert w.opener.calls == [] and w.net.quiet() and w.gate == [], (key, name, "a request was made")
                assert "64 hexadecimal characters" in (result["error"] or ""), (key, name, result["error"])
                assert og._tree(w.plugins) == before and og._tree(w.scratch) == [], (key, name)
        for given in (digest.upper(), f"  {digest}\n", f"\t{digest.upper()} "):
            og._reset(w)
            shutil_free = og._installer(w, "install:\n  require_hash: true\n").install_from_url(
                url, expected_sha256=given)
            assert shutil_free["success"] is True, (repr(given), shutil_free)
        og._reset(w)
        before = og._tree(w.plugins)
        for leftover in ("payload.bin", "plugin.py", "manifest.yaml"):
            assert (w.plugins / og._PLUGIN / leftover).exists()
        (w.plugins / og._PLUGIN / "payload.bin").write_bytes(b"installed before")
        before = og._tree(w.plugins)
        result = og._installer(w, "install:\n  require_hash: true\n").install_from_url(url, expected_sha256="0" * 64)
        assert result["success"] is False and "Hash mismatch" in (result["error"] or ""), result
        assert len(w.fetches) == 1, "a right-shaped digest is checked against the download"
        assert og._tree(w.plugins) == before, "a mismatch installed something"
        assert (w.plugins / og._PLUGIN / "payload.bin").read_bytes() == b"installed before"
        assert og._tree(w.scratch) == [], "the download was left behind"


# --------------------------------------------------------------------------- DK3


def test_dk3_the_marketplace_install_is_gated_destination_checked_and_configured(tmp_path):
    _installer, _serve, _reset, _tree = og._installer, og._serve, og._reset, og._tree
    _texts, _refused, _guard, _called_names = og._texts, og._refused, og._guard, og._called_names
    _PLUGIN, _PAYLOAD, _MIB = og._PLUGIN, og._PAYLOAD, og._MIB
    archive = og._zip_archive()
    with og._plugin_world(tmp_path) as w:
        installer = _installer(w)

        # c1 -- outside Daily, switch engaged, switch unreadable: refused in
        # the front door's words, before any request, nothing written.
        url = "https://example.org/p.zip"
        for state in og._REFUSAL_STATES:
            with _refused(w, state) as refusal:
                _reset(w)
                before = _tree(w.plugins)
                result = installer.install_from_url(url)
                assert w.opener.calls == [] and w.net.quiet(), (state, w.opener.calls, w.net.lookups, w.net.connections)
                assert result["success"] is False, (state, result)
                assert result["error"] == _texts("Plugin install")[refusal], (state, result["error"])
                assert _tree(w.plugins) == before and _tree(w.scratch) == [], (state, _tree(w.scratch))
        _reset(w)
        installer.install_from_url(url)
        assert len(w.opener.calls) + len(w.net.lookups) >= 1, "in Daily the install reaches the network layer"

        # c2 -- in Daily, every destination the page fetch refuses is refused,
        # named, with no connection to the refused address and nothing written.
        for literal in ("http://127.0.0.1/p.zip", "http://[::1]/p.zip", "http://10.0.0.1/p.zip",
                        "http://100.64.0.1/p.zip", "http://[::ffff:127.0.0.1]/p.zip"):
            _serve(w, literal, archive)
        _serve(w, "https://lan.example.org/p.zip", archive, address="192.168.1.2")
        _serve(w, "https://moved.example.org/p.zip", status=302, location="http://127.0.0.1/")
        _serve(w, "http://127.0.0.1/", archive)
        for target, why, address in og._REFUSED_DESTINATIONS:
            _reset(w)
            before = _tree(w.plugins)
            result = installer.install_from_url(target)
            reached = {host for host, _port in w.net.connections}
            assert w.opener.calls == [] and address not in reached, (target, w.opener.calls, reached)
            if address is None:
                assert w.net.quiet(), (target, w.net.lookups, w.net.connections)
            assert result["success"] is False, (target, result)
            assert why in (result["error"] or ""), (target, result["error"])
            assert _tree(w.plugins) == before and _tree(w.scratch) == [], target

        # c4 -- the GitHub rewrite reads the host, not a substring of the URL.
        main = "https://github.com/u/r/archive/refs/heads/main.zip"
        assert installer._normalize_url("https://github.com/u/r") == main
        for look_alike in ("https://evil.example/github.com/x", "https://github.com.evil.example/u/r"):
            assert installer._normalize_url(look_alike) == look_alike, look_alike

        # c5 -- witness, Daily: a served archive installs, it reached the
        # installer as bytes, and the gate was asked.
        _reset(w)
        _serve(w, "https://plugins.example.org/p.zip", archive)
        result = installer.install_from_url("https://plugins.example.org/p.zip")
        assert result["success"] is True and result["name"] == _PLUGIN, result
        assert (w.plugins / _PLUGIN / "payload.bin").read_bytes() == _PAYLOAD
        assert len(w.gate) >= 1, "the install asked the web gate"
        assert len(w.fetches) == 1, [f.url for f in w.fetches]
        assert w.fetches[0].page.text == "" and w.fetches[0].page.body == archive, "the archive was decoded as text"

        # c7 -- a release asset redirected to a signed URL keeps its name.
        _reset(w)
        signed = "https://cdn.example.org/x?X-Sig=" + "a" * 400
        _serve(w, "https://example.org/r/v1/p.tar.gz", status=302, location=signed)
        _serve(w, signed, og._tar_archive())
        result = installer.install_from_url("https://example.org/r/v1/p.tar.gz")
        assert result["success"] is True, result
        assert (w.plugins / _PLUGIN / "payload.bin").read_bytes() == _PAYLOAD

        # c3 -- the configuration is read, key by key; a configuration that
        # does not give require_hash asks a digest, so the two clauses that
        # reach the download give the archive's.
        big = og._zip_archive(padding=2 * _MIB)
        _serve(w, "https://big.example.org/p.zip", big)
        _reset(w)
        before = _tree(w.plugins)
        result = _installer(w, "install:\n  allow_remote_install: false\n").install_from_url(
            "https://plugins.example.org/p.zip")
        assert result["success"] is False and w.opener.calls == [] and w.net.quiet(), result
        assert "allow_remote_install" in (result["error"] or ""), result["error"]
        assert w.gate == [], "remote install off is refused before the gate is asked"
        result = _installer(w, "install:\n  max_download_size_mb: 1\n").install_from_url(
            "https://big.example.org/p.zip", expected_sha256=_sha(big))
        assert result["success"] is False and "too large" in (result["error"] or "").lower(), result
        assert _tree(w.plugins) == before
        _reset(w)
        result = _installer(w, "install:\n  require_hash: true\n").install_from_url(
            "https://plugins.example.org/p.zip")
        assert result["success"] is False and w.opener.calls == [] and w.net.quiet(), result
        low = (result["error"] or "").lower()
        assert "hash" in low or "sha256" in low or "digest" in low, result["error"]
        _reset(w)
        _installer(w, "install:\n  timeout_s: 7\n").install_from_url(
            "https://plugins.example.org/p.zip", expected_sha256=_sha(archive))
        assert [f.kwargs.get("timeout") for f in w.fetches] == [7], [f.kwargs for f in w.fetches]

    # c6 -- the installer holds no sink and fetches through the page fetch.
    text = source("plugin_installer.py").read_text(encoding="utf-8")
    guard, restore = _guard()
    try:
        assert guard.count_sinks("import urllib.request\n\ndef f(u):\n    return urllib.request.urlopen(u)\n") >= 1
        assert guard.count_sinks(text) == 0, guard.sink_sites(text)
    finally:
        restore()
    assert "fetch_page" in _called_names(ast.parse(text))


# --------------------------------------------------------------------------- the routes' world


@pytest.fixture
def world():
    """A window of the routes, in Daily unless told; one window at a time, the last closed at teardown."""
    made = []

    def make(people=((A, "alice", PW_A), (B, "bob", PW_B)), **kwargs):
        while made:
            made.pop().close()
        kwargs.setdefault("bulbe", False)
        accounts = tf._Accounts(*people)
        accounts.config["users"]["allow_registration"] = False
        w = tf._world(accounts=accounts, **kwargs)
        w.while_hashed, w.refuse_rules = None, False
        w.accounts.get_audit_log = lambda user_id=None, project_id=None, limit=50: list(w.accounts.audit)[-limit:]
        made.append(w)
        w.made, w.asked = _with_create_user(w)
        return w

    yield make
    while made:
        made.pop().close()


def _with_create_user(w):
    """The account store's ``create_user``, as the routes call it: ``first_only`` makes an account only in a store
    that holds none, as the real store does in its transaction. ``asked`` keeps each call (a call is a hash made)."""
    made, asked = [], []

    def create_user(username, password, email="", role=None, metadata=None, first_only=False):
        if w.while_hashed is not None:
            w.while_hashed()
        first_only = first_only() if callable(first_only) else first_only
        asked.append((username, first_only))
        if first_only and w.accounts.order:
            return None
        if any(u.username == username for u in w.accounts.users.values()) or w.refuse_rules:
            return None
        user_id = f"uid-{username}"
        w.accounts.add(user_id, username, password, role or "user")
        made.append(user_id)
        return w.accounts.users[user_id]

    w.accounts.create_user = create_user
    return made, asked


def _mode_unreadable():
    def is_bulbe():
        raise RuntimeError("the mode lock cannot be read")

    sys.modules["opti_oignon.security_mode"].is_bulbe = is_bulbe


def _register(w, name):
    return tf._post(w, REGISTER, {"username": name, "password": f"{name}-own-password"})


def _status(w):
    answer = w.client.get(STATUS)
    assert answer.status_code == 200, answer.text
    return answer.json()["registration_enabled"]


# The cases of DK4 and DK7: (name, the store's people, Bulbe, registration open, how the mode reads, how the
# configuration loaded, what the registration answers).
_EMPTY = ()
_TWO = ((A, "alice", PW_A), (B, "bob", PW_B))
_CASES = (
    ("first account, Daily, closed", _EMPTY, False, False, "read", None, 200),
    ("first account, Bulbe, closed", _EMPTY, True, False, "read", None, 200),
    ("first account, mode unreadable, closed", _EMPTY, False, False, "unreadable", None, 200),
    ("first account, Daily, open", _EMPTY, False, True, "read", None, 200),
    ("second account, Daily, closed", _TWO, False, False, "read", None, 403),
    ("second account, Bulbe, closed", _TWO, True, False, "read", None, 403),
    ("second account, Daily, open", _TWO, False, True, "read", None, 200),
    ("second account, Daily, open, at max_users", _TWO, False, True, "read", None, 403),
    ("second account, Daily, open, below max_users", _TWO, False, True, "read", None, 200),
    ("second account, Bulbe, open", _TWO, True, True, "read", None, 403),
    ("second account, mode unreadable, open", _TWO, False, True, "unreadable", None, 403),
    ("first account, configuration unread", _EMPTY, False, True, "read", "it holds a list", 403),
    ("second account, configuration unread", _TWO, False, True, "read", "it holds a list", 403),
)


def _case(world, people, bulbe, opened, mode, unread, name=""):
    w = world(people=people, bulbe=bulbe)
    w.accounts.config["users"]["allow_registration"] = opened
    if "max_users" in name:
        w.accounts.config["users"]["max_users"] = 2 if "at max_users" in name else 3
    if unread:
        w.accounts.config_unread = unread
    if mode == "unreadable":
        _mode_unreadable()
    return w


# --------------------------------------------------------------------------- DK4


def test_dk4_who_registers(world):
    for name, people, bulbe, opened, mode, unread, status in _CASES:
        w = _case(world, people, bulbe, opened, mode, unread, name)
        held = list(w.accounts.order)
        answer = _register(w, "carol")
        assert answer.status_code == status, (name, answer.status_code, answer.text)
        if status == 403:
            assert w.made == [] and w.asked == [], (name, "an account was made, or a password hashed")
            assert w.accounts.order == held and w.accounts.tokens == {}, name
            detail = answer.json()["detail"]
            if unread:
                assert "auth configuration does not load" in detail, (name, detail)
            elif "max_users" in name:
                assert detail == "Registration is closed: the platform holds its users.max_users accounts.", detail
            else:
                assert detail == ("Registration is closed: the platform's owner adds accounts (Workshop > "
                                  "Security > Authentication mode)."), (name, detail)
        else:
            assert w.made == ["uid-carol"], (name, w.made)
            assert w.asked == [("carol", not (opened and not bulbe and mode == "read"))], (name, w.asked)

    # A store that cannot be read registers no one, past the configuration's say.
    w = world(people=_EMPTY)
    w.accounts.broken = True
    answer = _register(w, "carol")
    assert answer.status_code == 503 and "cannot be read" in answer.json()["detail"], answer.text
    assert w.asked == [], "a password was hashed for a store that cannot be read"

    # A registration that finds the first account made meanwhile is refused as closed.
    w = world(people=_EMPTY)
    made, asked = w.made, w.asked
    real = w.accounts.create_user

    def first_made_meanwhile(username, password, email="", role=None, metadata=None, first_only=False):
        w.accounts.add("uid-first", "first", "first-own-password")
        return real(username, password, email, role, metadata, first_only)

    w.accounts.create_user = first_made_meanwhile
    answer = _register(w, "carol")
    assert answer.status_code == 403 and "Registration is closed" in answer.json()["detail"], answer.text
    assert made == [] and asked == [("carol", True)] and w.accounts.order == ["uid-first"], (made, asked)

    # The mode raised to Bulbe while the password is hashed: read again when the account is written, refused.
    w = world()
    w.accounts.config["users"]["allow_registration"] = True

    def raised():
        w.mode.bulbe = True

    w.while_hashed = raised
    answer = _register(w, "carol")
    assert answer.status_code == 403 and "Registration is closed" in answer.json()["detail"], answer.text
    assert w.made == [] and w.asked == [("carol", True)], (w.made, w.asked)

    # A first account the store refuses for its own rules (a name, a password) is a 400, not a closed registration.
    w = world(people=_EMPTY)
    w.refuse_rules = True
    answer = _register(w, "carol")
    assert answer.status_code == 400, answer.text
    assert answer.json()["detail"] == ("Registration failed: the name may be taken, or the password refused by the "
                                       "rules.")
    assert w.made == [] and w.asked == [("carol", True)] and w.accounts.order == [], (w.made, w.asked)

    # In place of en5: the first account registered in Bulbe enrols before its first session.
    w = world(people=_EMPTY, bulbe=True)
    answer = _register(w, "carol")
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body.get("requires_2fa_enrolment") is True and "access_token" not in body, body
    assert w.made == ["uid-carol"] and w.accounts.tokens == {}, "the account made, no session"
    secret = tf._post(w, BEGIN, {"challenge_id": body["challenge_id"]}).json()["secret"]
    done = tf._post(w, COMPLETE, {"challenge_id": body["challenge_id"], "code": tf._code(w, secret)})
    assert done.status_code == 200 and done.json()["user_id"] == "uid-carol", done.text

    w = world(bulbe=False)
    w.accounts.config["users"]["allow_registration"] = True
    given = _register(w, "dave")
    assert given.status_code == 200 and given.json().get("access_token") == "tok-uid-dave", given.text

    for broken in ("module", "totp"):
        w = world(people=_EMPTY, bulbe=True)
        if broken == "module":
            w.routes._get_2fa_manager = lambda: None
        else:
            w.tfa.TOTP_AVAILABLE = False
        refused = _register(w, "erin")
        assert refused.status_code == 503 and "registration refused" in refused.json()["detail"], (broken,
                                                                                                    refused.text)
        assert w.made == [] and w.accounts.tokens == {}, (broken, "no account left behind")


# --------------------------------------------------------------------------- DK5


class _Rendezvous:
    """Holds each count of the accounts until a second one is read, or a second and a half: counts read together
    meet; a count that waits on another writer's transaction is read alone."""

    def __init__(self):
        self.barrier = None
        self.counts = 0


def _store_window(tmpdir, rendezvous):
    """The real auth module, its ``safe_connect`` on files under ``tmpdir``, each connection holding its count of
    the accounts at the rendezvous while it is armed."""

    class Connection(sqlite3.Connection):
        def execute(self, sql, *args):
            if rendezvous.barrier is not None and " ".join(sql.split()).startswith("SELECT COUNT(*) FROM users"):
                rendezvous.counts += 1
                try:
                    rendezvous.barrier.wait(timeout=1.5)
                except threading.BrokenBarrierError:
                    pass
            return super().execute(sql, *args)

    stub = types.ModuleType("opti_oignon.db_utils")

    def safe_connect(db_path, *, check_same_thread=True, timeout=5.0):
        target = os.path.join(tmpdir, hashlib.md5(str(db_path).encode()).hexdigest() + ".db")
        return sqlite3.connect(target, timeout=timeout, check_same_thread=check_same_thread, factory=Connection)

    stub.safe_connect = safe_connect
    loaded, restore = isolate(targets={ac_t._AUTH: source("auth.py")}, blocked=("ollama",),
                              seeded={"opti_oignon.db_utils": stub})
    return loaded[ac_t._AUTH], restore


def _together(rendezvous, calls):
    """Run ``calls`` at once, the rendezvous armed; their answers, in order."""
    answers = [None] * len(calls)
    rendezvous.barrier = threading.Barrier(len(calls))

    def run(i, call):
        answers[i] = call()

    threads = [threading.Thread(target=run, args=(i, call)) for i, call in enumerate(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    rendezvous.barrier = None
    assert not any(thread.is_alive() for thread in threads), "a call never came back"
    return answers


def test_dk5_accounts_sent_together_are_counted_one_after_the_other(tmp_path):
    tmpdir = tempfile.mkdtemp(prefix="dk5_", dir=tmp_path)
    rendezvous = _Rendezvous()
    auth, restore = _store_window(tmpdir, rendezvous)
    try:
        for name, users, first_only in (("first accounts", "  allow_registration: false\n", True),
                                        ("max_users: 1", "  allow_registration: true\n  max_users: 1\n", False)):
            config = tmp_path / f"{uuid.uuid4().hex}.yaml"
            config.write_text("single_user_mode: false\nusers:\n" + users + "password:\n  bcrypt_rounds: 4\n",
                              encoding="utf-8")
            manager = auth.AuthManager(config_path=config, db_path=os.path.join(tmpdir, uuid.uuid4().hex + ".db"))
            assert manager.count_users() == 0, name
            rendezvous.counts = 0
            answers = _together(rendezvous, [
                lambda: manager.create_user("carol", "carol-own-password", first_only=first_only),
                lambda: manager.create_user("dave", "dave-own-password", first_only=first_only),
            ])
            assert rendezvous.counts == 2, (name, "both calls did not reach the count", rendezvous.counts)
            assert sum(answer is not None for answer in answers) == 1, (name, "accounts sent together", answers)
            assert manager.count_users() == 1, (name, manager.count_users())
            assert manager.create_user("erin", "erin-own-password", first_only=first_only) is None, name
            assert manager.count_users() == 1, name
            made = next(answer for answer in answers if answer is not None)
            assert manager.owners() == [made.user_id], (name, "the account made is not the owner")
            assert manager.authenticate(made.username, f"{made.username}-own-password") is not None, name
        # A duplicate name inside the transaction leaves the store writable.
        config = tmp_path / "open.yaml"
        config.write_text("single_user_mode: false\nusers:\n  allow_registration: true\npassword:\n"
                          "  bcrypt_rounds: 4\n", encoding="utf-8")
        manager = auth.AuthManager(config_path=config, db_path=os.path.join(tmpdir, uuid.uuid4().hex + ".db"))
        assert manager.create_user("carol", "carol-own-password") is not None
        assert manager.create_user("carol", "carol-own-password") is None
        assert manager.create_user("dave", "dave-own-password") is not None, "a refused duplicate held the store"
        assert manager.count_users() == 2

        # first_only given as a callable is asked after the count, under the write lock, and only past the first.
        asked = []

        def first_only():
            asked.append(manager.count_users.__name__)
            return True

        manager = auth.AuthManager(config_path=config, db_path=os.path.join(tmpdir, uuid.uuid4().hex + ".db"))
        assert manager.create_user("carol", "carol-own-password", first_only=first_only) is not None
        assert asked == [], "asked on an empty store, where the first account is made whatever it says"
        assert manager.create_user("dave", "dave-own-password", first_only=first_only) is None
        assert asked == ["count_users"] and manager.count_users() == 1
        assert manager.create_user("dave", "dave-own-password", first_only=lambda: False) is not None

        # A password bcrypt does not read whole is refused as a rule, nothing written; 72 bytes are taken.
        manager = auth.AuthManager(config_path=config, db_path=os.path.join(tmpdir, uuid.uuid4().hex + ".db"))
        for long in ("a" * 73, "\u00e9" * 37):
            assert manager.create_user("carol", long) is None, len(long.encode("utf-8"))
        assert manager.count_users() == 0
        made = manager.create_user("carol", "b" * 72)
        assert made is not None and manager.authenticate("carol", "b" * 72) is not None
        assert manager.change_password(made.user_id, "c" * 73) is False
        assert manager.change_password(made.user_id, "\u00e9" * 37) is False
        assert manager.authenticate("carol", "b" * 72) is not None, "a refused change changed the password"
        assert manager.change_password(made.user_id, "d" * 72) is True
        assert manager.authenticate("carol", "d" * 72) is not None

        # A cap that is no whole number reads 1 at load, for the route and the store alike: never a raise.
        for cap in ("2.0", "true", "'2'", "null", "[2]"):
            config = tmp_path / f"{uuid.uuid4().hex}.yaml"
            config.write_text("single_user_mode: false\nusers:\n  allow_registration: true\n  max_users: "
                              f"{cap}\npassword:\n  bcrypt_rounds: 4\n", encoding="utf-8")
            manager = auth.AuthManager(config_path=config, db_path=os.path.join(tmpdir, uuid.uuid4().hex + ".db"))
            assert manager.config["users"]["max_users"] == 1, (cap, manager.config["users"])
            assert manager.create_user("carol", "carol-own-password") is not None, cap
            assert manager.create_user("dave", "dave-own-password") is None, cap
            assert manager.count_users() == 1, cap
        config.write_text("single_user_mode: false\nusers:\n  allow_registration: true\n  max_users: 3\n",
                          encoding="utf-8")
        assert auth.AuthManager(config_path=config, db_path=os.path.join(tmpdir, uuid.uuid4().hex + ".db")).config[
            "users"]["max_users"] == 3, "a whole number was not kept"
    finally:
        restore()


# --------------------------------------------------------------------------- DK6


def _add(w, body):
    return tf._post(w, USERS, body)


def _claims():
    return sys.modules["opti_oignon.ceremony_factors"]


def test_dk6_the_owner_adds_an_account(world):
    w = world()
    tf._as(w, A)
    answer = _add(w, {"username": "carol", "password": "carol-own-password", "owner_password": PW_A})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["user_id"] == "uid-carol" and body["role"] == "user", body
    assert "access_token" not in body and "set-cookie" not in answer.headers, "a session was given"
    assert w.made == ["uid-carol"] and w.asked == [("carol", False)] and w.accounts.tokens == {}
    assert ("uid-alice", "create_user", "user", "uid-carol") in w.accounts.audit, w.accounts.audit

    # An admin's account made here is the owner from then on (``owners``: the admins', or the first registered
    # while none is an admin's): the first registered adds no more.
    answer = _add(w, {"username": "dave", "password": "dave-own-password", "owner_password": PW_A, "role": "admin"})
    assert answer.status_code == 200 and answer.json()["role"] == "admin", answer.text
    assert w.accounts.owners() == ["uid-dave"], w.accounts.owners()
    w.accounts.config["users"]["default_role"] = "viewer"
    answer = _add(w, {"username": "erin", "password": "erin-own-password", "owner_password": PW_A})
    assert answer.status_code == 403 and w.made == ["uid-carol", "uid-dave"], answer.text
    tf._as(w, "uid-dave")
    answer = _add(w, {"username": "erin", "password": "erin-own-password", "owner_password": "dave-own-password"})
    assert answer.status_code == 200 and answer.json()["role"] == "viewer", answer.text

    # Refusals: nothing made.
    w = world()
    for who, body, status, words in (
        (B, {"owner_password": PW_B}, 403, "owner"),
        (A, {"owner_password": "not-alice-password"}, 403, "password"),
        (A, {"owner_password": PW_A, "role": "root"}, 400, "admin, user or viewer"),
        (A, {"owner_password": PW_A, "role": "Admin"}, 400, "admin, user or viewer"),
    ):
        tf._as(w, who)
        asked = len(w.limiter.asked)
        failures = len(w.limiter.failures)
        answer = _add(w, {"username": "carol", "password": "carol-own-password", **body})
        assert answer.status_code == status, (who, body, answer.text)
        assert words in answer.json()["detail"], (who, body, answer.json()["detail"])
        assert w.made == [] and w.asked == [], (who, body, "an account was made")
        if status == 400:
            assert len(w.limiter.asked) == asked, "the password was asked before the role was read"
        if body["owner_password"] == "not-alice-password":
            assert len(w.limiter.failures) == failures + 1, "a wrong owner password was not counted"
    tf._as(w, A)
    assert _claims()._claim(A)
    try:
        answer = _add(w, {"username": "carol", "password": "carol-own-password", "owner_password": PW_A})
    finally:
        _claims()._release(A)
    assert answer.status_code == 429 and w.made == [], answer.text

    # The owner lists and removes accounts (an owner by the store, its token's role "user"); another account cannot.
    w = world()
    tf._as(w, A)
    assert _add(w, {"username": "carol", "password": "carol-own-password", "owner_password": PW_A}).status_code == 200
    listed = w.client.get(USERS)
    assert listed.status_code == 200 and [u["user_id"] for u in listed.json()] == [A, B, "uid-carol"], listed.text
    assert w.client.get("/api/auth/audit").status_code == 200, "the owner cannot read the audit"
    tf._as(w, B, role="admin")
    assert w.client.get(USERS).status_code == 403, "a token's role made an owner"
    assert w.client.delete(f"{USERS}/uid-carol").status_code == 403, "a token's role removed an account"
    assert w.client.get("/api/auth/audit").status_code == 403, "a token's role read the audit"
    tf._as(w, A)
    gone = w.client.delete(f"{USERS}/uid-carol")
    assert gone.status_code == 200 and "uid-carol" not in w.accounts.users, gone.text
    assert ("uid-alice", "delete_user", "user", "uid-carol") in w.accounts.audit, w.accounts.audit
    assert w.client.delete(f"{USERS}/{A}").status_code == 400 and A in w.accounts.users

    # The single-user session stands for the owner when there is one.
    w = world(people=((A, "alice", PW_A),))
    w.current = {"sub": "local", "username": "local", "role": "admin", "type": "access"}
    answer = _add(w, {"username": "carol", "password": "carol-own-password", "owner_password": PW_A})
    assert answer.status_code == 200 and w.made == ["uid-carol"], answer.text
    assert ("uid-alice", "create_user", "user", "uid-carol") in w.accounts.audit, w.accounts.audit
    gone = w.client.delete(f"{USERS}/uid-carol")
    assert gone.status_code == 200 and ("uid-alice", "delete_user", "user", "uid-carol") in w.accounts.audit, (
        gone.text, w.accounts.audit)
    mine = w.client.delete(f"{USERS}/{A}")
    assert mine.status_code == 400 and A in w.accounts.users, ("the single-user session removed the owner", mine.text)
    w = world(people=_EMPTY)
    w.current = {"sub": "local", "username": "local", "role": "admin", "type": "access"}
    answer = _add(w, {"username": "carol", "password": "carol-own-password", "owner_password": "anything-at-all"})
    assert answer.status_code == 403 and w.made == [], ("an install with no account adds none", answer.text)

    # In Bulbe: made where a factor can be set up, the account enrolling at its first login; refused where none can.
    w = world(bulbe=True)
    tf._as(w, A)
    answer = _add(w, {"username": "carol", "password": "carol-own-password", "owner_password": PW_A})
    assert answer.status_code == 200 and w.made == ["uid-carol"], answer.text
    login = tf._post(w, "/api/auth/login", {"username": "carol", "password": "carol-own-password"})
    assert login.status_code == 200 and login.json().get("requires_2fa_enrolment") is True, login.text
    w = world(bulbe=True)
    w.tfa.TOTP_AVAILABLE = False
    tf._as(w, A)
    answer = _add(w, {"username": "carol", "password": "carol-own-password", "owner_password": PW_A})
    assert answer.status_code == 503 and w.made == [] and w.asked == [], answer.text


# --------------------------------------------------------------------------- DK7


def test_dk7_the_status_says_what_the_registration_does(world):
    for name, people, bulbe, opened, mode, unread, status in _CASES:
        w = _case(world, people, bulbe, opened, mode, unread, name)
        said = _status(w)
        assert said is (status == 200), (name, said, status)
        assert _register(w, "carol").status_code == status, name
        if status == 200:
            assert _status(w) is (opened and not bulbe and mode == "read" and not unread
                                  and "max_users" not in name), (name, "the status after the account")


# --------------------------------------------------------------------------- DK8


def test_dk8_a_file_that_loads_or_none(tmp_path, caplog):
    import logging

    auth, restore, _counter, tmpdir = ac_t._load()
    seen = {}
    try:
        files = {
            "single-user on": "single_user_mode: true\nusers:\n  allow_registration: true\n",
            "single-user off": uc._CLOSED_KEYS,
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
                manager = uc._manager(auth, tmpdir, path)
            seen[name] = (uc._state(manager), len(uc._said(caplog, "gives no")), manager.config)
        seen["no file"] = (uc._state(uc._manager(auth, tmpdir, tmp_path / "absent.yaml")), 0, None)
    finally:
        restore()
    expected = {
        "single-user on": ((True, True, None), 0),
        "single-user off": ((False, False, None), 0),
        "no file": ((True, False, None), 0),
        "empty": ((False, False, None), 2),
        "cut before its settings": ((False, False, None), 2),
        "no registration setting": ((True, False, None), 1),
        "a misspelt false": ((False, False, None), 1),
        "a quoted false": ((False, False, None), 2),
    }
    for name, (state, warned) in expected.items():
        assert seen[name][:2] == (state, warned), (
            f"{name}: a setting a loaded file does not give as true or false was not read closed and said: "
            f"{seen[name][:2]}")
    assert seen["cut before its settings"][2]["password"]["min_length"] == 12, (
        "a loaded file's other settings were not kept when a missing one was read closed")


# --------------------------------------------------------------------------- DK9


def _roles_of_auth():
    tree = ast.parse(_read("opti_oignon", "auth.py"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == ["VALID_ROLES"]:
            return ast.literal_eval(node.value)
    raise AssertionError("auth.VALID_ROLES not found")


def test_dk9_the_texts():
    import yaml

    auth_yaml = yaml.safe_load(_read("opti_oignon", "config", "auth.yaml"))
    assert auth_yaml["users"]["allow_registration"] is False, "the shipped auth.yaml opens registration"
    marketplace = _read("opti_oignon", "config", "plugin_marketplace.yaml")
    assert "require_hash: true" in marketplace and "refused before anything is fetched" in marketplace

    routes = _read("opti_oignon", "api", "routes_auth.py")
    tree = ast.parse(routes)
    roles = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and [ast.unparse(t) for t in n.targets] == ["_ROLES"])
    assert roles == _roles_of_auth(), "the route's roles are not auth.VALID_ROLES"
    assert "Registration is disabled. Contact an administrator." not in routes

    market_route = _read("opti_oignon", "api", "routes_plugin_marketplace.py")
    assert "Expected SHA-256 hash (optional)" not in market_route
    assert "required unless install.require_hash" in market_route

    marketplace_ui = _read("frontend", "src", "lib", "components", "settings", "PluginMarketplace.svelte")
    assert 'placeholder="SHA-256 of the archive"' in marketplace_ui and "SHA-256 hash (optional)" not in marketplace_ui
    register_ui = _read("frontend", "src", "routes", "register", "+page.svelte")
    assert "the platform's owner adds accounts" in register_ui and "Contact an administrator" not in register_ui
    mode_ui = _read("frontend", "src", "lib", "components", "settings", "sections", "AccountAuthMode.svelte")
    assert "createUser({" in mode_ui and "owner_password: ownerPassword" in mode_ui
    assert "toastError(httpRefusal(e)?.detail || 'Failed to add the account');" in mode_ui
    client = _read("frontend", "src", "lib", "api", "auth.ts")
    assert "apiPost<AuthUser>(`${BASE}/users`, req)" in client
    login_ui = _read("frontend", "src", "routes", "login", "+page.svelte")
    link = login_ui.index('<a href="/register"')
    assert login_ui.rfind("{#if $authStatus?.registration_enabled}", 0, link) > login_ui.rfind("{/if}", 0, link), (
        "the login page offers registration whatever the server says")

    owners = {node.name: [ast.unparse(d) for d in node.args.defaults] for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name in ("list_users", "delete_user", "create_account")}
    assert "Depends(_require_owner)" in owners["list_users"] and "Depends(_require_owner)" in owners["delete_user"]
    audit = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_audit_log")
    assert "Depends(_require_owner)" in [ast.unparse(d) for d in audit.args.defaults]
    assert "_require_admin" not in routes, "a route reads the token's role for an owner's act"
    user_guide = " ".join(_read("docs", "user-guide", "plugins.md").split())
    assert '"expected_sha256":' in user_guide and '"url":' in user_guide and '"name": "my-plugin"' not in user_guide

    authentication = _read("docs", "security", "authentication.md")
    flat = " ".join(authentication.split())
    for stale in ("registration is open by default", "the login itself checks tries sent together before it "
                  "counts the first"):
        assert stale not in flat, stale
    for said in ("Registration makes the first account of an install that has none",
                 "`users.allow_registration: false`", "`POST /api/auth/users`",
                 "in Bulbe mode, or a mode that cannot be read, it never does"):
        assert said in flat, said
    bulbe = " ".join(_read("docs", "security", "bulbe-mode.md").split())
    assert "`install.require_hash` is true as shipped" in bulbe
    assert "Registration makes the first account of an install with none and no other" in bulbe
    first_run = " ".join(_read("docs", "getting-started", "first-run.md").split())
    assert "Registration then closes" in first_run
    guide = " ".join(_read("docs", "PLUGIN_DEVELOPMENT_GUIDE.md").split())
    assert "Optionally provide a SHA-256 hash" not in guide and '"sha256": ""' not in guide
    assert "an install without one is refused unless `install.require_hash` is false" in guide
    changelog = " ".join(_read("CHANGELOG.md").split())
    assert "[SECURITY] A plugin installed from a URL asks the archive's SHA-256, and registration closes past the " \
           "first account." in changelog
    assert ("Contracts dk1-dk10 in `tests/test_plugin_digest_and_closed_registration_contracts.py`; og2, en5, uc2 "
            "and a36 give way to them.") in changelog
    deselected = _read("pyproject.toml")
    for gone in ("test_outbound_gate_contracts.py::test_og2_",
                 "test_bulbe_login_enrols_a_factor_contracts.py::test_en5_",
                 "test_auth_config_unread_closes_the_door_contracts.py::test_uc2_",
                 "test_auth_core_contracts.py::test_a36_"):
        assert f"--deselect=tests/{gone}" in deselected, gone
    assert "The owner also lists and removes accounts (`GET` and `DELETE /api/auth/users`) and reads the account " \
           "audit (`GET /api/auth/audit`)" in flat
    assert "A new password of more than 72 bytes is refused -- at registration, when the owner adds an account and " \
           "when a password is changed --" in flat


# --------------------------------------------------------------------------- DK10


def test_dk10_sql_is_parameterised_never_f_string_the_accounts_written_in_one_transaction():
    tree = ast.parse(source("auth.py").read_text())
    sites = ac_t._sql_sites(tree)
    assert len(sites) == ac_t._SQL_SITES + 2

    fstring = 0
    dynamic = 0
    for node in sites:
        arg0 = node.args[0] if node.args else None
        if isinstance(arg0, ast.JoinedStr):
            fstring += 1
        elif isinstance(arg0, ast.Call):
            dynamic += 1
            # The dynamic site must be ``<constant>.format(...)``.
            assert isinstance(arg0.func, ast.Attribute)
            assert arg0.func.attr == "format"
            assert isinstance(arg0.func.value, ast.Constant)
            assert isinstance(arg0.func.value.value, str)

    assert fstring == ac_t._SQL_FSTRING_SITES
    assert dynamic == ac_t._SQL_DYNAMIC_SITES
    owners = [node for node in sites if isinstance(node.args[0], ast.Constant)
              and node.args[0].value == "SELECT user_id, role, rowid FROM users"]
    assert len(owners) == 1, "the owners' read is the one site added"

    create = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "create_user")
    statements = [" ".join(node.args[0].value.split()) for node in sorted(ac_t._sql_sites(create),
                                                                          key=lambda n: (n.lineno, n.col_offset))]
    assert statements[:2] == ["BEGIN IMMEDIATE", "SELECT COUNT(*) FROM users"], statements
    assert statements[2].startswith("INSERT INTO users "), statements
    assert len(statements) == 3, "a count of the accounts outside the transaction"
