#!/usr/bin/env python3
"""Replacements for four contracts that pinned a seam by WHEN it was reached.

Each of the four asserted that the encrypted-connection seam is exercised at
import, because the store behind it opened its database while constructing.
That timing was the defect, not the property: importing the package opened
twenty databases before any caller had asked for anything, with encryption
not enforced, at a moment where a refusal could not be handled.

The property those four protect is real and is kept here in full -- storage
is reached ONLY through the seam, the path asked for is the anchored
configured one, a refusing seam is absorbed rather than raised, and a caller
value travels as a bound parameter. What changes is the moment: each is now
observed at the first use instead of at the import, and each replacement
also pins that the import itself reaches nothing. A contract that only moved
the assertion later would have let the eager open come back.

  * ls1 supersedes a1  -- the authentication module.
  * ls2 supersedes b35 -- the anchored configured path.
  * ls3 supersedes b30 -- the refusing seam.
  * ls4 supersedes fp1 -- the parameterised round trip.

A fifth, rg4, is superseded without a replacement here: it pinned which
databases the guard's ledger still carried at a moment when six were owed,
and the ledger is now pinned entry by entry where the decisions that emptied
it are recorded.
"""

import hashlib
import os
import sqlite3
import tempfile
import types

from _isolation import isolate, source

_AUTH = "opti_oignon.auth"
_FP = "opti_oignon.session_fingerprint"

# Read off the shipped configuration in the reference environment.
_CFG_SQLITE_PATH = "fingerprint.db"


# --- Seeded connectors ------------------------------------------------------


class _Recorder:
    """A stand-in ``safe_connect`` that journals what it is asked to do.

    Equal paths map to one database so a round trip can be observed; every
    statement is journaled, which is what makes the bound-parameter claim an
    observation rather than a reading of the source.
    """

    def __init__(self, tmpdir):
        self._tmpdir = tmpdir
        self._stores = {}
        self.calls = 0
        self.paths = []
        self.statements = []

    def connect(self, db_path, **kwargs):
        self.calls += 1
        key = str(db_path)
        self.paths.append(key)
        real = self._stores.get(key)
        if real is None:
            target = os.path.join(
                self._tmpdir,
                hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()
                + ".db",
            )
            real = sqlite3.connect(target, check_same_thread=False)
            self._stores[key] = real
        return _RecordingConnection(real, self)

    def inserts(self, table):
        return [
            (sql, params)
            for sql, params in self.statements
            if f"INSERT INTO {table}" in sql
        ]

    def close_all(self):
        for real in self._stores.values():
            try:
                real.close()
            except sqlite3.Error:
                pass
        self._stores.clear()


class _RecordingConnection:
    """Delegates to a real connection, journaling every statement."""

    def __init__(self, real, recorder):
        self._real = real
        self._recorder = recorder

    def execute(self, sql, params=()):
        self._recorder.statements.append((sql, tuple(params)))
        return self._real.execute(sql, params)

    def executescript(self, sql):
        self._recorder.statements.append((sql, ()))
        return self._real.executescript(sql)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def close(self):
        # The shared store outlives one caller's connection.
        return None


def _seeded(recorder):
    stub = types.ModuleType("opti_oignon.db_utils")
    stub.safe_connect = recorder.connect
    return stub


def _refusing(counter):
    stub = types.ModuleType("opti_oignon.db_utils")

    def _safe_connect(db_path, **kwargs):
        counter["n"] += 1
        raise sqlite3.OperationalError("connection refused by the seam")

    stub.safe_connect = _safe_connect
    return stub


def _load(module_key, filename, recorder):
    loaded, restore = isolate(
        targets={module_key: source(filename)},
        blocked=("ollama",),
        seeded={"opti_oignon.db_utils": _seeded(recorder)},
    )
    return loaded[module_key], restore


# --- ls1: the authentication module -----------------------------------------


def test_ls1_the_auth_module_reaches_the_seam_on_first_use_not_at_import():
    """Supersedes a1, whose first assertion became false by design.

    a1 pinned that the seeded connector is reached at import, because the
    module-level manager opened its database while constructing. Both halves
    of what a1 protected are kept: the singletons and the availability flag
    still land at import, and the manager still reaches storage exclusively
    through the seeded seam. Only the moment moved, and the import reaching
    nothing is asserted here so it cannot move back.
    """
    tmpdir = tempfile.mkdtemp(prefix="ls_auth_")
    recorder = _Recorder(tmpdir)
    auth, restore = _load(_AUTH, "auth.py", recorder)
    try:
        assert recorder.calls == 0, (
            f"importing the module reached the connector {recorder.calls} "
            f"time(s) for {recorder.paths}; constructing a manager must not "
            "open a database"
        )
        assert auth.AUTH_AVAILABLE is True
        assert auth.auth_manager is not None
        assert type(auth.auth_manager).__name__ == "AuthManager"
        assert auth.login_rate_limiter is not None
        assert type(auth.login_rate_limiter).__name__ == "LoginRateLimiter"

        # The seam is still the only way to storage, one use later.
        manager = auth.AuthManager(db_path=os.path.join(tmpdir, "named.db"))
        assert recorder.calls == 0, (
            "a second manager opened a database while constructing"
        )
        manager._get_conn().close()
        assert recorder.calls > 0, (
            "the first connection did not reach the seeded connector, so "
            "this contract has observed nothing"
        )
        assert any(
            os.path.basename(one) == "named.db" for one in recorder.paths
        ), f"the manager opened {recorder.paths!r} rather than its own path"
    finally:
        recorder.close_all()
        restore()


def test_ls1b_the_signing_secret_is_minted_without_the_seam():
    """The half of the constructor that did NOT move.

    The secret is generated into the configuration in memory and never
    touches the database, which is why deferring the schema could leave it
    where it was. Asserted rather than assumed, because the two sat on
    adjacent lines and moving both would have looked identical here.
    """
    tmpdir = tempfile.mkdtemp(prefix="ls_secret_")
    recorder = _Recorder(tmpdir)
    auth, restore = _load(_AUTH, "auth.py", recorder)
    try:
        manager = auth.AuthManager(db_path=os.path.join(tmpdir, "secret.db"))
        assert recorder.calls == 0, "minting the secret opened a database"
        assert manager._get_jwt_secret(), (
            "the secret must still be minted at construction"
        )
    finally:
        recorder.close_all()
        restore()


# --- ls2: the anchored configured path --------------------------------------


def test_ls2_the_store_opens_the_anchored_configured_path_on_first_use():
    """Supersedes b35, whose first assertion became false by design.

    b35 pinned that the seeded connector is reached at import and that the
    path it asks for is the configured store file, anchored on the package
    data directory rather than resolved against the caller's directory. The
    anchoring claim is what mattered and it is re-asserted in full; what
    moved is that the open now happens when the manager's store is used.
    """
    tmpdir = tempfile.mkdtemp(prefix="ls_anchor_")
    recorder = _Recorder(tmpdir)
    mod, restore = _load(_FP, "session_fingerprint.py", recorder)
    try:
        assert recorder.calls == 0, (
            f"importing the module opened {recorder.paths!r}"
        )
        mod.fingerprint_manager._preferences.get_ratios()
        anchored = [
            one for one in recorder.paths
            if os.path.basename(one) == _CFG_SQLITE_PATH
        ]
        assert anchored, (
            f"the store must open its configured file, got {recorder.paths!r}"
        )
        for one in anchored:
            assert os.path.isabs(one), (
                "a relative path resolves against the caller's directory, "
                f"which is the defect b35 fixed: {one}"
            )
            assert os.path.basename(os.path.dirname(one)) == "data", (
                "the store belongs beside the others in the package data "
                f"directory, got {one}"
            )
    finally:
        recorder.close_all()
        restore()


# --- ls3: the refusing seam -------------------------------------------------


def test_ls3_a_refused_seam_is_absorbed_at_first_use_not_at_import():
    """Supersedes b30, whose first assertion became false by design.

    b30 pinned that a connector refusing every open is reached at import and
    that the module loads anyway -- the store's own guard absorbs it and the
    availability latch stays up. That fail-secure behaviour is unchanged and
    is re-asserted at the moment the refusal now happens. The import
    reaching nothing is asserted alongside it, because a refusal absorbed at
    a moment nobody reaches is not evidence of anything.
    """
    counter = {"n": 0}
    loaded, restore = isolate(
        targets={_FP: source("session_fingerprint.py")},
        blocked=("ollama",),
        seeded={"opti_oignon.db_utils": _refusing(counter)},
    )
    mod = loaded[_FP]
    try:
        assert counter["n"] == 0, (
            "the import reached the refusing connector, so a database was "
            "opened before any caller asked for one"
        )
        assert mod.FINGERPRINT_AVAILABLE is True
        assert mod.fingerprint_manager is not None

        store = mod.UserPreferencesStore(db_path="refused.db")
        assert counter["n"] == 0, "construction reached the connector"
        assert store.record("approve", "plan", "ctx") is None
        assert counter["n"] > 0, (
            "the refusing connector was never reached, so nothing was "
            "absorbed and this contract has observed nothing"
        )
        assert store.get_ratios() == {
            "approve": 0.0, "modify": 0.0, "abort": 0.0,
        }
        assert store.get_phase_preferences() == {}
        assert store.total_decisions == 0
        assert mod.FINGERPRINT_AVAILABLE is True
    finally:
        restore()


# --- ls4: the parameterised round trip --------------------------------------


def test_ls4_a_preference_round_trips_through_the_seam_with_bound_values():
    """Supersedes fp1, whose first assertion became false by design.

    fp1 pinned that the store opens its named path through the seam as it is
    constructed, then that a caller value travels as a bound parameter and
    the decision round-trips. Everything after the first assertion is kept
    verbatim in substance; the open is observed at the first write.
    """
    tmpdir = tempfile.mkdtemp(prefix="ls_round_")
    recorder = _Recorder(tmpdir)
    mod, restore = _load(_FP, "session_fingerprint.py", recorder)
    try:
        store = mod.UserPreferencesStore(db_path="prefs-under-test.db")
        assert "prefs-under-test.db" not in recorder.paths, (
            "constructing the store opened its database"
        )
        marker = "ctx-with-'quote--marker"
        store.record("approve", "plan", marker)
        assert "prefs-under-test.db" in recorder.paths, (
            "the store must open its path through the seeded encrypted-"
            "connection seam"
        )
        inserts = recorder.inserts("preferences")
        assert len(inserts) == 1, f"expected one insert, got {len(inserts)}"
        sql, params = inserts[0]
        assert "?" in sql, "the statement must use bind placeholders"
        assert marker not in sql, f"a caller value leaked into SQL: {sql!r}"
        assert marker in [str(one) for one in params], (
            "the caller value must travel as a bound parameter"
        )
        ratios = store.get_ratios()
        assert ratios["approve"] == 1.0, (
            f"the recorded decision must round-trip, got {ratios}"
        )
        assert store.total_decisions == 1
    finally:
        recorder.close_all()
        restore()
