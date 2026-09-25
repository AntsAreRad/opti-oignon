#!/usr/bin/env python3
"""Contracts for the componion's store: one encrypted file per person, refused by name, never adopted.

A being lives in one SQLite file under the data directory, named by its
owner's tag and its soil. A new file is probed after its schema commits and
refused if it was written in clear; an existing file is probed by its header
before any connection. Birth needs a readable key and SQLCipher, and every
birth is anchored in the signed audit log, so a store that went missing is
never mistaken for a being that was never sown.

  * AS1 -- an unkeyed or plain connection on a new file is refused by name,
    and nothing is left; an existing file is refused by its header before
    any connection, or by the connection when it answers no cipher; the
    probe can say yes.
  * AS2 -- without a key and with encryption required, a birth is refused
    ``no_soil`` without drawing and without creating the directory, and the
    status is ``awaiting_soil``; a plaintext seam leaves no file and the
    status stays ``ready``.
  * AS3 -- an unknown being is ``None``; every refused store refuses a birth
    without drawing and without touching its bytes; a key that cannot be
    read never reaches the audit.
  * AS4 -- the store's files stay inside the data directory: a path that
    climbs out, is absolute, empty or not a string, a tag that is not hex,
    and a directory that resolves elsewhere are refused ``path``; with no
    data directory at all the store is refused and creates nothing; a store
    name that is a symbolic link is refused ``path``, and a temp name that is
    one is removed as a link, never through it.
  * AS5 -- Bulbe ignores the glass option: in any mode that is not exactly
    Daily -- a mode that cannot be read included -- a glass jar is sealed
    for every action and never connected, and no glass jar is sown; the mode
    is read once per action, and the real mode manager is re-read when
    another process changes its file; a key that appears after a glass
    birth keeps the jar open, labelled, through its own plain connection;
    only the YAML boolean ``false`` loosens encryption.
  * AS6 -- one store per person: once the single-user latch is off, the
    local being is unclaimed, never shown to another account and never
    adopted, and listed only to an administrator.
  * AS7 -- two processes, under two string hash seeds, compute the same
    checkpoint of the same being: identical bytes, one row; a different
    state at the same place is refused ``divergence`` and never overwrites.
  * AS9 -- a store missing after its birth is ``missing``, never ``ready``;
    looking creates no audit file; a wipe makes it ``ready`` again; a birth
    under a foreign key blocks only its owner; an interrupted sowing can be
    finished, and one that was never anchored is cleared by the next birth.
  * AS10 -- every destructible key is a drawn key; forgetting the rhythm and
    ending a heard season remove the keys, the rows and the key bytes; a
    being sown without consent keeps no rhythm; the trunk read never yields
    a local row; old root pages spliced back are refused by the anchor, and
    a resume removes what they restored.
  * AS12 -- the vacuum debt is written by the destroying transaction and
    cleared only after a VACUUM ran: a reader held across the VACUUM, or a
    process that stops at it, leaves ``vacuum_owed`` on record, and the next
    write runs it and clears it.
  * AS13 -- a checkpoint never predates the event it runs through, so a
    forget reaches every state that could hold what it forgets; the same
    checkpoint again changes nothing; a kept blob edited in place is refused
    by the read-back.
  * AS14 -- a law the engine cannot vouch for leaves the being unavailable:
    a genesis naming a law no longer carried is refused ``law`` at open and
    at resume, never read under another law, and so is a damaged table.
  * AS15 -- a long-lived process never sows over a birth another process
    anchored: once the store is missing, it says so and refuses the birth.
  * AS16 -- the unclaimed listing reads the mode and decides with it: outside
    Daily a glass jar is left out, as the jar itself is sealed; an encrypted
    being is listed in every mode.
  * AS17 -- the shipped settings file requires encryption, and a busy
    timeout out of range falls back to 5000 and says so.

Local-only. The platform loads through the shared isolation window with the
platform's configuration, keys, mode, audit log and user modules proven
unreachable; every seam is injected (``tests/_allium_store_support.py``).
"""

import ast
import hashlib
import json
import logging
import os
import sqlite3
import stat
import subprocess
import sys
import threading
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_store_support as support  # noqa: E402
from _isolation import REPO, source  # noqa: E402

BUDGET_S = {
    "test_as1_an_unkeyed_or_plain_connection_is_refused_by_the_header": 2.0,
    "test_as2_without_a_key_birth_is_refused_and_nothing_is_left": 2.0,
    "test_as3_an_unknown_being_is_none_and_a_refused_store_never_draws_a_seed": 2.0,
    "test_as4_store_paths_stay_inside_the_data_directory": 2.0,
    "test_as5_bulbe_ignores_the_glass_option_and_the_mode_is_read_per_action": 2.0,
    "test_as6_one_store_per_person_and_the_local_being_is_never_adopted": 2.0,
    "test_as7_two_processes_compute_the_same_checkpoint_identical_bytes_one_row": 2.0,
    "test_as9_a_store_missing_after_sowing_is_missing_never_ready": 2.0,
    "test_as10_every_destructible_key_is_random_and_destroying_it_removes_its_bytes": 2.0,
    "test_as12_the_vacuum_debt_is_written_before_the_vacuum_and_cleared_only_after_it": 2.0,
    "test_as13_a_checkpoint_never_predates_its_event_and_what_is_kept_is_read_back": 2.0,
    "test_as14_a_law_the_engine_cannot_vouch_for_leaves_the_being_unavailable": 2.0,
    "test_as15_a_long_lived_process_never_sows_over_a_birth_another_process_anchored": 2.0,
    "test_as16_outside_daily_the_unclaimed_listing_leaves_out_a_glass_jar": 2.0,
    "test_as17_the_shipped_settings_require_encryption_and_a_bad_busy_timeout_is_said": 2.0,
}


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def p():
    platform, restore = support.open_platform()
    try:
        yield platform
    finally:
        restore()


def _status(p, given, user="local"):
    target = support.store(p, given)
    try:
        return target.status(user)
    finally:
        target.close()


def _code(p, call):
    try:
        call()
    except p.membrane.MembraneRefused as refusal:
        return "membrane:" + refusal.code
    except p.store.StoreRefused as refusal:
        return refusal.code
    return None


def _real_probe(p, tmp_path, suite, **overrides):
    """The store's own probe, a readable key and SQLCipher said to be there."""
    return support.seams(p, tmp_path, suite=suite, probe=None, **overrides)


def _random_file(p, path, size, suite):
    stream = p.rng.Stream(bytes(32), "test." + suite + ".file", size)
    data = bytearray()
    while len(data) < size:
        data += stream.next_u64().to_bytes(8, "big")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(data[:size]))
    assert bytes(data[:16]) != support.MAGIC


def _plain_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE note (x TEXT)")
        conn.execute("INSERT INTO note VALUES ('plain')")
        conn.commit()
    finally:
        conn.close()
    assert path.read_bytes()[:16] == support.MAGIC and path.stat().st_size >= 512


# ---------------------------------------------------------------------------
# AS1 -- the header decides, before and after the connection
# ---------------------------------------------------------------------------
def test_as1_an_unkeyed_or_plain_connection_is_refused_by_the_header(p, tmp_path):
    tag = p.anchors.owner_tag("local")
    for seam, cause in ((support.sqlite_seam(), "no cipher version"), (support.answering_seam(), "not keyed")):
        given = _real_probe(p, tmp_path.joinpath(cause.replace(" ", "_")), "as1", connect=seam)
        target = support.store(p, given)
        with pytest.raises(p.store.PlaintextRefused) as info:
            support.sow(p, target)
        assert info.value.code == "plaintext" and cause in str(info.value), str(info.value)
        assert seam.calls >= 1, "the new file was connected, then refused"
        assert sorted(support.directory(given).glob(tag + "*")) == [], "nothing is left"
        assert given["entropy"].calls == 0
        target.close()

    given = _real_probe(p, tmp_path.joinpath("existing"), "as1")
    path = support.store_path(p, given)
    _plain_file(path)
    status = _status(p, given)
    assert (status.status, status.reason) == ("unreadable", "soil"), status
    assert given["connect"].calls == 0, "a plain file under the encrypted name is never connected"

    path.unlink()
    _random_file(p, path, 300, "as1")
    status = _status(p, given)
    assert (status.status, status.reason) == ("unreadable", "soil") and "short file" in status.detail, status
    assert given["connect"].calls == 0

    path.unlink()
    _random_file(p, path, 4096, "as1")
    plain = dict(given, connect=support.Connector(lambda path, same, timeout: sqlite3.connect(
        path, check_same_thread=same, timeout=timeout)))
    status = _status(p, plain)
    assert (status.status, status.reason) == ("unavailable", "cipher"), status
    assert plain["connect"].calls >= 1

    # Witness: the probe can say yes.
    conn = support.CipherAnswering(sqlite3.connect(str(path)))
    try:
        assert p.store.probe(path, conn) == "encrypted"
        assert p.store.probe(path) == "encrypted"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# AS2 -- no key, no birth
# ---------------------------------------------------------------------------
def test_as2_without_a_key_birth_is_refused_and_nothing_is_left(p, tmp_path):
    tag = p.anchors.owner_tag("local")
    given = _real_probe(p, tmp_path.joinpath("nokey"), "as2", anchor_secret=lambda: ("none", None, "nokey"))
    target = support.store(p, given)
    assert _code(p, lambda: support.sow(p, target)) == "no_soil"
    assert target.status("local").status == "awaiting_soil"
    assert given["entropy"].calls == 0, "a refused soil never draws"
    assert not support.directory(given).exists(), "the directory is not created"
    target.close()

    given = _real_probe(p, tmp_path.joinpath("plainseam"), "as2")
    target = support.store(p, given)
    with pytest.raises(p.store.PlaintextRefused) as info:
        support.sow(p, target)
    assert info.value.code == "plaintext"
    assert sorted(support.directory(given).glob(tag + "*")) == [], "the file written in clear is deleted"
    assert given["entropy"].calls == 0
    assert target.status("local").status == "ready"
    target.close()

    # Witness: the header half can fire even when the connection answers a cipher version.
    path = tmp_path.joinpath("witness.db")
    _plain_file(path)
    conn = support.CipherAnswering(sqlite3.connect(str(path)))
    try:
        assert p.store.probe(path, conn) == "plaintext"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# AS3 -- None for no being; a refused store never draws
# ---------------------------------------------------------------------------
def test_as3_an_unknown_being_is_none_and_a_refused_store_never_draws_a_seed(p, tmp_path):
    given = support.seams(p, tmp_path.joinpath("fresh"), suite="as3")
    target = support.store(p, given)
    assert target.status("local").status == "ready"
    assert target.open("local") is None
    target.close()

    def refused(given, expected_status):
        path = support.store_path(p, given)
        before = path.read_bytes() if path.exists() else None
        status = _status(p, given)
        assert status.status == expected_status, status
        entropy = support.CountingEntropy(p.rng, "as3.refused", 0)
        attempt = support.store(p, dict(given, entropy=entropy))
        assert _code(p, lambda: support.sow(p, attempt)) is not None, "the birth is refused"
        attempt.close()
        assert entropy.calls == 0, "a refused store never draws"
        after = path.read_bytes() if path.exists() else None
        assert after == before, "the store's bytes are unchanged"
        return status

    plain = _real_probe(p, tmp_path.joinpath("plain"), "as3")
    _plain_file(support.store_path(p, plain))
    assert refused(plain, "unreadable").reason == "soil"

    wrong = _real_probe(p, tmp_path.joinpath("wrongkey"), "as3",
                        connect=support.raising_seam("SQLCipher key verification failed"))
    _random_file(p, support.store_path(p, wrong), 4096, "as3")
    assert refused(wrong, "unavailable").reason == "key"
    assert wrong["connect"].calls >= 1

    broken = support.seams(p, tmp_path.joinpath("broken"), suite="as3", index=1)
    grown = support.store(p, broken)
    being = support.sow(p, grown)
    support.build(being, 9, support.cli(p))
    grown.close()

    def flip(conn):
        eid = conn.execute("SELECT eid FROM links WHERE seq = 5").fetchone()[0]
        conn.execute("UPDATE bodies SET body = ? WHERE eid = ?", (b'{"act":"none"}', eid))
    support.edit(support.store_path(p, broken), flip)
    status = refused(broken, "unreadable")
    assert status.offer is not None and status.offer.kept_seq == 4, status

    unread = support.seams(p, tmp_path.joinpath("unread"), suite="as3",
                           anchor_secret=lambda: ("unreadable", None, "nokey"))
    _random_file(p, support.store_path(p, unread), 4096, "as3")
    assert refused(unread, "unavailable").reason == "key"
    assert unread["connect"].calls == 0, "a key that cannot be read never connects"

    nofile = support.seams(p, tmp_path.joinpath("nofile"), suite="as3",
                           anchor_secret=lambda: ("unreadable", None, "nokey"))
    assert refused(nofile, "unavailable").reason == "key"
    assert nofile["audit"].calls == 0, "the audit is never touched"

    # Witness: a birth that is allowed draws four times, 88 bytes.
    given = support.seams(p, tmp_path.joinpath("sown"), suite="as3", index=2)
    target = support.store(p, given)
    being = support.sow(p, target)
    assert being is not None
    assert (given["entropy"].calls, given["entropy"].drawn) == (4, 88)
    assert given["audit"].calls >= 1, "witness: the audit's counter reads above zero when the audit is reached"
    status = target.status("local")
    assert (status.status, status.labels) == ("alive", ("prototype",)), status
    target.close()


# ---------------------------------------------------------------------------
# AS4 -- inside the data directory, always
# ---------------------------------------------------------------------------
def test_as4_store_paths_stay_inside_the_data_directory(p, tmp_path, monkeypatch):
    elsewhere = tmp_path.joinpath("elsewhere")
    elsewhere.mkdir()
    outside = tmp_path.joinpath("outside")
    outside.mkdir()
    monkeypatch.chdir(elsewhere)
    base = support.seams(p, tmp_path, suite="as4")
    data = Path(base["data_dir"])
    data.mkdir()

    for bad in ("../x", "/abs", "a/../../b", "", 3, "a//b"):
        given = dict(base, persistence=dict(base["persistence"], path=bad))
        target = support.store(p, given)
        status = target.status("local")
        assert (status.status, status.reason) == ("unavailable", "path"), (bad, status)
        assert _code(p, lambda target=target: support.sow(p, target)) == "path", bad
        target.close()
    directory = data.joinpath("allium")
    for bad_tag in ("xyz", "A" * 32, "0" * 31 + "g", "../" + "0" * 29):
        assert _code(p, lambda bad_tag=bad_tag: p.store.store_file(directory, bad_tag, "encrypted")) == "path"
    assert base["entropy"].calls == 0

    directory.symlink_to(outside, target_is_directory=True)
    target = support.store(p, base)
    assert _code(p, lambda: support.sow(p, target)) == "path"
    assert (target.status("local").status, target.status("local").reason) == ("unavailable", "path")
    target.close()
    assert list(outside.iterdir()) == [], "nothing lands outside"
    directory.unlink()

    target = support.store(p, base)
    being = support.sow(p, target)
    assert being is not None
    target.close()
    placed = support.store_path(p, base)
    assert placed.exists() and placed.parent == data.resolve().joinpath("allium")
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(placed).st_mode) == 0o600
    assert list(elsewhere.iterdir()) == [], "the working directory is never used"

    # With no data directory given and the configuration blocked, the store is refused and creates nothing.
    before = sorted(str(item) for item in tmp_path.rglob("*"))
    defaulted = {key: value for key, value in base.items() if key != "data_dir"}
    target = support.store(p, defaulted)
    status = target.status("local")
    assert (status.status, status.reason) == ("unavailable", "path"), status
    assert _code(p, lambda: support.sow(p, target)) == "path"
    target.close()
    assert sorted(str(item) for item in tmp_path.rglob("*")) == before

    # A symbolic link inside the store's own directory is never followed.
    shared = support.seams(p, tmp_path.joinpath("links"), suite="as4", index=1, single_user=lambda: False)
    now = shared["clock"].wall
    target = support.store(p, shared)
    assert target.sow(transport=support.web(p, "bob", now), law="fixture", tz_minutes=0,
                      rhythm_consent=False) is not None
    target.close()
    bob_file = support.store_path(p, shared, "bob")
    bob_sha = support.sha256_file(bob_file)
    victim = outside.joinpath("victim.db")
    victim.write_bytes(b"outside" * 100)
    # A store name linked outside the directory, or to another person's store, is refused ``path``, never read.
    alice_file = support.store_path(p, shared, "alice")
    for pointee in (victim, bob_file):
        alice_file.symlink_to(pointee)
        fresh = support.store(p, shared)
        status = fresh.status("alice")
        assert (status.status, status.reason) == ("unavailable", "path"), (pointee.name, status)
        assert _code(p, lambda fresh=fresh: fresh.open("alice")) == "path", pointee.name
        fresh.close()
        alice_file.unlink()
    assert victim.read_bytes() == b"outside" * 100
    # A temp name linked to another person's store: only the link is removed.
    alice_temp = support.store_path(p, shared, "alice", ".sowing.db")
    alice_temp.symlink_to(bob_file.name)
    fresh = support.store(p, shared)
    connected = shared["connect"].calls
    with pytest.raises(p.store.ResumeRefused):
        fresh.finish_sowing(transport=support.web(p, "alice", now), confirm="0" * 8)
    fresh.close()
    assert shared["connect"].calls == connected, "what the link names is never opened"
    assert not alice_temp.is_symlink() and not alice_temp.exists(), "the link itself is removed"
    assert support.sha256_file(bob_file) == bob_sha, "the store it named is untouched"
    # A temp name linked to its own store: the next birth removes the link, never the store.
    bob_temp = support.store_path(p, shared, "bob", ".sowing.db")
    bob_temp.symlink_to(bob_file.name)
    fresh = support.store(p, shared)
    assert _code(p, lambda: fresh.sow(transport=support.web(p, "bob", now), law="fixture", tz_minutes=0,
                                      rhythm_consent=False)) == "exists"
    fresh.close()
    assert not bob_temp.is_symlink()
    assert support.sha256_file(bob_file) == bob_sha
    assert _status(p, shared, "bob").status == "alive"
    # The probe itself never follows a link.
    _random_file(p, outside.joinpath("header.db"), 4096, "as4")
    link = tmp_path.joinpath("links", "header-link.db")
    link.symlink_to(outside.joinpath("header.db"))
    assert p.store.probe(outside.joinpath("header.db")) == "encrypted", "witness: the file itself reads as encrypted"
    assert p.store.probe(link) == "short"


# ---------------------------------------------------------------------------
# AS5 -- Bulbe seals the glass jar; the mode is read once per action
# ---------------------------------------------------------------------------
_NOT_DAILY = ("bulbe", RuntimeError("the mode cannot be read"), "", "unknown", "Daily ", "Daily", "DAILY", " daily",
              b"daily")


def _glass(p, tmp_path, suite, index=0, **overrides):
    """Seams for a glass jar: no key configured, the YAML option exactly false, the store's own probe."""
    out = support.seams(p, tmp_path, suite=suite, index=index, probe=None,
                        anchor_secret=lambda: ("none", None, "nokey"),
                        persistence={"busy_timeout_ms": 5000, "path": "allium", "require_encryption": False})
    out.update(overrides)
    return out


def _outcome(p, call):
    """What an action gave: ``None`` when it answered, else the refusal's code."""
    try:
        call()
    except p.membrane.MembraneRefused as refusal:
        return "membrane:" + refusal.code
    except p.store.ResumeRefused as refusal:
        return "resume:" + refusal.code
    except p.store.StoreRefused as refusal:
        return refusal.code
    return None


def _being_actions(p, being, eid):
    cli = support.cli(p)
    hook = p.membrane.Transport("light_hook")
    return {
        "append": lambda: being.append("act", {"act": "touch"}, transport=cli),
        "rhythm_put": lambda: being.rhythm_put(hook, 9, True),
        "heard_note": lambda: being.heard_note(1, "tuft", {"turn": 1}),
        "heard_end_season": lambda: being.heard_end_season(1),
        "checkpoint_put": lambda: being.checkpoint_put(0, being.laws, eid, {"events": [eid]}),
        "trunk": lambda: list(being.trunk()),
        "head": lambda: being.head(),
        "verify": lambda: being.verify(),
        "unseal": lambda: being.unseal("0" * 32),
    }


def _store_actions(p, target):
    cli = support.cli(p)
    return {
        "status": lambda: target.status("local"),
        "open": lambda: target.open("local"),
        "resume": lambda: target.resume(transport=cli, confirm=(0, 0)),
        "finish_sowing": lambda: target.finish_sowing(transport=cli, confirm="0" * 8),
        "unclaimed": lambda: target.unclaimed(transport=support.cli(p, attended=True)),
    }


def test_as5_bulbe_ignores_the_glass_option_and_the_mode_is_read_per_action(p, tmp_path, monkeypatch):
    mode = support.Mode("daily")
    keyed, plain = support.sqlite_seam(), support.sqlite_seam()
    given = _glass(p, tmp_path.joinpath("jar"), "as5", connect=keyed, plain_connect=plain, mode=mode)
    target = support.store(p, given)
    being = support.sow(p, target, rhythm_consent=True)
    assert mode.reads == 1, "a sowing reads the mode once"
    glass = support.store_path(p, given, suffix=".glass.db")
    assert glass.exists() and not support.store_path(p, given).exists()
    assert keyed.calls == 0 and plain.calls >= 1
    status = target.status("local")
    assert (status.status, status.labels) == ("alive", ("glass_jar", "prototype")), status
    [(eid,)] = support.read(glass, "SELECT eid FROM links WHERE seq = 0")

    # In Daily, every action reads the mode exactly once, whether it answers or refuses.
    actions = dict(_being_actions(p, being, eid), **_store_actions(p, target))
    for name, call in actions.items():
        before = mode.reads
        expected = {"unseal": "membrane:payload", "resume": "resume:nothing", "finish_sowing": "resume:nothing"}
        assert _outcome(p, call) == expected.get(name), name
        assert mode.reads - before == 1, name

    # Any reading that is not exactly Daily seals the jar: never connected, every action refused.
    for value in _NOT_DAILY:
        mode.value = "daily"
        being = target.open("local")
        live = plain.connections[-1]
        counts = (keyed.calls, plain.calls)
        mode.value = value
        status = target.status("local")
        assert (status.status, status.reason) == ("sealed_bulbe", "sealed"), (value, status)
        with pytest.raises(sqlite3.ProgrammingError):
            live.execute("SELECT 1")
        actions = dict(_being_actions(p, being, eid), **_store_actions(p, target))
        for name, call in actions.items():
            before = mode.reads
            got = _outcome(p, call)
            assert mode.reads - before == 1, (value, name)
            if name not in ("status", "finish_sowing", "unclaimed"):
                assert got == "sealed", (value, name, got)
        assert (keyed.calls, plain.calls) == counts, value

    # In Bulbe no glass jar is sown.
    elsewhere = _glass(p, tmp_path.joinpath("bulbe_birth"), "as5", index=1, mode=support.Mode("bulbe"))
    attempt = support.store(p, elsewhere)
    assert _outcome(p, lambda: support.sow(p, attempt)) == "no_soil"
    attempt.close()
    assert elsewhere["entropy"].calls == 0 and not support.directory(elsewhere).exists()

    # Back in Daily the jar is alive, labelled.
    mode.value = "daily"
    status = target.status("local")
    assert (status.status, status.labels) == ("alive", ("glass_jar", "prototype")), status
    target.close()

    # A key appears after the birth: still a glass jar, through its own plain connection, never the keyed one.
    keyed_later, plain_later = support.sqlite_seam(), support.sqlite_seam()
    later = dict(given, anchor_secret=lambda: ("readable", support.KEY, support.KEY_ID),
                 cipher_available=lambda: True, connect=keyed_later, plain_connect=plain_later,
                 mode=support.Mode("daily"))
    target = support.store(p, later)
    status = target.status("local")
    assert status.status == "alive" and "glass_jar" in status.labels and "repot" in status.hints, status
    appended = target.open("local").append("act", {"act": "water"}, transport=support.cli(p))
    assert isinstance(appended, p.membrane.Appended), appended
    target.close()
    assert keyed_later.calls == 0 and plain_later.calls >= 1
    anchor = p.wire.parse(bytes(support.read(glass, "SELECT value FROM meta WHERE key = 'anchor'")[0][0]))
    assert anchor["key"] == "nokey", "a glass jar keeps its advisory anchor for life"
    strict = dict(given, persistence=dict(given["persistence"], require_encryption=True), mode=support.Mode("daily"))
    status = _status(p, strict)
    assert status.status == "alive" and "repot" in status.hints, "the YAML governs births, not a jar already born"

    # Only the YAML boolean false loosens encryption.
    folder = tmp_path.joinpath("yaml")
    folder.mkdir()
    cases = (
        ("missing", "persistence:\n  path: allium\n", True),
        ("string", 'persistence:\n  require_encryption: "false"\n', True),
        ("zero", "persistence:\n  require_encryption: 0\n", True),
        ("null", "persistence:\n  require_encryption: null\n", True),
        ("unreadable", "persistence: [unclosed\n", True),
        ("false", "persistence:\n  require_encryption: false\n", False),
    )
    for name, text, required in cases:
        file = folder.joinpath(name + ".yaml")
        file.write_text(text, encoding="ascii")
        assert p.settings.persistence(path=file)["require_encryption"] is required, name

    # The real mode manager: a mode file changed by another process is seen at the next action.
    blocked = tuple(name for name in support.BLOCKED if name != "opti_oignon.security_mode")
    real, restore = support.open_platform(blocked=blocked,
                                          extra={"opti_oignon.security_mode": source("security_mode.py")})
    try:
        manager = real.loaded["opti_oignon.security_mode"]
        home = tmp_path.joinpath("mode")
        home.mkdir()
        mode_file = home.joinpath("security.yaml")
        mode_file.write_text("security_mode: daily\n", encoding="ascii")
        monkeypatch.setattr(manager, "_SECURITY_YAML", mode_file)
        monkeypatch.setattr(manager, "_LOCKFILE_PATH", home.joinpath(".security_mode_lock"))
        monkeypatch.setattr(manager, "_DEFAULT_KEYFILE", home.joinpath(".keyfile"))
        wired = _glass(real, tmp_path.joinpath("real"), "as5", index=2, mode=None)
        target = support.store(real, wired)
        assert support.sow(real, target) is not None, "the first action read Daily from the real manager"
        assert target.status("local").status == "alive"
        # Another process switches to Bulbe; with no lockfile the file is trusted.
        switched = home.joinpath("security.yaml.next")
        switched.write_text("security_mode: bulbe\n", encoding="ascii")
        os.replace(switched, mode_file)
        status = target.status("local")
        assert (status.status, status.reason) == ("sealed_bulbe", "sealed"), status
        target.close()
    finally:
        restore()


# ---------------------------------------------------------------------------
# AS6 -- one store per person, never adopted
# ---------------------------------------------------------------------------
def _user_isolation_local_user():
    tree = ast.parse(REPO.joinpath("opti_oignon", "user_isolation.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "DEFAULT_LOCAL_USER" for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def test_as6_one_store_per_person_and_the_local_being_is_never_adopted(p, tmp_path):
    single = support.seams(p, tmp_path, suite="as6")
    target = support.store(p, single)
    local = support.sow(p, target)
    local_prefix = local.being_tag[:8]
    target.close()

    latched = dict(single, single_user=lambda: False)
    now = latched["clock"].wall
    target = support.store(p, latched)
    alice = target.status("alice")
    assert (alice.status, alice.offer) == ("ready", None), alice
    assert target.open("alice") is None, "another account never sees the local being"
    status = target.status("local")
    assert (status.status, status.reason) == ("unavailable", "unclaimed"), status
    listed = target.unclaimed(transport=support.web(p, "root", now, role="admin"))
    assert listed == [(p.anchors.owner_tag("local"), local_prefix)], listed
    assert target.unclaimed(transport=support.cli(p, attended=True)) == listed
    assert _code(p, lambda: target.unclaimed(transport=support.web(p, "alice", now))) == "membrane:owner"
    assert _code(p, lambda: target.unclaimed(transport=support.cli(p))) == "membrane:owner"

    bob = target.sow(transport=support.web(p, "bob", now), law="fixture", tz_minutes=0, rhythm_consent=False)
    assert bob is not None and bob.being_tag[:8] != local_prefix
    target.close()
    fresh = support.store(p, latched)
    try:
        assert fresh.status("bob").status == "alive"
        assert fresh.status("alice").status == "ready", "alice never sees bob's being"
        assert fresh.open("alice") is None
    finally:
        fresh.close()
    assert p.membrane.LOCAL_USER == _user_isolation_local_user() == "local"


# ---------------------------------------------------------------------------
# AS7 -- two processes, one checkpoint
# ---------------------------------------------------------------------------
# A child process: its own window, the configuration proven unreachable, the
# parent's audit entries on stdin, one checkpoint of a state derived from the
# eids, and on stdout what it computed.
_CHILD = r"""
import hashlib
import importlib
import json
import sys

sys.path.insert(0, sys.argv[1])
import _allium_store_support as support

given_in = json.loads(sys.stdin.read())
p, restore = support.open_platform()
try:
    try:
        importlib.import_module("opti_oignon.config")
    except ImportError:
        pass
    else:
        raise SystemExit("the platform's configuration is reachable from the child")

    class Audit(support.MemoryAudit):
        def verify_chain(self):
            return (True, None, 0)

    audit = Audit()
    audit.entries = given_in["entries"]
    given = support.seams(p, given_in["tmp"], suite="as7", audit=audit)
    target = support.store(p, given)
    being = target.open("local")
    trunk = list(being.trunk())
    eids = [row[0] for row in support.read(support.store_path(p, given), "SELECT eid FROM links ORDER BY seq")]
    t = trunk[-1][0]["t"]
    state = {"eids": eids, "kinds": [envelope["kind"] for envelope, _body in trunk], "t": t}
    made = being.checkpoint_put(t, being.laws, eids[-1], state)
    target.close()
    print(json.dumps({"blob": hashlib.sha256(made.blob).hexdigest(), "state_hash": made.state_hash}))
finally:
    restore()
"""


def _tree(root):
    return sorted(str(item.relative_to(root)) for item in root.rglob("*"))


def test_as7_two_processes_compute_the_same_checkpoint_identical_bytes_one_row(p, tmp_path):
    assert REPO.resolve() not in tmp_path.resolve().parents, "the children run outside the repository"
    given = support.seams(p, tmp_path, suite="as7")
    target = support.store(p, given)
    being = support.sow(p, target)
    support.build(being, 4, support.cli(p))
    target.close()
    path = support.store_path(p, given)
    before = _tree(tmp_path)
    handed = json.dumps({"entries": given["audit"].entries, "tmp": str(tmp_path)})
    outputs = []
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE="1")
        run = subprocess.run([sys.executable, "-c", _CHILD, str(REPO.joinpath("tests"))], cwd=tmp_path, env=env,
                             input=handed, capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stderr[-2000:]
        outputs.append(json.loads(run.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1], outputs
    rows = support.read(path, "SELECT blob, state_hash FROM checkpoints")
    assert len(rows) == 1, "one row"
    assert hashlib.sha256(bytes(rows[0][0])).hexdigest() == outputs[0]["blob"]
    assert rows[0][1] == outputs[0]["state_hash"]
    assert _tree(tmp_path) == before, "nothing else exists"

    # Witness: another state at the same place is refused by the read-back and never overwrites.
    [(t, laws, through)] = support.read(path, "SELECT t, laws, through FROM checkpoints")
    target = support.store(p, given)
    being = target.open("local")
    assert _outcome(p, lambda: being.checkpoint_put(t, laws, through, {"another": "state"})) == "divergence"
    target.close()
    assert support.read(path, "SELECT blob, state_hash FROM checkpoints") == rows


# ---------------------------------------------------------------------------
# AS9 -- missing is never ready
# ---------------------------------------------------------------------------
def _sow_entry(audit, owner):
    for event in audit.get_events(limit=256, event_type="allium_sow"):
        if event["details"].get("owner") == owner:
            return event["details"]
    return None


def _web_sow(p, target, sub, now):
    return target.sow(transport=support.web(p, sub, now), law="fixture", tz_minutes=0, rhythm_consent=False)


def test_as9_a_store_missing_after_sowing_is_missing_never_ready(p, tmp_path):
    audit = support.load_audit(tmp_path)
    given = support.seams(p, tmp_path, suite="as9", audit=audit, single_user=lambda: False)
    now = given["clock"].wall
    audit_file = Path(audit._db_path)
    assert _status(p, given, "alice").status == "ready"
    assert not audit_file.exists(), "looking writes nothing, not even the audit's own file"

    target = support.store(p, given)
    assert _web_sow(p, target, "alice", now) is not None
    target.close()
    assert audit_file.exists(), "witness: the first birth writes the very file looking never created"
    path = support.store_path(p, given, "alice")
    support.remove_with_journals(path)
    status = _status(p, given, "alice")
    assert status.status == "missing" and status.snapshot is None, status
    assert path.name in status.detail, status.detail
    with pytest.raises(p.store.StoreRefused):
        support.store(p, given).open("alice")

    entry = _sow_entry(audit, p.anchors.owner_tag("alice"))
    audit.append_event("allium_wipe", source="allium", action="wipe", severity="INFO", details=entry)
    assert _status(p, given, "alice").status == "ready", "a wipe entry closes the birth"

    target = support.store(p, given)
    assert _web_sow(p, target, "dave", now) is not None
    target.close()
    assert _status(p, given, "dave").status == "alive"

    audit.append_event("allium_sow", source="allium", action="sow", severity="INFO", details={
        "being": "ab" * 16, "key": "f00dfeedf00dfeed", "mac": "00" * 32, "owner": p.anchors.owner_tag("carol")})
    status = _status(p, given, "carol")
    assert (status.status, status.reason) == ("unavailable", "foreign"), status
    assert _status(p, given, "bob").status == "ready", "a foreign birth blocks only its owner"

    # An interrupted sowing, finished.
    def cut(name):
        if name == "link":
            raise RuntimeError("the sowing is cut before the link")

    target = support.store(p, dict(given, stage=cut))
    with pytest.raises(RuntimeError):
        _web_sow(p, target, "erin", now)
    target.close()
    prefix = _sow_entry(audit, p.anchors.owner_tag("erin"))["being"][:8]
    status = _status(p, given, "erin")
    assert status.status == "missing" and status.offer == p.store.Finish(prefix), status
    finisher = support.store(p, given)
    erin = support.web(p, "erin", now)
    with pytest.raises(p.store.ResumeRefused) as info:
        finisher.finish_sowing(transport=erin, confirm="0" * 8)
    assert info.value.code == "confirm"
    assert finisher.finish_sowing(transport=erin, confirm=prefix) is not None
    finisher.close()
    assert _status(p, given, "erin").status == "alive"
    assert not support.store_path(p, given, "erin", ".sowing.db").exists()

    # A sowing never anchored for good: its temp file leaves the status ready and the next birth clears it.
    target = support.store(p, dict(given, stage=cut))
    with pytest.raises(RuntimeError):
        _web_sow(p, target, "frank", now)
    target.close()
    temp = support.store_path(p, given, "frank", ".sowing.db")
    assert temp.exists()
    entry = _sow_entry(audit, p.anchors.owner_tag("frank"))
    audit.append_event("allium_wipe", source="allium", action="wipe", severity="INFO", details=entry)
    status = _status(p, given, "frank")
    assert (status.status, status.offer) == ("ready", None), status
    entropy = support.CountingEntropy(p.rng, "as9.frank", 1)
    target = support.store(p, dict(given, entropy=entropy))
    frank = _web_sow(p, target, "frank", now)
    target.close()
    assert frank is not None and entropy.calls == 4
    assert not temp.exists()
    assert _sow_entry(audit, p.anchors.owner_tag("frank"))["being"] != entry["being"]


# ---------------------------------------------------------------------------
# AS10 -- random keys, destroyed bytes, spliced pages, a trunk without local rows
# ---------------------------------------------------------------------------
def _meta_value(p, path, key):
    return p.wire.parse(bytes(support.read(path, "SELECT value FROM meta WHERE key = ?", (key,))[0][0]))


def _field_names(schema):
    names = []
    for name, spec in schema.items():
        names.append(name)
        if spec.get("type") == "object":
            names.extend(_field_names(spec["fields"]))
    return names


def test_as10_every_destructible_key_is_random_and_destroying_it_removes_its_bytes(p, tmp_path):
    given = support.seams(p, tmp_path, suite="as10")
    target = support.store(p, given)
    being = support.sow(p, target, rhythm_consent=True)
    path = support.store_path(p, given)
    entries = len(given["audit"].entries)
    hook = p.membrane.Transport("light_hook")
    assert [being.rhythm_put(hook, hour, hour == 8) for hour in (7, 8, 9)] == [0, 1, 2]
    [(genesis_eid,)] = support.read(path, "SELECT eid FROM links WHERE seq = 0")
    being.checkpoint_put(0, being.laws, genesis_eid, {"events": [genesis_eid]})
    being.heard_note(0, "hello", {"turn": 1})
    being.heard_note(0, "it's", {"turn": 2})
    keys = {name: bytes(value) for name, value in support.read(path, "SELECT name, key FROM keys")}
    assert sorted(keys) == ["being_secret", "heard:0", "rhythm"], sorted(keys)
    drawn = [chunk for chunk in given["entropy"].chunks if len(chunk) == 32]
    for name, value in keys.items():
        assert value in drawn, f"{name} is a drawn key, computed from nothing"
    rhythm_key, heard_key = keys["rhythm"], keys["heard:0"]
    assert support.count_bytes(path, rhythm_key) >= 1, "witness: the key's bytes can be counted"
    assert support.read(path, "SELECT COUNT(*) FROM rhythm_facts") == [(3,)]
    assert support.read(path, "SELECT COUNT(*) FROM heard") == [(2,)]
    # The trunk read while the local layers hold rows: the journal, and nothing local.
    early = list(being.trunk())
    assert support.read(path, "SELECT COUNT(*) FROM rhythm_facts") == [(3,)], "witness: the rhythm rows are there"
    assert len(early) == len(support.read(path, "SELECT seq FROM links")), early
    law_kinds = p.membrane.law_pin("fixture")["table"]["kinds"]
    assert all(law_kinds[envelope["kind"]]["scope"] == "trunk" for envelope, _body in early), early
    # No local-layer write reaches the audit log, whose entries carry the time the rows keep sealed.
    assert len(given["audit"].entries) == entries
    target.close()
    before_forget = path.read_bytes()

    target = support.store(p, given)
    being = target.open("local")
    assert isinstance(being.append("forget_rhythm", {}, transport=support.cli(p)), p.membrane.Appended)
    assert len(given["audit"].entries) > entries, "witness: a destruction does reach the audit log"
    assert support.read(path, "SELECT COUNT(*) FROM rhythm_facts") == [(0,)]
    assert support.read(path, "SELECT COUNT(*) FROM keys WHERE name = 'rhythm'") == [(0,)]
    assert support.count_bytes(path, rhythm_key) == 0
    assert support.read(path, "SELECT COUNT(*) FROM checkpoints") == [(0,)]
    assert _meta_value(p, path, "rhythm_floor") == 3
    being.heard_end_season(0)
    assert support.read(path, "SELECT COUNT(*) FROM heard") == [(0,)]
    assert support.read(path, "SELECT COUNT(*) FROM keys WHERE name = 'heard:0'") == [(0,)]
    assert support.count_bytes(path, heard_key) == 0
    assert _meta_value(p, path, "heard_ended") == [0]

    # The trunk read yields the journal and nothing local.
    kinds = p.membrane.law_pin("fixture")["table"]["kinds"]
    trunk = list(being.trunk())
    assert len(trunk) == len(support.read(path, "SELECT seq FROM links")) >= 2
    assert all(kinds[envelope["kind"]]["scope"] == "trunk" for envelope, _body in trunk), trunk
    for kind, entry in kinds.items():
        if entry["scope"] == "trunk" and entry["body"] is not None:
            assert "observed_hour" not in _field_names(entry["body"]), kind
    target.close()

    # Without consent at sowing there is no rhythm.
    refusing = support.seams(p, tmp_path.joinpath("no_consent"), suite="as10", index=1)
    other = support.store(p, refusing)
    unconsented = support.sow(p, other)
    assert _code(p, lambda: unconsented.rhythm_put(hook, 7, True)) == "membrane:consent"
    other.close()
    assert support.read(support.store_path(p, refusing), "SELECT COUNT(*) FROM rhythm_facts") == [(0,)]

    # Old root pages spliced back: the file's pages check, the anchor does not.
    for table in ("keys", "rhythm_facts"):
        support.splice(path, before_forget, table)
    names = [row[0] for row in support.read(path, "SELECT name FROM keys ORDER BY name")]
    assert "rhythm" in names and support.read(path, "SELECT COUNT(*) FROM rhythm_facts") == [(3,)], names
    status = _status(p, given)
    assert (status.status, status.reason) == ("unreadable", "anchor"), status
    resumer = support.store(p, given)
    resumer.resume(transport=support.cli(p), confirm=(status.offer.kept_seq, status.offer.discarded))
    resumer.close()
    assert [row[0] for row in support.read(path, "SELECT name FROM keys")] == ["being_secret"]
    assert support.read(path, "SELECT COUNT(*) FROM rhythm_facts") == [(0,)]
    assert support.count_bytes(path, rhythm_key) == 0 and support.count_bytes(path, heard_key) == 0
    assert _status(p, given).status == "alive"


# ---------------------------------------------------------------------------
# AS12 -- the vacuum debt is written before the VACUUM and cleared only after it
# ---------------------------------------------------------------------------
class _Stopped(BaseException):
    """The process stops: nothing any layer of the store handles."""


class _AtVacuum:
    """A standard-library connection whose ``VACUUM`` meets ``hook`` first."""

    def __init__(self, conn, hook):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_hook", hook)

    def execute(self, sql, *args):
        if " ".join(sql.split()).upper() == "VACUUM":
            self._hook()
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        setattr(self._conn, name, value)


def _at_vacuum(hook):
    def opener(path, same, timeout):
        conn = sqlite3.connect(path, check_same_thread=same, timeout=timeout)
        conn.execute("PRAGMA secure_delete = OFF")
        return conn

    return support.Connector(opener, wrap=lambda conn: _AtVacuum(conn, hook))


def test_as12_the_vacuum_debt_is_written_before_the_vacuum_and_cleared_only_after_it(p, tmp_path):
    cli = support.cli(p)
    fast = {"busy_timeout_ms": 20, "path": "allium", "require_encryption": True}

    # (1) A reader arrives as the VACUUM starts and holds its read across every wait.
    readers = []

    def hold_a_reader():
        if not readers:
            reader = sqlite3.connect(str(path), isolation_level=None)
            reader.execute("BEGIN")
            reader.execute("SELECT COUNT(*) FROM facts").fetchall()
            readers.append(reader)

    given = support.seams(p, tmp_path.joinpath("reader"), suite="as12", connect=_at_vacuum(hold_a_reader),
                          persistence=fast)
    target = support.store(p, given)
    being = support.sow(p, target)
    path = support.store_path(p, given)
    taught = being.append("lang_teach", {}, transport=cli, payload="shallot")
    assert _meta_value(p, path, "vacuum_owed") == 0
    try:
        assert isinstance(being.append("lang_forget", {"target": taught.eid}, transport=cli), p.membrane.Appended)
        assert len(readers) == 1, "witness: the reader held the file while the VACUUM ran"
        assert _meta_value(p, path, "vacuum_owed") == 1, "the debt is on record although the VACUUM failed"
    finally:
        for reader in readers:
            reader.close()
    assert "vacuum_owed" in target.status("local").hints
    assert isinstance(being.append("act", {"act": "warm"}, transport=cli), p.membrane.Appended)
    assert _meta_value(p, path, "vacuum_owed") == 0, "the next write ran the VACUUM and only then cleared the debt"
    assert "vacuum_owed" not in target.status("local").hints
    assert support.read(path, "PRAGMA freelist_count") == [(0,)]
    target.close()

    # (2) The process stops at the VACUUM, after the destruction committed.
    def stop():
        raise _Stopped("the process stops between the commit and the VACUUM")

    given = support.seams(p, tmp_path.joinpath("stop"), suite="as12", index=1, connect=_at_vacuum(stop))
    target = support.store(p, given)
    being = support.sow(p, target)
    path = support.store_path(p, given)
    taught = being.append("lang_teach", {}, transport=cli, payload="shallot")
    with pytest.raises(_Stopped):
        being.append("lang_forget", {"target": taught.eid}, transport=cli)
    target.close()
    restarted = dict(given, connect=support.sqlite_seam())
    assert _meta_value(p, path, "vacuum_owed") == 1
    status = _status(p, restarted)
    assert status.status == "alive" and "vacuum_owed" in status.hints, status
    target = support.store(p, restarted)
    assert isinstance(target.open("local").append("act", {"act": "warm"}, transport=cli), p.membrane.Appended)
    target.close()
    assert _meta_value(p, path, "vacuum_owed") == 0
    assert "vacuum_owed" not in _status(p, restarted).hints

    # (3) The VACUUM fails by itself, and nothing else stands in the way of a write: the debt still stays.
    failed = []

    def disk_full():
        if not failed:
            failed.append(True)
            raise sqlite3.OperationalError("database or disk is full")

    given = support.seams(p, tmp_path.joinpath("full"), suite="as12", index=2, connect=_at_vacuum(disk_full))
    target = support.store(p, given)
    being = support.sow(p, target)
    path = support.store_path(p, given)
    taught = being.append("lang_teach", {}, transport=cli, payload="shallot")
    assert isinstance(being.append("lang_forget", {"target": taught.eid}, transport=cli), p.membrane.Appended)
    assert failed == [True], "witness: the VACUUM failed"
    assert _meta_value(p, path, "vacuum_owed") == 1, "a VACUUM that failed never clears the debt"
    assert _meta_value(p, path, "cross_pending") == [], "witness: the writes around it went through"
    assert isinstance(being.append("act", {"act": "warm"}, transport=cli), p.membrane.Appended)
    assert _meta_value(p, path, "vacuum_owed") == 0
    target.close()


# ---------------------------------------------------------------------------
# AS13 -- a checkpoint never predates its event, and what is kept is read back
# ---------------------------------------------------------------------------
def test_as13_a_checkpoint_never_predates_its_event_and_what_is_kept_is_read_back(p, tmp_path):
    cli = support.cli(p)
    given = support.seams(p, tmp_path, suite="as13")
    target = support.store(p, given)
    being = support.sow(p, target)
    path = support.store_path(p, given)
    word = support.canary(p, "as13")
    given["clock"].wall += 10 * 60
    taught = being.append("lang_teach", {}, transport=cli, payload=word)
    given["clock"].wall += 10 * 60
    later = being.append("act", {"act": "play"}, transport=cli)
    assert (taught.t, later.t) == (10, 20), (taught, later)
    state = {"lexicon": [word]}

    # A checkpoint through the act of minute 20, named at minute 0, would outlive a forget of minute 10.
    before = (support.counts(path), _meta_value(p, path, "gen"))
    assert _outcome(p, lambda: being.checkpoint_put(0, being.laws, later.eid, state)) == "local"
    assert (support.counts(path), _meta_value(p, path, "gen")) == before, "nothing is written"
    made = being.checkpoint_put(20, being.laws, later.eid, state)
    assert made.t == 20, "witness: at the event's own minute it is kept"

    # The same checkpoint again: the kept row answers, and the generation does not move.
    gen = _meta_value(p, path, "gen")
    assert being.checkpoint_put(20, being.laws, later.eid, state) == made
    assert _meta_value(p, path, "gen") == gen
    being.checkpoint_put(21, being.laws, later.eid, {"lexicon": []})
    assert _meta_value(p, path, "gen") == gen + 1, "witness: a new checkpoint moves it"

    # A kept blob edited in place is refused by the read-back, never handed out as the state's bytes.
    support.edit(path, lambda conn: conn.execute("UPDATE checkpoints SET blob = ? WHERE t = 20",
                                                 (zlib.compress(b'{"events":[]}'),)))
    assert _outcome(p, lambda: being.checkpoint_put(20, being.laws, later.eid, state)) == "divergence"

    # The forget of minute 10 leaves no checkpoint that holds the word.
    assert isinstance(being.append("lang_forget", {"target": taught.eid}, transport=cli), p.membrane.Appended)
    kept = [zlib.decompress(bytes(blob)) for (blob,) in support.read(path, "SELECT blob FROM checkpoints")]
    assert all(word.encode("ascii") not in blob for blob in kept), kept
    target.close()


# ---------------------------------------------------------------------------
# AS14 -- a law the engine cannot vouch for: unavailable, never read under another law
# ---------------------------------------------------------------------------
def test_as14_a_law_the_engine_cannot_vouch_for_leaves_the_being_unavailable(p, tmp_path, monkeypatch):
    given = support.seams(p, tmp_path, suite="as14")
    target = support.store(p, given)
    being = support.sow(p, target, law="v0_1")
    assert being.law == "v0_1"
    target.close()
    path = support.store_path(p, given)
    sown = support.sha256_file(path)
    assert _status(p, given).status == "alive", "witness: under the law it names, the being opens"

    # The genesis names a law this engine no longer carries: refused ``law``, never read under the fixture law.
    carried = p.lawfiles.LAWS
    monkeypatch.setattr(p.lawfiles, "LAWS", tuple(name for name in carried if name != "v0_1"))
    status = _status(p, given)
    assert (status.status, status.reason) == ("unavailable", "law"), status
    resumer = support.store(p, given)
    assert _outcome(p, lambda: resumer.resume(transport=support.cli(p), confirm=(0, 0))) == "law"
    resumer.close()
    monkeypatch.setattr(p.lawfiles, "LAWS", carried)

    # The engine's own table is damaged, in a process that has not read it yet: unavailable, not unreadable.
    real = p.lawfiles.table_bytes
    damaged = real("journal_v1").replace(b'"move_pot"', b'"move_pog"', 1)
    assert damaged != real("journal_v1")
    monkeypatch.setattr(p.lawfiles, "table_bytes", lambda name: damaged if name == "journal_v1" else real(name))
    monkeypatch.setattr(p.membrane, "_LAWS", {})
    monkeypatch.setattr(p.membrane, "_PINS", {})
    status = _status(p, given)
    assert (status.status, status.reason) == ("unavailable", "law"), status
    assert support.sha256_file(path) == sown, "nothing is written"


# ---------------------------------------------------------------------------
# AS15 -- a long-lived process never sows over a birth another process anchored
# ---------------------------------------------------------------------------
def test_as15_a_long_lived_process_never_sows_over_a_birth_another_process_anchored(p, tmp_path):
    given = support.seams(p, tmp_path, suite="as15")
    server = support.store(p, dict(given))
    assert server.status("local").status == "ready", "the server looked before any birth"
    sower = support.store(p, dict(given))
    assert support.sow(p, sower) is not None
    sower.close()
    support.remove_with_journals(support.store_path(p, given))
    assert _status(p, given).status == "missing", "witness: a fresh process"
    status = server.status("local")
    assert status.status == "missing", status
    drawn = given["entropy"].calls
    assert _code(p, lambda: support.sow(p, server)) == "exists"
    server.close()
    assert given["entropy"].calls == drawn, "the refused birth draws nothing"
    assert [entry["event_type"] for entry in given["audit"].entries].count("allium_sow") == 1


# ---------------------------------------------------------------------------
# AS16 -- outside Daily the unclaimed listing leaves out a glass jar
# ---------------------------------------------------------------------------
def test_as16_outside_daily_the_unclaimed_listing_leaves_out_a_glass_jar(p, tmp_path):
    owner = p.anchors.owner_tag("local")
    attended = support.cli(p, attended=True)
    mode = support.Mode("daily")
    given = _glass(p, tmp_path.joinpath("jar"), "as16", mode=mode)
    target = support.store(p, given)
    prefix = support.sow(p, target).being_tag[:8]
    target.close()
    target = support.store(p, dict(given, single_user=lambda: False))
    assert target.unclaimed(transport=attended) == [(owner, prefix)], "witness: in Daily the jar is listed"
    for value in _NOT_DAILY:
        mode.value = value
        before = mode.reads
        assert target.unclaimed(transport=attended) == [], value
        assert mode.reads - before == 1, value
    mode.value = "daily"
    assert target.unclaimed(transport=attended) == [(owner, prefix)]
    target.close()

    # An encrypted being is listed in every mode.
    sealed = support.Mode("bulbe")
    keyed = support.seams(p, tmp_path.joinpath("keyed"), suite="as16", index=1, mode=sealed)
    target = support.store(p, keyed)
    keyed_prefix = support.sow(p, target).being_tag[:8]
    target.close()
    target = support.store(p, dict(keyed, single_user=lambda: False))
    assert target.unclaimed(transport=attended) == [(owner, keyed_prefix)]
    target.close()

# ---------------------------------------------------------------------------
# AS17 -- the shipped settings
# ---------------------------------------------------------------------------
def test_as17_the_shipped_settings_require_encryption_and_a_bad_busy_timeout_is_said(p, caplog, tmp_path):
    shipped = p.settings.config_file()
    assert shipped == REPO.joinpath("opti_oignon", "config", "allium.yaml") and shipped.is_file(), shipped
    expected = {"busy_timeout_ms": 5000, "path": "allium", "require_encryption": True}
    assert p.settings.persistence() == expected
    assert p.settings.persistence() == expected, "read again, from what the file's time says is unchanged"
    other = tmp_path.joinpath("allium.yaml")
    other.write_text("persistence:\n  path: garden/beds\n  require_encryption: false\n  busy_timeout_ms: 1234\n",
                     encoding="ascii")
    assert p.settings.persistence(other) == {"busy_timeout_ms": 1234, "path": "garden/beds",
                                             "require_encryption": False}, "the file is read, not the defaults"
    assert p.settings.persistence(tmp_path.joinpath("absent.yaml"))["require_encryption"] is True
    name = p.settings.logger.name
    for bad in (0, 60001, -5, True, "5000", None):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=name):
            assert p.settings.normalise({"busy_timeout_ms": bad})["busy_timeout_ms"] == 5000, bad
        assert [r for r in caplog.records if "busy_timeout_ms" in r.getMessage()], ("the fallback is said", bad)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=name):
        assert [p.settings.normalise({"busy_timeout_ms": ms})["busy_timeout_ms"] for ms in (1, 60000)] == [1, 60000]
    assert not caplog.records, "witness: a timeout in range is taken, and nothing is said"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
