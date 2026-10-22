#!/usr/bin/env python3
"""Contracts for the security mode's verdict, its lockfile under the key that loads, and the key that makes it.

The mode reads security.yaml and its HMAC-signed lockfile; when they cannot
be trusted it fails secure to Bulbe and records why in the audit chain --
whose store, as it opens, asks the mode again.

  * MV1 -- A FAIL-SECURE READ RETURNS, RECORDED ONCE: with the real audit
    chain, a lockfile signed under another key than the one that loads, or
    sources that disagree, read Bulbe at once, and the chain holds one
    entry of it, which verifies.
  * MV2 -- ON THE THREAD THAT RECORDS, THE VERDICT ANSWERS: a read of the
    mode on the thread recording a verdict is answered that verdict, even
    when the files changed meanwhile; nothing is recorded twice.
  * MV3 -- ANOTHER THREAD GETS THE KEPT VERDICT: a read on another thread
    while a verdict is being recorded is answered it at once, and records
    nothing.
  * MV4 -- THE KEY THE MODE READS: the key file's key when it loads, wrapped
    or not; none without a key file; a key file that is there and does not
    load is refused by name, and its bytes are never a key.
  * MV5 -- A KEY FILE THAT DOES NOT LOAD: the mode reads Bulbe, recorded as
    ``lockfile_unverifiable``, even under a lockfile signed with the bytes
    every wrapped key file starts with; escalation and the downgrade
    ceremony refuse ``no_signing_key``, and the mode is not set: nothing is
    written.
  * MV6 -- CHECK SAYS THE MODE AND WHY: for each state of the two sources
    and the key, ``check`` says the mode the platform reads and why -- the
    lockfile verifies, is not verified, cannot be, does not verify (with
    when it says it was signed and when the key file changed), the sources
    disagree -- security.yaml saying a mode, no mode, or nothing --, there
    is no lockfile, or one that cannot be read -- read as the platform reads
    it, and recorded nowhere.
  * MV7 -- A NEW KEY FILE KEEPS THE MODE: made by setup, by the platform's
    two key setups or by ``encrypt``, the key file leaves the mode the
    platform reads as it was: an agreeing lockfile is signed under it, one
    that disagrees is left as it is, none is never made.
  * MV8 -- A KEY FILE NOT MADE LEAVES THE LOCKFILE AS IT WAS: refused as it
    is written, not loading once written, undone by ``encrypt`` when a store
    does not switch, or refused because the lockfile cannot be written --
    said by its name: the lockfile is as it was, byte for byte.
  * MV9 -- MIGRATE WITH THE REAL MODE AND THE REAL CHAIN: in a process of
    its own, a wrapped key file, a lockfile signed under the key before it
    and the installed chromadb: the run ends, moves the memory's store, and
    the chain holds one ``lockfile_hmac_failure``, which verifies. Skipped
    where chromadb does not import.
  * MV10 -- WHOEVER MADE THE KEY FILE, THE LOCKFILE IS SIGNED UNDER IT: a
    rival maker that links its key file first keeps its lockfile when this
    one is refused; one refused after this one puts back this one's; a maker
    interrupted once its key file is linked leaves the lockfile signed under
    it.
  * MV11 -- A KEY FILE THAT CANNOT BE REMOVED KEEPS ITS LOCKFILE: ``encrypt``
    undoing a key file it cannot remove says so, by name, and the lockfile
    stays signed under it: the mode reads as before.
  * MV12 -- THE EMERGENCY STOP ESCALATES WHEN NO KEY LOADS: without a key
    file, or with one that does not load, a forced escalation writes
    security.yaml -- and, with no key file, a lockfile saying Bulbe unsigned
    -- and the mode reads Bulbe at each start, recording nothing; a key file
    made later signs the agreed Bulbe, and security.yaml edited back alone
    leaves it Bulbe. In a folder that takes no new file the lockfile is
    written in place; one that cannot be written at all is kept and said --
    one saying Daily keeps the sources disagreeing, which reads Bulbe -- and
    the stop is recorded whatever became of it. Unforced, the escalation is
    refused.
  * MV13 -- THE LOCKFILE IS WRITTEN WHOLE: a write of the lockfile that
    fails before it is in place leaves the one there as it was, and no
    temporary file; one that succeeds is in place, read-only; in a folder
    that takes no new file it is written in place, as it was before, never
    through a symbolic link at its place.

The modules are loaded from source in an isolation window, every path they
reach -- the key file, the lockfile, security.yaml, the audit chain, the
places of the stores, the process table -- moved into a temporary folder.
"""

import json
import os
import site
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
_MODULES = ("encryption", "db_encryption", "db_utils", "security_mode", "signed_audit_log")
# The key before (the lockfile was signed under it), the key that loads, and a rival maker's.
K0, K1, K2 = bytes(range(32)), bytes(range(1, 33)), bytes(range(2, 34))
PHRASE = "a phrase only these contracts know"
# 2026-06-18 03:33 UTC and 2026-10-21 04:31 UTC.
SIGNED, CHANGED = 1781753580.0, 1792557060.0


def _world(tmp_path, monkeypatch):
    """The five modules in a window, every path they reach moved under ``tmp_path``; the audit chain a new one."""
    pytest.importorskip("sqlcipher3")
    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME"):
        monkeypatch.delenv(name, raising=False)
    row = sqlite3.Row
    loaded, restore = isolate(targets={f"opti_oignon.{name}": source(f"{name}.py") for name in _MODULES},
                              blocked=("opti_oignon.secure_bytes",))

    def closed():
        try:
            restore()
        finally:
            sqlite3.Row = row

    try:
        enc, dbe, sm, sal = (loaded[f"opti_oignon.{name}"] for name in
                             ("encryption", "db_encryption", "security_mode", "signed_audit_log"))
        data, package, proc = tmp_path / "data", tmp_path / "package" / "data", tmp_path / "proc"
        for folder in (data, package, proc):
            folder.mkdir(parents=True)
        keyfile = data / ".keyfile"
        enc._DEFAULT_KEYFILE = sm._DEFAULT_KEYFILE = keyfile
        sm._DATA_DIR, sm._LOCKFILE_PATH, sm._SECURITY_YAML = data, data / ".security_mode_lock", tmp_path / "security.yaml"
        dbe._STORE_PLACES, dbe._LONE_STORES, dbe._PROC = (data, package), [], proc
        sal.signed_audit_log = sal.SignedAuditLog(data / "audit_chain.db")
        sm.security_mode_manager = sm.SecurityModeManager()
        world = SimpleNamespace(enc=enc, dbe=dbe, sm=sm, sal=sal, data=data, package=package, keyfile=keyfile,
                                lock=sm._LOCKFILE_PATH, yaml=sm._SECURITY_YAML, restore=closed)
    except BaseException:
        closed()
        raise
    return world


def _key_file(world, monkeypatch, key=K1, *, wrapped=False):
    """The key file, made as setup makes one: wrapped under the passphrase, or in the unprotected format."""
    if wrapped:
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", PHRASE)
        world.enc.save_keyfile(key, kdf_name="random")
    else:
        world.enc.save_keyfile(key, kdf_name="random", allow_unprotected=True)


def _sources(world, yaml_mode, lock_mode=None, key=None, *, stamp=None):
    """security.yaml saying ``yaml_mode``; the lockfile saying ``lock_mode`` signed under ``key``, at ``stamp``."""
    world.yaml.write_text(f"security_mode: {yaml_mode}\n", encoding="utf-8")
    if lock_mode is None:
        return
    if stamp is None:
        world.sm._write_lockfile(lock_mode, "leon", key)
        return
    signature = world.sm._compute_lockfile_hmac(lock_mode, stamp, "leon", key)
    world.lock.write_text(f"MODE:{lock_mode}\nTIMESTAMP:{stamp}\nUSER_ID:leon\nHMAC:{signature}\n", encoding="utf-8")


def _recorder(world):
    """The mode's records, kept in a list instead of the chain."""
    recorded = []
    world.sm._audit_log = lambda event, severity="INFO", **details: recorded.append(event)
    return recorded


def _chain(world):
    """The audit chain's event types, oldest first."""
    log = world.sal.signed_audit_log
    return [event["event_type"] for event in sorted(log.get_events(limit=50), key=lambda event: event["id"])]


def _in_a_thread(call, name="reader"):
    """``call()`` on a thread of its own, waited on for twenty seconds: ``(still running, what it returned)``."""
    got = {}
    worker = threading.Thread(target=lambda: got.setdefault("value", call()), name=name, daemon=True)
    worker.start()
    worker.join(20)
    return worker.is_alive(), got.get("value", "nothing returned")


def _raw(key):
    return key.as_bytes() if hasattr(key, "as_bytes") else key


def _state(path):
    """A file's bytes and permission bits, or None."""
    return (path.read_bytes(), os.stat(path).st_mode & 0o777) if path.exists() else None


# ---------------------------------------------------------------------------
# MV1 -- a fail-secure read returns, recorded once
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("failure", ["another_key", "disagree"])
def test_mv1_a_fail_secure_read_returns_bulbe_recorded_once_in_the_real_chain(tmp_path, monkeypatch, failure):
    world = _world(tmp_path, monkeypatch)
    try:
        _key_file(world, monkeypatch)
        if failure == "another_key":
            _sources(world, "daily", "daily", K0)
        else:
            _sources(world, "daily", "bulbe", K1)
        running, mode = _in_a_thread(world.sm.get_current_mode)
        assert not running, "the mode's fail-secure read waited on its own record"
        assert mode == "bulbe" and world.sm.get_current_mode() == "bulbe"
        event = "lockfile_hmac_failure" if failure == "another_key" else "security_mode_mismatch"
        assert _chain(world) == [event], "recorded once, by the read that found it"
        assert world.sal.signed_audit_log.verify_chain() == (True, None, 1)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# MV2 -- on the thread that records, the verdict answers
# ---------------------------------------------------------------------------
def test_mv2_on_the_thread_recording_a_verdict_a_read_of_the_mode_is_answered_that_verdict(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        _key_file(world, monkeypatch)
        _sources(world, "daily", "daily", K0)
        recorded, answers = [], []

        def record(event, severity="INFO", **details):
            recorded.append(event)
            if len(recorded) == 1:
                # The files change while the verdict is recorded, as the record's own store asks the mode.
                world.sm._write_lockfile("daily", "leon", K0)
                answers.append(world.sm.get_current_mode())

        world.sm._audit_log = record
        running, mode = _in_a_thread(world.sm.get_current_mode)
        assert not running and mode == "bulbe"
        assert recorded == ["lockfile_hmac_failure"] and answers == ["bulbe"], (recorded, answers)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# MV3 -- another thread gets the kept verdict
# ---------------------------------------------------------------------------
def test_mv3_another_thread_reading_while_a_verdict_is_recorded_gets_it_at_once(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    recording, release = threading.Event(), threading.Event()
    try:
        _key_file(world, monkeypatch)
        _sources(world, "daily", "daily", K0)
        recorded = []

        def record(event, severity="INFO", **details):
            recorded.append(event)
            if len(recorded) == 1:
                recording.set()
                release.wait(20)

        world.sm._audit_log = record
        first = threading.Thread(target=world.sm.get_current_mode, name="recording", daemon=True)
        first.start()
        assert recording.wait(20)
        running, mode = _in_a_thread(world.sm.get_current_mode, name="other")
        release.set()
        first.join(20)
        assert not running and mode == "bulbe"
        assert recorded == ["lockfile_hmac_failure"], "another thread reads the kept verdict, and records nothing"
    finally:
        release.set()
        world.restore()


# ---------------------------------------------------------------------------
# MV4 -- the key the mode reads
# ---------------------------------------------------------------------------
def test_mv4_the_key_the_mode_reads_is_the_key_file_s_when_it_loads_and_never_its_bytes(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        sm = world.sm
        assert sm._load_signing_key() is None, "no key file: no key"
        _key_file(world, monkeypatch, wrapped=True)
        assert _raw(sm._load_signing_key()) == K1
        monkeypatch.delenv("OPTI_KEYFILE_PASSPHRASE")
        with pytest.raises(sm.SigningKeyUnavailable) as refused:
            sm._load_signing_key()
        assert str(world.keyfile) in str(refused.value) and "OPTI_KEYFILE_PASSPHRASE" in str(refused.value)
        world.keyfile.unlink()
        _key_file(world, monkeypatch, K0)
        assert _raw(sm._load_signing_key()) == K0 != world.keyfile.read_bytes()[:32], "the key, not the file's bytes"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# MV5 -- a key file that does not load
# ---------------------------------------------------------------------------
def test_mv5_a_key_file_that_does_not_load_reads_bulbe_and_nothing_is_signed(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        sm = world.sm
        _key_file(world, monkeypatch, wrapped=True)
        monkeypatch.delenv("OPTI_KEYFILE_PASSPHRASE")
        # Signed with the first 32 bytes of the wrapped key file: text every wrapped key file starts with.
        _sources(world, "daily", "daily", world.keyfile.read_bytes()[:32])
        recorded = _recorder(world)
        assert sm.get_current_mode() == "bulbe" and recorded == ["lockfile_unverifiable"], recorded

        world.lock.unlink()
        _sources(world, "daily")
        manager = sm.SecurityModeManager()
        assert manager.get_current_mode() == "daily", "no lockfile: security.yaml"
        before = (_state(world.yaml), _state(world.lock))
        refused = manager.escalate_to_bulbe("leon")
        assert refused["success"] is False and refused["error"] == "no_signing_key", refused
        assert str(world.keyfile) in refused["message"]
        assert manager.initialize_mode("bulbe", "leon") is False
        assert (_state(world.yaml), _state(world.lock)) == before, "nothing written"

        _sources(world, "bulbe")
        manager = sm.SecurityModeManager()
        monkeypatch.setattr(sm, "DOWNGRADE_COOLDOWN_SECONDS", 0)
        pending = manager.request_downgrade("leon")
        assert pending["pending"] is True
        before = (_state(world.yaml), _state(world.lock))
        refused = manager.confirm_downgrade("leon", pending["request_id"], manager.get_pending_visual_code(), "pw")
        assert refused["success"] is False and refused["error"] == "no_signing_key", refused
        assert (_state(world.yaml), _state(world.lock)) == before and manager.get_current_mode() == "bulbe"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# MV6 -- check says the mode and why
# ---------------------------------------------------------------------------
_SAID = {
    "no_lock": "security mode: daily, from security.yaml: no lock file",
    "verified": "security mode: daily; its lock file verifies under the key that loads",
    "no_key_file": "security mode: daily; its lock file is not verified: no key file",
    "another_key": ("security mode: bulbe, failing secure: its lock file (daily, signed 2026-06-18 03:33 UTC by its own "
                    "account) does not verify under the key that loads (key file changed 2026-10-21 04:31 UTC): it "
                    "was signed under another key, or written without one; the platform records "
                    "lockfile_hmac_failure at each start until it is signed again under this key, which leaving "
                    "Bulbe through the downgrade ceremony does"),
    "does_not_load": "if it is wrapped under a passphrase, set OPTI_KEYFILE_PASSPHRASE: its lock file cannot be verified",
    "disagree": "security mode: bulbe, failing secure: security.yaml says daily, its lock file bulbe",
    "yaml_no_mode": "security mode: bulbe, failing secure: security.yaml says 'Daily' (no mode: read as bulbe), its "
                    "lock file daily",
    "yaml_missing": "security mode: bulbe, failing secure: security.yaml says nothing (it is missing: read as daily), "
                    "its lock file bulbe",
    "lock_unread": "security mode: daily, from security.yaml: its lock file cannot be read here (PermissionError), read "
                   "as none",
    "yaml_no_mapping": "security mode: bulbe, failing secure: security.yaml says 'daily', where a mapping is due (read as "
                       "bulbe), its lock file daily",
    "yaml_no_mapping_agreeing": "security mode: bulbe; its lock file verifies under the key that loads; security.yaml "
                                "says 'bulbe', where a mapping is due (read as bulbe)",
    "does_not_load_yaml_no_mode": "its lock file cannot be verified; security.yaml says 'Daily' (no mode: read as bulbe)",
    "another_key_yaml_no_mode": ("which leaving Bulbe through the downgrade ceremony does; security.yaml says 'Daily' (no "
                                 "mode: read as bulbe)"),
}


@pytest.mark.parametrize("state", list(_SAID))
def test_mv6_check_says_the_mode_the_platform_reads_and_why(tmp_path, monkeypatch, capsys, state):
    world = _world(tmp_path, monkeypatch)
    try:
        if state in ("verified", "another_key", "disagree", "yaml_no_mode", "yaml_missing", "yaml_no_mapping",
                     "yaml_no_mapping_agreeing", "another_key_yaml_no_mode"):
            _key_file(world, monkeypatch)
        if state in ("does_not_load", "does_not_load_yaml_no_mode"):
            _key_file(world, monkeypatch, wrapped=True)
            monkeypatch.delenv("OPTI_KEYFILE_PASSPHRASE")
        lock = {"no_lock": (None, None), "verified": ("daily", K1), "no_key_file": ("daily", K0),
                "another_key": ("daily", K0), "does_not_load": ("daily", K1), "disagree": ("bulbe", K1),
                "yaml_no_mode": ("daily", K1), "yaml_missing": ("bulbe", K1), "lock_unread": ("daily", K0),
                "yaml_no_mapping": ("daily", K1), "yaml_no_mapping_agreeing": ("bulbe", K1),
                "does_not_load_yaml_no_mode": ("daily", K1), "another_key_yaml_no_mode": ("daily", K0)}[state]
        _sources(world, "Daily" if state.endswith("yaml_no_mode") else "daily", *lock,
                 stamp=SIGNED if state == "another_key" else None)
        if state == "yaml_missing":
            world.yaml.unlink()
        if state in ("yaml_no_mapping", "yaml_no_mapping_agreeing"):
            world.yaml.write_text("daily\n" if state == "yaml_no_mapping" else "bulbe\n", encoding="utf-8")
        if state == "lock_unread":
            if os.geteuid() == 0:
                pytest.skip("root reads a file of mode 0")
            os.chmod(world.lock, 0)
        if world.keyfile.exists():
            os.utime(world.keyfile, (CHANGED, CHANGED))
        recorded = _recorder(world)
        report = world.sm.mode_report()
        assert recorded == [], "check records nothing"
        assert report["mode"] == world.sm.SecurityModeManager().get_current_mode(), "said as the platform reads it"
        assert world.dbe.main(["check"]) == 0
        said = [line for line in capsys.readouterr().out.splitlines() if line.startswith("security mode: ")]
        assert len(said) == 1 and _SAID[state] in said[0], said
        if state == "does_not_load":
            assert said[0].startswith(f"security mode: bulbe, failing secure: the key file {world.keyfile} does not "
                                      "load here"), said
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# MV7 -- a new key file keeps the mode
# ---------------------------------------------------------------------------
def _make_key_file(world, monkeypatch, capsys, maker):
    """The key file, made by ``maker``; what ``encrypt`` said, when it made it."""
    if maker == "save_keyfile":
        world.enc.save_keyfile(K1, kdf_name="random", allow_unprotected=True)
    elif maker == "setup_random_key":
        assert world.enc.EncryptionManager().setup_random_key() is True
    elif maker == "setup_from_passphrase":
        assert world.enc.EncryptionManager().setup_from_passphrase(PHRASE) is True
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", PHRASE)
    else:
        conn = sqlite3.connect(str(world.data / "notes.db"))
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, words TEXT NOT NULL)")
        conn.execute("INSERT INTO notes (words) VALUES ('a line')")
        conn.commit()
        conn.close()
        capsys.readouterr()
        assert world.dbe.main(["encrypt"]) == 0
        return capsys.readouterr().out
    return None


@pytest.mark.parametrize("lock", ["absent", "daily", "bulbe", "disagree"])
@pytest.mark.parametrize("maker", ["save_keyfile", "setup_random_key", "setup_from_passphrase", "encrypt"])
def test_mv7_a_new_key_file_keeps_the_mode_the_platform_reads(tmp_path, monkeypatch, capsys, lock, maker):
    world = _world(tmp_path, monkeypatch)
    try:
        yaml_mode, lock_mode = {"absent": ("daily", None), "daily": ("daily", "daily"), "bulbe": ("bulbe", "bulbe"),
                                "disagree": ("daily", "bulbe")}[lock]
        _sources(world, yaml_mode, lock_mode, K0)
        recorded = _recorder(world)
        before, kept = world.sm.mode_report()["mode"], _state(world.lock)
        world.enc.save_keyfile(K0, kdf_name="random", path=tmp_path / "elsewhere" / ".keyfile", allow_unprotected=True)
        assert _state(world.lock) == kept, "another key file than the mode's leaves its lockfile as it is"
        said = _make_key_file(world, monkeypatch, capsys, maker)
        assert world.keyfile.exists()
        assert world.sm.SecurityModeManager().get_current_mode() == before == ("bulbe" if lock in ("bulbe", "disagree")
                                                                               else "daily")
        if lock == "disagree":
            assert _state(world.lock) == kept, "sources that disagree are left as they are"
            assert recorded == ["lockfile_hmac_failure"]
        else:
            assert recorded == [], "an agreeing lockfile is signed under the new key: nothing to record"
        if lock == "absent":
            assert not world.lock.exists(), "no lockfile is made for a key"
        if lock in ("daily", "bulbe"):
            assert world.sm.mode_report()["verified"] is True
            assert "USER_ID:leon\n" in world.lock.read_text(encoding="utf-8"), "signed again for the same user"
        if said is not None:
            assert f"security mode: {before}" in said, said
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# MV8 -- a key file not made leaves the lockfile as it was
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["refused_as_written", "does_not_load_once_written", "a_store_does_not_switch",
                                 "the_lockfile_cannot_be_written"])
def test_mv8_a_key_file_not_made_leaves_the_lockfile_as_it_was(tmp_path, monkeypatch, capsys, how):
    world = _world(tmp_path, monkeypatch)
    try:
        _sources(world, "daily", "daily", K0)
        os.chmod(world.lock, 0o440)
        kept = _state(world.lock)
        if how == "refused_as_written":
            def refused(fpath, data):
                raise world.enc.KeyfileExists(str(fpath))
            monkeypatch.setattr(world.enc, "_write_keyfile", refused)
            with pytest.raises(world.enc.KeyfileExists):
                world.enc.save_keyfile(K1, kdf_name="random", allow_unprotected=True)
        else:
            for name in ("notes.db", "other.db"):
                conn = sqlite3.connect(str(world.data / name))
                conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, words TEXT NOT NULL)")
                conn.execute("INSERT INTO notes (words) VALUES ('a line')")
                conn.commit()
                conn.close()
            if how == "does_not_load_once_written":
                monkeypatch.setattr(world.dbe, "_get_db_encryption_key", lambda: b"another key entirely, 32 bytes!")
            elif how == "the_lockfile_cannot_be_written":
                def unwritable(mode, user_id, key):
                    raise PermissionError(13, "Permission denied")
                monkeypatch.setattr(world.sm, "_write_lockfile", unwritable)
            else:
                switch, seen = world.dbe._switch, []

                def second_fails(store, copy, folder, records):
                    seen.append(store)
                    if len(seen) == 2:
                        raise OSError("the second store does not switch")
                    return switch(store, copy, folder, records)

                monkeypatch.setattr(world.dbe, "_switch", second_fails)
            capsys.readouterr()
            assert world.dbe.main(["encrypt"]) == 1
            refused = capsys.readouterr().err
            assert "nothing was changed" in refused, refused
            if how == "the_lockfile_cannot_be_written":
                assert ("the security mode's lockfile cannot be signed under the new key (Permission denied)" in refused
                        and str(world.lock) in refused), "refused by the lockfile's name"
        assert not world.keyfile.exists()
        assert _state(world.lock) == kept, "the lockfile as it was, byte for byte"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# MV10 -- a rival maker's key file keeps the lockfile signed under it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("first", ["the_rival", "this_maker", "an_interrupt_after_the_link"])
def test_mv10_the_key_file_made_keeps_the_lockfile_signed_under_it_whoever_made_it(tmp_path, monkeypatch, first):
    world = _world(tmp_path, monkeypatch)
    try:
        sm, enc = world.sm, world.enc
        _sources(world, "daily", "daily", K0)
        recorded = _recorder(world)
        real, rival = enc._write_keyfile, {}

        def contested(fpath, data):
            monkeypatch.setattr(enc, "_write_keyfile", real)
            if first == "an_interrupt_after_the_link":
                # The key file is linked, then the maker is interrupted before it returns.
                real(fpath, data)
                raise KeyboardInterrupt
            if first == "the_rival":
                # The rival signs under its key and links its key file first: this one is refused.
                enc.save_keyfile(K2, kdf_name="random", allow_unprotected=True)
                raise enc.KeyfileExists(str(fpath))
            # The rival signs under its key before this key file is linked, and is refused once it is.
            rival["before"] = sm.sign_for_a_new_key(K2, world.keyfile)
            real(fpath, data)

        monkeypatch.setattr(enc, "_write_keyfile", contested)
        if first == "an_interrupt_after_the_link":
            with pytest.raises(KeyboardInterrupt):
                enc.save_keyfile(K1, kdf_name="random", allow_unprotected=True)
        elif first == "the_rival":
            with pytest.raises(enc.KeyfileExists):
                enc.save_keyfile(K1, kdf_name="random", allow_unprotected=True)
        else:
            enc.save_keyfile(K1, kdf_name="random", allow_unprotected=True)
            sm.put_lock_back(rival["before"], K2)
        assert _raw(sm._load_signing_key()) == (K2 if first == "the_rival" else K1)
        assert sm.mode_report()["verified"] is True, "the lockfile is signed under the key file that was made"
        assert sm.SecurityModeManager().get_current_mode() == "daily" and recorded == []
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# MV11 -- a key file that cannot be removed keeps its lockfile, and is said
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["does_not_load_once_written", "a_store_does_not_switch"])
def test_mv11_a_key_file_that_cannot_be_removed_keeps_its_lockfile_and_is_said(tmp_path, monkeypatch, capsys, how):
    world = _world(tmp_path, monkeypatch)
    try:
        _sources(world, "daily", "daily", K0)
        for name in ("notes.db", "other.db"):
            conn = sqlite3.connect(str(world.data / name))
            conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, words TEXT NOT NULL)")
            conn.execute("INSERT INTO notes (words) VALUES ('a line')")
            conn.commit()
            conn.close()
        real_os = world.dbe.os

        class Stuck:
            def __getattr__(self, name):
                return getattr(real_os, name)

            def unlink(self, path, *args, **kwargs):
                if Path(path) == world.keyfile:
                    raise PermissionError(13, "Permission denied", str(path))
                return real_os.unlink(path, *args, **kwargs)

        monkeypatch.setattr(world.dbe, "os", Stuck())
        if how == "does_not_load_once_written":
            monkeypatch.setattr(world.dbe, "_get_db_encryption_key", lambda: b"another key entirely, 32 bytes!")
        else:
            switch, seen = world.dbe._switch, []

            def second_fails(store, copy, folder, records):
                seen.append(store)
                if len(seen) == 2:
                    raise OSError("the second store does not switch")
                return switch(store, copy, folder, records)

            monkeypatch.setattr(world.dbe, "_switch", second_fails)
        capsys.readouterr()
        assert world.dbe.main(["encrypt"]) == 1
        refused = capsys.readouterr().err
        assert world.keyfile.exists() and str(world.keyfile) in refused, refused
        assert ("could not be removed (PermissionError)" if how == "does_not_load_once_written" else
                "nothing was changed but the key file; the key file") in refused, refused
        assert "(PermissionError), under which no store is encrypted: delete it" in refused or (
            how == "does_not_load_once_written"), refused
        assert world.sm.mode_report()["verified"] is True, "the lockfile stays signed under the key file that stays"
        assert world.sm.SecurityModeManager().get_current_mode() == "daily"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# MV12 -- the emergency stop escalates when no key loads
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_file", ["none", "does_not_load", "none_lockfile_unwritable", "none_folder_read_only",
                                      "none_lockfile_a_folder"])
def test_mv12_the_emergency_stop_escalates_when_no_key_loads(tmp_path, monkeypatch, key_file):
    world = _world(tmp_path, monkeypatch)
    try:
        sm = world.sm
        if key_file == "does_not_load":
            _key_file(world, monkeypatch, wrapped=True)
            monkeypatch.delenv("OPTI_KEYFILE_PASSPHRASE")
            _sources(world, "daily")
        elif key_file == "none_lockfile_a_folder":
            # A folder where the lockfile goes, with something in it: nothing can replace it, and it reads as none.
            _sources(world, "daily")
            (world.lock / "kept").mkdir(parents=True)
        else:
            _sources(world, "daily", "daily", K0)
        if key_file == "none_lockfile_unwritable":
            write = sm._write_lockfile

            def unwritable(mode, user_id, key):
                if key is None:
                    raise PermissionError(13, "Permission denied")
                return write(mode, user_id, key)

            monkeypatch.setattr(sm, "_write_lockfile", unwritable)
        if key_file == "none_folder_read_only":
            if os.geteuid() == 0:
                pytest.skip("root writes a folder of mode 0555")
            os.chmod(world.data, 0o555)
        recorded = _recorder(world)
        manager = sm.SecurityModeManager()
        assert manager.get_current_mode() == "daily"
        kept = None if key_file == "none_lockfile_a_folder" else _state(world.lock)
        refused = manager.escalate_to_bulbe("leon")
        assert refused["success"] is False and refused["error"] == "no_signing_key", "without force, refused"
        forced = manager.escalate_to_bulbe("emergency-stop", force=True)
        assert forced["success"] is True and forced["mode"] == "bulbe", forced
        assert recorded[-1:] == ["security_mode_escalated"], "the stop is recorded, whatever became of the lockfile"
        assert {"none": "security.yaml and an unsigned lockfile say it",
                "does_not_load": "the key file does not load: security.yaml says it",
                "none_lockfile_unwritable": "the lockfile could not be written (PermissionError): it says daily, which "
                                            "disagrees with security.yaml, and the mode reads Bulbe from that",
                "none_folder_read_only": "security.yaml and an unsigned lockfile say it",
                "none_lockfile_a_folder": "the lockfile could not be written (IsADirectoryError): security.yaml alone "
                                          "says it"}[key_file] in forced["message"], forced
        assert "security_mode: bulbe" in world.yaml.read_text(encoding="utf-8")
        if key_file == "none_folder_read_only":
            # A folder that takes no new file: the lockfile was written in place.
            os.chmod(world.data, 0o755)
        del recorded[:]
        if key_file == "none_lockfile_unwritable":
            assert _state(world.lock) == kept, "the lockfile that could not be written is kept, as it was"
            assert manager.get_current_mode() == "bulbe" and recorded == ["security_mode_mismatch"], (
                "it says Daily: the sources disagree and the mode reads Bulbe from that, recorded", recorded)
            return
        if key_file == "none_lockfile_a_folder":
            assert world.lock.is_dir() and manager.get_current_mode() == "bulbe" and recorded == [], (
                "a lockfile that reads as none: security.yaml alone says Bulbe", recorded)
            return
        assert manager.get_current_mode() == "bulbe" and sm.SecurityModeManager().get_current_mode() == "bulbe"
        assert recorded == [], "the two sources agree on Bulbe: a start records nothing"
        if key_file == "does_not_load":
            assert _state(world.lock) == kept is None, "a key file that does not load keeps the lockfile it has"
            monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", PHRASE)
            assert sm.SecurityModeManager().get_current_mode() == "bulbe" and recorded == [], "its passphrase back"
            return
        assert world.lock.read_text(encoding="utf-8").startswith("MODE:bulbe\n"), "the lockfile says Bulbe, unsigned"
        status = manager.status()
        assert (status["lockfile_exists"], status["sources_agree"], status["hmac_valid"]) == (True, True, None), (
            "a lockfile no key can check is said not verified, never invalid", status)
        world.enc.save_keyfile(K1, kdf_name="random", allow_unprotected=True)
        assert sm.SecurityModeManager().get_current_mode() == "bulbe" and recorded == [], (
            "a key file made later signs the agreed Bulbe", recorded)
        assert sm.mode_report()["verified"] is True and sm.SecurityModeManager().status()["hmac_valid"] is True
        world.yaml.write_text("security_mode: daily\n", encoding="utf-8")
        assert sm.SecurityModeManager().get_current_mode() == "bulbe", "security.yaml edited back alone: still Bulbe"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# MV13 -- the lockfile is written whole
# ---------------------------------------------------------------------------
def test_mv13_a_lockfile_write_that_fails_leaves_the_one_there_as_it_was(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        sm = world.sm
        _sources(world, "daily", "daily", K0)
        kept = _state(world.lock)
        real_os = sm.os

        class FailingSync:
            def __getattr__(self, name):
                return getattr(real_os, name)

            def fsync(self, fd):
                raise OSError(28, "No space left on device")

        monkeypatch.setattr(sm, "os", FailingSync())
        with pytest.raises(OSError):
            sm._write_lockfile("bulbe", "leon", K1)
        monkeypatch.setattr(sm, "os", real_os)
        assert _state(world.lock) == kept, "a write that fails before it is in place leaves the lockfile as it was"
        assert [entry.name for entry in world.data.iterdir() if ".tmp-" in entry.name] == [], "no temporary file left"
        sm._write_lockfile("bulbe", "leon", K1)
        assert world.lock.read_text(encoding="utf-8").startswith("MODE:bulbe\n") and _state(world.lock)[1] == 0o400
        assert sm._verify_lockfile(sm._read_lockfile(), K1)
        if os.geteuid() != 0:
            os.chmod(world.data, 0o555)
            try:
                sm._write_lockfile("daily", "leon", K1)
            finally:
                os.chmod(world.data, 0o755)
            assert world.lock.read_text(encoding="utf-8").startswith("MODE:daily\n") and _state(world.lock)[1] == 0o400, (
                "a folder that takes no new file: the lockfile is written in place")
            # A symbolic link at the lockfile's place, in such a folder: refused, its target never touched.
            outside = tmp_path / "outside.txt"
            outside.write_text("not the lockfile\n", encoding="utf-8")
            os.chmod(outside, 0o644)
            world.lock.unlink()
            world.lock.symlink_to(outside)
            os.chmod(world.data, 0o555)
            try:
                with pytest.raises(OSError):
                    sm._write_lockfile("bulbe", "leon", K1)
            finally:
                os.chmod(world.data, 0o755)
            assert _state(outside) == (b"not the lockfile\n", 0o644), "the link's target is never written nor made 0600"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# MV9 -- migrate with the real mode and the real chain
# ---------------------------------------------------------------------------
_REAL = r'''
import json, os, socket, sys
from pathlib import Path
repo, root = sys.argv[1], Path(sys.argv[2])
for name in ("OPTI_ENCRYPTION_KEY", "RAG_HOME"):
    os.environ.pop(name, None)
os.environ["OPTI_KEYFILE_PASSPHRASE"] = "a phrase only this contract knows"


def refuse(*args, **kwargs):
    raise OSError("the network is refused in this contract")


socket.socket.connect = refuse
socket.socket.connect_ex = refuse
socket.create_connection = refuse
socket.getaddrinfo = refuse
import chromadb
from chromadb.config import Settings
sys.meta_path[:] = [finder for finder in sys.meta_path if "editable" not in repr(finder).lower()]
sys.path.insert(0, repo)
from opti_oignon import (chroma_migration as cm, db_encryption as dbe, encryption as enc, security_mode as sm,
                         signed_audit_log as sal, vector_db as vdb)
modules = (cm, dbe, enc, sm, sal, vdb)
assert all(module.__file__.startswith(repo) for module in modules), [module.__file__ for module in modules]
data, package, proc = root / "data", root / "package" / "data", root / "proc"
for folder in (data, package, proc):
    folder.mkdir(parents=True, exist_ok=True)
dbe._STORE_PLACES, dbe._LONE_STORES, dbe._PROC = (data, package), [], proc
enc._DEFAULT_KEYFILE = sm._DEFAULT_KEYFILE = data / ".keyfile"
sm._DATA_DIR, sm._LOCKFILE_PATH, sm._SECURITY_YAML = data, data / ".security_mode_lock", root / "security.yaml"
sal.signed_audit_log = sal.SignedAuditLog(data / "audit_chain.db")
sm.security_mode_manager = sm.SecurityModeManager()
# The key file wrapped under the passphrase; the mode Daily, from security.yaml alone.
enc.save_keyfile(bytes(range(1, 33)), kdf_name="random")
sm._SECURITY_YAML.write_text("security_mode: daily\n", encoding="utf-8")
# The memory's facts, encrypted under the key that loads, and ChromaDB's store of the memory.
conn = vdb.safe_connect(package / "memory_facts.db")
conn.execute("CREATE TABLE memory_facts (id TEXT PRIMARY KEY, text TEXT NOT NULL, active INTEGER NOT NULL)")
conn.execute("INSERT INTO memory_facts VALUES ('m1', 'a fact', 1)")
conn.commit()
conn.close()
client = chromadb.PersistentClient(path=str(package / "rag" / "chroma_db"), settings=Settings(anonymized_telemetry=False))
client.get_or_create_collection(name="oo_memories", metadata={"hnsw:space": "cosine"}).add(
    ids=["m1"], documents=["a fact"], metadatas=[{"user_id": "local"}], embeddings=[[0.5, 0.5]])
del client
# Then the lockfile, signed under the key before this one, as on the machine; the command starts afresh.
sm._write_lockfile("daily", "leon", bytes(range(32)))
sm.security_mode_manager = sm.SecurityModeManager()
code = dbe.main(["migrate"])
events = [event["event_type"] for event in sal.signed_audit_log.get_events(limit=50)]
print("RESULT " + json.dumps({"code": code, "events": events, "chain": list(sal.signed_audit_log.verify_chain()),
                              "mode": sm.get_current_mode()}))
'''


def test_mv9_migrate_with_the_real_mode_and_the_real_chain_ends_and_records_once(tmp_path):
    import importlib.util

    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("chromadb does not import here")
    pytest.importorskip("sqlcipher3")
    # Its home in a temporary folder, its user site kept: where this interpreter finds chromadb's numpy.
    env = dict(os.environ, HOME=str(tmp_path / "home"), PYTHONUSERBASE=site.getuserbase(), PYTHONDONTWRITEBYTECODE="1")
    (tmp_path / "home").mkdir()
    try:
        child = subprocess.run([sys.executable, "-c", _REAL, str(_REPO), str(tmp_path / "root")],
                               capture_output=True, text=True, timeout=180, env=env)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"the run never ended (stdout: {str(exc.stdout)[-1500:]!r}; stderr: {str(exc.stderr)[-1500:]!r})")
    lines = [line for line in child.stdout.splitlines() if line.startswith("RESULT ")]
    assert child.returncode == 0 and lines, (child.returncode, child.stdout[-2000:], child.stderr[-3000:])
    result = json.loads(lines[-1][len("RESULT "):])
    assert result["code"] == 0 and result["mode"] == "bulbe", result
    assert result["events"] == ["lockfile_hmac_failure"] and result["chain"] == [True, None, 1], result
    assert "chroma_db: 1 collection(s) moved into the knowledge base's store" in child.stdout, child.stdout
    assert "encrypted under the key that loads" in child.stdout, child.stdout
    assert child.stderr.count("Lockfile HMAC verification FAILED") <= 1, child.stderr[-3000:]
