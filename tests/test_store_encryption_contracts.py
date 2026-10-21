#!/usr/bin/env python3
"""Contracts for the encryption of every store at rest: offline, all of the stores or none.

The platform keeps its SQLite stores in two places, the repository's data
directory and the package's. Every connection it opens keys the store once
a key loads, so a key made while one store stays in plaintext leaves that
store unreadable. The stores are therefore encrypted together, the platform
stopped, by ``python3 -m opti_oignon.db_encryption encrypt``
(``encrypt_all_stores``), and the server makes no key and moves no store
while one is kept in plaintext.

  * SE1 -- EVERY STORE, BOTH PLACES: the plaintext stores of both places,
    of the knowledge base's folder and the sandbox's audit log are
    encrypted, and the platform's own connection reads each as it was --
    its rows, those a log held that no process folded in, its full-text
    matches, its user_version, application_id and auto_vacuum, its file's
    mode; no file SQLite keeps beside a store is left there; the plaintext
    stores sit in a folder of their own, 0700, beside each; the key file
    made is 0600 and said; an empty file, a file that is no store, a
    plugin's own store (its plugin reads it in plaintext) and a user's file
    in a project are left alone, the last two named; a second run moves
    nothing, and says the folders of plaintext copies the first one kept.
  * SE2 -- ALL OR NONE, BEFORE ANY MOVE: a store that fails its own
    integrity check, or a copy that does not hold what its store holds,
    refuses the run by name: every store as it was, no copy, no folder,
    and no key file -- the one this run made goes, one that was there
    stays.
  * SE3 -- ALL OR NONE, WHILE SWITCHING: a switch that fails puts back
    every step made; the run is refused by name; every store as it was, no
    copy, no folder, and the key file the run wrote removed -- a key file
    it did not write left as it was, the stores encrypted under it still
    opening. A store that cannot be put back keeps its plaintext, its
    folder and the key file, and the command says where its plaintext is,
    ending with 3: neither a success nor a refusal; copies that cannot be
    removed once every store is put back are said, to delete, the run a
    refusal. A directory that cannot be synced once every store is
    switched undoes nothing, and is said.
  * SE4 -- NEVER UNDER A RUNNING PLATFORM, NEVER TWICE: a process that holds
    a store or a file beside it open -- a deleted one included -- or whose
    command line runs the platform refuses the run by name, read from the
    process table, before and again after the copies, changing nothing;
    this process, its parent (the shell that runs the command), a process
    that cannot be read and a command line that only passes through a
    folder named like the platform are not counted; the real process table
    names a child that holds a store; no process table is a refusal by
    name; so is another run under way, and what an interrupted run left
    that is not plainly a copy to make again -- a key file's temp among
    them --, each named with what is known of it: a second name for a file
    kept elsewhere, to delete, or the only one, to move -- never judged
    through a symbolic link, and a temp a linked key file may lead to kept.
    The parent counts when it runs the platform without naming the
    command.
  * SE5 -- THE KEY: one that loads -- a key file, or OPTI_ENCRYPTION_KEY --
    is taken and never replaced; a new one is held in memory while the
    copies are made, written only before the switch, and removed when it
    does not load once written; a key file that does not load refuses the
    run naming OPTI_KEYFILE_PASSPHRASE; with OPTI_KEYFILE_PASSPHRASE set,
    the key file made is wrapped under it; a store already encrypted under
    the key is left; one the key does not open, one encrypted with no key
    here and a store that is a symbolic link refuse the run, nothing made.
  * SE6 -- THE SERVER REFUSES WHAT WOULD BREAK THE STORES: setup makes no
    key while a store is kept in plaintext (409 naming the command, no key
    file, security.yaml untouched), and makes one when none is; the
    migration route moves nothing (409 naming the command); the status
    route counts the stores of both places.
  * SE7 -- THE COMMAND: ``check`` says the places, each store's state, who
    holds the stores and every folder of plaintext copies, to delete once
    the key is kept safe elsewhere; ``encrypt``
    encrypts and says the places, where the key file is and to keep a copy
    of it elsewhere, and where the plaintext copies are, its own and those
    an earlier run kept, 0; a refusal says why on standard error, 1.
  * SE8 -- A RUN THAT DIES: killed outright at any step, a run leaves every
    store whole under its name -- plaintext, or encrypted under a key file
    that loads -- and the next run finishes it; SIGTERM, SIGHUP or SIGINT,
    before or after any step, once or twice, is said at once and stops the
    run at its next safe point -- before a copy, before the processes are
    read again, before the key is written, before a store is switched --,
    nothing done after it but putting everything back; once the last
    store's switch has begun, it lets the run finish, and is said -- never a
    store lost; a second signal while the run puts everything back changes
    nothing it does; a hang-up the command was started with ignored (nohup)
    stays ignored, unsaid; with standard error closed, the notice lands in
    no file that took its number.
  * SE9 -- THE SANDBOX'S LOG, WHERE ITS MANAGER OPENS IT: for the shipped
    configuration and each shape its manager reads -- one with a value that
    does not build among them, where the manager opens its defaults --, the
    command names the audit log at the path the manager joins from it, and
    a manager built on a base of the test's opens its log there; the
    command reads the manager's own configuration file, names the
    configured log whether it holds anything or not, and takes it once it
    does -- never importing the manager, whose import builds one; a log
    configured below wherever the platform starts, or one at the manager's
    default path while the configuration names another, is said by check
    and refuses encrypt, nothing changed, taken by neither -- with advice
    that never writes over the configured log, given only from what the
    command can see of it; the default path under another spelling is the
    same file; a relative log is never resolved from where the command
    runs, so never taken for the default by a path that leads there from
    there.
  * SE10 -- EVERY DATA DIRECTORY OF THE PACKAGE IS A PLACE: read from the
    sources, every binding of a directory of databases (DATA_DIR, DB_DIR,
    STORE_DIR, with or without a leading _ or DEFAULT_) made in a module
    or a class -- in every branch, by ``/=``, unpacked, imported from the
    package, or set on an object -- is a place of the command or its store
    folder, and every ChromaDB directory and RAG_HOME lies below a place;
    one an environment variable moves counts only when the command follows
    that variable; each place is bound, and each store folder joined to a
    data directory, by some module; a binding the census cannot read, or
    one below whatever directory the platform starts in, is named, never
    passed over. Not read: a binding made inside a function. The plugins'
    folders aside: each plugin keeps its own stores, in plaintext. The
    census looks at no directory on disk, and walks none named data.
  * SE11 -- WHAT STAYS IN PLAINTEXT IS SAID, AND WHAT WAS NOT CHECKED: every
    file below the places, and below RAG_HOME when it is set, that begins
    as a SQLite file no cipher wrote and is no ``*.db`` -- ChromaDB's
    vector store, a database under another name, one whose name the locale
    cannot write, said escaped -- is named by ``check`` and by ``encrypt``,
    in one order, and never encrypted or moved; a symbolic link is never
    followed and an entry that cannot be read is never passed over: both
    are said, not checked; a pipe is never read; the folders of plaintext
    copies beside the stores are passed over, a folder of that name
    elsewhere is not; what an interrupted run left is said by ``check`` as
    such; below RAG_HOME a ``*.db`` is named too; a relative RAG_HOME is
    said, never walked; one above a place walks the place once, as a place;
    one that cannot be looked at is said; one reached through a link inside
    a place is walked, the link never said unchecked.
  * SE12 -- GIT NEVER TAKES WHAT A RUN LEFT: each leftover of a run killed
    outright, as the command names it -- a second name for a store's
    plaintext, a copy whose store is missing, a key file's temp --, the
    databases of the folders of plaintext copies and their sidecars, and
    the key file are ignored by git where the platform keeps them.
  * SE13 -- NO KEY, NO DATA DIRECTORY FROM THE SHELL: the suites'
    configuration drops OPTI_ENCRYPTION_KEY, OPTI_KEYFILE_PASSPHRASE and
    RAG_HOME from the environment when it is imported, before any suite or
    child process -- every variable the command follows to a directory
    among them, so a contract never walks a real one.
  * SE14 -- THE SERVER SAYS WHICH KEY: setup answers ``key: "existing"``
    when a key here loads -- a key file the command wrote, no key made,
    the key file as it was, a passphrase sent not used; OPTI_ENCRYPTION_KEY,
    no key file written -- and ``"made"`` when it makes one; ``wrapped``
    says whether the key file it made is wrapped, under the passphrase
    sent or under the server's OPTI_KEYFILE_PASSPHRASE; ``plaintext_stores``
    counts the stores still kept in plaintext.
  * SE15 -- A FOLDER OF STORES THAT CANNOT BE READ: a place, a store folder
    or the sandbox's folder that is there and cannot be read or searched is
    said by check, never a trace, and refuses encrypt, nothing changed; a
    store folder that is a symbolic link is followed as its stores are
    taken through it, never said unchecked, what stays in plaintext and any
    other *.db below it named, and RAG_HOME at its real target walks it no
    second time; one linked to another folder of its own place is one
    store folder, its plaintext copies never said as other databases.
  * SE16 -- A KEY VARIABLE THAT IS NO KEY: an OPTI_ENCRYPTION_KEY that is
    set and does not decode to a key refuses encrypt -- even when a key file
    loads -- and is said by check, no key file made, check reading the
    stores under the key file the platform passes over to, or saying that
    key file does not load here either; the server's setup answers 409
    naming it, no key made, nothing saved.

Local-only (the public distribution ships no tests). The real
``db_encryption.py`` and ``encryption.py`` -- and ``api/routes_security.py``
for SE6 and SE14, a byte-exact copy of ``sandbox_manager.py`` for SE9 --
are loaded through the shared isolation window; SQLCipher is the
installed ``sqlcipher3``; the two places, the key file, the process table
and security.yaml are the test's own, under its temporary directory.
OPTI_ENCRYPTION_KEY, OPTI_KEYFILE_PASSPHRASE and RAG_HOME are cleared unless a
clause sets them.
"""

import ast
import asyncio
import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_census import modules as package_modules  # noqa: E402
from _isolation import isolate, source  # noqa: E402

_PLAIN = b"SQLite format 3\x00"
_SIDES = ("-wal", "-shm", "-journal")
_REPO = Path(__file__).resolve().parent.parent
# A run of the command in a process of its own, its places, process table and key file the test's, whose os is the
# real one but for a signal it sends itself before or after its n-th step (0: never) -- a rename, link, unlink or
# sync, a copy, a read of the processes, the key's write --, each named, and said at the end; a second signal after
# the same step, or after the step named by "@n"; a file opened first, which takes number 2 when standard error is
# closed. It imports the package from the tree under test and says so: never another checkout.
_DRIVER = """
import os, signal, sys
from pathlib import Path
repo, root, step, how, when, second, ignored, taken = sys.argv[1:9]
root, step = Path(root), int(step)
second, _, at = second.partition("@")
at = int(at) if at else step
held = open(taken, "wb") if taken != "-" else None
if held is not None:
    print("taken", held.fileno(), flush=True)
sys.path.insert(0, repo)
for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME"):
    os.environ.pop(name, None)
from opti_oignon import db_encryption as dbe, encryption as enc
assert dbe.__file__.startswith(repo) and enc.__file__.startswith(repo), (dbe.__file__, enc.__file__)
dbe._STORE_PLACES = (root / "data", root / "package" / "data")
dbe._LONE_STORES = []
dbe._PROC = root / "proc"
enc._DEFAULT_KEYFILE = root / "keys" / ".keyfile"
if ignored != "-":
    signal.signal(getattr(signal, ignored), signal.SIG_IGN)
real, calls = os, []


def _step(name, call, *args, **kwargs):
    calls.append(name)
    index = len(calls)
    if index == step and when == "before":
        real.kill(real.getpid(), getattr(signal, how))
    try:
        return call(*args, **kwargs)
    finally:
        if index == step and when == "after":
            real.kill(real.getpid(), getattr(signal, how))
        if index == at and second != "-":
            real.kill(real.getpid(), getattr(signal, second))


def _names(*paths):
    return ">".join(Path(path).name for path in paths)


class Dying:
    def __getattr__(self, name):
        return getattr(real, name)

    def link(self, src, dst):
        return _step(f"link:{_names(src, dst)}", real.link, src, dst)

    def replace(self, src, dst):
        return _step(f"replace:{_names(src, dst)}", real.replace, src, dst)

    def unlink(self, path):
        return _step(f"unlink:{_names(path)}", real.unlink, path)

    def fsync(self, fd):
        return _step("fsync", real.fsync, fd)


def _counted(name, call):
    return lambda *args, **kwargs: _step(name, call, *args, **kwargs)


dbe.os = Dying()
dbe._encrypted_copy = _counted("copy", dbe._encrypted_copy)
dbe.store_holders = _counted("holders", dbe.store_holders)
dbe._write_master = _counted("key", dbe._write_master)
code = dbe.main(["encrypt"])
print("steps", ",".join(calls), flush=True)
sys.exit(code)
"""


def _window(tmp_path, monkeypatch, *, routes=False):
    """The real modules in a window, the places, the key file, the process table and security.yaml redirected."""
    pytest.importorskip("sqlcipher3")
    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME"):
        monkeypatch.delenv(name, raising=False)
    targets = {"opti_oignon.encryption": source("encryption.py"),
               "opti_oignon.db_encryption": source("db_encryption.py")}
    if routes:
        targets["opti_oignon.api.routes_security"] = source("api", "routes_security.py")
    row = sqlite3.Row
    loaded, restore = isolate(targets=targets, blocked=("opti_oignon.secure_bytes",),
                              packages=("opti_oignon.api",) if routes else ())

    def closed():
        try:
            restore()
        finally:
            sqlite3.Row = row

    try:
        enc, dbe = loaded["opti_oignon.encryption"], loaded["opti_oignon.db_encryption"]
        places = (tmp_path / "data", tmp_path / "package" / "data")
        for place in places:
            place.mkdir(parents=True)
        keydir, proc = tmp_path / "keys", tmp_path / "proc"
        keydir.mkdir()
        proc.mkdir()
        enc._DEFAULT_KEYFILE = keydir / ".keyfile"
        dbe._STORE_PLACES = places
        dbe._LONE_STORES = []
        dbe._PROC = proc
        world = SimpleNamespace(enc=enc, dbe=dbe, places=places, keydir=keydir, keyfile=keydir / ".keyfile", proc=proc,
                                restore=closed)
        if routes:
            config = tmp_path / "security.yaml"
            config.write_text("# security settings as the test left them\n"
                              + yaml.safe_dump({"encryption": {"enabled": False}}), encoding="utf-8")
            enc._load_encryption_config = lambda: (
                (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("encryption", {}))
            world.routes = loaded["opti_oignon.api.routes_security"]
            world.routes._SECURITY_YAML_PATH = str(config)
            world.config = config
    except BaseException:
        closed()
        raise
    return world


def _store(path, rows, *, version=0, application=0, vacuum=0):
    """A plaintext store: a table and its index, a full-text table, rows of words, a user_version, an application_id,
    an auto_vacuum."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(f"PRAGMA auto_vacuum = {vacuum}")
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, words TEXT NOT NULL)")
        conn.execute("CREATE INDEX notes_words ON notes(words)")
        conn.execute("CREATE VIRTUAL TABLE search USING fts5(words)")
        conn.executemany("INSERT INTO notes (words) VALUES (?)", [(f"note {i} of {path.stem}",) for i in range(rows)])
        conn.executemany("INSERT INTO search (words) VALUES (?)", [(f"words {i}",) for i in range(rows)])
        conn.execute(f"PRAGMA user_version = {version}")
        conn.execute(f"PRAGMA application_id = {application}")
        conn.commit()
    finally:
        conn.close()


def _pending(path, rows):
    """Rows committed to ``path``'s write-ahead log by a process that ended without folding them into the store."""
    code = ("import os, sqlite3, sys\n"
            "c = sqlite3.connect(sys.argv[1])\n"
            "c.execute('PRAGMA journal_mode=WAL')\n"
            "c.execute('PRAGMA wal_autocheckpoint=0')\n"
            "c.executemany('INSERT INTO notes (words) VALUES (?)', [('pending %d' % i,) for i in range(int(sys.argv[2]))])\n"
            "c.commit()\n"
            "os._exit(0)\n")
    subprocess.run([sys.executable, "-c", code, str(path), str(rows)], check=True, timeout=60)


def _held(conn):
    """What a store holds, read through ``conn``: every note in order, the full-text matches, user_version and
    application_id."""
    return ([row[0] for row in conn.execute("SELECT words FROM notes ORDER BY id")],
            conn.execute("SELECT count(*) FROM search WHERE search MATCH 'words'").fetchone()[0],
            *(conn.execute(f"PRAGMA {name}").fetchone()[0] for name in ("user_version", "application_id",
                                                                         "auto_vacuum")))


def _held_plain(path, look):
    """What a plaintext store holds, its log included, read from a copy: the store and its log are left as they are."""
    look.mkdir(exist_ok=True)
    for side in ("",) + _SIDES:
        if Path(str(path) + side).exists():
            shutil.copy2(str(path) + side, look / (path.name + side))
    conn = sqlite3.connect(str(look / path.name))
    try:
        return _held(conn)
    finally:
        conn.close()
        shutil.rmtree(look)


def _tree(*roots):
    """Every entry under ``roots``, by path, with the md5 of a file's bytes, "dir" for a folder, or its link."""
    seen = {}
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if path.is_symlink():
                seen[str(path)] = "link " + os.readlink(path)
            elif path.is_dir():
                seen[str(path)] = "dir"
            else:
                seen[str(path)] = hashlib.md5(path.read_bytes()).hexdigest()
    return seen


def _corrupt(path):
    """Break one page of ``path`` past the header: the file still begins as a plaintext store."""
    with open(path, "r+b") as handle:
        handle.seek(2 * 4096)
        handle.write(b"\xff" * 64)


def _process(proc, pid, argv, held=()):
    """A process in the test's own process table: its command line and the files it holds open."""
    entry = proc / str(pid)
    entry.mkdir()
    if argv is not None:
        (entry / "cmdline").write_bytes(b"\0".join(part.encode() for part in argv) + b"\0")
    if held is not None:
        (entry / "fd").mkdir()
        for n, target in enumerate(held):
            os.symlink(target, entry / "fd" / str(n))


# ---------------------------------------------------------------------------
# SE1 -- every store, both places
# ---------------------------------------------------------------------------
def test_se1_every_store_of_both_places_is_encrypted_and_reads_as_it_did(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        data, package = world.places
        _store(data / "auth.db", 3, version=2)
        _store(package / "conversations.db", 40, version=7, application=0x4F4F)
        _store(package / "notes.db", 12, vacuum=1)
        os.chmod(package / "notes.db", 0o600)
        _pending(package / "conversations.db", 5)
        assert Path(str(package / "conversations.db") + "-wal").stat().st_size > 0, "control: rows still in the log"
        (package / "notes.db-journal").write_bytes(b"")
        (package / "empty.db").write_bytes(b"")
        (package / "notes.txt").write_text("not a store", encoding="utf-8")
        for folder in ("rag", "plugin_data/scratchpad", "projects/p1"):
            (package / folder).mkdir(parents=True)
        _store(package / "rag" / "rag_documents.db", 6)
        _store(package / "plugin_data" / "scratchpad" / "scratchpad.db", 2)
        _store(package / "projects" / "p1" / "user.db", 2)
        sandbox = tmp_path / "sandboxes"
        sandbox.mkdir()
        _store(sandbox / "sandbox_audit.db", 4)
        world.dbe._LONE_STORES = [sandbox / "sandbox_audit.db"]
        untouched = {path: path.read_bytes() for path in (package / "projects" / "p1" / "user.db",
                                                          package / "plugin_data" / "scratchpad" / "scratchpad.db")}
        stores = [data / "auth.db", package / "conversations.db", package / "notes.db",
                  package / "rag" / "rag_documents.db", sandbox / "sandbox_audit.db"]
        before = {path.name: _held_plain(path, tmp_path / "look") for path in stores}
        assert len(before["conversations.db"][0]) == 45, "control: the log's rows are the store's"
        assert world.dbe.plaintext_stores() == stores, "control: five stores in plaintext"
        done = world.dbe.encrypt_all_stores()
        names = ["auth.db", "conversations.db", "notes.db", "rag_documents.db", "sandbox_audit.db"]
        assert (done["encrypted"], done["already"], done["key"]) == (names, [], "made"), done
        assert done["left"] == [str(package / "plugin_data" / "scratchpad" / "scratchpad.db"),
                                str(package / "projects" / "p1" / "user.db")], "the plugin's and the user's named"
        assert all(path.read_bytes() == kept for path, kept in untouched.items()), "and left as they are"
        assert (done["key_path"], done["places"]) == (str(world.keyfile), [str(data), str(package)]), done
        assert stat.S_IMODE((package / "notes.db").stat().st_mode) == 0o600, "the store's mode kept"
        for path in stores:
            assert path.read_bytes()[:16] != _PLAIN, ("encrypted at rest", path.name)
            conn = world.dbe.get_encrypted_connection(str(path))
            try:
                assert _held(conn) == before[path.name], ("the platform's own connection reads it as it was", path.name)
            finally:
                conn.close()
            assert not [side for side in _SIDES if Path(str(path) + side).exists()], ("nothing left beside it", path.name)
        assert world.dbe.plaintext_stores() == [], "none left in plaintext"
        folders = [Path(folder) for folder in done["plaintext_copies"]]
        assert [folder.parent for folder in folders] == [data, package, package / "rag", sandbox], folders
        for folder in folders:
            assert stat.S_IMODE(folder.stat().st_mode) == 0o700, folder
        for path in stores:
            kept = next(folder for folder in folders if folder.parent == path.parent) / path.name
            assert kept.read_bytes()[:16] == _PLAIN and _held_plain(kept, tmp_path / "look") == before[path.name], (
                "the plaintext store kept as it was", path.name)
        assert (folders[1] / "notes.db-journal").exists(), "the file SQLite kept beside it went with it"
        assert (package / "empty.db").read_bytes() == b"" and (package / "notes.txt").exists(), "the rest left alone"
        assert stat.S_IMODE(world.keyfile.stat().st_mode) == 0o600, "the key file made is 0600"
        snapshot = _tree(*world.places, world.keydir)
        again = world.dbe.encrypt_all_stores()
        assert (again["encrypted"], again["already"], again["plaintext_copies"], again["key"], again["left"]) == (
            [], names, [], "existing", done["left"]), "the plaintext copies are never taken again"
        assert sorted(again["earlier_copies"]) == sorted(done["plaintext_copies"]), (
            "the folders the first run kept, said again", again["earlier_copies"])
        assert _tree(*world.places, world.keydir) == snapshot, "a second run moves nothing"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SE2 -- all or none, before any move
# ---------------------------------------------------------------------------
def test_se2_a_store_or_a_copy_that_fails_leaves_every_store_as_it_was_and_no_key_made(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        data, package = world.places
        _store(data / "auth.db", 3)
        _store(package / "conversations.db", 300)
        _store(package / "notes.db", 300)
        _corrupt(package / "notes.db")
        snapshot = _tree(*world.places, world.keydir)
        with pytest.raises(world.dbe.EncryptionRefused, match="notes.db does not pass its own integrity check"):
            world.dbe.encrypt_all_stores()
        assert _tree(*world.places, world.keydir) == snapshot, "every store as it was, no copy, no folder, no key file"
        # A copy that does not hold what its store holds, under a key that was there before.
        _store(package / "fresh.db", 2)
        shutil.copy2(tmp_path / "data" / "auth.db", tmp_path / "auth.copy")
        (package / "notes.db").unlink()
        world.enc.save_keyfile(world.enc.generate_key(), kdf_name="random", allow_unprotected=True)
        snapshot = _tree(*world.places, world.keydir)
        shape, calls = world.dbe._shape, []

        def drifting(conn):
            calls.append(1)
            got = shape(conn)
            return got if len(calls) != 4 else got[:2] + (got[2] + 1,) + got[3:]

        world.dbe._shape = drifting
        with pytest.raises(world.dbe.EncryptionRefused, match="the encrypted copy of conversations.db does not hold"):
            world.dbe.encrypt_all_stores()
        assert len(calls) == 4, "control: the copy of the second store was the one read otherwise"
        assert _tree(*world.places, world.keydir) == snapshot, "every store as it was, the key that was there kept"
        world.dbe._shape = shape
        # A copy whose rows are as many but not the same: its words read back otherwise.
        keyed, opened = world.dbe._keyed, []

        def tampering(path, key):
            conn = keyed(path, key)
            opened.append(str(path))
            if str(path).endswith("conversations.db.encrypting") and opened.count(str(path)) == 2:
                conn.execute("UPDATE notes SET words = 'changed' WHERE id = 1")
                conn.commit()
            return conn

        world.dbe._keyed = tampering
        try:
            with pytest.raises(world.dbe.EncryptionRefused, match="the encrypted copy of conversations.db does not hold"):
                world.dbe.encrypt_all_stores()
        finally:
            world.dbe._keyed = keyed
        assert _tree(*world.places, world.keydir) == snapshot, "the same rows counted, other words: refused"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SE3 -- all or none, while moving
# ---------------------------------------------------------------------------
def test_se3_a_move_that_fails_puts_back_every_move_and_the_run_is_refused(tmp_path, monkeypatch, capsys):
    world = _window(tmp_path, monkeypatch)
    try:
        data, package = world.places
        _store(data / "auth.db", 3)
        _store(package / "conversations.db", 20)
        snapshot = _tree(*world.places, world.keydir)
        real, moved, written = world.dbe.os, [], []

        class Failing:
            def __getattr__(self, name):
                return getattr(real, name)

            def replace(self, src, dst):
                moved.append((Path(src).name, Path(dst).name))
                written.append(world.keyfile.exists())
                if len(moved) == 2:
                    raise OSError(28, "No space left on device")
                return real.replace(src, dst)

        world.dbe.os = Failing()
        try:
            with pytest.raises(world.dbe.EncryptionRefused, match="every step was put back: nothing was changed"):
                world.dbe.encrypt_all_stores()
        finally:
            world.dbe.os = real
        assert moved[:2] == [("auth.db.encrypting", "auth.db"), ("conversations.db.encrypting", "conversations.db")], (
            "control: the second store's switch failed", moved)
        assert written[:2] == [True, True], "control: the key file was written before the switch"
        assert ("auth.db.restoring", "auth.db") in moved, "the first store put back by a rename of its own"
        assert _tree(*world.places, world.keydir) == snapshot, "every store as it was, no copy, no folder, no key file"
        # Copies that cannot be removed once every store is put back are said, to delete; the run is a refusal, the
        # key file it wrote removed.

        class Sticky(Failing):
            def replace(self, src, dst):
                if str(src).endswith(".encrypting"):
                    raise OSError(28, "No space left on device")
                return real.replace(src, dst)

            def unlink(self, path):
                if str(path).endswith(".encrypting"):
                    raise PermissionError(13, "Permission denied")
                return real.unlink(path)

        world.dbe.os = Sticky()
        try:
            with pytest.raises(world.dbe.EncryptionRefused, match="every step was put back: nothing was changed") as left:
                world.dbe.encrypt_all_stores()
        finally:
            world.dbe.os = real
        for copy in (data / "auth.db.encrypting", package / "conversations.db.encrypting"):
            assert f"{copy} stays (PermissionError): delete it" in str(left.value), left.value
            copy.unlink()
        assert _tree(*world.places, world.keydir) == snapshot, "every store as it was, no folder, no key file"
        # A directory that cannot be synced once every store is switched undoes nothing: it is said.
        sync, synced = world.dbe._fsync_dir, []

        def unsyncable(path):
            synced.append(path)
            if len(synced) == 1:
                raise OSError(22, "Invalid argument")
            return sync(path)

        capsys.readouterr()
        world.dbe._fsync_dir = unsyncable
        try:
            assert world.dbe.main(["encrypt"]) == 0
        finally:
            world.dbe._fsync_dir = sync
        said = capsys.readouterr().out
        assert "2 store(s) encrypted, 0 already" in said and said.count(" could not be synced to the disk") == 1 and (
            f"{synced[0]} could not be synced to the disk: the switch stands; run sync before the next power cut"
            in said), said
        assert world.dbe.plaintext_stores() == [], "the switch stands"
        for path in (data / "auth.db", package / "conversations.db"):
            path.unlink()
        for folder in [*data.glob("plaintext-*"), *package.glob("plaintext-*")]:
            shutil.rmtree(folder)
        world.keyfile.unlink()
        _store(data / "auth.db", 3)
        _store(package / "conversations.db", 20)
        # A put-back that fails: the store stays encrypted, and its plaintext, its folder and the key file stay too.
        before = _held_plain(data / "auth.db", tmp_path / "look")
        moved.clear()

        class Stuck(Failing):
            def link(self, src, dst):
                if Path(dst).name == "auth.db.restoring":
                    raise OSError(28, "No space left on device")
                return real.link(src, dst)

        capsys.readouterr()
        world.dbe.os = Stuck()
        try:
            assert world.dbe.main(["encrypt"]) == 3, "a run stopped part way ends as neither a success nor a refusal"
        finally:
            world.dbe.os = real
        said = capsys.readouterr().err
        assert said.startswith("stopped part way: the stores could not all be put back where they were: "), said
        assert "auth.db is encrypted, its plaintext is" in said and "the key file stay" in said, said
        kept = Path(said.split("its plaintext is ")[1].split(" (")[0])
        assert kept.exists() and _held_plain(kept, tmp_path / "look") == before, "the plaintext kept whole"
        assert world.keyfile.exists(), "the key file the store is encrypted under kept"
        conn = world.dbe.get_encrypted_connection(str(data / "auth.db"))
        try:
            assert _held(conn) == before, "the store, encrypted, whole"
        finally:
            conn.close()
        # A run under a key it did not write, whose switch fails, leaves that key file as it was.
        key_bytes = world.keyfile.read_bytes()
        plain = _held_plain(package / "conversations.db", tmp_path / "look")
        assert world.dbe.plaintext_stores() == [package / "conversations.db"], "control: one store left in plaintext"

        class FirstSwitchFails(Failing):
            def replace(self, src, dst):
                if str(src).endswith(".encrypting"):
                    raise OSError(28, "No space left on device")
                return real.replace(src, dst)

        world.dbe.os = FirstSwitchFails()
        try:
            with pytest.raises(world.dbe.EncryptionRefused, match="every step was put back: nothing was changed"):
                world.dbe.encrypt_all_stores()
        finally:
            world.dbe.os = real
        assert world.keyfile.read_bytes() == key_bytes, "the key file this run did not write, as it was"
        assert _held_plain(package / "conversations.db", tmp_path / "look") == plain, "the plaintext store as it was"
        conn = world.dbe.get_encrypted_connection(str(data / "auth.db"))
        try:
            assert _held(conn) == before, "the store encrypted under that key still opens"
        finally:
            conn.close()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SE4 -- never under a running platform
# ---------------------------------------------------------------------------
def test_se4_a_process_that_holds_a_store_or_runs_the_platform_refuses_the_run(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        data, package = world.places
        _store(data / "auth.db", 3)
        _store(package / "conversations.db", 20)
        stores = world.dbe.store_files()
        wal = os.path.realpath(package / "conversations.db") + "-wal"
        proc = world.proc
        _process(proc, os.getpid(), ["python3", "-m", "uvicorn", "opti_oignon.api.app:app"],
                 [os.path.realpath(data / "auth.db")])
        _process(proc, 104, None, None)
        _process(proc, 105, ["bash", "/home/someone/Git_Opti-Oignon/backup.sh"], [])
        _process(proc, os.getppid(), ["bash", "-c", "python3 -m opti_oignon.db_encryption encrypt"], [])
        (proc / "self").mkdir()
        assert world.dbe.store_holders(stores) == {}, (
            "this process, its parent, one that cannot be read, a folder's name: none")
        snapshot = _tree(*world.places, world.keydir)
        for pid, argv, held, why in (
                (101, ["python3", "worker.py"], [wal], "holds conversations.db-wal open"),
                (102, ["python3", "-m", "uvicorn", "opti_oignon.api.app:app"], [], "runs the platform"),
                (103, ["/usr/bin/python3", "/home/someone/.local/bin/oo", "chat"], [], "runs the platform (oo)"),
                (106, ["python3", "worker.py"], [os.path.realpath(data / "auth.db") + " (deleted)"],
                 "holds auth.db (deleted) open")):
            _process(proc, pid, argv, held)
            assert world.dbe.store_holders(stores) == {pid: world.dbe.store_holders(stores)[pid]}, pid
            assert why in world.dbe.store_holders(stores)[pid], (pid, world.dbe.store_holders(stores))
            with pytest.raises(world.dbe.EncryptionRefused, match=f"stop every process of the platform first: process "
                                                                  f"{pid}"):
                world.dbe.encrypt_all_stores()
            assert _tree(*world.places, world.keydir) == snapshot, ("nothing changed, no key made", pid)
            shutil.rmtree(proc / str(pid))
        # A process that starts while the copies are made: refused before any move.
        holders, reads = world.dbe.store_holders, []

        def late(paths, *, proc=None):
            reads.append(1)
            return {} if len(reads) == 1 else {107: "runs the platform (oo)"}

        world.dbe.store_holders = late
        try:
            with pytest.raises(world.dbe.EncryptionRefused, match="a process of the platform started meanwhile"):
                world.dbe.encrypt_all_stores()
        finally:
            world.dbe.store_holders = holders
        assert len(reads) == 2 and _tree(*world.places, world.keydir) == snapshot, "no copy, no move, no key made"
        # The real process table names a child that holds a store.
        child = subprocess.Popen([sys.executable, "-c", "import sqlite3, sys, time\n"
                                  "c = sqlite3.connect(sys.argv[1]); c.execute('SELECT count(*) FROM notes').fetchone()\n"
                                  "print('ready', flush=True); time.sleep(60)\n", str(data / "auth.db")],
                                 stdout=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline().strip() == "ready"
            seen = world.dbe.store_holders(stores, proc="/proc")
            assert "holds auth.db open" in seen.get(child.pid, ""), (child.pid, seen.get(child.pid))
        finally:
            child.kill()
            child.wait(10)
        with pytest.raises(world.dbe.EncryptionRefused, match="shows no open files per process"):
            world.dbe.store_holders(stores, proc=tmp_path / "nowhere")
        # The parent counts when it runs the platform without naming the command, and for what it holds open.
        shutil.rmtree(proc / str(os.getppid()))
        _process(proc, os.getppid(), ["python3", "-m", "opti_oignon.cli", "chat"], [])
        assert world.dbe.store_holders(stores) == {os.getppid(): "runs the platform (opti_oignon.cli)"}
        shutil.rmtree(proc / str(os.getppid()))
        _process(proc, os.getppid(), ["bash"], [os.path.realpath(data / "auth.db")])
        assert world.dbe.store_holders(stores) == {os.getppid(): "holds auth.db open"}
        shutil.rmtree(proc / str(os.getppid()))
        # Another run under way.
        lock = os.open(data, os.O_RDONLY)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with pytest.raises(world.dbe.EncryptionRefused, match="another encryption of the stores is under way"):
                world.dbe.encrypt_all_stores()
        finally:
            os.close(lock)
        snapshot = _tree(*world.places, world.keydir)
        # What an interrupted run left: a plaintext half put back, a copy whose store is missing.
        for leftover, what in (
                (package / "conversations.db.restoring", "conversations.db's plaintext, left by a run killed while "
                 "putting conversations.db back, and in no plaintext-* folder beside it: move it into a new one, as "
                 "conversations.db)"),
                (package / "ghost.db.encrypting", "an encrypted copy of ghost.db, which is missing: put ghost.db back, "
                 "or delete the copy if it was removed on purpose)"),
                (world.keydir / "..keyfile.tmp-0a1b2c3d", "left by a write of the key file killed before it was in "
                 "place: nothing was encrypted under it; delete it)")):
            leftover.write_bytes(b"left by a run that died")
            with pytest.raises(world.dbe.EncryptionRefused, match=re.escape(f"an interrupted run left {leftover} "
                                                                            f"({what}")):
                world.dbe.encrypt_all_stores()
            leftover.unlink()
        assert _tree(*world.places, world.keydir) == snapshot, "nothing changed"
        # A second name for a file kept elsewhere is said so, to delete.
        folder = package / "plaintext-20261021T000000Z"
        folder.mkdir()
        (folder / "conversations.db").write_bytes(b"a plaintext kept")
        os.link(folder / "conversations.db", package / "conversations.db.restoring")
        world.keyfile.write_bytes(b"a key file")
        os.link(world.keyfile, world.keydir / "..keyfile.tmp-0a1b2c3d")
        with pytest.raises(world.dbe.EncryptionRefused) as refused:
            world.dbe.encrypt_all_stores()
        assert (f"{package / 'conversations.db.restoring'} (a second name for {folder / 'conversations.db'}, left by a "
                "run killed while putting conversations.db back: delete it)") in str(refused.value), refused.value
        assert (f"{world.keydir / '..keyfile.tmp-0a1b2c3d'} (a second name for the key file, left by a write killed "
                "once it was in place: delete it)") in str(refused.value), refused.value
        # A namesake is no second name: the file half put back is then the only one, and the temp holds another key.
        for path in (package / "conversations.db.restoring", world.keydir / "..keyfile.tmp-0a1b2c3d"):
            path.unlink()
        (package / "conversations.db.restoring").write_bytes(b"a plaintext kept")
        (world.keydir / "..keyfile.tmp-0a1b2c3d").write_bytes(b"another key")
        with pytest.raises(world.dbe.EncryptionRefused) as refused:
            world.dbe.encrypt_all_stores()
        assert ("conversations.db.restoring (conversations.db's plaintext, left by a run killed while putting "
                "conversations.db back, and in no plaintext-* folder beside it: move it into a new one, as "
                "conversations.db)") in str(refused.value), refused.value
        assert ("..keyfile.tmp-0a1b2c3d (left by a write of the key file killed before it was in place: nothing was "
                "encrypted under it; delete it)") in str(refused.value), refused.value
        # Never through a symbolic link: a link in a folder to the file half put back is no second name for it, and a
        # key file that is a link to a temp keeps that temp.
        (folder / "conversations.db").unlink()
        (folder / "conversations.db").symlink_to(package / "conversations.db.restoring")
        world.keyfile.unlink()
        world.keyfile.symlink_to(world.keydir / "..keyfile.tmp-0a1b2c3d")
        with pytest.raises(world.dbe.EncryptionRefused) as refused:
            world.dbe.encrypt_all_stores()
        assert "move it into a new one, as conversations.db)" in str(refused.value), refused.value
        assert ("..keyfile.tmp-0a1b2c3d (a key file's temp, and the key file is a symbolic link: if it leads here, this "
                "holds the key in use; keep it)") in str(refused.value), refused.value
        for path in (package / "conversations.db.restoring", world.keydir / "..keyfile.tmp-0a1b2c3d", world.keyfile):
            path.unlink()
        shutil.rmtree(folder)
        assert _tree(*world.places, world.keydir) == snapshot, "nothing changed"
        # A copy of a store that is there is made again.
        (package / "conversations.db.encrypting").write_bytes(b"a copy a run left")
        done = world.dbe.encrypt_all_stores()
        assert done["encrypted"] == ["auth.db", "conversations.db"] and not (
            package / "conversations.db.encrypting").exists(), done
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SE5 -- the key
# ---------------------------------------------------------------------------
def test_se5_a_key_that_loads_is_taken_one_that_does_not_refuses_and_none_is_ever_replaced(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        data, package = world.places
        _store(data / "auth.db", 3)
        # A key file that does not load: refused naming the passphrase, nothing changed.
        world.keyfile.write_bytes(b"not a key file\n")
        snapshot = _tree(*world.places, world.keydir)
        with pytest.raises(world.dbe.EncryptionRefused, match="OPTI_KEYFILE_PASSPHRASE"):
            world.dbe.encrypt_all_stores()
        assert _tree(*world.places, world.keydir) == snapshot
        # A key file that loads is taken, never replaced.
        world.keyfile.unlink()
        world.enc.save_keyfile(world.enc.generate_key(), kdf_name="random", allow_unprotected=True)
        keys = _tree(world.keydir)
        done = world.dbe.encrypt_all_stores()
        assert (done["encrypted"], done["key"]) == (["auth.db"], "existing") and _tree(world.keydir) == keys, done
        # A store already encrypted under it is left; a plaintext one beside it is encrypted.
        _store(package / "notes.db", 4)
        done = world.dbe.encrypt_all_stores()
        assert (done["encrypted"], done["already"]) == (["notes.db"], ["auth.db"]), done
        # A store the key does not open refuses the run.
        (package / "foreign.db").write_bytes(os.urandom(8192))
        _store(package / "later.db", 2)
        snapshot = _tree(*world.places, world.keydir)
        with pytest.raises(world.dbe.EncryptionRefused, match="foreign.db do not open under the key that loads here"):
            world.dbe.encrypt_all_stores()
        assert _tree(*world.places, world.keydir) == snapshot
        # With no key here at all, an encrypted store refuses the run and no key is made.
        world.keyfile.unlink()
        snapshot = _tree(*world.places, world.keydir)
        with pytest.raises(world.dbe.EncryptionRefused, match="encrypted under a key that does not load here"):
            world.dbe.encrypt_all_stores()
        assert _tree(*world.places, world.keydir) == snapshot, "no key made"
        # A store that is a symbolic link refuses the run.
        for path in package.glob("*.db"):
            path.unlink()
        for path in data.glob("*.db"):
            path.unlink()
        _store(tmp_path / "elsewhere.db", 2)
        os.symlink(tmp_path / "elsewhere.db", package / "linked.db")
        snapshot = _tree(*world.places, world.keydir)
        with pytest.raises(world.dbe.EncryptionRefused, match="linked.db are symbolic links"):
            world.dbe.encrypt_all_stores()
        assert _tree(*world.places, world.keydir) == snapshot
        (package / "linked.db").unlink()
        # A new key is held in memory while the copies are made, and written before the switch.
        _store(data / "auth.db", 3)
        copy, seen = world.dbe._encrypted_copy, []
        world.dbe._encrypted_copy = lambda *args: seen.append(world.keyfile.exists()) or copy(*args)
        try:
            done = world.dbe.encrypt_all_stores()
        finally:
            world.dbe._encrypted_copy = copy
        assert seen == [False] and done["key"] == "made" and world.keyfile.exists(), (seen, done)
        world.keyfile.unlink()
        for path in data.glob("*.db"):
            path.unlink()
        for path in data.glob("plaintext-*"):
            shutil.rmtree(path)
        # A key file that does not load once written goes, and the run is refused.
        _store(data / "auth.db", 3)
        derive = world.dbe._get_db_encryption_key
        world.dbe._get_db_encryption_key = lambda: None
        snapshot = _tree(*world.places, world.keydir)
        try:
            with pytest.raises(world.dbe.EncryptionRefused, match="the key does not load once written"):
                world.dbe.encrypt_all_stores()
        finally:
            world.dbe._get_db_encryption_key = derive
        assert _tree(*world.places, world.keydir) == snapshot, "the key file written goes, nothing else moved"
        (data / "auth.db").unlink()
        # OPTI_ENCRYPTION_KEY is taken, and no key file is made.
        _store(data / "auth.db", 3)
        monkeypatch.setenv("OPTI_ENCRYPTION_KEY", base64.urlsafe_b64encode(os.urandom(32)).decode())
        done = world.dbe.encrypt_all_stores()
        assert (done["encrypted"], done["key"], done["key_file"]) == (["auth.db"], "existing", "environment"), done
        assert not world.keyfile.exists(), "no key file made beside the environment's key"
        conn = world.dbe.get_encrypted_connection(str(data / "auth.db"))
        conn.close()
        monkeypatch.delenv("OPTI_ENCRYPTION_KEY")
        # With OPTI_KEYFILE_PASSPHRASE set, the key file made is wrapped under it.
        for path in data.glob("*.db"):
            path.unlink()
        _store(data / "auth.db", 3)
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", "a passphrase the key file is wrapped under")
        done = world.dbe.encrypt_all_stores()
        assert (done["key"], done["key_file"]) == ("made", "wrapped"), done
        envelope = json.loads(world.keyfile.read_text(encoding="utf-8"))
        assert envelope.get("version") == "envelope-v1", "wrapped, not the raw key"
        conn = world.dbe.get_encrypted_connection(str(data / "auth.db"))
        conn.close()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SE6 -- the server refuses what would break the stores
# ---------------------------------------------------------------------------
def test_se6_the_server_makes_no_key_and_moves_no_store_while_one_is_in_plaintext(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch, routes=True)
    try:
        data, package = world.places
        routes = world.routes
        _store(data / "auth.db", 3)
        _store(package / "conversations.db", 5)
        made = []
        generate = world.enc.generate_key
        world.enc.generate_key = lambda *args, **kwargs: made.append(1) or generate(*args, **kwargs)
        config, snapshot = world.config.read_bytes(), _tree(*world.places, world.keydir)
        with pytest.raises(routes.HTTPException) as refused:
            routes.setup_encryption(routes.EncryptionSetupRequest(mode="random"))
        assert refused.value.status_code == 409 and "2 database(s)" in refused.value.detail, refused.value.detail
        assert "python3 -m opti_oignon.db_encryption encrypt" in refused.value.detail
        assert made == [] and world.config.read_bytes() == config, "no key made, security.yaml untouched"
        assert _tree(*world.places, world.keydir) == snapshot
        with pytest.raises(routes.HTTPException) as moved:
            asyncio.run(routes.migrate_databases(routes.MigrateDBRequest()))
        assert moved.value.status_code == 409 and "python3 -m opti_oignon.db_encryption encrypt" in moved.value.detail
        assert _tree(*world.places, world.keydir) == snapshot, "no store moved"
        status = asyncio.run(routes.get_db_encryption_status())
        assert (status["total_databases"], status["unencrypted_databases"]) == (2, 2), status
        # With no store in plaintext, setup makes its key as before.
        for path in (data / "auth.db", package / "conversations.db"):
            path.unlink()
        answer = routes.setup_encryption(routes.EncryptionSetupRequest(mode="random"))
        assert answer["setup"] is True and world.keyfile.exists() and made == [1], answer
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SE7 -- the command
# ---------------------------------------------------------------------------
def test_se7_the_command_says_each_stores_state_and_encrypts_them_or_says_why_not(tmp_path, monkeypatch, capsys):
    world = _window(tmp_path, monkeypatch)
    try:
        data, package = world.places
        _store(data / "auth.db", 3)
        _store(package / "notes.db", 4)
        (package / "projects").mkdir()
        _store(package / "projects" / "user.db", 1)
        assert world.dbe.main(["check"]) == 0
        said = capsys.readouterr().out
        assert said.startswith(f"places: {data}, {package}\n"), said
        assert f"{data / 'auth.db'}: plaintext" in said and f"{package / 'notes.db'}: plaintext" in said, said
        assert f"{package / 'projects' / 'user.db'}: no store of the platform, left as it is" in said, said
        assert "2 store(s), 2 in plaintext; SQLCipher installed; a key does not load" in said, said
        _process(world.proc, 201, ["python3", "-m", "opti_oignon.cli", "chat"], [])
        assert world.dbe.main(["check"]) == 0
        assert "process 201 runs the platform" in capsys.readouterr().out
        assert world.dbe.main(["encrypt"]) == 1
        refused = capsys.readouterr()
        assert refused.err.startswith("refused: stop every process of the platform first: process 201"), refused.err
        assert refused.out == "" and not world.keyfile.exists(), "nothing done, nothing said done"
        shutil.rmtree(world.proc / "201")
        assert world.dbe.main(["encrypt"]) == 0
        said = capsys.readouterr().out
        assert "2 store(s) encrypted, 0 already; key made (file)" in said, said
        assert (f"key file: {world.keyfile} -- every store opens only with it: keep a copy of it somewhere safe, off "
                "this machine and out of any repository, before deleting the plaintext copies") in said, said
        assert said.count("plaintext copies kept in ") == 2 and "delete it once the platform reads its stores again" in said
        assert f"{package / 'projects' / 'user.db'}: no store of the platform, left as it is" in said, said
        assert world.dbe.main(["check"]) == 0
        said = capsys.readouterr().out
        assert f"{data / 'auth.db'}: encrypted" in said and "2 store(s), 0 in plaintext" in said and "a key loads" in said
        kept = [*data.glob("plaintext-*"), *package.glob("plaintext-*")]
        assert len(kept) == 2 and all(f"plaintext copies kept in {folder}: delete it once the platform reads its "
                                      "stores again and the key is kept safe elsewhere" in said for folder in kept), said
        assert world.dbe.main(["encrypt"]) == 0
        said = capsys.readouterr().out
        assert "0 store(s) encrypted, 2 already" in said and said.count("plaintext copies kept in ") == 0, said
        assert all(f"plaintext copies an earlier run kept in {folder}: delete it once the platform reads its stores "
                   "again" in said for folder in kept), said
        world.keyfile.write_bytes(b"not a key file\n")
        assert world.dbe.main(["check"]) == 0, "a key file that does not load is said, never raised"
        said = capsys.readouterr().out
        assert "key: a key file exists and does not load here" in said and f"{data / 'auth.db'}: unreadable here" in said
    finally:
        world.restore()



# ---------------------------------------------------------------------------
# SE8 -- a run that dies
# ---------------------------------------------------------------------------
def _ground(world, root):
    """A fresh world under ``root``: two places with a store each, the process table and the key file there, the
    modules turned to it; what each store holds, by path."""
    data, package, keys, proc = root / "data", root / "package" / "data", root / "keys", root / "proc"
    for folder in (data, package, keys, proc):
        folder.mkdir(parents=True)
    _store(data / "auth.db", 3, version=2)
    _store(package / "conversations.db", 30, version=7)
    world.dbe._STORE_PLACES = (data, package)
    world.dbe._PROC = proc
    world.enc._DEFAULT_KEYFILE = keys / ".keyfile"
    stores = [data / "auth.db", package / "conversations.db"]
    return stores, {path: _held_plain(path, root / "look") for path in stores}


def _work(name):
    """Whether a step of a run is work, not putting back: a copy, a read of the processes, the key's write, a store's
    plaintext linked into its folder, an encrypted copy taking a store's name."""
    kind, _, names = name.partition(":")
    src, _, dst = names.partition(">")
    return (kind in ("copy", "holders", "key") or (kind == "link" and src == dst and src.endswith(".db"))
            or (kind == "replace" and src.endswith(".encrypting")))


def _safe_points(sequence):
    """The steps of a whole run that a safe point comes just before: each copy, the second read of the processes,
    the key's write, each store's switch (its plaintext linked into its folder)."""
    first = sequence.index("holders")
    return [i for i, name in enumerate(sequence)
            if (i != first and name in ("copy", "holders", "key")) or (name.startswith("link:") and _work(name))]


def _steps(ended):
    last = ended.stdout.strip().splitlines()[-1] if ended.stdout.strip() else ""
    assert last.startswith("steps "), (ended.returncode, ended.stdout, ended.stderr)
    return last.split()[1].split(",")


def _run(root, step, how, when="before", second="-", ignored="-", taken="-", closed=False):
    argv = [sys.executable, "-c", _DRIVER, str(_REPO), str(root), str(step), how, when, second, ignored, taken]
    if closed:
        argv = ["bash", "-c", 'exec "$@" 2>&-', "bash", *argv]
    return subprocess.run(argv, capture_output=True, text=True, timeout=120)


def test_se8_a_run_killed_at_any_step_leaves_every_store_whole_and_one_ended_puts_everything_back(tmp_path,
                                                                                                  monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        _ground(world, tmp_path / "dry")
        dry = _run(tmp_path / "dry", 0, "SIGKILL")
        assert dry.returncode == 0 and " noted: " not in dry.stderr, (dry.returncode, dry.stderr)
        sequence = _steps(dry)
        steps = len(sequence)
        kinds = [name.split(":")[0] for name in sequence]
        assert kinds[:11] == ["holders", "copy", "fsync", "copy", "fsync", "holders", "key", "link", "replace", "link",
                              "replace"] and set(kinds[11:]) == {"fsync"}, (
            "control: the processes read, two copies made and synced, read again, the key written, two stores "
            "switched, the directories synced", sequence)
        safe = _safe_points(sequence)
        assert [kinds[i] for i in safe] == ["copy", "copy", "holders", "key", "link", "link"], ("control", safe)
        for step in range(1, steps + 1):
            root = tmp_path / f"killed{step}"
            stores, before = _ground(world, root)
            died = _run(root, step, "SIGKILL")
            assert died.returncode == -signal.SIGKILL, (step, died.returncode, died.stderr)
            for path in stores:
                assert path.exists(), ("a store's name is never missing", step, path.name)
                if path.read_bytes()[:16] == _PLAIN:
                    assert _held_plain(path, root / "look") == before[path], ("whole, in plaintext", step, path.name)
                else:
                    assert world.enc._DEFAULT_KEYFILE.exists(), ("an encrypted store has its key file", step)
                    conn = world.dbe.get_encrypted_connection(str(path))
                    try:
                        assert _held(conn) == before[path], ("whole, encrypted", step, path.name)
                    finally:
                        conn.close()
            done = world.dbe.encrypt_all_stores()
            assert sorted(done["encrypted"] + done["already"]) == ["auth.db", "conversations.db"], (step, done)
            for path in stores:
                conn = world.dbe.get_encrypted_connection(str(path))
                try:
                    assert _held(conn) == before[path], ("the next run finished it", step, path.name)
                finally:
                    conn.close()
        # A signal is noted where it lands, and the run stops at its next safe point: everything is put back, or,
        # when it landed while the last store was switched, the run finishes. Never a store lost either way.
        outcomes = set()
        for how, when, second in (("SIGTERM", "before", "-"), ("SIGTERM", "after", "-"), ("SIGHUP", "after", "-"),
                                  ("SIGINT", "before", "-"), ("SIGTERM", "after", "SIGINT")):
            for step in range(1, steps + 1):
                root = tmp_path / f"{how}{when}{second}{step}"
                stores, before = _ground(world, root)
                snapshot = _tree(root / "data", root / "package", root / "keys")
                ended = _run(root, step, how, when, second)
                outcomes.add(ended.returncode)
                ran = _steps(ended)
                assert ended.stderr.count(" noted: the run stops at its next safe point") == 1, (
                    "said at once, once", how, when, step, ended.stderr)
                # The next safe point after the signal is noted -- before step ``stop + 1`` -- stops the run there.
                stop = next((i for i in safe if i >= step), None)
                if stop is not None:
                    assert ended.returncode == 1, (how, when, step, ended.returncode, ended.stderr)
                    assert ran[:stop] == sequence[:stop], ("the run went on to its next safe point", how, when, step,
                                                           ran)
                    assert not [name for name in ran[stop:] if _work(name)], (
                        "then only put everything back", how, when, step, ran)
                    said = [line for line in ended.stderr.splitlines() if line.startswith("refused: ")]
                    assert len(said) == 1 and "stopped by signal" in said[0] and "nothing was changed" in said[0], (
                        how, when, step, ended.stderr)
                    assert _tree(root / "data", root / "package", root / "keys") == snapshot, (
                        "everything put back", how, when, step)
                else:
                    assert ended.returncode == 0 and ran == sequence, (how, when, step, ended.returncode, ran,
                                                                       ended.stderr)
                    assert "came after the last safe point: the run finished" in ended.stderr, (
                        "a signal that let the run finish is said", how, when, step, ended.stderr)
                    for path in stores:
                        conn = world.dbe.get_encrypted_connection(str(path))
                        try:
                            assert _held(conn) == before[path], ("finished whole", how, when, step, path.name)
                        finally:
                            conn.close()
        assert outcomes == {0, 1}, ("control: signals that stopped the run and signals that let it finish", outcomes)
        # A signal the command was started with ignored stays ignored.
        root = tmp_path / "ignored"
        _ground(world, root)
        kept = _run(root, 1, "SIGHUP", "before", "-", "SIGHUP")
        assert kept.returncode == 0 and " noted: " not in kept.stderr, (
            "a hang-up ignored by nohup does not stop the run, and is not said", kept.stderr)
        # A second signal while the run puts everything back changes nothing it does.
        for stop in (safe[3], safe[5]):
            root = tmp_path / f"stopped{stop}"
            _ground(world, root)
            ended = _run(root, stop, "SIGTERM", "after")
            put_back = _steps(ended)
            assert ended.returncode == 1 and put_back[:stop] == sequence[:stop] and len(put_back) > stop, (
                "control: stopped at the key's write, and at the second store's switch, then put back", put_back)
            for at in range(stop + 1, len(put_back) + 1):
                root = tmp_path / f"stopped{stop}-{at}"
                _ground(world, root)
                snapshot = _tree(root / "data", root / "package", root / "keys")
                twice = _run(root, stop, "SIGTERM", "after", f"SIGINT@{at}")
                assert twice.returncode == 1 and _steps(twice) == put_back, (stop, at, twice.returncode, twice.stderr)
                assert twice.stderr.count(" noted: ") == 1, ("said once", stop, at, twice.stderr)
                assert _tree(root / "data", root / "package", root / "keys") == snapshot, ("put back", stop, at)
        # Standard error closed at the start: the notice never lands in the file that then takes its number.
        root = tmp_path / "closed"
        _ground(world, root)
        snapshot = _tree(root / "data", root / "package", root / "keys")
        taken = root / "taken.log"
        closed = _run(root, 2, "SIGTERM", "before", taken=str(taken), closed=True)
        assert "taken 2" in closed.stdout, ("control: the file took standard error's number", closed.stdout)
        assert closed.returncode == 1 and "refused: stopped by signal" in closed.stdout, (closed.returncode,
                                                                                         closed.stdout)
        assert taken.read_bytes() == b"", ("no notice in it", taken.read_bytes()[:120])
        assert _tree(root / "data", root / "package", root / "keys") == snapshot, "everything put back"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SE9 -- the sandbox's audit log, where its manager opens it
# ---------------------------------------------------------------------------
def _sandbox_window(tmp_path):
    """The real ``sandbox_manager.py``, copied byte for byte beside a configuration that keeps its import-time manager
    off (that one would make the configured folders), in the shared window; its keyed connection a plain one."""
    package = tmp_path / "sandbox_copy"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text("enabled: false\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = lambda path, *args, **kwargs: sqlite3.connect(path)
    loaded, restore = isolate(targets={"opti_oignon.sandbox_manager": copy}, seeded={"opti_oignon.db_utils": db_utils})
    sm = loaded["opti_oignon.sandbox_manager"]
    if sm.sandbox_manager is not None:
        restore()
        raise AssertionError("the import-time manager must stay off in this window")
    return sm, restore


def _sandbox_shapes(root):
    """Configurations of the sandbox, by name: the shipped one, the shapes its manager reads, one with a value that
    does not build (the manager then opens its defaults), one with a base relative to where the platform starts."""
    return {
        "shipped": source("config", "sandbox.yaml").read_text(encoding="utf-8"),
        "defaults": "",
        "a base": f"workspace_base: {root / 'base'}\n",
        "a base and a relative log": f"workspace_base: {root / 'base'}\naudit_db_path: audit/log.db\n",
        "an absolute log": f"audit_db_path: {root / 'elsewhere' / 'log.db'}\n",
        "turned off": f"enabled: false\nworkspace_base: {root / 'off'}\n",
        "a value that does not build": f"workspace_base: {root / 'unbuilt'}\nhost_share_roots: 5\n",
        "a relative base": "workspace_base: sandboxes\n",
    }


def test_se9_the_sandboxs_audit_log_is_taken_where_its_manager_opens_it(tmp_path, monkeypatch, capsys):
    files = {}
    for index, (name, text) in enumerate(_sandbox_shapes(tmp_path).items()):
        files[name] = tmp_path / f"sandbox-{index}.yaml"
        files[name].write_text(text, encoding="utf-8")
    sm, restore = _sandbox_window(tmp_path)
    try:
        opened = {}
        for name, path in files.items():
            sm._CONFIG_PATH = str(path)
            config = sm._load_config()
            opened[name] = os.path.join(config.workspace_base, config.audit_db_path)
        default = sm.SandboxConfig()
        assert opened["defaults"] == opened["a value that does not build"] == os.path.join(
            default.workspace_base, default.audit_db_path), ("control: the manager falls back to its defaults", opened)
        assert not os.path.isabs(opened["a relative base"]), opened["a relative base"]
        # The manager opens its log at that join: built on a base of the test's, a relative log and an absolute one.
        sm._detect_bwrap = lambda: (False, "no bubblewrap in this contract")
        (tmp_path / "logs").mkdir()
        logs = []
        for audit in ("audit.db", str(tmp_path / "logs" / "audit.db")):
            config = sm.SandboxConfig(isolation_backend="auto", strict_mode=False, require_degraded_confirmation=False,
                                      workspace_base=str(tmp_path / "built"), audit_db_path=audit,
                                      reconcile_on_start=False)
            logs.append((sm.SandboxManager(config)._audit._db_path,
                         os.path.join(config.workspace_base, config.audit_db_path)))
        assert all(got == want for got, want in logs), logs
    finally:
        restore()
    world = _window(tmp_path, monkeypatch)
    try:
        dbe = world.dbe
        bound = _census_module(source("sandbox_manager.py"))[0]
        assert os.path.realpath(bound["_CONFIG_PATH"][0]) == os.path.realpath(dbe._SANDBOX_CONFIG), (
            "the command reads the manager's own configuration", bound.get("_CONFIG_PATH"), dbe._SANDBOX_CONFIG)
        dbe._LONE_STORES = None
        for name, path in files.items():
            dbe._SANDBOX_CONFIG = path
            configured, default = dbe._sandbox_logs()
            assert default == opened["defaults"], ("the command's default is the manager's", name, default)
            if name == "a value that does not build":
                assert configured != opened[name] == default, ("the manager fell back to its defaults", configured)
            else:
                assert configured == opened[name], (name, configured, opened[name])
            assert dbe._lone_store_paths() == ([Path(configured)] if os.path.isabs(configured) else []), name
        # From here the command runs: its default base is the test's, never the real one, which it would read.
        dbe._SANDBOX_DEFAULT = str(tmp_path / "default-base")
        data = world.places[0]
        _store(data / "auth.db", 2)
        # A log configured below wherever the platform starts: said by check, and encrypt refuses, nothing changed.
        dbe._SANDBOX_CONFIG = files["a relative base"]
        assert dbe.main(["check"]) == 0
        said = capsys.readouterr().out
        assert ("sandbox: the sandbox's audit log is configured below whatever directory the platform starts in "
                "(sandboxes/sandbox_audit.db): set an absolute workspace_base in config/sandbox.yaml; encrypt refuses "
                "until then") in said, said
        snapshot = _tree(*world.places, world.keydir)
        assert dbe.main(["encrypt"]) == 1
        refused = capsys.readouterr().err
        assert "set an absolute workspace_base" in refused and _tree(*world.places, world.keydir) == snapshot, refused
        # A log at the manager's default path while the configuration names another: taken by neither, refused.
        dbe._SANDBOX_CONFIG = files["a base"]
        stale = tmp_path / "default-base" / "sandbox_audit.db"
        stale.parent.mkdir()
        _store(stale, 1)
        kept = stale.read_bytes()
        assert stale not in dbe.store_files() and len(dbe._sandbox_problems()) == 1, dbe._sandbox_problems()
        assert dbe.main(["check"]) == 0
        assert (f"sandbox: {stale} holds a sandbox audit log at the manager's default path while its configuration "
                f"names {opened['a base']}") in capsys.readouterr().out
        assert dbe.main(["encrypt"]) == 1
        refused = capsys.readouterr().err
        assert "holds a sandbox audit log" in refused and "move it there, as no log is there yet" in refused, refused
        assert stale.read_bytes() == kept and _tree(*world.places, world.keydir) == snapshot, "nothing changed"
        # The configured log there too: this one is stale, and moving it would write over the live one.
        Path(opened["a base"]).parent.mkdir(parents=True)
        _store(Path(opened["a base"]), 1)
        assert "the configured log is there too, so this one is stale" in dbe._sandbox_problems()[0], (
            dbe._sandbox_problems())
        Path(opened["a base"]).unlink()
        # Where the configured log lies cannot be looked at -- its folder not searchable, or relative: no advice to move.
        os.chmod(Path(opened["a base"]).parent, 0o600)
        try:
            assert "cannot be looked at from here: look at both before moving anything" in dbe._sandbox_problems()[0], (
                dbe._sandbox_problems())
        finally:
            os.chmod(Path(opened["a base"]).parent, 0o700)
        dbe._SANDBOX_CONFIG = files["a relative base"]
        assert len(dbe._sandbox_problems()) == 2 and "cannot be looked at from here" in dbe._sandbox_problems()[1], (
            dbe._sandbox_problems())
        # Never resolved from where the command runs, which need not be where the platform starts: from here its
        # spelling leads to the default path, and the log there is still said.
        (tmp_path / "start").mkdir()
        (tmp_path / "start" / "sandboxes").symlink_to(stale.parent, target_is_directory=True)
        monkeypatch.chdir(tmp_path / "start")
        assert os.path.realpath(dbe._sandbox_logs()[0]) == os.path.realpath(stale), "control: from here it leads there"
        assert len(dbe._sandbox_problems()) == 2 and "cannot be looked at from here" in dbe._sandbox_problems()[1], (
            "a relative log taken for the default because of where the command runs", dbe._sandbox_problems())
        dbe._SANDBOX_CONFIG = files["a base"]
        # The default path under another spelling is the same file: nothing to say.
        same = tmp_path / "sandbox-same.yaml"
        same.write_text(f"workspace_base: {tmp_path / 'default-base'}\naudit_db_path: ./sandbox_audit.db\n",
                        encoding="utf-8")
        dbe._SANDBOX_CONFIG = same
        assert dbe._sandbox_problems() == [], dbe._sandbox_problems()
        dbe._SANDBOX_CONFIG = files["a base"]
        stale.unlink()
        # Named whether it holds anything or not; taken once it does.
        log = Path(opened["a base"])
        assert dbe._lone_store_paths() == [log] and log not in dbe.store_files(), "named, not taken: it is not there"
        _store(log, 2)
        assert dbe.store_files()[-1] == log and log in dbe.plaintext_stores(), "taken once it holds something"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SE10 -- every data directory the package binds is a place of the command
# ---------------------------------------------------------------------------
# The names a module binds a directory of stores to: each must be a place or a store folder of one.
_STORE_BINDING = re.compile(r"^_?(?:DEFAULT_)?(?:DATA|DB|STORE)_DIR$")
# The names a module binds a directory of other data to (ChromaDB's, the knowledge base's): each must lie below a
# place, where the command looks for what stays in plaintext.
_WALK_BINDING = re.compile(r"^_?(?:DEFAULT_)?CHROMA_(?:DIR|BASE)$|^RAG_HOME$")
# The package's folders the census passes over: each plugin keeps its own stores, in plaintext.
_NOT_THE_PLATFORMS = ("plugins",)


class _Unread(Exception):
    """An expression the census cannot read."""


def _evaluate(node, names, here):
    """(the path ``node`` names in the module at ``here``, the environment variables it depends on), its names bound
    so far in ``names`` as such pairs: a string, a name bound before, ``__file__``, ``Path(...)``, ``.parent``,
    ``.resolve()``, ``.absolute()``, the ``/`` operator, ``os.path.join``, ``dirname`` and ``abspath``,
    ``Path.home()``, and ``os.environ.get`` or ``os.getenv`` of a named variable with a default -- read as that
    default, the variable counted. Anything else is unread."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, frozenset()
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return str(here), frozenset()
        if names.get(node.id) is not None and names[node.id][0] is not None:
            return names[node.id]
        raise _Unread(node.id)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        value, needs = _evaluate(node.value, names, here)
        return Path(value).parent, needs
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        (left, first), (right, second) = _evaluate(node.left, names, here), _evaluate(node.right, names, here)
        return Path(left) / right, first | second
    if isinstance(node, ast.Call) and not node.keywords:
        call, args = ast.unparse(node.func), node.args
        if call in ("Path", "pathlib.Path") and len(args) == 1:
            value, needs = _evaluate(args[0], names, here)
            return Path(value), needs
        if call in ("Path.home", "pathlib.Path.home") and not args:
            return Path.home(), frozenset()
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("resolve", "absolute") and not args:
            value, needs = _evaluate(node.func.value, names, here)
            return Path(os.path.abspath(value)), needs
        if call == "os.path.join" and args:
            parts = [_evaluate(arg, names, here) for arg in args]
            return os.path.join(*(str(value) for value, _ in parts)), frozenset().union(*(n for _, n in parts))
        if call in ("os.path.dirname", "os.path.abspath") and len(args) == 1:
            value, needs = _evaluate(args[0], names, here)
            return getattr(os.path, call.rsplit(".", 1)[1])(str(value)), needs
        if (call in ("os.environ.get", "os.getenv") and len(args) == 2 and isinstance(args[0], ast.Constant)
                and isinstance(args[0].value, str)):
            value, needs = _evaluate(args[1], names, here)
            return value, needs | {args[0].value}
    raise _Unread(ast.unparse(node))


def _module_file(here, node):
    """The package's source file an ``import from`` in the module at ``here`` names, or None."""
    package = source()
    if node.level:
        base = Path(here).parent
        for _ in range(node.level - 1):
            base = base.parent
        parts = node.module.split(".") if node.module else []
    elif node.module and node.module.split(".")[0] == package.name:
        base, parts = package.parent, node.module.split(".")
    else:
        return None
    target = base.joinpath(*parts)
    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
        if candidate.is_file() and package in candidate.resolve().parents:
            return candidate
    return None


def _census_module(path, *, cache=None):
    """(names, bindings) of the module at ``path``. ``names``: each name it binds at module level, to its last
    (value, environment variables) -- value None when unread --, a name imported from the package followed to its
    module. ``bindings``: every binding, in order, as (label, name, value, variables): each branch of an ``if``, a
    ``try`` or a ``with``, a ``/=``, a tuple unpacked, a class's attribute, an attribute of an object assigned at
    module level. Not read: a binding inside a function (``global`` included), a loop's, a ``with``'s, a ``match``'s
    or an assignment expression's target. A name of a directory imported from elsewhere, unpacked from what is no
    tuple, or a ``/=`` on an object's attribute, is unread."""
    cache = {} if cache is None else cache
    path = Path(path)
    if path in cache:
        return cache[path]
    names, bindings = {}, []
    cache[path] = (names, bindings)

    def read(node, scope):
        try:
            return _evaluate(node, scope, path)
        except _Unread:
            return None, frozenset()

    def bind(label, name, pair, scope, *, named=True):
        """Record a binding; a name -- not an object's attribute -- also binds in its scope, module or class."""
        bindings.append((label, name, *pair))
        if named:
            scope[name] = pair

    def assign(target, value, scope, prefix):
        """Bind what ``target`` names to what ``value`` reads -- a name in its scope, an object's attribute recorded
        only, a tuple matched element by element, recursively -- or, where the two do not match, as unread."""
        if isinstance(target, (ast.Tuple, ast.List)):
            if (value is not None and isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts)
                    and not any(isinstance(elt, ast.Starred) for elt in target.elts)):
                for each, part in zip(target.elts, value.elts):
                    assign(each, part, scope, prefix)
            else:
                for each in target.elts:
                    assign(each, None, scope, prefix)
        elif isinstance(target, ast.Starred):
            assign(target.value, None, scope, prefix)
        elif isinstance(target, ast.Name):
            if value is None or not (isinstance(value, ast.Constant) and value.value is None):
                bind(f"{path.name}: {prefix}{target.id}", target.id,
                     read(value, scope) if value is not None else (None, frozenset()), scope)
        elif isinstance(target, ast.Attribute):
            bind(f"{path.name}: {prefix}{ast.unparse(target)}", target.attr,
                 read(value, scope) if value is not None else (None, frozenset()), scope, named=False)

    def walk(body, scope, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.ClassDef):
                walk(node.body, dict(scope), f"{prefix}{node.name}.")
                continue
            if isinstance(node, ast.ImportFrom):
                module = _module_file(path, node)
                theirs = _census_module(module, cache=cache)[0] if module is not None else {}
                for alias in node.names:
                    if alias.name in theirs or _STORE_BINDING.match(alias.name) or _WALK_BINDING.match(alias.name):
                        bind(f"{path.name}: {prefix}{alias.asname or alias.name}", alias.asname or alias.name,
                             theirs.get(alias.name, (None, frozenset())), scope)
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and getattr(node, "value", None) is not None:
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    assign(target, node.value, scope, prefix)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                value = ast.BinOp(left=ast.Name(id=node.target.id, ctx=ast.Load()), op=node.op, right=node.value)
                assign(node.target, value, scope, prefix)
            elif isinstance(node, ast.AugAssign):
                assign(node.target, None, scope, prefix)
            for field in ("body", "orelse", "finalbody", "handlers", "cases"):
                inner = getattr(node, field, None)
                if isinstance(inner, list):
                    walk(inner, scope, prefix)

    walk(ast.parse(path.read_text(encoding="utf-8")).body, names, "")
    return names, bindings


def _census(paths, places, folders, followed):
    """Each directory the modules at ``paths`` bind, sorted: (found {directory: [label]}, outside, unread,
    unfollowed). A store directory must be a place or a store folder of one, and no environment variable may move
    it -- the command takes stores only in its places; another data directory must lie at or below a place, and one
    an environment variable moves counts only when the command follows that variable. Each directory is compared
    as the module spells it, made absolute: the census looks at no directory on disk."""
    stores = {Path(os.path.abspath(place)) for place in places}
    stores |= {place / folder for place in stores for folder in folders}
    roots = {Path(os.path.abspath(place)) for place in places}
    cache, found, outside, unread, unfollowed = {}, {}, [], [], []
    for path in paths:
        for label, name, value, needs in _census_module(path, cache=cache)[1]:
            kind = "store" if _STORE_BINDING.match(name) else "walk" if _WALK_BINDING.match(name) else None
            if kind is None:
                continue
            if value is None:
                unread.append(label)
                continue
            unfollowed += [f"{label} ({need})" for need in sorted(needs) if need not in followed or kind == "store"]
            if not Path(value).is_absolute():
                outside.append(f"{label} = {value} (below whatever directory the platform starts in)")
                continue
            named = Path(os.path.abspath(value))
            found.setdefault(named, []).append(label)
            if (named not in stores) if kind == "store" else not any(named == root or root in named.parents
                                                                      for root in roots):
                outside.append(f"{label} = {named}")
    return found, outside, unread, unfollowed


def _joined(paths, folder):
    """The modules that join a data directory with ``folder`` anywhere -- ``DATA_DIR / folder`` or
    ``Path(DATA_DIR) / folder``, in a function too."""
    joins = []
    for path in paths:
        for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant)
                    and node.right.value == folder and _STORE_BINDING.match(
                        re.sub(r"^Path\((.*)\)$", r"\1", ast.unparse(node.left)))):
                joins.append(Path(path).name)
    return joins


def test_se10_every_data_directory_the_package_binds_is_a_place_of_the_command(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        dbe = world.dbe
        dbe._STORE_PLACES = None
        places = [Path(os.path.abspath(place)) for place in dbe._places()]
        package = source()
        # Walked without entering a data directory: what lies there is the maintainer's, never a module.
        modules = sorted(path for path in package_modules().values()
                         if path.relative_to(package).parts[0] not in _NOT_THE_PLATFORMS)
        found, outside, unread, unfollowed = _census(modules, places, dbe._STORE_FOLDERS, dbe._FOLLOWED)
        assert not unread, ("bindings of a data directory the census cannot read: bind it in a form it reads, or teach "
                            "it the form", unread)
        assert not outside, ("data directories the command does not take: their stores would stay in plaintext",
                             outside)
        assert not unfollowed, ("data directories an environment variable moves, which the command does not follow",
                                unfollowed)
        assert all(place in found for place in places), ("a place no module binds", places, sorted(found))
        assert all(_joined(modules, folder) for folder in dbe._STORE_FOLDERS), (
            "a store folder no module joins to a data directory", dbe._STORE_FOLDERS)
        assert len(found[places[1]]) >= 10 and "config.py: DATA_DIR" in found[places[1]], (
            "control: the census reads the package's bindings", found[places[1]][:12])
        assert any("RAG_HOME" in label for labels in found.values() for label in labels), (
            "control: the knowledge base's home is read, the variable that moves it followed")
        # Control: every form the census reads, sorted -- a place, outside, unread, an unfollowed variable.
        sample = tmp_path / "sample" / "opti_oignon_sample.py"
        sample.parent.mkdir()
        sample.write_text(
            "import os\nfrom pathlib import Path\nfrom elsewhere import DB_DIR\n"
            f"DATA_DIR = Path({str(places[0])!r})\n"
            "try:\n    _DB_DIR = Path.home() / 'oo'\nexcept ImportError:\n    _DB_DIR = DATA_DIR\n"
            "_STORE_DIR = locate()\n"
            "RAG_HOME = Path(os.environ.get('OO_SAMPLE_HOME', DATA_DIR))\n"
            "_DEFAULT_DB_DIR = Path(os.environ.get('RAG_HOME', DATA_DIR))\n"
            "DATA_DIR /= 'v2'\n"
            "class Settings:\n    _BASE = Path('/srv')\n    CHROMA_DIR = _BASE / 'chroma'\n"
            "settings = Settings()\nsettings.DATA_DIR = Path('/srv/data')\nsettings.DATA_DIR /= 'x'\n"
            "STORE_DIR, other = Path('/srv/db'), 1\nDB_DIR, rest = locate()\nCHROMA_BASE, *more = (DATA_DIR, 1)\n"
            "match os.name:\n    case 'posix':\n        _DATA_DIR = Path('/srv/posix')\n"
            "settings.DB_DIR, x = locate()\n(a, (_DB_DIR, b)) = (1, (Path('/srv/nested'), 2))\n", encoding="utf-8")
        found, outside, unread, unfollowed = _census([sample], places, dbe._STORE_FOLDERS, dbe._FOLLOWED)
        assert sorted(unread) == sorted(f"opti_oignon_sample.py: {name}" for name in (
            "DB_DIR", "_STORE_DIR", "settings.DATA_DIR", "DB_DIR", "CHROMA_BASE", "settings.DB_DIR")), unread
        assert unfollowed == ["opti_oignon_sample.py: RAG_HOME (OO_SAMPLE_HOME)",
                              "opti_oignon_sample.py: _DEFAULT_DB_DIR (RAG_HOME)"], unfollowed
        assert sorted(item.split(" = ")[0] for item in outside) == sorted(
            f"opti_oignon_sample.py: {name}" for name in ("_DB_DIR", "DATA_DIR", "Settings.CHROMA_DIR",
                                                           "settings.DATA_DIR", "STORE_DIR", "_DATA_DIR",
                                                           "_DB_DIR")), outside
        assert sorted(found[places[0]]) == sorted(f"opti_oignon_sample.py: {name}" for name in (
            "DATA_DIR", "_DB_DIR", "RAG_HOME", "_DEFAULT_DB_DIR")), found
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SE11 -- what stays in plaintext is said, and what was not checked
# ---------------------------------------------------------------------------
def test_se11_every_sqlite_file_left_in_plaintext_below_the_places_is_said(tmp_path, monkeypatch, capsys):
    world = _window(tmp_path, monkeypatch)
    locked = []
    try:
        data, package = world.places
        _store(data / "auth.db", 2)
        _store(package / "notes.db", 2)
        for folder in ("rag/chroma_db", "rag/chroma_v2", "projects/p1", "projects/plaintext-notes",
                       "plaintext-20990101T000000Z", "closed"):
            (package / folder).mkdir(parents=True)
        chroma = package / "rag" / "chroma_db" / "chroma.sqlite3"
        _store(chroma, 3)
        other = package / "projects" / "p1" / "tables.sqlite"
        _store(other, 1)
        named_plaintext = package / "projects" / "plaintext-notes" / "chroma.sqlite3"
        _store(named_plaintext, 1)
        _store(package / "projects" / "p1" / "user.db", 1)
        _store(package / "plaintext-20990101T000000Z" / "kept.sqlite", 1)
        outside = tmp_path / "outside"
        outside.mkdir()
        _store(outside / "far.sqlite", 1)
        os.symlink(outside / "far.sqlite", package / "projects" / "p1" / "linked.sqlite")
        os.symlink(outside, package / "projects" / "elsewhere")
        os.mkfifo(package / "projects" / "p1" / "pipe.sqlite")
        (package / "projects" / "p1" / "readme.txt").write_text("SQLite format 3, said in words\n", encoding="utf-8")
        shut = package / "projects" / "p1" / "shut.sqlite"
        _store(shut, 1)
        _store(package / "closed" / "inside.sqlite", 1)
        for path in (shut, package / "closed"):
            os.chmod(path, 0)
            locked.append(path)
        odd = os.path.join(os.fsencode(package / "projects" / "p1"), b"caf\xe9.sqlite")
        with open(odd, "wb") as handle:
            handle.write(other.read_bytes())
        odd = Path(os.fsdecode(odd))
        shutil.copyfile(package / "notes.db", package / "notes.db.restoring")
        sealed = package / "rag" / "chroma_v2" / "chroma.sqlite3"
        conn = world.dbe._keyed(str(sealed), os.urandom(32))
        try:
            conn.execute("CREATE TABLE t (x)")
            conn.commit()
        finally:
            conn.close()
        assert sealed.read_bytes()[:16] != _PLAIN, "control: a database no plaintext reader opens"
        home = tmp_path / "knowledge"
        (home / "chroma_db").mkdir(parents=True)
        moved = home / "chroma_db" / "chroma.sqlite3"
        _store(moved, 1)
        (home / "cache").mkdir()
        jobs = home / "cache" / "jobs.db"
        _store(jobs, 1)
        monkeypatch.setenv("RAG_HOME", str(home))
        others = world.dbe.plaintext_others()
        assert others == [odd, other, named_plaintext, chroma, jobs, moved] and world.dbe.plaintext_others() == others, (
            "every plaintext database below the places, and where RAG_HOME moves the knowledge base -- a *.db there "
            "no store --, in one order", others)
        unchecked = world.dbe.unchecked_entries()
        assert sorted(unchecked) == sorted([
            (package / "closed", "could not be read"), (package / "projects" / "elsewhere",
                                                        "a symbolic link, never followed"),
            (package / "projects" / "p1" / "linked.sqlite", "a symbolic link, never followed"),
            (shut, "could not be read")]), unchecked
        said_of = {path: (f"{path}: a ChromaDB vector store, its texts and vectors in plaintext: ChromaDB has no "
                          "encryption at rest, and this command does not encrypt it") if path.name == "chroma.sqlite3"
                   else f"{path}: a database in plaintext that is no store of the platform: this command does not "
                   "encrypt it" for path in others}
        assert world.dbe.main(["check"]) == 0
        said = capsys.readouterr().out
        assert all(line.replace(str(odd), "") in said.replace("caf\\udce9.sqlite", "") for line in said_of.values()), said
        assert "caf\\udce9.sqlite: a database in plaintext" in said, ("a name the locale cannot write, said escaped", said)
        assert said.count("this command does not encrypt it") == 6, said
        assert all(f"{path}: not checked, {why}" in said for path, why in unchecked), said
        assert f"left by an interrupted run: {package / 'notes.db.restoring'}" in said, (
            "what a killed run left said as such, not as a database of another kind", said)
        (package / "notes.db.restoring").unlink()
        before = {path: path.read_bytes() for path in (chroma, other, named_plaintext, jobs, moved)}
        assert world.dbe.main(["encrypt"]) == 0
        said = capsys.readouterr().out
        assert "2 store(s) encrypted" in said and said.count("this command does not encrypt it") == 6, said
        assert all(f"{path}: not checked, {why}" in said for path, why in unchecked), said
        assert all(path.read_bytes() == kept for path, kept in before.items()), "never encrypted, never moved"
        again = world.dbe.encrypt_all_stores()
        assert again["plaintext_others"] == [str(path) for path in others] and again["unchecked"] == [
            [str(path), why] for path, why in unchecked], "said by each run"
        # RAG_HOME relative to wherever the platform starts: said, never walked from here.
        monkeypatch.chdir(tmp_path)  # where "knowledge" lies: a walk of it would be seen
        monkeypatch.setenv("RAG_HOME", "knowledge")
        assert world.dbe.plaintext_others() == others[:4], world.dbe.plaintext_others()
        assert world.dbe.main(["check"]) == 0
        assert ("RAG_HOME is knowledge, below whatever directory the platform starts in: not checked"
                in capsys.readouterr().out)
        # RAG_HOME above a place: the place walked once, as a place -- no file said twice, no store said as another.
        monkeypatch.setenv("RAG_HOME", str(package.parent))
        assert world.dbe.plaintext_others() == others[:4], world.dbe.plaintext_others()
        # RAG_HOME that cannot be looked at: said, never passed over.
        shut_home = tmp_path / "shut-home"
        (shut_home / "kb").mkdir(parents=True)
        os.chmod(shut_home, 0)
        locked.append(shut_home)
        monkeypatch.setenv("RAG_HOME", str(shut_home / "kb"))
        assert (shut_home / "kb", "could not be read") in world.dbe.unchecked_entries(), world.dbe.unchecked_entries()
        os.chmod(shut_home, 0o700)
        # RAG_HOME through a link inside a place: its target walked, the link never said unchecked.
        os.symlink(home, package / "kb")
        monkeypatch.setenv("RAG_HOME", str(package / "kb"))
        assert jobs.name in [path.name for path in world.dbe.plaintext_others()], world.dbe.plaintext_others()
        assert not [path for path, _why in world.dbe.unchecked_entries() if path == package / "kb"], (
            world.dbe.unchecked_entries())
    finally:
        for path in locked:
            os.chmod(path, 0o700)
        world.restore()


# ---------------------------------------------------------------------------
# SE12 -- git never takes what a run left
# ---------------------------------------------------------------------------
def test_se12_git_ignores_what_a_run_killed_outright_leaves_where_the_platform_keeps_it(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        dbe, (data, package) = world.dbe, world.places
        (package / "rag").mkdir()
        _store(data / "auth.db", 2)
        _store(package / "rag" / "rag_documents.db", 2)
        kept = data / "plaintext-20990101T000000Z"
        kept.mkdir()
        os.link(data / "auth.db", kept / "auth.db")
        os.link(data / "auth.db", data / "auth.db.restoring")
        _store(package / "notes.db.encrypting", 1)
        _store(package / "rag" / "rag_documents.db.restoring", 1)
        (world.keydir / "..keyfile.tmp-0a1b2c3d").write_bytes(b"half a key file")
        named = [Path(item.split(" (", 1)[0]) for item in dbe._leftovers(dbe.store_files(), world.places)]
        left = [data / "auth.db.restoring", package / "rag" / "rag_documents.db.restoring",
                package / "notes.db.encrypting", world.keydir / "..keyfile.tmp-0a1b2c3d"]
        assert sorted(named) == sorted(left), ("control: the command names each", named)
        # Where the platform keeps them: the places as the command reads them, the key file as encryption.py binds it.
        dbe._STORE_PLACES = None
        real = [Path(place).relative_to(_REPO) for place in dbe._places()]
        keyfile = Path(_census_module(source("encryption.py"))[0]["_DEFAULT_KEYFILE"][0]).relative_to(_REPO)
        rooms = {data: real[0], package: real[1], world.keydir: keyfile.parent}
        paths = [str(rooms[path.parent] if path.parent in rooms else rooms[path.parent.parent] / path.parent.name)
                 + "/" + path.name for path in left]
        for room in (real[0], real[1], real[1] / "rag"):
            paths += [f"{room}/plaintext-20990101T000000Z/conversations.db{side}" for side in ("", *_SIDES)]
        paths.append(str(keyfile))
        probe = "opti_oignon/db_encryption.py"
        answer = subprocess.run(["git", "-C", str(_REPO), "check-ignore", "--no-index", "-v", "-n", *paths, probe],
                                capture_output=True, text=True, timeout=60)
        rules = dict(reversed(line.split("\t", 1)) for line in answer.stdout.splitlines())

        def ignored(path):
            """Whether git ignores ``path``: a rule matched it, and not a negated one (``!``)."""
            source_, _, rest = rules.get(path, "::").partition(":")
            return bool(source_) and not rest.partition(":")[2].startswith("!")

        assert not ignored(probe) and probe in rules, ("control: a source file is not ignored", answer.stdout,
                                                       answer.stderr)
        taken = [path for path in paths if not ignored(path)]
        assert not taken, ("git would take these", taken, answer.stderr)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SE13 -- no key from the shell
# ---------------------------------------------------------------------------
_KEYS = ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME")


def test_se13_the_suites_drop_the_key_the_passphrase_and_the_data_directory_a_shell_carries():
    conftest = Path(__file__).resolve().parent / "conftest.py"
    env = dict(os.environ, OPTI_ENCRYPTION_KEY="a key from the shell", OPTI_KEYFILE_PASSPHRASE="a phrase from the shell",
               RAG_HOME="/a/knowledge/base/of/the/shell")
    tell = "import os, sys; print(sorted(name for name in sys.argv[2:] if name in os.environ))"
    plain = subprocess.run([sys.executable, "-c", tell, "-", *_KEYS], env=env, capture_output=True, text=True,
                           timeout=60)
    assert plain.stdout.strip() == str(sorted(_KEYS)), ("control: the child is given both", plain.stdout, plain.stderr)
    loaded = subprocess.run([sys.executable, "-c", "import runpy, sys; runpy.run_path(sys.argv[1]); " + tell,
                             str(conftest), *_KEYS], env=env, capture_output=True, text=True, timeout=60)
    assert loaded.returncode == 0 and loaded.stdout.strip().splitlines()[-1] == "[]", (
        "the suites' configuration drops each when it is imported", loaded.stdout, loaded.stderr)
    assert not [name for name in _KEYS if name in os.environ], "and this session carries none of them"
    followed = next(ast.literal_eval(node.value) for node in ast.parse(source("db_encryption.py").read_text(
        encoding="utf-8")).body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_FOLLOWED")
    assert followed and set(followed) <= set(_KEYS), (
        "every variable the command follows to a directory is dropped too: the suites never walk a real one",
        followed)


# ---------------------------------------------------------------------------
# SE14 -- the server says which key, what stays in plaintext, whether the key file is wrapped
# ---------------------------------------------------------------------------
def _setup(root, monkeypatch, request, *, env=(), stores=(), encrypt=False):
    """Setup's answer in a world of its own under ``root``: ``env`` set, ``stores`` made in plaintext, encrypted
    first when asked; (the answer, whether a key was made, the key file's bytes before and after)."""
    root.mkdir()
    world = _window(root, monkeypatch, routes=True)
    try:
        for name, value in env:
            monkeypatch.setenv(name, value)
        for name in stores:
            _store(world.places[0] / name, 2)
        if encrypt:
            world.dbe.encrypt_all_stores()
        made = []
        generate = world.enc.generate_key
        world.enc.generate_key = lambda *args, **kwargs: made.append(1) or generate(*args, **kwargs)
        before = world.keyfile.read_bytes() if world.keyfile.exists() else None
        answer = world.routes.setup_encryption(world.routes.EncryptionSetupRequest(**request))
        return answer, bool(made), before, world.keyfile.read_bytes() if world.keyfile.exists() else None
    finally:
        for name, _value in env:
            monkeypatch.delenv(name, raising=False)
        world.restore()


def test_se14_setup_says_whether_it_enabled_the_key_in_place_or_made_one(tmp_path, monkeypatch):
    phrase = {"mode": "passphrase", "passphrase": "a passphrase never used"}
    answer, made, before, after = _setup(tmp_path / "encrypted", monkeypatch, phrase, stores=["auth.db"], encrypt=True)
    assert (answer["setup"], answer["key"], answer["detail"], answer["plaintext_stores"], answer["wrapped"]) == (
        True, "existing", "Encryption enabled with the existing key", 0, None), answer
    assert not made and before is not None and after == before, "no key made, the key file as it was"
    environment = [("OPTI_ENCRYPTION_KEY", base64.urlsafe_b64encode(os.urandom(32)).decode("ascii"))]
    answer, made, _, after = _setup(tmp_path / "environment", monkeypatch, {"mode": "random"}, env=environment,
                                    stores=["auth.db"])
    assert (answer["key"], answer["plaintext_stores"], made, after) == ("existing", 1, False, None), (
        "the environment's key taken, the store it leaves unopenable counted", answer)
    answer, made, _, after = _setup(tmp_path / "fresh", monkeypatch, {"mode": "random"})
    assert (answer["key"], answer["wrapped"], answer["plaintext_stores"], made) == ("made", False, 0, True), answer
    answer, *_ = _setup(tmp_path / "phrased", monkeypatch, {"mode": "random"},
                        env=[("OPTI_KEYFILE_PASSPHRASE", "the server's own passphrase")])
    assert (answer["key"], answer["wrapped"]) == ("made", True), ("wrapped under the server's passphrase", answer)
    answer, *_ = _setup(tmp_path / "typed", monkeypatch, {"mode": "passphrase", "passphrase": "a typed passphrase"})
    assert (answer["key"], answer["wrapped"]) == ("made", True), ("wrapped under the typed passphrase", answer)


# ---------------------------------------------------------------------------
# SE15 -- a folder of stores that cannot be read refuses the run; a linked one is followed
# ---------------------------------------------------------------------------
def test_se15_a_folder_of_stores_that_cannot_be_read_refuses_and_a_linked_one_is_followed(tmp_path, monkeypatch,
                                                                                            capsys):
    world = _window(tmp_path, monkeypatch)
    locked = []

    def lock(path, mode):
        os.chmod(path, mode)
        locked.append(path)

    try:
        dbe, (data, package) = world.dbe, world.places
        _store(data / "auth.db", 2)
        rag = package / "rag"
        rag.mkdir()
        _store(rag / "rag_documents.db", 2)
        snapshot = _tree(data, world.keydir)
        # Neither readable nor searchable; readable, not searchable: its stores out of sight either way.
        for mode in (0, 0o600):
            lock(rag, mode)
            assert dbe.main(["check"]) == 0
            assert f"{rag}: cannot be read: its stores are not checked, and encrypt refuses until it can" in (
                capsys.readouterr().out), mode
            assert dbe.main(["encrypt"]) == 1
            refused = capsys.readouterr().err
            assert f"{rag} cannot be read: the stores in it cannot be taken: nothing was changed" in refused, (
                mode, refused)
            assert _tree(data, world.keydir) == snapshot, ("nothing changed", mode)
        os.chmod(rag, 0o700)
        # A place that cannot be read: said, never a trace.
        lock(package, 0)
        assert dbe.main(["check"]) == 0
        assert f"{package}: cannot be read: its stores are not checked" in capsys.readouterr().out
        assert dbe.main(["encrypt"]) == 1 and f"{package} cannot be read" in capsys.readouterr().err
        os.chmod(package, 0o700)
        # The sandbox's folder, read from its configuration, the same.
        dbe._LONE_STORES, dbe._SANDBOX_DEFAULT = None, str(tmp_path / "default-base")
        config = tmp_path / "sandbox.yaml"
        config.write_text(f"workspace_base: {tmp_path / 'sandbox'}\n", encoding="utf-8")
        dbe._SANDBOX_CONFIG = config
        (tmp_path / "sandbox").mkdir()
        _store(tmp_path / "sandbox" / "sandbox_audit.db", 1)
        lock(tmp_path / "sandbox", 0)
        assert dbe.main(["encrypt"]) == 1
        assert f"{tmp_path / 'sandbox'} cannot be read" in capsys.readouterr().err
        assert _tree(data, world.keydir) == snapshot, "nothing changed"
        os.chmod(tmp_path / "sandbox", 0o700)
        dbe._LONE_STORES = []
        # A store folder that is a symbolic link -- a knowledge base on another disk -- is followed, as its stores are.
        elsewhere = tmp_path / "other-disk" / "rag"
        elsewhere.mkdir(parents=True)
        shutil.move(str(rag / "rag_documents.db"), str(elsewhere / "rag_documents.db"))
        rag.rmdir()
        os.symlink(elsewhere, rag)
        (elsewhere / "chroma_db").mkdir()
        _store(elsewhere / "chroma_db" / "chroma.sqlite3", 1)
        (elsewhere / "archive").mkdir()
        _store(elsewhere / "archive" / "old.db", 1)
        chroma = rag / "chroma_db" / "chroma.sqlite3"
        assert dbe.plaintext_others() == [chroma], dbe.plaintext_others()
        assert dbe.unchecked_entries() == [], ("the store folder is no link left unchecked", dbe.unchecked_entries())
        assert dbe.other_databases() == [rag / "archive" / "old.db"], ("a *.db below it named", dbe.other_databases())
        # RAG_HOME at the link's real target: the same folder, walked once, its stores never said as others.
        monkeypatch.setenv("RAG_HOME", str(elsewhere))
        assert dbe.plaintext_others() == [chroma], dbe.plaintext_others()
        assert dbe.main(["encrypt"]) == 0
        said = capsys.readouterr().out
        assert "2 store(s) encrypted" in said and f"{rag}: not checked" not in said, said
        assert said.count(": a ChromaDB vector store") == 1 and "rag_documents.db: a database in plaintext" not in said, (
            said)
        assert (elsewhere / "rag_documents.db").read_bytes()[:16] != _PLAIN, "encrypted through the link"
        assert f"{chroma}: a ChromaDB vector store" in said, said
        assert dbe.main(["check"]) == 0
        said = capsys.readouterr().out
        assert "a database in plaintext that is no store" not in said, ("the plaintext copy never said as another", said)
    finally:
        for path in locked:
            if os.path.lexists(path):
                os.chmod(path, 0o700)
        world.restore()
    # A store folder linked to another folder of its own place: one store, never said as another database.
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    world = _window(sibling, monkeypatch)
    try:
        package = world.places[1]
        (package / "rag-2026" / "sub").mkdir(parents=True)
        _store(package / "rag-2026" / "rag_documents.db", 2)
        _store(package / "rag-2026" / "sub" / "old.db", 1)
        os.symlink(package / "rag-2026", package / "rag")
        assert world.dbe.other_databases() == [package / "rag-2026" / "sub" / "old.db"], world.dbe.other_databases()
        world.dbe.encrypt_all_stores()
        assert world.dbe.other_databases() == [package / "rag-2026" / "sub" / "old.db"], (
            "the plaintext copy, kept under the folder's other name, never said as another database",
            world.dbe.other_databases())
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# SE16 -- a key variable that is no key refuses, in the command and in the server
# ---------------------------------------------------------------------------
def test_se16_an_environment_key_that_is_no_key_is_refused_never_passed_over(tmp_path, monkeypatch, capsys):
    hexadecimal = "ab" * 32  # what `openssl rand -hex 32` prints: 48 bytes once decoded, no key
    world = _window(tmp_path, monkeypatch)
    try:
        data = world.places[0]
        _store(data / "auth.db", 2)
        monkeypatch.setenv("OPTI_ENCRYPTION_KEY", hexadecimal)
        snapshot = _tree(data, world.keydir)
        assert world.dbe.main(["encrypt"]) == 1
        refused = capsys.readouterr().err
        assert ("OPTI_ENCRYPTION_KEY is set and is no key the platform reads (32 bytes, base64-encoded with its "
                "padding: 44 characters ending in a single '=')") in refused, refused
        assert _tree(data, world.keydir) == snapshot and not world.keyfile.exists(), "no key file made, nothing moved"
        assert world.dbe.main(["check"]) == 0
        assert "key: OPTI_ENCRYPTION_KEY is set and is no key" in capsys.readouterr().out
        # Even with a key file that loads, the variable is never passed over in silence.
        monkeypatch.delenv("OPTI_ENCRYPTION_KEY")
        world.dbe.encrypt_all_stores()
        monkeypatch.setenv("OPTI_ENCRYPTION_KEY", hexadecimal)
        assert world.dbe.main(["encrypt"]) == 1 and "is no key" in capsys.readouterr().err
        # check reads the stores as the platform does, under the key file it passes over to.
        assert world.dbe.main(["check"]) == 0
        said = capsys.readouterr().out
        assert "meanwhile the platform passes over it to the key file, which loads" in said and (
            f"{data / 'auth.db'}: encrypted" in said) and "a key loads" in said, said
    finally:
        world.restore()
    # Behind the variable, a key file wrapped under a passphrase this shell does not carry: both are said.
    wrapped = tmp_path / "wrapped"
    wrapped.mkdir()
    world = _window(wrapped, monkeypatch)
    try:
        _store(world.places[0] / "auth.db", 2)
        monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", "a passphrase of this contract")
        world.dbe.encrypt_all_stores()
        monkeypatch.delenv("OPTI_KEYFILE_PASSPHRASE")
        monkeypatch.setenv("OPTI_ENCRYPTION_KEY", hexadecimal)
        assert world.dbe.main(["check"]) == 0
        said = capsys.readouterr().out
        assert ("the key file it passes over to does not load here either (if it is wrapped under a passphrase, set "
                "OPTI_KEYFILE_PASSPHRASE)") in said, said
    finally:
        world.restore()
    server = tmp_path / "server"
    server.mkdir()
    world = _window(server, monkeypatch, routes=True)
    try:
        monkeypatch.setenv("OPTI_ENCRYPTION_KEY", hexadecimal)
        config = world.config.read_bytes()
        with pytest.raises(world.routes.HTTPException) as refused:
            world.routes.setup_encryption(world.routes.EncryptionSetupRequest(mode="random"))
        assert refused.value.status_code == 409 and "OPTI_ENCRYPTION_KEY is set" in refused.value.detail, (
            refused.value.detail)
        assert not world.keyfile.exists() and world.config.read_bytes() == config, "no key made, nothing saved"
    finally:
        world.restore()
