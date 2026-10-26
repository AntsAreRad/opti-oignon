#!/usr/bin/env python3
"""Contracts for forgetting: deleting a conversation forgets what the onion drew from it.

The onion's store kept a deleted conversation in its twelve tables -- its Core, its Cellar spans, its turns, its
peels, what the librarian read -- and the librarian kept its state in the process until a restart; both were served
again, and a writer of generation 0 could write it all back. A deletion now forgets the onion's copy first: the store
deletes its rows, overwritten, and keeps a tombstone that refuses any save of it; the process lets go of its state. A
copy that cannot be forgotten refuses the deletion by name, nothing deleted.

  * FO1 -- A DELETED CONVERSATION LEAVES EVERY TABLE OF THE ONION'S STORE, its cursor too; another conversation's
    rows stay as they were.
  * FO2 -- THE STORE'S FILE NO LONGER HOLDS ITS WORDS: what is deleted is overwritten, on an engine that would not
    overwrite by default.
  * FO3 -- THE TOMBSTONE HOLDS ITS ID AND THE TIME, NOTHING ELSE.
  * FO4 -- NO WRITER BRINGS IT BACK: a save at generation 0 or at the generation read before the deletion, from this
    store or another on the same file, and the librarian's own save of a state it held, refused by name, nothing
    written.
  * FO5 -- NOTHING OF IT IS SERVED: the librarian's peek is None, the store loads nothing, a turn that asks for its
    state gets one that is never kept.
  * FO6 -- NOTHING OF IT IS HELD IN THE PROCESS: its state, watermark, counter and refusals are let go; curating it
    does nothing.
  * FO7 -- A DELETION THE ONION CANNOT FORGET IS REFUSED BY NAME: the conversation, its messages and its runs remain;
    the route answers 409 with the reason.
  * FO8 -- A PEER'S DELETION FORGETS IT TOO; one the onion cannot forget is not applied, and the conversation stays.
  * FO9 -- WITH NO STORE, A DELETION LETS GO OF THE PROCESS'S STATE AND CREATES NO FILE.
  * FO10 -- A FILE WRITTEN BEFORE THE TOMBSTONES OPENS AT SCHEMA 5, KEEPS ITS CONVERSATIONS AND FORGETS ONE.
  * FO11 -- A TREE WITHOUT THE MEMORY PACKAGE STILL DELETES, AND SAYS IT.
  * FO12 -- A FILE WRITTEN BEFORE IS REWRITTEN ONCE, AND ITS FREE PAGES KEEP NOTHING: held back at its opening, the
    rewrite is said and done at the next forgetting.
  * FO13 -- A FORGETTING ANOTHER CONNECTION HOLDS BACK IS REFUSED BY NAME AND FORGETS NOTHING; once it lets go, the
    deletion goes through.
  * FO14 -- THE STORE REFUSING TO FORGET REFUSES THE DELETION, through the real librarian, which lets go of nothing.
  * FO15 -- A STATE READ BEFORE THE DELETION IS NEVER INSTALLED AFTER IT, by a peek or by a reading again.
  * FO16 -- A STORE THAT CANNOT BE LOOKED INTO REFUSES THE DELETION, by the reason's name, a file there or not: a
    plaintext connection where encryption is required, a volume not mounted; no file is left.
  * FO17 -- A WIPE STOPS AT A DELETION REFUSED, ASKS NO MORE, AND SAYS HOW MANY ARE LEFT.
  * FO18 -- A MEMORY PACKAGE THAT IS THERE AND BREAKS REFUSES THE DELETION.
  * FO19 -- A SAVE THAT TAKES A TURN BACK LEAVES NONE OF ITS WORDS IN THE FILE: every connection overwrites what it
    deletes.

Local-only (the public distribution ships no tests). conversation.py and the memory package loaded from source in one
shared isolation window; the onion's store a plain file in tmp, on a connection that does not overwrite deleted pages
by default; the run journal a recorder of the conversations it forgot.
"""

import dataclasses
import json
import logging
import sqlite3
import sys
import time
import types
from contextlib import closing
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

LIB = "opti_oignon.memory.librarian"
STORE = "opti_oignon.memory.onion_store"
CONV = "opti_oignon.conversation"
_MEMORY = ("probes", "core_store", "receipts", "composer", "peels", "librarian", "onion_store")
BUDGET_S = {
    "test_fo1_a_deleted_conversation_leaves_every_table_of_the_onion_store": 3.0,
    "test_fo2_the_store_file_no_longer_holds_its_words": 3.0,
    "test_fo3_the_tombstone_holds_its_id_and_the_time_nothing_else": 3.0,
    "test_fo4_no_writer_brings_it_back": 3.0,
    "test_fo5_nothing_of_it_is_served": 3.0,
    "test_fo6_nothing_of_it_is_held_in_the_process": 3.0,
    "test_fo7_a_deletion_the_onion_cannot_forget_is_refused_by_name": 3.0,
    "test_fo8_a_peers_deletion_forgets_it_too": 3.0,
    "test_fo9_with_no_store_a_deletion_lets_go_of_the_process_state_and_creates_no_file": 3.0,
    "test_fo10_a_file_written_before_the_tombstones_opens_at_schema_5_and_forgets_one": 3.0,
    "test_fo11_a_tree_without_the_memory_package_still_deletes_and_says_it": 3.0,
    "test_fo12_a_file_written_before_is_rewritten_once_and_its_free_pages_keep_nothing": 3.0,
    "test_fo13_a_forgetting_another_connection_holds_back_is_refused_and_forgets_nothing": 3.0,
    "test_fo14_the_store_refusing_to_forget_refuses_the_deletion_through_the_real_librarian": 3.0,
    "test_fo15_a_state_read_before_the_deletion_is_never_installed_after_it": 3.0,
    "test_fo16_a_store_that_cannot_be_looked_into_refuses_the_deletion": 3.0,
    "test_fo17_a_wipe_stops_at_a_deletion_refused_and_says_how_many_are_left": 3.0,
    "test_fo18_a_memory_package_that_is_there_and_breaks_refuses_the_deletion": 3.0,
    "test_fo19_a_save_that_takes_a_turn_back_leaves_none_of_its_words_in_the_file": 3.0,
}


def _plain(path):
    """A plain connection as the store's own are -- the rollback journal, a wait for a lock (the store's five seconds,
    shortened here) -- that, like many builds, does not overwrite deleted pages unless asked to."""
    conn = sqlite3.connect(str(path), timeout=0.3)
    conn.execute("PRAGMA secure_delete = OFF")
    return conn


class _HeldBack:
    """A connection another process holds back from rewriting the file, as a reader that has it open would."""

    def __init__(self, conn):
        object.__setattr__(self, "_conn", conn)

    def execute(self, sql, *args):
        if sql.strip().upper() == "VACUUM":
            raise sqlite3.OperationalError("database is locked")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        setattr(self._conn, name, value)


def _world(tmp_path, *, store=True, librarian=None, memory=True, broken_memory=None, wipe=False):
    """conversation.py with the real memory package beside it -- or ``librarian`` in the librarian's place, or, when
    not ``memory``, no memory package at all -- over ``tmp_path``'s data; the run journal records what it forgot;
    the onion's store, when ``store``, a plain file there, the librarian's seam pointed at it."""
    data = tmp_path / "data"
    data.mkdir(parents=True)
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda path, **kwargs: sqlite3.connect(
        str(path), check_same_thread=kwargs.get("check_same_thread", False))
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = data
    forgot = []
    journal = types.ModuleType("opti_oignon.run_journal")
    journal.forget_conversation = forgot.append
    targets = {CONV: source("conversation.py")}
    if wipe:
        targets["opti_oignon.conversation_wipe"] = source("conversation_wipe.py")
    seeded = {"opti_oignon.db_utils": db, "opti_oignon.config": cfg, "opti_oignon.run_journal": journal}
    packages = ("opti_oignon.memory",)
    if broken_memory is not None:
        # The package is there, and asking it for its librarian raises ``broken_memory``, as an __init__ that
        # breaks, or one that needs a module this tree lacks, would.
        broken = types.ModuleType("opti_oignon.memory")
        broken.__path__ = []

        def _breaks(name):
            raise broken_memory

        broken.__getattr__ = _breaks
        seeded["opti_oignon.memory"] = broken
        packages = ()
    elif librarian is not None:
        seeded[LIB] = librarian
    elif memory:
        targets.update({f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _MEMORY})
    loaded, restore = isolate(targets=targets, seeded=seeded, packages=packages)
    try:
        conv = loaded[CONV]
        manager = conv.ConversationManager(db_path=tmp_path / "conversations.db", publish_to_sync=False)
        lib = onion = real_store = None
        if librarian is None and memory and broken_memory is None:
            lib = loaded[LIB]
            lib.reset_librarian()
            real_store = lib.onion_store
            if store:
                # The machine's configuration names the store's file, as a local.yaml does, and the store is the
                # librarian's seam: every reading and every forgetting goes through it.
                onion = loaded[STORE].OnionStore(data / "onion.db", connect=_plain, require_encryption=False)
                named = dataclasses.replace(lib.load_config(), persist_path=str(onion.path), require_encryption=False)
                lib.load_config = lambda *args, **kwargs: named
            lib.onion_store = lambda config=None: onion
    except BaseException:
        restore()
        raise
    return SimpleNamespace(loaded=loaded, restore=restore, conv=conv, manager=manager, lib=lib, onion=onion,
                           data=data, forgot=forgot, real_store=real_store)


def _messages(word, n=12):
    return [{"role": "user" if i % 2 else "assistant",
             "content": f"Turn {i}: {word} reviewed service {i} on 2026-03-{i:02d} and {word} keeps it on the cluster."}
            for i in range(1, n + 1)]


def _faithful(turns):
    return " ".join(str(t.get("text", "")) for t in turns)


def _conversation(world, word):
    """A conversation in the store, its turns the user's ``word``, and the onion's state of it grown and saved."""
    cid = world.manager.create_conversation(title=f"about {word}").id
    for message in _messages(word):
        world.manager.add_message(cid, message["role"], message["content"])
    peels = world.loaded["opti_oignon.memory.peels"]
    composer = world.loaded["opti_oignon.memory.composer"]
    core_store = world.loaded["opti_oignon.memory.core_store"]
    gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
    budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)
    state = world.lib.state_for(cid)
    state.core.add(f"The user is {word}.", actor=core_store.USER)
    state.mirror(_messages(word))
    while world.lib.curate(state, _faithful, gate=gate, budget=budget).evicted:
        pass
    if world.onion is not None:
        world.onion.save(cid, state, generation=state.generation)
    return cid, state


def _rows(world, cid):
    """How many rows of ``cid`` each table of the onion's file with a ``conversation`` column holds -- every one the
    file has, found in the file, not the store's own list of them -- the tombstones apart."""
    with closing(sqlite3.connect(str(world.onion.path))) as conn:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        keyed = [table for table in tables if table != "onion_forgotten"
                 and "conversation" in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}]
        return {table: conn.execute(f"SELECT COUNT(*) FROM {table} WHERE conversation = ?", (cid,)).fetchone()[0]
                for table in keyed}


def _file_bytes(world):
    return b"".join(p.read_bytes() for p in world.data.glob("onion.db*") if p.is_file())


def test_fo1_a_deleted_conversation_leaves_every_table_of_the_onion_store(tmp_path):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        kept, _ = _conversation(world, "Kilimanjaro")
        before = _rows(world, kept)
        assert sum(_rows(world, gone).values()) > 0 and before["onion_cursor"] == 1, "control: both were stored"
        assert world.manager.delete_conversation(gone) is True
        left = _rows(world, gone)
        assert sum(left.values()) == 0, f"the onion's store kept the deleted conversation: {left}"
        assert _rows(world, kept) == before, "another conversation's rows moved"
        assert world.onion.conversations() == [kept]
        assert world.manager.get_conversation(gone) is None and world.forgot == [gone]
    finally:
        world.restore()


def test_fo2_the_store_file_no_longer_holds_its_words(tmp_path):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        _conversation(world, "Kilimanjaro")
        held = _file_bytes(world)
        assert b"Zanzibar" in held and b"Kilimanjaro" in held, "control: the file holds both conversations' words"
        world.manager.delete_conversation(gone)
        held = _file_bytes(world)
        assert b"Zanzibar" not in held, (
            "the deleted conversation's words still wait in the onion's file, in its free pages or its log")
        assert b"Kilimanjaro" in held, "the other conversation's words went with it"
    finally:
        world.restore()


def test_fo3_the_tombstone_holds_its_id_and_the_time_nothing_else(tmp_path):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        world.manager.delete_conversation(gone)
        with sqlite3.connect(str(world.onion.path)) as conn:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(onion_forgotten)").fetchall()]
            rows = conn.execute("SELECT * FROM onion_forgotten").fetchall()
        assert columns == ["conversation", "forgotten_at"], columns
        assert len(rows) == 1 and rows[0][0] == gone, rows
        assert datetime.fromisoformat(rows[0][1]).year >= 2026, f"a time, not a word: {rows[0][1]}"
    finally:
        world.restore()


def test_fo4_no_writer_brings_it_back(tmp_path):
    world = _world(tmp_path)
    try:
        gone, held = _conversation(world, "Zanzibar")
        read_at = held.generation
        assert read_at >= 1, "control: the state was saved"
        world.manager.delete_conversation(gone)
        smod = world.loaded[STORE]
        other = smod.OnionStore(world.onion.path, connect=_plain, require_encryption=False)
        fresh = world.loaded[LIB].OnionState()
        for store, state, generation in ((world.onion, fresh, 0), (other, fresh, 0), (other, held, read_at),
                                         (world.onion, held, None)):
            refused = None
            try:
                store.save(gone, state, generation=generation)
            except smod.OnionStateForgotten as exc:
                refused = exc
            assert refused is not None, f"a save at generation {generation!r} wrote a forgotten conversation back"
        refused = None
        try:
            world.lib._save_state(gone, held, None)
        except smod.OnionStoreError as exc:
            refused = exc
        assert isinstance(refused, smod.OnionStateForgotten), (
            f"the librarian's save of a deleted conversation must be refused as forgotten, not as another process's "
            f"save: {refused!r}")
        assert sum(_rows(world, gone).values()) == 0, "a refused save wrote rows"
    finally:
        world.restore()


def test_fo5_nothing_of_it_is_served(tmp_path):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        kept, _ = _conversation(world, "Kilimanjaro")
        world.manager.delete_conversation(gone)
        world.lib._states.pop(kept)
        assert world.lib.peek_state(kept) is not None, (
            "control: a peek of a conversation the process let go asks the store")
        assert world.lib.peek_state(gone) is None, "the onion still serves a deleted conversation"
        opened = None
        try:
            world.lib.open_onion(gone)
        except world.lib.LibrarianError as exc:
            opened = str(exc)
        assert opened and "nothing persisted" in opened, f"the deleted conversation was opened again: {opened}"
        assert world.onion.load_snapshot(gone)[0] is None, "the store still loads the deleted conversation"
        racing = world.lib.state_for(gone)
        assert racing.flesh.turns() == [] and not racing.core.all(), "a turn got the deleted conversation's state"
        assert gone not in world.lib._states, "a turn racing the deletion installed a state that would be kept"
        assert world.lib.peek_state(gone) is None
    finally:
        world.restore()


def test_fo6_nothing_of_it_is_held_in_the_process(tmp_path):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        lib = world.lib
        lib._watermark[gone] = 3
        lib._counters[gone] = ("a counter", "a model")
        lib._said_refused(gone, RuntimeError("a store locked a moment"))
        assert gone in lib._states and any(said[0] == gone for said in lib._refused), "control: held"
        world.manager.delete_conversation(gone)
        for holder in ("_states", "_watermark", "_counters"):
            assert gone not in getattr(lib, holder), f"{holder} still holds the deleted conversation"
        assert not any(said[0] == gone for said in lib._refused), "a refusal said of it is still held"
        # The onion on, and no store for a save to refuse: curating the deleted conversation must do nothing at all.
        lib.onion_store = lambda config=None: None
        dispatched = []
        on = dataclasses.replace(lib.load_config(), enabled=True)
        assert lib.maybe_curate(gone, _messages("Zanzibar"), config=on, runner=dispatched.append) is False
        assert gone not in lib._states and gone not in lib._watermark and dispatched == [], (
            "curating a deleted conversation marked it again or sent it to be curated")
        assert lib.maybe_curate("another", _messages("Kilimanjaro"), config=on, runner=dispatched.append) is True, (
            "control: a conversation that was not deleted is curated")
    finally:
        world.restore()


# A peer's deletion made after every write of these contracts, so the last writer's rule applies it.
_LATER = "2099-01-01T00:00:00"


def _refusing_librarian():
    lib = types.ModuleType(LIB)

    def forget_conversation(conversation_id, config=None):
        raise RuntimeError("the onion store at onion.db is locked by another process")

    lib.forget_conversation = forget_conversation
    return lib


def _routes(manager, librarian):
    """The conversations route in a window of its own, its deps the store under test; the librarian the deletion asks,
    seeded again here, since the deletion imports it when it runs."""
    deps = types.ModuleType("opti_oignon.api.deps")
    deps.CONVERSATION_AVAILABLE = True
    deps.conversation_manager = manager
    loaded, restore = isolate(
        targets={"opti_oignon.api.schemas": source("api", "schemas.py"),
                 "opti_oignon.api.routes_conversations": source("api", "routes_conversations.py")},
        seeded={"opti_oignon.api.deps": deps, LIB: librarian}, packages=("opti_oignon.api", "opti_oignon.memory"))
    return loaded["opti_oignon.api.routes_conversations"], restore


def test_fo7_a_deletion_the_onion_cannot_forget_is_refused_by_name(tmp_path):
    refusing = _refusing_librarian()
    world = _world(tmp_path, librarian=refusing)
    try:
        cid = world.manager.create_conversation(title="kept").id
        world.manager.add_message(cid, "user", "Zanzibar is where the user lives.")
        refused = None
        try:
            world.manager.delete_conversation(cid)
        except world.conv.ConversationNotForgotten as exc:
            refused = str(exc)
        assert refused and "could not be forgotten" in refused and "locked by another process" in refused, refused
        kept = world.manager.get_conversation(cid)
        assert kept is not None and [m.content for m in world.manager.get_messages(cid)] == [
            "Zanzibar is where the user lives."], "a refused deletion deleted"
        assert world.forgot == [], "a refused deletion forgot the conversation's runs"
        routes, restore = _routes(world.manager, refusing)
        try:
            answered = None
            try:
                routes.delete_conversation(cid)
            except Exception as exc:
                answered = exc
            assert getattr(answered, "status_code", None) == 409, answered
            detail = str(getattr(answered, "detail", ""))
            assert "could not be forgotten (RuntimeError)" in detail and "locked by another process" not in detail, (
                f"the answer must name the reason, never carry the message (a path on this machine): {detail}")
        finally:
            restore()
        assert world.manager.get_conversation(cid) is not None
    finally:
        world.restore()


def test_fo8_a_peers_deletion_forgets_it_too(tmp_path):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        assert world.manager.apply_synced_conversation({"conversation": {"id": gone}}, deleted=True,
                                                       updated_at=_LATER) is True
        assert sum(_rows(world, gone).values()) == 0 and world.lib.peek_state(gone) is None, (
            "a peer's deletion left the onion's copy")
    finally:
        world.restore()

    world = _world(tmp_path / "refusing", librarian=_refusing_librarian())
    try:
        cid = world.manager.create_conversation(title="kept").id
        assert world.manager.apply_synced_conversation({"conversation": {"id": cid}}, deleted=True,
                                                       updated_at=_LATER) is False
        assert world.manager.get_conversation(cid) is not None, "a peer's deletion the onion refused deleted"
        assert world.forgot == []
    finally:
        world.restore()


def test_fo9_with_no_store_a_deletion_lets_go_of_the_process_state_and_creates_no_file(tmp_path):
    world = _world(tmp_path, store=False)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        before = sorted(p.name for p in world.data.iterdir())
        assert gone in world.lib._states, "control: the process holds it"
        assert world.manager.delete_conversation(gone) is True
        assert gone not in world.lib._states and world.lib.peek_state(gone) is None
        assert sorted(p.name for p in world.data.iterdir()) == before, "a deletion with no store created a file"
    finally:
        world.restore()


def test_fo10_a_file_written_before_the_tombstones_opens_at_schema_5_and_forgets_one(tmp_path):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        kept, _ = _conversation(world, "Kilimanjaro")
        # Schema 4 is schema 5 without its one table: built here by taking it away, every other table as it was.
        with sqlite3.connect(str(world.onion.path)) as conn:
            conn.execute("DROP TABLE onion_forgotten")
            conn.execute("PRAGMA user_version = 4")
        smod = world.loaded[STORE]
        opened = smod.OnionStore(world.onion.path, connect=_plain, require_encryption=False)
        with sqlite3.connect(str(world.onion.path)) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert version == smod.SCHEMA_VERSION == 5 and "onion_forgotten" in tables, (version, sorted(tables))
        with closing(sqlite3.connect(str(world.onion.path))) as conn:
            assert conn.execute("SELECT COUNT(*) FROM onion_scrub").fetchone()[0] == 0, (
                "the file written before was marked, and not rewritten though nothing held it")
        assert sorted(opened.conversations()) == sorted([gone, kept]), "the file's conversations were not kept"
        world.lib.onion_store = lambda config=None: opened
        world.manager.delete_conversation(gone)
        assert opened.conversations() == [kept] and opened.load_snapshot(gone)[0] is None
    finally:
        world.restore()


def test_fo11_a_tree_without_the_memory_package_still_deletes_and_says_it(tmp_path, caplog):
    world = _world(tmp_path, memory=False)
    try:
        cid = world.manager.create_conversation(title="gone").id
        with caplog.at_level(logging.WARNING, logger=CONV):
            assert world.manager.delete_conversation(cid) is True
        assert world.manager.get_conversation(cid) is None and world.forgot == [cid]
        assert any("could not be asked to forget" in record.getMessage() for record in caplog.records), (
            "a deletion the onion could not be asked about went unsaid")
    finally:
        world.restore()


def _older(world, words):
    """The store's file as an older build left it: the fourth schema, and in its free pages a deleted row's ``words``,
    written and deleted on a connection that does not overwrite. Built by taking away what the fifth schema adds."""
    with closing(_plain(world.onion.path)) as conn:
        conn.execute("INSERT INTO onion_cellar (conversation, seq, key, span) VALUES (?, ?, ?, ?)",
                     ("older", 0, "k", json.dumps({"text": words * 64})))
        conn.commit()
        conn.execute("DELETE FROM onion_cellar WHERE conversation = ?", ("older",))
        conn.execute("DROP TABLE onion_forgotten")
        conn.execute("DROP TABLE onion_scrub")
        conn.execute("PRAGMA user_version = 4")
        conn.commit()


def test_fo12_a_file_written_before_is_rewritten_once_and_its_free_pages_keep_nothing(tmp_path, caplog):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        _older(world, "Mombasa ")
        assert b"Mombasa" in _file_bytes(world), "control: the deleted row's words wait in the file's free pages"
        smod = world.loaded[STORE]
        # One store, opened while another process holds the file, and asked to forget once it let go.
        holding = {"on": True}

        def connect(path):
            return _HeldBack(_plain(path)) if holding["on"] else _plain(path)

        with caplog.at_level(logging.WARNING, logger=STORE):
            held = smod.OnionStore(world.onion.path, connect=connect, require_encryption=False)
        assert any("not rewritten yet" in r.getMessage() for r in caplog.records), "a rewrite held back was not said"
        assert b"Mombasa" in _file_bytes(world), "control: the rewrite was held back"
        with closing(sqlite3.connect(str(world.onion.path))) as conn:
            assert conn.execute("SELECT COUNT(*) FROM onion_scrub").fetchone()[0] == 1, "the mark was lost"
        assert held.conversations() == [gone]
        holding["on"] = False
        world.lib.onion_store = lambda config=None: held
        world.manager.delete_conversation(gone)
        assert b"Mombasa" not in _file_bytes(world), (
            "the free pages of a file written before kept a deleted row's words")
        with closing(sqlite3.connect(str(world.onion.path))) as conn:
            assert conn.execute("SELECT COUNT(*) FROM onion_scrub").fetchone()[0] == 0, "rewritten, and still marked"
    finally:
        world.restore()


def test_fo13_a_forgetting_another_connection_holds_back_is_refused_and_forgets_nothing(tmp_path):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        reader = sqlite3.connect(str(world.onion.path), isolation_level=None)
        try:
            reader.execute("BEGIN")
            reader.execute("SELECT COUNT(*) FROM onion_flesh").fetchone()
            refused = None
            try:
                world.manager.delete_conversation(gone)
            except world.conv.ConversationNotForgotten as exc:
                refused = exc
            assert refused is not None and refused.reason == "OperationalError", (
                f"a forgetting another connection held back past its wait was not refused by name: {refused!r}")
            assert world.manager.get_conversation(gone) is not None and world.forgot == []
            assert sum(_rows(world, gone).values()) > 0, "a refused forgetting deleted rows"
            assert gone in world.lib._states and gone not in world.lib._forgotten
        finally:
            reader.execute("ROLLBACK")
            reader.close()
        assert world.manager.delete_conversation(gone) is True, "once the reader let go, the deletion went through"
        assert sum(_rows(world, gone).values()) == 0
    finally:
        world.restore()


def test_fo14_the_store_refusing_to_forget_refuses_the_deletion_through_the_real_librarian(tmp_path):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        smod = world.loaded[STORE]

        def locked(conversation_id):
            raise smod.OnionStoreError("the onion store is locked by another process")

        world.onion.forget = locked
        refused = None
        try:
            world.manager.delete_conversation(gone)
        except world.conv.ConversationNotForgotten as exc:
            refused = exc
        assert refused is not None and refused.reason == "OnionStoreError", refused
        assert world.manager.get_conversation(gone) is not None and world.forgot == [], "a refused deletion deleted"
        assert gone in world.lib._states and gone not in world.lib._forgotten, (
            "a refused deletion let go of the onion of a conversation that is still there")
    finally:
        world.restore()


def test_fo15_a_state_read_before_the_deletion_is_never_installed_after_it(tmp_path):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        lib = world.lib
        lib._states.pop(gone)
        reader = lib._load_state

        def read_then_deleted(conversation_id, config):
            loaded = reader(conversation_id, config)
            # The deletion lands between the read and the install.
            world.manager.delete_conversation(conversation_id)
            return loaded

        lib._load_state = read_then_deleted
        assert lib.peek_state(gone) is None and gone not in lib._states, (
            "a peek installed a state read before the deletion")
    finally:
        world.restore()

    world = _world(tmp_path / "again")
    try:
        gone, held = _conversation(world, "Zanzibar")
        lib = world.lib
        # Another process saved the conversation since: the generation this state was read at moved.
        with closing(sqlite3.connect(str(world.onion.path))) as conn:
            conn.execute("UPDATE onion_cursor SET generation = generation + 1 WHERE conversation = ?", (gone,))
            conn.commit()
        reader = lib._load_state

        def read_then_deleted(conversation_id, config):
            loaded = reader(conversation_id, config)
            world.manager.delete_conversation(conversation_id)
            return loaded

        lib._load_state = read_then_deleted
        assert lib._current(gone, held, lib.load_config()) is None and gone not in lib._states, (
            "a state read again was installed after the deletion")
    finally:
        world.restore()


def test_fo16_a_store_that_cannot_be_looked_into_refuses_the_deletion(tmp_path):
    world = _world(tmp_path, store=False)
    try:
        lib = world.lib
        lib.onion_store = world.real_store
        cases = (("a plaintext connection where encryption is required", world.data / "onion.db", True,
                  "PlaintextRefused"),
                 ("a volume not mounted", world.data / "volume" / "onion.db", False, "OperationalError"))
        for name, path, encrypted, reason in cases:
            named = dataclasses.replace(lib.load_config(), persist_path=str(path), require_encryption=encrypted)
            lib.load_config = lambda *args, named=named, **kwargs: named
            cid = world.manager.create_conversation(title=name).id
            refused = None
            try:
                world.manager.delete_conversation(cid)
            except world.conv.ConversationNotForgotten as exc:
                refused = exc
            assert refused is not None and refused.reason == reason, (
                f"{name}: a store that cannot be looked into may hold a copy; the deletion must be refused: "
                f"{refused!r}")
            assert world.manager.get_conversation(cid) is not None, f"{name}: a refused deletion deleted"
            assert not path.exists(), f"{name}: a store file was left behind"
    finally:
        world.restore()


def test_fo17_a_wipe_stops_at_a_deletion_refused_and_says_how_many_are_left(tmp_path, caplog):
    asked = []
    lib = types.ModuleType(LIB)

    def forget_conversation(conversation_id, config=None):
        asked.append(conversation_id)
        raise RuntimeError("the onion store is held by another process")

    lib.forget_conversation = forget_conversation
    world = _world(tmp_path, librarian=lib, wipe=True)
    try:
        ids = [world.manager.create_conversation(title=f"c{i}").id for i in range(3)]
        wiper = world.loaded["opti_oignon.conversation_wipe"].ConversationWipeManager()
        wiper._get_conversation_manager = lambda: world.manager
        with caplog.at_level(logging.WARNING, logger="opti_oignon.conversation_wipe"):
            purged = wiper._purge_disk_all()
        assert purged == set() and len(asked) == 1, (
            f"the store refuses every deletion alike: asking again waits as long for each ({len(asked)} asked)")
        assert all(world.manager.get_conversation(i) is not None for i in ids)
        assert any("stopped, 3 conversation(s) left" in r.getMessage() and "RuntimeError" in r.getMessage()
                   for r in caplog.records), "the wipe stopped without saying how many are left, or why"
    finally:
        world.restore()


def test_fo18_a_memory_package_that_is_there_and_breaks_refuses_the_deletion(tmp_path):
    for name, breaking in (("raises", RuntimeError("its __init__ breaks")),
                           ("lacks", ModuleNotFoundError("No module named 'numpy'", name="numpy"))):
        world = _world(tmp_path / name, broken_memory=breaking)
        try:
            cid = world.manager.create_conversation(title="kept").id
            refused = None
            try:
                world.manager.delete_conversation(cid)
            except world.conv.ConversationNotForgotten as exc:
                refused = exc
            assert refused is not None and refused.reason == type(breaking).__name__, (
                f"{name}: a memory package that breaks may hold a store: the deletion must be refused: {refused!r}")
            assert world.manager.get_conversation(cid) is not None and world.forgot == []
        finally:
            world.restore()


def test_fo19_a_save_that_takes_a_turn_back_leaves_none_of_its_words_in_the_file(tmp_path):
    world = _world(tmp_path)
    try:
        cid, state = _conversation(world, "Zanzibar")
        moved = {"role": "user", "content": "Ushuaia " * 40 + "is where the user moves next year."}
        messages = _messages("Zanzibar") + [moved]
        state.mirror(messages)
        world.onion.save(cid, state, generation=state.generation)
        assert b"Ushuaia" in _file_bytes(world), "control: the turn was saved"
        state.mirror(messages[:-1])
        assert all("Ushuaia" not in str(turn.get("text", "")) for turn in state.flesh.turns()), (
            "control: the mirror took the turn back")
        world.onion.save(cid, state, generation=state.generation)
        assert b"Ushuaia" not in _file_bytes(world), (
            "a save that replaced the conversation's rows left a turn it took back in the file's free pages")
    finally:
        world.restore()
