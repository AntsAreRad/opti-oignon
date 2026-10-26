#!/usr/bin/env python3
"""Contracts for forgetting: a deletion forgets what every store keyed by the conversation holds of it.

After the onion (its own suite), a deletion left the conversation's branches and their turns, the facts proposed in
it, the feedback given in it, the questions and answers the semantic cache kept of it, its links to projects, and
the artifacts, tool calls and coding session the process held of it; the conversation store left its messages' words
in its free pages. A deletion now walks one registry (``conversation._FORGETTERS``) after the onion and before the
conversation store's own write; a store that keeps the conversation's words and cannot forget refuses the deletion
by name; a store that forgot takes nothing new about the conversation in the process; every connection of a store
overwrites what it frees; a census holds the registry complete against every table of the package keyed by a
conversation.

  * FE1 -- ITS BRANCHES AND THEIR TURNS ARE FORGOTTEN, their words gone from the file and its log; another's
    stay.
  * FE2 -- THE FACTS PROPOSED IN IT ARE FORGOTTEN, whatever their decision; another's stay.
  * FE3 -- THE FEEDBACK GIVEN IN IT IS FORGOTTEN; another's stays.
  * FE4 -- WHAT THE SEMANTIC CACHE KEPT OF IT IS FORGOTTEN, under the global scope too; another's stays.
  * FE5 -- ITS LINKS TO EVERY PROJECT ARE FORGOTTEN, the outputs it gave kept by the project, their source cleared.
  * FE6 -- WHAT THE PROCESS HOLDS OF IT IN MEMORY IS LET GO: its artifacts, its tool calls, its coding session; a
    module the process never loaded is not loaded to be asked.
  * FE7 -- ITS MESSAGES' WORDS ARE GONE FROM THE CONVERSATION STORE'S FILE.
  * FE8 -- A STORE THAT KEEPS ITS WORDS AND CANNOT FORGET REFUSES THE DELETION BY NAME, naming what was forgotten
    before it; the conversation stays; once the store can, the deletion goes through.
  * FE9 -- A STORE OF IDS ONLY THAT CANNOT FORGET IS SAID, AND THE DELETION GOES THROUGH.
  * FE10 -- A PEER'S DELETION FORGETS THEM TOO.
  * FE11 -- A STORE THIS TREE LACKS IS SAID AND PASSED; ONE THAT IS THERE WITHOUT ITS FORGETTING, OR WITHOUT A
    DEPENDENCY OF ITS OWN, REFUSES.
  * FE12 -- THE CENSUS: every table of the package keyed by a conversation -- a conversation column, or records of
    any kind under (kind, record_id), f-strings read too -- is forgotten at a deletion, or named with the reason it
    is not; the census finds the tables it must.
  * FE13 -- AN OLDER SEMANTIC CACHE DROPS WHAT NO DELETION COULD REACH: the entries it cached with no
    conversation, or an older build caches into a newer file, go with the first deletion, overwritten -- the file
    given the column by an ordinary open, or still without it.
  * FE14 -- A DELETION EMPTIES THE WRITE-AHEAD LOG ANOTHER CONNECTION KEEPS, whose older frames hold the words; a
    read in progress keeps it, said, and the next deletion empties it.
  * FE15 -- A PEER'S CONVERSATION AND ITS DELETION, APPLIED, NEVER PUBLISH AGAIN (supersedes syn01's
    test_conversation_apply_does_not_republish, whose window reaches the stores only through modules another test
    cached).
  * FE16 -- EVERY WRITE OF A STORE OVERWRITES WHAT IT FREES: a deleted conversation's words a lookup's rewrite or an
    invalidation freed before the deletion are gone from the file too.
  * FE17 -- A STORE THAT FORGOT A CONVERSATION TAKES NOTHING NEW ABOUT IT in the process -- a cache put, a proposal
    (the agent told so), a branch, a branch turn, feedback --, a write already on its way when the deletion came
    included.
  * FE18 -- THE ONION IS ASKED FIRST: its refusal leaves every other store as it was, a peer's deletion too; a
    conversation the store does not hold asks no store.
  * FE19 -- A DELETION CREATES NO STORE it finds absent.
  * FE20 -- UNDER A DRIVER WHOSE ERRORS ARE NOT SQLITE3'S (the cipher's, sqlcipher3), the cache's schema step never
    refuses a deletion, and a log that cannot be emptied is said while the deletion stands.
  * FE21 -- THE DELETION ROUTE STOPS THE CONVERSATION'S LIVE TURNS FIRST, before the deletion; chat routes not
    loaded, or a stop that fails, never fail it.
  * FE22 -- A CONVERSATION THAT STAYS OR COMES BACK IS TAKEN AGAIN BY EVERY STORE: a deletion refused at any store,
    the conversation store's own write failing, a peer's later record bringing it back.
  * FE23 -- THE CONVERSATION STORE IS LOCKED FIRST AND THE PROCESS LETS GO ONCE COMMITTED: a store another
    connection writes to refuses the deletion before anything is forgotten; the process's memory of it goes after
    the commit.

Local-only (the public distribution ships no tests). conversation.py and the six store modules loaded from source
in one shared isolation window, every store in tmp -- the two whose file lives beside their source loaded from a
copy of it there --, on connections that do not overwrite deleted pages by default, as the cipher's driver does
not (sqlcipher3: secure_delete 0, probed 26 October); the librarian and the run journal recorders.
"""

import ast
import logging
import re
import sqlite3
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_census import modules as package_modules  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

CONV = "opti_oignon.conversation"
BUDGET_S = {
    "test_fe1_its_branches_and_their_turns_are_forgotten": 3.0,
    "test_fe2_the_facts_proposed_in_it_are_forgotten": 3.0,
    "test_fe3_the_feedback_given_in_it_is_forgotten": 3.0,
    "test_fe4_what_the_semantic_cache_kept_of_it_is_forgotten": 3.0,
    "test_fe5_its_links_to_every_project_are_forgotten_and_the_outputs_kept": 3.0,
    "test_fe6_what_the_process_holds_of_it_in_memory_is_let_go": 3.0,
    "test_fe7_its_messages_words_are_gone_from_the_conversation_stores_file": 3.0,
    "test_fe8_a_store_that_keeps_its_words_and_cannot_forget_refuses_by_name": 3.0,
    "test_fe9_a_store_of_ids_only_that_cannot_forget_is_said_and_the_deletion_goes_on": 3.0,
    "test_fe10_a_peers_deletion_forgets_them_too": 3.0,
    "test_fe11_a_store_this_tree_lacks_is_passed_one_without_its_forgetting_refuses": 3.0,
    "test_fe12_every_table_keyed_by_a_conversation_is_forgotten_or_named": 3.0,
    "test_fe13_an_older_semantic_cache_drops_what_no_deletion_could_reach": 3.0,
    "test_fe14_a_deletion_empties_the_log_another_connection_keeps": 3.0,
    "test_fe15_a_peers_conversation_and_its_deletion_applied_never_publish_again": 3.0,
    "test_fe16_every_write_of_a_store_overwrites_what_it_frees": 3.0,
    "test_fe17_a_store_that_forgot_a_conversation_takes_nothing_new_about_it": 5.0,
    "test_fe18_the_onion_is_asked_first_and_its_refusal_leaves_every_store": 3.0,
    "test_fe19_a_deletion_creates_no_store": 3.0,
    "test_fe20_a_driver_whose_errors_are_not_sqlite3s_never_turns_a_deletion": 3.0,
    "test_fe21_the_deletion_route_stops_the_conversations_live_turns_first": 3.0,
    "test_fe22_a_conversation_that_stays_or_comes_back_is_taken_again_by_every_store": 8.0,
    "test_fe23_the_store_is_locked_first_and_the_process_lets_go_once_committed": 3.0,
}
# The registry's modules, and the two whose file lives beside their source.
_STORES = ("conversation_branches", "pending_writes", "feedback", "semantic_cache", "projects", "artifacts")
_BESIDE = ("feedback", "projects")
# A peer's deletion made after every write of these contracts.
_LATER = "2099-01-01T00:00:00"


def _plain(path, *, wait=0.3, check_same_thread=False, **_options):
    """A plain connection that, like many builds, does not overwrite deleted pages unless asked to; it waits
    ``wait`` seconds for a lock (production's stores wait 5, db_utils.safe_connect)."""
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=wait)
    conn.execute("PRAGMA secure_delete = OFF")
    return conn


class _CipherError(Exception):
    """The cipher driver's errors, which are not sqlite3's (sqlcipher3.dbapi2.Error, probed 26 October)."""


class _CipherOperational(_CipherError):
    pass


class _CipherLike:
    """A connection whose errors, like sqlcipher3's, are not sqlite3's: what an ``except sqlite3.Error`` misses. A
    statement holding ``refuse`` raises one."""

    def __init__(self, conn, refuse=None):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_refuse", refuse)

    def execute(self, sql, *args):
        if self._refuse and self._refuse in sql:
            raise _CipherOperational(f"refused: {sql}")
        try:
            return self._conn.execute(sql, *args)
        except sqlite3.OperationalError as exc:
            raise _CipherOperational(str(exc)) from None
        except sqlite3.Error as exc:
            raise _CipherError(str(exc)) from None

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        setattr(self._conn, name, value)


def _world(tmp_path, *, lacking=(), seeded_stores=None, wait=0.3, memory=None, refuse=None):
    """conversation.py and the store modules over ``tmp_path``: the registry's modules but ``lacking``, and
    ``seeded_stores`` (module name -> stand-in) in the place of some; ``memory`` (module name -> stand-in) the
    process-memory modules loaded; the librarian and the run journal record. With ``refuse``, every connection is
    one whose errors are not sqlite3's, refusing the statements holding it."""
    data = tmp_path / "data"
    data.mkdir(parents=True)
    beside = tmp_path / "beside"
    beside.mkdir()
    def connect(db_path, **options):
        conn = _plain(db_path, wait=wait, **options)
        return _CipherLike(conn, refuse) if refuse else conn

    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = connect
    # The encryption layer's plain path, as production opens a store with no key (db_encryption.py): no journal mode
    # set, nothing overwritten unless the store asks.
    enc = types.ModuleType("opti_oignon.db_encryption")
    enc.get_encrypted_connection = connect
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = data
    asked = SimpleNamespace(onion=[], runs=[])
    librarian = types.ModuleType("opti_oignon.memory.librarian")
    librarian.forget_conversation = lambda conversation_id, config=None: asked.onion.append(conversation_id) or 0
    journal = types.ModuleType("opti_oignon.run_journal")
    journal.forget_conversation = asked.runs.append
    targets = {CONV: source("conversation.py")}
    seeded = {"opti_oignon.db_utils": db, "opti_oignon.db_encryption": enc, "opti_oignon.config": cfg,
              "opti_oignon.run_journal": journal, "opti_oignon.memory.librarian": librarian}
    for name, module in (memory or {}).items():
        seeded[f"opti_oignon.{name}"] = module
    for name in _STORES:
        if name in lacking:
            continue
        if seeded_stores and name in seeded_stores:
            seeded[f"opti_oignon.{name}"] = seeded_stores[name]
            continue
        if name in _BESIDE:
            copy = beside / f"{name}.py"
            copy.write_text(Path(source(f"{name}.py")).read_text(encoding="utf-8"), encoding="utf-8")
            targets[f"opti_oignon.{name}"] = copy
        else:
            targets[f"opti_oignon.{name}"] = source(f"{name}.py")
    loaded, restore = isolate(targets=targets, seeded=seeded, packages=("opti_oignon.memory",))
    try:
        conv = loaded[CONV]
        manager = conv.ConversationManager(db_path=data / "conversations.db", publish_to_sync=False)
    except BaseException:
        restore()
        raise
    return SimpleNamespace(loaded=loaded, restore=restore, conv=conv, manager=manager, data=data, beside=beside,
                           asked=asked, librarian=librarian, mod=lambda name: loaded[f"opti_oignon.{name}"])


def _conversation(world, word):
    cid = world.manager.create_conversation(title=f"about {word}").id
    first = world.manager.add_message(cid, "user", f"{word} is where the user grew up.")
    world.manager.add_message(cid, "assistant", f"Noted: {word}.")
    return cid, first


def _bytes(path):
    return b"".join(p.read_bytes() for p in path.parent.glob(path.name + "*") if p.is_file())


def _keeper(path):
    """Another process's connection to a store, idle: the store's write-ahead log outlives the store's own."""
    conn = _plain(path)
    conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchall()
    return conn


def test_fe1_its_branches_and_their_turns_are_forgotten(tmp_path):
    world = _world(tmp_path)
    try:
        branches = world.mod("conversation_branches")
        gone, first = _conversation(world, "Zanzibar")
        kept, kept_first = _conversation(world, "Kilimanjaro")
        for cid, message, word in ((gone, first, "Zanzibar"), (kept, kept_first, "Kilimanjaro")):
            branch = branches.branch_manager.fork(cid, fork_message_id=message.id, name="another way")
            branches.branch_manager.add_branch_message(branch.branch_id, cid, "user", f"{word} again, " * 20)
        path = world.data / "branches.db"
        assert b"Zanzibar again" in _bytes(path), "control: the branch's turn was written"
        keeper = _keeper(path)
        assert world.manager.delete_conversation(gone) is True
        with sqlite3.connect(str(path)) as conn:
            left = [conn.execute(f"SELECT COUNT(*) FROM {t} WHERE conversation_id = ?", (cid,)).fetchone()[0]
                    for t in ("branches", "branch_messages") for cid in (gone, kept)]
        assert left[0] == 0 and left[2] == 0, f"the deleted conversation's branches stayed: {left}"
        assert left[1] >= 1 and left[3] >= 1, f"another conversation's branches went with it: {left}"
        assert b"Zanzibar again" not in _bytes(path), "a deleted branch turn's words stayed in the file or its log"
        keeper.close()
    finally:
        world.restore()


def test_fe2_the_facts_proposed_in_it_are_forgotten(tmp_path):
    world = _world(tmp_path)
    try:
        queue = world.mod("pending_writes").get_pending_store()
        gone, _ = _conversation(world, "Zanzibar")
        kept, _ = _conversation(world, "Kilimanjaro")
        proposed = [queue.propose("memory", "add", {"text": f"The user grew up in {word}."}, {"origin": "extraction"},
                                  conversation_id=cid)[0] for cid, word in ((gone, "Zanzibar"), (gone, "Mombasa"),
                                                                            (kept, "Kilimanjaro"))]
        assert queue.decline(proposed[1].id), "control: one proposal of the deleted conversation was declined"
        path = Path(queue.db_path)
        assert b"Mombasa" in _bytes(path), "control: the proposals were written"
        assert world.manager.delete_conversation(gone) is True
        with sqlite3.connect(str(path)) as conn:
            counts = {cid: conn.execute("SELECT COUNT(*) FROM pending_writes WHERE conversation_id = ?",
                                        (cid,)).fetchone()[0] for cid in (gone, kept)}
        assert counts == {gone: 0, kept: 1}, (
            f"proposals of the deleted conversation stayed, or another's went: {counts}")
        held = _bytes(path)
        assert b"Mombasa" not in held and b"Zanzibar" not in held, "a forgotten proposal's words stayed in the file"
        assert b"Kilimanjaro" in held
    finally:
        world.restore()


def test_fe3_the_feedback_given_in_it_is_forgotten(tmp_path):
    world = _world(tmp_path)
    try:
        feedback = world.mod("feedback")
        gone, _ = _conversation(world, "Zanzibar")
        kept, _ = _conversation(world, "Kilimanjaro")
        for cid, word in ((gone, "Zanzibar"), (kept, "Kilimanjaro")):
            feedback.feedback_store.add_feedback(feedback.FeedbackEntry(conversation_id=cid, message_id="1",
                                                                        feedback_text=f"{word} was wrong"))
        path = Path(feedback.feedback_store._db_path)
        assert path == world.beside / "data" / "feedback.db", "control: the store is in tmp"
        assert b"Zanzibar was wrong" in _bytes(path), "control: the feedback was written"
        keeper = _keeper(path)
        assert world.manager.delete_conversation(gone) is True
        assert feedback.feedback_store.list_by_conversation(gone) == [], "the deleted conversation's feedback stayed"
        assert len(feedback.feedback_store.list_by_conversation(kept)) == 1
        assert b"Zanzibar was wrong" not in _bytes(path), "forgotten feedback's words stayed in the file or its log"
        keeper.close()
    finally:
        world.restore()


def test_fe4_what_the_semantic_cache_kept_of_it_is_forgotten(tmp_path):
    world = _world(tmp_path)
    try:
        cache = world.mod("semantic_cache").semantic_cache
        cache._config["enabled"] = True
        cache.embeddings_available = False
        gone, _ = _conversation(world, "Zanzibar")
        kept, _ = _conversation(world, "Kilimanjaro")
        for cid, word in ((gone, "Zanzibar"), (kept, "Kilimanjaro")):
            cache.put(f"Where did the user grow up, {word}?", f"In {word}.", model="m", conversation_id=cid)
        cache.put("What is a harbour?", "A sheltered water.", model="m")  # outside any conversation: no deletion's
        # The global scope caches with no conversation: each entry keeps the one it came from, the last to put it.
        cache._config["scope"] = "global"
        cache.put("What did the user dream of?", "Of the harbour of Zanzibar.", model="m", conversation_id=gone)
        cache.put("Which town, said again?", "The town of Kilimanjaro.", model="m", conversation_id=kept)
        cache.put("Which town, said again?", "The town of Zanzibar.", model="m", conversation_id=gone)
        cache.put("What did the user climb?", "The slopes of Kilimanjaro.", model="m", conversation_id=kept)
        path = Path(cache._db_path)
        assert b"Zanzibar" in _bytes(path), "control: the entries were written"
        assert world.manager.delete_conversation(gone) is True
        with sqlite3.connect(str(path)) as conn:
            responses = sorted(row[0] for row in conn.execute("SELECT response FROM cache_entries"))
        assert responses == ["A sheltered water.", "In Kilimanjaro.", "The slopes of Kilimanjaro."], (
            f"the cache kept the deleted conversation's entries, or lost another's: {responses}")
        assert b"Zanzibar" not in _bytes(path), "a forgotten entry's words stayed in the cache's file"
    finally:
        world.restore()


def test_fe5_its_links_to_every_project_are_forgotten_and_the_outputs_kept(tmp_path):
    world = _world(tmp_path)
    try:
        projects = world.mod("projects").project_store
        gone, _ = _conversation(world, "Zanzibar")
        kept, _ = _conversation(world, "Kilimanjaro")
        first, second = projects.create_project("one"), projects.create_project("two")
        for project in (first, second):
            projects.link_conversation(project.id, gone)
            projects.link_conversation(project.id, kept)
        output = projects.add_output(first.id, "notes.md", b"what the user decided", description="a summary",
                                     source_conversation_id=gone)
        assert world.manager.delete_conversation(gone) is True
        with sqlite3.connect(str(projects._db_path)) as conn:
            links = conn.execute("SELECT project_id, conversation_id FROM project_conversations").fetchall()
            sources = conn.execute("SELECT id, source_conversation_id FROM project_outputs").fetchall()
        assert sorted(links) == sorted([(first.id, kept), (second.id, kept)]), f"links left or lost: {links}"
        assert sources == [(output.id, "")], f"the output went, or still names the deleted conversation: {sources}"
    finally:
        world.restore()


def _recorder(name):
    """A process-memory module that records what it was asked to forget."""
    stub = types.ModuleType(f"opti_oignon.{name}")
    stub.asked = []
    stub.forget_conversation = lambda conversation_id: stub.asked.append(conversation_id) or 1
    return stub


def _function(module, name):
    """The syntax tree of the top-level function or method ``name`` of a package module (``Class.method``)."""
    tree = ast.parse((REPO / "opti_oignon" / module).read_text(encoding="utf-8"))
    owner, _, leaf = name.rpartition(".")
    scope = tree.body
    if owner:
        scope = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner).body
    return next(node for node in scope if isinstance(node, ast.FunctionDef) and node.name == leaf)


def _calls(node):
    return {ast.unparse(call.func) for call in ast.walk(node) if isinstance(call, ast.Call)}


def test_fe6_what_the_process_holds_of_it_in_memory_is_let_go(tmp_path, caplog):
    # The real forgetters, which the window stands in for: each asks its process's singleton, marks the
    # conversation, and says a sandbox it could not destroy.
    tools_forget = _function("agentic_executor.py", "forget_conversation")
    assert {"agentic_executor.clear_tool_history", "_FORGOTTEN.add"} <= _calls(tools_forget), _calls(tools_forget)
    assert "in _FORGOTTEN" in ast.unparse(_function("agentic_executor.py", "AgenticExecutor._record_tool_calls"))
    coding_forget = _function("chat_coding_agent.py", "forget_conversation")
    assert "chat_coding_manager.destroy_session" in _calls(coding_forget), _calls(coding_forget)
    assert any(isinstance(node, ast.Raise) for node in ast.walk(coding_forget)), "a failed sandbox is never said"
    tools, coding = _recorder("agentic_executor"), _recorder("chat_coding_agent")
    world = _world(tmp_path, memory={"agentic_executor": tools, "chat_coding_agent": coding})
    try:
        manager = world.mod("artifacts").artifact_manager
        gone, _ = _conversation(world, "Zanzibar")
        kept, _ = _conversation(world, "Kilimanjaro")
        manager._cache[gone] = [object(), object()]
        manager._cache[kept] = [object()]
        assert world.manager.delete_conversation(gone) is True
        assert gone not in manager._cache and len(manager._cache[kept]) == 1
        assert tools.asked == [gone], "the tool calls the process remembers of it were kept"
        assert coding.asked == [gone], "its coding session, and its sandbox, were kept"
    finally:
        world.restore()
    # A process that never loaded them holds nothing in their memory: they are not loaded to be asked.
    world = _world(tmp_path / "unloaded")
    try:
        gone, _ = _conversation(world, "Zanzibar")
        with caplog.at_level(logging.WARNING):
            assert world.manager.delete_conversation(gone) is True
        # Absent, or held off by the window (a key set to None): either way nothing was loaded to be asked.
        assert sys.modules.get("opti_oignon.agentic_executor") is None
        assert sys.modules.get("opti_oignon.chat_coding_agent") is None
        said = [r.getMessage() for r in caplog.records
                if "tool history" in r.getMessage() or "coding session" in r.getMessage()]
        assert said == [], f"a module the process never loaded was looked for: {said}"
    finally:
        world.restore()


def test_fe7_its_messages_words_are_gone_from_the_conversation_stores_file(tmp_path):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        _conversation(world, "Kilimanjaro")
        path = world.data / "conversations.db"
        assert b"Zanzibar" in _bytes(path), "control: the messages were written"
        assert world.manager.delete_conversation(gone) is True
        held = _bytes(path)
        assert b"Zanzibar" not in held, "a deleted message's words stayed in the conversation store's file"
        assert b"Kilimanjaro" in held
    finally:
        world.restore()


# The semantic cache's entries as block 185 created them, before each entry kept the conversation it came from.
_CACHE_185 = """
    CREATE TABLE IF NOT EXISTS cache_entries (
        query_hash TEXT NOT NULL,
        query_text TEXT NOT NULL,
        response TEXT NOT NULL,
        model TEXT NOT NULL DEFAULT '',
        conversation_id TEXT NOT NULL DEFAULT '',
        embedding BLOB,
        embedding_dim INTEGER DEFAULT 0,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at REAL NOT NULL,
        last_accessed REAL NOT NULL,
        hit_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (query_hash, conversation_id)
    )
"""


def test_fe13_an_older_semantic_cache_drops_what_no_deletion_could_reach(tmp_path):
    world = _world(tmp_path)
    try:
        cache = world.mod("semantic_cache").semantic_cache
        gone, _ = _conversation(world, "Zanzibar")
        kept, _ = _conversation(world, "Kilimanjaro")
        path = Path(cache._db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = _plain(path)
        conn.execute(_CACHE_185)
        for query, response, cid in (("Where did the user grow up?", "In Zanzibar.", gone),
                                     ("What did the user climb?", "Kilimanjaro.", kept),
                                     ("What did the user dream of?", "Of the harbour of Mombasa.", "")):
            conn.execute("INSERT INTO cache_entries (query_hash, query_text, response, conversation_id, created_at, "
                         "last_accessed) VALUES (?, ?, ?, ?, 1.0, 1.0)", (query, query, response, cid))
        conn.commit()
        conn.close()
        assert b"Mombasa" in _bytes(path), "control: the older file holds an entry with no conversation"
        # An ordinary open gives the older file the column, its entries' origin not known; the deletion after it
        # drops those with no conversation.
        cache._get_connection().close()
        assert world.manager.delete_conversation(gone) is True
        with sqlite3.connect(str(path)) as conn:
            responses = sorted(row[0] for row in conn.execute("SELECT response FROM cache_entries"))
        assert responses == ["Kilimanjaro."], (
            f"an older entry with no conversation stayed, or a conversation's was not forgotten: {responses}")
        held = _bytes(path)
        assert b"Mombasa" not in held and b"Zanzibar" not in held, "a dropped entry's words stayed in the file"
    finally:
        world.restore()

    # An older build caching into a newer file names no origin: the file's default says it is not known.
    world = _world(tmp_path / "newer")
    try:
        cache = world.mod("semantic_cache").semantic_cache
        gone, _ = _conversation(world, "Zanzibar")
        cache._get_connection().close()
        path = Path(cache._db_path)
        conn = _plain(path)
        conn.execute("INSERT INTO cache_entries (query_hash, query_text, response, conversation_id, created_at, "
                     "last_accessed) VALUES (?, ?, ?, '', 1.0, 1.0)", ("q", "q", "Under the floor in Pemba."))
        conn.commit()
        conn.close()
        assert world.manager.delete_conversation(gone) is True
        assert b"Pemba" not in _bytes(path), "an older build's entry with no conversation outlived the first deletion"
    finally:
        world.restore()

    # An older file still without the column, its schema step held back: every entry it holds with no conversation
    # goes, the schema left as it is.
    world = _world(tmp_path / "stuck")
    try:
        cache = world.mod("semantic_cache").semantic_cache
        gone, _ = _conversation(world, "Zanzibar")
        path = Path(cache._db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = _plain(path)
        conn.execute(_CACHE_185)
        conn.execute("INSERT INTO cache_entries (query_hash, query_text, response, conversation_id, created_at, "
                     "last_accessed) VALUES ('q', 'q', 'Of the harbour of Mombasa.', '', 1.0, 1.0)")
        conn.commit()
        conn.close()
        cache._init_db = _raising("the schema step is held back")
        assert world.manager.delete_conversation(gone) is True
        assert b"Mombasa" not in _bytes(path), "an entry no deletion could reach outlived a failed schema step"
    finally:
        world.restore()


def test_fe14_a_deletion_empties_the_log_another_connection_keeps(tmp_path, caplog):
    # Every lock waited for as production waits (5 s): a deletion never waits for a read in progress.
    world = _world(tmp_path, wait=5.0)
    try:
        branches = world.mod("conversation_branches").branch_manager
        feedback = world.mod("feedback")
        ids = {}
        for word in ("Zanzibar", "Mombasa"):
            cid, first = _conversation(world, word)
            branch = branches.fork(cid, fork_message_id=first.id, name="b")
            branches.add_branch_message(branch.branch_id, cid, "user", f"{word} again")
            feedback.feedback_store.add_feedback(feedback.FeedbackEntry(conversation_id=cid, message_id="1",
                                                                        feedback_text=f"{word} was wrong"))
            ids[word] = cid
        _conversation(world, "Kilimanjaro")
        paths = (world.data / "conversations.db", world.data / "branches.db", Path(feedback.feedback_store._db_path))
        keepers = [_keeper(path) for path in paths]
        readers = []
        for path in paths:
            reader = _plain(path)
            reader.execute("BEGIN")
            reader.execute("SELECT COUNT(*) FROM sqlite_master").fetchall()
            readers.append(reader)
        started = time.monotonic()
        with caplog.at_level(logging.WARNING):
            assert world.manager.delete_conversation(ids["Zanzibar"]) is True
        took = time.monotonic() - started
        said = [r.getMessage() for r in caplog.records
                if "a read in progress keeps the write-ahead log" in r.getMessage()]
        assert len(said) == 3, f"each store whose log a read kept says so: {said}"
        assert took < 2.5, f"a deletion waited {took:.1f} s for a read in progress"
        for reader in readers:
            reader.rollback()
            reader.close()
        assert all(b"Zanzibar" in _bytes(path) for path in paths), "control: the logs reads kept hold the words"
        assert world.manager.delete_conversation(ids["Mombasa"]) is True
        for path in paths:
            held = _bytes(path)
            assert b"Zanzibar" not in held and b"Mombasa" not in held, (
                f"{path.name} or its log kept a deleted conversation's words once no read was in progress")
        assert b"Kilimanjaro" in _bytes(paths[0])
        for keeper in keepers:
            keeper.close()
    finally:
        world.restore()


def _peer_payload(cid):
    """A peer's conversation as the sync package hands it over (syn01's payload)."""
    return {"user_id": "u", "conversation": {
        "id": cid, "title": "Hello", "created_at": "t0", "updated_at": "t1", "model": "m", "task_type": None,
        "preset": None, "metadata": {}, "messages": [
            {"role": "user", "content": "hi", "timestamp": "t0", "token_estimate": 1, "model": None, "metadata": {}}]}}


def test_fe15_a_peers_conversation_and_its_deletion_applied_never_publish_again(tmp_path):
    # Supersedes syn01's test_conversation_apply_does_not_republish, its assertions word for word: a peer's deletion
    # now asks every store, which syn01's window (a stand-in package with no path) reaches only through modules an
    # earlier test cached.
    world = _world(tmp_path)
    try:
        calls = []
        world.conv._sync_publish_conversation = lambda *a, **k: calls.append((a, k))
        manager = world.conv.ConversationManager(db_path=world.data / "c.db")
        assert manager.apply_synced_conversation(_peer_payload("c1")) is True
        assert manager.apply_synced_conversation({"conversation": {"id": "c1"}}, deleted=True) is True
        assert calls == []  # apply never re-publishes (no apply->write->publish echo)
        assert manager.get_conversation("c1") is None, "control: the peer's deletion was applied"
    finally:
        world.restore()


def test_fe16_every_write_of_a_store_overwrites_what_it_frees(tmp_path):
    # The deleted conversations' words that a store's own writes freed before the deletion: a lookup's rewrite of an
    # entry, an invalidation, a feedback or a branch deleted one by one, a proposal declined.
    world = _world(tmp_path)
    try:
        cache = world.mod("semantic_cache").semantic_cache
        cache._config["enabled"] = True
        cache.embeddings_available = False
        feedback = world.mod("feedback")
        branches = world.mod("conversation_branches").branch_manager
        queue = world.mod("pending_writes").get_pending_store()
        gone, first = _conversation(world, "Zanzibar")
        other, _ = _conversation(world, "Mombasa")
        cache.put("Where is the user's house?", "Under the floor in Pemba.", model="m", conversation_id=gone)
        for _ in range(5):
            assert cache.get("Where is the user's house?", conversation_id=gone, model="m") is not None
        cache.put("Where is the user's boat?", "At the quay of Chake.", model="m", conversation_id=other)
        cache.invalidate(other)
        entry = feedback.feedback_store.add_feedback(feedback.FeedbackEntry(
            conversation_id=gone, message_id="1", feedback_text="Wete was wrong"))
        assert feedback.feedback_store.delete_feedback(entry.feedback_id)
        branch = branches.fork(gone, fork_message_id=first.id, name="another way")
        branches.add_branch_message(branch.branch_id, gone, "user", "Mkoani again, " * 20)
        assert branches.delete_branch(branch.branch_id)
        proposal, _new = queue.propose("memory", "add", {"text": "The user grew up in Kengeja."},
                                       {"origin": "extraction"}, conversation_id=gone)
        assert queue.decline(proposal.id)
        projects = world.mod("projects").project_store
        project = projects.create_project("Kiwengwa plans", description="the house at Kiwengwa")
        assert projects.delete_project(project.id)
        assert world.manager.delete_conversation(gone) is True
        assert world.manager.delete_conversation(other) is True
        for path, word, freed in ((Path(cache._db_path), b"Pemba", "a lookup's rewrite"),
                                  (Path(cache._db_path), b"Chake", "an invalidation"),
                                  (Path(feedback.feedback_store._db_path), b"Wete", "a feedback deleted"),
                                  (world.data / "branches.db", b"Mkoani", "a branch deleted"),
                                  (Path(queue.db_path), b"Kengeja", "a proposal declined"),
                                  (Path(projects._db_path), b"Kiwengwa", "a project deleted")):
            assert word not in _bytes(path), f"{freed} left a deleted conversation's words in {path.name}"
    finally:
        world.restore()


def _raising(message="the store is held by another process"):
    def refuse(*_args, **_kwargs):
        raise RuntimeError(message)
    return refuse


def test_fe17_a_store_that_forgot_a_conversation_takes_nothing_new_about_it(tmp_path):
    world = _world(tmp_path)
    try:
        branches = world.mod("conversation_branches")
        pending = world.mod("pending_writes")
        feedback = world.mod("feedback")
        module = world.mod("semantic_cache")
        cache = module.semantic_cache
        cache._config["enabled"] = True
        cache.embeddings_available = False
        gone, first = _conversation(world, "Zanzibar")
        other, _ = _conversation(world, "Kilimanjaro")
        queue = pending.get_pending_store()
        queue.propose("memory", "add", {"text": "The user climbs."}, {"origin": "extraction"}, conversation_id=other)
        assert world.manager.delete_conversation(gone) is True
        # A turn that outlived the deletion, a page left open: each write lands nowhere.
        assert cache.put("Where?", "In Zanzibar.", model="m", conversation_id=gone) == ""
        late = None
        try:
            queue.propose("memory", "add", {"text": "The user grew up in Zanzibar."}, {"origin": "extraction"},
                          conversation_id=gone)
        except pending.ConversationForgotten as exc:
            late = exc
        assert late is not None, "a proposal about a deleted conversation was queued"
        told = pending.WriteGate(pending=queue, conversation_id=gone, max_per_run=5).write(
            "memory", "add", {"text": "The user grew up in Zanzibar."})
        assert told.startswith("Not proposed: this conversation was deleted"), told
        assert branches.branch_manager.fork(gone, fork_message_id=first.id, name="late") is None
        # A branch that outlived the deletion takes no turn.
        branches._FORGOTTEN.discard(gone)
        outlived = branches.branch_manager.fork(gone, fork_message_id=first.id, name="outlived")
        branches._FORGOTTEN.add(gone)
        assert branches.branch_manager.add_branch_message(outlived.branch_id, gone, "user", "Zanzibar again") is None
        refused = None
        try:
            feedback.feedback_store.add_feedback(feedback.FeedbackEntry(
                conversation_id=gone, message_id="1", feedback_text="Zanzibar was wrong"))
        except ValueError as exc:
            refused = exc
        assert refused is not None, "feedback about a deleted conversation was kept"

        # A write already on its way when the deletion came -- the deletion runs in its slow step -- is refused at
        # the write, under the lock the forgetting held.
        def deleting(word, then=None):
            """The write's slow step, the deletion run inside it the first time (a step may run twice)."""
            def step(*args, **kwargs):
                if word not in deleted:
                    deleted[word] = world.manager.delete_conversation(ids[word])
                return then(*args, **kwargs) if then else None
            return step

        deleted = {}

        ids = {}
        for word in ("Pemba", "Wete", "Chake", "Mkoani"):
            ids[word], _ = _conversation(world, word)
        cache.embeddings_available = True
        plain_embedding = module._get_embedding
        module._get_embedding = deleting("Pemba")
        try:
            assert cache.put("Where is the house?", "Under the floor in Pemba.", model="m",
                             conversation_id=ids["Pemba"]) == ""
        finally:
            module._get_embedding = plain_embedding
            cache.embeddings_available = False
        queue._uid = deleting("Wete", queue._uid)
        raced = None
        try:
            queue.propose("memory", "add", {"text": "The user grew up in Wete."}, {"origin": "extraction"},
                          conversation_id=ids["Wete"])
        except pending.ConversationForgotten as exc:
            raced = exc
        finally:
            del queue._uid
        assert raced is not None, "a proposal on its way when the deletion came was queued"
        branches.branch_manager._count_branches = deleting("Chake", branches.branch_manager._count_branches)
        try:
            assert branches.branch_manager.fork(ids["Chake"], fork_message_id=first.id, name="raced") is None
        finally:
            del branches.branch_manager._count_branches
        entry = feedback.FeedbackEntry(conversation_id=ids["Mkoani"], message_id="1", feedback_text="Mkoani was wrong")
        entry.validate = deleting("Mkoani", entry.validate)
        raced = None
        try:
            feedback.feedback_store.add_feedback(entry)
        except ValueError as exc:
            raced = exc
        assert raced is not None, "feedback on its way when the deletion came was kept"
        assert deleted == {"Pemba": True, "Wete": True, "Chake": True, "Mkoani": True}, (
            f"control: each deletion ran inside its write: {deleted}")
        for path in (Path(cache._db_path), Path(queue.db_path), world.data / "branches.db",
                     Path(feedback.feedback_store._db_path)):
            held = _bytes(path)
            for word in (b"Zanzibar", b"Pemba", b"Wete", b"Chake", b"Mkoani"):
                assert word not in held, f"{word!r} landed in {path.name} after its conversation's deletion"
    finally:
        world.restore()


def _writes_taken(world, cid, first, tag):
    """Whether every store of words takes the conversation's writes: a branch, a proposal, feedback, a cache put."""
    branches = world.mod("conversation_branches").branch_manager
    queue = world.mod("pending_writes").get_pending_store()
    feedback = world.mod("feedback")
    cache = world.mod("semantic_cache").semantic_cache
    cache._config["enabled"] = True
    cache.embeddings_available = False
    assert branches.fork(cid, fork_message_id=first.id, name=f"again {tag}") is not None, f"{tag}: the branches"
    queue.propose("memory", "add", {"text": f"The user sails, {tag}."}, {"origin": "extraction"}, conversation_id=cid)
    feedback.feedback_store.add_feedback(feedback.FeedbackEntry(conversation_id=cid, message_id=tag))
    assert cache.put(f"Where, {tag}?", "In Mombasa.", model="m", conversation_id=cid) != "", f"{tag}: the cache"


def test_fe22_a_conversation_that_stays_or_comes_back_is_taken_again_by_every_store(tmp_path):
    # A deletion refused at any store of words.
    for at in ("branches", "proposed facts", "feedback", "semantic cache"):
        world = _world(tmp_path / at.replace(" ", "_"))
        try:
            pending = world.mod("pending_writes")
            feedback = world.mod("feedback")
            cache = world.mod("semantic_cache").semantic_cache
            cache._config["enabled"] = True
            cache.embeddings_available = False
            cid, first = _conversation(world, "Mombasa")
            queue = pending.get_pending_store()
            queue.propose("memory", "add", {"text": "The user sails."}, {"origin": "extraction"}, conversation_id=cid)
            feedback.feedback_store.add_feedback(feedback.FeedbackEntry(conversation_id=cid, message_id="0"))
            cache.put("Where?", "In Mombasa.", model="m", conversation_id=cid)
            broken = {"branches": (world.mod("conversation_branches").branch_manager, "_get_connection"),
                      "proposed facts": (queue, "_connect"), "feedback": (feedback.feedback_store, "_get_conn"),
                      "semantic cache": (cache, "_forget_entries")}[at]
            setattr(*broken, _raising())
            refused = None
            try:
                world.manager.delete_conversation(cid)
            except world.conv.ConversationNotForgotten as exc:
                refused = exc
            finally:
                delattr(*broken)
            assert refused is not None and refused.reason == f"{at}: RuntimeError", (at, refused)
            _writes_taken(world, cid, first, at)
        finally:
            world.restore()
    # The queue asked on its own, its forgetting failing: it takes the conversation back.
    world = _world(tmp_path / "queue")
    try:
        pending = world.mod("pending_writes")
        cid, _ = _conversation(world, "Mombasa")
        queue = pending.get_pending_store()
        queue._connect = _raising()
        failed = None
        try:
            queue.forget_conversation(cid)
        except RuntimeError as exc:
            failed = exc
        finally:
            del queue._connect
        assert failed is not None
        queue.propose("memory", "add", {"text": "The user sails."}, {"origin": "extraction"}, conversation_id=cid)
    finally:
        world.restore()
    # The conversation store's own write failing once every store forgot: a user's deletion, a peer's.
    for path, refuse in (("user", "DELETE FROM conversations"), ("peer", "DELETE FROM messages")):
        world = _world(tmp_path / path, refuse=refuse)
        try:
            cid, first = _conversation(world, "Mombasa")
            if path == "user":
                assert world.manager.delete_conversation(cid) is False
            else:
                assert world.manager.apply_synced_conversation({"conversation": {"id": cid}}, deleted=True,
                                                               updated_at=_LATER) is False
            assert world.manager.get_conversation(cid) is not None
            _writes_taken(world, cid, first, path)
        finally:
            world.restore()
    # A conversation that comes back under its id: a peer's later record winning over its deletion.
    world = _world(tmp_path / "back")
    try:
        cid, first = _conversation(world, "Mombasa")
        assert world.manager.delete_conversation(cid) is True
        assert world.manager.apply_synced_conversation(_peer_payload(cid)) is True
        _writes_taken(world, cid, first, "back")
    finally:
        world.restore()


def test_fe23_the_store_is_locked_first_and_the_process_lets_go_once_committed(tmp_path):
    tools, coding = _recorder("agentic_executor"), _recorder("chat_coding_agent")
    world = _world(tmp_path, memory={"agentic_executor": tools, "chat_coding_agent": coding})
    try:
        branches = world.mod("conversation_branches").branch_manager
        gone, first = _conversation(world, "Zanzibar")
        branch = branches.fork(gone, fork_message_id=first.id, name="b")
        writer = _plain(world.data / "conversations.db")
        writer.execute("BEGIN IMMEDIATE")
        try:
            assert world.manager.delete_conversation(gone) is False, "a deletion went through a store locked"
            assert world.manager.apply_synced_conversation({"conversation": {"id": gone}}, deleted=True,
                                                           updated_at=_LATER) is False
        finally:
            writer.rollback()
            writer.close()
        assert world.manager.get_conversation(gone) is not None
        assert branches.get_branch(branch.branch_id) is not None, "a store forgot before the store's lock was taken"
        assert world.asked.onion == [], "the onion forgot before the store's lock was taken"
        assert tools.asked == [] and coding.asked == [], "the process let go of a conversation that stays"
        assert world.manager.delete_conversation(gone) is True
        assert tools.asked == [gone] and coding.asked == [gone]
    finally:
        world.restore()
    # The conversation store's own write failing once the stores of words forgot: the process keeps its memory.
    tools, coding = _recorder("agentic_executor"), _recorder("chat_coding_agent")
    world = _world(tmp_path / "failing", memory={"agentic_executor": tools, "chat_coding_agent": coding},
                   refuse="DELETE FROM conversations")
    try:
        cid, _ = _conversation(world, "Zanzibar")
        assert world.manager.delete_conversation(cid) is False
        assert tools.asked == [] and coding.asked == [], "the process let go before the deletion was committed"
    finally:
        world.restore()


def test_fe18_the_onion_is_asked_first_and_its_refusal_leaves_every_store(tmp_path):
    world = _world(tmp_path)
    try:
        world.librarian.forget_conversation = _raising()
        branches = world.mod("conversation_branches").branch_manager
        gone, first = _conversation(world, "Zanzibar")
        branch = branches.fork(gone, fork_message_id=first.id, name="b")
        refused = None
        try:
            world.manager.delete_conversation(gone)
        except world.conv.ConversationNotForgotten as exc:
            refused = exc
        assert refused is not None, "the onion's refusal did not refuse the deletion"
        assert branches.get_branch(branch.branch_id) is not None, "a store forgot before the onion refused"
        assert branches.fork(gone, fork_message_id=first.id, name="again") is not None, (
            "a store refuses the writes of a conversation the onion kept")
        assert world.manager.apply_synced_conversation({"conversation": {"id": gone}}, deleted=True,
                                                       updated_at=_LATER) is False
        assert branches.get_branch(branch.branch_id) is not None, "a peer's deletion forgot a store before the onion"
    finally:
        world.restore()
    world = _world(tmp_path / "never")
    try:
        branches = world.mod("conversation_branches").branch_manager
        _, first = _conversation(world, "Zanzibar")
        assert world.manager.delete_conversation("never-held") is False
        assert world.manager.apply_synced_conversation({"conversation": {"id": "never-held-either"}}, deleted=True,
                                                       updated_at=_LATER) is True
        assert world.asked.onion == [], "the onion was asked of a conversation the store does not hold"
        for cid in ("never-held", "never-held-either"):
            assert branches.fork(cid, fork_message_id=first.id, name="b") is not None, (
                f"a store marked {cid}, a conversation the store does not hold")
    finally:
        world.restore()


def test_fe19_a_deletion_creates_no_store(tmp_path):
    world = _world(tmp_path)
    try:
        gone, _ = _conversation(world, "Zanzibar")
        stores = {"feedback": Path(world.mod("feedback").feedback_store._db_path),
                  "project links": Path(world.mod("projects").project_store._db_path),
                  "semantic cache": Path(world.mod("semantic_cache").semantic_cache._db_path),
                  "proposed facts": Path(world.mod("pending_writes")._default_db_path())}
        assert not any(path.exists() for path in stores.values()), f"control: no store yet: {stores}"
        assert world.manager.delete_conversation(gone) is True
        made = sorted(name for name, path in stores.items() if path.exists())
        assert made == [], f"a deletion created a store it found absent: {made}"
        # The queue it did not open takes no proposal about the conversation when it opens.
        pending = world.mod("pending_writes")
        late = None
        try:
            pending.get_pending_store().propose("memory", "add", {"text": "The user grew up in Zanzibar."},
                                                {"origin": "extraction"}, conversation_id=gone)
        except pending.ConversationForgotten as exc:
            late = exc
        assert late is not None, "a queue opened after the deletion took a proposal about it"
    finally:
        world.restore()


def test_fe20_a_driver_whose_errors_are_not_sqlite3s_never_turns_a_deletion(tmp_path, caplog):
    world = _world(tmp_path)
    try:
        module = world.mod("semantic_cache")
        cache = module.semantic_cache
        cache._config["enabled"] = True
        cache.embeddings_available = False
        gone, _ = _conversation(world, "Zanzibar")
        cache.put("Where did the user grow up?", "In Zanzibar.", model="m", conversation_id=gone)
        # A new process opens the file the cipher's way: its schema step runs again, under errors not sqlite3's.
        cache._schema_ready = False
        plain = module._safe_connect
        module._safe_connect = lambda db_path, **options: _CipherLike(plain(db_path, **options))
        assert cache.put("Where did the user dream?", "Of Zanzibar.", model="m", conversation_id=gone) != "", (
            "the cache's schema step failed under the cipher's driver")
        assert world.manager.delete_conversation(gone) is True
        with sqlite3.connect(str(cache._db_path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0] == 0
    finally:
        world.restore()

    # Every connection's errors foreign, the logs' emptying refused: said, and the deletion stands.
    world = _world(tmp_path / "logs", refuse="wal_checkpoint")
    try:
        branches = world.mod("conversation_branches").branch_manager
        feedback = world.mod("feedback")
        gone, first = _conversation(world, "Zanzibar")
        branches.fork(gone, fork_message_id=first.id, name="b")
        feedback.feedback_store.add_feedback(feedback.FeedbackEntry(conversation_id=gone, message_id="1"))
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert world.manager.delete_conversation(gone) is True, "a log not emptied turned the deletion"
        said = [r.getMessage() for r in caplog.records if "write-ahead log could not be emptied" in r.getMessage()]
        assert len(said) == 3, f"each store whose log could not be emptied says so: {said}"
        assert world.manager.get_conversation(gone) is None
    finally:
        world.restore()


def _route(order, chat=None):
    """The conversations route in a window of its own: the store's deletion recorded in ``order``, and the chat
    routes' live turns ``chat`` (None: the chat routes not loaded)."""
    deps = types.ModuleType("opti_oignon.api.deps")
    deps.CONVERSATION_AVAILABLE = True
    deps.conversation_manager = SimpleNamespace(
        delete_conversation=lambda conv_id: order.append(("delete", conv_id)) or True)
    seeded = {"opti_oignon.api.deps": deps}
    if chat is not None:
        module = types.ModuleType("opti_oignon.api.routes_chat")
        module.live_chat_turns = chat
        seeded["opti_oignon.api.routes_chat"] = module
    loaded, restore = isolate(
        targets={"opti_oignon.api.schemas": source("api", "schemas.py"),
                 "opti_oignon.api.routes_conversations": source("api", "routes_conversations.py")},
        seeded=seeded, packages=("opti_oignon.api",))
    return loaded["opti_oignon.api.routes_conversations"], restore


def test_fe21_the_deletion_route_stops_the_conversations_live_turns_first(tmp_path):
    # The real stop, which the window stands in for: it stops the turns of that conversation.
    stop = _function("api/routes_chat.py", "LiveChatTurns.stop")
    assert ast.unparse(stop.body[-1]) == "return _stop_conversation_turns(conversation_id)", ast.unparse(stop)
    order = []
    routes, restore = _route(order, SimpleNamespace(stop=lambda conv_id: order.append(("stop", conv_id)) or 1))
    try:
        assert routes.delete_conversation("c1").status_code == 204
        assert order == [("stop", "c1"), ("delete", "c1")], f"the live turns were not stopped first: {order}"
    finally:
        restore()
    order = []
    routes, restore = _route(order)
    try:
        assert routes.delete_conversation("c1").status_code == 204
        assert order == [("delete", "c1")], order
    finally:
        restore()
    order = []
    routes, restore = _route(order, SimpleNamespace(stop=_raising("the turns could not be reached")))
    try:
        assert routes.delete_conversation("c1").status_code == 204, "a stop that failed failed the deletion"
        assert order == [("delete", "c1")], order
    finally:
        restore()


def _refusing(name, reason="the store is locked by another process"):
    stub = types.ModuleType(f"opti_oignon.{name}")

    def forget_conversation(conversation_id):
        raise RuntimeError(reason)

    stub.forget_conversation = forget_conversation
    return stub


def test_fe8_a_store_that_keeps_its_words_and_cannot_forget_refuses_by_name(tmp_path):
    refusing = _refusing("feedback")
    world = _world(tmp_path, seeded_stores={"feedback": refusing})
    try:
        gone, _ = _conversation(world, "Zanzibar")
        branches = world.mod("conversation_branches").branch_manager
        branch = branches.fork(gone, fork_message_id=world.manager.get_messages(gone)[0].id, name="b")
        refused = None
        try:
            world.manager.delete_conversation(gone)
        except world.conv.ConversationNotForgotten as exc:
            refused = exc
        assert refused is not None and refused.reason == "feedback: RuntimeError", refused
        assert "already forgotten: the onion, branches, proposed facts" in str(refused), str(refused)
        assert world.manager.get_conversation(gone) is not None and len(world.manager.get_messages(gone)) == 2
        assert world.asked.runs == [], "a refused deletion forgot the conversation's runs"
        assert branches.get_branch(branch.branch_id) is None, "control: the stores before the refusal forgot"
        refusing.forget_conversation = lambda conversation_id: 0
        assert world.manager.delete_conversation(gone) is True, "once the store could forget, the deletion went"
    finally:
        world.restore()


def test_fe9_a_store_of_ids_only_that_cannot_forget_is_said_and_the_deletion_goes_on(tmp_path, caplog):
    world = _world(tmp_path, seeded_stores={"projects": _refusing("projects")})
    try:
        gone, _ = _conversation(world, "Zanzibar")
        with caplog.at_level(logging.WARNING, logger=CONV):
            assert world.manager.delete_conversation(gone) is True
        assert any("project links could not be forgotten" in r.getMessage() for r in caplog.records), (
            "a store of ids that could not forget was not said")
    finally:
        world.restore()


def test_fe10_a_peers_deletion_forgets_them_too(tmp_path):
    world = _world(tmp_path)
    try:
        gone, first = _conversation(world, "Zanzibar")
        _conversation(world, "Kilimanjaro")
        branches = world.mod("conversation_branches").branch_manager
        branch = branches.fork(gone, fork_message_id=first.id, name="b")
        path = world.data / "conversations.db"
        keeper = _keeper(path)
        assert world.manager.apply_synced_conversation({"conversation": {"id": gone}}, deleted=True,
                                                       updated_at=_LATER) is True
        assert branches.get_branch(branch.branch_id) is None, "a peer's deletion left the conversation's branches"
        held = _bytes(path)
        assert b"Zanzibar" not in held, "a peer's deletion left the conversation's words in the file or its log"
        assert b"Kilimanjaro" in held
        keeper.close()
    finally:
        world.restore()

    world = _world(tmp_path / "refusing", seeded_stores={"feedback": _refusing("feedback")})
    try:
        cid, _ = _conversation(world, "Zanzibar")
        assert world.manager.apply_synced_conversation({"conversation": {"id": cid}}, deleted=True,
                                                       updated_at=_LATER) is False
        assert world.manager.get_conversation(cid) is not None
    finally:
        world.restore()


def _missing_dependency(name):
    """A store module that is there, one of its own imports missing: asking it fails as such a module does."""
    stub = types.ModuleType(f"opti_oignon.{name}")

    def __getattr__(attr):
        if attr == "forget_conversation":
            raise ModuleNotFoundError("No module named 'yaml'", name="yaml")
        raise AttributeError(attr)

    stub.__getattr__ = __getattr__
    return stub


def test_fe11_a_store_this_tree_lacks_is_passed_one_without_its_forgetting_refuses(tmp_path, caplog):
    world = _world(tmp_path, lacking=("feedback",))
    try:
        gone, _ = _conversation(world, "Zanzibar")
        with caplog.at_level(logging.WARNING, logger=CONV):
            assert world.manager.delete_conversation(gone) is True
        assert any("no feedback in this tree" in r.getMessage() for r in caplog.records), "the absence was not said"
    finally:
        world.restore()

    world = _world(tmp_path / "older", seeded_stores={"feedback": types.ModuleType("opti_oignon.feedback")})
    try:
        cid, _ = _conversation(world, "Zanzibar")
        refused = None
        try:
            world.manager.delete_conversation(cid)
        except world.conv.ConversationNotForgotten as exc:
            refused = exc
        assert refused is not None and refused.reason == "feedback: ImportError", (
            f"a store that is there and cannot be asked to forget was taken for an absent one: {refused!r}")
        assert world.manager.get_conversation(cid) is not None
    finally:
        world.restore()

    # A module that is there, a dependency of its own missing: broken, never absent.
    world = _world(tmp_path / "dependency", seeded_stores={"semantic_cache": _missing_dependency("semantic_cache")})
    try:
        cid, _ = _conversation(world, "Zanzibar")
        refused = None
        try:
            world.manager.delete_conversation(cid)
        except world.conv.ConversationNotForgotten as exc:
            refused = exc
        assert refused is not None and refused.reason == "semantic cache: ModuleNotFoundError", (
            f"a store whose own dependency is missing was taken for an absent one: {refused!r}")
        assert world.manager.get_conversation(cid) is not None
    finally:
        world.restore()


# Every table of the package keyed by a conversation, and who forgets it at a deletion: the conversation store
# itself (its messages go with the row), the onion (its own forgetting), or the registry's store; or the reason it
# is kept. A new table keyed by a conversation fails this census until it is forgotten or named.
_FORGOTTEN_BY = {
    "conversation.py": "the conversation store: its messages go with the conversation's row",
    "memory/onion_store.py": "the onion: conversation._forget_onion",
}
_KEPT = {
    "analytics.py": "how each turn performed: ids, models, timings and counts, never a turn's words",
    "context_ledger.py": "what each window held, counted: numbers and labels, never a turn's words",
    "memory/legacy.py": "the old store's memories, frozen since their migration into the facts, still read into "
                        "the window when the facts give nothing: owed, with the facts, to the next block",
    "veilid/change_feed.py": "the sync change feed: each record's latest payload under (kind, record_id), a "
                             "conversation's whole included: owed, with its compaction, to the next block",
    "veilid/deferred_ledger.py": "what a peer offered that waits its turn, under (kind, record_id): owed, with the "
                                 "change feed, to the next block",
}
_KEYS = {"conversation", "conversation_id", "conv_id", "source_conversation_id"}
# Records of any kind under their kind and id: a conversation's among them.
_GENERIC = {"kind", "record_id"}


def _sql_texts(tree):
    """The SQL a module spells: its string constants, and its f-strings with each placeholder as ``{}``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            yield "".join(part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else "{}"
                          for part in node.values)


def _keyed_tables():
    """Each module of the package whose SQL creates a table keyed by a conversation -- a conversation column, or
    records of any kind under (kind, record_id) -- and those tables; the package walked as the census walks it (no
    ``data`` directory entered), f-strings read with their placeholders."""
    root = REPO / "opti_oignon"
    found = {}
    for path in sorted({Path(path) for path in package_modules().values()}):
        relative = path.relative_to(root)
        for text in _sql_texts(ast.parse(path.read_text(encoding="utf-8"))):
            for match in re.finditer(r"CREATE TABLE(?: IF NOT EXISTS)?\s+([\w{}]+)\s*\((.*?)\)\s*(?:;|$)", text,
                                     re.S | re.I):
                columns = {part.strip().split()[0].lower().strip("()") for part in match.group(2).split(",")
                           if part.strip().split()}
                if columns & _KEYS or _GENERIC <= columns:
                    found.setdefault(relative.as_posix(), set()).add(match.group(1))
    return found


def test_fe12_every_table_keyed_by_a_conversation_is_forgotten_or_named(tmp_path):
    world = _world(tmp_path)
    try:
        registry = {module.split(".", 1)[1].replace(".", "/") + ".py" for _store, module, _f, _w in
                    world.conv._FORGETTERS}
    finally:
        world.restore()
    found = _keyed_tables()
    # The census finds what it must, or it proves nothing.
    assert {"branches", "branch_messages"} <= found.get("conversation_branches.py", set()), found
    assert "pending_writes" in found.get("pending_writes.py", set()), found
    assert "messages" in found.get("conversation.py", set()), found
    assert "veilid/change_feed.py" in found, f"the census reads no f-string, or no generic key: {sorted(found)}"
    unaccounted = {module: sorted(tables) for module, tables in found.items()
                   if module not in registry and module not in _FORGOTTEN_BY and module not in _KEPT}
    assert unaccounted == {}, (
        f"tables keyed by a conversation that a deletion neither forgets nor names: {unaccounted}")
    stale = sorted(set(_KEPT) & registry)
    assert stale == [], f"named as kept and forgotten too: {stale}"
