#!/usr/bin/env python3
"""Contracts for what a deletion still left: the response cache, the offline queue, the sync feed, the sandbox audit.

The response cache kept the model's answers under a hash of the turn and its history, with no conversation: a
deletion could not reach them, and a new conversation repeating a deleted one's first turn within the hour was served
the deleted answer. The offline queue kept each turn asked while the engine was offline, for a replay nothing ever
ran, and the turn was told it would be processed. The sync change feed kept every version of every conversation and
of every fact, for ever: a deletion only appended a tombstone. The sandbox audit kept every command a sandbox ran,
whole, and nothing ever forgot it; a conversation's code sandbox and its workspace binding outlived it. The account
wipe left every conversation in single-user mode, where they are all the one user's, and never touched the old
memory store. Each answer now keeps its conversation, which forgets it; the queue is given no turn; the feed keeps one
version of each record, so a tombstone erases what came before it; the audit keeps the record of what ran and forgets
its words; the wipe, in single-user mode, deletes every conversation through its deletion and empties the stores of
the user's requests. Every erasure overwrites.

  * DL1 -- A DELETION FORGETS THE ANSWERS THE RESPONSE CACHE KEPT OF THE CONVERSATION, overwritten; another
    conversation's and a request's made outside any stay; a turn that outlived the deletion lands nowhere, until the
    conversation comes back under its id.
  * DL2 -- AN OLDER RESPONSE CACHE DROPS WHAT NO DELETION COULD REACH: its entries are served, of an origin not
    known, until the first forgetting drops them; a forgetting that comes first changes no schema and drops them all;
    a cache whose file is not there is not created to be asked.
  * DL3 -- EVERY CONNECTION OF THE RESPONSE CACHE OVERWRITES WHAT IT FREES: an expiry, an eviction, a clear.
  * DL4 -- A STORE OF WORDS THAT CANNOT FORGET REFUSES THE DELETION, and the response cache takes the conversation
    back; once it can, the deletion goes through.
  * DL5 -- THE SANDBOX AUDIT FORGETS A DELETED CONVERSATION'S WORDS AND KEEPS THE RECORD OF WHAT RAN: its sessions'
    commands, block reasons and approval details replaced, overwritten; the time, the outcome, the sizes stay; an
    older row reached through its chat coding session's id; another conversation's and another session's rows stay;
    a command of the conversation after it is recorded without its words, until a refused deletion hands it back.
  * DL6 -- THE CODE SANDBOX AND THE WORKSPACE BINDING LET GO: a conversation's code sandbox is destroyed, its
    workspace released, each asked only of a process that loaded it, never loaded to be asked.
  * DL7 -- THE SYNC FEED KEEPS ONE VERSION OF EACH RECORD: each append erases the record's earlier versions,
    overwritten, and every delta and high-water a peer is served is the one the whole journal gave.
  * DL8 -- A TOMBSTONE ERASES EVERY VERSION OF ITS RECORD AND EMPTIES THE LOG: none of the record's words stay in the
    file or its log; a read in progress keeps the log, said, and the next deletion empties it.
  * DL9 -- THE FEED'S PASS ERASES WHAT AN OLDER BUILD LEFT, creates no feed, and never raises.
  * DL10 -- THE DEFERRED LEDGER HOLDS SKILLS ONLY AND OVERWRITES WHAT IT REMOVES: no conversation, fact or note is
    ever deferred.
  * DL11 -- AN OFFLINE TURN IS TOLD THE TRUTH AND KEPT NOWHERE: the executor queues no turn, its notice and status
    say not sent, and the queue's emptying erases what an older version queued, overwritten, creating nothing.
  * DL12 -- THE EXECUTOR'S CACHE PUT CARRIES ITS CONVERSATION.
  * DL13 -- THE BOOT ERASES THE FEED'S SUPERSEDED VERSIONS AND THE OFFLINE QUEUE'S TURNS.
  * DL14 -- THE SINGLE-USER WIPE DELETES EVERY CONVERSATION THROUGH ITS DELETION and empties the stores of the user's
    requests and the old memory store; the answer no longer names them.
  * DL15 -- A MULTI-USER WIPE TOUCHES NO INSTALL-WIDE STORE AND NAMES THEM, nor does a wipe that cannot read the mode.
  * DL16 -- A REFUSED DELETION STOPS THE SINGLE-USER WIPE AND SAYS HOW MANY ARE LEFT, and why.
  * DL17 -- THE CENSUS (supersedes ft13 of tests/test_facts_forgotten_contracts.py): every table of the package keyed
    by a conversation is forgotten at a deletion or named with its reason, the response cache and the sandbox audit
    forgotten, the change feed and the deferred ledger named for what they keep.
  * DL18 -- THE LOADER SAYS AN OFFLINE TURN WAS NOT SENT (supersedes ld8's status half of
    tests/test_chat_loader_contracts.py, whose table said queued).
  * DL19-DL23 -- THE EXECUTOR'S GOLDENS, V2 (supersede jw13a, jw13b and jw13e of
    tests/test_executor_onion_off_golden_contracts.py, jw33a and jw33d of tests/test_tool_path_off_golden_contracts.py,
    which pinned v1): each put of the response cache carries its conversation, an offline turn is told it was not
    sent and reaches no queue; nothing else differs from v1. DL19 to DL22 run the contracts they keep word for word.
  * DL24 -- AN OFFLINE TURN RECORDS ITS OUTCOME WITHOUT GENERATION AND QUEUES NOTHING (supersedes e5 of
    tests/test_executor_ledger_emission_contracts.py, which asked the turn to say queued).
  * DL25 -- THE ADMISSION STANDS BETWEEN THE PLAN AND THE COMPOSITION, AND AN OFFLINE TURN IS QUEUED NOWHERE
    (supersedes jw6 of tests/test_executor_onion_window_contracts.py, which asked the offline turn to reach the queue).
  * DL26 -- A DELETION ERASES ITS RECORD'S VERSIONS IN THE SYNC FEED BEFORE ITS COMMIT, WHATEVER THE TOMBSTONE DOES:
    a conversation's, and an erased fact's; a feed that cannot be written refuses the deletion by name, or keeps the
    fact.
  * DL27 -- THE SEMANTIC CACHE'S EMPTYING OVERWRITES, AND RAISES WHEN IT CANNOT: the wipe names it.
  * DL28 -- THE TOMBSTONE OUT-CLOCKS EVERY VERSION THE DELETION ERASED, AND A COPY COMING BACK LOSES: the erased clock
    stays as the record's floor until the tombstone lands, or for good when it cannot be written.

Local-only (the public distribution ships no tests). Every store real, loaded from source in isolation windows, in
tmp, on connections that do not overwrite deleted pages by default, as the cipher's driver does not (sqlcipher3:
secure_delete 0); the file's bytes read after each erasure.
"""

import ast
import json
import logging
import re
import sqlite3
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_census import modules as package_modules  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

CONV = "opti_oignon.conversation"
BUDGET_S = {
    "test_dl1_a_deletion_forgets_the_answers_the_response_cache_kept_of_the_conversation": 3.0,
    "test_dl2_an_older_response_cache_drops_what_no_deletion_could_reach": 3.0,
    "test_dl3_every_connection_of_the_response_cache_overwrites_what_it_frees": 3.0,
    "test_dl4_a_store_of_words_that_cannot_forget_refuses_and_the_cache_takes_the_conversation_back": 3.0,
    "test_dl5_the_sandbox_audit_forgets_a_deleted_conversations_words_and_keeps_the_record": 5.0,
    "test_dl6_the_code_sandbox_and_the_workspace_binding_let_go": 3.0,
    "test_dl7_the_sync_feed_keeps_one_version_of_each_record_and_serves_the_same_deltas": 3.0,
    "test_dl8_a_tombstone_erases_every_version_of_its_record_and_empties_the_log": 3.0,
    "test_dl9_the_feeds_pass_erases_what_an_older_build_left_and_creates_nothing": 3.0,
    "test_dl10_the_deferred_ledger_holds_skills_only_and_overwrites_what_it_removes": 3.0,
    "test_dl11_an_offline_turn_is_told_the_truth_and_kept_nowhere": 3.0,
    "test_dl12_the_executors_cache_put_carries_its_conversation": 3.0,
    "test_dl13_the_boot_erases_the_feeds_superseded_versions_and_the_offline_queues_turns": 3.0,
    "test_dl14_the_single_user_wipe_deletes_every_conversation_through_its_deletion": 5.0,
    "test_dl15_a_multi_user_wipe_touches_no_install_wide_store_and_names_them": 5.0,
    "test_dl16_a_refused_deletion_stops_the_single_user_wipe_and_says_how_many_are_left": 5.0,
    "test_dl17_every_table_keyed_by_a_conversation_is_forgotten_or_named": 3.0,
    "test_dl18_the_loader_says_an_offline_turn_was_not_sent": 30.0,
    # The first of DL19 to DL23 to run pays for the golden captures (about a second for both seeds), whatever the
    # order: each is given room for it, as the contracts they keep.
    "test_dl19_the_isolated_window_sends_the_v2_golden_requests": 10.0,
    "test_dl20_the_integrated_window_sends_the_v2_golden_requests": 10.0,
    "test_dl21_the_v2_golden_options_leave_the_llama_server_heads_with_no_max_tokens": 10.0,
    "test_dl22_every_tool_path_case_does_what_the_v2_golden_holds": 10.0,
    "test_dl23_a_check_on_an_absent_v2_golden_file_fails_by_name_and_creates_nothing": 10.0,
    "test_dl24_an_offline_turn_records_its_outcome_without_generation_and_queues_nothing": 5.0,
    "test_dl25_the_admission_stands_between_the_plan_and_the_composition_and_an_offline_turn_is_queued_nowhere": 10.0,
    "test_dl26_a_deletion_erases_its_records_versions_in_the_sync_feed_before_its_commit": 5.0,
    "test_dl27_the_semantic_caches_emptying_overwrites_and_raises_when_it_cannot": 3.0,
    "test_dl28_the_tombstone_out_clocks_every_erased_version_and_a_copy_coming_back_loses": 5.0,
}


def _plain(path, *, wait=0.3, check_same_thread=False, **_options):
    """A plain connection that, like many builds, does not overwrite deleted pages unless asked to."""
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=wait)
    conn.execute("PRAGMA secure_delete = OFF")
    return conn


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _bytes(path):
    path = Path(path)
    return b"".join(p.read_bytes() for p in path.parent.glob(path.name + "*") if p.is_file())


def _world(tmp_path, *, stores=None, wipe=False, single=True, auth=True, packages=()):
    """conversation.py and the response cache over ``tmp_path``; ``stores`` seeds stand-ins (the sandbox manager, the
    code executor, the workspace bindings, the old memory store); with ``wipe``, the account wipe, the offline queue
    and the every-conversation purge, the auth manager in single-user mode or not (``auth=False``: not readable)."""
    data = tmp_path / "data"
    data.mkdir(parents=True)
    seeded = {
        "opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain),
        "opti_oignon.db_encryption": _module("opti_oignon.db_encryption", get_encrypted_connection=_plain,
                                             SQLCIPHER_AVAILABLE=False),
        "opti_oignon.config": _module("opti_oignon.config", DATA_DIR=data),
        "opti_oignon.run_journal": _module("opti_oignon.run_journal", forget_conversation=lambda cid: None,
                                           wipe=lambda: {"wiped": True}),
        "opti_oignon.veilid.guard": _module("opti_oignon.veilid.guard", veilid_available=lambda: False),
    }
    seeded.update(stores or {})
    targets = {"opti_oignon.response_cache": source("response_cache.py"), CONV: source("conversation.py")}
    if wipe:
        if auth:
            # One account, "u-1": a single-user install's one user is "local" or that account.
            seeded["opti_oignon.auth"] = _module("opti_oignon.auth", auth_manager=SimpleNamespace(
                single_user_mode=single, count_users=lambda: 1,
                get_user=lambda uid: SimpleNamespace(id=uid) if uid == "u-1" else None))
        targets["opti_oignon.sync_queue"] = source("sync_queue.py")
        targets["opti_oignon.conversation_wipe"] = source("conversation_wipe.py")
        targets["opti_oignon.user_data_manager"] = source("user_data_manager.py")
    loaded, restore = isolate(targets=targets, seeded=seeded, packages=("opti_oignon.veilid", *packages))
    try:
        conv = loaded[CONV]
        manager = conv.ConversationManager(db_path=data / "conversations.db", publish_to_sync=False)
        if wipe:
            queue = loaded["opti_oignon.sync_queue"]
            queue.sync_queue = queue.SyncQueue(db_path=data / "sync_queue.db")
            loaded[CONV].conversation_manager = manager
    except BaseException:
        restore()
        raise
    return SimpleNamespace(loaded=loaded, restore=restore, conv=conv, manager=manager, data=data,
                           mod=lambda name: loaded[f"opti_oignon.{name}"])


def _conversation(world, word):
    cid = world.manager.create_conversation(title=f"about {word}").id
    world.manager.add_message(cid, "user", f"{word} is where the user grew up.")
    return cid


def _origins(path):
    with sqlite3.connect(str(path)) as conn:
        return sorted(row[0] for row in conn.execute("SELECT conversation_id FROM response_cache"))


# ---------------------------------------------------------------------------
# DL1 -- the response cache's answers go with their conversation
# ---------------------------------------------------------------------------
def test_dl1_a_deletion_forgets_the_answers_the_response_cache_kept_of_the_conversation(tmp_path):
    world = _world(tmp_path)
    try:
        cache = world.mod("response_cache").response_cache
        path = world.data / "response_cache.db"
        gone, kept = _conversation(world, "Zanzibar"), _conversation(world, "Lamu")
        assert cache.put("m", "sys", "tides?", "Zanzibar tides follow the moon.", conversation_id=gone)
        assert cache.put("m", "sys", "spice?", "Zanzibar cloves came by dhow.", explicit_key="k-two",
                         conversation_id=gone)
        assert cache.put("m", "sys", "donkeys?", "Lamu donkeys carry the shopping.", conversation_id=kept)
        assert cache.put("m", "sys", "tusks?", "Mombasa tusks arch over the road.")
        assert b"Zanzibar tides" in _bytes(path) and b"Zanzibar cloves" in _bytes(path), "control: the answers landed"
        assert world.manager.delete_conversation(gone), "control: the conversation was deleted"
        held = _bytes(path)
        assert b"Zanzibar" not in held, "the deleted conversation's answers stayed in the response cache's file"
        assert b"Lamu donkeys" in held and b"Mombasa tusks" in held, "another conversation's answers went too"
        assert _origins(path) == sorted([kept, ""]), f"each entry keeps the conversation it came from: {_origins(path)}"
        assert cache.put("m", "sys", "late?", "Zanzibar late answer.", conversation_id=gone) == "", (
            "a turn that outlived its deletion was cached")
        assert b"Zanzibar late" not in _bytes(path), "a forgotten conversation's answer landed"
        world.conv._taken_back(gone)
        assert cache.put("m", "sys", "back?", "Zanzibar is back.", conversation_id=gone), (
            "a conversation back under its id is never cached again")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DL2 -- an older response cache drops what no deletion could reach
# ---------------------------------------------------------------------------
# The table as the cache built it before it kept each entry's conversation.
_OLD_CACHE = (
    "CREATE TABLE IF NOT EXISTS response_cache (cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, prompt_hash TEXT "
    "NOT NULL, query_hash TEXT NOT NULL, response TEXT NOT NULL, task_type TEXT DEFAULT '', created_at REAL NOT "
    "NULL, last_accessed REAL NOT NULL, hit_count INTEGER DEFAULT 0, ttl INTEGER NOT NULL)"
)


def _old_cache(path, *answers):
    with sqlite3.connect(str(path)) as conn:
        conn.execute(_OLD_CACHE)
        for key, answer in answers:
            conn.execute("INSERT INTO response_cache VALUES (?, 'm', 'p', 'q', ?, '', ?, ?, 0, 3600)",
                         (key, answer, time.time(), time.time()))


def test_dl2_an_older_response_cache_drops_what_no_deletion_could_reach(tmp_path):
    world = _world(tmp_path)
    try:
        rc = world.mod("response_cache")
        first = tmp_path / "first.db"
        _old_cache(first, ("k-old", "Zanzibar old answer."), ("k-old-2", "Pemba old answer."))
        cache = rc.ResponseCache(db_path=first)
        served = cache.get("k-old")
        assert served is not None and served.response == "Zanzibar old answer.", "an older entry is no longer served"
        assert _origins(first) == ["?", "?"], f"an older entry's origin is not said not known: {_origins(first)}"
        assert cache.put("m", "sys", "q", "Lamu new answer.", conversation_id="conv-kept")
        assert cache.forget_conversation("conv-other") == 2, "the first forgetting does not drop the older entries"
        held = _bytes(first)
        assert b"Zanzibar old" not in held and b"Pemba old" not in held, "an older entry's words stayed in the file"
        assert _origins(first) == ["conv-kept"], f"a known conversation's entry went: {_origins(first)}"

        before = tmp_path / "before.db"
        _old_cache(before, ("k-a", "Mombasa old answer."), ("k-b", "Malindi old answer."))
        assert rc.ResponseCache(db_path=before).forget_conversation("conv-x") == 2, (
            "a forgetting that comes before any lookup leaves the older entries")
        with sqlite3.connect(str(before)) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(response_cache)")}
        assert "conversation_id" not in columns, "the forgetting changed the schema"
        assert b"Mombasa old" not in _bytes(before), "an older entry's words stayed in the file"

        missing = tmp_path / "missing.db"
        assert rc.ResponseCache(db_path=missing).forget_conversation("conv-x") == 0
        assert not missing.exists(), "a cache that was not there was created to be asked"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DL3 -- every connection of the response cache overwrites what it frees
# ---------------------------------------------------------------------------
def test_dl3_every_connection_of_the_response_cache_overwrites_what_it_frees(tmp_path):
    world = _world(tmp_path)
    try:
        rc = world.mod("response_cache")
        path = tmp_path / "cache.db"
        cache = rc.ResponseCache(db_path=path, max_entries=2)
        cache.put("m", "sys", "a", "Zanzibar expired answer.", ttl=-1, conversation_id="c1")
        assert b"Zanzibar expired" in _bytes(path), "control: the answer landed"
        assert cache.get(rc.ResponseCache.make_cache_key("m", "sys", "a")) is None, "control: the entry expired"
        assert b"Zanzibar expired" not in _bytes(path), "an expired answer stayed in the file's free pages"
        cache.put("m", "sys", "b", "Pemba evicted answer.", conversation_id="c1")
        time.sleep(0.01)
        cache.put("m", "sys", "c", "Lamu second answer.", conversation_id="c1")
        time.sleep(0.01)
        cache.put("m", "sys", "d", "Mombasa third answer.", conversation_id="c1")
        assert b"Pemba evicted" not in _bytes(path), "an evicted answer stayed in the file's free pages"
        assert cache.empty() == 2
        held = _bytes(path)
        assert b"Lamu second" not in held and b"Mombasa third" not in held, "a cleared answer stayed in the file"
        missing = tmp_path / "missing.db"
        assert rc.ResponseCache(db_path=missing).empty() == 0 and not missing.exists(), (
            "an emptying created the cache it found missing")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DL4 -- a store that cannot forget refuses; the cache takes the conversation back
# ---------------------------------------------------------------------------
def test_dl4_a_store_of_words_that_cannot_forget_refuses_and_the_cache_takes_the_conversation_back(tmp_path):
    state = SimpleNamespace(refuse=True, asked=[])

    def forget(conversation_id):
        state.asked.append(conversation_id)
        if state.refuse:
            raise OSError("the audit log is locked")
        return 1

    audit = _module("opti_oignon.sandbox_manager", forget_conversation=forget,
                    unforget_conversation=lambda conversation_id: None)
    world = _world(tmp_path, stores={"opti_oignon.sandbox_manager": audit})
    try:
        cache = world.mod("response_cache").response_cache
        gone = _conversation(world, "Zanzibar")
        assert cache.put("m", "sys", "q", "Zanzibar answer.", conversation_id=gone)
        with pytest.raises(world.conv.ConversationNotForgotten) as refused:
            world.manager.delete_conversation(gone)
        assert "sandbox audit" in str(refused.value), f"the refusal does not name the sandbox audit: {refused.value}"
        assert state.asked == [gone], "the sandbox audit was not asked"
        assert world.manager.get_conversation(gone) is not None, "the conversation went with the refusal"
        assert cache.put("m", "sys", "q2", "Zanzibar answer kept.", conversation_id=gone), (
            "the response cache refuses a conversation whose deletion was refused")
        state.refuse = False
        assert world.manager.delete_conversation(gone), "once the audit can forget, the deletion goes through"
        assert b"Zanzibar" not in _bytes(world.data / "response_cache.db"), "the answers stayed after the deletion"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DL5 -- the sandbox audit forgets the words and keeps the record
# ---------------------------------------------------------------------------
# The audit's tables as an older version built them, before each row kept its conversation.
_OLD_AUDIT = (
    "CREATE TABLE IF NOT EXISTS sandbox_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
    "timestamp REAL NOT NULL, command TEXT NOT NULL, return_code INTEGER, blocked INTEGER DEFAULT 0, block_reason "
    "TEXT DEFAULT '', timed_out INTEGER DEFAULT 0, stdout_len INTEGER DEFAULT 0, stderr_len INTEGER DEFAULT 0, "
    "isolation_backend TEXT DEFAULT '')"
)


def _sandbox(tmp_path):
    """sandbox_manager.py from a byte-exact copy whose settings keep its import-time manager off and put its audit in
    tmp; a manager of the contract's own over the same base."""
    package = tmp_path / "copy"
    (package / "config").mkdir(parents=True)
    base = tmp_path / "sandboxes"
    base.mkdir()
    (package / "config" / "sandbox.yaml").write_text(f"enabled: false\nworkspace_base: {base}\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    loaded, restore = isolate(targets={"opti_oignon.sandbox_manager": copy},
                              seeded={"opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain)})
    try:
        sm = loaded["opti_oignon.sandbox_manager"]
        assert sm.sandbox_manager is None, "the import-time manager must stay off in this window"
        sm._detect_bwrap = lambda: (False, "no bubblewrap in this contract")
        config = sm.SandboxConfig(isolation_backend="auto", strict_mode=False, require_degraded_confirmation=False,
                                  workspace_base=str(base), reconcile_on_start=False)
        manager = sm.SandboxManager(config)
    except BaseException:
        restore()
        raise
    return sm, manager, base / "sandbox_audit.db", restore


def _audit_rows(path):
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM sandbox_audit ORDER BY id")]


def test_dl5_the_sandbox_audit_forgets_a_deleted_conversations_words_and_keeps_the_record(tmp_path):
    sm, manager, path, restore = _sandbox(tmp_path)
    try:
        gone, kept = "conv-zanzibar-0001-abcd", "conv-lamu-0002-efgh"
        coding = f"cc-{gone[:12]}-0a0b0c0d"
        for sid, conversation in ((coding, gone), ("cc-lamu-session", kept), ("tool-session-1", None)):
            manager.create_sandbox(sid)
            manager.set_conversation(sid, conversation)
        result = sm.CommandResult(stdout="done", stderr="", return_code=3, blocked=True,
                                  block_reason="blocked: Zanzibar reason")
        manager.audit.log_command(coding, "echo Zanzibar secret > notes.txt", result)
        manager.audit.log_command("cc-lamu-session", "echo Lamu words", sm.CommandResult(stdout="x", return_code=0))
        manager.audit.log_command("tool-session-1", "echo Pemba words", sm.CommandResult(stdout="y", return_code=0))
        manager.audit.log_approval(coding, "approve", paths=["Zanzibar/plan.txt"], dest_dir="/home/Zanzibar",
                                   detail="Zanzibar copy-out")
        # A workspace runs for no conversation of its own: its rows keep the conversation it is bound to.
        manager.create_sandbox("ws-zanzibar")
        manager.set_binding("ws-zanzibar", gone)
        manager.audit.log_command("ws-zanzibar", "echo Zanzibar in the workspace",
                                  sm.CommandResult(stdout="w", return_code=0))
        with sqlite3.connect(str(path)) as conn:
            # Rows an older version wrote: of no conversation, the chat coding session's id still naming it.
            conn.execute("INSERT INTO sandbox_audit (session_id, timestamp, command, return_code) "
                         "VALUES (?, 1.0, 'echo Zanzibar older', 0)", (f"cc-{gone[:12]}-ffffffff",))
            conn.execute("INSERT INTO sandbox_audit (session_id, timestamp, command, return_code) "
                         "VALUES ('tool-session-2', 1.0, 'echo Malindi older', 0)")
        assert b"Zanzibar secret" in _bytes(path), "control: the command was audited"
        rows = {row["command"]: row for row in _audit_rows(path)}
        assert rows["echo Zanzibar secret > notes.txt"]["conversation_id"] == gone, (
            "a session's audit row does not keep its conversation")

        assert sm.forget_conversation(gone) == 4, "not every row of the conversation lost its words"
        held = _bytes(path)
        assert b"Zanzibar" not in held, "the deleted conversation's words stayed in the sandbox audit's file"
        assert b"Lamu words" in held and b"Pemba words" in held and b"Malindi older" in held, (
            "another conversation's or another session's rows lost their words")
        assert [row["command"] for row in _audit_rows(path) if row["session_id"] == "ws-zanzibar"] == [
            sm._FORGOTTEN_WORDS], "a workspace bound to the conversation kept its commands"
        forgotten = [row for row in _audit_rows(path) if row["session_id"].startswith(f"cc-{gone[:12]}-")]
        assert len(forgotten) == 2 and all(row["command"] == sm._FORGOTTEN_WORDS and row["block_reason"] == ""
                                           for row in forgotten), f"the words were not replaced: {forgotten}"
        record = next(row for row in forgotten if row["session_id"] == coding)
        assert (record["return_code"], record["blocked"], record["stdout_len"]) == (3, 1, 4), (
            f"the record of what ran was not kept: {record}")
        approval = manager.audit.get_approval_log(coding)[0]
        assert (approval["paths"], approval["dest_dir"], approval["detail"], approval["action"]) == (
            "[]", "", "", "approve"), f"the approval's words stayed, or its record went: {approval}"

        manager.audit.log_command(coding, "echo Zanzibar late", sm.CommandResult(stdout="z", return_code=0))
        manager.audit.log_approval(coding, "copy_out", paths=["Zanzibar/late.txt"], dest_dir="/home/Zanzibar/late",
                                   detail="Zanzibar late copy-out")
        held = _bytes(path)
        assert b"Zanzibar late" not in held and b"Zanzibar/late" not in held, (
            "a command or a copy-out of a forgotten conversation was recorded whole")
        late = [row for row in manager.audit.get_approval_log(coding) if row["action"] == "copy_out"]
        assert late and (late[0]["paths"], late[0]["detail"]) == ("[]", ""), f"the late copy-out kept words: {late}"
        sm.unforget_conversation(gone)
        manager.audit.log_command(coding, "echo Kilwa kept", sm.CommandResult(stdout="k", return_code=0))
        assert b"Kilwa kept" in _bytes(path), "a conversation handed back is still recorded without its words"
        # The user's wipe: every row loses its words, the record of what ran kept.
        assert sm.forget_every_word() >= 4, "the wipe forgot no word"
        held = _bytes(path)
        assert b"Lamu words" not in held and b"Pemba words" not in held and b"Kilwa kept" not in held, (
            "the wipe left words in the sandbox audit")
        assert all(row["return_code"] is not None for row in _audit_rows(path)), "the wipe took the record too"

        older = tmp_path / "older.db"
        with sqlite3.connect(str(older)) as conn:
            conn.execute(_OLD_AUDIT)
            conn.execute("INSERT INTO sandbox_audit (session_id, timestamp, command) VALUES (?, 1.0, ?)",
                         (f"cc-{gone[:12]}-12345678", "echo Zanzibar before"))
            conn.execute("INSERT INTO sandbox_audit (session_id, timestamp, command) VALUES ('w-1', 1.0, ?)",
                         ("echo Pemba before",))
        assert sm.AuditLog(str(older)).forget_conversation(gone) == 1, "an older file's row was not reached"
        assert b"Zanzibar before" not in _bytes(older) and b"Pemba before" in _bytes(older)
        with sqlite3.connect(str(older)) as conn:
            assert "conversation_id" not in {row[1] for row in conn.execute("PRAGMA table_info(sandbox_audit)")}, (
                "the forgetting changed the schema")
        sm.AuditLog(str(older), conversation_of=lambda sid: gone).log_command(
            "w-2", "echo Kilwa after", sm.CommandResult(stdout="", return_code=0))
        with sqlite3.connect(str(older)) as conn:
            after = conn.execute("SELECT conversation_id FROM sandbox_audit WHERE session_id = 'w-2'").fetchall()
        assert after == [(gone,)], f"an older log's schema step did not give its rows their conversation: {after}"
        missing = tmp_path / "missing.db"
        assert sm.AuditLog(str(missing)).forget_conversation(gone) == 0 and not missing.exists(), (
            "a log that was not there was created to be asked")
    finally:
        restore()


# ---------------------------------------------------------------------------
# DL6 -- the code sandbox and the workspace binding let go
# ---------------------------------------------------------------------------
class _Sandboxes:
    """A sandbox manager: the sessions made, the conversations given, the sessions destroyed and unbound."""

    def __init__(self, workspace):
        self.workspace = str(workspace)
        self.made, self.given, self.destroyed, self.bindings = [], [], [], []

    def create_sandbox(self, session_id=None, label=""):
        sid = session_id or f"sandbox-{len(self.made) + 1}"
        self.made.append(sid)
        return SimpleNamespace(session_id=sid)

    def set_conversation(self, session_id, conversation_id):
        self.given.append((session_id, conversation_id))

    def get_workspace_path(self, session_id):
        return self.workspace if session_id not in self.destroyed else None

    def destroy_sandbox(self, session_id):
        self.destroyed.append(session_id)

    def get_session(self, session_id):
        return SimpleNamespace(active=True, owner_user_id="local")

    def set_binding(self, session_id, conversation_id):
        self.bindings.append((session_id, conversation_id))


def test_dl6_the_code_sandbox_and_the_workspace_binding_let_go(tmp_path, caplog):
    asked = []
    stand_ins = {name: _module(name, forget_conversation=lambda cid, name=name: asked.append((name, cid)) or 1)
                 for name in ("opti_oignon.code_executor", "opti_oignon.sandbox_workspace")}
    world = _world(tmp_path, stores=stand_ins)
    try:
        gone = _conversation(world, "Zanzibar")
        assert world.manager.delete_conversation(gone)
        assert sorted(asked) == [("opti_oignon.code_executor", gone), ("opti_oignon.sandbox_workspace", gone)], (
            f"the code sandbox or the workspace binding was not asked: {asked}")
    finally:
        world.restore()
    world = _world(tmp_path / "unloaded")
    try:
        caplog.set_level(logging.WARNING)
        gone = _conversation(world, "Pemba")
        assert world.manager.delete_conversation(gone)
        said = caplog.text
        assert "no code sandbox in this tree" not in said and "no workspace binding in this tree" not in said, (
            "a store of this process's memory was loaded only to be asked")
        assert "no sandbox audit in this tree" not in said, "the sandbox manager was loaded only to be asked"
    finally:
        world.restore()

    tree = ast.parse(source("chat_coding_agent.py").read_text(encoding="utf-8"))
    ensure = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                   and node.name == "_ensure_sandbox"), None)
    assert ensure is not None, "the census reads no chat coding sandbox's creation"
    body = ast.unparse(ensure)
    assert body.index("create_sandbox(self._session_id)") < body.index(
        "attribute(self._session_id, self._conversation_id)"), (
        "a chat coding session's sandbox does not say its conversation, so its audit rows keep none")

    sandboxes = _Sandboxes(tmp_path)
    loaded, restore = isolate(targets={"opti_oignon.code_executor": source("code_executor.py")})
    try:
        ce = loaded["opti_oignon.code_executor"]
        executor = ce.code_executor
        executor._sandbox_mgr = sandboxes
        sid = executor._conversation_session(sandboxes, "conv-zanzibar")
        assert sandboxes.given == [(sid, "conv-zanzibar")], "the code sandbox does not say its conversation"
        assert ce.forget_conversation("conv-zanzibar") == 1 and sandboxes.destroyed == [sid], (
            "the deleted conversation's code sandbox was not destroyed")
        assert ce.forget_conversation("conv-zanzibar") == 0, "a sandbox already gone was destroyed again"
    finally:
        restore()

    sandboxes = _Sandboxes(tmp_path)
    loaded, restore = isolate(targets={"opti_oignon.sandbox_workspace": source("sandbox_workspace.py")},
                              seeded={"opti_oignon.sandbox_manager": _module(
                                  "opti_oignon.sandbox_manager", sandbox_manager=sandboxes)})
    try:
        sw = loaded["opti_oignon.sandbox_workspace"]
        assert sw.forget_conversation("conv-zanzibar") == 0, "a binding store never made held a binding"
        bindings = sw.get_workspace_bindings()
        bindings.bind("conv-zanzibar", "ws-1")
        bindings.bind("conv-lamu", "ws-2")
        assert sw.forget_conversation("conv-zanzibar") == 1, "the deleted conversation's workspace stayed bound"
        assert bindings.get_conversation_for("ws-1") is None and bindings.get_conversation_for("ws-2") == "conv-lamu"
        assert sandboxes.bindings[-1] == ("ws-1", None), "the workspace's session still names the conversation"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DL7-DL9 -- the sync feed
# ---------------------------------------------------------------------------
def _feed(tmp_path):
    loaded, restore = isolate(
        targets={"opti_oignon.veilid.records": source("veilid", "records.py"),
                 "opti_oignon.veilid.change_feed": source("veilid", "change_feed.py")},
        seeded={"opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain)},
        packages=("opti_oignon.veilid",))
    return loaded["opti_oignon.veilid.records"], loaded["opti_oignon.veilid.change_feed"], restore


def _version(records, record_id, clock, words, *, deleted=False, kind="conversation"):
    payload = {} if deleted else {"conversation": {"id": record_id, "messages": [words]}}
    return records.new_record(kind, record_id, payload, device="dev-a", clock=clock, deleted=deleted)


def _feed_rows(path):
    with sqlite3.connect(str(path)) as conn:
        return [tuple(row) for row in conn.execute(
            "SELECT seq, kind, record_id, clock, deleted FROM veilid_change_feed ORDER BY seq")]


def test_dl7_the_sync_feed_keeps_one_version_of_each_record_and_serves_the_same_deltas(tmp_path):
    records, cf, restore = _feed(tmp_path)
    try:
        feed = cf.ChangeFeed(root=tmp_path)
        appended = []
        fact = feed.record(_version(records, "zan", 1, "Zanzibar fact of another kind", kind="memory_canonical"))
        for record_id, clock, words in (("zan", 1, "Zanzibar first"), ("lamu", 1, "Lamu first"),
                                        ("zan", 2, "Zanzibar second"), ("pemba", 1, "Pemba first"),
                                        ("zan", 3, "Zanzibar third"), ("lamu", 2, "Lamu second")):
            appended.append((feed.record(_version(records, record_id, clock, words)), record_id, clock))
        batch = feed.record_many([_version(records, "pemba", 2, "Pemba second"),
                                  _version(records, "pemba", 3, "Pemba third")])
        appended += [(batch[0], "pemba", 2), (batch[1], "pemba", 3)]
        path = tmp_path / "veilid_change_feed.db"
        rows = _feed_rows(path)
        assert [(r[1], r[2], r[3]) for r in rows] == [("memory_canonical", "zan", 1), ("conversation", "zan", 3),
                                                      ("conversation", "lamu", 2), ("conversation", "pemba", 3)], (
            f"the feed keeps more than one version of a record, or erased another kind's: {rows}")
        assert rows[0][0] == fact, "a record of another kind under the same id was erased"
        feed._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        held = _bytes(path)
        assert b"Zanzibar first" not in held and b"Lamu first" not in held and b"Pemba second" not in held, (
            "a superseded version's words stayed in the feed's file")
        assert b"Zanzibar third" in held and b"Pemba third" in held, "control: the latest versions are kept"
        top = appended[-1][0]
        for watermark in range(top + 1):
            latest = {("memory_canonical", "zan"): (fact, 1)}
            for seq, record_id, clock in appended:
                latest[("conversation", record_id)] = (seq, clock)
            want = sorted((kind, record_id, clock) for (kind, record_id), (seq, clock) in latest.items()
                          if seq > watermark)
            delta = feed.since(watermark)
            got = sorted((r.kind.value, r.record_id, r.clock) for r in delta.records)
            assert got == want, f"the delta after {watermark} is not the whole journal's: {got} != {want}"
            assert delta.high_water == top, f"the high-water moved: {delta.high_water} != {top}"
        assert feed.current_clock("conversation", "zan") == 3, "a record's clock went back"
        feed.record(_version(records, "kilwa", 9, "Kilwa ninth"))
        feed.record(_version(records, "kilwa", 4, "Kilwa fourth, appended out of order"))
        assert feed.current_clock("conversation", "kilwa") == 9, (
            "an append out of order erased a version of a higher clock: the record's clock went back")
    finally:
        restore()


def test_dl8_a_tombstone_erases_every_version_of_its_record_and_empties_the_log(tmp_path, caplog):
    records, cf, restore = _feed(tmp_path)
    try:
        feed = cf.ChangeFeed(root=tmp_path)
        path = tmp_path / "veilid_change_feed.db"
        feed.record(_version(records, "zan", 1, "Zanzibar first words"))
        feed.record(_version(records, "zan", 2, "Zanzibar second words"))
        feed.record(_version(records, "lamu", 1, "Lamu words"))
        feed.record(_version(records, "zan", 3, "", deleted=True))
        assert [(r[2], r[4]) for r in _feed_rows(path)] == [("lamu", 0), ("zan", 1)], (
            f"a tombstone left its record's versions: {_feed_rows(path)}")
        held = _bytes(path)
        assert b"Zanzibar" not in held, "a deleted record's words stayed in the feed's file or its log"
        assert b"Lamu words" in held, "control: another record is kept"
        feed.record(_version(records, "kilwa", 9, "Kilwa ninth words"))
        feed.record(_version(records, "kilwa", 4, "Kilwa fourth words, out of order"))
        feed.record(_version(records, "kilwa", 6, "", deleted=True))
        assert [(r[2], r[3], r[4]) for r in _feed_rows(path) if r[2] == "kilwa"] == [("kilwa", 6, 1)], (
            "a tombstone left a version of a higher clock")
        assert b"Kilwa" not in _bytes(path), "a deleted record's version of a higher clock kept its words"
        feed.record(_version(records, "chole", 9, "Chole ninth words"))
        feed.record(_version(records, "chole", 4, "Chole fourth words, out of order"))
        feed.record_many([_version(records, "chole", 6, "", deleted=True)])
        assert [(r[3], r[4]) for r in _feed_rows(path) if r[2] == "chole"] == [(6, 1)], (
            "a batch's tombstone left a version of a higher clock")
        assert b"Chole" not in _bytes(path), "a batch's deleted record kept its words"

        own_wait = feed._connection.execute("PRAGMA busy_timeout").fetchone()[0]
        assert own_wait == 300, f"control: the contract's connections wait 300 ms: {own_wait}"
        feed.record(_version(records, "pemba", 1, "Pemba first words"))
        keeper = _plain(path)
        keeper.execute("BEGIN")
        keeper.execute("SELECT COUNT(*) FROM veilid_change_feed").fetchall()
        caplog.set_level(logging.WARNING)
        feed.record_many([_version(records, "pemba", 2, "", deleted=True)])
        assert "keeps the write-ahead log" in caplog.text, "a log a read kept was not said"
        keeper.rollback()
        keeper.close()
        feed.record(_version(records, "mafia", 1, "Mafia words"))
        feed.record(_version(records, "mafia", 2, "", deleted=True))
        assert b"Pemba first" not in _bytes(path), "the next deletion did not empty the log a read had kept"
        assert feed._connection.execute("PRAGMA busy_timeout").fetchone()[0] == own_wait, (
            "emptying the log changed how long the feed's connection waits")
    finally:
        restore()


def test_dl9_the_feeds_pass_erases_what_an_older_build_left_and_creates_nothing(tmp_path, caplog):
    records, cf, restore = _feed(tmp_path)
    try:
        old = tmp_path / "old"
        old.mkdir()
        path = old / "veilid_change_feed.db"
        older = cf.ChangeFeed(root=old)
        older.record(_version(records, "lamu", 1, "Lamu kept"))
        conn = sqlite3.connect(str(path))
        try:
            # As an older build appended: every version kept.
            for clock, words in ((1, "Zanzibar first"), (2, "Zanzibar second"), (3, "Zanzibar third")):
                version = _version(records, "zan", clock, words)
                conn.execute(cf._INSERT, cf.ChangeFeed._row_params(version, "2026-01-01T00:00:00+00:00"))
            conn.commit()
        finally:
            conn.close()
        older.close()
        feed = cf.ChangeFeed(root=old)
        assert feed.tend() == 2, "the pass did not erase the versions an older build left"
        assert [(r[2], r[3]) for r in _feed_rows(path)] == [("lamu", 1), ("zan", 3)], _feed_rows(path)
        held = _bytes(path)
        assert b"Zanzibar first" not in held and b"Zanzibar second" not in held, (
            "a superseded version's words stayed in the feed's file")
        assert feed.tend() == 0, "a second pass erased something"
        conn = sqlite3.connect(str(path))
        try:
            # An older build's out-of-order append: a version of a higher clock before one of a lower clock.
            for clock, words in ((9, "Kilwa ninth"), (4, "Kilwa fourth")):
                version = _version(records, "kilwa", clock, words)
                conn.execute(cf._INSERT, cf.ChangeFeed._row_params(version, "2026-01-01T00:00:00+00:00"))
            conn.commit()
        finally:
            conn.close()
        assert feed.tend() == 0 and feed.current_clock("conversation", "kilwa") == 9, (
            "the pass erased a version of a higher clock: the record's clock went back")
        conn = sqlite3.connect(str(path))
        try:
            # An older build's tombstone after a version of a higher clock, and two versions of one clock.
            for record_id, clock, words, deleted in (("mafia", 9, "Mafia ninth words", False), ("mafia", 6, "", True),
                                                     ("pemba", 5, "Pemba first of five", False),
                                                     ("pemba", 5, "Pemba second of five", False)):
                version = _version(records, record_id, clock, words, deleted=deleted)
                conn.execute(cf._INSERT, cf.ChangeFeed._row_params(version, "2026-01-01T00:00:00+00:00"))
            conn.commit()
        finally:
            conn.close()
        assert feed.tend() == 2, "the pass kept a version a later tombstone or a version of the same clock superseded"
        held = _bytes(path)
        assert b"Mafia ninth" not in held and b"Pemba first of five" not in held, (
            "a superseded version's words stayed after the pass")
        assert b"Pemba second of five" in held, "control: the latest version of the same clock is kept"

        empty = tmp_path / "none"
        empty.mkdir()
        assert cf.ChangeFeed(root=empty).tend() == 0 and not (empty / "veilid_change_feed.db").exists(), (
            "the pass created a feed it found missing")
        broken = SimpleNamespace(tend=lambda: (_ for _ in ()).throw(OSError("disk gone")))
        cf.set_change_feed(broken)
        caplog.set_level(logging.WARNING)
        assert cf.tend() == 0 and "could not be erased" in caplog.text, "a pass that failed raised, or said nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DL10 -- the deferred ledger: skills only, overwritten
# ---------------------------------------------------------------------------
def test_dl10_the_deferred_ledger_holds_skills_only_and_overwrites_what_it_removes(tmp_path):
    tree = ast.parse(source("veilid", "sync_engine.py").read_text(encoding="utf-8"))
    kinds = next((node.value for node in ast.walk(tree) if isinstance(node, ast.AnnAssign)
                  and getattr(node.target, "id", "") == "SENSITIVE_KINDS"), None)
    assert kinds is not None, "the census reads no SENSITIVE_KINDS"
    assert ast.unparse(kinds) == "frozenset({RecordKind.SKILL.value})", (
        f"the ledger may now hold a conversation's, a fact's or a note's record: {ast.unparse(kinds)}")
    loaded, restore = isolate(
        targets={"opti_oignon.veilid.records": source("veilid", "records.py"),
                 "opti_oignon.veilid.reconcile": source("veilid", "reconcile.py"),
                 "opti_oignon.veilid.deferred_ledger": source("veilid", "deferred_ledger.py")},
        seeded={"opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain)},
        packages=("opti_oignon.veilid",))
    try:
        records, ledger_module = loaded["opti_oignon.veilid.records"], loaded["opti_oignon.veilid.deferred_ledger"]
        ledger = ledger_module.DeferredLedger(root=tmp_path)
        skill = records.new_record("skill", "zan-skill", {"skill": {"body": "Zanzibar skill body"}},
                                   device="peer", clock=1)
        assert ledger.offer(skill, peer_id="peer-1") == "inserted"
        path = tmp_path / ledger_module.DB_FILENAME
        assert b"Zanzibar skill" in _bytes(path), "control: the envelope was kept"
        assert ledger.remove("skill", "zan-skill")
        ledger._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert b"Zanzibar skill" not in _bytes(path), "a removed envelope stayed in the ledger's free pages"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DL11 -- an offline turn is told the truth and kept nowhere
# ---------------------------------------------------------------------------
def test_dl11_an_offline_turn_is_told_the_truth_and_kept_nowhere(tmp_path):
    text = source("executor.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not any(call.endswith(".enqueue") for call in calls), "the executor still queues a turn"
    assert "sync_queue" not in text, "the executor still reaches the offline queue"
    notice = next((node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(getattr(t, "id", "") == "_OFFLINE_NOTICE" for t in node.targets)), None)
    assert notice is not None, "the executor has no notice for an offline turn"
    said = ast.literal_eval(notice)
    assert "not sent" in said and "queued" not in said and "processed" not in said, f"the notice says: {said!r}"
    statuses = re.findall(r'status\("(\[!\] Ollama offline[^"]*)"\)', text)
    assert statuses and set(statuses) == {"[!] Ollama offline -- request not sent"}, statuses
    agentic = source("agentic_executor.py").read_text(encoding="utf-8")
    assert "request queued" not in agentic and "[!] Ollama offline -- request not sent" in agentic, (
        "the agentic path still says a turn was queued")

    loaded, restore = isolate(targets={"opti_oignon.sync_queue": source("sync_queue.py")},
                              seeded={"opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain)})
    try:
        sq = loaded["opti_oignon.sync_queue"]
        path = tmp_path / "queue.db"
        queue = sq.SyncQueue(db_path=path)
        queue.enqueue("Zanzibar question asked offline")
        queue.enqueue("Pemba question asked offline")
        assert b"Zanzibar question" in _bytes(path), "control: an older version's turn was queued"
        assert queue.empty() == 2
        held = _bytes(path)
        assert b"Zanzibar question" not in held and b"Pemba question" not in held, (
            "an emptied turn's words stayed in the queue's free pages")
        missing = tmp_path / "missing.db"
        assert sq.SyncQueue(db_path=missing).empty() == 0 and not missing.exists(), (
            "the emptying created the queue it found missing")
    finally:
        restore()


# ---------------------------------------------------------------------------
# DL12 -- the executor's cache put carries its conversation
# ---------------------------------------------------------------------------
def test_dl12_the_executors_cache_put_carries_its_conversation():
    root = REPO / "opti_oignon"
    puts = {}
    for path in sorted({Path(path) for path in package_modules().values()}):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in ("_response_cache.put", "response_cache.put"):
                puts.setdefault(path.relative_to(root).as_posix(), []).append(node)
    assert {"executor.py", "semantic_cache.py"} <= set(puts), f"the census reads no put of the response cache: {puts}"
    for module, calls in sorted(puts.items()):
        for put in calls:
            given = {keyword.arg: ast.unparse(keyword.value) for keyword in put.keywords}
            assert given.get("conversation_id") == "conversation_id", (
                f"{module}: a put of the response cache does not carry its conversation: {ast.unparse(put)}")


# ---------------------------------------------------------------------------
# DL13 -- the boot erases the feed's superseded versions and the queue's turns
# ---------------------------------------------------------------------------
def test_dl13_the_boot_erases_the_feeds_superseded_versions_and_the_offline_queues_turns():
    tree = ast.parse(source("api", "app.py").read_text(encoding="utf-8"))
    lifespan = next((node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
                     and any(isinstance(sub, ast.Yield) for sub in ast.walk(node))
                     and "_feed.tend()" in ast.unparse(node)), None)
    assert lifespan is not None, "the server's life does not tend the sync feed"
    body = ast.unparse(lifespan)
    assert "from opti_oignon.veilid import change_feed as _feed" in body, "the boot does not reach the sync feed"
    assert "from opti_oignon.sync_queue import sync_queue as _offline" in body, "the boot does not reach the queue"
    assert "_offline.empty()" in body, "the boot does not empty the offline queue"
    assert body.index("_feed.tend()") < body.index("yield") and body.index("_offline.empty()") < body.index("yield"), (
        "the feed's pass or the queue's emptying is not part of the boot")


# ---------------------------------------------------------------------------
# DL14-DL16 -- the account wipe
# ---------------------------------------------------------------------------
class _OldStore:
    """The old memory store: its rows, emptied whole."""

    def __init__(self, path, rows):
        self.db_path = path
        path.write_bytes(b"")
        self.rows = list(rows)
        self.emptied = 0

    def held(self):
        return list(self.rows)

    def clear_all(self):
        count, self.rows = len(self.rows), []
        return count

    def empty_log(self):
        self.emptied += 1
        return True


def _wipe_world(tmp_path, *, single=True, auth=True, old=None, sandbox=None, semantic=None):
    stores, packages = {}, ()
    if semantic is not None:
        stores["opti_oignon.semantic_cache"] = semantic
    if old is not None:
        stores["opti_oignon.memory.legacy"] = _module("opti_oignon.memory.legacy", memory_manager=old)
        packages = ("opti_oignon.memory",)
    if sandbox is not None:
        stores["opti_oignon.sandbox_manager"] = sandbox
    return _world(tmp_path, stores=stores, wipe=True, single=single, auth=auth, packages=packages)


def _filled(world):
    cache = world.mod("response_cache").response_cache
    queue = world.mod("sync_queue").sync_queue
    ids = [_conversation(world, word) for word in ("Zanzibar", "Lamu", "Pemba")]
    for cid, word in zip(ids, ("Zanzibar", "Lamu", "Pemba")):
        assert cache.put("m", "sys", f"{word}?", f"{word} cached answer.", conversation_id=cid)
    assert cache.put("m", "sys", "outside?", "Mombasa cached outside any conversation.")
    assert queue.enqueue("Malindi asked offline long ago")
    return ids


def test_dl14_the_single_user_wipe_deletes_every_conversation_through_its_deletion(tmp_path):
    forgotten, emptied = [], []
    sandbox = _module("opti_oignon.sandbox_manager", forget_conversation=lambda cid: forgotten.append(cid) or 0,
                      unforget_conversation=lambda cid: None, forget_every_word=lambda: forgotten.append("*") or 0)
    semantic = _module("opti_oignon.semantic_cache",
                       semantic_cache=SimpleNamespace(empty=lambda: emptied.append("semantic") or 0),
                       forget_conversation=lambda cid: 0, unforget_conversation=lambda cid: None)
    old = _OldStore(tmp_path / "memories.db", ["Kilwa old memory"])
    world = _wipe_world(tmp_path, old=old, sandbox=sandbox, semantic=semantic)
    try:
        udm = world.mod("user_data_manager")
        ids = _filled(world)
        answer = udm.UserDataDeleter().delete_all("local")
        assert answer["conversations"] == 3, f"not every conversation was deleted: {answer['conversations']}"
        assert world.manager.list_conversations() == [], "a conversation stayed"
        assert sorted(cid for cid in forgotten if cid != "*") == sorted(ids) and "*" in forgotten, (
            "a conversation was deleted without its deletion, or the audit's words stayed")
        held = _bytes(world.data / "response_cache.db") + _bytes(world.data / "sync_queue.db")
        for word in (b"Zanzibar", b"Lamu", b"Pemba", b"Mombasa", b"Malindi"):
            assert word not in held, f"{word!r} stayed in the stores of the user's requests"
        assert old.rows == [] and old.emptied == 1, "the old memory store was not emptied"
        assert emptied == ["semantic"], "the semantic cache was not emptied"
        assert udm._the_one_user("u-1") and not udm._the_one_user("no-such-user"), (
            "a single-user install's one account is not its one user, or another name is")
        assert not set(udm.WIPE_SINGLE_USER) & set(answer["not_covered"]), (
            f"the answer names stores the wipe removed: {sorted(set(udm.WIPE_SINGLE_USER) & set(answer['not_covered']))}")
        assert "notes and their updates" in answer["not_covered"], "the answer no longer names what stays"
        assert "sandbox audit: the record of what ran, without its words" in answer["retained_by_design"]
    finally:
        world.restore()


def test_dl15_a_multi_user_wipe_touches_no_install_wide_store_and_names_them(tmp_path):
    for case, options, asked in (("multi-user", {"single": False}, "local"), ("unreadable", {"auth": False}, "local"),
                                 ("not the one user", {}, "no-such-user")):
        world = _wipe_world(tmp_path / case, **options)
        try:
            udm = world.mod("user_data_manager")
            ids = _filled(world)
            answer = udm.UserDataDeleter().delete_all(asked)
            assert answer["conversations"] == 0 and len(world.manager.list_conversations()) == len(ids), (
                f"{case}: an install-wide conversation was deleted")
            held = _bytes(world.data / "response_cache.db") + _bytes(world.data / "sync_queue.db")
            assert b"Lamu cached" in held and b"Malindi asked" in held, f"{case}: an install-wide store was emptied"
            assert set(udm.WIPE_SINGLE_USER) <= set(answer["not_covered"]), (
                f"{case}: the answer does not name what it left: "
                f"{sorted(set(udm.WIPE_SINGLE_USER) - set(answer['not_covered']))}")
        finally:
            world.restore()


def test_dl16_a_refused_deletion_stops_the_single_user_wipe_and_says_how_many_are_left(tmp_path):
    state = SimpleNamespace(seen=0)

    def forget(conversation_id):
        state.seen += 1
        if state.seen == 2:
            raise OSError("the audit log is locked")
        return 0

    sandbox = _module("opti_oignon.sandbox_manager", forget_conversation=forget,
                      unforget_conversation=lambda cid: None, forget_every_word=lambda: 0)
    world = _wipe_world(tmp_path, sandbox=sandbox)
    try:
        udm = world.mod("user_data_manager")
        _filled(world)
        answer = udm.UserDataDeleter().delete_all("local")
        assert answer["conversations"] == 1, f"the wipe went on past a refusal: {answer['conversations']}"
        left = [label for label in answer["not_covered"] if label.startswith("conversations (")]
        assert left == ["conversations (2 left: a deletion was refused, sandbox audit: OSError)"], (
            f"the answer does not say how many are left, and why: {left}")
        assert len(world.manager.list_conversations()) == 2, "control: two conversations stayed"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DL17 -- the census
# ---------------------------------------------------------------------------
_FORGOTTEN_BY = {
    "conversation.py": "the conversation store: its messages go with the conversation's row",
    "memory/onion_store.py": "the onion: conversation._forget_onion",
}
_KEPT = {
    "analytics.py": "how each turn performed: ids, models, timings and counts, never a turn's words",
    "context_ledger.py": "what each window held, counted: numbers and labels, never a turn's words",
    "memory/legacy.py": "the old store's memories: moved into the facts, then the store emptied, its pages "
                        "overwritten; its conversation column is the panel's open conversation, which cannot tell a "
                        "fact drawn from it from one written by hand, and nothing reads it into a window",
    "veilid/deferred_ledger.py": "what a peer offered that waits for approval, under (kind, record_id): only the "
                                 "sensitive kinds, skills, are deferred -- never a conversation's record",
}
_KEYS = {"conversation", "conversation_id", "conv_id", "source_conversation_id"}
# Records of any kind under their kind and id: a conversation's among them.
_GENERIC = {"kind", "record_id"}


def _texts(tree):
    """The strings a module spells: its constants, and its f-strings with each placeholder as ``{}``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            yield "".join(part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else "{}"
                          for part in node.values)


def _census():
    """Each module's tables keyed by a conversation."""
    root = REPO / "opti_oignon"
    tables = {}
    for path in sorted({Path(path) for path in package_modules().values()}):
        relative = path.relative_to(root).as_posix()
        for text in _texts(ast.parse(path.read_text(encoding="utf-8"))):
            for match in re.finditer(r"CREATE TABLE(?: IF NOT EXISTS)?\s+([\w{}]+)\s*\((.*?)\)\s*(?:;|$)", text,
                                     re.S | re.I):
                columns = {part.strip().split()[0].lower().strip("()") for part in match.group(2).split(",")
                           if part.strip().split()}
                if columns & _KEYS or _GENERIC <= columns:
                    tables.setdefault(relative, set()).add(match.group(1))
    return tables


def test_dl17_every_table_keyed_by_a_conversation_is_forgotten_or_named(tmp_path):
    world = _world(tmp_path)
    try:
        registry = {module.split(".", 1)[1].replace(".", "/") + ".py" for _store, module, _f, _w in
                    world.conv._FORGETTERS}
        words = {module for _store, module, _f, holds in world.conv._FORGETTERS if holds}
    finally:
        world.restore()
    tables = _census()
    # The census finds what it must, or it proves nothing.
    assert {"branches", "branch_messages"} <= tables.get("conversation_branches.py", set()), tables
    assert "veilid/change_feed.py" in tables, f"the census reads no f-string, or no generic key: {sorted(tables)}"
    assert "response_cache" in tables.get("response_cache.py", set()), (
        "the response cache keeps no conversation, so no deletion reaches its answers")
    assert {"sandbox_audit", "sandbox_approval_audit"} <= tables.get("sandbox_manager.py", set()), (
        "the sandbox audit keeps no conversation, so no deletion reaches its words")
    unaccounted = {module: sorted(found) for module, found in tables.items()
                   if module not in registry and module not in _FORGOTTEN_BY and module not in _KEPT}
    assert unaccounted == {}, f"tables keyed by a conversation that a deletion neither forgets nor names: {unaccounted}"
    assert sorted(set(_KEPT) & registry) == [], "named as kept and forgotten too"
    assert {"opti_oignon.response_cache", "opti_oignon.sandbox_manager", "opti_oignon.veilid.change_feed"} <= words, (
        "a store of the conversation's words is asked after the commit: one that cannot forget would not refuse")


# ---------------------------------------------------------------------------
# DL18 -- the loader says an offline turn was not sent
# ---------------------------------------------------------------------------
def test_dl18_the_loader_says_an_offline_turn_was_not_sent():
    import test_chat_loader_contracts as loader

    mapped = [(status, words) for status, words in loader._MAPPED if not status.startswith("[!] Ollama offline")]
    mapped.append(("[!] Ollama offline -- request not sent", "Ollama is offline: not sent"))
    unmapped = [*loader._UNMAPPED, "[!] Ollama offline -- request queued"]
    out = loader._node("words", {
        "sample": loader._SAMPLE,
        "ranAs": list(loader._RAN_AS) + ["some_new_code"],
        "statuses": [status for status, _ in mapped] + unmapped,
        "replays": {"failed": loader._failed_run(), "unmapped": loader._unmapped_run()},
    })
    statuses = out["statuses"]
    said = statuses[:len(mapped)]
    assert said == [words for _, words in mapped], (
        "each status the server sends is said in the table's words:\n"
        + "\n".join(f"{status!r} -> {got!r}" for (status, _), got in zip(mapped, said)))
    assert statuses[len(mapped):] == [None] * len(unmapped), (
        f"a status the server no longer sends, or never did, has words: {statuses[len(mapped):]}")
    every = json.dumps(out["tables"]["WORDS"])
    assert "Queued" not in every and "queued" not in every, "the loader still says a turn was queued"


# ---------------------------------------------------------------------------
# DL19-DL23 -- the executor's goldens, v2: each cache put carries its conversation, an offline turn is queued nowhere
# ---------------------------------------------------------------------------
# The digests of the v2 golden files, written by ``tests/_onion_off_golden.py --write`` and
# ``tests/_tool_path_off_golden.py --write``: v1 with every put of the response cache carrying its conversation and the
# offline cases' turns told they were not sent, never queued; nothing else differs.
_ONION_OFF_V2 = "c7694eb12355cf50a8ffce928137d446994231fe0e7d68545beff63ebb4d5884"
_TOOL_PATH_OFF_V2 = "20a67e46dd31d621b8c3aa294eede57f02d96b13d3e58443d29693d891fef83e"
# The onion-off matrix's digest: its offline cases now fire when each turn ends offline and reaches no queue.
_ONION_OFF_MATRIX_V2 = "70adb201b8e2339b19a6f85eebeea415e488f2dddc7149b783d6fd5df735ce3e"


def _onion_off(monkeypatch):
    import test_executor_onion_off_golden_contracts as suite

    monkeypatch.setattr(suite, "_GOLDEN_SHA256", _ONION_OFF_V2)
    monkeypatch.setattr(suite, "_MATRIX_SHA256", _ONION_OFF_MATRIX_V2)
    assert suite.golden.FIXTURE.relative_to(REPO).as_posix() == "tests/executor_golden/v2/onion_off.json"
    return suite


def _tool_path_off(monkeypatch):
    import test_tool_path_off_golden_contracts as suite

    monkeypatch.setattr(suite, "_GOLDEN_SHA256", _TOOL_PATH_OFF_V2)
    assert suite._FIXTURE.relative_to(REPO).as_posix() == "tests/executor_golden/v2/tool_path_off.json"
    return suite


def test_dl19_the_isolated_window_sends_the_v2_golden_requests(monkeypatch):
    """jw13a of tests/test_executor_onion_off_golden_contracts.py, word for word, on the v2 golden."""
    _onion_off(monkeypatch).test_jw13a_the_isolated_window_sends_the_golden_requests_and_the_wiring_contracts_own()


def test_dl20_the_integrated_window_sends_the_v2_golden_requests(monkeypatch):
    """jw13b of tests/test_executor_onion_off_golden_contracts.py, word for word, on the v2 golden."""
    _onion_off(monkeypatch).test_jw13b_the_integrated_window_sends_the_golden_requests_and_reaches_its_seams_as_it_did()


def test_dl21_the_v2_golden_options_leave_the_llama_server_heads_with_no_max_tokens(monkeypatch):
    """jw13e of tests/test_executor_onion_off_golden_contracts.py, word for word, on the v2 golden."""
    _onion_off(monkeypatch).test_jw13e_the_golden_options_leave_the_llama_server_heads_with_no_max_tokens()


def test_dl22_every_tool_path_case_does_what_the_v2_golden_holds(monkeypatch):
    """jw33a of tests/test_tool_path_off_golden_contracts.py, word for word, on the v2 golden."""
    _tool_path_off(monkeypatch).test_jw33a_every_case_sends_admits_asks_saves_and_yields_what_the_golden_holds()


def test_dl23_a_check_on_an_absent_v2_golden_file_fails_by_name_and_creates_nothing(tmp_path, monkeypatch, capsys):
    """jw33d of tests/test_tool_path_off_golden_contracts.py, its assertions kept, the v2 file named."""
    suite = _tool_path_off(monkeypatch)
    golden, fixture = suite.golden, suite._FIXTURE
    captured = dict(suite._capture())
    monkeypatch.setattr(golden, "capture_in_children", lambda *args, **kwargs: dict(captured))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    missing = tmp_path / "executor_golden" / "v2" / fixture.name
    monkeypatch.setattr(golden, "FIXTURE", missing)
    assert golden.main(["--check"]) == 1, "a check on an absent golden file did not fail"
    assert capsys.readouterr().out.splitlines()[-1:] == ["golden: ABSENT"], "the check did not name the absence"
    assert list(tmp_path.iterdir()) == [], "a check created something where the golden file would be"
    assert golden.compare(fixture.read_text(encoding="ascii"), missing) == "absent"
    with pytest.raises(AssertionError, match="the golden file is missing"):
        suite._pinned(missing)
    # Controls: the same check on the golden file in the tree agrees, and another text is a difference.
    monkeypatch.setattr(golden, "FIXTURE", fixture)
    assert golden.main(["--check"]) == 0 and capsys.readouterr().out.splitlines()[-1:] == ["golden: same"], (
        "control: a check on the golden file in the tree agrees")
    text = fixture.read_text(encoding="ascii")
    assert golden.compare(text + "\n") == "differs", "control: a different text is a difference"
    assert fixture.relative_to(REPO).as_posix() == "tests/executor_golden/v2/tool_path_off.json"


# ---------------------------------------------------------------------------
# DL24 -- an offline turn's ledger record, told the truth (supersedes e5)
# ---------------------------------------------------------------------------
def test_dl24_an_offline_turn_records_its_outcome_without_generation_and_queues_nothing():
    """e5 of tests/test_executor_ledger_emission_contracts.py, its assertions kept, the turn told it was not sent."""
    import test_executor_ledger_emission_contracts as ledger

    queued = []
    nm = types.ModuleType("opti_oignon.network_manager")
    nm.network_manager = SimpleNamespace(is_online=False)
    sq = types.ModuleType("opti_oignon.sync_queue")
    sq.sync_queue = SimpleNamespace(enqueue=lambda query, task_type, model: queued.append(query))
    mod, scripted, recorder, restore = ledger._load(
        extra_seeded={"opti_oignon.network_manager": nm, "opti_oignon.sync_queue": sq})
    try:
        ex = mod.Executor()
        chunks, (refined, response) = ledger._drive(ex.execute("What is a monoid?", ledger._routing(), refine=False))
        assert "not sent" in response and "queued" not in response, response
        assert queued == [], "an offline turn was queued"
        assert scripted.calls == []
        assert len(recorder.records) == 1
        assert recorder.records[0]["outcome"] == "offline_queued"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DL25 -- the admission between the plan and the composition (supersedes jw6)
# ---------------------------------------------------------------------------
def test_dl25_the_admission_stands_between_the_plan_and_the_composition_and_an_offline_turn_is_queued_nowhere():
    """jw6 of tests/test_executor_onion_window_contracts.py, its assertions kept, an offline turn reaching no queue."""
    import test_executor_onion_window_contracts as suite

    w, golden, oj = suite.w, suite.golden, suite.oj
    world, close = w.open_world(gov={}, projects=True)
    try:
        records = w.run_turns(world, suite._TWO)
    finally:
        close()
    for index, record in enumerate(records):
        kinds = w.kinds(record)
        places = [suite._first(kinds, kind) for kind in suite._ORDER]
        assert None not in places and places == sorted(places), (index, list(zip(suite._ORDER, places)))
        assert [kind for kind in kinds if kind in suite._ORDER] == list(suite._ORDER), (index, kinds)
        (admitted,) = suite._events(record, "admit")
        assert admitted["requested_ctx"] == world.journal.rung and admitted["requested_ctx"] in suite._LADDER
        assert kinds.count("admit") == 1 and kinds.count("compose") == 1, index
    world, close = w.open_world(gov={}, journal={"plan_refusal": ("turn", (900, 512))})
    try:
        (record,) = w.run_turns(world, suite._TWO[:1])
    finally:
        close()
    assert "plan" in w.kinds(record) and "admit" not in w.kinds(record), "a plan refused is never admitted"
    assert "compose" not in w.kinds(record)
    world, close = w.open_world(gov={}, offline=True)
    try:
        (record,) = w.run_turns(world, suite._TWO[:1])
    finally:
        close()
    kinds = w.kinds(record)
    assert kinds.count("network.online") == 1 and kinds.count("network.enqueue") == 0, kinds
    assert kinds.index("plan") < kinds.index("network.online"), kinds
    assert not {"admit", "compose", "cache.get", "stream"} & set(kinds), kinds
    assert [row["outcome"] for row in world.rows] == ["offline_queued"], world.rows
    assert record["yields"] and record["yields"][0].startswith("[Offline]"), record["yields"]
    assert "not sent" in record["yields"][0], record["yields"]

    # A descent: the context admitted under the plan's rung, composed and sent under it.
    world, close = w.open_world(gov={"num_ctx": (8192, 16384)})
    try:
        records = w.run_turns(world, suite._TWO)
    finally:
        close()
    assert [suite._events(record, "compose")[0]["num_ctx"] for record in records] == [8192, 16384]
    assert [call["options"]["num_ctx"] for call in w.sent(world)] == [8192, 16384]
    assert [(row["win_plan_rung"], row["win_num_ctx"], row["win_descent"]) for row in world.rows] == [
        (16384, 8192, True), (16384, 16384, False)]

    # The same with the onion itself: the descent laid over a shadow, the conversation's state untouched, the
    # next plan back at the rung.
    history = {"conv-1": golden.history_of(10, size=450)}
    world, close = w.open_real(gov={"num_ctx": (8192, 16384)}, history=history)
    try:
        lib = world.librarian
        composing = world.executor_module._onion_compose_window
        digests = []

        def digesting(plan, *, num_ctx=None):
            state = lib._states["conv-1"]
            before = suite._state_digest(state)
            window = composing(plan, num_ctx=num_ctx)
            digests.append((num_ctx, before, suite._state_digest(state), window))
            return window

        world.executor_module._onion_compose_window = digesting
        records = w.run_turns(world, suite._TWO)
        rows = world.rows
        assert [(row["outcome"], row["win_plan_rung"], row["win_num_ctx"], row["win_descent"]) for row in rows] == [
            ("completed", 16384, 8192, True), ("completed", 16384, 16384, False)], rows
        assert all(row["win_plan_rung"] in suite._LADDER for row in rows)
        assert [suite._events(record, "admit")[0]["requested_ctx"] for record in records] == [16384, 16384]
        (descent, before, after, window), (_ctx, _before, _after, whole) = digests
        assert descent == 8192 and before == after, "the composition of a descent leaves the state as it was"
        assert len(window.messages) < len(whole.messages) - 2, "control: the descent laid the Flesh over a shadow"
        state = lib._states["conv-1"]
        held = suite._state_digest(state)
        oj.bare(state, 2)
        assert suite._state_digest(state) != held, "control: an eviction moves the digest"
    finally:
        close()


# ---------------------------------------------------------------------------
# DL26 -- a deletion erases its record's versions in the feed before its commit
# ---------------------------------------------------------------------------
class _LockedEngine:
    """A sync engine whose every publish fails, as a feed another connection holds locked would."""

    def __init__(self):
        self.asked = []

    def current_clock(self, kind, record_id):
        return 0

    def publish_conversation(self, conversation_id, *args, **kwargs):
        self.asked.append(("conversation", conversation_id))
        raise OSError("database is locked")

    def publish_memory_canonical(self, fact_id, *args, **kwargs):
        self.asked.append(("memory_canonical", fact_id))
        raise OSError("database is locked")


def _feed_world(tmp_path):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    engine = _LockedEngine()
    seeded = {
        "opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain),
        "opti_oignon.db_encryption": _module("opti_oignon.db_encryption", get_encrypted_connection=_plain,
                                             SQLCIPHER_AVAILABLE=False),
        "opti_oignon.config": _module("opti_oignon.config", DATA_DIR=data),
        "opti_oignon.run_journal": _module("opti_oignon.run_journal", forget_conversation=lambda cid: None),
        "opti_oignon.veilid.guard": _module("opti_oignon.veilid.guard", veilid_available=lambda: True),
        "opti_oignon.veilid.sync_engine": _module("opti_oignon.veilid.sync_engine", get_sync_engine=lambda: engine),
    }
    targets = {
        "opti_oignon.veilid.records": source("veilid", "records.py"),
        "opti_oignon.veilid.change_feed": source("veilid", "change_feed.py"),
        "opti_oignon.memory.canonical_store": source("memory", "canonical_store.py"),
        "opti_oignon.response_cache": source("response_cache.py"),
        CONV: source("conversation.py"),
    }
    loaded, restore = isolate(targets=targets, seeded=seeded, packages=("opti_oignon.veilid", "opti_oignon.memory"))
    try:
        manager = loaded[CONV].ConversationManager(db_path=data / "conversations.db", publish_to_sync=False)
    except BaseException:
        restore()
        raise
    return SimpleNamespace(loaded=loaded, restore=restore, conv=loaded[CONV], manager=manager, data=data,
                           engine=engine, records=loaded["opti_oignon.veilid.records"],
                           cf=loaded["opti_oignon.veilid.change_feed"],
                           cs=loaded["opti_oignon.memory.canonical_store"])


def _journal(world, kind, record_id, *words):
    feed = world.cf.get_change_feed()
    for clock, said in enumerate(words, start=1):
        payload = {"text": said} if kind == "memory_canonical" else {"conversation": {"id": record_id,
                                                                                       "messages": [said]}}
        feed.record(world.records.new_record(kind, record_id, payload, device="dev-a", clock=clock))


def test_dl26_a_deletion_erases_its_records_versions_in_the_sync_feed_before_its_commit(tmp_path, caplog):
    world = _feed_world(tmp_path)
    try:
        path = world.data / "veilid_change_feed.db"
        first = _conversation(world, "Malindi")
        assert world.manager.delete_conversation(first) and not path.exists(), (
            "a deletion created a sync feed it found missing")
        gone, kept, locked = (_conversation(world, word) for word in ("Zanzibar", "Lamu", "Pemba"))
        _journal(world, "conversation", gone, "Zanzibar first words", "Zanzibar second words")
        _journal(world, "conversation", kept, "Lamu words")
        _journal(world, "conversation", locked, "Pemba words")
        assert b"Zanzibar second" in _bytes(path), "control: the conversation's versions were journalled"
        assert world.manager.delete_conversation(gone), "control: the deletion went through"
        rows = {row[2] for row in _feed_rows(path)}
        assert gone not in rows and kept in rows, f"the deleted conversation's versions stayed in the feed: {rows}"
        assert b"Zanzibar" not in _bytes(path), "the deleted conversation's words stayed in the feed or its log"

        keeper = _plain(path)
        keeper.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(world.conv.ConversationNotForgotten) as refused:
                world.manager.delete_conversation(locked)
        finally:
            keeper.rollback()
            keeper.close()
        assert "sync change feed" in str(refused.value), f"the refusal does not name the feed: {refused.value}"
        assert world.manager.get_conversation(locked) is not None, "the conversation went with the refusal"
        assert world.manager.delete_conversation(locked), "once the feed can be written, the deletion goes through"
        assert b"Pemba" not in _bytes(path), "the deletion asked again left the words in the feed"

        store = world.cs.CanonicalMemoryStore()
        fact = store.add("Kilwa ruins face the sea.", "fact", source="manual")
        _journal(world, "memory_canonical", fact.id, "Kilwa ruins face the sea.", "Kilwa ruins face the sea twice.")
        assert b"Kilwa ruins" in _bytes(path), "control: the fact's versions were journalled"
        assert store.erase(fact.id), "control: the fact was erased"
        assert ("memory_canonical", fact.id) in world.engine.asked, "control: its tombstone was asked and failed"
        assert b"Kilwa" not in _bytes(path), "an erased fact's words stayed in the feed when its tombstone failed"

        kept_fact = store.add("Mafia reefs shelter turtles.", "fact", source="manual")
        _journal(world, "memory_canonical", kept_fact.id, "Mafia reefs shelter turtles.")
        keeper = _plain(path)
        keeper.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(sqlite3.OperationalError):
                store.erase(kept_fact.id)
        finally:
            keeper.rollback()
            keeper.close()
        assert store.get(kept_fact.id) is not None, "a fact whose feed versions could not be erased was erased"
        broken = _module("opti_oignon.veilid.change_feed")

        def missing(name):
            raise ModuleNotFoundError("No module named 'opti_oignon.veilid.records'", name="opti_oignon.veilid.records")

        broken.__getattr__ = missing
        held_feed = sys.modules["opti_oignon.veilid.change_feed"]
        sys.modules["opti_oignon.veilid.change_feed"] = broken
        try:
            with pytest.raises(ModuleNotFoundError):
                store.erase(kept_fact.id)
        finally:
            sys.modules["opti_oignon.veilid.change_feed"] = held_feed
        assert store.get(kept_fact.id) is not None, "a sync package missing a module was taken for no sync package"
        for word, drop in (("Chake", lambda fid: store.hard_delete(fid)), ("Wete", lambda fid: store.clear())):
            other = store.add(f"{word} market sells cloves.", "fact", source="manual")
            _journal(world, "memory_canonical", other.id, f"{word} market sells cloves.")
            assert word.encode() in _bytes(path), "control: the fact's version was journalled"
            assert drop(other.id), f"control: the fact went ({word})"
            assert word.encode() not in _bytes(path), (
                f"a fact's versions stayed in the feed when its tombstone failed ({word})")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DL27 -- the semantic cache's emptying
# ---------------------------------------------------------------------------
def test_dl27_the_semantic_caches_emptying_overwrites_and_raises_when_it_cannot(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    loaded, restore = isolate(
        targets={"opti_oignon.semantic_cache": source("semantic_cache.py")},
        seeded={"opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain),
                "opti_oignon.config": _module("opti_oignon.config", DATA_DIR=data)})
    try:
        sc = loaded["opti_oignon.semantic_cache"]
        path = tmp_path / "semantic.db"
        cache = sc.SemanticCache(db_path=path)
        conn = cache._get_connection()
        assert conn is not None, "control: the semantic cache opens its file"
        try:
            conn.execute("INSERT INTO semantic_embeddings (cache_key, model, query_text, embedding, embedding_dim, "
                         "created_at) VALUES ('k', 'm', 'Zanzibar question embedded', x'00', 1, 1.0)")
            columns = [row[1] for row in conn.execute("PRAGMA table_info(cache_entries)")]
            values = {column: "Zanzibar cached answer" if column in ("response", "query") else 1
                      for column in columns}
            conn.execute(f"INSERT INTO cache_entries ({', '.join(values)}) VALUES ({', '.join('?' for _ in values)})",
                         list(values.values()))
            conn.commit()
        finally:
            conn.close()
        assert b"Zanzibar" in _bytes(path), "control: the cache holds the words"
        assert cache.empty() == 1
        assert b"Zanzibar" not in _bytes(path), "an emptied cache's words stayed in its free pages"
        missing = tmp_path / "missing.db"
        assert sc.SemanticCache(db_path=missing).empty() == 0 and not missing.exists(), (
            "an emptying created the cache it found missing")
        embedded, put = [], []
        cache.store_embedding = lambda **kwargs: embedded.append(kwargs["query_text"]) or True
        answers = SimpleNamespace(put=lambda **kwargs: put.append(kwargs.get("conversation_id")) or "k1")
        assert cache.put_with_embedding(answers, "m", "sys", "Zanzibar asked in c1", "a",
                                        conversation_id="c1") == ("k1", False)
        assert cache.put_with_embedding(answers, "m", "sys", "Lamu asked outside", "a") == ("k1", True)
        assert put == ["c1", None] and embedded == ["Lamu asked outside"], (
            f"a conversation's request was embedded, or its answer cached without it: {put} {embedded}")
        keeper = _plain(path)
        keeper.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(sqlite3.OperationalError):
                cache.empty()
        finally:
            keeper.rollback()
            keeper.close()
    finally:
        restore()


# ---------------------------------------------------------------------------
# DL28 -- the tombstone out-clocks every version the deletion erased, and a copy coming back loses
# ---------------------------------------------------------------------------
class _FeedEngine:
    """A sync engine that mints and appends as the real one does: the next clock from the feed, the record journalled;
    with ``failing``, every publish fails, as a locked feed would make it."""

    def __init__(self, world):
        self.world, self.failing = world, False

    def current_clock(self, kind, record_id):
        return self.world.cf.get_change_feed().current_clock(kind, record_id)

    def _publish(self, kind, record_id, payload, clock, deleted, updated_at):
        if self.failing:
            raise OSError("database is locked")
        self.world.cf.get_change_feed().record(self.world.records.new_record(
            kind, record_id, payload or {}, device="dev-a", clock=clock, deleted=deleted, updated_at=updated_at))

    def publish_conversation(self, conversation_id, payload, *, clock, deleted=False, updated_at=""):
        self._publish("conversation", conversation_id, payload, clock, deleted, updated_at)

    def publish_memory_canonical(self, fact_id, payload, *, clock, deleted=False, updated_at=""):
        self._publish("memory_canonical", fact_id, payload, clock, deleted, updated_at)


def test_dl28_the_tombstone_out_clocks_every_erased_version_and_a_copy_coming_back_loses(tmp_path):
    data = tmp_path / "data"
    data.mkdir(parents=True)
    world = SimpleNamespace()
    engine = _FeedEngine(world)
    seeded = {
        "opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain),
        "opti_oignon.db_encryption": _module("opti_oignon.db_encryption", get_encrypted_connection=_plain,
                                             SQLCIPHER_AVAILABLE=False),
        "opti_oignon.config": _module("opti_oignon.config", DATA_DIR=data),
        "opti_oignon.run_journal": _module("opti_oignon.run_journal", forget_conversation=lambda cid: None),
        "opti_oignon.veilid.guard": _module("opti_oignon.veilid.guard", veilid_available=lambda: True,
                                            assert_sync_allowed=lambda: None),
        "opti_oignon.veilid.sync_engine": _module("opti_oignon.veilid.sync_engine", get_sync_engine=lambda: engine),
    }
    targets = {
        "opti_oignon.veilid.records": source("veilid", "records.py"),
        "opti_oignon.veilid.reconcile": source("veilid", "reconcile.py"),
        "opti_oignon.veilid.change_feed": source("veilid", "change_feed.py"),
        "opti_oignon.veilid.protocol": source("veilid", "protocol.py"),
        "opti_oignon.memory.canonical_store": source("memory", "canonical_store.py"),
        "opti_oignon.response_cache": source("response_cache.py"),
        CONV: source("conversation.py"),
    }
    loaded, restore = isolate(targets=targets, seeded=seeded, packages=("opti_oignon.veilid", "opti_oignon.memory"))
    try:
        world.cf, world.records = loaded["opti_oignon.veilid.change_feed"], loaded["opti_oignon.veilid.records"]
        protocol = loaded["opti_oignon.veilid.protocol"]
        manager = loaded[CONV].ConversationManager(db_path=data / "conversations.db")
        feed = world.cf.get_change_feed()

        def batch(*records):
            return protocol.RecordBatch(device="dev-b", high_water=0, records=list(records))

        def latest(kind, record_id):
            return [r for r in feed.current_records() if r.kind.value == kind and r.record_id == record_id]

        gone = manager.create_conversation(title="about Zanzibar").id
        manager.add_message(gone, "user", "Zanzibar is where the user grew up.")
        manager.add_message(gone, "assistant", "Zanzibar sounds lovely.")
        (held,) = latest("conversation", gone)
        assert held.clock >= 2 and not held.deleted, f"control: the saves were journalled: {held}"
        assert manager.delete_conversation(gone)
        (tomb,) = latest("conversation", gone)
        assert tomb.deleted and tomb.clock > held.clock, (
            f"the tombstone does not out-clock the version it deleted: {tomb.clock} <= {held.clock}")
        result = protocol.apply_local_batch(feed, batch(held))
        assert result.applied == 0 and latest("conversation", gone)[0].deleted, (
            "a paired device's copy of the deleted conversation came back")
        assert feed.clock_floor("conversation", gone) == 0, "the tombstone landed and the floor stayed"

        engine.failing = True
        kept = manager.create_conversation(title="about Lamu").id
        engine.failing = False
        manager.add_message(kept, "user", "Lamu is where the user grew up.")
        (copy,) = latest("conversation", kept)
        engine.failing = True
        assert manager.delete_conversation(kept), "control: a failing tombstone never fails the deletion"
        assert latest("conversation", kept) == [], "control: the versions were erased, the tombstone failed"
        assert feed.current_clock("conversation", kept) == copy.clock, (
            "a deletion whose tombstone failed sent the record's clock back")
        engine.failing = False
        result = protocol.apply_local_batch(feed, batch(copy))
        assert result.applied == 0 and latest("conversation", kept) == [], (
            "a copy at the erased clock came back while the tombstone could not be written")
        later = world.records.new_record("conversation", kept, {"conversation": {"id": kept}}, device="dev-b",
                                         clock=copy.clock + 1)
        assert protocol.apply_local_batch(feed, batch(later)).applied == 1, (
            "control: a paired device's later edit wins, as the guide says")
        assert feed.clock_floor("conversation", kept) == 0, "a version above the floor landed and the floor stayed"
        feed.record_many([world.records.new_record("note", "n-1", {"text": "Pemba note"}, device="dev-a", clock=c)
                          for c in (1, 2)])
        assert feed.forget_record("note", "n-1") == 1 and feed.clock_floor("note", "n-1") == 2
        feed.record_many([world.records.new_record("note", "n-1", {}, device="dev-a", clock=3, deleted=True)])
        assert feed.clock_floor("note", "n-1") == 0, "a batch's version above the floor landed and the floor stayed"

        store = loaded["opti_oignon.memory.canonical_store"].CanonicalMemoryStore()
        fact = store.add("Kilwa ruins face the sea.", "fact", source="manual")
        store.update(fact.id, text="Kilwa ruins face the Indian Ocean.")
        (version,) = latest("memory_canonical", fact.id)
        assert store.erase(fact.id)
        (erased,) = latest("memory_canonical", fact.id)
        assert erased.deleted and erased.clock > version.clock, (
            f"an erased fact's tombstone does not out-clock its version: {erased.clock} <= {version.clock}")
        assert protocol.apply_local_batch(feed, batch(version)).applied == 0, "an erased fact came back"
    finally:
        restore()
