#!/usr/bin/env python3
"""Contracts for the facts a deletion leaves: the old store, a conversation's facts, a deleted fact.

The window carried the old memory store's block whenever the facts gave nothing, and the old store was never
emptied after its migration: a fact deleted in the facts came back from it, through that block or through a forced
second migration, and right after an account wipe the window carried the old memories again. A conversation's
deletion forgot none of the facts the model drew from it, and the capture did not say which conversation it drew
from. A deleted fact stayed, its text kept with no end, and the facts' store left an erased fact's words in its free
pages. The old store is now moved, then emptied; a deletion erases the facts drawn from its conversation; a deleted
fact waits in a trash for its days, then is erased for good; every erasure overwrites and tells the peers.

  * FT1 -- THE WINDOW NEVER CARRIES THE OLD STORE'S BLOCK: with no fact placed, the composer raising, or the
    composer missing, the turn has no memory block, and the old store is never asked.
  * FT2 -- A DELETION ERASES THE FACTS THE MODEL DREW FROM THE CONVERSATION -- extracted or captured, in service or
    in the trash, a peer's own -- their vectors with them, the peers told, their words gone from the file and its
    log; the facts the user made theirs, and another conversation's, stay.
  * FT3 -- A CAPTURE WRITES UNDER ITS CONVERSATION, and one landing after the deletion is refused; one writing when
    the deletion comes is erased with the others, its vector too; a refused deletion hands the conversation back;
    the capture's watermark is let go.
  * FT4 -- THE OLD STORE MOVES (supersedes test_memory_migration's test_migrates_all_facts): each old fact lands
    under legacy-import, its category mapped, one out of service in the trash, a duplicate merged; each old row is
    deleted once its fact is in, overwritten; a fact whose add fails keeps its row and the marker waits, until the
    next run finishes.
  * FT5 -- AN OLD STORE MIGRATED BEFORE IS EMPTIED, NOTHING ADDED, forced or not (supersedes test_memory_migration's
    test_force_ignores_marker): a fact the user deleted since never comes back; an old fact out of service that
    landed in service goes to the trash, unless the user adopted it since.
  * FT6 -- AN OLD STORE THAT CANNOT BE READ IS NEVER TAKEN FOR AN EMPTY ONE: no marker, no row deleted, said.
  * FT7 -- A DELETED FACT IS IN THE TRASH, through the panel's routes: out of service, listed with the day it is
    erased, restorable, erased at once on request, or all of them; "trash" is never taken for a fact's id, and a
    fact in service is never erased from there.
  * FT8 -- THE TRASH ERASES WHAT IT HELD PAST ITS DAYS, every user's: overwritten, the peers told; a fact restored
    meanwhile, or in service, stays.
  * FT9 -- THE TRASH'S DAYS COME FROM THE YAML, a whole number within bounds; anything else keeps 30, said.
  * FT10 -- THE TRASH IS TENDED AT BOOT AND EVERY HOUR, off the event loop, and the pass is cancelled at shutdown.
  * FT11 -- EVERY CONNECTION OF THE FACTS OVERWRITES WHAT IT FREES: an edit's old text, a fact deleted.
  * FT12 -- A FACTS STORE THAT CANNOT FORGET REFUSES THE DELETION BY NAME, and takes the conversation's facts again;
    once it can, the deletion goes through. A process that finds no facts creates none, nor does a start's migration
    or trash's pass, and it still refuses a fact drawn from the conversation after.
  * FT13 -- THE CENSUS (supersedes fe12 of tests/test_conversation_forgets_everything_contracts.py): every table of
    the package keyed by a conversation is forgotten at a deletion or named with its reason, the old store named as
    emptied; every source that names a conversation is one the facts' forgetting erases.
  * FT14 -- THE DASHBOARD COUNTS THE FACTS IN SERVICE, in the store the window reads, never the old store.
  * FT15 -- A PEER'S DELETION ERASES THE DRAWN FACTS TOO.
  * FT16 -- A TREE WITHOUT THE MEMORY PACKAGE PASSES THE FACTS, said: a module's package missing is the module
    missing.

Local-only (the public distribution ships no tests). conversation.py, the canonical store, the memory store, the
trash, the old store, the migration and the capture loaded from source in one shared isolation window, every store
in tmp, on connections that do not overwrite deleted pages by default, as the cipher's driver does not (sqlcipher3:
secure_delete 0, probed 26 October); the vector layer, the extraction, the sync engine and the onion stand-ins.
"""

import ast
import hashlib
import logging
import re
import sqlite3
import sys
import threading
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_census import modules as package_modules  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

CONV = "opti_oignon.conversation"
EXECUTOR = "opti_oignon.executor"
BUDGET_S = {
    "test_ft1_the_window_never_carries_the_old_stores_block": 8.0,
    "test_ft2_a_deletion_erases_the_facts_the_model_drew_from_the_conversation": 3.0,
    "test_ft3_a_capture_writes_under_its_conversation_and_one_landing_after_the_deletion_is_refused": 5.0,
    "test_ft4_the_old_store_moves_each_row_deleted_once_its_fact_is_in": 3.0,
    "test_ft5_an_old_store_migrated_before_is_emptied_nothing_added": 3.0,
    "test_ft6_an_old_store_that_cannot_be_read_is_never_taken_for_an_empty_one": 3.0,
    "test_ft7_a_deleted_fact_is_in_the_trash_through_the_panels_routes": 5.0,
    "test_ft8_the_trash_erases_what_it_held_past_its_days_and_nothing_else": 3.0,
    "test_ft9_the_trash_days_come_from_the_yaml_within_bounds": 3.0,
    "test_ft10_the_trash_is_tended_at_boot_and_every_hour": 3.0,
    "test_ft11_every_connection_of_the_facts_overwrites_what_it_frees": 3.0,
    "test_ft12_a_facts_store_that_cannot_forget_refuses_the_deletion_by_name": 3.0,
    "test_ft13_every_table_keyed_by_a_conversation_is_forgotten_or_named": 3.0,
    "test_ft14_the_dashboard_counts_the_facts_in_service_never_the_old_store": 3.0,
    "test_ft15_a_peers_deletion_erases_the_drawn_facts_too": 3.0,
    "test_ft16_a_tree_without_the_memory_package_passes_the_facts": 3.0,
    "test_ft17_a_fact_the_user_made_theirs_stays_when_its_conversation_goes": 5.0,
    "test_ft18_a_peers_erasure_drops_this_devices_vector_and_a_forgotten_conversations_fact_is_refused": 3.0,
    "test_ft19_a_restore_racing_the_forgetting_leaves_no_vector_and_the_trash_keeps_its_clock": 5.0,
    "test_ft20_a_peers_drawn_fact_never_lands_after_its_conversations_deletion": 3.0,
    "test_ft21_a_process_that_never_opened_the_vectors_owes_them_to_the_trash": 3.0,
}
# The memory modules loaded from source, in an order where each finds what it imports at its first use.
_MEMORY = ("canonical_store", "dedup", "trash", "legacy", "migration", "auto_capture")
# A peer's deletion made after every write of these contracts.
_LATER = "2099-01-01T00:00:00"


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


class _Vectors:
    """The vector layer: each fact's text by its id, as the layer keeps it. ``hold`` (a word, an event) makes the
    add of a text holding the word wait for the event; ``refuse`` (an id) makes its deletion raise; ``near`` (an id)
    is the near duplicate every search finds, as a vector the layer kept for it would be."""

    def __init__(self):
        self.items = {}
        self.hold = None
        self.entered = threading.Event()
        self.refuse = None
        self.near = None
        self.embed_hold = None
        self.embedding = threading.Event()

    def embed(self, text):
        if self.embed_hold and self.embed_hold[0] in text:
            self.embedding.set()
            assert self.embed_hold[1].wait(5), "the held embedding was never released"
        return [float(len(text)), 1.0]

    def find_similar(self, embedding, **_kwargs):
        return [SimpleNamespace(id=self.near, similarity=0.99)] if self.near else []

    def add(self, fact_id, text, *, embedding=None, user_id=None, category=None, source=None):
        if self.hold and self.hold[0] in text:
            self.entered.set()
            assert self.hold[1].wait(5), "the held add was never released"
        self.items[fact_id] = text
        return fact_id

    def update(self, fact_id, *, text=None, **_kwargs):
        if text is not None and fact_id in self.items:
            self.items[fact_id] = text
        return True

    def delete(self, fact_id, *, user_id=None):
        if self.refuse == fact_id:
            raise RuntimeError("the vector layer is down")
        return self.items.pop(fact_id, None) is not None

    def get(self, fact_id):
        return {"id": fact_id, "document": self.items[fact_id]} if fact_id in self.items else None


class _Feed:
    """The sync engine: what was published, fact by fact."""

    def __init__(self):
        self.published = []

    def current_clock(self, kind, key):
        return 0

    def publish_memory_canonical(self, fact_id, payload, *, clock, deleted, updated_at):
        self.published.append((fact_id, deleted))


def _world(tmp_path, *, vectors=None, routes=False, memory=True, health=None):
    """conversation.py and the memory modules over ``tmp_path``; with ``routes``, the memory routes and their
    schemas; with ``health`` (the old store's stand-in), the health routes; without ``memory``, no memory package."""
    data = tmp_path / "data"
    data.mkdir(parents=True)
    layer = vectors if vectors is not None else _Vectors()
    feed = _Feed()
    asked = SimpleNamespace(onion=[], extracted=[], refused=[], hold=None)

    def extract_and_store(messages, *, source="", **_kwargs):
        if asked.hold is not None:
            assert asked.hold.wait(5), "the held capture was never released"
        store = sys.modules["opti_oignon.memory.dedup"].get_memory_store()
        asked.extracted.append(source)
        results = []
        for message in messages:
            try:
                results.append(store.add(message["content"], "fact", source=source))
            except Exception as exc:  # noqa: BLE001 - as the extraction does: one refused fact never stops the rest
                asked.refused.append((source, type(exc).__name__))
        return results

    seeded = {
        "opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain),
        "opti_oignon.db_encryption": _module("opti_oignon.db_encryption", get_encrypted_connection=_plain,
                                             SQLCIPHER_AVAILABLE=False),
        "opti_oignon.config": _module("opti_oignon.config", DATA_DIR=data),
        "opti_oignon.run_journal": _module("opti_oignon.run_journal", forget_conversation=lambda cid: None),
        "opti_oignon.veilid.guard": _module("opti_oignon.veilid.guard", veilid_available=lambda: True),
        "opti_oignon.veilid.records": _module("opti_oignon.veilid.records",
                                              RecordKind=SimpleNamespace(MEMORY_CANONICAL="memory_canonical")),
        "opti_oignon.veilid.sync_engine": _module("opti_oignon.veilid.sync_engine", get_sync_engine=lambda: feed),
    }
    targets = {}
    packages = ["opti_oignon.veilid"]
    if memory:
        packages.append("opti_oignon.memory")
        # As the real module: the layer is open (``_vector_store``) once it was asked for.
        vectors_module = _module("opti_oignon.memory.vector_store", _vector_store=None)

        def get_vector_store():
            vectors_module._vector_store = layer
            return layer

        vectors_module.get_vector_store = get_vector_store
        seeded["opti_oignon.memory.vector_store"] = vectors_module
        seeded["opti_oignon.memory.librarian"] = _module(
            "opti_oignon.memory.librarian",
            forget_conversation=lambda conversation_id, config=None: asked.onion.append(conversation_id) or 0)
        seeded["opti_oignon.memory.extraction"] = _module("opti_oignon.memory.extraction",
                                                          extract_and_store=extract_and_store)
        for name in _MEMORY:
            targets[f"opti_oignon.memory.{name}"] = source("memory", f"{name}.py")
        targets["opti_oignon.pending_writes"] = source("pending_writes.py")
    targets[CONV] = source("conversation.py")
    if routes or health is not None:
        packages.append("opti_oignon.api")
        targets["opti_oignon.api.schemas"] = source("api", "schemas.py")
    if routes:
        seeded["opti_oignon.api.deps"] = _module("opti_oignon.api.deps", MEMORY_AVAILABLE=True, memory_manager=None)
        targets["opti_oignon.api.routes_memory"] = source("api", "routes_memory.py")
    if health is not None:
        flags = ("ARTIFACT_AVAILABLE", "BENCHMARK_AVAILABLE", "CODE_EXECUTOR_AVAILABLE", "CONTEXT_WINDOW_AVAILABLE",
                 "CONVERSATION_AVAILABLE", "EXECUTOR_AVAILABLE", "MODEL_WARMUP_AVAILABLE", "PIPELINE_AVAILABLE",
                 "PRESET_AVAILABLE", "RESPONSE_CACHE_AVAILABLE", "SEMANTIC_CACHE_AVAILABLE")
        deps = _module("opti_oignon.api.deps", MEMORY_AVAILABLE=True, memory_manager=health, conversation_manager=None,
                       executor=None, model_warmup=None, perf_benchmark_runner=None, response_cache=None,
                       **{flag: False for flag in flags})
        seeded["opti_oignon.api.deps"] = deps
        targets["opti_oignon.api.routes_health"] = source("api", "routes_health.py")
    loaded, restore = isolate(targets=targets, seeded=seeded, packages=tuple(packages))
    try:
        conv = loaded[CONV]
        manager = conv.ConversationManager(db_path=data / "conversations.db", publish_to_sync=False)
    except BaseException:
        restore()
        raise
    return SimpleNamespace(loaded=loaded, restore=restore, conv=conv, manager=manager, data=data, vectors=layer,
                           feed=feed, asked=asked, mod=lambda name: loaded[f"opti_oignon.{name}"],
                           store=lambda: loaded["opti_oignon.memory.dedup"].get_memory_store())


def _conversation(world, word):
    cid = world.manager.create_conversation(title=f"about {word}").id
    world.manager.add_message(cid, "user", f"{word} is where the user grew up.")
    return cid


def _bytes(path):
    return b"".join(p.read_bytes() for p in path.parent.glob(path.name + "*") if p.is_file())


def _keeper(path):
    """Another connection to a store, idle: the store's write-ahead log outlives the store's own connections."""
    conn = _plain(path)
    conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchall()
    return conn


def _rows(path):
    with sqlite3.connect(str(path)) as conn:
        return {row[0]: (row[1], row[2], bool(row[3])) for row in
                conn.execute("SELECT id, text, source, active FROM memory_facts")}


def _peer_fact(world, fact_id, text, source, *, active=True, owner="bob", updated_at=None):
    """A fact a peer brought, kept with the owner it came with."""
    stamp = updated_at or datetime.now(timezone.utc).isoformat()
    canonical = world.mod("memory.canonical_store").get_canonical_store()
    assert canonical.apply_synced_memory_canonical(fact_id, {"user_id": owner, "fact": {
        "id": fact_id, "text": text, "category": "fact", "source": source, "created_at": stamp, "updated_at": stamp,
        "active": active}}), "control: the peer's fact was applied"


def _join_captures():
    for thread in threading.enumerate():
        if thread.name == "oo-auto-capture":
            thread.join(5)


# ---------------------------------------------------------------------------
# FT1 -- the window never carries the old store's block
# ---------------------------------------------------------------------------
class _Scripted:
    def chat(self, **kwargs):
        return iter([{"message": {"content": "Hello"}, "done": True}])


def _memory_context(composer):
    """The working block the real executor lays for a turn, ``composer`` standing in for the unified composer
    (None: missing); the old store offered where the old executor reached for it. Returns (block, label, asked)."""
    asked = []
    legacy = _module("opti_oignon.memory.memory_manager",
                     format_for_prompt=lambda max_tokens=500: asked.append(max_tokens) or "[User Memory]\n- Zanzibar")
    retrieval = _module("opti_oignon.memory.retrieval")
    if composer is not None:
        retrieval.build_memory_block = lambda question, **kwargs: composer(question)[0]
        retrieval.working_memory_block = lambda question, **kwargs: composer(question)[0]
        retrieval.compose_memory_block = lambda question, **kwargs: composer(question)
    cfg = _module("opti_oignon.config", config=SimpleNamespace(get_model=lambda *a, **k: "m:1b",
                                                               get_temperature=lambda *a, **k: 0.2))
    seeded = {
        "opti_oignon.config": cfg,
        "opti_oignon.router": _module("opti_oignon.router", RoutingResult=type("RoutingResult", (), {})),
        "opti_oignon.memory.retrieval": retrieval,
        "opti_oignon.memory.memory_manager": legacy,
    }
    seed_registry(seeded, _Scripted())
    had, prev = "ollama" in sys.modules, sys.modules.get("ollama")
    sys.modules["ollama"] = _module("ollama", chat=_Scripted().chat)
    try:
        loaded, restore = isolate(
            targets={"opti_oignon.agent.untrusted_context": source("agent", "untrusted_context.py"),
                     "opti_oignon.context_dedup": source("context_dedup.py"),
                     EXECUTOR: source("executor.py")},
            blocked=["opti_oignon.memory.librarian"],
            seeded=seeded,
            packages=("opti_oignon.agent", "opti_oignon.memory"),
        )
        try:
            block, label = loaded[EXECUTOR].Executor()._compose_memory_context_labelled("Where?", "conv-1")
        finally:
            restore()
    finally:
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)
    return block, label, asked


def test_ft1_the_window_never_carries_the_old_stores_block():
    block, _label, asked = _memory_context(lambda question: ("- The user drinks green tea.", [("f1", True)]))
    assert "green tea" in block and asked == [], "control: a working block is laid, and the old store not asked"

    def raising(question):
        raise RuntimeError("the composer broke")

    for name, composer in (("no fact placed", lambda question: ("", [])), ("the composer raising", raising),
                           ("the composer missing", None)):
        block, label, asked = _memory_context(composer)
        assert block == "" and "Zanzibar" not in block, f"{name}: a block stood in for an empty one: {block!r}"
        assert label == ([], []), f"{name}: an empty block carries no label: {label!r}"
        assert asked == [], f"{name}: the old store was asked for its block"


# ---------------------------------------------------------------------------
# FT2 -- a deletion erases the facts the model drew from the conversation
# ---------------------------------------------------------------------------
def test_ft2_a_deletion_erases_the_facts_the_model_drew_from_the_conversation(tmp_path):
    world = _world(tmp_path)
    try:
        store = world.store()
        gone = _conversation(world, "Zanzibar")
        kept = _conversation(world, "Kilimanjaro")
        drawn = [store.add("Zanzibar harbour stories filled every childhood evening.", "fact",
                           source=f"extract:{gone}")[0],
                 store.add("Spring sailing trips along Zanzibar beaches.", "fact", source=f"auto-capture:{gone}")[0],
                 store.add("A jar of Zanzibar cloves sits on the kitchen shelf.", "fact",
                           source=f"extract:{gone}")[0]]
        assert store.soft_delete(drawn[2].id), "control: one drawn fact is in the trash"
        _peer_fact(world, "peer-1", "Zanzibar ferries remind Bob of home.", f"extract:{gone}")
        stays = {"manual": "Mombasa markets open early on Sundays.", "agent": "Lamu has no cars at all.",
                 "accepted:p1": "Malindi beaches face the Indian Ocean.", "legacy-import": "Nairobi sits on a plateau.",
                 f"extract:{kept}": "Kilimanjaro snow is melting fast.",
                 f"extract:{gone}x": "Arusha coffee grows on volcanic soil."}
        for source_, text in stays.items():
            store.add(text, "fact", source=source_)
        path = world.data / "memory_facts.db"
        assert b"Zanzibar harbour" in _bytes(path), "control: the facts were written"
        keeper = _keeper(path)
        assert world.manager.delete_conversation(gone) is True
        rows = _rows(path)
        assert sorted(source_ for _text, source_, _active in rows.values()) == sorted(stays), (
            f"the deletion left a drawn fact, or erased one of the user's: {sorted(rows.values())}")
        erased = [record.id for record in drawn] + ["peer-1"]
        assert not set(erased) & set(world.vectors.items), "a drawn fact's vector stayed"
        assert all(text in world.vectors.items.values() for text in stays.values()), "a kept fact lost its vector"
        tombstones = {fact_id for fact_id, deleted in world.feed.published if deleted}
        assert set(erased) <= tombstones, f"the peers were not told of every erasure: {tombstones}"
        held = _bytes(path)
        assert b"Zanzibar" not in held, "an erased fact's words stayed in the file or its log"
        assert b"Mombasa" in held, "control: the kept facts are read from the same bytes"
        assert world.asked.onion == [gone], "control: the onion was asked"
        keeper.close()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# FT3 -- a capture writes under its conversation; one landing after the deletion is refused
# ---------------------------------------------------------------------------
def _typed(text):
    return [{"role": "user", "content": text, "origin": "typed", "segments": []}]


def test_ft3_a_capture_writes_under_its_conversation_and_one_landing_after_the_deletion_is_refused(
        tmp_path, monkeypatch):
    monkeypatch.setenv("OPTI_AUTO_CAPTURE", "1")
    world = _world(tmp_path)
    try:
        capture, dedup = world.mod("memory.auto_capture"), world.mod("memory.dedup")
        store = world.store()
        path = world.data / "memory_facts.db"
        cid = _conversation(world, "Zanzibar")
        assert capture.maybe_capture(cid, _typed("Zanzibar spice markets thrill the user."), min_new=1) is True
        _join_captures()
        assert world.asked.extracted == [f"auto-capture:{cid}"], (
            f"the capture did not write under its conversation: {world.asked.extracted}")
        assert [s for _t, s, _a in _rows(path).values()] == [f"auto-capture:{cid}"], "control: the fact was written"
        assert cid in capture._capture_watermark, "control: the capture keeps its watermark"
        # A capture on its way when the deletion comes: drawn already, written after.
        world.asked.hold = threading.Event()
        assert capture.maybe_capture(cid, _typed("x") + _typed("Pemba reefs shelter turtles."), min_new=1) is True
        assert world.manager.delete_conversation(cid) is True
        assert cid not in capture._capture_watermark, "the deleted conversation's watermark was kept"
        world.asked.hold.set()
        _join_captures()
        world.asked.hold = None
        assert ("auto-capture:" + cid, "ConversationForgotten") in world.asked.refused, (
            f"a capture landing after the deletion was taken: {world.asked.refused}")
        assert _rows(path) == {}, f"a fact drawn from the deleted conversation is kept: {_rows(path)}"
        # A write already holding the store when the deletion comes is erased with the others, its vector too.
        other = _conversation(world, "Pemba")
        world.vectors.hold = ("Pemba", threading.Event())
        writer = threading.Thread(target=store.add, args=("Pemba forests hide flying foxes.", "fact"),
                                  kwargs={"source": f"extract:{other}"})
        writer.start()
        assert world.vectors.entered.wait(5), "control: the write holds the store"
        deleted = []
        deletion = threading.Thread(target=lambda: deleted.append(world.manager.delete_conversation(other)))
        deletion.start()
        deadline = time.monotonic() + 5
        while other not in world.asked.onion and time.monotonic() < deadline:
            time.sleep(0.01)
        assert other in world.asked.onion, "control: the deletion reached the stores"
        # The deletion asks the facts next: it waits for the write, or, if nothing makes it wait, erases now.
        time.sleep(0.1)
        world.vectors.hold[1].set()
        writer.join(5)
        deletion.join(5)
        assert deleted == [True], "control: the deletion went through"
        assert _rows(path) == {} and not any("Pemba" in text for text in world.vectors.items.values()), (
            "a fact written while the deletion came outlived it, in its row or its vector")
        # A refused deletion hands the conversation back: its facts are taken again.
        back = _conversation(world, "Lamu")
        dedup.forget_conversation(back)
        with pytest.raises(dedup.ConversationForgotten):
            store.add("Lamu dhows sail at dawn.", "fact", source=f"extract:{back}")
        world.conv._remember_again(back, ["opti_oignon.memory.dedup"])
        assert store.add("Lamu dhows sail at dawn.", "fact", source=f"extract:{back}")[0] is not None
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# FT4, FT5, FT6 -- the old store moves, then is emptied
# ---------------------------------------------------------------------------
def _old(world, tmp_path, facts):
    """An old store in tmp holding ``facts`` -- (text, category, in service) -- and its marker's path."""
    manager = world.mod("memory.legacy").MemoryManager(db_path=tmp_path / "old" / "memories.db")
    for text, category, active in facts:
        fact = manager.add_fact(text, category)
        assert fact is not None, f"control: the old store took {text!r}"
        if not active:
            assert manager.deactivate_fact(fact.id), "control: the old fact was put out of service"
    return manager, tmp_path / "old" / ".legacy_memory_migrated"


def _state(store):
    return {record.text: (record.source, record.category, record.active) for record in store.list(active_only=False)}


class _TrashRefusing:
    """The store, whose trash refuses once."""

    def __init__(self, store):
        self._store, self.refused = store, False

    def soft_delete(self, fact_id, **kwargs):
        if not self.refused:
            self.refused = True
            raise RuntimeError("the store is busy")
        return self._store.soft_delete(fact_id, **kwargs)

    def __getattr__(self, name):
        return getattr(self._store, name)


class _Refusing:
    """The store, whose add refuses a text holding ``word`` once."""

    def __init__(self, store, word):
        self._store, self._word = store, word

    def add(self, text, category="fact", **kwargs):
        if self._word in text:
            raise RuntimeError("the store is busy")
        return self._store.add(text, category, **kwargs)

    def __getattr__(self, name):
        return getattr(self._store, name)


def test_ft4_the_old_store_moves_each_row_deleted_once_its_fact_is_in(tmp_path):
    world = _world(tmp_path)
    try:
        migration, store = world.mod("memory.migration"), world.store()
        manager, marker = _old(world, tmp_path, [
            ("The user's name is Leonie.", "personal", True),
            ("The user likes green tea in the morning.", "preference", True),
            ("Zanzibar was the user's first long journey.", "weird-category", True),
            ("The user smoked Zanzibar cigars as a student.", "context", False)])
        keeper = _keeper(tmp_path / "old" / "memories.db")
        result = migration.migrate_legacy_to_store(manager=manager, store=store, marker_path=marker)
        assert result == {"scanned": 4, "added": 4, "merged": 0, "trashed": 1, "erased": 4, "skipped_marker": False,
                          "error": None}, result
        assert _state(store) == {
            "The user's name is Leonie.": ("legacy-import", "fact", True),
            "The user likes green tea in the morning.": ("legacy-import", "preference", True),
            "Zanzibar was the user's first long journey.": ("legacy-import", "fact", True),
            "The user smoked Zanzibar cigars as a student.": ("legacy-import", "fact", False)}, _state(store)
        assert manager.held() == [], "an old row stayed after its fact moved"
        assert marker.exists(), "the marker was not written once the old store was empty"
        assert b"Zanzibar" not in _bytes(tmp_path / "old" / "memories.db"), (
            "a moved row's words stayed in the file or the log another connection keeps")
        keeper.close()
        # A fact whose add fails keeps its row, and the marker waits; a duplicate merges.
        second, marker2 = _old(world, tmp_path / "second", [
            ("Pemba island lies north of Zanzibar.", "fact", True),
            ("The user likes green tea in the morning, always.", "preference", True)])
        result = migration.migrate_legacy_to_store(manager=second, store=_Refusing(store, "Pemba"),
                                                   marker_path=marker2)
        assert (result["added"], result["merged"], result["erased"]) == (0, 1, 1), result
        assert [fact.fact for fact in second.held()] == ["Pemba island lies north of Zanzibar."], second.held()
        assert not marker2.exists(), "the marker was written with an old row left"
        assert _state(store)["The user likes green tea in the morning."][2] is True, "a merge changed the fact"
        result = migration.migrate_legacy_to_store(manager=second, store=store, marker_path=marker2)
        assert (result["added"], result["erased"]) == (1, 1) and second.held() == [] and marker2.exists(), result
        # An old fact out of service whose copy cannot go to the trash: the copy is taken back, never left in
        # service, and its old row waits for the next run.
        third, marker3 = _old(world, tmp_path / "third", [("The user once smoked Pemba pipes.", "fact", False)])
        result = migration.migrate_legacy_to_store(manager=third, store=_TrashRefusing(store), marker_path=marker3)
        assert (result["added"], result["trashed"], result["erased"]) == (0, 0, 0), result
        assert "The user once smoked Pemba pipes." not in _state(store), "a copy that could not go to the trash stayed"
        assert len(third.held()) == 1 and not marker3.exists(), "the old row went, or the marker came, too early"
        result = migration.migrate_legacy_to_store(manager=third, store=store, marker_path=marker3)
        assert (result["added"], result["trashed"], result["erased"]) == (1, 1, 1) and marker3.exists(), result
        assert _state(store)["The user once smoked Pemba pipes."][2] is False, "the old fact out of service came back"
    finally:
        world.restore()


def test_ft5_an_old_store_migrated_before_is_emptied_nothing_added(tmp_path):
    world = _world(tmp_path)
    try:
        migration, store = world.mod("memory.migration"), world.store()
        texts = {"a": "Nairobi sits high on a plateau.", "b": "The user once kept Zanzibar parrots.",
                 "c": "Lamu dhows sail out at dawn.", "d": "Zanzibar doors are carved by hand.",
                 "e": "Mombasa ferries cross the creek."}
        # Beside them, texts no copy of this store's first migration carries out of service: one the user wrote by
        # hand, one the old store also keeps in service, one a fact written long before that migration carries.
        kept = {"x": "Kilwa ruins face the open sea.", "y": "Mafia island sharks are gentle giants.",
                "z": "Tanga markets sell sisal rope.", "w": "Ngorongoro crater floor is wide and green."}
        manager, marker = _old(world, tmp_path, [(texts["a"], "fact", True), (texts["b"], "fact", False),
                                                 (texts["c"], "fact", True), (texts["d"], "fact", True),
                                                 (texts["e"], "fact", False), (kept["x"], "fact", False),
                                                 (kept["y"], "fact", True), (kept["y"], "fact", False),
                                                 (kept["z"], "fact", False), (kept["w"], "fact", False)])
        # The store as the first migration left it -- every old fact copied in service -- and the user's acts since.
        copies = {key: store.add(text, "fact", source="legacy-import")[0] for key, text in texts.items()}
        assert store.soft_delete(copies["c"].id), "control: the user put one in the trash"
        assert store.hard_delete(copies["d"].id), "control: the user erased one"
        assert store.adopt(copies["e"].id, hashlib.sha256(texts["e"].encode("utf-8")).hexdigest())
        own = store.add("Malindi beaches face east.", "fact", source="manual")[0]
        store.add(kept["x"], "fact", source="manual")
        store.add(kept["y"], "fact", source="legacy-import")
        _peer_fact(world, "long-before", kept["z"], "legacy-import", owner="local",
                   updated_at=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat())
        _peer_fact(world, "long-after", kept["w"], "legacy-import", owner="local",
                   updated_at=(datetime.now(timezone.utc) + timedelta(hours=2)).isoformat())
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("migrated\n", encoding="utf-8")
        result = migration.migrate_legacy_to_store(manager=manager, store=store, marker_path=marker, force=True)
        assert result == {"scanned": 10, "added": 0, "merged": 0, "trashed": 1, "erased": 10, "skipped_marker": True,
                          "error": None}, result
        state = {text: active for text, (_s, _c, active) in _state(store).items()}
        assert state == {texts["a"]: True, texts["b"]: False, texts["c"]: False, texts["e"]: True,
                         own.text: True, kept["x"]: True, kept["y"]: True, kept["z"]: True, kept["w"]: True}, (
            f"a deleted fact came back, or a fact kept the wrong state: {state}")
        assert manager.held() == [], "an old row stayed"
        assert b"Zanzibar" not in _bytes(tmp_path / "old" / "memories.db"), "an old row's words stayed in the file"
        again = migration.migrate_legacy_to_store(manager=manager, store=store, marker_path=marker, force=True)
        assert (again["scanned"], again["added"], again["erased"]) == (0, 0, 0), again
    finally:
        world.restore()


def test_ft6_an_old_store_that_cannot_be_read_is_never_taken_for_an_empty_one(tmp_path, caplog):
    world = _world(tmp_path)
    try:
        migration, store = world.mod("memory.migration"), world.store()
        manager, marker = _old(world, tmp_path, [("Zanzibar was the user's first journey.", "fact", True),
                                                 ("The user likes green tea.", "preference", True)])

        def unreadable():
            raise sqlite3.DatabaseError("file is not a database")

        manager.held = unreadable
        with caplog.at_level(logging.WARNING, logger="opti_oignon.memory.migration"):
            result = migration.migrate_legacy_to_store(manager=manager, store=store, marker_path=marker)
        assert result["error"] and not marker.exists(), f"an unreadable old store was taken for empty: {result}"
        assert "migration failed" in caplog.text, "an unreadable old store went unsaid"
        marker.write_text("migrated\n", encoding="utf-8")
        result = migration.migrate_legacy_to_store(manager=manager, store=store, marker_path=marker)
        assert result["error"] and result["erased"] == 0, result
        with sqlite3.connect(str(tmp_path / "old" / "memories.db")) as conn:
            assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2, "an old row went unread"
        assert store.count() == 0, "a fact was added from an old store that could not be read"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# FT7, FT8, FT9, FT10 -- the trash
# ---------------------------------------------------------------------------
def test_ft7_a_deleted_fact_is_in_the_trash_through_the_panels_routes(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    world = _world(tmp_path, routes=True)
    try:
        app = FastAPI()
        app.include_router(world.mod("api.routes_memory").router)
        client = TestClient(app)
        store = world.store()
        one, two, three = (store.add(text, "fact", source="manual")[0] for text in (
            "Zanzibar tide tables hang by the door.", "Lamu donkeys carry the shopping uphill.",
            "Mombasa tusks arch over the avenue."))
        assert client.delete(f"/api/memory/{one.id}").status_code == 200
        assert [fact["id"] for fact in client.get("/api/memory").json()] == [three.id, two.id], (
            "a deleted fact stayed in service")
        trash = client.get("/api/memory/trash").json()
        assert trash["days"] == 30 and [fact["id"] for fact in trash["facts"]] == [one.id], trash
        deleted_at = datetime.fromisoformat(trash["facts"][0]["updated_at"])
        assert datetime.fromisoformat(trash["facts"][0]["erase_at"]) == deleted_at + timedelta(days=30), trash
        restored = client.post(f"/api/memory/{one.id}/restore")
        assert restored.status_code == 200 and restored.json()["active"] is True, restored.text
        assert client.get("/api/memory/trash").json()["facts"] == [], "a restored fact stayed in the trash"
        assert client.post(f"/api/memory/{one.id}/restore").status_code == 404, "a fact in service was restored"
        for fact in (one, two):
            assert client.delete(f"/api/memory/{fact.id}").status_code == 200
        path = world.data / "memory_facts.db"
        keeper = _keeper(path)
        erased = client.delete(f"/api/memory/trash/{two.id}")
        assert erased.status_code == 200 and two.id not in _rows(path), erased.text
        assert b"Lamu donkeys" not in _bytes(path), "an erased fact's words stayed in the file or the log kept open"
        keeper.close()
        assert (two.id, True) in world.feed.published, "the peers were not told of the erasure"
        assert client.delete(f"/api/memory/trash/{three.id}").status_code == 404
        assert _rows(path)[three.id][2] is True, "a fact in service was erased from the trash"
        emptied = client.delete("/api/memory/trash")
        assert emptied.status_code == 200 and emptied.json() == {"erased": 1}, (
            f"'trash' was taken for a fact's id, or the trash was not emptied: {emptied.text}")
        assert set(_rows(path)) == {three.id} and b"Zanzibar" not in _bytes(path), _rows(path)
    finally:
        world.restore()


class _Restoring:
    """The store, which restores ``fact_id`` right after the trash reads what it holds past its days."""

    def __init__(self, store, fact_id):
        self._store, self._fact_id = store, fact_id

    def trashed_before(self, cutoff):
        held = self._store.trashed_before(cutoff)
        assert self._store.restore(self._fact_id), "control: the fact was restored meanwhile"
        return held

    def __getattr__(self, name):
        return getattr(self._store, name)


def _age(path, fact_id, days):
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with sqlite3.connect(str(path)) as conn:
        conn.execute("UPDATE memory_facts SET updated_at = ? WHERE id = ?", (stamp, fact_id))


def test_ft8_the_trash_erases_what_it_held_past_its_days_and_nothing_else(tmp_path):
    world = _world(tmp_path)
    try:
        trash, store = world.mod("memory.trash"), world.store()
        path = world.data / "memory_facts.db"
        old, recent, restored, active = (store.add(text, "fact", source="manual")[0] for text in (
            "Zanzibar sunsets glow orange over the sea.", "Lamu nights are quiet and warm.",
            "Malindi reefs are full of fish.", "Mombasa tusks arch over the avenue."))
        for fact in (old, recent, restored):
            assert store.soft_delete(fact.id)
        _peer_fact(world, "peer-1", "Pemba forests hide flying foxes.", "manual", active=False)
        for fact_id in (old.id, restored.id, "peer-1"):
            _age(path, fact_id, 31)
        _age(path, recent.id, 29)
        keeper = _keeper(path)
        assert trash.tend(store=_Restoring(store, restored.id)) == 2
        rows = _rows(path)
        assert set(rows) == {recent.id, restored.id, active.id}, f"the trash erased the wrong facts: {rows}"
        assert rows[restored.id][2] is True and rows[recent.id][2] is False, rows
        assert restored.id in world.vectors.items, "a fact restored meanwhile lost the vector its restore gave it"
        # Another user's fact in the trash is not this user's to erase.
        _peer_fact(world, "peer-2", "Kilwa ruins face the open sea.", "manual", active=False)
        assert trash.erase("peer-2", store=store) is False and "peer-2" in _rows(path), (
            "a fact in another user's trash was erased")
        held = _bytes(path)
        assert b"Zanzibar" not in held and b"Pemba" not in held, "an erased fact's words stayed in the file or log"
        assert {(old.id, True), ("peer-1", True)} <= set(world.feed.published), "the peers were not told"
        assert trash.erase(active.id, store=store) is False and active.id in _rows(path), (
            "a fact in service was erased as if it were in the trash")
        assert trash.tend(store=store) == 0, "control: nothing left past its days"
        keeper.close()
    finally:
        world.restore()


def test_ft9_the_trash_days_come_from_the_yaml_within_bounds(tmp_path, caplog):
    world = _world(tmp_path)
    try:
        trash = world.mod("memory.trash")
        assert trash.trash_days() == 30, "the shipped YAML keeps a deleted fact 30 days"
        written = tmp_path / "memory.yaml"
        written.write_text("trash_days: 7\n", encoding="utf-8")
        assert trash.load_config(written)["trash_days"] == 7
        for bad in ("trash_days: 0", "trash_days: 3651", "trash_days: 7.5", "trash_days: '7'", "trash_days: true"):
            written.write_text(bad + "\n", encoding="utf-8")
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="opti_oignon.memory.trash"):
                assert trash.load_config(written)["trash_days"] == 30, f"{bad} was taken"
            assert "trash_days" in caplog.text, f"{bad} was refused unsaid"
        assert trash.load_config(tmp_path / "missing.yaml")["trash_days"] == 30
    finally:
        world.restore()


def test_ft10_the_trash_is_tended_at_boot_and_every_hour():
    tree = ast.parse(source("api", "app.py").read_text(encoding="utf-8"))
    calls = {ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert "_trash.tend()" in calls, "the trash is not tended at boot"
    assert "asyncio.sleep(_trash.PRUNE_INTERVAL_S)" in calls and "asyncio.to_thread(_trash.tend)" in calls, (
        "the trash is not tended every hour, off the event loop")
    assert "asyncio.create_task(_trash_retention())" in calls and "trash.cancel()" in calls, (
        "the hourly pass is not started, or not cancelled at shutdown")
    lifespan = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
                    and any(isinstance(sub, ast.Yield) for sub in ast.walk(node))
                    and "_trash.tend()" in ast.unparse(node))
    body = ast.unparse(lifespan)
    assert body.index("_trash.tend()") < body.index("yield") < body.index("trash.cancel()"), (
        "the boot's pass and the shutdown's cancel are not on either side of the server's life")


# ---------------------------------------------------------------------------
# FT11 -- every connection of the facts overwrites what it frees
# ---------------------------------------------------------------------------
def test_ft11_every_connection_of_the_facts_overwrites_what_it_frees(tmp_path):
    world = _world(tmp_path)
    try:
        store = world.store()
        path = world.data / "memory_facts.db"
        edited = store.add("Zanzibar dhow captains read the stars at night.", "fact", source="manual")[0]
        deleted = store.add("Pemba clove farmers dry their harvest on mats.", "fact", source="manual")[0]
        assert b"Zanzibar dhow" in _bytes(path) and b"Pemba clove" in _bytes(path), "control: the facts were written"
        assert store.update(edited.id, text="Dhow captains read the stars at night.") is not None
        assert store.hard_delete(deleted.id)
        held = _bytes(path)
        assert b"Zanzibar dhow" not in held, "an edit left the old text in the file's free space"
        assert b"Pemba clove" not in held, "a deleted fact's words stayed in the file's free pages"
        assert b"Dhow captains" in held, "control: the edited text is in the file"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# FT12 -- a facts store that cannot forget refuses by name
# ---------------------------------------------------------------------------
def test_ft12_a_facts_store_that_cannot_forget_refuses_the_deletion_by_name(tmp_path):
    world = _world(tmp_path)
    try:
        dedup, store = world.mod("memory.dedup"), world.store()
        gone = _conversation(world, "Zanzibar")
        fact = store.add("Zanzibar harbour stories filled every evening.", "fact", source=f"extract:{gone}")[0]
        world.vectors.refuse = fact.id
        with pytest.raises(world.conv.ConversationNotForgotten) as refused:
            world.manager.delete_conversation(gone)
        assert "facts" in str(refused.value), f"the refusal does not name the facts: {refused.value}"
        assert world.manager.get_conversation(gone) is not None, "the conversation went with the refusal"
        assert fact.id in _rows(world.data / "memory_facts.db"), "the fact went with the refusal"
        assert store.add("Spring sailing along the Zanzibar coast.", "fact", source=f"auto-capture:{gone}")[0], (
            "a refused deletion left the facts refusing the conversation")
        world.vectors.refuse = None
        assert world.manager.delete_conversation(gone) is True, "once the facts can forget, the deletion goes through"
        assert _rows(world.data / "memory_facts.db") == {}
    finally:
        world.restore()
    fresh = _world(tmp_path / "fresh")
    try:
        cid = _conversation(fresh, "Lamu")
        assert fresh.manager.delete_conversation(cid) is True
        assert not (fresh.data / "memory_facts.db").exists(), "a deletion created the facts' store"
        # Nor does a start: its migration finds no old store, and its trash's pass no facts.
        marker = fresh.data / ".legacy_memory_migrated"
        result = fresh.mod("memory.migration").migrate_legacy_to_store(marker_path=marker)
        assert result["error"] is None and marker.exists(), f"control: the start's migration ran: {result}"
        assert fresh.mod("memory.trash").tend() == 0
        assert not (fresh.data / "memories.db").exists(), "the start created the old store to find it empty"
        assert not (fresh.data / "memory_facts.db").exists(), "the start created the facts' store"
        dedup = fresh.mod("memory.dedup")
        with pytest.raises(dedup.ConversationForgotten):
            fresh.store().add("Lamu dhows sail at dawn.", "fact", source=f"extract:{cid}")
    finally:
        fresh.restore()


# ---------------------------------------------------------------------------
# FT13 -- the census
# ---------------------------------------------------------------------------
# Every table of the package keyed by a conversation, and who forgets it at a deletion: the conversation store
# itself, the onion, or the registry's store; or the reason it is kept. A new table keyed by a conversation fails
# this census until it is forgotten or named.
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
    "veilid/change_feed.py": "the sync change feed: each record's latest payload under (kind, record_id), a "
                             "conversation's whole included: owed, with its compaction, to the next block",
    "veilid/deferred_ledger.py": "what a peer offered that waits its turn, under (kind, record_id): owed, with the "
                                 "change feed, to the next block",
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
    """Each module's tables keyed by a conversation, and the sources naming a conversation it writes under."""
    root = REPO / "opti_oignon"
    tables, sources = {}, {}
    for path in sorted({Path(path) for path in package_modules().values()}):
        relative = path.relative_to(root).as_posix()
        for text in _texts(ast.parse(path.read_text(encoding="utf-8"))):
            for match in re.finditer(r"CREATE TABLE(?: IF NOT EXISTS)?\s+([\w{}]+)\s*\((.*?)\)\s*(?:;|$)", text,
                                     re.S | re.I):
                columns = {part.strip().split()[0].lower().strip("()") for part in match.group(2).split(",")
                           if part.strip().split()}
                if columns & _KEYS or _GENERIC <= columns:
                    tables.setdefault(relative, set()).add(match.group(1))
            head = re.fullmatch(r"([a-z-]+:)\{\}", text)
            if head:
                sources.setdefault(relative, set()).add(head.group(1))
    return tables, sources


def test_ft13_every_table_keyed_by_a_conversation_is_forgotten_or_named(tmp_path):
    world = _world(tmp_path)
    try:
        registry = {module.split(".", 1)[1].replace(".", "/") + ".py" for _store, module, _f, _w in
                    world.conv._FORGETTERS}
        drawn = set(world.mod("memory.dedup").DRAWN_SOURCES)
    finally:
        world.restore()
    tables, sources = _census()
    # The census finds what it must, or it proves nothing.
    assert {"branches", "branch_messages"} <= tables.get("conversation_branches.py", set()), tables
    assert "messages" in tables.get("conversation.py", set()), tables
    assert "memories" in tables.get("memory/legacy.py", set()), tables
    assert "veilid/change_feed.py" in tables, f"the census reads no f-string, or no generic key: {sorted(tables)}"
    unaccounted = {module: sorted(found) for module, found in tables.items()
                   if module not in registry and module not in _FORGOTTEN_BY and module not in _KEPT}
    assert unaccounted == {}, f"tables keyed by a conversation that a deletion neither forgets nor names: {unaccounted}"
    assert sorted(set(_KEPT) & registry) == [], "named as kept and forgotten too"
    # A fact names its conversation in its source: each writer's prefix is one the facts' forgetting erases.
    assert "memory/dedup.py" in registry, "the facts are not asked at a deletion"
    assert "extract:" in sources.get("api/routes_memory.py", set()), f"control: the census reads sources: {sources}"
    assert "auto-capture:" in sources.get("memory/auto_capture.py", set()), sources
    written = {prefix for found in sources.values() for prefix in found if prefix in ("extract:", "auto-capture:")}
    assert written == drawn, f"a source naming a conversation the forgetting does not erase: {written} != {drawn}"


# ---------------------------------------------------------------------------
# FT14 -- the dashboard counts the facts in service
# ---------------------------------------------------------------------------
def test_ft14_the_dashboard_counts_the_facts_in_service_never_the_old_store(tmp_path):
    old = SimpleNamespace(count_facts=lambda active_only=True: 99)
    world = _world(tmp_path, health=old)
    try:
        store = world.store()
        facts = [store.add(text, "fact", source="manual")[0] for text in (
            "Zanzibar tide tables hang by the door.", "Lamu donkeys carry the shopping.", "Mombasa tusks arch high.")]
        assert store.soft_delete(facts[0].id)
        dashboard = world.mod("api.routes_health").health_dashboard()
        assert dashboard.memory_fact_count == 2, (
            f"the dashboard does not count the facts in service: {dashboard.memory_fact_count}")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# FT15 -- a peer's deletion erases the drawn facts too
# ---------------------------------------------------------------------------
def test_ft15_a_peers_deletion_erases_the_drawn_facts_too(tmp_path):
    world = _world(tmp_path)
    try:
        store = world.store()
        gone = _conversation(world, "Zanzibar")
        drawn = store.add("Zanzibar harbour stories filled every evening.", "fact", source=f"extract:{gone}")[0]
        own = store.add("Mombasa markets open early.", "fact", source="manual")[0]
        assert world.manager.apply_synced_conversation({"conversation": {"id": gone}}, deleted=True,
                                                       updated_at=_LATER) is True
        assert world.manager.get_conversation(gone) is None, "control: the peer's deletion was applied"
        rows = _rows(world.data / "memory_facts.db")
        assert drawn.id not in rows and own.id in rows, f"a peer's deletion left the drawn fact: {rows}"
        assert drawn.id not in world.vectors.items and b"Zanzibar" not in _bytes(world.data / "memory_facts.db")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# FT16 -- a tree without the memory package passes the facts
# ---------------------------------------------------------------------------
def test_ft16_a_tree_without_the_memory_package_passes_the_facts(tmp_path, caplog):
    world = _world(tmp_path, memory=False)
    try:
        cid = _conversation(world, "Zanzibar")
        with caplog.at_level(logging.WARNING, logger=CONV):
            assert world.manager.delete_conversation(cid) is True, "a missing package refused the deletion"
        assert "no facts in this tree" in caplog.text, f"the facts passed unsaid: {caplog.text}"
        assert world.manager.get_conversation(cid) is None
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# FT17 -- a fact the user made theirs stays when its conversation goes
# ---------------------------------------------------------------------------
def test_ft17_a_fact_the_user_made_theirs_stays_when_its_conversation_goes(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    world = _world(tmp_path, routes=True)
    try:
        app = FastAPI()
        app.include_router(world.mod("api.routes_memory").memories_router)
        client = TestClient(app)
        store, pending = world.store(), world.mod("pending_writes")
        gone = _conversation(world, "Zanzibar")
        other = _conversation(world, "Kilimanjaro")

        def drawn(text, source=f"extract:{gone}"):
            return store.add(text, "fact", source=source)[0]

        by_hand = drawn("The user grew up near the harbour of Zanzibar town.")
        merged, decision = store.add("The user grew up near the harbour of Zanzibar.", "fact", source="manual")
        assert decision.action == "merge" and merged.id == by_hand.id, "control: the typed fact merged"
        asked = drawn("Spring sailing trips along the Zanzibar beaches.", f"auto-capture:{gone}")
        pending.apply_write("memory", "add", {"text": "Spring sailing trips along the Zanzibar beaches!"},
                            source="agent", memory_store=store)
        accepted = drawn("A jar of Zanzibar cloves on the kitchen shelf.")
        store.add("A jar of Zanzibar cloves on the kitchen shelf, still.", "fact", source="accepted:p7")
        edited = drawn("Zanzibar doors are carved by hand.")
        assert client.patch(f"/api/memories/{edited.id}", json={"text": "Zanzibar doors are carved in teak."}
                            ).status_code == 200
        rewritten = drawn("Zanzibar dhows sail out at dawn.")
        pending.apply_write("memory", "update", {"fact_id": rewritten.id, "text": "Zanzibar dhows sail at dawn."},
                            source="agent", memory_store=store)
        _peer_fact(world, "peer-3", "Zanzibar ferries run every hour.", f"extract:{gone}", owner="local")
        digest = hashlib.sha256(b"Zanzibar ferries run every hour.").hexdigest()
        assert client.post("/api/memories/adopt", json={"items": [{"id": "peer-3", "digest": digest}]}
                           ).json()["adopted"] == ["peer-3"], "control: the peer's drawn fact was adopted"
        # A capture from another conversation merged into a drawn fact takes nothing: the fact is still drawn.
        twice = drawn("Zanzibar spice tours start at nine.")
        store.add("Zanzibar spice tours start at nine sharp.", "fact", source=f"auto-capture:{other}")
        plain = drawn("Zanzibar tides rise twice a day.")
        # A fact that is already the user's keeps its own source when another act of theirs merges into it.
        agents = store.add("Lamu donkeys carry the shopping uphill.", "fact", source="agent")[0]
        store.add("Lamu donkeys carry the shopping uphill!", "fact", source="manual")
        # An old fact moved in service makes a drawn fact it merges into the user's; one out of service does not.
        kept_old = drawn("Zanzibar markets close at noon on Fridays.")
        withdrawn = drawn("Zanzibar buses leave from the old fort.")
        old, marker = _old(world, tmp_path, [("Zanzibar markets close at noon on Fridays!", "fact", True),
                                             ("Zanzibar buses leave from the old fort!", "fact", False)])
        moved = world.mod("memory.migration").migrate_legacy_to_store(manager=old, store=store, marker_path=marker)
        assert moved["merged"] == 2, f"control: both old facts merged into the drawn ones: {moved}"
        assert world.manager.delete_conversation(gone) is True
        rows = _rows(world.data / "memory_facts.db")
        assert {fact_id: source_ for fact_id, (_t, source_, _a) in rows.items()} == {
            by_hand.id: "manual", asked.id: "agent", accepted.id: "accepted:p7", edited.id: "manual",
            rewritten.id: "agent", "peer-3": "manual", agents.id: "agent", kept_old.id: "legacy-import"}, (
            f"a fact the user made theirs went with its conversation, or a drawn one stayed: {rows}")
        assert withdrawn.id not in rows, "an old fact out of service made a drawn fact the user's"
        assert twice.id not in rows and plain.id not in rows
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# FT18 -- a peer's erasure drops this device's vector; a forgotten conversation's fact is refused
# ---------------------------------------------------------------------------
def _state_of(fact_id, text, source, active):
    stamp = datetime.now(timezone.utc).isoformat()
    return {"user_id": "local", "fact": {"id": fact_id, "text": text, "category": "fact", "source": source,
                                         "created_at": stamp, "updated_at": stamp, "active": active}}


def test_ft18_a_peers_erasure_drops_this_devices_vector_and_a_forgotten_conversations_fact_is_refused(tmp_path):
    world = _world(tmp_path)
    try:
        store = world.store()
        canonical = world.mod("memory.canonical_store").get_canonical_store()
        path = world.data / "memory_facts.db"
        erased, trashed, kept = (store.add(text, "fact", source="manual")[0] for text in (
            "Zanzibar harbour lights glow at night.", "Lamu dhows creak in the wind.",
            "Mombasa tusks arch over the avenue."))
        assert {erased.id, trashed.id, kept.id} <= set(world.vectors.items), "control: this device holds the vectors"
        # A peer puts one out of service, and erases another.
        assert canonical.apply_synced_memory_canonical(
            trashed.id, _state_of(trashed.id, trashed.text, "manual", False)) is True
        assert trashed.id not in world.vectors.items, "a fact out of service on a peer kept its vector here"
        keeper = _keeper(path)
        assert canonical.apply_synced_memory_canonical(erased.id, {}, deleted=True) is True
        assert erased.id not in world.vectors.items and erased.id not in _rows(path), "a peer's erasure left it here"
        assert b"Zanzibar" not in _bytes(path), "a peer's erasure left the words in the file or the log kept open"
        keeper.close()
        # A vector layer that is there and fails never refuses the record -- the round would not bring it again:
        # the row goes, and the vector is owed to the trash's next pass.
        world.vectors.refuse = kept.id
        assert canonical.apply_synced_memory_canonical(kept.id, {}, deleted=True) is True
        assert kept.id not in _rows(path) and kept.id in world.vectors.items, "control: the vector layer failed"
        # So for a fact a peer put out of service; and one restored here before the pass keeps its vector.
        paused, restored = (store.add(text, "fact", source="manual")[0] for text in (
            "Kilwa ruins face the open sea.", "Tanga markets sell sisal rope."))
        for fact in (paused, restored):
            world.vectors.refuse = fact.id
            assert canonical.apply_synced_memory_canonical(
                fact.id, _state_of(fact.id, fact.text, "manual", False)) is True
        world.vectors.refuse = None
        assert {paused.id, restored.id} <= set(world.vectors.items), "control: the vector layer failed"
        assert store.restore(restored.id), "control: the fact was restored here"
        trash = world.mod("memory.trash")
        # The start's pass (no store handed): nothing past its days, vectors owed.
        trash.tend()
        assert kept.id not in world.vectors.items and paused.id not in world.vectors.items, (
            "an owed vector was never dropped")
        assert world.vectors.items.get(restored.id) == restored.text, "an owed drop took a restored fact's vector"
        # Back in service on the peer: this device's vector comes back at the next pass.
        assert canonical.apply_synced_memory_canonical(
            trashed.id, _state_of(trashed.id, trashed.text, "manual", True)) is True
        trash.tend(store=store)
        assert world.vectors.items.get(trashed.id) == trashed.text, "a fact back in service got no vector back"
        # A peer's fact drawn from a conversation this process forgot is refused, as this device's own would be.
        cid = _conversation(world, "Pemba")
        assert world.manager.delete_conversation(cid) is True
        late = _state_of("peer-late", "Pemba reefs shelter green turtles.", f"extract:{cid}", True)
        assert canonical.apply_synced_memory_canonical("peer-late", late) is False
        assert "peer-late" not in _rows(path), "a fact drawn from a forgotten conversation landed from a peer"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# FT19 -- a restore racing the forgetting; the trash's clock
# ---------------------------------------------------------------------------
def test_ft19_a_restore_racing_the_forgetting_leaves_no_vector_and_the_trash_keeps_its_clock(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    world = _world(tmp_path, routes=True)
    try:
        store, trash = world.store(), world.mod("memory.trash")
        path = world.data / "memory_facts.db"
        gone = _conversation(world, "Pemba")
        elsewhere = _conversation(world, "Watamu")
        fact = store.add("Pemba forests hide flying foxes.", "fact", source=f"extract:{gone}")[0]
        assert store.soft_delete(fact.id), "control: the drawn fact is in the trash"
        # A restore holding the store, its vector on the way back, when the deletion comes.
        world.vectors.hold = ("Pemba", threading.Event())
        restorer = threading.Thread(target=store.restore, args=(fact.id,))
        restorer.start()
        assert world.vectors.entered.wait(5), "control: the restore holds the store"
        deleted = []
        deletion = threading.Thread(target=lambda: deleted.append(world.manager.delete_conversation(gone)))
        deletion.start()
        deadline = time.monotonic() + 5
        while gone not in world.asked.onion and time.monotonic() < deadline:
            time.sleep(0.01)
        assert gone in world.asked.onion, "control: the deletion reached the stores"
        time.sleep(0.1)
        world.vectors.hold[1].set()
        restorer.join(5)
        deletion.join(5)
        world.vectors.hold = None
        assert deleted == [True], "control: the deletion went through"
        assert fact.id not in _rows(path) and not any("Pemba" in text for text in world.vectors.items.values()), (
            "a restore racing the forgetting left the erased text in its row or its vector")
        # A fact in the trash is not edited: an edit would restart the days it waits there.
        app = FastAPI()
        app.include_router(world.mod("api.routes_memory").memories_router)
        client = TestClient(app)
        waiting = store.add("Lamu nights are quiet and warm.", "fact", source="manual")[0]
        assert store.soft_delete(waiting.id)
        _age(path, waiting.id, 31)
        assert client.patch(f"/api/memories/{waiting.id}", json={"text": "Lamu nights are quiet."}).status_code == 404
        # Nor does a near duplicate merge into it: a vector this device kept for it finds it, and the new fact is
        # written anew.
        stale = store.add("Malindi reefs are full of fish.", "fact", source="manual")[0]
        assert store.soft_delete(stale.id)
        _age(path, stale.id, 31)
        world.vectors.near = stale.id
        new, decision = store.add("Coral gardens line the Watamu coast.", "fact", source="manual")
        world.vectors.near = None
        assert decision.action == "insert" and new is not None and new.id != stale.id, (
            f"a near duplicate merged into a fact in the trash: {decision}")
        # Nor does a proposal accepted, or the agent, edit a fact in the trash.
        pending = world.mod("pending_writes")
        queued = store.add("Watamu turtles nest in March.", "fact", source=f"extract:{elsewhere}")[0]
        assert store.soft_delete(queued.id)
        _age(path, queued.id, 31)
        with pytest.raises(pending.TargetMissing):
            pending.apply_write("memory", "update", {"fact_id": queued.id, "text": "Watamu turtles nest in April."},
                                source="accepted:p9", memory_store=store)
        assert trash.tend(store=store) == 3, "a fact in the trash had its days restarted"
        assert set(_rows(path)) == {new.id}, _rows(path)
        # A restore gives a vector back only for the text it was made from: a peer's new text that lands while it
        # computes it gets none of the old one.
        canonical = world.mod("memory.canonical_store").get_canonical_store()
        moved = store.add("Kilifi creek is deep and calm.", "fact", source="manual")[0]
        assert store.soft_delete(moved.id)
        world.vectors.embed_hold = ("Kilifi", threading.Event())
        restorer = threading.Thread(target=store.restore, args=(moved.id,))
        restorer.start()
        assert world.vectors.embedding.wait(5), "control: the restore computes the vector"
        assert canonical.apply_synced_memory_canonical(
            moved.id, _state_of(moved.id, "Takaungu creek is shallow.", "manual", False)) is True
        world.vectors.embed_hold[1].set()
        restorer.join(5)
        world.vectors.embed_hold = None
        assert moved.id not in world.vectors.items, (
            "a restore gave back a vector made from a text the fact no longer holds")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# FT20 -- a peer's drawn fact never lands after its conversation's deletion
# ---------------------------------------------------------------------------
def test_ft20_a_peers_drawn_fact_never_lands_after_its_conversations_deletion(tmp_path):
    world = _world(tmp_path)
    try:
        world.store()
        canonical = world.mod("memory.canonical_store").get_canonical_store()
        dedup = world.mod("memory.dedup")
        path = world.data / "memory_facts.db"
        cid = _conversation(world, "Pemba")
        entered, release = threading.Event(), threading.Event()
        checked = dedup.forgotten

        def check_then_wait(conversation_id):
            # The peer's fact passes the check before the deletion comes, and waits right after it.
            answer = checked(conversation_id)
            if not entered.is_set():
                entered.set()
                assert release.wait(5), "the held apply was never released"
            return answer

        dedup.forgotten = check_then_wait
        late = _state_of("peer-late", "Pemba reefs shelter green turtles.", f"extract:{cid}", True)
        applied = []
        applier = threading.Thread(target=lambda: applied.append(canonical.apply_synced_memory_canonical(
            "peer-late", late)))
        applier.start()
        assert entered.wait(5), "control: the apply checked the conversation"
        deleted = []
        deletion = threading.Thread(target=lambda: deleted.append(world.manager.delete_conversation(cid)))
        deletion.start()
        deadline = time.monotonic() + 5
        while cid not in world.asked.onion and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cid in world.asked.onion, "control: the deletion reached the stores"
        time.sleep(0.1)
        release.set()
        applier.join(5)
        deletion.join(5)
        assert deleted == [True], "control: the deletion went through"
        assert "peer-late" not in _rows(path), "a peer's fact drawn from a conversation landed after its deletion"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# FT21 -- a process that never opened the vector layer owes its vectors to the trash
# ---------------------------------------------------------------------------
def test_ft21_a_process_that_never_opened_the_vectors_owes_them_to_the_trash(tmp_path):
    world = _world(tmp_path)
    try:
        store, dedup = world.store(), world.mod("memory.dedup")
        canonical = world.mod("memory.canonical_store").get_canonical_store()
        path = world.data / "memory_facts.db"
        gone = _conversation(world, "Zanzibar")
        drawn = store.add("Zanzibar harbour stories filled every evening.", "fact", source=f"extract:{gone}")[0]
        erased = store.add("Lamu dhows creak in the wind.", "fact", source="manual")[0]
        assert {drawn.id, erased.id} <= set(world.vectors.items), "control: the vectors were written"
        # Another process: its facts never opened whole, its vector layer closed; the file holds the vectors still.
        dedup._memory_store = None
        sys.modules["opti_oignon.memory.vector_store"]._vector_store = None
        assert world.manager.delete_conversation(gone) is True
        assert canonical.apply_synced_memory_canonical(erased.id, {}, deleted=True) is True
        assert sys.modules["opti_oignon.memory.vector_store"]._vector_store is None, (
            "the vector layer was opened only to drop a vector")
        assert drawn.id not in _rows(path) and erased.id not in _rows(path), "a row stayed"
        assert {fact_id for fact_id, act in canonical.owed_vectors() if act == "drop"} == {drawn.id, erased.id}, (
            "a vector the closed layer holds was not owed")
        # The trash's pass, which opens the layer, settles them.
        world.mod("memory.trash").tend()
        assert drawn.id not in world.vectors.items and erased.id not in world.vectors.items, "an owed vector stayed"
        assert canonical.owed_vectors() == [], "a settled vector stayed owed"
    finally:
        world.restore()
