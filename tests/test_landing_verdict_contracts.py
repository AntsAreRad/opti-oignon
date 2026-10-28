#!/usr/bin/env python3
"""Contracts for a sync round that writes a received version into its store only when the journal keeps it.

A round wrote every version a paired device sent into its store -- a note, a conversation, a fact, a skill -- and only
then let the journal's merge choose the version it keeps: a copy that lost was already written. A paired device's
journal served again after its file was remade brought a deleted record back, with its words, while the journal kept
the tombstone; a deletion older than an edit made here destroyed a conversation, a fact or a skill the journal kept
alive; an older version overwrote a newer one. Now every landing asks the journal's verdict first
(``reconcile.received_wins``): a paired device is the user's own, and what it sends is ordered by the journal's
last-writer-wins, never written when it loses. The census counts the four landings behind that gate, and the engine
owes nothing.

  * LW1 -- THE VERDICT IS THE JOURNAL'S MERGE: over generated sides, a received version wins exactly when the merge
    journals it, or holds it already; at or below a deletion's floor it never does. Pure.
  * LW2 -- A JOURNAL SERVED AGAIN BRINGS NOTHING DELETED BACK: a paired device's journal remade (its epoch changed)
    serves the versions of a note, a conversation, a fact and a skill deleted here; none comes back into its store, no
    file under the data folder holds their words, and the journal keeps the tombstones.
  * LW3 -- AN EDIT MADE THERE BEFORE THE DELETION ARRIVED: at equal clocks the tie-break decides, and the store holds
    exactly what the journal holds, either way.
  * LW4 -- A TOMBSTONE THAT LOSES DESTROYS NOTHING: a deletion older than the version here leaves the note, the
    conversation and its messages, the fact and the skill whole.
  * LW5 -- AN OLDER VERSION NEVER REGRESSES THE STORE.
  * LW6 -- SEVERAL VERSIONS OF ONE RECORD IN A BATCH: only the winner is written, whatever the order; a winner its
    store refuses takes its losers out of the merge with it, so the journal keeps what the store keeps.
  * LW7 -- THE FLOOR: a version at or below the clock a deletion here erased is never written, with no tombstone
    journalled; one above it is.
  * LW8 -- A LOSER GOES ON TO THE MERGE, UNWRITTEN: the conflict log keeps a concurrent divergence, and a journal that
    cannot be read lets nothing be written.
  * LW9 -- THE CENSUS: the engine owes nothing, its four landings are gated behind the verdict, and a landing with no
    verdict in front of it is a finding.
  * LW10 -- AN INJECTED SINK IS BEHIND THE SAME VERDICT: it is handed a record's winner and none of its losers.
  * LW11 -- A PEER'S DELETION THAT WINS THE TIE-BREAK AT THE CLOCK HELD HERE IS JOURNALLED, and no file keeps the
    conversation's words.
  * LW12 -- A SKILL APPROVED FROM THE PANEL LANDS BEHIND THE SAME VERDICT: written into the registry, marked as
    received; one the registry cannot take waits, unjournalled; one that lost meanwhile is written nowhere.
  * LW13 -- A CONVERSATION'S DELETION MAKES THE LEG READ THE JOURNAL AGAIN: its forgetting journals the tombstones of
    what other stores drew from it, and the landers after it judge on what the journal holds then.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import ast
import base64
import random
import re
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_lw1_the_verdict_is_the_journals_merge": 5.0,
    "test_lw2_a_journal_served_again_brings_nothing_deleted_back": 10.0,
    "test_lw3_an_edit_made_there_before_the_deletion_arrived_lands_as_the_journal_decides": 10.0,
    "test_lw4_a_tombstone_that_loses_destroys_nothing": 10.0,
    "test_lw5_an_older_version_never_regresses_the_store": 10.0,
    "test_lw6_only_a_batchs_winner_is_written_and_a_refused_winner_takes_its_losers": 10.0,
    "test_lw7_a_version_at_the_floor_of_a_deletion_is_never_written": 10.0,
    "test_lw8_a_loser_goes_on_to_the_merge_unwritten": 10.0,
    "test_lw9_the_engine_owes_nothing_and_a_landing_without_its_verdict_is_a_finding": 90.0,
    "test_lw10_an_injected_sink_is_behind_the_same_verdict": 10.0,
    "test_lw11_a_peers_deletion_that_wins_the_tie_break_is_journalled_and_keeps_no_words": 10.0,
    "test_lw12_a_skill_approved_from_the_panel_lands_behind_the_same_verdict": 10.0,
    "test_lw13_a_conversations_deletion_makes_the_leg_read_the_journal_again": 15.0,
}

# The words planted, each a byte string no file holds by chance.
NOTE_TITLE, TALK, FACT, SKILL_STEP = "NOTE-WORDS-8", "TALK-WORDS-8", "FACT-WORDS-8", "SKILL-WORDS-8"
UPDATE = b"UPDATE-WORDS-8"
_WORDS = tuple(word.encode() for word in (NOTE_TITLE, TALK, FACT, SKILL_STEP))
_GUARD = REPO / ".github" / "scripts" / "write_census_guard.py"
_ENGINE = "opti_oignon/veilid/sync_engine.py"


def _plain(path, *, check_same_thread=False, **_options):
    return sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=0.3)


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _held(data, *words):
    """Which of ``words`` any file under ``data`` still holds."""
    found = set()
    for path in data.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            found.update(w for w in words if w in raw)
    return found


class _Signer:
    """A device with no signing backend: records are journalled unsigned and accepted unverified, the pre-VL-01
    posture, so the round's landings are what these contracts read."""

    def verify_available(self):
        return False

    def public_key(self):
        return b""

    def sign(self, _data):
        raise RuntimeError("no signing backend")


class _Peer:
    """The paired device: its journal as a list of versions, served from the watermark asked, under its epoch."""

    def __init__(self, protocol, records, versions=(), epoch="epoch-1"):
        self.protocol, self.records = protocol, records
        self.versions, self.epoch = list(versions), epoch

    def fetch(self, request):
        cursor = int(request["watermark"])
        chunk = self.versions[cursor:]
        return {"v": self.protocol.PROTOCOL_VERSION, "type": self.protocol.MSG_RECORD_BATCH, "device": "dev-b",
                "high_water": len(self.versions) if chunk else cursor,
                "records": self.records.encode_records(chunk), "epoch": self.epoch}


_VEILID = ("records", "reconcile", "change_feed", "signing", "deferred_ledger", "peers", "producers", "protocol",
           "review", "sync_engine")


def _world(tmp_path, **sinks):
    """This device, ``dev-a``, over ``tmp_path/data``: the sync engine, its journal, its paired set and its deferred
    ledger, and the four stores a round writes -- the notes, the conversations, the canonical facts, the skills --
    each from its source and each publishing its own changes into the journal through the engine; ``sinks`` injected
    into the engine in the place of the stores' own."""
    data = tmp_path / "data"
    data.mkdir(parents=True)

    def effective_user_id(user_id, single_user_mode=True):
        return "local" if (single_user_mode or user_id is None) else user_id

    seeded = {
        "opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain),
        "opti_oignon.db_encryption": _module("opti_oignon.db_encryption", get_encrypted_connection=_plain,
                                             SQLCIPHER_AVAILABLE=False),
        "opti_oignon.config": _module("opti_oignon.config", DATA_DIR=data),
        "opti_oignon.user_isolation": _module("opti_oignon.user_isolation", DEFAULT_LOCAL_USER="local",
                                              effective_user_id=effective_user_id),
        "opti_oignon.veilid.guard": _module("opti_oignon.veilid.guard", veilid_available=lambda: True,
                                            assert_sync_allowed=lambda: None),
        "opti_oignon.run_journal": _module("opti_oignon.run_journal", forget_conversation=lambda cid: None),
        "opti_oignon.memory.librarian": _module("opti_oignon.memory.librarian",
                                                forget_conversation=lambda conversation_id, config=None: 0),
    }
    targets = {f"opti_oignon.veilid.{name}": source("veilid", f"{name}.py") for name in _VEILID}
    targets.update({
        "opti_oignon.notes.blob_store": source("notes", "blob_store.py"),
        "opti_oignon.notes.notes_store": source("notes", "notes_store.py"),
        "opti_oignon.notes.note_updates_store": source("notes", "note_updates_store.py"),
        "opti_oignon.memory.canonical_store": source("memory", "canonical_store.py"),
        "opti_oignon.agent.skills": source("agent", "skills.py"),
        "opti_oignon.conversation": source("conversation.py"),
    })
    loaded, restore = isolate(targets=targets, seeded=seeded,
                              packages=("opti_oignon.veilid", "opti_oignon.notes", "opti_oignon.memory",
                                        "opti_oignon.agent"))
    try:
        mod = {name.rsplit(".", 1)[-1]: module for name, module in loaded.items()}
        feed_module, engine_module = mod["change_feed"], mod["sync_engine"]
        feed = feed_module.ChangeFeed(root=data)
        feed_module.set_change_feed(feed)
        peers = mod["peers"].PeerStore(root=data)
        engine = engine_module.SyncEngine(device="dev-a", feed=feed, store=peers, signer=_Signer(),
                                          ledger=mod["deferred_ledger"].DeferredLedger(root=data), **sinks)
        engine_module.set_sync_engine(engine)
        engine.register_peer("dev-b", "route-b")
        notes = mod["notes_store"].NotesStore(root=data)
        mod["notes_store"]._store = notes
        mod["note_updates_store"]._store = mod["note_updates_store"].NoteUpdatesStore(root=data)
        conversations = mod["conversation"].ConversationManager(db_path=data / "conversations.db",
                                                                publish_to_sync=True)
        mod["conversation"].conversation_manager = conversations
        facts = mod["canonical_store"].CanonicalMemoryStore(data / "memory_canonical.db")
        mod["canonical_store"]._store = facts
        skills = mod["skills"].SkillRegistry(data / "skills")
        mod["skills"]._REGISTRY = skills
    except BaseException:
        restore()
        raise

    def close():
        try:
            engine_module.set_sync_engine(None)
            feed_module.reset_change_feed()
        finally:
            restore()

    return SimpleNamespace(data=data, mod=mod, feed=feed, engine=engine, peers=peers, notes=notes,
                           conversations=conversations, facts=facts, skills=skills, restore=close)


def _plant(world):
    """One record of each kind, written here as the user writes it, each journalled by its store: their keys. The
    ids are fixed, so the content hashes a tie-break compares are the same at every run."""
    note = world.notes.add_note(NOTE_TITLE, body_crdt=b"", tags='["kept"]', note_id="lw-note")
    minted = world.mod["conversation"].uuid
    world.mod["conversation"].uuid = SimpleNamespace(uuid4=lambda: "lw-conversation")
    try:
        conversation = world.conversations.create_conversation(title="about the trip")
    finally:
        world.mod["conversation"].uuid = minted
    world.conversations.add_message(conversation.id, "user", f"{TALK} is where we stop first.")
    fact = world.facts.add(f"The user's sister is called {FACT}.", "fact", fact_id="lw-fact")
    skills = world.mod["skills"]
    world.skills.add("deploy", "ops", f"## When to Use\nShipping.\n\n## Procedure\n{SKILL_STEP}, then announce.",
                     source=skills.SOURCE_MANUAL, status=skills.STATUS_PUBLISHED)
    return {"note": note.id, "conversation": conversation.id, "memory_canonical": fact.id,
            "skill": skills._skill_sync_key("ops", "deploy")}


def _journalled(world, keys):
    """The version the journal holds of each planted record, by kind."""
    latest = {(r.kind.value, r.record_id): r for r in world.feed.current_records()}
    return {kind: latest[(kind, record_id)] for kind, record_id in keys.items()}


def _here(world, keys):
    """What each store holds of each planted record: None when it holds nothing."""
    note = world.notes.get_note(keys["note"])
    conversation = world.conversations.get_conversation(keys["conversation"])
    messages = world.conversations.get_messages(keys["conversation"]) if conversation is not None else []
    fact = world.facts.get(keys["memory_canonical"])
    skill = world.skills.get("deploy", "ops")
    return {"note": note.title if note is not None and not getattr(note, "deleted", False) else None,
            "conversation": (conversation.title, [m.content for m in messages]) if conversation is not None else None,
            "memory_canonical": (fact.text, fact.active) if fact is not None else None,
            "skill": skill.body if skill is not None else None}


def _delete(world, keys):
    """The user deletes each planted record here; each store journals its tombstone."""
    assert world.notes.delete_note(keys["note"]), "control: the note was deleted"
    assert world.conversations.delete_conversation(keys["conversation"]), "control: the conversation was deleted"
    assert world.facts.hard_delete(keys["memory_canonical"]), "control: the fact was deleted"
    assert world.skills.delete("deploy", "ops"), "control: the skill was deleted"


def _round(world, peer):
    return world.engine.run_round("dev-b", peer, approval_fn=lambda _conv, _label, _args: True)


def _version(world, record, *, clock, device="dev-b", deleted=False, payload=None, updated_at=None):
    """Another version of ``record``, made on the paired device."""
    records = world.mod["records"]
    return records.new_record(record.kind, record.record_id, record.payload if payload is None else payload,
                              device=device, clock=clock, deleted=deleted,
                              updated_at=updated_at or record.updated_at)


# ---------------------------------------------------------------------------
# LW1 -- the verdict is the journal's merge
# ---------------------------------------------------------------------------
class _Journal:
    """A journal the merge writes: its versions, and the floors a deletion left."""

    def __init__(self, held, floors):
        self.held, self.floors, self.written = list(held), dict(floors), []

    def current_records(self):
        return list(self.held)

    def clock_floor(self, kind, record_id):
        return self.floors.get((kind.value if hasattr(kind, "value") else kind, record_id), 0)

    def record(self, record):
        self.written.append(record)


def test_lw1_the_verdict_is_the_journals_merge():
    loaded, restore = isolate(
        targets={f"opti_oignon.veilid.{name}": source("veilid", f"{name}.py")
                 for name in ("records", "reconcile", "protocol")},
        seeded={"opti_oignon.veilid.guard": _module("opti_oignon.veilid.guard", assert_sync_allowed=lambda: None)},
        packages=("opti_oignon.veilid",))
    try:
        records, reconcile, protocol = (loaded[f"opti_oignon.veilid.{n}"] for n in ("records", "reconcile",
                                                                                    "protocol"))
        tree = ast.parse(source("veilid", "reconcile.py").read_text(encoding="utf-8"))
        imported = {n.module if isinstance(n, ast.ImportFrom) else a.name for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        assert imported <= {"__future__", "logging", "dataclasses", "typing", "opti_oignon.veilid.records"}, (
            f"the verdict is pure: it reads nothing but its arguments ({imported})")
        kinds = (records.RecordKind.NOTE, records.RecordKind.CONVERSATION)
        draw = random.Random(192)
        judged = wins = 0
        for _ in range(400):
            keys = [(draw.choice(kinds), f"r{draw.randint(1, 3)}") for _ in range(3)]

            def version(kind, rid):
                return records.new_record(kind, rid, {"text": f"t{draw.randint(1, 3)}"}, device=draw.choice("ab"),
                                          clock=draw.randint(1, 5), deleted=draw.random() < 0.3,
                                          updated_at=f"2026-10-28T1{draw.randint(0, 3)}:00:00+00:00")

            held = {key: version(*key) for key in set(keys) if draw.random() < 0.7}
            floors = {(k.value, rid): draw.randint(0, 4) for (k, rid) in set(keys) if draw.random() < 0.3}
            held = {key: r for key, r in held.items() if r.clock > floors.get((key[0].value, key[1]), 0)}
            incoming = [version(*draw.choice(keys)) for _ in range(draw.randint(1, 5))]
            journal = _Journal(held.values(), floors)
            batch = protocol.RecordBatch(device="dev-b", high_water=1, records=incoming, rejected=0, epoch=None)
            protocol.apply_record_batch(journal, batch)
            for r in incoming:
                key = records.key_of(r)
                mine = held.get((r.kind, r.record_id))
                verdict = reconcile.received_wins(r, mine, floor=floors.get(key, 0), rivals=incoming)
                # A version wins exactly when the merge journals it: what is written into a store, the journal takes.
                journalled = any(reconcile._selection_key(w) == reconcile._selection_key(r) for w in journal.written)
                assert verdict == journalled, (r, journal.written, mine, floors.get(key, 0))
                judged += 1
                wins += verdict
        assert judged > 500 and 0 < wins < judged, (judged, wins)
        note = records.RecordKind.NOTE
        v3 = records.new_record(note, "n", {"t": 1}, device="b", clock=3)
        assert reconcile.received_wins(v3, None), "nothing here"
        assert not reconcile.received_wins(v3, v3), "the version held here is not written again"
        relayed = records.new_record(note, "n", {"t": 1}, device="c", clock=3)
        assert not reconcile.received_wins(relayed, v3), "nor its content at its clock, from another device"
        assert not reconcile.received_wins(v3, None, floor=3), "at the floor of a deletion, it is the deleted one"
        tomb = records.new_record(note, "n", {}, device="a", clock=4, deleted=True)
        assert not reconcile.received_wins(v3, tomb), "under a tombstone, it never lands"
        assert reconcile.received_wins(tomb, v3) and not reconcile.received_wins(v3, None, rivals=[tomb])
    finally:
        restore()


# ---------------------------------------------------------------------------
# LW2 -- a journal served again brings nothing deleted back
# ---------------------------------------------------------------------------
def test_lw2_a_journal_served_again_brings_nothing_deleted_back(tmp_path):
    world = _world(tmp_path)
    try:
        keys = _plant(world)
        versions = _journalled(world, keys)
        assert all(_here(world, keys).values()), "control: every store holds its record"
        protocol, records = world.mod["protocol"], world.mod["records"]
        first = _round(world, _Peer(protocol, records, epoch="epoch-1"))
        assert first.applied == 0, "control: the paired device had nothing new"
        _delete(world, keys)
        assert not any(_here(world, keys).values()), "control: the deletions took everything here"
        tombs = _journalled(world, keys)
        assert all(t.deleted for t in tombs.values()), "control: each deletion journalled its tombstone"
        left = _held(world.data, *_WORDS)
        # The paired device's journal was remade: it serves its whole set again, the versions deleted here among it.
        result = _round(world, _Peer(protocol, records, versions.values(), epoch="epoch-2"))
        assert result.epoch_reset, "control: the round saw the journal remade and read it whole"
        assert _here(world, keys) == dict.fromkeys(keys), f"a deleted record came back: {_here(world, keys)}"
        assert _held(world.data, *_WORDS) <= left, f"the round wrote words of a deleted record back: {_held(world.data, *_WORDS)}"
        assert _journalled(world, keys) == tombs, "the journal keeps the tombstones"
        # A version made there after the deletion wins, and lands: the landing itself is not closed.
        later = {kind: _version(world, versions[kind], clock=tombs[kind].clock + 1) for kind in keys}
        _round(world, _Peer(protocol, records, list(versions.values()) + list(later.values()), epoch="epoch-2"))
        back = _here(world, keys)
        assert all(back.values()), f"a version that won did not land: {back}"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# LW3 -- an edit made there before the deletion arrived
# ---------------------------------------------------------------------------
def test_lw3_an_edit_made_there_before_the_deletion_arrived_lands_as_the_journal_decides(tmp_path):
    for attempt in range(2):
        world = _world(tmp_path / f"order{attempt}")
        try:
            keys = _plant(world)
            versions = _journalled(world, keys)
            _delete(world, keys)
            tombs = _journalled(world, keys)
            records, reconcile = world.mod["records"], world.mod["reconcile"]
            # The paired device edited each record from the version it held, at the clock the deletion took here.
            edits = {}
            for kind, version in versions.items():
                for salt in range(1024):
                    edit = _version(world, version, clock=tombs[kind].clock, payload=_edited(kind, version, salt))
                    if (reconcile._selection_key(edit) > reconcile._selection_key(tombs[kind])) == (attempt == 0):
                        edits[kind] = edit
                        break
            assert len(edits) == 4, f"control: an edit that wins, then one that loses, the tie-break ({sorted(edits)})"
            _round(world, _Peer(world.mod["protocol"], records, edits.values()))
            journal = _journalled(world, keys)
            here = _here(world, keys)
            for kind in keys:
                kept = journal[kind]
                assert kept.deleted == (attempt == 1), f"{kind}: control, the journal's tie-break"
                assert (here[kind] is None) == kept.deleted, f"{kind}: the store holds {here[kind]!r}, the journal {kept}"
        finally:
            world.restore()


def _named(kind, version):
    """A tombstone's payload as each store reads one: a conversation's and a skill's name their record."""
    if kind == "conversation":
        return {"conversation": {"id": version.record_id}}
    if kind == "skill":
        skill = version.payload["skill"]
        return {"skill": {"category": skill["category"], "name": skill["name"]}}
    return {}


def _edited(kind, version, salt):
    """The payload of ``version`` with its words changed, ``salt`` choosing among content hashes."""
    payload = dict(version.payload)
    if kind == "note":
        payload["title"] = f"{NOTE_TITLE} edited {salt}"
    elif kind == "conversation":
        payload["conversation"] = dict(payload["conversation"], title=f"edited {salt}")
    elif kind == "memory_canonical":
        payload["fact"] = dict(payload["fact"], text=f"The user's sister is called {FACT} ({salt}).")
    else:
        payload["skill"] = dict(payload["skill"], markdown=payload["skill"]["markdown"] + f"\n<!-- {salt} -->\n")
    return payload


# ---------------------------------------------------------------------------
# LW4 -- a tombstone that loses destroys nothing
# ---------------------------------------------------------------------------
def test_lw4_a_tombstone_that_loses_destroys_nothing(tmp_path):
    world = _world(tmp_path)
    try:
        keys = _plant(world)
        versions = _journalled(world, keys)
        # Each record edited here once more: the paired device deleted the version before.
        assert world.notes.update_note(keys["note"], title=f"{NOTE_TITLE} again")
        world.conversations.add_message(keys["conversation"], "assistant", f"Then {TALK} again.")
        assert world.facts.update(keys["memory_canonical"], text=f"The user's sister is called {FACT}, still.")
        skills = world.mod["skills"]
        world.skills.add("deploy", "ops", f"## When to Use\nShipping.\n\n## Procedure\n{SKILL_STEP}, twice.",
                         source=skills.SOURCE_MANUAL, status=skills.STATUS_PUBLISHED)
        before = _here(world, keys)
        held = _journalled(world, keys)
        assert all(held[k].clock > versions[k].clock for k in keys), "control: each edit raised the clock here"
        stale = [_version(world, versions[k], clock=held[k].clock - 1, deleted=True, payload=_named(k, versions[k]))
                 for k in keys]
        _round(world, _Peer(world.mod["protocol"], world.mod["records"], stale))
        assert _here(world, keys) == before, f"a deletion that lost destroyed: {_here(world, keys)} for {before}"
        assert _journalled(world, keys) == held, "the journal keeps the versions made here"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# LW5 -- an older version never regresses the store
# ---------------------------------------------------------------------------
def test_lw5_an_older_version_never_regresses_the_store(tmp_path):
    world = _world(tmp_path)
    try:
        keys = _plant(world)
        versions = _journalled(world, keys)
        old = {k: _version(world, versions[k], clock=versions[k].clock, payload=_edited(k, versions[k], 0))
               for k in keys}
        assert world.notes.update_note(keys["note"], title=f"{NOTE_TITLE} newer")
        world.conversations.add_message(keys["conversation"], "assistant", "newer")
        assert world.facts.update(keys["memory_canonical"], text=f"The user's sister is called {FACT}, newer.")
        skills = world.mod["skills"]
        world.skills.add("deploy", "ops", "## When to Use\nShipping.\n\n## Procedure\nnewer.",
                         source=skills.SOURCE_MANUAL, status=skills.STATUS_PUBLISHED)
        before = _here(world, keys)
        _round(world, _Peer(world.mod["protocol"], world.mod["records"], old.values()))
        assert _here(world, keys) == before, f"an older version overwrote a newer one: {_here(world, keys)}"
    finally:
        world.restore()
    # The version the journal holds, served again, never overwrites an edit made here that the journal could not take.
    world = _world(tmp_path / "unjournalled")
    try:
        keys = _plant(world)
        held = _journalled(world, keys)

        def locked(_record):
            raise sqlite3.OperationalError("database is locked")

        world.engine.publish = locked
        assert world.notes.update_note(keys["note"], title=f"{NOTE_TITLE} mine")
        world.conversations.add_message(keys["conversation"], "assistant", "mine")
        assert world.facts.update(keys["memory_canonical"], text=f"The user's sister is called {FACT}, mine.")
        skills = world.mod["skills"]
        world.skills.add("deploy", "ops", "## When to Use\nShipping.\n\n## Procedure\nmine.",
                         source=skills.SOURCE_MANUAL, status=skills.STATUS_PUBLISHED)
        del world.engine.publish
        assert _journalled(world, keys) == held, "control: the journal took none of the edits"
        mine = _here(world, keys)
        _round(world, _Peer(world.mod["protocol"], world.mod["records"], held.values()))
        assert _here(world, keys) == mine, f"the version the journal holds overwrote an edit here: {_here(world, keys)}"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# LW6 -- several versions of one record in a batch
# ---------------------------------------------------------------------------
def test_lw6_only_a_batchs_winner_is_written_and_a_refused_winner_takes_its_losers(tmp_path):
    for order in ((3, 5, 4), (5, 4, 3), (4, 3, 5)):
        world = _world(tmp_path / "".join(map(str, order)))
        try:
            keys = _plant(world)
            versions = _journalled(world, keys)
            batch = [_version(world, versions[k], clock=versions[k].clock + c, payload=_edited(k, versions[k], c))
                     for c in order for k in keys]
            _round(world, _Peer(world.mod["protocol"], world.mod["records"], batch))
            here, journal = _here(world, keys), _journalled(world, keys)
            for kind in keys:
                assert journal[kind].clock == versions[kind].clock + 5, f"{order} {kind}: control, the journal"
                assert _marked(here[kind], kind, 5), f"{order} {kind}: the store holds {here[kind]!r}"
        finally:
            world.restore()
    # Every store refuses: only each record's winner is ever asked, and none of its losers is journalled.
    asked = []

    def refusing(record):
        asked.append((record.kind.value, record.clock))
        return False

    world = _world(tmp_path / "refused", conversation_sink=refusing, note_sink=refusing,
                   memory_canonical_sink=refusing, skill_sink=refusing)
    try:
        keys = _plant(world)
        held = _journalled(world, keys)
        for order in ((5, 4), (4, 5)):
            asked.clear()
            batch = [_version(world, held[k], clock=held[k].clock + c, payload=_edited(k, held[k], c))
                     for c in order for k in keys]
            _round(world, _Peer(world.mod["protocol"], world.mod["records"], batch, epoch=f"epoch-{order[0]}"))
            assert sorted(asked) == sorted((k, held[k].clock + 5) for k in keys), (
                f"{order}: the stores were asked for {asked}: only a batch's winner is ever written")
            assert _journalled(world, keys) == held, f"{order}: a refused winner's loser was journalled in its place"
    finally:
        world.restore()


def _marked(value, kind, salt):
    """Whether a store's ``value`` (``_here``) is the version ``_edited`` made with ``salt``."""
    if value is None:
        return False
    if kind == "note":
        return value == f"{NOTE_TITLE} edited {salt}"
    if kind == "conversation":
        return value[0] == f"edited {salt}"
    if kind == "memory_canonical":
        return value[0].endswith(f"({salt}).")
    return f"<!-- {salt} -->" in value


# ---------------------------------------------------------------------------
# LW7 -- the floor
# ---------------------------------------------------------------------------
def test_lw7_a_version_at_the_floor_of_a_deletion_is_never_written(tmp_path):
    world = _world(tmp_path)
    try:
        keys = _plant(world)
        versions = _journalled(world, keys)
        before = _here(world, keys)
        # What a deletion leaves when its tombstone cannot be journalled: the versions erased, their clock kept as
        # the key's floor, no tombstone.
        for kind, record_id in keys.items():
            world.feed.forget_record(kind, record_id)
        floors = {k: world.feed.clock_floor(k, keys[k]) for k in keys}
        assert all(floors[k] == versions[k].clock for k in keys), f"control: the floors {floors}"
        at_floor = [_version(world, versions[k], clock=floors[k], payload=_edited(k, versions[k], 1)) for k in keys]
        _round(world, _Peer(world.mod["protocol"], world.mod["records"], at_floor))
        assert _here(world, keys) == before, f"a version at the floor was written: {_here(world, keys)}"
        above = [_version(world, versions[k], clock=floors[k] + 1, payload=_edited(k, versions[k], 2)) for k in keys]
        _round(world, _Peer(world.mod["protocol"], world.mod["records"], at_floor + above))
        after = _here(world, keys)
        assert all(after[k] != before[k] for k in keys), f"a version above the floor did not land: {after}"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# LW8 -- a loser goes on to the merge, unwritten
# ---------------------------------------------------------------------------
def test_lw8_a_loser_goes_on_to_the_merge_unwritten(tmp_path):
    asked = []
    world = _world(tmp_path, note_sink=lambda record: asked.append(record) or True)
    try:
        keys = _plant(world)
        reconcile = world.mod["reconcile"]
        rival = None
        for attempt in range(20):
            # A rival at the clock held that loses the tie-break; when none does, another version here, another hash.
            held = _journalled(world, keys)["note"]
            rival = next((r for r in (_version(world, held, clock=held.clock,
                                               payload=dict(held.payload, title=f"rival {salt}")) for salt in range(256))
                          if reconcile._selection_key(r) < reconcile._selection_key(held)), None)
            if rival is not None:
                break
            assert world.notes.update_note(keys["note"], title=f"{NOTE_TITLE} {attempt}")
        assert rival is not None, "control: a rival that loses the tie-break"
        result = _round(world, _Peer(world.mod["protocol"], world.mod["records"], [rival]))
        assert asked == [], "a version that lost the tie-break was written"
        assert result.conflicts == 1 and result.applied == 0, "the merge keeps the divergence in its conflict log"
        assert _journalled(world, keys)["note"] == held

        world.engine._feed = _unreadable(world.feed)
        later = _version(world, held, clock=held.clock + 1, payload=dict(held.payload, title="later"))
        records = world.mod["records"]
        update = records.new_record(records.RecordKind.NOTE_UPDATE, f"{keys['note']}:1", {
            "note_id": keys["note"], "seq": 1, "update_blob_b64": base64.b64encode(UPDATE).decode("ascii")},
            device="dev-b", clock=1)
        raised = False
        try:
            _round(world, _Peer(world.mod["protocol"], records, [rival, update, later]))
        except sqlite3.OperationalError:
            raised = True
        assert raised, "control: the round could not read the journal"
        assert asked == [], "a store was written while the journal could not be read"
        assert not _held(world.data, UPDATE), "a note's update was written while the journal could not be read"
    finally:
        world.restore()
    # A leg that brings a note's update and no full version: the journal is read all the same.
    world = _world(tmp_path / "updates")
    try:
        keys = _plant(world)
        records = world.mod["records"]
        update = records.new_record(records.RecordKind.NOTE_UPDATE, f"{keys['note']}:1", {
            "note_id": keys["note"], "seq": 1, "update_blob_b64": base64.b64encode(UPDATE).decode("ascii")},
            device="dev-b", clock=1)
        world.engine._feed = _unreadable(world.feed)
        raised = False
        try:
            _round(world, _Peer(world.mod["protocol"], records, [update]))
        except sqlite3.OperationalError:
            raised = True
        assert raised, "control: the round could not read the journal"
        assert not _held(world.data, UPDATE), "a note's update was written while the journal could not be read"
    finally:
        world.restore()


def _unreadable(feed):
    """The journal as a round sees it when another writer holds it: every read of its versions refused."""

    class _Unreadable:
        def __getattr__(self, name):
            return getattr(feed, name)

        def current_records(self):
            raise sqlite3.OperationalError("database is locked")

    return _Unreadable()


# ---------------------------------------------------------------------------
# LW9 -- the census
# ---------------------------------------------------------------------------
def test_lw9_the_engine_owes_nothing_and_a_landing_without_its_verdict_is_a_finding():
    loaded, restore = isolate(targets={"write_census_guard": _GUARD})
    try:
        guard = loaded["write_census_guard"]
        assert _ENGINE not in guard.LEDGER, "the engine still owes"
        gate = guard.GATES["sync.order"]
        assert (gate.kind, gate.home, gate.name) == ("verdict", "opti_oignon/veilid/reconcile.py",
                                                          "received_wins"), gate
        stores = {"conversation", "facts_canonical", "notes", "skills"}
        assert {store for (rel, store) in guard.GATED if rel == _ENGINE} == stores
        assert all(guard.GATED[(_ENGINE, store)] == ("sync.order",) for store in stores)
        estate = guard.read_estate(REPO)
        result = guard.run(REPO)
        assert result.code == 0, "\n".join(result.lines)
        sites = sorted((s.store, s.method, s.function) for s in result.census.sites()[_ENGINE])
        assert sites == [("conversation", "apply_synced_conversation", "_default_conversation_sink"),
                         ("facts_canonical", "apply_synced_memory_canonical", "_default_memory_canonical_sink"),
                         ("notes", "apply_synced_note", "_default_note_sink"),
                         ("skills", "apply_synced_skill", "_default_skill_sink")], sites
        text = estate.modules[_ENGINE]
        # Two landers broken at a time, one with no verdict, one asking it after its store is resolved.
        for silenced, moved in ((("_land_notes", "notes"), ("_land_conversations", "conversation")),
                                (("_land_skills", "skills"), ("_land_memory_canonical", "facts_canonical"))):
            broken = _after_the_sink(_silenced(text, silenced[0]), moved[0])
            found = guard.find_all(guard.take_census(estate._replace(modules={**estate.modules, _ENGINE: broken})))
            ungated = sorted(f for f in found.get("ungated", []) if _ENGINE in f)
            assert len(ungated) == 2, (silenced, moved, ungated)
            assert {silenced[1], moved[1]} == {s for s in ("conversation", "facts_canonical", "notes", "skills")
                                               if any(f" {s} write " in f for f in ungated)}, ungated
    finally:
        restore()


_VERDICT = "            if not received_wins("


def _silenced(text, lander):
    """The engine's text with ``lander`` asking no verdict."""
    at = text.index(_VERDICT, text.index(f"    def {lander}("))
    return text[:at] + "            if False and not received_wins(" + text[at + len(_VERDICT):]


def _after_the_sink(text, lander):
    """The engine's text with ``lander`` resolving its store before it asks the verdict."""
    start = text.index(f"    def {lander}(")
    block = re.search(r"            if not sink_resolved:\n                sink_resolved = True\n"
                      r"                sink = self\._resolve_\w+\(\)\n", text[start:]).group(0)
    moved = text[:start] + text[start:].replace(block, "", 1)
    at = moved.index(_VERDICT, start)
    return moved[:at] + block + moved[at:]


# ---------------------------------------------------------------------------
# LW10 -- an injected sink is behind the same verdict
# ---------------------------------------------------------------------------
def test_lw10_an_injected_sink_is_behind_the_same_verdict(tmp_path):
    asked = []

    def spy(record):
        asked.append((record.kind.value, record.clock, record.deleted))
        return True

    world = _world(tmp_path, conversation_sink=spy, note_sink=spy, memory_canonical_sink=spy, skill_sink=spy)
    try:
        keys = _plant(world)
        versions = _journalled(world, keys)
        _delete(world, keys)
        tombs = _journalled(world, keys)
        served = list(versions.values())
        _round(world, _Peer(world.mod["protocol"], world.mod["records"], served))
        assert asked == [], f"an injected sink was handed a version that lost: {asked}"
        # Each record's winner first, then a version it beats: the sink resolved for the winner never takes it.
        later = [_version(world, versions[k], clock=tombs[k].clock + c, payload=_edited(k, versions[k], c))
                 for k in keys for c in (2, 1)]
        _round(world, _Peer(world.mod["protocol"], world.mod["records"], served + later))
        assert sorted(asked) == sorted((k, tombs[k].clock + 2, False) for k in keys), asked
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# LW11 -- a peer's deletion that wins at the clock held here is journalled
# ---------------------------------------------------------------------------
def test_lw11_a_peers_deletion_that_wins_the_tie_break_is_journalled_and_keeps_no_words(tmp_path):
    world = _world(tmp_path)
    try:
        keys = _plant(world)
        records, reconcile = world.mod["records"], world.mod["reconcile"]
        cid = keys["conversation"]
        # The paired device deleted the conversation from the version it held, at the clock the version here has:
        # edits made here until the tie-break gives the deletion the win.
        for salt in range(200):
            held = _journalled(world, keys)["conversation"]
            tomb = records.new_record(records.RecordKind.CONVERSATION, cid, {"conversation": {"id": cid}},
                                      device="dev-b", clock=held.clock, deleted=True)
            if reconcile._selection_key(tomb) > reconcile._selection_key(held):
                break
            world.conversations.add_message(cid, "user", f"{TALK} once more ({salt}).")
        assert reconcile._selection_key(tomb) > reconcile._selection_key(held), "control: the deletion wins the tie"
        result = _round(world, _Peer(world.mod["protocol"], records, [tomb]))
        assert _here(world, keys)["conversation"] is None, "control: the deletion was applied here"
        kept = _journalled(world, keys)["conversation"]
        assert kept.deleted and kept.clock == held.clock, f"the deletion was not journalled: {kept}"
        assert result.applied == 1, result
        assert not _held(world.data, TALK.encode()), "the conversation's words stayed on disk, its journal included"
    finally:
        world.restore()
    # The journal's write failing at the merge, for a tombstone at the clock held and for one above it: the words
    # leave the journal with the store's, an older version served meanwhile still loses, and the tombstone delivered
    # again is journalled.
    for above in (0, 1):
        world = _world(tmp_path / f"failing{above}")
        try:
            keys = _plant(world)
            records, reconcile = world.mod["records"], world.mod["reconcile"]
            cid = keys["conversation"]
            first = _journalled(world, keys)["conversation"]
            # One edit here at least: the version served again below is older than the tombstone.
            world.conversations.add_message(cid, "user", f"{TALK} again.")
            for salt in range(200):
                held = _journalled(world, keys)["conversation"]
                tomb = records.new_record(records.RecordKind.CONVERSATION, cid, {"conversation": {"id": cid}},
                                          device="dev-b", clock=held.clock + above, deleted=True)
                if above or reconcile._selection_key(tomb) > reconcile._selection_key(held):
                    break
                world.conversations.add_message(cid, "user", f"{TALK} once more ({salt}).")

            def refused(_record):
                raise sqlite3.OperationalError("database is locked")

            assert first.clock < tomb.clock, "control: the version served again is older than the tombstone"
            world.feed.record = refused
            try:
                _round(world, _Peer(world.mod["protocol"], records, [tomb]))
            except sqlite3.OperationalError:
                pass
            del world.feed.record
            assert _here(world, keys)["conversation"] is None, "control: the deletion was applied here"
            assert ("conversation", cid) not in {(r.kind.value, r.record_id) for r in world.feed.current_records()}
            assert not _held(world.data, TALK.encode()), "a peer's deletion left the conversation's words in the journal"
            assert world.feed.clock_floor("conversation", cid) == tomb.clock - 1, "the floor is not under the tombstone"
            _round(world, _Peer(world.mod["protocol"], records, [first]))
            assert _here(world, keys)["conversation"] is None, "an older version came back while the tombstone waited"
            _round(world, _Peer(world.mod["protocol"], records, [first, tomb], epoch="epoch-2"))
            kept = _journalled(world, keys)["conversation"]
            assert kept.deleted and kept.clock == tomb.clock, f"the tombstone delivered again was not journalled: {kept}"
        finally:
            world.restore()


# ---------------------------------------------------------------------------
# LW12 -- a skill approved from the panel lands behind the same verdict
# ---------------------------------------------------------------------------
def test_lw12_a_skill_approved_from_the_panel_lands_behind_the_same_verdict(tmp_path):
    world = _world(tmp_path)
    try:
        skills, records = world.mod["skills"], world.mod["records"]
        key = skills._skill_sync_key("ops", "release")
        markdown = skills.Skill(name="release", category="ops", status=skills.STATUS_PUBLISHED,
                                body=f"## When to Use\nReleasing.\n\n## Procedure\n{SKILL_STEP}, then tag.").to_markdown()
        version = records.new_record(records.RecordKind.SKILL, key,
                                     {"skill": {"category": "ops", "name": "release", "markdown": markdown}},
                                     device="dev-b", clock=1)
        world.engine.run_round("dev-b", _Peer(world.mod["protocol"], records, [version]),
                               approval_fn=lambda _conv, _label, _args: False)
        assert [(e.kind, e.record_id) for e in world.engine.list_deferred()] == [("skill", key)], "control: deferred"
        assert world.skills.get("release", "ops") is None, "control: nothing written before the approval"
        answer = world.engine.approve_deferred("skill", key)
        assert answer["approved"] and answer["applied"] == 1, answer
        landed = world.skills.get("release", "ops")
        assert landed is not None and SKILL_STEP in landed.body, "the approved skill never reached the registry"
        assert world.skills.sync_state("release", "ops") == skills.SYNC_UNADOPTED, "a received skill is marked"
        latest = {(r.kind.value, r.record_id): r for r in world.feed.current_records()}
        assert latest[("skill", key)] == version, "the journal holds what was written"
        # The registry refusing at the approval: nothing journalled, the entry waits, and it lands at the next one.
        skills_older = records.new_record(records.RecordKind.SKILL, skills._skill_sync_key("ops", "audit"), {
            "skill": {"category": "ops", "name": "audit", "markdown": markdown.replace("release", "audit")}},
            device="dev-b", clock=1)
        world.engine.run_round("dev-b", _Peer(world.mod["protocol"], records, [version, skills_older]),
                               approval_fn=lambda _conv, _label, _args: False)
        world.engine._skill_sink = lambda _record: False
        answer = world.engine.approve_deferred("skill", skills_older.record_id)
        assert not answer["approved"] and not answer["refused"] and answer["reason"] == "unwritten", answer
        assert [e.record_id for e in world.engine.list_deferred()] == [skills_older.record_id], "the entry went"
        assert ("skill", skills_older.record_id) not in {(r.kind.value, r.record_id)
                                                         for r in world.feed.current_records()}
        world.engine._skill_sink = None
        assert world.engine.approve_deferred("skill", skills_older.record_id)["applied"] == 1
        assert world.skills.get("audit", "ops") is not None and world.engine.list_deferred() == []
        # A deferred version that an edit here beat meanwhile: approved, it is written nowhere.
        older = records.new_record(records.RecordKind.SKILL, key, {"skill": {
            "category": "ops", "name": "release", "markdown": markdown.replace("then tag", "then wait")}},
            device="dev-b", clock=2)
        world.engine.run_round("dev-b", _Peer(world.mod["protocol"], records, [version, skills_older, older]),
                               approval_fn=lambda _conv, _label, _args: False)
        assert [(e.kind, e.record_id) for e in world.engine.list_deferred()] == [("skill", key)], "control: deferred"
        for _ in range(2):
            world.skills.add("release", "ops", "## When to Use\nReleasing.\n\n## Procedure\nThen sign.",
                             source=skills.SOURCE_MANUAL, status=skills.STATUS_PUBLISHED)
        mine = {(r.kind.value, r.record_id): r for r in world.feed.current_records()}[("skill", key)]
        assert mine.clock == 3 and mine.device == "dev-a", f"control: the edits here were journalled: {mine}"
        answer = world.engine.approve_deferred("skill", key)
        assert answer["approved"] and answer["applied"] == 0, answer
        assert "Then sign." in world.skills.get("release", "ops").body, "an approval wrote a version that lost"
        assert {(r.kind.value, r.record_id): r for r in world.feed.current_records()}[("skill", key)] == mine
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# LW13 -- a landing that writes the journal makes the leg read it again
# ---------------------------------------------------------------------------
def test_lw13_a_conversations_deletion_makes_the_leg_read_the_journal_again(tmp_path):
    for salt in range(12):
        world = _world(tmp_path / f"salt{salt}")
        try:
            keys = _plant(world)
            held = _journalled(world, keys)
            records, fact = world.mod["records"], keys["memory_canonical"]
            manager, facts = world.conversations, world.facts

            def forgetting(record):
                # The conversation's deletion, and its forgetting of a fact drawn from it, whose tombstone enters the
                # journal (the facts' own erasure, as the conversation's forgetters ask it).
                landed = manager.apply_synced_conversation(record.payload, deleted=record.deleted,
                                                           updated_at=record.updated_at)
                facts.erase(fact)
                return landed

            world.engine._conversation_sink = forgetting
            cid = keys["conversation"]
            tomb = records.new_record(records.RecordKind.CONVERSATION, cid, {"conversation": {"id": cid}},
                                      device="dev-b", clock=held["conversation"].clock + 1, deleted=True)
            edit = _version(world, held["memory_canonical"], clock=held["memory_canonical"].clock + 1,
                            payload=_edited("memory_canonical", held["memory_canonical"], salt))
            _round(world, _Peer(world.mod["protocol"], records, [tomb, edit]))
            kept = _journalled(world, keys)["memory_canonical"]
            here = _here(world, keys)["memory_canonical"]
            assert (here is None) == kept.deleted, f"salt {salt}: the store holds {here!r}, the journal {kept}"
        finally:
            world.restore()
