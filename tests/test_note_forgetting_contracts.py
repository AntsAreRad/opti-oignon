#!/usr/bin/env python3
"""Contracts for a note's deletion that keeps no words, and the wipe that removes the notes when they are named.

A deleted note kept everything: ``delete_note`` set ``deleted = 1`` and left the title, the body, the tags, every
attachment with its transcript, caption and OCR text, every blob, the whole update tail and its checkpoint, and the
sync journal kept serving the body's increments to every paired device -- an update is never a tombstone. A
paired device that received the tombstone kept the same. Now a deletion, this device's or a peer's, forgets: the
row blanked to its tombstone, the attachments and their blobs (overwritten), the tail, the journal's versions and
updates of the note, the freed pages overwritten and the logs emptied; a sweep finishes what an older or an
interrupted deletion left. The account wipe removes every note -- single-user mode, the install's one user -- only
when the request names them and repeats the phrase its preview gave; the export carries what that wipe removes; the
settings show the user their data, the export first.

  * DN1 -- A DELETED NOTE KEEPS NO WORDS: read back in the bytes of every file, the log included; the blob
    overwritten before its removal; a note not deleted keeps all of its own.
  * DN2 -- THE SYNC JOURNAL FORGETS THE NOTE'S UPDATES AND VERSIONS: none served after the deletion, the tombstone
    is, out-clocking what was erased; another note's updates stay, a note whose id starts the same included.
  * DN3 -- A PEER'S TOMBSTONE FORGETS ONCE THE JOURNAL HOLDS IT (it won here), hook-free; landed before the verdict,
    it destroys nothing.
  * DN4 -- THE SWEEP FORGETS WHAT EARLIER DELETIONS LEFT, and never touches a live note.
  * DN5 -- A VERSION THAT BRINGS A FORGOTTEN NOTE BACK LANDS WITH ITS BODY; one for a live note touches only its
    metadata.
  * DN6 -- AN UPDATE FOR A DEAD NOTE IS REFUSED, AND NEVER JOURNALLED.
  * DN7 -- THE PREVIEW, AND EVERY REFUSAL BEFORE ANYTHING IS REMOVED: the phrase binds the counts; 422 for a word
    the wipe does not know or a missing phrase, 409 outside single-user mode, for another name, or when the counts
    moved, 503 when the notes cannot be read -- through the routes too.
  * DN8 -- THE WIPE WITH THE NOTES NAMED FORGETS EVERY NOTE, its tombstone published, counted, and its labels leave
    what stays; without them the notes are untouched and named.
  * DN9 -- A STEP THAT FAILS IS NAMED, AND STOPS NONE OF THE OTHERS: the blob kept with its row for the next sweep.
  * DN10 -- THE EXPORT CARRIES WHAT THE WIPE REMOVES: every note with its updates and attachments' text, every
    conversation of the one user; what could not be read named; in multi-user mode, neither, and both named.
  * DN11 -- THE REQUEST'S WORDS ARE CLOSED AND DERIVED from the inventory.
  * DN12 -- THE PANEL OFFERS THE EXPORT FIRST AND ASKS THE PHRASE: its rules run under Node; it is placed and wired.
  * DN13 -- A PEER'S TOMBSTONE THAT LOSES DESTROYS NOTHING: the note dead here and alive in the journal keeps its
    words through every sweep, and the wipe deletes it as the user's own deletion, its tombstone out-clocking.
  * DN14 -- AN UPDATE RACING THE DELETION IS NEVER KEPT: the deletion waits for an append in flight, then forgets it.
  * DN15 -- A DELETION THE JOURNAL COULD NOT TAKE IS SAID, AND TOLD AT THE NEXT: the peers not told are named, and
    the next deletion or sweep publishes the tombstone owed.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import ast
import asyncio
import base64
import os
import sqlite3
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_dn1_a_deleted_note_keeps_no_words": 5.0,
    "test_dn2_the_sync_journal_forgets_the_notes_updates_and_versions": 5.0,
    "test_dn3_a_peers_tombstone_forgets_once_the_journal_holds_it": 5.0,
    "test_dn4_the_sweep_forgets_what_earlier_deletions_left_and_never_a_live_note": 5.0,
    "test_dn5_a_version_that_brings_a_forgotten_note_back_lands_with_its_body": 5.0,
    "test_dn6_an_update_for_a_dead_note_is_refused_and_never_journalled": 5.0,
    "test_dn7_the_preview_and_every_refusal_before_anything_is_removed": 10.0,
    "test_dn8_the_wipe_with_the_notes_named_forgets_every_note": 10.0,
    "test_dn9_a_step_that_fails_is_named_and_stops_none_of_the_others": 5.0,
    "test_dn10_the_export_carries_what_the_wipe_removes": 5.0,
    "test_dn11_the_requests_words_are_closed_and_derived": 3.0,
    "test_dn12_the_panel_offers_the_export_first_and_asks_the_phrase": 30.0,
    "test_dn13_a_peers_tombstone_that_loses_destroys_nothing": 10.0,
    "test_dn14_an_update_racing_the_deletion_is_never_kept": 10.0,
    "test_dn15_a_deletion_the_journal_could_not_take_is_said_and_told_at_the_next": 5.0,
}

# The words planted, each a byte string no file holds by chance.
TITLE, BODY, TAG = "TITLE-WORDS-7", b"BODY-WORDS-7", "TAG-WORDS-7"
TRANSCRIPT, CAPTION, OCR = "TRANSCRIPT-WORDS-7", "CAPTION-WORDS-7", "OCR-WORDS-7"
UPDATE, BLOB, THUMB = b"UPDATE-WORDS-7", b"BLOB-WORDS-7", b"THUMB-WORDS-7"
KEEP = "KEEP-WORDS-7"
_WORDS = (TITLE.encode(), BODY, TAG.encode(), TRANSCRIPT.encode(), CAPTION.encode(), OCR.encode(), UPDATE, BLOB,
          THUMB, base64.b64encode(BODY), base64.b64encode(UPDATE))
_NOTE_LABELS = ("notes and their updates", "note attachments (voice, images)")


def _plain(path, *, check_same_thread=False, **_options):
    """A plain connection that, like many builds, does not overwrite deleted pages unless asked to."""
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=0.3)
    conn.execute("PRAGMA secure_delete = OFF")
    return conn


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _bytes(path):
    """A database file's bytes with its log and its shared memory."""
    path = Path(path)
    return b"".join(p.read_bytes() for p in path.parent.glob(path.name + "*") if p.is_file())


def _held(data, *words):
    """Which of ``words`` any file under ``data`` still holds."""
    found = set()
    for path in data.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            found.update(w for w in words if w in raw)
    return found


class _Engine:
    """The sync engine as the stores see it: the clock of a key, and the publication of a note or of an update,
    journalled in the process feed as the real engine journals them."""

    def __init__(self, change_feed, records):
        self.feed, self.records = change_feed, records
        self.published = []

    def current_clock(self, kind, record_id):
        return self.feed.get_change_feed().current_clock(kind, record_id)

    def publish_note(self, note_id, payload=None, *, clock, deleted=False, updated_at=""):
        self.published.append(("note", note_id, deleted))
        return self.feed.get_change_feed().record(self.records.new_record(
            self.records.RecordKind.NOTE, note_id, payload or {}, device="dev-a", clock=clock, deleted=deleted,
            updated_at=updated_at))

    def publish_note_update(self, note_id, seq, payload=None, *, clock):
        self.published.append(("note_update", f"{note_id}:{seq}", False))
        return self.feed.get_change_feed().record(self.records.new_record(
            self.records.RecordKind.NOTE_UPDATE, f"{note_id}:{seq}", payload or {}, device="dev-a", clock=clock))


def _world(tmp_path, *, wipe=False, single=True, auth=True, engine=True, routes=False, seeded=None):
    """The notes stores and the sync journal over ``tmp_path/data``, from their sources; with ``engine``, the
    engine's publication into the journal; with ``wipe``, the account wipe, the auth manager in single-user mode or
    not (``auth=False``: not readable), one account, ``u-1``; with ``routes``, the users' routes."""
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    stand_ins = {
        "opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain),
        "opti_oignon.config": _module("opti_oignon.config", DATA_DIR=data),
    }
    if engine:
        stand_ins["opti_oignon.veilid.guard"] = _module("opti_oignon.veilid.guard", veilid_available=lambda: True)
    targets = {
        "opti_oignon.veilid.records": source("veilid", "records.py"),
        "opti_oignon.veilid.change_feed": source("veilid", "change_feed.py"),
        "opti_oignon.notes.blob_store": source("notes", "blob_store.py"),
        "opti_oignon.notes.notes_store": source("notes", "notes_store.py"),
        "opti_oignon.notes.note_updates_store": source("notes", "note_updates_store.py"),
    }
    if wipe or routes:
        stand_ins["opti_oignon.run_journal"] = _module("opti_oignon.run_journal", wipe=lambda: {"wiped": True})
        if auth:
            stand_ins["opti_oignon.auth"] = _module("opti_oignon.auth", auth_manager=SimpleNamespace(
                single_user_mode=single, count_users=lambda: 1,
                get_user=lambda uid: SimpleNamespace(id=uid) if uid == "u-1" else None))
        targets["opti_oignon.user_data_manager"] = source("user_data_manager.py")
    if routes:
        targets["opti_oignon.api.routes_users"] = source("api", "routes_users.py")
    stand_ins.update(seeded or {})
    holder = {}
    if engine:
        stand_ins["opti_oignon.veilid.sync_engine"] = _module(
            "opti_oignon.veilid.sync_engine", get_sync_engine=lambda: holder["engine"])
    loaded, restore = isolate(targets=targets, seeded=stand_ins,
                              packages=("opti_oignon.notes", "opti_oignon.veilid", "opti_oignon.api"))
    feed_module, records = loaded["opti_oignon.veilid.change_feed"], loaded["opti_oignon.veilid.records"]
    feed_module.set_change_feed(feed_module.ChangeFeed(root=data))
    holder["engine"] = _Engine(feed_module, records)
    notes_module = loaded["opti_oignon.notes.notes_store"]
    updates_module = loaded["opti_oignon.notes.note_updates_store"]
    store = notes_module.NotesStore(root=data)
    notes_module._store = store
    updates = updates_module.NoteUpdatesStore(root=data)
    updates_module._store = updates

    def close():
        try:
            feed_module.reset_change_feed()
        finally:
            restore()

    return SimpleNamespace(data=data, loaded=loaded, notes=store, updates=updates, feed=feed_module, records=records,
                           engine=holder["engine"], notes_module=notes_module, updates_module=updates_module,
                           guard=stand_ins.get("opti_oignon.veilid.guard"), restore=close)


def _plant(world, *, title=TITLE, body=BODY, tag=TAG, attachment="att-1", witness=None):
    """A note holding words everywhere a note keeps them: its row, an attachment with its derived text and its
    blob, two updates and a checkpoint, and their versions in the journal; ``witness``, a hard link to the blob."""
    note = world.notes.add_note(title, body_crdt=body, tags=f'["{tag}"]')
    world.notes.add_attachment(note.id, "audio", blob_ref=attachment, mime="audio/ogg", byte_size=len(BLOB),
                               attachment_id=attachment, transcript_text=TRANSCRIPT, caption_text=CAPTION,
                               ocr_text=OCR, thumbnail=THUMB)
    folder = world.data / "notes_blobs"
    folder.mkdir(exist_ok=True)
    blob = folder / f"{attachment}.blob"
    blob.write_bytes(BLOB * 64)
    # A sealing interrupted: its partial file beside the blob.
    (folder / f"{attachment}.blob.tmp").write_bytes(BLOB)
    if witness is not None:
        os.link(blob, witness)
    world.updates.append_update(note.id, UPDATE + b"-1")
    world.updates.append_update(note.id, UPDATE + b"-2")
    world.updates.set_checkpoint_watermark(note.id, 1)
    return note


def _served(world):
    return world.feed.get_change_feed().since(0).records


def _journal_tombstone(world, note_id, *, device="dev-b"):
    """A note's tombstone journalled as a round journals a peer's that won here: above the version held."""
    feed = world.feed.get_change_feed()
    return feed.record(world.records.new_record(world.records.RecordKind.NOTE, note_id, {}, device=device,
                                                clock=feed.current_clock("note", note_id) + 1, deleted=True))


# ---------------------------------------------------------------------------
# DN1 -- a deleted note keeps no words
# ---------------------------------------------------------------------------
def _reader(path):
    """A connection that has read the database and stays open: the last connection's close no longer empties the
    write-ahead log, which only the deletion's own emptying then does."""
    conn = sqlite3.connect(str(path))
    conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
    return conn


def test_dn1_a_deleted_note_keeps_no_words(tmp_path):
    world = _world(tmp_path)
    readers = []
    try:
        witness = tmp_path / "witness"
        note = _plant(world, witness=witness)
        kept = world.notes.add_note(KEEP, body_crdt=KEEP.encode(), tags=f'["{KEEP}"]')
        world.updates.append_update(kept.id, KEEP.encode() + b"-update")
        # Both stores overwrite what they free, whatever their connection's default.
        for store in (world.notes, world.updates):
            conn = store._connect()
            try:
                assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1, f"{store.db_path} keeps freed pages"
            finally:
                conn.close()
        readers = [_reader(world.data / "notes.db"), _reader(world.data / "note_updates.db")]
        assert world.notes.delete_note(note.id) is True
        assert world.notes.delete_note(note.id) is False, "a dead note is not deleted twice"

        assert _held(world.data, *_WORDS) == set(), f"a deleted note's words stay: {_held(world.data, *_WORDS)}"
        assert not (world.data / "notes_blobs" / "att-1.blob").exists(), "the blob stays"
        assert not (world.data / "notes_blobs" / "att-1.blob.tmp").exists(), "the partial blob stays"
        assert set(witness.read_bytes()) == {0}, "the blob was removed, not overwritten first"
        raw = sqlite3.connect(str(world.data / "notes.db"))
        try:
            row = raw.execute("SELECT title, body_crdt, tags, pinned, deleted FROM note WHERE id = ?",
                              (note.id,)).fetchone()
            assert row == ("", b"", "[]", 0, 1), f"the tombstone is not bare: {row}"
            assert raw.execute("SELECT COUNT(*) FROM attachment WHERE note_id = ?", (note.id,)).fetchone() == (0,)
        finally:
            raw.close()
        assert world.updates.list_updates(note.id) == [], "the update tail stays"
        assert world.updates.get_checkpoint_watermark(note.id) in (None, 0), "the checkpoint stays"
        # The other note keeps all of its own.
        assert KEEP.encode() in _bytes(world.data / "notes.db")
        assert world.notes.get_note(kept.id).title == KEEP
        assert [u.update_blob for u in world.updates.list_updates(kept.id)] == [KEEP.encode() + b"-update"]
        # A deletion without its sweep -- the wipe's, one note after the other -- and with no tombstone journalled
        # to forget after it leaves its own row bare already.
        alone = world.notes.add_note("ALONE-WORDS-7", body_crdt=b"ALONE-BODY-7", tags='["ALONE-TAG-7"]')
        world.guard.veilid_available = lambda: False
        died, left = world.notes.forget_note(alone.id, sweep=False)
        assert died and [line for line in left if not line.startswith("paired devices (not told")] == [], left
        assert [w for w in (b"ALONE-WORDS-7", b"ALONE-BODY-7", b"ALONE-TAG-7")
                if w in _bytes(world.data / "notes.db")] == [], "a deletion alone keeps the note's words"
    finally:
        for conn in readers:
            conn.close()
        world.restore()


# ---------------------------------------------------------------------------
# DN2 -- the sync journal forgets the note's updates and versions
# ---------------------------------------------------------------------------
def test_dn2_the_sync_journal_forgets_the_notes_updates_and_versions(tmp_path):
    world = _world(tmp_path)
    try:
        note = _plant(world)
        # A note whose id begins as this one's does, and an unrelated one, with updates of their own.
        sibling = world.notes.add_note("sibling", note_id=note.id + "0")
        world.updates.append_update(sibling.id, b"sibling-update")
        lettered = world.notes.add_note("lettered", note_id=note.id + "z")
        world.updates.append_update(lettered.id, b"lettered-update")
        before = {(r.kind.value, r.record_id): r.clock for r in _served(world)}
        assert ("note_update", f"{note.id}:1") in before and ("note", note.id) in before, before

        assert world.notes.delete_note(note.id) is True
        served = {(r.kind.value, r.record_id): r for r in _served(world)}
        assert [k for k in served if k[0] == "note_update" and k[1].startswith(note.id + ":")] == [], \
            "the journal still serves the dead note's updates"
        tomb = served.get(("note", note.id))
        assert tomb is not None and tomb.deleted, "the tombstone is not served"
        assert tomb.clock > before[("note", note.id)], "the tombstone does not out-clock the version it erased"
        assert ("note_update", f"{sibling.id}:1") in served and ("note", sibling.id) in served, \
            "another note's records went with it"
        assert ("note_update", f"{lettered.id}:1") in served, "another note's records went with it"
        # An id with a colon would share another note's update keys: refused here and from a peer.
        with pytest.raises(ValueError):
            world.notes.add_note("colon", note_id=f"{lettered.id}:1")
        assert world.notes.apply_synced_note(f"{sibling.id}:x", {"title": "colon"}) is False
        assert world.notes.get_note(f"{sibling.id}:x") is None
        journal = _bytes(world.data / "veilid_change_feed.db")
        assert [w for w in (TITLE.encode(), base64.b64encode(BODY), base64.b64encode(UPDATE + b"-1"))
                if w in journal] == [], "the journal's file keeps the words"
        # Deleted again -- a retry, a second tab -- the note keeps the tombstone the journal serves.
        assert world.notes.forget_note(note.id) == (False, [])
        assert [(r.clock, r.deleted) for r in _served(world) if r.kind.value == "note" and r.record_id == note.id] \
            == [(tomb.clock, True)], "a second deletion erased the tombstone"
        # Deleted from a stale screen after a peer's tombstone won here: the peer's tombstone stays.
        peer = world.notes.add_note("peer-deleted")
        assert world.notes.apply_synced_note(peer.id, None, deleted=True)
        _journal_tombstone(world, peer.id)
        world.engine.published.clear()
        assert world.notes.forget_note(peer.id) == (False, [])
        assert [(r.device, r.deleted) for r in _served(world) if r.kind.value == "note" and r.record_id == peer.id] \
            == [("dev-b", True)], "a stale deletion erased the peer's tombstone"
        assert world.engine.published == []
    finally:
        world.restore()
    # With no tombstone to publish (the sync framework absent), the deletion still erases the note's versions and
    # updates from the journal, its clock kept as the key's floor.
    world = _world(tmp_path / "unpublished")
    try:
        note = _plant(world)
        top = world.feed.get_change_feed().current_clock("note", note.id)
        world.guard.veilid_available = lambda: False
        # The deletion alone, without the sweep that would also erase what a dead note left; the peers not told,
        # said.
        died, left = world.notes.forget_note(note.id, sweep=False)
        assert died and [line for line in left if line.startswith("paired devices (not told")] == left and left, left
        assert [r for r in _served(world) if r.record_id == note.id or r.record_id.startswith(note.id + ":")] == []
        assert world.feed.get_change_feed().clock_floor("note", note.id) == top
        journal = _bytes(world.data / "veilid_change_feed.db")
        assert [w for w in (TITLE.encode(), base64.b64encode(BODY), base64.b64encode(UPDATE + b"-1"))
                if w in journal] == [], "the journal's file keeps the words"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DN3 -- a peer's tombstone forgets once the journal holds it
# ---------------------------------------------------------------------------
def test_dn3_a_peers_tombstone_forgets_once_the_journal_holds_it(tmp_path):
    world = _world(tmp_path)
    try:
        note = _plant(world)
        world.engine.published.clear()
        assert world.notes.apply_synced_note(note.id, None, deleted=True, updated_at="2026-10-27T17:00:00+00:00")
        # Landed before the journal's verdict: the flag and the manifest, nothing destroyed -- it may lose.
        assert world.engine.published == [], "a peer's tombstone published something"
        assert world.notes.get_note(note.id).deleted
        assert (world.data / "notes_blobs" / "att-1.blob").exists(), "a blob destroyed before the verdict"
        assert len(world.updates.list_updates(note.id)) == 2, "the tail destroyed before the verdict"
        assert any(r.record_id == f"{note.id}:1" for r in _served(world)), "the journal's updates erased too early"

        # The round journals the tombstone: it won here, and the note is forgotten.
        _journal_tombstone(world, note.id)
        assert world.engine.published == [], "a peer's tombstone published something"
        assert not (world.data / "notes_blobs" / "att-1.blob").exists(), "the blob stays"
        assert world.updates.list_updates(note.id) == [], "the update tail stays"
        assert [r for r in _served(world) if r.kind.value == "note_update"
                and r.record_id.startswith(note.id + ":")] == [], "the journal still serves the updates"
        raw = sqlite3.connect(str(world.data / "notes.db"))
        try:
            assert raw.execute("SELECT title, body_crdt, tags, deleted FROM note WHERE id = ?",
                               (note.id,)).fetchone() == ("", b"", "[]", 1)
        finally:
            raw.close()
        stored = _bytes(world.data / "notes.db") + _bytes(world.data / "note_updates.db")
        assert [w for w in (TITLE.encode(), BODY, TAG.encode(), TRANSCRIPT.encode(), CAPTION.encode(), OCR.encode(),
                            UPDATE, THUMB) if w in stored] == [], "the stores keep the words"
        # The journal's file, its log included, no longer holds the updates; the tombstone is what it serves.
        journal = _bytes(world.data / "veilid_change_feed.db")
        assert [w for w in (base64.b64encode(UPDATE + b"-1"), base64.b64encode(UPDATE + b"-2")) if w in journal] == []
        assert [r.deleted for r in _served(world) if r.kind.value == "note" and r.record_id == note.id] == [True]
        # A tombstone for a note never seen here is recorded, bare.
        assert world.notes.apply_synced_note("ghost", None, deleted=True)
        assert world.notes.get_note("ghost").deleted
        # A batch journalled at once forgets the same.
        second = _plant(world, attachment="att-2")
        assert world.notes.apply_synced_note(second.id, None, deleted=True)
        feed = world.feed.get_change_feed()
        feed.record_many([world.records.new_record(world.records.RecordKind.NOTE, second.id, {}, device="dev-b",
                                                   clock=feed.current_clock("note", second.id) + 1, deleted=True)])
        assert not (world.data / "notes_blobs" / "att-2.blob").exists() and world.updates.list_updates(second.id) == []
        # A tombstone journalled for a note this store holds live -- out of step -- destroys nothing here.
        live = _plant(world, title="LIVE-WORDS-7", attachment="att-live")
        _journal_tombstone(world, live.id)
        assert world.notes.get_note(live.id).title == "LIVE-WORDS-7" and not world.notes.get_note(live.id).deleted
        assert (world.data / "notes_blobs" / "att-live.blob").exists() and len(world.updates.list_updates(live.id)) == 2
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DN4 -- the sweep forgets what earlier deletions left, and never a live note
# ---------------------------------------------------------------------------
def test_dn4_the_sweep_forgets_what_earlier_deletions_left_and_never_a_live_note(tmp_path):
    world = _world(tmp_path)
    try:
        old = _plant(world)
        live = _plant(world, title=KEEP, body=KEEP.encode(), tag="LIVE-TAG", attachment="att-live")
        # The deletion of an older build: the flag alone, everything else kept, its tombstone journalled.
        raw = sqlite3.connect(str(world.data / "notes.db"))
        try:
            raw.execute("UPDATE note SET deleted = 1 WHERE id = ?", (old.id,))
            raw.commit()
        finally:
            raw.close()
        forget, world.notes_module.forget_tombstoned = world.notes_module.forget_tombstoned, lambda note_id: []
        try:
            _journal_tombstone(world, old.id, device="dev-a")
        finally:
            world.notes_module.forget_tombstoned = forget
        assert world.notes.get_note(old.id).title == TITLE, "the setup forgot the note already"
        assert world.notes.sweep_dead_notes() == []

        # The live note holds derived text and updates of its own: the dead note's own words are read here.
        assert [w for w in (TITLE.encode(), BODY, TAG.encode()) if w in _bytes(world.data / "notes.db")] == [], \
            "the dead note's row keeps its words"
        assert world.notes.get_attachment("att-1") is None, "the dead note's attachment stays"
        assert not (world.data / "notes_blobs" / "att-1.blob").exists(), "the dead note's blob stays"
        assert world.updates.list_updates(old.id) == [], "the dead note's tail stays"
        assert [r for r in _served(world) if r.record_id.startswith(old.id + ":")] == [], \
            "the journal still serves the dead note's updates"
        assert [r.deleted for r in _served(world) if r.kind.value == "note" and r.record_id == old.id] == [True], \
            "the journal no longer serves the dead note's tombstone"
        # The live note is whole.
        assert world.notes.get_note(live.id).title == KEEP and not world.notes.get_note(live.id).deleted
        assert (world.data / "notes_blobs" / "att-live.blob").exists()
        assert len(world.updates.list_updates(live.id)) == 2
        assert world.notes.get_attachment("att-live").transcript_text == TRANSCRIPT
        assert any(r.record_id == f"{live.id}:1" for r in _served(world))
        assert any(r.kind.value == "note" and r.record_id == live.id and not r.deleted for r in _served(world)), \
            "the live note's version went"
        # A second sweep finds nothing to do.
        assert world.notes.sweep_dead_notes() == []
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DN5 -- a version that brings a forgotten note back lands with its body
# ---------------------------------------------------------------------------
def test_dn5_a_version_that_brings_a_forgotten_note_back_lands_with_its_body(tmp_path):
    world = _world(tmp_path)
    try:
        note = _plant(world)
        assert world.notes.delete_note(note.id)
        payload = {"title": "back", "body_crdt_b64": base64.b64encode(b"peer body").decode(), "tags": "[]",
                   "pinned": False, "created_at": "2026-10-01T00:00:00+00:00", "attachments": []}
        assert world.notes.apply_synced_note(note.id, payload, updated_at="2026-10-27T18:00:00+00:00")
        back = world.notes.get_note(note.id)
        assert not back.deleted and back.title == "back" and bytes(back.body_crdt) == b"peer body", back
        # A live note's body is the updates' to converge: a version touches its metadata only.
        payload2 = dict(payload, title="renamed", body_crdt_b64=base64.b64encode(b"stale").decode())
        assert world.notes.apply_synced_note(note.id, payload2)
        again = world.notes.get_note(note.id)
        assert again.title == "renamed" and bytes(again.body_crdt) == b"peer body", again
        # An attachment id that is no plain file name is refused at the landing.
        bad = dict(payload, attachments=[{"id": "../a", "kind": "audio"}, {"id": "a/b", "kind": "audio"}])
        assert world.notes.apply_synced_note(note.id, bad)
        assert world.notes.list_attachments(note.id) == []
        # A tombstone that lost dropped the manifest; the version that won names one attachment and not the other:
        # the blob it names stays, the other goes.
        two = _plant(world, attachment="att-kept")
        world.notes.add_attachment(two.id, "image", blob_ref="att-gone", attachment_id="att-gone")
        (world.data / "notes_blobs" / "att-gone.blob").write_bytes(BLOB)
        assert world.notes.apply_synced_note(two.id, None, deleted=True)
        kept = {"id": "att-kept", "kind": "audio", "blob_ref": "att-kept"}
        assert world.notes.apply_synced_note(two.id, dict(payload, attachments=[kept]))
        assert (world.data / "notes_blobs" / "att-kept.blob").exists(), "a blob the winning version names went"
        assert not (world.data / "notes_blobs" / "att-gone.blob").exists(), "a blob no version names stays"
        # A live note's manifest that loses an attachment: its blob goes with its row.
        assert world.notes.apply_synced_note(two.id, dict(payload, attachments=[]))
        assert not (world.data / "notes_blobs" / "att-kept.blob").exists(), "a pruned attachment's blob stays"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DN6 -- an update for a dead note is refused, and never journalled
# ---------------------------------------------------------------------------
def test_dn6_an_update_for_a_dead_note_is_refused_and_never_journalled(tmp_path):
    world = _world(tmp_path)
    try:
        note = _plant(world)
        assert world.notes.delete_note(note.id)
        journal = len(_served(world))
        with pytest.raises(world.updates_module.NoteUpdateRefused):
            world.updates.append_update(note.id, UPDATE + b"-late")
        with pytest.raises(world.updates_module.NoteUpdateRefused):
            world.updates.append_update(note.id, UPDATE + b"-late", seq=9, sync_publish=False)
        assert len(_served(world)) == journal and world.updates.list_updates(note.id) == []
        # Pruning what a live note holds is refused too: its tail is its body's history.
        live = world.notes.add_note("live")
        world.updates.append_update(live.id, b"live-update")
        with pytest.raises(world.updates_module.NoteUpdateRefused):
            world.updates.forget_dead_note(live.id)
        assert len(world.updates.list_updates(live.id)) == 1
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DN7 -- the preview, and every refusal before anything is removed
# ---------------------------------------------------------------------------
def _refused(call, status):
    manager_error = None
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - the class lives in the window
        manager_error = exc
    assert manager_error is not None and getattr(manager_error, "status", None) == status, manager_error
    return manager_error


def test_dn7_the_preview_and_every_refusal_before_anything_is_removed(tmp_path):
    world = _world(tmp_path, wipe=True, routes=True)
    try:
        udm = world.loaded["opti_oignon.user_data_manager"]
        routes = world.loaded["opti_oignon.api.routes_users"]
        for _ in range(3):
            _plant(world, attachment=f"att-{_}")
        preview = udm.wipe_preview("local", ["notes"])
        counted = preview["counts"]["notes"]
        assert {k: counted[k] for k in ("notes", "attachments", "disputed", "deleted_notes")} == \
            {"notes": 3, "attachments": 3, "disputed": 0, "deleted_notes": 0}, counted
        assert len(counted["code"]) == 4 and counted["code"] == world.notes.fingerprint()
        assert preview["phrase"] == f"remove 3 notes {counted['code']}" and preview["notice"] == udm.NOTES_NOTICE
        assert set(_NOTE_LABELS) <= set(preview["removed"]) and not set(_NOTE_LABELS) & set(preview["left"])
        assert preview["on_request"] == {w: list(ls) for w, ls in udm.WIPE_WHEN_NAMED.items()}
        assert udm._phrase({"notes": {"notes": 1, "attachments": 0, "deleted_notes": 0}}) == "remove 1 note"
        plain = udm.wipe_preview("local")
        assert "phrase" not in plain and set(_NOTE_LABELS) <= set(plain["left"]), plain
        deleter = udm.UserDataDeleter()

        _refused(lambda: udm.wipe_preview("local", ["projects"]), 422)
        _refused(lambda: deleter.delete_all("local", remove=["notes"]), 422)
        _refused(lambda: deleter.delete_all("local", remove=["notes"], confirm="remove 2 notes"), 409)
        _refused(lambda: deleter.delete_all("u-2", remove=["notes"], confirm="remove 3 notes"), 409)
        _refused(lambda: deleter.delete_all("local", remove=["notes"], confirm="remove 3 notes"), 409)
        # The same count, other notes: one gone and another come since the preview.
        phrase = preview["phrase"]
        assert world.notes.delete_note(world.notes.live_ids()[0])
        world.notes.add_note("swapped in")
        assert world.notes.counts()["notes"] == 3
        _refused(lambda: deleter.delete_all("local", remove=["notes"], confirm=phrase), 409)
        # An attachment come, the notes the same.
        phrase = udm.wipe_preview("local", ["notes"])["phrase"]
        world.notes.add_attachment(world.notes.live_ids()[0], "image", blob_ref="late", attachment_id="late")
        _refused(lambda: deleter.delete_all("local", remove=["notes"], confirm=phrase), 409)
        # An attachment replaced by another on the same note, the counts the same.
        phrase = udm.wipe_preview("local", ["notes"])["phrase"]
        assert world.notes.delete_attachment("late")
        world.notes.add_attachment(world.notes.live_ids()[0], "image", blob_ref="other", attachment_id="other")
        _refused(lambda: deleter.delete_all("local", remove=["notes"], confirm=phrase), 409)
        # The counts moved since the preview: a note landed.
        world.notes.add_note("arrived")
        _refused(lambda: deleter.delete_all("local", remove=["notes"], confirm=phrase), 409)
        assert len(world.notes.live_ids()) == 4, "a refused wipe removed notes"
        # Through the routes: the same refusals, as HTTP.
        request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"), headers={})
        for call, status in (
            (routes.preview_user_data_wipe("local", request, remove=["projects"], current_user={"sub": "local"}),
             422),
            (routes.delete_user_data("local", request, remove=["notes"], confirm=None, current_user={"sub": "local"}),
             422),
            (routes.delete_user_data("local", request, remove=["notes"], confirm="remove 3 notes",
                                     current_user={"sub": "local"}), 409),
        ):
            with pytest.raises(Exception) as caught:
                asyncio.run(call)
            assert getattr(caught.value, "status_code", None) == status, caught.value
        assert len(world.notes.live_ids()) == 4
        shown = asyncio.run(routes.preview_user_data_wipe("local", request, remove=["notes"],
                                                          current_user={"sub": "local"}))
        assert shown["phrase"] == f"remove 4 notes {world.notes.fingerprint()}"
    finally:
        world.restore()
    # Outside single-user mode, and when the mode cannot be read: refused, nothing read.
    for single, auth in ((False, True), (True, False)):
        world = _world(tmp_path / f"m{int(single)}{int(auth)}", wipe=True, single=single, auth=auth)
        try:
            udm = world.loaded["opti_oignon.user_data_manager"]
            _plant(world)
            _refused(lambda: udm.wipe_preview("local", ["notes"]), 409)
            _refused(lambda: udm.UserDataDeleter().delete_all("local", remove=["notes"], confirm="remove 1 note"),
                     409)
            assert len(world.notes.live_ids()) == 1
        finally:
            world.restore()
    # Notes that cannot be read: 503, nothing removed.
    world = _world(tmp_path / "broken", wipe=True)
    try:
        udm = world.loaded["opti_oignon.user_data_manager"]
        _plant(world)
        world.notes_module._store = SimpleNamespace(counts=lambda: (_ for _ in ()).throw(OSError("unreadable")))
        _refused(lambda: udm.wipe_preview("local", ["notes"]), 503)
        _refused(lambda: udm.UserDataDeleter().delete_all("local", remove=["notes"], confirm="remove 1 note"), 503)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DN8 -- the wipe with the notes named forgets every note
# ---------------------------------------------------------------------------
def test_dn8_the_wipe_with_the_notes_named_forgets_every_note(tmp_path):
    world = _world(tmp_path, wipe=True)
    try:
        udm = world.loaded["opti_oignon.user_data_manager"]
        notes = [_plant(world, attachment=f"att-{i}") for i in range(3)]
        old = _plant(world, attachment="att-old")
        # A note dead here by the flag alone while the journal holds it alive; what an older deletion left, its
        # tombstone journalled; and a blob no row names.
        raw = sqlite3.connect(str(world.data / "notes.db"))
        try:
            raw.execute("UPDATE note SET deleted = 1 WHERE id = ?", (notes[2].id,))
            raw.execute("UPDATE note SET deleted = 1 WHERE id = ?", (old.id,))
            raw.commit()
        finally:
            raw.close()
        forget, world.notes_module.forget_tombstoned = world.notes_module.forget_tombstoned, lambda note_id: []
        try:
            _journal_tombstone(world, old.id, device="dev-a")
        finally:
            world.notes_module.forget_tombstoned = forget
        (world.data / "notes_blobs" / "orphan.blob").write_bytes(BLOB)
        (world.data / "notes_blobs" / "orphan.blob.tmp").write_bytes(BLOB)
        # A blob written while the wipe runs -- an upload in flight, its row not yet written -- is no orphan.
        arriving = world.data / "notes_blobs" / "arriving.blob"
        arriving.write_bytes(b"ARRIVING")
        os.utime(arriving, (time.time() + 3600, time.time() + 3600))
        # Without the notes named, they are untouched and named.
        result = udm.UserDataDeleter().delete_all("local")
        assert result["content_removed"] == {} and set(_NOTE_LABELS) <= set(result["not_covered"])
        assert len(world.notes.live_ids()) == 2 and TITLE.encode() in _bytes(world.data / "notes.db")

        phrase = udm.wipe_preview("local", ["notes"])["phrase"]
        world.engine.published.clear()
        assert phrase.startswith("remove 3 notes "), phrase  # the note hidden here and alive in the journal counted
        result = udm.UserDataDeleter().delete_all("local", remove=["notes"], confirm=phrase)
        assert result["content_removed"] == {"notes": 3}, result["content_removed"]
        assert not set(_NOTE_LABELS) & set(result["not_covered"]), result["not_covered"]
        assert not [line for line in result["not_covered"] if line.startswith(("note", "sync journal"))], \
            result["not_covered"]
        assert world.notes.live_ids() == []
        assert _held(world.data, *_WORDS) == set(), f"the wipe left words: {_held(world.data, *_WORDS)}"
        assert sorted(p.name for p in (world.data / "notes_blobs").iterdir()) == ["arriving.blob"]
        # The note dead here by the flag alone while the journal holds it alive is deleted as the user's own.
        assert sorted(n for kind, n, deleted in world.engine.published if kind == "note" and deleted) == \
            sorted(n.id for n in notes), "each note's tombstone is not published"
        served = [r for r in _served(world) if r.kind.value in ("note", "note_update")]
        assert all(r.kind.value == "note" and r.deleted for r in served if r.record_id in {n.id for n in notes}), \
            served
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DN9 -- a step that fails is named, and stops none of the others
# ---------------------------------------------------------------------------
def test_dn9_a_step_that_fails_is_named_and_stops_none_of_the_others(tmp_path):
    world = _world(tmp_path, wipe=True)
    try:
        udm = world.loaded["opti_oignon.user_data_manager"]
        note = _plant(world)
        stuck = world.data / "notes_blobs" / "att-1.blob"
        stuck.unlink()
        stuck.mkdir()  # a blob that cannot be overwritten
        died, left = world.notes.forget_note(note.id)
        assert died and [line for line in left if line.startswith("note attachment att-1 (not removed:")], left
        # The other steps went: the row, the derived text, the tail, the journal's updates.
        assert {w for w in (TITLE.encode(), BODY, TRANSCRIPT.encode(), CAPTION.encode(), OCR.encode(), THUMB)
                if w in _bytes(world.data / "notes.db")} == set(), "the row kept for the next sweep keeps words"
        assert world.updates.list_updates(note.id) == []
        assert [r for r in _served(world) if r.record_id.startswith(note.id + ":")] == []
        # The row stays, named, for the next sweep.
        assert world.notes.get_attachment("att-1") is not None
        # In the wipe, the same failure is named in what stays.
        second = _plant(world, attachment="att-2")
        stuck2 = world.data / "notes_blobs" / "att-2.blob"
        stuck2.unlink()
        stuck2.mkdir()
        phrase = udm.wipe_preview("local", ["notes"])["phrase"]
        result = udm.UserDataDeleter().delete_all("local", remove=["notes"], confirm=phrase)
        assert result["content_removed"] == {"notes": 1}
        assert [line for line in result["not_covered"] if line.startswith("note attachment att-2 (not removed:")], \
            result["not_covered"]
        assert not world.notes.live_ids() and world.notes.get_note(second.id).deleted
        # A journal that cannot be written: each note's own failure is named in the wipe's answer, besides the
        # sweep's.
        third = world.notes.add_note("third")

        class _Broken:
            def __getattr__(self, name):
                def refuse(*_args, **_kwargs):
                    raise OSError("the journal cannot be written")
                return refuse

        world.feed.set_change_feed(_Broken())
        phrase = udm.wipe_preview("local", ["notes"])["phrase"]
        result = udm.UserDataDeleter().delete_all("local", remove=["notes"], confirm=phrase)
        assert result["content_removed"] == {"notes": 1} and world.notes.get_note(third.id).deleted
        assert "sync journal (not forgotten: OSError)" in result["not_covered"], result["not_covered"]
        assert "sync journal (not read: OSError)" in result["not_covered"], result["not_covered"]
        world.feed.set_change_feed(world.feed.ChangeFeed(root=world.data))
        # A blob a peer's tombstone took off the manifest that cannot be removed: named, its id kept, retried.
        awaited = _plant(world, attachment="att-awaited")
        stuck3 = world.data / "notes_blobs" / "att-awaited.blob"
        stuck3.unlink()
        stuck3.mkdir()
        assert world.notes.apply_synced_note(awaited.id, None, deleted=True)
        _journal_tombstone(world, awaited.id)
        raw = sqlite3.connect(str(world.data / "notes.db"))
        try:
            assert raw.execute("SELECT attachment_id FROM awaited_blob WHERE note_id = ?",
                               (awaited.id,)).fetchall() == [("att-awaited",)], "the id let go of a blob still there"
        finally:
            raw.close()
        stuck3.rmdir()
        stuck3.write_bytes(BLOB)
        # Once the blob can go, the sweep finishes.
        stuck.rmdir()
        stuck2.rmdir()
        assert world.notes.sweep_dead_notes() == []
        assert world.notes.get_attachment("att-1") is None and world.notes.get_attachment("att-2") is None
        assert not stuck3.exists(), "the awaited blob never retried"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DN10 -- the export carries what the wipe removes
# ---------------------------------------------------------------------------
class _Conversations:
    """The conversation store as the exporter reads it: pages of conversations, each read with its messages; one
    that cannot be read."""

    def __init__(self, count, broken=None, gone=None, short_page=None):
        self.ids = [f"c{i:03d}" for i in range(count)]
        self.broken, self.gone, self.short_page = broken, gone, short_page

    def list_conversations(self, limit=50, offset=0):
        if offset == self.short_page:
            return []  # as the store answers a page it cannot read
        return [SimpleNamespace(id=i) for i in self.ids[offset:offset + limit]]

    def get_stats(self):
        return {"total_conversations": len(self.ids)}

    def get_conversation(self, conv_id):
        if conv_id == self.broken:
            raise OSError("unreadable")
        if conv_id == self.gone:
            return None  # as the store answers one it cannot read
        return SimpleNamespace(to_dict=lambda: {"id": conv_id, "messages": [{"content": f"words of {conv_id}"}]})


def test_dn10_the_export_carries_what_the_wipe_removes(tmp_path):
    conversations = _Conversations(205, broken="c007", gone="c009")
    world = _world(tmp_path, wipe=True, seeded={"opti_oignon.conversation": _module(
        "opti_oignon.conversation", conversation_manager=conversations)})
    try:
        udm = world.loaded["opti_oignon.user_data_manager"]
        note = _plant(world)
        data = udm.UserDataExporter().export("local")
        meta = data["export_metadata"]
        assert meta["format_version"] == "1.3"
        (exported,) = data["notes"]
        assert exported["title"] == TITLE and exported["body_text"] == BODY.decode()
        assert base64.b64decode(exported["body_b64"]) == BODY and exported["tags"] == f'["{TAG}"]'
        assert [base64.b64decode(u["update_b64"]) for u in exported["updates"]] == [UPDATE + b"-1", UPDATE + b"-2"]
        (attachment,) = exported["attachments"]
        assert (attachment["transcript_text"], attachment["caption_text"], attachment["ocr_text"]) == \
            (TRANSCRIPT, CAPTION, OCR)
        assert sorted(c["id"] for c in data["conversations"]) == [i for i in conversations.ids
                                                                   if i not in ("c007", "c009")]
        assert meta["failed"] == ["conversation c007 (not exported: OSError)",
                                  "conversation c009 (not exported: not read)"], meta["failed"]
        assert not {"conversations (store not user-scoped)", *_NOTE_LABELS} & set(meta["not_covered"])
        assert udm.EXPORT_NOT_COVERED[1] in meta["not_covered"], "the attachments' bytes are not named as left out"
        assert world.notes.get_note(note.id).title == TITLE, "the export changed a note"
        # A page the store could not read comes back empty: the conversations it counts and the listing missed
        # are named.
        world.loaded["opti_oignon.user_data_manager"]._get_conversation_manager = \
            lambda: _Conversations(450, short_page=200)
        meta = udm.UserDataExporter().export("local")["export_metadata"]
        assert "conversations (250 of 450 not listed, not exported)" in meta["failed"], meta["failed"]
        # A store that cannot read itself answers an empty count and an empty listing: never taken for none.
        unreadable = _Conversations(0)
        unreadable.get_stats = lambda: {}
        world.loaded["opti_oignon.user_data_manager"]._get_conversation_manager = lambda: unreadable
        meta = udm.UserDataExporter().export("local")["export_metadata"]
        assert "conversations (not counted; the listing may be short)" in meta["failed"], meta["failed"]
    finally:
        world.restore()
    world = _world(tmp_path / "multi", wipe=True, single=False, seeded={"opti_oignon.conversation": _module(
        "opti_oignon.conversation", conversation_manager=conversations)})
    try:
        udm = world.loaded["opti_oignon.user_data_manager"]
        _plant(world)
        data = udm.UserDataExporter().export("local")
        assert data["notes"] == [] and data["conversations"] == []
        assert {"conversations (store not user-scoped)", *_NOTE_LABELS} <= set(data["export_metadata"]["not_covered"])
        assert udm.EXPORT_NOT_COVERED[1] not in data["export_metadata"]["not_covered"]
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DN11 -- the request's words are closed and derived
# ---------------------------------------------------------------------------
def _inventory():
    """The inventory, run from its source: data only, nothing of the package imported."""
    path = source("store_inventory.py")
    module = types.ModuleType("_dn_store_inventory")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    try:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    return module


def _literal(name):
    for node in ast.parse(source("user_data_manager.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"no {name} in the wipe")


def test_dn11_the_requests_words_are_closed_and_derived():
    inventory = _inventory()
    asked = [store for store in inventory.STORES if store.request]
    assert asked, "no store is removed on request"
    for store in inventory.STORES:
        if store.request:
            assert store.request in inventory.REQUESTS, f"{store.label}: the word {store.request!r}"
            assert store.wipe == inventory.NOT_COVERED, f"{store.label}: named, yet {store.wipe!r}"
            assert store.holds in (inventory.WORDS, inventory.DRAWN), store.label
    derived = {}
    for store in asked:
        derived.setdefault(store.request, []).append(store.label)
    assert _literal("WIPE_WHEN_NAMED") == {word: tuple(labels) for word, labels in derived.items()}
    assert _literal("WIPE_REMOVED") == tuple(s.label for s in inventory.STORES if s.wipe == inventory.REMOVED)
    assert sorted(derived) == sorted(inventory.REQUESTS), "a word no store answers to"


# ---------------------------------------------------------------------------
# DN12 -- the panel offers the export first and asks the phrase
# ---------------------------------------------------------------------------
_RULES = "frontend/src/lib/settings/yourData.ts"
_PANEL = "frontend/src/lib/components/settings/YourDataPanel.svelte"
_DRIVER = r"""
const m = await import(process.env.OO_YOUR_DATA);
const clause = process.argv[2];
const ok = (cond, what) => { if (!cond) { console.log('FAIL ' + what); process.exit(1); } };
const gate = { exportOffered: true, typedWipe: 'delete my data', named: [], phrase: undefined, typedPhrase: '',
               busy: false };
if (clause === 'rules') {
  ok(m.wipeQuery([]) === '', 'no query without words');
  ok(m.wipeQuery([' notes', 'notes'], ' remove 3 notes ') === '?remove=notes&confirm=remove%203%20notes', 'query');
  ok(m.wipeQuery([], 'remove 3 notes') === '', 'no phrase without words');
  ok(m.typedMatches(' remove 3 notes ', 'remove 3 notes') && !m.typedMatches('remove 2 notes', 'remove 3 notes'),
     'typed');
  ok(!m.typedMatches('', undefined) && !m.typedMatches('', ''), 'nothing matches no phrase');
  ok(m.canWipe(gate), 'the plain wipe, its words typed');
  ok(!m.canWipe({ ...gate, exportOffered: false }), 'the export comes first');
  ok(!m.canWipe({ ...gate, typedWipe: 'delete' }), 'the words typed');
  ok(!m.canWipe({ ...gate, busy: true }), 'not while busy');
  ok(!m.canWipe({ ...gate, named: ['notes'], phrase: 'remove 3 notes', typedPhrase: '' }), 'the phrase asked');
  ok(!m.canWipe({ ...gate, named: ['notes'], phrase: undefined, typedPhrase: 'remove 3 notes' }),
     'no phrase, no wipe of the notes');
  ok(m.canWipe({ ...gate, named: ['notes'], phrase: 'remove 3 notes', typedPhrase: 'remove 3 notes' }), 'named');
  const lines = m.answerLines(['a store', 'notes (2 arrived while the wipe ran, left)'], ['a store'],
                              { conversations: 2, notes: 0, facts: 5 });
  ok(JSON.stringify(lines.failed) === JSON.stringify(['notes (2 arrived while the wipe ran, left)']), 'failed');
  ok(JSON.stringify(lines.left) === JSON.stringify(['a store']), 'left');
  ok(JSON.stringify(lines.removed) === JSON.stringify(['conversations: 2', 'facts: 5']), 'removed');
  console.log('PASS rules');
}
"""


def test_dn12_the_panel_offers_the_export_first_and_asks_the_phrase():
    from _frontend import read, run_ts

    run_ts({"OO_YOUR_DATA": _RULES}, _DRIVER, "rules")
    panel = read(_PANEL)
    # The export is offered before the confirmation, the gate decides the button, the phrase goes to the wipe.
    assert panel.index("downloadFirst") < panel.index("3. Confirm"), "the export is not offered first"
    assert "canWipe({" in panel and 'disabled={!allowed}' in panel, "the button is not the gate's"
    assert "wipeUserData(userId, named, named.length > 0 ? preview.phrase : undefined)" in panel
    assert "answerLines(result.not_covered, preview.left" in panel, "the answer is not shown failures first"
    assert "if (asked === previewAsked) preview = answer;" in panel, "a stale preview shown over the last asked"
    client = read("frontend/src/lib/api/userData.ts")
    assert "/data/wipe-preview${wipeQuery(remove)}" in client and "/data${wipeQuery(remove, phrase)}" in client
    # Placed in the settings, and loaded by the hub.
    assert "panel: 'YourDataPanel'" in read("frontend/src/lib/settings/catalog.ts")
    assert "YourDataPanel: () => import('$lib/components/settings/YourDataPanel.svelte')" in read(
        "frontend/src/lib/components/settings/SettingsHub.svelte")


# ---------------------------------------------------------------------------
# DN13 -- a peer's tombstone that loses destroys nothing
# ---------------------------------------------------------------------------
def test_dn13_a_peers_tombstone_that_loses_destroys_nothing(tmp_path):
    world = _world(tmp_path, wipe=True)
    try:
        udm = world.loaded["opti_oignon.user_data_manager"]
        note = _plant(world)
        # Edited here after the peer deleted it: the peer's tombstone lands in the store first, and the journal,
        # which holds the later version, never takes it.
        assert world.notes.apply_synced_note(note.id, None, deleted=True)
        other = world.notes.add_note("OTHER-WORDS-7", body_crdt=b"OTHER-BODY-7")
        assert world.notes.delete_note(other.id), "the next deletion, whose sweep runs"
        for _ in range(2):
            assert world.notes.sweep_dead_notes() == []
        assert TITLE.encode() in _bytes(world.data / "notes.db"), "the note's words destroyed"
        assert (world.data / "notes_blobs" / "att-1.blob").exists(), "the note's blob destroyed"
        assert len(world.updates.list_updates(note.id)) == 2, "the note's tail destroyed"
        alive = [r for r in _served(world) if r.kind.value == "note" and r.record_id == note.id]
        assert [r.deleted for r in alive] == [False], "the journal's winning version erased"
        assert any(r.record_id == f"{note.id}:1" for r in _served(world)), "the journal's updates erased"
        # A journal that cannot be read: every dead note left as it is, said.
        held = world.feed.get_change_feed()

        class _Unreadable:
            def __getattr__(self, name):
                def refuse(*_args, **_kwargs):
                    raise OSError("the journal cannot be read")
                return refuse

        world.feed.set_change_feed(_Unreadable())
        assert world.notes.sweep_dead_notes() == ["sync journal (not read: OSError)"]
        world.feed.set_change_feed(held)
        assert TITLE.encode() in _bytes(world.data / "notes.db") and len(world.updates.list_updates(note.id)) == 2
        # A restart between the landing and the verdict loses nothing: the dropped attachments are kept on disk.
        restarted = world.notes_module.NotesStore(root=world.data)
        world.notes_module._store = restarted
        world.notes = restarted
        # A second disputed note, deleted here directly while sync is off (no tombstone journalled to forget after
        # it): its dropped blob goes with it.
        second = _plant(world, title="SECOND-WORDS-7", attachment="att-second")
        assert world.notes.apply_synced_note(second.id, None, deleted=True)
        world.guard.veilid_available = lambda: False
        left = world.notes.forget_note(second.id, sweep=False)[1]
        world.guard.veilid_available = lambda: True
        assert [line for line in left if line.startswith("paired devices (not told")] == left and left, left
        assert not (world.data / "notes_blobs" / "att-second.blob").exists(), "a disputed note's blob kept"
        # A peer's tombstone that wins while this process has no notes store open: the sweep finishes it.
        third = _plant(world, title="THIRD-WORDS-7", attachment="att-third")
        assert world.notes.apply_synced_note(third.id, None, deleted=True)
        world.notes_module._store = None
        _journal_tombstone(world, third.id)
        world.notes_module._store = restarted
        assert (world.data / "notes_blobs" / "att-third.blob").exists()
        assert world.notes.sweep_dead_notes() == []
        assert not (world.data / "notes_blobs" / "att-third.blob").exists(), "the sweep kept a dropped blob"
        # That sweep, sync back on, told the peers what the deletion made with sync off owed.
        assert world.feed.get_change_feed().note_states([second.id]) == {second.id: "tombstone"}
        # The journal unreadable: the preview cannot count the hidden note, and says so; a wipe names it as left.
        world.feed.set_change_feed(_Unreadable())
        unread = udm.wipe_preview("local", ["notes"])
        assert unread["counts"]["notes"]["disputed"] is None and udm.NOTES_UNREAD_NOTICE in unread["notice"], unread
        result = udm.UserDataDeleter().delete_all("local", remove=["notes"], confirm=unread["phrase"])
        assert [line for line in result["not_covered"] if line.startswith("notes and their updates (those hidden")], \
            result["not_covered"]
        assert TITLE.encode() in _bytes(world.data / "notes.db"), "the hidden note's words, said left, are gone"
        world.feed.set_change_feed(held)
        # The wipe counts the disputed note and deletes it as the user's own deletion: a tombstone that out-clocks
        # the version alive, the blob its landing took off the manifest removed.
        world.engine.published.clear()
        preview = udm.wipe_preview("local", ["notes"])
        assert preview["counts"]["notes"]["disputed"] == 1 and preview["phrase"].startswith("remove 1 note "), preview

        phrase = preview["phrase"]
        result = udm.UserDataDeleter().delete_all("local", remove=["notes"], confirm=phrase)
        assert not (world.data / "notes_blobs" / "att-1.blob").exists(), "the disputed note's blob kept"
        assert result["content_removed"] == {"notes": 1}, result["content_removed"]
        assert ("note", note.id, True) in world.engine.published, "the disputed note's deletion is not published"
        tomb = [r for r in _served(world) if r.kind.value == "note" and r.record_id == note.id]
        assert [r.deleted for r in tomb] == [True] and tomb[0].clock > alive[0].clock, tomb
        assert _held(world.data, TITLE.encode(), BODY, UPDATE, BLOB) == set()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DN14 -- an update racing the deletion is never kept
# ---------------------------------------------------------------------------
def _race(world, delete):
    """An update appended while ``delete`` runs: the append checks the note live and pauses; the deletion is let go
    once it asks for the append lock (or after a second, if it never asks), then the append."""
    note = world.notes.add_note("racing")
    checked, go, asking = threading.Event(), threading.Event(), threading.Event()
    sibling = world.updates._sibling_lookup
    holding = world.notes_module._holding_appends

    def slow(note_id, user_id):
        # The note found live; the deletion asked before the update is written.
        live = sibling(note_id, user_id)
        checked.set()
        go.wait(2.0)
        return live

    def signalled():
        asking.set()
        return holding()

    racer = world.updates_module.NoteUpdatesStore(root=world.data, parent_lookup=slow)
    outcome = {}

    def append():
        try:
            outcome["appended"] = racer.append_update(note.id, b"RACE-WORDS-7")
        except Exception as exc:  # noqa: BLE001 - a refusal is an outcome too
            outcome["refused"] = exc

    world.notes_module._holding_appends = signalled
    try:
        appending = threading.Thread(target=append)
        appending.start()
        assert checked.wait(2.0)
        deleting = threading.Thread(target=lambda: outcome.setdefault("deleted", delete(note.id)))
        deleting.start()
        asking.wait(1.0)
        time.sleep(0.05)
        go.set()
        appending.join(5.0)
        deleting.join(5.0)
    finally:
        go.set()
        world.notes_module._holding_appends = holding
    return note, outcome


def test_dn14_an_update_racing_the_deletion_is_never_kept(tmp_path):
    world = _world(tmp_path)
    try:
        # This device's deletion.
        note, outcome = _race(world, world.notes.delete_note)
        assert outcome.get("deleted") is True, outcome
        assert world.updates.list_updates(note.id) == [], "the racing update stays"
        assert [r for r in _served(world) if r.record_id.startswith(note.id + ":")] == [], \
            "the journal serves the racing update"
        # This device's deletion with sync off: no tombstone journalled to forget after it.
        world.guard.veilid_available = lambda: False
        note, outcome = _race(world, world.notes.delete_note)
        world.guard.veilid_available = lambda: True
        assert outcome.get("deleted") is True, outcome
        assert world.updates.list_updates(note.id) == [], "the racing update stays when nothing is journalled"
        # A peer's tombstone, journalled once it lands.

        def peers(note_id):
            world.notes.apply_synced_note(note_id, None, deleted=True)
            return _journal_tombstone(world, note_id)

        note, outcome = _race(world, peers)
        assert "deleted" in outcome, outcome
        assert world.updates.list_updates(note.id) == [], "the update racing a peer's tombstone stays"
        assert [r for r in _served(world) if r.record_id.startswith(note.id + ":")] == [], \
            "the journal serves the update racing a peer's tombstone"
        assert _held(world.data, b"RACE-WORDS-7", base64.b64encode(b"RACE-WORDS-7")) == set()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# DN15 -- a deletion the journal could not take is said, and told at the next
# ---------------------------------------------------------------------------
def test_dn15_a_deletion_the_journal_could_not_take_is_said_and_told_at_the_next(tmp_path):
    world = _world(tmp_path)
    try:
        note = _plant(world)
        working = world.engine.publish_note

        def failing(*_args, **_kwargs):
            raise OSError("the journal is locked")

        # The publication fails: the peers not told, said; the tombstone owed.
        world.engine.publish_note = failing
        died, left = world.notes.forget_note(note.id, sweep=False)
        assert died and [line for line in left
                         if line.startswith(f"paired devices (not told of the note {note.id}'s deletion")], left
        assert world.feed.get_change_feed().note_states([note.id]) == {note.id: "owed"}
        # Deleted again, the publication back: the peers are told -- by the deletion itself, its sweep aside.
        world.engine.publish_note = working
        world.engine.published.clear()
        assert world.notes.forget_note(note.id, sweep=False) == (False, [])
        assert ("note", note.id, True) in world.engine.published
        assert [r.deleted for r in _served(world) if r.kind.value == "note" and r.record_id == note.id] == [True]
        # Another note's deletion, through its sweep, tells what an earlier one owed.
        other = _plant(world, attachment="att-other")
        world.engine.publish_note = failing
        assert world.notes.forget_note(other.id, sweep=False)[0]
        world.engine.publish_note = working
        world.engine.published.clear()
        assert world.notes.delete_note(world.notes.add_note("third").id)
        assert ("note", other.id, True) in world.engine.published, "the sweep did not tell what was owed"
    finally:
        world.restore()
    # Sync never used here: nothing owed, nothing said.
    world = _world(tmp_path / "unsynced", engine=False)
    try:
        note = _plant(world)
        assert world.notes.forget_note(note.id) == (True, [])
        assert not (world.data / "veilid_change_feed.db").exists()
    finally:
        world.restore()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
