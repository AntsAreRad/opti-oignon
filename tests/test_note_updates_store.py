#!/usr/bin/env python3
"""Tests for the N.8 note-update store (notes.note_updates_store).

NoteUpdatesStore is the append-only log of opaque Yjs update blobs that backs
collaborative note sync. Its append seam is security-sensitive: an update that
cannot be attributed, gated against a live parent, or persisted is REFUSED
(NOTES_CRDT_SPEC.md section 5), never silently appended. The parent-liveness
gate is injectable, so these tests drive it with a small mutable Gate -- which
also lets one store append while the note is live and then prune once it dies.

The store keeps SQLite at rest (via safe_connect, degrading to plaintext here);
every test uses a fresh temp root.
"""

import pytest

from opti_oignon.notes.note_updates_store import (
    NoteUpdateRefused,
    NoteUpdatesStore,
)


class Gate:
    """A mutable parent-liveness probe: live by default, can die or fail."""

    def __init__(self, *, live=True, raise_it=False):
        self.live = live
        self.raise_it = raise_it

    def __call__(self, note_id, user_id):
        if self.raise_it:
            raise RuntimeError("liveness probe down")
        return self.live


def _store(tmp_path, gate=None):
    return NoteUpdatesStore(tmp_path, parent_lookup=gate or Gate())


def _append(store, note_id="n1", blob=b"u", **kw):
    kw.setdefault("sync_publish", False)   # don't touch the Veilid journal glue
    return store.append_update(note_id, blob, **kw)


# ===========================================================================
# append_update -- success
# ===========================================================================

def test_append_live_parent_succeeds(tmp_path):
    store = _store(tmp_path)
    rec = _append(store, blob=b"hello")
    assert rec.seq == 1
    assert rec.note_id == "n1"
    assert rec.update_blob == b"hello"
    assert store.count_updates("n1") == 1


def test_seq_auto_increments(tmp_path):
    store = _store(tmp_path)
    assert _append(store).seq == 1
    assert _append(store).seq == 2
    assert store.latest_seq("n1") == 2


def test_explicit_seq_honored(tmp_path):
    # The remote-apply path preserves the author's order by passing seq.
    store = _store(tmp_path)
    rec = _append(store, blob=b"x", seq=5)
    assert rec.seq == 5
    assert store.latest_seq("n1") == 5


def test_append_preserves_opaque_blob(tmp_path):
    store = _store(tmp_path)
    blob = b"\x00\xff\x01\x02not-text"
    _append(store, blob=blob)
    assert store.list_updates("n1")[0].update_blob == blob   # byte-exact


def test_author_device_recorded(tmp_path):
    store = _store(tmp_path)
    rec = _append(store, author_device="device-A")
    assert rec.author_device == "device-A"


# ===========================================================================
# append_update -- fail-secure refusals (section 5)
# ===========================================================================

def test_refuse_missing_blob(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(NoteUpdateRefused):
        store.append_update("n1", None, sync_publish=False)
    assert store.count_updates("n1") == 0       # nothing persisted


def test_refuse_dead_parent(tmp_path):
    store = _store(tmp_path, Gate(live=False))
    with pytest.raises(NoteUpdateRefused):
        _append(store)
    assert store.count_updates("n1") == 0


def test_refuse_indeterminable_parent(tmp_path):
    store = _store(tmp_path, Gate(raise_it=True))
    with pytest.raises(NoteUpdateRefused):
        _append(store)
    assert store.count_updates("n1") == 0


def test_refuse_duplicate_seq_never_replaces(tmp_path):
    store = _store(tmp_path)
    _append(store, blob=b"first", seq=1)
    with pytest.raises(NoteUpdateRefused):
        _append(store, blob=b"second", seq=1)   # same (user, note, seq)
    # the original row is intact -- a duplicate never overwrites
    rows = store.list_updates("n1")
    assert len(rows) == 1
    assert rows[0].update_blob == b"first"


# ===========================================================================
# reads
# ===========================================================================

def test_list_updates_after_seq_and_order(tmp_path):
    store = _store(tmp_path)
    for _ in range(3):
        _append(store)
    tail = store.list_updates("n1", after_seq=1)
    assert [r.seq for r in tail] == [2, 3]       # seq > 1, ascending


def test_list_updates_limit(tmp_path):
    store = _store(tmp_path)
    for _ in range(3):
        _append(store)
    assert len(store.list_updates("n1", limit=2)) == 2


def test_count_and_latest_seq_empty_note(tmp_path):
    store = _store(tmp_path)
    assert store.count_updates("absent") == 0
    assert store.latest_seq("absent") == 0


# ===========================================================================
# checkpoint watermark (section 4)
# ===========================================================================

def test_watermark_default_zero(tmp_path):
    assert _store(tmp_path).get_checkpoint_watermark("n1") == 0


def test_watermark_set_and_get(tmp_path):
    store = _store(tmp_path)
    assert store.set_checkpoint_watermark("n1", 5) is True
    assert store.get_checkpoint_watermark("n1") == 5


def test_watermark_is_monotonic(tmp_path):
    store = _store(tmp_path)
    store.set_checkpoint_watermark("n1", 5)
    assert store.set_checkpoint_watermark("n1", 3) is False   # regression no-op
    assert store.get_checkpoint_watermark("n1") == 5
    assert store.set_checkpoint_watermark("n1", 7) is True
    assert store.get_checkpoint_watermark("n1") == 7


# ===========================================================================
# pruning (section 4: local, lazy, never over-prune)
# ===========================================================================

def test_prune_below_watermark_noop_without_watermark(tmp_path):
    store = _store(tmp_path)
    for _ in range(3):
        _append(store)
    assert store.prune_below_watermark("n1") == 0   # nothing provably folded
    assert store.count_updates("n1") == 3


def test_prune_below_watermark_deletes_folded_tail(tmp_path):
    store = _store(tmp_path)
    for _ in range(3):
        _append(store)                              # seq 1, 2, 3
    store.set_checkpoint_watermark("n1", 2)
    assert store.prune_below_watermark("n1") == 2    # seq 1 and 2 removed
    remaining = store.list_updates("n1")
    assert [r.seq for r in remaining] == [3]


def test_prune_for_tombstone_prunes_dead_parent(tmp_path):
    gate = Gate(live=True)
    store = _store(tmp_path, gate)
    _append(store)
    _append(store)
    store.set_checkpoint_watermark("n1", 1)
    gate.live = False                               # the note is now tombstoned
    assert store.prune_for_tombstone("n1") == 2     # full tail gone
    assert store.count_updates("n1") == 0
    assert store.get_checkpoint_watermark("n1") == 0  # checkpoint row dropped too


def test_prune_for_tombstone_refuses_live_parent(tmp_path):
    store = _store(tmp_path, Gate(live=True))
    _append(store)
    _append(store)
    assert store.prune_for_tombstone("n1") == 0     # live -> preserved
    assert store.count_updates("n1") == 2


def test_prune_for_tombstone_refuses_indeterminable(tmp_path):
    gate = Gate(live=True)
    store = _store(tmp_path, gate)
    _append(store)
    gate.raise_it = True                            # probe down -> doubt
    assert store.prune_for_tombstone("n1") == 0     # preserved on doubt
    assert store.count_updates("n1") == 1
