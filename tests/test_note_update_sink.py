#!/usr/bin/env python3
"""Tests for the N.8 note_update apply sink (veilid.sync_engine._update_sink_for).

This is the receiving half of collaborative note sync: it lands a received
``note_update`` record into the append-only store. It is fail-secure at the
landing seam (NOTES_CRDT_SPEC.md section 5) -- a record that cannot be
attributed refuses (returns False, nothing appended), and any store refusal
propagates as False too.

The sink factory takes any object exposing ``append_update``, so most cases use
a recording fake store (to assert the exact append arguments), plus one
integration round through a real NoteUpdatesStore at a temp root. The record is
duck-typed: the sink only reads ``.payload``, ``.record_id``, and ``.device``.
"""

import base64
import types

from opti_oignon.notes.note_updates_store import NoteUpdatesStore
from opti_oignon.veilid.sync_engine import _update_sink_for


class FakeStore:
    """Records append_update calls; can be set to refuse (raise)."""

    def __init__(self, *, raise_on_append=False):
        self.calls = []
        self._raise = raise_on_append

    def append_update(self, note_id, blob, *, author_device=None, seq=None, sync_publish=True):
        self.calls.append({
            "note_id": note_id, "blob": blob, "author_device": author_device,
            "seq": seq, "sync_publish": sync_publish,
        })
        if self._raise:
            raise RuntimeError("store refused")


def _record(*, note_id="n1", seq=3, blob=b"hi", device="origin-dev",
            record_id=None, blob_b64=None):
    payload = {"note_id": note_id, "seq": seq}
    payload["update_blob_b64"] = (
        blob_b64 if blob_b64 is not None
        else base64.b64encode(blob).decode("ascii")
    )
    r = types.SimpleNamespace()
    r.payload = payload
    r.record_id = record_id if record_id is not None else f"{note_id}:{seq}"
    r.device = device
    return r


# ===========================================================================
# Success: a valid record lands with the author's order, device, no re-publish
# ===========================================================================

def test_sink_lands_valid_record():
    store = FakeStore()
    assert _update_sink_for(store)(_record(note_id="n1", seq=3, blob=b"hello")) is True
    call = store.calls[-1]
    assert call["note_id"] == "n1"
    assert call["blob"] == b"hello"          # decoded from the wire
    assert call["author_device"] == "origin-dev"
    assert call["seq"] == 3
    assert call["sync_publish"] is False     # never re-sign the author's update


def test_sink_preserves_author_seq():
    store = FakeStore()
    _update_sink_for(store)(_record(seq=7))
    assert store.calls[-1]["seq"] == 7       # the wire seq, never a local re-mint


def test_sink_uses_record_device_as_author():
    store = FakeStore()
    _update_sink_for(store)(_record(device="phone-42"))
    assert store.calls[-1]["author_device"] == "phone-42"


# ===========================================================================
# Fail-secure attribution refusals (return False, nothing appended)
# ===========================================================================

def _refused(record):
    store = FakeStore()
    result = _update_sink_for(store)(record)
    return result is False and store.calls == []


def test_refuse_non_string_note_id():
    assert _refused(_record(note_id=None))


def test_refuse_empty_note_id():
    assert _refused(_record(note_id=""))


def test_refuse_bool_seq():
    # bool is an int subclass; the sink must reject it explicitly.
    assert _refused(_record(seq=True, record_id="n1:True"))


def test_refuse_non_positive_seq():
    assert _refused(_record(seq=0))


def test_refuse_non_int_seq():
    assert _refused(_record(seq="3"))


def test_refuse_mismatched_record_id():
    # A re-coordinated payload (record identity != note_id:seq) cannot be
    # attributed and refuses.
    assert _refused(_record(note_id="n1", seq=3, record_id="n1:999"))


def test_refuse_non_string_blob():
    assert _refused(_record(blob_b64=12345))


def test_refuse_invalid_base64():
    assert _refused(_record(blob_b64="!!!not-base64!!!"))


# ===========================================================================
# Store-refusal propagation
# ===========================================================================

def test_refuse_when_store_rejects():
    # A well-formed record whose store append refuses (dead parent, duplicate,
    # ...) lands as False -- the sink never swallows a refusal into success.
    store = FakeStore(raise_on_append=True)
    assert _update_sink_for(store)(_record()) is False


# ===========================================================================
# Integration: land into a real NoteUpdatesStore
# ===========================================================================

def test_sink_lands_into_real_store(tmp_path):
    store = NoteUpdatesStore(tmp_path, parent_lookup=lambda nid, uid: True)
    sink = _update_sink_for(store)
    assert sink(_record(note_id="note-A", seq=4, blob=b"\x01\x02opaque")) is True
    rows = store.list_updates("note-A")
    assert len(rows) == 1
    assert rows[0].seq == 4                       # author's order preserved
    assert rows[0].update_blob == b"\x01\x02opaque"
    assert rows[0].author_device == "origin-dev"


def test_real_store_rejects_duplicate_landing(tmp_path):
    store = NoteUpdatesStore(tmp_path, parent_lookup=lambda nid, uid: True)
    sink = _update_sink_for(store)
    assert sink(_record(note_id="note-A", seq=1)) is True
    assert sink(_record(note_id="note-A", seq=1)) is False   # duplicate refused
    assert store.count_updates("note-A") == 1
