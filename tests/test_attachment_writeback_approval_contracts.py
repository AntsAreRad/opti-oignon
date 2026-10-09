#!/usr/bin/env python3
"""Approval contracts for the two model write-backs into a note's attachment.

A caption, an OCR reading and a transcript are written by a model, inside a
disposable sandbox, about an attachment the user put in a note. The note is
read back by the agent, so the write-back is a write into a store a model
reads: it waits for the user. ``caption_attachment`` and
``transcribe_attachment`` take the user's approval as an argument. The write
census still owes both write-backs: the approved request runs the model
again, so the text written is not the text the user approved. These
contracts hold the approval itself.

  * WB1 -- without approval a caption is returned and nothing is written back;
    with it, the caption and the OCR reading are written back once, exactly,
    for the user who owns the note.
  * WB2 -- without approval a transcript is returned and nothing is written
    back; with it, the transcript is written back once, exactly.

Each module is loaded alone in an isolation window. The sandbox, the blob
store and the note store are stood in for: no process runs, nothing is
decrypted, no database is opened. Local-only.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


class _Store:
    """A note store that holds one attachment of ``kind`` and records every write-back."""

    def __init__(self, kind):
        self.kind = kind
        self.updates = []

    def get_attachment(self, attachment_id, user_id=None):
        return types.SimpleNamespace(id=attachment_id, kind=self.kind)

    def update_attachment(self, attachment_id, user_id=None, **legs):
        self.updates.append((attachment_id, user_id, legs))


class _Blobs:
    def open(self, attachment_id):
        return b"\x00attachment bytes"


class _Sandbox:
    """A sandbox seam that runs bubblewrap and hands out one workspace."""

    bwrap_available = True
    bwrap_in_use = True

    def __init__(self, root):
        self.root = root
        self.destroyed = []

    def create_sandbox(self, session_id=None, label="", owner_user_id=""):
        return types.SimpleNamespace(session_id="wb-session")

    def get_active_workspace_path(self, session_id):
        return str(self.root)

    def destroy_sandbox(self, session_id):
        self.destroyed.append(session_id)


def _load(module, filename):
    loaded, restore = isolate(
        targets={f"opti_oignon.notes.{module}": source("notes", filename)},
        packages=("opti_oignon.notes",),
    )
    return loaded[f"opti_oignon.notes.{module}"], restore


def test_wb1_a_caption_is_written_back_only_when_the_user_approves(tmp_path):
    caption, restore = _load("caption", "caption.py")
    try:
        held, held_box = _Store("image"), _Sandbox(tmp_path)
        unapproved = caption.caption_attachment(
            "wb1", user_id="owner", store=held, blobs=_Blobs(), sandbox=held_box,
            captioner=lambda sandbox, session, name: ("a cat on a mat", "MAT"),
        )
        kept, kept_box = _Store("image"), _Sandbox(tmp_path)
        approved = caption.caption_attachment(
            "wb1", user_id="owner", store=kept, blobs=_Blobs(), sandbox=kept_box,
            captioner=lambda sandbox, session, name: ("a cat on a mat", "MAT"), approve=True,
        )
    finally:
        restore()
    # Without approval: the reading comes back, nothing is written.
    assert unapproved.ok and not unapproved.written_back, (unapproved.ok, unapproved.written_back)
    assert unapproved.caption_text == "a cat on a mat" and unapproved.ocr_text == "MAT"
    assert held.updates == [], held.updates
    assert held_box.destroyed == ["wb-session"], held_box.destroyed
    # With approval: one write-back, exactly what was read, for the owner.
    assert approved.ok and approved.written_back, (approved.ok, approved.written_back)
    assert kept.updates == [("wb1", "owner", {"caption_text": "a cat on a mat", "ocr_text": "MAT"})], kept.updates
    assert kept_box.destroyed == ["wb-session"], kept_box.destroyed


def test_wb2_a_transcript_is_written_back_only_when_the_user_approves(tmp_path):
    transcription, restore = _load("transcription", "transcription.py")
    try:
        held, held_box = _Store("audio"), _Sandbox(tmp_path)
        unapproved = transcription.transcribe_attachment(
            "wb2", user_id="owner", store=held, blobs=_Blobs(), sandbox=held_box,
            transcriber=lambda sandbox, session, name: "hello from the garden",
        )
        kept, kept_box = _Store("audio"), _Sandbox(tmp_path)
        approved = transcription.transcribe_attachment(
            "wb2", user_id="owner", store=kept, blobs=_Blobs(), sandbox=kept_box,
            transcriber=lambda sandbox, session, name: "hello from the garden", approve=True,
        )
    finally:
        restore()
    assert unapproved.ok and not unapproved.written_back, (unapproved.ok, unapproved.written_back)
    assert unapproved.transcript_text == "hello from the garden"
    assert held.updates == [], held.updates
    assert held_box.destroyed == ["wb-session"], held_box.destroyed
    assert approved.ok and approved.written_back, (approved.ok, approved.written_back)
    assert kept.updates == [("wb2", "owner", {"transcript_text": "hello from the garden"})], kept.updates
    assert kept_box.destroyed == ["wb-session"], kept_box.destroyed
