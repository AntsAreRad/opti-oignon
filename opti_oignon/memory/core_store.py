#!/usr/bin/env python3
"""The Core of the onion memory: pinned invariants that change only by supersession.

The Core holds what must be true in every window -- persona, hard
constraints, canonical decisions. Three properties make it trustworthy
rather than merely present. It is content-addressed: an entry's id is the
hash of its bytes, so the same text is the same entry and a changed text is
a different one. It changes only by supersession on an explicit user action:
there is no edit surface, the superseded text stays with a link to its
successor, and nothing the model or the librarian does can move it. And it
can prove itself: every read re-hashes every entry, and an entry whose bytes
no longer answer to its id is refused by name, never repaired.

The root is the hash of the canonical text of the active set, in insertion
order. The composer anchors every assembled prompt to it, so a prompt whose
Core bytes differ from the registry's is detectable by anyone holding the
root. This module keeps the store in memory; the onion store writes its
rows to the encrypted table and re-hashes every one on the way back. The
executor reaches this module through the librarian and nothing else does;
a contract on the tree says so.
"""

import hashlib
from dataclasses import dataclass, replace

checkpoint_before_apply = True

# The one actor allowed to pin or supersede. A string rather than a bool so
# that a caller has to say who it is, and a wrong answer reads as one.
USER = "user"


class CoreIntegrityError(ValueError):
    """An entry's bytes no longer answer to its id."""


def _native():
    """The native core, or None: asked at the call, never at import."""
    try:
        from opti_oignon.native import load
    except Exception:  # noqa: BLE001 - absence is the reference path
        return None
    return load()


def entry_hash(text):
    """The id of an entry with this text: SHA-256 of its UTF-8 bytes."""
    core = _native()
    if core is not None:
        return core.entry_hash(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CoreEntry:
    """One pinned statement, addressed by its bytes."""

    id: str
    text: str
    superseded_by: str = None

    @property
    def status(self):
        return "superseded" if self.superseded_by else "active"


class CoreStore:
    """Content-addressed, supersession-only, self-verifying."""

    def __init__(self):
        self._entries = {}
        self._order = []

    @staticmethod
    def _require_user(actor):
        if actor != USER:
            raise PermissionError(
                f"the Core changes only on an explicit user action; refused for actor {actor!r}"
            )

    def add(self, text, *, actor):
        """Pin ``text``. The same bytes pinned twice are one entry."""
        self._require_user(actor)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("an empty statement cannot be pinned")
        entry_id = entry_hash(text)
        if entry_id not in self._entries:
            self._entries[entry_id] = CoreEntry(entry_id, text)
            self._order.append(entry_id)
        return entry_id

    def supersede(self, old_id, text, *, actor):
        """Pin ``text`` as the successor of ``old_id``. The old text is untouched."""
        self._require_user(actor)
        old = self._entries[old_id]
        if old.superseded_by:
            raise ValueError(f"entry {old_id!r} is already superseded by {old.superseded_by!r}")
        new_id = self.add(text, actor=actor)
        if new_id == old_id:
            raise ValueError("an entry cannot supersede itself")
        self._entries[old_id] = replace(old, superseded_by=new_id)
        return new_id

    def get(self, entry_id):
        return self._entries[entry_id]

    def head(self, entry_id):
        """Follow supersession from ``entry_id`` to the entry that currently holds."""
        entry = self._entries[entry_id]
        seen = {entry_id}
        while entry.superseded_by:
            if entry.superseded_by in seen:
                raise ValueError(f"supersession cycle at {entry.superseded_by!r}")
            seen.add(entry.superseded_by)
            entry = self._entries[entry.superseded_by]
        return entry

    def all(self):
        return [self._entries[i] for i in self._order]

    def active(self):
        return [e for e in self.all() if e.status == "active"]

    def verify(self):
        """Refuse, by name, any entry whose bytes no longer hash to its id."""
        for entry in self.all():
            if entry_hash(entry.text) != entry.id:
                raise CoreIntegrityError(
                    f"Core entry {entry.id} no longer answers to its bytes: refused, not repaired"
                )
        return None

    def text(self):
        """The canonical text of the active set, verified first."""
        self.verify()
        return "\n".join(e.text for e in self.active())

    def root(self):
        """The hash the composer anchors a prompt to."""
        return entry_hash(self.text())
