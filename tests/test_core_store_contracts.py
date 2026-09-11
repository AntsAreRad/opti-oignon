#!/usr/bin/env python3
"""Contracts for the Core store of the onion memory.

The Core holds pinned invariants -- persona, hard constraints, canonical
decisions. It is content-addressed, changes only by supersession on an
explicit user action, and can prove that its bytes are the bytes it was
given: an entry whose text no longer answers to its hash is refused, never
repaired.

  * RK1 -- content-addressed: an entry's id is the hash of its bytes, and
    the same text is one entry.
  * RK2 -- supersession only: no edit surface exists; the superseded text is
    untouched, leaves the active set, and links to its successor.
  * RK3 -- explicit user action: an add or a supersession from any other actor
    is refused.
  * RK4 -- tamper refused: an entry whose bytes moved is named and refused
    by the integrity check.
  * RK5 -- the root anchors the content: equal content gives an equal root,
    any change in the active set changes it, and it is the hash of the
    canonical text.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _open():
    loaded, restore = isolate(
        targets={"opti_oignon.memory.core_store": source("memory", "core_store.py")},
        packages=("opti_oignon.memory",),
    )
    return loaded["opti_oignon.memory.core_store"], restore


# ---------------------------------------------------------------------------
# RK1 -- content-addressed
# ---------------------------------------------------------------------------
def test_rk1_an_entry_is_addressed_by_the_hash_of_its_bytes():
    mod, restore = _open()
    try:
        store = mod.CoreStore()
        text = "The user is called Alice."
        first = store.add(text, actor=mod.USER)
        assert first == hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert first == mod.entry_hash(text)
        second = store.add(text, actor=mod.USER)
        assert second == first, "the same bytes are the same entry"
        assert len(store.all()) == 1
        other = store.add("Answers are concise.", actor=mod.USER)
        assert other != first
        assert len(store.all()) == 2
        with pytest.raises(ValueError):
            store.add("   ", actor=mod.USER)
    finally:
        restore()


# ---------------------------------------------------------------------------
# RK2 -- supersession only
# ---------------------------------------------------------------------------
def test_rk2_supersession_links_and_never_edits():
    mod, restore = _open()
    try:
        store = mod.CoreStore()
        for name in ("edit", "update", "remove", "delete", "set", "replace"):
            assert not hasattr(store, name), f"no {name} surface: supersede or nothing"
        old = store.add("Alice lives in Berlin.", actor=mod.USER)
        new = store.supersede(old, "Alice lives in Oslo.", actor=mod.USER)
        assert store.get(old).text == "Alice lives in Berlin.", "the superseded text is untouched"
        assert store.get(old).superseded_by == new
        assert store.get(old).status == "superseded"
        assert store.head(old).id == new
        assert [e.id for e in store.active()] == [new]
        assert len(store.all()) == 2
        with pytest.raises(ValueError):
            store.supersede(old, "Alice lives in Rome.", actor=mod.USER)
        with pytest.raises(KeyError):
            store.supersede("0" * 64, "nothing", actor=mod.USER)
    finally:
        restore()


# ---------------------------------------------------------------------------
# RK3 -- explicit user action
# ---------------------------------------------------------------------------
def test_rk3_only_the_user_pins_or_supersedes():
    mod, restore = _open()
    try:
        store = mod.CoreStore()
        for actor in ("librarian", "model", "", None):
            with pytest.raises(PermissionError):
                store.add("Pinned by someone else.", actor=actor)
        assert store.all() == []
        held = store.add("Pinned by the user.", actor=mod.USER)
        with pytest.raises(PermissionError):
            store.supersede(held, "Rewritten by the model.", actor="model")
        assert store.get(held).superseded_by is None
        assert len(store.all()) == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# RK4 -- tamper refused
# ---------------------------------------------------------------------------
def test_rk4_an_entry_whose_bytes_moved_is_refused_by_name():
    mod, restore = _open()
    try:
        store = mod.CoreStore()
        held = store.add("The demo must not use Docker.", actor=mod.USER)
        store.add("Bob reviews the release.", actor=mod.USER)
        assert store.verify() is None, "control: an untouched store verifies"
        entry = store.get(held)
        object.__setattr__(entry, "text", "The demo must use Docker.")
        with pytest.raises(mod.CoreIntegrityError) as caught:
            store.verify()
        assert held in str(caught.value)
        with pytest.raises(mod.CoreIntegrityError):
            store.text()
    finally:
        restore()


# ---------------------------------------------------------------------------
# RK5 -- the root anchors the content
# ---------------------------------------------------------------------------
def test_rk5_the_root_is_the_hash_of_the_canonical_text():
    mod, restore = _open()
    try:
        a, b = mod.CoreStore(), mod.CoreStore()
        for store in (a, b):
            store.add("One.", actor=mod.USER)
            store.add("Two.", actor=mod.USER)
        assert a.root() == b.root()
        assert a.text() == "One.\nTwo."
        assert a.root() == hashlib.sha256(a.text().encode("utf-8")).hexdigest()
        before = a.root()
        a.add("Three.", actor=mod.USER)
        assert a.root() != before
        added = a.root()
        a.supersede(mod.entry_hash("Two."), "Two, revised.", actor=mod.USER)
        assert a.root() not in (before, added)
        assert a.text() == "One.\nThree.\nTwo, revised."
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
