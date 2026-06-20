#!/usr/bin/env python3
"""S263 store suite: the N.8 first implementation lot, the note_update store.

NOTES_CRDT_SPEC.md (decided at S262) binds this lot; sections 2, 4 and 5 are
its text. The lot is store-only and purely additive: ONE new module,
``opti_oignon/notes/note_updates_store.py``, carrying the at-rest half of the
collaboration model -- the append-only ``note_update`` table on the spec's
binding shape (unique ``(user_id, note_id, seq)``, per-user isolation via
``effective_user_id``, ``safe_connect`` with the documented degradation,
parameterized SQL only), the section-4 watermark and prune primitives (a
``note_checkpoint`` sibling table; prune at-or-below the checkpoint
watermark; prune-on-tombstone guarded by liveness), and the section-5
fail-secure refusal at the append seam (an unknown or dead parent refuses;
an indeterminable liveness refuses; refused means not appended and loggable,
never silent). The journaling glue on the S256 seam is the NEXT lot; zero
existing sources are edited at S263.

Red-before discipline: on the pristine S262 tree (no
``note_updates_store.py``) every module, source-discipline and behavior test
FAILS as an assertion -- the guarded loader returns ``None`` and the source
read helper returns an empty string, so absence is an assertion failure,
never a collection error -- while the final family passes by design (it pins
the spec sentence and the existing Notes surfaces this lot relies on).
Declared red-before split: 36 red / 3 design-green over 39.

Behavior tests build the store on a pytest ``tmp_path`` root; the refusal
family seeds a real ``NotesStore`` at the same root so the DEFAULT parent
lookup (the sibling ``notes.db`` read, fail-secure) is exercised end to end;
the other families inject a permissive lookup to isolate the mechanics under
test. Nothing here touches the repository ``data/`` posture.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "NOTES_CRDT_SPEC.md"
MODULE_PATH = REPO / "opti_oignon" / "notes" / "note_updates_store.py"
ROUTES_NOTES = REPO / "opti_oignon" / "api" / "routes_notes.py"
NOTES_STORE = REPO / "opti_oignon" / "notes" / "notes_store.py"


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _flat(text: str) -> str:
    """Collapse all whitespace runs to single spaces (reflow-immune pins)."""
    return re.sub(r"\s+", " ", text)


def _load_module():
    """Import the new store module; ``None`` when absent (assert-before-call)."""
    try:
        import importlib

        return importlib.import_module("opti_oignon.notes.note_updates_store")
    except Exception:
        return None


def _permissive(_note_id: str, _user_id: str) -> bool:
    return True


def _make_store(root: Path, **kwargs):
    """Build a NoteUpdatesStore at ``root``; asserts the module first."""
    mod = _load_module()
    assert mod is not None, "note_updates_store module is absent"
    return mod, mod.NoteUpdatesStore(root, **kwargs)


# ---------------------------------------------------------------------------
# Family 1 -- module surface (red-before: the module does not exist yet)
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_module_importable(self):
        assert _load_module() is not None, (
            "opti_oignon/notes/note_updates_store.py is absent"
        )

    def test_checkpoint_before_apply_hardcoded_true(self):
        mod = _load_module()
        assert mod is not None, "module absent"
        assert mod.checkpoint_before_apply is True

    def test_feature_available_flag(self):
        mod = _load_module()
        assert mod is not None, "module absent"
        assert mod.FEATURE_AVAILABLE is True

    def test_refusal_exception_shape(self):
        mod = _load_module()
        assert mod is not None, "module absent"
        exc = mod.NoteUpdateRefused("the reason", note_id="n1")
        assert isinstance(exc, Exception)
        assert exc.reason == "the reason"

    def test_singleton_hooks_exposed(self):
        mod = _load_module()
        assert mod is not None, "module absent"
        assert callable(mod.get_note_updates_store)
        assert callable(mod.reset_note_updates_store)
        mod.reset_note_updates_store()


# ---------------------------------------------------------------------------
# Family 2 -- source discipline (the spec's pinned house-rules sentence)
# ---------------------------------------------------------------------------


class TestSourceDiscipline:
    def test_source_present_and_ascii(self):
        raw = b""
        try:
            raw = MODULE_PATH.read_bytes()
        except OSError:
            raw = b""
        assert raw != b"", "note_updates_store.py source is absent"
        raw.decode("ascii")

    def test_no_sql_fstrings_four_verbs(self):
        src = _read(MODULE_PATH)
        assert src != "", "note_updates_store.py source is absent"
        for verb in ("SELECT", "INSERT", "UPDATE", "DELETE"):
            assert 'f"' + verb not in src
            assert "f'" + verb not in src

    def test_house_rule_imports_present(self):
        src = _read(MODULE_PATH)
        assert src != "", "note_updates_store.py source is absent"
        assert "safe_connect" in src
        assert "effective_user_id" in src
        assert "checkpoint_before_apply = True" in src

    def test_no_update_statement_touches_update_blob(self):
        src = _read(MODULE_PATH)
        assert src != "", "note_updates_store.py source is absent"
        assert "UPDATE note_update " not in src
        assert "update_blob =" not in src


# ---------------------------------------------------------------------------
# Family 3 -- append mechanics (permissive lookup; tmp_path root)
# ---------------------------------------------------------------------------


class TestAppend:
    def test_append_returns_record_minted_seq_one(self, tmp_path):
        _mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        rec = store.append_update("n1", b"\x01\x02")
        assert rec.seq == 1
        assert rec.note_id == "n1"

    def test_append_mints_sequential_seqs(self, tmp_path):
        _mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        first = store.append_update("n1", b"a")
        second = store.append_update("n1", b"b")
        assert (first.seq, second.seq) == (1, 2)
        assert store.latest_seq("n1") == 2

    def test_append_explicit_seq_honoured(self, tmp_path):
        _mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        rec = store.append_update("n1", b"x", seq=7)
        assert rec.seq == 7
        assert store.latest_seq("n1") == 7

    def test_append_roundtrips_blob_bytes(self, tmp_path):
        _mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        payload = b"\x00\xffyjs-update\x10"
        store.append_update("n1", payload)
        rows = store.list_updates("n1")
        assert len(rows) == 1
        assert bytes(rows[0].update_blob) == payload

    def test_append_records_author_device_and_created_at(self, tmp_path):
        _mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        rec = store.append_update("n1", b"u", author_device="dev-a")
        assert rec.author_device == "dev-a"
        assert rec.created_at != ""


# ---------------------------------------------------------------------------
# Family 4 -- per-user isolation (the memory-store reference pattern)
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_list_and_count_scoped_per_user(self, tmp_path):
        _mod, store = _make_store(
            tmp_path, single_user_mode=False, parent_lookup=_permissive
        )
        store.append_update("n1", b"a", user_id="alice")
        assert store.count_updates("n1", user_id="alice") == 1
        assert store.count_updates("n1", user_id="bob") == 0
        assert store.list_updates("n1", user_id="bob") == []

    def test_latest_seq_scoped_per_user(self, tmp_path):
        _mod, store = _make_store(
            tmp_path, single_user_mode=False, parent_lookup=_permissive
        )
        store.append_update("n1", b"a", user_id="alice")
        store.append_update("n1", b"b", user_id="alice")
        assert store.latest_seq("n1", user_id="alice") == 2
        assert store.latest_seq("n1", user_id="bob") == 0

    def test_seq_namespaces_per_user_same_note(self, tmp_path):
        _mod, store = _make_store(
            tmp_path, single_user_mode=False, parent_lookup=_permissive
        )
        store.append_update("n1", b"a", user_id="alice")
        store.append_update("n1", b"b", user_id="alice")
        rec = store.append_update("n1", b"c", user_id="bob")
        assert rec.seq == 1


# ---------------------------------------------------------------------------
# Family 5 -- the unique (user_id, note_id, seq) constraint
# ---------------------------------------------------------------------------


class TestUniqueConstraint:
    def test_duplicate_seq_refuses_and_preserves_original(self, tmp_path):
        mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        store.append_update("n1", b"original", seq=3)
        with pytest.raises(mod.NoteUpdateRefused):
            store.append_update("n1", b"intruder", seq=3)
        rows = store.list_updates("n1")
        assert len(rows) == 1
        assert bytes(rows[0].update_blob) == b"original"

    def test_same_seq_distinct_notes_ok(self, tmp_path):
        _mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        store.append_update("n1", b"a", seq=3)
        rec = store.append_update("n2", b"b", seq=3)
        assert rec.seq == 3

    def test_same_seq_distinct_users_ok(self, tmp_path):
        _mod, store = _make_store(
            tmp_path, single_user_mode=False, parent_lookup=_permissive
        )
        store.append_update("n1", b"a", seq=3, user_id="alice")
        rec = store.append_update("n1", b"b", seq=3, user_id="bob")
        assert rec.seq == 3


# ---------------------------------------------------------------------------
# Family 6 -- the section-5 refusal at the append seam (fail-secure)
# ---------------------------------------------------------------------------


class TestRefusal:
    def test_unknown_parent_refuses_default_lookup(self, tmp_path):
        mod, store = _make_store(tmp_path)
        with pytest.raises(mod.NoteUpdateRefused):
            store.append_update("ghost", b"u")
        assert store.count_updates("ghost") == 0
        assert not (tmp_path / "notes.db").exists(), (
            "the default lookup must never create the sibling notes.db"
        )

    def test_known_live_parent_appends_default_lookup(self, tmp_path):
        _mod, store = _make_store(tmp_path)
        from opti_oignon.notes.notes_store import NotesStore

        notes = NotesStore(tmp_path)
        note = notes.add_note("live parent")
        rec = store.append_update(note.id, b"u")
        assert rec.seq == 1

    def test_tombstoned_parent_refuses_default_lookup(self, tmp_path):
        mod, store = _make_store(tmp_path)
        from opti_oignon.notes.notes_store import NotesStore

        notes = NotesStore(tmp_path)
        note = notes.add_note("doomed parent")
        assert notes.delete_note(note.id) is True
        with pytest.raises(mod.NoteUpdateRefused):
            store.append_update(note.id, b"u")
        assert store.count_updates(note.id) == 0

    def test_lookup_exception_refuses_fail_secure(self, tmp_path):
        def _broken(_note_id: str, _user_id: str) -> bool:
            raise RuntimeError("liveness backend down")

        mod, store = _make_store(tmp_path, parent_lookup=_broken)
        with pytest.raises(mod.NoteUpdateRefused):
            store.append_update("n1", b"u")
        assert store.count_updates("n1") == 0

    def test_missing_blob_refuses(self, tmp_path):
        mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        with pytest.raises(mod.NoteUpdateRefused):
            store.append_update("n1", None)
        assert store.count_updates("n1") == 0

    def test_refusal_is_logged_warning(self, tmp_path, caplog):
        mod, store = _make_store(tmp_path)
        with caplog.at_level(logging.WARNING):
            with pytest.raises(mod.NoteUpdateRefused):
                store.append_update("ghost", b"u")
        warned = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warned) >= 1, "a refusal must be loggable, never silent"


# ---------------------------------------------------------------------------
# Family 7 -- the section-4 checkpoint watermark
# ---------------------------------------------------------------------------


class TestWatermark:
    def test_latest_seq_zero_when_empty(self, tmp_path):
        _mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        assert store.latest_seq("n1") == 0

    def test_set_then_get_watermark(self, tmp_path):
        _mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        assert store.get_checkpoint_watermark("n1") == 0
        assert store.set_checkpoint_watermark("n1", 5) is True
        assert store.get_checkpoint_watermark("n1") == 5

    def test_watermark_monotonic_non_decreasing(self, tmp_path):
        _mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        assert store.set_checkpoint_watermark("n1", 5) is True
        assert store.set_checkpoint_watermark("n1", 3) is False
        assert store.get_checkpoint_watermark("n1") == 5
        assert store.set_checkpoint_watermark("n1", 7) is True
        assert store.get_checkpoint_watermark("n1") == 7

    def test_watermark_scoped_per_user(self, tmp_path):
        _mod, store = _make_store(
            tmp_path, single_user_mode=False, parent_lookup=_permissive
        )
        store.set_checkpoint_watermark("n1", 4, user_id="alice")
        assert store.get_checkpoint_watermark("n1", user_id="alice") == 4
        assert store.get_checkpoint_watermark("n1", user_id="bob") == 0


# ---------------------------------------------------------------------------
# Family 8 -- prune at-or-below the watermark (local, lazy, never over-prune)
# ---------------------------------------------------------------------------


class TestPruneWatermark:
    def test_prunes_rows_at_or_below_watermark(self, tmp_path):
        _mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        for i in range(1, 6):
            store.append_update("n1", b"u%d" % i)
        assert store.set_checkpoint_watermark("n1", 3) is True
        pruned = store.prune_below_watermark("n1")
        assert pruned == 3
        remaining = [r.seq for r in store.list_updates("n1")]
        assert remaining == [4, 5]

    def test_no_watermark_prunes_nothing(self, tmp_path):
        _mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        store.append_update("n1", b"a")
        store.append_update("n1", b"b")
        assert store.prune_below_watermark("n1") == 0
        assert store.count_updates("n1") == 2

    def test_serving_tail_survives_after_prune(self, tmp_path):
        _mod, store = _make_store(tmp_path, parent_lookup=_permissive)
        for i in range(1, 6):
            store.append_update("n1", b"u%d" % i)
        store.set_checkpoint_watermark("n1", 3)
        store.prune_below_watermark("n1")
        tail = [r.seq for r in store.list_updates("n1", after_seq=3)]
        assert tail == [4, 5]
        assert store.latest_seq("n1") == 5


# ---------------------------------------------------------------------------
# Family 9 -- prune on tombstone (full tail, guarded by liveness)
# ---------------------------------------------------------------------------


class TestPruneTombstone:
    def test_dead_parent_prunes_full_tail(self, tmp_path):
        _mod, store = _make_store(tmp_path)
        from opti_oignon.notes.notes_store import NotesStore

        notes = NotesStore(tmp_path)
        note = notes.add_note("short-lived")
        for i in range(3):
            store.append_update(note.id, b"u%d" % i)
        assert notes.delete_note(note.id) is True
        assert store.prune_for_tombstone(note.id) == 3
        assert store.count_updates(note.id) == 0

    def test_live_parent_refuses_full_prune(self, tmp_path):
        _mod, store = _make_store(tmp_path)
        from opti_oignon.notes.notes_store import NotesStore

        notes = NotesStore(tmp_path)
        note = notes.add_note("still alive")
        store.append_update(note.id, b"u")
        assert store.prune_for_tombstone(note.id) == 0
        assert store.count_updates(note.id) == 1

    def test_indeterminable_liveness_refuses_prune(self, tmp_path):
        def _flaky(_note_id: str, _user_id: str) -> bool:
            raise RuntimeError("liveness backend down")

        mod, writer = _make_store(tmp_path, parent_lookup=_permissive)
        writer.append_update("n1", b"u")
        pruner = mod.NoteUpdatesStore(tmp_path, parent_lookup=_flaky)
        assert pruner.prune_for_tombstone("n1") == 0
        assert pruner.count_updates("n1") == 1


# ---------------------------------------------------------------------------
# Family 10 -- the spec and the Notes surfaces stay alive (design-green)
# ---------------------------------------------------------------------------


class TestSpecAndSurfacesAlive:
    def test_crdt_spec_atrest_sentence_alive(self):
        flat = _flat(_read(SPEC_PATH))
        assert (
            "an append-only note_update table, isolated per user via "
            "effective_user_id" in flat
        )
        assert "unique on (user_id, note_id, seq)" in flat

    def test_five_notes_routes_exact(self):
        src = _read(ROUTES_NOTES)
        assert src.count("@notes_router.") == 5
        assert '@notes_router.patch("/{note_id}"' in src

    def test_notes_store_house_rules_alive(self):
        src = _read(NOTES_STORE)
        assert "safe_connect" in src
        assert "_ORDERABLE_COLUMNS" in src
