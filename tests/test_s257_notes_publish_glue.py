#!/usr/bin/env python3
"""S257 -- the notes publisher glue (the N.8/N.9 delivery seam).

Through S256, notes had NO production publisher: ``publish_note`` was the
S252 engine seam with zero callers, so a note created, edited, flagged, or
tombstoned on this desktop never reached the journal outside tests. S257
lands the best-effort glue the skills and conversations already ride
(``_sync_publish_skill`` / ``_sync_publish_conversation``), at the STORE
layer so every caller is covered -- the routes, the gated ``manage_notes``
tool (which calls the store's mutation methods directly), and any future
caller -- without touching ``tools.py`` (the s256 negative pin) or
``routes_notes.py`` (the s245 five-routes-exact pin).

The contract under test (SYN-01, the established idiom):

- ``_sync_publish_note(note_id, payload_fn, *, deleted=False,
  updated_at="")`` at module level in ``notes_store.py``, serialised by a
  module-level ``_SYNC_LOCK`` (the skills adaptation for a store whose own
  lock must not be held across the journal append).
- Called AFTER the domain commit at every mutation seam: ``add_note``
  (create), ``update_note`` (only when a column was actually applied and
  the refreshed record exists), ``delete_note`` (tombstone, only when a row
  actually flipped), and ``set_mobile_allowed`` (the runbook's
  republish-on-opt-in delivery contract: a flag flip journals a fresh
  full-state record in BOTH directions -- delivery, not security; the
  serve filter's live lookup stays the security authority, N9-D1/N9-D2).
- Best-effort throughout: the availability probe runs first and an absent
  framework pays nothing (no payload build, no journal append); any
  failure -- a raising engine, an unreachable sync package under a stubbed
  interpreter -- is swallowed and the save is unaffected.
- The payload is the full JSON-safe note state (title, the opaque CRDT
  body base64-encoded, the opaque OR-Set tags string verbatim, pinned,
  created_at) and deliberately EXCLUDES ``mobile_allowed``: carrying the
  flag would make a future apply path a writer of it, contradicting N9-D3
  (the route's dedicated setter is the only writer).

Loading follows the S243/S244/S256 isolation idiom: the store flat-loaded
under a suite-private name; the guard and engine dotted-loaded under
``_ensure_pkg`` seeds so the glue's lazy imports resolve to patchable
modules. Design-green families (the reassertions, structure, the retained
roadmap rolls) pass on the pristine S256 tree; every other family reds
there as an assertion failure, never a collection error.
"""

from __future__ import annotations

import ast
import base64
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "opti_oignon"
VEILID = PKG / "veilid"

NOTES_STORE_SRC = PKG / "notes" / "notes_store.py"
TOOLS_SRC = PKG / "agent" / "tools.py"
ALLOWLISTS_SRC = PKG / "agent" / "allowlists.py"
ROUTES_NOTES_SRC = PKG / "api" / "routes_notes.py"
PRODUCERS_SRC = VEILID / "producers.py"
ENGINE_SRC = VEILID / "sync_engine.py"
GUARD_SRC = VEILID / "guard.py"
VERSION_PATH = PKG / "__version__.py"
ROADMAP_PATH = ROOT / "NOTES_FEATURE_ROADMAP.md"


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _flat(text: str) -> str:
    """Whitespace-collapsed text for phrase pins across wrapped lines."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Isolation harness (the S243 lesson, the S244 / S256 idiom)
# ---------------------------------------------------------------------------


def _ensure_pkg(name: str, path: Path) -> None:
    """Ensure ``name`` exists in sys.modules and is package-like.

    Non-destructive: keeps any pre-existing stub object (an earlier suite's),
    only granting it a ``__path__`` so a dotted load of a submodule resolves.
    """
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if not hasattr(mod, "__path__"):
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]


def _load_dotted(name: str, path: Path):
    """Load a module under its real dotted name, reusing an existing load."""
    existing = sys.modules.get(name)
    if existing is not None and hasattr(existing, "__file__"):
        return existing
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_pkg("opti_oignon", PKG)
_ensure_pkg("opti_oignon.veilid", VEILID)
_ensure_pkg("opti_oignon.agent", PKG / "agent")


_ISO: dict = {}


def _isolated_flat(name: str, rel: str):
    """Load a module under a FLAT name private to this suite."""
    if name not in _ISO:
        spec = importlib.util.spec_from_file_location(name, str(PKG / rel))
        if spec is None or spec.loader is None:
            raise ImportError(name)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _ISO[name] = mod
    return _ISO[name]


def _store_module():
    """The real NotesStore module, flat-loaded (plaintext-sqlite fallback)."""
    return _isolated_flat("s257_notes_store_iso", "notes/notes_store.py")


def _make_store(tmp_path):
    return _store_module().NotesStore(root=str(tmp_path))


def _guard_module():
    """The real guard module under its dotted name (the glue's probe seam)."""
    return _load_dotted("opti_oignon.veilid.guard", GUARD_SRC)


def _engine_module():
    """The real sync_engine module under its dotted name (patchable)."""
    return _load_dotted("opti_oignon.veilid.sync_engine", ENGINE_SRC)


class _StubEngine:
    """Records ``publish_note`` calls and mints clocks like the journal."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._clocks: dict[str, int] = {}

    def current_clock(self, kind, key: str) -> int:
        return self._clocks.get(key, 0)

    def publish_note(
        self,
        note_id: str,
        payload=None,
        *,
        clock: int,
        deleted: bool = False,
        updated_at: str = "",
    ) -> int:
        self._clocks[note_id] = clock
        self.calls.append(
            {
                "note_id": note_id,
                "payload": payload,
                "clock": clock,
                "deleted": deleted,
                "updated_at": updated_at,
            }
        )
        return len(self.calls)


class _RaisingEngine(_StubEngine):
    """A journal that always fails: the best-effort posture under test."""

    def publish_note(self, *args, **kwargs) -> int:  # noqa: D102
        raise RuntimeError("journal down")


@pytest.fixture
def live_glue(monkeypatch):
    """Veilid 'present' plus a recording stub engine behind the glue."""
    guard = _guard_module()
    engine_mod = _engine_module()
    stub = _StubEngine()
    monkeypatch.setattr(guard, "veilid_available", lambda: True)
    monkeypatch.setattr(engine_mod, "get_sync_engine", lambda: stub)
    return stub


@pytest.fixture
def raising_glue(monkeypatch):
    """Veilid 'present' plus a journal that raises on every publish."""
    guard = _guard_module()
    engine_mod = _engine_module()
    stub = _RaisingEngine()
    monkeypatch.setattr(guard, "veilid_available", lambda: True)
    monkeypatch.setattr(engine_mod, "get_sync_engine", lambda: stub)
    return stub


@pytest.fixture
def absent_glue(monkeypatch):
    """Veilid explicitly absent (the container default, pinned explicit)."""
    guard = _guard_module()
    monkeypatch.setattr(guard, "veilid_available", lambda: False)
    return guard


# ---------------------------------------------------------------------------
# Family 1 -- the glue surface (source and structure; red before the edit)
# ---------------------------------------------------------------------------


class TestGlueSurface:
    def test_glue_function_exists(self):
        mod = _store_module()
        fn = getattr(mod, "_sync_publish_note", None)
        assert callable(fn)

    def test_module_sync_lock_present(self):
        # The skills adaptation: a module-level lock serialises mint + append
        # so the store's own lock is never held across the journal.
        mod = _store_module()
        lock = getattr(mod, "_SYNC_LOCK", None)
        assert lock is not None
        assert hasattr(lock, "acquire") and hasattr(lock, "release")

    def test_glue_source_probes_guard_first(self):
        src = _read(NOTES_STORE_SRC)
        assert "_sync_publish_note" in src
        assert "veilid_available" in src

    def test_glue_source_swallows_failures(self):
        # The best-effort posture is visible in source: the publish path is
        # protected and a failure is logged, never raised into the save.
        src = _flat(_read(NOTES_STORE_SRC))
        assert "save unaffected" in src

    def test_payload_builder_exists(self):
        mod = _store_module()
        fn = getattr(mod, "_note_sync_payload", None)
        assert callable(fn)

    def test_mutation_seams_call_the_glue(self):
        # Source-level: each mutation method body references the glue. AST
        # walk so a comment can never satisfy the pin.
        src = _read(NOTES_STORE_SRC)
        tree = ast.parse(src)
        wanted = {"add_note", "update_note", "delete_note", "set_mobile_allowed"}
        calling: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name)
                        and inner.func.id == "_sync_publish_note"
                    ):
                        calling.add(node.name)
        assert calling == wanted


# ---------------------------------------------------------------------------
# Family 2 -- the absent-framework posture (pays nothing)
# ---------------------------------------------------------------------------


class TestAbsentPosture:
    def test_direct_call_skips_payload_build_when_absent(self, absent_glue):
        mod = _store_module()
        fn = getattr(mod, "_sync_publish_note", None)
        assert callable(fn)
        built = {"n": 0}

        def payload_fn():
            built["n"] += 1
            return {"title": "t"}

        fn("n1", payload_fn)
        assert built["n"] == 0

    def test_add_note_publishes_nothing_when_absent(
        self, tmp_path, absent_glue, monkeypatch
    ):
        # Even with an engine stub installed, the probe gates everything.
        engine_mod = _engine_module()
        stub = _StubEngine()
        monkeypatch.setattr(engine_mod, "get_sync_engine", lambda: stub)
        store = _make_store(tmp_path)
        store.add_note("quiet")
        assert stub.calls == []


# ---------------------------------------------------------------------------
# Family 3 -- journaling at every mutation seam (the positive contract)
# ---------------------------------------------------------------------------


class TestJournalingSeams:
    def test_create_journals_once(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        record = store.add_note("first")
        assert len(live_glue.calls) == 1
        call = live_glue.calls[0]
        assert call["note_id"] == record.id
        assert call["deleted"] is False
        assert call["clock"] == 1

    def test_create_payload_is_full_state(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        store.add_note(
            "shape", body_crdt=b"\x00\x01body", tags='["a","b"]', pinned=True
        )
        assert len(live_glue.calls) == 1
        payload = live_glue.calls[0]["payload"]
        assert payload is not None
        assert payload.get("title") == "shape"
        assert base64.b64decode(payload.get("body_crdt_b64", "")) == b"\x00\x01body"
        assert payload.get("tags") == '["a","b"]'
        assert payload.get("pinned") is True
        assert payload.get("created_at")

    def test_payload_excludes_mobile_allowed(self, tmp_path, live_glue):
        # N9-D3 adjacency: the flag never rides the wire payload, so a
        # future apply path can never become a writer of it.
        store = _make_store(tmp_path)
        nid = store.add_note("local trust").id
        store.set_mobile_allowed(nid, True)
        for call in live_glue.calls:
            payload = call["payload"]
            if payload is not None:
                assert "mobile_allowed" not in payload

    def test_update_journals_fresh_state(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        nid = store.add_note("before").id
        store.update_note(nid, title="after")
        assert len(live_glue.calls) == 2
        call = live_glue.calls[1]
        assert call["note_id"] == nid
        assert call["payload"] is not None
        assert call["payload"].get("title") == "after"
        assert call["clock"] == 2

    def test_empty_update_does_not_journal(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        nid = store.add_note("steady").id
        store.update_note(nid)
        assert len(live_glue.calls) == 1

    def test_update_unknown_note_does_not_journal(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        store.add_note("anchor")
        store.update_note("missing", title="ghost")
        assert len(live_glue.calls) == 1

    def test_tombstone_journals_deleted(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        nid = store.add_note("doomed").id
        assert store.delete_note(nid) is True
        assert len(live_glue.calls) == 2
        call = live_glue.calls[1]
        assert call["note_id"] == nid
        assert call["deleted"] is True
        assert call["payload"] is None

    def test_double_delete_journals_once(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        nid = store.add_note("once").id
        store.delete_note(nid)
        assert store.delete_note(nid) is False
        assert len(live_glue.calls) == 2

    def test_delete_unknown_note_does_not_journal(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        store.add_note("anchor")
        assert store.delete_note("missing") is False
        assert len(live_glue.calls) == 1

    def test_flag_flip_to_allowed_journals(self, tmp_path, live_glue):
        # The runbook's republish-on-opt-in delivery contract, now
        # container-proven at the journaling half: a phone whose watermark
        # has advanced past the filtered entries sees the fresh record.
        store = _make_store(tmp_path)
        nid = store.add_note("opted in").id
        assert store.set_mobile_allowed(nid, True) is True
        assert len(live_glue.calls) == 2
        call = live_glue.calls[1]
        assert call["note_id"] == nid
        assert call["deleted"] is False
        assert call["payload"] is not None

    def test_flag_flip_to_disallowed_also_journals(self, tmp_path, live_glue):
        # Delivery in both directions; the serve filter's live lookup stays
        # the security authority regardless (N9-D1/N9-D2).
        store = _make_store(tmp_path)
        nid = store.add_note("opted out").id
        store.set_mobile_allowed(nid, True)
        assert store.set_mobile_allowed(nid, False) is True
        assert len(live_glue.calls) == 3

    def test_flag_flip_unknown_note_does_not_journal(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        store.add_note("anchor")
        assert store.set_mobile_allowed("missing", True) is False
        assert len(live_glue.calls) == 1

    def test_flag_flip_on_tombstone_does_not_journal(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        nid = store.add_note("gone").id
        store.delete_note(nid)
        assert store.set_mobile_allowed(nid, True) is False
        assert len(live_glue.calls) == 2

    def test_clock_monotonic_per_note(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        nid = store.add_note("ticks").id
        store.update_note(nid, title="tick")
        store.set_mobile_allowed(nid, True)
        store.delete_note(nid)
        clocks = [c["clock"] for c in live_glue.calls if c["note_id"] == nid]
        assert clocks == [1, 2, 3, 4]

    def test_clocks_independent_across_notes(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        a = store.add_note("a").id
        b = store.add_note("b").id
        store.update_note(a, title="a2")
        by_note = {}
        for call in live_glue.calls:
            by_note.setdefault(call["note_id"], []).append(call["clock"])
        assert by_note.get(a) == [1, 2]
        assert by_note.get(b) == [1]

    def test_updated_at_rides_the_record(self, tmp_path, live_glue):
        store = _make_store(tmp_path)
        nid = store.add_note("when").id
        store.update_note(nid, title="when2")
        assert len(live_glue.calls) == 2
        refreshed = store.get_note(nid)
        assert live_glue.calls[1]["updated_at"] == refreshed.updated_at
        assert live_glue.calls[1]["updated_at"]


# ---------------------------------------------------------------------------
# Family 4 -- best-effort: a failing journal never breaks the save
# ---------------------------------------------------------------------------


class TestBestEffort:
    def test_create_survives_publish_error(self, tmp_path, raising_glue):
        store = _make_store(tmp_path)
        record = store.add_note("kept")
        assert store.get_note(record.id) is not None

    def test_update_survives_publish_error(self, tmp_path, raising_glue):
        store = _make_store(tmp_path)
        nid = store.add_note("kept").id
        record = store.update_note(nid, title="kept2")
        assert record is not None
        assert store.get_note(nid).title == "kept2"

    def test_delete_survives_publish_error(self, tmp_path, raising_glue):
        store = _make_store(tmp_path)
        nid = store.add_note("kept").id
        assert store.delete_note(nid) is True
        assert store.get_note(nid).deleted is True

    def test_flag_flip_survives_publish_error(self, tmp_path, raising_glue):
        store = _make_store(tmp_path)
        nid = store.add_note("kept").id
        assert store.set_mobile_allowed(nid, True) is True
        assert store.is_mobile_allowed(nid) is True

    def test_probe_error_is_swallowed(self, tmp_path, monkeypatch):
        guard = _guard_module()

        def boom():
            raise RuntimeError("probe down")

        monkeypatch.setattr(guard, "veilid_available", boom)
        store = _make_store(tmp_path)
        record = store.add_note("still saved")
        assert store.get_note(record.id) is not None

    def test_engine_resolution_error_is_swallowed(self, tmp_path, monkeypatch):
        guard = _guard_module()
        engine_mod = _engine_module()
        monkeypatch.setattr(guard, "veilid_available", lambda: True)

        def boom():
            raise RuntimeError("no engine")

        monkeypatch.setattr(engine_mod, "get_sync_engine", boom)
        store = _make_store(tmp_path)
        record = store.add_note("still saved")
        assert store.get_note(record.id) is not None


# ---------------------------------------------------------------------------
# Family 5 -- every caller is covered: the gated tool rides the store glue
# ---------------------------------------------------------------------------


class TestToolCoverage:
    def _handler(self, tmp_path):
        allow = _load_dotted("opti_oignon.agent.allowlists", ALLOWLISTS_SRC)
        assert allow is not None  # load order: allowlists before tools
        tools = _load_dotted("opti_oignon.agent.tools", TOOLS_SRC)
        store = _make_store(tmp_path)
        return tools.make_manage_notes_handler(store), store

    def test_tool_make_journals_via_store(self, tmp_path, live_glue):
        handler, _store = self._handler(tmp_path)
        out = handler({"action": "make", "title": "via tool"})
        assert "Note created" in out
        assert len(live_glue.calls) == 1
        payload = live_glue.calls[0]["payload"]
        assert payload is not None
        assert payload.get("title") == "via tool"

    def test_tool_delete_journals_tombstone(self, tmp_path, live_glue):
        handler, store = self._handler(tmp_path)
        nid = store.add_note("tool doomed").id
        out = handler({"action": "delete", "note_id": nid})
        assert "deleted" in out
        assert len(live_glue.calls) == 2
        assert live_glue.calls[-1]["deleted"] is True

    def test_tool_still_cannot_flip_the_flag(self, tmp_path, live_glue):
        # The N9-D3 structural guard survives the glue: the generic update
        # path raises on the flag, and the tool exposes no verb for it.
        store = _make_store(tmp_path)
        nid = store.add_note("guarded").id
        with pytest.raises(ValueError):
            store.update_note(nid, mobile_allowed=True)


# ---------------------------------------------------------------------------
# Family 6 -- design-green reassertions (must pass on the pristine tree)
# ---------------------------------------------------------------------------


class TestReassertions:
    def test_tools_source_zero_flag_occurrences(self):
        # The s256 negative pin, reasserted: the glue lives in the store,
        # never on the tool surface.
        assert "mobile_allowed" not in _read(TOOLS_SRC)

    def test_flag_still_excluded_from_updatable_columns(self):
        mod = _store_module()
        assert "mobile_allowed" not in mod.NotesStore._UPDATABLE_COLUMNS

    def test_routes_source_untouched_five_routes_exact(self):
        # The s245 pin by construction: the glue adds no route leg.
        src = _read(ROUTES_NOTES_SRC)
        assert src.count("@notes_router.") == 5

    def test_producers_note_record_source_present(self):
        # The s252 presence pin: producers themselves do not change.
        assert "def note_record" in _read(PRODUCERS_SRC)

    def test_engine_publish_note_source_present(self):
        assert "def publish_note" in _read(ENGINE_SRC)

    def test_version_held_3_12_0(self):
        ns: dict = {}
        exec(_read(VERSION_PATH), ns)
        assert ns.get("__version__") == "3.12.0"


# ---------------------------------------------------------------------------
# Family 7 -- the roadmap roll
# ---------------------------------------------------------------------------


class TestRoadmap:
    def test_roadmap_glue_rolled(self):
        text = _flat(_read(ROADMAP_PATH))
        assert "publisher glue LANDED at S257" in text

    def test_roadmap_retains_prior_rolls(self):
        # Design-green: the N.1 (s243) and N.9 (s256) rolls survive.
        text = _flat(_read(ROADMAP_PATH))
        assert "LANDED at S243" in text
        assert "contract / seam half LANDED at S256" in text


# ---------------------------------------------------------------------------
# Family 8 -- structure: AST, ASCII, this suite
# ---------------------------------------------------------------------------


class TestStructure:
    def test_store_source_parses(self):
        ast.parse(_read(NOTES_STORE_SRC))

    def test_store_source_is_ascii(self):
        raw = NOTES_STORE_SRC.read_bytes()
        assert all(b < 128 for b in raw)

    def test_this_suite_parses(self):
        ast.parse(_read(Path(__file__)))
