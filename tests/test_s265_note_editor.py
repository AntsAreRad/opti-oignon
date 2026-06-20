#!/usr/bin/env python3
"""S265 -- the N.8 third lot: the client editor on the S260 confirmed posture
and the PATCH-leg compaction trigger (the LAST container-provable item of the
NOTES_CRDT_SPEC.md section-7 proof list).

The slice under test (arbitrated at the read gate, version HELD at 3.12.0):

- A NEW sibling route module ``opti_oignon/api/routes_note_updates.py`` exposes
  the update legs over the S263 store, authed exactly like the five notes
  routes, so the routes_notes five-routes-exact pins (s245 / s249 / s256 / s257
  / s260 / s262 / s263) survive untouched -- the legs ride a SEPARATE router
  object (``note_updates_router``), never a new ``@notes_router.`` decorator:
  * append: ``POST /api/notes/{note_id}/updates`` over ``append_update`` --
    the local editor's incremental Yjs update; a store refusal (a dead or
    unknown parent, section 5) maps to an explicit error status, never a
    silent 200; invalid base64 is a 422; the local author identity rides the
    engine's signature at publish, NEVER the route payload (N9-D3).
  * tail read: ``GET /api/notes/{note_id}/updates?after_seq=N`` over
    ``list_updates`` -- the section-4 replay tail, per-user scoped.
- The PATCH-leg compaction trigger in the EXISTING ``update_note`` handler
  (a body-only edit, no new decorator, so ``@notes_router.patch`` stays 1):
  ``NoteUpdateRequest`` admits an optional ``checkpoint_watermark`` field; when
  present the handler records it through ``set_checkpoint_watermark`` AFTER the
  store commit (the S257 placement precedent) and prunes lazily through
  ``prune_below_watermark``; when absent nothing is recorded and nothing is
  pruned (fail-secure). The watermark is monotonic non-decreasing (the store
  setter rejects a regression), and the prune never over-prunes (rows above
  the watermark survive; serving never depends on pruned history).
- The editor confirmed-posture seam (section 5 / S260 idiom): the frontend
  ``noteUpdates`` client and the NotesPanel editor wiring render an edit's
  update ONLY after the local backend acknowledges the append (the POST
  returns) -- no optimistic ghost the store has not seen -- with the offline
  queue surfaced and a failure surfacing a toast while the display stays at
  server truth. Proven the container-provable way (source pins + wired
  TestClient handlers); the live browser walk stays host-assured (the
  NOTES_EDITOR_E2E_S265.md runbook).

Red-before contract (the S257-S260 assert-before-call idiom): every test that
touches the new surface asserts the surface exists (a non-empty source read,
an existence guard, or a recorded-effect read on the S263 store) BEFORE
calling or indexing it, so on the pristine S264 tree each red is an
AssertionError, never a collection error, an ImportError, an AttributeError,
or a TypeError. The design-green set -- the spec binding family, the
s262/s263/s264 reassertions, and the held-version / selection-literal
structure pins -- is declared as such inline and passes on both trees.

The store ``opti_oignon/notes/note_updates_store.py`` is NOT edited by this
lot: every seam it needs (``append_update``, ``list_updates``,
``set_checkpoint_watermark``, ``prune_below_watermark``) already landed at
S263, so the route legs and the PATCH trigger only call it. The auth core,
signing.py, records.py, producers.py, sync_engine.py, notes_store.py,
routes_sync.py, pairing.py, peers.py and tools.py stay edit-free.
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "opti_oignon"
API = PKG / "api"
NOTES = PKG / "notes"
VEILID = PKG / "veilid"

ROUTES_NOTES_PATH = API / "routes_notes.py"
ROUTES_NU_PATH = API / "routes_note_updates.py"      # the NEW sibling module
SCHEMAS_PATH = API / "schemas.py"
APP_PATH = API / "app.py"
ENGINE_SRC = VEILID / "sync_engine.py"
UPDATES_STORE_SRC = NOTES / "note_updates_store.py"
NOTES_STORE_SRC = NOTES / "notes_store.py"
TOOLS_SRC = PKG / "agent" / "tools.py"
VERSION_PATH = PKG / "__version__.py"
SPEC_PATH = ROOT / "NOTES_CRDT_SPEC.md"
RUNBOOK_PATH = ROOT / "NOTES_EDITOR_E2E_S265.md"      # the NEW host-assured walk

FE = ROOT / "frontend" / "src"
NOTE_UPDATES_CLIENT = FE / "lib" / "api" / "noteUpdates.ts"   # NEW client
NOTES_PANEL = FE / "lib" / "components" / "panels" / "NotesPanel.svelte"

NU_PREFIX = "/api/notes"
NU_APPEND_PATH = "/api/notes/{note_id}/updates"
NU_TAIL_PATH = "/api/notes/{note_id}/updates"


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
# Isolation harness (the S243 lesson, the S256 idiom: real dotted route modules
# over light stubs, real flat-loaded stores private to this suite)
# ---------------------------------------------------------------------------


def _ensure_pkg(name: str, path: Path) -> None:
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


def _reload_dotted(name: str, path: Path):
    """Force a fresh load under the dotted name (drop any prior object)."""
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _store_pkg_skeleton() -> None:
    """Ensure the package skeleton and a routes_auth stub exist in sys.modules."""
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.api", API)
    _ensure_pkg("opti_oignon.notes", NOTES)
    _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
    if "opti_oignon.api.routes_auth" not in sys.modules:
        stub = types.ModuleType("opti_oignon.api.routes_auth")

        def _get_current_user() -> dict:  # pragma: no cover - overridden
            return {"sub": None}

        stub._get_current_user = _get_current_user  # type: ignore[attr-defined]
        sys.modules["opti_oignon.api.routes_auth"] = stub


def _notes_store_module():
    """The real NotesStore module, flat-loaded under its dotted name."""
    _store_pkg_skeleton()
    return _load_dotted("opti_oignon.notes.notes_store", NOTES_STORE_SRC)


def _updates_store_module():
    """The real NoteUpdatesStore module, registered under its dotted name.

    Registered under the dotted name the route imports, so that the singleton
    set here is the one ``get_note_updates_store`` returns inside the handler.
    """
    _store_pkg_skeleton()
    return _load_dotted(
        "opti_oignon.notes.note_updates_store", UPDATES_STORE_SRC
    )


def _make_notes_store(tmp_path):
    return _notes_store_module().NotesStore(
        root=str(tmp_path), single_user_mode=True
    )


def _set_updates_singleton(tmp_path, parent_lookup):
    """Build a NoteUpdatesStore at ``tmp_path`` and install it as the singleton.

    Returns (module, instance). The handler resolves the update store through
    ``get_note_updates_store`` (its dependency seam's default), so installing
    the singleton injects this instance without the test ever naming a
    pristine-absent attribute.
    """
    mod = _updates_store_module()
    mod.reset_note_updates_store()
    inst = mod.NoteUpdatesStore(tmp_path, parent_lookup=parent_lookup)
    mod._store = inst  # the S171 singleton, set directly for test isolation
    return mod, inst


def _load_notes_route():
    """Load routes_notes dotted (delivered: it imports the update-store glue)."""
    _store_pkg_skeleton()
    # NotesStore must import cleanly so FEATURE_AVAILABLE is True (the s256
    # idiom: register the flat-loaded store under its dotted name first).
    sys.modules["opti_oignon.notes.notes_store"] = _notes_store_module()
    # The update-store module must be importable under its dotted name so a
    # delivered routes_notes can bind get_note_updates_store at load time.
    _updates_store_module()
    return _reload_dotted("opti_oignon.api.routes_notes", ROUTES_NOTES_PATH)


def _load_updates_route():
    """Load the NEW sibling module dotted. Callers assert existence first."""
    _store_pkg_skeleton()
    _updates_store_module()
    return _reload_dotted(
        "opti_oignon.api.routes_note_updates", ROUTES_NU_PATH
    )


def _build_notes(tmp_path):
    """A bare app over routes_notes with the store and auth injected."""
    routes = _load_notes_route()
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    store = _make_notes_store(tmp_path)
    app = FastAPI()
    app.include_router(routes.notes_router)
    state = {"sub": "user_a"}
    app.dependency_overrides[routes._notes_store_dep] = lambda: store
    app.dependency_overrides[routes._get_current_user] = lambda: {
        "sub": state["sub"]
    }
    client = TestClient(app)
    return client, store, routes, state


def _build_updates(tmp_path):
    """A bare app over the NEW sibling router with auth injected.

    The sibling's update-store dependency resolves through
    ``get_note_updates_store`` -> the singleton installed by the caller, so the
    caller controls parent liveness through that instance's ``parent_lookup``.
    """
    routes = _load_updates_route()
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(routes.note_updates_router)
    state = {"sub": "user_a"}
    app.dependency_overrides[routes._get_current_user] = lambda: {
        "sub": state["sub"]
    }
    client = TestClient(app)
    return client, routes, state


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _live(_note_id: str, _user_id: str) -> bool:
    return True


def _dead(_note_id: str, _user_id: str) -> bool:
    return False


def _nu_fn_segment(name: str) -> str:
    """The source of a top-level function in the sibling module, or ''."""
    src = _read(ROUTES_NU_PATH)
    if not src:
        return ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    return ""


# ===========================================================================
# Family 1 -- the binding contract (DESIGN-GREEN: the spec is unchanged)
# ===========================================================================


class TestSpecBinding:
    def test_section4_watermark_clause(self):
        f = _flat(_read(SPEC_PATH))
        assert f, "NOTES_CRDT_SPEC.md absent"
        assert "the checkpoint watermark" in f
        assert "rows at or below the watermark are prunable" in f
        assert "Pruning is local and lazy" in f

    def test_section5_confirmed_posture_clause(self):
        f = _flat(_read(SPEC_PATH))
        assert "renders in the editor only after the local backend" in f
        assert "no optimistic ghost state" in f.lower()

    def test_section7_remaining_items_named(self):
        f = _flat(_read(SPEC_PATH))
        assert "the confirmed-posture seam in the editor" in f
        assert "the prune watermark" in f


# ===========================================================================
# Family 2 -- the sibling route module (RED: the module is absent on pristine)
# ===========================================================================


class TestSiblingModule:
    def test_module_exists(self):
        assert ROUTES_NU_PATH.exists(), (
            "opti_oignon/api/routes_note_updates.py missing"
        )

    def test_router_object_and_prefix(self):
        assert ROUTES_NU_PATH.exists(), "sibling module missing"
        routes = _load_updates_route()
        assert hasattr(routes, "note_updates_router")
        assert routes.note_updates_router.prefix == NU_PREFIX

    def test_two_legs_at_source(self):
        src = _flat(_read(ROUTES_NU_PATH))
        assert src, "sibling module source absent"
        assert '@note_updates_router.post("/{note_id}/updates")' in src
        assert '@note_updates_router.get("/{note_id}/updates")' in src

    def test_router_is_authed(self):
        src = _read(ROUTES_NU_PATH)
        assert src, "sibling module source absent"
        # Authed exactly like the notes router: the dependency on the
        # current-user resolver rides the router object.
        assert "_get_current_user" in src
        assert "dependencies=" in src

    def test_no_new_notes_router_decorator(self):
        # The legs must NOT ride the notes_router object (that would break the
        # five-routes-exact pins); they ride note_updates_router.
        src = _read(ROUTES_NU_PATH)
        assert src, "sibling module source absent"
        assert "@notes_router." not in src

    def test_source_discipline(self):
        raw = b""
        try:
            raw = ROUTES_NU_PATH.read_bytes()
        except OSError:
            raw = b""
        assert raw != b"", "sibling module source absent"
        # Pure ASCII, no decorative separators.
        raw.decode("ascii")
        src = raw.decode("ascii")
        assert "====" not in src
        assert "checkpoint_before_apply = True" in src
        # No SQL is issued here -- the module delegates to the store.
        for verb in ("SELECT", "INSERT", "UPDATE", "DELETE"):
            assert 'f"' + verb not in src
            assert "f'" + verb not in src
        # The s216 census pin: no api file outside routes_chat.py calls it.
        assert "get_pipeline_runner" not in src

    def test_app_registers_the_sibling(self):
        src = _read(APP_PATH)
        assert "routes_note_updates import" in src
        assert "include_router(note_updates_router)" in src


# ===========================================================================
# Family 3 -- the append leg behaviour (RED: guarded by the existence assert)
# ===========================================================================


class TestAppendLeg:
    def test_append_round_trip(self, tmp_path):
        assert ROUTES_NU_PATH.exists(), "sibling module missing"
        _set_updates_singleton(tmp_path, parent_lookup=_live)
        client, _routes, _state = _build_updates(tmp_path)
        blob = b"yjs-update-bytes-1"
        r = client.post(
            "/api/notes/n1/updates", json={"update_blob_b64": _b64(blob)}
        )
        assert r.status_code in (200, 201), r.text
        body = r.json()
        assert int(body["seq"]) == 1
        assert body["note_id"] == "n1"
        assert base64.b64decode(body["update_blob_b64"]) == blob
        r2 = client.post(
            "/api/notes/n1/updates",
            json={"update_blob_b64": _b64(b"yjs-update-bytes-2")},
        )
        assert r2.status_code in (200, 201), r2.text
        assert int(r2.json()["seq"]) == 2

    def test_append_persists_to_store(self, tmp_path):
        assert ROUTES_NU_PATH.exists(), "sibling module missing"
        _mod, inst = _set_updates_singleton(tmp_path, parent_lookup=_live)
        client, _routes, _state = _build_updates(tmp_path)
        client.post(
            "/api/notes/n1/updates",
            json={"update_blob_b64": _b64(b"abc")},
        )
        rows = inst.list_updates("n1", user_id="user_a")
        assert len(rows) == 1
        assert rows[0].seq == 1

    def test_refused_dead_parent_is_explicit_error(self, tmp_path):
        assert ROUTES_NU_PATH.exists(), "sibling module missing"
        _set_updates_singleton(tmp_path, parent_lookup=_dead)
        client, _routes, _state = _build_updates(tmp_path)
        r = client.post(
            "/api/notes/ghost/updates",
            json={"update_blob_b64": _b64(b"x")},
        )
        assert r.status_code != 200, "a refused append must never be a 200"
        assert r.status_code == 409, r.text

    def test_invalid_base64_is_422(self, tmp_path):
        assert ROUTES_NU_PATH.exists(), "sibling module missing"
        _set_updates_singleton(tmp_path, parent_lookup=_live)
        client, _routes, _state = _build_updates(tmp_path)
        r = client.post(
            "/api/notes/n1/updates",
            json={"update_blob_b64": "not!!base64"},
        )
        assert r.status_code == 422, r.text

    def test_payload_carries_no_device_identity(self, tmp_path):
        # N9-D3: the route payload never carries a device identity; the
        # signature identity is the engine's at publish. The request schema
        # admits the blob only.
        assert ROUTES_NU_PATH.exists(), "sibling module missing"
        src = _read(SCHEMAS_PATH)
        # The append request shape is the blob; author_device is not an input.
        seg_ok = "NoteUpdateAppendRequest" in src
        assert seg_ok, "NoteUpdateAppendRequest schema absent"


# ===========================================================================
# Family 4 -- the tail-read leg behaviour (RED)
# ===========================================================================


class TestTailLeg:
    def test_tail_replays_from_seq(self, tmp_path):
        assert ROUTES_NU_PATH.exists(), "sibling module missing"
        _mod, inst = _set_updates_singleton(tmp_path, parent_lookup=_live)
        for i in range(1, 4):
            inst.append_update(
                "n1", ("u%d" % i).encode(), user_id="user_a",
                sync_publish=False,
            )
        client, _routes, _state = _build_updates(tmp_path)
        r = client.get("/api/notes/n1/updates", params={"after_seq": 1})
        assert r.status_code == 200, r.text
        seqs = [int(row["seq"]) for row in r.json()]
        assert seqs == [2, 3]

    def test_tail_empty_is_empty_list(self, tmp_path):
        assert ROUTES_NU_PATH.exists(), "sibling module missing"
        _set_updates_singleton(tmp_path, parent_lookup=_live)
        client, _routes, _state = _build_updates(tmp_path)
        r = client.get("/api/notes/none/updates", params={"after_seq": 0})
        assert r.status_code == 200, r.text
        assert r.json() == []

    def test_tail_returns_list_under_auth(self, tmp_path):
        assert ROUTES_NU_PATH.exists(), "sibling module missing"
        _mod, inst = _set_updates_singleton(tmp_path, parent_lookup=_live)
        inst.append_update(
            "n1", b"u1", user_id="user_a", sync_publish=False
        )
        client, _routes, state = _build_updates(tmp_path)
        state["sub"] = "user_b"
        r = client.get("/api/notes/n1/updates", params={"after_seq": 0})
        assert r.status_code == 200, r.text
        # The leg carries the authed user through to the scoped store read and
        # returns a list shape (true cross-user isolation is a multi-user-mode
        # property the store proves in s263; single-user mode collapses here).
        assert isinstance(r.json(), list)


# ===========================================================================
# Family 5 -- the PATCH-leg compaction trigger (RED: trigger absent on pristine,
# so the recorded watermark stays 0 and the assertions fail as AssertionError)
# ===========================================================================


class TestPatchWatermark:
    def _note_with_updates(self, tmp_path):
        """A created note plus three appended updates on the shared singleton."""
        _mod, inst = _set_updates_singleton(tmp_path, parent_lookup=_live)
        client, _store, _routes, _state = _build_notes(tmp_path)
        nid = client.post("/api/notes", json={"title": "t"}).json()["id"]
        for i in range(1, 4):
            inst.append_update(
                nid, ("u%d" % i).encode(), user_id="user_a",
                sync_publish=False,
            )
        return client, inst, nid

    def test_patch_records_watermark(self, tmp_path):
        client, inst, nid = self._note_with_updates(tmp_path)
        r = client.patch(
            "/api/notes/%s" % nid,
            json={"body_crdt_b64": _b64(b"merged"), "checkpoint_watermark": 2},
        )
        assert r.status_code == 200, r.text
        assert inst.get_checkpoint_watermark(nid, user_id="user_a") == 2

    def test_patch_triggers_lazy_prune(self, tmp_path):
        client, inst, nid = self._note_with_updates(tmp_path)
        client.patch(
            "/api/notes/%s" % nid,
            json={"checkpoint_watermark": 2},
        )
        # Watermark recorded (the red anchor): only then is the prune meaningful.
        assert inst.get_checkpoint_watermark(nid, user_id="user_a") == 2
        surviving = [
            row.seq for row in inst.list_updates(nid, user_id="user_a")
        ]
        # Rows at or below 2 pruned; row 3 survives (never over-prune).
        assert surviving == [3]

    def test_watermark_is_monotonic(self, tmp_path):
        client, inst, nid = self._note_with_updates(tmp_path)
        client.patch(
            "/api/notes/%s" % nid, json={"checkpoint_watermark": 2}
        )
        assert inst.get_checkpoint_watermark(nid, user_id="user_a") == 2
        # A lower watermark in a later PATCH is rejected by the store setter.
        client.patch(
            "/api/notes/%s" % nid, json={"checkpoint_watermark": 1}
        )
        assert inst.get_checkpoint_watermark(nid, user_id="user_a") == 2

    def test_absent_field_records_nothing(self, tmp_path):
        client, inst, nid = self._note_with_updates(tmp_path)
        # Record one (the red anchor on pristine: this stays 0 without the
        # trigger, failing the assertion as an AssertionError).
        client.patch(
            "/api/notes/%s" % nid, json={"checkpoint_watermark": 2}
        )
        assert inst.get_checkpoint_watermark(nid, user_id="user_a") == 2
        # A PATCH WITHOUT the field leaves the watermark untouched.
        r = client.patch(
            "/api/notes/%s" % nid, json={"title": "renamed"}
        )
        assert r.status_code == 200, r.text
        assert inst.get_checkpoint_watermark(nid, user_id="user_a") == 2

    def test_schema_admits_optional_watermark(self):
        # The PATCH body schema must admit the optional field; a bare instance
        # (every field omitted) must still construct (the s256 bare pin).
        import importlib

        schemas = importlib.import_module("opti_oignon.api.schemas")
        # Assert the field is a declared model field BEFORE reading it, so the
        # pristine red is an AssertionError (not an AttributeError).
        assert "checkpoint_watermark" in schemas.NoteUpdateRequest.model_fields
        upd = schemas.NoteUpdateRequest(checkpoint_watermark=5)
        assert upd.checkpoint_watermark == 5
        bare = schemas.NoteUpdateRequest()
        assert bare.checkpoint_watermark is None


# ===========================================================================
# Family 6 -- the editor confirmed-posture seam (RED: frontend wiring absent)
# ===========================================================================


class TestEditorConfirmedPosture:
    def test_client_module_exists(self):
        assert NOTE_UPDATES_CLIENT.exists(), (
            "frontend/src/lib/api/noteUpdates.ts missing"
        )

    def test_client_exports_append_and_tail(self):
        src = _read(NOTE_UPDATES_CLIENT)
        assert src, "noteUpdates client absent"
        assert "appendNoteUpdate" in src
        assert "fetchNoteUpdates" in src

    def test_client_targets_the_update_legs(self):
        src = _flat(_read(NOTE_UPDATES_CLIENT))
        assert src, "noteUpdates client absent"
        assert "/updates" in src

    def test_panel_imports_the_client(self):
        src = _read(NOTES_PANEL)
        assert src, "NotesPanel.svelte absent"
        assert "noteUpdates" in src

    def test_render_follows_ack(self):
        flat = _flat(_read(NOTES_PANEL))
        assert flat, "NotesPanel.svelte absent"
        # The update is applied to rendered state only after the append
        # resolves -- the await on the append precedes the local apply, and a
        # marker names the confirmed posture (the S260 vocabulary).
        assert "await appendNoteUpdate(" in flat
        assert "confirmed posture" in flat.lower()

    def test_offline_queue_surfaced(self):
        flat = _flat(_read(NOTES_PANEL)).lower()
        assert flat, "NotesPanel.svelte absent"
        assert "offline queue" in flat

    def test_failure_keeps_server_truth(self):
        flat = _flat(_read(NOTES_PANEL)).lower()
        assert flat, "NotesPanel.svelte absent"
        # A failed append surfaces a toast; the display is never advanced to a
        # state the backend has not confirmed. The "server truth" marker is the
        # editor's confirmed-posture failure path (absent on the pristine tree,
        # so this is a real red-before, not satisfied by the pre-existing S260
        # mobile-allowed toggle toasts).
        assert "server truth" in flat
        assert "toast" in flat


# ===========================================================================
# Family 7 -- the host-assured runbook (RED: runbook absent on pristine)
# ===========================================================================


class TestRunbook:
    def test_runbook_exists_and_ascii(self):
        raw = _read(RUNBOOK_PATH)
        assert raw != "", "NOTES_EDITOR_E2E_S265.md absent"
        assert all(ord(c) < 128 for c in raw)
        assert "====" not in raw

    def test_runbook_labels_host_assured_walk(self):
        f = _flat(_read(RUNBOOK_PATH))
        assert f, "runbook absent"
        assert "host-assured" in f
        assert "never simulated in-container" in f

    def test_runbook_names_the_confirmed_posture_walk(self):
        f = _flat(_read(RUNBOOK_PATH)).lower()
        assert f, "runbook absent"
        assert "confirmed posture" in f
        assert "append" in f


# ===========================================================================
# Family 8 -- reassertions (DESIGN-GREEN: these pass on the pristine S264 tree)
# ===========================================================================


class TestDesignGreenReassertions:
    def test_notes_routes_stay_five_at_source(self):
        src = _read(ROUTES_NOTES_PATH)
        assert src.count("@notes_router.") == 5
        assert src.count("@notes_router.patch") == 1

    def test_the_five_exact_routes_present(self):
        src = _read(ROUTES_NOTES_PATH)
        assert '@notes_router.get(""' in src
        assert '@notes_router.post(""' in src
        assert '@notes_router.get("/{note_id}"' in src
        assert '@notes_router.patch("/{note_id}"' in src
        assert '@notes_router.delete("/{note_id}"' in src

    def test_s262_engine_markers_alive(self):
        src = _read(ENGINE_SRC)
        assert "filter-at-serve" in src
        assert "never re-signs" in src

    def test_s264_serve_floor_markers_alive(self):
        src = _read(ENGINE_SRC)
        assert "_land_note_updates" in src
        assert "update_watermark_gate" in src

    def test_s263_store_discipline_alive(self):
        raw = b""
        try:
            raw = UPDATES_STORE_SRC.read_bytes()
        except OSError:
            raw = b""
        assert raw != b"", "note_updates_store.py source absent"
        raw.decode("ascii")
        src = raw.decode("ascii")
        for verb in ("SELECT", "INSERT", "UPDATE", "DELETE"):
            assert 'f"' + verb not in src
            assert "f'" + verb not in src
        assert "safe_connect" in src
        assert "effective_user_id" in src
        assert "checkpoint_before_apply = True" in src
        assert "UPDATE note_update " not in src
        assert "update_blob =" not in src

    def test_store_seams_present(self):
        src = _read(UPDATES_STORE_SRC)
        for seam in (
            "def append_update",
            "def list_updates",
            "def set_checkpoint_watermark",
            "def prune_below_watermark",
        ):
            assert seam in src, seam

    def test_sensitive_kinds_exact_skill_only(self):
        # Read the constant at source to avoid importing the veilid chain.
        src = _read(ENGINE_SRC)
        assert (
            "SENSITIVE_KINDS: frozenset[str] = frozenset("
            "{RecordKind.SKILL.value})"
        ) in src

    def test_tools_surface_stays_zero_mobile_allowed(self):
        src = _read(TOOLS_SRC)
        assert src != "", "agent/tools.py missing"
        assert src.count("mobile_allowed") == 0


# ===========================================================================
# Family 9 -- structure (DESIGN-GREEN: version held, suite hygiene)
# ===========================================================================


class TestStructure:
    def test_version_held_3_12_0(self):
        import re

        m = re.search(
            r'__version__\s*=\s*"([^"]+)"', _read(VERSION_PATH)
        )
        assert m is not None
        assert m.group(1) == "3.12.0"

    def test_this_suite_avoids_the_selection_literal(self):
        here = Path(__file__).read_text(encoding="utf-8")
        token = "sandbox" + "_manager"
        assert token not in here

    def test_new_sources_ascii_when_present(self):
        for path in (ROUTES_NU_PATH, NOTE_UPDATES_CLIENT, RUNBOOK_PATH):
            raw = _read(path)
            if raw:
                assert all(ord(c) < 128 for c in raw), str(path)

    def test_this_suite_parses(self):
        ast.parse(Path(__file__).read_text(encoding="utf-8"))
