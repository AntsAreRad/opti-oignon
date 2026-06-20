"""S249 -- the shared notes-attachment route (the media blocs' container-provable
backend prerequisite): a FastAPI router exposing the N.1 ``attachment`` manifest
and the two-layer ``NotesBlobStore`` over HTTP.

S243 landed N.1 (``opti_oignon/notes/``: the ``NotesStore`` with the ``attachment``
manifest table already carrying ``transcript_text`` / ``caption_text`` /
``ocr_text``, the ``{audio, image, drawing}`` kind allowlist, and the two-layer
per-attachment AES-256-GCM ``NotesBlobStore``). S244 landed N.4 (``manage_notes``);
S245 landed N.2's route (``routes_notes.py``); S246 landed N.3 (``note_actions``);
S247 landed N.3's route (``routes_note_actions.py``); S248 landed the N.2 / N.3
SvelteKit UI. The read gate found that N.1 already built the full media data layer
for all three kinds, so the only missing piece to open the N.5 / N.6 / N.7 media
front is the shared HTTP surface that moves the encrypted blobs and the manifest
rows -- not any one kind's data layer.

This lot adds ``routes_notes_attachments.py``: a SEPARATE router
(``notes_attachments_router`` at ``/api/notes/attachments``), NOT folded into the
S245 ``notes_router`` -- so the S245 ``test_five_routes_exact`` pin stays green
(the ``routes_note_actions`` precedent) and this is a pure chain addition. It
exposes upload (multipart, sealed via ``NotesBlobStore`` under a per-attachment
HKDF-domain subkey, the manifest row via ``add_attachment``, ``kind`` allowlisted),
list-by-note, metadata, download (decrypted in memory, never a plaintext temp),
and delete (blob plus manifest). Per-user via the existing auth dependency; the
availability guard is a 503 mirroring ``routes_notes._check_store``.

The one data-layer extension this route needs is ``NotesStore.delete_attachment``
(additive: removes the manifest row, parameterized, per-user). The transcript /
caption / OCR write-back (``update_attachment``) and the opt-in, sandboxed
whisper.cpp / vision post-processing are LATER blocs (N.5 / N.6 transcription /
OCR), as are the in-browser capture / gallery / canvas UIs; this lot is the
shared backend they all ride.

Six families, the S247 idiom:

 1. Source / structure -- the module exists, ``checkpoint_before_apply = True``,
    ``FEATURE_AVAILABLE``, the ``/api/notes/attachments`` prefix, the five
    (path, method) routes on ``notes_attachments_router``, the route seals via
    the blob store and never writes plaintext (uses the blob store, no plaintext
    temp), no f-string SQL and no direct ``.execute`` (it delegates to the store),
    the not-a-tool property, AST + pure ASCII.
 2. Registration -- ``app.py`` imports and includes ``notes_attachments_router``.
 3. Schemas -- ``AttachmentSchema`` / ``AttachmentDeleteResponse`` in
    ``schemas.py``; they load and validate a sample.
 4. Behavioural (TestClient, injected real NotesStore on tmp + NotesBlobStore with
    an injected master key) -- upload seals ciphertext and returns the manifest;
    download round-trips the exact bytes; list returns the note's attachments;
    metadata fetch; per-user isolation (a second user cannot read or delete the
    first's attachment -> 404); a bad kind is a 422 (not a 500) and leaves no
    orphan blob; an upload to a missing note is a 404; delete removes both the
    blob and the manifest, and a second delete is a 404; the no-master-key path
    is a clean 503 (NotesBlobUnavailable), never a plaintext write.
 5. Premise guards (green before AND after) -- the N.1 attachment data layer is
    intact (``add_attachment`` / ``get_attachment`` / ``list_attachments``, the
    ``attachment`` table, ``ATTACHMENT_KINDS``); the ``NotesBlobStore`` surface is
    intact (``seal`` / ``open`` / ``delete`` / ``NotesBlobUnavailable``); and the
    S245 ``notes_router`` still exposes EXACTLY its five routes (the attachment
    endpoints are on a separate router -- a future fold-in turns this red).
 6. AST / ASCII of the new route and of this suite.

Red-before on the pristine S248 tree (no ``routes_notes_attachments.py``, no
attachment schemas, no app registration, no ``delete_attachment``): every
family-1 / 2 / 3 contract pin and the family-4 behavioural pins FAIL -- the read
helpers return empty strings so absence is a failure, and the behavioural family
loads the route INSIDE the test (so absence is an ImportError failure, never a
collection error). The ``delete_attachment`` existence pin (family 1) is red until
the method lands. Every family-5 premise guard, the family-6 suite-parse pin, and
the family-1 negative invariants (vacuous on the absent module) PASS by design.

Isolation (the S243 / S245 / S246 / S247 lesson): the behavioural family loads the
route under its dotted name into package-like stubs, pre-loading the real (light)
schemas dotted and the real ``notes_store`` / ``blob_store`` dotted (their guarded
relative imports resolve as submodules of the stub package, with the REAL
AES-256-GCM primitive, not a stub), and stubbing ``routes_auth`` (the auth dep is
overridden per test anyway). No fastapi/ollama package import is forced at
collection; ollama is never invoked (no model path here).
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path

# Defensive: never pull the real ollama during collection.
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "opti_oignon"
ROUTE_PATH = PKG / "api" / "routes_notes_attachments.py"
SCHEMAS_PATH = PKG / "api" / "schemas.py"
APP_PATH = PKG / "api" / "app.py"
NOTES_STORE_PATH = PKG / "notes" / "notes_store.py"
BLOB_STORE_PATH = PKG / "notes" / "blob_store.py"
ROUTES_NOTES_PATH = PKG / "api" / "routes_notes.py"

EXPECTED_PREFIX = "/api/notes/attachments"
# The five (path, method) routes the surface exposes.
EXPECTED_ROUTES = frozenset(
    {
        ("/api/notes/attachments/note/{note_id}", "POST"),
        ("/api/notes/attachments/note/{note_id}", "GET"),
        ("/api/notes/attachments/{attachment_id}", "GET"),
        ("/api/notes/attachments/{attachment_id}/blob", "GET"),
        ("/api/notes/attachments/{attachment_id}", "DELETE"),
    }
)
# The S245 notes_router's five routes (the fold-in guard).
NOTES_ROUTER_ROUTES = frozenset(
    {
        ("/api/notes", "GET"),
        ("/api/notes", "POST"),
        ("/api/notes/{note_id}", "GET"),
        ("/api/notes/{note_id}", "PATCH"),
        ("/api/notes/{note_id}", "DELETE"),
    }
)


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Isolation harness (the S243 / S247 idiom)
# ---------------------------------------------------------------------------


def _ensure_pkg(name: str, path: Path) -> None:
    """Ensure ``name`` is in sys.modules and package-like (has __path__).

    Non-destructive: keeps any pre-existing stub object (an earlier suite's),
    only granting it a ``__path__`` so a dotted ``spec_from_file_location`` load
    of a submodule resolves without executing ``opti_oignon/__init__.py``.
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


def _notes_modules():
    """The real notes_store / blob_store, dotted (relative imports resolved)."""
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.notes", PKG / "notes")
    ns = _load_dotted("opti_oignon.notes.notes_store", NOTES_STORE_PATH)
    bs = _load_dotted("opti_oignon.notes.blob_store", BLOB_STORE_PATH)
    return ns, bs


def _load_route():
    """Load routes_notes_attachments under its dotted name into package stubs.

    Pre-loads the (light) schemas dotted and the real notes_store / blob_store
    dotted (so the route's absolute imports resolve to the real modules), and
    stubs routes_auth (the auth dep is overridden per test). On the pristine tree
    the route module is absent and this raises ImportError INSIDE the calling
    test -- a failure, never a collection error.
    """
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.api", PKG / "api")
    _ensure_pkg("opti_oignon.notes", PKG / "notes")
    _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
    _notes_modules()
    if "opti_oignon.api.routes_auth" not in sys.modules:
        stub = types.ModuleType("opti_oignon.api.routes_auth")

        def _get_current_user() -> dict:  # pragma: no cover - overridden in tests
            return {"sub": None}

        stub._get_current_user = _get_current_user  # type: ignore[attr-defined]
        sys.modules["opti_oignon.api.routes_auth"] = stub
    return _load_dotted("opti_oignon.api.routes_notes_attachments", ROUTE_PATH)


def _routes_of(router) -> set:
    out = set()
    for r in getattr(router, "routes", []):
        methods = getattr(r, "methods", None) or set()
        path = getattr(r, "path", "")
        for m in methods:
            if m in {"GET", "POST", "PATCH", "DELETE", "PUT"}:
                out.add((path, m))
    return out


def _build(tmp_path, *, sub: str = "alice", master=None):
    """Build a bare app over the route with a real store, blob store, and auth
    injected. Returns (client_app, routes, store, blobs, state)."""
    routes = _load_route()
    ns = sys.modules["opti_oignon.notes.notes_store"]
    bs = sys.modules["opti_oignon.notes.blob_store"]
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    store = ns.NotesStore(root=tmp_path, single_user_mode=False)
    if master is None:
        master = os.urandom(32)
    blobs = bs.NotesBlobStore(root=tmp_path, master_key=master) if master else (
        bs.NotesBlobStore(root=tmp_path)
    )
    state = {"sub": sub}
    app = FastAPI()
    app.include_router(routes.notes_attachments_router)
    app.dependency_overrides[routes._notes_store_dep] = lambda: store
    app.dependency_overrides[routes._blob_store_dep] = lambda: blobs
    app.dependency_overrides[routes._get_current_user] = lambda: {"sub": state["sub"]}
    return TestClient(app), routes, store, blobs, state


# ---------------------------------------------------------------------------
# Family 1 -- source / structure
# ---------------------------------------------------------------------------


class TestRouteSource:
    def test_module_exists(self):
        assert ROUTE_PATH.exists(), "opti_oignon/api/routes_notes_attachments.py missing"

    def test_checkpoint_before_apply_hardcoded(self):
        assert "checkpoint_before_apply = True" in _read(ROUTE_PATH)

    def test_feature_available_flag(self):
        assert "FEATURE_AVAILABLE" in _read(ROUTE_PATH)

    def test_api_notes_attachments_prefix(self):
        assert EXPECTED_PREFIX in _read(ROUTE_PATH)

    def test_route_uses_blob_store_and_store(self):
        src = _read(ROUTE_PATH)
        assert "NotesBlobStore" in src or "get_notes_blob_store" in src
        assert "get_notes_store" in src
        assert ".seal(" in src
        assert ".open(" in src

    def test_route_no_direct_sql(self):
        src = _read(ROUTE_PATH)
        assert "import sqlite3" not in src
        assert ".execute(" not in src

    def test_route_no_fstring_sql(self):
        # No f-string carrying an SQL verb (the house no-f-string-SQL discipline).
        src = _read(ROUTE_PATH)
        for verb in ("SELECT", "INSERT", "UPDATE", "DELETE FROM"):
            assert f'f"{verb}' not in src and f"f'{verb}" not in src, verb

    def test_route_is_not_a_model_tool(self):
        src = _read(ROUTE_PATH)
        assert "ToolSchema(" not in src
        assert "register_tool" not in src

    def test_delete_attachment_store_method_present(self):
        # The one additive data-layer extension this route needs (red before).
        src = _read(NOTES_STORE_PATH)
        assert "def delete_attachment(" in src

    def test_pure_ascii_no_decoration(self):
        raw = _read(ROUTE_PATH)
        assert raw != ""
        assert all(ord(c) < 128 for c in raw)
        assert "====" not in raw


class TestRouteShape:
    def test_router_prefix_runtime(self):
        routes = _load_route()
        assert routes.notes_attachments_router.prefix == EXPECTED_PREFIX

    def test_five_routes_exact(self):
        routes = _load_route()
        assert _routes_of(routes.notes_attachments_router) == EXPECTED_ROUTES

    def test_seams_present(self):
        routes = _load_route()
        assert hasattr(routes, "_notes_store_dep")
        assert hasattr(routes, "_blob_store_dep")
        assert hasattr(routes, "_get_current_user")
        assert hasattr(routes, "_check")


# ---------------------------------------------------------------------------
# Family 2 -- registration
# ---------------------------------------------------------------------------


class TestAppRegistration:
    def test_app_imports_attachments_router(self):
        src = _read(APP_PATH)
        assert "routes_notes_attachments import" in src
        assert "notes_attachments_router" in src

    def test_app_includes_attachments_router(self):
        src = _read(APP_PATH)
        assert "include_router(notes_attachments_router)" in src


# ---------------------------------------------------------------------------
# Family 3 -- schemas
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_schema_symbols_in_source(self):
        src = _read(SCHEMAS_PATH)
        assert "class AttachmentSchema" in src
        assert "class AttachmentDeleteResponse" in src

    def test_schemas_load_and_validate(self):
        schemas = _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
        att = schemas.AttachmentSchema(
            id="a1", note_id="n1", kind="audio", mime="audio/webm", byte_size=12
        )
        assert att.id == "a1"
        assert att.kind == "audio"
        assert att.transcript_text is None
        assert att.caption_text is None
        assert att.ocr_text is None
        dele = schemas.AttachmentDeleteResponse(deleted=True, id="a1")
        assert dele.deleted is True
        assert dele.id == "a1"


# ---------------------------------------------------------------------------
# Family 4 -- behavioural (TestClient, injected real store + blob store)
# ---------------------------------------------------------------------------


def _make_note(store, *, user_id="alice", title="N"):
    return store.add_note(title, body_crdt=b"", tags=None, pinned=False, user_id=user_id)


class TestBehaviour:
    def test_upload_seals_and_returns_manifest(self, tmp_path):
        client_app, _routes, store, blobs, _state = _build(tmp_path)
        note = _make_note(store)
        payload = b"the-audio-bytes-1234"
        r = client_app.post(
            f"/api/notes/attachments/note/{note.id}",
            files={"file": ("clip.webm", payload, "audio/webm")},
            data={"kind": "audio"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["note_id"] == note.id
        assert data["kind"] == "audio"
        assert data["byte_size"] == len(payload)
        assert data["mime"] == "audio/webm"
        assert data["transcript_text"] is None
        # The manifest row exists and the blob is sealed (ciphertext on disk).
        att_id = data["id"]
        assert store.get_attachment(att_id, user_id="alice") is not None
        assert blobs.exists(att_id)
        on_disk = blobs._blob_path(att_id).read_bytes()
        assert payload not in on_disk

    def test_download_round_trips_exact_bytes(self, tmp_path):
        client_app, _routes, store, _blobs, _state = _build(tmp_path)
        note = _make_note(store)
        payload = b"\x00\x01binary\xffpayload"
        up = client_app.post(
            f"/api/notes/attachments/note/{note.id}",
            files={"file": ("x.bin", payload, "application/octet-stream")},
            data={"kind": "image"},
        )
        att_id = up.json()["id"]
        r = client_app.get(f"/api/notes/attachments/{att_id}/blob")
        assert r.status_code == 200, r.text
        assert r.content == payload

    def test_list_attachments_for_note(self, tmp_path):
        client_app, _routes, store, _blobs, _state = _build(tmp_path)
        note = _make_note(store)
        for i in range(3):
            client_app.post(
                f"/api/notes/attachments/note/{note.id}",
                files={"file": (f"d{i}.svg", b"<svg/>", "image/svg+xml")},
                data={"kind": "drawing"},
            )
        r = client_app.get(f"/api/notes/attachments/note/{note.id}")
        assert r.status_code == 200, r.text
        items = r.json()
        assert len(items) == 3
        assert all(it["kind"] == "drawing" for it in items)

    def test_metadata_fetch(self, tmp_path):
        client_app, _routes, store, _blobs, _state = _build(tmp_path)
        note = _make_note(store)
        up = client_app.post(
            f"/api/notes/attachments/note/{note.id}",
            files={"file": ("a.webm", b"snd", "audio/webm")},
            data={"kind": "audio"},
        )
        att_id = up.json()["id"]
        r = client_app.get(f"/api/notes/attachments/{att_id}")
        assert r.status_code == 200, r.text
        assert r.json()["id"] == att_id

    def test_per_user_isolation(self, tmp_path):
        client_app, _routes, store, _blobs, state = _build(tmp_path, sub="alice")
        note = _make_note(store, user_id="alice")
        up = client_app.post(
            f"/api/notes/attachments/note/{note.id}",
            files={"file": ("a.webm", b"snd", "audio/webm")},
            data={"kind": "audio"},
        )
        att_id = up.json()["id"]
        # Switch the active user to bob; alice's attachment is invisible.
        state["sub"] = "bob"
        assert client_app.get(f"/api/notes/attachments/{att_id}").status_code == 404
        assert client_app.get(f"/api/notes/attachments/{att_id}/blob").status_code == 404
        assert client_app.delete(f"/api/notes/attachments/{att_id}").status_code == 404

    def test_bad_kind_is_422_and_no_orphan_blob(self, tmp_path):
        client_app, _routes, store, blobs, _state = _build(tmp_path)
        note = _make_note(store)
        before = list(blobs.blob_dir.iterdir())
        r = client_app.post(
            f"/api/notes/attachments/note/{note.id}",
            files={"file": ("v.mp4", b"video", "video/mp4")},
            data={"kind": "video"},
        )
        assert r.status_code == 422, r.text
        # No orphan blob was sealed for the rejected kind.
        after = list(blobs.blob_dir.iterdir())
        assert after == before

    def test_upload_to_missing_note_is_404(self, tmp_path):
        client_app, _routes, _store, _blobs, _state = _build(tmp_path)
        r = client_app.post(
            "/api/notes/attachments/note/does-not-exist",
            files={"file": ("a.webm", b"snd", "audio/webm")},
            data={"kind": "audio"},
        )
        assert r.status_code == 404, r.text

    def test_delete_removes_blob_and_manifest(self, tmp_path):
        client_app, _routes, store, blobs, _state = _build(tmp_path)
        note = _make_note(store)
        up = client_app.post(
            f"/api/notes/attachments/note/{note.id}",
            files={"file": ("a.webm", b"snd", "audio/webm")},
            data={"kind": "audio"},
        )
        att_id = up.json()["id"]
        assert blobs.exists(att_id)
        r = client_app.delete(f"/api/notes/attachments/{att_id}")
        assert r.status_code == 200, r.text
        assert r.json()["deleted"] is True
        assert not blobs.exists(att_id)
        assert store.get_attachment(att_id, user_id="alice") is None
        # A second delete is a 404 (already gone).
        assert client_app.delete(f"/api/notes/attachments/{att_id}").status_code == 404

    def test_no_master_key_is_clean_503_not_plaintext(self, tmp_path):
        # A blob store with no master key must refuse (NotesBlobUnavailable) and
        # the route must surface a 503, never persist a plaintext blob.
        client_app, _routes, store, blobs, _state = _build(tmp_path, master=False)
        note = _make_note(store)
        r = client_app.post(
            f"/api/notes/attachments/note/{note.id}",
            files={"file": ("a.webm", b"snd", "audio/webm")},
            data={"kind": "audio"},
        )
        assert r.status_code == 503, r.text
        # Nothing was sealed.
        assert list(blobs.blob_dir.iterdir()) == []

    def test_check_guard_is_503(self):
        routes = _load_route()
        from fastapi import HTTPException

        original = routes.FEATURE_AVAILABLE
        try:
            routes.FEATURE_AVAILABLE = False
            with pytest.raises(HTTPException) as exc:
                routes._check()
            assert exc.value.status_code == 503
        finally:
            routes.FEATURE_AVAILABLE = original


# ---------------------------------------------------------------------------
# Family 5 -- premise guards (green before AND after)
# ---------------------------------------------------------------------------


class TestPremiseGuards:
    def test_n1_attachment_data_layer_intact(self):
        src = _read(NOTES_STORE_PATH)
        assert "def add_attachment(" in src
        assert "def get_attachment(" in src
        assert "def list_attachments(" in src
        assert "CREATE TABLE IF NOT EXISTS attachment" in src
        assert "ATTACHMENT_KINDS" in src

    def test_blob_store_surface_intact(self):
        src = _read(BLOB_STORE_PATH)
        assert "class NotesBlobStore" in src
        assert "def seal(" in src
        assert "def open(" in src
        assert "def delete(" in src
        assert "class NotesBlobUnavailable" in src

    def test_notes_router_still_exactly_five_routes(self):
        # The attachment endpoints live on a SEPARATE router; folding them into
        # notes_router would grow this set and turn this guard red.
        _ensure_pkg("opti_oignon", PKG)
        _ensure_pkg("opti_oignon.api", PKG / "api")
        _ensure_pkg("opti_oignon.notes", PKG / "notes")
        _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
        _notes_modules()
        if "opti_oignon.api.routes_auth" not in sys.modules:
            stub = types.ModuleType("opti_oignon.api.routes_auth")
            stub._get_current_user = lambda: {"sub": None}  # type: ignore[attr-defined]
            sys.modules["opti_oignon.api.routes_auth"] = stub
        rn = _load_dotted("opti_oignon.api.routes_notes", ROUTES_NOTES_PATH)
        assert _routes_of(rn.notes_router) == NOTES_ROUTER_ROUTES


# ---------------------------------------------------------------------------
# Family 6 -- AST / ASCII
# ---------------------------------------------------------------------------


class TestASTValid:
    def test_route_source_parses(self):
        src = _read(ROUTE_PATH)
        assert src != ""
        ast.parse(src, filename=str(ROUTE_PATH))

    def test_this_suite_parses(self):
        src = _read(Path(__file__))
        assert src != ""
        ast.parse(src, filename=__file__)
