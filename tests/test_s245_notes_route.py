"""S245 -- N.2 (re-arbitrated to the container-provable half): the FastAPI notes
route binding the N.1 NotesStore to HTTP.

S243 landed N.1 (the ``opti_oignon/notes/`` data layer); S244 landed N.4 (the
gated ``manage_notes`` tool, the LLM-from-chat write surface). This lot adds the
HTTP surface the SvelteKit notes client (N.2 proper) will ride: a
``routes_notes.py`` binding the ``NotesStore`` -- list / get / create / update /
delete -- per-user via the existing auth dependency, registered on the app
exactly like the per-user ``memories_router``.

Read-gate arbitration (S245): N.2's SvelteKit UI is not pytest-provable, so the
container-provable half -- the FastAPI route, the natural prerequisite for the
UI -- is this lot; the UI itself is a labelled Playwright/runbook lot for a later
session. The route is the user's own manual surface and is NOT route-level
mode-gated (it mirrors ``routes_memory`` exactly: ``manage_memory`` is a
Bulbe-forbidden tool, yet ``routes_memory`` carries no mode gate; the user
creates/edits notes manually in both modes per NOTES_FEATURE_ROADMAP). The Bulbe
restriction on notes lives at the ``manage_notes`` tool layer, not this HTTP
surface.

The body is an opaque, client-owned CRDT, so the route never interprets it: the
body crosses the wire base64-encoded and is stored as opaque bytes. ``tags``
cross as a JSON array (``list[str]``) and are stored as the opaque JSON-array
string the store and the ``manage_notes`` tool already use. Cross-device CRDT
relay is N.8; this route persists whole note state (the editor's save), it does
not merge.

Six families:

 1. Source / structure -- ``routes_notes.py`` exists, ``checkpoint_before_apply
    = True``, ``FEATURE_AVAILABLE``, the ``/api/notes`` prefix, the five
    (path, method) routes on ``notes_router``, the route delegates (no
    ``sqlite3`` / no ``.execute`` in the route), AST + pure ASCII.
 2. Registration -- ``app.py`` imports ``notes_router`` and includes it
    (mirroring ``memories_router``).
 3. Schemas -- ``NoteSchema`` / ``NoteCreateRequest`` / ``NoteUpdateRequest`` in
    ``schemas.py``; the models load and validate a sample.
 4. Behavioural (TestClient, injected store) -- create returns the note with the
    body base64 round-tripping and tags round-tripping; list includes it; get
    returns it and a missing id is 404; update mutates title / tags / pinned /
    body and a missing id is 404; delete tombstones (404 thereafter, excluded
    from the default list) and a missing id is 404; per-user isolation (user_b
    never sees user_a's note); the ``_check_store`` guard is a 503.
 5. Premise guards -- green before and after: the N.1 ``NotesStore`` module
    loads, ``routes_memory`` is the template (``memories_router`` +
    ``_get_current_user``), the N.4 ``manage_notes`` tool is intact,
    ``schemas.py`` exists.
 6. AST / ASCII of the new route and of this suite.

Red-before: on the pristine S244 tree (no ``routes_notes.py``, no notes schemas,
no app registration) every family-1/2/3/4 contract pin FAILS -- the read helpers
return empty strings so absence is a failure, and the behavioural family loads
the route INSIDE the test (so absence is an ImportError failure, never a
collection error) -- while every family-5 premise guard and the family-6 "this
suite parses" pin PASS by design.

Isolation (the S243 lesson): the behavioural family loads the route under its
dotted name into package-like stubs, pre-loading the real (light) schemas dotted,
stubbing ``routes_auth`` so the auth chain never fires (the auth dep is overridden
per test anyway), and loading the real ``notes_store`` under a FLAT body
registered at the dotted key so the route's absolute import resolves to a store
whose guarded relative imports fall back to plaintext sqlite (the documented
in-container posture). No fastapi/ollama package import is forced at collection.
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import json
import sys
import types
from pathlib import Path

# Defensive: never pull the real ollama during collection.
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "opti_oignon"
ROUTES_NOTES_PATH = PKG / "api" / "routes_notes.py"
SCHEMAS_PATH = PKG / "api" / "schemas.py"
APP_PATH = PKG / "api" / "app.py"
ROUTES_MEMORY_PATH = PKG / "api" / "routes_memory.py"
TOOLS_PATH = PKG / "agent" / "tools.py"
NOTES_STORE_PATH = PKG / "notes" / "notes_store.py"

EXPECTED_PREFIX = "/api/notes"
# The five (path, method) routes the binding exposes.
EXPECTED_ROUTES = frozenset(
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
# Isolation harness (the S243 lesson, the S244 idiom)
# ---------------------------------------------------------------------------


def _ensure_pkg(name: str, path: Path) -> None:
    """Ensure ``name`` exists in sys.modules and is package-like.

    Non-destructive: keeps any pre-existing stub object (an earlier suite's),
    only granting it a ``__path__`` so a dotted ``spec_from_file_location`` load
    of a submodule resolves.
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


_ISO: dict = {}


def _isolated_flat(name: str, rel: str):
    """Load a module under a FLAT name (robust in the sweep)."""
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
    return _isolated_flat("s245_notes_store_iso", "notes/notes_store.py")


def _make_store(tmp_path, single_user_mode: bool = True):
    """A fresh NotesStore rooted in a tmp dir (single-user by default).

    The per-user isolation test builds a multi-user store so the store honours
    the user_id the route threads through; the real single-user singleton
    collapses every caller to the single user, which is the correct posture
    for a single-user deployment.
    """
    return _store_module().NotesStore(
        root=str(tmp_path), single_user_mode=single_user_mode
    )


def _load_route():
    """Load routes_notes under its dotted name into package-like stubs.

    Pre-loads the (light) schemas dotted, stubs routes_auth (the auth chain
    never fires; the dep is overridden per test), and registers the flat-loaded
    real notes_store at the dotted key so the route's absolute import resolves.
    On the pristine tree routes_notes is absent and this raises ImportError
    INSIDE the calling test -- a failure, never a collection error.
    """
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.api", PKG / "api")
    _ensure_pkg("opti_oignon.notes", PKG / "notes")
    _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
    if "opti_oignon.api.routes_auth" not in sys.modules:
        stub = types.ModuleType("opti_oignon.api.routes_auth")

        def _get_current_user() -> dict:  # pragma: no cover - overridden in tests
            return {"sub": None}

        stub._get_current_user = _get_current_user  # type: ignore[attr-defined]
        sys.modules["opti_oignon.api.routes_auth"] = stub
    sys.modules["opti_oignon.notes.notes_store"] = _store_module()
    return _load_dotted("opti_oignon.api.routes_notes", ROUTES_NOTES_PATH)


def _build(tmp_path, single_user_mode: bool = True):
    """Build a bare app over the route with the store and auth injected."""
    routes = _load_route()
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    store = _make_store(tmp_path, single_user_mode=single_user_mode)
    app = FastAPI()
    app.include_router(routes.notes_router)
    state = {"sub": "user_a"}
    app.dependency_overrides[routes._notes_store_dep] = lambda: store
    app.dependency_overrides[routes._get_current_user] = lambda: {"sub": state["sub"]}
    client = TestClient(app)
    return client, store, routes, state


# ---------------------------------------------------------------------------
# Family 1 -- source / structure
# ---------------------------------------------------------------------------


class TestRouteSource:
    def test_module_exists(self):
        assert ROUTES_NOTES_PATH.exists(), "opti_oignon/api/routes_notes.py missing"

    def test_checkpoint_before_apply_hardcoded(self):
        assert "checkpoint_before_apply = True" in _read(ROUTES_NOTES_PATH)

    def test_feature_available_flag(self):
        assert "FEATURE_AVAILABLE" in _read(ROUTES_NOTES_PATH)

    def test_api_notes_prefix(self):
        assert 'prefix="/api/notes"' in _read(ROUTES_NOTES_PATH)

    def test_route_delegates_no_direct_sql(self):
        src = _read(ROUTES_NOTES_PATH)
        assert src != ""
        # The route delegates to the store; it never touches the DB directly.
        assert "import sqlite3" not in src
        assert ".execute(" not in src

    def test_pure_ascii_no_decoration(self):
        raw = _read(ROUTES_NOTES_PATH)
        assert raw != ""
        assert all(ord(c) < 128 for c in raw)
        assert "====" not in raw


class TestRouteShape:
    def test_router_prefix_runtime(self):
        routes = _load_route()
        assert routes.notes_router.prefix == EXPECTED_PREFIX

    def test_five_routes_exact(self):
        routes = _load_route()
        found = set()
        for r in routes.notes_router.routes:
            path = getattr(r, "path", None)
            methods = getattr(r, "methods", None) or set()
            for m in methods:
                if m in ("GET", "POST", "PATCH", "DELETE", "PUT"):
                    found.add((path, m))
        assert found == set(EXPECTED_ROUTES), found

    def test_store_dep_and_check_present(self):
        routes = _load_route()
        assert hasattr(routes, "_notes_store_dep")
        assert hasattr(routes, "_check_store")
        assert hasattr(routes, "_record_to_schema")


# ---------------------------------------------------------------------------
# Family 2 -- registration on the app
# ---------------------------------------------------------------------------


class TestAppRegistration:
    def test_app_imports_notes_router(self):
        src = _read(APP_PATH)
        assert "routes_notes import" in src
        assert "notes_router" in src

    def test_app_includes_notes_router(self):
        src = _read(APP_PATH)
        assert "include_router(notes_router)" in src


# ---------------------------------------------------------------------------
# Family 3 -- schemas
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_schema_symbols_in_source(self):
        src = _read(SCHEMAS_PATH)
        assert "class NoteSchema" in src
        assert "class NoteCreateRequest" in src
        assert "class NoteUpdateRequest" in src

    def test_schemas_load_and_validate(self):
        schemas = _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
        note = schemas.NoteSchema(
            id="n1",
            title="t",
            body_crdt_b64=base64.b64encode(b"x").decode("ascii"),
            tags=["a", "b"],
            pinned=True,
        )
        assert note.id == "n1"
        assert note.tags == ["a", "b"]
        create = schemas.NoteCreateRequest(title="hello")
        assert create.title == "hello"
        upd = schemas.NoteUpdateRequest(pinned=True)
        assert upd.pinned is True


# ---------------------------------------------------------------------------
# Family 4 -- behavioural (TestClient, injected store)
# ---------------------------------------------------------------------------


class TestBehaviour:
    def test_create_returns_note_body_and_tags_roundtrip(self, tmp_path):
        client, _store, _routes, _state = _build(tmp_path)
        body = b"\x00\x01opaque-crdt"
        payload = {
            "title": "My note",
            "body_crdt_b64": base64.b64encode(body).decode("ascii"),
            "tags": ["alpha", "beta"],
            "pinned": True,
        }
        r = client.post("/api/notes", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["id"]
        assert data["title"] == "My note"
        assert base64.b64decode(data["body_crdt_b64"]) == body
        assert data["tags"] == ["alpha", "beta"]
        assert data["pinned"] is True
        assert data["deleted"] is False

    def test_list_includes_created(self, tmp_path):
        client, _store, _routes, _state = _build(tmp_path)
        client.post("/api/notes", json={"title": "one"})
        client.post("/api/notes", json={"title": "two"})
        r = client.get("/api/notes")
        assert r.status_code == 200
        titles = {n["title"] for n in r.json()}
        assert {"one", "two"} <= titles

    def test_get_returns_note(self, tmp_path):
        client, _store, _routes, _state = _build(tmp_path)
        nid = client.post("/api/notes", json={"title": "x"}).json()["id"]
        r = client.get(f"/api/notes/{nid}")
        assert r.status_code == 200
        assert r.json()["id"] == nid

    def test_get_missing_is_404(self, tmp_path):
        client, _store, _routes, _state = _build(tmp_path)
        r = client.get("/api/notes/does-not-exist")
        assert r.status_code == 404

    def test_update_metadata(self, tmp_path):
        client, _store, _routes, _state = _build(tmp_path)
        nid = client.post("/api/notes", json={"title": "old"}).json()["id"]
        r = client.patch(
            f"/api/notes/{nid}",
            json={"title": "new", "tags": ["t"], "pinned": True},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["title"] == "new"
        assert data["tags"] == ["t"]
        assert data["pinned"] is True

    def test_update_body_blob(self, tmp_path):
        client, _store, _routes, _state = _build(tmp_path)
        nid = client.post("/api/notes", json={"title": "x"}).json()["id"]
        new_body = b"\x02\x03new-state"
        r = client.patch(
            f"/api/notes/{nid}",
            json={"body_crdt_b64": base64.b64encode(new_body).decode("ascii")},
        )
        assert r.status_code == 200, r.text
        assert base64.b64decode(r.json()["body_crdt_b64"]) == new_body

    def test_update_missing_is_404(self, tmp_path):
        client, _store, _routes, _state = _build(tmp_path)
        r = client.patch("/api/notes/nope", json={"title": "x"})
        assert r.status_code == 404

    def test_delete_tombstones(self, tmp_path):
        client, _store, _routes, _state = _build(tmp_path)
        nid = client.post("/api/notes", json={"title": "x"}).json()["id"]
        r = client.delete(f"/api/notes/{nid}")
        assert r.status_code == 200, r.text
        assert r.json()["deleted"] is True
        # gone from the default list, and a get is 404
        assert client.get(f"/api/notes/{nid}").status_code == 404
        titles = {n["title"] for n in client.get("/api/notes").json()}
        assert "x" not in titles

    def test_delete_missing_is_404(self, tmp_path):
        client, _store, _routes, _state = _build(tmp_path)
        r = client.delete("/api/notes/nope")
        assert r.status_code == 404

    def test_per_user_isolation(self, tmp_path):
        client, _store, _routes, state = _build(tmp_path, single_user_mode=False)
        # user_a creates a note
        state["sub"] = "user_a"
        nid = client.post("/api/notes", json={"title": "secret"}).json()["id"]
        # user_b sees nothing and cannot fetch it
        state["sub"] = "user_b"
        assert client.get("/api/notes").json() == []
        assert client.get(f"/api/notes/{nid}").status_code == 404
        # back to user_a: still there
        state["sub"] = "user_a"
        assert client.get(f"/api/notes/{nid}").status_code == 200

    def test_check_store_guard_is_503(self, tmp_path):
        routes = _load_route()
        from fastapi import HTTPException

        original = routes.FEATURE_AVAILABLE
        try:
            routes.FEATURE_AVAILABLE = False
            with pytest.raises(HTTPException) as exc:
                routes._check_store()
            assert exc.value.status_code == 503
        finally:
            routes.FEATURE_AVAILABLE = original


# ---------------------------------------------------------------------------
# Family 5 -- premise guards (green before and after)
# ---------------------------------------------------------------------------


class TestPremiseGuards:
    def test_notes_store_module_loads(self):
        mod = _store_module()
        assert hasattr(mod, "NotesStore")
        assert hasattr(mod, "get_notes_store")

    def test_routes_memory_is_the_template(self):
        src = _read(ROUTES_MEMORY_PATH)
        assert "memories_router" in src
        assert "_get_current_user" in src

    def test_manage_notes_tool_intact(self):
        src = _read(TOOLS_PATH)
        assert "MANAGE_NOTES_SCHEMA" in src
        assert "make_manage_notes_handler" in src

    def test_schemas_module_exists(self):
        assert SCHEMAS_PATH.exists()


# ---------------------------------------------------------------------------
# Family 6 -- AST / ASCII
# ---------------------------------------------------------------------------


class TestASTValid:
    def test_route_source_parses(self):
        src = _read(ROUTES_NOTES_PATH)
        assert src != ""
        ast.parse(src, filename=str(ROUTES_NOTES_PATH))

    def test_this_suite_parses(self):
        src = _read(Path(__file__))
        assert src != ""
        ast.parse(src, filename=__file__)
