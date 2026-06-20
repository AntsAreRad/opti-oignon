"""S250 -- the N.5 voice-transcription backend (the first media post-processing
bloc): the opt-in, sandboxed whisper.cpp transcription orchestration, the
``NotesStore.update_attachment`` write-back, the opt-in dependency group, and the
HTTP trigger on a SEPARATE router.

S249 landed the shared notes-attachment route (``notes_attachments_router`` at
``/api/notes/attachments``: upload / list / metadata / download / delete over the
N.1 ``attachment`` manifest and the two-layer ``NotesBlobStore``), the additive
``NotesStore.delete_attachment``, and the two attachment schemas. The S249 carry
-forward named the one data-layer gap a media post-processing bloc must fill:
``NotesStore.update_attachment`` (absent then), the per-user, parameterized write
-back that sets ``transcript_text`` / ``caption_text`` / ``ocr_text`` on an
existing manifest row.

This lot lands the container-provable half of N.5 (voice transcription):

 - ``NotesStore.update_attachment`` -- additive, per-user, parameterized; sets
   transcript_text / caption_text / ocr_text on an existing row (N.6 reuses the
   caption / ocr legs).
 - ``opti_oignon/notes/transcription.py`` -- the orchestration. It fetches the
   manifest (per-user, kind must be ``audio``), is FAIL-SECURE on the disposable
   bubblewrap floor (with no real bwrap it REFUSES, it never falls back to a
   degraded tempdir for this file-touching post-processing), decrypts the blob in
   memory via ``NotesBlobStore.open``, copies the decrypted bytes INTO the
   disposable sandbox workspace (no plaintext temp on the host), runs the
   transcriber inside the sandbox, requires human approval before the durable
   write-back, then calls ``update_attachment``, and destroys the sandbox in a
   ``finally`` (wiping the plaintext audio). The transcriber and the sandbox are
   injected seams; the live builder wires whisper.cpp + the real SandboxManager,
   the host-assured piece.
 - the opt-in ``transcribe`` dependency group in pyproject, off by default (the
   ``veilid`` precedent: not pulled by the base install, not in ``all``).
 - ``opti_oignon/api/routes_notes_transcription.py`` -- a SEPARATE router
   (``notes_transcription_router`` at ``/api/notes/transcription``), NOT folded
   into ``notes_attachments_router`` (so the S249 ``test_five_routes_exact`` pin
   stays green -- the ``routes_note_actions`` / ``routes_notes_attachments``
   precedent), a pure chain addition. One ``POST /{attachment_id}`` drives the
   orchestration and returns the structured result; the only HTTP error is the
   503 availability guard.

What is host-assured and NOT in this suite: the live whisper.cpp run inside a real
disposable bubblewrap (both whisper.cpp and bwrap are absent in-container), in
NOTES_TRANSCRIPTION_E2E_S250.md, never simulated here.

Ten families:

 1. update_attachment -- the additive write-back over a real NotesStore on tmp.
 2. transcription.py source / structure.
 3. transcription.py behavioural -- fail-secure, not_found, not_audio,
    blob_unavailable, preview (approve False), commit (approve True), tool_failed,
    the disposable create+destroy lifecycle, no host plaintext temp.
 4. route source / structure.
 5. route behavioural (TestClient, injected real store + blobs, fake sandbox +
    transcriber) -- preview, commit, fail-secure, per-user, 503 guard.
 6. registration -- app.py imports and includes notes_transcription_router.
 7. schemas -- TranscriptionRequest / TranscriptionResultSchema load and validate.
 8. the opt-in transcribe dependency group, off by default.
 9. premise guards (green before AND after) -- the N.1 attachment data layer, the
    NotesBlobStore surface, the S245 notes_router still five routes, the S249
    notes_attachments_router still five routes.
 10. AST / ASCII of the orchestration, the route, and this suite.

Red-before on the pristine S249 tree (no transcription.py, no route, no
update_attachment, no schemas, no transcribe group): every family-1 / 2 / 4 / 7 /
8 contract pin and the family-3 / 5 behavioural pins FAIL (the read helpers return
empty strings so absence is a failure; the behavioural families load the modules
INSIDE the test so absence is an ImportError failure, never a collection error),
while the family-9 premise guards, the family-10 suite-parse pin, and the
family-2 / 4 negative invariants (vacuous on the absent modules) PASS by design.

Isolation (the S243 / S247 / S249 idiom): the behavioural families load the
orchestration and the route under their dotted names into package-like stubs,
pre-loading the real (light) schemas dotted and the real notes_store / blob_store
dotted (their guarded relative imports resolve as submodules of the stub package,
with the REAL AES-256-GCM primitive), and stub routes_auth (the auth dep is
overridden per test). No fastapi/ollama package import is forced at collection,
and the real SandboxManager is never imported (the sandbox is a fake seam here);
the live whisper.cpp / bwrap path is never reached.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

# Defensive: never pull the real ollama during collection.
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "opti_oignon"
ORCH_PATH = PKG / "notes" / "transcription.py"
ROUTE_PATH = PKG / "api" / "routes_notes_transcription.py"
SCHEMAS_PATH = PKG / "api" / "schemas.py"
APP_PATH = PKG / "api" / "app.py"
NOTES_STORE_PATH = PKG / "notes" / "notes_store.py"
BLOB_STORE_PATH = PKG / "notes" / "blob_store.py"
ROUTES_NOTES_PATH = PKG / "api" / "routes_notes.py"
ROUTES_ATTACH_PATH = PKG / "api" / "routes_notes_attachments.py"
PYPROJECT_PATH = REPO / "pyproject.toml"

EXPECTED_PREFIX = "/api/notes/transcription"
# The single (path, method) route the trigger surface exposes.
EXPECTED_ROUTES = frozenset({("/api/notes/transcription/{attachment_id}", "POST")})
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
# The S249 notes_attachments_router's five routes (the fold-in guard).
ATTACH_ROUTER_ROUTES = frozenset(
    {
        ("/api/notes/attachments/note/{note_id}", "POST"),
        ("/api/notes/attachments/note/{note_id}", "GET"),
        ("/api/notes/attachments/{attachment_id}", "GET"),
        ("/api/notes/attachments/{attachment_id}/blob", "GET"),
        ("/api/notes/attachments/{attachment_id}", "DELETE"),
    }
)


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Isolation harness (the S243 / S247 / S249 idiom)
# ---------------------------------------------------------------------------


def _ensure_pkg(name: str, path: Path) -> None:
    """Ensure ``name`` is in sys.modules and package-like (has __path__)."""
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


def _load_orch():
    """Load opti_oignon.notes.transcription dotted into package stubs.

    On the pristine tree the module is absent and this raises ImportError INSIDE
    the calling test -- a failure, never a collection error.
    """
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.notes", PKG / "notes")
    _notes_modules()
    return _load_dotted("opti_oignon.notes.transcription", ORCH_PATH)


def _load_route():
    """Load routes_notes_transcription under its dotted name into package stubs."""
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.api", PKG / "api")
    _ensure_pkg("opti_oignon.notes", PKG / "notes")
    _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
    _notes_modules()
    _load_orch()
    if "opti_oignon.api.routes_auth" not in sys.modules:
        stub = types.ModuleType("opti_oignon.api.routes_auth")

        def _get_current_user() -> dict:  # pragma: no cover - overridden in tests
            return {"sub": None}

        stub._get_current_user = _get_current_user  # type: ignore[attr-defined]
        sys.modules["opti_oignon.api.routes_auth"] = stub
    return _load_dotted("opti_oignon.api.routes_notes_transcription", ROUTE_PATH)


def _routes_of(router) -> set:
    out = set()
    for r in getattr(router, "routes", []):
        methods = getattr(r, "methods", None) or set()
        path = getattr(r, "path", "")
        for m in methods:
            if m in {"GET", "POST", "PATCH", "DELETE", "PUT"}:
                out.add((path, m))
    return out


# ---------------------------------------------------------------------------
# Fake seams (the sandbox and the transcriber; the live ones are host-assured)
# ---------------------------------------------------------------------------


class _FakeSandbox:
    """A minimal SandboxManager-like fake: a real tmp workspace per session, the
    bwrap-availability signal, and a recorded create / destroy lifecycle.

    It exposes only the handful of members the orchestration drives. ``destroy``
    actually removes the workspace so the suite can assert the disposable
    teardown (the plaintext audio does not linger)."""

    def __init__(self, tmp: Path, *, bwrap_available: bool = True) -> None:
        self._tmp = tmp
        self.bwrap_available = bwrap_available
        self.created: list[str] = []
        self.destroyed: list[str] = []
        self._ws: dict[str, Path] = {}

    def create_sandbox(
        self,
        session_id=None,
        allow_degraded: bool = False,
        label: str = "",
        owner_user_id: str = "local",
        timeout_override=None,
    ):
        sid = session_id or ("sbx-%d" % len(self.created))
        ws = self._tmp / ("ws_" + sid)
        ws.mkdir(parents=True, exist_ok=True)
        self._ws[sid] = ws
        self.created.append(sid)
        return types.SimpleNamespace(session_id=sid, workspace_path=str(ws))

    def get_active_workspace_path(self, session_id: str) -> str:
        return str(self._ws[session_id])

    def destroy_sandbox(self, session_id: str) -> bool:
        self.destroyed.append(session_id)
        ws = self._ws.get(session_id)
        if ws is not None and ws.exists():
            shutil.rmtree(ws)
        return True


def _ok_transcriber(sandbox, session_id, input_name):
    """A fake transcriber that confirms the decrypted bytes were written into the
    sandbox workspace, then returns a canned transcript (whisper.cpp is absent)."""
    ws = Path(sandbox.get_active_workspace_path(session_id))
    assert (ws / input_name).exists(), "decrypted bytes not injected into sandbox"
    return "hello there, this is the transcript"


def _boom_transcriber(sandbox, session_id, input_name):
    raise RuntimeError("whisper.cpp crashed")


def _seed_audio(store, blobs, *, sub: str, body: bytes = b"RIFF....WAVEfmt audio"):
    """Create a note and an AUDIO attachment (manifest + sealed blob)."""
    note = store.add_note(title="voice", body_crdt=b"", user_id=sub)
    rec = store.add_attachment(
        note.id, "audio", blob_ref="", mime="audio/wav", byte_size=len(body), user_id=sub
    )
    blobs.seal(rec.id, body)
    return note, rec


def _stores(tmp_path, *, master=True):
    ns, bs = _notes_modules()
    store = ns.NotesStore(root=tmp_path, single_user_mode=False)
    key = os.urandom(32) if master else None
    blobs = bs.NotesBlobStore(root=tmp_path, master_key=key) if key else (
        bs.NotesBlobStore(root=tmp_path)
    )
    return store, blobs


# ---------------------------------------------------------------------------
# Family 1 -- NotesStore.update_attachment (the additive write-back)
# ---------------------------------------------------------------------------


class TestUpdateAttachment:
    def test_method_present_in_source(self):
        assert "def update_attachment(" in _read(NOTES_STORE_PATH)

    def test_update_sets_transcript_text(self, tmp_path):
        store, _ = _stores(tmp_path)
        note = store.add_note(title="n", body_crdt=b"", user_id="alice")
        rec = store.add_attachment(note.id, "audio", blob_ref="", user_id="alice")
        assert store.get_attachment(rec.id, user_id="alice").transcript_text is None
        changed = store.update_attachment(
            rec.id, transcript_text="spoken words", user_id="alice"
        )
        assert changed is True
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.transcript_text == "spoken words"

    def test_update_sets_caption_and_ocr(self, tmp_path):
        store, _ = _stores(tmp_path)
        note = store.add_note(title="n", body_crdt=b"", user_id="alice")
        rec = store.add_attachment(note.id, "image", blob_ref="", user_id="alice")
        store.update_attachment(
            rec.id, caption_text="a cat", ocr_text="MEOW", user_id="alice"
        )
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text == "a cat"
        assert got.ocr_text == "MEOW"

    def test_update_is_per_user(self, tmp_path):
        store, _ = _stores(tmp_path)
        note = store.add_note(title="n", body_crdt=b"", user_id="alice")
        rec = store.add_attachment(note.id, "audio", blob_ref="", user_id="alice")
        # Bob cannot write back onto Alice's row.
        changed = store.update_attachment(
            rec.id, transcript_text="hijack", user_id="bob"
        )
        assert changed is False
        assert store.get_attachment(rec.id, user_id="alice").transcript_text is None

    def test_update_missing_row_is_false(self, tmp_path):
        store, _ = _stores(tmp_path)
        assert store.update_attachment(
            "nope", transcript_text="x", user_id="alice"
        ) is False

    def test_update_no_fields_leaves_row(self, tmp_path):
        store, _ = _stores(tmp_path)
        note = store.add_note(title="n", body_crdt=b"", user_id="alice")
        rec = store.add_attachment(
            note.id, "audio", blob_ref="", transcript_text="orig", user_id="alice"
        )
        # A no-op update (no fields) must not blank an existing value.
        store.update_attachment(rec.id, user_id="alice")
        assert store.get_attachment(rec.id, user_id="alice").transcript_text == "orig"

    def test_update_no_fstring_sql(self):
        src = _read(NOTES_STORE_PATH)
        # The update_attachment method body must not build SQL by f-string.
        for verb in ("UPDATE", "SELECT", "INSERT", "DELETE FROM"):
            assert f'f"{verb}' not in src and f"f'{verb}" not in src, verb


# ---------------------------------------------------------------------------
# Family 2 -- transcription.py source / structure
# ---------------------------------------------------------------------------


class TestOrchSource:
    def test_module_exists(self):
        assert ORCH_PATH.exists(), "opti_oignon/notes/transcription.py missing"

    def test_checkpoint_before_apply_hardcoded(self):
        assert "checkpoint_before_apply = True" in _read(ORCH_PATH)

    def test_feature_available_flag(self):
        assert "FEATURE_AVAILABLE" in _read(ORCH_PATH)

    def test_disposable_sandbox_lifecycle_referenced(self):
        src = _read(ORCH_PATH)
        assert "create_sandbox" in src
        assert "destroy_sandbox" in src

    def test_fail_secure_language(self):
        src = _read(ORCH_PATH)
        assert "bwrap_available" in src
        assert "sandbox_unavailable" in src

    def test_decrypts_via_blob_open(self):
        src = _read(ORCH_PATH)
        assert ".open(" in src
        assert "update_attachment" in src

    def test_no_direct_sql(self):
        src = _read(ORCH_PATH)
        assert "import sqlite3" not in src
        assert ".execute(" not in src

    def test_not_a_model_tool(self):
        src = _read(ORCH_PATH)
        assert "ToolSchema(" not in src
        assert "register_tool" not in src

    def test_pure_ascii_no_decoration(self):
        raw = _read(ORCH_PATH)
        assert raw != ""
        assert all(ord(c) < 128 for c in raw)
        assert "====" not in raw


# ---------------------------------------------------------------------------
# Family 3 -- transcription.py behavioural (the orchestration)
# ---------------------------------------------------------------------------


class TestOrchBehaviour:
    def test_fail_secure_when_no_bwrap(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_audio(store, blobs, sub="alice")
        sbx = _FakeSandbox(tmp_path, bwrap_available=False)
        res = orch.transcribe_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, transcriber=_ok_transcriber, approve=True,
        )
        assert res.refused is True
        assert res.reason == "sandbox_unavailable"
        assert res.written_back is False
        assert sbx.created == []  # never even created a workspace
        assert store.get_attachment(rec.id, user_id="alice").transcript_text is None

    def test_not_found(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        sbx = _FakeSandbox(tmp_path)
        res = orch.transcribe_attachment(
            "ghost", user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, transcriber=_ok_transcriber, approve=True,
        )
        assert res.refused is True
        assert res.reason == "not_found"
        assert sbx.created == []

    def test_not_audio(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        note = store.add_note(title="pic", body_crdt=b"", user_id="alice")
        rec = store.add_attachment(note.id, "image", blob_ref="", user_id="alice")
        blobs.seal(rec.id, b"PNG-bytes")
        sbx = _FakeSandbox(tmp_path)
        res = orch.transcribe_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, transcriber=_ok_transcriber, approve=True,
        )
        assert res.refused is True
        assert res.reason == "not_audio"
        assert sbx.created == []

    def test_blob_unavailable(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path, master=False)  # keyless blob store
        note = store.add_note(title="voice", body_crdt=b"", user_id="alice")
        rec = store.add_attachment(note.id, "audio", blob_ref="", user_id="alice")
        sbx = _FakeSandbox(tmp_path)
        res = orch.transcribe_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, transcriber=_ok_transcriber, approve=True,
        )
        assert res.refused is True
        assert res.reason == "blob_unavailable"
        assert sbx.created == []  # refused before creating a sandbox
        assert store.get_attachment(rec.id, user_id="alice").transcript_text is None

    def test_preview_does_not_write_back(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_audio(store, blobs, sub="alice")
        sbx = _FakeSandbox(tmp_path)
        res = orch.transcribe_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, transcriber=_ok_transcriber, approve=False,
        )
        assert res.ok is True
        assert res.transcript_text == "hello there, this is the transcript"
        assert res.written_back is False
        # Preview only: the manifest row is NOT persisted.
        assert store.get_attachment(rec.id, user_id="alice").transcript_text is None
        # The disposable sandbox was created and torn down.
        assert sbx.created and sbx.destroyed == sbx.created

    def test_commit_writes_back_on_approval(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_audio(store, blobs, sub="alice")
        sbx = _FakeSandbox(tmp_path)
        res = orch.transcribe_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, transcriber=_ok_transcriber, approve=True,
        )
        assert res.ok is True
        assert res.written_back is True
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.transcript_text == "hello there, this is the transcript"
        assert sbx.created and sbx.destroyed == sbx.created

    def test_tool_failure_is_clean(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_audio(store, blobs, sub="alice")
        sbx = _FakeSandbox(tmp_path)
        res = orch.transcribe_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, transcriber=_boom_transcriber, approve=True,
        )
        assert res.ok is False
        assert res.refused is False
        assert res.reason == "transcription_failed"
        assert store.get_attachment(rec.id, user_id="alice").transcript_text is None
        # The sandbox is still torn down on the failure path (finally).
        assert sbx.destroyed == sbx.created and sbx.created

    def test_no_host_plaintext_temp_remains(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_audio(store, blobs, sub="alice", body=b"SECRET-AUDIO-BYTES")
        sbx = _FakeSandbox(tmp_path)
        orch.transcribe_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, transcriber=_ok_transcriber, approve=True,
        )
        # After teardown, no workspace dir and no file carrying the plaintext.
        leaks = [p for p in tmp_path.rglob("*") if p.is_file()
                 and b"SECRET-AUDIO-BYTES" in _safe_bytes(p)]
        assert leaks == [], leaks


def _safe_bytes(p: Path) -> bytes:
    try:
        return p.read_bytes()
    except OSError:
        return b""


# ---------------------------------------------------------------------------
# Family 4 -- route source / structure
# ---------------------------------------------------------------------------


class TestRouteSource:
    def test_module_exists(self):
        assert ROUTE_PATH.exists(), "opti_oignon/api/routes_notes_transcription.py missing"

    def test_checkpoint_before_apply_hardcoded(self):
        assert "checkpoint_before_apply = True" in _read(ROUTE_PATH)

    def test_feature_available_flag(self):
        assert "FEATURE_AVAILABLE" in _read(ROUTE_PATH)

    def test_prefix(self):
        assert EXPECTED_PREFIX in _read(ROUTE_PATH)

    def test_route_no_direct_sql(self):
        src = _read(ROUTE_PATH)
        assert "import sqlite3" not in src
        assert ".execute(" not in src

    def test_route_is_not_a_model_tool(self):
        src = _read(ROUTE_PATH)
        assert "ToolSchema(" not in src
        assert "register_tool" not in src

    def test_pure_ascii_no_decoration(self):
        raw = _read(ROUTE_PATH)
        assert raw != ""
        assert all(ord(c) < 128 for c in raw)
        assert "====" not in raw


class TestRouteShape:
    def test_router_prefix_runtime(self):
        routes = _load_route()
        assert routes.notes_transcription_router.prefix == EXPECTED_PREFIX

    def test_one_route_exact(self):
        routes = _load_route()
        assert _routes_of(routes.notes_transcription_router) == EXPECTED_ROUTES

    def test_seams_present(self):
        routes = _load_route()
        assert hasattr(routes, "_notes_store_dep")
        assert hasattr(routes, "_blob_store_dep")
        assert hasattr(routes, "_sandbox_dep")
        assert hasattr(routes, "_transcriber_dep")


# ---------------------------------------------------------------------------
# Family 5 -- route behavioural (TestClient)
# ---------------------------------------------------------------------------


def _build(tmp_path, *, sub: str = "alice", master: bool = True,
           bwrap: bool = True, transcriber=_ok_transcriber):
    routes = _load_route()
    store, blobs = _stores(tmp_path, master=master)
    sbx = _FakeSandbox(tmp_path, bwrap_available=bwrap)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    state = {"sub": sub}
    app = FastAPI()
    app.include_router(routes.notes_transcription_router)
    app.dependency_overrides[routes._notes_store_dep] = lambda: store
    app.dependency_overrides[routes._blob_store_dep] = lambda: blobs
    app.dependency_overrides[routes._sandbox_dep] = lambda: sbx
    app.dependency_overrides[routes._transcriber_dep] = lambda: transcriber
    app.dependency_overrides[routes._get_current_user] = lambda: {"sub": state["sub"]}
    return TestClient(app), routes, store, blobs, sbx, state


class TestRouteBehaviour:
    def test_preview_returns_transcript_no_write_back(self, tmp_path):
        client, routes, store, blobs, sbx, _ = _build(tmp_path)
        _, rec = _seed_audio(store, blobs, sub="alice")
        r = client.post("/api/notes/transcription/" + rec.id, json={"approve": False})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["transcript_text"] == "hello there, this is the transcript"
        assert body["written_back"] is False
        assert store.get_attachment(rec.id, user_id="alice").transcript_text is None

    def test_commit_writes_back(self, tmp_path):
        client, routes, store, blobs, sbx, _ = _build(tmp_path)
        _, rec = _seed_audio(store, blobs, sub="alice")
        r = client.post("/api/notes/transcription/" + rec.id, json={"approve": True})
        assert r.status_code == 200, r.text
        assert r.json()["written_back"] is True
        assert store.get_attachment(rec.id, user_id="alice").transcript_text == (
            "hello there, this is the transcript"
        )

    def test_fail_secure_refusal(self, tmp_path):
        client, routes, store, blobs, sbx, _ = _build(tmp_path, bwrap=False)
        _, rec = _seed_audio(store, blobs, sub="alice")
        r = client.post("/api/notes/transcription/" + rec.id, json={"approve": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["refused"] is True
        assert body["reason"] == "sandbox_unavailable"

    def test_per_user_isolation_is_not_found(self, tmp_path):
        # Bob triggers transcription on Alice's attachment: a refused not_found,
        # never a served transcript and never a write-back.
        client, routes, store, blobs, sbx, state = _build(tmp_path, sub="alice")
        _, rec = _seed_audio(store, blobs, sub="alice")
        state["sub"] = "bob"
        r = client.post("/api/notes/transcription/" + rec.id, json={"approve": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["refused"] is True
        assert body["reason"] == "not_found"
        assert store.get_attachment(rec.id, user_id="alice").transcript_text is None

    def test_availability_guard_503(self, tmp_path):
        routes = _load_route()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(routes.notes_transcription_router)
        app.dependency_overrides[routes._get_current_user] = lambda: {"sub": "alice"}
        orig = routes.FEATURE_AVAILABLE
        try:
            routes.FEATURE_AVAILABLE = False
            client = TestClient(app)
            r = client.post("/api/notes/transcription/x", json={"approve": True})
            assert r.status_code == 503, r.text
        finally:
            routes.FEATURE_AVAILABLE = orig


# ---------------------------------------------------------------------------
# Family 6 -- registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_app_imports_router(self):
        assert "notes_transcription_router" in _read(APP_PATH)

    def test_app_includes_router(self):
        src = _read(APP_PATH)
        assert "include_router(notes_transcription_router)" in src


# ---------------------------------------------------------------------------
# Family 7 -- schemas
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_symbols_present(self):
        src = _read(SCHEMAS_PATH)
        assert "class TranscriptionRequest(" in src
        assert "class TranscriptionResultSchema(" in src

    def test_schemas_load_and_validate(self):
        _ensure_pkg("opti_oignon", PKG)
        _ensure_pkg("opti_oignon.api", PKG / "api")
        schemas = _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
        req = schemas.TranscriptionRequest(approve=True)
        assert req.approve is True
        # Default is the safe one: no durable write-back without approval.
        assert schemas.TranscriptionRequest().approve is False
        res = schemas.TranscriptionResultSchema(
            attachment_id="a1", ok=True, transcript_text="t", written_back=True
        )
        assert res.ok is True and res.written_back is True


# ---------------------------------------------------------------------------
# Family 8 -- the opt-in transcribe dependency group (off by default)
# ---------------------------------------------------------------------------


class TestOptInDependencyGroup:
    def _optional(self):
        import tomllib

        data = tomllib.loads(_read(PYPROJECT_PATH))
        return data["project"].get("optional-dependencies", {})

    def test_transcribe_group_exists(self):
        assert "transcribe" in self._optional()

    def test_transcribe_group_nonempty(self):
        assert len(self._optional()["transcribe"]) >= 1

    def test_transcribe_off_by_default(self):
        # Not pulled by the base install nor by the aggregate ``all`` extra (the
        # veilid precedent: opt-in, off by default).
        import tomllib

        data = tomllib.loads(_read(PYPROJECT_PATH))
        base = data["project"].get("dependencies", [])
        joined = " ".join(base).lower()
        assert "whisper" not in joined and "transcribe" not in joined
        all_extra = " ".join(self._optional().get("all", [])).lower()
        assert "transcribe" not in all_extra

    def test_existing_groups_preserved(self):
        opt = self._optional()
        for group in ("llama", "auth", "sqlcipher", "veilid"):
            assert group in opt, group


# ---------------------------------------------------------------------------
# Family 9 -- premise guards (green before AND after)
# ---------------------------------------------------------------------------


class TestPremiseGuards:
    def test_n1_attachment_data_layer_intact(self):
        src = _read(NOTES_STORE_PATH)
        assert "def add_attachment(" in src
        assert "def get_attachment(" in src
        assert "def list_attachments(" in src
        assert "ATTACHMENT_KINDS" in src

    def test_blob_store_surface_intact(self):
        src = _read(BLOB_STORE_PATH)
        assert "def seal(" in src
        assert "def open(" in src
        assert "class NotesBlobUnavailable" in src

    def test_s245_notes_router_still_five_routes(self):
        # The transcription trigger is a SEPARATE router; notes_router is untouched.
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

    def test_s249_attachment_router_still_five_routes(self):
        # The S249 five-routes pin must stay green (we did not fold in).
        from importlib import import_module  # noqa: F401  (documents intent)

        _ensure_pkg("opti_oignon", PKG)
        _ensure_pkg("opti_oignon.api", PKG / "api")
        _ensure_pkg("opti_oignon.notes", PKG / "notes")
        _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
        _notes_modules()
        if "opti_oignon.api.routes_auth" not in sys.modules:
            stub = types.ModuleType("opti_oignon.api.routes_auth")
            stub._get_current_user = lambda: {"sub": None}  # type: ignore[attr-defined]
            sys.modules["opti_oignon.api.routes_auth"] = stub
        ra = _load_dotted(
            "opti_oignon.api.routes_notes_attachments", ROUTES_ATTACH_PATH
        )
        assert _routes_of(ra.notes_attachments_router) == ATTACH_ROUTER_ROUTES


# ---------------------------------------------------------------------------
# Family 10 -- AST / ASCII
# ---------------------------------------------------------------------------


class TestASTValid:
    def test_orchestration_parses(self):
        src = _read(ORCH_PATH)
        assert src != ""
        ast.parse(src, filename=str(ORCH_PATH))

    def test_route_parses(self):
        src = _read(ROUTE_PATH)
        assert src != ""
        ast.parse(src, filename=str(ROUTE_PATH))

    def test_this_suite_parses(self):
        src = _read(Path(__file__))
        assert src != ""
        ast.parse(src, filename=__file__)
