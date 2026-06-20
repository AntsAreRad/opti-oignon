"""S251 -- the N.6 picture caption / OCR backend (the second media post-processing
bloc): the opt-in, sandboxed vision caption / OCR orchestration, the opt-in
``vision`` dependency group, and the HTTP trigger on a SEPARATE router. The
dependency-correct continuation of S250 by exact symmetry: S249 gave the blobs a
route, S250 gave the AUDIO blob its transcription path, S251 gives the IMAGE blob
its caption / OCR path.

S250 landed the container-provable half of N.5 (voice transcription): the
sandboxed whisper.cpp orchestration (``opti_oignon/notes/transcription.py``), the
additive ``NotesStore.update_attachment`` write-back (the transcript / caption /
ocr legs), the opt-in ``transcribe`` group, and the separate
``notes_transcription_router`` at ``/api/notes/transcription`` (one POST). The
S250 carry-forward named N.6 (reuse the orchestration via the vision pipeline;
``update_attachment`` already carries the caption / ocr legs) as a natural next
slice.

This lot lands the container-provable half of N.6 (picture caption / OCR):

 - ``opti_oignon/notes/caption.py`` -- the orchestration, a sibling of
   ``transcription.py`` (NOT a second function bolted onto it). It fetches the
   manifest (per-user, kind must be ``image``), is FAIL-SECURE on the disposable
   bubblewrap floor (with no real bwrap it REFUSES, never a degraded tempdir for
   this file-touching post-processing), decrypts the blob in memory via
   ``NotesBlobStore.open``, copies the decrypted bytes INTO the disposable
   sandbox workspace (no plaintext temp on the host), runs the captioner inside
   the sandbox, requires human approval before the durable write-back, then calls
   ``update_attachment`` (the caption_text / ocr_text legs, already in place at
   S250 -- so ``notes_store.py`` is UNTOUCHED here), and destroys the sandbox in a
   ``finally`` (wiping the plaintext image). The captioner and the sandbox are
   injected seams; the live builder wires the vision/OCR tool + the real
   SandboxManager, the host-assured piece. The captioner returns a
   ``(caption_text, ocr_text)`` pair; only the legs it actually produced (the
   non-None ones) are written, so a leg the tool did not produce never blanks an
   existing value (the ``update_attachment`` no-blank property).
 - the opt-in ``vision`` dependency group in pyproject, off by default (the
   ``transcribe`` / ``veilid`` precedent: not pulled by the base install, not in
   ``all``).
 - ``opti_oignon/api/routes_notes_caption.py`` -- a SEPARATE router
   (``notes_caption_router`` at ``/api/notes/caption``), NOT folded into
   ``notes_transcription_router`` (so the S250 ``test_one_route_exact`` pin stays
   green -- the ``routes_note_actions`` / ``routes_notes_attachments`` /
   ``routes_notes_transcription`` precedent), a pure chain addition. One
   ``POST /{attachment_id}`` drives the orchestration and returns the structured
   result; the only HTTP error is the 503 availability guard.

What is host-assured and NOT in this suite: the live vision/OCR run inside a real
disposable bubblewrap (both the vision tooling and bwrap are absent in-container),
in NOTES_CAPTION_E2E_S251.md, never simulated here.

Nine families:

 1. caption.py source / structure.
 2. caption.py behavioural -- captioner_unavailable, fail-secure, not_found,
    not_image, blob_unavailable, preview (approve False), commit (approve True)
    writing BOTH legs, the partial commit (ocr-only) that does not blank caption,
    the both-None commit (no write), tool_failed, the disposable create+destroy
    lifecycle, no host plaintext temp.
 3. route source / structure.
 4. route behavioural (TestClient, injected real store + blobs, fake sandbox +
    captioner) -- preview, commit, fail-secure, per-user, 503 guard.
 5. registration -- app.py imports and includes notes_caption_router.
 6. schemas -- CaptionRequest / CaptionResultSchema load and validate.
 7. the opt-in vision dependency group, off by default (transcribe preserved).
 8. premise guards (green before AND after) -- update_attachment's caption / ocr
    legs over a real NotesStore (the write-back target), the N.1 attachment data
    layer, the NotesBlobStore surface, the S245 notes_router still five routes,
    the S249 notes_attachments_router still five routes, and the S250
    notes_transcription_router still EXACTLY ONE route (the fold-in guard: N.6 did
    not fold into the transcription router).
 9. AST / ASCII of the orchestration, the route, and this suite.

Red-before on the pristine S250 tree (no caption.py, no route, no caption
schemas, no vision group, no app.py registration): every family-1 / 3 / 6 / 7
contract pin and the family-2 / 4 behavioural pins FAIL (the read helpers return
empty strings so absence is a failure; the behavioural families load the modules
INSIDE the test so absence is an ImportError failure, never a collection error),
while the family-8 premise guards and the family-9 suite-parse pin PASS by design
(they pin pre-existing invariants this step relies on, including that
update_attachment's caption / ocr legs already work). The family-9 orchestration
and route AST pins FAIL on the pristine tree (the files are absent).

Isolation (the S243 / S247 / S249 / S250 idiom): the behavioural families load
the orchestration and the route under their dotted names into package-like stubs,
pre-loading the real (light) schemas dotted and the real notes_store / blob_store
dotted (their guarded relative imports resolve as submodules of the stub package,
with the REAL AES-256-GCM primitive), and stub routes_auth (the auth dep is
overridden per test). No fastapi/ollama package import is forced at collection,
and the real SandboxManager is never imported (the sandbox is a fake seam here);
the live vision / bwrap path is never reached.
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

import pytest  # noqa: F401  (parity with the suite idiom; fixtures via tmp_path)

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "opti_oignon"
ORCH_PATH = PKG / "notes" / "caption.py"
ROUTE_PATH = PKG / "api" / "routes_notes_caption.py"
SCHEMAS_PATH = PKG / "api" / "schemas.py"
APP_PATH = PKG / "api" / "app.py"
NOTES_STORE_PATH = PKG / "notes" / "notes_store.py"
BLOB_STORE_PATH = PKG / "notes" / "blob_store.py"
ROUTES_NOTES_PATH = PKG / "api" / "routes_notes.py"
ROUTES_ATTACH_PATH = PKG / "api" / "routes_notes_attachments.py"
TRANS_ORCH_PATH = PKG / "notes" / "transcription.py"
TRANS_ROUTE_PATH = PKG / "api" / "routes_notes_transcription.py"
PYPROJECT_PATH = REPO / "pyproject.toml"

EXPECTED_PREFIX = "/api/notes/caption"
# The single (path, method) route the trigger surface exposes.
EXPECTED_ROUTES = frozenset({("/api/notes/caption/{attachment_id}", "POST")})
# The S250 notes_transcription_router's single route (the fold-in guard).
TRANS_ROUTER_ROUTES = frozenset(
    {("/api/notes/transcription/{attachment_id}", "POST")}
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
# Isolation harness (the S243 / S247 / S249 / S250 idiom)
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
    """Load opti_oignon.notes.caption dotted into package stubs.

    On the pristine tree the module is absent and this raises ImportError INSIDE
    the calling test -- a failure, never a collection error.
    """
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.notes", PKG / "notes")
    _notes_modules()
    return _load_dotted("opti_oignon.notes.caption", ORCH_PATH)


def _stub_auth() -> None:
    if "opti_oignon.api.routes_auth" not in sys.modules:
        stub = types.ModuleType("opti_oignon.api.routes_auth")

        def _get_current_user() -> dict:  # pragma: no cover - overridden in tests
            return {"sub": None}

        stub._get_current_user = _get_current_user  # type: ignore[attr-defined]
        sys.modules["opti_oignon.api.routes_auth"] = stub


def _load_route():
    """Load routes_notes_caption under its dotted name into package stubs."""
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.api", PKG / "api")
    _ensure_pkg("opti_oignon.notes", PKG / "notes")
    _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
    _notes_modules()
    _load_orch()
    _stub_auth()
    return _load_dotted("opti_oignon.api.routes_notes_caption", ROUTE_PATH)


def _load_trans_route():
    """Load the S250 routes_notes_transcription (for the fold-in guard)."""
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.api", PKG / "api")
    _ensure_pkg("opti_oignon.notes", PKG / "notes")
    _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
    _notes_modules()
    _load_dotted("opti_oignon.notes.transcription", TRANS_ORCH_PATH)
    _stub_auth()
    return _load_dotted(
        "opti_oignon.api.routes_notes_transcription", TRANS_ROUTE_PATH
    )


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
# Fake seams (the sandbox and the captioner; the live ones are host-assured)
# ---------------------------------------------------------------------------


class _FakeSandbox:
    """A minimal SandboxManager-like fake: a real tmp workspace per session, the
    bwrap-availability signal, and a recorded create / destroy lifecycle.

    It exposes only the handful of members the orchestration drives. ``destroy``
    actually removes the workspace so the suite can assert the disposable
    teardown (the plaintext image does not linger)."""

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


def _ok_captioner(sandbox, session_id, input_name):
    """A fake captioner that confirms the decrypted bytes were written into the
    sandbox workspace, then returns a canned (caption, ocr) pair (the live
    vision/OCR tool is absent in-container)."""
    ws = Path(sandbox.get_active_workspace_path(session_id))
    assert (ws / input_name).exists(), "decrypted bytes not injected into sandbox"
    return ("a photo of a black cat on a sofa", "MEOW MEOW")


def _ocr_only_captioner(sandbox, session_id, input_name):
    """Produces only OCR text, no caption -- exercises the partial-leg write-back
    (the caption leg must stay untouched, not blanked)."""
    ws = Path(sandbox.get_active_workspace_path(session_id))
    assert (ws / input_name).exists()
    return (None, "INVOICE 12345")


def _empty_captioner(sandbox, session_id, input_name):
    """Produces neither leg -- the tool ran but yielded nothing to write."""
    return (None, None)


def _boom_captioner(sandbox, session_id, input_name):
    raise RuntimeError("vision/OCR tool crashed")


def _seed_image(store, blobs, *, sub: str, body: bytes = b"\x89PNG\r\n image-bytes"):
    """Create a note and an IMAGE attachment (manifest + sealed blob)."""
    note = store.add_note(title="picture", body_crdt=b"", user_id=sub)
    rec = store.add_attachment(
        note.id, "image", blob_ref="", mime="image/png", byte_size=len(body),
        user_id=sub,
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


def _safe_bytes(p: Path) -> bytes:
    try:
        return p.read_bytes()
    except OSError:
        return b""


# ---------------------------------------------------------------------------
# Family 1 -- caption.py source / structure
# ---------------------------------------------------------------------------


class TestOrchSource:
    def test_module_exists(self):
        assert ORCH_PATH.exists(), "opti_oignon/notes/caption.py missing"

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

    def test_decrypts_via_blob_open_and_writes_back(self):
        src = _read(ORCH_PATH)
        assert ".open(" in src
        assert "update_attachment" in src

    def test_writes_caption_and_ocr_legs(self):
        src = _read(ORCH_PATH)
        assert "caption_text" in src
        assert "ocr_text" in src

    def test_image_kind_gate(self):
        src = _read(ORCH_PATH)
        assert "not_image" in src

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
# Family 2 -- caption.py behavioural (the orchestration)
# ---------------------------------------------------------------------------


class TestOrchBehaviour:
    def test_captioner_unavailable_when_none(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_image(store, blobs, sub="alice")
        sbx = _FakeSandbox(tmp_path)
        res = orch.caption_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, captioner=None, approve=True,
        )
        assert res.refused is True
        assert res.reason == "captioner_unavailable"
        assert res.written_back is False
        assert sbx.created == []

    def test_fail_secure_when_no_bwrap(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_image(store, blobs, sub="alice")
        sbx = _FakeSandbox(tmp_path, bwrap_available=False)
        res = orch.caption_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, captioner=_ok_captioner, approve=True,
        )
        assert res.refused is True
        assert res.reason == "sandbox_unavailable"
        assert res.written_back is False
        assert sbx.created == []  # never even created a workspace
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text is None and got.ocr_text is None

    def test_not_found(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        sbx = _FakeSandbox(tmp_path)
        res = orch.caption_attachment(
            "ghost", user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, captioner=_ok_captioner, approve=True,
        )
        assert res.refused is True
        assert res.reason == "not_found"
        assert sbx.created == []

    def test_not_image(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        note = store.add_note(title="voice", body_crdt=b"", user_id="alice")
        rec = store.add_attachment(note.id, "audio", blob_ref="", user_id="alice")
        blobs.seal(rec.id, b"WAVE-bytes")
        sbx = _FakeSandbox(tmp_path)
        res = orch.caption_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, captioner=_ok_captioner, approve=True,
        )
        assert res.refused is True
        assert res.reason == "not_image"
        assert sbx.created == []

    def test_blob_unavailable(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path, master=False)  # keyless blob store
        note = store.add_note(title="pic", body_crdt=b"", user_id="alice")
        rec = store.add_attachment(note.id, "image", blob_ref="", user_id="alice")
        sbx = _FakeSandbox(tmp_path)
        res = orch.caption_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, captioner=_ok_captioner, approve=True,
        )
        assert res.refused is True
        assert res.reason == "blob_unavailable"
        assert sbx.created == []  # refused before creating a sandbox
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text is None and got.ocr_text is None

    def test_preview_does_not_write_back(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_image(store, blobs, sub="alice")
        sbx = _FakeSandbox(tmp_path)
        res = orch.caption_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, captioner=_ok_captioner, approve=False,
        )
        assert res.ok is True
        assert res.caption_text == "a photo of a black cat on a sofa"
        assert res.ocr_text == "MEOW MEOW"
        assert res.written_back is False
        # Preview only: the manifest row is NOT persisted.
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text is None and got.ocr_text is None
        # The disposable sandbox was created and torn down.
        assert sbx.created and sbx.destroyed == sbx.created

    def test_commit_writes_back_both_legs(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_image(store, blobs, sub="alice")
        sbx = _FakeSandbox(tmp_path)
        res = orch.caption_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, captioner=_ok_captioner, approve=True,
        )
        assert res.ok is True
        assert res.written_back is True
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text == "a photo of a black cat on a sofa"
        assert got.ocr_text == "MEOW MEOW"
        assert sbx.created and sbx.destroyed == sbx.created

    def test_commit_partial_leg_ocr_only_does_not_blank_caption(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        note = store.add_note(title="pic", body_crdt=b"", user_id="alice")
        rec = store.add_attachment(
            note.id, "image", blob_ref="", caption_text="pre-existing caption",
            user_id="alice",
        )
        blobs.seal(rec.id, b"\x89PNG image")
        sbx = _FakeSandbox(tmp_path)
        res = orch.caption_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, captioner=_ocr_only_captioner, approve=True,
        )
        assert res.ok is True
        assert res.written_back is True
        got = store.get_attachment(rec.id, user_id="alice")
        # The ocr leg is written; the caption leg is NOT blanked (no-blank).
        assert got.ocr_text == "INVOICE 12345"
        assert got.caption_text == "pre-existing caption"

    def test_commit_both_none_writes_nothing(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_image(store, blobs, sub="alice")
        sbx = _FakeSandbox(tmp_path)
        res = orch.caption_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, captioner=_empty_captioner, approve=True,
        )
        assert res.ok is True
        assert res.written_back is False  # the tool ran but produced no legs
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text is None and got.ocr_text is None
        assert sbx.created and sbx.destroyed == sbx.created

    def test_tool_failure_is_clean(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_image(store, blobs, sub="alice")
        sbx = _FakeSandbox(tmp_path)
        res = orch.caption_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, captioner=_boom_captioner, approve=True,
        )
        assert res.ok is False
        assert res.refused is False
        assert res.reason == "caption_failed"
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text is None and got.ocr_text is None
        # The sandbox is still torn down on the failure path (finally).
        assert sbx.destroyed == sbx.created and sbx.created

    def test_per_user_isolation_is_not_found(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_image(store, blobs, sub="alice")
        sbx = _FakeSandbox(tmp_path)
        # Bob cannot caption Alice's attachment.
        res = orch.caption_attachment(
            rec.id, user_id="bob", store=store, blobs=blobs,
            sandbox=sbx, captioner=_ok_captioner, approve=True,
        )
        assert res.refused is True
        assert res.reason == "not_found"
        assert sbx.created == []
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text is None and got.ocr_text is None

    def test_no_host_plaintext_temp_remains(self, tmp_path):
        orch = _load_orch()
        store, blobs = _stores(tmp_path)
        _, rec = _seed_image(
            store, blobs, sub="alice", body=b"SECRET-IMAGE-BYTES"
        )
        sbx = _FakeSandbox(tmp_path)
        orch.caption_attachment(
            rec.id, user_id="alice", store=store, blobs=blobs,
            sandbox=sbx, captioner=_ok_captioner, approve=True,
        )
        # After teardown, no workspace dir and no file carrying the plaintext.
        leaks = [p for p in tmp_path.rglob("*") if p.is_file()
                 and b"SECRET-IMAGE-BYTES" in _safe_bytes(p)]
        assert leaks == [], leaks


# ---------------------------------------------------------------------------
# Family 3 -- route source / structure
# ---------------------------------------------------------------------------


class TestRouteSource:
    def test_module_exists(self):
        assert ROUTE_PATH.exists(), "opti_oignon/api/routes_notes_caption.py missing"

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
        assert routes.notes_caption_router.prefix == EXPECTED_PREFIX

    def test_one_route_exact(self):
        routes = _load_route()
        assert _routes_of(routes.notes_caption_router) == EXPECTED_ROUTES

    def test_seams_present(self):
        routes = _load_route()
        assert hasattr(routes, "_notes_store_dep")
        assert hasattr(routes, "_blob_store_dep")
        assert hasattr(routes, "_sandbox_dep")
        assert hasattr(routes, "_captioner_dep")


# ---------------------------------------------------------------------------
# Family 4 -- route behavioural (TestClient)
# ---------------------------------------------------------------------------


def _build(tmp_path, *, sub: str = "alice", master: bool = True,
           bwrap: bool = True, captioner=_ok_captioner):
    routes = _load_route()
    store, blobs = _stores(tmp_path, master=master)
    sbx = _FakeSandbox(tmp_path, bwrap_available=bwrap)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    state = {"sub": sub}
    app = FastAPI()
    app.include_router(routes.notes_caption_router)
    app.dependency_overrides[routes._notes_store_dep] = lambda: store
    app.dependency_overrides[routes._blob_store_dep] = lambda: blobs
    app.dependency_overrides[routes._sandbox_dep] = lambda: sbx
    app.dependency_overrides[routes._captioner_dep] = lambda: captioner
    app.dependency_overrides[routes._get_current_user] = lambda: {"sub": state["sub"]}
    return TestClient(app), routes, store, blobs, sbx, state


class TestRouteBehaviour:
    def test_preview_returns_legs_no_write_back(self, tmp_path):
        client, routes, store, blobs, sbx, _ = _build(tmp_path)
        _, rec = _seed_image(store, blobs, sub="alice")
        r = client.post("/api/notes/caption/" + rec.id, json={"approve": False})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["caption_text"] == "a photo of a black cat on a sofa"
        assert body["ocr_text"] == "MEOW MEOW"
        assert body["written_back"] is False
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text is None and got.ocr_text is None

    def test_commit_writes_back(self, tmp_path):
        client, routes, store, blobs, sbx, _ = _build(tmp_path)
        _, rec = _seed_image(store, blobs, sub="alice")
        r = client.post("/api/notes/caption/" + rec.id, json={"approve": True})
        assert r.status_code == 200, r.text
        assert r.json()["written_back"] is True
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text == "a photo of a black cat on a sofa"
        assert got.ocr_text == "MEOW MEOW"

    def test_fail_secure_refusal(self, tmp_path):
        client, routes, store, blobs, sbx, _ = _build(tmp_path, bwrap=False)
        _, rec = _seed_image(store, blobs, sub="alice")
        r = client.post("/api/notes/caption/" + rec.id, json={"approve": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["refused"] is True
        assert body["reason"] == "sandbox_unavailable"

    def test_per_user_isolation_is_not_found(self, tmp_path):
        # Bob triggers caption on Alice's attachment: a refused not_found, never a
        # served caption and never a write-back.
        client, routes, store, blobs, sbx, state = _build(tmp_path, sub="alice")
        _, rec = _seed_image(store, blobs, sub="alice")
        state["sub"] = "bob"
        r = client.post("/api/notes/caption/" + rec.id, json={"approve": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["refused"] is True
        assert body["reason"] == "not_found"
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text is None and got.ocr_text is None

    def test_availability_guard_503(self, tmp_path):
        routes = _load_route()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(routes.notes_caption_router)
        app.dependency_overrides[routes._get_current_user] = lambda: {"sub": "alice"}
        orig = routes.FEATURE_AVAILABLE
        try:
            routes.FEATURE_AVAILABLE = False
            client = TestClient(app)
            r = client.post("/api/notes/caption/x", json={"approve": True})
            assert r.status_code == 503, r.text
        finally:
            routes.FEATURE_AVAILABLE = orig


# ---------------------------------------------------------------------------
# Family 5 -- registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_app_imports_router(self):
        assert "notes_caption_router" in _read(APP_PATH)

    def test_app_includes_router(self):
        src = _read(APP_PATH)
        assert "include_router(notes_caption_router)" in src


# ---------------------------------------------------------------------------
# Family 6 -- schemas
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_symbols_present(self):
        src = _read(SCHEMAS_PATH)
        assert "class CaptionRequest(" in src
        assert "class CaptionResultSchema(" in src

    def test_schemas_load_and_validate(self):
        _ensure_pkg("opti_oignon", PKG)
        _ensure_pkg("opti_oignon.api", PKG / "api")
        schemas = _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
        req = schemas.CaptionRequest(approve=True)
        assert req.approve is True
        # Default is the safe one: no durable write-back without approval.
        assert schemas.CaptionRequest().approve is False
        res = schemas.CaptionResultSchema(
            attachment_id="a1", ok=True, caption_text="a cat", ocr_text="MEOW",
            written_back=True,
        )
        assert res.ok is True and res.written_back is True
        assert res.caption_text == "a cat" and res.ocr_text == "MEOW"


# ---------------------------------------------------------------------------
# Family 7 -- the opt-in vision dependency group (off by default)
# ---------------------------------------------------------------------------


class TestOptInDependencyGroup:
    def _optional(self):
        import tomllib

        data = tomllib.loads(_read(PYPROJECT_PATH))
        return data["project"].get("optional-dependencies", {})

    def test_vision_group_exists(self):
        assert "vision" in self._optional()

    def test_vision_group_nonempty(self):
        assert len(self._optional()["vision"]) >= 1

    def test_vision_off_by_default(self):
        # Not pulled by the base install nor by the aggregate ``all`` extra (the
        # transcribe / veilid precedent: opt-in, off by default).
        import tomllib

        data = tomllib.loads(_read(PYPROJECT_PATH))
        base = data["project"].get("dependencies", [])
        joined = " ".join(base).lower()
        assert "vision" not in joined
        all_extra = " ".join(self._optional().get("all", [])).lower()
        assert "vision" not in all_extra

    def test_existing_groups_preserved(self):
        opt = self._optional()
        for group in ("llama", "auth", "sqlcipher", "veilid", "transcribe"):
            assert group in opt, group


# ---------------------------------------------------------------------------
# Family 8 -- premise guards (green before AND after)
# ---------------------------------------------------------------------------


class TestPremiseGuards:
    def test_update_attachment_caption_ocr_legs_work(self, tmp_path):
        # The write-back target N.6 reuses; already landed at S250 (so this is
        # green before AND after, and notes_store.py is untouched by S251).
        store, _ = _stores(tmp_path)
        note = store.add_note(title="pic", body_crdt=b"", user_id="alice")
        rec = store.add_attachment(note.id, "image", blob_ref="", user_id="alice")
        changed = store.update_attachment(
            rec.id, caption_text="a cat", ocr_text="MEOW", user_id="alice"
        )
        assert changed is True
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text == "a cat"
        assert got.ocr_text == "MEOW"

    def test_update_attachment_no_blank_on_omitted_leg(self, tmp_path):
        store, _ = _stores(tmp_path)
        note = store.add_note(title="pic", body_crdt=b"", user_id="alice")
        rec = store.add_attachment(
            note.id, "image", blob_ref="", caption_text="keep me", user_id="alice"
        )
        # Writing only ocr_text must not blank caption_text.
        store.update_attachment(rec.id, ocr_text="text", user_id="alice")
        got = store.get_attachment(rec.id, user_id="alice")
        assert got.caption_text == "keep me"
        assert got.ocr_text == "text"

    def test_n1_attachment_data_layer_intact(self):
        src = _read(NOTES_STORE_PATH)
        assert "def add_attachment(" in src
        assert "def get_attachment(" in src
        assert "def list_attachments(" in src
        assert "def update_attachment(" in src
        assert "ATTACHMENT_KINDS" in src

    def test_blob_store_surface_intact(self):
        src = _read(BLOB_STORE_PATH)
        assert "def seal(" in src
        assert "def open(" in src
        assert "class NotesBlobUnavailable" in src

    def test_s245_notes_router_still_five_routes(self):
        # The caption trigger is a SEPARATE router; notes_router is untouched.
        _ensure_pkg("opti_oignon", PKG)
        _ensure_pkg("opti_oignon.api", PKG / "api")
        _ensure_pkg("opti_oignon.notes", PKG / "notes")
        _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
        _notes_modules()
        _stub_auth()
        rn = _load_dotted("opti_oignon.api.routes_notes", ROUTES_NOTES_PATH)
        assert _routes_of(rn.notes_router) == NOTES_ROUTER_ROUTES

    def test_s249_attachment_router_still_five_routes(self):
        _ensure_pkg("opti_oignon", PKG)
        _ensure_pkg("opti_oignon.api", PKG / "api")
        _ensure_pkg("opti_oignon.notes", PKG / "notes")
        _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
        _notes_modules()
        _stub_auth()
        ra = _load_dotted(
            "opti_oignon.api.routes_notes_attachments", ROUTES_ATTACH_PATH
        )
        assert _routes_of(ra.notes_attachments_router) == ATTACH_ROUTER_ROUTES

    def test_s250_transcription_router_still_one_route(self):
        # The fold-in guard: N.6 did NOT fold into the transcription router, so
        # its single-route pin stays exactly one route.
        rt = _load_trans_route()
        assert _routes_of(rt.notes_transcription_router) == TRANS_ROUTER_ROUTES


# ---------------------------------------------------------------------------
# Family 9 -- AST / ASCII
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
