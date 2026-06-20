"""S256 -- N.9, the phone-app note sync contract: the container-provable
contract / seam half of the per-item mobile-allowed flag
(MOBILE_THREAT_MODEL.md section 3: notes cross to the phone only behind a
per-item flag, never wholesale).

The slice under test:

- The ``note`` table gains an additive ``mobile_allowed INTEGER NOT NULL
  DEFAULT 0`` column (the secure default: nothing crosses until the user opts
  the item in), with the guarded ALTER migration for pre-N.9 databases (the
  AU-06 / peers-registry idiom).
- The store API: ``NoteRecord.mobile_allowed``, the dedicated
  ``set_mobile_allowed`` setter, and the fail-secure ``is_mobile_allowed``
  reader. The flag is deliberately NOT in ``_UPDATABLE_COLUMNS``: the generic
  update path -- and therefore the gated ``manage_notes`` tool -- can never
  flip it (decision N9-D3: the flag is a human trust decision at the desktop).
- The peer registry gains an additive nullable ``device_class`` column (the
  cas 7 grant migration shape). NULL stays the grandfathered desktop class
  (every pre-N.9 row is a desktop by construction); ``"phone"`` marks a
  phone-class peer; the marker is a local trust decision never changed by a
  re-pair.
- The sync responder filters AT SERVE (decision N9-D1): the journal stays
  whole (desktop-to-desktop N.8 note sync is untouched) and ``serve_request``
  drops NOTE records toward a phone-class asker unless a LIVE lookup through
  the injected note gate affirms the note's current flag. Fail-secure
  throughout (decision N9-D2): no resolvable gate, an unknown note, a raised
  error, an unknown identified peer -- anything indeterminable -- EXCLUDES
  the record; an absent or unreadable flag means NOT allowed. With an EMPTY
  ``peer_id`` (today's production posture, the private route as the implicit
  authenticator) the filter, exactly like PAIR-02, acts only where an
  identity is actually supplied; the identity binding at the serve seam is
  the mobile cycle's host-assured half.
- The route surface: the flag rides the EXISTING legs (``NoteSchema`` carries
  it, ``NoteUpdateRequest`` accepts it optionally on the PATCH) through the
  dedicated setter only -- no new route leg, so the s245 five-routes-exact
  pin survives by construction.
- The runbook NOTES_MOBILE_SYNC_N9_S256.md labels the host-assured remainder
  (the live phone round, the pairing payload carrying the class, the identity
  binding at serve, the republish-on-opt-in wiring) and is never simulated
  in-container.

Design-green on the pristine S255 tree (must PASS before the edits): the
generic-update guard (an unknown column already raises), the five-routes-exact
reassertion, the empty-peer-id and no-class serve postures (N.8 behaviour
preserved), the PAIR-02 pending refusal, the SENSITIVE_KINDS skill-only
reassertion, the tool-source negative pin, the retained roadmap N.1 roll, the
held version, and the AST / ASCII structure family. Everything else is RED
before by assertion (never a collection error).

Version is HELD at 3.12.0 (a contract / seam slice whose live round is
host-assured never bumps). The auth core (auth.py, auth_2fa.py,
emergency_stop.py), signing.py, and pairing.py are untouched by this bloc.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import hmac as hmac_mod
import importlib
import importlib.util
import sqlite3
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "opti_oignon"
VEILID = PKG / "veilid"

NOTES_STORE_SRC = PKG / "notes" / "notes_store.py"
PEERS_SRC = VEILID / "peers.py"
ENGINE_SRC = VEILID / "sync_engine.py"
TOOLS_SRC = PKG / "agent" / "tools.py"
SCHEMAS_PATH = PKG / "api" / "schemas.py"
ROUTES_NOTES_PATH = PKG / "api" / "routes_notes.py"
VERSION_PATH = PKG / "__version__.py"
RUNBOOK_PATH = ROOT / "NOTES_MOBILE_SYNC_N9_S256.md"
ROADMAP_PATH = ROOT / "NOTES_FEATURE_ROADMAP.md"

EXPECTED_PREFIX = "/api/notes"
EXPECTED_ROUTES = frozenset(
    {
        ("/api/notes", "GET"),
        ("/api/notes", "POST"),
        ("/api/notes/{note_id}", "GET"),
        ("/api/notes/{note_id}", "PATCH"),
        ("/api/notes/{note_id}", "DELETE"),
    }
)

# The pre-N.9 note table DDL (the S243 shape as shipped through v3.12.0),
# used to build an old-format database the migration must upgrade in place.
_OLD_NOTE_DDL = """
CREATE TABLE IF NOT EXISTS note (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'local',
    title TEXT NOT NULL DEFAULT '',
    body_crdt BLOB NOT NULL DEFAULT x'',
    tags TEXT NOT NULL DEFAULT '[]',
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS attachment (
    id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'local',
    kind TEXT NOT NULL,
    blob_ref TEXT NOT NULL,
    mime TEXT NOT NULL DEFAULT '',
    bytes INTEGER NOT NULL DEFAULT 0,
    nonce TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    transcript_text TEXT,
    caption_text TEXT,
    ocr_text TEXT
);
"""


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
# Isolation harness (the S243 lesson, the S244 / S245 idiom)
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
    return _isolated_flat("s256_notes_store_iso", "notes/notes_store.py")


def _make_store(tmp_path, single_user_mode: bool = True):
    return _store_module().NotesStore(
        root=str(tmp_path), single_user_mode=single_user_mode
    )


def _load_route():
    """Load routes_notes dotted (the s245 loader, s256-scoped)."""
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.api", PKG / "api")
    _ensure_pkg("opti_oignon.notes", PKG / "notes")
    _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
    if "opti_oignon.api.routes_auth" not in sys.modules:
        stub = types.ModuleType("opti_oignon.api.routes_auth")

        def _get_current_user() -> dict:  # pragma: no cover - overridden
            return {"sub": None}

        stub._get_current_user = _get_current_user  # type: ignore[attr-defined]
        sys.modules["opti_oignon.api.routes_auth"] = stub
    sys.modules["opti_oignon.notes.notes_store"] = _store_module()
    return _load_dotted("opti_oignon.api.routes_notes", ROUTES_NOTES_PATH)


def _build(tmp_path, single_user_mode: bool = True):
    """A bare app over the route with the store and auth injected."""
    routes = _load_route()
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    store = _make_store(tmp_path, single_user_mode=single_user_mode)
    app = FastAPI()
    app.include_router(routes.notes_router)
    state = {"sub": "user_a"}
    app.dependency_overrides[routes._notes_store_dep] = lambda: store
    app.dependency_overrides[routes._get_current_user] = lambda: {
        "sub": state["sub"]
    }
    client = TestClient(app)
    return client, store, routes, state


# ---------------------------------------------------------------------------
# Veilid harness (the s252 idiom: real modules over light stubs)
# ---------------------------------------------------------------------------

_MODE = {"fn": lambda: "daily"}
_AUDIT: dict = {"events": []}


def _set_mode(value: str = "daily") -> None:
    def _gm() -> str:
        return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


def _record_audit(**kwargs):
    _AUDIT["events"].append(kwargs)


def _ensure_stubs() -> None:
    for name, sub in (
        ("opti_oignon", PKG),
        ("opti_oignon.veilid", VEILID),
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(sub)]
            sys.modules[name] = mod
    if "opti_oignon.security_mode" not in sys.modules:
        sm = types.ModuleType("opti_oignon.security_mode")
        sm.get_current_mode = _MODE["fn"]  # type: ignore[attr-defined]
        sys.modules["opti_oignon.security_mode"] = sm
    if "opti_oignon.signed_audit_log" not in sys.modules:
        al = types.ModuleType("opti_oignon.signed_audit_log")
        al.chain_log = _record_audit  # type: ignore[attr-defined]
        sys.modules["opti_oignon.signed_audit_log"] = al


def _veilid() -> dict:
    """The real veilid modules, imported lazily inside the calling test."""
    _ensure_stubs()
    mods = {
        "signing": importlib.import_module("opti_oignon.veilid.signing"),
        "change_feed": importlib.import_module(
            "opti_oignon.veilid.change_feed"
        ),
        "peers": importlib.import_module("opti_oignon.veilid.peers"),
        "records": importlib.import_module("opti_oignon.veilid.records"),
        "protocol": importlib.import_module("opti_oignon.veilid.protocol"),
        "ledger": importlib.import_module(
            "opti_oignon.veilid.deferred_ledger"
        ),
        "engine": importlib.import_module("opti_oignon.veilid.sync_engine"),
    }
    return mods


class FakeSigner:
    """A deterministic HMAC-SHA256 'signature' scheme keyed per device."""

    def __init__(self, secret: bytes) -> None:
        self._secret = secret

    def public_key(self) -> bytes:
        return hmac_mod.new(self._secret, b"pub", hashlib.sha256).digest()

    def sign(self, data: bytes) -> bytes:
        return hmac_mod.new(
            self._secret + self.public_key(), data, hashlib.sha256
        ).digest()

    def verify(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        secret = _PUB_REGISTRY.get(public_key)
        expected_like = hmac_mod.new(
            (secret or b"\x00") + public_key, data, hashlib.sha256
        ).digest()
        return hmac_mod.compare_digest(expected_like, signature)


_PUB_REGISTRY: dict[bytes, bytes] = {}


def _make_signer(seed: str) -> FakeSigner:
    secret = hashlib.sha256(seed.encode()).digest()
    s = FakeSigner(secret)
    _PUB_REGISTRY[s.public_key()] = secret
    return s


@pytest.fixture()
def serve_world(tmp_path):
    """A serving engine with a journaled note and conversation, plus peers.

    Yields ``(mods, build_engine, request, ids)`` where ``build_engine`` takes
    the optional engine kwargs (``note_gate`` among them) and returns a fresh
    engine over the SAME feed and peer store, so one journal serves every
    posture under test. The asking peers registered on the server:

    - ``phone-1``: confirmed, marked phone-class.
    - ``desk-1``: confirmed, NO class (the grandfathered NULL).
    - ``pending-1``: a fresh pending pairing (PAIR-02).
    """
    mods = _veilid()
    _set_mode("daily")
    sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]
    _AUDIT["events"].clear()
    feed = mods["change_feed"].ChangeFeed(root=tmp_path / "server")
    store = mods["peers"].PeerStore(root=tmp_path / "server")
    ledger = mods["ledger"].DeferredLedger(root=tmp_path / "server")
    signer = _make_signer("server-seed")

    def build_engine(**kwargs):
        return mods["engine"].SyncEngine(
            device="server",
            feed=feed,
            store=store,
            signer=signer,
            ledger=ledger,
            **kwargs,
        )

    base = build_engine()
    base.publish_note("n1", {"body": "opaque"}, clock=1)
    base.publish_conversation("c1", {"title": "t"}, clock=1)

    store.add_peer("desk-1", "rk-desk")
    store.add_peer("phone-1", "rk-phone")
    if hasattr(store, "set_device_class"):
        store.set_device_class("phone-1", "phone")
    store.add_peer("pending-1", "rk-pending", pending=True)

    request = mods["protocol"].build_delta_request(device="asker", watermark=0)
    yield mods, build_engine, request, {"note": "n1", "conversation": "c1"}
    _AUDIT["events"].clear()


def _kinds(batch: dict) -> list[str]:
    return [w.get("kind") for w in batch.get("records", [])]


# ---------------------------------------------------------------------------
# Family 1 -- the note schema and its migration (store source + behaviour)
# ---------------------------------------------------------------------------


class TestNoteSchemaColumn:
    def test_create_table_carries_mobile_allowed(self):
        src = _read(NOTES_STORE_SRC)
        assert "mobile_allowed INTEGER NOT NULL DEFAULT 0" in src

    def test_guarded_migration_present(self):
        src = _read(NOTES_STORE_SRC)
        assert "ADD COLUMN" in src
        assert "mobile_allowed" in src
        assert "PRAGMA table_info" in src

    def test_prior_columns_intact(self):
        # Design-green: the S243 presence pins survive the additive column.
        src = _read(NOTES_STORE_SRC)
        for needle in (
            "CREATE TABLE IF NOT EXISTS note",
            "body_crdt",
            "tags",
            "deleted",
            "WHERE id = ? AND user_id = ?",
        ):
            assert needle in src, needle

    def test_migration_upgrades_a_pre_n9_database(self, tmp_path):
        # Build the old-shape database by hand, then open it through the
        # store: the guarded ALTER must add the column in place and the row
        # must read fail-secure False.
        db = tmp_path / "notes.db"
        conn = sqlite3.connect(str(db))
        conn.executescript(_OLD_NOTE_DDL)
        conn.execute(
            "INSERT INTO note (id, user_id, title, created_at, updated_at) "
            "VALUES ('old1', 'local', 'pre-n9', 't0', 't0')"
        )
        conn.commit()
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(note)").fetchall()
        }
        conn.close()
        assert "mobile_allowed" not in cols  # the premise: a true old shape

        store = _make_store(tmp_path)
        try:
            cols_after = {
                row[1]
                for row in sqlite3.connect(str(db))
                .execute("PRAGMA table_info(note)")
                .fetchall()
            }
            assert "mobile_allowed" in cols_after
            rec = store.get_note("old1")
            assert rec is not None
            assert rec.mobile_allowed is False
            assert store.is_mobile_allowed("old1") is False
            assert store.set_mobile_allowed("old1", True) is True
            assert store.is_mobile_allowed("old1") is True
        finally:
            store.close()


class TestStoreFlagBehaviour:
    def test_new_note_defaults_not_allowed(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            rec = store.add_note("fresh")
            assert rec.mobile_allowed is False
            assert store.get_note(rec.id).mobile_allowed is False
            assert store.is_mobile_allowed(rec.id) is False
        finally:
            store.close()

    def test_set_and_clear_roundtrip(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            rec = store.add_note("toggle")
            assert store.set_mobile_allowed(rec.id, True) is True
            assert store.get_note(rec.id).mobile_allowed is True
            assert store.is_mobile_allowed(rec.id) is True
            assert store.set_mobile_allowed(rec.id, False) is True
            assert store.get_note(rec.id).mobile_allowed is False
            assert store.is_mobile_allowed(rec.id) is False
        finally:
            store.close()

    def test_unknown_note_is_fail_secure(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            assert store.set_mobile_allowed("ghost", True) is False
            assert store.is_mobile_allowed("ghost") is False
        finally:
            store.close()

    def test_tombstone_is_fail_secure(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            rec = store.add_note("doomed")
            assert store.set_mobile_allowed(rec.id, True) is True
            assert store.delete_note(rec.id) is True
            # A tombstoned note never reads allowed and never updates.
            assert store.is_mobile_allowed(rec.id) is False
            assert store.set_mobile_allowed(rec.id, True) is False
        finally:
            store.close()

    def test_per_user_scope(self, tmp_path):
        store = _make_store(tmp_path, single_user_mode=False)
        try:
            rec = store.add_note("alice note", user_id="alice")
            assert store.set_mobile_allowed(rec.id, True, user_id="bob") is False
            assert store.is_mobile_allowed(rec.id, user_id="bob") is False
            assert store.is_mobile_allowed(rec.id, user_id="alice") is False
            assert (
                store.set_mobile_allowed(rec.id, True, user_id="alice") is True
            )
            assert store.is_mobile_allowed(rec.id, user_id="alice") is True
            assert store.is_mobile_allowed(rec.id, user_id="bob") is False
        finally:
            store.close()

    def test_flag_excluded_from_generic_update(self, tmp_path):
        # Design-green (the N9-D3 structural guard): the flag is not an
        # updatable column, so the generic kwargs path -- the one the gated
        # manage_notes tool rides -- raises rather than flips it. This holds
        # on the pristine tree (an unknown column already raises) and MUST
        # keep holding after the column exists.
        store = _make_store(tmp_path)
        try:
            rec = store.add_note("guarded")
            with pytest.raises(ValueError):
                store.update_note(rec.id, mobile_allowed=True)
            assert store.is_mobile_allowed(rec.id) is False
        finally:
            store.close()

    def test_updatable_columns_never_name_the_flag(self):
        mod = _store_module()
        assert "mobile_allowed" not in mod.NotesStore._UPDATABLE_COLUMNS


# ---------------------------------------------------------------------------
# Family 2 -- the peer registry device class
# ---------------------------------------------------------------------------


class TestPeerDeviceClassSource:
    def test_column_in_create_table(self):
        src = _read(PEERS_SRC)
        assert "device_class TEXT" in src

    def test_guarded_alter_present(self):
        src = _read(PEERS_SRC)
        assert "ADD COLUMN device_class" in src

    def test_allowlist_constants_present(self):
        src = _read(PEERS_SRC)
        assert "DEVICE_CLASS_PHONE" in src
        assert "DEVICE_CLASSES" in src
        # The grant columns stay (design-green reassertion of the s235 pins).
        assert "remote_chat_grant" in src
        assert "rag_subgrant" in src


class TestPeerDeviceClassBehaviour:
    def test_fresh_peer_has_no_class(self, tmp_path):
        mods = _veilid()
        store = mods["peers"].PeerStore(root=tmp_path)
        rec = store.add_peer("p1", "rk1")
        assert rec.device_class is None
        assert store.get_peer("p1").device_class is None

    def test_set_and_clear_class(self, tmp_path):
        mods = _veilid()
        store = mods["peers"].PeerStore(root=tmp_path)
        store.add_peer("p1", "rk1")
        assert store.set_device_class("p1", "phone") is True
        assert store.get_peer("p1").device_class == "phone"
        assert store.set_device_class("p1", None) is True
        assert store.get_peer("p1").device_class is None

    def test_unknown_peer_returns_false(self, tmp_path):
        mods = _veilid()
        store = mods["peers"].PeerStore(root=tmp_path)
        assert store.set_device_class("ghost", "phone") is False

    def test_invalid_class_raises(self, tmp_path):
        mods = _veilid()
        store = mods["peers"].PeerStore(root=tmp_path)
        store.add_peer("p1", "rk1")
        with pytest.raises(ValueError):
            store.set_device_class("p1", "toaster")

    def test_re_pair_preserves_class(self, tmp_path):
        mods = _veilid()
        store = mods["peers"].PeerStore(root=tmp_path)
        store.add_peer("p1", "rk1")
        store.set_device_class("p1", "phone")
        store.add_peer("p1", "rk1-rotated", label="renamed")
        rec = store.get_peer("p1")
        assert rec.routing_key == "rk1-rotated"
        assert rec.device_class == "phone"


# ---------------------------------------------------------------------------
# Family 3 -- the serve-side filter (N9-D1 / N9-D2)
# ---------------------------------------------------------------------------


class TestServeFilter:
    def test_ctor_accepts_note_gate(self, serve_world):
        mods, build_engine, request, ids = serve_world
        eng = build_engine(note_gate=lambda nid: False)
        assert eng is not None

    def test_phone_class_excludes_disallowed_note(self, serve_world):
        mods, build_engine, request, ids = serve_world
        eng = build_engine(note_gate=lambda nid: False)
        batch = eng.serve_request(request, peer_id="phone-1")
        kinds = _kinds(batch)
        assert "conversation" in kinds
        assert "note" not in kinds

    def test_phone_class_passes_allowed_note(self, serve_world):
        mods, build_engine, request, ids = serve_world
        eng = build_engine(note_gate=lambda nid: nid == ids["note"])
        kinds = _kinds(eng.serve_request(request, peer_id="phone-1"))
        assert "note" in kinds
        assert "conversation" in kinds

    def test_gate_lookup_is_live_not_snapshot(self, serve_world):
        mods, build_engine, request, ids = serve_world
        flags = {ids["note"]: False}
        eng = build_engine(note_gate=lambda nid: flags.get(nid, False))
        assert "note" not in _kinds(eng.serve_request(request, peer_id="phone-1"))
        flags[ids["note"]] = True
        assert "note" in _kinds(eng.serve_request(request, peer_id="phone-1"))

    def test_gate_error_is_fail_secure(self, serve_world):
        mods, build_engine, request, ids = serve_world

        def broken(nid: str) -> bool:
            raise RuntimeError("gate down")

        eng = build_engine(note_gate=broken)
        batch = eng.serve_request(request, peer_id="phone-1")
        assert "note" not in _kinds(batch)
        assert "conversation" in _kinds(batch)

    def test_no_resolvable_gate_is_fail_secure(self, serve_world):
        # No injected gate, and the journaled note id is unknown to any
        # process-default store: an absent or unreadable flag means NOT
        # allowed, so the note never leaves toward the phone.
        mods, build_engine, request, ids = serve_world
        eng = build_engine()
        batch = eng.serve_request(request, peer_id="phone-1")
        assert "note" not in _kinds(batch)
        assert "conversation" in _kinds(batch)

    def test_unknown_identified_peer_is_restrictive(self, serve_world):
        mods, build_engine, request, ids = serve_world
        eng = build_engine(note_gate=lambda nid: False)
        batch = eng.serve_request(request, peer_id="never-registered")
        assert "note" not in _kinds(batch)
        assert "conversation" in _kinds(batch)

    def test_high_water_untouched_by_filtering(self, serve_world):
        mods, build_engine, request, ids = serve_world
        eng = build_engine(note_gate=lambda nid: False)
        filtered = eng.serve_request(request, peer_id="phone-1")
        unfiltered = eng.serve_request(request, peer_id="desk-1")
        assert int(filtered.get("high_water", -1)) == int(
            unfiltered.get("high_water", -2)
        )
        assert int(filtered.get("high_water", 0)) >= 2

    def test_filter_is_audited(self, serve_world):
        mods, build_engine, request, ids = serve_world
        eng = build_engine(note_gate=lambda nid: False)
        _AUDIT["events"].clear()
        eng.serve_request(request, peer_id="phone-1")
        serve_events = [
            e for e in _AUDIT["events"] if e.get("event") == "sync_serve"
        ] or _AUDIT["events"]
        assert any(
            int(e.get("notes_filtered", 0)) >= 1 for e in serve_events
        ), serve_events

    def test_desktop_class_is_unfiltered(self, serve_world):
        # Design-green: a NULL-class peer is the grandfathered desktop and
        # keeps the whole N.8 note sync (built WITHOUT the gate kwarg so the
        # pin also holds on the pristine tree).
        mods, build_engine, request, ids = serve_world
        eng = build_engine()
        kinds = _kinds(eng.serve_request(request, peer_id="desk-1"))
        assert "note" in kinds
        assert "conversation" in kinds

    def test_empty_peer_id_posture_unchanged(self, serve_world):
        # Design-green, stated honestly: with no supplied identity (today's
        # production posture; the private route is the implicit
        # authenticator) there is nothing to key the filter on, exactly like
        # PAIR-02. The identity binding at the serve seam is the mobile
        # cycle's host-assured half.
        mods, build_engine, request, ids = serve_world
        eng = build_engine()
        kinds = _kinds(eng.serve_request(request))
        assert "note" in kinds
        assert "conversation" in kinds

    def test_pending_peer_still_refused(self, serve_world):
        # Design-green: PAIR-02 is untouched by the filter.
        mods, build_engine, request, ids = serve_world
        eng = build_engine()
        with pytest.raises(mods["engine"].PeerNotConfirmed):
            eng.serve_request(request, peer_id="pending-1")

    def test_sensitive_kinds_still_skill_only(self):
        # Design-green reassertion: the filter is NOT a sensitive-kind; the
        # human gate set is unchanged.
        mods = _veilid()
        records = mods["records"]
        assert mods["engine"].SENSITIVE_KINDS == frozenset(
            {records.RecordKind.SKILL.value}
        )


# ---------------------------------------------------------------------------
# Family 4 -- the route surface (no new leg; the dedicated setter only)
# ---------------------------------------------------------------------------


class TestRouteSurface:
    def test_schema_models_carry_flag(self):
        _ensure_pkg("opti_oignon", PKG)
        _ensure_pkg("opti_oignon.api", PKG / "api")
        schemas = _load_dotted("opti_oignon.api.schemas", SCHEMAS_PATH)
        note = schemas.NoteSchema(id="n1", title="t")
        assert note.mobile_allowed is False
        upd = schemas.NoteUpdateRequest(mobile_allowed=True)
        assert upd.mobile_allowed is True
        bare = schemas.NoteUpdateRequest()
        assert bare.mobile_allowed is None

    def test_create_shows_flag_false(self, tmp_path):
        client, _store, _routes, _state = _build(tmp_path)
        data = client.post("/api/notes", json={"title": "x"}).json()
        assert data["mobile_allowed"] is False

    def test_patch_flag_roundtrip(self, tmp_path):
        client, store, _routes, _state = _build(tmp_path)
        nid = client.post("/api/notes", json={"title": "x"}).json()["id"]
        r = client.patch(f"/api/notes/{nid}", json={"mobile_allowed": True})
        assert r.status_code == 200, r.text
        assert r.json()["mobile_allowed"] is True
        assert store.is_mobile_allowed(nid) is True
        assert client.get(f"/api/notes/{nid}").json()["mobile_allowed"] is True
        r2 = client.patch(f"/api/notes/{nid}", json={"mobile_allowed": False})
        assert r2.status_code == 200, r2.text
        assert r2.json()["mobile_allowed"] is False
        assert store.is_mobile_allowed(nid) is False

    def test_patch_without_flag_preserves_it(self, tmp_path):
        client, store, _routes, _state = _build(tmp_path)
        nid = client.post("/api/notes", json={"title": "x"}).json()["id"]
        client.patch(f"/api/notes/{nid}", json={"mobile_allowed": True})
        r = client.patch(f"/api/notes/{nid}", json={"title": "renamed"})
        assert r.status_code == 200, r.text
        assert r.json()["title"] == "renamed"
        assert r.json()["mobile_allowed"] is True
        assert store.is_mobile_allowed(nid) is True

    def test_five_routes_exact_unchanged(self):
        # Design-green by construction: the flag rides the existing PATCH;
        # no new leg, so the s245 exact set survives.
        routes = _load_route()
        found = set()
        for r in routes.notes_router.routes:
            path = getattr(r, "path", None)
            methods = getattr(r, "methods", None) or set()
            for m in methods:
                if m in ("GET", "POST", "PATCH", "DELETE", "PUT"):
                    found.add((path, m))
        assert found == set(EXPECTED_ROUTES), found

    def test_route_uses_dedicated_setter(self):
        src = _read(ROUTES_NOTES_PATH)
        assert "set_mobile_allowed(" in src

    def test_llm_tool_surface_never_names_flag(self):
        # Design-green negative pin (N9-D3): the gated manage_notes tool
        # never names the flag -- absent on pristine and it must STAY absent.
        assert "mobile_allowed" not in _read(TOOLS_SRC)


# ---------------------------------------------------------------------------
# Family 5 -- the runbook (host-assured half) and the roadmap roll
# ---------------------------------------------------------------------------


class TestRunbookAndRoadmap:
    def test_runbook_exists_and_is_ascii(self):
        raw = _read(RUNBOOK_PATH)
        assert raw != ""
        assert raw == raw.encode("ascii", errors="ignore").decode("ascii")

    def test_runbook_names_the_seam_and_decisions(self):
        text = _flat(_read(RUNBOOK_PATH))
        assert "filter-at-serve" in text
        assert "N9-D1" in text
        assert "N9-D2" in text
        assert "N9-D3" in text

    def test_runbook_states_fail_secure(self):
        text = _flat(_read(RUNBOOK_PATH))
        assert "an absent or unreadable flag means NOT allowed" in text

    def test_runbook_labels_host_assured_items(self):
        text = _flat(_read(RUNBOOK_PATH))
        assert "host-assured" in text
        assert "never simulated in-container" in text
        assert "pairing payload" in text
        assert "republish" in text

    def test_runbook_holds_version_and_auth_core(self):
        text = _flat(_read(RUNBOOK_PATH))
        assert "held at 3.12.0" in text
        assert "the auth core (auth.py, auth_2fa.py)" in text
        assert "emergency_stop.py" in text
        assert "edit-free" in text

    def test_roadmap_n9_rolled_to_seam_landed(self):
        text = _flat(_read(ROADMAP_PATH))
        assert "N.9" in text
        assert "contract / seam half LANDED at S256" in text

    def test_roadmap_retains_prior_rolls(self):
        # Design-green: the N.1 roll (the s243 pin) survives the N.9 roll.
        text = _flat(_read(ROADMAP_PATH))
        assert "LANDED at S243" in text


# ---------------------------------------------------------------------------
# Family 6 -- structure: held version, AST, ASCII
# ---------------------------------------------------------------------------


class TestStructure:
    def test_version_held(self):
        # Design-green: a contract / seam slice whose live round is
        # host-assured never bumps.
        ns: dict = {}
        exec(_read(VERSION_PATH), ns)
        assert ns.get("__version__") == "3.12.0"

    def test_touched_sources_parse(self):
        for path in (
            NOTES_STORE_SRC,
            PEERS_SRC,
            ENGINE_SRC,
            SCHEMAS_PATH,
            ROUTES_NOTES_PATH,
        ):
            ast.parse(_read(path))

    def test_this_suite_parses_and_is_ascii(self):
        raw = _read(Path(__file__))
        ast.parse(raw)
        assert raw == raw.encode("ascii", errors="ignore").decode("ascii")
