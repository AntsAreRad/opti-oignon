#!/usr/bin/env python3
"""S260 -- the UI toggles lot (N.9: the last container-provable half).

Two trust controls become visible and operable at the desktop, both riding
seams earlier sessions landed, with NO new notes route and NO tool surface:

- The per-note ``mobile_allowed`` toggle (the S256 flag) in the Notes UI:
  the frontend client and store gain the field and a ``toggleMobileAllowed``
  action that rides the EXISTING PATCH leg (the s245 five-routes-exact pin
  tolerates no new leg; the backend route already flips the flag ONLY
  through the store's dedicated setter, decision N9-D3), and NotesPanel
  shows the control in the editor plus a list marker. CONFIRMED posture,
  never optimistic: the rendered state comes only from the server-returned
  record (the store replaces the row from the PATCH response), the control
  is disabled while the request is in flight, and a failure surfaces a
  toast while the display stays at the server truth -- a trust flag is
  never shown in a state the backend has not confirmed.
- The pairing panel surfaces ``device_class`` (the S258 read half: the
  class rides ``_peer_to_dict``, the peers list, and the pending-pairings
  payload) next to the confirmation code and on each paired device, and
  gains the class SETTER the S258 design anticipated as "the control
  surface": a NEW auth-gated sync leg
  ``POST /api/sync/peers/{peer_id}/device-class`` on the relabel pattern --
  covered by the router-level ``_auth_dep``, a strict pure normaliser
  (only ``phone`` / ``desktop`` / a clear pass; free text raises and the
  route maps the ``ValueError`` to a 400 BEFORE anything reaches the
  store), lookup-before-write (404 ``Peer not paired`` without probing the
  engine), and the engine's audited ``set_device_class`` as the domain
  seam so the flip lands in the hash-chain audit log. The s235 sync-route
  pins are presence-only substrings and tolerate the addition; no
  exact-route-set pin exists on the sync router (verified at the gate).

Red-before contract: every test that touches the new surface asserts the
surface exists (getattr / signature / non-empty-read guards) BEFORE calling
or indexing it, so on the pristine S259 tree each red is an AssertionError,
never a collection error, an AttributeError, or a TypeError (the
S257-S259 assert-before-call idiom). The design-green set -- the
reassertions, the carried negative pins, the panel hygiene invariants, the
structure family -- is declared as such inline and passes on both trees.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "opti_oignon"
FE = ROOT / "frontend" / "src"

ROUTES_SYNC_SRC = PKG / "api" / "routes_sync.py"
ROUTES_NOTES_SRC = PKG / "api" / "routes_notes.py"
TOOLS_SRC = PKG / "agent" / "tools.py"
ENGINE_SRC = PKG / "veilid" / "sync_engine.py"
NOTES_CLIENT = FE / "lib" / "api" / "notes.ts"
SYNC_CLIENT = FE / "lib" / "api" / "sync.ts"
NOTES_STORE_TS = FE / "lib" / "stores" / "notes.ts"
NOTES_PANEL = FE / "lib" / "components" / "panels" / "NotesPanel.svelte"
SYNC_PANEL = FE / "lib" / "components" / "panels" / "SyncPanel.svelte"
RUNBOOK_PATH = ROOT / "UI_TOGGLES_E2E_S260.md"
ROADMAP_PATH = ROOT / "NOTES_FEATURE_ROADMAP.md"
VERSION_PATH = PKG / "__version__.py"
PYPROJECT_PATH = ROOT / "pyproject.toml"


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _flat(text: str) -> str:
    """Collapse whitespace runs to single spaces (reflow-immune pins)."""
    return re.sub(r"\s+", " ", text)


def _strip_token_fallbacks(text: str) -> str:
    """Remove var(--oo-..., #fallback) occurrences so only RAW hex remains."""
    return re.sub(r"var\(--oo-[^)]*\)", "", text)


def _has_raw_hex(text: str) -> bool:
    """True if a hardcoded hex colour survives outside a --oo-* fallback."""
    return re.search(r"#[0-9a-fA-F]{3,8}\b", _strip_token_fallbacks(text)) is not None


def _block_balanced(text: str) -> bool:
    """Svelte block + script/style balance (a lightweight structural pass)."""
    for blk in ("if", "each", "await", "key"):
        if text.count("{#" + blk) != text.count("{/" + blk + "}"):
            return False
    if text.count("<script") != text.count("</script>"):
        return False
    if text.count("<style") != text.count("</style>"):
        return False
    return True


# ---------------------------------------------------------------------------
# Isolation harness (the s252 / s256 / s258 idiom: real modules over light
# stubs, package stubs granted a __path__ non-destructively)
# ---------------------------------------------------------------------------


def _ensure_pkg(name: str, path: Path) -> None:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if not hasattr(mod, "__path__"):
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]


def _ensure_stubs() -> None:
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.veilid", PKG / "veilid")
    _ensure_pkg("opti_oignon.api", PKG / "api")
    if "opti_oignon.security_mode" not in sys.modules:
        sm = types.ModuleType("opti_oignon.security_mode")
        sm.get_current_mode = lambda: "daily"  # type: ignore[attr-defined]
        sys.modules["opti_oignon.security_mode"] = sm
    if "opti_oignon.signed_audit_log" not in sys.modules:
        al = types.ModuleType("opti_oignon.signed_audit_log")
        al.chain_log = lambda **kw: None  # type: ignore[attr-defined]
        sys.modules["opti_oignon.signed_audit_log"] = al


def _routes_sync():
    _ensure_stubs()
    try:
        return importlib.import_module("opti_oignon.api.routes_sync")
    except Exception:
        return None


def _peers():
    _ensure_stubs()
    return importlib.import_module("opti_oignon.veilid.peers")


def _guard_attr(mod, name: str):
    obj = getattr(mod, name, None)
    assert obj is not None, f"S260 surface absent: {name!r}"
    return obj


# ---------------------------------------------------------------------------
# Fakes (the s258 shapes: event-recording, no web stack, no disk)
# ---------------------------------------------------------------------------


class FakeRecord:
    def __init__(self, **kw):
        self.peer_id = kw.get("peer_id", "p1")
        self.routing_key = kw.get("routing_key", "rk")
        self.label = kw.get("label", "")
        self.watermark = kw.get("watermark", 0)
        self.added_at = kw.get("added_at", "t0")
        self.updated_at = kw.get("updated_at", "t0")
        for k, v in kw.items():
            setattr(self, k, v)


class FakeStore:
    """A get_peer-only store over a mutable row table."""

    def __init__(self, rows=None, events=None):
        self.rows = dict(rows or {})
        self.events = events if events is not None else []

    def get_peer(self, peer_id):
        self.events.append(("lookup", peer_id))
        return self.rows.get(peer_id)


class FakeEngine:
    """An audited-setter stand-in that mutates the paired FakeStore row."""

    def __init__(self, store, events=None, result=True):
        self.store = store
        self.events = events if events is not None else []
        self.result = result
        self.set_calls = []

    def set_device_class(self, peer_id, device_class):
        self.events.append(("set", peer_id, device_class))
        self.set_calls.append((peer_id, device_class))
        rec = self.store.rows.get(peer_id)
        if rec is not None and self.result:
            rec.device_class = device_class
        return self.result


# ---------------------------------------------------------------------------
# Family 1 -- the pure normaliser (free text never reaches the store)
# ---------------------------------------------------------------------------


class TestNormaliser:
    def _fn(self):
        mod = _routes_sync()
        assert mod is not None, "routes_sync import failed"
        return _guard_attr(mod, "_normalise_device_class")

    def test_exists_and_pure_signature(self):
        fn = self._fn()
        params = inspect.signature(fn).parameters
        assert list(params) == ["value"]

    def test_phone_passes(self):
        assert self._fn()("phone") == "phone"

    def test_desktop_passes(self):
        assert self._fn()("desktop") == "desktop"

    def test_none_clears(self):
        assert self._fn()(None) is None

    def test_blank_clears(self):
        fn = self._fn()
        assert fn("") is None
        assert fn("   ") is None

    def test_case_and_padding_normalise(self):
        assert self._fn()("  Phone ") == "phone"

    def test_free_text_raises_value_error(self):
        fn = self._fn()
        with pytest.raises(ValueError):
            fn("tablet")

    def test_non_string_raises_value_error(self):
        fn = self._fn()
        with pytest.raises(ValueError):
            fn(7)


# ---------------------------------------------------------------------------
# Family 2 -- the web-free setter payload helper (the relabel pattern)
# ---------------------------------------------------------------------------


class TestSetterPayloadHelper:
    def _fn(self):
        mod = _routes_sync()
        assert mod is not None, "routes_sync import failed"
        return mod, _guard_attr(mod, "set_device_class_payload")

    def test_signature_is_store_engine_peer_value(self):
        _, fn = self._fn()
        params = list(inspect.signature(fn).parameters)
        assert params == ["store", "engine", "peer_id", "device_class"]

    def test_happy_path_sets_and_returns_fresh_dict(self):
        _, fn = self._fn()
        store = FakeStore({"p1": FakeRecord(device_class=None)})
        engine = FakeEngine(store)
        out = fn(store, engine, "p1", "phone")
        assert engine.set_calls == [("p1", "phone")]
        assert out["device_class"] == "phone"
        assert out["peer_id"] == "p1"

    def test_clear_path_writes_none(self):
        _, fn = self._fn()
        store = FakeStore({"p1": FakeRecord(device_class="phone")})
        engine = FakeEngine(store)
        out = fn(store, engine, "p1", None)
        assert engine.set_calls == [("p1", None)]
        assert out["device_class"] is None

    def test_unknown_peer_raises_before_any_engine_write(self):
        mod, fn = self._fn()
        store = FakeStore({})
        engine = FakeEngine(store)
        with pytest.raises(mod.PeerNotFound):
            fn(store, engine, "ghost", "phone")
        assert engine.set_calls == []

    def test_free_text_raises_before_any_engine_write(self):
        _, fn = self._fn()
        store = FakeStore({"p1": FakeRecord(device_class=None)})
        engine = FakeEngine(store)
        with pytest.raises(ValueError):
            fn(store, engine, "p1", "tablet")
        assert engine.set_calls == []

    def test_engine_false_maps_to_peer_not_found(self):
        mod, fn = self._fn()
        store = FakeStore({"p1": FakeRecord(device_class=None)})
        engine = FakeEngine(store, result=False)
        with pytest.raises(mod.PeerNotFound):
            fn(store, engine, "p1", "phone")


# ---------------------------------------------------------------------------
# Family 3 -- the route leg source (auth-gated, 400 / 404 mapped)
# ---------------------------------------------------------------------------


def _route_fn_segment() -> str:
    """The sync_set_device_class function source, or '' when absent."""
    src = _read(ROUTES_SYNC_SRC)
    if not src:
        return ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "sync_set_device_class":
            return ast.get_source_segment(src, node) or ""
    return ""


class TestRouteLeg:
    def test_decorator_present(self):
        src = _flat(_read(ROUTES_SYNC_SRC))
        assert '@router.post("/peers/{peer_id}/device-class")' in src

    def test_handler_function_present(self):
        assert _route_fn_segment(), "sync_set_device_class absent"

    def test_handler_wraps_the_helper(self):
        seg = _route_fn_segment()
        assert seg, "sync_set_device_class absent"
        assert "set_device_class_payload(store, engine, peer_id," in _flat(seg)

    def test_value_error_maps_to_400(self):
        seg = _route_fn_segment()
        assert seg, "sync_set_device_class absent"
        assert "except ValueError" in seg
        assert "status_code=400" in seg

    def test_peer_not_found_maps_to_404(self):
        seg = _route_fn_segment()
        assert seg, "sync_set_device_class absent"
        assert "except PeerNotFound" in seg
        assert "status_code=404" in seg
        assert "Peer not paired" in seg

    def test_leg_sits_under_the_authed_router(self):
        # Design-green half (the dependency line predates S260) plus the
        # red half (the new decorator rides THAT router object).
        src = _read(ROUTES_SYNC_SRC)
        assert 'dependencies=_auth_dep' in src
        assert '@router.post("/peers/{peer_id}/device-class")' in _flat(src)


# ---------------------------------------------------------------------------
# Family 4 -- the sync TS client (the read field and the setter call)
# ---------------------------------------------------------------------------


class TestSyncClientTs:
    def test_peer_interface_carries_class(self):
        s = _read(SYNC_CLIENT)
        assert s, "sync.ts absent"
        assert "device_class?: string | null;" in s

    def test_setter_function_exported(self):
        s = _read(SYNC_CLIENT)
        assert s, "sync.ts absent"
        assert "export async function setDeviceClass" in s

    def test_setter_posts_the_new_leg(self):
        s = _read(SYNC_CLIENT)
        assert s, "sync.ts absent"
        assert "/device-class" in s

    def test_setter_value_type_is_the_allowlist(self):
        s = _flat(_read(SYNC_CLIENT))
        assert s, "sync.ts absent"
        assert "'phone' | 'desktop' | null" in s


# ---------------------------------------------------------------------------
# Family 5 -- the notes TS client (the flag rides the existing PATCH)
# ---------------------------------------------------------------------------


class TestNotesClientTs:
    def test_record_carries_flag(self):
        s = _read(NOTES_CLIENT)
        assert s, "notes.ts absent"
        assert "mobile_allowed: boolean;" in s

    def test_update_carries_optional_flag(self):
        s = _read(NOTES_CLIENT)
        assert s, "notes.ts absent"
        assert "mobile_allowed?: boolean;" in s

    def test_no_new_endpoint_in_client(self):
        # Design-green: the client still talks to /api/notes only (the flag
        # rides the existing PATCH; no dedicated path appears).
        s = _read(NOTES_CLIENT)
        assert s, "notes.ts absent"
        assert "mobile-allowed" not in s
        assert "'/api/notes'" in s


# ---------------------------------------------------------------------------
# Family 6 -- the notes Svelte store (confirmed posture by construction)
# ---------------------------------------------------------------------------


class TestNotesStoreTs:
    def test_action_exported(self):
        s = _read(NOTES_STORE_TS)
        assert s, "stores/notes.ts absent"
        assert "export async function toggleMobileAllowed" in s

    def test_action_rides_save_note(self):
        s = _flat(_read(NOTES_STORE_TS))
        assert s, "stores/notes.ts absent"
        assert "saveNote(note.id, { mobile_allowed: !note.mobile_allowed })" in s

    def test_save_note_replaces_row_from_response(self):
        # Design-green: the posture's mechanical half predates S260 -- the
        # store row is replaced from the PATCH RESPONSE, never flipped
        # locally first.
        s = _flat(_read(NOTES_STORE_TS))
        assert "notes.update((list) => list.map((n) => (n.id === id ? updated : n)))" in s


# ---------------------------------------------------------------------------
# Family 7 -- NotesPanel (the editor control and the list marker)
# ---------------------------------------------------------------------------


class TestNotesPanelToggle:
    def test_imports_the_action(self):
        s = _read(NOTES_PANEL)
        assert s, "NotesPanel absent"
        assert "toggleMobileAllowed" in s

    def test_control_label(self):
        s = _read(NOTES_PANEL)
        assert s, "NotesPanel absent"
        assert "Allow on phone" in s

    def test_busy_flag_disables_the_control(self):
        s = _flat(_read(NOTES_PANEL))
        assert s, "NotesPanel absent"
        assert "mobileBusy" in s
        assert "disabled={mobileBusy" in s

    def test_flip_is_confirmed_posture(self):
        # The component awaits the store action (whose state comes from the
        # server response) and surfaces failure; it never binds the checkbox
        # to a local draft of the flag.
        s = _read(NOTES_PANEL)
        assert s, "NotesPanel absent"
        assert "async function flipMobileAllowed" in s
        assert "await toggleMobileAllowed(" in s
        assert "bind:checked={editMobileAllowed}" not in s

    def test_failure_surfaces_a_toast(self):
        seg = _read(NOTES_PANEL)
        assert seg, "NotesPanel absent"
        assert "Failed to update phone sync" in seg

    def test_list_marker(self):
        s = _flat(_read(NOTES_PANEL))
        assert s, "NotesPanel absent"
        assert "{#if note.mobile_allowed}" in s

    def test_hygiene_invariants_hold(self):
        # Design-green on both trees: token discipline and block balance.
        raw = _read(NOTES_PANEL)
        assert raw, "NotesPanel absent"
        assert not _has_raw_hex(raw)
        assert _block_balanced(raw)


# ---------------------------------------------------------------------------
# Family 8 -- SyncPanel (the class shown, the setter wired)
# ---------------------------------------------------------------------------


class TestSyncPanelClass:
    def test_pending_card_shows_declared_class(self):
        s = _read(SYNC_PANEL)
        assert s, "SyncPanel absent"
        assert "p.device_class" in s

    def test_peer_row_shows_class(self):
        s = _read(SYNC_PANEL)
        assert s, "SyncPanel absent"
        assert "peer.device_class" in s

    def test_setter_imported_and_called(self):
        s = _read(SYNC_PANEL)
        assert s, "SyncPanel absent"
        assert "setDeviceClass" in s
        assert "await setDeviceClass(" in s

    def test_setter_controls_present(self):
        s = _read(SYNC_PANEL)
        assert s, "SyncPanel absent"
        assert "Treat as phone" in s
        assert "Treat as desktop" in s

    def test_phone_meaning_stated(self):
        # The panel says what phone-class MEANS (the serve-side filter),
        # so the human decision is informed at the point of action.
        s = _read(SYNC_PANEL)
        assert s, "SyncPanel absent"
        assert "mobile-allowed notes" in s

    def test_carried_negative_pins_hold(self):
        # Design-green on both trees (the s207 provenance-only rule): the
        # record body never enters the panel.
        s = _read(SYNC_PANEL)
        assert s, "SyncPanel absent"
        assert "d.payload" not in s
        assert "envelope" not in s

    def test_hygiene_invariants_hold(self):
        # Design-green on both trees: token discipline and block balance.
        raw = _read(SYNC_PANEL)
        assert raw, "SyncPanel absent"
        assert not _has_raw_hex(raw)
        assert _block_balanced(raw)


# ---------------------------------------------------------------------------
# Family 9 -- the host runbook (container-provable vs host-assured split)
# ---------------------------------------------------------------------------


class TestRunbook:
    def test_exists(self):
        assert RUNBOOK_PATH.exists(), "UI_TOGGLES_E2E_S260.md absent"

    def test_title_names_the_lot(self):
        s = _read(RUNBOOK_PATH)
        assert s, "runbook absent"
        assert "S260" in s
        assert "UI toggles" in s

    def test_container_vs_host_split_stated(self):
        s = _read(RUNBOOK_PATH)
        assert s, "runbook absent"
        assert "host-assured" in s
        assert "never simulated in-container" in s

    def test_notes_toggle_walk_present(self):
        s = _read(RUNBOOK_PATH)
        assert s, "runbook absent"
        assert "Allow on phone" in s
        assert "mobile_allowed" in s

    def test_device_class_walk_present(self):
        s = _read(RUNBOOK_PATH)
        assert s, "runbook absent"
        assert "device-class" in s
        assert "Treat as phone" in s

    def test_findings_register_and_version(self):
        s = _read(RUNBOOK_PATH)
        assert s, "runbook absent"
        assert "Findings register" in s
        assert "3.12.0" in s


# ---------------------------------------------------------------------------
# Family 10 -- the roadmap roll (dated, additive)
# ---------------------------------------------------------------------------


class TestRoadmapRoll:
    def test_dated_sentence_added(self):
        s = _flat(_read(ROADMAP_PATH))
        assert s, "roadmap absent"
        assert "LANDED at S260" in s

    def test_sentence_names_both_toggles(self):
        s = _flat(_read(ROADMAP_PATH))
        assert s, "roadmap absent"
        assert "Allow on phone" in s
        assert "device-class" in s

    def test_prior_rolls_intact(self):
        # Design-green: the additive idiom never erases the earlier dated
        # sentences or the N9-D3 statement.
        s = _flat(_read(ROADMAP_PATH))
        assert "LANDED at S256" in s or "half LANDED at S256" in s
        assert "LANDED at S257" in s
        assert "LANDED at S258" in s
        assert "N9-D3" in s


# ---------------------------------------------------------------------------
# Family 11 -- reassertions (design-green on both trees)
# ---------------------------------------------------------------------------


class TestReassertions:
    def test_tools_surface_stays_zero(self):
        src = _read(TOOLS_SRC)
        assert src, "tools.py absent"
        assert "mobile_allowed" not in src

    def test_notes_routes_stay_five_at_source(self):
        src = _read(ROUTES_NOTES_SRC)
        assert src, "routes_notes absent"
        assert src.count("@notes_router.") == 5
        assert src.count("@notes_router.patch") == 1

    def test_patch_leg_still_uses_the_dedicated_setter(self):
        src = _flat(_read(ROUTES_NOTES_SRC))
        assert "store.set_mobile_allowed(" in src

    def test_peer_dict_carries_class_defensively(self):
        mod = _routes_sync()
        assert mod is not None, "routes_sync import failed"
        d = mod._peer_to_dict(FakeRecord(device_class="phone"))
        assert d.get("device_class") == "phone"
        bare = FakeRecord()
        if hasattr(bare, "device_class"):
            delattr(bare, "device_class")
        assert mod._peer_to_dict(bare).get("device_class") is None

    def test_store_allowlist_is_the_two_classes(self):
        peers = _peers()
        assert peers.DEVICE_CLASSES == frozenset({"phone", "desktop"})

    def test_engine_setter_audits_the_flip(self):
        src = _read(ENGINE_SRC)
        assert src, "sync_engine absent"
        assert "peer_device_class" in src


# ---------------------------------------------------------------------------
# Family 12 -- structure (AST, ASCII, the selection literal)
# ---------------------------------------------------------------------------


_ASCII_ENFORCED = [
    ROUTES_SYNC_SRC,
    RUNBOOK_PATH,
    Path(__file__),
]


class TestStructure:
    def test_routes_sync_parses(self):
        src = _read(ROUTES_SYNC_SRC)
        assert src, "routes_sync absent"
        ast.parse(src)

    @pytest.mark.parametrize("path", _ASCII_ENFORCED, ids=lambda p: p.name)
    def test_sources_ascii(self, path):
        raw = _read(path)
        assert raw, f"{path.name} absent"
        assert all(ord(ch) < 128 for ch in raw), f"non-ASCII in {path.name}"

    def test_this_suite_avoids_the_selection_literal(self):
        here = Path(__file__).read_text(encoding="utf-8")
        token = "sandbox" + "_manager"
        assert token not in here

    def test_version_held(self):
        assert '__version__ = "3.12.0"' in _read(VERSION_PATH)
        assert 'version = "3.12.0"' in _read(PYPROJECT_PATH)
