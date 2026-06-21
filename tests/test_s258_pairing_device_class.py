#!/usr/bin/env python3
"""S258 -- the pairing payload carries ``device_class`` (N.9, PAIR-03).

The slice closes the host runbook's least-assured step: through S257 a phone
pairing was marked phone-class only by a manual ``set_device_class`` after the
ceremony. S258 lands the class in the pairing payload (the joining device
declares itself), folds it into the payload INTEGRITY when present (the S205
signing_pub recipe: a stripped or garbled class fails the digest and the
payload is rejected, never silently degraded), and records it at the accept
seam through the engine's new audited ``set_device_class`` under a MONOTONE
policy: a declaration may keep or restrict a stored class, never escalate it
(phone -> desktop stays human-only at the control surface). Fail-secure
mappings, decided at the read gate:

- A FRESH row with no declaration records ``phone`` explicitly (an undeclared
  new device is least-trusted; N9-D2 carried).
- An EXISTING row re-paired with no declaration keeps its class untouched
  (absence is no statement; the NULL grandfathered desktop only ever applies
  to rows that predate the ceremony's declaration).
- An out-of-vocabulary declaration normalises to ``phone`` with a warning
  (free text never reaches the store's setter; a future vocabulary never
  bricks the ceremony).
- An indeterminable prior (no store handle at the seam) only ever writes a
  ``phone`` resolution -- never a blind escalation.

The PAIR-02 confirmation code deliberately EXCLUDES the class from its
material (the read-gate correction): the code must recompute identically on
both devices from material both can derive, and the registry's post-policy
class can lawfully differ from the payload's declared one (fail-secure
default, normalisation, a monotone keep), while a legacy joiner pins
class-less material. The residual -- an active in-channel substitution can
flip the device-class field of a FRESH pairing without moving the code -- is
stated in the pairing module docstring; the compensating controls are the
integrity digest (garble and strip detection), the monotone re-pair rule, and
the class now riding ``_peer_to_dict`` so the pending panel shows what was
recorded next to the code.

Red-before contract: every test that touches the new surface asserts the
surface exists (getattr / signature / dataclass-field guards) BEFORE calling
it, so on the pristine S257 tree each red is an AssertionError, never a
collection error or a TypeError (the S257 assert-before-index lesson,
generalised to assert-before-call). The design-green set -- the legacy digest
formula, the stdlib-only pin, the code determinism, the reassertions, the
retained doc rolls, and structure -- is declared as such inline.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "opti_oignon"
VEILID = PKG / "veilid"

PAIRING_SRC = VEILID / "pairing.py"
PEERS_SRC = VEILID / "peers.py"
ENGINE_SRC = VEILID / "sync_engine.py"
SIGNING_SRC = VEILID / "signing.py"
ROUTES_SYNC_SRC = PKG / "api" / "routes_sync.py"
TOOLS_SRC = PKG / "agent" / "tools.py"
NOTES_STORE_SRC = PKG / "notes" / "notes_store.py"
VERSION_PATH = PKG / "__version__.py"
ROADMAP_PATH = ROOT / "NOTES_FEATURE_ROADMAP.md"
RUNBOOK_PATH = ROOT / "NOTES_MOBILE_SYNC_N9_S256.md"


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
# Isolation harness (the s252 / s256 idiom: real modules over light stubs,
# package stubs granted a __path__ non-destructively)
# ---------------------------------------------------------------------------

_AUDIT: dict = {"events": []}


def _record_audit(**kwargs):
    _AUDIT["events"].append(kwargs)


def _ensure_pkg(name: str, path: Path) -> None:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if not hasattr(mod, "__path__"):
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]


def _ensure_stubs() -> None:
    _ensure_pkg("opti_oignon", PKG)
    _ensure_pkg("opti_oignon.veilid", VEILID)
    _ensure_pkg("opti_oignon.api", PKG / "api")
    if "opti_oignon.security_mode" not in sys.modules:
        sm = types.ModuleType("opti_oignon.security_mode")
        sm.get_current_mode = lambda: "daily"  # type: ignore[attr-defined]
        sys.modules["opti_oignon.security_mode"] = sm
    if "opti_oignon.signed_audit_log" not in sys.modules:
        al = types.ModuleType("opti_oignon.signed_audit_log")
        al.chain_log = _record_audit  # type: ignore[attr-defined]
        sys.modules["opti_oignon.signed_audit_log"] = al


def _pairing():
    _ensure_stubs()
    return importlib.import_module("opti_oignon.veilid.pairing")


def _peers():
    _ensure_stubs()
    return importlib.import_module("opti_oignon.veilid.peers")


def _engine_mod():
    _ensure_stubs()
    return importlib.import_module("opti_oignon.veilid.sync_engine")


def _routes_sync():
    _ensure_stubs()
    try:
        return importlib.import_module("opti_oignon.api.routes_sync")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Manual digest recomputation (independent of the module under test)
# ---------------------------------------------------------------------------


def _manual_material(
    peer_id: str,
    routing_key: str,
    signing_pub=None,
    device_class=None,
) -> str:
    canonical = {
        "v": 1,
        "type": "veilid_pairing",
        "peer_id": peer_id,
        "routing_key": routing_key,
    }
    if signing_pub is not None:
        canonical["signing_pub"] = signing_pub
    if device_class is not None:
        canonical["device_class"] = device_class
    return json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _manual_integrity(
    peer_id: str,
    routing_key: str,
    signing_pub=None,
    device_class=None,
) -> str:
    blob = _manual_material(peer_id, routing_key, signing_pub, device_class)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Guards (assert-before-call: pristine reds are AssertionErrors)
# ---------------------------------------------------------------------------


def _guard_param(fn, name: str) -> None:
    assert fn is not None, "S258 surface absent"
    params = inspect.signature(fn).parameters
    assert name in params, (
        "S258 surface absent: parameter {!r} missing on {!r}".format(
            name, getattr(fn, "__name__", fn)
        )
    )


def _guard_attr(mod, name: str):
    obj = getattr(mod, name, None)
    assert obj is not None, f"S258 surface absent: {name!r}"
    return obj


def _resolve(mod, declared, prior_exists, prior_class):
    fn = _guard_attr(mod, "resolve_pairing_device_class")
    return fn(declared, prior_exists, prior_class)


def _build_with_class(mod, peer_id, routing_key, signing_pub, device_class):
    _guard_param(mod.build_pairing_payload, "device_class")
    return mod.build_pairing_payload(
        peer_id, routing_key, signing_pub, device_class=device_class
    )


def _handcraft(mod, peer_id, routing_key, signing_pub, device_class):
    """A payload carrying an arbitrary class string, integrity sender-true."""
    _guard_param(mod.pairing_integrity, "device_class")
    payload = {
        "v": 1,
        "type": "veilid_pairing",
        "peer_id": peer_id,
        "routing_key": routing_key,
    }
    if signing_pub is not None:
        payload["signing_pub"] = signing_pub
    if device_class is not None:
        payload["device_class"] = device_class
    payload["integrity"] = mod.pairing_integrity(
        peer_id, routing_key, signing_pub, device_class=device_class
    )
    return payload


# ---------------------------------------------------------------------------
# Fakes for the accept seam and the routes helpers
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
    """A get_peer-only store; records lookup events into a shared log."""

    def __init__(self, rows=None, events=None, raise_on_get=False):
        self.rows = dict(rows or {})
        self.events = events if events is not None else []
        self.raise_on_get = raise_on_get

    def get_peer(self, peer_id):
        self.events.append(("lookup", peer_id))
        if self.raise_on_get:
            raise RuntimeError("store down")
        return self.rows.get(peer_id)


class FakeEngine:
    """register_peer + optional set_device_class, event-recording."""

    def __init__(self, events=None, with_setter=True, setter_raises=False):
        self.events = events if events is not None else []
        self.device = "self-device"
        self.set_calls = []
        if with_setter:
            if setter_raises:

                def _boom(peer_id, device_class):
                    self.events.append(("set", peer_id, device_class))
                    raise RuntimeError("setter down")

                self.set_device_class = _boom
            else:

                def _setter(peer_id, device_class):
                    self.events.append(("set", peer_id, device_class))
                    self.set_calls.append((peer_id, device_class))
                    return True

                self.set_device_class = _setter

    def register_peer(self, peer_id, routing_key, **kw):
        self.events.append(("register", peer_id))
        return FakeRecord(peer_id=peer_id, routing_key=routing_key, **kw)


def _accept(mod, engine, obj, store):
    _guard_param(mod.accept_pairing_payload, "store")
    return mod.accept_pairing_payload(engine, obj, label="", store=store)


# ---------------------------------------------------------------------------
# Family 1 -- the wire vocabulary
# ---------------------------------------------------------------------------


class TestWireVocabulary:
    def test_vocab_constants_exist_and_exact(self):
        mod = _pairing()
        vocab = _guard_attr(mod, "PAIRING_DEVICE_CLASSES")
        assert vocab == frozenset({"phone", "desktop"})
        assert _guard_attr(mod, "PAIRING_DEVICE_CLASS_PHONE") == "phone"
        assert _guard_attr(mod, "PAIRING_DEVICE_CLASS_DESKTOP") == "desktop"

    def test_vocab_matches_registry_allowlist(self):
        # Anti-drift: the wire vocabulary and the registry allowlist are the
        # same two words, pinned EQUAL across the two modules without a
        # runtime import in pairing.py.
        mod = _pairing()
        peers = _peers()
        vocab = _guard_attr(mod, "PAIRING_DEVICE_CLASSES")
        assert vocab == peers.DEVICE_CLASSES

    def test_pairing_stays_stdlib_only(self):
        # Design-green: the module's loadable-in-isolation property survives
        # the slice (no opti_oignon import appears in pairing.py).
        src = _read(PAIRING_SRC)
        assert src != ""
        assert "from opti_oignon" not in src
        assert "import opti_oignon" not in src


# ---------------------------------------------------------------------------
# Family 2 -- the payload and its integrity (the signing_pub recipe)
# ---------------------------------------------------------------------------


class TestPayloadIntegrity:
    def test_legacy_digest_formula_holds(self):
        # Design-green: the historical digest is independently recomputable
        # on BOTH trees -- the formula can never drift.
        mod = _pairing()
        assert mod.pairing_integrity("A", "rk-A") == _manual_integrity(
            "A", "rk-A"
        )
        assert mod.pairing_integrity(
            "A", "rk-A", "PUB"
        ) == _manual_integrity("A", "rk-A", "PUB")

    def test_build_signature_gains_device_class(self):
        mod = _pairing()
        _guard_param(mod.build_pairing_payload, "device_class")
        _guard_param(mod.pairing_integrity, "device_class")
        _guard_param(mod.pairing_canonical_material, "device_class")

    def test_classless_build_keeps_legacy_shape_and_digest(self):
        mod = _pairing()
        _guard_param(mod.build_pairing_payload, "device_class")
        p = mod.build_pairing_payload("A", "rk-A", "PUB")
        assert "device_class" not in p
        assert p["integrity"] == _manual_integrity("A", "rk-A", "PUB")
        q = mod.build_pairing_payload("A", "rk-A", "PUB", device_class=None)
        assert q == p

    def test_class_rides_payload_and_integrity(self):
        mod = _pairing()
        p = _build_with_class(mod, "A", "rk-A", "PUB", "phone")
        assert p.get("device_class") == "phone"
        assert p["integrity"] == _manual_integrity(
            "A", "rk-A", "PUB", "phone"
        )

    def test_builder_rejects_out_of_vocab(self):
        mod = _pairing()
        _guard_param(mod.build_pairing_payload, "device_class")
        with pytest.raises(ValueError):
            mod.build_pairing_payload(
                "A", "rk-A", None, device_class="toaster"
            )
        with pytest.raises(ValueError):
            mod.build_pairing_payload("A", "rk-A", None, device_class="")

    def test_canonical_material_folds_class_sorted(self):
        mod = _pairing()
        _guard_param(mod.pairing_canonical_material, "device_class")
        mat = mod.pairing_canonical_material(
            "A", "rk-A", "PUB", device_class="phone"
        )
        assert mat == _manual_material("A", "rk-A", "PUB", "phone")
        legacy = mod.pairing_canonical_material("A", "rk-A", "PUB")
        assert legacy == _manual_material("A", "rk-A", "PUB")

    def test_strip_in_transit_rejected(self):
        mod = _pairing()
        p = _build_with_class(mod, "A", "rk-A", None, "phone")
        del p["device_class"]
        assert mod.parse_pairing_payload(p) is None

    def test_tampered_class_rejected(self):
        mod = _pairing()
        p = _build_with_class(mod, "A", "rk-A", None, "phone")
        p["device_class"] = "desktop"
        assert mod.parse_pairing_payload(p) is None


# ---------------------------------------------------------------------------
# Family 3 -- the defensive parse (the VL-01 idiom)
# ---------------------------------------------------------------------------


class TestParseDefensive:
    def test_parsed_dataclass_gains_field(self):
        mod = _pairing()
        names = {f.name for f in dataclasses.fields(mod.ParsedPairing)}
        assert "device_class" in names

    def test_parse_carries_class(self):
        mod = _pairing()
        names = {f.name for f in dataclasses.fields(mod.ParsedPairing)}
        assert "device_class" in names
        out = mod.parse_pairing_payload(
            _build_with_class(mod, "A", "rk-A", "PUB", "phone")
        )
        assert out is not None
        assert out.device_class == "phone"

    def test_classless_parses_with_none(self):
        mod = _pairing()
        names = {f.name for f in dataclasses.fields(mod.ParsedPairing)}
        assert "device_class" in names
        out = mod.parse_pairing_payload(
            mod.build_pairing_payload("A", "rk-A", "PUB")
        )
        assert out is not None
        assert out.device_class is None

    def test_mistyped_class_fails_sender_integrity(self):
        # The VL-01 idiom: present-but-mistyped is treated absent for the
        # recomputation, so a payload whose class was garbled into a
        # non-string fails its own digest -- tampering never degrades
        # silently into "no class".
        mod = _pairing()
        p = _build_with_class(mod, "A", "rk-A", None, "phone")
        p["device_class"] = 12345
        assert mod.parse_pairing_payload(p) is None
        q = _build_with_class(mod, "A", "rk-A", None, "phone")
        q["device_class"] = ""
        assert mod.parse_pairing_payload(q) is None

    def test_unknown_vocab_parses_for_apply_side_normalisation(self):
        # Forward compatibility: a future class word rides the wire and the
        # digest; the VOCABULARY is judged at the apply seam, never at parse.
        mod = _pairing()
        p = _handcraft(mod, "A", "rk-A", None, "tablet")
        out = mod.parse_pairing_payload(p)
        assert out is not None
        assert out.device_class == "tablet"


# ---------------------------------------------------------------------------
# Family 4 -- the confirmation code posture (deliberate exclusion)
# ---------------------------------------------------------------------------


class TestConfirmationCodePosture:
    def test_code_deterministic_and_order_normalized(self):
        # Design-green: the PAIR-02 derivation itself is untouched.
        mod = _pairing()
        a = mod.pairing_canonical_material("A", "rk-A")
        b = mod.pairing_canonical_material("B", "rk-B")
        assert mod.confirmation_code(a, b) == mod.confirmation_code(b, a)

    def test_module_states_the_fresh_pairing_residual(self):
        src = _read(PAIRING_SRC)
        assert "device-class field of a FRESH pairing" in _flat(src)


# ---------------------------------------------------------------------------
# Family 5 -- the monotone decision function (the D2-D4 table)
# ---------------------------------------------------------------------------


class TestResolveDecision:
    def test_fresh_row_records_affirmatively(self):
        mod = _pairing()
        assert _resolve(mod, None, False, None) == (True, "phone")
        assert _resolve(mod, "phone", False, None) == (True, "phone")
        assert _resolve(mod, "desktop", False, None) == (True, "desktop")

    def test_fresh_row_normalises_free_text(self):
        mod = _pairing()
        assert _resolve(mod, "toaster", False, None) == (True, "phone")
        assert _resolve(mod, "", False, None) == (True, "phone")

    def test_existing_null_grandfathered(self):
        mod = _pairing()
        assert _resolve(mod, None, True, None) == (False, None)
        assert _resolve(mod, "phone", True, None) == (True, "phone")
        assert _resolve(mod, "desktop", True, None) == (True, "desktop")
        assert _resolve(mod, "tablet", True, None) == (True, "phone")

    def test_existing_desktop(self):
        mod = _pairing()
        assert _resolve(mod, None, True, "desktop") == (False, None)
        assert _resolve(mod, "phone", True, "desktop") == (True, "phone")
        assert _resolve(mod, "desktop", True, "desktop") == (False, None)
        assert _resolve(mod, "tablet", True, "desktop") == (True, "phone")

    def test_existing_phone_never_escalates(self):
        mod = _pairing()
        assert _resolve(mod, None, True, "phone") == (False, None)
        assert _resolve(mod, "phone", True, "phone") == (False, None)
        assert _resolve(mod, "desktop", True, "phone") == (False, None)
        assert _resolve(mod, "tablet", True, "phone") == (False, None)

    def test_garbage_prior_ranks_lowest(self):
        # A registry value outside the vocabulary (defensive: the setter
        # cannot write one) ranks lowest: a phone resolution normalises it,
        # a desktop one is refused.
        mod = _pairing()
        assert _resolve(mod, "phone", True, "weird") == (True, "phone")
        assert _resolve(mod, "desktop", True, "weird") == (False, None)

    def test_indeterminable_prior_never_escalates_blind(self):
        mod = _pairing()
        assert _resolve(mod, None, None, None) == (False, None)
        assert _resolve(mod, "phone", None, None) == (True, "phone")
        assert _resolve(mod, "desktop", None, None) == (False, None)
        assert _resolve(mod, "tablet", None, None) == (True, "phone")


# ---------------------------------------------------------------------------
# Family 6 -- the accept seam
# ---------------------------------------------------------------------------


class TestAcceptSeam:
    def test_fresh_declared_phone_applies(self):
        mod = _pairing()
        engine = FakeEngine()
        rec = _accept(
            mod,
            engine,
            _build_with_class(mod, "p1", "rk1", None, "phone"),
            FakeStore(),
        )
        assert rec is not None
        assert engine.set_calls == [("p1", "phone")]

    def test_fresh_declared_desktop_applies(self):
        mod = _pairing()
        engine = FakeEngine()
        rec = _accept(
            mod,
            engine,
            _build_with_class(mod, "p1", "rk1", None, "desktop"),
            FakeStore(),
        )
        assert rec is not None
        assert engine.set_calls == [("p1", "desktop")]

    def test_fresh_absent_class_fail_secure_phone(self):
        mod = _pairing()
        engine = FakeEngine()
        rec = _accept(
            mod,
            engine,
            mod.build_pairing_payload("p1", "rk1"),
            FakeStore(),
        )
        assert rec is not None
        assert engine.set_calls == [("p1", "phone")]

    def test_fresh_unknown_vocab_normalised_phone(self):
        mod = _pairing()
        engine = FakeEngine()
        rec = _accept(
            mod,
            engine,
            _handcraft(mod, "p1", "rk1", None, "tablet"),
            FakeStore(),
        )
        assert rec is not None
        assert engine.set_calls == [("p1", "phone")]

    def test_repair_escalation_blocked(self):
        mod = _pairing()
        engine = FakeEngine()
        store = FakeStore({"p1": FakeRecord(device_class="phone")})
        rec = _accept(
            mod,
            engine,
            _build_with_class(mod, "p1", "rk1", None, "desktop"),
            store,
        )
        assert rec is not None
        assert engine.set_calls == []

    def test_repair_restriction_applies(self):
        mod = _pairing()
        engine = FakeEngine()
        store = FakeStore({"p1": FakeRecord(device_class="desktop")})
        rec = _accept(
            mod,
            engine,
            _build_with_class(mod, "p1", "rk1", None, "phone"),
            store,
        )
        assert rec is not None
        assert engine.set_calls == [("p1", "phone")]

    def test_repair_absent_declaration_keeps(self):
        mod = _pairing()
        engine = FakeEngine()
        store = FakeStore({"p1": FakeRecord(device_class=None)})
        rec = _accept(mod, engine, mod.build_pairing_payload("p1", "rk1"), store)
        assert rec is not None
        assert engine.set_calls == []

    def test_repair_null_to_desktop_explicitation(self):
        mod = _pairing()
        engine = FakeEngine()
        store = FakeStore({"p1": FakeRecord(device_class=None)})
        rec = _accept(
            mod,
            engine,
            _build_with_class(mod, "p1", "rk1", None, "desktop"),
            store,
        )
        assert rec is not None
        assert engine.set_calls == [("p1", "desktop")]

    def test_no_store_only_phone_writes(self):
        mod = _pairing()
        engine = FakeEngine()
        rec = _accept(
            mod,
            engine,
            _build_with_class(mod, "p1", "rk1", None, "desktop"),
            None,
        )
        assert rec is not None
        assert engine.set_calls == []
        engine2 = FakeEngine()
        rec2 = _accept(
            mod,
            engine2,
            _build_with_class(mod, "p2", "rk2", None, "phone"),
            None,
        )
        assert rec2 is not None
        assert engine2.set_calls == [("p2", "phone")]

    def test_raising_store_lookup_is_indeterminable(self):
        mod = _pairing()
        engine = FakeEngine()
        store = FakeStore(raise_on_get=True)
        rec = _accept(
            mod,
            engine,
            _build_with_class(mod, "p1", "rk1", None, "desktop"),
            store,
        )
        assert rec is not None
        assert engine.set_calls == []

    def test_setter_raising_never_voids_registration(self):
        mod = _pairing()
        engine = FakeEngine(setter_raises=True)
        rec = _accept(
            mod,
            engine,
            _build_with_class(mod, "p1", "rk1", None, "phone"),
            FakeStore(),
        )
        assert rec is not None
        assert ("register", "p1") in engine.events

    def test_engine_without_setter_registration_stands(self):
        mod = _pairing()
        engine = FakeEngine(with_setter=False)
        rec = _accept(
            mod,
            engine,
            _build_with_class(mod, "p1", "rk1", None, "phone"),
            FakeStore(),
        )
        assert rec is not None
        assert ("register", "p1") in engine.events

    def test_prior_lookup_happens_before_register(self):
        mod = _pairing()
        events: list = []
        engine = FakeEngine(events=events)
        store = FakeStore(events=events)
        _accept(
            mod,
            engine,
            _build_with_class(mod, "p1", "rk1", None, "phone"),
            store,
        )
        assert ("lookup", "p1") in events
        assert ("register", "p1") in events
        assert events.index(("lookup", "p1")) < events.index(
            ("register", "p1")
        )

    def test_invalid_and_self_payloads_still_refused(self):
        mod = _pairing()
        engine = FakeEngine()
        assert _accept(mod, engine, {"v": 99}, FakeStore()) is None
        own = mod.build_pairing_payload("self-device", "rk-self")
        assert _accept(mod, engine, own, FakeStore()) is None
        assert engine.set_calls == []


# ---------------------------------------------------------------------------
# Family 7 -- the engine's audited setter
# ---------------------------------------------------------------------------


def _real_world(tmp_path):
    """A real engine over a real PeerStore, audit recorded."""
    peers = _peers()
    em = _engine_mod()
    sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]
    _AUDIT["events"].clear()
    store = peers.PeerStore(root=tmp_path / "server")
    engine = em.SyncEngine(device="server", store=store)
    return store, engine


class TestEngineSetter:
    def test_engine_exposes_setter(self, tmp_path):
        _store, engine = _real_world(tmp_path)
        assert callable(getattr(engine, "set_device_class", None))

    def test_engine_setter_delegates_and_returns(self, tmp_path):
        store, engine = _real_world(tmp_path)
        setter = getattr(engine, "set_device_class", None)
        assert callable(setter)
        store.add_peer("p1", "rk1")
        assert setter("p1", "phone") is True
        assert store.get_peer("p1").device_class == "phone"
        assert setter("p1", None) is True
        assert store.get_peer("p1").device_class is None
        assert setter("ghost", "phone") is False

    def test_engine_setter_validates_via_store(self, tmp_path):
        store, engine = _real_world(tmp_path)
        setter = getattr(engine, "set_device_class", None)
        assert callable(setter)
        store.add_peer("p1", "rk1")
        with pytest.raises(ValueError):
            setter("p1", "toaster")

    def test_engine_setter_audits_effective_change_only(self, tmp_path):
        store, engine = _real_world(tmp_path)
        setter = getattr(engine, "set_device_class", None)
        assert callable(setter)
        store.add_peer("p1", "rk1")
        _AUDIT["events"].clear()
        assert setter("p1", "phone") is True
        flips = [
            e
            for e in _AUDIT["events"]
            if e.get("action") == "peer_device_class"
        ]
        assert len(flips) == 1
        assert setter("p1", "phone") is True
        flips = [
            e
            for e in _AUDIT["events"]
            if e.get("action") == "peer_device_class"
        ]
        assert len(flips) == 1


# ---------------------------------------------------------------------------
# Family 8 -- the production threading (routes_sync)
# ---------------------------------------------------------------------------


class TestRoutesThreading:
    def test_self_payload_helper_threads_class(self):
        mod = _routes_sync()
        assert mod is not None
        _guard_param(mod.self_pairing_payload, "device_class")
        out = mod.self_pairing_payload("d1", "rk1", None, device_class="desktop")
        assert out["payload"].get("device_class") == "desktop"
        assert '"device_class":"desktop"' in out["text"]

    def test_self_payload_helper_classless_stays_legacy(self):
        mod = _routes_sync()
        assert mod is not None
        _guard_param(mod.self_pairing_payload, "device_class")
        out = mod.self_pairing_payload("d1", "rk1", None)
        assert "device_class" not in out["payload"]

    def test_route_source_declares_desktop_constant(self):
        src = _flat(_read(ROUTES_SYNC_SRC))
        assert "DEVICE_CLASS_DESKTOP, PeerStore" in src
        assert "device_class=DEVICE_CLASS_DESKTOP" in src

    def test_accept_route_threads_store(self):
        src = _flat(_read(ROUTES_SYNC_SRC))
        assert (
            "accept_pairing_payload(engine, obj, label=label, store=store)"
            in src
        )

    def test_accept_helper_applies_class_through_fakes(self):
        mod = _routes_sync()
        assert mod is not None
        pairing = _pairing()
        engine = FakeEngine()
        payload = _build_with_class(pairing, "p1", "rk1", None, "phone")
        out = mod.accept_pairing(engine, payload, label="", store=FakeStore())
        assert out["peer_id"] == "p1"
        assert engine.set_calls == [("p1", "phone")]

    def test_peer_dict_carries_class_defensively(self):
        mod = _routes_sync()
        assert mod is not None
        d = mod._peer_to_dict(FakeRecord(device_class="phone"))
        assert "device_class" in d
        assert d["device_class"] == "phone"
        bare = FakeRecord()
        if hasattr(bare, "device_class"):
            delattr(bare, "device_class")
        d2 = mod._peer_to_dict(bare)
        assert d2.get("device_class") is None

    def test_self_pin_material_stays_classless(self):
        # Design-green: the pinned self material (the PAIR-02 code's local
        # half) deliberately stays class-less, the legacy-compatible code
        # material on both trees.
        src = _flat(_read(ROUTES_SYNC_SRC))
        assert (
            "pairing_canonical_material( peer_id, routing_key, signing_pub )"
            in src
        )


# ---------------------------------------------------------------------------
# Family 9 -- reassertions (design-green on both trees)
# ---------------------------------------------------------------------------


class TestReassertions:
    def test_fresh_peer_has_no_class(self, tmp_path):
        peers = _peers()
        store = peers.PeerStore(root=tmp_path / "a")
        rec = store.add_peer("p1", "rk1")
        assert rec.device_class is None

    def test_setter_roundtrip_and_allowlist(self, tmp_path):
        peers = _peers()
        store = peers.PeerStore(root=tmp_path / "b")
        store.add_peer("p1", "rk1")
        assert store.set_device_class("p1", "phone") is True
        assert store.get_peer("p1").device_class == "phone"
        with pytest.raises(ValueError):
            store.set_device_class("p1", "toaster")

    def test_repair_upsert_preserves_class(self, tmp_path):
        peers = _peers()
        store = peers.PeerStore(root=tmp_path / "c")
        store.add_peer("p1", "rk1")
        store.set_device_class("p1", "phone")
        store.add_peer("p1", "rk1-rotated", label="renamed")
        assert store.get_peer("p1").device_class == "phone"

    def test_tools_never_name_the_flag(self):
        src = _read(TOOLS_SRC)
        assert src != ""
        assert "mobile_allowed" not in src

    def test_notes_glue_seams_still_wired(self):
        src = _read(NOTES_STORE_SRC)
        assert src.count("_sync_publish_note(") >= 5

    def test_registry_and_signing_pins_hold(self):
        peers_src = _read(PEERS_SRC)
        assert "device_class TEXT" in peers_src
        assert "DEVICE_CLASSES" in peers_src
        assert "sqlite3.connect" in peers_src
        signing_src = _read(SIGNING_SRC)
        assert "ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS = False" in signing_src


# ---------------------------------------------------------------------------
# Family 10 -- the documentation rolls
# ---------------------------------------------------------------------------


class TestDocs:
    def test_roadmap_rolls_s258(self):
        text = _flat(_read(ROADMAP_PATH))
        assert "carrying the class LANDED at S258" in text

    def test_roadmap_retains_prior_rolls(self):
        # Design-green: the three pinned rolls survive the S258 roll.
        text = _flat(_read(ROADMAP_PATH))
        assert "LANDED at S243" in text
        assert "contract / seam half LANDED at S256" in text
        assert "publisher glue LANDED at S257" in text

    def test_runbook_gains_the_live_ceremony_section(self):
        text = _flat(_read(RUNBOOK_PATH))
        assert "S258" in text
        assert "the live two-device ceremony" in text

    def test_runbook_retains_pins(self):
        # Design-green: the s256 presence pins survive the additive edit.
        text = _flat(_read(RUNBOOK_PATH))
        assert "pairing payload" in text
        assert "republish" in text
        assert "held at 3.12.0" in text
        assert "edit-free" in text


# ---------------------------------------------------------------------------
# Family 11 -- structure: held version, AST, ASCII, untouched cores
# ---------------------------------------------------------------------------


class TestStructure:
    def test_version_held(self):
        # Design-green: a payload / seam slice whose live ceremony is
        # host-assured never bumps.
        ns: dict = {}
        exec(_read(VERSION_PATH), ns)  # noqa: S102 - version file is ours
        assert ns["__version__"] == "3.12.0"

    def test_touched_sources_parse_and_are_ascii(self):
        for path in (PAIRING_SRC, ENGINE_SRC, ROUTES_SYNC_SRC):
            src = _read(path)
            assert src != "", str(path)
            ast.parse(src, filename=str(path))
            assert src == src.encode("ascii", errors="ignore").decode(
                "ascii"
            ), str(path)

    def test_auth_core_pins_hold(self):
        # Design-green: the edit-free constraint's targets exist and parse;
        # byte-identity to pristine is the session-level proof.
        for name in ("auth.py", "auth_2fa.py", "emergency_stop.py"):
            src = _read(PKG / name)
            assert src != "", name
            ast.parse(src, filename=name)
