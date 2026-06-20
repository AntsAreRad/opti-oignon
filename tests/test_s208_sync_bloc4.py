#!/usr/bin/env python3
"""S208 -- Sync cycle Bloc 4: the grace flip, the republish surface, 3.7.0.

The lot under test: the VL-01 migration window is CLOSED --
``signing.ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS`` is False as a hard constant,
so an unsigned record from an origin with no registered signing key REFUSES
like the rest (counted, never applied, never holding the watermark), while the
pre-VL-01 verify-incapable posture is unchanged. The fleet order ships with
the flip: ``POST /api/sync/republish`` and the SyncPanel button expose the
one-time ``republish_signed`` ceremony step (mode-free, audited, honest 503
when signing is unavailable). The panel polish surfaces a failed
pending-approvals load and the unverified round count. The 3.7.0 release:
version sites, CHANGELOG (the cycle story), README/docs, the closed roadmap,
and the prepared host walk (BLOC4_LIVE_WALK.md).

Designed supersessions (deselect-plus-reassert; originals never edited):
- test_s205_sync_bloc2_lot1.py::TestGraceWindow::test_unkeyed_origin_accepted_and_counted_under_grace
  (pinned the True default and the open-window acceptance)
- test_s205_sync_bloc2_lot1.py::TestGraceWindow::test_grace_never_admits_unsigned_from_keyed_origin
  (pin only; the keyed-origin refusal itself is re-asserted here)
- test_s206_sync_bloc2_lot2.py::TestPendingStateMachine::test_verify_refuses_pending_origin_never_grace
  (pin only; the pending-origin refusal itself is re-asserted here)
- test_s207_sync_bloc3.py::TestApprovalPath::test_grace_closed_unkeyed_origin_refuses_at_approval
  (its deferral leg ran under the open default; re-created here with the
  window re-opened by monkeypatch for that leg)
- test_s182_release.py::TestVersionBump::test_version_file_is_final
- test_s182_release.py::TestVersionBump::test_version_file_contains_360
- test_s182_release.py::TestVersionBump::test_pyproject_version_is_final
- test_s182_release.py::TestChangelog::test_top_entry_is_360
- test_s182_release.py::TestOptionalGroups::test_version_is_hardcoded
  (the five 3.6.0 pins; each re-asserted at 3.7.0 here, alongside the s197
  f10a/f10b/f10d version pins added to the pyproject addopts lineage)

Harness: the s205/s206/s207 idiom -- opti_oignon stubbed, the security mode
driven, the audit log a recorder, FakeSigner per device, the real protocol
responder as the fake peer, routes driven through TestClient with injected
singletons.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"
API = OO / "api"
FRONTEND = ROOT / "frontend" / "src" / "lib"

_MODE = {"fn": lambda: "daily"}
_AUDIT: dict = {"events": []}


def set_mode(value: str = "daily", *, raises: bool = False) -> None:
    if raises:
        def _gm() -> str:
            raise RuntimeError("mode undeterminable")
    else:
        def _gm() -> str:
            return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


def _record_audit(**kwargs):
    _AUDIT["events"].append(kwargs)


def _ensure_stubs() -> None:
    for name, sub in (
        ("opti_oignon", OO),
        ("opti_oignon.veilid", VEILID),
        ("opti_oignon.api", API),
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


_ensure_stubs()

signing_mod = importlib.import_module("opti_oignon.veilid.signing")
guard = importlib.import_module("opti_oignon.veilid.guard")
_change_feed_mod = importlib.import_module("opti_oignon.veilid.change_feed")
_peers_mod = importlib.import_module("opti_oignon.veilid.peers")
_records_mod = importlib.import_module("opti_oignon.veilid.records")
_ledger_mod = importlib.import_module("opti_oignon.veilid.deferred_ledger")
_engine_mod = importlib.import_module("opti_oignon.veilid.sync_engine")
_status_mod = importlib.import_module("opti_oignon.veilid.sync_status")
rs = importlib.import_module("opti_oignon.api.routes_sync")

ChangeFeed = _change_feed_mod.ChangeFeed
PeerStore = _peers_mod.PeerStore
DeferredLedger = _ledger_mod.DeferredLedger
RecordKind = _records_mod.RecordKind
new_record = _records_mod.new_record
attach_signature = signing_mod.attach_signature
SigningUnavailable = signing_mod.SigningUnavailable
SyncEngine = _engine_mod.SyncEngine
VeilidDisabledInBulbe = guard.VeilidDisabledInBulbe


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]
    _AUDIT["events"].clear()
    _change_feed_mod.reset_change_feed()
    _peers_mod.reset_peer_store()
    _ledger_mod.reset_deferred_ledger()
    _engine_mod.reset_sync_engine()
    _status_mod.reset_sync_status_store()
    signing_mod.reset_record_signer()
    rs.reset_peer_resolver()
    rs.reset_self_routing_resolver()
    yield
    _change_feed_mod.reset_change_feed()
    _peers_mod.reset_peer_store()
    _ledger_mod.reset_deferred_ledger()
    _engine_mod.reset_sync_engine()
    _status_mod.reset_sync_status_store()
    signing_mod.reset_record_signer()
    rs.reset_peer_resolver()
    rs.reset_self_routing_resolver()
    set_mode("daily")


# ---------------------------------------------------------------------------
# The deterministic fake signer and the unavailable one (the S205 seams)
# ---------------------------------------------------------------------------


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
        expected_like = hmac_mod.new(
            self._mac_key_for(public_key), data, hashlib.sha256
        ).digest()
        return hmac_mod.compare_digest(expected_like, signature)

    def _mac_key_for(self, public_key: bytes) -> bytes:
        secret = _PUB_REGISTRY.get(public_key)
        return (secret or b"\x00") + public_key


_PUB_REGISTRY: dict[bytes, bytes] = {}


def make_signer(seed: str) -> FakeSigner:
    secret = hashlib.sha256(seed.encode()).digest()
    s = FakeSigner(secret)
    _PUB_REGISTRY[s.public_key()] = secret
    return s


class UnavailableSigner:
    """A signer whose backend is absent: sign raises, verify_available False."""

    def public_key(self) -> bytes:
        raise SigningUnavailable("no backend")

    def sign(self, data: bytes) -> bytes:
        raise SigningUnavailable("no backend")

    def verify(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        return False

    def verify_available(self) -> bool:
        return False


def b64(raw: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Records, peers, engines
# ---------------------------------------------------------------------------


def conv(rid: str, clock: int = 1, *, device: str = "dev-a") -> object:
    return new_record(
        RecordKind.CONVERSATION,
        rid,
        {"title": "hello " + rid},
        device=device,
        clock=clock,
        updated_at="2026-01-01T00:00:00+00:00",
    )


def skill(rid: str = "s1", clock: int = 1, *, device: str = "dev-a") -> object:
    return new_record(
        RecordKind.SKILL,
        rid,
        {"body": "code"},
        device=device,
        clock=clock,
        updated_at="2026-01-01T00:00:00+00:00",
    )


class CountingPeer:
    """A peer answering from its own feed through the real responder."""

    def __init__(self, feed, device: str) -> None:
        self._feed = feed
        self._device = device
        self.fetch_calls = 0

    def fetch(self, request):
        from opti_oignon.veilid.protocol import respond_to_request

        self.fetch_calls += 1
        return respond_to_request(self._feed, request, device=self._device)


def asker(tmp_path, *, device="dev-b", seed="asker-b", signer=None):
    """An asking engine: its own feed, store, ledger, and signer."""
    feed = ChangeFeed(root=tmp_path / device)
    store = PeerStore(root=tmp_path / device)
    ledger = DeferredLedger(root=tmp_path / device)
    eng = SyncEngine(
        device=device,
        feed=feed,
        store=store,
        signer=signer if signer is not None else make_signer(seed),
        ledger=ledger,
    )
    return eng, feed, store, ledger


def audit_actions() -> list:
    return [a.get("action") for a in _AUDIT["events"]]


def _read(*parts) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def _norm(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# A. The grace flip: the closed window and every surviving refusal
# ---------------------------------------------------------------------------


class TestGraceFlip:
    def test_default_is_false_pinned(self):
        # The S208 posture: the migration window is closed, hard constant.
        assert signing_mod.ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS is False

    def test_unkeyed_origin_unsigned_refuses_at_round(self, tmp_path):
        # The flip's core behaviour (supersedes the s205 open-window
        # acceptance): no registered key for the origin, no grace -- the
        # record refuses, is never applied, and never holds the watermark.
        eng, feed, store, ledger = asker(tmp_path)
        o_feed = ChangeFeed(root=tmp_path / "raw-a")
        o_feed.record(conv("c1", device="dev-a"))  # journalled unsigned
        store.add_peer("dev-a", "rk")  # no signing key registered
        result = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"))
        assert result.refused == 1
        assert result.unverified == 0
        assert result.applied == 0
        assert result.advanced is True  # a refusal never pins convergence
        assert all(r.record_id != "c1" for r in feed.current_records())

    def test_keyed_origin_unsigned_still_refuses(self, tmp_path):
        # Reassert (s205 supersession): the keyed-origin refusal never
        # depended on the window -- unchanged under the closed default.
        signer_a = make_signer("origin-a")
        eng, feed, store, ledger = asker(tmp_path)
        o_feed = ChangeFeed(root=tmp_path / "raw-a")
        o_feed.record(conv("c1", device="dev-a"))  # unsigned, origin keyed
        store.add_peer("dev-a", "rk", signing_pub=b64(signer_a.public_key()))
        result = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"))
        assert result.refused == 1 and result.unverified == 0
        assert result.applied == 0

    def test_pending_origin_still_refuses_never_grace(self, tmp_path):
        # Reassert (s206 supersession): the relay case. B pulls from
        # CONFIRMED peer C whose feed carries a record ORIGINATED by dev-a,
        # correctly signed by dev-a's key, which B holds for a PENDING dev-a
        # entry. Refused (counted), never trusted, never grace-admitted --
        # and with the window closed there is no grace to fall into anyway.
        signer_a = make_signer("origin-a")
        signer_c = make_signer("relay-c")
        feed_c = ChangeFeed(root=tmp_path / "c")
        engine_c = SyncEngine(device="dev-c", feed=feed_c, signer=signer_c)
        signed_a = attach_signature(conv("z1", device="dev-a"), signer_a)
        engine_c.publish(signed_a)  # foreign provenance, journalled verbatim
        engine_c.publish_conversation("c2", {"title": "own"}, clock=1)

        eng, feed, store, ledger = asker(tmp_path)
        store.add_peer("dev-c", "rk-c", signing_pub=b64(signer_c.public_key()))
        store.add_peer(
            "dev-a", "rk-a", signing_pub=b64(signer_a.public_key()), pending=True
        )
        result = eng.run_round("dev-c", CountingPeer(feed_c, "dev-c"))
        assert result.refused == 1  # the pending-origin record
        assert result.unverified == 0  # never any grace path
        assert result.applied == 1  # dev-c's own record applies
        assert result.advanced is True

    def test_monkeypatch_reopens_historical_window(self, tmp_path, monkeypatch):
        # The seam stays testable: the constant is read at call time, so a
        # monkeypatch restores the documented S205..S207 behaviour.
        eng, feed, store, ledger = asker(tmp_path)
        o_feed = ChangeFeed(root=tmp_path / "raw-a")
        o_feed.record(conv("c1", device="dev-a"))
        store.add_peer("dev-a", "rk")
        monkeypatch.setattr(
            signing_mod, "ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS", True
        )
        result = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"))
        assert result.unverified == 1 and result.refused == 0
        assert result.applied == 1

    def test_grace_open_deferral_then_closed_approval_refuses(
        self, tmp_path, monkeypatch
    ):
        # Reassert (s207 supersession), the historical scenario re-created:
        # a sensitive unsigned record from an unkeyed origin entered the
        # ledger while the window was OPEN (re-opened here by monkeypatch);
        # by approval time the default (closed) stands: the re-verification
        # refuses instead of applying, and the entry is removed.
        eng, feed, store, ledger = asker(tmp_path)
        o_feed = ChangeFeed(root=tmp_path / "raw-a")
        o_feed.record(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk")  # no signing key registered
        monkeypatch.setattr(
            signing_mod, "ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS", True
        )
        res = eng.run_round(
            "dev-a", CountingPeer(o_feed, "dev-a"),
            approval_fn=lambda c, l, a: False,
        )
        assert res.deferred == 1 and res.unverified == 1
        monkeypatch.setattr(
            signing_mod, "ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS", False
        )
        out = eng.approve_deferred("skill", "s1")
        assert out["refused"] is True and out["applied"] == 0
        assert ledger.count() == 0
        assert all(r.record_id != "s1" for r in feed.current_records())

    def test_verify_incapable_posture_unchanged(self, tmp_path):
        # The pre-VL-01 branch (no PQC backend) is NOT the grace: it accepts
        # everything as unverified with a warning, flip or no flip --
        # refusing what it cannot check would partition the fleet.
        feed_b = ChangeFeed(root=tmp_path / "b")
        store_b = PeerStore(root=tmp_path / "b")
        eng = SyncEngine(
            device="dev-b", feed=feed_b, store=store_b, signer=UnavailableSigner()
        )
        o_feed = ChangeFeed(root=tmp_path / "raw-a")
        o_feed.record(conv("c1", device="dev-a"))
        store_b.add_peer("dev-a", "rk")
        result = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"))
        assert result.applied == 1
        assert result.unverified == 1
        assert result.refused == 0

    def test_republish_signed_under_closed_window(self, tmp_path):
        # Reassert (the s205 republish behaviour, now under the closed
        # default): the local unsigned set re-journals WITH signatures at the
        # same clocks; a foreign record is untouched.
        feed = ChangeFeed(root=tmp_path / "b")
        unsigned_engine = SyncEngine(
            device="dev-b", feed=feed, signer=UnavailableSigner()
        )
        unsigned_engine.publish_conversation("c1", {"title": "x"}, clock=1)
        unsigned_engine.publish_conversation("gone", {}, clock=1, deleted=True)
        foreign = attach_signature(
            conv("z1", device="dev-z"), make_signer("z")
        )
        feed.record(foreign)
        engine = SyncEngine(device="dev-b", feed=feed, signer=make_signer("b"))
        count = engine.republish_signed()
        assert count == 2  # c1 + the tombstone; the foreign record untouched
        by_id = {r.record_id: r for r in feed.current_records()}
        assert by_id["c1"].signature and by_id["gone"].signature
        assert by_id["z1"].signature == foreign.signature

    def test_flip_documented_with_fleet_order_and_hardness(self):
        # The fleet-order documentation ships WITH the flip (the mandate):
        # the constant's comment names the order, the honest recovery, and
        # the hard-flip decision; the module docstring states the closure.
        src = _read("opti_oignon", "veilid", "signing.py")
        assert "ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS = False" in src
        assert "ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS = True" not in src
        norm = _norm(src)
        assert "hard constant" in norm
        assert "Fleet upgrade order" in norm
        assert "republish_signed" in norm
        assert "re-arrive SIGNED" in norm
        assert "CLOSED at S208" in norm
        assert "verify-incapable posture" in norm


# ---------------------------------------------------------------------------
# B. The republish operator surface: helper, route, mode posture, audit
# ---------------------------------------------------------------------------


class TestRepublishSurface:
    def setup_method(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

    def _client(self, tmp_path, *, signer=None):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        feed = ChangeFeed(root=tmp_path / "local")
        store = PeerStore(root=tmp_path / "store")
        ledger = DeferredLedger(root=tmp_path / "ledger")
        eng = SyncEngine(
            device="dev-b",
            feed=feed,
            store=store,
            signer=signer if signer is not None else make_signer("router-b"),
            ledger=ledger,
        )
        _peers_mod.set_peer_store(store)
        _engine_mod.set_sync_engine(eng)
        _ledger_mod.set_deferred_ledger(ledger)
        app = FastAPI()
        rs.register(app)
        return TestClient(app), eng, feed, store, ledger

    def _seed_unsigned(self, feed, n: int = 2):
        unsigned = SyncEngine(
            device="dev-b", feed=feed, signer=UnavailableSigner()
        )
        for i in range(n):
            unsigned.publish_conversation(f"c{i}", {"title": str(i)}, clock=1)

    def test_payload_helper_shape(self, tmp_path):
        feed = ChangeFeed(root=tmp_path / "b")
        self._seed_unsigned(feed, 1)
        eng = SyncEngine(device="dev-b", feed=feed, signer=make_signer("b"))
        assert rs.republish_payload(eng) == {"republished": 1}

    def test_route_republishes_and_counts(self, tmp_path):
        client, eng, feed, store, ledger = self._client(tmp_path)
        self._seed_unsigned(feed, 2)
        r = client.post("/api/sync/republish")
        assert r.status_code == 200
        assert r.json() == {"republished": 2}
        assert all(rec.signature for rec in feed.current_records())

    def test_route_idempotent_second_call_zero(self, tmp_path):
        client, eng, feed, store, ledger = self._client(tmp_path)
        self._seed_unsigned(feed, 1)
        assert client.post("/api/sync/republish").json()["republished"] == 1
        # Nothing unsigned remains: the honest 0.
        assert client.post("/api/sync/republish").json()["republished"] == 0

    def test_route_503_when_signing_unavailable(self, tmp_path):
        # The caller asked for a signed set and must know: never a silent
        # unsigned republish.
        client, eng, feed, store, ledger = self._client(
            tmp_path, signer=UnavailableSigner()
        )
        self._seed_unsigned(feed, 1)
        r = client.post("/api/sync/republish")
        assert r.status_code == 503
        assert all(not rec.signature for rec in feed.current_records())

    def test_republish_mode_free_under_bulbe_while_round_refuses(self, tmp_path):
        # The producers posture: signing the local set is a local edit,
        # permitted in any mode; only the wire is Daily-gated.
        client, eng, feed, store, ledger = self._client(tmp_path)
        self._seed_unsigned(feed, 1)
        store.add_peer("dev-a", "rk")
        set_mode("bulbe")
        assert (
            client.post("/api/sync/peers/dev-a/run").status_code == 403
        )  # the wire refuses
        r = client.post("/api/sync/republish")
        assert r.status_code == 200 and r.json()["republished"] == 1

    def test_republish_audited(self, tmp_path):
        client, eng, feed, store, ledger = self._client(tmp_path)
        self._seed_unsigned(feed, 1)
        assert client.post("/api/sync/republish").status_code == 200
        assert "republish_signed" in audit_actions()


# ---------------------------------------------------------------------------
# C. The SyncPanel polish slice (source assertions; no new component)
# ---------------------------------------------------------------------------


class TestPanelPolish:
    def setup_method(self):
        self.panel = _read(
            "frontend", "src", "lib", "components", "panels", "SyncPanel.svelte"
        )
        self.api = _read("frontend", "src", "lib", "api", "sync.ts")

    def test_api_mirror_has_republish(self):
        assert "export async function republishSigned" in self.api
        assert "/republish" in self.api

    def test_panel_imports_handler_and_button(self):
        assert "republishSigned," in self.panel
        assert "async function republishNow()" in self.panel
        assert "Republish signed records" in self.panel
        assert "Nothing to republish" in self.panel  # the honest 0 toast

    def test_deferred_load_failure_surfaced_not_silent(self):
        # A failed pending-approvals load must be distinguishable from an
        # empty queue.
        assert "deferredError" in self.panel
        assert "Pending approvals could not be loaded" in self.panel
        assert (
            "listDeferredRecords().catch(() => ({ deferred: [], count: 0 }))"
            not in self.panel
        )

    def test_round_toast_reports_unverified_and_points_below(self):
        assert "unverified (no signature check)" in self.panel
        assert "awaiting approval below" in self.panel

    def test_pairing_hint_names_the_signing_key(self):
        assert "signing keys plus an integrity check" in self.panel
        assert "carries a public routing key and an integrity check" not in self.panel

    def test_token_hygiene_no_raw_hex(self):
        for m in re.finditer(r"#[0-9a-fA-F]{3,8}\b", self.panel):
            ctx = self.panel[max(0, m.start() - 40):m.start()]
            assert "var(--oo-" in ctx, f"raw hex outside token fallback: {m.group()}"

    def test_block_balance_holds(self):
        opens = len(re.findall(r"\{#(if|each|await)\b", self.panel))
        closes = len(re.findall(r"\{/(if|each|await)\}", self.panel))
        assert opens == closes

    def test_panel_still_registered_no_new_component(self):
        spec = _read("FRONTEND_REDESIGN_SPEC.md")
        assert "SyncPanel" in spec


# ---------------------------------------------------------------------------
# D. The 3.7.0 release: version sites, CHANGELOG, docs, roadmap, the walk
# ---------------------------------------------------------------------------


FINAL_VERSION = "3.7.0"


class TestVersionRelease:
    """Reasserts the five superseded s182 pins (and the f10 ones) at 3.7.0."""

    def test_version_file_is_370(self):
        src = _read("opti_oignon", "__version__.py")
        assert '"3.7.0"' in src
        assert "3.6.0" not in src

    def test_version_bare_no_rc(self):
        m = re.search(r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py"))
        assert m and re.match(r"^\d+\.\d+\.\d+$", m.group(1))
        assert m.group(1) == FINAL_VERSION

    def test_pyproject_version_is_370_and_hardcoded(self):
        src = _read("pyproject.toml")
        assert f'version = "{FINAL_VERSION}"' in src
        assert 'version = "3.6.0"' not in src
        import tomllib

        data = tomllib.loads(src)
        assert "dynamic" not in data["project"]
        assert data["project"]["version"] == FINAL_VERSION

    def test_pyproject_consistent_with_version_file(self):
        m = re.search(r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py"))
        import tomllib

        data = tomllib.loads(_read("pyproject.toml"))
        assert data["project"]["version"] == m.group(1)

    def test_addopts_carries_the_version_supersessions(self):
        src = _read("pyproject.toml")
        for node in (
            "test_s182_release.py::TestVersionBump::test_version_file_is_final",
            "test_s182_release.py::TestVersionBump::test_version_file_contains_360",
            "test_s182_release.py::TestVersionBump::test_pyproject_version_is_final",
            "test_s182_release.py::TestChangelog::test_top_entry_is_360",
            "test_s182_release.py::TestOptionalGroups::test_version_is_hardcoded",
            "test_s197_f10a.py::test_ds03_version_file_is_360",
            "test_s197_f10b.py::test_version_file_is_360_f10b",
            "test_s197_f10d.py::test_version_hardcoded_not_dynamic",
        ):
            assert f"--deselect=tests/{node}" in src, node


class TestChangelogRelease:
    def setup_method(self):
        self.c = _read("CHANGELOG.md")

    def test_top_entry_is_370(self):
        entries = re.findall(r"## v(\d+\.\d+\.\d+)", self.c)
        assert entries and entries[0] == FINAL_VERSION

    def test_unreleased_block_absorbed(self):
        assert "## Unreleased" not in self.c

    def test_entry_tells_the_cycle_story(self):
        entry = self.c.split("## v3.7.0")[1].split("## v3.6.0")[0]
        for term in (
            "SYN-01", "PRT-04", "CHF-05", "VL-01", "PAIR-02", "SYN-05",
            "[SECURITY]", "Kerckhoffs", "Bulbe", "republish", "grace",
        ):
            assert term in entry, term

    def test_entry_states_the_fleet_order_and_recovery(self):
        entry = _norm(self.c.split("## v3.7.0")[1].split("## v3.6.0")[0])
        assert "Fleet upgrade order" in entry
        assert "refused" in entry
        assert "nothing is lost" in entry

    def test_entry_records_the_declined_liveness(self):
        entry = _norm(self.c.split("## v3.7.0")[1].split("## v3.6.0")[0])
        assert "challenge-response" in entry and "DECLINED" in entry

    def test_prior_entries_retained(self):
        assert "## v3.6.0 -- 2026-06-02 (S182)" in self.c
        assert "## v3.5.0 -- 2026-06-02 (S177)" in self.c
        assert "## v3.4.0 -- 2026-06-01 (S171)" in self.c


class TestDocsRelease:
    def test_readme_refreshed_to_370_with_sync_section(self):
        src = _read("README.md")
        assert "Opti-Oignon v3.7.0 sits between" in src
        assert "Opti-Oignon v3.5.0 sits between" not in src
        assert "## Features Added in v3.7.0" in src
        assert "Republish signed records" in src

    def test_sync_docs_page_carries_ceremony_and_fleet_order(self):
        src = _norm(_read("docs", "sync", "veilid-sync.md"))
        assert "## Record signing and the 3.7.0 upgrade order" in src
        assert "Republish signed records" in src
        assert "confirmation code" in src
        assert "pending-approvals ledger" in src

    def test_spec_section8_grace_closed_and_liveness_declined(self):
        src = _norm(_read("VEILID_SPEC.md"))
        assert "window CLOSED at S208" in src
        assert "DECLINED at S208" in src
        assert "loose end is formally closed" in src
        assert "test_s208_sync_bloc4" in src  # the section 12 row

    def test_roadmap_marked_closed_with_residue_routed(self):
        src = _read("ROADMAP_SYNC_CYCLE.md")
        assert "Status: CLOSED at S208" in src
        assert "BLOC4_LIVE_WALK.md" in src
        assert "DONE: 3.7.0 shipped at S208" in src


class TestLiveWalkPrepared:
    def setup_method(self):
        self.walk = _read("BLOC4_LIVE_WALK.md")

    def test_walk_document_has_the_nine_items(self):
        for n in range(1, 10):
            assert f"## W{n}." in self.walk, f"W{n}"

    def test_walk_covers_the_mandated_sequence(self):
        norm = _norm(self.walk)
        for term in (
            "VLD-03",
            "demotion drill",
            "conflict",
            "tombstone",
            "watermark ADVANCES",
            "Rotate A's origin signing key",
            "Bulbe",
            "Republish signed records",
            "refused count falls to zero",
        ):
            assert term in norm, term

    def test_walk_states_it_is_host_bound_not_simulated(self):
        norm = _norm(self.walk)
        assert "PREPARES but cannot execute" in norm
        assert "real machine" in norm

    def test_handoff_delegates_to_the_walk(self):
        src = _read("SHAKEDOWN_S198_HANDOFF.md")
        assert src.count("BLOC4_LIVE_WALK.md") >= 2  # section C + Routing
        assert "LANDED at S206" in src  # the PAIR-02 note refreshed
