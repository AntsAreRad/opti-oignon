#!/usr/bin/env python3
"""Tests for S181 Goal 1 -- the live Veilid transport (Theme 4 / Veilid Sync).

Covers opti_oignon/veilid/transport.py and the client's request/response surface
(opti_oignon/veilid/client.py: app_call / aapp_call), plus the sync route's live
peer resolution and its error mapping (opti_oignon/api/routes_sync.py):

- The client app_call surface: a request/response is submitted to the dedicated
  loop and bounded by the timeout; a stall surfaces as VeilidTimeout, an underlying
  error is wrapped as VeilidError, and app_call before connect is a typed error.
- VeilidPeer.fetch: it gates under Bulbe before sending; it serialises the request,
  drives the messenger to the peer's routing key, and parses the reply; a transport
  timeout propagates (the route maps it), while a malformed reply degrades to None
  (the engine treats it as an empty round). A full round through the engine drives
  the live peer end to end against a fake messenger answering from a remote feed.
- The messengers: ClientRouteMessenger forwards to the client with its timeout;
  decode_answer parses bytes / str / dict defensively and never raises.
- resolve_live_peer: None when the framework is absent (the sandbox), None for an
  unpaired peer; with an injected client and a paired peer it builds a VeilidPeer.
- The route surface: the injected resolver drives a paired Daily round to 200; an
  unavailable transport is 503; a peer timeout maps to 504 (fastapi-guarded).

Loaded via spec_from_file_location with opti_oignon stubbed; the security mode is
driven through a stubbed opti_oignon.security_mode and the audit log is a no-op.
The transport is exercised with a fake messenger and a fake client -- no veilid
framework and no live server.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"
API = OO / "api"

_MODE = {"fn": (lambda: "daily")}


def set_mode(value: str = "daily", *, raises: bool = False) -> None:
    if raises:
        def _gm() -> str:
            raise RuntimeError("mode undeterminable")
    else:
        def _gm() -> str:
            return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


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
        al.chain_log = lambda **kwargs: None  # type: ignore[attr-defined]
        sys.modules["opti_oignon.signed_audit_log"] = al


def _load_veilid(name: str):
    full = f"opti_oignon.veilid.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(VEILID / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod  # register before exec (3.12+ dataclass processing)
    spec.loader.exec_module(mod)
    return mod


def _load_api(name: str):
    full = f"opti_oignon.api.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(API / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
guard = _load_veilid("guard")
records = _load_veilid("records")
reconcile = _load_veilid("reconcile")
change_feed = _load_veilid("change_feed")
peers = _load_veilid("peers")
protocol = _load_veilid("protocol")
sync_engine = _load_veilid("sync_engine")
client_mod = _load_veilid("client")
node_mod = _load_veilid("node")
transport = _load_veilid("transport")
rs = _load_api("routes_sync")

RecordKind = records.RecordKind
VeilidClient = client_mod.VeilidClient
VeilidDisabledInBulbe = guard.VeilidDisabledInBulbe
VeilidTimeout = guard.VeilidTimeout
VeilidError = guard.VeilidError


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    node_mod.reset_node()
    rs.reset_peer_resolver()
    yield
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    node_mod.reset_node()
    rs.reset_peer_resolver()
    set_mode("daily")


def _rec(record_id, clock, *, device, payload=None, kind=None):
    return records.new_record(
        kind=kind or RecordKind.CONVERSATION,
        record_id=record_id,
        payload=payload if payload is not None else {"v": clock},
        device=device,
        clock=clock,
    )


# --- Fakes -----------------------------------------------------------------


def _update(name):
    return types.SimpleNamespace(
        attachment=types.SimpleNamespace(state=types.SimpleNamespace(name=name))
    )


class FakeCallAPI:
    """A connected api stand-in exposing app_call; can sleep or fail."""

    def __init__(self, callback, *, reply=b"{}", call_sleep=0.0, fail=False):
        self.callback = callback
        self.calls = []
        self._reply = reply
        self._call_sleep = call_sleep
        self._fail = fail

    async def attach(self):
        self.callback(_update("AttachedGood"))

    async def detach(self):
        self.callback(_update("Detached"))

    async def release(self):
        pass

    async def app_call(self, target, message):
        import asyncio

        self.calls.append((target, message))
        if self._call_sleep:
            await asyncio.sleep(self._call_sleep)
        if self._fail:
            raise ValueError("app_call kaboom")
        return self._reply


def make_call_factory(**kwargs):
    holder = {}

    def factory(callback):
        api = FakeCallAPI(callback, **kwargs)
        holder["api"] = api
        return api

    factory.holder = holder  # type: ignore[attr-defined]
    return factory


class RecordingMessenger:
    """A route messenger that records calls and returns a fixed reply."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def call(self, routing_key, payload, *, timeout=None):
        self.calls.append((routing_key, payload, timeout))
        return self.reply


class FeedMessenger:
    """A messenger that answers a request from a remote change feed (no socket)."""

    def __init__(self, feed, device):
        self.feed = feed
        self.device = device

    def call(self, routing_key, payload, *, timeout=None):
        request = transport.decode_answer(payload)
        batch = protocol.respond_to_request(self.feed, request, device=self.device)
        return transport._encode_message(batch)


class TimeoutMessenger:
    def call(self, routing_key, payload, *, timeout=None):
        raise VeilidTimeout("peer stalled")


class FakeClient:
    """A client stand-in exposing app_call, recording the budget it was given."""

    def __init__(self, reply=b"{}"):
        self.reply = reply
        self.calls = []

    def app_call(self, target, message, *, timeout=None):
        self.calls.append((target, message, timeout))
        return self.reply


class FakeAttachedNode:
    def __init__(self, connector, attached=True):
        self._connector = connector
        self._attached = attached

    def is_attached(self):
        return self._attached

    def connector(self):
        return self._connector


# --- The client request/response surface -----------------------------------


class TestClientAppCall:
    def test_app_call_round_trips(self):
        factory = make_call_factory(reply=b'{"ok":1}')
        c = VeilidClient(api_factory=factory)
        c.connect()
        try:
            out = c.app_call("ROUTE", b"hello")
            assert out == b'{"ok":1}'
            assert factory.holder["api"].calls == [("ROUTE", b"hello")]
        finally:
            c.shutdown()

    def test_app_call_times_out(self):
        factory = make_call_factory(call_sleep=3.0)
        c = VeilidClient(api_factory=factory, timeout=0.3)
        c.connect()
        try:
            with pytest.raises(VeilidTimeout):
                c.app_call("ROUTE", b"slow")
        finally:
            c.shutdown()

    def test_app_call_error_is_wrapped(self):
        factory = make_call_factory(fail=True)
        c = VeilidClient(api_factory=factory)
        c.connect()
        try:
            with pytest.raises(VeilidError) as ei:
                c.app_call("ROUTE", b"x")
            assert not isinstance(ei.value, ValueError)
        finally:
            c.shutdown()

    def test_app_call_before_connect_is_typed(self):
        c = VeilidClient(api_factory=make_call_factory())
        try:
            with pytest.raises(VeilidError):
                c.app_call("ROUTE", b"x")
        finally:
            c.shutdown()

    async def test_aapp_call_off_caller_loop(self):
        import asyncio

        factory = make_call_factory(reply=b'{"a":2}')
        c = VeilidClient(api_factory=factory)
        await c.aconnect()
        try:
            out = await c.aapp_call("ROUTE", b"y")
            assert out == b'{"a":2}'
            assert c._loop is not asyncio.get_running_loop()
        finally:
            await c.ashutdown()


# --- decode_answer / messengers --------------------------------------------


class TestDecodeAnswer:
    def test_bytes(self):
        assert transport.decode_answer(b'{"x":1}') == {"x": 1}

    def test_str(self):
        assert transport.decode_answer('{"y":2}') == {"y": 2}

    def test_dict_passthrough(self):
        assert transport.decode_answer({"z": 3}) == {"z": 3}

    def test_none_and_garbage_are_none(self):
        assert transport.decode_answer(None) is None
        assert transport.decode_answer(b"not json") is None
        assert transport.decode_answer("[1,2,3]") is None  # not a dict
        assert transport.decode_answer(42) is None


class TestClientRouteMessenger:
    def test_forwards_with_default_timeout(self):
        cl = FakeClient(reply=b'{"hi":1}')
        m = transport.ClientRouteMessenger(cl, timeout=2.5)
        out = m.call("RK", b"req")
        assert out == b'{"hi":1}'
        assert cl.calls == [("RK", b"req", 2.5)]

    def test_per_call_timeout_overrides(self):
        cl = FakeClient()
        m = transport.ClientRouteMessenger(cl, timeout=2.5)
        m.call("RK", b"req", timeout=0.5)
        assert cl.calls[-1][2] == 0.5

    def test_requires_client(self):
        with pytest.raises(ValueError):
            transport.ClientRouteMessenger(None)


# --- VeilidPeer.fetch -------------------------------------------------------


class TestVeilidPeerFetch:
    def test_fetch_sends_request_and_parses_reply(self):
        m = RecordingMessenger(reply=b'{"v":1,"type":"record_batch","device":"B","high_water":0,"records":[]}')
        peer = transport.VeilidPeer(m, "RKB", device="A")
        out = peer.fetch({"v": 1, "type": "delta_request", "device": "A", "watermark": 0})
        assert out["type"] == "record_batch"
        # the request was serialised and addressed to the routing key
        rk, payload, _ = m.calls[0]
        assert rk == "RKB"
        assert transport.decode_answer(payload)["type"] == "delta_request"

    def test_fetch_refuses_in_bulbe_before_sending(self):
        m = RecordingMessenger(reply=b"{}")
        peer = transport.VeilidPeer(m, "RKB")
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            peer.fetch({"v": 1, "type": "delta_request", "device": "A", "watermark": 0})
        assert m.calls == []  # the gate precedes the send

    def test_fetch_refuses_when_mode_undeterminable(self):
        m = RecordingMessenger(reply=b"{}")
        peer = transport.VeilidPeer(m, "RKB")
        set_mode(raises=True)
        with pytest.raises(VeilidDisabledInBulbe):
            peer.fetch({"v": 1, "type": "delta_request", "device": "A", "watermark": 0})

    def test_timeout_propagates(self):
        peer = transport.VeilidPeer(TimeoutMessenger(), "RKB")
        with pytest.raises(VeilidTimeout):
            peer.fetch({"v": 1, "type": "delta_request", "device": "A", "watermark": 0})

    def test_malformed_reply_is_none(self):
        peer = transport.VeilidPeer(RecordingMessenger(reply=b"garbage"), "RKB")
        assert peer.fetch({"v": 1, "type": "delta_request", "device": "A", "watermark": 0}) is None

    def test_requires_routing_key(self):
        with pytest.raises(ValueError):
            transport.VeilidPeer(RecordingMessenger(reply=b"{}"), "")

    def test_requires_messenger(self):
        with pytest.raises(ValueError):
            transport.VeilidPeer(None, "RKB")


# --- A full round through the engine over the live peer --------------------


class TestRoundOverLivePeer:
    def _engine(self, tmp_path, device="A"):
        feed = change_feed.ChangeFeed(root=tmp_path / "local")
        store = peers.PeerStore(root=tmp_path / "store")
        return sync_engine.SyncEngine(device=device, feed=feed, store=store), feed, store

    def test_round_applies_over_live_peer(self, tmp_path):
        eng, feed, store = self._engine(tmp_path)
        store.add_peer("B", "RKB")
        rfeed = change_feed.ChangeFeed(root=tmp_path / "remote")
        rfeed.record(_rec("c1", 1, device="B"))
        rfeed.record(_rec("c2", 1, device="B"))
        peer = transport.VeilidPeer(FeedMessenger(rfeed, "B"), "RKB", device="A")
        res = eng.run_round("B", peer)
        assert res.applied == 2
        assert res.advanced is True
        keys = {(r.kind.value, r.record_id) for r in feed.current_records()}
        assert keys == {("conversation", "c1"), ("conversation", "c2")}

    def test_round_timeout_propagates_through_engine(self, tmp_path):
        eng, feed, store = self._engine(tmp_path)
        store.add_peer("B", "RKB")
        peer = transport.VeilidPeer(TimeoutMessenger(), "RKB")
        with pytest.raises(VeilidTimeout):
            eng.run_round("B", peer)

    def test_round_malformed_reply_holds_watermark(self, tmp_path):
        eng, feed, store = self._engine(tmp_path)
        store.add_peer("B", "RKB")
        store.advance_watermark("B", 3)
        peer = transport.VeilidPeer(RecordingMessenger(reply=b"garbage"), "RKB")
        res = eng.run_round("B", peer)
        assert res.applied == 0
        assert res.advanced is False
        assert res.new_watermark == 3


# --- resolve_live_peer ------------------------------------------------------


class TestResolveLivePeer:
    def test_none_without_framework(self, tmp_path):
        # veilid is absent in the sandbox -> the production resolver returns None.
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "RKB")
        assert transport.resolve_live_peer("B", store=store) is None

    def test_none_for_unpaired_even_if_forced_available(self, tmp_path, monkeypatch):
        store = peers.PeerStore(root=tmp_path / "store")
        monkeypatch.setattr(transport, "veilid_available", lambda: True)
        assert transport.resolve_live_peer("ghost", store=store) is None

    def test_builds_peer_with_injected_client(self, tmp_path, monkeypatch):
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "RKB", label="phone")
        monkeypatch.setattr(transport, "veilid_available", lambda: True)
        cl = FakeClient(reply=b'{"v":1,"type":"record_batch","device":"B","high_water":0,"records":[]}')
        peer = transport.resolve_live_peer("B", store=store, client=cl, device="A")
        assert isinstance(peer, transport.VeilidPeer)
        assert peer.routing_key == "RKB"

    def test_builds_peer_from_attached_node(self, tmp_path, monkeypatch):
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "RKB")
        monkeypatch.setattr(transport, "veilid_available", lambda: True)
        cl = FakeClient()
        node = FakeAttachedNode(cl, attached=True)
        peer = transport.resolve_live_peer("B", store=store, node=node)
        assert isinstance(peer, transport.VeilidPeer)

    def test_none_when_node_not_attached(self, tmp_path, monkeypatch):
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "RKB")
        monkeypatch.setattr(transport, "veilid_available", lambda: True)
        node = FakeAttachedNode(FakeClient(), attached=False)
        assert transport.resolve_live_peer("B", store=store, node=node) is None


# --- Route surface: injected resolver + error mapping (fastapi-guarded) -----


class TestRouteLiveSurface:
    @pytest.fixture
    def client(self, tmp_path):
        fastapi = pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        if rs.router is None:  # pragma: no cover - defensive
            pytest.skip("sync router unavailable")
        st = peers.PeerStore(root=tmp_path / "store")
        feed = change_feed.ChangeFeed(root=tmp_path / "local")
        peers.set_peer_store(st)
        sync_engine.set_sync_engine(
            sync_engine.SyncEngine(device="A", feed=feed, store=st)
        )
        app = fastapi.FastAPI()
        assert rs.register(app) is True
        return TestClient(app), st, tmp_path

    def test_injected_resolver_runs_round_200(self, client):
        tc, st, tmp_path = client
        st.add_peer("B", "RKB")
        rfeed = change_feed.ChangeFeed(root=tmp_path / "remote")
        rfeed.record(_rec("c1", 1, device="B"))
        rs.set_peer_resolver(
            lambda pid, store: transport.VeilidPeer(FeedMessenger(rfeed, "B"), "RKB")
        )
        r = tc.post("/api/sync/peers/B/run")
        assert r.status_code == 200
        body = r.json()
        assert body["applied"] == 1
        assert body["advanced"] is True

    def test_transport_unavailable_is_503(self, client):
        # No resolver injected and veilid absent -> production resolver yields None.
        tc, st, _ = client
        st.add_peer("B", "RKB")
        r = tc.post("/api/sync/peers/B/run")
        assert r.status_code == 503

    def test_peer_timeout_is_504(self, client):
        tc, st, _ = client
        st.add_peer("B", "RKB")
        rs.set_peer_resolver(
            lambda pid, store: transport.VeilidPeer(TimeoutMessenger(), "RKB")
        )
        r = tc.post("/api/sync/peers/B/run")
        assert r.status_code == 504

    def test_bulbe_still_403_before_resolution(self, client):
        tc, st, _ = client
        st.add_peer("B", "RKB")
        rs.set_peer_resolver(
            lambda pid, store: (_ for _ in ()).throw(AssertionError("must not resolve"))
        )
        set_mode("bulbe")
        r = tc.post("/api/sync/peers/B/run")
        assert r.status_code == 403
