"""S235 -- cas 7 Lot 2: rate limiting, the control surface, the panel.

Container-provable proof of the abuse-control gate (REMOTE_INFERENCE_SPEC
section 7), the desktop control surface (section 11, SYN-06 auth parity), and
the SyncPanel-family panel registration (section 11, FRONTEND_REDESIGN_SPEC).

What is proven here:

  - the per-device rate-limit gate: a fixed window per device, a structured
    refusal on breach (the Lot 1 envelope shape, reason ``rate_limited``), and an
    alert recorded in the channel telemetry the panel reads; the window resets on
    a deterministic injected clock;
  - the handler wires the gate: a request from a device over its rate is refused
    ``rate_limited`` BEFORE the funnel (an alert, never absorbed work);
  - the control-surface payload helpers: view a device's grant, set
    enable/disable plus the RAG sub-grant, revoke (which kills live sessions),
    and read the channel rate/telemetry state;
  - SYN-06 auth parity: the new endpoints sit on the ``/api/sync`` router, which
    carries the router-level authentication dependency, so they inherit it;
  - the SyncPanel family is the home: a new ``RemoteChannelPanel.svelte``
    registered in FRONTEND_REDESIGN_SPEC, mounted in ``SyncPanel.svelte``, with
    the API client functions in ``sync.ts``; the Svelte tags balance.

Red-before on the pristine tree: the rate gate, the telemetry, the payload
helpers, the routes, and the panel do not exist, so every assertion is RED.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


def _streaming():
    try:
        return importlib.import_module("opti_oignon.veilid.remote_streaming")
    except Exception:
        return None


def _remote_inference():
    try:
        return importlib.import_module("opti_oignon.veilid.remote_inference")
    except Exception:
        return None


def _routes_sync():
    try:
        return importlib.import_module("opti_oignon.api.routes_sync")
    except Exception:
        return None


protocol = importlib.import_module("opti_oignon.veilid.protocol")


class FakeRecord:
    def __init__(self, *, pending=False, signing_pub="PUB",
                 remote_chat_enabled=True, rag_subgrant=False):
        self.pending = pending
        self.signing_pub = signing_pub
        self.remote_chat_enabled = remote_chat_enabled
        self.rag_subgrant = rag_subgrant


class FakePeerStore:
    """A peer store stand-in with the grant setters the control surface drives."""

    def __init__(self, peers=None):
        self._peers = dict(peers or {})

    def get_peer(self, peer_id):
        return self._peers.get(peer_id)

    def set_remote_chat_grant(self, peer_id, enabled):
        rec = self._peers.get(peer_id)
        if rec is None:
            return False
        rec.remote_chat_enabled = bool(enabled)
        return True

    def set_rag_subgrant(self, peer_id, granted):
        rec = self._peers.get(peer_id)
        if rec is None:
            return False
        rec.rag_subgrant = bool(granted)
        return True


class _AuditSpy:
    def __init__(self):
        self.events = []

    def __call__(self, action, **details):
        self.events.append((action, details))


def _fake_router(prompt):
    import types

    return types.SimpleNamespace(model="fake-model")


class FakeExecutor:
    def __init__(self, reply="canned", boom=False):
        self.calls = []
        self._reply = reply
        self._boom = boom

    def execute(self, question, routing, **kwargs):
        self.calls.append(question)
        if self._boom:
            raise AssertionError("the funnel must not be entered")
        reply = self._reply

        def _gen():
            yield reply
            return (reply, "chat")

        return _gen()


@pytest.fixture(autouse=True)
def _reset_streaming():
    mod = _streaming()
    if mod is not None and hasattr(mod, "reset_for_tests"):
        mod.reset_for_tests()
    yield
    if mod is not None and hasattr(mod, "reset_for_tests"):
        mod.reset_for_tests()


# ---------------------------------------------------------------------------
# Family 1 -- the per-device rate-limit gate (fixed window, injected clock)
# ---------------------------------------------------------------------------


class TestRateLimitGate:
    def test_allows_up_to_limit_then_breaches(self):
        mod = _streaming()
        assert mod is not None
        # three allowed, the fourth in the same window is a breach
        assert mod.check_rate("phone-A", now=1000.0, limit=3, window=60.0) is True
        assert mod.check_rate("phone-A", now=1000.0, limit=3, window=60.0) is True
        assert mod.check_rate("phone-A", now=1000.0, limit=3, window=60.0) is True
        assert mod.check_rate("phone-A", now=1000.0, limit=3, window=60.0) is False

    def test_window_resets(self):
        mod = _streaming()
        assert mod is not None
        for _ in range(3):
            mod.check_rate("phone-A", now=1000.0, limit=3, window=60.0)
        assert mod.check_rate("phone-A", now=1000.0, limit=3, window=60.0) is False
        # a new window: allowed again
        assert mod.check_rate("phone-A", now=1061.0, limit=3, window=60.0) is True

    def test_per_device_isolation(self):
        mod = _streaming()
        assert mod is not None
        for _ in range(3):
            mod.check_rate("phone-A", now=1000.0, limit=3, window=60.0)
        assert mod.check_rate("phone-A", now=1000.0, limit=3, window=60.0) is False
        # a different device has its own bucket
        assert mod.check_rate("phone-B", now=1000.0, limit=3, window=60.0) is True

    def test_breach_records_alert_in_telemetry(self):
        mod = _streaming()
        assert mod is not None
        for _ in range(3):
            mod.check_rate("phone-A", now=1000.0, limit=3, window=60.0)
        mod.check_rate("phone-A", now=1000.0, limit=3, window=60.0)  # breach
        tel = mod.telemetry("phone-A")
        assert int(tel.get("alerts", 0)) >= 1


# ---------------------------------------------------------------------------
# Family 2 -- the handler wires the rate gate (refused before the funnel)
# ---------------------------------------------------------------------------


class TestHandlerRateWiring:
    def test_rate_breach_refuses_before_funnel(self, monkeypatch):
        ri = _remote_inference()
        stream = _streaming()
        assert ri is not None and stream is not None
        # force the gate to report a breach
        monkeypatch.setattr(stream, "check_rate", lambda *a, **k: False)
        store = FakePeerStore({"phone-A": FakeRecord()})
        out = ri.serve_remote_inference(
            {"v": protocol.PROTOCOL_VERSION, "type": "remote_infer",
             "device": "phone-A", "request_id": "req-1", "prompt": "hi"},
            peer_id="phone-A",
            peer_store=store,
            router=_fake_router,
            audit=_AuditSpy(),
            executor=FakeExecutor(boom=True),  # the funnel must NOT be entered
        )
        assert out.get("ok") is False
        assert out.get("reason") == "rate_limited"

    def test_rate_refusal_is_audit_chained(self, monkeypatch):
        ri = _remote_inference()
        stream = _streaming()
        assert ri is not None and stream is not None
        monkeypatch.setattr(stream, "check_rate", lambda *a, **k: False)
        spy = _AuditSpy()
        ri.serve_remote_inference(
            {"v": protocol.PROTOCOL_VERSION, "type": "remote_infer",
             "device": "phone-A", "request_id": "req-1", "prompt": "hi"},
            peer_id="phone-A",
            peer_store=FakePeerStore({"phone-A": FakeRecord()}),
            router=_fake_router,
            audit=spy,
            executor=FakeExecutor(boom=True),
        )
        reasons = [d.get("reason") for a, d in spy.events if a == "remote_infer_refused"]
        assert "rate_limited" in reasons


# ---------------------------------------------------------------------------
# Family 3 -- the control surface payload helpers
# ---------------------------------------------------------------------------


class TestControlSurfacePayloads:
    def test_grant_view_payload(self):
        mod = _routes_sync()
        assert mod is not None and hasattr(mod, "remote_chat_grant_payload")
        store = FakePeerStore({"phone-A": FakeRecord(remote_chat_enabled=True, rag_subgrant=False)})
        out = mod.remote_chat_grant_payload(store, "phone-A")
        assert out["peer_id"] == "phone-A"
        assert out["remote_chat_enabled"] is True
        assert out["rag_subgrant"] is False

    def test_grant_view_unknown_raises_peer_not_found(self):
        mod = _routes_sync()
        assert mod is not None
        store = FakePeerStore({})
        with pytest.raises(Exception):
            mod.remote_chat_grant_payload(store, "ghost")

    def test_set_grant_payload_enables_and_subgrants(self):
        mod = _routes_sync()
        assert mod is not None and hasattr(mod, "set_remote_chat_grant_payload")
        store = FakePeerStore({"phone-A": FakeRecord(remote_chat_enabled=True, rag_subgrant=False)})
        out = mod.set_remote_chat_grant_payload(store, "phone-A", enabled=False, rag=True)
        assert out["remote_chat_enabled"] is False
        assert out["rag_subgrant"] is True

    def test_revoke_payload_kills_live_sessions(self):
        mod = _routes_sync()
        stream = _streaming()
        assert mod is not None and stream is not None
        assert hasattr(mod, "revoke_remote_chat_payload")
        store = FakePeerStore({"phone-A": FakeRecord(remote_chat_enabled=True)})
        stream.open_session("phone-A", "req-1", ["a", "b"])
        out = mod.revoke_remote_chat_payload(store, "phone-A")
        assert out["revoked"] is True
        assert store.get_peer("phone-A").remote_chat_enabled is False
        assert stream.active_session_count() == 0

    def test_telemetry_payload_shape(self):
        mod = _routes_sync()
        assert mod is not None and hasattr(mod, "remote_chat_telemetry_payload")
        out = mod.remote_chat_telemetry_payload()
        assert isinstance(out, dict)
        assert "active_sessions" in out


# ---------------------------------------------------------------------------
# Family 4 -- SYN-06 auth parity and the routes are declared
# ---------------------------------------------------------------------------


class TestRoutesAndAuthParity:
    def test_router_carries_router_level_auth(self):
        src = (_REPO / "opti_oignon/api/routes_sync.py").read_text(encoding="utf-8")
        assert "dependencies=_auth_dep" in src
        assert "_auth_dep = [Depends(_get_current_user)]" in src

    def test_four_control_routes_declared(self):
        src = (_REPO / "opti_oignon/api/routes_sync.py").read_text(encoding="utf-8")
        assert '/peers/{peer_id}/remote-chat"' in src
        assert '/peers/{peer_id}/remote-chat/revoke"' in src
        assert '/remote-chat/telemetry"' in src


# ---------------------------------------------------------------------------
# Family 5 -- the SyncPanel-family panel and the API client
# ---------------------------------------------------------------------------


def _balanced(src: str, open_tag: str, close_tag: str) -> bool:
    return src.count(open_tag) == src.count(close_tag)


class TestFrontendPanel:
    def test_remote_channel_panel_exists(self):
        p = _REPO / "frontend/src/lib/components/panels/RemoteChannelPanel.svelte"
        assert p.exists(), "RemoteChannelPanel.svelte must exist"

    def test_remote_channel_panel_registered_in_spec(self):
        spec = (_REPO / "FRONTEND_REDESIGN_SPEC.md").read_text(encoding="utf-8")
        assert "RemoteChannelPanel.svelte" in spec

    def test_syncpanel_mounts_remote_channel_panel(self):
        src = (_REPO / "frontend/src/lib/components/panels/SyncPanel.svelte").read_text(encoding="utf-8")
        assert "RemoteChannelPanel" in src

    def test_sync_api_exports_remote_chat_functions(self):
        src = (_REPO / "frontend/src/lib/api/sync.ts").read_text(encoding="utf-8")
        assert "getRemoteChatGrant" in src
        assert "setRemoteChatGrant" in src
        assert "revokeRemoteChat" in src
        assert "getRemoteChatTelemetry" in src

    def test_remote_channel_panel_uses_the_api(self):
        src = (_REPO / "frontend/src/lib/components/panels/RemoteChannelPanel.svelte").read_text(encoding="utf-8")
        assert "getRemoteChatGrant" in src or "getRemoteChatTelemetry" in src

    def test_svelte_tags_balance(self):
        for rel in (
            "frontend/src/lib/components/panels/RemoteChannelPanel.svelte",
            "frontend/src/lib/components/panels/SyncPanel.svelte",
        ):
            src = (_REPO / rel).read_text(encoding="utf-8")
            assert _balanced(src, "<script", "</script>"), rel
            assert _balanced(src, "<style", "</style>"), rel
            # no obviously unbalanced block markers
            assert src.count("{#if") == src.count("{/if"), rel
            assert src.count("{#each") == src.count("{/each"), rel
