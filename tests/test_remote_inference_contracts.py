#!/usr/bin/env python3
"""Security contracts for the served remote-inference handler.

The desktop responder lets a paired, lower-trust device (the phone) borrow the
desktop's local models without the model, prompt, or response leaving the
user's two machines (REMOTE_INFERENCE_SPEC). This suite pins the load-bearing
refusals of that seam -- it is the first coverage of the handler:

  * the network-isolation mode is re-asserted at the seam: under it the handler
    refuses by PROPAGATING (it sends no reply), audit-chained -- defence in
    depth on top of the binding-layer gate;
  * any field outside the tier-1 allow-set is REFUSED, never silently dropped,
    so a state-mutation / sandbox / filesystem / shell / config / pipeline
    field can never reach the model;
  * a remote RAG-read scope is denied unless the asking device's separate
    read-only sub-grant is on (default-deny);
  * a device cannot impersonate another: a claimed origin that does not match
    the route-authenticated peer is refused.

The handler is loaded in isolation: the optional veilid framework
(``guard`` / ``protocol`` / ``remote_streaming``) is replaced by faithful
stubs, and the real injectable seams (executor, router, peer store, audit sink)
are passed as fakes -- so the path pulls no model backend and every gate is
exercised deterministically.

Local-only. Runs under pytest or the ``__main__`` runner.
"""

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _make_stubs():
    """Faithful stand-ins for the three optional veilid deps the handler imports."""
    guard = types.ModuleType("opti_oignon.veilid.guard")

    class VeilidDisabledInBulbe(Exception):
        pass

    guard.VeilidDisabledInBulbe = VeilidDisabledInBulbe
    guard.NETWORK_ISOLATED = False

    def assert_sync_allowed():
        if guard.NETWORK_ISOLATED:
            raise VeilidDisabledInBulbe("network isolated")

    guard.assert_sync_allowed = assert_sync_allowed

    proto = types.ModuleType("opti_oignon.veilid.protocol")
    proto.PROTOCOL_VERSION = 1
    proto.MSG_REMOTE_INFER = "remote_infer"
    proto.MSG_REMOTE_INFER_CONT = "remote_infer_cont"

    rs = types.ModuleType("opti_oignon.veilid.remote_streaming")
    rs.rate_ok = True
    rs.sessions = {}

    def check_rate(peer):
        return rs.rate_ok

    def open_session(origin, rid, chunks):
        rs.sessions[(origin, rid)] = list(chunks)

    def pull(origin, rid, cursor):
        ch = rs.sessions.get((origin, rid))
        if not ch or cursor >= len(ch):
            return None
        return {"content": ch[cursor], "cursor": cursor + 1,
                "done": cursor + 1 == len(ch)}

    rs.check_rate, rs.open_session, rs.pull = check_rate, open_session, pull
    return guard, proto, rs


def _load():
    keys = ("opti_oignon", "opti_oignon.veilid",
            "opti_oignon.veilid.guard", "opti_oignon.veilid.protocol",
            "opti_oignon.veilid.remote_streaming",
            "opti_oignon.veilid.remote_inference")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    vpkg = types.ModuleType("opti_oignon.veilid")
    vpkg.__path__ = []
    sys.modules["opti_oignon.veilid"] = vpkg

    guard, proto, rs = _make_stubs()
    for name, mod in (("guard", guard), ("protocol", proto),
                      ("remote_streaming", rs)):
        sys.modules[f"opti_oignon.veilid.{name}"] = mod
        setattr(vpkg, name, mod)

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.veilid.remote_inference",
        _OO / "veilid" / "remote_inference.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.veilid.remote_inference"] = mod
    spec.loader.exec_module(mod)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod, guard, rs, restore


class _Rec:
    """A peer record: confirmed and chat-enabled by default, RAG sub-grant off."""

    def __init__(self, pending=False, enabled=True, rag=False):
        self.pending = pending
        self.remote_chat_enabled = enabled
        self.rag_subgrant = rag


class _Store:
    def __init__(self, mapping):
        self._m = mapping

    def get_peer(self, origin):
        return self._m.get(origin)


class _Exec:
    def execute(self, **kw):
        return "hello from desktop"


def _router(_prompt):
    return object()


def _req(**over):
    base = {"v": 1, "type": "remote_infer", "device": "phoneA",
            "request_id": "r1", "prompt": "hi"}
    base.update(over)
    return base


def _audit_sink():
    events: list = []
    return events, (lambda action, **d: events.append((action, d)))


def test_network_isolation_is_re_asserted_and_propagates():
    """Under network isolation the seam refuses by propagating -- no reply."""
    mod, guard, _rs, restore = _load()
    try:
        store = _Store({"phoneA": _Rec()})
        guard.NETWORK_ISOLATED = True
        events, sink = _audit_sink()
        raised = False
        try:
            mod.serve_remote_inference(
                _req(), peer_id="phoneA", executor=_Exec(), router=_router,
                peer_store=store, audit=sink)
        except guard.VeilidDisabledInBulbe:
            raised = True
        assert raised, "isolation must propagate, never return a served reply"
        assert events and events[0][1].get("reason") == "bulbe"
    finally:
        restore()


def test_out_of_surface_fields_are_refused_not_dropped():
    """Any field outside the tier-1 allow-set is refused, never dropped."""
    mod, _guard, _rs, restore = _load()
    try:
        store = _Store({"phoneA": _Rec()})
        # an unrecognised field is refused (not silently ignored and served)
        out = mod.serve_remote_inference(
            _req(foo="bar"), peer_id="phoneA", executor=_Exec(),
            router=_router, peer_store=store)
        assert out.get("refused") is True
        assert out["reason"] == "out_of_surface"
        # a capability-class field (here: a tool handle) is likewise refused
        out = mod.serve_remote_inference(
            _req(tool="shell_exec"), peer_id="phoneA", executor=_Exec(),
            router=_router, peer_store=store)
        assert out.get("refused") is True
        assert out["reason"] == "out_of_surface"
    finally:
        restore()


def test_remote_rag_is_denied_without_subgrant():
    """A RAG-read scope is denied unless the device's sub-grant is on."""
    mod, _guard, _rs, restore = _load()
    try:
        # sub-grant off -> refused
        store_off = _Store({"phoneA": _Rec(rag=False)})
        out = mod.serve_remote_inference(
            _req(rag={"q": "x"}), peer_id="phoneA", executor=_Exec(),
            router=_router, peer_store=store_off)
        assert out["reason"] == "rag_not_granted"
        # sub-grant on -> the scope passes the surface gate (served)
        store_on = _Store({"phoneA": _Rec(rag=True)})
        out = mod.serve_remote_inference(
            _req(rag={"q": "x"}), peer_id="phoneA", executor=_Exec(),
            router=_router, peer_store=store_on)
        assert out.get("ok") is True
    finally:
        restore()


def test_a_device_cannot_impersonate_another():
    """A claimed origin that does not match the route peer is refused."""
    mod, _guard, _rs, restore = _load()
    try:
        store = _Store({"phoneA": _Rec()})
        # route-authenticated as phoneA, but the request claims to be phoneB
        out = mod.serve_remote_inference(
            _req(device="phoneB"), peer_id="phoneA", executor=_Exec(),
            router=_router, peer_store=store)
        assert out.get("refused") is True
        assert out["reason"] == "provenance_mismatch"
        # a matching claim is not a provenance refusal (it is served)
        out = mod.serve_remote_inference(
            _req(device="phoneA"), peer_id="phoneA", executor=_Exec(),
            router=_router, peer_store=store)
        assert out.get("ok") is True
    finally:
        restore()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'OK' if failures == 0 else 'FAILED'} - {failures} failure(s)")
    sys.exit(1 if failures else 0)
