#!/usr/bin/env python3
"""Re-authentication gates for the served remote-inference handlers.

Before a paired phone may borrow the desktop's local model -- or pull the next
chunk of an already-streamed reply -- the responder re-authenticates the
route peer against the peer store on every entry point (REMOTE_INFERENCE_SPEC).
Three trust gates stand in front of both the initial handler and the
continuation handler, in the same order:

  * ``unknown_device`` -- the asking device is not a registered peer;
  * ``peer_not_confirmed`` -- the pairing still awaits mutual confirmation, so
    it grants nothing, serving included;
  * ``remote_chat_disabled`` -- the device's remote-chat grant is off (revoked).

This suite pins all three on each handler. The gates are not labels: each refusal
is shown to be a real serve-versus-refuse decision. On the continuation side the
buffer is opened for real (the stdlib streaming module is loaded by spec, not
stubbed), the device is then made unregistered / pending / revoked, and the
continuation is sent: the gate refuses BEFORE the buffer is read. Re-arming the
peer, the owner then pulls the very chunk the refusal withheld -- proof the data
was live and the refusal never consumed it. On the initial side the gate refuses
before any generation runs; re-arming the peer, the same request is served.

The handler is loaded in isolation with stubbed guard / protocol and the real
streaming buffer; the peer store, executor, and router are injected fakes.

Local-only. Runs under pytest or the ``__main__`` runner.
"""

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load():
    keys = ("opti_oignon", "opti_oignon.veilid", "opti_oignon.veilid.guard",
            "opti_oignon.veilid.protocol", "opti_oignon.veilid.remote_streaming",
            "opti_oignon.veilid.remote_inference")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    vpkg = types.ModuleType("opti_oignon.veilid")
    vpkg.__path__ = []
    sys.modules["opti_oignon.veilid"] = vpkg

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

    for name, mod in (("guard", guard), ("protocol", proto)):
        sys.modules[f"opti_oignon.veilid.{name}"] = mod
        setattr(vpkg, name, mod)

    # the real streaming buffer (stdlib-only) so the continuation gates run in
    # front of a genuine live buffer, not a stub
    rs_spec = importlib.util.spec_from_file_location(
        "opti_oignon.veilid.remote_streaming",
        _OO / "veilid" / "remote_streaming.py")
    rs = importlib.util.module_from_spec(rs_spec)
    sys.modules["opti_oignon.veilid.remote_streaming"] = rs
    setattr(vpkg, "remote_streaming", rs)
    rs_spec.loader.exec_module(rs)

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


def _cont(**over):
    base = {"v": 1, "type": "remote_infer_cont", "device": "phoneA",
            "request_id": "r1", "cursor": 1}
    base.update(over)
    return base


# --- continuation handler: the three gates in front of a live buffer ----------

def test_continuation_unknown_device_is_refused_before_the_live_buffer():
    """An unregistered device cannot read an in-flight stream it does not own."""
    mod, _guard, rs, restore = _load()
    try:
        rs.reset_for_tests()
        rs.open_session("phoneA", "r1", ["a", "b", "c"])  # the buffer is live
        out = mod.serve_remote_inference_continuation(
            _cont(cursor=1), peer_id="phoneA", peer_store=_Store({}))
        assert out.get("refused") is True
        assert out["reason"] == "unknown_device"
        assert out.get("ok") is not True
        # the buffer was NOT read: once the device is registered, the owner
        # pulls the very chunk the refusal withheld
        out2 = mod.serve_remote_inference_continuation(
            _cont(cursor=1), peer_id="phoneA", peer_store=_Store({"phoneA": _Rec()}))
        assert out2.get("ok") is True
        assert out2["content"] == "b"
    finally:
        restore()


def test_continuation_unconfirmed_peer_is_refused_before_the_live_buffer():
    """A pending (PAIR-02-unconfirmed) peer is refused before the buffer read."""
    mod, _guard, rs, restore = _load()
    try:
        rs.reset_for_tests()
        rs.open_session("phoneA", "r1", ["a", "b", "c"])
        out = mod.serve_remote_inference_continuation(
            _cont(cursor=1), peer_id="phoneA",
            peer_store=_Store({"phoneA": _Rec(pending=True)}))
        assert out.get("refused") is True
        assert out["reason"] == "peer_not_confirmed"
        assert out.get("ok") is not True
        out2 = mod.serve_remote_inference_continuation(
            _cont(cursor=1), peer_id="phoneA",
            peer_store=_Store({"phoneA": _Rec(pending=False)}))
        assert out2.get("ok") is True
        assert out2["content"] == "b"
    finally:
        restore()


def test_continuation_revoked_grant_is_refused_before_the_live_buffer():
    """A grant revoked mid-stream is refused even while the buffer is still live.

    Defence in depth: the durable grant flip and the live-buffer drop are
    separate halves; this gate refuses on the grant alone, before any pull, so a
    revoked device cannot drain a stream that has not yet been torn down.
    """
    mod, _guard, rs, restore = _load()
    try:
        rs.reset_for_tests()
        rs.open_session("phoneA", "r1", ["a", "b", "c"])
        out = mod.serve_remote_inference_continuation(
            _cont(cursor=1), peer_id="phoneA",
            peer_store=_Store({"phoneA": _Rec(enabled=False)}))
        assert out.get("refused") is True
        assert out["reason"] == "remote_chat_disabled"
        assert out.get("ok") is not True
        out2 = mod.serve_remote_inference_continuation(
            _cont(cursor=1), peer_id="phoneA",
            peer_store=_Store({"phoneA": _Rec(enabled=True)}))
        assert out2.get("ok") is True
        assert out2["content"] == "b"
    finally:
        restore()


# --- initial handler: the same three gates in front of generation -------------

def test_initial_unknown_device_is_refused_before_generation():
    """An unregistered device is refused before any chat request is built."""
    mod, _guard, rs, restore = _load()
    try:
        rs.reset_for_tests()
        out = mod.serve_remote_inference(
            _req(), peer_id="phoneA", executor=_Exec(), router=_router,
            peer_store=_Store({}))
        assert out.get("refused") is True
        assert out["reason"] == "unknown_device"
        assert out.get("ok") is not True
        # once registered, the same request is served
        out2 = mod.serve_remote_inference(
            _req(), peer_id="phoneA", executor=_Exec(), router=_router,
            peer_store=_Store({"phoneA": _Rec()}))
        assert out2.get("ok") is True
        assert out2["content"] == "hello from desktop"
    finally:
        restore()


def test_initial_unconfirmed_peer_is_refused_before_generation():
    """A pending (PAIR-02-unconfirmed) peer grants nothing, serving included."""
    mod, _guard, rs, restore = _load()
    try:
        rs.reset_for_tests()
        out = mod.serve_remote_inference(
            _req(), peer_id="phoneA", executor=_Exec(), router=_router,
            peer_store=_Store({"phoneA": _Rec(pending=True)}))
        assert out.get("refused") is True
        assert out["reason"] == "peer_not_confirmed"
        assert out.get("ok") is not True
        out2 = mod.serve_remote_inference(
            _req(), peer_id="phoneA", executor=_Exec(), router=_router,
            peer_store=_Store({"phoneA": _Rec(pending=False)}))
        assert out2.get("ok") is True
        assert out2["content"] == "hello from desktop"
    finally:
        restore()


def test_initial_revoked_grant_is_refused_before_generation():
    """A device whose remote-chat grant is off is refused before generation."""
    mod, _guard, rs, restore = _load()
    try:
        rs.reset_for_tests()
        out = mod.serve_remote_inference(
            _req(), peer_id="phoneA", executor=_Exec(), router=_router,
            peer_store=_Store({"phoneA": _Rec(enabled=False)}))
        assert out.get("refused") is True
        assert out["reason"] == "remote_chat_disabled"
        assert out.get("ok") is not True
        out2 = mod.serve_remote_inference(
            _req(), peer_id="phoneA", executor=_Exec(), router=_router,
            peer_store=_Store({"phoneA": _Rec(enabled=True)}))
        assert out2.get("ok") is True
        assert out2["content"] == "hello from desktop"
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
