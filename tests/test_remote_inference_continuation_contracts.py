#!/usr/bin/env python3
"""Continuation-seam security contracts for remote inference.

After the desktop responder streams the first chunk of a borrowed-model reply,
the phone pulls the rest by request id (REMOTE_INFERENCE_SPEC). This suite pins
the load-bearing properties of that pull path -- the companion to the served
handler's gates:

  * a device can only read its OWN in-flight stream: a continuation for a
    request id buffered under a different device is refused (buffer_mismatch),
    never a cross-read -- the buffer is keyed on (device, request id) and the
    handler pulls with the route-authenticated peer, not a client field;
  * any field outside the continuation allow-set is refused, not dropped;
  * the network-isolation mode is re-asserted on this second entry point too:
    under it the handler refuses by propagating (it sends no reply).

The handler is loaded in isolation with the REAL streaming buffer (so the
cross-read key is exercised for real) and stubbed guard / protocol; the
injectable peer store is a fake. The streaming registry is reset before each
test.

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

    # the real streaming buffer (stdlib-only) so the cross-read key is genuine
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
    pending = False
    remote_chat_enabled = True
    rag_subgrant = False


class _Store:
    def __init__(self, names):
        self._m = {n: _Rec() for n in names}

    def get_peer(self, origin):
        return self._m.get(origin)


def _cont(**over):
    base = {"v": 1, "type": "remote_infer_cont", "device": "phoneA",
            "request_id": "r1", "cursor": 1}
    base.update(over)
    return base


def test_a_device_cannot_read_another_devices_stream():
    """A continuation for another device's buffered stream is a mismatch."""
    mod, _guard, rs, restore = _load()
    try:
        store = _Store(["phoneA", "phoneB"])
        rs.reset_for_tests()
        rs.open_session("phoneA", "r1", ["a", "b", "c"])
        # phoneB, route-authenticated as itself, asks for phoneA's request id
        out = mod.serve_remote_inference_continuation(
            _cont(device="phoneB"), peer_id="phoneB", peer_store=store)
        assert out.get("refused") is True
        assert out["reason"] == "buffer_mismatch"
        # the owner still reads its own next chunk (the buffer is intact)
        out = mod.serve_remote_inference_continuation(
            _cont(device="phoneA"), peer_id="phoneA", peer_store=store)
        assert out.get("ok") is True
        assert out["content"] == "b"
    finally:
        restore()


def test_continuation_out_of_surface_field_is_refused():
    """A field outside the continuation allow-set is refused, not dropped."""
    mod, _guard, rs, restore = _load()
    try:
        store = _Store(["phoneA"])
        rs.reset_for_tests()
        rs.open_session("phoneA", "r1", ["a", "b", "c"])
        out = mod.serve_remote_inference_continuation(
            _cont(device="phoneA", extra="x"), peer_id="phoneA", peer_store=store)
        assert out.get("refused") is True
        assert out["reason"] == "out_of_surface"
    finally:
        restore()


def test_continuation_re_asserts_network_isolation():
    """Under network isolation the continuation seam refuses by propagating."""
    mod, guard, rs, restore = _load()
    try:
        store = _Store(["phoneA"])
        rs.reset_for_tests()
        rs.open_session("phoneA", "r1", ["a", "b", "c"])
        guard.NETWORK_ISOLATED = True
        raised = False
        try:
            mod.serve_remote_inference_continuation(
                _cont(device="phoneA"), peer_id="phoneA", peer_store=store)
        except guard.VeilidDisabledInBulbe:
            raised = True
        assert raised, "isolation must propagate, never return a served reply"
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
