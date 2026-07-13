#!/usr/bin/env python3
"""Sync pairing and trust-state contracts: integrity first, honest ceremony.

Pairing is the ceremony that populates the peer registry, and every trust
flip on that registry is a human action with an honest answer: the
device's own payload only exists when the live transport can mint a
routing key, its canonical material is pinned so the confirmation code
recomputes from local disk, a peer payload is integrity-checked before
anything is stored, confirmation and rejection stay asymmetric so a
trusted device is never fat-finger removed, and the device-class flip is
allowlisted at the wire before the audited setter runs. This suite pins
that behavior:

  * PC1 -- no routing key means a 503 with no payload built and no audit;
  * PC2 -- a built payload pins the canonical self material and audits
    the generation, and the QR text is exactly the encoded payload;
  * PC3 -- a pin failure degrades to a payload without a later code,
    never a 500, and the generation is still audited;
  * PC4 -- an invalid or tampered payload is a 400 after the defensive
    parse, nothing registered;
  * PC5 -- the accept envelope threads the label and the resolved store,
    a pending registration carries the confirmation code, and a bare
    body is accepted as the payload itself;
  * PC6 -- confirming an unknown peer is a 404 before the engine; a
    known pending peer flows the engine and returns the fresh record;
  * PC7 -- rejecting a confirmed peer is a 409 that points at unpair,
    only a pending peer is removed, and the rejection is audited;
  * PC8 -- the device-class wire gate refuses free text with a 400
    before any write, normalises case and whitespace into the audited
    setter, and maps a vanished row to a 404.

Loads the sync REST facade in isolation under a stand-in package; every
``opti_oignon.*`` entry plus the web-framework and model-client entries
is snapshotted and evicted first, and the seeds are deterministic
recorders: a minimal framework stand-in whose refusal type carries the
status code, a veilid sub-package of configurable recording stubs
(guard, pairing, peers, engine, status, node, transport, streaming), an
emergency-stop stub, and a hash-chain audit recorder. A meta-path guard
refuses any project submodule that was not seeded, so the load behaves
identically whether or not the project is installed. Local-only. Runs
under pytest or the __main__ runner.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the
    test's back -- silently importing live code. This guard sits ahead of
    every finder and refuses the names that were not seeded, so a load
    behaves identically whether the project is installed or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


class _HTTPRefusal(Exception):
    """Framework stand-in refusal carrying the status code and detail."""

    def __init__(self, status_code, detail=""):
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class _Router:
    """Framework stand-in router journaling the registration order."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.registered = []

    def _decorate(self, method, path):
        def deco(fn):
            self.registered.append((method, path))
            return fn
        return deco

    def get(self, path, **kwargs):
        return self._decorate("GET", path)

    def post(self, path, **kwargs):
        return self._decorate("POST", path)

    def delete(self, path, **kwargs):
        return self._decorate("DELETE", path)


def _peer(peer_id="peer-a", **over):
    rec = SimpleNamespace(
        peer_id=peer_id,
        routing_key="rk-" + peer_id,
        label="",
        watermark=0,
        added_at="t0",
        updated_at="t1",
        pending=False,
        key_changed=False,
        device_class=None,
        remote_chat_enabled=True,
        rag_subgrant=False,
    )
    for key, value in over.items():
        setattr(rec, key, value)
    return rec


def _round_result(**over):
    ns = SimpleNamespace(
        peer_id="peer-a",
        applied=1,
        deferred=0,
        conflicts=0,
        rejected=0,
        refused=0,
        unverified=0,
        previous_watermark=0,
        new_watermark=1,
        advanced=True,
        parsed=True,
    )
    for key, value in over.items():
        setattr(ns, key, value)
    return ns


class _Store:
    """Recording peer-store stand-in backed by a plain dict."""

    def __init__(self, cfg):
        self._cfg = cfg
        self.peers = {}
        self.grant_calls = []
        self.rag_calls = []
        self.pin_calls = []

    def add(self, rec):
        self.peers[rec.peer_id] = rec
        return rec

    def list_peers(self):
        return list(self.peers.values())

    def get_peer(self, peer_id):
        return self.peers.get(peer_id)

    def has_peer(self, peer_id):
        return peer_id in self.peers

    def set_remote_chat_grant(self, peer_id, flag):
        self.grant_calls.append((peer_id, bool(flag)))
        rec = self.peers.get(peer_id)
        if rec is not None:
            rec.remote_chat_enabled = bool(flag)

    def set_rag_subgrant(self, peer_id, flag):
        self.rag_calls.append((peer_id, bool(flag)))
        rec = self.peers.get(peer_id)
        if rec is not None:
            rec.rag_subgrant = bool(flag)

    def get_self_pairing_material(self):
        return self._cfg.self_material

    def pin_self_pairing_material(self, material):
        if self._cfg.pin_error is not None:
            raise self._cfg.pin_error
        self.pin_calls.append(material)


class _Engine:
    """Recording sync-engine stand-in configured through the window cfg."""

    def __init__(self, cfg):
        self._cfg = cfg
        self.device = "desk-self"
        self.run_calls = []
        self.confirm_calls = []
        self.unregister_calls = []
        self.register_calls = []
        self.setdc_calls = []
        self.approve_calls = []
        self.refuse_calls = []
        self.republish_calls = []

    def self_signing_pub(self):
        return self._cfg.signing_pub

    def run_round(self, peer_id, peer, approval_fn=None,
                  conversation_id="", approval_manager=None):
        self.run_calls.append((peer_id, peer, conversation_id))
        if self._cfg.run_error is not None:
            raise self._cfg.run_error
        return self._cfg.run_result

    def confirm_peer(self, peer_id):
        self.confirm_calls.append(peer_id)
        rec = self._cfg.store.peers.get(peer_id)
        if rec is not None:
            rec.pending = False

    def unregister_peer(self, peer_id):
        self.unregister_calls.append(peer_id)
        present = peer_id in self._cfg.store.peers
        if not (present and self._cfg.unregister_result):
            return False
        self._cfg.store.peers.pop(peer_id, None)
        return True

    def register_peer(self, peer_id, routing_key, label=""):
        self.register_calls.append((peer_id, routing_key, label))
        rec = self._cfg.store.peers.get(peer_id)
        if rec is None:
            rec = self._cfg.store.add(_peer(peer_id, routing_key=routing_key))
        rec.label = label
        return rec

    def set_device_class(self, peer_id, value):
        self.setdc_calls.append((peer_id, value))
        if not self._cfg.setdc_result:
            return False
        rec = self._cfg.store.peers.get(peer_id)
        if rec is not None:
            rec.device_class = value
        return True

    def list_deferred(self):
        return list(self._cfg.deferred_entries)

    def approve_deferred(self, kind, record_id):
        self.approve_calls.append((kind, record_id))
        if self._cfg.approve_error is not None:
            raise self._cfg.approve_error
        return dict(self._cfg.approve_result)

    def refuse_deferred(self, kind, record_id):
        self.refuse_calls.append((kind, record_id))
        if self._cfg.refuse_error is not None:
            raise self._cfg.refuse_error
        return self._cfg.refuse_entry

    def republish_signed(self):
        self.republish_calls.append(1)
        if self._cfg.republish_error is not None:
            raise self._cfg.republish_error
        return self._cfg.republish_count


class _Status:
    """Recording status-store stand-in."""

    def __init__(self):
        self.failures = []
        self.rounds = []

    def record_failure(self, peer_id, reason):
        self.failures.append((peer_id, reason))

    def record_round(self, payload):
        self.rounds.append(dict(payload))

    def last_for(self, peer_id):
        return None

    def last_round(self):
        return None


def _load(seed_veilid=True):
    """Load the sync REST facade under a stand-in package."""
    lateral = [
        k for k in list(sys.modules)
        if k == "fastapi" or k.startswith("fastapi.")
    ]
    keys = ["ollama"] + lateral + [
        k
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    ]
    saved = {k: sys.modules[k] for k in keys if k in sys.modules}
    for k in keys:
        sys.modules.pop(k, None)
    sys.modules["ollama"] = None  # no client import exists; drift fails loud

    framework = types.ModuleType("fastapi")
    framework.APIRouter = _Router
    framework.HTTPException = _HTTPRefusal
    framework.Depends = lambda fn=None: ("dependency", fn)
    framework.Body = lambda default=None, **_kw: default
    sys.modules["fastapi"] = framework

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    sys.modules["opti_oignon"] = root

    api = types.ModuleType("opti_oignon.api")
    api.__path__ = []
    sys.modules["opti_oignon.api"] = api
    root.api = api

    cfg = SimpleNamespace(
        bulbe=False,
        guard_calls=[],
        stopped=False,
        stop_calls=[],
        audit=[],
        store=None,
        engine=None,
        status=_Status(),
        node=None,
        store_resolutions=[],
        engine_resolutions=[],
        self_material=None,
        pin_error=None,
        signing_pub="SPUB",
        run_error=None,
        run_result=None,
        unregister_result=True,
        setdc_result=True,
        deferred_entries=[],
        approve_error=None,
        approve_result={"approved": True, "refused": False},
        refuse_error=None,
        refuse_entry=None,
        republish_error=None,
        republish_count=0,
        kill_calls=[],
        kill_result=0,
        kill_error=None,
        telemetry={"devices": {}, "active_sessions": 0},
        build_calls=[],
        accept_calls=[],
        accept_result=None,
        exc=None,
    )
    cfg.store = _Store(cfg)
    cfg.engine = _Engine(cfg)

    stop = types.ModuleType("opti_oignon.emergency_stop")

    def _guard_http():
        cfg.stop_calls.append(1)
        if cfg.stopped:
            raise _HTTPRefusal(503, "system stopped")

    stop.guard_http = _guard_http
    sys.modules["opti_oignon.emergency_stop"] = stop
    root.emergency_stop = stop

    audit_mod = types.ModuleType("opti_oignon.signed_audit_log")
    audit_mod.chain_log = lambda **kw: cfg.audit.append(dict(kw))
    sys.modules["opti_oignon.signed_audit_log"] = audit_mod
    root.signed_audit_log = audit_mod

    if seed_veilid:
        ve = types.ModuleType("opti_oignon.veilid")
        ve.__path__ = []
        sys.modules["opti_oignon.veilid"] = ve
        root.veilid = ve

        guard_mod = types.ModuleType("opti_oignon.veilid.guard")
        for name in ("VeilidDisabledInBulbe", "VeilidTimeout",
                     "VeilidUnavailable"):
            setattr(guard_mod, name, type(name, (Exception,), {}))

        def _bulbe_disabled():
            cfg.guard_calls.append(1)
            return cfg.bulbe

        guard_mod.bulbe_disabled = _bulbe_disabled
        sys.modules["opti_oignon.veilid.guard"] = guard_mod
        ve.guard = guard_mod

        pairing = types.ModuleType("opti_oignon.veilid.pairing")

        def _build(peer_id, routing_key, signing_pub=None, device_class=None):
            cfg.build_calls.append(
                (peer_id, routing_key, signing_pub, device_class)
            )
            return {
                "peer_id": peer_id,
                "routing_key": routing_key,
                "signing_pub": signing_pub,
                "device_class": device_class,
                "check": "CHK",
            }

        pairing.build_pairing_payload = _build
        pairing.encode_pairing_json = lambda p: json.dumps(p, sort_keys=True)
        pairing.pairing_canonical_material = (
            lambda peer_id, routing_key, signing_pub=None:
            ("MAT", peer_id, routing_key, signing_pub)
        )
        pairing.confirmation_code = lambda a, b: "1111-2222"

        def _accept(engine, obj, label="", store=None):
            cfg.accept_calls.append((obj, label, store))
            return cfg.accept_result

        pairing.accept_pairing_payload = _accept
        sys.modules["opti_oignon.veilid.pairing"] = pairing
        ve.pairing = pairing

        streaming = types.ModuleType("opti_oignon.veilid.remote_streaming")

        def _kill(peer_id):
            if cfg.kill_error is not None:
                raise cfg.kill_error
            cfg.kill_calls.append(peer_id)
            return cfg.kill_result

        streaming.kill_sessions_for_device = _kill
        streaming.telemetry = lambda: dict(cfg.telemetry)
        sys.modules["opti_oignon.veilid.remote_streaming"] = streaming
        ve.remote_streaming = streaming

        transport = types.ModuleType("opti_oignon.veilid.transport")

        def _unconfigured(*_args, **_kwargs):
            raise AssertionError(
                "transport stub reached; tests inject resolvers"
            )

        transport.resolve_live_peer = _unconfigured
        transport.resolve_self_routing_key = _unconfigured
        sys.modules["opti_oignon.veilid.transport"] = transport
        ve.transport = transport

        node_mod = types.ModuleType("opti_oignon.veilid.node")
        node_mod.get_node = lambda: cfg.node
        sys.modules["opti_oignon.veilid.node"] = node_mod
        ve.node = node_mod

        peers_mod = types.ModuleType("opti_oignon.veilid.peers")
        peers_mod.DEVICE_CLASS_DESKTOP = "desktop"
        peers_mod.PeerStore = type("PeerStore", (object,), {})

        def _get_store():
            cfg.store_resolutions.append(1)
            return cfg.store

        peers_mod.get_peer_store = _get_store
        sys.modules["opti_oignon.veilid.peers"] = peers_mod
        ve.peers = peers_mod

        engine_mod = types.ModuleType("opti_oignon.veilid.sync_engine")
        for name in ("PeerNotFound", "PeerNotConfirmed", "DeferredNotFound"):
            setattr(engine_mod, name, type(name, (Exception,), {}))
        engine_mod.SyncEngine = type("SyncEngine", (object,), {})

        def _get_engine():
            cfg.engine_resolutions.append(1)
            return cfg.engine

        engine_mod.get_sync_engine = _get_engine
        sys.modules["opti_oignon.veilid.sync_engine"] = engine_mod
        ve.sync_engine = engine_mod

        status_mod = types.ModuleType("opti_oignon.veilid.sync_status")
        status_mod.get_sync_status_store = lambda: cfg.status
        sys.modules["opti_oignon.veilid.sync_status"] = status_mod
        ve.sync_status = status_mod

        cfg.exc = SimpleNamespace(
            bulbe=guard_mod.VeilidDisabledInBulbe,
            timeout=guard_mod.VeilidTimeout,
            unavailable=guard_mod.VeilidUnavailable,
            peer_missing=engine_mod.PeerNotFound,
            not_confirmed=engine_mod.PeerNotConfirmed,
            deferred_missing=engine_mod.DeferredNotFound,
        )

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    def restore():
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        for k in list(sys.modules):
            if k == "opti_oignon" or k.startswith("opti_oignon."):
                del sys.modules[k]
            elif k == "fastapi" or k.startswith("fastapi."):
                del sys.modules[k]
        sys.modules.pop("ollama", None)
        for k, v in saved.items():
            sys.modules[k] = v

    full = "opti_oignon.api.routes_sync"
    spec = importlib.util.spec_from_file_location(
        full, _OO / "api" / "routes_sync.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    api.routes_sync = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        restore()
        raise

    return SimpleNamespace(mod=mod, cfg=cfg, restore=restore)


def _expect_refusal(status, fn, *args, **kwargs):
    """Call the route and assert the stand-in refusal with the given code."""
    try:
        fn(*args, **kwargs)
    except _HTTPRefusal as exc:
        assert exc.status_code == status, (
            f"expected {status}, got {exc.status_code}: {exc.detail}"
        )
        return exc
    raise AssertionError(f"expected a {status} refusal, nothing was raised")


def test_self_payload_refuses_503_without_a_routing_key():
    """PC1: no mintable routing key is a 503; nothing built, nothing audited."""
    env = _load()
    try:
        env.mod.set_self_routing_resolver(lambda engine: None)
        exc = _expect_refusal(503, env.mod.sync_pairing_self)
        assert "Routing key" in exc.detail
        assert env.cfg.build_calls == []
        assert env.cfg.audit == []
        env.mod.reset_self_routing_resolver()
    finally:
        env.restore()


def test_self_payload_pins_material_and_audits():
    """PC2: the payload pins canonical self material and audits itself."""
    env = _load()
    try:
        cfg = env.cfg
        env.mod.set_self_routing_resolver(lambda engine: "RKEY")
        out = env.mod.sync_pairing_self()
        assert out["peer_id"] == "desk-self"
        assert out["routing_key"] == "RKEY"
        assert out["payload"] == {
            "peer_id": "desk-self",
            "routing_key": "RKEY",
            "signing_pub": "SPUB",
            "device_class": "desktop",
            "check": "CHK",
        }
        assert out["text"] == json.dumps(out["payload"], sort_keys=True)
        assert cfg.build_calls == [("desk-self", "RKEY", "SPUB", "desktop")]
        assert cfg.store.pin_calls == [("MAT", "desk-self", "RKEY", "SPUB")]
        assert cfg.audit and cfg.audit[-1]["action"] == "pairing_self"
        env.mod.reset_self_routing_resolver()
    finally:
        env.restore()


def test_self_pin_failure_degrades_without_a_500():
    """PC3: a pin failure still returns the payload and still audits."""
    env = _load()
    try:
        cfg = env.cfg
        cfg.pin_error = RuntimeError("meta row unavailable")
        env.mod.set_self_routing_resolver(lambda engine: "RKEY")
        out = env.mod.sync_pairing_self()
        assert out["payload"]["routing_key"] == "RKEY"
        assert cfg.store.pin_calls == []
        assert cfg.audit and cfg.audit[-1]["action"] == "pairing_self"
        env.mod.reset_self_routing_resolver()
    finally:
        env.restore()


def test_invalid_pairing_payload_is_a_400():
    """PC4: a payload the parser rejects is a 400; nothing registered."""
    env = _load()
    try:
        cfg = env.cfg
        cfg.accept_result = None
        exc = _expect_refusal(
            400, env.mod.sync_pairing_accept, {"routing_key": "garbled"}
        )
        assert "Invalid" in exc.detail
        assert len(cfg.accept_calls) == 1
    finally:
        env.restore()


def test_accept_envelope_threads_label_store_and_code():
    """PC5: the envelope threads label and store; a bare body is the payload."""
    env = _load()
    try:
        cfg = env.cfg
        cfg.accept_result = _peer("peer-new", pending=True)
        cfg.self_material = ("MAT", "desk-self", "RK0", "SPUB")
        out = env.mod.sync_pairing_accept(
            {"payload": {"marker": 1}, "label": "Phone of A"}
        )
        assert cfg.accept_calls[-1] == ({"marker": 1}, "Phone of A", cfg.store)
        assert out["pending"] is True
        assert out["confirmation_code"] == "1111-2222"

        env.mod.sync_pairing_accept({"peer_id": "raw"})
        assert cfg.accept_calls[-1] == ({"peer_id": "raw"}, "", cfg.store)
    finally:
        env.restore()


def test_confirm_gates_unknown_then_flows_the_engine():
    """PC6: confirming an unknown peer is a 404 before the engine runs."""
    env = _load()
    try:
        cfg = env.cfg
        _expect_refusal(404, env.mod.sync_pairing_confirm, "ghost")
        assert cfg.engine.confirm_calls == []
        cfg.store.add(_peer(pending=True))
        out = env.mod.sync_pairing_confirm("peer-a")
        assert cfg.engine.confirm_calls == ["peer-a"]
        assert out["peer_id"] == "peer-a"
        assert out["pending"] is False
    finally:
        env.restore()


def test_reject_is_pending_only_and_audited():
    """PC7: rejection removes only a pending peer and is audited."""
    env = _load()
    try:
        cfg = env.cfg
        cfg.store.add(_peer("done", pending=False))
        exc = _expect_refusal(409, env.mod.sync_pairing_reject, "done")
        assert "unpair" in exc.detail
        assert cfg.engine.unregister_calls == []

        cfg.store.add(_peer("pend", pending=True))
        out = env.mod.sync_pairing_reject("pend")
        assert out == {"peer_id": "pend", "rejected": True}
        assert cfg.engine.unregister_calls == ["pend"]
        assert "pend" not in cfg.store.peers
        assert cfg.audit and cfg.audit[-1]["action"] == "pairing_reject"
    finally:
        env.restore()


def test_device_class_wire_gate_is_strict():
    """PC8: free text is a 400 before any write; the setter is normalised."""
    env = _load()
    try:
        cfg = env.cfg
        cfg.store.add(_peer())
        exc = _expect_refusal(
            400, env.mod.sync_set_device_class, "peer-a",
            {"device_class": "tablet"},
        )
        assert "device_class" in exc.detail
        assert cfg.engine.setdc_calls == []

        out = env.mod.sync_set_device_class(
            "peer-a", {"device_class": "  Phone "}
        )
        assert cfg.engine.setdc_calls == [("peer-a", "phone")]
        assert out["device_class"] == "phone"

        cfg.store.add(_peer("gone"))
        cfg.setdc_result = False
        _expect_refusal(
            404, env.mod.sync_set_device_class, "gone",
            {"device_class": "desktop"},
        )
    finally:
        env.restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
