#!/usr/bin/env python3
"""S264 -- the N.8 second lot: the note_update transport half.

NOTES_CRDT_SPEC.md section 3 made concrete (with the section-7 proof list):

1. The ``note_update`` record kind, a sibling of ``note`` on the S256 seam
   (honoured, never bypassed): the kind declaration, the producer, and the
   engine's ``publish_note_update`` convenience.
2. The journaling glue at the S263 store's append seam in the S257 idiom:
   best-effort, probe-first, covering every writer; the payload opaque;
   ``mobile_allowed`` and ``user_id`` never riding any payload (N9-D3); the
   remote-apply landing suppresses the glue (``sync_publish=False``) because
   the received record is already journalled verbatim -- re-publishing would
   re-sign it as ours, exactly what "never re-signs" forbids.
3. The apply leg: a received ``note_update`` lands through the store's append
   seam with the AUTHOR's explicit seq BEFORE the feed journal; the section-5
   refusal semantics apply verbatim (an update that cannot be attributed,
   gated, or persisted is refused: not appended, not served, not rendered,
   loggable -- never silent).
4. The S258 device-class serve gate extended as a FLOOR over the live
   ``is_mobile_allowed`` lookup for update records, fail-secure, and the
   checkpoint-watermark-forward serving rule (only ``seq > watermark`` is
   served toward a phone; the flip honours the S257 republish contract).
5. The relayed-signature property across a serve hop: an update authored on
   device A and served by device B carries A's signature end to end.

Red-before on the pristine S263 tree for every family except the declared
design-green reassertions (``TestDesignGreenReassertions``); every red is a
bare assertion (assert-before-call), never a collection error.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import importlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "opti_oignon"
VEILID = PKG / "veilid"

RECORDS_SRC = VEILID / "records.py"
PRODUCERS_SRC = VEILID / "producers.py"
ENGINE_SRC = VEILID / "sync_engine.py"
UPDATES_STORE_SRC = PKG / "notes" / "note_updates_store.py"


def _read(path: Path) -> str:
    """Raw text; empty string when absent so absence FAILS, never errors."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Isolation harness (the s252 / s256 idiom: real modules over light stubs)
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
            mod.__path__ = [str(sub)]  # type: ignore[attr-defined]
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
    sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]
    return {
        "signing": importlib.import_module("opti_oignon.veilid.signing"),
        "guard": importlib.import_module("opti_oignon.veilid.guard"),
        "change_feed": importlib.import_module(
            "opti_oignon.veilid.change_feed"
        ),
        "peers": importlib.import_module("opti_oignon.veilid.peers"),
        "records": importlib.import_module("opti_oignon.veilid.records"),
        "producers": importlib.import_module("opti_oignon.veilid.producers"),
        "protocol": importlib.import_module("opti_oignon.veilid.protocol"),
        "ledger": importlib.import_module(
            "opti_oignon.veilid.deferred_ledger"
        ),
        "engine": importlib.import_module("opti_oignon.veilid.sync_engine"),
    }


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


class FakeServingPeer:
    """A peer answering from its own feed through the real protocol."""

    def __init__(self, feed, device: str) -> None:
        self._feed = feed
        self._device = device

    def fetch(self, request):
        from opti_oignon.veilid.protocol import respond_to_request

        return respond_to_request(self._feed, request, device=self._device)


# ---------------------------------------------------------------------------
# The S263 update store, flat-loaded (the s263 idiom; plaintext fallback)
# ---------------------------------------------------------------------------

_ISO: dict = {}


def _updates_module():
    if "mod" not in _ISO:
        spec = importlib.util.spec_from_file_location(
            "s264_note_updates_iso", str(UPDATES_STORE_SRC)
        )
        if spec is None or spec.loader is None:
            _ISO["mod"] = None
        else:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["s264_note_updates_iso"] = mod
            try:
                spec.loader.exec_module(mod)
                _ISO["mod"] = mod
            except Exception:
                _ISO["mod"] = None
    return _ISO["mod"]


def _permissive(note_id: str, user_id: str) -> bool:
    return True


def _make_updates_store(tmp_path, parent_lookup=_permissive):
    mod = _updates_module()
    assert mod is not None, "note_updates_store module failed to load"
    return mod, mod.NoteUpdatesStore(
        root=str(tmp_path), parent_lookup=parent_lookup
    )


def _b64(blob: bytes) -> str:
    return base64.b64encode(blob).decode("ascii")


def _update_payload(note_id: str, seq: int, blob: bytes) -> dict:
    return {
        "note_id": note_id,
        "seq": seq,
        "update_blob_b64": _b64(blob),
        "author_device": "",
    }


def _wire_kinds(batch: dict) -> list[str]:
    return [w.get("kind") for w in batch.get("records", [])]


def _wire_update_seqs(batch: dict) -> list[int]:
    out = []
    for w in batch.get("records", []):
        if w.get("kind") == "note_update":
            out.append((w.get("payload") or {}).get("seq"))
    return out


# ---------------------------------------------------------------------------
# Family 1 -- the record kind (red before the edit)
# ---------------------------------------------------------------------------


class TestRecordKindNoteUpdate:
    def test_kind_member_present(self):
        records = _veilid()["records"]
        member = getattr(records.RecordKind, "NOTE_UPDATE", None)
        assert member is not None, "RecordKind.NOTE_UPDATE missing"

    def test_kind_value_is_note_update(self):
        records = _veilid()["records"]
        member = getattr(records.RecordKind, "NOTE_UPDATE", None)
        assert member is not None, "RecordKind.NOTE_UPDATE missing"
        assert member.value == "note_update"

    def test_kind_in_decoder_allowlist(self):
        records = _veilid()["records"]
        assert "note_update" in records.RECORD_KINDS

    def test_kind_is_not_sensitive(self):
        mods = _veilid()
        member = getattr(mods["records"].RecordKind, "NOTE_UPDATE", None)
        assert member is not None, "RecordKind.NOTE_UPDATE missing"
        assert member.value not in mods["engine"].SENSITIVE_KINDS

    def test_decode_roundtrip_note_update(self):
        records = _veilid()["records"]
        assert "note_update" in records.RECORD_KINDS
        rec = records.new_record(
            "note_update",
            "n1:1",
            _update_payload("n1", 1, b"\x01\x02"),
            device="dev-a",
            clock=1,
        )
        back = records.decode_record(records.encode_record(rec))
        assert back is not None
        assert back.kind.value == "note_update"
        assert back.record_id == "n1:1"
        assert back.payload["seq"] == 1


# ---------------------------------------------------------------------------
# Family 2 -- the producer (red before the edit)
# ---------------------------------------------------------------------------


class TestNoteUpdateProducer:
    def test_producer_exists(self):
        producers = _veilid()["producers"]
        fn = getattr(producers, "note_update_record", None)
        assert fn is not None, "producers.note_update_record missing"

    def test_record_identity_is_note_id_colon_seq(self):
        producers = _veilid()["producers"]
        fn = getattr(producers, "note_update_record", None)
        assert fn is not None, "producers.note_update_record missing"
        rec = fn(
            "n1", 7, _update_payload("n1", 7, b"\x07"), device="dev-a", clock=1
        )
        assert rec.record_id == "n1:7"
        assert rec.kind.value == "note_update"

    def test_no_tombstone_parameter_and_never_deleted(self):
        producers = _veilid()["producers"]
        fn = getattr(producers, "note_update_record", None)
        assert fn is not None, "producers.note_update_record missing"
        assert "deleted" not in inspect.signature(fn).parameters
        rec = fn(
            "n1", 1, _update_payload("n1", 1, b"\x01"), device="dev-a", clock=1
        )
        assert rec.deleted is False

    def test_content_hash_round(self):
        mods = _veilid()
        fn = getattr(mods["producers"], "note_update_record", None)
        assert fn is not None, "producers.note_update_record missing"
        rec = fn(
            "n1", 1, _update_payload("n1", 1, b"\x01"), device="dev-a", clock=1
        )
        assert mods["records"].verify_record_hash(rec) is True


# ---------------------------------------------------------------------------
# Family 3 -- publish_note_update (red before the edit)
# ---------------------------------------------------------------------------


def _bare_engine(tmp_path, mods, *, device="server", signer=None, **kwargs):
    feed = mods["change_feed"].ChangeFeed(root=tmp_path / device)
    store = mods["peers"].PeerStore(root=tmp_path / device)
    ledger = mods["ledger"].DeferredLedger(root=tmp_path / device)
    eng = mods["engine"].SyncEngine(
        device=device,
        feed=feed,
        store=store,
        signer=signer or _make_signer(device + "-seed"),
        ledger=ledger,
        **kwargs,
    )
    return eng, feed, store


class TestPublishNoteUpdate:
    def test_publish_helper_exists(self, tmp_path):
        mods = _veilid()
        _set_mode("daily")
        eng, _feed, _store = _bare_engine(tmp_path, mods)
        assert getattr(eng, "publish_note_update", None) is not None

    def test_publish_journals_and_returns_sequence(self, tmp_path):
        mods = _veilid()
        _set_mode("daily")
        eng, feed, _store = _bare_engine(tmp_path, mods)
        pn = getattr(eng, "publish_note_update", None)
        assert pn is not None, "engine.publish_note_update missing"
        seq = pn("n1", 1, _update_payload("n1", 1, b"\x01"), clock=1)
        assert isinstance(seq, int) and seq >= 1
        recs = feed.current_records()
        assert any(
            r.kind.value == "note_update" and r.record_id == "n1:1"
            for r in recs
        )

    def test_publish_signs_local_records(self, tmp_path):
        mods = _veilid()
        _set_mode("daily")
        eng, feed, _store = _bare_engine(tmp_path, mods)
        pn = getattr(eng, "publish_note_update", None)
        assert pn is not None, "engine.publish_note_update missing"
        pn("n1", 1, _update_payload("n1", 1, b"\x01"), clock=1)
        rec = next(
            r for r in feed.current_records() if r.kind.value == "note_update"
        )
        assert rec.signature != ""

    def test_publish_mode_free_local(self, tmp_path):
        mods = _veilid()
        _set_mode("bulbe")
        try:
            eng, feed, _store = _bare_engine(tmp_path, mods)
            pn = getattr(eng, "publish_note_update", None)
            assert pn is not None, "engine.publish_note_update missing"
            pn("n1", 1, _update_payload("n1", 1, b"\x01"), clock=1)
            assert any(
                r.kind.value == "note_update" for r in feed.current_records()
            )
        finally:
            _set_mode("daily")


# ---------------------------------------------------------------------------
# Family 4 -- the journaling glue, by source pin (red before the edit)
# ---------------------------------------------------------------------------


class TestGlueSource:
    def test_glue_function_present(self):
        src = _read(UPDATES_STORE_SRC)
        assert src != "", "note_updates_store.py source is absent"
        assert "def _sync_publish_note_update(" in src

    def test_glue_probes_guard_first(self):
        src = _read(UPDATES_STORE_SRC)
        assert "def _sync_publish_note_update(" in src
        assert "veilid_available" in src

    def test_append_seam_calls_the_glue(self):
        src = _read(UPDATES_STORE_SRC)
        assert src.count("_sync_publish_note_update") >= 2
        assert "sync_publish" in src

    def test_payload_keys_and_exclusions(self):
        src = _read(UPDATES_STORE_SRC)
        assert '"update_blob_b64"' in src
        assert '"note_id"' in src
        assert '"author_device"' in src
        assert '"mobile_allowed"' not in src
        assert "'mobile_allowed'" not in src
        assert '"user_id":' not in src


# ---------------------------------------------------------------------------
# Family 5 -- the glue, behaviourally (red before the edit)
# ---------------------------------------------------------------------------


class _StubEngine:
    def __init__(self) -> None:
        self.calls: list = []

    def current_clock(self, kind, record_id) -> int:
        return 0

    def publish_note_update(self, note_id, seq, payload, *, clock, updated_at=""):
        self.calls.append(
            {
                "note_id": note_id,
                "seq": seq,
                "payload": payload,
                "clock": clock,
                "updated_at": updated_at,
            }
        )
        return len(self.calls)


class _RaisingEngine(_StubEngine):
    def publish_note_update(self, *a, **k):  # pragma: no cover - trivial
        raise RuntimeError("journal down")


def _require_glue(mod) -> None:
    assert mod is not None, "note_updates_store module failed to load"
    assert (
        getattr(mod, "_sync_publish_note_update", None) is not None
    ), "_sync_publish_note_update missing"


@pytest.fixture()
def live_update_glue(monkeypatch):
    """Veilid 'present' plus a recording stub engine behind the glue."""
    mods = _veilid()
    stub = _StubEngine()
    monkeypatch.setattr(mods["guard"], "veilid_available", lambda: True)
    monkeypatch.setattr(mods["engine"], "get_sync_engine", lambda: stub)
    return stub


@pytest.fixture()
def raising_update_glue(monkeypatch):
    mods = _veilid()
    stub = _RaisingEngine()
    monkeypatch.setattr(mods["guard"], "veilid_available", lambda: True)
    monkeypatch.setattr(mods["engine"], "get_sync_engine", lambda: stub)
    return stub


class TestGlueBehaviour:
    def test_append_journals_once_with_opaque_payload(
        self, tmp_path, live_update_glue
    ):
        mod, store = _make_updates_store(tmp_path)
        _require_glue(mod)
        rec = store.append_update("n1", b"\x01\x02", author_device="dev-a")
        assert rec.seq == 1
        assert len(live_update_glue.calls) == 1
        call = live_update_glue.calls[0]
        assert call["note_id"] == "n1"
        assert call["seq"] == 1
        payload = call["payload"]
        assert payload["note_id"] == "n1"
        assert payload["seq"] == 1
        assert payload["update_blob_b64"] == _b64(b"\x01\x02")
        assert payload["author_device"] == "dev-a"
        assert "mobile_allowed" not in payload
        assert "user_id" not in payload

    def test_each_update_is_its_own_record_key(
        self, tmp_path, live_update_glue
    ):
        mod, store = _make_updates_store(tmp_path)
        _require_glue(mod)
        store.append_update("n1", b"\x01")
        store.append_update("n1", b"\x02")
        clocks = [c["clock"] for c in live_update_glue.calls]
        seqs = [c["seq"] for c in live_update_glue.calls]
        assert seqs == [1, 2]
        assert clocks == [1, 1]

    def test_sync_publish_false_suppresses_the_glue(
        self, tmp_path, live_update_glue
    ):
        mod, store = _make_updates_store(tmp_path)
        _require_glue(mod)
        sig = inspect.signature(store.append_update)
        assert "sync_publish" in sig.parameters
        store.append_update(
            "n1", b"\x05", seq=5, author_device="dev-a", sync_publish=False
        )
        assert live_update_glue.calls == []
        assert store.count_updates("n1") == 1

    def test_append_survives_publish_error(
        self, tmp_path, raising_update_glue
    ):
        mod, store = _make_updates_store(tmp_path)
        _require_glue(mod)
        rec = store.append_update("n1", b"\x01")
        assert rec.seq == 1
        assert store.count_updates("n1") == 1

    def test_absent_framework_publishes_nothing(self, tmp_path, monkeypatch):
        mods = _veilid()
        stub = _StubEngine()
        monkeypatch.setattr(mods["guard"], "veilid_available", lambda: False)
        monkeypatch.setattr(mods["engine"], "get_sync_engine", lambda: stub)
        mod, store = _make_updates_store(tmp_path)
        _require_glue(mod)
        store.append_update("n1", b"\x01")
        assert stub.calls == []
        assert store.count_updates("n1") == 1

    def test_refused_append_never_journals(self, tmp_path, live_update_glue):
        mod, store = _make_updates_store(
            tmp_path, parent_lookup=lambda nid, uid: False
        )
        _require_glue(mod)
        with pytest.raises(mod.NoteUpdateRefused):
            store.append_update("ghost", b"\x01")
        assert live_update_glue.calls == []


# ---------------------------------------------------------------------------
# Family 6 -- the apply leg (red before the edit)
# ---------------------------------------------------------------------------


def _relay_pair(tmp_path, mods, *, receiver_kwargs=None):
    """dev-a authors; dev-b receives over the real protocol. Returns parts."""
    _set_mode("daily")
    _AUDIT["events"].clear()
    signer_a = _make_signer("origin-a")
    signer_b = _make_signer("receiver-b")
    eng_a, feed_a, _store_a = _bare_engine(
        tmp_path, mods, device="dev-a", signer=signer_a
    )
    feed_b = mods["change_feed"].ChangeFeed(root=tmp_path / "dev-b")
    store_b = mods["peers"].PeerStore(root=tmp_path / "dev-b")
    ledger_b = mods["ledger"].DeferredLedger(root=tmp_path / "dev-b")
    store_b.add_peer(
        "dev-a",
        "rk-a",
        signing_pub=mods["signing"].encode_public_key(signer_a.public_key()),
    )
    eng_b = mods["engine"].SyncEngine(
        device="dev-b",
        feed=feed_b,
        store=store_b,
        signer=signer_b,
        ledger=ledger_b,
        **(receiver_kwargs or {}),
    )
    return eng_a, feed_a, eng_b, feed_b, store_b


class TestApplyLeg:
    def test_engine_accepts_update_sink_kwarg(self):
        mods = _veilid()
        params = inspect.signature(mods["engine"].SyncEngine.__init__).parameters
        assert "update_sink" in params

    def test_default_sink_and_builder_present_in_source(self):
        src = _read(ENGINE_SRC)
        assert "_default_update_sink" in src
        assert "def _update_sink_for(" in src

    def test_apply_lands_with_authors_seq(self, tmp_path):
        mods = _veilid()
        engine_mod = mods["engine"]
        builder = getattr(engine_mod, "_update_sink_for", None)
        assert builder is not None, "_update_sink_for missing"
        umod, ustore = _make_updates_store(tmp_path / "store-b")
        eng_a, feed_a, eng_b, feed_b, _sb = _relay_pair(
            tmp_path, mods, receiver_kwargs={"update_sink": builder(ustore)}
        )
        eng_a.publish_note_update(
            "n1", 4, _update_payload("n1", 4, b"\x04"), clock=1
        )
        result = eng_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert result.refused == 0
        rows = ustore.list_updates("n1")
        assert [r.seq for r in rows] == [4]
        assert rows[0].author_device == "dev-a"
        assert any(
            r.kind.value == "note_update" and r.record_id == "n1:4"
            for r in feed_b.current_records()
        )

    def test_apply_refuses_dead_parent_everywhere(self, tmp_path):
        mods = _veilid()
        engine_mod = mods["engine"]
        builder = getattr(engine_mod, "_update_sink_for", None)
        assert builder is not None, "_update_sink_for missing"
        umod, ustore = _make_updates_store(
            tmp_path / "store-b", parent_lookup=lambda nid, uid: nid == "n1"
        )
        eng_a, feed_a, eng_b, feed_b, _sb = _relay_pair(
            tmp_path, mods, receiver_kwargs={"update_sink": builder(ustore)}
        )
        eng_a.publish_note_update(
            "ghost", 1, _update_payload("ghost", 1, b"\x01"), clock=1
        )
        eng_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert ustore.count_updates("ghost") == 0
        assert all(
            r.record_id != "ghost:1" for r in feed_b.current_records()
        )
        assert any(
            e.get("action") == "sync_note_update_refused"
            for e in _AUDIT["events"]
        )

    def test_apply_duplicate_redelivery_is_benign(self, tmp_path):
        mods = _veilid()
        engine_mod = mods["engine"]
        builder = getattr(engine_mod, "_update_sink_for", None)
        assert builder is not None, "_update_sink_for missing"
        umod, ustore = _make_updates_store(tmp_path / "store-b")
        eng_a, feed_a, eng_b, feed_b, store_b = _relay_pair(
            tmp_path, mods, receiver_kwargs={"update_sink": builder(ustore)}
        )
        eng_a.publish_note_update(
            "n1", 1, _update_payload("n1", 1, b"\x01"), clock=1
        )
        eng_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        store_b.add_peer(
            "dev-a-bis",
            "rk-a2",
            signing_pub=mods["signing"].encode_public_key(
                _make_signer("origin-a").public_key()
            ),
        )
        eng_b.run_round("dev-a-bis", FakeServingPeer(feed_a, "dev-a"))
        assert ustore.count_updates("n1") == 1
        assert (
            sum(
                1
                for r in feed_b.current_records()
                if r.record_id == "n1:1" and r.kind.value == "note_update"
            )
            == 1
        )

    def test_apply_unattributable_payload_refuses(self, tmp_path):
        mods = _veilid()
        engine_mod = mods["engine"]
        builder = getattr(engine_mod, "_update_sink_for", None)
        assert builder is not None, "_update_sink_for missing"
        umod, ustore = _make_updates_store(tmp_path / "store-b")
        eng_a, feed_a, eng_b, feed_b, _sb = _relay_pair(
            tmp_path, mods, receiver_kwargs={"update_sink": builder(ustore)}
        )
        rec = engine_mod.record_from_payload(
            "note_update",
            "n1:9",
            {"note_id": "n1", "update_blob_b64": _b64(b"\x09")},
            device="dev-a",
            clock=1,
        )
        eng_a.publish(rec)
        eng_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert ustore.count_updates("n1") == 0
        assert all(r.record_id != "n1:9" for r in feed_b.current_records())

    def test_sink_absent_refuses_fail_secure(self, tmp_path):
        mods = _veilid()
        params = inspect.signature(
            mods["engine"].SyncEngine.__init__
        ).parameters
        assert "update_sink" in params
        umod, ustore = _make_updates_store(tmp_path / "store-b")
        eng_a, feed_a, eng_b, feed_b, _sb = _relay_pair(
            tmp_path,
            mods,
            receiver_kwargs={"update_sink": lambda record: False},
        )
        eng_a.publish_note_update(
            "n1", 1, _update_payload("n1", 1, b"\x01"), clock=1
        )
        eng_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert all(
            r.kind.value != "note_update" for r in feed_b.current_records()
        )


# ---------------------------------------------------------------------------
# Family 7 -- the serve floor and the watermark-forward rule (red before)
# ---------------------------------------------------------------------------


@pytest.fixture()
def update_serve_world(tmp_path):
    """A serving engine with a note, two updates, and a conversation.

    Yields ``(mods, build_engine, request)``; ``build_engine`` takes the
    engine kwargs (``note_gate`` and ``update_watermark_gate`` among them)
    over the SAME feed and peer store. Peers: ``phone-1`` (confirmed,
    phone-class), ``desk-1`` (confirmed, grandfathered NULL class).
    """
    mods = _veilid()
    _set_mode("daily")
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
    pn = getattr(base, "publish_note_update", None)
    if pn is not None:
        pn("n1", 1, _update_payload("n1", 1, b"\x01"), clock=1)
        pn("n1", 2, _update_payload("n1", 2, b"\x02"), clock=1)

    store.add_peer("desk-1", "rk-desk")
    store.add_peer("phone-1", "rk-phone")
    store.set_device_class("phone-1", "phone")
    request = mods["protocol"].build_delta_request(device="asker", watermark=0)
    yield mods, build_engine, request
    _AUDIT["events"].clear()


def _require_serve_surface(mods) -> None:
    params = inspect.signature(mods["engine"].SyncEngine.__init__).parameters
    assert "update_watermark_gate" in params, "update_watermark_gate missing"
    assert (
        getattr(mods["engine"].SyncEngine, "publish_note_update", None)
        is not None
    ), "engine.publish_note_update missing"


class TestServeFloor:
    def test_engine_accepts_update_watermark_gate_kwarg(self):
        mods = _veilid()
        params = inspect.signature(
            mods["engine"].SyncEngine.__init__
        ).parameters
        assert "update_watermark_gate" in params

    def test_phone_gets_no_update_for_flag_off_note(self, update_serve_world):
        mods, build_engine, request = update_serve_world
        _require_serve_surface(mods)
        eng = build_engine(
            note_gate=lambda nid: False, update_watermark_gate=lambda nid: 0
        )
        batch = eng.serve_request(request, peer_id="phone-1")
        kinds = _wire_kinds(batch)
        assert "note_update" not in kinds
        assert "note" not in kinds
        assert "conversation" in kinds

    def test_phone_gets_updates_for_flag_on_note(self, update_serve_world):
        mods, build_engine, request = update_serve_world
        _require_serve_surface(mods)
        eng = build_engine(
            note_gate=lambda nid: True, update_watermark_gate=lambda nid: 0
        )
        batch = eng.serve_request(request, peer_id="phone-1")
        assert _wire_update_seqs(batch) == [1, 2]
        assert "note" in _wire_kinds(batch)

    def test_watermark_forward_serving(self, update_serve_world):
        mods, build_engine, request = update_serve_world
        _require_serve_surface(mods)
        eng = build_engine(
            note_gate=lambda nid: True, update_watermark_gate=lambda nid: 1
        )
        batch = eng.serve_request(request, peer_id="phone-1")
        assert _wire_update_seqs(batch) == [2]

    def test_indeterminable_watermark_drops_updates(self, update_serve_world):
        mods, build_engine, request = update_serve_world
        _require_serve_surface(mods)

        def _boom(nid):
            raise RuntimeError("watermark store down")

        eng = build_engine(
            note_gate=lambda nid: True, update_watermark_gate=_boom
        )
        batch = eng.serve_request(request, peer_id="phone-1")
        assert _wire_update_seqs(batch) == []
        assert "conversation" in _wire_kinds(batch)

    def test_no_gate_drops_updates_for_phone(self, update_serve_world):
        mods, build_engine, request = update_serve_world
        _require_serve_surface(mods)
        eng = build_engine(update_watermark_gate=lambda nid: 0)
        batch = eng.serve_request(request, peer_id="phone-1")
        assert _wire_update_seqs(batch) == []

    def test_unparseable_parent_drops_for_phone(self, update_serve_world):
        mods, build_engine, request = update_serve_world
        _require_serve_surface(mods)
        base = build_engine()
        rec = mods["engine"].record_from_payload(
            "note_update",
            "orphan:1",
            {"seq": 1, "update_blob_b64": _b64(b"\x01")},
            device="server",
            clock=1,
        )
        base.publish(rec)
        eng = build_engine(
            note_gate=lambda nid: True, update_watermark_gate=lambda nid: 0
        )
        batch = eng.serve_request(request, peer_id="phone-1")
        ids = [w.get("id") for w in batch.get("records", [])]
        assert "orphan:1" not in ids

    def test_desktop_gets_full_tail(self, update_serve_world):
        mods, build_engine, request = update_serve_world
        _require_serve_surface(mods)
        eng = build_engine(
            note_gate=lambda nid: False, update_watermark_gate=lambda nid: 99
        )
        batch = eng.serve_request(request, peer_id="desk-1")
        assert _wire_update_seqs(batch) == [1, 2]
        assert "note" in _wire_kinds(batch)

    def test_high_water_untouched_by_update_filtering(
        self, update_serve_world
    ):
        mods, build_engine, request = update_serve_world
        _require_serve_surface(mods)
        eng = build_engine(
            note_gate=lambda nid: False, update_watermark_gate=lambda nid: 0
        )
        phone = eng.serve_request(request, peer_id="phone-1")
        desk = eng.serve_request(request, peer_id="desk-1")
        assert phone.get("high_water") == desk.get("high_water")


# ---------------------------------------------------------------------------
# Family 8 -- the relayed-signature property across a serve hop (red before)
# ---------------------------------------------------------------------------


class TestRelayedSignature:
    def test_relay_preserves_signature_verbatim(self, tmp_path):
        mods = _veilid()
        engine_mod = mods["engine"]
        builder = getattr(engine_mod, "_update_sink_for", None)
        assert builder is not None, "_update_sink_for missing"
        umod, ustore_b = _make_updates_store(tmp_path / "store-b")
        eng_a, feed_a, eng_b, feed_b, _sb = _relay_pair(
            tmp_path, mods, receiver_kwargs={"update_sink": builder(ustore_b)}
        )
        eng_a.publish_note_update(
            "n1", 1, _update_payload("n1", 1, b"\x01"), clock=1
        )
        origin = next(
            r for r in feed_a.current_records() if r.kind.value == "note_update"
        )
        eng_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        relayed = next(
            r for r in feed_b.current_records() if r.kind.value == "note_update"
        )
        assert relayed.signature == origin.signature
        assert relayed.device == "dev-a"

    def test_third_device_verifies_against_origin_key(self, tmp_path):
        mods = _veilid()
        engine_mod = mods["engine"]
        builder = getattr(engine_mod, "_update_sink_for", None)
        assert builder is not None, "_update_sink_for missing"
        umod, ustore_b = _make_updates_store(tmp_path / "store-b")
        eng_a, feed_a, eng_b, feed_b, _sb = _relay_pair(
            tmp_path, mods, receiver_kwargs={"update_sink": builder(ustore_b)}
        )
        eng_a.publish_note_update(
            "n1", 1, _update_payload("n1", 1, b"\x01"), clock=1
        )
        eng_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))

        umod2, ustore_c = _make_updates_store(tmp_path / "store-c")
        feed_c = mods["change_feed"].ChangeFeed(root=tmp_path / "dev-c")
        store_c = mods["peers"].PeerStore(root=tmp_path / "dev-c")
        ledger_c = mods["ledger"].DeferredLedger(root=tmp_path / "dev-c")
        store_c.add_peer("dev-b", "rk-b")
        store_c.add_peer(
            "dev-a",
            "rk-a",
            signing_pub=mods["signing"].encode_public_key(
                _make_signer("origin-a").public_key()
            ),
        )
        eng_c = mods["engine"].SyncEngine(
            device="dev-c",
            feed=feed_c,
            store=store_c,
            signer=_make_signer("third-c"),
            ledger=ledger_c,
            update_sink=builder(ustore_c),
        )
        result = eng_c.run_round("dev-b", FakeServingPeer(feed_b, "dev-b"))
        assert result.refused == 0
        rows = ustore_c.list_updates("n1")
        assert [r.seq for r in rows] == [1]
        assert rows[0].author_device == "dev-a"

    def test_relay_refuses_without_origin_key(self, tmp_path):
        mods = _veilid()
        engine_mod = mods["engine"]
        builder = getattr(engine_mod, "_update_sink_for", None)
        assert builder is not None, "_update_sink_for missing"
        umod, ustore_b = _make_updates_store(tmp_path / "store-b")
        eng_a, feed_a, eng_b, feed_b, _sb = _relay_pair(
            tmp_path, mods, receiver_kwargs={"update_sink": builder(ustore_b)}
        )
        eng_a.publish_note_update(
            "n1", 1, _update_payload("n1", 1, b"\x01"), clock=1
        )
        eng_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))

        umod2, ustore_c = _make_updates_store(tmp_path / "store-c")
        feed_c = mods["change_feed"].ChangeFeed(root=tmp_path / "dev-c")
        store_c = mods["peers"].PeerStore(root=tmp_path / "dev-c")
        ledger_c = mods["ledger"].DeferredLedger(root=tmp_path / "dev-c")
        store_c.add_peer("dev-b", "rk-b")
        eng_c = mods["engine"].SyncEngine(
            device="dev-c",
            feed=feed_c,
            store=store_c,
            signer=_make_signer("third-c"),
            ledger=ledger_c,
            update_sink=builder(ustore_c),
        )
        result = eng_c.run_round("dev-b", FakeServingPeer(feed_b, "dev-b"))
        assert result.refused >= 1
        assert ustore_c.count_updates("n1") == 0


# ---------------------------------------------------------------------------
# Family 9 -- design-green reassertions (GREEN on pristine, must stay green)
# ---------------------------------------------------------------------------


class TestDesignGreenReassertions:
    def test_s262_engine_markers_alive(self):
        src = _read(ENGINE_SRC)
        assert "filter-at-serve" in src
        assert "never re-signs" in src

    def test_s263_store_discipline_alive(self):
        raw = b""
        try:
            raw = UPDATES_STORE_SRC.read_bytes()
        except OSError:
            raw = b""
        assert raw != b"", "note_updates_store.py source is absent"
        raw.decode("ascii")
        src = raw.decode("ascii")
        for verb in ("SELECT", "INSERT", "UPDATE", "DELETE"):
            assert 'f"' + verb not in src
            assert "f'" + verb not in src
        assert "safe_connect" in src
        assert "effective_user_id" in src
        assert "checkpoint_before_apply = True" in src
        assert "UPDATE note_update " not in src
        assert "update_blob =" not in src

    def test_sensitive_kinds_exact_skill_only(self):
        mods = _veilid()
        assert mods["engine"].SENSITIVE_KINDS == frozenset({"skill"})
