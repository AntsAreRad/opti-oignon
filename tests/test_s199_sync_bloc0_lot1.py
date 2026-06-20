#!/usr/bin/env python3
"""S199 -- sync cycle Bloc 0 lot 1: clock discipline + conversation producer.

One tight test group per fix (SYN-01 enablement, ROADMAP_SYNC_CYCLE Bloc 0):

- ``ChangeFeed.current_clock(kind, key)``: the read side of per-key clock
  discipline. Unseen key yields 0 (so the first minted clock is 1); the value
  is MAX(clock) for the key (tombstones included, so a re-create out-clocks a
  delete and wins the LWW merge); keys and kinds are independent; the helper
  accepts a ``RecordKind`` or its raw string value; clocks stay monotonic over
  save/apply interleavings (PRT-02 journals clock-only adoptions, so a local
  mint after a remote apply continues past the winner's clock).
  ``SyncEngine.current_clock`` delegates to the engine's resolved feed.

- Conversation producer wiring: ``create_conversation``, ``add_message``,
  ``rename_conversation``, ``update_conversation_metadata``,
  ``delete_last_message`` publish a self-sufficient full-state snapshot
  (state-based LWW: the feed collapses to latest-per-key and CHF-02 compaction
  will delete superseded rows); ``delete_conversation`` publishes a tombstone.
  The payload carries the owning user (``effective_user_id`` pattern, scoping
  in the payload, never the key), plaintext message content (cross-device
  portability; the S125 field key is per-install), and no local SQLite message
  ids (device-local identities).

- Hook contract: the domain commit happens first and a snapshot or publish
  failure never breaks the save; the hook is a no-op when the veilid framework
  is absent (``guard.veilid_available`` probe) and pays nothing then; the hook
  is mode-free (publishing works under Bulbe -- only the wire is Daily-gated).

Loader idiom: spec_from_file_location with sys.modules registration BEFORE
exec_module (3.12+ dataclass processing), package stubs for the absolute and
relative imports. conversation.py's hard deps are stubbed: config (DATA_DIR to
a session tmp), db_utils (plain sqlite3), encryption (a reversible marker
transform, so payload plaintext is assertable), context_manager,
user_isolation. Every feed gets an injected tmp root; the engine singleton is
injected per test via ``set_sync_engine``.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"

_MODE = {"fn": (lambda: "daily")}


def set_mode(value: str = "daily") -> None:
    def _gm() -> str:
        return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


_SESSION_DATA_DIR = Path(tempfile.mkdtemp(prefix="oo_s199_data_"))


def _ensure_stubs() -> None:
    for name, sub in (("opti_oignon", OO), ("opti_oignon.veilid", VEILID)):
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
    if "opti_oignon.config" not in sys.modules:
        cfg = types.ModuleType("opti_oignon.config")
        cfg.DATA_DIR = _SESSION_DATA_DIR  # type: ignore[attr-defined]
        sys.modules["opti_oignon.config"] = cfg
    if "opti_oignon.db_utils" not in sys.modules:
        dbu = types.ModuleType("opti_oignon.db_utils")
        dbu.safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)  # type: ignore[attr-defined]
        sys.modules["opti_oignon.db_utils"] = dbu
    if "opti_oignon.encryption" not in sys.modules:
        # Reversible marker transform: the snapshot must carry the DECRYPTED
        # content, so the at-rest marker must never appear in a payload.
        enc = types.ModuleType("opti_oignon.encryption")
        enc.encrypt_field = lambda v: "ENCMARK:" + v  # type: ignore[attr-defined]
        enc.decrypt_field = (  # type: ignore[attr-defined]
            lambda v: v[len("ENCMARK:"):] if isinstance(v, str) and v.startswith("ENCMARK:") else v
        )
        sys.modules["opti_oignon.encryption"] = enc
    if "opti_oignon.context_manager" not in sys.modules:
        cm = types.ModuleType("opti_oignon.context_manager")
        cm.estimate_tokens = lambda text, model=None: max(1, len(text) // 4)  # type: ignore[attr-defined]
        sys.modules["opti_oignon.context_manager"] = cm
    if "opti_oignon.user_isolation" not in sys.modules:
        ui = types.ModuleType("opti_oignon.user_isolation")
        ui.DEFAULT_LOCAL_USER = "local"  # type: ignore[attr-defined]
        ui.effective_user_id = (  # type: ignore[attr-defined]
            lambda user_id=None, single_user_mode=True: "local" if user_id is None else user_id
        )
        sys.modules["opti_oignon.user_isolation"] = ui


def _load(name: str, base: Path = VEILID, package: str = "opti_oignon.veilid"):
    full = f"{package}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(base / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod  # register before exec (3.12+ dataclass processing)
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
guard = _load("guard")
records = _load("records")
reconcile = _load("reconcile")
change_feed = _load("change_feed")
protocol = _load("protocol")
peers = _load("peers")
producers = _load("producers")
sync_engine = _load("sync_engine")
conversation = _load("conversation", base=OO, package="opti_oignon")
RecordKind = records.RecordKind

_REAL_VEILID_AVAILABLE = guard.veilid_available


@pytest.fixture(autouse=True)
def _daily_reset_and_available():
    set_mode("daily")
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    # The container has no veilid framework; the hook gates on this probe, so
    # force it on by default and let the no-op test force it off.
    guard.veilid_available = lambda: True
    yield
    guard.veilid_available = _REAL_VEILID_AVAILABLE
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    set_mode("daily")


def _feed(tmp_path, name="feed"):
    return change_feed.ChangeFeed(root=tmp_path / name)


def _install_engine(tmp_path, device="dev-a"):
    f = _feed(tmp_path)
    eng = sync_engine.SyncEngine(device=device, feed=f)
    sync_engine.set_sync_engine(eng)
    return f, eng


def _manager(tmp_path):
    return conversation.ConversationManager(db_path=tmp_path / "conversations.db")


def _rec(record_id, clock, *, device="remote", payload=None, kind=None, deleted=False):
    return records.new_record(
        kind=kind or RecordKind.CONVERSATION,
        record_id=record_id,
        payload=payload if payload is not None else {"v": clock},
        device=device,
        clock=clock,
        deleted=deleted,
    )


def _batch(device, high_water, recs):
    return {
        "v": protocol.PROTOCOL_VERSION,
        "type": "record_batch",
        "device": device,
        "high_water": high_water,
        "records": records.encode_records(recs),
    }


def _latest(feed, record_id):
    for r in feed.current_records():
        if r.record_id == record_id:
            return r
    return None


# --- current_clock: the read side of clock discipline (SYN-01) --------------


class TestCurrentClockHelper:
    def test_unseen_key_yields_zero_so_first_clock_is_one(self, tmp_path):
        f = _feed(tmp_path)
        assert f.current_clock(RecordKind.CONVERSATION, "ghost") == 0
        assert f.current_clock(RecordKind.CONVERSATION, "ghost") + 1 == 1

    def test_returns_max_clock_for_the_key(self, tmp_path):
        f = _feed(tmp_path)
        for c in (1, 2, 3):
            f.record(_rec("c1", c, device="dev-a"))
        assert f.current_clock(RecordKind.CONVERSATION, "c1") == 3

    def test_tombstone_clock_is_counted(self, tmp_path):
        # A re-create after a delete must out-clock the tombstone to win LWW.
        f = _feed(tmp_path)
        f.record(_rec("c1", 3, device="dev-a"))
        f.record(_rec("c1", 4, device="dev-a", payload={}, deleted=True))
        assert f.current_clock(RecordKind.CONVERSATION, "c1") == 4
        assert f.current_clock(RecordKind.CONVERSATION, "c1") + 1 == 5

    def test_keys_and_kinds_are_independent(self, tmp_path):
        f = _feed(tmp_path)
        f.record(_rec("c1", 7, device="dev-a"))
        f.record(_rec("c1", 2, device="dev-a", kind=RecordKind.SKILL))
        assert f.current_clock(RecordKind.CONVERSATION, "c1") == 7
        assert f.current_clock(RecordKind.SKILL, "c1") == 2
        assert f.current_clock(RecordKind.CONVERSATION, "other") == 0

    def test_kind_accepts_enum_and_raw_string(self, tmp_path):
        f = _feed(tmp_path)
        f.record(_rec("c1", 5, device="dev-a"))
        assert f.current_clock(RecordKind.CONVERSATION, "c1") == 5
        assert f.current_clock("conversation", "c1") == 5

    def test_empty_record_id_raises(self, tmp_path):
        f = _feed(tmp_path)
        with pytest.raises(ValueError):
            f.current_clock(RecordKind.CONVERSATION, "")

    def test_monotonic_over_save_apply_interleaving(self, tmp_path):
        # Local mint, then a remote winner applied at a higher clock, then the
        # next local mint must continue past the winner (PRT-02 journals it).
        f = _feed(tmp_path)
        first = f.current_clock(RecordKind.CONVERSATION, "c1") + 1
        assert first == 1
        f.record(_rec("c1", first, device="dev-a", payload={"body": "local"}))
        winner = _rec("c1", 5, device="dev-b", payload={"body": "remote"})
        result = protocol.apply_record_batch(f, _batch("dev-b", 9, [winner]))
        assert result.applied == 1
        assert f.current_clock(RecordKind.CONVERSATION, "c1") == 5
        assert f.current_clock(RecordKind.CONVERSATION, "c1") + 1 == 6

    def test_engine_delegates_to_its_resolved_feed(self, tmp_path):
        f, eng = _install_engine(tmp_path)
        f.record(_rec("c1", 4, device="dev-a"))
        assert eng.current_clock(RecordKind.CONVERSATION, "c1") == 4
        assert eng.current_clock(RecordKind.CONVERSATION, "unseen") == 0


# --- conversation producer wiring (the six save-path hooks) ------------------


class TestConversationProducerWiring:
    def test_create_publishes_initial_snapshot(self, tmp_path):
        feed, _ = _install_engine(tmp_path, device="dev-a")
        mgr = _manager(tmp_path)
        conv = mgr.create_conversation(title="Notes")
        rec = _latest(feed, conv.id)
        assert rec is not None
        assert rec.kind == RecordKind.CONVERSATION
        assert rec.clock == 1
        assert rec.deleted is False
        assert rec.device == "dev-a"
        assert rec.payload["user_id"] == "local"
        snap = rec.payload["conversation"]
        assert snap["id"] == conv.id
        assert snap["title"] == "Notes"
        assert snap["messages"] == []
        assert rec.updated_at == conv.updated_at

    def test_add_message_publishes_full_plaintext_state(self, tmp_path):
        feed, _ = _install_engine(tmp_path)
        mgr = _manager(tmp_path)
        conv = mgr.create_conversation(title="Chat")
        assert mgr.add_message(conv.id, "user", "salut") is not None
        assert mgr.add_message(conv.id, "assistant", "bonjour", model="m1") is not None
        rec = _latest(feed, conv.id)
        assert rec.clock == 3  # create=1, then one per message
        msgs = rec.payload["conversation"]["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        # Plaintext portability: the at-rest marker never rides the payload.
        assert msgs[0]["content"] == "salut"
        assert msgs[1]["content"] == "bonjour"
        assert all("ENCMARK:" not in m["content"] for m in msgs)
        # Local SQLite ids are device-local identities and are excluded.
        assert all("id" not in m for m in msgs)
        assert msgs[1]["model"] == "m1"

    def test_delete_last_message_publishes_reduced_state(self, tmp_path):
        feed, _ = _install_engine(tmp_path)
        mgr = _manager(tmp_path)
        conv = mgr.create_conversation()
        mgr.add_message(conv.id, "user", "q")
        mgr.add_message(conv.id, "assistant", "a")
        assert mgr.delete_last_message(conv.id, role="assistant") is True
        rec = _latest(feed, conv.id)
        assert rec.clock == 4
        assert rec.deleted is False
        msgs = rec.payload["conversation"]["messages"]
        assert [m["role"] for m in msgs] == ["user"]

    def test_delete_conversation_publishes_tombstone(self, tmp_path):
        feed, _ = _install_engine(tmp_path)
        mgr = _manager(tmp_path)
        conv = mgr.create_conversation()
        mgr.add_message(conv.id, "user", "q")
        assert mgr.delete_conversation(conv.id) is True
        rec = _latest(feed, conv.id)
        assert rec.deleted is True
        assert rec.payload == {}
        assert rec.clock == 3
        assert rec.updated_at != ""

    def test_rename_publishes_new_title(self, tmp_path):
        feed, _ = _install_engine(tmp_path)
        mgr = _manager(tmp_path)
        conv = mgr.create_conversation(title="Old")
        assert mgr.rename_conversation(conv.id, "New") is True
        rec = _latest(feed, conv.id)
        assert rec.clock == 2
        assert rec.payload["conversation"]["title"] == "New"

    def test_update_metadata_publishes_new_state(self, tmp_path):
        feed, _ = _install_engine(tmp_path)
        mgr = _manager(tmp_path)
        conv = mgr.create_conversation()
        assert mgr.update_conversation_metadata(conv.id, task_type="code") is True
        rec = _latest(feed, conv.id)
        assert rec.clock == 2
        assert rec.payload["conversation"]["task_type"] == "code"

    def test_clock_continues_past_a_remote_winner(self, tmp_path):
        # The interleaving end to end through the hook: a remote apply between
        # two local saves; the next local publish must out-clock the winner.
        feed, _ = _install_engine(tmp_path)
        mgr = _manager(tmp_path)
        conv = mgr.create_conversation(title="A")  # clock 1
        winner = _rec(conv.id, 7, device="dev-b", payload={"body": "remote"})
        protocol.apply_record_batch(feed, _batch("dev-b", 9, [winner]))
        assert mgr.add_message(conv.id, "user", "after") is not None
        rec = _latest(feed, conv.id)
        assert rec.clock == 8
        assert rec.device == "dev-a"

    def test_unhooked_reads_publish_nothing(self, tmp_path):
        feed, _ = _install_engine(tmp_path)
        mgr = _manager(tmp_path)
        conv = mgr.create_conversation()
        before = feed.count()
        mgr.get_conversation(conv.id)
        mgr.list_conversations()
        mgr.get_messages(conv.id)
        assert feed.count() == before


# --- hook contract: failure isolation, no-op, mode-free ----------------------


class _BrokenEngine(sync_engine.SyncEngine):
    def publish_conversation(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("journal down")


class TestHookContract:
    def test_publish_failure_never_breaks_the_save(self, tmp_path):
        f = _feed(tmp_path)
        sync_engine.set_sync_engine(_BrokenEngine(device="dev-a", feed=f))
        mgr = _manager(tmp_path)
        conv = mgr.create_conversation(title="Survives")
        assert conv is not None
        msg = mgr.add_message(conv.id, "user", "still saved")
        assert msg is not None
        assert [m.content for m in mgr.get_messages(conv.id)] == ["still saved"]
        assert mgr.delete_conversation(conv.id) is True
        assert f.count() == 0

    def test_snapshot_failure_never_breaks_the_save(self, tmp_path):
        feed, _ = _install_engine(tmp_path)
        mgr = _manager(tmp_path)
        conv = mgr.create_conversation()
        feed.clear()

        def _boom(conn, conv_id):
            raise RuntimeError("snapshot down")

        mgr._sync_snapshot = _boom  # type: ignore[method-assign]
        msg = mgr.add_message(conv.id, "user", "still saved")
        assert msg is not None
        assert [m.content for m in mgr.get_messages(conv.id)] == ["still saved"]
        assert feed.count() == 0

    def test_noop_when_veilid_unavailable(self, tmp_path):
        guard.veilid_available = lambda: False
        feed, _ = _install_engine(tmp_path)
        mgr = _manager(tmp_path)
        conv = mgr.create_conversation(title="Quiet")
        mgr.add_message(conv.id, "user", "no journal")
        mgr.rename_conversation(conv.id, "Still quiet")
        mgr.delete_conversation(conv.id)
        assert feed.count() == 0

    def test_publish_is_mode_free_and_works_in_bulbe(self, tmp_path):
        # Producing + journalling are local-disk operations permitted in ANY
        # mode (producers.py posture); only the wire is Daily-gated.
        set_mode("bulbe")
        feed, _ = _install_engine(tmp_path)
        mgr = _manager(tmp_path)
        conv = mgr.create_conversation(title="Bulbe local edit")
        rec = _latest(feed, conv.id)
        assert rec is not None
        assert rec.clock == 1
