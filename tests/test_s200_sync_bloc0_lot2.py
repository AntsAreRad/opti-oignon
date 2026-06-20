#!/usr/bin/env python3
"""S200 -- sync cycle Bloc 0 lot 2: the memory producer (canonical tier).

One tight test group per fix (SYN-01, ROADMAP_SYNC_CYCLE Bloc 0, the lot-1
precedents applied to ``memory/canonical_store.py``):

- Clock discipline through the hook: an unseen fact key mints clock 1; clocks
  stay strictly monotonic over local writes AND past a journalled remote
  winner (PRT-02 journals adoptions, so ``current_clock`` reflects them); the
  two memory kinds (``MEMORY_CANONICAL`` / ``MEMORY_ARCHIVE``) never collide
  on the same identity.

- Payload shape: full state per fact -- ``user_id`` hoisted to the top level
  (the lot-1 scoping rule: the bare fact id is the per-kind key, ownership
  rides in the payload; memory has REAL per-user scoping so the actual uid is
  carried), the nested fact is ``to_dict`` minus ``use_count`` (device-local
  telemetry, does not merge under LWW) and minus the hoisted ``user_id``. A
  soft delete publishes STATE (the ``active`` flag; restore round-trips);
  only ``hard_delete`` is a tombstone; ``clear`` (the UD-03 user-wipe path)
  publishes per-fact tombstones, the ids read BEFORE the DELETE behind the
  ``_sync_wanted`` probe.

- Silence where arbitrated: ``touch`` (the retrieval hot path bumps counters
  on every surfaced fact) publishes nothing, so the facade's dedup merge
  (add converging to touch) publishes nothing either; reads publish nothing.

- Hook contract: the domain commit happens first and a publish or payload
  failure never breaks the write; no-op when the veilid framework is absent
  (and pays nothing then -- the payload re-read and clear's pre-delete id
  read are both behind the guard); mode-free (publishing works under Bulbe,
  only the wire is Daily-gated).

- The mandatory lot-2 identity check, documented as a test: ``fact_id`` is
  globally unique across users at the schema level (PRIMARY KEY on ``id``
  alone), so the bare per-kind key cannot collide cross-user.

Loader idiom: lot-1's (spec_from_file_location, sys.modules registration
BEFORE exec_module, package stubs). canonical_store's guarded deps are
stubbed: db_encryption (plain sqlite3), user_isolation, config. The veilid
modules are the real ones over tmp feeds; the engine singleton is injected
per test via ``set_sync_engine``.
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
MEMORY = OO / "memory"

_MODE = {"fn": (lambda: "daily")}


def set_mode(value: str = "daily") -> None:
    def _gm() -> str:
        return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


_SESSION_DATA_DIR = Path(tempfile.mkdtemp(prefix="oo_s200_data_"))


def _ensure_stubs() -> None:
    for name, sub in (
        ("opti_oignon", OO),
        ("opti_oignon.veilid", VEILID),
        ("opti_oignon.memory", MEMORY),
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
    if "opti_oignon.config" not in sys.modules:
        cfg = types.ModuleType("opti_oignon.config")
        cfg.DATA_DIR = _SESSION_DATA_DIR  # type: ignore[attr-defined]
        sys.modules["opti_oignon.config"] = cfg
    if "opti_oignon.db_utils" not in sys.modules:
        dbu = types.ModuleType("opti_oignon.db_utils")
        dbu.safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)  # type: ignore[attr-defined]
        sys.modules["opti_oignon.db_utils"] = dbu
    if "opti_oignon.db_encryption" not in sys.modules:
        # canonical_store's guarded at-rest layer: a plain-SQLite stand-in so
        # the module loads in isolation without sqlcipher.
        dbe = types.ModuleType("opti_oignon.db_encryption")
        dbe.SQLCIPHER_AVAILABLE = False  # type: ignore[attr-defined]

        def _gec(db_path, *, check_same_thread=True, timeout=5.0, enforce_encryption=None):
            return sqlite3.connect(
                str(db_path), check_same_thread=check_same_thread, timeout=timeout
            )

        dbe.get_encrypted_connection = _gec  # type: ignore[attr-defined]
        sys.modules["opti_oignon.db_encryption"] = dbe
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
canonical_store = _load("canonical_store", base=MEMORY, package="opti_oignon.memory")
dedup = _load("dedup", base=MEMORY, package="opti_oignon.memory")
RecordKind = records.RecordKind

_REAL_VEILID_AVAILABLE = guard.veilid_available


@pytest.fixture(autouse=True)
def _daily_reset_and_available():
    set_mode("daily")
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    canonical_store.reset_canonical_store()
    # The container has no veilid framework; the hook gates on this probe, so
    # force it on by default and let the no-op tests force it off.
    guard.veilid_available = lambda: True
    yield
    guard.veilid_available = _REAL_VEILID_AVAILABLE
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    canonical_store.reset_canonical_store()
    set_mode("daily")


def _install_engine(tmp_path, device="dev-a"):
    f = change_feed.ChangeFeed(root=tmp_path / "feed")
    eng = sync_engine.SyncEngine(device=device, feed=f)
    sync_engine.set_sync_engine(eng)
    return f, eng


def _store(tmp_path, *, single_user_mode=True):
    return canonical_store.CanonicalMemoryStore(
        db_path=tmp_path / "memory_facts.db", single_user_mode=single_user_mode
    )


def _rows(feed, record_id=None, kind=None):
    out = []
    for r in feed.current_records():
        if record_id is not None and r.record_id != record_id:
            continue
        if kind is not None and str(getattr(r.kind, "value", r.kind)) != str(
            getattr(kind, "value", kind)
        ):
            continue
        out.append(r)
    return out


def _latest(feed, record_id, kind=RecordKind.MEMORY_CANONICAL):
    matches = _rows(feed, record_id=record_id, kind=kind)
    if not matches:
        return None
    return max(matches, key=lambda r: r.clock)


def _remote(record_id, clock, *, kind=RecordKind.MEMORY_CANONICAL, deleted=False):
    return records.new_record(
        kind=kind,
        record_id=record_id,
        payload={} if deleted else {"v": clock},
        device="remote",
        clock=clock,
        deleted=deleted,
    )


# --- Clock discipline through the hook (SYN-01) ------------------------------


class TestClockDiscipline:
    def test_first_publish_mints_clock_one(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        s = _store(tmp_path)
        rec = s.add("first fact", "fact")
        latest = _latest(f, rec.id)
        assert latest is not None
        assert latest.clock == 1

    def test_clock_monotonic_over_local_writes(self, tmp_path):
        # The feed's read view collapses to latest-per-key (state-based LWW),
        # so monotonicity is asserted as the latest clock stepping by exactly
        # one after each successive write.
        f, _ = _install_engine(tmp_path)
        s = _store(tmp_path)
        rec = s.add("evolving fact", "fact")
        assert _latest(f, rec.id).clock == 1
        s.update(rec.id, text="evolving fact, edited")
        assert _latest(f, rec.id).clock == 2
        s.soft_delete(rec.id)
        assert _latest(f, rec.id).clock == 3
        s.restore(rec.id)
        assert _latest(f, rec.id).clock == 4

    def test_clock_continues_past_a_remote_winner(self, tmp_path):
        # apply_record_batch journals winners (PRT-02 included), so the feed
        # is the merged latest view; a journalled remote row stands in for an
        # applied winner here, and the next local mint must out-clock it.
        f, _ = _install_engine(tmp_path)
        s = _store(tmp_path)
        rec = s.add("contested fact", "fact")
        f.record(_remote(rec.id, 7))
        updated = s.update(rec.id, text="local edit after remote winner")
        assert updated is not None
        assert _latest(f, rec.id).clock == 8

    def test_two_memory_kinds_never_collide_on_identity(self, tmp_path):
        f, eng = _install_engine(tmp_path)
        s = _store(tmp_path)
        s.add("shared identity fact", "fact", fact_id="shared-id")
        next_archive = eng.current_clock(RecordKind.MEMORY_ARCHIVE, "shared-id") + 1
        eng.publish_memory_archive(
            "shared-id", {"entry": "archive tier"}, clock=next_archive
        )
        assert eng.current_clock(RecordKind.MEMORY_CANONICAL, "shared-id") == 1
        assert eng.current_clock(RecordKind.MEMORY_ARCHIVE, "shared-id") == 1
        assert len(_rows(f, record_id="shared-id", kind=RecordKind.MEMORY_CANONICAL)) == 1
        assert len(_rows(f, record_id="shared-id", kind=RecordKind.MEMORY_ARCHIVE)) == 1


# --- Payload shape (full state, scoping in the payload) ----------------------


class TestPayloadShape:
    def test_user_id_rides_the_payload(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        s = _store(tmp_path, single_user_mode=False)
        rec = s.add("alice's fact", "fact", user_id="alice")
        payload = _latest(f, rec.id).payload
        assert payload["user_id"] == "alice"
        assert "user_id" not in payload["fact"]

    def test_full_state_without_device_local_telemetry(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        s = _store(tmp_path)
        rec = s.add("plain fact", "preference", source="manual")
        fact = _latest(f, rec.id).payload["fact"]
        assert set(fact.keys()) == {
            "id", "text", "category", "source", "created_at", "updated_at", "active",
        }
        assert "use_count" not in fact
        assert fact["id"] == rec.id
        assert fact["text"] == "plain fact"
        assert fact["category"] == "preference"
        assert fact["source"] == "manual"
        assert fact["active"] is True

    def test_soft_delete_restore_round_trip_as_state(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        s = _store(tmp_path)
        rec = s.add("flagged fact", "fact")
        assert s.soft_delete(rec.id) is True
        after_soft = _latest(f, rec.id)
        assert after_soft.deleted is False  # state, not a tombstone
        assert after_soft.payload["fact"]["active"] is False
        assert s.restore(rec.id) is True
        after_restore = _latest(f, rec.id)
        assert after_restore.deleted is False
        assert after_restore.payload["fact"]["active"] is True
        assert after_restore.clock > after_soft.clock

    def test_hard_delete_publishes_tombstone(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        s = _store(tmp_path)
        rec = s.add("doomed fact", "fact")
        assert s.hard_delete(rec.id) is True
        latest = _latest(f, rec.id)
        assert latest.deleted is True
        assert latest.payload == {}
        assert latest.clock == 2

    def test_clear_publishes_per_fact_tombstones(self, tmp_path):
        # clear is the UD-03 user-wipe path: the wipe must converge on peers.
        f, _ = _install_engine(tmp_path)
        s = _store(tmp_path)
        ids = [s.add(f"fact {i}", "fact").id for i in range(3)]
        assert s.clear() == 3
        assert s.count(active_only=False) == 0
        for fid in ids:
            latest = _latest(f, fid)
            assert latest.deleted is True
            assert latest.payload == {}
            assert latest.clock == 2  # one add (1) then the tombstone (2)


# --- Silence where arbitrated (touch, merge, reads) ---------------------------


class _SilentVector:
    """A vector layer that never embeds, so dedup stays on the Jaccard stage."""

    def embed(self, text):
        return None

    def find_similar(self, *a, **kw):
        return []

    def add(self, *a, **kw):
        return None

    def update(self, *a, **kw):
        return None

    def delete(self, *a, **kw):
        return None


class TestArbitratedSilence:
    def test_touch_publishes_nothing(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        s = _store(tmp_path)
        rec = s.add("touched fact", "fact")
        assert s.touch(rec.id) is True
        # Under the latest-per-key view a publish would bump the clock; the
        # latest record must still be the add at clock 1.
        assert _latest(f, rec.id).clock == 1
        # The counter moved locally, invisibly to the journal.
        assert s.get(rec.id).use_count == 1

    def test_facade_dedup_merge_publishes_nothing(self, tmp_path):
        # The coordinated facade converges a duplicate add into a touch on
        # the existing fact: no semantic change, so nothing is journalled.
        f, _ = _install_engine(tmp_path)
        s = _store(tmp_path)
        facade = dedup.MemoryStore(s, _SilentVector())
        first, decision_a = facade.add("the user prefers dark mode", "preference")
        assert decision_a.action == "insert"
        second, decision_b = facade.add("the user prefers dark mode", "preference")
        assert decision_b.action == "merge"
        assert second.id == first.id
        assert _latest(f, first.id).clock == 1
        assert len(_rows(f)) == 1

    def test_reads_publish_nothing(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        s = _store(tmp_path)
        rec = s.add("read-only fact", "fact")
        s.get(rec.id)
        s.list()
        s.count()
        assert _latest(f, rec.id).clock == 1
        assert len(_rows(f)) == 1


# --- Hook failure isolation and the absent/mode-free postures ----------------


class TestHookContract:
    def test_publish_failure_never_breaks_the_write(self, tmp_path):
        f, eng = _install_engine(tmp_path)

        def _boom(*a, **kw):
            raise RuntimeError("journal append failed (test)")

        eng.publish_memory_canonical = _boom  # type: ignore[assignment]
        s = _store(tmp_path)
        rec = s.add("resilient fact", "fact")
        assert s.get(rec.id) is not None  # the domain write held
        assert _rows(f) == []

    def test_payload_failure_never_breaks_the_write(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        s = _store(tmp_path)
        rec = s.add("fragile snapshot fact", "fact")
        original_get = s.get

        def _boom(*a, **kw):
            raise RuntimeError("snapshot read failed (test)")

        s.get = _boom  # type: ignore[assignment]
        try:
            assert s.soft_delete(rec.id) is True  # the write held
        finally:
            s.get = original_get  # type: ignore[assignment]
        assert s.get(rec.id, user_id=None).active is False
        # Only the add made it to the journal; the failed state publish is
        # at-least-once territory for the next write.
        assert [r.clock for r in _rows(f, record_id=rec.id)] == [1]

    def test_noop_when_veilid_unavailable_pays_nothing(self, tmp_path):
        f, _ = _install_engine(tmp_path)
        guard.veilid_available = lambda: False
        s = _store(tmp_path)
        rec = s.add("offline fact", "fact")
        original_get = s.get

        def _boom(*a, **kw):
            raise AssertionError("payload re-read ran while sync was absent")

        s.get = _boom  # type: ignore[assignment]
        try:
            assert s.soft_delete(rec.id) is True
            assert s.restore(rec.id) is True
        finally:
            s.get = original_get  # type: ignore[assignment]
        assert canonical_store._sync_wanted() is False  # clear's pre-read probe
        assert s.clear() == 1
        assert s.count(active_only=False) == 0
        assert _rows(f) == []

    def test_publish_is_mode_free_and_works_in_bulbe(self, tmp_path):
        # Producing + journalling are local-disk operations, permitted in any
        # mode (the producers.py posture); only the wire is Daily-gated.
        f, _ = _install_engine(tmp_path)
        set_mode("bulbe")
        s = _store(tmp_path)
        rec = s.add("bulbe local fact", "fact")
        latest = _latest(f, rec.id)
        assert latest is not None
        assert latest.clock == 1


# --- The mandatory lot-2 identity check ---------------------------------------


class TestIdentityRule:
    def test_fact_id_globally_unique_across_users(self, tmp_path):
        # PRIMARY KEY on id alone (not (id, user_id)): a cross-user collision
        # on the bare per-kind key is impossible at the schema level, so the
        # lot-1 bare-key rule holds for memory.
        _install_engine(tmp_path)
        s = _store(tmp_path, single_user_mode=False)
        s.add("alice's fact", "fact", user_id="alice", fact_id="collide-id")
        with pytest.raises(sqlite3.IntegrityError):
            s.add("bob's fact", "fact", user_id="bob", fact_id="collide-id")
