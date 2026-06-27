#!/usr/bin/env python3
"""Tests for the canonical-memory materialization apply method (SYN-01 receive half).

``CanonicalMemoryStore.apply_synced_memory_canonical`` is the receiving half of a
sync round for canonical memory facts: a winning MEMORY_CANONICAL record is
written into the local store so the fact surfaces on this device. This suite
loads ``canonical_store.py`` in isolation (``get_encrypted_connection`` stubbed
to a plain sqlite3 connection, ``effective_user_id`` to the single-user default,
the publish hook stubbed) and proves the load-bearing invariant plus the basics:

  * a record materialises the fact row (create), with use_count seeded at 0;
  * an UPDATE to an existing fact preserves the device-local ``use_count`` -- a
    remote LWW win on the fact's content must never zero this device's usage
    telemetry (the analogue of the vault nonce-preservation invariant);
  * applying the same record twice is idempotent (one row, same state) -- the
    property that keeps the apply -> write loop from inflating;
  * a tombstone hard-deletes the fact by id (a converged deletion);
  * a malformed payload fails secure (returns False, raises nothing, writes
    nothing);
  * the apply is HOOK-FREE: it never re-publishes (no apply -> write -> publish
    echo), proven by a publish-hook spy that stays at zero across an apply.

The engine-side wiring (the round lander that calls this) is validated
separately (the canonical landing harness), not here. Local-only. Runs under
pytest or the __main__ runner.
"""

import importlib.util
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

DB_NAME = "memory_facts.db"


def _load(tmpdir: str):
    """Load canonical_store.py in isolation with a plain-sqlite backend.

    sys.modules is saved/restored so sibling suites stay clean. The guarded
    relative imports (``..db_encryption``, ``..user_isolation``) are satisfied
    with stubs so no SQLCipher / fastapi / ollama is required, and the publish
    hook is neutralised so setup never reaches the (absent) veilid framework.
    """
    keys = (
        "opti_oignon",
        "opti_oignon.memory",
        "opti_oignon.db_encryption",
        "opti_oignon.user_isolation",
        "opti_oignon.memory.canonical_store",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    mem_pkg = types.ModuleType("opti_oignon.memory")
    mem_pkg.__path__ = []
    sys.modules["opti_oignon.memory"] = mem_pkg

    enc = types.ModuleType("opti_oignon.db_encryption")
    enc.SQLCIPHER_AVAILABLE = False

    def _get_encrypted_connection(path, **kw):
        return sqlite3.connect(
            path, check_same_thread=kw.get("check_same_thread", False)
        )

    enc.get_encrypted_connection = _get_encrypted_connection
    sys.modules["opti_oignon.db_encryption"] = enc

    ui = types.ModuleType("opti_oignon.user_isolation")
    ui.DEFAULT_LOCAL_USER = "local"
    ui.effective_user_id = lambda user_id, single_user_mode=True: (
        "local" if (single_user_mode or user_id is None) else user_id
    )
    sys.modules["opti_oignon.user_isolation"] = ui

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.memory.canonical_store",
        _OO / "memory" / "canonical_store.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.memory.canonical_store"] = mod
    spec.loader.exec_module(mod)

    # The apply path is hook-free; the create/setup helpers (add) call the
    # publish hook, so stub it by default so setup never probes for veilid.
    mod._sync_publish_memory_fact = lambda *a, **k: None

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


def _wire_payload(
    fact_id="f1",
    text="the user prefers French",
    category="fact",
    source="conv-1",
    created_at="t0",
    updated_at="t1",
    active=True,
    user_id="local",
):
    """The full-state wire payload a producer emits: user_id hoisted, the nested
    fact is ``to_dict`` minus the device-local ``use_count``."""
    return {
        "user_id": user_id,
        "fact": {
            "id": fact_id,
            "text": text,
            "category": category,
            "source": source,
            "created_at": created_at,
            "updated_at": updated_at,
            "active": active,
        },
    }


def _db(tmpdir):
    return str(Path(tmpdir) / DB_NAME)


def _raw(dbpath):
    c = sqlite3.connect(dbpath)
    c.row_factory = sqlite3.Row
    return c


def _seed_raw(dbpath, *, fact_id, text, use_count, user_id="local"):
    """Insert a fact directly (bypassing add) so a known use_count is in place
    and the publish-hook spy stays untouched by setup."""
    c = sqlite3.connect(dbpath)
    try:
        c.execute(
            "INSERT INTO memory_facts "
            "(id, text, category, source, user_id, created_at, updated_at, active, use_count) "
            "VALUES (?, ?, 'fact', '', ?, 't0', 't0', 1, ?)",
            (fact_id, text, user_id, use_count),
        )
        c.commit()
    finally:
        c.close()


def test_apply_creates_fact():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.CanonicalMemoryStore(db_path=_db(td))
            assert store.apply_synced_memory_canonical("f1", _wire_payload()) is True

            raw = _raw(_db(td))
            try:
                row = raw.execute(
                    "SELECT * FROM memory_facts WHERE id = ?", ("f1",)
                ).fetchone()
                assert row is not None
                assert row["text"] == "the user prefers French"
                assert row["category"] == "fact"
                assert row["source"] == "conv-1"
                assert row["user_id"] == "local"
                assert row["active"] == 1
                assert row["use_count"] == 0  # seeded at 0 on create
            finally:
                raw.close()
        finally:
            restore()


def test_apply_update_preserves_use_count():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.CanonicalMemoryStore(db_path=_db(td))
            _seed_raw(_db(td), fact_id="f1", text="old text", use_count=7)

            ok = store.apply_synced_memory_canonical(
                "f1", _wire_payload(text="new text", updated_at="t9")
            )
            assert ok is True

            raw = _raw(_db(td))
            try:
                row = raw.execute(
                    "SELECT text, updated_at, use_count FROM memory_facts WHERE id = ?",
                    ("f1",),
                ).fetchone()
                assert row["text"] == "new text"        # content updated
                assert row["updated_at"] == "t9"
                assert row["use_count"] == 7             # device-local, preserved
            finally:
                raw.close()
        finally:
            restore()


def test_apply_tombstone_hard_deletes():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.CanonicalMemoryStore(db_path=_db(td))
            _seed_raw(_db(td), fact_id="f1", text="doomed", use_count=3)

            ok = store.apply_synced_memory_canonical(
                "f1", _wire_payload(), deleted=True
            )
            assert ok is True

            raw = _raw(_db(td))
            try:
                row = raw.execute(
                    "SELECT * FROM memory_facts WHERE id = ?", ("f1",)
                ).fetchone()
                assert row is None
            finally:
                raw.close()
        finally:
            restore()


def test_apply_idempotent():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.CanonicalMemoryStore(db_path=_db(td))
            p = _wire_payload(text="stable")
            assert store.apply_synced_memory_canonical("f1", p) is True
            assert store.apply_synced_memory_canonical("f1", p) is True

            raw = _raw(_db(td))
            try:
                rows = raw.execute(
                    "SELECT * FROM memory_facts WHERE id = ?", ("f1",)
                ).fetchall()
                assert len(rows) == 1
                assert rows[0]["text"] == "stable"
            finally:
                raw.close()
        finally:
            restore()


def test_apply_malformed_returns_false():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.CanonicalMemoryStore(db_path=_db(td))
            # not a dict
            assert store.apply_synced_memory_canonical("f1", "nope") is False
            # missing the nested fact
            assert store.apply_synced_memory_canonical("f1", {"user_id": "local"}) is False
            # fact missing id
            assert (
                store.apply_synced_memory_canonical(
                    "f1", {"user_id": "local", "fact": {"text": "x"}}
                )
                is False
            )
            # text is not a string
            bad = _wire_payload()
            bad["fact"]["text"] = 123
            assert store.apply_synced_memory_canonical("f1", bad) is False
            # nested id does not match the record key (integrity)
            mismatch = _wire_payload(fact_id="other")
            assert store.apply_synced_memory_canonical("f1", mismatch) is False

            raw = _raw(_db(td))
            try:
                rows = raw.execute("SELECT * FROM memory_facts").fetchall()
                assert rows == []  # nothing written on any malformed apply
            finally:
                raw.close()
        finally:
            restore()


def test_apply_is_hook_free():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.CanonicalMemoryStore(db_path=_db(td))
            calls = []
            mod._sync_publish_memory_fact = lambda *a, **k: calls.append((a, k))
            # seed via raw SQL so setup leaves the spy untouched
            _seed_raw(_db(td), fact_id="f1", text="orig", use_count=0)

            assert (
                store.apply_synced_memory_canonical("f1", _wire_payload(text="upd"))
                is True
            )
            assert store.apply_synced_memory_canonical("f1", _wire_payload(), deleted=True) is True
            assert calls == []  # apply never re-publishes
        finally:
            restore()


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
