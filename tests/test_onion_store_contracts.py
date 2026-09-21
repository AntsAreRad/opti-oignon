#!/usr/bin/env python3
"""Contracts for the onion store: one conversation's memory on disk, proven
on the way back.

The four stores prove themselves in memory by re-hashing. The onion store
keeps that property across a process boundary: a load rebuilds every store
through the surface the librarian uses, re-hashes every row against the id
it was saved under, recomputes the root the state was saved under, and
refuses by name the first thing that no longer answers. A conversation the
file does not know is ``None``, never an empty state. The table holds
conversation text, so a plaintext connection is a refusal by name unless
the configuration allows it, and a store built where the connection seam
is unreachable refuses and creates no file.

  * OS1 -- a state round-trips byte for byte: Core with a supersession,
    Cellar spans, receipts with one resolved, Peels from the gate, the
    remaining Flesh, the mirror cursor and the root; the loaded state
    composes the same memory block.
  * OS2 -- a moved byte is refused by name on the way back: a Core entry, a
    Cellar span, a Peel text, and the saved root; an unknown conversation
    is ``None``, not an empty state; a refusal never leaves a partial
    state behind.
  * OS3 -- a plaintext connection is refused by name by default and leaves
    no file; allowed by configuration it is accepted; an encrypted
    connection is accepted; a connection that raises (a wrong key) is a
    refusal, not an empty state.
  * OS4 -- the store imports the standard library only at module scope,
    never names the plain client, and with the connection seam unreachable
    refuses and creates no file.
  * OS5 -- the root is a pure function of the four stores: equal on equal
    rows, different on a moved Core text, Cellar span, receipt flag or
    Peel, and unmoved by the Flesh and the cursor.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source; the connection seam is blocked, so a
connection can only come from an injected one -- the plain client on a
temporary path for the round trips, SQLCipher on a fabricated key for the
encrypted case.
"""

import ast
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian", "onion_store")
_STORE = source("memory", "onion_store.py")


def _open():
    loaded, restore = isolate(
        targets={f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _MODULES},
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils"),
        packages=("opti_oignon.memory",),
    )
    loaded["opti_oignon.memory.librarian"].reset_librarian()
    return loaded, restore


def _plain(path):
    return sqlite3.connect(str(path))


def _messages(n):
    out = []
    for i in range(1, n + 1):
        role = "user" if i % 2 else "assistant"
        out.append({"role": role, "content": f"Turn {i}: Alice reviewed service {i} on 2026-03-{i:02d} and we agreed that service {i} stays on the new cluster."})
    return out


def _faithful(turns):
    return " ".join(str(t.get("text", "")) for t in turns)


def _grown_state(loaded, cid="c1", turns=12):
    """A state with every layer populated: Core, Cellar, receipts, Peels, Flesh, cursor."""
    lib = loaded["opti_oignon.memory.librarian"]
    peels = loaded["opti_oignon.memory.peels"]
    core_store = loaded["opti_oignon.memory.core_store"]
    composer = loaded["opti_oignon.memory.composer"]
    gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
    budget = composer.Budget(window=2000, reserve=200, core=300, receipts=300, peels=800, flesh=200, turn=200)
    state = lib.state_for(cid)
    first = state.core.add("Answers are concise.", actor=core_store.USER)
    state.core.supersede(first, "Answers are concise and cite the source.", actor=core_store.USER)
    state.core.add("The user is Alice.", actor=core_store.USER)
    state.mirror(_messages(turns))
    while lib.curate(state, _faithful, gate=gate, budget=budget).evicted:
        pass
    assert len(state.tree.all()) >= 2, "control: peels were made"
    assert state.flesh.turns(), "control: some Flesh remains"
    receipts = state.ledger.all()
    assert len(receipts) >= 2, "control: receipts were left"
    state.ledger.resolve(receipts[0].key, state.cellar)
    return state, budget, gate


def _same(a, b):
    core_a = [(e.id, e.text, e.superseded_by) for e in a.core.all()]
    core_b = [(e.id, e.text, e.superseded_by) for e in b.core.all()]
    assert core_a == core_b, "the Core and its supersession links"
    assert a.cellar.keys() == b.cellar.keys() and all(a.cellar.get(k) == b.cellar.get(k) for k in a.cellar.keys())
    assert a.ledger.all() == b.ledger.all(), "receipts, order and resolved flags"
    assert a.tree.all() == b.tree.all(), "peels, every field"
    assert a.flesh.turns() == b.flesh.turns() and a.seen == b.seen


# ---------------------------------------------------------------------------
# OS1 -- round trip
# ---------------------------------------------------------------------------
def test_os1_a_state_round_trips_byte_for_byte_and_composes_the_same_block(tmp_path):
    loaded, restore = _open()
    try:
        lib = loaded["opti_oignon.memory.librarian"]
        store_mod = loaded["opti_oignon.memory.onion_store"]
        state, budget, gate = _grown_state(loaded)
        block_before = lib.memory_block("c1", "service reviewed by alice", budget=budget)
        assert block_before, "control: the grown state has a block"
        path = tmp_path / "onion.db"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        root = store.save("c1", state)
        assert isinstance(root, str) and len(root) == 64
        assert store.conversations() == ["c1"]

        again = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        fresh = again.load("c1", lib.OnionState())
        assert fresh is not None
        _same(state, fresh)
        assert store_mod.onion_root(store_mod.snapshot_of(fresh)) == root
        lib.reset_librarian()
        lib._states["c1"] = fresh
        assert lib.memory_block("c1", "service reviewed by alice", budget=budget) == block_before
        assert again.load("other", lib.OnionState()) is None, "a conversation the file does not know"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OS2 -- a moved byte is refused by name
# ---------------------------------------------------------------------------
def test_os2_a_moved_byte_is_refused_by_name_and_never_leaves_a_partial_state(tmp_path):
    loaded, restore = _open()
    try:
        lib = loaded["opti_oignon.memory.librarian"]
        store_mod = loaded["opti_oignon.memory.onion_store"]
        state, _budget, _gate = _grown_state(loaded)
        path = tmp_path / "onion.db"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        store.save("c1", state)
        core_id = state.core.all()[0].id
        span_key = state.cellar.keys()[0]
        peel_id = state.tree.all()[0].id

        def tamper(sql, *params):
            with sqlite3.connect(str(path)) as conn:
                conn.execute(sql, params)
                conn.commit()

        cases = [
            ("UPDATE onion_core SET text = 'moved' WHERE conversation = 'c1' AND id = ?", (core_id,), core_id),
            ("UPDATE onion_cellar SET span = '[{\"text\":\"moved\",\"turn_id\":\"t0001\",\"role\":\"user\"}]' WHERE conversation = 'c1' AND key = ?", (span_key,), span_key),
            ("UPDATE onion_peels SET text = 'moved' WHERE conversation = 'c1' AND id = ?", (peel_id,), peel_id),
        ]
        for sql, params, named in cases:
            store.save("c1", state)
            tamper(sql, *params)
            with pytest.raises(store_mod.OnionIntegrityError, match="refused, not repaired") as raised:
                store.load("c1", lib.OnionState())
            message = str(raised.value)
            assert "root" in message, "the root over the rows moves with any row, and is checked first"
        store.save("c1", state)
        tamper("UPDATE onion_cursor SET root = ? WHERE conversation = 'c1'", "0" * 64)
        with pytest.raises(store_mod.OnionIntegrityError, match="0{64}"):
            store.load("c1", lib.OnionState())

        # A row that still answers to the root but not to its own id: the
        # row-level re-hash is a second, independent layer.
        snapshot, _root = store.load_snapshot("c1")
        moved = store_mod.Snapshot(
            core=((core_id, "moved", snapshot.core[0][2]),) + snapshot.core[1:], cellar=snapshot.cellar,
            receipts=snapshot.receipts, peels=snapshot.peels, flesh=snapshot.flesh, seen=snapshot.seen,
        )
        tamper("UPDATE onion_core SET text = 'moved' WHERE conversation = 'c1' AND id = ?", core_id)
        tamper("UPDATE onion_cursor SET root = ? WHERE conversation = 'c1'", store_mod.onion_root(moved))
        target = lib.OnionState()
        with pytest.raises(store_mod.OnionIntegrityError, match=core_id):
            store.load("c1", target)
        assert target.core.all() == [] and target.cellar.keys() == [] and target.flesh.turns() == [], (
            "a refusal leaves nothing behind in the state it was asked to fill"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# OS3 -- plaintext is a refusal by name
# ---------------------------------------------------------------------------
def test_os3_plaintext_is_refused_by_name_unless_allowed_and_a_wrong_key_is_not_an_empty_state(tmp_path):
    loaded, restore = _open()
    try:
        store_mod = loaded["opti_oignon.memory.onion_store"]
        path = tmp_path / "plain.db"
        with pytest.raises(store_mod.PlaintextRefused, match="plaintext"):
            store_mod.OnionStore(path, connect=_plain)
        assert not path.exists() or path.stat().st_size == 0, "no plaintext table was written"
        allowed = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        assert allowed.conversations() == []

        sqlcipher3 = pytest.importorskip("sqlcipher3")
        encrypted = tmp_path / "onion.db"

        def keyed(p):
            conn = sqlcipher3.connect(str(p))
            conn.execute("PRAGMA key = 'contract-key'")
            return conn

        store = store_mod.OnionStore(encrypted, connect=keyed)
        state, _b, _g = _grown_state(loaded)
        store.save("c1", state)
        assert store_mod.OnionStore(encrypted, connect=keyed).conversations() == ["c1"]
        with pytest.raises(Exception):
            sqlite3.connect(str(encrypted)).execute("SELECT count(*) FROM onion_cursor")

        def wrong(p):
            raise RuntimeError("SQLCipher key verification failed")

        with pytest.raises(RuntimeError, match="key verification"):
            store_mod.OnionStore(encrypted, connect=wrong)
    finally:
        restore()


# ---------------------------------------------------------------------------
# OS4 -- module scope, the plain client, the unreachable seam
# ---------------------------------------------------------------------------
def test_os4_the_store_imports_stdlib_only_never_names_the_plain_client_and_refuses_without_the_seam(tmp_path):
    text = _STORE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0] if not node.level else "." + (node.module or ""))
    assert top <= {"hashlib", "json", "logging", "threading", "contextlib", "dataclasses", "pathlib"}, top
    assert "sqlite3" not in top, "the client is reached through the seam, never imported at module scope"
    assert "sqlite3.connect(" not in text, "the plain client is never called by the module"
    seam = {(n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert any(m and m.endswith("db_utils") and a == "safe_connect" for m, a in seam), "safe_connect is the default connection"

    loaded, restore = _open()
    try:
        store_mod = loaded["opti_oignon.memory.onion_store"]
        with pytest.raises(Exception):
            store_mod.OnionStore(tmp_path / "blocked.db")
        assert not (tmp_path / "blocked.db").exists(), "with the seam unreachable no file is created"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OS5 -- the root
# ---------------------------------------------------------------------------
def test_os5_the_root_is_a_pure_function_of_the_four_stores(tmp_path):
    loaded, restore = _open()
    try:
        store_mod = loaded["opti_oignon.memory.onion_store"]
        state, _b, _g = _grown_state(loaded)
        base = store_mod.snapshot_of(state)
        root = store_mod.onion_root(base)
        assert store_mod.onion_root(store_mod.snapshot_of(state)) == root, "equal rows, equal root"
        rep = lambda **over: store_mod.Snapshot(**{**base.__dict__, **over})  # noqa: E731
        moved_core = rep(core=((base.core[0][0], "moved", base.core[0][2]),) + base.core[1:])
        moved_span = rep(cellar=((base.cellar[0][0], [{"text": "moved"}]),) + base.cellar[1:])
        k, s, ids, resolved = base.receipts[0]
        moved_receipt = rep(receipts=((k, s, ids, not resolved),) + base.receipts[1:])
        moved_peel = rep(peels=((base.peels[0][0], "moved") + base.peels[0][2:],) + base.peels[1:])
        roots = {store_mod.onion_root(x) for x in (moved_core, moved_span, moved_receipt, moved_peel)}
        assert root not in roots and len(roots) == 4, "every store moves the root, each differently"
        assert store_mod.onion_root(rep(flesh=(), seen=0)) == root, "the Flesh and the cursor are not anchored"
    finally:
        restore()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
