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
  * OS6 -- a receipt's kind comes back from the store as it was saved, and
    the loaded digest names it.
  * OS7 -- the root covers the kinds: a kind moved in the file is refused by
    name on the way back.
  * OS8 -- what the queue adds comes back from the store as it was saved: a
    peel's rung, its stitched units and its residual, a held receipt's
    anchors.
  * OS9 -- the root covers the peels' marks: a residual moved in the file is
    refused by name on the way back.
  * OS10 -- the refusal marks come back from the store as they were saved.
  * OS11 -- a file written before schema versions is migrated once, by
    name: no receipt's line keeps a word of its span, the root is
    recomputed over the rows, and a second opening changes nothing.
  * OS12 -- a file of a newer schema is refused by name and left as it was.
  * OS13 -- a migration never launders a moved byte: a conversation whose
    root does not answer to its rows is left as it was and refused by name
    on its own load; the other conversations of the file migrate.
  * OS14 -- the schema is read and brought up inside one write transaction,
    so two processes opening an older file migrate it once.
  * OS15 -- a conversation that fails after its first line was written again
    is rolled back to its own savepoint: none of its lines change.
  * OS16 -- an error of the database itself during a migration rolls the
    whole file back and is told as it was, never masked by the rollback:
    nothing migrated, nothing marked, and the next opening migrates.
  * OS17-OS20 -- OS8, OS9, OS10 and OS12 over a state whose repaired peel
    references the user's words instead of copying them.

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


# ---------------------------------------------------------------------------
# OS6-OS7 -- the kinds of receipts on disk
# ---------------------------------------------------------------------------
def _kinded_state(loaded):
    """A state whose ledger holds one receipt of each kind, oldest first: accepted, held, bare."""
    lib = loaded["opti_oignon.memory.librarian"]
    state = lib.state_for("c1")
    state.mirror(_messages(6))
    for kind in ("accepted", "held", "bare"):
        state.flesh.evict_span(2, state.cellar, state.ledger, kind=kind)
    return state


def test_os6_a_receipt_kind_comes_back_from_the_store_as_it_was_saved(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state = _kinded_state(loaded)
        store = store_mod.OnionStore(tmp_path / "onion.db", connect=_plain, require_encryption=False)
        store.save("c1", state)
        back = store.load("c1", lib.OnionState())
        assert [r.kind for r in back.ledger.all()] == ["accepted", "held", "bare"]
        assert back.ledger.all() == state.ledger.all()
        assert back.ledger.digest(back.cellar) == state.ledger.digest(state.cellar)
    finally:
        restore()


def test_os7_a_kind_moved_in_the_file_is_refused_by_name(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state = _kinded_state(loaded)
        path = tmp_path / "onion.db"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        store.save("c1", state)
        conn = sqlite3.connect(str(path))
        moved = conn.execute("UPDATE onion_receipt_marks SET kind = 'bare' WHERE kind = 'held'").rowcount
        conn.commit()
        conn.close()
        assert moved == 1, "control: one kind moved in the file"
        with pytest.raises(store_mod.OnionIntegrityError, match="does not answer to its rows"):
            store.load("c1", lib.OnionState())
    finally:
        restore()


# ---------------------------------------------------------------------------
# OS8-OS9 -- what the queue adds to peels and receipts, on disk
# ---------------------------------------------------------------------------
_TYPED = [
    {"role": "user", "origin": "typed", "segments": [],
     "content": "Alice moved the build to Berlin on 2026-03-04. We keep Docker on the build server."},
    {"role": "assistant", "origin": "assistant", "segments": [],
     "content": "Noted: the Berlin build runs 12 jobs a day, a sensible load for that machine. "
                "Bob checks the logs every morning."},
]
_LOSSY = "Alice moved the build to Berlin on 2026-03-04. The Berlin build runs 12 jobs a day. Bob checks the logs every morning."


def _laddered_state(loaded):
    """A state with a repaired peel (stitched units), a held span (anchors) and an accepted peel with a residual."""
    from dataclasses import replace

    lib, peels = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.peels"]
    composer = loaded["opti_oignon.memory.composer"]
    tiny = composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=1, turn=60)
    gate = replace(peels.load_gate(), span_turns=2)
    state = lib.state_for("c1")
    state.mirror(_TYPED * 2)
    repaired = lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tiny,
                          ladder=replace(peels.load_ladder(), rho=1.0))
    held = lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tiny, ladder=replace(peels.load_ladder(), rho=0.5))
    quiet = [dict(m, content=m["content"].replace("agreed", "stated").replace("stays", "lives")) for m in _messages(2)]
    state.mirror(_TYPED * 2 + quiet)
    simple = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
    lossy = lambda turns: _faithful(turns).replace("Turn 2:", "Turn:").replace("service 2 ", "service ")  # noqa: E731
    accepted = lib.curate(state, lossy, gate=simple, budget=tiny)
    assert (repaired.rung, held.rung, accepted.rung) == ("repaired", "held", "accepted"), "control: one of each"
    assert repaired.peel.stitched and held.receipt.anchors and accepted.peel.residual, "control: each has its mark"
    return state


def test_os8_what_the_queue_adds_comes_back_from_the_store_as_it_was_saved(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state = _laddered_state(loaded)
        store = store_mod.OnionStore(tmp_path / "onion.db", connect=_plain, require_encryption=False)
        store.save("c1", state)
        back = store.load("c1", lib.OnionState())
        assert back.tree.all() == state.tree.all(), "rung, stitched units and residual, as made"
        assert back.ledger.all() == state.ledger.all(), "the held receipt's anchors too"
    finally:
        restore()


def test_os9_a_residual_moved_in_the_file_is_refused_by_name(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state = _laddered_state(loaded)
        path = tmp_path / "onion.db"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        store.save("c1", state)
        conn = sqlite3.connect(str(path))
        moved = conn.execute("UPDATE onion_peel_marks SET residual = '[]' WHERE residual != '[]'").rowcount
        conn.commit()
        conn.close()
        assert moved == 1, "control: one residual moved in the file"
        with pytest.raises(store_mod.OnionIntegrityError, match="does not answer to its rows"):
            store.load("c1", lib.OnionState())
    finally:
        restore()


def test_os10_the_refusal_marks_come_back_from_the_store_as_they_were_saved(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state = _laddered_state(loaded)
        state.refusals = {"a" * 64: "b" * 64, "c" * 64: "d" * 64}
        store = store_mod.OnionStore(tmp_path / "onion.db", connect=_plain, require_encryption=False)
        store.save("c1", state)
        back = store.load("c1", lib.OnionState())
        assert back.refusals == state.refusals
    finally:
        restore()


# ---------------------------------------------------------------------------
# OS11-OS13 -- the schema a file was written with
# ---------------------------------------------------------------------------

# The tables a build before schema versions did not have.
_ADDED_SINCE = ("onion_receipt_marks", "onion_peel_marks", "onion_proposals", "onion_refusals")


def _plain_state(loaded, cid):
    """A state as a build before the ladder left it: accepted receipts and peels only, no mark of any kind."""
    lib, peels = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.peels"]
    composer = loaded["opti_oignon.memory.composer"]
    tiny = composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=1, turn=60)
    gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
    state = lib.state_for(cid)
    state.mirror([dict(m, content=m["content"].replace("agreed", "stated").replace("stays", "lives"))
                  for m in _messages(4)])
    while lib.curate(state, _faithful, gate=gate, budget=tiny).evicted:
        pass
    assert len(state.tree.all()) == 2 and all(r.kind == "accepted" for r in state.ledger.all()), "control: plain"
    return state


def _pre_version_file(loaded, path, cids=("c1",), spans=None):
    """A file as a build before schema versions left it: six tables, version 0, each receipt's line its span's words.

    ``spans`` maps a conversation to {seq: span}: Cellar rows written in place
    of the saved ones, the conversation's root computed over them.
    """
    from dataclasses import replace

    store_mod = loaded["opti_oignon.memory.onion_store"]
    store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
    heads, rows = [], {}
    for cid in cids:
        store.save(cid, _plain_state(loaded, cid))
        snapshot, _root = store.load_snapshot(cid)
        assert not (snapshot.receipt_marks or snapshot.peel_marks), "control: nothing a later build adds"
        saved = dict(snapshot.cellar)
        old = []
        for key, _stub, ids, resolved in snapshot.receipts:
            head = " ".join(str(saved[key][0].get("text", "")).split())[:48]
            old.append((key, f"{key[:12]} turns {ids[0]}..{ids[-1]}: {head}", ids, resolved))
            heads.append(head)
        edits = (spans or {}).get(cid, {})
        cellar = tuple((key, edits.get(seq, span)) for seq, (key, span) in enumerate(snapshot.cellar))
        rows[cid] = (old, edits, store_mod.onion_root(replace(snapshot, receipts=tuple(old), cellar=cellar)))
    conn = sqlite3.connect(str(path))
    for table in _ADDED_SINCE:
        conn.execute(f"DROP TABLE {table}")
    for cid, (old, edits, root) in rows.items():
        for seq, span in edits.items():
            conn.execute("UPDATE onion_cellar SET span = ? WHERE conversation = ? AND seq = ?",
                         (store_mod._canonical(span), cid, seq))
        for seq, (_key, stub, _ids, _resolved) in enumerate(old):
            conn.execute("UPDATE onion_receipts SET stub = ? WHERE conversation = ? AND seq = ?", (stub, cid, seq))
        conn.execute("UPDATE onion_cursor SET root = ? WHERE conversation = ?", (root, cid))
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()
    return heads


def _tables(path):
    conn = sqlite3.connect(str(path))
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
    finally:
        conn.close()


def _version(path):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def test_os11_a_file_written_before_schema_versions_is_migrated_once_by_name(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        path = tmp_path / "onion.db"
        heads = _pre_version_file(loaded, path)
        assert heads and all(heads), "control: the old lines carry the first words of their spans"
        assert not set(_ADDED_SINCE) & _tables(path), "control: the file has the tables of its time, no other"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        assert set(_ADDED_SINCE) <= _tables(path), "the migration adds the tables of its schema"
        back = store.load("c1", lib.OnionState())
        lines = [r.stub for r in back.ledger.all()]
        assert not any(head in line for head in heads for line in lines), "no line keeps a word of its span"
        assert _version(path) == store_mod.SCHEMA_VERSION, "the file says the schema it is now of"
        before = path.read_bytes()
        store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        assert path.read_bytes() == before, "a second opening changes nothing"
    finally:
        restore()


def test_os12_a_file_of_a_newer_schema_is_refused_by_name_and_left_as_it_was(tmp_path):
    loaded, restore = _open()
    try:
        store_mod = loaded["opti_oignon.memory.onion_store"]
        path = tmp_path / "onion.db"
        store_mod.OnionStore(path, connect=_plain, require_encryption=False).save("c1", _laddered_state(loaded))
        conn = sqlite3.connect(str(path))
        conn.execute(f"PRAGMA user_version = {store_mod.SCHEMA_VERSION + 1}")
        conn.commit()
        conn.close()
        before = path.read_bytes()
        with pytest.raises(store_mod.OnionStoreError, match="newer than this build"):
            store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        assert path.read_bytes() == before
    finally:
        restore()


def test_os13_a_migration_never_launders_a_moved_byte(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        path = tmp_path / "onion.db"
        heads = _pre_version_file(loaded, path, cids=("c1", "c2", "c3"))
        conn = sqlite3.connect(str(path))
        conn.execute("UPDATE onion_peels SET text = text || ' moved' WHERE conversation = 'c2' AND seq = 0")
        conn.execute("UPDATE onion_cellar SET span = '{not json' WHERE conversation = 'c3' AND seq = 0")
        conn.commit()
        query = "SELECT stub FROM onion_receipts WHERE conversation = ? ORDER BY seq"
        moved = {cid: conn.execute(query, (cid,)).fetchall() for cid in ("c2", "c3")}
        conn.close()
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        back = store.load("c1", lib.OnionState())
        assert not any(head in r.stub for head in heads for r in back.ledger.all()), "the sound conversation migrates"
        for cid in ("c2", "c3"):
            with pytest.raises(store_mod.OnionIntegrityError, match=cid):
                store.load(cid, lib.OnionState())
        conn = sqlite3.connect(str(path))
        kept = {cid: conn.execute(query, (cid,)).fetchall() for cid in ("c2", "c3")}
        conn.close()
        assert kept == moved, "a moved byte or a row that does not decode is left as it was, never laundered"
        assert _version(path) == store_mod.SCHEMA_VERSION, "and the file is brought up for the others"
    finally:
        restore()


def test_os14_the_schema_is_read_and_brought_up_inside_one_write_transaction(tmp_path):
    loaded, restore = _open()
    try:
        store_mod = loaded["opti_oignon.memory.onion_store"]
        path = tmp_path / "onion.db"
        _pre_version_file(loaded, path)
        said = []

        class Recording:
            def __init__(self, conn):
                object.__setattr__(self, "_conn", conn)

            def execute(self, sql, *args):
                said.append(" ".join(sql.split())[:40])
                return self._conn.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self._conn, name)

            def __setattr__(self, name, value):
                said.append(f"set {name}={value!r}")
                setattr(self._conn, name, value)

        store_mod.OnionStore(path, connect=lambda p: Recording(sqlite3.connect(str(p))), require_encryption=False)
        assert "BEGIN IMMEDIATE" in said, "a write transaction is taken first"
        begin = said.index("BEGIN IMMEDIATE")
        assert "set isolation_level=None" in said[:begin], "with the driver's own transactions off, whichever it is"
        assert "COMMIT" in said[begin:], "and closed by name"
        read = next(i for i, sql in enumerate(said) if sql.startswith("PRAGMA user_version") and "=" not in sql)
        assert begin < read, "the version is read inside the transaction that brings the file up"
    finally:
        restore()


def test_os15_a_conversation_that_fails_after_its_first_rewrite_keeps_every_line(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        path = tmp_path / "onion.db"
        heads = _pre_version_file(loaded, path, cids=("c1", "c4"), spans={"c4": {1: [1]}})
        query = "SELECT stub FROM onion_receipts WHERE conversation = 'c4' ORDER BY seq"
        conn = sqlite3.connect(str(path))
        before = conn.execute(query).fetchall()
        conn.close()
        assert len(before) == 2, "control: its first line is written again before its second span fails"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        conn = sqlite3.connect(str(path))
        after = conn.execute(query).fetchall()
        conn.close()
        assert after == before, "the conversation that failed mid-way keeps every line, its first rewrite undone"
        back = store.load("c1", lib.OnionState())
        assert not any(head in r.stub for head in heads for r in back.ledger.all()), "control: the other migrated"
    finally:
        restore()


def test_os16_a_database_error_rolls_the_whole_migration_back_and_is_told_as_it_was(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        path = tmp_path / "onion.db"
        heads = _pre_version_file(loaded, path, cids=("c1", "c2"))

        class Failing:
            def __init__(self, conn):
                object.__setattr__(self, "_conn", conn)

            def execute(self, sql, *args):
                if sql.startswith("UPDATE onion_receipts") and args and args[0][1] == "c2":
                    raise sqlite3.OperationalError("injected: disk I/O error")
                if sql == "ROLLBACK":
                    self._conn.execute(sql)
                    raise sqlite3.OperationalError("cannot rollback - no transaction is active")
                return self._conn.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self._conn, name)

            def __setattr__(self, name, value):
                setattr(self._conn, name, value)

        before = path.read_bytes()
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            store_mod.OnionStore(path, connect=lambda p: Failing(sqlite3.connect(str(p))), require_encryption=False)
        assert path.read_bytes() == before and _version(path) == 0, "nothing migrated, nothing marked"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        back = store.load("c2", lib.OnionState())
        assert not any(head in r.stub for head in heads for r in back.ledger.all()), "the next opening migrates it"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OS17-OS20 -- os8, os9, os10 and os12 over a state whose repaired peel
# references the user's words: the queue copies no run into a peel any more,
# so the control their shared state held -- a repaired peel with stitched
# units -- is a repaired peel with references; every other assertion is the
# same.
# ---------------------------------------------------------------------------
def _referenced_state(loaded):
    """A state with a repaired peel (references), a held span (anchors) and an accepted peel with a residual."""
    from dataclasses import replace

    lib, peels = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.peels"]
    composer = loaded["opti_oignon.memory.composer"]
    tiny = composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=1, turn=60)
    gate = replace(peels.load_gate(), span_turns=2)
    state = lib.state_for("c1")
    state.mirror(_TYPED * 2)
    repaired = lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tiny,
                          ladder=replace(peels.load_ladder(), rho=1.0))
    held = lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tiny, ladder=replace(peels.load_ladder(), rho=0.5))
    quiet = [dict(m, content=m["content"].replace("agreed", "stated").replace("stays", "lives")) for m in _messages(2)]
    state.mirror(_TYPED * 2 + quiet)
    simple = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
    lossy = lambda turns: _faithful(turns).replace("Turn 2:", "Turn:").replace("service 2 ", "service ")  # noqa: E731
    accepted = lib.curate(state, lossy, gate=simple, budget=tiny)
    assert (repaired.rung, held.rung, accepted.rung) == ("repaired", "held", "accepted"), "control: one of each"
    assert repaired.peel.refs and held.receipt.anchors and accepted.peel.residual, "control: each has its mark"
    return state


def test_os17_what_the_queue_adds_comes_back_from_the_store_as_it_was_saved(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state = _referenced_state(loaded)
        store = store_mod.OnionStore(tmp_path / "onion.db", connect=_plain, require_encryption=False)
        store.save("c1", state)
        back = store.load("c1", lib.OnionState())
        assert back.tree.all() == state.tree.all(), "rung, references, dropped sentences and residual, as made"
        assert back.ledger.all() == state.ledger.all(), "the held receipt's anchors too"
    finally:
        restore()


def test_os18_a_residual_moved_in_the_file_is_refused_by_name(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state = _referenced_state(loaded)
        path = tmp_path / "onion.db"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        store.save("c1", state)
        conn = sqlite3.connect(str(path))
        moved = conn.execute("UPDATE onion_peel_marks SET residual = '[]' WHERE residual != '[]'").rowcount
        conn.commit()
        conn.close()
        assert moved == 1, "control: one residual moved in the file"
        with pytest.raises(store_mod.OnionIntegrityError, match="does not answer to its rows"):
            store.load("c1", lib.OnionState())
    finally:
        restore()


def test_os19_the_refusal_marks_come_back_from_the_store_as_they_were_saved(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state = _referenced_state(loaded)
        state.refusals = {"a" * 64: "b" * 64, "c" * 64: "d" * 64}
        store = store_mod.OnionStore(tmp_path / "onion.db", connect=_plain, require_encryption=False)
        store.save("c1", state)
        back = store.load("c1", lib.OnionState())
        assert back.refusals == state.refusals
    finally:
        restore()


def test_os20_a_file_of_a_newer_schema_is_refused_by_name_and_left_as_it_was(tmp_path):
    loaded, restore = _open()
    try:
        store_mod = loaded["opti_oignon.memory.onion_store"]
        path = tmp_path / "onion.db"
        store_mod.OnionStore(path, connect=_plain, require_encryption=False).save("c1", _referenced_state(loaded))
        conn = sqlite3.connect(str(path))
        conn.execute(f"PRAGMA user_version = {store_mod.SCHEMA_VERSION + 1}")
        conn.commit()
        conn.close()
        before = path.read_bytes()
        with pytest.raises(store_mod.OnionStoreError, match="newer than this build"):
            store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        assert path.read_bytes() == before
    finally:
        restore()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
