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
  * OS21 -- what the librarian read of each standing span comes back from
    the store as it was saved, a span that states nothing included, a
    pronoun's subject known from an earlier span too; the readings are in
    the root, and a state with none keeps the root it had; what was read of
    a span the mirror superseded since is not saved.
  * OS22 -- a reading moved in the file is refused by name; with the root
    made to answer, so is a reading of a span the file does not hold
    standing, a span read twice and a statement that does not decode; a
    reading the checks no longer keep exactly as it was -- a value its
    sentence does not hold, a value kept with its last punctuation, a
    statement twice -- is left out with the readings that knew a subject
    only from it, said, never refused.
  * OS23 -- a file of schema 2 is brought up by the migration of the
    readings, by name: the table added, every root unchanged, nothing read.
  * OS24 -- readings the saved root does not cover, where the rest answers
    to it -- a build that knew none saved the state over a file that held
    some -- are left out, and said, and the next save holds none of them;
    the reader of a reading is in the root, a digest or none: one moved is
    refused by name, and so is one that is no digest.
  * OS25 -- THE ONION CARRIES ANY TEXT: a surrogate, which UTF-8 cannot
    carry -- half of a pair, or a pair split in two -- in a message, in a
    summary the model wrote or in a pin is kept as the replacement
    character, one character for one, so every place in the text stays;
    the conversation seen again is the same; the state evicts, saves and
    loads; a label or a lineage that holds one is read outside the grammar,
    never raised, and a role that holds one is kept as words are; a text
    that holds none is kept as it was; a pin that is no string is
    refused by name.
  * OS26 -- EVERY WAY A SUMMARY BECOMES A PEEL KEEPS IT SO: a second
    asking, a leaf, a gated eviction and a parent the model summarised with
    a surrogate keep it as the replacement character; ``storable`` replaces
    every surrogate, the halves a byte-escaping decoder leaves among them,
    and nothing else: the characters on either side of the range, a NUL,
    marks not composed, blanks at either end, a long text, all as they were.
  * OS27 -- A SAVE OVER WHAT WAS READ: each save writes the conversation's
    next generation, beside the root; a save that names the generation it
    read writes only over it and leaves the new one on the state, and when
    another writer saved since it is refused by name and writes nothing; a
    state read as absent is refused once the file holds the conversation,
    and so is one ahead of the file (a copy restored from before); a save
    that names none writes as before; a generation that is no whole number
    from 0 is refused by name.
  * OS28 -- ONE TRANSACTION EACH WAY: a save reads the generation inside
    the write transaction it takes first, before any row goes; a load
    reads the rows and their generation inside one read transaction.
  * OS29 -- a file of schema 3 is brought up by the generations'
    migration, by name: the column added, each conversation at generation
    1, every root unchanged; a second opening changes nothing.
  * OS30 -- TWO PROCESSES, ONE FILE: a state whose generation the file
    still holds is used as it is, its rows not read again; one another
    process saved since is never written over -- the save refused by name,
    nothing written, said -- and is read again at its next use, by the
    turn's way and the block's, what this process alone kept going with
    it, its two locks first; a conversation the file no longer holds starts
    over, and a state this process no longer holds is never written,
    whatever its generation; a store that cannot be read at a use is said
    once for each kind of refusal and raised; on two threads of one
    process, a use waits for the save the other is making and never reads
    it as another process's; of two readings again, the first stands; with
    no store nothing is read, and a configuration that does not read leaves
    the state as held, by either way to it; a refused window names both
    stores that can refuse it.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source; the connection seam is blocked, so a
connection can only come from an injected one -- the plain client on a
temporary path for the round trips, SQLCipher on a fabricated key for the
encrypted case.
"""

import ast
import json
import logging
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_journal as oj  # noqa: E402
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


# ---------------------------------------------------------------------------
# OS21-OS23 -- what the librarian read of each span, on disk
# ---------------------------------------------------------------------------
_FACTS = ("Bob lives in Lyon.", "Alice lives in Berlin.", "She lives in Oslo now.")
# What the librarian reads of each span, by the span's first turn: the third's subject is a pronoun's, known from the
# second.
_LINES = {"t0001": "t0001 | Bob | lives in | Lyon", "t0003": "t0003 | Alice | lives in | Berlin",
          "t0005": "t0005 | Alice | lives in | Oslo now."}


def _read_state(loaded, *, read=True):
    """A state of four spans under accepted peels -- three the user typed a fact in, then a document's alone -- and
    the root it has before any is read; each read by the librarian unless ``read`` is false."""
    lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
    messages = []
    for message in [oj.typed(words) for words in _FACTS] + [oj.with_document("", "Carol works at Initech.")]:
        messages += [message, oj.answer("Noted.")]
    messages += oj.conversation(4, start=len(messages) + 1)
    state = lib.state_for("c1")
    state.mirror(messages)
    for n in range(4):
        oj.accept(loaded, state, 2, f"The user stated fact {n + 1}.")
    root = store_mod.onion_root(store_mod.snapshot_of(state))
    if read:
        reading = lambda turns, known=(): _LINES.get(str(turns[0]["turn_id"]), "NONE")  # noqa: E731
        assert lib.read_spans(state, reading, limit=10) == 3, "control: three spans read, the document's not asked"
    return state, messages, root


def _rewritten(store_mod, path, statements):
    """The readings of c1 in the file written as ``statements``, the root made to answer to them."""
    from dataclasses import replace

    conn = sqlite3.connect(str(path))
    conn.execute("DELETE FROM onion_statements WHERE conversation = 'c1'")
    conn.commit()
    conn.close()
    store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
    snapshot, _root = store.load_snapshot("c1")
    root = store_mod.onion_root(replace(snapshot, statements=tuple(statements)))
    conn = sqlite3.connect(str(path))
    conn.executemany("INSERT INTO onion_statements (conversation, seq, span_key, reader, statements) "
                     "VALUES ('c1', ?, ?, ?, ?)",
                     [(i, key, reader, json.dumps([list(row) for row in found]))
                      for i, (key, reader, found) in enumerate(statements)])
    conn.execute("UPDATE onion_cursor SET root = ? WHERE conversation = 'c1'", (root,))
    conn.commit()
    conn.close()
    return store


def test_os21_what_the_librarian_read_of_the_standing_spans_comes_back_as_saved_and_is_in_the_root(tmp_path):
    from dataclasses import replace

    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state, messages, unread = _read_state(loaded)
        keys = [r.key for r in state.ledger.all()]
        assert [[(s.subject, s.value) for s in state.statements[key]] for key in keys] == [
            [("Bob", "Lyon")], [("Alice", "Berlin")], [("Alice", "Oslo now")], []], "control: what was read"
        snapshot = store_mod.snapshot_of(state)
        assert [key for key, _reader, _found in snapshot.statements] == keys, "each span read, in the ledger's order"
        root = store_mod.onion_root(snapshot)
        assert root != unread, "the readings are in the root"
        assert store_mod.onion_root(replace(snapshot, statements=())) == unread, "a state with none keeps its root"
        path = tmp_path / "onion.db"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        assert store.save("c1", state) == root
        back = store_mod.OnionStore(path, connect=_plain, require_encryption=False).load("c1", lib.OnionState())
        assert back.statements == state.statements, "every reading, the pronoun's and a span that states nothing"
        assert store_mod.onion_root(store_mod.snapshot_of(back)) == root
        # A span the mirror superseded since: what was read of it is not saved.
        edited = list(messages)
        edited[4] = oj.typed("Alice is staying in Berlin.")
        state.mirror(edited)
        assert keys[2] in state.statements and lib._superseded(state, keys[2]), "control: superseded, still read"
        store.save("c1", state)
        conn = sqlite3.connect(str(path))
        saved = [row[0] for row in conn.execute("SELECT span_key FROM onion_statements ORDER BY seq").fetchall()]
        conn.close()
        assert saved == keys[:2], "the standing spans' readings alone"
        assert store.load("c1", lib.OnionState()).statements == {key: state.statements[key] for key in keys[:2]}
    finally:
        restore()


def test_os22_a_reading_moved_is_refused_and_one_the_checks_no_longer_keep_is_left_out_and_said(tmp_path, caplog):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state, messages, _unread = _read_state(loaded)
        keys = [r.key for r in state.ledger.all()]
        path = tmp_path / "onion.db"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        store.save("c1", state)
        read = list(store.load_snapshot("c1")[0].statements)
        (bob,), (alice,), (oslo,) = read[0][2], read[1][2], read[2][2]
        conn = sqlite3.connect(str(path))
        moved = conn.execute("UPDATE onion_statements SET statements = replace(statements, 'Bob', 'Rob') "
                             "WHERE seq = 0").rowcount
        conn.commit()
        conn.close()
        assert moved == 1, "control: one reading moved in the file"
        with pytest.raises(store_mod.OnionIntegrityError, match="does not answer to its rows"):
            store.load("c1", lib.OnionState())
        # The root made to answer: a span read twice, a span the file does not hold, a row that does not decode.
        _rewritten(store_mod, path, read + [read[0]])
        with pytest.raises(store_mod.OnionIntegrityError, match="read twice"):
            store.load("c1", lib.OnionState())
        _rewritten(store_mod, path, read + [("f" * 64, "", ())])
        with pytest.raises(store_mod.OnionIntegrityError, match="does not hold standing"):
            store.load("c1", lib.OnionState())
        _rewritten(store_mod, path, [(keys[0], read[0][1], ((bob[0], str(bob[1]), *bob[2:]),))] + read[1:])
        with pytest.raises(store_mod.OnionIntegrityError, match="does not decode"):
            store.load("c1", lib.OnionState())
        # A reading the checks no longer keep exactly as it was: left out, with what knew a subject only from it.
        caplog.set_level(logging.WARNING)
        cases = (
            ("a value its sentence does not hold", (*alice[:5], "Paris")),
            ("a value kept with its last punctuation", (*alice[:5], "Berlin.")),
        )
        for why, moved_row in cases:
            caplog.clear()
            _rewritten(store_mod, path, [read[0], (keys[1], read[1][1], (moved_row,))] + read[2:])
            back = store.load("c1", lib.OnionState())
            assert back.statements == {keys[0]: state.statements[keys[0]], keys[3]: ()}, (
                why, "the pronoun's reading knew Alice from it alone")
            assert any("2 span reading(s) no longer borne out" in r.getMessage() for r in caplog.records), why
        caplog.clear()
        _rewritten(store_mod, path, [(keys[0], read[0][1], (bob, bob))] + read[1:])
        back = store.load("c1", lib.OnionState())
        assert back.statements == {key: state.statements[key] for key in keys[1:]}, "a statement twice"
        assert any("1 span reading(s) no longer borne out" in r.getMessage() for r in caplog.records)
        assert oslo[3] == "Alice", "control: the pronoun's subject"
        # A reading of a span the file holds, superseded: not standing, refused.
        edited = list(messages)
        edited[4] = oj.typed("Alice is staying in Berlin.")
        state.mirror(edited)
        store.save("c1", state)
        _rewritten(store_mod, path, read[:3])
        with pytest.raises(store_mod.OnionIntegrityError, match="does not hold standing"):
            store.load("c1", lib.OnionState())
    finally:
        restore()


def test_os23_a_file_of_schema_two_is_brought_up_by_the_readings_migration_every_root_unchanged(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state, _messages, unread = _read_state(loaded, read=False)
        path = tmp_path / "onion.db"
        assert store_mod.OnionStore(path, connect=_plain, require_encryption=False).save("c1", state) == unread
        conn = sqlite3.connect(str(path))
        conn.execute("DROP TABLE onion_statements")
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        assert (3, "statements") in [(target, name) for target, name, _step in store_mod._MIGRATIONS]
        conn = sqlite3.connect(str(path))
        assert conn.execute("PRAGMA user_version").fetchone()[0] == store_mod.SCHEMA_VERSION >= 3
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert "onion_statements" in tables
        assert conn.execute("SELECT root FROM onion_cursor").fetchone()[0] == unread, "the root is unchanged"
        conn.close()
        back = store.load("c1", lib.OnionState())
        assert back.statements == {} and store_mod.onion_root(store_mod.snapshot_of(back)) == unread
    finally:
        restore()


def test_os24_readings_the_saved_root_does_not_cover_are_left_out_and_a_readers_identity_is_in_the_root(
        tmp_path, caplog):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state, _messages, unread = _read_state(loaded)
        keys = [r.key for r in state.ledger.all()]
        state.readers = dict(state.readers, **{keys[0]: "a" * 64})
        path = tmp_path / "onion.db"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        store.save("c1", state)
        assert store.load("c1", lib.OnionState()).readers[keys[0]] == "a" * 64, "control: the reader saved"
        # A build that knew no reading saved the state since: the root it wrote covers none of the rows left.
        conn = sqlite3.connect(str(path))
        conn.execute("UPDATE onion_cursor SET root = ? WHERE conversation = 'c1'", (unread,))
        conn.commit()
        conn.close()
        caplog.set_level(logging.WARNING)
        back = store.load("c1", lib.OnionState())
        assert back.statements == {} and back.tree.all() == state.tree.all(), "left out, the rest as saved"
        assert any("4 span reading(s) its saved root does not cover" in r.getMessage() for r in caplog.records)
        store.save("c1", back)
        conn = sqlite3.connect(str(path))
        assert conn.execute("SELECT COUNT(*) FROM onion_statements").fetchone()[0] == 0, "the next save holds none"
        conn.close()
        # The reader is in the root: moved, refused; no digest, refused.
        store.save("c1", state)
        conn = sqlite3.connect(str(path))
        moved = conn.execute("UPDATE onion_statements SET reader = ? WHERE seq = 0", ("b" * 64,)).rowcount
        conn.commit()
        conn.close()
        assert moved == 1, "control: one reader moved in the file"
        with pytest.raises(store_mod.OnionIntegrityError, match="does not answer to its rows"):
            store.load("c1", lib.OnionState())
        read = list(store_mod.snapshot_of(state).statements)
        for reader in ("the 4B", "g" * 64, "A" * 64):
            _rewritten(store_mod, path, [(keys[0], reader, read[0][2])] + read[1:])
            with pytest.raises(store_mod.OnionIntegrityError, match="does not decode"):
                store.load("c1", lib.OnionState())
    finally:
        restore()


def test_os25_the_onion_carries_any_text_a_surrogate_kept_as_the_replacement_character(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        peels, probes = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.probes"]
        composer, core_store = loaded["opti_oignon.memory.composer"], loaded["opti_oignon.memory.core_store"]
        lone, split, shown = chr(0xD800), chr(0xD83D) + chr(0xDE00), chr(0xFFFD)
        typed = f"Bob lives in Lyon{lone}. Alice moved{split} to Oslo on 2026-03-04."
        kept = typed.replace(lone, shown).replace(split, shown * 2)
        messages = [oj.typed(typed), oj.answer(f"Noted{lone}: Alice moved to Oslo."), oj.typed("And Carol?"),
                    oj.answer("Carol stays in Lille.")]
        state = lib.state_for("c1")
        assert state.mirror(messages) == 4
        first = state.flesh.turns()[0]
        assert first["text"] == kept and len(kept) == len(typed), "one character for one: every place stays"
        assert first["segments"] == [[0, len(typed), "typed"]], "the parts where they were"
        assert state.flesh.turns()[1]["text"] == f"Noted{shown}: Alice moved to Oslo."
        assert state.mirror(messages) == 0, "the same conversation seen again as the same"
        # A summary the model wrote with one, through the queue's ladder.
        gate = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        tiny = composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=1, turn=60)
        outcome = lib.curate(state, lambda turns: _faithful(turns) + lone, gate=gate, budget=tiny)
        assert outcome.evicted and outcome.peel is not None and outcome.peel.text.endswith(shown), outcome.rung
        assert [t["text"] for t in state.cellar.get(outcome.receipt.key)][0] == kept, "the span as mirrored"
        # A pin, and its successor, with one.
        entry = lib.pin("c1", f"Answer in French{lone}.", actor=core_store.USER)
        assert state.core.get(entry).text == f"Answer in French{shown}."
        successor = lib.supersede("c1", entry, f"Answer in French or English{lone}.", actor=core_store.USER)
        assert state.core.get(successor).text == f"Answer in French or English{shown}."
        # The state saves and loads, its root with it.
        path = tmp_path / "onion.db"
        root = store_mod.OnionStore(path, connect=_plain, require_encryption=False).save("c1", state)
        back = store_mod.OnionStore(path, connect=_plain, require_encryption=False).load("c1", lib.OnionState())
        _same(state, back)
        assert store_mod.onion_root(store_mod.snapshot_of(back)) == root
        # A label or a lineage that holds one: read outside the grammar, never raised; a role that holds one is kept
        # as the words are; the state saves.
        other = lib.state_for("c2")
        assert other.mirror([dict(oj.typed("Hi there."), context=["document" + lone], lineage=["document:" + lone]),
                             {"role": "assistant" + lone, "content": "Hello."}]) == 2
        first_other = other.flesh.turns()[0]
        assert first_other["text"] == "Hi there." and other.flesh.turns()[1]["role"] == "assistant" + shown
        assert first_other["context"] == ["legacy"] and other.lineage[first_other["turn_id"]] == (peels.CUT,), (
            "read outside the grammar")
        store_mod.OnionStore(path, connect=_plain, require_encryption=False).save("c2", other)
        # A text that holds none is kept as it was; a pin that is no string is refused by name, the Core untouched.
        plain = "Bob lives in Lyon. Caf" + chr(0xE9) + " " + chr(0x1F600) + "."
        assert probes.storable(plain) == plain
        held = [(e.id, e.text, e.superseded_by) for e in state.core.all()]
        for verb in (lambda: lib.pin("c1", None, actor=core_store.USER),
                     lambda: lib.pin("c1", 123, actor=core_store.USER),
                     lambda: lib.supersede("c1", successor, None, actor=core_store.USER)):
            with pytest.raises(TypeError, match="the user's words as a string"):
                verb()
        assert [(e.id, e.text, e.superseded_by) for e in state.core.all()] == held
    finally:
        restore()


def test_os26_every_way_a_summary_becomes_a_peel_keeps_a_surrogate_as_the_replacement_character():
    from dataclasses import replace

    loaded, restore = _open()
    try:
        lib, peels, probes = (loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.peels"],
                              loaded["opti_oignon.memory.probes"])
        lone, shown = chr(0xD800), chr(0xFFFD)
        state = lib.state_for("c1")
        state.mirror(_TYPED * 2)
        # A second asking: the first summary loses a decision, the second holds a surrogate.
        decision = "We keep Docker on the build server."
        outcome = peels.advance(flesh=state.flesh, cellar=state.cellar, ledger=state.ledger, tree=state.tree,
                                lock=state.lock, gate=replace(peels.load_gate(), span_turns=2),
                                ladder=peels.load_ladder(), summarize=lambda turns: _LOSSY,
                                reask=lambda turns, missing: _LOSSY + " " + decision + lone, refusals={})
        assert outcome.rung == "reasked" and outcome.peel.text.endswith(shown), outcome.rung
        simple = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        leaf, _decision = peels.build_leaf(outcome.receipt.key, state.cellar, lambda turns: _faithful(turns) + " " + lone,
                                           simple, state.tree)
        assert leaf is not None and leaf.text.endswith(shown), "a leaf"
        gated = peels.evict_gated(flesh=state.flesh, cellar=state.cellar, ledger=state.ledger, tree=state.tree,
                                  gate=simple, summarize=lambda turns: _faithful(turns) + lone)
        assert gated.evicted and gated.peel.text.endswith(shown), "a gated eviction"
        children = [p.id for p in state.tree.all() if p.level == 0][-2:]
        built = peels.build_parent(children, state.cellar, lambda turns: _faithful(turns) + lone, simple, state.tree)
        parent = built[0] if isinstance(built, tuple) else built
        assert parent is not None and parent.level == 1 and parent.text.endswith(shown), "a parent"
        # Every surrogate, the halves a byte-escaping decoder leaves among them, and nothing else.
        edges = chr(0xD7FF) + chr(0xD800) + chr(0xDBFF) + chr(0xDC00) + chr(0xDC80) + chr(0xDCFF) + chr(0xDFFF) + chr(0xE000)
        assert probes.storable(edges) == chr(0xD7FF) + shown * 6 + chr(0xE000)
        sample = "  e" + chr(0x301) + " " + chr(0) + chr(0x1F600) + " Caf" + chr(0xE9) + " " * 3 + "x" * 5000 + "  "
        assert probes.storable(sample) == sample, "a text that holds none, as it was"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OS27-OS30 -- two processes, one file
# ---------------------------------------------------------------------------
class _Recording:
    """A connection that says each statement it is handed, and each attribute set on it, in ``said``."""

    def __init__(self, conn, said):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_said", said)

    def execute(self, sql, *args):
        self._said.append(" ".join(sql.split())[:40])
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        self._said.append(f"set {name}={value!r}")
        setattr(self._conn, name, value)


def _active(state):
    return [e.text for e in state.core.active()]


def test_os27_a_save_that_names_the_generation_it_read_writes_only_over_it_and_a_later_one_is_refused(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        core_store = loaded["opti_oignon.memory.core_store"]
        state = _plain_state(loaded, "c1")
        path = tmp_path / "onion.db"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        assert store.generation("c1") == 0 and state.generation == 0, "a conversation the file does not hold: 0"
        store.save("c1", state, generation=0)
        assert store.generation("c1") == 1 and state.generation == 1, "the next generation, left on the state"
        # Another process reads it, pins, and saves over what it read.
        other = store.load("c1", lib.OnionState())
        assert other.generation == 1, "a load leaves the generation it read"
        other.core.add("Pinned elsewhere.", actor=core_store.USER)
        other_root = store.save("c1", other, generation=1)
        assert store.generation("c1") == 2 and other.generation == 2
        # The first is stale: refused by name, nothing written.
        state.core.add("Pinned here.", actor=core_store.USER)
        before = path.read_bytes()
        with pytest.raises(store_mod.OnionStateStale, match="saved by another writer since it was read"):
            store.save("c1", state, generation=1)
        assert path.read_bytes() == before and state.generation == 1, "nothing written, the generation it read kept"
        back = store.load("c1", lib.OnionState())
        assert "Pinned elsewhere." in _active(back) and "Pinned here." not in _active(back)
        assert store_mod.onion_root(store_mod.snapshot_of(back)) == other_root and back.generation == 2
        # A state read as absent is refused once the file holds the conversation, and so is one ahead of the file.
        with pytest.raises(store_mod.OnionStateStale):
            store.save("c1", lib.OnionState(), generation=0)
        with pytest.raises(store_mod.OnionStateStale):
            store.save("c1", state, generation=store.generation("c1") + 1)
        # A save that names no generation writes as before, and raises the generation too.
        assert store.save("c1", state) == store_mod.onion_root(store_mod.snapshot_of(state))
        assert store.generation("c1") == 3 and state.generation == 1, "written; the state's generation untouched"
        for wrong in (-1, True, 1.0, "3"):
            with pytest.raises(TypeError, match="a generation is a whole number from 0"):
                store.save("c1", state, generation=wrong)
        assert store.generation("c1") == 3, "nothing written by a refused generation"
    finally:
        restore()


def test_os28_a_save_and_a_load_each_read_the_generation_inside_one_transaction(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        state = _plain_state(loaded, "c1")
        said = []
        store = store_mod.OnionStore(tmp_path / "onion.db", connect=lambda p: _Recording(sqlite3.connect(str(p)), said),
                                     require_encryption=False)
        store.save("c1", state, generation=0)
        said.clear()
        store.save("c1", state, generation=1)
        begin = said.index("BEGIN IMMEDIATE")
        read = next(i for i, sql in enumerate(said) if sql.startswith("SELECT generation FROM onion_cursor"))
        delete = next(i for i, sql in enumerate(said) if sql.startswith("DELETE FROM"))
        assert "set isolation_level=None" in said[:begin], "the driver's own transactions off, whichever it is"
        assert begin < read < delete, "the generation read inside the write transaction, before any row goes"
        assert said[-1] == "COMMIT", ("closed by name", said[-3:])
        said.clear()
        back = store.load("c1", lib.OnionState())
        assert back.generation == 2
        begin, end = said.index("BEGIN"), said.index("ROLLBACK")
        selects = [i for i, sql in enumerate(said) if sql.startswith("SELECT")]
        assert "set isolation_level=None" in said[:begin], "the driver's own transactions off"
        assert begin < selects[0] and selects[-1] < end, ("every row and the generation read in one transaction", said)
        assert any(said[i].startswith("SELECT generation FROM onion_cursor") for i in selects)
        said.clear()
        with pytest.raises(store_mod.OnionStateStale):
            store.save("c1", state, generation=1)
        assert "ROLLBACK" in said and not any(sql.startswith(("DELETE", "INSERT")) for sql in said), said
    finally:
        restore()


def test_os29_a_file_of_schema_three_is_brought_up_by_the_generations_migration_every_root_unchanged(tmp_path):
    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        path = tmp_path / "onion.db"
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        roots = {cid: store.save(cid, _plain_state(loaded, cid)) for cid in ("c1", "c2")}
        conn = sqlite3.connect(str(path))
        conn.execute("ALTER TABLE onion_cursor DROP COLUMN generation")
        conn.execute("PRAGMA user_version = 3")
        conn.commit()
        assert "generation" not in {row[1] for row in conn.execute("PRAGMA table_info(onion_cursor)")}, "control"
        conn.close()
        store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        assert (4, "generations") in [(target, name) for target, name, _step in store_mod._MIGRATIONS]
        assert _version(path) == store_mod.SCHEMA_VERSION >= 4, "the file says the schema it is now of"
        conn = sqlite3.connect(str(path))
        assert dict(conn.execute("SELECT conversation, root FROM onion_cursor")) == roots, "every root unchanged"
        assert dict(conn.execute("SELECT conversation, generation FROM onion_cursor")) == {"c1": 1, "c2": 1}, (
            "each conversation as saved once")
        conn.close()
        back = store.load("c1", lib.OnionState())
        assert back.generation == 1 and store_mod.onion_root(store_mod.snapshot_of(back)) == roots["c1"]
        store.save("c1", back, generation=1)
        assert (store.generation("c1"), store.generation("c2")) == (2, 1)
        before = path.read_bytes()
        store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        assert path.read_bytes() == before, "a second opening changes nothing"
    finally:
        restore()


def test_os30_a_process_reads_what_another_saved_since_and_never_writes_over_it(tmp_path, caplog):
    from dataclasses import replace

    loaded, restore = _open()
    try:
        lib, store_mod = loaded["opti_oignon.memory.librarian"], loaded["opti_oignon.memory.onion_store"]
        core_store = loaded["opti_oignon.memory.core_store"]
        path = tmp_path / "onion.db"
        lib._store[(str(path), False)] = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        config = replace(lib.load_config(), persist_path=str(path), require_encryption=False)
        store = lib.onion_store(config)
        reads, asked = [], []
        load, generation = store.load, store.generation
        store.load = lambda cid, state: reads.append(cid) or load(cid, state)
        store.generation = lambda cid: asked.append(cid) or generation(cid)
        # This process mirrors the conversation and saves it; what it alone keeps of it besides.
        state = lib.state_for("c1", config)
        with state.lock:
            state.mirror(_messages(4))
        lib._save_state("c1", state, config)
        assert state.generation == 1 and generation("c1") == 1
        reads.clear()
        record = object()
        state.journal, state.folded, state.unanswered = record, {"k1"}, {("k2", "reader")}
        assert lib.state_for("c1", config) is state and lib.peek_state("c1", config) is state and reads == [], (
            "a state whose generation the file holds is used as it is, its rows not read again")
        # Another process -- the terminal that opened it -- pins, and saves over what it read.
        other_store = store_mod.OnionStore(path, connect=_plain, require_encryption=False)
        other = other_store.load("c1", lib.OnionState())
        other.core.add("Pinned in the terminal.", actor=core_store.USER)
        other_store.save("c1", other, generation=other.generation)
        # This process's state is stale: its save is refused by name and writes nothing.
        with state.lock:
            state.mirror(_messages(6))
        caplog.set_level(logging.WARNING)
        before = path.read_bytes()
        with pytest.raises(store_mod.OnionStateStale, match="saved by another writer since it was read"):
            lib._save_state("c1", state, config)
        assert path.read_bytes() == before, "nothing written over the terminal's"
        assert any("saved by another process since this one read it; not written over" in r.getMessage()
                   for r in caplog.records), "said"
        # Its next use reads what the terminal saved, with what this process alone kept.
        again = lib.state_for("c1", config)
        assert again is not state and reads == ["c1"] and lib._states["c1"] is again, "read again in its place"
        assert _active(again) == ["Pinned in the terminal."] and again.generation == 2
        assert (again.journal, again.folded, again.unanswered) == (record, {"k1"}, {("k2", "reader")})
        assert again.lock is state.lock and again.slot is state.slot, "its locks go with it"
        assert lib._watermark["c1"] == len(again.flesh.turns()), "its growth counted from what it holds"
        # What this process saves now keeps the terminal's.
        lib.pin("c1", "Pinned in the server.", actor=core_store.USER, config=config)
        assert _active(other_store.load("c1", lib.OnionState())) == ["Pinned in the terminal.", "Pinned in the server."]
        # The block's way to the state reads again too.
        other = other_store.load("c1", lib.OnionState())
        other.core.add("Pinned in the terminal again.", actor=core_store.USER)
        other_store.save("c1", other, generation=other.generation)
        assert _active(lib.peek_state("c1", config))[-1] == "Pinned in the terminal again."
        # A file that no longer holds the conversation: it starts over.
        conn = sqlite3.connect(str(path))
        for table in store_mod.TABLES:
            conn.execute(f"DELETE FROM {table} WHERE conversation = 'c1'")
        conn.commit()
        conn.close()
        assert lib.peek_state("c1", config) is None and "c1" not in lib._states, "neither knows it"
        assert "c1" not in lib._watermark, "nor its growth"
        fresh = lib.state_for("c1", config)
        assert fresh.core.all() == [] and fresh.generation == 0 and fresh.journal is None
        assert lib.counters()["store"] == {"stale": 1, "read_again": 2, "gone": 1}, lib.counters()["store"]
        # With no store nothing is read.
        asked.clear()
        plain = replace(config, persist_path="")
        held = lib.state_for("c2", plain)
        assert lib.state_for("c2", plain) is held and lib.peek_state("c2", plain) is held and asked == []
        load_config = lib.load_config
        lib.load_config = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("unreadable"))
        try:
            assert lib.peek_state("c2") is held, "a configuration that does not read leaves the state as held"
            assert lib.state_for("c2") is held, "by either way to it"
        finally:
            lib.load_config = load_config
        # A store that cannot be read at a use is said once for the conversation, and raised.
        lib.state_for("c3", config)
        store.generation = lambda cid: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked"))
        caplog.clear()
        for _ in range(2):
            with pytest.raises(sqlite3.OperationalError):
                lib.state_for("c3", config)
        assert [r.getMessage() for r in caplog.records if "refused, not replaced" in r.getMessage()] == [
            "onion memory for c3 refused, not replaced: database is locked"], "said once"
        # A refusal of another kind is said too: a passing one does not silence it.
        reading = store.load
        store.generation = lambda cid: 1
        store.load = lambda cid, state: (_ for _ in ()).throw(store_mod.OnionIntegrityError("a row moved"))
        try:
            with pytest.raises(store_mod.OnionIntegrityError):
                lib.state_for("c3", config)
        finally:
            store.load = reading
        assert [r.getMessage() for r in caplog.records if "refused, not replaced" in r.getMessage()] == [
            "onion memory for c3 refused, not replaced: database is locked",
            "onion memory for c3 refused, not replaced: a row moved"], "each kind said once"
        store.generation = generation
        # One process, two threads: a use waits for the save the other thread is making, and never reads it as
        # another process's.
        arrived, release = threading.Event(), threading.Event()

        class Paused(lib.OnionState):
            pause = False

            @property
            def generation(self):
                return self.__dict__.get("_generation", 0)

            @generation.setter
            def generation(self, value):
                if self.pause:
                    arrived.set()
                    release.wait(10)
                self.__dict__["_generation"] = value

        class Watched:
            """The state's lock, saying when a thread waits for it."""

            def __init__(self):
                self.inner, self.waiting = threading.RLock(), threading.Event()

            def __enter__(self):
                if not self.inner.acquire(blocking=False):
                    self.waiting.set()
                    self.inner.acquire()
                return self

            def __exit__(self, *exc_info):
                self.inner.release()

        paused = Paused()
        paused.lock = Watched()
        lib._states["c4"] = paused
        with paused.lock:
            paused.mirror(_messages(2))
        lib._save_state("c4", paused, config)
        paused.pause = True
        saving = threading.Thread(target=lib._save_state, args=("c4", paused, config))
        saving.start()
        assert arrived.wait(10), "control: the save reached the generation it leaves"
        used = []
        using = threading.Thread(target=lambda: used.append(lib.state_for("c4", config)))
        using.start()
        while using.is_alive() and not paused.lock.waiting.wait(0.01):
            pass
        release.set()
        saving.join(10)
        using.join(10)
        paused.pause = False
        assert len(used) == 1 and used[0] is paused and paused.generation == 2, "its own save, not another's"
        # Two threads reading it again: the first one's reading stands.
        mine = lib.state_for("c5", config)
        with mine.lock:
            mine.mirror(_messages(2))
        lib._save_state("c5", mine, config)
        theirs = other_store.load("c5", lib.OnionState())
        other_store.save("c5", theirs, generation=theirs.generation)
        first, load_state = lib.OnionState(), lib._load_state

        def raced(cid, cfg):
            loaded = load_state(cid, cfg)
            lib._states[cid] = first
            return loaded

        lib._load_state = raced
        try:
            assert lib.state_for("c5", config) is first and lib._states["c5"] is first, "the first reading stands"
        finally:
            lib._load_state = load_state
        assert lib.counters()["store"] == {"stale": 1, "read_again": 2, "gone": 1}, lib.counters()["store"]
        # A state this process no longer holds is never written, whatever its generation: a conversation the file
        # stopped holding starts its numbering over.
        old = lib.state_for("c6", config)
        with old.lock:
            old.mirror(_messages(2))
        lib._save_state("c6", old, config)
        conn = sqlite3.connect(str(path))
        for table in store_mod.TABLES:
            conn.execute(f"DELETE FROM {table} WHERE conversation = 'c6'")
        conn.commit()
        conn.close()
        renewed = lib.state_for("c6", config)
        renewed.core.add("Pinned after the reset.", actor=core_store.USER)
        lib._save_state("c6", renewed, config)
        assert renewed is not old and renewed.generation == old.generation == 1, "control: the numbering started over"
        with pytest.raises(store_mod.OnionStateStale, match="no longer the one this process holds"):
            lib._save_state("c6", old, config)
        assert _active(other_store.load("c6", lib.OnionState())) == ["Pinned after the reset."], "nothing written"
        assert lib.counters()["store"] == {"stale": 2, "read_again": 2, "gone": 2}, lib.counters()["store"]
        # A refused window names both stores that can refuse it.
        words = {}
        for node in ast.parse(source("executor.py").read_text(encoding="utf-8")).body:
            named = isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_ONION_REFUSAL_WORDS"
                                                         for t in node.targets)
            if named:
                words = ast.literal_eval(node.value)
        assert "onion's store" in words["store"] and "conversation store" in words["store"], words["store"]
    finally:
        restore()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
