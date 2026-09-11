#!/usr/bin/env python3
"""Contracts for the drift ledger's physical store.

The ledger of the harness becomes a table: facts with provenance, kind and
confidence, superseded and never edited, with the contradiction census of
the deterministic templates taken at every write so drift is a count the
store carries. The connection is the repository's ``safe_connect``; the
contracts inject a plain connection on a temporary path and never touch
the data directory.

  * LS1 -- a fact round-trips through the table; a duplicate id and an
    invalid fact are refused.
  * LS2 -- supersession only: no edit surface, the superseded statement is
    byte-equal after supersession, the head resolves, and no SQL in the
    module ever updates a statement.
  * LS3 -- the contradiction census is taken at write: a contradicting
    statement is recorded against the held fact, a compatible one is not,
    and the drift rate is per thousand turns, unknown for zero turns.
  * LS4 -- the store connects through safe_connect and never through the
    plain client.
  * LS5 -- migration maps every canonical fact to a valid ledger fact, with
    the kind mapped, the source kept as provenance, and inactive rows
    closed; nothing is dropped silently.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source with the harness's ledger module.
"""

import ast
import re
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _open():
    loaded, restore = isolate(
        targets={
            "opti_oignon.memory.drift": source("memory", "drift.py"),
            "opti_oignon.memory.ledger_store": source("memory", "ledger_store.py"),
        },
        blocked=("opti_oignon.db_utils",),
        packages=("opti_oignon.memory",),
    )
    return loaded["opti_oignon.memory.ledger_store"], loaded["opti_oignon.memory.drift"], restore


def _store(mod, tmp_path):
    return mod.LedgerStore(tmp_path / "ledger.db", connect=lambda p: sqlite3.connect(str(p)))


def _fact(drift, fid, statement, kind="fact", **over):
    fields = dict(id=fid, statement=statement, kind=kind, provenance=["t0001"], confidence=0.9, valid_from="2026-09-11")
    fields.update(over)
    return drift.LedgerFact(**fields)


# ---------------------------------------------------------------------------
# LS1 -- round trip
# ---------------------------------------------------------------------------
def test_ls1_a_fact_round_trips_and_bad_input_is_refused(tmp_path):
    mod, drift, restore = _open()
    try:
        store = _store(mod, tmp_path)
        fact = _fact(drift, "f1", "Alice lives in Berlin.", provenance=["t0001", "t0002"], confidence=0.8)
        fid, found = store.add(fact)
        assert fid == "f1" and found == []
        got = store.get("f1")
        assert got == fact, "every field survives the table, provenance included"
        assert store.active() == [fact] and store.all() == [fact]
        with pytest.raises(ValueError):
            store.add(fact)
        with pytest.raises(ValueError):
            store.add(_fact(drift, "f2", "", provenance=[]))
        assert store.all() == [fact], "a refused write leaves no row"
        reopened = _store(mod, tmp_path)
        assert reopened.get("f1") == fact, "the row is on disk, not in the object"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LS2 -- supersession only
# ---------------------------------------------------------------------------
def test_ls2_supersession_links_and_no_sql_ever_edits_a_statement(tmp_path):
    mod, drift, restore = _open()
    try:
        store = _store(mod, tmp_path)
        for name in ("edit", "update", "remove", "delete", "set", "replace"):
            assert not hasattr(store, name)
        store.add(_fact(drift, "f1", "Alice lives in Berlin."))
        new_id = store.supersede("f1", _fact(drift, "f2", "Alice lives in Oslo."))
        assert new_id == "f2"
        assert store.get("f1").statement == "Alice lives in Berlin."
        assert store.get("f1").superseded_by == "f2" and store.get("f1").status == "superseded"
        assert store.head("f1").id == "f2"
        assert [f.id for f in store.active()] == ["f2"]
        with pytest.raises(ValueError):
            store.supersede("f1", _fact(drift, "f3", "Alice lives in Rome."))
        with pytest.raises(KeyError):
            store.supersede("nope", _fact(drift, "f4", "Alice lives in Rome."))
        text = source("memory", "ledger_store.py").read_text(encoding="utf-8")
        updates = [m.group(0) for m in re.finditer(r"UPDATE\s+\{?\w+\}?\s+SET\s+[^\"']*", text, re.IGNORECASE)]
        assert len(updates) >= 1, "control: supersession is an UPDATE of the link"
        for u in updates:
            assert "statement" not in u.lower(), f"no SQL updates a statement: {u}"
            assert "kind" not in u.lower() and "provenance" not in u.lower()
    finally:
        restore()


# ---------------------------------------------------------------------------
# LS3 -- census at write
# ---------------------------------------------------------------------------
def test_ls3_the_contradiction_census_is_taken_at_write(tmp_path):
    mod, drift, restore = _open()
    try:
        store = _store(mod, tmp_path)
        store.add(_fact(drift, "f1", "Alice lives in Berlin."))
        _fid, found = store.add(_fact(drift, "f2", "Alice prefers tea.", kind="preference"))
        assert found == [], "compatible facts contradict nothing"
        _fid, found = store.add(_fact(drift, "f3", "Alice lives in Oslo."))
        assert [c.fact_id for c in found] == ["f1"] and found[0].template == "exclusive"
        recorded = store.contradictions()
        assert len(recorded) == 1
        assert recorded[0]["fact_id"] == "f1" and recorded[0]["statement"] == "Alice lives in Oslo."
        assert recorded[0]["new_fact_id"] == "f3"
        assert store.drift_rate(turns=2000) == 0.5, "one contradiction over two thousand turns"
        assert store.drift_rate(turns=0) is None, "no turns means unknown, never 0.0"
        assert store.drift_rate(turns=1000) == 1.0
    finally:
        restore()


# ---------------------------------------------------------------------------
# LS4 -- safe_connect
# ---------------------------------------------------------------------------
def test_ls4_the_store_connects_through_safe_connect_never_the_plain_client(tmp_path):
    mod, drift, restore = _open()
    try:
        text = source("memory", "ledger_store.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        imported = {(n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert ("..db_utils", "safe_connect") in imported or (".." + "db_utils", "safe_connect") in imported or any(
            m and m.endswith("db_utils") and a == "safe_connect" for m, a in imported
        ), "safe_connect is the default connection"
        assert "sqlite3.connect(" not in text, "the plain client is never called by the module"
        with pytest.raises(Exception):
            mod.LedgerStore(tmp_path / "blocked.db")
        assert not (tmp_path / "blocked.db").exists(), "with db_utils unreachable no plaintext file is created"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LS5 -- migration
# ---------------------------------------------------------------------------
def test_ls5_migration_maps_every_canonical_fact_and_drops_nothing(tmp_path):
    mod, drift, restore = _open()
    try:
        rows = [
            dict(id="m1", text="Alice lives in Berlin.", category="identity", source="extraction", created_at="2026-01-01", updated_at="2026-01-01", active=True),
            dict(id="m2", text="Alice prefers tea.", category="preference", source="manual", created_at="2026-01-02", updated_at="2026-01-02", active=True),
            dict(id="m3", text="Harvest ships in May.", category="project", source="", created_at="2026-01-03", updated_at="2026-01-03", active=True),
            dict(id="m4", text="Bob reviews the release.", category="contact", source="auto-capture", created_at="2026-01-04", updated_at="2026-02-01", active=False),
            dict(id="m5", text="The budget is 1200 euros.", category="fact", source="extraction", created_at="2026-01-05", updated_at="2026-01-05", active=True),
            dict(id="m6", text="Answers are concise.", category="odd-category", source="extraction", created_at="2026-01-06", updated_at="2026-01-06", active=True),
        ]
        facts = mod.migrate_facts(rows)
        assert len(facts) == len(rows), "nothing dropped"
        by_id = {f.id: f for f in facts}
        assert by_id["m1"].kind == "fact" and by_id["m2"].kind == "preference" and by_id["m3"].kind == "project"
        assert by_id["m4"].kind == "fact" and by_id["m6"].kind == "fact", "unknown categories land on fact, never refused"
        assert by_id["m1"].provenance == ["memory_facts:extraction"]
        assert by_id["m3"].provenance == ["memory_facts:unknown"]
        assert by_id["m4"].valid_until == "2026-02-01" and by_id["m4"].status == "active", (
            "an inactive row is closed by date; supersession is a link the old store never had"
        )
        assert by_id["m1"].valid_until == ""
        for f in facts:
            assert drift.validate_fact(f) == []
        store = _store(mod, tmp_path)
        recorded = 0
        for f in facts:
            _fid, found = store.add(f)
            recorded += len(found)
        assert len(store.all()) == len(rows)
        assert recorded == 0, "control: this fixture holds no contradiction"
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
