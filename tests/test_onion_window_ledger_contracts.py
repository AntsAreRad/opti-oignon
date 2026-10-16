#!/usr/bin/env python3
"""Contracts for the ledger row of a turn the onion's journal laid: its window's figures, kept and read back.

The executor reports a journal turn's figures on its ledger row -- the
rung planned, why a window was refused and with which whole numbers, the
window as composed, whether the engine's slot bound was read -- and, on a
cold window, the engine's own count of its prompt. The ledger keeps them
in columns of their own, numbers and bare-word labels, and a ledger
written before those columns gains them.

  * JW29 -- every exit of a journal turn records the window's figures as
    far as the turn reached them, and the ledger itself keeps them and
    reads them back, through its newest rows and through each request's
    row: a stop before the plan, a plan refused, a stop after the plan,
    offline, the governor's refusal, a stop during the admission, a
    composition refused, a guard failed, a cache hit, a stop at the
    lookup, before the stream and on its thread, a stream that failed, an
    answer that used all its room, a descent on an engine whose slots name
    no context, a turn cold then warm. The engine's count of the prompt is
    kept for the cold window alone. A turn with the onion off leaves every
    one of those columns empty.
  * JW30 -- a ledger written before those columns opens, gains them in the
    table's order and keeps every row it held, each reading them empty,
    then takes the new figures; opened again it is left as it is; a
    migration that fails leaves the file as it was, and the ledger
    unavailable.
  * JW30b -- a number no column can hold -- an infinity, a NaN, a whole
    number past 64 bits -- drops its field, never the row: the row lands
    with that field empty and every other figure kept, the bounds of a
    64-bit column kept whole.

Local-only (the public distribution ships no tests). The executor runs in
the onion window world (tests/_onion_window_world.py); the ledger is the
module itself, loaded through the shared isolation window, over a database
in a temporary directory.
"""

import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_window_world as w  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_jw29_every_exit_of_a_journal_turn_records_its_windows_figures_and_the_ledger_keeps_them": 10.0,
    "test_jw30_a_ledger_written_before_the_journals_columns_gains_them_and_keeps_its_rows": 3.0,
    "test_jw30b_a_number_no_column_can_hold_drops_its_field_never_the_row": 2.0,
}

_LEDGER = "opti_oignon.context_ledger"


def _plain_connect(path, **kwargs):
    return sqlite3.connect(str(path), check_same_thread=False)


def _ledger_module(data_dir, *, connect=_plain_connect):
    """The ledger module itself, from its file: its seam ``connect``, its data directory ``data_dir``."""
    seam = types.ModuleType("opti_oignon.db_utils")
    seam.safe_connect = connect
    config = types.ModuleType("opti_oignon.config")
    config.DATA_DIR = Path(data_dir)
    loaded, restore = isolate(targets={_LEDGER: source("context_ledger.py")},
                              seeded={"opti_oignon.db_utils": seam, "opti_oignon.config": config})
    restore()
    return loaded[_LEDGER]


# ---------------------------------------------------------------------------
# JW29 -- every exit's figures, kept and read back
# ---------------------------------------------------------------------------
_WINDOW = ("win_rung", "win_num_ctx", "win_num_predict", "win_tokens", "win_accounted", "win_epoch", "win_motive",
           "win_first_diff", "win_cold", "win_descent", "win_cut")
_FIGURES = ("win_slot_unknown", "win_plan_rung", "win_refusal", "win_figures") + _WINDOW + ("engine_prompt_tokens",)
_ONE = (("conv-1", w.FIRST, {}),)
_TWO = (("conv-1", w.FIRST, {}), ("conv-1", w.SECOND, {}))
_TWO_CONVERSATIONS = (("conv-1", w.FIRST, {}), ("conv-2", w.FIRST, {}))


def _failing(world):
    def chat(**kwargs):
        raise RuntimeError("the engine went away")

    world.scripted.chat = chat


def _stopped_at_the_switch(world):
    """A stop raised as the onion's switch is read: the turn ends before anything of its window is read."""
    reading = world.executor_module._onion_enabled

    def read(path=None):
        world.run.stop.set()
        return reading(path)

    world.executor_module._onion_enabled = read


# Each way a journal turn ends: its name, the world's keywords, its turns, a hook on the world opened, and each
# row's outcome, how far the turn reached -- not to its plan, a plan refused, a rung planned, a window composed --
# and its refusal.
_EXITS = (
    ("a stop before the plan", {"gov": {}}, _ONE, _stopped_at_the_switch, [("cancelled", "before", None)]),
    ("a plan refused", {"gov": {}, "journal": {"plan_refusal": ("turn", (900, 512))}}, _ONE, None,
     [("window_refused", "nothing", ("turn", [900, 512]))]),
    ("a stop after the plan", {"gov": {}, "stop_at": {"plan": {0}}}, _ONE, None, [("cancelled", "rung", None)]),
    ("offline", {"gov": {}, "offline": True}, _ONE, None, [("offline_queued", "rung", None)]),
    ("the governor refused", {"gov": {"admitted": False}}, _ONE, None, [("governor_refused", "rung", None)]),
    ("a stop during the admission", {"gov": {}, "stop_at": {"admission": {0}}}, _ONE, None,
     [("cancelled", "rung", None)]),
    ("a composition refused", {"gov": {}, "journal": {"compose_refusal": ("window", (17000, 16384))}}, _ONE, None,
     [("window_refused", "rung", ("window", [17000, 16384]))]),
    ("the guard failed", {"gov": {}, "journal": {"accounted": 16000, "num_predict": 1024}}, _ONE, None,
     [("window_refused", "rung", ("count", [16000, 1024, 16384]))]),
    ("a cache hit", {"gov": {}, "hit": True, "engine_extra": {"eval_count": 3}}, _TWO_CONVERSATIONS, None,
     [("completed", "window", None), ("cache_hit", "window", None)]),
    ("a stop at the lookup", {"gov": {}, "stop_at": {"lookup": {0}}}, _ONE, None, [("cancelled", "window", None)]),
    ("a stop before the stream", {"gov": {}, "stop_at": {"generating": {0}}}, _ONE, None,
     [("cancelled", "window", None)]),
    ("a stop on the stream's thread", {"gov": {}, "stop_at": {"thread": {0}}}, _ONE, None,
     [("cancelled", "window", None)]),
    ("a stream that failed", {"gov": {}}, _ONE, _failing, [("error", "window", None)]),
    ("an answer that used all its room", {"gov": {}, "journal": {"num_predict": 1024},
                                          "engine_extra": {"eval_count": 1024, "prompt_eval_count": 321}}, _ONE, None,
     [("length", "window", None)]),
    ("a descent on an engine whose slots name no context",
     {"gov": {"num_ctx": 8192}, "backend": "llama_server", "engine_extra": {"eval_count": 3}}, _ONE, None,
     [("completed", "window", None)]),
    ("a turn cold then warm", {"gov": {}, "journal": {"cut": (("memory", 2), ("rag", 1), ("web", 0))},
                               "engine_extra": {"eval_count": 5, "prompt_eval_count": 321}}, _TWO, None,
     [("completed", "window", None), ("completed", "window", None)]),
)


def _teed(world, ledger):
    """The executor's ledger seam, each row it writes kept whole by the world and written to ``ledger`` too; what
    each write to the ledger answered, in order."""
    seam, landed = world.executor_module._get_context_ledger, []

    def record(**fields):
        seam().record(**fields)
        landed.append(ledger.record(**fields))

    world.executor_module._get_context_ledger = lambda: SimpleNamespace(record=record)
    return landed


def _expected(world, exits, extra):
    """Each row's figures as the turn reached them: the slot bound read, the rung planned, the refusal and its
    whole numbers, the window composed, and the engine's count of a cold prompt."""
    windows, rows = iter(world.journal.composed), []
    plan_rung = world.journal.rung
    for outcome, reached, refusal in exits:
        row = dict.fromkeys(_FIGURES)
        if reached == "before":
            rows.append((outcome, row))
            continue
        row["win_slot_unknown"] = 1 if world.backend.name == "llama_server" else 0
        row["win_plan_rung"] = plan_rung if reached != "nothing" else None
        if refusal is not None:
            row["win_refusal"], row["win_figures"] = refusal
        if reached == "window":
            window = next(windows)
            row.update(win_rung=window.rung, win_num_ctx=window.num_ctx, win_num_predict=window.num_predict,
                       win_tokens=window.tokens, win_accounted=window.accounted, win_epoch=window.epoch,
                       win_motive=window.motive, win_first_diff=window.first_diff, win_cold=int(window.cold),
                       win_descent=int(window.num_ctx < plan_rung), win_cut=dict(window.cut))
            if window.cold and outcome in ("completed", "length") and "prompt_eval_count" in extra:
                row["engine_prompt_tokens"] = extra["prompt_eval_count"]
        rows.append((outcome, row))
    return rows


def test_jw29_every_exit_of_a_journal_turn_records_its_windows_figures_and_the_ledger_keeps_them(tmp_path):
    mod = _ledger_module(tmp_path)
    seen = set()
    for index, (name, keywords, turns, hook, exits) in enumerate(_EXITS):
        ledger = mod.ContextLedger(db_path=tmp_path / f"exit-{index}.db")
        assert ledger.available, name
        world, close = w.open_world(**keywords)
        try:
            landed = _teed(world, ledger)
            if hook is not None:
                hook(world)
            w.run_turns(world, turns)
        finally:
            close()
        expected = _expected(world, exits, keywords.get("engine_extra") or {})
        assert [row["outcome"] for row in world.rows] == [outcome for outcome, _row in expected], (name, world.rows)
        assert landed == [True] * len(world.rows), (name, landed)
        newest = ledger.recent(limit=50)
        assert [row["request_id"] for row in newest] == [row["request_id"] for row in reversed(world.rows)], name
        for emitted, read, (outcome, figures) in zip(world.rows, reversed(newest), expected):
            kept = ledger.get(emitted["request_id"])
            assert kept == read, ("both readers give the same row", name)
            assert kept["outcome"] == outcome, (name, kept["outcome"])
            assert {key: kept[key] for key in _FIGURES} == figures, (name, outcome, kept, figures)
            seen.add(outcome)
    assert seen == {"window_refused", "cancelled", "offline_queued", "governor_refused", "completed", "cache_hit",
                    "error", "length"}, ("control: every exit was reached", seen)

    # The onion off: the columns exist, and its row leaves every one of them empty.
    ledger = mod.ContextLedger(db_path=tmp_path / "off.db")
    world, close = w.open_world(gov={}, journal={"enabled": False}, engine_extra={"prompt_eval_count": 321})
    try:
        _teed(world, ledger)
        w.run_turns(world, _ONE)
    finally:
        close()
    (kept,) = ledger.recent(limit=5)
    assert kept["outcome"] == "completed" and "plan" not in w.kinds({"trace": world.trace.turns[0]})
    assert {key: kept[key] for key in _FIGURES} == dict.fromkeys(_FIGURES), kept


# ---------------------------------------------------------------------------
# JW30 -- a ledger written before the journal's columns
# ---------------------------------------------------------------------------
# The table as the ledger wrote it before the journal's figures had columns.
_OLD_SCHEMA = """
CREATE TABLE context_ledger_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL,
    conversation_id TEXT,
    model TEXT NOT NULL,
    caller TEXT NOT NULL,
    outcome TEXT NOT NULL,
    token_method TEXT NOT NULL,
    tokens_system INTEGER,
    tokens_history INTEGER,
    tokens_user INTEGER,
    tokens_project INTEGER,
    tokens_manifest INTEGER,
    tokens_total INTEGER,
    zones_json TEXT,
    cache_hit INTEGER NOT NULL,
    cache_hit_type TEXT,
    cache_similarity REAL,
    cache_stored INTEGER NOT NULL,
    retrieval_count INTEGER NOT NULL,
    retrieval_top_score REAL,
    gov_action TEXT,
    gov_admitted INTEGER,
    gov_requested_ctx INTEGER,
    gov_num_ctx INTEGER,
    gov_conditional_eviction INTEGER,
    gov_keep_alive TEXT,
    gov_reason TEXT,
    duration_ms REAL
);
CREATE INDEX idx_context_ledger_request ON context_ledger_entries(request_id);
CREATE INDEX idx_context_ledger_ts ON context_ledger_entries(ts);
"""
_OLD_ROWS = (
    ("old-1", "completed", 120, '[{"zone":"system","budgeted":100,"actual":90,"trimmed":0,"strategy":"fixed"}]', 1,
     16384, "admit"),
    ("old-2", "cache_hit", 64, None, 0, None, None),
)


def _older(path):
    """A ledger file in the table of before, holding two rows."""
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(_OLD_SCHEMA)
        conn.executemany(
            "INSERT INTO context_ledger_entries (ts, request_id, model, caller, outcome, token_method, tokens_total, "
            "zones_json, cache_hit, cache_stored, retrieval_count, gov_requested_ctx, gov_action) "
            "VALUES (1.0, ?, 'm', 'chat', ?, 'estimated', ?, ?, 0, ?, 0, ?, ?)", _OLD_ROWS)
    conn.close()


def _columns(path):
    conn = sqlite3.connect(str(path))
    try:
        return [row[1] for row in conn.execute("PRAGMA table_info(context_ledger_entries)").fetchall()]
    finally:
        conn.close()


class _Faulting:
    """A connection whose statements naming ``fault`` fail, as a disk that gives out midway would."""

    def __init__(self, conn, fault):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_fault", fault)

    def execute(self, sql, *args):
        if self._fault in sql:
            raise sqlite3.OperationalError("disk I/O error")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        setattr(self._conn, name, value)


def test_jw30_a_ledger_written_before_the_journals_columns_gains_them_and_keeps_its_rows(tmp_path):
    path = tmp_path / "ledger.db"
    _older(path)
    before = _columns(path)
    assert "win_rung" not in before and "engine_prompt_tokens" not in before, "control: the table of before"
    mod = _ledger_module(tmp_path)
    added = [column for column, _definition in mod._WINDOW_COLUMNS]
    ledger = mod.ContextLedger(db_path=path)
    assert ledger.available
    assert _columns(path) == ["id", *mod._COLUMNS] == before + added, _columns(path)
    held = ledger.recent(limit=10)
    assert [(row["id"], row["request_id"], row["outcome"], row["tokens_total"]) for row in held] == [
        (2, "old-2", "cache_hit", 64), (1, "old-1", "completed", 120)], held
    old = ledger.get("old-1")
    assert old["gov_requested_ctx"] == 16384 and old["gov_action"] == "admit" and old["cache_stored"] == 1
    assert old["zones"] == [{"zone": "system", "budgeted": 100, "actual": 90, "trimmed": 0, "strategy": "fixed"}]
    for row in held:
        assert all(row[column] is None for column in added if column in row), row
        assert row["win_figures"] is None and row["win_cut"] is None, row
    assert ledger.record(request_id="new-1", model="m", outcome="window_refused", win_plan_rung=16384,
                         win_refusal="count", win_figures=[16000, 1024, 16384], win_cut={"rag": 1}, win_cold=True,
                         engine_prompt_tokens=777) is True
    new = ledger.get("new-1")
    assert (new["win_plan_rung"], new["win_refusal"], new["win_figures"], new["win_cut"], new["win_cold"],
            new["engine_prompt_tokens"]) == (16384, "count", [16000, 1024, 16384], {"rag": 1}, 1, 777), new
    again = mod.ContextLedger(db_path=path)
    assert again.available and _columns(path) == ["id", *mod._COLUMNS], "opened again, the file is left as it is"
    assert [row["request_id"] for row in again.recent(limit=10)] == ["new-1", "old-2", "old-1"]

    # A migration that fails midway leaves the file as it was, rows and all, and the ledger unavailable.
    broken = tmp_path / "broken" / "ledger.db"
    broken.parent.mkdir()
    _older(broken)
    faulting = _ledger_module(tmp_path, connect=lambda p, **kw: _Faulting(_plain_connect(p), "ADD COLUMN win_epoch"))
    assert faulting.ContextLedger(db_path=broken).available is False
    assert _columns(broken) == before, _columns(broken)
    conn = sqlite3.connect(str(broken))
    try:
        assert [row[0] for row in conn.execute("SELECT request_id FROM context_ledger_entries ORDER BY id")] == [
            "old-1", "old-2"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# JW30b -- a number no column can hold
# ---------------------------------------------------------------------------
def test_jw30b_a_number_no_column_can_hold_drops_its_field_never_the_row(tmp_path):
    mod = _ledger_module(tmp_path)
    ledger = mod.ContextLedger(db_path=tmp_path / "ledger.db")
    rows = (
        ("past-64-bits", {"engine_prompt_tokens": 1 << 63, "win_tokens": 6000}, ("engine_prompt_tokens",)),
        ("infinite", {"win_tokens": float("inf"), "engine_prompt_tokens": 6000}, ("win_tokens",)),
        ("not-a-number", {"win_accounted": float("nan"), "win_tokens": 6000}, ("win_accounted",)),
        ("below-64-bits", {"win_first_diff": -(1 << 63) - 1, "win_tokens": 6000}, ("win_first_diff",)),
    )
    for request_id, figures, dropped in rows:
        assert ledger.record(request_id=request_id, model="m", outcome="completed", win_plan_rung=16384,
                             **figures) is True, ("the row lands", request_id)
        kept = ledger.get(request_id)
        assert kept is not None and kept["win_plan_rung"] == 16384, request_id
        for field, value in figures.items():
            expected = None if field in dropped else value
            assert kept[field] == expected, (request_id, field, kept[field])
    assert ledger.record(request_id="bounds", model="m", outcome="completed", win_tokens=(1 << 63) - 1,
                         win_first_diff=-(1 << 63)) is True
    bounds = ledger.get("bounds")
    assert (bounds["win_tokens"], bounds["win_first_diff"]) == ((1 << 63) - 1, -(1 << 63)), bounds
    assert ledger.stats()["rows"] == len(rows) + 1, "every row landed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
