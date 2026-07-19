#!/usr/bin/env python3
"""What the coding-history store promises about the tasks and analytics it keeps.

The store is a per-domain SQLite ledger for the coding agent: it records a task,
the steps taken under it, the test runs, the human checkpoints, and a working-
memory blob, and it rolls all of that up into analytics computed in SQL rather
than in memory. Every write is DATA -- a task id, a task text, a model name may
carry SQL metacharacters, and each has to round-trip as a string rather than
execute, so every statement is parameterised and the batch delete loops single-id
parameterised deletes rather than assembling an IN clause.

Three shapes of the store's behaviour are worded by the code more narrowly than a
first reading of the docstrings suggests, and this suite pins the code:

  * The overall step average over an EMPTY ledger comes back with null figures,
    not zeroes. The zero-default branch guards on the row being absent, but an
    aggregate query always returns exactly one row, so that branch never runs and
    an empty ledger yields ``avg_steps`` of None. The suite pins the null the
    code produces, not the zero the guard was written to hand back.

  * Foreign keys are DECLARED on every child table but never turned on: the
    connection factory sets journal mode and nothing else. A step recorded
    against a task id that was never started therefore lands as an orphan rather
    than being refused, and referential integrity is held instead by the manual
    cascade every delete path performs -- children first, parent last. The suite
    pins both halves: the orphan a missing enforcement admits, and the complete
    cascade the code runs in its place.

  * The enabled flag gates WRITES only. A disabled store drops a task start, a
    step, a test, a checkpoint, and a working-memory save on the floor, and its
    ``delete_working_memory`` returns False without touching a row -- but
    ``delete_task``, ``prune``, the batch deletes, the reads, and the analytics
    all run regardless of the flag. The suite pins that asymmetry as the
    behaviour it is.

Reads and roll-ups degrade rather than raise: a detail lookup, a listing, an
analytics payload over a ledger with no matching rows comes back None, empty, or
zeroed instead of throwing. Stored blobs are bounded before they reach disk --
task text, step and test output, and the plan and memory JSON are each truncated
to a configured ceiling -- so a runaway field cannot grow the row without limit.

The store reaches for exactly one project seam: the connection factory. It is
seeded with a plain sqlite connection so a real temporary database backs every
contract, and one contract removes it entirely to prove the module's own import
fallback still connects. The module builds a singleton on load whose paths derive
from its own file, into the git-ignored package data directory; every contract
here builds its own store with an explicit temporary db path and a config path
that does not exist, so none depends on that singleton, its location, or the seed
config on disk. Loaded through the shared isolation window; no real backend is
ever touched.
"""

import sqlite3
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.coding_history"


def _default_connect(path, **kwargs):
    return sqlite3.connect(str(path), **kwargs)


def _load(tmp_path, *, connect=_default_connect, block_db=False):
    """Load the coding-history store in isolation.

    connect   -- the stand-in ``safe_connect``. The default is a plain sqlite
                 connection; a caller can pass a counting factory to pin the
                 seam itself.
    block_db  -- when true the connection module is declared UNREACHABLE and
                 proven so, so the module's own import-fallback connection
                 factory is what runs. There is no data-directory seam to
                 redirect: the module derives its default path from its own
                 file, and the module-level singleton it builds on load writes
                 into the (git-ignored) package data directory. Every contract
                 builds its own store with an explicit temporary path, so no
                 contract ever depends on that singleton or its location.
    """
    seeded = {}
    blocked = []

    if block_db:
        blocked.append("opti_oignon.db_utils")
    else:
        du = types.ModuleType("opti_oignon.db_utils")
        du.safe_connect = connect
        seeded["opti_oignon.db_utils"] = du

    loaded, restore = isolate(
        targets={_TARGET: source("coding_history.py")},
        blocked=blocked,
        seeded=seeded,
    )
    return loaded[_TARGET], restore


# --- fixtures -------------------------------------------------------------
# A store is always built with an explicit db path distinct from the singleton's
# and a config path that does not exist, which makes the loader fall through to
# the built-in defaults. A contract that needs non-default limits passes a
# written config file instead.

def _store(ch, tmp_path, *, db="h.db", config_path=None):
    return ch.CodingHistoryStore(
        db_path=Path(tmp_path) / db,
        config_path=(
            config_path if config_path is not None
            else Path(tmp_path) / "absent.yaml"
        ),
    )


def _cfg_file(tmp_path, name="coding_history.yaml", **cfg):
    """Write a flat coding-history config the loader reads through yaml."""
    lines = [f"{key}: {value!r}" for key, value in cfg.items()]
    path = Path(tmp_path) / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _raw(tmp_path, db="h.db"):
    return sqlite3.connect(str(Path(tmp_path) / db))


def _touch_created(tmp_path, task_id, value, db="h.db"):
    raw = _raw(tmp_path, db)
    raw.execute("UPDATE tasks SET created_at=? WHERE task_id=?", (value, task_id))
    raw.commit()
    raw.close()


# =========================================================================
# Config and data classes
# =========================================================================

def test_h1_load_config_missing_and_malformed_fall_back_to_defaults(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        missing = _store(ch, tmp_path, config_path=Path(tmp_path) / "none.yaml")
        assert missing._enabled is True
        assert missing._max_tasks == 200 and missing._retention_days == 30
        assert missing._max_output == 10000 and missing._max_plan == 50000, (
            "an absent config file leaves every default property standing"
        )
        bad = Path(tmp_path) / "bad.yaml"
        bad.write_text(":\n  - [unbalanced\n", encoding="utf-8")
        store = _store(ch, tmp_path, db="bad.db", config_path=bad)
        assert store._enabled is True and store._max_tasks == 200, (
            "a config that will not parse falls back to defaults, never raises "
            "out of construction"
        )
        cfg = _cfg_file(tmp_path, enabled=False, max_tasks=7, retention_days=3)
        over = _store(ch, tmp_path, db="over.db", config_path=cfg)
        assert over._enabled is False and over._max_tasks == 7 and over._retention_days == 3, (
            "config values override the defaults where present"
        )
    finally:
        restore()


def test_h2_dataclasses_roundtrip_to_dict(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        s = ch.TaskSummary(
            task_id="t", task_text="do", project_path="/p", model="m",
            status="started", step_count=3, completed_steps=2, test_runs=1,
            last_passed=True, created_at=1.0, completed_at=None,
        )
        assert s.to_dict() == {
            "task_id": "t", "task_text": "do", "project_path": "/p", "model": "m",
            "status": "started", "step_count": 3, "completed_steps": 2,
            "test_runs": 1, "last_passed": True, "created_at": 1.0,
            "completed_at": None,
        }
        d = ch.TaskDetail(
            task_id="t", task_text="do", project_path="/p", model="m",
            status="started", plan_json={"k": [1, 2]}, created_at=1.0,
            completed_at=None,
        )
        assert d.steps == [] and d.tests == [] and d.checkpoints == [], (
            "the nested collections default to empty lists"
        )
        d.steps.append({"n": 1})
        assert d.to_dict()["steps"] == [{"n": 1}] and d.to_dict()["plan_json"] == {"k": [1, 2]}
        cp = ch.CheckpointState(
            task_id="t", task_text="do", project_path="/p", model="m",
            plan_json=None, current_step=4, phase="apply", originals_hash="abc",
        )
        assert cp.current_step == 4 and cp.phase == "apply" and cp.originals_hash == "abc"
    finally:
        restore()


# =========================================================================
# Seam and construction
# =========================================================================

def test_h3_all_db_access_flows_through_the_connection_seam(tmp_path):
    calls = {"n": 0}

    def counting(path, **kwargs):
        calls["n"] += 1
        return sqlite3.connect(str(path), **kwargs)

    ch, restore = _load(tmp_path, connect=counting)
    try:
        store = _store(ch, tmp_path)
        calls["n"] = 0
        store.record_task_start("t", "text", model="m")
        store.record_step("t", 1)
        store.record_test("t", 1, True)
        store.list_tasks()
        store.get_task_detail("t")
        store.get_stats()
        store.delete_task("t")
        assert calls["n"] > 0, (
            "every database touch must be opened through safe_connect; a path "
            "that reached sqlite directly would bypass the encrypted-connection "
            "seam and never increment this counter"
        )
    finally:
        restore()


def test_h4_missing_connection_module_falls_back_and_still_connects(tmp_path):
    ch, restore = _load(tmp_path, block_db=True)
    try:
        assert callable(ch._safe_connect), (
            "with the connection module unreachable the import fallback binds a "
            "plain sqlite connection factory"
        )
        store = _store(ch, tmp_path)
        store.record_task_start("t", "via_fallback", model="m")
        d = store.get_task_detail("t")
        assert d is not None and d.task_text == "via_fallback", (
            "the store initialises and round-trips a task on the fallback seam"
        )
    finally:
        restore()


def test_h5_construction_creates_schema_and_is_idempotent(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        _store(ch, tmp_path)
        raw = _raw(tmp_path)
        names = {
            r[0] for r in raw.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        raw.close()
        assert {"tasks", "steps", "tests", "checkpoints", "working_memory"} <= names, (
            "construction lays out every table the store writes to"
        )
        # A second store on the SAME path must not raise on the existing schema.
        again = _store(ch, tmp_path)
        again.record_task_start("t", "x")
        assert again.count_tasks() == 1, (
            "re-opening an existing database is idempotent and still writes"
        )
    finally:
        restore()


# =========================================================================
# Task lifecycle and the enabled gate
# =========================================================================

def test_h6_record_task_start_replaces_on_same_id(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("t", "first", project_path="/a", model="m1")
        store.record_task_start("t", "second", project_path="/b", model="m2")
        assert store.count_tasks() == 1, "a repeat of the same id replaces rather than duplicates"
        d = store.get_task_detail("t")
        assert d.task_text == "second" and d.model == "m2" and d.project_path == "/b", (
            "the replacement carries the latest values"
        )
        assert d.status == "started" and d.completed_at is None
    finally:
        restore()


def test_h7_update_task_status_stamps_completion_only_on_terminal(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("t", "x")
        store.update_task_status("t", "running")
        assert store.get_task_detail("t").completed_at is None, (
            "a non-terminal status leaves the completion timestamp unset"
        )
        store.update_task_status("t", "completed", plan_json={"steps": [1, 2]})
        d = store.get_task_detail("t")
        assert d.status == "completed"
        assert isinstance(d.completed_at, float) and d.completed_at > 0, (
            "a terminal status stamps the completion time"
        )
        assert d.plan_json == {"steps": [1, 2]}, "the supplied plan is stored and decoded back"
    finally:
        restore()


def test_h8_disabled_store_drops_every_write(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        cfg = _cfg_file(tmp_path, enabled=False)
        store = _store(ch, tmp_path, config_path=cfg)
        store.record_task_start("t", "x")
        store.record_step("t", 1)
        store.record_test("t", 1, True)
        store.record_checkpoint("t", "apply")
        store.save_working_memory("t", {"k": 1})
        assert store.count_tasks() == 0, "a disabled store records no task"
        assert store.get_task_detail("t") is None
        assert store.load_working_memory("t") is None, (
            "every write is a no-op while the store is disabled"
        )
    finally:
        restore()


# =========================================================================
# Steps, tests, checkpoints, working memory
# =========================================================================

def test_h9_record_step_admits_multiple_and_orders_by_number(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("t", "x")
        store.record_step("t", 2, step_type="edit", file_path="b.py")
        store.record_step("t", 1, step_type="read", file_path="a.py")
        store.record_step("t", 1, step_type="read", file_path="a.py")
        steps = store.get_task_detail("t").steps
        assert [s["step_number"] for s in steps] == [1, 1, 2], (
            "steps are listed by step number and a repeat number is a second row, "
            "not a replacement"
        )
    finally:
        restore()


def test_h10_record_test_stores_flag_as_int_and_reads_back_bool(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("t", "x")
        store.record_test("t", 1, True, output="ok")
        store.record_test("t", 2, False, output="boom")
        raw = _raw(tmp_path)
        stored = [
            r[0] for r in raw.execute(
                "SELECT passed FROM tests WHERE task_id='t' ORDER BY run_number"
            ).fetchall()
        ]
        raw.close()
        assert stored == [1, 0], "the boolean is persisted as one or zero"
        tests = store.get_task_detail("t").tests
        assert tests[0]["passed"] is True and tests[1]["passed"] is False, (
            "the stored integer is decoded back to a real bool on read"
        )
    finally:
        restore()


def test_h11_last_checkpoint_is_the_most_recent_or_none(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("t", "x")
        assert store.get_last_checkpoint("t") is None, (
            "a task with no checkpoint resolves to None"
        )
        assert store.get_last_checkpoint("missing") is None, (
            "a checkpoint on a task that does not exist does not resolve -- the "
            "lookup joins the task in"
        )
        store.record_checkpoint("t", "plan", current_step=1, plan_snapshot={"a": 1})
        store.record_checkpoint("t", "apply", current_step=5, plan_snapshot={"b": 2})
        raw = _raw(tmp_path)
        raw.execute("UPDATE checkpoints SET timestamp=1000 WHERE phase='plan'")
        raw.execute("UPDATE checkpoints SET timestamp=2000 WHERE phase='apply'")
        raw.commit()
        raw.close()
        cp = store.get_last_checkpoint("t")
        assert cp is not None and cp.phase == "apply" and cp.current_step == 5, (
            "the latest checkpoint by timestamp is the one returned"
        )
        assert cp.plan_json == {"b": 2}, "its plan snapshot is decoded back"
    finally:
        restore()


def test_h12_working_memory_upserts_and_delete_is_gated(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("t", "x")
        assert store.load_working_memory("t") is None
        store.save_working_memory("t", {"k": 1})
        store.save_working_memory("t", {"k": 2})
        assert store.load_working_memory("t") == {"k": 2}, (
            "a second save replaces rather than appends"
        )
        raw = _raw(tmp_path)
        raw.execute("UPDATE working_memory SET memory_json='{bad' WHERE task_id='t'")
        raw.commit()
        raw.close()
        assert store.load_working_memory("t") is None, (
            "a memory blob that will not parse decodes to None rather than raising"
        )
        assert store.delete_working_memory("t") is True
        assert store.delete_working_memory("missing") is False
        # Delete is gated: repopulate, disable, and the delete is refused.
        store.save_working_memory("t", {"k": 3})
        store._enabled = False
        assert store.delete_working_memory("t") is False, (
            "delete_working_memory returns False without touching a row while the "
            "store is disabled"
        )
        store._enabled = True
        assert store.load_working_memory("t") == {"k": 3}, "the disabled delete left the row intact"
    finally:
        restore()


# =========================================================================
# Query methods
# =========================================================================

def test_h13_resumable_tasks_exclude_terminal_and_aggregate(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("open", "x", model="m")
        store.record_step("open", 1, status="completed")
        store.record_step("open", 2, status="pending")
        store.record_test("open", 1, True)
        store.record_task_start("run", "y")
        store.update_task_status("run", "running")
        store.record_task_start("done", "z")
        store.update_task_status("done", "completed")
        _touch_created(tmp_path, "open", 200)
        _touch_created(tmp_path, "run", 100)
        rows = store.get_resumable_tasks()
        ids = [r.task_id for r in rows]
        assert ids == ["open", "run"], (
            "only non-terminal tasks are resumable, most recent first; a completed "
            "task is excluded"
        )
        top = rows[0]
        assert top.step_count == 2 and top.completed_steps == 1 and top.test_runs == 1, (
            "the summary aggregates step, completed-step, and test counts in SQL"
        )
        assert top.last_passed is True
    finally:
        restore()


def test_h14_list_tasks_paginates_and_filters_by_status(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        for i, tid in enumerate(["a", "b", "c", "d"]):
            store.record_task_start(tid, tid)
            _touch_created(tmp_path, tid, 100 + i)
        store.update_task_status("b", "completed")
        page = store.list_tasks(limit=2, offset=1)
        assert [t.task_id for t in page] == ["c", "b"], (
            "the listing is ordered by created_at desc and honours limit and offset"
        )
        done = store.list_tasks(status="completed")
        assert [t.task_id for t in done] == ["b"], "a status filter narrows the listing"
        assert store.list_tasks(status="nope") == []
    finally:
        restore()


def test_h15_task_detail_assembles_children_or_none(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        assert store.get_task_detail("missing") is None
        store.record_task_start("t", "x", model="m")
        store.record_step("t", 2)
        store.record_step("t", 1)
        store.record_test("t", 1, True)
        store.record_test("t", 2, False)
        store.record_checkpoint("t", "apply", current_step=3)
        d = store.get_task_detail("t")
        assert {s["step_number"] for s in d.steps} == {1, 2} and len(d.steps) == 2, (
            "the detail carries every step of the task"
        )
        assert len(d.tests) == 2, "and every test run"
        assert len(d.checkpoints) == 1 and d.checkpoints[0]["phase"] == "apply", (
            "and every checkpoint, with its phase"
        )
    finally:
        restore()


def test_h16_count_tasks_total_and_by_status(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        assert store.count_tasks() == 0
        store.record_task_start("a", "x")
        store.record_task_start("b", "y")
        store.record_task_start("c", "z")
        store.update_task_status("a", "completed")
        store.update_task_status("b", "completed")
        assert store.count_tasks() == 3
        assert store.count_tasks(status="completed") == 2, "the count narrows to a status"
        assert store.count_tasks(status="started") == 1
    finally:
        restore()


# =========================================================================
# Deletes, foreign keys, and maintenance
# =========================================================================

def test_h17_delete_task_cascades_every_child_and_ignores_the_gate(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        assert store.delete_task("missing") is False
        store.record_task_start("t", "x")
        store.record_step("t", 1)
        store.record_test("t", 1, True)
        store.record_checkpoint("t", "apply")
        store.save_working_memory("t", {"k": 1})
        assert store.delete_task("t") is True
        assert store.get_task_detail("t") is None and store.count_tasks() == 0
        raw = _raw(tmp_path)
        remaining = {
            "steps": raw.execute("SELECT COUNT(*) FROM steps WHERE task_id='t'").fetchone()[0],
            "tests": raw.execute("SELECT COUNT(*) FROM tests WHERE task_id='t'").fetchone()[0],
            "checkpoints": raw.execute("SELECT COUNT(*) FROM checkpoints WHERE task_id='t'").fetchone()[0],
            "working_memory": raw.execute("SELECT COUNT(*) FROM working_memory WHERE task_id='t'").fetchone()[0],
        }
        raw.close()
        assert remaining == {"steps": 0, "tests": 0, "checkpoints": 0, "working_memory": 0}, (
            "delete drives a manual cascade through every child table"
        )
        # Not gated: a disabled store still deletes.
        store.record_task_start("t2", "y")
        store._enabled = False
        assert store.delete_task("t2") is True and store.count_tasks() == 0, (
            "delete_task runs regardless of the enabled flag"
        )
    finally:
        restore()


def test_h18_foreign_keys_are_declarative_not_enforced(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        # A child written against a task id that was never started is admitted:
        # the connection factory never turns foreign keys on.
        store.record_step("ghost", 1, step_type="edit")
        raw = _raw(tmp_path)
        orphans = raw.execute(
            "SELECT COUNT(*) FROM steps WHERE task_id='ghost'"
        ).fetchone()[0]
        raw.close()
        assert orphans == 1, (
            "a step against a missing task lands as an orphan; enforcement is off"
        )
        assert store.get_task_detail("ghost") is None, (
            "the orphan is invisible through the task-keyed detail lookup"
        )
    finally:
        restore()


def test_h19_prune_by_age_and_by_count(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        import time
        # Age: a task older than the retention window is pruned with its children.
        age_cfg = _cfg_file(tmp_path, retention_days=30)
        store = _store(ch, tmp_path, db="age.db", config_path=age_cfg)
        store.record_task_start("old", "x")
        store.record_step("old", 1)
        store.record_task_start("new", "y")
        _touch_created(tmp_path, "old", time.time() - 90 * 86400, db="age.db")
        assert store.prune() == 1, "a task past the retention window is pruned"
        assert store.count_tasks() == 1 and store.get_task_detail("new") is not None
        raw = _raw(tmp_path, "age.db")
        left = raw.execute("SELECT COUNT(*) FROM steps WHERE task_id='old'").fetchone()[0]
        raw.close()
        assert left == 0, "the pruned task took its steps with it"

        # Count: the oldest overflow beyond max_tasks is pruned. The timestamps
        # stay inside the retention window so only the count rule fires.
        cnt_cfg = _cfg_file(tmp_path, name="cnt.yaml", max_tasks=2)
        s2 = _store(ch, tmp_path, db="cnt.db", config_path=cnt_cfg)
        now = time.time()
        for i, tid in enumerate(["a", "b", "c"]):
            s2.record_task_start(tid, tid)
            _touch_created(tmp_path, tid, now - (3 - i), db="cnt.db")
        assert s2.prune() == 1, "the single overflow beyond the cap is pruned"
        assert {t.task_id for t in s2.list_tasks()} == {"b", "c"}, "the oldest is the one dropped"
    finally:
        restore()


def test_h20_batch_delete_by_ids_counts_real_deletions(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        assert store.batch_delete_by_ids([]) == 0, "an empty id list deletes nothing"
        for tid in ("a", "b", "c"):
            store.record_task_start(tid, tid)
        store.record_step("a", 1)
        deleted = store.batch_delete_by_ids(["a", "b", "does-not-exist"])
        assert deleted == 2, (
            "the count reflects rows actually removed; a missing id contributes zero"
        )
        assert store.count_tasks() == 1 and store.get_task_detail("c") is not None
        raw = _raw(tmp_path)
        left = raw.execute("SELECT COUNT(*) FROM steps WHERE task_id='a'").fetchone()[0]
        raw.close()
        assert left == 0, "the batch cascade cleared the children too"
    finally:
        restore()


def test_h21_batch_delete_before_date(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        for i, tid in enumerate(["old1", "old2", "keep"]):
            store.record_task_start(tid, tid)
            _touch_created(tmp_path, tid, 100 + i * 100)
        store.record_step("old1", 1)
        removed = store.batch_delete_before_date(250)
        assert removed == 2, "every task created before the cutoff is removed and counted"
        assert {t.task_id for t in store.list_tasks()} == {"keep"}
        raw = _raw(tmp_path)
        left = raw.execute("SELECT COUNT(*) FROM steps WHERE task_id='old1'").fetchone()[0]
        raw.close()
        assert left == 0
    finally:
        restore()


# =========================================================================
# Export
# =========================================================================

def test_h22_export_json_nests_children_and_computes_fields(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("a", "x", model="m")
        store.record_step("a", 1)
        store.record_step("a", 2)
        store.record_test("a", 1, True)
        store.record_test("a", 2, False)
        store.record_task_start("b", "y")            # no steps, no tests, no completion
        store.record_task_start("c", "z")            # completed before it started
        _touch_created(tmp_path, "a", 100)
        _touch_created(tmp_path, "b", 200)
        _touch_created(tmp_path, "c", 300)
        raw = _raw(tmp_path)
        raw.execute("UPDATE tasks SET completed_at=160 WHERE task_id='a'")   # after created
        raw.execute("UPDATE tasks SET completed_at=250 WHERE task_id='c'")   # before created (300)
        raw.commit()
        raw.close()
        rows = {r["task_id"]: r for r in store.export_tasks_json()}
        a = rows["a"]
        assert a["step_count"] == 2 and a["test_runs"] == 2
        assert a["pass_rate"] == 50.0, "the pass rate is the share of passing runs"
        assert a["duration_seconds"] == 60.0, "duration is completed minus created when positive"
        assert a["tests"][0]["passed"] is True, "nested test flags are real bools"
        b = rows["b"]
        assert b["step_count"] == 0 and b["test_runs"] == 0 and b["pass_rate"] == 0.0, (
            "a task with no runs reports a zero pass rate rather than dividing by zero"
        )
        assert b["duration_seconds"] is None, "a task that never completed has no duration"
        assert rows["c"]["duration_seconds"] is None, (
            "a completion that predates the start yields no duration"
        )
    finally:
        restore()


def test_h23_export_csv_rows_are_flat_and_aggregated(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("a", "x", model="m")
        store.record_step("a", 1)
        store.record_test("a", 1, True)
        store.record_test("a", 2, False)
        store.record_task_start("b", "y")
        _touch_created(tmp_path, "a", 100)
        _touch_created(tmp_path, "b", 200)
        raw = _raw(tmp_path)
        raw.execute("UPDATE tasks SET completed_at=140 WHERE task_id='a'")
        raw.commit()
        raw.close()
        rows = {r["task_id"]: r for r in store.export_tasks_csv_rows()}
        a = rows["a"]
        assert a["step_count"] == 1 and a["test_runs"] == 2 and a["pass_rate"] == 50.0
        assert a["duration_seconds"] == 40.0
        assert "steps" not in a and "tests" not in a, "csv rows carry no nested structures"
        b = rows["b"]
        assert b["test_runs"] == 0 and b["pass_rate"] == 0.0 and b["duration_seconds"] is None, (
            "a task with no runs coalesces to a zero rate and a null duration"
        )
    finally:
        restore()


# =========================================================================
# Stats and analytics
# =========================================================================

def test_h24_get_stats_counts_every_table(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("a", "x")
        store.record_task_start("b", "y")
        store.update_task_status("a", "completed")
        store.record_step("a", 1)
        store.record_step("a", 2)
        store.record_test("a", 1, True)
        store.record_test("a", 2, False)
        store.record_checkpoint("a", "apply")
        st = store.get_stats()
        assert st["total_tasks"] == 2
        assert st["by_status"] == {"completed": 1, "started": 1}, "tasks are tallied by status"
        assert st["total_steps"] == 2 and st["total_tests"] == 2
        assert st["passed_tests"] == 1 and st["total_checkpoints"] == 1
    finally:
        restore()


def test_h25_success_rate_by_model_excludes_blank_model(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("a", "x", model="m1")
        store.record_task_start("b", "y", model="m1")
        store.update_task_status("a", "completed")
        store.record_task_start("c", "z", model="m2")
        store.record_task_start("d", "w", model="")     # blank model
        rows = store.get_success_rate_by_model()
        by_model = {r["model"]: r for r in rows}
        assert "" not in by_model, "a task with no model name is excluded from the breakdown"
        assert by_model["m1"]["success_rate"] == 50.0 and by_model["m1"]["total"] == 2
        assert by_model["m2"]["success_rate"] == 0.0
        assert [r["model"] for r in rows] == ["m1", "m2"], "models are ordered by success rate desc"
    finally:
        restore()


def test_h26_avg_steps_by_model_excludes_stepless_tasks(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("a", "x", model="m1")
        store.record_step("a", 1)
        store.record_step("a", 2)
        store.record_task_start("b", "y", model="m2")   # no steps at all
        rows = {r["model"]: r for r in store.get_avg_steps_by_model()}
        assert "m2" not in rows, (
            "a model whose tasks recorded no steps does not appear -- the join drops it"
        )
        assert rows["m1"]["avg_steps"] == 2.0 and rows["m1"]["task_count"] == 1
        assert rows["m1"]["min_steps"] == 2 and rows["m1"]["max_steps"] == 2
    finally:
        restore()


def test_h27_avg_steps_overall_is_null_on_empty(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        empty = store.get_avg_steps_overall()
        assert empty["task_count"] == 0
        assert empty["avg_steps"] is None and empty["min_steps"] is None and empty["max_steps"] is None, (
            "over an empty ledger the average comes back null, not zero: the "
            "zero-default guard checks for an absent row, and an aggregate always "
            "returns one -- this pins the null the code produces"
        )
        store.record_task_start("a", "x")
        store.record_step("a", 1)
        store.record_step("a", 2)
        store.record_step("a", 3)
        got = store.get_avg_steps_overall()
        assert got["avg_steps"] == 3.0 and got["task_count"] == 1, (
            "once a task has steps the overall average is computed"
        )
    finally:
        restore()


def test_h28_failure_reasons_group_last_phase_of_failed_tasks(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("f1", "x")
        store.record_checkpoint("f1", "apply")
        store.update_task_status("f1", "failed")
        store.record_task_start("f2", "y")           # failed, no checkpoint
        store.update_task_status("f2", "failed")
        store.record_task_start("ok", "z")
        store.update_task_status("ok", "completed")
        rows = {r["failure_phase"]: r["count"] for r in store.get_failure_reasons()}
        assert rows.get("apply") == 1, "a failed task's last checkpoint phase is its failure point"
        assert rows.get("unknown") == 1, "a failed task with no checkpoint falls into 'unknown'"
        assert "completed" not in str(rows) and sum(rows.values()) == 2, (
            "only failed tasks are counted"
        )
    finally:
        restore()


def test_h29_time_trends_only_positive_durations(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("done", "x", model="m")
        store.record_task_start("open", "y")         # never completed
        store.record_task_start("bad", "z")          # completed before started
        _touch_created(tmp_path, "done", 100)
        _touch_created(tmp_path, "bad", 300)
        raw = _raw(tmp_path)
        raw.execute("UPDATE tasks SET completed_at=180 WHERE task_id='done'")
        raw.execute("UPDATE tasks SET completed_at=250 WHERE task_id='bad'")
        raw.commit()
        raw.close()
        rows = store.get_time_trends()
        assert [r["task_id"] for r in rows] == ["done"], (
            "only a task with both timestamps and a completion after its start appears"
        )
        assert rows[0]["duration_seconds"] == 80.0
    finally:
        restore()


def test_h30_test_pass_rate_excludes_tasks_without_runs(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        store.record_task_start("a", "x", model="m")
        store.record_test("a", 1, True)
        store.record_test("a", 2, True)
        store.record_test("a", 3, False)
        store.record_task_start("b", "y")            # no test runs
        rows = {r["task_id"]: r for r in store.get_test_pass_rate_per_task()}
        assert "b" not in rows, "a task with no test runs is excluded from the pass-rate table"
        a = rows["a"]
        assert a["total_runs"] == 3 and a["passed_runs"] == 2
        assert a["pass_rate"] == 66.7, "the pass rate is rounded to one decimal"
    finally:
        restore()


def test_h31_steps_distribution_buckets_by_count(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        for tid, n in (("a", 1), ("b", 1), ("c", 2)):
            store.record_task_start(tid, tid)
            for i in range(n):
                store.record_step(tid, i + 1)
        dist = {r["step_count"]: r["task_count"] for r in store.get_steps_distribution()}
        assert dist == {1: 2, 2: 1}, (
            "the distribution counts how many tasks recorded each number of steps"
        )
    finally:
        restore()


def test_h32_get_analytics_combines_the_payload(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        empty = store.get_analytics()
        assert empty["total_tasks"] == 0 and empty["overall_success_rate"] == 0.0, (
            "an empty ledger yields a zero overall success rate rather than dividing "
            "by zero"
        )
        for key in (
            "success_rate_by_model", "avg_steps_by_model", "avg_steps_overall",
            "failure_reasons", "time_trends", "test_pass_rate_per_task",
            "steps_distribution",
        ):
            assert key in empty, "the payload carries every analytics section"
        store.record_task_start("a", "x", model="m")
        store.update_task_status("a", "completed")
        store.record_task_start("b", "y", model="m")
        got = store.get_analytics()
        assert got["total_tasks"] == 2 and got["completed_tasks"] == 1
        assert got["overall_success_rate"] == 50.0
        assert got["success_rate_by_model"][0]["model"] == "m"
    finally:
        restore()


# =========================================================================
# Injection safety and bounded writes
# =========================================================================

def test_h33_metacharacters_round_trip_as_data(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)
        evil_id = "t'); DROP TABLE tasks;--"
        evil_text = 'do "this"; DELETE FROM steps;--'
        evil_model = "m'); DROP TABLE tests;--"
        store.record_task_start(evil_id, evil_text, model=evil_model)
        store.record_step(evil_id, 1, result="r'); DROP TABLE checkpoints;--")
        d = store.get_task_detail(evil_id)
        assert d is not None and d.task_text == evil_text and d.model == evil_model, (
            "an id, text, and model carrying SQL metacharacters round-trip as data; "
            "that only holds under parameterisation"
        )
        # The tables are still standing.
        assert store.count_tasks() == 1 and len(d.steps) == 1
    finally:
        restore()


def test_h34_stored_blobs_are_truncated_to_their_ceilings(tmp_path):
    ch, restore = _load(tmp_path)
    try:
        store = _store(ch, tmp_path)   # defaults: text 2000, output 10000, plan/memory 50000
        store.record_task_start("t", "T" * 5000)
        store.record_step("t", 1, result="R" * 20000)
        store.record_test("t", 1, True, output="O" * 20000)
        store.save_working_memory("t", {"blob": "M" * 80000})
        d = store.get_task_detail("t")
        assert len(d.task_text) == 2000, "the task text is capped at its ceiling before it is stored"
        assert len(d.steps[0]["result"]) == 10000, "step output is capped"
        assert len(d.tests[0]["output"]) == 10000, "test output is capped"
        mem = store.load_working_memory("t")
        import json
        assert len(json.dumps(mem)) <= 50000, "the memory blob is capped before it reaches disk"
    finally:
        restore()
