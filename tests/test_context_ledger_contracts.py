#!/usr/bin/env python3
"""What the context ledger promises about keeping figures and nothing else.

The ledger is the sink where each request's measurements land: token
figures per zone, cache hit labels, retrieval scores, the admission
verdict. Three promises carry the design, and every contract here leans
on one of them.

Numbers and labels only. The schema's column set is pinned column for
column -- there is no column a prompt, a response or a document could
land in, string fields are clipped to tight bounds, zone reports are
reduced to five allowed keys with any narrative field dropped, and a
hostile value in a numeric slot becomes NULL rather than a row lost or a
statement executed. Values carrying SQL metacharacters round-trip as
data, because every statement is parameterised.

Fail-open writing, fail-quiet reading. ``record()`` returns False on any
fault -- a seam that never initialised, a factory that starts raising
mid-life -- and never raises, because an observability sink that can take
the chat path down is worse than no sink. Unknown keyword fields are
ignored rather than rejected, so an emitter one release ahead degrades to
a partial row instead of a lost one. Reads on an empty, absent or broken
store come back empty or None, never as an exception.

A bounded table. Every insert enforces the retention cap by pruning the
oldest overflow, newest rows always survive, and a non-positive or
unparseable cap clamps to a floor of one row -- misconfiguration shrinks
the window, it never produces an unbounded log and never a self-erasing
one.

All persistence flows through the project's safe_connect seam, proven by
a counting stand-in: every operation that touches disk crosses it, and
with the seam absent the ledger reports itself unavailable instead of
reaching for a bare connection. The shared instance is a plain singleton
with an explicit reset, and its default file lands under the seeded data
directory.

Loaded through the shared isolation window. The connection factory and
the data directory are the only project seams the module reaches; each is
seeded or blocked, so no real database and no real key management is ever
touched by these contracts.
"""

import sqlite3
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.context_ledger"


def _default_connect(path, **kwargs):
    kwargs.pop("check_same_thread", None)
    return sqlite3.connect(str(path), check_same_thread=False)


def _load(data_dir, *, connect=_default_connect, block_db=False):
    """Load the ledger module in isolation.

    data_dir -- seeded as ``config.DATA_DIR``; the default database path
                resolves under it.
    connect  -- the stand-in ``safe_connect``. The default is a plain
                sqlite connection; a caller can pass a counting or a
                raising factory to pin the seam itself.
    block_db -- when true, the db_utils name is blocked entirely so the
                seam-absent posture runs.
    """
    seeded = {}
    blocked = []

    if block_db:
        blocked.append("opti_oignon.db_utils")
    else:
        du = types.ModuleType("opti_oignon.db_utils")
        du.safe_connect = connect
        seeded["opti_oignon.db_utils"] = du

    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = Path(data_dir)
    seeded["opti_oignon.config"] = cfg

    loaded, restore = isolate(
        targets={_TARGET: source("context_ledger.py")},
        blocked=blocked,
        seeded=seeded,
    )
    return loaded[_TARGET], restore


def _fill(ledger, n, prefix="r", **overrides):
    for i in range(n):
        fields = {
            "request_id": f"{prefix}{i}",
            "model": "m",
            "outcome": "completed",
            "tokens_total": 100 + i,
        }
        fields.update(overrides)
        assert ledger.record(**fields) is True
    return ledger


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------

def test_g1_initialisation_flows_through_the_seeded_seam(tmp_path):
    seen = []

    def counting(path, **kwargs):
        seen.append(str(path))
        return _default_connect(path, **kwargs)

    mod, restore = _load(tmp_path, connect=counting)
    try:
        ledger = mod.ContextLedger(db_path=tmp_path / "ledger.db")
        assert ledger.available is True
        assert seen, "initialisation never crossed the connection seam"
        assert all(p.endswith("ledger.db") for p in seen)
    finally:
        restore()


def test_g2_every_disk_operation_crosses_the_connection_seam(tmp_path):
    count = {"n": 0}

    def counting(path, **kwargs):
        count["n"] += 1
        return _default_connect(path, **kwargs)

    mod, restore = _load(tmp_path, connect=counting)
    try:
        ledger = mod.ContextLedger(db_path=tmp_path / "ledger.db")
        after_init = count["n"]
        assert after_init >= 1

        ledger.record(request_id="a", model="m", outcome="completed")
        after_write = count["n"]
        assert after_write > after_init

        ledger.recent(limit=5)
        after_recent = count["n"]
        assert after_recent > after_write

        ledger.get("a")
        after_get = count["n"]
        assert after_get > after_recent

        ledger.stats()
        assert count["n"] > after_get
    finally:
        restore()


def test_g3_sql_metacharacters_round_trip_as_data(tmp_path):
    mod, restore = _load(tmp_path)
    try:
        ledger = mod.ContextLedger(db_path=tmp_path / "ledger.db")
        hostile = "x'); DROP TABLE context_ledger_entries;--"
        assert ledger.record(
            request_id=hostile,
            model="m'--",
            outcome='out";--',
            gov_reason=hostile,
        ) is True
        row = ledger.get(hostile)
        assert row is not None
        assert row["request_id"] == hostile
        assert row["model"] == "m'--"
        assert row["gov_reason"] == hostile
        assert ledger.stats()["rows"] == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# The bounded table
# ---------------------------------------------------------------------------

def test_g4_retention_prunes_the_oldest_and_keeps_the_newest(tmp_path):
    mod, restore = _load(tmp_path)
    try:
        ledger = _fill(
            mod.ContextLedger(db_path=tmp_path / "ledger.db", max_rows=3), 5
        )
        rows = ledger.recent(limit=50)
        assert [r["request_id"] for r in rows] == ["r4", "r3", "r2"]
        assert ledger.get("r0") is None
        assert ledger.get("r4")["tokens_total"] == 104
    finally:
        restore()


def test_g5_non_positive_caps_clamp_to_a_floor_of_one_row(tmp_path):
    mod, restore = _load(tmp_path)
    try:
        for cap in (0, -5):
            ledger = _fill(
                mod.ContextLedger(
                    db_path=tmp_path / f"cap{cap}.db", max_rows=cap
                ),
                3,
            )
            assert ledger.max_rows == 1
            rows = ledger.recent(limit=50)
            assert [r["request_id"] for r in rows] == ["r2"]
    finally:
        restore()


def test_g6_unparseable_cap_falls_back_to_the_default(tmp_path):
    mod, restore = _load(tmp_path)
    try:
        ledger = mod.ContextLedger(
            db_path=tmp_path / "ledger.db", max_rows="abc"
        )
        assert ledger.max_rows == mod.DEFAULT_MAX_ROWS
    finally:
        restore()


# ---------------------------------------------------------------------------
# Fail-open writing
# ---------------------------------------------------------------------------

def test_g7_absent_seam_means_unavailable_not_a_bare_connection(tmp_path):
    mod, restore = _load(tmp_path, block_db=True)
    try:
        assert mod.SAFE_CONNECT_AVAILABLE is False
        ledger = mod.ContextLedger(db_path=tmp_path / "ledger.db")
        assert ledger.available is False
        assert ledger.record(request_id="a", model="m", outcome="o") is False
        assert ledger.recent() == []
        assert ledger.get("a") is None
        assert ledger.stats()["available"] is False
        assert not (tmp_path / "ledger.db").exists()
    finally:
        restore()


def test_g8_factory_raising_mid_life_turns_writes_into_false_never_raises(tmp_path):
    state = {"healthy": True}

    def flaky(path, **kwargs):
        if not state["healthy"]:
            raise RuntimeError("disk gone")
        return _default_connect(path, **kwargs)

    mod, restore = _load(tmp_path, connect=flaky)
    try:
        ledger = mod.ContextLedger(db_path=tmp_path / "ledger.db")
        assert ledger.record(request_id="a", model="m", outcome="o") is True
        state["healthy"] = False
        assert ledger.record(request_id="b", model="m", outcome="o") is False
    finally:
        restore()


def test_g9_hostile_field_types_coerce_to_null_and_the_row_still_lands(tmp_path):
    mod, restore = _load(tmp_path)
    try:
        ledger = mod.ContextLedger(db_path=tmp_path / "ledger.db")
        assert ledger.record(
            request_id="rx",
            model="m",
            outcome="completed",
            tokens_total="not-an-int",
            tokens_system=[1, 2],
            cache_similarity="not-a-float",
            zones="garbage-not-a-list",
            duration_ms=object(),
        ) is True
        row = ledger.get("rx")
        assert row is not None
        assert row["tokens_total"] is None
        assert row["tokens_system"] is None
        assert row["cache_similarity"] is None
        assert row["zones"] == []
        assert row["duration_ms"] is None
    finally:
        restore()


def test_g10_unknown_keyword_fields_are_ignored_not_rejected(tmp_path):
    mod, restore = _load(tmp_path)
    try:
        ledger = mod.ContextLedger(db_path=tmp_path / "ledger.db")
        assert ledger.record(
            request_id="fw",
            model="m",
            outcome="completed",
            some_future_field=123,
            another_one={"deep": True},
        ) is True
        assert ledger.get("fw") is not None
    finally:
        restore()


# ---------------------------------------------------------------------------
# Numbers and labels only
# ---------------------------------------------------------------------------

def test_g11_column_set_is_pinned_and_strings_clip_to_their_bounds(tmp_path):
    mod, restore = _load(tmp_path)
    try:
        ledger = mod.ContextLedger(db_path=tmp_path / "ledger.db")
        with _default_connect(tmp_path / "ledger.db") as conn:
            cols = [
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(context_ledger_entries)"
                ).fetchall()
            ]
        assert cols == ["id", *mod._COLUMNS]

        long_reason = "y" * 1000
        long_label = "z" * 1000
        long_id = "i" * 1000
        assert ledger.record(
            request_id=long_id,
            model=long_label,
            outcome=long_label,
            gov_reason=long_reason,
            gov_keep_alive=long_label,
        ) is True
        row = ledger.recent(limit=1)[0]
        assert len(row["request_id"]) == mod._ID_MAX
        assert len(row["model"]) == mod._ID_MAX
        assert len(row["outcome"]) == mod._LABEL_MAX
        assert len(row["gov_reason"]) == mod._REASON_MAX
        assert len(row["gov_keep_alive"]) == mod._LABEL_MAX
    finally:
        restore()


def test_g12_zone_reports_reduce_to_allowed_keys_and_the_list_is_capped(tmp_path):
    mod, restore = _load(tmp_path)
    try:
        ledger = mod.ContextLedger(db_path=tmp_path / "ledger.db")
        zones = [
            {
                "zone": "system",
                "budgeted_tokens": 100,
                "actual_tokens": 90,
                "trimmed_tokens": 0,
                "strategy": "fixed",
                "detail": "a narrative sentence that must not be kept",
            },
            {
                "zone": "history",
                "budgeted": 400,
                "actual": 350,
                "trimmed": 50,
                "strategy": "compressed",
            },
            "not-a-dict",
        ] + [{"zone": f"extra{i}"} for i in range(30)]
        assert ledger.record(
            request_id="z", model="m", outcome="completed", zones=zones
        ) is True
        stored = ledger.get("z")["zones"]
        assert len(stored) == mod._ZONES_MAX
        assert stored[0] == {
            "zone": "system",
            "budgeted": 100,
            "actual": 90,
            "trimmed": 0,
            "strategy": "fixed",
        }
        assert stored[1]["budgeted"] == 400
        assert all("detail" not in z for z in stored)
    finally:
        restore()


# ---------------------------------------------------------------------------
# Fail-quiet reading
# ---------------------------------------------------------------------------

def test_g13_recent_orders_newest_first_and_bounds_its_limit(tmp_path):
    mod, restore = _load(tmp_path)
    try:
        ledger = _fill(
            mod.ContextLedger(db_path=tmp_path / "ledger.db", max_rows=100), 6
        )
        top = ledger.recent(limit=2)
        assert [r["request_id"] for r in top] == ["r5", "r4"]
        assert len(ledger.recent(limit="garbage")) == 6
        assert len(ledger.recent(limit=0)) >= 1
        assert len(ledger.recent(limit=10_000)) == 6
    finally:
        restore()


def test_g14_get_answers_the_newest_row_or_none(tmp_path):
    mod, restore = _load(tmp_path)
    try:
        ledger = mod.ContextLedger(db_path=tmp_path / "ledger.db")
        ledger.record(request_id="dup", model="m", outcome="first")
        ledger.record(request_id="dup", model="m", outcome="second")
        assert ledger.get("dup")["outcome"] == "second"
        assert ledger.get("unknown") is None
        assert ledger.get("") is None
    finally:
        restore()


def test_g15_stats_aggregate_and_answer_a_zeroed_shape_when_empty(tmp_path):
    mod, restore = _load(tmp_path)
    try:
        ledger = mod.ContextLedger(db_path=tmp_path / "ledger.db", max_rows=50)
        empty = ledger.stats()
        assert empty["available"] is True
        assert empty["rows"] == 0
        assert empty["outcomes"] == {}
        assert empty["cache_hits"] == 0

        _fill(ledger, 2)
        ledger.record(
            request_id="h",
            model="m",
            outcome="cache_hit",
            cache_hit=True,
            token_method="exact",
            duration_ms=10.0,
        )
        stats = ledger.stats()
        assert stats["rows"] == 3
        assert stats["outcomes"] == {"completed": 2, "cache_hit": 1}
        assert stats["methods"] == {"estimated": 2, "exact": 1}
        assert stats["cache_hits"] == 1
        assert stats["max_rows"] == 50
    finally:
        restore()


def test_g16_reads_on_a_broken_store_come_back_empty_never_raise(tmp_path):
    state = {"healthy": True}

    def flaky(path, **kwargs):
        if not state["healthy"]:
            raise RuntimeError("disk gone")
        return _default_connect(path, **kwargs)

    mod, restore = _load(tmp_path, connect=flaky)
    try:
        ledger = _fill(mod.ContextLedger(db_path=tmp_path / "ledger.db"), 2)
        state["healthy"] = False
        assert ledger.recent() == []
        assert ledger.get("r0") is None
        assert ledger.stats()["available"] is False
    finally:
        restore()


def test_g17_shared_ledger_is_a_singleton_under_the_data_dir_with_a_reset(tmp_path):
    mod, restore = _load(tmp_path)
    try:
        first = mod.get_context_ledger()
        again = mod.get_context_ledger()
        assert first is again
        assert first.available is True
        assert Path(first.db_path).parent == Path(tmp_path)
        assert Path(first.db_path).name == "context_ledger.db"
        mod.reset_context_ledger()
        rebuilt = mod.get_context_ledger()
        assert rebuilt is not first
    finally:
        restore()
