#!/usr/bin/env python3
"""Contracts for what the resource governor learns of a load, and which loads that prices.

After a load the governor learns what the engine reports the model takes:
its total, the context it was loaded at, and now the part of it the GPU
held. Measured on a machine whose card had dropped out for a week (a driver
upgrade without a reboot), the 27B model it then ran on the CPU alone was
learned at 29 GiB for a 17 GiB file: the CPU keeps buffers and copies a GPU
load never takes, and the KV of its 16384 tokens rode in the total. Once the
card was back, the split placement put on the GPU everything the file's
tensors did not explain, some 12 GiB, more than the card had free, and
refused the load ("not one of its layers fits").

  * GL1 -- WHERE IT LAY: the store keeps the GPU's part of a learned cost
    beside its total and context; a store written before gains the column
    in place, its rows kept and saying nothing of where they lay; and the
    attribution after a load writes all three, with the context the engine
    says it loaded.
  * GL2 -- LIKE FOR LIKE: on a machine with a card, a cost learned of a load
    the CPU held alone, or of one whose placement was never kept, prices
    nothing: the next estimate answers. One learned of a load that held part
    of itself on the GPU prices it; with no card in the view, a CPU load's
    cost stands; with no view at all, the stricter rule holds.
  * GL3 -- THE MIGRATION NEVER STOPS THE STORE: an old store that cannot
    gain the column, being read-only, still opens and answers, its costs
    saying nothing of where they lay; and one that gains it from another
    process at the same moment opens with it.
  * GL4 -- THE INCIDENT: a 17 GiB model learned at 29 GiB on the CPU alone,
    asked at 4096 on an 8 GiB card with RAM to spare, is admitted split with
    layers on the GPU.

Each contract opens the shared isolation window of ``tests/_isolation.py`` on
the governor's source, with a seeded ``db_utils``; snapshots are injected and
the store is a temporary file, so no reading of the machine enters a decision.
"""
from __future__ import annotations

import os
import sqlite3
import stat
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_gl1_the_store_keeps_where_a_load_lay_and_an_old_store_gains_the_column": 5.0,
    "test_gl2_a_cost_prices_only_a_load_like_the_one_it_was_learned_of": 5.0,
    "test_gl3_the_migration_never_stops_the_store": 5.0,
    "test_gl4_a_model_learned_on_the_cpu_alone_is_split_once_the_card_is_back": 5.0,
}

_RG = "opti_oignon.resource_governor"
_GIB = 1024 ** 3


def _db_utils():
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **kw: sqlite3.connect(str(p), check_same_thread=kw.get("check_same_thread", False))
    return db


def _open():
    loaded, restore = isolate(
        targets={_RG: source("resource_governor.py")},
        blocked=("opti_oignon.context_manager", "opti_oignon.emergency_stop"),
        seeded={"opti_oignon.db_utils": _db_utils()},
    )
    return loaded[_RG], restore


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class _Hardware:
    def others_cpu_pressure(self):
        return None

    def pressure(self):
        return None

    def placement(self):
        return None


class _Engine:
    """An engine that tells each model's file size, as Ollama's model_info does, and nothing else."""

    name = "ollama"

    def __init__(self, sizes):
        self._sizes = sizes

    def model_info(self, model):
        size = self._sizes.get(model)
        if size is None:
            return None
        return types.SimpleNamespace(parameter_size=None, quantization_level=None, path=None, size=size, extra={})


class _Registry:
    def __init__(self, *engines):
        self._engines = list(engines)

    def backends(self):
        return list(self._engines)


def _governor(rg, *, db_path=None, registry=None):
    """A real governor on its shipped file, with a store of its own and only the engines handed in to ask."""
    clock = _Clock()
    gov = rg.ResourceGovernor(
        config_path="/nonexistent-resource-governor-config",
        db_path=db_path or str(Path(tempfile.mkdtemp()) / "governor.db"),
        warmup=None,
        registry=registry,
        clock=clock,
        meminfo_path="/nonexistent-meminfo",
        hardware=_Hardware(),
    )
    gov._config = rg.load_config(Path(source("config", "resource_governor.yaml")))
    gov.kv_geometry = lambda model, engine=None: None
    return gov


def _view(rg, *, capacity, in_use=0.0, loaded=(), ram_mb=64000.0, total_mb=96000.0):
    return dict(taken_at=1000.0, ttl_s=9999.0, loaded=list(loaded), capacity_gb=capacity, vram_in_use_gb=in_use,
                ram_available_mb=ram_mb, ram_total_mb=total_mb, sources=["S1"])


def _snap(rg, gov, **view):
    gov._snapshot = rg.ResourceSnapshot(**_view(rg, **view))


# ---------------------------------------------------------------------------
# GL1 -- where it lay
# ---------------------------------------------------------------------------
def test_gl1_the_store_keeps_where_a_load_lay_and_an_old_store_gains_the_column():
    """GL1: the GPU's part is kept and read back; an old store gains it; the attribution writes it."""
    rg, restore = _open()
    try:
        store = rg.AdaptStore(str(Path(tempfile.mkdtemp()) / "fresh.db"))
        store.record_model_cost("m", "sha256:a", 18 * _GIB, num_ctx=4096, observed_at=1.0, on_gpu_bytes=6 * _GIB)
        store.record_model_cost("n", "", 3 * _GIB, num_ctx=8192, observed_at=1.0)
        got = store.get_model_cost("m", "sha256:a")
        assert (got["size_vram_bytes"], got["num_ctx"], got["on_gpu_bytes"]) == (18 * _GIB, 4096, 6 * _GIB), got
        assert store.get_model_cost("n")["on_gpu_bytes"] is None, "a cost recorded with no placement says none"

        old = Path(tempfile.mkdtemp()) / "old.db"
        conn = sqlite3.connect(str(old))
        conn.executescript(
            """CREATE TABLE model_costs (name TEXT NOT NULL, digest TEXT NOT NULL DEFAULT '',
                   size_vram_bytes INTEGER NOT NULL, num_ctx INTEGER, observed_at REAL NOT NULL,
                   PRIMARY KEY (name, digest));
               INSERT INTO model_costs VALUES ('big', '', 31138512896, 16384, 5.0);"""
        )
        conn.commit()
        conn.close()
        migrated = rg.AdaptStore(str(old))
        kept = migrated.get_model_cost("big")
        assert (kept["size_vram_bytes"], kept["num_ctx"], kept["on_gpu_bytes"]) == (31138512896, 16384, None), kept
        migrated.record_model_cost("big", "", 18 * _GIB, num_ctx=4096, observed_at=6.0, on_gpu_bytes=7 * _GIB)
        assert migrated.get_model_cost("big")["on_gpu_bytes"] == 7 * _GIB, "the migrated store writes the column"
        rg.AdaptStore(str(old))
        assert migrated.get_model_cost("big")["on_gpu_bytes"] == 7 * _GIB, "opening it again changes nothing"

        gov = _governor(rg)
        gov._pending_attribution["big"] = 8192
        split = rg.LoadedModelView(name="big", size_vram_bytes=6 * _GIB, size_bytes=18 * _GIB, context_length=4096,
                                   digest="sha256:b")
        cpu = rg.LoadedModelView(name="cpu-only", size_vram_bytes=0, size_bytes=29 * _GIB, context_length=16384)
        gov._pending_attribution["cpu-only"] = 16384
        gov._attribute_pending([split, cpu])
        learned = gov._store.get_model_cost("big", "sha256:b")
        assert (learned["size_vram_bytes"], learned["num_ctx"], learned["on_gpu_bytes"]) == (
            18 * _GIB, 4096, 6 * _GIB), ("the total, the context the engine loaded, the GPU's part", learned)
        assert gov._store.get_model_cost("cpu-only")["on_gpu_bytes"] == 0, "a load the CPU held alone says so"
    finally:
        restore()


# ---------------------------------------------------------------------------
# GL2 -- like for like
# ---------------------------------------------------------------------------
def test_gl2_a_cost_prices_only_a_load_like_the_one_it_was_learned_of():
    """GL2: on a machine with a card, only a cost learned with part of the load on the GPU prices a load."""
    rg, restore = _open()
    try:
        gov = _governor(rg)
        store = gov._store
        store.record_model_cost("gpu", "", 10 * _GIB, num_ctx=8192, observed_at=1.0, on_gpu_bytes=10 * _GIB)
        store.record_model_cost("split", "", 18 * _GIB, num_ctx=8192, observed_at=1.0, on_gpu_bytes=6 * _GIB)
        store.record_model_cost("cpu", "", 29 * _GIB, num_ctx=8192, observed_at=1.0, on_gpu_bytes=0)
        store.record_model_cost("legacy", "", 29 * _GIB, num_ctx=8192, observed_at=1.0)

        _snap(rg, gov, capacity=8.0)
        with_card = {name: gov.estimate_model_vram_gb(name)[1] for name in ("gpu", "split", "cpu", "legacy")}
        assert with_card == {"gpu": "learned", "split": "learned", "cpu": "unknown", "legacy": "unknown"}, (
            "with a card, a CPU load's cost and one of unknown placement price nothing", with_card)

        _snap(rg, gov, capacity=None)
        without = {name: gov.estimate_model_vram_gb(name)[1] for name in ("gpu", "split", "cpu", "legacy")}
        assert set(without.values()) == {"learned"}, ("with no card in the view, every cost stands", without)

        gov._snapshot = None
        unseen = {name: gov.estimate_model_vram_gb(name)[1] for name in ("gpu", "split", "cpu", "legacy")}
        assert unseen == with_card, ("no view is no proof of a machine without a card", unseen)
    finally:
        restore()


# ---------------------------------------------------------------------------
# GL3 -- the migration never stops the store
# ---------------------------------------------------------------------------
def test_gl3_the_migration_never_stops_the_store():
    """GL3: a read-only old store still serves; a column another process added meanwhile is taken as it is."""
    rg, restore = _open()
    try:
        def _old(path):
            conn = sqlite3.connect(str(path))
            conn.executescript(
                """CREATE TABLE model_costs (name TEXT NOT NULL, digest TEXT NOT NULL DEFAULT '',
                       size_vram_bytes INTEGER NOT NULL, num_ctx INTEGER, observed_at REAL NOT NULL,
                       PRIMARY KEY (name, digest));
                   INSERT INTO model_costs VALUES ('big', '', 19327352832, 4096, 5.0);"""
            )
            conn.commit()
            conn.close()

        # An old store as a past version left it: every table written, the GPU's part absent; then read-only.
        locked = Path(tempfile.mkdtemp()) / "locked.db"
        _old(locked)
        first = rg.AdaptStore(str(locked))
        assert first.get_model_cost("big")["on_gpu_bytes"] is None
        conn = sqlite3.connect(str(locked))
        conn.execute("ALTER TABLE model_costs DROP COLUMN on_gpu_bytes")
        conn.commit()
        conn.close()
        os.chmod(locked, stat.S_IRUSR)
        try:
            assert not os.access(locked, os.W_OK), "control: the store is read-only for this process (not run as root)"
            store = rg.AdaptStore(str(locked))
            assert store._gpu_column is False, "read-only, the store could not gain the column, and knows it"
            got = store.get_model_cost("big")
            assert (got["size_vram_bytes"], got["num_ctx"], got["on_gpu_bytes"]) == (19327352832, 4096, None), got
            gov = _governor(rg, db_path=str(locked))
            _snap(rg, gov, capacity=8.0)
            assert gov.estimate_model_vram_gb("big")[1] == "unknown", "on a card, a cost of unknown placement"
        finally:
            os.chmod(locked, stat.S_IRUSR | stat.S_IWUSR)

        raced = Path(tempfile.mkdtemp()) / "raced.db"
        _old(raced)
        connect, asked = rg._safe_connect, []

        class _Stale:
            """A connection that reads the columns as they were before another process added one, once."""

            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, *args):
                if sql.startswith("PRAGMA table_info(model_costs)") and not asked:
                    asked.append(sql)
                    other = sqlite3.connect(str(raced))
                    other.execute("ALTER TABLE model_costs ADD COLUMN on_gpu_bytes INTEGER")
                    other.commit()
                    other.close()
                    return iter([(0, "name"), (1, "digest"), (2, "size_vram_bytes"), (3, "num_ctx"),
                                 (4, "observed_at")])
                return self._conn.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self._conn, name)

        rg._safe_connect = lambda path, **kw: _Stale(connect(path, **kw))
        try:
            store = rg.AdaptStore(str(raced))
        finally:
            rg._safe_connect = connect
        assert asked, "control: the other process added the column between the reading and the adding"
        store.record_model_cost("big", "", 18 * _GIB, num_ctx=4096, observed_at=6.0, on_gpu_bytes=7 * _GIB)
        assert store.get_model_cost("big")["on_gpu_bytes"] == 7 * _GIB, "opened with the column the other added"
    finally:
        restore()


# ---------------------------------------------------------------------------
# GL4 -- the incident
# ---------------------------------------------------------------------------
def test_gl4_a_model_learned_on_the_cpu_alone_is_split_once_the_card_is_back():
    """GL4: 17 GiB of tensors, learned at 29 GiB on the CPU at 16384, asked 4096 on an 8 GiB card: split."""
    rg, restore = _open()
    try:
        gov = _governor(rg, registry=_Registry(_Engine({"big": "17 GB"})))
        # A hybrid model's KV is small: 1 GiB at 16384 tokens, which takes 28 of the 29 learned for weights.
        gov._config.kv_override_models = {"big": 1.0 / 16.0}
        gov._store.record_model_cost("big", "", 29 * _GIB, num_ctx=16384, observed_at=1.0, on_gpu_bytes=0)
        table = types.SimpleNamespace(blocks={i: _GIB // 4 for i in range(64)}, total_bytes=17 * _GIB,
                                      output_bytes=_GIB // 2)
        gov.tensor_table = lambda model, engine=None: table if model == "big" else None
        _snap(rg, gov, capacity=8.0, ram_mb=45000.0, total_mb=61000.0)
        decision = gov.admit("big", 4096, caller="chat")
        assert decision.admitted is True, ("admitted once the card is back", decision.reason)
        assert decision.partial_offload and decision.gpu_layers and decision.gpu_layers > 0, (
            "split, with layers on the GPU", decision.gpu_share, decision.gpu_layers)
        assert decision.num_ctx == 4096 and decision.load_expected is True, decision
        assert gov.estimate_model_vram_gb("big")[1] == "file_size", "the file prices it, not the CPU's cost"
    finally:
        restore()
