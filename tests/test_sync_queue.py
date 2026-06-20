#!/usr/bin/env python3
"""
Tests for SyncQueue -- S71 Step 2: SQLite-backed request queue.

Covers:
- Enqueue/dequeue ordering (priority + FIFO)
- Persistence across reload
- Queue overflow (max size)
- Mark completed/failed
- Requeue failed entries
- Queue size counting with status filter
- List entries with status filter
- Clear entries
- Process queue with executor function
- Process queue with failure
- QueueEntry dataclass
"""

import importlib.util
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Direct module import
# ---------------------------------------------------------------------------

_mod_path = Path(__file__).resolve().parent.parent / "opti_oignon" / "sync_queue.py"
_spec = importlib.util.spec_from_file_location("sync_queue_mod", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("ollama", MagicMock())
_spec.loader.exec_module(_mod)

SyncQueue = _mod.SyncQueue
QueueEntry = _mod.QueueEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_queue(tmp_path: Path, max_size: int = 100) -> SyncQueue:
    """Create a SyncQueue with a temp database."""
    db_path = tmp_path / "test_queue.db"
    config_path = tmp_path / "network.yaml"
    cfg = {"max_queue_size": max_size}
    with open(config_path, "w") as f:
        yaml.safe_dump({"queue": cfg}, f)
    return SyncQueue(db_path=db_path, config_path=config_path)


# ===========================================================================
# TEST CLASSES
# ===========================================================================


class TestQueueEntry:
    """Tests for QueueEntry dataclass."""

    def test_default_values(self):
        entry = QueueEntry()
        assert entry.id == ""
        assert entry.query == ""
        assert entry.task_type == "general"
        assert entry.priority == 5
        assert entry.status == "pending"

    def test_to_dict(self):
        entry = QueueEntry(id="abc", query="hello", task_type="code_python", priority=2)
        d = entry.to_dict()
        assert d["id"] == "abc"
        assert d["task_type"] == "code_python"
        assert d["priority"] == 2


class TestEnqueueDequeue:
    """Tests for basic enqueue/dequeue operations."""

    def test_enqueue_returns_entry(self, tmp_path):
        sq = _make_queue(tmp_path)
        entry = sq.enqueue("hello world", task_type="general")
        assert entry is not None
        assert entry.query == "hello world"
        assert entry.status == "pending"
        assert len(entry.id) > 0

    def test_dequeue_returns_entry(self, tmp_path):
        sq = _make_queue(tmp_path)
        sq.enqueue("hello")
        entry = sq.dequeue()
        assert entry is not None
        assert entry.query == "hello"
        assert entry.status == "processing"

    def test_dequeue_empty_returns_none(self, tmp_path):
        sq = _make_queue(tmp_path)
        assert sq.dequeue() is None

    def test_fifo_ordering(self, tmp_path):
        sq = _make_queue(tmp_path)
        sq.enqueue("first")
        time.sleep(0.01)
        sq.enqueue("second")
        e1 = sq.dequeue()
        e2 = sq.dequeue()
        assert e1.query == "first"
        assert e2.query == "second"

    def test_priority_ordering(self, tmp_path):
        sq = _make_queue(tmp_path)
        sq.enqueue("low priority", priority=9)
        sq.enqueue("high priority", priority=1)
        sq.enqueue("medium priority", priority=5)
        e1 = sq.dequeue()
        e2 = sq.dequeue()
        e3 = sq.dequeue()
        assert e1.query == "high priority"
        assert e2.query == "medium priority"
        assert e3.query == "low priority"

    def test_priority_clamped(self, tmp_path):
        sq = _make_queue(tmp_path)
        entry = sq.enqueue("test", priority=0)
        assert entry.priority == 1  # clamped to min
        entry2 = sq.enqueue("test2", priority=20)
        assert entry2.priority == 10  # clamped to max


class TestQueueOverflow:
    """Tests for max queue size enforcement."""

    def test_overflow_rejects(self, tmp_path):
        sq = _make_queue(tmp_path, max_size=2)
        sq.enqueue("one")
        sq.enqueue("two")
        result = sq.enqueue("three")
        assert result is None
        assert sq.size() == 2


class TestQueuePersistence:
    """Tests for persistence across SyncQueue instances."""

    def test_persistence_across_reload(self, tmp_path):
        db_path = tmp_path / "persist.db"
        config_path = tmp_path / "network.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump({}, f)

        sq1 = SyncQueue(db_path=db_path, config_path=config_path)
        sq1.enqueue("persistent query")

        sq2 = SyncQueue(db_path=db_path, config_path=config_path)
        assert sq2.size() == 1
        entry = sq2.dequeue()
        assert entry.query == "persistent query"


class TestMarkStatus:
    """Tests for mark_completed, mark_failed, requeue_failed."""

    def test_mark_completed(self, tmp_path):
        sq = _make_queue(tmp_path)
        entry = sq.enqueue("test")
        sq.dequeue()
        sq.mark_completed(entry.id)
        assert sq.size(status="completed") == 1
        assert sq.size(status="pending") == 0

    def test_mark_failed(self, tmp_path):
        sq = _make_queue(tmp_path)
        entry = sq.enqueue("test")
        sq.dequeue()
        sq.mark_failed(entry.id, "timeout")
        entries = sq.list_entries(status="failed")
        assert len(entries) == 1
        assert entries[0].error == "timeout"

    def test_requeue_failed(self, tmp_path):
        sq = _make_queue(tmp_path)
        entry = sq.enqueue("test")
        sq.dequeue()
        sq.mark_failed(entry.id, "error")
        count = sq.requeue_failed()
        assert count == 1
        assert sq.size(status="pending") == 1


class TestListAndClear:
    """Tests for list_entries and clear."""

    def test_list_all(self, tmp_path):
        sq = _make_queue(tmp_path)
        sq.enqueue("a")
        sq.enqueue("b")
        entries = sq.list_entries()
        assert len(entries) == 2

    def test_list_filtered(self, tmp_path):
        sq = _make_queue(tmp_path)
        sq.enqueue("a")
        sq.enqueue("b")
        sq.dequeue()  # marks one as processing
        assert len(sq.list_entries(status="pending")) == 1
        assert len(sq.list_entries(status="processing")) == 1

    def test_clear_all(self, tmp_path):
        sq = _make_queue(tmp_path)
        sq.enqueue("a")
        sq.enqueue("b")
        removed = sq.clear()
        assert removed == 2
        assert sq.size() == 0

    def test_clear_by_status(self, tmp_path):
        sq = _make_queue(tmp_path)
        e1 = sq.enqueue("a")
        sq.enqueue("b")
        sq.dequeue()
        sq.mark_completed(e1.id)
        removed = sq.clear(status="completed")
        assert removed == 1
        assert sq.size() == 1


class TestProcessQueue:
    """Tests for process_queue() with executor function."""

    def test_process_queue_success(self, tmp_path):
        sq = _make_queue(tmp_path)
        sq.enqueue("q1")
        sq.enqueue("q2")

        def executor(entry):
            return f"response to {entry.query}"

        results = sq.process_queue(executor_fn=executor)
        assert len(results) == 2
        assert results[0]["status"] == "completed"
        assert results[0]["response"] == "response to q1"
        assert sq.size(status="completed") == 2

    def test_process_queue_with_failure(self, tmp_path):
        sq = _make_queue(tmp_path)
        sq.enqueue("good")
        sq.enqueue("bad")

        def executor(entry):
            if entry.query == "bad":
                raise RuntimeError("generation failed")
            return "ok"

        results = sq.process_queue(executor_fn=executor)
        assert len(results) == 2
        assert results[0]["status"] == "completed"
        assert results[1]["status"] == "failed"
        assert "generation failed" in results[1]["error"]

    def test_process_queue_no_executor(self, tmp_path):
        sq = _make_queue(tmp_path)
        sq.enqueue("test")
        results = sq.process_queue()
        assert len(results) == 1
        assert results[0]["status"] == "completed"
