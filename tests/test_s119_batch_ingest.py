"""
Tests for S119 -- RAG Batch Ingestion Backend.

Validates:
- Part 1: batch_ingest.py core (SQLite DB, job/file CRUD, counters, enums)
- Part 2: Folder scanner (scan_folder, extension filtering, skip patterns)
- Part 3: BatchIngestEngine (create jobs, start/cancel, worker lifecycle)
- Part 4: API routes (schemas, batch upload, folder ingest, job CRUD, enhanced docs)
- Part 5: AST / code quality checks
- Part 6: Integration wiring (__init__.py exports, app.py registration)
- Zero regressions

Target: ~45 tests
"""

import ast
import importlib.util
import json
import os
import re
import sqlite3
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
RAG_DIR = os.path.join(BACKEND_DIR, "rag")
API_DIR = os.path.join(BACKEND_DIR, "api")


def _load_module(name, path):
    """Load a Python module from file path without triggering __init__ imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load batch_ingest without triggering the full rag __init__ chain
_batch_mod = _load_module("batch_ingest", os.path.join(RAG_DIR, "batch_ingest.py"))

JobStatus = _batch_mod.JobStatus
FileStatus = _batch_mod.FileStatus
IngestFileRecord = _batch_mod.IngestFileRecord
IngestJobRecord = _batch_mod.IngestJobRecord
_IngestJobsDatabase = _batch_mod._IngestJobsDatabase
BatchIngestEngine = _batch_mod.BatchIngestEngine
scan_folder = _batch_mod.scan_folder
_should_include_file = _batch_mod._should_include_file
SUPPORTED_EXTENSIONS = _batch_mod.SUPPORTED_EXTENSIONS


# =========================================================================
# PART 1: DATABASE LAYER
# =========================================================================

class TestIngestJobsDatabase(unittest.TestCase):
    """Test the SQLite-backed job tracking database."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = _IngestJobsDatabase(os.path.join(self.tmp, "test_jobs.db"))

    def test_create_job_basic(self):
        """Creating a job stores it with pending status."""
        self.db.create_job("j1", "papers", "batch")
        job = self.db.get_job("j1")
        self.assertIsNotNone(job)
        self.assertEqual(job.job_id, "j1")
        self.assertEqual(job.status, JobStatus.PENDING.value)
        self.assertEqual(job.collection, "papers")
        self.assertEqual(job.source_type, "batch")
        self.assertGreater(job.created_at, 0)

    def test_create_folder_job(self):
        """Folder jobs store source_path."""
        self.db.create_job("j2", "default", "folder", source_path="/home/data")
        job = self.db.get_job("j2")
        self.assertEqual(job.source_type, "folder")
        self.assertEqual(job.source_path, "/home/data")

    def test_get_job_not_found(self):
        """Getting a non-existent job returns None."""
        self.assertIsNone(self.db.get_job("nonexistent"))

    def test_add_files_and_get(self):
        """Adding files to a job and retrieving them."""
        self.db.create_job("j1", "default", "batch")
        self.db.add_file("f1", "j1", "/tmp/a.pdf", "a.pdf", 1024)
        self.db.add_file("f2", "j1", "/tmp/b.txt", "b.txt", 512)

        files = self.db.get_files_for_job("j1")
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0].file_id, "f1")
        self.assertEqual(files[0].filename, "a.pdf")
        self.assertEqual(files[0].file_size, 1024)
        self.assertEqual(files[0].status, FileStatus.QUEUED.value)

    def test_next_queued_file_fifo(self):
        """get_next_queued_file returns files in FIFO order."""
        self.db.create_job("j1", "default", "batch")
        self.db.add_file("f1", "j1", "/tmp/a.pdf", "a.pdf", 100)
        self.db.add_file("f2", "j1", "/tmp/b.txt", "b.txt", 200)

        nxt = self.db.get_next_queued_file("j1")
        self.assertEqual(nxt.file_id, "f1")

        # Mark first as processing, next should be f2
        self.db.update_file_status("f1", FileStatus.PROCESSING.value)
        nxt = self.db.get_next_queued_file("j1")
        self.assertEqual(nxt.file_id, "f2")

    def test_next_queued_file_none_when_empty(self):
        """Returns None when no queued files remain."""
        self.db.create_job("j1", "default", "batch")
        self.assertIsNone(self.db.get_next_queued_file("j1"))

    def test_update_file_status_done(self):
        """Updating file status to done records doc_id and chunk_count."""
        self.db.create_job("j1", "default", "batch")
        self.db.add_file("f1", "j1", "/tmp/a.pdf", "a.pdf", 100)

        self.db.update_file_status(
            "f1", FileStatus.DONE.value,
            doc_id="doc123", chunk_count=15,
        )
        files = self.db.get_files_for_job("j1")
        self.assertEqual(files[0].status, FileStatus.DONE.value)
        self.assertEqual(files[0].doc_id, "doc123")
        self.assertEqual(files[0].chunk_count, 15)
        self.assertIsNotNone(files[0].completed_at)

    def test_update_file_status_error(self):
        """Updating file status to error records error_message."""
        self.db.create_job("j1", "default", "batch")
        self.db.add_file("f1", "j1", "/tmp/a.pdf", "a.pdf", 100)

        self.db.update_file_status(
            "f1", FileStatus.ERROR.value,
            error_message="Could not parse PDF",
        )
        files = self.db.get_files_for_job("j1")
        self.assertEqual(files[0].status, FileStatus.ERROR.value)
        self.assertEqual(files[0].error_message, "Could not parse PDF")

    def test_update_job_counters(self):
        """Job counters recompute from file statuses."""
        self.db.create_job("j1", "default", "batch")
        self.db.add_file("f1", "j1", "/tmp/a.pdf", "a.pdf", 100)
        self.db.add_file("f2", "j1", "/tmp/b.txt", "b.txt", 200)
        self.db.add_file("f3", "j1", "/tmp/c.md", "c.md", 300)

        self.db.update_file_status("f1", FileStatus.DONE.value, chunk_count=10)
        self.db.update_file_status("f2", FileStatus.ERROR.value, error_message="fail")
        self.db.update_file_status("f3", FileStatus.SKIPPED.value)

        self.db.update_job_counters("j1")
        job = self.db.get_job("j1")
        self.assertEqual(job.total_files, 3)
        self.assertEqual(job.completed_files, 1)
        self.assertEqual(job.failed_files, 1)
        self.assertEqual(job.skipped_files, 1)
        self.assertEqual(job.total_chunks, 10)

    def test_update_job_status_running(self):
        """Setting job to running records started_at."""
        self.db.create_job("j1", "default", "batch")
        self.db.update_job_status("j1", JobStatus.RUNNING.value)
        job = self.db.get_job("j1")
        self.assertEqual(job.status, JobStatus.RUNNING.value)
        self.assertIsNotNone(job.started_at)

    def test_update_job_status_completed(self):
        """Setting job to completed records completed_at."""
        self.db.create_job("j1", "default", "batch")
        self.db.update_job_status("j1", JobStatus.COMPLETED.value)
        job = self.db.get_job("j1")
        self.assertEqual(job.status, JobStatus.COMPLETED.value)
        self.assertIsNotNone(job.completed_at)

    def test_update_job_status_failed_with_message(self):
        """Setting job to failed records error_message."""
        self.db.create_job("j1", "default", "batch")
        self.db.update_job_status("j1", JobStatus.FAILED.value, error_message="boom")
        job = self.db.get_job("j1")
        self.assertEqual(job.status, JobStatus.FAILED.value)
        self.assertEqual(job.error_message, "boom")

    def test_list_jobs_all(self):
        """List all jobs."""
        self.db.create_job("j1", "default", "batch")
        self.db.create_job("j2", "papers", "folder")
        jobs = self.db.list_jobs()
        self.assertEqual(len(jobs), 2)

    def test_list_jobs_by_status(self):
        """List jobs filtered by status."""
        self.db.create_job("j1", "default", "batch")
        self.db.create_job("j2", "papers", "folder")
        self.db.update_job_status("j1", JobStatus.RUNNING.value)

        running = self.db.list_jobs(status=JobStatus.RUNNING.value)
        self.assertEqual(len(running), 1)
        self.assertEqual(running[0].job_id, "j1")

    def test_list_jobs_pagination(self):
        """List jobs with limit and offset."""
        for i in range(5):
            self.db.create_job(f"j{i}", "default", "batch")
            time.sleep(0.01)  # Ensure different created_at

        jobs = self.db.list_jobs(limit=2, offset=0)
        self.assertEqual(len(jobs), 2)
        jobs2 = self.db.list_jobs(limit=2, offset=2)
        self.assertEqual(len(jobs2), 2)
        # No overlap
        ids1 = {j.job_id for j in jobs}
        ids2 = {j.job_id for j in jobs2}
        self.assertEqual(len(ids1 & ids2), 0)

    def test_delete_job(self):
        """Deleting a job removes it and its files."""
        self.db.create_job("j1", "default", "batch")
        self.db.add_file("f1", "j1", "/tmp/a.pdf", "a.pdf", 100)

        ok = self.db.delete_job("j1")
        self.assertTrue(ok)
        self.assertIsNone(self.db.get_job("j1"))
        self.assertEqual(len(self.db.get_files_for_job("j1")), 0)

    def test_delete_job_not_found(self):
        """Deleting a non-existent job returns False."""
        self.assertFalse(self.db.delete_job("nonexistent"))

    def test_job_with_files_attached(self):
        """get_job returns files attached to the job."""
        self.db.create_job("j1", "default", "batch")
        self.db.add_file("f1", "j1", "/tmp/a.pdf", "a.pdf", 100)
        self.db.add_file("f2", "j1", "/tmp/b.txt", "b.txt", 200)

        job = self.db.get_job("j1")
        self.assertEqual(len(job.files), 2)


# =========================================================================
# PART 2: ENUMS AND DATA STRUCTURES
# =========================================================================

class TestEnumsAndDataStructures(unittest.TestCase):
    """Test enums and dataclass methods."""

    def test_job_status_values(self):
        """JobStatus enum has expected values."""
        self.assertEqual(JobStatus.PENDING.value, "pending")
        self.assertEqual(JobStatus.RUNNING.value, "running")
        self.assertEqual(JobStatus.COMPLETED.value, "completed")
        self.assertEqual(JobStatus.FAILED.value, "failed")
        self.assertEqual(JobStatus.CANCELLED.value, "cancelled")

    def test_file_status_values(self):
        """FileStatus enum has expected values."""
        self.assertEqual(FileStatus.QUEUED.value, "queued")
        self.assertEqual(FileStatus.PROCESSING.value, "processing")
        self.assertEqual(FileStatus.DONE.value, "done")
        self.assertEqual(FileStatus.ERROR.value, "error")
        self.assertEqual(FileStatus.SKIPPED.value, "skipped")

    def test_ingest_job_record_to_dict(self):
        """IngestJobRecord.to_dict() serializes correctly."""
        rec = IngestJobRecord(
            job_id="j1", status="pending", collection="papers",
            source_type="batch", total_files=3, created_at=100.0,
        )
        d = rec.to_dict()
        self.assertEqual(d["job_id"], "j1")
        self.assertEqual(d["collection"], "papers")
        self.assertEqual(d["total_files"], 3)
        self.assertNotIn("files", d)  # include_files defaults to False

    def test_ingest_job_record_to_dict_with_files(self):
        """to_dict(include_files=True) includes file list."""
        frec = IngestFileRecord(
            file_id="f1", job_id="j1", filepath="/tmp/a.pdf",
            filename="a.pdf", file_size=100, status="done",
        )
        rec = IngestJobRecord(
            job_id="j1", status="completed", collection="default",
            source_type="batch", files=[frec], created_at=100.0,
        )
        d = rec.to_dict(include_files=True)
        self.assertIn("files", d)
        self.assertEqual(len(d["files"]), 1)
        self.assertEqual(d["files"][0]["file_id"], "f1")

    def test_ingest_job_progress(self):
        """IngestJobRecord.progress property computes correctly."""
        rec = IngestJobRecord(
            job_id="j1", status="running", collection="default",
            source_type="batch", total_files=10, completed_files=3,
            failed_files=2, skipped_files=1, created_at=100.0,
        )
        self.assertAlmostEqual(rec.progress, 0.6)

    def test_ingest_job_progress_zero_files(self):
        """Progress is 1.0 when total_files is 0."""
        rec = IngestJobRecord(
            job_id="j1", status="completed", collection="default",
            source_type="batch", total_files=0, created_at=100.0,
        )
        self.assertAlmostEqual(rec.progress, 1.0)

    def test_ingest_file_record_to_dict(self):
        """IngestFileRecord.to_dict() serializes all fields."""
        frec = IngestFileRecord(
            file_id="f1", job_id="j1", filepath="/tmp/a.pdf",
            filename="a.pdf", file_size=1024, status="done",
            doc_id="doc1", chunk_count=15,
        )
        d = frec.to_dict()
        self.assertEqual(d["file_id"], "f1")
        self.assertEqual(d["doc_id"], "doc1")
        self.assertEqual(d["chunk_count"], 15)

    def test_supported_extensions_include_required(self):
        """SUPPORTED_EXTENSIONS includes PDF, TXT, MD, HTML, DOCX."""
        for ext in [".pdf", ".txt", ".md", ".html", ".docx"]:
            self.assertIn(ext, SUPPORTED_EXTENSIONS, f"{ext} missing")


# =========================================================================
# PART 3: FOLDER SCANNER
# =========================================================================

class TestFolderScanner(unittest.TestCase):
    """Test the scan_folder utility."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_scan_finds_supported_files(self):
        """scan_folder discovers supported file types."""
        for fn in ["a.pdf", "b.txt", "c.md", "d.html", "e.docx"]:
            Path(self.tmp, fn).touch()
        found = scan_folder(self.tmp)
        names = [p.name for p in found]
        self.assertEqual(len(names), 5)
        self.assertIn("a.pdf", names)
        self.assertIn("e.docx", names)

    def test_scan_skips_unsupported(self):
        """scan_folder ignores unsupported extensions."""
        Path(self.tmp, "a.pdf").touch()
        Path(self.tmp, "b.exe").touch()
        Path(self.tmp, "c.bin").touch()
        found = scan_folder(self.tmp)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "a.pdf")

    def test_scan_skips_hidden_files(self):
        """scan_folder skips files starting with a dot."""
        Path(self.tmp, ".hidden.txt").touch()
        Path(self.tmp, "visible.txt").touch()
        found = scan_folder(self.tmp)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "visible.txt")

    def test_scan_skips_large_files(self):
        """scan_folder skips files exceeding max size."""
        fp = Path(self.tmp, "big.txt")
        fp.write_bytes(b"x" * 1024)  # 1KB
        found = scan_folder(self.tmp, max_file_size_mb=0.0005)  # ~0.5KB
        self.assertEqual(len(found), 0)

    def test_scan_recursive(self):
        """scan_folder recurses into subdirectories."""
        sub = Path(self.tmp, "sub")
        sub.mkdir()
        Path(self.tmp, "a.txt").touch()
        Path(sub, "b.md").touch()
        found = scan_folder(self.tmp, recursive=True)
        names = [p.name for p in found]
        self.assertIn("a.txt", names)
        self.assertIn("b.md", names)

    def test_scan_non_recursive(self):
        """Non-recursive scan does not enter subdirectories."""
        sub = Path(self.tmp, "sub")
        sub.mkdir()
        Path(self.tmp, "a.txt").touch()
        Path(sub, "b.md").touch()
        found = scan_folder(self.tmp, recursive=False)
        names = [p.name for p in found]
        self.assertIn("a.txt", names)
        self.assertNotIn("b.md", names)

    def test_scan_skips_pycache(self):
        """scan_folder skips __pycache__ directories."""
        pycache = Path(self.tmp, "__pycache__")
        pycache.mkdir()
        Path(pycache, "module.py").touch()
        Path(self.tmp, "real.py").touch()
        found = scan_folder(self.tmp)
        names = [p.name for p in found]
        self.assertIn("real.py", names)
        self.assertNotIn("module.py", names)

    def test_scan_skips_git_dir(self):
        """scan_folder skips .git directories."""
        git = Path(self.tmp, ".git")
        git.mkdir()
        Path(git, "config.txt").touch()
        found = scan_folder(self.tmp)
        self.assertEqual(len(found), 0)

    def test_scan_custom_extensions(self):
        """scan_folder with custom extensions filter."""
        for fn in ["a.pdf", "b.txt", "c.py"]:
            Path(self.tmp, fn).touch()
        found = scan_folder(self.tmp, extensions={".pdf"})
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "a.pdf")

    def test_scan_invalid_directory(self):
        """scan_folder raises ValueError for non-directory."""
        with self.assertRaises(ValueError):
            scan_folder("/nonexistent/path")

    def test_scan_results_sorted(self):
        """scan_folder returns results sorted by name."""
        for fn in ["c.txt", "a.txt", "b.txt"]:
            Path(self.tmp, fn).touch()
        found = scan_folder(self.tmp)
        names = [p.name for p in found]
        self.assertEqual(names, ["a.txt", "b.txt", "c.txt"])


# =========================================================================
# PART 4: BATCH INGEST ENGINE
# =========================================================================

class TestBatchIngestEngine(unittest.TestCase):
    """Test the BatchIngestEngine class."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.engine = BatchIngestEngine(data_dir=self.tmp)

    def test_create_batch_job(self):
        """Create a batch job from file paths."""
        # Create temp files
        f1 = Path(self.tmp, "a.pdf")
        f2 = Path(self.tmp, "b.txt")
        f1.touch()
        f2.touch()

        job = self.engine.create_batch_job(
            filepaths=[str(f1), str(f2)],
            collection="papers",
        )
        self.assertEqual(job.status, JobStatus.PENDING.value)
        self.assertEqual(job.collection, "papers")
        self.assertEqual(job.source_type, "batch")
        self.assertEqual(job.total_files, 2)
        self.assertEqual(len(job.files), 2)

    def test_create_batch_job_skips_nonexistent(self):
        """Batch job creation skips files that don't exist."""
        f1 = Path(self.tmp, "exists.pdf")
        f1.touch()

        job = self.engine.create_batch_job(
            filepaths=[str(f1), "/nonexistent/file.txt"],
            collection="default",
        )
        self.assertEqual(job.total_files, 1)

    def test_create_folder_job(self):
        """Create a folder scan job."""
        for fn in ["a.pdf", "b.txt", "c.md"]:
            Path(self.tmp, fn).touch()

        job = self.engine.create_folder_job(
            directory=self.tmp,
            collection="docs",
        )
        self.assertEqual(job.source_type, "folder")
        self.assertEqual(job.source_path, str(Path(self.tmp).resolve()))
        self.assertEqual(job.collection, "docs")
        self.assertGreaterEqual(job.total_files, 3)

    def test_create_folder_job_empty_dir(self):
        """Folder job with empty directory has 0 files."""
        empty = Path(self.tmp, "empty")
        empty.mkdir()
        job = self.engine.create_folder_job(directory=str(empty))
        self.assertEqual(job.total_files, 0)

    def test_get_job(self):
        """Get a job by ID."""
        f1 = Path(self.tmp, "a.txt")
        f1.touch()
        job = self.engine.create_batch_job(filepaths=[str(f1)])
        retrieved = self.engine.get_job(job.job_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.job_id, job.job_id)

    def test_get_job_not_found(self):
        """Get non-existent job returns None."""
        self.assertIsNone(self.engine.get_job("nonexistent"))

    def test_list_jobs(self):
        """List all jobs."""
        f1 = Path(self.tmp, "a.txt")
        f1.touch()
        self.engine.create_batch_job(filepaths=[str(f1)])
        self.engine.create_batch_job(filepaths=[str(f1)], collection="papers")

        jobs = self.engine.list_jobs()
        self.assertEqual(len(jobs), 2)

    def test_delete_job(self):
        """Delete a pending job."""
        f1 = Path(self.tmp, "a.txt")
        f1.touch()
        job = self.engine.create_batch_job(filepaths=[str(f1)])
        ok = self.engine.delete_job(job.job_id)
        self.assertTrue(ok)
        self.assertIsNone(self.engine.get_job(job.job_id))

    def test_cancel_pending_job(self):
        """Cancel a pending job."""
        f1 = Path(self.tmp, "a.txt")
        f1.touch()
        job = self.engine.create_batch_job(filepaths=[str(f1)])
        ok = self.engine.cancel_job(job.job_id)
        self.assertTrue(ok)
        refreshed = self.engine.get_job(job.job_id)
        self.assertEqual(refreshed.status, JobStatus.CANCELLED.value)

    def test_cancel_already_completed(self):
        """Cannot cancel an already completed job."""
        f1 = Path(self.tmp, "a.txt")
        f1.touch()
        job = self.engine.create_batch_job(filepaths=[str(f1)])
        self.engine.db.update_job_status(job.job_id, JobStatus.COMPLETED.value)
        ok = self.engine.cancel_job(job.job_id)
        self.assertFalse(ok)

    def test_start_job_not_found(self):
        """Starting a non-existent job returns False."""
        self.assertFalse(self.engine.start_job("nonexistent"))

    def test_start_job_already_running(self):
        """Starting an already running job returns False."""
        f1 = Path(self.tmp, "a.txt")
        f1.touch()
        job = self.engine.create_batch_job(filepaths=[str(f1)])
        self.engine.db.update_job_status(job.job_id, JobStatus.RUNNING.value)
        ok = self.engine.start_job(job.job_id)
        self.assertFalse(ok)

    @patch.object(BatchIngestEngine, "_get_rag_store")
    def test_start_job_processes_files(self, mock_get_store):
        """Worker processes files and updates counters."""
        # Create a mock store
        mock_store = MagicMock()
        mock_doc = MagicMock()
        mock_doc.doc_id = "doc_abc"
        mock_doc.chunk_count = 5
        mock_store.ingest_file.return_value = mock_doc
        mock_get_store.return_value = mock_store

        # Create files
        f1 = Path(self.tmp, "a.txt")
        f2 = Path(self.tmp, "b.md")
        f1.write_text("hello world")
        f2.write_text("# Title\nContent")

        job = self.engine.create_batch_job(filepaths=[str(f1), str(f2)])
        self.engine.start_job(job.job_id)

        # Wait for completion
        for _ in range(50):
            refreshed = self.engine.get_job(job.job_id)
            if refreshed.status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                break
            time.sleep(0.05)

        refreshed = self.engine.get_job(job.job_id)
        self.assertEqual(refreshed.status, JobStatus.COMPLETED.value)
        self.assertEqual(refreshed.completed_files, 2)
        self.assertEqual(refreshed.total_chunks, 10)  # 5 per file
        self.assertEqual(mock_store.ingest_file.call_count, 2)

    @patch.object(BatchIngestEngine, "_get_rag_store")
    def test_start_job_handles_missing_file(self, mock_get_store):
        """Worker skips files that no longer exist."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # Create job with a path that will be deleted
        f1 = Path(self.tmp, "will_delete.txt")
        f1.write_text("temp")
        job = self.engine.create_batch_job(filepaths=[str(f1)])

        # Delete before processing
        f1.unlink()

        self.engine.start_job(job.job_id)
        for _ in range(50):
            refreshed = self.engine.get_job(job.job_id)
            if refreshed.status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                break
            time.sleep(0.05)

        refreshed = self.engine.get_job(job.job_id)
        self.assertEqual(refreshed.status, JobStatus.COMPLETED.value)
        self.assertEqual(refreshed.skipped_files, 1)
        mock_store.ingest_file.assert_not_called()

    @patch.object(BatchIngestEngine, "_get_rag_store")
    def test_start_job_handles_ingest_error(self, mock_get_store):
        """Worker marks files as error when ingest_file raises."""
        mock_store = MagicMock()
        mock_store.ingest_file.side_effect = RuntimeError("Parse failed")
        mock_get_store.return_value = mock_store

        f1 = Path(self.tmp, "bad.pdf")
        f1.write_text("not a pdf")
        job = self.engine.create_batch_job(filepaths=[str(f1)])
        self.engine.start_job(job.job_id)

        for _ in range(50):
            refreshed = self.engine.get_job(job.job_id)
            if refreshed.status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                break
            time.sleep(0.05)

        refreshed = self.engine.get_job(job.job_id)
        self.assertEqual(refreshed.status, JobStatus.COMPLETED.value)
        self.assertEqual(refreshed.failed_files, 1)
        self.assertIn("Parse failed", refreshed.files[0].error_message)

    @patch.object(BatchIngestEngine, "_get_rag_store")
    def test_start_job_store_unavailable(self, mock_get_store):
        """Job fails when RAG store is unavailable."""
        mock_get_store.return_value = None

        f1 = Path(self.tmp, "a.txt")
        f1.write_text("hello")
        job = self.engine.create_batch_job(filepaths=[str(f1)])
        self.engine.start_job(job.job_id)

        for _ in range(50):
            refreshed = self.engine.get_job(job.job_id)
            if refreshed.status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                break
            time.sleep(0.05)

        refreshed = self.engine.get_job(job.job_id)
        self.assertEqual(refreshed.status, JobStatus.FAILED.value)
        self.assertIn("unavailable", refreshed.error_message)


# =========================================================================
# PART 5: API ROUTES (AST-BASED)
# =========================================================================

class TestRoutesRagAST(unittest.TestCase):
    """Validate API route structure via AST parsing."""

    @classmethod
    def setUpClass(cls):
        routes_path = os.path.join(API_DIR, "routes_rag.py")
        with open(routes_path) as f:
            cls.source = f.read()
        cls.tree = ast.parse(cls.source)

    def _get_route_decorators(self):
        """Extract (method, path, func_name) tuples from the AST."""
        routes = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                        method = dec.func.attr  # get, post, delete, etc.
                        if dec.args and isinstance(dec.args[0], ast.Constant):
                            path = dec.args[0].value
                            routes.append((method, path, node.name))
        return routes

    def test_batch_endpoint_exists(self):
        """POST /ingest/batch endpoint exists."""
        routes = self._get_route_decorators()
        batch = [r for r in routes if r[1] == "/ingest/batch" and r[0] == "post"]
        self.assertEqual(len(batch), 1, "POST /ingest/batch not found")

    def test_folder_endpoint_exists(self):
        """POST /ingest/folder endpoint exists."""
        routes = self._get_route_decorators()
        folder = [r for r in routes if r[1] == "/ingest/folder" and r[0] == "post"]
        self.assertEqual(len(folder), 1, "POST /ingest/folder not found")

    def test_jobs_list_endpoint_exists(self):
        """GET /ingest/jobs endpoint exists."""
        routes = self._get_route_decorators()
        jobs = [r for r in routes if r[1] == "/ingest/jobs" and r[0] == "get"]
        self.assertEqual(len(jobs), 1, "GET /ingest/jobs not found")

    def test_job_detail_endpoint_exists(self):
        """GET /ingest/jobs/{job_id} endpoint exists."""
        routes = self._get_route_decorators()
        detail = [r for r in routes if r[1] == "/ingest/jobs/{job_id}" and r[0] == "get"]
        self.assertEqual(len(detail), 1, "GET /ingest/jobs/{job_id} not found")

    def test_job_delete_endpoint_exists(self):
        """DELETE /ingest/jobs/{job_id} endpoint exists."""
        routes = self._get_route_decorators()
        delete = [r for r in routes if r[1] == "/ingest/jobs/{job_id}" and r[0] == "delete"]
        self.assertEqual(len(delete), 1, "DELETE /ingest/jobs/{job_id} not found")

    def test_documents_endpoint_has_search_param(self):
        """GET /documents has search query parameter."""
        self.assertIn("search", self.source)

    def test_documents_endpoint_has_file_type_param(self):
        """GET /documents has file_type query parameter."""
        self.assertIn("file_type", self.source)

    def test_batch_returns_202(self):
        """POST /ingest/batch returns 202 status code."""
        self.assertIn("status_code=202", self.source)

    def test_schemas_exist(self):
        """All required Pydantic schemas are defined."""
        for cls_name in [
            "IngestFolderRequest",
            "IngestFileStatusResponse",
            "IngestJobResponse",
            "IngestJobsListResponse",
            "IngestJobDeleteResponse",
        ]:
            self.assertIn(f"class {cls_name}", self.source,
                          f"Schema {cls_name} not found")

    def test_helper_functions_exist(self):
        """Helper functions are defined."""
        self.assertIn("def _get_batch_engine", self.source)
        self.assertIn("def _job_to_response", self.source)


# =========================================================================
# PART 6: INTEGRATION WIRING
# =========================================================================

class TestIntegrationWiring(unittest.TestCase):
    """Verify module exports and app registration."""

    def test_rag_init_exports_batch_ingest(self):
        """rag/__init__.py exports batch ingestion symbols."""
        init_path = os.path.join(RAG_DIR, "__init__.py")
        with open(init_path) as f:
            src = f.read()
        for name in [
            "BatchIngestEngine",
            "get_batch_ingest_engine",
            "scan_folder",
            "JobStatus",
            "FileStatus",
            "IngestJobRecord",
            "IngestFileRecord",
        ]:
            self.assertIn(name, src, f"{name} not in rag/__init__.py")

    def test_batch_ingest_ast_valid(self):
        """batch_ingest.py is valid Python."""
        path = os.path.join(RAG_DIR, "batch_ingest.py")
        with open(path) as f:
            ast.parse(f.read())

    def test_routes_rag_ast_valid(self):
        """routes_rag.py is valid Python."""
        path = os.path.join(API_DIR, "routes_rag.py")
        with open(path) as f:
            ast.parse(f.read())

    def test_no_french_in_batch_ingest(self):
        """No French comments in batch_ingest.py."""
        path = os.path.join(RAG_DIR, "batch_ingest.py")
        with open(path) as f:
            src = f.read()
        french_words = ["fichier", "dossier", "erreur", "indexation", "supprimer", "chercher"]
        for line in src.split("\n"):
            if "#" in line:
                comment = line[line.index("#"):].lower()
                for w in french_words:
                    self.assertNotIn(w, comment,
                                     f"French word '{w}' found in comment: {line.strip()}")

    def test_no_hardcoded_hex_colors(self):
        """No hardcoded hex colors in batch_ingest.py (N/A but verify)."""
        path = os.path.join(RAG_DIR, "batch_ingest.py")
        with open(path) as f:
            src = f.read()
        # Skip hash computation lines
        for line in src.split("\n"):
            if "hashlib" in line or "hex()" in line or "hex[:12]" in line:
                continue
            if re.search(r'#[0-9a-fA-F]{6}\b', line) and "color" in line.lower():
                self.fail(f"Hardcoded hex color found: {line.strip()}")

    def test_sqlite_wal_mode(self):
        """batch_ingest.py uses WAL mode for SQLite."""
        path = os.path.join(RAG_DIR, "batch_ingest.py")
        with open(path) as f:
            src = f.read()
        self.assertIn("journal_mode=WAL", src)

    def test_app_includes_rag_router(self):
        """app.py includes the RAG router."""
        app_path = os.path.join(API_DIR, "app.py")
        with open(app_path) as f:
            src = f.read()
        self.assertIn("rag_router", src)

    def test_module_singleton_pattern(self):
        """batch_ingest.py follows singleton pattern."""
        path = os.path.join(RAG_DIR, "batch_ingest.py")
        with open(path) as f:
            src = f.read()
        self.assertIn("_engine_instance", src)
        self.assertIn("def get_batch_ingest_engine", src)


# =========================================================================
# PART 7: SQLITE THREAD SAFETY
# =========================================================================

class TestDatabaseThreadSafety(unittest.TestCase):
    """Verify DB can handle concurrent access from multiple threads."""

    def test_concurrent_file_updates(self):
        """Multiple threads can update file status concurrently."""
        tmp = tempfile.mkdtemp()
        db = _IngestJobsDatabase(os.path.join(tmp, "thread_test.db"))
        db.create_job("j1", "default", "batch")

        for i in range(20):
            db.add_file(f"f{i}", "j1", f"/tmp/{i}.txt", f"{i}.txt", 100)

        errors = []

        def update_file(file_id, idx):
            try:
                db.update_file_status(
                    file_id, FileStatus.DONE.value,
                    doc_id=f"doc{idx}", chunk_count=idx,
                )
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=update_file, args=(f"f{i}", i))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")

        db.update_job_counters("j1")
        job = db.get_job("j1")
        self.assertEqual(job.completed_files, 20)


if __name__ == "__main__":
    unittest.main()
