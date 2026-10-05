#!/usr/bin/env python3
"""
RAG BATCH INGESTION ENGINE
==================================

Provides batch file ingestion with background processing, per-file progress
tracking, and folder scanning for the RAG knowledge base.

Features:
- Batch upload: ingest multiple files in one request
- Folder scan: recursively discover and ingest supported files from a directory
- SQLite-backed job tracking (rag_ingest_jobs.db)
- Background worker thread for non-blocking ingestion
- Per-file status: queued -> processing -> done | error | skipped
- Job lifecycle: pending -> running -> completed | failed | cancelled

Author: Leon
"""

import functools
import importlib
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from concurrent.futures import CancelledError, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The governor's caller for the job's embeddings: a background class, so
# they wait for the machine rather than take it from interactive work.
INDEX_CALLER = "index"
# What _admit returns when the job is cancelled while it waits.
_CANCELLED = object()
# How long the job waits for the chunking at the head of its flight between
# two looks at its cancel, in seconds: a cancel is seen within one slice, as
# in the governor's queue, whose wait is sliced the same.
_HEAD_WAIT_SLICE_S = 0.5
# Audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)



# =========================================================================
# CONSTANTS & ENUMS
# =========================================================================

class JobStatus(str, Enum):
    """Status of a batch ingestion job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileStatus(str, Enum):
    """Status of an individual file within a job."""
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"
    SKIPPED = "skipped"


# Supported file extensions for batch ingestion
SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf", ".txt", ".md", ".html", ".htm", ".docx", ".doc",
    ".csv", ".tsv", ".xlsx", ".xls",
    ".py", ".r", ".R", ".rmd", ".Rmd",
    ".json", ".yaml", ".yml", ".toml",
    ".js", ".ts", ".css", ".sql", ".sh",
}

# Maximum file size for batch ingestion (MB)
MAX_FILE_SIZE_MB: float = 50.0


# =========================================================================
# DATA STRUCTURES
# =========================================================================

@dataclass
class IngestFileRecord:
    """Tracks one file within a batch ingestion job."""
    file_id: str
    job_id: str
    filepath: str
    filename: str
    file_size: int
    status: str
    doc_id: str | None = None
    chunk_count: int = 0
    error_message: str | None = None
    started_at: float | None = None
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "job_id": self.job_id,
            "filepath": self.filepath,
            "filename": self.filename,
            "file_size": self.file_size,
            "status": self.status,
            "doc_id": self.doc_id,
            "chunk_count": self.chunk_count,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class IngestJobRecord:
    """Tracks a batch ingestion job."""
    job_id: str
    status: str
    collection: str
    source_type: str  # "batch" or "folder"
    source_path: str | None = None  # folder path if source_type == "folder"
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    created_at: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None
    files: list[IngestFileRecord] = field(default_factory=list)

    def to_dict(self, include_files: bool = False) -> dict[str, Any]:
        d = {
            "job_id": self.job_id,
            "status": self.status,
            "collection": self.collection,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "failed_files": self.failed_files,
            "skipped_files": self.skipped_files,
            "total_chunks": self.total_chunks,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
        }
        if include_files:
            d["files"] = [f.to_dict() for f in self.files]
        return d

    @property
    def progress(self) -> float:
        """Return job progress as a float 0.0 - 1.0."""
        if self.total_files == 0:
            return 1.0
        return (self.completed_files + self.failed_files + self.skipped_files) / self.total_files


# =========================================================================
# SQLITE DATABASE
# =========================================================================

class _IngestJobsDatabase:
    """SQLite-backed storage for batch ingestion job tracking."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = _safe_connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id          TEXT PRIMARY KEY,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    collection      TEXT NOT NULL DEFAULT 'default',
                    source_type     TEXT NOT NULL DEFAULT 'batch',
                    source_path     TEXT,
                    total_files     INTEGER NOT NULL DEFAULT 0,
                    completed_files INTEGER NOT NULL DEFAULT 0,
                    failed_files    INTEGER NOT NULL DEFAULT 0,
                    skipped_files   INTEGER NOT NULL DEFAULT 0,
                    total_chunks    INTEGER NOT NULL DEFAULT 0,
                    created_at      REAL NOT NULL,
                    started_at      REAL,
                    completed_at    REAL,
                    error_message   TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status
                    ON ingest_jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_created
                    ON ingest_jobs(created_at DESC);

                CREATE TABLE IF NOT EXISTS ingest_files (
                    file_id         TEXT PRIMARY KEY,
                    job_id          TEXT NOT NULL,
                    filepath        TEXT NOT NULL,
                    filename        TEXT NOT NULL,
                    file_size       INTEGER NOT NULL DEFAULT 0,
                    status          TEXT NOT NULL DEFAULT 'queued',
                    doc_id          TEXT,
                    chunk_count     INTEGER NOT NULL DEFAULT 0,
                    error_message   TEXT,
                    started_at      REAL,
                    completed_at    REAL,
                    FOREIGN KEY (job_id) REFERENCES ingest_jobs(job_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_files_job
                    ON ingest_files(job_id);
                CREATE INDEX IF NOT EXISTS idx_files_status
                    ON ingest_files(status);
            """)

    # -- Job CRUD --

    def create_job(
        self,
        job_id: str,
        collection: str,
        source_type: str,
        source_path: str | None = None,
    ) -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO ingest_jobs
                   (job_id, status, collection, source_type, source_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (job_id, JobStatus.PENDING.value, collection, source_type, source_path, now),
            )

    def get_job(self, job_id: str) -> IngestJobRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if not row:
                return None
            job = self._row_to_job(row)
            # Attach files
            file_rows = conn.execute(
                "SELECT * FROM ingest_files WHERE job_id = ? ORDER BY rowid",
                (job_id,),
            ).fetchall()
            job.files = [self._row_to_file(r) for r in file_rows]
            return job

    def list_jobs(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IngestJobRecord]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM ingest_jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (status, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ingest_jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def update_job_status(
        self,
        job_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        now = time.time()
        with self._conn() as conn:
            if status == JobStatus.RUNNING.value:
                conn.execute(
                    "UPDATE ingest_jobs SET status = ?, started_at = ? WHERE job_id = ?",
                    (status, now, job_id),
                )
            elif status in (
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            ):
                conn.execute(
                    "UPDATE ingest_jobs SET status = ?, completed_at = ?, error_message = ? WHERE job_id = ?",
                    (status, now, error_message, job_id),
                )
            else:
                conn.execute(
                    "UPDATE ingest_jobs SET status = ? WHERE job_id = ?",
                    (status, job_id),
                )

    def update_job_counters(self, job_id: str) -> None:
        """Recompute job counters from file statuses."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT
                    COUNT(*) AS total,
                    COALESCE(SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END), 0) AS completed,
                    COALESCE(SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END), 0) AS failed,
                    COALESCE(SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END), 0) AS skipped,
                    COALESCE(SUM(CASE WHEN status = 'done' THEN chunk_count ELSE 0 END), 0) AS chunks
                FROM ingest_files WHERE job_id = ?""",
                (job_id,),
            ).fetchone()
            conn.execute(
                """UPDATE ingest_jobs SET
                    total_files = ?, completed_files = ?, failed_files = ?,
                    skipped_files = ?, total_chunks = ?
                WHERE job_id = ?""",
                (
                    row["total"],
                    row["completed"],
                    row["failed"],
                    row["skipped"],
                    row["chunks"],
                    job_id,
                ),
            )

    def delete_job(self, job_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT job_id FROM ingest_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM ingest_files WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM ingest_jobs WHERE job_id = ?", (job_id,))
            return True

    # -- File CRUD --

    def add_file(
        self,
        file_id: str,
        job_id: str,
        filepath: str,
        filename: str,
        file_size: int,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO ingest_files
                   (file_id, job_id, filepath, filename, file_size, status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (file_id, job_id, filepath, filename, file_size, FileStatus.QUEUED.value),
            )

    def get_next_queued_file(self, job_id: str) -> IngestFileRecord | None:
        """Get the next queued file for processing (FIFO by rowid)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM ingest_files WHERE job_id = ? AND status = ? ORDER BY rowid LIMIT 1",
                (job_id, FileStatus.QUEUED.value),
            ).fetchone()
            if not row:
                return None
            return self._row_to_file(row)

    def update_file_status(
        self,
        file_id: str,
        status: str,
        doc_id: str | None = None,
        chunk_count: int = 0,
        error_message: str | None = None,
    ) -> None:
        now = time.time()
        with self._conn() as conn:
            if status == FileStatus.PROCESSING.value:
                conn.execute(
                    "UPDATE ingest_files SET status = ?, started_at = ? WHERE file_id = ?",
                    (status, now, file_id),
                )
            else:
                conn.execute(
                    """UPDATE ingest_files SET
                        status = ?, doc_id = ?, chunk_count = ?,
                        error_message = ?, completed_at = ?
                    WHERE file_id = ?""",
                    (status, doc_id, chunk_count, error_message, now, file_id),
                )

    def requeue_file(self, file_id: str) -> None:
        """Put a file back in the queue, as if it had never been taken."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE ingest_files SET status = ?, started_at = NULL, completed_at = NULL WHERE file_id = ?",
                (FileStatus.QUEUED.value, file_id),
            )

    def get_files_for_job(self, job_id: str) -> list[IngestFileRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM ingest_files WHERE job_id = ? ORDER BY rowid",
                (job_id,),
            ).fetchall()
        return [self._row_to_file(r) for r in rows]

    # -- Helpers --

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> IngestJobRecord:
        return IngestJobRecord(
            job_id=row["job_id"],
            status=row["status"],
            collection=row["collection"],
            source_type=row["source_type"],
            source_path=row["source_path"],
            total_files=row["total_files"],
            completed_files=row["completed_files"],
            failed_files=row["failed_files"],
            skipped_files=row["skipped_files"],
            total_chunks=row["total_chunks"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error_message=row["error_message"],
        )

    @staticmethod
    def _row_to_file(row: sqlite3.Row) -> IngestFileRecord:
        return IngestFileRecord(
            file_id=row["file_id"],
            job_id=row["job_id"],
            filepath=row["filepath"],
            filename=row["filename"],
            file_size=row["file_size"],
            status=row["status"],
            doc_id=row["doc_id"],
            chunk_count=row["chunk_count"],
            error_message=row["error_message"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )


# =========================================================================
# FOLDER SCANNER
# =========================================================================

def scan_folder(
    directory: str | Path,
    recursive: bool = True,
    extensions: set[str] | None = None,
    max_file_size_mb: float = MAX_FILE_SIZE_MB,
) -> list[Path]:
    """
    Scan a directory for files with supported extensions.

    Args:
        directory: Path to scan.
        recursive: Whether to recurse into subdirectories.
        extensions: Allowed extensions (default: SUPPORTED_EXTENSIONS).
        max_file_size_mb: Skip files larger than this.

    Returns:
        Sorted list of file paths found.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    allowed = extensions or SUPPORTED_EXTENSIONS

    # Directories to skip
    skip_dirs = {
        "__pycache__", ".git", ".svn", "node_modules",
        ".venv", "venv", ".env", ".pytest_cache",
        ".mypy_cache", ".ruff_cache", ".tox",
    }

    results: list[Path] = []

    if recursive:
        for root, dirs, filenames in os.walk(directory):
            # Prune skipped directories
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for fn in filenames:
                fp = Path(root) / fn
                if _should_include_file(fp, allowed, max_file_size_mb):
                    results.append(fp)
    else:
        for fp in directory.iterdir():
            if fp.is_file() and _should_include_file(fp, allowed, max_file_size_mb):
                results.append(fp)

    results.sort(key=lambda p: p.name.lower())
    return results


def _should_include_file(
    fp: Path,
    allowed_extensions: set[str],
    max_size_mb: float,
) -> bool:
    """Check if a file should be included in a folder scan."""
    if not fp.is_file():
        return False
    if fp.name.startswith("."):
        return False
    if fp.suffix.lower() not in allowed_extensions:
        return False
    try:
        size_mb = fp.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return False
    except OSError:
        return False
    return True


# =========================================================================
# BATCH INGESTION ENGINE
# =========================================================================

class BatchIngestEngine:
    """
    Manages batch ingestion jobs with background processing.

    Usage::

        engine = BatchIngestEngine(data_dir="/path/to/data/rag")
        job = engine.create_batch_job(
            filepaths=["/tmp/a.pdf", "/tmp/b.txt"],
            collection="papers",
        )
        engine.start_job(job.job_id)

        # Poll for progress
        status = engine.get_job(job.job_id)
        print(status.progress, status.completed_files)
    """

    def __init__(self, data_dir: str | Path | None = None):
        if data_dir is None:
            try:
                from opti_oignon.config import DATA_DIR
                data_dir = Path(DATA_DIR) / "rag"
            except ImportError:
                data_dir = Path.home() / ".opti-oignon" / "data" / "rag"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db = _IngestJobsDatabase(self.data_dir / "rag_ingest_jobs.db")

        # Active worker threads keyed by job_id
        self._workers: dict[str, threading.Thread] = {}
        self._cancel_flags: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        # The background pool the jobs chunk in, and the task they send it
        # (None: the chunker's own module task).
        self._pool = _background_pool
        self._chunk_task: Any = None
        # The device (st_dev) of each file a job has sent the pool, by job,
        # with the path of the first file sent from it, and how a file's
        # device and a device's disks are read: the job's status names the
        # disks the background reads.
        self._devices: dict[str, dict[int, str]] = {}
        self._device_of: Any = _device_of
        self._disks_of: Any = _disks_of
        # The estimated parse of each file in flight, by doc id, for every job
        # of this engine: the room the governor gives the background is
        # shared, and a job waiting for room waits on _room.
        self._weights: dict[str, int] = {}
        self._room = threading.Condition(self._lock)
        self._waiting = 0  # jobs waiting for room, served before the others send more

    # -----------------------------------------------------------------
    # JOB CREATION
    # -----------------------------------------------------------------

    def create_batch_job(
        self,
        filepaths: list[str | Path],
        collection: str = "default",
    ) -> IngestJobRecord:
        """
        Create a batch ingestion job from a list of file paths.

        The job is created in PENDING status. Call start_job() to begin.

        Args:
            filepaths: List of file paths to ingest.
            collection: Target RAG collection name.

        Returns:
            The created IngestJobRecord.
        """
        job_id = uuid.uuid4().hex[:12]
        self.db.create_job(
            job_id=job_id,
            collection=collection,
            source_type="batch",
        )

        for fp_raw in filepaths:
            fp = Path(fp_raw).resolve()
            if not fp.is_file():
                logger.warning("Skipping non-existent file: %s", fp)
                continue

            file_id = uuid.uuid4().hex[:12]
            try:
                file_size = fp.stat().st_size
            except OSError:
                file_size = 0

            self.db.add_file(
                file_id=file_id,
                job_id=job_id,
                filepath=str(fp),
                filename=fp.name,
                file_size=file_size,
            )

        self.db.update_job_counters(job_id)
        return self.db.get_job(job_id)

    def create_folder_job(
        self,
        directory: str | Path,
        collection: str = "default",
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> IngestJobRecord:
        """
        Create a batch ingestion job by scanning a directory.

        Args:
            directory: Folder to scan.
            collection: Target RAG collection name.
            recursive: Recurse into subdirectories.
            extensions: Allowed file extensions (default: SUPPORTED_EXTENSIONS).

        Returns:
            The created IngestJobRecord.
        """
        directory = Path(directory).resolve()
        files = scan_folder(directory, recursive=recursive, extensions=extensions)

        job_id = uuid.uuid4().hex[:12]
        self.db.create_job(
            job_id=job_id,
            collection=collection,
            source_type="folder",
            source_path=str(directory),
        )

        for fp in files:
            file_id = uuid.uuid4().hex[:12]
            try:
                file_size = fp.stat().st_size
            except OSError:
                file_size = 0

            self.db.add_file(
                file_id=file_id,
                job_id=job_id,
                filepath=str(fp),
                filename=fp.name,
                file_size=file_size,
            )

        self.db.update_job_counters(job_id)
        return self.db.get_job(job_id)

    # -----------------------------------------------------------------
    # JOB CONTROL
    # -----------------------------------------------------------------

    def start_job(self, job_id: str) -> bool:
        """
        Start processing a pending job in a background thread.

        Returns:
            True if the job was started, False if already running or not found.
        """
        job = self.db.get_job(job_id)
        if not job:
            return False
        if job.status not in (JobStatus.PENDING.value,):
            return False

        with self._lock:
            if job_id in self._workers and self._workers[job_id].is_alive():
                return False

            cancel_event = threading.Event()
            self._cancel_flags[job_id] = cancel_event

            thread = threading.Thread(
                target=self._worker_loop,
                args=(job_id, cancel_event),
                daemon=True,
                name=f"batch-ingest-{job_id}",
            )
            self._workers[job_id] = thread
            thread.start()

        return True

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running or pending job.

        Returns:
            True if cancelled, False if not found or already finished.
        """
        job = self.db.get_job(job_id)
        if not job:
            return False
        if job.status in (
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        ):
            return False

        # Signal the worker to stop
        with self._lock:
            if job_id in self._cancel_flags:
                self._cancel_flags[job_id].set()

        self.db.update_job_status(job_id, JobStatus.CANCELLED.value)
        return True

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job and its file records.

        Running jobs are cancelled first.

        Returns:
            True if deleted.
        """
        # Cancel if running
        with self._lock:
            if job_id in self._cancel_flags:
                self._cancel_flags[job_id].set()
            self._devices.pop(job_id, None)

        return self.db.delete_job(job_id)

    # -----------------------------------------------------------------
    # JOB QUERIES
    # -----------------------------------------------------------------

    def get_job(self, job_id: str) -> IngestJobRecord | None:
        """Get a job with its file records."""
        return self.db.get_job(job_id)

    def list_jobs(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IngestJobRecord]:
        """List ingestion jobs."""
        return self.db.list_jobs(status=status, limit=limit, offset=offset)

    def job_disks(self, job_id: str) -> list[dict[str, Any]]:
        """The disks behind each device the job has sent the background a
        file from, in device order, each with what it does with the idle
        I/O class the workers read in; none for a job that has sent none
        since the server started. Each device is read with the path of the
        first file sent from it, so a file system whose files carry a
        device number no mount shows is found by where the file lies. A
        device whose disks cannot be read is said unknown, with why."""
        with self._lock:
            noted = sorted(self._devices.get(job_id, {}).items())
        named = []
        for device, path in noted:
            try:
                named.append(self._disks_of(device, path=path))
            except Exception as exc:  # noqa: BLE001 - one unreadable device leaves the others named
                named.append({
                    "device": f"{os.major(device)}:{os.minor(device)}",
                    "disks": [],
                    "policy": None,
                    "idle_class": "unknown",
                    "reason": str(exc),
                })
        return named

    # -----------------------------------------------------------------
    # BACKGROUND WORKER
    # -----------------------------------------------------------------

    def _worker_loop(self, job_id: str, cancel_event: threading.Event) -> None:
        """Background thread: process files one by one."""
        logger.info("Batch ingestion worker started for job %s", job_id)
        self.db.update_job_status(job_id, JobStatus.RUNNING.value)

        # Get RAG store (lazy)
        store = self._get_rag_store()
        if store is None:
            self.db.update_job_status(
                job_id, JobStatus.FAILED.value,
                error_message="RAG store unavailable",
            )
            logger.error("Batch ingestion failed: RAG store unavailable")
            return

        job = self.db.get_job(job_id)
        if not job:
            return

        try:
            self._run_files(job_id, job.collection, store, cancel_event)

            # Determine final status
            if cancel_event.is_set():
                self.db.update_job_status(job_id, JobStatus.CANCELLED.value)
                logger.info("Batch ingestion job %s cancelled", job_id)
            else:
                self.db.update_job_status(job_id, JobStatus.COMPLETED.value)
                logger.info("Batch ingestion job %s completed", job_id)

        except Exception as exc:
            logger.error("Batch ingestion job %s failed: %s", job_id, exc)
            self.db.update_job_status(
                job_id, JobStatus.FAILED.value,
                error_message=str(exc),
            )
        finally:
            # Cleanup references
            with self._lock:
                self._workers.pop(job_id, None)
                self._cancel_flags.pop(job_id, None)

    def _run_files(
        self,
        job_id: str,
        collection: str,
        store: Any,
        cancel_event: threading.Event,
    ) -> None:
        """Chunk ahead in the background pool; store in order on this thread.

        Up to the pool's in-flight limit of files are chunked at once, each
        by a background worker; this thread stores each result as it comes,
        oldest first, under a ticket the governor admits for the "index"
        caller. A worker holds a whole document as it parses it, so a file is
        sent only while its estimated parse (its size times the plan's
        factor for its kind), with those of the files in flight of every
        job of this engine, fits the room the governor gives the background
        (the RAM available less the reserve); a file larger than the room
        goes once nothing is in flight, and a job with nothing of its own in
        flight waits for room. A file sent alone is charged as any other.
        While the governor says memory is short, one file is in flight at a
        time, as before the pool.

        The pool is one per server and its workers are shared by every job.
        A task that breaks tells the pool which task broke (an executor
        already replaced is left serving) and makes its file a suspect.
        Suspects are sent one at a time, once nothing of the job is in
        flight, each in a worker of its own that no other task shares (the
        pool's submit_alone): only a file whose own worker dies under it
        fails, by name, and only while the pool is open; a worker the pool's
        own shutdown ends blames nothing, and the job ends, its files given
        back. A task that comes back cancelled (its executor
        retired under it) says nothing of its file, which is sent again
        before any new one.
        Cancelled, the job gives the files in flight back to the queue and
        does not wait for their chunking: the wait on the head of the flight
        looks at the cancel every slice, and a result that arrives after the
        cancel is not stored. Whatever ends the loop, a cancel or an error
        (the pool refusing a task, a database failing), every file the job
        still holds goes back to the queue and the error still propagates.
        """
        pool = self._pool()
        task = self._chunk_task or _chunk_file_task()
        chunk_size, chunk_overlap = store.chunker_settings()
        factors = self._parse_factors()
        flight: deque = deque()  # (file record, doc id, future, sent alone)
        alone: deque = deque()  # (file record, doc id): suspects, each sent in a worker of its own
        # (file record, doc id): to send before any new file, its task
        # cancelled under the job or its parse not yet fitting the room.
        again: deque = deque()
        sending: IngestFileRecord | None = None  # taken and not yet in flight
        head: tuple | None = None  # out of the flight, its status not yet written
        try:
            while not cancel_event.is_set():
                limit = pool.in_flight_limit()
                while not cancel_event.is_set():
                    if alone:
                        if flight:
                            break
                        sending, doc_id = alone.popleft()
                        weight = _parse_weight(sending.filepath, factors)
                        if not self._reserve(doc_id, weight, wait=True, cancel_event=cancel_event):
                            alone.appendleft((sending, doc_id))
                            sending = None
                            break
                        future = self._send(pool.submit_alone, task, sending.filepath, doc_id, chunk_size, chunk_overlap)
                        flight.append((sending, doc_id, future, True))
                        sending = None
                        continue
                    if len(flight) >= limit:
                        break
                    if flight and self._memory_short():
                        break
                    if again:
                        sending, doc_id = again.popleft()
                    else:
                        sending = self._take_file(job_id)
                        if sending is None:
                            break
                        doc_id = uuid.uuid4().hex[:12]
                        self._note_device(job_id, sending.filepath)
                    weight = _parse_weight(sending.filepath, factors)
                    if not self._reserve(doc_id, weight, wait=not flight, cancel_event=cancel_event):
                        again.appendleft((sending, doc_id))
                        sending = None
                        break
                    future = self._send(pool.submit, task, sending.filepath, doc_id, chunk_size, chunk_overlap)
                    flight.append((sending, doc_id, future, False))
                    sending = None
                if not flight:
                    break
                head = flight.popleft()
                file_rec, doc_id, future, solo = head
                if not _finished(future, cancel_event):
                    break
                # The parse is over, whatever came of it: its room is free.
                self._release(doc_id)
                try:
                    result = future.result()
                except CancelledError:
                    if solo:
                        alone.appendleft((file_rec, doc_id))
                    else:
                        again.append((file_rec, doc_id))
                    head = None
                    continue
                except BrokenProcessPool:
                    if not solo:
                        pool.reset(future)
                        alone.append((file_rec, doc_id))
                        head = None
                        continue
                    if getattr(pool, "closed", False):
                        # The pool's own shutdown ended the worker (the server
                        # stops): the file is given back, not blamed.
                        raise RuntimeError("the background pool is shut down") from None
                    self._fail(file_rec, f"the background worker died while chunking {file_rec.filename}")
                except Exception as exc:
                    self._fail(file_rec, str(exc))
                else:
                    self._store_file(file_rec, result, store, collection, cancel_event)
                head = None
                self.db.update_job_counters(job_id)
        finally:
            self._give_back(flight, alone, again, sending, head)

    def _give_back(
        self,
        flight: deque,
        alone: deque,
        again: deque,
        sending: IngestFileRecord | None,
        head: tuple | None,
    ) -> None:
        """Put back in the queue every file the job still holds, whatever
        ended its loop: the head whose status was not written and the files
        in flight (their tasks cancelled), those waiting to be sent alone or
        again, and the one being sent. Each is given back on its own: one
        that cannot be leaves the others to be. The room their parses held
        is freed for the other jobs once each parse is over: a task already
        running cannot be cancelled, and its parse goes on in its worker."""
        held = []
        for file_rec, doc_id, future, _solo in ([head] if head is not None else []) + list(flight):
            future.cancel()
            # At once when cancelled or done; when its running parse ends.
            future.add_done_callback(functools.partial(self._parse_over, doc_id))
            held.append(file_rec)
        held.extend(file_rec for file_rec, _doc_id in (*alone, *again))
        if sending is not None:
            held.append(sending)
        for file_rec in held:
            try:
                self.db.requeue_file(file_rec.file_id)
            except Exception as exc:  # noqa: BLE001 - the other files are still given back
                logger.warning("Could not give %s back to the queue: %s", file_rec.filename, exc)

    def _reserve(self, doc_id: str, weight: int, *, wait: bool, cancel_event: threading.Event) -> bool:
        """Charge ``weight``, the estimated parse of ``doc_id``, against the
        room the governor gives the background, which every job of this
        engine shares; True once charged. It fits when the room bounds
        nothing, when nothing of any job is in flight (a file larger than the
        room goes alone), or when the weights in flight and its own fit.
        Not fitting: False at once, or, with ``wait`` (the job has nothing of
        its own in flight), wait for another job's file to leave, looking at
        the cancel every slice, and False once cancelled. While a job waits
        so, a job with files in flight sends no more, even where its file
        would fit: the room goes first to the job that waits."""
        waiting = False
        try:
            while True:
                room = self._memory_room()
                with self._room:
                    fits = room is None or not self._weights or sum(self._weights.values()) + weight <= room
                    if fits and (wait or not self._waiting):
                        self._weights[doc_id] = weight
                        return True
                    if not wait:
                        return False
                    if not waiting:
                        self._waiting += 1
                        waiting = True
                    self._room.wait(timeout=_HEAD_WAIT_SLICE_S)
                if cancel_event.is_set():
                    return False
        finally:
            if waiting:
                with self._room:
                    self._waiting -= 1
                    self._room.notify_all()

    def _release(self, doc_id: str) -> None:
        """Take ``doc_id``'s parse off the shared room and wake the jobs that
        wait for room."""
        with self._room:
            if self._weights.pop(doc_id, None) is not None:
                self._room.notify_all()

    def _parse_over(self, doc_id: str, _future: Any) -> None:
        """A task given back is done (cancelled, or its parse ended): its
        room is free."""
        self._release(doc_id)

    def _send(self, entry: Any, task: Any, path: str, doc_id: str, chunk_size: int, chunk_overlap: int) -> Any:
        """Send ``path`` through the pool's ``entry`` (submit or
        submit_alone); a send that raises takes its parse off the room."""
        try:
            return entry(task, path, doc_id, chunk_size, chunk_overlap, caller=INDEX_CALLER)
        except BaseException:
            self._release(doc_id)
            raise

    def _note_device(self, job_id: str, path: str) -> None:
        """Note the device ``path`` is read from, with the path when it is the
        first file sent from that device, for the job's status; a file that
        cannot be read now is not noted (its chunking says why)."""
        try:
            device = self._device_of(path)
        except OSError:
            return
        with self._lock:
            self._devices.setdefault(job_id, {}).setdefault(device, path)

    def _memory_short(self) -> str | None:
        """Why memory is short for the background, as the governor says it
        (background_memory_short); None when it is not, and when there is no
        governor to ask, it has no such word, or the asking fails."""
        rg = _governor_module()
        if rg is None:
            return None
        try:
            ask = getattr(rg.get_resource_governor(), "background_memory_short", None)
            reason = ask() if callable(ask) else None
        except Exception as exc:  # noqa: BLE001 - no word on memory is no reason to wait
            logger.debug("Background memory state unavailable to the index: %s", exc)
            return None
        return reason if isinstance(reason, str) and reason else None

    def _memory_room(self) -> int | None:
        """The memory the governor gives background work now, in bytes
        (background_memory_room); None when it bounds nothing: no governor,
        no such word, or the asking fails."""
        rg = _governor_module()
        if rg is None:
            return None
        try:
            ask = getattr(rg.get_resource_governor(), "background_memory_room", None)
            room = ask() if callable(ask) else None
        except Exception as exc:  # noqa: BLE001 - no word on memory bounds nothing
            logger.debug("Background memory room unavailable to the index: %s", exc)
            return None
        return room if isinstance(room, int) and not isinstance(room, bool) and room >= 0 else None

    def _parse_factors(self) -> dict[str, float]:
        """How many times its size a file's parse takes in memory, by
        extension, as the background plan says (parse_expansion); none when
        there is no plan to read, and a file then weighs nothing."""
        rg = _governor_module()
        if rg is None:
            return {}
        try:
            factors = getattr(rg.get_resource_governor().plan_background(), "parse_expansion", None)
        except Exception as exc:  # noqa: BLE001 - no factors: nothing is weighed
            logger.debug("Background parse factors unavailable to the index: %s", exc)
            return {}
        return dict(factors) if isinstance(factors, dict) else {}

    def _take_file(self, job_id: str) -> IngestFileRecord | None:
        """The next queued file, marked processing; a file gone from disk
        is skipped and the next one taken."""
        while True:
            file_rec = self.db.get_next_queued_file(job_id)
            if file_rec is None:
                return None
            self.db.update_file_status(file_rec.file_id, FileStatus.PROCESSING.value)
            if Path(file_rec.filepath).is_file():
                return file_rec
            self.db.update_file_status(file_rec.file_id, FileStatus.SKIPPED.value, error_message="File not found")
            self.db.update_job_counters(job_id)

    def _store_file(
        self,
        file_rec: IngestFileRecord,
        result: Any,
        store: Any,
        collection: str,
        cancel_event: threading.Event,
    ) -> None:
        """Store one chunked file on this thread, under the governor's
        ticket for the "index" caller."""
        decision = self._admit(store, cancel_event)
        if decision is _CANCELLED:
            self.db.requeue_file(file_rec.file_id)
            return
        if decision is not None and not getattr(decision, "admitted", True):
            self._fail(file_rec, f"held by the resource governor: {getattr(decision, 'reason', '')}")
            return
        try:
            with _holding(decision):
                doc = store.store_chunked(
                    result,
                    collection=collection,
                    metadata={"batch_job_id": file_rec.job_id, "original_filename": file_rec.filename},
                )
        except Exception as exc:
            self._fail(file_rec, str(exc))
            return
        self.db.update_file_status(
            file_rec.file_id,
            FileStatus.DONE.value,
            doc_id=doc.doc_id,
            chunk_count=doc.chunk_count,
        )
        logger.debug("Ingested %s: %d chunks", file_rec.filename, doc.chunk_count)

    def _admit(self, store: Any, cancel_event: threading.Event) -> Any:
        """The governor's decision for this file's embeddings, asked as the
        background "index" caller through the queue that lets it wait.

        A refusal a wait can lift is asked again after the plan's pause, so
        the file waits for the machine rather than fails; a refusal no wait
        can lift is returned. The governor is given the job's cancel, so a
        cancel takes the job out of its queue, and the cancel is looked at
        before each ask and after each answer. None when there is no
        governor or no model to ask for (the engine head then admits the
        call on its own); _CANCELLED when the job is cancelled while it
        waits.
        """
        rg = _governor_module()
        model = getattr(store, "embedding_model", None)
        if rg is None or not model:
            return None
        try:
            governor = rg.get_resource_governor()
        except Exception as exc:  # noqa: BLE001 - fail open, as every funnel
            logger.debug("Resource governor unavailable to the index: %s", exc)
            return None
        while True:
            if cancel_event.is_set():
                return _CANCELLED
            try:
                decision = _admission(governor, model, cancel_event)
            except Exception as exc:  # noqa: BLE001 - fail open, as every funnel
                logger.warning("Index admission failed; the engine admits the call itself: %s", exc)
                return None
            if cancel_event.is_set():
                _hand_back(governor, decision)
                return _CANCELLED
            if decision is None or getattr(decision, "admitted", True) or rg.refusal_is_final(decision):
                return decision
            pause = _held_retry_s(governor)
            if pause is None:
                return decision
            if cancel_event.wait(pause):
                return _CANCELLED

    def _fail(self, file_rec: IngestFileRecord, message: str) -> None:
        logger.error("Failed to ingest %s: %s", file_rec.filename, message)
        self.db.update_file_status(file_rec.file_id, FileStatus.ERROR.value, error_message=message[:500])

    def _get_rag_store(self) -> Any:
        """Get the RAGVectorStore, returning None if unavailable."""
        try:
            from opti_oignon.rag_store import get_rag_store
            return get_rag_store()
        except Exception as exc:
            logger.error("Cannot get RAG store: %s", exc)
            return None


# =========================================================================
# BACKGROUND SEAMS
# =========================================================================

def _background_pool() -> Any:
    """The server's background pool."""
    from opti_oignon.background_pool import get_background_pool

    return get_background_pool()


def _device_of(path: str) -> int:
    """The device a file is read from (its st_dev, through any link)."""
    return os.stat(path).st_dev


def _disks_of(device: int, path: str | None = None) -> dict[str, Any]:
    """The disks behind ``device``, ``path`` a file on it, and what each
    does with the idle I/O class the background reads in."""
    from opti_oignon.background_pool import device_disks

    return device_disks(device, path=path)


def _chunk_file_task() -> Any:
    """The chunker's own task, the one a background worker runs."""
    from opti_oignon.rag_chunker import chunk_file_task

    return chunk_file_task


def _governor_module() -> Any:
    """The resource governor's module, or None when it cannot be loaded."""
    try:
        return importlib.import_module("opti_oignon.resource_governor")
    except Exception as exc:  # noqa: BLE001 - absence is an answer here
        logger.debug("Resource governor unavailable to the index: %s", exc)
        return None


def _admission(governor: Any, model: str, cancel_event: threading.Event) -> Any:
    """The governor's answer for the index's embeddings, the job's cancel
    given so a cancel takes it out of the queue; a governor whose admission
    takes no cancel is asked again without one."""
    try:
        return governor.admit_or_wait(model, caller=INDEX_CALLER, cancel=cancel_event)
    except TypeError as exc:
        if "cancel" not in str(exc):
            raise
    return governor.admit_or_wait(model, caller=INDEX_CALLER)


def _parse_weight(path: str, factors: dict[str, float]) -> int:
    """The memory the parse of ``path`` is estimated to take, in bytes: its
    size times the factor for its extension, else the default; 0 when its
    size cannot be read or no factor applies (it then weighs nothing)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return 0
    factor = factors.get(os.path.splitext(path)[1].lower(), factors.get("default"))
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor <= 0:
        return 0
    return int(size * factor)


def _hand_back(governor: Any, decision: Any) -> None:
    """An admission the job will not use, its cancel having landed as the
    governor answered: the load it admitted will not happen, so the governor
    is told (end_pending_load) rather than count it until it expires. A
    refusal, or a governor that cannot be told, leaves nothing to do."""
    if decision is None or not getattr(decision, "admitted", False):
        return
    end = getattr(governor, "end_pending_load", None)
    ticket = getattr(decision, "ticket_id", None)
    if not callable(end) or not ticket:
        return
    try:
        end(ticket)
    except Exception as exc:  # noqa: BLE001 - the load then expires on its own
        logger.debug("Index admission not handed back: %s", exc)


def _finished(future: Any, cancel_event: threading.Event) -> bool:
    """Wait for ``future`` in slices, looking at the cancel between them:
    True once it is done and the job not cancelled, False as soon as the
    job is cancelled, whether or not its result has arrived."""
    while not future.done():
        if cancel_event.is_set():
            return False
        wait([future], timeout=_HEAD_WAIT_SLICE_S)
    return not cancel_event.is_set()


def _held_retry_s(governor: Any) -> float | None:
    """The plan's pause before asking again, or None when it has none."""
    try:
        pause = governor.plan_background().held_retry_s
    except Exception as exc:  # noqa: BLE001 - no pause known: the refusal stands
        logger.debug("Background plan unavailable to the index: %s", exc)
        return None
    if isinstance(pause, bool) or not isinstance(pause, (int, float)) or pause < 0:
        return None
    return float(pause)


@contextmanager
def _holding(decision: Any):
    """Hold the index's own ticket around the store call, so the embed head
    accounts the admission it was given instead of admitting again."""
    if decision is None:
        yield
        return
    with _governor_module().ticket_scope(decision):
        yield


# =========================================================================
# MODULE-LEVEL SINGLETON
# =========================================================================

_engine_instance: BatchIngestEngine | None = None
_engine_lock = threading.Lock()


def get_batch_ingest_engine(
    data_dir: str | Path | None = None,
) -> BatchIngestEngine:
    """Return the module-level BatchIngestEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = BatchIngestEngine(data_dir=data_dir)
    return _engine_instance
