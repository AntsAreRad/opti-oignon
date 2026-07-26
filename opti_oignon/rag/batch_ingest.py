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

import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
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
            while not cancel_event.is_set():
                file_rec = self.db.get_next_queued_file(job_id)
                if file_rec is None:
                    break  # All files processed

                self._process_file(file_rec, store, job.collection, cancel_event)
                self.db.update_job_counters(job_id)

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

    def _process_file(
        self,
        file_rec: IngestFileRecord,
        store: Any,
        collection: str,
        cancel_event: threading.Event,
    ) -> None:
        """Process a single file: ingest into the RAG store."""
        fp = Path(file_rec.filepath)

        # Mark as processing
        self.db.update_file_status(file_rec.file_id, FileStatus.PROCESSING.value)

        # Check file still exists
        if not fp.is_file():
            self.db.update_file_status(
                file_rec.file_id,
                FileStatus.SKIPPED.value,
                error_message="File not found",
            )
            return

        # Check cancellation
        if cancel_event.is_set():
            return

        try:
            doc = store.ingest_file(
                filepath=str(fp),
                collection=collection,
                metadata={"batch_job_id": file_rec.job_id, "original_filename": file_rec.filename},
            )
            self.db.update_file_status(
                file_rec.file_id,
                FileStatus.DONE.value,
                doc_id=doc.doc_id,
                chunk_count=doc.chunk_count,
            )
            logger.debug("Ingested %s: %d chunks", fp.name, doc.chunk_count)

        except Exception as exc:
            logger.error("Failed to ingest %s: %s", fp.name, exc)
            self.db.update_file_status(
                file_rec.file_id,
                FileStatus.ERROR.value,
                error_message=str(exc)[:500],
            )

    def _get_rag_store(self) -> Any:
        """Get the RAGVectorStore, returning None if unavailable."""
        try:
            from opti_oignon.rag_store import get_rag_store
            return get_rag_store()
        except Exception as exc:
            logger.error("Cannot get RAG store: %s", exc)
            return None


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
