#!/usr/bin/env python3
"""
Parallel RAG ingestion engine (S160).

Provides concurrent chunk processing for RAG document ingestion using
``concurrent.futures.ThreadPoolExecutor``.  Each chunk is processed
independently so that a single failure does not abort the entire batch.

Key features:
- Configurable parallelism level (default: 4 workers)
- Per-chunk error isolation with detailed error reporting
- Progress tracking callback (called after each chunk completes)
- Batch-level statistics (success/fail/skip counts, elapsed time)
- Thread-safe result aggregation
- Integration point for ``ConnectionPool`` from S159

Usage::

    from opti_oignon.rag.parallel_ingest import ParallelIngestWorker

    worker = ParallelIngestWorker(max_workers=4)
    results = worker.ingest_chunks(
        chunks=chunk_list,
        embed_fn=my_embedder.embed_single,
        store_fn=my_store.add_chunk,
        progress_cb=lambda done, total: print(f"{done}/{total}"),
    )
    print(results.summary())
"""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, Sequence

logger = logging.getLogger(__name__)

# Hardcoded, never overridable
checkpoint_before_apply = True

FEATURE_AVAILABLE = True

# -- Constants ---------------------------------------------------------------

DEFAULT_MAX_WORKERS: int = 4
DEFAULT_CHUNK_TIMEOUT_S: float = 120.0


# -- Enums and data structures -----------------------------------------------

class ChunkStatus(str, Enum):
    """Processing status for an individual chunk."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ChunkResult:
    """Result of processing a single chunk."""

    chunk_index: int
    chunk_id: str
    status: ChunkStatus
    elapsed_s: float = 0.0
    error_message: str | None = None
    embedding_dim: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        d: dict[str, Any] = {
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "status": self.status.value,
            "elapsed_s": round(self.elapsed_s, 4),
        }
        if self.error_message:
            d["error_message"] = self.error_message
        if self.embedding_dim is not None:
            d["embedding_dim"] = self.embedding_dim
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class IngestBatchResult:
    """Aggregated results for a parallel ingestion batch."""

    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    elapsed_s: float = 0.0
    chunk_results: list[ChunkResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict."""
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "skipped": self.skipped,
            "elapsed_s": round(self.elapsed_s, 3),
            "success_rate": round(self.success / self.total, 4) if self.total else 0.0,
        }

    @property
    def all_succeeded(self) -> bool:
        """True if every chunk succeeded."""
        return self.failed == 0 and self.skipped == 0

    def failed_chunks(self) -> list[ChunkResult]:
        """Return only the failed chunk results."""
        return [r for r in self.chunk_results if r.status == ChunkStatus.FAILED]


# -- Chunk protocol ----------------------------------------------------------

class ChunkLike(Protocol):
    """Minimal protocol for objects that can be treated as chunks."""

    @property
    def chunk_id(self) -> str: ...

    @property
    def text(self) -> str: ...


@dataclass
class SimpleChunk:
    """Lightweight chunk container for cases where no full Chunk object exists."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# -- Progress tracker --------------------------------------------------------

class ProgressTracker:
    """Thread-safe progress counter with optional callback."""

    def __init__(
        self,
        total: int,
        callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self._total = total
        self._done = 0
        self._lock = threading.Lock()
        self._callback = callback

    @property
    def total(self) -> int:
        return self._total

    @property
    def done(self) -> int:
        with self._lock:
            return self._done

    def increment(self) -> int:
        """Increment the done counter and fire callback. Returns new count."""
        with self._lock:
            self._done += 1
            current = self._done
        if self._callback:
            try:
                self._callback(current, self._total)
            except Exception as exc:
                logger.debug("Progress callback error: %s", exc)
        return current


# -- Parallel ingest worker --------------------------------------------------

class ParallelIngestWorker:
    """Processes RAG chunks in parallel using a thread pool.

    Parameters
    ----------
    max_workers : int
        Maximum number of concurrent worker threads (default: 4).
    chunk_timeout : float
        Per-chunk processing timeout in seconds (default: 120).
    """

    def __init__(
        self,
        *,
        max_workers: int = DEFAULT_MAX_WORKERS,
        chunk_timeout: float = DEFAULT_CHUNK_TIMEOUT_S,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._max_workers = max_workers
        self._chunk_timeout = chunk_timeout

    @property
    def max_workers(self) -> int:
        """Maximum number of concurrent workers."""
        return self._max_workers

    @property
    def chunk_timeout(self) -> float:
        """Per-chunk timeout in seconds."""
        return self._chunk_timeout

    def ingest_chunks(
        self,
        chunks: Sequence[Any],
        embed_fn: Callable[[str], list[float] | None],
        store_fn: Callable[[str, list[float], dict[str, Any]], Any],
        *,
        progress_cb: Callable[[int, int], None] | None = None,
        skip_empty: bool = True,
    ) -> IngestBatchResult:
        """Process a batch of chunks in parallel.

        Each chunk is independently embedded via *embed_fn* and stored
        via *store_fn*.  A failure in one chunk does not affect others.

        Parameters
        ----------
        chunks : sequence
            Objects with ``.chunk_id`` and ``.text`` attributes, or dicts
            with ``"chunk_id"`` and ``"text"`` keys.
        embed_fn : callable
            ``(text: str) -> list[float] | None`` -- produces an embedding
            vector for the chunk text.
        store_fn : callable
            ``(chunk_id: str, embedding: list[float], metadata: dict) -> Any``
            -- persists the chunk with its embedding.
        progress_cb : callable, optional
            ``(done: int, total: int) -> None`` -- called after each chunk.
        skip_empty : bool
            If True, chunks with empty text are skipped instead of failing.

        Returns
        -------
        IngestBatchResult
            Aggregated results with per-chunk detail.
        """
        total = len(chunks)
        if total == 0:
            return IngestBatchResult()

        tracker = ProgressTracker(total, callback=progress_cb)
        results: list[ChunkResult] = [None] * total  # type: ignore[list-item]
        start_time = time.monotonic()

        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, total),
            thread_name_prefix="oo-rag-ingest",
        ) as pool:
            future_to_index: dict[Future[ChunkResult], int] = {}  # type: ignore[type-arg]
            for i, chunk in enumerate(chunks):
                fut = pool.submit(
                    self._process_single_chunk,
                    index=i,
                    chunk=chunk,
                    embed_fn=embed_fn,
                    store_fn=store_fn,
                    tracker=tracker,
                    skip_empty=skip_empty,
                )
                future_to_index[fut] = i

            for fut in as_completed(future_to_index):
                idx = future_to_index[fut]
                try:
                    results[idx] = fut.result(timeout=self._chunk_timeout)
                except Exception as exc:
                    results[idx] = ChunkResult(
                        chunk_index=idx,
                        chunk_id=f"unknown-{idx}",
                        status=ChunkStatus.FAILED,
                        error_message=f"Future error: {exc}",
                    )

        elapsed = time.monotonic() - start_time

        # Aggregate
        batch = IngestBatchResult(
            total=total,
            elapsed_s=elapsed,
            chunk_results=results,
        )
        for r in results:
            if r.status == ChunkStatus.SUCCESS:
                batch.success += 1
            elif r.status == ChunkStatus.FAILED:
                batch.failed += 1
            elif r.status == ChunkStatus.SKIPPED:
                batch.skipped += 1

        logger.info(
            "Parallel ingest complete: %d/%d success, %d failed, %d skipped (%.2fs, %d workers)",
            batch.success, batch.total, batch.failed, batch.skipped,
            batch.elapsed_s, self._max_workers,
        )
        return batch

    def _process_single_chunk(
        self,
        index: int,
        chunk: Any,
        embed_fn: Callable[[str], list[float] | None],
        store_fn: Callable[[str, list[float], dict[str, Any]], Any],
        tracker: ProgressTracker,
        skip_empty: bool,
    ) -> ChunkResult:
        """Process one chunk: extract text, embed, store."""
        t0 = time.monotonic()

        # Extract chunk_id and text
        chunk_id, text, meta = self._extract_chunk_data(chunk, index)

        # Skip empty
        if skip_empty and not text.strip():
            tracker.increment()
            return ChunkResult(
                chunk_index=index,
                chunk_id=chunk_id,
                status=ChunkStatus.SKIPPED,
                elapsed_s=time.monotonic() - t0,
                error_message="Empty text",
            )

        try:
            # Embed
            embedding = embed_fn(text)
            if embedding is None:
                tracker.increment()
                return ChunkResult(
                    chunk_index=index,
                    chunk_id=chunk_id,
                    status=ChunkStatus.FAILED,
                    elapsed_s=time.monotonic() - t0,
                    error_message="Embedding returned None",
                )

            # Store
            store_fn(chunk_id, embedding, meta)

            tracker.increment()
            return ChunkResult(
                chunk_index=index,
                chunk_id=chunk_id,
                status=ChunkStatus.SUCCESS,
                elapsed_s=time.monotonic() - t0,
                embedding_dim=len(embedding),
                metadata=meta,
            )

        except Exception as exc:
            tracker.increment()
            return ChunkResult(
                chunk_index=index,
                chunk_id=chunk_id,
                status=ChunkStatus.FAILED,
                elapsed_s=time.monotonic() - t0,
                error_message=str(exc)[:500],
            )

    @staticmethod
    def _extract_chunk_data(
        chunk: Any, index: int,
    ) -> tuple[str, str, dict[str, Any]]:
        """Extract chunk_id, text, and metadata from a chunk-like object.

        Supports:
        - Objects with ``.chunk_id`` and ``.text`` attributes
        - Dicts with ``"chunk_id"`` and ``"text"`` keys
        - Plain strings (chunk_id auto-generated)
        """
        if isinstance(chunk, str):
            return f"chunk-{index}", chunk, {"source_index": index}

        if isinstance(chunk, dict):
            cid = chunk.get("chunk_id", f"chunk-{index}")
            text = chunk.get("text", "")
            meta = {k: v for k, v in chunk.items() if k not in ("chunk_id", "text")}
            meta["source_index"] = index
            return cid, text, meta

        # Object with attributes
        cid = getattr(chunk, "chunk_id", f"chunk-{index}")
        text = getattr(chunk, "text", "")
        meta: dict[str, Any] = {"source_index": index}
        if hasattr(chunk, "metadata"):
            chunk_meta = chunk.metadata
            if isinstance(chunk_meta, dict):
                meta.update(chunk_meta)
        return cid, text, meta


# -- Convenience function ----------------------------------------------------

def parallel_ingest(
    chunks: Sequence[Any],
    embed_fn: Callable[[str], list[float] | None],
    store_fn: Callable[[str, list[float], dict[str, Any]], Any],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    progress_cb: Callable[[int, int], None] | None = None,
) -> IngestBatchResult:
    """One-shot convenience wrapper around :class:`ParallelIngestWorker`.

    Parameters
    ----------
    chunks : sequence
        Chunk objects, dicts, or plain strings.
    embed_fn : callable
        Embedding function ``(text) -> list[float] | None``.
    store_fn : callable
        Storage function ``(chunk_id, embedding, metadata) -> Any``.
    max_workers : int
        Parallelism level (default: 4).
    progress_cb : callable, optional
        Progress callback ``(done, total) -> None``.

    Returns
    -------
    IngestBatchResult
    """
    worker = ParallelIngestWorker(max_workers=max_workers)
    return worker.ingest_chunks(
        chunks, embed_fn, store_fn,
        progress_cb=progress_cb,
    )
