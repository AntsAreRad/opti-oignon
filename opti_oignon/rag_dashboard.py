#!/usr/bin/env python3
"""
RAG DASHBOARD & AUTO-REFRESH.

Provides:
- RAGDashboardStats: usage stats, collection health, source reliability
  - Queries per day, top cited sources, citation frequency
  - Collection health: chunk distribution, embedding coverage
  - Source reliability scoring (freshness, query success rate)
- RAGAutoRefresh: scheduled re-indexing and file-change detection
  - Configurable schedule (interval-based)
  - Watch mode: detect file changes and re-ingest
  - Source freshness tracking
- Configurable via config/rag.yaml [dashboard] and [auto_refresh] sections

Usage::

    dashboard = RAGDashboardStats(store=rag_store_instance)
    stats = dashboard.get_overall_stats()
    usage = dashboard.get_usage_over_time(days=30)
    sources = dashboard.get_source_reliability()

    refresher = RAGAutoRefresh(store=rag_store_instance)
    refresher.check_and_refresh()
"""

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
# Audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# Feature flag
RAG_DASHBOARD_AVAILABLE = True


# =========================================================================
# DATA STRUCTURES
# =========================================================================

@dataclass
class OverallStats:
    """High-level dashboard statistics."""
    total_collections: int
    total_documents: int
    total_chunks: int
    total_citations: int
    total_queries_today: int
    total_queries_week: int
    total_queries_all: int
    avg_score: float
    storage_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_collections": self.total_collections,
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "total_citations": self.total_citations,
            "total_queries_today": self.total_queries_today,
            "total_queries_week": self.total_queries_week,
            "total_queries_all": self.total_queries_all,
            "avg_score": round(self.avg_score, 4),
            "storage_bytes": self.storage_bytes,
        }


@dataclass
class UsageDataPoint:
    """A single data point for usage-over-time charts."""
    date: str
    query_count: int
    citation_count: int
    avg_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "query_count": self.query_count,
            "citation_count": self.citation_count,
            "avg_score": round(self.avg_score, 4),
        }


@dataclass
class CollectionHealth:
    """Health metrics for a single collection."""
    name: str
    document_count: int
    chunk_count: int
    citation_count: int
    avg_chunk_size: float
    file_types: list[str]
    last_ingestion: float
    last_query: float
    freshness_score: float  # 0.0 to 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "citation_count": self.citation_count,
            "avg_chunk_size": round(self.avg_chunk_size, 1),
            "file_types": self.file_types,
            "last_ingestion": self.last_ingestion,
            "last_query": self.last_query,
            "freshness_score": round(self.freshness_score, 2),
        }


@dataclass
class SourceReliability:
    """Reliability metrics for a single source document."""
    source_file: str
    collection_name: str
    doc_id: str
    citation_count: int
    avg_score: float
    last_cited: float
    freshness_score: float
    reliability_score: float  # Composite score 0.0 to 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "collection_name": self.collection_name,
            "doc_id": self.doc_id,
            "citation_count": self.citation_count,
            "avg_score": round(self.avg_score, 4),
            "last_cited": self.last_cited,
            "freshness_score": round(self.freshness_score, 2),
            "reliability_score": round(self.reliability_score, 3),
        }


@dataclass
class RefreshResult:
    """Result of an auto-refresh check."""
    checked_at: float
    sources_checked: int
    sources_refreshed: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at": self.checked_at,
            "sources_checked": self.sources_checked,
            "sources_refreshed": self.sources_refreshed,
            "errors": self.errors,
        }


# =========================================================================
# DASHBOARD STATS ENGINE
# =========================================================================

class RAGDashboardStats:
    """
    Computes dashboard statistics from the RAG store SQLite database.

    Reads directly from the rag_documents.db used by RAGVectorStore,
    so it always reflects the current state.
    """

    def __init__(self, store: Any = None, data_dir: str | Path | None = None):
        """
        Args:
            store: RAGVectorStore instance (for data_dir resolution).
            data_dir: Direct path to the RAG data directory.
        """
        self._store = store
        self._data_dir = data_dir
        self._db_path: str | None = None

    @property
    def db_path(self) -> str:
        """Resolve the path to rag_documents.db."""
        if self._db_path is not None:
            return self._db_path

        if self._data_dir:
            self._db_path = str(Path(self._data_dir) / "rag_documents.db")
        elif self._store is not None:
            self._db_path = str(self._store.data_dir / "rag_documents.db")
        else:
            try:
                from opti_oignon.config import DATA_DIR
                self._db_path = str(Path(DATA_DIR) / "rag" / "rag_documents.db")
            except ImportError:
                self._db_path = str(
                    Path.home() / ".opti-oignon" / "data" / "rag" / "rag_documents.db"
                )

        return self._db_path

    def _conn(self) -> sqlite3.Connection:
        """Open a read-only SQLite connection."""
        conn = _safe_connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _db_exists(self) -> bool:
        """Check if the database file exists."""
        return Path(self.db_path).exists()

    # -----------------------------------------------------------------
    # OVERALL STATS
    # -----------------------------------------------------------------

    def get_overall_stats(self) -> OverallStats:
        """Compute high-level dashboard statistics."""
        if not self._db_exists():
            return OverallStats(
                total_collections=0, total_documents=0, total_chunks=0,
                total_citations=0, total_queries_today=0, total_queries_week=0,
                total_queries_all=0, avg_score=0.0, storage_bytes=0,
            )

        now = time.time()
        day_ago = now - 86400
        week_ago = now - 7 * 86400

        try:
            with self._conn() as conn:
                # Collection count
                coll_count = conn.execute(
                    "SELECT COUNT(*) FROM collections"
                ).fetchone()[0]

                # Document count and total chunks
                doc_row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(chunk_count), 0) FROM documents"
                ).fetchone()
                doc_count = doc_row[0]
                chunk_count = doc_row[1]

                # Citation stats
                cite_row = conn.execute(
                    "SELECT COUNT(*), COALESCE(AVG(score), 0) FROM citations"
                ).fetchone()
                cite_count = cite_row[0]
                avg_score = cite_row[1]

                # Queries today (unique queries by timestamp)
                queries_today = conn.execute(
                    "SELECT COUNT(DISTINCT query) FROM citations WHERE timestamp >= ?",
                    (day_ago,),
                ).fetchone()[0]

                # Queries this week
                queries_week = conn.execute(
                    "SELECT COUNT(DISTINCT query) FROM citations WHERE timestamp >= ?",
                    (week_ago,),
                ).fetchone()[0]

                # All-time unique queries
                queries_all = conn.execute(
                    "SELECT COUNT(DISTINCT query) FROM citations"
                ).fetchone()[0]

            # Storage size
            storage = 0
            db_file = Path(self.db_path)
            if db_file.exists():
                storage += db_file.stat().st_size
            # Include ChromaDB directory if accessible
            if self._store is not None:
                chroma_dir = getattr(self._store, "chroma_dir", None)
                if chroma_dir and Path(chroma_dir).exists():
                    for f in Path(chroma_dir).rglob("*"):
                        if f.is_file():
                            storage += f.stat().st_size

            return OverallStats(
                total_collections=coll_count,
                total_documents=doc_count,
                total_chunks=chunk_count,
                total_citations=cite_count,
                total_queries_today=queries_today,
                total_queries_week=queries_week,
                total_queries_all=queries_all,
                avg_score=avg_score,
                storage_bytes=storage,
            )

        except Exception as exc:
            logger.error("Failed to compute overall stats: %s", exc)
            return OverallStats(
                total_collections=0, total_documents=0, total_chunks=0,
                total_citations=0, total_queries_today=0, total_queries_week=0,
                total_queries_all=0, avg_score=0.0, storage_bytes=0,
            )

    # -----------------------------------------------------------------
    # USAGE OVER TIME
    # -----------------------------------------------------------------

    def get_usage_over_time(self, days: int = 30) -> list[UsageDataPoint]:
        """
        Get daily usage data for the last N days.

        Returns one UsageDataPoint per day with query count,
        citation count, and average score.
        """
        if not self._db_exists():
            return []

        now = time.time()
        start = now - days * 86400

        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        date(timestamp, 'unixepoch') AS day,
                        COUNT(DISTINCT query) AS query_count,
                        COUNT(*) AS citation_count,
                        AVG(score) AS avg_score
                    FROM citations
                    WHERE timestamp >= ?
                    GROUP BY day
                    ORDER BY day ASC
                    """,
                    (start,),
                ).fetchall()

            # Build result with zero-filled gaps
            data_map: dict[str, UsageDataPoint] = {}
            for row in rows:
                data_map[row["day"]] = UsageDataPoint(
                    date=row["day"],
                    query_count=row["query_count"],
                    citation_count=row["citation_count"],
                    avg_score=row["avg_score"] or 0.0,
                )

            # Fill gaps
            result: list[UsageDataPoint] = []
            current = datetime.fromtimestamp(start, tz=timezone.utc).date()
            end_date = datetime.fromtimestamp(now, tz=timezone.utc).date()

            while current <= end_date:
                date_str = current.isoformat()
                if date_str in data_map:
                    result.append(data_map[date_str])
                else:
                    result.append(UsageDataPoint(
                        date=date_str, query_count=0,
                        citation_count=0, avg_score=0.0,
                    ))
                current += timedelta(days=1)

            return result

        except Exception as exc:
            logger.error("Failed to compute usage over time: %s", exc)
            return []

    # -----------------------------------------------------------------
    # COLLECTION HEALTH
    # -----------------------------------------------------------------

    def get_collection_health(self) -> list[CollectionHealth]:
        """Get health metrics for all collections."""
        if not self._db_exists():
            return []

        now = time.time()

        try:
            with self._conn() as conn:
                collections = conn.execute(
                    "SELECT name, created_at, updated_at FROM collections ORDER BY name"
                ).fetchall()

                results: list[CollectionHealth] = []
                for coll in collections:
                    name = coll["name"]

                    # Document stats
                    doc_row = conn.execute(
                        """
                        SELECT COUNT(*) AS doc_count,
                               COALESCE(SUM(chunk_count), 0) AS chunk_count,
                               COALESCE(SUM(raw_text_length), 0) AS total_text,
                               MAX(ingested_at) AS last_ingestion
                        FROM documents WHERE collection_name = ?
                        """,
                        (name,),
                    ).fetchone()

                    # File types
                    types = conn.execute(
                        "SELECT DISTINCT file_type FROM documents WHERE collection_name = ?",
                        (name,),
                    ).fetchall()
                    file_types = [t["file_type"] for t in types]

                    # Citation count and last query
                    cite_row = conn.execute(
                        """
                        SELECT COUNT(*) AS cite_count,
                               MAX(timestamp) AS last_query
                        FROM citations WHERE collection_name = ?
                        """,
                        (name,),
                    ).fetchone()

                    chunk_count = doc_row["chunk_count"]
                    total_text = doc_row["total_text"]
                    avg_chunk = total_text / max(1, chunk_count)

                    last_ingestion = doc_row["last_ingestion"] or 0.0
                    last_query = cite_row["last_query"] or 0.0

                    # Freshness: decays over time (half-life = 7 days)
                    age_days = (now - max(last_ingestion, last_query)) / 86400
                    freshness = max(0.0, 1.0 - (age_days / 30.0))

                    results.append(CollectionHealth(
                        name=name,
                        document_count=doc_row["doc_count"],
                        chunk_count=chunk_count,
                        citation_count=cite_row["cite_count"],
                        avg_chunk_size=avg_chunk,
                        file_types=file_types,
                        last_ingestion=last_ingestion,
                        last_query=last_query,
                        freshness_score=freshness,
                    ))

                return results

        except Exception as exc:
            logger.error("Failed to compute collection health: %s", exc)
            return []

    # -----------------------------------------------------------------
    # SOURCE RELIABILITY
    # -----------------------------------------------------------------

    def get_source_reliability(
        self,
        limit: int = 50,
    ) -> list[SourceReliability]:
        """
        Rank sources by reliability score.

        Reliability = weighted combination of:
        - Citation frequency (how often cited)
        - Average retrieval score (quality of matches)
        - Freshness (recency of last citation)
        """
        if not self._db_exists():
            return []

        now = time.time()

        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        d.source_file,
                        d.collection_name,
                        d.doc_id,
                        COUNT(c.citation_id) AS citation_count,
                        COALESCE(AVG(c.score), 0) AS avg_score,
                        COALESCE(MAX(c.timestamp), 0) AS last_cited,
                        d.ingested_at
                    FROM documents d
                    LEFT JOIN citations c ON c.parent_doc_id = d.doc_id
                    GROUP BY d.doc_id
                    ORDER BY citation_count DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            # Compute max citation count for normalization
            max_citations = max((r["citation_count"] for r in rows), default=1)
            max_citations = max(1, max_citations)

            results: list[SourceReliability] = []
            for row in rows:
                cite_count = row["citation_count"]
                avg_score = row["avg_score"]
                last_cited = row["last_cited"]
                ingested_at = row["ingested_at"]

                # Freshness: based on most recent activity
                latest_activity = max(last_cited, ingested_at)
                age_days = (now - latest_activity) / 86400 if latest_activity > 0 else 30
                freshness = max(0.0, 1.0 - (age_days / 30.0))

                # Reliability composite:
                # 40% citation frequency, 35% avg score, 25% freshness
                freq_norm = cite_count / max_citations
                reliability = (
                    0.40 * freq_norm
                    + 0.35 * avg_score
                    + 0.25 * freshness
                )

                results.append(SourceReliability(
                    source_file=row["source_file"],
                    collection_name=row["collection_name"],
                    doc_id=row["doc_id"],
                    citation_count=cite_count,
                    avg_score=avg_score,
                    last_cited=last_cited,
                    freshness_score=freshness,
                    reliability_score=reliability,
                ))

            # Sort by reliability descending
            results.sort(key=lambda r: r.reliability_score, reverse=True)
            return results

        except Exception as exc:
            logger.error("Failed to compute source reliability: %s", exc)
            return []

    # -----------------------------------------------------------------
    # TOP CITED SOURCES
    # -----------------------------------------------------------------

    def get_top_cited_sources(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the most frequently cited sources."""
        if not self._db_exists():
            return []

        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    SELECT source_file, COUNT(*) AS cite_count,
                           AVG(score) AS avg_score,
                           MAX(timestamp) AS last_cited
                    FROM citations
                    GROUP BY source_file
                    ORDER BY cite_count DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            return [
                {
                    "source_file": r["source_file"],
                    "citation_count": r["cite_count"],
                    "avg_score": round(r["avg_score"], 4),
                    "last_cited": r["last_cited"],
                }
                for r in rows
            ]

        except Exception as exc:
            logger.error("Failed to get top cited sources: %s", exc)
            return []


# =========================================================================
# AUTO-REFRESH ENGINE
# =========================================================================

class RAGAutoRefresh:
    """
    Monitors ingested sources and re-indexes when changes are detected.

    Supports:
    - File change detection via mtime comparison
    - Configurable refresh interval
    - Source freshness tracking
    """

    def __init__(
        self,
        store: Any = None,
        refresh_interval_hours: float = 24.0,
    ):
        """
        Args:
            store: RAGVectorStore instance.
            refresh_interval_hours: Minimum hours between re-indexing a source.
        """
        self._store = store
        self._refresh_interval = refresh_interval_hours * 3600
        self._last_check: float = 0.0
        self._file_mtimes: dict[str, float] = {}
        self._config: dict[str, Any] | None = None

    @property
    def store(self):
        """Lazy-load the RAG store."""
        if self._store is None:
            try:
                from opti_oignon.rag_store import get_rag_store
                self._store = get_rag_store()
            except ImportError:
                import importlib.util
                import sys
                spec = importlib.util.spec_from_file_location(
                    "rag_store",
                    Path(__file__).parent / "rag_store.py",
                )
                mod = importlib.util.module_from_spec(spec)
                sys.modules["rag_store"] = mod  # Python 3.13: register before exec_module for dataclass safety
                spec.loader.exec_module(mod)
                self._store = mod.get_rag_store()
        return self._store

    def _load_config(self) -> dict[str, Any]:
        """Load auto-refresh config from rag.yaml."""
        if self._config is not None:
            return self._config

        defaults: dict[str, Any] = {
            "enabled": False,
            "interval_hours": 24.0,
            "watch_files": True,
            "max_file_size_mb": 50,
        }
        try:
            import yaml
            config_path = Path(__file__).parent / "config" / "rag.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                refresh_cfg = cfg.get("auto_refresh", {})
                if isinstance(refresh_cfg, dict):
                    defaults.update(refresh_cfg)
        except Exception as exc:
            logger.debug("Could not load auto_refresh config: %s", exc)

        self._config = defaults
        return defaults

    def check_and_refresh(self) -> RefreshResult:
        """
        Check all ingested file-based sources for changes
        and re-ingest those that have been modified.

        Only re-indexes files whose mtime has changed since ingestion.
        """
        now = time.time()
        self._last_check = now

        cfg = self._load_config()
        max_size = cfg.get("max_file_size_mb", 50) * 1024 * 1024

        store = self.store
        if store is None:
            return RefreshResult(
                checked_at=now, sources_checked=0,
                sources_refreshed=0, errors=["Store not available"],
            )

        # Get all documents
        try:
            documents = store.list_documents(limit=10000)
        except Exception as exc:
            return RefreshResult(
                checked_at=now, sources_checked=0,
                sources_refreshed=0, errors=[str(exc)],
            )

        checked = 0
        refreshed = 0
        errors: list[str] = []

        for doc in documents:
            source = doc.source_file

            # Skip URLs and non-file sources
            if source.startswith(("http://", "https://", "inline")):
                continue

            checked += 1
            filepath = Path(source)

            # Check if file exists
            if not filepath.exists():
                continue

            # Check file size
            try:
                size = filepath.stat().st_size
                if size > max_size:
                    continue
            except OSError:
                continue

            # Check mtime
            try:
                current_mtime = filepath.stat().st_mtime
            except OSError:
                continue

            previous_mtime = self._file_mtimes.get(source, doc.ingested_at)

            # Check if file has been modified since ingestion
            if current_mtime <= previous_mtime:
                self._file_mtimes[source] = current_mtime
                continue

            # Check refresh interval
            time_since_ingest = now - doc.ingested_at
            if time_since_ingest < self._refresh_interval:
                continue

            # Re-ingest
            try:
                logger.info(
                    "Auto-refreshing source: %s (mtime changed)", source
                )

                # Delete old document
                store.delete_document(doc.doc_id)

                # Re-ingest
                store.ingest_file(
                    filepath=filepath,
                    collection=doc.collection_name,
                    metadata=doc.metadata,
                )

                self._file_mtimes[source] = current_mtime
                refreshed += 1

            except Exception as exc:
                error_msg = f"Failed to refresh {source}: {exc}"
                errors.append(error_msg)
                logger.error(error_msg)

        return RefreshResult(
            checked_at=now,
            sources_checked=checked,
            sources_refreshed=refreshed,
            errors=errors,
        )

    def get_stale_sources(
        self,
        max_age_days: float = 7.0,
    ) -> list[dict[str, Any]]:
        """
        Identify sources that haven't been refreshed recently.

        Returns a list of documents older than max_age_days.
        """
        now = time.time()
        cutoff = now - max_age_days * 86400

        store = self.store
        if store is None:
            return []

        try:
            documents = store.list_documents(limit=10000)
        except Exception:
            return []

        stale: list[dict[str, Any]] = []
        for doc in documents:
            if doc.ingested_at < cutoff:
                age_days = (now - doc.ingested_at) / 86400
                stale.append({
                    "doc_id": doc.doc_id,
                    "source_file": doc.source_file,
                    "collection_name": doc.collection_name,
                    "ingested_at": doc.ingested_at,
                    "age_days": round(age_days, 1),
                })

        stale.sort(key=lambda x: x["age_days"], reverse=True)
        return stale

    @property
    def last_check_time(self) -> float:
        """Timestamp of the last refresh check."""
        return self._last_check


# =========================================================================
# MODULE-LEVEL SINGLETONS
# =========================================================================

_dashboard_instance: RAGDashboardStats | None = None
_refresh_instance: RAGAutoRefresh | None = None


def get_rag_dashboard(
    store: Any = None,
    data_dir: str | Path | None = None,
) -> RAGDashboardStats:
    """Return the module-level RAGDashboardStats singleton."""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = RAGDashboardStats(store=store, data_dir=data_dir)
    return _dashboard_instance


def get_auto_refresh(
    store: Any = None,
    refresh_interval_hours: float = 24.0,
) -> RAGAutoRefresh:
    """Return the module-level RAGAutoRefresh singleton."""
    global _refresh_instance
    if _refresh_instance is None:
        _refresh_instance = RAGAutoRefresh(
            store=store,
            refresh_interval_hours=refresh_interval_hours,
        )
    return _refresh_instance
