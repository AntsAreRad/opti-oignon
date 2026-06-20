#!/usr/bin/env python3
"""
Tests for S100 -- External Knowledge Sources.

Covers:
- BM25Scorer: tokenization, scoring, normalization
- HybridSearchEngine: fusion, dedup, config
- External connectors: abstract interface, registration, query
- ExternalVectorStoreManager: registration, query, dedup, backends
- RAGDashboardStats: overall stats, usage over time, collection health, sources
- RAGAutoRefresh: stale detection, refresh logic
- routes_rag_dashboard: endpoint schemas
"""

import importlib.util
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# =========================================================================
# MODULE LOADING (importlib isolation)
# =========================================================================

ROOT = Path(__file__).resolve().parent.parent

def _load_module(name: str, filepath: Path) -> ModuleType:
    """Load a module by file path without requiring the full package."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    # Stub opti_oignon.config if needed
    if "opti_oignon.config" not in sys.modules:
        cfg_stub = ModuleType("opti_oignon.config")
        cfg_stub.DATA_DIR = tempfile.mkdtemp()  # type: ignore[attr-defined]
        sys.modules["opti_oignon.config"] = cfg_stub
    if "opti_oignon" not in sys.modules:
        parent = ModuleType("opti_oignon")
        parent.__path__ = [str(ROOT / "opti_oignon")]
        sys.modules["opti_oignon"] = parent
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load modules
hybrid_mod = _load_module(
    "opti_oignon.rag_hybrid_search",
    ROOT / "opti_oignon" / "rag_hybrid_search.py",
)
external_mod = _load_module(
    "opti_oignon.rag_external",
    ROOT / "opti_oignon" / "rag_external.py",
)
dashboard_mod = _load_module(
    "opti_oignon.rag_dashboard",
    ROOT / "opti_oignon" / "rag_dashboard.py",
)

BM25Scorer = hybrid_mod.BM25Scorer
HybridSearchEngine = hybrid_mod.HybridSearchEngine
HybridResult = hybrid_mod.HybridResult
HybridSearchResponse = hybrid_mod.HybridSearchResponse

BaseVectorConnector = external_mod.BaseVectorConnector
QdrantConnector = external_mod.QdrantConnector
WeaviateConnector = external_mod.WeaviateConnector
PineconeConnector = external_mod.PineconeConnector
ExternalVectorStoreManager = external_mod.ExternalVectorStoreManager
ExternalSearchResult = external_mod.ExternalSearchResult
ConnectorStatus = external_mod.ConnectorStatus

RAGDashboardStats = dashboard_mod.RAGDashboardStats
RAGAutoRefresh = dashboard_mod.RAGAutoRefresh
OverallStats = dashboard_mod.OverallStats
UsageDataPoint = dashboard_mod.UsageDataPoint
CollectionHealth = dashboard_mod.CollectionHealth
SourceReliability = dashboard_mod.SourceReliability
RefreshResult = dashboard_mod.RefreshResult


# =========================================================================
# HELPERS
# =========================================================================

def _make_rag_db(db_path: str) -> None:
    """Create a minimal RAG database with test data."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS collections (
            name TEXT PRIMARY KEY,
            description TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            collection_name TEXT NOT NULL,
            source_file TEXT NOT NULL,
            file_type TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            raw_text_length INTEGER NOT NULL DEFAULT 0,
            ingested_at REAL NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (collection_name) REFERENCES collections(name) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS citations (
            citation_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            collection_name TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            parent_doc_id TEXT NOT NULL,
            source_file TEXT NOT NULL,
            section TEXT,
            score REAL NOT NULL,
            timestamp REAL NOT NULL
        );
    """)

    now = time.time()
    conn.execute(
        "INSERT INTO collections VALUES (?, ?, ?, ?)",
        ("default", "Default collection", now - 86400, now),
    )
    conn.execute(
        "INSERT INTO collections VALUES (?, ?, ?, ?)",
        ("papers", "Research papers", now - 3600, now),
    )
    # Documents
    conn.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("doc1", "default", "/tmp/test.pdf", "pdf", 10, 5000, now - 7200, "{}"),
    )
    conn.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("doc2", "papers", "/tmp/paper.pdf", "pdf", 20, 12000, now - 3600, "{}"),
    )
    conn.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("doc3", "default", "https://example.com/page", "html", 5, 2000, now - 1800, "{}"),
    )
    # Citations
    for i in range(15):
        conn.execute(
            "INSERT INTO citations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"cite{i}",
                f"query {i % 5}",
                "default" if i < 10 else "papers",
                f"chunk{i}",
                "doc1" if i < 7 else ("doc2" if i < 12 else "doc3"),
                "/tmp/test.pdf" if i < 7 else "/tmp/paper.pdf",
                None,
                0.5 + (i % 5) * 0.1,
                now - (i * 3600),
            ),
        )
    conn.commit()
    conn.close()


class MockConnector(BaseVectorConnector):
    """Test connector implementation."""

    def __init__(self, name: str = "mock", results: list | None = None):
        super().__init__(name=name)
        self._results = results or []

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def query(self, query_text, query_embedding=None, top_k=5, filters=None):
        return self._results[:top_k]

    def get_document_count(self) -> int:
        return len(self._results)


# =========================================================================
# BM25 SCORER TESTS
# =========================================================================

class TestBM25Scorer:
    """Tests for BM25Scorer."""

    def test_tokenize_basic(self):
        scorer = BM25Scorer()
        tokens = scorer._tokenize("Hello World 123")
        assert tokens == ["hello", "world", "123"]

    def test_tokenize_special_chars(self):
        scorer = BM25Scorer()
        tokens = scorer._tokenize("hello-world! foo@bar.com")
        assert "hello" in tokens
        assert "world" in tokens

    def test_score_empty_query(self):
        scorer = BM25Scorer()
        result = scorer.score_chunks("", [{"chunk_id": "a", "content": "hello"}])
        assert result == []

    def test_score_empty_chunks(self):
        scorer = BM25Scorer()
        result = scorer.score_chunks("hello", [])
        assert result == []

    def test_score_single_match(self):
        scorer = BM25Scorer()
        chunks = [
            {"chunk_id": "a", "content": "the cat sat on the mat"},
            {"chunk_id": "b", "content": "the dog ran in the park"},
        ]
        scores = scorer.score_chunks("cat mat", chunks)
        assert len(scores) == 2
        # "a" should score higher (has both terms)
        assert scores[0][0] == "a"
        assert scores[0][1] > scores[1][1]

    def test_score_no_match(self):
        scorer = BM25Scorer()
        chunks = [{"chunk_id": "a", "content": "hello world"}]
        scores = scorer.score_chunks("xyz zzz", chunks)
        assert scores[0][1] == 0.0

    def test_score_sorted_descending(self):
        scorer = BM25Scorer()
        chunks = [
            {"chunk_id": "a", "content": "alpha"},
            {"chunk_id": "b", "content": "alpha beta alpha"},
            {"chunk_id": "c", "content": "gamma"},
        ]
        scores = scorer.score_chunks("alpha", chunks)
        for i in range(len(scores) - 1):
            assert scores[i][1] >= scores[i + 1][1]

    def test_normalize_scores_basic(self):
        scorer = BM25Scorer()
        raw = [("a", 10.0), ("b", 5.0), ("c", 0.0)]
        norm = scorer.normalize_scores(raw)
        assert norm[0] == ("a", 1.0)
        assert norm[2] == ("c", 0.0)
        assert 0.0 <= norm[1][1] <= 1.0

    def test_normalize_identical_scores(self):
        scorer = BM25Scorer()
        raw = [("a", 5.0), ("b", 5.0)]
        norm = scorer.normalize_scores(raw)
        # All same -> 0.5 (non-zero)
        assert norm[0][1] == 0.5
        assert norm[1][1] == 0.5

    def test_normalize_all_zero(self):
        scorer = BM25Scorer()
        raw = [("a", 0.0), ("b", 0.0)]
        norm = scorer.normalize_scores(raw)
        assert norm[0][1] == 0.0

    def test_normalize_empty(self):
        scorer = BM25Scorer()
        assert scorer.normalize_scores([]) == []

    def test_custom_k1_b_params(self):
        scorer = BM25Scorer(k1=2.0, b=0.5)
        assert scorer.k1 == 2.0
        assert scorer.b == 0.5
        chunks = [{"chunk_id": "a", "content": "test document text"}]
        scores = scorer.score_chunks("test", chunks)
        assert len(scores) == 1
        assert scores[0][1] > 0


# =========================================================================
# HYBRID SEARCH ENGINE TESTS
# =========================================================================

class TestHybridSearchEngine:
    """Tests for HybridSearchEngine."""

    def test_init_defaults(self):
        engine = HybridSearchEngine(store=MagicMock())
        assert engine.DEFAULT_ALPHA == 0.7

    def test_chunk_key_deterministic(self):
        k1 = HybridSearchEngine._chunk_key("doc1", 0)
        k2 = HybridSearchEngine._chunk_key("doc1", 0)
        assert k1 == k2
        assert len(k1) == 16

    def test_chunk_key_unique(self):
        k1 = HybridSearchEngine._chunk_key("doc1", 0)
        k2 = HybridSearchEngine._chunk_key("doc1", 1)
        k3 = HybridSearchEngine._chunk_key("doc2", 0)
        assert k1 != k2
        assert k1 != k3

    def test_fuse_scores_pure_vector(self):
        engine = HybridSearchEngine(store=MagicMock())
        v_results = {
            "key1": {"score": 0.9, "content": "hello", "source_file": "a.pdf",
                     "file_type": "pdf", "chunk_index": 0, "total_chunks": 1,
                     "parent_doc_id": "d1", "collection_name": "default"},
        }
        fused = engine._fuse_scores(v_results, {}, alpha=1.0)
        assert len(fused) == 1
        assert fused[0].fused_score == 0.9
        assert fused[0].search_mode == "vector"

    def test_fuse_scores_pure_keyword(self):
        engine = HybridSearchEngine(store=MagicMock())
        k_results = {
            "key1": {"score": 0.8, "content": "hello", "source_file": "a.pdf",
                     "file_type": "pdf", "chunk_index": 0, "total_chunks": 1,
                     "parent_doc_id": "d1", "collection_name": "default"},
        }
        fused = engine._fuse_scores({}, k_results, alpha=0.0)
        assert len(fused) == 1
        assert fused[0].fused_score == 0.8
        assert fused[0].search_mode == "keyword"

    def test_fuse_scores_hybrid(self):
        engine = HybridSearchEngine(store=MagicMock())
        entry = {"score": 0.0, "content": "hello", "source_file": "a.pdf",
                 "file_type": "pdf", "chunk_index": 0, "total_chunks": 1,
                 "parent_doc_id": "d1", "collection_name": "default"}
        v = {"key1": {**entry, "score": 0.8}}
        k = {"key1": {**entry, "score": 0.6}}
        fused = engine._fuse_scores(v, k, alpha=0.7)
        assert len(fused) == 1
        expected = 0.7 * 0.8 + 0.3 * 0.6
        assert abs(fused[0].fused_score - expected) < 0.001
        assert fused[0].search_mode == "hybrid"

    def test_fuse_scores_dedup(self):
        engine = HybridSearchEngine(store=MagicMock())
        entry = {"score": 0.5, "content": "hello", "source_file": "a.pdf",
                 "file_type": "pdf", "chunk_index": 0, "total_chunks": 1,
                 "parent_doc_id": "d1", "collection_name": "default"}
        # Same key in both -> should produce only 1 result
        v = {"key1": {**entry, "score": 0.9}}
        k = {"key1": {**entry, "score": 0.4}}
        fused = engine._fuse_scores(v, k, alpha=0.5)
        assert len(fused) == 1

    def test_fuse_scores_sorted(self):
        engine = HybridSearchEngine(store=MagicMock())
        entry_base = {"content": "x", "source_file": "a.pdf",
                      "file_type": "pdf", "chunk_index": 0, "total_chunks": 1,
                      "parent_doc_id": "d1", "collection_name": "default"}
        v = {
            "k1": {**entry_base, "score": 0.9},
            "k2": {**entry_base, "score": 0.3},
        }
        k = {
            "k3": {**entry_base, "score": 0.7},
        }
        fused = engine._fuse_scores(v, k, alpha=0.5)
        for i in range(len(fused) - 1):
            assert fused[i].fused_score >= fused[i + 1].fused_score

    def test_get_config(self):
        engine = HybridSearchEngine(store=MagicMock(), alpha=0.6)
        cfg = engine.get_config()
        assert cfg["alpha"] == 0.6
        assert "bm25_k1" in cfg

    def test_hybrid_result_to_dict(self):
        r = HybridResult(
            content="test", vector_score=0.8, keyword_score=0.5,
            fused_score=0.7, source_file="a.pdf", file_type="pdf",
            chunk_index=0, total_chunks=5, parent_doc_id="d1",
            collection_name="default", chunk_id="abc123",
        )
        d = r.to_dict()
        assert d["fused_score"] == 0.7
        assert d["search_mode"] == "hybrid"

    def test_hybrid_response_to_dict(self):
        resp = HybridSearchResponse(
            query="test", results=[], total_results=0,
            alpha=0.7, search_time_ms=12.5,
            vector_count=3, keyword_count=5,
        )
        d = resp.to_dict()
        assert d["alpha"] == 0.7
        assert d["search_time_ms"] == 12.5


# =========================================================================
# EXTERNAL CONNECTOR TESTS
# =========================================================================

class TestExternalConnectors:
    """Tests for external vector store connectors."""

    def test_mock_connector_connect(self):
        conn = MockConnector(name="test")
        assert not conn.connected
        assert conn.connect()
        assert conn.connected

    def test_mock_connector_disconnect(self):
        conn = MockConnector()
        conn.connect()
        conn.disconnect()
        assert not conn.connected

    def test_mock_connector_query(self):
        results = [
            ExternalSearchResult(
                content="test content", score=0.9, source="test",
                connector_name="mock", chunk_id="c1",
            ),
        ]
        conn = MockConnector(results=results)
        conn.connect()
        out = conn.query("test")
        assert len(out) == 1
        assert out[0].score == 0.9

    def test_connector_health(self):
        conn = MockConnector(name="test-conn")
        conn.connect()
        status = conn.health()
        assert status.name == "test-conn"
        assert status.connected
        assert isinstance(status, ConnectorStatus)

    def test_connector_type_property(self):
        conn = MockConnector()
        assert conn.connector_type == "MockConnector"

    def test_qdrant_unavailable(self):
        """QdrantConnector should handle missing library gracefully."""
        conn = QdrantConnector(name="test-qdrant")
        if not external_mod.QDRANT_AVAILABLE:
            assert not conn.connect()
            assert conn._last_error is not None

    def test_weaviate_unavailable(self):
        """WeaviateConnector should handle missing library gracefully."""
        conn = WeaviateConnector(name="test-weaviate")
        if not external_mod.WEAVIATE_AVAILABLE:
            assert not conn.connect()
            assert conn._last_error is not None

    def test_pinecone_no_key(self):
        """PineconeConnector should fail without API key."""
        conn = PineconeConnector(api_key="", name="test-pinecone")
        assert not conn.connect()
        assert conn._last_error is not None

    def test_external_result_to_dict(self):
        r = ExternalSearchResult(
            content="hello", score=0.85, source="qdrant",
            connector_name="my-qdrant", chunk_id="xyz",
            metadata={"key": "val"},
        )
        d = r.to_dict()
        assert d["score"] == 0.85
        assert d["connector_name"] == "my-qdrant"

    def test_connector_status_to_dict(self):
        s = ConnectorStatus(
            name="test", connector_type="Qdrant",
            connected=True, document_count=100,
            last_query_time_ms=15.3, error=None,
        )
        d = s.to_dict()
        assert d["connected"]
        assert d["document_count"] == 100


# =========================================================================
# EXTERNAL VECTOR STORE MANAGER TESTS
# =========================================================================

class TestExternalVectorStoreManager:
    """Tests for ExternalVectorStoreManager."""

    def test_register_connector(self):
        mgr = ExternalVectorStoreManager()
        conn = MockConnector(name="test")
        mgr.register_connector("test", conn)
        assert mgr.get_connector("test") is conn

    def test_unregister_connector(self):
        mgr = ExternalVectorStoreManager()
        conn = MockConnector()
        conn.connect()
        mgr.register_connector("test", conn)
        assert mgr.unregister_connector("test")
        assert mgr.get_connector("test") is None
        assert not conn.connected

    def test_unregister_nonexistent(self):
        mgr = ExternalVectorStoreManager()
        assert not mgr.unregister_connector("nope")

    def test_list_connectors(self):
        mgr = ExternalVectorStoreManager()
        mgr.register_connector("a", MockConnector())
        mgr.register_connector("b", MockConnector())
        statuses = mgr.list_connectors()
        assert len(statuses) == 2

    def test_connect_all(self):
        mgr = ExternalVectorStoreManager()
        mgr.register_connector("a", MockConnector())
        mgr.register_connector("b", MockConnector())
        results = mgr.connect_all()
        assert results["a"]
        assert results["b"]

    def test_disconnect_all(self):
        mgr = ExternalVectorStoreManager()
        c1 = MockConnector()
        c2 = MockConnector()
        mgr.register_connector("a", c1)
        mgr.register_connector("b", c2)
        mgr.connect_all()
        mgr.disconnect_all()
        assert not c1.connected
        assert not c2.connected

    def test_query_across_connectors(self):
        mgr = ExternalVectorStoreManager()
        r1 = [ExternalSearchResult("alpha", 0.9, "src1", "c1", "id1")]
        r2 = [ExternalSearchResult("beta", 0.8, "src2", "c2", "id2")]
        c1 = MockConnector(name="c1", results=r1)
        c2 = MockConnector(name="c2", results=r2)
        c1.connect()
        c2.connect()
        mgr.register_connector("c1", c1)
        mgr.register_connector("c2", c2)

        results = mgr.query("test", top_k=10)
        assert len(results) == 2
        # Sorted by score desc
        assert results[0].score >= results[1].score

    def test_query_subset(self):
        mgr = ExternalVectorStoreManager()
        c1 = MockConnector(name="c1", results=[
            ExternalSearchResult("a", 0.9, "s", "c1", "1"),
        ])
        c2 = MockConnector(name="c2", results=[
            ExternalSearchResult("b", 0.8, "s", "c2", "2"),
        ])
        c1.connect()
        c2.connect()
        mgr.register_connector("c1", c1)
        mgr.register_connector("c2", c2)

        results = mgr.query("test", connector_names=["c1"])
        assert len(results) == 1
        assert results[0].connector_name == "c1"

    def test_dedup_identical_content(self):
        mgr = ExternalVectorStoreManager()
        same_content = "this is duplicate content"
        r1 = [ExternalSearchResult(same_content, 0.9, "s1", "c1", "id1")]
        r2 = [ExternalSearchResult(same_content, 0.8, "s2", "c2", "id2")]
        c1 = MockConnector(name="c1", results=r1)
        c2 = MockConnector(name="c2", results=r2)
        c1.connect()
        c2.connect()
        mgr.register_connector("c1", c1)
        mgr.register_connector("c2", c2)

        results = mgr.query("test", top_k=10)
        assert len(results) == 1  # Deduped

    def test_get_available_backends(self):
        mgr = ExternalVectorStoreManager()
        backends = mgr.get_available_backends()
        assert "qdrant" in backends
        assert "weaviate" in backends
        assert "pinecone" in backends
        assert all(isinstance(v, bool) for v in backends.values())


# =========================================================================
# RAG DASHBOARD STATS TESTS
# =========================================================================

class TestRAGDashboardStats:
    """Tests for RAGDashboardStats."""

    @pytest.fixture
    def dashboard_with_data(self, tmp_path):
        db_path = str(tmp_path / "rag_documents.db")
        _make_rag_db(db_path)
        return RAGDashboardStats(data_dir=str(tmp_path))

    @pytest.fixture
    def empty_dashboard(self, tmp_path):
        return RAGDashboardStats(data_dir=str(tmp_path))

    def test_overall_stats_with_data(self, dashboard_with_data):
        stats = dashboard_with_data.get_overall_stats()
        assert isinstance(stats, OverallStats)
        assert stats.total_collections == 2
        assert stats.total_documents == 3
        assert stats.total_citations == 15
        assert stats.avg_score > 0

    def test_overall_stats_empty(self, empty_dashboard):
        stats = empty_dashboard.get_overall_stats()
        assert stats.total_collections == 0
        assert stats.total_documents == 0

    def test_overall_stats_to_dict(self, dashboard_with_data):
        stats = dashboard_with_data.get_overall_stats()
        d = stats.to_dict()
        assert "total_collections" in d
        assert "storage_bytes" in d

    def test_usage_over_time(self, dashboard_with_data):
        data = dashboard_with_data.get_usage_over_time(days=7)
        assert isinstance(data, list)
        assert len(data) >= 1
        # Should have at least some data points
        for dp in data:
            assert isinstance(dp, UsageDataPoint)
            assert dp.date  # Non-empty
            assert dp.query_count >= 0

    def test_usage_over_time_empty(self, empty_dashboard):
        data = empty_dashboard.get_usage_over_time(days=7)
        assert data == []

    def test_usage_datapoint_to_dict(self):
        dp = UsageDataPoint(date="2025-03-01", query_count=5, citation_count=12, avg_score=0.75)
        d = dp.to_dict()
        assert d["date"] == "2025-03-01"
        assert d["query_count"] == 5

    def test_collection_health(self, dashboard_with_data):
        health = dashboard_with_data.get_collection_health()
        assert len(health) == 2
        names = {h.name for h in health}
        assert "default" in names
        assert "papers" in names

    def test_collection_health_freshness(self, dashboard_with_data):
        health = dashboard_with_data.get_collection_health()
        for h in health:
            assert 0.0 <= h.freshness_score <= 1.0

    def test_collection_health_to_dict(self):
        h = CollectionHealth(
            name="test", document_count=5, chunk_count=50,
            citation_count=10, avg_chunk_size=200.0,
            file_types=["pdf", "txt"], last_ingestion=time.time(),
            last_query=time.time(), freshness_score=0.95,
        )
        d = h.to_dict()
        assert d["name"] == "test"
        assert d["freshness_score"] == 0.95

    def test_source_reliability(self, dashboard_with_data):
        sources = dashboard_with_data.get_source_reliability(limit=10)
        assert len(sources) >= 1
        for s in sources:
            assert isinstance(s, SourceReliability)
            assert 0.0 <= s.reliability_score <= 1.0

    def test_source_reliability_sorted(self, dashboard_with_data):
        sources = dashboard_with_data.get_source_reliability()
        for i in range(len(sources) - 1):
            assert sources[i].reliability_score >= sources[i + 1].reliability_score

    def test_source_reliability_to_dict(self):
        s = SourceReliability(
            source_file="test.pdf", collection_name="default",
            doc_id="d1", citation_count=10, avg_score=0.8,
            last_cited=time.time(), freshness_score=0.9,
            reliability_score=0.85,
        )
        d = s.to_dict()
        assert d["reliability_score"] == 0.85

    def test_top_cited_sources(self, dashboard_with_data):
        top = dashboard_with_data.get_top_cited_sources(limit=5)
        assert len(top) >= 1
        assert "source_file" in top[0]
        assert "citation_count" in top[0]


# =========================================================================
# RAG AUTO-REFRESH TESTS
# =========================================================================

class TestRAGAutoRefresh:
    """Tests for RAGAutoRefresh."""

    def test_init(self):
        refresher = RAGAutoRefresh(store=MagicMock(), refresh_interval_hours=12)
        assert refresher._refresh_interval == 12 * 3600

    def test_get_stale_sources_no_store(self):
        refresher = RAGAutoRefresh(store=None)
        refresher._store = None  # Force no store
        # Override store property
        stale = refresher.get_stale_sources()
        # With None store, returns empty
        assert isinstance(stale, list)

    def test_check_and_refresh_no_store(self):
        refresher = RAGAutoRefresh(store=None)
        refresher._store = "none"  # Break store
        # Should handle gracefully
        result = refresher.check_and_refresh()
        assert isinstance(result, RefreshResult)

    def test_refresh_result_to_dict(self):
        r = RefreshResult(
            checked_at=time.time(),
            sources_checked=5,
            sources_refreshed=2,
            errors=["some error"],
        )
        d = r.to_dict()
        assert d["sources_checked"] == 5
        assert len(d["errors"]) == 1

    def test_last_check_time(self):
        refresher = RAGAutoRefresh(store=MagicMock())
        assert refresher.last_check_time == 0.0


# =========================================================================
# ROUTES TESTS (schema validation)
# =========================================================================

class TestRouteSchemas:
    """Test that route Pydantic schemas are importable and valid."""

    def test_import_routes(self):
        routes_mod = _load_module(
            "opti_oignon.api.routes_rag_dashboard",
            ROOT / "opti_oignon" / "api" / "routes_rag_dashboard.py",
        )
        assert hasattr(routes_mod, "router")
        assert hasattr(routes_mod, "OverallStatsResponse")
        assert hasattr(routes_mod, "UsageResponse")
        assert hasattr(routes_mod, "SourcesResponse")
        assert hasattr(routes_mod, "HealthResponse")
        assert hasattr(routes_mod, "RefreshResponse")
        assert hasattr(routes_mod, "ConnectorsResponse")
        assert hasattr(routes_mod, "BackendsResponse")

    def test_overall_stats_schema(self):
        routes_mod = sys.modules.get("opti_oignon.api.routes_rag_dashboard")
        if routes_mod is None:
            pytest.skip("routes module not loaded")
        schema = routes_mod.OverallStatsResponse
        instance = schema(
            total_collections=2, total_documents=5, total_chunks=50,
            total_citations=20, total_queries_today=3, total_queries_week=15,
            total_queries_all=100, avg_score=0.75, storage_bytes=1024000,
        )
        assert instance.total_collections == 2

    def test_connector_status_schema(self):
        routes_mod = sys.modules.get("opti_oignon.api.routes_rag_dashboard")
        if routes_mod is None:
            pytest.skip("routes module not loaded")
        schema = routes_mod.ConnectorStatusResponse
        instance = schema(
            name="qdrant", connector_type="QdrantConnector",
            connected=True, document_count=500,
            last_query_time_ms=12.3, error=None,
        )
        assert instance.connected


# =========================================================================
# MODULE-LEVEL SINGLETON TESTS
# =========================================================================

class TestSingletons:
    """Test module-level singleton functions."""

    def test_hybrid_engine_singleton(self):
        # Reset
        hybrid_mod._hybrid_engine = None
        engine = hybrid_mod.get_hybrid_engine(store=MagicMock())
        assert engine is not None
        engine2 = hybrid_mod.get_hybrid_engine()
        assert engine2 is engine
        hybrid_mod._hybrid_engine = None  # Cleanup

    def test_external_manager_singleton(self):
        external_mod._external_manager = None
        mgr = external_mod.get_external_manager()
        assert mgr is not None
        mgr2 = external_mod.get_external_manager()
        assert mgr2 is mgr
        external_mod._external_manager = None

    def test_dashboard_singleton(self):
        dashboard_mod._dashboard_instance = None
        dash = dashboard_mod.get_rag_dashboard(data_dir="/tmp")
        assert dash is not None
        dash2 = dashboard_mod.get_rag_dashboard()
        assert dash2 is dash
        dashboard_mod._dashboard_instance = None

    def test_auto_refresh_singleton(self):
        dashboard_mod._refresh_instance = None
        ref = dashboard_mod.get_auto_refresh(store=MagicMock())
        assert ref is not None
        ref2 = dashboard_mod.get_auto_refresh()
        assert ref2 is ref
        dashboard_mod._refresh_instance = None


# =========================================================================
# FEATURE FLAG TESTS
# =========================================================================

class TestFeatureFlags:
    """Test feature availability flags."""

    def test_hybrid_search_available(self):
        assert hybrid_mod.HYBRID_SEARCH_AVAILABLE is True

    def test_external_stores_available(self):
        assert external_mod.EXTERNAL_STORES_AVAILABLE is True

    def test_rag_dashboard_available(self):
        assert dashboard_mod.RAG_DASHBOARD_AVAILABLE is True
