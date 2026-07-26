#!/usr/bin/env python3
"""
RAG VECTOR STORE -- Collection management, ingestion & retrieval.

Provides:
- ChromaDB collection CRUD (create, list, delete, stats)
- Document ingestion pipeline: file/text -> RAGChunker -> embeddings -> ChromaDB
- Retrieval: query -> embedding -> top-k nearest -> optional reranking
- Citation tracking: SQLite-backed log of which chunk answered which query
- Metadata filtering (by source, date, collection)
- Configurable embedding model (Ollama embeddings)

Each collection maps to a ChromaDB collection.  Documents are tracked in
a SQLite ``rag_documents.db`` so they can be listed and deleted cleanly.
"""

import hashlib
import html as html_module
import json
import logging
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
# Audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# -- Feature flags for optional deps ---------------------------------------

CHROMADB_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore[assignment]
    ChromaSettings = None  # type: ignore[assignment,misc]

RAG_STORE_AVAILABLE = CHROMADB_AVAILABLE

REQUESTS_AVAILABLE = False
try:
    import requests as _requests_lib
    REQUESTS_AVAILABLE = True
except ImportError:
    _requests_lib = None  # type: ignore[assignment]


# =========================================================================
# DATA STRUCTURES
# =========================================================================

@dataclass
class CollectionInfo:
    """Metadata about a ChromaDB collection."""
    name: str
    description: str
    document_count: int
    chunk_count: int
    created_at: float
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class IngestedDocument:
    """Record of a document stored in the vector store."""
    doc_id: str
    collection_name: str
    source_file: str
    file_type: str
    chunk_count: int
    raw_text_length: int
    ingested_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "collection_name": self.collection_name,
            "source_file": self.source_file,
            "file_type": self.file_type,
            "chunk_count": self.chunk_count,
            "raw_text_length": self.raw_text_length,
            "ingested_at": self.ingested_at,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """A single retrieval hit with score and provenance."""
    content: str
    score: float
    source_file: str
    file_type: str
    chunk_index: int
    total_chunks: int
    parent_doc_id: str
    collection_name: str
    section: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "score": round(self.score, 4),
            "source_file": self.source_file,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "parent_doc_id": self.parent_doc_id,
            "collection_name": self.collection_name,
            "section": self.section,
            "page": self.page,
        }


@dataclass
class CitationRecord:
    """Tracks which chunk answered which query."""
    citation_id: str
    query: str
    collection_name: str
    chunk_id: str
    parent_doc_id: str
    source_file: str
    section: str | None
    score: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "query": self.query,
            "collection_name": self.collection_name,
            "chunk_id": self.chunk_id,
            "parent_doc_id": self.parent_doc_id,
            "source_file": self.source_file,
            "section": self.section,
            "score": round(self.score, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class QueryResponse:
    """Full response to a RAG query including results and citations."""
    query: str
    results: list[RetrievalResult]
    citations: list[CitationRecord]
    total_results: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "citations": [c.to_dict() for c in self.citations],
            "total_results": self.total_results,
        }


# =========================================================================
# SQLITE BACKING STORE (documents + citations)
# =========================================================================

class _RAGDatabase:
    """SQLite database for document tracking and citation logging."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS collections (
                    name        TEXT PRIMARY KEY,
                    description TEXT NOT NULL DEFAULT '',
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    doc_id          TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    source_file     TEXT NOT NULL,
                    file_type       TEXT NOT NULL,
                    chunk_count     INTEGER NOT NULL DEFAULT 0,
                    raw_text_length INTEGER NOT NULL DEFAULT 0,
                    ingested_at     REAL NOT NULL,
                    metadata_json   TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (collection_name)
                        REFERENCES collections(name) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_docs_collection
                    ON documents(collection_name);

                CREATE TABLE IF NOT EXISTS citations (
                    citation_id     TEXT PRIMARY KEY,
                    query           TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    chunk_id        TEXT NOT NULL,
                    parent_doc_id   TEXT NOT NULL,
                    source_file     TEXT NOT NULL,
                    section         TEXT,
                    score           REAL NOT NULL,
                    timestamp       REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_citations_collection
                    ON citations(collection_name);
                CREATE INDEX IF NOT EXISTS idx_citations_doc
                    ON citations(parent_doc_id);
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = _safe_connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    # -- Collections --

    def create_collection(self, name: str, description: str = "") -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO collections (name, description, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (name, description, now, now),
            )

    def delete_collection(self, name: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM citations WHERE collection_name = ?", (name,))
            conn.execute("DELETE FROM documents WHERE collection_name = ?", (name,))
            conn.execute("DELETE FROM collections WHERE name = ?", (name,))

    def list_collections(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.name, c.description, c.created_at, c.updated_at,
                       COALESCE(d.doc_count, 0) AS document_count,
                       COALESCE(d.total_chunks, 0) AS chunk_count
                FROM collections c
                LEFT JOIN (
                    SELECT collection_name,
                           COUNT(*) AS doc_count,
                           SUM(chunk_count) AS total_chunks
                    FROM documents GROUP BY collection_name
                ) d ON d.collection_name = c.name
                ORDER BY c.created_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def touch_collection(self, name: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE collections SET updated_at = ? WHERE name = ?",
                (time.time(), name),
            )

    # -- Documents --

    def insert_document(self, doc: IngestedDocument) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO documents
                   (doc_id, collection_name, source_file, file_type,
                    chunk_count, raw_text_length, ingested_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc.doc_id,
                    doc.collection_name,
                    doc.source_file,
                    doc.file_type,
                    doc.chunk_count,
                    doc.raw_text_length,
                    doc.ingested_at,
                    json.dumps(doc.metadata),
                ),
            )

    def delete_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if not row:
                return None
            info = dict(row)
            conn.execute("DELETE FROM citations WHERE parent_doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return info

    def list_documents(
        self,
        collection_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[IngestedDocument]:
        with self._conn() as conn:
            if collection_name:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE collection_name = ? ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
                    (collection_name, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
        result: list[IngestedDocument] = []
        for r in rows:
            result.append(IngestedDocument(
                doc_id=r["doc_id"],
                collection_name=r["collection_name"],
                source_file=r["source_file"],
                file_type=r["file_type"],
                chunk_count=r["chunk_count"],
                raw_text_length=r["raw_text_length"],
                ingested_at=r["ingested_at"],
                metadata=json.loads(r["metadata_json"]) if r["metadata_json"] else {},
            ))
        return result

    def get_document(self, doc_id: str) -> IngestedDocument | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        if not row:
            return None
        return IngestedDocument(
            doc_id=row["doc_id"],
            collection_name=row["collection_name"],
            source_file=row["source_file"],
            file_type=row["file_type"],
            chunk_count=row["chunk_count"],
            raw_text_length=row["raw_text_length"],
            ingested_at=row["ingested_at"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )

    def get_doc_ids_for_collection(self, collection_name: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT doc_id FROM documents WHERE collection_name = ?",
                (collection_name,),
            ).fetchall()
        return [r["doc_id"] for r in rows]

    # -- Citations --

    def insert_citation(self, citation: CitationRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO citations
                   (citation_id, query, collection_name, chunk_id,
                    parent_doc_id, source_file, section, score, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    citation.citation_id,
                    citation.query,
                    citation.collection_name,
                    citation.chunk_id,
                    citation.parent_doc_id,
                    citation.source_file,
                    citation.section,
                    citation.score,
                    citation.timestamp,
                ),
            )

    def list_citations(
        self,
        collection_name: str | None = None,
        doc_id: str | None = None,
        limit: int = 50,
    ) -> list[CitationRecord]:
        with self._conn() as conn:
            if doc_id:
                rows = conn.execute(
                    "SELECT * FROM citations WHERE parent_doc_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (doc_id, limit),
                ).fetchall()
            elif collection_name:
                rows = conn.execute(
                    "SELECT * FROM citations WHERE collection_name = ? ORDER BY timestamp DESC LIMIT ?",
                    (collection_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM citations ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            CitationRecord(
                citation_id=r["citation_id"],
                query=r["query"],
                collection_name=r["collection_name"],
                chunk_id=r["chunk_id"],
                parent_doc_id=r["parent_doc_id"],
                source_file=r["source_file"],
                section=r["section"],
                score=r["score"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]


# =========================================================================
# VECTOR STORE
# =========================================================================

class RAGVectorStore:
    """
    ChromaDB-backed vector store with collection management,
    document ingestion, retrieval, and citation tracking.

    Usage::

        store = RAGVectorStore(data_dir="/path/to/data/rag")
        store.create_collection("papers", description="Research papers")
        doc = store.ingest_file("/path/to/paper.pdf", collection="papers")
        response = store.query("What is Shannon diversity?", collection="papers")
        for r in response.results:
            print(r.score, r.source_file, r.section)
    """

    DEFAULT_COLLECTION = "default"

    def __init__(
        self,
        data_dir: str | Path | None = None,
        embedding_model: str = "mxbai-embed-large",
        ollama_url: str = "http://localhost:11434",
    ):
        # Resolve data directory
        if data_dir is None:
            try:
                from opti_oignon.config import DATA_DIR
                data_dir = Path(DATA_DIR) / "rag"
            except ImportError:
                data_dir = Path.home() / ".opti-oignon" / "data" / "rag"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_dir = self.data_dir / "chroma_v2"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        # SQLite tracking DB
        self.db = _RAGDatabase(self.data_dir / "rag_documents.db")

        # ChromaDB client.
        # ChromaDB has no native at-rest encryption, so
        # the ingested chunk text and embedding vectors persisted under
        # chroma_dir are PLAINTEXT on disk. The RAG metadata DB above is
        # encrypted via safe_connect, but this vector store is not. Full-disk
        # encryption (LUKS) is a deployment requirement for a sensitive RAG
        # corpus (see SECURITY.md). Application-layer encryption of the chunk
        # text before upsert (decrypt on retrieval) is a planned cycle
        # (ROADMAP_POST_S183, RAG-at-rest cycle).
        if CHROMADB_AVAILABLE:
            self._chroma = chromadb.PersistentClient(
                path=str(self.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            self._chroma = None
            logger.warning("chromadb not installed -- vector store disabled")

        # Embedding config (lazy: embedder created on first use)
        self._embedding_model = embedding_model
        self._ollama_url = ollama_url
        self._embedder = None  # Lazy init

        # Chunker (lazy)
        self._chunker = None

    # -----------------------------------------------------------------
    # LAZY HELPERS
    # -----------------------------------------------------------------

    def _get_embedder(self):
        """Lazy-init the Ollama embedder."""
        if self._embedder is None:
            try:
                from opti_oignon.rag.config import EmbeddingConfig
                from opti_oignon.rag.embeddings import OllamaEmbeddings
                cfg = EmbeddingConfig(
                    model=self._embedding_model,
                    ollama_url=self._ollama_url,
                )
                self._embedder = OllamaEmbeddings(cfg)
            except ImportError:
                # Fallback: try direct import from file
                try:
                    import importlib.util
                    import sys
                    rag_dir = Path(__file__).parent / "rag"
                    spec_cfg = importlib.util.spec_from_file_location(
                        "rag_config", rag_dir / "config.py"
                    )
                    cfg_mod = importlib.util.module_from_spec(spec_cfg)
                    sys.modules["rag_config"] = cfg_mod  # Python 3.13: register before exec_module for dataclass safety
                    spec_cfg.loader.exec_module(cfg_mod)

                    spec_emb = importlib.util.spec_from_file_location(
                        "rag_embeddings", rag_dir / "embeddings.py"
                    )
                    emb_mod = importlib.util.module_from_spec(spec_emb)
                    emb_mod.config = cfg_mod  # wire dependency
                    sys.modules["rag_embeddings"] = emb_mod  # Python 3.13: register before exec_module for dataclass safety
                    spec_emb.loader.exec_module(emb_mod)

                    ecfg = cfg_mod.EmbeddingConfig(
                        model=self._embedding_model,
                        ollama_url=self._ollama_url,
                    )
                    self._embedder = emb_mod.OllamaEmbeddings(ecfg)
                except Exception as exc:
                    logger.error("Cannot initialize embedder: %s", exc)
                    self._embedder = None
        return self._embedder

    def _get_chunker(self):
        """Lazy-init the RAGChunker."""
        if self._chunker is None:
            try:
                from opti_oignon.rag_chunker import RAGChunker
            except ImportError:
                import importlib.util
                import sys
                spec = importlib.util.spec_from_file_location(
                    "rag_chunker",
                    Path(__file__).parent / "rag_chunker.py",
                )
                mod = importlib.util.module_from_spec(spec)
                sys.modules["rag_chunker"] = mod  # Python 3.13: register before exec_module for dataclass safety
                spec.loader.exec_module(mod)
                RAGChunker = mod.RAGChunker
            self._chunker = RAGChunker()
        return self._chunker

    def _get_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        if not self._chroma:
            raise RuntimeError("ChromaDB not available")
        return self._chroma.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    # -----------------------------------------------------------------
    # COLLECTION MANAGEMENT
    # -----------------------------------------------------------------

    def create_collection(
        self, name: str, description: str = ""
    ) -> CollectionInfo:
        """Create a new collection."""
        if not self._chroma:
            raise RuntimeError("ChromaDB not available")
        self._chroma.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        self.db.create_collection(name, description)
        now = time.time()
        return CollectionInfo(
            name=name,
            description=description,
            document_count=0,
            chunk_count=0,
            created_at=now,
            updated_at=now,
        )

    def list_collections(self) -> list[CollectionInfo]:
        """List all collections with stats."""
        rows = self.db.list_collections()
        results: list[CollectionInfo] = []
        for r in rows:
            results.append(CollectionInfo(
                name=r["name"],
                description=r.get("description", ""),
                document_count=r.get("document_count", 0),
                chunk_count=r.get("chunk_count", 0),
                created_at=r.get("created_at", 0),
                updated_at=r.get("updated_at", 0),
            ))
        return results

    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its documents/chunks."""
        try:
            if self._chroma:
                try:
                    self._chroma.delete_collection(name)
                except Exception:
                    pass  # Collection may not exist in ChromaDB
            self.db.delete_collection(name)
            return True
        except Exception as exc:
            logger.error("Failed to delete collection %s: %s", name, exc)
            return False

    def get_collection_stats(self, name: str) -> dict[str, Any]:
        """Get detailed stats for a collection."""
        docs = self.db.list_documents(collection_name=name, limit=10000)
        chunk_count = 0
        if self._chroma:
            try:
                coll = self._chroma.get_collection(name)
                chunk_count = coll.count()
            except Exception:
                chunk_count = sum(d.chunk_count for d in docs)
        else:
            chunk_count = sum(d.chunk_count for d in docs)

        return {
            "name": name,
            "document_count": len(docs),
            "chunk_count": chunk_count,
            "file_types": list({d.file_type for d in docs}),
            "total_text_length": sum(d.raw_text_length for d in docs),
        }

    # -----------------------------------------------------------------
    # DOCUMENT INGESTION
    # -----------------------------------------------------------------

    def ingest_file(
        self,
        filepath: str | Path,
        collection: str | None = None,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestedDocument:
        """
        Ingest a file: extract text -> chunk -> embed -> store in ChromaDB.

        Args:
            filepath: Path to the file.
            collection: Target collection name (default: 'default').
            doc_id: Optional document ID.
            metadata: Extra metadata to attach.

        Returns:
            IngestedDocument record.
        """
        collection = collection or self.DEFAULT_COLLECTION
        doc_id = doc_id or uuid.uuid4().hex[:12]

        # Ensure collection exists
        self.db.create_collection(collection)

        chunker = self._get_chunker()
        result = chunker.chunk_file(filepath, doc_id=doc_id)

        if not result.chunks:
            logger.warning("No chunks produced for %s", filepath)
            doc = IngestedDocument(
                doc_id=doc_id,
                collection_name=collection,
                source_file=str(filepath),
                file_type=result.file_type,
                chunk_count=0,
                raw_text_length=result.raw_text_length,
                ingested_at=time.time(),
                metadata=metadata or {},
            )
            self.db.insert_document(doc)
            return doc

        # Store chunks in ChromaDB
        self._store_chunks(result.chunks, collection)
        self.db.touch_collection(collection)

        doc = IngestedDocument(
            doc_id=doc_id,
            collection_name=collection,
            source_file=str(filepath),
            file_type=result.file_type,
            chunk_count=result.chunk_count,
            raw_text_length=result.raw_text_length,
            ingested_at=time.time(),
            metadata=metadata or {},
        )
        self.db.insert_document(doc)
        logger.info(
            "Ingested %s: %d chunks into collection '%s'",
            filepath, result.chunk_count, collection,
        )
        return doc

    def ingest_text(
        self,
        text: str,
        source: str = "inline",
        file_type: str = "text",
        collection: str | None = None,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestedDocument:
        """
        Ingest raw text content.

        Args:
            text: Text to ingest.
            source: Label for the source.
            file_type: File type hint.
            collection: Target collection.
            doc_id: Optional document ID.
            metadata: Extra metadata.

        Returns:
            IngestedDocument record.
        """
        collection = collection or self.DEFAULT_COLLECTION
        doc_id = doc_id or uuid.uuid4().hex[:12]

        self.db.create_collection(collection)

        chunker = self._get_chunker()
        result = chunker.chunk_text(text, source=source, file_type=file_type, doc_id=doc_id)

        if result.chunks:
            self._store_chunks(result.chunks, collection)
            self.db.touch_collection(collection)

        doc = IngestedDocument(
            doc_id=doc_id,
            collection_name=collection,
            source_file=source,
            file_type=file_type,
            chunk_count=result.chunk_count,
            raw_text_length=result.raw_text_length,
            ingested_at=time.time(),
            metadata=metadata or {},
        )
        self.db.insert_document(doc)
        return doc

    def _store_chunks(self, chunks: list, collection_name: str) -> None:
        """Embed and store chunks in ChromaDB."""
        if not self._chroma:
            raise RuntimeError("ChromaDB not available")

        coll = self._get_collection(collection_name)
        embedder = self._get_embedder()

        # Prepare data
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            metadatas.append(chunk.metadata)

        # Generate embeddings
        if embedder:
            embeddings = embedder.embed(
                documents, show_progress=False
            )
            # Filter out None embeddings
            valid = [
                (i, doc, meta, emb)
                for i, doc, meta, emb in zip(ids, documents, metadatas, embeddings)
                if emb is not None
            ]
            if valid:
                v_ids, v_docs, v_metas, v_embs = zip(*valid)
                coll.upsert(
                    ids=list(v_ids),
                    documents=list(v_docs),
                    metadatas=list(v_metas),
                    embeddings=list(v_embs),
                )
            else:
                logger.warning("All embeddings failed -- storing without vectors")
                coll.upsert(ids=ids, documents=documents, metadatas=metadatas)
        else:
            # No embedder: store documents only (ChromaDB default embedding)
            logger.warning("No embedder available -- using ChromaDB default embedding")
            coll.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # -----------------------------------------------------------------
    # DOCUMENT MANAGEMENT
    # -----------------------------------------------------------------

    def list_documents(
        self,
        collection: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[IngestedDocument]:
        """List ingested documents."""
        return self.db.list_documents(collection_name=collection, limit=limit, offset=offset)

    def get_document(self, doc_id: str) -> IngestedDocument | None:
        """Get a single document by ID."""
        return self.db.get_document(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks from ChromaDB + SQLite."""
        info = self.db.delete_document(doc_id)
        if not info:
            return False

        # Remove chunks from ChromaDB
        if self._chroma:
            try:
                coll_name = info["collection_name"]
                coll = self._chroma.get_collection(coll_name)
                # Find chunk IDs by parent_doc_id
                results = coll.get(
                    where={"parent_doc_id": {"$eq": doc_id}},
                    include=[],
                )
                if results and results.get("ids"):
                    coll.delete(ids=results["ids"])
            except Exception as exc:
                logger.error("Error removing chunks from ChromaDB: %s", exc)

        return True

    # -----------------------------------------------------------------
    # RETRIEVAL
    # -----------------------------------------------------------------

    def query(
        self,
        query_text: str,
        collection: str | None = None,
        n_results: int = 5,
        min_score: float = 0.3,
        source_filter: str | None = None,
        file_type_filter: str | None = None,
        rerank: bool = True,
        track_citations: bool = True,
    ) -> QueryResponse:
        """
        Query the vector store with optional reranking and citation tracking.

        Args:
            query_text: The search query.
            collection: Collection to search (None = default).
            n_results: Maximum results to return.
            min_score: Minimum similarity score (0-1).
            source_filter: Filter by source file path.
            file_type_filter: Filter by file type.
            rerank: Apply simple reranking heuristics.
            track_citations: Log citations to SQLite.

        Returns:
            QueryResponse with results and citations.
        """
        collection = collection or self.DEFAULT_COLLECTION

        if not self._chroma:
            return QueryResponse(query=query_text, results=[], citations=[], total_results=0)

        try:
            coll = self._chroma.get_collection(collection)
        except Exception:
            return QueryResponse(query=query_text, results=[], citations=[], total_results=0)

        if coll.count() == 0:
            return QueryResponse(query=query_text, results=[], citations=[], total_results=0)

        # Generate query embedding
        embedder = self._get_embedder()
        query_embedding = None
        if embedder:
            query_embedding = embedder.embed_single(query_text)

        # Build ChromaDB filter
        where_filter = self._build_where(source_filter, file_type_filter)

        # Query ChromaDB
        try:
            fetch_n = min(n_results * 3, 50)  # Fetch extra for filtering
            query_kwargs: dict[str, Any] = {
                "n_results": fetch_n,
                "include": ["documents", "metadatas", "distances"],
            }
            if query_embedding:
                query_kwargs["query_embeddings"] = [query_embedding]
            else:
                query_kwargs["query_texts"] = [query_text]

            if where_filter:
                query_kwargs["where"] = where_filter

            raw = coll.query(**query_kwargs)
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return QueryResponse(query=query_text, results=[], citations=[], total_results=0)

        if not raw or not raw.get("documents") or not raw["documents"][0]:
            return QueryResponse(query=query_text, results=[], citations=[], total_results=0)

        # Build results
        results: list[RetrievalResult] = []
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        dists = raw["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Convert distance to similarity score
            # Cosine distance: score = 1 - distance (for hnsw:space=cosine)
            score = max(0.0, 1.0 - dist)
            if score < min_score:
                continue

            page_val = meta.get("page", -1)
            results.append(RetrievalResult(
                content=doc,
                score=score,
                source_file=meta.get("source_file", ""),
                file_type=meta.get("file_type", ""),
                chunk_index=meta.get("chunk_index", 0),
                total_chunks=meta.get("total_chunks", 0),
                parent_doc_id=meta.get("parent_doc_id", ""),
                collection_name=collection,
                section=meta.get("section") or None,
                page=page_val if page_val >= 0 else None,
                metadata=meta,
            ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Rerank: boost results where query terms appear in content
        if rerank and results:
            results = self._rerank(query_text, results)

        # Trim to n_results
        results = results[:n_results]

        # Track citations
        citations: list[CitationRecord] = []
        if track_citations and results:
            now = time.time()
            for r in results:
                cid = uuid.uuid4().hex[:12]
                chunk_id = hashlib.sha256(
                    f"{r.parent_doc_id}::{r.chunk_index}".encode()
                ).hexdigest()[:16]
                citation = CitationRecord(
                    citation_id=cid,
                    query=query_text,
                    collection_name=collection,
                    chunk_id=chunk_id,
                    parent_doc_id=r.parent_doc_id,
                    source_file=r.source_file,
                    section=r.section,
                    score=r.score,
                    timestamp=now,
                )
                citations.append(citation)
                try:
                    self.db.insert_citation(citation)
                except Exception as exc:
                    logger.debug("Citation insert error: %s", exc)

        return QueryResponse(
            query=query_text,
            results=results,
            citations=citations,
            total_results=len(results),
        )

    def _build_where(
        self,
        source_filter: str | None,
        file_type_filter: str | None,
    ) -> dict | None:
        """Build a ChromaDB where filter."""
        conditions: list[dict] = []
        if source_filter:
            conditions.append({"source_file": {"$eq": source_filter}})
        if file_type_filter:
            conditions.append({"file_type": {"$eq": file_type_filter}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _rerank(query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """
        Simple reranking: boost score when query terms appear in content.

        This is a lightweight heuristic; no external reranker model needed.
        """
        query_terms = set(query.lower().split())
        if not query_terms:
            return results

        for r in results:
            content_lower = r.content.lower()
            match_count = sum(1 for t in query_terms if t in content_lower)
            boost = 0.05 * (match_count / max(1, len(query_terms)))
            r.score = min(1.0, r.score + boost)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # -----------------------------------------------------------------
    # CITATION MANAGEMENT
    # -----------------------------------------------------------------

    def list_citations(
        self,
        collection: str | None = None,
        doc_id: str | None = None,
        limit: int = 50,
    ) -> list[CitationRecord]:
        """List citation records."""
        return self.db.list_citations(
            collection_name=collection, doc_id=doc_id, limit=limit
        )

    # -----------------------------------------------------------------
    # WEB PAGE INGESTION
    # -----------------------------------------------------------------

    def ingest_url(
        self,
        url: str,
        collection: str | None = None,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestedDocument:
        """
        Fetch a web page, extract readable text, chunk, and ingest.

        Uses a readability-style approach: strip nav/ads/boilerplate,
        extract main content, chunk, embed, store.

        Args:
            url: The URL to fetch.
            collection: Target collection.
            doc_id: Optional document ID.
            metadata: Extra metadata.

        Returns:
            IngestedDocument record.

        Raises:
            RuntimeError: If requests library is not available.
            ValueError: If the URL is invalid or fetch fails.
        """
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not installed. Install with: pip install requests")

        collection = collection or self.DEFAULT_COLLECTION
        doc_id = doc_id or uuid.uuid4().hex[:12]

        self.db.create_collection(collection)

        # Load web ingestion config
        web_cfg = self._load_web_config()

        # Validate URL
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http/https supported.")

        # Fetch page
        try:
            resp = _requests_lib.get(
                url,
                timeout=web_cfg.get("timeout", 30),
                headers={"User-Agent": web_cfg.get("user_agent", "Opti-Oignon RAG/1.0")},
                allow_redirects=True,
            )
            resp.raise_for_status()
        except Exception as exc:
            raise ValueError(f"Failed to fetch URL {url}: {exc}") from exc

        max_size = web_cfg.get("max_page_size", 5 * 1024 * 1024)
        if len(resp.content) > max_size:
            raise ValueError(f"Page too large: {len(resp.content)} bytes (max {max_size})")

        # Extract text
        content_type = resp.headers.get("content-type", "")
        if "html" in content_type.lower() or resp.text.strip().startswith("<"):
            text = self._extract_html_text(resp.text, web_cfg)
        else:
            text = resp.text

        min_len = web_cfg.get("min_text_length", 100)
        if not text or len(text.strip()) < min_len:
            logger.warning("Extracted text too short from %s (%d chars)", url, len(text or ""))
            doc = IngestedDocument(
                doc_id=doc_id,
                collection_name=collection,
                source_file=url,
                file_type="html",
                chunk_count=0,
                raw_text_length=len(text) if text else 0,
                ingested_at=time.time(),
                metadata={**(metadata or {}), "url": url, "domain": parsed.netloc},
            )
            self.db.insert_document(doc)
            return doc

        # Chunk and ingest
        return self.ingest_text(
            text=text,
            source=url,
            file_type="text",
            collection=collection,
            doc_id=doc_id,
            metadata={**(metadata or {}), "url": url, "domain": parsed.netloc},
        )

    @staticmethod
    def _extract_html_text(
        html_content: str,
        web_cfg: dict[str, Any],
    ) -> str:
        """
        Extract readable text from HTML, stripping boilerplate.

        Readability-style approach:
        1. Remove script/style/nav/header/footer tags
        2. Remove elements matching boilerplate CSS classes/IDs
        3. Decode HTML entities
        4. Collapse whitespace
        """
        text = html_content

        # Step 1: Remove specified tags and their contents
        strip_tags = web_cfg.get("strip_tags", [
            "nav", "header", "footer", "aside", "script", "style",
            "noscript", "iframe", "svg", "form",
        ])
        for tag in strip_tags:
            # Remove <tag ...>...</tag> (non-greedy, case-insensitive)
            text = re.sub(
                rf'<{tag}[\s>].*?</{tag}>',
                ' ',
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            # Remove self-closing <tag ... />
            text = re.sub(
                rf'<{tag}\s[^>]*/?>',
                ' ',
                text,
                flags=re.IGNORECASE,
            )

        # Step 2: Remove elements with boilerplate class/id patterns
        boilerplate = web_cfg.get("boilerplate_patterns", [
            "sidebar", "menu", "navbar", "footer", "advertisement",
            "cookie", "popup", "modal", "banner", "social", "share", "comment",
        ])
        for pattern in boilerplate:
            # Remove divs/sections with matching class or id
            text = re.sub(
                rf'<(?:div|section|aside|ul|ol)[^>]*(?:class|id)=["\'][^"\']*{re.escape(pattern)}[^"\']*["\'][^>]*>.*?</(?:div|section|aside|ul|ol)>',
                ' ',
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )

        # Step 3: Strip all remaining HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Step 4: Decode HTML entities
        text = html_module.unescape(text)

        # Step 5: Collapse whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = text.strip()

        return text

    def _load_web_config(self) -> dict[str, Any]:
        """Load web ingestion config from rag.yaml."""
        try:
            import yaml
            config_path = Path(__file__).parent / "config" / "rag.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                return cfg.get("web_ingestion", {})
        except Exception as exc:
            logger.debug("Could not load rag.yaml web_ingestion: %s", exc)
        return {}


# =========================================================================
# YAML CONFIG LOADER
# =========================================================================

def load_rag_config() -> dict[str, Any]:
    """
    Load the full RAG configuration from config/rag.yaml.

    Returns a dict with keys: chunking, embedding, retrieval,
    web_ingestion, collections, storage.
    Falls back to defaults if the file is missing.
    """
    defaults: dict[str, Any] = {
        "chunking": {"chunk_size": 500, "chunk_overlap": 50},
        "embedding": {
            "model": "mxbai-embed-large",
            "fast_model": "nomic-embed-text",
            "ollama_url": "http://localhost:11434",
            "batch_size": 32,
            "timeout": 120,
        },
        "retrieval": {
            "n_results": 5,
            "min_score": 0.3,
            "rerank": True,
            "track_citations": True,
        },
        "web_ingestion": {"enabled": True, "timeout": 30},
        "collections": {"default": "default"},
        "storage": {
            "chroma_subdir": "chroma_v2",
            "db_filename": "rag_documents.db",
        },
    }
    try:
        import yaml
        config_path = Path(__file__).parent / "config" / "rag.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            # Merge loaded over defaults
            for key in defaults:
                if key in loaded:
                    if isinstance(defaults[key], dict) and isinstance(loaded[key], dict):
                        defaults[key].update(loaded[key])
                    else:
                        defaults[key] = loaded[key]
    except Exception as exc:
        logger.debug("Could not load rag.yaml: %s", exc)
    return defaults


# =========================================================================
# MODULE-LEVEL SINGLETON
# =========================================================================

_store_instance: RAGVectorStore | None = None


def get_rag_store(
    data_dir: str | Path | None = None,
    embedding_model: str = "mxbai-embed-large",
    ollama_url: str = "http://localhost:11434",
) -> RAGVectorStore:
    """Return the module-level RAGVectorStore singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = RAGVectorStore(
            data_dir=data_dir,
            embedding_model=embedding_model,
            ollama_url=ollama_url,
        )
    return _store_instance
