#!/usr/bin/env python3
"""
RAG VECTOR STORE -- Collection management, ingestion & retrieval.

Provides:
- Collection CRUD (create, list, delete, stats)
- Document ingestion pipeline: file/text -> RAGChunker -> embeddings -> the
  knowledge base's store
- Retrieval: query -> embedding -> top-k nearest -> optional reranking
- Citation tracking: SQLite-backed log of which chunk answered which query
- Metadata filtering (by source, date, collection)
- Configurable embedding model (Ollama embeddings)

Each collection maps to a collection of the namespace ``knowledge`` of the
knowledge base's store (``opti_oignon.vector_db``), which keeps the chunks'
texts, metadata and vectors encrypted like every other store. Documents are
tracked in a SQLite ``rag_documents.db`` so they can be listed and deleted
cleanly.
"""

import bisect
import hashlib
import html as html_module
import json
import logging
import re
import sqlite3
import threading
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

VECTOR_STORE_AVAILABLE = False
try:
    from opti_oignon import vector_db
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    vector_db = None  # type: ignore[assignment]

RAG_STORE_AVAILABLE = VECTOR_STORE_AVAILABLE
# The namespace of the knowledge base's store this store's collections live in.
VECTOR_NAMESPACE = "knowledge"
# The embedding model, and its server, this store embeds with when its caller names none -- the platform's
# ``get_rag_store()``: ``db_encryption migrate`` computes the vectors of ChromaDB's collections again with them.
EMBEDDING_MODEL = "mxbai-embed-large"
OLLAMA_URL = "http://localhost:11434"


# Every write of the store -- a collection made, a document's landing, a removal -- holds this one lock of the
# process: a removal never meets a document half landed, and a landing finds its collection still there or nothing.
_WRITES = threading.RLock()
# How many times each collection of each store was removed in this process: a landing that began before a removal
# refuses, even when the collection was made again since under the same name.
_REMOVALS: dict[tuple[str, str], int] = {}

# The inventory's labels (``store_inventory``) what a removal could not finish is named under.
_DOCUMENTS_LABEL = "the knowledge base: documents, their chunks and vectors (not user-scoped)"
_AUDIT_LABEL = "the knowledge base's injection audit (the flagged chunks' text)"
_JOBS_LABEL = "knowledge-base ingest jobs"


class CollectionRemoved(RuntimeError):
    """The collection a document was being stored into was removed meanwhile: nothing of the document landed."""


class RowsLeft(RuntimeError):
    """A removal took the chunks, and the rows could not go after them: listed until removed again."""


def _waits(exc: BaseException) -> bool:
    """Whether ``exc`` says the knowledge base waits for its migration."""
    return vector_db is not None and isinstance(exc, vector_db.MigrationPending)


class EmbeddingsUnavailable(RuntimeError):
    """No chunk of an ingestion has an embedding: nothing was stored."""


# =========================================================================
# DATA STRUCTURES
# =========================================================================

@dataclass
class CollectionInfo:
    """Metadata about a collection of the knowledge base."""
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
        # A document, a collection or a search deleted leaves no word in the file's pages.
        conn.execute("PRAGMA secure_delete = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def empty_log(self) -> bool:
        """Empty the write-ahead log, never waiting: whether a read or a write in progress kept it."""
        conn = _safe_connect(self.db_path)
        try:
            conn.execute("PRAGMA busy_timeout = 0")
            return bool(conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0])
        finally:
            conn.close()

    def has_collection(self, name: str) -> bool:
        with self._conn() as conn:
            return conn.execute("SELECT 1 FROM collections WHERE name = ?", (name,)).fetchone() is not None

    def collection_names(self) -> list[str]:
        with self._conn() as conn:
            return [row[0] for row in conn.execute("SELECT name FROM collections ORDER BY created_at")]

    def document_ids(self) -> list[str]:
        with self._conn() as conn:
            return [row[0] for row in conn.execute("SELECT doc_id FROM documents ORDER BY ingested_at")]

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


def _real_or_kept(match: re.Match) -> str:
    """A blank for a real element; for an opening left open, taken at once, the text as it was."""
    return " " if match.group("real") is not None else match.group(0)


_BOILERPLATE_OPENING = re.compile(r"<(?:div|section|aside|ul|ol)", re.IGNORECASE)
_BOILERPLATE_ATTRIBUTE = re.compile(r"(?:class|id)=[\"']", re.IGNORECASE)
_BOILERPLATE_CLOSING = re.compile(r"</(?:div|section|aside|ul|ol)>", re.IGNORECASE)
_QUOTE = re.compile(r"[\"']")


def _strip_boilerplate(text: str, word: str) -> str:
    """``text`` with each element whose class or id holds ``word`` taken out, up to the first closing tag after it.

    What ``re.sub(rf'<(?:div|section|aside|ul|ol)[^>]*(?:class|id)=["\\'][^"\\']*{re.escape(word)}[^"\\']*["\\']'
    rf'[^>]*>.*?</(?:div|section|aside|ul|ol)>', ' ', text, flags=re.DOTALL | re.IGNORECASE)`` gives, in time
    linear in the text: that pattern backtracks over the attributes of every opening and over every quoted value,
    and reads from every opening left open to the end of the text. Here each ``class=`` or ``id=`` with its quote is
    found once, and whether the pattern goes on from it -- the next quote ends a value that holds the word, a ``>``
    follows that quote, then a closing tag -- is decided once. An opening goes on from the last such attribute
    before its first ``>``: the pattern's greedy run over the attributes tries the longest first. Two values never
    overlap (each ends at or before the next attribute's quote), so each is read once. A word that holds a quote
    names no element here (the pattern would let it run across one); it is left out with a warning.
    """
    if '"' in word or "'" in word:
        logger.warning("Boilerplate word %r left out: a word cannot hold a quote", word)
        return text
    holds = re.compile(re.escape(word), re.IGNORECASE)
    angles = [found.start() for found in re.finditer(">", text)]
    quotes = [found.start() for found in _QUOTE.finditer(text)]
    closings = list(_BOILERPLATE_CLOSING.finditer(text))
    closing_starts = [found.start() for found in closings]
    starts, ends = [], []
    for attribute in _BOILERPLATE_ATTRIBUTE.finditer(text):
        opened = attribute.end() - 1
        at = bisect.bisect_right(quotes, opened)
        if at == len(quotes) or holds.search(text, opened + 1, quotes[at]) is None:
            continue
        at = bisect.bisect_left(angles, quotes[at] + 1)
        if at == len(angles):
            continue
        at = bisect.bisect_left(closing_starts, angles[at] + 1)
        if at == len(closings):
            continue
        starts.append(attribute.start())
        ends.append(closings[at].end())
    out, last = [], 0
    for opening in _BOILERPLATE_OPENING.finditer(text):
        if opening.start() < last:
            continue
        at = bisect.bisect_left(angles, opening.end())
        bound = angles[at] if at < len(angles) else len(text)
        at = bisect.bisect_left(starts, bound) - 1
        if at < 0 or starts[at] < opening.end():
            continue
        out.append(text[last:opening.start()])
        out.append(" ")
        last = ends[at]
    out.append(text[last:])
    return "".join(out)


# =========================================================================
# VECTOR STORE
# =========================================================================

class WebIngestionDisabled(RuntimeError):
    """Page ingestion refused by the configuration's own switch, by name."""

    refusal = "disabled"

    def __init__(self):
        super().__init__(
            "Ingesting a web page is refused: web_ingestion.enabled in rag.yaml is not true; "
            "nothing was fetched."
        )


class RAGVectorStore:
    """
    The knowledge base's vector store, with collection management,
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
        embedding_model: str = EMBEDDING_MODEL,
        ollama_url: str = OLLAMA_URL,
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

        # Where ChromaDB kept this store's collections before the knowledge
        # base's store: never made, never read (``db_encryption migrate``
        # moves them).
        self.chroma_dir = self.data_dir / "chroma_v2"

        # SQLite tracking DB
        self.db = _RAGDatabase(self.data_dir / "rag_documents.db")

        # The chunks' texts, metadata and vectors: the namespace ``knowledge``
        # of the knowledge base's store, beside the tracking DB, opened like
        # it through safe_connect (encrypted under the platform's key).
        if VECTOR_STORE_AVAILABLE:
            self.vectors_path = self.data_dir / vector_db.FILENAME
            self._chroma = vector_db.Client(self.vectors_path, VECTOR_NAMESPACE, legacy=self.chroma_dir)
        else:
            self._chroma = None
            logger.warning("the knowledge base's store does not import -- vector store disabled")

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

    @property
    def embedding_model(self) -> str:
        """The model this store embeds with."""
        return self._embedding_model

    def chunker_settings(self) -> tuple[int, int]:
        """(chunk_size, chunk_overlap) of this store's chunker, for a chunk
        task that runs elsewhere and must cut as the store does."""
        chunker = self._get_chunker()
        return chunker.chunk_size, chunker.chunk_overlap

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

    def _gate(self) -> None:
        """Refuse, before anything is written, while the knowledge base waits
        for its migration (``vector_db.MigrationPending``). A client with
        ChromaDB's interface alone has no migration to wait for."""
        ready = getattr(self._chroma, "ready", None)
        if ready is not None:
            ready()

    def _get_collection(self, name: str):
        """Get or create a collection of the knowledge base."""
        if not self._chroma:
            raise RuntimeError("the knowledge base's store is not available")
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
            raise RuntimeError("the knowledge base's store is not available")
        with _WRITES:
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
        """Delete a collection and all its documents/chunks (``forget_collection``): whether it is gone -- True
        for a collection that was not there.

        While the knowledge base waits for its migration nothing is deleted
        (``vector_db.MigrationPending``): the chunks still in ChromaDB's store
        would come back with it.
        """
        try:
            self.forget_collection(name)
            return True
        except Exception as exc:
            if _waits(exc):
                raise
            logger.error("Failed to delete collection %s: %s", name, type(exc).__name__)
            return False

    def forget_collection(self, name: str, *, sweep: bool = True, empty: bool = True) -> tuple[bool, list[str]]:
        """Forget a collection: its chunks first (the vector collection, overwritten), then the audit's entries and
        the ingest jobs' records of its documents, then its rows (overwritten), both logs emptied; whether it was
        there, and what could not be removed, named under the inventory's labels. Chunks that cannot be removed
        refuse it, the collection kept, the failure raised as it came; a knowledge base that waits for its
        migration refuses before anything goes (``MigrationPending``). With ``sweep``, what removals made before
        this left goes too (``sweep_orphans``); without ``empty``, the logs are left to the caller (the wipe empties
        them once)."""
        with _WRITES:
            self._gate()
            if not self._chroma:
                raise RuntimeError("the knowledge base's store is not available: nothing was removed")
            doc_ids = self.db.get_doc_ids_for_collection(name)
            sources = [d.source_file for d in self.db.list_documents(collection_name=name, limit=1_000_000)]
            chunk_ids: list[str] = []
            held = self.db.has_collection(name)
            try:
                coll = self._chroma.get_collection(name)
            except vector_db.CollectionNotFound:
                coll = None
            if coll is not None:
                chunk_ids = list(coll.get(include=[]).get("ids") or [])
                self._chroma.delete_collection(name)
                held = True
            # A landing that began before this removal refuses, the collection made again or not.
            _REMOVALS[(str(self.data_dir), name)] = self.removals(name) + 1
            left = self._forget_drawn(chunk_ids, doc_ids, sources, collection=name, whole=True)
            try:
                self.db.delete_collection(name)
            except Exception as exc:
                raise RowsLeft(f"the chunks of collection {name!r} were removed; its rows could not be "
                               f"({type(exc).__name__}): it stays listed, delete it again") from exc
            if sweep:
                left += self.sweep_orphans()
            if empty:
                left += self.empty_logs()
        logger.info("Forgot a collection: %d document(s), %d chunk(s)", len(doc_ids), len(chunk_ids))
        return held, list(dict.fromkeys(left))

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
        Ingest a file: extract text -> chunk -> embed -> store in the
        knowledge base's store.

        Args:
            filepath: Path to the file.
            collection: Target collection name (default: 'default').
            doc_id: Optional document ID.
            metadata: Extra metadata to attach.

        Returns:
            IngestedDocument record.
        """
        doc_id = doc_id or uuid.uuid4().hex[:12]
        # Counted before the file is read: a removal of its collection while it is read refuses its landing.
        since = self.removals(collection or self.DEFAULT_COLLECTION)
        chunker = self._get_chunker()
        result = chunker.chunk_file(filepath, doc_id=doc_id)
        return self.store_chunked(result, collection=collection, metadata=metadata, since=since)

    def store_chunked(
        self,
        result: Any,
        collection: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        since: int | None = None,
    ) -> IngestedDocument:
        """
        Store a file already chunked: embed its chunks into the store and
        record the document. The storage half of ingest_file, for a job
        whose chunking ran elsewhere (a background worker).

        Args:
            result: The chunker's ChunkingResult for the file.
            collection: Target collection name (default: 'default').
            metadata: Extra metadata to attach.
            since: The collection's removals counted when the file began to be read (``removals``), by the caller
                that read it; now, when not given.

        Returns:
            IngestedDocument record.
        """
        collection = collection or self.DEFAULT_COLLECTION
        filepath = result.source_file
        doc_id = result.doc_id
        since = self.removals(collection) if since is None else since

        # Ensure collection exists -- unless it was removed while the file was read: then nothing is made again.
        self._gate()
        self._still(collection, since, doc_id)
        self.db.create_collection(collection)

        if not result.chunks:
            logger.warning("No chunks produced for document %s", doc_id)
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
            self._land(doc, since=since)
            return doc

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
        # Store chunks in the knowledge base's store, and the document with them
        self._store_chunks(result.chunks, collection, doc=doc, since=since)
        logger.info(
            "Ingested document %s: %d chunks into collection '%s'",
            doc_id, result.chunk_count, collection,
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
        *,
        since: int | None = None,
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
        since = self.removals(collection) if since is None else since

        self._gate()
        self._still(collection, since, doc_id)
        self.db.create_collection(collection)

        chunker = self._get_chunker()
        result = chunker.chunk_text(text, source=source, file_type=file_type, doc_id=doc_id)

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
        if result.chunks:
            self._store_chunks(result.chunks, collection, doc=doc, since=since)
        else:
            self._land(doc, since=since)
        return doc

    def _store_chunks(self, chunks: list, collection_name: str, *, doc: IngestedDocument | None = None,
                      since: int | None = None) -> None:
        """Embed and store chunks in the knowledge base's store, and ``doc``'s row with them.

        A chunk whose embedding fails is dropped and counted; when no chunk has
        an embedding -- the embedder is missing, or every embedding failed --
        nothing is stored and the ingestion is refused by name: the store
        computes no vector of its own. The embedding runs first, outside the
        store's lock; the chunks and the row land under it, once ``doc``'s
        collection is found still there (``_land``).
        """
        if not self._chroma:
            raise RuntimeError("the knowledge base's store is not available")

        # The collection is there before the embedding, as a search of it finds it; its chunks land only with
        # the document's row.
        self._get_collection(collection_name)
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
        if not embedder:
            raise EmbeddingsUnavailable(
                f"no embedder for the {len(ids)} chunk(s) of collection '{collection_name}': "
                "nothing was stored (the store computes no vector of its own)"
            )
        embeddings = embedder.embed(
            documents, show_progress=False
        )
        # Filter out None embeddings
        valid = [
            (i, doc, meta, emb)
            for i, doc, meta, emb in zip(ids, documents, metadatas, embeddings)
            if emb is not None
        ]
        if not valid:
            raise EmbeddingsUnavailable(
                f"every embedding failed for the {len(ids)} chunk(s) of collection "
                f"'{collection_name}': nothing was stored"
            )
        if len(valid) < len(ids):
            logger.warning("%d of %d chunk(s) dropped: their embedding failed",
                           len(ids) - len(valid), len(ids))
        v_ids, v_docs, v_metas, v_embs = zip(*valid)

        def upsert() -> None:
            self._get_collection(collection_name).upsert(
                ids=list(v_ids),
                documents=list(v_docs),
                metadatas=list(v_metas),
                embeddings=list(v_embs),
            )

        if doc is None:
            with _WRITES:
                upsert()
            return
        self._land(doc, upsert, since=since)

    def _still(self, collection: str, since: int, doc_id: str) -> None:
        """Refuse, before the collection is made again, a document whose collection was removed while it was read
        or fetched (``CollectionRemoved``)."""
        if self.removals(collection) != since:
            raise CollectionRemoved(
                f"the collection {collection!r} was removed while document {doc_id} was being read: nothing of it "
                "was kept")

    def _land(self, doc: IngestedDocument, chunks: Any = None, *, since: int | None = None) -> None:
        """Land a document under the store's lock: its chunks (``chunks``, a call that stores them), then its row,
        only while its collection is still there and was not removed since ``since`` (its removals counted when
        the document's storing began) -- one removed while the document was read or embedded refuses, nothing of
        it stored, even when it was made again since under its name (``CollectionRemoved``)."""
        with _WRITES:
            removed = since is not None and self.removals(doc.collection_name) != since
            if removed or not self.db.has_collection(doc.collection_name):
                raise CollectionRemoved(
                    f"the collection {doc.collection_name!r} was removed while document {doc.doc_id} was being "
                    "stored: nothing of it was kept")
            if chunks is not None:
                chunks()
                self.db.touch_collection(doc.collection_name)
            self.db.insert_document(doc)

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
        """Delete a document and its chunks from the knowledge base's store + SQLite (``forget_document``): whether
        it went.

        While the knowledge base waits for its migration nothing is deleted
        (``vector_db.MigrationPending``): the chunks still in ChromaDB's store
        would come back with it.
        """
        try:
            return self.forget_document(doc_id)[0]
        except Exception as exc:
            if _waits(exc):
                raise
            logger.error("Failed to delete document %s: %s", doc_id, type(exc).__name__)
            return False

    def forget_document(self, doc_id: str, *, sweep: bool = True, empty: bool = True) -> tuple[bool, list[str]]:
        """Forget a document: its chunks first (overwritten), then the audit's entries of those chunks and the
        ingest jobs' records of the document, then its citations and its row (overwritten), both logs emptied;
        whether it was there, and what could not be removed, named under the inventory's labels. Chunks that cannot
        be removed refuse it: the document stays listed with them, the failure raised as it came, never a row gone
        with its chunks still searchable. A knowledge base that waits for its migration refuses before anything goes.
        ``sweep`` and ``empty`` as ``forget_collection``."""
        with _WRITES:
            record = self.db.get_document(doc_id)
            if record is None:
                return False, []
            self._gate()
            if not self._chroma:
                raise RuntimeError("the knowledge base's store is not available: nothing was removed")
            chunk_ids: list[str] = []
            try:
                coll = self._chroma.get_collection(record.collection_name)
            except vector_db.CollectionNotFound:
                coll = None
            if coll is not None:
                chunk_ids = list(coll.get(where={"parent_doc_id": {"$eq": doc_id}}, include=[]).get("ids") or [])
                if chunk_ids:
                    coll.delete(ids=chunk_ids)
            # The audit's entries keyed by the file in its collection go too, unless another document there was
            # read from the same file.
            others = [d for d in self.db.list_documents(collection_name=record.collection_name, limit=1_000_000)
                      if d.doc_id != doc_id and d.source_file == record.source_file]
            left = self._forget_drawn(chunk_ids, [doc_id], [] if others else [record.source_file],
                                      collection=record.collection_name)
            try:
                self.db.delete_document(doc_id)
            except Exception as exc:
                raise RowsLeft(f"the chunks of document {doc_id} were removed; its row and its citations could "
                               f"not be ({type(exc).__name__}): it stays listed, delete it again") from exc
            if sweep:
                left += self.sweep_orphans()
            if empty:
                left += self.empty_logs()
        logger.info("Forgot document %s: %d chunk(s)", doc_id, len(chunk_ids))
        return True, list(dict.fromkeys(left))

    def _forget_drawn(self, chunk_ids: list[str], doc_ids: list[str], sources: list[str], *, collection: str,
                      whole: bool = False) -> list[str]:
        """What the audit and the ingest jobs hold of these chunks and documents, forgotten: the audit's entries of
        the chunks, and those of these ``sources`` in ``collection`` (with ``whole``, every entry and every job of
        the collection); what could not be, named. An audit out of reach now is not reached again by a later
        removal: its clear, or the wipe, removes what it kept."""
        left: list[str] = []
        try:
            from opti_oignon.rag_sanitizer import existing_audit_log

            audit = existing_audit_log()
            if audit is not None:
                _gone, busy = audit.forget(chunk_ids, sources=sources, collection=collection, whole=whole)
                if busy:
                    left.append(f"{_AUDIT_LABEL} (a read or a write in progress keeps what was erased in the "
                                "database's file until it ends)")
        except Exception as exc:  # noqa: BLE001 - named: no later removal finds these entries again
            left.append(f"{_AUDIT_LABEL} (the removed chunks' entries not removed: {type(exc).__name__}; the "
                        "audit's clear removes them)")
        try:
            from opti_oignon.rag.batch_ingest import existing_jobs

            jobs = existing_jobs(self.data_dir)
            if jobs is not None:
                busy = jobs.forget_documents(doc_ids)[1]
                if whole:
                    from opti_oignon.rag.batch_ingest import stop_jobs_of

                    # A job storing into the collection stops: what it still holds refuses to land.
                    stop_jobs_of(jobs, collection)
                    busy = jobs.forget_collection(collection)[1] or busy
                if busy:
                    left.append(f"{_JOBS_LABEL} (a read or a write in progress keeps what was erased in the "
                                "database's file until it ends)")
        except Exception as exc:  # noqa: BLE001 - named, and swept by the next removal
            left.append(f"{_JOBS_LABEL} (the removed documents' records not removed: {type(exc).__name__})")
        return left

    def sweep_orphans(self) -> list[str]:
        """What removals made before this left: in a collection the knowledge base lists, the chunks whose document
        has no row (a removal took the row, and its chunks' removal failed and was only logged); the ingest records
        of documents gone -- removed, overwritten; what could not be, named. A vector collection no row names is
        left as it is: no removal leaves one, and nothing of the knowledge base reads it. Under the store's lock,
        so no document is ever half landed when it looks. A knowledge base that waits for its migration is not
        swept."""
        left: list[str] = []
        with _WRITES:
            try:
                live = set(self.db.document_ids())
                if self._chroma and getattr(self._chroma, "ready", None) is not None:
                    names = set(self.db.collection_names())
                    for coll in self._chroma.list_collections():
                        if coll.name not in names:
                            continue
                        got = coll.get(include=["metadatas"])
                        stale = [item_id for item_id, meta in zip(got.get("ids") or [], got.get("metadatas") or [])
                                 if (meta or {}).get("parent_doc_id") and meta["parent_doc_id"] not in live]
                        if stale:
                            coll.delete(ids=stale)
            except Exception as exc:  # noqa: BLE001 - named
                if _waits(exc):
                    return left
                left.append(f"{_DOCUMENTS_LABEL} (what earlier removals left, not swept: {type(exc).__name__})")
                return left
            try:
                from opti_oignon.rag.batch_ingest import existing_jobs

                jobs = existing_jobs(self.data_dir)
                if jobs is not None and jobs.forget_orphans(live)[1]:
                    left.append(f"{_JOBS_LABEL} (a read or a write in progress keeps what was erased in the "
                                "database's file until it ends)")
            except Exception as exc:  # noqa: BLE001 - named
                left.append(f"{_JOBS_LABEL} (what earlier removals left, not swept: {type(exc).__name__})")
        return left

    def empty_logs(self) -> list[str]:
        """Empty the tracking database's log and the chunks' store's, never waiting; what a read or a write in
        progress kept, named."""
        left: list[str] = []
        busy = []
        try:
            if self.db.empty_log():
                busy.append("rag_documents.db")
        except Exception as exc:  # noqa: BLE001 - named
            left.append(f"{_DOCUMENTS_LABEL} (the log not emptied: {type(exc).__name__})")
        try:
            if self._chroma and not self._chroma.empty_log():
                busy.append("vectors.db")
        except Exception as exc:  # noqa: BLE001 - named
            left.append(f"{_DOCUMENTS_LABEL} (the chunks' log not emptied: {type(exc).__name__})")
        if busy:
            left.append(f"{_DOCUMENTS_LABEL} (a read or a write in progress keeps what was erased in the database's "
                        f"file until it ends: {', '.join(busy)})")
        return left

    def removals(self, name: str) -> int:
        """How many times this store's collection ``name`` was removed in this process."""
        return _REMOVALS.get((str(self.data_dir), name), 0)

    def counts(self) -> dict[str, int]:
        """How many collections, documents and chunks the knowledge base lists, as a wipe's preview counts them."""
        rows = self.db.list_collections()
        return {"collections": len(rows),
                "documents": sum(int(r.get("document_count") or 0) for r in rows),
                "chunks": sum(int(r.get("chunk_count") or 0) for r in rows)}

    def fingerprint(self) -> str:
        """Four characters drawn from every collection and document the knowledge base lists: a document that came
        while another went, the counts unchanged, changes them."""
        listed = sorted(self.db.collection_names()) + ["--"] + sorted(self.db.document_ids())
        return hashlib.sha256("\n".join(listed).encode("utf-8")).hexdigest()[:4]

    def export_all(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Every collection, its description, and each of its documents -- source, type, times, metadata -- with
        its chunks' text in their order; and what could not be read, named."""
        out: list[dict[str, Any]] = []
        missing: list[str] = []
        for row in self.db.list_collections():
            name = row["name"]
            chunks: dict[str, list[tuple[int, str]]] = {}
            if self._chroma:
                try:
                    coll = self._chroma.get_collection(name)
                    got = coll.get(include=["metadatas", "documents"])
                    for text, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
                        meta = meta or {}
                        chunks.setdefault(str(meta.get("parent_doc_id", "")), []).append(
                            (int(meta.get("chunk_index", 0) or 0), text or ""))
                except vector_db.CollectionNotFound:
                    pass
                except Exception as exc:  # noqa: BLE001 - named, the rows still exported
                    missing.append(f"{_DOCUMENTS_LABEL} (the chunks of a collection not exported: "
                                   f"{type(exc).__name__})")
            documents = []
            for doc in self.db.list_documents(collection_name=name, limit=1_000_000):
                documents.append({
                    "doc_id": doc.doc_id,
                    "source_file": doc.source_file,
                    "file_type": doc.file_type,
                    "ingested_at": doc.ingested_at,
                    "metadata": doc.metadata,
                    "chunks": [text for _index, text in sorted(chunks.get(doc.doc_id, []), key=lambda c: c[0])],
                })
            out.append({"name": name, "description": row.get("description", ""), "created_at": row.get("created_at"),
                        "documents": documents})
        return out, missing

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
        except Exception as exc:
            if _waits(exc):
                logger.warning("nothing searched: %s", exc)
            return QueryResponse(query=query_text, results=[], citations=[], total_results=0)

        if coll.count() == 0:
            return QueryResponse(query=query_text, results=[], citations=[], total_results=0)

        # Generate query embedding: without one there is nothing to search
        # with, as the store computes no vector of its own.
        embedder = self._get_embedder()
        query_embedding = None
        if embedder:
            query_embedding = embedder.embed_single(query_text)
        if not query_embedding:
            logger.warning("no embedding for the query: nothing searched in collection '%s'", collection)
            return QueryResponse(query=query_text, results=[], citations=[], total_results=0)

        # Build the metadata filter
        where_filter = self._build_where(source_filter, file_type_filter)

        # Query the knowledge base's store
        try:
            fetch_n = min(n_results * 3, 50)  # Fetch extra for filtering
            query_kwargs: dict[str, Any] = {
                "n_results": fetch_n,
                "include": ["documents", "metadatas", "distances"],
                "query_embeddings": [query_embedding],
            }

            if where_filter:
                query_kwargs["where"] = where_filter

            raw = coll.query(**query_kwargs)
        except Exception as exc:
            logger.error("knowledge base query failed: %s", exc)
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
        """Build a metadata where filter."""
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
        extract main content, chunk, embed, store. The page is fetched
        through the web search module's page fetcher, behind the web gate
        and from a public address only (see ``_fetch_page``); nothing is
        recorded, not even the collection, until the page is in hand.

        Args:
            url: The URL to fetch.
            collection: Target collection.
            doc_id: Optional document ID.
            metadata: Extra metadata.

        Returns:
            IngestedDocument record.

        Raises:
            RuntimeError: If the page fetcher is not available, or its gate
                refuses (outside Daily mode, or with the search kill switch
                engaged or unreadable), or ``web_ingestion.enabled`` in
                ``rag.yaml`` is anything but true
                (``WebIngestionDisabled``); a refusal carries its name.
            ValueError: If the URL is invalid, its destination is refused,
                the page is too large, or the fetch fails.
        """
        # Validate URL
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http/https supported.")

        # Load web ingestion config; its own switch is asked before any
        # request, and a value that is not true refuses.
        web_cfg = self._load_web_config()
        if web_cfg.get("enabled", True) is not True:
            raise WebIngestionDisabled()

        # Nothing is fetched while the knowledge base waits for its migration.
        self._gate()
        collection = collection or self.DEFAULT_COLLECTION
        # Counted before the page is fetched: a removal of its collection meanwhile refuses its landing.
        since = self.removals(collection)

        # Fetch page
        page = self._fetch_page(url, web_cfg)

        doc_id = doc_id or uuid.uuid4().hex[:12]

        self._still(collection, since, doc_id)
        self.db.create_collection(collection)

        # Extract text
        content_type = page.content_type
        if "html" in content_type.lower() or page.text.strip().startswith("<"):
            text = self._extract_html_text(page.text, web_cfg)
        else:
            text = page.text

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
            self._land(doc, since=since)
            return doc

        # Chunk and ingest
        return self.ingest_text(
            text=text,
            source=url,
            file_type="text",
            collection=collection,
            doc_id=doc_id,
            metadata={**(metadata or {}), "url": url, "domain": parsed.netloc},
            since=since,
        )

    @staticmethod
    def _fetch_page(url: str, web_cfg: dict[str, Any]):
        """Fetch ``url`` through the page fetcher, with the configured bounds.

        The fetcher opens the web gate before every request and reaches only
        a public address; the store asks nothing of the network itself.
        """
        try:
            from opti_oignon.web_search import fetch_page
        except ImportError as exc:
            raise RuntimeError("The page fetcher is not available; nothing was fetched.") from exc
        ports = web_cfg.get("allowed_ports") or ()
        return fetch_page(
            url,
            timeout=web_cfg.get("timeout", 30),
            max_bytes=web_cfg.get("max_page_size", 5 * 1024 * 1024),
            user_agent=web_cfg.get("user_agent", "Opti-Oignon RAG/1.0"),
            allowed_ports=tuple(ports) if isinstance(ports, (list, tuple)) else (),
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
            # The tag's name is read as written, never as a pattern. Linear on an opening left open: one with no
            # closing after it takes the rest at once -- no opening after it has one either -- and is kept, as only
            # an element sets ``real``.
            name = re.escape(tag)
            # Remove <tag ...>...</tag> (non-greedy, case-insensitive)
            text = re.sub(
                rf'<{name}[\s>](?:.*?</{name}>(?P<real>)|.*)',
                _real_or_kept,
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            # Remove self-closing <tag ... />
            text = re.sub(
                rf'<{name}\s(?:[^>]*>(?P<real>)|[^>]*)',
                _real_or_kept,
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
            text = _strip_boilerplate(text, pattern)

        # Step 3: Strip all remaining HTML tags. Linear on a "<" left open: one with no ">" after it takes the rest at
        # once -- no "<" after it can close either -- and is kept, as only a tag sets ``real``.
        text = re.sub(r'<[^>]+>(?P<real>)|<[^>]*',
                      lambda m: " " if m.group("real") is not None else m.group(0), text)

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


def _default_data_dir() -> Path:
    """The knowledge base's folder when none is given, by the store's own rule (``RAGVectorStore.__init__``)."""
    try:
        from opti_oignon.config import DATA_DIR
        return Path(DATA_DIR) / "rag"
    except ImportError:
        return Path.home() / ".opti-oignon" / "data" / "rag"


def forget_every_query(data_dir: str | Path | None = None) -> int:
    """Delete every search the knowledge base logged -- each query, word for word, with what it cited -- overwritten,
    the log emptied; how many. The documents and the collections stay.

    The folder is the store's, else the one given, else the default. A file that is not there holds none, and is not
    created; one that is there and cannot be emptied raises: never taken for an empty one. The account wipe asks it
    for a single-user install's one user: the log is the install's.
    """
    if data_dir is None:
        data_dir = _store_instance.data_dir if _store_instance is not None else _default_data_dir()
    path = Path(data_dir) / "rag_documents.db"
    if not path.exists():
        return 0
    conn = _safe_connect(path)
    try:
        conn.execute("PRAGMA secure_delete = ON")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        gone = conn.execute("DELETE FROM citations").rowcount if "citations" in tables else 0
        conn.commit()
        conn.execute("PRAGMA busy_timeout = 0")
        if conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
            logger.warning("the knowledge base's searches forgotten; a read in progress keeps the write-ahead log, "
                           "whose older frames hold them until the last connection closes")
        return gone
    finally:
        conn.close()


def existing_rag_store() -> RAGVectorStore | None:
    """The module's store when it is built, or built now when its tracking file is there in the default folder; None
    otherwise: asking an install that has no knowledge base makes nothing."""
    if _store_instance is not None:
        return _store_instance
    if not (_default_data_dir() / "rag_documents.db").exists():
        return None
    return get_rag_store()


def get_rag_store(
    data_dir: str | Path | None = None,
    embedding_model: str = EMBEDDING_MODEL,
    ollama_url: str = OLLAMA_URL,
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
