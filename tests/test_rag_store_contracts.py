#!/usr/bin/env python3
"""What the RAG store promises about the bytes it keeps and the hits it returns.

The store has two halves and they answer to different masters. The SQLite half
tracks which documents exist, which collection owns them, and which chunk once
answered which query -- provenance the rest of the system reads as fact. That
half must be parameterised end to end: a document title or a collection name is
DATA, and a name carrying SQL metacharacters has to round-trip as a string,
never execute. It must also cascade: a document removed takes its citations with
it, because a citation whose parent is gone is a dangling claim of provenance.

The vector half answers to availability. A query is a read on a best-effort
index that may be missing, empty, or throwing, and none of those is an error the
caller should see -- every one degrades to an empty response, never an
exception, because a retrieval that raises takes the whole answer down with it.
When it does return hits it converts cosine distance to a similarity floored at
zero (a distance above one is legal and must not become a negative score),
drops everything under the score floor, ranks, applies the term-overlap rerank
heuristic, trims, and logs one citation per surviving hit under a chunk id
derived from parent-and-index so the same chunk always lands the same id.

The rerank is a documented heuristic, not a model, and these contracts pin it
exactly so the day a real cross-encoder replaces it, that is a deliberate,
contract-visible change and not a silent drift.

Ingestion records even the empty case: a document that produced no chunks is
still written, so it stays visible and deletable rather than vanishing. URL
ingestion admits only http/https at the scheme gate and refuses an oversize page
-- the network-facing guards, pinned as-is; their wider gaps (no block on
private or loopback targets, a size cap applied only after the body is already
in memory) are carried findings for a security cycle, not defects fixed here.

The connection factory is the seam: every DB touch flows through the module's
``safe_connect``. When that seam's backing module is unreachable the store falls
back to a bare sqlite connection and keeps working -- the documented fail-open
in Daily mode -- and that fallback is pinned too, so a regression that turned it
into a hard failure would surface here.

Loaded through the shared isolation window. Every project module the store might
reach for is either blocked (and proven unreachable before the module runs) or
seeded; the lazy embedder/chunker seams are never left to resolve on their own,
so no real weight-loading code is ever touched by these contracts.
"""

import hashlib
import sqlite3
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.rag_store"
_BLOCKED = [
    "opti_oignon.config",
    "opti_oignon.rag_chunker",
    "opti_oignon.rag.config",
    "opti_oignon.rag.embeddings",
]


def _load(*, seed_db_utils=True):
    """Load rag_store in isolation.

    With ``seed_db_utils`` the connection seam resolves to a stand-in whose
    ``safe_connect`` is a plain sqlite connection; without it the name is
    blocked so the module's own fallback path is exercised instead.
    """
    seeded = {}
    blocked = list(_BLOCKED)
    if seed_db_utils:
        du = types.ModuleType("opti_oignon.db_utils")
        du.safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)
        seeded["opti_oignon.db_utils"] = du
    else:
        blocked.append("opti_oignon.db_utils")
    loaded, restore = isolate(targets={_TARGET: source("rag_store.py")},
                              blocked=blocked, seeded=seeded)
    return loaded[_TARGET], restore


# --- test doubles ---------------------------------------------------------
# In-memory stand-ins for the ChromaDB client/collection, the Ollama embedder,
# and the chunker. They are injected onto the store instance so the lazy
# getters return them without reaching any real backend.

class _FakeCollection:
    def __init__(self, *, count_val=0, query_payload=None,
                 raise_on_query=False, get_ids=None):
        self._count = count_val
        self._payload = query_payload if query_payload is not None else {}
        self._raise_on_query = raise_on_query
        self._get_ids = list(get_ids or [])
        self.upserts = []
        self.deleted = []

    def count(self):
        return self._count

    def upsert(self, ids=None, documents=None, metadatas=None, embeddings=None):
        self.upserts.append({
            "ids": ids, "documents": documents,
            "metadatas": metadatas, "embeddings": embeddings,
        })

    def query(self, **kwargs):
        if self._raise_on_query:
            raise RuntimeError("backend query failure")
        return self._payload

    def get(self, where=None, include=None):
        return {"ids": list(self._get_ids)}

    def delete(self, ids=None):
        self.deleted.append(list(ids or []))


class _FakeChroma:
    def __init__(self, collections=None, *, raise_on_get=False):
        self._collections = dict(collections or {})
        self._raise_on_get = raise_on_get
        self.created = []
        self.deleted = []

    def get_or_create_collection(self, name=None, metadata=None):
        self.created.append(name)
        coll = self._collections.get(name)
        if coll is None:
            coll = _FakeCollection()
            self._collections[name] = coll
        return coll

    def get_collection(self, name):
        if self._raise_on_get:
            raise RuntimeError("get_collection failure")
        if name not in self._collections:
            raise RuntimeError("no such collection: " + str(name))
        return self._collections[name]

    def delete_collection(self, name):
        self.deleted.append(name)
        self._collections.pop(name, None)


class _FakeEmbedder:
    def __init__(self, *, per_doc=None, single=None):
        self._per_doc = per_doc
        self._single = single if single is not None else [0.1, 0.2, 0.3]

    def embed(self, documents, show_progress=False):
        if self._per_doc is not None:
            return list(self._per_doc)
        return [[float(len(d)), 0.0, 1.0] for d in documents]

    def embed_single(self, text):
        return list(self._single)


class _FakeChunk:
    def __init__(self, chunk_id, content, metadata):
        self.chunk_id = chunk_id
        self.content = content
        self.metadata = metadata


class _FakeChunkResult:
    def __init__(self, chunks, file_type="text", raw_text_length=0):
        self.chunks = list(chunks)
        self.file_type = file_type
        self.chunk_count = len(self.chunks)
        self.raw_text_length = raw_text_length


class _FakeChunker:
    def __init__(self, result=None):
        self._result = result

    def _mk(self, doc_id):
        if self._result is not None:
            return self._result
        return _FakeChunkResult(
            [_FakeChunk((doc_id or "auto") + "::0", "chunk zero",
                        {"parent_doc_id": doc_id or "auto", "chunk_index": 0,
                         "source_file": "s"})],
            raw_text_length=42,
        )

    def chunk_file(self, filepath, doc_id=None):
        return self._mk(doc_id)

    def chunk_text(self, text, source=None, file_type=None, doc_id=None):
        return self._mk(doc_id)


class _FakeResp:
    def __init__(self, *, content=b"", text="", headers=None, status_ok=True):
        self.content = content
        self.text = text
        self.headers = headers or {}
        self._ok = status_ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("HTTP status error")


class _FakeRequests:
    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc
        self.calls = []

    def get(self, url, timeout=None, headers=None, allow_redirects=None):
        self.calls.append({"url": url, "timeout": timeout,
                           "headers": headers, "allow_redirects": allow_redirects})
        if self._exc:
            raise self._exc
        return self._resp


def _store(rs, tmp_path):
    return rs.RAGVectorStore(data_dir=str(tmp_path))


def _doc(rs, doc_id, collection, **kw):
    base = dict(source_file="f", file_type="text", chunk_count=0,
                raw_text_length=0, ingested_at=1.0)
    base.update(kw)
    return rs.IngestedDocument(doc_id=doc_id, collection_name=collection, **base)


def _cit(rs, cid, collection, parent, **kw):
    base = dict(query="q", chunk_id="ch", source_file="f",
                section=None, score=0.5, timestamp=1.0)
    base.update(kw)
    return rs.CitationRecord(citation_id=cid, collection_name=collection,
                             parent_doc_id=parent, **base)


def _rr(rs, content, score, **kw):
    base = dict(source_file="f", file_type="text", chunk_index=0,
                total_chunks=1, parent_doc_id="p", collection_name="c")
    base.update(kw)
    return rs.RetrievalResult(content=content, score=score, **base)


# =========================================================================
# _RAGDatabase -- SQLite persistence (documents + citations)
# =========================================================================

def test_a1_collection_write_is_parameterised(tmp_path):
    rs, restore = _load()
    try:
        db = rs._RAGDatabase(tmp_path / "rag.db")
        db.create_collection("papers", "research")
        cols = db.list_collections()
        assert [c["name"] for c in cols] == ["papers"]
        assert cols[0]["description"] == "research"
        assert cols[0]["document_count"] == 0 and cols[0]["chunk_count"] == 0

        evil = "papers'); DROP TABLE collections;--"
        db.create_collection(evil)
        names = {c["name"] for c in db.list_collections()}
        assert evil in names and "papers" in names, (
            "a collection name with SQL metacharacters must round-trip as data; "
            "that only holds if every write is parameterised, never f-string SQL"
        )
    finally:
        restore()


def test_a2_document_roundtrips_through_metadata_json(tmp_path):
    rs, restore = _load()
    try:
        db = rs._RAGDatabase(tmp_path / "rag.db")
        db.create_collection("c")
        meta = {"nested": {"k": [1, 2]}, "unicode": "cafe\u0301", "q": "a'b\"c"}
        db.insert_document(_doc(rs, "d", "c", file_type="pdf", chunk_count=3,
                                raw_text_length=10, ingested_at=1.5, metadata=meta))
        got = db.get_document("d")
        assert got is not None and got.metadata == meta, (
            "metadata must round-trip through JSON, quotes and unicode included"
        )
        assert got.chunk_count == 3 and got.file_type == "pdf"

        db.insert_document(_doc(rs, "d", "c", source_file="s2", file_type="txt"))
        assert db.get_document("d").source_file == "s2", (
            "same doc_id must upsert (INSERT OR REPLACE), not duplicate"
        )
    finally:
        restore()


def test_a3_delete_collection_cascades_documents_and_citations(tmp_path):
    rs, restore = _load()
    try:
        db = rs._RAGDatabase(tmp_path / "rag.db")
        db.create_collection("c")
        db.insert_document(_doc(rs, "d", "c"))
        db.insert_citation(_cit(rs, "x", "c", "d"))
        db.delete_collection("c")
        assert db.list_collections() == []
        assert db.get_document("d") is None
        assert db.list_citations(collection_name="c") == []
    finally:
        restore()


def test_a4_delete_document_cascades_citations_and_returns_row(tmp_path):
    rs, restore = _load()
    try:
        db = rs._RAGDatabase(tmp_path / "rag.db")
        db.create_collection("c")
        db.insert_document(_doc(rs, "d1", "c", chunk_count=1))
        db.insert_citation(_cit(rs, "x1", "c", "d1"))
        assert db.list_citations(doc_id="d1"), "precondition: citation present"

        info = db.delete_document("d1")
        assert info is not None and info["doc_id"] == "d1", (
            "delete must return the removed row so the caller can act on it"
        )
        assert db.get_document("d1") is None
        assert db.list_citations(doc_id="d1") == [], (
            "deleting a document must also remove its citations; a citation whose "
            "parent document is gone is a dangling provenance record"
        )
        assert db.delete_document("absent") is None
    finally:
        restore()


def test_a5_list_documents_honours_collection_and_pagination(tmp_path):
    rs, restore = _load()
    try:
        db = rs._RAGDatabase(tmp_path / "rag.db")
        db.create_collection("c1")
        db.create_collection("c2")
        for i in range(3):
            db.insert_document(_doc(rs, f"c1-{i}", "c1", ingested_at=float(i)))
        db.insert_document(_doc(rs, "c2-0", "c2", ingested_at=9.0))

        assert {d.doc_id for d in db.list_documents(collection_name="c1")} == \
            {"c1-0", "c1-1", "c1-2"}
        # ORDER BY ingested_at DESC, so the newest comes first
        assert [d.doc_id for d in db.list_documents(collection_name="c1", limit=1)] \
            == ["c1-2"]
        assert [d.doc_id for d in
                db.list_documents(collection_name="c1", limit=1, offset=1)] == ["c1-1"]
        assert len(db.list_documents()) == 4
    finally:
        restore()


def test_a6_list_citations_filters_by_doc_then_collection(tmp_path):
    rs, restore = _load()
    try:
        db = rs._RAGDatabase(tmp_path / "rag.db")
        db.create_collection("c")
        db.insert_citation(_cit(rs, "x1", "c", "d1", timestamp=1.0))
        db.insert_citation(_cit(rs, "x2", "c", "d2", timestamp=2.0))
        assert [c.citation_id for c in db.list_citations(doc_id="d1")] == ["x1"]
        assert {c.citation_id for c in db.list_citations(collection_name="c")} == \
            {"x1", "x2"}
        assert len(db.list_citations()) == 2
    finally:
        restore()


def test_a7_all_db_access_flows_through_safe_connect_seam(tmp_path):
    calls = []

    def rec(p, **kw):
        calls.append(str(p))
        return sqlite3.connect(str(p), **kw)

    du = types.ModuleType("opti_oignon.db_utils")
    du.safe_connect = rec
    loaded, restore = isolate(targets={_TARGET: source("rag_store.py")},
                              blocked=list(_BLOCKED),
                              seeded={"opti_oignon.db_utils": du})
    try:
        rs = loaded[_TARGET]
        db = rs._RAGDatabase(tmp_path / "rag.db")
        db.create_collection("c")
        assert calls, (
            "every _RAGDatabase connection must be opened through the module's "
            "safe_connect seam, never a bare sqlite3.connect in the body"
        )
    finally:
        restore()


def test_a8_db_utils_unreachable_falls_back_to_sqlite(tmp_path):
    rs, restore = _load(seed_db_utils=False)
    try:
        db = rs._RAGDatabase(tmp_path / "rag.db")
        db.create_collection("c")
        db.insert_document(_doc(rs, "d", "c"))
        assert db.get_document("d") is not None, (
            "with db_utils unreachable the store must still operate on a bare "
            "sqlite connection (documented Daily-mode fail-open), not hard-fail"
        )
    finally:
        restore()


# =========================================================================
# _build_where -- structured metadata filter
# =========================================================================

def test_a9_build_where_none_single_and_conjunction(tmp_path):
    rs, restore = _load()
    try:
        w = rs.RAGVectorStore._build_where
        assert w(None, None, None) is None
        assert w(None, "a.pdf", None) == {"source_file": {"$eq": "a.pdf"}}
        assert w(None, None, "pdf") == {"file_type": {"$eq": "pdf"}}
        assert w(None, "a.pdf", "pdf") == {
            "$and": [
                {"source_file": {"$eq": "a.pdf"}},
                {"file_type": {"$eq": "pdf"}},
            ]
        }, "two filters compose as a conjunction of structured $eq conditions"
    finally:
        restore()


# =========================================================================
# _rerank -- term-overlap heuristic (documented, pinned exactly)
# =========================================================================

def test_a10_rerank_boosts_by_term_overlap_and_caps_and_sorts(tmp_path):
    rs, restore = _load()
    try:
        rerank = rs.RAGVectorStore._rerank
        # no query terms -> untouched
        assert rerank("", [_rr(rs, "x", 0.5)])[0].score == 0.5

        # the term-matching hit is boosted enough to overtake a higher base score
        a = _rr(rs, "alpha beta here", 0.58)   # 2/2 terms -> +0.05 -> 0.63
        b = _rr(rs, "gamma delta", 0.60)       # 0 terms -> +0.00 -> 0.60
        out = rerank("alpha beta", [b, a])
        assert out[0].content == "alpha beta here" and round(out[0].score, 4) == 0.63
        assert out[1].content == "gamma delta"

        # boost is capped so the score never exceeds 1.0
        assert rerank("alpha", [_rr(rs, "alpha", 0.99)])[0].score == 1.0
    finally:
        restore()


# =========================================================================
# _extract_html_text -- boilerplate removal + entity decode
# =========================================================================

def test_a11_extract_html_strips_boilerplate_and_decodes_entities(tmp_path):
    rs, restore = _load()
    try:
        html = (
            "<html><head><style>.x{color:red}</style></head>"
            "<body><script>alert('boom')</script>"
            "<nav>menu links here</nav>"
            "<p>Hello &amp; welcome &lt;friend&gt;</p></body></html>"
        )
        out = rs.RAGVectorStore._extract_html_text(html, {})
        assert "alert(" not in out and "color:red" not in out, (
            "script and style contents must be removed, not just their tags"
        )
        assert "menu links here" not in out, "nav boilerplate must be stripped"
        assert "Hello & welcome <friend>" in out, "HTML entities must be decoded"
        assert "<p>" not in out and "</p>" not in out, "residual tags must be gone"
    finally:
        restore()


# =========================================================================
# load_rag_config -- schema completeness + extra-key filtering
# =========================================================================

def test_a12_load_rag_config_is_complete_and_filters_extras(tmp_path):
    rs, restore = _load()
    try:
        cfg = rs.load_rag_config()
        assert set(cfg.keys()) == {
            "chunking", "embedding", "retrieval",
            "web_ingestion", "collections", "storage",
        }, (
            "load_rag_config exposes exactly its own schema; unrelated top-level "
            "keys present in rag.yaml must not leak through"
        )
        assert isinstance(cfg["retrieval"]["rerank"], bool)
        assert isinstance(cfg["retrieval"]["n_results"], int)
        assert isinstance(cfg["embedding"]["model"], str)
    finally:
        restore()


# =========================================================================
# query -- fail-secure ladder + happy path + score floor
# =========================================================================

def test_a13_query_is_fail_secure_and_never_raises(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)

        store._chroma = None
        r = store.query("q", collection="default")
        assert r.results == [] and r.total_results == 0, "no backend -> empty"

        store._chroma = _FakeChroma({}, raise_on_get=True)
        assert store.query("q", collection="default").results == [], \
            "get_collection failure -> empty"

        store._chroma = _FakeChroma({"default": _FakeCollection(count_val=0)})
        assert store.query("q", collection="default").results == [], \
            "empty collection -> empty"

        store._chroma = _FakeChroma(
            {"default": _FakeCollection(count_val=3, raise_on_query=True)})
        store._get_embedder = lambda: None
        assert store.query("q", collection="default").results == [], (
            "a backend failure during query must degrade to an empty response, "
            "never propagate an exception to the caller"
        )
    finally:
        restore()


def test_a14_query_scores_filters_sorts_reranks_and_cites(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        docs = ["alpha match", "weak", "strong hit alpha"]
        metas = [
            {"source_file": "a.txt", "file_type": "text", "chunk_index": 0,
             "total_chunks": 3, "parent_doc_id": "d1", "section": "S", "page": 2},
            {"source_file": "b.txt", "file_type": "text", "chunk_index": 1,
             "total_chunks": 3, "parent_doc_id": "d2"},
            {"source_file": "c.txt", "file_type": "text", "chunk_index": 2,
             "total_chunks": 3, "parent_doc_id": "d3"},
        ]
        dists = [0.20, 0.90, 0.10]  # similarities 0.80, 0.10, 0.90
        coll = _FakeCollection(count_val=3, query_payload={
            "documents": [docs], "metadatas": [metas], "distances": [dists]})
        store._chroma = _FakeChroma({"default": coll})
        store._embedder = _FakeEmbedder()

        resp = store.query("alpha", collection="default", n_results=5,
                           min_score=0.3, track_citations=True)

        contents = [r.content for r in resp.results]
        assert "weak" not in contents, "similarity 0.10 is below the 0.3 floor"
        assert resp.results[0].content == "strong hit alpha", (
            "highest-similarity hit ranks first"
        )
        top = [r for r in resp.results if r.parent_doc_id == "d1"][0]
        assert top.page == 2 and top.section == "S", "page/section carried through"

        assert len(resp.citations) == len(resp.results) == resp.total_results
        want = hashlib.sha256(b"d1::0").hexdigest()[:16]
        assert any(c.chunk_id == want for c in resp.citations), (
            "chunk id must be sha256(parent::index)[:16] so a chunk is stable"
        )
        assert store.db.list_citations(collection_name="default"), "citations persisted"
    finally:
        restore()


def test_a15_query_score_floor_is_never_negative(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        coll = _FakeCollection(count_val=1, query_payload={
            "documents": [["zzz"]],
            "metadatas": [[{"parent_doc_id": "d", "chunk_index": 0}]],
            "distances": [[1.4]],  # raw similarity 1 - 1.4 = -0.4
        })
        store._chroma = _FakeChroma({"default": coll})
        store._get_embedder = lambda: None  # query_texts path, no embedding

        resp = store.query("qqq", collection="default", n_results=5,
                           min_score=-1.0, rerank=False, track_citations=False)
        assert resp.results, "min_score=-1 keeps the hit so the floor is observable"
        assert resp.results[0].score == 0.0, (
            "cosine distance can exceed 1; similarity must be floored at 0, "
            "never returned as a negative score"
        )
    finally:
        restore()


# =========================================================================
# ingestion -- ingest_text / _store_chunks / delete_document
# =========================================================================

def test_a16_ingest_text_stores_chunks_and_records_document(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        coll = _FakeCollection()
        store._chroma = _FakeChroma({"default": coll})
        store._embedder = _FakeEmbedder()
        store._chunker = _FakeChunker(result=_FakeChunkResult([
            _FakeChunk("k0", "chunk zero", {"parent_doc_id": "d", "chunk_index": 0}),
            _FakeChunk("k1", "chunk one", {"parent_doc_id": "d", "chunk_index": 1}),
        ], raw_text_length=99))

        doc = store.ingest_text("some text", collection="default", doc_id="d")
        assert doc.chunk_count == 2 and doc.raw_text_length == 99
        assert store.db.get_document("d") is not None
        assert coll.upserts, "chunks must be upserted into the vector store"
        assert any(c["name"] == "default" for c in store.db.list_collections()), (
            "the target collection row is created on ingest"
        )
    finally:
        restore()


def test_a17_ingest_empty_text_still_records_zero_chunk_document(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        store._chroma = _FakeChroma()
        store._embedder = _FakeEmbedder()
        store._chunker = _FakeChunker(result=_FakeChunkResult([], raw_text_length=0))

        doc = store.ingest_text("", collection="default", doc_id="empty")
        assert doc.chunk_count == 0
        assert store.db.get_document("empty") is not None, (
            "a document that produced no chunks must still be recorded, so it is "
            "visible and deletable rather than silently dropped"
        )
    finally:
        restore()


def test_a18_store_chunks_drops_failed_embeddings(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        coll = _FakeCollection()
        store._chroma = _FakeChroma({"default": coll})
        store._embedder = _FakeEmbedder(per_doc=[[1.0], None, [3.0]])
        chunks = [_FakeChunk("i0", "c0", {}), _FakeChunk("i1", "c1", {}),
                  _FakeChunk("i2", "c2", {})]

        store._store_chunks(chunks, "default")
        up = coll.upserts[-1]
        assert up["ids"] == ["i0", "i2"] and up["embeddings"] == [[1.0], [3.0]], (
            "a chunk whose embedding came back None must be dropped from the "
            "upsert, not stored against a null vector"
        )
    finally:
        restore()


def test_a19_store_chunks_all_failed_embeddings_stores_without_vectors(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        coll = _FakeCollection()
        store._chroma = _FakeChroma({"default": coll})
        store._embedder = _FakeEmbedder(per_doc=[None, None])
        chunks = [_FakeChunk("i0", "c0", {}), _FakeChunk("i1", "c1", {})]

        store._store_chunks(chunks, "default")
        up = coll.upserts[-1]
        assert up["ids"] == ["i0", "i1"] and up["embeddings"] is None, (
            "when every embedding fails, the chunks are still stored (ChromaDB "
            "default embedding), never lost"
        )
    finally:
        restore()


def test_a20_store_chunks_without_embedder_uses_default_embedding(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        coll = _FakeCollection()
        store._chroma = _FakeChroma({"default": coll})
        store._get_embedder = lambda: None
        chunks = [_FakeChunk("i0", "c0", {})]

        store._store_chunks(chunks, "default")
        assert coll.upserts[-1]["embeddings"] is None
    finally:
        restore()


def test_a21_store_delete_document_removes_chunks_and_returns_bool(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        store.db.create_collection("c")
        store.db.insert_document(_doc(rs, "d", "c", chunk_count=2))
        coll = _FakeCollection(get_ids=["d::0", "d::1"])
        store._chroma = _FakeChroma({"c": coll})

        assert store.delete_document("d") is True
        assert coll.deleted == [["d::0", "d::1"]], (
            "the document's chunks must be removed from the vector store too"
        )
        assert store.delete_document("missing") is False
    finally:
        restore()


# =========================================================================
# ingest_url -- network-facing guards (pinned as-is)
# =========================================================================

def test_a22_ingest_url_rejects_non_http_schemes(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        store._chroma = _FakeChroma()
        store._load_web_config = lambda: {"max_page_size": 10_000,
                                          "min_text_length": 100}
        rs.REQUESTS_AVAILABLE = True
        # a valid short response, so the ONLY thing that can raise is the scheme
        # gate -- if the gate widened, these would fall through and return a doc
        rs._requests_lib = _FakeRequests(resp=_FakeResp(
            content=b"short", text="short", headers={"content-type": "text/plain"}))

        for bad in ("file:///etc/passwd", "ftp://host/x", "gopher://host/x"):
            with pytest.raises(ValueError):
                store.ingest_url(bad)
    finally:
        restore()


def test_a23_ingest_url_requires_requests(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        store._chroma = _FakeChroma()
        rs.REQUESTS_AVAILABLE = False
        with pytest.raises(RuntimeError):
            store.ingest_url("http://example.com")
    finally:
        restore()


def test_a24_ingest_url_rejects_oversize_page(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        store._chroma = _FakeChroma()
        store._load_web_config = lambda: {"max_page_size": 1000,
                                          "min_text_length": 100}
        rs.REQUESTS_AVAILABLE = True
        rs._requests_lib = _FakeRequests(resp=_FakeResp(
            content=b"x" * 2000, text="x" * 10,
            headers={"content-type": "text/html"}))
        with pytest.raises(ValueError):
            store.ingest_url("http://example.com")
    finally:
        restore()


def test_a25_ingest_url_short_text_records_zero_chunk_doc(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        store._chroma = _FakeChroma()
        store._load_web_config = lambda: {"max_page_size": 10_000,
                                          "min_text_length": 100}
        rs.REQUESTS_AVAILABLE = True
        rs._requests_lib = _FakeRequests(resp=_FakeResp(
            content=b"hi", text="hi", headers={"content-type": "text/plain"}))

        doc = store.ingest_url("http://example.com/page", collection="web")
        assert doc.chunk_count == 0 and doc.file_type == "html"
        assert doc.metadata.get("url") == "http://example.com/page"
        assert doc.metadata.get("domain") == "example.com"
        assert store.db.get_document(doc.doc_id) is not None
    finally:
        restore()


def test_a26_ingest_url_happy_path_chunks_and_tags_domain(tmp_path):
    rs, restore = _load()
    try:
        store = _store(rs, tmp_path)
        coll = _FakeCollection()
        store._chroma = _FakeChroma({"web": coll})
        store._embedder = _FakeEmbedder()
        store._chunker = _FakeChunker()
        store._load_web_config = lambda: {"max_page_size": 100_000,
                                          "min_text_length": 5}
        rs.REQUESTS_AVAILABLE = True
        page = "<html><body><p>" + "content here " * 20 + "</p></body></html>"
        rs._requests_lib = _FakeRequests(resp=_FakeResp(
            content=page.encode(), text=page,
            headers={"content-type": "text/html"}))

        doc = store.ingest_url("https://site.example/article", collection="web")
        assert doc.chunk_count >= 1
        assert doc.metadata.get("domain") == "site.example"
        assert doc.file_type == "text", "delegates to ingest_text with file_type=text"
    finally:
        restore()


# =========================================================================
# get_rag_store -- process-wide singleton
# =========================================================================

def test_a27_get_rag_store_is_a_singleton(tmp_path):
    rs, restore = _load()
    try:
        s1 = rs.get_rag_store(data_dir=str(tmp_path / "a"))
        s2 = rs.get_rag_store(data_dir=str(tmp_path / "b"))
        assert s1 is s2, "get_rag_store returns the process-wide singleton"
        assert s1.data_dir == (tmp_path / "a"), (
            "the second call's data_dir is ignored once the instance exists"
        )
    finally:
        restore()
